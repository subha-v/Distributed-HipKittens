"""exp_24 Instrument A: the external ladder, five arms, one pool, two protocols.

Generalizes exp_10's `mp_vs_rank1.py` from two arms to five and from one protocol
to two. Everything that made that instrument trustworthy is kept: the same 8
processes, the same inputs, one fresh pool per shape, correctness of every gated
arm before a single sample is taken, and per-rank fd-level stderr (a GPU memory
fault aborts the process from the runtime, so anything buffered in Python's
`sys.stderr` object is lost).

What is new:

  * five arms instead of two -- `ours`, `ours_null` (the SAME file under a second
    module name, so the ladder's noise floor is measured in the run that produces
    the ratios), `reference`, `rank1`, and `harness_floor`;
  * `harness_floor` returns a preallocated output and nothing else, so the graded
    protocol's own constant is MEASURED here rather than quoted from memory;
  * two protocols in the same pool -- the evaluator's graded per-call region
    verbatim, and a pipelined burst -- never blended;
  * complete rotation: arm order is cyclically rotated so every arm is first
    exactly once, and protocol order flips per rep;
  * duration-based warmup, because idle sclk on this node is ~125 MHz against
    ~1900 pinned and a fixed-iteration warmup produced a 60% wrong number once.

ONE SHAPE PER INVOCATION, with a fresh pool. rank-1 caches its compiled kernel in
module globals keyed by NOTHING, so the cache is only valid for the shape it was
built for; the evaluator survives this by destroying the process group between
shapes.

Usage: python3 ladder_mp.py <shape_index> <iters> <reps> <port> [burst] [pipe_iters]
Env:   LAD_OUT (output path stem), LAD_ARMS (csv subset), LAD_WARM_MS,
       VS_FORCE_BIAS (should be 1; see design.md 2.7)
"""

import importlib.util
import json
import os
import statistics
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
ON = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight"
HARNESS = f"{ON}/harness"
ARM_REF = f"{ON}/compbench/reference"
ARM_R1 = f"{ON}/compbench/rank1"

SCORED = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]

# (key, module name, source file). `ours` and `ours_null` are deliberately the
# same path under two names: that is what makes the null arm a null.
ARM_SPECS = [
    ("ours", "ours_submission", f"{HARNESS}/submission.py"),
    ("ours_null", "ours_null_submission", f"{HARNESS}/submission.py"),
    ("reference", "reference_submission", f"{ARM_REF}/submission.py"),
    ("rank1", "rank1_submission", f"{ARM_R1}/submission.py"),
    ("harness_floor", None, None),
]

# Arms whose correctness is gated. `harness_floor` is deliberately wrong.
UNGATED = {"harness_floor"}
# Arms excluded from the ratio tables.
NON_RATIO = {"ours_null", "harness_floor"}


def load_module(name, path):
    """Both real submissions are called `submission`; load under distinct names."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def generate_input(rank, world_size, m, n, k, has_bias, seed):
    """reference.generate_input verbatim, plus the VS_FORCE_BIAS reproduction.

    VS_FORCE_BIAS=1 reproduces what the official evaluator ACTUALLY does: its
    cases parser does `int(val)` and keeps the raw string on ValueError, so
    `has_bias: False` becomes the string "False", which is truthy. See
    design.md 2.7 -- without it rank-1 raises on its second call.
    """
    if os.environ.get("VS_FORCE_BIAS") == "1":
        has_bias = True
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)
    local_k = k // world_size
    x = (torch.rand((m, local_k), dtype=torch.bfloat16, device=device,
                    generator=gen) * 2 - 1) * 0.01
    w = (torch.rand((n, local_k), dtype=torch.bfloat16, device=device,
                    generator=gen) * 2 - 1) * 0.01
    bias = None
    if has_bias:
        gen.manual_seed(seed)
        bias = (torch.rand((n,), dtype=torch.bfloat16, device=device,
                           generator=gen) * 2 - 1) * 0.01
    return (x, w, bias)


def clear_l2_cache():
    """Verbatim from the evaluator's utils.py:169."""
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
    dummy.fill_(42)
    del dummy


def _clone_data(data):
    return tuple(t.clone() if torch.is_tensor(t) else t for t in data)


def make_floor_kernel(m, n, rank):
    """The harness-constant arm: allocate the output once, then return it.

    Under the graded protocol its time IS the protocol's constant --
    clear_l2 -> sync -> barrier -> clone -> sync -> barrier -- so the netting in
    plan.md 7 rests on a measurement from this run instead of a remembered 92 us.
    """
    out = torch.empty((m // WORLD, n), dtype=torch.bfloat16,
                      device=f"cuda:{rank}")

    def kernel(_data):
        return out

    return kernel


def warm(kernel, data, warm_ms, min_calls, rank, block=5):
    """Duration-based warmup, with the call count agreed across ranks.

    Duration-based because idle sclk on this node is ~125 MHz against ~1900
    pinned and a fixed-iteration warmup produced a 60% wrong number once.

    But a per-rank deadline gives per-rank call COUNTS, and two of these arms are
    collectives -- `reference` calls reduce_scatter_tensor and rank-1 calls its
    own dist_barrier -- so unequal counts deadlock. Rank 0 therefore owns the
    stop decision and broadcasts it, and every rank runs the same whole number of
    fixed-size blocks. Duration-based sizing, identical collective counts.
    """
    stop = torch.zeros(1, dtype=torch.int32, device=f"cuda:{rank}")
    calls = 0
    out = None
    deadline = None
    while True:
        for _ in range(block):
            out = kernel(_clone_data(data))
            calls += 1
        torch.cuda.synchronize()
        # The clock starts AFTER the first block, so one-time setup cannot eat the
        # warm window. In the LAD_QUICK validation `reference` and `rank1` spent
        # the whole 400 ms on RCCL channel setup and Triton JIT respectively and
        # landed at the 20-call minimum, while `ours` got 965 calls -- the window
        # is meant to buy steady-state work at ramped clocks, not to time setup.
        if deadline is None:
            deadline = time.perf_counter() + warm_ms / 1000.0
        if rank == 0:
            done = calls >= min_calls and time.perf_counter() >= deadline
            stop.fill_(1 if done else 0)
        dist.broadcast(stop, 0)
        if int(stop.item()):
            break
    torch.cuda.synchronize()
    return out, calls


def time_graded(kernel, data, iters):
    """eval.py:_run_distributed_benchmark lines 349-366, verbatim.

    Deviation, documented: eval.py times on rank 0 only. Every rank is timing the
    same per-call wall time here, so all eight are recorded and pooled; the
    rank-0 subset stays separable in the JSON.
    """
    samples = []
    out = None
    for _ in range(iters):
        clear_l2_cache()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter_ns()
        out = kernel(_clone_data(data))
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1000.0)
    return samples, out


def time_pipelined(kernel, data, iters, burst):
    """Back-to-back calls, one sync/barrier pair per burst: the throughput number.

    The clone is hoisted out of the burst deliberately -- the clone and the
    trailing barrier ARE the per-call harness tax the graded/pipelined difference
    is there to expose. All three submissions treat the inputs as read-only, and
    the caller re-checks the final output against the oracle afterwards.
    """
    clone = _clone_data(data)
    samples = []
    out = None
    for _ in range(iters):
        clear_l2_cache()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter_ns()
        for _ in range(burst):
            out = kernel(clone)
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1000.0 / burst)
    return samples, out


def oracle(data, rank, m):
    x, w, bias = data
    partial = torch.matmul(x, w.T)
    if bias is not None:
        partial = partial + bias
    dist.all_reduce(partial)
    rows = m // WORLD
    return partial[rank * rows:(rank + 1) * rows, :]


def check(got, want):
    diff = float((got.float() - want.float()).abs().max().item())
    return {
        "allclose_1e-2": bool(torch.allclose(want, got, rtol=1e-2, atol=1e-2)),
        "allclose_2e-3": bool(torch.allclose(want, got, rtol=2e-3, atol=2e-3)),
        "max_abs_diff": diff,
    }


def _worker(rank, shape_index, iters, reps, port, burst, pipe_iters, arm_keys,
            out_path):
    err = open(f"{out_path}.rank{rank}.stderr", "wb", 0)
    os.dup2(err.fileno(), 2)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("HK_DEBUG", "0")
    warm_ms = float(os.environ.get("LAD_WARM_MS", "400"))

    def say(message):
        if rank == 0:
            print(f"[rank0] {message}", flush=True)

    # rank-1's C++ writes ipc_handles_rank*.bin into cwd and does
    # `from task import ...`; our submission needs HARNESS on the path.
    os.chdir(ARM_R1)
    for path in (ARM_R1, ARM_REF, HARNESS):
        sys.path.insert(0, path)

    torch.cuda.set_device(rank)

    specs = [s for s in ARM_SPECS if s[0] in arm_keys]
    modules = {}
    for key, mod_name, src in specs:
        if mod_name is not None:
            modules[key] = load_module(mod_name, src)

    dist.init_process_group("nccl", init_method="env://", rank=rank,
                            world_size=WORLD,
                            device_id=torch.device(f"cuda:{rank}"))

    result = {}
    try:
        m, n, k, has_bias, seed = SCORED[shape_index]
        arms = [s[0] for s in specs]
        say(f"--- shape {shape_index + 1}: {m}x{n}x{k} bias={int(has_bias)} "
            f"arms={arms} iters={iters} reps={reps} "
            f"pipe={pipe_iters}x{burst}")
        data = generate_input(rank, WORLD, m, n, k, has_bias, seed)

        impls = {}
        for key in arms:
            if key == "harness_floor":
                impls[key] = make_floor_kernel(m, n, rank)
            else:
                impls[key] = modules[key].custom_kernel

        # Warm EVERY arm before ANY arm is timed, so no arm is measured while
        # another is still cold. First call is where each does its IPC exchange
        # or JIT and where RCCL builds channels; none of that may land in a
        # timed run.
        outs, warm_calls = {}, {}
        for key in arms:
            outs[key], warm_calls[key] = warm(impls[key], data, warm_ms, 20, rank)
        torch.cuda.synchronize()
        dist.barrier()
        say(f"    warm calls: {warm_calls}")

        # Correctness of every gated arm BEFORE any timing.
        want = oracle(data, rank, m)
        correctness = {}
        for key in arms:
            if key in UNGATED:
                continue
            correctness[key] = check(outs[key], want)
            say(f"    {key:>14} {correctness[key]}")
        bad = [key for key, c in correctness.items() if not c["allclose_1e-2"]]
        if bad:
            raise RuntimeError(f"refusing to time: arms wrong at 1e-2: {bad}")

        samples = {key: {"graded": [], "pipelined": []} for key in arms}
        rank0_flag = (rank == 0)
        for rep in range(reps):
            shift = rep % len(arms)
            order = arms[shift:] + arms[:shift]
            protos = (("graded", "pipelined") if rep % 2 == 0
                      else ("pipelined", "graded"))
            say(f"  rep {rep}: order={order} protos={protos}")
            for proto in protos:
                for key in order:
                    if proto == "graded":
                        got, out = time_graded(impls[key], data, iters)
                    else:
                        got, out = time_pipelined(impls[key], data, pipe_iters,
                                                  burst)
                    samples[key][proto].extend(got)
                    # A mutating arm would make the reused pipelined clone
                    # produce a fast wrong answer; catch it here, not in a plot.
                    if key not in UNGATED:
                        post = check(out, want)
                        if not post["allclose_1e-2"]:
                            raise RuntimeError(
                                f"arm {key} wrong AFTER {proto} timing: {post}")

        result = {
            "shape_index": shape_index,
            "shape": [m, n, k, int(has_bias)],
            "shape_label": f"{m}x{n}x{k}",
            "bias_forced": os.environ.get("VS_FORCE_BIAS") == "1",
            "iters": iters, "reps": reps,
            "pipe_iters": pipe_iters, "burst": burst,
            "warm_ms": warm_ms, "warm_calls": warm_calls,
            "is_rank0": rank0_flag,
            "correctness": correctness,
            "arms": {key: samples[key] for key in arms},
        }
        for key in arms:
            for proto in ("graded", "pipelined"):
                s = samples[key][proto]
                if not s:
                    continue
                say(f"    {key:>14} {proto:>9}  best={min(s):9.2f} "
                    f"med={statistics.median(s):9.2f} "
                    f"mean={statistics.mean(s):9.2f} "
                    f"sd={statistics.stdev(s) if len(s) > 1 else 0.0:8.2f} "
                    f"max={max(s):9.2f} us  n={len(s)}")
    except Exception:
        print(f"[rank {rank}] EXCEPTION\n{traceback.format_exc()}", flush=True)
        raise
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    with open(f"{out_path}.rank{rank}.json", "w") as handle:
        json.dump(result, handle, indent=2)


def main():
    shape_index = int(sys.argv[1])
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 13100
    burst = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    pipe_iters = int(sys.argv[6]) if len(sys.argv) > 6 else 20

    arm_keys = os.environ.get("LAD_ARMS")
    arm_keys = ([k.strip() for k in arm_keys.split(",") if k.strip()]
                if arm_keys else [s[0] for s in ARM_SPECS])
    unknown = set(arm_keys) - {s[0] for s in ARM_SPECS}
    if unknown:
        print(f"unknown arms: {sorted(unknown)}", flush=True)
        return 2

    out_path = os.environ.get("LAD_OUT", f"lad_shape{shape_index}")
    print(f"=== exp_24 ladder, shape[{shape_index}]={SCORED[shape_index]} "
          f"arms={arm_keys} iters={iters} reps={reps} "
          f"pipelined={pipe_iters}x{burst} port={port} ===", flush=True)

    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(WORLD):
        p = ctx.Process(target=_worker,
                        args=(rank, shape_index, iters, reps, port, burst,
                              pipe_iters, arm_keys, out_path))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    codes = [p.exitcode for p in procs]
    print(f"exit codes: {codes}", flush=True)
    return 0 if all(c == 0 for c in codes) else 2


if __name__ == "__main__":
    sys.exit(main())

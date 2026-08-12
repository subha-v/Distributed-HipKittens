"""Same-run, interleaved, paired comparison: OUR kernel vs the frozen rank-1.

Why interleaved rather than two evaluator runs: two separately-staged evaluator
runs already gave this project a misleading answer once (1.37x when the
interleaved harness said 1.06x). Both arms here run in the SAME process pool,
on the SAME inputs, alternating, with the arm order reversed every rep so
neither is systematically first.

Protocol per timed iteration is the evaluator's `full` protocol verbatim: clone
the input, flush L2, synchronize + barrier, time, synchronize + barrier.

ONE SHAPE PER INVOCATION, with a fresh pool. That is deliberate: rank-1 caches
its compiled kernel in module globals (`pre_compile_cache`,
`pre_compile_cache2`) that are keyed by NOTHING, so they are only valid for the
shape they were built for -- the evaluator gets away with it because it destroys
the process group between shapes, which runs rank-1's own cache-clearing patch.
Running one shape per process avoids reaching into rank-1's state at all.

Usage: python3 mp_vs_rank1.py <shape_index> <iters> <reps> <port>
"""

import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
HARNESS = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/harness"
ARM = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/compbench/rank1"

SCORED = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]

ARMS = ["ours", "rank1"]


def load_module(name, path):
    """Both submissions are called `submission`; load them under distinct names."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def generate_input(rank, world_size, m, n, k, has_bias, seed):
    # VS_FORCE_BIAS reproduces what the official evaluator ACTUALLY does. Its
    # cases parser (eval.py get_test_cases) does `int(val)` and on ValueError
    # keeps the raw string, so `has_bias: False` becomes the string "False",
    # which is truthy -- reference.generate_input therefore builds a bias for
    # all six graded shapes. rank-1's author noticed ("bench all have bias?")
    # and its cached launch path dereferences bias unconditionally, so a
    # genuinely absent bias makes rank-1 raise on its second call.
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


def _worker(rank, shape_index, iters, reps, port, out_path):
    err = open(f"{out_path}.rank{rank}.stderr", "wb", 0)
    os.dup2(err.fileno(), 2)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["HK_DEBUG"] = "0"

    def say(message):
        if rank == 0:
            print(f"[rank0] {message}", flush=True)

    os.chdir(ARM)          # rank-1's C++ writes ipc_handles_rank*.bin into cwd
    sys.path.insert(0, ARM)      # rank-1's `from task import ...`
    sys.path.insert(0, HARNESS)  # our submission

    torch.cuda.set_device(rank)
    ours = load_module("ours_submission", os.path.join(HARNESS, "submission.py"))
    r1 = load_module("rank1_submission", os.path.join(ARM, "submission.py"))

    dist.init_process_group("nccl", init_method="env://", rank=rank,
                            world_size=WORLD,
                            device_id=torch.device(f"cuda:{rank}"))

    impls = {"ours": ours.custom_kernel, "rank1": r1.custom_kernel}
    result = {}
    try:
        m, n, k, has_bias, seed = SCORED[shape_index]
        say(f"--- shape {shape_index+1}: {m}x{n}x{k} bias={int(has_bias)}")
        data = generate_input(rank, WORLD, m, n, k, has_bias, seed)

        # Warm both arms: first call is where each does its IPC exchange / JIT,
        # and where RCCL builds channels. None of that may land in a timed run.
        outs = {}
        for arm in ARMS:
            for _ in range(20):
                outs[arm] = impls[arm](_clone_data(data))
        torch.cuda.synchronize()
        dist.barrier()

        # Correctness of BOTH arms before any timing. A number for a wrong
        # kernel is worthless, and rank-1 silently falls back to torch for
        # shapes it has no config for -- checked separately in p11.
        x, w, bias = data
        partial = torch.matmul(x, w.T)
        if bias is not None:
            partial = partial + bias
        dist.all_reduce(partial)
        rows = m // WORLD
        want = partial[rank * rows:(rank + 1) * rows, :]
        correctness = {}
        for arm in ARMS:
            got = outs[arm]
            diff = float((got.float() - want.float()).abs().max().item())
            correctness[arm] = {
                "allclose_1e-2": bool(torch.allclose(want, got, rtol=1e-2,
                                                     atol=1e-2)),
                "allclose_2e-3": bool(torch.allclose(want, got, rtol=2e-3,
                                                     atol=2e-3)),
                "max_abs_diff": diff,
            }
            say(f"    {arm:>6} correctness {correctness[arm]}")

        samples = {arm: [] for arm in ARMS}
        for rep in range(reps):
            order = ARMS if rep % 2 == 0 else list(reversed(ARMS))
            for arm in order:
                kernel = impls[arm]
                for _ in range(iters):
                    clear_l2_cache()
                    torch.cuda.synchronize()
                    dist.barrier()
                    t0 = time.perf_counter_ns()
                    kernel(_clone_data(data))
                    torch.cuda.synchronize()
                    dist.barrier()
                    t1 = time.perf_counter_ns()
                    samples[arm].append((t1 - t0) / 1000.0)

        result = {
            "shape": [m, n, k, int(has_bias)],
            "bias_forced": os.environ.get("VS_FORCE_BIAS") == "1",
            "bias_present": bias is not None,
            "correctness": correctness,
            "arms": {arm: samples[arm] for arm in ARMS},
        }
        for arm in ARMS:
            s = samples[arm]
            say(f"    {arm:>6}  best={min(s):9.2f} med={statistics.median(s):9.2f} "
                f"mean={statistics.mean(s):9.2f} sd={statistics.stdev(s):8.2f} "
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
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 12500
    out_path = os.environ.get("VS_OUT", f"vs_shape{shape_index}")

    print(f"=== interleaved ours-vs-rank1, shape[{shape_index}]="
          f"{SCORED[shape_index]} iters={iters} reps={reps} ===", flush=True)
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(WORLD):
        p = ctx.Process(target=_worker, args=(rank, shape_index, iters, reps,
                                              port, out_path))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    codes = [p.exitcode for p in procs]
    print(f"exit codes: {codes}", flush=True)
    return 0 if all(c == 0 for c in codes) else 2


if __name__ == "__main__":
    sys.exit(main())

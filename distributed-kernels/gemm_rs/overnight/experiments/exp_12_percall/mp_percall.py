"""exp_12 step 1: split the graded protocol's per-call fixed cost four ways.

The graded protocol (eval.py:349-360) is

    clear_l2_cache(); torch.cuda.synchronize(); dist.barrier()
    t0 = perf_counter_ns()
    output = custom_kernel(_clone_data(data, rank))
    torch.cuda.synchronize(); dist.barrier()
    t1 = perf_counter_ns()

and t1 - t0 is ~100 us larger than our own pipelined device time on every
shape. This file runs that structure verbatim in eight processes and inserts
three more timestamps inside the region, which costs a few hundred ns and
splits it:

    t0    -> t_pay   the clone
    t_pay -> t_iss   the host path (python + pybind + hipLaunchKernel)
    t_iss -> t_syn   the device (execution + drain)
    t_syn -> t1      the trailing barrier

perf_counter_ns() is CLOCK_MONOTONIC, so the eight ranks' stamps are
comparable on one host. That yields barrier-exit skew for free -- which matters
because no rank's grid can finish before every other rank's grid is up.

Arms run interleaved in one pool with the order flipped every rep. Null arms
launch a probe kernel from `nullk.so` that matches our launch geometry (512
threads, launch_bounds(512,1), optional 64 KB dynamic LDS) but does nothing.

Usage: python3 mp_percall.py <shapes_csv> <arms_csv> <iters> <reps> <tag> <port>
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

ON = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight"
HARNESS = f"{ON}/harness"
BUILD = f"{HARNESS}/build"
sys.path.insert(0, HARNESS)

WORLD = 8
CU_COUNT = 304
CTA_THREADS = 512
LDS_FULL = 65536

SCORED = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]

# kind: "none" (nothing in the timed region), "null"/"epoch" (probe kernel),
# "ours" (submission.custom_kernel). clone/flush reproduce the evaluator's two
# non-kernel components.
ARM_SPECS = {
    "floor":     dict(kind="none"),
    "null1":     dict(kind="null", grid=1, lds=0),
    "null8":     dict(kind="null", grid=8, lds=LDS_FULL),
    "null76":    dict(kind="null", grid=76, lds=LDS_FULL),
    "null152":   dict(kind="null", grid=152, lds=LDS_FULL),
    "null304":   dict(kind="null", grid=CU_COUNT, lds=LDS_FULL),
    "null304l0": dict(kind="null", grid=CU_COUNT, lds=0),
    "null608":   dict(kind="null", grid=2 * CU_COUNT, lds=LDS_FULL),
    "epoch304":  dict(kind="epoch", grid=CU_COUNT, lds=LDS_FULL),
    "ours":      dict(kind="ours"),
    "oursclone": dict(kind="ours", clone=True),
    "oursfull":  dict(kind="ours", clone=True, flush=True),
    # The pre-exp_12 host path, reached by turning the prebound fast path off
    # in the submission module. Same process, same pool, same inputs,
    # interleaved with `ours*` -- the only denominator that is same-run.
    "oldbare":   dict(kind="ours", fast=False),
    "oldfull":   dict(kind="ours", clone=True, flush=True, fast=False),
}

STAMPS = ["t0", "t_pay", "t_iss", "t_syn", "t1"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def generate_input(rank, world_size, m, n, k, has_bias, seed):
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


def _worker(rank, indices, arms, iters, reps, port, out_path):
    err_file = open(f"{out_path}.rank{rank}.stderr", "wb", 0)
    os.dup2(err_file.fileno(), 2)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("HK_DEBUG", "0")

    def say(message):
        if rank == 0:
            print(f"[rank0] {message}", flush=True)

    torch.cuda.set_device(rank)
    import submission as S
    from submission import custom_kernel
    fast_available = getattr(S, "_FAST", False)

    nullk = load_module("nullk", f"{BUILD}/nullk.so")
    # Probe scratch: the epoch cells use the same 2*304 u32 layout the real
    # kernel's ep_cell does, so the atomics hit the same number of cache lines.
    cells = torch.zeros(2 * CU_COUNT, dtype=torch.int32, device=f"cuda:{rank}")
    errbuf = torch.zeros(1, dtype=torch.int32, device=f"cuda:{rank}")
    sink = torch.zeros(4, dtype=torch.int32, device=f"cuda:{rank}")
    cells_p, err_p, sink_p = (int(cells.data_ptr()), int(errbuf.data_ptr()),
                              int(sink.data_ptr()))

    dist.init_process_group("nccl", init_method="env://", rank=rank,
                            world_size=WORLD,
                            device_id=torch.device(f"cuda:{rank}"))
    results = {}
    try:
        for index in indices:
            m, n, k, has_bias, seed = SCORED[index]
            say(f"--- shape {index+1}: {m}x{n}x{k} bias={int(has_bias)}")
            data = generate_input(rank, WORLD, m, n, k, has_bias, seed)

            # Warm every arm: the first custom_kernel call does the IPC
            # exchange, the first probe launch does its hipFuncSetAttribute and
            # module load, and the clocks need to be up. None of that may land
            # in a timed iteration.
            stream = torch.cuda.current_stream(rank).cuda_stream
            for _ in range(20):
                out = custom_kernel(_clone_data(data))
                nullk.launch(0, stream, CU_COUNT, CTA_THREADS, LDS_FULL,
                             cells_p, err_p, sink_p)
                nullk.launch(1, stream, CU_COUNT, CTA_THREADS, LDS_FULL,
                             cells_p, err_p, sink_p)
            torch.cuda.synchronize()
            dist.barrier()

            samples = {arm: [] for arm in arms}
            for rep in range(reps):
                order = arms if rep % 2 == 0 else list(reversed(arms))
                for arm in order:
                    spec = ARM_SPECS[arm]
                    kind = spec["kind"]
                    do_clone = spec.get("clone", False)
                    do_flush = spec.get("flush", False)
                    grid = spec.get("grid", 0)
                    lds = spec.get("lds", 0)
                    probe = 0 if kind == "null" else 1
                    bucket = samples[arm]
                    S._FAST = fast_available and spec.get("fast", True)
                    for _ in range(iters):
                        if do_flush:
                            clear_l2_cache()
                        torch.cuda.synchronize()
                        dist.barrier()
                        t0 = time.perf_counter_ns()
                        payload = _clone_data(data) if do_clone else data
                        t_pay = time.perf_counter_ns()
                        if kind == "ours":
                            out = custom_kernel(payload)
                        elif kind != "none":
                            nullk.launch(probe, stream, grid, CTA_THREADS, lds,
                                         cells_p, err_p, sink_p)
                        t_iss = time.perf_counter_ns()
                        torch.cuda.synchronize()
                        t_syn = time.perf_counter_ns()
                        dist.barrier()
                        t1 = time.perf_counter_ns()
                        bucket.append((t0, t_pay, t_iss, t_syn, t1))

            # Correctness after the timed blocks, same oracle mp_smoke uses, so
            # no number is ever reported for a run that silently failed.
            x, w, bias = data
            partial = torch.matmul(x, w.T)
            if bias is not None:
                partial = partial + bias
            dist.all_reduce(partial)
            rows = m // WORLD
            want = partial[rank * rows:(rank + 1) * rows, :]
            loose = bool(torch.allclose(out, want, rtol=1e-2, atol=1e-2))
            tight = bool(torch.allclose(out, want, rtol=2e-3, atol=2e-3))
            diff = float((out.float() - want.float()).abs().max().item())
            say(f"    correctness 1e-2={loose} 2e-3={tight} "
                f"max|diff|={diff:.3e}")

            results[index] = {
                "shape": [m, n, k, int(has_bias)],
                "allclose_1e-2": loose,
                "allclose_2e-3": tight,
                "max_abs_diff": diff,
                "arms": {arm: samples[arm] for arm in arms},
            }
    except Exception:
        print(f"[rank {rank}] EXCEPTION\n{traceback.format_exc()}", flush=True)
        raise
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    with open(f"{out_path}.rank{rank}.json", "w") as handle:
        json.dump(results, handle)


def geo(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def decompose(per_rank, index, arm):
    """Split each iteration of one arm, aligned across the eight ranks.

    Iteration i is the same iteration on every rank: the ranks run one loop in
    lockstep with a barrier at both ends of every iteration.

    Device time is measured from `max_r(t_iss)` rather than from each rank's
    own issue, because the grid that starts last gates all eight: a producer on
    rank A writes into rank B's heap and B's reducers wait on it.
    """
    series = [per_rank[r][str(index)]["arms"][arm] for r in range(WORLD)]
    count = min(len(s) for s in series)
    rows = []
    for i in range(count):
        stamp = [dict(zip(STAMPS, series[r][i])) for r in range(WORLD)]
        t0s = [s["t0"] for s in stamp]
        iss = [s["t_iss"] for s in stamp]
        syn = [s["t_syn"] for s in stamp]
        rows.append({
            "total": (stamp[0]["t1"] - stamp[0]["t0"]) / 1e3,
            "total_max": max(s["t1"] - s["t0"] for s in stamp) / 1e3,
            "clone": statistics.mean(s["t_pay"] - s["t0"]
                                     for s in stamp) / 1e3,
            "host": statistics.mean(s["t_iss"] - s["t_pay"]
                                    for s in stamp) / 1e3,
            "host_max": max(s["t_iss"] - s["t_pay"] for s in stamp) / 1e3,
            "device": (max(syn) - max(iss)) / 1e3,
            "barrier": statistics.mean(s["t1"] - s["t_syn"]
                                       for s in stamp) / 1e3,
            "skew_t0": (max(t0s) - min(t0s)) / 1e3,
            "skew_iss": (max(iss) - min(iss)) / 1e3,
        })
    return rows


def report(per_rank, indices, arms):
    print("\n" + "=" * 118)
    print("exp_12: graded protocol, 8 processes, per-call cost split "
          "(median over iterations, us)")
    print("=" * 118)
    header = (f"{'#':>2}{'arm':>11}{'best':>9}{'med':>9}{'p90':>9}"
              f"{'clone':>8}{'host':>8}{'hostMx':>8}{'device':>9}"
              f"{'barrier':>9}{'resid':>7}{'skew0':>8}{'skewIss':>8}{'n':>5}")
    print(header)
    print("-" * len(header))
    table = {}
    for index in indices:
        for arm in arms:
            rows = decompose(per_rank, index, arm)
            totals = sorted(r["total"] for r in rows)
            med = lambda key: statistics.median(r[key] for r in rows)  # noqa: E731
            table[(index, arm)] = {
                "best": totals[0],
                "median": statistics.median(totals),
                "p90": totals[min(len(totals) - 1, int(0.9 * len(totals)))],
                "clone": med("clone"), "host": med("host"),
                "host_max": med("host_max"), "device": med("device"),
                "barrier": med("barrier"), "skew_t0": med("skew_t0"),
                "skew_iss": med("skew_iss"), "n": len(rows),
            }
            e = table[(index, arm)]
            # resid is what the four components do not explain: the wait for
            # the slowest rank to issue, which is real and is charged to the
            # score, but belongs to no single component.
            e["resid"] = (e["median"] - e["clone"] - e["host"] - e["device"]
                          - e["barrier"])
            print(f"{index+1:>2}{arm:>11}{e['best']:>9.2f}{e['median']:>9.2f}"
                  f"{e['p90']:>9.2f}{e['clone']:>8.2f}{e['host']:>8.2f}"
                  f"{e['host_max']:>8.2f}{e['device']:>9.2f}"
                  f"{e['barrier']:>9.2f}{e['resid']:>7.1f}{e['skew_t0']:>8.2f}"
                  f"{e['skew_iss']:>8.2f}{e['n']:>5}")
        print("-" * len(header))

    for arm in arms:
        present = [i for i in indices if (i, arm) in table]
        if len(present) == len(SCORED):
            print(f"geomean {arm:>11}: best={geo([table[(i, arm)]['best'] for i in present]):8.2f}"
                  f"  median={geo([table[(i, arm)]['median'] for i in present]):8.2f} us")
    return table


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "1,4"
    arm_arg = sys.argv[2] if len(sys.argv) > 2 else "all"
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    tag = sys.argv[5] if len(sys.argv) > 5 else "a"
    port = int(sys.argv[6]) if len(sys.argv) > 6 else 12610
    indices = [int(x) - 1 for x in which.split(",")]
    arms = list(ARM_SPECS) if arm_arg == "all" else arm_arg.split(",")
    for arm in arms:
        if arm not in ARM_SPECS:
            raise SystemExit(f"unknown arm {arm}; have {list(ARM_SPECS)}")
    os.makedirs("logs", exist_ok=True)
    out_path = f"logs/percall_{tag}"

    print(f"shapes={[i+1 for i in indices]} arms={arms} iters={iters} "
          f"reps={reps} tag={tag}", flush=True)
    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(WORLD):
        p = mp.Process(target=_worker, args=(rank, indices, arms, iters, reps,
                                             port, out_path))
        p.start()
        procs.append(p)
    deadline = time.time() + 1500
    for p in procs:
        p.join(timeout=max(5, deadline - time.time()))
    alive = [i for i, p in enumerate(procs) if p.is_alive()]
    if alive:
        print(f"TIMED OUT, ranks still alive: {alive}")
        for p in procs:
            if p.is_alive():
                p.terminate()
        return 1
    codes = [p.exitcode for p in procs]
    print(f"exit codes: {codes}", flush=True)
    if not all(c == 0 for c in codes):
        return 1

    per_rank = []
    for rank in range(WORLD):
        with open(f"{out_path}.rank{rank}.json") as handle:
            per_rank.append(json.load(handle))
    ok = all(per_rank[r][str(i)]["allclose_2e-3"]
             for r in range(WORLD) for i in indices)
    table = report(per_rank, indices, arms)
    print(f"\ncorrectness 2e-3 on all ranks and shapes: {ok}")
    with open(f"{out_path}.summary.json", "w") as handle:
        json.dump({f"{i+1}:{a}": v for (i, a), v in table.items()}, handle,
                  indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

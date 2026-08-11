"""exp_07 step 3: reproduce the GRADED protocol in our own 8-process harness.

The single-process harness and the official evaluator disagree by 3.4x on shape
4 and 2.7x on shape 6 while agreeing within 1.2x on the other four. exp_07 step
2 falsified the cold-L2 explanation in the single-process harness. The largest
untested structural difference is then: one process driving eight devices
(our harness) versus eight processes each driving one (the evaluator).

This file runs the evaluator's exact timed structure, eval.py:349-360,

    clear_l2_cache()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter_ns()
    output = custom_kernel(_clone_data(data, rank))
    torch.cuda.synchronize()
    dist.barrier()
    t1 = time.perf_counter_ns()

against the same `submission.custom_kernel` the evaluator imports, and ablates
its two non-kernel components so the cost can be attributed:

    full     clone + flush   (the graded protocol, verbatim)
    noclone  flush only      (is the timed clone the cost?)
    noflush  clone only      (does the flush matter multi-process?)
    bare     neither         (pure kernel under barrier/sync, no pipelining)

Arms are interleaved within a shape and the order is flipped every rep, so no
arm is systematically first. Nothing here writes to the kernel or to harness/.
"""

import json
import math
import os
import statistics
import sys
import time
import traceback

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

HARNESS = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/harness"
sys.path.insert(0, HARNESS)

WORLD = 8

SCORED = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]

ARMS = ["full", "noclone", "noflush", "bare"]

# mode "vs" pits our submission against the reference GEMM+RCCL through the
# identical graded protocol, in the same process pool, interleaved -- the only
# denominator that is actually same-run. Both are measured on the `full` arm
# because that is verbatim what the evaluator times.
VS_ARMS = ["ours:full", "ref:full"]


def ref_custom_kernel(data):
    """The reference implementation: torch.matmul + reduce_scatter_tensor.

    Semantically exp026's own submission.py, i.e. the naive fused-free baseline
    the evaluator scored at a geomean of 511.6 us (best) on this node.
    """
    x, w, bias = data
    rank = dist.get_rank()
    partial = torch.matmul(x, w.T)
    if bias is not None:
        partial = partial + bias
    rows = x.shape[0] // WORLD
    out = torch.empty((rows, w.shape[0]), dtype=partial.dtype,
                      device=partial.device)
    dist.reduce_scatter_tensor(out, partial.contiguous())
    return out


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


def _clone_data(data, rank):
    return tuple(t.clone() if torch.is_tensor(t) else t for t in data)


def _worker(rank, shape_indices, iters, reps, port, out_path, mode):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["HK_DEBUG"] = "0"

    def say(message):
        if rank == 0:
            print(f"[rank0] {message}", flush=True)

    import submission  # noqa: F401
    torch.cuda.set_device(rank)
    from submission import custom_kernel

    dist.init_process_group("nccl", init_method="env://", rank=rank,
                            world_size=WORLD,
                            device_id=torch.device(f"cuda:{rank}"))
    arms = VS_ARMS if mode == "vs" else ARMS
    impls = {"ours": custom_kernel, "ref": ref_custom_kernel}
    results = {}
    try:
        for index in shape_indices:
            m, n, k, has_bias, seed = SCORED[index]
            say(f"--- shape {index+1}: {m}x{n}x{k} bias={int(has_bias)}")
            data = generate_input(rank, WORLD, m, n, k, has_bias, seed)

            # Bring the shape's state up and warm the clocks. Also the first
            # call is where hk_submission does its IPC exchange, and the first
            # reduce_scatter is where RCCL builds its channels; neither may
            # land inside a timed iteration.
            for _ in range(20):
                custom_kernel(_clone_data(data, rank))
                if mode == "vs":
                    ref_custom_kernel(_clone_data(data, rank))
            torch.cuda.synchronize()
            dist.barrier()

            samples = {arm: [] for arm in arms}
            for rep in range(reps):
                order = arms if rep % 2 == 0 else list(reversed(arms))
                for arm in order:
                    if mode == "vs":
                        impl, protocol = arm.split(":")
                    else:
                        impl, protocol = "ours", arm
                    kernel = impls[impl]
                    do_clone = protocol in ("full", "noflush")
                    do_flush = protocol in ("full", "noclone")
                    for _ in range(iters):
                        if do_flush:
                            clear_l2_cache()
                        torch.cuda.synchronize()
                        dist.barrier()
                        t0 = time.perf_counter_ns()
                        payload = (_clone_data(data, rank) if do_clone
                                   else data)
                        out = kernel(payload)
                        torch.cuda.synchronize()
                        dist.barrier()
                        t1 = time.perf_counter_ns()
                        samples[arm].append((t1 - t0) / 1000.0)

            # Correctness on this shape, same oracle mp_smoke uses, so a number
            # can never be reported for a run that silently failed.
            x, w, bias = data
            partial = torch.matmul(x, w.T)
            if bias is not None:
                partial = partial + bias
            dist.all_reduce(partial)
            rows = m // WORLD
            want = partial[rank * rows:(rank + 1) * rows, :]
            ok = bool(torch.allclose(out, want, rtol=2e-2, atol=2e-2))
            diff = float((out.float() - want.float()).abs().max().item())

            results[index] = {
                "shape": [m, n, k, int(has_bias)],
                "allclose": ok,
                "max_abs_diff": diff,
                "arms": {arm: samples[arm] for arm in arms},
            }
            say(f"    correctness allclose={ok} max|diff|={diff:.3e}")
            for arm in arms:
                s = samples[arm]
                say(f"    {arm:>8}  best={min(s):9.2f} mean={statistics.mean(s):9.2f} "
                    f"med={statistics.median(s):9.2f} "
                    f"sd={statistics.stdev(s):8.2f} max={max(s):9.2f} us")
    except Exception:
        print(f"[rank {rank}] EXCEPTION\n{traceback.format_exc()}", flush=True)
        raise
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    with open(f"{out_path}.rank{rank}.json", "w") as handle:
        json.dump(results, handle, indent=2)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "4,5"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    out_path = sys.argv[4] if len(sys.argv) > 4 else "mp_graded"
    mode = sys.argv[5] if len(sys.argv) > 5 else "abl"
    indices = [int(x) - 1 for x in which.split(",")]
    arms = VS_ARMS if mode == "vs" else ARMS

    print(f"shapes={[i+1 for i in indices]} iters={iters} reps={reps} "
          f"mode={mode} arms={arms}")
    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(WORLD):
        p = mp.Process(target=_worker,
                       args=(rank, indices, iters, reps, 12401, out_path,
                             mode))
        p.start()
        procs.append(p)

    deadline = time.time() + 1200
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
    print(f"exit codes: {codes}")
    if not all(c == 0 for c in codes):
        return 1

    # Rank 0's clock is the evaluator's clock, but the max over ranks is what
    # the slowest rank actually experienced; both are reported.
    per_rank = []
    for rank in range(WORLD):
        with open(f"{out_path}.rank{rank}.json") as handle:
            per_rank.append(json.load(handle))

    print("\n" + "=" * 104)
    print("exp_07 step 3: GRADED protocol, 8 processes, our submission")
    print("=" * 104)
    header = (f"{'#':>2}{'shape':>22}{'arm':>11}{'best':>10}{'mean':>10}"
              f"{'median':>10}{'sd%':>7}{'max':>10}{'ok':>4}")
    print(header)
    print("-" * len(header))
    best_by_arm = {arm: [] for arm in arms}
    mean_by_arm = {arm: [] for arm in arms}
    for index in indices:
        entry = per_rank[0][str(index)]
        m, n, k, bias = entry["shape"]
        ok = all(per_rank[r][str(index)]["allclose"] for r in range(WORLD))
        for arm in arms:
            s = entry["arms"][arm]
            best_by_arm[arm].append(min(s))
            mean_by_arm[arm].append(statistics.mean(s))
            print(f"{index+1:>2}{f'{m}x{n}x{k}':>22}{arm:>11}"
                  f"{min(s):>10.2f}{statistics.mean(s):>10.2f}"
                  f"{statistics.median(s):>10.2f}"
                  f"{100*statistics.stdev(s)/statistics.mean(s):>7.1f}"
                  f"{max(s):>10.2f}{'yes' if ok else 'NO':>4}")
        print("-" * len(header))
    if len(indices) == len(SCORED):
        def geo(values):
            return math.exp(sum(math.log(v) for v in values) / len(values))
        print("geometric mean over all six graded shapes:")
        for arm in arms:
            print(f"  {arm:>11}  best={geo(best_by_arm[arm]):9.2f} us   "
                  f"mean={geo(mean_by_arm[arm]):9.2f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())

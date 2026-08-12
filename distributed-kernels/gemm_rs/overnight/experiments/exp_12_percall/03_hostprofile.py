"""exp_12 step 1b: itemize the host path through custom_kernel.

The graded decomposition leaves exactly one reducible term that is ours: the
20-36 us spent in python + pybind + hipLaunchKernel between `t0` and the point
the kernel is enqueued. Under the graded protocol nothing overlaps it, so every
microsecond is charged to the score.

This runs in the real eight-process pool -- eight python interpreters spinning
on one host is itself part of the cost and cannot be reproduced single-process.
Each component is timed in a tight loop; the launch components enqueue without
synchronizing, which is what the graded protocol's host path does too.

Usage: python3 03_hostprofile.py <shape_index> <iters> <port>
"""

import json
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


def generate_input(rank, m, n, k, has_bias, seed):
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)
    local_k = k // WORLD
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


def _worker(rank, shape_index, iters, port, out_path):
    err_file = open(f"{out_path}.rank{rank}.stderr", "wb", 0)
    os.dup2(err_file.fileno(), 2)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("HK_DEBUG", "0")
    torch.cuda.set_device(rank)

    import submission as S
    dist.init_process_group("nccl", init_method="env://", rank=rank,
                            world_size=WORLD,
                            device_id=torch.device(f"cuda:{rank}"))
    timings = {}
    try:
        m, n, k, has_bias, seed = SCORED[shape_index]
        data = generate_input(rank, m, n, k, has_bias, seed)
        for _ in range(20):
            S.custom_kernel(data)
        torch.cuda.synchronize()
        dist.barrier()

        x, w, bias = data
        key = (rank, m, n, k, bias is not None)
        state = S._STATES[key]
        plan = state.plan
        stream = torch.cuda.current_stream(rank).cuda_stream
        entry = S._entry
        Tensor = S.Tensor

        def bench(name, fn):
            fn()
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter_ns()
            for _ in range(iters):
                fn()
            elapsed = (time.perf_counter_ns() - start) / 1e3 / iters
            torch.cuda.synchronize()
            timings[name] = elapsed

        bench("dist.get_rank", lambda: dist.get_rank())
        bench("set_device", lambda: torch.cuda.set_device(rank))
        bench("_get_default_group",
              lambda: dist.distributed_c10d._get_default_group())
        bench("current_stream.cuda_stream",
              lambda: torch.cuda.current_stream(rank).cuda_stream)
        bench("data_ptr x3",
              lambda: (x.data_ptr(), w.data_ptr(), state.out.data_ptr()))
        bench("Tensor() x2",
              lambda: (Tensor(1, (1, 1, m, k // WORLD)),
                       Tensor(1, (1, 1, n, k // WORLD))))
        bench("8 int(plan[..])",
              lambda: (int(plan["lrow_count"]), int(plan["col_count"]),
                       int(plan["eb"]), int(plan["num_gemm_ctas"]),
                       int(plan["ready_words"]),
                       1 if plan["packet_fast_path"] else 0,
                       int(plan["config_row"]),
                       1 if plan["even_k"] else 0))
        bench("3 f-strings", lambda: (f"enter custom_kernel key={key}",
                                      f"cache hit settled=True "
                                      f"keys={len(S._STATES)}",
                                      "state ready"))
        bench("state.launch", lambda: state.launch(x, w, bias))
        bench("custom_kernel", lambda: S.custom_kernel(data))

        # The floor: the same pybind entry with every argument prebuilt, only
        # the three data pointers refreshed. This is what the fast path can be.
        xt = Tensor(x.data_ptr(), (1, 1, m, k // WORLD))
        wt = Tensor(w.data_ptr(), (1, 1, n, k // WORLD))
        args = [xt, wt, state.c_view, state.out_view, state.desc_c,
                state.sig_local, state.desc_sig, state.ep_cell,
                0 if bias is None else bias.data_ptr(), state.err,
                stream, S.SPIN_LIMIT, rank, m, n, k // WORLD,
                int(plan["lrow_count"]), int(plan["col_count"]),
                int(plan["eb"]), int(plan["num_gemm_ctas"]),
                int(plan["ready_words"]),
                1 if plan["packet_fast_path"] else 0, 0, 0, 0,
                int(plan["config_row"]), 1 if plan["even_k"] else 0]

        def prebuilt():
            xt._pointer = x.data_ptr()
            wt._pointer = w.data_ptr()
            entry(*args)

        bench("PREBUILT entry(*args)", prebuilt)
        bench("entry(*args) no ptr refresh", lambda: entry(*args))
    except Exception:
        print(f"[rank {rank}] EXCEPTION\n{traceback.format_exc()}", flush=True)
        raise
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    with open(f"{out_path}.rank{rank}.json", "w") as handle:
        json.dump(timings, handle, indent=2)


def main():
    shape_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 12620
    os.makedirs("logs", exist_ok=True)
    out_path = f"logs/hostprof_s{shape_index+1}"

    print(f"host-path profile, shape {shape_index+1}={SCORED[shape_index]}, "
          f"{iters} iters, 8 processes", flush=True)
    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(WORLD):
        p = mp.Process(target=_worker,
                       args=(rank, shape_index, iters, port, out_path))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=900)
    codes = [p.exitcode for p in procs]
    print(f"exit codes: {codes}", flush=True)
    if not all(c == 0 for c in codes):
        return 1

    per_rank = []
    for rank in range(WORLD):
        with open(f"{out_path}.rank{rank}.json") as handle:
            per_rank.append(json.load(handle))
    print("\n" + "=" * 74)
    print("host path through custom_kernel, us per call (8 concurrent ranks)")
    print("=" * 74)
    print(f"{'component':<32}{'rank0':>10}{'mean8':>10}{'max8':>10}")
    print("-" * 62)
    for name in per_rank[0]:
        values = [per_rank[r][name] for r in range(WORLD)]
        print(f"{name:<32}{values[0]:>10.2f}"
              f"{statistics.mean(values):>10.2f}{max(values):>10.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

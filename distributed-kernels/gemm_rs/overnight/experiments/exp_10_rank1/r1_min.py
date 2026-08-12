"""Minimal 8-rank driver for the frozen rank-1 submission.

Why this exists: every previous attempt drove rank-1 through eval.py, where any
failure surfaces ~25 minutes later as an opaque `el.get(60)` timeout. This does
the smallest thing that exercises the same code path -- one shape, one
custom_kernel call, one correctness check -- and it captures each rank's stderr
into its own file at the file-descriptor level, so a GPU memory fault (which
aborts the process from the HSA runtime, bypassing Python) is still readable
afterwards.

Usage: python3 r1_min.py <shape_index> [iters]
"""

import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8

# The six graded benchmark shapes, verbatim from task.yml `benchmarks:`.
SHAPES = [
    dict(m=64,   n=7168, k=18432, has_bias=False, seed=1234),
    dict(m=512,  n=4096, k=12288, has_bias=True,  seed=663),
    dict(m=2048, n=2880, k=2880,  has_bias=True,  seed=166),
    dict(m=4096, n=4096, k=4096,  has_bias=False, seed=1371),
    dict(m=8192, n=4096, k=14336, has_bias=True,  seed=7168),
    dict(m=8192, n=8192, k=29568, has_bias=False, seed=42),
]


def _clone_data(data, rank):
    device = torch.device(f"cuda:{rank}")
    return tuple(
        t.clone().to(device) if torch.is_tensor(t) else t for t in data
    )


def _worker(rank, shape_index, iters, port, outdir, queue):
    # fd-level redirect: a GPU fault aborts the process from the runtime, so
    # anything buffered in Python's sys.stderr object would be lost.
    err = open(os.path.join(outdir, f"stderr_rank{rank}.txt"), "wb", 0)
    os.dup2(err.fileno(), 2)
    out = open(os.path.join(outdir, f"stdout_rank{rank}.txt"), "wb", 0)
    os.dup2(out.fileno(), 1)

    stage = "start"
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(rank)

        stage = "init_process_group"
        dist.init_process_group(
            "nccl", init_method="env://", rank=rank, world_size=WORLD,
            device_id=torch.device(f"cuda:{rank}"),
        )

        stage = "generate_input"
        from reference import generate_input, check_implementation
        args = dict(SHAPES[shape_index])
        data = generate_input(rank=rank, world_size=WORLD, **args)
        check_copy = _clone_data(data, rank)

        stage = "import submission"
        print(f"[rank {rank}] importing submission", flush=True)
        from submission import custom_kernel

        stage = "custom_kernel #1"
        print(f"[rank {rank}] calling custom_kernel, shape={args}", flush=True)
        t0 = time.perf_counter()
        output = custom_kernel(_clone_data(data, rank))
        stage = "synchronize after #1"
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"[rank {rank}] custom_kernel #1 returned in "
              f"{(t1 - t0) * 1e3:.1f} ms, out={tuple(output.shape)} "
              f"{output.dtype} {output.device}", flush=True)

        stage = "check_implementation"
        good, message = check_implementation(check_copy, output)
        print(f"[rank {rank}] correctness good={good} msg={message[:200]}",
              flush=True)

        durations = []
        for i in range(iters):
            stage = f"custom_kernel iter {i}"
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = custom_kernel(_clone_data(data, rank))
            torch.cuda.synchronize()
            durations.append((time.perf_counter() - t0) * 1e6)
        if durations:
            print(f"[rank {rank}] {iters} iters, best={min(durations):.2f} us",
                  flush=True)

        queue.put((rank, "OK", good, message[:300], durations))

        stage = "destroy_process_group"
        dist.destroy_process_group()
        print(f"[rank {rank}] destroyed process group cleanly", flush=True)
    except BaseException as exc:
        traceback.print_exc()
        sys.stderr.flush()
        print(f"[rank {rank}] FAILED at stage={stage}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        queue.put((rank, "FAIL", stage, f"{type(exc).__name__}: {exc}", []))
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


def main():
    shape_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 12377
    outdir = os.environ.get("R1_OUTDIR", "r1min_logs")
    os.makedirs(outdir, exist_ok=True)

    print(f"driver: shape[{shape_index}]={SHAPES[shape_index]} iters={iters} "
          f"port={port} outdir={outdir}", flush=True)
    print(f"driver: torch {torch.__version__} devices "
          f"{torch.cuda.device_count()}", flush=True)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for rank in range(WORLD):
        p = ctx.Process(target=_worker,
                        args=(rank, shape_index, iters, port, outdir, queue))
        p.start()
        procs.append(p)

    results = {}
    deadline = time.time() + float(os.environ.get("R1_TIMEOUT", "420"))
    while len(results) < WORLD and time.time() < deadline:
        try:
            item = queue.get(timeout=5)
        except Exception:
            alive = [p.pid for p in procs if p.is_alive()]
            if not alive:
                print("driver: all children exited without reporting",
                      flush=True)
                break
            continue
        results[item[0]] = item

    for p in procs:
        p.join(timeout=20)
    for p in procs:
        if p.is_alive():
            print(f"driver: terminating still-alive pid {p.pid}", flush=True)
            p.terminate()
            p.join(timeout=10)

    print("\n===== driver summary =====", flush=True)
    print(f"reported: {len(results)}/{WORLD} ranks", flush=True)
    for rank in range(WORLD):
        if rank in results:
            item = results[rank]
            print(f"  rank {rank}: {item[1]} {item[2]} {item[3]}", flush=True)
        else:
            code = procs[rank].exitcode
            print(f"  rank {rank}: NO REPORT exitcode={code}", flush=True)
    ok = all(results.get(r, (None, "X"))[1] == "OK" and results[r][2] is True
             for r in range(WORLD))
    print(f"ALL_RANKS_OK={ok}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

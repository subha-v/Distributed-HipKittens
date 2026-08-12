"""Bisect rank-1's fault: is it the C++ dist_barrier or the Triton GEMM?

The heaps are all mapped rw-s and host-side peer writes succeed, so the IPC
mapping is sound. This runs the two device-side peer writers separately, in
order, with the pointer table and /proc/self/maps captured in the SAME process
so the fault address can be attributed to a region afterwards.

Stages, each reported before it is attempted:
  1. load_heap_base_ptr  -> C++ init_shmem (hipExtMallocWithFlags + IPC)
  2. torch.ops.my_ops.barrier(rank) -> the dist_barrier kernel, peer atomics
  3. custom_kernel -> the Triton GEMM epilogue + reduce
"""

import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
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
    return tuple(t.clone().to(device) if torch.is_tensor(t) else t
                 for t in data)


def _worker(rank, shape_index, port, outdir, queue):
    err = open(os.path.join(outdir, f"stderr_rank{rank}.txt"), "wb", 0)
    os.dup2(err.fileno(), 2)
    out = open(os.path.join(outdir, f"stdout_rank{rank}.txt"), "wb", 0)
    os.dup2(out.fileno(), 1)

    def say(msg):
        print(f"[rank {rank}] {msg}", flush=True)

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

        stage = "import submission"
        import submission
        from reference import generate_input, check_implementation

        args = dict(SHAPES[shape_index])
        data = generate_input(rank=rank, world_size=WORLD, **args)
        check_copy = _clone_data(data, rank)
        payload = _clone_data(data, rank)

        # ---- stage 1: the C++ init_shmem / IPC bootstrap -------------------
        stage = "1_load_heap_base_ptr"
        say("STAGE 1 load_heap_base_ptr (C++ init_shmem)")
        base_ptr = submission.load_heap_base_ptr(payload[0])
        torch.cuda.synchronize()
        hack = submission.A_ptr_index_hack.data_ptr()
        heaps = [base_ptr[i].item() for i in range(WORLD)]
        say(f"STAGE 1 OK. hack=0x{hack:x}")
        for i, p in enumerate(heaps):
            say(f"  heap[{i}] = 0x{p:x} .. 0x{p + (1 << 30):x} "
                f"(1 GiB)  base_addr={submission.base_addrs[i]}")
        say(f"  input a.data_ptr()=0x{payload[0].data_ptr():x} "
            f"b=0x{payload[1].data_ptr():x}")
        with open(os.path.join(outdir, f"maps_rank{rank}.txt"), "w") as fh:
            fh.write(open("/proc/self/maps").read())

        # ---- stage 2: the dist_barrier kernel, peer atomic writes ----------
        stage = "2_my_ops_barrier"
        dist.barrier()
        say("STAGE 2 torch.ops.my_ops.barrier (dist_barrier kernel, "
            "peer atomic stores)")
        t0 = time.perf_counter()
        torch.ops.my_ops.barrier(rank)
        torch.cuda.synchronize()
        say(f"STAGE 2 OK in {(time.perf_counter() - t0) * 1e3:.1f} ms")

        # ---- stage 3: the full kernel -------------------------------------
        stage = "3_custom_kernel"
        dist.barrier()
        say(f"STAGE 3 custom_kernel shape={args}")
        t0 = time.perf_counter()
        output = submission.custom_kernel(payload)
        torch.cuda.synchronize()
        say(f"STAGE 3 OK in {(time.perf_counter() - t0) * 1e3:.1f} ms "
            f"out={tuple(output.shape)}")

        stage = "4_check"
        good, message = check_implementation(check_copy, output)
        say(f"STAGE 4 correctness good={good} msg={message[:300]}")

        queue.put((rank, "OK", stage, f"good={good}", message[:300]))
        dist.destroy_process_group()
        say("clean exit")
    except BaseException as exc:
        traceback.print_exc()
        say(f"FAILED at stage={stage}: {type(exc).__name__}: {exc}")
        queue.put((rank, "FAIL", stage, f"{type(exc).__name__}: {exc}", ""))
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


def main():
    shape_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 12399
    outdir = os.environ.get("R1_OUTDIR", "r1bisect_logs")
    os.makedirs(outdir, exist_ok=True)
    print(f"driver: shape[{shape_index}]={SHAPES[shape_index]}", flush=True)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for rank in range(WORLD):
        p = ctx.Process(target=_worker,
                        args=(rank, shape_index, port, outdir, queue))
        p.start()
        procs.append(p)

    results = {}
    deadline = time.time() + float(os.environ.get("R1_TIMEOUT", "300"))
    while len(results) < WORLD and time.time() < deadline:
        try:
            item = queue.get(timeout=5)
        except Exception:
            if not any(p.is_alive() for p in procs):
                break
            continue
        results[item[0]] = item

    for p in procs:
        p.join(timeout=15)
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)

    print("\n===== driver summary =====", flush=True)
    for rank in range(WORLD):
        item = results.get(rank)
        if item:
            print(f"  rank {rank}: {item[1]} stage={item[2]} {item[3]}",
                  flush=True)
        else:
            print(f"  rank {rank}: NO REPORT exitcode={procs[rank].exitcode}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

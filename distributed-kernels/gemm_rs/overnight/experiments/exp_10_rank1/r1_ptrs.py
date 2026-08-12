"""Dump rank-1's symmetric-heap pointer table and its page permissions.

The fault is `Write access to a read-only page` during the first peer write.
This stops immediately after rank-1's own `load_heap_base_ptr` (which runs its
C++ init: hipExtMallocWithFlags(finegrained) -> hipIpcGetMemHandle -> file
rendezvous -> hipIpcOpenMemHandle x7) and asks the kernel what permissions
those mappings actually have, before any GEMM is launched.
"""

import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8


def maps_for(addr):
    """Which /proc/self/maps region contains addr, and with what perms?"""
    try:
        with open("/proc/self/maps") as handle:
            for line in handle:
                span = line.split()[0]
                lo, hi = (int(x, 16) for x in span.split("-"))
                if lo <= addr < hi:
                    return line.rstrip()
    except Exception as exc:
        return f"<maps read failed: {exc}>"
    return "<no mapping contains this address>"


def _worker(rank, port, outdir, queue):
    err = open(os.path.join(outdir, f"ptr_stderr_rank{rank}.txt"), "wb", 0)
    os.dup2(err.fileno(), 2)
    out = open(os.path.join(outdir, f"ptr_stdout_rank{rank}.txt"), "wb", 0)
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

        stage = "import submission"
        import submission

        stage = "load_heap_base_ptr"
        probe = torch.empty(8, dtype=torch.bfloat16,
                            device=torch.device(f"cuda:{rank}"))
        base_ptr = submission.load_heap_base_ptr(probe)
        torch.cuda.synchronize()

        hack = submission.A_ptr_index_hack.data_ptr()
        lines = [f"[rank {rank}] A_ptr_index_hack = 0x{hack:x}",
                 f"[rank {rank}]   maps: {maps_for(hack)}"]
        for i in range(WORLD):
            p = base_ptr[i].item()
            delta = p - hack
            lines.append(
                f"[rank {rank}] heap[{i}] = 0x{p:x}  "
                f"delta={delta}  delta/2={delta // 2}  "
                f"fits_int32={-2**31 <= delta // 2 < 2**31}")
            lines.append(f"[rank {rank}]   maps: {maps_for(p)}")
        lines.append(f"[rank {rank}] base_addrs = {submission.base_addrs}")
        print("\n".join(lines), flush=True)

        # Does a plain torch write to the peer heap work? This is the exact
        # operation the barrier kernel performs, minus rank-1's code.
        stage = "peer write probe"
        import ctypes
        lib = ctypes.CDLL("libamdhip64.so")
        # ctypes defaults arguments to C int, which truncates 64-bit device
        # pointers; declare them explicitly or the probe tests garbage.
        lib.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int]
        lib.hipMemcpy.restype = ctypes.c_int
        D2H, H2D = 2, 1
        results = []
        for i in range(WORLD):
            p = base_ptr[i].item()
            try:
                buf = ctypes.c_int32(0)
                rc_r = lib.hipMemcpy(ctypes.byref(buf), ctypes.c_void_p(p),
                                     4, D2H)
                val = buf.value
                buf2 = ctypes.c_int32(0x5A5A)
                rc_w = lib.hipMemcpy(ctypes.c_void_p(p), ctypes.byref(buf2),
                                     4, H2D)
                rc_r2 = lib.hipMemcpy(ctypes.byref(buf), ctypes.c_void_p(p),
                                      4, D2H)
                results.append(
                    f"heap[{i}] @0x{p:x}: read rc={rc_r} val={val} | "
                    f"write rc={rc_w} | reread rc={rc_r2} val={buf.value} "
                    f"{'WRITE_OK' if buf.value == 0x5A5A else 'WRITE_LOST'}")
            except Exception as exc:
                results.append(f"heap[{i}]: probe raised {exc!r}")
        print(f"[rank {rank}] host-side D2D probe:\n  " +
              "\n  ".join(results), flush=True)

        queue.put((rank, "OK", [base_ptr[i].item() for i in range(WORLD)],
                   hack, results))
        dist.destroy_process_group()
    except BaseException as exc:
        traceback.print_exc()
        print(f"[rank {rank}] FAILED at {stage}: {type(exc).__name__}: {exc}",
              flush=True)
        queue.put((rank, "FAIL", stage, f"{type(exc).__name__}: {exc}", []))
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 12388
    outdir = os.environ.get("R1_OUTDIR", "r1ptr_logs")
    os.makedirs(outdir, exist_ok=True)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for rank in range(WORLD):
        p = ctx.Process(target=_worker, args=(rank, port, outdir, queue))
        p.start()
        procs.append(p)

    results = {}
    deadline = time.time() + 240
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
        if p.is_alive():
            p.terminate()

    print("\n===== summary =====", flush=True)
    for rank in range(WORLD):
        item = results.get(rank)
        print(f"  rank {rank}: {item[1] if item else 'NO REPORT'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

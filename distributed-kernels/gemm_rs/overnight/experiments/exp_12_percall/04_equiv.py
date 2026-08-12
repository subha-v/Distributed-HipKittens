"""exp_12: prove the prebound launch path is byte-identical to the old one.

Correctness against a torch oracle is not the interesting test here -- both
paths run the SAME kernel, so the only thing that can differ is the
mi300x_globals POD they build, and a wrong field would most likely still pass
allclose on some shapes. So this compares the two paths' outputs BIT-EXACTLY on
the same inputs, in the same eight-process pool, on all six graded shapes.

Both arms launch exactly once per rank per shape, so the per-CTA device-derived
epochs stay in lockstep across ranks, which is the one thing the protocol
requires of its caller.

Usage: python3 04_equiv.py <port>
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ON = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight"
sys.path.insert(0, f"{ON}/harness")

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


def _worker(rank, port, out_path):
    err_file = open(f"{out_path}.rank{rank}.stderr", "wb", 0)
    os.dup2(err_file.fileno(), 2)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["HK_DEBUG"] = "0"
    torch.cuda.set_device(rank)

    import submission as S
    dist.init_process_group("nccl", init_method="env://", rank=rank,
                            world_size=WORLD,
                            device_id=torch.device(f"cuda:{rank}"))
    lines = []
    ok_all = True
    try:
        if S._run is None:
            print(f"[rank {rank}] FAIL: module has no prebound entry")
            raise SystemExit(2)
        for index, (m, n, k, has_bias, seed) in enumerate(SCORED):
            data = generate_input(rank, m, n, k, has_bias, seed)
            x, w, bias = data
            for _ in range(3):
                S.custom_kernel(data)
            torch.cuda.synchronize()
            dist.barrier()
            state = S._STATES[(rank, m, n, k, bias is not None)]

            state.out.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            state.launch(x, w, bias)
            torch.cuda.synchronize()
            dist.barrier()
            ref = state.out.clone()

            state.out.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            state.launch_fast(x, w, bias)
            torch.cuda.synchronize()
            dist.barrier()
            got = state.out.clone()

            same = bool(torch.equal(ref, got))
            nonzero = bool(ref.abs().sum().item() > 0)
            bits = state.error_bits()
            worst = float((ref.float() - got.float()).abs().max().item())
            ok = same and nonzero and bits == 0
            ok_all = ok_all and ok
            lines.append(f"  shape {index+1} {m}x{n}x{k} bias={int(has_bias)}: "
                         f"bit_identical={same} ref_nonzero={nonzero} "
                         f"errbits={bits} max|diff|={worst:.3e} "
                         f"{'OK' if ok else 'FAIL'}")
    except Exception:
        print(f"[rank {rank}] EXCEPTION\n{traceback.format_exc()}", flush=True)
        raise
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    with open(f"{out_path}.rank{rank}.txt", "w") as handle:
        handle.write(f"rank {rank} all_ok={ok_all}\n" + "\n".join(lines) + "\n")
    if rank == 0:
        print(f"rank 0 all_ok={ok_all}")
        print("\n".join(lines), flush=True)
    return 0 if ok_all else 3


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 12630
    os.makedirs("logs", exist_ok=True)
    out_path = "logs/equiv"
    print("prebound vs duck-typed launch: bit-exact equivalence, 6 shapes",
          flush=True)
    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(WORLD):
        p = mp.Process(target=_worker, args=(rank, port, out_path))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=900)
    codes = [p.exitcode for p in procs]
    print(f"exit codes: {codes}", flush=True)
    verdict = True
    for rank in range(WORLD):
        try:
            text = open(f"{out_path}.rank{rank}.txt").read()
        except OSError:
            print(f"rank {rank}: NO REPORT")
            verdict = False
            continue
        if "all_ok=True" not in text or "FAIL" in text:
            print(f"rank {rank} REPORT:\n{text}")
            verdict = False
    print(f"\nEQUIVALENCE {'PASS' if verdict and all(c == 0 for c in codes) else 'FAIL'}")
    return 0 if verdict and all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())

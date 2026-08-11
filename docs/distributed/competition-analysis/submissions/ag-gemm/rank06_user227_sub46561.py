#!/usr/bin/env python3
# ag_gemm_single.py
# Contains both the submission kernel AND a torchrun runner.
# The harness will only care about `custom_kernel`.

import os
# ---- Set env BEFORE importing torch/distributed ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NCCL_DEBUG", "WARN")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("RCCL_P2P_ENABLE", "1")
os.environ.setdefault("HSA_FORCE_FINE_GRAIN_PCIE", "1")

from datetime import timedelta
import argparse
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F


# ---------------- Submission kernel ----------------
@torch.no_grad()
def custom_kernel(data):
    """
    Stable AG-GEMM (matches the reference contract exactly):
      1) AllGather(local inputs) -> [local_M * world_size, K]
      2) One GEMM per rank: out = full_input @ weight.T (+ bias)

    Args:
        data: tuple(x_local, weight, bias)
              x_local: [local_M, K]
              weight : [local_N, K]
              bias   : [local_N] or None
    Returns:
        out: [local_M * world_size, local_N]
    """
    x_local, w, b = data

    # World size (fallback to 1 if dist isn't initialized)
    if dist.is_available() and dist.is_initialized():
        ws = dist.get_world_size()
    else:
        ws = 1

    n_local, k = x_local.shape

    # Single collective with identical send shapes across ranks (exact ref behavior)
    full_input = torch.empty((n_local * ws, k), dtype=x_local.dtype, device=x_local.device)
    if ws > 1:
        # This will obey the process-group timeout set by the harness; the env above
        # helps make failures raise instead of silently hanging.
        dist.all_gather_into_tensor(full_input, x_local)
    else:
        full_input.copy_(x_local)

    # Post-collective: lean GEMM. Keep it contiguous-friendly if needed.
    if not w.is_contiguous():
        w = w.contiguous()
    if b is not None and not b.is_contiguous():
        b = b.contiguous()

    # One big GEMM (+ optional bias), bf16 path is fast on MI300
    # F.linear(full_input, w, b) == full_input @ w.T (+ b)
    out = F.linear(full_input, w, b)
    return out
# ---------------------------------------------------


# ---------------- Runner utilities -----------------
def init_dist():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=60),
    )
    return rank, local_rank, world_size


def rccl_health_check():
    if not (dist.is_available() and dist.is_initialized()):
        return
    rank = dist.get_rank()
    ws = dist.get_world_size()
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    x = torch.tensor([rank], device=dev, dtype=torch.int32)
    buf = torch.empty(ws, device=dev, dtype=x.dtype)
    dist.all_gather_into_tensor(buf, x)
    assert int(buf.sum().item()) == ws * (ws - 1) // 2, "RCCL health check failed"


def generate_input(rank, world_size, m, n, k, has_bias, seed):
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)
    local_m = m // world_size
    local_n = n // world_size
    x = (torch.rand((local_m, k), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01
    w = (torch.rand((local_n, k), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01
    b = None
    if has_bias:
        b = (torch.rand((local_n,), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01
    return (x, w, b)
# ---------------------------------------------------


# ---------------- Main for local runs --------------
def main():
    ap = argparse.ArgumentParser(description="One-file AG-GEMM runner")
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    rank, local_rank, world_size = init_dist()
    rccl_health_check()
    dist.barrier()

    data = generate_input(local_rank, world_size, args.m, args.n, args.k, args.bias, args.seed)

    # Warmup
    for _ in range(args.warmup):
        _ = custom_kernel(data)
        dist.barrier()

    # Timed
    times = []
    for _ in range(args.repeat):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = custom_kernel(data)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    if rank == 0:
        print(f"[m={args.m} n={args.n} k={args.k} bias={args.bias}] "
              f"avg_ms={sum(times)/len(times):.3f} p50_ms={sorted(times)[len(times)//2]:.3f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

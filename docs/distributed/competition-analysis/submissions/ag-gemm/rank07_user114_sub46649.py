#!/usr/bin/env python3
# ag_gemm_single.py
# Contains both the submission kernel AND a torchrun runner.
# The harness will only care about `custom_kernel`.

import os
# ---- Set env BEFORE importing torch/distributed ----
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TORCH_NCCL_DEBUG", "WARN")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("RCCL_P2P_ENABLE", "1")
os.environ.setdefault("HSA_FORCE_FINE_GRAIN_PCIE", "1")

#os.environ.setdefault("NCCL_IB_DISABLE","0")
#os.environ.setdefault("NCCL_P2P_DISABLE","0")
#os.environ.setdefault("NCCL_SHM_DISABLE","0")

os.environ.setdefault("NCCL_IB_HCA","mlx5_0")   # pick HCA from ibv_devinfo
os.environ.setdefault("NCCL_IB_GID_INDEX","3")  # typical for RoCE/IB networks
os.environ.setdefault("NCCL_IB_DISABLE","0")

#os.environ.setdefault("NCCL_NTHREADS","512")
#os.environ.setdefault("NCCL_BUFFSIZE","2097152")

#os.environ.setdefault("NCCL_SOCKET_IFNAME","ib0")

os.environ.setdefault("NCCL_DEBUG","INFO")
os.environ.setdefault("NCCL_DEBUG_SUBSYS","NET")


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
    ws = dist.get_world_size()

    n_local, k = x_local.shape

    # Single collective with identical send shapes across ranks (exact ref behavior)
    full_input = torch.empty((n_local * ws, k), dtype=x_local.dtype, device=x_local.device)
    dist.all_gather_into_tensor(full_input, x_local)


    # One big GEMM (+ optional bias), bf16 path is fast on MI300
    # F.linear(full_input, w, b) == full_input @ w.T (+ b)
    out = F.linear(full_input, w, b)
    return out

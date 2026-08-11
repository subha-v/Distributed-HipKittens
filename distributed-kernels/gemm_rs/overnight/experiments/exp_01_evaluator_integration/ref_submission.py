"""CONTROL submission: plain torch.matmul + reduce_scatter, no IPC, no globals.

Exists for exactly one question: does a second init_process_group in the same
worker process hang even when our module is not involved? If this control hangs
at mp_smoke's case 1 the same way hk_submission does, the fault is in
torch/NCCL/the container and not in our retained state.
"""

import torch
import torch.distributed as dist

WORLD = 8


def custom_kernel(data):
    x, w, bias = data
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    partial = torch.matmul(x, w.T)
    if bias is not None:
        partial = partial + bias
    rows = x.shape[0] // WORLD
    out = torch.empty((rows, w.shape[0]), dtype=partial.dtype,
                      device=partial.device)
    dist.reduce_scatter_tensor(out, partial.contiguous())
    return out

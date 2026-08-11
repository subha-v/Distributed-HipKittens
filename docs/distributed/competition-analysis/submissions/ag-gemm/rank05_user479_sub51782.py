# ---- Perf env: set BEFORE importing torch/distributed ----
import torch.nn.functional as F
import torch.distributed as dist
import torch
from task import input_t, output_t
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ROCm / RCCL knobs (PyTorch honors NCCL_* aliases on ROCm too)
# "INFO" for verbose debug
os.environ.setdefault("RCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_DEBUG", "WARN")                 # alias; optional
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING",
                      "1")     # better failure handling
# stable debug; set "0" for max perf
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
# enable P2P when topology allows
os.environ.setdefault("RCCL_P2P_ENABLE", "1")
os.environ.setdefault("HSA_FORCE_FINE_GRAIN_PCIE",
                      "1")     # fine-grain PCIe on ROCm


@torch.no_grad()
def custom_kernel(data: input_t) -> output_t:
    full_input = torch.empty(
        (data[0].shape[0] * 8, data[0].shape[1]),
        dtype=data[0].dtype,
        device=data[0].device
    )
    dist.all_gather_into_tensor(full_input, data[0])
    return F.linear(full_input, data[1], data[2])

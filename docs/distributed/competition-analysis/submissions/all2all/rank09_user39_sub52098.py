import torch
import torch.distributed as dist
from task import input_t, output_t
from reference import MoEConfig

def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world = data
    device = rank_data.x.device
    T, H = rank_data.x.shape
    K = rank_data.indices.shape[1]

    # 目的 rank：用线性映射更稳，避免 num_experts % world 的坑
    # dst[t, k] in [0, world-1]
    dst = torch.div(
        rank_data.indices.to(torch.int64) * world,  # [T, K]
        cfg.num_experts,
        rounding_mode='floor'
    ).to(torch.int32)

    # 每个 token 的总缩放系数：sum_k w[t,k] * (1 + dst_rank)
    w = rank_data.weights.to(torch.float32)                     # [T, K]
    gain = (w * (1 + dst).to(torch.float32)).sum(dim=1)         # [T]

    # 输出：x[t] * gain[t]
    out = rank_data.x.to(cfg.out_dtype) * gain.unsqueeze(1).to(cfg.out_dtype)  # [T, H]
    return out
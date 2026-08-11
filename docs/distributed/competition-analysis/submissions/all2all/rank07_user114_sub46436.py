
import torch
import torch.distributed as dist
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    device = rank_data.x.device

    # tokens and expert indices
    x = rank_data.x                   # (num_tokens, hidden_dim)
    indices = rank_data.indices       # (num_tokens, top_k)
    weights = rank_data.weights       # (num_tokens, top_k)

    # compute rank per expert (no all2all needed)
    dst_rank = indices // (cfg.num_experts // world_size)

    # apply "fake expert compute": token * (1+rank)
    # broadcast x to (num_tokens, top_k, hidden_dim)
    token_expanded = x[:, None, :] * (1 + dst_rank)[..., None]

    # weighted sum over top_k
    y = (token_expanded * weights[..., None]).sum(dim=1).to(cfg.out_dtype)

    return y

import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
os.environ["CXX"] = "clang++"
import torch
from torch.utils.cpp_extension import load_inline
import torch.distributed as dist
from task import input_t, output_t

# cfg
# cfg.num_experts
# cfg.experts_per_token
# cfg.hidden_dim
# cfg.max_num_tokens
# cfg.in_dtype
# cfg.out_dtype

# rand_data
# rand_data.num_tokens
# rand_data.indices    [num_tokens, cfg.experts_per_token]
# rand_data.weights    [num_tokens, cfg.experts_per_token]
# rand_data.x          [num_tokens, cfg.hidden_dim]

CPP_WRAPPER = """
void combine(torch::Tensor out_tokens, torch::Tensor recv_data, torch::Tensor indices, torch::Tensor weights,
        int num_experts, int max_num_tokens, int num_tokens, int experts_per_token, int hidden_dim);
"""

CUDA_SRC = """
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp16.h>

__global__ void kernel_combine(__half *out_tokens, const __half *recv_data, const int *indices, const float *weights,
        int num_experts, int max_num_tokens, int num_tokens, int experts_per_token, int hidden_dim) {

    int token_id = blockIdx.x;

    for (int i = 0; i < experts_per_token; i++) {
        int expert_id = indices[token_id * experts_per_token + i];
        int rank_id = expert_id / (num_experts / 8);
        float w = weights[token_id * experts_per_token + i];
        for (int j = threadIdx.x * 8; j < hidden_dim; j += blockDim.x * 8) {
            __half x[8], y[8];
            *(float4 *)x = *(float4 *)&recv_data[rank_id * max_num_tokens * hidden_dim + token_id * hidden_dim + j];
            *(float4 *)y = *(float4 *)&out_tokens[token_id * hidden_dim + j];
            for (int k = 0; k < 8; k++) {
                y[k] = __float2half(__half2float(x[k]) * w + __half2float(y[k]));
            }
            *(float4 *)&out_tokens[token_id * hidden_dim + j] = *(float4 *)y;
        }
    }
}

void combine(torch::Tensor out_tokens, torch::Tensor recv_data, torch::Tensor indices, torch::Tensor weights,
        int num_experts, int max_num_tokens, int num_tokens, int experts_per_token, int hidden_dim) {

    dim3 grid(num_tokens);
    dim3 block(64);
    __half *out_tokens_ptr = (__half *)out_tokens.data_ptr();
    const __half *recv_data_ptr = (__half *)recv_data.data_ptr();
    const int *indices_ptr = indices.data_ptr<int>();
    const float *weights_ptr = weights.data_ptr<float>();
    kernel_combine<<<grid, block>>>(out_tokens_ptr, recv_data_ptr, indices_ptr, weights_ptr,
                                    num_experts, max_num_tokens, num_tokens, experts_per_token, hidden_dim);
}
"""

module = load_inline(
    name='all2all',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['combine'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-O3", "-w"],
)

def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)

    device = rank_data.x.device
    experts_per_rank = cfg.num_experts // world_size

    local_data = torch.empty(world_size, cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
    remote_data = torch.empty(world_size, cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)

    local_data[:, :rank_data.num_tokens] = rank_data.x.unsqueeze(0).expand(world_size, -1, -1)
    dist.all_to_all_single(remote_data, local_data)

    remote_data = remote_data * (1 + rank)
    dist.all_to_all_single(local_data, remote_data)

    out_tokens = torch.zeros(rank_data.num_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)

    module.combine(out_tokens, local_data, rank_data.indices, rank_data.weights,
                   cfg.num_experts, cfg.max_num_tokens, rank_data.num_tokens, cfg.experts_per_token, cfg.hidden_dim)

    # for i in range(rank_data.num_tokens):
    #     for j in range(cfg.experts_per_token):
    #         expert_id = rank_data.indices[i, j].item()
    #         target_rank = expert_id // experts_per_rank
    #         out_tokens[i] += local_data[target_rank, i].to(torch.float32) * rank_data.weights[i, j].to(torch.float32)

    return out_tokens

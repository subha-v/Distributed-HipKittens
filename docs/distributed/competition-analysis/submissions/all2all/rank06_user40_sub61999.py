import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t


CPP_WRAPPER = """
void init(
    int device,
    int rank,
    int world_size,
    int max_tokens,
    int experts_per_token,
    int num_local_experts,
    bool is_pow2,
    int shift);

torch::Tensor pack(
    torch::Tensor indices,
    torch::Tensor weights,
    int num_tokens);

torch::Tensor compute(int device);

torch::Tensor get_gather_buffer(int device);

torch::Tensor get_rs_buffer(int device);
"""

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#define CHECK_HIP(call)                                                             \
    do {                                                                            \
        hipError_t _status = (call);                                                \
        if (_status != hipSuccess) {                                                \
            throw std::runtime_error(std::string("HIP error: ") +                   \
                                     hipGetErrorString(_status));                   \
        }                                                                           \
    } while (0)


namespace {

union HalfBits {
    __half h;
    uint16_t u;
};

__device__ inline uint16_t half_from_float(float v) {
    HalfBits bits;
    bits.h = __float2half(v);
    return bits.u;
}

__device__ inline float half_to_float(uint16_t v) {
    HalfBits bits;
    bits.u = v;
    return __half2float(bits.h);
}

__global__ void pack_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    uint16_t* __restrict__ send_buffer,
    int num_tokens,
    int max_tokens,
    int experts_per_token,
    int num_local_experts,
    int is_pow2,
    int shift) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = max_tokens * experts_per_token;
    if (tid >= total) {
        return;
    }

    int token_idx = tid / experts_per_token;
    int slot = tid % experts_per_token;
    uint16_t* row = send_buffer + token_idx * experts_per_token * 2;
    uint16_t* dest_ptr = row + slot;
    uint16_t* weight_ptr = row + experts_per_token + slot;

    if (token_idx < num_tokens) {
        int index_val = indices[token_idx * experts_per_token + slot];
        int dest_rank = is_pow2 ? (index_val >> shift)
                                : (index_val / num_local_experts);
        int16_t dest_short = static_cast<int16_t>(dest_rank);
        *dest_ptr = static_cast<uint16_t>(dest_short);
        float w = weights[token_idx * experts_per_token + slot];
        *weight_ptr = half_from_float(w);
    } else {
        *dest_ptr = 0;
        *weight_ptr = 0;
    }
}

__global__ void compute_weights_kernel(
    const uint16_t* __restrict__ gathered,
    uint16_t* __restrict__ weights_matrix,
    int total_tokens,
    int experts_per_token,
    int rank,
    float scale) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_tokens) {
        return;
    }

    const uint16_t* row = gathered + tid * experts_per_token * 2;
    float accum = 0.f;
    for (int i = 0; i < experts_per_token; ++i) {
        int16_t dest = static_cast<int16_t>(row[i]);
        if (dest == rank) {
            uint16_t weight_bits = row[experts_per_token + i];
            accum += half_to_float(weight_bits);
        }
    }
    accum *= scale;
    weights_matrix[tid] = half_from_float(accum);
}

struct DeviceContext {
    bool initialized = false;
    int device = -1;
    int rank = -1;
    int world_size = 1;
    int max_tokens = 0;
    int experts_per_token = 0;
    int num_local_experts = 0;
    int is_pow2 = 0;
    int shift = 0;
    at::Tensor send_buffer;
    at::Tensor gather_buffer;
    at::Tensor weights_matrix;
    at::Tensor rs_buffer;
};

std::unordered_map<int, DeviceContext> g_contexts;
std::mutex g_mutex;

DeviceContext* fetch_context(int device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_contexts.find(device);
    if (it == g_contexts.end() || !it->second.initialized) {
        return nullptr;
    }
    return &it->second;
}

}  // namespace

void init(
    int device,
    int rank,
    int world_size,
    int max_tokens,
    int experts_per_token,
    int num_local_experts,
    bool is_pow2,
    int shift) {

    std::lock_guard<std::mutex> lock(g_mutex);
    DeviceContext& ctx = g_contexts[device];
    if (ctx.initialized) {
        bool same =
            ctx.world_size == world_size &&
            ctx.max_tokens == max_tokens &&
            ctx.experts_per_token == experts_per_token &&
            ctx.num_local_experts == num_local_experts;
        if (same) {
            return;
        }
        ctx.send_buffer = at::Tensor();
        ctx.gather_buffer = at::Tensor();
        ctx.weights_matrix = at::Tensor();
        ctx.rs_buffer = at::Tensor();
        ctx.initialized = false;
    }

    CHECK_HIP(hipSetDevice(device));

    ctx.device = device;
    ctx.rank = rank;
    ctx.world_size = world_size;
    ctx.max_tokens = max_tokens;
    ctx.experts_per_token = experts_per_token;
    ctx.num_local_experts = num_local_experts;
    ctx.is_pow2 = is_pow2 ? 1 : 0;
    ctx.shift = shift;

    auto options = torch::TensorOptions()
                       .dtype(torch::kHalf)
                       .device(torch::kCUDA, device);
    ctx.send_buffer = torch::zeros({max_tokens, experts_per_token * 2}, options);
    ctx.gather_buffer =
        torch::zeros({max_tokens * world_size, experts_per_token * 2}, options);
    ctx.weights_matrix = torch::zeros({world_size, max_tokens}, options);
    ctx.rs_buffer = torch::zeros({max_tokens}, options);

    ctx.initialized = true;
}

torch::Tensor pack(
    torch::Tensor indices,
    torch::Tensor weights,
    int num_tokens) {
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA tensor");
    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D");
    TORCH_CHECK(weights.dim() == 2, "weights must be 2D");
    TORCH_CHECK(indices.size(0) == weights.size(0), "mismatched token dimension");
    TORCH_CHECK(indices.size(1) == weights.size(1), "mismatched experts per token");

    indices = indices.contiguous();
    weights = weights.contiguous();

    int device = indices.get_device();
    DeviceContext* ctx_ptr = fetch_context(device);
    TORCH_CHECK(ctx_ptr != nullptr, "Device context not initialized");
    DeviceContext& ctx = *ctx_ptr;
    TORCH_CHECK(num_tokens <= ctx.max_tokens, "num_tokens exceeds max_tokens");

    CHECK_HIP(hipSetDevice(device));
    hipStream_t stream = at::hip::getCurrentHIPStream();

    int total_slots = ctx.max_tokens * ctx.experts_per_token;
    int threads = 256;
    int blocks = (total_slots + threads - 1) / threads;

    pack_kernel<<<blocks, threads, 0, stream>>>(
        indices.data_ptr<int32_t>(),
        weights.data_ptr<float>(),
        reinterpret_cast<uint16_t*>(ctx.send_buffer.data_ptr<at::Half>()),
        num_tokens,
        ctx.max_tokens,
        ctx.experts_per_token,
        ctx.num_local_experts,
        ctx.is_pow2,
        ctx.shift);
    CHECK_HIP(hipGetLastError());

    return ctx.send_buffer;
}

torch::Tensor compute(int device) {
    DeviceContext* ctx_ptr = fetch_context(device);
    TORCH_CHECK(ctx_ptr != nullptr, "Device context not initialized");
    DeviceContext& ctx = *ctx_ptr;

    CHECK_HIP(hipSetDevice(device));
    hipStream_t stream = at::hip::getCurrentHIPStream();

    int total_tokens = ctx.max_tokens * ctx.world_size;
    int threads = 256;
    int blocks = (total_tokens + threads - 1) / threads;
    compute_weights_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(ctx.gather_buffer.data_ptr<at::Half>()),
        reinterpret_cast<uint16_t*>(ctx.weights_matrix.data_ptr<at::Half>()),
        total_tokens,
        ctx.experts_per_token,
        ctx.rank,
        1.0f + static_cast<float>(ctx.rank));
    CHECK_HIP(hipGetLastError());

    return ctx.weights_matrix;
}

torch::Tensor get_gather_buffer(int device) {
    DeviceContext* ctx_ptr = fetch_context(device);
    TORCH_CHECK(ctx_ptr != nullptr, "Device context not initialized");
    return ctx_ptr->gather_buffer;
}

torch::Tensor get_rs_buffer(int device) {
    DeviceContext* ctx_ptr = fetch_context(device);
    TORCH_CHECK(ctx_ptr != nullptr, "Device context not initialized");
    return ctx_ptr->rs_buffer;
}
"""


BASE_DIR = Path(__file__).resolve().parent

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx942")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(BASE_DIR / ".torch_extensions"))
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

_module = None
_initialized_configs: Dict[Tuple[int, int, int, int], bool] = {}


def _load_module():
    global _module
    if _module is not None:
        return _module

    _module = load_inline(
        name="all2all_nccl_inline",
        cpp_sources=[CPP_WRAPPER],
        cuda_sources=[HIP_SRC],
        functions=[
            "init",
            "pack",
            "compute",
            "get_gather_buffer",
            "get_rs_buffer",
        ],
        verbose=False,
        extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-O3"],
        extra_ldflags=["-L/opt/rocm/lib", "-lrccl"],
    )
    return _module


def _ensure_initialized(cfg, device_index: int, rank: int, world_size: int) -> None:
    key = (
        device_index,
        cfg.max_num_tokens,
        cfg.experts_per_token,
        world_size,
        cfg.num_experts,
    )
    if _initialized_configs.get(key, False):
        return

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before custom_kernel")

    module = _load_module()
    num_local_experts = cfg.num_experts // world_size
    is_pow2 = (num_local_experts & (num_local_experts - 1)) == 0
    shift = num_local_experts.bit_length() - 1 if is_pow2 else 0

    module.init(
        device_index,
        rank,
        world_size,
        cfg.max_num_tokens,
        cfg.experts_per_token,
        num_local_experts,
        bool(is_pow2),
        shift,
    )
    _initialized_configs[key] = True


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data

    device_index = rank_data.x.device.index
    if device_index is None:
        raise RuntimeError("Expected rank_data.x to be on a CUDA device")
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)

    module = _load_module()
    _ensure_initialized(cfg, device_index, rank, world_size)

    indices = rank_data.indices
    weights = rank_data.weights
    if not indices.is_contiguous():
        indices = indices.contiguous()
    if not weights.is_contiguous():
        weights = weights.contiguous()

    send_buffer = module.pack(indices, weights, rank_data.num_tokens)
    gather_buffer = module.get_gather_buffer(device_index)
    dist.all_gather_into_tensor(gather_buffer, send_buffer)

    weights_matrix = module.compute(device_index)
    rs_buffer = module.get_rs_buffer(device_index)
    dist.reduce_scatter_tensor(rs_buffer, weights_matrix)

    local_weights = rs_buffer.narrow(0, 0, cfg.max_num_tokens)
    sliced_weights = local_weights[: rank_data.num_tokens]

    x_slice = rank_data.x.narrow(0, 0, rank_data.num_tokens)
    weight_factors = sliced_weights.to(x_slice.dtype).view(-1, 1)
    output = x_slice * weight_factors

    return output

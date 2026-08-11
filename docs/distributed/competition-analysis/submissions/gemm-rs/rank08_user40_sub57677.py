import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

CPP_WRAPPER = """
void init(int device);
void matmul(
    torch::Tensor output,
    torch::Tensor left,
    torch::Tensor right,
    c10::optional<torch::Tensor> bias);
"""

CUDA_SRC = r"""
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/util/Optional.h>
#include <unordered_map>
#include <tuple>
#include <array>
#include <string>

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
        } \
    } while (0)

#define CHECK_HIPBLASLT(call) \
    do { \
        hipblasStatus_t status = call; \
        if (status != HIPBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("hipBLASLt error: ") + std::to_string(static_cast<int>(status))); \
        } \
    } while (0)

struct MatrixSizeHash {
    size_t operator()(const std::tuple<int, int, int>& t) const {
        auto h1 = std::hash<int>{}(std::get<0>(t));
        auto h2 = std::hash<int>{}(std::get<1>(t));
        auto h3 = std::hash<int>{}(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct CachedMatmulResources {
    hipblasLtMatrixLayout_t matA;
    hipblasLtMatrixLayout_t matB;
    hipblasLtMatrixLayout_t matD;
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatmulAlgo_t algo;
    size_t workspaceSize;
};

struct DeviceContext {
    hipblasLtHandle_t handle;
    hipblasLtMatmulPreference_t preference;
    void* workspace;
    size_t workspaceSize;
    std::unordered_map<std::tuple<int, int, int>, CachedMatmulResources, MatrixSizeHash> cache;

    DeviceContext() : handle(nullptr), preference(nullptr), workspace(nullptr), workspaceSize(0) {}
};

static std::unordered_map<int, DeviceContext> g_contexts;

static CachedMatmulResources& get_or_create_resources(DeviceContext& ctx, int M, int N, int K);
static void ensure_workspace(DeviceContext& ctx, size_t workspaceSize);

static DeviceContext& get_device_context(int device) {
    auto it = g_contexts.find(device);
    if (it == g_contexts.end()) {
        CHECK_HIP(hipSetDevice(device));
        DeviceContext ctx;
        CHECK_HIPBLASLT(hipblasLtCreate(&ctx.handle));
        CHECK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&ctx.preference));
        uint64_t maxWorkspaceSize = 1024ULL * 1024 * 1024;
        CHECK_HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(
            ctx.preference,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &maxWorkspaceSize,
            sizeof(maxWorkspaceSize)));
        auto inserted = g_contexts.emplace(device, std::move(ctx));
        it = inserted.first;
    }
    return it->second;
}

void init(int device) {
    CHECK_HIP(hipSetDevice(device));
    auto& ctx = get_device_context(device);

    constexpr std::array<std::tuple<int, int, int>, 6> warmup_shapes = {
        std::tuple<int, int, int>{64, 7168, 2304},
        std::tuple<int, int, int>{512, 4096, 1536},
        std::tuple<int, int, int>{2048, 2880, 360},
        std::tuple<int, int, int>{4096, 4096, 4096},
        std::tuple<int, int, int>{8192, 4096, 1792},
        std::tuple<int, int, int>{8192, 8192, 3696},
    };

    for (const auto& shape : warmup_shapes) {
        auto& res = get_or_create_resources(ctx, std::get<0>(shape), std::get<1>(shape), std::get<2>(shape));
        ensure_workspace(ctx, res.workspaceSize);
    }
}

static CachedMatmulResources& get_or_create_resources(DeviceContext& ctx, int M, int N, int K) {
    auto key = std::make_tuple(M, N, K);
    auto it = ctx.cache.find(key);
    if (it == ctx.cache.end()) {
        CachedMatmulResources res{};
        CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&res.matA, HIP_R_16BF, K, M, K));
        CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&res.matB, HIP_R_16BF, K, N, K));
        CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&res.matD, HIP_R_16BF, N, M, N));

        CHECK_HIPBLASLT(hipblasLtMatmulDescCreate(&res.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        hipblasOperation_t transA = HIPBLAS_OP_T;
        hipblasOperation_t transB = HIPBLAS_OP_N;
        CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
            res.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
        CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
            res.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

        int returnedAlgoCount = 0;
        hipblasLtMatmulHeuristicResult_t heuristicResults[1];
        CHECK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(
            ctx.handle,
            res.desc,
            res.matA,
            res.matB,
            res.matD,
            res.matD,
            ctx.preference,
            1,
            heuristicResults,
            &returnedAlgoCount));
        if (returnedAlgoCount == 0) {
            throw std::runtime_error("No suitable hipBLASLt algorithm found");
        }
        res.algo = heuristicResults[0].algo;
        res.workspaceSize = heuristicResults[0].workspaceSize;

        auto inserted = ctx.cache.emplace(key, std::move(res));
        it = inserted.first;
    }
    return it->second;
}

static void ensure_workspace(DeviceContext& ctx, size_t workspaceSize) {
    if (workspaceSize <= ctx.workspaceSize) {
        return;
    }
    if (ctx.workspace) {
        CHECK_HIP(hipFree(ctx.workspace));
        ctx.workspace = nullptr;
        ctx.workspaceSize = 0;
    }
    if (workspaceSize > 0) {
        CHECK_HIP(hipMalloc(&ctx.workspace, workspaceSize));
        ctx.workspaceSize = workspaceSize;
    }
}

void matmul(
    torch::Tensor output,
    torch::Tensor left,
    torch::Tensor right,
    c10::optional<torch::Tensor> bias_opt) {
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(left.is_cuda(), "left must be a CUDA tensor");
    TORCH_CHECK(right.is_cuda(), "right must be a CUDA tensor");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "Expected BF16 output");
    TORCH_CHECK(left.dtype() == torch::kBFloat16, "Expected BF16 inputs");
    TORCH_CHECK(right.dtype() == torch::kBFloat16, "Expected BF16 inputs");
    TORCH_CHECK(left.size(1) == right.size(1), "K dimensions must match");
    TORCH_CHECK(output.size(0) == left.size(0), "Output M mismatch");
    TORCH_CHECK(output.size(1) == right.size(0), "Output N mismatch");

    auto device = left.device().index();
    TORCH_CHECK(device >= 0, "left tensor has invalid device index");
    TORCH_CHECK(device == right.device().index(), "Inputs must be on the same device");
    TORCH_CHECK(device == output.device().index(), "Output must be on the same device");
    CHECK_HIP(hipSetDevice(device));

    int M = left.size(0);
    int N = right.size(0);
    int K = left.size(1);

    auto& ctx = get_device_context(device);
    auto& res = get_or_create_resources(ctx, M, N, K);
    ensure_workspace(ctx, res.workspaceSize);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    hipStream_t stream = at::hip::getCurrentHIPStream();

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    void* bias_ptr = nullptr;
    torch::Tensor bias_tensor;
    if (bias_opt.has_value()) {
        bias_tensor = bias_opt.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.dtype() == torch::kBFloat16, "bias must be BF16");
        TORCH_CHECK(bias_tensor.device().index() == device, "bias must be on the same device");
        TORCH_CHECK(bias_tensor.numel() == N, "bias shape mismatch");
        if (!bias_tensor.is_contiguous()) {
            bias_tensor = bias_tensor.contiguous();
        }
        epilogue = HIPBLASLT_EPILOGUE_BIAS;
        bias_ptr = bias_tensor.data_ptr();
    }

    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        res.desc,
        HIPBLASLT_MATMUL_DESC_EPILOGUE,
        &epilogue,
        sizeof(epilogue)));

    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        res.desc,
        HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
        &bias_ptr,
        sizeof(bias_ptr)));

    CHECK_HIPBLASLT(hipblasLtMatmul(
        ctx.handle,
        res.desc,
        &alpha,
        right.data_ptr(),
        res.matB,
        left.data_ptr(),
        res.matA,
        &beta,
        output.data_ptr(),
        res.matD,
        output.data_ptr(),
        res.matD,
        &res.algo,
        ctx.workspace,
        ctx.workspaceSize,
        stream));
}
"""

BASE_DIR = Path(__file__).resolve().parent

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx942")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(BASE_DIR / ".torch_extensions"))
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

module = None
_initialized_devices: set[int] = set()
_buffer_cache: Dict[Tuple[int, int, int, int, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_buffers(
    device_index: int,
    M: int,
    N: int,
    world_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (device_index, M, N, world_size, dtype)
    buffers = _buffer_cache.get(key)
    if buffers is None:
        with torch.cuda.device(device_index):
            full_output = torch.empty((M, N), dtype=dtype, device=device)
            rs_output = torch.empty((M // world_size, N), dtype=dtype, device=device)
        buffers = (full_output, rs_output)
        _buffer_cache[key] = buffers
    return buffers

def _load_module():
    global module
    if module is not None:
        return module
    
    # Redirect stdout temporarily to avoid issues in multiprocessing
    import sys
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        if sys.stdout is None:
            sys.stdout = open('/dev/null', 'w')
        if sys.stderr is None:
            sys.stderr = open('/dev/null', 'w')
            
        module = load_inline(
            name="gemm_rs",
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[CUDA_SRC],
            functions=["init", "matmul"],
            verbose=False,  # Disable verbose to avoid stdout issues
            extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-O3"],
        )
    finally:
        if old_stdout is not None:
            sys.stdout = old_stdout
        if old_stderr is not None:
            sys.stderr = old_stderr
    
    return module


def _ensure_extension_initialized(device_index: int) -> None:
    if device_index in _initialized_devices:
        return
    mod = _load_module()
    torch.cuda.set_device(device_index)
    mod.init(device_index)
    _initialized_devices.add(device_index)


def custom_kernel(data: input_t) -> output_t:
    input_tensor, weight, bias = data

    if not input_tensor.is_cuda or not weight.is_cuda:
        raise RuntimeError("custom_kernel expects CUDA tensors")

    device_index = input_tensor.device.index or 0
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)
    _ensure_extension_initialized(device_index)

    world_size = torch.distributed.get_world_size()
    if input_tensor.size(0) % world_size != 0:
        raise ValueError("M must be divisible by world_size")

    left = input_tensor if input_tensor.is_contiguous() else input_tensor.contiguous()
    right = weight if weight.is_contiguous() else weight.contiguous()

    bias_tensor = None
    if bias is not None:
        bias_tensor = bias if bias.is_contiguous() else bias.contiguous()

    mod = _load_module()
    full_output, rs_output = _get_buffers(
        device_index,
        left.size(0),
        right.size(0),
        world_size,
        left.dtype,
        left.device,
    )
    mod.matmul(full_output, left, right, bias_tensor)

    torch.distributed.reduce_scatter_tensor(rs_output, full_output)
    return rs_output

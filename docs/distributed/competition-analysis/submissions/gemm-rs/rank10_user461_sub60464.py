from task import input_t, output_t
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline


hip_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

#define BM 128
#define BN 128
#define BK 32
#define TM 8
#define TN 8

__global__ void __launch_bounds__(256, 2)
gemm_kernel_fast(
    const __hip_bfloat16* __restrict__ A,
    const __hip_bfloat16* __restrict__ B,
    const __hip_bfloat16* __restrict__ bias,
    __hip_bfloat16* __restrict__ C,
    const int M, const int N, const int K, const bool has_bias
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int ty = tid >> 4;
    const int tx = tid & 15;
    
    __shared__ __hip_bfloat16 As[BM * BK];
    __shared__ __hip_bfloat16 Bs[BK * BN];
    
    float acc[TM][TN];
    #pragma unroll
    for(int i = 0; i < TM; i++) {
        #pragma unroll
        for(int j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    const int A_row = by * BM;
    const int B_col = bx * BN;
    
    #pragma unroll 1
    for(int ko = 0; ko < K; ko += BK) {
        
        const int A_load_count = (BM * BK) / 256;
        #pragma unroll
        for(int l = 0; l < A_load_count; l++) {
            const int idx = tid + l * 256;
            const int i = idx / BK;
            const int k = idx % BK;
            const int gi = A_row + i;
            const int gk = ko + k;
            As[i * BK + k] = (gi < M && gk < K) ? __ldg(&A[gi * K + gk]) : __hip_bfloat16(0);
        }
        
        const int B_load_count = (BK * BN) / 256;
        #pragma unroll
        for(int l = 0; l < B_load_count; l++) {
            const int idx = tid + l * 256;
            const int k = idx / BN;
            const int j = idx % BN;
            const int gk = ko + k;
            const int gj = B_col + j;
            Bs[k * BN + j] = (gk < K && gj < N) ? __ldg(&B[gj * K + gk]) : __hip_bfloat16(0);
        }
        
        __syncthreads();
        
        #pragma unroll
        for(int k = 0; k < BK; k++) {
            #pragma unroll
            for(int i = 0; i < TM; i++) {
                const float a_reg = __bfloat162float(As[(ty * TM + i) * BK + k]);
                #pragma unroll
                for(int j = 0; j < TN; j++) {
                    const float b_reg = __bfloat162float(Bs[k * BN + tx * TN + j]);
                    acc[i][j] = __fmaf_rn(a_reg, b_reg, acc[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for(int i = 0; i < TM; i++) {
        const int gi = A_row + ty * TM + i;
        if(gi < M) {
            #pragma unroll
            for(int j = 0; j < TN; j++) {
                const int gj = B_col + tx * TN + j;
                if(gj < N) {
                    float val = acc[i][j];
                    if(has_bias) {
                        val += __bfloat162float(__ldg(&bias[gj]));
                    }
                    C[gi * N + gj] = __float2bfloat16(val);
                }
            }
        }
    }
}

torch::Tensor gemm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    bool has_bias
) {
    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);
    
    auto output = torch::empty({M, N}, input.options());
    
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(256);
    
    hipStream_t stream = c10::hip::getCurrentHIPStream();
    
    gemm_kernel_fast<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
        has_bias ? reinterpret_cast<const __hip_bfloat16*>(bias.data_ptr()) : nullptr,
        reinterpret_cast<__hip_bfloat16*>(output.data_ptr()),
        M, N, K, has_bias
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor gemm_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, bool has_bias);
"""

try:
    gemm_module = load_inline(
        name='gemm_final_fastest',
        cpp_sources=cpp_source,
        cuda_sources=hip_source,
        functions=['gemm_forward'],
        extra_cuda_cflags=[
            '-O3',
            '-march=gfx942',
            '--offload-arch=gfx942',
            '-ffast-math',
            '-munsafe-fp-atomics',
            '-mwavefrontsize64',
            '-mllvm', '-amdgpu-early-inline-all=true',
            '-mllvm', '-amdgpu-function-calls=false',
        ],
        with_cuda=True,
        build_directory='./gemm_final_fastest',
        verbose=False,
    )
    HAS_KERNEL = True
except Exception as e:
    print(f"Compilation failed: {e}")
    HAS_KERNEL = False


def custom_kernel(data: input_t) -> output_t:
    input, weight, bias = data
    M, K = input.shape
    N = weight.shape[0]
    world_size = dist.get_world_size()
    
    input = input.contiguous()
    weight = weight.contiguous()
    has_bias = bias is not None
    if has_bias:
        bias = bias.contiguous()
    
    chunk_size = M // world_size
    
    if HAS_KERNEL:
        output = gemm_module.gemm_forward(
            input, weight,
            bias if has_bias else torch.empty(0, dtype=torch.bfloat16, device=input.device),
            has_bias
        )
    else:
        output = torch.nn.functional.linear(input, weight, bias)
    
    rs_output = torch.empty((chunk_size, N), dtype=output.dtype, device=input.device)
    dist.reduce_scatter_tensor(rs_output, output, op=dist.ReduceOp.SUM)
    
    return rs_output

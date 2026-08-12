// exp_22 toolchain capability probe (CPU-only compile test, throwaway).
// Each variant isolates one API whose exact spelling on ROCm 7.2 / gfx950 the
// ubench depends on. A variant that fails to compile prints the expected
// signature in its error, which is the information we want.
#include <hip/hip_runtime.h>
#include <cstdint>

using v16f32 = float __attribute__((ext_vector_type(16)));
using v4f32  = float __attribute__((ext_vector_type(4)));
using v8bf16 = __bf16 __attribute__((ext_vector_type(8)));
using v8s16  = short __attribute__((ext_vector_type(8)));
using v4s16  = short __attribute__((ext_vector_type(4)));

#if defined(PROBE_VARIANT_A)
// CDNA4 doubled-K bf16 MFMA, __bf16 vector operands.
__global__ void k(v16f32* o, const v8bf16* a, const v8bf16* b) {
    v16f32 acc = o[0];
    acc = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a[0], b[0], acc, 0, 0, 0);
    o[0] = acc;
}
#elif defined(PROBE_VARIANT_B)
// same instruction, short-vector operands (CDNA3 _1k spelling style)
__global__ void k(v16f32* o, const v8s16* a, const v8s16* b) {
    v16f32 acc = o[0];
    acc = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a[0], b[0], acc, 0, 0, 0);
    o[0] = acc;
}
#elif defined(PROBE_VARIANT_C)
// CDNA3-era fallback that gfx950 also implements.
__global__ void k(v16f32* o, const v4s16* a, const v4s16* b) {
    v16f32 acc = o[0];
    acc = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a[0], b[0], acc, 0, 0, 0);
    o[0] = acc;
}
#elif defined(PROBE_VARIANT_D)
// 16x16x32 bf16 (CDNA4), 4-wide accumulator.
__global__ void k(v4f32* o, const v8bf16* a, const v8bf16* b) {
    v4f32 acc = o[0];
    acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a[0], b[0], acc, 0, 0, 0);
    o[0] = acc;
}
#elif defined(PROBE_VARIANT_E)
// uncached device allocation (mori symmetric-heap HeapType parity) + big LDS +
// s_memrealtime + peer-capable atomics, all in one.
__device__ __forceinline__ std::uint64_t rt_now() {
    std::uint64_t t;
    asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)" : "=s"(t) :: "memory");
    return t;
}
__global__ __launch_bounds__(256) void k(std::uint64_t* o, unsigned* c) {
    __shared__ char big[98304];
    big[threadIdx.x] = (char)threadIdx.x;
    __syncthreads();
    o[0] = rt_now() + (std::uint64_t)big[255];
    (void)__hip_atomic_fetch_add(c, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
}
int host_probe() {
    void* p = nullptr;
    if (hipExtMallocWithFlags(&p, 1024, hipDeviceMallocUncached) != hipSuccess) return 1;
    int n = 0; (void)hipGetDeviceCount(&n);
    int can = 0; (void)hipDeviceCanAccessPeer(&can, 0, 1);
    return n + can;
}
#endif

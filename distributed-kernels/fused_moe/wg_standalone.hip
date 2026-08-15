// Standalone harness for the T2B wgrad phase (n2_wgrad_gm_t2b.cpp): one
// kernel, descriptor-style arg table so k0_train.hip_loader's 3-arg launch
// works unchanged.  Validates the MFMA lane layout + scale folding against
// a torch reference before the mega integration (wg_test.py).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <cstdint>

#include "n2_wgrad_gm_t2b.cpp"

extern "C" __global__ __launch_bounds__(256, 1) void wg_standalone(
    const long long* desc, int* pperr, long long spin) {
  (void)pperr;
  (void)spin;
  __shared__ __hip_bfloat16
      ldsA[production_fused_moe::n2::kWgMT * production_fused_moe::n2::kWgLdsRow];
  __shared__ __hip_bfloat16
      ldsB[production_fused_moe::n2::kWgMT * production_fused_moe::n2::kWgLdsRow];
  __shared__ float ldsMeta[4 * 32];
  __shared__ int ldsRange[64];
  production_fused_moe::n2::n2p6t2b_wgrad_body(
      (const std::uint8_t*)desc[0], (const float*)desc[1],
      (const std::uint8_t*)desc[2], (const float*)desc[3],
      (const std::uint8_t*)desc[4], (const float*)desc[5],
      (const std::uint8_t*)desc[6], (const float*)desc[7],
      (const int*)desc[8], (const float*)desc[9], (const int*)desc[10],
      (const int*)desc[11], (__hip_bfloat16*)desc[12],
      (__hip_bfloat16*)desc[13], (int)desc[14], (int)blockIdx.x,
      (int)gridDim.x, ldsA, ldsB, ldsMeta, ldsRange);
}

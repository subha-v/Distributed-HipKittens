// T2B PHASE 1z — z regeneration: z = W13 · x_rows, quantized to Zq/DQZ.
//
// WHY THIS PHASE EXISTS (measured, 2026-08-15): saving z from the FORWARD's
// epilogue costs 784 B/lane of scratch — the forward epilogue's register
// ceiling cannot hold acc addressable for a second consumer (13 ms
// pathology, phase-map risk 9; reordering does not help).  Regenerating z
// in the BACKWARD costs one GEMM1-shaped pass (~2.7 ms class) in a phase
// where the accumulators have exactly ONE consumer — the plain forward
// stays untouched at its 4.75 ms, and the forward saves NOTHING beyond its
// natural outputs (the plan + a_dst/sc_stage x-rows).
//
// Chassis: the forward phase-1 VERBATIM (paired gate/up GEMM over W13
// [32, 4096, 7168], K = 7168) with the epilogue replaced: no SiLU, no A2q —
// the raw (g, u) accumulators are quantized per-(row, K128) into
// Zq [rowcap, 4096] (g-half cols [0,2048), u-half [2048,4096)) + DQZ
// [rowcap, 32], phase-1b's exact read layout.  Inputs come from the
// FORWARD-SAVED buffers: x rows (a_dst) + token-major scales (sc_stage),
// addressed through the SAVED plan (sorted_ids/tile_desc/nvi).
// ===========================================================================

#include "kittens.cuh"
#include "n2_fused_moe.hpp"
#include "aiter_gfx950_fp8_layout.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <cstddef>
#include <cstdint>

#include "n2_device_common.cuh"

namespace production_fused_moe::n2 {

constexpr int kW13NZ = 4096;
constexpr int kNGroups13Z = kW13NZ / 128;          // 32
constexpr int kChunksZ = 8;
constexpr int kChunkColsZ = kInter / kChunksZ;     // 256
constexpr int kWaveColsZ = kChunkColsZ / kWaves;   // 64
constexpr int kColTilesZ = kWaveColsZ / 16;        // 4
constexpr int kKGroupsZ = kHidden / 128;           // 56
constexpr int kDZColsZ = 2 * kInter;               // 4096
constexpr int kKGroupsDZZ = kDZColsZ / 128;        // 32
constexpr int kA2StrideZ = 272;

static_assert(kKGroupsZ % 2 == 0, "the K loop is unrolled by two");
static_assert(kThreads == kBlockM * kAChunks, "A tile fill contract");

#ifndef N2_P1Z_QUAL
#define N2_P1Z_QUAL __global__ __launch_bounds__(kThreads, 1)
#define N2_P1Z_NAME n2_phase1z_kernel
#endif
#ifndef N2_HOOK_CTX_ARG
#define N2_HOOK_CTX_ARG
#endif
#ifndef N2GM_P1Z_TASK_START
#define N2GM_P1Z_TASK_START blockIdx.x
#endif
#ifndef N2GM_P1Z_TASK_STRIDE
#define N2GM_P1Z_TASK_STRIDE kCTAs
#endif

N2_P1Z_QUAL void N2_P1Z_NAME(
    const std::uint8_t* __restrict__ X_bytes,    // fwd-saved x rows [R, 7168]
    const float* __restrict__ X_scale,           // fwd-saved token-major [R,56]
    const std::uint8_t* __restrict__ W13,        // ORIGINAL [32, 4096, 7168]
    const float* __restrict__ S13,               // ORIGINAL [32, 32, 56]
    const int* __restrict__ sorted_ids,          // SAVED plan
    const int* __restrict__ sorted_eid,          // SAVED plan
    const int* __restrict__ nvi,                 // SAVED plan
    std::uint8_t* __restrict__ Zq,               // OUT [rowcap, 4096] fp8
    float* __restrict__ DQZ,                     // OUT [rowcap, 32] f32
    const int* __restrict__ tile_desc,           // SAVED plan
    int num_tiles
    N2_HOOK_CTX_ARG) {
#ifndef N2GM_G
#define N2GM_G 1
#endif
  constexpr int kGM = N2GM_G;
  constexpr int kMrows = kBlockM * kGM;

  // shared arrays: the t2b namespace-scope overlay (k0pf6gm_t2b_shared.hpp)

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 63;
  const int wv = tid >> 6;
  const int q = lane >> 4;
  const int r = lane & 15;
  const int T = nvi[1];

  constexpr int kNChunksP1Z = kChunksZ;
  const int num_tasks = num_tiles * kNChunksP1Z;
  auto tile_b0 = [&](int t) __attribute__((always_inline)) {
    return tile_desc[t] >> 4;
  };
  auto tile_gcount = [&](int t) __attribute__((always_inline)) {
    return tile_desc[t] & 0xF;
  };

  const std::uint32_t a_num_records =
      static_cast<std::uint32_t>(T) * static_cast<std::uint32_t>(kHidden);
  const i32x4 a_rsrc = make_srsrc(X_bytes, a_num_records, 0);

  const int a_row = tid >> 3;
  const int a_chunk = tid & 7;
  const int a_cp = a_chunk ^ (a_row & 7);
  const int j_w = wv >> 1;

  for (int task = (int)(N2GM_P1Z_TASK_START); task < num_tasks;
       task += (int)(N2GM_P1Z_TASK_STRIDE)) {
    const int tile = task / kNChunksP1Z;
    const int g = task - tile * kNChunksP1Z;
    const int b0 = tile_b0(tile);
    const int e = sorted_eid[b0];
    const int gcount = tile_gcount(tile);
    const int ew = e;

    for (int i = tid; i < kMrows; i += kThreads) {
      const int lb32 = i >> 5;
      t2bsh_tok_lds[i] = (lb32 < gcount)
                       ? (sorted_ids[(b0 + lb32) * kBlockM + (i & 31)] &
                          0x00FFFFFF)
                       : T;
    }
    for (int i = tid; i < 2 * kMrows; i += kThreads) {
      t2bsh_amax_lds[i / kMrows][i % kMrows] = 0.0f;
    }
    if (tid < 2 * 2 * kKGroupsZ) {
      const int k = tid % kKGroupsZ;
      const int j = (tid / kKGroupsZ) & 1;
      const int gu = tid / (2 * kKGroupsZ);
      const int ng = (gu == 0) ? (2 * g + j) : (kInter / 128 + 2 * g + j);
      t2bsh_b1s_lds[gu][j][k] = S13[(ew * kNGroups13Z + ng) * kKGroupsZ + k];
    }
    __syncthreads();

    static_assert(kKGroupsZ % 4 == 0, "float4 scale quads");
    constexpr int kScaleQuads = kKGroupsZ / 4;
    for (int idx = tid; idx < kScaleQuads * kMrows; idx += kThreads) {
      const int i = idx % kMrows;
      const int c = idx / kMrows;
      const int token = t2bsh_tok_lds[i];
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (token < T) {
        v = *reinterpret_cast<const float4*>(
            X_scale + static_cast<std::size_t>(token) * kKGroupsZ + 4 * c);
      }
      t2bsh_ascale_lds[4 * c + 0][i] = v.x;
      t2bsh_ascale_lds[4 * c + 1][i] = v.y;
      t2bsh_ascale_lds[4 * c + 2][i] = v.z;
      t2bsh_ascale_lds[4 * c + 3][i] = v.w;
    }
    __syncthreads();

    std::uint32_t a_voff0[kGM];
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
      a_voff0[sb] =
          static_cast<std::uint32_t>(t2bsh_tok_lds[sb * kBlockM + a_row]) *
              static_cast<std::uint32_t>(kHidden) +
          static_cast<std::uint32_t>(a_chunk * kAChunkBytes);
    }

    racc acc[kGM][2][2][kColTilesZ];
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
#pragma unroll
      for (int gu = 0; gu < 2; ++gu) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
          for (int c = 0; c < kColTilesZ; ++c) {
            zero(acc[sb][gu][m][c]);
          }
        }
      }
    }

    const int n_gate = kChunkColsZ * g + kWaveColsZ * wv;
    const int n_up = kInter + n_gate;

    rfp8 af[kGM][2];
    __uint128_t aTmp[kGM][2];
    rfp8 bf[2][2][kColTilesZ];

    auto load_a = [&](int k, int buf) __attribute__((always_inline)) {
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
        aTmp[sb][buf] = llvm_amdgcn_raw_buffer_load_b128(
            a_rsrc, a_voff0[sb] + static_cast<std::uint32_t>(k * 128), 0u, 0u);
      }
    };
    auto store_a = [&](int buf, int lb) __attribute__((always_inline)) {
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
        *reinterpret_cast<__uint128_t*>(
            &t2bsh_a_lds[lb][sb * kBlockM + a_row][a_cp][0]) = aTmp[sb][buf];
      }
    };
    auto read_a = [&](int lb) __attribute__((always_inline)) {
      const int xr = r & 7;
      const int cp0 = q ^ xr;
      const int cp1 = cp0 ^ 4;
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const int row = sb * kBlockM + 16 * m + r;
          float4* d = reinterpret_cast<float4*>(&af[sb][m].tiles[0][0].data[0]);
          d[0] = *reinterpret_cast<const float4*>(&t2bsh_a_lds[lb][row][cp0][0]);
          d[1] = *reinterpret_cast<const float4*>(&t2bsh_a_lds[lb][row][cp1][0]);
        }
      }
    };
    auto load_b = [&](int k, int buf) __attribute__((always_inline)) {
      const int kbyte = k * 128;
#pragma unroll
      for (int c = 0; c < kColTilesZ; ++c) {
        const std::size_t tg = aiter::fp8_weight_byte_offset(
            ew, n_gate + 16 * c, kbyte, kW13NZ, kHidden);
        const std::size_t tu = aiter::fp8_weight_byte_offset(
            ew, n_up + 16 * c, kbyte, kW13NZ, kHidden);
        load_bfrag(bf[buf][0][c], W13 + tg + 16 * lane);
        load_bfrag(bf[buf][1][c], W13 + tu + 16 * lane);
        stage_agpr(bf[buf][0][c]);
        stage_agpr(bf[buf][1][c]);
      }
    };
    auto mfma_k = [&](int k, int buf) __attribute__((always_inline)) {
      const float bsg = t2bsh_b1s_lds[0][j_w][k];
      const float bsu = t2bsh_b1s_lds[1][j_w][k];
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
        const float4 as0 =
            *reinterpret_cast<const float4*>(&t2bsh_ascale_lds[k][sb * kBlockM + 4 * q]);
        const float4 as1 = *reinterpret_cast<const float4*>(
            &t2bsh_ascale_lds[k][sb * kBlockM + 16 + 4 * q]);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const float4 as = (m == 0) ? as0 : as1;
#pragma unroll
          for (int gu = 0; gu < 2; ++gu) {
            const float bs = (gu == 0) ? bsg : bsu;
            const f32x2 bsv = {bs, bs};
            const f32x2 s0 = f32x2{as.x, as.y} * bsv;
            const f32x2 s1 = f32x2{as.z, as.w} * bsv;
#pragma unroll
            for (int c = 0; c < kColTilesZ; ++c) {
              racc p;
              zero(p);
              mma_ABt(p, af[sb][m], bf[buf][gu][c], p);
              const f32x2* pf = accv(p);
              f32x2* cf = accv(acc[sb][gu][m][c]);
              cf[0] += pf[0] * s0;
              cf[1] += pf[1] * s1;
            }
          }
        }
      }
    };

    load_a(0, 0);
    load_a(1, 1);
    load_b(0, 0);
    store_a(0, 0);
    lds_cta_barrier();

#pragma unroll 1
    for (int k = 0; k < kKGroupsZ; k += 2) {
      const int ka2 = (k + 2 < kKGroupsZ) ? (k + 2) : (kKGroupsZ - 1);
      const int ka3 = (k + 3 < kKGroupsZ) ? (k + 3) : (kKGroupsZ - 1);
      const int kb1 = (k + 1 < kKGroupsZ) ? (k + 1) : (kKGroupsZ - 1);
      const int kb2 = (k + 2 < kKGroupsZ) ? (k + 2) : (kKGroupsZ - 1);

      read_a(0);
      load_a(ka2, 0);
      load_b(kb1, 1);
      mfma_k(k, 0);
      store_a(1, 1);
      __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
      __builtin_amdgcn_sched_group_barrier(0x020, 17, 0);
      __builtin_amdgcn_sched_group_barrier(0x008, 16, 0);
      __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);
      lds_cta_barrier();

      read_a(1);
      load_a(ka3, 1);
      load_b(kb2, 0);
      mfma_k(kb1, 1);
      store_a(0, 0);
      __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
      __builtin_amdgcn_sched_group_barrier(0x020, 17, 0);
      __builtin_amdgcn_sched_group_barrier(0x008, 16, 0);
      __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);
      lds_cta_barrier();
    }

    // =======================================================================
    // EPILOGUE: quantize (g, u) per (row, K128) — acc's ONLY consumer, so
    // the register ceiling holds (the in-forward save's 784 B scratch was
    // the dual-consumer pathology this phase exists to avoid).
    // =======================================================================
#pragma unroll
    for (int gu = 0; gu < 2; ++gu) {
      if (gu == 1) {
        for (int i = tid; i < 2 * kMrows; i += kThreads) {
          t2bsh_amax_lds[i / kMrows][i % kMrows] = 0.0f;
        }
        __syncthreads();
      }
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
        const int rbase = sb * kBlockM;
        float lm[2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
          for (int t = 0; t < 4; ++t) lm[m][t] = 0.0f;
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
          for (int c = 0; c < kColTilesZ; ++c) {
            const float* zf = accf(acc[sb][gu][m][c]);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
              const bool lv = t2bsh_tok_lds[rbase + 16 * m + 4 * q + t] < T;
              lm[m][t] = fmaxf(lm[m][t], lv ? fabsf(zf[t]) : 0.0f);
            }
          }
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
          for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int mask = 1; mask < 16; mask <<= 1) {
              lm[m][t] = fmaxf(lm[m][t], __shfl_xor(lm[m][t], mask, 64));
            }
          }
        }
        if (r == 0) {
#pragma unroll
          for (int m = 0; m < 2; ++m) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
              atomicMax(reinterpret_cast<int*>(
                            &t2bsh_amax_lds[j_w][rbase + 16 * m + 4 * q + t]),
                        __float_as_int(lm[m][t]));
            }
          }
        }
      }
      __syncthreads();
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
        const int rbase = sb * kBlockM;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          float dqz[4];
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            dqz[t] = fmaxf(t2bsh_amax_lds[j_w][rbase + 16 * m + 4 * q + t],
                           1.0e-6f) /
                     448.0f;
          }
#pragma unroll
          for (int c = 0; c < kColTilesZ; ++c) {
            const float* zf = accf(acc[sb][gu][m][c]);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
              const bool lv = t2bsh_tok_lds[rbase + 16 * m + 4 * q + t] < T;
              const float zv = lv ? zf[t] : 0.0f;
              const float qv = fminf(fmaxf(zv / dqz[t], -448.0f), 448.0f);
              t2bsh_a2_lds[rbase + 16 * m + 4 * q + t]
                    [kWaveColsZ * wv + 16 * c + r] =
                  __hip_cvt_float_to_fp8(qv, __HIP_SATFINITE, __HIP_E4M3);
            }
          }
        }
      }
      __syncthreads();
      {
        const int cr = tid >> 3;
        const int cc = tid & 7;
#pragma unroll
        for (int sb = 0; sb < kGM; ++sb) {
          if (sb >= gcount) break;
          const uint4 s0 = *reinterpret_cast<const uint4*>(
              &t2bsh_a2_lds[sb * kBlockM + cr][32 * cc]);
          const uint4 s1 = *reinterpret_cast<const uint4*>(
              &t2bsh_a2_lds[sb * kBlockM + cr][32 * cc + 16]);
          const std::size_t dst =
              (static_cast<std::size_t>(b0 + sb) * kBlockM + cr) * kDZColsZ +
              static_cast<std::size_t>(gu) * kInter + kChunkColsZ * g +
              32 * cc;
          *reinterpret_cast<uint4*>(Zq + dst) = s0;
          *reinterpret_cast<uint4*>(Zq + dst + 16) = s1;
        }
        if (q == 0) {
#pragma unroll
          for (int sb = 0; sb < kGM; ++sb) {
            if (sb >= gcount) break;
            const int rbase = sb * kBlockM;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
#pragma unroll
              for (int kb = 0; kb < 2; ++kb) {
                DQZ[(static_cast<std::size_t>(b0 + sb) * kBlockM + 16 * m +
                     r) *
                        kKGroupsDZZ +
                    gu * (kKGroupsDZZ / 2) + 2 * g + kb] =
                    fmaxf(t2bsh_amax_lds[kb][rbase + 16 * m + r], 1.0e-6f) / 448.0f;
              }
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

}  // namespace production_fused_moe::n2

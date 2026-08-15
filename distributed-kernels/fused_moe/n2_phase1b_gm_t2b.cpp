// T2B PHASE 1b — dH2 = W2^T · dY, the backward's first grouped GEMM, with the
// swiglu-prime epilogue emitting dZq (both halves) and the dw row-dots.
// Derived from n2_phase1_gm_mps.cpp (the forward's GEMM-1); design:
// overnight/aug14/T2_BACKWARD_MEGA_DESIGN.md, blueprint T2_PHASE_MAP.md §3.
//
// WHAT IS IDENTICAL to the forward phase 1 (byte-level intent, not aspiration):
//   * the A side entirely: dYq rows are [R, 7168] fp8 + 56 token-major
//     per-K128 scales — the forward's exact input shape, so the A tile fill,
//     the XOR swizzle, the double-buffered pipeline, and the bounded-srsrc
//     discipline port unchanged;
//   * the task decomposition: task = (expert-aligned tile, chunk g in [0,8)),
//     kGM G-stacking, tile_desc/num_tiles from the FORWARD'S SAVED PLAN;
//     8 chunk-tasks per 32-block keeps the a2_done >= 8 gate exactly
//     (phase map risk 4);
//   * the K-loop shape: K = 7168 = 56 K128 groups, unroll-2, 17 loads in
//     flight, the donor sched hints at mask 0.
//
// WHAT CHANGES:
//   * ONE weight set instead of the gate/up pair: W2T [32, 2048, 7168] fp8
//     AITER-shuffled (host-pre-transposed per step) + S2T [32, 16, 56] f32.
//     The gu dimension is DELETED — acc halves to [kGM][2][kColTiles],
//     freeing the registers the new epilogue spends on z loads.
//   * the epilogue: dH2 (in acc) meets the forward-saved z (Zq/DQZ) and
//     act(z) (A2q/DQ2):
//         sig    = 1/(1+e^-g)
//         dZ_g   = dH2 * u * sig * (1 + g*(1-sig))
//         dZ_u   = dH2 * g * sig
//         dw_row += dH2 * act(z)          (row-dot partial, sorted-row space)
//     dZ is quantized per-(row, K128) into dZq [rowcap, 4096] + DQZ
//     [rowcap, 32], g-half at cols [0,2048), u-half at [2048,4096) — the
//     W13-column order phase-2b consumes.  The staging LDS is REUSED
//     SEQUENTIALLY (g-half stored, then u-half) so the LDS budget is the
//     forward's.
//   * dw output is SORTED-ROW space (dw_sorted [rowcap] f32, one value per
//     (token,expert) sorted row): the M1 dedup makes receive rows
//     multi-expert aggregates, so (token,k) attribution needs the saved
//     plan — the HOST wrapper scatters dw_sorted -> [T,8] via sti/sei.
//     Accumulation here is one atomicAdd per (lane,m,t) partial.
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

// fp8 e4m3 byte -> f32 (the epilogue's z/act(z) dequant loads).
__device__ __forceinline__ float k0p6t2b_fp8_to_f32(std::uint8_t b) {
  const __half_raw h = __hip_cvt_fp8_to_halfraw(b, __HIP_E4M3);
  return __half2float(__half(h));
}

// ---- backward geometry -----------------------------------------------------
// dH2 = W2T · dY:  N = 2048 (kInter), K = 7168 (kHidden).
constexpr int kW2TN = 2048;                        // W2T rows (output cols)
constexpr int kNGroupsW2T = kW2TN / 128;           // 16 blockscale N-groups
constexpr int kDZCols = 2 * kInter;                // 4096: dgate|dup halves
constexpr int kKGroupsDZ = kDZCols / 128;          // 32 DQZ scale groups

constexpr int kChunksB = 8;                        // 2048 / 256
constexpr int kChunkColsB = kW2TN / kChunksB;      // 256
constexpr int kWaveColsB = kChunkColsB / kWaves;   // 64
constexpr int kColTilesB = kWaveColsB / 16;        // 4

constexpr int kKGroupsB = kHidden / 128;           // 56 (the A/K side)

constexpr int kA2StrideB = 272;                    // staging LDS row stride

static_assert(kKGroupsB % 2 == 0, "the K loop is unrolled by two");
static_assert(kChunkColsB % (16 * kWaves) == 0, "W2T chunk / wave split");
static_assert(kThreads == kBlockM * kAChunks,
              "the A tile fill must be exactly one 16 B load per thread");
static_assert(kAChunks * kAChunkBytes == 128, "one K128 group per A tile row");

#ifndef N2GM_P1B_TASK_START
#define N2GM_P1B_TASK_START blockIdx.x
#endif
#ifndef N2GM_P1B_TASK_STRIDE
#define N2GM_P1B_TASK_STRIDE kCTAs
#endif
#ifndef N2GM_P1B_TASK_DONE_HOOK
#define N2GM_P1B_TASK_DONE_HOOK
#endif

#ifndef N2_P1B_QUAL
#define N2_P1B_QUAL __global__ __launch_bounds__(kThreads, 1)
#define N2_P1B_NAME n2_phase1b_kernel
#endif

N2_P1B_QUAL void N2_P1B_NAME(
    const std::uint8_t* __restrict__ dY_bytes,   // dYq rows [R, 7168] fp8
    const float* __restrict__ dY_scale,          // token-major [R, 56] f32
    const std::uint8_t* __restrict__ W2T,        // [32, 2048, 7168] AITER fp8
    const float* __restrict__ S2T,               // [32, 16, 56] f32
    const std::uint8_t* __restrict__ Zq,         // fwd-saved [rowcap, 4096] fp8
    const float* __restrict__ DQZ,               // fwd-saved [rowcap, 32] f32
    const std::uint8_t* __restrict__ A2q,        // fwd-saved act(z) [rowcap,2048]
    const float* __restrict__ DQ2,               // fwd-saved [rowcap, 16] f32
    const int* __restrict__ sorted_ids,          // SAVED plan: token|slot<<24
    const int* __restrict__ sorted_eid,          // SAVED plan
    const int* __restrict__ nvi,                 // SAVED plan: [padded_rows, T]
    std::uint8_t* __restrict__ dZq,              // OUT [rowcap, 4096] fp8
    float* __restrict__ DQdZ,                    // OUT [rowcap, 32] f32
    float* __restrict__ dw_sorted,               // OUT [rowcap] f32 (pre-zeroed)
    const int* __restrict__ tile_desc,           // SAVED plan
    int num_tiles) {
#ifndef N2GM_G
#define N2GM_G 1
#endif
  constexpr int kGM = N2GM_G;
  constexpr int kMrows = kBlockM * kGM;

  __shared__ int tok_lds[kMrows];
  __shared__ int srow_lds[kMrows];                 // absolute sorted row ids
  __shared__ float ascale_lds[kKGroupsB][kMrows];
  __shared__ float b1s_lds[2][kKGroupsB];          // [n128 half][k128] (no gu)
  __shared__ float amax_lds[2][kMrows];
  __shared__ __align__(16) std::uint8_t a2_lds[kMrows][kA2StrideB];
  __shared__ __align__(16)
      std::uint8_t a_lds[2][kMrows][kAChunks][kAChunkBytes];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 63;
  const int wv = tid >> 6;
  const int q = lane >> 4;
  const int r = lane & 15;
  const int T = nvi[1];

  constexpr int kNChunksP1B = kChunksB;
  const int num_tasks = num_tiles * kNChunksP1B;
  auto tile_b0 = [&](int t) __attribute__((always_inline)) {
    return tile_desc[t] >> 4;
  };
  auto tile_gcount = [&](int t) __attribute__((always_inline)) {
    return tile_desc[t] & 0xF;
  };

  const std::uint32_t a_num_records =
      static_cast<std::uint32_t>(T) * static_cast<std::uint32_t>(kHidden);
  const i32x4 a_rsrc = make_srsrc(dY_bytes, a_num_records, 0);

  const int a_row = tid >> 3;
  const int a_chunk = tid & 7;
  const int a_cp = a_chunk ^ (a_row & 7);

  const int j_w = wv >> 1;

  for (int task = (int)(N2GM_P1B_TASK_START); task < num_tasks;
       task += (int)(N2GM_P1B_TASK_STRIDE)) {
    const int tile = task / kNChunksP1B;
    const int g = task - tile * kNChunksP1B;   // chunk of the 2048 cols
    const int b0 = tile_b0(tile);
    const int e = sorted_eid[b0];
    const int gcount = tile_gcount(tile);

    const std::uint8_t* const W2Te = W2T;
    const float* const S2Te = S2T;
    const int ew = e;

    for (int i = tid; i < kMrows; i += kThreads) {
      const int lb32 = i >> 5;
      const int srow = (b0 + lb32) * kBlockM + (i & 31);
      srow_lds[i] = srow;
      tok_lds[i] = (lb32 < gcount)
                       ? (sorted_ids[srow] & 0x00FFFFFF)
                       : T;
    }
    for (int i = tid; i < 2 * kMrows; i += kThreads) {
      amax_lds[i / kMrows][i % kMrows] = 0.0f;
    }
    // One weight-scale set: S2T[e][n128][k128] for the two n128 halves this
    // chunk's 256 columns span (2g, 2g+1).
    if (tid < 2 * kKGroupsB) {
      const int k = tid % kKGroupsB;
      const int j = tid / kKGroupsB;   // n128 half 0/1
      b1s_lds[j][k] = S2Te[(ew * kNGroupsW2T + 2 * g + j) * kKGroupsB + k];
    }
    __syncthreads();

    // dY scales are TOKEN-MAJOR [R, 56] (the backward M2 writes sc_stage in
    // the m15 kernel's K0P6_MPS_ASCALE_TM=1 layout): the forward's token-major
    // gather arm, verbatim.
    static_assert(kKGroupsB % 4 == 0, "float4 scale quads");
    constexpr int kScaleQuads = kKGroupsB / 4;
    for (int idx = tid; idx < kScaleQuads * kMrows; idx += kThreads) {
      const int i = idx % kMrows;
      const int c = idx / kMrows;
      const int token = tok_lds[i];
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (token < T) {
        v = *reinterpret_cast<const float4*>(
            dY_scale + static_cast<std::size_t>(token) * kKGroupsB + 4 * c);
      }
      ascale_lds[4 * c + 0][i] = v.x;
      ascale_lds[4 * c + 1][i] = v.y;
      ascale_lds[4 * c + 2][i] = v.z;
      ascale_lds[4 * c + 3][i] = v.w;
    }
    __syncthreads();

    std::uint32_t a_voff0[kGM];
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
      a_voff0[sb] =
          static_cast<std::uint32_t>(tok_lds[sb * kBlockM + a_row]) *
              static_cast<std::uint32_t>(kHidden) +
          static_cast<std::uint32_t>(a_chunk * kAChunkBytes);
    }

    // =======================================================================
    // GEMM: dH2 chunk = dYq · W2T-chunk.  The forward pipeline minus gu.
    // =======================================================================
    racc acc[kGM][2][kColTilesB];
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int c = 0; c < kColTilesB; ++c) {
          zero(acc[sb][m][c]);
        }
      }
    }

    const int n_col = kChunkColsB * g + kWaveColsB * wv;

    rfp8 af[kGM][2];
    __uint128_t aTmp[kGM][2];
    rfp8 bf[2][kColTilesB];

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
            &a_lds[lb][sb * kBlockM + a_row][a_cp][0]) = aTmp[sb][buf];
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
          d[0] = *reinterpret_cast<const float4*>(&a_lds[lb][row][cp0][0]);
          d[1] = *reinterpret_cast<const float4*>(&a_lds[lb][row][cp1][0]);
        }
      }
    };
    auto load_b = [&](int k, int buf) __attribute__((always_inline)) {
      const int kbyte = k * 128;
#pragma unroll
      for (int c = 0; c < kColTilesB; ++c) {
        const std::size_t t0 = aiter::fp8_weight_byte_offset(
            ew, n_col + 16 * c, kbyte, kW2TN, kHidden);
        load_bfrag(bf[buf][c], W2Te + t0 + 16 * lane);
        stage_agpr(bf[buf][c]);
      }
    };

    auto mfma_k = [&](int k, int buf) __attribute__((always_inline)) {
      const float bs = b1s_lds[j_w][k];
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
        const float4 as0 =
            *reinterpret_cast<const float4*>(&ascale_lds[k][sb * kBlockM + 4 * q]);
        const float4 as1 = *reinterpret_cast<const float4*>(
            &ascale_lds[k][sb * kBlockM + 16 + 4 * q]);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const float4 as = (m == 0) ? as0 : as1;
          const f32x2 bsv = {bs, bs};
          const f32x2 s0 = f32x2{as.x, as.y} * bsv;
          const f32x2 s1 = f32x2{as.z, as.w} * bsv;
#pragma unroll
          for (int c = 0; c < kColTilesB; ++c) {
            racc p;
            zero(p);
            mma_ABt(p, af[sb][m], bf[buf][c], p);
            const f32x2* pf = accv(p);
            f32x2* cf = accv(acc[sb][m][c]);
            cf[0] += pf[0] * s0;
            cf[1] += pf[1] * s1;
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
    for (int k = 0; k < kKGroupsB; k += 2) {
      const int ka2 = (k + 2 < kKGroupsB) ? (k + 2) : (kKGroupsB - 1);
      const int ka3 = (k + 3 < kKGroupsB) ? (k + 3) : (kKGroupsB - 1);
      const int kb1 = (k + 1 < kKGroupsB) ? (k + 1) : (kKGroupsB - 1);
      const int kb2 = (k + 2 < kKGroupsB) ? (k + 2) : (kKGroupsB - 1);

      read_a(0);
      load_a(ka2, 0);
      load_b(kb1, 1);
      mfma_k(k, 0);
      store_a(1, 1);
      // Donor literals at mask 0; the gu deletion halves the true MFMA count
      // per half, so these hints are conservative rather than exact — retune
      // is a perf pass, not a correctness item.
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
    // EPILOGUE: dH2 -> (dZ_g, dZ_u) via swiglu' on the saved z; dw row-dots.
    // The staging LDS is used twice sequentially: g-half then u-half.
    // Column identity: this lane's dH2 value (m,c,t) is intermediate index
    //   i = 256*g + 64*wv + 16*c + r      (in [0,2048))
    // for row 16*m + 4*q + t of the sub-block.  z_g = Zq[row][i],
    // z_u = Zq[row][2048 + i]; act(z) = A2q[row][i].
    // =======================================================================
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
      const int rbase = sb * kBlockM;
      bool live[2][4];
      int srow[2][4];
      float dzg[2][kColTilesB][4];
      float dzu[2][kColTilesB][4];
      float dwacc[2][4];
      float lmax_g[2][4];
      float lmax_u[2][4];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const int rr = rbase + 16 * m + 4 * q + t;
          live[m][t] = tok_lds[rr] < T;
          srow[m][t] = srow_lds[rr];
          dwacc[m][t] = 0.0f;
          lmax_g[m][t] = 0.0f;
          lmax_u[m][t] = 0.0f;
        }
      }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int c = 0; c < kColTilesB; ++c) {
          const float* hf = accf(acc[sb][m][c]);
          const int i_col = kChunkColsB * g + kWaveColsB * wv + 16 * c + r;
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const int rr = srow[m][t];
            float dh2 = live[m][t] ? hf[t] : 0.0f;
            // saved z (both halves) + act(z), dequantized per K128 group.
            const std::size_t zbase = static_cast<std::size_t>(rr) * kDZCols;
            const float zscale_g = DQZ[rr * kKGroupsDZ + (i_col >> 7)];
            const float zscale_u =
                DQZ[rr * kKGroupsDZ + ((kInter + i_col) >> 7)];
            const float ascale =
                DQ2[rr * (kInter / 128) + (i_col >> 7)];
            const float gz =
                k0p6t2b_fp8_to_f32(Zq[zbase + i_col]) * zscale_g;
            const float uz =
                k0p6t2b_fp8_to_f32(Zq[zbase + kInter + i_col]) * zscale_u;
            const float az =
                k0p6t2b_fp8_to_f32(A2q[static_cast<std::size_t>(rr) * kInter +
                                       i_col]) *
                ascale;
            const float sig = 1.0f / (1.0f + __expf(-gz));
            const float dg = dh2 * uz * sig * (1.0f + gz * (1.0f - sig));
            const float du = dh2 * gz * sig;
            dzg[m][c][t] = dg;
            dzu[m][c][t] = du;
            dwacc[m][t] += dh2 * az;
            lmax_g[m][t] = fmaxf(lmax_g[m][t], fabsf(dg));
            lmax_u[m][t] = fmaxf(lmax_u[m][t], fabsf(du));
          }
        }
      }
      // dw row-dot partials: reduce across the 16 r-lanes (columns), then one
      // atomic per (row) per wave-quad slice.
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
          for (int mask = 1; mask < 16; mask <<= 1) {
            dwacc[m][t] += __shfl_xor(dwacc[m][t], mask, 64);
          }
          if (r == 0 && live[m][t]) {
            atomicAdd(dw_sorted + srow[m][t], dwacc[m][t]);
          }
        }
      }

      // ---- g-half: amax -> quant -> stage -> store ------------------------
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float (*dzv)[kColTilesB][4] = (half == 0) ? dzg : dzu;
        float (*lmax)[4] = (half == 0) ? lmax_g : lmax_u;
        // re-zero the shared amax accumulators for this half
        for (int i = tid; i < 2 * kMrows; i += kThreads) {
          amax_lds[i / kMrows][i % kMrows] = 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
          for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int mask = 1; mask < 16; mask <<= 1) {
              lmax[m][t] = fmaxf(lmax[m][t], __shfl_xor(lmax[m][t], mask, 64));
            }
          }
        }
        if (r == 0) {
#pragma unroll
          for (int m = 0; m < 2; ++m) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
              atomicMax(reinterpret_cast<int*>(
                            &amax_lds[j_w][rbase + 16 * m + 4 * q + t]),
                        __float_as_int(lmax[m][t]));
            }
          }
        }
        __syncthreads();

#pragma unroll
        for (int m = 0; m < 2; ++m) {
          float dq[4];
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            dq[t] = fmaxf(amax_lds[j_w][rbase + 16 * m + 4 * q + t], 1.0e-6f) /
                    448.0f;
          }
#pragma unroll
          for (int c = 0; c < kColTilesB; ++c) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
              const float qv =
                  fminf(fmaxf(dzv[m][c][t] / dq[t], -448.0f), 448.0f);
              a2_lds[rbase + 16 * m + 4 * q + t]
                    [kWaveColsB * wv + 16 * c + r] =
                  __hip_cvt_float_to_fp8(qv, __HIP_SATFINITE, __HIP_E4M3);
            }
          }
        }
        __syncthreads();

        // coalesced store of this half's 256-col chunk + its DQdZ scales
        {
          const int cr = tid >> 3;
          const int cc = tid & 7;
          if ((rbase >> 5) < gcount) {
            const uint4 s0 = *reinterpret_cast<const uint4*>(
                &a2_lds[rbase + cr][32 * cc]);
            const uint4 s1 = *reinterpret_cast<const uint4*>(
                &a2_lds[rbase + cr][32 * cc + 16]);
            const std::size_t dst =
                (static_cast<std::size_t>(b0 + sb) * kBlockM + cr) * kDZCols +
                static_cast<std::size_t>(half) * kInter + kChunkColsB * g +
                32 * cc;
            *reinterpret_cast<uint4*>(dZq + dst) = s0;
            *reinterpret_cast<uint4*>(dZq + dst + 16) = s1;
          }
        }
        if (q == 0 && (rbase >> 5) < gcount) {
#pragma unroll
          for (int m = 0; m < 2; ++m) {
#pragma unroll
            for (int kb = 0; kb < 2; ++kb) {
              DQdZ[(static_cast<std::size_t>(b0 + sb) * kBlockM + 16 * m + r) *
                       kKGroupsDZ +
                   half * (kKGroupsDZ / 2) + 2 * g + kb] =
                  fmaxf(amax_lds[kb][rbase + 16 * m + r], 1.0e-6f) / 448.0f;
            }
          }
        }
        __syncthreads();
      }
    }
    __syncthreads();

#ifdef N2GM_P1B_ARRIVAL_HOOK
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
      if (sb >= gcount) break;
      const int b = b0 + sb;
      N2GM_P1B_ARRIVAL_HOOK
    }
#endif
    N2GM_P1B_TASK_DONE_HOOK
  }
}

}  // namespace production_fused_moe::n2

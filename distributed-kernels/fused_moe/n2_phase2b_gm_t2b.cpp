// T2B PHASE 2b — dX rows = W13^T · dZ, the backward's second grouped GEMM,
// with the forward phase-2 WRITE-ONCE EPILOGUE VERBATIM: per-sorted-row
// sorted_w weighting (the ONLY correct w-application point — the M1
// match_any dedup makes receive rows multi-expert aggregates, so folding w
// at M8 combine is forbidden; T2_PHASE_MAP.md §4) and the exp_21/exp_24
// remote-accumulate transport into the token owners' slots (depth-throttled
// packed-bf16x2 RMWs, slot_off pre-folded into the peer table).
// Derived from n2_phase2_gm_mps.cpp; design: T2_BACKWARD_MEGA_DESIGN.md.
//
// WHAT IS IDENTICAL to the forward phase 2:
//   * the ENTIRE epilogue: throttle_vmcnt/throttle_plan/epilogue_write are
//     byte-for-byte the forward's (this file replaces, not accompanies, the
//     forward P2 in the t2b TU — no ODR overlap);
//   * the task decomposition (tile, nc in [0,16)), the [kMrows x 448] output
//     tile, N = kHidden = 7168, the nc -> XCD stability property, the
//     G-stacked A-tile pipeline, the m7tab construction, the deferred
//     drain discipline.
//
// WHAT CHANGES (the phase map's caveat (c), exactly):
//   * A = dZq [rowcap, 4096] fp8 + DQdZ [rowcap, 32] f32 (phase-1b's
//     output): kKGroupsP2B = 32 K128 groups (K-loop trip doubles),
//     dq2-analog LDS doubles, srsrc num_records = nvi[0] * 4096;
//   * W = W13T [32, 7168, 4096] fp8 AITER-shuffled (host-pre-transposed per
//     step) + S13T [32, 56, 32] f32 (the transposed 128x128 scale grid —
//     exact, the blockscale grid is symmetric);
//   * the M20 pool indirection is stripped (t2b #errors on M20).
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

// ---- backward geometry -----------------------------------------------------
constexpr int kAColsP2B = 2 * kInter;             // 4096: dZ row width
constexpr int kKGroupsP2B = kAColsP2B / 128;      // 32 K128 groups
constexpr int kNGroupsS13T = kHidden / 128;       // 56 scale n-groups (W13T)

constexpr int kNChunksB2 = 16;                    // output n-chunks per block
constexpr int kChunkNB = kHidden / kNChunksB2;    // 448 output cols per task
constexpr int kWaveTilesB = kChunkNB / 16 / kWaves;  // 7 N16 tiles per wave
constexpr int kWaveCols2B = 16 * kWaveTilesB;        // 112 cols per wave
constexpr int kTaskTilesB = kChunkNB / 16;           // 28 N16 tiles per task

static_assert(kKGroupsP2B % 2 == 0, "the K loop is unrolled by two");
static_assert(kChunkNB % (16 * kWaves) == 0, "N chunk / wave split");
static_assert(kNChunksB2 % 8 == 0, "nc -> XCD (nc mod 8) must be stable");
static_assert(kThreads == kBlockM * kAChunks,
              "the A tile fill must be exactly one 16 B load per thread");
static_assert(kAChunks * kAChunkBytes == 128, "one K128 group per A tile row");

// ---- the forward phase-2 epilogue machinery, VERBATIM ----------------------
template <int VmCnt>
__device__ __forceinline__ void throttle_vmcnt();

#if defined(__HIP_DEVICE_COMPILE__)
#define K0P6_T2B_THROTTLE_VMCNT(N)                                    \
  template <>                                                         \
  __device__ __forceinline__ void throttle_vmcnt<N>() {               \
    asm volatile("s_waitcnt vmcnt(" #N ")" ::: "memory");             \
  }
#else
#define K0P6_T2B_THROTTLE_VMCNT(N)                                    \
  template <>                                                         \
  __device__ __forceinline__ void throttle_vmcnt<N>() {}
#endif
K0P6_T2B_THROTTLE_VMCNT(4)
K0P6_T2B_THROTTLE_VMCNT(8)
K0P6_T2B_THROTTLE_VMCNT(16)
K0P6_T2B_THROTTLE_VMCNT(32)
#undef K0P6_T2B_THROTTLE_VMCNT

inline constexpr unsigned int kThrModeOff = 0u;
inline constexpr unsigned int kThrMode8 = 1u;
inline constexpr unsigned int kThrMode4 = 2u;
inline constexpr unsigned int kThrMode16 = 3u;
inline constexpr unsigned int kThrMode32 = 4u;

struct throttle_plan {
  bool at8;
  bool at4;
  bool at16;
  bool at32;
};

__device__ __forceinline__ throttle_plan make_throttle_plan(
    unsigned int thr_mode) {
  return throttle_plan{thr_mode == kThrMode8, thr_mode == kThrMode4,
                       thr_mode == kThrMode16, thr_mode == kThrMode32};
}

__device__ __forceinline__ void throttle_epilogue_rmw(const throttle_plan& p) {
  if (p.at8) {
    throttle_vmcnt<8>();
  } else if (p.at4) {
    throttle_vmcnt<4>();
  } else if (p.at16) {
    throttle_vmcnt<16>();
  } else if (p.at32) {
    throttle_vmcnt<32>();
  }
}

template <int JMAX>
__device__ __forceinline__ void epilogue_write(
    std::uint32_t (&xp)[kBlockM][32], racc (&acc)[JMAX][2],
    const float (&sw2)[2], const bool (&lv2)[2], const int (&xtok)[16], int T,
    int q, int r, int rowh, int dcol, std::size_t col_base,
    __hip_bfloat16* __restrict__ OUT,
    const unsigned long long* __restrict__ peer_tab,
    int maxtok_sh, unsigned int tok_mask,
    bool dual, const throttle_plan& thr) {
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    const int row = 16 * m + r;
    const float sw = lv2[m] ? sw2[m] : 0.0f;
    const int xorm = 2 * (row & 15);
#pragma unroll
    for (int j = 0; j < JMAX; ++j) {
      const float* cf = accf(acc[j][m]);
      __hip_bfloat162 v0, v1;
      v0.x = __float2bfloat16(cf[0] * sw);
      v0.y = __float2bfloat16(cf[1] * sw);
      v1.x = __float2bfloat16(cf[2] * sw);
      v1.y = __float2bfloat16(cf[3] * sw);
      const int phys = (8 * j + 2 * q) ^ xorm;
      uint2 packed;
      packed.x = *reinterpret_cast<const std::uint32_t*>(&v0);
      packed.y = *reinterpret_cast<const std::uint32_t*>(&v1);
      *reinterpret_cast<uint2*>(&xp[row][phys]) = packed;
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
  __builtin_amdgcn_s_waitcnt(0xc07f);
  __builtin_amdgcn_wave_barrier();

  if (dcol < 8 * JMAX) {
    if (peer_tab != nullptr) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int row = 2 * i + rowh;
        const std::uint32_t d = xp[row][dcol ^ (2 * (row & 15))];
        if (xtok[i] < T) {
          const unsigned int xr = (unsigned int)xtok[i];
          const std::uintptr_t a =
              (std::uintptr_t)peer_tab[xr >> maxtok_sh] +
              ((std::size_t)(xr & tok_mask) * kHidden + col_base + 2 * dcol) *
                  2u;
          kittens::distributed::accumulate_peer_bf162(
              reinterpret_cast<void*>(a), d);
          throttle_epilogue_rmw(thr);
          if (dual) {
            __hip_bfloat162* pl = reinterpret_cast<__hip_bfloat162*>(
                OUT + static_cast<std::size_t>(xtok[i]) * kHidden + col_base +
                2 * dcol);
            unsafeAtomicAdd(pl, *reinterpret_cast<const __hip_bfloat162*>(&d));
          }
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int row = 2 * i + rowh;
        const std::uint32_t d = xp[row][dcol ^ (2 * (row & 15))];
        if (xtok[i] < T) {
          __hip_bfloat162* p = reinterpret_cast<__hip_bfloat162*>(
              OUT + static_cast<std::size_t>(xtok[i]) * kHidden + col_base +
              2 * dcol);
          unsafeAtomicAdd(p, *reinterpret_cast<const __hip_bfloat162*>(&d));
        }
      }
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
}

// ===========================================================================
// PHASE 2b: split-N full-K W13T over dZ, write-once epilogue into dX slots
// ===========================================================================
#ifndef N2_P2B_QUAL
#define N2_P2B_QUAL __global__ __launch_bounds__(kThreads, 1)
#define N2_P2B_NAME n2_phase2b_kernel
#endif
#ifndef N2_HOOK_CTX_ARG
#define N2_HOOK_CTX_ARG
#endif
N2_P2B_QUAL void N2_P2B_NAME(
    const std::uint8_t* __restrict__ dZq,       // [rowcap, 4096] FP8 (phase 1b)
    const float* __restrict__ DQdZ,             // [rowcap, 32] FP32 (phase 1b)
    const std::uint8_t* __restrict__ W13T,      // [32, 7168, 4096] AITER
    const float* __restrict__ S13T,             // [32, 56, 32] FP32
    const int* __restrict__ sorted_ids,         // SAVED plan: token|slot<<24
    const float* __restrict__ sorted_w,         // SAVED plan: the topk weights
    const int* __restrict__ sorted_eid,         // SAVED plan
    const int* __restrict__ nvi,                // SAVED plan: [padded_rows, T]
    __hip_bfloat16* __restrict__ OUT,           // dX [R, 7168] BF16, pre-zeroed
    const int* __restrict__ tile_desc,          // SAVED plan
    int num_tiles
    N2_HOOK_CTX_ARG) {
#ifndef N2GM_G
#define N2GM_G 1
#endif
  constexpr int kGM = N2GM_G;
  constexpr int kMrows = kBlockM * kGM;

  __shared__ int tok_lds[kMrows];
  __shared__ float w_lds[kMrows];
  __shared__ float b2s_lds[kTaskTilesB][kKGroupsP2B];  // [28][32] — doubled
  __shared__ float dq2_lds[kKGroupsP2B][kMrows];       // [32][rows] — doubled
  __shared__ __align__(16)
      std::uint8_t a_lds[2][kMrows][kAChunks][kAChunkBytes];
  __shared__ __align__(16) std::uint32_t xp_lds[kWaves][kBlockM][32];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 63;
  const int wv = tid >> 6;
  const int q = lane >> 4;
  const int r = lane & 15;
  const int T = nvi[1];

  __shared__ unsigned long long m7tab[8];
  const unsigned long long* m7_peer_tab = nullptr;
  int m7_sh = 0;
  unsigned int m7_tok_mask = 0u;
  bool m7_dual = false;
  throttle_plan m7_thr = make_throttle_plan(kThrModeOff);
#ifdef N2GM_TASK_DONE_HOOK
  {
    const unsigned long long m7cfg =
        (unsigned long long)k0p6_dread(k0p6_desc, K0P6_D_MPS_CFG);
    if (hk_moe::mps::mode_is_direct_accum(
            hk_moe::mps::decode_config(m7cfg))) {
#ifndef N2GM_M7TAB_FILL
      const auto* m7sym = k0p6_symmetric(k0p6_desc);
      const unsigned long long m7_slots =
          (unsigned long long)k0p6_dread(k0p6_desc, K0P6_D_MPS_SLOTS);
      const int m7_cur = (int)k0p6_dread(k0p6_desc, K0P6_D_CUR);
      const unsigned int m7_mtok =
          (unsigned int)k0p6_dread(k0p6_desc, K0P6_D_MAXTOK);
      const unsigned long long m7_slot_off =
          (m7_slots - (unsigned long long)(std::uintptr_t)m7sym->local_heap_base)
          + (unsigned long long)m7_cur * (unsigned long long)m7_mtok *
                ((unsigned long long)kHidden * 2ull);
      if (tid < 8) {
        const void* pb = (tid == m7_cur)
            ? (const void*)m7sym->local_heap_base
            : (const void*)m7sym->heap_bases.select<8>(tid);
        m7tab[tid] = (unsigned long long)(std::uintptr_t)pb + m7_slot_off;
      }
#else
      N2GM_M7TAB_FILL
#endif
      m7_sh = 31 - __clz((unsigned int)m7_mtok);
      m7_tok_mask = m7_mtok - 1u;
      m7_peer_tab = m7tab;
      const hk_moe::mps::config m7dec = hk_moe::mps::decode_config(m7cfg);
      m7_dual = hk_moe::mps::detect_dual(m7dec);
      m7_thr = make_throttle_plan((unsigned int)__builtin_amdgcn_readfirstlane(
          (int)(hk_moe::mps::throttle_enabled(m7dec)
                    ? (kThrMode8 + hk_moe::mps::throttle_depth_sel(m7dec))
                    : kThrModeOff)));
    }
  }
#endif

  constexpr int kNChunksP2 = kNChunksB2;
  const int num_tasks = num_tiles * kNChunksP2;

#ifndef N2GM_TASK_START
#define N2GM_TASK_START blockIdx.x
#endif
#ifndef N2GM_TASK_STRIDE
#define N2GM_TASK_STRIDE kCTAs
#endif
  auto n2gm_tile_b0 = [&](int t) __attribute__((always_inline)) {
    return tile_desc[t] >> 4;
  };
  auto n2gm_tile_gcount = [&](int t) __attribute__((always_inline)) {
    return tile_desc[t] & 0xF;
  };

  const std::uint32_t a2_num_records =
      static_cast<std::uint32_t>(nvi[0]) *
      static_cast<std::uint32_t>(kAColsP2B);
  const i32x4 a2_rsrc = make_srsrc(dZq, a2_num_records, 0);

  const int a_row = tid >> 3;
  const int a_chunk = tid & 7;
  const int a_cp = a_chunk ^ (a_row & 7);

  const int rowh = lane >> 5;
  const int dcol = lane & 31;

#ifdef N2GM_TASK_DONE_HOOK
  k0p6_defer m7_pend{-1, 0, 0};
#endif

#if defined(N2GM_TASK_LO) && defined(N2GM_TASK_HI)
  for (int task = (int)(N2GM_TASK_LO) + (int)(N2GM_TASK_START);
       task < (int)(N2GM_TASK_HI); task += (int)(N2GM_TASK_STRIDE)) {
#else
  for (int task = (int)(N2GM_TASK_START); task < num_tasks;
       task += (int)(N2GM_TASK_STRIDE)) {
#endif
#ifdef N2GM_TASK_LOOP_HEAD_HOOK
    N2GM_TASK_LOOP_HEAD_HOOK
#endif
#ifndef N2GM_TASK_DECODE
    const int tile = task / kNChunksP2;
    const int nc = task - tile * kNChunksP2;
#else
    N2GM_TASK_DECODE(task, tile, nc, num_tiles)
#endif
    const int b0 = n2gm_tile_b0(tile);
    const int e = sorted_eid[b0];
    const int gcount = n2gm_tile_gcount(tile);
    const std::uint8_t* const W13Te = W13T;
    const float* const S13Te = S13T;
    const int ew = e;
#ifdef N2GM_TASK_WAIT_HOOK
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
      if (sb >= gcount) break;
      const int b = b0 + sb;
      N2GM_TASK_WAIT_HOOK
    }
#endif

    // ---- per-task LDS setup ------------------------------------------------
    for (int i = tid; i < kMrows; i += kThreads) {
      const int lb32 = i >> 5;
      if (lb32 < gcount) {
        tok_lds[i] = sorted_ids[(b0 + lb32) * kBlockM + (i & 31)] & 0x00FFFFFF;
        w_lds[i] = sorted_w[(b0 + lb32) * kBlockM + (i & 31)];
      } else {
        tok_lds[i] = T;
        w_lds[i] = 0.0f;
      }
    }
    // b2s_lds[t][k] = S13T[e, ng(tile t), k], k in [0,32).
    for (int idx = tid; idx < kTaskTilesB * kKGroupsP2B; idx += kThreads) {
      const int t = idx >> 5;      // /32
      const int k = idx & 31;      // %32
      const int ng = (kChunkNB * nc + 16 * t) >> 7;
      b2s_lds[t][k] = S13Te[(ew * kNGroupsS13T + ng) * kKGroupsP2B + k];
    }
    // dq2_lds[k][row] = DQdZ[(b0*32+row), k] over kMrows, k in [0,32).
    for (int idx = tid; idx < kKGroupsP2B * kMrows; idx += kThreads) {
      const int k = idx / kMrows;
      const int i = idx % kMrows;
      const int lb32 = i >> 5;
      dq2_lds[k][i] =
          (lb32 < gcount)
              ? DQdZ[(static_cast<std::size_t>(b0 + lb32) * kBlockM +
                      (i & 31)) *
                         kKGroupsP2B +
                     k]
              : 0.0f;
    }
    __syncthreads();

    std::uint32_t a2_voff0[kGM];
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
      a2_voff0[sb] =
          (static_cast<std::uint32_t>(b0 + sb) * kBlockM + a_row) * kAColsP2B +
          static_cast<std::uint32_t>(a_chunk * kAChunkBytes);
    }

    racc acc[kGM][kWaveTilesB][2];
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
#pragma unroll
      for (int t = 0; t < kWaveTilesB; ++t) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          zero(acc[sb][t][m]);
        }
      }
    }

    rfp8 af[kGM][2];
    __uint128_t aTmp[kGM][2];
    rfp8 bf[2][kWaveTilesB];

    auto load_a2 = [&](int k, int buf) __attribute__((always_inline)) {
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
        aTmp[sb][buf] = llvm_amdgcn_raw_buffer_load_b128(
            a2_rsrc, a2_voff0[sb] + static_cast<std::uint32_t>(k * 128), 0u, 0u);
      }
    };
    auto store_a2 = [&](int buf, int lb) __attribute__((always_inline)) {
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
        *reinterpret_cast<__uint128_t*>(
            &a_lds[lb][sb * kBlockM + a_row][a_cp][0]) = aTmp[sb][buf];
      }
    };
    auto read_a2 = [&](int lb) __attribute__((always_inline)) {
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
    auto load_w2 = [&](int k, int buf) __attribute__((always_inline)) {
#pragma unroll
      for (int t = 0; t < kWaveTilesB; ++t) {
        const std::size_t tb = aiter::fp8_weight_byte_offset(
            ew, kChunkNB * nc + kWaveCols2B * wv + 16 * t, 128 * k, kHidden,
            kAColsP2B);
        load_bfrag(bf[buf][t], W13Te + tb + 16 * lane);
        stage_agpr(bf[buf][t]);
      }
    };
    auto mfma_k = [&](int k, int buf) __attribute__((always_inline)) {
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
        const float sdq0 = dq2_lds[k][sb * kBlockM + r];
        const float sdq1 = dq2_lds[k][sb * kBlockM + 16 + r];
#pragma unroll
        for (int t = 0; t < kWaveTilesB; ++t) {
          const float fs = b2s_lds[kWaveTilesB * wv + t][k];
#pragma unroll
          for (int m = 0; m < 2; ++m) {
            racc p;
            zero(p);
            mma_ABt(p, bf[buf][t], af[sb][m], p);
            const float s = fs * (m == 0 ? sdq0 : sdq1);
            const f32x2 sv = {s, s};
            const f32x2* pf = accv(p);
            f32x2* cf = accv(acc[sb][t][m]);
            cf[0] += pf[0] * sv;
            cf[1] += pf[1] * sv;
          }
        }
      }
    };

    load_a2(0, 0);
    load_a2(1, 1);
    load_w2(0, 0);
    store_a2(0, 0);
    lds_cta_barrier();

#pragma unroll 1
    for (int k = 0; k < kKGroupsP2B; k += 2) {
      const int ka2 = (k + 2 < kKGroupsP2B) ? (k + 2) : (kKGroupsP2B - 1);
      const int ka3 = (k + 3 < kKGroupsP2B) ? (k + 3) : (kKGroupsP2B - 1);
      const int kb1 = (k + 1 < kKGroupsP2B) ? (k + 1) : (kKGroupsP2B - 1);
      const int kb2 = (k + 2 < kKGroupsP2B) ? (k + 2) : (kKGroupsP2B - 1);

      read_a2(0);
      load_a2(ka2, 0);
      load_w2(kb1, 1);
      mfma_k(k, 0);
      store_a2(1, 1);
      __builtin_amdgcn_sched_group_barrier(0x100, 4 * kGM, 0);
      __builtin_amdgcn_sched_group_barrier(0x020, 8 + 7 * kGM, 0);
      __builtin_amdgcn_sched_group_barrier(0x008, 14 * kGM, 0);
      __builtin_amdgcn_sched_group_barrier(0x200, kGM, 0);
      lds_cta_barrier();

      read_a2(1);
      load_a2(ka3, 1);
      load_w2(kb2, 0);
      mfma_k(kb1, 1);
      store_a2(0, 0);
      __builtin_amdgcn_sched_group_barrier(0x100, 4 * kGM, 0);
      __builtin_amdgcn_sched_group_barrier(0x020, 8 + 7 * kGM, 0);
      __builtin_amdgcn_sched_group_barrier(0x008, 14 * kGM, 0);
      __builtin_amdgcn_sched_group_barrier(0x200, kGM, 0);
      lds_cta_barrier();
    }

    // ---- the forward's write-once epilogue, per live sub-block --------------
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
      if (sb >= gcount) break;
      const int rbase = sb * kBlockM;
      const int tok2[2] = {tok_lds[rbase + r], tok_lds[rbase + 16 + r]};
      const float sw2[2] = {w_lds[rbase + r], w_lds[rbase + 16 + r]};
      const bool lv2[2] = {tok2[0] < T, tok2[1] < T};
      int xtok[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        xtok[i] = tok_lds[rbase + 2 * i + rowh];
      }
      epilogue_write<4>(xp_lds[wv],
                        *reinterpret_cast<racc(*)[4][2]>(&acc[sb][0]), sw2, lv2,
                        xtok, T, q, r, rowh, dcol,
                        static_cast<std::size_t>(kChunkNB) * nc +
                            kWaveCols2B * wv,
                        OUT, m7_peer_tab, m7_sh, m7_tok_mask,
                        m7_dual, m7_thr);
      epilogue_write<3>(xp_lds[wv],
                        *reinterpret_cast<racc(*)[3][2]>(&acc[sb][4]), sw2, lv2,
                        xtok, T, q, r, rowh, dcol,
                        static_cast<std::size_t>(kChunkNB) * nc +
                            kWaveCols2B * wv + 64,
                        OUT, m7_peer_tab, m7_sh, m7_tok_mask,
                        m7_dual, m7_thr);
    }
#ifdef N2GM_TASK_DONE_DRAIN_HOOK
    N2GM_TASK_DONE_DRAIN_HOOK
#endif
    __syncthreads();
#ifdef N2GM_TASK_DONE_HOOK
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
      if (sb >= gcount) break;
      const int b = b0 + sb;
      N2GM_TASK_DONE_HOOK
    }
#endif
  }
#ifdef N2GM_TASK_LOOP_TAIL_HOOK
  N2GM_TASK_LOOP_TAIL_HOOK
#endif
}

}  // namespace production_fused_moe::n2

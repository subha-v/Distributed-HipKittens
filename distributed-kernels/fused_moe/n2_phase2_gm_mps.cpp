// MPS VENDORED COPY — derived from amd-master's pinned donor
//   auto-gpu-kernel/k0_fused_moe/solution/hip/n2_phase2_gm.cpp
//   sha256 7d8beb039b8224e614eacdbe9b34b21837095ffa6ec4842f08d72358b7b8e025
// Donor is byte-identical to the object included by the exp_62/exp_64 winners and
// the exp_35 G=3 champion. This copy exists so the MPS sibling kernel can (a)
// re-stride the task loop over a dense logical compute-id space and (b) drain
// per-thread VMEM before the done hook, WITHOUT touching the frozen upstream.
// The two deltas are marked "// MPS-DELTA (n)". Everything else, including all
// MFMA/LDS/epilogue arithmetic, is byte-identical to the donor. The includer's
// done-hook contract also gains `nc`: k0p6_mps_task_done(b, nc, tid, desc).
//
// exp_47 (N2, phase 2 of 2) — W2 as SPLIT-N FULL-K: every output element written ONCE.
//
// THE ONE VARIABLE (with n2_phase1): the W2 K=2048 reduction decomposition.
// n1g/stock split K into 8 chunks whose bf16 partials collide in
// `global_atomic_pk_add_bf16` — 183.5 MB of atomic writes against a 22.9 MB
// minimum (measured 2026-07-24, input ablation: n1g epilogue+per-token 43.45 us,
// AITER 29.50 us).  Here task = (sorted block b, output n-chunk nc in [0,16)):
// one CTA owns a [32 x 448] output tile and reduces ALL 16 K128 groups of A2 in
// REGISTERS (56 VGPR of accumulators per lane), then writes the tile ONCE.
// The only atomics left are the intrinsic cross-block top-k collisions — total
// atomic traffic 22.9 MB, 8x less than n1g/stock.
//
// Structure per task:
//   * A2 (the phase-1 handoff, [rowcap x 2048] fp8) streams through the SAME
//     CTA-shared, XOR-swizzled, double-buffered LDS tile idiom as n1g's A gather
//     (memory order, 8 fully-used cache lines per instruction, bounded
//     descriptor num_records = nvi[0]*2048).
//   * W2 streams as AITER-preshuffled fragments straight from global (no LDS),
//     7 N16 tiles per wave, double-buffered, AGPR-staged by the backend
//     (-mllvm -amdgpu-mfma-vgpr-form=1, n1g's exp_10 flag).
//   * Per K128 group kb: a FRESH FP32 partial per MFMA (srcC=0, §L0 audit item
//     4), folded with s = fc2_scale[e, ng(tile), kb] * DQ2[row, kb] — the exact
//     dequant-before-write expression from §L0.
//   * Epilogue: x sorted_weights, bf16 pack, n1g's wave-local XOR-swizzled LDS
//     transpose (2 cache lines per atomic), `global_atomic_pk_add_bf16` with
//     `xtok < T` exec-masking (the atomic is UNBOUNDED — the mask is memory
//     safety, §L0).
//
// XCD discipline: nc -> XCD (nc mod 8) is STABLE across blocks (16 mod 8 == 0),
// so every block of an expert re-reads its W2 slices through the SAME XCD's L2 —
// the 1.003x HBM disjointness (§L88) is preserved exactly.
//
// Pipeline: K-loop over 16 K128 groups, unroll-2, 15 loads in flight (14 W2 +
// 1 A2), no vmcnt(0) inside the loop (Gate 1 asserts from the compiled ISA).
// ===========================================================================

#include "kittens.cuh"
#include "n2_fused_moe.hpp"
#include "aiter_gfx950_fp8_layout.hpp"
#ifndef N2_KERNELS_ONLY
#include "pyutils/pyutils.cuh"
#endif

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "n2_device_common.cuh"

namespace production_fused_moe::n2 {

constexpr int kNChunks = 16;                   // output n-chunks per block
constexpr int kChunkN = kHidden / kNChunks;    // 448 output cols per task
constexpr int kWaveTiles = kChunkN / 16 / kWaves;  // 7 N16 tiles per wave
constexpr int kWaveCols2 = 16 * kWaveTiles;        // 112 cols per wave

constexpr int kNGroups2 = kHidden / 128;       // 56 fc2_scale n-groups
constexpr int kTaskTiles = kChunkN / 16;       // 28 N16 tiles per task

static_assert(kKGroups2 % 2 == 0, "the K loop is unrolled by two");
static_assert(kChunkN % (16 * kWaves) == 0, "N chunk / wave split");
static_assert(kNChunks % 8 == 0, "nc -> XCD (nc mod 8) must be stable");
static_assert(kThreads == kBlockM * kAChunks,
              "the A2 tile fill must be exactly one 16 B load per thread");
static_assert(kAChunks * kAChunkBytes == 128, "one K128 group per A2 tile row");

// exp_21 (MPS-DELTA (3)) — DIRECT REMOTE ACCUMULATE target override for the
// write-once epilogue. When the body built a remote target (peer_tab != null),
// each row's atomic goes to the OWNER's slot instead of local `part`:
//   addr = tab[xtok>>sh] + slot_off + ((xtok & mask)*kHidden + col)*2,
// the exact translate_peer arithmetic pre-tabulated into 64 B of LDS (tab[cur]
// holds the local heap base, so the uniform expression covers own-rank rows
// with no select). MAXTOK is a power of two (entry guard), so div/mod are a
// shift/mask and nothing here is a spill candidate. `dual` additionally keeps
// the legacy local-part write -- the review-mandated lost-update DETECTOR
// (M8 compares the two towers row-by-row). When peer_tab == null the whole
// construct folds away and the donor's addressing is byte-identical.
// exp_24 B (MPS-DELTA (5)) — the throttle DEPTH becomes a swept axis.
//
// `s_waitcnt vmcnt(N)` encodes N in the instruction's simm16 and has no
// register form, so the depth must be a compile-time literal selected at run
// time. Two shapes were built and measured (see exp_24_dead_plan_work/build.md):
//
//   REJECTED — template the whole 16-iteration accumulate loop on the depth and
//   switch between four instantiations. Correct, and every literal reached the
//   ISA, but the four-way join at peak epilogue pressure made the allocator
//   spill one value and reload it from scratch ONCE PER ATOMIC: scratch_load
//   went 9 -> 436 with 389 reloads of a single 4 B slot, and the regression hit
//   the depth-8 (control) path too. A measurement contaminated in the control
//   arm is worse than no measurement.
//
//   SHIPPED — one copy of the loop; the throttle guard becomes four mutually
//   exclusive SGPR lane masks (`throttle_plan`) decided before the loop, and the
//   64-bit `slot_off` is folded into the peer table (MPS-DELTA (6)) to pay for
//   them. Measured tuple against the exp_21 build, same compiler and flags:
//   SGPR 106 / VGPR 256 / AGPR 256 unchanged, scratch 144 -> 128 B/lane, LDS
//   unchanged, v_mfma 180 unchanged, flat_atomic_pk_add_bf16 282 unchanged, zero
//   scratch ops inside either MFMA span, and zero scratch ops within 24
//   instructions of any remote atomic (the accidental-spill gate above).
template <int VmCnt>
__device__ __forceinline__ void throttle_vmcnt();

#if defined(__HIP_DEVICE_COMPILE__)
#define K0P6_MPS_THROTTLE_VMCNT(N)                                    \
  template <>                                                         \
  __device__ __forceinline__ void throttle_vmcnt<N>() {               \
    asm volatile("s_waitcnt vmcnt(" #N ")" ::: "memory");             \
  }
#else
#define K0P6_MPS_THROTTLE_VMCNT(N)                                    \
  template <>                                                         \
  __device__ __forceinline__ void throttle_vmcnt<N>() {}
#endif
K0P6_MPS_THROTTLE_VMCNT(4)
K0P6_MPS_THROTTLE_VMCNT(8)
K0P6_MPS_THROTTLE_VMCNT(16)
K0P6_MPS_THROTTLE_VMCNT(32)
#undef K0P6_MPS_THROTTLE_VMCNT

// 0 = throttle off; 1..4 = cap outstanding remote RMWs at 8 / 4 / 16 / 32.
// Mode 1 (depth 8) is exp_21's shipped behaviour and is deliberately the value
// the hot compare tests for.
inline constexpr unsigned int kThrModeOff = 0u;
inline constexpr unsigned int kThrMode8 = 1u;
inline constexpr unsigned int kThrMode4 = 2u;
inline constexpr unsigned int kThrMode16 = 3u;
inline constexpr unsigned int kThrMode32 = 4u;

// The four depths as MUTUALLY EXCLUSIVE booleans, decided once before the
// accumulate loop. This shape is load-bearing and was arrived at by measurement
// (build.md): carrying the 5-valued `thr_mode` itself into the loop cost ONE
// EXTRA LIVE VGPR, which at the epilogue's 256-VGPR ceiling evicted the peer
// table's LDS base address to scratch and put a `scratch_load_dword` 5
// instructions ahead of EVERY remote atomic -- 96 static sites, on the shipped
// depth-8 path too. As four booleans the compiler materialises four SGPR lane
// masks instead (SGPR spills go to v255 lanes via readlane, not to memory), the
// integer dies before the loop, and the hot arm costs exactly what exp_21's
// `if (throttle)` cost: one mask test plus one branch.
struct throttle_plan {
  bool at8;    // exp_21's shipped depth
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
    // g-bit 0x20 at depth 8: exp_21's winning lever, verbatim.
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
  // PHASE 1 (write, MFMA layout): lane 16q+r holds sorted rows {r, 16+r}.
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
  // The transpose is WAVE-LOCAL, so this RAW needs NO CTA barrier.
  __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
  __builtin_amdgcn_s_waitcnt(0xc07f);  // lgkmcnt(0); vmcnt/expcnt free
  __builtin_amdgcn_wave_barrier();

  // PHASE 2 (read, ATOMIC layout): 2 cache lines per atomic.  The output
  // atomic is UNBOUNDED: `xtok[i] < T` is uniform across each 32-lane half, so
  // a padding row's half-wave is EXEC-masked off and forms no address at all.
  if (dcol < 8 * JMAX) {
    if (peer_tab != nullptr) {
      // exp_21 remote-accumulate: the uniform branch is hoisted out of the
      // unrolled body deliberately; the per-row work is shift/mask + one ds
      // read2 + one wide IMAD on top of the donor's atomic form, and its
      // 128B-per-half-wave contiguity is IDENTICAL to the donor's (the
      // fabric's merging behavior is what exp_21 measures).
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int row = 2 * i + rowh;
        const std::uint32_t d = xp[row][dcol ^ (2 * (row & 15))];
        if (xtok[i] < T) {
          const unsigned int xr = (unsigned int)xtok[i];
          // peer_tab already carries slot_off (MPS-DELTA (6)).
          const std::uintptr_t a =
              (std::uintptr_t)peer_tab[xr >> maxtok_sh] +
              ((std::size_t)(xr & tok_mask) * kHidden + col_base + 2 * dcol) *
                  2u;
          kittens::distributed::accumulate_peer_bf162(
              reinterpret_cast<void*>(a), d);
          // exp_24 B: was `if (throttle) asm("s_waitcnt vmcnt(8)")`. Same one
          // mask test on the shipped path; three more depths behind it.
          throttle_epilogue_rmw(thr);
          if (dual) {   // detector: keep the local tower in exact lock-step
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
  // WAR: the next pass's ds_write must not overtake these ds_reads.
  __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
}

// ===========================================================================
// PHASE 2: split-N full-K W2, write-once epilogue
// ===========================================================================
// N2_KERNELS_ONLY includers (k0pf6_mega) redefine the qualifier/name to call the
// same body as a device function; the defaults preserve the standalone kernel.
#ifndef N2_P2_QUAL
#define N2_P2_QUAL __global__ __launch_bounds__(kThreads, 1)
#define N2_P2_NAME n2_phase2_kernel
#endif
#ifndef N2_HOOK_CTX_ARG
#define N2_HOOK_CTX_ARG
#endif
N2_P2_QUAL void N2_P2_NAME(
    const std::uint8_t* __restrict__ A2q,       // [rowcap, 2048] FP8 (phase 1)
    const float* __restrict__ DQ2,              // [rowcap, 16] FP32 (phase 1)
    const std::uint8_t* __restrict__ W2,        // down [32, 7168, 2048] AITER
    const float* __restrict__ S2,               // fc2_scale [32, 56, 16]
    const int* __restrict__ sorted_ids,         // packed: token|slot<<24
    const float* __restrict__ sorted_w,
    const int* __restrict__ sorted_eid,
    const int* __restrict__ nvi,                // [padded_rows, T]
    __hip_bfloat16* __restrict__ OUT,           // out [R, 7168] BF16, pre-zeroed
    const int* __restrict__ tile_desc,          // exp_59: [num_tiles] (b0<<4)|gcount
    int num_tiles                               // exp_59: expert-aligned tile count
    N2_HOOK_CTX_ARG) {
  // exp_59 G-stacking: a task spans kGM consecutive same-expert 32-blocks.
#ifndef N2GM_G
#define N2GM_G 1
#endif
  constexpr int kGM = N2GM_G;
  constexpr int kMrows = kBlockM * kGM;              // 32*G sorted rows per task

  __shared__ int tok_lds[kMrows];
  __shared__ float w_lds[kMrows];
  __shared__ float b2s_lds[kTaskTiles][kKGroups2];   // [28 tiles][kb] — per expert
  __shared__ float dq2_lds[kKGroups2][kMrows];       // [kb][sorted row]
  __shared__ __align__(16)
      std::uint8_t a_lds[2][kMrows][kAChunks][kAChunkBytes];
  // xp transpose scratch stays per-32-block; the epilogue reuses it per sub-block.
  __shared__ __align__(16) std::uint32_t xp_lds[kWaves][kBlockM][32];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 63;
  const int wv = tid >> 6;
  const int q = lane >> 4;
  const int r = lane & 15;
  const int T = nvi[1];

  // MPS-DELTA (3) — exp_21 mode 12 target construction. Runs ONCE per body.
  // The live footprint across the task loop is three scalars + 64 B of LDS:
  // the descriptor's ten-pointer carriage never enters a register live range
  // near the MFMA peak, which is the resource discipline the donor demands.
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
      // exp_24 (MPS-DELTA (6)) — `slot_off` is PRE-ADDED into the peer table
      // instead of being carried into the epilogue as a live 64-bit value.
      // It is uniform across owners, so `tab[i] + slot_off` is exact, and the
      // epilogue's address collapses from
      //   tab[owner] + slot_off + (pos*kHidden + col)*2
      // to
      //   tab'[owner] + (pos*kHidden + col)*2.
      // This is a REGISTER-PRESSURE ENABLER, not a free lunch we took on the
      // side: the epilogue sits at 256 VGPR / 256 AGPR, and exp_24 B's four
      // depth masks could not be added without evicting something. With
      // `slot_off` still live the allocator evicted the peer table's LDS base
      // and reloaded it from scratch once per remote atomic (96 static sites).
      // Folding it frees the AGPR pair that held it and removes two
      // `v_accvgpr_read` plus one `v_lshl_add_u64` from every atomic. Measured
      // in build.md; it also means exp_24's control arm is exp_21's protocol
      // with a slightly lighter address computation, which is why the batch
      // carries its own in-batch control at first AND last position.
      const unsigned long long m7_slot_off =
          (m7_slots - (unsigned long long)(std::uintptr_t)m7sym->local_heap_base)
          + (unsigned long long)m7_cur * (unsigned long long)m7_mtok *
                ((unsigned long long)kHidden * 2ull)
#if K0P6_MPS_ENABLE_TBO
          // exp_02 (MPS-DELTA (7)) — producer-side slot PARITY for mode 16's
          // deferred combine: launch e accumulates into parity (e&1); the
          // owner's deferred M8 consumes that same generation inside launch
          // e+1, and launch e+2 reuses the first half once its retirement
          // edge passes. Stride = all 8 peers' per-epoch slot bytes. The term
          // is uniform and folds to +0 in every non-TBO mode, so mode 12's
          // epilogue arithmetic is bit-stable in a TBO build too.
          + (hk_moe::mps::mode_is_tbo(hk_moe::mps::decode_config(m7cfg))
                 ? (k0p6_epoch(k0p6_desc) & 1ull)
                 : 0ull) *
                (8ull * (unsigned long long)m7_mtok *
                 (unsigned long long)kHidden * 2ull)
#endif
          ;
      if (tid < 8) {
        const void* pb = (tid == m7_cur)
            ? (const void*)m7sym->local_heap_base
            : (const void*)m7sym->heap_bases.select<8>(tid);
        m7tab[tid] = (unsigned long long)(std::uintptr_t)pb + m7_slot_off;
      }
#else
      // M15-DELTA (B): includer-supplied target-table fill (e.g. the staged
      // local-fold arm points m7tab at a LOCAL stage laid out per owner so
      // the epilogue's shift/mask/atomic instruction stream is bit-identical
      // and only the pointer VALUES change). Contract: the macro must declare
      // `const unsigned int m7_mtok` and store m7tab[tid] for tid < 8. The
      // undefined arm above is the donor's token stream verbatim.
      N2GM_M7TAB_FILL
#endif
      m7_sh = 31 - __clz((unsigned int)m7_mtok);  // MAXTOK = 2^m7_sh (guard)
      m7_tok_mask = m7_mtok - 1u;
      m7_peer_tab = m7tab;
      // exp_24: the throttle enable and its new depth selector both come from
      // the adapter's accessors now, so the `g` bit table has exactly one
      // definition (moe_mps_adapter.cuh). `throttle_enabled` is the same bit
      // test exp_21 open-coded here as `((m7cfg >> 8) & 0x20) != 0`, and the
      // depth selector's 0 (today's depth 8) maps to kThrMode8, so `g = 33`
      // still lands on exp_21's depth.
      const hk_moe::mps::config m7dec = hk_moe::mps::decode_config(m7cfg);
      m7_dual = hk_moe::mps::detect_dual(m7dec);
      // readfirstlane is load-bearing, not decoration: the mode arrives through
      // a volatile descriptor read, so the compiler cannot prove it uniform and
      // would park it in a VGPR, extending a VECTOR live range across the
      // epilogue. make_throttle_plan then turns it into four SGPR lane masks
      // whose spills go to v255 lanes rather than to memory.
      m7_thr = make_throttle_plan((unsigned int)__builtin_amdgcn_readfirstlane(
          (int)(hk_moe::mps::throttle_enabled(m7dec)
                    ? (kThrMode8 + hk_moe::mps::throttle_depth_sel(m7dec))
                    : kThrModeOff)));
      // The first epilogue reads the table at the END of task 0; the task
      // loop's own LDS-fill __syncthreads() orders the fill before it, so no
      // extra barrier is spent here.
    }
  }
#endif

  // exp_59: task = (tile, n-chunk nc); 16 n-chunks per tile. tile_desc packs
  // (b0<<4)|gcount. At kGM==1, num_tiles == padded/32 and this equals the donor's
  // (padded/32)*16.
  constexpr int kNChunksP2 = 16;
  const int num_tasks = num_tiles * kNChunksP2;

  // MPS-DELTA (1): task-loop start/stride are macro-parameterized so the
  // includer can map a dense logical compute-id space onto the same task set.
  // Defaults reproduce the donor exactly (physical bid, full grid stride).
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

  // Bounded A2 descriptor: rows < nvi[0] are the only ones phase 2 can name,
  // and phase 1 wrote every one of them this epoch.
  const std::uint32_t a2_num_records =
      static_cast<std::uint32_t>(nvi[0]) * static_cast<std::uint32_t>(kInter);
  const i32x4 a2_rsrc = make_srsrc(A2q, a2_num_records, 0);

  const int a_row = tid >> 3;
  const int a_chunk = tid & 7;
  const int a_cp = a_chunk ^ (a_row & 7);

  const int rowh = lane >> 5;   // 0 or 1: which of the atomic's two rows
  const int dcol = lane & 31;   // which dword (= 2 adjacent output columns)

  // MPS-DELTA (4) — deferred event publication for remote-accumulate modes.
  // The epilogue's REMOTE atomics get a whole task of fabric-ACK slack: task
  // t-1's (b,nc) events publish at the TOP of task t (where the drain's
  // vmcnt(0) finds them already acknowledged) instead of stalling the
  // epilogue's critical path per task on a remote ACK. pend state is written
  // uniformly by all threads (the hook visits per sub-block), so the
  // __syncthreads() in the flush is convergent.
#ifdef N2GM_TASK_DONE_HOOK
  k0p6_defer m7_pend{-1, 0, 0};
#endif

  // M15-DELTA (A): optional half-open task range + runtime-order decode. Both
  // default-undefined, and the undefined arms below are the donor's exact
  // token stream (gate: .text identity of every existing includer's build).
  // An includer that defines N2GM_TASK_LO/N2GM_TASK_HI (e.g. as names bound
  // through N2_HOOK_CTX_ARG) slices the task space into slabs; one that
  // defines N2GM_TASK_DECODE picks a different task->(tile,nc) order (the
  // nc-major order of exp_29/exp_35 rung (e)).
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
    const int nc = task - tile * kNChunksP2;   // output n-chunk 0..15
#else
    N2GM_TASK_DECODE(task, tile, nc, num_tiles)
#endif
    const int b0 = n2gm_tile_b0(tile);         // first 32-block of the tile
    const int e = sorted_eid[b0];
    const int gcount = n2gm_tile_gcount(tile); // live sub-blocks (1..kGM)
#ifdef N2GM_TASK_WAIT_HOOK
    // exp_59: wait a2 for each LIVE sub-block (a2_done indexed by 32-block b0+sb).
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
      if (sb >= gcount) break;
      const int b = b0 + sb;
      N2GM_TASK_WAIT_HOOK
    }
#endif

    // ---- per-task LDS setup: tok/w/dq2 over kMrows (per-row); b2s per e ------
    for (int i = tid; i < kMrows; i += kThreads) {
      const int lb32 = i >> 5;
      if (lb32 < gcount) {
        tok_lds[i] = sorted_ids[(b0 + lb32) * kBlockM + (i & 31)] & 0x00FFFFFF;
        w_lds[i] = sorted_w[(b0 + lb32) * kBlockM + (i & 31)];
      } else {
        tok_lds[i] = T;      // pad sub-block -> sentinel (never live)
        w_lds[i] = 0.0f;
      }
    }
    // b2s_lds[t][kb] = fc2_scale[e, ng(tile t), kb], tile t in [0,28):
    // ng = (448*nc + 16*t) / 128.
    for (int idx = tid; idx < kTaskTiles * kKGroups2; idx += kThreads) {
      const int t = idx >> 4;
      const int k = idx & 15;
      const int ng = (kChunkN * nc + 16 * t) >> 7;
      b2s_lds[t][k] = S2[(e * kNGroups2 + ng) * kKGroups2 + k];
    }
    // dq2_lds[kb][row] = DQ2[(b0*32+row), kb] over kMrows.
    for (int idx = tid; idx < kKGroups2 * kMrows; idx += kThreads) {
      const int k = idx / kMrows;
      const int i = idx % kMrows;
      const int lb32 = i >> 5;
      dq2_lds[k][i] = (lb32 < gcount)
                          ? DQ2[(static_cast<std::size_t>(b0 + lb32) * kBlockM +
                                 (i & 31)) *
                                    kKGroups2 +
                                k]
                          : 0.0f;
    }
    __syncthreads();

    // A2 tile fill coordinates per sub-block (a_row is row within a 32-block).
    std::uint32_t a2_voff0[kGM];
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
      a2_voff0[sb] =
          (static_cast<std::uint32_t>(b0 + sb) * kBlockM + a_row) * kInter +
          static_cast<std::uint32_t>(a_chunk * kAChunkBytes);
    }

    racc acc[kGM][kWaveTiles][2];   // [sub-block][n tile][rowtile]
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
#pragma unroll
      for (int t = 0; t < kWaveTiles; ++t) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          zero(acc[sb][t][m]);
        }
      }
    }

    rfp8 af[kGM][2];        // [sub-block][rowtile] A2(k)
    __uint128_t aTmp[kGM][2];
    rfp8 bf[2][kWaveTiles]; // [buf][n tile] W2 fragments — shared across sub-blocks

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
      for (int t = 0; t < kWaveTiles; ++t) {
        const std::size_t tb = aiter::fp8_weight_byte_offset(
            e, kChunkN * nc + kWaveCols2 * wv + 16 * t, 128 * k, kHidden,
            kInter);
        load_bfrag(bf[buf][t], W2 + tb + 16 * lane);
        stage_agpr(bf[buf][t]);
      }
    };
    auto mfma_k = [&](int k, int buf) __attribute__((always_inline)) {
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
        const float sdq0 = dq2_lds[k][sb * kBlockM + r];
        const float sdq1 = dq2_lds[k][sb * kBlockM + 16 + r];
#pragma unroll
        for (int t = 0; t < kWaveTiles; ++t) {
          const float fs = b2s_lds[kWaveTiles * wv + t][k];
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

    // ---- the pipeline: K-loop over the FULL intermediate dim. ---------------
    load_a2(0, 0);
    load_a2(1, 1);
    load_w2(0, 0);
    store_a2(0, 0);
    lds_cta_barrier();

#pragma unroll 1
    for (int k = 0; k < kKGroups2; k += 2) {
      const int ka2 = (k + 2 < kKGroups2) ? (k + 2) : (kKGroups2 - 1);
      const int ka3 = (k + 3 < kKGroups2) ? (k + 3) : (kKGroups2 - 1);
      const int kb1 = (k + 1 < kKGroups2) ? (k + 1) : (kKGroups2 - 1);
      const int kb2 = (k + 2 < kKGroups2) ? (k + 2) : (kKGroups2 - 1);

      // sched_group_barrier hints scale with kGM: read_a2/mfma issue kGM* the
      // per-sub-block work; VMEM read (W2 load) and DS write are per-sub-block-A
      // (kGM* store_a2) plus the shared W2 load (kWaveTiles). Kept proportional.
      // ---- half 0: step k.   LDS buf 0, B buf 0. -------------------------
      read_a2(0);
      load_a2(ka2, 0);
      load_w2(kb1, 1);
      mfma_k(k, 0);
      store_a2(1, 1);
      __builtin_amdgcn_sched_group_barrier(0x100, 4 * kGM, 0);   // DS read
      __builtin_amdgcn_sched_group_barrier(0x020, 8 + 7 * kGM, 0);  // VMEM read
      __builtin_amdgcn_sched_group_barrier(0x008, 14 * kGM, 0);  // MFMA
      __builtin_amdgcn_sched_group_barrier(0x200, kGM, 0);       // DS write
      lds_cta_barrier();

      // ---- half 1: step k+1.  LDS buf 1, B buf 1. ------------------------
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

    // =======================================================================
    // WRITE-ONCE EPILOGUE, per LIVE sub-block. xp_lds is per-32-block scratch
    // shared across sub-blocks; the wave_barrier inside epilogue_write serializes
    // its reuse, so sub-blocks run sequentially through it. Row indices into
    // tok_lds/w_lds/acc are offset by sb*kBlockM.
    // =======================================================================
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
                        static_cast<std::size_t>(kChunkN) * nc + kWaveCols2 * wv,
                        OUT, m7_peer_tab, m7_sh, m7_tok_mask,
                        m7_dual, m7_thr);
      epilogue_write<3>(xp_lds[wv],
                        *reinterpret_cast<racc(*)[3][2]>(&acc[sb][4]), sw2, lv2,
                        xtok, T, q, r, rowh, dcol,
                        static_cast<std::size_t>(kChunkN) * nc +
                            kWaveCols2 * wv + 64,
                        OUT, m7_peer_tab, m7_sh, m7_tok_mask,
                        m7_dual, m7_thr);
    }
    // MPS-DELTA (2): per-thread VMEM drain BEFORE the task-end barrier. The
    // epilogue's global_atomic_pk_add_bf16 stores are asynchronous;
    // __syncthreads() does not wait for them. The MPS stream mode publishes
    // (b, nc) completion from tid0 immediately after the barrier, so every
    // lane first drains its own atomics — the exp_56 M1 release discipline
    // applied at task end. Default expands to nothing (donor schedule).
#ifdef N2GM_TASK_DONE_DRAIN_HOOK
    N2GM_TASK_DONE_DRAIN_HOOK
#endif
    // The next task refills every LDS buffer this one just read.
    __syncthreads();
    // exp_59: part arrival fires per LIVE sub-block (part_done indexed by the
    // absolute 32-block b0+sb). Exact 1:1 tile ownership => each part_done[b]
    // reaches 16 from its owning tile's 16 n-chunk tasks only.
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

#ifndef N2_KERNELS_ONLY
namespace production_fused_moe::n2 {
// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------
struct n2_phase2_globals {
  tensor_bf16 a2q_bytes;          // scratch [rowcap, 1024] BF16-view (=2048 B rows)
  tensor_fp32 dq2;                // scratch [rowcap, 16] FP32
  tensor_bf16 w2_bytes;           // BF16 view of production `down` (AITER)
  tensor_fp32 w2_scales;          // production `fc2_scale`
  tensor_i32 sorted_token_ids;
  tensor_fp32 sorted_weights;
  tensor_i32 sorted_expert_ids;
  tensor_i32 num_valid_ids;
  tensor_bf16 out_bf16;           // production `out`, pre-zeroed in [0,T)
  std::uintptr_t stream_ptr;
};

[[noreturn]] static void invalid(const char* message) {
  throw std::invalid_argument(std::string("n2_phase2: ") + message);
}

static void require(bool condition, const char* message) {
  if (!condition) {
    invalid(message);
  }
}

void validate_contract(const n2_phase2_globals& g) {
  require(aiter::is_supported_blockscale_shape(kExperts, kHidden, kInter),
          "W2 AITER shape unsupported");
  require(g.a2q_bytes.cols() == kInter / 2,
          "a2q must have BF16-view cols 1024 (=2048 B rows)");
  require(g.dq2.cols() == kKGroups2, "dq2 must have FP32 cols 16");
  require(g.w2_bytes.batch() == 1 && g.w2_bytes.depth() == 1 &&
              g.w2_bytes.rows() == kExperts * kHidden &&
              g.w2_bytes.cols() == kInter / 2,
          "down must have BF16-view shape [32*7168,1024]");
  require(g.w2_scales.batch() == 1 && g.w2_scales.depth() == kExperts &&
              g.w2_scales.rows() == kNGroups2 &&
              g.w2_scales.cols() == kKGroups2,
          "fc2_scale must have FP32 shape [32,56,16]");
  require(g.num_valid_ids.cols() == 2, "num_valid_ids must hold two values");
  require(g.out_bf16.batch() == 1 && g.out_bf16.depth() == 1 &&
              g.out_bf16.cols() == kHidden,
          "out must have BF16 cols 7168");
  require(g.sorted_token_ids.cols() / kBlockM <= g.sorted_expert_ids.cols(),
          "sorted_token_ids capacity implies more blocks than sorted_expert_ids "
          "can name");
  require(g.sorted_weights.cols() == g.sorted_token_ids.cols(),
          "sorted_weights and sorted_token_ids must have equal capacity");
  require(static_cast<std::size_t>(g.a2q_bytes.rows()) >=
              static_cast<std::size_t>(g.sorted_token_ids.cols()),
          "a2q must cover the sorted row capacity");
}

void dispatch_n2_phase2(n2_phase2_globals g) {
  validate_contract(g);
  hipStream_t stream = reinterpret_cast<hipStream_t>(g.stream_ptr);
  n2_phase2_kernel<<<kCTAs, kThreads, 0, stream>>>(
      reinterpret_cast<const std::uint8_t*>(g.a2q_bytes.raw_ptr),
      g.dq2.raw_ptr,
      reinterpret_cast<const std::uint8_t*>(g.w2_bytes.raw_ptr),
      g.w2_scales.raw_ptr, g.sorted_token_ids.raw_ptr,
      g.sorted_weights.raw_ptr, g.sorted_expert_ids.raw_ptr,
      g.num_valid_ids.raw_ptr,
      reinterpret_cast<__hip_bfloat16*>(g.out_bf16.raw_ptr));
}

}  // namespace production_fused_moe::n2

void bind_n2_phase2(pybind11::module_& module) {
  using production_fused_moe::n2::dispatch_n2_phase2;
  using production_fused_moe::n2::n2_phase2_globals;
  py::bind_function<dispatch_n2_phase2>(
      module, "n2_phase2",
      &n2_phase2_globals::a2q_bytes,
      &n2_phase2_globals::dq2,
      &n2_phase2_globals::w2_bytes,
      &n2_phase2_globals::w2_scales,
      &n2_phase2_globals::sorted_token_ids,
      &n2_phase2_globals::sorted_weights,
      &n2_phase2_globals::sorted_expert_ids,
      &n2_phase2_globals::num_valid_ids,
      &n2_phase2_globals::out_bf16,
      &n2_phase2_globals::stream_ptr);
}
#endif  // N2_KERNELS_ONLY

// MPS VENDORED COPY — derived from amd-master's pinned donor
//   auto-gpu-kernel/k0_fused_moe/solution/hip/n2_phase1_gm.cpp
//   sha256 1d90b26658b6a524db69434ddcfa0b2dcec2418c852f83874468d55dd0197dc2
// Donor located and hashed on gbt350-odcdh2-c05-1 2026-08-11 (585 lines,
// 23,662 B); byte-identical after LF normalization to the read-only mirror
// .node/mps_n2_phase1_gm.cpp, and to the source_snapshot copies under
// experiments/exp_59_large_m_tiles and exp_63_large_m_rerun. This is the object
// k0pf6gm_device_tile{,_mps}.hip includes today at :395.
//
// This copy exists so the MPS sibling kernel can (a) scale phase 1's four
// sched_group_barrier hints by kGM — the donor's literals are hardcoded for
// kGM==1 and steer only a third of the shipped kGM==3 instruction mix — and
// (b) carry four empty sub-phase instrumentation hooks and a parameterized task
// loop, WITHOUT touching the frozen upstream. Deltas are marked
// "// MPS-DELTA (n)". Everything else, including all MFMA/LDS/epilogue
// arithmetic, the pipeline depth, the barrier placement and every scale, is
// byte-identical to the donor.
//
// The ten MPS-DELTA sites, in file order (donor line numbers in brackets):
//   (1) [:73]  N2GM_P1_SCHED_GSCALE, a 4-BIT MASK, default 0 (= donor literals),
//              + static_assert + the four N2GM_P1_N_* group-size macros.
//   (2) [:92]  Four instrumentation hook macros, empty by default.
//   (3) [:92]  N2GM_P1_TASK_START / N2GM_P1_TASK_STRIDE, defaulting to the
//              donor's blockIdx.x / kCTAs.
//   (4) [:156] The task loop header, bounds taken from (3).
//   (5) [:161] N2GM_P1_TASK_HEAD_HOOK     — top of the task loop.
//   (6) [:317] N2GM_P1_KLOOP_ENTER_HOOK   — after lds_cta_barrier, before for(k).
//   (7) [:331] K-loop half 0 sched hints, counts from (1)'s mask.
//   (8) [:344] K-loop half 1 sched hints, counts from (1)'s mask.
//   (9) [:351] N2GM_P1_KLOOP_EXIT_HOOK    — immediately after the K-loop closes.
//  (10) [:473] N2GM_P1_EPILOGUE_DONE_HOOK — after the A2q/DQ2 stores, before the
//              task-end __syncthreads().
//
// (5),(6),(9),(10) expand to nothing unless the includer defines them, and all
// four sit OUTSIDE both MFMA spans. With N2GM_P1_SCHED_GSCALE at its default 0,
// N2GM_P1_TASK_{START,STRIDE} at their defaults and no hook defined, this file
// is semantically the donor: (1)'s #else arms hold the donor's four literals
// verbatim, and (4) is the donor's loop with two no-op casts. That claim is
// MEASURED, not asserted — the resulting .text hashes byte-identical to a build
// of the donor itself (exp_26 build.md §5, §8).
//
// The kGM scaling in (7)/(8) is a PARAMETERIZATION, NOT A RETUNE. Per-half
// instruction census (exp_26 design.md §2), every expression evaluating to the
// donor's existing literal at kGM==1:
//   DS read  0x100  read_a: kGM * m(2) * 2 float4        = 4*kGM  -> 4  at kGM=1
//   VMEM rd  0x020  load_b 8 frags * 2 + load_a kGM      = 16+kGM -> 17 at kGM=1
//   MFMA     0x008  kGM * m(2) * gu(2) * kColTiles(4)    = 16*kGM -> 16 at kGM=1
//   DS write 0x200  store_a: kGM * ds_write_b128         = kGM    -> 1  at kGM=1
// The 2*kGM ascale_lds + 2 b1s_lds ds_reads issued inside mfma_k stay UNHINTED,
// exactly as in the donor and exactly as in phase 2 (whose 0x100 hint is
// likewise read_a2-only, n2_phase2_gm.cpp:358). Folding them in would make the
// expression 6*kGM+2 = 8 at kGM==1, contradicting the donor's literal 4 — i.e.
// it would be a retune, and it is deliberately not done here.
// ===========================================================================

// exp_47 (N2, phase 1 of 2) — W13 with the split-K W2 DELETED and A2 handed off
// through a small GLOBAL buffer instead of LDS.
//
// THE ONE VARIABLE (the pair n2_phase1 + n2_phase2 forms one mechanism): the W2
// K=2048 reduction decomposition.  n1g/stock: split-K 8 chunks x bf16 atomic
// partials (8x write amplification, 183.5 MB vs a 22.9 MB minimum; measured 2026-07-24
// by input ablation: n1g's epilogue+per-token = 43.45 us, AITER's = 29.50 us).
// n2: split-N full-K — every output element is written ONCE.
//
// PHASE 1 IS n1g's GEMM-1 VERBATIM (task = (sorted block b, intermediate chunk g),
// the CTA-shared A tile, the 56-step paired gate/up K-loop with 17 loads in flight,
// register SiLU, dynamic per-(row,K128) amax, FP8 e4m3 quant) — the math, the
// schedule, and every scale are bit-identical to n1g's.  What changes is ONLY where
// the result goes:
//
//   A2q[row_global, 256g : +256]  (fp8 e4m3, row-major, 2048 B rows)
//   DQ2[row_global, 2g + kb]      (fp32 dequant scale = max(amax,1e-6)/448)
//
// Both are fully written for every row < nvi[0] on every call (padding rows carry
// exact zeros from the live-select), so phase 2 never reads a stale byte — no
// zeroing, no epoch state, route-swap safe by construction.
//
// REMOVED from n1g: the entire split-K GEMM-2 (load_n / compute_n / the N-loop),
// b2s_lds, xp_lds, and every output atomic.  This kernel writes NO output rows.
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

// ---- production geometry (CLAUDE.md I/O contract) --------------------------
constexpr int kW13N = 4096;      // rows [0,2048) gate, [2048,4096) up

// ---- the decomposition (stock's) ------------------------------------------
constexpr int kChunks = 8;                        // intermediate chunks
constexpr int kChunkCols = kInter / kChunks;      // 256
constexpr int kWaveCols = kChunkCols / kWaves;    // 64 intermediate cols/wave
constexpr int kColTiles = kWaveCols / 16;         // 4 x 16-col MFMA tiles

constexpr int kKGroups = kHidden / 128;           // 56 K128 groups in W13
constexpr int kNGroups13 = kW13N / 128;           // 32

// A2 LDS row stride: 256 + 16.
constexpr int kA2Stride = 272;

static_assert(kKGroups % 2 == 0, "the K loop is unrolled by two");
static_assert(kChunkCols % (16 * kWaves) == 0, "W13 chunk / wave split");
static_assert(kThreads == kBlockM * kAChunks,
              "the A tile fill must be exactly one 16 B load per thread");
static_assert(kAChunks * kAChunkBytes == 128, "one K128 group per A tile row");

static_assert(rfp8::height == 1 && rfp8::width == 1);
static_assert(racc::height == 1 && racc::width == 1);

#ifndef N2_FORCE_VMCNT0
#define N2_FORCE_VMCNT0 0
#endif
static_assert(N2_FORCE_VMCNT0 == 0 || N2_FORCE_VMCNT0 == 1);

// MPS-DELTA (1): opt-in kGM scaling of the K-loop sched_group_barrier hints,
// as a 4-BIT MASK so each hint class is independently A/B-able from one source.
//
//   bit 0 (1)  DS read    0x100   donor 4    ->  4 * kGM   (12 at kGM=3)
//   bit 1 (2)  VMEM read  0x020   donor 17   ->  16 + kGM  (19 at kGM=3)
//   bit 2 (4)  MFMA       0x008   donor 16   ->  16 * kGM  (48 at kGM=3)
//   bit 3 (8)  DS write   0x200   donor 1    ->  kGM       ( 3 at kGM=3)
//
// 0 (the default) emits the donor's four literals verbatim, so the default
// build IS the donor schedule; 15 scales all four. Each scaled expression
// equals its donor literal at kGM == 1 INDEPENDENTLY of the other three bits,
// so all 16 masks are donor-identical at kGM == 1 and the acceptance property
// of design.md §2.2 holds bit by bit, not just for the all-on case.
//
// Why a mask and not a bool: the first all-on build (exp_26 B2) bought the
// schedule win but also migrated 96 scratch accesses into the
// M7-epilogue/M8/M9 region -- mode 12's hot fabric path. The mask exists to
// separate the two effects and find the cheapest subset that still moves the
// drain. See build.md §7 for the measured per-subset table.
#ifndef N2GM_P1_SCHED_GSCALE
#define N2GM_P1_SCHED_GSCALE 0
#endif
static_assert(N2GM_P1_SCHED_GSCALE >= 0 && N2GM_P1_SCHED_GSCALE <= 15,
              "N2GM_P1_SCHED_GSCALE is a 4-bit mask: DSR|VMEM|MFMA|DSW");

// The four group sizes, resolved once at preprocessing time so each K-loop half
// keeps the donor's four-line shape. Every #else arm below is the donor's
// literal from n2_phase1_gm.cpp:331-334 / :344-347, verbatim.
#if (N2GM_P1_SCHED_GSCALE & 1)
#define N2GM_P1_N_DSREAD (4 * kGM)
#else
#define N2GM_P1_N_DSREAD 4
#endif
#if (N2GM_P1_SCHED_GSCALE & 2)
#define N2GM_P1_N_VMEMRD (16 + kGM)
#else
#define N2GM_P1_N_VMEMRD 17
#endif
#if (N2GM_P1_SCHED_GSCALE & 4)
#define N2GM_P1_N_MFMA (16 * kGM)
#else
#define N2GM_P1_N_MFMA 16
#endif
#if (N2GM_P1_SCHED_GSCALE & 8)
#define N2GM_P1_N_DSWRITE (kGM)
#else
#define N2GM_P1_N_DSWRITE 1
#endif

// MPS-DELTA (9): the memory layout the activation-scale gather reads.
//
//   0 (default) = donor. GROUP-major, A_scale[k * T + token], T = nvi[1] = T_ext
//                 = 32,768, so one token's 56 scales are 131,072 B apart. The
//                 96x56 gather touches 5,376 distinct 64 B lines to deliver
//                 21,504 B: 16x line amplification, 344 KB per task.
//   1           = TOKEN-major, A_scale[token * kKGroups + k]. A row is 224 B and
//                 224*token mod 64 is always 0 or 32, so a row spans EXACTLY 4
//                 lines: 384 lines for the same 21,504 B. 14.0x less line
//                 traffic, and the gather sits between two __syncthreads() where
//                 no MFMA can cover it.
//
// The two arrays hold the same values BY CONSTRUCTION, so this switch is
// bit-identical by definition rather than by tolerance. M5's
// hk_moe::mps::scale_transpose_row (moe_mps_adapter.cuh:298-305) writes
// sc_dst[lane * T_ext + t] = sc_stage[t * 56 + lane] for every t in [0, T_ext)
// and every lane < 56, and hkp_sort.hpp's scan() sets nvi[1] = T_loc = T_ext, so
// the group-major array M6 reads today is a pure transposed copy of the
// token-major one it reads at 1. The transposed copy has exactly one consumer,
// which is this gather, so at 1 the M5 transpose is dead and is deleted.
#ifndef N2GM_P1_ASCALE_TOKEN_MAJOR
#define N2GM_P1_ASCALE_TOKEN_MAJOR 0
#endif

// T2B-DELTA: opt-in save of z = (gate, up) PRE-activation for the backward
// megakernel (Zq [rowcap, 4096] fp8 + DQZ [rowcap, 32] f32, g-half cols
// [0,2048), u-half [2048,4096) — phase-1b's read layout).  Default 0 keeps
// this file's .text-identity claim intact; when enabled the includer must
// also define N2GM_P1_SAVEZ_PTR / N2GM_P1_SAVEZ_DQ_PTR.
#ifndef N2GM_P1_SAVE_Z
#define N2GM_P1_SAVE_Z 0
#endif
static_assert(N2GM_P1_ASCALE_TOKEN_MAJOR == 0 ||
                  N2GM_P1_ASCALE_TOKEN_MAJOR == 1,
              "N2GM_P1_ASCALE_TOKEN_MAJOR is a bool: 0 = donor group-major");

__device__ __forceinline__ void n2_completion_observation_probe() {
#if N2_FORCE_VMCNT0
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}

// ===========================================================================
// PHASE 1: W13 -> register SiLU -> dynamic amax -> FP8 quant -> GLOBAL A2 handoff
// ===========================================================================
// N2_KERNELS_ONLY includers (k0pf6_mega) redefine the qualifier/name to call the
// same body as a device function; the defaults preserve the standalone kernel.
#ifndef N2_P1_QUAL
#define N2_P1_QUAL __global__ __launch_bounds__(kThreads, 1)
#define N2_P1_NAME n2_phase1_kernel
#endif
#ifndef N2_HOOK_CTX_ARG
#define N2_HOOK_CTX_ARG
#endif
// MPS-DELTA (2): four sub-phase instrumentation hooks, EMPTY by default, so an
// includer that defines none of them gets the donor body. They exist so a
// timestamp experiment can split (task head | prologue | K-loop | epilogue)
// without re-vendoring this file. All four sites are outside both MFMA spans.
// Definitions carry their own trailing semicolon (the N2GM_TASK_DONE_HOOK
// convention); the includer is expected to #undef them after the #include.
#ifndef N2GM_P1_TASK_HEAD_HOOK
#define N2GM_P1_TASK_HEAD_HOOK
#endif
#ifndef N2GM_P1_KLOOP_ENTER_HOOK
#define N2GM_P1_KLOOP_ENTER_HOOK
#endif
#ifndef N2GM_P1_KLOOP_EXIT_HOOK
#define N2GM_P1_KLOOP_EXIT_HOOK
#endif
#ifndef N2GM_P1_EPILOGUE_DONE_HOOK
#define N2GM_P1_EPILOGUE_DONE_HOOK
#endif
// MPS-DELTA (3): task-loop start/stride, macro-parameterized exactly the way
// phase 2 already is (n2_phase2_gm_mps.cpp:286-294). The defaults reproduce the
// donor (physical bid, full grid stride), so an includer that overrides nothing
// gets the donor loop. Added for exp_25's M6/M7 interleave and for the F1
// measurement T6(128)/T6(256) -- M6's CTA scaling, which no experiment has ever
// varied (m6_m7_structure.md §0, §7 item 2). Inert at the defaults: nothing in
// this file reads them outside the loop header.
#ifndef N2GM_P1_TASK_START
#define N2GM_P1_TASK_START blockIdx.x
#endif
#ifndef N2GM_P1_TASK_STRIDE
#define N2GM_P1_TASK_STRIDE kCTAs
#endif
N2_P1_QUAL void N2_P1_NAME(
    const std::uint8_t* __restrict__ A_bytes,   // input  [R, 7168] FP8
    const float* __restrict__ A_scale,          // input_scale, group-major live
    const std::uint8_t* __restrict__ W13,       // gate   [32, 4096, 7168] AITER
    const float* __restrict__ S13,              // fc1_scale [32, 32, 56]
    const int* __restrict__ sorted_ids,         // packed: token|slot<<24
    const int* __restrict__ sorted_eid,
    const int* __restrict__ nvi,                // [padded_rows, T]
    std::uint8_t* __restrict__ A2q,             // OUT [rowcap, 2048] FP8
    float* __restrict__ DQ2,                    // OUT [rowcap, 16] FP32
    const int* __restrict__ tile_desc,          // exp_59: [num_tiles] (b0<<4)|gcount
    int num_tiles                               // exp_59: expert-aligned tile count
    N2_HOOK_CTX_ARG) {
  // exp_59 G-stacking: a task now spans kGM consecutive SAME-expert 32-blocks
  // (effective M = 32*kGM). Row-indexed LDS grows by kGM; per-expert arrays
  // (b1s_lds) do NOT — every sub-block of a task shares expert e. kGM==1 is
  // textually identical to the donor n2_phase1.cpp.
#ifndef N2GM_G
#define N2GM_G 1
#endif
  constexpr int kGM = N2GM_G;
  constexpr int kMrows = kBlockM * kGM;            // 32*G sorted rows per task

  __shared__ int tok_lds[kMrows];
  __shared__ float ascale_lds[kKGroups][kMrows];   // [k128][sorted row]
  __shared__ float b1s_lds[2][2][kKGroups];        // [gate/up][n128 half][k128]
  __shared__ float amax_lds[2][kMrows];            // [kb][sorted row]
  __shared__ __align__(16) std::uint8_t a2_lds[kMrows][kA2Stride];

  // The CTA-shared, double-buffered A tile (N1e), now kGM sub-blocks wide.
  __shared__ __align__(16)
      std::uint8_t a_lds[2][kMrows][kAChunks][kAChunkBytes];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 63;
  const int wv = tid >> 6;   // wave 0..3
  const int q = lane >> 4;   // 0..3
  const int r = lane & 15;   // 0..15
  const int T = nvi[1];      // live token count -- read ON DEVICE

  // exp_59: work count is expert-aligned tiles * 8 intermediate chunks. tile_desc
  // packs (b0<<4)|gcount, gcount in [1,kGM]. At kGM==1, num_tiles == padded/32 and
  // this equals the donor's (padded/32)*8.
  constexpr int kNChunksP1 = 8;
  const int num_tasks = num_tiles * kNChunksP1;
  auto n2gm_tile_b0 = [&](int t) __attribute__((always_inline)) {
    return tile_desc[t] >> 4;
  };
  auto n2gm_tile_gcount = [&](int t, int, int) __attribute__((always_inline)) {
    return tile_desc[t] & 0xF;
  };

  // THE BOUNDED INPUT DESCRIPTOR (§L12, UNCHANGED).
  const std::uint32_t a_num_records =
      static_cast<std::uint32_t>(T) * static_cast<std::uint32_t>(kHidden);
  const i32x4 a_rsrc = make_srsrc(A_bytes, a_num_records, 0);

  const int a_row = tid >> 3;              // 0..31
  const int a_chunk = tid & 7;             // 0..7
  const int a_cp = a_chunk ^ (a_row & 7);  // the XOR swizzle

  const int j_w = wv >> 1;

  // MPS-DELTA (4): loop bounds come from MPS-DELTA (3)'s macros. At the
  // defaults this is the donor's
  //   for (int task = blockIdx.x; task < num_tasks; task += kCTAs)
  // with two no-op casts -- phase 2's convention, present so an override that
  // hands back a non-int (a descriptor field, a role-partition id) still
  // compiles. Verified .text-byte-identical to the donor build; build.md §8.
  for (int task = (int)(N2GM_P1_TASK_START); task < num_tasks;
       task += (int)(N2GM_P1_TASK_STRIDE)) {
    // exp_59: a task is (tile T, chunk g). tile_b0[T] is the tile's first 32-block
    // (all sub-blocks share expert e); gcount = live 32-blocks in this tile
    // (< kGM only in an expert's ragged tail). kGM==1 reduces to the donor with
    // tile == 32-block, gcount == 1.
    // MPS-DELTA (5): task-head hook. Nothing executes between the loop's `{`
    // and here, so this is the statement top of the task loop.
    N2GM_P1_TASK_HEAD_HOOK
    const int tile = task / kNChunksP1;
    const int g = task - tile * kNChunksP1;   // intermediate chunk 0..7
    const int b0 = n2gm_tile_b0(tile);        // first 32-block of the tile
    const int e = sorted_eid[b0];
    const int gcount = n2gm_tile_gcount(tile, b0, e);   // live sub-blocks (1..kGM)
#if K0P6_M20_SLOTPOOL
    // M20 slot pool: replica slots (e >= 32 owned experts) address the
    // parity-resolved pool bases baked into this layer's descriptor.  Tile
    // grouping above keeps the raw slot id; only weight indexing remaps.
    const bool m20_pooled = e >= 32;
    const std::uint8_t* const W13e =
        m20_pooled ? (const std::uint8_t*)k0p6_desc[K0P6_D_M20_W13P] : W13;
    const float* const S13e =
        m20_pooled ? (const float*)k0p6_desc[K0P6_D_M20_S13P] : S13;
    const int ew = m20_pooled ? (e - 32) : e;
#else
    const std::uint8_t* const W13e = W13;
    const float* const S13e = S13;
    const int ew = e;
#endif

    // ---- per-task LDS setup: tok/ascale/amax over kMrows (per-row); b1s per e --
    for (int i = tid; i < kMrows; i += kThreads) {
      const int lb32 = i >> 5;   // sub-block index 0..kGM-1
      tok_lds[i] = (lb32 < gcount)
                       ? (sorted_ids[(b0 + lb32) * kBlockM + (i & 31)] & 0x00FFFFFF)
                       : T;   // pad sub-block rows -> sentinel (>= T, never live)
    }
    for (int i = tid; i < 2 * kMrows; i += kThreads) {
      amax_lds[i / kMrows][i % kMrows] = 0.0f;
    }
    if (tid < 2 * 2 * kKGroups) {
      const int k = tid % kKGroups;
      const int j = (tid / kKGroups) & 1;
      const int gu = tid / (2 * kKGroups);
      const int ng = (gu == 0) ? (2 * g + j) : (kInter / 128 + 2 * g + j);
      b1s_lds[gu][j][k] = S13e[(ew * kNGroups13 + ng) * kKGroups + k];
    }
    __syncthreads();

#if N2GM_P1_ASCALE_TOKEN_MAJOR
    // MPS-DELTA (9) arm 1. input_scale is TOKEN-MAJOR: input_scale[token * 56 + k].
    // One row is kKGroups * 4 = 224 B contiguous, so a row spans exactly 4 lines
    // (224*token mod 64 is 0 or 32, never 5 lines): 96 rows = 384 distinct lines
    // against the group-major path's 96 * 56 = 5,376.
    //
    // float4, not float: the byte offset is 224*token + 16*c and 224 = 14*16, so
    // 16 B alignment holds for every (token, c). Cuts 5,376 scalar loads to 1,344
    // vector loads over the same 384 lines, and the trip count from 21 to 6.
    //
    // Row-fast (i from idx % kMrows), not quad-fast: consecutive lanes then write
    // consecutive ascale_lds dwords, which is bank-conflict-free. Quad-fast would
    // merge the line requests but stride the LDS writes by 4*kMrows dwords -- a
    // multiple of 32 -- putting every lane of a quad in one bank.
    static_assert(kKGroups % 4 == 0, "float4 scale quads need kKGroups % 4 == 0");
    constexpr int kScaleQuads = kKGroups / 4;   // 14, no tail
    for (int idx = tid; idx < kScaleQuads * kMrows; idx += kThreads) {
      const int i = idx % kMrows;
      const int c = idx / kMrows;
      const int token = tok_lds[i];
      // Same predicate as the donor, so pad sub-block rows still get exact +0.0f.
      // Note T is no longer part of any ADDRESS in this arm, only of the mask.
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (token < T) {
        v = *reinterpret_cast<const float4*>(
            A_scale + static_cast<std::size_t>(token) * kKGroups + 4 * c);
      }
      ascale_lds[4 * c + 0][i] = v.x;
      ascale_lds[4 * c + 1][i] = v.y;
      ascale_lds[4 * c + 2][i] = v.z;
      ascale_lds[4 * c + 3][i] = v.w;
    }
#else
    // input_scale live prefix is GROUP-MAJOR: input_scale[k128 * T + token].
    for (int idx = tid; idx < kKGroups * kMrows; idx += kThreads) {
      const int k = idx / kMrows;
      const int i = idx % kMrows;
      const int token = tok_lds[i];
      ascale_lds[k][i] = (token < T)
                             ? A_scale[static_cast<std::size_t>(k) * T + token]
                             : 0.0f;
    }
#endif
    __syncthreads();

    // Per-sub-block A base offset (a_row is the row WITHIN a 32-block).
    std::uint32_t a_voff0[kGM];
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
      a_voff0[sb] =
          static_cast<std::uint32_t>(tok_lds[sb * kBlockM + a_row]) *
              static_cast<std::uint32_t>(kHidden) +
          static_cast<std::uint32_t>(a_chunk * kAChunkBytes);
    }

    // =======================================================================
    // GEMM-1 : PAIRED gate/up, G-stacked. Weights (bf) loaded ONCE per K-step
    // and reused across all kGM sub-blocks — the L2-reuse win. Accumulators and
    // A-fragments gain the sub-block dimension — the register cost (+16*G racc).
    // =======================================================================
    racc acc[kGM][2][2][kColTiles];
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
#pragma unroll
      for (int gu = 0; gu < 2; ++gu) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
          for (int c = 0; c < kColTiles; ++c) {
            zero(acc[sb][gu][m][c]);
          }
        }
      }
    }

    const int n_gate = kChunkCols * g + kWaveCols * wv;
    const int n_up = kInter + n_gate;

    rfp8 af[kGM][2];
    __uint128_t aTmp[kGM][2];
    rfp8 bf[2][2][kColTiles];

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
      for (int c = 0; c < kColTiles; ++c) {
        const std::size_t tg = aiter::fp8_weight_byte_offset(
            ew, n_gate + 16 * c, kbyte, kW13N, kHidden);
        const std::size_t tu = aiter::fp8_weight_byte_offset(
            ew, n_up + 16 * c, kbyte, kW13N, kHidden);
        load_bfrag(bf[buf][0][c], W13e + tg + 16 * lane);
        load_bfrag(bf[buf][1][c], W13e + tu + 16 * lane);
        stage_agpr(bf[buf][0][c]);
        stage_agpr(bf[buf][1][c]);
      }
    };

    auto mfma_k = [&](int k, int buf) __attribute__((always_inline)) {
      const float bsg = b1s_lds[0][j_w][k];
      const float bsu = b1s_lds[1][j_w][k];
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
        const float4 as0 =
            *reinterpret_cast<const float4*>(&ascale_lds[k][sb * kBlockM + 4 * q]);
        const float4 as1 = *reinterpret_cast<const float4*>(
            &ascale_lds[k][sb * kBlockM + 16 + 4 * q]);
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
            for (int c = 0; c < kColTiles; ++c) {
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

    // ---- the pipeline.  UNCHANGED from n1g.
    load_a(0, 0);
    load_a(1, 1);
    load_b(0, 0);
    store_a(0, 0);
    lds_cta_barrier();
    // MPS-DELTA (6): K-loop entry hook — after the pipeline-prologue barrier,
    // before the first MFMA. Outside the K-loop.
    N2GM_P1_KLOOP_ENTER_HOOK

#pragma unroll 1
    for (int k = 0; k < kKGroups; k += 2) {
      const int ka2 = (k + 2 < kKGroups) ? (k + 2) : (kKGroups - 1);
      const int ka3 = (k + 3 < kKGroups) ? (k + 3) : (kKGroups - 1);
      const int kb1 = (k + 1 < kKGroups) ? (k + 1) : (kKGroups - 1);
      const int kb2 = (k + 2 < kKGroups) ? (k + 2) : (kKGroups - 1);

      // ---- half 0: step k.   LDS buf 0, B buf 0. -------------------------
      read_a(0);
      load_a(ka2, 0);
      load_b(kb1, 1);
      mfma_k(k, 0);
      store_a(1, 1);
      // MPS-DELTA (7): the donor's literals here are hardcoded for kGM==1. At
      // the shipped kGM==3 the real per-half mix is 12 DS-read / 19 VMEM-read /
      // 48 MFMA / 3 DS-write, so 4/17/16/1 leaves two thirds of the DS reads,
      // two thirds of the MFMA and two thirds of the DS writes outside every
      // hinted group. The four counts now come from MPS-DELTA (1)'s mask; at
      // mask 0 they expand to the donor's 4 / 17 / 16 / 1.
      __builtin_amdgcn_sched_group_barrier(0x100, N2GM_P1_N_DSREAD, 0);   // DS read
      __builtin_amdgcn_sched_group_barrier(0x020, N2GM_P1_N_VMEMRD, 0);   // VMEM read
      __builtin_amdgcn_sched_group_barrier(0x008, N2GM_P1_N_MFMA, 0);     // MFMA
      __builtin_amdgcn_sched_group_barrier(0x200, N2GM_P1_N_DSWRITE, 0);  // DS write
      n2_completion_observation_probe();
      lds_cta_barrier();

      // ---- half 1: step k+1.  LDS buf 1, B buf 1. ------------------------
      read_a(1);
      load_a(ka3, 1);
      load_b(kb2, 0);
      mfma_k(kb1, 1);
      store_a(0, 0);
      // MPS-DELTA (8): half 1 — census verified identical to half 0 (one
      // read_a / load_a / load_b / mfma_k / store_a each, same arguments in
      // shape, only the LDS and B buffer indices differ), so it takes the same
      // four counts from MPS-DELTA (1)'s mask.
      __builtin_amdgcn_sched_group_barrier(0x100, N2GM_P1_N_DSREAD, 0);   // DS read
      __builtin_amdgcn_sched_group_barrier(0x020, N2GM_P1_N_VMEMRD, 0);   // VMEM read
      __builtin_amdgcn_sched_group_barrier(0x008, N2GM_P1_N_MFMA, 0);     // MFMA
      __builtin_amdgcn_sched_group_barrier(0x200, N2GM_P1_N_DSWRITE, 0);  // DS write
      n2_completion_observation_probe();
      lds_cta_barrier();
    }
    // MPS-DELTA (9): K-loop exit hook — the last MFMA of the task has issued.
    // Outside the K-loop.
    N2GM_P1_KLOOP_EXIT_HOOK

    // =======================================================================
    // REGISTER SiLU + DYNAMIC per-(row, K128) amax + FP8 quant — per sub-block.
    // amax_lds row index is the ABSOLUTE row within the tile (sb*32 + 16m+4q+t).
    // =======================================================================
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
      const int rbase = sb * kBlockM;
      bool live[2][4];
      float a2v[2][kColTiles][4];
      float lmax[2][4];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          live[m][t] = tok_lds[rbase + 16 * m + 4 * q + t] < T;
          lmax[m][t] = 0.0f;
        }
      }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int c = 0; c < kColTiles; ++c) {
          const float* gf = accf(acc[sb][0][m][c]);
          const float* uf = accf(acc[sb][1][m][c]);
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const float gv = gf[t];
            const float hv = (gv / (1.0f + __expf(-gv))) * uf[t];
            const float h = live[m][t] ? hv : 0.0f;
            a2v[m][c][t] = h;
            lmax[m][t] = fmaxf(lmax[m][t], fabsf(h));
          }
        }
      }
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

      // THE EPSILON IS NOT OPTIONAL (rank 2 has empty experts; 448/0 = NaN).
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        float dq[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          dq[t] = fmaxf(amax_lds[j_w][rbase + 16 * m + 4 * q + t], 1.0e-6f) /
                  448.0f;
        }
#pragma unroll
        for (int c = 0; c < kColTiles; ++c) {
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const float qv =
                fminf(fmaxf(a2v[m][c][t] / dq[t], -448.0f), 448.0f);
            a2_lds[rbase + 16 * m + 4 * q + t][kWaveCols * wv + 16 * c + r] =
                __hip_cvt_float_to_fp8(qv, __HIP_SATFINITE, __HIP_E4M3);
          }
        }
      }
      __syncthreads();
    }

    // =======================================================================
    // A2 -> GLOBAL, coalesced. Each of the 256 threads writes 32 B of one
    // 32-block's tile; loop kGM sub-blocks. Pad sub-blocks (sb>=gcount) are NOT
    // written — phase2 never reads them and A2q rows beyond the live region are
    // out of every bounded descriptor.
    // =======================================================================
    {
      const int cr = tid >> 3;   // row within a 32-block 0..31
      const int cc = tid & 7;    // 32-B chunk 0..7
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
        if (sb >= gcount) break;
        const uint4 s0 = *reinterpret_cast<const uint4*>(
            &a2_lds[sb * kBlockM + cr][32 * cc]);
        const uint4 s1 = *reinterpret_cast<const uint4*>(
            &a2_lds[sb * kBlockM + cr][32 * cc + 16]);
        const std::size_t dst =
            (static_cast<std::size_t>(b0 + sb) * kBlockM + cr) * kInter +
            kChunkCols * g + 32 * cc;
        *reinterpret_cast<uint4*>(A2q + dst) = s0;
        *reinterpret_cast<uint4*>(A2q + dst + 16) = s1;
      }
    }
    // DQ2[row, 2g+kb] — one lane per row writes (q==0), per live sub-block.
    if (q == 0) {
#pragma unroll
      for (int sb = 0; sb < kGM; ++sb) {
        if (sb >= gcount) break;
        const int rbase = sb * kBlockM;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
          for (int kb = 0; kb < 2; ++kb) {
            DQ2[(static_cast<std::size_t>(b0 + sb) * kBlockM + 16 * m + r) *
                    kKGroups2 +
                2 * g + kb] =
                fmaxf(amax_lds[kb][rbase + 16 * m + r], 1.0e-6f) / 448.0f;
          }
        }
      }
    }
#if N2GM_P1_SAVE_Z
    // T2B-DELTA: persist z per (row, K128) with the epilogue's own amax
    // machinery.  acc[sb][gu] is still live; a2_lds is reused sequentially
    // per half (the barrier at each round's head orders the A2q store's LDS
    // reads before the overwrite).  Same quant expression, same live-select,
    // same coalesced store shape as the A2q path.
    for (int gu = 0; gu < 2; ++gu) {
      for (int i = tid; i < 2 * kMrows; i += kThreads) {
        amax_lds[i / kMrows][i % kMrows] = 0.0f;
      }
      __syncthreads();
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
          for (int c = 0; c < kColTiles; ++c) {
            const float* zf = accf(acc[sb][gu][m][c]);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
              const bool lv = tok_lds[rbase + 16 * m + 4 * q + t] < T;
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
                            &amax_lds[j_w][rbase + 16 * m + 4 * q + t]),
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
            dqz[t] =
                fmaxf(amax_lds[j_w][rbase + 16 * m + 4 * q + t], 1.0e-6f) /
                448.0f;
          }
#pragma unroll
          for (int c = 0; c < kColTiles; ++c) {
            const float* zf = accf(acc[sb][gu][m][c]);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
              const bool lv = tok_lds[rbase + 16 * m + 4 * q + t] < T;
              const float zv = lv ? zf[t] : 0.0f;
              const float qv = fminf(fmaxf(zv / dqz[t], -448.0f), 448.0f);
              a2_lds[rbase + 16 * m + 4 * q + t][kWaveCols * wv + 16 * c + r] =
                  __hip_cvt_float_to_fp8(qv, __HIP_SATFINITE, __HIP_E4M3);
            }
          }
        }
      }
      __syncthreads();
      {
        std::uint8_t* const zq_out = N2GM_P1_SAVEZ_PTR;
        float* const dqz_out = N2GM_P1_SAVEZ_DQ_PTR;
        const int cr = tid >> 3;
        const int cc = tid & 7;
#pragma unroll
        for (int sb = 0; sb < kGM; ++sb) {
          if (sb >= gcount) break;
          const uint4 s0 = *reinterpret_cast<const uint4*>(
              &a2_lds[sb * kBlockM + cr][32 * cc]);
          const uint4 s1 = *reinterpret_cast<const uint4*>(
              &a2_lds[sb * kBlockM + cr][32 * cc + 16]);
          const std::size_t dst =
              (static_cast<std::size_t>(b0 + sb) * kBlockM + cr) *
                  (2 * static_cast<std::size_t>(kInter)) +
              static_cast<std::size_t>(gu) * kInter + kChunkCols * g + 32 * cc;
          *reinterpret_cast<uint4*>(zq_out + dst) = s0;
          *reinterpret_cast<uint4*>(zq_out + dst + 16) = s1;
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
                dqz_out[(static_cast<std::size_t>(b0 + sb) * kBlockM + 16 * m +
                         r) *
                            (2 * kKGroups2) +
                        gu * kKGroups2 + 2 * g + kb] =
                    fmaxf(amax_lds[kb][rbase + 16 * m + r], 1.0e-6f) / 448.0f;
              }
            }
          }
        }
      }
      __syncthreads();
    }
#endif
    // MPS-DELTA (10): epilogue-done hook — after the A2q/DQ2 stores, before the
    // task-end __syncthreads(). NOT a VMEM drain: the stores above are still in
    // flight here, exactly as in the donor.
    N2GM_P1_EPILOGUE_DONE_HOOK
    // The next task refills every LDS buffer this one just read.
    __syncthreads();
    // exp_59: a2 arrival fires per LIVE sub-block only (a2_done indexed by the
    // absolute 32-block b0+sb). Expert-aligned tiling gives exact 1:1 tile
    // ownership of every 32-block, so each a2_done[b] is incremented exactly 8
    // times by its owning tile's 8 chunk-tasks — NEVER by a neighbouring tile's
    // pad sub-block. Firing for sb>=gcount would double-count and break the >=8
    // gate (an exp_56-class defect); do not.
#ifdef N2GM_TASK_DONE_HOOK
#pragma unroll
    for (int sb = 0; sb < kGM; ++sb) {
      if (sb >= gcount) break;
      const int b = b0 + sb;
      N2GM_TASK_DONE_HOOK
    }
#endif
  }
}

}  // namespace production_fused_moe::n2

#ifndef N2_KERNELS_ONLY
namespace production_fused_moe::n2 {
// ---------------------------------------------------------------------------
// Host side.  Production tensors + the two scratch handoff buffers (allocated
// once at setup by the caller; no host work inside the measured region).
// input capacity is a RUNTIME property (device-driven walk) — only column
// counts are contract-checked, so the same module serves decode and prefill.
// ---------------------------------------------------------------------------
struct n2_phase1_globals {
  tensor_bf16 input_bytes;        // BF16 view of production `input`
  tensor_fp32 input_scale;        // production `input_scale`
  tensor_bf16 w13_bytes;          // BF16 view of production `gate` (AITER)
  tensor_fp32 w13_scales;         // production `fc1_scale`
  tensor_i32 sorted_token_ids;
  tensor_i32 sorted_expert_ids;
  tensor_i32 num_valid_ids;
  tensor_bf16 a2q_bytes;          // scratch [rowcap, 1024] BF16-view (=2048 B rows)
  tensor_fp32 dq2;                // scratch [rowcap, 16] FP32
  std::uintptr_t stream_ptr;
};

[[noreturn]] static void invalid(const char* message) {
  throw std::invalid_argument(std::string("n2_phase1: ") + message);
}

static void require(bool condition, const char* message) {
  if (!condition) {
    invalid(message);
  }
}

void validate_contract(const n2_phase1_globals& g) {
  require(aiter::is_supported_blockscale_shape(kExperts, kW13N, kHidden),
          "W13 AITER shape unsupported");
  require(g.input_bytes.batch() == 1 && g.input_bytes.depth() == 1 &&
              g.input_bytes.cols() == kHidden / 2,
          "input must have BF16-view cols 3584");
  require(g.input_scale.batch() == 1 && g.input_scale.depth() == 1 &&
              g.input_scale.cols() == kKGroups,
          "input_scale must have FP32 cols 56");
  require(g.w13_bytes.batch() == 1 && g.w13_bytes.depth() == 1 &&
              g.w13_bytes.rows() == kExperts * kW13N &&
              g.w13_bytes.cols() == kHidden / 2,
          "gate must have BF16-view shape [32*4096,3584]");
  require(g.w13_scales.batch() == 1 && g.w13_scales.depth() == kExperts &&
              g.w13_scales.rows() == kNGroups13 &&
              g.w13_scales.cols() == kKGroups,
          "fc1_scale must have FP32 shape [32,32,56]");
  require(g.num_valid_ids.cols() == 2, "num_valid_ids must hold two values");
  require(g.a2q_bytes.cols() == kInter / 2,
          "a2q must have BF16-view cols 1024 (=2048 B rows)");
  require(g.dq2.cols() == kKGroups2, "dq2 must have FP32 cols 16");
  require(g.sorted_token_ids.cols() / kBlockM <= g.sorted_expert_ids.cols(),
          "sorted_token_ids capacity implies more blocks than sorted_expert_ids "
          "can name");
  require(static_cast<std::size_t>(g.a2q_bytes.rows()) >=
              static_cast<std::size_t>(g.sorted_token_ids.cols()),
          "a2q must cover the sorted row capacity");
}

void dispatch_n2_phase1(n2_phase1_globals g) {
  validate_contract(g);
  hipStream_t stream = reinterpret_cast<hipStream_t>(g.stream_ptr);
  n2_phase1_kernel<<<kCTAs, kThreads, 0, stream>>>(
      reinterpret_cast<const std::uint8_t*>(g.input_bytes.raw_ptr),
      g.input_scale.raw_ptr,
      reinterpret_cast<const std::uint8_t*>(g.w13_bytes.raw_ptr),
      g.w13_scales.raw_ptr, g.sorted_token_ids.raw_ptr,
      g.sorted_expert_ids.raw_ptr, g.num_valid_ids.raw_ptr,
      reinterpret_cast<std::uint8_t*>(g.a2q_bytes.raw_ptr),
      g.dq2.raw_ptr);
}

}  // namespace production_fused_moe::n2

void bind_n2_phase1(pybind11::module_& module) {
  using production_fused_moe::n2::dispatch_n2_phase1;
  using production_fused_moe::n2::n2_phase1_globals;
  py::bind_function<dispatch_n2_phase1>(
      module, "n2_phase1",
      &n2_phase1_globals::input_bytes,
      &n2_phase1_globals::input_scale,
      &n2_phase1_globals::w13_bytes,
      &n2_phase1_globals::w13_scales,
      &n2_phase1_globals::sorted_token_ids,
      &n2_phase1_globals::sorted_expert_ids,
      &n2_phase1_globals::num_valid_ids,
      &n2_phase1_globals::a2q_bytes,
      &n2_phase1_globals::dq2,
      &n2_phase1_globals::stream_ptr);
}
#endif  // N2_KERNELS_ONLY

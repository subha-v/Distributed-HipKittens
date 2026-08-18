// ================================================================================================
// T2V6 — the backward megakernel with WGRAD-IN-BUBBLES (the v6 filler design).
// Additive sibling of k0pf6gm_device_tile_t2b.hip (which stays untouched, the
// same way t2b was an additive sibling of the m15 forward).  Design record:
// the WGRAD LADDER (v1 host fp32 -> v1.5 host bf16 -> v2/M8.5 in-kernel serial
// -> v2.2 256x256 AGPR tiles -> v3b bmm side-stream -> v4 serial e2e 1,912 ms
// -> v5 sliced side-stream WEDGE) proved (a) stream co-scheduling against an
// all-CU grid-barrier kernel is impossible on gfx950 (rocBLAS grids yield,
// ours don't: 3-way residency standoff), so wgrad must live INSIDE the one
// kernel; and (b) serial in-kernel placement puts it on the critical path.
// v6 therefore executes wgrad tiles inside the kernel's own WAIT WINDOWS:
//
//   window A: service-CTA slab 0 of M7   (today: no code path at all — the
//             pool falls straight through to the slab rendezvous and spins)
//   window B: service-CTA slab 1 after the flush_rows combine quota
//   window C: the M8 slab-word wait, all 256 CTAs (poll-and-fill: one
//             nonblocking sweep of the 16 certificates, then one tile, loop;
//             the verbatim bounded poll remains as the fail-closed fallback)
//   tail    : M8.5 becomes a cursor DRAIN of whatever the windows left over
//             (mode 1 keeps the v4 grid-stride whole-space behavior).
//
// READINESS LATTICE (why two cursors): dW2 = (swt·dY)^T·act(z) reads only
// M2-certified dY rows + the z-regen output behind its grid barrier + the
// static saved plan — legal from slab 0 on.  dW13 = (swt·dZ)^T·X reads dZq,
// a phase-1b output with NO terminating barrier (CTAs flow straight into M7;
// only a2_done certifies per-block) — globally final only after the slab-0
// rendezvous, so dW13 tiles are claimable from slab 1 on.  Cursor A walks the
// dW2 tile space, cursor B the dW13 space; the tail drains A then B.  Every
// tile is claimed exactly once per launch (fetch_add), preserving the wgrad
// body's one-CTA-per-tile store contract (overwrite or bf16-RMW accumulate).
//
// QUOTA DISCIPLINE (the R1-hostage rule, same law as the combine sweep): all
// window claims are quota-bounded so no filler tile can delay a rendezvous
// beyond one tile's tail.  Quotas ride the DWMODE descriptor word:
//   bits [3:0]  mode: 0 = off (host wgrad), 1 = v4-compat post-M8 grid-stride
//               overwrite, 2 = filler (cursors + windows + tail)
//   bits [7:4]  nmb (microbatches/iteration): accumulate is DERIVED in-kernel
//               as (epoch-1) mod nmb != 0 — the first microbatch overwrites,
//               the rest bf16-RMW accumulate, and the host patches NOTHING
//               per launch (it adds arenas into main_grad once per layer per
//               iteration); nmb <= 1 never accumulates (gate harnesses)
//   bits [15:8]  window-A quota (tiles per service CTA, slab 0)
//   bits [23:16] window-B quota (tiles per service CTA, slab 1 post-sweep)
//   bits [31:24] window-C quota (tiles per CTA inside the M8 wait)
// A zero quota disables that window at runtime — one .co carries the sweep.
//
// PROF ARM (K0P6_T2V6_PROF): the bubble ledger that sizes all of this.  New
// descriptor slot 73 = u64[32] arena, MONOTONIC (atomicAdd/atomicMax only,
// never kernel-zeroed) so the async host can read+diff at any cadence.
// Realtime clock (s_memrealtime, 100 MHz) bracketed OUTSIDE the poll bodies:
// zero interference with the protocol.  Compiled out by default; the default
// build's wait paths are the donor's byte-for-byte.
//
// exp_38 discipline: dead donor-parity arms (M17/M18/M19/M20/staged) are
// STRIPPED from this sibling, not carried compiled-out; one textual copy of
// the slab rendezvous; every v6 addition is compile-gated so the FILLER=0
// PROF=0 build is the t2b control under this TU's own symbol.
// ================================================================================================

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>
#include <cstddef>
#include <cstdint>

#include "hkp_sort.hpp"
#include "hkp_sync.hpp"
#include "hkp_quant.hpp"
#include "k0pf5_ll128.hpp"
#include "k0pf6_chunk_release.hpp"
#include "moe_hk_adapter.cuh"
#include "moe_mps_adapter.cuh"
#include "k0pf6gm_t2b_abi.hpp"

// JIT-cache guard (own lineage; bump on any change to this TU).
#define K0P6_T2V6_SRC_REV 2

// ---- v6 compile arms ----------------------------------------------------------------------
// PROF: the bubble ledger (slot 73 required nonzero).  FILLER: the wgrad
// window scheduler.  Both default 0: the default build is the t2b control.
#ifndef K0P6_T2V6_PROF
#define K0P6_T2V6_PROF 0
#endif
#ifndef K0P6_T2V6_FILLER
#define K0P6_T2V6_FILLER 0
#endif

// This sibling is the FULL backward only (no transport-skeleton stage).
#define K0P6_T2B_PHASES 1

// Donor-parity arms do not exist in this TU (stripped, not compiled out);
// the forced-0 defines keep every shared header on the default path.
#define K0P6_M15_STAGED 0
#define K0P6_M15_SCATTER_RR 0
#define K0P6_M15_REPLICATE 0
#define K0P6_M15_ADAPTIVE 0
#define K0P6_M20_SLOTPOOL 0

// exp_27 shipped arm: M6 reads token-major sc_stage; M5's transpose is dead.
#ifndef K0P6_MPS_ASCALE_TM
#define K0P6_MPS_ASCALE_TM 1
#endif

#define K0P6C_NCHUNK_MAX 8

#define K0P6_MAXE 64
#define K0P6_NG 56
#define K0P6_H 7168
#define K0P6_SC_ERB 0
#define K0P6_SC_CNT 80
#define K0P6_SC_CUR 160

// ---- descriptor offsets (MIRRORED by the host driver; identical to t2b) -------------------
#define K0P6_D_MYIDS 0
#define K0P6_D_MYWG 1
#define K0P6_D_HIDDEN 2
#define K0P6_D_A_LL 3
#define K0P6_D_DESTCTR 4
#define K0P6_D_ROWS_DONE 5
#define K0P6_D_PUSHED 6
#define K0P6_D_QPUSH_DONE 7
#define K0P6_D_A_DST 8
#define K0P6_D_SC_STAGE 9
#define K0P6_D_RECV_EID 10
#define K0P6_D_RECV_WGT 11
#define K0P6_D_PULL_STAGE 12
#define K0P6_D_PULL_CNT 13
#define K0P6_D_STI 14
#define K0P6_D_SWT 15
#define K0P6_D_SEI 16
#define K0P6_D_NVI 17
#define K0P6_D_PULL_PTR 18
#define K0P6_D_PULL_SRC 19
#define K0P6_D_SC_DST 20
#define K0P6_D_PART 21
#define K0P6_D_SCRATCH 22
#define K0P6_D_GBAR 23
#define K0P6_D_HCNT 24
#define K0P6_D_ROW_READY 25
#define K0P6_D_ROW_REM 26
#define K0P6_D_A2Q 27
#define K0P6_D_DQ2 28
#define K0P6_D_A2_DONE 29
#define K0P6_D_PART_DONE 30
#define K0P6_D_RETIRED 31
#define K0P6_D_OUT 32
#define K0P6_D_COMB_DONE 33
#define K0P6_D_EPOCH_CELL 34
#define K0P6_D_W13 35
#define K0P6_D_S13 36
#define K0P6_D_W2 37
#define K0P6_D_S2 38
#define K0P6_D_CUR 39
#define K0P6_D_WORLD 40
#define K0P6_D_SPIN 41
#define K0P6_D_PPERR 42
#define K0P6_D_MAXB 43
#define K0P6_D_T 44
#define K0P6_D_TOPK 45
#define K0P6_D_E 46
#define K0P6_D_TLOCMAX 47
#define K0P6_D_PADMAX 48
#define K0P6_D_DEBUG 49
#define K0P6_D_CHUNK_READY 50
#define K0P6_D_MAXTOK 51
#define K0P6_D_SPIN_DBG 52
#define K0P6_D_TILE_DESC 53
#define K0P6_D_NUM_TILES 54
#define K0P6_D_SYMMETRIC 55
// MPS slots 56..62 come from moe_mps_adapter.cuh; t2b slots 63..72 from the
// ABI header.  v6 extends by one:
#define K0P6_D_T2V6_PROF 73   // u64[32] monotonic bubble ledger (PROF arm)
#define K0P6_T2V6_D_LEN 74
#define K0P6_M15_D_LEN K0P6_MPS_D_LEN

// ---- prof arena layout (u64 indices; MONOTONIC, host reads+diffs) -------------------------
// Cycle cells are CTA-cycle SUMS at 100 MHz (10 ns ticks): wall estimate =
// diff / launches / participating-CTAs.  MIN cells store ~t via atomicMax
// (the ts_first convention); recover with ~cell.
#define T2V6_PB_RETIRE 0        // M0 retire poll (tid0's own source wait)
#define T2V6_PB_M0BAR 1         // M0 grid barrier
#define T2V6_PB_M2ROWS 2        // M2 rows_done spin (tid0)
#define T2V6_PB_M2CHUNK 3       // M2 chunk polls + grid barrier
#define T2V6_PB_M3BAR 4         // M3-position barrier (post dw zero)
#define T2V6_PB_ZBAR 5          // post-z-regen barrier
#define T2V6_PB_RDV0_C 6        // slab-0 rendezvous wait, compute CTAs
#define T2V6_PB_RDV0_S 7        // slab-0 rendezvous wait, service CTAs
#define T2V6_PB_RDV1_C 8        // slab-1 rendezvous wait, compute CTAs
#define T2V6_PB_RDV1_S 9        // slab-1 rendezvous wait, service CTAs
#define T2V6_PB_M8WAIT 10       // M8 slab-word wait (incl. window-C fill)
#define T2V6_PB_M8ENTER_MIN 11  // ~min over CTAs of M8 ticket-loop entry
#define T2V6_PB_M8EXIT_MAX 12   // max over CTAs of M8 ticket-loop exit
#define T2V6_PB_ENTER_MIN 13    // ~min over CTAs of post-guard kernel entry
#define T2V6_PB_M9EXIT_MAX 14   // max over CTAs of M9 exit
#define T2V6_PB_LAUNCHES 15     // += 1 per launch (bid0/tid0)
#define T2V6_PB_FILL_A 16       // tiles run in window A (slab-0 service)
#define T2V6_PB_FILL_B 17       // tiles run in window B (slab-1 service)
#define T2V6_PB_FILL_C 18       // tiles run in window C (M8 wait)
#define T2V6_PB_FILL_TAIL 19    // tiles run in the M8.5 tail drain
#define T2V6_PB_FILL_CYC 20     // CTA-cycles spent inside filler tiles
#define T2V6_PB_SVC_CTAS 21     // += C per launch
#define T2V6_PB_WORDS 32

#if K0P6_T2V6_PROF
#define T2V6_PROF_ONLY(x) x
#else
#define T2V6_PROF_ONLY(x)
#endif

// Kernel symbol name, parametric like the siblings'.
#ifndef K0P6_T2V6_KERNEL_NAME
#define K0P6_T2V6_KERNEL_NAME k0pf6gm_t2v6_mega
#endif
#undef K0P6_M15_KERNEL_NAME
#define K0P6_M15_KERNEL_NAME K0P6_T2V6_KERNEL_NAME

// Slab count. S=2 is the only value where the 448-column nc chunks and M8's
// 1024-byte chunks co-align (column 3584 = nc 8 = M8 chunk 7); it is also the
// fence-law optimum (each extra boundary is an M7.5-class rendezvous).
#define K0P6_M15_SLABS 2
#define K0P6_M15_NC_PER_SLAB 8
#define K0P6_M15_M8_CSPLIT 7   // M8 chunk boundary: [0,7) front, [7,14) back

#ifndef K0P6GM_G
#define K0P6GM_G 3
#endif
#if K0P6GM_G != 3
#error "fused_moe is pinned to the measured K0P6GM_G=3 configuration"
#endif

using k0p6_sort = hkp::destination_counting_sort<K0P6_MAXE, 32, K0P6_SC_ERB, K0P6_SC_CNT,
                                                 K0P6_SC_CUR>;

__device__ __forceinline__ long long k0p6_dread(const long long* desc, int off) {
  return *reinterpret_cast<volatile const long long*>(desc + off);
}

__device__ __forceinline__ const hk_moe::symmetric_heap_descriptor*
k0p6_symmetric(const long long* desc) {
  return reinterpret_cast<const hk_moe::symmetric_heap_descriptor*>(
      k0p6_dread(desc, K0P6_D_SYMMETRIC));
}

__device__ __forceinline__ std::uint64_t k0p6_epoch(const long long* desc) {
  const unsigned int* epoch_cell =
      (const unsigned int*)k0p6_dread(desc, K0P6_D_EPOCH_CELL);
  return (std::uint64_t)
             hk_moe::load_relaxed<hk_moe::scope::agent>(epoch_cell) +
         1ull;
}

// ---- task hooks (donor forms, verbatim) ---------------------------------------------------
__device__ __forceinline__ void k0p6_a2_arrive(int b, int tid, const long long* desc) {
  if (tid == 0) {
    hk_moe::arrive_local_count<true>(
        ((unsigned int*)desc[K0P6_D_A2_DONE]) + b);
  }
}

__device__ __forceinline__ bool k0p6_a2_wait(int b, int tid, const long long* desc) {
  if (tid == 0) {
    const unsigned int* a2_done = (const unsigned int*)desc[K0P6_D_A2_DONE];
    const long long spin_limit = desc[K0P6_D_SPIN];
    int* pperr = (int*)desc[K0P6_D_PPERR];
    const bool ready = hk_moe::poll_local_count<8>(
        a2_done + b, (std::uint64_t)spin_limit,
        hk_moe::set_error_bit{pperr, 16777216});
    if (!ready) atomicOr(pperr, 16777216);
  }
  hk_moe::acquire_payload_agent();
  return !hk_moe::error_bit_set_agent(
      (const int*)desc[K0P6_D_PPERR], 16777216);
}

__device__ __forceinline__ int k0p6_mps_stride(const long long* desc, int nct) {
  return nct - (int)((unsigned long long)k0p6_dread(desc, K0P6_D_MPS_CFG) &
                     0xFFull);
}

__device__ __forceinline__ int k0p6_mps_task_start(const long long* desc) {
  return hk_moe::mps::compute_id_of(
      hk_moe::mps::decode_config(
          (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG)),
      (int)blockIdx.x);
}

// ---- M15 deferred-drain hooks: mode 12's drain schedule, ZERO publication ----
struct k0p6_defer {
  int b0;
  int gcount;
  int nc;
};

__device__ __forceinline__ void k0p6_m15_task_flush_defer(
    int tid, const long long* k0p6_desc, k0p6_defer& p) {
  if (p.b0 < 0) return;
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  p.b0 = -1;
}

__device__ __forceinline__ void k0p6_m15_task_done_defer(
    int b, int nc, int tid, const long long* k0p6_desc, k0p6_defer& p) {
  if (p.b0 < 0) {
    p.b0 = b;
    p.nc = nc;
  }
  p.gcount = 0;
}

// ---- N2 phase bodies ----------------------------------------------------------------------
#define N2GM_G K0P6GM_G
#define N2_P1Z_QUAL __device__ __forceinline__
#define N2_P1Z_NAME n2p6t2b_phase1z_body
#define N2_KERNELS_ONLY
#define N2_HOOK_CTX_ARG , const long long* __restrict__ k0p6_desc
#include "k0pf6gm_t2b_shared.hpp"
#include "n2_phase1z_gm_t2b.cpp"
#undef N2_HOOK_CTX_ARG
#undef N2_KERNELS_ONLY
#undef N2_P1Z_NAME
#undef N2_P1Z_QUAL

// K0P6_T2B_FASTCVT=1 (a BUILD flag of this TU's arms, never defaulted here)
// selects the hardware fp8->f32 dequant in the swiglu' epilogue; the ctl
// parity arm builds without it so the fastcvt delta is its own A/B.
#define N2_P1B_QUAL __device__ __forceinline__
#define N2_P1B_NAME n2p6t2b_phase1b_body
#define N2_KERNELS_ONLY
#define N2_HOOK_CTX_ARG , const long long* __restrict__ k0p6_desc
#define N2GM_P1B_ARRIVAL_HOOK k0p6_a2_arrive(b, tid, k0p6_desc);
#include "n2_phase1b_gm_t2b.cpp"
#undef N2GM_P1B_ARRIVAL_HOOK
#undef N2_HOOK_CTX_ARG
#undef N2_KERNELS_ONLY
#undef N2_P1B_NAME
#undef N2_P1B_QUAL

#define N2_P2B_QUAL __device__ __forceinline__
#define N2_P2B_NAME n2p6t2b_phase2b_body
#define N2_KERNELS_ONLY
#define N2_HOOK_CTX_ARG                                             \
  , const long long* __restrict__ k0p6_desc, int n2gm_task_lo,      \
      int n2gm_task_hi
#define N2GM_TASK_WAIT_HOOK \
  if (!k0p6_a2_wait(b, tid, k0p6_desc)) return;
#define N2GM_TASK_DONE_HOOK \
  k0p6_m15_task_done_defer(b, nc, tid, k0p6_desc, m7_pend);
#define N2GM_TASK_LOOP_HEAD_HOOK k0p6_m15_task_flush_defer(tid, k0p6_desc, m7_pend);
#define N2GM_TASK_LOOP_TAIL_HOOK k0p6_m15_task_flush_defer(tid, k0p6_desc, m7_pend);
#define N2GM_TASK_START k0p6_mps_task_start(k0p6_desc)
#define N2GM_TASK_STRIDE k0p6_mps_stride(k0p6_desc, (int)gridDim.x)
#define N2GM_TASK_LO n2gm_task_lo
#define N2GM_TASK_HI n2gm_task_hi
#define N2GM_TASK_DECODE(t_, tile_, nc_, ntiles_) \
  const int nc = (t_) / (ntiles_);                \
  const int tile = (t_) - nc * (ntiles_);
#include "n2_phase2b_gm_t2b.cpp"
#undef N2GM_TASK_DECODE
#undef N2GM_TASK_HI
#undef N2GM_TASK_LO
#undef N2GM_TASK_STRIDE
#undef N2GM_TASK_START
#undef N2GM_TASK_LOOP_TAIL_HOOK
#undef N2GM_TASK_LOOP_HEAD_HOOK
#undef N2GM_TASK_DONE_HOOK
#undef N2GM_TASK_WAIT_HOOK
#undef N2_HOOK_CTX_ARG
#undef N2_KERNELS_ONLY
#undef N2_P2B_NAME
#undef N2_P2B_QUAL
#undef N2GM_G

// The wgrad tile body (k0p6wg_tile + the M8.5-form driver), unchanged.
#include "n2_wgrad_gm_t2b.cpp"

// ---- small shared helpers -----------------------------------------------------------------
__device__ __forceinline__ float k0p6_bf16_to_f32(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}
__device__ __forceinline__ unsigned short k0p6_f32_to_bf16_rne(float f) {
  const unsigned int u = __float_as_uint(f);
  const unsigned int lsb = (u >> 16) & 1u;
  return (unsigned short)((u + 0x7FFFu + lsb) >> 16);
}
__device__ __forceinline__ float k0p6q_bf16_to_f32(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}

// One M8 combine batch over chunk range [CLO, CHI) (donor verbatim).
template <int NT, int CLO, int CHI>
__device__ __forceinline__ void k0p6_m15_m8_batch(
    int tok0, int T, const int* __restrict__ pull_ptr,
    const int* __restrict__ pull_src, int lane, int cur, int MAXTOK,
    unsigned short* __restrict__ slots, unsigned short* __restrict__ out) {
  int lo2[NT], fanout[NT];
  bool live[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int tok = tok0 + t;
    live[t] = (tok < T);
    lo2[t] = live[t] ? pull_ptr[tok] : 0;
    fanout[t] = live[t] ? (pull_ptr[tok + 1] - lo2[t]) : 0;
  }
  unsigned long long pbase[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    pbase[t] = 0ull;
    if ((lane >> 3) == t) {
      const int j2 = lane & 7;
      int p = cur;
      int row = cur * MAXTOK;
      if (j2 < fanout[t]) {
        p = pull_src[(size_t)(lo2[t] + j2) * 2 + 0];
        row = pull_src[(size_t)(lo2[t] + j2) * 2 + 1];
      }
      pbase[t] = (unsigned long long)(slots +
                 ((size_t)p * (size_t)MAXTOK + (size_t)row -
                  (size_t)cur * (size_t)MAXTOK) *
                     (size_t)K0P6_H);
    }
  }
  hk_moe::acquire_payload_system();
  __syncwarp();
#pragma unroll 1
  for (int c = CLO; c < CHI; ++c) {
    const size_t off = (size_t)(c << 10) + (size_t)(lane << 4);
    float acc[NT][8];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[t][e] = 0.f;
    }
#pragma unroll 1
    for (int jb = 0; jb < 8; jb += 4) {
      uint4 v2[NT][4];
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const unsigned long long pb = __shfl(pbase[t], 8 * t + jb + jj);
          v2[t][jj] = *reinterpret_cast<const uint4*>(
              reinterpret_cast<const unsigned char*>(pb) + off);
        }
      }
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          if (jb + jj < fanout[t]) {
            const unsigned short* const u16 =
                reinterpret_cast<const unsigned short*>(&v2[t][jj]);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[t][e] += k0p6_bf16_to_f32(u16[e]);
            const unsigned long long pz = __shfl(pbase[t], 8 * t + jb + jj);
            uint4 z4;
            z4.x = 0u; z4.y = 0u; z4.z = 0u; z4.w = 0u;
            *reinterpret_cast<uint4*>(
                reinterpret_cast<unsigned char*>(pz) + off) = z4;
          }
        }
      }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      if (!live[t]) continue;
      unsigned short* const orow = out + (size_t)(tok0 + t) * (size_t)K0P6_H;
      union {
        uint4 pack;
        unsigned short u16[8];
      } pk;
#pragma unroll
      for (int e = 0; e < 8; ++e) pk.u16[e] = k0p6_f32_to_bf16_rne(acc[t][e]);
      *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(orow) + off) = pk.pack;
    }
  }
}

#if K0P6_T2V6_FILLER
// ---- v6 filler machinery ------------------------------------------------------------------
// Tile-space maps.  The wgrad body's flat expert-major space per expert is
// [0, kT13) dW13 tiles then [kT13, kT13+kT2) dW2 tiles; the two cursors walk
// the two sub-spaces so claim order can respect the readiness lattice.
namespace t2v6 {

using production_fused_moe::n2::kWgMT;
using production_fused_moe::n2::kWgNT;
inline constexpr int kN13 = 7168 / kWgNT;
inline constexpr int kN2 = 2048 / kWgNT;
inline constexpr int kT13 = (4096 / kWgMT) * kN13;
inline constexpr int kT2 = (7168 / kWgMT) * kN2;

// Everything one wgrad tile needs, bound once after phase-1b.
struct fill_ctx {
  const std::uint8_t* dzq;
  const float* dqdz;
  const std::uint8_t* zq;
  const float* dqz;
  const std::uint8_t* xrows;
  const float* xsc;
  const std::uint8_t* dyq;
  const float* dysc;
  const int* sti;
  const float* swt;
  const int* sei;
  const int* nvi;
  __hip_bfloat16* dw13;
  __hip_bfloat16* dw2;
  int n_experts;
  bool accumulate;
  // cursors (mps_state words 4/5: zeroed by M0 every epoch)
  std::uint32_t* cur_dw2;
  std::uint32_t* cur_dw13;
  // LDS overlays (the M8.5 aliases: dead t2bsh storage post-phase-1b)
  __hip_bfloat16* ldsA;
  __hip_bfloat16* ldsB;
  float* ldsMeta;
  int* ldsRange;      // [n_experts][2] block ranges, t2bsh_amax alias
  int* ldsClaim;      // one int broadcast cell, t2bsh_srow alias
};

__device__ __forceinline__ void bind_ctx(const long long* desc,
                                         fill_ctx& c, bool accumulate,
                                         int em) {
  c.dzq = (const std::uint8_t*)k0p6_dread(desc, K0P6_D_A2Q);
  c.dqdz = (const float*)k0p6_dread(desc, K0P6_D_DQ2);
  c.zq = (const std::uint8_t*)k0p6_dread(desc, K0P6_D_T2B_ZQ);
  c.dqz = (const float*)k0p6_dread(desc, K0P6_D_T2B_DQZ);
  c.xrows = (const std::uint8_t*)k0p6_dread(desc, K0P6_D_T2B_XROWS);
  c.xsc = (const float*)k0p6_dread(desc, K0P6_D_T2B_XSC);
  c.dyq = (const std::uint8_t*)k0p6_dread(desc, K0P6_D_A_DST);
  c.dysc = (const float*)k0p6_dread(desc, K0P6_D_SC_STAGE);
  c.sti = (const int*)k0p6_dread(desc, K0P6_D_STI);
  c.swt = (const float*)k0p6_dread(desc, K0P6_D_SWT);
  c.sei = (const int*)k0p6_dread(desc, K0P6_D_SEI);
  c.nvi = (const int*)k0p6_dread(desc, K0P6_D_NVI);
  c.dw13 = (__hip_bfloat16*)k0p6_dread(desc, K0P6_D_T2B_DW13);
  c.dw2 = (__hip_bfloat16*)k0p6_dread(desc, K0P6_D_T2B_DW2);
  // The in-kernel PARTITION: cursors/tail cover experts [0, n_experts)
  // only; the host's side-stream slice drains the rest (flat tidx
  // [em*per_e, total) via the wgrad body's own tile_lo).  em == 0 or
  // >= E means everything is in-kernel.
  {
    const int e_all = (int)k0p6_dread(desc, K0P6_D_E);
    c.n_experts = (em > 0 && em < e_all) ? em : e_all;
  }
  c.accumulate = accumulate;
  std::uint32_t* st = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
  c.cur_dw2 = st + 4;
  c.cur_dw13 = st + 5;
  c.ldsA = reinterpret_cast<__hip_bfloat16*>(
      &production_fused_moe::n2::t2bsh_a2_lds[0][0]);
  c.ldsB = reinterpret_cast<__hip_bfloat16*>(
      &production_fused_moe::n2::t2bsh_a_lds[0][0][0][0]);
  c.ldsMeta = &production_fused_moe::n2::t2bsh_ascale_lds[0][0];
  c.ldsRange = reinterpret_cast<int*>(
      &production_fused_moe::n2::t2bsh_amax_lds[0][0]);
  c.ldsClaim = &production_fused_moe::n2::t2bsh_srow_lds[0];
}

// Per-CTA lazy expert-range scan into the LDS overlay (the wgrad driver's
// scan, hoisted so window quanta don't re-scan).  The scanned-once flag
// lives in dead LDS (ldsClaim[1], zeroed once post-phase-1b) so NOTHING
// filler-related is register-live outside the windows — the whole-function
// allocator must never see filler state at the phase-2b pressure peak.
__device__ __forceinline__ void ensure_ranges(const fill_ctx& c) {
  __syncthreads();
  if (c.ldsClaim[1] != 0) {
    __syncthreads();
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int padded = c.nvi[0];
  const int nblk = (padded + 31) >> 5;
  __syncthreads();
  for (int e = tid; e < 2 * c.n_experts; e += (int)blockDim.x)
    c.ldsRange[e] = 0;
  __syncthreads();
  for (int b = tid; b < nblk; b += (int)blockDim.x) {
    const int e = c.sei[b];
    if (e < 0 || e >= c.n_experts) continue;
    if (b == 0 || c.sei[b - 1] != e) c.ldsRange[2 * e + 0] = b;
    if (b == nblk - 1 || c.sei[b + 1] != e) c.ldsRange[2 * e + 1] = b + 1;
  }
  __syncthreads();
  if (tid == 0) c.ldsClaim[1] = 1;
  __syncthreads();
}

// Run one tile of the dW2 sub-space (flat index i in [0, E*kT2)).
__device__ __forceinline__ void run_dw2(const fill_ctx& c, int i) {
  const int e = i / kT2;
  const int v = i - e * kT2;
  const int m0 = (v / kN2) * kWgMT;
  const int n0 = (v % kN2) * kWgNT;
  const int s = c.ldsRange[2 * e + 0] * 32;
  const int t = c.ldsRange[2 * e + 1] * 32;
  production_fused_moe::n2::k0p6wg_tile<1>(
      c.dzq, c.dqdz, c.zq, c.dqz, c.xrows, c.xsc, c.dyq, c.dysc, c.sti,
      c.swt, c.nvi[1], c.nvi[0], s, t, m0, n0,
      c.dw2 + (std::size_t)e * 7168 * 2048, 2048, c.accumulate, c.ldsA,
      c.ldsB, c.ldsMeta);
}

// Run one tile of the dW13 sub-space (flat index i in [0, E*kT13)).
__device__ __forceinline__ void run_dw13(const fill_ctx& c, int i) {
  const int e = i / kT13;
  const int u = i - e * kT13;
  const int m0 = (u / kN13) * kWgMT;
  const int n0 = (u % kN13) * kWgNT;
  const int s = c.ldsRange[2 * e + 0] * 32;
  const int t = c.ldsRange[2 * e + 1] * 32;
  production_fused_moe::n2::k0p6wg_tile<0>(
      c.dzq, c.dqdz, c.zq, c.dqz, c.xrows, c.xsc, c.dyq, c.dysc, c.sti,
      c.swt, c.nvi[1], c.nvi[0], s, t, m0, n0,
      c.dw13 + (std::size_t)e * 4096 * 7168, 7168, c.accumulate, c.ldsA,
      c.ldsB, c.ldsMeta);
}

// CTA-collective claim off one cursor.  Returns the claimed index or -1 when
// the sub-space is exhausted.  tid0 fetch_adds, LDS-broadcasts; the trailing
// __syncthreads keeps the claim cell single-writer per round (the tile body
// opens with its own __syncthreads before touching ldsA/ldsB).
__device__ __forceinline__ int claim(std::uint32_t* cursor, int space,
                                     int* cell) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid == 0) {
    const std::uint32_t i =
        kittens::distributed::fetch_add_relaxed<hk_moe::scope::agent>(cursor,
                                                                      1u);
    *cell = ((int)i < space) ? (int)i : -1;
  }
  __syncthreads();
  const int got = *cell;
  __syncthreads();
  return got;
}

// One quota-bounded window: claim dW2 tiles (and, when `allow13`, fall over
// to dW13 tiles once dW2 is exhausted).  Returns tiles actually run.
__device__ __forceinline__ int window(const fill_ctx& c, int quota,
                                      bool allow13) {
  int ran = 0;
  if (quota <= 0) return 0;
  ensure_ranges(c);
  const int space2 = c.n_experts * kT2;
  const int space13 = c.n_experts * kT13;
  while (ran < quota) {
    int i = claim(c.cur_dw2, space2, c.ldsClaim);
    if (i >= 0) {
      run_dw2(c, i);
      ++ran;
      continue;
    }
    if (!allow13) break;
    i = claim(c.cur_dw13, space13, c.ldsClaim);
    if (i < 0) break;
    run_dw13(c, i);
    ++ran;
  }
  return ran;
}

}  // namespace t2v6
#endif  // K0P6_T2V6_FILLER

// ================================================================================================
extern "C" __global__ void __launch_bounds__(256, 1) K0P6_M15_KERNEL_NAME(
    const long long* __restrict__ desc, int* __restrict__ pperr, long long spin_limit) {
  if (desc == nullptr || pperr == nullptr) return;
  const int TOPK = (int)k0p6_dread(desc, K0P6_D_TOPK);
  const int E = (int)k0p6_dread(desc, K0P6_D_E);
  const int cur = (int)k0p6_dread(desc, K0P6_D_CUR);
  const int world = (int)k0p6_dread(desc, K0P6_D_WORLD);
  const int T_loc_max = (int)k0p6_dread(desc, K0P6_D_TLOCMAX);
  const int PADMAX = (int)k0p6_dread(desc, K0P6_D_PADMAX);
  const int MAXTOK = (int)k0p6_dread(desc, K0P6_D_MAXTOK);
  const int input_tokens = (int)k0p6_dread(desc, K0P6_D_T);
  const long long symmetric_address = k0p6_dread(desc, K0P6_D_SYMMETRIC);

  __shared__ __align__(16) unsigned char s_row[4][7488];
  __shared__ int s_cnt[K0P6_MAXE];
  __shared__ unsigned int s_total;
  __shared__ int s_fill[64];
  __shared__ int s_ns[8];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nct = (int)gridDim.x;
  const int lane = tid & 63;
  const int wid = tid >> 6;

  if (symmetric_address == 0 ||
      (static_cast<unsigned long long>(symmetric_address) & 7ull) != 0ull ||
      TOPK != 8 || E <= 0 || E > K0P6_MAXE || world != 8 ||
      cur < 0 || cur >= world || T_loc_max <= 0 ||
      T_loc_max > 0x00FFFFFF || PADMAX <= 0 || MAXTOK <= 0 ||
      MAXTOK > T_loc_max / 8 || input_tokens < 0 ||
      input_tokens > MAXTOK || spin_limit < 0 || nct != 256 ||
      (MAXTOK + K0P6_CHUNK - 1) / K0P6_CHUNK > K0P6C_NCHUNK_MAX) {
    if (bid == 0 && tid == 0) atomicOr(pperr, 131072);
    return;
  }
  {
    const hk_moe::mps::config entry_cfg = hk_moe::mps::decode_config(
        (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
    if (!hk_moe::mps::config_is_valid(entry_cfg) || PADMAX / 32 > 16383 ||
        !hk_moe::mps::mode_is_direct_accum(entry_cfg) ||
        (MAXTOK & (MAXTOK - 1)) != 0 ||
        k0p6_dread(desc, K0P6_D_MPS_STATE) == 0 ||
        (k0p6_dread(desc, K0P6_D_MPS_STATE) & 7) != 0 ||
        k0p6_dread(desc, K0P6_D_MPS_SLOTS) == 0 ||
        (k0p6_dread(desc, K0P6_D_MPS_SLOTS) & 15) != 0 ||
        k0p6_dread(desc, K0P6_D_ROW_READY) == 0 ||
        world * MAXTOK + K0P6_M15_SLABS > T_loc_max
        // t2b: the saved-plan save buffers must be bound.
        || k0p6_dread(desc, K0P6_D_T2B_ZQ) == 0
        || k0p6_dread(desc, K0P6_D_T2B_DQZ) == 0
        || k0p6_dread(desc, K0P6_D_T2B_ACTQ) == 0
        || k0p6_dread(desc, K0P6_D_T2B_ACTDQ) == 0
        || k0p6_dread(desc, K0P6_D_T2B_XROWS) == 0
        || k0p6_dread(desc, K0P6_D_T2B_XSC) == 0
        || k0p6_dread(desc, K0P6_D_T2B_DW13) == 0
        || k0p6_dread(desc, K0P6_D_T2B_DW2) == 0
        || k0p6_dread(desc, K0P6_D_T2B_DWTOPK) == 0
#if K0P6_T2V6_PROF
        || k0p6_dread(desc, K0P6_D_T2V6_PROF) == 0
        || (k0p6_dread(desc, K0P6_D_T2V6_PROF) & 7) != 0
#endif
        ) {
      if (bid == 0 && tid == 0) atomicOr(pperr, K0P6_MPS_ERR_CONFIG);
      return;
    }
  }
  const int EL = E;
  const int T_ext = world * MAXTOK;
  const int lo = cur * E;
  const long long debug_stop = k0p6_dread(desc, K0P6_D_DEBUG);

  // v6: the packed DWMODE word (see file header).  Mode 0/1 preserve the t2b
  // meanings; mode 2 enables the window scheduler.
  const unsigned int dwword =
      (unsigned int)(unsigned long long)k0p6_dread(desc, K0P6_D_T2B_DWMODE);
  const int dwmode = (int)(dwword & 0xFu);
  (void)dwmode;
#if K0P6_T2V6_FILLER
  // accumulate is DERIVED, not host-patched: this layer's protocol instance
  // launches exactly once per microbatch, so (epoch-1) mod nmb is the
  // microbatch index — 0 overwrites (first mb of the iteration), the rest
  // bf16-RMW accumulate.  nmb rides DWMODE bits [7:4]; nmb <= 1 (gates,
  // single-mb harnesses) never accumulates.
  const int dw_nmb = (int)((dwword >> 4) & 0xFu);
  const bool dw_accumulate =
      (dw_nmb > 1) &&
      (((k0p6_epoch(desc) - 1ull) % (unsigned long long)dw_nmb) != 0ull);
  const int dw_quota_a = (int)((dwword >> 8) & 0xFFu);
  const int dw_quota_b = (int)((dwword >> 16) & 0xFFu);
  // [31:24] = E_m, the expert-space partition: experts [0, E_m) drain
  // IN-KERNEL (windows + tail); experts [E_m, E) belong to the host's
  // side-stream slice launch (small-grid wg_standalone over flat tidx
  // [E_m*per_e, total)), which hides under the non-MoE region the way the
  // bmm arm proved.  0 => everything in-kernel.  The M8-wait window quota
  // is self-limiting (certificate arrival ends it), so it is a constant.
  const int dw_em = (int)((dwword >> 24) & 0xFFu);
  const int dw_quota_c = 32;
#endif
#if K0P6_T2V6_PROF
  std::uint64_t* t2v6_prof =
      (std::uint64_t*)k0p6_dread(desc, K0P6_D_T2V6_PROF);
  if (tid == 0) {
    const std::uint64_t t = hk_moe::mps::realtime_now();
    atomicMax((unsigned long long*)(t2v6_prof + T2V6_PB_ENTER_MIN),
              (unsigned long long)(~t));
    if (bid == 0) {
      atomicAdd((unsigned long long*)(t2v6_prof + T2V6_PB_LAUNCHES), 1ull);
      const hk_moe::mps::config pcfg = hk_moe::mps::decode_config(
          (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
      atomicAdd((unsigned long long*)(t2v6_prof + T2V6_PB_SVC_CTAS),
                (unsigned long long)pcfg.reserved_comm_ctas);
    }
  }
#endif

  // ================= M0: retire wait + per-block counter zero (donor verbatim) =============
  {
    const std::uint64_t epoch = k0p6_epoch(desc);
    std::uint64_t* retired =
        (std::uint64_t*)k0p6_dread(desc, K0P6_D_RETIRED);
    T2V6_PROF_ONLY(std::uint64_t t2v6_t0 = 0;
                   if (tid == 0) t2v6_t0 = hk_moe::mps::realtime_now();)
    if (tid < world) {
      (void)hk_moe::poll_value_at_least_system(
          retired + tid, epoch > 0 ? epoch - 1ull : 0ull,
          (std::uint64_t)spin_limit, pperr, 16777216);
    }
    T2V6_PROF_ONLY(if (tid == 0) {
      const std::uint64_t t1 = hk_moe::mps::realtime_now();
      atomicAdd((unsigned long long*)(t2v6_prof + T2V6_PB_RETIRE),
                (unsigned long long)(t1 - t2v6_t0));
      t2v6_t0 = t1;
    })
    {
      unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
      const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
      const hkp::fail_closed bar_err{pperr, 2097152};
      hkp::grid_barrier(bar, tid, bar_err);
    }
    T2V6_PROF_ONLY(if (tid == 0) {
      atomicAdd((unsigned long long*)(t2v6_prof + T2V6_PB_M0BAR),
                (unsigned long long)(hk_moe::mps::realtime_now() - t2v6_t0));
    })
    hk_moe::acquire_payload_agent();
    if (hk_moe::error_bit_set_agent(pperr, 16777216 | 2097152)) return;
    const int maxb = (int)k0p6_dread(desc, K0P6_D_MAXB);
    unsigned int* a2_done = (unsigned int*)k0p6_dread(desc, K0P6_D_A2_DONE);
    unsigned int* part_done = (unsigned int*)k0p6_dread(desc, K0P6_D_PART_DONE);
    for (int i = bid * (int)blockDim.x + tid; i < maxb; i += nct * (int)blockDim.x) {
      a2_done[i] = 0u;
      part_done[i] = 0u;
    }
    {
      std::uint32_t* mps_state = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
      const int gtid0 = bid * (int)blockDim.x + tid;
      const int gstride0 = nct * (int)blockDim.x;
      for (int i = gtid0; i < K0P6_MPS_ST_WORDS + 2 * K0P6_MPS_TS_COUNT;
           i += gstride0) {
        mps_state[i] = 0u;
      }
    }
    // exp_56 fix (6): grid-wide reset of the NEXT epoch's dest_counter parity.
    {
      unsigned int* dc_next =
          (unsigned int*)k0p6_dread(desc, K0P6_D_DESTCTR)
          + (size_t)(((k0p6_epoch(desc) + 1ull) & 1ull)) * (size_t)world;
      for (int i = bid * (int)blockDim.x + tid; i < world; i += nct * (int)blockDim.x)
        hk_moe::publish_relaxed<hk_moe::scope::system>(dc_next + i, 0u);
      {
        const int self = cur;
        for (int i = bid * (int)blockDim.x + tid; i < world; i += nct * (int)blockDim.x) {
          if (i == self &&
              hk_moe::load_relaxed<hk_moe::scope::system>(dc_next + i) != 0u)
            atomicOr(pperr, 262144);
        }
      }
    }
    __syncthreads();
  }
  if (debug_stop == 1) return;

  // ================= M1: qpush dispatch (donor verbatim, saved-plan form) ===================
  {
    const auto* symmetric = k0p6_symmetric(desc);
    const std::uint64_t epoch = k0p6_epoch(desc);
    const int par = (int)(epoch & 1ull);
    const int T = (int)k0p6_dread(desc, K0P6_D_T);
    const int* my_ids = (const int*)k0p6_dread(desc, K0P6_D_MYIDS);
    const float* my_wgt = (const float*)k0p6_dread(desc, K0P6_D_MYWG);
    const unsigned short* hidden = (const unsigned short*)k0p6_dread(desc, K0P6_D_HIDDEN);
    unsigned char* a_ll = (unsigned char*)k0p6_dread(desc, K0P6_D_A_LL);
    unsigned int* dest_counter =
        (unsigned int*)k0p6_dread(desc, K0P6_D_DESTCTR) + (size_t)par * (size_t)world;
    unsigned int* pushed_count = (unsigned int*)k0p6_dread(desc, K0P6_D_PUSHED);
    std::uint64_t* rows_done =
        (std::uint64_t*)k0p6_dread(desc, K0P6_D_ROWS_DONE);
    unsigned int* qpush_done = (unsigned int*)k0p6_dread(desc, K0P6_D_QPUSH_DONE);
    int* pull_stage = (int*)k0p6_dread(desc, K0P6_D_PULL_STAGE);
    int* pull_cnt = (int*)k0p6_dread(desc, K0P6_D_PULL_CNT);
    (void)pull_cnt;

    const int gw = bid * (blockDim.x >> 6) + wid;
    const int nw = nct * (blockDim.x >> 6);
    for (int tau = gw; tau < T; tau += nw) {
      const int s = lane;
      const int eid = (s < TOPK) ? my_ids[(size_t)tau * TOPK + s] : -1;
      const float wgt = (s < TOPK) ? my_wgt[(size_t)tau * TOPK + s] : 0.0f;
      // t2b/M1b: destinations and rows come from the FORWARD'S SAVED PLAN.
      const int dest =
          (s < TOPK) ? pull_stage[((size_t)tau * TOPK + s) * 2 + 0] : -1;
      const int saved_row =
          (s < TOPK) ? pull_stage[((size_t)tau * TOPK + s) * 2 + 1] : -1;
      const bool primary = (dest >= 0) && (saved_row >= 0);
      const unsigned long long pmask = __ballot(primary);
      const int fanout = __popcll(pmask);

      const unsigned int row = primary ? (unsigned int)saved_row : 0u;
      (void)dest_counter;  // bound for M0's parity zeroing; unused in M1b
      if (fanout == 0) continue;

      const unsigned short* hrow = hidden + (size_t)tau * K0P5_K;
      unsigned char* row_stream = s_row[wid];
      uint4 v[7];
#pragma unroll
      for (int c = 0; c < 7; ++c) {
        const uint4 raw0 = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const unsigned char*>(hrow) + (c << 11) + (lane << 5));
        const uint4 raw1 = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const unsigned char*>(hrow) + (c << 11) + (lane << 5) + 16);
        const unsigned short* h16 = reinterpret_cast<const unsigned short*>(&raw0);
        const unsigned short* h16b = reinterpret_cast<const unsigned short*>(&raw1);
        float a = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) a = fmaxf(a, fabsf(k0p6q_bf16_to_f32(h16[e])));
#pragma unroll
        for (int e = 0; e < 8; ++e) a = fmaxf(a, fabsf(k0p6q_bf16_to_f32(h16b[e])));
#pragma unroll
        for (int o = 1; o <= 4; o <<= 1) a = fmaxf(a, __shfl_xor(a, o, 64));
        const float scale = fmaxf(a, 1.0e-6f) / 448.0f;
        union { unsigned char b[16]; uint4 v; } qo;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float q = fminf(fmaxf(k0p6q_bf16_to_f32(h16[e]) / scale, -448.0f), 448.0f);
          qo.b[e] = __hip_cvt_float_to_fp8(q, __HIP_SATFINITE, __HIP_E4M3);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float q = fminf(fmaxf(k0p6q_bf16_to_f32(h16b[e]) / scale, -448.0f), 448.0f);
          qo.b[8 + e] = __hip_cvt_float_to_fp8(q, __HIP_SATFINITE, __HIP_E4M3);
        }
        v[c] = qo.v;
        if ((lane & 7) == 0)
          *reinterpret_cast<float*>(row_stream + K0P5_K + c * 32 + (lane >> 3) * 4) = scale;
      }
#pragma unroll
      for (int c = 0; c < 7; ++c)
        *reinterpret_cast<uint4*>(row_stream + (c << 10) + (lane << 4)) = v[c];
      {
        const uint4 ep0 = make_uint4((unsigned int)__shfl(eid, 0), (unsigned int)__shfl(eid, 1),
                                     (unsigned int)__shfl(eid, 2), (unsigned int)__shfl(eid, 3));
        if (lane == 0) *reinterpret_cast<uint4*>(row_stream + K0P5_K + K0P5_SC_BYTES) = ep0;
        const uint4 ep1 = make_uint4((unsigned int)__shfl(eid, 4), (unsigned int)__shfl(eid, 5),
                                     (unsigned int)__shfl(eid, 6), (unsigned int)__shfl(eid, 7));
        if (lane == 1) *reinterpret_cast<uint4*>(row_stream + K0P5_K + K0P5_SC_BYTES + 16) = ep1;
        const float4 wp0 = make_float4(__shfl(wgt, 0), __shfl(wgt, 1), __shfl(wgt, 2),
                                       __shfl(wgt, 3));
        if (lane == 2) *reinterpret_cast<float4*>(row_stream + K0P5_K + K0P5_SC_BYTES + 32) = wp0;
        const float4 wp1 = make_float4(__shfl(wgt, 4), __shfl(wgt, 5), __shfl(wgt, 6),
                                       __shfl(wgt, 7));
        if (lane == 3) *reinterpret_cast<float4*>(row_stream + K0P5_K + K0P5_SC_BYTES + 48) = wp1;
      }
      __syncwarp();

      const uint64_t* row64 = reinterpret_cast<const uint64_t*>(row_stream);
      const int g = lane >> 3, j = lane & 7;
      unsigned long long pm = pmask;
#pragma unroll 1
      for (int jj = 0; jj < fanout; ++jj) {
        const int src_lane = __ffsll((long long)pm) - 1;
        pm &= pm - 1;
        const int pe = __shfl(dest, src_lane);
        const int drow_id = __shfl((int)row, src_lane);
        if (lane == 0 && drow_id >= 0 && drow_id < T_loc_max)
          atomicAdd(pushed_count + pe, 1u);
        if (drow_id < 0 || drow_id >= T_loc_max) continue;
        unsigned char* dst = hk_moe::peer_ptr(
            a_ll + (size_t)drow_id * K0P5_ROW_STRIDE, pe,
            symmetric);
        hk_moe::store_dispatch_row(
            dst, row64, (size_t)K0P5_ROW_U64, g, j);
      }
      __syncwarp();
    }

    hk_moe::release_cta_payload_system();
    std::uint64_t* chunk_ready =
        (std::uint64_t*)k0p6_dread(desc, K0P6_D_CHUNK_READY);
    const unsigned int epoch32 = (unsigned int)epoch;
    const int nchunks = K0P6C_NCHUNK_MAX;
    if (tid == 0) {
      const unsigned int old = __hip_atomic_fetch_add(qpush_done, 1u, __ATOMIC_ACQ_REL,
                                                      __HIP_MEMORY_SCOPE_AGENT);
      if ((old % gridDim.x) == (gridDim.x - 1)) {
        hk_moe::release_signal_batch_system();
        for (int pe = 0; pe < world; ++pe) {
          const unsigned int n_s = pushed_count[pe];
          if (pe == cur) {
            hk_moe::publish_epoch_payload<hk_moe::scope::agent>(
                &rows_done[cur], epoch32, n_s);
          } else {
            std::uint64_t* peer =
                hk_moe::peer_ptr(rows_done + cur, pe, symmetric);
            hk_moe::publish_epoch_payload<hk_moe::scope::system>(
                peer, epoch32, n_s);
          }
          std::uint64_t* dest_cr =
              (pe == cur) ? chunk_ready
                          : hk_moe::peer_ptr(chunk_ready, pe, symmetric);
          for (int c = 0; c < nchunks; ++c) {
            const int lo_row = c * K0P6_CHUNK;
            int fill = (int)n_s - lo_row;
            if (fill < 0) fill = 0;
            else if (fill > K0P6_CHUNK) fill = K0P6_CHUNK;
            hk_moe::publish_epoch_payload<hk_moe::scope::system>(
                dest_cr + (size_t)cur * (size_t)nchunks + (size_t)c,
                epoch32, (unsigned int)fill);
          }
        }
      }
    }
  }
  if (debug_stop == 2) return;

  // ================= M2: chunk-acquire unpack + histogram (donor verbatim) =================
  int T_loc;
  {
    const std::uint64_t epoch = k0p6_epoch(desc);
    const unsigned int epoch32 = (unsigned int)epoch;
    unsigned char* a_ll = (unsigned char*)k0p6_dread(desc, K0P6_D_A_LL);
    const std::uint64_t* rows_done =
        (const std::uint64_t*)k0p6_dread(desc, K0P6_D_ROWS_DONE);
    const std::uint64_t* chunk_ready =
        (const std::uint64_t*)k0p6_dread(desc, K0P6_D_CHUNK_READY);
    unsigned char* a_dst = (unsigned char*)k0p6_dread(desc, K0P6_D_A_DST);
    float* sc_stage = (float*)k0p6_dread(desc, K0P6_D_SC_STAGE);
    int* recv_eid = (int*)k0p6_dread(desc, K0P6_D_RECV_EID);
    float* recv_wgt = (float*)k0p6_dread(desc, K0P6_D_RECV_WGT);
    unsigned int* row_remaining = (unsigned int*)k0p6_dread(desc, K0P6_D_ROW_REM);
    int* hcnt = (int*)k0p6_dread(desc, K0P6_D_HCNT);
    const int nchunks = K0P6C_NCHUNK_MAX;

    for (int i = tid; i < EL; i += blockDim.x) s_cnt[i] = 0;
    if (tid == 0) s_total = 0u;
    __syncthreads();

    const int g = lane >> 3, j = lane & 7;
    const int wave = bid * (blockDim.x >> 6) + (tid >> 6);
    const int wave_count = nct * (blockDim.x >> 6);

    T2V6_PROF_ONLY(std::uint64_t t2v6_t2 = 0;
                   if (tid == 0) t2v6_t2 = hk_moe::mps::realtime_now();)
    if (tid == 0) {
      long long spins = 0;
      int all = 0;
      while (!all) {
        all = 1;
        unsigned int tsum = 0;
        for (int src = 0; src < world; ++src) {
          const std::uint64_t v2 =
              hk_moe::load_relaxed<hk_moe::scope::system>(rows_done + src);
          if ((v2 >> 32) == epoch) tsum += (unsigned int)(v2 & 0xffffffffu);
          else all = 0;
        }
        if (all) { s_total = tsum; break; }
        if (++spins > spin_limit) { atomicOr(pperr, 16777216); break; }
        __builtin_amdgcn_s_sleep(4);
      }
    }
    T2V6_PROF_ONLY(if (tid == 0) {
      const std::uint64_t t1 = hk_moe::mps::realtime_now();
      atomicAdd((unsigned long long*)(t2v6_prof + T2V6_PB_M2ROWS),
                (unsigned long long)(t1 - t2v6_t2));
      t2v6_t2 = t1;
    })
    __syncthreads();
    const unsigned int total = s_total;

    const int npairs = world * nchunks;
    for (int k = wave; k < npairs; k += wave_count) {
      const int s = k / nchunks;
      const int c = k - s * nchunks;
      unsigned int fill = 0u;
      bool ok = true;
      if (lane == 0)
        ok = hk_moe::poll_epoch_word_system(
            chunk_ready + (size_t)s * (size_t)nchunks + (size_t)c,
            epoch32, (std::uint64_t)spin_limit, pperr, 8388608,
            &fill, (unsigned int*)k0p6_dread(desc, K0P6_D_SPIN_DBG));
      ok = (__shfl(ok ? 1 : 0, 0) != 0);
      (void)ok;
    }
    {
      unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
      const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
      const hkp::fail_closed bar_err{pperr, 2097152};
      hkp::grid_barrier(bar, tid, bar_err);
    }
    T2V6_PROF_ONLY(if (tid == 0) {
      atomicAdd((unsigned long long*)(t2v6_prof + T2V6_PB_M2CHUNK),
                (unsigned long long)(hk_moe::mps::realtime_now() - t2v6_t2));
    })
    hk_moe::acquire_payload_agent();
    if (hk_moe::error_bit_set_agent(
            pperr, 16777216 | 8388608 | 2097152)) return;
    if (tid == 0) {
      std::uint32_t* mps_state2 =
          (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
      const hk_moe::mps::config cfg2 = hk_moe::mps::decode_config(
          (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
      std::uint64_t* ts2 = (std::uint64_t*)(mps_state2 + K0P6_MPS_ST_WORDS);
      hk_moe::mps::ts_mark(ts2 + K0P6_MPS_TS_M2_DONE, ts2, bid,
                           K0P6_MPS_E23_M2_DONE, cfg2.timestamps);
    }
    for (int k = tid; k < npairs; k += (int)blockDim.x) {
      const int s = k / nchunks;
      const int c = k - s * nchunks;
      const std::uint64_t w = hk_moe::load_relaxed<hk_moe::scope::system>(
          chunk_ready + (size_t)s * (size_t)nchunks + (size_t)c);
      s_fill[k] = (int)(unsigned int)(w & 0xffffffffu);
    }
    __syncthreads();
    if (tid < world) {
      int ns = 0;
      for (int cc = 0; cc < nchunks; ++cc) ns += s_fill[tid * nchunks + cc];
      s_ns[tid] = ns;
    }
    __syncthreads();
    hk_moe::acquire_payload_system();
    __syncthreads();

    for (int r = wave; r < T_ext; r += wave_count) {
      const int s = r / MAXTOK;
      const int off = r - s * MAXTOK;
      if (off >= s_ns[s]) continue;
      k0p6_unpack_row_nopoll((size_t)r, a_ll + (size_t)r * (size_t)K0P5_ROW_STRIDE, a_dst,
                             sc_stage, recv_eid, recv_wgt, g, j);
      __syncwarp();
      if (lane < 8) {
        const int e = recv_eid[(size_t)r * 8 + lane];
        const bool live_e = (e >= lo && e < lo + E);
        if (live_e) atomicAdd(&s_cnt[e - lo], 1);
        const unsigned int live_ballot = __ballot(live_e) & 0xFFu;
        if (lane == 0) row_remaining[r] = (unsigned int)__popcll(live_ballot);
      }
      __syncwarp();
    }

    T_loc = (int)total;
    if (T_ext < 0 || T_ext > T_loc_max) {
      if (bid == 0 && tid == 0) atomicOr(pperr, 524288);
      return;
    }

    {
      int* recv_eid_h = (int*)k0p6_dread(desc, K0P6_D_RECV_EID);
      const int gtid = bid * (int)blockDim.x + tid;
      const int gstride = nct * (int)blockDim.x;
      for (int s = 0; s < world; ++s) {
        for (int c = 0; c < nchunks; ++c) {
          const std::uint64_t w = hk_moe::load_relaxed<hk_moe::scope::system>(
              chunk_ready + (size_t)s * (size_t)nchunks + (size_t)c);
          const unsigned int fill = (unsigned int)(w & 0xffffffffu);
          // exp_04_tgen (aug13) T-GENERALIZATION FIX: clamp hole-sentinel
          // chunk ranges to the segment.
          long long seg_row1 = (long long)(c + 1) * K0P6_CHUNK;
          if (seg_row1 > (long long)MAXTOK) seg_row1 = (long long)MAXTOK;
          const long long base = (long long)s * (long long)MAXTOK + (long long)c * K0P6_CHUNK;
          const long long hole0 = (base + (long long)fill) * TOPK;
          const long long hole1 = ((long long)s * (long long)MAXTOK + seg_row1) * TOPK;
          for (long long i = hole0 + gtid; i < hole1; i += gstride) recv_eid_h[i] = -1;
        }
      }
    }

    __syncthreads();
    for (int i = tid; i < EL; i += blockDim.x) hcnt[i * nct + bid] = s_cnt[i];
  }
  if (debug_stop == 3) return;

  // ================= M3-position: dw zero + ordering barrier (t2b form) =====================
  {
    float* dw0 = (float*)k0p6_dread(desc, K0P6_D_T2B_DWTOPK);
    for (int i = bid * (int)blockDim.x + tid; i < PADMAX;
         i += nct * (int)blockDim.x)
      dw0[i] = 0.0f;
    T2V6_PROF_ONLY(std::uint64_t t2v6_t3 = 0;
                   if (tid == 0) t2v6_t3 = hk_moe::mps::realtime_now();)
    unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
    const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
    const hkp::fail_closed bar_err{pperr, 2097152};
    hkp::grid_barrier(bar, tid, bar_err);
    T2V6_PROF_ONLY(if (tid == 0) {
      atomicAdd((unsigned long long*)(t2v6_prof + T2V6_PB_M3BAR),
                (unsigned long long)(hk_moe::mps::realtime_now() - t2v6_t3));
    })
    if (hk_moe::error_bit_set_agent(pperr, 2097152)) return;
  }
  if (debug_stop == 4) return;

  // ================= M5.9b: z regeneration (see n2_phase1z header) ==========================
  production_fused_moe::n2::n2p6t2b_phase1z_body(
      (const std::uint8_t*)k0p6_dread(desc, K0P6_D_T2B_XROWS),
      (const float*)k0p6_dread(desc, K0P6_D_T2B_XSC),
      (const std::uint8_t*)k0p6_dread(desc, K0P6_D_T2B_ACTQ),   // = W13 orig
      (const float*)k0p6_dread(desc, K0P6_D_T2B_ACTDQ),         // = S13 orig
      (const int*)k0p6_dread(desc, K0P6_D_STI),
      (const int*)k0p6_dread(desc, K0P6_D_SEI),
      (const int*)k0p6_dread(desc, K0P6_D_NVI),
      (std::uint8_t*)k0p6_dread(desc, K0P6_D_T2B_ZQ),
      (float*)k0p6_dread(desc, K0P6_D_T2B_DQZ),
      (const int*)k0p6_dread(desc, K0P6_D_TILE_DESC),
      (int)((const int*)k0p6_dread(desc, K0P6_D_NUM_TILES))[0], desc);
  // Zq must be complete before ANY CTA's phase-1b epilogue reads it.
  {
    T2V6_PROF_ONLY(std::uint64_t t2v6_tz = 0;
                   if (tid == 0) t2v6_tz = hk_moe::mps::realtime_now();)
    unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
    const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
    const hkp::fail_closed bar_err{pperr, 2097152};
    hkp::grid_barrier(bar, tid, bar_err);
    T2V6_PROF_ONLY(if (tid == 0) {
      atomicAdd((unsigned long long*)(t2v6_prof + T2V6_PB_ZBAR),
                (unsigned long long)(hk_moe::mps::realtime_now() - t2v6_tz));
    })
    if (hk_moe::error_bit_set_agent(pperr, 2097152)) return;
  }

  production_fused_moe::n2::n2p6t2b_phase1b_body(
      (const std::uint8_t*)k0p6_dread(desc, K0P6_D_A_DST),        // dY rows
      (const float*)k0p6_dread(desc, K0P6_D_SC_STAGE),            // dY scales
      (const std::uint8_t*)k0p6_dread(desc, K0P6_D_W13),          // = W2T
      (const float*)k0p6_dread(desc, K0P6_D_S13),                 // = S2T
      (const std::uint8_t*)k0p6_dread(desc, K0P6_D_T2B_ZQ),
      (const float*)k0p6_dread(desc, K0P6_D_T2B_DQZ),
      (const int*)k0p6_dread(desc, K0P6_D_STI),
      (const int*)k0p6_dread(desc, K0P6_D_SEI),
      (const int*)k0p6_dread(desc, K0P6_D_NVI),
      (std::uint8_t*)k0p6_dread(desc, K0P6_D_A2Q),                // = dZq
      (float*)k0p6_dread(desc, K0P6_D_DQ2),                       // = DQdZ
      (float*)k0p6_dread(desc, K0P6_D_T2B_DWTOPK),
      (const int*)k0p6_dread(desc, K0P6_D_TILE_DESC),
      (int)((const int*)k0p6_dread(desc, K0P6_D_NUM_TILES))[0], desc);
  if (debug_stop == 5) return;
  if (tid == 0) {
    std::uint32_t* mps_state6 =
        (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
    const hk_moe::mps::config cfg6 = hk_moe::mps::decode_config(
        (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
    std::uint64_t* ts6 = (std::uint64_t*)(mps_state6 + K0P6_MPS_ST_WORDS);
    hk_moe::mps::ts_mark(ts6 + K0P6_MPS_TS_M6_DONE, ts6, bid,
                         K0P6_MPS_E23_M6_DONE, cfg6.timestamps);
  }

#if K0P6_T2V6_FILLER
  // Zero the per-CTA ranges-scanned flag (dead t2bsh LDS post-phase-1b; the
  // filler context itself is (re)bound LOCALLY inside every window so no
  // filler state is register-live at the phase-2b pressure peak).
  if (dwmode >= 2) {
    if (tid == 0) production_fused_moe::n2::t2bsh_srow_lds[1] = 0;
    __syncthreads();
  }
#endif

  // ================= M7: NC-MAJOR slab loop + slab-certified rendezvous ====================
  // One textual copy of the rendezvous sequence, executed once per slab (the
  // exp_38/exp_34 lesson: a SECOND copy of release/barrier/acquire pushed
  // whole-function allocation over a spill cliff; a loop keeps one copy).
  {
    for (int slab = 0; slab < K0P6_M15_SLABS; ++slab) {
      const int ntiles =
          ((const int*)k0p6_dread(desc, K0P6_D_NUM_TILES))[0];
      const int t_lo = slab * K0P6_M15_NC_PER_SLAB * ntiles;
      const int t_hi = t_lo + K0P6_M15_NC_PER_SLAB * ntiles;
      T2V6_PROF_ONLY(bool t2v6_is_svc = false;)
      {
        const hk_moe::mps::config mrole = hk_moe::mps::decode_config(
            (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
        T2V6_PROF_ONLY(t2v6_is_svc =
                           hk_moe::mps::is_service_cta(mrole, bid, nct);)
        if (!hk_moe::mps::is_service_cta(mrole, bid, nct)) {
          production_fused_moe::n2::n2p6t2b_phase2b_body(
              (const std::uint8_t*)k0p6_dread(desc, K0P6_D_A2Q),   // dZq
              (const float*)k0p6_dread(desc, K0P6_D_DQ2),          // DQdZ
              (const std::uint8_t*)k0p6_dread(desc, K0P6_D_W2),    // = W13T
              (const float*)k0p6_dread(desc, K0P6_D_S2),           // = S13T
              (const int*)k0p6_dread(desc, K0P6_D_STI),
              (const float*)k0p6_dread(desc, K0P6_D_SWT),
              (const int*)k0p6_dread(desc, K0P6_D_SEI),
              (const int*)k0p6_dread(desc, K0P6_D_NVI),
              (__hip_bfloat16*)k0p6_dread(desc, K0P6_D_PART),
              (const int*)k0p6_dread(desc, K0P6_D_TILE_DESC), ntiles, desc,
              t_lo, t_hi);
        } else {
          if (slab == 1) {
            // ---- pool front-half combine sweep (RMW arm, donor verbatim) --
            const unsigned int epoch32 = (unsigned int)k0p6_epoch(desc);
            unsigned int* row_ready =
                (unsigned int*)k0p6_dread(desc, K0P6_D_ROW_READY);
            bool cells_ok = true;
            if (tid == 0) {
              for (int p = 0; p < world; ++p) {
                cells_ok = hk_moe::poll_epoch_system(
                               row_ready + (size_t)p * T_loc_max +
                                   (size_t)T_ext + 0,
                               epoch32, (std::uint64_t)spin_limit,
                               hk_moe::set_error_bit{pperr, 33554432}) &&
                           cells_ok;
                if (!cells_ok) break;
              }
            }
            __syncthreads();
            if (!hk_moe::error_bit_set_agent(
                    pperr, 16777216 | 2097152 | 33554432 |
                               K0P6_MPS_ERR_CONFIG)) {
              const int T = (int)k0p6_dread(desc, K0P6_D_T);
              const int* pull_ptr =
                  (const int*)k0p6_dread(desc, K0P6_D_PULL_PTR);
              const int* pull_src =
                  (const int*)k0p6_dread(desc, K0P6_D_PULL_SRC);
              unsigned short* slots =
                  (unsigned short*)k0p6_dread(desc, K0P6_D_MPS_SLOTS);
              unsigned short* out =
                  (unsigned short*)k0p6_dread(desc, K0P6_D_OUT);
              std::uint32_t* mps_state =
                  (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
              std::uint32_t* m8_front = mps_state + K0P6_MPS_ST_M8NEXT;
              constexpr int NT = 4;
              const int nbatches = (T + NT - 1) / NT;
              // QUOTA, not exhaustion (the R1-hostage rule).
              const int quota = (int)hk_moe::mps::effective_flush_rows(
                  hk_moe::mps::decode_config(
                      (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG)));
              int claimed = 0;
              int* batch_slot = &s_ns[wid];
              while (claimed < quota) {
                if (lane == 0) {
                  *batch_slot =
                      kittens::distributed::fetch_add_relaxed<hk_moe::scope::agent>(
                          m8_front, 1u);
                }
                __syncwarp();
                const int batch = *batch_slot;
                if (batch >= nbatches) break;
                ++claimed;
                k0p6_m15_m8_batch<NT, 0, K0P6_M15_M8_CSPLIT>(
                    batch * NT, T, pull_ptr, pull_src, lane, cur, MAXTOK,
                    slots, out);
              }
            }
          }
#if K0P6_T2V6_FILLER
          // ---- v6 windows A/B: service-CTA wgrad filler -------------------
          // Slab 0: dW2 tiles only (dZq not yet globally final).  Slab 1:
          // dW2 leftovers then dW13.  Quota-bounded so the pool's R1 arrival
          // stays inside the compute shadow (the combine-sweep law).
          if (dwmode >= 2) {
            const int q = (slab == 0) ? dw_quota_a : dw_quota_b;
            if (q > 0) {
              T2V6_PROF_ONLY(std::uint64_t t2v6_tf = 0;
                             if (tid == 0) t2v6_tf =
                                 hk_moe::mps::realtime_now();)
              t2v6::fill_ctx wgc;
              t2v6::bind_ctx(desc, wgc, dw_accumulate, dw_em);
              const int ran = t2v6::window(wgc, q, slab == 1);
              T2V6_PROF_ONLY(if (tid == 0 && ran > 0) {
                atomicAdd((unsigned long long*)(t2v6_prof +
                                                (slab == 0 ? T2V6_PB_FILL_A
                                                           : T2V6_PB_FILL_B)),
                          (unsigned long long)ran);
                atomicAdd((unsigned long long*)(t2v6_prof + T2V6_PB_FILL_CYC),
                          (unsigned long long)(hk_moe::mps::realtime_now() -
                                               t2v6_tf));
              })
              (void)ran;
            }
          }
#endif
        }
      }
      // ---- slab rendezvous (one textual copy; per-CTA drain + grid barrier
      // + leader publication of the (rank, slab) word to all 8 peers) --------
      T2V6_PROF_ONLY(std::uint64_t t2v6_tr = 0;
                     if (tid == 0) t2v6_tr = hk_moe::mps::realtime_now();)
      hk_moe::release_cta_payload_system();
      {
        unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
        const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
        const hkp::fail_closed bar_err{pperr, 2097152};
        hkp::grid_barrier(bar, tid, bar_err);
      }
      T2V6_PROF_ONLY(if (tid == 0) {
        const int cell =
            (slab == 0) ? (t2v6_is_svc ? T2V6_PB_RDV0_S : T2V6_PB_RDV0_C)
                        : (t2v6_is_svc ? T2V6_PB_RDV1_S : T2V6_PB_RDV1_C);
        atomicAdd((unsigned long long*)(t2v6_prof + cell),
                  (unsigned long long)(hk_moe::mps::realtime_now() - t2v6_tr));
      })
      hk_moe::acquire_payload_agent();
      const bool payload_ok = !hk_moe::error_bit_set_agent(
          pperr, 16777216 | 2097152);
      // RMW arm: the epilogue's remote accumulates are fabric-visible after
      // the drain+writeback+barrier above, so the slab word certifies them
      // now.
      if (payload_ok && tid == 0 && bid == 0) {
        const auto* symmetric = k0p6_symmetric(desc);
        unsigned int* row_ready =
            (unsigned int*)k0p6_dread(desc, K0P6_D_ROW_READY);
        const unsigned int epoch32 = (unsigned int)k0p6_epoch(desc);
        hk_moe::release_signal_batch_system();
        unsigned int* self_slot =
            row_ready + (size_t)cur * T_loc_max + (size_t)T_ext + slab;
        for (int R = 0; R < world; ++R) {
          if (R == cur) {
            hk_moe::publish_epoch<hk_moe::scope::agent>(self_slot, epoch32);
          } else {
            hk_moe::publish_epoch<hk_moe::scope::system>(
                hk_moe::peer_ptr(self_slot, R, symmetric), epoch32);
          }
        }
      }
    }
  }
  if (debug_stop == 6) return;
  if (tid == 0) {
    std::uint32_t* mps_state7 =
        (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
    const hk_moe::mps::config cfg7 = hk_moe::mps::decode_config(
        (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
    std::uint64_t* ts7 = (std::uint64_t*)(mps_state7 + K0P6_MPS_ST_WORDS);
    hk_moe::mps::ts_mark(ts7 + K0P6_MPS_TS_M7_DONE, ts7, bid,
                         K0P6_MPS_E23_M7_DONE, cfg7.timestamps);
  }

  // ================= M8: coarse-certified combine, front then back =========================
  // Cross-rank wait = 2*world slab-word polls by tid0 + one CTA barrier.  The
  // v6 FILLER arm converts the wait into poll-and-fill: one NONBLOCKING sweep
  // of the 16 certificates, and while any is missing the whole CTA runs one
  // cursor-claimed wgrad tile, up to the window-C quota; the verbatim bounded
  // poll then runs regardless, so the fail-closed ladder (bit 25) and the
  // settled-certificate acquire order are the donor's exactly.
  {
    const unsigned int epoch32 = (unsigned int)k0p6_epoch(desc);
    const unsigned int* row_ready =
        (const unsigned int*)k0p6_dread(desc, K0P6_D_ROW_READY);
    T2V6_PROF_ONLY(std::uint64_t t2v6_t8 = 0;
                   if (tid == 0) t2v6_t8 = hk_moe::mps::realtime_now();)
#if K0P6_T2V6_FILLER
    if (dwmode >= 2 && dw_quota_c > 0) {
      t2v6::fill_ctx wgc;
      t2v6::bind_ctx(desc, wgc, dw_accumulate, dw_em);
      int claimed8 = 0;
      while (claimed8 < dw_quota_c) {
        if (tid == 0) {
          int ready = 1;
          for (int s = 0; s < K0P6_M15_SLABS && ready; ++s) {
            for (int p = 0; p < world; ++p) {
              if (hk_moe::load_relaxed<hk_moe::scope::system>(
                      row_ready + (size_t)p * T_loc_max + (size_t)T_ext +
                      s) != epoch32) {
                ready = 0;
                break;
              }
            }
          }
          *wgc.ldsClaim = ready;
        }
        __syncthreads();
        const int all_ready = *wgc.ldsClaim;
        __syncthreads();
        if (all_ready) break;
        // Not ready: run one tile (dW2 leftovers first, then dW13).
        ensure_ranges(wgc);
        int i = t2v6::claim(wgc.cur_dw2, wgc.n_experts * t2v6::kT2,
                            wgc.ldsClaim);
        if (i >= 0) {
          t2v6::run_dw2(wgc, i);
        } else {
          i = t2v6::claim(wgc.cur_dw13, wgc.n_experts * t2v6::kT13,
                          wgc.ldsClaim);
          if (i < 0) break;  // both spaces drained; fall to the bounded poll
          t2v6::run_dw13(wgc, i);
        }
        ++claimed8;
        T2V6_PROF_ONLY(if (tid == 0) atomicAdd(
            (unsigned long long*)(t2v6_prof + T2V6_PB_FILL_C), 1ull);)
      }
    }
#endif
    if (tid == 0) {
      for (int s = 0; s < K0P6_M15_SLABS; ++s) {
        for (int p = 0; p < world; ++p) {
          (void)hk_moe::poll_epoch_system(
              row_ready + (size_t)p * T_loc_max + (size_t)T_ext + s, epoch32,
              (std::uint64_t)spin_limit,
              hk_moe::set_error_bit{pperr, 33554432});
        }
      }
    }
    __syncthreads();
    T2V6_PROF_ONLY(if (tid == 0) {
      const std::uint64_t t1 = hk_moe::mps::realtime_now();
      atomicAdd((unsigned long long*)(t2v6_prof + T2V6_PB_M8WAIT),
                (unsigned long long)(t1 - t2v6_t8));
      atomicMax((unsigned long long*)(t2v6_prof + T2V6_PB_M8ENTER_MIN),
                (unsigned long long)(~t1));
    })
    const bool payload_ok = !hk_moe::error_bit_set_agent(
        pperr, 16777216 | 2097152 | 33554432 | K0P6_MPS_ERR_CONFIG);
    const int T = payload_ok ? (int)k0p6_dread(desc, K0P6_D_T) : 0;
    const int* pull_ptr = (const int*)k0p6_dread(desc, K0P6_D_PULL_PTR);
    const int* pull_src = (const int*)k0p6_dread(desc, K0P6_D_PULL_SRC);
    unsigned short* slots = (unsigned short*)k0p6_dread(desc, K0P6_D_MPS_SLOTS);
    unsigned short* out = (unsigned short*)k0p6_dread(desc, K0P6_D_OUT);
    std::uint32_t* mps_state = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
    std::uint64_t* mps_ts = (std::uint64_t*)(mps_state + K0P6_MPS_ST_WORDS);
    constexpr int NT = 4;
    const int nbatches = (T + NT - 1) / NT;
    // RMW arm: dynamic wave tickets.  FRONT cursor shared with the pool's
    // mid-slab-1 sweep; BACK cursor is the dead event cursor.
    std::uint32_t* m8_front = mps_state + K0P6_MPS_ST_M8NEXT;
    std::uint32_t* m8_back = mps_state + K0P6_MPS_ST_EVNEXT;
    int* batch_slot = &s_ns[wid];
    while (true) {
      if (lane == 0) {
        *batch_slot =
            kittens::distributed::fetch_add_relaxed<hk_moe::scope::agent>(
                m8_front, 1u);
      }
      __syncwarp();
      const int batch = *batch_slot;
      if (batch >= nbatches) break;
      k0p6_m15_m8_batch<NT, 0, K0P6_M15_M8_CSPLIT>(
          batch * NT, T, pull_ptr, pull_src, lane, cur, MAXTOK, slots, out);
    }
    while (true) {
      if (lane == 0) {
        *batch_slot =
            kittens::distributed::fetch_add_relaxed<hk_moe::scope::agent>(
                m8_back, 1u);
      }
      __syncwarp();
      const int batch = *batch_slot;
      if (batch >= nbatches) break;
      k0p6_m15_m8_batch<NT, K0P6_M15_M8_CSPLIT, 14>(
          batch * NT, T, pull_ptr, pull_src, lane, cur, MAXTOK, slots, out);
    }
    (void)nbatches;
    T2V6_PROF_ONLY(if (tid == 0) {
      atomicMax((unsigned long long*)(t2v6_prof + T2V6_PB_M8EXIT_MAX),
                (unsigned long long)hk_moe::mps::realtime_now());
    })
    if (tid == 0) {
      const hk_moe::mps::config m8cfg = hk_moe::mps::decode_config(
          (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
      hk_moe::mps::ts_mark(mps_ts + K0P6_MPS_TS_REDUCE_DONE, mps_ts, bid,
                           K0P6_MPS_E23_REDUCE_DONE, m8cfg.timestamps);
    }
  }

  // ================= M8.5: wgrad tail ======================================================
  // Mode 1: the v4-compat whole-space grid-stride overwrite (t2b verbatim).
  // Mode 2 (FILLER): drain whatever the windows left on the two cursors —
  // same one-CTA-per-tile store contract, M9's counted arrival is completion.
  if (dwmode == 1) {
    production_fused_moe::n2::n2p6t2b_wgrad_body(
        (const std::uint8_t*)k0p6_dread(desc, K0P6_D_A2Q),        // dZq
        (const float*)k0p6_dread(desc, K0P6_D_DQ2),               // DQdZ
        (const std::uint8_t*)k0p6_dread(desc, K0P6_D_T2B_ZQ),
        (const float*)k0p6_dread(desc, K0P6_D_T2B_DQZ),
        (const std::uint8_t*)k0p6_dread(desc, K0P6_D_T2B_XROWS),  // fwd x rows
        (const float*)k0p6_dread(desc, K0P6_D_T2B_XSC),
        (const std::uint8_t*)k0p6_dread(desc, K0P6_D_A_DST),      // dY rows
        (const float*)k0p6_dread(desc, K0P6_D_SC_STAGE),
        (const int*)k0p6_dread(desc, K0P6_D_STI),
        (const float*)k0p6_dread(desc, K0P6_D_SWT),
        (const int*)k0p6_dread(desc, K0P6_D_SEI),
        (const int*)k0p6_dread(desc, K0P6_D_NVI),
        (__hip_bfloat16*)k0p6_dread(desc, K0P6_D_T2B_DW13),
        (__hip_bfloat16*)k0p6_dread(desc, K0P6_D_T2B_DW2),
        (int)k0p6_dread(desc, K0P6_D_E), bid, nct,
        /*tile_lo=*/0, /*tile_hi=*/0, /*accumulate=*/false,
        reinterpret_cast<__hip_bfloat16*>(
            &production_fused_moe::n2::t2bsh_a2_lds[0][0]),
        reinterpret_cast<__hip_bfloat16*>(
            &production_fused_moe::n2::t2bsh_a_lds[0][0][0][0]),
        &production_fused_moe::n2::t2bsh_ascale_lds[0][0],
        reinterpret_cast<int*>(
            &production_fused_moe::n2::t2bsh_amax_lds[0][0]));
  }
#if K0P6_T2V6_FILLER
  else if (dwmode >= 2) {
    T2V6_PROF_ONLY(std::uint64_t t2v6_tt = 0;
                   if (tid == 0) t2v6_tt = hk_moe::mps::realtime_now();)
    t2v6::fill_ctx wgc;
    t2v6::bind_ctx(desc, wgc, dw_accumulate, dw_em);
    ensure_ranges(wgc);
    int drained = 0;
    while (true) {
      int i = t2v6::claim(wgc.cur_dw2, wgc.n_experts * t2v6::kT2,
                          wgc.ldsClaim);
      if (i < 0) break;
      t2v6::run_dw2(wgc, i);
      ++drained;
    }
    while (true) {
      int i = t2v6::claim(wgc.cur_dw13, wgc.n_experts * t2v6::kT13,
                          wgc.ldsClaim);
      if (i < 0) break;
      t2v6::run_dw13(wgc, i);
      ++drained;
    }
    T2V6_PROF_ONLY(if (tid == 0 && drained > 0) {
      atomicAdd((unsigned long long*)(t2v6_prof + T2V6_PB_FILL_TAIL),
                (unsigned long long)drained);
      atomicAdd((unsigned long long*)(t2v6_prof + T2V6_PB_FILL_CYC),
                (unsigned long long)(hk_moe::mps::realtime_now() - t2v6_tt));
    })
    (void)drained;
  }
#endif

  // ================= M9: counted arrival, resets, retired pokes (donor + scope fix) ========
  {
    __syncthreads();
    T2V6_PROF_ONLY(if (tid == 0) {
      atomicMax((unsigned long long*)(t2v6_prof + T2V6_PB_M9EXIT_MAX),
                (unsigned long long)hk_moe::mps::realtime_now());
    })
    if (tid == 0) {
      unsigned int* combine_done =
          (unsigned int*)k0p6_dread(desc, K0P6_D_COMB_DONE);
      const unsigned int old = __hip_atomic_fetch_add(combine_done, 1u, __ATOMIC_ACQ_REL,
                                                      __HIP_MEMORY_SCOPE_AGENT);
      if ((old % gridDim.x) == (gridDim.x - 1)) {
        const auto* symmetric_m9 = k0p6_symmetric(desc);
        const std::uint64_t epoch = k0p6_epoch(desc);
        std::uint64_t* retired =
            (std::uint64_t*)k0p6_dread(desc, K0P6_D_RETIRED);
        unsigned int* pushed_count = (unsigned int*)k0p6_dread(desc, K0P6_D_PUSHED);
        unsigned int* qpush_done = (unsigned int*)k0p6_dread(desc, K0P6_D_QPUSH_DONE);
        unsigned int* mega_count = (unsigned int*)k0p6_dread(desc, K0P6_D_EPOCH_CELL);
        for (int pe = 0; pe < world; ++pe) pushed_count[pe] = 0u;
        qpush_done[0] = 0u;
        combine_done[0] = 0u;
        mega_count[0] = (unsigned int)epoch;
        hk_moe::release_signal_batch_system();
        for (int pe = 0; pe < world; ++pe) {
          if (pe == cur) {
            hk_moe::publish_value<hk_moe::scope::agent>(
                &retired[cur], epoch);
          } else {
            std::uint64_t* peer =
                hk_moe::peer_ptr(retired + cur, pe, symmetric_m9);
            hk_moe::publish_value<hk_moe::scope::system>(
                peer, epoch);
          }
        }
      }
    }
  }
  (void)T_loc;
}

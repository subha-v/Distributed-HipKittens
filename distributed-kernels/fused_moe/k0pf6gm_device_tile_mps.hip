// ================================================================================================
// RUN PIN (ablations-tgen) — THIS BRANCH COMPILES THE M15 SLAB KERNEL UNDER THE
// HARNESS'S KERNEL NAME, WITH THE exp_04_tgen T-GENERALIZATION FIX IN THE
// CHASSIS (M2 hole-sentinel chunk ranges clamped to the segment). Never merge
// to ablations. Mode-12 control numbers must come from the flag-off ablations
// tree, never from this pin (the exp_38 rule generalized: this binary's .text
// is a DIFFERENT KERNEL).
// The body below is k0pf6gm_device_tile_m15.hip verbatim at this commit, with
// only the symbol name redirected; keeping the content inline (not #include)
// keeps it inside the mori JIT content hash (the exp_04 -I-not-hashed trap).
// ================================================================================================
#define K0P6_M15_KERNEL_NAME k0pf6gm_mps_mega
// ================================================================================================
// M15 SIBLING — "slab-certified pipelined combine" — UNMEASURED until the local
// validation gates pass on 8x gfx950. Built ADDITIVELY as its own translation
// unit next to k0pf6gm_device_tile_mps.hip (which is UNTOUCHED); design
// contract: overnight/aug12/M15_DESIGN.md. Donor for every phase M0-M6 and the
// M9 retire: the mps sibling at rev 26 .text 642646fc (the mode-12 depth-4
// ratchet, 6,482.7 us = 0.8408x production).
//
// Mechanism deltas vs the mode-12 ratchet (evidence-cited in M15_DESIGN.md):
//   * M7 task order is NC-MAJOR (task = nc*num_tiles + tile) and the task
//     space is cut into S=2 column slabs: slab 0 = nc 0..7 (output columns
//     [0,3584)), slab 1 = nc 8..15. Column-disjointness of the two slabs is
//     what makes mid-epoch combine sound: slab-1 epilogue RMWs never touch a
//     front-half slot byte. (exp_29's staircase; exp_35 rung (e).)
//   * After each slab, ONE M7.5-class rendezvous (per-CTA drain + one leader
//     L2 writeback + grid barrier) and ONE coarse epoch word per (rank,slab)
//     published to all 8 peers -- mode-14's protocol economics at S=2, parked
//     in the row_ready spare tail like mode 14's m7_done cells. No per-row
//     readiness protocol exists anywhere in this kernel.
//   * The C-pool finally has a job: during slab 1 it polls the 8 slab-0 words
//     and reduces FRONT-half combine batches (dynamic wave tickets) while the
//     compute CTAs run the back-half GEMM. M8 after the final rendezvous is
//     mode-14-shaped (Ready: no per-row polls, no ballot), split front/back.
//   * The M7 epilogue transport is mode 12's remote-accumulate VERBATIM,
//     including the g-encoded depth-4 vmcnt throttle and the deferred drain
//     (task t's drain paid at task t+1's head). The event queue is deleted:
//     the done hook buffers only the drain obligation and publishes nothing
//     (exp_34-C's measured-null shape).
//   * OPTIONAL, compile-gated K0P6_M15_STAGED=1 arm ("mode 15b"): the same
//     epilogue instruction stream is pointed at a LOCAL stage (only the
//     m7tab pointer VALUES change), intra-producer collisions fold locally,
//     and carrier CTAs push folded half-rows as wide posted packet stores
//     into the owners' slots (~1.44x fewer remote bytes than the RMW stream;
//     op class 0.127 vs 0.571 us/op on the measured issue ladder). Requires
//     one new descriptor slot (63) holding a host-zeroed [T_ext][7168] bf16
//     stage. Default build contains NONE of this code.
//
// pperr bits are the donor's; bit 25 (33554432) here means "slab-word
// cross-rank rendezvous timeout" (the mode-14 reuse rule: the per-row combine
// poll that owned this bit is compiled out of this kernel entirely).
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
#include "moe_mps_adapter.cuh"   // config word, MPS descriptor slots, role predicates

// JIT-cache guard (same contract as the mps sibling): this .hip IS content-
// hashed; bump on any .cuh-adjacent edit and confirm a fresh .hsaco mtime.
#define K0P6_M15_SRC_REV 2

// The staged wide-push transport arm (mode 15b). DEFAULT 0: the default build
// must contain zero staged code (exp_38: reachable-but-unused code near the
// M7 epilogue is how the ratchet lost +726.9 us).
#ifndef K0P6_M15_STAGED
#define K0P6_M15_STAGED 0
#endif

// M17: round-robin source-interleaved scatter. The donor scatter's atomic
// landing order is approximately source-banded, so consecutive positions in
// every expert block share an owner and the epilogue's depth-4 vmcnt window
// concentrates its outstanding RMWs on ONE XGMI link at a time (per-link
// bursts + head-of-line blocking under the throttle). Iterating pairs in a
// source-rotating permutation makes consecutive cursor positions land on
// rotating owners: the same depth spans ~7 links. Plan-side only — zero M7
// code touched, zero protocol. DEFAULT 0: the default arm's scatter call is
// the donor's, textually identical.
#ifndef K0P6_M15_SCATTER_RR
#define K0P6_M15_SCATTER_RR 0
#endif

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

// ---- descriptor offsets (MIRRORED by the host driver; identical to the mps sibling) ------
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
// MPS slots 56..62 come from moe_mps_adapter.cuh.
#if K0P6_M15_STAGED
// Staged arm only: local [world*MAXTOK][7168] bf16 stage, HOST-ZEROED before
// the first epoch (thereafter consume-and-zeroed by the push pass). Extends
// the descriptor length by one slot; the default arm needs no host change.
#define K0P6_D_M15_STAGE 63
#define K0P6_M15_D_LEN 64
#else
#define K0P6_M15_D_LEN K0P6_MPS_D_LEN
#endif

// Kernel symbol name, parametric so a RUN PIN branch can compile this body
// under the harness's expected name (the ablations-tbo16 pin pattern). The
// default is the sibling name; ablations-m15 sets it to k0pf6gm_mps_mega.
#ifndef K0P6_M15_KERNEL_NAME
#define K0P6_M15_KERNEL_NAME k0pf6gm_m15_mega
#endif

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
// exp_34-C measured this exact shape (drain kept, publication deleted) as a
// +3.2 us null relative to publishing -- and m15 has no drain phase to feed,
// so the event queue and its ~16.7k enqueues/rank/epoch are simply gone. The
// pend state is written uniformly by every thread so the flush's
// __syncthreads is convergent (donor discipline).
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
#define N2_P1_QUAL __device__ __forceinline__
#define N2_P1_NAME n2p6m15_phase1_body
#define N2_KERNELS_ONLY
#define N2_HOOK_CTX_ARG , const long long* __restrict__ k0p6_desc
#define N2GM_TASK_DONE_HOOK k0p6_a2_arrive(b, tid, k0p6_desc);
#define N2GM_P1_SCHED_GSCALE 0
#define N2GM_P1_ASCALE_TOKEN_MAJOR K0P6_MPS_ASCALE_TM
#include "n2_phase1_gm_mps.cpp"
#undef N2GM_P1_ASCALE_TOKEN_MAJOR
#undef N2GM_P1_SCHED_GSCALE
#undef N2GM_TASK_DONE_HOOK
#undef N2_HOOK_CTX_ARG
#undef N2_KERNELS_ONLY
#undef N2_P1_NAME
#undef N2_P1_QUAL

#define N2_P2_QUAL __device__ __forceinline__
#define N2_P2_NAME n2p6m15_phase2_body
#define N2_KERNELS_ONLY
// The slab bounds ride the hook-context extension point, so the shared body
// file needs no signature change (M15-DELTA (A) there).
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
// NC-MAJOR decode (M15-DELTA (A)): consecutive tasks share nc, tiles ascend.
#define N2GM_TASK_DECODE(t_, tile_, nc_, ntiles_) \
  const int nc = (t_) / (ntiles_);                \
  const int tile = (t_) - nc * (ntiles_);
#if K0P6_M15_STAGED
// Staged arm (M15-DELTA (B) in the shared body): identical epilogue
// instruction stream, LOCAL pointer values. tab[i] = stage + i*MAXTOK*H*2 so
// tab[xr>>sh] + (xr&mask)*H*2 lands on stage[r] for receive row r.
#define N2GM_M7TAB_FILL                                                     \
  const unsigned long long m15_stage =                                      \
      (unsigned long long)k0p6_dread(k0p6_desc, K0P6_D_M15_STAGE);          \
  const unsigned int m7_mtok =                                              \
      (unsigned int)k0p6_dread(k0p6_desc, K0P6_D_MAXTOK);                   \
  if (tid < 8) {                                                            \
    m7tab[tid] = m15_stage + (unsigned long long)tid *                      \
                                 (unsigned long long)m7_mtok *              \
                                 ((unsigned long long)kHidden * 2ull);      \
  }
#endif
#include "n2_phase2_gm_mps.cpp"
#if K0P6_M15_STAGED
#undef N2GM_M7TAB_FILL
#endif
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
#undef N2_P2_NAME
#undef N2_P2_QUAL
#undef N2GM_G

#if K0P6_M15_SCATTER_RR
// M17 scatter: the donor's stage-5 live-pair scatter (hkp_sort.hpp) with the
// iteration order permuted source-round-robin. Bijective over the pair space
// (q -> (src = q & 7, off = q >> 3); world == 8 is entry-guarded), same
// per-expert atomic cursors, same sti/swt packing, same overflow bit — only
// the atomic LANDING order (and therefore the owner sequence within each
// sorted expert block) changes.
__device__ __forceinline__ void k0p6_m15_scatter_rr(
    const int* __restrict__ recv_eid, const float* __restrict__ recv_wgt,
    int world, int MAXTOK, int TOPK, int lo, int E, int PADMAX,
    int* __restrict__ scratch, int* __restrict__ sti, float* __restrict__ swt,
    int* __restrict__ pperr, int ovf_bit) {
  const int rows = world * MAXTOK;
  const int pairs = rows * TOPK;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pairs;
       i += gridDim.x * blockDim.x) {
    const int q = i / TOPK;
    const int s = i - q * TOPK;
    const int src = q & (world - 1);
    const int off = q >> 3;
    const int row = src * MAXTOK + off;
    const int j = row * TOPK + s;
    const int e = recv_eid[j];
    if (e < lo || e >= lo + E) continue;
    const int pos = atomicAdd(&scratch[K0P6_SC_CUR + (e - lo)], 1);
    if (pos >= PADMAX) {
      atomicOr(pperr, ovf_bit);
      continue;
    }
    sti[pos] = (row & 0x00FFFFFF) | (s << 24);
    swt[pos] = recv_wgt[j];
  }
}
#endif

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

// One M8 combine batch over chunk range [CLO, CHI) -- the mode-14 "Ready"
// shape specialized for m15: owner-LOCAL slot reads, coarse readiness already
// settled by the slab words (no per-row poll, no ballot), consume-and-zero
// always on (mode-12 slot lifetime). Arithmetic is byte-stable with the donor
// batch body; only the chunk bounds and the inlined slot base differ.
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
      // Out-of-fanout lanes still need a dereferenceable base (the consumer
      // shuffles all 8 group lanes unconditionally); (cur, row=0) names this
      // rank's own slots[0] under the slot address form, bytes discarded.
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
  // The acquire half of the slab rendezvous stays with the reading thread
  // (the mode-14 discipline: the poll was the cost, never the acquire).
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
            // Consume-and-zero, same geometry as the read (mode-12 slot
            // lifetime: M9's retirement orders this before any peer's
            // next-epoch accumulate). Dummy bases excluded by the guard.
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
  // M15 guard: mode-12 (direct-accumulate) config only; power-of-two MAXTOK
  // (epilogue shift/mask); TWO spare slab cells per source segment in the
  // row_ready tail (indices T_ext and T_ext+1 -- the mode-14 parking rule,
  // one wider); every referenced MPS buffer present.
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
#if K0P6_M15_STAGED
        || k0p6_dread(desc, K0P6_D_M15_STAGE) == 0 ||
        (k0p6_dread(desc, K0P6_D_M15_STAGE) & 15) != 0
#endif
        ) {
      if (bid == 0 && tid == 0) atomicOr(pperr, K0P6_MPS_ERR_CONFIG);
      return;
    }
  }
  const int T_ext = world * MAXTOK;
  const int lo = cur * E;
  const long long debug_stop = k0p6_dread(desc, K0P6_D_DEBUG);

  // ================= M0: retire wait + per-block counter zero (donor verbatim,
  // minus the dead event-queue/arrival/claim zeroing -- m15 has no consumers
  // for them; mps_state IS zeroed: it carries the M8 wave tickets) ============
  {
    const std::uint64_t epoch = k0p6_epoch(desc);
    std::uint64_t* retired =
        (std::uint64_t*)k0p6_dread(desc, K0P6_D_RETIRED);
    if (tid < world) {
      (void)hk_moe::poll_value_at_least_system(
          retired + tid, epoch > 0 ? epoch - 1ull : 0ull,
          (std::uint64_t)spin_limit, pperr, 16777216);
    }
    {
      unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
      const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
      const hkp::fail_closed bar_err{pperr, 2097152};
      hkp::grid_barrier(bar, tid, bar_err);
    }
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

  // ================= M1: qpush dispatch (donor verbatim) ====================================
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

    const int gw = bid * (blockDim.x >> 6) + wid;
    const int nw = nct * (blockDim.x >> 6);
    for (int tau = gw; tau < T; tau += nw) {
      const int s = lane;
      const int eid = (s < TOPK) ? my_ids[(size_t)tau * TOPK + s] : -1;
      const float wgt = (s < TOPK) ? my_wgt[(size_t)tau * TOPK + s] : 0.0f;
      const int dest = (eid >= 0) ? eid / E : -1;
      const unsigned long long match = __match_any_sync(__activemask(), dest);
      const bool primary = (s < TOPK) && ((match & ((1ULL << s) - 1ULL)) == 0ULL);
      const unsigned long long pmask = __ballot(primary);
      const int fanout = __popcll(pmask);

      unsigned int row = 0u;
      if (primary) {
        unsigned int pos;
        if (dest == cur) {
          pos = hk_moe::reserve_row<hk_moe::scope::agent>(
              dest_counter + cur);
        } else {
          unsigned int* peer =
              hk_moe::peer_ptr(dest_counter + cur, dest, symmetric);
          pos = hk_moe::reserve_row<hk_moe::scope::system>(
              peer);
        }
        if (pos >= (unsigned int)MAXTOK) atomicOr(pperr, 65536);
        row = (unsigned int)cur * (unsigned int)MAXTOK + pos;
      }
      if (s < TOPK) {
        pull_stage[((size_t)tau * TOPK + s) * 2 + 0] = primary ? dest : -1;
        pull_stage[((size_t)tau * TOPK + s) * 2 + 1] = primary ? (int)row : -1;
      }
      if (lane == 0) pull_cnt[tau] = fanout;
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

    for (int i = tid; i < E; i += blockDim.x) s_cnt[i] = 0;
    if (tid == 0) s_total = 0u;
    __syncthreads();

    const int g = lane >> 3, j = lane & 7;
    const int wave = bid * (blockDim.x >> 6) + (tid >> 6);
    const int wave_count = nct * (blockDim.x >> 6);

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
          // exp_04_tgen (aug13) T-GENERALIZATION FIX: chunk c's rows live in
          // [c*CHUNK, min((c+1)*CHUNK, MAXTOK)) of segment s. The unclamped
          // (c+1)*CHUNK end was only in-segment because NCHUNK_MAX*CHUNK ==
          // MAXTOK at T=4096. At MAXTOK<4096 the phantom chunks (fill=0)
          // mapped past the segment end onto FOLLOWING segments' LIVE rows,
          // spraying recv_eid=-1 over every segment except s=0 (exp_36's
          // T=1024/2048 failure). Clamping keeps hole writes inside their own
          // segment; at MAXTOK=4096 seg_row1 == (c+1)*CHUNK for all 8 chunks,
          // so the written ranges are IDENTICAL to the pre-fix kernel's.
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
    for (int i = tid; i < E; i += blockDim.x) hcnt[i * nct + bid] = s_cnt[i];
  }
  if (debug_stop == 3) return;

  // ================= M3-M5: plan + scatter (donor verbatim) ================================
  {
    unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
    const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
    const hkp::fail_closed bar_err{pperr, 2097152};
    const int T = (int)k0p6_dread(desc, K0P6_D_T);
    int* scratch = (int*)k0p6_dread(desc, K0P6_D_SCRATCH);
    int* nvi = (int*)k0p6_dread(desc, K0P6_D_NVI);
    int* sei = (int*)k0p6_dread(desc, K0P6_D_SEI);
    const int* pull_cnt = (const int*)k0p6_dread(desc, K0P6_D_PULL_CNT);
    int* pull_ptr = (int*)k0p6_dread(desc, K0P6_D_PULL_PTR);
    const float* sc_stage = (const float*)k0p6_dread(desc, K0P6_D_SC_STAGE);
    float* sc_dst = (float*)k0p6_dread(desc, K0P6_D_SC_DST);
    unsigned short* part = (unsigned short*)k0p6_dread(desc, K0P6_D_PART);
    int* sti = (int*)k0p6_dread(desc, K0P6_D_STI);
    float* swt = (float*)k0p6_dread(desc, K0P6_D_SWT);
    const int* recv_eid = (const int*)k0p6_dread(desc, K0P6_D_RECV_EID);
    const float* recv_wgt = (const float*)k0p6_dread(desc, K0P6_D_RECV_WGT);
    const int* pull_stage = (const int*)k0p6_dread(desc, K0P6_D_PULL_STAGE);
    int* pull_src = (int*)k0p6_dread(desc, K0P6_D_PULL_SRC);
    const int* hcnt = (const int*)k0p6_dread(desc, K0P6_D_HCNT);

    hkp::grid_barrier(bar, tid, bar_err);   // M3
    if (hk_moe::error_bit_set_agent(pperr, 2097152)) return;

    if (bid == 0) {
      k0p6_sort::count_reduce(hcnt, E, nct, s_cnt, tid);
      k0p6_sort::scan(s_cnt, E, T_ext, scratch, nvi, tid);
      k0p6_sort::cursors_sei(scratch, E, PADMAX, sei, pperr, 4194304, tid);
      __syncthreads();
      if (tid == 0) {
        int* tile_desc = (int*)k0p6_dread(desc, K0P6_D_TILE_DESC);
        int* num_tiles_p = (int*)k0p6_dread(desc, K0P6_D_NUM_TILES);
        const int G = K0P6GM_G;
        const int bcap = PADMAX / 32;
        int nt = 0;
        for (int e = 0; e < E; ++e) {
          const int blk0 = scratch[K0P6_SC_ERB + e] / 32;
          const int blk1 = scratch[K0P6_SC_ERB + e + 1] / 32;
          const int nsb = blk1 - blk0;
          for (int t = 0; t * G < nsb; ++t) {
            const int b0 = blk0 + t * G;
            int gc = nsb - t * G;
            if (gc > G) gc = G;
            if (nt >= bcap) { atomicOr(pperr, 4194304); break; }
            tile_desc[nt++] = (b0 << 4) | (gc & 0xF);
          }
        }
        num_tiles_p[0] = nt;
        num_tiles_p[1] = G;
      }
    } else if (bid == 1) {
      __shared__ int s_tmp[256];
      __shared__ int s_carry;
      hkp::csr_scan_block256(pull_cnt, pull_ptr, T, tid, s_tmp, &s_carry);
    } else {
      const int lane2 = tid & 63;
      const int w = (bid - 2) * (blockDim.x >> 6) + (tid >> 6);
      const int nw2 = (nct - 2) * (blockDim.x >> 6);
      const hk_moe::mps::config cfg4 = hk_moe::mps::decode_config(
          (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
#if K0P6_MPS_ASCALE_TM
      if (!hk_moe::mps::skip_dead_part_zero(cfg4)) {
        for (int t = w; t < T_ext; t += nw2) {
          hkp::zero_part_scale_transpose<14>(part + (std::size_t)t * K0P6_H, sc_dst,
                                             sc_stage + (std::size_t)t * K0P6_NG, t, T_ext,
                                             K0P6_NG, lane2);
        }
      }
#else
      if (hk_moe::mps::skip_dead_part_zero(cfg4)) {
        for (int t = w; t < T_ext; t += nw2) {
          hk_moe::mps::scale_transpose_row(sc_dst,
                                           sc_stage + (std::size_t)t * K0P6_NG, t,
                                           T_ext, K0P6_NG, lane2);
        }
      } else {
        for (int t = w; t < T_ext; t += nw2) {
          hkp::zero_part_scale_transpose<14>(part + (std::size_t)t * K0P6_H, sc_dst,
                                             sc_stage + (std::size_t)t * K0P6_NG, t, T_ext,
                                             K0P6_NG, lane2);
        }
      }
#endif
    }

    hkp::grid_barrier(bar, tid, bar_err);   // M4
    if (hk_moe::error_bit_set_agent(pperr, 2097152)) return;

    const int pairs = T_ext * TOPK;
    k0p6_sort::pad(scratch, E, T_ext, TOPK, PADMAX, sti, swt, pperr, 4194304);
#if K0P6_M15_SCATTER_RR
    k0p6_m15_scatter_rr(recv_eid, recv_wgt, world, MAXTOK, TOPK, lo, E, PADMAX,
                        scratch, sti, swt, pperr, 1048576);
    (void)pairs;
#else
    k0p6_sort::scatter(recv_eid, recv_wgt, pairs, lo, E, TOPK, PADMAX, scratch, sti, swt,
                       pperr, 1048576);
#endif
    hkp::pull_src_fill(pull_stage, pull_ptr, T, TOPK, T * world, pull_src, pperr, 1048576);

    hkp::grid_barrier(bar, tid, bar_err);   // M5
    if (hk_moe::error_bit_set_agent(pperr, 2097152)) return;
    if (tid == 0) {
      std::uint32_t* mps_state5 =
          (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
      const hk_moe::mps::config cfg5 = hk_moe::mps::decode_config(
          (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
      std::uint64_t* ts5 = (std::uint64_t*)(mps_state5 + K0P6_MPS_ST_WORDS);
      hk_moe::mps::ts_mark(ts5 + K0P6_MPS_TS_M5_DONE, ts5, bid,
                           K0P6_MPS_E23_M5_DONE, cfg5.timestamps);
    }
  }
  if (debug_stop == 4) return;

  // ================= M6: N2 phase1 (donor verbatim) =========================================
  production_fused_moe::n2::n2p6m15_phase1_body(
      (const std::uint8_t*)k0p6_dread(desc, K0P6_D_A_DST),
#if K0P6_MPS_ASCALE_TM
      (const float*)k0p6_dread(desc, K0P6_D_SC_STAGE),
#else
      (const float*)k0p6_dread(desc, K0P6_D_SC_DST),
#endif
      (const std::uint8_t*)k0p6_dread(desc, K0P6_D_W13),
      (const float*)k0p6_dread(desc, K0P6_D_S13),
      (const int*)k0p6_dread(desc, K0P6_D_STI),
      (const int*)k0p6_dread(desc, K0P6_D_SEI),
      (const int*)k0p6_dread(desc, K0P6_D_NVI),
      (std::uint8_t*)k0p6_dread(desc, K0P6_D_A2Q),
      (float*)k0p6_dread(desc, K0P6_D_DQ2),
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

  // ================= M7: NC-MAJOR slab loop + slab-certified rendezvous ====================
  // One textual copy of the rendezvous sequence, executed once per slab (the
  // exp_38/exp_34 lesson: a SECOND copy of release/barrier/acquire pushed
  // whole-function allocation over a spill cliff; a loop keeps one copy).
  //
  // Pool job (RMW arm): during slab 1 the C service CTAs poll the 8 slab-0
  // words once (bounded) and reduce FRONT-half combine batches on dynamic
  // wave tickets -- the first mid-M7 combine start in this kernel family.
  // Front-half slot bytes (columns [0,3584)) are column-disjoint from every
  // slab-1 epilogue RMW (nc >= 8 writes columns >= 3584), so no ordering
  // beyond the slab-0 words is needed.
  {
    for (int slab = 0; slab < K0P6_M15_SLABS; ++slab) {
      const int ntiles =
          ((const int*)k0p6_dread(desc, K0P6_D_NUM_TILES))[0];
      const int t_lo = slab * K0P6_M15_NC_PER_SLAB * ntiles;
      const int t_hi = t_lo + K0P6_M15_NC_PER_SLAB * ntiles;
      {
        const hk_moe::mps::config mrole = hk_moe::mps::decode_config(
            (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
        if (!hk_moe::mps::is_service_cta(mrole, bid, nct)) {
          production_fused_moe::n2::n2p6m15_phase2_body(
              (const std::uint8_t*)k0p6_dread(desc, K0P6_D_A2Q),
              (const float*)k0p6_dread(desc, K0P6_D_DQ2),
              (const std::uint8_t*)k0p6_dread(desc, K0P6_D_W2),
              (const float*)k0p6_dread(desc, K0P6_D_S2),
              (const int*)k0p6_dread(desc, K0P6_D_STI),
              (const float*)k0p6_dread(desc, K0P6_D_SWT),
              (const int*)k0p6_dread(desc, K0P6_D_SEI),
              (const int*)k0p6_dread(desc, K0P6_D_NVI),
              (__hip_bfloat16*)k0p6_dread(desc, K0P6_D_PART),
              (const int*)k0p6_dread(desc, K0P6_D_TILE_DESC), ntiles, desc,
              t_lo, t_hi);
        } else if (slab == 1) {
#if !K0P6_M15_STAGED
          // ---- pool front-half combine sweep (RMW arm) --------------------
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
            // QUOTA, not exhaustion: an unbounded sweep would hold the R1
            // barrier hostage once slab-1 compute finishes (the pool covers
            // only ~20% of the front half per exp_29's 6.1 GB/s/CTA law;
            // draining all 1,024 batches would outlast slab 1 several times
            // over). flush_rows (default 16) batches per wave keeps the
            // pool's R1 arrival inside the compute shadow; leftovers are
            // claimed post-R1 by the whole grid on the same cursor.
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
#else
          // ---- pool front-half stage push (staged arm) --------------------
          // Ticketed 8-row quanta of FRONT halves (columns [0,3584)) of live
          // receive rows, wide posted packet stores into owner slots.
          // Liveness from the settled chunk_ready fills (dense-prefix
          // segments; NEVER row_remaining -- its holes are stale, the
          // exp_30 landmine).
          const auto* symmetric = k0p6_symmetric(desc);
          const std::uint64_t* chunk_ready =
              (const std::uint64_t*)k0p6_dread(desc, K0P6_D_CHUNK_READY);
          if (tid < world) {
            int ns = 0;
            for (int c = 0; c < K0P6C_NCHUNK_MAX; ++c)
              ns += (int)(unsigned int)(hk_moe::load_relaxed<hk_moe::scope::agent>(
                        chunk_ready + (size_t)tid * K0P6C_NCHUNK_MAX + c) &
                    0xffffffffu);
            s_ns[tid] = ns;
          }
          __syncthreads();
          const unsigned char* stage =
              (const unsigned char*)k0p6_dread(desc, K0P6_D_M15_STAGE);
          unsigned short* slots =
              (unsigned short*)k0p6_dread(desc, K0P6_D_MPS_SLOTS);
          std::uint32_t* mps_state =
              (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
          std::uint32_t* push_front = mps_state + K0P6_MPS_ST_TAIL;
          const int nquanta = (T_ext + 7) / 8;
          // Same quota logic as the RMW pool sweep: never let the push hold
          // the R1 barrier hostage. flush_rows*8 quanta per pool CTA
          // (default 16 -> 128 quanta = 1,024 rows/CTA) sits inside the
          // slab-1 shadow at the measured per-CTA push rates.
          const int push_quota = 8 * (int)hk_moe::mps::effective_flush_rows(
              hk_moe::mps::decode_config(
                  (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG)));
          int pushed_q = 0;
          __shared__ int s_q;
          while (pushed_q < push_quota) {
            if (tid == 0) {
              s_q = (int)kittens::distributed::fetch_add_relaxed<hk_moe::scope::agent>(
                  push_front, 1u);
            }
            __syncthreads();
            const int q = s_q;
            __syncthreads();
            if (q >= nquanta) break;
            ++pushed_q;
            for (int r = q * 8; r < q * 8 + 8 && r < T_ext; ++r) {
              const int owner = r / MAXTOK;
              const int pos = r - owner * MAXTOK;
              if (pos >= s_ns[owner]) continue;
              const std::byte* src = reinterpret_cast<const std::byte*>(
                  stage + (size_t)r * (size_t)K0P6_H * 2u);
              std::byte* dst_local = reinterpret_cast<std::byte*>(
                  slots + ((size_t)cur * (size_t)MAXTOK + (size_t)pos) *
                              (size_t)K0P6_H);
              std::byte* dst = (owner == cur)
                  ? dst_local
                  : hk_moe::peer_ptr(dst_local, owner, symmetric);
              kittens::distributed::store_peer_packets(
                  dst, src, (size_t)K0P6_M15_M8_CSPLIT * 1024u,
                  (unsigned int)tid, (unsigned int)blockDim.x);
            }
          }
#endif
        }
      }
      // ---- slab rendezvous (one textual copy; per-CTA drain + grid barrier
      // + leader publication of the (rank, slab) word to all 8 peers) --------
      hk_moe::release_cta_payload_system();
      {
        unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
        const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
        const hkp::fail_closed bar_err{pperr, 2097152};
        hkp::grid_barrier(bar, tid, bar_err);
      }
      hk_moe::acquire_payload_agent();
      const bool payload_ok = !hk_moe::error_bit_set_agent(
          pperr, 16777216 | 2097152);
#if !K0P6_M15_STAGED
      // RMW arm: the epilogue's remote accumulates are fabric-visible after
      // the drain+writeback+barrier above, so the slab word certifies them
      // now. (Staged arm publishes after its push pass instead -- below.)
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
#else
      (void)payload_ok;
#endif
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

#if K0P6_M15_STAGED
  // ================= M7.7 (staged arm only): remaining pushes + R2 + publish ===============
  // All CTAs: finish FRONT-half quanta (same ticket the pool used), then push
  // BACK halves grid-strided, zeroing each stage row after its back half is
  // read (consume-and-zero; ordering to the next epoch is M9's retirement
  // edge, the mode-12 slot discipline verbatim). Then one grid barrier and
  // ONE leader publication of BOTH slab words.
  {
    const auto* symmetric = k0p6_symmetric(desc);
    const std::uint64_t* chunk_ready =
        (const std::uint64_t*)k0p6_dread(desc, K0P6_D_CHUNK_READY);
    if (tid < world) {
      int ns = 0;
      for (int c = 0; c < K0P6C_NCHUNK_MAX; ++c)
        ns += (int)(unsigned int)(hk_moe::load_relaxed<hk_moe::scope::agent>(
                  chunk_ready + (size_t)tid * K0P6C_NCHUNK_MAX + c) &
              0xffffffffu);
      s_ns[tid] = ns;
    }
    __syncthreads();
    unsigned char* stage =
        (unsigned char*)k0p6_dread(desc, K0P6_D_M15_STAGE);
    unsigned short* slots =
        (unsigned short*)k0p6_dread(desc, K0P6_D_MPS_SLOTS);
    std::uint32_t* mps_state =
        (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
    std::uint32_t* push_front = mps_state + K0P6_MPS_ST_TAIL;
    const int nquanta = (T_ext + 7) / 8;
    __shared__ int s_q2;
    while (true) {
      if (tid == 0) {
        s_q2 = (int)kittens::distributed::fetch_add_relaxed<hk_moe::scope::agent>(
            push_front, 1u);
      }
      __syncthreads();
      const int q = s_q2;
      __syncthreads();
      if (q >= nquanta) break;
      for (int r = q * 8; r < q * 8 + 8 && r < T_ext; ++r) {
        const int owner = r / MAXTOK;
        const int pos = r - owner * MAXTOK;
        if (pos >= s_ns[owner]) continue;
        const std::byte* src = reinterpret_cast<const std::byte*>(
            stage + (size_t)r * (size_t)K0P6_H * 2u);
        std::byte* dst_local = reinterpret_cast<std::byte*>(
            slots + ((size_t)cur * (size_t)MAXTOK + (size_t)pos) *
                        (size_t)K0P6_H);
        std::byte* dst = (owner == cur)
            ? dst_local
            : hk_moe::peer_ptr(dst_local, owner, symmetric);
        kittens::distributed::store_peer_packets(
            dst, src, (size_t)K0P6_M15_M8_CSPLIT * 1024u,
            (unsigned int)tid, (unsigned int)blockDim.x);
      }
    }
    // Back halves, grid-strided (all CTAs are present post-R1 -- no ticket
    // needed). Stage zeroing does NOT happen here: another CTA's front-half
    // packet loads of the same row may still be in flight until the R2
    // drain+barrier below proves every push's reads retired grid-wide.
    for (int r = bid; r < T_ext; r += nct) {
      const int owner = r / MAXTOK;
      const int pos = r - owner * MAXTOK;
      if (pos >= s_ns[owner]) continue;
      const std::byte* src = reinterpret_cast<const std::byte*>(
          stage + (size_t)r * (size_t)K0P6_H * 2u +
          (size_t)K0P6_M15_M8_CSPLIT * 1024u);
      std::byte* dst_local = reinterpret_cast<std::byte*>(
          slots + ((size_t)cur * (size_t)MAXTOK + (size_t)pos) *
                      (size_t)K0P6_H);
      std::byte* dst = (owner == cur)
          ? dst_local
          : hk_moe::peer_ptr(dst_local, owner, symmetric);
      kittens::distributed::store_peer_packets(
          dst + (size_t)K0P6_M15_M8_CSPLIT * 1024u, src,
          (size_t)K0P6_H * 2u - (size_t)K0P6_M15_M8_CSPLIT * 1024u,
          (unsigned int)tid, (unsigned int)blockDim.x);
    }
    kittens::distributed::producer_drain_release<kittens::distributed::memory_scope::system>();
    {
      unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
      const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
      const hkp::fail_closed bar_err{pperr, 2097152};
      hkp::grid_barrier(bar, tid, bar_err);   // R2
    }
    hk_moe::acquire_payload_agent();
    // Consume-and-zero of the stage, post-R2: every push's loads have retired
    // grid-wide (each CTA's producer_drain_release precedes the barrier), so
    // zeroing cannot race an in-flight packet read. Local stores; ordering
    // against the next epoch's first stage accumulate is M9's retirement
    // edge, the mode-12 slot-lifetime discipline verbatim. First epoch: the
    // host zeroes the stage at allocation.
    for (int r = bid; r < T_ext; r += nct) {
      const int owner = r / MAXTOK;
      const int pos = r - owner * MAXTOK;
      if (pos >= s_ns[owner]) continue;
      unsigned char* zrow = stage + (size_t)r * (size_t)K0P6_H * 2u;
      for (size_t i = (size_t)tid * 16u; i < (size_t)K0P6_H * 2u;
           i += (size_t)blockDim.x * 16u) {
        uint4 z4;
        z4.x = 0u; z4.y = 0u; z4.z = 0u; z4.w = 0u;
        *reinterpret_cast<uint4*>(zrow + i) = z4;
      }
    }
    const bool push_ok = !hk_moe::error_bit_set_agent(
        pperr, 16777216 | 2097152);
    if (push_ok && tid == 0 && bid == 0) {
      unsigned int* row_ready =
          (unsigned int*)k0p6_dread(desc, K0P6_D_ROW_READY);
      const unsigned int epoch32 = (unsigned int)k0p6_epoch(desc);
      hk_moe::release_signal_batch_system();
      for (int slab = 0; slab < K0P6_M15_SLABS; ++slab) {
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
#endif

  // ================= M8: coarse-certified combine, front then back =========================
  // Cross-rank wait = 2*world slab-word polls by tid0 + one CTA barrier (the
  // mode-14 rendezvous shape: sequential bounded polls that become
  // satisfiable at the same instant cost nothing extra). No per-row polls,
  // no ballot; bit 25 is the whole fail-closed path.
  {
    const unsigned int epoch32 = (unsigned int)k0p6_epoch(desc);
    const unsigned int* row_ready =
        (const unsigned int*)k0p6_dread(desc, K0P6_D_ROW_READY);
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
#if !K0P6_M15_STAGED
    // RMW arm: dynamic wave tickets. The FRONT cursor is the one the pool has
    // been claiming from since mid-slab-1 -- leftovers only here. The BACK
    // cursor is a fresh cell (the dead event cursor).
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
#else
    // Staged arm: all CTAs arrive together post-R2; static wave stripes.
    {
      const int gw = bid * (blockDim.x >> 6) + wid;
      const int nw = nct * (blockDim.x >> 6);
      for (int tok0 = gw * NT; tok0 < T; tok0 += nw * NT) {
        k0p6_m15_m8_batch<NT, 0, K0P6_M15_M8_CSPLIT>(
            tok0, T, pull_ptr, pull_src, lane, cur, MAXTOK, slots, out);
      }
      for (int tok0 = gw * NT; tok0 < T; tok0 += nw * NT) {
        k0p6_m15_m8_batch<NT, K0P6_M15_M8_CSPLIT, 14>(
            tok0, T, pull_ptr, pull_src, lane, cur, MAXTOK, slots, out);
      }
    }
#endif
    (void)nbatches;
    if (tid == 0) {
      const hk_moe::mps::config m8cfg = hk_moe::mps::decode_config(
          (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
      hk_moe::mps::ts_mark(mps_ts + K0P6_MPS_TS_REDUCE_DONE, mps_ts, bid,
                           K0P6_MPS_E23_REDUCE_DONE, m8cfg.timestamps);
    }
  }

  // ================= M9: counted arrival, resets, retired pokes (donor + scope fix) ========
  {
    __syncthreads();
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
        // M15 scope fix (exp_29 protocol review, KRN:1848 class): the
        // consumers of `retired` are SYSTEM-scope pollers on peer GPUs; the
        // donor's agent-scope release issues no cross-device writeback and
        // was only ever saved by the uncached heap. Priced at ~+22 us inside
        // a +/-40 band by exp_21 mode 9 -- taken for soundness.
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
}

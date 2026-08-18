// ================================================================================================
// T3 — the COUNTER-DATAFLOW backward megakernel (the MoK synthesis).
// Additive sibling of t2b/t2v6; design record: overnight/aug18/
// T3_COUNTER_DATAFLOW_DESIGN.md; ABI: k0pf6gm_t3_abi.hpp.
//
// The t2b skeleton executed six grid barriers and two slab rendezvous per
// launch; every measured pathology of the v6 campaign (residency standoffs,
// tiny filler bubbles, thrice-realized cross-rank drift, microbatch-boundary
// export) traces to that lockstep phase structure.  T3 keeps ONE grid
// barrier (sealing M0's retire wait + counter zeroing) and replaces every
// other ordering edge with flat HBM counters at the consumer's frontier:
//
//   task families, in global claim order (atomic cursor per family):
//     F0  M1b dY dispatch            all-CTA cooperative, saved plan (verbatim)
//     F1  z-regen (tile, g) tasks    NO dependencies — they absorb the
//                                    dispatch flight time and cross-rank
//                                    drift that M2's barrier used to realize
//     F2  unpack (source, chunk)     gate: chunk_ready[s][c] (exists today);
//                                    signal: unpack_done[s] counted arrival
//     F3  phase-1b (tile, g) tasks   gate: z_done[same ticket] (p1z and p1b
//                                    share the identical task space, so the
//                                    z edge is 1:1) + unpack_done[s]==8 for
//                                    every source in the tile's sti window;
//                                    signal: a2_done[b] (existing hook)
//     F4  phase-2b slab-0 tasks      gate: in-body a2_done>=8 (existing);
//                                    completion: counted arrival -> the LAST
//                                    task's finisher publishes the slab-0
//                                    epoch word to all 8 peers (M9's
//                                    counted-arrival visibility argument;
//                                    NO rendezvous behind it)
//     F5  phase-2b slab-1 tasks      same, slab-1 word
//     F6  M8 combine front           gate: 8 slab-0 words (bounded poll with
//                                    wgrad-tile filler); warp tickets as today
//     F7  M8 combine back            gate: all 16 words (+ filler)
//     F8  wgrad tail                 the t2v6 cursor drain (dW2 then dW13)
//     M9  counted retire             verbatim
//
//   A CTA walks the families in order, claiming one task at a time; family
//   cursors are global, so drained CTAs flow forward elastically (the M15
//   C-pool early-combine becomes an emergent property: whoever runs out of
//   slab-1 work combines the certified front half).  Deadlock-freedom: a
//   claimed task's gate waits only on tasks of EARLIER families, all of
//   which are claimed by construction (cursors hand out a total order).
//
// wgrad here is HONEST WORK on the ticket (MoK's lesson: their wgrad is 50%
// of backward FLOPs and is not hidden — it is simply executed without
// separate-kernel overhead, gather passes, or allocator churn), plus filler
// duty inside the two M8 word waits.  Cross-microbatch accumulation is the
// t2v6 epoch-derived contract (DWMODE mode 3).
//
// exp_38 discipline: no donor-parity arms are carried; every wait is a
// bounded poll on the existing fail-closed pperr ladder; the phase bodies
// are driven one-task-per-invocation purely through their macro seams (the
// only body edit in T3 is the default-empty N2GM_P1Z_TASK_DONE_HOOK).
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
#include "k0pf6gm_t3_abi.hpp"

#define K0P6_T3_SRC_REV 1

// The hardware fp8 dequant in the swiglu' epilogue (proven −21 ms/iter).
#define K0P6_T2B_FASTCVT 1

// Donor-parity arms do not exist in this TU.
#define K0P6_T2B_PHASES 1
#define K0P6_M15_STAGED 0
#define K0P6_M15_SCATTER_RR 0
#define K0P6_M15_REPLICATE 0
#define K0P6_M15_ADAPTIVE 0
#define K0P6_M20_SLOTPOOL 0

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

// ---- descriptor offsets (identical to t2b; slots 63..72 from the ABI header) --------------
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
#define K0P6_M15_D_LEN K0P6_MPS_D_LEN

#ifndef K0P6_T3_KERNEL_NAME
#define K0P6_T3_KERNEL_NAME k0pf6gm_t3_mega
#endif
#undef K0P6_M15_KERNEL_NAME
#define K0P6_M15_KERNEL_NAME K0P6_T3_KERNEL_NAME

// Slab split retained for the combine certificates (column 3584 = nc 8 =
// M8 chunk 7); the RENDEZVOUS is gone — only the counted-arrival words.
#define K0P6_M15_SLABS 2
#define K0P6_M15_NC_PER_SLAB 8
#define K0P6_M15_M8_CSPLIT 7

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

// ---- task hooks (donor forms) -------------------------------------------------------------
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

// ---- T3 counter helpers -------------------------------------------------------------------
__device__ __forceinline__ unsigned int* k0p6t3_ctrs(const long long* desc) {
  return (unsigned int*)k0p6_dread(desc, K0P6_D_T3_CTRS);
}

// CTA-collective family-cursor claim: tid0 fetch_adds, LDS-broadcast via a
// dead t2bsh cell (dispatcher-only use; every body call re-fills its own
// LDS behind __syncthreads, so the cell is free between invocations).
__device__ __forceinline__ int k0p6t3_claim(unsigned int* cursor, int space,
                                            int* cell, int tid) {
  if (tid == 0) {
    const unsigned int i =
        kittens::distributed::fetch_add_relaxed<hk_moe::scope::agent>(cursor,
                                                                      1u);
    *cell = ((int)i < space) ? (int)i : -1;
  }
  __syncthreads();
  const int got = *cell;
  __syncthreads();
  return got;
}

// Bounded relaxed-spin on a release-published u32 flag (z_done cells).
__device__ __forceinline__ bool k0p6t3_poll_flag(const unsigned int* cell,
                                                 long long spin_limit,
                                                 int* pperr, int bit) {
  long long spins = 0;
  while (__hip_atomic_load(cell, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT) == 0u) {
    if (++spins > spin_limit) {
      atomicOr(pperr, bit);
      return false;
    }
    __builtin_amdgcn_s_sleep(2);
  }
  return true;
}

// Bounded spin until counter >= target (unpack_done cells).
__device__ __forceinline__ bool k0p6t3_poll_count(const unsigned int* cell,
                                                  unsigned int target,
                                                  long long spin_limit,
                                                  int* pperr, int bit) {
  long long spins = 0;
  while (__hip_atomic_load(cell, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT) < target) {
    if (++spins > spin_limit) {
      atomicOr(pperr, bit);
      return false;
    }
    __builtin_amdgcn_s_sleep(2);
  }
  return true;
}

// ---- M15 deferred-drain hooks (donor) -----------------------------------------------------
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

// ---- N2 phase bodies, one-task-per-invocation wiring --------------------------------------
// P1Z: START = the claimed ticket, STRIDE >= num_tasks => the internal loop
// runs exactly once.  The DONE hook drains stores and release-publishes
// z_done[ticket] (p1z and p1b share the identical (tile, g) task space).
#define N2GM_G K0P6GM_G
#define N2_P1Z_QUAL __device__ __forceinline__
#define N2_P1Z_NAME n2p6t2b_phase1z_body
#define N2_KERNELS_ONLY
#define N2_HOOK_CTX_ARG \
  , const long long* __restrict__ k0p6_desc, int n2gm_p1z_task
#define N2GM_P1Z_TASK_START n2gm_p1z_task
#define N2GM_P1Z_TASK_STRIDE (1 << 28)
#define N2GM_P1Z_TASK_DONE_HOOK                                            \
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");                         \
  __syncthreads();                                                         \
  if (tid == 0) {                                                          \
    unsigned int* t3zd = k0p6t3_ctrs(k0p6_desc) + K0P6_T3_CTR_ZDONE;       \
    __hip_atomic_store(t3zd + task, 1u, __ATOMIC_RELEASE,                  \
                       __HIP_MEMORY_SCOPE_AGENT);                          \
  }
#include "k0pf6gm_t2b_shared.hpp"
#include "n2_phase1z_gm_t2b.cpp"
#undef N2GM_P1Z_TASK_DONE_HOOK
#undef N2GM_P1Z_TASK_STRIDE
#undef N2GM_P1Z_TASK_START
#undef N2_HOOK_CTX_ARG
#undef N2_KERNELS_ONLY
#undef N2_P1Z_NAME
#undef N2_P1Z_QUAL

// P1B: same one-task wiring; the a2 arrival hook is the existing edge.
#define N2_P1B_QUAL __device__ __forceinline__
#define N2_P1B_NAME n2p6t2b_phase1b_body
#define N2_KERNELS_ONLY
#define N2_HOOK_CTX_ARG \
  , const long long* __restrict__ k0p6_desc, int n2gm_p1b_task
#define N2GM_P1B_TASK_START n2gm_p1b_task
#define N2GM_P1B_TASK_STRIDE (1 << 28)
#define N2GM_P1B_ARRIVAL_HOOK k0p6_a2_arrive(b, tid, k0p6_desc);
#include "n2_phase1b_gm_t2b.cpp"
#undef N2GM_P1B_ARRIVAL_HOOK
#undef N2GM_P1B_TASK_STRIDE
#undef N2GM_P1B_TASK_START
#undef N2_HOOK_CTX_ARG
#undef N2_KERNELS_ONLY
#undef N2_P1B_NAME
#undef N2_P1B_QUAL

// P2B: driven through the LO/HI arm — one claimed task t => LO=t, HI=t+1,
// START=0, STRIDE=1.  The per-call LOOP_TAIL flush drains the task's remote
// RMWs before its counted arrival, which is what makes the slab words'
// counted-arrival certificate sound (M9's visibility argument).
#define N2_P2B_QUAL __device__ __forceinline__
#define N2_P2B_NAME n2p6t2b_phase2b_body
#define N2_KERNELS_ONLY
#define N2_HOOK_CTX_ARG                                             \
  , const long long* __restrict__ k0p6_desc, int n2gm_task_lo,      \
      int n2gm_task_hi, k0p6_defer* n2gm_pend
#define N2GM_TASK_WAIT_HOOK \
  if (!k0p6_a2_wait(b, tid, k0p6_desc)) return;
#define N2GM_TASK_DONE_HOOK \
  k0p6_m15_task_done_defer(b, nc, tid, k0p6_desc, *n2gm_pend);
#define N2GM_TASK_LOOP_HEAD_HOOK \
  k0p6_m15_task_flush_defer(tid, k0p6_desc, *n2gm_pend);
#define N2GM_TASK_LOOP_TAIL_HOOK ;
#define N2GM_TASK_START 0
#define N2GM_TASK_STRIDE 1
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

// The wgrad tile body (k0p6wg_tile + driver), unchanged.
#include "n2_wgrad_gm_t2b.cpp"

// ---- small shared helpers (donor) ---------------------------------------------------------
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

// ---- wgrad ticket machinery (the t2v6 filler, self-contained here) ------------------------
namespace t3wg {

using production_fused_moe::n2::kWgMT;
using production_fused_moe::n2::kWgNT;
inline constexpr int kN13 = 7168 / kWgNT;
inline constexpr int kN2 = 2048 / kWgNT;
inline constexpr int kT13 = (4096 / kWgMT) * kN13;
inline constexpr int kT2 = (7168 / kWgMT) * kN2;

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
  std::uint32_t* cur_dw2;
  std::uint32_t* cur_dw13;
  __hip_bfloat16* ldsA;
  __hip_bfloat16* ldsB;
  float* ldsMeta;
  int* ldsRange;
  int* ldsClaim;
};

__device__ __forceinline__ void bind_ctx(const long long* desc,
                                         fill_ctx& c, bool accumulate) {
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
  c.n_experts = (int)k0p6_dread(desc, K0P6_D_E);
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

}  // namespace t3wg

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
      MAXTOK > T_loc_max / 8 || (MAXTOK & (MAXTOK - 1)) != 0 ||
      input_tokens < 0 || input_tokens > MAXTOK || spin_limit < 0 ||
      nct != 256 ||
      (MAXTOK + K0P6_CHUNK - 1) / K0P6_CHUNK > K0P6C_NCHUNK_MAX) {
    if (bid == 0 && tid == 0) atomicOr(pperr, 131072);
    return;
  }
  {
    const hk_moe::mps::config entry_cfg = hk_moe::mps::decode_config(
        (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
    if (!hk_moe::mps::config_is_valid(entry_cfg) || PADMAX / 32 > 16383 ||
        !hk_moe::mps::mode_is_direct_accum(entry_cfg) ||
        k0p6_dread(desc, K0P6_D_MPS_STATE) == 0 ||
        (k0p6_dread(desc, K0P6_D_MPS_STATE) & 7) != 0 ||
        k0p6_dread(desc, K0P6_D_MPS_SLOTS) == 0 ||
        (k0p6_dread(desc, K0P6_D_MPS_SLOTS) & 15) != 0 ||
        k0p6_dread(desc, K0P6_D_ROW_READY) == 0 ||
        world * MAXTOK + K0P6_M15_SLABS > T_loc_max
        || k0p6_dread(desc, K0P6_D_T2B_ZQ) == 0
        || k0p6_dread(desc, K0P6_D_T2B_DQZ) == 0
        || k0p6_dread(desc, K0P6_D_T2B_ACTQ) == 0
        || k0p6_dread(desc, K0P6_D_T2B_ACTDQ) == 0
        || k0p6_dread(desc, K0P6_D_T2B_XROWS) == 0
        || k0p6_dread(desc, K0P6_D_T2B_XSC) == 0
        || k0p6_dread(desc, K0P6_D_T2B_DW13) == 0
        || k0p6_dread(desc, K0P6_D_T2B_DW2) == 0
        || k0p6_dread(desc, K0P6_D_T2B_DWTOPK) == 0
        || k0p6_dread(desc, K0P6_D_T3_CTRS) == 0
        || (k0p6_dread(desc, K0P6_D_T3_CTRS) & 7) != 0) {
      if (bid == 0 && tid == 0) atomicOr(pperr, K0P6_MPS_ERR_CONFIG);
      return;
    }
  }
  const int EL = E;
  const int T_ext = world * MAXTOK;
  const int lo = cur * E;
  (void)EL;
  (void)lo;
  const long long debug_stop = k0p6_dread(desc, K0P6_D_DEBUG);

  // DWMODE (t2v6 contract): [3:0] mode (3 = T3 in-kernel wgrad; 0 = host
  // wgrad — wgrad families skipped), [7:4] nmb, accumulate derived from
  // the epoch counter.
  const unsigned int dwword =
      (unsigned int)(unsigned long long)k0p6_dread(desc, K0P6_D_T2B_DWMODE);
  const int dwmode = (int)(dwword & 0xFu);
  const int dw_nmb = (int)((dwword >> 4) & 0xFu);
  const bool dw_accumulate =
      (dw_nmb > 1) &&
      (((k0p6_epoch(desc) - 1ull) % (unsigned long long)dw_nmb) != 0ull);

  const int ntiles = ((const int*)k0p6_dread(desc, K0P6_D_NUM_TILES))[0];
  const int n_p1z = ntiles * 8;   // == n_p1b: shared (tile, g) task space
  const int n_p2b_half = ntiles * K0P6_M15_NC_PER_SLAB;
  unsigned int* ctrs = k0p6t3_ctrs(desc);
  int* claim_cell = &production_fused_moe::n2::t2bsh_srow_lds[2];

  // ================= M0: retire wait + resets + THE one grid barrier ======================
  {
    const std::uint64_t epoch = k0p6_epoch(desc);
    std::uint64_t* retired =
        (std::uint64_t*)k0p6_dread(desc, K0P6_D_RETIRED);
    if (tid < world) {
      (void)hk_moe::poll_value_at_least_system(
          retired + tid, epoch > 0 ? epoch - 1ull : 0ull,
          (std::uint64_t)spin_limit, pperr, 16777216);
    }
    const int maxb = (int)k0p6_dread(desc, K0P6_D_MAXB);
    unsigned int* a2_done = (unsigned int*)k0p6_dread(desc, K0P6_D_A2_DONE);
    unsigned int* part_done = (unsigned int*)k0p6_dread(desc, K0P6_D_PART_DONE);
    const int gtid = bid * (int)blockDim.x + tid;
    const int gstride = nct * (int)blockDim.x;
    for (int i = gtid; i < maxb; i += gstride) {
      a2_done[i] = 0u;
      part_done[i] = 0u;
    }
    {
      std::uint32_t* mps_state = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
      for (int i = gtid; i < K0P6_MPS_ST_WORDS + 2 * K0P6_MPS_TS_COUNT;
           i += gstride) {
        mps_state[i] = 0u;
      }
    }
    // T3 counter arena: cursors, unpack/slab counters, and the z_done flags
    // for this launch's task count.
    for (int i = gtid; i < K0P6_T3_CTR_WORDS + n_p1z; i += gstride)
      ctrs[i] = 0u;
    // dw_topk pre-zero (the M3-position role, folded behind the same seal).
    {
      float* dw0 = (float*)k0p6_dread(desc, K0P6_D_T2B_DWTOPK);
      for (int i = gtid; i < PADMAX; i += gstride) dw0[i] = 0.0f;
    }
    // exp_56 fix (6): grid-wide reset of the NEXT epoch's dest_counter parity.
    {
      unsigned int* dc_next =
          (unsigned int*)k0p6_dread(desc, K0P6_D_DESTCTR)
          + (size_t)(((k0p6_epoch(desc) + 1ull) & 1ull)) * (size_t)world;
      for (int i = gtid; i < world; i += gstride)
        hk_moe::publish_relaxed<hk_moe::scope::system>(dc_next + i, 0u);
      const int self = cur;
      for (int i = gtid; i < world; i += gstride) {
        if (i == self &&
            hk_moe::load_relaxed<hk_moe::scope::system>(dc_next + i) != 0u)
          atomicOr(pperr, 262144);
      }
    }
    {
      unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
      const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
      const hkp::fail_closed bar_err{pperr, 2097152};
      hkp::grid_barrier(bar, tid, bar_err);
    }
    hk_moe::acquire_payload_agent();
    if (hk_moe::error_bit_set_agent(pperr, 16777216 | 2097152)) return;
    __syncthreads();
  }
  if (debug_stop == 1) return;

  // ================= F0 — M1b: qpush dispatch (donor verbatim, saved plan) =================
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

    const int gw = bid * (blockDim.x >> 6) + wid;
    const int nw = nct * (blockDim.x >> 6);
    for (int tau = gw; tau < T; tau += nw) {
      const int s = lane;
      const int eid = (s < TOPK) ? my_ids[(size_t)tau * TOPK + s] : -1;
      const float wgt = (s < TOPK) ? my_wgt[(size_t)tau * TOPK + s] : 0.0f;
      const int dest =
          (s < TOPK) ? pull_stage[((size_t)tau * TOPK + s) * 2 + 0] : -1;
      const int saved_row =
          (s < TOPK) ? pull_stage[((size_t)tau * TOPK + s) * 2 + 1] : -1;
      const bool primary = (dest >= 0) && (saved_row >= 0);
      const unsigned long long pmask = __ballot(primary);
      const int fanout = __popcll(pmask);

      const unsigned int row = primary ? (unsigned int)saved_row : 0u;
      (void)dest_counter;
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

  // ================= F1 — unpack (source, chunk, sub) tasks ================================
  // FIRST family after dispatch: the 512 sub-tasks (64 rows each) gate on
  // their chunk's epoch word; CTAs that find the cursor drained flow
  // straight into z-regen, so arrival-bound unpackers never idle the
  // grid.  signal: unpack_done[source] counted arrival, target 64.
  {
    const std::uint64_t epoch = k0p6_epoch(desc);
    const unsigned int epoch32 = (unsigned int)epoch;
    unsigned char* a_ll = (unsigned char*)k0p6_dread(desc, K0P6_D_A_LL);
    unsigned char* a_dst = (unsigned char*)k0p6_dread(desc, K0P6_D_A_DST);
    float* sc_stage = (float*)k0p6_dread(desc, K0P6_D_SC_STAGE);
    int* recv_eid = (int*)k0p6_dread(desc, K0P6_D_RECV_EID);
    float* recv_wgt = (float*)k0p6_dread(desc, K0P6_D_RECV_WGT);
    const std::uint64_t* chunk_ready =
        (const std::uint64_t*)k0p6_dread(desc, K0P6_D_CHUNK_READY);
    const int nchunks = K0P6C_NCHUNK_MAX;
    const int nsub = K0P6_CHUNK / 64;
    const int g = lane >> 3, j = lane & 7;
    while (true) {
      const int t = k0p6t3_claim(ctrs + K0P6_T3_CTR_UNPK_CUR,
                                 world * nchunks * nsub, claim_cell, tid);
      if (t < 0) break;
      const int s = t / (nchunks * nsub);
      const int rem = t - s * (nchunks * nsub);
      const int c = rem / nsub;
      const int sub = rem - c * nsub;
      unsigned int fill = 0u;
      bool ok = true;
      if (tid == 0)
        ok = hk_moe::poll_epoch_word_system(
            chunk_ready + (size_t)s * (size_t)nchunks + (size_t)c,
            epoch32, (std::uint64_t)spin_limit, pperr, 8388608,
            &fill, (unsigned int*)k0p6_dread(desc, K0P6_D_SPIN_DBG));
      if (tid == 0) s_fill[0] = ok ? (int)fill : -1;
      __syncthreads();
      const int chunk_fill = s_fill[0];
      __syncthreads();
      if (chunk_fill < 0) return;  // fail-closed: pperr already set
      hk_moe::acquire_payload_system();
      const int sub0 = sub * 64;
      int sub_fill = chunk_fill - sub0;
      if (sub_fill < 0) sub_fill = 0;
      else if (sub_fill > 64) sub_fill = 64;
      const int r0 = s * MAXTOK + c * K0P6_CHUNK + sub0;
      for (int r = r0 + wid; r < r0 + sub_fill; r += (int)(blockDim.x >> 6)) {
        k0p6_unpack_row_nopoll((size_t)r,
                               a_ll + (size_t)r * (size_t)K0P5_ROW_STRIDE,
                               a_dst, sc_stage, recv_eid, recv_wgt, g, j);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (tid == 0) {
        __hip_atomic_fetch_add(ctrs + K0P6_T3_CTR_UNPACK + s, 1u,
                               __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
  if (debug_stop == 3) return;

  // ================= F2 — z-regen tasks (NO dependencies) ==================================
  // These absorb the dY dispatch flight time and the cross-rank drift the
  // M2 barrier used to realize: a CTA that finishes its M1b share flows
  // straight into z tiles while peers' rows are still on the wire.
  {
    const std::uint8_t* xrows = (const std::uint8_t*)k0p6_dread(desc, K0P6_D_T2B_XROWS);
    const float* xsc = (const float*)k0p6_dread(desc, K0P6_D_T2B_XSC);
    const std::uint8_t* w13o = (const std::uint8_t*)k0p6_dread(desc, K0P6_D_T2B_ACTQ);
    const float* s13o = (const float*)k0p6_dread(desc, K0P6_D_T2B_ACTDQ);
    const int* sti = (const int*)k0p6_dread(desc, K0P6_D_STI);
    const int* sei = (const int*)k0p6_dread(desc, K0P6_D_SEI);
    const int* nvi = (const int*)k0p6_dread(desc, K0P6_D_NVI);
    std::uint8_t* zq = (std::uint8_t*)k0p6_dread(desc, K0P6_D_T2B_ZQ);
    float* dqz = (float*)k0p6_dread(desc, K0P6_D_T2B_DQZ);
    const int* tile_desc = (const int*)k0p6_dread(desc, K0P6_D_TILE_DESC);
    while (true) {
      const int t = k0p6t3_claim(ctrs + K0P6_T3_CTR_P1Z_CUR, n_p1z,
                                 claim_cell, tid);
      if (t < 0) break;
      production_fused_moe::n2::n2p6t2b_phase1z_body(
          xrows, xsc, w13o, s13o, sti, sei, nvi, zq, dqz, tile_desc, ntiles,
          desc, t);
    }
  }
  if (debug_stop == 4) return;

  // ================= F3 — phase-1b tasks (gates: z 1:1, sources by sti scan) ===============
  {
    const std::uint8_t* a_dst = (const std::uint8_t*)k0p6_dread(desc, K0P6_D_A_DST);
    const float* sc_stage = (const float*)k0p6_dread(desc, K0P6_D_SC_STAGE);
    const std::uint8_t* w2t = (const std::uint8_t*)k0p6_dread(desc, K0P6_D_W13);
    const float* s2t = (const float*)k0p6_dread(desc, K0P6_D_S13);
    const std::uint8_t* zq = (const std::uint8_t*)k0p6_dread(desc, K0P6_D_T2B_ZQ);
    const float* dqz = (const float*)k0p6_dread(desc, K0P6_D_T2B_DQZ);
    const int* sti = (const int*)k0p6_dread(desc, K0P6_D_STI);
    const int* sei = (const int*)k0p6_dread(desc, K0P6_D_SEI);
    const int* nvi = (const int*)k0p6_dread(desc, K0P6_D_NVI);
    std::uint8_t* dzq = (std::uint8_t*)k0p6_dread(desc, K0P6_D_A2Q);
    float* dqdz = (float*)k0p6_dread(desc, K0P6_D_DQ2);
    float* dwtopk = (float*)k0p6_dread(desc, K0P6_D_T2B_DWTOPK);
    const int* tile_desc = (const int*)k0p6_dread(desc, K0P6_D_TILE_DESC);
    const unsigned int* zdone = ctrs + K0P6_T3_CTR_ZDONE;
    const int mt_shift = 31 - __clz((unsigned int)MAXTOK);
    // srow cells: [0..1] wgrad claim/ranges, [2..3] dispatcher claim pair,
    // [4] the source-mask accumulator — disjoint from claim_cell[1] (=[3]).
    int* mask_cell = &production_fused_moe::n2::t2bsh_srow_lds[4];
    while (true) {
      const int t = k0p6t3_claim(ctrs + K0P6_T3_CTR_P1B_CUR, n_p1z,
                                 claim_cell, tid);
      if (t < 0) break;
      // gate (i): the 1:1 z edge.
      if (tid == 0) {
        const bool zok = k0p6t3_poll_flag(zdone + t, spin_limit, pperr,
                                          16777216);
        claim_cell[1] = zok ? 1 : 0;
      }
      __syncthreads();
      const int zok = claim_cell[1];
      __syncthreads();
      if (!zok) return;
      // gate (ii): source set of the tile window from the saved plan.
      const int tile = t / 8;
      const int b0 = tile_desc[tile] >> 4;
      const int gcount = tile_desc[tile] & 0xF;
      if (tid == 0) *mask_cell = 0;
      __syncthreads();
      {
        const int rows = gcount * 32;
        if (tid < rows) {
          const int recv = sti[b0 * 32 + tid] & 0x00FFFFFF;
          if (recv < T_ext) atomicOr(mask_cell, 1 << (recv >> mt_shift));
        }
      }
      __syncthreads();
      const int smask = *mask_cell;
      __syncthreads();
      if (tid == 0) {
        bool sok = true;
        for (int s = 0; s < world && sok; ++s) {
          if ((smask >> s) & 1)
            sok = k0p6t3_poll_count(ctrs + K0P6_T3_CTR_UNPACK + s, 64u,
                                    spin_limit, pperr, 16777216);
        }
        claim_cell[1] = sok ? 1 : 0;
      }
      __syncthreads();
      const int sok = claim_cell[1];
      __syncthreads();
      if (!sok) return;
      production_fused_moe::n2::n2p6t2b_phase1b_body(
          a_dst, sc_stage, w2t, s2t, zq, dqz, sti, sei, nvi, dzq, dqdz,
          dwtopk, tile_desc, ntiles, desc, t);
    }
  }
  if (debug_stop == 5) return;

  // ================= F4/F5 — phase-2b tasks, slab-certified by counted arrival =============
  {
    const std::uint8_t* dzq = (const std::uint8_t*)k0p6_dread(desc, K0P6_D_A2Q);
    const float* dqdz = (const float*)k0p6_dread(desc, K0P6_D_DQ2);
    const std::uint8_t* w13t = (const std::uint8_t*)k0p6_dread(desc, K0P6_D_W2);
    const float* s13t = (const float*)k0p6_dread(desc, K0P6_D_S2);
    const int* sti = (const int*)k0p6_dread(desc, K0P6_D_STI);
    const float* swt = (const float*)k0p6_dread(desc, K0P6_D_SWT);
    const int* sei = (const int*)k0p6_dread(desc, K0P6_D_SEI);
    const int* nvi = (const int*)k0p6_dread(desc, K0P6_D_NVI);
    __hip_bfloat16* part = (__hip_bfloat16*)k0p6_dread(desc, K0P6_D_PART);
    const int* tile_desc = (const int*)k0p6_dread(desc, K0P6_D_TILE_DESC);
    const unsigned int epoch32 = (unsigned int)k0p6_epoch(desc);
    k0p6_defer p2b_pend{-1, 0, 0};
#pragma unroll 1
    for (int slab = 0; slab < K0P6_M15_SLABS; ++slab) {
      unsigned int* cursor =
          ctrs + (slab == 0 ? K0P6_T3_CTR_P2B0_CUR : K0P6_T3_CTR_P2B1_CUR);
      unsigned int* done_ctr =
          ctrs + (slab == 0 ? K0P6_T3_CTR_SLAB0 : K0P6_T3_CTR_SLAB1);
      const int base = slab * n_p2b_half;
      while (true) {
        const int u = k0p6t3_claim(cursor, n_p2b_half, claim_cell, tid);
        if (u < 0) break;
        const int t = base + u;
        production_fused_moe::n2::n2p6t2b_phase2b_body(
            dzq, dqdz, w13t, s13t, sti, swt, sei, nvi, part, tile_desc,
            ntiles, desc, t, t + 1, &p2b_pend);
      }
      // drain the last task's deferred RMWs before this CTA certifies.
      k0p6_m15_task_flush_defer(tid, desc, p2b_pend);
      // Slab certificate WITHOUT a rendezvous: every CTA (task-claiming or
      // not) drains its vmem, writes back its L2 (the release half of the
      // old rendezvous — vmcnt alone does not make remote RMWs fabric-
      // visible), then arrives ONCE on the slab's CTA counter.  The 256th
      // arrival's acquire orders it after every CTA's writeback, and it
      // publishes the slab word to all 8 peers.  CTAs flow straight into
      // the next family — the barrier's WAIT half is gone, its visibility
      // half is preserved verbatim.
      hk_moe::release_cta_payload_system();
      __syncthreads();
      if (tid == 0) {
        const unsigned int old = __hip_atomic_fetch_add(
            done_ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if ((int)old == nct - 1) {
          const auto* symmetric = k0p6_symmetric(desc);
          unsigned int* row_ready =
              (unsigned int*)k0p6_dread(desc, K0P6_D_ROW_READY);
          hk_moe::release_signal_batch_system();
          unsigned int* self_slot =
              row_ready + (size_t)cur * T_loc_max + (size_t)T_ext + slab;
          for (int R = 0; R < world; ++R) {
            if (R == cur) {
              hk_moe::publish_epoch<hk_moe::scope::agent>(self_slot,
                                                          epoch32);
            } else {
              hk_moe::publish_epoch<hk_moe::scope::system>(
                  hk_moe::peer_ptr(self_slot, R, symmetric), epoch32);
            }
          }
        }
      }
      __syncthreads();
    }
  }
  if (debug_stop == 6) return;

  // ================= F6/F7 — M8 combine, word-gated with wgrad filler ======================
  {
    const unsigned int epoch32 = (unsigned int)k0p6_epoch(desc);
    const unsigned int* row_ready =
        (const unsigned int*)k0p6_dread(desc, K0P6_D_ROW_READY);
    t3wg::fill_ctx wgc;
    if (dwmode >= 3) {
      t3wg::bind_ctx(desc, wgc, dw_accumulate);
      // zero the per-CTA ranges-scanned flag (dead LDS, garbage post-p1b).
      if (tid == 0) production_fused_moe::n2::t2bsh_srow_lds[1] = 0;
      __syncthreads();
    }
    // front gate: this rank + 7 peers' slab-0 words; while any is missing,
    // run one wgrad tile (poll-and-fill — the whole grid participates).
    for (int stage = 0; stage < 2; ++stage) {
      const int nwords = (stage == 0) ? 1 : K0P6_M15_SLABS;
      while (true) {
        if (tid == 0) {
          int ready = 1;
          for (int s = 0; s < nwords && ready; ++s) {
            for (int p = 0; p < world; ++p) {
              if (hk_moe::load_relaxed<hk_moe::scope::system>(
                      row_ready + (size_t)p * T_loc_max + (size_t)T_ext +
                      s) != epoch32) {
                ready = 0;
                break;
              }
            }
          }
          *claim_cell = ready;
        }
        __syncthreads();
        const int all_ready = *claim_cell;
        __syncthreads();
        if (all_ready) break;
        if (dwmode >= 3) {
          t3wg::ensure_ranges(wgc);
          int i = t3wg::claim(wgc.cur_dw2, wgc.n_experts * t3wg::kT2,
                              wgc.ldsClaim);
          if (i >= 0) {
            t3wg::run_dw2(wgc, i);
            continue;
          }
          i = t3wg::claim(wgc.cur_dw13, wgc.n_experts * t3wg::kT13,
                          wgc.ldsClaim);
          if (i >= 0) {
            t3wg::run_dw13(wgc, i);
            continue;
          }
        }
        // nothing to fill with: bounded poll, fail-closed.
        if (tid == 0) {
          for (int s = 0; s < nwords; ++s) {
            for (int p = 0; p < world; ++p) {
              (void)hk_moe::poll_epoch_system(
                  row_ready + (size_t)p * T_loc_max + (size_t)T_ext + s,
                  epoch32, (std::uint64_t)spin_limit,
                  hk_moe::set_error_bit{pperr, 33554432});
            }
          }
        }
        __syncthreads();
        break;
      }
      const bool payload_ok = !hk_moe::error_bit_set_agent(
          pperr, 16777216 | 2097152 | 33554432 | K0P6_MPS_ERR_CONFIG);
      const int T = payload_ok ? (int)k0p6_dread(desc, K0P6_D_T) : 0;
      const int* pull_ptr = (const int*)k0p6_dread(desc, K0P6_D_PULL_PTR);
      const int* pull_src = (const int*)k0p6_dread(desc, K0P6_D_PULL_SRC);
      unsigned short* slots = (unsigned short*)k0p6_dread(desc, K0P6_D_MPS_SLOTS);
      unsigned short* out = (unsigned short*)k0p6_dread(desc, K0P6_D_OUT);
      std::uint32_t* mps_state = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
      constexpr int NT = 4;
      const int nbatches = (T + NT - 1) / NT;
      std::uint32_t* ticket = mps_state + (stage == 0 ? K0P6_MPS_ST_M8NEXT
                                                      : K0P6_MPS_ST_EVNEXT);
      int* batch_slot = &s_ns[wid];
      while (true) {
        if (lane == 0) {
          *batch_slot =
              kittens::distributed::fetch_add_relaxed<hk_moe::scope::agent>(
                  ticket, 1u);
        }
        __syncwarp();
        const int batch = *batch_slot;
        if (batch >= nbatches) break;
        if (stage == 0) {
          k0p6_m15_m8_batch<NT, 0, K0P6_M15_M8_CSPLIT>(
              batch * NT, T, pull_ptr, pull_src, lane, cur, MAXTOK, slots,
              out);
        } else {
          k0p6_m15_m8_batch<NT, K0P6_M15_M8_CSPLIT, 14>(
              batch * NT, T, pull_ptr, pull_src, lane, cur, MAXTOK, slots,
              out);
        }
      }
      __syncthreads();
    }

    // ================= F8 — wgrad tail drain ==============================================
    if (dwmode >= 3) {
      t3wg::ensure_ranges(wgc);
      while (true) {
        int i = t3wg::claim(wgc.cur_dw2, wgc.n_experts * t3wg::kT2,
                            wgc.ldsClaim);
        if (i < 0) break;
        t3wg::run_dw2(wgc, i);
      }
      while (true) {
        int i = t3wg::claim(wgc.cur_dw13, wgc.n_experts * t3wg::kT13,
                            wgc.ldsClaim);
        if (i < 0) break;
        t3wg::run_dw13(wgc, i);
      }
    }
  }

  // ================= M9: counted arrival, resets, retired pokes (donor) ====================
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

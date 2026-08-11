// ================================================================================================
// MPS (MINIMUM-PROGRESS SPECIALIZATION) SIBLING — UNMEASURED until the local
// validation gates pass on 8x gfx950. This is a COMET/MoK-style role-split
// experiment built ADDITIVELY on k0pf6gm_device_tile.hip; the abstraction-parity
// port is untouched. Design contract: DESIGN_MPS.md. Performance target: the
// exact exp_35 G=3/c4 donor at 6,919.8 µs vs 7,698.0 µs production.
//
// Frozen donor chain:
//   experiments/exp_64_m2_row_striping/source_snapshot/k0pf6gm_mega.hip
//   sha256 57152a087c62aee1ffaae464cd2972e321d5e283e0681bc5a248039a6dc1ae2c
//   via the parity port (same file, this directory).
//
// Mechanism deltas vs the parity port:
//   * M6 end: finish-order role partition. The first C finishers become
//     minimum-progress service CTAs; everyone else receives a dense logical
//     compute id and M7 strides the pool of (256-C) so no GEMM task is lost.
//   * M7 task-done hook publishes (b, nc) completion to a capacity-exact event
//     queue with one agent release on tid0 — no system fence, no remote work,
//     no row loop on the GEMM CTA (replaces the semantic role of part_done's
//     count with per-(block, N-chunk) readiness records).
//   * Service CTAs consume events (wave-striped), do (row,chunk) arrival
//     bookkeeping, claim adjacent chunk groups, and PUSH completed slices into
//     owner-resident slots [producer][source-row-offset][chunk] with controlled
//     16-byte stores. Completed rows are flagged to their owner only, batched
//     behind one system release per flush group.
//   * M7.5's bulk publication is gone in stream mode: row_ready cells and
//     arrival/push counts make per-slice completion explicit, so the grid
//     barrier between M7 and combine is removed as a consequence.
//   * M8 reduces owner-LOCAL slots (same arithmetic; base pointer changes from
//     peer part rows to local slot rows); batches are claimed by dynamic
//     ticket so drained compute CTAs — and finally the service CTAs — flow
//     into reduction work.
//   * Modes (descriptor word K0P6_D_MPS_CFG): 0 = reserve-only control (parity
//     protocol, M7 strided by the reservation); 1 = bulk push after the parity
//     barrier (copy/layout without overlap); 2 = full stream.
//
// The phase-2 GEMM body is the vendored n2_phase2_gm_mps.cpp (pinned donor sha
// recorded below + two behavioral deltas: task start/stride macros and a
// per-thread VMEM drain hook before task-end publication). Generated
// ISA/resources must be compared against the parity port before any claim.
// ================================================================================================
// k0pf6_mega.hip — FULL-REGION FUSED MEGAKERNEL (v4): dispatch -> destination sort -> expert
// GEMM (N2 phase1 + phase2) -> combine, ONE launch per MoE layer per epoch.
//
// v4 fixes the residual exp_55 resource findings (v2/v3 attempts recorded in git history):
//   * The N2 phase2 body alone consumes the entire 256-VGPR budget at 1 wave/SIMD, so the
//     80-144 B/lane scratch was the allocator spilling whatever wrapper state it had to keep
//     live across M7.  v4's rule: NOTHING but `desc` may live across the GEMM phases.
//     - epoch is re-derived per phase from the mega_count cell (a desc read), never held;
//     - every phase reads its buffer pointers at phase start through k0p6_dread (a
//       non-CSE-able asm load), so pointers die at phase end;
//     - M8 reads even its scalars from desc at its own start.
//   * The v3 __noinline__ experiment is reverted (its ABI frames cost MORE scratch).
//
// Carried from v2 (see header history / exp_55 result.md): descriptor ABI (three kernargs);
// per-receive-row liveness (row_remaining written by M2 per epoch), a2_done/part_done zeroed at
// M0 (v1 M9 reset race removed), static row stripes, contiguous LDS row staging + ProtoLL128
// putPackets, coherent poll loads, and resets-before-poke in M9. The old per-task row finisher
// was replaced by exp_62's coarse post-M7 publication, corrected below for same-thread release.
//
// Structure (256 CTAs x 256 threads, phase-split uniform — every CTA runs every phase):
//   M0 retire wait + a2_done/part_done zero.  M1 qpush (LL128 rows, rows_done hints).
//   M2 streaming verify/unpack/histogram + row_remaining (overlaps other CTAs' M1).
//   M3 grid barrier.  M4 plan split.  M5 grid barrier + pad/scatter/pull_src_fill.
//   M6 N2 phase1 (a2_done arrivals). M7 N2 phase2 (a2 waits at task start). M7.5
//   grid-converged CTA-leader row certification. M8 streaming mtab4 combine (row_ready polls).
//   M9 counted arrival, resets, retired pokes.
//
// pperr bits: 131072 shape guard, 524288 T_loc out of range, 8388608 stream poll timeout,
//   16777216 a2_done/retire wait timeout, 33554432 row_ready combine poll timeout,
//   2097152 grid-barrier timeout, 65536 receive-row overflow, 1048576 scatter/CSR
//   overflow, 4194304 pad overflow, 262144 dest_counter base-nonzero at epoch start (exp_56 (ii)).
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
#include "k0pf6_chunk_release.hpp"   // exp_56: counted per-chunk release/acquire (additive)
#include "moe_hk_adapter.cuh"
#include "moe_mps_adapter.cuh"       // MPS: roles, event queue, slice push/flags

// ---- JIT CACHE INVALIDATION GUARD -- read this before editing any .cuh -------
// The mori JIT cache key is a content hash over its own _jit-sources tree,
// restricted to .hpp/.h/.cpp/.hip. Our .cuh headers arrive through -I from the
// read-only Distributed-HipKittens mount and are NOT hashed, so an edit
// confined to moe_mps_adapter.cuh / moe_hk_adapter.cuh / include/cdna4/**
// leaves the key unchanged and the campaign SILENTLY REUSES THE PREVIOUS
// k0pf6gm_mps_mega.hsaco. exp_04 lost a measurement to exactly this.
// This .hip IS hashed. Bump K0P6_MPS_SRC_REV in the same commit as any
// .cuh-only change, then confirm a new hsaco mtime under
// ~/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5/*/ before believing
// any number. The value is never read; only its presence in the hash matters.
#define K0P6_MPS_SRC_REV 3

// exp_56 (Option C): per-source SEGMENTED receive layout + chunk-counted release.
//  * Row index = source*MAXTOK + fetch_add(dest_counter[source] on dest, 1)  (waitless).
//  * Publication: ONE per-CTA system release fence at M1-end + ONE system store per (source,chunk)
//    flag (chunk = K0P6_CHUNK receive-rows). Consumer acquires once per chunk then reads payload
//    with plain non-polling loads. Replaces the per-128B-packet inline flag entirely.
//  * COUNT vs EXTENT: T_count = live arrivals (sum rows_done); T_ext = world*MAXTOK = swept extent.
//    Every loop bound / range filter / pad sentinel / phase mask uses T_ext; only nvi[0] statistics
//    use live per-expert counts (layout-independent). See exp_56 design.md §4b.
#define K0P6C_NCHUNK_MAX 8   // ceil(MAXTOK/CHUNK) = ceil(4096/512); chunk_ready inner dim

#define K0P6_MAXE 64
#define K0P6_NG 56
#define K0P6_H 7168
#define K0P6_SC_ERB 0
#define K0P6_SC_CNT 80
#define K0P6_SC_CUR 160

// ---- descriptor offsets (MIRRORED by the host driver; keep in sync) ----------------------
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
// exp_56 Option C additive slots (appended; existing offsets unchanged):
#define K0P6_D_CHUNK_READY 50   // uint64[world][K0P6C_NCHUNK_MAX] dest-side (epoch<<32|fill) chunk word
#define K0P6_D_MAXTOK 51        // per-source segment stride (= T = 4096 this campaign)
#define K0P6_D_SPIN_DBG 52      // uint32[2] diag: [0]=max spins to SUCCESS, [1]=max spins at FAIL
// exp_59 large-M G-stacking slots (appended; existing offsets unchanged):
#define K0P6_D_TILE_DESC 53     // int[bcap] packed (b0<<4)|gcount, one per expert-aligned tile
#define K0P6_D_NUM_TILES 54     // int[1] tile count (device-written in M4)
// HipKittens/IRIS ABI v1: device pointer to hk_moe::symmetric_heap_descriptor.
// The descriptor contains the local heap base plus eight host-snapshotted peer
// heap bases. It replaces workload-wide hidden peer-translation state.
#define K0P6_D_SYMMETRIC 55
// MPS slots 56..62 and K0P6_MPS_D_LEN (=63) are defined in moe_mps_adapter.cuh.
// K0P6_D_LEN is renamed: this kernel requires the extended descriptor.
#define K0P6_D_LEN K0P6_MPS_D_LEN

// G-stack width must match N2GM_G in both included phase bodies. This port fixes
// the best measured G=3 build; another value is a separate candidate and needs
// its own source/JIT cache key and complete gate run.
#ifndef K0P6GM_G
#define K0P6GM_G 3
#endif
#if K0P6GM_G != 3
#error "fused_moe is pinned to the measured K0P6GM_G=3 configuration"
#endif

using k0p6_sort = hkp::destination_counting_sort<K0P6_MAXE, 32, K0P6_SC_ERB, K0P6_SC_CNT,
                                                 K0P6_SC_CUR>;

// Phase-scoped descriptor read: the memory clobber keeps the compiler from CSE-hoisting
// descriptor values into kernel-long registers (the spill mechanism being avoided).
__device__ __forceinline__ long long k0p6_dread(const long long* desc, int off) {
  // Keep this as a compiler-visible load so SIInsertWaitcnts can track its VMEM dependency.
  // Volatile preserves the original phase-scoped re-read contract without a hand-written
  // full VMEM drain after every descriptor access.
  return *reinterpret_cast<volatile const long long*>(desc + off);
}

__device__ __forceinline__ const hk_moe::symmetric_heap_descriptor*
k0p6_symmetric(const long long* desc) {
  return reinterpret_cast<const hk_moe::symmetric_heap_descriptor*>(
      k0p6_dread(desc, K0P6_D_SYMMETRIC));
}

// The current epoch: derived from the monotonic cell, never held across phases.
__device__ __forceinline__ std::uint64_t k0p6_epoch(const long long* desc) {
  const unsigned int* epoch_cell =
      (const unsigned int*)k0p6_dread(desc, K0P6_D_EPOCH_CELL);
  return (std::uint64_t)
             hk_moe::load_relaxed<hk_moe::scope::agent>(epoch_cell) +
         1ull;
}

// ---- task hooks (referenced by the N2_TASK_*_HOOK macros below) ---------------------------
__device__ __forceinline__ void k0p6_a2_arrive(int b, int tid, const long long* desc) {
  if (tid == 0) {
    // Release-only arrival: the completion RMW itself is deliberately relaxed.
    // This removes the donor's unused acquire half while preserving its edge.
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
  // Only thread 0 polls. Converge the CTA and acquire on every consuming
  // execution mask before phase 2 reads the address-reused A2/DQ2 payload.
  hk_moe::acquire_payload_agent();
  return !hk_moe::error_bit_set_agent(
      (const int*)desc[K0P6_D_PPERR], 16777216);
}

__device__ __forceinline__ void k0p6_part_arrive(int b, int tid, const long long* desc) {
  // exp_62: HOISTED combine-release tail. The per-task epilogue no longer does ANY
  // cross-scope release work (no per-task system release, no row_remaining
  // decrement loop, no cross-rank row_ready pokes). That row-scaling per-task work
  // (profiler: the residual +11,198 us of M7) is moved OUT of the GEMM wave into a
  // single coarse post-M7 bulk pass (see the M6/M7 epilogue in the launch region:
  // one grid barrier + one grid-strided row_ready publication). This body keeps only
  // a plain LOCAL, AGENT-scope part_done bump so M0's part_done zeroing stays
  // meaningful; nothing on the correctness path reads part_done after the hoist.
  if (tid == 0) {
    unsigned int* part_done = (unsigned int*)desc[K0P6_D_PART_DONE];
    hk_moe::arrive_local_count(part_done + b);
  }
}

// M7 task stride for the density-preserving role split. The phase-2 body
// re-derives it from the descriptor config word at body entry, the same
// volatile channel its own hooks already use; nothing tuneable stays live
// across the MFMA mainloop. Mode 1 must carry C=0 (validation enforces it),
// so this is the donor's full-grid stride exactly.
__device__ __forceinline__ int k0p6_mps_stride(const long long* desc, int nct) {
  return nct - (int)((unsigned long long)k0p6_dread(desc, K0P6_D_MPS_CFG) &
                     0xFFull);
}

// ---- MPS phase-2 done hook -----------------------------------------------------
// Stream mode (2): publish the (b, nc) tile event off the GEMM wave — one
// agent release + one relaxed 32-bit store by tid0, after the vendored body
// drained every lane's epilogue atomics. Other modes keep the parity part_done
// bump so M0's zeroing contract is unchanged. The config word is re-read per
// task through the same volatile descriptor channel the a2/part bumps use, so
// nothing tuneable is kept live across the MFMA mainloop.
__device__ __forceinline__ void k0p6_mps_task_done(int b, int nc, int tid,
                                                   const long long* k0p6_desc) {
  if (tid != 0) return;
  if (((unsigned long long)k0p6_dread(k0p6_desc, K0P6_D_MPS_CFG) >> 16 &
       0xFFull) == 2ull) {
    std::uint32_t* mps_q = (std::uint32_t*)k0p6_dread(k0p6_desc, K0P6_D_MPS_Q);
    std::uint32_t* mps_state =
        (std::uint32_t*)k0p6_dread(k0p6_desc, K0P6_D_MPS_STATE);
    hk_moe::mps::enqueue_tile_release(
        mps_q, mps_state + K0P6_MPS_ST_TAIL, b, nc);
  } else {
    hk_moe::arrive_local_count(
        ((unsigned int*)k0p6_dread(k0p6_desc, K0P6_D_PART_DONE)) + b);
  }
}

// ---- N2 phase bodies, exp_59 G-stacked variants (n2_phase{1,2}_gm.cpp) --------------------
// The _gm bodies add tile_desc/num_tiles kernargs and the per-live-sub-block hooks. The
// hooks fire per LIVE sub-block b=b0+sb (a2_done/part_done indexed by absolute 32-block),
// reusing the SAME k0p6_a2_arrive/k0p6_a2_wait/k0p6_part_arrive helpers unchanged.
#define N2GM_G K0P6GM_G
#define N2_P1_QUAL __device__ __forceinline__
#define N2_P1_NAME n2p6gm_phase1_body
#define N2_KERNELS_ONLY
#define N2_HOOK_CTX_ARG , const long long* __restrict__ k0p6_desc
#define N2GM_TASK_DONE_HOOK k0p6_a2_arrive(b, tid, k0p6_desc);
#include "n2_phase1_gm.cpp"
#undef N2GM_TASK_DONE_HOOK
#undef N2_HOOK_CTX_ARG
#undef N2_KERNELS_ONLY
#undef N2_P1_NAME
#undef N2_P1_QUAL

#define N2_P2_QUAL __device__ __forceinline__
#define N2_P2_NAME n2p6gm_mps_phase2_body
#define N2_KERNELS_ONLY
#define N2_HOOK_CTX_ARG , const long long* __restrict__ k0p6_desc
#define N2GM_TASK_WAIT_HOOK \
  if (!k0p6_a2_wait(b, tid, k0p6_desc)) return;
#define N2GM_TASK_DONE_HOOK k0p6_mps_task_done(b, nc, tid, k0p6_desc);
#define N2GM_TASK_DONE_DRAIN_HOOK asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
// Donor-identical start (free hardware uniform); the stride is re-derived
// from the descriptor inside the body — the only value the role split costs.
#define N2GM_TASK_START blockIdx.x
#define N2GM_TASK_STRIDE k0p6_mps_stride(k0p6_desc, (int)gridDim.x)
#include "n2_phase2_gm_mps.cpp"
#undef N2GM_TASK_STRIDE
#undef N2GM_TASK_START
#undef N2GM_TASK_DONE_DRAIN_HOOK
#undef N2GM_TASK_DONE_HOOK
#undef N2GM_TASK_WAIT_HOOK
#undef N2_HOOK_CTX_ARG
#undef N2_KERNELS_ONLY
#undef N2_P2_NAME
#undef N2_P2_QUAL
#undef N2GM_G

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

// One M8 combine batch (NT tokens) over a caller-selected row base function.
// The arithmetic is byte-stable with the parity M8 body; splitting the loop
// into pull/slot instantiations at the call site keeps the two address forms
// out of each other's live ranges (resource-gate requirement).
template<int NT, typename BaseFor>
__device__ __forceinline__ void k0p6_mps_m8_batch(
    int tok0, int T, unsigned int epoch32, long long spin_limit,
    const int* __restrict__ pull_ptr, const int* __restrict__ pull_src,
    const unsigned int* __restrict__ row_ready, int T_loc_max,
    int* __restrict__ pperr, int lane, int cur,
    unsigned short* __restrict__ out, BaseFor&& base_for) {
  bool batch_ready = true;
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
      // The consumer below shuffles all 8 lanes of the group unconditionally
      // and only the accumulate is fanout-guarded, so an out-of-fanout lane
      // must still hold a dereferenceable base. `(cur, 0)` names this rank's
      // own row 0 under either address form; the bytes it loads are discarded.
      int p = cur;
      int row = 0;
      if (j2 < fanout[t]) {
        p = pull_src[(size_t)(lo2[t] + j2) * 2 + 0];
        row = pull_src[(size_t)(lo2[t] + j2) * 2 + 1];
        // exp_56 fix (8, site 1/4 — the combine poll = 33554432 bit): CROSS-GPU
        // poll of row_ready (relaxed SYSTEM store). nt-only load never
        // invalidates => spins on a stale row_ready line forever. System-scope
        // atomic load.
        batch_ready = hk_moe::poll_epoch_system(
            row_ready + (size_t)p * T_loc_max + row, epoch32,
            (std::uint64_t)spin_limit,
            hk_moe::set_error_bit{pperr, 33554432});
      }
      pbase[t] = base_for(p, row);
    }
  }
  // A timeout must not mint a usable peer payload. Skip this token batch
  // uniformly across the wave after recording pperr; M9 still retires the
  // launch, so the negative control fails without wedging the node.
  if (__ballot(batch_ready) != ~0ull) return;
  // Only lanes 0..31 poll readiness, while all lanes consume the shuffled
  // bases. Apply one convergent wave acquire before any payload load.
  hk_moe::acquire_payload_system();
  __syncwarp();
#pragma unroll 1
  for (int c = 0; c < 14; ++c) {
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
extern "C" __global__ void __launch_bounds__(256, 1) k0pf6gm_mps_mega(
    const long long* __restrict__ desc, int* __restrict__ pperr, long long spin_limit) {
  if (desc == nullptr || pperr == nullptr) return;
  const int TOPK = (int)k0p6_dread(desc, K0P6_D_TOPK);
  const int E = (int)k0p6_dread(desc, K0P6_D_E);
  const int cur = (int)k0p6_dread(desc, K0P6_D_CUR);
  const int world = (int)k0p6_dread(desc, K0P6_D_WORLD);
  const int T_loc_max = (int)k0p6_dread(desc, K0P6_D_TLOCMAX);
  const int PADMAX = (int)k0p6_dread(desc, K0P6_D_PADMAX);
  const int MAXTOK = (int)k0p6_dread(desc, K0P6_D_MAXTOK);   // exp_56: per-source segment stride
  const int input_tokens = (int)k0p6_dread(desc, K0P6_D_T);
  const long long symmetric_address = k0p6_dread(desc, K0P6_D_SYMMETRIC);
  // exp_05 stage 0: epoch origin for the phase attribution. Scoped to this
  // block so nothing crosses into a later phase; the cell is min-valued
  // (ts_first stores ~t) so the earliest CTA entry wins.
  if (threadIdx.x == 0) {
    std::uint32_t* mps_state0 =
        (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
    const hk_moe::mps::config cfg0 = hk_moe::mps::decode_config(
        (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
    hk_moe::mps::ts_first(
        (std::uint64_t*)(mps_state0 + K0P6_MPS_ST_WORDS) + K0P6_MPS_TS_KSTART,
        cfg0.timestamps);
  }
  __shared__ __align__(16) unsigned char s_row[4][7488];
  __shared__ int s_cnt[K0P6_MAXE];
  __shared__ unsigned int s_total;
  // exp_64: M2 row-striping. Pass A records each (source,chunk) live fill; the per-source sum
  // s_ns[s] is the dense live extent of source s's segment. world*nchunks <= 64, world <= 8.
  // MPS lifetime note (resource-gate): s_fill/s_ns are WRITTEN and last READ
  // only inside M2 (Pass-A tables, Pass-B extents, sentinel pass) and are dead
  // from the M3 barrier onward. M6.9 reuses s_ns[4] as the role broadcast slot
  // and M8's dynamic ticket loop reuses s_ns[0..3] as its per-wave broadcast
  // words. No overlapping live ranges exist for those pairs.
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
  // MPS guard: the config word must be valid, the event wire format must fit
  // every block of this run (bcap = PADMAX/32 <= 16383), and every MPS buffer
  // must be present. The decoded config fields and capacity math stay LOCAL to
  // this scope — the donor's no-cross-phase-state rule applies to MPS tuning
  // words too: every consuming phase reloads its own from k0p6_dread.
  {
    const hk_moe::mps::config entry_cfg = hk_moe::mps::decode_config(
        (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
    if (!hk_moe::mps::config_is_valid(entry_cfg) || PADMAX / 32 > 16383 ||
        k0p6_dread(desc, K0P6_D_MPS_Q) == 0 ||
        k0p6_dread(desc, K0P6_D_MPS_NCARR) == 0 ||
        k0p6_dread(desc, K0P6_D_MPS_PUSHED) == 0 ||
        k0p6_dread(desc, K0P6_D_MPS_CLAIM) == 0 ||
        k0p6_dread(desc, K0P6_D_MPS_STATE) == 0 ||
        (k0p6_dread(desc, K0P6_D_MPS_STATE) & 7) != 0 ||
        k0p6_dread(desc, K0P6_D_MPS_SLOTS) == 0 ||
        (k0p6_dread(desc, K0P6_D_MPS_SLOTS) & 15) != 0) {
      if (bid == 0 && tid == 0) atomicOr(pperr, K0P6_MPS_ERR_CONFIG);
      return;
    }
  }
  const int T_ext = world * MAXTOK;   // exp_56: swept segmented extent
  const int lo = cur * E;
  const long long debug_stop = k0p6_dread(desc, K0P6_D_DEBUG);   // bisection knob (0=run all)

  // ================= M0: retire wait + per-block counter zero =============================
  {
    const std::uint64_t epoch = k0p6_epoch(desc);
    std::uint64_t* retired =
        (std::uint64_t*)k0p6_dread(desc, K0P6_D_RETIRED);
    unsigned int* mega_count = (unsigned int*)k0p6_dread(desc, K0P6_D_EPOCH_CELL);
    (void)mega_count;
    if (tid < world) {
      // Directed replay lifetime: peer retirement must reach epoch-1 before
      // this launch reuses the same addresses.
      (void)hk_moe::poll_value_at_least_system(
          retired + tid, epoch > 0 ? epoch - 1ull : 0ull,
          (std::uint64_t)spin_limit, pperr, 16777216);
    }
    // Retirement protects every address this launch is about to reuse. A CTA-local
    // return would strand peers at later grid barriers, while proceeding after one
    // timeout could overwrite payload a prior peer still consumes. Rendezvous the
    // complete local grid, acquire the published error bit, and return uniformly
    // before the first reset or M1 store. local_grid_epoch is cumulative, so this
    // extra phase requires no counter reset and cannot be satisfied by an old phase.
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
    // MPS per-epoch state: event queue, slice arrivals, push/claim counters,
    // role ticket + reduction ticket + diagnostic timestamps. Zeroed under the
    // same contract as a2_done/part_done above (used no earlier than M6.9;
    // the M3/M5 grid barriers order this zeroing before any reader).
    {
      std::uint32_t* mps_q = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_Q);
      std::uint32_t* mps_arr = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_NCARR);
      std::uint32_t* mps_pushed = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_PUSHED);
      std::uint32_t* mps_claim = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_CLAIM);
      std::uint32_t* mps_state = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
      const int gtid0 = bid * (int)blockDim.x + tid;
      const int gstride0 = nct * (int)blockDim.x;
      for (int i = gtid0; i < (PADMAX / 32) * 16; i += gstride0) mps_q[i] = 0u;
      for (int i = gtid0; i < T_ext * 16; i += gstride0) mps_arr[i] = 0u;
      for (int i = gtid0; i < T_ext; i += gstride0) {
        mps_pushed[i] = 0u;
        mps_claim[i] = 0u;
      }
      for (int i = gtid0; i < K0P6_MPS_ST_WORDS + 2 * K0P6_MPS_TS_COUNT;
           i += gstride0) {
        mps_state[i] = 0u;
      }
    }
    // exp_56 fix (6): GRID-WIDE reset of the NEXT epoch's dest_counter parity, replacing the
    // single-CTA M9 reset. The M9 reset was one store by the elected last-combine-arriver; the
    // retire-wait gates this rank only against PEERS, never against ITSELF, so a different CTA of
    // this same rank reached the next epoch's M0/M1 and read/incremented the slice before the
    // arriver's store propagated => self-slot nonzero (the sound-guard failure on all ranks).
    // CORRECTNESS (accurate reason): this zeroes parity (epoch+1)&1, which NO CTA in THIS epoch
    // reads or writes (M1 dispatch touches only epoch&1) — slice disjointness within the epoch. The
    // ONE-EPOCH-PER-LAUNCH boundary then guarantees all 256 CTAs finish M0 (hence this zeroing)
    // before epoch N+1's launch exists, so N+1 starts with its parity provably zero. No post-reset
    // barrier is needed (the M0 retirement barrier precedes these stores and the M3 barrier is
    // irrelevant to this buffer). Grid-stride over exactly the `world` slots of the next parity,
    // same pattern as a2_done/part_done above.
    {
      unsigned int* dc_next =
          (unsigned int*)k0p6_dread(desc, K0P6_D_DESTCTR)
          + (size_t)(((k0p6_epoch(desc) + 1ull) & 1ull)) * (size_t)world;
      // exp_56 fix (7 / H2): the counter is INCREMENTED via __hip_atomic_fetch_add(SYSTEM) and READ
      // via __hip_atomic_load(SYSTEM). A PLAIN store reset may not be coherent with that atomic path
      // on the MoRI symmetric (possibly Uncached) heap — unlike a2_done/part_done which are read with
      // plain/relaxed AGENT loads. Match the access mode: reset with system-scope atomic stores so
      // reset/increment/read all share one coherent path.
      for (int i = bid * (int)blockDim.x + tid; i < world; i += nct * (int)blockDim.x)
        hk_moe::publish_relaxed<hk_moe::scope::system>(dc_next + i, 0u);
      // exp_56 fix (8) — SOUND base-zero guard (pperr 262144). The EARLIER guard read
      // dest_counter[epoch&1][cur], but M0 has NO post-reset grid barrier before M1, so THIS RANK's
      // own CTAs racing ahead into M1's local dest==cur fetch_add on [epoch&1][cur] concurrently write the
      // slot the guard reads => false positive (my prior "no concurrent writer" argument missed
      // same-rank CTAs, only ruled out peers). RACE-FREE version: verify the slice THIS epoch's M0
      // JUST ZEROED — [(epoch+1)&1][cur] — read back by the SAME thread that zeroed slot `cur`.
      // No CTA in this epoch writes (epoch+1)&1 (M1 touches only epoch&1), so this read has no
      // concurrent writer at all and certifies exactly the reset we just performed.
      {
        const int self = cur;   // the (epoch+1)&1 slot for this rank's own future self-index
        // find the (bid,tid) that owns grid-stride index `self` and let it verify.
        for (int i = bid * (int)blockDim.x + tid; i < world; i += nct * (int)blockDim.x) {
          if (i == self &&
              hk_moe::load_relaxed<hk_moe::scope::system>(dc_next + i) != 0u)
            atomicOr(pperr, 262144);
        }
      }
    }
    // exp_56 NOTE: recv_eid sentinel-init is NOT done here. Doing it in M0 raced M2's recv_eid
    // writes across CTAs (M0 has no post-reset grid barrier before M1||M2; row R is init'd by one CTA's
    // grid-stride but unpacked by another, so a late -1 init clobbered a real value => intermittent
    // stale rows). Instead the sentinel is written at the END of M2 to ONLY the hole rows, which are
    // DISJOINT from live rows, and the existing M3 grid barrier orders it before scatter. See M2.
    __syncthreads();
  }
  if (debug_stop == 1) return;

  // ================= M1: qpush phase (contiguous LDS row + ProtoLL128 putPackets) =========
  int T_loc;
  {
    // Phase-local by design: do not carry peer-descriptor state through M2-M7
    // or either N2 MFMA loop.
    const auto* symmetric = k0p6_symmetric(desc);
    const std::uint64_t epoch = k0p6_epoch(desc);
    // exp_56 fix (5): epoch-parity DOUBLE-BUFFERED dest_counter[2][world]. Producers of epoch N
    // fetch_add into parity N&1; M9 zeroes the OPPOSITE parity (which epoch N+2 will use, a full
    // epoch of slack away). A peer racing ahead into epoch N+1 touches parity (N+1)&1, disjoint
    // from the reset target — so NO reset/dispatch race exists and base-zero is STRUCTURAL, making
    // pos in [0,n_s) hold with no offset. This dissolves the cross-epoch counter lifetime bug.
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

      // exp_56 Option C: SEGMENTED index. dest_counter is per-SOURCE on the destination
      // (dest_counter[src]); this source `cur` increments its own slot on `dest`. The global
      // receive-row is cur*MAXTOK + local_pos, so a row's owner (source) is implicit in its
      // index and holes stay within this source's segment. Waitless: index known immediately.
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
      // Register-lean quant (byte-identical numerics): hold only the packed bf16 raws and
      // unpack twice (amax pass, then quantize pass) instead of holding f[16]; scales and
      // meta land in the LDS tail directly instead of living in registers.  This removes
      // ~35 VGPR of phase pressure (the exp_55-v2/v3 scratch source).
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
      // meta vectors: computed in convergent context (shfl needs all lanes), staged by one
      // lane each, one vector live at a time.
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
        // exp_56: payload-only store, NO per-packet flag (chunk release replaces it).
        hk_moe::store_dispatch_row(
            dst, row64, (size_t)K0P5_ROW_U64, g, j);
      }
      __syncwarp();
    }

    // exp_56 chunk-counted release. Two-part publish (design.md §3, §4):
    //  (1) EVERY producer CTA executes ONE system release fence (buffer_wbl2 sc1), flushing all of
    //      this rank's payload L2 lines over xGMI. 256 fences/rank, NOT per-row (the rejected
    //      39.5x path) and NOT per-packet (the incorrect fast path).
    //  (2) The LAST grid arriver, AFTER all CTAs have fenced, publishes one epoch32 flag per
    //      (source=cur, chunk) on every destination, plus the per-source rows_done count so the
    //      consumer knows exactly how many live chunks to wait on (empty/partial-safe, design.md §4).
    // (1) CHEAP per-CTA release fence (plan cost model: 1 L2 writeback per CTA = 256/rank).
    // Each thread first drains ITS OWN outstanding payload VMEM stores with a near-free
    // s_waitcnt vmcnt(0) (NOT an L2 writeback) so every lane's stores are past L1 before the
    // barrier; then ONE thread issues the directional system release (expected gfx950
    // buffer_wbl2 sc1, the L2 writeback over xGMI). __syncthreads() cannot substitute for the
    // per-thread vmcnt drain
    // (it does not wait on outstanding VMEM on AMD), which is why the tid0-only fence left a few
    // rows stale. The barrier after guarantees all lanes drained before the writeback + count.
    hk_moe::release_cta_payload_system();
    std::uint64_t* chunk_ready =
        (std::uint64_t*)k0p6_dread(desc, K0P6_D_CHUNK_READY);
    const unsigned int epoch32 = (unsigned int)epoch;
    const int nchunks = K0P6C_NCHUNK_MAX;
    if (tid == 0) {
      const unsigned int old = __hip_atomic_fetch_add(qpush_done, 1u, __ATOMIC_ACQ_REL,
                                                      __HIP_MEMORY_SCOPE_AGENT);
      if ((old % gridDim.x) == (gridDim.x - 1)) {
        // System release fence: the AGENT-scope qpush_done acquire proves every producer CTA's
        // per-CTA fence completed locally, but does NOT order those CTAs' system writebacks before
        // this thread's chunk-word stores AS OBSERVED BY A PEER. This fence makes every subsequent
        // system-scope chunk-word store observed-after all prior flushed payloads.
        hk_moe::release_signal_batch_system();
        for (int pe = 0; pe < world; ++pe) {
          // exp_56 fix (i-b): n_s = pushed_count is the TRUE count of rows this source wrote to pe.
          // It equals the placement range [0, n_s) ONLY IF dest_counter started at 0 this epoch —
          // which is now a HARD, verified invariant (M0 zero-check, pperr bit 262144) guaranteeing
          // pos ranges exactly [0, n_s). A readback of dest_counter would return residue+count and
          // reintroduce an offset bug, and would also race peers at this AGENT-scope point; keeping
          // pushed_count + the base-zero guarantee is the smaller, sound fix.
          const unsigned int n_s = pushed_count[pe];   // rows this source sent to dest pe
          // rows_done: STATISTICS ONLY now (T_count for nvi). No correctness path reads it.
          if (pe == cur) {
            hk_moe::publish_epoch_payload<hk_moe::scope::agent>(
                &rows_done[cur], epoch32, n_s);
          } else {
            std::uint64_t* peer =
                hk_moe::peer_ptr(rows_done + cur, pe, symmetric);
            hk_moe::publish_epoch_payload<hk_moe::scope::system>(
                peer, epoch32, n_s);
          }
          // Publish ALL nchunks self-describing chunk words (fill_count=0 for unfilled chunks) so
          // the consumer needs no separate count and no terminator. fill_count = rows of this chunk
          // = clamp(n_s - c*CHUNK, 0, CHUNK). One 64-bit (epoch,fill) store per chunk = the SOLE
          // authority for both readiness and extent, riding the single release fence above.
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
  // no barrier: CTAs flow straight into the streaming phase (M1||M2 overlap).

  // ================= M2: chunk-acquire unpack + histogram + row_remaining (exp_56 Option C) ===
  // Consumer: for each (source s, chunk c) it owns in a wave-stripe, acquire chunk_ready[s][c]
  // ONCE (one buffer_inv sc1), then unpack that chunk's rows with plain non-polling loads. The
  // set of live chunks per source is derived from rows_done[s]'s count (empty/partial-safe).
  // Rows live at segmented indices s*MAXTOK + [c*CHUNK, min((c+1)*CHUNK, n_s)). Holes never
  // touched. T_count = sum n_s (statistics); T_ext used for all bounds/scan/pad.
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

    // Step A: wait until every source has published its rows_done for this epoch, so each n_s is
    // known. This is the same all-in test as the dense path; no barrier, per-source poll.
    if (tid == 0) {
      long long spins = 0;
      int all = 0;
      while (!all) {
        all = 1;
        unsigned int tsum = 0;
        for (int src = 0; src < world; ++src) {
          // exp_56 fix (8, site 3/4): CROSS-GPU poll of rows_done (peer system store). nt-only
          // load never invalidates. Statistics-only NOW (T_count), but it is ALSO the Step-A all-in
          // WAIT loop, so a stale line here can hang Step A (16777216). System-scope atomic load.
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
    const unsigned int total = s_total;   // T_count (live arrivals)

    // exp_64: SPLIT M2 into (A) a coarse acquire/wait pass over the 64 (source,chunk) pairs and
    // (B) a row-parallel unpack+histogram pass over all 1,024 waves. The fused loop striped unpack
    // over npairs=64, so only ~48 of 1,024 resident waves ever ran (M2 analysis 20260806); Pass B
    // stripes the ~21,816 live rows over EVERY wave. The RELEASE PROTOCOL IS UNCHANGED: the chunk
    // word stays the sole readiness+extent authority; Pass A performs the identical per-(source,
    // chunk) coherent SYSTEM poll + convergent buffer_inv acquire that the fused loop did.
    const int npairs = world * nchunks;
    // ---- Pass A: wave-striped chunk-readiness WAIT (the M1||M2 peer-producer overlap). Same
    // coherent SYSTEM poll as before; it now only WAITS (records nothing) — fills are re-derived
    // per-CTA below from the settled words. This preserves the ~42-64 release/acquire events.
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
      (void)ok;   // pperr records failure; every CTA still reaches the rendezvous below
    }
    // ---- Pass A->B grid barrier. Every pair's poll (grid-wide) has resolved to the current epoch,
    // so the per-CTA SYSTEM re-read below is guaranteed to observe current-epoch chunk words. Same
    // local-grid-epoch counter (gbar) as M3-M5; cumulative target makes this a fresh phase, no reset.
    {
      unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
      const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
      const hkp::fail_closed bar_err{pperr, 2097152};
      hkp::grid_barrier(bar, tid, bar_err);
    }
    // Both bounded M2 wait classes converge here. The barrier's agent release/acquire
    // makes any timeout bit visible grid-wide; no CTA may reread a chunk word or touch
    // a_ll unless every rows_done and chunk_ready wait succeeded. A uniform return
    // intentionally leaves the epoch unretired. This timeout is terminal for the
    // current protocol instance: the host must reinitialize all distributed state
    // before another launch, not clear pperr and retry the same epoch in place.
    hk_moe::acquire_payload_agent();
    if (hk_moe::error_bit_set_agent(
            pperr, 16777216 | 8388608 | 2097152)) return;
    // exp_05 stage 0: the dispatch is complete grid-wide at this point (M1 send
    // + M2 per-(source,chunk) acquire, closed by the barrier above). This is the
    // measurement that turns the 757 us M1+M2 figure from INFERRED to measured.
    if (tid == 0) {
      std::uint32_t* mps_state2 =
          (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
      const hk_moe::mps::config cfg2 = hk_moe::mps::decode_config(
          (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
      hk_moe::mps::ts_last(
          (std::uint64_t*)(mps_state2 + K0P6_MPS_ST_WORDS) +
              K0P6_MPS_TS_M2_DONE,
          cfg2.timestamps);
    }
    // ---- Per-CTA fill/ns table from the (now settled, current-epoch) chunk words. Every CTA needs
    // ALL 64 fills because Pass B waves unpack rows from any source. SYSTEM atomic load invalidates
    // and re-fetches; the barrier guarantees the word carries this epoch. s_ns[s] = sum of source s's
    // chunk fills = its dense live-row extent (segments are prefix-dense: exp_56 base-zero invariant).
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
    // ---- Per-CTA convergent SYSTEM acquire. Pass A's per-poll buffer_inv was per-CU; Pass B waves
    // read a_ll on CUs that never polled. All producer a_ll payload landed BEFORE the barrier (Pass A
    // confirmed publication), and no a_ll write happens this epoch after it, so ONE buffer_inv sc1
    // here orders every subsequent a_ll load after the invalidate — the exact hazard documented in
    // k0pf6_chunk_release.hpp:113-118, hoisted from per-chunk to once-per-CTA.
    hk_moe::acquire_payload_system();   // convergent pure acquire
    __syncthreads();

    // ---- Pass B: row-parallel unpack + histogram over the segmented EXTENT, warp-per-row, striped
    // across ALL waves. Row r belongs to source s=r/MAXTOK at offset off=r%MAXTOK; it is LIVE iff
    // off < s_ns[s] (dense-prefix), the EXACT union of the per-chunk [base,base+fill) ranges the
    // fused loop unpacked. Each r has a unique owner warp (r mod wave_count), so every live row is
    // unpacked once, counted once, and row_remaining[r] written once — bit-identical grid histogram.
    for (int r = wave; r < T_ext; r += wave_count) {
      const int s = r / MAXTOK;
      const int off = r - s * MAXTOK;
      if (off >= s_ns[s]) continue;   // hole / beyond this source's live extent: never unpack
      k0p6_unpack_row_nopoll((size_t)r, a_ll + (size_t)r * (size_t)K0P5_ROW_STRIDE, a_dst,
                             sc_stage, recv_eid, recv_wgt, g, j);
      __syncwarp();
      if (lane < 8) {   // TOPK = 8 (shape-guarded)
        const int e = recv_eid[(size_t)r * 8 + lane];
        const bool live_e = (e >= lo && e < lo + E);
        if (live_e) atomicAdd(&s_cnt[e - lo], 1);
        const unsigned int live_ballot = __ballot(live_e) & 0xFFu;
        if (lane == 0) row_remaining[r] = (unsigned int)__popcll(live_ballot);
      }
      __syncwarp();
    }

    T_loc = (int)total;   // T_count; extent-driven bounds use T_ext below
    if (T_ext < 0 || T_ext > T_loc_max) {
      if (bid == 0 && tid == 0) atomicOr(pperr, 524288);
      return;
    }

    // exp_56 sentinel (RACE-FREE, no new barrier, CHUNK-WORD authority — no rows_done read).
    // Write recv_eid = -1 to EVERY hole row so scatter's blind [0,pairs) scan skips them via its
    // expert-range filter. Per (source s, chunk c): live rows are [base, base+fill), holes are
    // [base+fill, base+512). Summed over all 8 chunks this covers the whole segment, including
    // fully-empty chunks (fill=0 => the entire 512-row chunk is holes). Derived from the SETTLED
    // self-describing chunk words (the sole authority), NOT rows_done — so no correctness path reads
    // rows_done. Holes are DISJOINT from the live rows M2 unpacked => no cross-CTA race with unpack.
    // The existing M3 grid barrier orders every CTA's hole write before scatter reads recv_eid.
    {
      int* recv_eid_h = (int*)k0p6_dread(desc, K0P6_D_RECV_EID);
      const int gtid = bid * (int)blockDim.x + tid;
      const int gstride = nct * (int)blockDim.x;
      for (int s = 0; s < world; ++s) {
        for (int c = 0; c < nchunks; ++c) {
          // exp_56 fix (8, site 4/4): SECOND cross-GPU read of chunk_ready (peer system store),
          // used to derive hole extents. nt-only load never invalidates => could read a stale word
          // and sentinel the wrong hole range. System-scope atomic load, same as the Step-B poll.
          const std::uint64_t w = hk_moe::load_relaxed<hk_moe::scope::system>(
              chunk_ready + (size_t)s * (size_t)nchunks + (size_t)c);
          // word is settled: Step B already polled/acquired every chunk this rank consumes, and the
          // producer published all nchunks unconditionally. epoch match guaranteed by Step B success.
          const unsigned int fill = (unsigned int)(w & 0xffffffffu);
          const long long base = (long long)s * (long long)MAXTOK + (long long)c * K0P6_CHUNK;
          const long long hole0 = (base + (long long)fill) * TOPK;
          const long long hole1 = (base + (long long)K0P6_CHUNK) * TOPK;
          for (long long i = hole0 + gtid; i < hole1; i += gstride) recv_eid_h[i] = -1;
        }
      }
    }

    __syncthreads();
    for (int i = tid; i < E; i += blockDim.x) hcnt[i * nct + bid] = s_cnt[i];
  }
  if (debug_stop == 3) return;

  // ================= M3-M5: plan + scatter (k0pf4_dsort verbatim, inline) ==================
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
      // exp_56: nvi[1] must be the EXTENT (pad-sentinel / xtok<T mask boundary), not live count.
      k0p6_sort::scan(s_cnt, E, T_ext, scratch, nvi, tid);
      k0p6_sort::cursors_sei(scratch, E, PADMAX, sei, pperr, 4194304, tid);
      // exp_59: build the expert-aligned tile table on device. scratch[OffErb+e]
      // (=K0P6_SC_ERB+e) is the per-expert padded row-begin from scan(); each
      // expert e contributes ceil(nsb_e/G) tiles of first-32-block (blk0_e + t*G)
      // and gcount = min(G, nsb_e - t*G). Serial on tid 0 (E<=64, <=~ceil(1110/32)/G
      // tiles/expert): negligible vs the GEMM. Packs (b0<<4)|gcount into tile_desc.
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
          const int nsb = blk1 - blk0;              // live+pad 32-blocks for e
          for (int t = 0; t * G < nsb; ++t) {
            const int b0 = blk0 + t * G;
            int gc = nsb - t * G;
            if (gc > G) gc = G;                      // live sub-blocks in this tile
            if (nt >= bcap) { atomicOr(pperr, 4194304); break; }
            tile_desc[nt++] = (b0 << 4) | (gc & 0xF);
          }
        }
        num_tiles_p[0] = nt;
        // exp_59 safety: stamp the COMPILED G so the host fails closed on a mismatch
        // vs its own PF6GM_G (a silent kernel/host G disagreement already contaminated
        // one run this session). num_tiles is int[2]: [0]=count, [1]=compiled G.
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
      // exp_56: zero/transpose over the segmented EXTENT (holes must be zeroed too, else stale
      // part rows corrupt combine). T_ext, not the live count.
      for (int t = w; t < T_ext; t += nw2) {
        hkp::zero_part_scale_transpose<14>(part + (std::size_t)t * K0P6_H, sc_dst,
                                           sc_stage + (std::size_t)t * K0P6_NG, t, T_ext,
                                           K0P6_NG, lane2);
      }
    }

    hkp::grid_barrier(bar, tid, bar_err);   // M4 publishes scratch/nvi/sei/pull_ptr
    if (hk_moe::error_bit_set_agent(pperr, 2097152)) return;

    // exp_56: scatter SCAN bound and pad sentinel use T_ext (recv_eid holes are -1 => skipped).
    const int pairs = T_ext * TOPK;
    k0p6_sort::pad(scratch, E, T_ext, TOPK, PADMAX, sti, swt, pperr, 4194304);
    k0p6_sort::scatter(recv_eid, recv_wgt, pairs, lo, E, TOPK, PADMAX, scratch, sti, swt,
                       pperr, 1048576);
    hkp::pull_src_fill(pull_stage, pull_ptr, T, TOPK, T * world, pull_src, pperr, 1048576);

    hkp::grid_barrier(bar, tid, bar_err);   // M5 publishes sti/swt/pull_src/part/sc_dst
    if (hk_moe::error_bit_set_agent(pperr, 2097152)) return;
  }
  if (debug_stop == 4) return;

  // ================= M6: N2 phase1 (G-stacked; a2_done arrivals per live sub-block) =========
  production_fused_moe::n2::n2p6gm_phase1_body(
      (const std::uint8_t*)k0p6_dread(desc, K0P6_D_A_DST),
      (const float*)k0p6_dread(desc, K0P6_D_SC_DST),
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
  // exp_05 stage 0: M6 (N2 phase 1) is done for this CTA. With KSTART and
  // M2_DONE this closes the pre-M7 decomposition: dispatch = M2_DONE - KSTART,
  // plan+M6 = M6_DONE - M2_DONE.
  if (tid == 0) {
    std::uint32_t* mps_state6 =
        (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
    const hk_moe::mps::config cfg6 = hk_moe::mps::decode_config(
        (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
    hk_moe::mps::ts_last(
        (std::uint64_t*)(mps_state6 + K0P6_MPS_ST_WORDS) + K0P6_MPS_TS_M6_DONE,
        cfg6.timestamps);
  }

  // ================= M6.9: minimum-progress role selection (static tail) =======
  // The C TAIL CTAs of this grid become the minimum-progress service pool; all
  // other CTAs stride the M7 task space by (256-C), so every (tile, nc) task
  // still executes exactly once. Mode 1 has C=0 by validation: no reservation,
  // no overlap, full-grid stride exactly like the donor.
  //
  // Why static-tail, not finish-order tickets: the resource gate showed that
  // ANY value crossing the M6→M7 boundary costs VGPR-pair spill slots at the
  // phase-2 pointer peak (measured +24 B/lane and the SGPR spill blowup).
  // Physical tail CTAs are a *free* role test — blockIdx.x is a hardware
  // uniform, and stride (256-C) is re-derived inside the body from the
  // descriptor. On this kernel's uniform-cost bid-interleaved task stripes,
  // M6 completion order has no bid-correlated bias, so the tail reservation's
  // service-start statistics match a random C-subset; the queue absorbs the
  // skew that finish-order tickets would have covered. No grid barrier added.
  //
  // Absolutely NO MPS value is allowed to live across the M7 MFMA body: the
  // service predicate below and the M7-tail timestamp decode are the ONLY
  // sites needing C/mode here, and each reloads the config word locally (the
  // donor's per-phase descriptor discipline, verbatim).
  // ================= M7: N2 phase2 over the logical compute pool ================
  {
    const std::uint32_t mps_C =
        (std::uint32_t)((unsigned long long)k0p6_dread(desc, K0P6_D_MPS_CFG) &
                        0xFFull);
    if (bid < nct - (int)mps_C) {
      production_fused_moe::n2::n2p6gm_mps_phase2_body(
          (const std::uint8_t*)k0p6_dread(desc, K0P6_D_A2Q),
          (const float*)k0p6_dread(desc, K0P6_D_DQ2),
          (const std::uint8_t*)k0p6_dread(desc, K0P6_D_W2),
          (const float*)k0p6_dread(desc, K0P6_D_S2),
          (const int*)k0p6_dread(desc, K0P6_D_STI),
          (const float*)k0p6_dread(desc, K0P6_D_SWT),
          (const int*)k0p6_dread(desc, K0P6_D_SEI),
          (const int*)k0p6_dread(desc, K0P6_D_NVI),
          (__hip_bfloat16*)k0p6_dread(desc, K0P6_D_PART),
          (const int*)k0p6_dread(desc, K0P6_D_TILE_DESC),
          (int)((const int*)k0p6_dread(desc, K0P6_D_NUM_TILES))[0], desc);
      if (tid == 0) {
        std::uint32_t* mps_state =
            (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
        std::uint64_t* mps_ts = (std::uint64_t*)(mps_state + K0P6_MPS_ST_WORDS);
        const hk_moe::mps::config m7cfg = hk_moe::mps::decode_config(
            (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
        hk_moe::mps::ts_last(mps_ts + K0P6_MPS_TS_M7_DONE, m7cfg.timestamps);
      }
    }
  }
  if (debug_stop == 6) return;

  // ================= M7.6: minimum-progress service waves (stream mode) ========
  // Wave-striped consumption of the (b, nc) event queue: slice arrival
  // bookkeeping, group claiming, payload push into owner-resident slots, and
  // batched owner-only row_ready publication. Runs ONLY on the reserved
  // service CTAs (stride 0) in mode 2. All waits are bounded and watch pperr;
  // a failure exits the loop and the launch fails closed through the usual
  // paths.
  {
    // Phase-local service predicate (same formula as the M7 guard above).
    const hk_moe::mps::config scfg = hk_moe::mps::decode_config(
        (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
    const bool service_cta_here =
        scfg.mode == 2u && bid >= nct - (int)scfg.reserved_comm_ctas;
    if (service_cta_here) {
      const auto* symmetric = k0p6_symmetric(desc);
      const int* nvi = (const int*)k0p6_dread(desc, K0P6_D_NVI);
      const std::uint32_t events_total = ((std::uint32_t)nvi[0] >> 5) * 16u;
      const std::uint32_t queue_capacity = (std::uint32_t)(PADMAX / 32) * 16u;
      std::uint32_t* mps_state =
          (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
      if (events_total > queue_capacity) {
        if (tid == 0) atomicOr(pperr, K0P6_MPS_ERR_CONFIG);
      } else {
        hk_moe::mps::service_env env;
        env.q = (const std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_Q);
        env.nc_arr = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_NCARR);
        env.pushed = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_PUSHED);
        env.claim = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_CLAIM);
        env.row_rem = (std::uint32_t*)k0p6_dread(desc, K0P6_D_ROW_REM);
        env.sti = (const int*)k0p6_dread(desc, K0P6_D_STI);
        env.part = (const unsigned short*)k0p6_dread(desc, K0P6_D_PART);
        env.slots = (unsigned short*)k0p6_dread(desc, K0P6_D_MPS_SLOTS);
        env.row_ready = (std::uint32_t*)k0p6_dread(desc, K0P6_D_ROW_READY);
        env.pperr = pperr;
        env.ts = (std::uint64_t*)(mps_state + K0P6_MPS_ST_WORDS);
        env.symmetric = symmetric;
        env.cur = cur;
        env.world = world;
        env.maxtok = MAXTOK;
        env.t_ext = T_ext;
        env.t_loc_max = T_loc_max;
        env.epoch32 = (std::uint32_t)k0p6_epoch(desc);
        env.group_slices = scfg.group_slices;
        env.flush_rows = scfg.flush_rows;
        env.spin_limit = (std::uint64_t)spin_limit;
        env.events_total = events_total;
        env.queue_capacity = queue_capacity;
        env.ts_enable = scfg.timestamps;
        // The wave scratch unions with the M1 staging slice: M1's s_row use
        // is long dead by M7.6, and 524 B per wave fits s_row[wid]'s 7488 B.
        static_assert(sizeof(hk_moe::mps::wave_scratch) <= 7488,
                      "MPS wave scratch must fit the reused M1 staging slice");
        hk_moe::mps::wave_scratch* mps_scratch =
            reinterpret_cast<hk_moe::mps::wave_scratch*>(&s_row[wid][0]);
        hk_moe::mps::run_service(
            env, *mps_scratch,
            (std::uint32_t)(bid - (nct - (int)scfg.reserved_comm_ctas)),
            scfg.reserved_comm_ctas, wid, lane);
      }
    }
  }

  // ================= M7.5 (exp_62): HOISTED coarse combine-release =========================
  // MODE 0 ONLY (reserve-only control): the parity publication path verbatim,
  // including the mandatory local-grid rendezvous. Modes 1/2 replace it (bulk
  // push below; streaming in M7.6). Monotonic launch-fatal state: if any A2
  // wait timed out, all task bodies returned before touching stale A2/DQ2.
  // Every CTA still participates in the rendezvous below, then suppresses
  // readiness and executes an empty M8 so M9 can retire the failed launch
  // without wedging.
  // The per-task combine-release tail was removed from k0p6_part_arrive (see its body above):
  // phase2's GEMM tasks now pay NO cross-scope release work in-wave. Publish row_ready for every
  // LIVE receive-row ONCE here, AFTER all phase2 tasks have written their part rows. This is the
  // A corrected three-stage publication built from the exp_56 M1 release discipline:
  //  (1) every producer CTA drains its OWN part VMEM stores (s_waitcnt vmcnt(0)) then ONE thread
  //      issues one directional system release (expected gfx950 buffer_wbl2 sc1), flushing this
  //      CTA's part L2 lines
  //      over xGMI. 256 fences/rank, NOT per-task (the old M7 tail) and NOT per-row (rejected 39.5x).
  //  (2) one grid barrier orders ALL CTAs' part writes + flushes before ANY row_ready store.
  //  (3) after acquiring that rendezvous, each CTA leader executes its own SYSTEM release and publishes
  //      a bid/nct stripe of row_ready[cur][r]=epoch32 (self AGENT, peers SYSTEM — the exact scopes
  //      the folded tail used). The same leader performs both its release and every relaxed store;
  //      one lane's fence is never used to order another lane's publication. row_remaining[r] (built by
  //      M2 = popcount of this row's live experts) is the per-epoch liveness bitmap: nonzero => a
  //      real receive-row, 0 => a hole (skipped). A partial/empty grid slice still publishes every
  //      live row it owns, so no consumer can hang. Each relaxed row_ready store is release-ordered
  //      by its publishing leader's preceding fence, and the M8 poll SYSTEM-acquires. The second
  //      leader release is necessary to carry the grid-acquired payload ordering to remote flags.
  if (hk_moe::mps::decode_config(
          (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG)).mode == 0u) {
    unsigned int* row_remaining = (unsigned int*)k0p6_dread(desc, K0P6_D_ROW_REM);
    unsigned int* row_ready = (unsigned int*)k0p6_dread(desc, K0P6_D_ROW_READY);
    const unsigned int epoch32 = (unsigned int)k0p6_epoch(desc);
    // (1) per-CTA system release of this CTA's part payload (exp_56 M1 idiom, lines ~532-534):
    // drain THIS thread's stores to L2 (cheap vmcnt(0), NOT an L2 writeback), barrier so all lanes
    // are drained, then ONE thread issues the single L2 writeback for the CTA.
    hk_moe::release_cta_payload_system();
    // (2) grid barrier: every phase2 task's part write + every CTA's writeback is now globally
    // ordered before any row_ready store. Reuse the M3-M5 local grid-epoch counter (gbar); its
    // cumulative target makes this a fresh phase, no reset needed.
    {
      unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
      const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
      const hkp::fail_closed bar_err{pperr, 2097152};
      hkp::grid_barrier(bar, tid, bar_err);
    }
    hk_moe::acquire_payload_agent();
    const bool payload_ok = !hk_moe::error_bit_set_agent(
        pperr, 16777216 | 2097152);
    // CTA-leader-only publication keeps the formal release sequence and limits the second-stage
    // release count to 256/rank. Nonleaders may enter M8 immediately and poll while leaders publish.
    if (payload_ok && tid == 0) {
      const auto* symmetric = k0p6_symmetric(desc);
      hk_moe::release_signal_batch_system();
      // Each CTA leader owns r = bid (mod nct) over the receive-row EXTENT. row_remaining!=0
      // selects live rows; holes are 0 and skipped.
      for (int r = bid; r < T_ext; r += nct) {
        if (hk_moe::load_relaxed<hk_moe::scope::agent>(row_remaining + r) == 0u)
          continue;   // hole row: nothing certified it in M2, no combine reads it
        unsigned int* ready_slot = row_ready + (size_t)cur * T_loc_max + r;
        for (int R = 0; R < world; ++R) {
          if (R == cur) {
            hk_moe::publish_epoch<hk_moe::scope::agent>(
                ready_slot, epoch32);
          } else {
            unsigned int* peer = hk_moe::peer_ptr(
                ready_slot, R, symmetric);
            hk_moe::publish_epoch<hk_moe::scope::system>(
                peer, epoch32);
          }
        }
        // Self-clean: restore the old folded invariant (every live row ended each epoch at 0 via the
        // per-task decrement; holes stayed 0). Row r is owned by exactly one CTA leader in this
        // grid-strided pass, so this store has no concurrent reader/writer. Keeps the free-running
        // soak/campaign from accumulating stale-nonzero holes that would bloat later epochs' pass.
        hk_moe::publish_relaxed<hk_moe::scope::agent>(row_remaining + r, 0u);
      }
    }
  }

  // ================= M7.6b (mode 1): BULK PUSH after the parity barrier =========
  // Isolates the copy/layout improvement from the overlap improvement: all 256
  // CTAs ran M7 unreserved; payload transport switches to posted pushes into
  // owner-resident slots; flags go to the row's owner ONLY (1 store vs 8);
  // publication timing is still bulk (one barrier, then stripes).
  if (hk_moe::mps::decode_config(
          (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG)).mode == 1u) {
    unsigned int* row_remaining = (unsigned int*)k0p6_dread(desc, K0P6_D_ROW_REM);
    std::uint32_t* row_ready = (std::uint32_t*)k0p6_dread(desc, K0P6_D_ROW_READY);
    unsigned short* slots = (unsigned short*)k0p6_dread(desc, K0P6_D_MPS_SLOTS);
    unsigned short* part_mut = (unsigned short*)k0p6_dread(desc, K0P6_D_PART);
    const unsigned int epoch32 = (unsigned int)k0p6_epoch(desc);
    // (1) each CTA drains its OWN part atomics; (2) one grid barrier orders
    // every phase-2 task's part writes before any CTA reads part rows for
    // pushing; both are identical to the mode-0 rendezvous above.
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
    if (payload_ok) {
      const auto* symmetric = k0p6_symmetric(desc);
      const size_t row_bytes = (size_t)K0P6_H * 2u;
      // (3) push whole rows, grid-strided by CTA: bid owns rows r ≡ bid (mod
      // nct). Controlled 16-byte packet stores, 896 packets per row.
      for (int r = bid; r < T_ext; r += nct) {
        if (hk_moe::load_relaxed<hk_moe::scope::agent>(row_remaining + r) == 0u)
          continue;
        const int owner = r / MAXTOK;
        const int pos = r - owner * MAXTOK;
        const std::byte* src =
            reinterpret_cast<const std::byte*>(part_mut) + (size_t)r * row_bytes;
        std::byte* dst_local = reinterpret_cast<std::byte*>(
            slots + ((size_t)cur * (size_t)MAXTOK + (size_t)pos) *
                        (size_t)K0P6_H);
        std::byte* dst = (owner == cur) ? dst_local
            : hk_moe::peer_ptr(dst_local, owner, symmetric);
        kittens::distributed::store_peer_packets(dst, src, row_bytes,
                                                 (unsigned int)tid,
                                                 (unsigned int)blockDim.x);
      }
      // (4) drain the whole CTA's posted pushes, one leader system release,
      // then the leader owns every flag store of its stripe (same-thread
      // discipline). Flags go to the owner rank only; self-clean preserved.
      kittens::distributed::producer_drain_release<kittens::distributed::memory_scope::system>();
      if (tid == 0) {
        for (int r = bid; r < T_ext; r += nct) {
          if (hk_moe::load_relaxed<hk_moe::scope::agent>(row_remaining + r) == 0u)
            continue;
          const int owner = r / MAXTOK;
          std::uint32_t* ready_slot =
              row_ready + (size_t)cur * (size_t)T_loc_max + (size_t)r;
          if (owner == cur) {
            hk_moe::publish_epoch<hk_moe::scope::agent>(ready_slot, epoch32);
          } else {
            hk_moe::publish_epoch<hk_moe::scope::system>(
                hk_moe::peer_ptr(ready_slot, owner, symmetric), epoch32);
          }
          hk_moe::publish_relaxed<hk_moe::scope::agent>(row_remaining + r, 0u);
        }
      }
    }
  }

  // ================= M8: streaming mtab4 combine (everything re-read from desc) ============
  // Four structural paths, one stable arithmetic body (k0p6_mps_m8_batch):
  //   mode 0 (reserve-only): parity behavior — parity's exact static wave
  //                          stripes over peer-part pulls.
  //   mode 1 (bulk-push):    static wave stripes over owner-LOCAL slot rows.
  //   mode 2 (stream):       DYNAMIC per-wave batch tickets so early-drained
  //                          compute CTAs start immediately and late-joining
  //                          service CTAs mix in; owner-local slot reads (or
  //                          peer-part pulls under the pull_fallback diagnostic).
  // The pull and local-slot address forms live in DISJOINT branches now, and
  // the static loop is restored separately from the dynamic-ticket loop:
  // resource-gate terms. Payload validity is computed phase-locally here (the
  // parity block's cross-phase bool is gone): any error bit on this CTA
  // suppresses all batches; the counted M9 arrival still retires a failed
  // launch.
  {
    const hk_moe::mps::config m8cfg = hk_moe::mps::decode_config(
        (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
    const bool m8_pull = (m8cfg.mode == 0u) || m8cfg.pull_fallback;
    const bool m8_dynamic = (m8cfg.mode == 2u);
    std::uint32_t* mps_state = (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
    std::uint64_t* mps_ts = (std::uint64_t*)(mps_state + K0P6_MPS_ST_WORDS);
    // Keep every M8 descriptor-derived value inside this phase scope. M9
    // deliberately reloads its own state after the scope ends. Only the pull
    // paths materialize the IRIS descriptor (one of the six counted sites).
    const auto* symmetric = m8_pull ? k0p6_symmetric(desc) : nullptr;
    const unsigned short* part = m8_pull
        ? (const unsigned short*)k0p6_dread(desc, K0P6_D_PART) : nullptr;
    const unsigned short* slots = m8_pull
        ? nullptr : (const unsigned short*)k0p6_dread(desc, K0P6_D_MPS_SLOTS);
    const std::uint64_t epoch = k0p6_epoch(desc);
    const unsigned int epoch32 = (unsigned int)epoch;
    const bool payload_ok = !hk_moe::error_bit_set_agent(
        pperr, 16777216 | 2097152 |
        (m8_dynamic ? (K0P6_MPS_ERR_SERVICE | K0P6_MPS_ERR_CONFIG) : 0));
    const int T = payload_ok ? (int)k0p6_dread(desc, K0P6_D_T) : 0;
    unsigned short* out = (unsigned short*)k0p6_dread(desc, K0P6_D_OUT);
    const int* pull_ptr = (const int*)k0p6_dread(desc, K0P6_D_PULL_PTR);
    const int* pull_src = (const int*)k0p6_dread(desc, K0P6_D_PULL_SRC);
    const unsigned int* row_ready =
        (const unsigned int*)k0p6_dread(desc, K0P6_D_ROW_READY);

    // mode 0 (+pull_fallback): the producer's remote part row over xGMI.
    // slot modes: the producer's slice-landing slot in THIS rank's own memory
    // — row names a row on producer p in [cur*MAXTOK, (cur+1)*MAXTOK).
    const auto base_pull =
        [part, symmetric](int p, int row) -> unsigned long long {
      return (unsigned long long)hk_moe::peer_ptr(
          part + (size_t)row * (size_t)K0P6_H, p, symmetric);
    };
    const auto base_slot =
        [slots, cur, MAXTOK](int p, int row) -> unsigned long long {
      return (unsigned long long)(slots +
             ((size_t)p * (size_t)MAXTOK + (size_t)row -
              (size_t)cur * (size_t)MAXTOK) *
                 (size_t)K0P6_H);
    };

    const int gw = bid * (blockDim.x >> 6) + wid;
    const int nw = nct * (blockDim.x >> 6);
    constexpr int NT = 4;
    if (!m8_dynamic) {
      // Parity's static wave stripes, one per address form.
      if (m8_pull) {
        for (int tok0 = gw * NT; tok0 < T; tok0 += nw * NT) {
          k0p6_mps_m8_batch<NT>(tok0, T, epoch32, spin_limit, pull_ptr,
                                pull_src, row_ready, T_loc_max, pperr, lane,
                                cur, out, base_pull);
        }
      } else {
        for (int tok0 = gw * NT; tok0 < T; tok0 += nw * NT) {
          k0p6_mps_m8_batch<NT>(tok0, T, epoch32, spin_limit, pull_ptr,
                                pull_src, row_ready, T_loc_max, pperr, lane,
                                cur, out, base_slot);
        }
      }
    } else {
      // Stream mode: dynamic per-wave batch claiming through the M8 ticket.
      // The broadcast word reuses dead M2 storage (see s_ns lifetime note).
      const int nbatches = (T + NT - 1) / NT;
      std::uint32_t* m8_next = mps_state + K0P6_MPS_ST_M8NEXT;
      int* batch_slot = &s_ns[wid];
      if (m8_pull) {
        while (true) {
          if (lane == 0) {
            *batch_slot =
                kittens::distributed::fetch_add_relaxed<hk_moe::scope::agent>(
                    m8_next, 1u);
          }
          __syncwarp();
          const int batch = *batch_slot;
          if (batch >= nbatches) break;
          k0p6_mps_m8_batch<NT>(batch * NT, T, epoch32, spin_limit, pull_ptr,
                                pull_src, row_ready, T_loc_max, pperr, lane,
                                cur, out, base_pull);
        }
      } else {
        while (true) {
          if (lane == 0) {
            *batch_slot =
                kittens::distributed::fetch_add_relaxed<hk_moe::scope::agent>(
                    m8_next, 1u);
          }
          __syncwarp();
          const int batch = *batch_slot;
          if (batch >= nbatches) break;
          k0p6_mps_m8_batch<NT>(batch * NT, T, epoch32, spin_limit, pull_ptr,
                                pull_src, row_ready, T_loc_max, pperr, lane,
                                cur, out, base_slot);
        }
      }
    }
    if (tid == 0) {
      hk_moe::mps::ts_last(mps_ts + K0P6_MPS_TS_REDUCE_DONE,
                           m8cfg.timestamps);
    }
  }

  // ================= M9: counted arrival, resets (BEFORE poke), retired pokes ============
  {
    __syncthreads();
    if (tid == 0) {
      unsigned int* combine_done =
          (unsigned int*)k0p6_dread(desc, K0P6_D_COMB_DONE);
      const unsigned int old = __hip_atomic_fetch_add(combine_done, 1u, __ATOMIC_ACQ_REL,
                                                      __HIP_MEMORY_SCOPE_AGENT);
      if ((old % gridDim.x) == (gridDim.x - 1)) {
        // Reload for M9 instead of retaining M8 state as a cross-phase value.
        const auto* symmetric_m9 = k0p6_symmetric(desc);
        const std::uint64_t epoch = k0p6_epoch(desc);
        std::uint64_t* retired =
            (std::uint64_t*)k0p6_dread(desc, K0P6_D_RETIRED);
        unsigned int* dest_counter = (unsigned int*)k0p6_dread(desc, K0P6_D_DESTCTR);
        unsigned int* pushed_count = (unsigned int*)k0p6_dread(desc, K0P6_D_PUSHED);
        unsigned int* qpush_done = (unsigned int*)k0p6_dread(desc, K0P6_D_QPUSH_DONE);
        unsigned int* mega_count = (unsigned int*)k0p6_dread(desc, K0P6_D_EPOCH_CELL);
        for (int pe = 0; pe < world; ++pe) pushed_count[pe] = 0u;
        // exp_56 fix (6): dest_counter reset MOVED to M0 (grid-wide, next-parity). It is NOT reset
        // here anymore — a single-CTA reset had no intra-rank ordering against the next epoch. Do
        // not reintroduce it: a stale second writer would race the M0 grid-wide reset.
        (void)dest_counter;
        qpush_done[0] = 0u;
        combine_done[0] = 0u;
        mega_count[0] = (unsigned int)epoch;
        hk_moe::release_signal_batch_agent();
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

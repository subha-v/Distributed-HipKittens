#pragma once

// Minimum-progress specialization (MPS) workload adapter for the PF6 MoE
// megakernel. This header owns operator policy ONLY: descriptor slot ids, the
// runtime role/config word, the (b, nc) tile-event wire format, slice arrival
// accounting, owner-slot posting/fan-out, and diagnostic timestamps. Generic
// mechanisms stay in kittens::distributed (see include/cdna4/ops/group/
// distributed/roles.cuh); the LL128/GEMM/sort plumbing remains in the donor
// helpers. Design contract: DESIGN_MPS.md.

#include "moe_hk_adapter.cuh"

#include <cstddef>
#include <cstdint>

namespace hk_moe::mps {

// ---- descriptor extension (appended after K0P6_D_SYMMETRIC = 55) ------------
// Host and device must agree exactly; mirrored by moe_host_abi.hpp.
#define K0P6_D_MPS_Q 56        // uint32[qcap] tile-event slots (MARK|(b<<4)|nc)
#define K0P6_D_MPS_NCARR 57    // uint32[T_ext*16] slice arrival counters
#define K0P6_D_MPS_PUSHED 58   // uint32[T_ext] pushed slice counts (target 16)
#define K0P6_D_MPS_CLAIM 59    // uint32[T_ext] per-row push-group claim bits
#define K0P6_D_MPS_STATE 60    // uint32[8] scalars + uint64[8] diag timestamps
#define K0P6_D_MPS_SLOTS 61    // bf16[world][MAXTOK][7168] owner landing slots
#define K0P6_D_MPS_CFG 62      // packed uint64 role/config word (host-written)
#define K0P6_MPS_D_LEN 63

// ---- pperr bits added by MPS (donor/parity bits 16..25 are unchanged) -------
#define K0P6_MPS_ERR_SERVICE 67108864    // 1<<26: service/event poll timeout
#define K0P6_MPS_ERR_CONFIG 268435456    // 1<<27: MPS config/shape guard

// ---- K0P6_D_MPS_STATE scalar word lanes -------------------------------------
#define K0P6_MPS_ST_TICKET 0   // finish-order role tickets (agent monotonic)
#define K0P6_MPS_ST_TAIL 1     // event tail ticket (monotonic, capacity-bound)
#define K0P6_MPS_ST_M8NEXT 2   // dynamic reduction batch ticket
#define K0P6_MPS_ST_WORDS 8    // scalar words precede the uint64 diag block
#define K0P6_MPS_TS_FIRST_READY 0   // max(~t) over event enqueues (invert: min)
#define K0P6_MPS_TS_LAST_READY 1    // max(t)   over event enqueues
#define K0P6_MPS_TS_DRAIN 2         // max(t)   over service-wave stripe ends
#define K0P6_MPS_TS_M7_DONE 3       // max(t)   over compute-CTA task drain
#define K0P6_MPS_TS_REDUCE_DONE 4   // max(t)   over CTA reduction exits
// exp_05 stage 0: whole-kernel phase attribution. With these three the epoch
// decomposes end to end without any debug_stop bisect (which aborts the ranks
// on this build): dispatch = M2_DONE - KSTART, plan+M6 = M6_DONE - M2_DONE,
// M7 = M7_DONE - M6_DONE, combine = REDUCE_DONE - M7_DONE. Diagnostic only:
// every write is behind cfg.timestamps and off by default.
#define K0P6_MPS_TS_KSTART 5        // min(t)   over CTA entry (stored inverted)
#define K0P6_MPS_TS_M2_DONE 6       // max(t)   over CTAs leaving the M2 barrier
#define K0P6_MPS_TS_M6_DONE 7       // max(t)   over CTAs leaving M6
#define K0P6_MPS_TS_COUNT 8

// ---- packed config word (K0P6_D_MPS_CFG) --------------------------------------
// [0:8)   C    reserved minimum-progress service CTAs (finish order)
// [8:16)  g    adjacent N-chunks grouped per push (1, 2, 4, 16)
// [16:24) mode 0 reserve-only | 1 bulk-push-after-M7 | 2 stream
// [24:32) flush_rows  completed-row flags per system release (1..64)
// [32]    pull_fallback: stream flags, but M8 pulls remote part (diagnostic)
// [33]    timestamps_enable
struct config {
    std::uint32_t reserved_comm_ctas;
    std::uint32_t group_slices;
    std::uint32_t mode;
    std::uint32_t flush_rows;
    bool pull_fallback;
    bool timestamps;
};

__host__ __device__ __forceinline__ std::uint64_t encode_config(
        std::uint32_t reserved_comm_ctas, std::uint32_t group_slices,
        std::uint32_t mode, std::uint32_t flush_rows, bool pull_fallback,
        bool timestamps) {
    return (std::uint64_t)reserved_comm_ctas
           | ((std::uint64_t)group_slices << 8)
           | ((std::uint64_t)mode << 16)
           | ((std::uint64_t)flush_rows << 24)
           | ((std::uint64_t)(pull_fallback ? 1u : 0u) << 32)
           | ((std::uint64_t)(timestamps ? 1u : 0u) << 33);
}

__host__ __device__ __forceinline__ config decode_config(std::uint64_t word) {
    config c;
    c.reserved_comm_ctas = (std::uint32_t)(word & 0xFFu);
    c.group_slices = (std::uint32_t)((word >> 8) & 0xFFu);
    c.mode = (std::uint32_t)((word >> 16) & 0xFFu);
    c.flush_rows = (std::uint32_t)((word >> 24) & 0xFFu);
    c.pull_fallback = ((word >> 32) & 1u) != 0u;
    c.timestamps = ((word >> 33) & 1u) != 0u;
    return c;
}

__host__ __device__ __forceinline__ bool config_is_valid(config c) {
    if (c.reserved_comm_ctas > 64u) return false;
    if (c.reserved_comm_ctas >= 256u) return false;
    if (c.group_slices != 1u && c.group_slices != 2u && c.group_slices != 4u &&
        c.group_slices != 16u) return false;
    if (c.mode > 2u) return false;
    if (c.mode == 2u && c.reserved_comm_ctas == 0u) return false;
    // Mode 1 (bulk, no overlap) strides the full grid: the M7 stride derives
    // from C everywhere, so mode 1 must carry C == 0.
    if (c.mode == 1u && c.reserved_comm_ctas != 0u) return false;
    if (c.flush_rows == 0u || c.flush_rows > 64u) return false;
    return true;
}

// ---- tile-event wire format ---------------------------------------------------
// One 32-bit slot per (b, nc). The MARK bit makes zero safely mean "empty";
// b < 2^14 is enforced by the kernel shape/config guard (bcap = PADMAX/32).
inline constexpr std::uint32_t kEventMark = 0x80000000u;

__host__ __device__ __forceinline__ std::uint32_t encode_event(int b, int nc) {
    return kEventMark | ((std::uint32_t)b << 4) | (std::uint32_t)nc;
}

__host__ __device__ __forceinline__ int event_block(std::uint32_t word) {
    return (int)((word >> 4) & 0x3FFFu);
}

__host__ __device__ __forceinline__ int event_chunk(std::uint32_t word) {
    return (int)(word & 0xFu);
}

// ---- device diagnostics -------------------------------------------------------
__device__ __forceinline__ std::uint64_t realtime_now() {
#if defined(__HIP_DEVICE_COMPILE__)
    std::uint64_t t;
    // `s_memrealtime` is an SMEM instruction: its destination SGPR pair is NOT
    // valid until `s_waitcnt lgkmcnt(0)`. Without the wait the caller reads
    // whatever the register happened to hold, so sites that follow LDS traffic
    // (which carries its own lgkmcnt wait) read a real clock while sites that
    // do not read garbage. That is exactly the observed failure: LAST_READY and
    // M7_DONE carried real clocks while DRAIN, REDUCE_DONE, M2_DONE and
    // M6_DONE all read 1. Destination is a scalar pair, so the constraint is
    // "s", not "r".
    asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)"
                 : "=s"(t) :: "memory");
    return t;
#else
    return 0;
#endif
}

// min-timestamp: store max(~t) so the M0 zeroing (all-zero) is a valid floor and
// no special init value is required. Recover with ~cell.
__device__ __forceinline__ void ts_first(std::uint64_t* cell, bool enable) {
    if (!enable || cell == nullptr) return;
    atomicMax(reinterpret_cast<unsigned long long*>(cell),
              ~realtime_now());
}

__device__ __forceinline__ void ts_last(std::uint64_t* cell, bool enable) {
    if (!enable || cell == nullptr) return;
    atomicMax(reinterpret_cast<unsigned long long*>(cell), realtime_now());
}

// ---- producer side: M7 task-done hook body -------------------------------------
// Release-then-publish of this CTA's drained block-chunk completion. Finding
// (exp_56 M1 discipline): the per-thread `s_waitcnt vmcnt(0)` drain + CTA
// barrier runs in the phase-2 body's N2GM_TASK_DONE_DRAIN_HOOK immediately
// before this hook, so tid0's agent release carries the whole CTA's part
// atomics. No system fence and no row loop ever execute on the GEMM CTA.
// Readiness timestamps (first/last-ready) are captured by the SERVICE waves
// consuming these events, keeping the memory-clobbering s_memrealtime asm
// entirely off the MFMA wave (resource gate).
__device__ __forceinline__ void enqueue_tile_release(
        std::uint32_t* __restrict__ q, std::uint32_t* __restrict__ tail,
        int b, int nc) {
    const std::uint32_t ticket =
        kittens::distributed::fetch_add_relaxed<scope::agent>(tail, 1u);
    kittens::distributed::publish_tile_release<scope::agent>(
        q + ticket, encode_event(b, nc));
}

// ---- service side --------------------------------------------------------------
struct service_env {
    const std::uint32_t* __restrict__ q;
    std::uint32_t* __restrict__ nc_arr;      // [T_ext*16]
    std::uint32_t* __restrict__ pushed;      // [T_ext]
    std::uint32_t* __restrict__ claim;       // [T_ext]
    std::uint32_t* __restrict__ row_rem;     // [T_ext] (M2 popcount; self-clean)
    const int* __restrict__ sti;             // sorted token ids
    const unsigned short* __restrict__ part; // [T_ext, 7168] bf16 (local)
    unsigned short* __restrict__ slots;      // [world, MAXTOK, 7168] bf16
    std::uint32_t* __restrict__ row_ready;   // [world*T_loc_max]
    int* pperr;
    std::uint64_t* ts;                       // [8] diag block (or nullptr)
    const symmetric_heap_descriptor* symmetric;
    int cur;
    int world;
    int maxtok;
    int t_ext;
    int t_loc_max;
    std::uint32_t epoch32;
    std::uint32_t group_slices;              // g in {1,2,4,16}
    std::uint32_t flush_rows;                // flags per system release
    std::uint64_t spin_limit;
    std::uint32_t events_total;              // E = (nvi[0] >> 5) * 16
    std::uint32_t queue_capacity;            // (PADMAX/32)*16
    bool ts_enable;
};

// Leader-polled, pperr-watching bounded wait on one queue slot. Returns the
// decoded event word, or 0 on failure (pperr bit already recorded). Word 0 is
// never a real event because of the MARK bit.
__device__ __forceinline__ std::uint32_t wait_event_nonempty(
        const std::uint32_t* slot, int* pperr, std::uint64_t spin_limit) {
    std::uint64_t spins = 0;
    while (true) {
        const std::uint32_t v = load_relaxed<scope::agent>(slot);
        if (v != 0u) return v;
        const int err = load_relaxed<scope::agent>(pperr);
        if ((err & (16777216 | K0P6_MPS_ERR_SERVICE)) != 0) return 0u;
        if (++spins > spin_limit) {
            atomicOr(pperr, K0P6_MPS_ERR_SERVICE);
            return 0u;
        }
        kittens::distributed::detail::pause();
    }
}

struct wave_scratch {
    std::uint32_t rows[32];        // rows with a group push from this event
    std::uint32_t group_base[32];  // first nc of the claimed group
    std::uint32_t push_count;
    std::uint32_t flag_rows[64];   // completed rows pending publication
    std::uint32_t flag_count;
    std::uint32_t event;           // leader-polled event word (0 = failed)
};

static_assert(sizeof(wave_scratch) == 524,
              "wave_scratch must stay inside the reused M1 staging slice");

// exp_07: arrival-counter address, CHUNK-MAJOR.
//
// The natural layout `nc_arr[r*16 + nc]` puts consecutive rows exactly 16
// uint32 = 64 B apart, i.e. **exactly one cache line apart**, so the 32 live
// lanes of one event — which handle 32 different rows of the same block —
// touch 32 DISTINCT cache lines on every arrival. exp_05/exp_06 measured that
// footprint as the larger half of a g-independent +1,159 us M7 interference
// floor (scope accounted for only ~30%).
//
// Chunk-major `nc_arr[nc*T_ext + r]` puts those same 32 rows in 32 CONSECUTIVE
// uint32 = 2 cache lines. Identical semantics, identical buffer and size, no
// ABI change: only the two service call sites and this helper know the layout,
// and M0 still zeroes the whole array.
//
// Predicted trade: the group-completion probe loop walks `nc` for a FIXED row,
// which was one shared cache line under row-major and becomes `g` separate
// lines here. So this should WIN at g=1 (probe loop degenerate, arrivals
// dominate) and LOSE at g=16. That asymmetry is the experiment.
__device__ __forceinline__ std::uint32_t* nc_arr_at(
        const service_env& env, std::uint32_t r, std::uint32_t nc) {
    return env.nc_arr + (std::size_t)nc * (std::size_t)env.t_ext +
           (std::size_t)r;
}

inline constexpr std::size_t kSliceBytes = 448u * 2u;   // 896
// How many claimed row groups the service wave copies concurrently. A single
// group is 896*g bytes = at most 56*g packets, so at g=1 it does not even give
// the wave's 64 lanes one packet each and no pipelining is possible WITHIN a
// group. Batching whole groups is the only way to give a lane more than one
// load in flight. See exp_03.
inline constexpr unsigned int kPushBatch = 4u;

// Source (this rank's `part` row slice) and destination (the owner's landing
// slot, local or peer-translated) of one claimed group. Split out of
// push_slice_group so a wave can compute several groups' addresses before
// issuing any of their loads.
__device__ __forceinline__ const void* slice_group_src(
        const service_env& env, std::uint32_t r, std::uint32_t group_base) {
    return reinterpret_cast<const std::byte*>(env.part) +
        ((std::size_t)r * 7168u + (std::size_t)group_base * 448u) * 2u;
}

__device__ __forceinline__ void* slice_group_dst(
        const service_env& env, std::uint32_t r, std::uint32_t group_base) {
    const std::uint32_t owner = r / (std::uint32_t)env.maxtok;
    const std::uint32_t pos = r % (std::uint32_t)env.maxtok;
    std::byte* dst_local = reinterpret_cast<std::byte*>(
        env.slots + ((std::size_t)env.cur * (std::size_t)env.maxtok + pos) *
                        7168u + (std::size_t)group_base * 448u);
    return (owner == (std::uint32_t)env.cur)
        ? dst_local
        : hk_moe::peer_ptr(dst_local, (int)owner, env.symmetric);
}

// Push one claimed group of adjacent N-chunk slices for row r to its owner's
// landing slot. Controlled 16-byte stores, coalesced by the wave's 64 lanes;
// no fence, no completion work here (flush_pending owns both).
__device__ __forceinline__ void push_slice_group(const service_env& env,
        std::uint32_t r, std::uint32_t group_base, int lane) {
    kittens::distributed::store_peer_packets(
        slice_group_dst(env, r, group_base),
        slice_group_src(env, r, group_base),
        kSliceBytes * env.group_slices, (unsigned int)lane, 64u);
}

// Push the kPushBatch claimed groups starting at `first` together: every
// group's load is issued before any group's store, so each lane holds
// kPushBatch loads in flight instead of one. Byte-for-byte the same traffic as
// kPushBatch calls to push_slice_group, in an unspecified order between groups
// (they are disjoint rows). The row/base reads come straight from LDS and the
// pointer arrays are indexed only under `#pragma unroll`, so nothing here is
// dynamically indexed in registers.
__device__ __forceinline__ void push_slice_group_batch(const service_env& env,
        const wave_scratch& s, std::uint32_t first, int lane) {
    void* dsts[kPushBatch];
    const void* srcs[kPushBatch];
#pragma unroll
    for (unsigned int b = 0; b < kPushBatch; ++b) {
        const std::uint32_t r = s.rows[first + b];
        const std::uint32_t gb = s.group_base[first + b];
        dsts[b] = slice_group_dst(env, r, gb);
        srcs[b] = slice_group_src(env, r, gb);
    }
    kittens::distributed::store_peer_packets_multi<kPushBatch>(
        dsts, srcs, kSliceBytes * env.group_slices, (unsigned int)lane, 64u);
}

// Publish every completed-row flag accumulated by this wave behind ONE system
// release (the same-thread fence-then-store discipline of the parity M7.5
// publisher, at wave granularity). The row's unique publisher also restores
// the exp_62 row_remaining self-clean invariant. Whole-wave convergent call.
__device__ __forceinline__ void flush_pending(const service_env& env,
        wave_scratch& s, int lane) {
    if (s.flag_count == 0u) return;
    // Wave-granular producer drain: vmcnt counters are per-wavefront, so each
    // lane's s_waitcnt covers the whole wave's outstanding packet stores.
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
    __syncwarp();
    if (lane == 0) {
        hk_moe::release_signal_batch_system();
        for (std::uint32_t i = 0; i < s.flag_count; ++i) {
            const std::uint32_t r = s.flag_rows[i];
            const std::uint32_t owner = r / (std::uint32_t)env.maxtok;
            std::uint32_t* ready_slot =
                env.row_ready + (std::size_t)env.cur *
                                    (std::size_t)env.t_loc_max + r;
            if (owner == (std::uint32_t)env.cur) {
                hk_moe::publish_epoch<scope::agent>(ready_slot, env.epoch32);
            } else {
                hk_moe::publish_epoch<scope::system>(
                    hk_moe::peer_ptr(ready_slot, (int)owner, env.symmetric),
                    env.epoch32);
            }
            // exp_62 self-clean invariant: every published row ends the epoch
            // at row_remaining == 0. Unique writer: this wave alone flagged r.
            hk_moe::publish_relaxed<scope::agent>(env.row_rem + r, 0u);
        }
        s.flag_count = 0u;
    }
    __syncwarp();
}

// Account one pushed group against its row, and when all 16 of the row's
// slices have been pushed, queue the row's completed-row flag.
//
// The flush test follows EVERY append (not once per event): an event may append
// up to 32 rows, so a per-event test could let flag_count overshoot
// flag_rows[64] whenever flush_rows > 32. flag_count stays <= flush_rows <= 64
// under this discipline. Whole-wave convergent call.
__device__ __forceinline__ void retire_pushed_row(const service_env& env,
        wave_scratch& s, std::uint32_t rr, std::uint32_t g, int lane) {
    if (lane == 0) {
        const std::uint32_t oldp =
            kittens::distributed::fetch_add_relaxed<scope::agent>(
                env.pushed + rr, g);
        if (oldp + g == 16u) {
            s.flag_rows[s.flag_count] = rr;
            ++s.flag_count;
        }
    }
    __syncwarp();
    if (s.flag_count >= env.flush_rows) flush_pending(env, s, lane);
}

// The minimum-progress service loop: wave-striped consumption of the tile
// event queue on reserved CTAs. Failure exits the loop; the bounded waits
// elsewhere (owner polls, M9 arrival) make the overall failure terminal.
// `s` is this wave's private scratch slice (LDS), carved by the caller.
__device__ __forceinline__ void run_service(
        const service_env& env, wave_scratch& s,
        std::uint32_t service_id, std::uint32_t service_count, int wid,
        int lane) {
    constexpr int kWaves = 4;
    const std::uint32_t waves_total = service_count * (std::uint32_t)kWaves;
    const std::uint32_t wave_global =
        service_id * (std::uint32_t)kWaves + (std::uint32_t)wid;
    const std::uint32_t g = env.group_slices;
    if (lane == 0) {
        s.push_count = 0u;
        s.flag_count = 0u;
    }
    __syncwarp();
    for (std::uint32_t k = wave_global; k < env.events_total;
         k += waves_total) {
        if (lane == 0) {
            s.event = wait_event_nonempty(env.q + k, env.pperr, env.spin_limit);
        }
        __syncwarp();
        const std::uint32_t ev = s.event;
        if (ev == 0u) break;
        // Wave-local payload acquire (a CTA-scope cta_acquire would deadlock:
        // the waves of one CTA iterate different stripe lengths).
        kittens::distributed::thread_acquire<scope::agent>();
        // Readiness timestamps live on the SERVICE wave (off the MFMA wave):
        // first/last tile-event observed. Service waves already poll at the
        // queue tail, so this is the earliest observable readiness point
        // without instrumenting producer CTAs.
        if (lane == 0 && env.ts != nullptr) {
            ts_first(env.ts + K0P6_MPS_TS_FIRST_READY, env.ts_enable);
            ts_last(env.ts + K0P6_MPS_TS_LAST_READY, env.ts_enable);
        }
        const int b = event_block(ev);
        const int nc = event_chunk(ev);
        const int group_base = nc & ~(int)(g - 1u);
        // Lanes 0..31 account for the 32 sorted rows of block b.
        int r = 0x7FFFFFFF;
        bool live = false;
        std::uint32_t target = 0u;
        if (lane < 32) {
            r = env.sti[(std::size_t)b * 32 + (std::size_t)lane] & 0x00FFFFFF;
            live = r < env.t_ext;
            if (live) {
                target = kittens::distributed::load_relaxed<scope::agent>(
                    env.row_rem + r);
            }
        }
        bool push_lead = false;
        if (live) {
            // exp_06 measured what this site's SCOPE costs: relaxing it to
            // fetch_add_relaxed recovered 354 us of the 1,159 us M7 interference
            // floor at g=1 (157 us at g=16), i.e. ~30%. The other ~70% is
            // FOOTPRINT, which exp_07 attacks by transposing the counter array.
            const std::uint32_t old =
                kittens::distributed::detail::fetch_add_acq_rel<scope::agent>(
                    nc_arr_at(env, r, (std::uint32_t)nc), 1u);
            if (old + 1u == target) {
                // Slice (r,nc) final. Group-check with acq_rel RMW probes so
                // (a) the check itself is in the per-counter RMW order: the
                // *last* completing bumper of the group provably observes
                // every other chunk at target; and (b) the pushing wave holds
                // an acquire edge per chunk of the group before it reads that
                // chunk's part payload — including chunks whose events were
                // consumed by OTHER service waves.
                bool all = true;
                for (std::uint32_t n2 = (std::uint32_t)group_base;
                     n2 < (std::uint32_t)group_base + g; ++n2) {
                    const std::uint32_t cur_count =
                        kittens::distributed::detail::fetch_add_acq_rel<scope::agent>(
                            nc_arr_at(env, r, n2), 0u);
                    if (cur_count != target) { all = false; break; }
                }
                if (all) {
                    const std::uint32_t claim_bit =
                        1u << ((std::uint32_t)group_base / g);
                    // Plain atomicOr dedupes the claimant (same idiom as the
                    // kernel's pperr stores): agent-domain RMW, no ordering
                    // role — the acquire edge is carried by the probes above.
                    const std::uint32_t old_claim =
                        atomicOr(env.claim + r, claim_bit);
                    push_lead = (old_claim & claim_bit) == 0u;
                }
            }
        }
        const unsigned long long ballot = __ballot(push_lead);
        if (lane == 0) s.push_count = 0u;
        __syncwarp();
        unsigned long long pending = ballot;
        while (pending != 0u) {
            const int src = __ffsll((long long)pending) - 1;
            pending &= pending - 1u;
            if (lane == src) {
                const std::uint32_t idx = atomicAdd(&s.push_count, 1u);
                s.rows[idx] = (std::uint32_t)r;
                s.group_base[idx] = (std::uint32_t)group_base;
            }
        }
        __syncwarp();
        const std::uint32_t npush = s.push_count;
        // Copy kPushBatch groups per pass so each lane keeps kPushBatch loads
        // in flight; the per-row accounting stays strictly one row at a time so
        // the flag/flush discipline above is unchanged.
        std::uint32_t i = 0;
        for (; i + kPushBatch <= npush; i += kPushBatch) {
            push_slice_group_batch(env, s, i, lane);
#pragma unroll
            for (unsigned int b = 0; b < kPushBatch; ++b) {
                retire_pushed_row(env, s, s.rows[i + b], g, lane);
            }
        }
        for (; i < npush; ++i) {
            push_slice_group(env, s.rows[i], s.group_base[i], lane);
            retire_pushed_row(env, s, s.rows[i], g, lane);
        }
    }
    flush_pending(env, s, lane);
    if (lane == 0 && env.ts != nullptr) {
        ts_last(env.ts + K0P6_MPS_TS_DRAIN, env.ts_enable);
    }
}

} // namespace hk_moe::mps

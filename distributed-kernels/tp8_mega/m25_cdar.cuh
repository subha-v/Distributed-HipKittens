/**
 * @file
 * @brief CDAR — the counter-dataflow all-reduce for M25 (TP8+EP boundaries).
 *
 * Composition over the distributed primitive layer; no new synchronization
 * mechanism is introduced here. A CDAR instance replaces one per-layer
 * all-reduce with two overlapped halves (M25_TP8_MEGA_DESIGN.md section 2):
 *
 *   ERS  (producer half): every rank's producing GEMM epilogue commits its
 *        partial rows for token-slab s directly into slab s's OWNER-rank
 *        accumulator, depth-bounded (credit.cuh), traversing slabs in
 *        owner_major_staggered order (order.cuh) so the emergent schedule is
 *        ring reduce-scatter with every link busy from the first slab.
 *   MAG  (consumer half): once the owner has all eight contributions it
 *        certifies the slab (slab.cuh); consumer ranks PULL the reduced slab
 *        into local staging with a quota-bounded CONSUMING pool — the M15
 *        front-half-combine economics (-93/-444 us), never a carrier pool
 *        (+340 us).
 *
 * FAN-IN IS TWO-LEVEL, deliberately:
 *   level 1 (local, agent scope): several producer CTAs on one rank
 *        contribute fragments to the same slab; they count in with
 *        counted_arrive_dynamic_release_into (runtime fragment count — MoE
 *        fan-in varies per slab). The LAST local arriver owns the rank-level
 *        publication.
 *   level 2 (cross-rank, system scope): the publication is ONE per-source
 *        epoch word stored to the owner's src_done[slab][source] cell —
 *        the proven M15 rows_done pattern. No cross-rank RMW counters:
 *        per-source words keep remote traffic store-class, keep contention
 *        at zero, and give the owner a self-describing view (which source is
 *        late) for free. The owner certifies on observing all World words.
 *
 * EMPTY-CONTRIBUTION RULE (the M24 lesson, load-bearing): a rank that
 * contributes NOTHING to slab s this epoch must still publish its src_done
 * word (ers_publish_empty) — the arming agent does it at plan time. A
 * certificate that some peer counts on must exist on every path.
 *
 * TRANSPORT is a compile-time knob (gfx950 arch card: coalesced remote
 * atomics 52.8 GB/s vs stores 54.9; scattered atomics 4.1 — the atomic path
 * is legal only in coalesced row-major form; gfx942 must use towers):
 *   atomic_accumulate — throttled packed-bf16 remote RMW straight into the
 *        owner accumulator. One buffer, no owner reduce step.
 *   store_towers     — throttled packet16 stores into a per-source tower on
 *        the owner; the owner reduces World towers locally after gather
 *        (the m15b fold-then-wide-push twin; also the lost-update detector
 *        shape: towers can be compared row-by-row in a debug arm).
 *
 * MEMORY (per boundary, per rank, caller-allocated on the symmetric heap or
 * any peer-visible allocation; layout helpers below):
 *   accum   : num_slabs_local x slab_bytes           (owned slabs only)
 *   towers  : World x num_slabs_local x slab_bytes   (store_towers only)
 *   signals : src_done[num_slabs][World] u64, mag_ready[num_slabs] u64,
 *             local_arrive[num_slabs] u32 (level-1 cells, zeroed at arm)
 * Reuse across epochs goes through lifetime.cuh slot retirement; every
 * signal cell is epoch-word addressed so stale observations are impossible
 * by construction (exact-epoch match, completion.cuh).
 */

#pragma once

#include <cstdint>

#include "../../include/cdna4/ops/group/distributed/packet.cuh"
#include "../../include/cdna4/ops/group/distributed/credit.cuh"
#include "../../include/cdna4/ops/group/distributed/order.cuh"
#include "../../include/cdna4/ops/group/distributed/sync.cuh"
#include "../../include/cdna4/ops/group/distributed/completion.cuh"
#include "../../include/cdna4/ops/group/distributed/slab.cuh"
#include "../../include/cdna4/ops/group/distributed/counter.cuh"

namespace kittens::distributed::cdar {

enum class transport : unsigned int {
    atomic_accumulate = 0,
    store_towers = 1,
};

/** Geometry of one CDAR boundary. World is compile-time (fabric constant). */
template<unsigned int World>
struct geometry {
    unsigned int tokens;      ///< global rows this step (runtime, from the fill/step descriptor)
    unsigned int row_bytes;   ///< bf16 hidden row: 7168 * 2 = 14336
    unsigned int slab_rows;   ///< rows per slab (prefill default 512; decode mode shrinks it)
    unsigned int rank;        ///< this rank

    KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE unsigned int num_slabs() const {
        return (tokens + slab_rows - 1u) / slab_rows;
    }
    KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE unsigned int owner_of(
            unsigned int slab) const {
        return slab % World;
    }
    KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE unsigned int rows_in(
            unsigned int slab) const {
        const unsigned int first = slab * slab_rows;
        const unsigned int left = tokens - first;
        return left < slab_rows ? left : slab_rows;
    }
    KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE std::size_t slab_bytes() const {
        return static_cast<std::size_t>(slab_rows) * row_bytes;
    }
};

/**
 * Per-rank signal block for one boundary. All three arrays live at the SAME
 * offsets on every rank (symmetric rule), so peer cells are addressed by
 * base swap alone — the pgl/translate_peer discipline.
 */
struct signals {
    std::uint64_t* src_done;   ///< [num_slabs][World], written by remote sources
    std::uint64_t* mag_ready;  ///< [num_slabs], written by the owner's multicast
    std::uint32_t* local_arrive; ///< [num_slabs], level-1 cells, zero at arm

    template<unsigned int World>
    KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE std::uint64_t* src_done_cell(
            unsigned int slab, unsigned int source) const {
        return src_done + static_cast<std::size_t>(slab) * World + source;
    }
    KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE std::uint64_t* mag_ready_cell(
            unsigned int slab) const {
        return mag_ready + slab;
    }
    KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE std::uint32_t* local_cell(
            unsigned int slab) const {
        return local_arrive + slab;
    }
};

/**
 * One boundary's peer view: base pointers for every rank's accumulator,
 * towers (may be null under atomic_accumulate) and signal block. Indexed by
 * rank; the local rank's entries point at local memory (uniform expressions,
 * no self-select — the shipped epilogue's addressing discipline).
 */
template<unsigned int World>
struct peer_view {
    std::byte* accum[World];
    std::byte* towers[World];
    signals sig[World];
};

// ---------------------------------------------------------------------------
// ERS: producer half
// ---------------------------------------------------------------------------

/**
 * Commit one contiguous row-range fragment of slab `slab` to its owner.
 *
 * `fragment` is the caller's staged partial (LDS or registers spilled to a
 * local staging row — the GEMM epilogue already owns such staging), laid out
 * as `rows x row_bytes` contiguous. Lanes stride packed-dword-wise across
 * the fragment so the remote stream is fully coalesced (the 52.8 GB/s form).
 *
 * Contract: epilogue position — no interleaved vector loads between commits
 * in the same span, or Depth is renegotiated (credit.cuh). This function
 * performs NO synchronization: arrival accounting is ers_local_arrive().
 */
template<transport T, unsigned int Depth, unsigned int World>
KITTENS_DISTRIBUTED_DEVICE_INLINE void ers_commit_fragment(
        const geometry<World>& g, const peer_view<World>& pv,
        unsigned int slab, unsigned int first_row_in_slab,
        unsigned int rows, const std::byte* __restrict__ fragment,
        unsigned int tid, unsigned int threads) {
    const unsigned int owner = g.owner_of(slab);
    // Owned slabs are stored at their LOCAL slab ordinal on the owner:
    // local_ordinal = slab / World (owner stripe).
    const std::size_t local_ordinal = slab / World;
    const std::size_t byte_off =
        local_ordinal * g.slab_bytes() +
        static_cast<std::size_t>(first_row_in_slab) * g.row_bytes;
    const std::size_t bytes =
        static_cast<std::size_t>(rows) * g.row_bytes;

    if constexpr (T == transport::atomic_accumulate) {
        std::byte* const dst = pv.accum[owner] + byte_off;
        const auto* const src =
            reinterpret_cast<const unsigned int*>(fragment);
        auto* const out = reinterpret_cast<unsigned int*>(dst);
        const std::size_t dwords = bytes >> 2;
#pragma unroll 1
        for (std::size_t i = tid; i < dwords; i += threads) {
            throttled_accumulate_bf162<Depth>(out + i, src[i]);
        }
    } else {
        // Tower path: this source's private stripe on the owner; plain
        // stores, owner reduces after gather.
        std::byte* const dst = pv.towers[owner] +
            static_cast<std::size_t>(g.rank) *
                ((g.num_slabs() + World - 1u) / World) * g.slab_bytes() +
            byte_off;
        const auto* const in = reinterpret_cast<const packet16*>(fragment);
        auto* const out = reinterpret_cast<packet16*>(dst);
        const std::size_t packets = bytes >> 4;
#pragma unroll 1
        for (std::size_t i = tid; i < packets; i += threads) {
            throttled_store_packet16<Depth>(out + i, in[i]);
        }
    }
}

/**
 * Level-1 arrival for one producer CTA's slab contribution, and — on the
 * last local arriver — the level-2 per-source publication to the owner.
 *
 * Call once per contributing CTA per slab, one elected thread, AFTER that
 * CTA's commits for the slab have drained (throttle_vmcnt<0>() or the
 * caller's producer_drain_release covering the epilogue span; the release
 * to SYSTEM scope is performed here by the last arriver).
 *
 * `expected_local` is this rank's fragment count for this slab this epoch
 * (runtime — the rolling planner computes it when it schedules the slab).
 * `payload` rides in the certificate's low half (e.g. rows contributed).
 */
template<unsigned int World>
KITTENS_DISTRIBUTED_DEVICE_INLINE void ers_local_arrive(
        const geometry<World>& g, const peer_view<World>& pv,
        unsigned int slab, unsigned int expected_local,
        std::uint32_t epoch, std::uint32_t payload) {
    counted_arrival a;
    counted_arrive_dynamic_release_into<memory_scope::system>(
        pv.sig[g.rank].local_cell(slab), expected_local, a);
    if (a.last) {
        const unsigned int owner = g.owner_of(slab);
        certify_slab_to(
            pv.sig[owner].template src_done_cell<World>(slab, g.rank),
            epoch, payload);
    }
}

/**
 * Empty-contribution publication (the arming agent's duty when
 * expected_local == 0 for (rank, slab) this epoch). One elected thread.
 * The payload 0 is meaningful: the owner may sum payloads as a cross-check
 * against the step's n_orig accounting.
 */
template<unsigned int World>
KITTENS_DISTRIBUTED_DEVICE_INLINE void ers_publish_empty(
        const geometry<World>& g, const peer_view<World>& pv,
        unsigned int slab, std::uint32_t epoch) {
    const unsigned int owner = g.owner_of(slab);
    // An empty contribution has no payload to release; the relaxed word is
    // self-contained (exact-epoch matched by the observer).
    certify_slab_to(
        pv.sig[owner].template src_done_cell<World>(slab, g.rank), epoch, 0u);
}

// ---------------------------------------------------------------------------
// Owner side: gather-complete observation, optional tower reduce, certify
// ---------------------------------------------------------------------------

struct owner_gather {
    std::uint32_t payload_sum;
    bool ready;
};

/**
 * Bounded observation that all World sources have published slab `slab` at
 * `epoch`, with a single acquire on success covering every source's payload
 * (each source's release happened before its word; exact-epoch matching
 * makes stale words invisible). One elected thread polls; a whole-CTA owner
 * role follows the completion.cuh wider-role discipline instead.
 *
 * Poll order is source-rotated by rank so eight owners do not all hammer
 * source 0's cell first (m17's spreading-by-order, applied to signals).
 */
template<unsigned int World>
KITTENS_DISTRIBUTED_DEVICE_INLINE void owner_gather_ready_into(
        const geometry<World>& g, const peer_view<World>& pv,
        unsigned int slab, std::uint32_t epoch,
        std::uint64_t spin_limit, owner_gather& out) {
    out.payload_sum = 0u;
    out.ready = false;
    wait_result64 w;
#pragma unroll 1
    for (unsigned int i = 0u; i < World; ++i) {
        const unsigned int source = (g.rank + 1u + i) % World;
        init_wait_result(w);
        bounded_poll_epoch_word_relaxed_into<memory_scope::system>(
            pv.sig[g.rank].template src_done_cell<World>(slab, source),
            epoch, spin_limit, w);
        if (!w.ready) return;
        out.payload_sum += epoch_word_payload(w.observed);
    }
    thread_acquire<memory_scope::system>();
    out.ready = true;
}

/**
 * Owner's certificate multicast after its post-gather work (tower reduce
 * and/or fused residual+RMSNorm on the owned slab) has drained. The caller
 * performs producer_drain_release<system>() covering that work first; this
 * is the one-textual-copy multicast (slab.cuh).
 */
template<unsigned int World>
KITTENS_DISTRIBUTED_DEVICE_INLINE void owner_certify(
        const geometry<World>& g, const peer_view<World>& pv,
        unsigned int slab, std::uint32_t epoch, std::uint32_t payload) {
    std::uint64_t* cells[World];
#pragma unroll
    for (unsigned int r = 0u; r < World; ++r) {
        cells[r] = pv.sig[r].mag_ready_cell(slab);
    }
    certify_slab<World>(cells, epoch, payload);
}

// ---------------------------------------------------------------------------
// MAG: consumer half
// ---------------------------------------------------------------------------

/**
 * Bounded wait for slab `slab`'s certificate on the LOCAL mag_ready cell,
 * calling-thread acquire on success. Pool CTAs consuming whole slabs use
 * the relaxed poll + converge + cta_acquire discipline instead.
 */
template<unsigned int World>
KITTENS_DISTRIBUTED_DEVICE_INLINE void mag_wait_into(
        const geometry<World>& g, const peer_view<World>& pv,
        unsigned int slab, std::uint32_t epoch,
        std::uint64_t spin_limit, wait_result64& result) {
    bounded_wait_slab_into(pv.sig[g.rank].mag_ready_cell(slab), epoch,
                           spin_limit, result);
}

/**
 * Pull one certified slab from its owner into local staging (CTA-cooperative
 * copy; consuming-pool placement is the caller's role decision). The owner's
 * accumulator is CONSUMED here, never written: the pool operates strictly in
 * the producer's shadow on certified work.
 */
template<unsigned int World>
KITTENS_DISTRIBUTED_DEVICE_INLINE void mag_pull(
        const geometry<World>& g, const peer_view<World>& pv,
        unsigned int slab, std::byte* __restrict__ local_dst,
        unsigned int tid, unsigned int threads) {
    const unsigned int owner = g.owner_of(slab);
    const std::size_t local_ordinal = slab / World;
    const std::byte* const src =
        pv.accum[owner] + local_ordinal * g.slab_bytes();
    const std::size_t bytes =
        static_cast<std::size_t>(g.rows_in(slab)) * g.row_bytes;
    load_peer_packets(local_dst, src, bytes, tid, threads);
}

// ---------------------------------------------------------------------------
// Host-side layout arithmetic (sizes for the boundary allocations)
// ---------------------------------------------------------------------------

template<unsigned int World>
KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE std::size_t accum_bytes(
        const geometry<World>& g) {
    const unsigned int owned = (g.num_slabs() + World - 1u) / World;
    return static_cast<std::size_t>(owned) * g.slab_bytes();
}

template<unsigned int World>
KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE std::size_t towers_bytes(
        const geometry<World>& g, transport t) {
    return t == transport::store_towers ? accum_bytes(g) * World : 0u;
}

template<unsigned int World>
KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE std::size_t signal_words64(
        const geometry<World>& g) {
    // src_done + mag_ready
    return static_cast<std::size_t>(g.num_slabs()) * (World + 1u);
}

// ---------------------------------------------------------------------------
// E-B1 / E-B2 — device phase ledger  (compile-time gated: M25_LEDGER, def. 0)
// ---------------------------------------------------------------------------
//
// A per-CTA RAW EVENT RING, not a set of running maxima. Every record is one
// closed interval {t0, t1} in the device's own 100 MHz realtime domain, tagged
// with a phase code and an aux payload (rows or bytes). All reduction — phase
// spans, unions, overlap, the E-B2 duty histogram, and the cross-rank rank-max
// — happens on the HOST after the launch joins. That choice is what discharges
// four of LAW-63's five traps by construction rather than by discipline.
//
// LAW-63 acceptance criteria, and how each is met:
//   (a) `s_waitcnt lgkmcnt(0)` after every `s_memrealtime`.  realtime_now()
//       below is the single call site; the wait is inside the same asm block,
//       so no caller can forget it (LAW-63a killed four of six prior sites).
//   (b) the clock is the CONSTANT-RATE 100 MHz realtime counter, NOT the
//       ~2.2 GHz shader clock: 1 tick = 10 ns = 0.01 us exactly. Mixing the
//       two is a 22x error. The host prints hipDeviceAttributeWallClockRate
//       next to kLedgerHz every run so the conversion carries a receipt.
//   (c) running maxima are RESET between epochs — here there are no running
//       maxima at all: the host zeroes `count[]` before every iteration, so a
//       ring only ever describes the iteration it was armed for. A soak can
//       not leak into a timed iteration.
//   (d) spans are reduced ACROSS RANKS to rank-max on the host (rank-0-only
//       reporting understates a combine-class phase by 38%). The bench prints
//       every rank AND an explicit RANKMAX row; there is no rank-0 fast path.
//   (e) timestamps from different kernel launches are NEVER subtracted: the
//       ring is indexed by launch slot, every launch gets its own time origin
//       (min over that launch's kernel-entry stamps on that rank), and the
//       host analysis refuses to combine slots.
//
// Cost: one `s_memrealtime` pair per phase per CTA under `threadIdx.x == 0`,
// with every bracket OUTSIDE the poll bodies and outside the store loops —
// the t2v6 PROF-arm shape (`k0pf6gm_device_tile_t2v6.hip`, desc slot 73),
// measured cost-free on the training chassis. Wall parity against the
// M25_LEDGER=0 build is a run-time acceptance criterion, not an assumption.
#ifndef M25_LEDGER
#define M25_LEDGER 0
#endif

#if M25_LEDGER

/// s_memrealtime ticks per second on CDNA (constant-rate wall counter).
/// 1 tick = 10 ns. NEVER mix with s_getreg/clock() shader ticks (LAW-63b).
inline constexpr std::uint64_t kLedgerHz = 100000000ull;
inline constexpr double kLedgerTickUs = 0.01;

/// Phase codes. `lg_kernel` brackets the whole launch on this CTA and is the
/// only event guaranteed present; the loop-span codes bracket a whole loop,
/// the rest bracket one iteration's phase.
enum ledger_code : std::uint32_t {
    lg_kernel      = 1u,   ///< CTA entry stamp (t0 == t1); origin of the slot
    lg_prod_loop   = 2u,   ///< whole producer loop (aux = producer_block)
    lg_mfma1       = 3u,   ///< one item's pre-boundary MFMA burst (aux = rows)
    lg_commit      = 4u,   ///< ERS remote store/atomic issue (aux = bytes)
    lg_drain       = 5u,   ///< vmcnt drain + local arrive (aux = slab)
    lg_duty_loop   = 6u,   ///< whole owner-duty loop
    lg_gather_wait = 7u,   ///< owner gather poll (bracketed OUTSIDE the poll)
    lg_reduce      = 8u,   ///< owner tower reduce (aux = slab)
    lg_certify     = 9u,   ///< counted arrival + certificate multicast
    lg_cons_loop   = 10u,  ///< whole consume loop
    lg_mag_wait    = 11u,  ///< consumer certificate wait (OUTSIDE the poll)
    lg_pull        = 12u,  ///< mag_pull remote read (aux = bytes)
    lg_verify      = 13u,  ///< verification sample
    lg_mfma2       = 14u,  ///< post-boundary MFMA burst (aux = rows)
    lg_tx_loop     = 15u,  ///< whole exposed-transport loop (phased arm)
    lg_kernel_end  = 16u,  ///< CTA exit stamp (t0 == t1)
    lg_code_count  = 17u
};

/// One closed interval. 24 B, 8 B aligned; written by one thread, read only
/// by the host after the launch joins (the join is the release edge).
struct ledger_event {
    std::uint64_t t0;
    std::uint64_t t1;
    std::uint32_t code;
    std::uint32_t aux;
};

/// Device-side view of one LAUNCH SLOT's ring: `cap` events per CTA.
struct ledger_view {
    ledger_event* ev;
    std::uint32_t* count;   ///< [blocks]; low 16 = written, high 16 = dropped
    unsigned int cap;
};

/// Per-CTA cursor, register-resident, owned by threadIdx.x == 0.
struct ledger_cursor {
    ledger_event* ring;
    unsigned int cap;
    unsigned int n;
    unsigned int dropped;
};

/// The 100 MHz realtime read. `s_memrealtime` is an SMEM instruction: its
/// destination SGPR pair is INVALID until `s_waitcnt lgkmcnt(0)`, so the wait
/// is welded into the same asm block (LAW-63a). Destination is a scalar pair
/// => the constraint is "s", not "r".
KITTENS_DISTRIBUTED_DEVICE_INLINE std::uint64_t realtime_now() {
#if defined(__HIP_DEVICE_COMPILE__)
    std::uint64_t t;
    asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)"
                 : "=s"(t) :: "memory");
    return t;
#else
    return 0ull;
#endif
}

KITTENS_DISTRIBUTED_DEVICE_INLINE ledger_cursor ledger_open(
        const ledger_view& v, unsigned int block) {
    ledger_cursor c;
    c.ring = v.ev + static_cast<std::size_t>(block) * v.cap;
    c.cap = v.cap;
    c.n = 0u;
    c.dropped = 0u;
    return c;
}

/// Plain 24 B store: not atomic, not volatile, no fence — (CTA, slot) has
/// exactly one writer and the only reader is the host after the join.
///
/// `store` predicates only the WRITE; the cursor arithmetic runs on every
/// thread so `n`/`dropped` stay CTA-uniform and the whole cursor lives in
/// SGPRs instead of VGPRs. That is what keeps the instrument off the vector
/// register budget of an occupancy-1 kernel (LAW-32: a megakernel can be made
/// slow purely by the register allocator).
KITTENS_DISTRIBUTED_DEVICE_INLINE void ledger_mark(
        ledger_cursor& c, std::uint32_t code, std::uint64_t t0,
        std::uint64_t t1, std::uint32_t aux, bool store) {
    if (c.n < c.cap) {
        if (store) {
            ledger_event e;
            e.t0 = t0;
            e.t1 = t1;
            e.code = code;
            e.aux = aux;
            c.ring[c.n] = e;
        }
        ++c.n;
    } else {
        ++c.dropped;
    }
}

KITTENS_DISTRIBUTED_DEVICE_INLINE void ledger_close(
        const ledger_view& v, unsigned int block, const ledger_cursor& c) {
    v.count[block] = (c.n & 0xffffu) | (c.dropped << 16);
}

#endif // M25_LEDGER

} // namespace kittens::distributed::cdar

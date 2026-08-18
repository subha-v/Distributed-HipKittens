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

} // namespace kittens::distributed::cdar

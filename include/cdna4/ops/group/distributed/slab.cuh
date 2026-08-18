/**
 * @file
 * @brief Prefix certification over coarse regions (knob K3 as a primitive).
 *
 * Extracted from the M15 megakernel's open-coded two-slab protocol
 * (k0pf6gm_device_tile_m15.hip, drain -> barrier -> per-peer epoch-word
 * publication -> bounded wait -> acquire), which replaced ~926,000 row-level
 * protocol operations with 2 x 8 slab certificates: the mode-14 economics
 * without mode-14's loss (docs/distributed/OVERLAP_ABSTRACTIONS.md sections
 * 3-4). Granularity is the free axis (a 64x signal-count change measured
 * ~nothing); EXPOSURE is the deadly one — these helpers publish and observe,
 * but the caller decides where certification sits so it stays folded into a
 * producer tail or consumer head, never a standalone phase.
 *
 * Division of labor, deliberately narrow:
 *  - Grid-wide synchronization stays in the KERNEL (the one-grid-barrier
 *    budget is the kernel's to spend; a primitive must not hide one).
 *  - Payload visibility is the caller's release/acquire chain (sync.cuh),
 *    exactly as packet.cuh requires.
 *  - This header owns only the certificate cells' protocol: one
 *    self-describing epoch word per (region, peer), multicast by ONE textual
 *    copy of the publication loop — exp_34/38 measured per-peer code
 *    duplication costing a register cliff; the one-copy discipline is
 *    load-bearing and is the reason this is a primitive at all.
 *
 * The slab cut must sit where producer tile shape and consumer vector shape
 * co-align (the M15 cut: column 3,584 for 448-col nc chunks x 1,024-byte
 * combine chunks; for reduce-scatter schedules the owner block boundary is
 * the natural cut and needs no re-chunking). The cut is the caller's
 * geometry; certificates are agnostic to it.
 */

#pragma once

#include <cstdint>

#include "detail/config.cuh"
#include "sync.cuh"
#include "completion.cuh"

namespace kittens::distributed {

/**
 * Publish one region's certificate to every peer: a single epoch word
 * (epoch, payload) stored relaxed to each peer's signal cell.
 *
 * PRECONDITIONS (protocol errors if violated, not checked):
 *  - The caller has already made the region's payload visible at
 *    `Scope` (producer_drain_release<Scope>() or an equivalent chain
 *    covering every contributing wave — for multi-CTA producers that means
 *    the caller's counted-arrival or grid-step established completion).
 *  - Exactly one thread calls this per (region, epoch): the elected
 *    certifier (the M15 leader, or a counted_arrive_*'s last arriver, or an
 *    ERS owner after dynamic fan-in completes).
 *
 * `peer_signals[r]` is the certificate cell for this region as visible to
 * rank r (self included; pass the local cell at the local index — the
 * uniform loop covers own-rank with no select, the same discipline as the
 * peer-table addressing in the shipped epilogue). One textual copy: callers
 * must NOT unroll this per peer by hand.
 */
template<unsigned int World, memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void certify_slab(
        std::uint64_t* const (&peer_signals)[World],
        std::uint32_t epoch, std::uint32_t payload) {
#pragma unroll 1
    for (unsigned int rank = 0u; rank < World; ++rank) {
        publish_epoch_word_relaxed<Scope>(peer_signals[rank], epoch, payload);
    }
}

/** Single-destination form: certify to one consumer's cell (ERS owner-side). */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void certify_slab_to(
        std::uint64_t* signal, std::uint32_t epoch, std::uint32_t payload) {
    publish_epoch_word_relaxed<Scope>(signal, epoch, payload);
}

/**
 * Bounded wait for one region's certificate, calling-thread acquire on
 * success. `result.observed`'s low half carries the producer's payload
 * (row count, live-tile count — the certificate is self-describing so the
 * consumer never re-derives extent from memory it has not acquired).
 *
 * Wider consumer roles (a whole pool CTA consuming one slab) follow the
 * completion.cuh discipline instead: selected lanes call the relaxed poll,
 * converge, then cta_acquire<Scope>() once — do not give every lane its own
 * acquire.
 */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void bounded_wait_slab_into(
        const std::uint64_t* signal, std::uint32_t expected_epoch,
        std::uint64_t spin_limit, wait_result64& result) {
    bounded_poll_epoch_word_relaxed_into<Scope>(signal, expected_epoch,
                                                spin_limit, result);
    if (result.ready) thread_acquire<Scope>();
}

} // namespace kittens::distributed

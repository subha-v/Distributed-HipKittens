/**
 * @file
 * @brief Directed credit protocol for address-reused peer slots.
 */

#pragma once

#include <cstdint>

#include "completion.cuh"

namespace kittens::distributed {

/**
 * Wait for the unique prior consumer to retire a reusable slot.
 *
 * This is a relaxed control observation, not a payload acquire. Epoch one is
 * immediately reusable because the slot has never held a live payload. The
 * caller MUST call init_wait_result(result) before every invocation, including
 * the epoch-one fast path; that path intentionally writes only `ready`.
 */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void bounded_wait_slot_reusable_into(
        const std::uint32_t* credit, std::uint32_t next_epoch,
        std::uint64_t spin_limit, wait_result& result) {
    if (next_epoch <= 1u) {
        result.ready = true;
        return;
    }
    const std::uint32_t required = next_epoch - 1u;
    while (true) {
        result.observed = load_relaxed<Scope>(credit);
        if (epoch_ready(result.observed, required)) {
            result.ready = true;
            break;
        }
        ++result.spins;
        if (result.spins > spin_limit) break;
        detail::pause();
    }
}

/**
 * Drain every consuming wave's VMEM reads, then return one relaxed directed
 * lifetime credit.
 *
 * The pointer must name the producer-visible credit cell for exactly this
 * slot/dependency. This function is whole-CTA convergent; it hides no world or
 * grid rendezvous. The credit grants permission to overwrite; it does not
 * publish consumer writes or result data, which require a separate release.
 */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void drain_and_retire_slot(
        std::uint32_t* credit, std::uint32_t consumed_epoch) {
    consumer_drain();
    if (detail::is_leader()) {
        store_relaxed<Scope>(credit, consumed_epoch);
    }
}

} // namespace kittens::distributed

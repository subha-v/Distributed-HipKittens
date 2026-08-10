/**
 * @file
 * @brief Monotonic CTA epochs and cumulative device-local fan-in.
 */

#pragma once

#include <cstdint>

#include "sync.cuh"

namespace kittens::distributed {

struct counted_arrival {
    std::uint32_t previous;
    std::uint32_t epoch;
    std::uint32_t ordinal;
    bool last;
};

/**
 * Increment a CTA-private monotonic epoch and broadcast its new value.
 *
 * Every thread in the CTA must call this function convergently. The cell must
 * be initialized before launch and must not be shared by concurrently running
 * CTAs. Epoch zero is reserved for initialization; the first call returns one.
 * Quiesce and reinitialize the protocol before this 32-bit cell wraps.
 * The post-increment agent load is an LDS-free broadcast.
 */
KITTENS_DISTRIBUTED_DEVICE_INLINE std::uint32_t epoch32(
        std::uint32_t* monotonic_cell) {
#if defined(__HIP_DEVICE_COMPILE__)
    if (detail::is_leader()) {
        (void)fetch_add_relaxed<memory_scope::agent>(monotonic_cell, 1u);
    }
    detail::cta_barrier();
    const std::uint32_t epoch =
        load_relaxed<memory_scope::agent>(monotonic_cell);
    // Prevent the leader from incrementing the next convergent epoch while a
    // slower wave is still capturing this one.
    detail::cta_barrier();
    return epoch;
#else
    return __atomic_add_fetch(monotonic_cell, 1u, __ATOMIC_RELAXED);
#endif
}

/**
 * Record one local producer after its payload release.
 *
 * Call from one elected thread per producer CTA, after
 * producer_drain_release(). The cumulative counter is zeroed once at setup and
 * never reset. `expected` is the fixed producer count for this dependency key;
 * changing it across epochs breaks the quotient/remainder identity.
 *
 * The agent-scope acq_rel RMW chains producer arrivals locally. This raw form
 * does not export the effects newly acquired by the last RMW to another agent.
 * Before a relaxed remote readiness store, either use
 * counted_arrive_release_into() or issue thread_release<system>() after this
 * call on the last arriver.
 */
KITTENS_DISTRIBUTED_DEVICE_INLINE void counted_arrive_into(
        std::uint32_t* cumulative_counter, std::uint32_t expected,
        counted_arrival& result) {
    if (expected == 0u) {
        result = {0u, 0u, 0u, false};
        return;
    }
    const std::uint32_t previous =
        detail::fetch_add_acq_rel<memory_scope::agent>(cumulative_counter, 1u);
    result.previous = previous;
    result.epoch = previous / expected + 1u;
    result.ordinal = previous % expected;
    result.last = result.ordinal == expected - 1u;
}

/**
 * Count one local producer and release the completed fan-in for publication.
 *
 * Only the last arriver executes the selected-scope release. A following
 * relaxed readiness store from that same thread may then publish every
 * producer payload ordered through the local acq_rel RMW chain.
 */
template<memory_scope PublicationScope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void counted_arrive_release_into(
        std::uint32_t* cumulative_counter, std::uint32_t expected,
        counted_arrival& result) {
    counted_arrive_into(cumulative_counter, expected, result);
    if (result.last) thread_release<PublicationScope>();
}

} // namespace kittens::distributed

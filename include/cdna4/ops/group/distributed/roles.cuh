/**
 * @file
 * @brief Minimum-progress role specialization for persistent CDNA4 kernels.
 *
 * These are progress-partition primitives, not a scheduler or task graph. The
 * service/compute split is fixed at one rendezvous point per phase boundary
 * (COMET/MoK style: host-select the count, never resize inside the phase);
 * compute capacity is allowed to flow elastically into service/reduction work
 * only *after* its own queue drains.
 */

#pragma once

#include <cstdint>

#include "sync.cuh"
#include "completion.cuh"

namespace kittens::distributed {

/**
 * One CTA's role assignment from a finish-order partition.
 *
 * `ordinal` is the raw finish ticket. When `service` is true the CTA is one of
 * the first `service_count` finishers and owns dense `service_id`; otherwise it
 * owns dense compute id `compute_id` of a pool of `compute_count`. Dense ids
 * exist so downstream task loops may stride by `compute_count` and cover every
 * task exactly once regardless of which physical CTAs were reserved.
 */
struct role_partition {
    std::uint32_t ordinal;
    std::uint32_t service_id;
    std::uint32_t compute_id;
    std::uint16_t service_count;
    std::uint16_t compute_count;
    bool service;

    [[nodiscard]] KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE bool is_service()
            const {
        return service;
    }
    [[nodiscard]] KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE bool is_compute()
            const {
        return !service;
    }
};

/**
 * Partition a persistent grid into compute vs minimum-progress service roles.
 *
 * Whole-CTA convergent: every thread calls. One elected thread per CTA takes
 * a ticket from `ticket_cell` (a monotonic agent-scope counter, zeroed by the
 * caller once per phase/epoch); the assignment is broadcast CTA-locally. The
 * first `service_ctas` finishers become service CTAs; everyone else receives a
 * dense compute id in `[0, total_ctas - service_ctas)`.
 *
 * Choosing the *first finishers* avoids any additional grid barrier and makes
 * the service pool available at the earliest instant the preceding phase lets
 * anyone go. There is deliberately no reordering, preemption, or in-phase
 * resizing: the split is a plain function of arrival order at one point.
 */
[[nodiscard]] KITTENS_DISTRIBUTED_DEVICE_INLINE role_partition
finish_order_partition(std::uint32_t* ticket_cell, std::uint32_t service_ctas,
                       std::uint32_t total_ctas) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ std::uint32_t ordinal_shared;
    if (detail::is_leader()) {
        ordinal_shared =
            fetch_add_relaxed<memory_scope::agent>(ticket_cell, 1u);
    }
    detail::cta_barrier();
    const std::uint32_t ordinal = ordinal_shared;
    detail::cta_barrier();  // allow ordinal_shared reuse by a later call
#else
    (void)ticket_cell;
    const std::uint32_t ordinal = 0u;
    (void)total_ctas;
#endif
    role_partition role;
    role.ordinal = ordinal;
    role.service_count = static_cast<std::uint16_t>(service_ctas);
    role.compute_count =
        static_cast<std::uint16_t>(total_ctas - service_ctas);
    role.service = ordinal < service_ctas;
    role.service_id = role.service ? ordinal : 0u;
    role.compute_id = role.service ? 0u : ordinal - service_ctas;
    return role;
}

/**
 * Publish a completed independent output tile after a caller-established
 * payload release... or establish it here: this is the convenience form that
 * performs the release fence itself. Use when the publishing thread's prior
 * payload stores are already drained at the scope of its consumers (AMD:
 * `s_waitcnt vmcnt(0)` completed for the publishing wave/CTA).
 */
template<memory_scope Scope = memory_scope::agent>
KITTENS_DISTRIBUTED_DEVICE_INLINE void publish_tile_release(
        std::uint32_t* tile_key, std::uint32_t tile_value) {
    thread_release<Scope>();
    store_relaxed<Scope>(tile_key, tile_value);
}

/**
 * Bounded observation of one published tile key, with a pure acquire on
 * success before the caller touches payload. Caller initializes `result`.
 * Mirrors the tile-completion half of the split-release discipline: poll the
 * key, acquire, then read.
 */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void wait_tile_acquire_into(
        const std::uint32_t* tile_key, std::uint32_t expected_value,
        std::uint64_t spin_limit, wait_result& result) {
    bounded_poll_relaxed_into<Scope>(tile_key, expected_value, spin_limit,
                                     result);
    if (result.ready) thread_acquire<Scope>();
}

/**
 * Retire a completed protocol epoch on a monotonic 64-bit cell. The caller
 * owns the release that makes the epoch's payload writes visible; this is a
 * semantic alias so protocol code names its intent.
 */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void retire_epoch(std::uint64_t* epoch_cell,
                                                    std::uint64_t epoch) {
    store_relaxed<Scope>(epoch_cell, epoch);
}

} // namespace kittens::distributed

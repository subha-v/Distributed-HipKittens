/**
 * @file
 * @brief Monotonic publication and bounded readiness observation.
 */

#pragma once

#include <cstdint>

#include "sync.cuh"

namespace kittens::distributed {

struct wait_result {
    std::uint32_t observed;
    std::uint64_t spins;
    bool ready;
};

/** Caller-owned result for 64-bit completion words and counters. */
struct wait_result64 {
    std::uint64_t observed;
    std::uint64_t spins;
    bool ready;
};

KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE void init_wait_result(
        wait_result& result) {
    result.observed = 0u;
    result.spins = 0u;
    result.ready = false;
}

KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE void init_wait_result(
        wait_result64& result) {
    result.observed = 0u;
    result.spins = 0u;
    result.ready = false;
}

/** Pack an epoch and extent/payload into one self-describing completion word. */
[[nodiscard]] KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE constexpr std::uint64_t
epoch_word(std::uint32_t epoch, std::uint32_t payload) {
    return (static_cast<std::uint64_t>(epoch) << 32u) | payload;
}

[[nodiscard]] KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE constexpr std::uint32_t
epoch_word_epoch(std::uint64_t word) {
    return static_cast<std::uint32_t>(word >> 32u);
}

[[nodiscard]] KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE constexpr std::uint32_t
epoch_word_payload(std::uint64_t word) {
    return static_cast<std::uint32_t>(word);
}

[[nodiscard]] KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE bool epoch_ready(
        std::uint32_t observed, std::uint32_t expected) {
    // Preserve the donor's plain monotonic comparison. A protocol instance
    // must be quiesced and reinitialized before this 32-bit epoch wraps; zero
    // is reserved for setup and modular rollover is not inferred.
    return observed >= expected;
}

/**
 * Cheap relaxed publication after a caller-established release.
 *
 * This separation permits one producer_drain_release() to cover several
 * completion cells. Calling it without the release is a protocol error.
 */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void publish_epoch_relaxed(
        std::uint32_t* signal, std::uint32_t epoch) {
    store_relaxed<Scope>(signal, epoch);
}

/** Publish one 64-bit epoch/payload word after a caller-established release. */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void publish_epoch_word_relaxed(
        std::uint64_t* signal, std::uint32_t epoch, std::uint32_t payload) {
    store_relaxed<Scope>(signal, epoch_word(epoch, payload));
}

/** Whole-CTA store drain + selected-scope release + leader publication. */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void release_and_publish(
        std::uint32_t* signal, std::uint32_t epoch) {
    producer_drain_release<Scope>();
    if (detail::is_leader()) publish_epoch_relaxed<Scope>(signal, epoch);
}

/**
 * Bounded relaxed monotonic observation with no payload acquire.
 *
 * The caller must initialize `result` with init_wait_result(). Keeping result
 * storage caller-owned is code-generation sensitive: gfx942 exp_011's
 * aggregate return changed the bounded-poll CFG, while exp_012's reference
 * parameter restored manual/helper ISA and resource equality.
 *
 * A wider consumer role can have selected lanes poll here, converge, and call
 * cta_acquire<Scope>() before reading payload.
 */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void bounded_poll_relaxed_into(
        const std::uint32_t* signal, std::uint32_t expected_epoch,
        std::uint64_t spin_limit, wait_result& result) {
    while (true) {
        result.observed = load_relaxed<Scope>(signal);
        if (epoch_ready(result.observed, expected_epoch)) {
            result.ready = true;
            break;
        }
        ++result.spins;
        if (result.spins > spin_limit) break;
        detail::pause();
    }
}

/** 64-bit monotonic-counter overload with the same caller-owned shape. */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void bounded_poll_relaxed_into(
        const std::uint64_t* signal, std::uint64_t expected,
        std::uint64_t spin_limit, wait_result64& result) {
    while (true) {
        result.observed = load_relaxed<Scope>(signal);
        if (result.observed >= expected) {
            result.ready = true;
            break;
        }
        ++result.spins;
        if (result.spins > spin_limit) break;
        detail::pause();
    }
}

/**
 * Poll a self-describing 64-bit word until its high-half epoch equals expected.
 * Exact equality is intentional for address-reused word cells; the low half is
 * available to the caller as epoch_word_payload(result.observed).
 */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void bounded_poll_epoch_word_relaxed_into(
        const std::uint64_t* signal, std::uint32_t expected_epoch,
        std::uint64_t spin_limit, wait_result64& result) {
    while (true) {
        result.observed = load_relaxed<Scope>(signal);
        if (epoch_word_epoch(result.observed) == expected_epoch) {
            result.ready = true;
            break;
        }
        ++result.spins;
        if (result.spins > spin_limit) break;
        detail::pause();
    }
}

/**
 * Bounded monotonic observation with a success-only pure acquire.
 *
 * Only the calling thread is ordered to consume payload after this function.
 * A wider consumer role must use bounded_poll_relaxed_into(), converge, then
 * call cta_acquire<Scope>() explicitly.
 */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void bounded_observe_acquire_into(
        const std::uint32_t* signal, std::uint32_t expected_epoch,
        std::uint64_t spin_limit, wait_result& result) {
    bounded_poll_relaxed_into<Scope>(signal, expected_epoch, spin_limit,
                                     result);
    if (result.ready) thread_acquire<Scope>();
}

/** 64-bit monotonic observation plus calling-thread pure acquire. */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void bounded_observe_acquire_into(
        const std::uint64_t* signal, std::uint64_t expected,
        std::uint64_t spin_limit, wait_result64& result) {
    bounded_poll_relaxed_into<Scope>(signal, expected, spin_limit, result);
    if (result.ready) thread_acquire<Scope>();
}

} // namespace kittens::distributed

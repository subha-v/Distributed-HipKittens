/**
 * @file
 * @brief CDNA3 ordering building blocks for distributed payload lifetime.
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "detail/config.cuh"

namespace kittens::distributed {

/** Visibility domain for device-side synchronization metadata. */
enum class memory_scope : std::uint8_t {
    /** All agents on the current GPU. */
    agent,
    /** All agents in the peer-visible system. */
    system,
};

namespace detail {

template<memory_scope Scope, typename T>
KITTENS_DISTRIBUTED_DEVICE_INLINE T load_relaxed(const T* pointer) {
    static_assert(std::is_integral_v<T>);
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr int hip_scope = Scope == memory_scope::system
                                  ? __HIP_MEMORY_SCOPE_SYSTEM
                                  : __HIP_MEMORY_SCOPE_AGENT;
    return __hip_atomic_load(pointer, __ATOMIC_RELAXED,
                             hip_scope);
#else
    return __atomic_load_n(pointer, __ATOMIC_RELAXED);
#endif
}

template<memory_scope Scope, typename T>
KITTENS_DISTRIBUTED_DEVICE_INLINE void store_relaxed(T* pointer, T value) {
    static_assert(std::is_integral_v<T>);
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr int hip_scope = Scope == memory_scope::system
                                  ? __HIP_MEMORY_SCOPE_SYSTEM
                                  : __HIP_MEMORY_SCOPE_AGENT;
    __hip_atomic_store(pointer, value, __ATOMIC_RELAXED,
                       hip_scope);
#else
    __atomic_store_n(pointer, value, __ATOMIC_RELAXED);
#endif
}

template<memory_scope Scope, typename T>
KITTENS_DISTRIBUTED_DEVICE_INLINE T fetch_add_relaxed(T* pointer, T value) {
    static_assert(std::is_integral_v<T>);
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr int hip_scope = Scope == memory_scope::system
                                  ? __HIP_MEMORY_SCOPE_SYSTEM
                                  : __HIP_MEMORY_SCOPE_AGENT;
    return __hip_atomic_fetch_add(pointer, value, __ATOMIC_RELAXED,
                                  hip_scope);
#else
    return __atomic_fetch_add(pointer, value, __ATOMIC_RELAXED);
#endif
}

template<memory_scope Scope, typename T>
KITTENS_DISTRIBUTED_DEVICE_INLINE T fetch_add_acq_rel(T* pointer, T value) {
    static_assert(std::is_integral_v<T>);
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr int hip_scope = Scope == memory_scope::system
                                  ? __HIP_MEMORY_SCOPE_SYSTEM
                                  : __HIP_MEMORY_SCOPE_AGENT;
    return __hip_atomic_fetch_add(pointer, value, __ATOMIC_ACQ_REL,
                                  hip_scope);
#else
    return __atomic_fetch_add(pointer, value, __ATOMIC_ACQ_REL);
#endif
}

KITTENS_DISTRIBUTED_DEVICE_INLINE void pause() {
#if defined(__HIP_DEVICE_COMPILE__)
    // Immediate 4 is code-generation sensitive; s_sleep rejects runtime values.
    __builtin_amdgcn_s_sleep(4);
#endif
}

template<memory_scope Scope>
KITTENS_DISTRIBUTED_DEVICE_INLINE void release_fence() {
#if defined(__HIP_DEVICE_COMPILE__)
    // Keep scope strings explicit: they select different gfx942 cache actions.
    if constexpr (Scope == memory_scope::system) {
        // Pure release, unlike __threadfence_system()'s bidirectional fence.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    } else {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    }
#else
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
}

template<memory_scope Scope>
KITTENS_DISTRIBUTED_DEVICE_INLINE void acquire_fence() {
#if defined(__HIP_DEVICE_COMPILE__)
    // Pure acquire: do not replace with __threadfence_system(), which adds an
    // unnecessary release/writeback at the consumer.
    if constexpr (Scope == memory_scope::system) {
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    } else {
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
#else
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
}

} // namespace detail

/** Relaxed atomic metadata load at an explicit visibility scope. */
template<memory_scope Scope, typename T>
[[nodiscard]] KITTENS_DISTRIBUTED_DEVICE_INLINE T load_relaxed(
        const T* pointer) {
    return detail::load_relaxed<Scope>(pointer);
}

/** Relaxed atomic metadata store at an explicit visibility scope. */
template<memory_scope Scope, typename T>
KITTENS_DISTRIBUTED_DEVICE_INLINE void store_relaxed(T* pointer, T value) {
    detail::store_relaxed<Scope>(pointer, value);
}

/**
 * Reserve `count` entries from a monotonic agent- or peer-visible counter.
 * Returns the first reserved index (the pre-increment value).
 */
template<memory_scope Scope, typename T>
[[nodiscard]] KITTENS_DISTRIBUTED_DEVICE_INLINE T fetch_add_relaxed(
        T* pointer, T count) {
    return detail::fetch_add_relaxed<Scope>(pointer, count);
}

/** Semantic alias for peer row-slot allocation. */
template<memory_scope Scope = memory_scope::system, typename T>
[[nodiscard]] KITTENS_DISTRIBUTED_DEVICE_INLINE T reserve_rows_relaxed(
        T* next_row, T rows) {
    return fetch_add_relaxed<Scope>(next_row, rows);
}

/** Calling-thread pure release/acquire; no CTA convergence is hidden. */
template<memory_scope Scope>
KITTENS_DISTRIBUTED_DEVICE_INLINE void thread_release() {
    detail::release_fence<Scope>();
}

template<memory_scope Scope>
KITTENS_DISTRIBUTED_DEVICE_INLINE void thread_acquire() {
    detail::acquire_fence<Scope>();
}

/**
 * Drain every producer wave, converge the CTA, and issue exactly one release
 * at the selected scope before publication. On wave64/max-WG64, the barriers
 * may legally disappear; multi-wave kernels require both convergence points.
 */
template<memory_scope Scope = memory_scope::system>
KITTENS_DISTRIBUTED_DEVICE_INLINE void producer_drain_release() {
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
    detail::cta_barrier();
    if (detail::is_leader()) detail::release_fence<Scope>();
    detail::cta_barrier();
}

/** Converge a consuming CTA before a pure acquire at the selected scope. */
template<memory_scope Scope>
KITTENS_DISTRIBUTED_DEVICE_INLINE void cta_acquire() {
    detail::cta_barrier();
    detail::acquire_fence<Scope>();
}

/** Finish all consumer VMEM accesses before a directed reuse credit. */
KITTENS_DISTRIBUTED_DEVICE_INLINE void consumer_drain() {
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    detail::cta_barrier();
}

} // namespace kittens::distributed

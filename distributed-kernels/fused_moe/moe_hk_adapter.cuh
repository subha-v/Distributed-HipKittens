#pragma once

// HipKittens mechanism adapter for the preserved PF6 MoE megakernel. Routing,
// task descriptors, LL128 wire bytes, grid barriers, and GEMM scheduling remain
// operator-local. IRIS supplies peer heap bases; MoRI remains only where the
// frozen LL128 row primitive requires its device declarations.

#include "kittens.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hk_moe {

using scope = kittens::distributed::memory_scope;

struct set_error_bit {
    int* error;
    int bit;

    __device__ __forceinline__ void operator()() const {
        if (error != nullptr) atomicOr(error, bit);
    }
};

struct symmetric_heap_descriptor {
    const std::byte* local_heap_base;
    kittens::peer_bases<std::byte> heap_bases;
};

static_assert(std::is_standard_layout_v<symmetric_heap_descriptor>);
static_assert(std::is_trivially_copyable_v<symmetric_heap_descriptor>);
static_assert(sizeof(void*) == 8,
              "MoE IRIS descriptor ABI requires 64-bit pointers");
static_assert(sizeof(symmetric_heap_descriptor) == 72,
              "MoE IRIS descriptor ABI must remain exactly 72 bytes");
static_assert(alignof(symmetric_heap_descriptor) == 8,
              "MoE IRIS descriptor ABI must remain 8-byte aligned");
static_assert(offsetof(symmetric_heap_descriptor, local_heap_base) == 0);
static_assert(offsetof(symmetric_heap_descriptor, heap_bases) == 8);

[[nodiscard]] inline symmetric_heap_descriptor make_symmetric_heap_descriptor(
        const void* local_heap_base,
        kittens::peer_bases<std::byte> heap_bases) {
    return {reinterpret_cast<const std::byte*>(local_heap_base), heap_bases};
}

template<typename T>
__device__ __forceinline__ T* peer_ptr(
        T* local, int rank, const symmetric_heap_descriptor* descriptor) {
    return kittens::distributed::translate_peer<8>(
        local, descriptor->local_heap_base, descriptor->heap_bases, rank);
}

template<typename T>
__device__ __forceinline__ const T* peer_ptr(
        const T* local, int rank,
        const symmetric_heap_descriptor* descriptor) {
    return kittens::distributed::translate_peer<8>(
        local, descriptor->local_heap_base, descriptor->heap_bases, rank);
}

__device__ __forceinline__ void store_dispatch_row(
        unsigned char* __restrict__ peer_row,
        const std::uint64_t* __restrict__ staged_row,
        std::size_t word_count, int lane_group, int lane_in_group) {
    k0p6_put_row_payload(peer_row, staged_row, word_count, lane_group,
                         lane_in_group);
}

__device__ __forceinline__ void release_cta_payload_system() {
    kittens::distributed::producer_drain_release();
}

__device__ __forceinline__ void release_signal_batch_system() {
    kittens::distributed::thread_release<scope::system>();
}

__device__ __forceinline__ void release_signal_batch_agent() {
    kittens::distributed::thread_release<scope::agent>();
}

__device__ __forceinline__ void acquire_payload_system() {
    kittens::distributed::thread_acquire<scope::system>();
}

__device__ __forceinline__ void acquire_payload_agent() {
    kittens::distributed::cta_acquire<scope::agent>();
}

template<bool Release = false>
__device__ __forceinline__ std::uint32_t arrive_local_count(
        std::uint32_t* count) {
    if constexpr (Release) release_signal_batch_agent();
    return kittens::distributed::fetch_add_relaxed<scope::agent>(count, 1u);
}

template<scope Scope>
__device__ __forceinline__ std::uint32_t reserve_row(
        std::uint32_t* counter) {
    return kittens::distributed::reserve_rows_relaxed<Scope>(counter, 1u);
}

template<std::uint32_t Expected, typename Timeout>
__device__ __forceinline__ bool poll_local_count(
        const std::uint32_t* count, std::uint64_t spin_limit,
        Timeout timeout) {
    kittens::distributed::wait_result result;
    kittens::distributed::init_wait_result(result);
    kittens::distributed::bounded_poll_relaxed_into<scope::agent>(
        count, Expected, spin_limit, result);
    if (!result.ready) timeout();
    return result.ready;
}

template<scope Scope, typename T>
__device__ __forceinline__ void publish_relaxed(T* signal, T value) {
    kittens::distributed::store_relaxed<Scope>(signal, value);
}

template<scope Scope, typename T>
__device__ __forceinline__ T load_relaxed(const T* signal) {
    return kittens::distributed::load_relaxed<Scope>(signal);
}

template<scope Scope>
__device__ __forceinline__ void publish_epoch(std::uint32_t* signal,
                                              std::uint32_t epoch) {
    if constexpr (Scope == scope::system) {
        kittens::distributed::publish_epoch_relaxed(signal, epoch);
    } else {
        publish_relaxed<Scope>(signal, epoch);
    }
}

template<scope Scope>
__device__ __forceinline__ void publish_epoch_payload(
        std::uint64_t* signal, std::uint32_t epoch, std::uint32_t payload) {
    kittens::distributed::publish_epoch_word_relaxed<Scope>(
        signal, epoch, payload);
}

template<scope Scope>
__device__ __forceinline__ void publish_value(std::uint64_t* signal,
                                              std::uint64_t value) {
    publish_relaxed<Scope>(signal, value);
}

template<typename Timeout>
__device__ __forceinline__ bool poll_epoch_system(
        const std::uint32_t* signal, std::uint32_t epoch,
        std::uint64_t spin_limit, Timeout timeout) {
    kittens::distributed::wait_result result;
    kittens::distributed::init_wait_result(result);
    kittens::distributed::bounded_poll_relaxed_into<scope::system>(
        signal, epoch, spin_limit, result);
    if (!result.ready) timeout();
    return result.ready;
}

__device__ __forceinline__ bool poll_epoch_word_system(
        const std::uint64_t* signal, std::uint32_t epoch,
        std::uint64_t spin_limit, int* error, int error_bit,
        std::uint32_t* payload, std::uint32_t* spin_debug) {
    kittens::distributed::wait_result64 result;
    kittens::distributed::init_wait_result(result);
    kittens::distributed::bounded_poll_epoch_word_relaxed_into<scope::system>(
        signal, epoch, spin_limit, result);
    if (result.ready) {
        *payload = kittens::distributed::epoch_word_payload(result.observed);
        if (spin_debug != nullptr && result.spins > 0) {
            atomicMax(spin_debug, static_cast<std::uint32_t>(result.spins));
        }
        return true;
    }
    if (error != nullptr) atomicOr(error, error_bit);
    if (spin_debug != nullptr) {
        atomicMax(spin_debug + 1,
                  static_cast<std::uint32_t>(result.spins));
    }
    return false;
}

__device__ __forceinline__ bool poll_value_at_least_system(
        const std::uint64_t* signal, std::uint64_t expected,
        std::uint64_t spin_limit, int* error, int error_bit) {
    kittens::distributed::wait_result64 result;
    kittens::distributed::init_wait_result(result);
    kittens::distributed::bounded_poll_relaxed_into<scope::system>(
        signal, expected, spin_limit, result);
    if (!result.ready && error != nullptr) atomicOr(error, error_bit);
    return result.ready;
}

__device__ __forceinline__ bool error_bit_set_agent(const int* error,
                                                    int bit) {
    if (error == nullptr) return false;
    int observed = 0;
    if ((threadIdx.x & 63u) == 0u) {
        observed = kittens::distributed::load_relaxed<scope::agent>(error);
    }
    observed = __shfl(observed, 0);
    return (observed & bit) != 0;
}

} // namespace hk_moe

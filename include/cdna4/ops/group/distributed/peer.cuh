/**
 * @file
 * @brief Typed symmetric-allocation peer address translation.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "../../../types/global/pgl.cuh"
#include "detail/config.cuh"

namespace kittens::distributed {

/**
 * Translate a pointer inside a local symmetric allocation to rank Rank.
 *
 * The caller supplies both allocation bases and owns their lifetime. The byte
 * offset is deliberately computed from the allocation base, not from a tensor
 * view base, so suballocations remain intact. No allocation, host call, or
 * synchronization is hidden here.
 */
template<int Rank, typename T>
[[nodiscard]] KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE T* translate_peer(
        T* local_pointer, const void* local_allocation_base,
        const peer_bases<std::byte>& bases) {
    static_assert(Rank >= 0 && Rank < pgl_max_world_size,
                  "peer rank must be in [0, 8)");
    const auto offset = reinterpret_cast<std::uintptr_t>(local_pointer) -
                        reinterpret_cast<std::uintptr_t>(local_allocation_base);
    std::byte* const selected = bases.template get<Rank>();
    if (selected == nullptr) return nullptr;
    const auto peer_base = reinterpret_cast<std::uintptr_t>(selected);
    return reinterpret_cast<T*>(peer_base + offset);
}

/** Runtime named-base selection; invalid/inactive ranks fail closed. */
template<int World = pgl_max_world_size, typename T>
[[nodiscard]] KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE T* translate_peer(
        T* local_pointer, const void* local_allocation_base,
        const peer_bases<std::byte>& bases, int rank) {
    std::byte* const peer_base = bases.template select<World>(rank);
    if (peer_base == nullptr) return nullptr;
    const auto offset = reinterpret_cast<std::uintptr_t>(local_pointer) -
                        reinterpret_cast<std::uintptr_t>(local_allocation_base);
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(peer_base) +
                                offset);
}

/** Form a typed pointer from a caller-provided byte offset on one peer. */
template<int Rank, typename T>
[[nodiscard]] KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE T* peer_offset(
        const peer_bases<std::byte>& bases, std::size_t byte_offset) {
    std::byte* const selected = bases.template get<Rank>();
    return selected == nullptr
               ? nullptr
               : reinterpret_cast<T*>(
                     reinterpret_cast<std::uintptr_t>(selected) + byte_offset);
}

template<int World = pgl_max_world_size, typename T>
[[nodiscard]] KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE T* peer_offset(
        const peer_bases<std::byte>& bases, int rank,
        std::size_t byte_offset) {
    std::byte* const peer_base = bases.template select<World>(rank);
    return peer_base == nullptr
               ? nullptr
               : reinterpret_cast<T*>(
                     reinterpret_cast<std::uintptr_t>(peer_base) + byte_offset);
}

} // namespace kittens::distributed

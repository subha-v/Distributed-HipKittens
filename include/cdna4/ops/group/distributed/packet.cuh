/**
 * @file
 * @brief Explicit 16-byte local/peer packet movement.
 */

#pragma once

#include <cstddef>

#include "detail/config.cuh"

namespace kittens::distributed {

using packet16 = detail::packet16;

/**
 * One aligned posted packet store. This is payload movement only: it contains
 * no wait, fence, completion update, or cache hint. The gfx950 donors used the
 * same direct uint4 assignment; keep the 16-byte type intact in future ISA A/Bs.
 */
KITTENS_DISTRIBUTED_DEVICE_INLINE void store_packet16(
        void* __restrict__ destination, const packet16& value) {
    *reinterpret_cast<packet16*>(destination) = value;
}

KITTENS_DISTRIBUTED_DEVICE_INLINE packet16 load_packet16(
        const void* __restrict__ source) {
    return *reinterpret_cast<const packet16*>(source);
}

[[nodiscard]] KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE bool packet_contract(
        const void* destination, const void* source, std::size_t bytes) {
    return detail::is_aligned_16(destination) && detail::is_aligned_16(source) &&
           (bytes & 15u) == 0u;
}

/**
 * CTA-cooperative staged-memory to peer-memory transfer.
 *
 * Runtime packet counts are appropriate for staged LDS/global memory. Do not
 * use this overload for a register-resident `packet16[N]`: dynamic indexing can
 * demote that array to scratch. Use store_packet_row<N> for register packets.
 */
KITTENS_DISTRIBUTED_DEVICE_INLINE void store_peer_packets(
        void* __restrict__ destination, const void* __restrict__ source,
        std::size_t bytes, unsigned int tid, unsigned int threads) {
    auto* const out = reinterpret_cast<packet16*>(destination);
    const auto* const in = reinterpret_cast<const packet16*>(source);
    const std::size_t count = bytes >> 4;
#pragma unroll 1
    for (std::size_t packet = tid; packet < count; packet += threads) {
        out[packet] = in[packet];
    }
}

KITTENS_DISTRIBUTED_DEVICE_INLINE void store_peer_packets(
        void* __restrict__ destination, const void* __restrict__ source,
        std::size_t bytes) {
    store_peer_packets(destination, source, bytes, detail::thread_id(),
                       detail::thread_count());
}

[[nodiscard]] KITTENS_DISTRIBUTED_DEVICE_INLINE bool store_peer_packets_checked(
        void* __restrict__ destination, const void* __restrict__ source,
        std::size_t bytes, unsigned int tid, unsigned int threads) {
    if (threads == 0u || !packet_contract(destination, source, bytes)) {
        return false;
    }
    store_peer_packets(destination, source, bytes, tid, threads);
    return true;
}

[[nodiscard]] KITTENS_DISTRIBUTED_DEVICE_INLINE bool store_peer_packets_checked(
        void* __restrict__ destination, const void* __restrict__ source,
        std::size_t bytes) {
    return store_peer_packets_checked(destination, source, bytes,
                                      detail::thread_id(),
                                      detail::thread_count());
}

/** Peer/local source to caller-selected destination with the same packet form. */
KITTENS_DISTRIBUTED_DEVICE_INLINE void load_peer_packets(
        void* __restrict__ destination, const void* __restrict__ source,
        std::size_t bytes, unsigned int tid, unsigned int threads) {
    auto* const out = reinterpret_cast<packet16*>(destination);
    const auto* const in = reinterpret_cast<const packet16*>(source);
    const std::size_t count = bytes >> 4;
#pragma unroll 1
    for (std::size_t packet = tid; packet < count; packet += threads) {
        out[packet] = in[packet];
    }
}

template<std::size_t Chunks>
KITTENS_DISTRIBUTED_DEVICE_INLINE void store_packet_row(
        std::byte* destination, const packet16 (&values)[Chunks],
        unsigned int lane, std::size_t lane_stride_bytes = 64u * 16u) {
    static_assert(Chunks > 0, "a packet row must contain at least one packet");
#pragma unroll
    for (std::size_t chunk = 0; chunk < Chunks; ++chunk) {
        store_packet16(destination + chunk * lane_stride_bytes + lane * 16u,
                       values[chunk]);
    }
}

} // namespace kittens::distributed

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

/**
 * Streaming (non-temporal) peer copy.
 *
 * Byte-for-byte the same traffic as `store_peer_packets`, with `nt` requested
 * on BOTH the source load and the peer store.
 *
 * Rationale, from measurement rather than taste. The payload is read exactly
 * once and written exactly once and is never reused by either side, yet the
 * default lowering caches it fully — the ISA read in
 * `overnight/experiments/exp_03_push_throughput/isa_push_loop.md` confirmed the
 * emitted `flat_store_dwordx4` carries no `sc0`, no `sc1` and no `nt` at all.
 * A service pool streaming hundreds of MB through the caches therefore evicts
 * the concurrent GEMM's weights from the per-XCD L2 and the 256 MB LLC.
 * `exp_16` bounded that payload contention at roughly 1,000 us of the
 * concurrent phase, which is the largest single cost left in the kernel.
 *
 * `nt` is a replacement-policy hint (ISA 9.1.10.2, LLC Hit Evict), not a
 * coherence bit, so it does not alter the release/acquire discipline the caller
 * wraps around this copy.
 */
KITTENS_DISTRIBUTED_DEVICE_INLINE void store_peer_packets_streaming(
        void* __restrict__ destination, const void* __restrict__ source,
        std::size_t bytes, unsigned int tid, unsigned int threads) {
    using vector4 = unsigned int __attribute__((ext_vector_type(4)));
    auto* const out = reinterpret_cast<vector4*>(destination);
    const auto* const in = reinterpret_cast<const vector4*>(source);
    const std::size_t count = bytes >> 4;
#pragma unroll 1
    for (std::size_t packet = tid; packet < count; packet += threads) {
        __builtin_nontemporal_store(__builtin_nontemporal_load(in + packet),
                                    out + packet);
    }
}

/**
 * Batched multi-region variant of `store_peer_packets`.
 *
 * The single-region form is a dependent load-then-store chain, so each lane
 * keeps exactly ONE memory operation in flight and the copy runs at one round
 * trip per 16 bytes per lane. That is the right shape when one region refills
 * the lanes many times over. It is the wrong shape for a caller holding MANY
 * SMALL regions: a region narrower than `threads * 16` bytes cannot even give
 * every lane one packet, so there is nothing to pipeline within it.
 *
 * This overload copies `Regions` equally-sized regions together, issuing every
 * load before any store, so each lane keeps `Regions` loads in flight while
 * each individual region keeps the identical fully-coalesced access pattern.
 * The pointer arrays and the staging buffer are indexed only under
 * `#pragma unroll`, so they stay register-resident (see the note on
 * `store_peer_packets` about dynamic indexing demoting a packet array to
 * scratch).
 *
 * Every region must have the same `bytes`. Regions may alias only if the
 * caller would also accept `Regions` independent copies in an unspecified
 * order.
 */
template<unsigned int Regions>
KITTENS_DISTRIBUTED_DEVICE_INLINE void store_peer_packets_multi(
        void* const (&destinations)[Regions],
        const void* const (&sources)[Regions],
        std::size_t bytes, unsigned int tid, unsigned int threads) {
    static_assert(Regions > 0, "a multi-region copy needs at least one region");
    const std::size_t count = bytes >> 4;
#pragma unroll 1
    for (std::size_t packet = tid; packet < count; packet += threads) {
        packet16 staged[Regions];
#pragma unroll
        for (unsigned int region = 0; region < Regions; ++region) {
            staged[region] =
                reinterpret_cast<const packet16*>(sources[region])[packet];
        }
#pragma unroll
        for (unsigned int region = 0; region < Regions; ++region) {
            reinterpret_cast<packet16*>(destinations[region])[packet] =
                staged[region];
        }
    }
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

/**
 * @file
 * @brief Parallel global layouts for rank-aware symmetric-memory addressing.
 *
 * A parallel global layout (PGL) is an ordinary global layout plus the
 * allocation bases for the same symmetric allocation on each rank. Calling
 * `on<Rank>()` or `on(rank)` returns a copy of the ordinary layout with only
 * its `raw_ptr` rebased. Shape, strides, dtype, and a nonzero suballocation
 * offset are preserved.
 *
 * This type owns no memory and carries no synchronization, epoch, route,
 * completion, task, or scheduling state.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__HIPCC__)
#define KITTENS_PGL_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define KITTENS_PGL_HOST_DEVICE_INLINE inline
#endif

namespace kittens {

inline constexpr int pgl_max_world_size = 8;

namespace ducks {
namespace pgl {

struct identifier {};

/** A lightweight global layout that can be copied and have its pointer rebound. */
template<typename T>
concept compatible_global_layout = requires(T layout, typename T::dtype* pointer) {
    typename T::dtype;
    layout.raw_ptr = pointer;
};

/** Concept for all parallel global layouts. */
template<typename T>
concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

} // namespace pgl
} // namespace ducks

namespace detail {

template<typename T, typename Address>
KITTENS_PGL_HOST_DEVICE_INLINE T* pgl_address_cast(Address address) {
    using address_type = std::remove_cv_t<std::remove_reference_t<Address>>;
    if constexpr (std::is_pointer_v<address_type>) {
        return reinterpret_cast<T*>(address);
    }
    else {
        return reinterpret_cast<T*>(static_cast<std::uintptr_t>(address));
    }
}

template<typename T>
KITTENS_PGL_HOST_DEVICE_INLINE T* pgl_offset_base(
        std::byte* base, std::size_t byte_offset) {
    if (base == nullptr) return nullptr;
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(base) +
                                byte_offset);
}

template<int Rank, int World, typename T, typename PeerSource>
KITTENS_PGL_HOST_DEVICE_INLINE T* pgl_source_base(const PeerSource& source) {
    static_assert(Rank >= 0 && Rank < pgl_max_world_size);
    if constexpr (Rank < World) {
        // IRIS's device view exposes get_heap_base(int). Rank is a literal in
        // every instantiation here, avoiding a variable GEP into its base table.
        return pgl_address_cast<T>(source.get_heap_base(Rank));
    }
    else {
        return nullptr;
    }
}

} // namespace detail

/**
 * @brief Named corresponding peer bases with no runtime-indexed pointer array.
 *
 * Named members are deliberate. On AMDGPU, variable indexing into an embedded
 * eight-pointer array can block scalar replacement and create private scratch.
 * The runtime selector below touches every member at a constant field offset.
 */
template<typename T = std::byte>
struct peer_bases {
    using value_type = T;

    T* rank0;
    T* rank1;
    T* rank2;
    T* rank3;
    T* rank4;
    T* rank5;
    T* rank6;
    T* rank7;

    template<int Rank>
    [[nodiscard]] KITTENS_PGL_HOST_DEVICE_INLINE T* get() const {
        static_assert(Rank >= 0 && Rank < pgl_max_world_size,
                      "peer rank must be in [0, 8)");
        if constexpr (Rank == 0) return rank0;
        if constexpr (Rank == 1) return rank1;
        if constexpr (Rank == 2) return rank2;
        if constexpr (Rank == 3) return rank3;
        if constexpr (Rank == 4) return rank4;
        if constexpr (Rank == 5) return rank5;
        if constexpr (Rank == 6) return rank6;
        return rank7;
    }

    /** Invalid or inactive ranks fail closed with nullptr. */
    template<int World = pgl_max_world_size>
    [[nodiscard]] KITTENS_PGL_HOST_DEVICE_INLINE T* select(int rank) const {
        static_assert(World > 0 && World <= pgl_max_world_size,
                      "PGL world size must be in [1, 8]");
        T* selected = nullptr;
        if constexpr (World > 0) if (rank == 0) selected = rank0;
        if constexpr (World > 1) if (rank == 1) selected = rank1;
        if constexpr (World > 2) if (rank == 2) selected = rank2;
        if constexpr (World > 3) if (rank == 3) selected = rank3;
        if constexpr (World > 4) if (rank == 4) selected = rank4;
        if constexpr (World > 5) if (rank == 5) selected = rank5;
        if constexpr (World > 6) if (rank == 6) selected = rank6;
        if constexpr (World > 7) if (rank == 7) selected = rank7;
        return selected;
    }
};

/**
 * @brief Build named heap/arena bases from an IRIS-like non-owning view.
 *
 * `PeerSource` only needs `get_heap_base(int)`. This copies only the active
 * address values; it allocates and copies no payload. Use `offset_peer_bases`
 * before pairing these heap bases with a later allocation base.
 */
template<typename T = std::byte, int World = pgl_max_world_size, typename PeerSource>
[[nodiscard]] KITTENS_PGL_HOST_DEVICE_INLINE peer_bases<T>
make_peer_bases(const PeerSource& source) {
    static_assert(World > 0 && World <= pgl_max_world_size,
                  "PGL world size must be in [1, 8]");
    return {
        detail::pgl_source_base<0, World, T>(source),
        detail::pgl_source_base<1, World, T>(source),
        detail::pgl_source_base<2, World, T>(source),
        detail::pgl_source_base<3, World, T>(source),
        detail::pgl_source_base<4, World, T>(source),
        detail::pgl_source_base<5, World, T>(source),
        detail::pgl_source_base<6, World, T>(source),
        detail::pgl_source_base<7, World, T>(source),
    };
}

/**
 * @brief Shift peer heap/arena anchors to one corresponding suballocation.
 *
 * A PGL's local anchor and every peer anchor must identify the same byte in
 * their respective symmetric heaps. Backends such as IRIS expose heap bases,
 * while an operator commonly owns a later allocation. This helper applies the
 * allocation's heap-relative offset to every active named base on the host,
 * before the descriptor reaches hot device code.
 */
template<int World = pgl_max_world_size, typename T = std::byte>
[[nodiscard]] KITTENS_PGL_HOST_DEVICE_INLINE peer_bases<T>
offset_peer_bases(const peer_bases<std::byte>& bases,
                  std::size_t byte_offset) {
    static_assert(World > 0 && World <= pgl_max_world_size,
                  "PGL world size must be in [1, 8]");
    return {
        World > 0 ? detail::pgl_offset_base<T>(bases.rank0, byte_offset) : nullptr,
        World > 1 ? detail::pgl_offset_base<T>(bases.rank1, byte_offset) : nullptr,
        World > 2 ? detail::pgl_offset_base<T>(bases.rank2, byte_offset) : nullptr,
        World > 3 ? detail::pgl_offset_base<T>(bases.rank3, byte_offset) : nullptr,
        World > 4 ? detail::pgl_offset_base<T>(bases.rank4, byte_offset) : nullptr,
        World > 5 ? detail::pgl_offset_base<T>(bases.rank5, byte_offset) : nullptr,
        World > 6 ? detail::pgl_offset_base<T>(bases.rank6, byte_offset) : nullptr,
        World > 7 ? detail::pgl_offset_base<T>(bases.rank7, byte_offset) : nullptr,
    };
}

/**
 * @brief One ordinary layout projected onto the same allocation on each rank.
 */
template<ducks::pgl::compatible_global_layout GL, int World>
struct parallel_global_layout {
    static_assert(World > 0 && World <= pgl_max_world_size,
                  "PGL world size must be in [1, 8]");

    using identifier = ducks::pgl::identifier;
    using layout_type = GL;
    using dtype = typename GL::dtype;

    static constexpr int world_size = World;

    GL local;
    const std::byte* local_allocation_base;
    peer_bases<std::byte> peers;

    /** Compile-time rank projection. The descriptor is assumed valid. */
    template<int Rank>
    [[nodiscard]] KITTENS_PGL_HOST_DEVICE_INLINE GL on() const {
        static_assert(Rank >= 0 && Rank < World,
                      "peer rank must be inside the PGL world");
        return rebase(peers.template get<Rank>());
    }

    /** Runtime rank projection. Invalid or inactive ranks return a null view. */
    [[nodiscard]] KITTENS_PGL_HOST_DEVICE_INLINE GL on(int rank) const {
        std::byte* const peer_base = peers.template select<World>(rank);
        if (peer_base == nullptr) {
            GL out = local;
            out.raw_ptr = nullptr;
            return out;
        }
        return rebase(peer_base);
    }

private:
    [[nodiscard]] KITTENS_PGL_HOST_DEVICE_INLINE GL
    rebase(std::byte* peer_allocation_base) const {
        GL out = local;
        const auto local_address = reinterpret_cast<std::uintptr_t>(local.raw_ptr);
        const auto allocation_address =
            reinterpret_cast<std::uintptr_t>(local_allocation_base);
        const auto offset_bytes = local_address - allocation_address;
        const auto peer_address =
            reinterpret_cast<std::uintptr_t>(peer_allocation_base);
        out.raw_ptr = reinterpret_cast<dtype*>(peer_address + offset_bytes);
        return out;
    }
};

/** Concise public spelling used at kernel call sites. */
template<ducks::pgl::compatible_global_layout GL, int World>
using pgl = parallel_global_layout<GL, World>;

/** Build a PGL from already-materialized named bases. */
template<int World, ducks::pgl::compatible_global_layout GL, typename LocalBase>
[[nodiscard]] KITTENS_PGL_HOST_DEVICE_INLINE parallel_global_layout<GL, World>
make_parallel_global_layout(GL local, LocalBase local_allocation_base,
                            peer_bases<std::byte> peers) {
    return {
        local,
        detail::pgl_address_cast<const std::byte>(local_allocation_base),
        peers,
    };
}

static_assert(std::is_standard_layout_v<peer_bases<std::byte>>);
static_assert(std::is_trivially_copyable_v<peer_bases<std::byte>>);
static_assert(sizeof(peer_bases<std::byte>) ==
              pgl_max_world_size * sizeof(std::byte*));

} // namespace kittens

#undef KITTENS_PGL_HOST_DEVICE_INLINE

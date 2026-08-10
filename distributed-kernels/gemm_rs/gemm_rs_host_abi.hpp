#pragma once

// Host-only construction of the POD descriptors consumed by GEMM-RS.
//
// IRIS owns the symmetric allocations and peer mappings. This bridge only
// snapshots the corresponding allocation bases into the exact device ABI
// type; allocating, uploading, retaining, and launching remain caller-owned.

#include "../common/iris_adapter.hpp"
#include "gemm_rs_hk_adapter.cuh"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace hk_gemm_rs::host_abi {

inline constexpr int world_size = 8;

static_assert(sizeof(void*) == 8,
              "the GEMM-RS descriptor ABI requires 64-bit pointers");
static_assert(sizeof(symmetric_descriptor) == 9 * sizeof(void*));

namespace detail {

inline void validate_all_peer_bases(
        const kittens::peer_bases<std::byte>& bases) {
    for (int rank = 0; rank < world_size; ++rank) {
        if (bases.template select<world_size>(rank) == nullptr) {
            throw std::invalid_argument(
                "IRIS returned a null GEMM-RS peer allocation base");
        }
    }
}

} // namespace detail

/**
 * Snapshot one symmetric allocation into the exact GEMM-RS device POD.
 *
 * `local_allocation_base` must be the allocation anchor, not a later view into
 * that allocation. The common IRIS adapter applies its heap-relative offset to
 * every peer before the descriptor is constructed.
 */
[[nodiscard]] inline symmetric_descriptor snapshot_allocation_descriptor(
        const iris::iris_device_view& view,
        const void* local_allocation_base) {
    if (local_allocation_base == nullptr) {
        throw std::invalid_argument(
            "GEMM-RS local allocation base must not be null");
    }
    const auto bases =
        kittens::distributed::iris_backend::
            snapshot_peer_allocation_bases<world_size>(
                view, local_allocation_base);
    detail::validate_all_peer_bases(bases);
    return make_symmetric_descriptor(local_allocation_base, bases);
}

/** The two independently-uploaded descriptors required by the selected port. */
struct allocation_descriptors {
    symmetric_descriptor c_heap;
    symmetric_descriptor signals;
};

static_assert(std::is_standard_layout_v<allocation_descriptors>);
static_assert(std::is_trivially_copyable_v<allocation_descriptors>);

[[nodiscard]] inline allocation_descriptors snapshot_allocation_descriptors(
        const iris::iris_device_view& view, const void* local_c_heap,
        const void* local_signals) {
    return {
        snapshot_allocation_descriptor(view, local_c_heap),
        snapshot_allocation_descriptor(view, local_signals),
    };
}

} // namespace hk_gemm_rs::host_abi

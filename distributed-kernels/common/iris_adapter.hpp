#pragma once

// Host-side bridge between IRIS's symmetric heap and HipKittens PGLs.
//
// IRIS continues to own allocation, IPC exchange, and rank lifecycle.  Kernels
// receive only the eight named allocation bases plus the ordinary HipKittens
// layout.  Keeping iris_device_view out of hot-loop globals avoids a
// runtime-indexed heap_bases_[8] aggregate in device code.

#include "kittens.cuh"

#include <iris/iris.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace kittens::distributed::iris_backend {

template<int World>
[[nodiscard]] inline peer_bases<std::byte>
snapshot_peer_heap_bases(const iris::iris_device_view& view) {
    static_assert(World > 0 && World <= pgl_max_world_size,
                  "IRIS PGL world size must be in [1, 8]");
    if (view.world_size() != World) {
        throw std::invalid_argument(
            "IRIS runtime world size does not match the PGL World");
    }
    if (view.cur_rank() < 0 || view.cur_rank() >= view.world_size()) {
        throw std::invalid_argument("IRIS current rank is outside its world");
    }
    return make_peer_bases<std::byte, World>(view);
}

template<int World>
[[nodiscard]] inline peer_bases<std::byte>
snapshot_peer_allocation_bases(const iris::iris_device_view& view,
                               const void* local_allocation_base) {
    const auto heaps = snapshot_peer_heap_bases<World>(view);
    const auto local_heap_address = view.get_heap_base(view.cur_rank());
    const auto allocation_address =
        reinterpret_cast<std::uintptr_t>(local_allocation_base);
    if (allocation_address < local_heap_address) {
        throw std::invalid_argument(
            "IRIS allocation base precedes the local symmetric heap");
    }
    return offset_peer_bases<World>(
        heaps, static_cast<std::size_t>(allocation_address -
                                        local_heap_address));
}

template<int World, ducks::pgl::compatible_global_layout GL>
[[nodiscard]] inline pgl<GL, World>
snapshot_layout(GL local, const void* local_allocation_base,
                const iris::iris_device_view& view) {
    return make_parallel_global_layout<World>(
        local, local_allocation_base,
        snapshot_peer_allocation_bases<World>(view,
                                              local_allocation_base));
}

} // namespace kittens::distributed::iris_backend

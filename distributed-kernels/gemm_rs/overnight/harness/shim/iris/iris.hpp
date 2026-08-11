#pragma once

// Minimal iris_device_view for the single-process, 8-device validation harness.
//
// The production runtime obtains these heap bases from IRIS's MPI/IPC exchange.
// This harness instead allocates one fine-grained symmetric heap per device
// inside a single process, so every peer base is directly addressable and the
// eight bases can be handed to the *real* host ABI
// (hk_gemm_rs::host_abi::snapshot_allocation_descriptors) unchanged. Only the
// three accessors that ABI actually consumes are provided, matching the shape
// of the repository's own host-test fixture.

#include <array>
#include <cstddef>
#include <cstdint>

namespace iris {

class iris_device_view {
public:
    int rank = 0;
    int world = 1;
    std::array<std::uintptr_t, 8> heaps{};

    [[nodiscard]] int cur_rank() const { return rank; }
    [[nodiscard]] int world_size() const { return world; }
    [[nodiscard]] std::uintptr_t get_heap_base(int peer) const {
        return heaps[static_cast<std::size_t>(peer)];
    }
};

} // namespace iris

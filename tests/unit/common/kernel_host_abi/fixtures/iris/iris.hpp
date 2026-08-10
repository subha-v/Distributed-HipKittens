#pragma once

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

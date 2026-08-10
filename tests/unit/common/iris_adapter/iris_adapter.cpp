#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "../../../../distributed-kernels/common/iris_adapter.hpp"

namespace {

struct mock_layout {
    using dtype = std::uint16_t;

    dtype* raw_ptr;
    std::uint32_t rows;
    std::uint32_t columns;
};

struct alignas(16) rank_heap {
    std::array<std::byte, 4096> bytes{};
};

std::array<rank_heap, 8> heaps;

iris::iris_device_view make_view(int rank = 3, int world = 8) {
    iris::iris_device_view view;
    view.rank = rank;
    view.world = world;
    for (std::size_t peer = 0; peer < heaps.size(); ++peer) {
        view.heaps[peer] =
            reinterpret_cast<std::uintptr_t>(heaps[peer].bytes.data());
    }
    return view;
}

template<typename Function>
void expect_invalid_argument(Function&& function) {
    bool threw = false;
    try {
        function();
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main() {
    using namespace kittens;
    using namespace kittens::distributed::iris_backend;

    constexpr std::size_t allocation_offset = 0x300;
    constexpr std::size_t view_offset = 0x120;
    const auto view = make_view();
    const void* const local_allocation =
        heaps[3].bytes.data() + allocation_offset;
    const mock_layout local{
        reinterpret_cast<std::uint16_t*>(
            heaps[3].bytes.data() + allocation_offset + view_offset),
        17u,
        29u,
    };

    const auto peer_heaps = snapshot_peer_heap_bases<8>(view);
    assert(peer_heaps.rank5 == heaps[5].bytes.data());

    const auto peer_allocations =
        snapshot_peer_allocation_bases<8>(view, local_allocation);
    assert(peer_allocations.rank5 ==
           heaps[5].bytes.data() + allocation_offset);

    const auto layout = snapshot_layout<8>(local, local_allocation, view);
    const auto peer5 = layout.on<5>();
    assert(peer5.raw_ptr == reinterpret_cast<std::uint16_t*>(
                                heaps[5].bytes.data() + allocation_offset +
                                view_offset));
    assert(peer5.rows == local.rows && peer5.columns == local.columns);

    expect_invalid_argument([&] {
        (void)snapshot_peer_heap_bases<8>(make_view(3, 7));
    });
    expect_invalid_argument([&] {
        (void)snapshot_peer_heap_bases<8>(make_view(8, 8));
    });
    expect_invalid_argument([&] {
        const auto before_heap = reinterpret_cast<const void*>(
            view.get_heap_base(view.cur_rank()) - 1u);
        (void)snapshot_peer_allocation_bases<8>(view, before_heap);
    });

    return 0;
}

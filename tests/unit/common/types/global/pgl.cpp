#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifndef PGL_HEADER
#error "PGL_HEADER must name the CDNA3 or CDNA4 PGL header"
#endif

#include PGL_HEADER

namespace {

using element = std::uint16_t;

struct mock_global_layout {
    using dtype = element;

    element* raw_ptr;
    std::uint32_t batch;
    std::uint32_t depth;
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint64_t batch_stride;
    std::uint64_t depth_stride;
    std::uint64_t row_stride;
    std::uint64_t col_stride;
};

struct iris_like_device_view {
    std::array<std::uintptr_t, kittens::pgl_max_world_size> heap_bases;

    [[nodiscard]] std::uintptr_t get_heap_base(int rank) const {
        return heap_bases[static_cast<std::size_t>(rank)];
    }
};

struct alignas(16) rank_storage {
    std::array<std::byte, 4096> bytes{};
};

std::array<rank_storage, kittens::pgl_max_world_size> storage;

kittens::peer_bases<std::byte> named_bases(std::size_t offset = 0u) {
    return {
        storage[0].bytes.data() + offset, storage[1].bytes.data() + offset,
        storage[2].bytes.data() + offset, storage[3].bytes.data() + offset,
        storage[4].bytes.data() + offset, storage[5].bytes.data() + offset,
        storage[6].bytes.data() + offset, storage[7].bytes.data() + offset,
    };
}

iris_like_device_view iris_view() {
    iris_like_device_view result{};
    for (std::size_t rank = 0; rank < result.heap_bases.size(); ++rank) {
        result.heap_bases[rank] =
            reinterpret_cast<std::uintptr_t>(storage[rank].bytes.data());
    }
    return result;
}

void assert_metadata_preserved(const mock_global_layout& actual,
                               const mock_global_layout& expected) {
    assert(actual.batch == expected.batch);
    assert(actual.depth == expected.depth);
    assert(actual.rows == expected.rows);
    assert(actual.cols == expected.cols);
    assert(actual.batch_stride == expected.batch_stride);
    assert(actual.depth_stride == expected.depth_stride);
    assert(actual.row_stride == expected.row_stride);
    assert(actual.col_stride == expected.col_stride);
}

template<int Rank, typename PGL>
void check_rank(const PGL& pgl, const mock_global_layout& local,
                std::size_t offset_bytes) {
    const mock_global_layout compiletime = pgl.template on<Rank>();
    const mock_global_layout runtime = pgl.on(Rank);
    element* const expected = reinterpret_cast<element*>(
        storage[Rank].bytes.data() + offset_bytes);

    assert(compiletime.raw_ptr == expected);
    assert(runtime.raw_ptr == expected);
    assert_metadata_preserved(compiletime, local);
    assert_metadata_preserved(runtime, local);
}

template<typename PGL>
void check_all_ranks(const PGL& pgl, const mock_global_layout& local,
                     std::size_t offset_bytes) {
    check_rank<0>(pgl, local, offset_bytes);
    check_rank<1>(pgl, local, offset_bytes);
    check_rank<2>(pgl, local, offset_bytes);
    check_rank<3>(pgl, local, offset_bytes);
    check_rank<4>(pgl, local, offset_bytes);
    check_rank<5>(pgl, local, offset_bytes);
    check_rank<6>(pgl, local, offset_bytes);
    check_rank<7>(pgl, local, offset_bytes);
}

} // namespace

int main() {
    constexpr std::size_t allocation_offset_bytes = 0x300;
    constexpr std::size_t view_offset_bytes = 0x120;
    constexpr std::size_t expected_heap_offset =
        allocation_offset_bytes + view_offset_bytes;
    std::byte* const local_allocation_base =
        storage[3].bytes.data() + allocation_offset_bytes;
    const mock_global_layout local{
        reinterpret_cast<element*>(local_allocation_base + view_offset_bytes),
        2, 3, 5, 7,
        105, 35, 7, 1,
    };

    using pgl_type = kittens::pgl<mock_global_layout, 8>;
    static_assert(std::is_same_v<
                  pgl_type,
                  kittens::parallel_global_layout<mock_global_layout, 8>>);
    static_assert(kittens::ducks::pgl::compatible_global_layout<mock_global_layout>);
    static_assert(kittens::ducks::pgl::all<pgl_type>);
    static_assert(std::is_standard_layout_v<kittens::peer_bases<std::byte>>);
    static_assert(std::is_trivially_copyable_v<kittens::peer_bases<std::byte>>);
    static_assert(std::is_standard_layout_v<pgl_type>);
    static_assert(std::is_trivially_copyable_v<pgl_type>);

    const pgl_type from_named = kittens::make_parallel_global_layout<8>(
        local, local_allocation_base, named_bases(allocation_offset_bytes));
    check_all_ranks(from_named, local, expected_heap_offset);

    const iris_like_device_view view = iris_view();
    const auto peer_heap_bases =
        kittens::make_peer_bases<std::byte, 8>(view);
    const auto peer_allocation_bases = kittens::offset_peer_bases<8>(
        peer_heap_bases, allocation_offset_bytes);
    const pgl_type from_iris = kittens::make_parallel_global_layout<8>(
        local, local_allocation_base, peer_allocation_bases);
    check_all_ranks(from_iris, local, expected_heap_offset);

    assert(from_named.on(-1).raw_ptr == nullptr);
    assert(from_named.on(8).raw_ptr == nullptr);

    const auto world4 = kittens::make_parallel_global_layout<4>(
        local, local_allocation_base,
        kittens::offset_peer_bases<4>(
            kittens::make_peer_bases<std::byte, 4>(view),
            allocation_offset_bytes));
    assert(world4.on(3).raw_ptr ==
           reinterpret_cast<element*>(storage[3].bytes.data() +
                                      expected_heap_offset));
    assert(world4.on(4).raw_ptr == nullptr);
    assert(world4.peers.rank4 == nullptr);

    return 0;
}

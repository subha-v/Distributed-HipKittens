#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../../../../distributed-kernels/gemm_rs/gemm_rs_host_abi.hpp"
#include "../../../../distributed-kernels/fused_moe/moe_host_abi.hpp"

namespace {

struct alignas(64) rank_heap {
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
    static_assert(std::is_standard_layout_v<
                  hk_gemm_rs::symmetric_descriptor>);
    static_assert(std::is_trivially_copyable_v<
                  hk_gemm_rs::symmetric_descriptor>);
    static_assert(std::is_standard_layout_v<
                  hk_moe::symmetric_heap_descriptor>);
    static_assert(std::is_trivially_copyable_v<
                  hk_moe::symmetric_heap_descriptor>);

    constexpr std::size_t c_offset = 0x180;
    constexpr std::size_t signal_offset = 0x580;
    const auto view = make_view();
    const void* const local_c = heaps[3].bytes.data() + c_offset;
    const void* const local_signals =
        heaps[3].bytes.data() + signal_offset;

    const auto gemm =
        hk_gemm_rs::host_abi::snapshot_allocation_descriptors(
            view, local_c, local_signals);
    assert(gemm.c_heap.local_allocation_base == local_c);
    assert(gemm.c_heap.bases.rank6 ==
           heaps[6].bytes.data() + c_offset);
    assert(gemm.signals.local_allocation_base == local_signals);
    assert(gemm.signals.bases.rank1 ==
           heaps[1].bytes.data() + signal_offset);

    const auto moe = hk_moe::host_abi::snapshot_heap_descriptor(view);
    assert(moe.local_heap_base == heaps[3].bytes.data());
    assert(moe.heap_bases.rank0 == heaps[0].bytes.data());
    assert(moe.heap_bases.rank7 == heaps[7].bytes.data());

    using hk_moe::host_abi::descriptor_word;
    std::vector<descriptor_word> words(
        hk_moe::host_abi::donor_descriptor_words);
    for (std::size_t index = 0; index < words.size(); ++index) {
        words[index] = static_cast<descriptor_word>(index + 1);
    }
    const auto donor_words = words;

    constexpr std::uintptr_t first_uploaded_address = 0x1000;
    hk_moe::host_abi::append_symmetric_heap_descriptor(
        words,
        hk_moe::host_abi::device_visible_descriptor_address(
            first_uploaded_address));
    assert(words.size() == hk_moe::host_abi::descriptor_words);
    for (std::size_t index = 0; index < donor_words.size(); ++index) {
        assert(words[index] == donor_words[index]);
    }
    const auto first_checked = hk_moe::host_abi::validate_descriptor(
        std::span<const descriptor_word>(words));
    assert(first_checked.value() == first_uploaded_address);

    constexpr std::uintptr_t second_uploaded_address = 0x2000;
    hk_moe::host_abi::patch_symmetric_heap_descriptor(
        std::span<descriptor_word>(words),
        hk_moe::host_abi::device_visible_descriptor_address(
            second_uploaded_address));
    const auto second_checked = hk_moe::host_abi::validate_descriptor(
        std::span<const descriptor_word>(words));
    assert(second_checked.value() == second_uploaded_address);
    for (std::size_t index = 0; index < donor_words.size(); ++index) {
        assert(words[index] == donor_words[index]);
    }

    expect_invalid_argument([&] {
        (void)hk_gemm_rs::host_abi::snapshot_allocation_descriptor(
            view, nullptr);
    });
    expect_invalid_argument([&] {
        const auto before_heap = reinterpret_cast<const void*>(
            view.get_heap_base(view.cur_rank()) - 1u);
        (void)hk_gemm_rs::host_abi::snapshot_allocation_descriptor(
            view, before_heap);
    });
    expect_invalid_argument([&] {
        auto invalid = make_view();
        invalid.heaps[5] = 0;
        (void)hk_moe::host_abi::snapshot_heap_descriptor(invalid);
    });
    expect_invalid_argument([&] {
        auto invalid = make_view();
        invalid.heaps[2] = 0;
        (void)hk_gemm_rs::host_abi::snapshot_allocation_descriptor(
            invalid, local_c);
    });
    expect_invalid_argument([&] {
        (void)hk_moe::host_abi::snapshot_heap_descriptor(make_view(3, 7));
    });
    expect_invalid_argument([] {
        (void)hk_moe::host_abi::device_visible_descriptor_address(0);
    });
    expect_invalid_argument([] {
        (void)hk_moe::host_abi::device_visible_descriptor_address(0x1001);
    });
    expect_invalid_argument([&] {
        std::vector<descriptor_word> short_words(
            hk_moe::host_abi::descriptor_words - 1);
        hk_moe::host_abi::patch_symmetric_heap_descriptor(
            std::span<descriptor_word>(short_words),
            hk_moe::host_abi::device_visible_descriptor_address(0x1000));
    });
    expect_invalid_argument([&] {
        auto extended_words = words;
        hk_moe::host_abi::append_symmetric_heap_descriptor(
            extended_words,
            hk_moe::host_abi::device_visible_descriptor_address(0x1000));
    });
    expect_invalid_argument([&] {
        auto missing_slot = donor_words;
        (void)hk_moe::host_abi::validate_descriptor(
            std::span<const descriptor_word>(missing_slot));
    });
    expect_invalid_argument([&] {
        std::vector<descriptor_word> null_slot(
            hk_moe::host_abi::descriptor_words, 0);
        (void)hk_moe::host_abi::validate_descriptor(
            std::span<const descriptor_word>(null_slot));
    });
    expect_invalid_argument([&] {
        auto misaligned_slot = words;
        misaligned_slot[hk_moe::host_abi::symmetric_heap_descriptor_slot] =
            static_cast<descriptor_word>(0x1001);
        (void)hk_moe::host_abi::validate_descriptor(
            std::span<const descriptor_word>(misaligned_slot));
    });

    return 0;
}

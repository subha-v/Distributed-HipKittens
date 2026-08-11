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

    // ---- MPS descriptor extension (slots 56..62, length 63) ----------------
    {
        static_assert(hk_moe::host_abi::mps_descriptor_words == 63);
        static_assert(hk_moe::host_abi::mps_slot_queue == 56 &&
                      hk_moe::host_abi::mps_slot_config == 62);
        // Sizing formulas (PADMAX=263136, MAXTOK=4096, T_ext=32768 campaign).
        static_assert(hk_moe::host_abi::mps_queue_bytes(263136) ==
                      131568 * 4);
        static_assert(hk_moe::host_abi::mps_arrivals_bytes(32768) ==
                      32768 * 16 * 4);
        static_assert(hk_moe::host_abi::mps_pushed_bytes(32768) == 32768 * 4);
        static_assert(hk_moe::host_abi::mps_claim_bytes(32768) == 32768 * 4);
        static_assert(hk_moe::host_abi::mps_state_bytes() == 96);
        static_assert(hk_moe::host_abi::mps_slots_bytes(4096) ==
                      8 * 4096 * 7168 * 2);
        // Config round trip over the campaign sweep space.
        const std::uint64_t cfg = hk_moe::mps::encode_config(
            8, 2, 2, 16, false, true);
        const auto decoded = hk_moe::mps::decode_config(cfg);
        assert(decoded.reserved_comm_ctas == 8 && decoded.group_slices == 2 &&
               decoded.mode == 2 && decoded.flush_rows == 16 &&
               !decoded.pull_fallback && decoded.timestamps);
        assert(hk_moe::mps::config_is_valid(decoded));
        assert(!hk_moe::mps::config_is_valid(
            hk_moe::mps::decode_config(
                hk_moe::mps::encode_config(0, 2, 2, 16, false, false))));
        assert(!hk_moe::mps::config_is_valid(
            hk_moe::mps::decode_config(3)));

        hk_moe::host_abi::mps_buffer_binding binding;
        binding.queue = 0x10000;
        binding.arrivals = 0x20000;
        binding.pushed = 0x30000;
        binding.claim = 0x40000;
        binding.state = 0x50000;
        binding.slots = 0x60000;
        binding.config = cfg;
        hk_moe::host_abi::validate_mps_binding(binding);

        // Append to the frozen 55-word donor vector -> exact 63 words.
        std::vector<descriptor_word> mps_words(
            hk_moe::host_abi::donor_descriptor_words);
        for (std::size_t index = 0; index < mps_words.size(); ++index) {
            mps_words[index] = static_cast<descriptor_word>(index + 1);
        }
        hk_moe::host_abi::append_mps_descriptor(
            mps_words,
            hk_moe::host_abi::device_visible_descriptor_address(0x1000),
            binding);
        assert(mps_words.size() == hk_moe::host_abi::mps_descriptor_words);
        for (std::size_t index = 0;
             index < hk_moe::host_abi::donor_descriptor_words; ++index) {
            assert(mps_words[index] ==
                   static_cast<descriptor_word>(index + 1));
        }
        const auto checked = hk_moe::host_abi::validate_mps_descriptor(
            std::span<const descriptor_word>(mps_words));
        assert(checked.value() == 0x1000);
        assert(mps_words[hk_moe::host_abi::mps_slot_queue] ==
               static_cast<descriptor_word>(0x10000));
        assert(mps_words[hk_moe::host_abi::mps_slot_config] ==
               std::bit_cast<descriptor_word>(cfg));

        // Rejections: length, zero, alignment, invalid config.
        expect_invalid_argument([&] {
            hk_moe::host_abi::append_mps_descriptor(
                mps_words,  // already 63 words: wrong input length
                hk_moe::host_abi::device_visible_descriptor_address(0x1000),
                binding);
        });
        expect_invalid_argument([&] {
            auto bad = binding;
            bad.slots = 0x60004;  // slots need 16-byte alignment
            hk_moe::host_abi::validate_mps_binding(bad);
        });
        expect_invalid_argument([&] {
            auto bad = binding;
            bad.state = 0x50004;  // state carries u64 cells
            hk_moe::host_abi::validate_mps_binding(bad);
        });
        expect_invalid_argument([&] {
            auto bad = binding;
            bad.queue = 0;
            hk_moe::host_abi::validate_mps_binding(bad);
        });
        expect_invalid_argument([&] {
            auto bad = binding;
            bad.config = hk_moe::mps::encode_config(0, 2, 2, 16, false, false);
            hk_moe::host_abi::validate_mps_binding(bad);
        });
    }

    return 0;
}

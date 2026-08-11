// Host-only ABI/config checks for the MI300X/gfx942 GEMM-RS megakernel.
// Exercises the exact constants header, shape resolution, allocation sizing,
// descriptor snapshots (reused gfx950 PODs), and failure modes of
// gemm_rs_mi300x_host_abi.hpp under strict C++20 with the shared IRIS stub.

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "../../../../distributed-kernels/gemm_rs/gemm_rs_mi300x_host_abi.hpp"

namespace {

namespace m3h = hk_gemm_rs_mi300x::host_abi;
namespace m3 = hk_gemm_rs_mi300x;

struct alignas(64) rank_heap {
    std::array<std::byte, 8192> bytes{};
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

void check_scored_rows() {
    // The six scored shapes from the evaluator contract.
    const int ms[6] = {64, 512, 2048, 4096, 8192, 8192};
    const int ns[6] = {7168, 4096, 2880, 4096, 4096, 8192};
    const int ks[6] = {18432, 12288, 2880, 4096, 14336, 29568};
    const bool bias[6] = {false, true, true, false, true, false};
    const int bms[6] = {32, 64, 128, 256, 256, 256};
    const int bns[6] = {256, 64, 256, 256, 256, 256};
    const int bks[6] = {32, 64, 32, 32, 32, 32};
    const int nreds[6] = {32, 48, 48, 48, 32, 8};
    for (int i = 0; i < 6; ++i) {
        const auto plan = m3h::resolve_shape(ms[i], ns[i], ks[i], bias[i]);
        assert(plan.key.k_local == ks[i] / 8);                    // gate 4
        assert(plan.cfg.bm == bms[i] && plan.cfg.bn == bns[i]);
        assert(plan.cfg.bk == bks[i] && plan.cfg.num_reducer_ctas == nreds[i]);
        assert(plan.cfg.config_row == i + 1);
        assert(plan.num_gemm_ctas + plan.cfg.num_reducer_ctas == 304);
        assert(plan.num_gemm_ctas >= 256);
        assert(plan.eb >= 1 && plan.slice_rows % plan.eb == 0);   // gate 5
        assert(plan.cfg.bm % plan.eb == 0);
        assert(plan.lrow_count == plan.slice_rows / plan.eb);
        assert(plan.lrow_count <= 32 && plan.col_count <= 128);
        assert(plan.gemm_tiles == (ms[i] / bms[i]) * plan.col_count);
        assert(plan.red_tiles == plan.lrow_count * plan.col_count);
        assert(plan.signal_words_total <= m3::SIGNAL_U32_CAP);    // gate 8
        assert(plan.packet_fast_path == ((ns[i] % 8) == 0));      // gate 9
        // LDS per config must fit 64 KiB with a 32-row staging window.
        const int lds = 2 * (bms[i] + bns[i]) * bks[i] * 2;
        assert(lds <= 65536);
        assert(32 * bns[i] * 2 <= lds);
    }
}

void check_specific_geometry() {
    {   // shape 1: BM=32 bands of EB=8 straddle four owners per tile
        const auto plan = m3h::resolve_shape(64, 7168, 18432, false);
        assert(plan.slice_rows == 8 && plan.eb == 8 && plan.lrow_count == 1);
        assert(plan.col_count == 28 && plan.gemm_tiles == 56);
        assert(plan.red_tiles == 28 && plan.num_gemm_ctas == 272);
        assert(!(plan.key.has_bias));
    }
    {   // shape 3: N tail 2880 % 256 == 64; K tail 360 % 32 == 8
        const auto plan = m3h::resolve_shape(2048, 2880, 2880, true);
        assert(!plan.even_n && !plan.even_k);
        assert(plan.col_count == 12 && plan.lrow_count == 2);
        assert(plan.slice_rows == 256 && plan.eb == 128);
    }
    {   // shape 6: K tail 3696 % 32 == 16
        const auto plan = m3h::resolve_shape(8192, 8192, 29568, false);
        assert(!plan.even_k && plan.even_n);
        assert(plan.lrow_count == 4 && plan.col_count == 32);
        assert(plan.gemm_tiles == 1024 && plan.red_tiles == 128);
    }
}

void check_generic_path() {
    {   // public-style generic shape (m=64, n=2880)
        const auto plan = m3h::resolve_shape(64, 2880, 2880, true);
        assert(plan.cfg.config_row == 0);
        assert(plan.cfg.bm == 32 && plan.cfg.bn == 64 && plan.cfg.bk == 64);
        assert(plan.eb == m3::gcd_int(32, 8));
        assert(plan.num_gemm_ctas == 304 - 24);
    }
    {   // large generic shape exercises the maximum signal geometry
        const auto plan = m3h::resolve_shape(8192, 8192, 28672, false);
        assert(plan.key.k_local == 3584);
        assert(plan.lrow_count == 32 && plan.col_count == 128);
        assert(plan.signal_words_total ==
               64 + 2 * 8u * 32u * 128u);
    }
    // Failure modes: the launch resolution must fail closed.
    expect_invalid_argument([] { (void)m3h::resolve_shape(63, 4096, 4096, false); });
    expect_invalid_argument([] { (void)m3h::resolve_shape(64, 0, 4096, false); });
    expect_invalid_argument([] { (void)m3h::resolve_shape(64, 4096, 4095, false); });
    expect_invalid_argument([] { (void)m3h::resolve_shape(40, 4096, 4096, false); });
}

void check_key_and_descriptors() {
    const auto plan = m3h::resolve_shape(8192, 4096, 14336, true);
    const auto key = m3h::make_cache_key(plan);
    assert(key.m == 8192 && key.n == 4096 && key.k_local == 1792);
    assert(key.has_bias && key.config_row == 5 && key.even_k && key.even_n);
    (void)key.arch; (void)key.dtype;

    // Descriptor snapshots reuse the unchanged gfx950 PODs (72 bytes).
    constexpr std::size_t c_offset = 0x180;
    constexpr std::size_t signal_offset = 0x580;
    const auto view = make_view();
    const void* const local_c = heaps[3].bytes.data() + c_offset;
    const void* const local_signals = heaps[3].bytes.data() + signal_offset;
    const auto desc = m3h::snapshot_allocation_descriptors(
        view, local_c, local_signals);
    static_assert(sizeof(desc.c_heap) == 72);
    assert(desc.c_heap.local_allocation_base == local_c);
    assert(desc.c_heap.bases.rank6 == heaps[6].bytes.data() + c_offset);
    assert(desc.signals.bases.rank1 == heaps[1].bytes.data() + signal_offset);

    expect_invalid_argument([&] {
        (void)m3h::snapshot_allocation_descriptors(view, nullptr,
                                                   local_signals);
    });
    expect_invalid_argument([&] {
        auto invalid = make_view();
        invalid.heaps[4] = 0;
        (void)m3h::snapshot_allocation_descriptors(invalid, local_c,
                                                   local_signals);
    });
}

void check_layout_and_epoch_math() {
    // Dependency-key helpers: collision-freeness in the small, per gate 7.
    for (int src = 0; src < 8; ++src) {
        for (int lr = 0; lr < 4; ++lr) {
            for (int c = 0; c < 32; ++c) {
                assert(m3::ready_idx(src, lr, c, 4, 32) ==
                       (src * 4 + lr) * 32 + c);
                assert(m3::credit_idx(src, lr, c, 4, 32) ==
                       (src * 4 + lr) * 32 + c);
            }
        }
    }
    assert(m3::ready_idx(7, 3, 31, 4, 32) + 1 == 8 * 4 * 32);
    assert(m3::EP_GEMM_U32 == 0 && m3::EP_RED_U32 == 304 &&
           m3::EP_U32 == 608);
    assert(m3::ERR_PRODUCER_CREDIT != m3::ERR_REDUCER_READY);
}

} // namespace

int main() {
    static_assert(m3::WORLD_SIZE == 8);                // gate 1
    static_assert(m3::CU_COUNT == 304);
    static_assert(m3::CTA_THREADS == 512);
    check_scored_rows();
    check_specific_geometry();
    check_generic_path();
    check_key_and_descriptors();
    check_layout_and_epoch_math();
    return 0;
}

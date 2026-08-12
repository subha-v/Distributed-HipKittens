#pragma once

// Host-only shape configuration and launch plumbing for the MI300X/gfx942
// GEMM -> ReduceScatter persistent megakernel.
//
// This header owns: the scored-shape table, the generic fallback rule, every
// derived count (tile maps, signal geometry, allocation sizes), and the
// compile/launch cache key. It performs no allocation, upload, or launch;
// those remain caller-owned. Buffer snapshots reuse the unchanged gfx950
// helpers and the unchanged 72-byte descriptors.

#include "../common/iris_adapter.hpp"
#include "gemm_rs_host_abi.hpp"          // hk_gemm_rs 72 B descriptors + snapshots
#include "gemm_rs_mi300x_constants.cuh"  // constants shared with device code

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace hk_gemm_rs_mi300x::host_abi {

inline constexpr int world_size = WORLD_SIZE;   // compile-time world, gate 1
inline constexpr int cu_count   = CU_COUNT;
inline constexpr int cta_threads = CTA_THREADS;

// ---------------------------------------------------------------------------
// Shape table. Key = (M, N, K_LOCAL = K_global / 8, has_bias): the evaluator
// shards K evenly (verified against the official evaluator; gate 4).
// BM/BN/BK are the frozen gfx942 rank-1 tile choices with BM | (M/8) checked
// at construction and again statically.
//
// Reducer-CTA counts were originally carried over verbatim from RadeonFlow's
// submitted scored values (32/48/48/48/32/8); exp_02 replaced them with a
// uniform NR=32, worth 8.3% on the largest shape. exp_13 re-swept the axis
// {4..80} after the retile, the mainloop overlap, the WGM tile order and the
// release grouping had all moved the balance, and two rows moved.
//
// Both roles are strided persistent loops, so what the split buys is not CTAs
// but ROUNDS: the producer critical path is ceil(gemm_tiles / (304 - NR)) tile
// waves and the reducer's is ceil(red_tiles / NR). Waves are flat in NR over a
// wide plateau and then step, which makes the optimum the largest NR that does
// not add a producer wave:
//   row 1 (224 tiles, 112 red tiles): 1 wave for all NR <= 80, so producers are
//     free above 32 while reduce rounds fall 4 -> 2 at NR=56 (112/56 = exactly
//     2 each). Measured -15.4%. exp_04a is what created this: retiling to
//     32/64/64 took red_tiles 28 -> 112 and quadrupled the reduce rounds at
//     NR=32, and the split was never re-swept.
//   row 6 (1024 tiles, 128 red tiles): NG=256 is exactly 4 balanced waves and
//     is the LAST split with 4; reduce rounds fall 4 -> 3. Measured -4.9%.
//     NR=56 costs a fifth wave and is 11% worse.
// Rows 2-5 are flat: every candidate is inside the null-arm floor of the
// measuring instrument, so they keep 32. Do not read the flatness as "the axis
// does not matter" -- it means those rows sit mid-plateau in both roles.
// ---------------------------------------------------------------------------
struct shape_key {
    int m, n, k_local;
    bool has_bias;
};
struct shape_config {
    int bm, bn, bk;
    int num_reducer_ctas;             // in {8, 16, 24, 32, 40, 48, ...}
    int config_row;                   // unique row id, part of the cache key
};

struct shape_entry { shape_key key; shape_config cfg; };

// exp_14 (E4b) re-swept BM/BN/BK with the same round-counting model, against an
// exhaustive enumeration of the legal space (see experiments/exp_14_tile_waves).
// The correction that made the sweep productive: `waves` alone is not a cost,
// because moving the tile also moves what ONE tile body costs. Three columns
// have to be minimized together, and they disagree:
//
//   waves * BM*BN   MFMA-bound cost. Equals (M*N/NG)/avg_fill, i.e. this IS the
//                   fill criterion, expressed as a time.
//   waves * k_iters OVERHEAD-bound cost. Every k-iteration pays a full
//                   __syncthreads(), a vmcnt(0), an lgkmcnt(0) and an LDS round
//                   trip whatever the tile size, and shape 6 measures ~4700
//                   cycles/iteration against ~2050 of MFMA -- so this is the
//                   LARGER half, and raising BK is the only knob that cuts it at
//                   constant MFMA work and constant total bytes.
//   1/BM + 1/BN     global bytes, which is what buying fill with a smaller tile
//                   actually costs. Measured, not assumed: on row 3 an arm with
//                   identical waves*BM*BN AND identical waves*k_iters but 1.78x
//                   the traffic was 6.2% SLOWER in 15 of 15 draws.
//
//   row 1: 32/64/64 -> 32/64/128. Same 224 tiles, same 1 wave, same 90% fill,
//     same 112 columns (so the NR=56 reduce optimum is untouched); k_iters
//     36 -> 18. +1.7% best, p=0.0013 by rank test over 15 allocation draws --
//     real but under this row's own null-arm floor, which is why it needed 15.
//   row 2: 64/64/64 -> 64/128/64. 512 tiles/2 waves -> 256 tiles/1 wave at
//     identical MFMA work, k_iters unchanged at 24 so waves*k_iters halves,
//     traffic 0.75x, and red_tiles 64 -> 32 so reduce rounds fall 2 -> 1.
//     +32.8%, disjoint over 15 draws. The largest single win of the project.
//   row 3: 128/256/32 -> 128/192/32. 2880/192 = 15 EXACTLY, so this also
//     removes 6.7% of ragged-N padding and an out-of-bounds read: at BN=256 the
//     last B tile's load_issue reads rows 2880..3071 of a 2880-row operand.
//     192 tiles/71% fill -> 240/88%. +6.8%, disjoint over 15 draws.
//   rows 4, 5, 6: UNCHANGED, and that is a result rather than an omission. Each
//     is the argmin of the composite over all 38 legal points for its shape,
//     and row 6 sits exactly at the waves*BM*BN floor (1024 tiles = 4 x 256 NG,
//     100% fill). Row 5's 88% last-wave fill is NOT recoverable: one wave would
//     need BM*BN >= 123362, i.e. 241 accumulator VGPRs. All three are also
//     pinned against the LDS cap -- (BM+BN)*BK = 512*32 = 16384 is exactly the
//     limit -- so BK cannot rise on the three largest rows and their
//     waves*k_iters, the larger half of the mainloop, is untouchable from this
//     table. That is an E1 (double-buffer) problem, not a tile-table one.
inline constexpr std::array<shape_entry, 6> scored_shapes{{
    {{  64, 7168, 2304, false}, { 32,  64, 128, 56, 1}},
    {{ 512, 4096, 1536,  true}, { 64, 128,  64, 32, 2}},
    {{2048, 2880,  360,  true}, {128, 192,  32, 32, 3}},
    {{4096, 4096,  512, false}, {256, 256,  32, 32, 4}},
    {{8192, 4096, 1792,  true}, {256, 256,  32, 32, 5}},
    {{8192, 8192, 3696, false}, {256, 256,  32, 48, 6}},
}};

// Generic fallback row (correctness path for every other evaluator-legal
// shape): gcd-derived emit bands; gate 2-14 checks below apply identically.
inline constexpr shape_config generic_config{32, 64, 64, 24, 0};
inline constexpr int generic_m_alignment = 32;  // guard for ragged A reads (s6)

// ---------------------------------------------------------------------------
// Resolved launch plan: every derived count, precomputed and validated.
// ---------------------------------------------------------------------------
struct launch_config {
    shape_key key;
    shape_config cfg;
    int slice_rows;        // M / 8
    int eb;                // emit band height = gcd(BM, M/8); EB | M/8, EB | BM
    int lrow_count;        // slice_rows / EB
    int col_count;         // ceil(N / BN)
    int gemm_tiles;        // (M / BM) * col_count
    int red_tiles;         // lrow_count * col_count
    int num_gemm_ctas;     // CU_COUNT - num_reducer_ctas
    std::size_t c_heap_bytes;          // 8 slots * slice_rows * N * sizeof(bf16)
    std::size_t ready_words;           // 8 * lrow_count * col_count
    std::size_t credit_words;          // 8 * lrow_count * col_count
    std::size_t signal_words_total;    // guard + ready + credit
    bool even_k;           // K_local % BK == 0
    bool even_n;           // N % BN == 0
    bool packet_fast_path; // (N % 8) == 0: 16 B packet rows align everywhere
};

// Deliberately does NOT compare has_bias, and that is worth explaining because
// the obvious reading of the table says it should.
//
// The evaluator's case parser is `try: val = int(val) except ValueError: pass`,
// so `has_bias: False` survives as the *string* "False", which is truthy. Every
// graded shape therefore arrives with has_bias == true, whatever the case file
// says. Keying on it meant the tuned rows for the three shapes declared
// has_bias=false (1, 4 and 6) could never match under the real evaluator and
// silently fell through to generic_config{32,64,64,24}: shape 4 went 256 ->
// 8192 tiles and shape 6 went 1024 -> 32768, i.e. 32x the tiles at a third of
// the tile size. That is the whole of the unexplained 2.5-3.3x inflation those
// two shapes showed under eval.py while reproducing fine in our own harness --
// our harness passed has_bias honestly, so it never triggered the fallback.
//
// Ignoring the field is correct, not a workaround: BM/BN/BK and the reducer
// split are pure geometry, and bias is applied from a runtime pointer in the
// epilogue rather than by any compile-time branch (it is not a template
// parameter of the kernel). So one row serves both cases exactly.
[[nodiscard]] inline const shape_config* find_scored_config(
        const shape_key& key) {
    for (const auto& entry : scored_shapes) {
        if (entry.key.m == key.m && entry.key.n == key.n &&
            entry.key.k_local == key.k_local) {
            return &entry.cfg;
        }
    }
    return nullptr;
}

[[nodiscard]] inline launch_config resolve_shape(
        int m, int n, int k_global, bool has_bias) {
    if (m <= 0 || n <= 0 || k_global <= 0) {
        throw std::invalid_argument("GEMM-RS mi300x: non-positive dimension");
    }
    if (m % world_size != 0) {                       // gate 2
        throw std::invalid_argument("GEMM-RS mi300x: M % world != 0");
    }
    if (k_global % world_size != 0) {                // gate 4
        throw std::invalid_argument("GEMM-RS mi300x: K % world != 0");
    }
    const int k_local = k_global / world_size;

    const shape_key key{m, n, k_local, has_bias};
    const shape_config* picked = find_scored_config(key);
    shape_config cfg = picked ? *picked : generic_config;
    if (picked == nullptr && m % generic_m_alignment != 0) {
        throw std::invalid_argument(
            "GEMM-RS mi300x: generic path requires M % 32 == 0");
    }

    const int slice_rows = m / world_size;
    launch_config plan{};
    plan.key = key;
    plan.cfg = cfg;
    plan.slice_rows = slice_rows;
    plan.eb = gcd_int(cfg.bm, slice_rows);
    if (plan.eb <= 0 || slice_rows % plan.eb != 0 || cfg.bm % plan.eb != 0) {
        throw std::invalid_argument("GEMM-RS mi300x: emit band does not divide");
    }                                                            // gate 5
    if (cfg.bm % (WARPS_M * 16) != 0 || cfg.bn % (WARPS_N * 16) != 0) {
        throw std::invalid_argument("GEMM-RS mi300x: warp fragment mismatch");
    }
    plan.lrow_count = slice_rows / plan.eb;
    plan.col_count = (n + cfg.bn - 1) / cfg.bn;
    if (m % cfg.bm != 0) {
        throw std::invalid_argument("GEMM-RS mi300x: BM must divide M");
    }
    plan.gemm_tiles = (m / cfg.bm) * plan.col_count;
    plan.red_tiles = plan.lrow_count * plan.col_count;
    if (plan.lrow_count > LROW_MAX || plan.col_count > COL_MAX) {
        throw std::invalid_argument(
            "GEMM-RS mi300x: signal geometry exceeds LROW_MAX/COL_MAX");
    }
    plan.num_gemm_ctas = cu_count - cfg.num_reducer_ctas;
    if (plan.num_gemm_ctas <= 0 || plan.num_gemm_ctas > cu_count) {
        throw std::invalid_argument("GEMM-RS mi300x: degenerate CTA split");
    }
    plan.c_heap_bytes =
        static_cast<std::size_t>(world_size) *
        static_cast<std::size_t>(slice_rows) *
        static_cast<std::size_t>(n) * sizeof(std::uint16_t);
    plan.ready_words = static_cast<std::size_t>(world_size) *
                       static_cast<std::size_t>(plan.lrow_count) *
                       static_cast<std::size_t>(plan.col_count);
    plan.credit_words = plan.ready_words;
    plan.signal_words_total =
        SIGNAL_GUARD_U32 + plan.ready_words + plan.credit_words;
    if (plan.signal_words_total > SIGNAL_U32_CAP) {              // gate 8
        throw std::invalid_argument("GEMM-RS mi300x: signal heap too small");
    }
    plan.even_k = (k_local % cfg.bk) == 0;
    plan.even_n = (n % cfg.bn) == 0;
    plan.packet_fast_path = (n % 8) == 0;                        // gate 9
    return plan;
}

// ---------------------------------------------------------------------------
// Compile/launch cache key (gate 20): every codegen-relevant variable is in
// the key. arch/dtype/bias/tail are explicit; the config row identities the
// exact tile instantiation; eb/lrow/col strides follow from the key.
// ---------------------------------------------------------------------------
struct launch_cache_key {
    int m, n, k_local;
    bool has_bias;
    int config_row;
    bool even_k, even_n;
    const char* arch;            // "gfx942" for this port's binding
    const char* dtype;           // "bf16"
};
[[nodiscard]] inline launch_cache_key make_cache_key(
        const launch_config& plan) {
    return {plan.key.m, plan.key.n, plan.key.k_local, plan.key.has_bias,
            plan.cfg.config_row, plan.even_k, plan.even_n, "gfx942", "bf16"};
}

// ---------------------------------------------------------------------------
// Descriptor plumbing (unchanged gfx950 helpers; exact 72-byte PODs).
// ---------------------------------------------------------------------------
using hk_gemm_rs::host_abi::snapshot_allocation_descriptors;
using hk_gemm_rs::host_abi::allocation_descriptors;

// Zero-epoch setup contract: sig and ep_cell are zeroed ONCE before the first
// launch. There is no per-call host reset (gate 17).
inline constexpr std::size_t epoch_words_per_rank = EP_U32;

} // namespace hk_gemm_rs_mi300x::host_abi

static_assert(hk_gemm_rs_mi300x::host_abi::world_size == 8,
              "GEMM-RS mi300x fixes world size 8");                 // gate 1

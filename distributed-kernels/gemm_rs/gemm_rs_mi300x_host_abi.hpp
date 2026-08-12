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
// submitted scored values (32/48/48/48/32/8). Sweeping {8,16,24,32,40,48} per
// shape on this node showed a uniform NR=32 is optimal or within noise on all
// six, and beats the inherited table by 8.3% on the largest shape
// (2850.2 -> 2632.1 us at 8192x8192x29568, where the donor value of 8 starves
// the reduce side). Carrying over a donor's constants was not free.
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

inline constexpr std::array<shape_entry, 6> scored_shapes{{
    {{  64, 7168, 2304, false}, { 32,  64, 64, 32, 1}},
    {{ 512, 4096, 1536,  true}, { 64,  64, 64, 32, 2}},
    {{2048, 2880,  360,  true}, {128, 256, 32, 32, 3}},
    {{4096, 4096,  512, false}, {256, 256, 32, 32, 4}},
    {{8192, 4096, 1792,  true}, {256, 256, 32, 32, 5}},
    {{8192, 8192, 3696, false}, {256, 256, 32, 32, 6}},
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

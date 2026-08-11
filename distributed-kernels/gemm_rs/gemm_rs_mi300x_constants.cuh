#pragma once

// Pure constants and dependency-key layout for the MI300X/gfx942 GEMM-RS
// megakernel. This header is compilable as strict host C++20 and device HIP:
// it intentionally has no kittens, IRIS, or HIP-API dependency, so the host
// ABI tests exercise the exact geometry the kernel uses.

namespace hk_gemm_rs_mi300x {

// ---------------------------------------------------------------------------
// Compile-time geometry. One kernel, one CTA class, exact 304-CTA grid.
// ---------------------------------------------------------------------------
inline constexpr int WORLD_SIZE       = 8;     // gate 1
inline constexpr int CU_COUNT         = 304;
inline constexpr int CTA_THREADS      = 512;
inline constexpr int NUM_WARPS        = CTA_THREADS / 64;
inline constexpr int WARPS_M          = 2;
inline constexpr int WARPS_N          = 4;
static_assert(WARPS_M * WARPS_N == NUM_WARPS);

// Signal-geometry caps: lrow_count = (M/8)/EB with EB = gcd(BM, M/8); the
// generic path (BM=32) at M=8192 yields 32. col_count = ceil(N/BN) with
// BN >= 64 and N <= 8192 yields at most 128. Host sizing validates the fit.
inline constexpr int LROW_MAX         = 32;
inline constexpr int COL_MAX          = 128;
inline constexpr int SIGNAL_GUARD_U32 = 64;
inline constexpr int SIGNAL_U32_CAP   =
    SIGNAL_GUARD_U32 + 2 * WORLD_SIZE * LROW_MAX * COL_MAX;   // 65600
static_assert(SIGNAL_U32_CAP == 65600);

// Local per-CTA device-derived epoch cells: producer window then reducer
// window; disjoint so roles never collide (gate 7).
inline constexpr int EP_GEMM_U32 = 0;
inline constexpr int EP_RED_U32  = CU_COUNT;
inline constexpr int EP_U32      = 2 * CU_COUNT;              // 608

// Fail-closed error bits (LOCAL int).
inline constexpr int ERR_PRODUCER_CREDIT = 1 << 25;
inline constexpr int ERR_REDUCER_READY   = 1 << 26;

// Negative-control flag bits (a separate module; never in the production
// binding). Production kernels are compiled with
// HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS == 0 and these branches vanish.
inline constexpr unsigned CTRL_DROP_PUBLICATION = 1u << 0;
inline constexpr unsigned CTRL_REROUTE_SLOT     = 1u << 1;
inline constexpr unsigned CTRL_DROP_CREDIT      = 1u << 2;

// ---------------------------------------------------------------------------
// Dependency-key layout helpers. constexpr => valid in both host and device
// code; identical formulas are restated in the Python simulation.
// ---------------------------------------------------------------------------
constexpr int owner_of_row(int row, int slice_rows) {
    return row / slice_rows;
}
constexpr int ready_idx(int src, int lrow, int col, int lrows, int cols) {
    return (src * lrows + lrow) * cols + col;
}
constexpr int credit_idx(int owner, int lrow, int col, int lrows, int cols) {
    return (owner * lrows + lrow) * cols + col;
}
constexpr int gcd_int(int a, int b) {
    return b == 0 ? a : gcd_int(b, a % b);
}

} // namespace hk_gemm_rs_mi300x

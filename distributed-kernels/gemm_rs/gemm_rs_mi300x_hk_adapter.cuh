#pragma once

// HipKittens communication boundary for the MI300X/gfx942 GEMM -> ReduceScatter
// persistent megakernel. Tile ownership, signal indices, epochs, and reduction
// order remain operator policy (MI300X_DESIGN.md); this file maps only peer
// projection, packet movement, ordering, and directed slot lifetime onto
// kittens::pgl / kittens::distributed, and adds the MI300X-specific emit and
// reduction tile helpers. Shared geometry constants live in
// gemm_rs_mi300x_constants.cuh.
//
// No IRIS headers here: this compilation path stays device-buildable. Host
// plumbing (shape table, sizing, snapshots) lives in
// gemm_rs_mi300x_host_abi.hpp.

#include "kittens.cuh"
#include "gemm_rs_hk_adapter.cuh"
#include "gemm_rs_mi300x_constants.cuh"

#include <cstddef>
#include <cstdint>

namespace hk_gemm_rs_mi300x {

using bf16 = kittens::bf16;

// ---------------------------------------------------------------------------
// Protocol helpers: thin spellings of the gfx950-validated adapter so the
// MI300X operator never touches raw peer-pointer arrays.
// ---------------------------------------------------------------------------

__device__ __forceinline__ std::uint32_t next_epoch_workgroup(
        std::uint32_t* cell) {
    return hk_gemm_rs::next_epoch_workgroup(cell);
}

__device__ __forceinline__ void release_payload_system() {
    hk_gemm_rs::release_payload_system();
}

__device__ __forceinline__ void acquire_payload_system() {
    hk_gemm_rs::acquire_payload_system();
}

__device__ __forceinline__ void publish_band_epoch(
        std::uint32_t* sig_local,
        const hk_gemm_rs::symmetric_descriptor* sig_peers,
        int dest, int me, int signal_index, std::uint32_t epoch) {
    hk_gemm_rs::publish_tile_epoch(sig_local, sig_peers, dest, me,
                                   signal_index, epoch);
}

__device__ __forceinline__ bool wait_band_epoch(
        const std::uint32_t* local_ready, std::uint32_t epoch,
        std::uint64_t spin_limit, int* errp, int err_bit) {
    return hk_gemm_rs::wait_tile_epoch(local_ready, epoch, spin_limit,
                                       errp, err_bit);
}

__device__ __forceinline__ bool wait_reuse_credit(
        const std::uint32_t* local_credit, std::uint32_t next_epoch,
        std::uint64_t spin_limit, int* errp, int err_bit) {
    return hk_gemm_rs::wait_reuse_credit(local_credit, next_epoch, spin_limit,
                                         errp, err_bit);
}

__device__ __forceinline__ void finish_tile_consumption() {
    hk_gemm_rs::finish_tile_consumption();
}

__device__ __forceinline__ void publish_reuse_credit(
        std::uint32_t* sig_local,
        const hk_gemm_rs::symmetric_descriptor* sig_peers,
        int source, int me, int credit_index, std::uint32_t epoch) {
    hk_gemm_rs::publish_reuse_credit(sig_local, sig_peers, source, me,
                                     credit_index, epoch);
}

__device__ __forceinline__ bool error_bit_set(const int* errp, int bit) {
    return hk_gemm_rs::error_bit_set(errp, bit);
}

// ---------------------------------------------------------------------------
// Global -> shared, split into an ISSUE half and a COMMIT half (exp_03/E1(b)).
//
// kittens::load(ST&, const GL&, COORD) fuses four things into one call:
// global_load_dwordx4 into a per-thread float4 staging buffer, s_waitcnt
// vmcnt(0), the ds_write_b64 pair, and s_waitcnt lgkmcnt(0). The fusion is why
// a k+1 "prefetch" cannot overlap iteration k's MFMAs: the round trip is
// drained before control leaves the call. These two helpers are that function
// cut in half at its vmcnt(0), so the caller can put work in between. Index
// arithmetic, the `row < rows` predicate, the unit_coord/src_ptr derivation and
// the dst.idx() addressing are copied unchanged from
// include/cdna3/ops/warp/memory/tile/global_to_shared.cuh (which is shared with
// other kernels and is not ours to edit).
//
// One deliberate deviation, provably behaviour-identical: the donor iterates
// j over `small_calls` (16) inside a `big_calls` loop, relying on dead-code
// elimination of the slots whose `row < rows` predicate is unsatisfiable. Here
// the loop bound is `total_calls`, which is exactly the number of slots any
// lane can satisfy the predicate for (row < rows <=> j*N_THREADS + laneid <
// rows*memcpy_per_row <=> j < total_calls), so no slot that could load is
// dropped and the staging buffer's register footprint no longer depends on the
// optimizer folding threadIdx range information. big_calls == 1 is asserted
// rather than assumed, because a split buffer cannot express the donor's reuse
// of buf[] across successive big calls.
// ---------------------------------------------------------------------------

template<kittens::ducks::st::all ST, int N_THREADS>
struct g2s_plan {
    using T = typename ST::dtype;
    static constexpr int elem_per_memcpy      = sizeof(float4) / sizeof(T);
    static constexpr int elem_per_half_memcpy = sizeof(float2) / sizeof(T);
    static constexpr int memcpy_per_row       = ST::cols / elem_per_memcpy;
    static constexpr int total_calls =
        (ST::cols * ST::rows + N_THREADS * elem_per_memcpy - 1) /
        (N_THREADS * elem_per_memcpy);
    static constexpr int small_calls = 16;
    static constexpr int big_calls   = (total_calls + small_calls - 1) / small_calls;
    static constexpr int slots       = total_calls;
    static_assert(big_calls == 1,
                  "issue/commit split needs the whole tile in one staging pass");
};

// Number of float4 staging slots the caller must provide for one tile.
template<kittens::ducks::st::all ST, int N_THREADS>
inline constexpr int g2s_slots = g2s_plan<ST, N_THREADS>::slots;

// Full-drain waits only. Counted waits make the count an invariant the
// scheduler can silently invalidate (see gemm_rs_device_tile.cpp:786-812 for a
// 30x numerical error that still passed a 2e-2 gate); the win here comes from
// where the wait sits, not from making it counted.
__device__ __forceinline__ void wait_vmcnt0() {
#ifdef BUILTINS_ONLY
    __builtin_amdgcn_s_waitcnt(0);
#else
    asm volatile("s_waitcnt vmcnt(0)");
#endif
}
__device__ __forceinline__ void wait_lgkmcnt0() {
#ifdef BUILTINS_ONLY
    __builtin_amdgcn_s_waitcnt(0);
#else
    asm volatile("s_waitcnt lgkmcnt(0)");
#endif
}

// A bare s_waitcnt is NOT enough to protect a register filled from LDS, and the
// failure is silent. The wait has no operands, so it has no data dependence on
// anything; the ds_read that fills a fragment is an asm volatile whose output
// the compiler believes is available the instant the asm ends. Volatile asm
// blocks keep their order relative to each other, but an MFMA is a plain
// intrinsic with no memory effects, so the machine scheduler is free to hoist
// it above the wait -- and it does, in exactly the instantiations where nothing
// else fragments the scheduling region. Measured on this kernel: MFMAs hoisted
// above the wait in all three K_TAIL=false instantiations (3 of 4 MFMAs per
// k-iteration in <32,64,64,false>), producing ~1e19 garbage on 14 of 17 shapes,
// while the K_TAIL=true ones happened to be clean because mask_a_k_tail split
// the block.
//
// frag_anchor gives the drain teeth: after the wait, every base tile's register
// pair is laundered through an empty asm volatile with a "+v" tie, which is a
// real def. Any consumer of that register is now ordered after it by data
// dependence rather than by scheduler goodwill. Emits no instructions. One
// operand per base tile, matching the ds_read_b64 that filled it (which writes
// data[0..1] as one 64-bit value).
template<typename RT>
__device__ __forceinline__ void frag_anchor(RT& t) {
    #pragma unroll
    for (int h = 0; h < RT::height; ++h) {
        #pragma unroll
        for (int w = 0; w < RT::width; ++w) {
            asm volatile("" : "+v"(
                *reinterpret_cast<std::uint64_t*>(&t.tiles[h][w].data[0])));
        }
    }
}

// Drain the ds_reads that filled these two fragments, then anchor both.
template<typename RTA, typename RTB>
__device__ __forceinline__ void acquire_frags(RTA& a, RTB& b) {
    wait_lgkmcnt0();
    frag_anchor(a);
    frag_anchor(b);
}

// Order every accumulator-writing MFMA BEFORE this point (exp_09 E1(c) arm A3).
//
// Needed because scheduling *hints* do not move MFMAs in this TU. Two arms
// proved it: sched_group_barrier asking for VMEM spread through the MFMA block
// changed not one instruction, and sched_barrier(0x7F6) -- every class except
// MFMA allowed to cross -- left the commit's vmcnt(0) sitting after 33 of the
// 64 MFMAs exactly as before. What DOES work on this compiler is the same
// mechanism frag_anchor uses: an empty asm volatile with a "+v" tie is a real
// def, so whatever defined that register must precede it, and being volatile it
// cannot be reordered against load_commit's volatile waits and ds_writes. That
// is a data dependence, not scheduler goodwill.
//
// One operand per base tile is sufficient: a single v_mfma defines the whole
// 4-float accumulator tile, so tying its first register pair orders that MFMA.
// Emits no instructions.
template<typename RT>
__device__ __forceinline__ void acc_anchor(RT& t) {
    #pragma unroll
    for (int h = 0; h < RT::height; ++h) {
        #pragma unroll
        for (int w = 0; w < RT::width; ++w) {
            asm volatile("" : "+v"(
                *reinterpret_cast<std::uint64_t*>(&t.tiles[h][w].data[0])));
        }
    }
}

// Issue the tile's global loads into `buf`. NO waitcnt: on return the data is
// in flight, not in registers, and `buf` must not be read until load_commit.
template<kittens::ducks::st::all ST, int N_THREADS, int axis = 2,
         bool assume_aligned = false, kittens::ducks::gl::all GL,
         kittens::ducks::coord::tile COORD = kittens::coord<ST>>
__device__ __forceinline__ void load_issue(float4* buf, const GL& src,
                                           const COORD& idx) {
    using P = g2s_plan<ST, N_THREADS>;
    const int row_stride = src.template stride<axis>();
    kittens::coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype* src_ptr = (typename GL::dtype*)&src[unit_coord];
    const int laneid = threadIdx.x % N_THREADS;

    #pragma unroll
    for (int j = 0; j < P::slots; ++j) {
        const int load_idx = j * N_THREADS + laneid;
        const int row = load_idx / P::memcpy_per_row;
        const int col = (load_idx % P::memcpy_per_row) * P::elem_per_memcpy;
        if (row < ST::rows) {
            buf[j] = kittens::load_global_vec4_async(
                (float4*)(src_ptr + (row * row_stride + col)));
        }
    }
}

// Land the issued loads and publish them to LDS: vmcnt(0), the ds_writes, then
// lgkmcnt(0) so the caller's __syncthreads() only has to order the barrier.
// (__syncthreads() alone would NOT do it: every LDS op in this tree is inside
// asm volatile, so SIInsertWaitcnts never sees an LDS event and emits no
// lgkmcnt with the s_barrier.)
template<int N_THREADS, kittens::ducks::st::all ST>
__device__ __forceinline__ void load_commit(ST& dst, const float4* buf) {
    using P = g2s_plan<ST, N_THREADS>;
    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;

    wait_vmcnt0();

    #pragma unroll
    for (int j = 0; j < P::slots; ++j) {
        const int load_idx = j * N_THREADS + laneid;
        const int row = load_idx / P::memcpy_per_row;
        const int col = (load_idx % P::memcpy_per_row) * P::elem_per_memcpy;
        if (row < dst.rows) {
            kittens::store_shared_vec(dst.idx(dst_ptr, {row, col}),
                                      {buf[j].x, buf[j].y});
            kittens::store_shared_vec(
                dst.idx(dst_ptr, {row, col + P::elem_per_half_memcpy}),
                {buf[j].z, buf[j].w});
        }
    }

    wait_lgkmcnt0();
}

// ---------------------------------------------------------------------------
// MI300X emit: FP32 accumulators are converted to bf16 through the already
// dead A/B LDS and emitted as 16-byte peer packets (exp-23 EV=1 discipline at
// EB-row band granularity). Every store is bounded by `valid_cols` (N tail).
// ---------------------------------------------------------------------------

// Stage one fragment warp-tile of bf16 values into the row-major staging
// window. Lane L of a col-layout fp32 accumulator holds, per base tile (h,w):
// col = 16*w + L%16, rows = 16*h + 4*(L/16) + {0,1,2,3}, with data[p] giving
// rows lr + 2*p + {0,1}. Consecutive lanes cover consecutive columns, so a
// half-wave writes 16 bf16 = 32 B contiguous per staging row: conflict-free.
// `row0/col0` are this warp's tile-local bases. Only rows inside the window
// `[win_row0, win_row0 + win_rows)` (tile-local) are written, at window-local
// coordinates; callers iterate windows over the emit band.
// Bias is the per-rank epilogue term of the oracle: added in FP32 before the
// tile's single RNE bf16 pack. The bias read is index-clamped so N-tail
// columns (never emitted) cannot read out of range.
template<typename RT>
__device__ __forceinline__ void stage_fragment_bf16(
        bf16* stage, int stage_stride, const RT& acc,
        const bf16* bias, int row0, int col0,
        int win_row0, int win_rows,
        int global_col_base, int bias_limit) {
    const int lane = kittens::laneid();
    const int lc   = lane & 15;
    const int lr   = (lane >> 4) << 2;
    #pragma unroll
    for (int h = 0; h < RT::height; ++h) {
        #pragma unroll
        for (int w = 0; w < RT::width; ++w) {
            const int c_local = col0 + 16 * w + lc;
            float b0 = 0.f;
            if (bias != nullptr) {
                const int gc = global_col_base + c_local;
                const int clamped = gc < bias_limit
                    ? gc : (bias_limit > 0 ? bias_limit - 1 : 0);
                b0 = hk_gemm_rs::bf16_to_float(
                    std::bit_cast<std::uint16_t>(bias[clamped]));
            }
            #pragma unroll
            for (int p = 0; p < RT::packed_per_tile; ++p) {
                const float2 v = acc.tiles[h][w].data[p];
                const int r = row0 + 16 * h + lr + 2 * p;
                if (r >= win_row0 && r < win_row0 + win_rows) {
                    stage[(r - win_row0) * stage_stride + c_local] =
                        std::bit_cast<bf16>(
                            hk_gemm_rs::float_to_bf16_rne(v.x + b0));
                }
                if (r + 1 >= win_row0 && r + 1 < win_row0 + win_rows) {
                    stage[(r + 1 - win_row0) * stage_stride + c_local] =
                        std::bit_cast<bf16>(
                            hk_gemm_rs::float_to_bf16_rne(v.y + b0));
                }
            }
        }
    }
}

// Preflight the 16-byte packet contract for all packet-covered columns of a
// band. NO store is issued here; the caller picks the packet path only when
// every check passes, otherwise the bounded scalar path. On the scored table
// (N % 256 == 0 or N % 64 == 0) the contract holds constructively; the runtime
// check is belt-and-braces for gate 9.
__device__ __forceinline__ bool emit_band_preflight(
        const bf16* dst_base, long dst_row_stride,
        const bf16* stage, int stage_stride, int rows, int cols,
        int valid_cols, unsigned tid, unsigned threads) {
    constexpr int ELEMS_PER_LANE = 8;
    const int packets_per_row = cols / ELEMS_PER_LANE;
    for (int packet = static_cast<int>(tid); packet < rows * packets_per_row;
         packet += static_cast<int>(threads)) {
        const int r = packet / packets_per_row;
        const int c = (packet % packets_per_row) * ELEMS_PER_LANE;
        if (c + ELEMS_PER_LANE <= valid_cols) {
            const bf16* d = dst_base + (long)r * dst_row_stride + c;
            const bf16* s = stage + r * stage_stride + c;
            if (!kittens::distributed::packet_contract(d, s, 16u)) return false;
        }
    }
    return true;
}

// Emit one band of `rows` x `cols` bf16 from staging to the owner rank.
// 16-byte packets for every column group of 8 fully inside `valid_cols`;
// a shorter bounded tail finishes a partial group; columns at or past
// `valid_cols` are never emitted. `dst_base` is the band's first element on
// the owner rank (PGL-rebased by the caller); strides are in elements.
__device__ __forceinline__ void emit_band_packets(
        bf16* dst_base, long dst_row_stride,
        const bf16* stage, int stage_stride, int rows, int cols,
        int valid_cols, unsigned tid, unsigned threads) {
    constexpr int ELEMS_PER_LANE = 8;                          // 16 B
    const int packets_per_row = cols / ELEMS_PER_LANE;         // cols % 8 == 0
    for (int packet = static_cast<int>(tid); packet < rows * packets_per_row;
         packet += static_cast<int>(threads)) {
        const int r = packet / packets_per_row;
        const int c = (packet % packets_per_row) * ELEMS_PER_LANE;
        if (c + ELEMS_PER_LANE <= valid_cols) {
            bf16* d = dst_base + (long)r * dst_row_stride + c;
            const bf16* s = stage + r * stage_stride + c;
            hk_gemm_rs::store_peer_packet16(d, s);
        } else {
            // N-tail: bounded element stores, no packet API (gate 9's contract
            // is honored by construction: 16 B packets only where 16 B held).
            for (int e = 0; e < ELEMS_PER_LANE; ++e) {
                if (c + e >= valid_cols) break;
                dst_base[(long)r * dst_row_stride + c + e] =
                    stage[r * stage_stride + c + e];
            }
        }
    }
}

// Fully scalar bounded emit for configurations where 16-byte packet alignment
// cannot be guaranteed (N % 8 != 0). Same bounds contract; no packet API.
__device__ __forceinline__ void emit_band_scalar(
        bf16* dst_base, long dst_row_stride,
        const bf16* stage, int stage_stride, int rows, int cols,
        int valid_cols, unsigned tid, unsigned threads) {
    for (int elem = static_cast<int>(tid); elem < rows * valid_cols;
         elem += static_cast<int>(threads)) {
        const int r = elem / valid_cols;
        const int c = elem % valid_cols;
        dst_base[(long)r * dst_row_stride + c] = stage[r * stage_stride + c];
    }
}

// ---------------------------------------------------------------------------
// MI300X reduce: bounded REDV=1 pull of one EB-row strip. All eight source
// 16 B loads issue before any consumption; FP32 accumulate runs in ascending
// source order; one RNE bf16 pack at output; columns bounded by N. No bias
// here (gate 15): bias lives in the producer epilogue.
// `rows`/`cols` are runtime so the same spelling serves every shape's EB.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void pull_sum_bf16_strip_mlp8(
        std::uint16_t* __restrict__ destination,
        hk_gemm_rs::bf16_sources8 sources,
        std::size_t row_stride, std::size_t row_offset,
        std::size_t column_offset, std::size_t valid_columns,
        int rows, int cols, bool packet_ok,
        unsigned int tid, unsigned int thread_count) {
    const int packets_per_row = cols / 8;            // cols % 8 == 0 by config
    const int packet_count = rows * packets_per_row;
    for (int packet = static_cast<int>(tid); packet < packet_count;
         packet += static_cast<int>(thread_count)) {
        const std::size_t row = row_offset +
            static_cast<std::size_t>(packet / packets_per_row);
        const std::size_t column = column_offset +
            static_cast<std::size_t>(packet % packets_per_row) * 8u;
        const std::size_t offset = row * row_stride + column;
        if (packet_ok && column + 8u <= valid_columns) {
            const auto p0 = hk_gemm_rs::load_packet(sources.source0, offset);
            const auto p1 = hk_gemm_rs::load_packet(sources.source1, offset);
            const auto p2 = hk_gemm_rs::load_packet(sources.source2, offset);
            const auto p3 = hk_gemm_rs::load_packet(sources.source3, offset);
            const auto p4 = hk_gemm_rs::load_packet(sources.source4, offset);
            const auto p5 = hk_gemm_rs::load_packet(sources.source5, offset);
            const auto p6 = hk_gemm_rs::load_packet(sources.source6, offset);
            const auto p7 = hk_gemm_rs::load_packet(sources.source7, offset);
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            float a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
            hk_gemm_rs::add_packet(p0, a0, a1, a2, a3, a4, a5, a6, a7);
            hk_gemm_rs::add_packet(p1, a0, a1, a2, a3, a4, a5, a6, a7);
            hk_gemm_rs::add_packet(p2, a0, a1, a2, a3, a4, a5, a6, a7);
            hk_gemm_rs::add_packet(p3, a0, a1, a2, a3, a4, a5, a6, a7);
            hk_gemm_rs::add_packet(p4, a0, a1, a2, a3, a4, a5, a6, a7);
            hk_gemm_rs::add_packet(p5, a0, a1, a2, a3, a4, a5, a6, a7);
            hk_gemm_rs::add_packet(p6, a0, a1, a2, a3, a4, a5, a6, a7);
            hk_gemm_rs::add_packet(p7, a0, a1, a2, a3, a4, a5, a6, a7);
            hk_gemm_rs::packet_bf16x8 output;
            output.value[0] = hk_gemm_rs::float_to_bf16_rne(a0);
            output.value[1] = hk_gemm_rs::float_to_bf16_rne(a1);
            output.value[2] = hk_gemm_rs::float_to_bf16_rne(a2);
            output.value[3] = hk_gemm_rs::float_to_bf16_rne(a3);
            output.value[4] = hk_gemm_rs::float_to_bf16_rne(a4);
            output.value[5] = hk_gemm_rs::float_to_bf16_rne(a5);
            output.value[6] = hk_gemm_rs::float_to_bf16_rne(a6);
            output.value[7] = hk_gemm_rs::float_to_bf16_rne(a7);
            *reinterpret_cast<uint4*>(destination + offset) = output.packet;
        } else {
            // N tail: byte-exact scalar tail, still source-ascending FP32 with
            // a single RNE pack per surviving element.
            for (int e = 0; e < 8; ++e) {
                const std::size_t cpos = column + static_cast<std::size_t>(e);
                if (cpos >= valid_columns) break;
                const std::size_t off = offset + static_cast<std::size_t>(e);
                float acc = hk_gemm_rs::bf16_to_float(sources.source0[off]);
                acc += hk_gemm_rs::bf16_to_float(sources.source1[off]);
                acc += hk_gemm_rs::bf16_to_float(sources.source2[off]);
                acc += hk_gemm_rs::bf16_to_float(sources.source3[off]);
                acc += hk_gemm_rs::bf16_to_float(sources.source4[off]);
                acc += hk_gemm_rs::bf16_to_float(sources.source5[off]);
                acc += hk_gemm_rs::bf16_to_float(sources.source6[off]);
                acc += hk_gemm_rs::bf16_to_float(sources.source7[off]);
                destination[off] = hk_gemm_rs::float_to_bf16_rne(acc);
            }
        }
    }
}

} // namespace hk_gemm_rs_mi300x

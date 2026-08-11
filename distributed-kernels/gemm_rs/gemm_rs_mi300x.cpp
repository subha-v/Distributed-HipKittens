// ================================================================================================
// gemm_rs_mi300x — single-launch persistent GEMM -> ReduceScatter for 8x MI300X/gfx942.
//
// Design, proofs, and claim boundary: MI300X_DESIGN.md (read sections 2, 3, 7 before the
// wait loops below). Provenance: MI300X_PROVENANCE.md. This file is ADDITIVE: the gfx950
// two-launch port in this directory is untouched.
//
// Topology: one persistent grid of exactly 304 CTAs of 512 threads. CTA pid < num_gemm_ctas
// is a persistent GEMM producer; pids [num_gemm_ctas, 304) are persistent reducers
// (COMET-style CTA-level specialization; no intra-CTA producer-warp split because every
// branch of one HIP kernel inherits the kernel's max resource footprint). Per shape, the
// split NUM_REDUCER_CTAS is a table entry initialised from RadeonFlow's submitted values.
//
// Producer per tile: HipKittens cdna3 double-buffered mainloop (st_bf/rt_bf/mma_ABt) ->
// bias in FP32 -> per-band directed credit wait -> dead-LDS staging -> 16 B peer packets
// into the owner's slot `me` through hk_gemm_rs::peer_view -> one
// producer_drain_release<system> covering all bands -> leader publishes per-band tile epochs
// on the destination rank.
//
// Reducer per tile: lanes s in [0,8) bounded-poll ready[s][lrow][col] >= e -> converge ->
// cta_acquire<system> -> pull_sum_bf16_strip_mlp8 (8 loads issued, source-ascending FP32
// consumption, single RNE pack, N-bounded) -> consumer_drain -> leader returns directed
// credit[me][lrow][col] = e to every source rank.
//
// Epochs are device-derived per CTA (epoch32); no host-supplied epoch exists.
// The production module exports exactly one entry point and exactly one kernel launch per
// call (gate 16). Negative controls exist only under HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS
// as a separate module (gate 19).
// ================================================================================================

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#include "gemm_rs_mi300x_hk_adapter.cuh"

#include <stdexcept>
#include <string>

#ifndef HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS
#define HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS 0
#endif

using namespace kittens;
namespace m3 = hk_gemm_rs_mi300x;
using G = kittens::group<m3::NUM_WARPS>;

using _gl_A = gl<bf16, -1, -1, -1, -1>;   // [M, K_local]
using _gl_B = gl<bf16, -1, -1, -1, -1>;   // [N, K_local] (row-major weight shard)
using _gl_C = gl<bf16, -1, -1, -1, -1>;   // heap/out [*, 1, rows, N]

// ================================================================================================
// Kernel argument POD. All pointer smuggling goes through uintptr_t exactly as the gfx950
// donor did; the two 72-byte symmetric descriptors live in device-visible memory built once
// by hk_gemm_rs::host_abi::snapshot_allocation_descriptors.
// ================================================================================================
struct mi300x_globals {
    _gl_A a;                     // LOCAL [M, K_local]
    _gl_B b;                     // LOCAL [N, K_local]
    _gl_C c_heap;                // SYMMETRIC bf16 [8, 1, M/8, N] (local view)
    _gl_C out;                   // LOCAL bf16 [1, 1, M/8, N]
    std::uintptr_t c_heap_peers; // device ptr -> hk_gemm_rs::symmetric_descriptor
    std::uintptr_t sig;          // SYMMETRIC u32 signals (local base)
    std::uintptr_t sig_peers;    // device ptr -> hk_gemm_rs::symmetric_descriptor
    std::uintptr_t ep_cell;      // LOCAL u32[m3::EP_U32]
    std::uintptr_t bias;         // const bf16* [N], or 0
    std::uintptr_t err;          // LOCAL int[1]
    std::uintptr_t stream_ptr;
    std::uint64_t spin_limit;
    int me;
    int M, N, K;                 // K = local K shard
    int lrows, cols;             // lrow_count, col_count
    int eb;                      // emit-band height (rows)
    int num_gemm_ctas;
    int ready_words;             // credit region offset in sig (words)
    int packet_fast_path;        // (N % 8) == 0
    // Negative-control fields: zero in the production binding. The consuming
    // branches exist only under HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS.
    unsigned ctrl_flags;
    int ctrl_rank, ctrl_arg0;
    int config_row;              // 0 = generic; [1..6] = scored table rows
    int even_k;                  // K_local % BK == 0 for the resolved config
    dim3 grid() const  { return dim3(m3::CU_COUNT); }
    dim3 block() const { return dim3(m3::CTA_THREADS); }
    size_t dynamic_shared_memory() const { return kittens::MAX_SHARED_MEMORY; }
};

// ================================================================================================
// K-tail register mask (exp-07 discipline, cdna3 layout). For a row-layout rt_bf base tile
// (16x16) lane L holds row = 16h + L%16 and K columns 16w + 4*(L/16) + {0,1,2,3}, packed as
// two bf16_2 in data[0..1] (see cdna3 ops/warp/memory/tile/shared_to_register.cuh's
// row_offset/col_offset). Zeroing A's padding columns kills the term unconditionally --
// B's ragged tail values multiply exact zeros, whatever they are. The only hardware
// assumption is that the ragged read itself does not fault (donor's oob_probe premise).
// ================================================================================================
template<typename RT>
__device__ __forceinline__ void mask_a_k_tail(RT &t, int tail_k) {
    using BT = __hip_bfloat16;
    const int lane = kittens::laneid();
    const int col0 = 4 * (lane / 16);            // first of this lane's 4 K columns
    #pragma unroll
    for (int h = 0; h < RT::height; ++h) {
        #pragma unroll
        for (int w = 0; w < RT::width; ++w) {
            #pragma unroll
            for (int p = 0; p < RT::packed_per_tile; ++p) {
                // each packed bf16_2 covers K columns 16w + col0 + 2p + {0,1}
                const int kc = 16 * w + col0 + 2 * p;
                if (kc >= tail_k) {
                    reinterpret_cast<BT*>(&t.tiles[h][w].data[p])[0] = BT(0.0f);
                }
                if (kc + 1 >= tail_k) {
                    reinterpret_cast<BT*>(&t.tiles[h][w].data[p])[1] = BT(0.0f);
                }
            }
        }
    }
}

// ================================================================================================
// THE MEGAKERNEL.
// ================================================================================================
template<int BM, int BN, int BK, bool K_TAIL>
__global__ __launch_bounds__(m3::CTA_THREADS, 1)
void gemm_rs_mi300x_kernel(const mi300x_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    static_assert(BM % (m3::WARPS_M * 16) == 0 && BN % (m3::WARPS_N * 16) == 0);
    static_assert((BM % m3::WARPS_M) == 0 && (BN % m3::WARPS_N) == 0);
    // LDS: A and B double buffers; the emit staging aliases them after the
    // last MFMA. Exact footprint must honor the gfx942 per-workgroup limit.
    constexpr int LDS_BYTES = 2 * (BM + BN) * BK * (int)sizeof(bf16);
    static_assert(LDS_BYTES <= 65536);
    constexpr int WM = BM / m3::WARPS_M;           // >= 16 rows per warp
    constexpr int WN = BN / m3::WARPS_N;           // >= 16 cols per warp
    static_assert(WM >= 16 && WN >= 16);
    // Staging window capacity check: max 32 rows x BN cols bf16 must alias the
    // dead A/B buffers (gate 3's bounds discipline is constructive).
    constexpr int STAGE_BYTES_MAX = 32 * BN * (int)sizeof(bf16);
    static_assert(STAGE_BYTES_MAX <= LDS_BYTES);

    const int pid = blockIdx.x;
    const int me = g.me;
    const int M = g.M, N = g.N, K = g.K;
    const int slice = M / m3::WORLD_SIZE;
    const int EB = g.eb;
    const int lrows = g.lrows, cols = g.cols;

    std::uint32_t* sig = reinterpret_cast<std::uint32_t*>(g.sig);
    std::uint32_t* epc = reinterpret_cast<std::uint32_t*>(g.ep_cell);
    int* errp = reinterpret_cast<int*>(g.err);
    const std::uint64_t spin = g.spin_limit;
    const auto* sig_peers =
        reinterpret_cast<const hk_gemm_rs::symmetric_descriptor*>(g.sig_peers);
    const auto* heap_peers =
        reinterpret_cast<const hk_gemm_rs::symmetric_descriptor*>(g.c_heap_peers);

    if (pid < g.num_gemm_ctas) {
        // ===================================================================
        // PRODUCER ROLE: persistent GEMM CTAs, tile loop strided by NG.
        // ===================================================================
        const std::uint32_t ep =
            m3::next_epoch_workgroup(epc + m3::EP_GEMM_U32 + pid);
        // Sticky abort: never emit into a poisoned protocol instance.
        if (m3::error_bit_set(errp, m3::ERR_PRODUCER_CREDIT |
                                    m3::ERR_REDUCER_READY)) return;

        st_bf<BM, BK> (&As)[2] = al.allocate<st_bf<BM, BK>, 2>();
        st_bf<BN, BK> (&Bs)[2] = al.allocate<st_bf<BN, BK>, 2>();
        bf16* const stage = reinterpret_cast<bf16*>(&__shm[0]);

        rt_bf<WM, BK, ducks::rt_layout::row> A_frag;
        rt_bf<WN, BK, ducks::rt_layout::row> B_frag;
        rt_fl<WM, WN, ducks::rt_layout::col> C_accum;

        const int warp = kittens::warpid() % m3::NUM_WARPS;
        const int warp_row = warp / m3::WARPS_N;   // warps cover (WM x WN)
        const int warp_col = warp % m3::WARPS_N;

        const int num_pid_m = M / BM;
        const int num_pid_n = cols;
        const int tiles = num_pid_m * num_pid_n;
        const int k_iters = (K + BK - 1) / BK;
        const int tail_k = K_TAIL ? (K - (k_iters - 1) * BK) : BK;
        constexpr int WGM = 4;
        // XCD remap is a pure scheduling transform; the initial design keeps
        // the natural loop stride and leaves XCD-tuned orders to the retune
        // table (MI300X_DESIGN.md section 8). Coverage below is exact: each
        // tile index is visited by exactly one producer CTA per launch.

        for (int t = pid; t < tiles; t += g.num_gemm_ctas) {
            // grouped L2-friendly tile order (same family as both donors).
            const int in_group = WGM * num_pid_n;
            const int group = t / in_group;
            const int first = group * WGM;
            const int gsize = (num_pid_m - first < WGM) ? (num_pid_m - first)
                                                        : WGM;
            const int tm = first + (t % in_group) % gsize;
            const int tn = (t % in_group) / gsize;

            zero(C_accum);
            // ---- double-buffered mainloop (correctness-first scheduling) ----
            G::load(As[0], g.a, {0, 0, tm, 0});
            G::load(Bs[0], g.b, {0, 0, tn, 0});
            __syncthreads();
            for (int k = 0; k < k_iters; ++k) {
                load(A_frag, subtile_inplace<WM, BK>(As[k & 1], {warp_row, 0}));
                load(B_frag, subtile_inplace<WN, BK>(Bs[k & 1], {warp_col, 0}));
                if (k + 1 < k_iters) {
                    G::load(As[(k + 1) & 1], g.a, {0, 0, tm, k + 1});
                    G::load(Bs[(k + 1) & 1], g.b, {0, 0, tn, k + 1});
                }
                // The A operand carries the only mask: whatever lands in the
                // K-padding columns of A is overwritten with exact zeros
                // before the MFMA (gates 3, 15).
                if constexpr (K_TAIL) {
                    if (k == k_iters - 1) mask_a_k_tail(A_frag, tail_k);
                }
                mma_ABt(C_accum, A_frag, B_frag, C_accum);
                __syncthreads();
            }

            // ---- egress: per-band credit wait, emit, one release, publish ---
            const int bands = BM / EB;             // EB | BM (gate 5)
            // Every band of this tile must be reusable before ANY payload
            // store of epoch ep: waits on leader lanes only, result broadcast
            // by the CTA barrier + shared error bit (fail-closed).
            if (threadIdx.x < (unsigned)bands) {
                const int b = (int)threadIdx.x;
                const int row0 = tm * BM + b * EB;
                const int dest = m3::owner_of_row(row0, slice);
                const int lrow = (row0 - dest * slice) / EB;
                const bool ok = m3::wait_reuse_credit(
                    sig + m3::SIGNAL_GUARD_U32 + g.ready_words +
                        m3::credit_idx(dest, lrow, tn, lrows, cols),
                    ep, spin, errp, m3::ERR_PRODUCER_CREDIT);
                if (!ok) atomicOr(errp, m3::ERR_PRODUCER_CREDIT);
            }
            __syncthreads();
            if (m3::error_bit_set(errp, m3::ERR_PRODUCER_CREDIT)) return;

            const bf16* biasp = reinterpret_cast<const bf16*>(g.bias);
            const int win_rows = EB < 32 ? EB : 32;      // EB % win_rows == 0
            const int vt = N - tn * BN < BN ? N - tn * BN : BN;   // <= BN

            for (int b = 0; b < bands; ++b) {
                const int row0 = tm * BM + b * EB;
                const int dest = m3::owner_of_row(row0, slice);
                const int lrow = (row0 - dest * slice) / EB;
                int dest_eff = dest;
#if HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS
                // REROUTE-SLOT control: land this band on the wrong rank while
                // publishing the true destination. Numerics MUST fail.
                if ((g.ctrl_flags & m3::CTRL_REROUTE_SLOT) &&
                    me == g.ctrl_rank && b == g.ctrl_arg0) {
                    dest_eff = (dest + 1) % m3::WORLD_SIZE;
                }
#endif
                auto dst = hk_gemm_rs::peer_view(g.c_heap, heap_peers, dest_eff);
                bf16* const dst_base =
                    &dst[{me, 0, lrow * EB, tn * BN}];
                const long dst_stride = dst.template stride<2>();

                for (int w0 = 0; w0 < EB; w0 += win_rows) {
                    m3::stage_fragment_bf16(
                        stage, BN, C_accum, biasp,
                        warp_row * WM, warp_col * WN,
                        /*win_row0=*/b * EB + w0, /*win_rows=*/win_rows,
                        /*global_col_base=*/tn * BN, /*bias_limit=*/N);
                    __syncthreads();
                    bf16* const wdst = dst_base + (long)w0 * dst_stride;
                    if (g.packet_fast_path) {
                        // Contract check first; fall back to scalar wholesale
                        // rather than violate a single 16-byte rule.
                        bool contract_ok = m3::emit_band_preflight(
                            wdst, dst_stride, stage, BN, win_rows, BN, vt,
                            threadIdx.x, m3::CTA_THREADS);
                        // Uniform decision: any lane's failure is everyone's.
                        if (__builtin_amdgcn_readfirstlane(
                                static_cast<unsigned>(!contract_ok))) {
                            m3::emit_band_scalar(
                                wdst, dst_stride, stage, BN, win_rows, BN, vt,
                                threadIdx.x, m3::CTA_THREADS);
                        } else {
                            m3::emit_band_packets(
                                wdst, dst_stride, stage, BN, win_rows, BN, vt,
                                threadIdx.x, m3::CTA_THREADS);
                        }
                    } else {
                        m3::emit_band_scalar(
                            wdst, dst_stride, stage, BN, win_rows, BN, vt,
                            threadIdx.x, m3::CTA_THREADS);
                    }
                    __syncthreads();
                }
            }

            // One grouped CTA release covers every band of this tile, then
            // the leader publishes each band's cheap completion cell on its
            // destination rank (publication separate from movement, F6-style).
            m3::release_payload_system();
            if (threadIdx.x == 0) {
                for (int b = 0; b < bands; ++b) {
                    const int row0 = tm * BM + b * EB;
                    const int dest = m3::owner_of_row(row0, slice);
                    const int lrow = (row0 - dest * slice) / EB;
#if HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS
                    // DROP-PUBLICATION control: this consumer MUST time out
                    // without reading. Production builds cannot see this code.
                    if ((g.ctrl_flags & m3::CTRL_DROP_PUBLICATION) &&
                        me == g.ctrl_rank && dest == g.ctrl_arg0) {
                        continue;
                    }
#endif
                    m3::publish_band_epoch(
                        sig, sig_peers, dest, me,
                        m3::SIGNAL_GUARD_U32 +
                            m3::ready_idx(me, lrow, tn, lrows, cols),
                        ep);
                }
            }
        }
        return;
    }

    // =======================================================================
    // REDUCER ROLE: persistent CTAs, tile loop strided by NUM_REDUCER_CTAS.
    // =======================================================================
    const int pid_r = pid - g.num_gemm_ctas;
    const int num_reducer_ctas = m3::CU_COUNT - g.num_gemm_ctas;
    const std::uint32_t ep =
        m3::next_epoch_workgroup(epc + m3::EP_RED_U32 + pid_r);
    if (m3::error_bit_set(errp, m3::ERR_PRODUCER_CREDIT |
                                m3::ERR_REDUCER_READY)) return;

    const int red_tiles = lrows * cols;
    const std::size_t slot_stride =
        static_cast<std::size_t>(slice) * static_cast<std::size_t>(N);
    const std::uint16_t* heap =
        reinterpret_cast<const std::uint16_t*>(g.c_heap.raw_ptr);
    std::uint16_t* out = reinterpret_cast<std::uint16_t*>(g.out.raw_ptr);

    for (int t = pid_r; t < red_tiles; t += num_reducer_ctas) {
        const int lr = t / cols;
        const int col = t % cols;

        // Bounded ready wait on exactly the eight source contributions of this
        // output tile (gate 6). Any lane's timeout is fail-closed: the bit is
        // set, the CTA converges, and NO payload read follows (gate 12).
        if (threadIdx.x < m3::WORLD_SIZE) {
            const bool ok = m3::wait_band_epoch(
                sig + m3::SIGNAL_GUARD_U32 +
                    m3::ready_idx((int)threadIdx.x, lr, col, lrows, cols),
                ep, spin, errp, m3::ERR_REDUCER_READY);
            if (!ok) atomicOr(errp, m3::ERR_REDUCER_READY);
        }
        __syncthreads();
        if (m3::error_bit_set(errp, m3::ERR_REDUCER_READY)) return;
        m3::acquire_payload_system();

        const hk_gemm_rs::bf16_sources8 srcs{
            heap + slot_stride * 0u, heap + slot_stride * 1u,
            heap + slot_stride * 2u, heap + slot_stride * 3u,
            heap + slot_stride * 4u, heap + slot_stride * 5u,
            heap + slot_stride * 6u, heap + slot_stride * 7u};
        m3::pull_sum_bf16_strip_mlp8(
            out, srcs, /*row_stride=*/(std::size_t)N,
            /*row_offset=*/(std::size_t)(lr * EB),
            /*column_offset=*/(std::size_t)(col * BN),
            /*valid_columns=*/(std::size_t)N,
            /*rows=*/EB, /*cols=*/BN,
            /*packet_ok=*/g.packet_fast_path != 0,
            threadIdx.x, m3::CTA_THREADS);

        // All consumer reads must drain before any retirement credit is
        // published (gate 13); the credit is what unblocks the epoch-(e+1)
        // producer overwrite of these exact slots (gate 14).
        m3::finish_tile_consumption();
        if (threadIdx.x == 0) {
            for (int s = 0; s < m3::WORLD_SIZE; ++s) {
#if HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS
                // DROP-CREDIT control: that source's epoch-2 producer MUST
                // time out without overwriting the slot.
                if ((g.ctrl_flags & m3::CTRL_DROP_CREDIT) &&
                    me == g.ctrl_rank && s == g.ctrl_arg0) {
                    continue;
                }
#endif
                m3::publish_reuse_credit(
                    sig, sig_peers, s, me,
                    m3::SIGNAL_GUARD_U32 + g.ready_words +
                        m3::credit_idx(me, lr, col, lrows, cols),
                    ep);
            }
        }
    }
}

// ================================================================================================
// Host dispatch: one launch per call (gate 16). Shape resolution and every
// validation live in host_abi::resolve_shape; this binding only selects the
// pre-compiled instantiation matching the plan's (config_row, even_k) pair.
// ================================================================================================
template<int BM, int BN, int BK, bool K_TAIL>
static void launch_fixed(const mi300x_globals& g_plan) {
    constexpr int LDS_BYTES = 2 * (BM + BN) * BK * (int)sizeof(bf16);
    static_assert(LDS_BYTES <= 65536);
    // The dynamic-LDS attribute is a property of the instantiation, not of the
    // call, so it is set once per process. Setting it per launch put a host
    // API round trip on the critical path of every call; measured on gfx942
    // that dominated the smallest shapes, whose entire SOL budget is 6.46 us.
    static const hipError_t attribute_status = hipFuncSetAttribute(
        (void*)gemm_rs_mi300x_kernel<BM, BN, BK, K_TAIL>,
        hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES);
    if (attribute_status != hipSuccess) {
        throw std::runtime_error(
            std::string("GEMM-RS mi300x: cannot reserve ") +
            std::to_string(LDS_BYTES) + " B of dynamic LDS: " +
            hipGetErrorString(attribute_status));
    }
    hipStream_t s = reinterpret_cast<hipStream_t>(g_plan.stream_ptr);
    gemm_rs_mi300x_kernel<BM, BN, BK, K_TAIL>
        <<<g_plan.grid(), g_plan.block(), (std::size_t)LDS_BYTES, s>>>(g_plan);
}

void dispatch_gemm_rs_mi300x(mi300x_globals g) {
    switch (g.config_row) {
        case 1: launch_fixed< 32, 256, 32, false>(g); return;
        case 2: launch_fixed< 64,  64, 64, false>(g); return;
        case 3: launch_fixed<128, 256, 32,  true>(g); return;
        case 4: launch_fixed<256, 256, 32, false>(g); return;
        case 5: launch_fixed<256, 256, 32, false>(g); return;
        case 6: launch_fixed<256, 256, 32,  true>(g); return;

        // Generic correctness row: bounded paths carry the tails.
        default:
            if (g.even_k) launch_fixed<32, 64, 64, false>(g);
            else          launch_fixed<32, 64, 64,  true>(g);
            return;
    }
}

// ================================================================================================
// Bindings. The production module exposes exactly one function; the negative
// control module (separate TU product) exposes only the control variants.
// ================================================================================================
PYBIND11_MODULE(TK_MODNAME, m) {
#if HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS
    m.doc() = "GEMM-RS mi300x NEGATIVE-CONTROL module (not production)";
    py::bind_function<dispatch_gemm_rs_mi300x>(m,
        "gemm_rs_mi300x_control",
        &mi300x_globals::a, &mi300x_globals::b, &mi300x_globals::c_heap,
        &mi300x_globals::out, &mi300x_globals::c_heap_peers,
        &mi300x_globals::sig, &mi300x_globals::sig_peers,
        &mi300x_globals::ep_cell, &mi300x_globals::bias, &mi300x_globals::err,
        &mi300x_globals::stream_ptr, &mi300x_globals::spin_limit,
        &mi300x_globals::me, &mi300x_globals::M, &mi300x_globals::N,
        &mi300x_globals::K, &mi300x_globals::lrows, &mi300x_globals::cols,
        &mi300x_globals::eb, &mi300x_globals::num_gemm_ctas,
        &mi300x_globals::ready_words, &mi300x_globals::packet_fast_path,
        &mi300x_globals::ctrl_flags, &mi300x_globals::ctrl_rank,
        &mi300x_globals::ctrl_arg0, &mi300x_globals::config_row,
        &mi300x_globals::even_k);
#else
    m.doc() = "GEMM-RS mi300x: one-launch persistent GEMM -> ReduceScatter";
    py::bind_function<dispatch_gemm_rs_mi300x>(m, "gemm_rs_mi300x",
        &mi300x_globals::a, &mi300x_globals::b, &mi300x_globals::c_heap,
        &mi300x_globals::out, &mi300x_globals::c_heap_peers,
        &mi300x_globals::sig, &mi300x_globals::sig_peers,
        &mi300x_globals::ep_cell, &mi300x_globals::bias, &mi300x_globals::err,
        &mi300x_globals::stream_ptr, &mi300x_globals::spin_limit,
        &mi300x_globals::me, &mi300x_globals::M, &mi300x_globals::N,
        &mi300x_globals::K, &mi300x_globals::lrows, &mi300x_globals::cols,
        &mi300x_globals::eb, &mi300x_globals::num_gemm_ctas,
        &mi300x_globals::ready_words, &mi300x_globals::packet_fast_path,
        &mi300x_globals::ctrl_flags, &mi300x_globals::ctrl_rank,
        &mi300x_globals::ctrl_arg0, &mi300x_globals::config_row,
        &mi300x_globals::even_k);
#endif
}

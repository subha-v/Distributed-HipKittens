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
#include <vector>

#ifndef HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS
#define HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS 0
#endif

// Restores the donors' WGM = 4 grouped tile order. Exists as exp_08's control
// arm; see the tile-order comment in the producer role for why 4 costs the two
// large shapes most of their xGMI egress bandwidth.
#ifndef HK_GEMM_RS_MI300X_WGM4
#define HK_GEMM_RS_MI300X_WGM4 0
#endif

// Release granularity (exp_05 / E3). The number of output tiles one producer
// CTA emits between consecutive producer_drain_release() calls: the tile loop
// becomes an outer group loop that emits RELEASE_GROUP tiles, issues ONE
// release, then publishes all of the group's (tile, band) ready flags.
//
// It is a compile-time constant with a hard cap, and it is applied identically
// on the generic config row. "One release per CTA per epoch" as an unbounded
// rule is rejected: the generic row is BM=32/BN=64 with 280 producers, so
// 8192x8192 is 32768 tiles = up to 118 tiles per CTA, and deferring every
// publication to the end of the GEMM there would make every reducer spin
// through the whole mainloop. 1 is the behaviour-preserving control arm.
#ifndef HK_GEMM_RS_MI300X_RELEASE_GROUP
#define HK_GEMM_RS_MI300X_RELEASE_GROUP 4
#endif

// Group only when a producer CTA owns at least RELEASE_GROUP tiles, i.e. only
// when the group can be full. 0 groups unconditionally (exp_05's arm rg4).
//
// This is measured, not assumed. Batching trades release cost for publication
// delay, and the two do not scale together:
//
//   * 8192x8192x29568 owns 4 tiles per CTA and 4 releases cost it ~135 us of
//     1778. Collapsing them to one saved 5.96% (paired, null-arm floor 0.10%),
//     which is within a few tenths of the 3/4 x 135 us the release attribution
//     predicts. The reduce that now has to happen in the tail is only ~4.9% of
//     the op and hides under the remaining producers.
//   * 512x4096x12288 owns 2 tiles per CTA and is an 88 us operation whose
//     reducers get 2 rounds of work. Deferring the first tile's publication to
//     the end of the second COSTS 3.75% -- more than halving its releases can
//     possibly save. 8192x4096x14336, also 2 tiles per CTA, is inside its own
//     noise either way.
//
// So the rule keys on the one quantity that separates them and that the kernel
// already knows: how many tiles a CTA owns. A partial group is never worth its
// publication delay in anything measured here. Shapes with 3 tiles per CTA are
// untested and take the ungrouped path, which is the conservative side.
#ifndef HK_GEMM_RS_MI300X_RELEASE_GROUP_FULL_ONLY
#define HK_GEMM_RS_MI300X_RELEASE_GROUP_FULL_ONLY 1
#endif

// exp_26. FULL_ONLY as written above is a step function of tiles_per_cta: a CTA
// that owns 4 tiles groups all four, and a CTA that owns 3, 2 or 1 groups none.
// The cliff is not a property of the mechanism, only of comparing against a
// single constant, and it costs exactly one shape. exp_20's attribution:
// release collapsed 134.9 -> 15.0 us on 8192x8192x29568 (4 tiles per CTA, which
// groups) and is still 65.4 us, 10.2% of 641.9, on 8192x4096x14336 (2 tiles per
// CTA, which does not) -- corroborated by the writeback-origin counters, where
// the release pushes 13.1% of L2 writebacks on the first and 56.4% on the
// second.
//
// So let the group size fall to what the shape can actually fill. Both terms are
// already computed; the cap and every invariant of the group loop are untouched.
// Per graded shape the effective group goes 1/1/1/1/1/4 -> 1/1/1/1/2/4: only
// 8192x4096x14336 moves, and the four one-tile rows are arithmetically pinned to
// 1 either way, which makes them controls rather than collateral.
//
// HOW that is spelled turns out to matter as much as the rule. See the value
// list on the macro below: `min(RELEASE_GROUP, tiles_per_cta)` is the obvious
// spelling and it is measurably the wrong one.
//
// What this rule does NOT do is revive the arm E3 rejected. rgroup <= the tile
// count of the CTA that owns the most, so no CTA ever waits on a tile that does
// not exist, and `emitted = min(left, rgroup)` still truncates the last group of
// a short-changed CTA to what it actually owns. The unconditional arm's defect
// was different in kind: at RELEASE_GROUP = 4 it made a 2-tile CTA's loop stride
// 4 CTAs' worth of tiles, so the group could only ever be half full and the
// publication of the first tile was deferred behind a mainloop that had no
// second tile to amortize it.
//
// The honest residual risk is publication delay, not fullness, and it is not
// removed by anything above: on 8192x4096x14336 the first of the two tiles is
// now published one mainloop later than before. That is the same trade E3
// measured as a 3.75% LOSS on the then-2-tile 512x4096x12288 (an 88 us shape
// whose reducers got 2 rounds of work) and as inside its own noise on
// 8192x4096x14336 itself, before WGM and the NR retune moved both. It is a
// measurement, and exp_26 makes it against a paired null arm.
// REVERTED TO 0 ON 2026-08-12. It was landed as 2 on a paired A/B that measured
// -6.56% on 8192x4096x14336 in 80 of 80 rounds against a -0.61% null, with a
// full green ladder including M9. **That win was an artifact of allocation
// order and does not exist.**
//
// The decisive test put both binaries in ONE pool as `ours` and `ours_prev`,
// bit-equality-asserted on all 8 ranks, arm order balanced by a complete Latin
// square (all 3! permutations, residual pinning exactly zero), 336 paired rounds
// per cell. Shape 5, negative = this rule faster:
//
//     `ours` allocated 1st : -3.11% pipelined, 336/336 wins
//     `ours` allocated 3rd : +5.85% pipelined,   0/336 wins
//
// Both at p ~ 1e-101. Same two binaries, same pool, same balanced ordering; the
// only difference is which arm was CONSTRUCTED first. A p of 1e-101 that
// reverses under permutation of the setup is measuring the instrument, and the
// perfect internal consistency is the signature -- noise widens, artifacts
// reverse.
//
// The clincher: on 8192x8192x29568 both arms compile to rgroup 4 and are
// literally the same computation, and the same arrangement still reports -1.67%
// against a -0.14% null, clearing its own null by 10x. The effect is
// manufacturable on identical code, at exp_26's shape and size class.
//
// Order-balanced, with the residual MEASURED on the five shapes where both rules
// agree and truth is exactly zero, shape 5 comes out +1.37% pipelined / +1.12%
// graded SLOWER. exp_26's -6.56% appears in no configuration.
//
// Allocation happens once per process, before the first round, so no
// within-round permutation can control it -- running both construction orders is
// the only control, and any future paired A/B in this tree must do so.
//
// Formally the outcome is INDETERMINATE, since the two orders disagree in sign.
// Reverted on burden of proof: a candidate ships only when it is shown better,
// and this one's sole support is withdrawn while the only estimate with a
// measured residual says mildly slower.
//
// It briefly defaulted to 1 before any of that had been run, which would have
// silently redefined the production binary for every other experiment building
// from this file, including the waterfall's rung (c) "shipped binary" arm. That
// was wrong then and the value below is only 2 now because the evidence exists.
//
// Values -- two spellings of the same rule, which do NOT measure the same:
//
//   0  the incumbent step rule, `tiles_per_cta >= RELEASE_GROUP ? RG : 1`.
//   1  `min(RELEASE_GROUP, tiles_per_cta)`, the obvious spelling. It makes
//      rgroup an arbitrary value in [1, RG] where the incumbent made it a
//      member of the two-element set {1, RG}, and the compiler prices that:
//      +2 VGPRs on five of the seven instantiations (246 -> 248 and 248 -> 250
//      on the 256x256 rows, which sit against the 256 arch cap) and a different
//      schedule everywhere, at unchanged occupancy, spills and ordering-op
//      counts. exp_09 recorded the same class of effect from the same variable:
//      folded constant, known bound and runtime value are three schedules.
//   2  the same rule spelled as a descending select over a COMPILE-TIME LADDER
//      of group sizes, so rgroup is again one of a handful of literals. Its
//      resource tuple is the incumbent's on all seven instantiations, to the
//      register, and it adds between -1 and +7 instructions -- the extra rung
//      and nothing else. Correct and fully gated, but NOT faster; see above.
//
// If this is ever revisited, 2 is the spelling to revisit -- never 1, which is
// the same rule with a strictly worse schedule. And it must be judged in BOTH
// construction orders, because one order alone fabricates a 1e-101 result here.
//
// The ladder is `{RELEASE_GROUP, 2, 1}` -- a strict generalization of the
// incumbent, one rung added. A CTA owning 3 tiles takes the 2 rung and its last
// group is truncated to 1 by `emitted`, the same conservative treatment every
// short CTA already gets.
#ifndef HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE
#define HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE 0
#endif

// exp_14 (E4b) tile screening. BM/BN/BK are template parameters, so unlike the
// reducer split they cannot be swept from the host without an instantiation per
// candidate. Off by default: the extra rows exist only in the sweep module, so
// the production binding compiles exactly the six instantiations gate M2
// expects and the screening arms cost the graded build neither code size nor
// compile time.
#ifndef HK_GEMM_RS_MI300X_TILE_SWEEP
#define HK_GEMM_RS_MI300X_TILE_SWEEP 0
#endif

// aug11/exp_22 phase-event ring: the per-CTA timestamps behind the paper's
// Fig 3 (per-layer resource-utilization timeline). DIAGNOSTIC ONLY.
//
// DEFAULT 0, AND IT MUST STAY 0. Everything under this flag exists to picture
// the kernel, not to be the kernel: it adds one scalar clock read and one
// relaxed store per phase boundary by tid 0. A default of 1 would silently
// redefine the production binary for every experiment that compiles this file
// -- including the waterfall's shipped-binary reference rung -- and the
// timeline would then be a picture of an instrumented kernel nobody ships.
// The resource-tuple parity gate (exp_22_timeline/parity_gate.sh) exists to
// prove flag-present-and-0 is byte-identical to flag-absent.
#ifndef HK_GEMM_RS_MI300X_TRACE
#define HK_GEMM_RS_MI300X_TRACE 0
#endif

using namespace kittens;
namespace m3 = hk_gemm_rs_mi300x;
using G = kittens::group<m3::NUM_WARPS>;

#if HK_GEMM_RS_MI300X_TRACE
// ================================================================================================
// exp_22 phase-event ring. One u64 per phase boundary crossed:
//     (phase_id << 56) | (s_memrealtime() & 0x00FFFFFFFFFFFFFF)
// written by tid 0 of each CTA with a PLAIN RELAXED STORE. No new fences and
// no new atomics: an instrument that adds ordering to the paths it pictures
// measures itself. The host reads the ring after the launch completes; nothing
// reads it in flight.
//
// Depth is derived from this kernel's own worst case, not inherited from the
// gfx950 sibling's 24. Events per CTA per epoch are
//     producer = 1 + groups*(6*tiles_in_group + 3) + 1
//     reducer  = 1 + 5*red_tiles_per_cta + 1
// which over the six graded rows peaks at 29 (shape 6; shape 5 is 20). 64 is
// 2.17x that at 304*64*8 = 155,648 B per rank. Overflow DROPS and counts in
// the last slot -- it never wraps, because a wrapped ring produces a
// plausible-looking but wrong timeline. See exp_22_timeline/design.md.
// ================================================================================================
namespace hk_trace {

inline constexpr int DEPTH  = 64;          // slots per CTA
inline constexpr int USABLE = DEPTH - 1;   // [DEPTH-1] is the drop counter

// Fixed and ordered by lifecycle. Write order within a CTA is monotone in
// timestamp but NOT in id -- a producer cycles MAINLOOP_BEG..PUBLISH_END once
// per tile group. Absence of an id means that CTA did not run that phase:
// producers and reducers run disjoint subsets.
enum : int {
    CTA_BEG = 0, MAINLOOP_BEG, MAINLOOP_END, CREDIT_WAIT_BEG, CREDIT_WAIT_END,
    EMIT_BEG, EMIT_END, RELEASE_BEG, RELEASE_END, PUBLISH_END,
    READY_WAIT_BEG, READY_WAIT_END, REDUCE_BEG, REDUCE_END, CREDIT_PUB_END,
    CTA_END
};

__device__ __forceinline__ void stamp(std::uint64_t* ring, int cta, int& n,
                                      int id) {
    if (ring == nullptr || threadIdx.x != 0u) return;
    std::uint64_t* const base = ring + (std::size_t)cta * DEPTH;
    const std::uint64_t t = __builtin_amdgcn_s_memrealtime();
    if (n < USABLE) {
        base[n++] = ((std::uint64_t)id << 56) |
                    (t & 0x00FFFFFFFFFFFFFFull);
    } else {
        // Non-atomic RMW is correct: exactly one thread in the process writes
        // this word.
        base[USABLE] += 1ull;
    }
}

}  // namespace hk_trace

#define HK_TRACE_STAMP(id) \
    ::hk_trace::stamp(trace_ring, pid, trace_n, ::hk_trace::id)
#else
#define HK_TRACE_STAMP(id) do {} while (0)
#endif

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
#if HK_GEMM_RS_MI300X_TRACE
    // exp_22 diagnostic ring, u64[CU_COUNT][hk_trace::DEPTH], or 0 for "do not
    // trace". LAST member on purpose: prebound::configure's aggregate
    // initializer stops at even_k, so this member is value-initialized to 0
    // there and the prebound launch path compiles unchanged and never traces.
    std::uintptr_t trace;
#endif
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
// The tile -> (tm, tn) map, as ONE function.
//
// It exists as a function rather than as straight-line code inside the tile
// loop because the deferred publish loop has to name the same tiles the emit
// loop just wrote, and the alternatives are worse: journalling the group in LDS
// is impossible (the emit staging buffer aliases the A/B double buffers, which
// are already at the 65536 B ceiling for BM=BN=256) and journalling it in
// registers demotes to scratch (a runtime-indexed local array on rows that
// already sit at 246-248 of 256 VGPRs). Recomputing costs a handful of scalar
// ops per published band and makes an emit/publish index drift impossible
// without editing one shared expression.
//
// `wgm` is a runtime argument because exp_08 made it one: WGM = 4 when every
// tile is resident at once, num_pid_m otherwise. Both callers must pass the
// same value, which they do -- it is loop-invariant and computed once.
// ================================================================================================
struct tile_id { int tm, tn; };

__device__ __forceinline__ tile_id decode_tile(int t, int num_pid_m,
                                               int num_pid_n, int wgm) {
    const int in_group = wgm * num_pid_n;
    const int group = t / in_group;
    const int first = group * wgm;
    const int gsize = (num_pid_m - first < wgm) ? (num_pid_m - first) : wgm;
    return {first + (t % in_group) % gsize, (t % in_group) / gsize};
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
#if HK_GEMM_RS_MI300X_TRACE
    std::uint64_t* const trace_ring =
        reinterpret_cast<std::uint64_t*>(g.trace);
    int trace_n = 0;
#endif
    HK_TRACE_STAMP(CTA_BEG);
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

        // The k-step is split into KH halves of KS columns each (exp_03/P3).
        // Only one half's operands are live at a time, which drops fragment
        // pressure from (WM+WN)*BK/32 to (WM+WN)*KS/32 registers -- 48 -> 24 on
        // the 256x256x32 rows, which are pinned at the 256 arch-VGPR cap.
        // Numerics are bit-exact: mma_ABt chains its k tiles ascending into the
        // same accumulator element, so splitting the k range does not reorder a
        // single addition.
        constexpr int KH = 2;
        constexpr int KS = BK / KH;
        static_assert(KS % 16 == 0 && KS * KH == BK);
        rt_bf<WM, KS, ducks::rt_layout::row> A_frag;
        rt_bf<WN, KS, ducks::rt_layout::row> B_frag;
        rt_fl<WM, WN, ducks::rt_layout::col> C_accum;

        // Register staging for the k+1 global->LDS copy (exp_03/E1(b)). These
        // live across the MFMA block, which is the whole point: the global
        // round trip is issued before the MFMAs and landed after them, so the
        // 64 MFMAs of iteration k cover it. Slot counts come from the same
        // formula the fused helper uses, so this is 2 float4 = 8 VGPRs per
        // operand on the 256-row configs and 1 float4 = 4 on the rest.
        using ST_A = st_bf<BM, BK>;
        using ST_B = st_bf<BN, BK>;
        constexpr int NT = G::GROUP_THREADS;
        float4 abuf[m3::g2s_slots<ST_A, NT>];
        float4 bbuf[m3::g2s_slots<ST_B, NT>];

        const int warp = kittens::warpid() % m3::NUM_WARPS;
        const int warp_row = warp / m3::WARPS_N;   // warps cover (WM x WN)
        const int warp_col = warp % m3::WARPS_N;

        const int num_pid_m = M / BM;
        const int num_pid_n = cols;
        const int tiles = num_pid_m * num_pid_n;
        const int k_iters = (K + BK - 1) / BK;
        const int tail_k = K_TAIL ? (K - (k_iters - 1) * BK) : BK;
        // WGM is the xGMI egress-link-concurrency knob, not the L2-locality one
        // it looks like. A producer CTA's whole tile lands on ONE peer --
        // dest = (tm*BM)/(M/8) -- so the set of `tm` values live across the
        // concurrently resident CTAs IS the set of egress links this rank is
        // using at that instant.
        //
        // When the tile loop runs more than one round, the donors' WGM = 4
        // makes in_group = 4*num_pid_n SMALLER than a round (128 vs 272 tiles
        // on 8192x8192x29568), so a round spans ~2 M-groups and therefore only
        // 2-3 of the 8 destinations: measured 117.44 MB of egress at 102 GB/s
        // against 2.02 effective links x ~47 GB/s = 95 GB/s, i.e. the links in
        // use were already saturated while five sat idle. WGM = num_pid_m makes
        // the decode column-major (group and first collapse to 0, gsize to
        // num_pid_m, so tm varies fastest), which puts all 8 destinations in
        // every round -- 7.53 effective links.
        //
        // When `tiles <= num_gemm_ctas` every tile is resident at once, so the
        // order cannot change egress concurrency at all and only its operand
        // locality is left. There WGM = 4 is better and column-major is a
        // measured regression, because column-major gives each XCD one A tile
        // per 8 pids and every B tile (2 + 16 distinct operand tiles per XCD on
        // 4096x4096x4096) where WGM = 4 gives it 4 + 8.
#if HK_GEMM_RS_MI300X_WGM4
        const int WGM = 4;
#else
        const int WGM = (tiles <= g.num_gemm_ctas) ? 4 : num_pid_m;
#endif
        // Coverage is unchanged and exact for either value: each tile index is
        // visited by exactly one producer CTA per launch, so every ready cell
        // is still published exactly once per rank per epoch.

        constexpr int RELEASE_GROUP = HK_GEMM_RS_MI300X_RELEASE_GROUP;
        static_assert(RELEASE_GROUP >= 1 && RELEASE_GROUP <= 8,
                      "RELEASE_GROUP is a hard-capped compile-time constant: an "
                      "unbounded group defers publication arbitrarily far on the "
                      "generic row, where a CTA owns up to 118 tiles");
        const int stride = g.num_gemm_ctas;
        const int bands = BM / EB;                 // EB | BM (gate 5)
        // Uniform across the grid: both terms come from the shape plan. The
        // compile-time cap still bounds rgroup, so the generic row's 118 tiles
        // per CTA group in fours exactly like the scored rows and nothing is
        // special-cased per config row.
        const int tiles_per_cta = (tiles + stride - 1) / stride;
#if HK_GEMM_RS_MI300X_RELEASE_GROUP_FULL_ONLY
#if HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE == 2
        // The ladder spelling. Every arm of this select is a compile-time
        // literal, so rgroup stays a member of {RELEASE_GROUP, 2, 1} exactly as
        // the incumbent kept it a member of {RELEASE_GROUP, 1} -- which is what
        // keeps the schedule and the register budget where they were. The
        // second rung is guarded by the cap so RELEASE_GROUP = 1 still collapses
        // to the behaviour-preserving control arm.
        const int rgroup =
            tiles_per_cta >= RELEASE_GROUP ? RELEASE_GROUP
            : ((RELEASE_GROUP >= 2 && tiles_per_cta >= 2) ? 2 : 1);
#elif HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE == 1
        // min(cap, tiles_per_cta), floored at 1. The floor is not defensive
        // decoration: rgroup == 0 makes `t0 += rgroup * stride` an infinite
        // loop, and it is the only value of this expression that does not
        // terminate. tiles >= 1 and stride >= 1 already give tiles_per_cta >= 1,
        // so the clamp costs one scalar op and removes the failure mode from
        // the reader's proof obligations entirely.
        const int rcap = tiles_per_cta < RELEASE_GROUP ? tiles_per_cta
                                                       : RELEASE_GROUP;
        const int rgroup = rcap > 1 ? rcap : 1;
#else
        const int rgroup = tiles_per_cta >= RELEASE_GROUP ? RELEASE_GROUP : 1;
#endif
#else
        const int rgroup = RELEASE_GROUP;
#endif

        // Grouping does not perturb coverage, because
        //   {t0 + j*NG : t0 in {pid, pid+G*NG, ...}, j in [0,G)} & [0,tiles)
        // is {pid + i*NG : i >= 0} & [0,tiles) -- the same tile set as the
        // ungrouped loop, in the same order, each visited once.
        for (int t0 = pid; t0 < tiles; t0 += rgroup * stride) {
            // This group's size, in closed form and BEFORE the tile body. A
            // counter incremented inside the body would be loop-carried
            // through the mainloop's register-critical region, and the 256x256
            // rows sit within 10 VGPRs of the 256 arch cap.
            //
            // t0 < tiles, so left >= 1 and emitted >= 1 on every iteration that
            // runs. That is what makes an unflushed tail structurally
            // impossible rather than a matter of vigilance: a zero-tile CTA
            // never enters this loop, so it can neither release nor publish,
            // and every group that emits reaches the release and the publish
            // below in the same iteration.
            const int left = (tiles - t0 + stride - 1) / stride;
            const int emitted = left < rgroup ? left : rgroup;

            for (int j = 0; j < emitted; ++j) {
                const int t = t0 + j * stride;
                // grouped tile order (same family as both donors), WGM as above.
                const tile_id id = decode_tile(t, num_pid_m, num_pid_n, WGM);
                const int tm = id.tm;
                const int tn = id.tn;

                HK_TRACE_STAMP(MAINLOOP_BEG);
                zero(C_accum);
                // ---- double-buffered mainloop, issue/commit split ----------------
                // Buffer lifetime, unchanged from the fused version: iteration k
                // ds_reads As[k&1] and ds_writes As[(k+1)&1]; (k+1)&1 == (k-1)&1, so
                // the buffer being written is the one iteration k-1 read, and the
                // __syncthreads() ending iteration k-1 separates them. Both of
                // iteration k's ds_read halves are drained (acquire_frags) before
                // that barrier, and iteration k's ds_writes are drained by
                // load_commit's trailing lgkmcnt(0), also before it.
                m3::load_issue<ST_A, NT>(abuf, g.a, {0, 0, tm, 0});
                m3::load_issue<ST_B, NT>(bbuf, g.b, {0, 0, tn, 0});
                m3::load_commit<NT>(As[0], abuf);
                m3::load_commit<NT>(Bs[0], bbuf);
                __syncthreads();
                // exp_09 E1(c): the prefetch is UNCONDITIONAL, with a clamped
                // source index, so the `if (more)` guards are gone.
                //
                // Those guards were the reason the k-loop body was six basic
                // blocks (ten for K_TAIL=true): one for the issue, one for the
                // MFMAs, one for the commit, plus the branches. A basic-block
                // boundary is a scheduling-region boundary -- the machine
                // scheduler cannot move an instruction across one, and neither can
                // sched_group_barrier, which is exactly why arm A0 (the builtin
                // alone, no restructure) changed not a single instruction of the
                // schedule. Straight-line, the four global_load_dwordx4 sit in the
                // same region as the 64 MFMAs and the address arithmetic can be
                // hoisted across the whole iteration.
                //
                // Peeling the last iteration instead was tried first (arm A1) and
                // is a hard fail: duplicating the MFMA block put both copies'
                // fragment addresses in one live range and cost 30 VGPR spills /
                // 50 scratch stores on the 256x256 rows, which sit at 246 of 256.
                //
                // Why the clamp is safe. On the last iteration `kn == k`, so the
                // prefetch re-reads the tile this iteration is already consuming:
                // a valid, in-bounds global read whose only effect is to land in
                // As[(k+1)&1] / Bs[(k+1)&1]. That is the buffer iteration k-1 read
                // and nothing reads it again -- the loop is over, and the epilogue
                // does not touch A/B LDS until after its own __syncthreads(). The
                // steady-state buffer invariant is untouched: iteration k ds_reads
                // As[k&1] and ds_writes As[(k+1)&1], (k+1)&1 == (k-1)&1, and the
                // __syncthreads() ending k-1 separates the two. Cost is one extra
                // tile read per tile, against k_iters of 116..924.
                for (int k = 0; k < k_iters; ++k) {
                    // ISSUE only -- no vmcnt here, that is the entire mechanism.
                    // Clamped, not guarded: a select, not a branch.
                    const int kn = (k + 1 < k_iters) ? (k + 1) : k;
                    m3::load_issue<ST_A, NT>(abuf, g.a, {0, 0, tm, kn});
                    m3::load_issue<ST_B, NT>(bbuf, g.b, {0, 0, tn, kn});
                    // exp_09 arm B: raise this wave's issue priority for the operand
                    // reads and the MFMA block, drop it for the commit. At 512
                    // threads and 1 CTA/CU there are 2 waves per SIMD competing for
                    // one issue port, so a wave grinding through ds_writes, waits
                    // and address arithmetic can starve its neighbour's MFMAs. The
                    // gfx950 donor does the same (gemm_rs_device_tile.cpp:884-935).
                    //
                    // The window is airtight without any scheduling hint, which
                    // matters because hints do not work here (arms A0, A2). MFMAs
                    // cannot leave through the top: they depend, through
                    // acquire_frags' anchor, on ds_reads that follow this setprio,
                    // and both setprio and the ds_read asm have side effects so
                    // their order is fixed. They cannot leave through the bottom
                    // either: acc_anchor ties the accumulator below them.
                    __builtin_amdgcn_s_setprio(1);
                    #pragma unroll
                    for (int kh = 0; kh < KH; ++kh) {
                        load(A_frag, subtile_inplace<WM, KS>(As[k & 1],
                                                             {warp_row, kh}));
                        load(B_frag, subtile_inplace<WN, KS>(Bs[k & 1],
                                                             {warp_col, kh}));
                        // Mandatory: every ds_read in this tree is inside an
                        // asm volatile, so SIInsertWaitcnts never observes the LDS
                        // event and emits no use-wait of its own. Nothing else
                        // between these reads and the MFMAs waits on lgkmcnt. The
                        // anchor is equally mandatory -- a bare wait is reorderable
                        // past an MFMA and was, silently, on 14 of 17 shapes.
                        m3::acquire_frags(A_frag, B_frag);
                        // The A operand carries the only mask: whatever lands in
                        // the K-padding columns of A is overwritten with exact
                        // zeros before the MFMA (gates 3, 15). The threshold is
                        // rebased into this half's local K coordinates; a half that
                        // starts at or past the tail gets threshold 0, i.e. is
                        // zeroed outright.
                        if constexpr (K_TAIL) {
                            if (k == k_iters - 1) {
                                const int th = tail_k - kh * KS;
                                mask_a_k_tail(A_frag, th > 0 ? th : 0);
                            }
                        }
                        mma_ABt(C_accum, A_frag, B_frag, C_accum);
                    }
                    // Keep every MFMA ABOVE the commit's vmcnt(0).
                    //
                    // Straight-lining the body (above) cost prefetch coverage, and
                    // the ISA says so precisely: with the commit in its own basic
                    // block the block boundary forced all 64 MFMAs to issue before
                    // the vmcnt(0), but in one region the scheduler sank 31 of
                    // them below it -- 21 even below the __syncthreads() -- leaving
                    // the global load only 33 MFMAs of cover instead of 64. The
                    // volatile asm chain cannot move (that is what pins the
                    // vmcnt(0)), so the MFMAs are what has to be held.
                    //
                    // sched_barrier(0x7F6) -- every class except MFMA allowed to
                    // cross -- was tried here first (arm A2) and did nothing: the
                    // vmcnt(0) still landed after 33 MFMAs. The anchor below is a
                    // data dependence on the accumulator instead of a hint, which
                    // is the mechanism that already works in this TU.
                    m3::acc_anchor(C_accum);
                    __builtin_amdgcn_s_setprio(0);
                    // COMMIT: vmcnt(0) lands what was issued before the MFMAs, then
                    // the ds_writes publish it and lgkmcnt(0) drains them.
                    m3::load_commit<NT>(As[(k + 1) & 1], abuf);
                    m3::load_commit<NT>(Bs[(k + 1) & 1], bbuf);
                    __syncthreads();
                }
                HK_TRACE_STAMP(MAINLOOP_END);

                // ---- egress: per-band credit wait, then emit ---------------
                // The credit wait STAYS PER TILE and stays before that tile's
                // first payload store; it is what makes overwriting an
                // unconsumed slot impossible, and hoisting it to the group
                // would grant permission for tile j+1 before tile j had it. A
                // failed wait still returns immediately: the pending group is
                // abandoned unreleased AND unpublished, which is fail-closed --
                // its reducers time out on bit 26 rather than read a slot
                // nobody finished writing.
                //
                // Every band of this tile must be reusable before ANY payload
                // store of epoch ep: waits on leader lanes only, result broadcast
                // by the CTA barrier + shared error bit (fail-closed).
                HK_TRACE_STAMP(CREDIT_WAIT_BEG);
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
                HK_TRACE_STAMP(CREDIT_WAIT_END);

                const bf16* biasp = reinterpret_cast<const bf16*>(g.bias);
                const int win_rows = EB < 32 ? EB : 32;      // EB % win_rows == 0
                const int vt = N - tn * BN < BN ? N - tn * BN : BN;   // <= BN

                HK_TRACE_STAMP(EMIT_BEG);
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
                HK_TRACE_STAMP(EMIT_END);

            }   // end of the group's tile loop

            // ONE release for the whole group. Whole-CTA convergent and outside
            // any divergent region (it contains two __syncthreads()), and it
            // covers every payload store issued above by every thread to every
            // destination: vmcnt is destination-agnostic and retires in order
            // on gfx9, so a store issued three tiles ago is drained exactly as
            // well as one issued three instructions ago, and buffer_wbl2
            // sc0 sc1 writes back the whole L2 of the executing XCD -- the only
            // L2 a workgroup's threads can have dirtied. Batching therefore
            // adds no hardware assumption; it applies the per-tile release's
            // own properties to a larger set of stores.
            //
            // The publications that follow are RELAXED stores by design. They
            // carry no ordering of their own and are correct only because this
            // release dominates them, which is why coarsening the release
            // without coarsening the publication is a protocol error rather
            // than a smaller version of this change.
            HK_TRACE_STAMP(RELEASE_BEG);
#if HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS
            // PUBLISH-EARLY control: announce readiness BEFORE the release that
            // covers the payload. A reducer may then acquire a slot whose bytes
            // are still in flight or still dirty in this XCD's L2, and its
            // buffer_inv pulls the PREVIOUS epoch's line -- which is the right
            // answer whenever the inputs did not change, so no tolerance can
            // see it. m9_stale_slot poisons every rank's heap with bf16 NaN to
            // make it visible; if this control does not fail that gate, the
            // gate has no power over the property batching puts at risk.
            //
            // ctrl_flags is a kernel argument and therefore CTA-uniform, so the
            // release stays convergent on both paths and exactly one of the two
            // sites executes per group.
            const bool publish_early =
                (g.ctrl_flags & m3::CTRL_PUBLISH_EARLY) != 0u;
            if (!publish_early) m3::release_payload_system();
#else
            m3::release_payload_system();
#endif
            HK_TRACE_STAMP(RELEASE_END);

            // ALL of the group's publications, after that single release: the
            // leader publishes each (tile, band)'s cheap completion cell on its
            // destination rank (publication separate from movement, F6-style).
            // Every tile is named by decode_tile -- the same helper the emit
            // loop above used -- so an emit/publish index drift cannot happen
            // without editing one shared expression, and nothing about the
            // group is journalled in LDS or in registers.
            if (threadIdx.x == 0) {
                for (int j = 0; j < emitted; ++j) {
                    const tile_id id =
                        decode_tile(t0 + j * stride, num_pid_m, num_pid_n, WGM);
                    for (int b = 0; b < bands; ++b) {
                        const int row0 = id.tm * BM + b * EB;
                        const int dest = m3::owner_of_row(row0, slice);
                        const int lrow = (row0 - dest * slice) / EB;
#if HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS
                        // DROP-PUBLICATION control: this consumer MUST time out
                        // without reading. Production cannot see this code.
                        if ((g.ctrl_flags & m3::CTRL_DROP_PUBLICATION) &&
                            me == g.ctrl_rank && dest == g.ctrl_arg0) {
                            continue;
                        }
#endif
                        m3::publish_band_epoch(
                            sig, sig_peers, dest, me,
                            m3::SIGNAL_GUARD_U32 +
                                m3::ready_idx(me, lrow, id.tn, lrows, cols),
                            ep);
                    }
                }
            }
            HK_TRACE_STAMP(PUBLISH_END);
#if HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS
            if (publish_early) m3::release_payload_system();
#endif
        }
        HK_TRACE_STAMP(CTA_END);
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
        HK_TRACE_STAMP(READY_WAIT_BEG);
        if (threadIdx.x < m3::WORLD_SIZE) {
            const bool ok = m3::wait_band_epoch(
                sig + m3::SIGNAL_GUARD_U32 +
                    m3::ready_idx((int)threadIdx.x, lr, col, lrows, cols),
                ep, spin, errp, m3::ERR_REDUCER_READY);
            if (!ok) atomicOr(errp, m3::ERR_REDUCER_READY);
        }
        __syncthreads();
        if (m3::error_bit_set(errp, m3::ERR_REDUCER_READY)) return;
        HK_TRACE_STAMP(READY_WAIT_END);
        m3::acquire_payload_system();
        HK_TRACE_STAMP(REDUCE_BEG);

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
        HK_TRACE_STAMP(REDUCE_END);

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
        HK_TRACE_STAMP(CREDIT_PUB_END);
    }
    HK_TRACE_STAMP(CTA_END);
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
        // Rows 1-3 retiled by exp_14 (E4b); see the shape table's comment for
        // the model and the per-row evidence. Rows 4-6 are the enumerated
        // argmin of their shapes and row 6 sits at the wave-cost floor.
        case 1: launch_fixed< 32,  64, 128, false>(g); return;
        case 2: launch_fixed< 64, 128,  64, false>(g); return;
        case 3: launch_fixed<128, 192,  32,  true>(g); return;
        case 4: launch_fixed<256, 256,  32, false>(g); return;
        case 5: launch_fixed<256, 256,  32, false>(g); return;
        case 6: launch_fixed<256, 256,  32,  true>(g); return;

        // exp_14 (E4b) screened seven more tiles through config rows 101-107.
        // Those rows are deliberately NOT here: gate 16 of
        // gemm_rs_mi300x_static_checks.py counts the scored dispatch arms and
        // requires exactly six, and a screening convenience is not worth
        // weakening a protocol invariant check. The block is preserved verbatim,
        // with the command to re-apply it, in
        // experiments/exp_14_tile_waves/sweep_rows.patch.txt.

        // Generic correctness row: bounded paths carry the tails.
        default:
            if (g.even_k) launch_fixed<32, 64, 64, false>(g);
            else          launch_fixed<32, 64, 64,  true>(g);
            return;
    }
}

// ================================================================================================
// Prebound launch path (exp_12). Same POD, same dispatch, same one launch per
// call; the only thing that differs is where the arguments come from.
//
// Why it exists. Under the graded protocol (barrier -> call -> synchronize ->
// barrier) nothing overlaps the host path, so every microsecond spent in python
// and pybind is charged to the score. pyutils' gl<> converter re-derives each
// tensor argument from a python object on every call: __class__.__name__,
// is_contiguous(), device.type, shape and data_ptr(), i.e. ~8 attribute
// lookups, two python calls and two std::string comparisons per tensor, four
// tensors deep. Measured in an 8-rank pool on this node that is ~7.5 us of the
// ~23 us the host path costs, and all of it recomputes values that cannot
// change between calls on a cached (rank, shape) state.
//
// So the invariant part is bound once, from the same place hk_submission builds
// its descriptors, and the call site passes only what actually changes: the
// three input pointers and the stream. The `gemm_rs_mi300x` binding below is
// untouched -- the single-process validation harness still drives it, and the
// two paths must construct byte-identical mi300x_globals.
// ================================================================================================
namespace prebound {

// One table per process, indexed by an opaque handle the caller stores next to
// the allocation it describes. Deliberately NOT a cache keyed on shape: a
// module-global keyed on nothing is precisely the hazard that makes the
// competitor's cached launch path raise on its second call with a new shape.
// The GIL serializes every access, so no lock is needed.
static std::vector<mi300x_globals>& slots() {
    static std::vector<mi300x_globals> table;
    return table;
}

static int configure(
        int a_rows, int a_cols, int b_rows, int b_cols,
        std::uintptr_t c_heap, int c_batch, int c_rows, int c_cols,
        std::uintptr_t out, int out_rows, int out_cols,
        std::uintptr_t c_heap_peers, std::uintptr_t sig,
        std::uintptr_t sig_peers, std::uintptr_t ep_cell, std::uintptr_t err,
        std::uintptr_t stream_ptr, std::uint64_t spin_limit,
        int me, int M, int N, int K, int lrows, int cols, int eb,
        int num_gemm_ctas, int ready_words, int packet_fast_path,
        int config_row, int even_k) {
    mi300x_globals g {
        make_gl<_gl_A>(0, 1, 1, a_rows, a_cols),
        make_gl<_gl_B>(0, 1, 1, b_rows, b_cols),
        make_gl<_gl_C>((std::uint64_t)c_heap, c_batch, 1, c_rows, c_cols),
        make_gl<_gl_C>((std::uint64_t)out, 1, 1, out_rows, out_cols),
        c_heap_peers, sig, sig_peers, ep_cell, /*bias=*/0, err,
        stream_ptr, spin_limit, me, M, N, K, lrows, cols, eb,
        num_gemm_ctas, ready_words, packet_fast_path,
        /*ctrl_flags=*/0u, /*ctrl_rank=*/0, /*ctrl_arg0=*/0,
        config_row, even_k,
    };
    slots().push_back(g);
    return (int)slots().size() - 1;
}

static void run(int handle, std::uintptr_t a, std::uintptr_t b,
                std::uintptr_t bias, std::uintptr_t stream_ptr) {
    if (handle < 0 || handle >= (int)slots().size()) {
        throw std::runtime_error("GEMM-RS mi300x: bad prebound handle " +
                                 std::to_string(handle));
    }
    mi300x_globals g = slots()[(std::size_t)handle];
    // Only the three per-call inputs and the stream move. Writing raw_ptr is
    // the whole of what make_gl would do here: gl<bf16,-1,-1,-1,-1> carries no
    // TMA descriptors on gfx942, so its constructor is a field copy.
    g.a.raw_ptr = reinterpret_cast<bf16*>(a);
    g.b.raw_ptr = reinterpret_cast<bf16*>(b);
    g.bias = bias;
    if (stream_ptr != 0) g.stream_ptr = stream_ptr;
    dispatch_gemm_rs_mi300x(g);
}

}  // namespace prebound

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
#if HK_GEMM_RS_MI300X_TRACE
        // exp_22 diagnostic: ONE extra trailing argument, so the flag-ON
        // module takes 28 where production takes 27. The exported symbol name
        // is deliberately unchanged, so the harness's calling convention is
        // otherwise untouched and the driver only appends a pointer.
        &mi300x_globals::even_k, &mi300x_globals::trace);
#else
        &mi300x_globals::even_k);
#endif
    // The prebound pair. Still exactly one kernel launch per call (gate 16),
    // still the same dispatch table and the same POD; see the comment above
    // namespace prebound for why the duck-typed path is too expensive to keep
    // on the graded critical path.
    m.def("gemm_rs_mi300x_configure", &prebound::configure,
          "bind everything invariant for one (rank, shape); returns a handle");
    m.def("gemm_rs_mi300x_run", &prebound::run,
          "launch a configured handle with (a, b, bias, stream)");
#endif
}

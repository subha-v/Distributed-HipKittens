/**
 * @file
 * @brief G25-1 rig: one TP8 layer boundary — fused CDAR vs phased CDAR vs
 *        compute-then-RCCL (production's schedule), with a real MFMA body.
 *
 * The question this answers (M25_TP8_MEGA_DESIGN.md gate G25-1): does hiding
 * the boundary collective inside a matrix-core-active producer beat running
 * the same collective as an exposed phase — and at what co-residency cost?
 *
 * ARMS (argv mode):
 *   compute : MFMA loop only — the compute floor A.
 *   phased  : MFMA loop, then the full CDAR as a separate phase — A + C
 *             sequential (the unfused control; isolates transport cost).
 *   fused   : MFMA loop with ERS commits interleaved per slab + owner/MAG
 *             overlap — the M25 schedule. Overlap win = (phased - fused).
 *             Co-residency tax = fused_compute_span vs compute (derived:
 *             fused - hidden_CDAR_share; report walls, let the reader do
 *             arithmetic from the printed A/B/C).
 *   rccl    : MFMA loop kernel, then ncclAllReduce(bf16 sum) enqueued on the
 *             same stream — production's GEMM -> collective schedule at the
 *             identical logical boundary (235 MB at 16K tokens).
 *
 * G-L0 ARMS ADDED (all host-side; the device .text is untouched at
 * M25_LEDGER=0 / M25_NOCOMPUTE=0 — see LEDGER_NOTES.md for the sha receipts):
 *   transport       : the phased schedule with k_inner forced to 0 — the same
 *             CDAR all-reduce, ZERO MFMA work. This is direct(ours) for
 *             beta, and it is what makes I_co ABSOLUTE rather than only
 *             differential: I_co(phased) = phased - compute - transport.
 *   transport_fused : the fused schedule with k_inner forced to 0 (the
 *             co-residency-free control for the fused protocol's own cost).
 *   compute0: the compute arm at k_inner=0 — the residual control that BOUNDS
 *             what "zero MFMA work" still costs (the two operand loads that
 *             survive a runtime-zero trip count). Build with
 *             -DM25_NOCOMPUTE=1 for the hard compile-out variant; compute0
 *             must then read ~0 on both.
 *   stdrccl : a bare, timed ncclAllReduce at the identical 235 MB payload in
 *             the SAME binary — direct(rccl) for beta, no subtraction. Honors
 *             NCCL_ALGO / NCCL_PROTO / NCCL_*NCHANNELS (printed per run).
 *   beta    : alternates transport and stdrccl within one invocation (LAW-60:
 *             cross-session lone-arm contrasts are void) and prints
 *             beta = median(transport) / median(stdrccl) directly.
 *
 * rho LADDER: `k_inner` (argv 6) scales ONLY the MFMA loop's trip count, so
 * rho = compute:transport is settable without a rebuild. Caveat for R5/Q4:
 * k_inner scales FLOPs and operand-load traffic together (one rotating load
 * per iteration) — intensity and duration are NOT independent here; the
 * `bytes_per_mfma` argv that separates them is R5's build item, not G-L0's.
 *
 * INSTRUMENT (E-B1/E-B2, -DM25_LEDGER=1): a per-CTA raw event ring in the
 * 100 MHz realtime domain (m25_cdar.cuh), reduced host-side into per-phase
 * spans, span-overlap h_ledger, and a per-layer fabric duty histogram, with
 * an explicit cross-rank RANKMAX row. Default build is byte-identical.
 *
 * HONESTY LABELS:
 *  - The MFMA body is a representative matrix-core load (dense
 *    mfma_f32_16x16x32_bf16 K-loop over the fragment data at occupancy 1),
 *    NOT the M15/M25 GEMM. It contends for the same resources a real
 *    epilogue would (VGPRs, vmcnt counter, CU issue), which is what the
 *    co-residency question needs; absolute FLOP/s is not the claim.
 *  - CDAR arms verify exact bf16 sums (rank+1 summed = 36.0); the rccl arm
 *    verifies the same value host-side. A faster wrong collective is a
 *    zero.
 *  - Depth matters again here (co-residency) — it is argv and should be
 *    re-swept in fused mode.
 *
 * Build (node): hipcc --offload-arch=gfx950 -std=c++20 -O3 -I <root> \
 *   m25_boundary_bench.hip -o m25_boundary_bench -lrccl
 *   (ledger arm: add -DM25_LEDGER=1 -o m25_boundary_bench_ledger)
 * Run: ./m25_boundary_bench <mode> [tokens=16384] [slab_rows=256]
 *      [iters=7] [depth=4] [k_inner=64] [bps=2]
 */

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rccl/rccl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "m25_cdar.cuh"

// ---- E-B1/E-B2 plumbing --------------------------------------------------
// Every device-side addition in this file is behind M25_LEDGER. With the flag
// at 0 the preprocessed device translation unit is textually identical to the
// pre-instrument source, which is what the .text sha parity gate asserts.
#ifndef M25_NOCOMPUTE
#define M25_NOCOMPUTE 0
#endif

#if M25_LEDGER
// Ring capacity per CTA per launch slot. Worst case in the shipped matrix is
// ~5 producer items x 3 events + duty 2 x 4 + consume 1 x 4 + 4 loop/kernel
// spans; 128 leaves an order of magnitude of headroom and any overflow is
// COUNTED (high 16 bits of count[]) and fails the run loudly.
static constexpr unsigned int kLedgerCap = 256u;
// Launch slots: the compute and rccl arms cross the boundary in TWO launches
// (stage 1, stage 2). Slots keep their timestamps apart — LAW-63(e) forbids
// subtracting stamps written by different launches.
static constexpr unsigned int kLedgerLaunches = 2u;
static constexpr unsigned int kDutyHistBins = 64u;

#define M25_LG_PARAM , cd::ledger_view lgv
#define M25_LG_ARG , lgv
// Every stamp is read by EVERY thread (the value is CTA-uniform, so it lands
// in SGPRs); only the 24 B ring write is predicated on thread 0. The brackets
// sit OUTSIDE every poll body and outside every store loop.
// The entry and exit stamps are recorded as their OWN degenerate events
// rather than held in a live register pair across the whole kernel — one less
// long-lived u64 in an already-tight scalar file.
#define M25_LG_OPEN()                                                         \
    cd::ledger_cursor lgc = cd::ledger_open(lgv, blockIdx.x);                 \
    {                                                                         \
        const std::uint64_t lg_e = cd::realtime_now();                        \
        cd::ledger_mark(lgc, cd::lg_kernel, lg_e, lg_e, 0u,                   \
                        threadIdx.x == 0u);                                   \
    }
#define M25_LG_T0(v) const std::uint64_t v = cd::realtime_now();
#define M25_LG_MARK(v, code, aux)                                             \
    cd::ledger_mark(lgc, (code), (v), cd::realtime_now(),                     \
                    static_cast<std::uint32_t>(aux), threadIdx.x == 0u)
#define M25_LG_EXIT()                                                         \
    do {                                                                      \
        const std::uint64_t lg_x = cd::realtime_now();                        \
        cd::ledger_mark(lgc, cd::lg_kernel_end, lg_x, lg_x, 0u,               \
                        threadIdx.x == 0u);                                   \
        if (threadIdx.x == 0u) cd::ledger_close(lgv, blockIdx.x, lgc);        \
    } while (0)
// launch-slot views passed at the two call sites in main()
#define M25_LG_A0(r) , lv0[r]
#define M25_LG_A1(r) , lv1[r]
#else
#define M25_LG_PARAM
#define M25_LG_ARG
#define M25_LG_OPEN()
#define M25_LG_T0(v)
#define M25_LG_MARK(v, code, aux)
#define M25_LG_EXIT()
#define M25_LG_A0(r)
#define M25_LG_A1(r)
#endif

namespace kd = kittens::distributed;
namespace cd = kittens::distributed::cdar;

static constexpr unsigned int kWorld = 8;
static constexpr unsigned int kRowBytes = 14336;
static constexpr unsigned int kBlocks = 256;
static constexpr unsigned int kThreads = 256;
static constexpr unsigned int kDutyBlocks = 32;
static constexpr unsigned int kReduceSegs = 8;

static constexpr std::uint64_t kSpin = 1u << 20;
static constexpr unsigned int kAttempts = 512;

#define HIP_CHECK(x)                                                          \
    do {                                                                      \
        hipError_t err_ = (x);                                                \
        if (err_ != hipSuccess) {                                             \
            std::fprintf(stderr, "HIP error %s at %s:%d\n",                   \
                         hipGetErrorString(err_), __FILE__, __LINE__);        \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define NCCL_CHECK(x)                                                         \
    do {                                                                      \
        ncclResult_t nres_ = (x);                                             \
        if (nres_ != ncclSuccess) {                                           \
            std::fprintf(stderr, "NCCL error %s at %s:%d\n",                  \
                         ncclGetErrorString(nres_), __FILE__, __LINE__);      \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

struct rank_report {
    unsigned int gather_timeouts;
    unsigned int mag_timeouts;
    unsigned int verify_errors;
    unsigned int slabs_certified;
    float acc_token;  // anti-DCE sink for the MFMA accumulators
};

__global__ void fill_bf16(std::byte* buf, std::size_t bytes,
                          unsigned int packed) {
    auto* out = reinterpret_cast<unsigned int*>(buf);
    const std::size_t n = bytes >> 2;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        out[i] = packed;
    }
}

#if defined(__HIP_DEVICE_COMPILE__)
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float floatx4 __attribute__((ext_vector_type(4)));

// One MFMA burst over a fragment: dense matrix-core work consuming the
// fragment's bytes as operands. k_inner controls FLOP density per slab —
// the knob that sets the compute shadow the collective must hide under.
__device__ __forceinline__ float mfma_burst(const std::byte* frag,
                                            unsigned int rows,
                                            unsigned int k_inner,
                                            unsigned int tid) {
#if M25_NOCOMPUTE
    // Transport-only build: the compute body is COMPILED OUT (no MFMA, no
    // operand loads at all). The same measurement is available in the default
    // binary as `mode=transport` (k_inner=0); this arm exists so the residue
    // of that runtime-zero trip count can be proven to be nothing.
    (void)frag; (void)rows; (void)k_inner; (void)tid;
    return 0.f;
#else
    const auto* v = reinterpret_cast<const bf16x8*>(frag);
    const std::size_t vecs =
        static_cast<std::size_t>(rows) * (kRowBytes / 16);
    floatx4 acc = {0.f, 0.f, 0.f, 0.f};
    bf16x8 a = v[tid % vecs];
    bf16x8 b = v[(tid * 7u + 13u) % vecs];
#pragma unroll 4
    for (unsigned int k = 0; k < k_inner; ++k) {
        acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, acc, 0, 0, 0);
        // rotate one operand through the fragment so loads stay live
        a = v[(tid + k * 17u) % vecs];
    }
    return acc[0] + acc[1] + acc[2] + acc[3];
#endif
}
#else
__device__ __forceinline__ float mfma_burst(const std::byte*, unsigned int,
                                            unsigned int, unsigned int) {
    return 0.f;
}
#endif

enum bench_mode : int { mode_compute = 0, mode_phased = 1, mode_fused = 2 };

// stage: 0 = full boundary crossing (burst1 [+ transport] + consume + burst2
// on each consumed slab); 1 = burst1 only; 2 = burst2 only over `staging`
// (the split lets the rccl arm interpose its AllReduce between two launches
// while fused/phased cross the boundary inside one kernel).
template<int Mode, unsigned int Depth>
__global__ __launch_bounds__(kThreads, 1) void boundary_kernel(
        cd::peer_view<kWorld> pv, unsigned int tokens, unsigned int slab_rows,
        unsigned int rank, std::uint32_t epoch, unsigned int k_inner,
        unsigned int stage, unsigned int bps, std::uint32_t* reduce_cells,
        const std::byte* __restrict__ src, std::byte* __restrict__ staging,
        rank_report* report, unsigned int expected_bits M25_LG_PARAM) {
    M25_LG_OPEN();
    const cd::geometry<kWorld> g{tokens, kRowBytes, slab_rows, rank};
    const unsigned int nslabs = g.num_slabs();
    const unsigned int block = blockIdx.x;
    const unsigned int base = (rank + 1u) % kWorld;
    float acc_sink = 0.f;
    bool prev_valid = false;
    unsigned int prev_slab = 0u;
    unsigned int prev_rows = 0u;

    if (stage == 2u) {
        // burst2 only: the post-boundary GEMM over already-reduced staging.
        M25_LG_T0(lg_s2);
        for (unsigned int slab = block; slab < nslabs; slab += kBlocks) {
            const std::byte* dst =
                staging + static_cast<std::size_t>(slab) * g.slab_bytes();
            M25_LG_T0(lg_m2);
            acc_sink += mfma_burst(dst, g.rows_in(slab), k_inner, threadIdx.x);
            M25_LG_MARK(lg_m2, cd::lg_mfma2, g.rows_in(slab));
        }
        M25_LG_MARK(lg_s2, cd::lg_cons_loop, 0u);
        if (threadIdx.x == 0u) atomicAdd(&report->acc_token, acc_sink);
        M25_LG_EXIT();
        return;
    }

    // ---- producer: MFMA burst per work item, ERS commit fused or deferred.
    // In fused mode the duty pool (blocks [0, kDutyBlocks)) produces NOTHING:
    // it enters gather immediately and consumes certified slabs in the
    // producers' shadow — the M15 consuming-pool economics. Producers fall
    // through into consumption after their items (also M15).
    const bool producer_block =
        (Mode != mode_fused) || (block >= kDutyBlocks) || (stage == 1u);
    const unsigned int pstride =
        (Mode == mode_fused && stage != 1u) ? (kBlocks - kDutyBlocks)
                                            : kBlocks;
    const unsigned int pfirst =
        (Mode == mode_fused && stage != 1u) ? (block - kDutyBlocks) : block;
    const unsigned int items = nslabs * bps;
    M25_LG_T0(lg_prod);
    for (unsigned int k = pfirst; producer_block && k < items; k += pstride) {
        const unsigned int slab = (base + k / bps) % nslabs;
        const unsigned int fragment = k % bps;
        const unsigned int rows_total = g.rows_in(slab);
        const unsigned int rows_per = (rows_total + bps - 1u) / bps;
        const unsigned int r0 = fragment * rows_per;
        const unsigned int rows =
            r0 >= rows_total ? 0u
                             : (rows_per < rows_total - r0 ? rows_per
                                                           : rows_total - r0);
        const std::byte* frag =
            src + static_cast<std::size_t>(slab) * g.slab_bytes() +
            static_cast<std::size_t>(r0) * kRowBytes;
        if (rows > 0u) {
            M25_LG_T0(lg_m1);
            acc_sink += mfma_burst(frag, rows, k_inner, threadIdx.x);
            M25_LG_MARK(lg_m1, cd::lg_mfma1, rows);
        }
        if constexpr (Mode == mode_fused) {
            // SOFTWARE-PIPELINED commit (v2): the previous item's stores
            // drained under THIS item's MFMA burst (the wave's in-order
            // vmcnt counter holds at most Depth of them past each
            // throttled store, so the burst's own operand loads wait
            // behind <=Depth stores — the co-residency contract depth-4
            // was measured for). Arrive for the PREVIOUS item now (its
            // stores are certainly retired: they precede this burst), then
            // commit this item WITHOUT a drain and move on.
            if (prev_valid) {
                M25_LG_T0(lg_dr);
                kd::throttle_vmcnt<0>();
                __syncthreads();
                if (threadIdx.x == 0u) {
                    cd::ers_local_arrive(g, pv, prev_slab, bps, epoch,
                                         prev_rows);
                }
                __syncthreads();
                M25_LG_MARK(lg_dr, cd::lg_drain, prev_slab);
            }
            if (rows > 0u) {
                // WRITE-ONCE FROM REGISTERS (v3): the M15 epilogue lesson —
                // the producer's OUTPUT WRITE is the transport. No reload of
                // the fragment: each lane stores a register-resident 16-byte
                // packet (bench artifact: the value is uniform rank+1, which
                // keeps the exact-sum verification while modeling the real
                // register->remote store profile). Posted stores drain under
                // the next item's MFMA burst; Depth bounds the queue the
                // burst's operand loads wait behind.
                const unsigned int owner = g.owner_of(slab);
                const std::size_t owned_stride =
                    ((nslabs + kWorld - 1u) / kWorld) * g.slab_bytes();
                std::byte* dst = pv.towers[owner] +
                    static_cast<std::size_t>(g.rank) * owned_stride +
                    static_cast<std::size_t>(slab / kWorld) * g.slab_bytes() +
                    static_cast<std::size_t>(r0) * kRowBytes;
                const kd::packet16 reg_packet =
                    *reinterpret_cast<const kd::packet16*>(frag);
                auto* out = reinterpret_cast<kd::packet16*>(dst);
                const std::size_t packets =
                    (static_cast<std::size_t>(rows) * kRowBytes) >> 4;
                M25_LG_T0(lg_cm);
#pragma unroll 1
                for (std::size_t p = threadIdx.x; p < packets;
                     p += blockDim.x) {
                    kd::throttled_store_packet16<Depth>(out + p, reg_packet);
                }
                M25_LG_MARK(lg_cm, cd::lg_commit,
                            static_cast<std::size_t>(rows) * kRowBytes);
                // no drain: the next iteration's mfma_burst is the shadow
            }
            prev_valid = true;
            prev_slab = slab;
            prev_rows = rows;
        }
    }
    if constexpr (Mode == mode_fused) {
        if (prev_valid) {
            M25_LG_T0(lg_dt);
            kd::throttle_vmcnt<0>();
            __syncthreads();
            if (threadIdx.x == 0u) {
                cd::ers_local_arrive(g, pv, prev_slab, bps, epoch,
                                     prev_rows);
            }
            __syncthreads();
            M25_LG_MARK(lg_dt, cd::lg_drain, prev_slab);
        }
    }
    M25_LG_MARK(lg_prod, cd::lg_prod_loop, producer_block ? 1u : 0u);

    if constexpr (Mode == mode_phased) {
        // The unfused control: identical transport, exposed as a phase.
        M25_LG_T0(lg_tx);
        for (unsigned int k = block; k < items; k += kBlocks) {
            const unsigned int slab = (base + k / bps) % nslabs;
            const unsigned int fragment = k % bps;
            const unsigned int rows_total = g.rows_in(slab);
            const unsigned int rows_per = (rows_total + bps - 1u) / bps;
            const unsigned int r0 = fragment * rows_per;
            const unsigned int rows =
                r0 >= rows_total
                    ? 0u
                    : (rows_per < rows_total - r0 ? rows_per
                                                  : rows_total - r0);
            if (rows > 0u) {
                M25_LG_T0(lg_pc);
                const std::byte* frag =
                    src + static_cast<std::size_t>(slab) * g.slab_bytes() +
                    static_cast<std::size_t>(r0) * kRowBytes;
                cd::ers_commit_fragment<cd::transport::store_towers, Depth>(
                    g, pv, slab, r0, rows, frag, threadIdx.x, blockDim.x);
                kd::throttle_vmcnt<0>();
                M25_LG_MARK(lg_pc, cd::lg_commit,
                            static_cast<std::size_t>(rows) * kRowBytes);
            }
            M25_LG_T0(lg_pa);
            __syncthreads();
            if (threadIdx.x == 0u) {
                cd::ers_local_arrive(g, pv, slab, bps, epoch, rows);
            }
            __syncthreads();
            M25_LG_MARK(lg_pa, cd::lg_drain, slab);
        }
        M25_LG_MARK(lg_tx, cd::lg_tx_loop, 0u);
    }

    if (stage == 1u) {
        if (threadIdx.x == 0u) atomicAdd(&report->acc_token, acc_sink);
        M25_LG_EXIT();
        return;
    }

    if constexpr (Mode != mode_compute) {
        // ---- owner duty (gather + parallel tower reduce + certify)
        if (block < kDutyBlocks) {
            const unsigned int owned =
                (nslabs > rank ? (nslabs - 1u - rank) / kWorld + 1u : 0u);
            const unsigned int duty_items = owned * kReduceSegs;
            M25_LG_T0(lg_duty);
            for (unsigned int k = block; k < duty_items; k += kDutyBlocks) {
                const unsigned int slab = rank + (k / kReduceSegs) * kWorld;
                const unsigned int seg = k % kReduceSegs;
                cd::owner_gather og{};
                // LAW-63: the bracket is OUTSIDE the poll body — one pair of
                // realtime reads per wait, never one per spin.
                M25_LG_T0(lg_gw);
                if (threadIdx.x == 0u) {
                    unsigned int tries = 0u;
                    do {
                        cd::owner_gather_ready_into(g, pv, slab, epoch, kSpin,
                                                    og);
                    } while (!og.ready && ++tries < kAttempts);
                    if (!og.ready) atomicAdd(&report->gather_timeouts, 1u);
                }
                M25_LG_MARK(lg_gw, cd::lg_gather_wait, slab);
                __syncthreads();
                M25_LG_T0(lg_rd);
                {
                    const std::size_t lo =
                        static_cast<std::size_t>(slab / kWorld) *
                        g.slab_bytes();
                    const std::size_t owned_stride =
                        ((nslabs + kWorld - 1u) / kWorld) * g.slab_bytes();
                    auto* acc2 = reinterpret_cast<__hip_bfloat162*>(
                        pv.accum[rank] + lo);
                    const std::size_t elems2 =
                        static_cast<std::size_t>(g.rows_in(slab)) *
                        (kRowBytes / 4);
                    const std::size_t per_seg =
                        (elems2 + kReduceSegs - 1u) / kReduceSegs;
                    const std::size_t s0 = seg * per_seg;
                    const std::size_t s1 =
                        s0 + per_seg < elems2 ? s0 + per_seg : elems2;
                    for (std::size_t i = s0 + threadIdx.x; i < s1;
                         i += blockDim.x) {
                        float ax = 0.f, ay = 0.f;
#pragma unroll
                        for (unsigned int r = 0u; r < kWorld; ++r) {
                            const auto* tw2 =
                                reinterpret_cast<const __hip_bfloat162*>(
                                    pv.towers[rank] + r * owned_stride + lo);
                            ax += __bfloat162float(tw2[i].x);
                            ay += __bfloat162float(tw2[i].y);
                        }
                        acc2[i] = __hip_bfloat162(__float2bfloat16(ax),
                                                  __float2bfloat16(ay));
                    }
                }
                __syncthreads();
                M25_LG_MARK(lg_rd, cd::lg_reduce, slab);
                M25_LG_T0(lg_ct);
                kd::producer_drain_release<kd::memory_scope::system>();
                if (threadIdx.x == 0u) {
                    kd::counted_arrival ra;
                    kd::counted_arrive_dynamic_release_into<
                        kd::memory_scope::system>(reduce_cells + slab,
                                                  kReduceSegs, ra);
                    if (ra.last) {
                        cd::owner_certify(g, pv, slab, epoch, g.rows_in(slab));
                        atomicAdd(&report->slabs_certified, 1u);
                    }
                }
                M25_LG_MARK(lg_ct, cd::lg_certify, slab);
            }
            M25_LG_MARK(lg_duty, cd::lg_duty_loop, 0u);
        }
        // ---- consume + verify
        M25_LG_T0(lg_cons);
        for (unsigned int slab = block; slab < nslabs; slab += kBlocks) {
            kd::wait_result64 w;
            // LAW-63: bracket OUTSIDE the poll body.
            M25_LG_T0(lg_mw);
            if (threadIdx.x == 0u) {
                unsigned int tries = 0u;
                do {
                    kd::init_wait_result(w);
                    cd::mag_wait_into(g, pv, slab, epoch, kSpin, w);
                } while (!w.ready && ++tries < kAttempts);
                if (!w.ready) atomicAdd(&report->mag_timeouts, 1u);
            }
            M25_LG_MARK(lg_mw, cd::lg_mag_wait, slab);
            __syncthreads();
            std::byte* dst =
                staging + static_cast<std::size_t>(slab) * g.slab_bytes();
            M25_LG_T0(lg_pl);
            cd::mag_pull(g, pv, slab, dst, threadIdx.x, blockDim.x);
            __syncthreads();
            M25_LG_MARK(lg_pl, cd::lg_pull,
                        static_cast<std::size_t>(g.rows_in(slab)) * kRowBytes);
            M25_LG_T0(lg_vf);
            if (threadIdx.x < 64u) {
                const auto* v = reinterpret_cast<const unsigned short*>(dst);
                const std::size_t elems =
                    static_cast<std::size_t>(g.rows_in(slab)) *
                    (kRowBytes / 2);
                const std::size_t idx =
                    (static_cast<std::size_t>(threadIdx.x) * 2654435761u) %
                    elems;
                if (v[idx] != static_cast<unsigned short>(expected_bits)) {
                    atomicAdd(&report->verify_errors, 1u);
                }
            }
            M25_LG_MARK(lg_vf, cd::lg_verify, slab);
            // burst2: the post-boundary GEMM consumes this slab AS IT LANDS
            // — boundary-N tail hides under layer-N+1 compute, per slab.
            M25_LG_T0(lg_m2b);
            acc_sink += mfma_burst(dst, g.rows_in(slab), k_inner, threadIdx.x);
            M25_LG_MARK(lg_m2b, cd::lg_mfma2, g.rows_in(slab));
        }
        M25_LG_MARK(lg_cons, cd::lg_cons_loop, 0u);
    }

    if (threadIdx.x == 0u) {
        atomicAdd(&report->acc_token, acc_sink);
    }
    M25_LG_EXIT();
}

static unsigned short f32_to_bf16_bits(float f) {
    unsigned int u;
    std::memcpy(&u, &f, 4);
    return static_cast<unsigned short>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

template<int Mode>
static void launch_mode_depth(unsigned int depth, hipStream_t s,
                              const cd::peer_view<kWorld>& pv,
                              unsigned int tokens, unsigned int slab_rows,
                              unsigned int rank, std::uint32_t epoch,
                              unsigned int k_inner, unsigned int stage,
                              unsigned int bps, std::uint32_t* rcells,
                              const std::byte* src, std::byte* stage_buf,
                              rank_report* rep, unsigned int bits
                              M25_LG_PARAM) {
    switch (depth) {
        case 8:
            boundary_kernel<Mode, 8u><<<kBlocks, kThreads, 0, s>>>(
                pv, tokens, slab_rows, rank, epoch, k_inner, stage, bps,
                rcells, src, stage_buf, rep, bits M25_LG_ARG);
            break;
        case 16:
            boundary_kernel<Mode, 16u><<<kBlocks, kThreads, 0, s>>>(
                pv, tokens, slab_rows, rank, epoch, k_inner, stage, bps,
                rcells, src, stage_buf, rep, bits M25_LG_ARG);
            break;
        case 32:
            boundary_kernel<Mode, 32u><<<kBlocks, kThreads, 0, s>>>(
                pv, tokens, slab_rows, rank, epoch, k_inner, stage, bps,
                rcells, src, stage_buf, rep, bits M25_LG_ARG);
            break;
        default:
            boundary_kernel<Mode, 4u><<<kBlocks, kThreads, 0, s>>>(
                pv, tokens, slab_rows, rank, epoch, k_inner, stage, bps,
                rcells, src, stage_buf, rep, bits M25_LG_ARG);
            break;
    }
}

// ---------------------------------------------------------------------------
// E-B1/E-B2 host-side reduction. Everything below is host code: it does not
// exist in the device image and cannot perturb the measured schedule.
// ---------------------------------------------------------------------------
#if M25_LEDGER
namespace lgh {

static const char* code_name(std::uint32_t c) {
    switch (c) {
        case cd::lg_kernel:      return "kernel";
        case cd::lg_prod_loop:   return "prod_loop";
        case cd::lg_mfma1:       return "mfma1";
        case cd::lg_commit:      return "commit";
        case cd::lg_drain:       return "drain";
        case cd::lg_duty_loop:   return "duty_loop";
        case cd::lg_gather_wait: return "gather_wait";
        case cd::lg_reduce:      return "reduce";
        case cd::lg_certify:     return "certify";
        case cd::lg_cons_loop:   return "cons_loop";
        case cd::lg_mag_wait:    return "mag_wait";
        case cd::lg_pull:        return "pull";
        case cd::lg_verify:      return "verify";
        case cd::lg_mfma2:       return "mfma2";
        case cd::lg_tx_loop:     return "tx_loop";
        case cd::lg_kernel_end:  return "kernel_end";
        default:                 return "?";
    }
}

struct interval { std::uint64_t a, b; };

/// Merge into a disjoint union, sorted ascending. Input is copied.
static std::vector<interval> merge_union(std::vector<interval> v) {
    std::sort(v.begin(), v.end(),
              [](const interval& x, const interval& y) { return x.a < y.a; });
    std::vector<interval> out;
    for (const auto& iv : v) {
        if (iv.b <= iv.a) continue;
        if (!out.empty() && iv.a <= out.back().b) {
            if (iv.b > out.back().b) out.back().b = iv.b;
        } else {
            out.push_back(iv);
        }
    }
    return out;
}

static std::uint64_t total_len(const std::vector<interval>& u) {
    std::uint64_t t = 0;
    for (const auto& iv : u) t += iv.b - iv.a;
    return t;
}

/// Length of the intersection of two disjoint unions (merge walk).
static std::uint64_t intersect_len(const std::vector<interval>& x,
                                   const std::vector<interval>& y) {
    std::uint64_t t = 0;
    std::size_t i = 0, j = 0;
    while (i < x.size() && j < y.size()) {
        const std::uint64_t lo = x[i].a > y[j].a ? x[i].a : y[j].a;
        const std::uint64_t hi = x[i].b < y[j].b ? x[i].b : y[j].b;
        if (hi > lo) t += hi - lo;
        if (x[i].b < y[j].b) ++i; else ++j;
    }
    return t;
}

/// Everything one (launch slot, rank) contributes to the report.
struct rank_slot {
    bool present = false;
    std::uint64_t origin = 0;       ///< min kernel-entry stamp on THIS rank
    std::uint64_t span = 0;         ///< max kernel-exit - origin (ticks)
    unsigned int dropped = 0;
    unsigned int nblocks = 0;
    std::uint64_t n[cd::lg_code_count] = {};
    std::uint64_t sum[cd::lg_code_count] = {};   ///< CTA-ticks (not wall)
    std::uint64_t bytes[cd::lg_code_count] = {};
    std::uint64_t uni[cd::lg_code_count] = {};   ///< union length, ticks
    std::uint64_t first[cd::lg_code_count] = {};
    std::uint64_t last[cd::lg_code_count] = {};
    std::uint64_t u_compute = 0, u_transport = 0, u_wait = 0, u_overlap = 0;
    std::vector<double> duty;       ///< kDutyHistBins fabric-busy fractions
};

/// Reduce one rank's raw ring for one launch slot. Compute class = the MFMA
/// bursts; transport class = the remote store commits + the remote pulls
/// (the fabric-touching spans); wait class = the two bounded polls + drain.
static rank_slot reduce_slot(const std::vector<cd::ledger_event>& ev,
                             const std::vector<std::uint32_t>& cnt,
                             unsigned int cap) {
    rank_slot rs;
    std::vector<interval> per_code[cd::lg_code_count];
    std::vector<interval> comp, trans, wait;
    bool have = false;
    std::uint64_t lo = 0, hi = 0;
    for (unsigned int b = 0; b < kBlocks; ++b) {
        const unsigned int n = cnt[b] & 0xffffu;
        rs.dropped += (cnt[b] >> 16);
        if (n) ++rs.nblocks;
        for (unsigned int i = 0; i < n; ++i) {
            const cd::ledger_event& e = ev[static_cast<std::size_t>(b) * cap + i];
            if (e.code >= cd::lg_code_count) continue;
            const std::uint64_t d = e.t1 > e.t0 ? e.t1 - e.t0 : 0;
            rs.n[e.code] += 1;
            rs.sum[e.code] += d;
            if (e.code == cd::lg_commit || e.code == cd::lg_pull)
                rs.bytes[e.code] += e.aux;
            if (!rs.first[e.code] || e.t0 < rs.first[e.code])
                rs.first[e.code] = e.t0;
            if (e.t1 > rs.last[e.code]) rs.last[e.code] = e.t1;
            per_code[e.code].push_back({e.t0, e.t1});
            if (e.code == cd::lg_mfma1 || e.code == cd::lg_mfma2)
                comp.push_back({e.t0, e.t1});
            if (e.code == cd::lg_commit || e.code == cd::lg_pull)
                trans.push_back({e.t0, e.t1});
            if (e.code == cd::lg_gather_wait || e.code == cd::lg_mag_wait ||
                e.code == cd::lg_drain)
                wait.push_back({e.t0, e.t1});
            if (e.code == cd::lg_kernel) {
                if (!have || e.t0 < lo) lo = e.t0;
                have = true;
            }
            if (e.code == cd::lg_kernel_end && e.t1 > hi) hi = e.t1;
        }
    }
    if (!have) return rs;
    rs.present = true;
    rs.origin = lo;
    rs.span = hi - lo;
    for (unsigned int c = 0; c < cd::lg_code_count; ++c)
        rs.uni[c] = total_len(merge_union(per_code[c]));
    const auto uc = merge_union(comp);
    const auto ut = merge_union(trans);
    rs.u_compute = total_len(uc);
    rs.u_transport = total_len(ut);
    rs.u_wait = total_len(merge_union(wait));
    rs.u_overlap = intersect_len(uc, ut);
    // E-B2: fabric duty over the launch timeline, per rank (a cross-rank
    // timeline is INVALID — s_memrealtime origins are per device).
    rs.duty.assign(kDutyHistBins, 0.0);
    if (rs.span) {
        const double w = static_cast<double>(rs.span) / kDutyHistBins;
        for (const auto& iv : ut) {
            const double a = static_cast<double>(iv.a - rs.origin) / w;
            const double b = static_cast<double>(iv.b - rs.origin) / w;
            for (unsigned int k = 0; k < kDutyHistBins; ++k) {
                const double s = a > k ? a : k;
                const double e = b < k + 1.0 ? b : k + 1.0;
                if (e > s) rs.duty[k] += e - s;
            }
        }
        for (auto& d : rs.duty) if (d > 1.0) d = 1.0;
    }
    return rs;
}

static double us(std::uint64_t ticks) {
    return static_cast<double>(ticks) * cd::kLedgerTickUs;
}

} // namespace lgh
#endif // M25_LEDGER

// One arm's schedule shape. Everything here is host-side dispatch: no arm
// adds a kernel instantiation, so the device image is arm-count-invariant.
struct arm_spec {
    int kmode;         ///< mode_compute / mode_phased / mode_fused
    bool zero_kinner;  ///< transport-only arms: k_inner forced to 0
    bool uses_rccl;    ///< enqueues ncclAllReduce at the boundary
    bool kernels;      ///< launches boundary kernels at all
    bool two_launch;   ///< burst1 and burst2 as separate launches
};

static bool arm_lookup(const std::string& a, arm_spec& s) {
    if (a == "compute")   { s = {mode_compute, false, false, true,  true }; return true; }
    if (a == "compute0")  { s = {mode_compute, true,  false, true,  true }; return true; }
    if (a == "phased")    { s = {mode_phased,  false, false, true,  false}; return true; }
    if (a == "fused")     { s = {mode_fused,   false, false, true,  false}; return true; }
    if (a == "transport") { s = {mode_phased,  true,  false, true,  false}; return true; }
    if (a == "transport_fused")
                          { s = {mode_fused,   true,  false, true,  false}; return true; }
    if (a == "rccl")      { s = {mode_compute, false, true,  true,  true }; return true; }
    if (a == "stdrccl")   { s = {mode_compute, false, true,  false, false}; return true; }
    return false;
}

static const char* env_or(const char* k) {
    const char* v = std::getenv(k);
    return v ? v : "(unset)";
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "fused";
    const unsigned int tokens = argc > 2 ? std::atoi(argv[2]) : 16384u;
    const unsigned int slab_rows = argc > 3 ? std::atoi(argv[3]) : 256u;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 7;
    const unsigned int depth = argc > 5 ? std::atoi(argv[5]) : 4u;
    const unsigned int k_inner = argc > 6 ? std::atoi(argv[6]) : 64u;
    const unsigned int bps = argc > 7 ? std::atoi(argv[7]) : 2u;

    // `beta` alternates our standalone transport against RCCL's standalone
    // all-reduce inside ONE invocation: LAW-60 voids cross-session lone-arm
    // contrasts, and beta is the ratio of two DIRECT measurements (B4), never
    // of two wall-minus-floor subtractions.
    std::string arm_name[2] = {mode, std::string()};
    int narms = 1;
    if (mode == "beta") {
        arm_name[0] = "transport";
        arm_name[1] = "stdrccl";
        narms = 2;
    }
    arm_spec spec[2]{};
    for (int a = 0; a < narms; ++a) {
        if (!arm_lookup(arm_name[a], spec[a])) {
            std::fprintf(stderr,
                         "unknown mode '%s'; expected one of compute compute0 "
                         "phased fused transport transport_fused rccl stdrccl "
                         "beta\n", arm_name[a].c_str());
            return 2;
        }
    }
    const bool needs_comm = spec[0].uses_rccl ||
                            (narms > 1 && spec[1].uses_rccl);

    cd::geometry<kWorld> g{tokens, kRowBytes, slab_rows, 0u};
    const unsigned int nslabs = g.num_slabs();
    const std::size_t accum_b = cd::accum_bytes(g);
    const std::size_t towers_b = accum_b * kWorld;
    const std::size_t src_b = static_cast<std::size_t>(nslabs) * g.slab_bytes();
    const std::size_t sig64 = cd::signal_words64(g);

    int ndev = 0;
    HIP_CHECK(hipGetDeviceCount(&ndev));
    if (ndev < static_cast<int>(kWorld)) return 2;
    for (unsigned int a = 0; a < kWorld; ++a) {
        HIP_CHECK(hipSetDevice(a));
        for (unsigned int b = 0; b < kWorld; ++b) {
            if (a == b) continue;
            hipError_t e = hipDeviceEnablePeerAccess(b, 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
                HIP_CHECK(e);
            (void)hipGetLastError();
        }
    }

    std::byte* accum[kWorld]; std::byte* twr[kWorld]; std::byte* src[kWorld];
    std::byte* stage[kWorld];
    std::uint64_t* s64[kWorld]; std::uint32_t* s32[kWorld];
    std::uint32_t* rcell[kWorld];
    rank_report* rep[kWorld];
    hipStream_t str[kWorld];
    ncclComm_t comms[kWorld];
    for (unsigned int r = 0; r < kWorld; ++r) {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipStreamCreate(&str[r]));
        HIP_CHECK(hipMalloc(&accum[r], accum_b));
        HIP_CHECK(hipMalloc(&twr[r], towers_b));
        HIP_CHECK(hipMalloc(&src[r], src_b));
        HIP_CHECK(hipMalloc(&stage[r], src_b));
        HIP_CHECK(hipMalloc(&s64[r], sig64 * 8));
        HIP_CHECK(hipMalloc(&s32[r], nslabs * 4));
        HIP_CHECK(hipMalloc(&rcell[r], nslabs * 4));
        HIP_CHECK(hipMalloc(&rep[r], sizeof(rank_report)));
        HIP_CHECK(hipMemset(s64[r], 0, sig64 * 8));
        const unsigned short b16 = f32_to_bf16_bits(static_cast<float>(r + 1));
        const unsigned int packed = (static_cast<unsigned int>(b16) << 16) | b16;
        fill_bf16<<<256, 256, 0, str[r]>>>(src[r], src_b, packed);
    }
    if (needs_comm) {
        int devs[kWorld];
        for (unsigned int r = 0; r < kWorld; ++r) devs[r] = static_cast<int>(r);
        NCCL_CHECK(ncclCommInitAll(comms, kWorld, devs));
    }
    for (unsigned int r = 0; r < kWorld; ++r) {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipStreamSynchronize(str[r]));
    }

    cd::peer_view<kWorld> pv{};
    for (unsigned int r = 0; r < kWorld; ++r) {
        pv.accum[r] = accum[r];
        pv.towers[r] = twr[r];
        pv.sig[r] = {s64[r], s64[r] + static_cast<std::size_t>(nslabs) * kWorld,
                     s32[r]};
    }

    const unsigned short expect_bits = f32_to_bf16_bits(36.0f);
    const std::size_t logical = static_cast<std::size_t>(tokens) * kRowBytes;

#if M25_LEDGER
    // ---- E-B1 arena: one raw event ring per (launch slot, CTA), per rank.
    // Counts are zeroed before EVERY iteration and read back after it, so a
    // ring only ever describes the iteration it was armed for (LAW-63c) and
    // no stamp is ever compared across launches (LAW-63e).
    const std::size_t lg_ev_n =
        static_cast<std::size_t>(kLedgerLaunches) * kBlocks * kLedgerCap;
    const std::size_t lg_cnt_n =
        static_cast<std::size_t>(kLedgerLaunches) * kBlocks;
    cd::ledger_event* lg_ev[kWorld];
    std::uint32_t* lg_cnt[kWorld];
    for (unsigned int r = 0; r < kWorld; ++r) {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipMalloc(&lg_ev[r], lg_ev_n * sizeof(cd::ledger_event)));
        HIP_CHECK(hipMalloc(&lg_cnt[r], lg_cnt_n * sizeof(std::uint32_t)));
        HIP_CHECK(hipMemset(lg_ev[r], 0, lg_ev_n * sizeof(cd::ledger_event)));
        HIP_CHECK(hipMemset(lg_cnt[r], 0, lg_cnt_n * sizeof(std::uint32_t)));
    }
    // [iter][rank] snapshots; analysed at the end for the MEDIAN-wall
    // iteration of each arm (the same iteration the headline wall quotes).
    std::vector<std::vector<std::vector<cd::ledger_event>>> snap_ev(iters);
    std::vector<std::vector<std::vector<std::uint32_t>>> snap_cnt(iters);
    int lg_clock_khz = 0;
    (void)hipDeviceGetAttribute(&lg_clock_khz, hipDeviceAttributeWallClockRate, 0);
#endif

    std::vector<double> walls[2];
    std::vector<int> wall_iter[2];
    unsigned int rccl_verify_errors = 0;
    for (int it = 0; it < iters; ++it) {
        const int ai = (narms > 1) ? (it & 1) : 0;
        const arm_spec& as = spec[ai];
        const unsigned int ki = as.zero_kinner ? 0u : k_inner;
        const std::uint32_t epoch = static_cast<std::uint32_t>(it + 1);
        for (unsigned int r = 0; r < kWorld; ++r) {
            HIP_CHECK(hipSetDevice(r));
            HIP_CHECK(hipMemsetAsync(s32[r], 0, nslabs * 4, str[r]));
            HIP_CHECK(hipMemsetAsync(rcell[r], 0, nslabs * 4, str[r]));
            HIP_CHECK(hipMemsetAsync(accum[r], 0, accum_b, str[r]));
            HIP_CHECK(hipMemsetAsync(rep[r], 0, sizeof(rank_report), str[r]));
#if M25_LEDGER
            HIP_CHECK(hipMemsetAsync(lg_cnt[r], 0,
                                     lg_cnt_n * sizeof(std::uint32_t), str[r]));
#endif
            if (as.uses_rccl) {
                // rccl arms reduce IN PLACE on a fresh copy of src in staging
                HIP_CHECK(hipMemcpyAsync(stage[r], src[r], src_b,
                                         hipMemcpyDeviceToDevice, str[r]));
            }
        }
        for (unsigned int r = 0; r < kWorld; ++r) {
            HIP_CHECK(hipSetDevice(r));
            HIP_CHECK(hipStreamSynchronize(str[r]));
        }
#if M25_LEDGER
        cd::ledger_view lv0[kWorld], lv1[kWorld];
        for (unsigned int r = 0; r < kWorld; ++r) {
            lv0[r] = {lg_ev[r], lg_cnt[r], kLedgerCap};
            lv1[r] = {lg_ev[r] + static_cast<std::size_t>(kBlocks) * kLedgerCap,
                      lg_cnt[r] + kBlocks, kLedgerCap};
        }
#endif
        const auto t0 = std::chrono::steady_clock::now();
        if (as.kernels) {
            for (unsigned int r = 0; r < kWorld; ++r) {
                HIP_CHECK(hipSetDevice(r));
                std::byte* dstbuf = as.uses_rccl || as.kmode != mode_compute
                                        ? stage[r] : src[r];
                if (as.kmode == mode_compute) {
                    launch_mode_depth<mode_compute>(
                        depth, str[r], pv, tokens, slab_rows, r, epoch, ki,
                        1u, bps, rcell[r], src[r], dstbuf, rep[r],
                        expect_bits M25_LG_A0(r));
                    if (!as.uses_rccl) {
                        launch_mode_depth<mode_compute>(
                            depth, str[r], pv, tokens, slab_rows, r, epoch, ki,
                            2u, bps, rcell[r], src[r], dstbuf, rep[r],
                            expect_bits M25_LG_A1(r));
                    }
                } else if (as.kmode == mode_phased) {
                    launch_mode_depth<mode_phased>(
                        depth, str[r], pv, tokens, slab_rows, r, epoch, ki,
                        0u, bps, rcell[r], src[r], stage[r], rep[r],
                        expect_bits M25_LG_A0(r));
                } else {
                    launch_mode_depth<mode_fused>(
                        depth, str[r], pv, tokens, slab_rows, r, epoch, ki,
                        0u, bps, rcell[r], src[r], stage[r], rep[r],
                        expect_bits M25_LG_A0(r));
                }
            }
        }
        if (as.uses_rccl) {
            NCCL_CHECK(ncclGroupStart());
            for (unsigned int r = 0; r < kWorld; ++r) {
                NCCL_CHECK(ncclAllReduce(stage[r], stage[r],
                                         logical / 2, ncclBfloat16, ncclSum,
                                         comms[r], str[r]));
            }
            NCCL_CHECK(ncclGroupEnd());
            if (as.kernels && as.two_launch) {
                for (unsigned int r = 0; r < kWorld; ++r) {
                    HIP_CHECK(hipSetDevice(r));
                    launch_mode_depth<mode_compute>(
                        depth, str[r], pv, tokens, slab_rows, r, epoch, ki,
                        2u, bps, rcell[r], src[r], stage[r], rep[r],
                        expect_bits M25_LG_A1(r));
                }
            }
        }
        for (unsigned int r = 0; r < kWorld; ++r) {
            HIP_CHECK(hipSetDevice(r));
            HIP_CHECK(hipStreamSynchronize(str[r]));
        }
        const auto t1 = std::chrono::steady_clock::now();
        walls[ai].push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
        wall_iter[ai].push_back(it);
        if (as.uses_rccl) {
            unsigned short sample[64];
            HIP_CHECK(hipSetDevice(0));
            HIP_CHECK(hipMemcpy(sample, stage[0], sizeof(sample),
                                hipMemcpyDeviceToHost));
            for (auto sv : sample) {
                if (sv != expect_bits) ++rccl_verify_errors;
            }
        }
#if M25_LEDGER
        // Read back OUTSIDE the timing bracket; the stream sync above is the
        // release edge for the plain device stores in the rings.
        snap_ev[it].resize(kWorld);
        snap_cnt[it].resize(kWorld);
        for (unsigned int r = 0; r < kWorld; ++r) {
            HIP_CHECK(hipSetDevice(r));
            snap_ev[it][r].resize(lg_ev_n);
            snap_cnt[it][r].resize(lg_cnt_n);
            HIP_CHECK(hipMemcpy(snap_ev[it][r].data(), lg_ev[r],
                                lg_ev_n * sizeof(cd::ledger_event),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(snap_cnt[it][r].data(), lg_cnt[r],
                                lg_cnt_n * sizeof(std::uint32_t),
                                hipMemcpyDeviceToHost));
        }
#endif
    }

    rank_report host_rep[kWorld];
    unsigned int errs = rccl_verify_errors, gt = 0, mt = 0;
    for (unsigned int r = 0; r < kWorld; ++r) {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipMemcpy(&host_rep[r], rep[r], sizeof(rank_report),
                            hipMemcpyDeviceToHost));
        errs += host_rep[r].verify_errors;
        gt += host_rep[r].gather_timeouts;
        mt += host_rep[r].mag_timeouts;
    }

    const bool pass = (errs == 0 && gt == 0 && mt == 0);
    std::printf("BOUNDARY_BENCH_CFG mode=%s depth=%u tokens=%u slab_rows=%u "
                "k_inner=%u bps=%u iters=%d payload_MB=%.1f ledger=%d "
                "nocompute=%d\n",
                mode.c_str(), depth, tokens, slab_rows, k_inner, bps, iters,
                static_cast<double>(logical) / 1.0e6, M25_LEDGER,
                M25_NOCOMPUTE);
    if (needs_comm) {
        std::printf("BOUNDARY_BENCH_RCCLENV NCCL_ALGO=%s NCCL_PROTO=%s "
                    "NCCL_NCHANNELS=%s NCCL_MIN_NCHANNELS=%s "
                    "NCCL_MAX_NCHANNELS=%s\n",
                    env_or("NCCL_ALGO"), env_or("NCCL_PROTO"),
                    env_or("NCCL_NCHANNELS"), env_or("NCCL_MIN_NCHANNELS"),
                    env_or("NCCL_MAX_NCHANNELS"));
    }
    double med[2] = {0.0, 0.0};
    int med_it[2] = {-1, -1};
    for (int a = 0; a < narms; ++a) {
        if (walls[a].empty()) continue;
        std::vector<std::size_t> ord(walls[a].size());
        for (std::size_t i = 0; i < ord.size(); ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](std::size_t x, std::size_t y) {
            return walls[a][x] < walls[a][y];
        });
        const std::size_t mid = ord[ord.size() / 2];
        med[a] = walls[a][mid];
        med_it[a] = wall_iter[a][mid];
        std::printf(
            "BOUNDARY_BENCH mode=%s depth=%u tokens=%u slab_rows=%u k_inner=%u "
            "iters=%d\n  wall_us median=%.1f min=%.1f max=%.1f n=%zu\n"
            "  gather_timeouts=%u mag_timeouts=%u verify_errors=%u\n",
            arm_name[a].c_str(), depth, tokens,
            slab_rows, spec[a].zero_kinner ? 0u : k_inner, iters, med[a],
            walls[a][ord.front()], walls[a][ord.back()], walls[a].size(),
            gt, mt, errs);
        // One bandwidth convention per line, both labelled (never mixed).
        const double gbps_logical =
            static_cast<double>(logical) / (med[a] * 1.0e3);
        const double gbps_bus = gbps_logical * 2.0 * (kWorld - 1) / kWorld;
        std::printf("  BW arm=%s logical_bytes_over_wall_GBps=%.1f "
                    "busbw_convention_GBps=%.1f\n",
                    arm_name[a].c_str(), gbps_logical, gbps_bus);
    }
    if (narms > 1 && med[1] > 0.0) {
        std::printf("BETA_DIRECT ours=%s_us=%.1f rccl=%s_us=%.1f "
                    "beta=%.4f  # ratio of two DIRECT measurements, one "
                    "binary, alternating arms (B4)\n",
                    arm_name[0].c_str(), med[0], arm_name[1].c_str(), med[1],
                    med[0] / med[1]);
    }
#if M25_LEDGER
    std::printf("LEDGER_CLOCK assumed_Hz=%llu hipWallClockRate_kHz=%d "
                "tick_us=%.4f\n",
                (unsigned long long)cd::kLedgerHz, lg_clock_khz,
                cd::kLedgerTickUs);
    if (lg_clock_khz > 0 &&
        (unsigned long long)lg_clock_khz * 1000ull != cd::kLedgerHz) {
        std::printf("LEDGER_CLOCK_WARN device reports %d kHz, ledger assumes "
                    "%llu Hz — rescale before quoting any span\n",
                    lg_clock_khz, (unsigned long long)cd::kLedgerHz);
    }
    for (int a = 0; a < narms; ++a) {
        if (med_it[a] < 0 || !spec[a].kernels) continue;
        const int it = med_it[a];
        for (unsigned int sl = 0; sl < kLedgerLaunches; ++sl) {
            const std::size_t evoff =
                static_cast<std::size_t>(sl) * kBlocks * kLedgerCap;
            const std::size_t cnoff = static_cast<std::size_t>(sl) * kBlocks;
            std::vector<lgh::rank_slot> rs(kWorld);
            bool any = false;
            for (unsigned int r = 0; r < kWorld; ++r) {
                std::vector<cd::ledger_event> ev(
                    snap_ev[it][r].begin() + evoff,
                    snap_ev[it][r].begin() + evoff +
                        static_cast<std::size_t>(kBlocks) * kLedgerCap);
                std::vector<std::uint32_t> cn(
                    snap_cnt[it][r].begin() + cnoff,
                    snap_cnt[it][r].begin() + cnoff + kBlocks);
                rs[r] = lgh::reduce_slot(ev, cn, kLedgerCap);
                any = any || rs[r].present;
            }
            if (!any) continue;
            for (unsigned int r = 0; r < kWorld; ++r) {
                if (!rs[r].present) continue;
                std::printf("LEDGER arm=%s launch=%u iter=%d rank=%u "
                            "span_us=%.1f blocks=%u dropped=%u\n",
                            arm_name[a].c_str(), sl, it, r, lgh::us(rs[r].span),
                            rs[r].nblocks, rs[r].dropped);
                for (unsigned int c = 1; c < cd::lg_code_count; ++c) {
                    if (!rs[r].n[c]) continue;
                    std::printf("LEDGER_PHASE arm=%s launch=%u rank=%u "
                                "phase=%s n=%llu cta_sum_us=%.1f union_us=%.1f "
                                "first_us=%.1f last_us=%.1f bytes=%llu\n",
                                arm_name[a].c_str(), sl, r, lgh::code_name(c),
                                (unsigned long long)rs[r].n[c],
                                lgh::us(rs[r].sum[c]), lgh::us(rs[r].uni[c]),
                                lgh::us(rs[r].first[c] - rs[r].origin),
                                lgh::us(rs[r].last[c] - rs[r].origin),
                                (unsigned long long)rs[r].bytes[c]);
                }
                const double ut = lgh::us(rs[r].u_transport);
                std::printf("LEDGER_OVERLAP arm=%s launch=%u rank=%u "
                            "U_compute_us=%.1f U_transport_us=%.1f "
                            "U_wait_us=%.1f overlap_us=%.1f h_ledger=%.4f "
                            "fabric_duty=%.4f\n",
                            arm_name[a].c_str(), sl, r,
                            lgh::us(rs[r].u_compute), ut,
                            lgh::us(rs[r].u_wait), lgh::us(rs[r].u_overlap),
                            ut > 0.0 ? lgh::us(rs[r].u_overlap) / ut : 0.0,
                            rs[r].span ? ut / lgh::us(rs[r].span) : 0.0);
                std::printf("LEDGER_DUTY arm=%s launch=%u rank=%u bins=%u "
                            "bin_us=%.2f", arm_name[a].c_str(), sl, r,
                            kDutyHistBins,
                            lgh::us(rs[r].span) / kDutyHistBins);
                for (unsigned int k = 0; k < kDutyHistBins; ++k)
                    std::printf(" %.3f", rs[r].duty[k]);
                std::printf("\n");
            }
            // LAW-63(d): the reported figure is the CROSS-RANK MAX, never
            // rank 0 (rank-0-only understates a combine-class phase by 38%).
            lgh::rank_slot mx;
            unsigned int drop = 0;
            for (unsigned int r = 0; r < kWorld; ++r) {
                if (!rs[r].present) continue;
                drop += rs[r].dropped;
                if (rs[r].span > mx.span) mx.span = rs[r].span;
                for (unsigned int c = 0; c < cd::lg_code_count; ++c) {
                    if (rs[r].sum[c] > mx.sum[c]) mx.sum[c] = rs[r].sum[c];
                    if (rs[r].uni[c] > mx.uni[c]) mx.uni[c] = rs[r].uni[c];
                }
                if (rs[r].u_compute > mx.u_compute) mx.u_compute = rs[r].u_compute;
                if (rs[r].u_transport > mx.u_transport) mx.u_transport = rs[r].u_transport;
                if (rs[r].u_wait > mx.u_wait) mx.u_wait = rs[r].u_wait;
                if (rs[r].u_overlap > mx.u_overlap) mx.u_overlap = rs[r].u_overlap;
            }
            std::printf("LEDGER_RANKMAX arm=%s launch=%u iter=%d span_us=%.1f "
                        "dropped_total=%u\n", arm_name[a].c_str(), sl, it,
                        lgh::us(mx.span), drop);
            for (unsigned int c = 1; c < cd::lg_code_count; ++c) {
                if (!mx.uni[c] && !mx.sum[c]) continue;
                std::printf("LEDGER_RANKMAX_PHASE arm=%s launch=%u phase=%s "
                            "cta_sum_us=%.1f union_us=%.1f\n",
                            arm_name[a].c_str(), sl, lgh::code_name(c),
                            lgh::us(mx.sum[c]), lgh::us(mx.uni[c]));
            }
            const double mut = lgh::us(mx.u_transport);
            std::printf("LEDGER_RANKMAX_OVERLAP arm=%s launch=%u "
                        "U_compute_us=%.1f U_transport_us=%.1f U_wait_us=%.1f "
                        "overlap_us=%.1f h_ledger=%.4f fabric_duty=%.4f\n",
                        arm_name[a].c_str(), sl, lgh::us(mx.u_compute), mut,
                        lgh::us(mx.u_wait), lgh::us(mx.u_overlap),
                        mut > 0.0 ? lgh::us(mx.u_overlap) / mut : 0.0,
                        mx.span ? mut / lgh::us(mx.span) : 0.0);
            if (drop) {
                std::printf("LEDGER_OVERFLOW arm=%s launch=%u dropped=%u — "
                            "spans are LOWER BOUNDS; raise kLedgerCap and "
                            "re-run before quoting\n",
                            arm_name[a].c_str(), sl, drop);
            }
        }
    }
    for (unsigned int r = 0; r < kWorld; ++r) {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipFree(lg_ev[r]));
        HIP_CHECK(hipFree(lg_cnt[r]));
    }
#endif
    std::printf("%s\n",
                pass ? "BOUNDARY_BENCH_PASS" : "BOUNDARY_BENCH_FAIL");
    if (needs_comm) {
        for (unsigned int r = 0; r < kWorld; ++r) ncclCommDestroy(comms[r]);
    }
    return pass ? 0 : 1;
}

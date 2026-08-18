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
 * Run: ./m25_boundary_bench <mode> [tokens=16384] [slab_rows=256]
 *      [iters=7] [depth=4] [k_inner=64]
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
        rank_report* report, unsigned int expected_bits) {
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
        for (unsigned int slab = block; slab < nslabs; slab += kBlocks) {
            const std::byte* dst =
                staging + static_cast<std::size_t>(slab) * g.slab_bytes();
            acc_sink += mfma_burst(dst, g.rows_in(slab), k_inner, threadIdx.x);
        }
        if (threadIdx.x == 0u) atomicAdd(&report->acc_token, acc_sink);
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
            acc_sink += mfma_burst(frag, rows, k_inner, threadIdx.x);
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
                kd::throttle_vmcnt<0>();
                __syncthreads();
                if (threadIdx.x == 0u) {
                    cd::ers_local_arrive(g, pv, prev_slab, bps, epoch,
                                         prev_rows);
                }
                __syncthreads();
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
#pragma unroll 1
                for (std::size_t p = threadIdx.x; p < packets;
                     p += blockDim.x) {
                    kd::throttled_store_packet16<Depth>(out + p, reg_packet);
                }
                // no drain: the next iteration's mfma_burst is the shadow
            }
            prev_valid = true;
            prev_slab = slab;
            prev_rows = rows;
        }
    }
    if constexpr (Mode == mode_fused) {
        if (prev_valid) {
            kd::throttle_vmcnt<0>();
            __syncthreads();
            if (threadIdx.x == 0u) {
                cd::ers_local_arrive(g, pv, prev_slab, bps, epoch,
                                     prev_rows);
            }
            __syncthreads();
        }
    }

    if constexpr (Mode == mode_phased) {
        // The unfused control: identical transport, exposed as a phase.
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
                const std::byte* frag =
                    src + static_cast<std::size_t>(slab) * g.slab_bytes() +
                    static_cast<std::size_t>(r0) * kRowBytes;
                cd::ers_commit_fragment<cd::transport::store_towers, Depth>(
                    g, pv, slab, r0, rows, frag, threadIdx.x, blockDim.x);
                kd::throttle_vmcnt<0>();
            }
            __syncthreads();
            if (threadIdx.x == 0u) {
                cd::ers_local_arrive(g, pv, slab, bps, epoch, rows);
            }
            __syncthreads();
        }
    }

    if (stage == 1u) {
        if (threadIdx.x == 0u) atomicAdd(&report->acc_token, acc_sink);
        return;
    }

    if constexpr (Mode != mode_compute) {
        // ---- owner duty (gather + parallel tower reduce + certify)
        if (block < kDutyBlocks) {
            const unsigned int owned =
                (nslabs > rank ? (nslabs - 1u - rank) / kWorld + 1u : 0u);
            const unsigned int duty_items = owned * kReduceSegs;
            for (unsigned int k = block; k < duty_items; k += kDutyBlocks) {
                const unsigned int slab = rank + (k / kReduceSegs) * kWorld;
                const unsigned int seg = k % kReduceSegs;
                cd::owner_gather og{};
                if (threadIdx.x == 0u) {
                    unsigned int tries = 0u;
                    do {
                        cd::owner_gather_ready_into(g, pv, slab, epoch, kSpin,
                                                    og);
                    } while (!og.ready && ++tries < kAttempts);
                    if (!og.ready) atomicAdd(&report->gather_timeouts, 1u);
                }
                __syncthreads();
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
            }
        }
        // ---- consume + verify
        for (unsigned int slab = block; slab < nslabs; slab += kBlocks) {
            kd::wait_result64 w;
            if (threadIdx.x == 0u) {
                unsigned int tries = 0u;
                do {
                    kd::init_wait_result(w);
                    cd::mag_wait_into(g, pv, slab, epoch, kSpin, w);
                } while (!w.ready && ++tries < kAttempts);
                if (!w.ready) atomicAdd(&report->mag_timeouts, 1u);
            }
            __syncthreads();
            std::byte* dst =
                staging + static_cast<std::size_t>(slab) * g.slab_bytes();
            cd::mag_pull(g, pv, slab, dst, threadIdx.x, blockDim.x);
            __syncthreads();
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
            // burst2: the post-boundary GEMM consumes this slab AS IT LANDS
            // — boundary-N tail hides under layer-N+1 compute, per slab.
            acc_sink += mfma_burst(dst, g.rows_in(slab), k_inner, threadIdx.x);
        }
    }

    if (threadIdx.x == 0u) {
        atomicAdd(&report->acc_token, acc_sink);
    }
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
                              rank_report* rep, unsigned int bits) {
    switch (depth) {
        case 8:
            boundary_kernel<Mode, 8u><<<kBlocks, kThreads, 0, s>>>(
                pv, tokens, slab_rows, rank, epoch, k_inner, stage, bps,
                rcells, src, stage_buf, rep, bits);
            break;
        case 16:
            boundary_kernel<Mode, 16u><<<kBlocks, kThreads, 0, s>>>(
                pv, tokens, slab_rows, rank, epoch, k_inner, stage, bps,
                rcells, src, stage_buf, rep, bits);
            break;
        case 32:
            boundary_kernel<Mode, 32u><<<kBlocks, kThreads, 0, s>>>(
                pv, tokens, slab_rows, rank, epoch, k_inner, stage, bps,
                rcells, src, stage_buf, rep, bits);
            break;
        default:
            boundary_kernel<Mode, 4u><<<kBlocks, kThreads, 0, s>>>(
                pv, tokens, slab_rows, rank, epoch, k_inner, stage, bps,
                rcells, src, stage_buf, rep, bits);
            break;
    }
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "fused";
    const unsigned int tokens = argc > 2 ? std::atoi(argv[2]) : 16384u;
    const unsigned int slab_rows = argc > 3 ? std::atoi(argv[3]) : 256u;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 7;
    const unsigned int depth = argc > 5 ? std::atoi(argv[5]) : 4u;
    const unsigned int k_inner = argc > 6 ? std::atoi(argv[6]) : 64u;
    const unsigned int bps = argc > 7 ? std::atoi(argv[7]) : 2u;

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
    if (mode == "rccl") {
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

    std::vector<double> walls;
    unsigned int rccl_verify_errors = 0;
    for (int it = 0; it < iters; ++it) {
        const std::uint32_t epoch = static_cast<std::uint32_t>(it + 1);
        for (unsigned int r = 0; r < kWorld; ++r) {
            HIP_CHECK(hipSetDevice(r));
            HIP_CHECK(hipMemsetAsync(s32[r], 0, nslabs * 4, str[r]));
            HIP_CHECK(hipMemsetAsync(rcell[r], 0, nslabs * 4, str[r]));
            HIP_CHECK(hipMemsetAsync(accum[r], 0, accum_b, str[r]));
            HIP_CHECK(hipMemsetAsync(rep[r], 0, sizeof(rank_report), str[r]));
            if (mode == "rccl") {
                // rccl arm reduces IN PLACE on a copy of src in staging
                HIP_CHECK(hipMemcpyAsync(stage[r], src[r], src_b,
                                         hipMemcpyDeviceToDevice, str[r]));
            }
        }
        for (unsigned int r = 0; r < kWorld; ++r) {
            HIP_CHECK(hipSetDevice(r));
            HIP_CHECK(hipStreamSynchronize(str[r]));
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (unsigned int r = 0; r < kWorld; ++r) {
            HIP_CHECK(hipSetDevice(r));
            if (mode == "compute") {
                launch_mode_depth<mode_compute>(depth, str[r], pv, tokens,
                                                slab_rows, r, epoch, k_inner,
                                                1u, bps, rcell[r], src[r], src[r],
                                                rep[r], expect_bits);
                launch_mode_depth<mode_compute>(depth, str[r], pv, tokens,
                                                slab_rows, r, epoch, k_inner,
                                                2u, bps, rcell[r], src[r], src[r],
                                                rep[r], expect_bits);
            } else if (mode == "rccl") {
                launch_mode_depth<mode_compute>(depth, str[r], pv, tokens,
                                                slab_rows, r, epoch, k_inner,
                                                1u, bps, rcell[r], src[r], stage[r],
                                                rep[r], expect_bits);
            } else if (mode == "phased") {
                launch_mode_depth<mode_phased>(depth, str[r], pv, tokens,
                                               slab_rows, r, epoch, k_inner,
                                               0u, bps, rcell[r], src[r], stage[r],
                                               rep[r], expect_bits);
            } else {
                launch_mode_depth<mode_fused>(depth, str[r], pv, tokens,
                                              slab_rows, r, epoch, k_inner,
                                              0u, bps, rcell[r], src[r], stage[r],
                                              rep[r], expect_bits);
            }
        }
        if (mode == "rccl") {
            NCCL_CHECK(ncclGroupStart());
            for (unsigned int r = 0; r < kWorld; ++r) {
                NCCL_CHECK(ncclAllReduce(stage[r], stage[r],
                                         logical / 2, ncclBfloat16, ncclSum,
                                         comms[r], str[r]));
            }
            NCCL_CHECK(ncclGroupEnd());
            for (unsigned int r = 0; r < kWorld; ++r) {
                HIP_CHECK(hipSetDevice(r));
                launch_mode_depth<mode_compute>(depth, str[r], pv, tokens,
                                                slab_rows, r, epoch, k_inner,
                                                2u, bps, rcell[r], src[r], stage[r],
                                                rep[r], expect_bits);
            }
        }
        for (unsigned int r = 0; r < kWorld; ++r) {
            HIP_CHECK(hipSetDevice(r));
            HIP_CHECK(hipStreamSynchronize(str[r]));
        }
        const auto t1 = std::chrono::steady_clock::now();
        walls.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
        if (mode == "rccl" && it == iters - 1) {
            unsigned short sample[64];
            HIP_CHECK(hipSetDevice(0));
            HIP_CHECK(hipMemcpy(sample, stage[0], sizeof(sample),
                                hipMemcpyDeviceToHost));
            for (auto sv : sample) {
                if (sv != expect_bits) ++rccl_verify_errors;
            }
        }
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
    std::sort(walls.begin(), walls.end());
    const double med = walls[walls.size() / 2];
    std::printf(
        "BOUNDARY_BENCH mode=%s depth=%u tokens=%u slab_rows=%u k_inner=%u "
        "iters=%d\n  wall_us median=%.1f min=%.1f max=%.1f\n"
        "  gather_timeouts=%u mag_timeouts=%u verify_errors=%u\n%s\n",
        mode.c_str(), depth, tokens, slab_rows, k_inner, iters, med,
        walls.front(), walls.back(), gt, mt, errs,
        (errs == 0 && gt == 0 && mt == 0) ? "BOUNDARY_BENCH_PASS"
                                          : "BOUNDARY_BENCH_FAIL");
    if (mode == "rccl") {
        for (unsigned int r = 0; r < kWorld; ++r) ncclCommDestroy(comms[r]);
    }
    return (errs == 0 && gt == 0 && mt == 0) ? 0 : 1;
}

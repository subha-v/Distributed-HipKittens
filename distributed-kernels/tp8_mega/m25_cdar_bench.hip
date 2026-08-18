/**
 * @file
 * @brief G25-0b: the CDAR microbench (M25_TP8_MEGA_DESIGN.md gate ladder).
 *
 * v1. Measures one full CDAR boundary (ERS commit -> owner gather/certify ->
 * MAG pull) end-to-end on 8 GPUs, single process, one kernel per device.
 * Reports effective all-reduce bandwidth (logical bytes = tokens x row_bytes
 * over median wall) plus exact-value verification (every rank contributes
 * value rank+1; every consumed element must equal 36.0, exact in bf16).
 *
 * v1 changes over v0, from the first measured matrix (41.1 / 22.4 GB/s at
 * the 235 MB class — 5-6x under the ~350-420 GB/s fabric ceiling, all
 * correctness PASS):
 *  - DEPTH IS A SWEPT AXIS (argv), dispatched over template instantiations.
 *    The depth-4 law was measured for epilogues CO-RESIDENT with MFMA
 *    waves; a pure-transport kernel is in-flight-starved at 4. K4's optimum
 *    is workload-dependent — this sweep measures the transport-only curve.
 *    (A bench kernel has no register cliff at stake; the exp_24 one-copy
 *    discipline still applies inside REAL kernels.)
 *  - BLOCKS-PER-SLAB is a swept axis: v0 left half the grid idle and one
 *    block's waves serialized each slab's entire remote stream.
 *  - The towers owner-reduce is vectorized (bf16x2 loads) and parallelized
 *    across 32 duty blocks with a per-slab dynamic reduce-arrival; the last
 *    reduce segment certifies. v0's single-block scalar loop dominated the
 *    towers wall.
 *
 * Still v0-honest: no co-resident MFMA (transport ceiling, not in-mega
 * number); memset+launch inside the timed window (noise at 235 MB, an upper
 * bound at the decode class); RCCL comparison runs separately.
 *
 * Build: hipcc --offload-arch=gfx950 -std=c++20 -O3 -I <root> \
 *        m25_cdar_bench.hip -o m25_cdar_bench
 * Run:   ./m25_cdar_bench <tokens> <slab_rows> <atomic|towers> <iters> \
 *        [depth=4|8|16|32] [blocks_per_slab]
 */

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

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
static constexpr unsigned int kRowBytes = 14336;  // 7168 x bf16
static constexpr unsigned int kBlocks = 64;
static constexpr unsigned int kThreads = 256;
static constexpr unsigned int kDutyBlocks = 32;   // owner-reduce/certify pool
static constexpr unsigned int kReduceSegs = 8;    // reduce segments per slab
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

struct rank_report {
    unsigned int gather_timeouts;
    unsigned int mag_timeouts;
    unsigned int verify_errors;
    unsigned int slabs_certified;
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

template<cd::transport T, unsigned int Depth>
__global__ __launch_bounds__(kThreads, 1) void cdar_bench_kernel(
        cd::peer_view<kWorld> pv, unsigned int tokens, unsigned int slab_rows,
        unsigned int rank, std::uint32_t epoch, unsigned int bps,
        std::uint32_t* reduce_cells,
        const std::byte* __restrict__ src, std::byte* __restrict__ staging,
        rank_report* report, unsigned int expected_bits) {
    const cd::geometry<kWorld> g{tokens, kRowBytes, slab_rows, rank};
    const unsigned int nslabs = g.num_slabs();
    const unsigned int block = blockIdx.x;
    const unsigned int base = (rank + 1u) % kWorld;

    // ---- produce: work item = (slab, fragment); all blocks participate.
    const unsigned int items = nslabs * bps;
    for (unsigned int k = block; k < items; k += kBlocks) {
        const unsigned int slab = (base + k / bps) % nslabs;
        const unsigned int fragment = k % bps;
        const unsigned int rows_total = g.rows_in(slab);
        const unsigned int rows_per = (rows_total + bps - 1u) / bps;
        const unsigned int r0 = fragment * rows_per;
        const unsigned int rows =
            r0 >= rows_total ? 0u
                             : (rows_per < rows_total - r0 ? rows_per
                                                           : rows_total - r0);
        if (rows > 0u) {
            const std::byte* frag =
                src + static_cast<std::size_t>(slab) * g.slab_bytes() +
                static_cast<std::size_t>(r0) * kRowBytes;
            cd::ers_commit_fragment<T, Depth>(g, pv, slab, r0, rows, frag,
                                              threadIdx.x, blockDim.x);
            kd::throttle_vmcnt<0>();
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            cd::ers_local_arrive(g, pv, slab, bps, epoch, rows);
        }
        __syncthreads();
    }

    // ---- owner duty: gather, (reduce), certify. Work item =
    // (owned_slab, reduce_segment) over kDutyBlocks blocks; the LAST reduce
    // segment (dynamic arrival) releases and certifies. For the atomic
    // transport the "reduce" is empty and segment 0 alone certifies.
    if (block < kDutyBlocks) {
        const unsigned int owned =
            (nslabs > rank ? (nslabs - 1u - rank) / kWorld + 1u : 0u);
        const unsigned int segs =
            (T == cd::transport::store_towers) ? kReduceSegs : 1u;
        const unsigned int duty_items = owned * segs;
        for (unsigned int k = block; k < duty_items; k += kDutyBlocks) {
            const unsigned int slab = rank + (k / segs) * kWorld;
            const unsigned int seg = k % segs;
            cd::owner_gather og{};
            if (threadIdx.x == 0u) {
                unsigned int tries = 0u;
                do {
                    cd::owner_gather_ready_into(g, pv, slab, epoch, kSpin, og);
                } while (!og.ready && ++tries < kAttempts);
                if (!og.ready) atomicAdd(&report->gather_timeouts, 1u);
            }
            __syncthreads();
            if (T == cd::transport::store_towers) {
                // Vectorized segment reduce: bf16x2 lanes over this
                // segment's element range of the owned slab.
                const std::size_t lo =
                    static_cast<std::size_t>(slab / kWorld) * g.slab_bytes();
                const std::size_t owned_stride =
                    ((nslabs + kWorld - 1u) / kWorld) * g.slab_bytes();
                auto* acc2 = reinterpret_cast<__hip_bfloat162*>(
                    pv.accum[rank] + lo);
                const std::size_t elems2 =
                    static_cast<std::size_t>(g.rows_in(slab)) *
                    (kRowBytes / 4);
                const std::size_t per_seg =
                    (elems2 + segs - 1u) / segs;
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
                    kd::memory_scope::system>(reduce_cells + slab, segs, ra);
                if (ra.last) {
                    cd::owner_certify(g, pv, slab, epoch, g.rows_in(slab));
                    atomicAdd(&report->slabs_certified, 1u);
                }
            }
        }
    }

    // ---- consume: every block pulls its share of ALL slabs and verifies.
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
                static_cast<std::size_t>(g.rows_in(slab)) * (kRowBytes / 2);
            const std::size_t idx =
                (static_cast<std::size_t>(threadIdx.x) * 2654435761u) % elems;
            if (v[idx] != static_cast<unsigned short>(expected_bits)) {
                atomicAdd(&report->verify_errors, 1u);
            }
        }
    }
}

static unsigned short f32_to_bf16_bits(float f) {
    unsigned int u;
    std::memcpy(&u, &f, 4);
    return static_cast<unsigned short>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

template<cd::transport T>
static void launch_depth(unsigned int depth, hipStream_t s,
                         const cd::peer_view<kWorld>& pv, unsigned int tokens,
                         unsigned int slab_rows, unsigned int rank,
                         std::uint32_t epoch, unsigned int bps,
                         std::uint32_t* rcells, const std::byte* src,
                         std::byte* stage, rank_report* rep,
                         unsigned int bits) {
    switch (depth) {
        case 8:
            cdar_bench_kernel<T, 8u><<<kBlocks, kThreads, 0, s>>>(
                pv, tokens, slab_rows, rank, epoch, bps, rcells, src, stage,
                rep, bits);
            break;
        case 16:
            cdar_bench_kernel<T, 16u><<<kBlocks, kThreads, 0, s>>>(
                pv, tokens, slab_rows, rank, epoch, bps, rcells, src, stage,
                rep, bits);
            break;
        case 32:
            cdar_bench_kernel<T, 32u><<<kBlocks, kThreads, 0, s>>>(
                pv, tokens, slab_rows, rank, epoch, bps, rcells, src, stage,
                rep, bits);
            break;
        default:
            cdar_bench_kernel<T, 4u><<<kBlocks, kThreads, 0, s>>>(
                pv, tokens, slab_rows, rank, epoch, bps, rcells, src, stage,
                rep, bits);
            break;
    }
}

int main(int argc, char** argv) {
    const unsigned int tokens = argc > 1 ? std::atoi(argv[1]) : 16384u;
    const unsigned int slab_rows = argc > 2 ? std::atoi(argv[2]) : 512u;
    const std::string transport = argc > 3 ? argv[3] : "atomic";
    const int iters = argc > 4 ? std::atoi(argv[4]) : 7;
    const unsigned int depth = argc > 5 ? std::atoi(argv[5]) : 4u;
    const unsigned int bps = argc > 6 ? std::atoi(argv[6]) : 2u;
    const bool towers = transport == "towers";

    cd::geometry<kWorld> g{tokens, kRowBytes, slab_rows, 0u};
    const unsigned int nslabs = g.num_slabs();
    const std::size_t accum_b = cd::accum_bytes(g);
    const std::size_t towers_b = towers ? accum_b * kWorld : 16u;
    const std::size_t src_b = static_cast<std::size_t>(nslabs) * g.slab_bytes();
    const std::size_t sig64 = cd::signal_words64(g);

    int ndev = 0;
    HIP_CHECK(hipGetDeviceCount(&ndev));
    if (ndev < static_cast<int>(kWorld)) {
        std::fprintf(stderr, "need %u GPUs, have %d\n", kWorld, ndev);
        return 2;
    }
    for (unsigned int a = 0; a < kWorld; ++a) {
        HIP_CHECK(hipSetDevice(a));
        for (unsigned int b = 0; b < kWorld; ++b) {
            if (a == b) continue;
            int can = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&can, a, b));
            if (!can) { std::fprintf(stderr, "no P2P %u->%u\n", a, b); return 2; }
            hipError_t e = hipDeviceEnablePeerAccess(b, 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                HIP_CHECK(e);
            }
            (void)hipGetLastError();
        }
    }

    std::byte* accum[kWorld]; std::byte* twr[kWorld]; std::byte* src[kWorld];
    std::byte* stage[kWorld];
    std::uint64_t* s64[kWorld]; std::uint32_t* s32[kWorld];
    std::uint32_t* rcell[kWorld];
    rank_report* rep[kWorld];
    hipStream_t str[kWorld];
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
    for (int it = 0; it < iters; ++it) {
        const std::uint32_t epoch = static_cast<std::uint32_t>(it + 1);
        for (unsigned int r = 0; r < kWorld; ++r) {
            HIP_CHECK(hipSetDevice(r));
            HIP_CHECK(hipMemsetAsync(s32[r], 0, nslabs * 4, str[r]));
            HIP_CHECK(hipMemsetAsync(rcell[r], 0, nslabs * 4, str[r]));
            HIP_CHECK(hipMemsetAsync(accum[r], 0, accum_b, str[r]));
            HIP_CHECK(hipMemsetAsync(rep[r], 0, sizeof(rank_report), str[r]));
        }
        for (unsigned int r = 0; r < kWorld; ++r) {
            HIP_CHECK(hipSetDevice(r));
            HIP_CHECK(hipStreamSynchronize(str[r]));
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (unsigned int r = 0; r < kWorld; ++r) {
            HIP_CHECK(hipSetDevice(r));
            if (towers) {
                launch_depth<cd::transport::store_towers>(
                    depth, str[r], pv, tokens, slab_rows, r, epoch, bps,
                    rcell[r], src[r], stage[r], rep[r], expect_bits);
            } else {
                launch_depth<cd::transport::atomic_accumulate>(
                    depth, str[r], pv, tokens, slab_rows, r, epoch, bps,
                    rcell[r], src[r], stage[r], rep[r], expect_bits);
            }
        }
        for (unsigned int r = 0; r < kWorld; ++r) {
            HIP_CHECK(hipSetDevice(r));
            HIP_CHECK(hipStreamSynchronize(str[r]));
        }
        const auto t1 = std::chrono::steady_clock::now();
        walls.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    rank_report host_rep[kWorld];
    unsigned int errs = 0, gt = 0, mt = 0, cert = 0;
    for (unsigned int r = 0; r < kWorld; ++r) {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipMemcpy(&host_rep[r], rep[r], sizeof(rank_report),
                            hipMemcpyDeviceToHost));
        errs += host_rep[r].verify_errors;
        gt += host_rep[r].gather_timeouts;
        mt += host_rep[r].mag_timeouts;
        cert += host_rep[r].slabs_certified;
    }
    std::sort(walls.begin(), walls.end());
    const double med = walls[walls.size() / 2];
    const double eff = static_cast<double>(logical) / (med * 1e-6) / 1e9;
    std::printf(
        "CDAR_BENCH transport=%s depth=%u bps=%u tokens=%u slab_rows=%u "
        "slabs=%u iters=%d\n"
        "  wall_us median=%.1f min=%.1f max=%.1f\n"
        "  logical_AR_bytes=%zu effective_AR_GBps=%.1f (per-iter incl. "
        "memset+launch)\n"
        "  certified=%u/%u(x last iter) gather_timeouts=%u mag_timeouts=%u "
        "verify_errors=%u\n"
        "%s\n",
        transport.c_str(), depth, bps, tokens, slab_rows, nslabs, iters, med,
        walls.front(), walls.back(), logical, eff, cert, nslabs, gt, mt, errs,
        (errs == 0 && gt == 0 && mt == 0) ? "CDAR_BENCH_PASS"
                                          : "CDAR_BENCH_FAIL");
    return (errs == 0 && gt == 0 && mt == 0) ? 0 : 1;
}

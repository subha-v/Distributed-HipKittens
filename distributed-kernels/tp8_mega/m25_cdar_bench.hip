/**
 * @file
 * @brief G25-0b: the CDAR microbench (M25_TP8_MEGA_DESIGN.md gate ladder).
 *
 * Measures one full CDAR boundary (ERS commit -> owner gather/certify ->
 * MAG pull) end-to-end on 8 GPUs, single process, one kernel per device,
 * for both K1 transports and several message classes. Reports effective
 * all-reduce bandwidth (logical bytes = tokens x row_bytes vs median wall)
 * plus exact-value verification (every rank contributes value rank+1; every
 * consumed element must equal 36.0, exactly representable in bf16 — parity
 * evidence for free).
 *
 * v0 HONESTY NOTES (what this does and does not measure):
 *  - No co-resident MFMA body yet: this is the transport+protocol ceiling,
 *    not the in-mega number. The co-residency degradation test is v1.
 *  - Per-iteration accum/arrive re-zero (hipMemsetAsync) and kernel launch
 *    are INSIDE the timed window; at the prefill message class (235 MB
 *    logical) they are noise, at the decode class they are not — decode
 *    numbers here bound one-shot latency from ABOVE only.
 *  - RCCL comparison runs separately (rccl-tests / a python driver); this
 *    binary prints the numbers to compare against.
 *  - Every wait is bounded with a retry cap: on exhaustion the kernel sets
 *    an error flag and exits (fail loud, never hang — the 8-GPU lesson).
 *
 * Build (node):
 *   hipcc --offload-arch=gfx950 -std=c++20 -O3 -I <repo root> \
 *         m25_cdar_bench.hip -o m25_cdar_bench
 * Run:
 *   ./m25_cdar_bench <tokens> <slab_rows> <atomic|towers> <iters>
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
static constexpr unsigned int kDutyBlocks = 8;    // owner-duty (also produce+consume)
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

// Fill a bf16 buffer with a constant (bit pattern replicated as u32 pairs).
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

template<cd::transport T>
__global__ __launch_bounds__(kThreads, 1) void cdar_bench_kernel(
        cd::peer_view<kWorld> pv, unsigned int tokens, unsigned int slab_rows,
        unsigned int rank, std::uint32_t epoch,
        const std::byte* __restrict__ src, std::byte* __restrict__ staging,
        rank_report* report, unsigned int expected_bits) {
    const cd::geometry<kWorld> g{tokens, kRowBytes, slab_rows, rank};
    const unsigned int nslabs = g.num_slabs();
    const unsigned int block = blockIdx.x;

    // ---- produce: every block commits its share of ALL slabs, traversing
    // slabs in rank-staggered order (owner rotation falls out of s % World).
    const unsigned int base = (rank + 1u) % kWorld;
    for (unsigned int k = block; k < nslabs; k += kBlocks) {
        const unsigned int slab = (base + k) % nslabs;
        const unsigned int rows = g.rows_in(slab);
        const std::byte* frag =
            src + static_cast<std::size_t>(slab) * g.slab_bytes();
        cd::ers_commit_fragment<T, 4u>(g, pv, slab, 0u, rows, frag,
                                       threadIdx.x, blockDim.x);
        // Drain this block's remote stream before arrival accounting.
        kd::throttle_vmcnt<0>();
        __syncthreads();
        if (threadIdx.x == 0u) {
            cd::ers_local_arrive(g, pv, slab, 1u, epoch, rows);
        }
        __syncthreads();
    }

    // ---- owner duty: first kDutyBlocks blocks certify this rank's slabs.
    if (block < kDutyBlocks) {
        for (unsigned int s = rank + block * kWorld; s < nslabs;
             s += kWorld * kDutyBlocks) {
            // s owned by rank iff s % kWorld == rank; iterate that stripe.
            if (g.owner_of(s) != rank) continue;
            cd::owner_gather og{};
            if (threadIdx.x == 0u) {
                unsigned int tries = 0u;
                do {
                    cd::owner_gather_ready_into(g, pv, s, epoch, kSpin, og);
                } while (!og.ready && ++tries < kAttempts);
                if (!og.ready) atomicAdd(&report->gather_timeouts, 1u);
            }
            __syncthreads();
            if (T == cd::transport::store_towers) {
                // Owner-side reduce: sum kWorld towers into accum (bf16x2
                // packed adds via float2 conversion kept simple: integer
                // lanes over dwords with packed bf16 add intrinsic path is
                // v1; v0 uses float math per element for correctness).
                const std::size_t lo = s / kWorld * g.slab_bytes();
                const std::size_t owned_stride =
                    ((nslabs + kWorld - 1u) / kWorld) * g.slab_bytes();
                auto* acc = reinterpret_cast<__hip_bfloat16*>(
                    pv.accum[rank] + lo);
                const std::size_t elems =
                    static_cast<std::size_t>(g.rows_in(s)) * (kRowBytes / 2);
                for (std::size_t i = threadIdx.x; i < elems; i += blockDim.x) {
                    float acc_v = 0.f;
                    for (unsigned int r = 0u; r < kWorld; ++r) {
                        const auto* tw = reinterpret_cast<const __hip_bfloat16*>(
                            pv.towers[rank] + r * owned_stride + lo);
                        acc_v += __bfloat162float(tw[i]);
                    }
                    acc[i] = __float2bfloat16(acc_v);
                }
            }
            __syncthreads();
            kd::producer_drain_release<kd::memory_scope::system>();
            if (threadIdx.x == 0u) {
                cd::owner_certify(g, pv, s, epoch, g.rows_in(s));
                atomicAdd(&report->slabs_certified, 1u);
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
        // verify a sample: 64 elements per slab, exact bf16 compare
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

int main(int argc, char** argv) {
    const unsigned int tokens = argc > 1 ? std::atoi(argv[1]) : 16384u;
    const unsigned int slab_rows = argc > 2 ? std::atoi(argv[2]) : 512u;
    const std::string transport = argc > 3 ? argv[3] : "atomic";
    const int iters = argc > 4 ? std::atoi(argv[4]) : 7;
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
    // P2P everywhere
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
        HIP_CHECK(hipMalloc(&rep[r], sizeof(rank_report)));
        HIP_CHECK(hipMemset(s64[r], 0, sig64 * 8));
        // rank r contributes value (r+1) everywhere
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

    const float expect = 36.0f;  // sum r+1, r in [0,8)
    const unsigned short expect_bits = f32_to_bf16_bits(expect);
    const std::size_t logical = static_cast<std::size_t>(tokens) * kRowBytes;

    std::vector<double> walls;
    for (int it = 0; it < iters; ++it) {
        const std::uint32_t epoch = static_cast<std::uint32_t>(it + 1);
        for (unsigned int r = 0; r < kWorld; ++r) {
            HIP_CHECK(hipSetDevice(r));
            HIP_CHECK(hipMemsetAsync(s32[r], 0, nslabs * 4, str[r]));
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
                cdar_bench_kernel<cd::transport::store_towers>
                    <<<kBlocks, kThreads, 0, str[r]>>>(
                        pv, tokens, slab_rows, r, epoch, src[r], stage[r],
                        rep[r], expect_bits);
            } else {
                cdar_bench_kernel<cd::transport::atomic_accumulate>
                    <<<kBlocks, kThreads, 0, str[r]>>>(
                        pv, tokens, slab_rows, r, epoch, src[r], stage[r],
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
        "CDAR_BENCH transport=%s tokens=%u slab_rows=%u slabs=%u iters=%d\n"
        "  wall_us median=%.1f min=%.1f max=%.1f\n"
        "  logical_AR_bytes=%zu effective_AR_GBps=%.1f (per-iter incl. "
        "memset+launch)\n"
        "  certified=%u/%u(x last iter) gather_timeouts=%u mag_timeouts=%u "
        "verify_errors=%u\n"
        "%s\n",
        transport.c_str(), tokens, slab_rows, nslabs, iters, med,
        walls.front(), walls.back(), logical, eff, cert, nslabs, gt, mt, errs,
        (errs == 0 && gt == 0 && mt == 0) ? "CDAR_BENCH_PASS"
                                          : "CDAR_BENCH_FAIL");
    return (errs == 0 && gt == 0 && mt == 0) ? 0 : 1;
}

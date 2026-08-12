// ===========================================================================
// exp_22 -- resource saturation vs CTA count on MI350X (NanoFlow Fig 7 analog)
//
// Standalone ubench. It is NOT part of the megakernel build: it links its own
// host driver, owns all 8 devices from ONE process, and reaches peer memory
// through HIP peer access rather than mori's symmetric heap. It does not
// include, edit, or depend on k0pf6gm_device_tile_mps.hip or
// moe_mps_adapter.cuh (another agent owns those tonight); the one device idiom
// it needs from that tree -- the s_memrealtime read -- is COPIED here, and the
// transport comes from the shared library headers (store_peer_packets_multi /
// translate_peer), which are the same headers the megakernel calls.
//
// Three resource classes, one code object, one role-split kernel:
//   mode a "mfma" -- LDS-staged bf16 MFMA task loop              -> TFLOPS
//   mode b "hbm"  -- M8-shaped 8:1 local slot reduce             -> GB/s
//   mode c "xgmi" -- MPS-shaped peer row push                    -> GB/s egress
//
// Every arm -- isolated, concurrent, reserved_only -- runs the SAME __global__
// with the SAME register/LDS footprint, so an arm-to-arm difference can never
// be an occupancy or register-allocation artifact. The three arms differ only
// in which role is given work.
//
// Bump E22_SRC_REV on every edit (the K0P6_MPS_SRC_REV discipline: a stale
// object is indistinguishable from a null result).
// ===========================================================================
#define E22_SRC_REV 9

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "cdna4/ops/group/distributed/packet.cuh"
#include "cdna4/ops/group/distributed/peer.cuh"
#include "cdna4/ops/group/distributed/sync.cuh"
#include "cdna4/types/global/pgl.cuh"

namespace kd = kittens::distributed;

// --------------------------------------------------------------- constants
static constexpr int kThreads = 256;   // 4 waves of 64, as in the megakernel
static constexpr int kWaves = kThreads / 64;
static constexpr int kMaxCtas = 256;   // 256 CUs on MI350X
// 96 KiB pins ONE workgroup per CU on gfx950's 160 KiB LDS. Occupancy 1 is a
// property of the kernel being modelled, and it is the premise of the figure:
// there is no oversubscription regime on CDNA4 to hide a second workload in.
static constexpr int kLdsBytes = 98304;
static constexpr int kStageBytesPerTile = kThreads * 4 * 16;   // 16 KiB per CTA
// store_peer_packets_multi takes its pointer arrays BY REFERENCE, so the
// caller's void*[Depth] + const void*[Depth] pair has its address taken and the
// compiler promotes it out of registers. MEASURED (see result.md): with this
// 96 KiB static array it promotes to LDS, +16 B/lane/region exactly (98,304 ->
// 114,688 at depth 4 -> 131,072 at depth 8) and keeps scratch at zero; shrink
// the static array to make room and it instead promotes to SCRATCH (80/144
// B/lane) and lets occupancy rise to 2 blocks/CU, which would destroy the
// premise of the figure. So the static size is held fixed at every depth: LDS
// differs across depths but occupancy -- the thing the CTA axis means -- does
// not. exp_04 saw the scratch half of this same behaviour in the megakernel.
template<int Depth>
struct LdsBudget {
    static constexpr int kStatic = kLdsBytes;
    static constexpr int kWave = kStatic / kWaves;
    static_assert(kStatic >= kStageBytesPerTile, "LDS budget too small to stage");
};
static constexpr int kRealtimeHz = 100000000;                  // s_memrealtime
static constexpr double kFlopsPerMfma = 2.0 * 32.0 * 32.0 * 16.0;   // 32,768

using v16f32 = float __attribute__((ext_vector_type(16)));
using v8bf16 = __bf16 __attribute__((ext_vector_type(8)));
// The nontemporal builtins reject HIP_vector_type; they need a native vector,
// which is also what store_peer_packets_streaming uses for the same reason.
using v4u32 = unsigned int __attribute__((ext_vector_type(4)));

enum : int { kModeNone = 0, kModeHbm = 1, kModeXgmi = 2 };
enum : int { kFanSingle = 0, kFanRr = 1 };
enum : int { kTsRole = 0, kTsBegin = 1, kTsEnd = 2, kTsWork = 3, kTsStride = 4 };

// s_memrealtime is an SMEM read: the destination SGPR pair is invalid until
// s_waitcnt lgkmcnt(0). Copied in full (including the reason) from
// moe_mps_adapter.cuh's realtime_now(): a site not preceded by LDS traffic
// silently reads a stale register without the explicit wait.
__device__ __forceinline__ std::uint64_t realtime_now() {
    std::uint64_t t;
    asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)" : "=s"(t) :: "memory");
    return t;
}

__device__ __forceinline__ void drain_vm() {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

struct Params {
    // ---- roles
    int res_ctas;                    // C: blockIdx.x < res_ctas => resource role
    int mode;                        // resource-role body
    unsigned long long res_rows;     // 0 => resource role idle (reserved_only)
    unsigned long long cmp_tiles;    // 0 => compute role idle (isolated)
    int mfma_per_tile;
    // ---- mode c (xgmi push)
    const std::byte* c_src;
    std::byte* c_dst_local;
    const void* c_dst_base;
    kittens::peer_bases<std::byte> c_peers;
    int me;
    int world;
    unsigned int row_bytes;
    unsigned int src_rows_cap;
    unsigned int dst_rows_cap;       // rows per source-rank partition
    int fanout;
    // ---- mode c protocol overlay (H3)
    int protocol;
    int proto_g;
    int flush_rows;
    unsigned int* proto_probe;
    unsigned int* proto_arrival;
    // ---- mode b (hbm reduce)
    const std::byte* b_slots;
    unsigned long long b_slot_stride;
    std::byte* b_out;
    unsigned int b_rows_cap;
    // ---- mode a (mfma)
    const uint4* a_src;
    unsigned int a_vec4_cap;
    // ---- outputs
    float* sink;
    std::uint64_t* ts;
};

// --------------------------------------------------------------- mode a body
// LDS-staged bf16 MFMA. Four independent accumulators keep the MAI pipe fed.
// `mfma_per_tile` is the arithmetic-intensity knob (MFMAs per 16 KiB staged),
// so one body serves both as the compute-capacity curve and as an overlay
// partner that really competes for memory instead of trivially not competing.
template<int WaveLds>
__device__ __forceinline__ void mfma_body(const Params p, char* lds,
                                          unsigned long long* work) {
    const unsigned int tid = threadIdx.x;
    const int wave = (int)(tid / 64u);
    const unsigned int lane = tid % 64u;
    const v8bf16* const win =
        reinterpret_cast<const v8bf16*>(lds + (size_t)wave * WaveLds);
    // Every index below is masked with (slots - 1), so it is in [0, slots - 1]
    // whether or not `slots` is a power of two.
    constexpr unsigned int kWinSlots = (unsigned int)WaveLds / 16u;

    v16f32 acc0 = (v16f32)(0.0f), acc1 = (v16f32)(0.0f);
    v16f32 acc2 = (v16f32)(0.0f), acc3 = (v16f32)(0.0f);

    unsigned long long mfmas = 0;
    for (unsigned long long tile = 0; tile < p.cmp_tiles; ++tile) {
        const unsigned int base =
            (unsigned int)(((tile * kThreads + tid) * 4ull) % p.a_vec4_cap);
        uint4* const stage = reinterpret_cast<uint4*>(lds) + (size_t)tid * 4u;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            stage[i] = p.a_src[(base + (unsigned int)i) % p.a_vec4_cap];
        }
        __syncthreads();
#pragma unroll 4
        for (int m = 0; m < p.mfma_per_tile; m += 4) {
            const unsigned int o = ((unsigned int)m * 64u + lane * 3u +
                                    (unsigned int)tile) & (kWinSlots - 1u);
            const v8bf16 a0 = win[o];
            const v8bf16 b0 = win[(o + 97u) & (kWinSlots - 1u)];
            const v8bf16 a1 = win[(o + 211u) & (kWinSlots - 1u)];
            const v8bf16 b1 = win[(o + 307u) & (kWinSlots - 1u)];
            acc0 = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a0, b0, acc0, 0, 0, 0);
            acc1 = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a1, b1, acc1, 0, 0, 0);
            acc2 = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a0, b1, acc2, 0, 0, 0);
            acc3 = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a1, b0, acc3, 0, 0, 0);
        }
        __syncthreads();
        mfmas += (unsigned long long)((p.mfma_per_tile + 3) / 4) * 4ull;
    }

    // Anti-DCE sink: the accumulators must be observable or the loop is dead.
    float s = 0.0f;
#pragma unroll
    for (int e = 0; e < 16; ++e) s += acc0[e] + acc1[e] + acc2[e] + acc3[e];
    if (lane == 0u && p.sink != nullptr) {
        p.sink[(size_t)blockIdx.x * kWaves + wave] = s;
    }
    *work = mfmas;   // MFMAs issued by THIS wave
}

// --------------------------------------------------------------- mode b body
// M8-shaped: 8 bf16 slot rows in, 1 row out; 14 chunks of 64 lanes x 16 B per
// row; one row per wave at a time (NT=4). Non-temporal on both sides -- the
// payload is read once and written once, exactly as in the combine. The reduce
// runs in fp32 on unpacked halves so the check below can be exact.
__device__ __forceinline__ void hbm_body(const Params p, unsigned long long* work) {
    const unsigned int tid = threadIdx.x;
    const int wave = (int)(tid / 64u);
    const unsigned int lane = tid % 64u;
    const unsigned int chunks = p.row_bytes / (64u * 16u);
    const unsigned long long rows_per_cta =
        p.res_rows / (unsigned long long)p.res_ctas;
    const unsigned long long first = (unsigned long long)blockIdx.x * rows_per_cta;
    unsigned long long done = 0;

    for (unsigned long long i = (unsigned long long)wave; i < rows_per_cta;
         i += kWaves) {
        const size_t roff = (size_t)((first + i) % p.b_rows_cap) * p.row_bytes;
        v4u32* const out = reinterpret_cast<v4u32*>(p.b_out + roff);
        for (unsigned int c = 0; c < chunks; ++c) {
            const size_t off = (size_t)c * 64u + lane;
            float acc[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[k] = 0.0f;
#pragma unroll
            for (int sl = 0; sl < 8; ++sl) {
                const v4u32* const in = reinterpret_cast<const v4u32*>(
                    p.b_slots + (size_t)sl * p.b_slot_stride + roff);
                const v4u32 v = __builtin_nontemporal_load(in + off);
                const __hip_bfloat16* const h =
                    reinterpret_cast<const __hip_bfloat16*>(&v);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[k] += __bfloat162float(h[k]);
            }
            v4u32 o;
            __hip_bfloat16* const oh = reinterpret_cast<__hip_bfloat16*>(&o);
#pragma unroll
            for (int k = 0; k < 8; ++k) oh[k] = __float2bfloat16(acc[k]);
            __builtin_nontemporal_store(o, out + off);
        }
        ++done;
    }
    *work = done;
}

// --------------------------------------------------------------- mode c body
// The MPS service pusher: local row -> peer row through translate_peer +
// store_peer_packets_multi<Depth>. Depth is the plan's MLP axis (loads issued
// before the first store). The isolated arm carries NO flag, fence, or probe:
// it prices payload transport alone. --protocol re-adds the megakernel's
// per-flush drain + system release + arrival RMW + g-wide probe read, so
// payload and protocol are separable inside one binary (exp_20 as a curve).
template<int Depth>
__device__ __forceinline__ void xgmi_body(const Params p,
                                          unsigned long long* work) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long rows_per_cta =
        p.res_rows / (unsigned long long)p.res_ctas;
    const unsigned long long first = (unsigned long long)blockIdx.x * rows_per_cta;
    const int npeer = p.world - 1;
    unsigned long long done = 0;
    unsigned int since_flush = 0;

    for (unsigned long long g = 0; g < rows_per_cta; g += Depth) {
        void* dsts[Depth];
        const void* srcs[Depth];
#pragma unroll
        for (int j = 0; j < Depth; ++j) {
            const unsigned long long r = first + g + (unsigned long long)j;
            srcs[j] = p.c_src + (size_t)(r % p.src_rows_cap) * p.row_bytes;
            int peer;
            unsigned long long drow;
            if (p.fanout == kFanSingle) {
                peer = (p.me + 1) % p.world;
                drow = r;
            } else {
                peer = (p.me + 1 + (int)(r % (unsigned long long)npeer)) % p.world;
                drow = r / (unsigned long long)npeer;
            }
            // Destination is partitioned by SOURCE rank, mirroring the
            // slots[world][MAXTOK][H] layout the kernel pushes into.
            const size_t off = ((size_t)p.me * p.dst_rows_cap +
                                (size_t)(drow % p.dst_rows_cap)) * p.row_bytes;
            dsts[j] = kd::translate_peer<8>(p.c_dst_local + off, p.c_dst_base,
                                            p.c_peers, peer);
        }
        kd::store_peer_packets_multi<Depth>(dsts, srcs, p.row_bytes, tid, kThreads);
        done += Depth;
        since_flush += Depth;

        if (p.protocol && since_flush >= (unsigned int)p.flush_rows) {
            since_flush = 0;
            drain_vm();
            kd::thread_release<kd::memory_scope::system>();
            if (tid < 32u) {   // the adapter's 32 live bookkeeping lanes
                (void)kd::fetch_add_relaxed<kd::memory_scope::system>(
                    p.proto_arrival + (tid & 7u), 1u);
                for (int q = 0; q < p.proto_g; ++q) {
                    // acquire-carrying read: fetch_add(., 0) at acq_rel, the
                    // adapter's group-completion probe idiom.
                    (void)kd::detail::fetch_add_acq_rel<kd::memory_scope::agent>(
                        p.proto_probe + ((unsigned int)q & 63u), 0u);
                }
            }
        }
    }
    if (p.protocol) {
        drain_vm();
        kd::thread_release<kd::memory_scope::system>();
    }
    *work = done;
}

// ------------------------------------------------------------ role-split kernel
template<int Depth>
__device__ __forceinline__ void sat_body(const Params p) {
    __shared__ char lds[LdsBudget<Depth>::kStatic];
    const bool resource = (int)blockIdx.x < p.res_ctas;
    unsigned long long work = 0;

    const std::uint64_t t0 = realtime_now();
    if (resource) {
        if (p.res_rows != 0ull) {
            if (p.mode == kModeXgmi) xgmi_body<Depth>(p, &work);
            else if (p.mode == kModeHbm) hbm_body(p, &work);
        }
    } else {
        if (p.cmp_tiles != 0ull) mfma_body<LdsBudget<Depth>::kWave>(p, lds, &work);
    }
    drain_vm();
    const std::uint64_t t1 = realtime_now();

    if (threadIdx.x == 0u && p.ts != nullptr) {
        std::uint64_t* const s = p.ts + (size_t)blockIdx.x * kTsStride;
        s[kTsRole] = resource ? 1ull : 2ull;
        s[kTsBegin] = t0;
        s[kTsEnd] = t1;
        s[kTsWork] = work;
    }
}

__global__ __launch_bounds__(kThreads) void sat_kernel_d1(Params p) { sat_body<1>(p); }
__global__ __launch_bounds__(kThreads) void sat_kernel_d4(Params p) { sat_body<4>(p); }
__global__ __launch_bounds__(kThreads) void sat_kernel_d8(Params p) { sat_body<8>(p); }

// ------------------------------------------------------------- fill / verify
__global__ void fill_words_kernel(std::byte* base, unsigned long long words,
                                  unsigned long long salt) {
    std::uint64_t* const p = reinterpret_cast<std::uint64_t*>(base);
    for (unsigned long long i =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < words; i += (unsigned long long)gridDim.x * blockDim.x) {
        std::uint64_t z = i * 0x9E3779B97F4A7C15ull + salt;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        p[i] = z ? z : 1ull;   // never emit the all-zero word
    }
}

__global__ void fill_bf16_slot_kernel(std::byte* base, unsigned long long elems,
                                      float value) {
    __hip_bfloat16* const p = reinterpret_cast<__hip_bfloat16*>(base);
    const __hip_bfloat16 v = __float2bfloat16(value);
    for (unsigned long long i =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < elems; i += (unsigned long long)gridDim.x * blockDim.x) {
        p[i] = v;
    }
}

__global__ void fold_kernel(const std::uint64_t* p, unsigned long long words,
                            std::uint64_t poison, unsigned long long* out) {
    std::uint64_t x = 0;
    unsigned long long bad = 0;
    for (unsigned long long i =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < words; i += (unsigned long long)gridDim.x * blockDim.x) {
        const std::uint64_t v = p[i];
        x ^= v;
        if (v == poison) ++bad;
    }
    atomicXor(reinterpret_cast<unsigned long long*>(out), (unsigned long long)x);
    atomicAdd(out + 1, bad);
}

__global__ void check_bf16_kernel(const std::byte* base, unsigned long long elems,
                                  float expect, unsigned long long* out) {
    const __hip_bfloat16* const p = reinterpret_cast<const __hip_bfloat16*>(base);
    unsigned long long bad = 0;
    for (unsigned long long i =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < elems; i += (unsigned long long)gridDim.x * blockDim.x) {
        if (__bfloat162float(p[i]) != expect) ++bad;
    }
    atomicAdd(out, bad);
}

// ================================================================ host driver
#define HC(call)                                                               \
    do {                                                                       \
        const hipError_t e_ = (call);                                          \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "[e22] %s:%d %s -> %s\n", __FILE__, __LINE__, \
                         #call, hipGetErrorString(e_));                        \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

namespace {

struct Sizes {
    unsigned int row_bytes = 14336;   // one bf16 row of H = 7168: the push unit
    unsigned int src_rows = 65536;    // 939 MB, >> 256 MB LLC: reads hit HBM
    unsigned int dst_rows = 32768;    // per source-rank partition, x8 = 3.8 GB;
                                      // >> the receiver's 256 MB LLC, so peer
                                      // writes are not absorbed by its cache
    unsigned int b_rows = 16384;      // per slot, x9 buffers = 2.1 GB
    unsigned long long a_bytes = 512ull << 20;
    int world = 8;
};

struct Config {
    Sizes sz;
    std::string out = "saturation.jsonl";
    std::string tier = "plan";
    int rotations = 5;
    int warmup = 2;
    int uncached = 1;              // mori's symmetric heap is HIP-VMM Uncached
    int verify = 1;
    int grid = kMaxCtas;
    double target_launch_us = 25000.0;
    double concurrent_slack = 1.35;
    double min_overlap = 0.90;
    int max_retries = 2;
    double peak_xgmi_link = 76.8;  // GB/s per xGMI link per direction
    double peak_xgmi_agg = 537.6;  // 7 x 76.8 GB/s aggregate egress
    double peak_hbm = 8000.0;      // GB/s HBM3E
    double peak_tflops = 2300.0;   // MI350X dense bf16 (spec sheet)
    double max_seconds = 0.0;      // 0 = no budget
    double xgmi_mb_per_cta = 45.0; // x(world-1) for rr7; see xgmi_rows_for
    double hbm_mb_per_cta = 48.0;
    int list_only = 0;
    int dry_run = 0;               // emit synthetic records, never touch a GPU
};

struct Dev {
    std::byte* c_src = nullptr;
    std::byte* c_dst = nullptr;
    std::byte* b_slots = nullptr;
    std::byte* b_out = nullptr;
    uint4* a_src = nullptr;
    unsigned int* proto_probe = nullptr;
    unsigned int* proto_arrival = nullptr;
    float* sink = nullptr;
    std::uint64_t* ts = nullptr;
    unsigned long long* scratch = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t ev0 = nullptr;
    hipEvent_t ev1 = nullptr;
};

struct Point {
    std::string mode = "mfma";       // mfma | hbm | xgmi
    int ctas = 0;
    int mlp = 0;                     // mode c: load depth; mode a: mfma_per_tile
    std::string fanout = "na";       // single | rr7 | na
    std::string conc = "isolated";   // isolated | concurrent | reserved_only
    int protocol = 0;
    int proto_g = 0;
    int pushers = 1;                 // 1 = one pusher device, 8 = all push
    unsigned int row_bytes = 14336;
    std::string work_mode = "per_cta";   // per_cta | fixed
    int heap_uncached = 1;

    std::string key() const {
        char b[256];
        std::snprintf(b, sizeof b, "%s|c%d|m%d|%s|%s|p%d|g%d|n%d|rb%u|%s|u%d",
                      mode.c_str(), ctas, mlp, fanout.c_str(), conc.c_str(),
                      protocol, proto_g, pushers, row_bytes, work_mode.c_str(),
                      heap_uncached);
        return std::string(b);
    }
    std::string pair_key() const {   // identical across the three arms
        char b[256];
        std::snprintf(b, sizeof b, "%s|c%d|m%d|%s|p%d|g%d|n%d|rb%u|%s|u%d",
                      mode.c_str(), ctas, mlp, fanout.c_str(), protocol,
                      proto_g, pushers, row_bytes, work_mode.c_str(),
                      heap_uncached);
        return std::string(b);
    }
};

Config g_cfg;
std::vector<Dev> g_dev;
int g_world = 0;
std::map<std::pair<int, int>, double> g_tile_rate;   // (shape, ctas) -> tiles/us
std::map<std::string, double> g_isolated_us;         // pair_key -> wall us
std::map<std::string, unsigned long long> g_paired_tiles;   // pair_key -> tiles

void* alloc_dev(std::size_t bytes, bool uncached) {
    void* p = nullptr;
    if (uncached) {
        HC(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached));
    } else {
        HC(hipMalloc(&p, bytes));
    }
    return p;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return (n & 1u) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = q * (double)(v.size() - 1);
    const std::size_t lo = (std::size_t)idx;
    const std::size_t hi = std::min(v.size() - 1, lo + 1);
    return v[lo] + (idx - (double)lo) * (v[hi] - v[lo]);
}

double stdev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = 0.0;
    for (double x : v) m += x;
    m /= (double)v.size();
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (double)(v.size() - 1));
}

double now_s() {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

// --------------------------------------------------------------- allocation
void setup_devices() {
    int n = 0;
    HC(hipGetDeviceCount(&n));
    g_world = std::min(n, g_cfg.sz.world);
    if (g_world < 2) {
        std::fprintf(stderr, "[e22] need >= 2 devices, saw %d\n", n);
        std::exit(2);
    }
    g_dev.resize((std::size_t)g_world);

    for (int d = 0; d < g_world; ++d) {
        HC(hipSetDevice(d));
        for (int q = 0; q < g_world; ++q) {
            if (q == d) continue;
            int can = 0;
            HC(hipDeviceCanAccessPeer(&can, d, q));
            if (!can) {
                std::fprintf(stderr, "[e22] no peer access %d->%d\n", d, q);
                std::exit(2);
            }
            const hipError_t e = hipDeviceEnablePeerAccess(q, 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                std::fprintf(stderr, "[e22] enable peer %d->%d: %s\n", d, q,
                             hipGetErrorString(e));
                std::exit(2);
            }
        }
    }

    const Sizes& s = g_cfg.sz;
    const bool unc = g_cfg.uncached != 0;
    const std::size_t src_bytes = (std::size_t)s.src_rows * s.row_bytes;
    const std::size_t dst_bytes = (std::size_t)s.dst_rows * s.row_bytes * 8u;
    const std::size_t b_slot_bytes = (std::size_t)s.b_rows * s.row_bytes;

    for (int d = 0; d < g_world; ++d) {
        HC(hipSetDevice(d));
        Dev& v = g_dev[(std::size_t)d];
        v.c_src = (std::byte*)alloc_dev(src_bytes, unc);
        v.c_dst = (std::byte*)alloc_dev(dst_bytes, unc);
        v.b_slots = (std::byte*)alloc_dev(b_slot_bytes * 8u, unc);
        v.b_out = (std::byte*)alloc_dev(b_slot_bytes, unc);
        // GEMM operands are ordinary device tensors in the real kernel, not
        // symmetric-heap memory: allocate them cached regardless of --uncached.
        v.a_src = (uint4*)alloc_dev(s.a_bytes, false);
        v.proto_probe = (unsigned int*)alloc_dev(64 * sizeof(unsigned int), unc);
        v.proto_arrival = (unsigned int*)alloc_dev(64 * sizeof(unsigned int), unc);
        HC(hipMalloc(&v.sink, (std::size_t)kMaxCtas * kWaves * sizeof(float)));
        HC(hipMalloc(&v.ts,
                     (std::size_t)kMaxCtas * kTsStride * sizeof(std::uint64_t)));
        HC(hipMalloc(&v.scratch, 8 * sizeof(unsigned long long)));
        HC(hipStreamCreate(&v.stream));
        HC(hipEventCreate(&v.ev0));
        HC(hipEventCreate(&v.ev1));

        hipLaunchKernelGGL(fill_words_kernel, dim3(2048), dim3(256), 0, v.stream,
                           v.c_src, src_bytes / 8u, 0x1234ull + (unsigned)d);
        hipLaunchKernelGGL(fill_words_kernel, dim3(2048), dim3(256), 0, v.stream,
                           (std::byte*)v.a_src, s.a_bytes / 8u,
                           0x77ull + (unsigned)d);
        for (int sl = 0; sl < 8; ++sl) {
            // 2^-sl, so the 8-slot sum is exactly 1.9921875 in bf16
            hipLaunchKernelGGL(fill_bf16_slot_kernel, dim3(2048), dim3(256), 0,
                               v.stream, v.b_slots + (std::size_t)sl * b_slot_bytes,
                               b_slot_bytes / 2u, std::ldexp(1.0, -sl));
        }
        HC(hipMemsetD8Async(v.c_dst, 0xA5, dst_bytes, v.stream));
        HC(hipMemsetD32Async((hipDeviceptr_t)v.proto_probe, 0u, 64, v.stream));
        HC(hipMemsetD32Async((hipDeviceptr_t)v.proto_arrival, 0u, 64, v.stream));
        HC(hipStreamSynchronize(v.stream));
    }
    HC(hipSetDevice(0));

    const double per_dev =
        (double)(src_bytes + dst_bytes + b_slot_bytes * 9u + s.a_bytes) /
        (double)(1u << 30);
    std::printf("[e22] rev=%d world=%d heap=%s per-device alloc=%.2f GiB "
                "threads=%d lds=%d\n",
                E22_SRC_REV, g_world, unc ? "uncached" : "cached", per_dev,
                kThreads, kLdsBytes);
}

kittens::peer_bases<std::byte> dst_bases() {
    kittens::peer_bases<std::byte> b{};
    b.rank0 = (g_world > 0) ? g_dev[0].c_dst : nullptr;
    b.rank1 = (g_world > 1) ? g_dev[1].c_dst : nullptr;
    b.rank2 = (g_world > 2) ? g_dev[2].c_dst : nullptr;
    b.rank3 = (g_world > 3) ? g_dev[3].c_dst : nullptr;
    b.rank4 = (g_world > 4) ? g_dev[4].c_dst : nullptr;
    b.rank5 = (g_world > 5) ? g_dev[5].c_dst : nullptr;
    b.rank6 = (g_world > 6) ? g_dev[6].c_dst : nullptr;
    b.rank7 = (g_world > 7) ? g_dev[7].c_dst : nullptr;
    return b;
}

Params make_params(const Point& pt, int dev, int res_ctas,
                   unsigned long long res_rows, unsigned long long cmp_tiles,
                   int mfma_per_tile) {
    Params p{};
    const Sizes& s = g_cfg.sz;
    p.res_ctas = res_ctas;
    p.mode = (pt.mode == "xgmi") ? kModeXgmi
             : (pt.mode == "hbm") ? kModeHbm
                                  : kModeNone;
    p.res_rows = res_rows;
    p.cmp_tiles = cmp_tiles;
    p.mfma_per_tile = mfma_per_tile;
    p.c_src = g_dev[(std::size_t)dev].c_src;
    p.c_dst_local = g_dev[(std::size_t)dev].c_dst;
    p.c_dst_base = g_dev[(std::size_t)dev].c_dst;
    p.c_peers = dst_bases();
    p.me = dev;
    p.world = g_world;
    p.row_bytes = pt.row_bytes;
    p.src_rows_cap =
        (unsigned int)((std::size_t)s.src_rows * s.row_bytes / pt.row_bytes);
    p.dst_rows_cap =
        (unsigned int)((std::size_t)s.dst_rows * s.row_bytes / pt.row_bytes);
    p.fanout = (pt.fanout == "rr7") ? kFanRr : kFanSingle;
    p.protocol = pt.protocol;
    p.proto_g = pt.proto_g;
    p.flush_rows = 16;
    p.proto_probe = g_dev[(std::size_t)dev].proto_probe;
    p.proto_arrival = g_dev[(std::size_t)dev].proto_arrival;
    p.b_slots = g_dev[(std::size_t)dev].b_slots;
    p.b_slot_stride = (unsigned long long)s.b_rows * s.row_bytes;
    p.b_out = g_dev[(std::size_t)dev].b_out;
    p.b_rows_cap =
        (unsigned int)((std::size_t)s.b_rows * s.row_bytes / pt.row_bytes);
    p.a_src = g_dev[(std::size_t)dev].a_src;
    p.a_vec4_cap = (unsigned int)(s.a_bytes / 16u);
    p.sink = g_dev[(std::size_t)dev].sink;
    p.ts = g_dev[(std::size_t)dev].ts;
    return p;
}

void launch(const std::string& mode, int mlp, int dev, const Params& p, int grid) {
    Dev& v = g_dev[(std::size_t)dev];
    const int depth = (mode == "xgmi") ? mlp : 1;
    if (depth >= 8) {
        hipLaunchKernelGGL(sat_kernel_d8, dim3(grid), dim3(kThreads), 0, v.stream, p);
    } else if (depth >= 4) {
        hipLaunchKernelGGL(sat_kernel_d4, dim3(grid), dim3(kThreads), 0, v.stream, p);
    } else {
        hipLaunchKernelGGL(sat_kernel_d1, dim3(grid), dim3(kThreads), 0, v.stream, p);
    }
}

struct Spans {
    double res_us = 0.0, cmp_us = 0.0, overlap_us = 0.0;
    unsigned long long res_work = 0, cmp_work = 0;
    int res_n = 0, cmp_n = 0;
};

Spans read_spans(int dev, int grid) {
    std::vector<std::uint64_t> h((std::size_t)grid * kTsStride);
    HC(hipSetDevice(dev));
    HC(hipMemcpy(h.data(), g_dev[(std::size_t)dev].ts,
                 h.size() * sizeof(std::uint64_t), hipMemcpyDeviceToHost));
    std::uint64_t rb = ~0ull, re = 0, cb = ~0ull, ce = 0;
    Spans s;
    for (int i = 0; i < grid; ++i) {
        const std::uint64_t* r = h.data() + (std::size_t)i * kTsStride;
        if (r[kTsRole] == 1ull) {
            rb = std::min(rb, r[kTsBegin]);
            re = std::max(re, r[kTsEnd]);
            s.res_work += r[kTsWork];
            ++s.res_n;
        } else if (r[kTsRole] == 2ull) {
            cb = std::min(cb, r[kTsBegin]);
            ce = std::max(ce, r[kTsEnd]);
            s.cmp_work += r[kTsWork];
            ++s.cmp_n;
        }
    }
    const double tick = 1e6 / (double)kRealtimeHz;   // us per tick
    if (s.res_n) s.res_us = (double)(re - rb) * tick;
    if (s.cmp_n) s.cmp_us = (double)(ce - cb) * tick;
    if (s.res_n && s.cmp_n) {
        const std::uint64_t lo = std::max(rb, cb);
        const std::uint64_t hi = std::min(re, ce);
        s.overlap_us = (hi > lo) ? (double)(hi - lo) * tick : 0.0;
    }
    return s;
}

// ------------------------------------------------------------------- verify
bool verify_xgmi(const Point& pt, int dev) {
    // Sized so every source row is read exactly once and every destination row
    // is written exactly once -- only then is an order-independent XOR fold a
    // valid end-to-end check that the payload really landed on the peer.
    const int npeer = g_world - 1;
    const int ctas = 8;
    const Sizes& s = g_cfg.sz;
    const unsigned int dst_cap =
        (unsigned int)((std::size_t)s.dst_rows * s.row_bytes / pt.row_bytes);
    unsigned long long rows =
        (unsigned long long)ctas * pt.mlp * npeer * 8ull;
    while (rows > (unsigned long long)dst_cap) rows /= 2ull;

    const std::size_t part = (std::size_t)dst_cap * pt.row_bytes;
    for (int d = 0; d < g_world; ++d) {
        HC(hipSetDevice(d));
        HC(hipMemsetD8(g_dev[(std::size_t)d].c_dst + (std::size_t)dev * part,
                       0xA5, part));
    }
    HC(hipSetDevice(dev));
    const Params p = make_params(pt, dev, ctas, rows, 0ull, 16);
    launch(pt.mode, pt.mlp, dev, p, ctas);
    HC(hipStreamSynchronize(g_dev[(std::size_t)dev].stream));
    HC(hipGetLastError());

    const std::uint64_t poison = 0xA5A5A5A5A5A5A5A5ull;
    const unsigned long long words = rows * pt.row_bytes / 8ull;
    unsigned long long expect[2] = {0, 0};
    HC(hipMemsetD32((hipDeviceptr_t)g_dev[(std::size_t)dev].scratch, 0u, 4));
    hipLaunchKernelGGL(fold_kernel, dim3(512), dim3(256), 0,
                       g_dev[(std::size_t)dev].stream,
                       (const std::uint64_t*)g_dev[(std::size_t)dev].c_src, words,
                       poison, g_dev[(std::size_t)dev].scratch);
    HC(hipStreamSynchronize(g_dev[(std::size_t)dev].stream));
    HC(hipMemcpy(expect, g_dev[(std::size_t)dev].scratch, sizeof expect,
                 hipMemcpyDeviceToHost));

    unsigned long long got = 0, poisoned = 0;
    for (int d = 0; d < g_world; ++d) {
        if (pt.fanout == "rr7") {
            if (d == dev) continue;
        } else if (d != (dev + 1) % g_world) {
            continue;
        }
        const unsigned long long r_here =
            (pt.fanout == "rr7") ? rows / (unsigned long long)npeer : rows;
        HC(hipSetDevice(d));
        HC(hipMemsetD32((hipDeviceptr_t)g_dev[(std::size_t)d].scratch, 0u, 4));
        hipLaunchKernelGGL(fold_kernel, dim3(512), dim3(256), 0,
                           g_dev[(std::size_t)d].stream,
                           (const std::uint64_t*)(g_dev[(std::size_t)d].c_dst +
                                                  (std::size_t)dev * part),
                           r_here * pt.row_bytes / 8ull, poison,
                           g_dev[(std::size_t)d].scratch);
        HC(hipStreamSynchronize(g_dev[(std::size_t)d].stream));
        unsigned long long h[2] = {0, 0};
        HC(hipMemcpy(h, g_dev[(std::size_t)d].scratch, sizeof h,
                     hipMemcpyDeviceToHost));
        got ^= h[0];
        poisoned += h[1];
    }
    HC(hipSetDevice(dev));
    const bool ok = (got == expect[0]) && (poisoned == 0);
    std::printf("[e22][verify] xgmi mlp=%d fanout=%s rb=%u rows=%llu fold=%s "
                "poison_words=%llu -> %s\n",
                pt.mlp, pt.fanout.c_str(), pt.row_bytes, rows,
                (got == expect[0]) ? "MATCH" : "MISMATCH", poisoned,
                ok ? "PASS" : "FAIL");
    return ok;
}

bool verify_hbm(const Point& pt, int dev) {
    HC(hipSetDevice(dev));
    const int ctas = 8;
    const unsigned long long rows = 1024;
    HC(hipMemsetD8(g_dev[(std::size_t)dev].b_out, 0x00,
                   (std::size_t)rows * pt.row_bytes));
    const Params p = make_params(pt, dev, ctas, rows, 0ull, 16);
    launch(pt.mode, pt.mlp, dev, p, ctas);
    HC(hipStreamSynchronize(g_dev[(std::size_t)dev].stream));
    HC(hipGetLastError());
    HC(hipMemsetD32((hipDeviceptr_t)g_dev[(std::size_t)dev].scratch, 0u, 4));
    hipLaunchKernelGGL(check_bf16_kernel, dim3(512), dim3(256), 0,
                       g_dev[(std::size_t)dev].stream,
                       g_dev[(std::size_t)dev].b_out, rows * pt.row_bytes / 2ull,
                       1.9921875f, g_dev[(std::size_t)dev].scratch);
    HC(hipStreamSynchronize(g_dev[(std::size_t)dev].stream));
    unsigned long long bad = 0;
    HC(hipMemcpy(&bad, g_dev[(std::size_t)dev].scratch, sizeof bad,
                 hipMemcpyDeviceToHost));
    std::printf("[e22][verify] hbm rows=%llu wrong_elems=%llu -> %s\n", rows, bad,
                bad == 0 ? "PASS" : "FAIL");
    return bad == 0;
}

// ------------------------------------------------------------------- sizing
unsigned long long xgmi_rows_for(const Point& pt) {
    // per_cta: constant volume PER CTA, so every launch lands in the same
    // 10-50 ms window at every CTA count. A rate is a ratio, so equal TOTAL
    // work across the axis is not required; equal steady-state residency is,
    // and holding the total fixed would put a 3 s launch at C=1 next to a 5 ms
    // launch at C=64. `fixed` keeps design.md's constant global list as a
    // cross-check at three CTA counts.
    // rr7 prices seven links instead of one, so it gets seven times the volume
    // to land in the same time window as the single-link arm.
    const int depth = std::max(1, pt.mlp);
    const double fan_scale =
        (pt.fanout == "rr7") ? (double)std::max(1, g_world - 1) : 1.0;
    const unsigned long long per_cta = std::max(
        (unsigned long long)depth,
        (unsigned long long)(g_cfg.xgmi_mb_per_cta * fan_scale * 1048576.0) /
            pt.row_bytes);
    unsigned long long rows = (pt.work_mode == "fixed")
                                  ? 292000ull * 14336ull / pt.row_bytes
                                  : per_cta * (unsigned long long)pt.ctas;
    const unsigned long long grain = (unsigned long long)pt.ctas *
                                     (unsigned long long)depth *
                                     (unsigned long long)std::max(1, g_world - 1);
    rows = (rows / grain) * grain;
    return rows < grain ? grain : rows;
}

unsigned long long hbm_rows_for(const Point& pt) {
    const unsigned long long per_cta =
        (unsigned long long)(g_cfg.hbm_mb_per_cta * 1048576.0) / pt.row_bytes;
    unsigned long long rows = (pt.work_mode == "fixed")
                                  ? 131072ull
                                  : per_cta * (unsigned long long)pt.ctas;
    const unsigned long long grain = (unsigned long long)pt.ctas * kWaves;
    rows = (rows / grain) * grain;
    return rows < grain ? grain : rows;
}

// Measured tiles-per-microsecond PER CTA for a given (mfma_per_tile, ctas).
// One short probe launch, cached. This replaces any assumption about how fast
// the compute role is, which is what sizes the overlay so it outlasts the
// resource role.
double tile_rate(int shape, int ctas) {
    const std::pair<int, int> k(shape, ctas);
    auto it = g_tile_rate.find(k);
    if (it != g_tile_rate.end()) return it->second;
    const int probe_tiles = 1500;
    Point pt;
    pt.mode = "mfma";
    pt.ctas = ctas;
    pt.mlp = shape;
    const Params p = make_params(pt, 0, 0, 0ull, (unsigned long long)probe_tiles,
                                 shape);
    HC(hipSetDevice(0));
    for (int i = 0; i < 2; ++i) {
        launch("mfma", 0, 0, p, ctas);
        HC(hipStreamSynchronize(g_dev[0].stream));
    }
    HC(hipEventRecord(g_dev[0].ev0, g_dev[0].stream));
    launch("mfma", 0, 0, p, ctas);
    HC(hipEventRecord(g_dev[0].ev1, g_dev[0].stream));
    HC(hipEventSynchronize(g_dev[0].ev1));
    HC(hipGetLastError());
    float ms = 0.0f;
    HC(hipEventElapsedTime(&ms, g_dev[0].ev0, g_dev[0].ev1));
    const double rate = (double)probe_tiles / std::max(1.0, (double)ms * 1000.0);
    g_tile_rate[k] = rate;
    return rate;
}

// -------------------------------------------------------------- point runner
struct Result {
    std::string metric;
    std::string value_source;
    double value = 0.0;
    std::vector<double> wall_us;
    std::vector<double> metric_wall;
    std::vector<double> metric_span;
    std::vector<double> res_span_us;
    std::vector<double> cmp_span_us;
    std::vector<double> overlap_pct;
    std::vector<double> cmp_tflops;
    double peak_fraction = 0.0;
    double bytes = 0.0;
    double flops = 0.0;
    unsigned long long res_rows = 0;
    unsigned long long cmp_tiles = 0;
    int mfma_per_tile = 0;
    int retries = 0;
    int verified = -1;
};

Result run_point(const Point& pt) {
    Result r;
    const int npush = (pt.pushers >= g_world) ? g_world : 1;
    const bool res_active = (pt.mode != "mfma") && (pt.conc != "reserved_only");
    const int res_ctas = (pt.mode == "mfma") ? 0 : pt.ctas;
    const int grid = (pt.conc == "isolated") ? std::max(1, pt.ctas) : g_cfg.grid;
    const int cmp_ctas = grid - res_ctas;
    const int shape = (pt.mode == "mfma") ? std::max(4, pt.mlp) : 16;

    unsigned long long res_rows = 0;
    if (res_active) {
        res_rows = (pt.mode == "xgmi") ? xgmi_rows_for(pt) : hbm_rows_for(pt);
    }

    unsigned long long cmp_tiles = 0;
    if (pt.mode == "mfma") {
        cmp_tiles = (unsigned long long)std::max(
            1.0, g_cfg.target_launch_us * tile_rate(shape, grid));
    } else if (pt.conc != "isolated") {
        auto paired = g_paired_tiles.find(pt.pair_key());
        if (paired != g_paired_tiles.end()) {
            cmp_tiles = paired->second;   // reserved_only reuses the concurrent
                                          // arm's compute size, exactly
        } else {
            auto iso = g_isolated_us.find(pt.pair_key());
            const double res_est = (iso != g_isolated_us.end())
                                       ? iso->second
                                       : g_cfg.target_launch_us;
            cmp_tiles = (unsigned long long)std::max(
                1.0, res_est * g_cfg.concurrent_slack * tile_rate(shape, cmp_ctas));
        }
    }

    for (int attempt = 0;; ++attempt) {
        r.wall_us.clear();
        r.metric_span.clear();
        r.res_span_us.clear();
        r.cmp_span_us.clear();
        r.overlap_pct.clear();
        r.cmp_tflops.clear();
        std::vector<Params> ps((std::size_t)npush);
        for (int d = 0; d < npush; ++d) {
            ps[(std::size_t)d] =
                make_params(pt, d, res_ctas, res_rows, cmp_tiles, shape);
        }
        for (int it = 0; it < g_cfg.warmup + g_cfg.rotations; ++it) {
            for (int d = 0; d < npush; ++d) {
                HC(hipSetDevice(d));
                HC(hipEventRecord(g_dev[(std::size_t)d].ev0,
                                  g_dev[(std::size_t)d].stream));
                launch(pt.mode, pt.mlp, d, ps[(std::size_t)d], grid);
                HC(hipEventRecord(g_dev[(std::size_t)d].ev1,
                                  g_dev[(std::size_t)d].stream));
            }
            double worst = 0.0;
            for (int d = 0; d < npush; ++d) {
                HC(hipSetDevice(d));
                HC(hipEventSynchronize(g_dev[(std::size_t)d].ev1));
                float ms = 0.0f;
                HC(hipEventElapsedTime(&ms, g_dev[(std::size_t)d].ev0,
                                       g_dev[(std::size_t)d].ev1));
                worst = std::max(worst, (double)ms * 1000.0);
            }
            HC(hipGetLastError());
            if (it < g_cfg.warmup) continue;
            const Spans sp = read_spans(0, grid);
            r.wall_us.push_back(worst);
            r.res_span_us.push_back(sp.res_us);
            r.cmp_span_us.push_back(sp.cmp_us);
            const double denom = (pt.mode == "mfma") ? sp.cmp_us : sp.res_us;
            r.overlap_pct.push_back(denom > 0.0 ? 100.0 * sp.overlap_us / denom
                                                : 0.0);
            // cmp_work is the MFMA count reported by ONE wave leader per CTA
            // (only threadIdx.x == 0 writes the stamp block) and all kWaves
            // waves of a CTA issue the identical count, so the CTA total is
            // kWaves x that. Dropping this factor is a silent 4x understatement
            // of TFLOPS; the roofline check in result.md is what catches it.
            if (sp.cmp_work) {
                r.cmp_tflops.push_back((double)sp.cmp_work * kFlopsPerMfma *
                                       (double)kWaves /
                                       (std::max(1.0, sp.cmp_us) * 1e6));
            }
            if (pt.mode == "mfma") {
                r.flops = (double)sp.cmp_work * kFlopsPerMfma * (double)kWaves;
            }
        }
        r.retries = attempt;
        const double ovl = median(r.overlap_pct);
        if (pt.conc != "concurrent" || ovl >= 100.0 * g_cfg.min_overlap ||
            attempt >= g_cfg.max_retries) {
            break;
        }
        cmp_tiles = (unsigned long long)((double)cmp_tiles * 1.6);
    }

    r.res_rows = res_rows;
    r.cmp_tiles = cmp_tiles;
    r.mfma_per_tile = shape;

    // ---- metric arithmetic. Hand-checked against a roofline in result.md.
    if (pt.mode == "xgmi" || pt.mode == "hbm") {
        r.metric = "GBps";
        const double per_row = (pt.mode == "hbm") ? 9.0 : 1.0;   // 8 reads + 1
        r.bytes = (double)res_rows * pt.row_bytes * per_row * (double)npush;
        for (double us : r.wall_us) r.metric_wall.push_back(r.bytes / (us * 1e3));
        for (double us : r.res_span_us) {
            r.metric_span.push_back(us > 0.0 ? r.bytes / (us * 1e3) : 0.0);
        }
        const bool span_based = (pt.conc == "concurrent");
        r.value = span_based ? median(r.metric_span) : median(r.metric_wall);
        r.value_source = span_based ? "role_span" : "wall";
        const double peak = (pt.mode == "hbm")
                                ? g_cfg.peak_hbm
                                : ((pt.fanout == "rr7") ? g_cfg.peak_xgmi_agg
                                                        : g_cfg.peak_xgmi_link);
        r.peak_fraction = r.value / (peak * (double)npush);
    } else {
        r.metric = "TFLOPS";
        for (double us : r.wall_us) {
            r.metric_wall.push_back(r.flops / (us * 1e6));
        }
        for (double us : r.cmp_span_us) {
            r.metric_span.push_back(us > 0.0 ? r.flops / (us * 1e6) : 0.0);
        }
        r.value = median(r.metric_wall);
        r.value_source = "wall";
        r.bytes = (double)cmp_tiles * kStageBytesPerTile * (double)cmp_ctas;
        r.peak_fraction = r.value / g_cfg.peak_tflops;
    }

    if (pt.conc == "isolated" && pt.mode != "mfma") {
        g_isolated_us[pt.pair_key()] = median(r.wall_us);
    }
    if (pt.conc == "concurrent") g_paired_tiles[pt.pair_key()] = cmp_tiles;
    return r;
}

// ------------------------------------------------------------------- output
void emit_series(FILE* f, const char* name, const std::vector<double>& v,
                 const char* fmt) {
    std::fprintf(f, ",\"%s\":[", name);
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::fprintf(f, "%s", i ? "," : "");
        std::fprintf(f, fmt, v[i]);
    }
    std::fprintf(f, "]");
}

void emit(FILE* f, const Point& pt, const Result& r) {
    const std::vector<double>& vm =
        (r.value_source == "role_span") ? r.metric_span : r.metric_wall;
    const double med = median(vm);
    std::fprintf(
        f,
        "{\"key\":\"%s\",\"pair_key\":\"%s\",\"mode\":\"%s\",\"ctas\":%d,"
        "\"mlp\":%d,\"fanout\":\"%s\",\"concurrency\":\"%s\",\"metric\":\"%s\","
        "\"value\":%.4f,\"value_source\":\"%s\",\"repeats\":%d,\"median\":%.4f,"
        "\"spread\":{\"p25\":%.4f,\"p75\":%.4f,\"min\":%.4f,\"max\":%.4f,"
        "\"sd\":%.4f,\"rel_iqr_pct\":%.3f},\"peak_fraction\":%.5f",
        pt.key().c_str(), pt.pair_key().c_str(), pt.mode.c_str(), pt.ctas, pt.mlp,
        pt.fanout.c_str(), pt.conc.c_str(), r.metric.c_str(), r.value,
        r.value_source.c_str(), (int)vm.size(), med, quantile(vm, 0.25),
        quantile(vm, 0.75), vm.empty() ? 0.0 : *std::min_element(vm.begin(), vm.end()),
        vm.empty() ? 0.0 : *std::max_element(vm.begin(), vm.end()), stdev(vm),
        med > 0.0 ? 100.0 * (quantile(vm, 0.75) - quantile(vm, 0.25)) / med : 0.0,
        r.peak_fraction);
    emit_series(f, "values", vm, "%.4f");
    emit_series(f, "values_wall", r.metric_wall, "%.4f");
    emit_series(f, "values_span", r.metric_span, "%.4f");
    emit_series(f, "wall_us", r.wall_us, "%.2f");
    emit_series(f, "res_span_us", r.res_span_us, "%.2f");
    emit_series(f, "cmp_span_us", r.cmp_span_us, "%.2f");
    emit_series(f, "overlap_pct", r.overlap_pct, "%.2f");
    emit_series(f, "cmp_tflops", r.cmp_tflops, "%.3f");
    std::fprintf(
        f,
        ",\"wall_us_median\":%.2f,\"res_span_us_median\":%.2f,"
        "\"cmp_span_us_median\":%.2f,\"overlap_pct_median\":%.2f,"
        "\"cmp_tflops_median\":%.3f,\"retries\":%d,\"protocol\":%d,"
        "\"proto_g\":%d,\"pushers\":%d,\"row_bytes\":%u,\"work_mode\":\"%s\","
        "\"heap\":\"%s\",\"res_rows\":%llu,\"cmp_tiles\":%llu,"
        "\"mfma_per_tile\":%d,\"bytes\":%.0f,\"flops\":%.0f,\"grid\":%d,"
        "\"threads\":%d,\"lds_bytes\":%d,\"rotations\":%d,\"warmup\":%d,"
        "\"src_rev\":%d,\"verified\":%d,\"status\":\"%s\"}\n",
        median(r.wall_us), median(r.res_span_us), median(r.cmp_span_us),
        median(r.overlap_pct), median(r.cmp_tflops), r.retries, pt.protocol,
        pt.proto_g, pt.pushers, pt.row_bytes, pt.work_mode.c_str(),
        pt.heap_uncached ? "uncached" : "cached", r.res_rows, r.cmp_tiles,
        r.mfma_per_tile, r.bytes, r.flops,
        (pt.conc == "isolated") ? std::max(1, pt.ctas) : g_cfg.grid, kThreads,
        kLdsBytes, g_cfg.rotations, g_cfg.warmup, E22_SRC_REV, r.verified,
        g_cfg.dry_run ? "dry_run" : "valid_diagnostic");
    std::fflush(f);
}

// ------------------------------------------------------------------ dry run
// Emits one synthetic record per grid point WITHOUT touching a device, purely
// so the real emit() format can be validated against summarize.py on CPU. The
// model is a saturating curve for the bandwidth modes and a linear one for
// MFMA; the numbers are meaningless and every dry record is stamped
// status="dry_run" so it can never be mistaken for data.
Result synthetic_result(const Point& pt) {
    Result r;
    const double c = (double)std::max(1, pt.ctas);
    const bool con = (pt.conc == "concurrent");
    double v;
    if (pt.mode == "mfma") {
        r.metric = "TFLOPS";
        v = 7.5 * c;
    } else if (pt.mode == "hbm") {
        r.metric = "GBps";
        v = 6000.0 * c / (c + 24.0);
    } else {
        r.metric = "GBps";
        const double ceil_ =
            (pt.fanout == "rr7") ? g_cfg.peak_xgmi_agg : g_cfg.peak_xgmi_link;
        v = ceil_ * c / (c + 3.0);
        if (pt.protocol) v *= 0.55;
    }
    if (con) v *= 0.97;
    if (pt.conc == "reserved_only") v = 0.0;
    const double t_us = 25000.0;
    for (int i = 0; i < g_cfg.rotations; ++i) {
        const double jitter = 1.0 + 0.004 * (double)(i - 2);
        r.wall_us.push_back(t_us / jitter);
        r.res_span_us.push_back(t_us * 0.98 / jitter);
        r.cmp_span_us.push_back(con ? t_us * 1.3 : 0.0);
        r.overlap_pct.push_back(con ? 97.5 : 0.0);
        r.metric_wall.push_back(v * jitter);
        r.metric_span.push_back(v * jitter * 1.02);
        if (pt.conc != "isolated") r.cmp_tflops.push_back(1500.0 * jitter);
    }
    r.value_source = con ? "role_span" : "wall";
    r.value = con ? median(r.metric_span) : median(r.metric_wall);
    const double peak = (pt.mode == "mfma") ? g_cfg.peak_tflops
                        : (pt.mode == "hbm")
                            ? g_cfg.peak_hbm
                            : ((pt.fanout == "rr7") ? g_cfg.peak_xgmi_agg
                                                    : g_cfg.peak_xgmi_link);
    r.peak_fraction = r.value / peak;
    r.res_rows = 1000;
    r.cmp_tiles = con ? 5000 : 0;
    r.mfma_per_tile = (pt.mode == "mfma") ? std::max(4, pt.mlp) : 16;
    r.bytes = 1e9;
    r.flops = (pt.mode == "mfma") ? 1e13 : 0.0;
    r.verified = 0;
    return r;
}

// --------------------------------------------------------------- grid tiers
std::vector<Point> build_grid(const std::string& tier) {
    std::vector<Point> g;
    const std::vector<int> a_ctas = {32, 64, 96, 128, 160, 192, 224, 256};
    const std::vector<int> b_ctas = {8, 16, 32, 64, 128, 256};
    const std::vector<int> c_ctas = {1, 2, 4, 8, 16, 32, 64};
    const std::vector<int> mlps = {1, 4, 8};
    const std::vector<std::string> fans = {"single", "rr7"};

    auto mfma = [&](int ctas, int shape) {
        Point p;
        p.mode = "mfma";
        p.ctas = ctas;
        p.mlp = shape;
        p.heap_uncached = g_cfg.uncached;
        g.push_back(p);
    };
    auto res = [&](const std::string& mode, int ctas, int mlp,
                   const std::string& fan, const std::string& conc, int protocol,
                   int proto_g, int pushers = 1, unsigned int rb = 14336,
                   const std::string& wm = "per_cta") {
        Point p;
        p.mode = mode;
        p.ctas = ctas;
        p.mlp = mlp;
        p.fanout = fan;
        p.conc = conc;
        p.protocol = protocol;
        p.proto_g = proto_g;
        p.pushers = pushers;
        p.row_bytes = rb;
        p.work_mode = wm;
        p.heap_uncached = g_cfg.uncached;
        g.push_back(p);
    };

    if (tier == "quick") {
        mfma(256, 32);
        res("hbm", 64, 0, "na", "isolated", 0, 0);
        res("hbm", 64, 0, "na", "concurrent", 0, 0);
        res("hbm", 64, 0, "na", "reserved_only", 0, 0);
        res("xgmi", 8, 4, "single", "isolated", 0, 0);
        res("xgmi", 8, 4, "rr7", "isolated", 0, 0);
        res("xgmi", 8, 4, "rr7", "concurrent", 0, 0);
        res("xgmi", 8, 4, "rr7", "reserved_only", 0, 0);
        res("xgmi", 8, 4, "rr7", "isolated", 1, 16);
        return g;
    }

    // panel a -- MFMA capacity. shape 4 is M7-like intensity (~30 flop/byte),
    // shape 256 is peak-seeking; H4 is a claim about the CTA axis, so the whole
    // axis is swept at both intensities. Shape 16 is the shape the overlay's
    // compute role runs, so this series is the compute-only denominator the
    // reserved_only arm is measured against (a=192 pairs exactly with the C=64
    // reservation, a=224 with C=32, and so on).
    for (int shape : {4, 16, 256})
        for (int c : a_ctas) mfma(c, shape);
    // panel b -- local HBM reduce
    for (int c : b_ctas) res("hbm", c, 0, "na", "isolated", 0, 0);
    for (int c : b_ctas) {
        res("hbm", c, 0, "na", "concurrent", 0, 0);
        res("hbm", c, 0, "na", "reserved_only", 0, 0);
    }
    // panel c -- xGMI push, isolated then the two overlay arms
    for (const std::string& f : fans)
        for (int m : mlps)
            for (int c : c_ctas) res("xgmi", c, m, f, "isolated", 0, 0);
    for (const std::string& f : fans)
        for (int m : mlps)
            for (int c : c_ctas) {
                res("xgmi", c, m, f, "concurrent", 0, 0);
                res("xgmi", c, m, f, "reserved_only", 0, 0);
            }
    // H3 -- protocol vs payload inside one binary, at MLP 4
    for (const std::string& f : fans)
        for (int c : c_ctas)
            for (int gg : {2, 16}) {
                res("xgmi", c, 4, f, "isolated", 1, gg);
                res("xgmi", c, 4, f, "concurrent", 1, gg);
                res("xgmi", c, 4, f, "reserved_only", 1, gg);
            }

    if (tier == "full") {
        // granularity: the production push unit is a 896 B slice, not a row
        for (const std::string& f : fans)
            for (int m : mlps)
                for (int c : c_ctas)
                    res("xgmi", c, m, f, "isolated", 0, 0, 1, 896);
        // every rank pushes at once: the fabric condition the combine sees
        for (int c : c_ctas) {
            res("xgmi", c, 4, "rr7", "isolated", 0, 0, 8);
            res("xgmi", c, 4, "rr7", "concurrent", 0, 0, 8);
        }
        // design.md's fixed global work list, as a work_mode cross-check
        for (int c : {4, 16, 64})
            res("xgmi", c, 4, "rr7", "isolated", 0, 0, 1, 14336, "fixed");
    }
    return g;
}

void load_existing(const std::string& path, std::set<std::string>* keys) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return;
    std::vector<char> line(1u << 16);
    while (std::fgets(line.data(), (int)line.size(), f)) {
        const char* k = std::strstr(line.data(), "\"key\":\"");
        if (!k) continue;
        k += 7;
        const char* e = std::strchr(k, '"');
        if (!e) continue;
        keys->insert(std::string(k, (std::size_t)(e - k)));
        // Recover the paired isolated wall time so a resumed run still sizes
        // the overlay from measurement instead of the fallback.
        const char* pk = std::strstr(line.data(), "\"pair_key\":\"");
        const char* wm = std::strstr(line.data(), "\"wall_us_median\":");
        const char* cc = std::strstr(line.data(), "\"concurrency\":\"isolated\"");
        const char* ct = std::strstr(line.data(), "\"cmp_tiles\":");
        const char* cn = std::strstr(line.data(), "\"concurrency\":\"concurrent\"");
        if (pk && wm && cc) {
            pk += 12;
            const char* pe = std::strchr(pk, '"');
            if (pe) {
                g_isolated_us[std::string(pk, (std::size_t)(pe - pk))] =
                    std::atof(wm + 17);
            }
        }
        if (pk && ct && cn) {
            const char* p2 = std::strstr(line.data(), "\"pair_key\":\"") + 12;
            const char* pe = std::strchr(p2, '"');
            if (pe) {
                g_paired_tiles[std::string(p2, (std::size_t)(pe - p2))] =
                    std::strtoull(ct + 12, nullptr, 10);
            }
        }
    }
    std::fclose(f);
}

const char* need(int argc, char** argv, int i) {
    if (i >= argc) {
        std::fprintf(stderr, "[e22] missing value for %s\n", argv[i - 1]);
        std::exit(2);
    }
    return argv[i];
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out") g_cfg.out = need(argc, argv, ++i);
        else if (a == "--tier") g_cfg.tier = need(argc, argv, ++i);
        else if (a == "--rotations") g_cfg.rotations = std::atoi(need(argc, argv, ++i));
        else if (a == "--warmup") g_cfg.warmup = std::atoi(need(argc, argv, ++i));
        else if (a == "--devices") g_cfg.sz.world = std::atoi(need(argc, argv, ++i));
        else if (a == "--src-rows") g_cfg.sz.src_rows = (unsigned)std::atoi(need(argc, argv, ++i));
        else if (a == "--dst-rows") g_cfg.sz.dst_rows = (unsigned)std::atoi(need(argc, argv, ++i));
        else if (a == "--b-rows") g_cfg.sz.b_rows = (unsigned)std::atoi(need(argc, argv, ++i));
        else if (a == "--target-us") g_cfg.target_launch_us = std::atof(need(argc, argv, ++i));
        else if (a == "--peak-tflops") g_cfg.peak_tflops = std::atof(need(argc, argv, ++i));
        else if (a == "--peak-hbm") g_cfg.peak_hbm = std::atof(need(argc, argv, ++i));
        else if (a == "--peak-xgmi-link") g_cfg.peak_xgmi_link = std::atof(need(argc, argv, ++i));
        else if (a == "--peak-xgmi-agg") g_cfg.peak_xgmi_agg = std::atof(need(argc, argv, ++i));
        else if (a == "--max-seconds") g_cfg.max_seconds = std::atof(need(argc, argv, ++i));
        else if (a == "--xgmi-mb-per-cta") g_cfg.xgmi_mb_per_cta = std::atof(need(argc, argv, ++i));
        else if (a == "--hbm-mb-per-cta") g_cfg.hbm_mb_per_cta = std::atof(need(argc, argv, ++i));
        else if (a == "--cached") g_cfg.uncached = 0;
        else if (a == "--uncached") g_cfg.uncached = 1;
        else if (a == "--no-verify") g_cfg.verify = 0;
        else if (a == "--list") g_cfg.list_only = 1;
        else if (a == "--dry-run") g_cfg.dry_run = 1;
        else {
            std::fprintf(stderr, "[e22] unknown arg %s\n", a.c_str());
            return 2;
        }
    }

    const std::vector<Point> grid = build_grid(g_cfg.tier);
    if (g_cfg.list_only) {
        for (const Point& p : grid) std::printf("%s\n", p.key().c_str());
        std::printf("[e22] tier=%s points=%zu\n", g_cfg.tier.c_str(), grid.size());
        return 0;
    }

    if (!g_cfg.dry_run) setup_devices();
    std::set<std::string> done;
    load_existing(g_cfg.out, &done);
    FILE* out = std::fopen(g_cfg.out.c_str(), "a");
    if (!out) {
        std::fprintf(stderr, "[e22] cannot append to %s\n", g_cfg.out.c_str());
        return 2;
    }
    std::printf("[e22] tier=%s points=%zu already_done=%zu out=%s "
                "rotations=%d warmup=%d\n",
                g_cfg.tier.c_str(), grid.size(), done.size(), g_cfg.out.c_str(),
                g_cfg.rotations, g_cfg.warmup);

    std::set<std::string> verified;
    const double t0 = now_s();
    int n_run = 0, n_fail = 0;
    for (const Point& pt : grid) {
        if (done.count(pt.key())) continue;
        if (g_cfg.max_seconds > 0.0 && now_s() - t0 > g_cfg.max_seconds) {
            std::printf("[e22] time budget reached after %d point(s); rerun the "
                        "same command to resume\n", n_run);
            break;
        }
        if (g_cfg.dry_run) {
            emit(out, pt, synthetic_result(pt));
            ++n_run;
            continue;
        }
        int verified_flag = -1;
        if (g_cfg.verify && pt.mode != "mfma") {
            char vk[160];
            std::snprintf(vk, sizeof vk, "%s|%d|%s|%u|%d", pt.mode.c_str(), pt.mlp,
                          pt.fanout.c_str(), pt.row_bytes, pt.protocol);
            if (!verified.count(vk)) {
                const bool ok =
                    (pt.mode == "xgmi") ? verify_xgmi(pt, 0) : verify_hbm(pt, 0);
                if (!ok) {
                    std::fprintf(stderr,
                                 "[e22] VERIFY FAILED for %s -- refusing to "
                                 "report a bandwidth for it\n", vk);
                    ++n_fail;
                    continue;
                }
                verified.insert(vk);
            }
            verified_flag = 1;
        }
        Result got = run_point(pt);
        got.verified = verified_flag;
        emit(out, pt, got);
        ++n_run;
        std::printf("[e22] %-52s %9.2f %-6s peak=%5.1f%% wall=%8.1fus "
                    "span=%8.1fus ovl=%5.1f%% cmp=%7.1fTF\n",
                    pt.key().c_str(), got.value, got.metric.c_str(),
                    100.0 * got.peak_fraction, median(got.wall_us),
                    median(got.res_span_us), median(got.overlap_pct),
                    median(got.cmp_tflops));
    }
    std::fclose(out);
    std::printf("[e22] done: ran %d point(s), %d verify failure(s), %.1f s\n",
                n_run, n_fail, now_s() - t0);
    return n_fail ? 1 : 0;
}

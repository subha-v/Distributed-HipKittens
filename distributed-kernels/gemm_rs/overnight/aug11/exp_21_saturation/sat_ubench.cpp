// ================================================================================================
// sat_ubench — exp_21 resource-saturation microbenchmark for GEMM-RS on 8x MI300X (gfx942).
//
// Paper Fig 2 / Q4(a); a NanoFlow-Fig-7 analog. Three resource curves against CTA count, each
// isolated and concurrent, with a C-matched reserve-only control:
//
//   mode a (MFMA)  this kernel's double-buffered mainloop over a fixed synthetic tile list
//   mode b (HBM)   the reducer body at REDV=1, m3::pull_sum_bf16_strip_mlp8, called unmodified
//   mode c (xGMI)  the real 16 B peer-packet emit: emit_band_packets' packet loop calling
//                  hk_gemm_rs::store_peer_packet16 into kittens::distributed::translate_peer<8>
//
// This is a DIAGNOSTIC module. It is standalone by construction: it shares no object with the
// production binding, exports its own pybind surface, writes into its own build directory, and
// can never enter a graded build. It edits NO production file -- the two adapter headers and the
// constants header are included read-only and their helpers are called, not copied, wherever a
// call is possible.
//
// Why a new module: there is no grid override anywhere in this codebase. The production grid is
// hardcoded dim3(m3::CU_COUNT) (gemm_rs_mi300x.cpp:140) and only the host scalar num_gemm_ctas
// varies, which moves the producer/reducer boundary inside a fixed 304-CTA grid. A saturation
// curve needs the grid itself as the x-axis. See design.md section 0.
//
// Plan, controls and pre-registered predictions: plan.md. Task graph, buffers, traffic model and
// rejected alternatives: design.md.
// ================================================================================================

#include "kittens.cuh"

#include "gemm_rs_mi300x_hk_adapter.cuh"     // read-only: the shipped emit / reduce bodies

#include <hip/hip_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace m3 = hk_gemm_rs_mi300x;
using namespace kittens;

using bf16_t = kittens::bf16;
using _gl_A  = gl<bf16_t, -1, -1, -1, -1>;    // [1, 1, M, K]
using _gl_B  = gl<bf16_t, -1, -1, -1, -1>;    // [1, 1, N, K]

// ------------------------------------------------------------------------------------------------
// Geometry. These mirror the production shape-6 config row (BM=BN=256, BK=32) because that row is
// where the attribution mass sits, and its LDS footprint 2*(BM+BN)*BK*2 == 65536 is the whole
// gfx942 per-workgroup budget -- which is what makes "1 CTA/CU, no oversubscription regime" a
// structural fact of this ubench rather than an assertion (plan.md H4).
// ------------------------------------------------------------------------------------------------
constexpr int SAT_BM       = 256;
constexpr int SAT_BN       = 256;
constexpr int SAT_BK       = 32;
constexpr int SAT_WIN_ROWS = 32;                    // production win_rows = min(EB, 32)
constexpr int SAT_THREADS  = m3::CTA_THREADS;       // 512
constexpr int SAT_WARPS_M  = m3::WARPS_M;           // 2
constexpr int SAT_WARPS_N  = m3::WARPS_N;           // 4
constexpr int SAT_GRID_MAX = m3::CU_COUNT;          // 304
constexpr int SAT_WORLD    = m3::WORLD_SIZE;        // 8
constexpr int SAT_LDS      = 2 * (SAT_BM + SAT_BN) * SAT_BK * (int)sizeof(bf16_t);
static_assert(SAT_LDS == 65536);
static_assert(SAT_LDS <= kittens::MAX_SHARED_MEMORY);
// Emit-window packet geometry, identical to emit_band_packets at cols = BN.
constexpr int SAT_ELEMS_PER_PACKET = 8;                                   // 16 B of bf16
constexpr int SAT_PACKETS_PER_ROW  = SAT_BN / SAT_ELEMS_PER_PACKET;       // 32
constexpr int SAT_PACKETS_PER_WIN  = SAT_WIN_ROWS * SAT_PACKETS_PER_ROW;  // 1024
constexpr int SAT_WIN_BYTES        = SAT_PACKETS_PER_WIN * 16;            // 16384

// ------------------------------------------------------------------------------------------------
// Argument POD. gl<> has a host-only constructor, so this stays an aggregate: the caller writes
// `sat_args g{ga, gb};` (remaining members value-initialised) and then fills fields by name.
// ------------------------------------------------------------------------------------------------
struct sat_args {
    _gl_A a;
    _gl_B b;

    // ---- mode a -------------------------------------------------------------------------------
    int k_iters;
    int a_tiles_m;
    int a_tiles_n;
    int a_work;                 // fixed tile-list length (isolated); modulus (filler)

    // ---- mode b -------------------------------------------------------------------------------
    hk_gemm_rs::bf16_sources8 red_srcs;
    std::uint16_t* red_dst;
    int red_row_stride;         // elements (N)
    int red_valid_cols;         // elements (N)
    int b_strip_rows;
    int b_cols_per_row;         // N / BN
    int b_row_bands;            // rows per slot / b_strip_rows
    int b_work;

    // ---- mode c -------------------------------------------------------------------------------
    std::uint16_t* c_local_base;                    // == the symmetric allocation base
    hk_gemm_rs::symmetric_descriptor heap_desc;     // 72 B, by value
    unsigned long long c_slot_elems;                // slot_rows * N
    int c_row_stride;                               // elements (N)
    int c_cols_per_row;                             // N / BN
    int c_work;                                     // windows in the work list
    int depth;                                      // in-flight remote-op bound; 0 = unbounded
    int fanout;                                     // 0 = single peer, 1 = round-robin-7

    // ---- protocol arm -------------------------------------------------------------------------
    int protocol;
    int release_group;
    std::uint32_t* sig_local;
    hk_gemm_rs::symmetric_descriptor sig_desc;

    // ---- roles / instrumentation --------------------------------------------------------------
    int me;
    int num_res_ctas;                   // C
    int num_gemm_ctas;                  // 304 - C, concurrent arms only
    unsigned long long deadline_ticks;  // reserve control: idle-spin length
    unsigned long long* stamps;         // [304][2]
    unsigned int* work_done;            // [304]
    unsigned int* res_done;             // [1]
};

// ------------------------------------------------------------------------------------------------
// The payload pattern. A pure function of the window-local coordinates, so the destination-side
// verify can regenerate it without any in-kernel accumulation in the timed path (design.md 7).
// ------------------------------------------------------------------------------------------------
__host__ __device__ __forceinline__ std::uint16_t sat_pattern(int r, int c) {
    unsigned h = (unsigned)r * 1315423911u ^ (unsigned)c * 2654435761u;
    h ^= h >> 13;
    h *= 0x9E3779B1u;
    h ^= h >> 16;
    return (std::uint16_t)(h | 0x0001u);          // never 0x0000, never the 0x7FC0 poison below
}
constexpr std::uint16_t SAT_POISON = 0x7FC0u;     // bf16 quiet NaN; sat_pattern can produce it
                                                  // only if h|1 == 0x7FC0, impossible (even).

__device__ __forceinline__ unsigned long long sat_clock() {
    return __builtin_amdgcn_s_memrealtime();
}

// ------------------------------------------------------------------------------------------------
// MODE A -- the production mainloop, transcribed.
//
// Every element of the schedule is load-bearing and is kept: the issue/commit split around the
// MFMA block, the KH=2 half split that drops fragment pressure on the 256/256/32 row, the
// unconditional clamped prefetch (so the body stays one basic block), the volatile-asm fragment
// anchor that stops the machine scheduler hoisting MFMAs above the lgkmcnt drain, the accumulator
// anchor that stops it sinking them below the commit's vmcnt(0), and the s_setprio window.
// Removing any of them changes the ISA, so a body without them would not be this kernel's MFMA
// curve. Provenance for each: gemm_rs_mi300x.cpp lines 258-482 and exp_03 / exp_09.
//
// FILLER=false: a fixed tile list, grid-strided by dense role id.
// FILLER=true : the concurrent overlay's duration-matched filler -- re-runs the list until the
//               resource role signals done, checked BEFORE each tile so the concurrent window has
//               no traffic-free tail. The stop flag travels through the first 4 bytes of the A
//               double buffer because mode a already owns the entire 64 KB LDS budget and a
//               __shared__ int would overflow it (design.md 2).
// ------------------------------------------------------------------------------------------------
template<bool FILLER>
__device__ void sat_mode_a(const sat_args& g, int role_id, int role_ctas, int* shm) {
    shared_allocator al(shm);
    st_bf<SAT_BM, SAT_BK> (&As)[2] = al.allocate<st_bf<SAT_BM, SAT_BK>, 2>();
    st_bf<SAT_BN, SAT_BK> (&Bs)[2] = al.allocate<st_bf<SAT_BN, SAT_BK>, 2>();

    constexpr int WM = SAT_BM / SAT_WARPS_M;
    constexpr int WN = SAT_BN / SAT_WARPS_N;
    constexpr int KH = 2;
    constexpr int KS = SAT_BK / KH;
    static_assert(KS % 16 == 0 && KS * KH == SAT_BK);

    rt_bf<WM, KS, ducks::rt_layout::row> A_frag;
    rt_bf<WN, KS, ducks::rt_layout::row> B_frag;
    rt_fl<WM, WN, ducks::rt_layout::col> C_accum;

    using ST_A = st_bf<SAT_BM, SAT_BK>;
    using ST_B = st_bf<SAT_BN, SAT_BK>;
    constexpr int NT = kittens::group<m3::NUM_WARPS>::GROUP_THREADS;
    float4 abuf[m3::g2s_slots<ST_A, NT>];
    float4 bbuf[m3::g2s_slots<ST_B, NT>];

    const int warp     = kittens::warpid() % m3::NUM_WARPS;
    const int warp_row = warp / SAT_WARPS_N;
    const int warp_col = warp % SAT_WARPS_N;

    const int k_iters = g.k_iters;
    const int tiles   = g.a_tiles_m * g.a_tiles_n;
    int* const lds_flag = shm;

    unsigned int done = 0;
    for (int i = 0;; ++i) {
        const int t = role_id + i * role_ctas;
        if constexpr (FILLER) {
            if (threadIdx.x == 0) {
                const unsigned int seen = __hip_atomic_load(
                    g.res_done, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                *lds_flag = (seen >= (unsigned int)g.num_res_ctas) ? 1 : 0;
            }
            __syncthreads();
            const int stop = *lds_flag;
            __syncthreads();          // every thread has read it before load_commit overwrites
            if (stop) break;
        } else {
            if (t >= g.a_work) break;
        }

        const int slot = t % tiles;
        const int tm   = slot / g.a_tiles_n;
        const int tn   = slot % g.a_tiles_n;

        zero(C_accum);
        m3::load_issue<ST_A, NT>(abuf, g.a, {0, 0, tm, 0});
        m3::load_issue<ST_B, NT>(bbuf, g.b, {0, 0, tn, 0});
        m3::load_commit<NT>(As[0], abuf);
        m3::load_commit<NT>(Bs[0], bbuf);
        __syncthreads();

        for (int k = 0; k < k_iters; ++k) {
            const int kn = (k + 1 < k_iters) ? (k + 1) : k;   // clamped, not guarded
            m3::load_issue<ST_A, NT>(abuf, g.a, {0, 0, tm, kn});
            m3::load_issue<ST_B, NT>(bbuf, g.b, {0, 0, tn, kn});
            __builtin_amdgcn_s_setprio(1);
            #pragma unroll
            for (int kh = 0; kh < KH; ++kh) {
                load(A_frag, subtile_inplace<WM, KS>(As[k & 1], {warp_row, kh}));
                load(B_frag, subtile_inplace<WN, KS>(Bs[k & 1], {warp_col, kh}));
                m3::acquire_frags(A_frag, B_frag);
                mma_ABt(C_accum, A_frag, B_frag, C_accum);
            }
            m3::acc_anchor(C_accum);
            __builtin_amdgcn_s_setprio(0);
            m3::load_commit<NT>(As[(k + 1) & 1], abuf);
            m3::load_commit<NT>(Bs[(k + 1) & 1], bbuf);
            __syncthreads();
        }
        // The accumulator anchor is also the only consumer this ubench needs: an empty asm
        // volatile with a "+v" tie is a real use, so nothing in the MFMA chain is dead.
        m3::acc_anchor(C_accum);
        ++done;
    }

    if (threadIdx.x == 0) g.work_done[blockIdx.x] = done;
}

// ------------------------------------------------------------------------------------------------
// MODE B -- the reducer body at REDV=1, calling the shipped helper unmodified. One work unit is
// one (b_strip_rows x BN) strip of the owner's output, exactly as the reducer's tile loop issues
// it: 8 x 16 B source loads per output packet, source-ascending FP32, one RNE pack, one 16 B store.
// ------------------------------------------------------------------------------------------------
__device__ void sat_mode_b(const sat_args& g, int role_id, int role_ctas) {
    // The work list wraps over the distinct strips of the source region, so the total work is
    // identical at every CTA count while the swept footprint stays above this node's 256 MB
    // Infinity Cache (plan.md section 4).
    const int distinct = g.b_row_bands * g.b_cols_per_row;
    unsigned int done = 0;
    for (int t = role_id; t < g.b_work; t += role_ctas) {
        const int slot = t % distinct;
        const int lr   = slot / g.b_cols_per_row;
        const int col  = slot % g.b_cols_per_row;
        m3::pull_sum_bf16_strip_mlp8(
            g.red_dst, g.red_srcs,
            /*row_stride=*/(std::size_t)g.red_row_stride,
            /*row_offset=*/(std::size_t)(lr * g.b_strip_rows),
            /*column_offset=*/(std::size_t)(col * SAT_BN),
            /*valid_columns=*/(std::size_t)g.red_valid_cols,
            /*rows=*/g.b_strip_rows, /*cols=*/SAT_BN,
            /*packet_ok=*/true, threadIdx.x, SAT_THREADS);
        ++done;
    }
    if (threadIdx.x == 0) g.work_done[blockIdx.x] = done;
}

// ------------------------------------------------------------------------------------------------
// MODE C -- the real 16 B peer-packet emit.
//
// The packet indexing, the LDS-staged source, and the dst_base + r*dst_row_stride + c destination
// arithmetic are emit_band_packets' (gemm_rs_mi300x_hk_adapter.cuh:355-379) at cols == BN == 256,
// so the destination row stride is N elements and consecutive rows of one window land 16 KB apart
// on the peer -- the real scatter, not a contiguous memcpy.
//
// `issued` is carried ACROSS windows, because that is what the in-flight remote-op bound means:
// the number of 16 B peer stores a thread has outstanding before it drains. depth == 0 issues no
// drain at all inside the work list, which is the production shape (production drains once per
// release group, i.e. after ~64 stores per thread).
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void sat_emit_window(
        std::uint16_t* dst_base, int dst_row_stride,
        const std::uint16_t* stage, int depth, int& issued) {
    for (int packet = (int)threadIdx.x; packet < SAT_PACKETS_PER_WIN; packet += SAT_THREADS) {
        const int r = packet / SAT_PACKETS_PER_ROW;
        const int c = (packet % SAT_PACKETS_PER_ROW) * SAT_ELEMS_PER_PACKET;
        std::uint16_t* d = dst_base + (long)r * (long)dst_row_stride + c;
        const std::uint16_t* s = stage + r * SAT_BN + c;
        hk_gemm_rs::store_peer_packet16(d, s);
        if (depth > 0 && ++issued >= depth) {
            issued = 0;
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
    }
}

__device__ void sat_mode_c(const sat_args& g, int role_id, int role_ctas, int* shm) {
    std::uint16_t* const stage = reinterpret_cast<std::uint16_t*>(shm);
    for (int i = (int)threadIdx.x; i < SAT_WIN_ROWS * SAT_BN; i += SAT_THREADS) {
        stage[i] = sat_pattern(i / SAT_BN, i % SAT_BN);
    }
    __syncthreads();

    int issued = 0;
    int grp = 0;
    unsigned int done = 0;
    for (int w = role_id; w < g.c_work; w += role_ctas) {
        // single: the whole grid pushes to (me+1)%8, so with all eight ranks live this is a ring
        // and each used link carries exactly one source -- a per-rank number is a per-link number.
        // rr7: window w goes to peer (me+1+(w%7))%8, which enumerates the seven non-self ranks.
        const int peer = (g.fanout == 0) ? ((g.me + 1) % SAT_WORLD)
                                         : ((g.me + 1 + (w % 7)) % SAT_WORLD);
        const int widx = (g.fanout == 0) ? w : (w / 7);
        const int wrow = (widx / g.c_cols_per_row) * SAT_WIN_ROWS;
        const int wcol = (widx % g.c_cols_per_row) * SAT_BN;

        std::uint16_t* const peer_base = kittens::distributed::translate_peer<SAT_WORLD>(
            g.c_local_base, g.heap_desc.local_allocation_base, g.heap_desc.bases, peer);
        if (peer_base == nullptr) break;                 // fail closed, never fake a byte

        std::uint16_t* const dst = peer_base
            + g.c_slot_elems * (unsigned long long)g.me       // slot index = SOURCE rank
            + (unsigned long long)wrow * (unsigned long long)g.c_row_stride
            + (unsigned long long)wcol;

        sat_emit_window(dst, g.c_row_stride, stage, g.depth, issued);
        ++done;

        if (g.protocol && ++grp >= g.release_group) {
            grp = 0;
            // The production per-group protocol, additively: one release covering every payload
            // store issued since the last one (s_waitcnt vmcnt(0) + CTA barrier + system release
            // fence -> buffer_wbl2 sc0 sc1), then the leader's probe RMW and its relaxed
            // system-scope publication. Cells are distinct per (CTA, peer), so the RMW is
            // uncontended exactly as production's per-(src,lrow,col) cells are.
            kittens::distributed::producer_drain_release<
                kittens::distributed::memory_scope::system>();
            if (threadIdx.x == 0) {
                const int idx = m3::SIGNAL_GUARD_U32 + 2 * (int)(blockIdx.x * SAT_WORLD + peer);
                std::uint32_t* tgt = kittens::distributed::translate_peer<SAT_WORLD>(
                    g.sig_local + idx, g.sig_desc.local_allocation_base,
                    g.sig_desc.bases, peer);
                if (tgt != nullptr) {
                    (void)kittens::distributed::fetch_add_relaxed<
                        kittens::distributed::memory_scope::system>(tgt, 1u);
                    kittens::distributed::store_relaxed<
                        kittens::distributed::memory_scope::system>(tgt + 1, (std::uint32_t)w);
                }
            }
        }
    }
    if (g.depth > 0 || g.protocol == 0) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    if (threadIdx.x == 0) g.work_done[blockIdx.x] = done;
}

// ------------------------------------------------------------------------------------------------
// The C-matched reserve-only control: occupy C CUs for the same wall duration the live arm's
// resource role took, with no traffic. One polling thread per CTA with s_sleep backoff, the rest
// parked on the barrier, so the "idle" control does not itself generate a load stream.
// ------------------------------------------------------------------------------------------------
__device__ void sat_idle_spin(const sat_args& g) {
    if (threadIdx.x == 0) {
        const unsigned long long t0 = sat_clock();
        while (sat_clock() - t0 < g.deadline_ticks) __builtin_amdgcn_s_sleep(8);
        g.work_done[blockIdx.x] = 0u;
    }
    __syncthreads();
}

// ------------------------------------------------------------------------------------------------
// THE UBENCH KERNEL. MODE 0 = a, 1 = b, 2 = c.
// ------------------------------------------------------------------------------------------------
template<int MODE, bool CONC, bool CONTROL>
__global__ __launch_bounds__(SAT_THREADS, 1)
void sat_kernel(const sat_args g) {
    extern __shared__ alignment_dummy __shm[];
    int* const shm = (int*)&__shm[0];
    const int pid = blockIdx.x;

    if (threadIdx.x == 0) g.stamps[2 * pid + 0] = sat_clock();

    if constexpr (CONC) {
        if (pid < g.num_res_ctas) {
            if constexpr (CONTROL) {
                sat_idle_spin(g);
            } else if constexpr (MODE == 1) {
                sat_mode_b(g, pid, g.num_res_ctas);
            } else if constexpr (MODE == 2) {
                sat_mode_c(g, pid, g.num_res_ctas, shm);
            }
            if (threadIdx.x == 0) {
                (void)__hip_atomic_fetch_add(g.res_done, 1u, __ATOMIC_RELEASE,
                                             __HIP_MEMORY_SCOPE_AGENT);
            }
        } else {
            sat_mode_a<true>(g, pid - g.num_res_ctas, g.num_gemm_ctas, shm);
        }
    } else {
        if constexpr (MODE == 0)      sat_mode_a<false>(g, pid, (int)gridDim.x, shm);
        else if constexpr (MODE == 1) sat_mode_b(g, pid, (int)gridDim.x);
        else                          sat_mode_c(g, pid, (int)gridDim.x, shm);
    }

    if (threadIdx.x == 0) g.stamps[2 * pid + 1] = sat_clock();
}

// ------------------------------------------------------------------------------------------------
// Support kernels.
// ------------------------------------------------------------------------------------------------

// Tick-rate calibration: spin on s_memrealtime for `target` ticks while the host times the
// launch. The gfx950 sibling's "100 MHz / 10 ns" is a hypothesis to confirm here, not an input.
__global__ void sat_calib_kernel(unsigned long long target, unsigned long long* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const unsigned long long t0 = sat_clock();
    unsigned long long t1 = t0;
    while (t1 - t0 < target) {
        __builtin_amdgcn_s_sleep(2);
        t1 = sat_clock();
    }
    out[0] = t0;
    out[1] = t1;
}

// mode 0: small-magnitude bf16 operands (no overflow across 3712 accumulations, and no denormal
//         or NaN inputs, so panel a measures MFMA issue and nothing else)
// mode 1: the SAT_POISON pre-fill for a mode-c destination extent
// mode 2: the payload pattern (used for the reduce sources; content is irrelevant there, but a
//         deterministic fill keeps runs comparable)
__global__ void sat_fill_kernel(std::uint16_t* p, unsigned long long n, int mode) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
        std::uint16_t v;
        if (mode == 1) {
            v = SAT_POISON;
        } else {
            const std::uint16_t h = sat_pattern((int)(i >> 8), (int)(i & 0xFF));
            v = (mode == 0) ? (std::uint16_t)(0x3B00u | (h & 0x00FFu)) : h;
        }
        p[i] = v;
    }
}

// Destination-side verification of one slot: walk exactly the written extent, compare against the
// generating function, count bitwise mismatches, and fold a u64 fingerprint. This is stronger than
// an xor-fold (an xor of k identical windows degenerates for even k) and costs nothing in the
// timed path, because nothing is accumulated inside the emit.
__global__ void sat_verify_kernel(const std::uint16_t* base, unsigned long long slot_elems,
                                  int slot, int row_stride, int cols_per_row, int window_count,
                                  unsigned long long* mismatch, unsigned long long* fold) {
    const std::uint16_t* const slot_base = base + slot_elems * (unsigned long long)slot;
    const long long per_win = SAT_WIN_ROWS * SAT_BN;
    const long long total   = (long long)window_count * per_win;
    unsigned long long bad = 0, acc = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        const int w    = (int)(i / per_win);
        const int off  = (int)(i % per_win);
        const int r    = off / SAT_BN;
        const int c    = off % SAT_BN;
        const int wrow = (w / cols_per_row) * SAT_WIN_ROWS;
        const int wcol = (w % cols_per_row) * SAT_BN;
        const std::uint16_t got  = slot_base[(long long)(wrow + r) * row_stride + wcol + c];
        const std::uint16_t want = sat_pattern(r, c);
        if (got != want) ++bad;
        acc ^= ((unsigned long long)got) << ((off & 3) * 16);
    }
    (void)__hip_atomic_fetch_add(mismatch, bad, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    (void)__hip_atomic_fetch_xor(fold, acc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// ================================================================================================
// Host surface.
// ================================================================================================
namespace {

void hip_ok(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
}
void use_device(int device) { hip_ok(hipSetDevice(device), "hipSetDevice"); }

int device_count() {
    int count = 0;
    hip_ok(hipGetDeviceCount(&count), "hipGetDeviceCount");
    return count;
}

// Allocation and peer access follow harness/dhk_rt.cpp exactly (one process, eight devices, peer
// access enabled, fine-grained where the protocol needs uncached memory). Copied rather than
// imported because that module is another experiment's build product and must not be rebuilt here.
std::uintptr_t fine_alloc(int device, std::size_t nbytes) {
    use_device(device);
    void* pointer = nullptr;
    hip_ok(hipExtMallocWithFlags(&pointer, nbytes, hipDeviceMallocFinegrained),
           "hipExtMallocWithFlags(fine-grained)");
    hip_ok(hipMemset(pointer, 0, nbytes), "hipMemset");
    hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize");
    return reinterpret_cast<std::uintptr_t>(pointer);
}
std::uintptr_t plain_alloc(int device, std::size_t nbytes) {
    use_device(device);
    void* pointer = nullptr;
    hip_ok(hipMalloc(&pointer, nbytes), "hipMalloc");
    hip_ok(hipMemset(pointer, 0, nbytes), "hipMemset");
    hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize");
    return reinterpret_cast<std::uintptr_t>(pointer);
}
void free_device(int device, std::uintptr_t pointer) {
    use_device(device);
    hip_ok(hipFree(reinterpret_cast<void*>(pointer)), "hipFree");
}
void device_synchronize(int device) {
    use_device(device);
    hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize");
}
void zero_region(int device, std::uintptr_t pointer, std::size_t nbytes) {
    use_device(device);
    hip_ok(hipMemset(reinterpret_cast<void*>(pointer), 0, nbytes), "hipMemset(zero)");
    hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize(zero)");
}
py::bytes device_to_host(int device, std::uintptr_t pointer, std::size_t nbytes) {
    use_device(device);
    std::string buffer;
    buffer.resize(nbytes);
    hip_ok(hipMemcpy(buffer.data(), reinterpret_cast<const void*>(pointer), nbytes,
                     hipMemcpyDeviceToHost), "hipMemcpy D2H");
    return py::bytes(buffer);
}

py::dict enable_peer_access(int world) {
    py::dict report;
    py::list failures;
    int enabled = 0, already = 0;
    for (int device = 0; device < world; ++device) {
        use_device(device);
        for (int peer = 0; peer < world; ++peer) {
            if (peer == device) continue;
            int can = 0;
            hip_ok(hipDeviceCanAccessPeer(&can, device, peer), "hipDeviceCanAccessPeer");
            if (can == 0) { failures.append(py::make_tuple(device, peer, "cannot access")); continue; }
            const hipError_t status = hipDeviceEnablePeerAccess(peer, 0);
            if (status == hipSuccess) { ++enabled; }
            else if (status == hipErrorPeerAccessAlreadyEnabled) { ++already; (void)hipGetLastError(); }
            else { failures.append(py::make_tuple(device, peer, hipGetErrorString(status)));
                   (void)hipGetLastError(); }
        }
    }
    report["enabled"] = enabled;
    report["already_enabled"] = already;
    report["failures"] = failures;
    return report;
}

py::dict device_props(int device) {
    hipDeviceProp_t p{};
    hip_ok(hipGetDeviceProperties(&p, device), "hipGetDeviceProperties");
    py::dict out;
    out["name"] = std::string(p.name);
    out["gcn_arch_name"] = std::string(p.gcnArchName);
    out["multi_processor_count"] = p.multiProcessorCount;
    out["clock_rate_khz"] = p.clockRate;
    out["memory_clock_rate_khz"] = p.memoryClockRate;
    out["memory_bus_width_bits"] = p.memoryBusWidth;
    out["l2_cache_size_bytes"] = p.l2CacheSize;
    out["total_global_mem_bytes"] = (unsigned long long)p.totalGlobalMem;
    out["max_shared_memory_per_block"] = (unsigned long long)p.sharedMemPerBlock;
    out["warp_size"] = p.warpSize;
    // Analytic HBM peak from what the device itself reports; the source string is recorded next
    // to it in saturation.json so no borrowed spec number can leak into the figure.
    const double hbm_gbps = 2.0 * (double)p.memoryClockRate * 1e3 *
                            (double)p.memoryBusWidth / 8.0 / 1e9;
    out["hbm_peak_gbps_from_props"] = hbm_gbps;
    return out;
}

void fill_region(int device, std::uintptr_t pointer, unsigned long long elems, int mode) {
    use_device(device);
    sat_fill_kernel<<<1024, 256>>>(reinterpret_cast<std::uint16_t*>(pointer), elems, mode);
    hip_ok(hipGetLastError(), "sat_fill_kernel launch");
    hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize(fill)");
}

py::tuple calibrate_ticks(int device, unsigned long long target) {
    use_device(device);
    void* out = nullptr;
    hip_ok(hipMalloc(&out, 2 * sizeof(unsigned long long)), "hipMalloc(calib)");
    hip_ok(hipDeviceSynchronize(), "sync");
    const auto t0 = std::chrono::steady_clock::now();
    sat_calib_kernel<<<1, 1>>>(target, reinterpret_cast<unsigned long long*>(out));
    hip_ok(hipGetLastError(), "sat_calib_kernel launch");
    hip_ok(hipDeviceSynchronize(), "sync(calib)");
    const auto t1 = std::chrono::steady_clock::now();
    unsigned long long host[2] = {0, 0};
    hip_ok(hipMemcpy(host, out, sizeof(host), hipMemcpyDeviceToHost), "hipMemcpy(calib)");
    (void)hipFree(out);
    const double wall_ns =
        (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return py::make_tuple(host[1] - host[0], wall_ns);
}

py::tuple verify_slot(int device, std::uintptr_t base, unsigned long long slot_elems, int slot,
                      int row_stride, int cols_per_row, int window_count) {
    use_device(device);
    void* out = nullptr;
    hip_ok(hipMalloc(&out, 2 * sizeof(unsigned long long)), "hipMalloc(verify)");
    hip_ok(hipMemset(out, 0, 2 * sizeof(unsigned long long)), "hipMemset(verify)");
    auto* o = reinterpret_cast<unsigned long long*>(out);
    sat_verify_kernel<<<SAT_GRID_MAX, 256>>>(
        reinterpret_cast<const std::uint16_t*>(base), slot_elems, slot, row_stride,
        cols_per_row, window_count, o, o + 1);
    hip_ok(hipGetLastError(), "sat_verify_kernel launch");
    hip_ok(hipDeviceSynchronize(), "sync(verify)");
    unsigned long long host[2] = {0, 0};
    hip_ok(hipMemcpy(host, out, sizeof(host), hipMemcpyDeviceToHost), "hipMemcpy(verify)");
    (void)hipFree(out);
    return py::make_tuple(host[0], host[1]);
}

// ---- launch ------------------------------------------------------------------------------------
// One py::dict of scalars and pointers rather than forty positional arguments: the per-launch host
// cost is irrelevant next to a 10-100 ms kernel, and a mis-ordered positional argument in a
// forty-argument signature is exactly the class of bug that silently produces a plausible number.
long long dget(const py::dict& d, const char* key) {
    if (!d.contains(key)) throw std::invalid_argument(std::string("missing key: ") + key);
    return d[key].cast<long long>();
}
unsigned long long dgetu(const py::dict& d, const char* key) {
    if (!d.contains(key)) throw std::invalid_argument(std::string("missing key: ") + key);
    return d[key].cast<unsigned long long>();
}

hk_gemm_rs::symmetric_descriptor make_desc(const std::vector<std::uintptr_t>& bases,
                                           std::uintptr_t local_base) {
    if (bases.size() != (std::size_t)SAT_WORLD) {
        throw std::invalid_argument("expected exactly 8 allocation bases");
    }
    kittens::peer_bases<std::byte> pb{
        reinterpret_cast<std::byte*>(bases[0]), reinterpret_cast<std::byte*>(bases[1]),
        reinterpret_cast<std::byte*>(bases[2]), reinterpret_cast<std::byte*>(bases[3]),
        reinterpret_cast<std::byte*>(bases[4]), reinterpret_cast<std::byte*>(bases[5]),
        reinterpret_cast<std::byte*>(bases[6]), reinterpret_cast<std::byte*>(bases[7])};
    return hk_gemm_rs::make_symmetric_descriptor(reinterpret_cast<const void*>(local_base), pb);
}

double sat_launch(const py::dict& d) {
    const int device = (int)dget(d, "device");
    const int mode   = (int)dget(d, "mode");
    const bool conc  = dget(d, "conc") != 0;
    const bool ctrl  = dget(d, "control") != 0;
    const int grid   = (int)dget(d, "grid");
    if (grid < 1 || grid > SAT_GRID_MAX) throw std::invalid_argument("grid outside [1, 304]");

    const int a_rows = (int)dget(d, "a_rows");
    const int a_cols = (int)dget(d, "a_cols");
    const int b_rows = (int)dget(d, "b_rows");
    _gl_A ga = kittens::make_gl<_gl_A>(dgetu(d, "a_ptr"), 1, 1, a_rows, a_cols);
    _gl_B gb = kittens::make_gl<_gl_B>(dgetu(d, "b_ptr"), 1, 1, b_rows, a_cols);

    sat_args g{ga, gb};
    g.k_iters   = (int)dget(d, "k_iters");
    g.a_tiles_m = (int)dget(d, "a_tiles_m");
    g.a_tiles_n = (int)dget(d, "a_tiles_n");
    g.a_work    = (int)dget(d, "a_work");

    const auto red_base   = dgetu(d, "red_src_base");
    const auto red_slot   = dgetu(d, "red_slot_elems");
    const auto* red_u16   = reinterpret_cast<const std::uint16_t*>(red_base);
    g.red_srcs = hk_gemm_rs::bf16_sources8{
        red_u16 + red_slot * 0u, red_u16 + red_slot * 1u, red_u16 + red_slot * 2u,
        red_u16 + red_slot * 3u, red_u16 + red_slot * 4u, red_u16 + red_slot * 5u,
        red_u16 + red_slot * 6u, red_u16 + red_slot * 7u};
    g.red_dst        = reinterpret_cast<std::uint16_t*>(dgetu(d, "red_dst"));
    g.red_row_stride = (int)dget(d, "red_row_stride");
    g.red_valid_cols = (int)dget(d, "red_valid_cols");
    g.b_strip_rows   = (int)dget(d, "b_strip_rows");
    g.b_cols_per_row = (int)dget(d, "b_cols_per_row");
    g.b_row_bands    = (int)dget(d, "b_row_bands");
    g.b_work         = (int)dget(d, "b_work");

    g.c_local_base   = reinterpret_cast<std::uint16_t*>(dgetu(d, "heap_local"));
    g.heap_desc      = make_desc(d["heap_bases"].cast<std::vector<std::uintptr_t>>(),
                                 dgetu(d, "heap_local"));
    g.c_slot_elems   = dgetu(d, "c_slot_elems");
    g.c_row_stride   = (int)dget(d, "c_row_stride");
    g.c_cols_per_row = (int)dget(d, "c_cols_per_row");
    g.c_work         = (int)dget(d, "c_work");
    g.depth          = (int)dget(d, "depth");
    g.fanout         = (int)dget(d, "fanout");

    g.protocol      = (int)dget(d, "protocol");
    g.release_group = (int)dget(d, "release_group");
    g.sig_local     = reinterpret_cast<std::uint32_t*>(dgetu(d, "sig_local"));
    g.sig_desc      = make_desc(d["sig_bases"].cast<std::vector<std::uintptr_t>>(),
                                dgetu(d, "sig_local"));

    g.me             = (int)dget(d, "me");
    g.num_res_ctas   = (int)dget(d, "num_res_ctas");
    g.num_gemm_ctas  = (int)dget(d, "num_gemm_ctas");
    g.deadline_ticks = dgetu(d, "deadline_ticks");
    g.stamps         = reinterpret_cast<unsigned long long*>(dgetu(d, "stamps"));
    g.work_done      = reinterpret_cast<unsigned int*>(dgetu(d, "work_done"));
    g.res_done       = reinterpret_cast<unsigned int*>(dgetu(d, "res_done"));

    use_device(device);
    // 65,536 B of dynamic LDS in EVERY arm, so occupancy is 1 CTA/CU everywhere and the
    // isolated arms are not measured at a different occupancy than the concurrent ones.
    const int shmem = SAT_LDS;
    const dim3 gr((unsigned)grid), bl((unsigned)SAT_THREADS);

    const auto t0 = std::chrono::steady_clock::now();
    if (!conc) {
        if (mode == 0)      sat_kernel<0, false, false><<<gr, bl, shmem>>>(g);
        else if (mode == 1) sat_kernel<1, false, false><<<gr, bl, shmem>>>(g);
        else                sat_kernel<2, false, false><<<gr, bl, shmem>>>(g);
    } else if (!ctrl) {
        if (mode == 1)      sat_kernel<1, true, false><<<gr, bl, shmem>>>(g);
        else if (mode == 2) sat_kernel<2, true, false><<<gr, bl, shmem>>>(g);
        else throw std::invalid_argument("mode a has no concurrent overlay");
    } else {
        if (mode == 1)      sat_kernel<1, true, true><<<gr, bl, shmem>>>(g);
        else if (mode == 2) sat_kernel<2, true, true><<<gr, bl, shmem>>>(g);
        else throw std::invalid_argument("mode a has no reserve control");
    }
    hip_ok(hipGetLastError(), "sat_kernel launch");
    const auto t1 = std::chrono::steady_clock::now();
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

} // namespace

PYBIND11_MODULE(sat_ubench, module) {
    module.doc() = "exp_21 resource-saturation microbenchmark for the gfx942 GEMM-RS kernel";

    module.def("device_count", &device_count);
    module.def("device_props", &device_props, py::arg("device") = 0);
    module.def("enable_peer_access", &enable_peer_access, py::arg("world") = 8);
    module.def("fine_alloc", &fine_alloc, py::arg("device"), py::arg("nbytes"));
    module.def("plain_alloc", &plain_alloc, py::arg("device"), py::arg("nbytes"));
    module.def("free_device", &free_device, py::arg("device"), py::arg("pointer"));
    module.def("device_synchronize", &device_synchronize, py::arg("device"));
    module.def("zero_region", &zero_region, py::arg("device"), py::arg("pointer"),
               py::arg("nbytes"));
    module.def("device_to_host", &device_to_host, py::arg("device"), py::arg("pointer"),
               py::arg("nbytes"));
    module.def("fill_region", &fill_region, py::arg("device"), py::arg("pointer"),
               py::arg("elems"), py::arg("mode"));
    module.def("calibrate_ticks", &calibrate_ticks, py::arg("device"), py::arg("target"));
    module.def("verify_slot", &verify_slot, py::arg("device"), py::arg("base"),
               py::arg("slot_elems"), py::arg("slot"), py::arg("row_stride"),
               py::arg("cols_per_row"), py::arg("window_count"));
    module.def("launch", &sat_launch, py::arg("cfg"));

    module.attr("BM") = SAT_BM;
    module.attr("BN") = SAT_BN;
    module.attr("BK") = SAT_BK;
    module.attr("WIN_ROWS") = SAT_WIN_ROWS;
    module.attr("THREADS") = SAT_THREADS;
    module.attr("GRID_MAX") = SAT_GRID_MAX;
    module.attr("WORLD") = SAT_WORLD;
    module.attr("LDS_BYTES") = SAT_LDS;
    module.attr("WIN_BYTES") = SAT_WIN_BYTES;
    module.attr("PACKETS_PER_WIN") = SAT_PACKETS_PER_WIN;
    module.attr("POISON") = (int)SAT_POISON;
    module.attr("SIGNAL_GUARD_U32") = m3::SIGNAL_GUARD_U32;
}

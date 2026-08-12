// exp_12 launch-cost probes: kernels that share our megakernel's launch
// geometry but do (almost) no work, so the graded protocol's per-call fixed
// cost can be split into "launch + sync machinery" and "our protocol".
//
// Geometry matched deliberately: 512 threads, __launch_bounds__(512, 1), and a
// dynamic-LDS request that can be set to the same 65536 B the 256x256x32 rows
// use -- at 64 KB only one workgroup fits per CU, so a 304-CTA grid is spread
// one-per-CU exactly as the real kernel's is. Any of those could plausibly be
// what a launch costs, so none of them is allowed to differ silently.
//
// Nothing here is part of the deliverable kernel; it is an instrument.

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

namespace {

constexpr int CTA_THREADS = 512;
constexpr int ERR_MASK = (1 << 25) | (1 << 26);   // the real kernel's two bits

// The `blockIdx.x == 0x7fffffff` guards are never taken for any grid we launch,
// but the compiler cannot prove that, so the dynamic-LDS allocation and the
// pointer arguments stay live without adding a memory op to the timed path.
__device__ __forceinline__ bool never() { return blockIdx.x == 0x7fffffffu; }

__global__ __launch_bounds__(CTA_THREADS, 1)
void k_empty(unsigned* sink) {
    extern __shared__ char shm[];
    if (never()) { shm[0] = 1; sink[0] = (unsigned)shm[0]; }
}

// kittens::distributed::epoch32 spelled out, on the same per-CTA cell layout
// (`ep_cell + EP_GEMM_U32 + blockIdx.x`), followed by the sticky error-bit
// load. This is exactly what all 304 of our CTAs execute before any useful
// work, and it is the only thing this kernel does.
__global__ __launch_bounds__(CTA_THREADS, 1)
void k_epoch(unsigned* cells, const int* err, unsigned* sink) {
    extern __shared__ char shm[];
    if (never()) { shm[0] = 1; sink[0] = (unsigned)shm[0]; }

    unsigned* cell = cells + blockIdx.x;
    if (threadIdx.x == 0) {
        __hip_atomic_fetch_add(cell, 1u, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    const unsigned epoch = __hip_atomic_load(cell, __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    const int bits = __hip_atomic_load(err, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
    if (bits & ERR_MASK) return;
    if (never() && threadIdx.x == 0) sink[0] = epoch;
}

void reserve_lds(const void* function, int lds_bytes) {
    // Per instantiation, not per call: our kernel learned the hard way that a
    // hipFuncSetAttribute on the launch path dominates the small shapes.
    const hipError_t status = hipFuncSetAttribute(
        function, hipFuncAttributeMaxDynamicSharedMemorySize, lds_bytes);
    if (status != hipSuccess) {
        throw std::runtime_error(std::string("nullk: cannot reserve ") +
                                 std::to_string(lds_bytes) + " B of LDS: " +
                                 hipGetErrorString(status));
    }
}

void launch(int kind, unsigned long long stream_ptr, int grid, int threads,
            int lds_bytes, unsigned long long cells, unsigned long long err,
            unsigned long long sink) {
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    static bool reserved_empty = false, reserved_epoch = false;
    if (kind == 0) {
        if (!reserved_empty) { reserve_lds((const void*)k_empty, 65536);
                               reserved_empty = true; }
        k_empty<<<dim3(grid), dim3(threads), (std::size_t)lds_bytes, stream>>>(
            reinterpret_cast<unsigned*>(sink));
    } else {
        if (!reserved_epoch) { reserve_lds((const void*)k_epoch, 65536);
                               reserved_epoch = true; }
        k_epoch<<<dim3(grid), dim3(threads), (std::size_t)lds_bytes, stream>>>(
            reinterpret_cast<unsigned*>(cells),
            reinterpret_cast<const int*>(err),
            reinterpret_cast<unsigned*>(sink));
    }
}

}  // namespace

PYBIND11_MODULE(nullk, m) {
    m.doc() = "exp_12 launch-cost probes (empty / epoch-handshake kernels)";
    m.def("launch", &launch,
          "kind 0 = empty, 1 = per-CTA epoch32 + error-bit check",
          pybind11::arg("kind"), pybind11::arg("stream"), pybind11::arg("grid"),
          pybind11::arg("threads"), pybind11::arg("lds"),
          pybind11::arg("cells") = 0ull, pybind11::arg("err") = 0ull,
          pybind11::arg("sink") = 0ull);
}

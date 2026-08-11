// exp_21 preamble — fabric RATE microbenchmark for the mode-7 design crux.
//
// Question: can xGMI sustain the remote packed-bf16 atomic OP RATE mode 7
// needs, or is the RMW path flit/op-limited far below the byte path? Mode 2
// moves ~312 MB/rank as ~19.5M 16-B vector stores; mode 7 moves the same
// bytes as ~78M 4-B atomic RMWs (4x ops), CONTIGUOUS 128 B per half-wave
// (the epilogue's exact pattern). Whether that is feasible depends entirely
// on whether the memory pipeline coalesces same-line atomics.
//
// This tests THROUGHPUT only, on coarse-grain hipMalloc + P2P (the exp_18
// heap already proved correctness there; RATE is a fabric-path property, not
// a heap-grain property). Mori-heap correctness of the atomic itself is
// covered by the mode-7 dual-write detector inside the real harness.
//
// Compile (in subha_k1):
//   hipcc -O2 --offload-arch=gfx950 -munsafe-fp-atomics ubench_fabric_rate.cpp -o ubench_fabric_rate
// Run: ./ubench_fabric_rate
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdint>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  printf("ERR %s:%d %s\n",#x,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)

constexpr int kRegionDwordsPerCTA = 1 << 19;      // 512K dwords = 2 MB per CTA
constexpr unsigned int kThreads = 256;
constexpr int kIters = 8;

__global__ void k_stores(std::uint4* base, int iters) {
  std::uint4* my = base + (std::size_t)blockIdx.x * (kRegionDwordsPerCTA >> 2);
  const int n = kRegionDwordsPerCTA >> 2;
  const std::uint4 v = {1u, 2u, 3u, 4u};
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll 1
    for (int i = threadIdx.x; i < n; i += kThreads) my[i] = v;
  }
}

__global__ void k_atomic(unsigned int* base, int iters) {
  unsigned int* my = base + (std::size_t)blockIdx.x * kRegionDwordsPerCTA;
  const unsigned int ob = 0x3F803F80u;
  const __hip_bfloat162 one = *reinterpret_cast<__hip_bfloat162*>(&ob);
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll 1
    for (int i = threadIdx.x; i < kRegionDwordsPerCTA; i += kThreads) {
      unsafeAtomicAdd(reinterpret_cast<__hip_bfloat162*>(my + i), one);
    }
  }
}

// Same op count, but every wave instruction touches 64 DISTINCT 128-B lines:
// the cannot-merge worst case.
__global__ void k_atomic_scatter(unsigned int* base, int iters) {
  unsigned int* my = base + (std::size_t)blockIdx.x * kRegionDwordsPerCTA;
  const unsigned int ob = 0x3F803F80u;
  const __hip_bfloat162 one = *reinterpret_cast<__hip_bfloat162*>(&ob);
  const unsigned int mask = (unsigned int)kRegionDwordsPerCTA - 1u;
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll 1
    for (int i = threadIdx.x; i < kRegionDwordsPerCTA; i += kThreads) {
      const unsigned int idx = (((unsigned int)(i & 63)) << 5) +
                               (((unsigned int)(i >> 6)) << 11);
      unsafeAtomicAdd(reinterpret_cast<__hip_bfloat162*>(my + (idx & mask)), one);
    }
  }
}

template <typename F>
double timeit(F&& f) {
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  CK(hipDeviceSynchronize());
  CK(hipEventRecord(a)); f(); CK(hipEventRecord(b));
  CK(hipEventSynchronize(b));
  float ms = 0.f; CK(hipEventElapsedTime(&ms, a, b));
  CK(hipGetLastError());
  CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
  return (double)ms * 1e3;   // us
}

static void run(const char* variant, int ncta, std::size_t bytes, double us,
                double gops) {
  printf("%-11s %4d %9.1fMB %9.0f us %9.1f GB/s", variant, ncta, bytes / 1e6,
         us, bytes / 1e3 / us);
  if (gops > 0.0) printf("  %7.2f Gop/s", gops / (us * 1e-6));
  printf("\n");
}

int main() {
  int nd = 0; CK(hipGetDeviceCount(&nd));
  if (nd < 2) { printf("need >=2 GPUs\n"); return 1; }
  CK(hipSetDevice(0));
  void* buf = nullptr; CK(hipMalloc(&buf, (std::size_t)512 << 20));
  CK(hipMemset(buf, 0, (std::size_t)512 << 20));
  CK(hipSetDevice(1));
  hipError_t pe = hipDeviceEnablePeerAccess(0, 0);
  if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) {
    printf("ERR p2p: %s\n", hipGetErrorString(pe)); return 1; }
  const int ctas_list[4] = {64, 128, 192, 256};
  printf("# writer on dev1 -> dev0 (PEER) and dev0 -> dev0 (LOCAL).");
  printf("  bytes/launch include iters.\n");
  for (int ci = 0; ci < 4; ++ci) {
    const int ncta = ctas_list[ci];
    const std::size_t bytes = (std::size_t)ncta *
        (std::size_t)kRegionDwordsPerCTA * 4u * (std::size_t)kIters;
    const double gops = (double)bytes / 4.0 / 1e9;
    CK(hipSetDevice(1));
    run("stores-PEER", ncta, bytes,
        timeit([&]{ hipLaunchKernelGGL(k_stores, dim3(ncta), dim3(kThreads),
            0, 0, (std::uint4*)buf, kIters); }), 0.0);
    run("atomic-PEER", ncta, bytes,
        timeit([&]{ hipLaunchKernelGGL(k_atomic, dim3(ncta), dim3(kThreads),
            0, 0, (unsigned int*)buf, kIters); }), gops);
    run("atomSC-PEER", ncta, bytes,
        timeit([&]{ hipLaunchKernelGGL(k_atomic_scatter, dim3(ncta),
            dim3(kThreads), 0, 0, (unsigned int*)buf, kIters); }), gops);
    CK(hipSetDevice(0));
    run("atomic-LOCAL", ncta, bytes,
        timeit([&]{ hipLaunchKernelGGL(k_atomic, dim3(ncta), dim3(kThreads),
            0, 0, (unsigned int*)buf, kIters); }), gops);
    run("stores-LOCAL", ncta, bytes,
        timeit([&]{ hipLaunchKernelGGL(k_stores, dim3(ncta), dim3(kThreads),
            0, 0, (std::uint4*)buf, kIters); }), 0.0);
  }
  return 0;
}

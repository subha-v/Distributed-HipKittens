/**
 * @file
 * @brief Private portability helpers for CDNA4 distributed operations.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define KITTENS_DISTRIBUTED_DEVICE_INLINE __device__ __forceinline__
#define KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define KITTENS_DISTRIBUTED_DEVICE_INLINE inline
#define KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE inline
#endif

namespace kittens::distributed::detail {

#if defined(__HIP_DEVICE_COMPILE__)
using packet16 = uint4;
#else
struct alignas(16) packet16 {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    std::uint32_t w;
};
#endif

static_assert(sizeof(packet16) == 16,
              "distributed packets must remain exactly 16 bytes");
static_assert(alignof(packet16) >= 16,
              "distributed packets must remain 16-byte aligned");

KITTENS_DISTRIBUTED_HOST_DEVICE_INLINE bool is_aligned_16(const void* pointer) {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 15u) == 0u;
}

KITTENS_DISTRIBUTED_DEVICE_INLINE unsigned int thread_id() {
#if defined(__HIP_DEVICE_COMPILE__)
    return threadIdx.x;
#else
    return 0u;
#endif
}

KITTENS_DISTRIBUTED_DEVICE_INLINE unsigned int thread_count() {
#if defined(__HIP_DEVICE_COMPILE__)
    return blockDim.x;
#else
    return 1u;
#endif
}

KITTENS_DISTRIBUTED_DEVICE_INLINE bool is_leader() {
    return thread_id() == 0u;
}

KITTENS_DISTRIBUTED_DEVICE_INLINE void cta_barrier() {
#if defined(__HIP_DEVICE_COMPILE__)
    __syncthreads();
#endif
}

} // namespace kittens::distributed::detail

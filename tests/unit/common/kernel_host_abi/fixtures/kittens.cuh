#pragma once

#ifndef PGL_HEADER
#error "PGL_HEADER must name the CDNA3 or CDNA4 PGL header"
#endif

#include <bit>
#include <cstddef>
#include <cstdint>

#define __device__
#define __host__
#define __forceinline__ inline

struct alignas(16) uint4 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
};

struct stub_thread_index {
    unsigned int x = 0;
};

inline stub_thread_index threadIdx{};

inline float __uint_as_float(std::uint32_t value) {
    return std::bit_cast<float>(value);
}

inline std::uint32_t __float_as_uint(float value) {
    return std::bit_cast<std::uint32_t>(value);
}

inline int __shfl(int value, int) { return value; }
inline int atomicOr(int* value, int bits) {
    const int prior = *value;
    *value |= bits;
    return prior;
}
inline std::uint32_t atomicMax(std::uint32_t* value, std::uint32_t next) {
    const auto prior = *value;
    if (next > *value) *value = next;
    return prior;
}

inline void k0p6_put_row_payload(unsigned char*, const std::uint64_t*,
                                 std::size_t, int, int) {}

#include PGL_HEADER

namespace kittens::distributed {

enum class memory_scope {
    agent,
    system,
};

struct wait_result {
    bool ready = false;
    std::uint64_t observed = 0;
    std::uint64_t spins = 0;
};

struct wait_result64 {
    bool ready = false;
    std::uint64_t observed = 0;
    std::uint64_t spins = 0;
};

template<typename Result>
inline void init_wait_result(Result& result) {
    result = {};
}

template<int World, typename T>
inline T* translate_peer(T* local, const std::byte*,
                         kittens::peer_bases<std::byte>, int) {
    return local;
}

template<int World, typename T>
inline const T* translate_peer(const T* local, const std::byte*,
                               kittens::peer_bases<std::byte>, int) {
    return local;
}

inline void store_peer_packets(void*, const void*, std::size_t, std::size_t,
                               std::size_t) {}
inline std::uint32_t epoch32(std::uint32_t*) { return 1; }
inline void producer_drain_release() {}
inline void consumer_drain() {}

template<memory_scope Scope>
inline void cta_acquire() {}

template<memory_scope Scope>
inline void thread_acquire() {}

template<memory_scope Scope>
inline void thread_release() {}

template<memory_scope Scope = memory_scope::system>
inline void publish_epoch_relaxed(std::uint32_t*, std::uint32_t) {}

template<memory_scope Scope, typename T>
inline void store_relaxed(T* target, T value) {
    *target = value;
}

template<memory_scope Scope, typename T>
inline T load_relaxed(const T* target) {
    return *target;
}

template<memory_scope Scope, typename T, typename Result>
inline void bounded_poll_relaxed_into(const T*, T, std::uint64_t,
                                     Result&) {}

inline void bounded_wait_slot_reusable_into(const std::uint32_t*,
                                            std::uint32_t, std::uint64_t,
                                            wait_result&) {}

template<memory_scope Scope>
inline std::uint32_t fetch_add_relaxed(std::uint32_t* target,
                                      std::uint32_t value) {
    const auto prior = *target;
    *target += value;
    return prior;
}

template<memory_scope Scope>
inline std::uint32_t reserve_rows_relaxed(std::uint32_t* target,
                                         std::uint32_t value) {
    return fetch_add_relaxed<Scope>(target, value);
}

template<memory_scope Scope>
inline void publish_epoch_word_relaxed(std::uint64_t*, std::uint32_t,
                                       std::uint32_t) {}

template<memory_scope Scope>
inline void bounded_poll_epoch_word_relaxed_into(const std::uint64_t*,
                                                 std::uint32_t,
                                                 std::uint64_t,
                                                 wait_result64&) {}

inline std::uint32_t epoch_word_payload(std::uint64_t word) {
    return static_cast<std::uint32_t>(word);
}

} // namespace kittens::distributed

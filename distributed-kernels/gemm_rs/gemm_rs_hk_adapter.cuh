#pragma once

// Thin HipKittens communication boundary for the preserved GEMM-RS schedule.
// Tile ownership, signal indices, epochs, and reduction order remain operator
// policy.  This file maps only peer projection, packet movement, ordering, and
// directed slot lifetime onto kittens::pgl / kittens::distributed.

#include "kittens.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hk_gemm_rs {

struct symmetric_descriptor {
    const std::byte* local_allocation_base;
    kittens::peer_bases<std::byte> bases;
};

static_assert(std::is_standard_layout_v<symmetric_descriptor>);
static_assert(std::is_trivially_copyable_v<symmetric_descriptor>);
static_assert(sizeof(void*) == 8,
              "GEMM-RS symmetric descriptor ABI requires 64-bit pointers");
static_assert(sizeof(symmetric_descriptor) == 72,
              "GEMM-RS symmetric descriptor ABI must remain exactly 72 bytes");
static_assert(alignof(symmetric_descriptor) == 8,
              "GEMM-RS symmetric descriptor ABI must remain 8-byte aligned");
static_assert(offsetof(symmetric_descriptor, local_allocation_base) == 0);
static_assert(offsetof(symmetric_descriptor, bases) == 8);

[[nodiscard]] inline symmetric_descriptor make_symmetric_descriptor(
        const void* local_allocation_base,
        kittens::peer_bases<std::byte> allocation_bases) {
    return {reinterpret_cast<const std::byte*>(local_allocation_base),
            allocation_bases};
}

template<typename GL>
__device__ __forceinline__ GL peer_view(
        const GL& local, const symmetric_descriptor* descriptor, int rank) {
    const auto layout = kittens::make_parallel_global_layout<8>(
        local, descriptor->local_allocation_base, descriptor->bases);
    return layout.on(rank);
}

__device__ __forceinline__ void store_peer_packet16(void* destination,
                                                     const void* staged) {
    kittens::distributed::store_peer_packets(
        destination, staged, 16u, 0u, 1u);
}

__device__ __forceinline__ std::uint32_t next_epoch_workgroup(
        std::uint32_t* cell) {
    return kittens::distributed::epoch32(cell);
}

__device__ __forceinline__ void release_payload_system() {
    kittens::distributed::producer_drain_release();
}

__device__ __forceinline__ void acquire_payload_system() {
    kittens::distributed::cta_acquire<
        kittens::distributed::memory_scope::system>();
}

__device__ __forceinline__ void publish_relaxed(
        std::uint32_t* cell, std::uint32_t epoch, bool local) {
    if (local) {
        kittens::distributed::publish_epoch_relaxed<
            kittens::distributed::memory_scope::agent>(cell, epoch);
    } else {
        kittens::distributed::publish_epoch_relaxed<
            kittens::distributed::memory_scope::system>(cell, epoch);
    }
}

__device__ __forceinline__ void publish_tile_epoch(
        std::uint32_t* local_signals, const symmetric_descriptor* descriptor,
        int destination, int me, int signal_index, std::uint32_t epoch) {
    std::uint32_t* target = kittens::distributed::translate_peer<8>(
        local_signals + signal_index, descriptor->local_allocation_base,
        descriptor->bases, destination);
    publish_relaxed(target, epoch, destination == me);
}

__device__ __forceinline__ bool wait_tile_epoch(
        const std::uint32_t* signal, std::uint32_t epoch,
        std::uint64_t spin_limit, int*, int) {
    kittens::distributed::wait_result result;
    kittens::distributed::init_wait_result(result);
    kittens::distributed::bounded_poll_relaxed_into<
        kittens::distributed::memory_scope::system>(
            signal, epoch, spin_limit, result);
    return result.ready;
}

__device__ __forceinline__ bool wait_reuse_credit(
        const std::uint32_t* credit, std::uint32_t next_epoch,
        std::uint64_t spin_limit, int*, int) {
    kittens::distributed::wait_result result;
    kittens::distributed::init_wait_result(result);
    kittens::distributed::bounded_wait_slot_reusable_into(
        credit, next_epoch, spin_limit, result);
    return result.ready;
}

__device__ __forceinline__ void finish_tile_consumption() {
    kittens::distributed::consumer_drain();
}

__device__ __forceinline__ void publish_reuse_credit(
        std::uint32_t* local_signals, const symmetric_descriptor* descriptor,
        int source, int me, int credit_index, std::uint32_t epoch) {
    std::uint32_t* target = kittens::distributed::translate_peer<8>(
        local_signals + credit_index, descriptor->local_allocation_base,
        descriptor->bases, source);
    publish_relaxed(target, epoch, source == me);
}

__device__ __forceinline__ bool error_bit_set(const int* error, int bit) {
    if (error == nullptr) return false;
    int observed = 0;
    if ((threadIdx.x & 63u) == 0u) {
        observed = kittens::distributed::load_relaxed<
            kittens::distributed::memory_scope::agent>(error);
    }
    observed = __shfl(observed, 0);
    return (observed & bit) != 0;
}

// Exact exp-24 REDV=1 owner-pull arithmetic. Eight 16-byte loads are issued
// before consumption; accumulation stays source-ascending in FP32 and bias is
// applied before the single RNE BF16 output pack.
struct bf16_sources8 {
    const std::uint16_t* source0;
    const std::uint16_t* source1;
    const std::uint16_t* source2;
    const std::uint16_t* source3;
    const std::uint16_t* source4;
    const std::uint16_t* source5;
    const std::uint16_t* source6;
    const std::uint16_t* source7;
};

union packet_bf16x8 {
    uint4 packet;
    std::uint16_t value[8];
};

__device__ __forceinline__ float bf16_to_float(std::uint16_t value) {
    return __uint_as_float(static_cast<std::uint32_t>(value) << 16);
}

__device__ __forceinline__ std::uint16_t float_to_bf16_rne(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<std::uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

__device__ __forceinline__ packet_bf16x8 load_packet(
        const std::uint16_t* source, std::size_t offset) {
    packet_bf16x8 result;
    result.packet = *reinterpret_cast<const uint4*>(source + offset);
    return result;
}

__device__ __forceinline__ void add_packet(
        packet_bf16x8 input, float& a0, float& a1, float& a2, float& a3,
        float& a4, float& a5, float& a6, float& a7) {
    a0 += bf16_to_float(input.value[0]);
    a1 += bf16_to_float(input.value[1]);
    a2 += bf16_to_float(input.value[2]);
    a3 += bf16_to_float(input.value[3]);
    a4 += bf16_to_float(input.value[4]);
    a5 += bf16_to_float(input.value[5]);
    a6 += bf16_to_float(input.value[6]);
    a7 += bf16_to_float(input.value[7]);
}

template<int Rows, int Columns>
__device__ __forceinline__ void pull_sum_bf16_tile_mlp8(
        std::uint16_t* __restrict__ destination, bf16_sources8 sources,
        std::size_t row_stride, std::size_t row_offset,
        std::size_t column_offset, const std::uint16_t* __restrict__ bias,
        float bias_scale, unsigned int tid, unsigned int thread_count) {
    static_assert(Columns % 8 == 0);
    constexpr int packets_per_row = Columns / 8;
    constexpr int packet_count = Rows * packets_per_row;
    for (int packet = static_cast<int>(tid); packet < packet_count;
         packet += static_cast<int>(thread_count)) {
        const std::size_t row = row_offset + packet / packets_per_row;
        const std::size_t column = column_offset +
            static_cast<std::size_t>(packet % packets_per_row) * 8u;
        const std::size_t offset = row * row_stride + column;
        const auto p0 = load_packet(sources.source0, offset);
        const auto p1 = load_packet(sources.source1, offset);
        const auto p2 = load_packet(sources.source2, offset);
        const auto p3 = load_packet(sources.source3, offset);
        const auto p4 = load_packet(sources.source4, offset);
        const auto p5 = load_packet(sources.source5, offset);
        const auto p6 = load_packet(sources.source6, offset);
        const auto p7 = load_packet(sources.source7, offset);
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        float a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
        add_packet(p0, a0, a1, a2, a3, a4, a5, a6, a7);
        add_packet(p1, a0, a1, a2, a3, a4, a5, a6, a7);
        add_packet(p2, a0, a1, a2, a3, a4, a5, a6, a7);
        add_packet(p3, a0, a1, a2, a3, a4, a5, a6, a7);
        add_packet(p4, a0, a1, a2, a3, a4, a5, a6, a7);
        add_packet(p5, a0, a1, a2, a3, a4, a5, a6, a7);
        add_packet(p6, a0, a1, a2, a3, a4, a5, a6, a7);
        add_packet(p7, a0, a1, a2, a3, a4, a5, a6, a7);
        if (bias != nullptr) {
            a0 += bias_scale * bf16_to_float(bias[column + 0]);
            a1 += bias_scale * bf16_to_float(bias[column + 1]);
            a2 += bias_scale * bf16_to_float(bias[column + 2]);
            a3 += bias_scale * bf16_to_float(bias[column + 3]);
            a4 += bias_scale * bf16_to_float(bias[column + 4]);
            a5 += bias_scale * bf16_to_float(bias[column + 5]);
            a6 += bias_scale * bf16_to_float(bias[column + 6]);
            a7 += bias_scale * bf16_to_float(bias[column + 7]);
        }
        packet_bf16x8 output;
        output.value[0] = float_to_bf16_rne(a0);
        output.value[1] = float_to_bf16_rne(a1);
        output.value[2] = float_to_bf16_rne(a2);
        output.value[3] = float_to_bf16_rne(a3);
        output.value[4] = float_to_bf16_rne(a4);
        output.value[5] = float_to_bf16_rne(a5);
        output.value[6] = float_to_bf16_rne(a6);
        output.value[7] = float_to_bf16_rne(a7);
        *reinterpret_cast<uint4*>(destination + offset) = output.packet;
    }
}

} // namespace hk_gemm_rs

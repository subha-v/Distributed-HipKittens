// One-process/two-GPU neutral transport diagnostic for methodology Stages 0/1.
//
// This is deliberately not the final rank-per-GPU MORI/IRIS comparison. It
// holds the materialized source, inbox, publication, and dependent digest
// semantics constant across CU push, CU pull, and hipMemcpyPeerAsync.
//
// Bump NEUTRAL_TRANSPORT_SRC_REV on every edit. The build script records both
// this revision and the source hash so a stale executable cannot masquerade as
// a null result.
#define NEUTRAL_TRANSPORT_SRC_REV 3

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cdna4/ops/group/distributed/completion.cuh"
#include "cdna4/ops/group/distributed/packet.cuh"
#include "cdna4/ops/group/distributed/peer.cuh"
#include "cdna4/ops/group/distributed/sync.cuh"
#include "cdna4/types/global/pgl.cuh"

namespace kd = kittens::distributed;

namespace {

constexpr int kWorld = 2;
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 256;
constexpr std::uint64_t kRealtimeHz = 100000000ull;
constexpr std::uint64_t kPoison = 0xA5A5A5A5A5A5A5A5ull;
constexpr std::size_t kVolume = 64ull << 20;
constexpr std::uint64_t kValidSpinLimit = 1ull << 38;
constexpr std::uint64_t kControlSpinLimit = 100000ull;
constexpr const char* kSchema = "exp01.transport-point.v1";

enum class Method { CuPush, CuPull, HostCopy };
enum class Mutation { Valid, Redirect, EarlyPublication, NoPublication };

const char* method_name(Method method) {
    switch (method) {
        case Method::CuPush: return "cu_push";
        case Method::CuPull: return "cu_pull";
        case Method::HostCopy: return "host_copy_path";
    }
    return "invalid";
}

const char* mutation_name(Mutation mutation) {
    switch (mutation) {
        case Mutation::Valid: return "valid";
        case Mutation::Redirect: return "redirected_destination";
        case Mutation::EarlyPublication: return "early_publication";
        case Mutation::NoPublication: return "no_publication";
    }
    return "invalid";
}

__host__ __device__ __forceinline__ std::uint64_t mix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

__host__ __device__ __forceinline__ std::uint64_t payload_value(
        std::uint32_t epoch, int source, int destination,
        std::uint64_t record, std::uint64_t word) {
    std::uint64_t seed = static_cast<std::uint64_t>(epoch) << 32;
    seed ^= static_cast<std::uint64_t>(source + 1) * 0xD6E8FEB86659FD93ull;
    seed ^= static_cast<std::uint64_t>(destination + 3) * 0xA0761D6478BD642Full;
    seed ^= record * 0xE7037ED1A0B428DBull;
    seed ^= word * 0x8EBC6AF09C88C6E3ull;
    std::uint64_t value = mix64(seed);
    if (value == kPoison) value ^= 1ull;
    return value;
}

__host__ __device__ __forceinline__ std::uint64_t digest_xor_word(
        std::uint64_t value, std::uint64_t record, std::uint64_t word) {
    return mix64(value ^ (record * 0x9E3779B97F4A7C15ull) ^
                 (word * 0xD1B54A32D192ED03ull));
}

__host__ __device__ __forceinline__ std::uint64_t digest_odd_constant(
        std::uint64_t record, std::uint64_t word) {
    return mix64((record << 32) ^ word ^ 0x94D049BB133111EBull) | 1ull;
}

__device__ __forceinline__ std::uint64_t realtime_now() {
    std::uint64_t ticks;
    asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)"
                 : "=s"(ticks) :: "memory");
    return ticks;
}

struct Stamp {
    std::uint64_t begin;
    std::uint64_t end;
    std::uint64_t work;
};

struct DeviceDigest {
    std::uint64_t xor_digest;
    std::uint64_t add_digest;
    std::uint64_t poison_survivors;
    std::uint64_t sample_mismatches;
    std::uint64_t first_word;
    std::uint64_t poll_spins;
    std::uint32_t observed_epoch;
    std::uint32_t ready_timeout;
};

struct GenerateParams {
    std::byte* output;
    std::size_t slot_stride;
    std::size_t total_bytes;
    std::size_t record_bytes;
    std::uint32_t slot;
    std::uint32_t epoch;
    int source;
    int destination;
    Stamp* stamps;
};

__global__ __launch_bounds__(kThreads) void generate_kernel(GenerateParams params) {
    const std::uint64_t begin = realtime_now();
    auto* output = reinterpret_cast<std::uint64_t*>(
        params.output + static_cast<std::size_t>(params.slot) * params.slot_stride);
    const std::uint64_t words = params.total_bytes / sizeof(std::uint64_t);
    const std::uint64_t words_per_record = params.record_bytes / sizeof(std::uint64_t);
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < words;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint64_t record = index / words_per_record;
        const std::uint64_t word = index % words_per_record;
        output[index] = payload_value(params.epoch, params.source,
                                      params.destination, record, word);
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    const std::uint64_t end = realtime_now();
    if (threadIdx.x == 0 && params.stamps != nullptr) {
        params.stamps[blockIdx.x] = {begin, end, words};
    }
}

struct TransportParams {
    std::byte* outbound_local;
    const void* outbound_base;
    kittens::peer_bases<std::byte> outbound_peers;
    std::byte* inbox_local;
    const void* inbox_base;
    kittens::peer_bases<std::byte> inbox_peers;
    std::uint32_t* ready_local;
    const void* ready_base;
    kittens::peer_bases<std::byte> ready_peers;
    unsigned int* completion_count;
    Stamp* stamps;
    std::size_t slot_stride;
    std::size_t total_bytes;
    std::size_t record_bytes;
    std::uint32_t records;
    std::uint32_t slot;
    std::uint32_t epoch;
    int me;
    int peer;
    int grid_blocks;
    int pull;
    int publish;
    int redirect;
};

__global__ __launch_bounds__(kThreads) void transport_kernel(TransportParams params) {
    const std::uint64_t begin = realtime_now();
    std::uint64_t copied = 0;
    const std::size_t slot_offset =
        static_cast<std::size_t>(params.slot) * params.slot_stride;

    for (std::uint64_t record = blockIdx.x; record < params.records;
         record += gridDim.x) {
        const std::size_t payload_offset =
            static_cast<std::size_t>(record) * params.record_bytes;
        if (payload_offset >= params.total_bytes) break;
        const std::size_t remaining = params.total_bytes - payload_offset;
        const std::size_t bytes =
            params.record_bytes < remaining ? params.record_bytes : remaining;
        std::size_t inbox_offset = slot_offset + payload_offset;
        if (params.redirect && record == 0) {
            // The allocation carries one guard record after the semantic
            // payload. Redirecting record zero there leaves the true record
            // poisoned and deterministically flips digest/sample/poison gates.
            inbox_offset = slot_offset + params.total_bytes;
        }

        if (!params.pull) {
            const void* source =
                params.outbound_local + slot_offset + payload_offset;
            std::byte* local_projection = params.inbox_local + inbox_offset;
            void* destination = kd::translate_peer<kWorld>(
                local_projection, params.inbox_base, params.inbox_peers,
                params.peer);
            kd::store_peer_packets(destination, source, bytes, threadIdx.x,
                                   blockDim.x);
        } else {
            std::byte* local_projection =
                params.outbound_local + slot_offset + payload_offset;
            const void* source = kd::translate_peer<kWorld>(
                local_projection, params.outbound_base, params.outbound_peers,
                params.peer);
            void* destination = params.inbox_local + inbox_offset;
            kd::load_peer_packets(destination, source, bytes, threadIdx.x,
                                  blockDim.x);
        }
        copied += bytes;
    }

    // packet.cuh supplies the legal payload operation for both directions.
    // A grid-wide completion primitive is not available: each CTA establishes
    // its release, then a local completion atomic elects the last CTA. This
    // benchmark-local last-CTA reduction is intentionally not promoted into a
    // distributed header before the diagnostic proves useful.
    kd::producer_drain_release<kd::memory_scope::system>();
    if (threadIdx.x == 0) {
        const unsigned int prior = atomicAdd(params.completion_count, 1u);
        if (prior + 1u == static_cast<unsigned int>(params.grid_blocks) &&
            params.publish) {
            kd::thread_release<kd::memory_scope::system>();
            std::uint32_t* ready = params.ready_local + params.slot;
            if (!params.pull) {
                ready = kd::translate_peer<kWorld>(
                    ready, params.ready_base, params.ready_peers, params.peer);
            }
            kd::publish_epoch_relaxed<kd::memory_scope::system>(
                ready, params.epoch);
        }
    }
    const std::uint64_t end = realtime_now();
    if (threadIdx.x == 0 && params.stamps != nullptr) {
        params.stamps[blockIdx.x] = {begin, end, copied};
    }
}

struct DigestParams {
    const std::byte* input;
    std::size_t slot_stride;
    std::size_t total_bytes;
    std::size_t record_bytes;
    std::uint32_t slot;
    std::uint32_t epoch;
    int source;
    int destination;
    int wait_for_ready;
    const std::uint32_t* ready;
    std::uint32_t* consumed;
    std::uint64_t spin_limit;
    DeviceDigest* output;
    Stamp* stamp;
};

__global__ __launch_bounds__(kThreads) void digest_kernel(DigestParams params) {
    __shared__ std::uint64_t xor_parts[kThreads];
    __shared__ std::uint64_t add_parts[kThreads];
    __shared__ std::uint64_t poison_parts[kThreads];
    __shared__ std::uint64_t sample_parts[kThreads];
    __shared__ std::uint64_t poll_spins;
    __shared__ std::uint32_t observed_epoch;
    __shared__ int ready;

    const std::uint64_t begin = realtime_now();
    if (threadIdx.x == 0) {
        ready = 1;
        poll_spins = 0;
        observed_epoch = params.epoch;
        if (params.wait_for_ready) {
            kd::wait_result result;
            kd::init_wait_result(result);
            kd::bounded_poll_relaxed_into<kd::memory_scope::system>(
                params.ready + params.slot, params.epoch, params.spin_limit,
                result);
            ready = result.ready ? 1 : 0;
            poll_spins = result.spins;
            observed_epoch = result.observed;
        }
    }
    __syncthreads();
    if (ready && params.wait_for_ready) {
        kd::cta_acquire<kd::memory_scope::system>();
    }

    std::uint64_t xor_value = 0;
    std::uint64_t add_value = 0;
    std::uint64_t poison = 0;
    std::uint64_t sample_bad = 0;
    const auto* input = reinterpret_cast<const std::uint64_t*>(
        params.input + static_cast<std::size_t>(params.slot) * params.slot_stride);
    const std::uint64_t words = params.total_bytes / sizeof(std::uint64_t);
    const std::uint64_t words_per_record = params.record_bytes / sizeof(std::uint64_t);
    const std::uint64_t pseudo =
        words == 0 ? 0 : mix64(params.epoch) % words;

    if (ready) {
        for (std::uint64_t index = threadIdx.x; index < words;
             index += blockDim.x) {
            const std::uint64_t record = index / words_per_record;
            const std::uint64_t word = index % words_per_record;
            const std::uint64_t value = input[index];
            xor_value ^= digest_xor_word(value, record, word);
            add_value += value * digest_odd_constant(record, word);
            poison += value == kPoison ? 1ull : 0ull;
            if (index == 0 || index == words / 2 || index + 1 == words ||
                index == pseudo) {
                const std::uint64_t expected =
                    payload_value(params.epoch, params.source,
                                  params.destination, record, word);
                sample_bad += value == expected ? 0ull : 1ull;
            }
        }
    }
    xor_parts[threadIdx.x] = xor_value;
    add_parts[threadIdx.x] = add_value;
    poison_parts[threadIdx.x] = poison;
    sample_parts[threadIdx.x] = sample_bad;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            xor_parts[threadIdx.x] ^= xor_parts[threadIdx.x + stride];
            add_parts[threadIdx.x] += add_parts[threadIdx.x + stride];
            poison_parts[threadIdx.x] += poison_parts[threadIdx.x + stride];
            sample_parts[threadIdx.x] += sample_parts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (ready && params.wait_for_ready) kd::consumer_drain();
    const std::uint64_t end = realtime_now();
    if (threadIdx.x == 0) {
        DeviceDigest result{};
        result.xor_digest = xor_parts[0];
        result.add_digest = add_parts[0];
        result.poison_survivors = poison_parts[0];
        result.sample_mismatches = sample_parts[0];
        result.first_word = ready && words ? input[0] : 0ull;
        result.poll_spins = poll_spins;
        result.observed_epoch = observed_epoch;
        result.ready_timeout = ready ? 0u : 1u;
        *params.output = result;
        if (ready && params.wait_for_ready && params.consumed != nullptr) {
            kd::publish_epoch_relaxed<kd::memory_scope::system>(
                params.consumed + params.slot, params.epoch);
        }
        if (params.stamp != nullptr) {
            *params.stamp = {begin, end, words};
        }
    }
}

__global__ void publish_kernel(std::uint32_t* ready, std::uint32_t slot,
                               std::uint32_t epoch) {
    kd::release_and_publish<kd::memory_scope::system>(ready + slot, epoch);
}

__global__ void stamp_begin_kernel(Stamp* stamp) {
    if (threadIdx.x == 0) stamp->begin = realtime_now();
}

__global__ void stamp_end_kernel(Stamp* stamp, std::uint64_t work) {
    if (threadIdx.x == 0) {
        stamp->end = realtime_now();
        stamp->work = work;
    }
}

#define HC(call)                                                                \
    do {                                                                        \
        const hipError_t error_ = (call);                                        \
        if (error_ != hipSuccess) {                                              \
            std::fprintf(stderr, "[neutral] %s:%d %s -> %s\n", __FILE__,        \
                         __LINE__, #call, hipGetErrorString(error_));             \
            std::exit(2);                                                        \
        }                                                                       \
    } while (0)

struct Config {
    std::string output = "transport_diagnostic.jsonl";
    std::vector<std::size_t> sizes = {65536};
    std::vector<Method> methods = {
        Method::CuPush, Method::CuPull, Method::HostCopy};
    std::vector<int> depths = {1, 2};
    int rotations = 5;
    int warmup = 1;
    int soak_epochs = 0;
    double max_seconds = 0.0;
    bool list_only = false;
};

Config config;
std::vector<std::string> command_line;

struct Device {
    std::byte* outbound = nullptr;
    std::byte* inbox = nullptr;
    std::uint32_t* ready = nullptr;
    std::uint32_t* consumed = nullptr;
    unsigned int* completion_count = nullptr;
    DeviceDigest* expected = nullptr;
    DeviceDigest* actual = nullptr;
    Stamp* producer_stamps = nullptr;
    Stamp* transport_stamps = nullptr;
    Stamp* consumer_stamp = nullptr;
    hipStream_t producer_stream = nullptr;
    hipStream_t transport_stream = nullptr;
    hipStream_t consumer_stream = nullptr;
    hipEvent_t producer_begin = nullptr;
    hipEvent_t producer_end = nullptr;
    hipEvent_t transport_begin = nullptr;
    hipEvent_t transport_end = nullptr;
    hipEvent_t wall_begin = nullptr;
    hipEvent_t wall_end = nullptr;
    std::string name;
    std::string pci_bus_id;
};

struct Buffers {
    std::array<Device, kWorld> devices;
    std::size_t record_bytes = 0;
    std::size_t records = 0;
    std::size_t tail_record_bytes = 0;
    std::size_t slot_stride = 0;
    int depth = 1;
    int generate_blocks = kMaxBlocks;
    int transport_blocks = 1;
    std::array<std::array<std::uint64_t, 2>, kWorld> last_first_word{};
    std::array<std::array<std::uint32_t, 2>, kWorld> last_consumed{};
    std::array<std::array<std::uint32_t, 2>, kWorld> last_valid_epoch{};
};

kittens::peer_bases<std::byte> byte_bases(std::byte* rank0, std::byte* rank1) {
    kittens::peer_bases<std::byte> bases{};
    bases.rank0 = rank0;
    bases.rank1 = rank1;
    return bases;
}

Buffers allocate_buffers(std::size_t record_bytes, int depth) {
    Buffers buffers;
    buffers.record_bytes = record_bytes;
    buffers.records = (kVolume + record_bytes - 1) / record_bytes;
    buffers.tail_record_bytes = kVolume % record_bytes;
    if (buffers.tail_record_bytes == 0) buffers.tail_record_bytes = record_bytes;
    buffers.slot_stride = kVolume + record_bytes;
    buffers.depth = depth;
    buffers.transport_blocks =
        std::max(1, std::min(kMaxBlocks, static_cast<int>(buffers.records)));

    for (int device = 0; device < kWorld; ++device) {
        HC(hipSetDevice(device));
        hipDeviceProp_t properties{};
        HC(hipGetDeviceProperties(&properties, device));
        buffers.devices[device].name = properties.name;
        char pci[32] = {};
        HC(hipDeviceGetPCIBusId(pci, sizeof(pci), device));
        buffers.devices[device].pci_bus_id = pci;
        for (int peer = 0; peer < kWorld; ++peer) {
            if (peer == device) continue;
            int can_access = 0;
            HC(hipDeviceCanAccessPeer(&can_access, device, peer));
            if (!can_access) {
                std::fprintf(stderr, "[neutral] device %d cannot access peer %d\n",
                             device, peer);
                std::exit(3);
            }
            const hipError_t enabled = hipDeviceEnablePeerAccess(peer, 0);
            if (enabled != hipSuccess && enabled != hipErrorPeerAccessAlreadyEnabled) {
                HC(enabled);
            }
            if (enabled == hipErrorPeerAccessAlreadyEnabled) {
                (void)hipGetLastError();
            }
        }

        Device& dev = buffers.devices[device];
        const std::size_t allocation_bytes =
            static_cast<std::size_t>(depth) * buffers.slot_stride;
        HC(hipMalloc(&dev.outbound, allocation_bytes));
        HC(hipMalloc(&dev.inbox, allocation_bytes));
        HC(hipMalloc(&dev.ready, static_cast<std::size_t>(depth) *
                                    sizeof(std::uint32_t)));
        HC(hipMalloc(&dev.consumed, static_cast<std::size_t>(depth) *
                                       sizeof(std::uint32_t)));
        HC(hipMalloc(&dev.completion_count, sizeof(unsigned int)));
        HC(hipMalloc(&dev.expected, static_cast<std::size_t>(depth) *
                                    sizeof(DeviceDigest)));
        HC(hipMalloc(&dev.actual, static_cast<std::size_t>(depth) *
                                  sizeof(DeviceDigest)));
        HC(hipMalloc(&dev.producer_stamps, static_cast<std::size_t>(kMaxBlocks) *
                                            sizeof(Stamp)));
        HC(hipMalloc(&dev.transport_stamps, static_cast<std::size_t>(kMaxBlocks) *
                                             sizeof(Stamp)));
        HC(hipMalloc(&dev.consumer_stamp, sizeof(Stamp)));
        HC(hipStreamCreateWithFlags(&dev.producer_stream, hipStreamNonBlocking));
        HC(hipStreamCreateWithFlags(&dev.transport_stream, hipStreamNonBlocking));
        HC(hipStreamCreateWithFlags(&dev.consumer_stream, hipStreamNonBlocking));
        HC(hipEventCreate(&dev.producer_begin));
        HC(hipEventCreate(&dev.producer_end));
        HC(hipEventCreate(&dev.transport_begin));
        HC(hipEventCreate(&dev.transport_end));
        HC(hipEventCreate(&dev.wall_begin));
        HC(hipEventCreate(&dev.wall_end));
        HC(hipMemset(dev.ready, 0,
                     static_cast<std::size_t>(depth) * sizeof(std::uint32_t)));
        HC(hipMemset(dev.consumed, 0,
                     static_cast<std::size_t>(depth) * sizeof(std::uint32_t)));
    }
    return buffers;
}

void free_buffers(Buffers& buffers) {
    for (int device = 0; device < kWorld; ++device) {
        HC(hipSetDevice(device));
        Device& dev = buffers.devices[device];
        HC(hipStreamSynchronize(dev.producer_stream));
        HC(hipStreamSynchronize(dev.transport_stream));
        HC(hipStreamSynchronize(dev.consumer_stream));
        HC(hipEventDestroy(dev.producer_begin));
        HC(hipEventDestroy(dev.producer_end));
        HC(hipEventDestroy(dev.transport_begin));
        HC(hipEventDestroy(dev.transport_end));
        HC(hipEventDestroy(dev.wall_begin));
        HC(hipEventDestroy(dev.wall_end));
        HC(hipStreamDestroy(dev.producer_stream));
        HC(hipStreamDestroy(dev.transport_stream));
        HC(hipStreamDestroy(dev.consumer_stream));
        HC(hipFree(dev.outbound));
        HC(hipFree(dev.inbox));
        HC(hipFree(dev.ready));
        HC(hipFree(dev.consumed));
        HC(hipFree(dev.completion_count));
        HC(hipFree(dev.expected));
        HC(hipFree(dev.actual));
        HC(hipFree(dev.producer_stamps));
        HC(hipFree(dev.transport_stamps));
        HC(hipFree(dev.consumer_stamp));
    }
}

double event_us(hipEvent_t begin, hipEvent_t end) {
    float milliseconds = 0.0f;
    HC(hipEventElapsedTime(&milliseconds, begin, end));
    return static_cast<double>(milliseconds) * 1000.0;
}

double stamp_span_us(int device, Stamp* stamps, int count) {
    std::vector<Stamp> host(static_cast<std::size_t>(count));
    HC(hipSetDevice(device));
    HC(hipMemcpy(host.data(), stamps, host.size() * sizeof(Stamp),
                 hipMemcpyDeviceToHost));
    std::uint64_t begin = ~0ull;
    std::uint64_t end = 0;
    for (const Stamp& stamp : host) {
        if (stamp.end <= stamp.begin || stamp.begin == 0) continue;
        begin = std::min(begin, stamp.begin);
        end = std::max(end, stamp.end);
    }
    if (end <= begin || begin == ~0ull) return 0.0;
    return static_cast<double>(end - begin) * 1.0e6 /
           static_cast<double>(kRealtimeHz);
}

struct DirectionSample {
    double producer_us = 0;
    double transport_us = 0;
    double consumer_us = 0;
    double producer_event_us = 0;
    double transport_event_us = 0;
    double local_wall_event_us = 0;
    double enqueue_us = 0;
    std::uint64_t poll_spins = 0;
};

struct EpochResult {
    double host_global_us = 0;
    std::array<DirectionSample, kWorld> directions;
    std::array<DeviceDigest, kWorld> expected{};
    std::array<DeviceDigest, kWorld> actual{};
    bool digest_pass = false;
    bool samples_pass = false;
    bool generation_pass = false;
    bool output_overwritten = false;
    std::uint64_t poison_survivors = 0;
    unsigned int all_rank_errors = 0;
};

void reset_epoch_state(Buffers& buffers, std::uint32_t slot) {
    for (int device = 0; device < kWorld; ++device) {
        HC(hipSetDevice(device));
        Device& dev = buffers.devices[device];
        const std::size_t slot_offset =
            static_cast<std::size_t>(slot) * buffers.slot_stride;
        HC(hipMemsetD8Async(dev.inbox + slot_offset, 0xA5,
                            buffers.slot_stride, dev.consumer_stream));
        HC(hipMemsetD8Async(
            reinterpret_cast<std::byte*>(dev.actual) +
                static_cast<std::size_t>(slot) * sizeof(DeviceDigest),
            0xA5, sizeof(DeviceDigest), dev.consumer_stream));
        HC(hipMemsetD8Async(
            reinterpret_cast<std::byte*>(dev.expected) +
                static_cast<std::size_t>(slot) * sizeof(DeviceDigest),
            0xA5, sizeof(DeviceDigest), dev.producer_stream));
        HC(hipMemsetAsync(dev.ready + slot, 0, sizeof(std::uint32_t),
                          dev.consumer_stream));
        HC(hipMemsetAsync(dev.consumed + slot, 0, sizeof(std::uint32_t),
                          dev.consumer_stream));
        HC(hipMemsetAsync(dev.completion_count, 0, sizeof(unsigned int),
                          dev.transport_stream));
        HC(hipMemsetAsync(dev.producer_stamps, 0,
                          static_cast<std::size_t>(kMaxBlocks) * sizeof(Stamp),
                          dev.producer_stream));
        HC(hipMemsetAsync(dev.transport_stamps, 0,
                          static_cast<std::size_t>(kMaxBlocks) * sizeof(Stamp),
                          dev.transport_stream));
        HC(hipMemsetAsync(dev.consumer_stamp, 0, sizeof(Stamp),
                          dev.consumer_stream));
    }
}

void generate_and_expected(Buffers& buffers, std::uint32_t slot,
                           std::uint32_t epoch) {
    for (int source = 0; source < kWorld; ++source) {
        const int destination = 1 - source;
        Device& dev = buffers.devices[source];
        HC(hipSetDevice(source));
        GenerateParams generate{
            dev.outbound,
            buffers.slot_stride,
            kVolume,
            buffers.record_bytes,
            slot,
            epoch,
            source,
            destination,
            dev.producer_stamps,
        };
        HC(hipEventRecord(dev.producer_begin, dev.producer_stream));
        hipLaunchKernelGGL(generate_kernel, dim3(buffers.generate_blocks),
                           dim3(kThreads), 0, dev.producer_stream, generate);
        HC(hipEventRecord(dev.producer_end, dev.producer_stream));
        DigestParams digest{
            dev.outbound,
            buffers.slot_stride,
            kVolume,
            buffers.record_bytes,
            slot,
            epoch,
            source,
            destination,
            0,
            nullptr,
            nullptr,
            0,
            dev.expected + slot,
            nullptr,
        };
        hipLaunchKernelGGL(digest_kernel, dim3(1), dim3(kThreads), 0,
                           dev.producer_stream, digest);
    }
    for (int source = 0; source < kWorld; ++source) {
        HC(hipSetDevice(source));
        HC(hipStreamSynchronize(buffers.devices[source].producer_stream));
        HC(hipGetLastError());
    }
}

void launch_consumers(Buffers& buffers, std::uint32_t slot,
                      std::uint32_t epoch, std::uint64_t spin_limit) {
    for (int destination = 0; destination < kWorld; ++destination) {
        const int source = 1 - destination;
        Device& dev = buffers.devices[destination];
        HC(hipSetDevice(destination));
        DigestParams digest{
            dev.inbox,
            buffers.slot_stride,
            kVolume,
            buffers.record_bytes,
            slot,
            epoch,
            source,
            destination,
            1,
            dev.ready,
            dev.consumed,
            spin_limit,
            dev.actual + slot,
            dev.consumer_stamp,
        };
        HC(hipEventRecord(dev.wall_begin, dev.consumer_stream));
        hipLaunchKernelGGL(digest_kernel, dim3(1), dim3(kThreads), 0,
                           dev.consumer_stream, digest);
        HC(hipEventRecord(dev.wall_end, dev.consumer_stream));
    }
}

TransportParams transport_params(Buffers& buffers, Method method, int device,
                                 std::uint32_t slot, std::uint32_t epoch,
                                 Mutation mutation) {
    Device& dev = buffers.devices[device];
    TransportParams params{};
    params.outbound_local = dev.outbound;
    params.outbound_base = dev.outbound;
    params.outbound_peers =
        byte_bases(buffers.devices[0].outbound, buffers.devices[1].outbound);
    params.inbox_local = dev.inbox;
    params.inbox_base = dev.inbox;
    params.inbox_peers =
        byte_bases(buffers.devices[0].inbox, buffers.devices[1].inbox);
    params.ready_local = dev.ready;
    params.ready_base = dev.ready;
    params.ready_peers = byte_bases(
        reinterpret_cast<std::byte*>(buffers.devices[0].ready),
        reinterpret_cast<std::byte*>(buffers.devices[1].ready));
    params.completion_count = dev.completion_count;
    params.stamps = dev.transport_stamps;
    params.slot_stride = buffers.slot_stride;
    params.total_bytes = kVolume;
    params.record_bytes = buffers.record_bytes;
    params.records = static_cast<std::uint32_t>(buffers.records);
    params.slot = slot;
    params.epoch = epoch;
    params.me = device;
    params.peer = 1 - device;
    params.grid_blocks = buffers.transport_blocks;
    params.pull = method == Method::CuPull ? 1 : 0;
    params.publish = mutation == Mutation::NoPublication ? 0 : 1;
    params.redirect = mutation == Mutation::Redirect ? 1 : 0;
    return params;
}

void launch_cu_transports(Buffers& buffers, Method method, std::uint32_t slot,
                          std::uint32_t epoch, Mutation mutation) {
    const int first = epoch & 1u;
    for (int order = 0; order < kWorld; ++order) {
        const int device = (first + order) % kWorld;
        Device& dev = buffers.devices[device];
        HC(hipSetDevice(device));
        TransportParams params =
            transport_params(buffers, method, device, slot, epoch, mutation);
        HC(hipEventRecord(dev.transport_begin, dev.transport_stream));
        hipLaunchKernelGGL(transport_kernel, dim3(buffers.transport_blocks),
                           dim3(kThreads), 0, dev.transport_stream, params);
        HC(hipEventRecord(dev.transport_end, dev.transport_stream));
    }
}

std::array<double, kWorld> launch_host_copies(
        Buffers& buffers, std::uint32_t slot, std::uint32_t epoch,
        Mutation mutation) {
    std::array<double, kWorld> enqueue_us{};
    const int first = epoch & 1u;
    for (int order = 0; order < kWorld; ++order) {
        const int destination = (first + order) % kWorld;
        const int source = 1 - destination;
        Device& dev = buffers.devices[destination];
        HC(hipSetDevice(destination));
        const std::size_t slot_offset =
            static_cast<std::size_t>(slot) * buffers.slot_stride;
        HC(hipEventRecord(dev.transport_begin, dev.transport_stream));
        hipLaunchKernelGGL(stamp_begin_kernel, dim3(1), dim3(1), 0,
                           dev.transport_stream, dev.transport_stamps);
        const auto begin = std::chrono::steady_clock::now();
        for (std::size_t record = 0; record < buffers.records; ++record) {
            const std::size_t payload_offset = record * buffers.record_bytes;
            if (payload_offset >= kVolume) break;
            const std::size_t bytes =
                std::min(buffers.record_bytes, kVolume - payload_offset);
            std::size_t destination_offset = slot_offset + payload_offset;
            if (mutation == Mutation::Redirect && record == 0) {
                destination_offset = slot_offset + kVolume;
            }
            HC(hipMemcpyPeerAsync(
                dev.inbox + destination_offset, destination,
                buffers.devices[source].outbound + slot_offset + payload_offset,
                source, bytes, dev.transport_stream));
        }
        const auto end = std::chrono::steady_clock::now();
        enqueue_us[destination] =
            std::chrono::duration<double, std::micro>(end - begin).count();
        hipLaunchKernelGGL(stamp_end_kernel, dim3(1), dim3(1), 0,
                           dev.transport_stream, dev.transport_stamps,
                           static_cast<std::uint64_t>(kVolume));
        if (mutation != Mutation::NoPublication) {
            hipLaunchKernelGGL(publish_kernel, dim3(1), dim3(64), 0,
                               dev.transport_stream, dev.ready, slot, epoch);
        }
        HC(hipEventRecord(dev.transport_end, dev.transport_stream));
    }
    return enqueue_us;
}

void publish_early(Buffers& buffers, std::uint32_t slot, std::uint32_t epoch) {
    for (int destination = 0; destination < kWorld; ++destination) {
        Device& dev = buffers.devices[destination];
        HC(hipSetDevice(destination));
        hipLaunchKernelGGL(publish_kernel, dim3(1), dim3(64), 0,
                           dev.transport_stream, dev.ready, slot, epoch);
        HC(hipStreamSynchronize(dev.transport_stream));
    }
}

EpochResult execute_epoch(Buffers& buffers, Method method, Mutation mutation,
                          std::uint32_t epoch, bool collect_timing) {
    const std::uint32_t slot = epoch % static_cast<std::uint32_t>(buffers.depth);
    bool credits_ok = true;
    if (mutation == Mutation::Valid) {
        for (int device = 0; device < kWorld; ++device) {
            const std::uint32_t previous =
                buffers.last_valid_epoch[device][slot];
            if (previous != 0 &&
                buffers.last_consumed[device][slot] != previous) {
                credits_ok = false;
            }
        }
    }
    reset_epoch_state(buffers, slot);
    for (int device = 0; device < kWorld; ++device) {
        HC(hipSetDevice(device));
        HC(hipStreamSynchronize(buffers.devices[device].consumer_stream));
        HC(hipStreamSynchronize(buffers.devices[device].transport_stream));
    }
    generate_and_expected(buffers, slot, epoch);

    EpochResult result;

    const auto host_begin = std::chrono::steady_clock::now();
    std::array<double, kWorld> enqueue_us{};
    if (mutation == Mutation::EarlyPublication) {
        publish_early(buffers, slot, epoch);
        launch_consumers(buffers, slot, epoch, kControlSpinLimit);
    } else {
        launch_consumers(
            buffers, slot, epoch,
            mutation == Mutation::NoPublication ? kControlSpinLimit
                                                : kValidSpinLimit);
        if (method == Method::HostCopy) {
            enqueue_us =
                launch_host_copies(buffers, slot, epoch, mutation);
        } else {
            launch_cu_transports(buffers, method, slot, epoch, mutation);
        }
    }

    for (int device = 0; device < kWorld; ++device) {
        HC(hipSetDevice(device));
        HC(hipEventSynchronize(buffers.devices[device].wall_end));
        if (mutation != Mutation::EarlyPublication) {
            HC(hipEventSynchronize(buffers.devices[device].transport_end));
        }
        HC(hipGetLastError());
    }
    const auto host_end = std::chrono::steady_clock::now();
    result.host_global_us =
        std::chrono::duration<double, std::micro>(host_end - host_begin).count();

    for (int device = 0; device < kWorld; ++device) {
        HC(hipSetDevice(device));
        Device& dev = buffers.devices[device];
        HC(hipMemcpy(&result.expected[device], dev.expected + slot,
                     sizeof(DeviceDigest), hipMemcpyDeviceToHost));
        HC(hipMemcpy(&result.actual[device], dev.actual + slot,
                     sizeof(DeviceDigest), hipMemcpyDeviceToHost));
        std::uint32_t consumed = 0;
        HC(hipMemcpy(&consumed, dev.consumed + slot, sizeof(consumed),
                     hipMemcpyDeviceToHost));
        if (mutation == Mutation::Valid) {
            buffers.last_consumed[device][slot] = consumed;
            buffers.last_valid_epoch[device][slot] = epoch;
        }

        const int source = 1 - device;
        const int transport_device =
            method == Method::CuPush ? source : device;
        DirectionSample& direction = result.directions[device];
        direction.producer_us = stamp_span_us(
            source, buffers.devices[source].producer_stamps,
            buffers.generate_blocks);
        direction.producer_event_us = event_us(
            buffers.devices[source].producer_begin,
            buffers.devices[source].producer_end);
        direction.consumer_us =
            stamp_span_us(device, dev.consumer_stamp, 1);
        direction.poll_spins = result.actual[device].poll_spins;
        direction.enqueue_us = enqueue_us[device];
        direction.local_wall_event_us = event_us(dev.wall_begin, dev.wall_end);
        if (mutation != Mutation::EarlyPublication) {
            direction.transport_us = stamp_span_us(
                transport_device,
                buffers.devices[transport_device].transport_stamps,
                method == Method::HostCopy ? 1 : buffers.transport_blocks);
            direction.transport_event_us = event_us(
                buffers.devices[transport_device].transport_begin,
                buffers.devices[transport_device].transport_end);
        }
    }

    result.digest_pass = true;
    result.samples_pass = true;
    result.generation_pass = credits_ok;
    result.output_overwritten = true;
    for (int destination = 0; destination < kWorld; ++destination) {
        const DeviceDigest& expected = result.expected[1 - destination];
        const DeviceDigest& actual = result.actual[destination];
        const bool digest_ok =
            actual.ready_timeout == 0 &&
            expected.xor_digest == actual.xor_digest &&
            expected.add_digest == actual.add_digest;
        result.digest_pass &= digest_ok;
        result.samples_pass &=
            actual.ready_timeout == 0 && actual.sample_mismatches == 0;
        result.poison_survivors += actual.poison_survivors;
        result.output_overwritten &=
            actual.xor_digest != kPoison || actual.add_digest != kPoison ||
            actual.ready_timeout != static_cast<std::uint32_t>(kPoison);
        const std::uint64_t previous =
            buffers.last_first_word[destination][slot];
        if (previous != 0 && previous == actual.first_word &&
            actual.ready_timeout == 0) {
            result.generation_pass = false;
        }
        if (mutation == Mutation::Valid && actual.ready_timeout == 0) {
            buffers.last_first_word[destination][slot] = actual.first_word;
        }
        if (!digest_ok || actual.sample_mismatches != 0 ||
            actual.poison_survivors != 0 || actual.ready_timeout != 0) {
            ++result.all_rank_errors;
        }
    }
    if (!collect_timing) result.host_global_us = 0.0;
    return result;
}

bool valid_epoch_pass(const EpochResult& result) {
    return result.digest_pass && result.samples_pass &&
           result.poison_survivors == 0 && result.generation_pass &&
           result.output_overwritten && result.all_rank_errors == 0;
}

struct ControlResult {
    bool applicable = true;
    bool detected = false;
    std::string detector;
};

struct GateResult {
    bool preflight_pass = false;
    bool digest_pass = false;
    bool samples_pass = false;
    bool generation_pass = false;
    bool output_overwritten = false;
    std::uint64_t poison_survivors = 0;
    unsigned int all_rank_errors = 0;
    std::map<std::string, ControlResult> controls;
    int soak_epochs = 0;
    bool soak_pass = false;
};

GateResult run_gates(Buffers& buffers, Method method, std::uint32_t* epoch) {
    GateResult gate;
    const EpochResult preflight =
        execute_epoch(buffers, method, Mutation::Valid, (*epoch)++, false);
    gate.preflight_pass = valid_epoch_pass(preflight);
    gate.digest_pass = preflight.digest_pass;
    gate.samples_pass = preflight.samples_pass;
    gate.generation_pass = preflight.generation_pass;
    gate.output_overwritten = preflight.output_overwritten;
    gate.poison_survivors = preflight.poison_survivors;
    gate.all_rank_errors = preflight.all_rank_errors;

    for (Mutation mutation : {Mutation::Redirect, Mutation::EarlyPublication,
                              Mutation::NoPublication}) {
        const EpochResult control =
            execute_epoch(buffers, method, mutation, (*epoch)++, false);
        ControlResult result;
        if (mutation == Mutation::NoPublication) {
            result.detector = "bounded_poll_timeout";
            result.detected = control.actual[0].ready_timeout != 0 &&
                              control.actual[1].ready_timeout != 0;
        } else {
            result.detector = "digest_sample_poison";
            result.detected =
                !control.digest_pass || !control.samples_pass ||
                control.poison_survivors != 0;
        }
        gate.controls[mutation_name(mutation)] = result;
    }

    gate.soak_epochs = config.soak_epochs;
    gate.soak_pass = config.soak_epochs == 600;
    for (int iteration = 0; iteration < config.soak_epochs; ++iteration) {
        const EpochResult soak =
            execute_epoch(buffers, method, Mutation::Valid, (*epoch)++, false);
        if (!valid_epoch_pass(soak)) {
            gate.soak_pass = false;
            break;
        }
    }
    return gate;
}

template<typename T>
double quantile(std::vector<T> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = static_cast<double>(values.size() - 1) * q;
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return static_cast<double>(values[lower]) * (1.0 - weight) +
           static_cast<double>(values[upper]) * weight;
}

struct DirectionSeries {
    std::vector<double> producer_us;
    std::vector<double> transport_us;
    std::vector<double> consumer_us;
    std::vector<double> producer_event_us;
    std::vector<double> transport_event_us;
    std::vector<double> local_wall_event_us;
    std::vector<double> enqueue_us;
    std::vector<std::uint64_t> poll_spins;
};

struct PointResult {
    Method method;
    GateResult gate;
    std::vector<double> global_us;
    std::array<DirectionSeries, kWorld> directions;
    bool timed_correctness_pass = true;
    bool timer_agreement_pass = false;
    std::vector<double> timer_delta_us;
};

void append_epoch(PointResult& point, const EpochResult& epoch) {
    point.global_us.push_back(epoch.host_global_us);
    point.timed_correctness_pass &= valid_epoch_pass(epoch);
    double local_wall_max = 0;
    for (int destination = 0; destination < kWorld; ++destination) {
        const DirectionSample& sample = epoch.directions[destination];
        DirectionSeries& series = point.directions[destination];
        series.producer_us.push_back(sample.producer_us);
        series.transport_us.push_back(sample.transport_us);
        series.consumer_us.push_back(sample.consumer_us);
        series.producer_event_us.push_back(sample.producer_event_us);
        series.transport_event_us.push_back(sample.transport_event_us);
        series.local_wall_event_us.push_back(sample.local_wall_event_us);
        series.enqueue_us.push_back(sample.enqueue_us);
        series.poll_spins.push_back(sample.poll_spins);
        local_wall_max = std::max(local_wall_max, sample.local_wall_event_us);
    }
    point.timer_delta_us.push_back(epoch.host_global_us - local_wall_max);
}

bool controls_pass(const GateResult& gate) {
    for (const auto& entry : gate.controls) {
        if (!entry.second.applicable || !entry.second.detected) return false;
    }
    return gate.controls.size() == 3;
}

std::string point_id(Method method, const Buffers& buffers) {
    std::ostringstream stream;
    stream << "diagnostic|" << method_name(method) << "|rb"
           << buffers.record_bytes << "|bytes" << kVolume << "|"
           << (buffers.depth == 1 ? "one_epoch" : "ping_pong")
           << "|soak" << config.soak_epochs << "|rev"
           << NEUTRAL_TRANSPORT_SRC_REV;
    return stream.str();
}

void emit_number_series(FILE* output, const char* name,
                        const std::vector<double>& values) {
    std::fprintf(output, "\"%s\":[", name);
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::fprintf(output, "%s%.4f", index ? "," : "", values[index]);
    }
    std::fprintf(output, "]");
}

void emit_u64_series(FILE* output, const char* name,
                     const std::vector<std::uint64_t>& values) {
    std::fprintf(output, "\"%s\":[", name);
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::fprintf(output, "%s%llu", index ? "," : "",
                     static_cast<unsigned long long>(values[index]));
    }
    std::fprintf(output, "]");
}

void emit_span(FILE* output, const char* name,
               const std::vector<double>& values) {
    std::fprintf(output, "\"%s\":{\"p50\":%.4f,\"p95\":%.4f,\"values\":[",
                 name, quantile(values, 0.50), quantile(values, 0.95));
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::fprintf(output, "%s%.4f", index ? "," : "", values[index]);
    }
    std::fprintf(output, "]}");
}

std::string joined_command() {
    std::ostringstream stream;
    for (std::size_t index = 0; index < command_line.size(); ++index) {
        if (index) stream << ' ';
        stream << command_line[index];
    }
    return stream.str();
}

void emit_point(FILE* output, const Buffers& buffers, PointResult& point) {
    const double p50 = quantile(point.global_us, 0.50);
    const double p95 = quantile(point.global_us, 0.95);
    const double event_p50 = std::max(
        quantile(point.directions[0].local_wall_event_us, 0.50),
        quantile(point.directions[1].local_wall_event_us, 0.50));
    const double timer_delta = std::abs(p50 - event_p50);
    const double timer_tolerance = std::max(2.0, 0.05 * p50);
    point.timer_agreement_pass = timer_delta <= timer_tolerance;

    bool gate_without_timer =
        point.gate.preflight_pass && controls_pass(point.gate) &&
        point.timed_correctness_pass;
    const char* status = "blocked";
    const char* blocker = "strict_600_epoch_soak_not_requested";
    if (!gate_without_timer) {
        status = "incorrect";
        blocker = "correctness_or_negative_control_failed";
    } else if (config.soak_epochs == 600 && !point.gate.soak_pass) {
        status = "incorrect";
        blocker = "strict_600_epoch_soak_failed";
    } else if (!point.timer_agreement_pass) {
        status = "blocked";
        blocker = "device_host_timer_disagreement";
    } else if (point.gate.soak_pass) {
        status = "gated";
        blocker = nullptr;
    }

    const char* executor =
        point.method == Method::HostCopy ? "unknown" : "cu";
    const bool executor_verified = point.method != Method::HostCopy;
    const std::size_t descriptors =
        point.method == Method::HostCopy ? 2 * buffers.records : 0;
    const std::size_t launches =
        point.method == Method::HostCopy ? 8 : 4;

    std::fprintf(
        output,
        "{\"schema_version\":\"%s\",\"point_id\":\"%s\","
        "\"status\":\"%s\",\"method\":\"%s\","
        "\"comparison_family\":\"transport_equivalent\",\"ranks\":2,"
        "\"record_bytes\":%zu,\"records_per_rank\":%zu,"
        "\"total_bytes_per_rank\":%zu,\"tail_record_bytes\":%zu,"
        "\"direction\":\"bidirectional\",\"fanout\":1,"
        "\"lifetime_mode\":\"%s\","
        "\"memory_type\":\"hip_device_peer_access\","
        "\"executor\":{\"label\":\"%s\",\"verified\":%s,"
        "\"trace_artifact\":null,\"transport_selector\":null},",
        kSchema, point_id(point.method, buffers).c_str(), status,
        method_name(point.method), buffers.record_bytes, buffers.records,
        kVolume, buffers.tail_record_bytes,
        buffers.depth == 1 ? "one_epoch" : "ping_pong",
        executor, executor_verified ? "true" : "false");

    std::fprintf(
        output,
        "\"work\":{\"payload_bytes\":%zu,\"control_bytes\":8,"
        "\"publications\":2,\"descriptors\":%zu,\"launches\":%zu,"
        "\"queue_depth\":null},",
        2 * kVolume, descriptors, launches);

    std::fprintf(
        output,
        "\"correctness\":{\"digest_pass\":%s,\"samples_pass\":%s,"
        "\"poison_survivors\":%llu,\"generation_pass\":%s,"
        "\"output_poison_overwritten\":%s,\"all_rank_errors\":%u,"
        "\"negative_controls\":{",
        point.gate.digest_pass && point.timed_correctness_pass ? "true" : "false",
        point.gate.samples_pass && point.timed_correctness_pass ? "true" : "false",
        static_cast<unsigned long long>(point.gate.poison_survivors),
        point.gate.generation_pass ? "true" : "false",
        point.gate.output_overwritten ? "true" : "false",
        point.gate.all_rank_errors);
    bool first_control = true;
    for (const auto& entry : point.gate.controls) {
        std::fprintf(
            output,
            "%s\"%s\":{\"applicable\":%s,\"detected\":%s,"
            "\"detector\":\"%s\"}",
            first_control ? "" : ",", entry.first.c_str(),
            entry.second.applicable ? "true" : "false",
            entry.second.detected ? "true" : "false",
            entry.second.detector.c_str());
        first_control = false;
    }
    std::fprintf(
        output,
        "},\"soak_epochs\":%d,\"soak_pass\":%s},",
        point.gate.soak_epochs, point.gate.soak_pass ? "true" : "false");

    std::fprintf(
        output,
        "\"timing\":{\"scope\":\"transport_plus_dependent_consumer\","
        "\"campaigns\":1,\"iterations_per_campaign\":%zu,"
        "\"global_makespan_us\":{\"p50\":%.4f,\"p95\":%.4f,\"values\":[",
        point.global_us.size(), p50, p95);
    for (std::size_t index = 0; index < point.global_us.size(); ++index) {
        std::fprintf(output, "%s%.4f", index ? "," : "", point.global_us[index]);
    }
    std::fprintf(output, "]},\"per_direction\":[");
    for (int destination = 0; destination < kWorld; ++destination) {
        const int source = 1 - destination;
        const int transport_device =
            point.method == Method::CuPush ? source : destination;
        const DirectionSeries& direction = point.directions[destination];
        std::fprintf(
            output,
            "%s{\"source_rank\":%d,\"destination_rank\":%d,"
            "\"transport_initiator_rank\":%d,"
            "\"timestamp_rule\":\"spans_subtract_ticks_only_within_one_gpu\","
            "\"component_spans_us\":{",
            destination ? "," : "", source, destination, transport_device);
        emit_span(output, "producer", direction.producer_us);
        std::fprintf(output, ",");
        emit_span(output, "transport", direction.transport_us);
        std::fprintf(output, ",");
        emit_span(output, "consumer", direction.consumer_us);
        std::fprintf(output, "},\"hip_events_us\":{");
        emit_span(output, "producer", direction.producer_event_us);
        std::fprintf(output, ",");
        emit_span(output, "transport", direction.transport_event_us);
        std::fprintf(output, ",");
        emit_span(output, "local_wall", direction.local_wall_event_us);
        std::fprintf(output, "},");
        emit_span(output, "enqueue_us", direction.enqueue_us);
        std::fprintf(output, ",");
        emit_u64_series(output, "poll_spins", direction.poll_spins);
        std::fprintf(output, "}");
    }
    const double gbps = p50 > 0 ? static_cast<double>(2 * kVolume) /
                                     (p50 * 1000.0)
                               : 0.0;
    std::fprintf(
        output,
        "],\"gbps\":%.6f,\"timer_agreement\":{\"host_p50_us\":%.4f,"
        "\"max_local_hip_event_p50_us\":%.4f,\"absolute_delta_us\":%.4f,"
        "\"tolerance_us\":%.4f},"
        "\"device_host_timer_agreement_pass\":%s},",
        gbps, p50, event_p50, timer_delta, timer_tolerance,
        point.timer_agreement_pass ? "true" : "false");

    int runtime_version = 0;
    int driver_version = 0;
    HC(hipRuntimeGetVersion(&runtime_version));
    HC(hipDriverGetVersion(&driver_version));
    std::fprintf(
        output,
        "\"provenance\":{\"diagnostic_scope\":\"one_process_two_gpu\","
        "\"source_rev\":%d,\"arch\":\"gfx950\",\"hip_runtime\":%d,"
        "\"hip_driver\":%d,\"devices\":[",
        NEUTRAL_TRANSPORT_SRC_REV, runtime_version, driver_version);
    for (int device = 0; device < kWorld; ++device) {
        std::fprintf(
            output,
            "%s{\"rank\":%d,\"name\":\"%s\",\"pci_bus_id\":\"%s\"}",
            device ? "," : "", device, buffers.devices[device].name.c_str(),
            buffers.devices[device].pci_bus_id.c_str());
    }
    std::fprintf(
        output,
        "],\"command\":\"%s\",\"consumer_digest\":\"xor_plus_modular_add"
        "_plus_deterministic_samples\",\"src_revision_macro\":"
        "\"NEUTRAL_TRANSPORT_SRC_REV\"},",
        joined_command().c_str());
    if (blocker) {
        std::fprintf(output, "\"blocker\":\"%s\"}\n", blocker);
    } else {
        std::fprintf(output, "\"blocker\":null}\n");
    }
    std::fflush(output);
}

std::set<std::string> load_completed(const std::string& path) {
    std::set<std::string> completed;
    FILE* input = std::fopen(path.c_str(), "r");
    if (!input) return completed;
    std::vector<char> line(1u << 20);
    while (std::fgets(line.data(), static_cast<int>(line.size()), input)) {
        const char* start = std::strstr(line.data(), "\"point_id\":\"");
        if (!start) continue;
        start += 12;
        const char* end = std::strchr(start, '"');
        if (end) completed.emplace(start, static_cast<std::size_t>(end - start));
    }
    std::fclose(input);
    return completed;
}

std::vector<std::size_t> sweep_sizes() {
    return {256,       1024,      4096,      14336,     65536,
            262144,    1048576,   4194304,   16777216,  67108864};
}

const char* need_value(int argc, char** argv, int index) {
    if (index >= argc) {
        std::fprintf(stderr, "[neutral] missing value after %s\n",
                     argv[index - 1]);
        std::exit(2);
    }
    return argv[index];
}

Method parse_method(const char* value) {
    if (std::strcmp(value, "cu_push") == 0) return Method::CuPush;
    if (std::strcmp(value, "cu_pull") == 0) return Method::CuPull;
    if (std::strcmp(value, "host_copy_path") == 0) return Method::HostCopy;
    std::fprintf(stderr, "[neutral] invalid method %s\n", value);
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    for (int index = 0; index < argc; ++index) command_line.emplace_back(argv[index]);
    bool sizes_replaced = false;
    bool methods_replaced = false;
    bool depths_replaced = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--out") {
            config.output = need_value(argc, argv, ++index);
        } else if (argument == "--size") {
            if (!sizes_replaced) config.sizes.clear();
            sizes_replaced = true;
            config.sizes.push_back(
                std::strtoull(need_value(argc, argv, ++index), nullptr, 10));
        } else if (argument == "--sweep") {
            config.sizes = sweep_sizes();
            sizes_replaced = true;
        } else if (argument == "--anchor") {
            config.sizes = {65536};
            sizes_replaced = true;
        } else if (argument == "--method") {
            if (!methods_replaced) config.methods.clear();
            methods_replaced = true;
            config.methods.push_back(
                parse_method(need_value(argc, argv, ++index)));
        } else if (argument == "--lifetime") {
            if (!depths_replaced) config.depths.clear();
            depths_replaced = true;
            const std::string value = need_value(argc, argv, ++index);
            if (value == "one_epoch") config.depths.push_back(1);
            else if (value == "ping_pong") config.depths.push_back(2);
            else {
                std::fprintf(stderr, "[neutral] invalid lifetime %s\n",
                             value.c_str());
                return 2;
            }
        } else if (argument == "--rotations") {
            config.rotations =
                std::atoi(need_value(argc, argv, ++index));
        } else if (argument == "--warmup") {
            config.warmup = std::atoi(need_value(argc, argv, ++index));
        } else if (argument == "--soak") {
            config.soak_epochs = 600;
        } else if (argument == "--max-seconds") {
            config.max_seconds =
                std::atof(need_value(argc, argv, ++index));
        } else if (argument == "--list") {
            config.list_only = true;
        } else {
            std::fprintf(stderr, "[neutral] unknown argument %s\n",
                         argument.c_str());
            return 2;
        }
    }
    if (config.rotations <= 0 || config.warmup < 0) {
        std::fprintf(stderr, "[neutral] rotations must be positive and warmup nonnegative\n");
        return 2;
    }
    for (std::size_t size : config.sizes) {
        if (size == 0 || size > kVolume || (size & 15u) != 0) {
            std::fprintf(stderr,
                         "[neutral] size %zu must be 16-byte aligned and <=64 MiB\n",
                         size);
            return 2;
        }
    }

    if (config.list_only) {
        for (std::size_t size : config.sizes) {
            for (int depth : config.depths) {
                Buffers descriptor;
                descriptor.record_bytes = size;
                descriptor.depth = depth;
                for (Method method : config.methods) {
                    std::printf("%s\n", point_id(method, descriptor).c_str());
                }
            }
        }
        return 0;
    }

    int device_count = 0;
    HC(hipGetDeviceCount(&device_count));
    if (device_count < kWorld) {
        std::fprintf(stderr, "[neutral] requires two GPUs, found %d\n",
                     device_count);
        return 3;
    }

    const std::set<std::string> completed = load_completed(config.output);
    FILE* output = std::fopen(config.output.c_str(), "a");
    if (!output) {
        std::fprintf(stderr, "[neutral] cannot append %s\n",
                     config.output.c_str());
        return 2;
    }
    const auto program_begin = std::chrono::steady_clock::now();
    std::uint32_t epoch = 1;
    int emitted = 0;

    for (std::size_t size : config.sizes) {
        for (int depth : config.depths) {
            if (config.max_seconds > 0) {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - program_begin).count();
                if (elapsed >= config.max_seconds) {
                    std::printf("[neutral] time budget reached; rerun to resume\n");
                    std::fclose(output);
                    return 0;
                }
            }

            Buffers buffers = allocate_buffers(size, depth);
            std::map<Method, PointResult> active;
            for (Method method : config.methods) {
                const std::string id = point_id(method, buffers);
                if (completed.count(id)) {
                    std::printf("[neutral] skip completed %s\n", id.c_str());
                    continue;
                }
                PointResult point;
                point.method = method;
                std::printf("[neutral] gates method=%s size=%zu lifetime=%s soak=%d\n",
                            method_name(method), size,
                            depth == 1 ? "one_epoch" : "ping_pong",
                            config.soak_epochs);
                point.gate = run_gates(buffers, method, &epoch);
                active.emplace(method, std::move(point));
            }

            std::vector<Method> order;
            for (Method method : config.methods) {
                if (active.count(method)) order.push_back(method);
            }
            for (int iteration = 0;
                 iteration < config.warmup + config.rotations && !order.empty();
                 ++iteration) {
                const int rotation =
                    iteration % static_cast<int>(order.size());
                for (std::size_t offset = 0; offset < order.size(); ++offset) {
                    const Method method =
                        order[(static_cast<std::size_t>(rotation) + offset) %
                              order.size()];
                    EpochResult measured = execute_epoch(
                        buffers, method, Mutation::Valid, epoch++, true);
                    if (iteration >= config.warmup) {
                        append_epoch(active.at(method), measured);
                    }
                }
            }

            for (Method method : order) {
                emit_point(output, buffers, active.at(method));
                ++emitted;
                std::printf(
                    "[neutral] emitted method=%s size=%zu lifetime=%s p50=%.2f us\n",
                    method_name(method), size,
                    depth == 1 ? "one_epoch" : "ping_pong",
                    quantile(active.at(method).global_us, 0.50));
            }
            free_buffers(buffers);
        }
    }
    std::fclose(output);
    std::printf("[neutral] complete emitted=%d out=%s\n", emitted,
                config.output.c_str());
    return 0;
}

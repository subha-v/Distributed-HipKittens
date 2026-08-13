// exp_01 Stage-0/1 final rank-per-GPU transport base.
//
// One MPI process owns one GPU.  Every method uses the same hipMalloc source
// and inbox allocations, exchanged with HIP IPC, and the same destination
// epoch + directed-credit protocol.  The three payload paths are separate
// kernels/host paths so an unused method cannot perturb another method's
// device code generation.
#define EXP01_RANK_SRC_REV 4

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "cdna4/ops/group/distributed/completion.cuh"
#include "cdna4/ops/group/distributed/lifetime.cuh"
#include "cdna4/ops/group/distributed/packet.cuh"
#include "cdna4/ops/group/distributed/peer.cuh"
#include "cdna4/ops/group/distributed/sync.cuh"
#include "cdna4/types/global/pgl.cuh"

namespace kd = kittens::distributed;

namespace {

constexpr std::uint64_t kMiB = 1ull << 20;
constexpr std::uint64_t kDefaultBytes = 64ull * kMiB;
constexpr std::uint64_t kPoison = 0xd15ea5edbad0c0deull;
constexpr std::uint64_t kOutputPoison = 0xfacefeedcafebeefull;
constexpr std::uint32_t kThreads = 256;
constexpr std::uint64_t kChunkBytes = 1ull << 20;
constexpr std::uint64_t kRealtimeHz = 100000000ull;
constexpr int kDigestBlocks = 256;
constexpr int kSampleCount = 8;
constexpr std::uint64_t kDefaultSpinLimit = 1ull << 22;
constexpr std::uint64_t kEarlyDelayCycles = 20ull * 1000ull * 1000ull;

int g_rank = -1;

[[noreturn]] void fail(const std::string& message, int code = 2) {
    std::fprintf(stderr, "RANK_TRANSPORT_FATAL rank=%d: %s\n", g_rank,
                 message.c_str());
    std::fflush(stderr);
    if (g_rank >= 0) MPI_Abort(MPI_COMM_WORLD, code);
    std::exit(code);
}

void mpi_check(int rc, const char* expression) {
    if (rc == MPI_SUCCESS) return;
    char text[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(rc, text, &length);
    fail(std::string(expression) + ": " + std::string(text, length));
}

void hip_check(hipError_t rc, const char* expression) {
    if (rc == hipSuccess) return;
    fail(std::string(expression) + ": " + hipGetErrorString(rc));
}

#define MC(expr) mpi_check((expr), #expr)
#define HC(expr) hip_check((expr), #expr)

enum class Phase { gate, soak, run };
enum class Method { cu_push, cu_pull, host_copy_path };
enum class Control {
    none,
    no_publication,
    redirected_destination,
    early_publication,
    dropped_credit,
    early_reuse,
};

struct Options {
    Phase phase = Phase::gate;
    Method method = Method::cu_push;
    std::uint64_t record_bytes = 65536;
    std::uint64_t total_bytes = kDefaultBytes;
    int depth = 2;
    int soak_epochs = 0;
    int iterations = 100;
    std::uint64_t spin_limit = kDefaultSpinLimit;
    std::string lifetime = "ping_pong";
    std::string output = "rank_points.jsonl";
    std::string artifact_dir = "raw/rank";
    std::string gate_artifact;
    std::string soak_artifact;
};

const char* method_name(Method method) {
    switch (method) {
        case Method::cu_push: return "cu_push";
        case Method::cu_pull: return "cu_pull";
        case Method::host_copy_path: return "host_copy_path";
    }
    return "invalid";
}

Method parse_method(const std::string& value) {
    if (value == "cu_push") return Method::cu_push;
    if (value == "cu_pull") return Method::cu_pull;
    if (value == "host_copy_path") return Method::host_copy_path;
    throw std::runtime_error("unknown method: " + value);
}

std::uint64_t parse_u64(const char* text, const char* option) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || end == text || *end != '\0') {
        throw std::runtime_error(std::string("invalid value for ") + option);
    }
    return static_cast<std::uint64_t>(value);
}

void usage(FILE* stream) {
    std::fprintf(
        stream,
        "Usage: rank_transport (--gate | --soak 600 | --run) [options]\n"
        "  --method cu_push|cu_pull|host_copy_path\n"
        "  --record-bytes N       Stage-0 default: 65536\n"
        "  --total-bytes N        fixed-volume default: 67108864\n"
        "  --lifetime one_epoch|ping_pong\n"
        "  --iterations N         --run iterations (default 100)\n"
        "  --spin-limit N         bounded device poll limit\n"
        "  --output PATH          resumable JSONL destination\n"
        "  --artifact-dir PATH    gate/soak artifact directory\n"
        "  --gate-artifact PATH   exact gate artifact override\n"
        "  --soak-artifact PATH   exact soak artifact override\n");
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool phase_seen = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* option) -> const char* {
            if (++i >= argc) {
                throw std::runtime_error(std::string("missing value for ") + option);
            }
            return argv[i];
        };
        if (arg == "--gate") {
            if (phase_seen) throw std::runtime_error("choose exactly one phase");
            options.phase = Phase::gate;
            phase_seen = true;
        } else if (arg == "--soak") {
            if (phase_seen) throw std::runtime_error("choose exactly one phase");
            options.phase = Phase::soak;
            options.soak_epochs = static_cast<int>(parse_u64(need("--soak"), "--soak"));
            phase_seen = true;
        } else if (arg == "--run") {
            if (phase_seen) throw std::runtime_error("choose exactly one phase");
            options.phase = Phase::run;
            phase_seen = true;
        } else if (arg == "--method") {
            options.method = parse_method(need("--method"));
        } else if (arg == "--record-bytes") {
            options.record_bytes = parse_u64(need("--record-bytes"), "--record-bytes");
        } else if (arg == "--total-bytes") {
            options.total_bytes = parse_u64(need("--total-bytes"), "--total-bytes");
        } else if (arg == "--lifetime") {
            options.lifetime = need("--lifetime");
        } else if (arg == "--iterations") {
            options.iterations =
                static_cast<int>(parse_u64(need("--iterations"), "--iterations"));
        } else if (arg == "--spin-limit") {
            options.spin_limit = parse_u64(need("--spin-limit"), "--spin-limit");
        } else if (arg == "--output") {
            options.output = need("--output");
        } else if (arg == "--artifact-dir") {
            options.artifact_dir = need("--artifact-dir");
        } else if (arg == "--gate-artifact") {
            options.gate_artifact = need("--gate-artifact");
        } else if (arg == "--soak-artifact") {
            options.soak_artifact = need("--soak-artifact");
        } else if (arg == "--help" || arg == "-h") {
            usage(stdout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (!phase_seen) throw std::runtime_error("one of --gate, --soak, --run is required");
    if (options.lifetime == "one_epoch") {
        options.depth = 1;
    } else if (options.lifetime == "ping_pong") {
        options.depth = 2;
    } else {
        throw std::runtime_error("--lifetime must be one_epoch or ping_pong");
    }
    if (options.record_bytes == 0 || options.total_bytes == 0 ||
        (options.record_bytes & 7u) != 0u || (options.total_bytes & 15u) != 0u) {
        throw std::runtime_error("record bytes must be 8-byte aligned and total bytes 16-byte aligned");
    }
    if (options.iterations <= 0) throw std::runtime_error("--iterations must be positive");
    if (options.phase == Phase::soak && options.soak_epochs != 600) {
        throw std::runtime_error("--soak requires exactly 600 epochs");
    }
    return options;
}

std::string point_id(const Options& options) {
    std::ostringstream out;
    out << "rank2|" << method_name(options.method) << "|rb" << options.record_bytes
        << "|tb" << options.total_bytes << "|" << options.lifetime << "|d"
        << options.depth << "|rev" << EXP01_RANK_SRC_REV;
    return out.str();
}

std::string safe_id(std::string value) {
    for (char& c : value) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) {
            c = '_';
        }
    }
    return value;
}

std::string default_artifact(const Options& options, const char* suffix) {
    return options.artifact_dir + "/" + safe_id(point_id(options)) + suffix;
}

std::string slurp(const std::string& path) {
    std::ifstream input(path);
    if (!input) return {};
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

bool artifact_passes(const std::string& path, const Options& options,
                     bool require_soak) {
    const std::string text = slurp(path);
    if (text.empty()) return false;
    const std::string id = "\"point_id\":\"" + point_id(options) + "\"";
    if (text.find(id) == std::string::npos ||
        text.find("\"source_rev\":" + std::to_string(EXP01_RANK_SRC_REV)) ==
            std::string::npos ||
        text.find("\"pass\":true") == std::string::npos) {
        return false;
    }
    return !require_soak ||
           text.find("\"soak_epochs\":600") != std::string::npos;
}

void write_atomic_text(const std::string& path, const std::string& text) {
    const std::string temporary = path + ".tmp." + std::to_string(::getpid());
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output) fail("cannot create artifact " + temporary);
        output << text << '\n';
        output.flush();
        if (!output) fail("failed writing artifact " + temporary);
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        fail("cannot rename artifact into " + path);
    }
}

__host__ __device__ inline std::uint64_t avalanche(std::uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

__host__ __device__ inline std::uint64_t payload_value(
    std::uint32_t epoch, int source, int destination, std::uint64_t record,
    std::uint64_t word) {
    std::uint64_t x = static_cast<std::uint64_t>(epoch) * 0x9e3779b97f4a7c15ull;
    x ^= static_cast<std::uint64_t>(source + 1) * 0xd6e8feb86659fd93ull;
    x ^= static_cast<std::uint64_t>(destination + 3) * 0xa0761d6478bd642full;
    x ^= record * 0xe7037ed1a0b428dbull;
    x ^= word * 0x8ebc6af09c88c6e3ull;
    x = avalanche(x);
    return x == kPoison ? (x ^ 0x0101010101010101ull) : x;
}

__host__ __device__ inline std::uint64_t digest_x(std::uint64_t value,
                                                  std::uint64_t index) {
    return avalanche(value ^ (index * 0x9e3779b97f4a7c15ull));
}

__host__ __device__ inline std::uint64_t digest_weight(std::uint64_t index) {
    return (avalanche(index ^ 0x243f6a8885a308d3ull) | 1ull);
}

struct DeviceDigest {
    unsigned long long actual_xor;
    unsigned long long actual_add;
    unsigned long long expected_xor;
    unsigned long long expected_add;
    unsigned long long mismatches;
    unsigned long long poison_survivors;
    unsigned long long sample_failures;
    unsigned long long completion_epoch;
};

struct DeviceStatus {
    std::uint64_t ready_spins;
    std::uint64_t credit_spins;
    std::uint32_t ready_observed;
    std::uint32_t credit_observed;
    std::uint32_t ready;
    std::uint32_t credit_ready;
};

struct DeviceStamps {
    unsigned long long transport_begin;
    unsigned long long transport_end;
    unsigned long long consumer_begin;
    unsigned long long consumer_end;
};

__device__ __forceinline__ std::uint64_t realtime_now() {
    std::uint64_t ticks;
    asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)"
                 : "=s"(ticks) :: "memory");
    return ticks;
}

__global__ void fill_payload_kernel(std::uint64_t* destination,
                                    std::uint64_t total_words,
                                    std::uint64_t words_per_record,
                                    std::uint32_t epoch, int source,
                                    int destination_rank) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total_words; index += stride) {
        const std::uint64_t record = index / words_per_record;
        const std::uint64_t word = index - record * words_per_record;
        destination[index] =
            payload_value(epoch, source, destination_rank, record, word);
    }
}

__global__ void poison_kernel(std::uint64_t* destination,
                              std::uint64_t total_words) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total_words; index += stride) {
        destination[index] = kPoison;
    }
}

__global__ void poison_output_kernel(DeviceDigest* digest,
                                     DeviceStatus* status,
                                     DeviceStamps* stamps) {
    if (blockIdx.x || threadIdx.x) return;
    auto* words = reinterpret_cast<unsigned long long*>(digest);
    for (std::size_t i = 0; i < sizeof(DeviceDigest) / sizeof(unsigned long long);
         ++i) {
        words[i] = kOutputPoison;
    }
    std::memset(status, 0, sizeof(*status));
    std::memset(stamps, 0, sizeof(*stamps));
}

__global__ void reset_digest_kernel(DeviceDigest* digest) {
    if (blockIdx.x || threadIdx.x) return;
    std::memset(digest, 0, sizeof(*digest));
}

__global__ void stamp_kernel(unsigned long long* destination) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *destination = realtime_now();
}

__global__ void delay_kernel(std::uint64_t cycles) {
    if (blockIdx.x || threadIdx.x) return;
    const std::uint64_t begin = clock64();
    while (clock64() - begin < cycles) {
        asm volatile("s_sleep 1");
    }
}

template<int PeerRank>
__global__ void cu_push_packets_rank(
    std::byte* local_inbox_base, kittens::peer_bases<std::byte> inbox_bases,
    const std::byte* local_source_base, std::uint64_t source_offset,
    std::uint64_t inbox_offset, std::uint64_t bytes,
    std::uint64_t destination_shift) {
    const std::uint64_t begin =
        static_cast<std::uint64_t>(blockIdx.x) * kChunkBytes;
    if (begin >= bytes) return;
    const std::uint64_t count = min(kChunkBytes, bytes - begin);
    std::byte* const local_projection =
        local_inbox_base + inbox_offset + destination_shift + begin;
    std::byte* const destination = kd::translate_peer<PeerRank>(
        local_projection, local_inbox_base, inbox_bases);
    const std::byte* const source =
        local_source_base + source_offset + begin;
    kd::store_peer_packets(destination, source, count, threadIdx.x, blockDim.x);
}

template<int PeerRank>
__global__ void cu_pull_packets_rank(
    std::byte* local_inbox_base, kittens::peer_bases<std::byte> source_bases,
    const std::byte* local_source_projection_base, std::uint64_t source_offset,
    std::uint64_t inbox_offset, std::uint64_t bytes,
    std::uint64_t destination_shift) {
    const std::uint64_t begin =
        static_cast<std::uint64_t>(blockIdx.x) * kChunkBytes;
    if (begin >= bytes) return;
    const std::uint64_t count = min(kChunkBytes, bytes - begin);
    const std::byte* const source = kd::translate_peer<PeerRank>(
        const_cast<std::byte*>(local_source_projection_base + source_offset + begin),
        local_source_projection_base, source_bases);
    std::byte* const destination =
        local_inbox_base + inbox_offset + destination_shift + begin;
    kd::load_peer_packets(destination, source, count, threadIdx.x, blockDim.x);
}

template<int PeerRank>
__global__ void publish_remote_rank(
    std::uint32_t* local_ready_base, kittens::peer_bases<std::byte> ready_bases,
    int slot, std::uint32_t epoch) {
    std::uint32_t* const signal = kd::translate_peer<PeerRank>(
        local_ready_base + slot, local_ready_base, ready_bases);
    kd::release_and_publish<kd::memory_scope::system>(signal, epoch);
}

__global__ void publish_local(std::uint32_t* ready, int slot,
                              std::uint32_t epoch) {
    kd::release_and_publish<kd::memory_scope::system>(ready + slot, epoch);
}

__global__ void wait_ready_kernel(const std::uint32_t* ready, int slot,
                                  std::uint32_t epoch,
                                  std::uint64_t spin_limit,
                                  DeviceStatus* status) {
    if (blockIdx.x || threadIdx.x) return;
    kd::wait_result result;
    kd::init_wait_result(result);
    kd::bounded_observe_acquire_into<kd::memory_scope::system>(
        ready + slot, epoch, spin_limit, result);
    status->ready_spins = result.spins;
    status->ready_observed = result.observed;
    status->ready = result.ready ? 1u : 0u;
}

__global__ void wait_credit_kernel(const std::uint32_t* credit, int slot,
                                   std::uint32_t generation,
                                   std::uint64_t spin_limit,
                                   DeviceStatus* status) {
    if (blockIdx.x || threadIdx.x) return;
    kd::wait_result result;
    kd::init_wait_result(result);
    kd::bounded_wait_slot_reusable_into<kd::memory_scope::system>(
        credit + slot, generation, spin_limit, result);
    status->credit_spins = result.spins;
    status->credit_observed = result.observed;
    status->credit_ready = result.ready ? 1u : 0u;
}

__global__ void digest_kernel(const std::uint64_t* inbox,
                              std::uint64_t total_words,
                              std::uint64_t words_per_record,
                              std::uint32_t epoch, int source,
                              int destination_rank,
                              const DeviceStatus* status,
                              DeviceDigest* digest) {
    if (!status->ready) return;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    unsigned long long ax = 0;
    unsigned long long aa = 0;
    unsigned long long ex = 0;
    unsigned long long ea = 0;
    unsigned long long mismatches = 0;
    unsigned long long poison = 0;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total_words; index += stride) {
        const std::uint64_t record = index / words_per_record;
        const std::uint64_t word = index - record * words_per_record;
        const std::uint64_t expected =
            payload_value(epoch, source, destination_rank, record, word);
        const std::uint64_t actual = inbox[index];
        ax ^= digest_x(actual, index);
        aa += actual * digest_weight(index);
        ex ^= digest_x(expected, index);
        ea += expected * digest_weight(index);
        mismatches += actual != expected;
        poison += actual == kPoison;
    }
    atomicXor(&digest->actual_xor, ax);
    atomicAdd(&digest->actual_add, aa);
    atomicXor(&digest->expected_xor, ex);
    atomicAdd(&digest->expected_add, ea);
    atomicAdd(&digest->mismatches, mismatches);
    atomicAdd(&digest->poison_survivors, poison);
}

__global__ void sample_kernel(const std::uint64_t* inbox,
                              std::uint64_t total_words,
                              std::uint64_t words_per_record,
                              std::uint32_t epoch, int source,
                              int destination_rank,
                              const DeviceStatus* status,
                              DeviceDigest* digest) {
    if (blockIdx.x || threadIdx.x || !status->ready) return;
    const std::uint64_t last = total_words - 1;
    const std::uint64_t samples[kSampleCount] = {
        0,
        min<std::uint64_t>(1, last),
        total_words / 4,
        total_words / 2,
        (3 * total_words) / 4,
        last > 0 ? last - 1 : 0,
        avalanche(epoch ^ 0x1234u) % total_words,
        avalanche((static_cast<std::uint64_t>(epoch) << 32) ^ 0x5678u) %
            total_words,
    };
    unsigned long long failures = 0;
    for (int i = 0; i < kSampleCount; ++i) {
        const std::uint64_t index = samples[i];
        const std::uint64_t record = index / words_per_record;
        const std::uint64_t word = index - record * words_per_record;
        failures += inbox[index] !=
                    payload_value(epoch, source, destination_rank, record, word);
    }
    digest->sample_failures = failures;
    digest->completion_epoch = epoch;
}

template<int PeerRank>
__global__ void retire_remote_rank(
    std::uint32_t* local_credit_base, kittens::peer_bases<std::byte> credit_bases,
    int slot, std::uint32_t generation, const DeviceStatus* status) {
    if (!status->ready) return;
    std::uint32_t* const peer_credit = kd::translate_peer<PeerRank>(
        local_credit_base + slot, local_credit_base, credit_bases);
    kd::drain_and_retire_slot<kd::memory_scope::system>(
        peer_credit, generation);
}

struct RankMetric {
    double producer_us = 0;
    double transport_event_us = 0;
    double consumer_event_us = 0;
    double transport_device_us = 0;
    double consumer_device_us = 0;
    double enqueue_us = 0;
    double completion_us = 0;
    double local_window_us = 0;
    std::uint64_t digest_actual_xor = 0;
    std::uint64_t digest_actual_add = 0;
    std::uint64_t digest_expected_xor = 0;
    std::uint64_t digest_expected_add = 0;
    std::uint64_t mismatches = 0;
    std::uint64_t poison_survivors = 0;
    std::uint64_t sample_failures = 0;
    std::uint64_t ready_spins = 0;
    std::uint32_t ready_observed = 0;
    std::uint32_t credit_observed = 0;
    std::uint32_t completion_epoch = 0;
    std::uint32_t ready = 0;
    std::uint32_t credit_ready = 0;
    std::uint32_t generation_pass = 0;
    std::uint32_t timer_agreement = 0;
};

struct IterationResult {
    double global_us = 0;
    double launch_skew_us = 0;
    double rank_tail_us = 0;
    int all_rank_errors = 0;
    int detected_ranks = 0;
    std::vector<RankMetric> ranks;
};

struct Buffers {
    int device = -1;
    int peer_device = -1;
    int peer_rank = -1;
    hipDeviceProp_t properties{};
    std::byte* source = nullptr;
    std::byte* inbox = nullptr;
    std::uint32_t* ready = nullptr;
    std::uint32_t* credit = nullptr;
    std::byte* peer_source = nullptr;
    std::byte* peer_inbox = nullptr;
    std::uint32_t* peer_ready = nullptr;
    std::uint32_t* peer_credit = nullptr;
    DeviceDigest* digest = nullptr;
    DeviceStatus* status = nullptr;
    DeviceStamps* stamps = nullptr;
    hipStream_t producer_stream = nullptr;
    hipStream_t transport_stream = nullptr;
    hipStream_t consumer_stream = nullptr;
    hipEvent_t producer_begin = nullptr;
    hipEvent_t producer_end = nullptr;
    hipEvent_t transport_begin = nullptr;
    hipEvent_t transport_end = nullptr;
    hipEvent_t consumer_begin = nullptr;
    hipEvent_t consumer_end = nullptr;
    hipEvent_t payload_done = nullptr;
    kittens::peer_bases<std::byte> source_bases{};
    kittens::peer_bases<std::byte> inbox_bases{};
    kittens::peer_bases<std::byte> ready_bases{};
    kittens::peer_bases<std::byte> credit_bases{};
    std::uint64_t allocation_bytes = 0;
};

void set_rank_bases(kittens::peer_bases<std::byte>& bases, std::byte* local,
                    std::byte* peer) {
    bases = {};
    if (g_rank == 0) {
        bases.rank0 = local;
        bases.rank1 = peer;
    } else {
        bases.rank0 = peer;
        bases.rank1 = local;
    }
}

template<typename T>
T* exchange_ipc_pointer(T* local, int peer_rank, int tag) {
    hipIpcMemHandle_t local_handle{};
    hipIpcMemHandle_t peer_handle{};
    HC(hipIpcGetMemHandle(&local_handle, local));
    MC(MPI_Sendrecv(&local_handle, sizeof(local_handle), MPI_BYTE, peer_rank, tag,
                    &peer_handle, sizeof(peer_handle), MPI_BYTE, peer_rank, tag,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    void* mapped = nullptr;
    HC(hipIpcOpenMemHandle(&mapped, peer_handle, hipIpcMemLazyEnablePeerAccess));
    return static_cast<T*>(mapped);
}

Buffers allocate_buffers(const Options& options, int local_rank, int peer_rank,
                         int peer_device) {
    Buffers buffers;
    buffers.device = local_rank;
    buffers.peer_device = peer_device;
    buffers.peer_rank = peer_rank;
    buffers.allocation_bytes = options.total_bytes * options.depth;
    HC(hipGetDeviceProperties(&buffers.properties, buffers.device));
    HC(hipMalloc(&buffers.source, buffers.allocation_bytes));
    HC(hipMalloc(&buffers.inbox, buffers.allocation_bytes));
    HC(hipMalloc(&buffers.ready, sizeof(std::uint32_t) * options.depth));
    HC(hipMalloc(&buffers.credit, sizeof(std::uint32_t) * options.depth));
    HC(hipMalloc(&buffers.digest, sizeof(DeviceDigest)));
    HC(hipMalloc(&buffers.status, sizeof(DeviceStatus)));
    HC(hipMalloc(&buffers.stamps, sizeof(DeviceStamps)));

    buffers.peer_source =
        exchange_ipc_pointer(buffers.source, peer_rank, 101);
    buffers.peer_inbox =
        exchange_ipc_pointer(buffers.inbox, peer_rank, 102);
    buffers.peer_ready =
        exchange_ipc_pointer(buffers.ready, peer_rank, 103);
    buffers.peer_credit =
        exchange_ipc_pointer(buffers.credit, peer_rank, 104);

    set_rank_bases(buffers.source_bases, buffers.source, buffers.peer_source);
    set_rank_bases(buffers.inbox_bases, buffers.inbox, buffers.peer_inbox);
    set_rank_bases(buffers.ready_bases,
                   reinterpret_cast<std::byte*>(buffers.ready),
                   reinterpret_cast<std::byte*>(buffers.peer_ready));
    set_rank_bases(buffers.credit_bases,
                   reinterpret_cast<std::byte*>(buffers.credit),
                   reinterpret_cast<std::byte*>(buffers.peer_credit));

    HC(hipStreamCreateWithFlags(&buffers.producer_stream, hipStreamNonBlocking));
    HC(hipStreamCreateWithFlags(&buffers.transport_stream, hipStreamNonBlocking));
    HC(hipStreamCreateWithFlags(&buffers.consumer_stream, hipStreamNonBlocking));
    HC(hipEventCreate(&buffers.producer_begin));
    HC(hipEventCreate(&buffers.producer_end));
    HC(hipEventCreate(&buffers.transport_begin));
    HC(hipEventCreate(&buffers.transport_end));
    HC(hipEventCreate(&buffers.consumer_begin));
    HC(hipEventCreate(&buffers.consumer_end));
    HC(hipEventCreateWithFlags(&buffers.payload_done, hipEventDisableTiming));
    return buffers;
}

void destroy_buffers(Buffers& buffers) {
    HC(hipSetDevice(buffers.device));
    for (hipEvent_t event : {buffers.producer_begin, buffers.producer_end,
                             buffers.transport_begin, buffers.transport_end,
                             buffers.consumer_begin, buffers.consumer_end,
                             buffers.payload_done}) {
        if (event) HC(hipEventDestroy(event));
    }
    for (hipStream_t stream : {buffers.producer_stream, buffers.transport_stream,
                               buffers.consumer_stream}) {
        if (stream) HC(hipStreamDestroy(stream));
    }
    for (void* pointer : {static_cast<void*>(buffers.peer_source),
                          static_cast<void*>(buffers.peer_inbox),
                          static_cast<void*>(buffers.peer_ready),
                          static_cast<void*>(buffers.peer_credit)}) {
        if (pointer) HC(hipIpcCloseMemHandle(pointer));
    }
    if (buffers.stamps) HC(hipFree(buffers.stamps));
    if (buffers.status) HC(hipFree(buffers.status));
    if (buffers.digest) HC(hipFree(buffers.digest));
    if (buffers.credit) HC(hipFree(buffers.credit));
    if (buffers.ready) HC(hipFree(buffers.ready));
    if (buffers.inbox) HC(hipFree(buffers.inbox));
    if (buffers.source) HC(hipFree(buffers.source));
}

void reinitialize_protocol(const Options& options, Buffers& buffers) {
    HC(hipMemsetAsync(buffers.ready, 0, sizeof(std::uint32_t) * options.depth,
                      buffers.producer_stream));
    HC(hipMemsetAsync(buffers.credit, 0, sizeof(std::uint32_t) * options.depth,
                      buffers.producer_stream));
    HC(hipMemsetAsync(buffers.status, 0, sizeof(DeviceStatus),
                      buffers.producer_stream));
    HC(hipMemsetAsync(buffers.digest, 0, sizeof(DeviceDigest),
                      buffers.producer_stream));
    HC(hipStreamSynchronize(buffers.producer_stream));
    MC(MPI_Barrier(MPI_COMM_WORLD));
}

int grid_for(std::uint64_t bytes) {
    return static_cast<int>((bytes + kChunkBytes - 1) / kChunkBytes);
}

void launch_push(const Options& options, Buffers& buffers,
                 std::uint64_t source_offset, std::uint64_t inbox_offset,
                 std::uint64_t bytes, std::uint64_t shift) {
    const dim3 grid(grid_for(bytes));
    const dim3 block(kThreads);
    if (g_rank == 0) {
        hipLaunchKernelGGL((cu_push_packets_rank<1>), grid, block, 0,
                           buffers.transport_stream, buffers.inbox,
                           buffers.inbox_bases, buffers.source, source_offset,
                           inbox_offset, bytes, shift);
    } else {
        hipLaunchKernelGGL((cu_push_packets_rank<0>), grid, block, 0,
                           buffers.transport_stream, buffers.inbox,
                           buffers.inbox_bases, buffers.source, source_offset,
                           inbox_offset, bytes, shift);
    }
    HC(hipGetLastError());
}

void launch_pull(const Options& options, Buffers& buffers,
                 std::uint64_t source_offset, std::uint64_t inbox_offset,
                 std::uint64_t bytes, std::uint64_t shift) {
    const dim3 grid(grid_for(bytes));
    const dim3 block(kThreads);
    if (g_rank == 0) {
        hipLaunchKernelGGL((cu_pull_packets_rank<1>), grid, block, 0,
                           buffers.transport_stream, buffers.inbox,
                           buffers.source_bases, buffers.source, source_offset,
                           inbox_offset, bytes, shift);
    } else {
        hipLaunchKernelGGL((cu_pull_packets_rank<0>), grid, block, 0,
                           buffers.transport_stream, buffers.inbox,
                           buffers.source_bases, buffers.source, source_offset,
                           inbox_offset, bytes, shift);
    }
    HC(hipGetLastError());
}

void launch_host_copy(Buffers& buffers, std::uint64_t source_offset,
                      std::uint64_t inbox_offset, std::uint64_t bytes,
                      std::uint64_t shift) {
    HC(hipMemcpyPeerAsync(buffers.peer_inbox + inbox_offset + shift,
                          buffers.peer_device,
                          buffers.source + source_offset, buffers.device, bytes,
                          buffers.transport_stream));
}

void launch_publication(Method method, Buffers& buffers, int slot,
                        std::uint32_t epoch) {
    if (method == Method::cu_pull) {
        hipLaunchKernelGGL(publish_local, 1, 1, 0, buffers.transport_stream,
                           buffers.ready, slot, epoch);
    } else if (g_rank == 0) {
        hipLaunchKernelGGL((publish_remote_rank<1>), 1, 1, 0,
                           buffers.transport_stream, buffers.ready,
                           buffers.ready_bases, slot, epoch);
    } else {
        hipLaunchKernelGGL((publish_remote_rank<0>), 1, 1, 0,
                           buffers.transport_stream, buffers.ready,
                           buffers.ready_bases, slot, epoch);
    }
    HC(hipGetLastError());
}

void launch_retire(Buffers& buffers, int slot, std::uint32_t generation) {
    if (g_rank == 0) {
        hipLaunchKernelGGL((retire_remote_rank<1>), 1, 1, 0,
                           buffers.consumer_stream, buffers.credit,
                           buffers.credit_bases, slot, generation,
                           buffers.status);
    } else {
        hipLaunchKernelGGL((retire_remote_rank<0>), 1, 1, 0,
                           buffers.consumer_stream, buffers.credit,
                           buffers.credit_bases, slot, generation,
                           buffers.status);
    }
    HC(hipGetLastError());
}

bool prepare_epoch(const Options& options, Buffers& buffers,
                   std::uint32_t epoch, int slot, std::uint32_t generation,
                   bool prepare_alternate) {
    HC(hipMemsetAsync(buffers.status, 0, sizeof(DeviceStatus),
                      buffers.producer_stream));
    hipLaunchKernelGGL(wait_credit_kernel, 1, 1, 0, buffers.producer_stream,
                       buffers.credit, slot, generation, options.spin_limit,
                       buffers.status);
    HC(hipGetLastError());
    HC(hipStreamSynchronize(buffers.producer_stream));
    DeviceStatus credit_status{};
    HC(hipMemcpy(&credit_status, buffers.status, sizeof(credit_status),
                 hipMemcpyDeviceToHost));
    if (!credit_status.credit_ready) return false;

    const std::uint64_t slot_offset =
        static_cast<std::uint64_t>(slot) * options.total_bytes;
    const std::uint64_t words = options.total_bytes / sizeof(std::uint64_t);
    const std::uint64_t words_per_record = options.record_bytes / 8;
    HC(hipEventRecord(buffers.producer_begin, buffers.producer_stream));
    hipLaunchKernelGGL(fill_payload_kernel, kDigestBlocks, kThreads, 0,
                       buffers.producer_stream,
                       reinterpret_cast<std::uint64_t*>(buffers.source +
                                                        slot_offset),
                       words, words_per_record, epoch, g_rank,
                       buffers.peer_rank);
    hipLaunchKernelGGL(poison_kernel, kDigestBlocks, kThreads, 0,
                       buffers.producer_stream,
                       reinterpret_cast<std::uint64_t*>(buffers.inbox +
                                                        slot_offset),
                       words);
    hipLaunchKernelGGL(poison_output_kernel, 1, 1, 0, buffers.producer_stream,
                       buffers.digest, buffers.status, buffers.stamps);
    if (prepare_alternate) {
        const int alternate = 1 - slot;
        const std::uint64_t alternate_offset =
            static_cast<std::uint64_t>(alternate) * options.total_bytes;
        hipLaunchKernelGGL(fill_payload_kernel, kDigestBlocks, kThreads, 0,
                           buffers.producer_stream,
                           reinterpret_cast<std::uint64_t*>(buffers.source +
                                                            alternate_offset),
                           words, words_per_record, epoch + 1, g_rank,
                           buffers.peer_rank);
    }
    HC(hipGetLastError());
    HC(hipEventRecord(buffers.producer_end, buffers.producer_stream));
    HC(hipEventSynchronize(buffers.producer_end));
    MC(MPI_Barrier(MPI_COMM_WORLD));  // source-production visibility for pull
    return true;
}

IterationResult run_iteration(const Options& options, Buffers& buffers,
                              std::uint32_t epoch, Control control) {
    const int slot = static_cast<int>((epoch - 1) % options.depth);
    const std::uint32_t generation =
        static_cast<std::uint32_t>((epoch - 1) / options.depth + 1);
    const bool early_reuse = control == Control::early_reuse;
    const bool prepared =
        prepare_epoch(options, buffers, epoch, slot, generation, early_reuse);

    int prepared_int = prepared ? 1 : 0;
    int all_prepared = 0;
    MC(MPI_Allreduce(&prepared_int, &all_prepared, 1, MPI_INT, MPI_MIN,
                     MPI_COMM_WORLD));
    IterationResult result;
    if (!all_prepared) {
        const bool expected = control == Control::dropped_credit;
        const int detected = expected && !prepared;
        MC(MPI_Allreduce(&detected, &result.detected_ranks, 1, MPI_INT, MPI_SUM,
                         MPI_COMM_WORLD));
        result.all_rank_errors = expected ? 0 : 2 - result.detected_ranks;
        return result;
    }

    float producer_ms = 0;
    HC(hipEventElapsedTime(&producer_ms, buffers.producer_begin,
                           buffers.producer_end));

    const std::uint64_t slot_offset =
        static_cast<std::uint64_t>(slot) * options.total_bytes;
    const std::uint64_t words = options.total_bytes / 8;
    const std::uint64_t words_per_record = options.record_bytes / 8;
    const std::uint64_t shift =
        control == Control::redirected_destination ? 16 : 0;
    const std::uint64_t bytes =
        control == Control::redirected_destination
            ? options.total_bytes - 16
            : options.total_bytes;

    MC(MPI_Barrier(MPI_COMM_WORLD));
    int release = static_cast<int>(epoch);
    double coordinator_begin = 0;
    if (g_rank == 0) coordinator_begin = MPI_Wtime();
    MC(MPI_Bcast(&release, 1, MPI_INT, 0, MPI_COMM_WORLD));
    const double local_begin = MPI_Wtime();

    HC(hipEventRecord(buffers.consumer_begin, buffers.consumer_stream));
    hipLaunchKernelGGL(stamp_kernel, 1, 1, 0, buffers.consumer_stream,
                       &buffers.stamps->consumer_begin);
    hipLaunchKernelGGL(wait_ready_kernel, 1, 1, 0, buffers.consumer_stream,
                       buffers.ready, slot, epoch, options.spin_limit,
                       buffers.status);
    if (control == Control::early_reuse) {
        hipLaunchKernelGGL(delay_kernel, 1, 1, 0, buffers.consumer_stream,
                           kEarlyDelayCycles);
    }
    hipLaunchKernelGGL(reset_digest_kernel, 1, 1, 0, buffers.consumer_stream,
                       buffers.digest);
    hipLaunchKernelGGL(digest_kernel, kDigestBlocks, kThreads, 0,
                       buffers.consumer_stream,
                       reinterpret_cast<const std::uint64_t*>(buffers.inbox +
                                                              slot_offset),
                       words, words_per_record, epoch, buffers.peer_rank, g_rank,
                       buffers.status, buffers.digest);
    hipLaunchKernelGGL(sample_kernel, 1, 1, 0, buffers.consumer_stream,
                       reinterpret_cast<const std::uint64_t*>(buffers.inbox +
                                                              slot_offset),
                       words, words_per_record, epoch, buffers.peer_rank, g_rank,
                       buffers.status, buffers.digest);
    if (control != Control::dropped_credit) {
        launch_retire(buffers, slot, generation);
    }
    hipLaunchKernelGGL(stamp_kernel, 1, 1, 0, buffers.consumer_stream,
                       &buffers.stamps->consumer_end);
    HC(hipEventRecord(buffers.consumer_end, buffers.consumer_stream));

    HC(hipEventRecord(buffers.transport_begin, buffers.transport_stream));
    hipLaunchKernelGGL(stamp_kernel, 1, 1, 0, buffers.transport_stream,
                       &buffers.stamps->transport_begin);
    const auto enqueue_begin = std::chrono::steady_clock::now();

    if (control == Control::early_publication) {
        launch_publication(options.method, buffers, slot, epoch);
        hipLaunchKernelGGL(delay_kernel, 1, 1, 0, buffers.transport_stream,
                           kEarlyDelayCycles);
    }

    if (options.method == Method::cu_push) {
        launch_push(options, buffers, slot_offset, slot_offset, bytes, shift);
    } else if (options.method == Method::cu_pull) {
        launch_pull(options, buffers, slot_offset, slot_offset, bytes, shift);
    } else {
        launch_host_copy(buffers, slot_offset, slot_offset, bytes, shift);
        HC(hipEventRecord(buffers.payload_done, buffers.transport_stream));
    }

    const auto enqueue_end = std::chrono::steady_clock::now();
    const double enqueue_us =
        std::chrono::duration<double, std::micro>(enqueue_end - enqueue_begin)
            .count();

    double completion_us = 0;
    if (options.method == Method::host_copy_path) {
        const auto completion_begin = std::chrono::steady_clock::now();
        // The positive host path cannot publish until the dedicated copy stream
        // has explicitly completed its payload operation.
        HC(hipEventSynchronize(buffers.payload_done));
        completion_us =
            std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - completion_begin)
                .count();
    }

    if (control != Control::no_publication &&
        control != Control::early_publication) {
        launch_publication(options.method, buffers, slot, epoch);
    }

    if (control == Control::early_reuse) {
        const int alternate = 1 - slot;
        const std::uint64_t alternate_offset =
            static_cast<std::uint64_t>(alternate) * options.total_bytes;
        if (options.method == Method::cu_push) {
            launch_push(options, buffers, alternate_offset, slot_offset,
                        options.total_bytes, 0);
        } else if (options.method == Method::cu_pull) {
            launch_pull(options, buffers, alternate_offset, slot_offset,
                        options.total_bytes, 0);
        } else {
            launch_host_copy(buffers, alternate_offset, slot_offset,
                             options.total_bytes, 0);
        }
    }

    hipLaunchKernelGGL(stamp_kernel, 1, 1, 0, buffers.transport_stream,
                       &buffers.stamps->transport_end);
    HC(hipEventRecord(buffers.transport_end, buffers.transport_stream));
    HC(hipGetLastError());
    HC(hipEventSynchronize(buffers.transport_end));
    HC(hipEventSynchronize(buffers.consumer_end));
    const double local_end = MPI_Wtime();
    MC(MPI_Barrier(MPI_COMM_WORLD));
    double coordinator_end = 0;
    if (g_rank == 0) coordinator_end = MPI_Wtime();

    float transport_ms = 0;
    float consumer_ms = 0;
    HC(hipEventElapsedTime(&transport_ms, buffers.transport_begin,
                           buffers.transport_end));
    HC(hipEventElapsedTime(&consumer_ms, buffers.consumer_begin,
                           buffers.consumer_end));

    DeviceDigest digest{};
    DeviceStatus status{};
    DeviceStamps stamps{};
    HC(hipMemcpy(&digest, buffers.digest, sizeof(digest), hipMemcpyDeviceToHost));
    HC(hipMemcpy(&status, buffers.status, sizeof(status), hipMemcpyDeviceToHost));
    HC(hipMemcpy(&stamps, buffers.stamps, sizeof(stamps), hipMemcpyDeviceToHost));
    std::uint32_t credit_value = 0;
    HC(hipMemcpy(&credit_value, buffers.credit + slot, sizeof(credit_value),
                 hipMemcpyDeviceToHost));

    const double cycles_per_us =
        static_cast<double>(kRealtimeHz) / 1.0e6;
    RankMetric local;
    local.producer_us = static_cast<double>(producer_ms) * 1000.0;
    local.transport_event_us = static_cast<double>(transport_ms) * 1000.0;
    local.consumer_event_us = static_cast<double>(consumer_ms) * 1000.0;
    local.transport_device_us =
        stamps.transport_end >= stamps.transport_begin && cycles_per_us > 0
            ? (stamps.transport_end - stamps.transport_begin) / cycles_per_us
            : 0;
    local.consumer_device_us =
        stamps.consumer_end >= stamps.consumer_begin && cycles_per_us > 0
            ? (stamps.consumer_end - stamps.consumer_begin) / cycles_per_us
            : 0;
    local.enqueue_us = enqueue_us;
    local.completion_us = completion_us;
    local.local_window_us = (local_end - local_begin) * 1.0e6;
    local.digest_actual_xor = digest.actual_xor;
    local.digest_actual_add = digest.actual_add;
    local.digest_expected_xor = digest.expected_xor;
    local.digest_expected_add = digest.expected_add;
    local.mismatches = digest.mismatches;
    local.poison_survivors = digest.poison_survivors;
    local.sample_failures = digest.sample_failures;
    local.ready_spins = status.ready_spins;
    local.ready_observed = status.ready_observed;
    local.credit_observed = credit_value;
    local.completion_epoch = static_cast<std::uint32_t>(digest.completion_epoch);
    local.ready = status.ready;
    local.credit_ready = status.credit_ready;
    local.generation_pass =
        status.ready && status.ready_observed == epoch &&
                digest.completion_epoch == epoch &&
                (control == Control::dropped_credit || credit_value == generation)
            ? 1u
            : 0u;
    const double transport_tolerance =
        std::max(2.0, 0.05 * local.transport_event_us);
    const double consumer_tolerance =
        std::max(2.0, 0.05 * local.consumer_event_us);
    const bool transport_agrees =
        local.transport_event_us <= 20.0 ||
        std::abs(local.transport_event_us - local.transport_device_us) <=
            transport_tolerance;
    const bool consumer_agrees =
        local.consumer_event_us <= 20.0 ||
        std::abs(local.consumer_event_us - local.consumer_device_us) <=
            consumer_tolerance;
    local.timer_agreement = transport_agrees && consumer_agrees;

    const bool correct =
        local.ready && local.digest_actual_xor == local.digest_expected_xor &&
        local.digest_actual_add == local.digest_expected_add &&
        local.mismatches == 0 && local.poison_survivors == 0 &&
        local.sample_failures == 0 && local.generation_pass;
    bool detected = false;
    switch (control) {
        case Control::none: detected = correct; break;
        case Control::no_publication: detected = !local.ready; break;
        case Control::redirected_destination:
        case Control::early_publication:
        case Control::early_reuse:
            detected = local.ready && !correct;
            break;
        case Control::dropped_credit:
            detected = correct && credit_value < generation;
            break;
    }
    const int local_error =
        control == Control::none ? (!correct || !local.timer_agreement) : !detected;
    MC(MPI_Allreduce(&local_error, &result.all_rank_errors, 1, MPI_INT, MPI_SUM,
                     MPI_COMM_WORLD));
    const int local_detected = detected ? 1 : 0;
    MC(MPI_Allreduce(&local_detected, &result.detected_ranks, 1, MPI_INT, MPI_SUM,
                     MPI_COMM_WORLD));

    result.ranks.resize(g_rank == 0 ? 2 : 0);
    MC(MPI_Gather(&local, sizeof(local), MPI_BYTE,
                  g_rank == 0 ? result.ranks.data() : nullptr, sizeof(local),
                  MPI_BYTE, 0, MPI_COMM_WORLD));
    double starts[2] = {};
    double durations[2] = {};
    MC(MPI_Gather(&local_begin, 1, MPI_DOUBLE, starts, 1, MPI_DOUBLE, 0,
                  MPI_COMM_WORLD));
    MC(MPI_Gather(&local.local_window_us, 1, MPI_DOUBLE, durations, 1,
                  MPI_DOUBLE, 0, MPI_COMM_WORLD));
    if (g_rank == 0) {
        result.global_us = (coordinator_end - coordinator_begin) * 1.0e6;
        result.launch_skew_us =
            (std::max(starts[0], starts[1]) - std::min(starts[0], starts[1])) *
            1.0e6;
        const double median = 0.5 * (durations[0] + durations[1]);
        result.rank_tail_us = std::max(durations[0], durations[1]) - median;
    }
    return result;
}

double quantile(std::vector<double> values, double q) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const double position = q * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - lower;
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

std::string hex64(std::uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

bool output_contains(const std::string& path, const std::string& id) {
    const std::string text = slurp(path);
    return text.find("\"point_id\":\"" + id + "\"") != std::string::npos;
}

void append_jsonl(const std::string& path, const std::string& record) {
    FILE* file = std::fopen(path.c_str(), "a");
    if (!file) fail("cannot append JSONL " + path);
    if (std::fprintf(file, "%s\n", record.c_str()) < 0 || std::fflush(file) != 0) {
        std::fclose(file);
        fail("failed writing JSONL " + path);
    }
#if defined(__linux__)
    ::fsync(::fileno(file));
#endif
    std::fclose(file);
}

std::string gate_artifact_json(const Options& options, bool pass,
                               const IterationResult& positive,
                               const std::array<bool, 5>& detected,
                               bool early_valid) {
    static constexpr const char* names[5] = {
        "no_publication", "redirected_destination", "early_publication",
        "dropped_credit", "early_reuse"};
    std::ostringstream out;
    out << "{\"artifact_version\":\"exp01.rank-gate.v1\",\"point_id\":\""
        << point_id(options) << "\",\"source_rev\":" << EXP01_RANK_SRC_REV
        << ",\"pass\":" << (pass ? "true" : "false")
        << ",\"positive\":{\"all_rank_errors\":"
        << positive.all_rank_errors << ",\"ranks\":[";
    for (std::size_t i = 0; i < positive.ranks.size(); ++i) {
        if (i) out << ',';
        const RankMetric& rank = positive.ranks[i];
        out << "{\"rank\":" << i
            << ",\"ready\":" << rank.ready
            << ",\"ready_observed\":" << rank.ready_observed
            << ",\"completion_epoch\":" << rank.completion_epoch
            << ",\"credit_ready\":" << rank.credit_ready
            << ",\"credit_observed\":" << rank.credit_observed
            << ",\"generation_pass\":" << rank.generation_pass
            << ",\"mismatches\":" << rank.mismatches
            << ",\"poison_survivors\":" << rank.poison_survivors
            << ",\"sample_failures\":" << rank.sample_failures
            << ",\"actual_xor\":\"" << hex64(rank.digest_actual_xor)
            << "\",\"expected_xor\":\"" << hex64(rank.digest_expected_xor)
            << "\",\"actual_add\":\"" << hex64(rank.digest_actual_add)
            << "\",\"expected_add\":\"" << hex64(rank.digest_expected_add)
            << "\",\"timer_agreement\":" << rank.timer_agreement
            << ",\"transport_event_us\":" << rank.transport_event_us
            << ",\"transport_device_us\":" << rank.transport_device_us
            << ",\"consumer_event_us\":" << rank.consumer_event_us
            << ",\"consumer_device_us\":" << rank.consumer_device_us
            << '}';
    }
    out << "]},\"negative_controls\":{";
    for (int i = 0; i < 5; ++i) {
        if (i) out << ',';
        const bool applicable = i < 3 || options.depth == 2;
        out << '"' << names[i] << "\":{\"applicable\":"
            << (applicable ? "true" : "false") << ",\"detected\":"
            << (detected[i] ? "true" : "false");
        if (i == 2) {
            out << ",\"valid\":" << (early_valid ? "true" : "false");
        }
        out << '}';
    }
    out << "}}";
    return out.str();
}

int run_gate(const Options& options, Buffers& buffers,
             const std::string& gate_path) {
    reinitialize_protocol(options, buffers);
    IterationResult positive =
        run_iteration(options, buffers, 1, Control::none);
    bool positive_pass = positive.all_rank_errors == 0;
    reinitialize_protocol(options, buffers);

    const std::array<Control, 5> controls = {
        Control::no_publication, Control::redirected_destination,
        Control::early_publication, Control::dropped_credit,
        Control::early_reuse};
    std::array<bool, 5> detected{};
    bool early_valid = false;
    for (std::size_t i = 0; i < controls.size(); ++i) {
        if (i >= 3 && options.depth != 2) {
            detected[i] = true;  // explicitly inapplicable, not silently omitted
            continue;
        }
        const int attempts = controls[i] == Control::early_publication ? 3 : 1;
        for (int attempt = 0; attempt < attempts && !detected[i]; ++attempt) {
            // Every control follows a full protocol reinitialization, so its
            // first slot generation is one even when the payload epoch differs
            // in later positive campaigns.
            const std::uint32_t epoch = 1;
            IterationResult control =
                run_iteration(options, buffers, epoch, controls[i]);
            detected[i] = control.detected_ranks == 2 &&
                          control.all_rank_errors == 0;
            if (controls[i] == Control::dropped_credit && detected[i]) {
                // The missing return is not considered detected merely because
                // the cell stayed zero: prove the next reuse bounded-times out.
                IterationResult reuse = run_iteration(
                    options, buffers,
                    static_cast<std::uint32_t>(epoch + options.depth),
                    Control::dropped_credit);
                detected[i] = reuse.detected_ranks == 2 &&
                              reuse.all_rank_errors == 0;
            }
            reinitialize_protocol(options, buffers);
        }
        if (controls[i] == Control::early_publication) {
            early_valid = detected[i];
        }
    }
    const bool pass =
        positive_pass && std::all_of(detected.begin(), detected.end(),
                                     [](bool value) { return value; }) &&
        early_valid;
    if (g_rank == 0) {
        write_atomic_text(gate_path,
                          gate_artifact_json(options, pass, positive, detected,
                                             early_valid));
        std::printf("RANK_GATE point=%s pass=%s artifact=%s\n",
                    point_id(options).c_str(), pass ? "true" : "false",
                    gate_path.c_str());
        std::fflush(stdout);
    }
    MC(MPI_Barrier(MPI_COMM_WORLD));
    return pass ? 0 : 3;
}

int run_soak(const Options& options, Buffers& buffers,
             const std::string& gate_path, const std::string& soak_path) {
    int gate_ok = 0;
    if (g_rank == 0) gate_ok = artifact_passes(gate_path, options, false);
    MC(MPI_Bcast(&gate_ok, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (!gate_ok) {
        if (g_rank == 0) {
            std::fprintf(stderr,
                         "RANK_SOAK_BLOCKED missing/failed exact gate artifact %s\n",
                         gate_path.c_str());
        }
        return 4;
    }
    reinitialize_protocol(options, buffers);
    int errors = 0;
    int first_failure_epoch = 0;
    IterationResult first_failure;
    for (int epoch = 1; epoch <= 600 && errors == 0; ++epoch) {
        IterationResult result = run_iteration(
            options, buffers, static_cast<std::uint32_t>(epoch), Control::none);
        if (result.all_rank_errors != 0) {
            first_failure_epoch = epoch;
            first_failure = result;
        }
        errors += result.all_rank_errors;
    }
    const bool pass = errors == 0;
    if (g_rank == 0) {
        std::ostringstream out;
        out << "{\"artifact_version\":\"exp01.rank-soak.v1\",\"point_id\":\""
            << point_id(options) << "\",\"source_rev\":" << EXP01_RANK_SRC_REV
            << ",\"soak_epochs\":600,\"all_rank_errors\":" << errors
            << ",\"pass\":" << (pass ? "true" : "false")
            << ",\"first_failure_epoch\":" << first_failure_epoch
            << ",\"first_failure_ranks\":[";
        for (std::size_t i = 0; i < first_failure.ranks.size(); ++i) {
            if (i) out << ',';
            const RankMetric& rank = first_failure.ranks[i];
            out << "{\"rank\":" << i
                << ",\"ready\":" << rank.ready
                << ",\"ready_observed\":" << rank.ready_observed
                << ",\"completion_epoch\":" << rank.completion_epoch
                << ",\"credit_observed\":" << rank.credit_observed
                << ",\"generation_pass\":" << rank.generation_pass
                << ",\"mismatches\":" << rank.mismatches
                << ",\"poison_survivors\":" << rank.poison_survivors
                << ",\"sample_failures\":" << rank.sample_failures
                << ",\"actual_xor\":\"" << hex64(rank.digest_actual_xor)
                << "\",\"expected_xor\":\"" << hex64(rank.digest_expected_xor)
                << "\",\"actual_add\":\"" << hex64(rank.digest_actual_add)
                << "\",\"expected_add\":\"" << hex64(rank.digest_expected_add)
                << "\",\"timer_agreement\":" << rank.timer_agreement
                << ",\"transport_event_us\":" << rank.transport_event_us
                << ",\"transport_device_us\":" << rank.transport_device_us
                << ",\"consumer_event_us\":" << rank.consumer_event_us
                << ",\"consumer_device_us\":" << rank.consumer_device_us
                << '}';
        }
        out << "]}";
        write_atomic_text(soak_path, out.str());
        std::printf("RANK_SOAK point=%s pass=%s epochs=600 artifact=%s\n",
                    point_id(options).c_str(), pass ? "true" : "false",
                    soak_path.c_str());
        std::fflush(stdout);
    }
    MC(MPI_Barrier(MPI_COMM_WORLD));
    return pass ? 0 : 5;
}

std::string run_record_json(const Options& options,
                            const std::vector<IterationResult>& iterations,
                            bool pass) {
    std::vector<double> global;
    std::vector<double> skew;
    std::vector<double> tails;
    std::array<std::vector<double>, 2> producer;
    std::array<std::vector<double>, 2> transport;
    std::array<std::vector<double>, 2> consumer;
    std::array<std::vector<double>, 2> transport_device;
    std::array<std::vector<double>, 2> consumer_device;
    std::array<std::vector<double>, 2> enqueue;
    std::array<std::vector<double>, 2> completion;
    int all_errors = 0;
    bool timer_pass = true;
    std::uint64_t poison = 0;
    bool samples_pass = true;
    bool digest_pass = true;
    bool generation_pass = true;
    for (const IterationResult& iteration : iterations) {
        global.push_back(iteration.global_us);
        skew.push_back(iteration.launch_skew_us);
        tails.push_back(iteration.rank_tail_us);
        all_errors += iteration.all_rank_errors;
        for (int rank = 0; rank < 2; ++rank) {
            const RankMetric& metric = iteration.ranks[rank];
            producer[rank].push_back(metric.producer_us);
            transport[rank].push_back(metric.transport_event_us);
            consumer[rank].push_back(metric.consumer_event_us);
            transport_device[rank].push_back(metric.transport_device_us);
            consumer_device[rank].push_back(metric.consumer_device_us);
            enqueue[rank].push_back(metric.enqueue_us);
            completion[rank].push_back(metric.completion_us);
            timer_pass &= metric.timer_agreement;
            poison += metric.poison_survivors;
            samples_pass &= metric.sample_failures == 0;
            digest_pass &= metric.mismatches == 0 &&
                           metric.digest_actual_xor == metric.digest_expected_xor &&
                           metric.digest_actual_add == metric.digest_expected_add;
            generation_pass &= metric.generation_pass;
        }
    }
    const double p50 = quantile(global, 0.50);
    const double p95 = quantile(global, 0.95);
    const std::uint64_t records =
        (options.total_bytes + options.record_bytes - 1) / options.record_bytes;
    const std::uint64_t tail =
        options.total_bytes % options.record_bytes == 0
            ? options.record_bytes
            : options.total_bytes % options.record_bytes;
    const bool gated = pass && all_errors == 0 && timer_pass && digest_pass &&
                       samples_pass && poison == 0 && generation_pass;
    std::ostringstream out;
    out << std::setprecision(10);
    out << "{\"schema_version\":\"exp01.transport-point.v1\""
        << ",\"point_id\":\"" << point_id(options) << "\""
        << ",\"status\":\"" << (gated ? "gated" : "incorrect") << "\""
        << ",\"method\":\"" << method_name(options.method) << "\""
        << ",\"comparison_family\":\"transport_equivalent\""
        << ",\"ranks\":2"
        << ",\"record_bytes\":" << options.record_bytes
        << ",\"records_per_rank\":" << records
        << ",\"total_bytes_per_rank\":" << options.total_bytes
        << ",\"tail_record_bytes\":" << tail
        << ",\"direction\":\"bidirectional\""
        << ",\"fanout\":1"
        << ",\"lifetime_mode\":\"" << options.lifetime << "\""
        << ",\"ping_pong_depth\":" << options.depth
        << ",\"memory_type\":\"hipMalloc_ipc\""
        << ",\"executor\":{\"label\":\""
        << (options.method == Method::host_copy_path ? "unknown" : "cu")
        << "\",\"verified\":"
        << (options.method == Method::host_copy_path ? "false" : "true")
        << ",\"trace_artifact\":null,\"transport_selector\":null}"
        << ",\"work\":{\"payload_bytes\":" << (2 * options.total_bytes)
        << ",\"control_bytes\":8,\"publications\":2,\"descriptors\":"
        << (options.method == Method::host_copy_path ? 2 : 0)
        << ",\"launches\":"
        << (options.method == Method::host_copy_path ? 8 : 10)
        << ",\"queue_depth\":null}"
        << ",\"correctness\":{\"digest_pass\":"
        << (digest_pass ? "true" : "false")
        << ",\"samples_pass\":" << (samples_pass ? "true" : "false")
        << ",\"poison_survivors\":" << poison
        << ",\"generation_pass\":" << (generation_pass ? "true" : "false")
        << ",\"all_rank_errors\":" << all_errors
        << ",\"negative_controls\":{"
        << "\"no_publication\":{\"applicable\":true,\"detected\":true,"
           "\"detector\":\"bounded_poll_timeout\"},"
        << "\"redirected_destination\":{\"applicable\":true,\"detected\":true,"
           "\"detector\":\"digest_sample_poison\"},"
        << "\"early_publication\":{\"applicable\":true,\"detected\":true,"
           "\"valid\":true,\"detector\":\"digest_sample_poison\"},"
        << "\"dropped_credit\":{\"applicable\":"
        << (options.depth == 2 ? "true" : "false")
        << ",\"detected\":true,\"detector\":\"bounded_credit_timeout\"},"
        << "\"early_reuse\":{\"applicable\":"
        << (options.depth == 2 ? "true" : "false")
        << ",\"detected\":true,\"detector\":\"generation_digest\"}}"
        << ",\"soak_epochs\":600,\"soak_pass\":true}"
        << ",\"timing\":{\"campaigns\":1,\"iterations_per_campaign\":"
        << iterations.size()
        << ",\"global_makespan_us\":{\"p50\":" << p50 << ",\"p95\":" << p95
        << ",\"values\":[";
    for (std::size_t i = 0; i < global.size(); ++i) {
        if (i) out << ',';
        out << global[i];
    }
    out << "]},\"tp_us\":" << quantile(producer[0], 0.5)
        << ",\"tm_us\":"
        << std::max(quantile(transport[0], 0.5), quantile(transport[1], 0.5))
        << ",\"tcons_us\":"
        << std::max(quantile(consumer[0], 0.5), quantile(consumer[1], 0.5))
        << ",\"tserial_us\":null,\"tideal_us\":null,\"tjoint_us\":" << p50
        << ",\"enqueue_us\":"
        << std::max(quantile(enqueue[0], 0.5), quantile(enqueue[1], 0.5))
        << ",\"completion_after_final_op_us\":"
        << std::max(quantile(completion[0], 0.5),
                    quantile(completion[1], 0.5))
        << ",\"gbps\":" << (2.0 * options.total_bytes / (p50 * 1000.0))
        << ",\"launch_skew_us\":" << quantile(skew, 0.5)
        << ",\"rank_tail_us\":" << quantile(tails, 0.5)
        << ",\"device_host_timer_agreement_pass\":"
        << (timer_pass ? "true" : "false")
        << ",\"per_direction\":[";
    for (int rank = 0; rank < 2; ++rank) {
        if (rank) out << ',';
        const RankMetric& final = iterations.back().ranks[rank];
        out << "{\"source_rank\":" << rank << ",\"destination_rank\":"
            << (1 - rank) << ",\"component_spans_us\":{\"producer\":"
            << quantile(producer[rank], 0.5) << ",\"transport_hip_event\":"
            << quantile(transport[rank], 0.5) << ",\"consumer_hip_event\":"
            << quantile(consumer[rank], 0.5)
            << ",\"transport_device_clock\":"
            << quantile(transport_device[rank], 0.5)
            << ",\"consumer_device_clock\":"
            << quantile(consumer_device[rank], 0.5)
            << "},\"digest\":{\"xor_actual\":\""
            << hex64(final.digest_actual_xor) << "\",\"xor_expected\":\""
            << hex64(final.digest_expected_xor) << "\",\"add_actual\":\""
            << hex64(final.digest_actual_add) << "\",\"add_expected\":\""
            << hex64(final.digest_expected_add) << "\"}}";
    }
    out << "]}"
        << ",\"provenance\":{\"source_rev\":" << EXP01_RANK_SRC_REV
        << ",\"arch\":\"gfx950\",\"process_model\":\"mpi_rank_per_gpu\""
        << ",\"mpi_world_size\":2,\"allocation\":\"hipMalloc_ipc\"}"
        << ",\"blocker\":null}";
    return out.str();
}

int run_timed(const Options& options, Buffers& buffers,
              const std::string& gate_path, const std::string& soak_path) {
    int artifacts_ok = 0;
    int already_done = 0;
    if (g_rank == 0) {
        artifacts_ok = artifact_passes(gate_path, options, false) &&
                       artifact_passes(soak_path, options, true);
        already_done = output_contains(options.output, point_id(options));
    }
    MC(MPI_Bcast(&artifacts_ok, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MC(MPI_Bcast(&already_done, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (!artifacts_ok) {
        if (g_rank == 0) {
            std::fprintf(stderr,
                         "RANK_TIMING_FORBIDDEN exact gate+soak artifacts are "
                         "missing or failed for %s\n",
                         point_id(options).c_str());
        }
        return 6;
    }
    if (already_done) {
        if (g_rank == 0) {
            std::printf("RANK_RUN_RESUME_SKIP point=%s output=%s\n",
                        point_id(options).c_str(), options.output.c_str());
        }
        return 0;
    }
    reinitialize_protocol(options, buffers);
    std::vector<IterationResult> iterations;
    iterations.reserve(options.iterations);
    bool pass = true;
    for (int i = 0; i < options.iterations && pass; ++i) {
        IterationResult result = run_iteration(
            options, buffers, static_cast<std::uint32_t>(i + 1), Control::none);
        pass &= result.all_rank_errors == 0;
        if (g_rank == 0) iterations.push_back(std::move(result));
    }
    if (g_rank == 0 && !iterations.empty()) {
        const std::string record = run_record_json(options, iterations, pass);
        append_jsonl(options.output, record);
        std::printf("RANK_RUN point=%s pass=%s iterations=%zu output=%s\n",
                    point_id(options).c_str(), pass ? "true" : "false",
                    iterations.size(), options.output.c_str());
        std::fflush(stdout);
    }
    MC(MPI_Barrier(MPI_COMM_WORLD));
    return pass ? 0 : 7;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MC(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    MC(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));
    int world = 0;
    MC(MPI_Comm_size(MPI_COMM_WORLD, &world));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI thread level is below FUNNELED");
    if (world != 2) fail("this Stage-0/1 base requires exactly two MPI ranks");

    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        if (g_rank == 0) {
            std::fprintf(stderr, "argument error: %s\n", error.what());
            usage(stderr);
        }
        MPI_Finalize();
        return 2;
    }

    MPI_Comm local_comm = MPI_COMM_NULL;
    MC(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank,
                           MPI_INFO_NULL, &local_comm));
    int local_rank = -1;
    int local_size = 0;
    MC(MPI_Comm_rank(local_comm, &local_rank));
    MC(MPI_Comm_size(local_comm, &local_size));
    if (local_size != 2) fail("both ranks must be on one host");

    int device_count = 0;
    HC(hipGetDeviceCount(&device_count));
    if (device_count < local_size) fail("fewer visible GPUs than local MPI ranks");
    HC(hipSetDevice(local_rank));
    std::array<int, 2> devices{};
    MC(MPI_Allgather(&local_rank, 1, MPI_INT, devices.data(), 1, MPI_INT,
                     MPI_COMM_WORLD));
    const int peer_rank = 1 - g_rank;
    const int peer_device = devices[peer_rank];
    int can_access = 0;
    HC(hipDeviceCanAccessPeer(&can_access, local_rank, peer_device));
    if (!can_access) fail("hipDeviceCanAccessPeer reports unavailable peer path");
    hipError_t enable = hipDeviceEnablePeerAccess(peer_device, 0);
    if (enable != hipSuccess && enable != hipErrorPeerAccessAlreadyEnabled) {
        hip_check(enable, "hipDeviceEnablePeerAccess");
    }

    Buffers buffers =
        allocate_buffers(options, local_rank, peer_rank, peer_device);
    if (g_rank == 0) {
        std::printf(
            "RANK_TRANSPORT_READY rev=%d world=2 memory=hipMalloc_ipc "
            "method=%s record_bytes=%llu total_bytes=%llu lifetime=%s\n",
            EXP01_RANK_SRC_REV, method_name(options.method),
            static_cast<unsigned long long>(options.record_bytes),
            static_cast<unsigned long long>(options.total_bytes),
            options.lifetime.c_str());
        std::fflush(stdout);
    }

    const std::string gate_path =
        options.gate_artifact.empty()
            ? default_artifact(options, ".gate.json")
            : options.gate_artifact;
    const std::string soak_path =
        options.soak_artifact.empty()
            ? default_artifact(options, ".soak.json")
            : options.soak_artifact;

    int rc = 0;
    if (options.phase == Phase::gate) {
        rc = run_gate(options, buffers, gate_path);
    } else if (options.phase == Phase::soak) {
        rc = run_soak(options, buffers, gate_path, soak_path);
    } else {
        rc = run_timed(options, buffers, gate_path, soak_path);
    }

    destroy_buffers(buffers);
    MC(MPI_Comm_free(&local_comm));
    MC(MPI_Finalize());
    return rc;
}

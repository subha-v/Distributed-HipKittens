#pragma once

// Host-only construction and descriptor-vector patching for fused MoE.
//
// This bridge deliberately stops at ordinary host POD values. The caller owns
// device allocation/upload, address-space validation, lifetime, descriptor
// vector upload, and launch ordering.

#include "../common/iris_adapter.hpp"
#include "moe_hk_adapter.cuh"
#include "moe_mps_adapter.cuh"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace hk_moe::host_abi {

inline constexpr int world_size = 8;
inline constexpr std::size_t donor_descriptor_words = 55;
inline constexpr std::size_t symmetric_heap_descriptor_slot = 55;
inline constexpr std::size_t descriptor_words = 56;

using descriptor_word = std::int64_t;

static_assert(sizeof(void*) == 8,
              "the fused-MoE descriptor ABI requires 64-bit pointers");
static_assert(sizeof(descriptor_word) == sizeof(std::uintptr_t));
static_assert(sizeof(symmetric_heap_descriptor) == 9 * sizeof(void*));

namespace detail {

inline void validate_all_peer_bases(
        const kittens::peer_bases<std::byte>& bases) {
    for (int rank = 0; rank < world_size; ++rank) {
        if (bases.template select<world_size>(rank) == nullptr) {
            throw std::invalid_argument(
                "IRIS returned a null fused-MoE peer heap base");
        }
    }
}

[[nodiscard]] inline descriptor_word encode_address(std::uintptr_t address) {
    return std::bit_cast<descriptor_word>(address);
}

[[nodiscard]] inline std::uintptr_t decode_address(descriptor_word word) {
    return std::bit_cast<std::uintptr_t>(word);
}

} // namespace detail

/** Build the heap-relative descriptor needed for MoE's many suballocations. */
[[nodiscard]] inline symmetric_heap_descriptor snapshot_heap_descriptor(
        const iris::iris_device_view& view) {
    const auto bases =
        kittens::distributed::iris_backend::
            snapshot_peer_heap_bases<world_size>(view);
    detail::validate_all_peer_bases(bases);
    const auto local_heap_address = view.get_heap_base(view.cur_rank());
    if (local_heap_address == 0) {
        throw std::invalid_argument(
            "IRIS returned a null fused-MoE local heap base");
    }
    return make_symmetric_heap_descriptor(
        reinterpret_cast<const void*>(local_heap_address), bases);
}

/**
 * Opaque address of an already-uploaded `symmetric_heap_descriptor`.
 *
 * Construction checks only the ABI-visible nonzero/alignment properties. It
 * cannot establish that the address is device-visible or that its allocation
 * will outlive a launch; those remain uploader responsibilities.
 */
class device_visible_descriptor_address {
public:
    explicit device_visible_descriptor_address(std::uintptr_t address)
        : address_(address) {
        if (address == 0) {
            throw std::invalid_argument(
                "uploaded fused-MoE heap descriptor address must not be null");
        }
        if (address % alignof(symmetric_heap_descriptor) != 0) {
            throw std::invalid_argument(
                "uploaded fused-MoE heap descriptor address is misaligned");
        }
    }

    [[nodiscard]] std::uintptr_t value() const { return address_; }

private:
    std::uintptr_t address_;
};

/** Patch slot 55 of an already-sized 56-word host descriptor vector. */
inline void patch_symmetric_heap_descriptor(
        std::span<descriptor_word> words,
        device_visible_descriptor_address uploaded) {
    if (words.size() != descriptor_words) {
        throw std::invalid_argument(
            "fused-MoE descriptor must contain exactly 56 int64 words");
    }
    words[symmetric_heap_descriptor_slot] =
        detail::encode_address(uploaded.value());
}

/** Append the new slot to the frozen donor's exact 55-word host vector. */
inline void append_symmetric_heap_descriptor(
        std::vector<descriptor_word>& words,
        device_visible_descriptor_address uploaded) {
    if (words.size() != donor_descriptor_words) {
        throw std::invalid_argument(
            "fused-MoE donor descriptor must contain exactly 55 int64 words");
    }
    words.push_back(detail::encode_address(uploaded.value()));
}

/** Validate the extended ABI and recover its opaque uploaded address. */
[[nodiscard]] inline device_visible_descriptor_address
validate_descriptor(std::span<const descriptor_word> words) {
    if (words.size() != descriptor_words) {
        throw std::invalid_argument(
            "fused-MoE descriptor must contain exactly 56 int64 words");
    }
    return device_visible_descriptor_address(
        detail::decode_address(words[symmetric_heap_descriptor_slot]));
}

// ============================================================================
// MPS (minimum-progress specialization) descriptor extension — additive.
//
// The MPS kernel (`k0pf6gm_device_tile_mps.hip`, entry `k0pf6gm_mps_mega`)
// consumes a 63-word descriptor: the symmetric heap descriptor in slot 55 as
// above, plus buffer pointers and the packed runtime config in slots 56..62.
// These helpers stop at host POD values exactly like the base ABI above:
// allocation, upload, address-space validation, lifetime, and launch remain
// caller-owned. Buffer sizing formulas are provided so the runtime can
// allocate before patching; see DESIGN_MPS.md for buffer semantics.
// ============================================================================

inline constexpr std::size_t mps_descriptor_words = 63;
inline constexpr std::size_t mps_slot_queue = 56;
inline constexpr std::size_t mps_slot_arrivals = 57;
inline constexpr std::size_t mps_slot_pushed = 58;
inline constexpr std::size_t mps_slot_claim = 59;
inline constexpr std::size_t mps_slot_state = 60;
inline constexpr std::size_t mps_slot_slots = 61;
inline constexpr std::size_t mps_slot_config = 62;

/** Device-visible addresses of the MPS buffers plus the packed config word. */
struct mps_buffer_binding {
    std::uintptr_t queue;        // uint32[(padmax/32)*16]
    std::uintptr_t arrivals;     // uint32[t_ext*16]
    std::uintptr_t pushed;       // uint32[t_ext]
    std::uintptr_t claim;        // uint32[t_ext]
    std::uintptr_t state;        // 96 bytes: 8 u32 scalars + 8 u64 timestamps
    std::uintptr_t slots;        // u16[world*maxtok*7168]; 16-byte aligned
    std::uint64_t config;        // hk_moe::mps::encode_config(...)
};

[[nodiscard]] inline constexpr std::size_t mps_queue_bytes(
        std::size_t padmax) {
    return (padmax / 32) * 16 * sizeof(std::uint32_t);
}

[[nodiscard]] inline constexpr std::size_t mps_arrivals_bytes(
        std::size_t t_ext) {
    return t_ext * 16 * sizeof(std::uint32_t);
}

[[nodiscard]] inline constexpr std::size_t mps_pushed_bytes(
        std::size_t t_ext) {
    return t_ext * sizeof(std::uint32_t);
}

[[nodiscard]] inline constexpr std::size_t mps_claim_bytes(
        std::size_t t_ext) {
    return t_ext * sizeof(std::uint32_t);
}

[[nodiscard]] inline constexpr std::size_t mps_state_bytes() {
    return 8 * sizeof(std::uint32_t) + 8 * sizeof(std::uint64_t);
}

[[nodiscard]] inline constexpr std::size_t mps_slots_bytes(
        std::size_t maxtok) {
    return static_cast<std::size_t>(world_size) * maxtok * 7168 *
           sizeof(std::uint16_t);
}

/** Validate address properties the kernel's entry guard re-checks on device. */
inline void validate_mps_binding(const mps_buffer_binding& binding) {
    if (binding.queue == 0 || binding.arrivals == 0 || binding.pushed == 0 ||
        binding.claim == 0 || binding.state == 0 || binding.slots == 0) {
        throw std::invalid_argument("MPS buffer addresses must be nonzero");
    }
    if (binding.queue % 4 != 0 || binding.arrivals % 4 != 0 ||
        binding.pushed % 4 != 0 || binding.claim % 4 != 0) {
        throw std::invalid_argument("MPS counter buffers must be 4-byte aligned");
    }
    if (binding.state % 8 != 0) {
        throw std::invalid_argument(
            "MPS state buffer must be 8-byte aligned (u64 timestamp cells)");
    }
    if (binding.slots % 16 != 0) {
        throw std::invalid_argument(
            "MPS slice-landing slots must be 16-byte aligned (packet16)");
    }
    if (!hk_moe::mps::config_is_valid(
            hk_moe::mps::decode_config(binding.config))) {
        throw std::invalid_argument("MPS packed config word failed validation");
    }
}

inline void patch_mps_slots(std::span<descriptor_word> words,
                            const mps_buffer_binding& binding) {
    if (words.size() != mps_descriptor_words) {
        throw std::invalid_argument(
            "fused-MoE MPS descriptor must contain exactly 63 int64 words");
    }
    validate_mps_binding(binding);
    words[mps_slot_queue] = detail::encode_address(binding.queue);
    words[mps_slot_arrivals] = detail::encode_address(binding.arrivals);
    words[mps_slot_pushed] = detail::encode_address(binding.pushed);
    words[mps_slot_claim] = detail::encode_address(binding.claim);
    words[mps_slot_state] = detail::encode_address(binding.state);
    words[mps_slot_slots] = detail::encode_address(binding.slots);
    words[mps_slot_config] = std::bit_cast<descriptor_word>(binding.config);
}

/**
 * Append the full MPS extension to the frozen donor's exact 55-word vector:
 * slot 55 (symmetric heap descriptor) plus slots 56..62.
 */
inline void append_mps_descriptor(
        std::vector<descriptor_word>& words,
        device_visible_descriptor_address uploaded,
        const mps_buffer_binding& binding) {
    if (words.size() != donor_descriptor_words) {
        throw std::invalid_argument(
            "fused-MoE donor descriptor must contain exactly 55 int64 words");
    }
    words.push_back(detail::encode_address(uploaded.value()));
    words.resize(mps_descriptor_words, 0);
    std::span<descriptor_word> span(words);
    patch_mps_slots(span.subspan(0), binding);
}

/** Validate the MPS-extended ABI (length, heap descriptor, buffers, config). */
[[nodiscard]] inline device_visible_descriptor_address
validate_mps_descriptor(std::span<const descriptor_word> words) {
    if (words.size() != mps_descriptor_words) {
        throw std::invalid_argument(
            "fused-MoE MPS descriptor must contain exactly 63 int64 words");
    }
    const mps_buffer_binding binding{
        detail::decode_address(words[mps_slot_queue]),
        detail::decode_address(words[mps_slot_arrivals]),
        detail::decode_address(words[mps_slot_pushed]),
        detail::decode_address(words[mps_slot_claim]),
        detail::decode_address(words[mps_slot_state]),
        detail::decode_address(words[mps_slot_slots]),
        std::bit_cast<std::uint64_t>(words[mps_slot_config])};
    validate_mps_binding(binding);
    return device_visible_descriptor_address(
        detail::decode_address(words[symmetric_heap_descriptor_slot]));
}

} // namespace hk_moe::host_abi

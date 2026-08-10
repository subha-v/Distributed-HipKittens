#pragma once

// Host-only construction and descriptor-vector patching for fused MoE.
//
// This bridge deliberately stops at ordinary host POD values. The caller owns
// device allocation/upload, address-space validation, lifetime, descriptor
// vector upload, and launch ordering.

#include "../common/iris_adapter.hpp"
#include "moe_hk_adapter.cuh"

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

} // namespace hk_moe::host_abi

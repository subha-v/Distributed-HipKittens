#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#ifndef DISTRIBUTED_HEADER
#error "DISTRIBUTED_HEADER must name the CDNA3 or CDNA4 distributed header"
#endif

#include DISTRIBUTED_HEADER

namespace {

using namespace kittens;
using namespace kittens::distributed;

struct alignas(16) rank_storage {
    std::array<std::byte, 4096> bytes{};
};

std::array<rank_storage, pgl_max_world_size> storage;

peer_bases<std::byte> named_bases(std::size_t allocation_offset = 0u) {
    return {
        storage[0].bytes.data() + allocation_offset,
        storage[1].bytes.data() + allocation_offset,
        storage[2].bytes.data() + allocation_offset,
        storage[3].bytes.data() + allocation_offset,
        storage[4].bytes.data() + allocation_offset,
        storage[5].bytes.data() + allocation_offset,
        storage[6].bytes.data() + allocation_offset,
        storage[7].bytes.data() + allocation_offset,
    };
}

void check_peer_addressing() {
    // Exercise allocation and view offsets independently: peer_bases names
    // each allocation, while translate_peer preserves the view's inner offset.
    constexpr std::size_t allocation_offset = 0x200;
    constexpr std::size_t view_offset = 0x120;
    auto bases = named_bases(allocation_offset);
    std::byte* const local_allocation_base =
        storage[3].bytes.data() + allocation_offset;
    auto* local = reinterpret_cast<std::uint32_t*>(
        local_allocation_base + view_offset);

    assert(translate_peer<5>(local, local_allocation_base, bases) ==
           reinterpret_cast<std::uint32_t*>(
               storage[5].bytes.data() + allocation_offset + view_offset));
    assert(translate_peer<8>(local, local_allocation_base, bases, 6) ==
           reinterpret_cast<std::uint32_t*>(
               storage[6].bytes.data() + allocation_offset + view_offset));
    assert(translate_peer<4>(local, local_allocation_base, bases, 6) ==
           nullptr);
    assert(translate_peer<8>(local, local_allocation_base, bases, -1) ==
           nullptr);
    assert((peer_offset<2, std::uint32_t>(bases, view_offset) ==
            reinterpret_cast<std::uint32_t*>(
                storage[2].bytes.data() + allocation_offset + view_offset)));
    assert((peer_offset<8, std::uint32_t>(bases, 7, view_offset) ==
            reinterpret_cast<std::uint32_t*>(
                storage[7].bytes.data() + allocation_offset + view_offset)));
    assert((peer_offset<4, std::uint32_t>(bases, 7, view_offset) == nullptr));
    assert((peer_offset<8, std::uint32_t>(bases, -1, view_offset) == nullptr));

    // Compile-time rank selection must still fail closed when an optional or
    // incompletely populated descriptor supplies no base for that rank.
    auto incomplete = bases;
    incomplete.rank5 = nullptr;
    assert(translate_peer<5>(local, local_allocation_base, incomplete) ==
           nullptr);
    assert((peer_offset<5, std::uint32_t>(incomplete, view_offset) ==
            nullptr));
}

void check_packets() {
    alignas(16) packet16 source[4] = {
        {1u, 2u, 3u, 4u}, {5u, 6u, 7u, 8u},
        {9u, 10u, 11u, 12u}, {13u, 14u, 15u, 16u},
    };
    alignas(16) packet16 destination[4]{};
    alignas(16) packet16 roundtrip[4]{};

    static_assert(sizeof(packet16) == 16);
    assert(packet_contract(destination, source, sizeof(source)));
    assert(store_peer_packets_checked(destination, source, sizeof(source),
                                      0u, 1u));
    assert(std::memcmp(destination, source, sizeof(source)) == 0);
    load_peer_packets(roundtrip, destination, sizeof(destination), 0u, 1u);
    assert(std::memcmp(roundtrip, source, sizeof(source)) == 0);
    assert(!store_peer_packets_checked(destination, source, sizeof(source),
                                       0u, 0u));
    assert(!packet_contract(reinterpret_cast<std::byte*>(destination) + 1,
                            source, sizeof(source)));
    assert(!packet_contract(destination, source, sizeof(source) - 1u));

    alignas(16) std::array<std::byte, 96> row{};
    packet16 values[2] = {{21u, 22u, 23u, 24u},
                          {25u, 26u, 27u, 28u}};
    store_packet_row(row.data(), values, 1u, 32u);
    assert(std::memcmp(row.data() + 16u, &values[0], sizeof(packet16)) == 0);
    assert(std::memcmp(row.data() + 48u, &values[1], sizeof(packet16)) == 0);
}

void check_scoped_operations() {
    std::uint32_t cell = 0u;

    store_relaxed<memory_scope::agent>(&cell, 3u);
    assert(load_relaxed<memory_scope::agent>(&cell) == 3u);
    assert(fetch_add_relaxed<memory_scope::agent>(&cell, 2u) == 3u);
    assert(cell == 5u);
    assert(reserve_rows_relaxed<memory_scope::system>(&cell, 4u) == 5u);
    assert(cell == 9u);

    thread_release<memory_scope::agent>();
    thread_acquire<memory_scope::agent>();
    thread_release<memory_scope::system>();
    thread_acquire<memory_scope::system>();
    producer_drain_release<memory_scope::agent>();
    producer_drain_release<memory_scope::system>();
    cta_acquire<memory_scope::agent>();
    cta_acquire<memory_scope::system>();

    publish_epoch_relaxed<memory_scope::agent>(&cell, 10u);
    assert(cell == 10u);
    release_and_publish<memory_scope::system>(&cell, 11u);
    assert(cell == 11u);
}

void check_completion() {
    static_assert(std::is_standard_layout_v<wait_result>);
    static_assert(std::is_standard_layout_v<wait_result64>);

    std::uint32_t signal = 7u;
    wait_result result;
    init_wait_result(result);
    bounded_poll_relaxed_into<memory_scope::agent>(&signal, 6u, 0u, result);
    assert(result.ready && result.observed == 7u && result.spins == 0u);

    signal = 0u;
    init_wait_result(result);
    bounded_poll_relaxed_into<memory_scope::system>(&signal, 1u, 0u, result);
    assert(!result.ready && result.observed == 0u && result.spins == 1u);

    // Success is represented out-of-band, so the high bit remains a valid
    // monotonic epoch value rather than colliding with timeout encoding.
    signal = 0x80000000u;
    init_wait_result(result);
    bounded_poll_relaxed_into<memory_scope::agent>(
        &signal, 0x80000000u, 0u, result);
    assert(result.ready && result.observed == 0x80000000u &&
           result.spins == 0u);

    signal = 9u;
    init_wait_result(result);
    bounded_observe_acquire_into<memory_scope::system>(
        &signal, 8u, 0u, result);
    assert(result.ready && result.observed == 9u && result.spins == 0u);

    std::uint64_t word = 0u;
    publish_epoch_word_relaxed<memory_scope::agent>(&word, 12u, 345u);
    assert(word == epoch_word(12u, 345u));
    assert(epoch_word_epoch(word) == 12u);
    assert(epoch_word_payload(word) == 345u);

    wait_result64 wide_result;
    init_wait_result(wide_result);
    bounded_poll_epoch_word_relaxed_into<memory_scope::system>(
        &word, 12u, 0u, wide_result);
    assert(wide_result.ready && wide_result.observed == word &&
           wide_result.spins == 0u);

    init_wait_result(wide_result);
    bounded_poll_epoch_word_relaxed_into<memory_scope::agent>(
        &word, 13u, 0u, wide_result);
    assert(!wide_result.ready && wide_result.observed == word &&
           wide_result.spins == 1u);

    std::uint64_t wide_counter = 20u;
    init_wait_result(wide_result);
    bounded_observe_acquire_into<memory_scope::system>(
        &wide_counter, 19u, 0u, wide_result);
    assert(wide_result.ready && wide_result.observed == 20u &&
           wide_result.spins == 0u);
}

void check_counters_and_lifetime() {
    std::uint32_t epoch_cell = 0u;
    assert(epoch32(&epoch_cell) == 1u);
    assert(epoch32(&epoch_cell) == 2u);

    std::uint32_t arrivals = 0u;
    counted_arrival arrival{};
    counted_arrive_into(&arrivals, 3u, arrival);
    assert(arrival.epoch == 1u && arrival.ordinal == 0u && !arrival.last);
    counted_arrive_into(&arrivals, 3u, arrival);
    assert(arrival.epoch == 1u && arrival.ordinal == 1u && !arrival.last);
    counted_arrive_into(&arrivals, 3u, arrival);
    assert(arrival.epoch == 1u && arrival.ordinal == 2u && arrival.last);
    counted_arrive_into(&arrivals, 3u, arrival);
    assert(arrival.epoch == 2u && arrival.ordinal == 0u && !arrival.last);
    counted_arrive_into(&arrivals, 0u, arrival);
    assert(arrival.previous == 0u && arrival.epoch == 0u &&
           arrival.ordinal == 0u && !arrival.last);

    std::uint32_t published_arrivals = 0u;
    counted_arrive_release_into<memory_scope::system>(
        &published_arrivals, 2u, arrival);
    assert(arrival.ordinal == 0u && !arrival.last);
    counted_arrive_release_into<memory_scope::system>(
        &published_arrivals, 2u, arrival);
    assert(arrival.ordinal == 1u && arrival.last);

    std::uint32_t credit = 0u;
    wait_result result;
    init_wait_result(result);
    bounded_wait_slot_reusable_into<memory_scope::system>(
        &credit, 1u, 0u, result);
    assert(result.ready && result.observed == 0u && result.spins == 0u);

    credit = 3u;
    init_wait_result(result);
    bounded_wait_slot_reusable_into<memory_scope::agent>(
        &credit, 4u, 0u, result);
    assert(result.ready && result.observed == 3u && result.spins == 0u);

    credit = 2u;
    init_wait_result(result);
    bounded_wait_slot_reusable_into<memory_scope::system>(
        &credit, 4u, 0u, result);
    assert(!result.ready && result.observed == 2u && result.spins == 1u);

    drain_and_retire_slot<memory_scope::agent>(&credit, 4u);
    assert(credit == 4u);
}

} // namespace

int main() {
    check_peer_addressing();
    check_packets();
    check_scoped_operations();
    check_completion();
    check_counters_and_lifetime();
    return 0;
}

/**
 * @file
 * @brief Producer task-order adapters (knob K2 as primitives).
 *
 * Pure index maps, zero protocol — the cheapest lever in the campaign
 * (docs/distributed/OVERLAP_ABSTRACTIONS.md sections 3-4). Order alone
 * measured ~0; order x certification measured the -144 us combine residue
 * and is what turns coarse certification (K3) from "coarse = late" into
 * "coarse = early enough": the producer's completion ORDER decides how early
 * a certified consumable prefix exists.
 *
 * These are the `N2GM_TASK_DECODE`-class hooks (n2_phase2_gm_mps.cpp:479)
 * expressed as reusable functions. Each returns a decomposition of a linear
 * task id; the caller binds the components to (tile, chunk/owner/...) itself.
 * Everything here must stay branch-free single-expression arithmetic: these
 * run inside hot task loops at peak register pressure, and the exp_24 lesson
 * (one spilled 4-byte value reloaded per atomic) applies to ANY code placed
 * there.
 *
 * Evidence base per map:
 *  - consumer_major: GEMM-natural (tile-major) order exposed no consumable
 *    row before 93.9% completion; nc-major decode (exp_29/exp_35 rung (e),
 *    shipped in the M15 kernel) is what made the consuming pool's early
 *    certified prefix exist at all.
 *  - owner_major_staggered: staggering the owner block by rank makes ring
 *    reduce-scatter FALL OUT of epilogue-carried stores + slab words: every
 *    link busy from the first slab, each owner's reduce starting after 1/Nth
 *    of every peer's GEMM (OVERLAP_ABSTRACTIONS section 6). This is the ERS
 *    producer order in the M25 TP8 design.
 *  - source_interleaved: m17's RR permutation — destination rotation via
 *    order buys injection spreading with no per-destination credit protocol.
 */

#pragma once

#include "detail/config.cuh"

namespace kittens::distributed {

struct task_point {
    unsigned int major;  ///< outer component (tile for consumer_major's owner axis, owner block, ...)
    unsigned int minor;  ///< inner component (chunk, index-within-owner, ...)
};

/**
 * Producer-natural order (the donor default): finish one tile's every chunk
 * before the next tile. Kept for completeness and A/B symmetry; as a
 * consumable-prefix generator it is the measured WORST case.
 */
KITTENS_DISTRIBUTED_DEVICE_INLINE task_point producer_major(
        unsigned int task, unsigned int chunks_per_tile) {
    return { task / chunks_per_tile, task % chunks_per_tile };
}

/**
 * Consumer-major (nc-major) order: complete chunk c across ALL tiles before
 * chunk c+1. The consumer keyed on chunks sees its first certified unit after
 * ~1/chunks of the producer's work instead of after ~all of it.
 * major = tile, minor = chunk.
 */
KITTENS_DISTRIBUTED_DEVICE_INLINE task_point consumer_major(
        unsigned int task, unsigned int num_tiles) {
    return { task % num_tiles, task / num_tiles };
}

/**
 * Owner-major order staggered by rank: rank r emits owner (r+1)'s block
 * first, then (r+2)'s, ... wrapping to its own. With K1 = epilogue-carried
 * depth-bounded transport and K3 = one certificate per owner block, the
 * emergent schedule is ring reduce-scatter. major = destination owner rank,
 * minor = task index within that owner's block.
 *
 * `tasks_per_owner` is the caller's owner-block size in tasks; total tasks
 * must be world * tasks_per_owner. `world` is a power of two on every fabric
 * we ship (8), so the modulo folds to a mask when the caller passes a
 * compile-time world.
 */
KITTENS_DISTRIBUTED_DEVICE_INLINE task_point owner_major_staggered(
        unsigned int task, unsigned int rank, unsigned int world,
        unsigned int tasks_per_owner) {
    const unsigned int block = task / tasks_per_owner;
    return { (rank + 1u + block) % world, task % tasks_per_owner };
}

/**
 * Coprime-stride interleave (m17's RR permutation): a bijection on
 * [0, count) that spreads consecutive tasks across destinations/banks.
 * REQUIRES gcd(stride, count) == 1 — the caller asserts this at plan time
 * (host side), never per task. With count a power of two, any odd stride is
 * valid.
 */
KITTENS_DISTRIBUTED_DEVICE_INLINE unsigned int source_interleaved(
        unsigned int index, unsigned int stride, unsigned int count) {
    return (index * stride) % count;
}

} // namespace kittens::distributed

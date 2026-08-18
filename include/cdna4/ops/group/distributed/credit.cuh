/**
 * @file
 * @brief Bounded-injection remote transport (knob K4 as a primitive).
 *
 * Extracted from the shipped open-coded throttle in
 * `distributed-kernels/fused_moe/n2_phase2_gm_mps.cpp` (exp_21 / exp_24 B),
 * where an UNBOUNDED remote-accumulate epilogue measured 7,110.8 us and the
 * depth-4 bound measured 6,495.8 us: the single largest knob in the fused-MoE
 * campaign (-615.0 us). See docs/distributed/OVERLAP_ABSTRACTIONS.md section 4,
 * item 1.
 *
 * CONTRACT NOTES THE EXTRACTION MUST NOT LOSE:
 *
 * 1. `s_waitcnt vmcnt(N)` encodes N in the instruction's simm16; there is no
 *    register form. Depth is therefore a compile-time literal. Runtime depth
 *    selection must NOT be done by instantiating a hot loop once per depth:
 *    exp_24 B measured that shape spilling ONE 4-byte value with 389 reloads
 *    (scratch_load 9 -> 436), contaminating the control arm. The shipped idiom
 *    is ONE textual copy of the loop with mutually exclusive SGPR lane masks
 *    ("throttle plan") chosen before the loop. This header ships the
 *    per-operation emitters; the plan remains the caller's loop-level
 *    composition, documented here so the rejected shape is never rebuilt.
 *
 * 2. `vmcnt` is one in-order counter per wave (exp_25, exp_38): a wave cannot
 *    overlap its own remote traffic with its own loads, and ANY vector memory
 *    op between the remote issue and the wait changes what the bound means.
 *    Callers place these emitters in epilogue position with no interleaved
 *    vector loads, or accept that the effective depth is renegotiated.
 *
 * 3. A bound that can silently die is not a primitive (exp_38: register
 *    pressure re-throttled an epilogue without any source change). Every
 *    kernel adopting this header must carry the ISA gate: (a) the literal
 *    `s_waitcnt vmcnt(N)` appears in the emitted ISA at the expected sites;
 *    (b) zero scratch operations within 24 instructions of any remote atomic;
 *    (c) the issue-run distribution matches the plan (the fused-MoE gate used
 *    282 atomics / ~12 runs / mean ~23.5). The gate script lives with the
 *    kernel build (see m24_build_gate.sh precedent), not in this header, but
 *    the header is the contract's home.
 *
 * Arch card (docs/distributed/OVERLAP_ABSTRACTIONS.md section 5): on gfx950,
 * coalesced remote atomics run at 52.8 GB/s vs 54.9 GB/s stores, so the
 * accumulate form is a legitimate transport ON THIS ARCH ONLY when coalesced
 * (scattered atomics collapse to 4.1 GB/s). On gfx942 remote atomics measure
 * ~3x slower than stores: use `throttled_store_packet16` plus an owner-side
 * local reduce there (the m15b fold-then-wide-push twin).
 */

#pragma once

#include "detail/config.cuh"
#include "packet.cuh"

namespace kittens::distributed {

/**
 * Cap this wave's outstanding vector-memory operations at `Depth`.
 *
 * Explicit specializations keep the literal in the instruction (no register
 * form exists). Depth 4 is the fused-MoE campaign's shipped value and the
 * measured optimum on gfx950 xGMI for remote pk_add epilogues; 8/16/32 exist
 * for response-curve sweeps, and 0 (full drain) is the release edge.
 */
template <unsigned int Depth>
KITTENS_DISTRIBUTED_DEVICE_INLINE void throttle_vmcnt();

#if defined(__HIP_DEVICE_COMPILE__)
#define KITTENS_DISTRIBUTED_THROTTLE_VMCNT(N)                                 \
    template <>                                                               \
    KITTENS_DISTRIBUTED_DEVICE_INLINE void throttle_vmcnt<N>() {              \
        asm volatile("s_waitcnt vmcnt(" #N ")" ::: "memory");                 \
    }
#else
#define KITTENS_DISTRIBUTED_THROTTLE_VMCNT(N)                                 \
    template <>                                                               \
    KITTENS_DISTRIBUTED_DEVICE_INLINE void throttle_vmcnt<N>() {}
#endif
KITTENS_DISTRIBUTED_THROTTLE_VMCNT(0)
KITTENS_DISTRIBUTED_THROTTLE_VMCNT(4)
KITTENS_DISTRIBUTED_THROTTLE_VMCNT(8)
KITTENS_DISTRIBUTED_THROTTLE_VMCNT(16)
KITTENS_DISTRIBUTED_THROTTLE_VMCNT(32)
#undef KITTENS_DISTRIBUTED_THROTTLE_VMCNT

/**
 * One bounded remote accumulate: issue the packed-bf16 RMW, then cap the
 * wave's outstanding vector ops at `Depth`.
 *
 * Inherits `accumulate_peer_bf162`'s full contract (allocation grain,
 * no-fence, caller-owned visibility). Adds only injection bounding: with the
 * emitter used exclusively in a store-free epilogue span, at most `Depth`
 * remote RMWs are in flight per wave — the congestion-collapse guard that
 * measured -615 us. NOT a release: the caller's drain
 * (`producer_drain_release` / `throttle_vmcnt<0>` + release chain) still ends
 * the span.
 */
template <unsigned int Depth>
KITTENS_DISTRIBUTED_DEVICE_INLINE void throttled_accumulate_bf162(
        void* destination, unsigned int packed_bf16x2) {
    accumulate_peer_bf162(destination, packed_bf16x2);
    throttle_vmcnt<Depth>();
}

/**
 * One bounded remote packet store (the gfx942-safe / store-transport twin;
 * also the ERS fold-then-wide-push variant on gfx950 when atomic contention
 * at an owner slab measures worse than store + owner-side local reduce).
 */
template <unsigned int Depth>
KITTENS_DISTRIBUTED_DEVICE_INLINE void throttled_store_packet16(
        void* __restrict__ destination, const packet16& value) {
    store_packet16(destination, value);
    throttle_vmcnt<Depth>();
}

/**
 * Loop-level composition note (the "throttle plan"): a caller that must pick
 * depth at RUNTIME builds one loop containing an unconditional issue and a
 * short ladder of mask-guarded `throttle_vmcnt<D>()` calls, with the masks
 * (at most one true) hoisted into SGPRs before the loop — the exact shipped
 * shape in n2_phase2_gm_mps.cpp. Do not template the loop on Depth; see the
 * header comment for the measured spill catastrophe.
 */

} // namespace kittens::distributed

#pragma once

// Minimum-progress specialization (MPS) workload adapter for the PF6 MoE
// megakernel. This header owns operator policy ONLY: descriptor slot ids, the
// runtime role/config word, the (b, nc) tile-event wire format, slice arrival
// accounting, owner-slot posting/fan-out, and diagnostic timestamps. Generic
// mechanisms stay in kittens::distributed (see include/cdna4/ops/group/
// distributed/roles.cuh); the LL128/GEMM/sort plumbing remains in the donor
// helpers. Design contract: DESIGN_MPS.md.

#include "moe_hk_adapter.cuh"

#include <cstddef>
#include <cstdint>

namespace hk_moe::mps {

// ---- descriptor extension (appended after K0P6_D_SYMMETRIC = 55) ------------
// Host and device must agree exactly; mirrored by moe_host_abi.hpp.
#define K0P6_D_MPS_Q 56        // uint32[qcap] tile-event slots (MARK|(b<<4)|nc)
#define K0P6_D_MPS_NCARR 57    // uint32[T_ext*16] slice arrival counters
#define K0P6_D_MPS_PUSHED 58   // uint32[T_ext] pushed slice counts (target 16)
#define K0P6_D_MPS_CLAIM 59    // uint32[T_ext] per-row push-group claim bits
#define K0P6_D_MPS_STATE 60    // uint32[8] scalars + uint64[8] diag timestamps
#define K0P6_D_MPS_SLOTS 61    // bf16[world][MAXTOK][7168] owner landing slots
#define K0P6_D_MPS_CFG 62      // packed uint64 role/config word (host-written)
#define K0P6_MPS_D_LEN 63

// ---- pperr bits added by MPS (donor/parity bits 16..25 are unchanged) -------
#define K0P6_MPS_ERR_SERVICE 67108864    // 1<<26: service/event poll timeout
#define K0P6_MPS_ERR_CONFIG 268435456    // 1<<28: MPS config/shape guard

// exp_34 mode 14 REUSES donor bit 25. Its donor meaning is "row_ready combine
// poll timeout" (see the pperr table in the kernel header) and mode 14 is the
// one mode that compiles that poll OUT, so the two meanings can never both be
// reachable in one executed path. Named here so kernel and adapter agree on it.
#define K0P6_MPS_ERR_M7DONE 33554432     // 1<<25: mode-14 M7-done rendezvous

// ---- exp_38: mode 14 IS COMPILE-TIME OPT-IN, AND THAT IS NOT TIDINESS --------
// Merely making mode-14 code REACHABLE cost the mode-12 ratchet +726.9 us
// (6,497.3 -> 7,224.2, same session, same config, same `production` denominator)
// with mode 12's own source lines untouched. The mechanism is register pressure
// and instruction scheduling in the SHARED functions mode 14 extends -- most of
// all `mode_is_direct_accum`, which the M7 epilogue branches on, and
// `k0p6_mps_task_done_maybe_defer`, which every task head runs. The visible
// symptom was that exp_24's `vmcnt` injection bound went INERT: the g=353 vs
// g=65 contrast collapsed from -613.5 us to -1.8 us while all four `vmcnt`
// instantiations stayed present at identical counts, and every field of the
// resource tuple still matched. So the tuple does not certify parity; only
// `.text` byte-identity does.
//
// Everything mode 14 adds therefore sits behind this flag, and the flag defaults
// OFF so that the DEFAULT build of this tree is `.text`-identical to rev 26.
// Turn it on with -DK0P6_MPS_ENABLE_MODE14=1 to run exp_34's arms; that build is
// the one that carries the +727 us, and it is a mode-14 measurement, not a
// mode-12 one. Never publish a mode-12 number from a MODE14=1 binary.
#ifndef K0P6_MPS_ENABLE_MODE14
#define K0P6_MPS_ENABLE_MODE14 0
#endif

// ---- exp_02 (aug12) mode 16: DEFERRED COMBINE (TBO-2 epoch pipelining, stage
// A+B). SAME DISCIPLINE AS MODE 14, FOR THE SAME MEASURED REASON: merely adding
// a third compare to `mode_is_direct_accum` once cost the mode-12 ratchet
// +726.9 us through whole-function register allocation (exp_38). Everything
// mode 16 adds sits behind this flag, default OFF, so the DEFAULT build of
// this tree stays `.text`-identical to the TBO-free build. Turn it on with
// -DK0P6_MPS_ENABLE_TBO=1 to run exp_02's mode-16 arms.
//
//   *** NEVER PUBLISH A MODE-12 NUMBER FROM A TBO=1 BINARY ***
//
// Mode 16 IS mode 12's transport (remote packed-bf16 epilogue accumulate with
// the depth-4 throttle) and mode 12's readiness protocol (event queue, drain,
// per-row row_ready) VERBATIM. The one semantic change is WHEN the combine
// (M8) runs: launch i runs M7/drain for epoch i but claims the combine batches
// of epoch i-1, in two windows -- the plan shadow (all CTAs, after
// pull_src_fill, before the M5 grid barrier, where mode 12's CTAs 2..255 are
// idle at the ratchet config) and the usual M8 window for leftovers. To make
// deferral sound, seven buffers are allocated at TWICE their size and indexed
// by epoch parity (write side = current epoch's parity, consume side = the
// deferred epoch's parity): slots(61), row_ready(25), pull_stage(12),
// pull_cnt(13), pull_ptr(18), pull_src(19), out(32). M0's retire-wait loosens
// from epoch-1 to epoch-2 (the ping-pong protection depth) and M9 pokes
// retired := epoch-1 (the epoch whose slots M8 just consumed+zeroed). See
// overnight/aug12/exp_02_tbo_deferred_combine/design.md for the full protocol
// delta and the required host allocation/gate-lag patch.
#ifndef K0P6_MPS_ENABLE_TBO
#define K0P6_MPS_ENABLE_TBO 1
#endif

// ---- K0P6_D_MPS_STATE scalar word lanes -------------------------------------
#define K0P6_MPS_ST_TICKET 0   // finish-order role tickets (agent monotonic)
#define K0P6_MPS_ST_TAIL 1     // event tail ticket (monotonic, capacity-bound)
#define K0P6_MPS_ST_M8NEXT 2   // dynamic reduction batch ticket
#define K0P6_MPS_ST_EVNEXT 3   // exp_12: dynamic tile-event consumption ticket
#define K0P6_MPS_ST_WORDS 8    // scalar words precede the uint64 diag block
#define K0P6_MPS_TS_FIRST_READY 0   // max(~t) over event enqueues (invert: min)
#define K0P6_MPS_TS_LAST_READY 1    // max(t)   over event enqueues
#define K0P6_MPS_TS_DRAIN 2         // max(t)   over service-wave stripe ends
#define K0P6_MPS_TS_M7_DONE 3       // max(t)   over compute-CTA task drain
#define K0P6_MPS_TS_REDUCE_DONE 4   // max(t)   over CTA reduction exits
// exp_05 stage 0: whole-kernel phase attribution. With these three the epoch
// decomposes end to end without any debug_stop bisect (which aborts the ranks
// on this build): dispatch = M2_DONE - KSTART, plan+M6 = M6_DONE - M2_DONE,
// M7 = M7_DONE - M6_DONE, combine = REDUCE_DONE - M7_DONE. Diagnostic only:
// every write is behind cfg.timestamps and off by default.
#define K0P6_MPS_TS_M5_DONE 5       // max(t)   over CTAs leaving the M5 barrier
#define K0P6_MPS_TS_M2_DONE 6       // max(t)   over CTAs leaving the M2 barrier
#define K0P6_MPS_TS_M6_DONE 7       // max(t)   over CTAs leaving M6
#define K0P6_MPS_TS_COUNT 8

// ---- exp_23: per-CTA phase event ring ---------------------------------------
// Fixed slots, not a queue: cell (b, k) is the LAST time CTA b crossed boundary
// k in this epoch. 256 CTAs x 16 slots x 8 B = 32 KiB, parked in the tail of
// K0P6_D_MPS_STATE immediately after the 8 coarse uint64 cells, so there is no
// new descriptor slot, no host-bridge signature change and no kernel ABI
// change -- only the host allocation of slot 60 grows (the harness edit is
// carried as exp_23_fig10_timeline/e23_ab.patch).
//
// DELIBERATELY NOT ZEROED BY M0. M0's reset loop covers ST_WORDS +
// 2*TS_COUNT uint32 words; widening it would put a grid-wide 32 KiB store into
// the reset path of an arm that ships with the ring off. The host zero-init is
// the floor instead. That is sound because within one launch every CTA
// rewrites each slot it owns on every epoch, so a final-epoch read is current,
// and a slot a CTA never writes in any epoch (M7_DONE on a service CTA) stays
// 0 and is read as "boundary never crossed" rather than as a stale time.
#define K0P6_MPS_E23_SLOTS 16
#define K0P6_MPS_E23_MAX_CTA 256
#define K0P6_MPS_E23_CELLS (K0P6_MPS_E23_SLOTS * K0P6_MPS_E23_MAX_CTA)

#define K0P6_MPS_E23_KSTART 0        // tier B
#define K0P6_MPS_E23_M2_DONE 1       // tier A
#define K0P6_MPS_E23_M5_DONE 2       // tier A
#define K0P6_MPS_E23_M6_DONE 3       // tier A
#define K0P6_MPS_E23_M7_DONE 4       // tier A
#define K0P6_MPS_E23_SVC_ENTER 5     // tier C
#define K0P6_MPS_E23_SVC_EXIT 6      // tier C
#define K0P6_MPS_E23_M75_ENTER 7     // tier C
#define K0P6_MPS_E23_M75_BAR 8       // tier C
#define K0P6_MPS_E23_M75_EXIT 9      // tier C
#define K0P6_MPS_E23_M8_ENTER 10     // tier B
#define K0P6_MPS_E23_REDUCE_DONE 11  // tier A
#define K0P6_MPS_E23_M9_DONE 12      // tier B
#define K0P6_MPS_E23_META 15         // tier C: (role << 32) | task count

// exp_38 DEMOTES THIS FROM 1 TO 0. The original plan was to ship the ring
// compiled in and gate it at runtime on cfg.timestamps, on the strength of a
// parity gate that compared RESOURCE TUPLES. exp_38 established that the tuple
// does not certify parity: the mode-14 commit matched every field of it and
// still cost the mode-12 ratchet +726.9 us. The ring was never GPU-timed, so it
// is not accused -- it is simply held to the gate we now trust, `.text`
// byte-identity against rev 26, which a compiled-in ring cannot meet by
// construction (it adds a store to every stamped boundary). Build exp_23's
// figure arms with -DK0P6_MPS_E23_RING=1 and treat that binary as its own arm.
#ifndef K0P6_MPS_E23_RING
#define K0P6_MPS_E23_RING 0
#endif
// Gate-only: fold `enable` to a compile-time true so the stamps cannot be
// sunk behind a branch. This is how the "ring on" resource tuple is measured.
// Never defined in a shipped build.
#ifndef K0P6_MPS_E23_FORCE_ON
#define K0P6_MPS_E23_FORCE_ON 0
#endif

// ---- packed config word (K0P6_D_MPS_CFG) --------------------------------------
// [0:8)   C    reserved minimum-progress service CTAs (finish order)
// [8:16)  g    adjacent N-chunks grouped per push (1, 2, 4, 16), LOW byte
// [16:24) mode 0 reserve-only | 1 bulk-push-after-M7 | 2 stream
// [24:32) flush_rows  completed-row flags per system release (1..64)
// [32]    pull_fallback: stream flags, but M8 pulls remote part (diagnostic)
// [33]    timestamps_enable
// [34:42) g    HIGH byte (exp_24). See the widening note below.
//
// exp_24 -- WHY THE `g` FIELD IS SPLIT. `g`'s byte at [8:16) was fully spoken
// for (0x0F physical | 0x10 detect | 0x20 throttle) with two bits left, and
// exp_24 needs three: one mechanism bit plus a two-bit depth selector. Passing
// `g > 0xFF` into the OLD encoder silently corrupted the word -- `g = 0x121`
// became `0x121 << 8 = 0x12100`, whose bit 16 lands in the MODE field, and the
// decode then read back `g = 0x21`. So the field is widened to 16 bits, with
// the high byte parked in the packed word's first free bits [34:42).
// This is bit-identical for every `g <= 0xFF`: the high byte is zero, so the
// extra term vanishes and `decode_config` returns exactly the old struct. No
// host-bridge signature, descriptor slot, or `K0_MPS_CFG` grammar changes.
struct config {
    std::uint32_t reserved_comm_ctas;
    std::uint32_t group_slices;
    std::uint32_t mode;
    std::uint32_t flush_rows;
    bool pull_fallback;
    bool timestamps;
};

__host__ __device__ __forceinline__ std::uint64_t encode_config(
        std::uint32_t reserved_comm_ctas, std::uint32_t group_slices,
        std::uint32_t mode, std::uint32_t flush_rows, bool pull_fallback,
        bool timestamps) {
    return (std::uint64_t)reserved_comm_ctas
           | ((std::uint64_t)(group_slices & 0xFFu) << 8)
           | ((std::uint64_t)mode << 16)
           | ((std::uint64_t)flush_rows << 24)
           | ((std::uint64_t)(pull_fallback ? 1u : 0u) << 32)
           | ((std::uint64_t)(timestamps ? 1u : 0u) << 33)
           | ((std::uint64_t)((group_slices >> 8) & 0xFFu) << 34);
}

__host__ __device__ __forceinline__ config decode_config(std::uint64_t word) {
    config c;
    c.reserved_comm_ctas = (std::uint32_t)(word & 0xFFu);
    c.group_slices = (std::uint32_t)((word >> 8) & 0xFFu)
                     | ((std::uint32_t)((word >> 34) & 0xFFu) << 8);
    c.mode = (std::uint32_t)((word >> 16) & 0xFFu);
    c.flush_rows = (std::uint32_t)((word >> 24) & 0xFFu);
    c.pull_fallback = ((word >> 32) & 1u) != 0u;
    c.timestamps = ((word >> 33) & 1u) != 0u;
    return c;
}

// exp_08 / axis A8 — XCD placement of the service pool.
//
// Workgroups go round-robin to XCDs with chunk size one (`xcd = bid % 8`), so
// the mode-2 static TAIL reservation spreads the pool one-eighth per die and
// touches all eight 4 MB L2s. exp_05 measured that a running service pool
// inflates M7 by 37-57% purely through memory-system interference, and exp_07
// showed the mechanism is atomic contention rather than cache-line footprint.
// If that contention is resolved in the per-XCD L2s, then giving the pool WHOLE
// DIES should leave the compute dies' L2s clean.
//
// Mode 3 is mode 2 with exactly that placement change and nothing else:
// residue classes [0, C/32) are service, [C/32, 8) are compute. One XCD is
// 256/8 = 32 CTAs, so C must be a multiple of 32; with the existing C <= 64 cap
// the legal points are C = 32 (one die) and C = 64 (two dies).
inline constexpr std::uint32_t kCtasPerXcd = 32u;
inline constexpr std::uint32_t kXcds = 8u;

__host__ __device__ __forceinline__ std::uint32_t service_dies(config c) {
    return c.reserved_comm_ctas / kCtasPerXcd;
}

// exp_20 diagnostic modes 4/5/6 -- Tier 1 of the ablation queue.
//
// Mode 4 IS mode 2 with a tunable pacing delay inserted in the push path (E1:
// rate vs volume). Modes 5 and 6 ARE mode 0 -- the reserve-only control, with
// the parity M7.5 publication and the parity M8 remote-`part` pull -- but the
// reserved pool, instead of idling into the rendezvous, runs a synthetic
// traffic generator (E2: read/write/fabric attribution) or an LDS-only spin
// (E3: occupancy without traffic) for a bounded window that covers M7.
//
// All three are CORRECTNESS-PRESERVING and run the full gate ladder. Mode 0's
// M8 pulls the producer's remote `part` row and never reads K0P6_D_MPS_SLOTS,
// so the slots buffer is DEAD in modes 5/6 and a generator may write it
// freely; the generator only ever READS `part`, which it does not own.
//
// Config-field reinterpretation, so no descriptor, host-ABI or harness change
// is needed (exactly how mode 3 was added):
//   mode 4: flush_rows -> 1 + pacing units; the real flag-batch depth is
//           pinned at 16 so pacing is the ONLY variable against mode 2.
//   mode 5: g -> traffic variant (1 read+write | 2 read-only | 4 write-only);
//           pull_fallback -> write this rank's own slots instead of a peer's;
//           flush_rows -> active window in units of 100 us.
//   mode 6: flush_rows -> active window in units of 100 us.
inline constexpr std::uint32_t kDiagTicksPer100us = 10000u;  // 100 MHz realtime
inline constexpr std::uint32_t kDiagVariantCopy = 1u;
inline constexpr std::uint32_t kDiagVariantRead = 2u;
inline constexpr std::uint32_t kDiagVariantWrite = 4u;

// Mode 7 is the sharpest of the set: mode 2 with the 896 B payload copy
// DELETED and nothing else touched -- same events, same arrival atomics, same
// event-queue spin, same flags, same fences, and above all the same
// event-driven rate, so there is no rate-matching problem to argue about. It
// stays correct because `pull_fallback` makes M8 read the producer's remote
// `part` row instead of the slot the pool would have filled; its matched
// reference is mode 2 + pull_fallback, which does identical work PLUS the copy.
// Mode 8 is mode 2 with a tunable BACKOFF added to the event-queue poll, and
// nothing else. It is the follow-on the exp_20 measurements pointed at: with
// the payload proven free (mode 7) and the pool proven event-starved for most
// of M7 (the pacing fit puts only ~240 paced pushes on the drain's critical
// path), what the pool actually does next to M7 is SPIN -- 256 waves each
// issuing a relaxed load on its queue slot plus a relaxed load on the single
// shared `pperr` word every ~0.12 us. Consecutive tickets are consecutive
// 32-bit slots, so ~16 waves share each queue cache line while compute CTAs
// are concurrently writing those same lines. Unlike push pacing, backing the
// poll off costs nothing when events ARE available.
// exp_34 note: mode 14 is deliberately NOT a member. `mode_is_stream` gates
// three unrelated things -- the drain entry (KRN M7.6), the "a stream mode needs
// a pool" validator rule, and M8's dynamic-ticket loop -- and mode 14 wants only
// the last one. Adding 14 here would have required exempting it at the other two
// sites; instead M8 asks for `mode_is_stream || mode_is_coarse` directly, and
// this predicate stays byte-identical for every mode that predates exp_34.
__host__ __device__ __forceinline__ bool mode_is_stream(config c) {
    return c.mode == 2u || c.mode == 3u || c.mode == 4u || c.mode == 7u ||
           c.mode == 8u || c.mode == 9u || c.mode == 12u || c.mode == 13u
#if K0P6_MPS_ENABLE_TBO
           || c.mode == 16u   // kModeDeferCombine (declared below; literal idiom
                             // matches this predicate's other arms)
#endif
        ;
}

// exp_02 mode 16. Kept out of the mode_is_stream comment chain above for the
// same reason mode 14 asked for its own predicate: mode 16 wants the drain AND
// the dynamic-ticket M8 half of mode_is_stream's gates, and the constant is
// declared below. One predicate, one compare, used under K0P6_MPS_ENABLE_TBO
// only.
__host__ __device__ __forceinline__ bool mode_is_tbo(config c) {
#if K0P6_MPS_ENABLE_TBO
    return c.mode == 16u;    // kModeDeferCombine (declared below)
#else
    (void)c;
    return false;
#endif
}

// ---- exp_21 mode 9: the flush-fence ATTRIBUTE diagnostic (exp_06-style,
// NOT correctness-preserving) ----------------------------------------------
// Mode 9 is mode 2 with flush_pending's wave-level release_signal_batch_system
// weakened to AGENT. exp_20 §4 named flush's per-batch (~2,400/epoch)
// system-scope releases the prime interference suspect: the only mechanism
// left invisible to a rate knob, a volume knob, and per-byte cost. Weakening
// the fence drops the acquire edge the owner's M8 acquire payload read
// formally needs, so its M7 attribution is valid but its combine output is
// not trusted -- a labelled diagnostic exactly like exp_06's scope swap.
inline constexpr std::uint32_t kModeFlushAgent = 9u;

// ---- exp_21 modes 12/13: DIRECT REMOTE ACCUMULATE (A11/M11) ------------------
// (Numbered 12+: exp_20's Tier-1 diagnostics hold 4-8.) The M7 epilogue
// accumulates each tile DIRECTLY into the owner's slot with the remote
// packed-bf16 atomic exp_18 proved over xGMI: the ~936 MB part-write /
// pool-read / pool-push protocol collapses to ~312 MB of remote RMWs and the
// pool keeps only the readiness bookkeeping (arrivals, chunk counts, flags).
// The owner zeroes each slot row right after consuming it in M8
// ("consume-and-zero", made sound by the retirement gate + this harness's
// terminal-pperr semantics -- see exp_21_direct_accumulate/design.md).
//
// Modes 12/13 differ ONLY in the completion-counting scheme:
//  - mode 12 = the mode-2 scheme verbatim: nc_arr[r*16+nc] per (row, chunk)
//    with `pushed[r]` counting chunk finals to 16 (the protocol-identical
//    control: same protocol atomics as the ratchet, payload freed).
//  - mode 13 = per-row target counters: nc_arr[r] with target 16*row_rem[r],
//    deleting the `pushed` counter and the entire per-chunk completion
//    machinery (~40% of the stream protocol's agent RMWs; STATUS's live-queue
//    item 1 with exp_09's objection withdrawn). SOUND here and only here:
//    the service wave reads NO payload in remote-accumulate modes
//    (exp_20 §2a: payload is free), so per-chunk acquire edges are
//    unnecessary and the single cell's release-sequence chains all
//    contributing producers into the flagging wave's acq_rel RMW.
//
// Field discipline (exp_20 idiom -- the host bridge signature never changes):
// physical g must be 1 and the `g` field's bit 4 carries the DUAL-WRITE
// DETECTOR (g = 1 | 0x10): the epilogue also writes local `part`, and M8
// compares each consumed slot row against the producer's part row, setting
// pperr bit 1<<27 on a lost update. This is the review-mandated lost-update
// detector; rel_L1 alone cannot see it.
inline constexpr std::uint32_t kModeRemoteAccum = 12u;
inline constexpr std::uint32_t kModeDirectRows = 13u;

// ---- exp_34 mode 14: COARSE READINESS ---------------------------------------
// Mode 14 rides mode 12's remote-accumulate TRANSPORT verbatim and deletes the
// entire per-row readiness PROTOCOL. Nothing is enqueued, nothing is drained,
// no arrival counter is bumped, no per-row flag is published or polled. In its
// place M7 ends with one rank-local rendezvous (`release_cta_payload_system` +
// `hkp::grid_barrier`) followed by 8 epoch publishes from a single CTA leader
// and an 8-word bounded poll on every CTA, after which every slot on this rank
// is final by construction and M8's per-row poll compiles out.
//
// Why it MUST be mode 12's transport and not mode 2's: on mode 2 `nc_arr` is
// also the producer-side PUSH TRIGGER (`target = row_rem[r]`, then the
// completing lane pushes the group), so deleting the counting there exposes the
// whole push behind a local barrier. In mode 12 the payload already arrived in
// the producer's own epilogue RMWs and `nc_arr` is pure bookkeeping, so the
// protocol's only consumer disappears with it. See exp_30/design.md §0.
//
// STATED PLAINLY, because it is the finding and not a footnote: with the
// bookkeeping, the push and the flags all gone, the service pool has no job
// left, so at C == 0 mode 14 is a HOMOGENEOUS megakernel. Its number is banked
// as the price of the readiness protocol, not ratcheted as a role-split result.
// C > 0 remains legal and meaningful (the tail CTAs skip M7 and join only M8),
// which is the placement arm exp_37 wants; `is_service_cta` is deliberately
// left UNCHANGED so the M7 stride `nct - C` and the reservation stay consistent.
inline constexpr std::uint32_t kModeCoarseReady = 14u;

// ---- exp_02 (aug12) mode 16: DEFERRED COMBINE (TBO-2 stage A+B) -------------
// Mode 16 rides mode 12's remote-accumulate transport AND mode 12's per-row
// readiness protocol unchanged. The experiment is purely TEMPORAL: epoch i's
// combine runs inside launch i+1, claimed in the plan shadow (M4->M5 window,
// where the ratchet config leaves CTAs 2..255 with no stripe work) and in the
// regular M8 window for leftovers. Its overlap claim: M8's poll ballots all
// pass on first check (the producing epoch completed last launch), so the
// combine becomes ~330 us of unconditional load/store work absorbable into
// plan's idle capacity, and the epoch's tail loses the whole combine phase.
// Epoch 0 has no previous epoch: M8 and the retired-poke fall back to the
// same-epoch values, so generation 0 is just mode 12 semantics.
// Fields: identical grammar to modes 12/13 (physical g must be 1; the legal
// g-bits are the direct-accumulate set; detect is REJECTED -- the deferred
// detector would race non-parity `part` across generations; throttle/depth
// unchanged; flush_rows keeps its drain meaning).
inline constexpr std::uint32_t kModeDeferCombine = 16u;
inline constexpr std::uint32_t kRemoteAccumGMask = 0xFu;   // physical g bits
inline constexpr std::uint32_t kRemoteAccumDetectBit = 0x10u;
#define K0P6_MPS_ERR_DUAL 134217728    // 1<<27: dual-write detector mismatch

// ---- exp_24: two more `g`-field bits, one per independent mechanism ----------
// The throttle ENABLE bit predates this experiment (exp_21 read it raw from the
// packed word in the phase-2 body); it is given a named constant and an
// accessor here so the whole `g` encoding is single-sourced. Bit table:
//
//   0x000F  physical g                       (must be 1 in modes 12/13)
//   0x0010  dual-write lost-update detector
//   0x0020  epilogue remote-RMW throttle ENABLE
//   0x0040  exp_24 A: skip the DEAD `part` zero-fill in M5
//   0x0080  exp_34 C: mode 14 KEEPS mode 12's per-task VMEM drain
//   0x0300  exp_24 B: throttle DEPTH select, 00->8 (today) 01->4 10->16 11->32
//   0xFC00  reserved (rejected)
//
// `00 -> 8` is deliberate: the all-zero selector reproduces exp_21's ratchet
// exactly, so `g = 33` stays the same kernel path it has always been.
inline constexpr std::uint32_t kRemoteAccumThrottleBit = 0x20u;
inline constexpr std::uint32_t kRemoteAccumSkipPartZeroBit = 0x40u;
inline constexpr std::uint32_t kRemoteAccumThrottleDepthMask = 0x300u;
inline constexpr std::uint32_t kRemoteAccumThrottleDepthShift = 8u;
// ---- exp_34 C: the drain-deletion CONFOUND CONTROL bit -----------------------
// Mode 14 changes two things at once against mode 12: the readiness signal's
// granularity, and -- because the per-task event publication disappears with the
// per-row protocol -- the per-task VMEM drain that only ever ordered that
// publication (~2,840 vmcnt(0) + 2,840 __syncthreads() per CTA). A single
// mode 12 -> mode 14 waterfall rung would therefore carry two variables. This
// bit restores the deferred drain under mode 14 so the two are priced
// separately, in ONE binary, from the config word the per-task hook already
// loads: rung (i) mode 14 + 0x80 = granularity alone, rung (ii) mode 14 = plus
// the drain deletion. Legal ONLY on mode 14 (rejected below elsewhere), because
// on modes 12/13 the drain is not deleted and the bit would mean nothing.
inline constexpr std::uint32_t kCoarseKeepDrainBit = 0x80u;
// Same bit as seen in the PACKED descriptor word: encode_config puts the low
// byte of `group_slices` at bit 8, so 0x80 lands at bit 15. The per-task hooks
// test the packed word directly (they already load it for the mode field) rather
// than decoding a config struct on the hot path.
inline constexpr std::uint64_t kCoarseKeepDrainWordBit = 0x8000ull;
inline constexpr std::uint32_t kRemoteAccumGLegalBits =
        kRemoteAccumGMask | kRemoteAccumDetectBit | kRemoteAccumThrottleBit |
        kRemoteAccumSkipPartZeroBit | kRemoteAccumThrottleDepthMask
#if K0P6_MPS_ENABLE_MODE14
        | kCoarseKeepDrainBit
#endif
        ;

__host__ __device__ __forceinline__ bool mode_is_coarse(config c) {
    return c.mode == kModeCoarseReady;
}

// exp_34 C: true when mode 14 is asked to keep mode 12's per-task drain. Host
// side only (reporting / validation); the device hook reads the packed word.
__host__ __device__ __forceinline__ bool coarse_keeps_drain(config c) {
    return mode_is_coarse(c) && (c.group_slices & kCoarseKeepDrainBit) != 0u;
}

// Mode 14 is a member: it needs every direct-accumulate property (physical
// g == 1, no `part` writes, no remote-part pull, the epilogue's peer-table
// target construction in n2_phase2_gm_mps.cpp:329, the throttle accessors, the
// power-of-two MAXTOK entry guard, and M8's consume-and-zero instantiation).
// Its ONLY divergence from mode 12 is the readiness protocol.
//
// exp_38: the third term is the single highest-value line in this file to keep
// behind the flag. Every `throttle_enabled`, `detect_dual` and epilogue-target
// decision in phase 2 funnels through here, so a third comparison lengthens the
// live range of the descriptor-derived mode in the exact block that carries the
// `vmcnt` throttle. This is where the +726.9 us lived.
__host__ __device__ __forceinline__ bool mode_is_direct_accum(config c) {
    return c.mode == kModeRemoteAccum || c.mode == kModeDirectRows
#if K0P6_MPS_ENABLE_MODE14
           || c.mode == kModeCoarseReady
#endif
#if K0P6_MPS_ENABLE_TBO
           || c.mode == kModeDeferCombine
#endif
        ;
}

__host__ __device__ __forceinline__ bool detect_dual(config c) {
    return mode_is_direct_accum(c) &&
           (c.group_slices & kRemoteAccumDetectBit) != 0u;
}

__host__ __device__ __forceinline__ bool throttle_enabled(config c) {
    return mode_is_direct_accum(c) &&
           (c.group_slices & kRemoteAccumThrottleBit) != 0u;
}

// 0..3. The phase-2 epilogue turns this into one of four compile-time
// `s_waitcnt vmcnt(N)` instantiations; it is never a runtime vmcnt.
__host__ __device__ __forceinline__ std::uint32_t throttle_depth_sel(config c) {
    return (c.group_slices & kRemoteAccumThrottleDepthMask) >>
           kRemoteAccumThrottleDepthShift;
}

// exp_24 A. In mode 12 the `part` buffer (slot 21, 560 MiB) is neither read nor
// written: M7's epilogue targets the owner's `slots` through `peer_tab`, M8
// takes `part = nullptr`, and the service pool's only `part` use sits behind an
// unreachable `continue`. See exp_24_dead_plan_work/design.md §1.2 for the
// line-cited enumeration. The M5 zero of all T_ext rows is therefore 448 MiB of
// stores per rank per epoch that nothing consumes.
//
// The guard is deliberately narrow and FAIL-CLOSED rather than best-effort:
// `config_is_valid` REJECTS the bit outside mode 12 and rejects it together
// with the dual-write detector (whose local `part` tower needs the zero), so a
// wrong config is refused instead of silently keeping or silently dropping the
// zero. Mode 13 satisfies the same deadness proof but stays out of scope until
// a mode-13 run re-derives it.
//
// exp_34: mode 14 is admitted because it satisfies the same deadness proof by
// construction -- it IS mode 12's transport, M8 takes `part = nullptr` on the
// same branch, and the dual-write detector (the one `part` consumer) is rejected
// for mode 14 below. This is not a convenience: the mode-14 rung must be able to
// carry the ratchet's `g = 353`, or the waterfall comparison against mode 12
// would silently also be an exp_24-A comparison.
__host__ __device__ __forceinline__ bool skip_dead_part_zero(config c) {
#if K0P6_MPS_ENABLE_MODE14
    return (c.mode == kModeRemoteAccum || mode_is_coarse(c)
#if K0P6_MPS_ENABLE_TBO
            || mode_is_tbo(c)
#endif
            ) &&
#else
#if K0P6_MPS_ENABLE_TBO
    return (c.mode == kModeRemoteAccum || mode_is_tbo(c)) &&
#else
    return c.mode == kModeRemoteAccum &&
#endif
#endif
           (c.group_slices & kRemoteAccumSkipPartZeroBit) != 0u &&
           !detect_dual(c) && !c.pull_fallback;
}

// exp_24 A, the surviving half. `hkp::zero_part_scale_transpose<Chunks1K>`
// (hkp_quant.hpp:130-143, OUTSIDE this repo and read-only to us) fuses two
// unrelated jobs into one per-row body: 14 KiB of `part` zero stores, and the
// row-major -> group-major scale transpose that writes `sc_dst`. Mode 12 needs
// the second and provably not the first, so this is the second on its own,
// byte-identical to hkp_quant.hpp:142 including the `lane < ng` predicate and
// the `sc_dst[lane * T_loc + t]` addressing that M6 reads (KRN:1321).
//
// It lives here rather than in hkp_quant.hpp because that header is not ours to
// edit. In hindsight the primitive should have been two composable pieces --
// `zero_row` and `scale_transpose_row` -- with the fused form as their
// composition; see exp_24_dead_plan_work/result.md `## Primitives`.
__device__ __forceinline__ void scale_transpose_row(float* sc_dst,
                                                    const float* sc_stage_row,
                                                    int t, int T_loc, int ng,
                                                    int lane) {
    if (lane < ng) {
        sc_dst[(std::size_t)lane * (std::size_t)T_loc + t] = sc_stage_row[lane];
    }
}

__host__ __device__ __forceinline__ std::uint32_t physical_g(config c) {
    return mode_is_direct_accum(c) ? (c.group_slices & kRemoteAccumGMask)
                                   : c.group_slices;
}

// Modes that reserve a pool but keep mode 0's publication and combine paths.
__host__ __device__ __forceinline__ bool mode_is_diag_pool(config c) {
    return c.mode == 5u || c.mode == 6u;
}

// Who runs the parity M7.5 bulk publication and the parity M8 remote pull.
__host__ __device__ __forceinline__ bool mode_is_parity_publish(config c) {
    return c.mode == 0u || mode_is_diag_pool(c);
}

__host__ __device__ __forceinline__ std::uint32_t effective_flush_rows(config c) {
    return (c.mode == 4u || c.mode == 8u) ? 16u : c.flush_rows;
}

__host__ __device__ __forceinline__ std::uint32_t pace_units(config c) {
    return (c.mode == 4u) ? (c.flush_rows - 1u) : 0u;
}

__host__ __device__ __forceinline__ std::uint32_t poll_backoff_units(config c) {
    return (c.mode == 8u) ? (c.flush_rows - 1u) : 0u;
}

__host__ __device__ __forceinline__ std::uint64_t diag_window_ticks(config c) {
    return (std::uint64_t)c.flush_rows * (std::uint64_t)kDiagTicksPer100us;
}

__host__ __device__ __forceinline__ bool config_is_valid(config c) {
    // exp_13: cap raised 64 -> 128. Before compute-CTA fall-through (exp_12)
    // the service cost obeyed a 1/C law and the tax curve made large C strictly
    // worse, so the cap was left alone deliberately. Fall-through inverts that:
    // the drain is no longer pool-limited, and larger C now buys more push
    // overlapped with M7 (combine falls 1,646 -> 649 us from C=4 to C=64) for a
    // net win at every point measured. 128 of 256 is 50%, above COMET's
    // 14-35% band, so this is an extrapolation the sweep has to justify.
    if (c.reserved_comm_ctas > 128u) return false;
    if (c.reserved_comm_ctas >= 256u) return false;
    if (mode_is_direct_accum(c)) {
        // exp_21 modes 12/13: physical g must be 1 (there are no push groups)
        // and every extra bit must be one this build knows how to honour.
        if ((c.group_slices & kRemoteAccumGMask) != 1u) return false;
        if ((c.group_slices & ~kRemoteAccumGLegalBits) != 0u) return false;
        // Part is never written in modes 12/13, so a remote-part pull has
        // nothing to read: forbid rather than silently corrupt.
        if (c.pull_fallback) return false;
        // exp_24 B: a depth selector without the throttle enabled selects
        // nothing. Reject instead of accepting a config that reads as swept.
        if ((c.group_slices & kRemoteAccumThrottleDepthMask) != 0u &&
            (c.group_slices & kRemoteAccumThrottleBit) == 0u) return false;
        // exp_24 A: the skip bit is honoured ONLY by mode 12's M5, and the
        // dual-write detector accumulates into `part` and needs the zero. Both
        // are rejections, so a config either engages the mechanism or fails --
        // it is never silently ignored.
        if ((c.group_slices & kRemoteAccumSkipPartZeroBit) != 0u) {
#if K0P6_MPS_ENABLE_MODE14
            if (c.mode != kModeRemoteAccum && !mode_is_coarse(c)
#if K0P6_MPS_ENABLE_TBO
                && !mode_is_tbo(c)
#endif
                ) return false;
#else
#if K0P6_MPS_ENABLE_TBO
            if (c.mode != kModeRemoteAccum && !mode_is_tbo(c)) return false;
#else
            if (c.mode != kModeRemoteAccum) return false;
#endif
#endif
            if ((c.group_slices & kRemoteAccumDetectBit) != 0u) return false;
        }
#if K0P6_MPS_ENABLE_MODE14
        // exp_34: mode 14 deletes the per-row `row_ready` protocol, and the
        // dual-write detector's M8 instantiation is the one slot-mode path that
        // still POLLS it (KRN's m7_detect branch is tested before the
        // direct-accumulate branch, so a mode-14 + detect config would take it
        // and spin on a flag nobody publishes). Reject rather than hang.
        if (mode_is_coarse(c) &&
            (c.group_slices & kRemoteAccumDetectBit) != 0u) return false;
        // exp_34 C: the drain-retention control has no meaning where the drain
        // was never deleted. Reject on 12/13 instead of silently ignoring it --
        // otherwise a mistyped waterfall arm would LOOK like the control arm.
        if ((c.group_slices & kCoarseKeepDrainBit) != 0u &&
            !mode_is_coarse(c)) return false;
#endif
#if K0P6_MPS_ENABLE_TBO
        // exp_02 mode 16: same rejection shape as mode 14's -- the dual-write
        // detector's M8 instantiation POLLS the per-row flags AND pulls the
        // producer's `part` tower. Under deferral `part` is NOT parity-indexed
        // (nothing else writes it), so a deferred detector would compare epoch
        // i-1's slot against epoch i's `part` -- a silent mix. Reject, exactly
        // as mode 14 does.
        if (mode_is_tbo(c) &&
            (c.group_slices & kRemoteAccumDetectBit) != 0u) return false;
#endif
    } else if (c.group_slices != 1u && c.group_slices != 2u &&
               c.group_slices != 4u && c.group_slices != 16u) return false;
#if K0P6_MPS_ENABLE_TBO
    if (c.mode > 16u) return false;
#elif K0P6_MPS_ENABLE_MODE14
    if (c.mode > 14u) return false;
#else
    if (c.mode > 13u) return false;
#endif
    // Mode 7 moves no payload into the slots, so M8 must take the pull path.
    if (c.mode == 7u && !c.pull_fallback) return false;
    if (mode_is_stream(c) && c.reserved_comm_ctas == 0u) return false;
    // exp_20 diagnostics: a pool of zero has nothing to run.
    if (mode_is_diag_pool(c) && c.reserved_comm_ctas == 0u) return false;
    // Mode 5 reads `g` as the traffic-variant selector, and 16 is not one.
    if (c.mode == 5u && c.group_slices == 16u) return false;
    // Mode 3 hands out whole dies, so the pool must be a whole number of them.
    if (c.mode == 3u && (c.reserved_comm_ctas % kCtasPerXcd) != 0u) return false;
    // Mode 1 (bulk, no overlap) strides the full grid: the M7 stride derives
    // from C everywhere, so mode 1 must carry C == 0.
    if (c.mode == 1u && c.reserved_comm_ctas != 0u) return false;
    if (c.flush_rows == 0u || c.flush_rows > 64u) return false;
    return true;
}

// Role predicate and dense role ids. Modes 0-2 keep the static-tail formula
// byte-for-byte; mode 3 selects by XCD residue class. The dense compute id is
// what the M7 task loop strides with, so both branches must produce a dense
// [0, nct - C) numbering or tasks would be skipped or run twice.
__device__ __forceinline__ bool is_service_cta(config c, int bid, int nct) {
    if (c.mode == 3u) {
        return ((std::uint32_t)bid & (kXcds - 1u)) < service_dies(c);
    }
    return bid >= nct - (int)c.reserved_comm_ctas;
}

__device__ __forceinline__ int compute_id_of(config c, int bid) {
    if (c.mode == 3u) {
        const std::uint32_t d = service_dies(c);
        return (int)(((std::uint32_t)bid >> 3) * (kXcds - d) +
                     (((std::uint32_t)bid & (kXcds - 1u)) - d));
    }
    return bid;
}

__device__ __forceinline__ std::uint32_t service_id_of(config c, int bid,
                                                       int nct) {
    if (c.mode == 3u) {
        const std::uint32_t d = service_dies(c);
        return ((std::uint32_t)bid >> 3) * d +
               ((std::uint32_t)bid & (kXcds - 1u));
    }
    return (std::uint32_t)(bid - (nct - (int)c.reserved_comm_ctas));
}

// ---- tile-event wire format ---------------------------------------------------
// One 32-bit slot per (b, nc). The MARK bit makes zero safely mean "empty";
// b < 2^14 is enforced by the kernel shape/config guard (bcap = PADMAX/32).
inline constexpr std::uint32_t kEventMark = 0x80000000u;

__host__ __device__ __forceinline__ std::uint32_t encode_event(int b, int nc) {
    return kEventMark | ((std::uint32_t)b << 4) | (std::uint32_t)nc;
}

__host__ __device__ __forceinline__ int event_block(std::uint32_t word) {
    return (int)((word >> 4) & 0x3FFFu);
}

__host__ __device__ __forceinline__ int event_chunk(std::uint32_t word) {
    return (int)(word & 0xFu);
}

// ---- device diagnostics -------------------------------------------------------
__device__ __forceinline__ std::uint64_t realtime_now() {
#if defined(__HIP_DEVICE_COMPILE__)
    std::uint64_t t;
    // `s_memrealtime` is an SMEM instruction: its destination SGPR pair is NOT
    // valid until `s_waitcnt lgkmcnt(0)`. Without the wait the caller reads
    // whatever the register happened to hold, so sites that follow LDS traffic
    // (which carries its own lgkmcnt wait) read a real clock while sites that
    // do not read garbage. That is exactly the observed failure: LAST_READY and
    // M7_DONE carried real clocks while DRAIN, REDUCE_DONE, M2_DONE and
    // M6_DONE all read 1. Destination is a scalar pair, so the constraint is
    // "s", not "r".
    asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)"
                 : "=s"(t) :: "memory");
    return t;
#else
    return 0;
#endif
}

// min-timestamp: store max(~t) so the M0 zeroing (all-zero) is a valid floor and
// no special init value is required. Recover with ~cell.
__device__ __forceinline__ void ts_first(std::uint64_t* cell, bool enable) {
    if (!enable || cell == nullptr) return;
    atomicMax(reinterpret_cast<unsigned long long*>(cell),
              ~realtime_now());
}

__device__ __forceinline__ void ts_last(std::uint64_t* cell, bool enable) {
    if (!enable || cell == nullptr) return;
    atomicMax(reinterpret_cast<unsigned long long*>(cell), realtime_now());
}

// ---- exp_23: per-CTA phase event ring, accessors -------------------------------
// Base of the ring: immediately past the 8 coarse uint64 cells.
__device__ __forceinline__ std::uint64_t* e23_ring(std::uint64_t* ts_base) {
    return (ts_base == nullptr) ? nullptr : (ts_base + K0P6_MPS_TS_COUNT);
}

// Plain 8 B store: not atomic, not volatile, no fence. Every call site is under
// `tid == 0`, so (CTA, slot) has exactly one writer, and the only reader is the
// host after the launch joins -- the join is the release edge.
__device__ __forceinline__ void e23_mark(std::uint64_t* ts_base, int bid,
                                         int slot, std::uint64_t t,
                                         bool enable) {
#if K0P6_MPS_E23_RING
    if (!enable || ts_base == nullptr) return;
    if ((unsigned)bid >= (unsigned)K0P6_MPS_E23_MAX_CTA) return;
    e23_ring(ts_base)[bid * K0P6_MPS_E23_SLOTS + slot] = t;
#else
    (void)ts_base; (void)bid; (void)slot; (void)t; (void)enable;
#endif
}

// FUSED coarse + per-CTA stamp. THE WHOLE POINT IS THE SINGLE CLOCK READ: one
// `t` feeds both the coarse atomicMax cell and this CTA's ring cell, which
// makes `max over CTAs (ring cell) == coarse cell` an identity rather than an
// approximation, and that identity is what proves the array was indexed into
// the right cell in the right epoch.
//
// DO NOT rewrite a call site as `ts_last(cell, e); e23_mark(base, ...)`. That
// compiles, it passes the resource tuple, the figure still looks right -- and
// the two cells now come from two different clock reads, which silently demotes
// the reconciliation check. Gate G7 (an `s_memrealtime` census in the ISA:
// tier A must add exactly zero) exists solely to catch that substitution.
//
// exp_38: with the ring compiled OUT this must degenerate to EXACTLY rev 26's
// `ts_last(cell, enable)` -- not to something equivalent-looking. The nullptr
// test and the named `t` are part of the fused form and both are absent from
// rev 26, so the ring-off definition forwards rather than sharing a body. That
// is what makes the default build's `.text` match rev 26 byte for byte with the
// call sites left in their fused form.
#if K0P6_MPS_E23_RING
__device__ __forceinline__ void ts_mark(std::uint64_t* cell,
                                        std::uint64_t* ts_base, int bid,
                                        int slot, bool enable) {
#if K0P6_MPS_E23_FORCE_ON
    (void)enable;
    const bool en = true;
#else
    const bool en = enable;
#endif
    if (!en) return;
    const std::uint64_t t = realtime_now();
    if (cell != nullptr) {
        atomicMax(reinterpret_cast<unsigned long long*>(cell),
                  (unsigned long long)t);
    }
    e23_mark(ts_base, bid, slot, t, en);
}
#else
__device__ __forceinline__ void ts_mark(std::uint64_t* cell,
                                        std::uint64_t* ts_base, int bid,
                                        int slot, bool enable) {
    (void)ts_base; (void)bid; (void)slot;
    ts_last(cell, enable);
}
#endif

// Tier C metadata cell: (role << 32) | per-CTA task count, in slot 15. Not a
// timestamp, so it does not go through ts_mark and reads no clock.
__device__ __forceinline__ void e23_meta(std::uint64_t* ts_base, int bid,
                                         std::uint32_t role,
                                         std::uint32_t count, bool enable) {
    e23_mark(ts_base, bid, K0P6_MPS_E23_META,
             ((std::uint64_t)role << 32) | (std::uint64_t)count, enable);
}

// ---- producer side: M7 task-done hook body -------------------------------------
// Release-then-publish of this CTA's drained block-chunk completion. Finding
// (exp_56 M1 discipline): the per-thread `s_waitcnt vmcnt(0)` drain + CTA
// barrier runs in the phase-2 body's N2GM_TASK_DONE_DRAIN_HOOK immediately
// before this hook, so tid0's agent release carries the whole CTA's part
// atomics. No system fence and no row loop ever execute on the GEMM CTA.
// Readiness timestamps (first/last-ready) are captured by the SERVICE waves
// consuming these events, keeping the memory-clobbering s_memrealtime asm
// entirely off the MFMA wave (resource gate).
// EnqueueScope: exp_21 mode 12 releases at SYSTEM scope, because the payload
// now lives in the OWNER's memory: the chain that publishes it (event store ->
// service acquire -> flush's system release -> owner acquire) is only as
// strong as its first link. Modes 2/3/4/7 keep agent (unchanged ratchet arm).
template<scope EnqueueScope = scope::agent>
__device__ __forceinline__ void enqueue_tile_release(
        std::uint32_t* __restrict__ q, std::uint32_t* __restrict__ tail,
        int b, int nc) {
    const std::uint32_t ticket =
        kittens::distributed::fetch_add_relaxed<scope::agent>(tail, 1u);
    kittens::distributed::publish_tile_release<EnqueueScope>(
        q + ticket, encode_event(b, nc));
}

// ---- service side --------------------------------------------------------------
struct service_env {
    const std::uint32_t* __restrict__ q;
    std::uint32_t* __restrict__ nc_arr;      // [T_ext*16]
    std::uint32_t* __restrict__ pushed;      // [T_ext]
    std::uint32_t* __restrict__ claim;       // [T_ext]
    std::uint32_t* __restrict__ ev_next;     // exp_12: event consumption ticket
    std::uint32_t* __restrict__ row_rem;     // [T_ext] (M2 popcount; self-clean)
    const int* __restrict__ sti;             // sorted token ids
    const unsigned short* __restrict__ part; // [T_ext, 7168] bf16 (local)
    unsigned short* __restrict__ slots;      // [world, MAXTOK, 7168] bf16
    std::uint32_t* __restrict__ row_ready;   // [world*T_loc_max]
    int* pperr;
    std::uint64_t* ts;                       // [8] diag block (or nullptr)
    const symmetric_heap_descriptor* symmetric;
    int cur;
    int world;
    int maxtok;
    int t_ext;
    int t_loc_max;
    std::uint32_t epoch32;
    std::uint32_t group_slices;              // g in {1,2,4,16} (physical)
    std::uint32_t flush_rows;                // flags per system release
    std::uint32_t pace;                      // exp_20 mode 4: s_sleep units/push
    std::uint32_t poll_backoff;              // exp_20 mode 8: s_sleep units/spin
    bool no_payload;                         // exp_20 mode 7: skip the copy
    std::uint32_t mode;                      // exp_21: 12/13 = remote accumulate
    std::uint64_t spin_limit;
    std::uint32_t events_total;              // E = (nvi[0] >> 5) * 16
    std::uint32_t queue_capacity;            // (PADMAX/32)*16
    bool ts_enable;
};

// exp_20 -- a delay that issues NO memory request. `s_sleep N` idles the wave
// for ~64*N core clocks, so a loop of these lowers a wave's request RATE while
// leaving its total request VOLUME untouched. Two callers, both diagnostic
// knobs on a single mode: mode 4 paces the payload push (E1: is the
// interference rate-driven or volume-driven?), mode 8 backs off the
// event-queue poll. One unit is `s_sleep 4` = 256 clocks ~= 0.12 us at 2.1 GHz.
__device__ __forceinline__ void pace_delay(std::uint32_t units) {
#if defined(__HIP_DEVICE_COMPILE__)
    for (std::uint32_t i = 0; i < units; ++i) {
        asm volatile("s_sleep 4");
    }
#else
    (void)units;
#endif
}

// Leader-polled, pperr-watching bounded wait on one queue slot. Returns the
// decoded event word, or 0 on failure (pperr bit already recorded). Word 0 is
// never a real event because of the MARK bit.
//
// `backoff` (mode 8 only, 0 everywhere else) adds idle time between spins. Each
// spin costs two relaxed global loads -- the queue slot, and the single shared
// `pperr` word that all 256 draining waves poll -- so the spin's aggregate
// request rate is set here and nowhere else.
__device__ __forceinline__ std::uint32_t wait_event_nonempty(
        const std::uint32_t* slot, int* pperr, std::uint64_t spin_limit,
        std::uint32_t backoff) {
    std::uint64_t spins = 0;
    while (true) {
        const std::uint32_t v = load_relaxed<scope::agent>(slot);
        if (v != 0u) return v;
        const int err = load_relaxed<scope::agent>(pperr);
        if ((err & (16777216 | K0P6_MPS_ERR_SERVICE)) != 0) return 0u;
        if (++spins > spin_limit) {
            atomicOr(pperr, K0P6_MPS_ERR_SERVICE);
            return 0u;
        }
        kittens::distributed::detail::pause();
        pace_delay(backoff);
    }
}

struct wave_scratch {
    std::uint32_t rows[32];        // rows with a group push from this event
    std::uint32_t group_base[32];  // first nc of the claimed group
    std::uint32_t push_count;
    std::uint32_t flag_rows[64];   // completed rows pending publication
    std::uint32_t flag_count;
    std::uint32_t event;           // leader-polled event word (0 = failed)
    std::uint32_t ticket;          // exp_12: claimed event index
};

static_assert(sizeof(wave_scratch) == 528,
              "wave_scratch must stay inside the reused M1 staging slice");

inline constexpr std::size_t kSliceBytes = 448u * 2u;   // 896
// How many claimed row groups the service wave copies concurrently. A single
// group is 896*g bytes = at most 56*g packets, so at g=1 it does not even give
// the wave's 64 lanes one packet each and no pipelining is possible WITHIN a
// group. Batching whole groups is the only way to give a lane more than one
// load in flight. See exp_03.
inline constexpr unsigned int kPushBatch = 4u;

// Source (this rank's `part` row slice) and destination (the owner's landing
// slot, local or peer-translated) of one claimed group. Split out of
// push_slice_group so a wave can compute several groups' addresses before
// issuing any of their loads.
__device__ __forceinline__ const void* slice_group_src(
        const service_env& env, std::uint32_t r, std::uint32_t group_base) {
    return reinterpret_cast<const std::byte*>(env.part) +
        ((std::size_t)r * 7168u + (std::size_t)group_base * 448u) * 2u;
}

__device__ __forceinline__ void* slice_group_dst(
        const service_env& env, std::uint32_t r, std::uint32_t group_base) {
    const std::uint32_t owner = r / (std::uint32_t)env.maxtok;
    const std::uint32_t pos = r % (std::uint32_t)env.maxtok;
    std::byte* dst_local = reinterpret_cast<std::byte*>(
        env.slots + ((std::size_t)env.cur * (std::size_t)env.maxtok + pos) *
                        7168u + (std::size_t)group_base * 448u);
    return (owner == (std::uint32_t)env.cur)
        ? dst_local
        : hk_moe::peer_ptr(dst_local, (int)owner, env.symmetric);
}

// Push one claimed group of adjacent N-chunk slices for row r to its owner's
// landing slot. Controlled 16-byte stores, coalesced by the wave's 64 lanes;
// no fence, no completion work here (flush_pending owns both).
__device__ __forceinline__ void push_slice_group(const service_env& env,
        std::uint32_t r, std::uint32_t group_base, int lane) {
    // exp_17 tried the streaming (nt) form here and it was a measured NULL --
    // and the `nt` bits provably reached the ISA (`flat_load_dwordx4 ... nt`,
    // `flat_store_dwordx4 ... nt`), so the hint was applied and simply did not
    // help. Conclusion: the pool's interference with the concurrent GEMM is
    // BANDWIDTH contention, not cache-capacity pollution, and no replacement
    // policy hint can move it. That also kills the M1/M2 cache-bit class for
    // this problem. The only remaining lever is moving fewer bytes -- see
    // exp_17_streaming_copy/result.md.
    kittens::distributed::store_peer_packets(
        slice_group_dst(env, r, group_base),
        slice_group_src(env, r, group_base),
        kSliceBytes * env.group_slices, (unsigned int)lane, 64u);
}

// Push the kPushBatch claimed groups starting at `first` together: every
// group's load is issued before any group's store, so each lane holds
// kPushBatch loads in flight instead of one. Byte-for-byte the same traffic as
// kPushBatch calls to push_slice_group, in an unspecified order between groups
// (they are disjoint rows). The row/base reads come straight from LDS and the
// pointer arrays are indexed only under `#pragma unroll`, so nothing here is
// dynamically indexed in registers.
__device__ __forceinline__ void push_slice_group_batch(const service_env& env,
        const wave_scratch& s, std::uint32_t first, int lane) {
    void* dsts[kPushBatch];
    const void* srcs[kPushBatch];
#pragma unroll
    for (unsigned int b = 0; b < kPushBatch; ++b) {
        const std::uint32_t r = s.rows[first + b];
        const std::uint32_t gb = s.group_base[first + b];
        dsts[b] = slice_group_dst(env, r, gb);
        srcs[b] = slice_group_src(env, r, gb);
    }
    kittens::distributed::store_peer_packets_multi<kPushBatch>(
        dsts, srcs, kSliceBytes * env.group_slices, (unsigned int)lane, 64u);
}

// Publish every completed-row flag accumulated by this wave behind ONE system
// release (the same-thread fence-then-store discipline of the parity M7.5
// publisher, at wave granularity). The row's unique publisher also restores
// the exp_62 row_remaining self-clean invariant. Whole-wave convergent call.
__device__ __forceinline__ void flush_pending(const service_env& env,
        wave_scratch& s, int lane) {
    if (s.flag_count == 0u) return;
    // Wave-granular producer drain: vmcnt counters are per-wavefront, so each
    // lane's s_waitcnt covers the whole wave's outstanding packet stores.
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
    __syncwarp();
    if (lane == 0) {
        // exp_21 mode 9 (diagnostic, exp_06-style): the flush's system release
        // weakened to agent to attribute its ~2,400/epoch cost. NOT trusted
        // for combine correctness.
        if (env.mode == kModeFlushAgent) {
            hk_moe::release_signal_batch_agent();
        } else {
            hk_moe::release_signal_batch_system();
        }
        for (std::uint32_t i = 0; i < s.flag_count; ++i) {
            const std::uint32_t r = s.flag_rows[i];
            const std::uint32_t owner = r / (std::uint32_t)env.maxtok;
            std::uint32_t* ready_slot =
                env.row_ready + (std::size_t)env.cur *
                                    (std::size_t)env.t_loc_max + r;
            if (owner == (std::uint32_t)env.cur) {
                hk_moe::publish_epoch<scope::agent>(ready_slot, env.epoch32);
            } else {
                hk_moe::publish_epoch<scope::system>(
                    hk_moe::peer_ptr(ready_slot, (int)owner, env.symmetric),
                    env.epoch32);
            }
            // exp_62 self-clean invariant: every published row ends the epoch
            // at row_remaining == 0. Unique writer: this wave alone flagged r.
            hk_moe::publish_relaxed<scope::agent>(env.row_rem + r, 0u);
        }
        s.flag_count = 0u;
    }
    __syncwarp();
}

// Account one pushed group against its row, and when all 16 of the row's
// slices have been pushed, queue the row's completed-row flag.
//
// The flush test follows EVERY append (not once per event): an event may append
// up to 32 rows, so a per-event test could let flag_count overshoot
// flag_rows[64] whenever flush_rows > 32. flag_count stays <= flush_rows <= 64
// under this discipline. Whole-wave convergent call.
__device__ __forceinline__ void retire_pushed_row(const service_env& env,
        wave_scratch& s, std::uint32_t rr, std::uint32_t g, int lane) {
    if (lane == 0) {
        const std::uint32_t oldp =
            kittens::distributed::fetch_add_relaxed<scope::agent>(
                env.pushed + rr, g);
        if (oldp + g == 16u) {
            s.flag_rows[s.flag_count] = rr;
            ++s.flag_count;
        }
    }
    __syncwarp();
    if (s.flag_count >= env.flush_rows) flush_pending(env, s, lane);
}

// The minimum-progress service loop: wave-striped consumption of the tile
// event queue on reserved CTAs. Failure exits the loop; the bounded waits
// elsewhere (owner polls, M9 arrival) make the overall failure terminal.
// `s` is this wave's private scratch slice (LDS), carved by the caller.
// exp_12: the event loop claims a TICKET instead of walking a static
// (service_id, service_count) stripe. Three things follow, and all three are
// motivated by measurement:
//
//  * **Any CTA can join.** A static stripe fixes the pool at launch and makes
//    exactly-once consumption depend on the pool size, so a late joiner would
//    double-count arrivals. A ticket makes the pool size irrelevant, which is
//    what lets compute CTAs fall through into the drain after M7 — and the
//    profile says the ENTIRE capacity tax is M7's (+27% at C=64), so refunding
//    M7's CTAs at the tail is the single largest tax we can give back.
//  * **It consumes in readiness order.** Producers enqueue with a monotonic
//    tail ticket, so the queue fills densely in completion order. Claiming the
//    next unclaimed slot therefore takes the next event that will be ready,
//    where the stripe forced each wave to wait for ITS events in index order —
//    head-of-line blocking behind a slot whose producer had not run yet.
//  * **It costs one relaxed atomic per event** (~16,720 per rank per epoch)
//    against the 535,040 the arrival counters already spend, so it is free at
//    the resolution we can measure.
__device__ __forceinline__ void run_service(
        const service_env& env, wave_scratch& s, int wid, int lane) {
    (void)wid;
    const std::uint32_t g = env.group_slices;
    if (lane == 0) {
        s.push_count = 0u;
        s.flag_count = 0u;
    }
    __syncwarp();
    while (true) {
        if (lane == 0) {
            s.ticket = kittens::distributed::fetch_add_relaxed<scope::agent>(
                env.ev_next, 1u);
        }
        __syncwarp();
        const std::uint32_t k = s.ticket;
        if (k >= env.events_total) break;
        if (lane == 0) {
            s.event = wait_event_nonempty(env.q + k, env.pperr, env.spin_limit,
                                          env.poll_backoff);
        }
        __syncwarp();
        const std::uint32_t ev = s.event;
        if (ev == 0u) break;
        // Wave-local payload acquire (a CTA-scope cta_acquire would deadlock:
        // the waves of one CTA iterate different stripe lengths). exp_21 keeps
        // this at AGENT scope for ALL modes including 12/13: exp_20's §4 puts
        // fence count at the top of the interference suspect list, and the
        // remote-accumulate chain needs no per-event system acquire -- the
        // acq_rel counting cells chain producer causality into the flagging
        // wave and the OWNER's acquire_payload_system closes the device
        // boundary (see k0p6_mps_task_done's scope note).
        kittens::distributed::thread_acquire<scope::agent>();
        // Readiness timestamps live on the SERVICE wave (off the MFMA wave):
        // first/last tile-event observed. Service waves already poll at the
        // queue tail, so this is the earliest observable readiness point
        // without instrumenting producer CTAs.
        if (lane == 0 && env.ts != nullptr) {
            ts_first(env.ts + K0P6_MPS_TS_FIRST_READY, env.ts_enable);
            ts_last(env.ts + K0P6_MPS_TS_LAST_READY, env.ts_enable);
        }
        const int b = event_block(ev);
        const int nc = event_chunk(ev);
        const int group_base = nc & ~(int)(g - 1u);
        // Lanes 0..31 account for the 32 sorted rows of block b.
        int r = 0x7FFFFFFF;
        bool live = false;
        std::uint32_t target = 0u;
        if (lane < 32) {
            r = env.sti[(std::size_t)b * 32 + (std::size_t)lane] & 0x00FFFFFF;
            live = r < env.t_ext;
            if (live) {
                target = kittens::distributed::load_relaxed<scope::agent>(
                    env.row_rem + r);
            }
        }
        bool push_lead = false;
        if (live) {
            // exp_06 measured what this site's SCOPE costs: relaxing it to
            // fetch_add_relaxed recovered 354 us of the 1,159 us M7 interference
            // floor at g=1 (157 us at g=16), i.e. ~30%. The other ~70% is the
            // atomic's 32-scattered-cache-lines-per-event footprint. The relaxed
            // form is NOT correctness-preserving (it drops the acquire edge the
            // probe loop below relies on) and was reverted; the measurement is
            // the justification for M4, whose per-XCD counters cut scope and
            // footprint together. See exp_06_atomic_scope/result.md.
            if (env.mode == kModeDirectRows) {
                // exp_21 mode 13: ONE counter per row, target 16*row_rem.
                // Deletes the per-chunk cells AND the `pushed` RMW. exp_09's
                // old objection ("fewer lines per event") is withdrawn: the 32
                // live lanes of an event address 32 DIFFERENT rows, so exp_07's
                // essential line-spread is preserved. Soundness: in a
                // remote-accumulate mode the service wave reads NO payload, so
                // no per-chunk acquire edge is needed; the last arriver's
                // acq_rel RMW on the one cell is in the release sequence of
                // every contributing producer's arrival RMW on that same cell,
                // chaining all 16 chunks' causalities into this wave's flush.
                const std::uint32_t old =
                    kittens::distributed::detail::fetch_add_acq_rel<scope::agent>(
                        env.nc_arr + (std::size_t)r, 1u);
                if (old + 1u == target * 16u) push_lead = true;
            } else {
            const std::uint32_t old =
                kittens::distributed::detail::fetch_add_acq_rel<scope::agent>(
                    env.nc_arr + (std::size_t)r * 16u + (std::size_t)nc, 1u);
            if (old + 1u == target) {
                // Slice (r,nc) final. Group-check with acq_rel RMW probes so
                // (a) the check itself is in the per-counter RMW order: the
                // *last* completing bumper of the group provably observes
                // every other chunk at target; and (b) the pushing wave holds
                // an acquire edge per chunk of the group before it reads that
                // chunk's part payload — including chunks whose events were
                // consumed by OTHER service waves.
                if (g == 1u) {
                    // exp_14: at g == 1 the group IS this single chunk, so the
                    // probe and the claim are PROVABLY redundant and skipped.
                    // (a) The probe would re-read the very counter whose
                    // fetch_add just returned `target - 1`, so it can only
                    // return `target`; and this wave already holds the acquire
                    // edge for that chunk from its own acq_rel above, with no
                    // other chunk in the group left to acquire. (b) The claim
                    // dedupes the pusher, but RMW order makes `old + 1 ==
                    // target` true for exactly ONE lane grid-wide, so the
                    // claimant is already unique.
                    //
                    // Deletes 2 of the 4 atomics per completing slice
                    // (~698,112 of ~1.58M per rank per epoch), aimed straight
                    // at the measured cause of M7's interference. g > 1 keeps
                    // the original path byte-for-byte.
                    push_lead = true;
                } else {
                    bool all = true;
                    for (std::uint32_t n2 = (std::uint32_t)group_base;
                         n2 < (std::uint32_t)group_base + g; ++n2) {
                        const std::uint32_t cur_count =
                            kittens::distributed::detail::fetch_add_acq_rel<scope::agent>(
                                env.nc_arr + (std::size_t)r * 16u + n2, 0u);
                        if (cur_count != target) { all = false; break; }
                    }
                    if (all) {
                        const std::uint32_t claim_bit =
                            1u << ((std::uint32_t)group_base / g);
                        // Plain atomicOr dedupes the claimant (same idiom as
                        // the kernel's pperr stores): agent-domain RMW, no
                        // ordering role — the acquire edge is carried by the
                        // probes above.
                        const std::uint32_t old_claim =
                            atomicOr(env.claim + r, claim_bit);
                        push_lead = (old_claim & claim_bit) == 0u;
                    }
                }
            }
            }
        }
        const unsigned long long ballot = __ballot(push_lead);
        if (lane == 0) s.push_count = 0u;
        __syncwarp();
        unsigned long long pending = ballot;
        while (pending != 0u) {
            const int src = __ffsll((long long)pending) - 1;
            pending &= pending - 1u;
            if (lane == src) {
                const std::uint32_t idx = atomicAdd(&s.push_count, 1u);
                s.rows[idx] = (std::uint32_t)r;
                s.group_base[idx] = (std::uint32_t)group_base;
            }
        }
        __syncwarp();
        const std::uint32_t npush = s.push_count;
        if (env.mode == kModeRemoteAccum
#if K0P6_MPS_ENABLE_TBO
            || env.mode == kModeDeferCombine
#endif
            ) {
            // exp_21 mode 12: bookkeeping only. The completing lane of each
            // (row, chunk) counter counts the CHUNK toward the row's 16; the
            // payload reached the owner in the producer's own epilogue RMWs.
            // `pushed` therefore reads "chunks done", target unchanged (16),
            // and retire_pushed_row's existing flush path is exact. No claim:
            // g is pinned 1, and the RMW order makes the chunk's completing
            // lane unique exactly as in stream mode's g==1 arm.
            for (std::uint32_t i = 0; i < npush; ++i) {
                retire_pushed_row(env, s, s.rows[i], 1u, lane);
            }
            continue;
        }
        if (env.mode == kModeDirectRows) {
            // exp_21 mode 13: the nc_arr RMW already told us this ROW is fully
            // arrived (single-cell ordering through all 16 chunks' producers),
            // so the completed rows flag DIRECTLY -- no `pushed` counter at
            // all. The per-row flush discipline below mirrors
            // retire_pushed_row's exactly.
            for (std::uint32_t i = 0; i < npush; ++i) {
                if (lane == 0) {
                    s.flag_rows[s.flag_count] = s.rows[i];
                    ++s.flag_count;
                }
                __syncwarp();
                if (s.flag_count >= env.flush_rows) flush_pending(env, s, lane);
            }
            continue;
        }
        // Copy kPushBatch groups per pass so each lane keeps kPushBatch loads
        // in flight; the per-row accounting stays strictly one row at a time so
        // the flag/flush discipline above is unchanged.
        std::uint32_t i = 0;
        for (; i + kPushBatch <= npush; i += kPushBatch) {
            pace_delay(env.pace * kPushBatch);
            if (!env.no_payload) push_slice_group_batch(env, s, i, lane);
#pragma unroll
            for (unsigned int b = 0; b < kPushBatch; ++b) {
                retire_pushed_row(env, s, s.rows[i + b], g, lane);
            }
        }
        for (; i < npush; ++i) {
            pace_delay(env.pace);
            if (!env.no_payload) push_slice_group(env, s.rows[i], s.group_base[i], lane);
            retire_pushed_row(env, s, s.rows[i], g, lane);
        }
    }
    flush_pending(env, s, lane);
    if (lane == 0 && env.ts != nullptr) {
        ts_last(env.ts + K0P6_MPS_TS_DRAIN, env.ts_enable);
    }
}

// ---- exp_20 E2/E3: diagnostic pool bodies (modes 5 and 6) --------------------
// These run on the SAME reserved CTAs as the real pool, in a mode whose
// publication and combine are mode 0's, so every one of them is compared
// against mode 0 at the SAME C and the only difference is the traffic.

struct diag_env {
    const unsigned short* __restrict__ part;    // [T_ext, 7168] bf16 (local)
    unsigned short* __restrict__ slots;         // [world, MAXTOK, 7168] bf16
    const std::uint32_t* __restrict__ row_rem;  // [T_ext] M2 popcount liveness
    const symmetric_heap_descriptor* symmetric;
    int cur;
    int maxtok;
    int t_ext;
    std::uint32_t variant;      // kDiagVariant{Copy,Read,Write}
    bool local_dst;             // write this rank's own slots, never a peer's
    std::uint64_t window_ticks;
};

// Read-side twin of store_peer_packets: identical loads, identical
// `#pragma unroll 1` MLP-1 shape, no store. The XOR chain is what keeps the
// loads alive AND what reproduces the copy's one-outstanding-load-per-lane
// dependency; the caller sinks the accumulator into a no-op asm so nothing is
// written to memory.
__device__ __forceinline__ unsigned int touch_packets(
        const void* __restrict__ source, std::size_t bytes, unsigned int tid,
        unsigned int threads) {
    const auto* const in =
        reinterpret_cast<const kittens::distributed::packet16*>(source);
    const std::size_t count = bytes >> 4;
    unsigned int acc = 0u;
#pragma unroll 1
    for (std::size_t packet = tid; packet < count; packet += threads) {
        const kittens::distributed::packet16 v = in[packet];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    return acc;
}

// Write-side twin: identical stores, no load.
__device__ __forceinline__ void fill_packets(
        void* __restrict__ destination, std::size_t bytes, unsigned int tid,
        unsigned int threads) {
    auto* const out = reinterpret_cast<kittens::distributed::packet16*>(destination);
    const std::size_t count = bytes >> 4;
    kittens::distributed::packet16 zero;
    zero.x = 0u; zero.y = 0u; zero.z = 0u; zero.w = 0u;
#pragma unroll 1
    for (std::size_t packet = tid; packet < count; packet += threads) {
        out[packet] = zero;
    }
}

// Mode 5. One wave sweeps a stride of receive rows, moving the real pusher's
// unit of work (16 slices of kSliceBytes) between the real pusher's buffers,
// with the read half, the write half, or both. Row liveness comes from the M2
// popcount, so the swept set and the byte count match the real pool's exactly.
// The sweep wraps until the window expires, so every variant occupies the same
// wall-clock window regardless of how fast it runs -- ratewise this is "run
// this traffic class flat out for the duration of M7", which is precisely what
// the real pool does.
__device__ __forceinline__ void run_diag_traffic(const diag_env& d, int wave_id,
                                                 int wave_count, int lane) {
    const std::uint64_t t0 = realtime_now();
    unsigned int acc = 0u;
    bool expired = false;
    while (!expired) {
        for (int r = wave_id; r < d.t_ext; r += wave_count) {
            if (realtime_now() - t0 >= d.window_ticks) { expired = true; break; }
            if (kittens::distributed::load_relaxed<scope::agent>(d.row_rem + r)
                    == 0u) continue;
            const std::uint32_t owner = (std::uint32_t)r / (std::uint32_t)d.maxtok;
            const std::uint32_t pos = (std::uint32_t)r % (std::uint32_t)d.maxtok;
            std::byte* dst_local = reinterpret_cast<std::byte*>(
                d.slots + ((std::size_t)d.cur * (std::size_t)d.maxtok +
                           (std::size_t)pos) * 7168u);
            std::byte* dst = (d.local_dst || owner == (std::uint32_t)d.cur)
                ? dst_local
                : reinterpret_cast<std::byte*>(
                      hk_moe::peer_ptr(dst_local, (int)owner, d.symmetric));
            const std::byte* src = reinterpret_cast<const std::byte*>(d.part) +
                (std::size_t)r * 7168u * 2u;
            for (std::uint32_t nc = 0; nc < 16u; ++nc) {
                const std::size_t off = (std::size_t)nc * kSliceBytes;
                if (d.variant == kDiagVariantCopy) {
                    kittens::distributed::store_peer_packets(
                        dst + off, src + off, kSliceBytes, (unsigned int)lane,
                        64u);
                } else if (d.variant == kDiagVariantRead) {
                    acc ^= touch_packets(src + off, kSliceBytes,
                                         (unsigned int)lane, 64u);
                } else {
                    fill_packets(dst + off, kSliceBytes, (unsigned int)lane, 64u);
                }
            }
        }
    }
#if defined(__HIP_DEVICE_COMPILE__)
    // Sink the read-only accumulator without issuing a store.
    asm volatile("" :: "v"(acc));
#else
    (void)acc;
#endif
}

// Mode 6. Occupied-but-silent: a dependent LDS read/write chain per lane, no
// global memory request at all except the realtime read that bounds the
// window. 17 words per lane is coprime with the 32 LDS banks, so lanes do not
// conflict. `volatile` is what guarantees the ds_read/ds_write pair survives.
inline constexpr std::uint32_t kDiagSpinWords = 17u;

__device__ __forceinline__ void run_diag_spin(std::uint32_t* lds_words,
                                              std::uint64_t window_ticks,
                                              int lane) {
    volatile std::uint32_t* mine =
        lds_words + (std::uint32_t)lane * kDiagSpinWords;
    std::uint32_t x = (std::uint32_t)lane + 1u;
    const std::uint64_t t0 = realtime_now();
    while (realtime_now() - t0 < window_ticks) {
        for (int i = 0; i < 512; ++i) {
            mine[x % kDiagSpinWords] = x;
            x = mine[(x + 7u) % kDiagSpinWords] + 1u;
        }
    }
}

} // namespace hk_moe::mps

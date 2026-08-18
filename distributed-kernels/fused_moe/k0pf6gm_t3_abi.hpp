// T3 — the counter-dataflow backward megakernel's descriptor ABI.
// Design record: overnight/aug18/T3_COUNTER_DATAFLOW_DESIGN.md (the MoK
// synthesis).  T3 keeps the ENTIRE t2b ABI (slots 0..72, meanings from
// k0pf6gm_t2b_abi.hpp, including the saved-plan reuse and the disjoint
// protocol ring) and extends it:
//
//   73  (reserved: the t2v6 PROF ledger slot keeps its number so the
//        prof machinery is shared verbatim; bind 0 when PROF is off)
//   74  T3 counter arena, u32[K0P6_T3_CTR_WORDS + maxb], zeroed by T3's
//       M0 region behind the kernel's ONE grid barrier:
//         [0..7]   unpack_done[source]   — counted arrivals, target = 8
//                  (one per (source, chunk) unpack task); phase-1b and
//                  dW2 tasks gate on the sources their sorted block
//                  actually uses (derived by scanning the block's 32
//                  saved sti entries — the plan is static, no pre-pass)
//         [8]      slab0_done — counted task arrivals; the LAST slab-0
//                  phase-2b task's release-arrive chain certifies every
//                  earlier task's remote RMWs (M9's counted-arrival
//                  reasoning), and the finisher publishes the slab-0
//                  epoch word to all 8 peers.  NO rendezvous behind it.
//         [9]      slab1_done — same for slab 1
//         [10]     p1z_cursor      [11] p1b_cursor
//         [12]     p2b_s0_cursor   [13] p2b_s1_cursor
//         [14]     unpack_cursor   [15] spare
//         [16..]   z_done[b] — per-32-row-block z-regen completion
//                  (arrive_local_count per covering tile; phase-1b and
//                  dW2 tasks poll their blocks)
//       (m8 front/back tickets stay in mps_state words 2/3; wgrad
//        cursors stay in mps_state words 4/5 — the t2v6 contract.)
//   75  (optional) T3 prof arena, u64[32], the t2v6 ledger layout —
//       bind 0 unless the PROF arm is compiled.
//
// DWMODE (slot 72) gains mode 3 = T3: wgrad is the UNIVERSAL FILLER —
// claimed as whole tiles inside every long gate-wait (unpack gates,
// phase-1b source gates, combine word waits) and drained at the tail;
// nmb/accumulate semantics are the t2v6 contract (epoch-derived).
//
// The ONLY grid barrier in T3 seals M0 (retire wait + counter/dw zeroing)
// before the dataflow starts.  Everything after is counters.
#pragma once

#define K0P6_D_T3_CTRS 74
#define K0P6_D_T3_PROF 75
#define K0P6_T3_D_LEN 76

#define K0P6_T3_CTR_UNPACK 0   // [8]
#define K0P6_T3_CTR_SLAB0 8
#define K0P6_T3_CTR_SLAB1 9
#define K0P6_T3_CTR_P1Z_CUR 10
#define K0P6_T3_CTR_P1B_CUR 11
#define K0P6_T3_CTR_P2B0_CUR 12
#define K0P6_T3_CTR_P2B1_CUR 13
#define K0P6_T3_CTR_UNPK_CUR 14
#define K0P6_T3_CTR_WORDS 16
#define K0P6_T3_CTR_ZDONE 16   // + b, b in [0, maxb)

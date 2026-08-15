// T2B — the backward megakernel's descriptor ABI (design: overnight/aug14/
// T2_BACKWARD_MEGA_DESIGN.md; blueprint: T2_PHASE_MAP.md).
//
// PRINCIPLE: the backward descriptor keeps the m15 base layout (slots 0..62)
// so k0p6_dread, the adapters, and both phase bodies work unchanged — the
// slots change MEANING, not position:
//
//   0  MYIDS   = the forward's routing ids (same tensor, saved by autograd)
//   1  MYWG    = the forward's topk weights (same tensor)
//   2  HIDDEN  = dY   [T, 7168] bf16   (M1b quantizes it in-flight, fwd path)
//   32 OUT     = dX   [T, 7168] bf16   (M8 writes it)
//   35 W13     = W2T  [E, 2048, 7168] fp8 AITER-shuffled   (phase-1b: dH2)
//   36 S13     = S2T  [E, 16, 56] f32 blockscales (transposed grid)
//   37 W2      = W13T [E, 7168, 4096] fp8 AITER-shuffled   (phase-2b: dX)
//   38 S2      = S13T [E, 56, 32] f32
//   27 A2Q     = dZq  [rowcap, 4096] fp8    (backward phase-1b OUTPUT:
//   28 DQ2     = DQdZ [rowcap, 32]  f32      dgate|dup in W13-column order)
//   8  A_DST   = dY unpacked rows (backward's own receive buffer)
//   9  SC_STAGE= dY row-group scales (backward's own)
//
//   SAVED-PLAN REUSE (read-only in backward; bound to the FORWARD's saved
//   arenas — reserve_row/scatter atomics are nondeterministic, so the plan
//   is never re-derived; M0.5 and M3..M5 do not exist in the backward):
//   12 PULL_STAGE, 13 PULL_CNT, 14 STI, 15 SWT, 16 SEI, 17 NVI,
//   18 PULL_PTR, 19 PULL_SRC, 53 TILE_DESC, 54 NUM_TILES
//
//   PROTOCOL INSTANCES (risk #1 of the phase map): the host binds the
//   backward to its OWN epoch cell (34), retired (31), dest_counter (4),
//   rows_done (5), chunk_ready (50), row_ready (25), slots (61), a_ll (3),
//   pushed/qpush (6/7), mps_state (60), gbar (23) — a disjoint symmetric
//   ring so forward and backward epoch lattices never interleave.
//   dest_counter is vestigial in M1b (no reservation) but stays bound so
//   M0's parity-zeroing remains verbatim.
//
// NEW SLOTS (63..):
#define K0P6_D_T2B_ZQ 63        // t2b-owned Zq [rowcap,4096] (M5.9b z-regen out)
#define K0P6_D_T2B_DQZ 64       // t2b-owned DQZ [rowcap,32] (M5.9b out)
#define K0P6_D_T2B_ACTQ 65      // W13 ORIGINAL [E,4096,7168] (z-regen input)
#define K0P6_D_T2B_ACTDQ 66     // S13 ORIGINAL [E,32,56] (z-regen input)
#define K0P6_D_T2B_XROWS 67     // fwd-saved a_dst (x rows) [T_loc_max, 7168] fp8
#define K0P6_D_T2B_XSC 68       // fwd-saved sc_stage [T_loc_max, 56] f32
#define K0P6_D_T2B_DW13 69      // dW13 accumulator [E, 4096, 7168] fp32|bf16
#define K0P6_D_T2B_DW2 70       // dW2  accumulator [E, 7168, 2048] fp32|bf16
#define K0P6_D_T2B_DWTOPK 71    // dw_topk out [T, 8] f32 (rowdot(dH2, act_z))
#define K0P6_D_T2B_DWMODE 72    // scalar: 0 = fp32 accum, 1 = bf16 accum
#define K0P6_T2B_D_LEN 73

// Backward GEMM geometry (the phase map's duality):
//   phase-1b (dH2 = W2T · dYq): N = 2048, K = 7168 — phase-1's exact chunk
//     geometry (8 chunks x 256), NO gate/up pairing; the paired-register
//     epilogue instead computes swiglu'(z) from Zq/DQZ and emits (dgate,dup)
//     into dZq, plus dw_topk row-dots against A2q/DQ2 while dH2 is live.
//   phase-2b (dX rows = W13T · dZq): N = 7168, K = 4096 — phase-2's exact
//     nc geometry with kKGroups 16->32 and DQ width doubled; the epilogue
//     applies the SAVED swt weight per sorted row (exactly the forward's
//     w-application point — fold-at-combine is forbidden by the match_any
//     dedup, see T2_PHASE_MAP.md section 4) and remote-accumulates dX rows
//     to token owners; M8 combine and M9 retire reuse byte-for-byte.
//
// Wgrad filler (carved service CTAs, M20 PF-engine precedent):
//   dW2 += swt · dYq-rows  x act(z)-rows   — schedulable from slab 0 on
//   dW1 += swt · dZq-rows  x x-rows        — schedulable from slab 1 on
//   Accumulation is rank-local (owned experts only), zero protocol; any
//   unfinished tiles run in a post-M8 tail before M9.
#define K0P6_T2B_WGRAD_CTAS 8

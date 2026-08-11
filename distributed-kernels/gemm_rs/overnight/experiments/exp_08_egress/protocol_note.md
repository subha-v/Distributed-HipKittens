# exp_08 — protocol note for the tile-order change (read before its first GPU run)

Written against `experiments/exp_05_release_granularity/protocol_review.md` §1–§2,
re-read in full before this candidate was built. Sources re-read:
`gemm_rs_mi300x.cpp`, `gemm_rs_mi300x_hk_adapter.cuh`,
`include/cdna3/ops/group/distributed/{sync,packet}.cuh`.

## What changed

One expression. `WGM` in the producer role goes from `constexpr int WGM = 4` to
`const int WGM = num_pid_m` (with `-DHK_GEMM_RS_MI300X_WGM4=1` restoring 4 as
the control arm). The decode expression itself is **byte-identical**; only the
constant feeding it moved. With `WGM = num_pid_m`, `in_group` becomes
`num_pid_m * num_pid_n == tiles`, so for every `t < tiles`: `group == 0`,
`first == 0`, `gsize == num_pid_m`, and the decode reduces to
`tm = t % num_pid_m`, `tn = t / num_pid_m` — column-major.

Nothing else is touched. In particular **no store, no fence, no barrier, no
wait, no cache-policy bit, no release site, no publication site, and no
signal index expression is modified**, and the GEMM mainloop (including the
ISA-verified `acquire_frags` anchor) is not touched.

## Why it is ordering-neutral

The change is a **permutation of the tile→CTA assignment**. Against the review's
invariant list:

1. **Release still dominates publication (§2.1, C1).** There is still exactly
   one `release_payload_system()` per tile at `:381`, and the leader's
   `publish_band_epoch` loop at `:382-401` still runs after it, in the same
   iteration. Unmoved.
2. **Coverage is exact and each ready cell is published exactly once per rank
   per epoch.** `{pid + i·NG} ∩ [0, tiles)` is unchanged as a *set of `t`*, and
   `t ↦ (tm, tn)` is a bijection onto `[0,num_pid_m) × [0,num_pid_n)` for either
   `WGM` — asserted mechanically in `destmap.py`, which fails on a duplicate or
   a shortfall, for all six scored shapes under both orders. `check_signals()`
   (every touched ready/credit cell `== n_calls`) is the runtime version of the
   same assertion and is part of M3/M5.
3. **Intra-CTA WAW on a heap slot stays impossible (§2.7 invariant 8).** A
   group's slots must be pairwise disjoint. `(tm, tn) ↦ (dest, lrow, tn)` is
   still injective: `tn` is carried through unchanged, and for `bands == 1`
   (shapes 2–6) `dest` and `lrow` are determined by `tm`; for `bands == 4`
   (shape 1, `BM=32`, `slice=8`, `EB=8`) `dest = 4·tm + b`, injective in
   `(tm, b)`. Distinct `t` ⇒ distinct `(tm, tn)` ⇒ distinct slot.
4. **The wait-for graph is unchanged, so §2.4's acyclicity carries over
   verbatim.** `R(d,e) → P(s,e)` and `P(r,e) → R(d,e-1)` are edges between
   *epochs*, and the tile order does not create, delete, or re-target one. The
   producer's credit wait is still `credit[dest][lrow][tn] ≥ ep-1` on the cell
   rank `dest` writes to us, still per tile, still before that tile's first
   payload store, still fail-closed via `atomicOr` + `__syncthreads()` +
   `error_bit_set`. Every wait remains bounded, so the worst case of a wrong
   analysis is a timeout and a poisoned instance, never a wedged node.
5. **`ep` remains loop-invariant** (computed at `:162-163`, above the tile
   loop). Batching/grouping is not involved.

## The one thing that genuinely changes: cross-peer arrival order

Producers now publish their `ready` cells in a different order, so packets and
flags arrive at each destination in a different sequence. The review settles
this directly: a reducer's only wait is `ready[s][lrow][col] ≥ e` for all eight
`s` of *its own* output tile (`:374-377`), it never reads any other tile's
slots, and `release_payload_system()` supplies a destination-agnostic ordering
(`s_waitcnt vmcnt(0)` covers every outstanding VMEM op of every thread to every
target, and `buffer_wbl2 sc0 sc1` is an L2-wide writeback that does not
discriminate by destination — §2.3). **No reducer, credit, or epoch path depends
on which peer lands first.** This is the same question the E2 research report
flagged for I4 (rank rotation) as a one-line review item; the answer is no.

Second-order, performance not correctness: reducer tile `(lr, col)` now becomes
ready at a different time relative to its neighbours, so reducer spin
distribution changes. Bounded and fail-closed either way.

## What must be true for the candidate to be accepted

- M3 must pass 17/17 at **both** `1e-2` and `2e-3`. Arithmetic is untouched, so
  `max|diff|` should be *identical* to the current best (`4.883e-04` on the
  scored set): the reduction is source-ascending FP32 with one RNE pack and does
  not depend on producer order. A moved `max|diff|` means something other than
  the tile order changed and is grounds to stop.
- `check_signals` / `check_epochs` clean (they are what would catch a coverage
  bug, which is the only real failure mode here).
- Shapes 1, 3 and 4 are **free built-in controls**: `destmap.py` says they
  already run at 8.00 of 8 effective egress links, so their egress cannot
  improve. If they *move*, the cause is the tile order's effect on operand
  reuse, not on egress, and that must be reported as such.

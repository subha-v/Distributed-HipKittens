# exp_04_tgen — T-generalization: root-cause + fix the T≠4096 correctness defect (exp_36 blocker)

**Owner:** tgen agent · branch `ablations-tgen` (from `ablations` @ `ba0f785a`) ·
node `gbt350-odcdh2-c05-1` (8× MI350X), worktree `~/DHK-tgen`.

## The defect being chased (measured, exp_36 aug11)

At `T ∈ {1024, 2048}` tokens/rank BOTH megakernel arms return wrong output while
`production` passes, with `pperr=0`:
- `mps_mega`: 7 of 8 ranks' outputs ENTIRELY unwritten (poison survivors = T·H
  per rank; `first_rows=[0..15]`). Rank 0 is the single written rank.
- `pf6gm_mega`: everything written (survivors=0) but wrong —
  `max_abs≈1.71, relative≈0.846` (≈ 7/8 of the value mass missing).
Identical signature with capacities pinned at T=4096 values AND scaled to T.
`T=4096` is correct (13 green campaigns).

## Root cause (found by code audit before any GPU run; to be confirmed by the gate ladder)

Two facts compose:

1. **The host always sets the kernel's segment stride to T.** The harness driver
   writes `int(T)` into descriptor slot 51 (`K0P6_D_MAXTOK`) —
   `e004pf_k0pf_ab.py:2807` / `:2824` ("MAXTOK == T for this campaign"). The
   `K0_MAXTOK` env only sizes host buffers; the kernel-visible MAXTOK is T in
   BOTH of exp_36's capacity modes. That is why "pinned" and "scaled" failed
   identically: the kernel ran with MAXTOK=1024 either way.

2. **The M2 hole-sentinel pass assumes `K0P6C_NCHUNK_MAX·K0P6_CHUNK == MAXTOK`.**
   All three chassis kernels (`k0pf6gm_device_tile.hip:779-794`,
   `k0pf6gm_device_tile_mps.hip:1381-1397`, `k0pf6gm_device_tile_m15.hip:906-917`
   at `ba0f785a`) sentinel holes per (source s, chunk c) as
   `[base+fill, base+CHUNK)·TOPK` with `base = s·MAXTOK + c·CHUNK`, iterating
   `c ∈ [0, 8)`. Only at MAXTOK = 8·512 = 4096 does every chunk end inside its
   segment. At MAXTOK=1024 chunks c=2..7 are phantom (producer publishes
   fill=0), and their ranges `s·MAXTOK + [1024, 4096)` land on the FULL LIVE
   segments s+1, s+2, s+3. Every segment s'≥1 is covered by segment s'−1's
   phantom chunks ⇒ `recv_eid = -1` sprayed over ALL live rows except source
   0's segment. (At MAXTOK=2048 the same holds via c=4..7 covering s+1.)
   Additionally at scaled capacities the s=7 ranges run past `recv_eid`'s
   allocation (rows up to 11·MAXTOK > T_ext) — an OOB heap write of −1s.

**Downstream, this reproduces every observed symptom:**
- scatter's expert-range filter drops the −1 pairs ⇒ the GEMM computes ONLY
  segment-0 rows on every rank.
- `pf6gm_mega`: M7.5 publishes `row_ready` from `row_remaining` (written by M2
  Pass B BEFORE the sentinel pass and not touched by it) ⇒ all polls succeed ⇒
  M8 combines `part` rows that M5 zeroed but the GEMM never wrote ⇒ output
  written, ranks ≥1 get (near-)zero contributions, rank 0 correct ⇒
  survivors=0, global max_abs = max|ref| ≈ 1.71, relative ≈ 7/8. pperr=0.
- `mps_mega`/M15 (mode 12): row flags are COUNT-driven (`retire_pushed_row` /
  the epilogue's per-row slice accounting) ⇒ never-computed rows are never
  flagged ⇒ owners 1..7's combine batches never fire ⇒ 7 of 8 ranks' outputs
  entirely unwritten; owner 0 (all producers' segment-0 rows) is written.

**The broken invariant, in one line:** the sentinel pass assumed the static
chunk grid (8×512) tiles the runtime segment (MAXTOK), which is true only at
MAXTOK=4096.

## The fix (this branch, commit 1)

Clamp each chunk's row range to its segment: `seg_row1 = min((c+1)·CHUNK,
MAXTOK)`; `hole1 = (s·MAXTOK + seg_row1)·TOPK`. Phantom chunks become empty
ranges. Applied to all three kernels (shared chassis). Exact-math at
MAXTOK=4096: `seg_row1 == (c+1)·CHUNK` for all 8 chunks, so the written ranges
are identical to the pre-fix kernel — but the `.text` changes, so per the
ratchet rule a same-session paired T=4096 campaign must prove no regression
(M15 C=28 control = 5,822.0 µs = 0.7544×; production ≈ 7,704–7,712).
`K0P6_MPS_SRC_REV 31→32`, `K0P6_M15_SRC_REV 1→2`.

Producer-side chunk publication is NOT touched: it already clamps fills
(`fill = clamp(n_s − c·CHUNK, 0, CHUNK)` ⇒ 0 for phantom chunks) and M2 Pass A
polls all 64 words that M1 unconditionally publishes — that pairing is sound at
every MAXTOK and must stay symmetric.

## Method / gates (per the corpus discipline)

1. Reproduce first: T=2048 smoke on the UNFIXED M15 pin (`~/DHK-m15`-style pin
   on this branch) — confirm the exp_36 signature.
2. Fixed build gate ladder at every claimed T ∈ {1024, 2048, 4096}:
   [MOK GATE] pass · [POISON] survivors=0 + selftest fires · [MPS SOAK]
   600/600 pperr=0 · negative control fails.
3. Campaigns: 5-rotation at T=1024 and T=2048 (production denominator), one
   confirming T=4096 campaign (M15 C=28 vs 5,822.0 µs control band).
4. pf6gm_mega smokes at T ∈ {1024, 2048} (same fix, same chassis).

T ≤ 512 remains refused by the host guard (`ab.py:554`) — separate issue, out
of scope here.

## Pre-registered expectations

- Unfixed T=2048 smoke reproduces: mps 7/8 unwritten, rank 0 written.
- Fixed: all gates green at all three T's; T=4096 M15 C=28 lands in the
  5,822 µs band (the clamp adds ~64 trivially-predicted scalar ops to a
  once-per-epoch grid-strided pass — no phase-stamp-visible cost).
- Batch response (corpus pre-registration): small T favors coarse signals; the
  M15 advantage (0.7544× at T=4096) is expected to SHRINK at T ∈ {1024, 2048}.

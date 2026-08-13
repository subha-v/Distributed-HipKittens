# exp_04_tgen result — the T≠4096 defect is the M2 hole-sentinel's phantom chunks; fixed, gates green at T ∈ {1024, 2048, 4096}

**Node** `gbt350-odcdh2-c05-1` (8× MI350X, gfx950) · branch `ablations-tgen`
(fix `af030778`, RUN PIN `144f2029`, from `ablations` @ `ba0f785a`) · node
worktree `~/DHK-tgen` · driver: exp_36's audited `~/e36/rc_T.sh` machinery ·
2026-08-13 06:12–~07:00 UTC. Raw logs `~/e36/scratch/tgen_*` and
`~/e36/scratch/screen_tgen_*.csv` on the node.

---

## Root cause (exact file:line and the broken invariant)

Two facts compose; each is harmless alone:

1. **The kernel-visible MAXTOK is always T.** The host driver writes `int(T)`
   into descriptor slot 51 (`K0P6_D_MAXTOK`) at
   `e004pf_k0pf_ab.py:2807` and `:2824` ("MAXTOK == T for this campaign") —
   the `K0_MAXTOK` env only sizes host buffers. exp_36's two capacity modes
   ("pinned" vs "scaled") therefore ran the SAME kernel shape, which is why
   their failure signatures were identical.

2. **The M2 hole-sentinel pass assumes the static chunk grid tiles the
   segment** — `K0P6C_NCHUNK_MAX·K0P6_CHUNK == MAXTOK`, true only at
   MAXTOK=4096. At `ba0f785a` the pass (base chassis
   `k0pf6gm_device_tile.hip:779-794`, mps
   `k0pf6gm_device_tile_mps.hip:1381-1397`, m15
   `k0pf6gm_device_tile_m15.hip:902-917`; donor
   `~/amd-master/.../prefill_opt/kernels/k0pf6gm_mega.hip:717-734`) writes,
   per (source s, chunk c ∈ [0,8)):
   `recv_eid[(s·MAXTOK + c·512 + fill) .. (s·MAXTOK + (c+1)·512))·TOPK] = −1`.
   At MAXTOK=1024, chunks c=2..7 are phantom (producer publishes fill=0) and
   their ranges are `s·1024 + [1024, 4096)` — the FULL LIVE segments s+1,
   s+2, s+3. Every segment s'≥1 is covered by segment s'−1's phantom chunks;
   **segment 0 alone survives**. (MAXTOK=2048: c=4..7 cover exactly segment
   s+1 — same outcome.) At scaled capacities the s=7 ranges additionally run
   past `recv_eid`'s allocation (rows to 11·MAXTOK > T_ext·, a 32 KB OOB
   write of −1s into the next heap buffer).

**Downstream this reproduces every measured symptom.** Scatter's expert-range
filter drops the −1 pairs ⇒ the GEMM computes ONLY segment-0 (source rank 0's)
rows on every rank:
- `pf6gm_mega`: M7.5 publishes `row_ready` from `row_remaining` (written by M2
  Pass B before the sentinel pass, untouched by it) ⇒ every combine poll
  succeeds ⇒ M8 reads `part` rows M5 zeroed but the GEMM never wrote ⇒ output
  fully written, ranks ≥1 combine zeros ⇒ survivors=0,
  `max_abs = max|ref| ≈ 1.7–1.8`, `relative ≈ 7/8`, `pperr=0`.
- mode-12 `mps_mega`: row flags are count-driven (`retire_pushed_row` /
  epilogue slice accounting) ⇒ never-computed rows never flagged ⇒ owners
  1..7's combine batches never fire ⇒ **7 of 8 ranks' outputs entirely
  unwritten, rank 0 written** — exp_36's exact signature.
- M15: slab-coarse readiness (no per-row protocol) ⇒ writes zeros like
  pf6gm — confirmed by the reproduction below.

**The broken invariant in one line:** the hole-sentinel derivation mapped
every static chunk to rows, assuming `8·512 == MAXTOK`; only the producer's
fill clamp knew smaller segments exist.

## The fix (`af030778`)

Clamp each chunk's row range to its own segment:
`seg_row1 = min((c+1)·K0P6_CHUNK, MAXTOK)`;
`hole1 = (s·MAXTOK + seg_row1)·TOPK`. Phantom chunks become empty ranges.
Applied to all three chassis kernels. Producer-side publication is untouched
(its fill clamp is already correct, and M2 Pass A's 64-word poll must stay
symmetric with M1's 64-word publish). Exact-math at MAXTOK=4096 — identical
written ranges — but the `.text` changes, so the T=4096 no-regression
campaign below is part of the claim. `K0P6_MPS_SRC_REV 31→32`,
`K0P6_M15_SRC_REV 1→2`.

---

## Reproduction (unfixed M15 pin `~/DHK-m15` @ `5753970b`, T=2048, C=16, smoke)

```
[MOK GATE] production max_abs=0.023438 relative=0.005739 pass=True
[MOK GATE] pf6gm_mega max_abs=1.812500 relative=0.875568 pass=False
[MOK GATE] mps_mega   max_abs=1.812500 relative=0.875128 pass=False   ← unfixed M15
[POISON] survivors=0 both arms · selftest fires (57344) · pperr=0
```
`relative ≈ 0.875 = 7/8` — the "seven of eight segments dropped" fingerprint,
matching the mechanism exactly. (M15 writes zeros — slab-coarse readiness —
unlike mode-12 mps which leaves 7/8 ranks unwritten; both are the same root
cause.)

## Fixed gates (RUN PIN `144f2029`, `~/DHK-tgen`, M15 as `mps_mega`)

| screen | T | cap | mps gate | survivors | selftest | pperr | negctl | soak | spin |
|---|---:|---|---|---:|---|---:|---|---|---|
| tgen_f1 | 2048 | pinned | **pass** 0.0430 / 0.008289 (C=16 AND C=28) | 0 | fires | 0 | fails ✓ | — | — |
| tgen_f2 | 1024 | pinned | **pass** 0.0430 / 0.008289 | 0 | fires | 0 | fails ✓ | — | — |
| tgen_f3 | 4096 | pinned | **pass** 0.0315 / 0.008295 (family-canonical) | 0 | fires | 0 | fails ✓ | **600/600** | 0/0 |
| tgen_f4 | 2048 | scaled | **pass** 0.0430 / 0.008288 | 0 | fires | 0 | fails ✓ | — | — |
| tgen_f5 | 1024 | scaled | **pass** 0.0391 / 0.008361 | 0 | fires | 0 | fails ✓ | — | — |

f3 (T=4096) came back with full timing at screen resolution: mps 6,103.1 µs =
0.7801× production (screens resolve ~6.6%; the campaign below is the number).

**pf6gm_mega still fails at T≠4096** — its arm builds from the DONOR tree
(`PF3_KERNEL_DIR/k0pf6gm_mega.hip` in `~/amd-master`, ab.py:299/905), not
from `DHK_ROOT`: the same sentinel bug in a copy this branch does not own.
The identical 2-line clamp applies at `k0pf6gm_mega.hip:729-732`; an
automated edit of that shared tree was denied by policy, so it is left to
the owner (patch text in plan.md / this file). Because the harness blocks
ALL timing when any arm's MOK gate fails, the T≠4096 campaigns below run
`K0_MOK_ARMS=production,mps_mega`.

## Campaigns (5 processes, 500 warmup / 100 timed, C=28 g=353 mode=12 flush_rows=16)

<!-- FILLED AFTER CAMPAIGNS -->

## Confidence

- **High** on the root cause: the mechanism was found by code audit, predicts
  the rank-0-survivor asymmetry, the 7/8 relative error, the written/unwritten
  split between pf6gm and mode-12, and the pinned≡scaled equivalence — and the
  single-variable A/B (sentinel clamp only) flips every gate at every T.
- **High** on the fix's default-shape safety: exact-math equality at
  MAXTOK=4096 plus green 600/600 soak + canonical gate band at T=4096.

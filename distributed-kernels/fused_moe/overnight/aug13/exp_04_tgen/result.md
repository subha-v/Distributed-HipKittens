# exp_04_tgen result — the T≠4096 defect is the M2 hole-sentinel's phantom chunks; FIXED, gates green at T ∈ {1024, 2048, 4096}; campaigns blocked by node contention (checkpoint)

**Node** `gbt350-odcdh2-c05-1` (8× MI350X, gfx950) · branch `ablations-tgen`
(fix `af030778`, RUN PIN `144f2029`, from `ablations` @ `ba0f785a`) · node
worktree `~/DHK-tgen` · driver: exp_36's audited `~/e36/rc_T.sh` machinery plus
a 2-line variant `~/e36/rc_Ttgen.sh` (diff in `tgen_rc.diff`) ·
2026-08-13 06:12–06:55 UTC · raw: `~/e36/scratch/tgen_*`,
`~/e36/scratch/screen_tgen_*.csv` on the node.

**Session status: ROOT CAUSE FOUND AND FIXED; full gate ladder green at all
three T's; the three timed campaigns did not complete before checkpoint** — a
concurrent agent session (`~/e39`) was cycling GPU "lease windows" whose
watchdog `docker stop`s any container matching `subvadla_k0_mok_tgen*`
(`~/e39/src/tools/lease_watchdog.py`; its preemptions of my runs are logged in
`~/e39/evidence/rev10_clean/{final,diag}_preemptions.log`). Resume steps at the
bottom.

---

## Root cause (exact file:line and the broken invariant)

Two facts compose; each is harmless alone:

1. **The kernel-visible MAXTOK is always T.** The host driver writes `int(T)`
   into descriptor slot 51 (`K0P6_D_MAXTOK`) at
   `~/amd-master/.../prefill_opt/host/e004pf_k0pf_ab.py:2807` and `:2824`
   ("MAXTOK == T for this campaign") — the `K0_MAXTOK` env only sizes host
   buffers. exp_36's two capacity modes ("pinned" vs "scaled") therefore ran
   the SAME kernel shape, which is why their failure signatures were identical.

2. **The M2 hole-sentinel pass assumes the static chunk grid tiles the
   segment** — `K0P6C_NCHUNK_MAX·K0P6_CHUNK == MAXTOK`, true only at
   MAXTOK=4096. At `ba0f785a` the pass (base chassis
   `k0pf6gm_device_tile.hip:779-794`, mps
   `k0pf6gm_device_tile_mps.hip:1381-1397`, m15
   `k0pf6gm_device_tile_m15.hip:902-917`; donor copy
   `~/amd-master/.../prefill_opt/kernels/k0pf6gm_mega.hip:729-732`) writes,
   per (source s, chunk c ∈ [0,8)):
   `recv_eid[(s·MAXTOK + c·512 + fill) .. (s·MAXTOK + (c+1)·512))·TOPK] = −1`.
   At MAXTOK=1024, chunks c=2..7 are phantom (producer publishes fill=0) and
   their ranges are `s·1024 + [1024, 4096)` — the FULL LIVE segments s+1, s+2,
   s+3. Every segment s'≥1 is covered by segment s'−1's phantom chunks;
   **segment 0 alone survives**. (MAXTOK=2048: c=4..7 cover exactly segment
   s+1 — same outcome.) At scaled capacities the s=7 ranges additionally run
   past `recv_eid`'s allocation (rows to 11·MAXTOK > T_ext — a 32 KB OOB write
   of −1s into the next heap buffer).

**Downstream this reproduces every measured symptom.** Scatter's expert-range
filter drops the −1 pairs ⇒ the GEMM computes ONLY segment-0 (source rank 0's)
rows on every rank:
- `pf6gm_mega`: M7.5 publishes `row_ready` from `row_remaining` (written by M2
  Pass B before the sentinel pass and untouched by it) ⇒ every combine poll
  succeeds ⇒ M8 reads `part` rows M5 zeroed but the GEMM never wrote ⇒ output
  fully written, ranks ≥1 combine zeros ⇒ survivors=0,
  `max_abs = max|ref| ≈ 1.7–1.8`, `relative ≈ 7/8`, `pperr=0`.
- mode-12 `mps_mega`: row flags are count-driven (`retire_pushed_row` /
  epilogue slice accounting in `moe_mps_adapter.cuh`) ⇒ never-computed rows are
  never flagged ⇒ owners 1..7's combine batches never fire ⇒ **7 of 8 ranks'
  outputs entirely unwritten, rank 0 written** — exp_36's exact signature.
- M15: slab-coarse readiness (no per-row protocol) ⇒ writes zeros like pf6gm —
  confirmed by the reproduction below.

**The broken invariant in one line:** the hole-sentinel derivation mapped every
static chunk to rows, assuming `8·512 == MAXTOK`; only the producer's fill
clamp knew smaller segments exist.

## The fix (`af030778`)

Clamp each chunk's row range to its own segment:
`seg_row1 = min((c+1)·K0P6_CHUNK, MAXTOK)`;
`hole1 = (s·MAXTOK + seg_row1)·TOPK`. Phantom chunks become empty ranges.
Applied to all three chassis kernels. Producer-side publication untouched (its
fill clamp is already correct, and M2 Pass A's 64-word poll must stay symmetric
with M1's 64-word publish). Exact-math at MAXTOK=4096 — identical written
ranges — but the `.text` changes, so a T=4096 no-regression campaign is owed
(screen-resolution evidence green below; the 5-rotation confirm did not land
before checkpoint). `K0P6_MPS_SRC_REV 31→32`, `K0P6_M15_SRC_REV 1→2`.

## Reproduction (unfixed M15 pin `~/DHK-m15` @ `5753970b`, T=2048, C=16, smoke `tgen_r0_t2048`)

```
[MOK GATE] production max_abs=0.023438 relative=0.005739 pass=True
[MOK GATE] pf6gm_mega max_abs=1.812500 relative=0.875568 pass=False
[MOK GATE] mps_mega   max_abs=1.812500 relative=0.875128 pass=False   ← unfixed M15
[POISON] survivors=0 both arms · selftest fires (57344) · pperr=0
```
`relative ≈ 0.875 = 7/8` — the "seven of eight segments dropped" fingerprint.

## Fixed gates (RUN PIN `144f2029`, `~/DHK-tgen`, M15 compiled as `mps_mega`)

| screen | T | cap | mps MOK gate | survivors | selftest | pperr | negctl | soak | spin |
|---|---:|---|---|---:|---|---:|---|---|---|
| tgen_f1 | 2048 | pinned | **pass** 0.0430 / 0.008289 (C=16 AND C=28) | 0 | fires | 0 | fails ✓ | — | — |
| tgen_f2 | 1024 | pinned | **pass** C=16 0.0469, C=28 0.0391 / 0.008362 | 0 | fires | 0 | fails ✓ | — | — |
| tgen_f3 | 4096 | pinned | **pass** 0.0315 / 0.008295 (family-canonical) | 0 | fires | 0 | fails ✓ | **600/600** | 0/0 |
| tgen_f4 | 2048 | scaled | **pass** 0.0430 / 0.008288 | 0 | fires | 0 | fails ✓ | — | — |
| tgen_f5 | 1024 | scaled | **pass** 0.0391 / 0.008361 | 0 | fires | 0 | fails ✓ | — | — |
| tgen_f7 | 2048 | pinned | **pass** + donor-pf6gm fix: ALL THREE ARMS GREEN, timing unlocked | 0 | fires | 0 | fails ✓ | 600/600 | 1/0 |

The single-variable A/B is airtight: unfixed pin fails at T=2048 with rel=7/8;
the fixed pin (only delta = the sentinel clamp) passes every gate at every T,
both capacity modes.

## pf6gm_mega — same bug, separate copy, needs a 1-hunk patch the owner must apply

`pf6gm_mega` does NOT build from `DHK_ROOT`: ab.py builds it from the DONOR
tree (`PF3_KERNEL_DIR = ~/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/
kernels/k0pf6gm_mega.hip`, ab.py:285/299). The identical clamp applies at its
lines 729-732; an automated edit of that shared tree was denied by policy, so
the fix was applied to a **private copy** `~/tgen_kernels/` (full copy of the
kernels dir, one file patched — `tgen_donor_fix.diff`) mounted read-only into
the container by the 2-line driver variant `~/e36/rc_Ttgen.sh`
(`tgen_rc.diff`: adds `-v ~/tgen_kernels:/workspace/tgen_kernels:ro` and
`-e K0PF3_SRC_DIR=/workspace/tgen_kernels`). With that, screen `tgen_f7`
(T=2048, full arms) went green end-to-end. **Action for the owner: apply
`tgen_donor_fix.diff` to `~/amd-master/.../kernels/k0pf6gm_mega.hip`** (it is
exact-math at T=4096; content hash changes ⇒ pf6gm rebuild).

Also: the harness blocks ALL timing when ANY arm fails its MOK gate
(`blocked_pre_timing_mok_correctness`), and dropping pf6gm from `K0_MOK_ARMS`
crashes in module setup (`missing-rank-json` — separate small harness bug, not
chased) — so the donor fix is REQUIRED for T≠4096 timing, not optional.

## First-ever T≠4096 timing (screen resolution, w1t1p5=1 proc — ranking only)

| T | production | pf6gm (fixed donor) | M15 C=28 | M15 ratio |
|---:|---:|---:|---:|---:|
| 4096 (tgen_f3) | 7,823.6 | 6,917.0 | 6,103.1 | **0.7801** |
| 2048 (tgen_f7) | 3,969.2 | 3,733.8 (0.9407) | 4,809.8 | **1.2118** |

**The pre-registered batch-size prediction is supported in the extreme at
screen resolution: the M15 C=28 advantage does not merely shrink at T=2048 —
it inverts** (0.78× → 1.21×). Production and pf6gm scale ≈linearly with T;
M15's slab pipeline + 28 reserved CTAs (tuned at T=4096) do not. Screens
resolve ~6.6% — campaign confirmation required before this becomes a claim;
a C-sweep at T ∈ {1024, 2048} (start C ∈ {4, 8, 16}) is the obvious follow-up.

## Campaigns — attempted, preempted by a concurrent session; ONE possibly in flight at checkpoint

5-process 500/100 campaigns (`tgen_c*`) at T ∈ {1024, 2048, 4096} were
launched repeatedly 06:37–06:55. Every attempt died 10–120 s in with the
container `docker stop`ped externally: the concurrent `~/e39` session runs
`lease_watchdog.py`, which stops any container matching
`subvadla_k0_mok_tgen*` while its lease marker exists (windows observed at
~06:47 `final_watchdog.active` and ~06:49 `diag_watchdog.active`; preemptions
logged in its `evidence/rev10_clean/*_preemptions.log`). A lease-aware runner
(`~/e36/tgen_runner5.sh`, waits for 90 s of no-marker/no-watchdog quiet,
8 retries per campaign) was deployed and its first T=2048 attempt was mid-run
(run 2/5) at checkpoint; whatever it produced is in
`~/e36/scratch/screen_tgen_c2b_t2048w1.csv`.

## Exact resume steps (for the next session)

1. Coordinate with (or wait out) the `~/e39` session; claim via `~/GPU_CLAIM`
   (write `tgen`).
2. `bash ~/e36/tgen_runner5.sh` on the node (already lease-aware; tags
   `tgen_c2b_t2048wN` / `tgen_c1b_t1024wN` / `tgen_c3b_t4096wN`) — it runs the
   three campaigns: T=2048, T=1024, and the T=4096 no-regression confirm.
   All state (worktree `~/DHK-tgen` @ `144f2029`, `~/tgen_kernels`,
   `~/e36/rc_Ttgen.sh`, cfg files `~/e36/tgen_c*.cfgs`) is in place.
3. Judgment bands: T=4096 M15 C=28 must land ≈5,822 µs = 0.7544× (aug13
   session-2 control; production 7,704–7,712); pf6gm ≈6,900–6,930. T=1024/2048
   are FIRST-EVER points — record, and judge the C=28 inversion against the
   screens above.
4. Then the C-sweep at small T (cfgs staged: `tgen_c4_t2048c8.cfgs`,
   `tgen_c5_t1024c8.cfgs`).
5. Apply `tgen_donor_fix.diff` to the donor tree (owner action) and drop the
   `rc_Ttgen.sh` deviation afterwards.
6. Out of scope, documented: the `T ≤ 512` host guard (ab.py:554); the
   arms-without-pf6gm harness crash; the summarize.py
   `KeyError: 'pass_all_ranks'` on failed runs (masks rank-side gate output).

## Confidence

- **High** on the root cause: found by code audit, predicts the
  rank-0-survivor asymmetry, the 7/8 relative error, the written/unwritten
  split between pf6gm and mode-12, and the pinned≡scaled equivalence — and the
  single-variable A/B (sentinel clamp only) flips every gate at every T.
- **High** on the fix's default-shape safety: exact-math equality at
  MAXTOK=4096, green 600/600 soak + canonical gate band + negative control at
  T=4096. The 5-rotation T=4096 paired campaign remains owed for the ratchet
  book (screen-resolution evidence shows 0.7801×, consistent with the 0.7544×
  campaign control).
- **Medium** on the T=2048 inversion (screen resolution, n=1) — campaign
  needed before any paper claim.

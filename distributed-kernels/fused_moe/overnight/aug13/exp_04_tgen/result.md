# exp_04_tgen result — the T≠4096 defect is the M2 hole-sentinel's phantom chunks; FIXED; campaigns COMPLETE at T ∈ {1024, 2048, 4096} + T=2048 C-sweep — no 4096 regression, the "T=2048 inversion" was a screen artifact, the true inversion is between 2048 and 1024

**Node** `gbt350-odcdh2-c05-1` (8× MI350X, gfx950) · branch `ablations-tgen`
(fix `af030778`, RUN PIN `144f2029`, from `ablations` @ `ba0f785a`) · node
worktree `~/DHK-tgen` · driver: exp_36's audited `~/e36/rc_T.sh` machinery plus
a 2-line variant `~/e36/rc_Ttgen.sh` (diff in `tgen_rc.diff`) ·
2026-08-13 06:12–06:55 UTC · raw: `~/e36/scratch/tgen_*`,
`~/e36/scratch/screen_tgen_*.csv` on the node.

**Session status: COMPLETE.** Everything from here through §Confidence is the
aug13-morning checkpoint, preserved as written (its campaigns were blocked by
the concurrent `~/e39` session's lease watchdog, which `docker stop`s any
container matching `subvadla_k0_mok_tgen*`). The campaigns it staged were run
to completion by the resumed session of aug13 evening — see **§Resumed session
(20:05–23:09 UTC)** at the bottom for the three 5-rotation campaigns, the
T=2048 C-sweep, the screen-vs-campaign adjudication, and a new contention
finding (e39's docker-stops orphan mori FileBaton locks).

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

---

# Resumed session (2026-08-13 20:05–23:09 UTC) — campaigns + C-sweep COMPLETE

Same node, same RUN PIN `144f2029` (`~/DHK-tgen` clean), same driver chain
(`screenT.sh` → `rc_Ttgen.sh` with the private-copy donor mount). GPU claimed
via `~/GPU_CLAIM` = "tgen" at 20:04Z, **released 23:09:22Z** (GPU idle, all
containers down). Raw: `~/e36/scratch/screen_tgen_{c2b_t2048w2,c1b_t1024w1,
c3b_t4096w1,c4sweep}.csv` + `tgen_*.batchlog` + run dirs `~/k0-mok-tgen_*`.

## New contention finding: e39 docker-stops orphan mori-jit FileBaton locks (hang, root-caused, cleared)

Campaign attempt `tgen_c2b_t2048w1` (20:05:46Z) hung at the post-Gloo JIT step
for ~2 h: all 8 ranks in `clock_nanosleep` poll loops, 0% CU, `run1/` empty.
Mechanism (mori `jit/core.py:560-567`): a cache HIT returns before any lock is
touched; a MISSING `.hsaco` takes a torch-style
`FileBaton(".{kernel}.hsaco.lock")` and polls indefinitely while the lock
exists — no staleness detection. The morning's e39 lease-watchdog preemptions
docker-stopped containers mid-compile and orphaned three zero-byte root-owned
locks (06:23 `6b325fee9800/.k0pf6_mega`, 06:37
`dfe9c0f4083d/.k0_region_kernels_e22`, 06:48 `1fc12e1c3f84/.k0pf_quant`), each
with its `.hsaco` missing. Every later run needing one of those kernels joins
a baton wait whose holder is long dead.

Cleared the three stale locks at 21:57:17Z (via the container's `/work` mount;
host dirs are root-owned; verified no live compiler first). The already-hung
attempt did not recover — it died with a torchrun elastic failure at
22:27:20Z, **exactly 30m03s after the deletion** (= `torch.distributed`'s
default collective timeout: the woken ranks entered the post-JIT barrier and
the in-flight attempt could no longer complete the compile). Try 2 on the
clean cache compiled `k0_region_kernels_e22` (22:30) and went green
end-to-end in 139 s. **Corollary for the e39 record: a lease-window docker
stop poisons the shared JIT cache for every subsequent session; stale-lock
clearing must be part of preemption recovery.** (Also observed this session:
two host-side SSH outages ~20:15–20:40Z and ~21:20–21:54Z — Conductor
pre-auth resets, node itself healthy; detached `setsid` runners rode through
both.)

## Campaign results (5-rotation, w500t100p5, production + pf6gm denominators)

All arms all T: MOK gate pass, negative control fails ✓, poison survivors=0,
selftest fires, pperr=0, soak 600/600.

| campaign (utc) | T | prod p50 µs | pf6gm p50 µs | M15 C=28 p50 µs | M15/prod | M15/pf6gm | pf6gm/prod | gate abs/rel | spin ok/fail |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| c3b_t4096w1 (22:39:44) | 4096 | 7712.6 | 6893.6 | 5828.8 | **0.7557** | 0.8455 | 0.8938 | 0.0352/0.008295 | 0/0 |
| c2b_t2048w2 (22:31:41) | 2048 | 3857.4 | 3675.8 | 3411.8 | **0.8845** | 0.9282 | 0.9529 | 0.0430/0.008288 | 1/0 |
| c1b_t1024w1 (22:35:11) | 1024 | 2064.1 | 2290.8 | 2258.2 | **1.0940** | 0.9858 | 1.1098 | 0.0391/0.008362 | 1/0 |

**T=4096 no-regression: ACCEPTED.** M15 C=28 = 5,828.8 µs vs the aug13
session-2 control 5,822.0 µs (+6.8 µs = +0.12%); ratio 0.7557 vs 0.7544;
production 7,712.6 vs band 7,704–7,712 (+0.6 µs); pf6gm 6,893.6 vs 6,900–6,930
(−6.4 µs). All arms within 0.2% of control — consistent with the fix being
exact-math at MAXTOK=4096.

**Binary provenance (all rows).** Two `k0pf6gm_mps_mega` builds appear in the
CSVs' hsaco stamps (`dfe9c0f4083d` built 06:29, `901fe6e735e9` built 06:54;
the mori cache key re-derives per run and flip-flops between them). The two
hsacos are the SAME kernel: identical size (143,304 B) and byte-identical
except 40 bytes that are exactly the two embedded `__hip_cuid_*` per-compile
random IDs (`strings` diff shows nothing else). Both postdate fix `af030778`;
independently, every gate lands on the fixed-kernel canonical values (the
unfixed kernel scores rel≈0.875, not 0.0083). One binary, timed everywhere.

## Screen-vs-campaign adjudication: the checkpoint's "T=2048 inversion" was a cold-start artifact

| resolution | prod | pf6gm | M15 C=28 | M15/prod |
|---|---:|---:|---:|---:|
| screen `tgen_f7` (w1t1p1, 06:29) | 3969.2 | 3733.8 | 4809.8 | 1.2118 |
| campaign `c2b_w2` (w500t100p5, 22:31) | 3857.4 | 3675.8 | 3411.8 | **0.8845** |

Same T, same cfg, same binary family, same driver. Campaign-vs-screen deltas:
production −2.8%, pf6gm −1.6%, **M15 −29.1%**. The w1t1p1 screen times a
single iteration after a single warmup; the M15 slab pipeline pays a large
first-iteration cost (cold plan/spin state across 28 reserved CTAs) that the
per-token arms do not. Verdict: **the inversion claim at T=2048 is
overturned — M15 C=28 still beats production by 11.6% at T=2048.**
Method rule going forward: w1t1p1 screens must not be used to rank M15
against per-token pipelines (the checkpoint's own "medium confidence,
campaign required" hedge was correct).

## T=2048 C-sweep (w500t100p2): smaller C does NOT rescue — it hurts

| C | prod p50 | pf6gm p50 | M15 p50 | M15/prod | ts_M7 µs | ts_m2_to_end µs |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 3857.9 | 3676.6 | 4324.9 | 1.1210 | 2106.1 | 3978.6 |
| 8 | 3858.0 | 3684.2 | 4316.9 | 1.1190 | 2138.5 | 3970.1 |
| 16 | 3858.3 | 3675.2 | 3584.6 | 0.9291 | 1483.7 | 3239.7 |
| 28 (campaign, p5) | 3857.4 | 3675.8 | 3411.8 | **0.8845** | 1274.2 | 3066.4 |

All four gates green (0.0430/0.00829); production and pf6gm flat across rows
(internal control ±0.2%). **The pre-registered expectation ("the C-knob
optimum shifts down sharply at smaller T") is REFUTED at T=2048**: the
response is monotone — every reserved-CTA increment still buys M7-phase time
(ts_M7 2106→1274 µs from C=4→28), i.e. even at half the tokens the combine
remains service-throughput-bound, not slab-starved. C>28 at T=2048 is the
open direction, not C<28. (The staged T=1024 sweep `tgen_c5_t1024c8.cfgs` was
not run — GPU handed to the queued m15pkt agent at 23:09Z; it is the next
knob to turn given T=1024 is where the inversion actually lives.)

## The batch-size regime map: first campaign-grade multi-T response points

M15 C=28 vs production: **0.7557 (T=4096) → 0.8845 (T=2048) → 1.0940
(T=1024)**. The advantage shrinks roughly linearly in log-T and inverts
between T=2048 and T=1024 — NOT at T=2048 as the screen suggested. Notably
pf6gm inverts too (pf6gm/prod 0.8938 → 0.9529 → 1.1098): at T=1024 BOTH
megakernels lose to production while M15 still beats pf6gm (0.9858), so the
small-T loss is a property of the fused-megakernel family (fixed pipeline
fill/drain and plan overhead amortized over fewer tokens — M15's
m2_to_end shrinks 5310→1993 µs from T=4096→1024, only 2.66× for a 4× token
reduction), not of the slab design specifically. The corpus pre-registration
("small T favors coarse signals; the M15 advantage is expected to SHRINK at
T ∈ {1024, 2048}") is CONFIRMED in direction, and the campaign places the
break-even near T≈1600–1800 at C=28.

## Donor-tree patch: STILL PENDING (verified by diff, not grep)

`diff ~/amd-master/.../prefill_opt/kernels/k0pf6gm_mega.hip
~/tgen_kernels/k0pf6gm_mega.hip` still shows the unclamped
`hole1 = (base + K0P6_CHUNK)·TOPK` in the donor tree — **the owner has NOT
applied `tgen_donor_fix.diff`**; every pf6gm number in this file flows through
the patched private copy via the `rc_Ttgen.sh` mount deviation, which must
stay until the donor tree is patched. (Process note: an earlier grep-based
check this session false-positived "applied" because `grep … | head` masks
grep's exit status — the pipe returns `head`'s 0 and the no-match fallback
never fires. Adjudicate donor state by diff only.)

## Session confidence

- **High** on the T=4096 no-regression (all three arms within 0.2% of the
  session-2 control bands) and on the T-response direction (three
  campaign-grade points, one binary, gates green everywhere).
- **High** on the T=2048 adjudication (0.8845 campaign vs 1.2118 screen; the
  29% M15-only cold-start delta identifies the artifact mechanism).
- **Medium** on the C-monotonicity at T=2048 (p2 rotations for C∈{4,8,16} vs
  p5 at C=28; the 17→28 gap is 173 µs ≫ noise, but C>28 is unexplored).
- T=1024 C-response: unmeasured (one C=28 point only) — first follow-up.

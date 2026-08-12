# exp_33 — bottleneck attribution at the ratchet (paper Q3)

**Date:** 2026-08-12 08:53–09:12Z · **Node:** `gbt350-odcdh2-c05-1` (8× MI350X, gfx950)
**Commit:** `ca5b683f` (`codex/distributed-hipkittens-scaffold`), `K0P6_MPS_SRC_REV 26`, `K0P6_MPS_ASCALE_TM 1`
**Config:** `K0_MPS_CFG="C=16,g=353,mode=12,flush_rows=16,timestamps=1"`, arms `production,pf6gm_mega,mps_mega`
**Resolution:** 2 campaigns × 5 rotations = **n=10**, 500 warmup / 100 timed, 600-epoch soak, poison on
**Data:** `phase_stamps.json` · **Raw:** `raw/` (both campaign logs, both `summary.json`, all 80 rank JSONs)

---

## 1. Verdict

**M7 (GEMM-2) is the bottleneck at the ratchet, at 2,701.8 ± 23.7 µs = 46.2 % of the
measured kernel interior**, narrowly ahead of M6 (GEMM-1) at 2,453.3 ± 2.5 µs =
41.9 %. Together the two GEMMs are **88.1 % of the interior**. Everything the
scheduling work has been arguing about — plan and combine — is now small: plan
M3–M5 is 372.8 µs (6.4 %) and combine M8/M9 is 324.2 µs (5.5 %).

Three findings beyond the headline:

1. **The M7/combine split point is not real.** M7 and combine are anti-correlated
   at **r = −0.904**; their sum is 3,026.1 ± 10.2 µs with sd 32.1 µs, against an
   sd of 100.5 µs that independence would predict from their individual sds
   (75.1 and 66.8). M7 and combine are **one coupled block**: when GEMM-2's last
   CTA lands early, the combine simply waits longer for peer arrivals. Do not
   attack combine on its own — the time will move, not disappear.
2. **The M7 epilogue surcharge is ~817–898 µs**, i.e. the payload carriage costs
   about **30–33 % of M7**. This is a *proxy*, not the specified measurement
   (§5): `pf6gm_mega` cannot emit stamps at all, so the specified
   `M7(mps) − M7(pf6gm)` is not computable in this harness.
3. **Dispatch peer wait is zero.** `[MPS SPIN]` reads `fail_max=0` in all 10
   rotations and `success_max ≤ 1` — at most one poll iteration *ever*, against a
   2,000,000 limit, over 500 warmup + 100 timed + 600 soak epochs per rotation.
   exp_10 reproduces exactly at this shape.

**The ratchet reproduced.** `mps_mega` = 6,487.5 µs (e33a) and 6,496.6 µs (e33b)
against the recorded 6,482.7 µs — +0.07 % and +0.21 %. Pooled 6,493.8 µs =
**0.8424× production**. Every gate read green in both campaigns.

---

## 2. `phase_stamps.json` schema

Top-level keys: `schema_version` (`"exp33-phase-stamps-1"`), `experiment`,
`generated_utc`, `commit`, `config`, `units`, `caveats`, `resolution`, `gates`,
`peer_wait`, `arms`, `supplementary`, `derived`, `per_rotation_raw`.

- **`commit`** — `repo_head`, `branch`, `K0P6_MPS_SRC_REV`, `K0P6_MPS_ASCALE_TM`,
  `mps_build_identity_per_rotation` (jit dir + hsaco sha256 + source sha256, taken
  from the in-container `pf6mps` record in each rank JSON), `build_identity_note`.
- **`config`** — `K0_MPS_CFG`, `arms`, `campaigns[]` (tag, output root, log,
  per-campaign `arm_p50_us_median`), `rotations_total`, `warmup_iters`,
  `timed_iters`, `soak_iters`, `K0_MOK_POISON_OUT`, `K0_SPIN_LIMIT`,
  `mps_config_decoded` (the harness's own decode of the packed cfg word), `shape`,
  `node`.
- **`units`** — `tick_us = 0.01` plus the note that the 2.2 GHz shader clock is
  the wrong divisor.
- **`caveats`** — 8 entries; the load-bearing ones are reproduced in §7.
- **`gates`** — `mok_gate_pass_counts` per arm, `mok_gate_failures[]`,
  `control_fails_values`, `soak[]` (one entry per rotation), `poison_selftest`,
  `poison_survivor_values`, `pperr_max`, and a single computed `all_green` bool.
- **`peer_wait`** — `[MPS SPIN]` success/fail maxima per rotation + interpretation.
- **`arms.<arm>`** — `p50_us` (rank-max), `p50_us_values_per_rotation`,
  `p50_us_per_campaign_median`, `p50_us_rank0`, `ratio_vs_production`,
  `ratio_vs_production_rank0`, `phase_instrument`, and **`phases`**, a dict keyed
  by phase name where each entry carries
  **`{phase, source, us_rank_max, us_rank0, n, stderr}`** plus `sd`,
  per-rotation `values_us`, and — where a value is absent — an explicit
  `reason` / `us_rank_max_reason` string. `mps_mega` additionally carries
  `stacked_bar_phases`, the five phases that partition the kernel.
- **`supplementary.pf_full_stage_profile_us`** — the same-run `pf_full` stage
  profile (`maxrank` and `rank0`), the source of the surcharge proxy.
- **`derived`** — `identity_check_us`, `dispatch_M0toM2_residual_estimate_us`,
  `M7_plus_combine_coupled_block_us` (incl. `pearson_r_M7_vs_combine`),
  `m7_epilogue_surcharge_proxy_us` (both proxies, each with confidence),
  `first_ready_decoded`.
- **`per_rotation_raw`** — all 10 rotations verbatim: campaign, run index, arm
  order, the eight raw `[MPS TS]` stamps, every derived µs value, and the spin
  counters. Every number in this document is in this file.

---

## 3. Phase decomposition — `mps_mega` (the ratchet)

Device stamps, **rank 0 only**, **final soak epoch**, n=10, 1 tick = 0.01 µs.

| phase | µs (rank-0) | stderr | sd | % of interior | % of p50 | rank-max |
|---|---:|---:|---:|---:|---:|---|
| dispatch M0–M2 | *not stamped* | — | — | — | (9.9 % residual) | n/a |
| plan M3–M5 | **372.79** | 0.67 | 2.11 | 6.4 % | 5.7 % | n/a |
| M6 GEMM-1 | **2,453.30** | 2.51 | 7.95 | 41.9 % | 37.8 % | n/a |
| M7 GEMM-2 (total) | **2,701.84** | 23.74 | 75.06 | 46.2 % | 41.6 % | n/a |
| — of which GEMM proper | ~1,804–1,885 *(proxy)* | — | — | ~31–32 % | — | n/a |
| — of which epilogue surcharge | **~817–898** *(proxy)* | — | — | ~14–15 % | — | n/a |
| combine M8/M9 | **324.21** | 21.12 | 66.78 | 5.5 % | 5.0 % | n/a |
| **interior M3–M9** | **5,852.14** | 9.56 | 30.23 | 100 % | 90.1 % | n/a |
| *service drain (overlaps M7+combine, **not additive**)* | *2,810.88* | *22.45* | *70.98* | — | — | n/a |
| *M7 + combine (coupled block)* | *3,026.05* | *10.16* | *32.14* | *51.7 %* | *46.6 %* | n/a |

`rank-max` is **n/a for every row by construction**, not by omission: the
`[MPS TS]`/`[MPS SPIN]` print block in `e004pf_k0pf_ab.py` sits inside
`if rank == 0:` and reads `pf6_state["mps_state"]`, which is never all-reduced and
never written to the per-rank JSON. The reported value is the device **CTA-max
within rank 0**.

**Identity check:** `plan + M6 + M7 + combine = interior` holds to
**0.0 µs in every one of the 10 rotations** — the stamps are mutually consistent
and the parse is by name, not position.

**Dispatch M0–M2** has no stamp in this build: the 8-slot layout has no
kernel-start entry (the former `KSTART` slot now holds `M5_DONE`), and
`FIRST_READY_inv` is a *min* stamp over all 600 soak epochs, so it cannot be
differenced against final-epoch maxima. The residual
`p50 − interior = 6,493.83 − 5,852.14 =` **641.69 µs** is recorded as
`derived.dispatch_M0toM2_residual_estimate_us` and is **instrument-mixed** (a
timed-iteration rank-max total minus a soak-epoch rank-0 interior); it absorbs
M0+M1+M2, launch, and epoch skew. Treat it as an upper bound, good to ~100 µs.
For scale, the archived `[PF6GM DECOMP]` M0+M1+M2 for `pf6gm_mega` is ≈768 µs.

## 4. Phase decomposition — `production` and `pf6gm_mega`

`production` **does** have a phase signal, contrary to the pre-registered
expectation that it would have none: the harness's own
`stage_profile.production` (`[K0PF PROFILE] production`, `K0_STAGE_PROFILE=1` by
default) gives three coarse phases with a genuine cross-rank `dist.all_reduce(MAX)`,
so both variants exist here.

| arm | phase | µs rank-max | stderr | µs rank-0 | stderr |
|---|---|---:|---:|---:|---:|
| production | dispatch | **918.70** | 2.01 | **902.85** | 1.93 |
| production | gemm | **5,945.49** | 7.64 | **5,904.68** | 6.20 |
| production | combine | **1,356.02** | 11.68 | **978.41** | 8.71 |
| production | *(sum)* | *8,220.21* | — | *7,785.94* | — |
| pf6gm_mega | all phases | **null** | — | **null** | — |

The `production` rank-max stages sum to **+6.6 % above** its p50 of 7,708.3 µs
(the rank-0 sum is only +1.0 % above the rank-0 p50) because the max over ranks is
taken **independently per stage**, and max-of-sums ≤ sum-of-maxes. This is a
different instrument from the MPS device stamps — HIP-event brackets around 5
eager reps with a host barrier between reps — so it is **not** comparable
tick-for-tick with §3. Note also that production's combine carries the largest
rank spread of any phase measured tonight (1,356 rank-max vs 978 rank-0, +38 %),
which is the all-to-all skew the megakernel arms hide inside their epilogue.

`pf6gm_mega` phase data is `null` with `reason` recorded in the JSON. It is not a
missing instrument, it is an absent one: `pf6gm_mega` is a **different kernel**
(`k0pf6gm_mega`, descriptor `desc_gm`, 55 slots) and is never passed the
`mps_state` timestamp block, so it emits no stamps. The one per-phase instrument
that does exist for it — the `K0_PF6GM_DECOMP` cumulative-prefix path
(`ab.py:5228`, which is what produced the archived `[PF6GM DECOMP]` module
tables) — **is absent from `run_campaign.sh`'s `-e` forwarding list** (lines
110–147), so it cannot be reached through the campaign at all, and a direct
`torchrun` is off-limits (it defaults the absolute tolerance to 1.0 and
`K0_PF6GM_G` to 2). Making it reachable is a one-line harness edit that exp_33
does not own; it is the single highest-value harness change for the paper's
Fig-4 waterfall, because it would turn every surcharge proxy below into a
same-run measurement.

## 5. The M7 epilogue surcharge

The specified formula `M7(mps_mega) − M7(pf6gm_mega)` is **not computable** — see
§4. Two clearly-labelled proxies, both recorded in
`derived.m7_epilogue_surcharge_proxy_us`:

| proxy | subtrahend (payload-free GEMM-2) | value | confidence |
|---|---:|---:|---|
| **A** — same-run `pf_full.n2_p2`, rank-max | 1,885.00 ± 2.74 | **816.84 µs** | medium |
| **A′** — same-run `pf_full.n2_p2`, rank-0 | 1,858.30 ± 2.66 | **843.54 µs** | medium |
| **B** — archived `[PF6GM DECOMP] M7_n2_phase2` (2026-08-06, different commit) | 1,803.98 | **897.86 µs** | low |

Proxy A's subtrahend is the **same donor `n2_phase2` body that `pf6gm_mega`'s M7
runs**, measured in these very runs, which is why it is the primary. All three
proxies mix instruments — the minuend is a device wall-clock delta from the final
soak epoch, the subtrahend is a HIP-event measure of a separately launched kernel
with its barriers outside the timing window. Read the result as **"the epilogue
payload costs order 800–900 µs, ~30–33 % of M7"**, not as a calibrated number.
The two independent subtrahends agreeing to 4.5 % (1,804 vs 1,885) is the reason
to trust the order of magnitude.

## 6. `[MPS SPIN]` — the peer-wait instrument

| quantity | value |
|---|---|
| `fail_max` | **0** in all 10 rotations |
| `success_max` | **0** in 9 rotations, **1** in 1 rotation (e33b run3) |
| `limit` | 2,000,000 |

Each rotation's counters are running maxima that are never reset, so they cover
500 warmup + 100 timed + **all 600 soak epochs**. A maximum of one successful
poll iteration ever means M2 essentially never waits on a peer chunk:
**dispatch peer wait is ~0 at this shape**, reproducing exp_10 at routing std=0.
This is a *fact about the shape*, not about the config — it is exactly why the
skew sweep (exp_36) is the experiment that can make a service pool re-enter.

## 7. Gate ladder — all green, both campaigns, all 10 rotations

Verbatim (one instance each; counts from `gates` in the JSON):

```
[MOK GATE] production max_abs=0.027344 relative=0.005776 pass=True
[MOK GATE] pf6gm_mega max_abs=0.039062 relative=0.008293 pass=True
[MOK GATE] mps_mega  max_abs=0.039062 relative=0.008293 pass=True
[MARK] control_fails=True
[MPS SOAK] completed=600/600 pperr=0 poison=0 poison_epoch=-1 pass=True
[POISON SELFTEST] arm=mps_mega one_row_poisoned_fails=True nonfinite=57344 relative=nan
[POISON SELFTEST] arm=pf6gm_mega one_row_poisoned_fails=True nonfinite=57344 relative=nan
[POISON] eager arm=mps_mega survivors=0
[POISON] post_timing arm=mps_mega survivors=0
[MOK SYNTHETIC EAGER] status=valid_diagnostic ... gate_ok=True
[K0PF GATE] plan_equivalence pass=True ... perr=0/0 padded=33440 T_loc=21816
[K0PF GATE] quant_exact pass=True byte_diffs=0 scale_diffs=0
[K0PF GATE] gather_exact pass=True byte_diffs=0 scale_diffs=0
[K0PF GATE] combine_bit_exact pass=True bf16_bit_diffs=0
```

- `[MOK GATE] pass=True` **10/10 for each of the three arms**, zero failures.
- `control_fails` takes exactly one distinct value across the run: `True`.
- All 10 `[MPS SOAK]` lines are identical: `600/600`, `pperr=0`, `poison=0`,
  `poison_epoch=-1`, `pass=True`. **`K0_MPS_SOAK_ITERS` was left unset so the
  harness default of exactly 600 applies; it was not touched.**
- **`pperr = 0` everywhere** — `gates.pperr_max = 0`, taken as the max over every
  `pperr=` occurrence in both logs.
- Poison survivors: the only value present anywhere is `0`.
- `gates.all_green = true` (computed from all of the above, not asserted).

**Build identity.** All 10 rotations loaded **one** build — jit `5635a2f2a370`,
hsaco `a6e4189ce224…`, source `e957bd00b48d…` — verified from the in-container
`pf6mps.hsaco_sha256` in each rank JSON. Note the host-side `latest/` symlink
stamp that `screen.sh` samples *outside* the run reported a flip between the two
campaigns; the in-container record shows that flip did not reach either run.
The symlink is not the authority (harness gotchas 3–4). `summary.json`'s
`kernel_hsaco_sha256` / `kernel_source_sha256` are empty dicts in this harness
build and must not be used.

## 8. Caveats (the load-bearing ones)

1. **FINAL SOAK EPOCH.** Every `[MPS TS]` value is a running device *max* that is
   never reset, so after the 600-epoch soak the stamps describe the **final soak
   epoch**, not a timed iteration. The phase table and the end-to-end p50 are two
   different measurements of the same kernel; do not sum the phases and compare
   the total to `arm_p50_us`. (The §3 "% of p50" column is a presentational
   normalisation only, and is why `dispatch` there is a residual.)
2. **1 tick = 0.01 µs** (100 MHz wall clock). The 2.2 GHz shader clock is the
   wrong divisor.
3. **Rank-0 only** for every `mps_mega` phase, and the value is the CTA-max within
   rank 0. No rank-max variant of the MPS phases exists in this harness.
4. **`pf6gm_mega` emits no stamps**, so the specified surcharge is a proxy (§5).
5. **`production`'s split is a different instrument** and its rank-max stages
   over-count by 6.6 % when summed.
6. **`service_drain` overlaps M7 and combine** and must never be added into the
   stacked bar. The five `stacked_bar_phases` in the JSON are the ones that
   partition the kernel.
7. `FIRST_READY` decodes as `2**64 − FIRST_READY_inv` but is a *min* over all 600
   epochs; it is recorded raw and must not be differenced against final-epoch
   stamps.
8. The two campaigns are true replicates (same commit, same build, same cfg, same
   seed, `K0_SYNTH_ROUTE` unset) and are pooled; per-campaign medians are kept in
   the JSON so any drift stays visible.

## 9. Confidence and measurement resolution

**High confidence in the ranking** M7 > M6 ≫ plan > combine, and in the coupled-block
finding. Measured stderr at n=10:

| quantity | stderr | as % of its own mean | resolves |
|---|---:|---:|---|
| plan M3–M5 | 0.67 µs | 0.18 % | sub-1 % changes easily |
| M6 GEMM-1 | 2.51 µs | 0.10 % | sub-1 % changes easily |
| M7 GEMM-2 | 23.74 µs | 0.88 % | ~1 % changes |
| combine M8/M9 | 21.12 µs | 6.5 % | only large changes |
| M7 + combine | 10.16 µs | 0.34 % | sub-1 % changes |
| interior M3–M9 | 9.56 µs | 0.16 % | sub-1 % changes |
| end-to-end p50 | — | 0.14 % between campaigns | ~0.5 % |

Per-rotation sd for M6 is 7.95 µs and for plan 2.11 µs, against the prior
measured sigmas of 4.8 µs (M6) and 4.7 µs (plan) — the same order, with M6 a
little wider and plan a little tighter here. The phase stamps continue to resolve
~1 %, as documented, while a screen resolves ~6.6 % end-to-end.

**The one number to distrust** is `combine M8/M9` in isolation: its stderr is
6.5 % of its mean and it is anti-correlated with M7 at r = −0.90. Any future
result that claims to have moved combine by less than ~60 µs, measured alone, is
inside the noise of a quantity that is really just M7's slack.

## 10. Largest remaining term, and what would attack it

**M7 is the term to attack: 2,701.8 µs, 46.2 % of the interior**, and it
decomposes into roughly 1,860–1,885 µs of GEMM-2 proper plus **~820–900 µs of
epilogue surcharge** (§5). But the r = −0.90 coupling with combine says the real
target is the **3,026.1 µs M7+combine block**, and that the surcharge is the only
part of it a scheduling change can actually delete — trimming GEMM-2's tail alone
would hand the time straight to the combine's peer wait.

That points, in expected-value order, at exactly the queued work:

- **mode 14 / coarse readiness (exp_34, then the exp_35 waterfall rung d)** is the
  direct attack on the surcharge: it deletes the per-row protocol
  (`nc_arr`/`pushed`/`row_ready`/M8 polls) that the epilogue is carrying, which is
  the ~820–900 µs term. The pre-registered band 5,990–6,440 µs implies a
  −54…−504 µs move, and the surcharge measured here is the first evidence that a
  win of that size is physically available inside M7 rather than assumed.
- **exp_31, the phase-2 VMEM hint (`14 + kGM`)** is a one-line change aimed at
  the GEMM-2 body, i.e. the other ~1,880 µs of M7.
- **exp_26's mask ladder and the M6 software-pipeline depth** target M6's
  2,453.3 µs, the second-largest term and the one with the *tightest* stamp
  (stderr 2.51 µs), so it is also the cheapest phase to judge a change on.
- **Plan M3–M5 is done as a target.** At 372.8 µs / 6.4 % it holds less headroom
  in total than the surcharge inside M7 alone; exp_24 and exp_27 took the fat out
  and there is nothing left worth a campaign.
- **Combine M8/M9 is not a target either**, for the coupling reason above. Its
  324.2 µs is slack absorbing M7's jitter, not work.

Re-run these stamps after the next landed optimisation and re-pick from the new
profile, not from this one.

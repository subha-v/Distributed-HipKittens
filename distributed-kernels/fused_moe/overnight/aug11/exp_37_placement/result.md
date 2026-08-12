# exp_37 — placement adjudication (paper Fig 5 / Q2)

**Does dedicating CTAs to communication ever win on AMD? — No. Not once, not
anywhere on the reachable axis, and the loss is monotone in the size of the
dedicated pool.** Six settings from `C=4` to the `C=64` dedicated-pool design
order themselves perfectly by how many CTAs are taken away from compute:

```
C=4   6463.3      C=8   6470.3      C=12  6490.4      C=16  6497.5      C=32  6735.3      C=64(mode 2)  6837.5   us
0.8393x           0.8406x           0.8428x           0.8436x           0.8746x           0.8881x     of production
```

**And yes, there is a new ratchet.** The standing ratchet `C=16` is beaten by
both smaller pools in **7 of 7 paired rounds** with intervals nowhere near
zero: `C=8` by **−27.2 µs [−32.3, −22.1]** and `C=4` by **−34.2 µs
[−39.5, −28.9]**. `C=4` and `C=8` are **not distinguishable from each other**
(−7.0 µs, CI [−15.5, **+1.5**], signs mixed 5:2) — the honest statement is that
the winning region is `C ≤ 8`, not that `C=4` specifically is best.
**Recommendation: move the ratchet to `C=4,g=353,mode=12,flush_rows=16`**, with
`C=8` an equally defensible choice; reasoning and the bar-clearing argument are
in *Ratchet recommendation* below.

The mechanism is visible and it is exactly the paper's claim: **M6, the GEMM,
does not move at all** (−1.0 µs, CI [−6.7, +4.8]; the arm means span 1.8 µs,
0.07 %, across the whole `C = 4…32` range), so the CTAs handed to the service pool are taken from
work that had nothing to gain. **The entire effect lives in the
payload-carrying phases** (`M7 + combine`: −41.9 µs for `C=8`, −31.1 µs for
`C=4`). Peer wait stayed at zero everywhere.

All **27 campaigns** (135 rotations) are gates-green with `pperr=0`.

---

## What was run

| | |
|---|---|
| pin | `b5215081b87cf6ba0619255cef87288e081903d3`, `SRC_REV 26`, `K0P6_MPS_ASCALE_TM 1` (verified before every batch; the launcher refuses otherwise) |
| node | `gbt350-odcdh2-c05-1`, 8× MI350X gfx950 |
| campaign | 5 rotations, 500 warmup / 100 timed, `K0_MPS_SOAK_ITERS=600` (untouched), T = 4096 |
| arms per run | `production, pf6gm_mega, mps_mega` — production is the **same-run** denominator |
| batches | **A** = 12 campaigns, 4 interleaved rounds of {C=16, C=8, C=4}; **B** = 6 campaigns, 2 rounds of {C=12, C=32, C=64/mode 2}; **C** = 3 more rounds of {C=16, C=8, C=4} |
| total | 27 campaigns, 12:48Z → 14:21Z, one 8-GPU job at a time |

Arms were **rotated inside each round** (the position of each arm within a
round is permuted round to round) so session drift cannot masquerade as a C
effect. It did not need to: the same-run production denominator moved only
**0.30 % peak-to-peak** across the whole 1.5 h session (7690.8 – 7713.8 µs).

Every campaign's `K0_MPS_CFG` is verified **from the kernel side**, not from the
launcher's intent: all four knobs are read back out of all 8 rank JSONs of all
5 rotations (`config.mps_config`) and required to be identical, together with
`mps_soak_iters = 600`. This is the check that catches the "partial config
LOOKS like a pass" failure.

---

## The arm table

`p50` = median over the campaign's 5 rotations; the arm's number is the median
over its campaigns. `ratio` is same-run `mps_mega / production`. Stamps are
device stamps (1 tick = 0.01 µs) averaged over every rotation of the arm.

| arm | mode | C | g | n | p50 median µs | mean | min–max | ratio | Δ vs C=16 | M6 | M7 | combine |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| **C=4** | 12 | 4 | 353 | **7** | **6464.2** | 6463.3 | 6455.4–6470.3 | **0.8393** | **−34.2** | 2450.0 | 2593.4 | 415.7 |
| **C=8** | 12 | 8 | 353 | **7** | **6472.0** | 6470.3 | 6462.0–6477.0 | **0.8406** | **−27.2** | 2449.5 | 2674.6 | 323.6 |
| C=12 | 12 | 12 | 353 | 2 | 6490.4 | 6490.4 | 6490.2–6490.7 | 0.8428 | −7.1 † | 2450.9 | 2731.0 | 268.3 |
| **C=16** (ratchet) | 12 | 16 | 353 | **7** | **6498.0** | 6497.5 | 6492.6–6501.8 | 0.8436 | — control | 2451.0 | 2696.1 | 344.1 |
| C=32 | 12 | 32 | 353 | 2 | 6735.3 | 6735.3 | 6734.1–6736.5 | 0.8746 | +237.8 † | 2449.2 | 3027.6 | 179.0 |
| **C=64 dedicated pool** | **2** | 64 | 1 | 2 | 6837.5 | 6837.5 | 6835.7–6839.4 | 0.8881 | +340.0 † | 2480.5 | 2916.1 | 442.3 |
| C=0 | 12 | 0 | — | — | `requires_mode_14` | | | | | | | |
| C=0 / 8 / 16 | 14 | — | — | — | `pending_exp_34` | | | | | | | |

† `C=12`, `C=32` and mode 2 were run in batch B, which contained no `C=16`
campaign, so their deltas are **unpaired** (cross-batch) and are quoted for the
shape of the axis only. Their within-arm replicates are tight (≤ 2.4 µs apart).
`C=0` in mode 12 is rejected by the validator (`moe_mps_adapter.cuh:373/375`);
it is recorded as `requires_mode_14` and **not** faked with a large-C proxy.
Mode 14 had not landed at this pin and was not run on my own initiative.

Every campaign value individually, in run order, is in `placement.json`
(`arms[].campaigns`, `arms[].campaign_ids`), along with all 5 rotation p50s per
campaign (`arms[].rotation_p50_us`).

---

## The paired statistics

**Clustering.** The unit of inference is the **campaign**, never the rank and
never the rotation. The 8 ranks share one input draw and one collective, and the
harness has already collapsed them (each rotation reports one rank-max-aligned
p50); the 5 rotations share the campaign's build, container launch and routing
draw. Campaign values are then **paired by interleave round**, so each
comparison is a within-round difference.

| comparison | n pairs | Δ µs | 95 % CI | t (clustered) | per-round Δ | sign test | rank-sum (unpaired) |
|---|---:|---:|---|---:|---|---:|---:|
| **C=8 − C=16** | 7 | **−27.2** | **[−32.3, −22.1]** | −13.03 | −22.0, −32.4, −23.2, −26.9, −37.0, −24.0, −24.8 | 0.016 | 0.00058 |
| **C=4 − C=16** | 7 | **−34.2** | **[−39.5, −28.9]** | −15.86 | −37.2, −30.2, −39.9, −41.7, −31.0, −25.7, −33.6 | 0.016 | 0.00058 |
| C=4 − C=8 | 7 | −7.0 | **[−15.5, +1.5]** | −2.02 | −15.2, +2.2, −16.7, −14.8, **+6.0**, −1.7, −8.8 | 0.45 | 0.038 |

**Both candidates clear the noise decisively.** Every one of the 7 rounds is
negative for both, the campaign value sets are completely disjoint
(`max C=8` 6477.0 < `min C=16` 6492.6; `max C=4` 6470.3 < 6492.6), and the
intervals sit 22–39 µs away from zero. The effect is ~0.42 % (C=8) and ~0.53 %
(C=4) end-to-end — far below the 5 % screen resolution, which is exactly why
this was run as 21 paired campaigns and not as a screen.

**`C=4` vs `C=8` is a genuine tie.** The paired interval spans zero and two of
the seven rounds have the *opposite* sign. The unpaired rank-sum test reports
p = 0.038, but it ignores the round blocking that the design exists to exploit
and pools two batches whose `C=4` values differ (batch A mean 6458.8, batch C
mean 6469.4, while `C=8` barely moved); the blocked test is the authoritative
one. **I am not calling `C=4` better than `C=8`.** Even taken at face value the
gap would be 0.1 %.

**On unclustered inference.** Had I treated the 105 rotations as independent
samples, the C=4-vs-C=8 comparison would have reported t = −2.76 instead of
−2.02, and every real effect would have been inflated similarly
(C=8 vs C=16: −11.88 unclustered vs −13.03 clustered). Both invalid values are
stored in `placement.json` under `INVALID_unclustered_rotation_t` so the
inflation is auditable rather than invisible.

---

## Which phase pays for the C change

Same clustering: campaign-mean stamp, paired by round, n = 7.

| phase | C=8 − C=16 | C=4 − C=16 | C=4 − C=8 |
|---|---:|---:|---:|
| plan (M3–M5) + M6 | −0.0 [−8.2, +8.1] | +1.0 [−5.9, +7.9] | — |
| **M6 (the GEMM)** | **−1.5 [−9.4, +6.4]** | **−1.0 [−6.7, +4.8]** | +0.5 [−6.7, +7.8] |
| M7 | −21.5 [−66.2, +23.3] | −102.6 [−162.8, −42.5] | −81.2 [−132.7, −29.7] |
| combine | −20.5 [−62.7, +21.8] | **+71.6** [+25.0, +118.2] | **+92.0** [+41.8, +142.3] |
| **M7 + combine** | **−41.9 [−47.9, −35.9]** | **−31.1 [−53.0, −9.1]** | +10.9 [−6.2, +27.9] |

**M6 is flat — confirmed, and much more tightly than before.** Across the whole
`C = 4…32` range the arm means are 2450.0 / 2449.5 / 2450.9 / 2451.0 / 2449.2 µs:
a **1.8 µs (0.07 %) spread**, with a paired CI of ±7 µs. The prior claim was
"flat within 1.5 %"; the real number is flat within 0.1 %. Taking 12 CTAs away
from the GEMM and giving them to the service pool changes the GEMM's duration by
nothing measurable, which is the whole argument: one block per CU means there is
no issue-slot contention for a dedicated pool to relieve, so the pool is a pure
N/(N−C) capacity tax. The only arm where M6 *does* move is the dedicated-pool
design itself (mode 2, `C=64`: 2480.5 µs, +30 µs).

**The prior claim that "the entire difference lives in M7" is too strong — I am
refuting the split, not the location.** M7 alone and combine alone move in
*opposite* directions and each spans zero for at least one candidate: the
M7/combine boundary shifts with C (at `C=4`, 102.6 µs leaves M7 and 71.6 µs
reappears in combine). Their **sum** is significant for both candidates and is
the right size to explain the end-to-end delta (−41.9 and −31.1 µs against
end-to-end −27.2 and −34.2 µs). The defensible statement: **the C effect is
entirely inside the payload-carrying phases, and not in the GEMM.**

---

## Gates

Aggregated over all **135 rotations** (27 campaigns × 5), with zero exceptions:

```
135  [MOK GATE] mps_mega max_abs=... relative=... pass=True
135  [MARK] control_fails=True
135  [MPS SOAK] completed=600/600 pperr=0 poison=0 poison_epoch=-1 pass=True
540  pperr=0                       (no nonzero pperr anywhere)
270  [POISON SELFTEST] one_row_poisoned_fails=True nonfinite=57344
540  [POISON] {eager,post_timing} arm={mps_mega,pf6gm_mega} survivors=0
120  [MPS SPIN] chunk_poll success_max=0 fail_max=0 limit=2000000
 15  [MPS SPIN] chunk_poll success_max=1 fail_max=0 limit=2000000
```

Verbatim per-campaign blocks are in `raw/gate_evidence.txt`.

**The one deviation from "expected 0/0":** 15 of 135 rotations (11 %) report
`success_max=1` instead of 0 — a single *successful* chunk poll, meaning one
peer-wait poll observed data not yet arrived and then found it immediately.
`fail_max` is **0 in all 135 rotations**, so no poll ever exhausted. It is
spread across every arm — `C=4`, `C=8`, `C=12`, `C=16`, `C=32` and mode 2 all
show it — and does not track the effect. Peer wait remains, for practical purposes, zero — which
matters for the interpretation above: the service pool is not winning back time
by hiding waits, because there are no waits to hide.

---

## Ratchet recommendation

**Yes — I believe this clears the bar to move the ratchet, from
`C=16,g=353,mode=12,flush_rows=16` to `C=4` (or `C=8`).** The reasoning:

1. **The delta is small but the evidence is not.** 7 of 7 paired rounds
   negative, CI [−39.5, −28.9] µs for `C=4`, disjoint campaign value sets,
   exact rank-sum p = 0.00058. A 0.5 % effect measured 21 times beats a 5 %
   effect measured once.
2. **It reproduces across sessions and across a re-launch.** Batch A and batch C
   are separated by 40 minutes and by batch B; both give the same ordering and
   overlapping intervals. And my `C=16` control lands at **6497.5 µs (n=7)**
   against exp_36's independent **6498.1 µs (n=3)** — two sessions agreeing to
   0.6 µs (0.01 %) on the control arm. Every exp_36 point is confirmed at higher
   n: `C=8` 6469.5 → 6470.3, `C=4` 6466.1 → 6463.3, `C=32` 6719.3 → 6735.3,
   mode 2 6829.6 → 6837.5.
3. **It is free.** `C` is a runtime config word, not a code change: no rebuild,
   no new protocol, no resource-tuple movement, nothing to gate beyond what is
   already green here.
4. **It is mechanistically coherent**, not a fluke of the timer: the phase
   stamps put the gain in the payload phases and show the GEMM untouched, and
   the whole axis is monotone over a 374 µs span.

**Which value.** `C=4` has the better point estimate but is statistically tied
with `C=8`; I would take `C=4` because it is also the direction the entire axis
points and it pre-positions the trend line for exp_34's `C=0`, while noting that
anyone preferring the more conservative interior point loses nothing measurable
by choosing `C=8`.

**One caveat the ratchet owner must apply.** Every arm here carries
`timestamps=1`, so the comparison is internally clean but the *absolute*
microseconds are not comparable to a stamps-off number. The standing published
ratchet of **6482.7 µs / 0.8408×** does not reproduce in this session — my
stamps-on `C=16` control is 6497.5 µs / 0.8436×, and it matches exp_36's
stamps-on `C=16` exactly. So transfer the **paired delta**, not the absolute:
the new ratchet should be quoted as **−34.2 µs against whatever `C=16`
measures under the same instrumentation** (≈ 6448 µs if applied to 6482.7), and
re-measured stamps-off before it is published as a headline number.

---

## `placement.json` schema

Top level: `schema_version` (`exp_37.placement.v1`), `generated_utc`, `commit`,
`experiment`, `host`, `harness` (campaign shape, T, stat unit, clustering
policy), `caveats`, `arms`, `phase_attribution`, `pairwise_comparisons`,
`campaigns_raw`.

`arms[]` — one record per arm:
`arm_id, mode, C, g, flush_rows, status` (`measured` | `requires_mode_14` |
`pending_exp_34`), `n_campaigns`, `campaigns` (p50 per campaign, in run order),
`campaign_ids`, `interleave_rounds`, `p50_median`, `p50_mean`, `p50_min`,
`p50_max`, `rotation_p50_us` (all 5 rotations per campaign),
`production_p50_same_run` (per campaign) and `..._median`,
`ratio_vs_production` (+ per campaign), `delta_vs_ratchet_us`, `paired_ci`
(n_pairs, mean, sd, CI, t, df, sign-test p, exact rank-sum p, disjointness,
per-round deltas, and the deliberately-labelled `INVALID_unclustered_rotation_t`),
`phase_stamps` (`plan`, `M6`, `M7`, `combine`, `servicedrain`; each with
per-campaign final-epoch values and rotation mean/median/sd/min/max),
`spin_success_max`, `spin_fail_max`, `gates_green`, `n_campaigns_gates_green`.

`phase_attribution.comparisons[]` — the clustered paired stamp deltas quoted
above, per phase, with CI, `spans_zero` and per-pair values.

`campaigns_raw[]` — every campaign: tag, idx, outdir, arm, the read-back
`mps_config`, soak iters, rank-file count, status, per-rotation p50s for
mps/production, all gate fields, and both the final-epoch and all-rotation
stamps.

Raw evidence in `raw/`: `e37_raw.json` (collector output), `gate_evidence.txt`
(verbatim gate lines per campaign), the three driver logs, the three screen
CSVs, the three config lists, and the launcher/collector/stats scripts as run.

---

## Confidence

- **High** — dedication never wins, and is monotonically harmful. Six settings,
  374 µs of span, tight replicates, two independent sessions agreeing, and a
  mechanism visible in the stamps. This is Fig-5 quality.
- **High** — `C=8` and `C=4` both beat `C=16`. 7/7 paired rounds, disjoint value
  sets, intervals far from zero.
- **High** — M6 is untouched by C (±7 µs paired CI, 0.07 % spread across the
  axis).
- **Medium-high** — the ratchet move itself. The statistics are strong; the
  reservation is only that the absolute number must be re-established
  stamps-off, since the published 6482.7 µs baseline does not reproduce under
  this instrumentation.
- **Medium** — the *ordering* of `C=12` and `C=32` and mode 2 relative to the
  core arms, which is unpaired cross-batch at n=2 (though the gaps, 238 and
  340 µs, dwarf any plausible drift given the 0.30 % production stability).
- **None** — anything about `C=0` or mode 14. Not run, not estimated, left as
  explicit placeholders.

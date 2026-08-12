# exp_24 RE-MEASURE — instrument A at exp_26's shipped config

**Verdict: the gap to rank-1 did NOT move, and exp_26's shape-5 win did not
reproduce in instrument A.** `ours/rank-1` reads **1.1165× graded / 1.1111×
pipelined** (per-shape best, geomean) against the previous run's 1.0971× /
1.1189× — both moves inside the ±2% ratio-noise floor, so both are *unchanged*,
not improvements and not regressions. exp_26's projected **≈1.085× graded did
not hold** (measured 1.1165× best, 1.0924× median). Shape 5, the only row whose
codegen changed, came back **flat under the graded protocol and ~3% WORSE under
the pipelined one**, against exp_26's paired −6.56%. That contradiction is the
most consequential thing in this run and section 6 names the one experiment that
settles it.

Data: `ladders.json` (2 252 816 B, instrument A only, `expect_b=0`).
Comparison: `remeasure_delta.json`, `compare_runs.py`.

---

## 1. The binary under test

| | value |
|---|---|
| `gemm_rs_mi300x.so` sha256 | `fb3d670b5ffb68bbfd469692098f11631abcd1dc30995c543fbca4349543ea6b` |
| pre-exp_26 binary (previous ladder) | `79599cce…` |
| assertion | **differs** — recorded in `logs/module_sha_under_test.txt` |
| `dhk_rt.so` | `a97031b24f251da6…` |
| `gemm_rs_mi300x_control.so` | `c84ea3679f1c4dcf…` |

The rebuild needed repairing first: `harness/build.sh` and `tools/m2_report.sh`
arrived with Windows CRLF and the build died mid-script after emitting only the
first module, while still printing a "sha differs" line. That line was a **false
pass** — the module it described had not been rebuilt in that invocation. After
stripping CR, syntax-checking, and deleting the three `.so` targets so "the file
exists" meant something, a from-scratch build returned rc=0 on all three
modules and reproduced `fb3d670b…` bit-for-bit, which is both the freshness
proof and a compiler-determinism check.

## 2. `rgroup` really is 2 on shape 5 — three independent checks

`geometry.json`, from the live shape planner (`dhk_rt.resolve_shape`, the same
planner the launch path uses):

| # | shape | row | tiles | producers | tiles/CTA | rgroup was | rgroup now |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 64×7168×18432 | 1 | 224 | 248 | 1 | 1 | 1 |
| 2 | 512×4096×12288 | 2 | 256 | 272 | 1 | 1 | 1 |
| 3 | 2048×2880×2880 | 3 | 240 | 272 | 1 | 1 | 1 |
| 4 | 4096×4096×4096 | 4 | 256 | 272 | 1 | 1 | 1 |
| 5 | **8192×4096×14336** | 5 | 512 | 272 | **2** | **1** | **2** |
| 6 | 8192×8192×29568 | 6 | 1024 | 256 | 4 | 4 | 4 |

Ladder `1/1/1/1/2/4`; exactly one row moves, and it is shape 5. **PASS.**

Check B, codegen reachability: recompiling the same source with the same
`TK_MODNAME` but `PERSHAPE=0` yields `2846864dbbbf2fba…` (410 232 B) against the
shipped `fb3d670b…` (410 264 B). The objects **differ**, so `PERSHAPE=2` is not
dead code. Check C: `gemm_rs_mi300x.cpp:162` reads `#define
HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE 2`, with `RELEASE_GROUP 4` and
`FULL_ONLY 1`.

The first version of check A **failed on the kernel's behalf** because I
hardcoded the graded table and got rows 3 and 4 wrong (`2048×4096×12288` for
`2048×2880×2880`, and every bias flag inverted); both fell through to generic
config row 0 at 15 and 30 tiles per CTA. `verify_rgroup.py` now imports `SCORED`
from `ladder_mp.py`, so the verification cannot disagree with the measurement.

## 3. The ladder — per-shape best (median), all five arms

**GRADED**, µs:

| # | shape | ours | ours_null | reference | rank1 | harness_floor |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 64×7168×18432 | 169.62 (177.96) | 167.09 (178.42) | 242.05 (254.84) | 196.62 (206.44) | 112.76 (123.49) |
| 2 | 512×4096×12288 | 172.51 (180.80) | 172.55 (180.20) | 257.34 (271.84) | 159.34 (169.16) | 102.51 (111.00) |
| 3 | 2048×2880×2880 | 175.96 (184.23) | 176.37 (185.88) | 230.10 (249.04) | 154.47 (166.53) | 79.22 (87.39) |
| 4 | 4096×4096×4096 | 293.15 (302.64) | 290.35 (301.08) | 332.78 (343.76) | 258.37 (280.45) | 77.65 (85.23) |
| 5 | 8192×4096×14336 | 724.84 (753.80) | 721.87 (746.03) | 657.16 (675.90) | 564.91 (603.82) | 101.94 (109.29) |
| 6 | 8192×8192×29568 | 1828.65 (1887.91) | 1839.64 (1926.45) | 1666.36 (1723.37) | 1462.26 (1525.32) | 149.30 (156.72) |
| | **geomean** | **354.97 (369.69)** | 353.78 (370.49) | 416.54 (436.41) | 317.94 (338.42) | 101.32 (109.76) |

**PIPELINED**, µs:

| # | shape | ours | ours_null | reference | rank1 | harness_floor |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 64×7168×18432 | 63.59 (64.55) | 63.61 (64.76) | 105.41 (110.48) | 76.45 (78.98) | 6.31 (7.33) |
| 2 | 512×4096×12288 | 72.39 (73.16) | 71.96 (72.93) | 107.56 (110.82) | 57.70 (59.54) | 6.54 (7.25) |
| 3 | 2048×2880×2880 | 90.44 (91.91) | 90.48 (92.55) | 122.52 (124.92) | 83.39 (86.59) | 5.42 (5.99) |
| 4 | 4096×4096×4096 | 207.66 (211.09) | 207.00 (209.76) | 224.18 (226.81) | 196.84 (203.37) | 5.24 (5.83) |
| 5 | 8192×4096×14336 | 645.84 (652.31) | 637.98 (644.66) | 542.12 (545.96) | 491.88 (504.51) | 5.43 (6.09) |
| 6 | 8192×8192×29568 | 1644.97 (1697.03) | 1678.49 (1742.91) | 1522.67 (1583.73) | 1370.40 (1418.00) | 4.99 (5.85) |
| | **geomean** | **212.41 (215.95)** | 212.39 (216.52) | 252.16 (258.73) | 191.17 (197.44) | 5.63 (6.36) |

Correctness: 0 gated arms failed either tolerance; max|diff| 9.766e-4 … 1.465e-3
across all gated arms and shapes. Parser refusals: 0. All 48 sample files carry
`rot_mode="shuffle"`, which is a content-based proof that none is a leftover
from the previous binary (see section 8).

## 4. The ratios, with each per-shape number marked against its own floor

Floors are this run's `|ours − ours_null|` per shape and statistic.

**ours / rank-1, best:**

| # | shape | graded | pipelined | floor% | resolved? |
|---:|---|---:|---:|---:|---|
| 1 | 64×7168×18432 | 0.8627 | 0.8318 | 1.52 / 0.04 | resolved — **we are ahead** |
| 2 | 512×4096×12288 | 1.0827 | 1.2544 | 0.02 / 0.59 | resolved |
| 3 | 2048×2880×2880 | 1.1391 | 1.0845 | 0.23 / 0.04 | resolved |
| 4 | 4096×4096×4096 | 1.1346 | 1.0550 | 0.96 / 0.32 | resolved |
| 5 | 8192×4096×14336 | 1.2831 | 1.3130 | 0.41 / 1.23 | resolved |
| 6 | 8192×8192×29568 | 1.2506 | 1.2004 | 0.60 / 2.00 | resolved |
| | **geomean** | **1.1165** | **1.1111** | | |

Every per-shape ratio is far outside its own floor, so all twelve are resolved.
Medians: 1.0924× graded, 1.0937× pipelined. We are ahead of rank-1 on **1 of 6**
shapes (shape 1) under either protocol.

**ours / reference**: 0.8522× graded, 0.8424× pipelined, ahead on 4 of 6 shapes
(behind on 5 and 6). **Do not read the apparent improvement from 0.8863× as
ours** — see section 5; the reference arm degraded in this run for reasons that
have nothing to do with our kernel, which flatters this ratio.

## 5. Delta versus the previous run, and why the graded column is not usable

Pre-registration: shape 5 improves, the other five flat. **Outcome: the reverse
on both counts.**

| # | shape | rgroup | graded best prev → new | Δ% | floor% | pipelined best prev → new | Δ% | floor% |
|---:|---|---:|---|---:|---:|---|---:|---:|
| 1 | 64×7168×18432 | 1 | 149.37 → 169.62 | **+13.56** | 1.52 | 62.88 → 63.59 | +1.13 | 0.36 |
| 2 | 512×4096×12288 | 1 | 154.03 → 172.51 | **+12.00** | 0.44 | 71.17 → 72.39 | +1.70 | 0.59 |
| 3 | 2048×2880×2880 | 1 | 178.12 → 175.96 | −1.22 | 0.48 | 90.95 → 90.44 | −0.56 | 0.43 |
| 4 | 4096×4096×4096 | 1 | 294.84 → 293.15 | −0.58 | 2.15 | 208.10 → 207.66 | −0.21 | 0.32 |
| 5 | 8192×4096×14336 | **1→2** | 717.71 → 724.84 | +0.99 | 2.00 | 627.84 → 645.84 | **+2.87** | 2.25 |
| 6 | 8192×8192×29568 | 4 | 1759.29 → 1828.65 | +3.94 | 4.31 | 1642.60 → 1644.97 | +0.14 | 0.14 |
| | geomean | | 339.29 → 354.97 | +4.62 | | 210.64 → 212.41 | +0.84 | |

Twelve control-shape moves land outside their floors, which by the dispatch's
rule is a blocker. It is an **instrument** blocker, not a kernel one, and the
common-mode table is why:

**graded best, Δ% versus the previous run, per arm:**

| # | shape | ours | ours_null | reference | rank1 | harness_floor | spread |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 64×7168×18432 | +13.56 | +12.25 | +36.01 | +11.48 | **+19.17** | 24.53 |
| 2 | 512×4096×12288 | +12.00 | +11.53 | +25.76 | +6.46 | **+29.26** | 22.80 |
| 3 | 2048×2880×2880 | −1.22 | −0.51 | −1.53 | −1.91 | +1.75 | 3.66 |
| 4 | 4096×4096×4096 | −0.58 | +0.59 | −0.66 | +2.01 | −3.38 | 5.39 |
| 5 | 8192×4096×14336 | +0.99 | +2.59 | −0.89 | −0.09 | +1.31 | 3.48 |
| 6 | 8192×8192×29568 | +3.94 | +0.06 | +0.10 | −0.53 | +0.50 | 4.47 |

On shapes 1 and 2 **every arm inflated, including `harness_floor`, which is a
no-op kernel** (+19.2%, +29.3%). Our shape-1 graded time is ~150 µs of which
~95–113 µs is the protocol's own per-call constant, so a ~19% shift in that
constant is ~+18 µs ≈ +12% of the total — the whole observed move. The same
shapes moved only +1.1% and +1.7% under the pipelined protocol, which does not
pay the constant per call. Shapes 3–6 are quiet on every arm (spreads 3.5–5.4%).

The `harness_floor` graded arm has an **rsd of 194.8 / 153.0 / 274.2 / 266.9 /
4.3 / 60.0%** across the six shapes. The graded per-call constant is not a
constant between runs, so **absolute graded µs is not comparable across runs**;
the within-run ratio is. (This also re-kills the netted table: floor spread
63.7%, `shape_independent=False`.)

Under the pipelined protocol the small shapes tell the same story from the other
side: `reference` inflated **+48.2%** on shape 1 and **+14.3%** on shape 2, and
`harness_floor` +17.4% / +22.1%, while `ours`, `ours_null` and `rank1` all moved
≈+1%. That is what makes the ours/reference "narrowing" an artifact: reference
got slower, we did not get faster.

## 6. Shape 5: exp_26's win did not reproduce, and the two measurements disagree

| statistic | prev → new | Δ | this shape's floor | read |
|---|---|---:|---:|---|
| graded best | 717.71 → 724.84 | +0.99% | 2.00% | under floor — no effect |
| graded median | 739.19 → 753.80 | +1.98% | 2.68% | under floor — no effect |
| pipelined best | 627.84 → 645.84 | **+2.87%** | 2.25% | resolved — **worse** |
| pipelined median | 632.95 → 652.31 | **+3.06%** | 2.60% | resolved — **worse** |

exp_26 measured −6.56% pipelined on this shape in 80 of 80 paired rounds across
both allocation orders against a 0.61% null. Instrument A, comparing across
runs, reads +2.9 to +3.1% — a **disagreement of ~9.5 percentage points,
including its sign.** Shape 5's environment was quiet in this run (`reference`
−0.07%, `rank1` +0.13%, `harness_floor` −1.41%), so unlike shapes 1–2 this move
cannot be waved off as common-mode drift; `ours` (+2.87%) and `ours_null`
(+3.90%) both moved, and they are the same module, so the null cannot separate
them either.

Which measurement to believe: **exp_26's is the stronger design.** It is a
paired same-run A/B of the two binaries in one pool; mine is a comparison of two
runs separated by six hours, and section 5 just demonstrated that this
instrument's cross-run absolute numbers carry environmental drift of up to
+29% on the floor arm. A cross-run delta of +2.9% against a 2.25% floor is a
weak instrument disagreeing with a strong one.

**The decisive experiment, and it is one run:** add the pre-exp_26 build as a
sixth arm (`ours_prev`, exp_26's `ps0.so` already exists) to `ladder_mp.py`'s
pool, so the two binaries are compared *inside* instrument A with the same
interleave, the same pool and the same shuffle that every other arm gets. That
converts the across-run comparison into a within-run one and either reproduces
exp_26's −6.56% under the graded protocol or falsifies it. Until then, exp_26's
landing stands on its own gate ladder and **the shape-5 win should be treated as
unconfirmed under the graded protocol**, which is the protocol that grades us.

## 7. Did exp_26's ≈1.085× projection hold? No

| | projected | measured |
|---|---|---|
| graded geomean ratio vs rank-1 | ≈1.085× | **1.1165×** best, 1.0924× median |
| shape 5 graded ratio | ≈1.20× | **1.2831×** |

The projection was explicitly an arithmetic extrapolation from a pipelined
measurement, and it was right to flag itself: the release saving did not survive
the per-call protocol. The graded *median* (1.0924×) lands nearer the projection
than the *best* (1.1165×), which is consistent with the saving being visible
only in the tail rather than in the fastest call.

## 8. `ladder_mp.py`'s rotation: DEFECTIVE, now fixed

**Verdict: it had exactly the defect exp_26 describes.** The scheme was
`order = arms[shift:] + arms[:shift]` with `shift = rep % len(arms)`, i.e.
`order[i] = arms[(rep + i) mod n]`. Every arm is first exactly once, which
removes an *absolute-position* bias — but the *relative* offset between any two
arms is `(j − k) mod n` in every rep, independent of `shift`. So a full rotation
over all positions is **not** immunity: arm *j* ran the same number of slots
after arm *k* in all five reps, and `ours_null` sat permanently one slot behind
`ours`, which is the worst possible placement for the arm whose whole job is to
price the instrument.

Fix (`ladder_mp.py`): a per-rep random permutation, seeded from the shape index
only — never from rank or wall clock, because all eight ranks must walk an
identical order or the collective arms deadlock against each other. The orders
are recorded in every sample file as `arm_orders` and the mode as `rot_mode`;
`LAD_ROT=cyclic` still reproduces the old scheme for a paired instrument study.

**The null arm confirms it mattered**, and reproduces exp_26's signature:

| shape | graded floor, cyclic | graded floor, shuffled |
|---|---:|---:|
| 1 | 0.34% | 1.52% |
| 2 | 0.44% | 0.02% |
| 3 | 0.48% | 0.23% |
| 4 | 2.15% | 0.96% |
| 5 | 2.00% | 0.41% |
| 6 | **4.31%** | **0.60%** |
| mean | 1.62% | **0.62%** |

Shape 6's graded floor collapsed 4.31% → 0.60% and shape 5's 2.00% → 0.41%,
on behaviour that cannot differ. Pipelined floors were already small and stayed
so (mean 0.66% → 0.70%).

**Residual defect, disclosed:** five reps is too few for a shuffle to guarantee
decorrelation, and `show_orders.py` found accidentally pinned pairs on 2 of the
6 shapes (shape 3 `harness_floor↔reference`, shape 5 `ours_null↔harness_floor`).
Neither pins `ours↔ours_null`, so the reported floors are not directly
compromised, but the proper fix is a balanced design — a Latin square over the
five positions, or more reps — rather than an unconstrained shuffle. Logged for
the next instrument revision.

## 9. Caveats

1. **Instrument B was not re-run** in this dispatch, as instructed. `raw/eval`
   still held the previous run's evaluator output, and the aggregator folded it
   into the new `ladders.json` as `evaluator_crosscheck` where it would have
   read as this run's cross-check. It is quarantined at
   `raw/eval_prev_79599cce_DO_NOT_PARSE` and archived at
   `prev_79599cce/raw_eval`; `ladders.json` was re-aggregated with `expect_b=0`
   and now carries no instrument-B block. **The 1.1349× ours/reference
   evaluator finding stands unchanged and still belongs to the previous
   binary.**
2. **Two variables moved between the runs**, not one: the module *and* the arm
   ordering. The five bit-identical shapes were the intended control for both,
   and they came back contaminated by environmental drift on shapes 1–2, so the
   confound is not fully resolved. `LAD_ROT=cyclic` on this same binary
   separates them in one run if the ordering is ever suspected of carrying a
   real effect rather than just inflating the null.
3. **`run_ladders.sh` exited rc=2 after a clean run.** All six shapes completed
   with rank exit codes `[0]×8` and aggregation wrote `ladders.json` before the
   failure; the error is a bash syntax error at line 454, because I pushed the
   tree twice while the script was still executing and bash reads a script
   incrementally. The script's own header documents this hazard from an earlier
   incident with `tools/gpu_lease.sh`; I reintroduced it. **Never sync the tree
   while a driver is running** — the probe scripts must be pushed before the
   launch, not during. The data is unaffected and the lease released cleanly.
4. **`LAD_KEEP` is not implemented** in `run_ladders.sh` — it never deletes
   previous samples, it overwrites `lad_s<shape>.rank<r>.json` per shape. That
   is safe only when all six shapes succeed; a failed shape silently leaves the
   previous binary's file to be parsed. `finalize.sh` now gates on content
   (`rot_mode == "shuffle"` in all 48 files) rather than trusting the overwrite.
5. One process drives all 8 devices here, which is not the evaluator's
   topology; every comparison against rank-1's published evaluator numbers
   carries that caveat.
6. Clocks pinned at 1900 MHz before timing; the node was clean at lease
   acquisition (0 s wait) and no preemption occurred.

## 10. Schema

`ladders.json`: `protocols.{graded,pipelined}.arms.<arm>.per_shape[i]` with
`{best_us, median_us, mean_us, worst_us, rsd_pct, runs}` plus `geomean_us`,
`geomean_median_us`, `geomean_mean_us`; `ratios.<ours_vs_X__proto__stat>` with
`{per_shape[].ratio, geomean}`; `config.rot_mode`; `correctness`.
`geometry.json`: `rows[]` with `{shape, config_row, tiles, producers,
tiles_per_cta, rgroup_prev, rgroup_now, num_reducer_ctas}`, `moved`, `pass`.
`remeasure_delta.json`: `{blockers[], geomeans}`.

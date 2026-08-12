# exp_25 — per-shape sensitivity readout (paper Q5)

**STATUS: COMPLETE. Zero GPU runs, zero lease time, no new measurement** —
derived entirely from `exp_20_attribution/ablation.json` and
`exp_23_waterfall/{waterfall,stats}.json`, whose sha256s are recorded in
`knob_by_shape.json`.

---

## 1. Verdict up front

**P1 — "order and granularity deltas grow with communication share" is
FALSIFIED, in sign, under all four candidate definitions of communication
share. The deltas are largest where the communication share is *lowest*.** And
the falsification is **uninformative as a sensitivity result**, because in this
shape set the structural mask that decides whether a knob acts at all is
*perfectly rank-anti-correlated* with communication share
(Spearman(K, tiles/NG) = **+1.00**, Spearman(K, D3) = **−1.00** over the four
usable shapes). The two axes cannot be separated by the six graded shapes at any
sample size. **The honest answer is that n is too small AND the design is
confounded, so no sensitivity claim about comm share is supportable from this
data — in either direction.**

**P2 — "the NR curve stays flat everywhere" is FALSIFIED as stated and
CONFIRMED in its correct form:** flat from NR = 32 upward on shapes 2-6 (every
contrast inside that shape's own widened floor), flat only from NR = 48 upward on
shape 1, and a resolved cliff below — NR = 8 is resolved slower on **6 of 6**
shapes by 13.75–51.48%, NR = 16 on 5 of 6. This is the one claim in this
experiment informed by all six shapes.

The two verdicts, with everything that follows them, in one table:

| prediction | verdict | shapes informing it | key numbers |
|---|---|---|---|
| **P1** order delta grows with comm share | **FALSIFIED in sign; then UNRESOLVABLE by confound** | **n = 2** where the knob is active (n = 4 for the mask variant) | s5 comm D3 = 0.525 → +15.11%; s6 comm D3 = 0.392 → +42.11%. Spearman (mask variant, n = 4) = **−0.40 / −0.80 / −1.00 / −0.80** under D1/D2/D3/D4 |
| **P1** granularity delta grows with comm share | **UNRESOLVED — n = 1** | **n = 1** | only shape 6 can express the knob; it has the lowest comm share of the four usable shapes under D2/D3/D4 |
| **P2** NR flat everywhere | **FALSIFIED** | n = 6 | NR = 8: −13.75 to −51.48%, resolved slower on 6/6 |
| **P2′** NR flat on the plateau | **CONFIRMED** | n = 6 | plateau lower edge NR = 32 on shapes 2-6, NR = 48 on shape 1; max \|gain\| on the plateau is inside every shape's widened floor |

---

## 2. The data, in one table

Best/median wall times and the scored contrasts come from exp_23; the stage
attribution and geometry from exp_20. `floor` is the exp_23 **widened**
union-of-identical-pairs floor, which is the one used for every verdict here.

| # | shape | K | tiles | NG | tiles/CTA | full µs | **D1** | **D2** | **D3** | **D4** | floor % | a→b | b→c |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 64×7168×18432 | 18432 | 224 | 248 | 1 | 67.1 | — | — | — | — | 2.82 | −0.60% inert | +1.74% inert |
| 2 | 512×4096×12288 | 12288 | 256 | 272 | 1 | 67.5 | — | — | — | — | 2.09 | −0.29% inert | −0.08% inert |
| 3 | 2048×2880×2880 | 2880 | 240 | 272 | 1 | 87.7 | 0.235 | 0.494 | **0.815** | 0.642 | 1.74 | −0.64% inert | +1.41% inert |
| 4 | 4096×4096×4096 | 4096 | 256 | 272 | 1 | 203.5 | **0.424** | **0.669** | 0.811 | **0.704** | 0.85 | −0.02% inert | +0.05% inert |
| 5 | 8192×4096×14336 | 14336 | 512 | 272 | **2** | 641.9 | 0.280 | 0.464 | 0.525 | 0.455 | 4.97 | **+15.11% RESOLVED** | +0.79% inert |
| 6 | 8192×8192×29568 | 29568 | 1024 | 256 | **4** | 1632.5 | 0.230 | **0.343** | **0.392** | **0.343** | 4.44 | **+42.11% RESOLVED** | **+12.22% RESOLVED** |

Shapes 1 and 2 carry no comm share by construction (§3). "inert" means the knob
is *arithmetically incapable* of changing behaviour on that shape, not that it
was measured and found small — every one of those eight contrasts is also inside
its shape's floor, which is exp_23's instrument check passing, not a result.

Derived arithmetic per shape (all of it reproduced independently by
`build_sensitivity.py --check` against exp_20's own `geometry` block):

| # | BM/BN/BK | NR | waves | k_iters | waves·k_iters | LDS B | at 64 KB cap | tiles/NG | flops per egress byte |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 32/64/128 | 56 | 1 | 18 | 18 | 49152 | no | 0.903 | 21065 |
| 2 | 64/128/64 | 32 | 1 | 24 | 24 | 49152 | no | 0.941 | 14043 |
| 3 | 128/192/32 | 32 | 1 | 12 | 12 | 40960 | no | 0.882 | 3291 |
| 4 | 256/256/32 | 32 | 1 | 16 | 16 | 65536 | **yes** | 0.941 | 4681 |
| 5 | 256/256/32 | 32 | 2 | 56 | 112 | 65536 | **yes** | 1.882 | 16384 |
| 6 | 256/256/32 | 48 | 4 | 116 | 464 | 65536 | **yes** | 4.000 | 33792 |

---

## 3. Communication share: four definitions, and why the choice matters less
## than the confound

exp_20's stage numbers are **single-cut deltas** (`full − arm`). They overlap
wherever the phases overlap and they do not partition `full` —
`stage_sum_over_full` runs from 0.05 (shape 1, all noise) to 1.02 (shape 5). So
`egress/full` is not a share of anything; it is a ratio of two quantities where
the numerator is a difference of overlapping durations. Four definitions were
therefore written into `plan.md` before any correlation was computed, and all
four are reported:

| id | formula | shape 3 | shape 4 | shape 5 | shape 6 |
|---|---|---:|---:|---:|---:|
| **D1** | `egress / full` | 0.235 | **0.424** | 0.280 | 0.230 |
| **D2** | `(egress + sync + release) / full` | 0.494 | **0.669** | 0.464 | 0.343 |
| **D3** *(primary)* | `(full − GEMM) / full` | **0.815** | 0.811 | 0.525 | 0.392 |
| **D4** | `(egress + sync + release) / Σ(5 stages)` | 0.642 | **0.704** | 0.455 | 0.343 |

D3 is the nominated primary because its numerator and denominator come from the
same single cut, so it cannot be inflated by double-counting overlapping pools.
It is also the only one with a closed form in the shape's own arithmetic:
`flops / egress byte = 2MNK / ((7/8)·MN·2) = **8K/7**`, i.e. **the
compute-per-communicated-byte axis of the graded set is a function of K alone**,
and D3 is monotone decreasing in K across all four usable shapes
(K = 2880 → 4096 → 14336 → 29568 gives 0.815 → 0.811 → 0.525 → 0.392).

### The correlation under each definition

Mask variant (all four usable shapes, inert shapes included at their measured
near-zero deltas), a→b rung:

| definition | Spearman ρ | Pearson r | shape order, most-comm-first |
|---|---:|---:|---|
| D1 `egress/full` | **−0.40** | −0.51 | 4, 5, 3, 6 |
| D2 | **−0.80** | −0.83 | 4, 3, 5, 6 |
| D3 *(primary)* | **−1.00** | −0.95 | 3, 4, 5, 6 |
| D4 | **−0.80** | −0.94 | 4, 3, 5, 6 |

b→c rung, same variant: ρ = −1.00 / −0.80 / −0.40 / −0.80 under D1/D2/D3/D4.

**Is the conclusion definition-dependent? Not in sign — in strength.** Every
definition gives a negative rank correlation for both rungs, so no definition
rescues P1. What *does* move is the detail: under D1 the two knob-active shapes
rank 2nd and 4th of four by comm share, while under D2/D3/D4 they are the two
**lowest**-comm shapes in the set. And no coefficient above is inferential: at
n = 4 the smallest attainable two-sided p is 2/4! = **0.083**, so even the
perfect ρ = −1.00 under D3 cannot reach p < 0.05. **No line is fitted and no
p-value is quoted as significant.**

---

## 4. Shapes 1 and 2: unresolvable, excluded, n = 4

Excluded from every comm share and every correlation, on exp_20's own reading:

- shape 1 has **three negative stage deltas** (sync −1.0, release −0.5, reduce
  −0.3 µs) and all five are at or inside its 2.34 µs floor; shape 2 has one
  (reduce −0.8 µs);
- the five cuts price **4.9%** of shape 1's wall and **9.5%** of shape 2's
  (`stage_sum_over_full` = 0.049 / 0.095), i.e. 90-95% of those calls is priced
  by no cut at all;
- M7 labels shape 1 `bound = HOST` at 10.39× SOL — 62.34 µs of host issue inside
  a 66.41 µs wall — and exp_20 shows shape 2 is host-bound too against its
  median (62.43 of 67.80 µs), its `device` label coming only from one 114 µs
  rotation inflating a mean.

A ratio built on a negative delta inside a noise floor is noise with a decimal
point. **n = 4 enters the correlation. That is far too few for a correlation
coefficient to mean much, which is why this readout gives ordering statements
and refuses fits.** The exclusion is scoped to attribution: shapes 1 and 2 keep
their rows in §6's NR adjudication, where the quantity is a same-shape paired
wall-time contrast against a measured floor rather than a stage delta — and
shape 1's NR contrasts are the largest in the whole set.

---

## 5. Structural activity vs sensitivity — the crux, stated cleanly

Two questions, and merging them is the single easiest way to get this experiment
backwards.

**(i) Is the knob active on this shape at all?** Pure arithmetic from the shape
plan, no measurement involved:

- `WGM4`: `WGM = (tiles <= NG) ? 4 : num_pid_m`, so a→b changes behaviour only
  where `tiles > NG`. tiles/NG = 0.903 / 0.941 / 0.882 / 0.941 / **1.882** /
  **4.000** → **shapes 5 and 6 only.**
- grouped release: `rgroup = tiles_per_cta >= 4 ? 4 : 1`, and `tiles_per_cta` =
  1/1/1/1/2/**4** → **shape 6 only.**

Both flags were re-derived here from `m, n, bm, bn, nr` and match exp_23's
`ab_rung_active` / `bc_rung_active` for all six shapes, and exp_20's
`effective_release_group` (1/1/1/1/1/4) independently.

**A zero delta on shapes 1-4 is arithmetic, not evidence about sensitivity.** Any
correlation that includes those rows is measuring the structural mask. That is
why the table in §3 is labelled "mask variant" everywhere it appears.

**(ii) Given that the knob is active, does its value scale with comm share?**
This is the only question P1 can be about, and the sample is:

| rung | active *and* usable shapes | n | comm share (D3) → delta |
|---|---|---:|---|
| a→b task order | 5, 6 | **2** | 0.525 → +15.11% ; 0.392 → +42.11% |
| b→c granularity | 6 | **1** | 0.392 → +12.22% |

**n = 2 and n = 1. That is not a trend and it is not presented as one.** Two
points define a slope by construction; one point defines nothing.
`knob_by_shape.json` carries `correlations.active_only.*.coefficient = null` with
the reason inline, rather than a number.

What the two active points *do* show is that the delta tracks a **task-graph**
property rather than a communication-intensity one: from shape 5 to shape 6,
tiles/NG goes 1.88 → 4.00 and `tiles_per_cta` 2 → 4 while comm share *falls*
(D3 0.525 → 0.392, D1 0.280 → 0.230), and the delta grows 2.8×. More producer
rounds to mis-spread, not more communication to hide.

### The confound, which is the actual finding

Across shapes 3-6 the graded set orders shapes **identically** by K and by
tiles/NG (Spearman = **+1.00**), and comm share is monotone in K
(Spearman(K, D3) = **−1.00**). So "how much communication this shape does" and
"whether these knobs can act on this shape" are perfectly anti-aligned in the
only shape set we have. **P1 is not merely underpowered at n = 4; it is not
identifiable from these six shapes at any n.** Getting an answer would need
shapes that break the collinearity — e.g. a small-K shape with `tiles_per_cta ≥ 4`
(large M·N, shallow K), which is not in the graded set.

### And the regression is ill-posed in a second, independent way

Every comm share here is measured at rung **c**, the top of the ladder, while
each rung delta is taken from the configuration *below* it. The knob changes the
very quantity it would be regressed against:

| rung / shape | delta µs | comm pool at c (D2 numerator) | delta ÷ comm pool | target pool at c | delta ÷ target pool |
|---|---:|---:|---:|---:|---:|
| a→b, shape 5 | 99.2 | 298.0 | 0.33× | egress 179.7 | 0.55× |
| a→b, shape 6 | **775.1** | 559.6 | **1.39×** | egress 376.0 | **2.06×** |
| b→c, shape 6 | **200.7** | 559.6 | 0.36× | release **15.0** (inside floor) | **13.4×** |

Shape 6's order rung is worth **more than the entire communication pool that
survives at the winner**, and its granularity rung is worth 13× the release pool
that survives — and ~1.5× the 134.9 µs release pool exp_08 measured *before* E3
at the same task order. A knob whose win exceeds the pool it targets did not
merely delete its own instructions; it unblocked something else. That is the
same superadditivity exp_23 found from the other direction: the granularity rung
was pre-registered at ≈4% from exp_05's **graded** protocol on the **aug10 task
order** and came in at **+12.22%** pipelined **on top of WGM**, because
coarsening the signal is worth more once the order change has removed the egress
serialization that was hiding it. **Consequence for the paper: the waterfall's
rungs are not additive contributions and the ladder's order is part of the
result. A per-shape "sensitivity" number for a rung is only defined relative to
the rung below it.**

---

## 6. P2 — the NR curve, and its correct form

Gains vs each shape's **shipped** NR (56/32/32/32/32/48), median, from the rung-c
binary — the NR sweep needs no rebuild because `num_reducer_ctas` is a host
scalar. `floor` is that shape's widened floor; a star marks the shipped point.

| # | shipped NR | NR=8 | NR=16 | NR=32 | NR=48 | floor % | flat from |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **56** | −51.48 slower | −28.11 slower | −7.96 slower | +0.10 | 2.82 | **48** |
| 2 | **32** | −25.00 slower | −8.94 slower | ★ +0.69 | +0.35 | 2.09 | 32 |
| 3 | **32** | −27.26 slower | −11.04 slower | ★ −0.04 | +0.48 | 1.74 | 32 |
| 4 | **32** | −24.55 slower | −9.80 slower | ★ −0.12 | +0.34 | 0.85 | 32 |
| 5 | **32** | −13.75 slower | −5.01 *(beyond floor, not rank-separated → unresolved)* | ★ −2.37 | +0.44 | 4.97 | 32 |
| 6 | **48** | −18.80 slower | −8.87 slower | −3.53 *(inside floor — see below)* | ★ +3.92 | 4.44 | 32 |

- **Flat on the plateau: CONFIRMED.** Every contrast at NR ≥ 32 is inside its
  shape's own widened floor on shapes 2-6; on shape 1 the flat region starts at
  48. Adding CTAs to the non-GEMM side above the plateau buys nothing measurable
  on any shape, which is the claim Q2 needs.
- **Flat everywhere: FALSIFIED.** NR = 8 is resolved slower on all six shapes,
  NR = 16 on five (shape 5's −5.01% exceeds its 4.97% floor but is not
  rank-separated, so it is unresolved rather than resolved).
- **The plateau's left edge is not a constant either.** It is 32 on shapes 2-6
  and 48 on shape 1 — the smallest, host-bound shape needs the *most* reducers —
  which is exactly why the shipped table is per-shape 56/32/32/32/32/48 rather
  than uniform. "Enough reducers, not more" is the right statement; "32 is
  enough" is not.
- **Marking the shipped points is not cosmetic.** NR = 32 is the shipped config
  for shapes 2-5 and NR = 48 for shape 6, so the ladder did not regress at its
  own best; a figure that omits the stars implies it did.
- **Why this survives as the Q2 answer, for a sharper reason than flatness.** The
  reducer CTAs are not a communication pool — the payload leaves in the
  producer's epilogue — they are the **owner-side reduce**, which must run on the
  owner. Below ~32 you are starving a computation (NR = 8 makes shape 6's reduce
  take 16 rounds instead of 3), not under-provisioning communication. Reported so
  nobody attributes the cliff to communication: the NR = 8 penalty correlates
  +0.60 with the reduce share, +0.80 with the not-GEMM share and **0.00 with
  egress/full** — and at n = 4 those three are indistinguishable, so **none of
  them is a claim.** The mechanism is the argument; the correlation is not.

### One internal inconsistency in the input, flagged and resolved

On shape 6, `nr32 vs c` is labelled **"RESOLVED slower"** by exp_23's
disjointness rule while carrying `beyond_floor = false` — its −3.53% sits inside
that shape's 4.44% widened floor. This readout treats the **floor as
authoritative** and reads it as unresolved, and
`knob_by_shape.json` flags it per point as
`verdict_conflicts_with_widened_floor`. It is the only such conflict in the 24
NR contrasts. It matters because it is exactly the failure mode exp_23 itself
documented on this shape: two *identically configured* arms (`c` vs `nr48`)
separated by +3.74/+4.44/+4.10/+2.53% — positive in all four draws and in **both
construction orders** — which the same rule certified as "RESOLVED faster". The
practical consequence is that the reported geomean ordering NR = 48 (0.992×) <
NR = 32 (1.023×) is **not resolvable per shape anywhere**; it is inside the floor
on every row.

---

## 7. What in this data pushes back on the paper's story

Stated plainly, because these are the parts a reviewer will find:

1. **The Q5 pre-registration is wrong for GEMM-RS, and the repair changes its
   meaning.** "Deltas grow with comm share" is false in sign under every
   definition. The defensible replacement is "deltas grow with the number of
   producer rounds a CTA must issue" — a **task-graph** property, not a
   communication-intensity one. This *strengthens* the central thesis
   (scheduling decides overlap) while removing a claim the paper currently makes
   about *when* the knobs pay.
2. **The two operator instances disagree on the direction of the sensitivity, and
   the disagreement is structural.** PAPER.md's Q5 registers "small M shifts
   value toward signal coarsening"; on GEMM-RS the coarsening knob is
   *arithmetically impossible* below `tiles_per_cta = 4` and therefore acts only
   on the **largest** shape. The paper must not present these as one trend.
3. **The knobs cannot reach the most communication-bound shape in the set.**
   Shape 4 has the highest comm share under D1, D2 and D4 (egress alone is
   **42.4%** of its 203.5 µs) and **no rung touches it** — one tile per CTA makes
   both knobs identities. Shapes 3 and 4 are likewise irrecoverable for release
   grouping at any setting. So "these knobs are how you fix a comm-bound
   GEMM-RS" is not supported; they fix a *multi-round* GEMM-RS.
4. **Communication is not the bottleneck at the winner, which supports Q3 and
   deflates Q1's framing.** At rung c the not-GEMM share is 0.392 on shape 6 and
   0.525 on shape 5 — the two shapes carrying the whole remaining graded gap —
   with the mainloop at 60.8% and 47.6%, and only 421.2 of shape 6's 992.2 µs
   GEMM pool being MFMA occupancy. The frontier has moved back into single-GPU
   GEMM quality, exactly as Q3 claims; a waterfall of communication-scheduling
   knobs is a history of how we got here, not a map of what is left.
5. **`egress/full` — the definition a reader would reach for — is the weakest
   one.** It gives the flattest correlation (ρ = −0.40) and it is the only
   definition under which the knob-active shapes are not the two lowest-comm
   shapes. Any figure quoting a single comm share must say which one it is.
6. **Two of six shapes cannot be attributed at all.** A third of the graded set
   is host-bound in this harness, so the paper's per-shape sensitivity panel
   plots four points, and it should say so on the panel rather than in a
   footnote.

---

## 8. `knob_by_shape.json` — schema, stated verbatim

Top level: `experiment`, `what`, `generated_utc`, `gpu_runs` (= 0),
`derivation`, `inputs`, `protocol`, `comm_share_definitions`, `caveats[]`,
`shapes[]`, `correlations`.

| key | meaning |
|---|---|
| `inputs.{ablation,waterfall,stats}` | `{path, sha256}` of each input artifact — the readout is void if a hash moves |
| `protocol` | pipelined, one process driving 8 devices (**not** the evaluator's topology); exp_20 at 40 iters/cell; exp_23 at 4 allocation draws × 8-arm rotation, best+median only |
| `comm_share_definitions` | the four formulas, `why_four`, `primary` (= `D3_full_minus_gemm_over_full`), `why_primary` |

Each `shapes[]` record:

| key | meaning |
|---|---|
| `shape_index`, `shape`, `m`, `n`, `k`, `has_bias` | shape identity, 1-based graded index |
| `geometry` | `bm/bn/bk`, `num_reducer_ctas`, `num_gemm_ctas` (= 304 − NR), `grid_ctas`, `tiles`, `tiles_per_cta` (= ⌈tiles/NG⌉), `producer_waves`, `last_wave_fill`, `k_local` (= K/8), `k_iters`, `waves_x_k_iters`, `lds_double_buffer_bytes` (= 2·(BM+BN)·BK·2), `at_64kb_lds_cap`, `effective_release_group` |
| `derived_arithmetic` | `flops` (= 2MNK), `egress_bytes_mb` (= 7/8·MN·2), `flops_per_egress_byte` and its closed form `8K/7`, `tiles_over_ng` |
| `attribution` | `full_us`, `stages_us{gemm,egress,sync,reduce,release}`, `stage_share_of_full`, `stage_clears_exp20_floor`, `stage_sum_over_full` (**not 1.0 — the cuts overlap**), `x_sol`, `m7_bound`, `m7_host_issue_us_per_op` |
| `comm_share` | `usable_for_correlation`, `exclusion_reason` (null when usable), the four definitions `D1…D4` (**null on shapes 1-2**), `stage_sum_over_full` |
| `floors` | `exp20_allocation_floor_{pct,us}`, `exp23_widened_floor_{pct,us}`, `exp23_single_pair_floor_pct`, `exp23_null_pairs`, `which_is_used` |
| `arm_medians_us`, `arm_bests_us` | rungs `a`/`b`/`c`/`null` wall times at this shape |
| `rungs.{a_to_b,b_to_c}` | `mechanism`, **`structurally_active`**, `gain_median_pct`, `gain_best_pct`, `gain_median_per_draw_range_pct`, `delta_us_from_pooled_{medians,bests}`, `floor_pct`, `floor_us`, `beyond_floor`, `verdict`, `resolved`, `note_delta_us`, `vs_pools_at_rung_c` |
| `rungs.*.vs_pools_at_rung_c` | `comm_pool_us_D2_numerator`, `target_pool` (+`_us`, +`_clears_exp20_floor`), `delta_over_comm_pool`, `delta_over_target_pool`, `delta_over_full_at_rung_c` — a ratio > 1 means the rung's win exceeds the pool that survives at the winner |
| `nr_curve` | `shipped_nr`, `reference_arm`, `points[]`, `plateau_points_nr_ge_32`, `flat_from_nr32_upward`, `plateau_lower_edge_nr`, `plateau_max_abs_gain_pct`, `points_within_widened_floor`, `points_resolved_slower`, `cliff_points_nr_lt_32`, `cliff_all_resolved_slower`, `cliff_worst_gain_pct` |
| `nr_curve.points[]` | `nr`, `num_gemm_ctas`, `is_shipped_config`, `median_us`, `best_us`, `gain_{median,best}_pct_vs_shipped`, `within_widened_floor`, `beyond_floor`, `verdict`, `is_null_pair`, `contrast_key`, **`verdict_conflicts_with_widened_floor`** |
| `knob_structurally_active` | `order_a_to_b`, `granularity_b_to_c`, `nr_reduction_below_plateau` — the answer to question (i) of §5, arithmetic only |

`correlations`:

| key | meaning |
|---|---|
| `n_shapes_entering` (= 4), `shapes_entering`, `shapes_excluded` | the correlation set and why the other two are out |
| `min_attainable_two_sided_p_at_this_n` | 2/4! = 0.0833 — the reason no p-value is quoted |
| `why_no_fit` | stated in-file |
| `by_definition.<D>.{a_to_b,b_to_c}_mask_variant` | `spearman_rho`, `pearson_r`, `n`, `points[]` (each with `structurally_active` and `resolved`), and an `interpretation` string saying this measures the mask |
| `by_definition.<D>.shape_order_most_comm_first` | ordering used for every ranking statement |
| `by_definition.<D>.order_active_shapes_are_the_two_lowest_comm` | true under D2/D3/D4, **false under D1** |
| `by_definition.<D>.comm_share_rank_of_order_active_shapes` | 1 = most comm; s5/s6 rank 2/4 under D1 and 3/4 under D2-D4 |
| `active_only.{a_to_b,b_to_c}` | `n_active_and_usable` (2 and 1), `coefficient: null`, `why_none`, and the raw points |
| `nr_cliff_axis` | NR = 8 penalty vs reduce share (+0.60), vs D3 (+0.80), vs D1 (0.00), n = 4, with the interpretation that none is a claim |
| `collinearity` | `spearman_k_vs_tiles_over_ng` (+1.00), `spearman_k_vs_D3` (−1.00), the raw K and tiles/NG values, and the identifiability argument |

`sensitivity_points.csv` is the same data flattened to one row per shape for a
plotter: shape identity, geometry, `flops_per_egress_byte`, the four comm shares,
the widened floor, both rungs' activity/gain/delta/verdict, the four NR gains and
`nr_plateau_flat`.

---

## 9. Method, and what would invalidate this

- **No GPU. No lease. No new measurement.** Two artifacts read, hashed, and
  re-derived from. Every number in §1-§7 comes out of
  `build_sensitivity.py`; none is typed by hand.
- **Independent re-derivation gate.** `python3 build_sensitivity.py --check`
  recomputes `tiles`, `tiles_per_cta`, `k_iters`, `num_gemm_ctas`,
  `waves × k_iters`, the effective release group and **both structural-activity
  flags** from `m, n, k, bm, bn, bk, nr` alone and asserts they reproduce
  exp_20's `geometry` block and exp_23's `ab_rung_active` / `bc_rung_active`.
  **PASSED for all six shapes**, so the structural mask in §5 is not taken on
  trust from either input.
- **Floors are the widened union-of-identical-pairs floors measured in the run
  that produced the ratios** (2.82 / 2.09 / 1.74 / 0.85 / 4.97 / 4.44 %), never
  the published table — shape 5's widened floor is 2.3× its published 2.17% and
  shape 4's 0.85% is *better* than its published 2.41%, so the published set is
  not uniformly conservative in either direction.
- **No means anywhere**, per the standing rule that means are unusable on this
  node; best and median only.
- Every µs here is pipelined with one process driving eight devices, which is not
  the evaluator's topology, so none of it is directly comparable to graded
  per-call numbers or to rank-1's evaluator figures.

**What would invalidate it:** a changed sha256 on either input; a `--check`
failure (the structural mask is then wrong and §5 collapses); a re-measured floor
that swallows shape 5's a→b or shape 6's two rungs; or a new shape that breaks
the K ↔ tiles/NG collinearity, which would make P1 identifiable and would
require re-running this analysis rather than amending it.

## 10. Files

| file | contents |
|---|---|
| `plan.md` | pre-registrations restated, the four comm-share definitions, the exclusion rule and the structural/sensitivity separation — all fixed before any correlation was computed |
| `build_sensitivity.py` | the derivation + the `--check` gate |
| `knob_by_shape.json` | the plot-ready record; schema in §8 |
| `sensitivity_points.csv` | one row per shape, for plotters |
| `plot_sensitivity.py` | Panel A (delta vs comm share, 2×2 over the four definitions, hollow = inert, bars = widened floor) and Panel B (NR curve with shipped points starred and floor bands). **Not executed here — the authoring host has no matplotlib**; the data is complete without it |
| `build_run.log` | the build + check transcript |

Ledger note, for whoever owns them: `aug11/PLOTS.md` needs an exp_25 row
(Q5 → `knob_by_shape.json` → DONE) and `STATUS.md`/`LESSONS.md` have no exp_25
entry yet. This experiment does not own those files and did not touch them.

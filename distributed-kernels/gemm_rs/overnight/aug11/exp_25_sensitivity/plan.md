# exp_25 — per-shape sensitivity readout (paper Q5)

**No GPU runs. No lease taken. No new measurement.** Both inputs already exist
on disk and this experiment is pure analysis over them:

| input | what it contributes |
|---|---|
| `aug11/exp_20_attribution/ablation.json` | per-shape stage attribution at the current best, per-shape geometry, per-shape allocation-noise floor |
| `aug11/exp_23_waterfall/waterfall.json` + `stats.json` | per-shape rung ladder (a → b → c), the NR sweep, the widened null floors, and the resolved/unresolved verdicts |

Everything else (`LESSONS.md`, `STATUS.md`, `exp_20/result.md`,
`exp_23/result.md`, `exp_23/plan.md`) is read for the measurement rules that
constrain what may be claimed.

## Honesty note on the ordering of this file

Both input experiments completed before exp_25 was dispatched, so **this
`plan.md` is not a blind pre-registration** and is not presented as one. The
predictions in §1 are *quoted from the charter and from `exp_23/plan.md`* — they
were registered before the data existed, by other experiments — and this file's
job is to fix the *adjudication rules* (§2–§4) before the numbers are computed,
so the analysis cannot be steered. In particular the shape-exclusion rule and
the four comm-share definitions were written down here, in this file, before any
correlation was evaluated, and **all four definitions are reported regardless of
which one flatters the prediction**.

---

## 1. The pre-registered predictions this experiment must adjudicate

Both come from the aug11 charter's exp_25 item, verbatim:

- **P1 — "order/granularity deltas grow with comm share."**
- **P2 — "the NR curve stays flat everywhere."**

Two supporting registrations from `exp_23/plan.md`, already scored there, are
carried in because P1 cannot be read without them:

- **S1 (structural)** — `WGM4` changes behaviour only where `tiles > NG`, so the
  a→b rung can act only on shapes 5 and 6; grouped release changes behaviour
  only where `tiles_per_cta >= RELEASE_GROUP = 4`, so the b→c rung can act only
  on shape 6. Shapes 1–4 are predicted flat *arithmetically*.
- **S2** — shape 6 b→c was pre-registered at ≈ −4% and measured at +12.22%.

Each of P1 and P2 is scored **CONFIRMED / FALSIFIED / UNRESOLVED**, with the
numbers, the floor beside every delta, and an explicit count of how many shapes
inform the claim.

## 2. "Communication share" — four candidate definitions, all reported

exp_20's stage deltas are **single-cut** (`full − arm`): they overlap wherever
the phases overlap and they do **not** partition `full`
(`stage_sum_over_full` runs 0.05 → 1.02 across the six shapes). So there is no
canonical comm share, and picking one silently would be the main way to fake
this result. Four definitions are computed for every usable shape:

| id | formula | what it assumes |
|---|---|---|
| **D1** | `egress / full` | only the peer/xGMI cut counts as communication |
| **D2** | `(egress + sync + release) / full` | communication = payload egress + cross-rank sync + publication release; the owner-side reduce is compute |
| **D3** | `(full − GEMM) / full` | everything the mainloop cut does not price is "not-GEMM", an upper bound on comm |
| **D4** | `(egress + sync + release) / Σ(five stages)` | D2 renormalised by the priced pools instead of by `full`, so the overlap/underpricing in the denominator cancels |

**The correlation is reported under all four.** D3 is nominated as the *primary*
axis before the numbers are looked at, for one reason: it is the only definition
whose denominator and numerator are both taken from the same single cut, so it
cannot be inflated by double-counting overlapping pools, and it is monotone in
the shape's arithmetic (`flops / egress byte = 8K/7`). If the verdict on P1
depends on the definition, **that dependence is the finding** and is stated as
such rather than resolved by choosing.

## 3. Shape exclusions, fixed before computing anything

**Shapes 1 and 2 are excluded from every comm-share number and every
correlation.** Grounds, all from exp_20:

- Three of shape 1's five stage deltas are **negative** and every one of the
  five is at or inside its 2.34 µs floor; shape 2 has one negative delta.
- The five cuts price only **4.9%** (shape 1) and **9.5%** (shape 2) of wall
  time (`stage_sum_over_full`), i.e. ~91–95% of those calls is priced by no cut.
- M7 labels shape 1 `bound = HOST` (62.34 µs of issue inside a 66.41 µs wall) at
  10.39× SOL, and exp_20 shows shape 2 is host-bound too against its median.

A ratio built on a negative delta inside a noise floor is noise with a decimal
point. Those two shapes are reported as **unresolvable, excluded**, and the
surviving **n = 4** is stated next to every correlation.

**The exclusion applies to attribution only.** Shapes 1 and 2 keep their rows in
the NR-curve adjudication (P2), because there the quantity being read is a
same-shape paired wall-time contrast against a measured floor, not a stage
delta — and shape 1's NR contrasts are large and resolved.

## 4. Structural activity is separated from sensitivity — the crux

Two different questions, never merged:

1. **Is the knob active on this shape at all?** Decided by arithmetic from the
   shape plan (`tiles > NG`; `tiles_per_cta >= 4`), independent of any
   measurement. A zero delta on a shape where the answer is "no" is
   **arithmetic, not evidence about sensitivity.**
2. **Given that it is active, does its value scale with comm share?** This is
   the only question P1 can be about, and it has **n = 2** for the order rung
   (shapes 5 and 6) and **n = 1** for the granularity rung (shape 6).

Consequences adopted in advance:

- Any correlation computed over all four usable shapes is labelled a
  **mask-variant** correlation and is explicitly described as measuring the
  structural mask, not a sensitivity trend.
- With n = 4 (mask variant) the smallest attainable two-sided p for a perfect
  rank reversal is 2/4! = 0.083, so **no p-value is quoted as significant** and
  a fitted line is not drawn. Monotonicity/ordering statements only.
- With n = 2 and n = 1 for the real question, **no coefficient is computed at
  all.** Two points define a slope by construction; one point defines nothing.

## 5. Floors

Every delta is printed beside its shape's floor, and two floors are carried
because they disagree:

| # | exp_20 allocation floor | exp_23 widened (union of all identical pairs) |
|---|---|---|
| 1 | 3.48% / 2.34 µs | **2.82%** |
| 2 | 0.93% / 0.63 µs | **2.09%** |
| 3 | 1.61% / 1.41 µs | **1.74%** |
| 4 | 2.41% / 4.90 µs | **0.85%** |
| 5 | 2.17% / 13.93 µs | **4.97%** |
| 6 | 4.28% / 69.87 µs | **4.44%** |

The exp_23 set is used for every rung and NR verdict, because it is the floor
measured **in the run that produced those ratios**, from the union of all
identically-configured pairs — the discipline `LESSONS.md` adopted after two
identical shape-6 arms separated by up to 4.44% consistently in both
construction orders and were certified "RESOLVED faster" by the disjointness
rule. Note shape 5's widened floor is 2.3× its published 2.17% and shape 4's is
*better* than its published 2.41%: the published table is not uniformly
conservative in either direction.

## 6. Deliverables

| file | contents |
|---|---|
| `build_sensitivity.py` | derives every number from the two input JSONs; no value is typed by hand |
| `knob_by_shape.json` | the plot-ready record; schema restated verbatim in `sensitivity.md` |
| `sensitivity_points.csv` | the scatter data (one row per shape) for a plotter that does not want to parse the JSON |
| `sensitivity.md` | the readout: P1/P2 verdicts, the correlation under each definition, the structural-vs-sensitivity separation, and how few shapes inform each claim |
| `plot_sensitivity.py` | scatter of rung delta vs comm share under all four definitions + the NR curve; matplotlib |

Validation command (no GPU):

```bash
python3 build_sensitivity.py --check
```

which re-derives the JSON, re-hashes both inputs, and asserts the derived
geometry (`tiles`, `NG`, `tiles_per_cta`, `waves × k_iters`, structural activity
flags) reproduces exp_20's `geometry` block and exp_23's `ab_rung_active` /
`bc_rung_active` flags independently. A mismatch there means one of the two
inputs has moved and the readout is void.

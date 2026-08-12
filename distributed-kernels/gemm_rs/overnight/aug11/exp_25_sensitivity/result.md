# exp_25 result — per-shape sensitivity readout (paper Q5)

**STATUS: COMPLETE.** Zero GPU runs, zero lease time, no new measurement. The
full readout — verdicts, the correlation under each communication-share
definition, the structural-vs-sensitivity separation, the schema, and the method
gate — is **`sensitivity.md`**. This file is the one-screen verdict so the
folder convention (`plan.md` + `result.md`) is honoured without duplicating
numbers.

## Verdicts

| pre-registered prediction | verdict | shapes informing it |
|---|---|---:|
| **P1** order and granularity deltas grow with communication share | **FALSIFIED in sign under all four comm-share definitions**, then **UNRESOLVABLE**: the structural mask is perfectly rank-anti-correlated with comm share in this shape set | **2** for order, **1** for granularity (4 for the mask variant) |
| **P2** the NR curve stays flat everywhere | **FALSIFIED as stated; CONFIRMED on the plateau** — flat from NR=32 upward on shapes 2-6 and from 48 on shape 1, resolved cliff below (NR=8 slower on 6/6 by 13.75-51.48%) | **6** |

Confidence: **high** on P2 (six shapes, resolved contrasts, floors measured in
the same run). **The P1 verdict is a statement about what this data cannot
support, and that is its value** — n = 2 active shapes for the order rung and
n = 1 for the granularity rung, on a shape set where comm share (a function of
K) and knob activity (a function of tiles/NG) are rank-identical in opposition.
No line is fitted; no p-value is quoted as significant (at n = 4 the smallest
attainable two-sided p is 0.083).

## The three things a reader should take away

1. The rung deltas track a **task-graph** property — producer rounds per CTA,
   tiles/NG = 1.88 → 4.00 — not communication intensity, which moves the other
   way (D3 comm share 0.525 → 0.392 between the same two shapes).
2. **The knobs cannot reach the most communication-bound shape in the set.**
   Shape 4 is 42.4% egress and no rung touches it: at one tile per CTA both
   knobs are identities.
3. Shape 6's order rung is worth **1.39× the entire communication pool that
   survives at the winner** and its granularity rung **13× the release pool**, so
   a rung delta cannot be regressed against a share measured after the rung was
   taken. The rungs compose superadditively and the ladder's order is part of the
   result.

## Artifacts

`sensitivity.md` (the readout) · `knob_by_shape.json` (plot-ready, schema in
`sensitivity.md` §8) · `sensitivity_points.csv` · `build_sensitivity.py`
(+ `--check` gate, PASSED for all six shapes) · `plot_sensitivity.py` (not
executed here — no matplotlib on the authoring host) · `build_run.log`.

No kernel source, harness, tool, ledger or other experiment directory was
touched. `aug11/PLOTS.md`, `STATUS.md` and `LESSONS.md` still have no exp_25
row; this experiment does not own them.

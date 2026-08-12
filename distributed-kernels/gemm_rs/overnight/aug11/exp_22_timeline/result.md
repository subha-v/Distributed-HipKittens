# exp_22 result — IN PROGRESS (Phase 1 complete, Phase 2 gated)

**Verdict: none yet.** No GPU job has been run and no kernel file has been
touched. This file exists so the morning read is not blank; it will be
rewritten with numbers when Phase 2 lands.

## Phase 1 — done, CPU only

| deliverable | state |
|---|---|
| `plan.md` | done — figure, arms, pre-registered expectations, publication gates |
| `design.md` | done — phase enum, ring sizing, traffic model, tick calibration, rejected alternatives, **the patch plan with file:line insertion points** (§9) |
| `parity_gate.sh` + `parity_check.py` | done — gate 1, three TUs, byte-identical OFF comparison against the post-exp_14 M2 table |
| `build_trace.sh` | done — flag-ON module into `exp_22_timeline/build/`, never `harness/build/` |
| `trace_run.py` | done — arm (b) driver, in-situ tick regression, both correctness tolerances |
| `b0_ref.py`, `b0_rank0_wrap.sh`, `b0_capture.sh`, `b0_kernel_map.py` | done — arm (a) capture and the trace-derived kernel map |
| `bin_timeline.py` | done — owns the whole traffic model for both arms; 10 µs bins; ±10% integral gate |
| `plot_timeline.py` | done — the 3×N grid |
| `c_rank1_probe.sh` | done — arm (c) feasibility probe, read-only, no GPU |

## Why the phase enum carries a credit-wait pair — preserved reasoning

The charter's nine-tuple has no producer credit-wait. Adding
`CREDIT_WAIT_BEG`/`CREDIT_WAIT_END` (ids 3 and 4) is not decoration; without
them the producer's **reuse-credit stall lands inside the emit interval**, and
because the xGMI strip is computed as (phase bytes ÷ phase duration), a longer
emit interval carrying the same bytes deflates the strip by exactly the length
of that stall. The figure would then show our egress running slower than it
does, and — worse — it would point the reader at the wrong resource, because
the missing time would look like fabric cost rather than protocol waiting.

The size of that mis-attribution is now measured, not hypothetical: exp_20's
refreshed profile ranks `sync` as the **second-largest non-GEMM pool on shape 6
at 168.6 µs**, ahead of both reduce and release. Folding a 168.6 µs stall into
the emit interval would have been one of the largest errors this figure could
contain.

The same argument is why `RELEASE_END` exists (the release is an interval worth
65.4 µs on shape 5, not an instant) and why `PUBLISH_END` exists (otherwise the
publish loop is charged to the next tile's mainloop).

Ring depth is derived the same way — from this kernel's own per-row worst case,
29 events on shape 6 and 20 on shape 5, giving depth 64 at 2.17× headroom.
The sibling's depth 24 is a gfx950 MoE fact about a different phase taxonomy
(M0-M9 plus service stripes) and was not inherited.

## Phase 2 — patch APPLIED, parity gate is the next deliverable

The kernel-edit gate was opened and the patch landed: **21 sites, +109 lines,
0 deletions**, all inside `#if HK_GEMM_RS_MI300X_TRACE`, **default 0**.
`design.md` §9 records the anchor resolution — the file had grown from 893 to
951 lines under exp_26, every planned line number shifted, three of this
document's own anchors were recorded at the wrong indentation and were caught
by the anchor check rather than by a mispatch, and one planned site turned out
to need no edit because aggregate initialization already value-initializes the
new trailing member to 0.

Still blocked on the **GPU lease** (exp_23 and exp_21 are ahead). Arm (a)
capture, the tick calibration and arm (b) all need the node; nothing has been
launched.

## Policy, stated up front and binding on everything below

- The instrumented build is a **diagnostic arm**. `HK_GEMM_RS_MI300X_TRACE` is
  OFF for every campaign timing run and **no microsecond it produces is ever
  reported as a performance number**. The single exception is the ON-vs-OFF
  perturbation estimate, which is labelled as such.
- The MFMA row is **"CTAs in MFMA phase — an occupancy proxy"**, never MFMA
  utilization, and its denominator is `304 − NR` (272 on shape 5), printed in
  the axis label and in every row of `timeline_bins.csv`.
- Rank 0 is plotted; rank-max is supplementary; **rank symmetry is stated and
  quantified, never averaged away**.
- One process drives all eight devices in arm (b) — this is not the
  evaluator's topology, and any comparison against evaluator-measured numbers
  carries that caveat.

## Schemas (to be restated verbatim with the data)

- `events.json` — `exp22.events.v1`. Common envelope: `arm`, `kind`
  (`phase_ring` | `kernel_trace`), `shape`, and either `ranks.<r>.events`
  (flat `[cta, phase_id, ticks]` triples, `ticks_per_us` alongside) or
  `intervals` (`{name, resource, begin_ns, end_ns}`).
- `timeline_bins.csv` — `arm, bin_index, t_us_start, t_us_end, mfma_frac,
  mfma_denominator, hbm_gbs, xgmi_gbs, tflops`.
- `b0_kernel_map.json` — `exp22.b0_kernel_map.v1`: `rules`, `classified`
  (name → count/ns/resource/matched_rule), and **`unclassified`**, which is
  reported here rather than silently bucketed.
- `parity.json` — `exp22.parity.v1`: per-arm, per-instantiation
  VGPR/AGPR/SGPR/scratch/spill/LDS tuples plus the verdict.
- `tick_rate.json` — `exp22.tick_rate.v1`: the regression points, the measured
  `ticks_per_us`, and the sibling's 100 MHz hypothesis recorded for comparison.
- `validation.json` — `exp22.validation.v1`: the ±10% integral report per arm.

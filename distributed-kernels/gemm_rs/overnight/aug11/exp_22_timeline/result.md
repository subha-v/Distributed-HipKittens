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

## Phase 2 — blocked on two gates held by the orchestrator

1. **Kernel-edit gate.** `design.md` §9 is a 22-site additive patch to
   `gemm_rs_mi300x.cpp`, entirely inside `#if HK_GEMM_RS_MI300X_TRACE`
   (default 0). It is **not applied**: another agent is compiling waterfall
   rung binaries from that exact file and an edit mid-build would silently
   contaminate its arms.
2. **GPU lease.** Arm (a) capture, the parity gate's flag-ON build check, and
   arm (b) all need the node. Nothing has been launched.

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

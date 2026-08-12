# aug11 GEMM-RS — PLOTS index

paper figure → data file → experiment → status. Every number that reaches a
figure must live in a `.json`/`.csv` here, never only in prose.

Numbering per the charter update: the figure queue is **exp_20…exp_25** under
`overnight/aug11/`, distinct from the optimization sessions' `exp_01…exp_14`
under `overnight/experiments/`.

| paper fig | question | data file | experiment | status |
|---|---|---|---|---|
| — | Q3 bottleneck attribution | `aug11/exp_20_attribution/ablation.json`, `counters.json` | exp_20 | **LANDED** — 6 shapes × 5 stages, freshness gate +0.96% |
| Fig 2 | Q4 saturation vs CTA count | `aug11/exp_21_saturation/saturation.json`, `knees.json`, `saturation.csv`, `ceilings.txt` | exp_21 | **LANDED** — 260 points, 224/224 checksums, 1 distinct fold. **Egress knee C=16 of 304 (5.3%)** for 7-peer, **C=2** single-link; falsifier not triggered. Protocol costs 0.44× of egress at the knee (identical payload bytes); conc/iso 0.957 with GEMM slowdown 1.221. Panel b has **no knee ≤ 304**. H4 linearity falsified past C≈160 (memory-path-bound: 4549 of 4593 GB/s). Ceilings all node-sourced: 64.0 GB/s per xGMI link, 448 GB/s aggregate, 5325 GB/s HBM, 1307.4 TFLOPS bf16. `coarse` cross-check queued |
| Fig 3 | Q4 per-layer resource timeline | `aug11/exp_22_timeline/events_ours_*.json`, `events_b0_reference.json`, `timeline_bins.csv`, `b0_kernel_map.json`, `parity.json`, `tick_rate.json`, `validation.json` | exp_22 | **COMPLETE, 2 arms.** Overlap quantified: MFMA and xGMI both live in **17/68 bins (25%)** for ours, **0/53 (0%)** for reference. Our emit peaks **420.4 GB/s** vs RCCL's **244.4** on the same 58.72 MB (**1.72×**), under the ~448 ceiling. Reference epoch strictly serialized: 243.2 GEMM / 43.4 bias / **241.3 RCCL = 45.7% comm share**. Parity PASS 0 failures; correct at both tolerances; **zero ring drops on 304/304 CTAs**; tick rate agrees 3 ways within 0.27%. Arm (c) deliberately not taken. **Note: the ±10% integral gate is a conservation identity (residuals ~1e-16) and is NOT independent validation — the ceiling comparison is** |
| — | Q4 tick-rate calibration (gfx942) | `aug11/exp_21_saturation/saturation.json`, `aug11/exp_22_timeline/tick_rate.json` | exp_21 + exp_22 | **LANDED** — `s_memrealtime` = **99.7366 MHz**, two independent methods, −0.264% from the sibling's assumed gfx950 100 MHz |
| — | Q1 rung-distinctness evidence | `aug11/exp_23_waterfall/fingerprints.json` | exp_23 | **LANDED** — proves the four rungs are four binaries, distinguished at the sites their mechanisms predict |
| Fig 4 | Q1 knob waterfall (**money figure**) | `aug11/exp_23_waterfall/waterfall.json`, `stats.json`, `fingerprints.json` | exp_23 | **LANDED** — 4 draws (2 fwd / 2 rev); a→b 1.084×, b→c 1.115× cumulative; null within 0.2% of c; structural prediction held |
| — | Q6 external ladders | `aug11/exp_24_ladders/ladders.json` | exp_24 | queued |
| — | Q5 per-shape sensitivity | `aug11/exp_25_sensitivity/knob_by_shape.json`, `sensitivity_points.csv` | exp_25 | **LANDED (negative)** — P1 falsified in sign under all 4 comm-share definitions, then shown **not identifiable** (mask and comm share confounded at ρ=±1.00); P2 falsified as written, confirmed on the 32-56 plateau |

## Data already in hand that figures can draw on

These were produced before tonight's charter but are plot-ready and directly
relevant. Sources are committed under `overnight/experiments/`.

| content | file | relevance |
|---|---|---|
| xGMI fabric counters at the WGM fix: 117.48 MB carried vs 117.44 MB useful (1.0003×), 99.9% at full 64 B, EA write latency −38.6% | `experiments/exp_08_egress/result.md` | Fig 4 supporting: a task-order change moved zero bytes and −27.9% |
| effective xGMI links 2.02 → 7.53 of 8 | `experiments/exp_08_egress/result.md` | Fig 2 / Fig 4 |
| **`s_memrealtime` = 99.7366 MHz on gfx942** (10.0264 ns/tick), 5 reps, spread 0.0101%, measured two-stage against `steady_clock` | `aug11/exp_21_saturation/saturation.json` → `tick_rate_hz` | **exp_22 needs this**; the sibling's 100 MHz is a gfx950 statement, this one is measured here |
| `NR` curve over {4,8,16,24,32,40,48,56,64,80} × 6 shapes, with null-arm floors | `experiments/exp_13_cta_split/result.md`, `sweep_*.json` | **Fig 4 rung (d)** — the placement-flatness exhibit, already measured |
| per-call decomposition: clone / host / device / barrier, plus a null-kernel floor at 1…608 CTAs | `experiments/exp_12_percall/result.md` | Q6 caveat: ~92 µs of every graded call is harness machinery both arms pay |
| ours vs reference GEMM+RCCL, same-run graded | `experiments/exp_07_cold_l2/result.md` | exp_17 prior arm |
| ours vs frozen rank-1, same-run graded, per shape | `experiments/exp_10_rank1/rematch.md`, `exp_14_tile_waves/result.md` | exp_17 prior arm |
| stage attribution at three successive configs (pre-WGM, post-WGM, post-E1b) | `experiments/logs/reattribute_*.log` | Q3 trend; **all stale vs current best** |

## Known-stale, must be regenerated at the current config

- The attribution table in `overnight/RESULTS.md` predates WGM, E3, the NR
  re-sweep and the tile re-sweep. exp_13 replaces it.
- Any "current best" vector quoted before `experiments/exp_14_tile_waves`.

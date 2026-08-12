# aug11 GEMM-RS — PLOTS index

paper figure → data file → experiment → status. Every number that reaches a
figure must live in a `.json`/`.csv` here, never only in prose.

Numbering per the charter update: the figure queue is **exp_20…exp_25** under
`overnight/aug11/`, distinct from the optimization sessions' `exp_01…exp_14`
under `overnight/experiments/`.

| paper fig | question | data file | experiment | status |
|---|---|---|---|---|
| — | Q3 bottleneck attribution | `aug11/exp_20_attribution/ablation.json`, `counters.json` | exp_20 | **LANDED** — 6 shapes × 5 stages, freshness gate +0.96% |
| Fig 2 | Q4 saturation vs CTA count | `aug11/exp_21_saturation/saturation.json` | exp_21 | queued |
| Fig 3 | Q4 per-layer resource timeline | `aug11/exp_22_timeline/events.json`, `timeline_bins.csv` | exp_22 | queued |
| Fig 4 | Q1 knob waterfall (**money figure**) | `aug11/exp_23_waterfall/waterfall.json`, `stats.json`, `fingerprints.json` | exp_23 | **LANDED** — 4 draws (2 fwd / 2 rev); a→b 1.084×, b→c 1.115× cumulative; null within 0.2% of c; structural prediction held |
| — | Q6 external ladders | `aug11/exp_24_ladders/ladders.json` | exp_24 | queued |
| — | Q5 per-shape sensitivity | `aug11/exp_25_sensitivity/knob_by_shape.json` | exp_25 | queued |

## Data already in hand that figures can draw on

These were produced before tonight's charter but are plot-ready and directly
relevant. Sources are committed under `overnight/experiments/`.

| content | file | relevance |
|---|---|---|
| xGMI fabric counters at the WGM fix: 117.48 MB carried vs 117.44 MB useful (1.0003×), 99.9% at full 64 B, EA write latency −38.6% | `experiments/exp_08_egress/result.md` | Fig 4 supporting: a task-order change moved zero bytes and −27.9% |
| effective xGMI links 2.02 → 7.53 of 8 | `experiments/exp_08_egress/result.md` | Fig 2 / Fig 4 |
| `NR` curve over {4,8,16,24,32,40,48,56,64,80} × 6 shapes, with null-arm floors | `experiments/exp_13_cta_split/result.md`, `sweep_*.json` | **Fig 4 rung (d)** — the placement-flatness exhibit, already measured |
| per-call decomposition: clone / host / device / barrier, plus a null-kernel floor at 1…608 CTAs | `experiments/exp_12_percall/result.md` | Q6 caveat: ~92 µs of every graded call is harness machinery both arms pay |
| ours vs reference GEMM+RCCL, same-run graded | `experiments/exp_07_cold_l2/result.md` | exp_17 prior arm |
| ours vs frozen rank-1, same-run graded, per shape | `experiments/exp_10_rank1/rematch.md`, `exp_14_tile_waves/result.md` | exp_17 prior arm |
| stage attribution at three successive configs (pre-WGM, post-WGM, post-E1b) | `experiments/logs/reattribute_*.log` | Q3 trend; **all stale vs current best** |

## Known-stale, must be regenerated at the current config

- The attribution table in `overnight/RESULTS.md` predates WGM, E3, the NR
  re-sweep and the tile re-sweep. exp_13 replaces it.
- Any "current best" vector quoted before `experiments/exp_14_tile_waves`.

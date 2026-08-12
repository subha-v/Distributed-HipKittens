# aug11 — figure ledger for `docs/distributed/PAPER.md`

One row per paper figure. **This is the morning read: start here.** Status
vocabulary is exactly four values — **DATA LANDED** (plot-ready file exists),
**PARTIAL** (some series/rungs measured, the missing ones named),
**PENDING** (instrument built and gated, data not yet collected),
**NOT STARTED** (no instrument).

## At a glance

| status | figures |
|---|---|
| DATA LANDED | Q3 attribution · §3.2 Finding-1 · PAPER Figure 4 (worked example) |
| PARTIAL | PAPER Figure 1 (comm fraction) · Q1 waterfall (3 rungs of 5) · task-order/readiness |
| PENDING | Q4a saturation |
| NOT STARTED | Q4b timeline · Q5 sensitivity · Q2 placement · Q6 external ladders |

**Two numbering schemes are in play and they disagree.** `PAPER.md` explicitly
numbers only **Figure 1** (§1, the comm-fraction motivation) and **Figure 4**
(§4.2, the worked example of one producer-consumer edge); every §6 evaluation
figure is referred to by its question (Q1…Q6), unnumbered. `aug11/CLAUDE.md`
uses a second scheme keyed to §6 — Fig 2 = saturation, Fig 3 = timeline,
**Fig 4 = the Q1 waterfall**, Fig 5 = placement, Fig 6 = task order, Fig 7 =
sensitivity. Only Fig 1 agrees between the two. **The charter's "Fig 4" (the
waterfall) is NOT `PAPER.md`'s Figure 4 (the edge worked example).** Rows below
are keyed by `PAPER.md` section/question first, with the charter's number in
brackets.

## The ledger

| figure (PAPER.md ref) | charter ref | what it plots | data file | generating experiment | status |
|---|---|---|---|---|---|
| **Figure 1** — §1 opening measurement | Fig 1 | communication fraction of layer time: dispatch + combine as a share of MoE layer time on 8×MI350X, alongside the GEMM-RS reduce-scatter epilogue share on 8×MI300X | `exp_33_attribution/phase_stamps.json` → `arms.production.phases` (MoE half only) | exp_33 (MoE half, derived); GEMM-RS half is on the `GEMM-RS` branch, not ours | **PARTIAL** |
| §2.1 token/expert flow | — | schematic: two GPUs, a token routed to experts, the output tile coming home (COMET Fig-2 style) | none — diagram | — | **NOT STARTED** (schematic, needs no campaign) |
| §3.2 **Finding 1** — "the paper's pivotal measurement, gets its own figure" | — | pool interference is coordination, not bytes: concurrent-GEMM inflation vs capacity tax; deleting the payload copy moves the GEMM +5.7 µs while 831 µs of inflation remains; inflation tracks atomic/fence count, and coarsening transfers *increases* it | aug10 `exp_20`, `exp_03`, `exp_05`–`exp_07` result files | aug10 (prior sessions) | **DATA LANDED** |
| **Figure 4** — §4.2 worked example | — | operation count on one cross-GPU edge: ~30 coordination ops across three CTAs (pool design) → amortized ~10⁻⁴ ops and no third CTA (final design); fabric byte-limit 52.8 GB/s coalesced 4 B atomics vs 54.9 GB/s 16 B stores | counts derived in §4.2; fabric numbers from aug10 `exp_21` ubench | aug10 exp_21 + design analysis | **DATA LANDED** (diagram + counts, not a campaign) |
| §6 **Q1** — the waterfall (the money figure) | **Fig 4** | homogeneous → +epilogue-carried payload → +injection bound → +arrival-sized signals → +task reorder, each rung's p50 and ratio vs `production`, paired same-run | `exp_35_waterfall/waterfall.json` (schema `exp35.waterfall.1`) | exp_35 | **PARTIAL — 3 rungs of 5.** (a) `pf6gm_mega` 6,900.2 sd 15.5 / 0.8950; (b) payload, throttle OFF `g=65` 7,110.8 sd 13.7 / 0.9226; (c) + injection bound depth 4 `g=353` 6,495.8 sd 6.9 / 0.8422. Missing: (d) mode 14 → `status: pending_exp_34`; (e) nc-major → `status: not_built`. Both are `null` rows with `blocked_by`, so the figure completes in place without re-running (a)–(c). |
| §6 **Q2** — do dedicated communication blocks ever win? | Fig 5 | paired best-vs-best (mode 2 pool at C=64 g=1 vs mode 12 at C=16 vs mode 14 at C ∈ {0, 8, 16}), plus the pool-size sweep at the final design (predicted flat→degenerate) and the reducer-count sweep | — | exp_37 | **NOT STARTED.** Hard-blocked for the C=0 arm: `C = 0` is illegal in mode 12 (`moe_mps_adapter.cuh:373`), so the dedication axis cannot reach zero without exp_34's mode 14. |
| §6 **Q3** — where does the time go now? | — | phase attribution at the ratchet: stacked bar over plan M3–M5 / M6 GEMM-1 / M7 GEMM-2 (GEMM proper vs epilogue surcharge) / combine M8–M9, for `mps_mega`, plus the `production` three-stage split | `exp_33_attribution/phase_stamps.json` (schema `exp33-phase-stamps-1`; `arms.mps_mega.phases.stacked_bar_phases` is the partition to plot) | exp_33 | **DATA LANDED.** n=10, all gates green. M7 2,701.8 ±23.74 (46.2 % of interior) > M6 2,453.3 ±2.51 (41.9 %); the two GEMMs are 88.1 %. plan 372.8 ±0.67, combine 324.2 ±21.12, interior 5,852.1 ±9.56. |
| §6 **Q4a** — resource saturation vs CTA count (NanoFlow Fig-7 analog) | Fig 2 | per-resource throughput vs CTA count (MFMA TFLOPS / HBM GB/s / xGMI GB/s), each curve **isolated** and **concurrent**, knee annotated; the gap between the two curve families *is* Finding 1 as a figure | `exp_22_fig7_saturation/saturation.json` (schema `exp22-saturation-1`) — **does not exist yet** | exp_22 | **PENDING.** Ubench `e22_saturation.hip` + one-command sweep + summarizer all built, CPU gate green; sweep is ~3 min of GPU. Plot contract in `exp_22_fig7_saturation/plots_row.md`: use `derived.knees[*].knee_ctas`, do not re-derive; drop/mark any point whose `concurrent_is_really_concurrent` is false. |
| §6 **Q4b** — per-layer utilization timelines (NanoFlow-v2 Fig-10 analog) | Fig 3 | per-CTA phase occupancy over time for three arms on one routing seed: RCCL-eager `production` (torch profiler) vs homogeneous `pf6gm_mega` vs the `mps_mega` ratchet | `exp_23_fig10_timeline/rank0_events.json` (per arm) + `timeline_bins.csv` — neither exists | exp_23 | **NOT STARTED.** Plan and design written. Needs the per-CTA phase event ring behind the diagnostics flag and a resource-tuple parity gate with the flag compiled in but OFF. |
| §6 **Q5** — sensitivity: batch, seqlen, imbalance | Fig 7 | best config and end-to-end µs over T ∈ {512, 1024, 2048, 4096} × routing std ∈ {0, 0.032, 0.05}, with `[MPS SPIN]` recorded at every point; at std = 0.05 also C ∈ {0, 8, 16} | `exp_36_sensitivity/sensitivity_grid.json` — does not exist | exp_36 | **NOT STARTED.** The one regime where a service pool can re-enter; `[MPS SPIN]` = 0/0 at std = 0 is already measured (exp_33 §6, exp_35), so std = 0 is the anchor column. |
| §4.3 / §3.3 — readiness curve and producer task order | Fig 6 | `P(ready by t) = (t/S)^8`, the measured "nothing consumed before the producer phase was ≥93.9 % complete", and the nc-major staircase that unblocks 15/16 of the combine early | readiness analysis in `exp_30_coarse_readiness/`; aug11 exp_29 result; nc-major measurement **does not exist** | exp_29 / exp_30 (analysis); the reorder is unbuilt in any commit | **PARTIAL.** The "signals buy an event of probability ≈ 0" half is analyzed and quantified (~926,000 deleted atomics per rank per epoch, itemized). The "task order is the lever" half has no measurement on MoE — the MoE reorder exists in no commit; GEMM-RS exp_08's link-order −27.9 % is the only measured instance. |
| §6 **Q6** — external ladders | — | MoE vs tuned production (AITER+MoRI) **and** vs PyTorch+RCCL eager, so fusion / scheduling / protocol are attributed separately | — | stretch (eager arm to build) | **NOT STARTED.** `production` (AITER+MoRI) is already the same-run denominator in every campaign; the missing arm is the eager dispatch→GEMM→combine reference through the same harness shapes. |

## Caveats a plotter must carry

1. **Figure 1's MoE fraction is derived here, not an exp_33 verdict.** From
   `exp_33` §4: `production` dispatch 918.70 ±2.01 + combine 1,356.02 ±11.68 =
   **2,274.7 µs**, i.e. **29.5 % of the 7,708.3 µs p50** (rank-max) or **24.4 %**
   on the rank-0 variant (902.85 + 978.41 = 1,881.3). **PAPER.md §1 currently
   claims "roughly a third"; at this shape the honest range is 24–30 %.** Either
   soften the caption or re-derive it on a shape that supports the stronger
   claim. Note also that `production`'s rank-max stages sum to +6.6 % above its
   own p50 (max is taken independently per stage), so a stacked bar built from
   rank-max stages over-counts.
2. **Q3's stacked bar and the end-to-end p50 are two different
   measurements.** Every `[MPS TS]` value is a running device max that is never
   reset, so the phases describe the **final soak epoch**, while `arm_p50_us` is
   a timed-iteration rank-max. Do not sum the phases and compare to the total.
   `plan + M6 + M7 + combine = interior` closes to **0.0 µs in all 10
   rotations**; the `p50 − interior = 641.7 µs` residual is instrument-mixed
   dispatch/launch/skew and is an upper bound good to ~100 µs.
3. **Never add `service_drain` into the Q3 stacked bar** — it overlaps M7 and
   combine. The five `stacked_bar_phases` in the JSON are the ones that
   partition the kernel.
4. **The M7 epilogue surcharge (~817–898 µs, ~30–33 % of M7) is a cross-run
   proxy, not a measurement.** `pf6gm_mega` emits no stamps, and the one
   instrument that could close it (`K0_PF6GM_DECOMP`) is absent from
   `run_campaign.sh`'s `-e` forwarding list. Label the surcharge segment as a
   proxy in the figure or split it out with a hatch.
5. **Q3's `combine` bar must carry its error bar.** stderr is 6.5 % of its own
   mean and it is anti-correlated with M7 at **r = −0.90**; the honest object is
   the 3,026.1 ±10.2 µs M7+combine block.
6. **Q1's rung (a)→(b) step is not single-variable** and cannot be made one
   (rung (a) is a different kernel with no MPS protocol). Step (b)→(c) is
   strictly single-variable. Annotate the figure accordingly.
7. **Q1's rungs (b) and (c) are different `.hsaco` builds from the same pinned
   source** — the throttle is a compile-time specialization — which is why `M6`
   drifts 47 µs between them. That drift is codegen, not mechanism.
8. **No arm's `out` is bit-reproducible** at this commit, so correctness in
   every figure's caption is the `[MOK GATE]` tolerance gate plus the NaN-poison
   detector, never a bit compare.

## Highest-value unblocking actions, in order

1. **One-line harness edit: add `K0_PF6GM_DECOMP` to `run_campaign.sh`'s `-e`
   forwarding list.** It converts every M7 surcharge proxy into a same-run
   measurement and it is the single highest-value change outstanding for the Q1
   waterfall and the Q3 stacked bar.
2. **exp_34 mode 14 through protocol review and the gate ladder** — it unblocks
   Q1 rung (d) *and* the only legal C=0 point for Q2.
3. **exp_22's 3-minute sweep** — Q4a goes from PENDING to DATA LANDED for the
   price of one GPU slot.

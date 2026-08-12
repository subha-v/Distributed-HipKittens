# aug11 — figure ledger for `docs/distributed/PAPER.md`

One row per paper figure. **This is the morning read: start here.** Status
vocabulary is exactly five values — **DATA LANDED** (plot-ready file exists),
**PARTIAL** (some series/rungs measured, the missing ones named),
**SPEC READY** (instrument specified and its off-path tooling built and
self-tested, but the code edit is not applied), **PENDING** (instrument built and
gated, data not yet collected), **NOT STARTED** (no instrument).

## At a glance

| status | figures |
|---|---|
| DATA LANDED | **Q4a saturation** · Q3 attribution · §3.2 Finding-1 · PAPER Figure 4 (worked example) |
| PARTIAL | **Q5 sensitivity** (T=4096 only) · PAPER Figure 1 (comm fraction) · Q1 waterfall (5 rungs of **6**, but two of them on a **broken baseline**) · **Q2 placement** (mode-14 C ∈ {0,8,16} landed, C=0 blocker cleared) · task-order/readiness |
| SPEC READY | Q4b timeline |
| PENDING | — |
| NOT STARTED | Q6 external ladders |

> **READ THIS BEFORE PLOTTING ANYTHING MEASURED AT `291dfa08` OR LATER.**
> exp_34 established that commit `291dfa08` (mode 14) **regresses the mode-12
> ratchet by +726.9 µs**, entirely in M7, and that the exp_24 injection throttle
> is **inert at that pin for both mode 12 and mode 14** (`g=353` ≡ `g=65` to
> within 2–7 µs, against −613.5 µs at rev 26). Every mode-12 denominator taken at
> `291dfa08` or later is a broken-transport number until that is fixed. exp_34
> §1 has the attribution, the phase localisation and the ISA census.

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
| **Figure 1** — §1 opening measurement | Fig 1 | communication fraction of layer time: dispatch + combine as a share of MoE layer time on 8×MI350X, alongside the GEMM-RS reduce-scatter epilogue share on 8×MI300X | `exp_33_attribution/phase_stamps.json` → `arms.production.phases` (MoE half only) | exp_33 (MoE half, derived); GEMM-RS half is on the `GEMM-RS` branch, not ours | **PARTIAL.** **Caption action required: `PAPER.md` §1 says "roughly a third"; the measured fraction is 24.4 % (rank-0) to 29.5 % (rank-max) and the caption must be softened to that range** or re-derived on a shape that supports the stronger claim. |
| §2.1 token/expert flow | — | schematic: two GPUs, a token routed to experts, the output tile coming home (COMET Fig-2 style) | none — diagram | — | **NOT STARTED** (schematic, needs no campaign) |
| §3.2 **Finding 1** — "the paper's pivotal measurement, gets its own figure" | — | pool interference is coordination, not bytes: concurrent-GEMM inflation vs capacity tax; deleting the payload copy moves the GEMM +5.7 µs while 831 µs of inflation remains; inflation tracks atomic/fence count, and coarsening transfers *increases* it | aug10 `exp_20`, `exp_03`, `exp_05`–`exp_07` result files | aug10 (prior sessions) | **DATA LANDED** |
| **Figure 4** — §4.2 worked example | — | operation count on one cross-GPU edge: ~30 coordination ops across three CTAs (pool design) → amortized ~10⁻⁴ ops and no third CTA (final design); fabric byte-limit 52.8 GB/s coalesced 4 B atomics vs 54.9 GB/s 16 B stores | counts derived in §4.2; fabric numbers from aug10 `exp_21` ubench | aug10 exp_21 + design analysis | **DATA LANDED** (diagram + counts, not a campaign) |
| §6 **Q1** — the waterfall (the money figure) | **Fig 4** | homogeneous → +epilogue-carried payload → +injection bound → +arrival-sized signals → +task reorder, each rung's p50 and ratio vs `production`, paired same-run | `exp_35_waterfall/waterfall.json` (schema `exp35.waterfall.1`) + `exp_34_mode14/mode14_arms.json` for (d)/(e) | exp_35 (+ exp_34 for the mode-14 rungs) | **PARTIAL — 5 rungs measured of 6, but (d)/(e) DO NOT SHARE (a)–(c)'s BASELINE and must not be plotted with them yet.** exp_34 measured (d) `g=481,mode=14` = **6,647.7** (n=4) and (e) `g=353,mode=14` = **6,650.9** (n=4), stamps-off, at pin `291dfa08` — where **mode 12 itself regressed to 7,224.2 from 6,497.3** (exp_34 §1, attributed to that very commit, +726.9 µs, all in M7). So at this pin rung (c) is broken and the (c)→(d) step reads −576.5 µs against a baseline no other rung shares. **Action: re-run (d)/(e) after the regression is fixed, then the figure is complete but for (f).** Also note the (d)→(e) step — deleting the per-task VMEM drain — measured **+3.2 µs, a null**, so the six-rung split can collapse back to five with the drain deletion reported as free. (f) nc-major task order, `status: not_built`. Prior text for (a)–(c) retained: **3 rungs of 6.** The rung count **grew from 5 to 6**: exp_34's confound fix split the mode-14 rung in two, because mode 14 also deleted the per-task VMEM drain and that deletion is now selectable (`g \|= 0x80`, `kCoarseKeepDrainBit`). Measured: (a) `pf6gm_mega` 6,900.2 sd 15.5 / 0.8950; (b) payload, throttle OFF `g=65` 7,110.8 sd 13.7 / 0.9226; (c) + injection bound depth 4 `g=353` 6,495.8 sd 6.9 / 0.8422. Owed: (d) `g=481,mode=14` — coarse readiness with the drain retained; (e) `g=353,mode=14` — + drain deletion; (f) nc-major task order, `status: not_built`. (d)/(e) are built and CPU-gated, not measured. Both `null` rows carry `blocked_by`, so the figure completes in place without re-running (a)–(c). |
| §6 **Q2** — do dedicated communication blocks ever win? | Fig 5 | paired best-vs-best (mode 2 pool at C=64 g=1 vs mode 12 at C=16 vs mode 14 at C ∈ {0, 8, 16}), plus the pool-size sweep at the final design (predicted flat→degenerate) and the reducer-count sweep | `exp_34_mode14/mode14_arms.json` for the C ∈ {0, 8, 16} mode-14 series; exp_36's C sweep is the mode-12 half | exp_37 (**campaign running**, pinned at `b5215081`) + exp_34 (mode-14 series) | **PARTIAL, and the C=0 blocker is now cleared.** exp_36's C sweep at T=4096 is **monotonically harmful in C** and the dedicated-pool arm (mode 2, C=64) is the worst point at +331.5 µs. exp_34 supplies the previously impossible **C=0** point (legal only in mode 14; `C = 0` is rejected in mode 12 at `moe_mps_adapter.cuh:373`) and the series is **monotone, not flat**: C=0 **6,650.9** (n=4) → C=8 **6,725.6** (n=2) → C=16 **6,780.4** (n=2), i.e. **8.1–9.3 µs per reserved CTA**. The prediction of flatness is **falsified**, and in the strongest possible way for the paper's claim: in mode 14 the pool provably has **no work at all** (bit 26 never set across 4,032 `pperr` readings; `[MPS TS] DRAIN=0`; `[MPS SPIN] 0/0`), so the penalty is **pure CTA-capacity loss with contention excluded by construction** — no placement policy can be rescued by giving the pool less to do. Caption caveat: this series sits at pin `291dfa08`, so it may be compared *within itself* but not against mode-12 numbers from other pins (exp_34 §1). |
| §6 **Q3** — where does the time go now? | — | phase attribution at the ratchet: stacked bar over plan M3–M5 / M6 GEMM-1 / M7 GEMM-2 (GEMM proper vs epilogue surcharge) / combine M8–M9, for `mps_mega`, plus the `production` three-stage split | `exp_33_attribution/phase_stamps.json` (schema `exp33-phase-stamps-1`; `arms.mps_mega.phases.stacked_bar_phases` is the partition to plot) | exp_33 | **DATA LANDED.** n=10, all gates green. M7 2,701.8 ±23.74 (46.2 % of interior) > M6 2,453.3 ±2.51 (41.9 %); the two GEMMs are 88.1 %. plan 372.8 ±0.67, combine 324.2 ±21.12, interior 5,852.1 ±9.56. |
| §6 **Q4a** — resource saturation vs CTA count (NanoFlow Fig-7 analog) | Fig 2 | per-resource throughput vs CTA count (MFMA TFLOPS / HBM GB/s / xGMI GB/s), each curve **isolated** and **concurrent**, knee annotated; the gap between the two curve families *is* Finding 1 as a figure | **`exp_22_fig7_saturation/saturation.json`** (schema `exp22-saturation-1`; raw `saturation_plan.jsonl`) | exp_22, `E22_SRC_REV 10`, 250/250 plan points, 2026-08-12T10:45Z | **DATA LANDED.** Knees: single xGMI link **8 CTAs** (plateau 56.9 GB/s = 74.1 % of 76.8); 7-link fabric **32 CTAs** at mlp4 (355.0 GB/s = 66.0 % of 537.6); HBM **128 CTAs** (4,151.9 GB/s = 51.9 %); **MFMA has no knee** — linear to 256 CTAs, R² ≥ 0.99989. Communication saturates at 8–32 CTAs against 256 CTAs of compute capacity. H4 SUPPORTED ×3; H3 `REFUTED_PAYLOAD_RIDES_FREE` (concurrent/isolated median **0.9951**, n=42, vs **0.878** with the protocol compiled in and −42 % at g=16); H1 falsifier fired, reported unsoftened. |
| §6 **Q4b** — per-layer utilization timelines (NanoFlow-v2 Fig-10 analog) | Fig 3 | per-CTA phase occupancy over time for three arms on one routing seed | `exp_23_fig10_timeline/rank0_events.json` (schema `exp23-events-1`) + `timeline_bins.csv` (`exp23-bins-1`) — **not collected** | exp_23 | **SPEC READY.** `patch_spec.md` done (Tier A = five one-line `ts_last`→`ts_mark` swaps at sites that already decode the config and read the clock, plus a one-line host buffer resize, **no ABI change**); `parse_events.py` / `bin_timeline.py` / `xcheck.py` / `selftest.py` all built and self-tested (8 mutation classes, every verdict proven to flip). **Verdict GO at Tier A**, ~15 % parity risk, ~45–55 min of GPU for all three arms. **Two plan corrections:** the homogeneous panel must be `mps_mega C=0,mode=0` labelled "bulk, no overlap" — `pf6gm_mega` cannot be instrumented (55-word descriptor, no MPS state slot); and the promised amd-smi bandwidth cross-check is replaced by a UMC duty-cycle check ±15 pp plus a 52.8 GB/s fabric-ceiling bound, because this node exposes no live xGMI throughput counter. |
| §6 **Q5** — sensitivity: batch, seqlen, imbalance | Fig 7 | best config and end-to-end µs over T × routing skew, with `[MPS SPIN]` at every point | **`exp_36_sensitivity/sensitivity_grid.json`** (49 points) | exp_36, pin `b5215081`, 15 campaigns + 10 screens, all gates green | **PARTIAL — one shape, and the missing axes are themselves the finding.** T = 4096 is the **only** feasible point: T ≤ 512 is refused by a host guard (`ab.py:554`), and T = 1024 / 2048 return **wrong output on both megakernel arms** while `production` passes. The routing-std axis **does not exist** — `K0_SYNTH_ROUTE` is rejected under the hard-wired `K0_INPUT_MODE=mok_synthetic` and `synthetic_routes.py` has no `std` parameter at all, so std = 0.032 / 0.05 were never settable. What the figure *can* show at T=4096: the throttle is worth **10–30× more than placement**, and **placement is monotonically harmful** — `C=4` 6,466.1 ≈ `C=8` **6,469.5** < `C=16` 6,498.1 (ratchet) < `C=32` 6,719.3 < `C=64` mode 2 6,829.6; depth 8 costs +72.7 and throttle-off +626.9. `[MPS SPIN]` 0/0 at all 25 green points. Reproduced under a second router seed with an identical ranking. |
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
9. **Q4a: three `knee_ctas` values in the JSON are artifacts and must not be
   annotated as knees.** `xgmi rr7` at mlp1 and mlp8 has **not** saturated at
   C=64 (C=64/C=32 = 1.87 and 1.73), so their `knee_ctas = 64` is the plateau
   definition applied to a still-rising curve — label them "no knee in range".
   Same for `mfma`, whose `knee_ctas = 224` is just linearity. And
   `reserved_only` points carry `value = 0` **by design** (that arm holds C CTAs
   idle to price the reservation): its payload is `cmp_tflops_median`, so never
   plot it on the bandwidth axis.
10. **Q4a's H1 falsifier fired and the mechanism was fine.** The gate asked for
   75 % of the 76.8 GB/s nominal (57.6 GB/s) but the achievable ceiling is
   56.9–57.3 GB/s at every C from 8 to 64, so **no CTA count could have
   passed** — the threshold was calibrated against a spec sheet instead of a
   measured ceiling. Report it unsoftened, and note that the substantive
   question still separates cleanly: C=8 55.2 → C=64 56.5 GB/s, so **8× the
   pushers buys +2.4 %**.
11. **Q4a's concurrency is proven, not assumed.** Every concurrent point
   satisfies `wall < res_span + cmp_span` at **75/75** (span-sum/wall
   1.62–1.82) — a stronger check than the by-construction `overlap_pct`. The
   headline pairing to quote: carrying the payload costs the compute pool
   **0.073 %** median, while *dedicating* the same 16 CTAs costs **6.25 %**
   before a byte moves. Two honest exceptions: HBM at C=128 (10.1 %) and the
   saturated-link corner (14.2 %).
12. **Every number in Q5 is a T=4096 number** and every figure caption that
   implies a batch/seqlen sweep must say so. There is no second shape: T ≤ 512
   is refused and T = 1024 / 2048 are incorrect on both megakernel arms.
13. **Rungs and series must carry their PIN, not just their config.** exp_34 §1
   is the reason: the same `C=16,g=353,mode=12` config measures 6,497.3 µs at
   `f113d73f` and 7,224.2 µs at `291dfa08` in the *same session*, with
   `production` and `pf6gm_mega` unchanged to <5 µs. Two rungs of the Q1
   waterfall now sit either side of that step. Any figure mixing pins is
   plotting a compiler artifact as a mechanism.
14. **`servicedrain` is not a number for mode 14 and must never be plotted as
   one.** No CTA enters the drain, so `K0P6_MPS_TS_DRAIN` is never stored and the
   cell reads as its never-written sentinel (`DRAIN=0`, and the derived delta
   comes out as a large negative). `mode14_arms.json` carries
   `servicedrain_status` for exactly this reason; the *absence* is the evidence,
   and it belongs in the caption as text, not on an axis.

## Highest-value unblocking actions, in order

1. **Fix the `291dfa08` mode-12 regression (exp_34 §1).** It is worth **+726.9 µs**
   — more than every remaining item on this list combined — it is live on `HEAD`,
   and it silently rebases the Q1 waterfall and every future mode-12 denominator.
   The mode-12 source path is byte-identical across the commit and the throttle's
   four `vmcnt` instantiations are still in the ISA, so this is a codegen /
   allocation effect in the shared function and it needs a source owner. **Then
   re-run Q1 rungs (d)/(e)** (~30 min of GPU) and the waterfall is complete but
   for (f).
2. ~~**exp_34 mode 14 on GPU**~~ — **DONE.** Full ladder green (world-8
   correctness, both negative controls, bit 26 never set, 600-epoch soak, poison
   selftest), 26 campaigns across 4 interleaved batches. Rung **falsified**
   against its 5,990–6,440 band at 6,650.9 µs; Q2's C=0 point landed; the drain
   deletion priced at a null. See `exp_34_mode14/result.md`.
2. **One-line harness edit: add `K0_PF6GM_DECOMP` to `run_campaign.sh`'s `-e`
   forwarding list.** It converts every M7 surcharge proxy into a same-run
   measurement and it is the single highest-value change outstanding for the Q1
   waterfall and the Q3 stacked bar.
3. **exp_23's Tier-A patch apply** (~20 min mechanical) + the 4-TU CPU parity
   gate, then ~45–55 min of GPU: Q4b goes SPEC READY → DATA LANDED.
4. **Fix the T = 1024 / 2048 correctness defect** — it is what stands between Q5
   and a real batch/seqlen axis, and it is a defect in both megakernels, not a
   figure problem.

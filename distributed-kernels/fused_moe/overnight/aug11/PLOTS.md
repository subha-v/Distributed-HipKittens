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
| DATA LANDED | **Q2 placement** · Q4a saturation · Q3 attribution · §3.2 Finding-1 · PAPER Figure 4 (worked example) |
| PARTIAL | Q1 waterfall (**4 of 5 rungs** — the count is back to five; rung (d) is on a **broken baseline** and its **re-run is now unblocked** at `275d2c2a`) · Q5 sensitivity (T=4096 only) · PAPER Figure 1 (comm fraction) · task-order/readiness |
| SPEC READY | Q4b timeline — instrument **built** at `d13acacf`, **default-off** since `275d2c2a`; its **tuple parity gate is demoted** and the ring owes its own timing arm |
| PENDING | — |
| NOT STARTED | Q6 external ladders |

> **THE `291dfa08` REGRESSION IS FIXED AT `275d2c2a` (exp_38) — AND THE PIN RULE
> STILL STANDS.** Mode 14 is now behind `K0P6_MPS_ENABLE_MODE14` (default 0) and
> the exp_23 ring behind `K0P6_MPS_E23_RING` (default 0); the default build's
> `.text` is **byte-identical to rev 26 `f113d73f`** (sha256 `642646fc…a541a7`,
> 179,520 B), the ratchet reproduces at **6,488.7 µs (+0.09 %)** and the injection
> bound is **alive again at −618.7 µs** against −613.5 at rev 26 and −1.8 at the
> broken pin. **Anything measured between `291dfa08` (inclusive) and `275d2c2a`
> (exclusive) is still a broken-transport number — that window contains exp_34's
> mode-14 series and nothing else** — and **Q1 rung (d) must be re-run at the
> healthy pin before it can go on the same axis as (a)–(c)**. exp_38's `.text`
> sha256 is now the parity gate of record; the resource tuple is demoted, because
> it reports scratch SIZE (pinned at 128 B) and was blind to scratch OP COUNT
> (19 → 168).

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
| §6 **Q1** — the waterfall (the money figure) | **Fig 4** | homogeneous → +epilogue-carried payload → +injection bound → +arrival-sized signals → +task reorder, each rung's p50 and ratio vs `production`, paired same-run | `exp_35_waterfall/waterfall.json` (schema `exp35.waterfall.1`) + `exp_34_mode14/mode14_arms.json` for (d)/(e) | exp_35 (rungs a–c) + exp_34 (rung d) | **PARTIAL — 4 of 5 rungs have a number, and rung (d)'s is not commensurable with the rest.** **The count is back to five**: exp_34 measured the drain deletion at **+3.2 µs — a null** — so the six-rung split collapses, the drain deletion is reported as free, and coarse readiness is one rung again. Plottable, all at rev-26 pins: (a) `pf6gm_mega` 6,900.2 sd 15.5 / 0.8950; (b) payload, throttle OFF `g=65` 7,110.8 sd 13.7 / 0.9226; (c) + injection bound depth 4 `g=353` 6,495.8 sd 6.9 / 0.8422. **(d) coarse readiness (mode 14, `C=0`) = 6,650.9 µs (n=4) — a FALSIFICATION of the pre-registered 5,990–6,440 band — but measured at pin `291dfa08`, where mode 12 itself regressed to 7,224.2 from 6,497.3 and the injection bound is inert.** So the (c)→(d) step reads −576.5 µs against a baseline no other rung shares, and against the *true* in-session ratchet the same arm is **+150.4 µs**. **Do not draw (d) on the same axis as (a)–(c); re-run it at the healthy pin — the regression fix landed at `275d2c2a` (exp_38), so the re-run is unblocked and needs ~30 min of GPU with `-DK0P6_MPS_ENABLE_MODE14=1` plus a default-build in-session control.** (e) nc-major task order, `status: not_built`, exists in no commit. |
| §6 **Q2** — do dedicated communication blocks ever win? | Fig 5 | paired best-vs-best (mode 2 pool at C=64 g=1 vs mode 12 at C=16 vs mode 14 at C ∈ {0, 8, 16}), plus the pool-size sweep at the final design (predicted flat→degenerate) and the reducer-count sweep | **`exp_37_placement/placement.json`** (schema `exp_37.placement.v1`) for the mode-12 axis; **`exp_34_mode14/mode14_arms.json`** for the mode-14 C ∈ {0, 8, 16} series | exp_37 (27 campaigns / 135 rotations at `b5215081`) + exp_34 (mode-14 series, C=0 point) | **DATA LANDED.** **Dedication never won — not once, nowhere on the reachable axis, monotone in pool size.** mode 12 at rev 26: `C=4` **6,464.2** (n=7) ≈ `C=8` **6,472.0** (n=7) < `C=12` 6,490.4 < `C=16` **6,498.0** (n=7) < `C=32` 6,735.3 < **mode 2 `C=64` 6,837.5** — the dedicated-pool design at its own best point is the worst arm, 374 µs of span. Paired by interleave round, n=7: **`C=8` −27.2 [−32.3, −22.1] t=−13.0; `C=4` −34.2 [−39.5, −28.9] t=−15.9**, all 7 rounds negative for both, value sets disjoint, exact rank-sum p = 0.00058. **`C=4` vs `C=8` is a TIE** (−7.0, CI [−15.5, +1.5]) — plot the winning *region* `C ≤ 8`, never `C=4` as the winner. **exp_34 supplies the C=0 point** that mode 12's validator makes unreachable, and the mode-14 series is **monotone, not flat**: C=0 6,650.9 → C=8 6,725.6 → C=16 6,780.4, i.e. **8.1–9.3 µs per reserved CTA with the pool provably jobless** (bit 26 never set across 4,032 `pperr` readings, `DRAIN=0`, `[MPS SPIN] 0/0`) — **pure capacity loss, contention excluded by construction, which is stronger evidence than the predicted flatness would have been.** Mechanism for the caption: **M6 does not move** (paired −1.0 µs CI [−6.7, +4.8], 0.07 % spread across C=4…32); the effect is entirely in the payload-carrying phases (M7+combine −41.9 and −31.1), **not** in M7 alone — M7 and combine move in opposite directions because their boundary shifts with C. Caption caveat: the mode-14 series sits at `291dfa08` and may be compared within itself but not against mode-12 numbers from other pins. |
| §6 **Q3** — where does the time go now? | — | phase attribution at the ratchet: stacked bar over plan M3–M5 / M6 GEMM-1 / M7 GEMM-2 (GEMM proper vs epilogue surcharge) / combine M8–M9, for `mps_mega`, plus the `production` three-stage split | `exp_33_attribution/phase_stamps.json` (schema `exp33-phase-stamps-1`; `arms.mps_mega.phases.stacked_bar_phases` is the partition to plot) | exp_33 | **DATA LANDED.** n=10, all gates green. M7 2,701.8 ±23.74 (46.2 % of interior) > M6 2,453.3 ±2.51 (41.9 %); the two GEMMs are 88.1 %. plan 372.8 ±0.67, combine 324.2 ±21.12, interior 5,852.1 ±9.56. |
| §6 **Q4a** — resource saturation vs CTA count (NanoFlow Fig-7 analog) | Fig 2 | per-resource throughput vs CTA count (MFMA TFLOPS / HBM GB/s / xGMI GB/s), each curve **isolated** and **concurrent**, knee annotated; the gap between the two curve families *is* Finding 1 as a figure | **`exp_22_fig7_saturation/saturation.json`** (schema `exp22-saturation-1`; raw `saturation_plan.jsonl`) | exp_22, `E22_SRC_REV 10`, 250/250 plan points, 2026-08-12T10:45Z | **DATA LANDED.** Knees: single xGMI link **8 CTAs** (plateau 56.9 GB/s = 74.1 % of 76.8); 7-link fabric **32 CTAs** at mlp4 (355.0 GB/s = 66.0 % of 537.6); HBM **128 CTAs** (4,151.9 GB/s = 51.9 %); **MFMA has no knee** — linear to 256 CTAs, R² ≥ 0.99989. Communication saturates at 8–32 CTAs against 256 CTAs of compute capacity. H4 SUPPORTED ×3; H3 `REFUTED_PAYLOAD_RIDES_FREE` (concurrent/isolated median **0.9951**, n=42, vs **0.878** with the protocol compiled in and −42 % at g=16); H1 falsifier fired, reported unsoftened. |
| §6 **Q4b** — per-layer utilization timelines (NanoFlow-v2 Fig-10 analog) | Fig 3 | per-CTA phase occupancy over time for three arms on one routing seed | `exp_23_fig10_timeline/rank0_events.json` (schema `exp23-events-1`) + `timeline_bins.csv` (`exp23-bins-1`) — **not collected** | exp_23 (Tier A built at `d13acacf`, `SRC_REV 29`) | **SPEC READY — instrument built, no data, and its parity gate is now DEMOTED.** Tier A is applied and the CPU parity gate is green: ring compiled in but runtime-off equals the arm on all eight tuple columns (SGPR 106 / VGPR 256 / AGPR 256 / scratch 128 / LDS 155,496 / occupancy **asserted** 1 / spills 217-17), MFMA 180 with 0 scratch ops per span, `pk_add_bf16` 282, while `.text` differs (192,640 vs 192,448 B) so the instrument is genuinely present. G7 green both halves (`ts_mark` 5 / `ts_last` 0 / `e23_mark` 0; ISA `s_memrealtime` 12 == 12). A fourth attribution build localises the whole ring-on cost (+16 B scratch, +4 VGPR spills) to `timestamps=1`, **not** the ring. **But `291dfa08` passed that identical tuple gate while costing 726.9 µs, so a green tuple no longer licenses "free": this ring owes an END-TO-END TIMING CONTROL against rev 26 (or `.text` byte-identity) before any timeline drawn with it may be published.** Host patch `e23_ab.patch` is `git apply -p1 --check` clean and deliberately parked so it cannot become a second variable in another agent's arm. **Two plan corrections:** the homogeneous panel must be `mps_mega C=0,mode=0` labelled "bulk, no overlap" — `pf6gm_mega` cannot be instrumented (55-word descriptor, no MPS state slot); and the amd-smi bandwidth cross-check is replaced by a UMC duty-cycle check ±15 pp plus a 52.8 GB/s fabric-ceiling bound, because this node exposes no live xGMI throughput counter. |
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
   plotting a compiler artifact as a mechanism. **exp_38 fixed the step at
   `275d2c2a` and the rule survives it**: the broken window is `291dfa08`
   inclusive to `275d2c2a` exclusive, and a `-DK0P6_MPS_ENABLE_MODE14=1` build
   still carries the +727 µs by design, so the pin **and the `-D` set** belong in
   every series' metadata.
14. **`servicedrain` is not a number for mode 14 and must never be plotted as
   one.** No CTA enters the drain, so `K0P6_MPS_TS_DRAIN` is never stored and the
   cell reads as its never-written sentinel (`DRAIN=0`, and the derived delta
   comes out as a large negative). `mode14_arms.json` carries
   `servicedrain_status` for exactly this reason; the *absence* is the evidence,
   and it belongs in the caption as text, not on an axis.

## Highest-value unblocking actions, in order

~~**0. Fix the `291dfa08` mode-12 regression.**~~ **DONE — exp_38, `275d2c2a`.**
`.text` byte-identity with rev 26, ratchet restored to +0.09 %, injection contrast
revived to −618.7 µs. It was a codegen/allocation effect exactly as diagnosed: four
of mode 14's nine sites spill inside the M7 epilogue, and each spill reload drags an
`s_waitcnt vmcnt(0)` full drain (21 → 117), collapsing the atomic issue run from a
mean of 23.5 in flight to 1.45 — so `vmcnt(4)` was capping something that never
exceeded 1.

1. **Re-run Q1 rung (d) at the healthy pin** (~30 min of GPU, **now unblocked**).
   Build with `-DK0P6_MPS_ENABLE_MODE14=1` and pair it in-session against a
   default-build `C=16,g=353,mode=12` control, because a mode-14 binary carries the
   +727 µs and **no mode-12 number may ever be published from it**. That is the only
   thing standing between the waterfall and 4 commensurable rungs, and it is also
   the only way to settle whether the injection bound and the coarse signal are
   complements or substitutes. Rung (e) remains unbuilt in any commit.
2. **A timing arm for exp_23's ring** before Q4b is drawn. exp_38 **exonerates** the
   ring as the cause of the regression (it predates the ring, and a ring-compiled-in
   build matches rev 26 on every epilogue-window metric), but the ring **cannot be
   `.text`-identical by construction**, so `K0P6_MPS_E23_RING=1` is a second arm and
   needs its own campaign — not a tuple gate, which is the gate `291dfa08` passed.
3. **One-line harness edit: add `K0_PF6GM_DECOMP` to `run_campaign.sh`'s `-e`
   forwarding list.** It converts every M7 surcharge proxy into a same-run
   measurement and it is the single highest-value change outstanding for the Q1
   waterfall and the Q3 stacked bar.
4. **Then Q4b's collection** — the parked host patch, the ring's timing arm,
   ~45–55 min of GPU.
5. **Fix the T = 1024 / 2048 correctness defect** — it is what stands between Q5
   and a real batch/seqlen axis, and it is a defect in both megakernels, not a
   figure problem.

## One statistical correction to carry into every caption

**"`[MPS SPIN]` 0/0 everywhere" is now "0/0 in 120 of 135 rotations".** exp_37
found 15 of 135 (11 %) reporting `success_max = 1` — a single *successful* poll —
spread across every arm and not tracking any effect, with **`fail_max = 0` in all
135**. Peer wait remains zero for practical purposes and no poll ever exhausted,
but the absolute claim in earlier sections is too strong and captions should say
"≤ 1 poll ever" rather than "zero".

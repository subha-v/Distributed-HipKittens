# OVERLAP_ATLAS — every comm/compute overlap opportunity, across every regime

**Written:** 2026-08-18, laptop, no node access. **Consumer:** the ablation-campaign
master doc (`../BRIEF.md`), specifically outputs (2) schedule state vector S,
(5) experiment ladder and (6) per-cell kernel predictions.

**Why this document exists.** The industry experts' summary of our position was that
we have exploited *one* overlap region — the MoE all-to-all under the expert GEMMs —
and that it is unclear how the schedule should change with the workload. The first
half of that is true and this document is the corrective enumeration: **every place
in every regime where a byte on the wire (or on HBM) could be moved under a FLOP**,
sized where the repo has a number, screened against the laws we have already
banked, and ranked.

**Rules of evidence, binding on every line below.**

* Numbers appear only with a source: a repo file (`path:line`), a git sha, a results
  artifact, or the BRIEF's draft8 digest.
* `MEASURED` / `DERIVED` (arithmetic shown) / `ASSUMED` / `SPECULATION` tags are used
  the same way `../grounding/TP8_STATE.md` uses them. Nothing is quoted without one.
* Where an idea is mine and unmeasured it says **SPECULATION (mine)** and carries a
  named falsifier.
* Retracted strings from `../grounding/BANKED_LAWS.md` §7 and
  `nightshift/REPORT.md:97-143` are not re-used. In particular: **the "−32% fewer
  bytes" CDAR claim is dead** (`../grounding/TP8_STATE.md` §1.3, ≈ −9.7%), and no
  serving number is quoted without its workload tuple.

---

## 0. How to read this atlas

### 0.1 The record schema

Every opportunity carries, in order: **the comm being hidden** · **the compute hiding
it** · **where the slack is, sized** · **which W cells it applies to** · **screens**
(pass/fail against the banked laws) · **implementation sketch** (skeleton + knobs from
`../grounding/KNOB_INVENTORY.md`) · **EV rank**.

### 0.2 W-cell notation

`TOPO ∈ {DP/EP, TP8+EP}` · `PHASE ∈ {prefill, decode}` · `CONC` (serving concurrency)
· `T` (tokens/rank/step) · `FILL` (real ÷ executed rows) · `SKEW` (max-rank load) ·
`ARCH ∈ {gfx950, gfx942}`. Where a cell is quoted it carries the tuple that produced
its number.

### 0.3 EV rank, defined so it is falsifiable

| rank | meaning |
|---|---|
| **HIGH** | ≥3% of the cell's wall, the slack pool is measured **and larger than the thing being hidden**, all screens pass, and it is reachable from an existing skeleton by knob turns or one seam |
| **MED** | 1–3%, or ≥3% but the slack pool is DERIVED/unmeasured, or it needs one new mechanism that has a precedent in-repo |
| **LOW** | <1%, or it fails a screen, or it needs machinery nobody has built (rollback, work migration, a new residency model) |

Rank is **expected value**, not probability of working. A HIGH item can still fail;
what makes it HIGH is that the arithmetic says the prize is worth the node hours.

---

## 1. The feasibility screens (apply all eight to every proposal)

These are the banked laws restated as *rejection tests*. An opportunity that fails a
screen is not forbidden — it is predicted ≈0 or negative, and its arm must be
pre-registered as a falsification attempt, not as a ratchet candidate.

**SCREEN-0 — Name the slack, and show it exceeds the comm.**
The G25-1 lesson: the rig hid ~2.1% of its collective against a ≥90% PASS bar
(`tp8_mega/results/G25_1_STATUS.md`, commit `3f7f0f3a`) at an operating point where
the collective was **~3× LONGER than the compute it had to hide under** (compute floor
1,439 µs for two bursts vs 1,880–2,214 µs of CDAR for one boundary,
`../grounding/TP8_STATE.md` §2.4). Best conceivable fused wall there is
`max(compute, transport)` ≈ 1,880 µs, i.e. **perfect hiding was Amdahl-capped near
zero**. Any proposal that does not state `slack_CTA·µs ≥ comm_CTA·µs` is untestable
before it is wrong.

**SCREEN-1 — The four doors (occupancy-1 selection rule).**
`docs/distributed/OVERLAP_ABSTRACTIONS.md:42-60` + LAW-12: 256-VGPR/256-AGPR bodies
and 155,428 B of LDS force **1 block/CU, 1 wave/SIMD**, so there are no spare waves
to fill stalls and phase partitioning is *work-conserving* (exp_25's static role
split measured **−173 µs**, a loss). Overlap pays only through four doors:
**(a)** move a *latency-bound* consumer into a compute shadow (M15 combine: −144 µs
of residue); **(b)** *delete* protocol work (926k per-row RMWs → 2×8 slab words);
**(c)** *swap the op class* on the fabric (RMW → posted store); **(d)** *bound
injection* (depth 4: −615 µs). "It fills stalls with other work" is door (e) and
**door (e) does not exist at occupancy 1**.

> **Corollary I am adding, and it is load-bearing for the decode cells (DERIVED).**
> LAW-12's own scope note says the axiom "breaks the moment LDS or register pressure
> drops — shrinking the ubench's LDS let occupancy rise 1 → 2 blocks/CU"
> (`aug11/LESSONS.md:99-109`). Decode needs **small-M tile variants**
> (`TP8_OVERLAP_ANALYSIS.md:80`, `M25_TP8_MEGA_DESIGN.md:219-221`), which are exactly
> the bodies that would drop below the 256-VGPR ceiling. ⇒ **the four-door rule is a
> PREFILL law. At decode, door (e) may re-open, and with it every "fill the stall"
> design the prefill ledger correctly rejects.** This is the single largest structural
> difference between the prefill and decode halves of this atlas and it has never
> been measured (`../grounding/BANKED_LAWS.md` §8 hole 4: "decode has never been
> measured at all in any of our rigs").

**SCREEN-2 — Critical rank.**
LAW-52 (`aug18/V6_EVIDENCE_AND_DESIGN.md:115-118`): iteration pace = the critical
rank's wall clock; **only work removed from THAT rank's wall moves e2e**. In-window
filler qualifies; **cross-rank wait absorption and host-side reshuffling do not**.
In serving the hot *expert* rank plays the role of training's hot *hardware* rank
(LAW-46: 51.9% of expert-slot traffic on rank 0 under contiguous placement).

**SCREEN-3 — Injection depth is a co-residency knob, not a fabric knob.**
LAW-20/21 are one **joint** cell: carrier relocation without the bound is
**+210.6 µs** (7,110.8 vs homogeneous 6,900.2); with depth 4 it is **6,495.8**
(−615.0 µs, t = −80.2). LAW-8/F6: depth is **flat {4,8,16,32} in pure transport**
(towers 93.6/93.4/93.3 GB/s, atomics 67.1/67.1/67.0 —
`tp8_mega/results/g25_0b_v1_results.txt`). ⇒ any new co-resident carrier must
**re-fit `d` in its own cell**; any pure-transport arm must not claim the −615 µs.

**SCREEN-4 — Residency / short-kernel side-stream.**
LAW-54: a 256-CTA all-CU megakernel creates a residency contract; "slice convoys
block mega residency where rocBLAS's sub-ms kernels slip through"
(`aug18/V6_EVIDENCE_AND_DESIGN.md:64-70`). **Only sub-millisecond kernels qualify for
a side stream.** Any "run it on another stream" proposal fails this screen by default.

**SCREEN-5 — Filler granularity: quota, never fence; the monolithic-slice
falsification.**
LAW-53's ledger: in-launch bubble supply **~150–180 CTA-ms** vs in-situ wgrad demand
**~3,360 CTA-ms**; every rescheduling variant lost — control 1,670/1,689 ms; pure
in-kernel filler 1,892; **partition + monolithic 128-CTA slice + fence 1,959–1,962**;
dribbled slices + fence 1,988–1,993; no-fence 1,927–1,954
(`aug18/V6_EVIDENCE_AND_DESIGN.md:30-49`). And SHEXP's arithmetic: **one filler task
is ~230 µs against a 642 µs window**, so "yield when real work arrives" is
arithmetically impossible and the discipline must be **quota**
(`SHARED_EXPERT_FILLER_DESIGN.md:354-383`).

**SCREEN-6 — Readiness must change when work can EXECUTE.**
LAW-23/24: under natural GEMM-2 order the median token is not reducible until
**91.7% of GEMM-2 completes** (`../BRIEF.md:59-61`); per-record flags gave the vendor
**no benefit** and an LL128-style per-record path was **44%/177% slower**
(`aug11/OVERLAP_METHODOLOGY_STUDY.md:338-351`). Signal the smallest early-**executable**
unit. Order (K2) is what makes "early-executable" exist; alone it is ≈0, with
certification it is −144 µs.

**SCREEN-7 — The instrument is part of the kernel.**
LAW-32/58/59: merely making mode-14 code **reachable** cost the mode-12 ratchet
**+726.9 µs** with mode-12 source untouched (`moe_mps_adapter.cuh:39-56`);
`.text` sha identity is the only parity gate we trust; the issue-run gate is
**282 atomics / ~12 runs / mean ~23.5**. Every arm in this atlas must pin
`K0P6_MPS_ENABLE_{MODE14,TBO}`, `K0P6_M15_SRC_REV` and the kernel-name.

**SCREEN-8 — Measurability.**
LAW-45 / `../grounding/WORKLOAD_EVIDENCE.md` §W10: e2e MDE is **±15% day / ±18%
position at c32p** but **≤1.8% throughput / ≤0.25% latency at c512p**. A predicted
−2% effect is unmeasurable at c32p and measurable at c512p. Boundary rigs are the
precision instrument; e2e is the validation instrument. SHEXP already pre-registers
this honestly: its −1.6…−4.6% e2e prediction "is **not measurable in serving at
n=1**" (`SHARED_EXPERT_FILLER_DESIGN.md:698-704`).

---

## 2. The slack ledger — every pool this project has sized

This table is the atlas's load-bearing asset: an overlap proposal is only as good as
the slack it targets. Blank sizes are named holes, not zeroes.

| # | slack pool | regime (W tuple) | size | class | source |
|---|---|---|---|---|---|
| **P1** | M0→M2 dispatch window | DP/EP prefill, T=4096, balanced | **642 µs** wall; **~102,720 CTA·µs** usable at F=160 | MEASURED (wall) / DERIVED (CTA·µs) | `aug12/OVERLAP_KERNEL_DESIGN_IDEAS.md:22-27`; `SHARED_EXPERT_FILLER_DESIGN.md:390-402` |
| **P2** | M2 chunk_ready poll | DP/EP prefill, **std=0** | **~0** (`[MPS SPIN] 0/0`, max-spins-to-success 0 vs a 2,000,000 limit) — grows with SKEW | MEASURED | LAW-31; `aug10/KERNEL_BOTTLENECKS.md:159-163` |
| **P3** | M6 K-loop stall pool | DP/EP prefill | **~84% of M6 = 2,453.3 µs**; MFMA duty 17.1% / 13.7% | MEASURED | `aug12/OVERLAP_KERNEL_DESIGN_IDEAS.md` L6 |
| **P4** | M7 fabric headroom | DP/EP prefill | epilogue moves 392 MiB xGMI in 2,701.8 µs = **152 GB/s ≈ 43% of the 349–355 GB/s egress card** | DERIVED (arithmetic below) | volumes at `aug12/…IDEAS.md:29-31`; ceiling `OVERLAP_ABSTRACTIONS.md:147` (**flagged unreproduced**) |
| **P5** | M8 combine as consumable prefix | DP/EP prefill | realized **−93 µs (C 8→16), −444 µs (C 16→24)**; residue **−144 µs** with order | MEASURED | `OVERLAP_ABSTRACTIONS.md:26-30, :80` |
| **P6** | dispatch+plan(i) ↔ GEMMs(i) | DP/EP prefill | **~1,015 µs** — but **structurally blocked** by the sort's global histogram dependency | MEASURED (size) / ANALYTIC (block) | `aug12/…IDEAS.md:33-47`, §5 |
| **P7** | epoch i ↔ epoch i±1 | DP/EP prefill | **~1,000 µs dispatch+plan + 324 µs combine** — "the biggest untouched axis"; the one realization tried (mode 16) **lost +75.8 µs** | MEASURED | `aug12/…IDEAS.md:33-47`; `OVERLAP_ABSTRACTIONS.md:52-53` |
| **P8** | training in-launch bubble supply | training bwd, 8 GPU | slab-0 idle **1.61 ms × 28 CTAs**; slab-1 post-quota **1.23–1.36 ms × 28**; M8 cert wait **36–177 µs × 256**; M0 barrier **~250 µs × 256** ⇒ **~150–180 CTA-ms** | MEASURED | `aug18/V6_EVIDENCE_AND_DESIGN.md:41-49` |
| **P9** | TP8 exposed ring-AR (the prize pool) | TP8+EP prefill, 16,384 tok/step | **~2.8 ms/layer of ~10.4 ms ≈ 27%** ⇒ **171 ms of a 634 ms step**; perfect hiding = 1.37× ⇒ 25.8k → **~35.4k input tok/s** | DERIVED, two routes agree; **never profiler-confirmed** | `../grounding/TP8_STATE.md` §1.4; commit `108def6e` |
| **P10** | TP8 compute available per layer | TP8+EP prefill | **7.2–7.6 ms/layer** vs 2.8 ms collective ⇒ **compute:transport ≈ 2.6–3.2 : 1** | DERIVED | same |
| **P11** | TP8 fabric duty required | TP8+EP prefill | **28–39%** (see §2.1) — the fabric is idle ~2/3 of every layer | DERIVED (mine) | §2.1 |
| **P12** | G25-1 rig slack | the rig as run | **NEGATIVE**: compute 1,439 µs vs transport 1,880–2,214 µs | DERIVED | `../grounding/TP8_STATE.md` §2.4 |
| **P13** | DP prefill co-scheduling | serving c32p, ISL 4096, OSL 8, closed-loop | **15–25% of node throughput**, kernel-independent; 3.19 vs 3.93 chunk-events per in-bucket step of 8 possible | ANALYTIC from measured step counts | LAW-50; `aug18-prefill/BOTTLENECK_SIGNALS.md:307-311, 483-496` |
| **P14** | padded MoE rows | serving c32p m15 arm | **62.4%** of every MoE row computed was padding ⇒ **≈65 s of a 217.6 s run** | MEASURED | LAW-42; `BOTTLENECK_SIGNALS.md:38-44, 264-271` |
| **P15** | dummy steps | serving c32p m15 arm | **56.0%** of in-bucket steps are 100% fake work (139.62/249.12) | MEASURED (receipt) | `BOTTLENECK_SIGNALS.md:341-343` |
| **P16** | shared-expert FLOPs available as filler | any, per MoE layer | **1/8 of routed** ⇒ **≈550–645 µs of grid-wide work** at the mega's measured throughput | DERIVED from measured M6/M7 | `SHARED_EXPERT_FILLER_DESIGN.md:636-651` |
| **P17** | serialized shared expert we would delete | DP/EP prefill serving | **S_out ≈ 450–700 µs per MoE layer**, fully in front of the mega, plus 4 launches + an add kernel | DERIVED | `SHARED_EXPERT_FILLER_DESIGN.md:671-683` |

### 2.1 The most important new arithmetic in this document (DERIVED, mine)

`../grounding/TP8_STATE.md` §1.3 establishes **411 MB of per-rank egress per 235 MB
boundary all-reduce** (two-shot RS `7/8·235` + AG `7·235/8`, identical to ring's
`2(N−1)/N·B`). Two boundaries per layer ⇒ **822 MB per rank per layer**. The layer is
**10.4 ms** (§1.4, DERIVED and cross-checked against `nightshift/LOG.md` 12:10 PDT).

```
required continuous egress = 822 MB / 10.4 ms = 79.0 GB/s
```

Against the measured achievable rates:

| transport | achieved per-rank egress rate | fabric-busy ms per layer | **duty cycle needed** |
|---|---:|---:|---:|
| CDAR store-towers, best (256-row slabs) | 205 GB/s | 4.01 | **38.6%** |
| CDAR inside the boundary rig | 186 GB/s | 4.42 | 42.5% |
| **RCCL in the rig** | **280 GB/s** | **2.94** | **28.2%** |
| the (unreproduced) egress ceiling | 349 GB/s | 2.36 | 22.6% |

Three consequences, and they reorder the M25 program:

1. **The fabric only needs a ~28–39% duty cycle to carry the entire TP8 layer's
   traffic.** Hiding the collective is therefore **not bandwidth-limited — it is
   scheduling-limited.** Even our own 1.5×-slow transport has 2.6× more capacity than
   the layer requires, *if the traffic is spread rather than bursted*.
2. **Cross-check that validates the prize pool.** RCCL's fabric-busy time is
   **2.94 ms/layer**, and the banked *exposed* ring-AR is **~2.8 ms/layer**. Those are
   the same number. ⇒ **production hides essentially none of its collective**; the
   27% pool is real and it is not an artifact of a bad estimate.
3. **It re-interprets K4.** If the binding problem is duty cycle rather than
   congestion, the injection depth knob is a **metering** control (stretch the
   transport across the compute), not only a congestion control — and its optimum at
   the deployment compute:transport ratio should be **lower** (more throttled) than
   the depth-4 that was tuned where the carrier had to finish inside one epilogue.
   **Pre-registered falsifier:** re-sweep depth {2,4,8,16,32} in the fused arm after
   the rig is put at the deployment ratio (TP8_STATE M2, one argv). If the optimum is
   still 4 or moves *up*, the metering interpretation is wrong. (Note depth 2 has
   never been measured — the `g` `0x300` selector has no room, so this needs one new
   specialization, `credit.cuh:63-82` already ships depths {0,4,8,16,32}.)

The DP/EP side of the same arithmetic, for symmetry (DERIVED from the volume anchors
at `aug12/…IDEAS.md:29-31`):

| DP/EP prefill phase | bytes | span | rate | duty vs 349 GB/s |
|---|---:|---:|---:|---:|
| M1 dispatch push | 162 MiB/rank | 642 µs | **264.6 GB/s** | **~76%** |
| M7 epilogue remote RMW (xGMI part) | 392 MiB/rank | 2,701.8 µs | 152.1 GB/s | ~43% |
| M7 epilogue amortised over the epoch | 392 MiB/rank | 5,822 µs | 70.6 GB/s | ~20% |

The 76% figure independently reproduces SHEXP's "our 642 µs is ~77% of that ceiling"
using MORI's 342 GB/s dispatch number (`SHARED_EXPERT_FILLER_DESIGN.md:390-394`).
⇒ **dispatch is near fabric-saturated (so its CU capacity is genuinely spare — the
SHEXP window hypothesis has a second, independent supporting arithmetic), while the
combine epilogue runs at ~43% duty (so M7's problem is congestion/layout/protocol,
not bandwidth — exactly what LAW-16 and LAW-3 say).**

---

## 3. Regime map — which comm exists at all, per cell

| cell | per-layer comm that exists | species | dominant cost | metric that decides |
|---|---|---|---|---|
| **DP/EP prefill** (c512p, φ≈0.985) | MoE all2all dispatch + combine, MoE layers only | sparse, data-dependent | expert GEMMs (M6+M7 = 88.1% of the interior) | input tok/s; TPOT p50 (we win 4.24×) |
| **DP/EP prefill** (c32p, φ=0.376) | same, but ~56% of steps carry **zero real tokens** | sparse | **padding and DP co-scheduling**, not comm | TTFT p99 / e2e p99 (we win 23–33%) |
| **TP8+EP prefill** (c32p) | **2 dense all-reduces on the critical path of every layer, incl. the 3 dense layers** — 122 collectives per token-step | dense, structural | exposed collective ≈27% + GEMMs | input tok/s, TTFT p50 |
| **DP/EP decode** | all2all with tiny payloads — **but today the mega executes it at prefill shape** (B4096 rescue) | sparse, latency-bound | **padding, by ~500×** (§6.1) | TPOT |
| **TP8+EP decode** | 2 tiny dense ARs/layer | dense, **latency-bound** | expert-weight streaming from HBM (§6.2) | TPOT / TTFT |
| **training fwd+bwd** | all2all fwd + bwd, plus DP gradient collectives | sparse + dense | self-inflicted FLOPs (30.5% more MoE FLOPs, LAW-57) | median of max-across-ranks iteration latency |

Sources: `TP8_OVERLAP_ANALYSIS.md:16-42, 76-87`; `../grounding/TP8_STATE.md` §1.2,
§1.5, §4.3; `../grounding/WORKLOAD_EVIDENCE.md` §W8.3, §3.2; LAW-65 (phase profile).

**The framing point that must survive into the master doc**
(`../grounding/TP8_STATE.md` §3.3): our high-concurrency **+8.40%** win is mediated by
**fill**, and fill is a *DP-topology* quantity — it does not exist under TP8 even in
principle. The TP8 cell must be won by a different mediating quantity: **exposed
collective fraction**. The campaign needs two laws, not one.

---

## 4. Family A — Intra-MoE (DP8/EP8): what is exploited, and what is left

### A1. Combine-under-GEMM2 — **EXPLOITED, this is the ratchet**

*Comm hidden:* the MoE combine (392 MiB/rank/epoch of remote packed-bf16 RMW over
xGMI). *Compute hiding it:* GEMM-2 (M7) itself, via the producer epilogue.
*Slack:* P4 + P5. *Realized:* production 7,712 µs → M15 C=28 **5,822.0 µs = 0.7544×**
(`../BRIEF.md:49-51`). *Mechanism split:* door (d) depth-4 bound (−615 µs), door (b)
protocol deletion (926k RMWs → 2×8 slab words), door (a) consuming pool
(−93/−444 µs).
**Residual room, and it is small:** exp_25 caps burst de-alignment's value at
**+211 µs** and measured M6→M7 availability ≡ consumption (2.133 = 2.133, **zero
slack**) (`aug12/…IDEAS.md` L4). **EV: exploited; residual LOW.**

### A2. Dispatch-under-GEMM1, same epoch — **STRUCTURALLY ABSENT, not merely unbuilt**

*Slack on the table:* P6, **~1,015 µs**. *Why it is blocked:* M6 needs
`sti/sei/tile_desc` from the sort; the sort's per-expert extents need **every**
source's histogram bins, and a source's chunk carries rows for all 32 local experts,
so **no partial-arrival prefix completes any expert**
(`aug12/OVERLAP_KERNEL_DESIGN_IDEAS.md:107-118`). M1↔M2 *is* already the overlap
(zero peer wait, LAW-31); the 642 µs residual is own work.
*Screens:* fails SCREEN-6 (no dependency-structure change makes an earlier
executable unit exist).
**EV: LOW as an overlap target — but this is a *result*, not a gap.** It is one of
the campaign's cleanest "why the regime boundary sits there" statements and should be
reported as such.

### A3. Cross-epoch pipelining (epoch i ↔ i±1) — **the biggest untouched axis, one arm tried and lost**

*Comm hidden:* epoch i's dispatch+plan under epoch i−1's GEMMs. *Slack:* P7,
**~1,000 µs + 324 µs**. *Tried:* mode 16 / TBO-2 deferred combine —
**falsified, 6,559.6 µs = 0.8513×, +75.8 µs vs the ratchet**; diagnosis: "the wait it
escaped was mostly not on the critical path, and the parity state it added was"
(`OVERLAP_ABSTRACTIONS.md:52-53`).
*Screens:* SCREEN-1 door — the deferred-combine form went through **no** door, which
is exactly why it measured ≈0-and-worse. A *dispatch*-side cross-epoch arm would go
through door (a) (the dispatch consumer is fabric-latency-bound and would land in the
prior epoch's compute shadow) — a different door, so the mode-16 falsification does
**not** transfer to it.
*Sketch:* skeleton S5 with the deferral moved from combine to **dispatch**; needs
parity buffers (descriptor slot 63 is contested — `#error`s at
`k0pf6gm_device_tile_m15.hip:103-105,159-165`) and the retire/epoch machinery.
**EV: MED.** Large pool, one arm's worth of evidence against a *different* form,
expensive (parity state is what killed the last attempt).

### A4. **C\*(W) — the consuming pool's size as a function of the workload** — the single most under-exploited knob we already own

*This is not a new overlap; it is the same overlap, mis-tuned everywhere except one
point.* `C` is the steepest and most non-monotonic knob in the inventory
(16 → 6,292.4; 24 → 5,848.5; 28 → **5,822.0**; 32 → unstable, hung 1 of 2 runs) and
**its optimum already flipped across designs** (mode 12 wanted C ≤ 8; M15 wants
24–28). The kernel itself documents that its tuned point is invalid elsewhere:
*"at `T_eff` … C / flush_rows are no longer at their tuned point"*
(`k0pf6gm_device_tile_m15.hip:2122-2127`).
*Slack:* P5. *W cells:* all of `T × FILL × CONC × SKEW`. *Screens:* all pass — it is
door (a) and it is a runtime byte of `K0P6_D_MPS_CFG`, so one HSACO serves the whole
sweep (`BENCHMARKING.md:174-186`).
*Sketch:* S4, sweep `C ∈ {8,16,24,28,32}` × `flush_rows ∈ {0,8,16,32}` inside **every**
W-cell; `flush_rows = 0` is a free control that isolates pure protocol deletion
(`aug12/M15_DESIGN.md:87-88`).
**EV: HIGH.** It is the direct, measured answer to the expert question *"does our M15
producer/consumer work best for all sizes?"*, it costs no code, and the law it
establishes — *C\* is set by the ratio of latency-bound consumer work to producer
shadow, and is a derived quantity of the cost model, never an input* — is exactly the
"if X then Y because Q crossed θ" shape the BRIEF demands.

### A5. Plan phase (M3–M5) — single-CTA scan while 255 CTAs idle

372.8 µs total with **three grid barriers inside**; the genuine same-epoch residual
is "worth ≤ tens of µs, in the noise" (`aug12/…IDEAS.md:107-118`). Fails SCREEN-5
(230 µs task granularity vs a window fragmented by three barriers —
`SHARED_EXPERT_FILLER_DESIGN.md:660`). **EV: LOW.**

### A6. Burst de-alignment (K4-prime, rotated-nc phase)

`nc' = (nc + gid·φ) mod 16` to de-align epilogue bursts across the 240 CTAs that
currently march in lockstep at 78.3% of egress ceiling. exp_25 caps the value at
**+211 µs**; unimplemented (`aug12/…IDEAS.md:124-127`). Passes SCREEN-1 door (d)-ish
(it is congestion shaping). **EV: LOW-MED** — cheap (one index-map change,
`order.cuh` already hosts the map family), honest to lose, and it prices the
de-alignment term for the cost model.

### A7. Per-destination credits (K6) — the skew-rescue arm

Today's bound is a **global** per-wave `vmcnt(4)`: destinations are indistinguishable.
Under skew the scarce resource concentrates per-owner
(`aug12/…IDEAS.md:128-131`). `credit.cuh:104-106` marks this a **documented non-goal
until a skew harness exists**, because m17's order-based rotation buys the spreading
with no protocol. *Screens:* SCREEN-3 (it is a depth refinement — must re-fit `d`);
predetermined falsifier already written: **must tie mode-12 within 1% at std=0**.
*Blocked by:* LAW-62(b) — the skew axis **has never been drivable** on the kernel
harness (`K0_SYNTH_ROUTE` rejected; reachable CV 0.0034–0.0086 vs COMET's 0.032).
**EV: MED, gated.** Highest-value *unbuilt* knob in the inventory, and it cannot be
pre-registered until the skew axis is shown settable (the route-replay rig,
`ROUTE_REPLAY_PLAN.md`, is code-complete and **has never been executed on GPU**).

### A8. **Wave-specialized transport (K1-w)** — the granularity nobody in this repo has tried

**SPECULATION (mine), with a mechanism grounded in three banked measurements.**

*The observation.* Every role discussion in this project is at **CTA** granularity —
`roles.cuh`, `is_service_cta`, the carry/idle/consume trichotomy. But
`__launch_bounds__(256,1)` means **256 threads = 4 waves of 64**, and LAW-12's
"1 wave/SIMD" over 4 SIMDs means those 4 waves are the CU's entire occupancy. The
occupancy-1 axiom's binding consequence is stated precisely as: *"`vmcnt` is one
in-order counter **per wave**, so a wave cannot overlap its own remote traffic with
its own loads"* (`OVERLAP_ABSTRACTIONS.md:42-47`). **A different wave has a different
`vmcnt`.**

*The proposal.* Split the CTA, not the grid: waves 0–2 run MFMA, wave 3 runs the
transport (ERS commits / MAG pulls / combine RMWs), handing off through LDS. This is
the only construction that yields an **independent injection counter without adding
occupancy** — precisely the resource the depth-4 cliff (LAW-20), the exp_38
involuntary re-throttle (LAW-32: issue runs collapsed 23.5 → 1.45 in flight), and the
G25-1 null (four producer-side interventions moving nothing) all point at.

*Why it is not just the falsified CTA-dedication result.* LAW-28's "dedicating CTAs
to communication never wins" is **carrier-conditional** (contradiction C3: dedication
*won* at C=64 in the staged-push carrier, 0.888× vs homogeneous 0.894×) and its
mechanism is the **capacity tax** (8–9 µs per reserved CTA, LAW-27) plus memory-system
interference (LAW-13, 2.8× the capacity tax). A wave-role split pays **no CTA capacity
tax at all** — the CU still runs one block — and it does not change the number of
CTAs touching each L2, so LAW-19's placement pathology does not apply either.
*What it does cost:* 1/4 of the CTA's MFMA issue slots. Against P3 (M6 is ~84%
K-loop-stalled at 17.1% MFMA duty) that is plausibly free; against M7 (CTA-throughput
bound, LAW-15) it is plausibly a straight 25% loss. **The screen is therefore
phase-dependent and that is itself the experiment.**

*Screens:* SCREEN-1 — it is a **new door**, or more precisely it is the only way to
open door (e) at prefill without raising occupancy; SCREEN-3 — it must re-fit `d`
because the transport wave's congestion profile is entirely different; SCREEN-7 —
high risk, this changes register allocation for the whole function and exp_38 is the
precedent (a bound that can silently die is not a primitive).
*Sketch:* cheapest first form is **not** in the megakernel: put it in the G25-1
boundary rig (`m25_boundary_bench.hip`), which already has arms {compute, phased,
fused} and a transport template argument (`m25_cdar.cuh:158`). Add arm `fused_wave`:
`if (threadIdx.x >= 192) { transport_loop(); } else { mfma_body(); }`. One rig, one
new arm, no serving risk.
*Falsifier:* if `fused_wave` hides <10% of the collective **at the corrected
compute:transport ratio** (TP8_STATE M2), the "independent vmcnt is the missing
resource" hypothesis is dead and the wall is elsewhere (which is what the phase ledger
M1 would then have to name).
**EV: MED-HIGH.** Highest-information single new arm in this atlas: it is cheap, it
is testable in an existing rig, and it discriminates a *resource-class* explanation
(B2 in `../grounding/TP8_STATE.md` §6) from a *schedule* explanation — which is
exactly the question four falsified producer-side mutations left open.

### A9. **Compute-side throttling ("reverse K4")** — throttle the GEMM, not the carrier

**SPECULATION (mine).** Every throttle in this repo throttles the *carrier*
(K4 depth, pool pacing, poll backoff). But LAW-13 says the cost is **memory-system
contention**, and LAW-7 says *"the pusher's slowness is load-bearing"* — driven flat
out in isolation, peer write cost **+624 µs vs +38/+105 for read/local-write**, a
**519 µs fabric surcharge at 5.96×**, which the real pusher never reaches because its
`load → wait → store` shape throttles it ~5.9×. Contention is symmetric: **either
side can yield.** Under TP8 at the deployment ratio the compute has slack by
construction (P10: compute is 2.6–3.2× the collective), so **slowing the MFMA body by
X% is free until X pushes compute past the collective**.
*Sketch:* an `s_sleep`/issue-rate knob or a K-step stride in the MFMA body, swept in
the boundary rig alongside `k_inner`. Falsifier: if fused wall is flat or worse across
the compute-throttle sweep, contention is not the mechanism and B2 is dead.
**EV: MED.** Cheap, and it is the *only* arm that tests the contention hypothesis
from the side nobody has pushed on.

### A10. What A-family says about the expert question

*"Does our M15 producer/consumer work best for all sizes?"* — the banked answer is
**no, and we can already name the axis**: at T=1,024 the fused megakernel **loses**
(1.0940×) against production, with break-even at **T ≈ 1,600–1,800**
(`../grounding/WORKLOAD_EVIDENCE.md` §W6, exp_04_tgen, 5-rotation campaigns, commit
`43291977`). **Provenance trap that must ride that number:** it was measured on the
**tgen body** (sha `20b8c6bb`), not the serving M15 body (`935f555e`), which fails the
MoK gate at T=2048/1024; the serving body has never produced a number at T≠4096, and
`K0_MAXTOK == T` makes the sweep a **batch-size** sweep, not a fill sweep.

---

## 5. Family B — The shared expert as universal overlap filler

The shared expert is the only same-epoch forward work with **zero dependency on
dispatch, plan, sort, or any peer** (`SHARED_EXPERT_FILLER_DESIGN.md:14-27`). It is
dense, it runs on every GPU, its input (`hidden_states`, descriptor slot
`K0P6_D_HIDDEN`) is "resident and immutable from the first instruction of the kernel"
(`:521-527`), and today vLLM runs it **serialized, outside the megakernel, before**
the routed region.

**The bound that must ride every claim:** the shared expert is **1/8 of routed
FLOPs** (4,096 rows vs 32,768 receive rows at the same 7168×2048×3 shape,
`:636-651`). It can fill at most ~1/8 of a layer's compute span. It is **real but
capped and must never be sold as the answer to a 28% pool** — this is already stated
as the B4 bound in `../grounding/TP8_STATE.md` §6.

### B1. SHEXP under the DP/EP dispatch window (prefill) — **the designed arm**

*Comm hidden:* M1's 162 MiB/rank dispatch push. *Compute hiding it:* shared-expert
GEMM-1 tiles. *Slack:* **P1 = 642 µs × F CTAs = 102,720 CTA·µs at F=160**, against
shexp p1's **79,000 CTA·µs** — with 23,000 CTA·µs of headroom
(`SHARED_EXPERT_FILLER_DESIGN.md:653-662`). SCREEN-0 **passes with the arithmetic
shown**, which is rare in this atlas.
*W cells:* `TOPO=DP/EP`, `PHASE=prefill`, any `CONC`; the window's existence is
`SKEW`-independent at std=0 and *grows* with skew (P2).
*Screens:* SCREEN-1 — the filler pool **carries nothing** (zero fabric traffic, zero
atomics on shared protocol cells except one `fetch_add_relaxed` per claimed task,
`:446-468`) so it is a *consuming* pool by the LAW-28/C3 taxonomy, not a carrier;
SCREEN-5 — quota (`Q ∈ {1,2,3}` × `F ∈ {64,96,128,160}`), never yield, because one
task is ~230 µs against a 642 µs window; SCREEN-7 — G2 requires flag-off `.text` sha
identity; SCREEN-8 — the predicted −1.6…−4.6% e2e **is below the c32p noise floor**
and the design says so.
*Sketch:* S4 + `K0P6_M15_SHEXP` (default 0, `F2` in the fusion-boundary table), two
new ticket words in `mps_state` (words 4–5, `moe_mps_adapter.cuh:91-95`), tail
placement reusing `compute_id_of`.
*The gate that decides it:* **G1** — `K0P6_M15_SHEXP_DRYRUN=F` makes the tail F CTAs
skip M1/M2 and read `K0P6_MPS_TS_M2_DONE`. Green if M2-done is flat to ±40 µs to
F≈160; red if it scales like `642 × 256/(256−F)`. Two hours, one macro, **and it is
worth running even if SHEXP is never built — it is the missing measurement behind
every "dispatch overlap" claim in the aug12 document** (`:712-721`). The design
itself prices the risk that the window does not exist at **P ≈ 0.4**.
*Expected value:* `−(450…700) + 272 = −180…−430 µs/layer` if G1 is green;
`−55…+195 µs` in the pure work-conservation fallback (`:685-691`).
**EV: HIGH at kernel level, MED at e2e** (SCREEN-8: unmeasurable at c32p n=1; it must
be claimed as a region-time delta plus an accuracy receipt, or not claimed).

### B2. SHEXP under TP8+EP (prefill) — **strictly simpler than B1**

Under TP8 the shared expert is TP-sharded and **its output folds into the same ERS
accumulator, deleting its separate all-reduce** (`TP8_OVERLAP_ANALYSIS.md:83`;
`M25_TP8_MEGA_DESIGN.md:76-83`, "it is just queue priority"). The routed tiles cannot
start until MAG#1 lands slab-wise, so **shared-expert tiles are the natural work-queue
fallback whenever routed tiles starve** — and the starvation window is exactly the
MAG#1 ramp, i.e. the collective we are trying to hide.
*Slack:* P16 (≈1/8 of the layer's routed compute) against P9's 2.8 ms/layer. At
7.2–7.6 ms/layer of compute, 1/8 is **~0.9–0.95 ms/layer** — i.e. **the shared expert
alone can cover ~32–34% of the exposed AR** (DERIVED). That is a third of the prize,
not the prize.
*Screens:* all pass; it is door (a) with a consuming-pool shape and zero new
machinery. SCREEN-2 passes trivially — under TP8 every rank owes every slab the same
bytes at boundary 1, so there is no hot rank to bypass
(`../grounding/TP8_STATE.md` §4.2).
*Sketch:* S6 (M25) + work-queue priority `routed → shared → next-boundary consumer`
(`M25_TP8_MEGA_DESIGN.md:126-133`).
**EV: HIGH**, and it should be sequenced **before** further transport work, because it
is the one hiding vehicle whose slack is bounded by construction rather than by an
unmeasured hypothesis.

### B3. Shared expert at decode

At decode the shared expert is a *dense* GEMM over the whole (small) batch while the
routed experts touch few rows each — so its **relative** share rises. But at decode
the MoE is HBM-weight-bound (§6.2) and the shared expert's own weights are one more
42 MB-class stream. *Screens:* SCREEN-1 corollary — if occupancy rises above 1 at
decode tile shapes, this becomes a genuine stall-fill and the arithmetic changes
qualitatively. **EV: MED, entirely unmeasured** (hole H1: no decode cell has ever run).

### B4. Shared expert (fwd+bwd) as training filler — **the one place the sizing already fits**

LAW-53's endgame: shared-expert GEMMs are "~2 ms/layer·mb of work vs ~150 CTA-ms of
window supply — **it fits where wgrad's 13 ms could not**"
(`aug18/V6_EVIDENCE_AND_DESIGN.md:88-92`). Shared-expert bwd is ~1/4 of routed bwd
(`aug15/OVERLAP_PROGRAM.md:9-33`). *Screens:* SCREEN-5 is the whole story here — the
five falsified wgrad variants died on granularity and fences, not on the idea.
**EV: MED-HIGH (training arm).**

---

## 6. Family C — TP8 attention-region overlaps (the region the experts said we ignore)

This family exists **only** under TP8+EP: under DP/EP attention/dense/shared are
replicated with **zero communication** (`../grounding/TP8_STATE.md` §1.2). That is why
the atlas's TP8 half is dense with opportunities the DP half structurally cannot have.

### C1. AR#1 (o-proj output) hidden in the o-proj GEMM epilogue — **ERS#1, built, red**

*Comm:* boundary-1 all-reduce, 235 MB, 411 MB/rank egress. *Compute:* the o-proj GEMM
itself. *Status:* built as CDAR ERS#1; **fused 3,606 µs vs GEMM→RCCL 2,909 µs**, with
**2.1% of the collective hidden against a ≥90% bar** (`G25_1_STATUS.md`, `3f7f0f3a`).
*Why the negative is not yet decisive:* SCREEN-0 — the rig ran with the collective
~3× **longer** than the compute (P12), the inverse of the deployment ratio (P10).
**"No hiding" has not yet been tested where hiding was possible**, and the fix is one
argv value (`../grounding/TP8_STATE.md` §2.4, M2).
*Prior art we own on this exact shape:* the GEMM-RS adapter already **beats RCCL**
on a comm-dominant shape — **376.7 µs vs RCCL 421.8 vs RadeonFlow 485.3**
(`docs/distributed/ARCHITECTURE.md:124`), with the caveat that
`docs/distributed/SOURCE_AUDIT.md:104` warns EV=1 and REDV=1 were never measured
together, so G25-0a **does not transfer to CDAR** unexamined.
*Screens:* SCREEN-1 door (a)+(c)+(d); SCREEN-3 must re-fit `d` at the corrected ratio;
SCREEN-6 satisfied structurally — under TP8 `owner_of(slab)` is known **before the
producing GEMM starts** and to **every rank without communication**, so
`owner_major_staggered` applies at plan time and the ring emerges by construction
(`../grounding/TP8_STATE.md` §4.1).
**EV: HIGH but currently RED.** It is the entire M25 thesis. Ordering: M2 (ratio fix)
→ M1 (phase ledger) → this.

### C2. Reduce-scatter *inside* the GEMM epilogue as the general primitive — **measured to work on both arches**

The gfx942 GEMM-RS bring-up (`distributed-kernels/gemm_rs/overnight/RESULTS.md`)
gives the honest cost decomposition for exactly this pattern at 8×MI300X:
GEMM mainloop **46%** of the largest shape, **XGMI egress 32%**
(~127 GB/s against ~448 GB/s per-GPU XGMI — "about 3.5× off, consistent with 16-byte
scattered peer stores"), **per-tile release 9%** (250 µs on shape 6), reduce ~8%,
cross-rank sync ~9%. Two transferable findings: (i) **a uniform
`NUM_REDUCER_CTAS = 32` is optimal or within noise on all six shapes** and beats the
inherited RadeonFlow per-shape table — carrying a donor's constants cost 8.3% on the
largest shape; (ii) **grouping `producer_drain_release` across the tiles a CTA
produces, instead of per tile, is 9% sitting on the table**.
*Relevance to A4/C1:* (i) is direct evidence that the consumer-pool size is a
**derived** quantity that must be re-swept per shape (the same law as A4, on a
different arch and a different skeleton — that is two independent regimes agreeing);
(ii) is a **door (b)** protocol-deletion item that CDAR has an exact analogue of
(the certify/publish cadence).
**EV: MED-HIGH for (ii) as a cheap CDAR arm; HIGH as cross-regime evidence for A4's
law.**

### C3. AR#1 landing overlapped with the router/gate + row-gather (counter-dataflow across the attention→MoE boundary)

*Comm:* MAG#1 (the pull half of boundary 1). *Compute:* the gate GEMM + the local
expert row-gather + the first expert tiles, per certified slab.
*Slack, sized (DERIVED):* gate GEMM at 16,384 tokens is
`16,384 × 7,168 × 256 × 2 = 60.1 GFLOP`; at the gfx942 bf16 MFMA roofline anchor used
in the GEMM-RS SOL table (`1.307e15`, `gemm_rs/overnight/RESULTS.md:160-165`) that is
**46 µs at roofline**, realistically 100–200 µs — **1–2% of a 10.4 ms layer**, and it
is *on the critical path* between AR#1 and the expert GEMMs. The larger, unmeasured
term is that **production serializes gate → sort → group as separate kernels**, whose
launch-gap cost is explicitly "sized by vLLM kernel-gap time in B0 profile"
(`M25_TP8_MEGA_DESIGN.md:164`) — **never measured** (hole).
*W cells:* TP8+EP, prefill; collapses at decode (chunking dies below ~2K tokens).
*Screens:* SCREEN-6 passes — this is the textbook early-executable unit: routing for
slab k is known the moment gate(k) finishes, and the rolling planner appends tiles
per certified slab. SCREEN-0 is **unmet**: the slack (gate+gather, ~100–200 µs) is far
smaller than MAG#1 (~1 ms class), so this overlaps a small consumer into a large comm
— it removes the gate from the critical path, it does not hide the AR.
*Sketch:* S6 + rolling planner CTA (`M25_TP8_MEGA_DESIGN.md:126-131`), `F4` in the
fusion-boundary table.
**EV: MED.** Correctly framed as *critical-path deletion*, not as hiding. Do not let
it be reported as AR hiding.

### C4. Owner-shard RMSNorm fused at the ERS certificate — and the mirror variant nobody has considered

*The built design:* norms on the owner's shard only, deleting the 8×-replicated norm
production pays: "8× less norm work + one less full HBM pass per boundary; small but
free" (`M25_TP8_MEGA_DESIGN.md:161`, `F3`).
*Sizing (DERIVED, ASSUMED bandwidth efficiency):* a full pass over the 235 MB boundary
tensor is 470 MB of HBM traffic; using the exp_29-implied HBM figure
(1.57 TB/s = 20% of HBM ⇒ **HBM ≈ 7.85 TB/s**, `OVERLAP_ABSTRACTIONS.md:26-30`) that
is **60 µs at peak, ~150 µs at 40% efficiency**. Owner-sharding saves 7/8 of it:
**~50–130 µs/layer**, i.e. **0.5–1.3% of the layer**.
*The mirror variant (SPECULATION, mine):* put the norm on the **consumer** side of
MAG instead — compute it as the pulled slab lands, in the consumer's ramp. It costs
8× redundant elementwise work (which is what production already pays) but it removes
the norm **from the certificate's critical path entirely**, so no consumer waits on a
norm that has not run. Which side wins is a pure `slack_consumer_ramp` vs `8×
elementwise` trade, and it is decidable from the phase ledger.
**EV: LOW-MED (both variants).** Small, free, and worth carrying as a control that
prices the certificate chain.

### C5. AG (MAG#2) overlapped with the next layer's attention — v1's structural bubble

*Comm:* MAG#2's tail. *Compute:* attention of layer L+1. *Problem:* attention is
**host-launched** in v1, so the last slabs' pull cannot hide under it; sized at
`~1/n_slabs of MAG#2 + launch gap` (`M25_TP8_MEGA_DESIGN.md:203-213`). At 64 slabs
that is **~1.6% of MAG#2 plus the gap** — small *if* the launch gap is small, and the
launch gap is **unmeasured**.
*Doors (from the design, choose after the phase ledger):* (a) MLA attention in-kernel
(largest lift), (b) chunked attention on a hipGraph with per-slab events (medium),
(c) persistent-attention co-kernel sharing the certificate space.
*Screens:* SCREEN-4 is decisive for (c) — a persistent co-kernel against a resident
256-CTA mega is exactly the standoff LAW-54 describes, so (c) is predicted to
serialize unless the mega stops being all-CU (the T3/S7 direction). (b) survives
SCREEN-4 **only if each chunked attention launch is sub-millisecond**, which at 16,384
tokens ÷ 64 slabs = 256 tokens/slab is plausible but unmeasured.
**EV: MED for (b), LOW for (c) as stated, HIGH-cost/HIGH-ceiling for (a).**

### C6. **Collective-under-collective: stop hiding the AR under compute and start smoothing it across the layer**

**SPECULATION (mine), but it follows directly from §2.1's arithmetic and it may be the
most important item in this document.**

The entire M25 framing is "hide the collective under the compute". §2.1 shows the
fabric needs only a **28–39% duty cycle** to carry the whole layer's 822 MB. So the
correct objective is not *"put the collective in the shadow of a compute burst"* — it
is ***"keep the fabric at ~1/3 duty continuously for the whole layer"***.

Three things follow that the design does not currently do:

1. **AR#1 and AR#2 of the same layer can pipeline against each other, not only
   against compute.** Slab s's expert GEMM starts as soon as MAG#1[s] lands, so slab
   0's ERS#2 can begin while slab 63's MAG#1 is still in flight. That is
   collective-collective overlap, and the fabric has the headroom for it (43% + 43%
   would still be under the ceiling at our *own* measured rate).
2. **The bursty epilogue is the problem, not the transport rate.** Our epilogue-carried
   ERS necessarily bursts at the GEMM's drain. A metered ERS that spreads commits
   across the whole producing GEMM (rather than across its epilogue) achieves the same
   bytes at lower instantaneous rate and therefore lower contention (LAW-13's
   interference term is a memory-system property, and LAW-16 says the interference is
   *protocol traffic*, which scales with event density in time).
3. **This re-scopes the transport track (A1–A6 in `TP8_STATE` §6).** Closing the 1.5×
   RCCL gap buys ~740 µs/boundary in-rig, but at deployment the fabric is idle ~2/3 of
   the time — so a 1.5× slower transport that runs continuously beats a fast transport
   that runs in bursts. **Track B (hiding) should be funded ahead of Track A
   (transport efficiency),** which agrees with TP8_STATE's own Amdahl ranking but for
   a sharper reason.

*Falsifier:* instrument the phase ledger (TP8_STATE M1) to emit **fabric duty over
time**, not just phase spans. If the fused arm already shows a flat ~35% duty across
the layer and still fails to hide, then smoothing is not the lever and the wall is
consumer-side read bandwidth (hypothesis A1) or HBM collision (A2).
**EV: HIGH as a reframing; MED as a specific arm** (it needs the phase ledger first —
"one instrumented run replaces the next four guesses", `G25_1_STATUS.md`).

### C7. Sequence-parallel RS+AG instead of AR

Halves norm-path comm and gives natural chunk boundaries, but token-sharded hidden
complicates the local row-gather that is the whole EP-inside-TP dividend
(`TP8_OVERLAP_ANALYSIS.md:117-120`). Explicitly **secondary**; evaluate only if the
other tracks leave exposed norm time. **EV: LOW.**

---

## 7. Family D — Cross-layer pipelining

### D1. Layer-k combine under layer-k+1 attention/router (the forward's one legal cross-region overlap)

`aug15/OVERLAP_PROGRAM.md:9-33` enumerates the forward dependency chain
(attn → router → dispatch → GEMMs → combine → residual → next attn) and concludes it
leaves **exactly one** legal cross-region forward overlap: **combine(L) tail vs
attention(L+1) prologue on finished tokens** — "fine-grained, expensive machinery,
low priority".
*Screens:* SCREEN-6 — "finished tokens" is a genuine early-executable unit only if the
combine's readiness staircase exists, which is K2's nc-major order (LAW-24);
SCREEN-4 — under DP/EP attention is a separate launch, so this needs either in-kernel
attention or a sub-ms chunked attention.
**EV: LOW-MED** (DP/EP), superseded by C5 under TP8 where the same edge is the
designed v2 door.

### D2. The counter-dataflow skeleton (T3 / S7) as the enabler for everything cross-layer

T3's diagnosis is the reason cross-layer work keeps failing: *"our all-256-CTA grid
barrier creates the residency contract that made every separate wgrad kernel standoff
or serialize, quantizes execution into lockstep phases whose bubbles are small
(~150 CTA-ms vs wgrad's ~3,360), and realizes cross-rank drift at **three points per
launch (M0/M2/M8) instead of one per layer**"*
(`aug18/T3_COUNTER_DATAFLOW_DESIGN.md:43-49`). The MoK counter-example builds none of
those constructs — no grid barriers, every dependency a flat HBM counter, a
work-stealing ticket pool with surplus clusters early-exiting.
*What is different from the falsified naive variants (G25-1 F1–F4, V6's five wgrad
variants):* those all kept the residency contract and moved work **inside** it. T3
removes the contract. **This is the structural precondition, not another mutation.**
*Cost/risk:* the projection band 1,280–1,330 ms training step is **explicitly a
projection, not a measurement** (`:85-91`). And T3 reintroduces fine-grained polling
that the serving campaign measured as toxic **in its carrier-pool form** — the stated
difference is that these counters gate **consumers on certified-enough frontiers**
(block/minibatch), not carriers on rows.
**EV: MED-HIGH as a skeleton bet; LOW as a near-term ratchet.** It is the only
skeleton in the inventory that makes SCREEN-4 stop binding.

### D3. Expert-weight prefetch across layers (built, gated)

`K0P6_M20_SLOTPOOL` + `K0P6_M20_PF_CTAS = 8` pool CTAs streaming the **next-next**
layer's replica weights during the M6..M8 window, where *"stream order **is** the
certification; no flags, no tags"* (`k0pf6gm_device_tile_m15.hip:134-142, 432,
2091-2100`). Deleting the M0.5 pre-pass and its two grid-barrier rendezvous is priced
at **+1.2 ms measured** (`:143-145`). Never isolated as a performance delta.
*Screens:* SCREEN-1 door (a) — it is a latency-bound consumer (HBM/PCIe stream) in a
compute shadow; SCREEN-5 — 8 CTAs is a quota by construction.
**EV: MED (prefill), HIGH (decode — see F3).**

### D4. Cross-epoch / TBO under DP/EP — see A3. **The launch boundary as a pipeline stage is falsified in its one built form (+75.8 µs).**

---

## 8. Family E — Router / gate / norms: fuse or overlap?

| item | size | fuse or overlap? | verdict |
|---|---|---|---|
| **gate GEMM** (tokens × 7168 × 256) | **46 µs at roofline**, ~100–200 µs realistic, per layer (DERIVED, §C3) | **fuse** (into the mega, appended per certified slab) | the win is deleting the serialized gate→sort→group **kernel gaps**, which is unmeasured — `M25_TP8_MEGA_DESIGN.md:164` |
| **sort / histogram** | part of plan M3–M5 = 372.8 µs with **three grid barriers inside** | **fuse, do not try to overlap** | it is a global function of all arrivals (A2) — the barriers are the dependency, not an implementation choice |
| **RMSNorm** (TP8) | ~50–130 µs/layer saved by owner-sharding (DERIVED, §C4) | **fuse, owner-sharded** | "small but free"; the consumer-side mirror is the untested alternative |
| **RMSNorm** (DP/EP) | replicated, **zero comm** | nothing to overlap | structurally absent |
| **SiluAndMul** | ~15 µs over `[4096,4096]` bf16 | already fused in M6 | — |
| **residual add** | folded into the ERS accumulator preload (`M25_TP8_MEGA_DESIGN.md:68`) | fuse | free |

**The law this family yields:** *small ops on the critical path should be **fused**
(to delete launch gaps and HBM passes), never **overlapped** (their slack is smaller
than the machinery that would hide them).* The one exception is the gate under TP8,
because its output (routing) is what unblocks the expert tiles — there, fusing is what
makes the *rolling planner* possible, and the planner is what makes C3's
early-executable unit exist at all.

---

## 9. Family F — Decode: a different mechanism, and an explicit regime flip

**Standing caveat (LAW/hole):** *"Decode has never been measured at all in any of our
rigs — every kernel law above is a prefill law"*
(`../grounding/BANKED_LAWS.md` §8 hole 4). The `c8/c16/c32` (ISL 1024, OSL 512) and
`c512` decode cells are **defined and never executed** (hole H1). Everything in this
family is DERIVED or SPECULATION, and it is stated that way.

### 9.1 Which comm even matters at decode, per topology

| topology | per-layer comm | payload at a decode-ish size | measured point we own |
|---|---|---|---|
| **TP8+EP** | 2 dense ARs | `B_global × 14,336 B`; at B=256 that is **3.67 MB** | CDAR at 256 tokens / 64-row slabs: **801 µs (atomic) / 1,017 µs (towers)** ⇒ **4.6 / 3.6 GB/s effective — latency-dominated, and towers LOSE** (`../grounding/TP8_STATE.md` §5, M5) |
| **DP/EP** | MoE all2all only, tiny payloads | — | **but today the mega executes decode steps at PREFILL shape**: post-M23 the uniform-decode rescue routes them through the B4096 graph, so the kernel runs 4,096 padded rows for ~1 real token per rank (`../grounding/WORKLOAD_EVIDENCE.md` §3.2; `CORPUS_FINDINGS.md:9-16`) |

**The DP/EP decode finding, stated plainly (DERIVED):** at decode the DP mega moves
and computes the *prefill* volume — 162 MiB of dispatch and 4,096 rows of MoE — for a
handful of real tokens. **The dominant decode cost under DP/EP is not comm at all; it
is padding, by two to three orders of magnitude.** Any decode overlap design for
DP/EP is second-order to the fill lever (Family G).

### 9.2 The regime flip, stated as a law table

| quantity | prefill (measured) | decode (predicted) | why |
|---|---|---|---|
| AR regime | **bandwidth-bound**: 235 MB, 2.8 ms/layer exposed | **latency-bound**: 3.67 MB, ~0.8–1.0 ms on our transport ⇒ **64× fewer bytes buys only ~2–3× less time** | the measured 256-token CDAR point, §9.1 |
| exposed collective **fraction** | ~27% of the layer | **rises**, unless a latency-optimal AR is used | numerator falls 3×, denominator falls much more |
| **K1** (transport) | store-towers win (117.2 vs 67.1 GB/s at 235 MB) | **REVERSES — towers lose** (1,017 vs 801 µs at 3.67 MB): the owner reduce is no longer amortised | `../grounding/TP8_STATE.md` §5 M5 |
| **K3** (slab granularity) | interior optimum at 256 rows (117.2 GB/s; 1024 rows are 2.1× worse) | **chunking collapses below ~2K tokens** — the slab concept dies | `M25_TP8_MEGA_DESIGN.md:219`; `KNOB_INVENTORY` B.3 |
| **K4** (depth) | −615 µs co-resident, flat in pure transport | **inert** — there is no bandwidth congestion to control | LAW-8/LAW-20 |
| **K5** (`C`) | 24–28 optimal | must be re-derived; consumer work is tiny | `k0pf6gm_device_tile_m15.hip:2122-2127` |
| **LAW-12 / SCREEN-1** | occupancy exactly 1 ⇒ door (e) closed | **may break**: small-M tile bodies drop below the 256-VGPR/155 KB-LDS budget ⇒ occupancy ≥2 ⇒ **door (e) re-opens** | LAW-12's own scope note (`aug11/LESSONS.md:99-109`) |
| primary metric | input tok/s, TTFT | **TPOT** | `../grounding/WORKLOAD_EVIDENCE.md` §W8.3 |

**The last row of that table is the biggest unexploited structural fact in the
project.** Every "fill the stall with other work" design the prefill ledger correctly
rejects becomes *legal again* at decode if occupancy rises. Nobody has checked.
**Cheapest possible test:** compile one small-M decode tile body and print
`-Rpass-analysis=kernel-resource-usage`. Zero GPU time. LAW-12's corollary is already
binding: *"occupancy must be asserted and printed, not assumed, in every arm."*

### F1. Latency-optimal AR at decode sizes (one-shot / LL-style)

*Status:* designed as G25-0b(iii) (`M25_TP8_MEGA_DESIGN.md:174-176`), **never run**;
`rccl-tests` is absent from the node but the boundary rig already links `-lrccl` and
calls `ncclAllReduce`, so the reference is one argv away
(`../grounding/TP8_STATE.md` §5, M5). **This is the entire go/no-go for every decode
cell — without it we have no basis for any claim below ~2K tokens.**
**EV: HIGH as a measurement, unrankable as a design until it runs.**

### F2. Persistent cross-layer megakernel / whole-layer fusion at decode

At decode the step is launch-overhead-dominated by construction: 61 layers × (attention
+ gate + sort + group + up to 4 MoE kernels + 2 collectives) is on the order of
hundreds of launches per token. A persistent kernel spanning layers deletes them.
*Screens:* SCREEN-4 inverts here — the residency contract is a *cost* when you want
side-stream work, and a *benefit* when the whole step is one kernel; SCREEN-1 depends
entirely on the occupancy question above.
*Sketch:* S6/S7 with certificate-gated layer transitions and **one grid barrier per
launch edge** (`M25_TP8_MEGA_DESIGN.md:134-139`).
**EV: MED-HIGH, wholly unmeasured.** It is the most plausible route to the TPOT metric,
and TPOT is the metric the market quotes.

### F3. Expert-weight prefetch during attention (decode's real overlap)

**DERIVED sizing.** At decode with a global batch of a few hundred tokens × top-8, the
routed-slot count exceeds 256 comfortably, so **essentially every expert is touched**.
Per rank that is 32 local experts × **42 MB/expert** (the figure used in the offline
replication frontier, `aug15/OVERLAP_PROGRAM.md:36-46`) = **1.34 GB/rank/layer** of
weight streaming. At the exp_29-implied HBM ≈ 7.85 TB/s that is **171 µs/layer at
peak, ~342 µs at 50% efficiency** — i.e. **10–21 ms per 61-layer step of pure weight
traffic**, which will dominate any 3.67 MB all-reduce by an order of magnitude.

⇒ **At decode the "comm" worth hiding is the HBM weight stream, not the fabric.** And
the compute that can hide it is **attention**, which is HBM-bound on the KV cache but
*not* on the expert weights — different buffers, same bus, so the overlap is partial
and must be measured, not assumed.
*The route-dependence twist (SPECULATION, mine):* which experts to prefetch is known
the moment layer L's gate finishes, which is *before* layer L's attention has even
been issued in the next layer's pipeline stage. So the prefetch is issuable one full
attention-span early. The mechanism already exists — M20's slot pool with prefetch
CTAs and "stream order **is** the certification" (D3) — it has simply never been
pointed at a decode workload.
**EV: HIGH (decode), MED confidence.** Named falsifier: if attention and expert-weight
streaming saturate the same HBM controller, the prefetch is work-conserving and
measures ≈0.

### F4. KV-cache-op overlap

The KV write for step t is independent of the MoE of step t (it happens at attention
time) and of the collective. Under TP8 it is per-rank-sharded (heads 16/rank) so it
carries no cross-rank traffic. It is therefore a legitimate small filler for the AR
window. **Sizing is entirely unmeasured in this repo** — no decode instrument exists.
**EV: LOW-MED, listed for completeness.**

---

## 10. Family G — Serving-stack-level "schedule knobs" outside the kernel

`../grounding/BANKED_LAWS.md` LAW-50 is blunt: **DP prefill co-scheduling is worth
15–25% of node throughput, kernel-independent — larger than any kernel delta in this
project's ledger.** The atlas would be dishonest if it ranked in-kernel overlaps above
these.

### G1. Dummy-step early-out (tier-1 fill)

*Prize:* **56.0% of in-bucket steps at c32p** are 100% fake work both arms pay
(139.62/249.12, `BOTTLENECK_SIGNALS.md:341-343`). Corpus classifier agrees on *shape*
(>95% of rows share one expert set; across-row weight STD ≈0.008) and the receipt on
*rate*.
*Feasibility:* DP-dummy calls are **perfectly rank-synchronous** in the capture window
— `c[i]` bimodal on {0,8} for all 64 indices, pairwise **κ = 1.000** over 28 pairs,
lag-0 `n_real` correlation **+0.996** — so a step-level skip carries 1:1 from
rank-calls to steps with **no synchrony discount** (commit `2ba124c8`).
*The binding constraint:* the seal predicate is a **collective** decision — a split
seal deadlocks, and an earlier revision let one rank refuse while seven replayed the
graph with the receipt still reporting green (commit `eeff2c74`). Coverage must ride
an all-reduced readiness bit.
*The caveat that bounds it:* the capture window's dummies are entirely ramp-in
(idx 0–13) and drain-out (idx 58–63); **zero mid-run dummy calls appear**, so the
whole-run ~56% *mid-run* rate is not covered by the synchrony result.
**EV: HIGH at c32p; ~zero at c512p** (2% dummy). This is a per-cell verdict, which is
itself the campaign's point.

### G2. Fill-aware (tier-2) row-proportional kernel — M24

Nine default-0 macros exist; **never compiled**; serving-side plumbing (G13a–G15) does
not exist (`../grounding/WORKLOAD_EVIDENCE.md` §3.2). The modeled verdict from the
corpus shape is that **tier-1 captures the entire modeled win and tier-2 adds nothing
under the measured bimodal distribution** (deciles of `n_real`:
`0, 0, 5, 1994, 4096, 4096, …` — ~31% near zero, ~60% completely full, only ~10%
genuinely partial).
*The load-bearing SPECULATION attached to that:* the bimodality may be an artifact of
**ISL 4096 == `max_num_batched_tokens` 4096**, i.e. one chunk per request. If so it
dissolves at any other ISL and **every tier-1 sizing number is cell-specific**
(hole H4, never visited).
**EV: LOW at c32p as currently modeled; UNKNOWN elsewhere — and the "elsewhere" is
one ISL sweep away.**

### G3. DP prefill co-scheduling / chunk packing — **the largest single number in the ledger**

Perfect co-scheduling needs **128 padded steps instead of the observed 260/322**; at
the fitted step costs that is stock 200.5 s → **171.2 s (−14.6%)** and m15
217.6 → **171.2 s (−21.3%)**; at the alternative fit both → **150.4 s (−25.0%/−30.9%)**
(`BOTTLENECK_SIGNALS.md:483-496`). Directly measured driver: **3.93 (stock) vs 3.19
(m15) prefill chunk-events per in-bucket step of 8 possible**.
*Screens:* SCREEN-2 — it removes work from **every** rank's wall including the
critical one, so it passes cleanly; SCREEN-8 — the effect is 3–10× the c32p noise
floor, so it is measurable at n small.
**EV: HIGH, and it is a scheduler change, not a kernel change.** It also restores the
replay's routing conditions (fewer pad rows ⇒ less padding-degenerate skew), so it
compounds with G4.

### G4. EPLB / RR owner placement as a schedule knob living outside the kernel

*Measured prize:* under linear placement, per-rank aggregate receive share is
**63.60% / 6.93 / … ⇒ max/mean 5.088×, max/min 13.84×**; under round-robin (`e % 8`)
it is **13.56 / 13.33 / … ⇒ max/mean 1.085×**. Worst-rank layer **5.593× → 1.296×**
(`rr_placement/RR_PLACEMENT_NOTES.md:208-236`).
*The ceiling nobody should quote past:* RR is a **static** redistribution and the
**ideal static bound still leaves per-call p95 at 3.05×** (`:246-262`); the two
per-call skew instruments disagree by ~2× (m18diag p50 5.15× vs corpus p50 2.61×) and
**must not be mixed** (hole H17, CPU-only to resolve).
*Composition status:* EPLB has **only ever run on stock-target servers, never composed
with the megakernel** — the 128-redundant configuration violates `PF4H-CONTRACT-001`;
placement-only EPLB0 is designed but unrun and costs **≈1.31 GiB/rank**. Its offline
predictor runs vLLM's own policy over our histograms with **no GPU**, and carries an
explicit gate: *if predicted worst-layer critical-rank relief is under ~15%, do not
run the arm* (`M15_EPLB0_COMPOSE.md` §8).
*Why it belongs in an overlap atlas:* SCREEN-2. Skew is worth **3.6× wall** in the
kernel replay rig (balanced 5,823 µs → aggregate-histogram 20,739 µs) — **far more
than any schedule knob in the ledger** — and it acts precisely by making one rank's
slab certificate late, which is the mechanism every overlap in Family A depends on.
**Placement is the cheapest overlap enabler we have.**
**EV: HIGH.**

### G5. `max_num_batched_tokens` / bucket size

Fixed at 4096 in **every run ever executed**. Named as "the single largest lever
nobody has pulled": the ~46–60% of step time that is *not* MoE also runs on 4,096
padded rows, and shrinking the bucket removes pad rows before the model sees them
(`FILL_AWARE_DESIGN.md:1337-1345`). **EV: HIGH, integration-side, cheap** (hole H12).

### G6. Use dummy steps as a maintenance window

**SPECULATION (mine), LOW EV, near-zero cost.** If ~56% of in-bucket steps at c32p are
pure fake work (P15) and EPLB rearrangements take ~3 s asynchronously
(`../grounding/WORKLOAD_EVIDENCE.md` §W4e, class MEMORY — a lead, not evidence), then
scheduling rearrangement/replica-refresh into dummy steps makes it free. Strictly
dominated by G1 (skipping is better than filling) **unless** the maintenance must
happen anyway. **EV: LOW.**

### G7. A staggering idea that this atlas rejects, pre-emptively

One might propose phase-offsetting the eight DP ranks so rank *i*'s all-to-all
overlaps rank *j*'s attention. **This is wrong and the repo already contains the
reason:** DP padding forces all eight ranks into the same bucket
(`should_dp_pad` ⇒ every rank padded to the max across ranks,
`M23_RAGGED_SEAL_DESIGN.md` §1.1), and lockstep is *why* dispatch has **zero exposed
peer wait** (LAW-31 — the transfer is already latency-hidden by symmetry). Staggering
would convert a free collective into an exposed one. Recorded here so nobody proposes
it twice.

---

## 11. Family H — Things the repo has not tried (creative, honestly marked)

### H1. **Padding-degeneracy deduplication** — compute the pad row once, not 2,555 times

**SPECULATION (mine); the underlying mechanism is a stated in-repo HYPOTHESIS with a
free decisive test.**

LAW-42's mechanism prediction: vLLM pads with token id 0 ⇒ **identical hidden states ⇒
identical router logits ⇒ the same top-8 experts**, so ~62% of dispatched rows land on
≤8 of 256 experts; this predicts **2.66× rank-load skew from padding alone**, rising to
~5× with two hot experts on one rank — *"quantitatively consistent with the measured
per-call max-rank load p50 of 5.15× and with essentially nothing else"*
(`BOTTLENECK_SIGNALS.md:516-535`).

**If those rows are identical, they are also identical *outputs*.** So instead of the
two designed tiers (skip the whole step; or plumb ragged row counts everywhere), a
third option exists: **compute one representative pad row per (expert, layer) and
replicate**. Properties that make it attractive relative to tier-2:

* It **preserves every shape and bound in the kernel** — the padded envelope, the
  4,096-row execution shape, the dispatch counts, the graph key. Tier-2's whole
  difficulty is that it must make every row-proportional cost track `T_eff` at five
  named substitution sites (`k0pf6gm_device_tile_m15.hip:182-193`).
* The pad block is **contiguous at the tail** (`n_orig .. 4096`), so a single scalar
  identifies it — the same `n_orig` vector M24 already routes through descriptor slot
  71.
* It attacks **P14 (62.4% of computed MoE rows) and the skew simultaneously**: removing
  the degenerate rows removes the 2.66× padding-induced skew, which is a SCREEN-2 win
  (it removes work specifically from the *hot* rank).
* Pad-row outputs are discarded downstream, so even numerically-loose handling is safe
  — but the honest framing is that this is a **compute deletion**, and its overlap
  relevance is that it *manufactures* slack, which then needs a filler (Family B).

*Risks to state:* (i) the degeneracy mechanism is a HYPOTHESIS, not a measurement —
the decisive check is `np.unique(topk_ids, axis=0)` on an existing capture, **free,
zero GPU time**, and it is already action #1 in `BOTTLENECK_SIGNALS.md:601-606`;
(ii) it changes the routing distribution, so every replay corpus and every skew number
measured *with* padding stops applying (LAW-42 already flags this: fill-awareness
"changes the *routing distribution*, not just the row count"); (iii) it is
DP-topology-specific — under TP8 the dummy/fill lever does not exist at all.
**EV: HIGH at c32p if the free test comes back positive; strictly zero at c512p.**

### H2. **Cross-rank expert work-stealing — nearly free under TP8+EP**

**SPECULATION (mine).** SCREEN-2 says only work removed from the hot rank's wall moves
e2e, and the repo's only answer is **moving compute statically** (M18 replication, M19
adaptive, M20 slot pool, RR/EPLB placement) — `aug15/OVERLAP_PROGRAM.md:26-33` states
it as doctrine: *"In serving, the only lever that moves the skewed critical path is
MOVING COMPUTE; overlap is the second-order term."*

**Dynamic** migration has never been proposed, and there is a structural reason it is
newly cheap: **under TP8+EP the post-AR hidden is REPLICATED on every rank**
(`TP8_OVERLAP_ANALYSIS.md:29-34`), so a cold rank *already holds the activations* for
any token in the batch. Stealing an overloaded expert's rows costs **zero token
movement** — only the weights must be resident somewhere, and the stolen partial joins
**the same ERS accumulator** the owner would have written into. Under DP/EP the same
idea requires shipping the activations, which is why it never looked affordable.

*Mechanism sketch:* the gate runs locally and redundantly on all 8 ranks
(`TP8_OVERLAP_ANALYSIS.md:29-34`), so **every rank computes the same load histogram at
the same time, with no communication**. A deterministic steal rule evaluated from that
shared histogram needs **no protocol at all** — every rank independently derives the
same assignment, exactly the "replication buys schedule determinism" dividend
(`../grounding/TP8_STATE.md` §4.1). Weight residency is the real cost, and M20's slot
pool (~1 GB at B=8, `k0pf6gm_device_tile_m15.hip:126-149`) is the existing vehicle.
*Screens:* SCREEN-2 **passes by construction** — this is work migration off the
critical rank, which is the exact thing LAW-52 says *does* move e2e (as opposed to
wait absorption, which it says does not). SCREEN-0: the slack is the cold ranks' idle
time, which under 5.088× aggregate skew is enormous. SCREEN-7: it is a planner change,
not an epilogue change, so the exp_38 codegen risk is lower than most arms here.
*Falsifiers:* (i) if the ideal-static bound is already close to the dynamic bound, the
extra machinery buys nothing — and RR's numbers say static gets aggregate skew to
1.085× but leaves **per-call p95 at 3.05×**, which is precisely the gap dynamic
stealing would attack; (ii) weight residency may not fit alongside the 940 MB
accumulator ring plus the unpriced towers overhead (≈+205 MB/rank/boundary, DERIVED in
`../grounding/TP8_STATE.md` §4.3).
**EV: MED-HIGH under TP8+EP, LOW under DP/EP.** Highest-ceiling item in this atlas
after C6, and the most expensive.

### H3. **Speculative routing under the collective**

**SPECULATION (mine), high risk.** The router only needs the hidden state to pick a
top-8 argmax. Run the gate on a **partially reduced** hidden (e.g. after 4 of 8
contributions have landed), start the row-gather and the first expert tiles
immediately, and validate against the exact hidden when the AR completes; repair the
(presumably few) rows whose top-8 changed.
*What it buys:* it hides **the whole of MAG#1's latency** behind the expert GEMM ramp,
rather than overlapping it slab-by-slab (C3). That is qualitatively more than the
rolling planner.
*What kills it:* no rollback mechanism exists anywhere in this codebase; a
mispredicted row must be recomputed, and the repair path is on the critical path of
the *slab certificate*, so a single mispredict could cost more than the whole gain.
Also SCREEN-7: a repair branch near the epilogue is exactly the class of unreachable-ish
code that cost +726.9 µs.
*Cheapest de-risking step:* **CPU-only.** Take a captured `topk_ids` npz
(`route_capture/skewhook_v2`) and measure how often the top-8 changes when the hidden
is reconstructed from a partial sum. If mispredict rate is >1%, drop the idea. Zero
GPU time.
**EV: LOW-MED** (high ceiling × low probability × high cost), but the screening test is
free, so it belongs on the list.

### H4. **Grouped certificate release (from the gfx942 GEMM-RS ablation)**

Not mine, but never carried into the MoE/CDAR line: `producer_drain_release` is issued
**once per tile**, each one a `buffer_wbl2 sc0 sc1` L2 writeback plus a `vmcnt(0)`
drain, and it costs **9% (250 µs on the largest shape)**; *"grouping the release across
the tiles a CTA produces, instead of per tile, is the obvious move"*
(`gemm_rs/overnight/RESULTS.md:188-196`). This is a pure **door (b)** protocol
deletion and CDAR has the same cadence question in its certify/publish path.
**EV: MED-HIGH** — cheap, measured on a sibling kernel, and it is exactly the kind of
"delete protocol work" item the four-door rule says actually pays.

### H5. **Instrument the fabric duty cycle, not just phase spans**

**SPECULATION (mine) as an instrument, not a schedule.** Every ledger we own measures
*spans* (`[MPS TS]` phase stamps, running device maxima never reset — LAW-63c). §2.1
argues the binding quantity under TP8 is **duty cycle over the layer**. There is no
live xGMI throughput counter on this node (`amd-smi --xgmi` returns N/A,
`--shownodesbw` returns 0-0 — LAW-11), so duty must be reconstructed from **per-CTA
commit timestamps binned into the layer's timeline**. The phase ledger (TP8_STATE M1)
should emit histograms, not just first/last stamps. Costs the same instrumented run
already scheduled.
**EV: HIGH as a measurement enabler** — it is what turns C6 from an argument into an
experiment.

---

## 12. Master ranking

Sorted by EV, then by cost. "Slack ≥ comm?" is SCREEN-0.

| # | opportunity | regime | slack ≥ comm? | screens | cost | **EV** |
|---|---|---|---|---|---|---|
| A4 | `C*(W)` — re-sweep the consuming pool in every W-cell | DP/EP, all | yes (P5) | all pass | **zero code** (runtime byte) | **HIGH** |
| G3 | DP prefill co-scheduling / chunk packing | serving c32p | n/a | pass | scheduler only | **HIGH** |
| G4 | RR / EPLB placement as the overlap enabler | serving, SKEW | n/a | SCREEN-2 pass | host + 1.31 GiB/rank | **HIGH** |
| G5 | `max_num_batched_tokens` sweep | serving, all | n/a | pass | integration only | **HIGH** |
| B2 | shared-expert filler under TP8 (covers ~32–34% of P9) | TP8 prefill | yes, bounded | all pass | queue priority | **HIGH** |
| C6 | smooth the collective to ~1/3 duty instead of bursting it | TP8 prefill | yes (P11) | needs H5 first | ledger + K4 re-sweep | **HIGH (reframe)** |
| F1 | RCCL reference + one-shot AR at decode sizes | TP8 decode | unknown | — | one argv | **HIGH (measurement)** |
| F3 | expert-weight prefetch during attention | decode | derived, large | pass | reuse M20 | **HIGH** |
| G1 | dummy-step early-out (tier-1) | serving c32p | n/a | collective-decision constraint | vLLM-side | **HIGH @ c32p** |
| H1 | padding-degeneracy dedup | serving c32p, DP | n/a | SCREEN-2 pass | free CPU test first | **HIGH if test green** |
| B1 | SHEXP under the DP dispatch window | DP prefill | **yes, arithmetic shown** | all pass | ~38 h, gated by G1 | **HIGH kernel / MED e2e** |
| C1 | ERS#1 in the o-proj epilogue (M25 core) | TP8 prefill | **currently NO** (P12) | fix ratio first | already built | **HIGH but RED** |
| H5 | duty-cycle instrumentation | TP8 | n/a | — | inside a scheduled run | **HIGH (enabler)** |
| A8 | wave-specialized transport (K1-w) | both | n/a | new door | one rig arm | **MED-HIGH** |
| H4 | grouped certificate release | TP8 / DP | n/a | door (b) | small | **MED-HIGH** |
| C2 | RS-in-epilogue lessons from gfx942 (uniform reducer count, release grouping) | TP8, ARCH | n/a | pass | small | **MED-HIGH** |
| H2 | cross-rank expert work-stealing | TP8, SKEW | yes, huge | SCREEN-2 pass | large | **MED-HIGH** |
| D2 | T3 / counter-dataflow skeleton | training, then serving | n/a | removes SCREEN-4 | skeleton swap | **MED-HIGH (strategic)** |
| B4 | shared expert as training filler | training | **yes** (P8 fits) | SCREEN-5 critical | medium | **MED-HIGH** |
| F2 | persistent cross-layer megakernel at decode | decode | unknown | occupancy question | large | **MED-HIGH** |
| A3 | cross-epoch pipelining, dispatch-side | DP prefill | yes (P7) | different door than mode 16 | large (parity state) | **MED** |
| A9 | compute-side throttling ("reverse K4") | TP8 | n/a | tests LAW-13 from the other side | one knob | **MED** |
| C3 | router/gate + gather under MAG#1 | TP8 prefill | **no** (small consumer) | SCREEN-6 pass | planner CTA | **MED** |
| C5 | MAG#2 tail under next-layer attention | TP8 prefill | ~1.6% + launch gap | SCREEN-4 binds (c) | medium–large | **MED** |
| A7 | per-destination credits (K6) | DP, SKEW | n/a | blocked by LAW-62b | medium | **MED, gated** |
| D3 | cross-layer weight prefetch (prefill) | DP prefill | unmeasured | pass | built, gated | **MED** |
| B3 | shared expert at decode | decode | unknown | occupancy question | small | **MED** |
| A6 | burst de-alignment (rotated-nc) | DP prefill | ≤ +211 µs | pass | one index map | **LOW-MED** |
| C4 | owner-shard vs consumer-side norm | TP8 | ~50–130 µs/layer | pass | small | **LOW-MED** |
| D1 | combine(L) tail vs attention(L+1) prologue | DP prefill | small | SCREEN-4 binds | expensive | **LOW-MED** |
| H3 | speculative routing under the collective | TP8 prefill | large ceiling | no rollback exists | large | **LOW-MED** |
| F4 | KV-cache-op overlap | decode | unmeasured | — | small | **LOW-MED** |
| G2 | tier-2 fill-aware kernel | serving | n/a | modeled ≈0 under bimodal fill | large | **LOW (cell-specific)** |
| A5 | plan-phase overlap | DP prefill | tens of µs | SCREEN-5 fails | — | **LOW** |
| C7 | sequence-parallel RS+AG | TP8 | — | secondary by design | large | **LOW** |
| G6 | dummy steps as a maintenance window | serving c32p | n/a | dominated by G1 | small | **LOW** |
| A2 | dispatch × GEMM-1 same epoch | DP prefill | 1,015 µs, **blocked** | SCREEN-6 fails | — | **LOW (report as a result)** |
| G7 | DP rank phase-staggering | DP | — | **rejected**: breaks LAW-31 | — | **negative** |

---

## 13. What this atlas says to do first

Ordered by information per node-hour, respecting the gate ladders already written.

1. **Free, CPU-only, today.** (a) `np.unique(topk_ids, axis=0)` on an existing
   `skewhook_v2` capture → decides H1 and confirms/kills LAW-42's degeneracy
   mechanism. (b) Reconcile the two per-call skew instruments (5.15× vs 2.61×,
   hole H17) → changes the modeled skew tax by ~2×. (c) Compile one small-M decode
   tile body and print `-Rpass-analysis=kernel-resource-usage` → decides whether
   SCREEN-1's four-door rule survives into the decode cells, which is the single
   biggest structural fork in this document.
2. **One argv.** Put the G25-1 rig at the deployment compute:transport ratio
   (TP8_STATE M2). Every TP8 hiding conclusion we currently hold was measured at a
   point where hiding was Amdahl-capped near zero.
3. **One instrumented run.** The device phase ledger (TP8_STATE M1), extended per H5
   to emit **fabric duty over time**. This is what turns C6 from an argument into an
   experiment, and it is already the named next step in `G25_1_STATUS.md`.
4. **Two hours, one macro.** SHEXP's G1 dry-run (`K0P6_M15_SHEXP_DRYRUN=F`, read
   `K0P6_MPS_TS_M2_DONE`). It decides B1 and it is *the missing measurement behind
   every "dispatch overlap" claim in the aug12 document* — worth running even if SHEXP
   is never built.
5. **One rig arm.** `fused_wave` in the boundary bench (A8). It discriminates the
   resource-class explanation from the schedule explanation, which four falsified
   producer-side mutations left open.
6. **The knob sweep that answers the experts' question directly.** A4: `C × flush_rows`
   inside every W-cell, one HSACO, manifest-addressed. This is the cheapest law in the
   campaign and it is the literal answer to *"does our M15 producer/consumer work best
   for all sizes?"*

**What this atlas refuses to promise.** The shared expert cannot cover a 27% pool
(it is 1/8 of routed FLOPs). Cross-epoch pipelining has one built form and it lost.
Dedicated communication CTAs are a carrier-conditional verdict, not a technique
verdict, and the one regime that might rescue them — skew — has never been drivable in
any kernel rig. And no decode claim in this document rests on a measurement, because
**this project has never measured decode**.

---

## 14. The four sentences this atlas contributes to the master doc

1. **Under TP8+EP the fabric needs only a ~28–39% duty cycle to carry the entire
   layer's 822 MB of per-rank egress, and production's RCCL fabric-busy time
   (2.94 ms/layer) equals its exposed time (~2.8 ms/layer) — so the collective is
   ~100% exposed, the prize is real, and the binding constraint is scheduling, not
   bandwidth.** (DERIVED, §2.1.)
2. **The occupancy-1 four-door selection rule is a prefill law; decode's small-M tile
   bodies may run at occupancy ≥2, which re-opens the "fill the stall" door that
   prefill correctly closes** — and that is checkable with a compiler flag and zero GPU
   time. (DERIVED from LAW-12's own scope note.)
3. **The largest overlap-adjacent levers in the ledger are not in the kernel:** DP
   prefill co-scheduling (15–25% of node throughput), padding (62.4% of computed MoE
   rows at c32p), and expert placement (skew is worth 3.6× wall in the replay rig).
   Every in-kernel overlap in Family A is smaller than all three.
4. **The one granularity this project has never tested is the wave**, and it is the
   only construction that yields an independent `vmcnt` without raising occupancy —
   which is precisely the resource the depth-4 cliff, the exp_38 involuntary
   re-throttle, and the four null G25-1 mutations all point at.

---

## Amendments from review (2026-08-18)

Ranking and screens survive intact; two items are corrected by BLOCKING findings in
`../review/MECHANISM_CRITIQUE.md`, and the atlas's duty-cycle reframe is promoted:

1. **A8 / `fused_wave` is re-sized from S to M and moved out of the core campaign**
   (MECHANISM B7). It is **not implementable as written**: `m25_boundary_bench.hip:147` is
   `__launch_bounds__(256,1)` and the fused path contains **10 `__syncthreads()`**, including inside
   the commit loop, so a `threadIdx.x >= 192` role split diverges across block-wide barriers. Any
   real version restructures the hot loop and its register allocation — the class of change that cost
   **+726.9 µs with source untouched**. Its mechanism claim is also restated: the argument addresses
   the CTA capacity tax and L2 placement, not LAW-13's interference term (2.8× larger), and a wave
   split **relocates** memory-system contention into the CU. The corrected hypothesis is *the missing
   resource is per-wave ordering, not per-CU memory capacity*, discriminated at fixed total vmem
   request rate, with a compute-only 3-wave control at the same `k_inner`.
2. **§C4/§F3 must not size against the back-derived "exp_29-implied HBM ≈ 7.85 TB/s"**
   (MECHANISM M10). Use the **measured** plateau, 4,151.9 GB/s at a 128-CTA knee. This *strengthens*
   F3's conclusion (~323 µs/layer of expert-weight traffic). A back-derived denominator is the same
   error class as the spec-sheet denominator that manufactured a refutation (LAW-62a).
3. **C6 (duty-cycle metering) is promoted from a reframe to a testable coordinate.** Burstiness —
   peak/mean fabric injection over the layer — has no coordinate in `Q` and no term in `M`, so two
   schedules identical in all eight `Q` scalars but different in injection profile are predicted to
   tie. It is registered as **FQ-4**, a fourth matched-`Q` falsifier, at zero marginal node cost
   because E-B2 builds the instrument anyway. The duty histogram gains an **HBM** counter alongside
   the fabric counter so store-towers' unpriced ~+205 MB/rank/boundary shows up (MECHANISM M7).
4. **A "residency" cell is added** (MECHANISM B8): `N_CTA ∈ {160,192,224,256} × {comm in-kernel, comm
   as a second concurrent kernel}`, which sweeps the one structural variable the +831 µs
   dedicated-comm-CTA negative rests on and pins LAW-28's scope to *CTA-granular dedication inside an
   all-CU, grid-barriered, occupancy-1 CDNA megakernel*.

# W_VECTOR — the workload state vector, and the mediating quantities it collapses into

**Written:** 2026-08-18, laptop, no node access. **Deliverable:** output (1) of
`../BRIEF.md:113-114`, extended with the layer the BRIEF's mission statement actually
requires: *"if you change X in the workload, the schedule should change by Y, because
mediating quantity **Q** crossed threshold **θ**"* (`../BRIEF.md:15-17`).

**Consumers:** `S_VECTOR` (the schedule state vector — already drafted as
`../grounding/KNOB_INVENTORY.md`), `COST_MODEL` (M(W, S; θ)), `GRID` (the experiment
ladder), `METRIC_MAP`.

**Sources.** This document derives nothing from memory. Its inputs are the four grounding
briefs — `../grounding/{KNOB_INVENTORY,BANKED_LAWS,WORKLOAD_EVIDENCE,TP8_STATE}.md` — the
BRIEF's draft8 digest, and direct repo reads where a constant was needed. Every number
carries a source. Arithmetic performed *here* is shown inline so it can be checked, and
tagged **DERIVED**.

**Evidence tags** (inherited from `../grounding/WORKLOAD_EVIDENCE.md` §0.1):
`SERVING` · `RECEIPT` · `KERNEL` · `CORPUS` · `MODEL` · `DERIVED` · `ASSUMED` ·
`SPECULATION`. A claim with no tag is a definition, not a measurement.

---

## 0. The problem this document solves, in one page

The expert feedback contains one risk and one demand, and they pull against each other:

* **The demand:** enumerate the workload properties, and say how the schedule must change
  as each is toggled (`../BRIEF.md:26-28`).
* **The risk:** *"Very expansive, unclear what the core contribution might be"*
  (`../BRIEF.md:29-30`).

Enumerating W and crossing it with S is exactly how the risk materialises. §1 below lands
at **21 workload axes**. `../grounding/KNOB_INVENTORY.md` lands at **9 skeletons × ~25
knobs**. A full factorial is not merely expensive, it is *uninterpretable*: a table of
which knob won in which cell is a lookup table, not a cost model, and it cannot predict a
cell nobody ran.

**The core move of this campaign is a rank reduction.** We assert — and make falsifiable —
that W acts on the optimal schedule **only through a small vector of derived mediating
quantities Q**, computable from W by arithmetic before any kernel runs:

> **SUFFICIENCY OF Q (the campaign's central testable claim).**
> `S*(W) = argmin_S M(Q(W), R(S); θ)`.
> Two workloads with equal Q have the same optimal schedule, even if their W differ on
> every coordinate. Two workloads with different S* differ in at least one coordinate of
> Q.

That single claim is the contribution. It converts "we swept a lot of kernels" into "we
identified the ~8 numbers that decide a megakernel schedule, and we can compute them from
a serving config with a spreadsheet." It is also *falsifiable in a day* — §2.4 pre-registers
three ways to kill it — which is what separates it from a taxonomy.

The rest of the document: §1 enumerates W honestly (including what is unsettable). §2
defines Q, the schedule-actuated response quantities R, and the hardware constants θ, and
states the sufficiency claim with its falsifiers. §3 computes Q for every cell we have
already measured and checks the known results against it — six retrodictions, and four
findings that fell out of the arithmetic and are new. §4 derives the reduced grid from
Q-space corners rather than W-space corners. §5 argues why interpolation must happen over
Q. §6 states what this document does not claim.

---

## 1. W — the workload state vector

### 1.1 Control classes (the column that decides who can move an axis)

| class | meaning | consequence for the campaign |
|---|---|---|
| **MODEL-FIXED** | a property of the checkpoint. Cannot be changed without changing the model. | Not an ablation axis for *this* model; it is the axis along which the results generalise to *other* models. Must appear in Q formulas symbolically. |
| **HW-FIXED** | a property of the silicon/fabric/node. | One point today (8×MI355X gfx950). gfx942 is reachable as a *microbench* only. |
| **OPERATOR** | set on the vLLM serve line or the launch config. Directly settable. | Cheapest axes to sweep. Includes topology, bucket size, prefix caching, EPLB. |
| **LOAD** | set by the client / the traffic. Settable by the benchmark driver. | Concurrency, ISL, OSL, arrival process. |
| **EMERGENT** | *not settable at all* — it is the output of (OPERATOR × LOAD × scheduler). | φ, dummy rate, κ_sched, per-call skew. **These are the ones that matter most and the ones we cannot dial**, which is why §4.2's settability audit exists and why the campaign needs synthetic actuators. |

The EMERGENT class is the reason this document exists. The BRIEF's flagship mechanism
("DP starves at low C; at high C we win", `../BRIEF.md:43-46`) is a claim about an
emergent quantity, φ, that no experiment has ever *set*. `../grounding/WORKLOAD_EVIDENCE.md`
§5 hole H7 states it plainly: fill has **zero controlled points**; the two cells that
support the story "differ in ~six variables at once."

### 1.2 The axes

#### Group A — structural / topology (5 axes)

| # | axis | symbol | measured range (source) | class | settable today? | feeds |
|---|---|---|---|---|---|---|
| A1 | phase | `ph` | prefill only. **Decode has never been measured in any rig** (`grounding/BANKED_LAWS.md` §8 item 4; `WORKLOAD_EVIDENCE.md` H1) | LOAD (via OSL) | e2e yes (cells defined, never run); kernel no | ρ, λ, P |
| A2 | parallelism topology | `topo` | DP8/EP8 (ours), TP8+EP (vendor ≤128 conc), DP8+EP (vendor ≥512). **128–512 undocumented by the vendor** (`WORKLOAD_EVIDENCE.md` W8.1, `nightshift/LOG.md:356-360`) | OPERATOR | yes (serve line) | B_bnd, κ, φ (φ *exists only under DP*) |
| A3 | world size / EP degree | `N`, `EP` | 8 only | OPERATOR (bounded by node) | 8 only | fan-in, pre-reduce value (`BANKED_LAWS.md` LAW-51 is explicitly EP-degree dependent) |
| A4 | node count | — | 1 (intra-node xGMI) | HW-FIXED | **no** | — |
| A5 | hybrid shapes (TP4×DP2, PP) | — | **never run, never configured** | OPERATOR | untested | B_bnd, κ |

**A4 recommendation, stated once:** multi-node is `WORKLOAD_EVIDENCE.md` hole H10 with
zero evidence. **Scope it out explicitly** in the master doc rather than gesture at it.
Every fabric law we own (`BANKED_LAWS.md` LAW-2/3/4) is an intra-node xGMI law.

**A5 note:** TP4×DP2 is the only hybrid with a principled motivation from our own data —
it halves the AR participant count (reducing `2(N−1)/N` egress modestly) while retaining
half the DP fill dilution. It is **SPECULATION**; no repo artifact configures it.

#### Group B — load / size (5 axes)

| # | axis | symbol | measured range (source) | class | settable today? | feeds |
|---|---|---|---|---|---|---|
| B1 | tokens per rank per step (kernel-visible batch) | `T` | 4096 (serving, both cells); 4096/2048/1024 on the **tgen body only** (`WORKLOAD_EVIDENCE.md` W6) | EMERGENT (from C, ISL, bucket) | kernel: yes on tgen body, **NO on the serving M15 body** (fails MoK gate at T=2048/1024, rel 0.874/0.839, `REPORT.md:88`) | α, C_phase, B_bnd, λ, P |
| B2 | concurrency | `C_req` | **32 and 512 only, 16× apart, nothing between** (`WORKLOAD_EVIDENCE.md` §6) | LOAD | yes | φ, κ_sched, T |
| B3 | ISL | — | **4096 only** — and ISL == `max_num_batched_tokens` exactly (`BOTTLENECK_SIGNALS.md:337`) | LOAD | yes | φ *shape*, chunk multiplicity |
| B4 | OSL | — | **8 only** | LOAD | yes | prefill:decode step ratio ⇒ seal-coverable fraction |
| B5 | arrival process | — | **closed-loop saturated only**; `o50p/o75p/o90p` Poisson cells specified at `OPEN_LOOP_BASE_RATE=4.97 req/s`, never run (`WORKLOAD_EVIDENCE.md` H6) | LOAD | yes (driver exists) | φ, dummy rate, κ_sched |

**B1 is the axis with the worst provenance trap in the project** and it must be repeated
in the master doc: the famous T-sweep `0.7557 / 0.8845 / 1.0940` is campaign-grade but was
measured on the **tgen** body (sha `20b8c6bb`), which exists *because* the M15 chassis has
a T≠4096 defect that was never merged (`WORKLOAD_EVIDENCE.md` W6). Also `K0_MAXTOK == T`
in the harness ⇒ **the T-sweep is a batch-size sweep, not a fill sweep** (`REPORT.md:435-438`).

#### Group C — composition / raggedness (4 axes, all EMERGENT)

| # | axis | symbol | measured range (source) | class | settable today? | feeds |
|---|---|---|---|---|---|---|
| C1 | fill | `φ` | 0.3757 (c32p m15) / 0.4329 (c32p stock) / ≈0.985 (c512p m15); declines within a run 0.417→0.370 (`WORKLOAD_EVIDENCE.md` W1) | EMERGENT | **no** — needs `K0P6_M24_NORIG_TABLE`, and M24 has **never compiled** (H7) | α_real, B_bnd(DP), W_pad |
| C2 | fill *shape* | `F(φ)` | strongly **bimodal**: deciles of n_real = 0,0,5,1994,4096,4096,… ⇒ ~31 % near zero, ~60 % full, ~10 % partial (`WORKLOAD_EVIDENCE.md` W3) | EMERGENT | no | decides tier-1 vs tier-2 fill design |
| C3 | dummy / degenerate step rate | `δ` | 56.0 % (c32p m15 whole run), 49.3 % (c32p stock), 2 % (c512p), 31.2 % (corpus ramp window) (`WORKLOAD_EVIDENCE.md` W2) | EMERGENT | no | wasted-work fraction; and it is **rank-synchronous** (κ=1.000 over 28 pairs) so a step-level skip needs no synchrony discount |
| C4 | DP co-scheduling density | `κ_sched` | prefill chunk-events per in-bucket step **3.19 (m15) vs 3.93 (stock) of 8**; padded-work-per-real-token W = 2.568 vs 2.154 ⇒ **1.1922** (`WORKLOAD_EVIDENCE.md` W5) | EMERGENT | no | **an A/B validity gate, not a kernel term** |

**C2 is the most decision-relevant shape fact we own and it inverts the intuition that
fill is a dial.** Consequence already banked (`WORKLOAD_EVIDENCE.md` W3): under this
distribution a whole-step skip captures the entire modeled win and row-granular plumbing
adds nothing. **And the probable cause is an artifact** (SPECULATION, cheap to test): ISL
4096 == bucket 4096 ⇒ a DP rank holds a whole chunk or nothing. If so, every tier-1 sizing
number is cell-specific and dissolves at any other ISL (hole H4). This is why §4 puts an
ISL sweep in tier 2 despite it being "just a client flag."

**C4 must never enter the cost model as a kernel term.** `BANKED_LAWS.md` LAW-40: at
f≈0.44 the W=1.1922 inflation alone is worth +8.4 % wall — the *entire* measured c32p gap.
It is a **matching requirement on serving A/Bs** (`grounding/BANKED_LAWS.md` LAW-40) and,
separately, the largest optimisation in the ledger (LAW-50: 15–25 % of node throughput,
kernel-independent).

#### Group D — routing (5 axes)

| # | axis | symbol | measured range (source) | class | settable today? | feeds |
|---|---|---|---|---|---|---|
| D1 | aggregate expert popularity | `gini` | gini **0.609** aggregate, per-layer 0.68–0.83; **experts 0–7 carry 51.93 %** of routed slots (`WORKLOAD_EVIDENCE.md` W4a) | EMERGENT (data) | replayable via `K0_MOK_ROUTE_HIST` (corr 0.998 on the marginal) | κ_rank |
| D2 | expert→rank placement | `π` | contiguous ("linear"): rank max/mean **5.088×**, max/min 13.84×. Round-robin (`e%8`): **1.085×** (`WORKLOAD_EVIDENCE.md` W4a) | **OPERATOR** | yes (`rr_patch.py`, offline-verified; campaign 4 never ran) | κ_rank |
| D3 | per-call skew | `κ_call` | **UNRECONCILED**: m18diag p50 **5.15×** / p95 6.25× (n=90,050) vs routecap1 p50 **2.61×** / p90 3.35× (n=512) (`WORKLOAD_EVIDENCE.md` W4b) | EMERGENT | GPU replay is code-complete and **never executed** (H8) | κ, the skew tax |
| D4 | run correlation | `r_corr` | **FALSIFIED as first-order**: consecutive real rows share 57.0 % of experts vs 55.7 % shuffled — a 1.3-point excess (`WORKLOAD_EVIDENCE.md` W4c) | EMERGENT | replay rig exists, unrun | destination interleaving value (demoted) |
| D5 | EPLB | on/off | **MEMORY class only**, no in-repo receipt; the 128-redundant config violates `PF4H-CONTRACT-001` ⇒ **has never been composed with the megakernel** (`WORKLOAD_EVIDENCE.md` W4e) | OPERATOR | no (contract violation); EPLB0 designed, unrun | κ_rank |

**The single most important scoping fact about D:** routing skew as measured in serving is
**partly synthetic**. `BANKED_LAWS.md` LAW-42: vLLM pads with token id 0 ⇒ identical hidden
⇒ identical top-8 ⇒ **2.66× rank-load skew from padding alone**, rising to ~5× when two hot
experts share a rank — "quantitatively consistent with the measured per-call p50 of 5.15×
and with essentially nothing else." So D and C are **not independent axes**: fill drives
skew. Any grid that sweeps them orthogonally is sweeping a counterfactual.

#### Group E — model shape (6 axes, all MODEL-FIXED)

| # | axis | symbol | value here (source) | why it is in W anyway |
|---|---|---|---|---|
| E1 | routed experts | `E` | 256, 32/rank at EP8 (`k0pf6gm_device_tile_m15.hip:333` `K0P6_MAXE 64` capacity; 32 owned) | sets fan-in and `A` (protocol event count) |
| E2 | top-k | `k` | 8 (`k0pf6gm_device_tile_m15.hip:1230` rejects `TOPK != 8`) | **the exponent in the readiness law** — see Q6; this is the axis on which K3's sign flips |
| E3 | hidden | `d_model` | 7168 ⇒ **14,336 B/token bf16** (`K0P6_H 7168`; `m25_boundary_bench.hip:59` `kRowBytes = 14336`) | the byte constant in every B_bnd formula |
| E4 | expert intermediate | `d_ff` | 2048 (`SHARED_EXPERT_FILLER_DESIGN.md:40`, "same shape as one routed expert") | the FLOP constant in C_phase |
| E5 | shared expert | `n_sh` | 1, same shape as one routed expert (same line) | the **only** routing-independent filler; capped at ~1/8 of routed FLOPs (`TP8_STATE.md` B4) |
| E6 | layers | `L` | 61 total, 3 dense + 58 MoE (`TP8_OVERLAP_ANALYSIS.md:10`) | TP8 pays 2 collectives on **all 61**; DP pays all2all on 58 ⇒ 122 vs 116 comm events/step (`TP8_STATE.md` §4.3) |

#### Group F — precision (2 axes)

| # | axis | value / range | class | evidence |
|---|---|---|---|---|
| F1 | compute/storage precision | bf16 activations, fp8 block weights; `--kv-cache-dtype fp8` on our arms only, shipped default on the vendor arm (`campaign_v5/ARMS.md:90-100`, `:190-210`) | OPERATOR | `BANKED_LAWS.md` LAW-56: **verify precision at runtime, not from config** — the production FP8 training bar ran its MoE in BF16, and fp8-in-MoE was worth ~0 ms |
| F2 | wire precision | bf16 today. fp8-on-wire designed (**1.939×** fewer bytes) but **structurally blocked**: gfx950 has no fp8 remote RMW, so it is reachable only on top of skeleton S8, whose scratch pathology is P0 (`KNOB_INVENTORY.md` F6, `FP8_WIRE_DESIGN.md:12-29`) | OPERATOR (if built) | halves B_bnd — the only axis that moves B_bnd without moving anything else |

#### Group G — hardware (4 axes)

| # | axis | value (source) | class |
|---|---|---|---|
| G1 | arch | gfx950/CDNA4 (all serving); gfx942/MI300X microbench only. **Decisive:** gfx942 remote atomics ~3× slower than remote stores, 15 vs 44–46 GiB/s (`OVERLAP_ABSTRACTIONS.md:146`) | HW-FIXED (2 points, 1 of them microbench-only) |
| G2 | link / fabric | nominal 76.8 GB/s per link; **achievable 56.9–57.3 GB/s (74 %)**, reached by ~8 CTAs (`BANKED_LAWS.md` LAW-4). Node egress ceiling 349–355 GB/s is flagged **"unreproduced; treat as unknown"** (`OVERLAP_ABSTRACTIONS.md:147`) | HW-FIXED |
| G3 | CU / XCD geometry | 8 XCD × 32 CU, 4 MB L2/XCD, 256 MB Infinity Cache; **occupancy is exactly 1 block/CU** at megakernel budgets (`BANKED_LAWS.md` LAW-12) | HW-FIXED |
| G4 | thermal/clock heterogeneity | root-caused only in the **training** arm: GPU2 hottest (77 °C), 4–5 % clock spread, rank-structural 92 µs→4,353 µs wait spread (`WORKLOAD_EVIDENCE.md` W10c). **Zero serving points** (H16) | HW-FIXED, EMERGENT in effect |

#### Group H — deployment / instrument (the axes that are neither model nor load, and are the cheapest levers)

| # | axis | value / range | class | why it is in W |
|---|---|---|---|---|
| H1 | `max_num_batched_tokens` (the bucket) | **ours 4096**, vendor native **16384** (`campaign_v5/ARMS.md:96`, `:194`) | OPERATOR | *It is the bucket size.* Everything in (512, 4096] collapses into one B4096 key; capture list stops at 512 (`WORKLOAD_EVIDENCE.md` W5). Named as the largest-reach unpulled lever (`FILL_AWARE_DESIGN.md:1337-1345`); **never swept** (H12) |
| H2 | `max_num_seqs` | ours 128, vendor 2048 | OPERATOR | at "concurrency 512" the client offers 512 against a **128 admission cap** — interaction never characterised (`WORKLOAD_EVIDENCE.md` H14). **Verify before quoting c512p as C=512.** |
| H3 | prefix caching | ON in camp3/m23pair, OFF in b0/b3 | OPERATOR | changes φ directly; absolute tok/s across those campaigns are **not comparable**, only ratios (`ARMS.md:294-310`) |
| H4 | seal coverage | 0.77 % → 96.9–100 % of in-bucket, cell-dependent (`WORKLOAD_EVIDENCE.md` W9) | EMERGENT | **standing rule:** no serving delta may be interpreted without its arm's `sealed/in_bucket` |
| H5 | scheduling policy | fcfs | OPERATOR | drives κ_sched (C4) |

### 1.3 The sampling map, and a finding that falls straight out of it

`WORKLOAD_EVIDENCE.md` §6 states the honest summary: **two serving points on a
~8-dimensional space**, one deep (c32p) and one shallow (c512p). Laying the *arms* out on
the (topology × concurrency) plane makes something sharper visible:

| | DP8/EP8 + M15 (ours) | DP8/EP8 native (`native_tuned_dp`) | TP8+EP native (`native_tuned_tp`) |
|---|---|---|---|
| **C=32** | 20,371.6 tok/s (b0v5) | patched-stock only, n=2 directional (camp3) | **25,844.0** (b0v5) |
| **C=512** | 42,788.0 / 42,835.7 (b3cal, b3rev2) | 39,474.0 / 40,183.1 | **NEVER RUN** |

**DERIVED FINDING (new — not stated in any grounding brief).** The flagship
"topology optimum flips with concurrency" law is currently measured on a **diagonal**, not
a 2×2. We have TP8-native only at C=32 and DP-native only at C=512. The claim *"TP8 stops
being the right topology at high concurrency"* — which is the mechanism half of the
draft8 story — **has no measurement at all**; it is inherited from AMD's recommendation
table, whose own 128–512 gap is undocumented (`WORKLOAD_EVIDENCE.md` W8.1). One vendor-arm
run (TP8 native at c512p, ~25 min) upgrades the flagship law from an extrapolation to a
2×2. It is grid cell **E1** in §4.

**Second DERIVED caveat on the same pair.** The b0v5 arms do not share Group-H
coordinates: our arm runs `--max-num-batched-tokens 4096 --max-num-seqs 128`, the vendor
TP8 arm runs `16384 / 2048` (`campaign_v5/ARMS.md:96`, `:194`). That is by design — the
protocol compares *each configuration at its vendor-recommended tuning* — but it means the
−21.17 % is an **arm-level** delta at two different W points, and cannot be read as a
kernel delta. The three-way decomposition already required by
`BENCHMARK_PROTOCOL.md:22-29` is the only valid reading.

---

## 2. Q — the mediating-quantity layer (the core move)

### 2.1 The three-way split

Muddling these three is what makes cost models unfalsifiable. Keep them apart:

| layer | symbol | definition | who sets it |
|---|---|---|---|
| **workload mediators** | **Q** | derived scalars computable from W (and θ) **by arithmetic, before any kernel runs**. Independent of the schedule. | the workload |
| **schedule response quantities** | **R** | the quantities the kernel's knobs actually move: readiness parallelism P, unblockable fraction u(t), protocol op count A, scatter width L, injection depth d, reserved consumers C, quota. | the schedule S (`KNOB_INVENTORY.md` K1–K5) |
| **hardware constants** | **θ** | per-arch calibration: achievable link BW, atomic:store ratio, per-CTA drain rate, effective FLOP/s per phase-limiter class, protocol op cost. | the silicon; measured once per arch |

The cost model is then `M(Q, R(S); θ)` and the sufficiency claim (§2.4) is precisely that
**W enters M only via Q**. Note that `../grounding/BANKED_LAWS.md`'s preamble table mixes
Q and R (it lists `d`, `A`, `L`, `u(t)` alongside `φ`, `σ_rank`, `f`). That mixing is
exactly why "the mediating quantity" has felt slippery: `d` is a *knob*, φ is a
*workload*. Splitting them is the first useful thing this document does.

### 2.2 The Q vector — 10 coordinates

---

#### **Q1 — B_bnd: bytes crossing each comm boundary, per token per layer**

```
DP8/EP8 (per token, per MoE layer, aggregate remote fabric bytes):
   B_dispatch = E[remote distinct dest ranks] · d_model · 2
   B_combine  = m · d_model · 2,     m = k·(N−1)/N unfolded, = E[remote ranks] folded
   with E[distinct dest ranks] = N·(1 − C(E−E/N, k)/C(E, k))

TP8+EP (per token, per layer, aggregate, ring or two-shot — identical):
   B_AR = 2 · (2(N−1)/N) · N · d_model · 2   ... = 2 ARs/layer
   per-rank egress per AR = 2(N−1)/N · B_tensor
```

Evaluated at E=256, k=8, N=8, d_model=7168 (routing arithmetic exact at
`FP8_WIRE_DESIGN.md:58-66`: E[distinct]=**5.2942**, E[remote distinct]=**4.6324**,
E[remote partials]=**7.000**, fold factor 1.511×):

| topology | dispatch | combine | **total/token/layer** | layers paying | **DERIVED** |
|---|---:|---:|---:|---|---|
| DP8/EP8, unfolded combine | 64.9 KB | 98.0 KB | **162.9 KB** | 58 MoE | this doc |
| DP8/EP8, source-folded (m15b) | 64.9 KB | 64.9 KB | **129.7 KB** | 58 | this doc |
| TP8+EP (2 ring ARs) | — | — | **392.0 KB** | **all 61** | `M25_TP8_MEGA_DESIGN.md:98` |

⇒ **TP8 moves 2.41–3.02× more fabric bytes per token per layer than DP/EP, on 5 % more
layers.** (DERIVED.) This is the byte half of the topology story and it is *large*. It is
also the number that kills the M25 design doc's "−32 % fewer bytes" claim: `TP8_STATE.md`
§1.3 re-derives CDAR at **354 vs 392 KB = −9.7 %**, because ERS+MAG is a two-shot AR that
moves exactly ring's bytes at boundary 1. **CDAR's case is overlap, not bytes** — that
retraction is now confirmed independently by the arithmetic above.

**Crucially, B_bnd's argument differs by topology:**
`B_bnd^DP = f(T_padded) = f(T_real/φ)` — DP dispatches pad rows, unsliced: the router emits
ordinary top-k for pad rows and Mori dispatches `a1` as `[4096, 7168]`
(`FILL_AWARE_DESIGN.md:68-79` via `WORKLOAD_EVIDENCE.md` W1).
`B_bnd^TP = f(T_real)` — there is no DP lockstep padding, so there is no `execute_dummy_batch`
and no fill lever at all (`TP8_STATE.md` §3.3).
**This asymmetry is the fill law, stated at the byte level, and it is the sharpest version
of the draft8 mechanism the project has.**

**Selects:** K1 (transport op class, via message size), F2 (fp8-on-wire), the fuse/don't-fuse
decision F1.

---

#### **Q2 — C_phase: compute time per phase, from GEMM shapes**

```
FLOP/token/MoE-layer = 2 · 3 · d_model · d_ff · (k + n_sh)
                     = 2·3·7168·2048·9 = 792.7 MFLOP        (DERIVED)
C_phase(p) = FLOP(p) / θ_flop(p, limiter class)
```

Calibrated against the banked exp_33 stamp (`BANKED_LAWS.md` LAW-65: interior 5,852.1 µs,
M6 2,453.3, M7 2,701.8, plan 372.8, combine 324.2), node tokens = 8 × 4096 = 32,768:

| phase | FLOP share | node FLOP | per-GPU | measured µs | **θ_flop DERIVED** |
|---|---|---:|---:|---:|---:|
| M6 (gate+up) | 2/3 | 15.39 TFLOP | 1.924 TFLOP | 2,453.3 | **784 TFLOP/s** |
| M7 (down) | 1/3 | 7.70 | 0.962 | 2,701.8 | **356 TFLOP/s** |
| M6+M7 | 1 | 23.09 | 2.886 | 5,155.1 | 560 TFLOP/s |

**The 2.2× spread between the two phases' θ is the whole point.** A pure-FLOP C_phase
model would predict M6 = 2×M7; it measures 0.91×. `BANKED_LAWS.md` LAW-15 already names the
mechanism: **M6 is weight-streaming/latency-bound and M7 is CTA-throughput-bound**, and M7
additionally carries the transport epilogue (LAW-30: M7 and combine are anti-correlated at
r = −0.904 and must be judged as a sum). ⇒ **C_phase must be a per-limiter-class model,
and θ_flop is a function of (phase, schedule), not of the hardware alone.** That is the
single largest source of model error to guard against, and it is why §4 puts a
θ-calibration tier *before* any sweep.

**Selects:** everything, indirectly — C_phase is the denominator of ρ.

---

#### **Q3 — ρ: the compute:transport ratio (the hideability ratio)**

```
ρ = C_phase(co-resident compute available at the boundary) / X(collective time)
max hidable fraction of the collective = min(ρ, 1)      (DERIVED, from
   wall_fused ≥ max(C, X) and wall_phased = C + X)
```

| cell | C | X | **ρ** | max hiding |
|---|---:|---:|---:|---:|
| TP8 deployment (per layer) | 7.2–7.6 ms | 2.8 ms | **2.6–3.2** | 100 % |
| G25-1 rig, `k_inner=2048` | 1,439 µs (2 bursts) | 2,167–2,214 µs (1 boundary) | **0.650–0.664** | **65 %** |

**DERIVED FINDING (new).** The G25-1 gate's PASS bar is **≥90 % of the AR hidden**
(`M25_TP8_MEGA_DESIGN.md:181`). At the rig's own operating point the arithmetic ceiling is
`min(ρ,1) = 0.65`. **The gate could not have passed, at any schedule, for any knob
setting.** `TP8_STATE.md` §2.4 identified the ratio problem and correctly called M2 (a
`k_inner` sweep) the cheapest open experiment; it did not notice that the *bar itself* is
unreachable. That upgrades M2 from "cheapest" to "blocking": every G25-1 number, including
the four falsifications F1–F4, was produced against an impossible criterion.

**Selects:** whether to fuse at all (F1); K5 (C); the value of any filler (F2 shared
expert). **Threshold θ:** fusion can only pay when ρ ≳ 1; below that, `max(C,X) = X` and
the honest schedule is "run the vendor collective and stop paying protocol."

---

#### **Q4 — co: co-residency (binary, and the cleanest law we own)**

```
co = 1  iff the carrier issues fabric ops while an MFMA body occupies the same CUs
```

Under occupancy-1 (`BANKED_LAWS.md` LAW-12: 256 VGPR + 256 AGPR + 155,428 B LDS force one
block/CU) there are no spare waves, so `co` is a property of the *phase*, not of a
scheduler heuristic.

**Measured both ways, which is what makes it a law and not a preference:**
`co=1` → injection depth 4 is worth **−615.0 µs** (7,110.8 → 6,495.8, t = −80.2) and depth
16/32 sit at the unthrottled cost (`BANKED_LAWS.md` LAW-20/21).
`co=0` → depth is **FLAT** across {4,8,16,32}: towers 93.6/93.4/93.3 GB/s, atomics
67.1/67.1/67.0 (`g25_0b_v1_results.txt`, commit `108def6e`).

⇒ **Injection depth is congestion control against co-resident compute, not against the
fabric.** This is already the campaign's model law; Q just gives it a name and a place.
Note also `BANKED_LAWS.md` LAW-7 (making the pusher faster *imports* fabric cost — 519 µs
surcharge at 5.96× when driven flat out) and LAW-22 (pacing pays iff the paced engine is
not the only thing in flight) are the same `co` law seen from two other sites.

**Selects:** K4 depth; transfer-unit size; MLP depth (LAW-7 corollary: these are injection-rate
knobs in disguise whenever `co=1`).

---

#### **Q5 — Σ: the slack map (where idle CU-time exists, per phase, per rank)**

```
Σ(phase) = (CTAs that can be removed without moving the phase) × phase duration   [CTA·ms]
```

This is the `BANKED_LAWS.md` LAW-53 bubble-ledger method, transferred. Measured instances:

| site | slack | source |
|---|---|---|
| M6 (DP/EP prefill) | removing **62 of 256 CTAs** moves M6 by **+0.5 %** (flat across C=2..64) ⇒ **Σ(M6) ≳ 62 × 2.4533 ms = 152 CTA·ms** (DERIVED) | `BANKED_LAWS.md` LAW-15 |
| M7 (same) | same removal moves M7 by **+27 %, linear in C** ⇒ **Σ(M7) ≈ 0** | LAW-15 |
| price of *taking* slack | reserve-but-idle costs **8–9 µs per CTA, monotone** ⇒ at C=28 that is ~238 µs of pure capacity tax (DERIVED) | LAW-27 |
| value of *using* slack | consuming a certified prefix in the producer's shadow: **−93 µs (C 8→16), −444 µs (C 16→24)** | `OVERLAP_ABSTRACTIONS.md:30` |
| training cross-check | in-launch bubble supply ~**150–180 CTA·ms** per backward launch vs wgrad demand ~3,360 CTA·ms | LAW-53 |

**The 152 CTA·ms (serving M6) and 150–180 CTA·ms (training backward) agreeing is a
coincidence worth naming as a hypothesis, not a law** (SPECULATION): at occupancy 1 with a
256-CTA grid, the in-launch slack of a megakernel appears to be an *order-of-magnitude
constant* ~10⁻¹ of the launch's CTA·ms budget, and any filler larger than that loses. Both
programs falsified every filler exceeding it.

**The placement corollary, which is the most repeatable mistake in the ledger** (LAW-15):
*"the old design reserved exactly at the boundary where the free phase ends and the starved
phase begins."* Σ is the map that prevents repeating it.

**Selects:** K5 (C and where to reserve), F2 (shared-expert filler — capped at ~1/8 of
routed FLOPs, so it fits Σ where wgrad's 13 ms could not), and skeleton choice S2 vs S4.

---

#### **Q6 — P and u(t): readiness parallelism and the unblockable fraction**

```
P   = number of independently early-executable consumer units per epoch
u(t)= fraction of consumer work executable by producer-progress t
t50 = producer progress at which the median consumer unit becomes executable
```

Under *natural* (tile-major) producer order with a top-k fan-in, each consumer token needs
all k partials, so `P(ready by t) = (t/S)^k` and

```
t50 = 0.5^(1/k)
   k=1 → 50.0 %   k=2 → 70.7 %   k=4 → 84.1 %   k=8 → 91.7 %   k=16 → 95.8 %   (DERIVED)
```

The k=8 value **reproduces the measured 91.7 %** exactly
(`aug11/OVERLAP_METHODOLOGY_STUDY.md:277`; the tile-major variant is recorded as 93.9 % at
`order.cuh:21-24`). ⇒ **top-k is a workload axis that sets a schedule law's sign.**
`BANKED_LAWS.md` LAW-23 already says it in words — *"at top-k 1–2 the same fine signals
would have a real shadow; at top-k 8 they do not"* — and calls it "the clearest example in
the ledger of a schedule knob whose sign flips on a workload parameter." The formula makes
it a threshold: fine-grained readiness signalling pays when `0.5^(1/k)` leaves a usable
shadow, i.e. roughly **k ≤ 2**.

**P by topology and skeleton (DERIVED):**

| configuration | P | why |
|---|---:|---|
| DP/EP M15, S=2 slabs | **2** consumable prefixes (×8 sources = 16 certificates) | `K0P6_M15_SLABS 2` at `k0pf6gm_device_tile_m15.hip:498-500`; S=2 is the only value where 448-column nc chunks and M8's 1,024-B chunks co-align |
| TP8 CDAR, 16,384 tok / 256-row slabs | **64** | `g25_0b_slab_sweep.txt` |
| unfused collective (S0) | **1** | the boundary degenerates to a phase barrier |

**This explains the K3 contradiction the knob inventory flagged as needing an experiment
cell.** M15 says granularity is free (a 64× signal-count change measured ~nothing,
`slab.cuh:10-12`); CDAR says it is worth **2.1×** (128 rows 116.0 / 256 rows 117.2 / 512
rows 93.4 / 1024 rows 55.4 GB/s). Under Q the two are not in conflict: in M15, changing
signal count at fixed P=2 changes only `A` (Q9) — free; in CDAR, changing `slab_rows`
changes P *and* the store-tower geometry *and* the gather width simultaneously. **Law
candidate:** *readiness granularity is free when it moves only A; it is expensive when it
also moves P or the transfer geometry.* (This is the reconciling hypothesis stated as
SPECULATION in `KNOB_INVENTORY.md` B.3; Q makes it a falsifiable decomposition — hold
geometry fixed and vary signal count only.)

**Selects:** K2 (producer order — it is what makes "early-executable" exist), K3 (slab
count/extent), K5 (there is nothing for a consuming pool to do if P=1).

---

#### **Q7 — κ: skew (hottest-rank load ÷ mean)**

```
κ_rank(π, gini)   = max_r load(r) / mean_r load(r)     — placement-dependent, aggregate
κ_call            = per-call version; the distribution, not the mean, is what costs
κ_pad             = skew contributed by degenerate pad routing alone
```

| instance | κ | source |
|---|---:|---|
| aggregate, contiguous placement | **5.088×** (max/min 13.84×) | `WORKLOAD_EVIDENCE.md` W4a |
| aggregate, round-robin `e%8` | **1.085×** | same |
| per-call p50 (m18diag, n=90,050) | **5.15×** (p95 6.25×) | W4b |
| per-call p50 (routecap1, n=512) | **2.61×** (p90 3.35×) | W4b — **unreconciled, ~2×** |
| ideal *static* redistribution floor | p95 still **3.05×** | `RR_PLACEMENT_NOTES.md:246-262` |
| from padding alone | **2.66×**, →~5× with two hot experts on one rank | `BANKED_LAWS.md` LAW-42 |
| TP8 boundary 1 | **1.000 by construction** — every rank owes every slab the same bytes | `TP8_STATE.md` §4.2 |

**κ is worth 3.6× wall in the kernel rig** — balanced 5,823 µs → aggregate-histogram
20,739 µs (`BANKED_LAWS.md` LAW-39 companion / `M18_REPLICATION_RESULTS.md:24-31`) — which
is *far larger than any schedule knob in the entire ledger*. And **it has never been
driven as a controlled axis on the kernel harness**: `K0_SYNTH_ROUTE` is rejected under
`mok_synthetic`, and the only reachable knob (seed) spans destination-load CV
0.0034–0.0086, 4–15× *less* imbalance than COMET's 0.032 (`BANKED_LAWS.md` LAW-62b). The
pre-registered claim "skew is the one regime where a service pool may re-enter" is
therefore **untestable, not refuted** — and it is the single largest open question about
whether M15's producer/consumer is right "for all sizes."

**Selects:** K2 (`source_interleaved`/m17 rotation), K6 (per-destination credits — the
highest-value *unbuilt* knob, gated on a skew harness existing, `credit.cuh:104-106`),
placement π, replication (M18/M19/M20), and — via LAW-52's critical-rank law — *whether any
schedule change moves e2e at all*.

---

#### **Q8 — φ and W_pad: fill, and padded work per real token**

```
φ      = real rows / rows the kernel executes           (per rank, per in-bucket step)
W_pad  = padded MoE rows per real token = 1/φ · (in-bucket step inflation)
cost per real token = (a + b·T_pad) / (φ · T_pad)
```

The last line is the load-bearing one: **fill and tokens-per-rank enter through the same
expression**, and they are the same mediating quantity seen twice. Evaluated on the
tgen-body fit of §3.2 (a=1,068 µs, b=1.1623 µs/token for M15; a=181 µs, b=1.8387 for
production):

| cell | φ | M15 µs / real token | production µs / real token | ratio |
|---|---:|---:|---:|---:|
| c32p (φ=0.3757) | 0.376 | **3.788** | 5.012 | 0.756 |
| c512p (φ=0.985) | 0.985 | **1.445** | 1.912 | 0.756 |

**DERIVED, and it is an important negative:** at fixed T_pad, **fill does not change the
DP kernel *ratio* at all** — both arms pay the padding identically (`BANKED_LAWS.md` LAW-42;
`VLLM_MOE_SKIP_PADDING` defaults to 0). Fill changes the **absolute** per-real-token cost
of *the whole DP topology* by 2.62×, and TP8 does not pay it. **So the honest mechanism for
the C=32/C=512 flip is not "our kernel is better when full" — it is "the DP topology's
per-real-token cost inflates by 1/φ and TP8's does not."** That is a *topology* law
mediated by φ, and it re-frames the draft8 headline correctly. It also predicts something
testable: a *fill-aware* kernel (M24) is not gap-closing versus production — it is
gap-closing versus **TP8**, by attacking the 1/φ term that only DP pays.

**Selects:** F10 (seal boundary), M24 fill-awareness, quota sizing (`flush_rows` is tuned
to slab-1's shadow at nbatches=1,024, i.e. T=4096 — explicitly de-tuned elsewhere,
`k0pf6gm_device_tile_m15.hip:2122-2127`).

---

#### **Q9 — A and L: protocol intensity and scatter width**

```
A = protocol atomic/flag operations per rank per epoch
L = cache lines touched per protocol event
```

Measured anchors: ~535,040 arrivals/rank/epoch in the mode-2 protocol
(`aug10/KERNEL_BOTTLENECKS.md:175-179`); ~1.58 M atomics/rank/epoch total, of which exp_14
removed 698,112 for **−306 µs** ⇒ **~0.44 µs per thousand atomics** (`BANKED_LAWS.md`
LAW-17), with a hard ceiling: removing *every* remaining arrival atomic is worth ≤235 µs
against +1,249 µs outstanding, **≤19 %**. M15's slab certificates replaced ~926,000
protocol ops with 2×8 words (`slab.cuh:6-9`).

Two counter-intuitive laws ride here and both are workload-parameterised:
* **Only reducing A helps; relocating cannot** (LAW-17).
* **Scattering is a feature**: coalescing 32 lanes' counters onto 2 cache lines instead of
  32 was **≥10× slower** (abandoned after 12 min vs a usual 25–95 s) because an atomic
  resolves *at the line* (LAW-18). ⇒ **L should be maximised, the opposite of the load rule.**

`A` scales with rows × chunks × contributing blocks — set by the *plan*, hence by
(T, k, E, tile size), not by the schedule. That makes A a genuine Q coordinate even though
the knobs that reduce it live in S.

**Selects:** K3 (signal class), and it is the term that makes fine granularity look free in
M15 (Q6).

---

#### **Q10 — λ and I_bnd: latency-vs-bandwidth regime, and boundary arithmetic intensity**

```
λ      = message size per protocol unit / knee size        (λ ≫ 1 ⇒ bandwidth-bound)
I_bnd  = FLOP per byte crossing the boundary
```

Measured knees: pair knee **256 KiB**; fan-out knee **64 KiB/peer (push)** vs **1 MiB
(pull)**; **pull collapses past 256 CTAs while push degrades gently**
(`OVERLAP_KERNEL_DESIGN_ADDENDUM.md:317-321`). Chunking collapses below ~2K tokens
(`M25_TP8_MEGA_DESIGN.md:212-215`).

**The regime flip is measured, at one point:** CDAR at 256 tokens / 64-row slabs / 3.67 MB
gives **801 µs (atomic) vs 1,017 µs (towers)** = 4.6 / 3.6 GB/s effective — latency-dominated,
and **store-towers LOSE** (`TP8_STATE.md` §5 M5). At 235 MB the same comparison is
117.2 (towers) vs 67.1 GB/s (atomics). ⇒ **K1 is message-size-dependent, i.e. a second
manifest axis, not a global decision.**

I_bnd (DERIVED, per token per MoE layer, using Q1 and Q2's 792.7 MFLOP):
DP unfolded **4,754 FLOP/B**; DP folded **5,968**; TP8 **1,975**. TP8's boundary is
**2.4× less arithmetically intense**, which is the structural reason its collective is
exposed at all.

**Selects:** K1 (transport op class), push/pull direction, K3 slab extent, and the entire
decode-cell design (`TP8_STATE.md` B8: decode is *a different mechanism*, one-shot/LL AR,
not a smaller version of prefill).

---

### 2.3 θ — the hardware constants (calibration targets, with current status)

| θ | value | status | consumer |
|---|---|---|---|
| achievable per-link BW | 56.9–57.3 GB/s (74 % of 76.8 nominal), reached by ~8 CTAs | MEASURED (LAW-4) | Q1→time |
| node egress ceiling | 349–355 GB/s | **flagged "unreproduced; treat as unknown, measure first"** (`OVERLAP_ABSTRACTIONS.md:147`) | the 58 %-vs-79 % efficiency framing in `TP8_STATE.md` §2.3 — **re-measure before quoting** |
| atomic:store ratio | gfx950 ≈ 1.0 at 4 B coalesced (52.8 vs 54.9 GB/s) but **0.57 at 235 MB** (67 vs 117); gfx942 **≈0.33** (15 vs 44–46 GiB/s) | MEASURED, arch-dependent, size-dependent | K1 |
| scatter cliff | coalesced 52.8 → scattered **4.1 GB/s (13×)** | MEASURED | layout, always |
| per-CTA consume rate | **6.1 GB/s/CTA** (exp_29) | MEASURED | quota sizing (`flush_rows`) |
| protocol op cost | ~0.44 µs per 1,000 atomics | MEASURED, one regime | Q9 |
| θ_flop per limiter class | M6 784, M7 356 TFLOP/s at this shape | **DERIVED here**, schedule-contaminated | C_phase |
| no live xGMI counter | `amd-smi --xgmi` = N/A, `--shownodesbw` = 0-0 | MEASURED (LAW-11) | **bounds what any calibration can promise** |

### 2.4 The central claim, stated so it can be killed

> **H-Q (SUFFICIENCY OF Q).** For the megakernel family in this repo, the optimal schedule
> depends on the workload only through
> `Q = (B_bnd, C_phase, ρ, co, Σ, P/u, κ, φ, A/L, λ/I_bnd)`.
> Formally: `S*(W₁) = S*(W₂)` whenever `Q(W₁) = Q(W₂)`, to within the measurement
> resolution of the boundary rig (≈3 % end-to-end screen delta, ≈0.09 % campaign
> reproducibility — `BANKED_LAWS.md` LAW-60).

**Three pre-registered falsifiers.** Each is a *matched-Q, mismatched-W* pair; if the
optimal knob setting differs across the pair, H-Q is false and the extra coordinate that
explains it must be added to Q (and named in the paper).

| falsifier | construction | matched | mismatched | prediction under H-Q |
|---|---|---|---|---|
| **FQ-1 (fill vs batch)** | (a) T_pad=4096, φ=0.5 via `K0P6_M24_NORIG_TABLE`; (b) T_pad=2048, φ=1.0 | φ·T_pad = 2,048 real rows; same κ, same k | φ, T_pad, A | **C\*** and **depth\*** identical. If they differ, φ and T are not one quantity and Q8 is wrong. |
| **FQ-2 (co-residency)** | CDAR transport at fixed message size, with and without a co-resident MFMA body of matched duration | B_bnd, λ, P, A | `co` | depth flat in one, cliff in the other — **already half-measured** (LAW-8 vs LAW-20); the missing half is doing both *on the same rig* |
| **FQ-3 (granularity decomposition)** | vary K3 signal count at **fixed** transfer geometry, then vary transfer geometry at **fixed** signal count | — | A alone / P+geometry alone | the first is free (M15's result), the second is 2.1× (CDAR's result). If the *first* is expensive, Q9 is not separable from Q6 |

**Known threats to H-Q, stated up front** (these are the honest reasons it might fail):

1. **Non-separability is already proven for at least one pair.** LAW-21: carrier relocation
   and injection bound are *not* additive (+210.6 µs regression without the bound). So M
   must be evaluated on joint cells of R, and any waterfall figure is wrong for that pair.
   H-Q is about *W→Q*, not about R being additive — but a reviewer will conflate them.
2. **Substitutes.** LAW-37: the injection bound (−613.5 µs) and coarse readiness (−573.3 µs)
   may be bounding the same resource; never measured on top of each other.
3. **Codegen is a hidden coordinate.** LAW-32: merely making mode-14 code *reachable* cost
   the mode-12 ratchet **+726.9 µs** with mode-12 source untouched. A manifest-driven knob
   space will silently re-time arms whose knobs it never touched. ⇒ every arm must carry a
   `.text`-sha or mechanism-invariant gate (LAW-58/59). **This is the biggest threat to the
   whole "hundreds of kernels as points in a knob space" plan.**
4. **An unidentified floor.** LAW-16 residual: a g-independent **+1,159 µs** of interference
   whose mechanism is unknown; only 36 % is localised. If that floor moves with a workload
   axis not in Q, H-Q fails and we will not know why.
5. **Nuisance terms that are real but not kernel-side**: κ_sched (C4) and coverage (H4) can
   each swamp any kernel effect. They belong in the *validity gate*, not in Q.

### 2.5 The shape of M(Q, R; θ)

Not the full cost model (that is a separate deliverable), but the composition rule Q forces:

```
wall = Σ_phases  max( C_phase(p),  X_exposed(p) )                     [makespan, not sum]
X_exposed(p) = X(p) · (1 − hidden(p)),   hidden(p) ≤ min(ρ(p), 1)
X(p) = B_bnd(p) / BW_eff(K1, λ, layout)  +  A(p)·c_atomic  +  stall(d, co)
consumer_start(p) = t50(k, K2) · C_phase(producer)                     [Q6]
capacity_tax = C · 8.5 µs   ;   consumption_credit = f(Σ, P, quota)    [Q5, LAW-27/30]
e2e = makespan of the CRITICAL RANK, = the rank maximising κ · (thermal drift)  [LAW-52]
```

Four properties this skeleton must have, each forced by a banked law:
* **max, not sum** — LAW-64: the estimand is synchronised global makespan of the complete
  dependent DAG.
* **critical-rank evaluation** — LAW-52: only work removed from the hottest/slowest rank's
  wall moves e2e.
* **coupled M7+combine** — LAW-30: r = −0.904; the objective is their sum.
* **no additive waterfall across (carrier, depth)** — LAW-21.

---

## 3. The collapse, computed on the cells we already have

### 3.1 Q across eight measured cells

| Q | c512p DP/EP (**win**) | c32p DP/EP (**loss**) | TP8+EP deployment | G25-1 rig | CDAR pure transport | T-sweep 4096 | T-sweep 1024 | gfx942 |
|---|---|---|---|---|---|---|---|---|
| **B_bnd** /tok/layer | 130–163 KB × 58 | same × 58, but on **2.66× more rows** | **392 KB × 61** | 235 MB/boundary | 235 MB | 163 KB | 163 KB | — |
| **C_phase** | 5.2 ms interior (M6+M7) | same | 7.2–7.6 ms/layer | 1,439 µs (2 bursts) | 0 | 5,829 µs | 2,258 µs | — |
| **ρ** | n/a (no exposed collective) | n/a | **2.6–3.2** | **0.65** | 0 | n/a | n/a | — |
| **co** | 1 | 1 | 1 (intended) | 1 | **0** | 1 | 1 | 1 |
| **Σ** | Σ(M6) ≳152 CTA·ms, Σ(M7)≈0 | same | unmeasured | unmeasured | n/a | 152 CTA·ms | scales down with T | — |
| **P** | 2 | 2 | 64 | 64 | 64 | 2 | 2 | — |
| **t50** | 91.7 % (k=8) | 91.7 % | 91.7 % | — | — | 91.7 % | 91.7 % | — |
| **κ** | 5.088× aggr. / 2.61–5.15× per call | same + **2.66× from padding** | **1.000** at bnd 1 | 1.000 | 1.000 | 1.0 (balanced synth) | 1.0 | — |
| **φ** | **0.985** | **0.376** | **n/a — does not exist** | n/a | n/a | 1.0 | 1.0 | — |
| **A** | ~1.58 M/rank/epoch (pre-M15) → 16 words | same | 2-level, 8 words/slab | same | same | — | — | — |
| **λ** | ≫1 | ≫1 | ≫1 (3.67 MB slabs) | ≫1 | ≫1 | ≫1 | ~1 (chunking collapses <2K) | — |
| **α** (fixed share) | 0.183 | 0.183 | — | — | — | **0.183** | **0.473** | — |
| **θ atomic:store** | 0.57 @235 MB | 0.57 | 0.57 | 0.57 | 0.57 | — | — | **0.33** |

### 3.2 Six retrodictions

**R1 — the T=1024 loss.** Fit `cost = a + b·T` to the exp_04_tgen campaign
(`git show 43291977:.../aug13/exp_04_tgen/result.md:226-300`; M15-C=28 p50
5,828.8 / 3,411.8 / 2,258.2 µs, production 7,712.6 / 3,857.4 / 2,064.1) on the T=4096 and
T=1024 endpoints and check the midpoint (DERIVED):

| arm | a (fixed, µs) | b (µs/token) | predicted @2048 | measured @2048 | error |
|---|---:|---:|---:|---:|---:|
| M15 C=28 | **1,068.2** | 1.1623 | 3,448.4 | 3,411.8 | **+1.1 %** |
| production | **181.3** | 1.8387 | 3,946.9 | 3,857.4 | **+2.3 %** |

Both linear fits hold within ~2 % on a point they were not fitted to. ⇒ **the whole T-axis
collapses to one number: α = a/(a+bT), the fixed-cost share.**

| T | α (M15) | α (production) | measured ratio |
|---:|---:|---:|---:|
| 4096 | 0.183 | 0.024 | 0.7557 |
| 2048 | 0.310 | 0.046 | 0.8845 |
| 1024 | **0.473** | 0.088 | **1.0940** |

Crossover: `T* = (a_m − a_p)/(b_p − b_m) = 886.9 / 0.6764 = **1,311 tokens/rank**`, at which
`α_m = **0.412**`. ⇒ **Law candidate: the fused megakernel wins iff its fixed-cost share
α < ≈0.41.** The fixed cost is nameable — ~415 µs plan phase, four pre-M6 grid barriers,
dispatch, per-epoch protocol setup (`BANKED_LAWS.md` LAW-39 / LAW-31).

**DERIVED DISCREPANCY, flagged rather than smoothed:** `WORKLOAD_EVIDENCE.md` W6 quotes
break-even ≈ **T 1,600–1,800**. Two independent interpolations here give **1,311**
(cost-fit) and **1,398–1,483** (ratio interpolation, log- and linear-in-T). The three
methods disagree by ~25 %. Since the crossover T is one of the most quotable numbers the
campaign could produce, **it must be measured directly rather than interpolated** — grid
cell K1 in §4.

**R2 — the c32p/c512p flip.** Under Q8, DP's per-real-token cost is `(a+b·T_pad)/(φ·T_pad)`
while TP8's collective scales with real tokens. At φ=0.376 the DP topology pays **2.62×**
per real token relative to φ=0.985 (DERIVED, §2.2 Q8), and the *ratio between DP arms is
unchanged* — both pay the padding. ⇒ Q retrodicts the sign of the flip and, importantly,
**corrects the mechanism**: it is a topology tax mediated by φ, not a kernel-quality
effect. This is consistent with the speaker notes' own caveat that the two φ figures "come
from different serving cells, not a controlled fill-only ablation"
(`WORKLOAD_EVIDENCE.md` W8.1).

**R3 — TP8's AR bytes scale with batch.** Q1: `B_AR = 2(N−1)/N · T · d_model · 2` per rank
per layer, linear in T with **no fixed term**, while DP's all2all is linear in T_pad. At
low concurrency T_real is small ⇒ TP8's collective shrinks with the batch while DP's stays
pinned at the bucket. Q therefore predicts what draft8 asserts (`../BRIEF.md:45-46`)
without needing a new mechanism.

**R4 — depth flat vs depth-cliff.** Q4 (`co`) alone separates LAW-8 from LAW-20. No other
coordinate differs between the two rigs at the message sizes involved.

**R5 — the C-optimum flip (mode 12 wanted C≤8; M15 wants C=24–28).** Under Q5+Q6: C buys
`Σ`-consumption only if `P ≥ 2` *and* a certified prefix exists early. Mode 12 has no
certificate ⇒ P=1 ⇒ every reserved CTA is pure capacity tax at 8–9 µs/CTA ⇒ C*→0. M15 has
P=2 with an nc-major order ⇒ consumption credit −93/−444 µs dominates the tax until the
prefix is exhausted, then it inverts (C=32 unstable). ⇒ **C* is a derived quantity of
(Σ, P, t50), never an input** — which is exactly what `OVERLAP_ABSTRACTIONS.md:117-123`
says when it ships "the response-curve method, never a default."

**R6 — K1 flips twice.** Once on λ (towers win at 235 MB, **lose** at 3.67 MB) and once on
θ_atomic:store (gfx950 0.57 vs gfx942 0.33). Both are Q/θ coordinates, neither is a
schedule preference. This retrodicts the F5 falsification (`TP8_STATE.md` §3.1) *and*
explains why the design doc got it wrong: it selected on the small-message arch card
(52.8 vs 54.9 GB/s), i.e. it evaluated θ at the wrong λ.

### 3.3 Four findings that fell out of the arithmetic (new here)

1. **The G25-1 PASS bar was arithmetically unreachable.** ρ_rig = 0.65 ⇒ max hiding 65 % vs
   a ≥90 % bar (§2.2 Q3). Every G25-1 verdict, including F1–F4, was scored against an
   impossible criterion. **Consequence:** re-run the `k_inner` sweep *before* re-scoring any
   of those four hypotheses, and restate the bar as `≥0.9·min(ρ,1)`.
2. **The topology law is measured on a diagonal, not a 2×2** (§1.3). TP8-native at C=512 has
   never been run; the "TP8 stops winning at high concurrency" half of the flagship story is
   inherited from a vendor recommendation table with an undocumented 128–512 gap.
3. **TP8 moves 2.4–3.0× more fabric bytes per token per layer than DP/EP** (Q1, DERIVED) —
   independently confirming the `TP8_STATE.md` §1.3 retraction of the "−32 % fewer bytes"
   claim from the opposite direction, and quantifying why the TP8 collective is exposed at
   all (I_bnd 1,975 vs 4,754 FLOP/B).
4. **C_phase cannot be a FLOP model.** θ_flop differs 2.2× between M6 (784 TFLOP/s) and M7
   (356) at the *same* shape on the *same* silicon (§2.2 Q2). Any cost model that prices
   compute by FLOPs will mis-place the reservation boundary — the exact failure LAW-15 calls
   "the single most repeatable placement mistake in the ledger."

### 3.4 Where Q is currently silent (and must be declared silent)

* **Decode.** No rig has ever measured it (`BANKED_LAWS.md` §8 item 4). Q6's t50, Q10's λ
  and Q3's ρ all have decode branches that are *formulas without measurements*. The
  megakernel's 4.24× TPOT p50 win at c512p (757 vs 3,209 ms) is **entirely unexplained** by
  anything in Q.
* **Tail latency.** Q predicts makespan. It says nothing about why M15 wins **every p99** at
  c32p (TTFT/TPOT/e2e by 23/24/33 %, `TP8_STATE.md` §1.5) while losing every median. The
  mechanism offered there — DP's eight independent engines decorrelate head-of-line
  effects — is SPECULATION and is not in Q. **This is a gap in the metric map, not in the
  kernel model, but it is the gap a C=32 TTFT-p99-SLO deployment would decide on.**
* **The 64 % unidentified interference floor** (LAW-16 residual).
* **κ_sched and coverage** — deliberately excluded from Q as validity gates.

---

## 4. The reduced experimental grid

### 4.1 The design rule

Do **not** sample corners of W (21 axes, most unsettable). Sample corners of **Q** (10
coordinates, of which 6 are cheaply reachable on a boundary rig). A cell earns its place
only if it moves a Q coordinate that no cheaper cell moves, **or** it breaks a collinearity
that the current sample cannot separate.

**The collinearities that must be broken** (these are why the two serving cells cannot
support the flagship law on their own — `WORKLOAD_EVIDENCE.md` W8.1 says the two cells
"differ in ~six variables at once"):

| collinearity in today's data | breaking cell |
|---|---|
| φ ⟂ C_req ⟂ κ_sched ⟂ variance envelope (all move together between c32p and c512p) | **K3** (synthetic fill at fixed T) and **E3** (ISL sweep at fixed C) |
| φ ⟂ T (K0_MAXTOK == T makes every T-sweep a batch sweep) | **K3** via `NORIG_TABLE` |
| κ ⟂ φ (padding *generates* skew, LAW-42) | **K2** (route replay with pad rows excluded) |
| ρ ⟂ everything in TP8 (one operating point) | **K4** (`k_inner` sweep) |
| topology ⟂ concurrency (the diagonal, §1.3) | **E1** |

### 4.2 Settability audit (LAW-62 is binding: prove a knob is settable before pre-registering an axis)

| axis | actuator | status | blocking work |
|---|---|---|---|
| T | `K0_T` / `K0_MAXTOK` | **tgen body only**; serving M15 fails the MoK gate at T=2048/1024 | port the tgen M2 hole-sentinel fix into the M15 body, or run the axis on tgen and label it |
| φ | `K0P6_M24_NORIG_CONST/_TABLE` | designed, **M24 has never compiled** | compile M24 behind default-0 macros; `_ZERO_PAD` is mandatory with `_FILL` (`#error`) |
| κ (aggregate) | `K0_MOK_ROUTE_HIST` | **works** (corr 0.998 on the marginal) | none — but it destroys per-chunk composition |
| κ (per-call, real) | `route_replay.py` captured-vs-shuffled | **code complete, 22 CPU tests pass, NEVER EXECUTED ON GPU** | one run; `mok_eager` only by design |
| κ (synthetic std) | `K0_SYNTH_ROUTE` | **REJECTED** under `mok_synthetic`; reachable CV 0.0034–0.0086 | do not pre-register this axis on that harness |
| placement π | `rr_patch.py` | offline-verified, campaign 4 never ran | one serving campaign |
| ρ | `k_inner` argv | **works, one value** | one argv |
| λ | `tokens`/`slab_rows` argv | works | + RCCL reference at decode sizes (`rccl-tests` absent, but the rig links `-lrccl`) |
| co | phased/fused arm selection | works | none |
| arch | gfx942 node | microbench only | schedule access |
| bucket size | `--max-num-batched-tokens` | **works, never swept** | none — cheapest unpulled lever in the project |
| EPLB | `num_redundant_experts=0` | designed; 128-redundant **violates PF4H-CONTRACT-001** | run `eplb0_predict.py` first; its own gate says skip the arm if worst-layer relief < ~15 % |

### 4.3 Tier 0 — calibration (must precede everything; ~1 node-hour)

| # | measurement | fixes | why blocking |
|---|---|---|---|
| **T0-a** | node egress ceiling + per-link BW re-measure | θ | the 349–355 GB/s denominator is flagged unreproduced; the whole 58 %-vs-79 % efficiency framing rests on it |
| **T0-b** | device phase ledger on `m25_boundary_bench.hip` (`TP8_STATE.md` M1) | Σ, and which of 4 walls | *"one instrumented run replaces the next four guesses"*; bake in the running-maxima reset (LAW-63c) |
| **T0-c** | θ_flop per limiter class: M6/M7 stamps at 2–3 shapes | C_phase | §3.3 finding 4 — a FLOP model mis-places the reservation |
| **T0-d** | profiler confirmation of the 2.8 ms/layer exposed AR on native TP8 (`TP8_STATE.md` M6) | ρ | *"until it is, '28 % of the step is all-reduce' is a model, not a measurement"* |
| **T0-e** | reconcile the two per-call skew instruments (5.15× vs 2.61×) | κ | **CPU-only**, and it changes the modeled skew tax by 2× |

### 4.4 Tier 1 — boundary cells (the precision instrument; a 5-rotation campaign is ~3 min 5 s, LAW-60)

| # | cell | Q corner it pins | actuator | law it establishes | kill criterion |
|---|---|---|---|---|---|
| **K1** | T ∈ {4096, 2048, 1536, 1311, 1024, 512} × {M15, production}, tgen body pinned | **α** low→high | `K0_T` | *the fused mega wins iff α < θ_α ≈ 0.41*; measures the crossover directly instead of interpolating three disagreeing estimates | if the measured crossover is outside [1,100, 1,900] the linear cost model is wrong and α is not the mediator |
| **K2** | routing ∈ {balanced, aggregate hist, worst layer, **captured**, **shuffled-captured**} × K2-order × K5-C | **κ** low→high | `K0_MOK_ROUTE_HIST` + the unrun GPU replay | *does a service pool re-enter under skew?* — the one pre-registered claim that is untestable rather than refuted (LAW-62b) | if captured ≈ shuffled to within 3 %, run-correlation is dead at kernel speed too and mechanism (a) is closed |
| **K3** | φ ∈ {0.25, 0.5, 0.75, 1.0} at **fixed** T_pad, plus a heterogeneous-φ row via `NORIG_TABLE` | **φ** ⟂ T (breaks the worst collinearity we have) | M24 `_FILL`+`_TGRAIN`+`_ZERO_PAD` | *row-proportional cost scales with real tokens; every quota tuned at padded capacity is wrong by 1/φ* | if C* and quota* do not move with φ, fill-awareness is not a schedule axis and M24 is a work-deletion feature only |
| **K4** | `k_inner` sweep until ρ ∈ {0.65, 1.0, 2.0, 3.0} × {phased, fused} × depth {4,8,16,32} × slab {128,256,512} | **ρ**, **co**, and re-scores F1–F4 | rig argv | *hiding ≤ min(ρ,1); the fused schedule pays only above ρ≈1* | **pre-registered (from `TP8_STATE.md` M2):** if hiding is still <10 % at ρ≈3, the "hide it in the epilogue" thesis is in serious trouble |
| **K5** | message size ∈ {3.67 MB … 235 MB} × {atomics, towers} × {push, pull} + **RCCL reference at each** | **λ**, and K1's sign flip | rig argv; the rig already links `-lrccl` | *K1 is chosen by issue rate at the deployment message size, not by the arch card* | this is the **entire go/no-go for the decode cells** |
| **K6** | granularity decomposition: signal count at fixed geometry, then geometry at fixed signal count | **A vs P** separation (falsifier FQ-3) | `K0P6_M15_SLABS` vs CDAR `slab_rows` | *granularity is free when it moves only A; expensive when it moves P or transfer geometry* | if signal-count-only is expensive, Q9 and Q6 are not separable |
| **K7** | gfx942 transport card at deployment message size | **θ_arch** | other node | *choose RMW vs store+local-reduce by the arch's atomic:store ratio at the deployment λ* | if gfx942 and gfx950 now agree (as F5 suggests), this is a **simplification** for the PR, and the cell can be one run |

Every Tier-1 row must carry the manifest schema of `KNOB_INVENTORY.md` Appendix — including
the instrument gates `K0P6_MPS_ENABLE_{MODE14,TBO}`, `K0P6_M15_SRC_REV` and the kernel-name
pin — plus a `.text`-sha or mechanism-invariant check (LAW-58/59). **A knob whose off-state
is not `.text`-identical is a second arm.**

### 4.5 Tier 2 — e2e validation (order-balanced ≥5 pairs, `BENCHMARK_PROTOCOL.md:50-53`)

| # | cell | Q corner | why it cannot be done at the boundary | cost |
|---|---|---|---|---|
| **E1** | **TP8-native at C=512** | completes the topology×concurrency 2×2 | it *is* the missing corner of the flagship law (§1.3) | 1 arm, ~25 min + protocol pairs |
| **E2** | concurrency ladder C ∈ {32, 64, 128, 256, 512} × {m15-DP, native-DP, native-TP} | **φ** and **κ_sched** as they actually emerge | φ is EMERGENT; no rig produces it | the crossover concurrency is the single most quotable number the campaign could produce (H3) |
| **E3** | ISL ∈ {1024, 2048, 4096} at fixed C | breaks φ-shape ⟂ concurrency; tests the **ISL == bucket** artifact hypothesis (C2) | bimodality is a scheduler property | if fill stops being bimodal at ISL≠bucket, **every tier-1 dummy-skip sizing number is cell-specific** |
| **E4** | OSL ∈ {8, 128, 512} — the decode-bound cells `c8/c16/c32` already defined and never run | **the entire decode branch of Q** | H1 | TPOT is the decode metric and our 4.24× TPOT win is unexplained |
| **E5** | open-loop Poisson at 50/75/90 % of measured ceiling | whether the dummy-skip advantage is **production value or a benchmark artifact** | `BENCHMARK_PROTOCOL.md:36-42` says the closed-loop dummy share is partly an artifact | H6 |
| **E6** | `max_num_batched_tokens` ∈ {2048, 4096, 8192, 16384} on our arm | **T_pad** as an OPERATOR axis | it is the bucket; it sets everything downstream | largest modeled reach, integration-side, never swept (H12) |
| **E7** | placement π ∈ {contiguous, RR} (+ EPLB0 if the predictor gate passes) | **κ** at serving level | κ's serving remedy has never been measured | LAW-47's frontier is offline-only |

**Sample sizes and gates.** e2e minimum detectable effect is set by LAW-45 (~15 % at c32p)
— but `WORKLOAD_EVIDENCE.md` W10b shows the envelope is itself a function of W (≤1.8 %
throughput / ≤0.25 % latency at c512p). ⇒ **measure the variance envelope per cell**; do
not apply the c32p band blanket. Any predicted schedule change smaller than the cell's own
envelope must be adjudicated at the boundary, not e2e.

### 4.6 Explicitly scoped out

Multi-node (H10, zero evidence); PP (no artifact); training-arm transfer beyond the three
laws already marked TRANSFERS (LAW-52/53/54); exact-token SHA as a parity gate (the kernel
is not bit-reproducible — 116/1024 requests differ between arms, and stock-vs-stock SHAs
differ, `WORKLOAD_EVIDENCE.md` W10d).

---

## 5. Why interpolation must happen over Q, not W

1. **Dimension.** W has 21 axes and 2 serving samples. Q has 10 coordinates, 6 of which the
   *boundary rigs* already span by 2–5× at ~3 min/campaign. Interpolating in W is
   extrapolating; interpolating in Q is interpolating.
2. **Q is computable before the run.** Every Q coordinate is arithmetic over (W, θ):
   B_bnd from routing combinatorics, C_phase from GEMM shapes, ρ from their ratio, P from
   slab geometry, α from a two-point fit, λ from message size. That is what lets the cost
   model *predict* a cell rather than look it up — the difference between a contribution and
   a table.
3. **Many W moves are redundant in Q.** Concurrency, ISL, arrival process and prefix caching
   all act on the kernel *only* through (T_pad, φ, κ_sched). A cost model in Q therefore
   answers questions about cells nobody ran, which is precisely the expert's demand
   ("as we toggle the list of workload properties, what are the properties of the kernel
   that change", `../BRIEF.md:27`).
4. **Q makes the contradictions in the ledger dissolve into scope statements.** Depth-flat
   vs depth-cliff → `co`. Granularity-free vs granularity-2.1× → `A` vs `P`+geometry.
   Dedication-wins vs dedication-never-wins → carrier job × (Σ, P). Three of the six
   contradictions in `BANKED_LAWS.md` §6 are resolved by naming one Q coordinate.
5. **Q is what makes the S vector finite.** `KNOB_INVENTORY.md` Part D already selects the
   ten highest-leverage knobs *because* their sign or magnitude is known to move with a
   workload axis. Q is the formal statement of which axis, and therefore of which knob must
   be re-swept in which cell. In particular **C and quota are derived quantities of
   (Σ, P, t50), not inputs** — the kernel itself says its tuned point is invalid at other
   T_eff (`k0pf6gm_device_tile_m15.hip:2122-2127`).

**The one-sentence version for the master doc:** *we did not sweep hundreds of kernels; we
identified ten numbers you can compute from a serving config, and showed that those ten
numbers — not the config — decide the schedule.*

---

## 6. What this document does not claim

* **H-Q is a hypothesis, not a result.** It has three pre-registered falsifiers (§2.4) and
  five named threats. It is stated so a reviewer can kill it in one experiment.
* **No number here is a new measurement.** Everything tagged DERIVED is arithmetic over
  cited artifacts, shown inline. Where my arithmetic disagrees with a banked figure — the
  T-crossover (1,311 / 1,398–1,483 vs the banked 1,600–1,800) — the disagreement is stated,
  not reconciled by choosing.
* **The retraction list of `WORKLOAD_EVIDENCE.md` §0.2 is binding here too**: no "1.19×
  padding multiplier", no "31 % of dummy *steps*", no "+2.7–5.2 %", no `t2048_m15_*` MoK
  ratios as M15 data, no "m15 beats production."
* **Two claims already retracted elsewhere are re-confirmed by this document's arithmetic
  and must stay retracted**: CDAR's "−32 % fewer bytes" (Q1 gives −9.7 %), and any reading
  of the c32p −21.17 % as a *kernel* delta (the arms differ in Group-H coordinates, §1.3).
* **φ has zero controlled points today.** Every statement of the form "fill explains the
  regime split" is, until grid cell K3 runs, an inference across two cells that differ in
  six variables.
* **Decode is a formula branch with no data behind it.** Q's decode terms are predictions,
  and the campaign should say so on the slide.

---

## Amendments from review (2026-08-18)

Superseded by `../ABLATION_METHODOLOGY.md`, which is now the authority on the contribution
sentence, the canonical `Q` table, and the cell registry. Changes forced by BLOCKING items in
`../review/CONTRIBUTION_CRITIQUE.md` and `../review/MECHANISM_CRITIQUE.md`:

1. **The contribution sentence in §0/§2.4/§5 is replaced.** The campaign defends one sentence:
   *one base megakernel plus a library of attachable schedule modules, with `M(Q(W), R(S); θ)` as
   the composition policy.* The two topology-specific sentences are the pre-committed **fallback**
   (regime map), not the headline.
2. **`Q` is reduced from ten coordinates to eight and made schedule-independent** (CONTRIBUTION
   B2/B3, MECHANISM B10). `P`, `u(t)`, `A`, `L`, `Σ`, `φ_p` and burstiness are demoted to the
   schedule-response vector `R` — §2.1 already defined them there, and §2.2 contradicted itself by
   re-listing them as Q6/Q9. `κ_sched`, seal coverage and `W_pad` move into a separate validity-gate
   vector `V`. Canonical table: `ABLATION_METHODOLOGY §2.2`. Arity is frozen at Phase-1 registration.
3. **`crit persistence` and burstiness are declared provisional coordinates**, pre-registered as the
   first two mediators to add if H-Q fails; burstiness gets falsifier FQ-4.
4. **§4.4's cell ids are superseded by the single registry** in `ABLATION_METHODOLOGY §7.2`. This
   document's K6 (granularity decomposition) is restored as **FQ-3**, and the producer-order ×
   certification law gets a cell for the first time (`K5×order`).
5. **§4.5's E1/E2 pair sizing** is repriced with rotation balancing, and the c512p cell is labelled
   `R-L3 FAIL` until a 10,240-prompt confirmatory pair runs.

# S_VECTOR — the formal SCHEDULE STATE VECTOR

**Deliverable:** output (2) of `../BRIEF.md:116-118` — the kernel-design degrees of freedom,
organized so that "hundreds of kernels" collapse into **parameterized skeletons + a manifest**
(`../BRIEF.md:105-108`), not hand-written one-offs.

**Written:** 2026-08-18, local design pass, no node access.
**Inputs (binding):** `../grounding/KNOB_INVENTORY.md`, `../grounding/BANKED_LAWS.md`,
`../grounding/WORKLOAD_EVIDENCE.md`, `../grounding/TP8_STATE.md`.
**Consumers:** the cost model `M(W, S; θ)` (BRIEF item 3), the experiment ladder (item 5), the
per-cell predictions (item 6).

**Rules obeyed here.** Every claim is anchored to a repo path (`file:line` where useful) or to a
grounding brief. Numbers appear **only** with a source. Anything not measured is marked
**SPECULATION** or **PREDICTION** (a prediction is a SPECULATION that names the banked law it is
extrapolated from and its falsifier). No number appears without its regime.

---

## 0. The object

### 0.1 Four tiers, deliberately separated

The single most common failure mode in this corpus is conflating things that live at different
tiers, and it has cost measured time: making mode-14 code merely *reachable* moved the mode-12
ratchet by **+726.9 µs with mode-12's source untouched**
(`KNOB_INVENTORY.md` B.12, from `moe_mps_adapter.cuh:39-56`; BANKED_LAWS LAW-32). So:

| tier | symbol | what it is | changing it means | identity check |
|---|---|---|---|---|
| **0 — STRUCTURE** | `K` | which *program* runs: who holds which role, where the launch boundaries are | a different binary; every knob optimum must be re-derived | source file + `K0P6_M15_SRC_REV` + kernel-name pin |
| **1 — SCHEDULE KNOBS** | `b` (compile), `r` (runtime) | legal settings *within* a skeleton | a point in the knob space | `.text` sha256 (compile) / device descriptor echo (runtime) |
| **2 — INSTRUMENT PINS** | `I` | things that do not change the intended schedule but change the measurement | the number's *identity*, not its value | must be in every row (LAW-58/59) |
| **3 — DERIVED OPTIMA** | `C*, d*, fr*, …` | knob settings the cost model is supposed to *predict* | nothing — these are outputs | must never appear as a campaign input |

**Tier-3 is not a formality.** `KNOB_INVENTORY.md` B.5 states it directly: the C-response is
steep and non-monotonic *across designs* (mode 12 wanted `C ≤ 8`; M15 wants `C = 24–28`), and the
kernel's own comment says its tuned point is invalid at other `T_eff`
(`k0pf6gm_device_tile_m15.hip:2122-2127`). `C`, `flush_rows`, and `depth` are therefore **derived
quantities of the cost model**, re-swept inside every workload cell — never carried.

### 0.2 The formal vector

```
S  =  ⟨ K , b , r , g , F ⟩            the SCHEDULE state (this document)
W  =  ⟨ T, CONC, TOPO, SKEW, FILL, PHASE, ARCH, … ⟩     the WORKLOAD state (sibling doc)
I  =  ⟨ build gates, src rev, kernel-name pin, timestamps bit, .text sha ⟩   the INSTRUMENT
H  =  ⟨ harness, protocol, n, rotation/pairing discipline ⟩                  the MEASUREMENT

one ARM   :=  (K, b, r, g, F)                      — a point in schedule space
one RUN   :=  (ARM, W-cell, I, H, n) → results     — one manifest row (§4)
```

where

* `K` ∈ the skeleton set of §1 (structurally distinct programs),
* `b` = compile-time knob vector (the `-D` set),
* `r` = runtime knob vector — for the DP/EP family this is **one packed 64-bit descriptor word**
  `K0P6_D_MPS_CFG` (slot 62; layout `moe_mps_adapter.cuh:164-171`, struct `:183-190`, encode/decode
  `:192-216`, validator `config_is_valid` `:583-673`), driven by env `K0_MPS_CFG`; for the TP8/CDAR
  family it is the rig argv (`m25_boundary_bench.hip:441-448`),
* `g` = geometry/extent (T, MAXTOK, world, TOPK, slab extent, grid),
* `F` = the fusion-boundary set: which ops live inside the launch (§2.12).

**Why this shape makes the campaign tractable.** `KNOB_INVENTORY.md` B.0 establishes the enabling
fact: *the entire runtime schedule surface of the M15/M24 serving mega is one 64-bit word*, and
`BENCHMARKING.md:174-186` already ships the discipline — **one HSACO serves the whole sweep, one
arm name, one campaign per sweep point**. The "hundreds of kernels as points in a knob space"
requirement is therefore already satisfiable for the shipped skeleton without new machinery.

### 0.3 Every knob is tagged with the mediating quantity it moves

The cost model consumes `S` through the mediating quantities BANKED_LAWS §preamble already names.
Each knob below carries these tags so `M(W, S; θ)` can be assembled mechanically:

| tag | mediating quantity |
|---|---|
| `d` | outstanding remote ops per producer thread (injection depth) |
| `co` | compute co-residency of the carrier (is a GEMM sharing the device?) |
| `A` | protocol atomic/flag operation **count** per epoch |
| `L` | cache lines touched per protocol event (scatter width) |
| `φ` | fraction of the payload phase that is CTA-throughput-bound |
| `u(t)` | consumer-unblockable fraction under the producer's order |
| `fill` | real rows ÷ executed rows |
| `σ_rank` | max-rank load ratio (routing + padding skew) |
| `crit` | identity of the critical (hottest/slowest) rank |
| `B` | in-launch bubble supply (CTA-ms) |

And with the workload axes from `KNOB_INVENTORY.md` §preamble: **T · CONC · TOPO · SKEW · FILL ·
PHASE · ARCH**.

---

## 1. PART I — SKELETONS

A skeleton fixes *who runs what role* and *where the launch boundaries are*. Skeletons are mutually
exclusive builds, not config points. The set below is `KNOB_INVENTORY.md` Part A (S0–S8) plus one
addition the grounding forces (`D1`, §1.10).

### 1.0 The master table

| id | skeleton | structural commitment | vehicle (file) | status | best measured point (source) |
|---|---|---|---|---|---|
| **S0** | Unfused production: GEMM kernels + separate collective | comm is its own kernel; no in-kernel cross-rank protocol; K3 degenerates to a phase barrier | vLLM 0.25.1 + AITER/MORI (external), arm `production` (`BENCHMARKING.md:198-206`) | **reference, always present** | DP prefill boundary **7,712 µs = 1.000×** (`../BRIEF.md:49`); TP8 boundary **GEMM→RCCL 2,909 µs** (`tp8_mega/results/G25_1_STATUS.md:14-19`) |
| **S1** | Homogeneous megakernel | 256 identical CTAs, grid barriers between phases M0…M9 | `k0pf6gm_device_tile.hip` (arm `pf6gm_mega`) | **shipped control** | **6,908.8 µs = 0.8958×** (`../BRIEF.md:50`) |
| **S2** | Carrier pool (dedicated comm CTAs, modes 1–9) | reserved CTAs *move payload* | `moe_mps_adapter.cuh` modes 1–9 + `k0pf6gm_device_tile_mps.hip`; `DESIGN_MPS.md:37-75` | **falsified at one point; retained as control** | pool-carried **+332…+340 µs** (`OVERLAP_ABSTRACTIONS.md:28`); C=64 dedicated comm CTAs **6,866 µs (0.888×)** with **+831 µs added traffic from the role itself** (`../BRIEF.md:52-53`) |
| **S3** | Producer-carried, homogeneous roles (modes 12/13) | the GEMM-2 epilogue itself issues the remote packed-bf16 accumulate; per-row readiness (~926k ops) | `n2_phase2_gm_mps.cpp:236-238`; modes at `moe_mps_adapter.cuh:352-353` | shipped ratchet | unbounded **7,110.8** (worse than S1); depth-4 **6,483.8 = 0.8407×** (`../BRIEF.md:50,54-55`) |
| **S4** | **M15 — slab-certified producer/consumer with a CONSUMING pool** | nc-major producer order + 2 column slabs + one epoch word per (rank, slab) + `C` reserved CTAs that *consume* certified front-half combine in the producer's shadow | `k0pf6gm_device_tile_m15.hip`; contract `aug12/M15_DESIGN.md:20-42` | **the shipped best** | C=16 6,292.4 → 24 **5,848.5 (0.7589×)** → 28 **5,822.0 (0.7544×)** → 32 unstable, hung 1 of 2 runs (`../BRIEF.md:50-51`) |
| **S5** | Cross-launch deferred combine (mode 16 / TBO-2) | launch boundary becomes a pipeline stage; epoch-parity buffers | `moe_mps_adapter.cuh:395`, gated `K0P6_MPS_ENABLE_TBO` | **falsified — retire to a named negative** | **6,559.6 µs = 0.8513×, +75.8 µs vs the ratchet** (`OVERLAP_ABSTRACTIONS.md:52-53,83`) |
| **S6** | **CDAR / counter-dataflow TP8 megakernel (M25)** | chunked all-reduce that never exists as a phase: ERS rides the producing GEMM's epilogue, MAG rides the consuming GEMM's ramp | `aug18-prefill/M25_TP8_MEGA_DESIGN.md`; rig `tp8_mega/{m25_cdar.cuh, m25_boundary_bench.hip}` | **RED on perf, GREEN on correctness** | fused CDAR **3,606** vs GEMM→RCCL **2,909 µs**; phased 3,653; compute floor 1,439 (`G25_1_STATUS.md:14-19`, commit `3f7f0f3a`) |
| **S7** | MoK-style counter dataflow, zero grid barriers (T3) | every dependency is a flat HBM counter; producers get lower task indices than consumers; no residency contract | `aug18/T3_COUNTER_DATAFLOW_DESIGN.md`; partial vehicle `k0pf6gm_device_tile_t3.hip` | **design only** | projection band 1,280–1,330 ms training step — **explicitly a projection, not a measurement** (`T3_COUNTER_DATAFLOW_DESIGN.md:85-91`) |
| **S8** | Staged wide-push transport (m15b) | epilogue folds partials into a *local* stage; carriers push folded half-rows as posted 16 B stores — swaps the fabric op class RMW → posted store, ~1.44× fewer remote bytes | `k0pf6gm_device_tile_m15.hip:63-70` (`K0P6_M15_STAGED`, default 0); `aug12/M15_DESIGN.md:63-76` | **built, never validated — P0 blocker** | `STAGED=1` compiles to **96 `scratch_load_dword` + 97 `vmcnt(0)`** within 24 instructions of the remote atomic vs 0/1 in the default build (`aug18-prefill/FP8_WIRE_DESIGN.md:24-29`) — the throttle is annihilated |
| **D1** | **Small-message / decode-shaped collective** (§1.10) | latency-shaped, not bandwidth-shaped: one-shot or LL-style AR, small-M tiles, slab extent collapsed, K1 reversed | **not built**; nearest vehicle is `m25_boundary_bench.hip` at `tokens=256, slab_rows=64` | **one data point exists** | 256 tokens / 64-row slabs / 3.67 MB: **801 µs (atomic) vs 1,017 µs (towers)** — 4.6 / 3.6 GB/s, latency-dominated, **towers LOSE** (`TP8_STATE.md` §5 M5) |

### 1.1 The structural law that separates them (and answers the experts' question directly)

The expert feedback asks: *"does our M15 producer/consumer work best for all sizes?"*
(`../BRIEF.md:21`). The answer lives in one three-way split, because **S2 and S4 use the *same*
reservation mechanism with opposite economics** (`KNOB_INVENTORY.md` A.1, from
`docs/distributed/OVERLAP_ABSTRACTIONS.md:26-32`):

| pool job | measured verdict | source |
|---|---|---|
| **carrying** payload | **+332…+340 µs** — never wins | `OVERLAP_ABSTRACTIONS.md:28` |
| **idling** (reserve-but-wait) | **+8–9 µs per reserved CTA**, monotone | `:29` |
| **consuming** certified-ready output in the producer's shadow | **−93 µs** (C 8→16), **−444 µs** (C 16→24) | `:30` |

The enabling difference is the *certification* (K4, §2.4) that creates early consumable work, plus
the *order* (K5, §2.5) that makes the certified prefix large. **Pool job is therefore a skeleton
axis, not a knob.**

**And the verdict is regime-conditional in a way the ledger already records** (BANKED_LAWS
contradiction C3): dedication **WON** at C=64 in the *staged-push* carrier (0.888× vs homogeneous
0.894×, `aug10/experiments/LESSONS.md:669-676`) and **LOST monotonically at every size** in the
*producer-carried* carrier (exp_37: C=4 6,464.2 … C=64 6,837.5, paired in 7 of 7 rounds,
`aug11/LESSONS.md:287-301`). Retracted-claims list item 8 is explicit: *"CTA specialization is
generally bad" is not established* (`aug11/OVERLAP_METHODOLOGY_STUDY.md:33-36`). **S2 stays in the
campaign as a live control arm, not as a closed question.**

### 1.2 The selection function every skeleton must pass (the occupancy-1 axiom)

Every kernel in this family runs 1 block/CU, 1 wave/SIMD with 256-VGPR GEMM bodies — 256 ArchVGPR
+ 256 AGPR and **155,428 B of LDS (of 163,840) independently force one block per CU**
(BANKED_LAWS LAW-12, `aug10/KERNEL_BOTTLENECKS.md:24-27`). Consequence: there are no spare waves to
fill stalls, `vmcnt` is one in-order counter per wave, and **phase partitioning is
work-conserving** (`OVERLAP_ABSTRACTIONS.md:42-47`). Overlap pays only through four doors
(`:55-60`):

> (a) a latency-bound consumer into a compute shadow; (b) deleting protocol work; (c) op-class
> swaps on the fabric; (d) bounding injection.

**Any proposed schedule that goes through none of these doors is predicted ≈0.** This is the
falsifiable selection function the campaign should test directly, and it is the cheapest possible
pruning rule: a candidate skeleton that cannot name its door does not get built.

**Corollary that binds the manifest:** occupancy must be *asserted and printed*, not assumed —
shrinking a ubench's LDS silently raised occupancy 1 → 2 blocks/CU and voided that experiment's
premise (LAW-12 workload-dependence note, `aug11/LESSONS.md:99-109`).

### 1.3 S0 — unfused production (the reference, in two topologies)

* **Commits to:** collective as its own kernel. No certificates can exist (K4 degenerates to
  `S=1`, a phase barrier — `KNOB_INVENTORY.md` C.3).
* **Knobs exposed:** none of ours. Its only degrees of freedom are the vendor's (RCCL algorithm,
  MORI dispatch order) and the serving-stack control plane (`WORKLOAD_EVIDENCE.md` §3.1).
* **Where it wins:** wherever the collective is **dense, regular and large** — exactly the regime
  the vendor library is tuned for. Measured: RCCL AR at 235 MB is **1.5× faster than our
  transport** (`TP8_STATE.md` §2.3: 1,470 vs 2,214 µs; 280 GB/s = **79% of the node egress card**
  vs our 58%). BANKED_LAWS LAW-10 states the expected direction: *the gap closes as the collective
  gets more data-dependent, not as it gets bigger.*
* **Campaign role:** mandatory in every cell as the ratio denominator. `BENCHMARKING.md:174-186`
  requires it inside the *same* campaign invocation, never cross-run.

### 1.4 S1 — homogeneous megakernel (the control that carries no payload)

* **Commits to:** one launch, grid barriers between phases, no role split.
* **Knobs exposed:** grid geometry only; `K0P6GM_G` is `#error`-pinned to 3
  (`k0pf6gm_device_tile_m15.hip:502-506`).
* **Why it must stay:** it is the *only* arm that measures "megakernel, no cross-rank protocol",
  which is the baseline LAW-21 is stated against (**producer-carried without the bound is +211.2 µs
  slower than the homogeneous baseline that carries no payload at all**, `aug11/LESSONS.md:41-54`).
* **Where it could win:** small T. LAW-39/`WORKLOAD_EVIDENCE.md` W6 records that the whole fused
  family inverts against production between T=2048 and T=1024 (pf6gm/prod 0.8938 → 0.9529 →
  **1.1098**), so at small T the question is not which fused skeleton but whether to fuse at all.

### 1.5 S2 — carrier pool (the control arm the experts' question requires)

* **Commits to:** reserved CTAs that *move payload*; a producer-enqueued event queue; the pool's
  drain becomes the post-GEMM critical path (LAW-25: at C=64 the drain runs **5,952 µs past M6**
  while M7+combine occupy 6,054 µs).
* **Knobs exposed:** `C`, `g` (group slices), `flush_rows` as pacing/backoff, XCD placement
  (mode 3), pull_fallback, dynamic ticket vs static stripe.
* **Falsified where:** balanced prefill, T=4096, producer-carried carrier (LAW-28).
* **Where grounding says it could still win — three named cells, each with its law:**
  1. **Decode / small-T (`PHASE`, `T`).** LAW-15 measured that the CTA-capacity tax falls *entirely*
     on the CTA-throughput-bound phase (M7 +27%, linear in C) and **not** on the latency-bound one
     (plan+M6 **+0.5%, flat across C = 2/16/32/48/64**). LAW-27's 8–9 µs/CTA reservation tax and
     LAW-14's marginal-interference curve are both properties of `φ` (CTA-throughput-boundedness).
     **PREDICTION:** at decode shapes `φ → 0`, so both the capacity tax and the interference term
     should flatten and dedication's cost approaches zero. **Falsifier:** measure the mode-14
     idle-pool C-sweep (LAW-27's instrument, `aug11/LESSONS.md:445-452`) at decode shape; if it is
     still 8–9 µs/CTA monotone, the prediction is dead.
  2. **Skew (`SKEW`).** LAW-31 explains why a pool buys nothing at balanced routing (dispatch has
     **zero exposed wait**, max-spins-to-success = 0). LAW-62(b) then records that the skew axis was
     **never settable** on that harness (`K0_SYNTH_ROUTE` rejected; reachable CV **0.0034–0.0086** vs
     COMET's 0.032), so "skew is the one regime where a service pool may re-enter" is
     **untestable, not refuted** (retraction list item 7).
  3. **Staged-push carrier (`K1`).** C3: dedication won at C=64 *within* that carrier.
* **Campaign role:** one C-response curve in each of the three cells above. This is the literal
  experiment that answers *"does producer/consumer work best for ALL sizes?"*

### 1.6 S3 — producer-carried, homogeneous roles

* **Commits to:** the GEMM-2 epilogue issues remote packed-bf16 accumulates; per-row readiness.
* **Knobs exposed:** depth (K3), `g` bits, `flush_rows`, `C` (as a pure tax — mode 12's optimum is
  `C ≤ 8`, `OVERLAP_ABSTRACTIONS.md:120-123`).
* **Its law (do not split it):** LAW-21 — carrier relocation and injection bound are **not
  separable and never additive**. Rung (b) epilogue-carried, throttle disabled = **7,110.8 µs**;
  rung (c) with depth-4 = **6,495.8 µs**; the throttle alone is **1.52× the entire (a)→(c) gap**.
  Any waterfall figure that sums their coefficients is wrong.

### 1.7 S4 — M15 consuming pool (the shipped best, and the campaign's spine)

* **Commits to:** nc-major producer order (K5) + 2 column slabs (K4) + one epoch word per
  (rank, slab) + `C` reserved CTAs that *consume* the certified front half in the producer's shadow
  (`k0pf6gm_device_tile_m15.hip:2102-2160`).
* **Knobs exposed:** the full runtime word (`C`, `g`, `mode`, `flush_rows`, `pull_fallback`,
  `timestamps`) plus the compile set (`SLABS`, `NC_PER_SLAB`, `M8_CSPLIT`, `SCATTER_RR`,
  `REPLICATE`, `ADAPTIVE`, `SAVE_Z`, `M20_SLOTPOOL`, the `M24_*` family).
* **Where it wins, measured:** T=4096 balanced prefill (0.7544×) and the c512p serving cell
  (**+8.40%** and, in the reverse-order pair, **+6.60%**; geomean **1.0750**, n=2 —
  `WORKLOAD_EVIDENCE.md` §W8.2, still below the ≥5-pair floor).
* **Where it is known to lose:** c32p (**−21.17%** vs TP8+EP) and T=1024 (**1.0940×**, i.e. we
  lose) — but the T-sweep's provenance trap is binding: the numbers are campaign-grade on the
  **tgen body**, and the serving M15 body **failed the MoK gate at T=2048/1024** on 2026-08-18
  (`WORKLOAD_EVIDENCE.md` §W6). **The serving M15 body has never produced a number at T ≠ 4096.**
* **Structural caveat the campaign must carry:** `C=32` *hung 1 of 2 runs* (`../BRIEF.md:51`). A
  hang is a schedule-space boundary, not noise — it must be recorded as a manifest outcome value,
  not dropped.

### 1.8 S6 — CDAR / counter-dataflow (the TOPO skeleton)

* **Commits to:** the collective never exists as a phase; certificate-gated layer transitions; two
  certificate levels — local `counted_arrive_dynamic_release_into` then **one store** of a
  per-source epoch word, *no cross-rank RMW counters* (`m25_cdar.cuh:20-31`).
* **Knobs exposed (all in `m25_boundary_bench.hip:441-448`):** `mode ∈ {compute, phased, fused,
  rccl}`, `tokens`, `slab_rows`, `iters`, `depth`, `k_inner`, `bps`, and `transport ∈
  {atomic_accumulate, store_towers}` (`m25_cdar.cuh:72-75`).
* **The structural dividend (why this skeleton is not just "M15 on TP"):** under DP the destination
  is *data-dependent, known only post-gate, and only to the sender*; under TP8+EP `owner_of(slab) =
  slab % World` is **structural, known before the producing GEMM starts, and known to every rank
  with zero communication** because routing is recomputed locally from the replicated post-AR hidden
  (`m25_cdar.cuh:88-93`; `TP8_STATE.md` §4.1). **Replication buys schedule determinism** —
  `owner_major_staggered(rank)` can be applied at plan time and the ring emerges by construction.
* **Honest state:** RED. Two separable deficits — transport **1.51×** slower than RCCL, and hiding
  **2.1%** against a PASS bar of ≥90% (`TP8_STATE.md` §2.3). Four producer-side hypotheses falsified
  in-rig (LAW-34).
* **The finding that reframes those falsifications:** the rig is at the wrong operating point. At
  `k_inner=2048` the compute floor is 1,439 µs for two bursts (~720 µs each) while one boundary's
  CDAR is 1,880–2,214 µs — **the collective is ~3× LONGER than the compute it must hide under**,
  whereas the deployment ratio is ~7.2 ms compute vs ~2.8 ms collective (**~3:1 the other way**).
  Hiding was Amdahl-capped near zero and F1–F4 were **tested blind**. Fixing it costs one argv value
  (`TP8_STATE.md` §2.4, M2). **This is the single most important entry in the whole knob space: the
  `k_inner` knob is not a rig detail, it is the axis that decides whether the experiment was
  possible.**
* **Byte-claim retraction that rides with this skeleton:** `M25_TP8_MEGA_DESIGN.md:98`'s "−32%
  fewer bytes" mixes per-rank and aggregate conventions in one row; re-derived it is **≈ −9.7%**,
  because ERS+MAG is a two-shot AR moving exactly ring's bytes at boundary 1. **CDAR's case is
  overlap, not bytes** (`TP8_STATE.md` §1.3).

### 1.9 S5, S7, S8 — the three that are not primary axes

* **S5 (mode 16 / TBO):** falsified, **+75.8 µs**; "the wait it escaped was mostly not on the
  critical path, and the parity state it added was" (`OVERLAP_ABSTRACTIONS.md:52-53`). Keep as a
  named negative in the teaching list; do not build a sweep.
* **S7 (T3 zero-barrier):** design only, with an explicit projection not a measurement. Its
  contribution to *this* document is a law about the fusion boundary, not an arm: the all-256-CTA
  grid barrier creates a residency contract that made every separate wgrad kernel standoff or
  serialize, and **realizes cross-rank drift at three points per launch instead of one per layer**
  (`T3_COUNTER_DATAFLOW_DESIGN.md:43-49`). Fusion boundary → barrier count → drift-realization
  count.
* **S8 (m15b staged push):** **P0 before any A/B.** The scratch pathology
  (`FP8_WIRE_DESIGN.md:24-29`) annihilates the depth throttle, so *every* S8 × depth arm is void
  until it is fixed — and LAW-32 says exactly this class of thing re-times arms whose knobs were
  never touched. S8 is also the **only** route to F6 (fp8 on the wire): gfx950 has **no fp8 remote
  RMW**, so fp8 is unreachable from the mode-12 epilogue at all (`FP8_WIRE_DESIGN.md:12-29`).

### 1.10 D1 — the decode-specific skeleton (new; forced by the grounding)

**Why a separate skeleton and not a knob setting.** Three independent measurements say the decode
regime does not merely re-tune the prefill skeleton, it *reverses* its structural choices:

1. **K1 reverses.** At the deployment prefill size (235 MB) store-towers beat atomics **117.2 vs
   67.1 GB/s**; at 3.67 MB (256 tokens, 64-row slabs) **towers LOSE**: 1,017 µs vs atomics' 801 µs,
   because the reduce step is no longer amortised (`TP8_STATE.md` §5 M5). K1 is therefore
   *message-size-dependent* — "a second manifest axis", in that document's words.
2. **K4 collapses.** `M25_TP8_MEGA_DESIGN.md:212-215`: "chunking collapses below ~2K tokens"; and
   `m25_cdar.cuh:82` already documents `slab_rows` as "prefill default 512; **decode mode shrinks
   it**".
3. **The mechanism class changes.** `TP8_STATE.md` §6 B8: decode is "one-shot/LL AR, small-M tiles,
   possibly fused AR+RMSNorm … **do not conflate with the prefill bandwidth track — different
   mechanisms, different cells**" (`TP8_OVERLAP_ANALYSIS.md:110-116`).

**What D1 commits to:** latency-shaped transport (no owner-local reduce pass, no tower fan-out),
slab extent at or below one wave's worth of rows, certificate scope per-owner rather than global,
and a fusion boundary that includes the norm (the AR+RMSNorm fusion of B8) because at these sizes
the norm's HBM pass is comparable to the collective.

**Status: NOT BUILT.** The nearest vehicle is the existing boundary rig run at `tokens=256,
slab_rows=64`, which is where the single decode data point above came from. The **go/no-go is M5**
— an RCCL reference at decode message sizes, designed as G25-0b(iii) and **never run**; the rig
already links `-lrccl` and calls `ncclAllReduce`, so it is one argv away (`TP8_STATE.md` §5 M5).
Without it we have **no basis for any claim below ~2K tokens.**

**Why it matters more than its current evidence suggests:** decode has **never been measured at all
in any of our rigs** (BANKED_LAWS §8 hole 4; `WORKLOAD_EVIDENCE.md` H1 — the `c8/c16/c32` (ISL 1024,
OSL 512) and `c512` decode cells are *defined* in
`campaign_v5/run_m15_campaign_eplb_v5.sh:1165-1210` and **never executed**), yet TPOT is the decode
metric and **m15 already wins TPOT p50 by 4.24× at c512p** (757.1 vs 3,208.8 ms,
`WORKLOAD_EVIDENCE.md` §W8.3) — entirely unexplained.

**One honest structural note on the DP side.** Today there is no DP decode skeleton to build,
because the megakernel does not *see* decode shapes: post-M23 the uniform-decode rescue routes those
steps through the B4096 graph, so the mega executes **4,096 padded rows for ~1 real token per rank**
(`WORKLOAD_EVIDENCE.md` §3.2, `CORPUS_FINDINGS.md:9-16`). *Whatever the kernel does on a decode
step, it does at prefill shape.* The DP-side "decode skeleton" is therefore an **integration**
knob — bucket size / `max_num_batched_tokens`, hole H12, **never swept**, and named as the lever
with the largest modeled reach (`FILL_AWARE_DESIGN.md:1337-1345`).

### 1.11 Skeleton → regime map (the prediction table this document owes the cost model)

| skeleton | predicted-best regime | mediating quantity that selects it | status of the prediction |
|---|---|---|---|
| S0 | dense/regular/large collective; T below the fused-family crossover (~T 1,600–1,800/rank at C=28) | vendor-tuned bandwidth efficiency; fixed-cost amortization | MEASURED at both ends (LAW-10, LAW-39) |
| S1 | small T, no cross-rank protocol worth its fixed cost | `A` (protocol op count) vs T | MEASURED (T-sweep inversion, tgen body) |
| S2 | decode / `φ→0` phases; skew-dominated cells; staged-push carrier | `φ`, `σ_rank` | **PREDICTION** — falsifiers named in §1.5 |
| S3 | co-resident carrier where the bound can be installed but no certified prefix exists | `d` × `co` | MEASURED (LAW-20/21) |
| S4 | high-fill DP prefill, T ≥ ~2048/rank, balanced-to-moderate skew | `u(t)` (certified prefix), `fill` | MEASURED at T=4096 and c512p |
| S6 | TP8+EP, large dense collectives, compute:transport ≥ ~3:1 | exposed collective fraction | RED; untested at the right ratio (M2) |
| D1 | decode / small message / latency-bound | message size crossing the reduce-amortization threshold | one point; go/no-go = M5 |

---

## 2. PART II — KNOBS WITHIN A SKELETON

Grouping follows the BRIEF's own list (`../BRIEF.md:116-118`). Every row carries its provenance in
`KNOB_INVENTORY.md` plus the underlying `file:line`. `c/r` = compile-time / runtime.

### 2.1 K1 — CARRIER ROLE and TRANSPORT OP

| field | value |
|---|---|
| **domain** | {producer-epilogue remote packed-bf16 RMW (modes 12/13) · staged pool push (mode 2) · staged wide posted-store push (m15b) · owner-gathered **store towers** (CDAR) · consumer **PULL** (`load_peer_packets`) · SDMA} |
| **provenance** | `KNOB_INVENTORY.md` B.1 |
| **sites** | epilogue RMW `n2_phase2_gm_mps.cpp:236-238`; primitives `include/cdna4/ops/group/distributed/credit.cuh:96-113` (`throttled_accumulate_bf162` / `throttled_store_packet16`); CDAR enum `tp8_mega/m25_cdar.cuh:72-75`; staged push `k0pf6gm_device_tile_m15.hip:2285` gated at `:68-70` |
| **c/r** | **mixed** — RMW-vs-pool-push is the runtime `mode`; RMW-vs-staged-local-fold is compile (`-DK0P6_M15_STAGED=1`); CDAR transport is a **compile-time template argument** (`m25_cdar.cuh:158`) |
| **default** | epilogue remote RMW (mode 12) in M15/M24; `store_towers` in the CDAR rig |
| **measured** | pool-carried **+340 µs** vs epilogue-carried (`OVERLAP_ABSTRACTIONS.md:79`). Packed-bf16 remote atomics **52.8 GB/s** coalesced vs remote stores **54.9**; **scattered lane addresses collapse to 4.1 GB/s** (`../BRIEF.md:57-58`) — a **13×** cliff. At 235 MB: atomics **67.1**, towers **93.6** (512-row) / **117.2 GB/s** (256-row) (`results/g25_0b_v1_results.txt`, `g25_0b_slab_sweep.txt`). At 3.67 MB the order **reverses** (§1.10) |
| **mediates** | `A`, `L`, `co` |
| **axes** | **ARCH** (decisive: gfx942 remote atomics ~3× slower than stores, 15 vs 44–46 GiB/s — `OVERLAP_ABSTRACTIONS.md:146`), **T**, **TOPO**, **SKEW** |
| **campaign role** | **PRIMARY.** The law it establishes: *choose RMW vs store+local-reduce by the arch's atomic:store ratio **at the deployment message size**, not by the small-message arch card* — F5 reversed `M25_TP8_MEGA_DESIGN.md:115` on exactly this (`TP8_STATE.md` §2.2) |

**Layout is inside this knob, not beside it.** The 13× coalesced/scattered cliff means the
tile→packet mapping must be audited **in ISA, not assumed** (`M25_TP8_MEGA_DESIGN.md:150-153`); the
BRIEF states it as a law: **layout is a schedule decision** (`../BRIEF.md:58`). See §2.9.

### 2.2 K2 — DIRECTION (push vs pull)

| field | value |
|---|---|
| **domain** | {push (producer-defined, many-to-one) · pull (consumer-defined, one-to-many)}; `pull_fallback` bit 32 of the config word |
| **provenance** | `KNOB_INVENTORY.md` B.8 |
| **c/r** | compile per boundary in CDAR (MAG pulls, `M25_TP8_MEGA_DESIGN.md:116`; ERS pushes); runtime diagnostic bit in the DP word (`moe_mps_adapter.cuh:164-171`) |
| **measured** | matched 64 KiB producer→transport→dependent-consumer wall **0.9936× (tie)**; 64 MiB single-link BW pull/push **1.078×** (`../BRIEF.md:63-65`, LAW-5, two independent rigs). Message-size knees: pair knee 256 KiB; fan-out knee 64 KiB/peer (push) vs 1 MiB (pull); **pull collapses past 256 CTAs while push degrades gently** (`aug12/OVERLAP_KERNEL_DESIGN_ADDENDUM.md:317-321`) |
| **mediates** | latency/ordering, not bandwidth |
| **axes** | T, TOPO, CONC |
| **campaign role** | **SECONDARY control**, with one exception: `TP8_STATE.md` A1 ranks *"the wall is consumer-side remote-READ bandwidth in MAG, and pull is the wrong direction at this message size"* as the **top** transport hypothesis, because the tie was measured at **64 KiB** and neither anchor is a 235 MB one-to-many case. Killer: M1 then M8 |

**Law already stated** (`../BRIEF.md:64-65`): *push when progress is producer-defined and
many-to-one; pull when consumer-defined and one-to-many.* BANKED_LAWS grades the tie
PROVEN-replicated and the **selection rule HYPOTHESIS** — neither corner has been measured.

### 2.3 K3 — INJECTION DEPTH (flow control)

| field | value |
|---|---|
| **domain** | enable ∈ {0,1}; depth ∈ **{8 (sel 00), 4 (01), 16 (10), 32 (11)}**; plus {none}; unbuilt: per-destination credits |
| **provenance** | `KNOB_INVENTORY.md` B.4 |
| **sites** | selector bits `0x0300` of `g` (`moe_mps_adapter.cuh:400-410`, accessors `:481-491`); device specializations `n2_phase2_gm_mps.cpp:116-127`, instantiation `:170-176`, SGPR throttle plan `:405-408`; primitive `credit.cuh:63-82` ships depths {0,4,8,16,32} |
| **c/r** | **both, deliberately.** `vmcnt(N)` encodes N in simm16 — **there is no register form** — so depth is a compile-time literal; runtime selection is one textual loop copy plus mutually-exclusive SGPR lane masks. Templating the loop on `Depth` spilled one 4-byte value with **389 reloads (scratch_load 9 → 436)**, contaminating the control arm (`credit.cuh:12-22`) |
| **default** | enabled, depth 4 (`g=353`) |
| **measured** | **the single largest knob in the DP campaign: unbounded 7,110.8 → depth-4 6,495.8 = −615.0 µs** (`credit.cuh:5-9`, LAW-20/21). It is a **cliff, not a curve**: M7 **4 → 2,659 \| 8 → 2,742 \| 16 → 3,204 \| 32 → 3,112 \| unthrottled 3,189 µs** (`aug10/experiments/LESSONS.md:981-990`). **AND depth is FLAT {4,8,16,32} in pure transport** — towers 93.6/93.4/93.3, atomics 67.1/67.1/67.0 (`results/g25_0b_v1_results.txt`) |
| **mediates** | `d`, gated by `co` |
| **axes** | ARCH, TOPO (effective depth capped by `min(nominal, fan-in × tokens interleaved)`, fan-in ≈5.3 — `OVERLAP_KERNEL_DESIGN_ADDENDUM.md:295-300`), SKEW, T |
| **campaign role** | **PRIMARY — the cleanest law the campaign already owns:** *depth binds iff the fabric shares the device with compute; θ is contention, not bytes.* Sweep depth × co-residency (transport-only vs fused) × ARCH |

**Fragility contract that must ride every depth arm** (`credit.cuh:30-38`): a bound that can
silently die is not a primitive — exp_38 saw register pressure re-throttle an epilogue with **no
source change**. The ISA gate is (a) literal `vmcnt(N)` at expected sites, (b) **zero scratch ops
within 24 instructions of any remote atomic**, (c) issue-run distribution matching the plan
(**~282 atomics / ~12 runs / mean ~23.5**). Depth-2 has never been measured — the `g` `0x300`
selector has no room left, so **the left side of the cliff is unexplored** (LAW-20).

### 2.4 K4 — READINESS / CERTIFICATE GRANULARITY and SCOPE

| field | value |
|---|---|
| **domain — granularity** | {per-row flags (~926k protocol ops) · per-(rank, slab) epoch words (2×8) · one phase barrier (S=1) · per-(owner, slab) CDAR cells}; slab extent in rows |
| **domain — scope** | {global rendezvous · per-owner} — the axis §2.4.1 argues is the real one |
| **provenance** | `KNOB_INVENTORY.md` B.3 |
| **sites** | M15 `K0P6_M15_SLABS 2`, `NC_PER_SLAB 8`, `M8_CSPLIT 7` at `k0pf6gm_device_tile_m15.hip:498-500`; slab loop/rendezvous `:2067`, `:2235-2260`; primitives `slab.cuh:62-97`; CDAR `slab_rows` is a **runtime** field of `cdar::geometry` (`m25_cdar.cuh:79-101`) and argv 3 of the rig |
| **c/r** | **compile in M15** (S=2 is the only value where 448-column nc chunks and M8's 1,024-byte chunks co-align at column 3,584, and is also the fence-law optimum — `:496-500`); **runtime in the CDAR rig** |
| **measured — M15** | per-row → slab words replaced ~926,000 protocol ops with 2×8 certificates; **granularity is the free axis, EXPOSURE is the deadly one** — a **64× signal-count change measured ~nothing** (`slab.cuh:10-12`) |
| **measured — CDAR** | granularity is **not** free: 128 rows **116.0** · 256 rows **117.2** · 512 rows **93.4** · 1024 rows **55.4 GB/s** — a **2.1× spread** with an **interior optimum at 256 rows = 3.67 MB/slab** (`results/g25_0b_slab_sweep.txt`) |
| **mediates** | `A`, `u(t)` |
| **axes** | T (slab count = tokens/slab_rows; **chunking collapses below ~2K tokens**, `M25_TP8_MEGA_DESIGN.md:212-215`), PHASE (decode wants smaller slabs), TOPO, CONC |
| **campaign role** | **PRIMARY — it owns a live contradiction** |

**2.4.1 The contradiction, and the experiment cell it demands.** M15/`slab.cuh` says granularity is
free; the CDAR sweep says it is worth 2.1×. Reconciling hypothesis (**SPECULATION**, from
`KNOB_INVENTORY.md` B.3): *granularity is free when it changes only signal **count**; it is expensive
when it also changes **transfer geometry*** — in M15 the signal count changed while the transfer
geometry did not; in CDAR `slab_rows` also changes the store-tower geometry and the gather width.
**Falsifier:** in the CDAR rig, sweep `slab_rows` with the transfer geometry held fixed (vary only
the certificate cut, keeping tower shape and gather width constant). If the 2.1× survives, the
hypothesis is dead and granularity is a first-order bandwidth knob.

**2.4.2 Scope is the axis the two topologies actually differ on** (`TP8_STATE.md` §4.2, DERIVED):
DP/EP places the certificate at the **destination expert**, fan-in from 8 unknown-size,
skew-correlated sources ⇒ **a global rendezvous whose latency is set by the hottest rank**.
TP8+EP places it at the **token-slab owner**, fan-in from 8 structurally-known sources ⇒ **64
independent per-slab rendezvous whose lateness is individually attributable**. Same primitives,
different observability and different failure mode. **The empty-contribution rule** — a rank that
contributes nothing must *still* publish its `src_done` word (`ers_publish_empty`,
`m25_cdar.cuh:33-36`, `:240-247`) — moves from edge case to hot path at TP8 boundary 2.

**2.4.3 The law that governs granularity regardless** (LAW-23): *signal the smallest
early-**EXECUTABLE** unit, not the smallest stored or transferred unit.* Under the natural GEMM-2
order **P(token ready by t) = (t/S)^8** and the **median token is not reducible until 91.7% of
GEMM-2 completes** (`../BRIEF.md:59-61`). The exponent is **top-k**: at top-k 1–2 fine signals would
have a real shadow; at top-k 8 they do not. This is the clearest example in the ledger of a knob
whose sign flips on a workload parameter. Independent vendor-side confirmation: an LL128-style
per-record flag path is **44% slower at nbs=1 and 177% slower at nbs=64**
(`aug11/OVERLAP_METHODOLOGY_STUDY.md:338-351`).

### 2.5 K5 — PRODUCER ORDER

| field | value |
|---|---|
| **domain** | {`producer_major` (tile-major, donor default) · `consumer_major` (nc-major, M15) · `owner_major_staggered(rank)` (ring inducer) · `source_interleaved` (m17 coprime-stride RR)} |
| **provenance** | `KNOB_INVENTORY.md` B.2 |
| **sites** | primitives `order.cuh:50-95` (all four, branch-free one-expression maps); live decode `k0pf6gm_device_tile_m15.hip:880-883`, consumed `n2_phase2_gm_mps.cpp:479-483`; m17 RR behind `K0P6_M15_SCATTER_RR` (`:81-83`, default 0) |
| **c/r** | **compile-time** — a macro-substituted decode inside the hot task loop; making it runtime is an explicit non-goal because order lives at peak register pressure (`order.cuh:15-18`) |
| **measured** | **order alone ≈ 0**; **order × certification = −144 µs of combine residue** (`OVERLAP_ABSTRACTIONS.md:80`). Mediating measurement: the 91.7% readiness pathology above; `order.cuh:21-24` records "no consumable row before 93.9% completion" under tile-major |
| **mediates** | `u(t)` |
| **axes** | SKEW (m17's destination rotation is the zero-protocol alternative to per-destination credits), TOPO (`owner_major_staggered` is what makes ring reduce-scatter *emerge* — `OVERLAP_ABSTRACTIONS.md:154-160`), T, ARCH (nc→XCD L2 residency invariant) |
| **campaign role** | **PRIMARY as an interaction term only.** The law: *order sets the arrival time of the certified prefix; K4 and K6 are worthless without it.* Never sweep it alone — it is the cheapest lever in the program and its main-effect is zero |

**Unbuilt sub-knob:** rotated-nc phase offset `nc' = (nc + gid·φ) mod 16` to de-align epilogue
bursts; ceiling priced at **+211 µs** from exp_25 (`aug12/OVERLAP_KERNEL_DESIGN_IDEAS.md:126`).

### 2.6 K6 — CONSUMER POOL (`C`, quota, fall-in policy)

| field | value |
|---|---|
| **domain — `C`** | integer 0…128 (`config_is_valid`: >128 rejected, ≥256 rejected, `moe_mps_adapter.cuh:591-592`); documented earlier cap `C ≤ 64` (`BENCHMARKING.md:192-194`). Mode 3 requires C a multiple of 32 ⇒ legal {32, 64}; mode 1 requires `C == 0`; stream/diagnostic-pool modes require `C != 0` (`:661-669`) |
| **domain — `flush_rows`** | 1…64 (`:670`); design sweep {0, 8, 16, 32} where **0 disables the mid-M7 sweep and isolates pure protocol deletion** (`aug12/M15_DESIGN.md:87-88`) |
| **domain — fall-in** | {static stripe · dynamic ticket + compute-CTA fall-through} |
| **provenance** | `KNOB_INVENTORY.md` B.5, B.6 |
| **sites** | role predicate `is_service_cta` (`moe_mps_adapter.cuh:673-682`); dense compute id `:684-691`; task start/stride `k0pf6gm_device_tile_m15.hip:793-802`; the pool's job `:2102-2160`; RMW quota `:2149-2152`; staged quota `:2196-2200`; `effective_flush_rows` `moe_mps_adapter.cuh:567-569` |
| **c/r** | **runtime** (config bytes 0 and 3) |
| **measured — `C`** | **steepest and most non-monotonic knob in the inventory.** 16 → 6,292.4; 24 → 5,848.5 (0.7589×); **28 → 5,822.0 (0.7544×)**; **32 → unstable, hung 1 of 2 runs**. Deltas −93 µs (8→16), −444 µs (16→24). Under mode 12 the same knob's optimum was `C ≤ 8`. Idle-pool tax +8–9 µs/CTA monotone |
| **measured — `flush_rows`** | **no isolated Δ in-repo.** Its design role is quantified: the pool covers only ~20% of the front half per exp_29's **6.1 GB/s/CTA** law, and "draining all 1,024 batches would outlast slab 1 several times over" (`k0pf6gm_device_tile_m15.hip:2142-2148`) |
| **measured — fall-in** | dynamic ticket + fall-through is worth up to **2.9×** on a role-split kernel (C=16 22,068 → 7,617; C=32 1.97×; C=64 1.34×) — static stripes cause head-of-line blocking (LAW-26) |
| **mediates** | `φ`, `u(t)`, `B` |
| **axes** | T (`:2122-2127`: at other `T_eff` "C / flush_rows are no longer at their tuned point"), FILL, CONC, PHASE, SKEW |
| **campaign role** | **PRIMARY. `C` must be re-swept inside every W-cell**; its optimum is a model *output* |

**Overloading trap for the manifest:** in mode 4 `flush_rows` means pacing units, in mode 8
poll-backoff units, in modes 5/6 a diagnostic window in 100 µs units
(`moe_mps_adapter.cuh:571-581`). **A naked `flush_rows` value is meaningless — every row must carry
`mode` to interpret it.**

**Unbuilt sibling worth naming:** `K0P6_M24_FILL_C` is literally *"runtime-adaptive
`reserved_comm_ctas`"* — the C-vs-T coupling as a knob, declared with a hard `#error`
(`k0pf6gm_device_tile_m15.hip:262-270`, `:294-302`). It is the mechanism the cost model would
*deploy* once it can predict `C*`.

### 2.7 K7 — SLAB COUNT / PARTITION / ITEM GRANULARITY

| knob | domain | site | c/r | default | measured | axes |
|---|---|---|---|---|---|---|
| `K0P6_M15_SLABS` | S (column slabs) | `:498-500` | compile | 2 | co-determined with `M8_CSPLIT=7` by the column-3,584 alignment argument | T |
| `NC_PER_SLAB` | nc chunks per slab | `:499` | compile | 8 | — | T |
| `M8_CSPLIT` | front/back split index | `:500` | compile | 7 | — | — |
| `slab_rows` (CDAR) | rows/slab | `m25_cdar.cuh:79-101`, argv 3 | **runtime** | 256 (rig) / 512 (doc) | interior optimum at 256; 1024 is **2.1×** worse (§2.4) | T, PHASE |
| `bps` (fragments/slab) | item granularity in a slab | `m25_boundary_bench.hip:448`, used `:185-190` | runtime | 2 | atomics @512-row: **bps 1 → 41.0, 2 → 67.1, 4 → 66.8 GB/s** — *one block per slab leaves the grid half-idle*; in the **fused** schedule bps=16 was **WORSE (+180 µs of protocol, still zero hiding)** | T, PHASE |
| `NT` (M8 combine batch) | rows per dynamic ticket | `k0pf6gm_device_tile_m15.hip:2440` | compile | 4 | not isolated | T, FILL, PHASE |

### 2.8 K8 — EXPERT-OWNER PLACEMENT (the SKEW actuator)

| knob | domain | site | c/r | default | measured | axes |
|---|---|---|---|---|---|---|
| **placement map** | contiguous ("linear") · round-robin `e % 8` · EPLB-derived | `aug18-prefill/rr_placement/rr_patch.py`, `RR_PLACEMENT_NOTES.md:208-236` | host | contiguous | linear: per-rank receive share **63.60% / 6.93 / … → max/mean 5.088×, max/min 13.84×**; RR: **max/mean 1.085×**; worst-rank layer **5.593× → 1.296×** | SKEW, TOPO |
| `K0P6_M15_REPLICATE` (M18) | static hot-expert replication, `E + nrep ≤ 64` | `k0pf6gm_device_tile_m15.hip:99-101` | compile | 0 | motivating: **51.9% of expert-slot traffic to experts 0–7, all on rank 0**; aggregate receive load **5.09×** fair share, worst layer 5.59×; both kernels degrade **~3.2–3.6×** under replay | SKEW |
| `K0P6_M15_ADAPTIVE` (M19) | per-chunk replication, theta = M18R header word 3 | `:120-124` | compile (theta runtime) | 0 | per-call coverage bimodal (**p5 = 1%, p75+ = 77%**); per-call max rank load **6.25× at p95**; serving pair #1 measured **−5.7% e2e against a 4× aggregate-replay kernel win** | SKEW, T |
| `K0P6_M20_SLOTPOOL` | replica weights in a triple-buffered cross-layer pool (~1 GB vs 41 GB) | `:147-149` | compile | 0 | deleting the M0.5 pre-pass + two grid-barrier rendezvous priced at **+1.2 ms measured** | SKEW, T |
| `K0P6_M15_SCATTER_RR` (m17) | source-rotating scatter so a depth-4 window spans ~7 links | `:81-83` | compile | 0 | RR **landing order** buys ~1–1.5% under skew (0.8467 → 0.8370 aggregate; 0.8559 → 0.8413 worst-layer): **the bottleneck is popularity concentration, not ordering** | SKEW, TOPO |
| **EPLB** (`num_redundant_experts=0`) | expert→slot map resolved inside the router | `M15_EPLB0_COMPOSE.md` | host | off | compose is sound; **not zero-memory: ≈1.31 GiB/rank** transfer buffer. **Never composed with the megakernel** (the 128-redundant config violates `PF4H-CONTRACT-001`) | SKEW, TOPO |

**The binding scope note:** the offline replication frontier (LAW-47) is ANALYTIC over aggregate
histograms — uniform-K **K=2 → 4.10×, K=4 → 3.23×, K=8 → 1.77×, K=16 → 1.09×**, greedy ≈ uniform
+ ε at equal bytes — and its named risk is that *the aggregate frontier cannot price the per-chunk
coverage bimodality*. And **ideal static redistribution still leaves per-call p95 at 3.05×**
(`RR_PLACEMENT_NOTES.md:246-262`). Placement is a static remedy against a per-call problem.

**The instrument problem that gates this whole group:** the two per-call skew instruments
**disagree by ~2×** (m18diag p50 **5.15×** / p95 6.25× at n=90,050 calls vs routecap1 p50 **2.61×**
/ p90 3.35× at n=512 rank-calls) and **must not be mixed** — reconciling them is CPU-only and
changes the modeled skew tax by ~2× (`WORKLOAD_EVIDENCE.md` W4b, H17).

### 2.9 K9 — LAYOUT / COALESCING

| knob | domain | evidence | rule |
|---|---|---|---|
| **payload address stream** | coalesced ⟷ scattered lane addresses | **52.8 → 4.1 GB/s, a 13× collapse** (`../BRIEF.md:57-58`, LAW-3) | always coalesce payload; audit in ISA (`M25_TP8_MEGA_DESIGN.md:150-153`) |
| **protocol counter layout** | 32 lanes over 32 lines ⟷ 2 lines | exp_07 transposed arrival counters onto 2 cache lines and the config made **no measurable progress in 12 minutes against a usual 25–95 s** ⇒ **≥10× is a lower bound** (LAW-18) | **never coalesce atomics** — an atomic resolves at the line, so RMWs to 2 lines serialize while 32 over 32 lines proceed across L2 banks |
| **consumer-side gather layout** | group-major ⟷ token-major | exp_27: **5,376 lines to deliver 21,504 B → 384 lines, 14.0× less line traffic, ΔM6 = −79.7 µs (t = −14.8)**; producing the group-major copy cost **−2.95 µs (n.s.)** (LAW-36) | **price a layout conversion on the CONSUMER's line traffic** |
| **record order across destinations** | grouped ⟷ destination-interleaved | MORI-measured **65%** on push-combine; their own **822 → 498 µs** RR-interleave fix (LAW-51) | interleave under **any** destination concentration; cheap |

**These four rows are one law with opposite signs for loads and atomics.** That inversion is
exactly the kind of thing the HipKittens PR should teach (§5.4).

### 2.10 K10 — GRID GEOMETRY / ROLE PLACEMENT

| knob | domain | site | c/r | default | measured | axes |
|---|---|---|---|---|---|---|
| grid size / occupancy | 256 CTAs, `__launch_bounds__(256,1)`, 1 block/CU | `M25_TP8_MEGA_DESIGN.md:120-121`; LAW-12 | compile | 256 / occ 1 | not swept — it is the axiom the selection rule rests on | ARCH, TOPO |
| pool XCD placement (mode 3) | tail reservation ⟷ whole-die residue classes | `moe_mps_adapter.cuh:218-232`, `:673-682` | runtime (mode) | tail | **XCD-confined placement 7–23% worse**; localization cut interference 36% but starved the pool 54% — **net loss** (LAW-19). Note the dispatch rule: `xcd = wg_id % 8`, chunk size one, so contiguous `bid < C` **spreads** and strided `bid % 8 == 0` **concentrates** | ARCH, TOPO |
| role-assignment mechanism | static tail split ⟷ `finish_order_partition` | `roles.cuh`; M25 keeps static — "finish_order_partition stays rolled back — 24 B/lane spill precedent" (`M25_TP8_MEGA_DESIGN.md:122-123`) | compile | static tail | qualitative | ARCH |
| prefetch CTAs `K0P6_M20_PF_CTAS` | CTAs streaming next-next layer replica weights | `k0pf6gm_device_tile_m15.hip:432`, used `:2091-2100` | compile | 8 | not isolated | T, SKEW |
| planner CTA | 1 rolling-planner CTA (M25) | `M25_TP8_MEGA_DESIGN.md:126-131` | design | 1 | **unmeasured (SPECULATION as a win)** | T, SKEW |
| `k_inner` (CDAR rig) | MFMA burst depth ⇒ the compute:transport ratio | `m25_boundary_bench.hip:447` | runtime (argv 6) | 64 default / 2048 as run | **the rig ran at ~1/5 the deployment compute:transport ratio** — see §1.8 | **PHASE, T** |

### 2.11 K11 — FILL / EXTENT (the FILL and T actuators)

| knob | domain | site | c/r | default | measured | axes |
|---|---|---|---|---|---|---|
| `MAXTOK` | per-rank slot capacity; **must be a power of two** in mode 12 | descriptor `K0P6_D_MAXTOK` (`:392`); `OVERLAP_KERNEL_DESIGN_ADDENDUM.md:268-285` | runtime (host) | over-provisioned by TOPK | **the only confirmed production win on this axis: `MAXTOK = T` measured −1.49% (~148 µs) at prefill** | T, PHASE, FILL |
| `K0P6_M24_FILL` | re-derive `T` from a per-layer `n_orig` vector | `:182-186` | compile | 0 | mechanism motivated by the **37.6% fill / 2.66× padding multiplier** at c32p; **never compiled** (`WORKLOAD_EVIDENCE.md` §3.2) | **FILL**, T, CONC |
| `K0P6_M24_TGRAIN` | rounding granularity of `T_eff` | `:189-190` | compile | **256** | conservative: the read-only `csr_scan_block256` / `pull_src_fill` headers "have only ever been exercised at T ∈ {4096, 2048, 1024}" | T |
| `K0P6_M24_ZERO_PAD` | zero `out[T_eff, T_cap)` | `:197-199` | compile | 0 but **mandatory** with FILL (`#error` `:279-281`) | correctness | FILL |
| `K0P6_M24_NORIG_CONST` / `_TABLE` | payload-value substitution; `_TABLE` is a per-rank brace list | `:228-229`, `:239-243` | compile | off | **the instrument for a synthetic fill/heterogeneous-fill sweep without serving** | FILL, SKEW |
| `K0P6_M24_NULLWORK` | receive-side null-work skip | `:215-217` | compile | 0 | **demoted to diagnostic** — a DP-dummy rank still owns 32 experts under EP8 and still receives from every busy peer | FILL, TOPO |
| `spin_limit` | fail-closed bound on every device poll | kernel arg (`aug12/M15_DESIGN.md:7-8`) | runtime | `20_000_000` in serving | it is the **failure boundary** for cross-rank arrival skew on ragged batches | SKEW, CONC, T |

**Fill is bimodal, not a dial** — the most decision-relevant shape fact in the corpus. Deciles of
`n_real` over 512 rank-calls: **0, 0, 5, 1994, 4096, 4096, 4096, …** ⇒ ~31% near zero, ~60%
completely full, **only ~10% genuinely partial** (`WORKLOAD_EVIDENCE.md` §W3). Probable cause
(**SPECULATION**, cheap to test): ISL 4096 == `max_num_batched_tokens` 4096, so a DP rank has either
a whole chunk or nothing. **If so, every tier-1 sizing number is cell-specific.**

### 2.12 F — FUSION BOUNDARY (which ops live inside the launch)

| id | boundary move | vehicle | c/r | status | evidence / price |
|---|---|---|---|---|---|
| **F1** | **collective inside vs outside** — RCCL AR vs in-epilogue CDAR | M25 | build | **interim negative** | fused 3,606 vs GEMM→RCCL 2,909 µs; RCCL AR 1.5× our transport. Amdahl prize: ~2.8 ms/layer exposed AR = **171 ms of a 634 ms step ≈ 27% ⇒ perfect hiding = 1.37× ⇒ ~35.4k tok/s** (DERIVED, `TP8_STATE.md` §1.4) |
| **F2** | shared expert inside (as bubble filler) | `K0P6_M15_SHEXP`, default 0 | compile | design only | it is the **only** same-epoch work with zero dependency on dispatch/plan/sort/peers (`SHARED_EXPERT_FILLER_DESIGN.md:17-30`); **bounded by construction at ~1/8 of routed FLOPs** — real but capped, must not be sold as the answer to a 28% pool (`TP8_STATE.md` B4) |
| **F3** | RMSNorm inside, owner-sharded | M25 step (b)/(g) | design | unbuilt | "8× less norm work + one less full HBM pass per boundary; small but free" (`M25_TP8_MEGA_DESIGN.md:161`) |
| **F4** | router/gate GEMM inside (rolling planner) | M25 step (c) + planner CTA | design | unbuilt | production serializes gate→sort→group kernels |
| **F5** | attention inside | M25 §8 doors (a) in-kernel MLA, (b) chunked attention on a hipGraph with per-slab events, (c) persistent co-kernel | design | **v1 deliberately leaves it out** | the exposed MAG#2 tail is v1's structural bubble, ~`1/n_slabs` + launch gap; at 64 slabs ~1.6% of MAG#2 + gap |
| **F6** | quantization on the wire (fp8 combine + source pre-reduce) | `FP8_WIRE_DESIGN.md` | design | **blocked** | **not reachable from the mode-12 RMW epilogue at all** — gfx950 has no fp8 remote RMW; reachable only atop S8, whose scratch pathology is P0. Under TP8 the sign of its central risk **flips in our favour**: the risk is "converting hidden wire time into exposed wire time", and under TP8 the wire time is already exposed (`TP8_STATE.md` A5) |
| **F7** | cross-layer weight prefetch inside | M20 slot pool | compile | built, gated | **stream order *is* the certification; no flags, no tags** (`:134-142`) |
| **F8** | backward / wgrad inside | T2B (`K0P6_M15_SAVE_Z`), T3 wgrad-as-tickets | compile / design | forward-save built | in-launch bubble supply is **~150–180 CTA-ms vs wgrad demand ~3,360 CTA-ms** — every wgrad rescheduling variant lost (LAW-53) |
| **F9** | launch boundary as a pipeline stage | mode 16 | runtime (TBO build) | **falsified** | **+75.8 µs** |
| **F10** | graph-capture / seal boundary — which serving steps the mega may run | M23 ragged seal (**zero lines of HIP**) | host | design ready → shipped | coverage **2% → 100% of in-bucket = 0.77% → 38.3% of all steps, ~50×**. Also: on an unsealed in-bucket step a PF4H-target server fell back to **no graph at all**, not the stock B4096 graph — **every pre-M23 serving A/B measured a doubly-handicapped candidate** |

**Why the fusion boundary is a schedule knob and not packaging** (`KNOB_INVENTORY.md` C.3), two
measured reasons: (1) **fusing the collective is what lets K4 certificates exist at all** — unfused,
the boundary degenerates to a phase barrier (S=1); (2) **fusion boundary sets barrier count which
sets drift-realization count** — three realizations per launch vs one per layer
(`T3_COUNTER_DATAFLOW_DESIGN.md:43-49`).

### 2.13 I — INSTRUMENT PINS (tier 2: not schedule, but part of the number's identity)

| pin | site | why load-bearing |
|---|---|---|
| `K0P6_MPS_ENABLE_MODE14` | `moe_mps_adapter.cuh:57-58` | making mode-14 code **reachable** cost the mode-12 ratchet **+726.9 µs** with mode 12's source untouched. ***Never publish a mode-12 number from a MODE14=1 binary.*** |
| `K0P6_MPS_ENABLE_TBO` | `:61-69` | same discipline, same measured reason |
| `K0P6_M15_SRC_REV` | `k0pf6gm_device_tile_m15.hip:63`, bumped to 3 at `:314-317` | JIT-cache guard; a stale `.hsaco` silently measures the wrong body |
| `K0P6_M15_KERNEL_NAME` | `:491-493` | symbol pin so a RUN PIN branch compiles under the harness's expected name |
| `timestamps` bit | `moe_mps_adapter.cuh:170` | phase stamps resolve ~1% vs ~6.6% for screens |
| descriptor slot exclusivity | `#error`s at `:103-105`, `:159-165`, `:279-312` | **slot 63 is claimed by STAGED / REPLICATE / SAVE_Z — three arms structurally cannot compose** |
| occupancy print | LAW-12 corollary | occupancy must be asserted, not assumed |
| `[MPS TS]` reset discipline | LAW-63(c) | stamps are **running maxima never reset** — after a 600-epoch soak they describe the final *soak* epoch |

**The general law:** *the instrument is part of the kernel* (`OVERLAP_ABSTRACTIONS.md:198`). A knob
manifest that does not pin these is not reproducible.

---

## 3. PART III — CONSTRAINT MATRIX

Pruning is how the campaign stays tractable. Four classes, in decreasing hardness.

### 3.1 Class A — HARD ILLEGAL (the build or the validator rejects it)

| # | constraint | enforced at | consequence for the grid |
|---|---|---|---|
| A1 | `C > 128` or `C ≥ 256` rejected; earlier documented cap `C ≤ 64` | `moe_mps_adapter.cuh:591-592`; `BENCHMARKING.md:192-194` | C domain is {0…64} in practice |
| A2 | mode 1 requires `C == 0`; stream/diagnostic modes require `C != 0` | `:661-669` | the (mode, C) product is not a rectangle |
| A3 | mode 3 requires `C % 32 == 0` ⇒ legal {32, 64} | `:667-668`, `:220-232` | XCD-placement arm has exactly 2 points |
| A4 | physical `g` must be 1 in direct-accumulate modes | `:596` | `g` sub-field is not free under S3/S4 |
| A5 | `flush_rows ∈ [1, 64]` | `:670` | "flush_rows = 0" is a *design* value (`M15_DESIGN.md:87-88`), not an encodable one — **must be implemented as a separate mechanism bit, not as the value 0** |
| A6 | **descriptor slot 63 exclusivity**: STAGED / REPLICATE / SAVE_Z | `#error`s at `:103-105`, `:159-165` | **S8 × M18 × T2B cannot compose** — three arms are mutually exclusive by construction |
| A7 | `K0P6GM_G != 3` is a hard `#error` | `:502-506` | the only compute-side knob with a quantified prediction (G=4: intensity 161.7→204.8, predicted M6 ≈ 1,957 µs) is **pinned shut** |
| A8 | `MAXTOK` must be a power of two in mode 12 | `OVERLAP_KERNEL_DESIGN_ADDENDUM.md:268-285` | MAXTOK sweep is dyadic only |
| A9 | slab entry guard `world·MAXTOK + SLABS ≤ T_loc_max` (32,770 ≤ 40,960 today) | `:1254` | binds any joint MAXTOK/S change |
| A10 | `K0P6_M24_ZERO_PAD` mandatory with `FILL` | `#error` `:279-281` | FILL arms are always (FILL, ZERO_PAD) pairs |
| A11 | depth cannot be a register (`vmcnt(N)` is simm16-only) | `credit.cuh:12-22` | depth is compile-time codegen + runtime *selection*; a "runtime depth" design is illegal |
| A12 | producer order is compile-time by design (peak register pressure) | `order.cuh:15-18` | 4 orders = 4 binaries, not 4 config points |
| A13 | `K0P6_M24_DENSE_SCATTER` / `_FILL_C` / `_M8_ADAPT` are declared but `#error` | `:262-270`, `:294-302` | the C-vs-T adaptive knob does not exist yet |

### 3.2 Class B — DOMINATED A PRIORI (a banked law rules it out; keep exactly one control point)

| # | pruned combination | banked law | keep as control? |
|---|---|---|---|
| B1 | S5 / F9 (cross-launch deferred combine) anywhere | **+75.8 µs**, `OVERLAP_ABSTRACTIONS.md:52-53` | 1 point, once, as a named negative |
| B2 | mode-3 XCD-confined pool placement | **7–23% worse**; 36% less interference but 54% pool starvation (LAW-19) | 1 point — a known-bad control that usefully bounds interference |
| B3 | `transport = atomic_accumulate` at bulk message size on gfx950 | F5: **67 vs 117 GB/s** at 235 MB (`TP8_STATE.md` §2.2) | **must be kept at decode size**, where it *wins* (801 vs 1,017 µs) |
| B4 | `slab_rows = 1024` (CDAR) | F7: **2.1× slower** than 256 | drop; keep 128/256/512 |
| B5 | `bps = 1` | one block per slab leaves the grid half-idle: **41.0 vs 67.1 GB/s** | drop |
| B6 | `bps = 16` in the fused arm | F4: **+180 µs of pure protocol, still zero hiding** | drop |
| B7 | pacing a *dedicated pool* | LAW-22: `M2→end` rises monotonically 6,229 → 8,134 µs, **no interior minimum ⇒ pacing 0 is already optimal**; "every Tier-3 scheduling idea premised on throttling the pusher is closed" | drop entirely |
| B8 | coalescing protocol atomics onto few lines | LAW-18: **≥10×** slower (lower bound) | never build |
| B9 | per-row certification at top-k 8 | LAW-23: median token not reducible until **91.7%** of GEMM-2; vendor LL128 path 44–177% slower | **keep at low top-k only** — the exponent is top-k, so the sign flips there |
| B10 | deleting the per-task VMEM drain | LAW-33: **+3.2 µs, sign reversed** | drop |
| B11 | S2 (carrying pool) at T=4096 balanced prefill with a producer-carried epilogue | LAW-28, 27 campaigns, monotone over a 374 µs span | **DO NOT prune elsewhere** — see 3.2.1 |
| B12 | "we move −32% fewer bytes" as a claim | `TP8_STATE.md` §1.3: re-derives to **≈ −9.7%** | retract; the claim itself is pruned, not an arm |

**3.2.1 The three prunes that must NOT be generalized.** Each of these is regime-scoped, and
over-pruning them would delete exactly the cells the experts asked about:

* **B11 is carrier- and regime-conditional** (contradiction C3): dedication *won* at C=64 in the
  staged-push carrier. And LAW-62(b) records that the one regime that could rescue a pool — skew —
  **was never settable** on that harness. "Dedication is bad" is not a technique verdict.
* **B3 reverses with message size** — K1 is a *second manifest axis*, not a global decision.
* **B9's exponent is top-k**, i.e. a workload parameter. At top-k 1–2 fine signals have a real
  shadow.

### 3.3 Class C — WORKLOAD-INVALID (the (skeleton × knob) is meaningless in that W-cell)

| # | cell | what becomes invalid | source |
|---|---|---|---|
| C1 | **TOPO = TP8+EP** | **the entire FILL lever dies**: no DP lockstep ⇒ no `execute_dummy_batch`, no 56%-dummy steps; the M23 ragged seal is a per-rank DP batch concept; the MORI all2all machinery is structurally absent | `TP8_STATE.md` §3.3 — *"our high-concurrency +8.40% win is mediated by fill, and fill is a DP-topology quantity. It does not generalise to TP8 even in principle."* |
| C2 | **co = 0** (pure transport) | **K3 (depth) is inert** — flat across {4,8,16,32} | F6 / LAW-8 |
| C3 | **T < ~2K tokens** | **K4 chunking collapses**; slab-based certification has no staircase | `M25_TP8_MEGA_DESIGN.md:212-215` |
| C4 | **T ≠ 4096 on the serving M15 body** | **every arm is void** — the body **failed the MoK correctness gate at T=2048/1024** (rel 0.874205 / 0.838890) while `production` passed in the same runs | `WORKLOAD_EVIDENCE.md` §W6 |
| C5 | **any SKEW cell on the MoK harness** | **the axis does not exist**: `K0_SYNTH_ROUTE` rejected under `K0_INPUT_MODE=mok_synthetic`, no `std` parameter, reachable CV **0.0034–0.0086** vs COMET's 0.032 | LAW-62(b) — *untestable, not refuted* |
| C6 | **any T-sweep framed as a fill sweep** | `K0_MAXTOK == K0_T` ⇒ capacity shrinks with T ⇒ **it is a batch-size sweep, never a fill sweep** | `WORKLOAD_EVIDENCE.md` §W6, `REPORT.md:435-438` |
| C7 | **any pre-M23 serving pair** | retired — the seal fired on ~2% of heavy steps and the candidate ran ~225 of ~230 heavy steps **eagerly** | LAW-49; retraction list item 6 |
| C8 | **exact-token SHA as a parity gate at c32p** | the kernel is **not bit-reproducible run to run** (bf16 remote atomics, order set by a per-step tile schedule); `ordered_output_token_id_stream_sha256` differs even **stock vs stock** | `WORKLOAD_EVIDENCE.md` §W10(d) |
| C9 | **decode claims below ~2K tokens** | **no basis exists** — the RCCL reference at decode message sizes (G25-0b(iii)) was never run | `TP8_STATE.md` §5 M5 |
| C10 | **serving deltas < ~15%** at c32p-era variance | must be adjudicated at the boundary, not e2e (but see the c512p refinement: **≤1.8% throughput / ≤0.25% latency** there — *the variance envelope is itself a function of W*) | LAW-45; `WORKLOAD_EVIDENCE.md` §W10(b) |

### 3.4 Class D — MEASUREMENT-VOID (the arm is legal and the number is still meaningless)

| # | void condition | law |
|---|---|---|
| D1 | a mode-12 number published from a `MODE14=1` or `TBO=1` binary | LAW-32 (**+726.9 µs** from unreachable code) |
| D2 | a `flush_rows` value reported without its `mode` | `moe_mps_adapter.cuh:571-581` (overloaded 4 ways) |
| D3 | any S8 (`STAGED=1`) × depth arm before the scratch P0 is fixed | `FP8_WIRE_DESIGN.md:24-29` — **96 scratch loads + 97 `vmcnt(0)`** within 24 instructions of the remote atomic |
| D4 | a "knob is inert" claim without `.text` sha identity, **and** a flag-ON build that differs | LAW-58/59 — a build passed **every** resource-tuple field and still cost **+726.9 µs**; "a flag-on build that came out identical would mean the flag never reached the code and the arm would be a lie" |
| D5 | a per-rank failure adjudicated from rank-0 stdout | LAW-61(c) — mode-14's negative control printed `pperr=0` on rank 0 while **rank 7 carried the pre-registered failure signature** |
| D6 | (run, rank) cells treated as independent units | LAW-61(a) — the same null reads **t = −5.01** clustered wrong and **t = −0.73** with runs as units |
| D7 | any phase number compared tick-for-tick across instruments | LAW-63(d) — the MPS phase print is the **CTA-max within rank 0**; production's cross-rank combine reads **1,356.0 rank-max vs 978.4 rank-0 (+38%)** |
| D8 | a serving delta quoted without its arm's `sealed/in_bucket` | LAW-49 standing rule |
| D9 | a knob "set" via an env var the harness does not forward | `BENCHMARKING.md:87-101` — see §5.3 |
| D10 | an end-to-end screen delta below ~3% | LAW-60 — documented σ = 0.52% but a repeated control drift of **6.6% inside one batch** |

### 3.5 The pruning arithmetic (why this matters)

Full crossing of S4's own knobs at **one** W-cell:
`C(6) × depth(5) × flush_rows(4) × order(4) × K1(3) × slabs(2) × mode(3) = 8,640` arms.
Over six W-cells that is **51,840 campaigns**; at LAW-60's measured **~3 min 5 s per 5-rotation
campaign** that is **≈2,679 node-hours**. The constraint matrix plus the screening design of §5.1
takes S4 to **~30 knob points per cell** — a **288× reduction** — and the whole campaign to the
budget in §5.5.

---

## 4. PART IV — MANIFEST SCHEMA AND NAMING

### 4.1 The rule

> **One row = one arm = one regenerable number.** A reported number that cannot be regenerated
> from its row is not a result; it is an anecdote.

Three identity checks are required, not one, because this repo has been burned by each failure mode
independently: `.text` sha (LAW-58), device-side descriptor echo (§5.3's env-forwarding trap), and
the mechanism invariant (LAW-59's ISA gate).

### 4.2 Where manifests live

```
distributed-kernels/fused_moe/overnight/aug18-ablations/manifests/
├── cells.jsonl      # one row per W-cell (the workload state vector)
├── builds.jsonl     # one row per HSACO  (compile-time knob vector b + .text sha)
├── arms.jsonl       # one row per schedule point S = (K, b_ref, r, g, F)
├── runs.jsonl       # one row per execution: arm × cell × n × gates × result refs
└── replay/<arm_id>.sh   # GENERATED from arms.jsonl, never hand-written
```

Result **artifacts** stay where the repo already puts them, and rows point at them by path:

* boundary/kernel rigs → `distributed-kernels/tp8_mega/results/*.txt` (precedent:
  `g25_0b_v1_results.txt`, `g25_0b_slab_sweep.txt`, `G25_1_STATUS.md`);
* e2e serving → `distributed-kernels/tp8_mega/results/serving/*.json` (precedent: the six banked
  manifests `b0v5_pair_01_*`, `b3cal_pair_01_*`, `b3rev2_pair_01_*`, schema
  `pf4h-exact-token-closed-concurrency-v2`).

Normalizing into four tables (rather than one flat CSV) is deliberate: **builds are shared across
many arms**, and a build's identity is the thing LAW-32 says silently changes results.

### 4.3 Naming scheme (legible at hundreds of arms)

```
arm_id  ::=  <cell>__<skeleton>__<delta>__<build8>
```

* `<cell>` — the W-cell id, itself structured: `<class>.<key>.<variant>`
  * `K.T4096.bal` (kernel rig, T=4096, balanced routing), `K.T1024.hist`, `K.T4096.agg`
  * `B.tp8.16k`, `B.tp8.256` (boundary rig, tokens)
  * `E.c32p.dp`, `E.c512p.dp`, `E.c32p.tp` (e2e serving cells)
* `<skeleton>` — `S0`…`S8`, `D1`
* `<delta>` — dot-joined `key=value` for **only the knobs that differ from the skeleton's declared
  default**, emitted in a fixed canonical key order so the string is deterministic and sortable:
  `C28.d4.fr16.o=nc.k1=rmw.sl=256`
* `<build8>` — first 8 hex of the build's `.text` sha256 (LAW-58: *the only parity gate we trust*)

**Examples**

```
K.T4096.bal__S4__C28.d4.fr16__9f2a1c07        # the shipped ratchet, kernel rig
K.T4096.bal__S4__C28.dOFF.fr16__9f2a1c07      # its depth control, same binary
K.T4096.bal__S3__C16.d4.fr16__9f2a1c07        # mode-12 rung, same binary
K.T4096.agg__S4__C24.d4.fr16.o=rr__41b8ee52   # m17 order variant -> different binary
B.tp8.16k__S6__k1=tow.sl=256.d4.bps2.ki8192__c70d19aa
B.tp8.256__D1__k1=atm.sl=64.d4.bps2__c70d19aa
E.c512p.dp__S4__C28.d4.fr16__9f2a1c07         # e2e validation of the same arm
```

Properties this buys: sortable (cell groups together), greppable (`grep '__S4__'`,
`grep 'd4\.'`), diff-legible (two arms differing in one knob differ in one token), and
**collision-proof across source revisions** because the build hash is in the key. Human labels
("the ratchet", "camp3 pair 2") may be attached as an `alias` field but are **never** the key —
`WORKLOAD_EVIDENCE.md` R1 records a real pin trap where two archived campaigns named
`t2048_m15_*` carried sha `20b8c6bb` (DHK-tgen), not M15's `935f555e`, and were retracted.

### 4.4 Schema — `builds.jsonl`

```jsonc
{
  "build_id": "9f2a1c07",                       // first 8 of text_sha256
  "skeleton": "S4",
  "source": "distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip",
  "src_rev": 3,                                  // K0P6_M15_SRC_REV
  "kernel_name_pin": "k0pf6gm_m15",              // K0P6_M15_KERNEL_NAME
  "defines": {                                   // the FULL -D set, no omissions
    "K0P6_M15_STAGED": 0, "K0P6_M15_SCATTER_RR": 0, "K0P6_M15_REPLICATE": 0,
    "K0P6_M15_ADAPTIVE": 0, "K0P6_M15_SAVE_Z": 0, "K0P6_M20_SLOTPOOL": 0,
    "K0P6_M24_FILL": 0, "K0P6_M24_TGRAIN": 256, "K0P6_M24_ZERO_PAD": 0,
    "K0P6_M24_NULLWORK": 0, "K0P6_M24_STRICT": 0,
    "K0P6_MPS_ENABLE_MODE14": 0, "K0P6_MPS_ENABLE_TBO": 0,   // instrument gates
    "K0P6_M15_SLABS": 2, "K0P6_M15_NC_PER_SLAB": 8, "K0P6_M15_M8_CSPLIT": 7,
    "K0P6GM_G": 3, "K0P6_M15_ORDER": "consumer_major"
  },
  "text_sha256": "9f2a1c07…",
  "resources": { "sgpr": 0, "vgpr": 0, "agpr": 0, "lds_bytes": 155428,
                 "scratch_bytes_per_lane": 128, "scratch_op_count": 19,
                 "sgpr_spills": 186, "vgpr_spills": 15 },   // LAW-58: counts, not just size
  "isa_gate": { "vmcnt_literal_sites": ["n2_phase2_gm_mps.cpp:116-127"],
                "scratch_ops_within_24_of_remote_atomic": 0,
                "atomic_issue_runs": {"atomics": 282, "runs": 12, "mean_in_flight": 23.5} },
  "parity": { "claimed_inert_vs": null, "text_identical_to": null, "flag_on_differs": true },
  "compiler": "…", "rocm": "…", "built_utc": "…"
}
```

`resources` and `isa_gate` are mandatory because **the resource tuple is not a parity gate**
(LAW-58) and because a depth bound *can silently die with no source change* (`credit.cuh:30-38`).

### 4.5 Schema — `arms.jsonl` (this is `S`)

```jsonc
{
  "arm_key": "S4__C28.d4.fr16",     // skeleton + delta; cell-independent
  "skeleton": "S4",
  "build_id": "9f2a1c07",
  "runtime": {                       // r — the packed config word, decoded
    "config_word_env": "C=28,g=353,mode=12,flush_rows=16,timestamps=1",
    "C": 28, "mode": 12, "flush_rows": 16,
    "g": { "raw": 353, "physical": 1, "dual_write_detect": 0, "throttle_enable": 1,
           "skip_part_zero": 1, "depth_sel": "01", "depth": 4, "mode14_keepdrain": 0 },
    "pull_fallback": false, "timestamps": true
  },
  "geometry": { "T": 4096, "MAXTOK": 4096, "world": 8, "TOPK": 8,
                "spin_limit": 20000000, "grid": 256, "occupancy_blocks_per_cu": 1 },
  "fusion": { "F1_collective_inside": true, "F2_shexp": false, "F7_prefetch": false,
              "F10_seal": "m23_ragged" },
  "knob_deltas_vs_skeleton_default": ["C=28"],   // what the name encodes
  "mediating_tags": ["u(t)", "phi"],
  "campaign_role": "primary"
}
```

For an S6/CDAR arm the `runtime` block is the rig argv, which **already is** the schema
(`m25_boundary_bench.hip:441-448`):

```jsonc
"runtime": { "mode": "fused", "tokens": 16384, "slab_rows": 256, "iters": 7,
             "depth": 4, "k_inner": 8192, "bps": 2, "transport": "store_towers" }
```

### 4.6 Schema — `runs.jsonl` (arm × cell × measurement)

```jsonc
{
  "run_id": "2026-08-19T04:12Z__K.T4096.bal__S4__C28.d4.fr16__9f2a1c07__r3",
  "arm_key": "S4__C28.d4.fr16", "build_id": "9f2a1c07", "cell_id": "K.T4096.bal",
  "harness": { "name": "mok_synthetic_prefill", "driver": "screen.sh",
               "arms_in_campaign": ["production","pf6gm_mega","mps_mega"],
               "warmup": 500, "timed": 100, "soak_epochs": 600, "rotations": 5 },
  "n": 5, "unit_of_analysis": "run",              // LAW-61(a): cluster by run
  "identity_checks": {
    "text_sha_matches_build": true,
    "descriptor_echo_matches_request": true,       // K0_MPS_DESC_DUMP, §5.3
    "mechanism_invariant_pass": true               // LAW-59
  },
  "gates": { "pperr": 0, "spin_fail_max": 0, "mok_rel_err": 0.0,
             "occupancy_printed": 1, "all_ranks_checked": true },
  "result": {
    "primary": {"metric": "boundary_p50_us", "value": 5822.0},
    "ratio_vs": {"production": 0.7544, "pf6gm_mega": 0.8427},   // 5822.0/6908.8, same-campaign

    "phase_stamps_us": {"m6": null, "m7": null, "combine": null},
    "outcome": "ok"                                // ok | hang | gate_fail | void
  },
  "artifacts": ["distributed-kernels/tp8_mega/results/…"],
  "notes": "",
  "voided_by": null                                // e.g. "D1: MODE14=1 binary"
}
```

For an e2e serving run, `result` carries the **whole metric vector**, never a scalar — the reporting
discipline (`../BRIEF.md:109`, and LAW-44's measured demonstration that **median TTFT fell 5.0%
while node throughput fell 7.84% in the same pair**):

```jsonc
"result": { "input_tok_s": 42835.7,
            "ttft_ms": {"p50": 43145.0, "p99": 47895.0},
            "tpot_ms": {"p50": 756.8, "p99": null},
            "e2e_ms": {"p50": 48452.0, "p99": null},
            "seal": {"sealed": 299, "in_bucket": 299, "ratio": 1.000},
            "fill_phi": 0.985, "uniform_rescued": 6, "steps": 300 }
```

`seal` and `fill_phi` are **required fields**, not optional: no serving delta may be interpreted
without its arm's `sealed/in_bucket` (LAW-49), and every quoted cell must carry its own measured
fill (`FILL_AWARE_DESIGN.md:1319-1323`).

### 4.7 Schema — `cells.jsonl` (the W half of the row)

```jsonc
{ "cell_id": "E.c512p.dp", "class": "SERVING",
  "topology": "DP8/EP8", "concurrency_offered": 512, "max_num_seqs": 128,
  "ISL": 4096, "OSL": 8, "num_prompts": 2048, "arrival": "closed",
  "prefix_cache": false, "max_num_batched_tokens": 4096, "kv_cache_dtype": "fp8",
  "prompt_source": "qsl mlperf-qsl-concat-v2-stride7", "seed": 320802,
  "prompt_stream_sha256": "…",
  "measured_fill_phi": 0.985, "measured_dummy_rate": 0.02,
  "skew_instrument": null, "eplb": "off", "placement": "contiguous",
  "hardware": "8xMI355X gfx950", "variance_envelope": {"tput": 0.018, "latency_p50": 0.0025},
  "known_confounds": ["H14: offered 512 vs max_num_seqs 128 — VERIFY before quoting as conc 512"] }
```

The `known_confounds` field is not decoration: `WORKLOAD_EVIDENCE.md` H14 flags that the c512p cell
offers 512 in flight against a **128-seq admission cap**, never characterized — *"verify before
quoting c512p as concurrency 512."*

### 4.8 The reproducibility contract

A `runs.jsonl` row is **valid** iff all four hold:

1. `replay/<arm_id>.sh` regenerates the invocation from the row alone (generated, not hand-edited);
2. `identity_checks.text_sha_matches_build` — the binary is the one the row names (LAW-58);
3. `identity_checks.descriptor_echo_matches_request` — the **device's own** decoded config equals
   the requested one. Precedent tooling exists:
   `distributed-kernels/fused_moe/overnight/aug11/tools/e35_11_descverify.sh` dumps the descriptor
   the host handed the kernel (`K0_MPS_DESC_DUMP`) and decodes slot `K0P6_D_MPS_CFG` with the exact
   bit formula from `moe_mps_adapter.cuh`;
4. `identity_checks.mechanism_invariant_pass` — the ISA gate of `credit.cuh:30-38` for any depth
   arm, and LAW-59's rule for any "inert" claim.

Rows failing any check are kept with `voided_by` set. **Voided rows are never deleted** — the
retraction ledger (`WORKLOAD_EVIDENCE.md` §0.2, BANKED_LAWS §7) exists because deleted-but-cited
numbers keep reappearing.

---

## 5. PART V — BUILDABLE-VARIANT COUNT AND BUILD STRATEGY

### 5.1 Screening design (not a full cross)

Three design rules, each forced by a banked law:

* **R1 — OFAT around the current best, with the control re-measured in the same session.** LAW-60:
  screens order points, campaigns set the ratchet, stamps attribute mechanisms; the smallest callable
  end-to-end screen delta is **~3%**.
* **R2 — joint cells where separability is already falsified.** LAW-21 proves **(carrier, depth)** is
  non-separable (the throttle alone is **1.52× the entire (a)→(c) gap**). LAW-30 proves **M7 and
  combine are one coupled block** at **r = −0.904** — judge their *sum*. LAW-37 flags **(injection
  bound, coarse readiness)** as possible **substitutes**, never measured on top of each other. Each
  of these gets a 2×2 cell, not two OFAT ladders.
* **R3 — `C` (and `flush_rows` with it) is re-swept inside every W-cell**, because its optimum
  already flipped across designs and the kernel documents it as de-tuned at other `T_eff`.

### 5.2 The count

**Skeletons actually built: 7** (target O(5–8)).

| skeleton | build? | why |
|---|---|---|
| S0, S1, S2, S3, S4, S6, D1 | **yes — 7** | reference, control, the falsified-but-live control, the ratchet, the shipped best, the TOPO arm, the PHASE arm |
| S5 | no — 1 archival point | falsified (+75.8 µs) |
| S7 | no | design only; contributes a law, not an arm |
| S8 | **conditional** | P0 scratch fix first; it is the only route to F6 |

**Knob points per skeleton per cell, after pruning** (target O(30–80)):

| skeleton | runtime points | compile variants | knob points/cell | derivation |
|---|---:|---:|---:|---|
| S0 | 1 | 0 | **1** | no knobs of ours |
| S1 | 1 | 1 (grid control) | **2** | `K0P6GM_G` is `#error`-pinned |
| S2 | C{0,8,16,32,48,64}=6 × mode{2,3}=2, minus A3 (mode 3 legal only at C∈{32,64}) | 0 | **8** | the C-response curve that answers "all sizes?" |
| S3 | C{0,4,8,16}=4 + depth{off,4,8,16,32}=5 + joint(carrier×depth)=4 | 0 | **13** | LAW-21 joint cell included |
| **S4** | C{0,8,16,24,28,32}=6 + depth{off,4,8,16,32}=5 + flush{1,8,16,32}=4 + mode{12,13,2}=3 + joint(C×flush)=4 + joint(depth×readiness)=4 | order{4} + slab-S{1, pinned} + M24{FILL} = 5 | **31** | §5.1 R1–R3 |
| S6 | transport{2} × slab{128,256,512}=3 × depth{4,8,16,32}=4 × bps{2,4}=2 × k_inner{2048, 8192, 16384}=3, screened not crossed | 0 (rig switches at `:430-439`) | **28** | k_inner is the §1.8 axis |
| D1 | transport{2} × slab{16,32,64}=3 × depth{4,8}=2, + RCCL reference{2} | 0 | **14** | gated on M5 |

**W-cells (from the sibling W doc; here only their count matters for the budget):** the kernel/rig
side needs ~4 (`K.T4096.bal`, `K.T4096.agg`, `B.tp8.16k`, `B.tp8.256`) and the e2e side ~3
(`E.c32p.*`, `E.c512p.dp`, plus one point inside the **32→512 gap**, which
`WORKLOAD_EVIDENCE.md` H3 calls *"the single most quotable number the campaign could produce, and we
cannot currently state it"*).

**Total buildable arms:**

```
rig/boundary arms  ≈ Σ (knob points × applicable cells)
                   ≈ 1·2 + 2·2 + 8·2 + 13·2 + 31·2 + 28·2 + 14·1     ≈  180 arms
e2e validation     ≈ 3 cells × 2 arms × 5 order-balanced pairs        ≈   30 pairs
```

≈ **180 rig arms + 30 serving pairs**, from **7 skeletons** and **≈12–16 HSACOs** (§5.4).

### 5.3 Compile-time vs runtime — and the env-forwarding trap

**The design rule the codegen forces** (`KNOB_INVENTORY.md` B.4, `credit.cuh:12-22`,
`order.cuh:15-18`):

> **Policy CHOICE stays runtime; policy CODEGEN stays compile-time.**

The mechanism is already shipped twice: depth is four `s_waitcnt vmcnt(N)` specializations selected
by mutually-exclusive SGPR lane masks (`n2_phase2_gm_mps.cpp:116-176`), and the CDAR rig switches
over compile-time `Depth` specializations at runtime (`m25_boundary_bench.hip:430-439`). Templating
the hot loop on `Depth` instead spilled one 4-byte value with **389 reloads**.

**The trap that voids runtime knobs (Class D9).** A runtime knob is only runtime *if the harness
forwards it*. `BENCHMARKING.md:87-101` records the measured instance: `e004pf_k0pf_ab.py:241` reads
`K0_PF6GM_G` with a default of `"2"`, and **`K0_PF6GM_G` is not forwarded by the frozen
`run_campaign.sh`**, so the container build is governed by the source default. Running the frozen
campaign unmodified compares a G=3 candidate against a **G=2 reference** — *"a ~330 µs handicap in
the candidate's favour, and an invalid claim."* The same document names the sanity check:
`pf6gm_mega/production_p50 ≈ 0.898` if G=3 was forwarded, **≈ 0.941 if it was not** — *"that single
number tells you which reference you actually built."*

**Three mitigations, all mandatory in the manifest:**

1. every runtime knob is carried on the **whitelisted** path that is already forwarded
   (`K0_MPS_CFG` — the config-word grammar, `BENCHMARKING.md:174-186`), never as a new ad-hoc env
   var;
2. every run records the **device-side echo** of the decoded descriptor, not the requested string
   (`K0_MPS_DESC_DUMP` + `e35_11_descverify.sh`);
3. every campaign carries the **reference sanity ratio** in the row, so a mis-forwarded build is
   detected by a number, not by inspection.

**The sweep vehicle already exists.** `overnight/aug11/tools/screen.sh` is a manifest-driven driver:
`bash screen.sh TAG /path/to/cfgs.txt # one K0_MPS_CFG per line`, with `SCREEN_ARMS`,
`SCREEN_WARMUP` etc. as env knobs and a failure-classifier that recognizes `"K0_MPS_CFG requires"` →
`bad-mps-cfg`. `arms.jsonl → cfgs.txt` is a projection, not new machinery.

### 5.4 The HSACO matrix (≈12–16 binaries)

| # | binary | skeleton(s) served | notes |
|---|---|---|---|
| 1 | `k0pf6gm_device_tile.hip` | S1 | reference build; G=3 forwarded and asserted |
| 2 | `k0pf6gm_device_tile_mps.hip` | S2 (modes 1–9) | carrier-pool control |
| 3 | m15 **base** | S3 + S4 | one binary if the validator's mode ceiling admits both under the same gate set (`moe_mps_adapter.cuh:652-658`) — **VERIFY on the node; do not assume** |
| 4–6 | m15 × `ORDER ∈ {producer_major, owner_major_staggered, source_interleaved}` | S4 | A12: order is compile-time |
| 7 | m15 × `SCATTER_RR=1` | S4/m17 | SKEW arm |
| 8 | m15 × `REPLICATE=1` | S4/M18 | **slot-63 exclusive** (A6) |
| 9 | m15 × `M24_FILL=1, ZERO_PAD=1` | S4/M24 | FILL arm; **has never compiled** — first job is to make it build |
| 10 | m15 × `M20_SLOTPOOL=1` | S4/M20 | placement arm |
| 11 | m15 × `STAGED=1` | S8 | **gated on the P0 scratch fix** (D3) |
| 12–13 | instrument builds: `MODE14=1`, `TBO=1` | diagnostics only | **no headline number may come from these** (D1) |
| 14 | `m25_boundary_bench.hip` | S6 + D1 | transport/mode/depth specializations selected at runtime |
| 15 | `m25_boundary_bench.hip` + RCCL reference arm | S0 (TP) + D1 go/no-go | `-lrccl` already linked; M5 is one argv |
| 16 | *(reserve)* T3 vehicle | S7 | only if the design arm is funded |

**Per-binary hygiene, non-negotiable** (LAW-58/59): record `.text` sha256; a flag-OFF build claimed
inert **must** be `.text`-identical to base; a flag-ON build **must** differ; record scratch **op
counts** and spill counts, not just `ScratchSize` (which stayed pinned at 128 B/lane while scratch
ops went **19 → 168**).

### 5.5 Node-hour budget (order of magnitude, from measured cadence)

| track | unit cost | count | node-hours |
|---|---|---|---|
| MoK 5-rotation campaigns (S1–S4, D1 kernel-side) | **~3 min 5 s** (LAW-60: 500 warmup / 100 timed / 600-epoch soak) | ~120 | **≈ 6.2** |
| CDAR boundary rig points (S6, 7 iters, median wall) | minutes; the existing depth×bps and slab sweeps are single result files | ~60 | **≈ 3–5** |
| Route-replay campaigns (SKEW cells) | **139 s per campaign** (`WORKLOAD_EVIDENCE.md` H8) | ~20 | **≈ 0.8** |
| e2e serving pairs | **~25 min per pair** (same source), plus `ARM_COOLDOWN=240 s` symmetric | 30 pairs | **≈ 12.5** |
| instrumentation runs (device phase ledger M1, M2 ratio sweep, M6 profile) | — | ~10 | **≈ 2** |
| **total** | | | **≈ 25–27 node-hours** |

Against `../BRIEF.md`'s single-node constraint this is roughly **two to three overnight sessions**,
which is what makes the pruning of §3 load-bearing rather than tidy.

### 5.6 Ordering (what to build first, and why)

1. **M2 — the CDAR `k_inner` ratio sweep.** One argv. It decides whether four falsified hypotheses
   were tested at an operating point where hiding was possible at all (`TP8_STATE.md` §2.4). The
   cheapest experiment in the entire campaign with the largest interpretive reach.
2. **M1 — the device phase ledger.** *"One instrumented run replaces the next four guesses."*
   With the `[MPS TS]` reset discipline baked in (LAW-63c) or the ledger lies.
3. **The `C`-response curve in a second W-cell.** The direct answer to the experts' question, and
   the arm that turns a tuning constant into a law.
4. **Make `K0P6_M24_FILL` compile.** It is the only actuator for the FILL axis, and FILL is the
   mediating quantity behind the flagship result — currently supported by *two cells that differ in
   ~six variables at once* (`WORKLOAD_EVIDENCE.md` H7).
5. **M5 — RCCL at decode message sizes.** The go/no-go for D1 and for every cell below ~2K tokens.

---

## 6. What is deliberately NOT in `S`

| excluded | why | where it lives |
|---|---|---|
| fill φ, dummy rate, concurrency, ISL/OSL, skew, arrival process, topology choice | these are **workload**, not schedule — conflating them is how a scheduler effect (κ = 1.192, worth 15–25% of node throughput) gets read as a kernel effect (LAW-40, LAW-50) | `W_VECTOR` (sibling) |
| `C*`, `d*`, `flush_rows*` | **outputs** of `M(W, S; θ)`, not inputs (§0.1 tier 3) | the cost model |
| instrument gates, src rev, kernel-name pin, timestamps bit | they change the *measurement*, not the schedule — but they are part of every number's identity | `I`, §2.13 + `builds.jsonl` |
| `max_num_batched_tokens`, bucket list, scheduling policy, prefix caching, EPLB config | serving-stack **control plane**; they set what the kernel sees but are not kernel schedule | `WORKLOAD_EVIDENCE.md` §3.1; H12 is the un-pulled lever |
| SDMA as a carrier | **never measured in situ**; the ledger neither supports nor refutes it (retraction list item 2), and the one API measured was CU-lowered, not SDMA (LAW-6) | open hole |

---

## 7. The five sentences this document is for

1. **Structure is not a knob:** carrying (+340 µs), idling (+8–9 µs/CTA), and consuming (−444 µs)
   are the *same* reservation mechanism with opposite signs, so "does producer/consumer work best
   for all sizes?" is a question about which of the three jobs the pool has, in which regime — and
   the answer already flipped once across carriers (C3).
2. **The runtime schedule surface is one 64-bit word**, so hundreds of arms are one HSACO plus a
   manifest — provided every arm pins its instrument gates, because unreachable code cost the
   ratchet **+726.9 µs** with the mode's source untouched.
3. **Depth is the campaign's cleanest law and its cleanest scoping failure:** **−615 µs**
   co-resident, **flat** in pure transport. The mediating quantity is co-residency, not bytes.
4. **Granularity has two contradictory verdicts** (free in M15, **2.1×** in CDAR) and reconciling
   them — signal count vs transfer geometry — is itself a cost-model contribution.
5. **The fusion boundary is a schedule knob** because it decides whether certificates can exist at
   all and how many times per launch cross-rank drift is realized; and at the TP8 boundary its
   honest current verdict is negative (3,606 vs 2,909 µs) at an operating point **~5× away from the
   deployment ratio**, which is the first thing to fix.

---

## Amendments from review (2026-08-18)

Superseded on organisation and budget by `../ABLATION_METHODOLOGY.md`:

1. **`S` is re-presented as one BASE kernel plus a module library**, not as disjoint skeletons
   (operator north star). S1 is the base; the depth bound, slab certificates, consumer-major order,
   consuming pool, transport op class, shared-expert filler, fill-awareness, prefetch, placement and
   seal are **modules with attach/detach conditions**. Mapping: `ABLATION_METHODOLOGY §3.2`.
2. **S6 (CDAR) and S7 (T3) are declared NOT expressible as modules on this base** — they replace it.
   That composability boundary is a claim the campaign must confirm or refute, and cell `B.res`
   (`N_CTA × {comm in-kernel, comm as a second concurrent kernel}`) sweeps the grid geometry that
   §2.10 K10 had exempted from the LAW-62 settability rule as "the axiom" (MECHANISM B8).
3. **§5.3's env-forwarding trap is promoted to a binding §0.4-class contract** for the whole
   campaign: every knob rides the whitelisted `K0_MPS_CFG` path with device-side descriptor echo and
   the reference sanity ratio in every row. Per-rank `C` must ride that path, not a new env var
   (FEASIBILITY B8).
4. **§5.5's ≈25–27 node-hour budget is superseded.** Repriced as `pts × K × unit` with a declared
   boundary-invocation range of [1, 3] min and a 0.6–0.7 node-availability derate:
   **≈38–48 node-hours over 6–8 overnight sessions** (`ABLATION_METHODOLOGY §7.3`).
5. **Manifest schema additions (binding):** `arms.jsonl` gains `primitive_touched` and
   `api_implication`; `runs.jsonl` gains `envelope_measured_this_cell`, the actuation receipt, and
   the per-run clock/temperature summary.

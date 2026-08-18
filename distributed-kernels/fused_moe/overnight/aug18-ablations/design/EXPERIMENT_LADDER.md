# EXPERIMENT_LADDER — the executable campaign plan

**Deliverable:** output (5) of `../BRIEF.md:122-123` ("calibration microbenches → boundary grid
sweeps → e2e validation; gates, sample sizes, kill criteria, node-hour budget") and output (6)
(`:124-125`, per-cell kernel predictions). **Written:** 2026-08-18, laptop, no node access.

**Consumes, and is subordinate to, all five siblings.** Where this document and a sibling
disagree and the disagreement is not named here, **the sibling wins**:

| sibling | what this ladder takes from it |
|---|---|
| `W_VECTOR.md` | the 10-coordinate mediating vector **Q**, the settability audit (§4.2), the Tier-0/1/2 grid skeleton, and the three matched-Q falsifiers FQ-1/2/3 |
| `S_VECTOR.md` | the 7 buildable skeletons (S0–S4, S6, D1), the four-class constraint matrix, the manifest schema, and the naming scheme `<cell>__<skeleton>__<delta>__<build8>` |
| `COST_MODEL.md` | the θ table (with its 8 holes), decision boundaries D1–D11, predictions P1–P11, the fit/hold-out split, and the accuracy bars |
| `METRICS_PROTOCOL.md` | the metric map (classes A–E), the ten-block workload tuple, the two-instrument doctrine, the validity gates, and the law-card format |
| `OVERLAP_ATLAS.md` | the eight feasibility screens, the 17-row slack ledger, the master EV ranking, and the "do this first" ordering |

**Binding rules for this document.**
1. Every number carries a source (repo `path:line`, git sha, BRIEF digest, or a sibling
   section). Arithmetic performed *here* is tagged **DERIVED** with its inputs shown.
2. Every cell states its **actuator** and whether the actuator exists today. LAW-62 is
   binding: *prove a knob is settable before pre-registering an axis.* Cells whose actuator
   does not exist are marked **GATED** and carry the engineering item that unblocks them.
3. Every arm is a manifest row (`S_VECTOR §4`), regenerable from `replay/<arm_id>.sh`.
4. **Predictions are registered before runs** (§2). A cell that runs before its prediction is
   committed produces a number, not evidence.
5. Anything not measured is **SPECULATION** or **PREDICTION** (a prediction names the banked
   law it extrapolates from and its falsifier).

---

## 0. The ladder in one page

### 0.1 The gate structure

```
PHASE 0  CALIBRATION            exit: θ table complete, every entry with a CI
   │     (fills the 8 holes in COST_MODEL §2; extends two existing harnesses)
   ▼
PHASE 1  REGISTRATION           exit: predictions.jsonl committed with a git sha, BEFORE
   │     (instantiate D1–D11 with calibrated θ)      any Phase-2 cell runs
   ▼
PHASE 2  BOUNDARY GRID          exit: every registered prediction adjudicated at the
   │     (the precision instrument)                  precision instrument
   ▼
PHASE 3  FIT + HOLD-OUT         exit: accuracy bars met, OR the pre-committed refinement
   │     (θ frozen; M predicts held-out cells)       ladder executed and the miss published
   ▼
PHASE 4  E2E VALIDATION         exit: ≥4 W-points confirmed under the serving protocol with
         (the validity instrument)                   coverage counters and the metric map
```

**The phases are gates, not a schedule.** Phase 2 may not start on a cell whose prediction is
unregistered. Phase 4 may not start on an arm that failed its Phase-2 kill criterion.

### 0.2 Budget

| phase | node-hours | wall (1 node) | notes |
|---|---:|---|---|
| **Phase 0** | **≈6.0** | ~1 overnight | includes 0-GPU CPU work (§1.1) done in parallel |
| **Phase 1** | 0 | ~1 day | CPU only; it is a commit, not a run |
| **Phase 2** | **≈14.0** | ~1.5 overnights | 17 cells, ~250 rig invocations + ~164 MoK campaigns |
| **Phase 3** | ≈2.0 (contingency) | ~1 day | CPU fit; node hours only on a bar miss |
| **Phase 4a** | **≈16.5** | ~2 overnights | the 4 mandated e2e cells |
| **Phase 4b** | ≈10.0 (conditional) | ~1 overnight | concurrency ladder + placement, fires only if Phase 3 passes |
| **total (4a)** | **≈38.5** | **≈4 overnight sessions** | |
| **total (4a+4b)** | ≈48.5 | ≈5 sessions | |

**Honest delta against `S_VECTOR §5.5`, which budgeted ≈25–27 node-hours.** Two reasons, both
stated rather than smoothed: (i) that estimate allotted **2 node-hours to "instrumentation
runs"**, whereas the θ table has **8 holes** (`COST_MODEL §2`, class `H`) and closing them
properly is ≈6 h; (ii) it budgeted 30 serving pairs at ≈12.5 h but did **not** include the
decode-heavy cell (long OSL runs are ~2× a prefill pair) or the mixed/open-loop cell. The
per-unit cadences are identical (LAW-60: 5-rotation MoK campaign = **3 min 5 s**; ~25 min per
serving pair + `ARM_COOLDOWN=240 s`).

### 0.3 The critical path is not node-hours — it is five unbuilt actuators

| # | blocker | what it gates | size | source |
|---|---|---|---|---|
| **X1** | `K0P6_M24_FILL` has **never compiled** | the entire **φ** axis — the mediating quantity behind the flagship result | **M** | `WORKLOAD_EVIDENCE §3.2`; `W_VECTOR §4.2` |
| **X2** | serving M15 body **fails the MoK gate at T=2048/1024** (rel 0.874205 / 0.838890) while `production` passes in the same runs | the entire **T** axis on the shipped body; the T-sweep otherwise runs on the tgen body (sha `20b8c6bb`), not M15 (`935f555e`) | **M** (port the tgen M2 hole-sentinel fix) | `WORKLOAD_EVIDENCE §W6`; `REPORT.md:88` |
| **X3** | `TOPK != 8` is a hard reject at `k0pf6gm_device_tile_m15.hip:1230` | the **top-k** sign-flip law (P4a), the cleanest "sign flips on a workload parameter" result the campaign owns | **M** | verified in-repo |
| **X4** | `m25_boundary_bench.hip:282` hardcodes `cd::transport::store_towers` | K1 **co-resident** — the decode K1 reversal cannot be tested with compute present | **S** (one template arg + argv) | verified in-repo |
| **X5** | S8 (`K0P6_M15_STAGED=1`) compiles to **96 `scratch_load_dword` + 97 `vmcnt(0)`** within 24 instructions of the remote atomic vs 0/1 in the default build | fp8-on-wire (F6) entirely — gfx950 has **no fp8 remote RMW**, so F6 is reachable only atop S8 | **M–L** | `FP8_WIRE_DESIGN.md:24-29` |

Plus one access gate: **gfx942 node** for the ARCH cell (`W_VECTOR §4.2`: "microbench only").

**Consequence for sequencing:** X1 and X2 must be started on day 1 in parallel with Phase 0,
because they are the two axes on which the campaign's flagship claims rest and they are the
only two whose lead time exceeds their run time.

---

## 1. PHASE 0 — CALIBRATION

**Objective.** Close every `H` (hole) entry in `COST_MODEL §2` and every Tier-0 item in
`W_VECTOR §4.3`, using **extensions of the two harnesses that already exist**, never a rewrite.

**Exit gate (all four must hold):**

* **G0-a — θ complete.** Every row of `COST_MODEL §2` is class `M` or `D`; no row is class `H`
  except those explicitly scoped out (§1.6).
* **G0-b — every θ entry carries a confidence interval** computed by the method in §1.6, not a
  point value. LAW-62(a) is the reason: exp_22's H1 demanded 75% of the *spec-sheet* 76.8 GB/s
  (57.6) when the **achievable** ceiling is 56.9–57.3 GB/s, so `REFUTED_NEVER_REACHED` fired on
  a threshold no CTA count could have passed — *"a spec-sheet denominator can manufacture a
  refutation."*
* **G0-c — the two named fudge factors are bounded by measurement, not by assumption.**
  `I₀ ≤ 1,159 µs` (64% of its mechanism unidentified) and `κ_couple ≤ 55 µs/CTA` must each be
  either *measured* by the phase ledger or *declared per-skeleton with the scope stated*
  (`COST_MODEL §1.6, §1.7`).
* **G0-d — instrument parity.** Every Phase-0 binary carries `.text` sha256, scratch **op
  count** and spill counts (not just `ScratchSize`, which stayed pinned at 128 B/lane while
  scratch ops went **19 → 168**), and the flag-ON build **differs** from flag-OFF (LAW-58/59).

### 1.1 P0-Z — the free triple (zero GPU time, ~4 engineer-hours, run first)

`OVERLAP_ATLAS §13.1` and `W_VECTOR §4.3` both put these first, and they cost nothing.

| id | measurement | what it decides | source |
|---|---|---|---|
| **P0-Z1** | `np.unique(topk_ids, axis=0)` on an existing `route_capture/skewhook_v2` capture | confirms or kills LAW-42's **degeneracy mechanism** (vLLM pads with token id 0 ⇒ identical hidden ⇒ identical top-8 ⇒ 2.66× rank skew from padding alone). Decides atlas item H1 (pad-row dedup). Already action #1 at `BOTTLENECK_SIGNALS.md:601-606` | LAW-42 |
| **P0-Z2** | reconcile the two per-call skew instruments — m18diag **p50 5.15× / p95 6.25×** (n=90,050 router calls) vs routecap1 **p50 2.61× / p90 3.35×** (n=512 rank-calls) | **changes the modeled skew tax by ~2×.** Until resolved, M carries σ as the interval `[2.61, 5.15]` and every σ-dependent prediction is a band, not a number | H17; `COST_MODEL §1.8` |
| **P0-Z3** | compile one small-M decode tile body, print `-Rpass-analysis=kernel-resource-usage` | decides whether **SCREEN-1's four-door rule survives into decode.** LAW-12's own scope note: shrinking a ubench's LDS let occupancy rise 1 → 2 blocks/CU and silently voided that experiment. If decode tiles run at occupancy ≥2, **door (e) re-opens and every prefill-rejected "fill the stall" design becomes legal** | `OVERLAP_ATLAS §1` corollary; `aug11/LESSONS.md:99-109` |

**P0-Z3 is the single largest structural fork in the campaign and it costs a compiler flag.**

### 1.2 P0-A — transport-only harness (`m25_cdar_bench.hip`), co = 0

**Why this harness and not a new one.** Its argv **already is** the manifest schema for the
transport skeleton: `tokens, slab_rows, transport{atomic|towers}, iters, depth, bps`
(`m25_cdar_bench.hip:265-272`). It is the only rig in the project where `co = 0`, which is the
coordinate that separates LAW-8 from LAW-20.

| θ id | parameter | how P0-A fixes it | points | prediction registered in Phase 1 |
|---|---|---|---:|---|
| **θ-F4** | node egress ceiling (currently *"unreproduced; treat as unknown, measure first"*) | all-pairs egress sweep at `tokens` ∈ {4K, 16K}, `bps` ∈ {2,4}, CTAs implied by `n_slabs × bps` | 12 | the 349–355 GB/s card is **not** reproduced; the achievable ceiling lands in [280, 330] GB/s. **This one number re-scales every "% of ceiling" framing in `TP8_STATE §2.3`** |
| **θ-F5/F6/F7** | atomic-vs-store efficiency **as a function of message size** | `b` ∈ {3.67, 15, 33, 60, 235} MB × transport {atomic, towers} × slab_rows matched | 30 | **b\* = 32.8 MB ≈ 2,289 tokens**, band [1,500, 4,000] (D3/P3) |
| **θ-F13** | **RCCL at decode message sizes — never run; the entire go/no-go for decode** | `m25_boundary_bench mode=rccl` at `tokens` ∈ {32, 128, 256, 1024, 4096, 16384} | 6 | see §2.3 row B2-d |
| **θ-P8** | `d*`'s left side — **depth 2 has never been measured on any arch** | add one `vmcnt(2)` specialization; sweep depth {2,4,8,16,32} at co=0 | 10 | depth stays **flat** at co=0 (LAW-8 replication); if depth 2 is *not* flat at co=0, LAW-8 is wrong and the whole `d(co)` structure of M collapses |
| **θ-F1/F2/F3** | per-link BW, knees, 7-link plateau — **re-measure, do not inherit** | CTA-count sweep at fixed message | 8 | knee at 8 CTAs single-link, 32 rr7; C=8→64 buys +2.4% |

**Sizing.** ~66 primary points; ~30 of them need 3 invocations (predicted delta <5%, §3.2)
⇒ ≈126 invocations × ~1 min (8-GPU single-process init dominates) ≈ **1.6 node-hours** with
retries.

**Two things P0-A explicitly does NOT do.** (i) It cannot measure `θ-F15` (SDMA in situ) —
that has **never been measured** and the ledger neither supports nor refutes it; the one API
that *was* measured (`hipMemcpyPeerAsync`) is **CU-lowered, not SDMA** (LAW-6), with zero
`hsa_amd_memory_async_copy_on_engine` calls in the trace. SDMA is **scoped out** (§1.6).
(ii) It cannot see live xGMI throughput — `amd-smi --xgmi` returns N/A and `--shownodesbw`
returns 0-0 (LAW-11), so every bandwidth figure is a computed rate, and duty cycle must be
reconstructed from per-CTA commit timestamps (§1.3).

### 1.3 P0-B — boundary harness (`m25_boundary_bench.hip`), co = 1

Four arms already exist (`compute | phased | fused | rccl`, `:494-618`), the rig runs the real
`v_mfma_f32_16x16x32_bf16` body at the real 14,336 B row, and `-lrccl` is already linked. Three
extensions, all small:

| ext | change | size | why it is blocking |
|---|---|---|---|
| **E-B1** | **device phase ledger**: per-block-class stamps for produce span, first/last commit, first/last certify, first/last pull, burst-2 span — **with the running-maxima reset baked in** | **M** | *"one instrumented run replaces the next four guesses"* (`G25_1_STATUS.md`). LAW-63(c): `[MPS TS]` values are running maxima **never reset**, so after a soak they describe the *soak* epoch — bake the reset in **or the ledger lies** |
| **E-B2** | **fabric-duty histogram**: bin per-CTA commit timestamps into the layer timeline, emit duty-over-time, not just spans | **S** (rides E-B1) | there is no live xGMI counter (LAW-11), and §2.1 of the atlas argues the binding quantity under TP8 is **duty cycle**, not span. This is what turns C6 from an argument into an experiment |
| **E-B3** | **transport as a template arg + argv** (currently `store_towers` is hardcoded at `:282`) | **S** | X4. Without it the **decode K1 reversal** (towers *lose* at 3.67 MB) cannot be tested with compute co-resident, and D3/P3 is only half-measurable |

**θ closed by P0-B:**

| θ id | parameter | method | CI method |
|---|---|---|---|
| **θ-P5** | `I₀`, the unattributed co-residency floor (`≤1,159 µs`, 64% of mechanism unidentified) | phase ledger separates produce-span from commit-span; the residue after attribution **is** `I₀` | 3 repeats, report the interval |
| **θ-P6** | `κ_couple` (`≤55 µs/CTA`) | ledger separates M7's **drain wait** from its **MFMA span** — the same instrument LAW-30 itself calls for | 3 repeats |
| **θ-P12** | co-residency protocol tax on the *phased* arm (~335 µs, DERIVED, **attributed by nobody**) | ledger, phased arm, `compute` arm as the control | 3 repeats |
| **ρ (Q3)** | compute:transport ratio as a **swept** axis | `k_inner` argv (`:447`) until ρ ∈ {0.65, 1.0, 2.0, 3.0, 4.0} | direct |

**The blocking arithmetic that makes P0-B mandatory, restated (DERIVED, `W_VECTOR §2.2 Q3`).**
At the rig's shipped operating point `k_inner=2048`, compute floor = **1,439 µs** (two bursts)
against a collective of **2,167–2,214 µs** ⇒ `ρ = 0.650–0.664`. Max hidable fraction is
`min(ρ, 1)` ⇒ the **arithmetic ceiling is 65%** against a G25-1 PASS bar of **≥90%**
(`M25_TP8_MEGA_DESIGN.md:181`). **The gate could not have passed at any schedule, for any knob
setting.** Every G25-1 verdict — including the four falsifications F1–F4 — was scored against
an impossible criterion. ⇒ **the bar is restated for the whole campaign as `h ≥ h*`** where
`h*` is D2's break-even (48.7% at today's θ), **not** as a fixed 90%.

**Sizing.** ρ-sweep 5 × 4 arms = 20 points; ledger 12; transport-argv validation 8 ⇒ 40 points
× ~1 min, plus 3 repeats on the 20 ρ points ⇒ ≈80 invocations ≈ **1.3 node-hours**.

### 1.4 P0-C — mega chassis (MoK synthetic prefill campaigns), DP/EP

| θ id | parameter | method | why re-measure |
|---|---|---|---|
| **θ-C3** | `ρ_mfma` for the M6/M7 bodies at the MoE tile shape | isolated-body timing at 2–3 shapes | **C_phase cannot be a FLOP model.** Calibrating against the exp_33 stamp gives `θ_flop` = **784 TFLOP/s for M6** and **356 for M7** at the *same* shape on the *same* silicon — a **2.2× spread**, because M6 is weight-streaming/latency-bound and M7 is CTA-throughput-bound (`W_VECTOR §3.3` finding 4). A FLOP-priced model mis-places the reservation boundary — LAW-15 calls that *"the single most repeatable placement mistake in the ledger"* |
| **θ-C4** | `φ_p` (capacity elasticity per phase) | reserve-but-idle C-sweep C ∈ {0,8,16,24,32,48,64}, phase stamps | LAW-65: **re-profile after every landed win.** The banked φ_p (plan+M6 ≈0, M7 = 0.84) was fitted at the *pre-M15* profile; at the M15 ratchet **M7 (46.2%) has already overtaken M6 (41.9%)** |
| **θ-P4** | `c_bar`, grid-barrier cost — **never measured**, though the structure is known (four barriers between M2 and M6, and the pre-M6 region is a "sum of maxima") | empty-phase barrier-count sweep in the chassis | fusion boundary → barrier count → **drift-realization count** (three per launch vs one per layer) |
| **θ-C6** | consumer rate `ρ = 6.1 GB/s/CTA` | re-measure on the M15 pool body | `C_sat` scales **inversely** with it, and it was measured on the combine body only |
| **θ-P3** | `ψ`, line-contention exponent — currently a **lower bound only** (`≥1`) | counter footprint sweep L ∈ {2,4,8,16,32} | LAW-18 inverts the coalescing intuition: 32 lanes onto **2 lines instead of 32** made the kernel **≥10× slower** (abandoned after 12 min against a usual 25–95 s). ⇒ `L` should be **maximised**, the opposite of the load rule |

**Sizing.** 14 + 6 + 6 + 4 + 5 = 35 campaigns × 3 min 5 s ≈ **1.8 node-hours**.

### 1.5 P0-D — serving-side calibration

| id | measurement | fixes | cost |
|---|---|---|---|
| **P0-D1** | **profiler confirmation of the 2.8 ms/layer exposed AR on the native TP8 arm** | `ρ` at deployment. Until it runs, *"production exposes 28% of its step as all-reduce"* is a **model, not a measurement** — it is DERIVED from throughput-vs-roofline plus the in-rig RCCL arm, and `TP8_OVERLAP_ANALYSIS.md:141-143` lists it as the first thing B0 must calibrate; **it was not done** | ~0.6 h |
| **P0-D2** | the **16,384-token chunk assumption** underneath the 634 ms/step arithmetic | ASSUMED today (AMD recommendation); it settles §1.4's whole decomposition | rides P0-D1 |
| **P0-D3** | **per-cell variance envelope** at c32p and c512p, ≥2 repeats each | R-N1: `envelope_source ∈ {measured-this-cell, inherited-c32p, unmeasured}`. **Inheriting the c32p band at c512p is over-conservative; inheriting the c512p band anywhere else is fraud** | ~1.3 h |

**Sizing:** ≈**1.9 node-hours**. Note P0-D2 is free if P0-D1 runs; note also that the c512p
envelope is already measurable from the two banked pairs (m15 M1 spread **+0.11%**,
native_tuned_dp **+1.80%**, latency percentiles ≤0.25%) — P0-D3 raises it from n=2 to n=4.

### 1.6 The θ table after Phase 0, and the CI method

**CI method (this is what G0-b requires).** Different instruments have different noise
structures, so one rule does not fit:

| instrument | statistic | CI construction | resolution (LAW-60 / `METRICS §4.1`) |
|---|---|---|---|
| MoK campaign | median of 5 rotations | **paired**, same-session, alternating `c,b,c,b`; CI from the paired differences across K campaign pairs | campaigns reproduce to **0.09%**; screens drift **6.6% within one batch** |
| device stamps | per-phase, rank-0 CTA-max | 3 repeats, report min/median/max; **final soak epoch only** | plan σ **4.7 µs (1.1%)**, M7 σ **63 µs (2.3%)** |
| boundary rig | median wall of 7 iters | 3 invocations, report the spread | depth arms separated at **0.1–0.3%**; slab arms at **2.1×** |
| serving pair | median of ≥5 order-balanced pairs | geometric mean of pair ratios + full spread | **±15% day / ±18% position** at c32p; **≤1.8% M1 / ≤0.25% latency** at c512p |

**Deliberately scoped out of the θ table (declared, not silently omitted):**

| θ | why scoped out |
|---|---|
| **θ-F15 (SDMA in situ)** | never measured; the one API measured was CU-lowered, not SDMA (LAW-6). Nothing in the ledger supports or refutes it. Scoping it out is cheaper than a bad measurement |
| **multi-node** | zero evidence (H10). `W_VECTOR §4.6` already scopes it out; every fabric law we own is an **intra-node xGMI** law |
| **energy** | **not instrumented anywhere in this repo.** `METRICS §1.6` gates it behind G-E1 (one node command proving `rocm-smi --showpower` returns a live per-GPU value at ≥1 Hz, plus a null-control pair). Until G-E1 passes, energy is reported as `not instrumented` — **never estimated from TDP** |
| **θ-F17 (gfx942 link/egress card)** | access-gated; folded into the ARCH cell K8 (§3.5) and reported as a separate arch card if the node materializes |

### 1.7 Phase 0 budget

| item | node-hours |
|---|---:|
| P0-Z (free CPU triple) | **0.0** |
| P0-A transport-only | 1.6 |
| P0-B boundary + ledger | 1.3 |
| P0-C mega chassis | 1.8 |
| P0-D serving | 1.9 |
| retries / gate failures (20%) | 1.3 |
| **Phase 0 total** | **≈6.0** |

---

## 2. PHASE 1 — REGISTERED PREDICTIONS

### 2.1 Why this phase exists at all

This is the phase that makes the campaign **science rather than search**. The corpus already
contains the counter-example: G25-1's four schedule hypotheses were tried, each returned null,
and the honest post-hoc reading (`TP8_STATE §3.1` meta-observation) is *"a parameter that four
independent producer-side interventions cannot move is almost certainly not on the producer
side."* That reading is worth something **only because the four were pre-specified as
mechanism claims**. A grid run without registration produces a lookup table, which
`W_VECTOR §0` correctly names as the failure mode the expert feedback warned about
("very expansive, unclear what the core contribution might be").

Registration also gives the campaign its **cheapest output**: the four-door screen (D11) has an
**11/11 retrospective hit rate** on the falsified ledger, so every zero-door arm can be
rejected before it costs a node-hour — and the registered forward prediction is that *every
zero-door arm measures inside ±1σ of its control*.

### 2.2 The artifact

```
distributed-kernels/fused_moe/overnight/aug18-ablations/manifests/
├── predictions.jsonl          # THIS phase's output; one row per (cell, prediction)
├── cells.jsonl  builds.jsonl  arms.jsonl  runs.jsonl        # S_VECTOR §4.2
└── replay/<arm_id>.sh
```

```jsonc
{ "pred_id": "P3",  "cell_id": "B.tp8.lambda",  "registered_utc": "…", "registered_sha": "…",
  "theta_version": "phase0",                      // or "banked" for the pre-registration
  "statement": "K1 flips from store_towers to atomic_accumulate below b* tokens/boundary",
  "quantity": "b*", "point": 32.8, "unit": "MB", "band": [15.0, 60.0],
  "derivation": "b* = (L_t - L_a)/(1/R_a - 1/R_t) = 243/7.410",
  "falsifier": "a slab_rows-matched sweep whose crossover falls outside [1500,4000] tokens",
  "on_falsify": "K1 is not affine in bytes; the fixed terms are contaminated by grid occupancy",
  "adjudicating_arms": ["B.tp8.lambda__S6__k1=atm.sl=256__…", "…"] }
```

### 2.3 The registered prediction table

Registered **twice**: once now with **banked θ** (`theta_version: "banked"`), once after Phase 0
with **calibrated θ** (`theta_version: "phase0"`). **Both are committed and the diff is
reported.** A prediction whose *sign* changes between the two registrations is itself a
finding: it names a θ entry the model is dangerously sensitive to.

| id | cell | registered prediction (banked θ) | falsifier / what a miss means |
|---|---|---|---|
| **P1** | `K.T*.bal` | crossover **T\* = 1,311 tok/rank**, band **[1,100, 1,900]**. Three methods disagree by ~25%: affine cost fit **1,311**; ratio interpolation **1,398–1,483**; the repo's own reading **1,600–1,800** | outside [1,100, 1,900] ⇒ the linear cost model is wrong and **α is not the mediator** |
| **P1b** | `E.c32p.dp` | a **fill-aware** M15 lands within **±10% of parity** on real work per step at c32p (`T_eff` = 1,539, barely above T\*) ⇒ **the c32p loss is not recoverable by fill-awareness alone**; that cell must be won by the ~56% dummy-step skip | fill-aware M15 wins c32p by >10% ⇒ the affine model under-states fusion's per-token rate |
| **P1c** | `K.T4096.phi` | with `M24_FILL=1` at φ=0.3757, T_pad=4096: `T_layer` = 1,068.2 + 1.1623×1,539 = **2,857 µs** vs **5,829 µs** at FILL=0 ⇒ **−51.0%** (DERIVED) | FILL=1 measures within 10% of FILL=0 ⇒ the five T-substitution sites do not actually make cost row-proportional, and M24 is a work-*deletion* feature only |
| **P1d** | `K.T4096.phi` | **at fixed T_pad, φ does NOT move the DP kernel *ratio*** — both arms pay the padding (`VLLM_MOE_SKIP_PADDING` defaults to 0; the router emits ordinary top-k for pads; Mori dispatches `a1` unsliced). Predict the M15/production ratio is **0.756 ± 0.02 at every φ** | the ratio moves >5% with φ at fixed T_pad ⇒ fill **is** a kernel-quality axis and `W_VECTOR` Q8's topology-tax reframe is wrong |
| **P2** | `B.tp8.ratio` | **h\* = 48.7%** = `1 − (1 − 335/1470)/1.506`. At the corrected ratio with an **unchanged schedule**, hiding lands in **[10%, 35%]** — i.e. still below break-even | **<10%** ⇒ the "hide it in the epilogue" thesis is in serious trouble (this is `TP8_STATE` M2's own pre-registered falsifier); **≥49%** ⇒ fusion already pays and the transport track is deprioritised |
| **P2c** | `B.tp8.ratio` | at β = 1.0 (RCCL parity) the bar falls to **22.8%**; at `I_co → 0` it falls to **0** ⇒ **the transport track is worth more than the hiding track until β < 1.2** | a run at h ≥ 49% that still loses ⇒ (D2) is missing a term, most likely a second co-residency tax on the *compute* side (I multiplicative, not additive) |
| **P3** | `B.tp8.lambda` | **b\* = 32.8 MB ≈ 2,289 tokens/boundary**, band [1,500, 4,000] — lands on the design doc's independent *"chunking collapses below ~2K tokens"* | crossover outside the band ⇒ the two endpoint configs' fixed terms are contaminated by grid occupancy and K1 is not affine in bytes |
| **P3b** | `K.gfx942.k1` | on gfx942 (`R_a` ÷3) **b\* falls to ≈11 MB ≈ 780 tokens** ⇒ **store-towers is right almost everywhere there**, which **simplifies the PR** | atomics winning above 1,000 tokens on gfx942 |
| **P3c** | `B.tp8.lambda` | **every decode-cell conclusion is blocked on θ-F13.** Registered: RCCL at 3.67 MB lands in **[150, 400] µs**, i.e. our 801 µs (atomic) / 1,017 µs (towers) point is **2–5× off the achievable latency floor** | RCCL within 20% of 801 µs ⇒ the decode transport track is dead and decode is a weight-prefetch problem only (§6.4) |
| **P4a** | `K.topk` | **producer-order restructuring + slab certification pays iff top-k > 3.** `Φ_slab(2) = 0.25` vs `Φ_natural(k) = 1/(k+1)`. At k=8, Φ goes 0.111 → 0.250 (**2.25×**); at **k=2** the inequality **reverses** (0.333 > 0.25) and per-row readiness becomes viable while slab certification becomes a pessimisation | no sign flip between k=2 and k=8 |
| **P4c** | `B.tp8.g` | `g* ∝ T^{1/(α+1)} ≈ T^{1/3}`. Fit on the measured slab sweep gives `a = 2.647 µs/slab`, `α ≈ 2`, `g* = 193` rows vs measured optimum **256** (one grid step). ⇒ at **T=4,096 predict 128 rows** (≈122); at T=2,048 predict 128 (≈96) | the optimum stays at 256 ⇒ `g*` is absolute (hardware quanta), not T-dependent, and (D4b)'s protocol term is mis-specified |
| **P5d** | `B.tp8.depth` | depth **re-acquires** its effect in the fused arm once `co` returns to 1 at the corrected ratio. **And the metering reinterpretation predicts `d*` moves DOWN**: `d*(fused, ρ≈3) ≤ 4`, plausibly 2 | `d*` stays 4 or moves **up** ⇒ the metering interpretation (duty-cycle, not congestion) is wrong. Either outcome is a campaign law |
| **P6** | `K.T4096.C` | **dedicate CTAs to consuming, never to carrying**, and only when `V_cert > 0`. Carrying **+332…+340 µs**; idling **+8–9 µs/CTA monotone**; consuming **−93 µs (C 8→16), −444 µs (C 16→24)** | a carrier pool that wins in any cell ⇒ `value(carry) > 0` somewhere and D6 needs a carrier-job term |
| **P6c** | `K.T4096.skew` | under **adversarial** skew a *carrier* pool (S2) still loses. This is the one pre-registered claim in the ledger that is **untestable, not refuted** (`K0_SYNTH_ROUTE` rejected; reachable CV 0.0034–0.0086 vs COMET's 0.032) | S2 wins at any C under adversarial skew ⇒ LAW-28 is regime-conditional in a second dimension and "dedication is bad" must be re-scoped again |
| **P7a** | `K.T4096.C` × φ | **`C*` is first-order INVARIANT in `T_eff`** (both `V_cert ∝ T_eff` and `τ_shadow ∝ T_eff`). `C_sat = 337 MB/(6.1 GB/s × 2.66 ms) = 20.8` ⇒ predict `C* = 24 ± 4` at every φ. **This contradicts the kernel's own comment** at `:2122-2127` | `C*` moves >1 grid step ⇒ `τ_shadow` is not linear in T_eff (likely `φ_M7 → 0` at small T) and M needs a `φ_p(T)` term |
| **P7c** | `K.T4096.skew` | under σ > 2 the hot rank's certified prefix arrives later ⇒ `τ_shadow` shrinks while `V_cert` is unchanged ⇒ **`C*` RISES with σ**, and a **single global C cannot be right on all 8 ranks** ⇒ per-rank C beats global C | a per-rank-C arm that **ties** global C under replayed skew |
| **P8** | `B.tp8.C` | **MAG's pull runs 256 blocks — exactly the measured pull-collapse edge** (*"pull collapses past 256 CTAs while push degrades gently"*). Reducing pullers to ≤128 recovers bandwidth | a puller-count sweep {64,128,192,256} that is **flat** ⇒ the collapse knee does not transfer to the 235 MB many-to-one case and θ-F10 must be re-measured at deployment size |
| **P9** | `E.c32p.rr` | RR/EPLB0 placement is worth **+27% to +46% ABSOLUTE** node throughput at c32p **for BOTH arms**, with **<3% relative A/B change**. Larger than LAW-50's 15–25% and larger than any kernel delta in the ledger. **Deliberately registered as an over-prediction** | **<10%** ⇒ `λ_serving ≪ λ_rig`, i.e. **the replay rig over-states skew because it replays skewed routing on 100% of rows while only 37.6% of serving rows are real** — which would invalidate every replay-based skew number. **[27%, 46%]** ⇒ placement, not schedule, is the headline. **ratio change >5%** ⇒ `λ_M15 ≠ λ_prod` matters at serving scale |
| **P9adv** | `K.T4096.skew` | at adversarial σ = 8 (all top-8 mass on one rank), (M13) predicts `5,823 × (0.3734 + 0.6266×8)` = **31,364 µs** (DERIVED, out of the fitted range) | measured outside ±10% ⇒ (M13) is not linear in σ beyond [1, 5.6], most likely becoming **super-linear** as the hot rank's certificate wait turns convex |
| **P11** | all | **every zero-door arm measures inside ±1σ of its control** (four-door screen, 11/11 retrospective) | any zero-door arm with a reproducible **>3%** effect ⇒ **there is a fifth door** and the occupancy-1 selection rule is incomplete. *This is a genuinely useful thing to be wrong about and must be reported loudly* |
| **PF8** | `B.tp8.fp8` (GATED on X5) | fp8-on-wire is **1.939× fewer bytes** ⇒ boundary transport falls by `(1 − 1/1.939) × 2,214` = **−1,072 µs**. Under TP8 the wire time is **already exposed**, so the FP8 doc's central risk (converting hidden wire time into exposed) **does not apply** | **<−500 µs** ⇒ the transport is not byte-limited at this size (consistent with the 58%-of-ceiling framing) and the lever is **issue rate**, not bytes |
| **PHY** | `B.hyb.world` | world 8 → 4 drops per-rank egress from `2·7/8 = 1.75×B` to `2·3/4 = 1.50×B` (**−14.3%**) and drops fan-in 8→4. Predict wall ratio ∈ **[0.78, 0.86]** | ratio ≥ 0.95 ⇒ the AR is latency/protocol-bound, not egress-bound, at this size — which would re-scope every byte-based topology argument |
| **PE1** | `E.c512p.tp` | **TP8-native at C=512 lands within ±10% of its own c32p value (25,844 → 23.3k–28.4k input tok/s)**, because its chunk size (and therefore per-step work) is concurrency-independent, while DP8 rises 20.4k → 42.8k because φ rises 0.376 → 0.985 | TP8 at C=512 exceeds **35k** ⇒ TP8 also benefits from concurrency, and the "topology flips with concurrency" law needs a mediator other than φ |
| **PE4** | `E.decode` | at OSL 512 / ISL 1024 the megakernel's advantage **collapses**: predict `m15/native ≥ 1.0` (we tie or lose) on M6, because post-M23 the mega executes decode steps **at prefill shape** (4,096 padded rows for ~1 real token per rank) | m15 wins TPOT p50 by **>20%** at OSL 512 ⇒ the 4.24× TPOT win at c512p is **not** a closed-loop batch artifact and there is a real decode mechanism we do not understand |
| **PE5** | `E.mixed` | the dummy-skip prize (**56.0%** of in-bucket steps at c32p) falls by **≥half** under open-loop Poisson at 75% of ceiling, because the closed-loop dummy share is *"partly an artifact of concurrency-32 with OSL=8 — ranks idle at the tail of every wave"* | dummy rate stays **≥45%** open-loop ⇒ it is production value, not a benchmark artifact. `METRICS §2.2` calls this *"the campaign's biggest single exposure"* |
| **PF4** | `K.T4096.doors` | **the injection bound and coarse readiness are SUBSTITUTES, not additive.** −613.5 µs and −573.3 µs close within 7% and both plausibly bound the same in-flight-remote-write resource. Predict `Δ(depth ∣ cert ON) ∈ [−150, 0] µs` | `Δ ≤ −400 µs` with cert ON ⇒ they **compose**, and every waterfall in the paper **is** additive after all. This **changes M's structure, not its coefficients**, which is why it is a failure mode and not a calibration item |

### 2.4 The re-registration rule

After Phase 0, each row is re-derived with calibrated θ and written again. Three outcomes,
each with a pre-committed response:

1. **Point moves <1 band-width** → publish both, proceed.
2. **Point moves >1 band-width but the sign is unchanged** → publish both; flag the θ entry as
   high-leverage in the θ-sensitivity table; proceed.
3. **Sign changes** → **stop.** A prediction whose sign flips on calibration means the boundary
   was inside the θ uncertainty all along; the cell is re-designed to bracket it before it runs.

---

## 3. PHASE 2 — THE BOUNDARY GRID

### 3.1 Design rule, and the settability gate

**Rule (from `W_VECTOR §4.1`).** Do not sample corners of **W** (21 axes, most unsettable).
Sample corners of **Q** (10 coordinates, 6 cheaply reachable on a boundary rig). A cell earns
its place only if it moves a Q coordinate no cheaper cell moves, **or** it breaks a
collinearity today's sample cannot separate:

| collinearity in today's two-serving-point sample | breaking cell |
|---|---|
| φ ⟂ C_req ⟂ κ_sched ⟂ variance envelope (all move together between c32p and c512p) | **K2** (synthetic φ at fixed T_pad) and **E3/E-F** (ISL and concurrency ladders) |
| φ ⟂ T (`K0_MAXTOK == T` makes every T-sweep a **batch-size** sweep) | **K2** via `NORIG_CONST/_TABLE` |
| κ ⟂ φ (**padding generates skew** — LAW-42) | **K3** (route replay with pad rows excluded) |
| ρ ⟂ everything in TP8 (one operating point) | **B1** (`k_inner`) |
| topology ⟂ concurrency (the measurement is a **diagonal**, not a 2×2) | **E-C** (TP8-native at C=512) |

**Settability gate (LAW-62, binding).** Every cell below carries `actuator` and
`actuator_status`. A cell whose actuator does not exist is **GATED** and does not appear in the
budget until its engineering item lands.

### 3.2 Sample size — deriving K instead of asserting it

**Boundary-level variance, from the grounding:**

| source | statistic | value |
|---|---|---|
| LAW-60 | campaign reproducibility | **0.09%** |
| exp_35 | campaign sd, three rungs | 13.7 / 15.5 / **6.9 µs** on 7,110.8 / 6,900.2 / 6,495.8 µs ⇒ **0.11–0.22%** |
| LAW-60 | screen control drift **within one batch** | **6.6%** (6,795 → 7,244) with no foreign process ⇒ *"treat ~3% as the smallest callable end-to-end screen delta"* |
| `METRICS §4.1` | boundary rig, 7-iteration median | depth arms separated at **0.1–0.3%** and correctly read as "flat" |

**Take `σ_campaign = 0.20%` of the wall** (the conservative end of exp_35's range). The 6.6%
drift is **common-mode** and is removed by the exp_35 discipline: same-session, alternating
`c,b,c,b` campaign pairs. For a paired contrast over **K campaign pairs**, with
`σ_diff = √2 · σ_campaign = 0.283%`, MDE at α=0.05 / power 0.8 is `2.80 · σ_diff / √K`:

| K (alternating campaign pairs) | MDE (% of wall) | MDE at T=4096 (µs) | resolves |
|---:|---:|---:|---|
| 1 | 0.79% | 46 | C 16→24 (−444), skew ladder (3.6×), slab 256↔1024 (2.1×) |
| **2** | **0.56%** | **33** | depth 4↔unbounded (−615), carrier relocation (+211), certificate deletion (−573) |
| **4** | **0.40%** | **23** | **C 24→28 (−26.5 µs = −0.45%)**, depth 4↔8 (−50.9 µs) |
| 6 | 0.32% | 19 | the residue floor; below this the ledger's own sds dominate |

**Rule R-K1.** `K = 1` for cells whose *smallest* registered delta exceeds 3% (screening /
ordering only). `K = 2` default. `K = 4` for any cell whose registered delta is <1%
(the C fine-structure and the depth fine-structure).
**Rule R-K2 (boundary rig).** 7-iteration median is the atom; `σ_rig ≈ 0.15%` ⇒ 3 invocations
resolve ≥0.35%, 1 invocation resolves ≥0.6%. Use 3 for any predicted delta <5%, 1 otherwise.
**Rule R-K3 (LAW-61).** **Runs are the unit of analysis, never (run, rank) cells** — the same
null reads `t = −5.01` treated as 160 independent cells and `t = −0.73` with runs as units.
**Rule R-K4.** A **hang is a manifest outcome value** (`outcome: "hang"`), never a dropped
point. `C=32` hung 1 of 2 runs, and `C*` sits one grid step below a hang — that is a
schedule-space boundary and it must be recorded as one.

### 3.3 The TP8 / boundary cell matrix (instrument: `m25_boundary_bench` + `m25_cdar_bench`)

| id | cell | axes swept | S-arms (after pruning) | K | pts | h | adjudicates |
|---|---|---|---|---:|---:|---:|---|
| **B1** | `B.tp8.ratio` | `k_inner` → ρ ∈ {0.65, 1.0, 2.0, 3.0, 4.0} | S0-TP (`rccl`), S6-phased, S6-fused, compute-floor | 3 | 20 | 0.5 | **P2, P2c, P5d**; re-scores F1–F4 |
| **B2** | `B.tp8.lambda` | `b` ∈ {3.67, 15, 33, 60, 235} MB × transport {atomic, towers} × co {0, 1} | S6, D1, S0-TP(`rccl`) at every size | 3 | 45 | 1.2 | **P3, P3b-prep, P3c** — and it **unblocks the entire decode branch** |
| **B3** | `B.tp8.g` | `slab_rows` ∈ {64,128,256,512} × tokens ∈ {2048, 4096, 16384} | S6-fused, transport-only | 1 | 24 | 0.4 | **P4c** |
| **B4** | `B.tp8.depth` | depth ∈ {2,4,8,16,32} × co ∈ {0,1} at ρ≈3 | S6-fused, transport-only | 3 | 20 | 0.7 | **P5d** + the metering reinterpretation |
| **B5** | `B.tp8.C` | consuming pool C ∈ {0,8,16,24,32}; MAG puller count ∈ {64,128,192,256} | S6-fused | 3 | 18 | 0.5 | **P8**, D7 under TP8 (F3 tested only the C=0 endpoint) |
| **B6** | `B.tp8.wave` | `fused_wave` arm (`threadIdx.x ≥ 192 → transport`) × depth {2,4} at ρ≈3 | S6-fused + new arm | 3 | 8 | 0.3 | the **new door**: discriminates the *resource-class* explanation from the *schedule* explanation that four null mutations left open |
| **B7** | `B.hyb.world` | world ∈ {2,4,8}, transport-only, matched bytes/rank | S6 transport | 3 | 9 | 0.3 | **PHY** — the hybrid/N-dependence cell |
| **B8** | `B.tp8.shexp` | a routing-independent MFMA burst sized at **1/8 of routed FLOPs** injected as work-queue fallback at ρ≈3 | S6-fused ± filler | 3 | 8 | 0.4 | atlas B2: the shared expert alone covers **~32–34%** of the exposed AR (0.9–0.95 ms of 7.2–7.6 ms) — **a third of the prize, not the prize** |
| **B9** | `B.tp8.fp8` **GATED (X5)** | wire dtype {bf16, fp8-e4m3 + per-128 fp32 scales} | S8-derived | 3 | 6 | 0.3 | **PF8** |

**Subtotal (B1–B8, B9 gated): ≈4.3 node-hours.**

**Two pruning notes that must not be over-generalised** (`S_VECTOR §3.2.1`):
* **B3 is kept at decode size** even though `atomic_accumulate` is pruned at bulk size —
  it *wins* there (801 vs 1,017 µs at 3.67 MB). **K1 is a second manifest axis, not a global
  decision.**
* `slab_rows = 1024` is dropped (2.1× slower) and `bps = 1` is dropped (one block per slab
  leaves the grid half-idle, 41.0 vs 67.1 GB/s); `bps = 16` is dropped in the fused arm
  (+180 µs of pure protocol, still zero hiding).

### 3.4 The DP/EP cell matrix (instrument: MoK synthetic prefill campaigns)

| id | cell | axes swept | S-arms (after pruning) | K | campaigns | h | adjudicates | gate |
|---|---|---|---|---:|---:|---:|---|---|
| **K5** | `K.T4096.doors` | {depth bound on/off} × {slab certification on/off} — a **2×2, not two ladders** | S3, S4 | **4** | 16 | 0.8 | **PF4** — the LAW-37 substitutes question. **Cheapest of all the F-tests; it changes M's structure if it fails** | — |
| **K4** | `K.T4096.C` | C ∈ {0,8,16,24,28,32} × `flush_rows` ∈ {0,8,16,32} (screened, not crossed) | S4, + S2 control at C ∈ {32,64}, + S1 | **4** | 36 | 1.9 | **P6, P7a** — the literal answer to *"does our M15 producer/consumer work best for all sizes?"* | — |
| **K1** | `K.T*.bal` | T ∈ {512, 1024, 1536, 2048, 3072, 4096, 8192} | S0, S1, S4-C28 (all three inside **one** campaign invocation) | 2 | 14 | 0.75 | **P1** — the single most quotable number the campaign could produce | **X2** |
| **K3** | `K.T4096.skew` | routing ∈ {balanced, aggregate-hist, worst-layer, **captured**, **shuffled-captured**, **adversarial-synthetic**} | S4-C24, S4-C28, S2-C32, S2-C64 | 2 | 48 | 2.5 | **P6c, P7c, P9adv, P9b** | — (see actuator note) |
| **K2** | `K.T4096.phi` | φ ∈ {0.25, 0.376, 0.5, 0.7, 1.0} at **fixed T_pad = 4096** × C ∈ {16, 24, 28} | S4 ± `M24_FILL` | 2 | 30 | 1.6 | **P1c, P1d, P7a** — breaks the **worst collinearity we have** | **X1** |
| **K6** | `K.topk` | top-k ∈ {2, 4, 8} × order ∈ {natural, nc-major + S=2} | S1, S4 | 2 | 12 | 0.6 | **P4a** — the cleanest sign-flip law in the campaign | **X3** |
| **K7** | `K.decode.dp` | per-rank T ∈ {8, 32, 128, 256} | S0, S1, S4 | 2 | 8 | 0.4 | the α → 1 limit; see §6.3 | **X2** |

**Subtotal: ≈8.5 node-hours** (164 campaigns × 3 min 5 s, plus route-replay campaigns at
**139 s** each for K3's captured/shuffled rows).

**Actuator notes for K3 (this is what makes the skew axis real).** `K0_SYNTH_ROUTE` is
**rejected** under `K0_INPUT_MODE=mok_synthetic`, the synthetic-route families are decode-only,
and the only reachable knob (seed) spans destination-load CV **0.0034–0.0086 — 4–15× *less*
imbalance than COMET's 0.032**. So the axis is built from three actuators that **do** exist:

1. **aggregate-hist** — `K0_MOK_ROUTE_HIST` (Gumbel-top-k), marginal fidelity corr **0.998**,
   gini 0.331 vs 0.341 target. Works today. **Destroys per-chunk composition** — say so.
2. **captured / shuffled-captured** — `mok_synthetic_prefill/route_replay.py`, **code-complete,
   22 CPU tests pass, NEVER EXECUTED ON GPU**. It is the *only* experiment that separates
   popularity concentration from run correlation at kernel speed, at 139 s/campaign vs ~25 min
   per serving pair. `route_multiset_sha256` equality proves only correlation changed.
3. **adversarial-synthetic** — a hand-built 256-bin histogram placing all top-8 mass on one
   rank's block (σ → 8.0), fed through the same `K0_MOK_ROUTE_HIST` path. **This is the cell
   that makes P6c settable for the first time**, and it is the honest answer to LAW-62(b)'s
   "untestable, not refuted."

**Registered kill-criterion for K3's correlation half:** if **captured ≈ shuffled within 3%**,
run-correlation is dead at kernel speed too (it is already falsified as first-order at the
corpus level: 57.0% vs 55.7% expert overlap, a **1.3-point excess**), and mechanism (a),
destination interleaving, is closed.

### 3.5 The remaining required axes: arch, precision, hybrid

| axis | cell | how it is actually swept | honest status |
|---|---|---|---|
| **ARCH** | **K8** `K.gfx942.k1` | run the *existing* `m25_cdar_bench` (transport is already argv there) on a gfx942 node: transport {atomic, towers} × `b` ∈ {3.67 … 235} MB, plus depth {2,4,8,16,32} | access-gated; ≈1.0 h; adjudicates **P3b** and **P5b**. If gfx950 and gfx942 now agree at deployment message size (as F5 suggests), **the cell is one run and the result is a simplification for the PR** |
| **PRECISION (F1)** | rides every serving arm | **not swept — verified.** LAW-56: the production FP8 training bar ran its MoE in **BF16** because the `delayed` recipe returns a null turbo quant config, and fp8-in-MoE was worth **~0 ms**. ⇒ one **runtime** proof line per arm (`moe_path_precision` in the workload tuple, block 2), never a config read | a config-derived precision claim is a **protocol violation**, not a style issue |
| **PRECISION (F2, wire)** | **B9** `B.tp8.fp8` | **GATED on X5.** gfx950 has **no fp8 remote RMW**, so F6 is *not reachable from the mode-12 epilogue at all*; it is reachable only atop S8, whose scratch pathology is P0 | if X5 does not land in the first two weeks, the precision axis is a **single declared coordinate**, not a swept one — and the ladder says so rather than pretending otherwise |
| **HYBRID TOPO** | **B7** `B.hyb.world` (boundary) + **E-H** (serving, deferred) | boundary: recompile the rig at `kWorld` ∈ {2,4,8} and price `2(N−1)/N` egress + the fan-in change. Serving TP4×DP2: **no repo artifact configures it** — SPECULATION, deferred to Phase 4b | TP4×DP2 is the only hybrid with a principled motivation from our own data (halves AR participants while retaining half the DP fill dilution) and it is **unbuilt** |

### 3.6 The pruned S-arm list, per cell class

Derived from `S_VECTOR §3` (four constraint classes). The pruning arithmetic is the
justification for the whole grid: full crossing of S4's own knobs at **one** cell is
`C(6) × depth(5) × flush(4) × order(4) × K1(3) × slabs(2) × mode(3) = 8,640` arms; over six
cells that is **51,840 campaigns ≈ 2,679 node-hours**. The matrix below is a **288× reduction**.

| cell class | S-arms carried | pruned out, with the law that prunes it |
|---|---|---|
| DP/EP prefill, T=4096 | S0, S1, S3-C{0,4,8,16}, S4-C{0,8,16,24,28,32} × flush{0,8,16,32}, S2-C{32,64} as the **control** | S5/F9 (+75.8 µs, keep 1 archival point); mode-3 XCD confinement (7–23% worse, keep 1 known-bad control); pacing a dedicated pool (`M2→end` rises monotonically 6,229 → 8,134 µs, **no interior minimum**); coalescing protocol atomics (≥10×); deleting the per-task VMEM drain (+3.2 µs, sign reversed) |
| DP/EP, T ≠ 4096 | **all S4 arms are VOID on the serving body** (Class C4) | the M15 body failed the MoK gate at T=2048/1024 while `production` passed **in the same runs**. Run on the tgen body and **label it**, or land X2 |
| TP8+EP | S0-TP(`rccl`), S6-{phased, fused, fused_wave}, D1 at decode sizes | `slab_rows=1024` (2.1×); `bps=1` (grid half-idle); `bps=16` fused (+180 µs, zero hiding); atomics at bulk size — **but kept at decode size**, where they win |
| any TP8 cell | — | **the entire FILL lever is Class-C invalid**: no DP lockstep ⇒ no `execute_dummy_batch`, no 56%-dummy steps; the M23 ragged seal is a per-rank DP-batch concept; the MORI all2all machinery is structurally absent |
| any arm | — | **Class D (measurement-void):** a mode-12 number from a `MODE14=1` or `TBO=1` binary; a `flush_rows` value without its `mode`; any S8 × depth arm before X5; an "inert" claim without `.text` sha identity **and** a flag-ON build that differs; a rank-N failure adjudicated from rank-0 stdout; (run, rank) treated as independent |

### 3.7 Ordering by information value

Ordering rule: **cells that discriminate between predicted decision boundaries first**, then
cheapest-first among ties. Formally, rank by (# registered predictions adjudicated × how
contested each is) ÷ node-hours.

| rank | cell | h | why here |
|---:|---|---:|---|
| 1 | **B1** `B.tp8.ratio` | 0.5 | **Blocking.** The G25-1 bar was arithmetically unreachable (ρ=0.65 ⇒ ceiling 65% vs a 90% bar). Until B1 runs, four falsifications and one RED gate are all scored against an impossible criterion. **One argv** |
| 2 | **K5** `K.T4096.doors` | 0.8 | Adjudicates a **structural failure mode** (F4). If the two mechanisms are substitutes, **no waterfall in the paper is additive** — that changes how every prior result is drawn |
| 3 | **B2** `B.tp8.lambda` | 1.2 | Adjudicates P3 **and unblocks the entire decode branch** (θ-F13). Every claim below ~2K tokens currently has **no basis** |
| 4 | **K4** `K.T4096.C` | 1.9 | The literal expert question, zero code, one runtime byte |
| 5 | **K1** `K.T*.bal` | 0.75 | The most quotable number; three methods disagree by 25% ⇒ **measure, do not interpolate** |
| 6 | **K3** `K.T4096.skew` | 2.5 | Skew is worth **3.6× wall** in the rig — 6× the largest schedule knob — and has **never been driven as a controlled axis** |
| 7 | **K2** `K.T4096.phi` | 1.6 | Breaks the collinearity the flagship result rests on (two cells differing in ~six variables at once) |
| 8 | **B4** `B.tp8.depth` | 0.7 | Either re-establishes the −615 µs law under a new topology or retires it as DP-specific — **either outcome is a campaign law** |
| 9 | **B3** `B.tp8.g` | 0.4 | P4c; cheap |
| 10 | **B6** `B.tp8.wave` | 0.3 | Highest information per node-hour of any **new** arm: discriminates resource-class from schedule |
| 11 | **B5** `B.tp8.C` | 0.5 | P8; F3 tested only the C=0 endpoint |
| 12 | **B8** `B.tp8.shexp` | 0.4 | Bounds the one hiding vehicle whose slack is bounded **by construction** |
| 13 | **K6** `K.topk` | 0.6 | The cleanest sign-flip law — gated on X3 |
| 14 | **B7** `B.hyb.world` | 0.3 | PHY |
| 15 | **K7** `K.decode.dp` | 0.4 | Prices the α → 1 limit — gated on X2 |
| 16 | **K8** `K.gfx942.k1` | 1.0 | Access-gated; a simplification if it agrees |
| 17 | **B9** `B.tp8.fp8` | 0.3 | Gated on X5 |

### 3.8 Phase 2 budget

| track | node-hours |
|---|---:|
| boundary/TP8 (B1–B8) | 4.3 |
| MoK/DP-EP (K1–K7) | 8.5 |
| ARCH (K8) | 1.0 |
| gated (B9) | +0.3 if X5 lands |
| **Phase 2 total** | **≈14.0** |

---

## 4. PHASE 3 — MODEL FIT AND HOLD-OUT TEST

### 4.1 The split — declared before Phase 2 runs

**Rule (binding, from `COST_MODEL §4.1`): θ is NEVER re-fitted on a held-out cell. If a
held-out cell needs new θ, that is recorded as a MODEL FAILURE, not a calibration update.**

| set | cells / points | what is fitted |
|---|---|---|
| **FIT** | Phase 0 entirely; **B2's two endpoints only** (3.67 and 235 MB); **B4 at co=0 only**; P0-C's reserve-but-idle sweep; **K3's balanced + aggregate-hist points only**; K4's `C` ladder at **φ = 1.0 only** | θ-F*, `η_size`, α, `a`, `c`; `φ_p`; `ι`, `ψ`; **λ (one point per arm)**; `ρ`, `C_sat` |
| **HELD OUT** | **B2's middle sizes** {15, 33, 60} MB; **B3** at T ∈ {2048, 4096}; **K1's interior T** {1536, 2048, 3072}; **K3's worst-layer and adversarial** points; **K6's k = 2 and k = 4**; **B1 at ρ ∈ {2, 3}**; **K4's C ladder at φ ∈ {0.376, 0.5}**; **K8 (gfx942) entirely**; **all Phase-4 e2e cells** | tests **P3, P4c, P1, P9adv, P4a, P2, P7a, P3b/P5b**, and the lift |

The split is deliberately asymmetric: **the fitted set is the cheap, already-half-measured
corners; the held-out set is everything the campaign wants to *claim*.** Two entries deserve
naming because they are the strongest prior evidence the model has, and both were obtained
*before any new measurement*:

* **the skew term validated to 0.47% on a held-out point with ONE fitted parameter** —
  `λ_M15 = 0.6266` fitted on the aggregate-histogram replay predicts the worst-layer point at
  **22,582 µs vs 22,475 measured**; the same procedure on production gives `λ_prod = 0.5322`
  and **+1.17%**. Consequence: `λ_M15 > λ_prod` ⇒ **skew hurts us more than production**,
  independently predicted from the opposite direction by `aug18/FAIRNESS_AUDIT.md:58`;
* **the affine T-model reproduces an unfitted midpoint to +1.07% (mega) and +2.31% (prod)**.

### 4.2 The accuracy bars (from `COST_MODEL §4.3`, unchanged)

| metric | bar | rationale |
|---|---|---|
| median regret `R = [t(â_M) − t(a*)]/t(a*)` | **≤ 3%** | 3% is the smallest callable screen delta (LAW-60) |
| p90 regret | ≤ 8% | one grid step of a coarse C or T sweep |
| max regret | ≤ 15% | worse means M would pick a **qualitatively wrong skeleton** |
| Spearman `ρ_S` per cell (≥6 arms) | **≥ 0.70** | below this M cannot order arms and is not a screening tool |
| boundary-location error | **≤ 1 grid step** on every swept axis | the standard M already meets on `C_sat` (21 predicted vs 24 measured) |
| sign accuracy | ≥ 90% | below this the "if X then Y" law form is not supportable |
| θ stability | each entry within **±2× its own measurement CI** across cells | an entry that moves more is not a hardware constant |

**Instrument-scoping rule (binding).** These bars apply at the **boundary** level. At e2e the
minimum detectable effect is `~15%` at c32p and `≤1.8%` at c512p — **the variance envelope is
itself a function of W.** ⇒ *no e2e cell may be used to adjudicate a prediction smaller than
its own measured envelope.*

### 4.3 What happens on failure — decided in advance, in this order

1. **Check the instrument first.** A missed bar caused by a stale `.hsaco`, an unreset stamp,
   or a `MODE14=1` binary **is not a model failure**. The precedent is decisive: a commit whose
   diff *cannot* change mode 12 changed it by **+726.9 µs (11%)** because making mode-14 code
   merely *reachable* let the register allocator install a stronger involuntary throttle
   (issue runs collapsed from **282 atomics / 12 runs / mean 23.5 in flight** to **194 runs /
   mean 1.45**).
2. **Check whether the missed cell is one of the two named fudge factors' regimes.** If yes,
   run the device phase ledger (θ-P5/P6) **before touching M's structure**.
3. **Add a mediator only if a specific matched-Q / different-S\* pair is exhibited**, and only
   from the named candidate list, in this order:
   `co` (already in) → `L` (scatter width) → **phase indicator** → **topology indicator** →
   σ as a **distribution** → κ. **Never add a mediator without a cell pair that demands it.**
4. **Split the regime**, and declare the indicator explicitly. **Topology is the leading
   candidate** and it is half-predicted already: *fill is a DP-topology quantity and does not
   generalise to TP8 even in principle* ⇒ the campaign may need **two laws, not one**.
5. **Publish the miss.** *A cost model with a documented, bounded failure region is a result; a
   cost model with an undocumented one is a liability.*

### 4.4 The pre-committed degradation ladder (what the campaign still delivers on a miss)

| outcome | what ships |
|---|---|
| all bars met | **a cost model**: `S*(W) = argmin_S M(Q(W), R(S); θ)`, with 11 instantiated decision boundaries |
| median regret >3% **but sign accuracy ≥90%** | **a screening tool**: M orders arms and rejects zero-door proposals for free, but does not size them. The four-door screen's 11/11 retrospective record already supports this weaker claim |
| sign accuracy <90% in **one** topology | **a regime map plus one law**: the topology indicator is declared, and the law is stated per-topology (DP: `T_eff`; TP8: exposed collective fraction) |
| sign accuracy <90% in **both** topologies | **a regime map**, explicitly the weaker-but-honest fallback named in `COST_MODEL §0.2` — plus the falsified ledger, which is itself a deliverable |
| **FQ-1/2/3 falsified** (matched Q, different S\*) | **H-Q is dead and we say so**, name the missing coordinate, and ship the regime map. The three falsifiers are pre-registered precisely so this outcome is publishable rather than embarrassing |

---

## 5. PHASE 4 — E2E VALIDATION

### 5.1 Doctrine

`METRICS §4.4`: **R-S1** — no schedule claim ships without boundary evidence **and** at least
one e2e confirmation at a **matching W cell**; a boundary-only result publishes as
*"mechanism, unvalidated at deployment"*, an e2e-only result as *"observed, mechanism
unattributed"*; **neither is a law.** **R-S2** — boundary rigs *rank and falsify*; only e2e
*sizes*. **R-S6** — runs are the unit; ranks in a run are not independent samples.

### 5.2 The four mandated cells, plus the missing corner

| id | cell | class | PRIMARY metric | guards that can veto | n | h |
|---|---|---|---|---|---:|---:|
| **E-A** | `E.c32p.tp` — **C=32 TP8+EP, the current loss** (m15-DP vs `native_tuned_tp`) | **B** interactive low-C | **`goodput_in(a,b)` attainment CURVE over `a` and `b`**, not a point — because the deployable arm **flips at a TTFT SLO of ≈3.5–4.0 s** (TP8 97.3% vs m15 93.1% at 3,500 ms; m15 98.3% vs 97.9% at 4,000 ms) and at a TPOT SLO of ≈600–700 ms | TTFT p99, TPOT p99 (**m15 wins every p99 by 23/24/33%**), M1, coverage, accuracy | 5 pairs | 4.6 |
| **E-B** | `E.c512p.dp` — **C=512 DP/EP, the current win** | **A** batch/offline | **M1** aggregate input tok/s, steady-state window | accuracy; `sealed/in_bucket ≥ 0.95`; **`W` matched: `\|W_a/W_b − 1\| ≤ 0.05`**; memory + achievable C; TPOT p99 as a stability guard | 5 pairs (2 banked) | 2.8 |
| **E-C** | `E.c512p.tp` — **TP8-native at C=512: the missing corner** | A | M1 | as E-B | folded into E-B as a 3rd arm | 2.0 |
| **E-D** | `E.decode` — ISL 1024 / **OSL 512**, C ∈ {8,16,32} (the `c8/c16/c32` cells, *defined and never executed*) | **D** decode-heavy | **M6 = 1000/TPOT p50** per-request decode rate, with **TPOT p99 co-primary** | **M2 is BANNED here** (`M2 ≡ M4 × OSL` exactly in all 6 banked manifests ⇒ at fixed OSL it carries zero decode information); TTFT p99; KV capacity → achievable C; accuracy | 3 pairs (directional) | 4.5 |
| **E-E** | `E.mixed` — open-loop Poisson at 75% of the measured ceiling (`OPEN_LOOP_BASE_RATE = 4.97 req/s`) | **E** mixed continuous batching | **`goodput_in(a,b)` + `goodput_out(a',b)` as a PAIR**, traffic mix declared | **`κ` is a mandatory reported field** — a kernel claim whose κ differs across arms by **>5%** is a scheduler result mislabelled; W; both p99s; coverage; accuracy | 3 pairs | 2.8 |

**Phase 4a total ≈ 16.7 node-hours.**

**Phase 4b (conditional on Phase 3 passing):** `E-F` concurrency ladder C ∈ {64,128,256} × 3
arms (**H3: the crossover concurrency is "the single most quotable number the campaign could
produce, and we cannot currently state it"**) ≈5 h; `E-G` placement π ∈ {contiguous, RR}
(**P9**, the biggest predicted absolute number in the campaign) ≈4.6 h. **≈10 h.**

### 5.3 Mandatory coverage and composition counters (every arm, every cell)

Per `METRICS §3.5`, computed from workload-tuple blocks 5 and 6 and reported alongside them:

| counter | formula | gate |
|---|---|---|
| fill `φ` | `Σ in_bucket_sum_orig / (Σ in_bucket × 8 × 4096)` | **per cell** — *"37.6% is c32p's number"*; a cell without its own φ is not quotable |
| padding multiplier | `1/φ` | c32p m15 **2.66×**, stock 2.31× |
| dummy rate | `uniform_rescued / in_bucket` | whole-run, **not** the ramp window (N4) |
| **seal ratio** | `sealed / in_bucket` | **≥ 0.95, else the arm did not run the traffic** |
| **`W`** | `(in_bucket × 4096 × 8) / Σ sum_orig` | **`\|W_a/W_b − 1\| ≤ 0.05`, else the A/B is a composition result.** The precedent: `W = 2.568 vs 2.154 = 1.1922`, worth **+8.4% wall — the entire measured gap** |
| **`κ`** | prefill chunk-events per in-bucket step, of 8 | class-E gate; measured **3.93 (stock) vs 3.19 (m15)** |
| Little's occupancy | `M4 × e2e_mean / C` | **≥ 0.95** to call a closed-loop cell saturated |
| waves | `num_prompts / C` | **≥ 20** |
| accuracy | bounded rel-err (**not** exact-token SHA) | the kernel is **not bit-reproducible**: 116/1024 requests (11.33%) produced different tokens between arms, and `ordered_output_token_id_stream_sha256` differs even **stock vs stock** |

**A finding that falls straight out of applying our own gates to our own banked cell
(DERIVED).** The c512p cell as configured **fails two of them**: Little's occupancy **0.899**
(gate 0.95) and waves **`2048/512 = 4.0`** (gate 20) — it is a **4-generation run whose TTFT
p50 of 43.1 s is ~3 generations of queueing**. To make c512p admissible, either raise
`num_prompts` to **≥10,240** (which makes each pair ~5× longer, ≈2 h/arm) or accept the cell
labelled `R-L3 FAIL`. **The ladder's recommendation:** run the concurrency ladder at C ∈ {32,
128} where waves ≥20 is cheap (640 and 2,560 prompts respectively), and run **one** confirmatory
long c512p pair at 10,240 prompts; label every other c512p number `R-L3 FAIL` rather than
silently quoting it.

**A second one, which is the strongest metric-map argument the campaign owns.** At c512p,
`attainment(5000, 200) = 0/2048` for **BOTH** arms ⇒ **c512p is a batch cell under this
driver**, so its TTFT p50 delta (1.49× worse) is **not decision-relevant** — only throughput
and TPOT are. Meanwhile m15 hits **100% TPOT attainment at any `b ≥ 800 ms`** while
`native_tuned_dp` needs **`b ≥ 3,400 ms`**: `goodput_in(60000,1000)` = **42,835.7 vs
1,471.5 tok/s (29×)** against a headline delta of only **+6.60%**.

### 5.4 The one open contradiction, carried forward rather than silently forked

`METRICS §6.3`: `nightshift/BENCHMARK_PROTOCOL.md:61-64` makes **TTFT p50/p99** primary
("what users feel"). That is wrong under a **closed-loop** driver — all six banked manifests
are `driver=closed, request_rate=null` — and **TTFT p50 alone selects the wrong arm at c32p**.
The proposed amendment demotes TTFT p50 to a guard and promotes `goodput_in(a,b)` to primary in
classes B/C/E, labelling all closed-loop latency `(closed-loop, backlog-inclusive)`.
**Until the operator accepts, BENCHMARK_PROTOCOL wins.** Phase 4 therefore reports **both**,
and flags every cell where the two conventions select different arms.

---

## 6. PER-CELL KERNEL PREDICTIONS — the build plan

Format per region: **S\*** (predicted-winning schedule) · **mechanism** (why, via M) ·
**starts from** (existing file) · **implement** (S/M/L) · **falsification arm** (run alongside,
same session) · **how this prediction loses**.

### 6.1 Region 1 — high-C prefill, DP8/EP8 (`E.c512p.dp`, φ ≈ 0.985, `T_eff` ≈ 4,035, κ_agg 5.088×)

**S\* = S4 (M15)** with `K1 = producer-epilogue packed-bf16 RMW`, `K2 = nc-major`, `K3 = S=2`,
`K4 = depth 4`, `K5 = C ∈ [24, 28]`, `flush_rows` **scaled with `T_eff`** rather than carried.

**Mechanism.** `T_eff = 4,035` is **3× above `T* = 1,311`**, so the fixed-cost share
`α = 0.183 ≪ 0.412` ⇒ fusion pays with margin. `C_sat = V_cert/(ρ·τ_shadow) =
337 MB/(6.1 GB/s × 2.66 ms) = **20.8 CTAs**`, predicted knee 21 against the measured knee 24 —
**one grid step, with no fitting**. The same formula reproduces exp_29's own negative
(`675 MB/(6.1 × 16) = 6.9 ms ≫ 2.66 ms` ⇒ a pool-only combine cannot cover the work at any
legal C), which is exactly why M15 takes the **front half only**.

**Starts from.** `distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip` — shipped, and the
entire runtime schedule surface is **one 64-bit descriptor word**, so one HSACO serves the whole
sweep.

**Implement.** **S** — nothing new: `C × flush_rows` via `K0_MPS_CFG`, driven by the existing
manifest driver `overnight/aug11/tools/screen.sh` (`arms.jsonl → cfgs.txt` is a projection, not
new machinery). **M, optional** — `K0P6_M24_FILL_C` ("runtime-adaptive `reserved_comm_ctas`"),
currently a hard `#error` at `k0pf6gm_device_tile_m15.hip:262-270, 294-302`; build it **only if
P7a fails**, i.e. only if `C*` actually moves.

**Falsification arms (same session, alternating).** (i) **S2 carrier pool at C ∈ {32, 64}** —
predicted **+332…+340 µs**, and it is the control that makes "consuming, not carrying" a
measurement rather than a slogan; (ii) **`flush_rows = 0`** — a free control that isolates
**pure protocol deletion**; (iii) **S1 homogeneous** — the only arm that measures "megakernel,
no cross-rank protocol", which is the baseline LAW-21 is stated against.

> **How this loses.** P7a says `C* = 24 ± 4` at **every** φ. The kernel's own comment says the
> opposite (*"at `T_eff` … C / flush_rows are no longer at their tuned point"*). If `C*` moves
> more than one grid step across `φ ∈ {0.376, 0.5, 1.0}` at fixed `T_pad`, then `τ_shadow` is
> **not** linear in `T_eff`, M needs a `φ_p(T)` term, and the "one C fits every cell" claim —
> which is the cheapest thing the campaign could ship — is dead.

### 6.2 Region 2 — low-C prefill, TP8+EP (`E.c32p.tp`, the open loss, −21.17%)

**Champion baseline to beat, named honestly: the RCCL-hybrid** — GEMM → RCCL for the collective
(**2,909 µs** in-rig, **280 GB/s per-rank egress = 79% of the egress card**) with our
megakernel owning the MoE region. Any fused claim is measured against this, not against the
phased CDAR control.

**S\* — ranked, and the top-ranked one is not the fused kernel.**

1. **RCCL-hybrid (S0-TP collective + S4-style MoE).** `h* = 48.7%` and measured hiding is
   **2.1%** ⇒ the arm is **23× short of the real bar**, and the bar itself was set **1.85× too
   high** by the G25-1 gate. **Registered prediction: fused CDAR does NOT beat GEMM→RCCL in
   Phase 2 unless β is closed below ~1.2 first** (P2c).
2. **S6 + shared-expert filler (B8/atlas B2).** The shared expert is **1/8 of routed FLOPs**
   ⇒ at 7.2–7.6 ms of compute per layer that is **0.9–0.95 ms**, covering **~32–34% of the
   2.8 ms/layer exposed AR** — *a third of the prize, not the prize*, with **zero new
   machinery** (it is queue priority, and under TP8 its output folds into the same ERS
   accumulator, **deleting its separate all-reduce**).
3. **Metered ERS (atlas C6).** The most important reframe in the atlas: 822 MB/rank/layer over
   10.4 ms needs only **79.0 GB/s = a 28–39% fabric duty cycle**. ⇒ **hiding is
   scheduling-limited, not bandwidth-limited**, and *a 1.5× slower transport that runs
   continuously beats a fast one that bursts.* The cross-check that validates the pool: RCCL's
   fabric-busy time is `822 MB / 280 GB/s = **2.94 ms/layer**` and the banked *exposed* ring-AR
   is **~2.8 ms/layer** — the same number ⇒ **production hides essentially 0% of its
   collective**, and the 27% pool is real, not an estimate artifact.
4. **`fused_wave` (A8).** `__launch_bounds__(256,1)` = 4 waves/CTA, and `vmcnt` is one in-order
   counter **per wave**. Dedicating wave 3 to transport is **the only construction that yields
   an independent injection counter without raising occupancy** — so it pays **no CTA capacity
   tax** (8–9 µs/CTA) and no L2-placement penalty, and LAW-28's CTA-dedication falsification
   **does not transfer**.

**Starts from.** `distributed-kernels/tp8_mega/m25_boundary_bench.hip` (622 lines, four arms,
real MFMA body, `-lrccl` linked) + `m25_cdar.cuh`.

**Implement.** **S** — `k_inner` (argv 6, exists); **S** — transport argv (X4: `:282` hardcodes
`store_towers`); **M** — device phase ledger + duty histogram (E-B1/E-B2); **M** — the
shared-expert filler arm; **M** — metered ERS (spread commits across the **whole producing
GEMM**, not its epilogue); **M** — `fused_wave`; **L** — rolling planner + F3 (owner-sharded
RMSNorm, ~50–130 µs/layer) + F4 (gate GEMM inside).

**Falsification arms.** `compute` floor, `phased` (unfused control), `rccl` — all four in the
same invocation set at the same ρ, so the operating point is common-mode.

> **How this loses — in both directions, which is the point.** If at ρ ≈ 3 with filler and
> metering the fused arm hides **≥49%**, the RCCL-hybrid champion is beaten and the whole
> transport track (A1–A6) should be defunded in favour of hiding. If it hides **<10%**, *"the
> 'hide it in the epilogue' thesis is in serious trouble"* (`TP8_STATE` M2's own pre-registered
> falsifier) and the honest deliverable for this cell is: **the RCCL-hybrid, plus a published
> negative with a mechanism.** Note also that the −21.17% is an **arm-level** delta measured at
> two different W points — our arm runs `--max-num-batched-tokens 4096 --max-num-seqs 128`, the
> vendor TP8 arm runs `16384 / 2048` — so it **cannot be read as a kernel delta** at all.

### 6.3 Region 3 — decode, DP8/EP8

**S\* = S0 / production, and the lever is not in the kernel.** This is the region's honest
finding and it should be reported as a **result**, not a gap: post-M23 the uniform-decode
rescue routes decode steps through the B4096 graph, so the mega executes **4,096 padded rows
for ~1 real token per rank**. *Whatever the kernel does on a decode step, it does at prefill
shape.* ⇒ **there is no DP decode kernel problem today**; the DP decode lever is
**integration-side**: `max_num_batched_tokens` / bucket size (**H12, fixed at 4096 in every run
ever executed, never swept**) plus the tier-1 dummy-step skip (**56.0% of in-bucket steps at
c32p are 100% fake work both arms pay**).

**Mechanism, as a number (DERIVED from the affine fit `a_M15 = 1,068.2`, `b_M15 = 1.1623`;
`a_prod = 181.3`, `b_prod = 1.8387`):**

| per-rank T | M15 (µs) | production (µs) | **ratio** | `α_M15` |
|---:|---:|---:|---:|---:|
| 256 | 1,365.8 | 652.0 | **2.10×** | 0.782 |
| 128 | 1,217.0 | 416.7 | **2.92×** | 0.878 |
| 32 | 1,105.4 | 240.1 | **4.60×** | 0.966 |
| 8 | 1,077.5 | 196.0 | **5.50×** | 0.991 |

**Starts from.** The **tgen body** (branch `ablations-tgen`, sha `20b8c6bb`) for T-generality;
`aug18-prefill/M23_RAGGED_SEAL_DESIGN.md` and the vLLM control plane for the bucket lever.

**Implement.** **M** — X2: port the tgen M2 hole-sentinel fix (`NCHUNK_MAX·CHUNK == MAXTOK` is
true only at 4096; at MAXTOK ≤ 2048 phantom chunks overwrite live segments) into the serving
M15 body. **S** — bucket-size sweep, integration-side, zero HIP.

**Falsification arm.** `MAXTOK = T` — the **only confirmed production win on this axis**
(**−1.49%, ~148 µs** at prefill) — run at each decode-shaped T to check whether capacity
compaction rescues any of the ratio.

> **How this loses.** The table above **extrapolates a fit made on T ∈ [1,024, 4,096] down to
> T = 8 — a 128× extrapolation**, which is exactly the kind of thing that breaks. If the
> measured ratio at T = 256 comes in below **1.5×**, the affine model's fixed term is not
> constant at small T (most likely because `φ_M7 → 0` and the kernel becomes latency-dominated,
> which the model has no term for), and P1's whole `α` framing is scoped to `T ≥ 512`.

### 6.4 Region 4 — decode, TP8+EP (the latency regime)

**S\* = D1**, the decode-specific skeleton, and it is a **skeleton not a knob setting** because
three structural choices **reverse**:

| knob | prefill | decode | source |
|---|---|---|---|
| **K1** | store-towers win (**117.2 vs 67.1 GB/s** at 235 MB) | **REVERSES — towers LOSE** (1,017 vs 801 µs at 3.67 MB): the owner reduce is no longer amortised | `TP8_STATE §5 M5` |
| **K3** | interior optimum at 256 rows | **chunking collapses below ~2K tokens** — the slab concept dies | `M25_TP8_MEGA_DESIGN.md:212-215` |
| **K4** | −615 µs co-resident | **inert** — no bandwidth congestion to control | LAW-8/20 |
| mechanism class | bandwidth-shaped AR | **one-shot / LL-style AR, small-M tiles, possibly fused AR+RMSNorm** | `TP8_OVERLAP_ANALYSIS.md:110-116` |

**But the sharper prediction is that the collective is the wrong target.** At decode with a
few-hundred-token global batch × top-8, essentially every expert is touched: **32 local experts
× 42 MB = 1.34 GB/rank/layer** of weight streaming; at the exp_29-implied HBM ≈ 7.85 TB/s that
is **171 µs/layer at peak, ~342 µs at 50% efficiency ⇒ 10–21 ms per 61-layer step** — an order
of magnitude above any 3.67 MB all-reduce. ⇒ **at decode the "comm" worth hiding is the HBM
weight stream, not the fabric**, the hiding compute is **attention**, and **which experts to
prefetch is known the moment layer L's gate finishes**.

**Starts from.** `m25_boundary_bench.hip` with `mode=rccl tokens=256 slab_rows=64`
(**one argv** — the go/no-go); and `k0pf6gm_device_tile_m15.hip:126-149, :432, :2091-2100`
(M20 slot pool + `PF_CTAS`, where *"stream order **is** the certification; no flags, no tags"*)
for the prefetch arm — **the mechanism already exists and has never been pointed at decode**.

**Implement.** **S** — the RCCL decode reference (argv); **S (0 GPU)** — P0-Z3, the occupancy
check; **L** — D1 proper (latency-shaped transport, no tower fan-out, per-owner certificate
scope, AR+RMSNorm fusion).

**Falsification arms.** (i) replicate the atomic-vs-towers reversal at 3.67 MB — it is
currently **n=1**; (ii) `atomic` vs `towers` **co-resident** (requires X4).

> **How this loses.** Registered: *at decode, expert-weight HBM streaming exceeds the fabric
> collective by ≥5× per layer, so a fused CDAR decode arm measures ≤3% better than RCCL because
> both are latency-floor-bound.* This **loses if RCCL at 3.67 MB comes in below ~250 µs** — in
> which case our 801 µs point is 3× off the achievable latency floor, transport (not weights)
> is the decode lever, and D1's transport design becomes the priority. Note the standing
> caveat that makes every sentence in this region provisional: **decode has never been measured
> at all in any of our rigs.**

### 6.5 Region 5 — skewed routing (all DP cells; `E.c32p.rr`, `K.T4096.skew`)

**S\* = placement first (RR / EPLB0), then per-owner certificates — in that order, and the
order is the finding.**

**Mechanism.** (M13): `T_layer(σ) = T_layer(1)·[(1−λ) + λσ]` with `λ_M15 = 0.6266` ⇒
`dT/dσ = T(1)·λ = **3,649 µs per unit σ per layer**`. Compare the entire schedule-knob ledger:
the largest single knob (depth-4) is **615 µs**. **Skew is worth 3.6× wall (5,823 → 20,739 µs)
and dominates every schedule knob by ~6×.** Placement moves σ from **5.088× to 1.085×**
(worst-rank layer **5.593× → 1.296×**), and it does so **outside the kernel**.

**And the second-order structure matters.** Under (M13) the *ratio* between two arms is nearly
σ-flat when the λ's are close ⇒ **placement fixes are enormous absolute wins and near-zero
relative wins** — which is exactly why P9 is registered as `+27–46% absolute for BOTH arms with
<3% relative A/B change`.

**Starts from.** `aug18-prefill/rr_placement/rr_patch.py` (offline-verified; campaign 4 never
ran); `k0pf6gm_device_tile_m15.hip:81-83` (`SCATTER_RR`/m17), `:99-101` (`REPLICATE`/M18),
`:120-124` (`ADAPTIVE`/M19), `:147-149` (`M20_SLOTPOOL`);
`mok_synthetic_prefill/route_replay.py` (**code-complete, never executed on GPU**);
`m15_eplb0/eplb0_predict.py` (runs vLLM's own policy over our histograms with **no GPU**, and
carries its own gate: *if predicted worst-layer critical-rank relief is under ~15%, do not run
the arm*).

**Implement.** **S** — run `route_replay.py` on GPU (139 s/campaign); **S** — build the
adversarial synthetic histogram for `K0_MOK_ROUTE_HIST`; **M** — per-rank `C` (the config word
is per-process, so per-rank C is an env-per-rank change plus a manifest field); **M** —
**per-owner certificates / per-destination credits (K6)**, marked at `credit.cuh:104-106` as a
*documented non-goal until a skew harness exists* — **the K3 cell is exactly the harness that
unblocks it**, and its predetermined falsifier is already written (*must tie mode-12 within 1%
at std=0*).

**Falsification arms.** (i) **m17 `SCATTER_RR`** — order-only rotation, predicted to buy
**~1–1.5% under skew and nothing more** (measured 0.8467 → 0.8370 aggregate, 0.8559 → 0.8413
worst-layer). *If `SCATTER_RR` buys >5% in the new cell, the "bottleneck is popularity
concentration, not ordering" verdict is wrong.* (ii) **S2 carrier pool at C ∈ {32, 64} under
adversarial skew** — the P6c arm, testing the one claim that is untestable-not-refuted.

> **How this loses — and it is deliberately the campaign's most exposed prediction.** P9's
> falsifier is pre-interpreted: a measured serving gain **<10%** means `λ_serving ≪ λ_rig`,
> i.e. **the replay rig over-states skew because it replays skewed routing on 100% of rows
> while only 37.6% of serving rows are real** — which would invalidate every replay-based skew
> number in the ledger, including the 3.6× that motivates this whole region. Two further
> honesty flags ride it: the two per-call skew instruments **disagree by ~2×** and must not be
> mixed; and **ideal static redistribution still leaves per-call p95 at 3.05×** — placement is
> a static remedy against a per-call problem.

### 6.6 Region 6 — gfx942 portability

**S\* = per-source non-overlapping slots + owner-local reduce (store-class), NOT
producer-carried atomic accumulation.**

**Mechanism.** `θ-F16`: gfx942 remote atomics are **~3× slower than remote stores** (15 vs
44–46 GiB/s). Through D3, `1/R_a` triples ⇒ **`b*` falls from 32.8 MB to ≈11 MB ≈ 780 tokens**
⇒ store-towers is right almost everywhere. **This is a simplification for the PR**: gfx950 and
gfx942 now *agree* at deployment message size, since F5 already reversed the design doc's
atomic choice on gfx950 (atomics cap ~67 GB/s at 235 MB, diagnosed as the **dword issue rate**).
Through D5, congestion forms at lower depth ⇒ **`d* < 4`, plausibly 2** — and **depth 2 has
never been measured on any arch** because the `g` `0x300` selector is full.

**Starts from.** `include/cdna4/ops/group/distributed/credit.cuh:63-82` (already ships depths
{0,4,8,16,32} and both `throttled_accumulate_bf162` / `throttled_store_packet16`);
`m25_cdar_bench.hip` (transport is already argv there).

**Implement.** **S** — run the existing transport-only bench on a gfx942 node; **S** — one new
`vmcnt(2)` specialization plus widening the `g` `0x300` selector by one bit
(`moe_mps_adapter.cuh:400-410`); **M** — the owner-local-reduce carrier in the **DP** mega
(it exists today only in CDAR as `store_towers`).

**Falsification arm.** `atomic_accumulate` at every message size on gfx942, run in the same
invocation set.

> **How this loses.** P3b's stated falsifier: **atomics winning above 1,000 tokens on gfx942**.
> And a second, cheaper way to lose: if `d*` on gfx942 is **not** below 4, the "depth is a rate
> bound scaling with op drain time" mechanism is wrong and depth must be re-fitted per arch
> with no transferable rule — which turns an arch **card** into an arch **table**.

---

## 7. KILL CRITERIA AND THE FIRST-TWO-WEEKS BACKLOG

### 7.1 Kill criteria, per arm family

Each row names **one** number that retires the family. A family that is killed produces a
**published negative**, which `METRICS` R-S3 makes mandatory, not optional: *"a falsification at
the boundary is publishable on its own and must be published — the ledger's value is largely
its null results."*

| family | kill criterion | what ships instead |
|---|---|---|
| **Fused collective (S6/CDAR)** | at ρ ≥ 2.5, with filler and metering, **h < 20%** across {phased, fused, fused_wave, fused+shexp} | the **RCCL-hybrid** as the TP8 answer, plus the law *"fuse the collective iff `h > 1 − (1 − I_co/T_vendor)/β`"*, instantiated at 48.7% today and 22.8% at transport parity |
| **Consuming pool `C*(W)` (S4/A4)** | `C*` lands at **24 ± 4 in every cell** (φ, σ, T) | the law becomes *"`C*` is a constant of the **skeleton**, not a function of W"* — still a law, and it **retires the adaptive-C build (`M24_FILL_C`)** before it is written |
| **Carrier pool (S2)** | S2 loses monotonically **including in the adversarial-skew cell** | LAW-28 is upgraded from carrier-conditional to **regime-general on this hardware**, which also retires **A7 (per-destination credits)** — the highest-value *unbuilt* knob in the inventory |
| **Fill-awareness (M24 / G2)** | `C*` and `quota*` **do not move with φ** at fixed `T_pad` | M24 is demoted to a **work-deletion** feature; the fill story collapses to **tier-1 dummy-skip** and the ladder stops there |
| **Decode transport (D1)** | RCCL at 3.67 MB is **within 20%** of our 801 µs atomic point | decode is declared a **weight-prefetch problem**, and F3 (M20 slot pool pointed at decode) becomes the only decode arm |
| **Skew / placement (G4)** | measured serving gain **<10%** | the finding is that **the replay rig over-states skew**, and *every* replay-based skew number in the ledger is re-scoped. This is a first-class result, not a failure |
| **Wave specialization (A8)** | `fused_wave` hides **<10%** at the corrected ratio | the "independent `vmcnt` is the missing resource" hypothesis is dead and the wall must be named by the phase ledger |
| **Any zero-door arm** | it measures **inside ±1σ** of its control (the registered P11 prediction) | nothing — it was screened out for free. **Conversely**, any zero-door arm with a reproducible **>3%** effect means **there is a fifth door** and the occupancy-1 selection rule is incomplete — *report loudly* |
| **fp8-on-wire (F6/B9)** | X5 (the S8 scratch pathology) does not land, **or** the boundary moves **<−500 µs** | the transport is declared **issue-rate-bound, not byte-bound** at this size — consistent with the 58%-of-ceiling framing |
| **The model itself (H-Q)** | any **matched-Q / different-S\*** pair from FQ-1/2/3 | the deliverable degrades along §4.4's pre-committed ladder, ending at a regime map — which is weaker but honest |

### 7.2 The first five concrete actions

| # | action | size | node-h | why it is in the first five |
|---|---|---|---:|---|
| **1** | **P0-Z, the free triple.** (a) `np.unique(topk_ids, axis=0)` on an existing `skewhook_v2` capture → decides atlas H1 and confirms/kills LAW-42's degeneracy mechanism; (b) reconcile the two per-call skew instruments (**5.15× vs 2.61×**) → changes the modeled skew tax by ~2×; (c) compile one small-M decode tile body with `-Rpass-analysis=kernel-resource-usage` → decides whether the four-door rule survives into decode | **S** (~4 engineer-h) | **0.0** | Zero GPU time, and (c) is *"the single largest structural fork"* in the atlas |
| **2** | **B1 — the `k_inner` ratio sweep**, ρ ∈ {0.65, 1, 2, 3} × {compute, phased, fused, rccl}; **plus the RCCL decode reference** at `tokens` ∈ {32, 128, 256, 1024} in the same invocation set (θ-F13) | **S** (argv only) | **0.6** | The G25-1 PASS bar was **arithmetically unreachable** (ceiling 65% vs a 90% bar), so four falsifications and one RED gate are unscored; and the decode branch has **no basis** without θ-F13. Two blockers, one session |
| **3** | **E-B1/E-B2 — the device phase ledger + fabric-duty histogram** on `m25_boundary_bench.hip`, with the running-maxima **reset** baked in | **M** (~1 build-day) | **1.0** | *"One instrumented run replaces the next four guesses."* It removes the ambiguity in **both** named fudge factors (`I₀`, `κ_couple`) with one instrument, and it turns the duty-cycle reframe (C6) from an argument into an experiment. **Without the reset, the ledger lies** |
| **4** | **K5 — the LAW-37 2×2**: {depth bound on/off} × {slab certification on/off}, same session, `.text`-gated, K=4 | **S** | **0.8** | −613.5 µs and −573.3 µs close within 7% and may bound the **same** resource. If they are substitutes, **no waterfall in the paper is additive** — this changes M's *structure*, not its coefficients, and it is the cheapest of all the F-tests |
| **5** | **K4 — the `C × flush_rows` response curve in a second W-cell**, one HSACO, 18 configs × K=2 | **S** (zero code) | **1.9** | The literal answer to *"does our M15 producer/consumer work best for all sizes?"*, and the arm that turns a tuning constant into a law: *`C*` is a **derived quantity** of the cost model, never an input* |

**Started in parallel on day 1, because their lead time exceeds their run time:**
**X1** (make `K0P6_M24_FILL` compile — the *only* actuator for the φ axis, and φ is the
mediating quantity behind the flagship result) and **X2** (port the tgen M2 hole-sentinel fix
into the serving M15 body — without it every T ≠ 4096 arm on that body is **Class-C void**).

### 7.3 Two weeks, laid out

| day | node work | bench work |
|---|---|---|
| 1 | — | P0-Z (a)(b)(c); start X1 and X2; scaffold `manifests/` |
| 2 | Action 2 (B1 + θ-F13) | E-B1/E-B2 build |
| 3 | P0-A transport ladder (1.6 h) | E-B3 (transport argv, X4) |
| 4 | Action 3 (ledger runs) | Phase-1 registration with **banked** θ, committed |
| 5 | P0-C mega chassis (1.8 h) | X1 / X2 continue |
| 6–7 | P0-D serving calibration (1.9 h) | **re-register** predictions with **calibrated** θ; publish the diff |
| 8 | Action 4 (K5) + B3, B4 | analysis |
| 9 | Action 5 (K4) | analysis |
| 10 | K1 (if X2 landed) + B5, B6, B7, B8 | analysis |
| 11–12 | K3 skew ladder (2.5 h) | build the adversarial histogram; first GPU run of `route_replay.py` |
| 13 | K2 (if X1 landed) | — |
| 14 | Phase-3 fit; go/no-go on Phase 4 | law cards for whatever survived |

---

## 8. What this ladder refuses to promise

* **The shared expert cannot cover the 27% pool.** It is **1/8 of routed FLOPs** ⇒ ~32–34% of
  the exposed AR under TP8. Real, capped, and it must never be sold as the answer.
* **Cross-epoch pipelining has one built form and it lost** (+75.8 µs): *"the wait it escaped
  was mostly not on the critical path, and the parity state it added was."*
* **Dedicated communication CTAs are a carrier-conditional verdict, not a technique verdict** —
  dedication **won** at C=64 in the staged-push carrier (0.888× vs 0.894×) and lost
  monotonically in the producer-carried one — and the one regime that might rescue them (skew)
  has never been drivable in any kernel rig. K3 is the first cell that makes it drivable.
* **No decode claim in this ladder rests on a measurement**, because *this project has never
  measured decode*. Every decode row is a formula branch, and the campaign should say so on the
  slide.
* **Three claims stay retracted and must not reappear:** CDAR's *"−32% fewer bytes"* (the honest
  figure is **≈ −9.7%**, confirmed independently from two directions); the corpus **"1.19×"**
  padding multiplier (nz-filter artifact; the honest all-call aggregate is **1.446×**); and any
  reading of the c32p **−21.17%** as a *kernel* delta (the two arms differ in Group-H
  coordinates — 4096/128 vs 16384/2048 — so only the three-way decomposition is a valid
  reading).
* **`φ` has zero controlled points today.** Every sentence of the form *"fill explains the
  regime split"* is, until cell **K2** runs, an inference across two cells that differ in six
  variables at once — and the speaker notes already say so: *"they explain the regime split,
  but they do not prove fill is the only changing term."*
* **Multi-node, PP, SDMA, and energy are scoped out**, each with the reason stated (§1.6),
  rather than gestured at.

---

## Amendments from review (2026-08-18)

Superseded on the grid, the budget, the split and the gates by `../ABLATION_METHODOLOGY.md §7`.
Changes forced by BLOCKING items across all three reviews:

1. **A single cell registry replaces the ladder's ids** (`ABLATION_METHODOLOGY §7.2`). The
   §3.4 renumbering against `W_VECTOR §4.4` had silently deleted the granularity-decomposition cell
   and left the producer-order × certification law with no cell. Restored as **FQ-3** and
   **K5×order** (K5 becomes a 2×2×2: depth × certification × order).
2. **FQ-1/2/3 get cells, ranks and node-hours** (≈1.4 h, rank 4 in the ordering), plus **FQ-4**
   (burstiness) at zero marginal cost. They were cited as the test of the central claim at §0, §4.4
   and §7.1 and appeared in **zero cells and zero budget rows**.
3. **§4.1's split is voided and re-declared** (`ABLATION_METHODOLOGY §4.6`): "Phase 0 entirely" as
   FIT collided with Phase 0 measuring most of the held-out points, and `COST_MODEL §4.1`'s CAL-T
   fitted the grid its own HO-2/HO-3 held out. Rule: **any Phase-0-measured point is FIT**; Phase-0
   ladders are cut to endpoints; a PRIOR-VALIDATED (in-sample) row is added; the split is committed
   with a git sha before Phase 2.
4. **The budget is repriced as `pts × K × unit`** with the boundary unit declared as a range
   [1, 3] min and measured in the first ten minutes of session 1. §0.2 (14.0 for Phase 2, 16.5 vs
   §5.2's 16.7 for Phase 4a, headline 38.5) and §1.7 (rows summing 7.9 against a stated 6.0; P0-A's
   2.1 h stated as 1.6 h "with retries") are corrected. K4 is 18 configs × K=4. New total:
   **≈38–48 node-hours ⇒ 6–8 overnight sessions** after a 0.6–0.7 availability derate.
5. **New Phase-0 arms:** P0-A0 (rig repeatability — `σ_rig` has never been measured), a boundary
   cadence measurement, a fan-in / destination-concentration sweep, and a decode-size geometry sweep
   with each transport at its own optimum. **P0-B absorbs B1** (they were the same ρ sweep budgeted
   twice on opposite sides of the fit line).
6. **Gates gain teeth:** G0-a's definitional satisfaction and G0-c's or-declare branch are deleted;
   **G0-e** (clock/temperature parity, first-run discard, foreign-process check, randomized order with
   an anchor) is added; the scoring denominator is pre-registered with NOT-ADJUDICATED counted
   visibly; sign accuracy is computed only over predictions exceeding the cell's measured MDE; the
   ≤1-grid-step boundary-location bar is replaced by a bootstrap profile-CI criterion, and cells whose
   location CI spans >2 grid steps are declared unable-to-adjudicate **before** they run.
   The 26 registered predictions are a confirmatory family under Holm/BH correction.
7. **A §0.4 forwarding contract is binding** (whitelisted `K0_MPS_CFG` only, device-side descriptor
   echo, reference sanity ratio per row), and **actuation receipts are per-cell admission gates**.
   Per-rank `C` must ride the whitelisted path.
8. **Phase-2 node-hours are gated on ≥2 of {X1, X2, X3} landing**, with the concurrency ladder named
   as the ungated fallback φ actuator. K6 is re-scoped as a model-generalization cell after
   `COST_MODEL` D4a was corrected two-sided (no sign flip at k=3 under banked θ).
9. **Nine cells are cut to an extensions appendix** (B7, B8, B9/X5, K7, K8, E-A as a run, E-D, B3,
   B5, B6) and **E-F is promoted out of the conditional phase**; the c512p cell is labelled
   `R-L3 FAIL` until a 10,240-prompt confirmatory pair runs, and Phase-4 pair counts are
   rotation-balanced with the two banked pairs kept as prior evidence, not as pair count.
10. **§7.3's calendar is re-published**: `predictions.jsonl` with banked θ commits on **day 1** (it is
    a `git add`, and §0.1's own gate forbids a Phase-2 cell running before its prediction is
    registered — day 2 ran B1 against a day-4 commit). Phases 0–2 occupy the two weeks at 0.65
    availability; Phase 4 gets its own block. A paired contrast spanning a session boundary is
    `voided_by: cross-session`; the runner must be idempotent and manifest-resumable.
11. **X6 is added to §0.3**: the manifest/analysis pipeline and the client v4 patch
    (`sent_s/issued_s/scheduled_s` are null, which blocks open-loop windowing).

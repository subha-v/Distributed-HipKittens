# ABLATION_METHODOLOGY — one base megakernel, a library of schedule modules, and a cost model that decides which to attach

**Date:** 2026-08-18 · **Audience:** Simran + the industry reviewers who saw draft8 ·
**Status:** campaign design, pre-registration pending. No number below is new; every number
carries its source. Predictions are labelled **REGISTERED PRIOR** and were written before the
runs they adjudicate.

**Relationship to the detailed design set.** This document is self-contained and is the
authority on the contribution, the canonical `Q` table, the cell registry, the budget, and the
fit/hold-out split. It **references** rather than duplicates:
`design/{W_VECTOR,S_VECTOR,COST_MODEL,METRICS_PROTOCOL,OVERLAP_ATLAS,EXPERIMENT_LADDER}.md`,
`grounding/{KNOB_INVENTORY,BANKED_LAWS,WORKLOAD_EVIDENCE,TP8_STATE}.md`, and the three
adversarial reviews in `review/`. Where this document and a sibling disagree, **this document
wins**, and the sibling carries an "Amendments from review" section saying so.

**Evidence tags:** `MEASURED` (in-repo artifact, cited) · `DERIVED` (arithmetic here over cited
artifacts, inputs shown) · `REGISTERED PRIOR` · `SPECULATION`. An untagged claim is a
definition, not a measurement.

---

## 0. Executive summary — the experts' comments, answered one at a time

> **"does our m15 producer/consumer work best for all sizes?"** (`BRIEF.md:21`)

No, and we can now say exactly when it does. M15's producing/consuming split is not one thing:
it is three modules — consumer-major producer order, slab certificates, and a *consuming* CTA
pool — and the pool only pays when the first two have created a certified prefix for it to
consume. The same reservation mechanism measures **+332…+340 µs when it carries payload**,
**+8–9 µs per CTA when it idles**, and **−93 µs (C 8→16) / −444 µs (C 16→24) when it consumes**
(`OVERLAP_ABSTRACTIONS.md:26-30`, MEASURED). Cell **K4** re-sweeps `C × flush_rows` in a second
workload cell and turns the shipped constant `C = 24–28` into a derived quantity `C* =
min(C_sat, C_tax)`; the model already predicts `C_sat = 337 MB / (6.1 GB/s × 2.66 ms) = 20.8
CTAs` against the measured knee of 24 — **one grid step, with no fitting** (`COST_MODEL §1.7`,
DERIVED).

> **"It's powerful premature to say that producer/consumer may not be best. Might be."**
> (`BRIEF.md:22`)

Agreed, and the campaign is built so the negative can lose. The dedicated-carrier verdict is
**carrier-conditional, not a technique verdict**: dedication *won* at C=64 inside the
staged-push carrier (0.888× vs homogeneous 0.894×, `aug10/experiments/LESSONS.md:669-676`) and
lost monotonically inside the producer-carried one (`aug11/LESSONS.md:287-301`). The one regime
that could rescue it — skew — was never *settable* on the kernel rig (`K0_SYNTH_ROUTE` rejected;
reachable destination-load CV **0.0034–0.0086** vs COMET's 0.032, LAW-62b), so it is
**untestable, not refuted**. Cell **K3** builds an adversarial synthetic histogram that makes
that axis settable for the first time, and cell **B.residency** tests the structural variable
the whole negative rests on (256 CTAs, occupancy 1) by sweeping `N_CTA` and by running the
carrier as a *second concurrent kernel*.

> **"List of what we'd like to teach people about"** (`BRIEF.md:23`)

§9. Ten laws, each with the cell that adjudicates it and a `verdict` column, published as law
cards with a Q-inequality, a θ, a predicted-vs-measured line, a failure scope, and the one
measurement that would retire it (`METRICS_PROTOCOL §5`). The card titles are pre-registered in
Phase 1 next to `predictions.jsonl`, so the teaching list is a committed artifact rather than a
hoped-for byproduct.

> **"What the pull request in HK looks like"** (`BRIEF.md:24`)

§10, and it is the same object as the contribution: the PR is **the module library plus the
composition interface**. Three API changes are already derivable and each is decided by a named
cell — `credit.cuh` needs a `co` scope note and depth-2 (which is an *encoding* change, because
the `g` `0x300` depth selector is a full 2-bit field, `moe_mps_adapter.cuh:400-410`);
`slab.cuh` conflates signal count with transfer geometry in one parameter and should take two;
`order.cuh` selection should be a function of top-k. `arms.jsonl` gains `primitive_touched` and
`api_implication` fields so every arm carries its PR consequence.

> **"Super curious about the findings before we even look at other workloads"** (`BRIEF.md:25`)

Four, all obtained from arithmetic over banked artifacts, all in §4: (i) the G25-1 PASS bar was
**arithmetically unreachable** — at the rig's own operating point `ρ = 0.650–0.664`, so maximum
hidable fraction is 65% against a ≥90% bar; every G25-1 verdict including four falsifications was
scored against an impossible criterion (`W_VECTOR §2.2 Q3`, DERIVED); (ii) the honest break-even
is `h* = 1 − (1 − I_co/T_vendor)/β = 48.7%`, i.e. the bar was **1.85× too high** and the arm is
23× short of the real one; (iii) the skew multiplier `T(σ) = T(1)[(1−λ) + λσ]` fits **one**
parameter and predicts a held-out point to **+0.47%** (λ_M15 = 0.6266; 22,582 predicted vs
22,475 measured, `M18_REPLICATION_RESULTS.md:24-31`); (iv) the flagship topology law is measured
on a **diagonal, not a 2×2** — TP8-native at C=512 has never been run.

> **"We need to have a cost model and enumerate what are the properties that control the
> workload"** (`BRIEF.md:26`)

§2 enumerates W (21 axes, with control class and settability) and collapses it to **eight**
mediating scalars `Q`, frozen in arity at Phase 1. The reviews caught two incompatible `Q`
vectors in circulation and, worse, coordinates that were *schedule-dependent* — which makes
`argmin_S M(Q(W), R(S))` ill-posed. Fixed: `Q` now contains only quantities computable from
`(W, θ)` before a kernel exists, evaluated at a declared canonical schedule; readiness
parallelism, protocol count, scatter width, slack and burstiness are demoted to the schedule
response vector `R`; and co-scheduling density, seal coverage and `W_pad` are moved into a
separate **validity-gate vector `V`** that can veto a comparison but never enters the model.

> **"As we toggle the list of workload [properties], what are the properties of the kernel that
> changes"** (`BRIEF.md:27`)

§8 answers this in the form the north star requires: each workload region gets a **module
composition diff** from the base kernel — *attach X, detach Y* — with the `Q`-inequality that
triggers the diff and a registered prediction that can lose. Example: crossing `T_eff` below
≈1,300 real rows/rank detaches the entire fused collective (`α` exceeds ≈0.41 and the vendor
path wins); crossing `co` from 1 to 0 detaches the depth bound (it measures **flat** in pure
transport); crossing `b` below ≈32.8 MB swaps `M-TOWERS` for `M-RMW`.

> **"Workload setting to schedule design — and can we do that in a cost model?"**
> (`BRIEF.md:28`)

Yes, with two named and bounded fudge factors (`I₀ ≤ 1,159 µs`, `κ_couple ≤ 55 µs/CTA`) that one
instrument removes, and with the failure mode pre-committed: if the sufficiency claim dies, the
deliverable degrades to a **regime map plus the falsified ledger** (§7.5). Three matched-`Q`
falsifiers now have cells and node-hours (they had zero in the draft the reviewers read).

> **Risk: "Very expansive, unclear what the core contribution might be"** (`BRIEF.md:29-30`)

Taken as the binding constraint. The contribution is now **one sentence** (§1), repeated
verbatim in every sibling document. The expansive grid is cut by ~45%: nine cells removed,
including the two whose own registered predictions said *our thing loses*. The core campaign is
**≈22–28 node-hours at the boundary instrument** (Phases 0–3, the part that actually defends the
claim) plus **≈16–20 node-hours of e2e sizing** — and the draft's "≈38.5 h / four overnights"
is replaced with the honest **≈38–48 node-hours over 6–8 overnight sessions** after a 0.6–0.7
node-availability derate.

> **"People keep posting TPS without batch size or concurrency, input and output lengths, or
> even saying whether it means per-request decode speed or aggregate throughput. On its own, the
> number means almost nothing."** (`BRIEF.md:31-33`)

We can quantify "almost nothing" from our own artifacts: the same run, same arm, same second
reports aggregate output **83.663 tok/s** and per-request decode **1.321 tok/s** — a **63.3×**
spread between two numbers a reader would both call "tok/s" (`b3rev2_pair_01_2_m15.json`,
DERIVED). §5 fixes one PRIMARY metric per workload class, bans bare "TPS" as a protocol
violation, and requires a ten-block workload tuple on every number. The discipline is
load-bearing, not pedantic: at C=32 the arm you should deploy **flips at a TTFT SLO of
≈3.5–4.0 s**, and at c512p `goodput_in(60000,1000)` is **42,835.7 vs 1,471.5 tok/s (29×)**
against a headline delta of **+6.60%**.

---

## 1. The contribution, in one sentence — and the minimum campaign that defends it

> **CONTRIBUTION.** A distributed MoE megakernel should be built as **one base kernel plus a
> library of attachable, removable schedule modules**, and the cost model `M(Q(W), R(S); θ)` is
> the **composition policy**: given eight workload-derived scalars `Q`, it outputs *which modules
> to attach and which to detach*, each decision being a threshold inequality in one coordinate
> of `Q`.

That sentence is the paper. Everything else is instantiation. It is falsifiable in three ways
(§4.6), it is cut down to a two-block campaign (below), and it converts "we swept hundreds of
kernels" into "we found the eight numbers that decide a schedule, we can compute them from a
serving config with a spreadsheet, and here is the module each number switches on or off."

**The pre-committed fallback, stated up front so it cannot be mistaken for the headline.** If
the composition policy fails to generalise across topology, the deliverable degrades to a
**two-law regime map** — DP/EP governed by `T_eff`, TP8+EP governed by the exposed-collective
fraction — plus the falsified ledger. That is weaker but honest, it is written into §7.5's
degradation ladder, and it is *not* what we are claiming today.

**The five laws the minimum campaign defends** (each is a genuine `W → S` sign or threshold, has
a settable actuator or a named unblocker, maps to a teaching-list law in §9, and maps to a DHK
primitive in §10):

| # | law (module diff it drives) | mediator | cells | node-h |
|---|---|---|---|---:|
| **L1** | Fuse at all iff the fixed-cost share `α < ≈0.41`, i.e. `T_eff > T* ≈ 1,300–1,800` — below it, **detach the whole fused collective** | `T_eff` | K1 (decode-shaped T folded in) | 0.8 |
| **L2** | **Attach `M-DEPTH` iff `co = 1`**: injection depth is congestion control against co-resident compute, not against the fabric | `co` | B4 + **FQ-2** | 1.3 |
| **L3** | Choose `M-TOWERS` vs `M-RMW` by the arch's atomic:store ratio **at the deployment message size**; `b* ≈ 32.8 MB ≈ 2,289 tokens` | `λ` | B2 (+ RCCL decode reference) | 2.8 |
| **L4** | A reserved CTA costs 8–9 µs unless it *consumes* a certified prefix; `C*` is derived from `(Σ, P, t50)`, never carried | `Σ, P` (via `R`) | K4 + K5×order + **B.residency** | 6.1 |
| **L5** | The DP topology's per-real-token cost inflates by `1/φ` and TP8's does not — the C=32/C=512 flip is a **topology tax mediated by fill**, not kernel quality | `φ` (→ `T_eff`) | K2 + **FQ-1** + E-F | 3.0 + e2e |

Plus **FQ-1/2/3/4**, the matched-`Q` sufficiency falsifiers, which are the test of the
contribution itself and which the reviewers correctly noted had **zero budgeted experiments** in
the draft.

**Everything in the expansive version that is not on this list is in Appendix A as an
extension**, including two cells whose own registered predictions were "our thing loses"
(the decode e2e cell and the C=32 arm-level e2e re-run).

---

## 2. W — the workload state vector — and `Q`, the mediating layer

Condensed from `design/W_VECTOR.md`; that document is the authority on the 21 axes and their
provenance traps.

### 2.1 W, by control class

| class | axes | campaign consequence |
|---|---|---|
| **MODEL-FIXED** | routed experts 256 (32/rank at EP8), **top-k 8**, hidden 7,168 ⇒ **14,336 B/token bf16**, `d_ff` 2,048, 1 shared expert, 61 layers (3 dense + 58 MoE) | not ablation axes for this checkpoint; they are the axes along which results generalise to other models, and they appear symbolically in every `Q` formula |
| **HW-FIXED** | gfx950 (gfx942 microbench only), xGMI intra-node, 8 XCD × 32 CU, **occupancy exactly 1 block/CU** (LAW-12), world 8 | one point today; multi-node has **zero evidence** and is scoped out |
| **OPERATOR** | topology, `max_num_batched_tokens` (ours 4096, vendor 16384), `max_num_seqs`, prefix caching, expert placement, EPLB | cheapest levers; `max_num_batched_tokens` is the largest-reach lever in the project and has **never been swept** |
| **LOAD** | concurrency, ISL, OSL, arrival process | settable by the driver; measured at **two points only**, C=32 and C=512, 16× apart |
| **EMERGENT** | fill `φ`, dummy rate, co-scheduling density `κ_sched`, per-call skew | *not settable at all* — and they matter most. This is why the campaign needs synthetic actuators (M24 `NORIG_TABLE`, route replay, adversarial histograms) |

Two provenance traps ride every W claim and are binding: `K0_MAXTOK == T` in the harness, so
**every T-sweep is a batch-size sweep, not a fill sweep** (`REPORT.md:435-438`); and vLLM pads
with token id 0 ⇒ identical hidden ⇒ identical top-8 ⇒ **2.66× rank-load skew from padding
alone** (LAW-42), so fill *generates* skew and the two axes are not orthogonal.

### 2.2 `Q` — the canonical eight (frozen in Phase 1)

Two incompatible `Q` vectors were in circulation, and several coordinates were derived *from the
schedule*, which makes the central formulation ill-posed. Resolved: **`Q` contains only
quantities computable from `(W, θ)` before any kernel exists**, each evaluated at a declared
canonical schedule (base kernel + `M-DEPTH` + `M-CERT` + `M-POOL-CONSUME` at C=24).

| # | symbol | definition | computed from | measured span in-repo |
|---|---|---|---|---|
| **Q1** | `T_eff = φ·T` | real rows per rank per step | fill receipt × bucket size | 1,539 (c32p) … 4,035 (c512p) |
| **Q2** | `k` | top-k | model card | 8 (the code hard-rejects `TOPK != 8`) |
| **Q3** | `B_bnd` | bytes crossing the boundary per token per layer | routing combinatorics × `d_model` × topology | DP 130–163 KB × 58 layers vs **TP8 392 KB × 61** ⇒ TP8 moves **2.41–3.02×** more fabric bytes (DERIVED) |
| **Q4** | `C_phase` | compute time per phase at the canonical schedule | GEMM shapes ÷ `θ_flop(limiter class)` | M6 2,453.3 µs, M7 2,701.8 µs at T=4096 (LAW-65 stamp) |
| **Q5** | `ρ = C_phase / X` | hideability ratio; max hiding `= min(ρ,1)` | Q3, Q4 | **0.650–0.664** in the G25-1 rig vs **2.6–3.2** at TP8 deployment |
| **Q6** | `co ∈ {0,1}` | is an MFMA body co-resident with the carrier? | phase × fusion boundary of the **base** kernel | both values measured, and they separate LAW-8 from LAW-20 |
| **Q7** | `σ` | max-rank load ratio (routing + placement + padding) | route corpus × placement map | 5.088× contiguous vs **1.085×** round-robin; per-call **unreconciled: 5.15× vs 2.61×** |
| **Q8** | `λ_msg = m / knee` | message size relative to the transport knee | `T` × row bytes × topology | 3.67 MB (decode-shaped) … 235 MB (prefill boundary) |

**Two provisional coordinates, declared not hidden.** `crit persistence` (cross-layer
correlation of the argmax rank — a scalar `σ` cannot distinguish "rank 0 hot in every layer"
from "a different rank hot in each layer", which is exactly what round-robin placement changes)
and **burstiness** (peak/mean fabric injection over the layer — the atlas's duty-cycle finding
has no coordinate anywhere). Both are **registered as the first two mediators to add if H-Q
fails**, and burstiness gets its own cheap falsifier (FQ-4) because the instrument that measures
it is being built anyway.

**`R` — schedule response quantities** (what the modules move, and what the model consumes on
the schedule side): readiness parallelism `P`, unblockable fraction `u(t)`, protocol op count
`A`, scatter width `L`, injection depth `d`, reserved consumers `C`, quota, slack map `Σ`,
CTA-throughput-bound fraction `φ_p`, injection burstiness.

**`V` — the validity-gate vector, explicitly banned from `Q`**: co-scheduling density
`κ_sched = 1.1922` (worth **+8.4% wall — the entire measured c32p gap** on its own, LAW-40), seal
coverage `sealed/in_bucket`, `W_pad`, and the cell's measured variance envelope. These can veto a
comparison; they never enter `M`. Conflating them with `Q` is how a scheduler effect gets read as
a kernel effect.

---

## 3. S — the base kernel and the module library

Condensed from `design/S_VECTOR.md` (skeletons, knobs, constraint matrix, manifest schema).
The reorganisation below is the north star made concrete: **S is not a set of disjoint
skeletons; it is one base plus attachments.**

### 3.1 The base

**BASE** = homogeneous megakernel: 256 CTAs, `__launch_bounds__(256,1)`, **occupancy 1 block/CU**
forced independently by 256 VGPR + 256 AGPR *and* 155,428 B LDS (LAW-12), grid barriers between
phases, no cross-rank protocol. MEASURED at **6,908.8 µs = 0.8958×** of production at the 4,096
tok/rank prefill boundary (`BRIEF.md:50`). It is the only arm that measures "megakernel, no
cross-rank protocol", which is the baseline every module's delta is stated against.

**The gate every module must pass** (the occupancy-1 selection rule,
`OVERLAP_ABSTRACTIONS.md:42-60`): at 1 block/CU there are no spare waves and `vmcnt` is one
in-order counter per wave, so phase partitioning is work-conserving and overlap pays through
exactly **four doors** — (a) a latency-bound consumer into a compute shadow, (b) deleting
protocol work, (c) an op-class swap on the fabric, (d) bounding injection. **A module that
cannot name its door is predicted ≈0 and is not built.** Retrospectively this screen is 11/11 on
the falsified ledger (`COST_MODEL §D11`).

### 3.2 The module library

| module | what it attaches | door | attach condition (`Q`-inequality) | banked effect | DHK primitive |
|---|---|---|---|---|---|
| **M-ORDER** | consumer-major (nc-major) producer order | (a), enabler | always with `M-CERT`; never alone | **≈0 alone**; **−144 µs with certification** | `order.cuh:50-95` |
| **M-CERT** | slab certificates: one epoch word per (rank, slab), replacing ~926k per-row protocol ops with 2×8 words | (b) | `(Φ_slab−Φ_natural)·V(k)/R_serial + κ_couple·ΔC_sat > ι·ΔA(k)` — see §4.5 | **−573.3 µs** of protocol work | `slab.cuh:62-97` |
| **M-DEPTH** | depth-bounded producer-carried delivery (`vmcnt(N)` specializations, depths {0,4,8,16,32}) | (d) | **`co = 1`** | **−615.0 µs** at `co=1` (7,110.8 → 6,495.8, t = −80.2); **flat** at `co=0` | `credit.cuh:63-82` |
| **M-POOL-CONSUME** | `C` reserved CTAs consuming the certified prefix in the producer's shadow | (a) | `V_cert > 0`, size `C* = min(C_sat, C_tax)` | −93 µs (C 8→16), **−444 µs** (C 16→24); C=32 **hung 1 of 2 runs** | `roles.cuh` |
| **M-POOL-CARRY** | reserved CTAs that *move payload* | none | `value(carry) > 0` — true only when the pool **is** the transport | **+332…+340 µs**; +831 µs of added traffic from the role itself at C=64 | `roles.cuh` |
| **M-TOWERS** | owner-gathered store towers + owner-local reduce | (c) | `b > b* ≈ 32.8 MB` | **117.2 GB/s** @256-row vs atomics' 67.1 at 235 MB | `packet.cuh`, `peer.cuh` |
| **M-RMW** | producer-epilogue packed-bf16 remote accumulate | (c) | `b < b*`, or arch atomic:store ≥ ~0.9 | **801 µs vs towers' 1,017 µs** at 3.67 MB (n=1, confounded — §4.5) | `credit.cuh` |
| **M-SHEXP** | shared expert as routing-independent filler | (a) | `Σ ≥ demand`; bounded by construction at **1/8 of routed FLOPs** | covers **~32–34%** of the TP8 exposed AR — a third of the prize | — (new) |
| **M-FILL** | `T_eff` substitution at five sites (M24) | (b) | `φ < 1` **and** topology = DP | **never compiled** (blocker X1) | — (new) |
| **M-PREFETCH** | cross-layer expert-weight prefetch (M20 slot pool) | (a) | decode, or `Σ > 0` in attention | *"stream order **is** the certification; no flags, no tags"* | `lifetime.cuh` |
| **M-PLACE** | owner round-robin / EPLB placement (host-side) | — | `σ > ~2` | σ **5.088× → 1.085×**; worst-rank layer 5.593× → 1.296× | outside the kernel |
| **M-SEAL** | ragged graph-capture seal (M23) | — | DP serving always | coverage **2% → 100% of in-bucket** (~50×) | outside the kernel |

### 3.3 What is **not** expressible as a module on this base — and why that is a finding

Two structures cannot be attached or removed; they replace the base:

* **S6 / CDAR (counter-dataflow TP8):** the collective *never exists as a phase*. ERS rides the
  producing GEMM's epilogue and MAG rides the consuming GEMM's ramp. You cannot express "no
  collective phase" as an attachment to a base that has one. Honest status: **RED** — fused
  3,606 µs vs GEMM→RCCL 2,909 µs, and our transport is **1.51×** slower than RCCL.
* **S7 / T3 (zero grid barriers):** every dependency becomes a flat HBM counter. The all-256-CTA
  grid barrier is a *residency contract*; removing it changes drift realization from three points
  per launch to one per layer. Design only; the 1,280–1,330 ms figure is an explicit projection.

**This is a claim the campaign must confirm or refute, not an assumption.** The composability
boundary is exactly the question "is a dedicated carrier a *module* on this base, or does it need
a different base?" — and the draft answered it with a CTA-granular negative (+831 µs) while
leaving grid geometry unswept, calling it "the axiom the selection rule rests on". Cell
**B.residency** (`N_CTA ∈ {160,192,224,256} × {comm in-kernel, comm as a second concurrent
kernel}`, ≈0.4–0.8 h) sweeps the axiom and pins LAW-28's scope to *"CTA-granular dedication
inside an all-CU, grid-barriered, occupancy-1 CDNA megakernel."*

### 3.4 Compositionality risk — modules are not additive, and the ladder tests it

Composability is a **hypothesis**, and the banked ledger already contains three counter-examples:

1. **Non-additive, measured.** Carrier relocation and the injection bound are not separable:
   relocation alone is a **+210.6 µs regression**; with the bound the joint cell is **−615.0 µs**
   — the throttle alone is **1.52× the entire gap** (LAW-21). Any waterfall summing their
   coefficients is arithmetically wrong.
2. **Possible substitutes, never co-measured.** The injection bound (−613.5 µs) and coarse
   readiness (−573.3 µs) close within 7% and plausibly bound the *same* in-flight-remote-write
   resource (LAW-37). If they are substitutes, `M`'s protocol and interference terms must share
   one resource variable rather than contribute independently — that is a **structural** change,
   not a coefficient change.
3. **Conditional value.** `M-DEPTH` is worth −615 µs at `co = 1` and **exactly nothing** at
   `co = 0`. A module's value depends on which other modules are attached and on the phase.
4. **Codegen coupling — the deepest risk.** Making mode-14 code merely *reachable* cost the
   mode-12 ratchet **+726.9 µs with mode-12's source untouched** (LAW-32), because the register
   allocator installed a stronger involuntary throttle. **A manifest of composable modules will
   silently re-time arms whose modules it never touched.** Mitigation is non-negotiable: `.text`
   sha256 per build, a flag-OFF build claimed inert must be `.text`-identical, a flag-ON build
   **must** differ, and every depth arm carries the ISA gate (literal `vmcnt(N)` at expected
   sites, zero scratch ops within 24 instructions of any remote atomic, issue-run distribution
   ~282 atomics / ~12 runs / mean 23.5 in flight).

**Consequences carried into the design:** the ladder contains **pairwise-interaction cells that
test additivity rather than assume it** — K5 is a **2×2×2** (`depth × certification × order`),
not three ladders; FQ-2 does both halves of the `co` contrast on one rig; and `M` subtracts the
overlap credit **once at layer scope, never phase-by-phase** (§4.1). Where `M` carries no
interaction term, it says so.

---

## 4. `M(Q, R; θ)` — structure, constants, decision boundaries, validation

Condensed from `design/COST_MODEL.md` as amended by the mechanism review.

### 4.1 Structure

```
T_step   = Σ_layers T_layer + H(S)                                   H bounded, |δ| ≲ 15 ms/step (prefill only)
T_layer  = MAX_ranks T_layer^(r)                                     … LAW-52: the critical rank sets the pace
T_layer^(r) = Σ_phases [ C_p + X_p + Q_p + I_p ] − O(S)              … O subtracted ONCE, at layer scope
C_p(n)   = C_p(N)·[(1−φ_p) + φ_p·N/n],  n = N − C                    … capacity elasticity, φ_M7 = 0.84, φ_M6 ≈ 0
X_p      = L(op, world, n_slabs, bps) + b_p / BW_eff                 … AMENDED: affine, with a latency intercept
BW_eff   = min( egress, links_used(π, order)·BW_link, ingress_dst/share ) · η_layout · η_par(blocks, bytes/block)
                                                                      … AMENDED: incast-aware; η_size and η_ctas collapsed
Q_p      = ι·A_p(L) + n_bar·c_bar + n_cert·c_cert + W_spin           … ι = 0.438 µs per 1,000 protocol atomics
I_p      = co·[ ι·A_p(L)·ξ_int + J(d) + I₀(skeleton) ]               … J(d)=0 for d ≤ 8; +470 µs for d ≥ 16
T_layer(σ) = T_layer(1)·[(1−λ) + λ·σ]                                … λ_M15 = 0.6266, λ_prod = 0.5322
```

**Four amendments the mechanism review forced, all landed above.** (i) The wire term had **no
incast / destination-concentration factor** and assumed the 7-link plateau even under 51.9%
concentration on rank 0 — so it structurally predicted ≈0 for exactly the class of remedy the
vendor measured at **65%** (MORI's push-combine destination interleaving, 822 → 498 µs).
(ii) It had **no fixed/latency intercept**, so at the one decode point we own (3.67 MB) it
predicted **31 µs against a measured 801 µs — 25× wrong** in the regime where the decode go/no-go
is decided; `L(·)` is now explicit and is calibrated for free from P0-A's existing points.
(iii) `η_size(m)` and `η_ctas(n_CTA)` were multiplied as independent factors although they are
the same saturating-parallelism measurement (`n_slabs × bps` sets the transport grid occupancy).
(iv) **Decode predictions become sign-only** — `M` has no expert-weight-stream term although the
atlas derives **1.34 GB/rank/layer** of weight streaming at decode, an order of magnitude above
the 3.67 MB collective.

### 4.2 `θ` — the constants, with class and status

`M` = measured & banked; `D` = derived here; `H` = hole with the microbench that closes it.

| θ | value | cls | consumer |
|---|---|---|---|
| per-link achievable BW | **56.9–57.3 GB/s** (74% of 76.8 nominal) — never use 76.8 | M | `X_p` |
| link knee | **8 CTAs** single-link, **32** rr7; C=8→64 buys **+2.4%** | M | pusher count is not a bandwidth knob above ~8–32 |
| node egress ceiling | 349–355 GB/s, flagged **"unreproduced; treat as unknown"** | **H** | every "% of ceiling" framing; P0-A |
| atomic:store, small msg | 52.8 vs 54.9 GB/s (0.962) | M | `η_op` |
| atomic cap, bulk | **67.1 GB/s** (diagnosed as dword issue rate) vs towers **117.2** @256-row | M | L3 |
| layout penalty | coalesced **52.8 → scattered 4.1 GB/s = 13×** | M | *layout is a schedule decision* |
| gfx942 atomic:store | **~3× slower** (15 vs 44–46 GiB/s) | M | L3 on gfx942 |
| RCCL AR @235 MB | **1,470 µs = 280 GB/s = 79%** of the (unreproduced) ceiling | M | `β` |
| our CDAR @235 MB | **2,214 µs ⇒ β = 1.506×** | M | `h*` |
| RCCL at decode sizes | — | **H** | **entire decode go/no-go**; one argv |
| `ι` protocol cost | **0.438 µs per 1,000 atomics**; census ~926,000/rank/epoch | D/M | `Q_p` |
| `ψ` line-contention exponent | **≥1** (32 lanes → 2 lines was **≥10× slower**) | **H** | `L` should be **maximised** — the opposite of the load rule |
| `J_hi` depth penalty | **+470 µs** at d ≥ 16; `d* = 8` is the cliff edge; **d=2 never measured on any arch** | M | L2 |
| consumer rate `ρ` | **6.1 GB/s per CTA** | M | `C_sat` |
| HBM knee / plateau | knee **128 CTAs**, **4,151.9 GB/s = 51.9%** of 8,000 (use this, never a back-derived peak) | M | decode branch |
| `I₀` co-residency floor | **≤1,159 µs**, **64% of the mechanism unidentified** | **H (bounded)** | fudge factor #1 |
| `κ_couple` producer coupling | **≤55 µs/CTA** for `C ≤ C_sat`; identically 0 without a certified prefix | **H (bounded)** | fudge factor #2 |
| `λ` row-proportional fraction | **0.6266** (M15), **0.5322** (production) | D | skew multiplier |
| `σ` per-call | **UNRECONCILED: 5.15× vs 2.61×** — carried as an interval, never mixed | M | every σ prediction is a band |

**Both fudge factors are removed by one instrument** — the device phase ledger (E-B1) that
separates produce-span from commit-span and M7's drain-wait from its MFMA span. The review
caught a real defect in *where* they are calibrated: the boundary rig's co-resident "compute" is
an MFMA burst with a **fixed** MFMA:load ratio and a small L2-resident working set, so it
structurally cannot produce a quantity whose declared scaling is "proportional to the co-resident
phase's memory intensity" (the real phase streams ~470 MB at ~296 GB/s). Fix, landed:
`bytes_per_mfma` becomes an argv so intensity and duration are separable axes, `I₀` is calibrated
on the **mega chassis** (real GEMM), and the boundary rig's number is scoped as TP8-specific
`I_co`.

### 4.3 The decision boundaries, as module diffs

| id | inequality | instantiated with banked θ | module diff it triggers |
|---|---|---|---|
| **D1** | `T_eff > T* = ΔF/Δv` | `886.7 / 0.6764 = **1,311 tok/rank**`, band **[1,300, 1,800]** | below `T*`: **detach everything**, run the vendor path |
| **D2** | `h > h* = 1 − (1 − I_co/T_vendor)/β` | `1 − (1−335/1470)/1.506 = **48.7%**`; at β=1, **22.8%** | fuse the collective, or leave it to RCCL |
| **D3** | `b > b* = (L_t − L_a)/(1/R_a − 1/R_t)` | `243/7.410 = **32.8 MB ≈ 2,289 tokens**`; on gfx942 **≈11 MB ≈ 780 tokens** | attach `M-TOWERS` vs `M-RMW` |
| **D4** | `(Φ_slab−Φ_natural)·V(k)/R_serial + κ_couple·ΔC_sat > ι·ΔA(k)` | see §4.5 — **corrected, two-sided** | attach `M-CERT` + `M-ORDER` |
| **D5** | `d* = ∞ if co=0; d* = 8 if co=1` | measured both ways | attach `M-DEPTH` |
| **D6** | `value(job)·C > C_p(N)·φ_p·[N/(N−C) − 1]` | carry = 0, idle = 0, consume > 0 iff `V_cert > 0` | `M-POOL-CONSUME` vs `M-POOL-CARRY` |
| **D7** | `C* = min(C_sat, C_tax)`, `C_sat = V_cert/(ρ·τ_shadow)` | **20.8 predicted vs 24 measured** | size `M-POOL-CONSUME` |
| **D9** | `gain = 1/[(1−f) + f·M(σ_RR)/M(σ_lin)]` | **+27% to +46%** absolute at c32p, **<3%** ratio change | attach `M-PLACE` (host-side) |
| **D11** | zero doors ⇒ predicted ≈0 | **11/11 retrospective** | reject before building |

### 4.4 Registered-prediction protocol

Predictions are committed to `manifests/predictions.jsonl` with a git sha **before** the cells
run — twice: once now with banked θ, once after Phase 0 with calibrated θ, **and the diff is
published**. A prediction whose *sign* flips between the two registrations means the boundary was
inside θ's uncertainty all along; the cell is re-designed to bracket it before it runs. Three
process fixes from the reviews are binding: (i) `Q`'s **arity is frozen** at registration, so a
miss cannot be answered with "add a coordinate"; (ii) the **scoring denominator is
pre-registered** — NOT-ADJUDICATED cells are counted visibly, and sign accuracy is computed only
over predictions whose effect exceeds the cell's own measured MDE; (iii) the 26 registered
predictions are a **confirmatory family under Holm/BH correction**; everything else is labelled
exploratory in every table (with ~60 pairwise contrasts, ~3 spurious "significant" results are
expected under a global null).

### 4.5 Two registered predictions corrected *before* measurement

**The top-k sign flip is retracted as stated.** D4a compared `Φ_slab(2) = 0.25` against
`Φ_natural(k) = 1/(k+1)` and concluded that certification pays iff `k > 3` — advertised as "the
cleanest sign-flip law the campaign owns". That inequality is **one-sided**: it omits `ι·ΔA`, the
protocol cost of the fine path, although `ι = 0.438 µs/1,000 ops` is in the same document. With
the model's own θ at k=2: the fine path buys `V(2)·(0.333−0.250) = 14 MB ⇒ ~9 µs` of extra
consumable shadow (≈49 µs even inflated by the full 5.4× `κ_couple`), against `ι·A(2) ≈ 101 µs`
of protocol cost. **Slab certification still wins at k=2 by 2–10×, and there is no sign flip at
k=3.** The vendor evidence cited in support (per-record flags buy nothing; an LL128-style path is
44%/177% slower) is a *protocol-cost-dominance* result and reads the same way. Re-registered:
`M-CERT` wins at every `k ≥ 2` under banked θ; the campaign measures where — if anywhere — the
flip actually is, and X3 (the `TOPK != 8` unlock) is **re-ranked down** accordingly, from the
flagship law to a model-generalization cell.

**The decode K1 reversal is labelled MEASURED-once-confounded.** The single point behind "towers
lose at decode" is `tokens=256, slab_rows=64` — 4 slabs, and at `bps=2` that is **8 blocks of a
256-block launch**. The project already diagnosed a half-idle grid as a 4.2× artifact, and
`bps` 1→2 alone is **+64%**. The stated mechanism ("the owner reduce is no longer amortised")
therefore competes with a known artifact. Fix: each transport is run at **its own** optimal
`(slab_rows, bps)` per message size, plus a decode-size geometry sweep that restores block count.
The reversal is not quoted as a law until it survives at matched block count.

### 4.6 Validation: fit/hold-out, bars, and the three falsifiers

**The split is declared here and nowhere else** (the two sibling documents contradicted each
other and the ladder's hold-out leaked: `CAL-T` fitted the very grid `HO-2/HO-3` held out, and
two "held-out" points already had published residuals). Binding rules:

* **Any point measured in Phase 0 is FIT.** Phase 0's transport ladder is cut to **endpoints
  only** (tokens {256, 16K}, slab_rows {128, 1024}, depth {4, 32}) so every interior point is
  genuinely unseen.
* A **PRIOR-VALIDATED (in-sample)** row exists and is labelled as such: the λ held-out point
  (+0.47%) and the affine-T midpoint (+1.07%) are prior evidence, not campaign hold-outs. The
  λ fit and its "hold-out" come from the same replay ladder and the same disputed σ instrument;
  it is demoted from "the model's strongest validation" to **internal consistency** until K3's
  RR-permuted and adversarial points land.
* **HELD OUT:** B2's interior sizes {15, 33, 60} MB · K1's interior T {1,536, 2,048, 3,072} ·
  K3's adversarial point · K2's interior φ · B4 at `co=1` · all Phase-4 cells. The split file is
  committed with a git sha **before** Phase 2 runs.

**Bars** (boundary instrument only; e2e may never adjudicate a prediction smaller than its own
measured envelope): median regret ≤3%, p90 ≤8%, max ≤15%, Spearman ≥0.70 per cell with ≥6 arms,
sign accuracy ≥90% over the pre-registered denominator, θ stability within ±2× its own CI.
**Boundary-location error is no longer scored as "≤1 grid step"** — that bar is trivially met by
any bracketed crossover on a coarse grid and its uncertainty (`σ_diff / slope`) was never
propagated, and near an optimum the slope is ~0. Replaced by a **bootstrap profile-CI criterion**
on the location, with the rule that a cell whose location CI spans >2 grid steps is declared
**unable-to-adjudicate before it runs**.

**The falsifiers of the contribution itself** (matched `Q`, mismatched `W`; if the optimal module
composition differs across the pair, the sufficiency claim is dead and we name the missing
coordinate):

| id | construction | matched | mismatched | prediction | h |
|---|---|---|---|---|---:|
| **FQ-1** | (a) `T_pad`=4096, φ=0.5; (b) `T_pad`=2048, φ=1.0 — one alternating same-session 2-arm campaign | `φ·T_pad` = 2,048 real rows; same k, same σ | φ, `T_pad`, A | `C*` and `d*` identical | 0.4 |
| **FQ-2** | CDAR transport at fixed message size, with and without a co-resident MFMA body of **matched duration** | `B_bnd`, λ, P, A | `co` | depth flat in one, cliff in the other | 0.3 |
| **FQ-3** | signal count at **fixed** geometry (certify every 1/2/4 slabs at `slab_rows`=256), then geometry at fixed signal count | — | A alone / P+geometry alone | the first is free, the second is 2.1× | 0.7 |
| **FQ-4** | two schedules with identical `Q` but different peak/mean fabric injection (metered vs bursting ERS) | all eight | burstiness | identical wall | 0.0 (rides E-B2) |

---

## 5. Metric map and reporting discipline

Condensed from `design/METRICS_PROTOCOL.md`, which amends `nightshift/BENCHMARK_PROTOCOL.md`
rather than forking it.

| class | deployment decision | PRIMARY | guards that can veto | evidence today |
|---|---|---|---|---|
| **A** prefill batch/offline | corpus per node-hour | **M1** aggregate input tok/s, steady-state window | accuracy; `sealed/in_bucket ≥ 0.95`; **`W` matched within 5%**; memory + achievable C | c512p, **n=2, below the ≥5 floor** |
| **B** interactive low-C | will users tolerate it | **`goodput_in(a,b)` attainment curve**, open-loop | TTFT p99, TPOT p99, M1, coverage | c32p closed-loop, n=1 unbalanced |
| **C** saturated high-C | max sustainable rate at SLO | `λ*` such that `attainment ≥ 0.99` | TTFT p99 stability, M1 at λ* | **ZERO** — open-loop cells never run |
| **D** decode-heavy | one user's stream rate | **M6** = 1000/TPOT p50, TPOT p99 co-primary | **M2 banned at fixed OSL** (`M2 ≡ M4 × OSL` exactly) | **ZERO** |
| **E** mixed | the real production regime | `goodput_in` + `goodput_out` as a pair | **`κ` mandatory**; >5% cross-arm κ is a scheduler result | **ZERO controlled** |

**Rules that are load-bearing, each with the measurement that forced it.** Bare "tok/s"/"TPS" is
a protocol violation (63.3× ambiguity in one run). Closed-loop TTFT/e2e are labelled
`(closed-loop, backlog-inclusive)` and never quoted against an SLO — at c512p the run is
**4 generations deep** and its 43.1 s TTFT p50 is ~3 generations of queueing. A closed-loop cell
needs `num_prompts/C ≥ 20` and Little's occupancy ≥0.95; **our own banked c512p cell scores 4.0
waves and 0.899** and is labelled `R-L3 FAIL` until the confirmatory 10,240-prompt pair runs.
TPOT's denominator is `OSL−1` and must be stated (at OSL=8 the conventions differ by **14.3% —
larger than the entire c512p result**). Runs, not (run, rank) cells, are the unit of analysis
(the same null reads `t = −5.01` one way and `t = −0.73` the other). Every delta is sized against
its **own cell's** envelope — **±15%/±18% at c32p** vs **≤1.8%/≤0.25% at c512p**; inheriting the
c512p band anywhere else is fraud.

**The one open contradiction, carried not forked.** `BENCHMARK_PROTOCOL.md:61-64` makes TTFT
p50/p99 primary. Until the operator rules, **BENCHMARK_PROTOCOL wins**: Phase 4's PRIMARY is the
standing protocol and the amendment metrics (`goodput_in(a,b)`, M1) are reported as guards, with
every cell flagged where the two conventions select different arms.

**Three new instrument gates, all admission gates rather than hygiene.** **Actuation receipts** —
a device-side skipped-rows/`T_eff` counter for the φ cell, an observed per-rank destination-load
CV for the skew cell, a per-rank descriptor echo for per-rank `C`; without them a silent no-op is
indistinguishable from the registered falsification, which is the 2%-coverage disaster in new
clothes. A **forwarding contract** — every knob rides the whitelisted `K0_MPS_CFG` path with a
device-side echo and the reference sanity ratio in every row (the precedent is a `K0_PF6GM_G`
default that gave a candidate a **~330 µs handicap in its own favour and an invalid claim**).
And **thermal/clock parity** — continuous `amd-smi` clock+temperature at ≥0.2 Hz, first
invocation of each session discarded, foreign-process check recorded, level order randomized with
a repeating anchor regressed out (GPU2 runs hottest at 77 °C with a 4–5% clock spread; there is a
**+18.7 ms first-run penalty and +3.4 ms monotone soak**).

---

## 6. Overlap atlas — the ranked opportunity table

Condensed from `design/OVERLAP_ATLAS.md` (39 opportunities across 8 families). The top of the
ranking, with the screen each passes:

| rank | opportunity | regime | slack ≥ comm? | cost | EV |
|---|---|---|---|---|---|
| 1 | **`C*(W)`** — re-sweep the consuming pool in every W-cell | DP/EP, all | yes | **zero code** (one runtime byte) | HIGH |
| 2 | DP prefill co-scheduling / chunk packing | serving c32p | n/a | scheduler only | HIGH (**15–25% of node throughput**, kernel-independent) |
| 3 | RR / EPLB owner placement | serving, skew | n/a | host + 1.31 GiB/rank | HIGH |
| 4 | `max_num_batched_tokens` sweep | serving, all | n/a | integration only | HIGH (never swept) |
| 5 | shared-expert filler under TP8 | TP8 prefill | yes, bounded | queue priority | HIGH (**~32–34% of the prize, not the prize**) |
| 6 | **metered collective** — smooth the AR to ~1/3 duty instead of bursting it | TP8 prefill | yes | ledger + re-sweep | HIGH (reframe) |
| 7 | RCCL reference + one-shot AR at decode sizes | TP8 decode | unknown | **one argv** | HIGH (measurement) |
| 8 | expert-weight prefetch during attention | decode | derived, large | reuse M20 | HIGH |
| 9 | dummy-step early-out (tier-1 fill) | serving c32p | n/a | vLLM-side | HIGH @ c32p |
| 10 | wave-specialized transport | both | n/a | **M, not S** (see below) | MED-HIGH |

**The atlas's single most important derived finding, and it reframes the TP8 track.** Under
TP8+EP the fabric needs only a **28–39% duty cycle** to carry the layer's 822 MB of per-rank
egress; RCCL's fabric-busy time (`822 MB / 280 GB/s = 2.94 ms/layer`) equals the banked *exposed*
ring-AR (~2.8 ms/layer), so **production hides essentially 0% of its collective** — the 27% prize
pool is real, and the binding constraint is **scheduling, not bandwidth**. Corollary: *a 1.5×
slower transport that runs continuously beats a fast one that bursts.* This has no coordinate in
`Q` today, which is exactly why burstiness is a declared provisional coordinate and FQ-4 exists.

**One atlas item was corrected and must not be run as written.** The wave-specialization arm
(`if (threadIdx.x >= 192) transport_loop(); else mfma_body();`) is **not an S-sized change**:
`m25_boundary_bench.hip:147` is `__launch_bounds__(256,1)` and the fused path contains **10
`__syncthreads()`**, so a role split puts waves on divergent paths across block-wide barriers.
Any real version restructures the hot loop and its register allocation — the class of change that
cost **+726.9 µs with source untouched**. Re-sized **M**, moved to Appendix A, hypothesis
restated: the missing resource is *per-wave ordering*, not per-CU memory capacity.

---

## 7. The experiment ladder

Condensed from `design/EXPERIMENT_LADDER.md`, repriced and re-gated per the feasibility review.

### 7.1 Gates

```
PHASE 0 CALIBRATION   exit: θ complete with CIs; σ_rig MEASURED; clock/temp parity; actuation receipts
PHASE 1 REGISTRATION  exit: predictions.jsonl + Q arity + law-card titles + scoring denominator committed with a sha
PHASE 2 BOUNDARY GRID exit: every registered prediction adjudicated at the precision instrument
PHASE 3 FIT+HOLD-OUT  exit: bars met, or the degradation ladder executed and the miss published
PHASE 4 E2E SIZING    exit: ≥3 W-points sized under the serving protocol with coverage + composition counters
```

**Two gates had no teeth and now do.** `G0-a` was satisfiable by definition (the residue after
attribution *is* `I₀`) and `G0-c` had an explicit "or declare it" escape branch — deleted. Added
**`G0-e`**: σ_rig must be *measured*, not inherited. The draft's `σ_rig` was a **between-arm
spread read as flat** (0.1–0.3% across four different depth arms), not repeatability, and **no
repeated identical boundary invocation exists anywhere in the repo** — yet every Phase-2 sample
size rests on it. P0-A0 (20 invocations of one fixed config spread across a session, plus 5 after
a 30-minute idle, ≈0.4–0.8 h) supplies it, and every `K` is re-derived from the result.

### 7.2 The cell registry (one canonical ID space — the renumbering deleted two cells)

`W_VECTOR §4.4` and `EXPERIMENT_LADDER §3.4` used `K1–K8` with **different meanings**, which
silently deleted the granularity-decomposition cell (= FQ-3) and left the producer-order ×
certification law — ranked the **#4 highest-leverage knob in the inventory** — with no cell at
all. One registry, used everywhere from now on:

| id | cell | axes | instrument | K | adjudicates | actuator status |
|---|---|---|---|---:|---|---|
| **P0-A0** | rig repeatability | 1 config × 25 invocations | boundary | — | σ_rig | exists |
| **P0-A** | transport endpoints + incast | b {3.67, 235} × transport × **fan-in k ∈ {1,2,4,7}** + concentrated-vs-uniform destinations | transport-only | 3 | θ-F*, incast term | exists |
| **P0-B** | ledger + ρ ladder (**absorbs the old B1**) | `k_inner` → ρ ∈ {0.65,1,2,3,4} × 4 arms; phase ledger; duty histogram | boundary | 3 | θ-P5/P6/P12, `h_ledger`, P2 | E-B1/E-B2/E-B3 to build |
| **P0-C** | mega chassis | θ_flop per limiter class, φ_p, `c_bar`, ψ (with per-point timeouts) | MoK | — | θ-C*, θ-P3/P4 | exists |
| **P0-D** | serving envelope + TP8 profile | ≥2 repeats per cell; profiler confirmation of the 2.8 ms/layer exposed AR | serving | — | envelope, ρ at deployment | exists |
| **FQ-1/2/3/4** | **the sufficiency falsifiers** | §4.6 | mixed | 2–3 | **H-Q itself** | X1 for FQ-1 (fallback: E-F) |
| **B2** | `λ` / K1 crossover | b ∈ {3.67,15,33,60,235} × transport × `co` ∈ {0,1}, each at its **own** optimal geometry + decode geometry sweep | boundary | 3 | L3, P3, P3c | X4 (transport argv) |
| **B4** | depth × co | depth {2,4,8,16,32} × co {0,1} at ρ≈3 | boundary | 3 | L2, P5d | one `vmcnt(2)` specialization |
| **B.res** | **the residency axiom** | `N_CTA` {160,192,224,256} × {comm in-kernel, comm as a 2nd kernel} | boundary | 3 | LAW-28's scope; the composability boundary | exists |
| **K5** | **doors 2×2×2** | depth × certification × **order** | MoK | 4 | PF4 (substitutes?), teaching-list law #4 | exists |
| **K4** | `C × flush_rows` | C {0,8,16,20,24,28,32} × flush {0,8,16,32}, screened | MoK | 4 | L4, P6, P7a | exists |
| **K1** | `T` ladder | T {128,256,512,1024,1536,2048,3072,4096}, 3 arms in one invocation | MoK | 2 | L1, P1 | **X2** |
| **K3′** | skew, reduced | {balanced, aggregate-hist, **adversarial**} × {S4-C24, S2-C64} + captured/shuffled replay | MoK + replay | 2 | P6c, P7c, P9adv | histogram path exists; replay never run on GPU |
| **K2** | φ at fixed `T_pad` | φ {0.25, 0.376, 0.5, 0.7, 1.0} × C {16,24,28} | MoK | 2 | L5, P1c, P1d | **X1** |
| **E-F** | concurrency ladder | C ∈ {32, 128} × {m15-DP, native-DP, native-TP} | serving | ≥6 rot | the crossover concurrency; the **ungated φ actuator** | exists |
| **E-B′** | admissible c512p | 10,240 prompts (waves ≥20), m15 vs native-DP | serving | 1 balanced pair | L1/L4 sizing | exists |

`C`'s local grid is made **uniform** — `{16, 20, 24, 28, 32}` — because the `C* = 24 ± 4`
falsifier was arithmetically ill-defined on the old `{…,24,28,32}` ladder (±4 was one step at the
top and half a step at the bottom). The bar is now stated in absolute CTAs.

### 7.3 Budget — repriced honestly

Unit costs, declared: boundary invocation **[1, 3] min, uncited** — measured in the first ten
minutes of session 1, and both endpoints published until then; MoK 5-rotation campaign
**3 min 5 s** (cited); route-replay campaign **139 s** (cited); serving pair **~55 min**
(back-derived). Every row is priced as `pts × K × unit`, which the draft's Phase-2 table omitted
(it priced off `pts` alone, understating boundary invocations by 1.6×).

| block | node-hours (1 min / 2 min boundary unit) |
|---|---:|
| Phase 0 (P0-Z free triple, cadence, A0, A, B, C, D + 20% retries) | **8.2 – 11.2** |
| Phase 1 (registration) | 0.0 |
| Phase 2 core (FQ-1..4, B2, B4, B.res, K5, K4, K1, K3′, K2) | **12.1 – 14.9** |
| Phase 3 (fit + contingency) | 2.0 |
| **Kernel-level subtotal — the part that defends the contribution** | **≈22 – 28** |
| Phase 4 (E-F at C ∈ {32,128} × 3 arms, ≥6 rotations; one admissible c512p pair) | **16 – 20** |
| **TOTAL** | **≈38 – 48 node-hours** |

Applying the node-availability derate the draft omitted — this week alone the node was
unreachable ~8 h, the queue was killed with work stranded, and the IP changed mid-campaign — at
**0.6–0.7 effective** this is **≈60–75 wall-hours on the node ⇒ 6–8 overnight sessions**, not
four. Additional execution rules: a paired contrast spanning a session boundary is
`voided_by: cross-session` unless the anchor series shows drift below the cell envelope; the
runner is idempotent and manifest-resumable, writing each `runs.jsonl` row before the next
invocation; a **hang is a manifest outcome value**, never a dropped point (C=32 hung 1 of 2 runs,
and `C*` sits one grid step below it), with an explicit hang allowance in the budget.

### 7.4 Ordering by information value

Re-ordered so that cells adjudicating a `W → S` law come first; engineering unblockers are named
as unblockers rather than ranked as science.

1. **P0-Z** — the free CPU triple (pad-row dedup test; reconcile the two σ instruments; compile
   one small-M decode tile body and read occupancy). Zero GPU time, and the third is the single
   largest structural fork in the campaign: if decode tiles run at occupancy ≥2, a fifth door
   opens and every prefill-rejected "fill the stall" design becomes legal at decode.
2. **P0-A0 + cadence** — measure `σ_rig` and the invocation cost before anything is sized on them.
3. **P0-B** — the ρ ladder and the device phase ledger. *One instrumented run replaces the next
   four guesses*; it removes both fudge factors' ambiguity and it re-scores four falsifications
   that were graded against an impossible bar. **With the running-maxima reset baked in, or the
   ledger lies.**
4. **FQ-1/2/3** — the test of the contribution, ≈1.4 h, run before the grid that assumes it.
5. **K5 (2×2×2)** — if depth and certification are substitutes, **no waterfall in the paper is
   additive**; this changes `M`'s structure, not its coefficients, and it is the cheapest of all
   the structural tests.
6. **K4** — the literal expert question, zero code, one runtime byte.
7. **B2, B4, B.res, K1, K3′, K2** — the remaining four laws.

Started day 1 in parallel because their lead time exceeds their run time: **X1** (make
`K0P6_M24_FILL` compile — the only actuator for φ) and **X2** (port the tgen hole-sentinel fix
into the serving M15 body — without it every `T ≠ 4096` arm on that body is void).

**Phase-2 node-hours are gated on ≥2 of {X1, X2, X3} landing.** Three of the four sign-flip laws
depend on unbuilt actuators; if all three slip, the campaign spends its budget re-confirming two
already-banked laws, which is precisely the "expansive, unclear contribution" failure mode. The
**ungated fallback for φ already exists**: the concurrency ladder produces φ ∈ {0.376 … 0.985}
emergently and is settable today.

### 7.5 Kill criteria and the degradation ladder

| family | kill criterion | what ships instead |
|---|---|---|
| fused collective (S6/CDAR) | at ρ ≥ 2.5 with filler and metering, **`h_ledger` < 20%** across all arms | the **RCCL-hybrid**, plus the law `fuse iff h > 1 − (1 − I_co/T_vendor)/β` (48.7% today, 22.8% at parity) |
| `M-POOL-CONSUME` | `C*` lands at 24 ± 4 in **every** cell | the law becomes *"`C*` is a constant of the skeleton, not a function of W"* — still a law, and it retires the adaptive-C build before it is written |
| `M-POOL-CARRY` | loses monotonically **including under adversarial skew and at every `N_CTA`** | LAW-28 upgrades from carrier-conditional to regime-general on this hardware |
| `M-FILL` | `C*` and `quota*` do not move with φ at fixed `T_pad` | M24 is demoted to a work-**deletion** feature; the fill story collapses to tier-1 dummy-skip |
| `M-PLACE` | measured serving gain **<10%** | the finding is that **the replay rig over-states skew** and every replay-based skew number is re-scoped — a first-class result |
| any zero-door module | measures inside its cell's equivalence margin (TOST at the 3% smallest-callable-screen margin) | nothing — it was screened for free. Conversely a reproducible >3% zero-door effect means **there is a fifth door**; report loudly |
| **the contribution (H-Q)** | any matched-`Q` / different-composition pair from FQ-1/2/3/4 | H-Q is dead, we name the missing coordinate, and we ship the regime map |

Degradation ladder, pre-committed: all bars met ⇒ **a composition policy**. Median regret >3% but
sign accuracy ≥90% ⇒ **a screening tool** (M orders modules and rejects zero-door proposals for
free). Sign accuracy <90% in one topology ⇒ **a regime map plus one law**. Both ⇒ **a regime map
plus the falsified ledger** — which is the fallback, stated in §1, and not the headline.

---

## 8. Per-region build plan — predictions as module composition diffs

Every row is a **diff from the base kernel**, the `Q`-inequality that triggers it, and a
registered prior that can lose. Cell ids are §7.2's.

### Region 1 — high-C DP/EP prefill (`φ ≈ 0.985`, `T_eff ≈ 4,035`, `σ_agg 5.088×`)

**Compose:** BASE **+ M-ORDER + M-CERT + M-DEPTH(d=4) + M-POOL-CONSUME(C ∈ [24,28], quota scaled
with `T_eff`)**. **Detach:** `M-POOL-CARRY`, `M-FILL` (φ≈1 ⇒ nothing to delete).
**Why:** `T_eff` is **3× above `T*`** so `α = 0.183 ≪ 0.412`; `C_sat = 20.8` vs measured knee 24.
**REGISTERED PRIOR (P7a):** `C* = 24 ± 4` at **every** φ, because `V_cert` and `τ_shadow` are both
∝ `T_eff`. **This contradicts the kernel's own comment** that its tuned point is invalid at other
`T_eff` — one of them is wrong and K4 decides which. **Falsifier:** `|C*_pred − C*_meas| > 4`
CTAs ⇒ `τ_shadow` is not linear in `T_eff`, `M` needs a `φ_p(T)` term, and "one C fits every
cell" — the cheapest thing the campaign could ship — is dead.
**Falsification arms, same session, alternating:** `M-POOL-CARRY` at C ∈ {32,64} (predicted
+332…+340 µs); `flush_rows = 0` (isolates pure protocol deletion); BASE alone.

### Region 2 — low-C TP8+EP prefill (the open C=32 arm-level gap)

**Champion baseline, named honestly: the RCCL-hybrid** — vendor collective (2,909 µs in-rig,
79% of the egress card) with our megakernel owning the MoE region.
**Compose (ranked):** (1) RCCL-hybrid; (2) CDAR **+ M-SHEXP** (covers ~32–34% of the exposed AR
with zero new machinery, and under TP8 the shared expert's output folds into the same
accumulator, deleting its separate all-reduce); (3) CDAR with a **metered** ERS (spread commits
across the whole producing GEMM, not its epilogue).
**REGISTERED PRIOR (P2/P2c):** at the corrected ratio with an unchanged schedule, `h_ledger` lands
in **[10%, 35%]** — still below `h* = 48.7%` — and **fused CDAR does not beat GEMM→RCCL until
`β < 1.2`**. **Falsifier:** `h_ledger ≥ 49%` and still losing ⇒ `D2` is missing a term (most
likely a co-residency tax on the compute side, i.e. `I` multiplicative not additive).
**Amendment that makes this measurable at all:** `h` was defined as a wall difference
`(phased − fused)/collective`, which **cannot separate hiding from co-residency tax** — an arm
genuinely hiding 20% while paying 400 µs more interference measures ~2%, indistinguishable from
the reported 2.1%. The kill criterion now rides `h_ledger`, computed from **span overlap** in the
device ledger; the gap between `h_ledger` and `h_wall` **is** `ΔI_co`, the term `M` needs.
**Standing caveat:** the −21.17% is an **arm-level** delta at two different W points (ours
4096/128, the vendor's 16384/2048) and **cannot be read as a kernel delta**; only the three-way
decomposition is a valid reading. It appears in this document only in that form.

### Region 3 — DP/EP decode

**Compose:** BASE only — **the lever is not in the kernel, and that is the finding.** Post-M23
the uniform-decode rescue routes decode steps through the B4096 graph, so the mega executes
**4,096 padded rows for ~1 real token per rank**: whatever the kernel does on a decode step, it
does at prefill shape. The DP decode levers are **integration-side** — bucket size (fixed at 4096
in every run ever executed, never swept) and the tier-1 dummy-step skip (**56.0%** of in-bucket
steps at c32p are fake work both arms pay). **REGISTERED PRIOR, sign-only:** the fused family's
ratio degrades monotonically as `T` falls, crossing 1.0 near `T* ≈ 1,311`. **This loses if** the
measured ratio at T=256 comes in below 1.5×, in which case the affine model's fixed term is not
constant at small T and `α`'s framing is scoped to `T ≥ 512`. Decode-shaped T folds into K1.

### Region 4 — TP8+EP decode (latency regime)

**Compose:** a **different base** (D1: latency-shaped transport, no tower fan-out, per-owner
certificate scope, AR+RMSNorm fusion) **+ M-PREFETCH**. Three structural choices reverse: `K1`
(towers lose), certification (chunking collapses below ~2K tokens), `M-DEPTH` (inert).
**But the sharper prediction is that the collective is the wrong target:** at decode, 32 local
experts × 42 MB = **1.34 GB/rank/layer** of weight streaming against a 3.67 MB collective; priced
against the **measured** 4,151.9 GB/s HBM plateau (never a back-derived peak) that is ~323 µs of
weight traffic per layer. **REGISTERED PRIOR, sign-only:** expert-weight HBM streaming exceeds the
fabric collective by ≥5× per layer, so a fused decode arm measures ≤3% better than RCCL because
both are latency-floor-bound; the decode lever is prefetch, and the mechanism already exists in
M20 and has never been pointed at decode. **This loses if RCCL at 3.67 MB comes in below ~250 µs**
— then our 801 µs point is 3× off the achievable floor and transport becomes the decode lever.
**Every sentence in this region is provisional: this project has never measured decode.**

### Region 5 — skewed routing

**Compose:** **M-PLACE first, then per-owner certificates — and the order is the finding.**
`dT/dσ = T(1)·λ = 3,649 µs per unit σ per layer`, against the largest schedule knob in the entire
ledger at 615 µs: **skew dominates every schedule knob by ~6×, and its remedy lives outside the
kernel.** **REGISTERED PRIOR (P9), deliberately the campaign's most exposed over-prediction:**
round-robin/EPLB0 placement is worth **+27% to +46% of absolute node throughput for BOTH arms**
with **<3%** change in the A/B ratio. **Falsifier, pre-interpreted:** a measured gain <10% means
the replay rig over-states skew (it replays skewed routing on 100% of rows while only 37.6% of
serving rows are real) — which would **invalidate every replay-based skew number in the ledger,
including the 3.6× that motivates this region.** Two honesty flags ride it: the two per-call skew
instruments disagree by ~2× and must not be mixed, and ideal static redistribution still leaves
per-call p95 at **3.05×** — placement is a static remedy against a per-call problem.

### Region 6 — gfx942 portability (access-gated, Appendix A)

**Compose:** BASE + **M-TOWERS**, not `M-RMW`: `b*` falls from 32.8 MB to **≈11 MB ≈ 780 tokens**,
so store-class transport is right almost everywhere — **a simplification for the PR**, since the
two arches then agree at deployment message size. **REGISTERED PRIOR:** `d* < 4`, plausibly 2
(atomics drain ~3× slower ⇒ congestion forms at lower depth). This competes with a second
registered model of depth (in-flight *bytes* vs in-flight *time*); B4 discriminates them by
varying op **width** and op **rate** independently.

---

## 9. What we teach people — the law-card list

The publishable unit is the **card**, not the number: statement, mediating `Q`, `Q`-inequality,
θ with its fit provenance, predicted-vs-measured, cells tested, instrument + operating point,
confidence tier, **failure scope**, **retirement condition**, manifest rows. A card with an empty
failure scope or retirement is a tuning constant wearing a law's clothes. **Titles are
pre-registered in Phase 1**, so this is a committed artifact.

| # | law we expect to teach | cell | status entering the campaign |
|---|---|---|---|
| 1 | **`C*` is derived from `(Σ, P, t50)`, never carried** — a reserved CTA costs 8–9 µs unless it consumes a certified prefix | K4 | measured at one cell; `C_sat` predicts the knee within one grid step |
| 2 | **Depth binds iff the fabric shares the device with compute** — θ is contention, not bytes | B4, FQ-2 | PROVEN-replicated at `co=1`, MEASURED-once at `co=0` |
| 3 | **Granularity is free when it moves only `A`; expensive when it moves `P` or transfer geometry** | **FQ-3** | two repo results disagree (free in M15, 2.1× in CDAR); reconciling them *is* the contribution |
| 4 | **Order sets the arrival time of the certified prefix; certification and pool are worthless without it** — ≈0 alone, −144 µs jointly | **K5×order** | the cheapest lever in the program; had **no cell** until this document |
| 5 | **Choose transport op class by the arch's atomic:store ratio at the deployment message size**, never off the small-message card | B2, gfx942 | reversed a design decision once already |
| 6 | **Carry vs idle vs consume: the same reservation, three signs** (+340 / +8–9 per CTA / −444 µs) | K4, B.res | the direct answer to the experts' first question |
| 7 | **Quota = shadow duration ÷ per-CTA consume rate (6.1 GB/s/CTA)** — re-derive, never carry | K4 | design-quantified, never isolated |
| 8 | **Row-proportional cost should scale with real tokens; every quota tuned at padded capacity is wrong by 1/φ** (2.66× measured) | K2 | gated on X1 |
| 9 | **Replication pays iff per-chunk coverage exceeds θ; aggregate skew statistics mispredict serving** | K3′ | the most instructive contradiction in the corpus (4× replay win → −5.7% e2e) |
| 10 | **Fuse the collective iff `h > 1 − (1 − I_co/T_vendor)/β`** — a bandwidth-ratio threshold, not a philosophy | P0-B | currently a measured negative, and the negative is the deliverable |

Plus the **negatives**, which are half the teaching value and must be published: the cross-launch
deferred combine (+75.8 µs — *"the wait it escaped was mostly not on the critical path, and the
parity state it added was"*); pacing a dedicated pool (monotone, no interior minimum ⇒ closed);
coalescing protocol atomics (**≥10× slower** — an atomic resolves *at the line*, so `L` should be
**maximised**, the opposite of the load rule); XCD-confined pool placement (36% less interference,
54% pool starvation, net 7–23% worse); and the four G25-1 producer-side falsifications, now
correctly labelled *tested at an operating point where hiding was Amdahl-capped near zero*.

---

## 10. The HipKittens PR — the module library and the composition interface

**The PR is the contribution, shipped as code.** It has three parts.

**(a) The module library.** Ten headers already exist under
`include/cdna4/ops/group/distributed/{completion,counter,credit,lifetime,order,packet,peer,roles,slab,sync}.cuh`,
covering PGL/peer translation, typed packet movement, packed-bf16 peer accumulation, directional
release/acquire, epoch publication with bounded waits, counted fan-in with reservations, directed
retirement credits (ready ≠ reusable), and CTA role helpers. Each §3.2 module maps to one of them
or names a new one (`M-SHEXP`, `M-FILL`, and the metered-injection policy are new).

**(b) The composition interface** — the part that does not exist yet and is the actual PR: a
declared schedule descriptor that names which modules are attached, carries each module's attach
condition as a `Q`-inequality, and is **echoed by the device** so an arm's composition can be
verified rather than assumed. The precedent for why the echo is mandatory is measured: a knob the
harness silently failed to forward produced a **~330 µs handicap in the candidate's favour and an
invalid claim**, detectable only by a sanity ratio.

**(c) Three API changes already derivable, each decided by a named cell:**

| primitive | law | API consequence | cell |
|---|---|---|---|
| `credit.cuh` (`throttled_accumulate_bf162`, depths {0,4,8,16,32}) | depth binds **iff `co = 1`**; `d*` predicted to move **down** to 2 in the fused arm and on gfx942 | the bound needs a **`co` scope note** in its contract, and **depth 2 does not exist**: the `g` `0x300` selector is a full 2-bit field, so adding it is an **encoding change, not a parameter** | B4, FQ-2, gfx942 |
| `slab.cuh` (`certify_slab`, `bounded_wait_slab_into`) | granularity is free when it moves only `A`, expensive when it moves `P` or transfer geometry | the primitive **conflates signal count with transfer geometry in one parameter**; it should take **two** | **FQ-3** |
| `order.cuh` (four branch-free maps) | order alone ≈0; order × certification = −144 µs; and `Φ_slab(2)` vs `Φ_natural(k)` cross at low k | order selection should be **a function of top-k**, exposed as policy rather than a compile-time default | K5×order, K6 |

**Manifest consequence, binding from Phase 1:** `arms.jsonl` gains `primitive_touched` and
`api_implication`, every cell row carries a primitive column, and Phase 1 pre-registers which
primitives the campaign expects to **confirm / generalize / retire**. A wrong pre-registration
there is as publishable as a right one, and it makes "what does the HK PR look like?" answerable
on the day the campaign ends rather than three weeks later.

**One structural constraint the PR must state rather than discover:** *policy CHOICE stays
runtime; policy CODEGEN stays compile-time.* `s_waitcnt vmcnt(N)` is simm16-only — there is no
register form — and templating the hot loop on `Depth` spilled one 4-byte value with **389
reloads**, contaminating the control arm. A "runtime depth" API is illegal on this hardware, and
the shipped mechanism (mutually-exclusive SGPR lane masks selecting compile-time specializations)
is the pattern to teach.

---

## Appendix A — Extensions cut from the core campaign

Cut with the cost of each cut stated. Total ≈17–18 node-hours (~45% of the expansive grid).
**The two largest cuts remove arms whose own registered predictions said "our thing loses."**

| extension | node-h saved | why it can go | what is lost |
|---|---:|---|---|
| `B.hyb.world` (world ∈ {2,4,8}) | 0.3 | world size is HW-FIXED at 8; the prediction is pure byte arithmetic `2(N−1)/N`, already computed | an axis with no deployment cell — publish the arithmetic instead |
| `B.tp8.shexp` as a *cell* | 0.4 | adjudicates no registered prediction; the atlas itself caps it at ~32–34% of the prize | it is an engineering arm, not a law — moved to the build track |
| fp8-on-wire (`B9`) + **X5** | 0.3 + an M–L build | the ladder already concedes that if X5 slips, precision is "a declared coordinate, not a swept one" | the largest engineering item serving the smallest cell |
| `K.decode.dp` as a separate cell | 0.4 | shares actuator, instrument and body with K1 | nothing — decode-shaped T folds into K1's ladder |
| gfx942 cell | 1.0 + access risk | access-gated; the predicted outcome is *a simplification for the PR* | defer; one run if the node materializes |
| **E-A as a run** (C=32 arm-level) | 4.6 | the −21.17% **cannot be read as a kernel delta**; the valuable output — *TP8 wins every SLO tighter than ~3.5 s TTFT; we win all tails* — is **already derivable from banked data** | converted to an analysis deliverable at 0 node-hours |
| **E-D** (decode e2e) | 4.5 | no decode claim rests on a measurement; its own registered prediction is *we tie or lose*; the mega runs decode at prefill shape; and it is below the ≥5-pair floor with an unmeasured envelope | the *prediction* stays published; the run waits for D1 to exist |
| E-E (open-loop) | — | **kept as exploratory**, labelled "observed, mechanism unattributed"; PE5 dropped from the scored set | it needs a client v4 patch (`sent_s/issued_s/scheduled_s` are null) — tracked as **X6** |
| `B3`, `B5`, `B6` (slab geometry, MAG pullers, wave split) | 1.2 | B3 is confounded and superseded by FQ-3; B5/B6 are TP8 transport engineering; the wave arm is **M, not S** and needs barrier restructuring plus a compute-only 3-wave control | TP8 transport tuning; re-fund only if P0-B shows `h_ledger ≥ 20%` |
| Phase 4b as a phase | — | the concurrency ladder is promoted into the core (it was hiding the campaign's best number in a conditional phase); placement folds into K3′ plus the EPLB predictor's own gate | the conditional structure |

**What must not be cut, despite being tempting:** P0-Z (free, and the decode occupancy check is
the largest structural fork in the campaign); the device phase ledger + duty histogram (it removes
**both** fudge factors with one instrument and turns the duty-cycle reframe from an argument into
an experiment); and the retraction/negative ledger, which is the most credible page in the set and
belongs **in the deck**, not an appendix.

---

## Appendix B — Unresolved critique points, with reasons

Deliberately not resolved in this document. Each names the review item, the reason, and what would
change the decision.

1. **The two per-call skew instruments remain unreconciled (5.15× vs 2.61×).** Reason: it is
   CPU-only work scheduled as P0-Z2, and reconciling it *on paper* would be guessing. Every
   σ-dependent prediction is therefore stated as a **band**, and the two instruments are never
   mixed. Changes when P0-Z2 lands, before Phase 1's registration commit.
2. **`I ⟂ C` vs LAW-14's +87% marginal-interference growth** (mechanism M1). The mechanistic
   resolution offered — interference is invariant at *fixed protocol volume*, and a carrier pool's
   protocol volume grows with `C` while a consuming pool's does not — is plausible and unmeasured.
   Deferred because resolving it silently inside the cost model is exactly what the grounding's
   reading rules forbid; it is filed as a **named contradiction** to be resolved by the phase
   ledger, not by assertion.
3. **`J(d) = 0` for `d ≤ 8` while K5 spends `K = 4` measuring the 50.9 µs depth-4-vs-8 difference**
   (mechanism M3). Kept as-is: the model states a tie below the model's declared resolution, and
   the cell measures whether the tie is real. If d=4 beats d=8 reproducibly, `M` gains a monotone
   sub-cliff term — a coefficient change, not a structural one.
4. **Two competing resource models for injection depth** (in-flight *bytes* vs in-flight *time*,
   mechanism M4). Both are registered as **competing hypotheses** rather than resolved, because
   the discriminating measurement (vary op width and op rate independently) is one B4 arm and
   guessing first would bias its interpretation.
5. **`κ_couple` absorbs at least three mechanisms and the ledger separates one** (mechanism M13).
   Accepted as a bounded fudge factor (`≤55 µs/CTA`, identically 0 without a certified prefix) with
   the elimination experiment named. A three-way decomposition is a second campaign.
6. **The four-door screen has no positive controls, and door membership was assigned knowing the
   outcomes** (mechanism M11). Partially resolved: three already-measured zero-door arms are named
   as `campaign_role: screen_validation` (~0.2 h) so P11 has an evidence path, and P11 is restated
   as a TOST equivalence test at the 3% margin. The 11/11 record remains retrospective and is
   labelled so.
7. **`PF8` implicitly assumes a byte-proportional transport the ledger contradicts** (mechanism
   M14). Moot for the core campaign — the cell is cut with X5 — but the prediction stays on the
   record as a clean test of byte-bound vs issue-rate-bound.
8. **Multi-node, PP, SDMA and energy remain scoped out**: zero evidence; no artifact; never measured
   in situ (the one API measured was CU-lowered, not SDMA); and not instrumented anywhere in the repo
   (energy is `not instrumented`, never estimated from TDP).
9. **`crit persistence` and `burstiness` are declared provisional rather than added to `Q`.**
   Reason: `Q`'s arity is frozen at Phase 1 precisely so that a miss cannot be answered by adding a
   coordinate. Both are pre-registered as the **first two mediators to add**, in that order, and
   burstiness carries its own falsifier (FQ-4) at zero marginal node cost.
10. **The composability hypothesis itself is not resolved by this campaign.** K5's 2×2×2 tests one
    triple; FQ-2 tests one conditional module; B.residency tests one composability boundary. A
    general additivity result over the whole library is out of scope, and the document says so
    rather than implying the library composes.

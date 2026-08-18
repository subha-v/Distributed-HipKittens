# CONTRIBUTION_CRITIQUE — adversarial review under LENS 1 (the "Simran test")

**Reviewed:** `BRIEF.md`, `grounding/{KNOB_INVENTORY,BANKED_LAWS,WORKLOAD_EVIDENCE,TP8_STATE}.md`,
`design/{W_VECTOR,S_VECTOR,COST_MODEL,METRICS_PROTOCOL,OVERLAP_ATLAS,EXPERIMENT_LADDER}.md`.
**Written:** 2026-08-18, laptop, no node access, no ssh. **Reviewer stance:** hostile.
**Scope:** contribution only — is there ONE crisp defensible claim, does every phase serve it,
does it produce falsifiable `toggle X → schedule Y` laws, what must be cut.

**Rules honored:** every issue names a file/section and proposes a fix; every number is quoted
from the draft or the BRIEF digest with its source; nothing is invented. Arithmetic done *here*
is tagged **DERIVED (review)**.

---

## 0. Verdict in one page

The draft is far stronger than the corpus it summarizes. Its grounding discipline (retraction
lists, settability audits, `.text`-sha parity, "a hang is a manifest outcome") is genuinely
better than most published kernel work. **That is not the risk.** The risk is exactly the one
Simran named, and the draft has it in a specific, diagnosable form:

> **The campaign has three different contribution sentences, two different definitions of the
> object those sentences are about, and zero budgeted experiments testing the claim it says is
> the contribution.**

Concretely:

* `W_VECTOR §2.4` says the contribution is **H-Q: sufficiency of a 10-coordinate Q**.
* `COST_MODEL §6` (`:1269`) says the contribution is **"Q and the four-door selection rule …
  eight scalars"** — a *different* Q, with a different arity and a coordinate (`κ`) that
  `W_VECTOR §3.4` explicitly *bans* from Q.
* `COST_MODEL §6` then closes with **two sentences, one per topology** — which is already the
  regime-map fallback of `EXPERIMENT_LADDER §4.4`, not the unified law.
* `EXPERIMENT_LADDER` budgets **≈38.5 node-hours across 17 cells and 26 registered predictions**
  and budgets **0.0 node-hours for FQ-1, FQ-2 and FQ-3** — the three matched-Q falsifiers that
  `§0` (`:12`), `§4.4` (`:581`) and `§7.1` (`:926`) all name as the test of the contribution.

A reviewer who reads the ladder and asks "which experiment tests your central claim?" gets no
answer. That is the whole Simran risk, reproduced inside the document written to defeat it.

**Second-order verdict:** BRIEF output **(7)** — *"Expected emergent abstractions → the shape of
the HipKittens PR; the list of what we'd like to teach people"* — **has no owner document.**
`W_VECTOR` is tagged output (1), `S_VECTOR` (2), `COST_MODEL` (3), `METRICS_PROTOCOL` (4),
`EXPERIMENT_LADDER` (5)+(6). Nothing claims (7). The HK-PR story floats free of the experiments.

**Third:** the budget table does not add up (§8.1). Phase 0 is stated at **≈6.0 h** and its own
rows sum to **7.9 h**; correcting P0-A's own invocation arithmetic gives **≈8.5 h**. The headline
"≈38.5, four overnights" is understated by ~2–2.6 h before any surprise.

**Recommended shape:** one contribution sentence, five laws, **≈20 node-hours, two overnights**
(§6). Everything in §6.3's cut list can go without weakening the claim; two of the cuts *improve*
it by removing arms whose predicted outcome is "our thing loses."

---

## 1. Is there ONE crisp contribution sentence? — **No. There are three, and they disagree.**

### BLOCKING-1 — Three incompatible contribution statements

| where | the sentence | arity of Q | shape of claim |
|---|---|---|---|
| `W_VECTOR §2.4` `:597-602`, restated `§5` `:893-895` | "we identified **ten numbers** you can compute from a serving config, and showed that those ten numbers — not the config — decide the schedule" | **10** | one universal law |
| `COST_MODEL §0.3` `:63-78` and §6 `:1269` | "the whole workload space acts on the schedule through **eight scalars**, and overlap pays through exactly four mechanisms" | **8** | law + selection rule |
| `COST_MODEL §6` `:1276-1280` | "**DP/EP:** governed by `T_eff`… **TP8+EP:** fill does not exist; governed by exposed-collective fraction" | n/a | **two laws = a regime map** |
| `METRICS_PROTOCOL §0` `:50-52` | "a **law card** … the concrete form of the expert's 'list of what we'd like to teach people about'" | n/a | a *format*, not a claim |

The third is the pre-committed *fallback* in `EXPERIMENT_LADDER §4.4` (`:579-580`, "sign accuracy
<90% in one topology ⇒ a regime map plus one law"). Shipping the fallback as the headline before
any measurement is a tell: **the authors do not currently believe the unified claim.** That is
fine — but then say so, and make the regime map the contribution, with the unification as the
stretch goal. What cannot survive contact with Simran is a deck that opens with H-Q and closes
with two topology-specific sentences.

> **FIX.** The master doc opens with **exactly one** sentence and every sibling is edited to
> repeat it verbatim. Recommended (it is the only version all six documents' evidence actually
> supports):
> *"A megakernel's optimal comm/compute schedule depends on the workload only through a small
> set of computable mediating quantities; we name them, measure the thresholds at which each one
> flips a schedule decision, and show the flip is predictable from a serving config with
> arithmetic — including where the rule needs two regimes instead of one."*
> Then state the arity **once**, in one table, and make `COST_MODEL §0.3` and `W_VECTOR §2.2`
> literally the same table.

### BLOCKING-2 — Q is not one vector; the two definitions are incompatible

`W_VECTOR §2.2` Q = `(B_bnd, C_phase, ρ, co, Σ, P/u, κ, φ, A/L, λ/I_bnd)`.
`COST_MODEL §0.3` Q = `(T_eff, co, A/L, φ_p, u/Φ, σ, m, f/W_pad/κ_sched)`.

These are not notational variants:

* `COST_MODEL` Q8 includes **`κ_sched` (DP co-scheduling density)** as a mediating quantity.
  `W_VECTOR §2.4` threat #5 and `§3.4` explicitly exclude it: *"κ_sched and coverage — deliberately
  excluded from Q as validity gates"* (`:630-631`, `:771`). One document's mediator is the other's
  A/B-validity gate. `METRICS §2.2` then makes `κ` a *veto guard* in class E. **Three roles for one
  quantity.**
* `COST_MODEL` has no `ρ`, no `Σ`, no `C_phase` — yet `ρ` is the single most load-bearing quantity
  in the whole draft (it is why the G25-1 PASS bar was unreachable, `W_VECTOR §3.3` finding 1) and
  `Σ` is the slack map that selects S2 vs S4.
* `EXPERIMENT_LADDER §0` (`:12`) declares it inherits **the 10-coordinate Q from W_VECTOR**, but
  every registered prediction in `§2.3` is derived from `COST_MODEL`'s D1–D11, i.e. from the
  8-scalar Q. **The ladder tests one Q and claims another.**

A sufficiency claim over an under-specified vector is unfalsifiable by construction: any miss is
answered with "add a coordinate." Simran will find this in about ninety seconds.

> **FIX.** One canonical table, in `COST_MODEL`, with three columns: `symbol | computed from W by |
> which documents consume it`. `W_VECTOR` and `EXPERIMENT_LADDER` cite it, never restate it.
> `κ_sched`, coverage and `W_pad` move to a separate, clearly-labelled **validity-gate vector V**
> (they already have that role in `METRICS §3.5`). Freeze the arity in Phase 1 alongside
> `predictions.jsonl` — an unfrozen Q is a moving goalpost.

### BLOCKING-3 — Q is not schedule-independent, so `argmin_S M(Q(W), R(S); θ)` is ill-posed

`W_VECTOR §2.1` (`:233-235`) defines the split precisely and correctly:

> **Q** = "derived scalars computable from W … **Independent of the schedule**"
> **R** = "the quantities the kernel's knobs actually move: **readiness parallelism P, unblockable
> fraction u(t), protocol op count A, scatter width L**, injection depth d, reserved consumers C, quota."

Then `§2.2` puts **P and u(t) in Q as Q6** (`:406`) and **A and L in Q as Q9** (`:524`). The same
document assigns the same four quantities to both vectors, twelve lines apart. Worse, `§2.2 Q6`'s
own table derives `P` **from the schedule**: *"DP/EP M15, S=2 slabs → P = 2; TP8 CDAR, 256-row
slabs → P = 64"* — `S` (`K0P6_M15_SLABS`) and `slab_rows` are knobs (`S_VECTOR §2.4`), so Q6 is a
function of S. The same defect hits Q2 (`θ_flop` is admitted to be *"a function of (phase,
schedule), not of the hardware alone"*, `:312-314`), Q5 (`Σ` is *measured* by removing CTAs — a
schedule action) and Q3 (`ρ = C_phase/X`, and `X` depends on the K1 transport choice).

Consequence: **you cannot hold Q fixed and vary S**, which is exactly what FQ-1/2/3 require and
exactly what the argmin formulation means. At least 4 of 10 coordinates move when you turn a knob.

> **FIX.** Demote `P`, `u(t)`, `A`, `L`, `Σ`, `φ_p` to **R** (where `W_VECTOR §2.1` and
> `S_VECTOR §0.3` already put them). Keep in Q only what is computable from `(W, θ)` before a
> kernel exists: `T_eff = φ·T`, `k`, `B_bnd`, `C_phase` (at a declared canonical schedule),
> `ρ`, `σ`, `λ/m`, `co` (a property of the *topology+phase*, not of the pool). That is **eight**,
> which also resolves BLOCKING-2's arity fight in favour of `COST_MODEL`.
> Then re-state the contribution as: *W → Q (arithmetic) → thresholds on Q select R → R is
> realized by S*. The three-layer version is defensible; the two-layer version is not.

---

## 2. Does every ladder phase serve the contribution? — **Phase 2 does not test it at all.**

### BLOCKING-4 — FQ-1/2/3 are cited four times and budgeted zero times

`EXPERIMENT_LADDER` references the falsifiers at `:12` (inherited), `:581` (a degradation outcome)
and `:926` (the kill criterion for "the model itself"). They appear in **no cell** of `§3.3`
(B1–B9), **no cell** of `§3.4` (K1–K7), **no row** of the `§2.3` prediction table, and **no line**
of the `§3.8` budget. The campaign's kill criterion for its own central hypothesis has no
instrument, no arm, no node-hour, and no session.

They are also nearly free, which makes the omission worse:

| falsifier | what it needs | nearest planned cell | gap | cost |
|---|---|---|---|---|
| **FQ-1** (fill vs batch) | (φ=0.5, T_pad=4096) vs (φ=1.0, T_pad=2048), matched at 2,048 real rows; compare `C*` and `d*` | K2 has arm (a), K1 has arm (b) — **different campaigns, different sessions** | the paired same-session discipline of `§3.2` R-K1 is violated across cells | **≈0.4 h** as one 2-arm alternating campaign |
| **FQ-2** (co-residency) | same rig, same message size, ± a co-resident MFMA body of **matched duration** | B4 sweeps depth × co, but at `ρ≈3` (not matched duration) and across two harnesses | "matched duration" is the whole point and is unspecified | **≈0.2 h** as two extra B4 arms |
| **FQ-3** (granularity decomposition) | signal count at **fixed** geometry, then geometry at **fixed** signal count | **none** | B3 moves `slab_rows`, which moves signal count *and* geometry *and* `P` together — the exact confound FQ-3 exists to break | **≈0.3 h** (certify every 1/2/4 slabs at `slab_rows=256`) |

**≈0.9 node-hours buys the campaign's central claim an actual test.** It currently spends 4.3 h on
the TP8 boundary track alone.

> **FIX.** Add a cell class **`FQ`** to `§3.3`/`§3.4` with three rows, budget 0.9 h, and move it to
> **rank 2** in `§3.7` (behind B1 only). Add three rows to `predictions.jsonl` whose `statement`
> is H-Q itself. Without this the answer to "what is the contribution?" is "a sweep."

### BLOCKING-5 — the cell-ID renumbering silently deleted FQ-3's cell and the producer-order cell

`W_VECTOR §4.4` defines Tier-1 cells **K1–K7** with one meaning; `EXPERIMENT_LADDER §3.4` defines
**K1–K8** with a *different* meaning, and `§3.1` reuses `W_VECTOR`'s collinearity table with the
ladder's numbering:

| id | `W_VECTOR §4.4` | `EXPERIMENT_LADDER §3.4` |
|---|---|---|
| K1 | T-sweep (α) | T-sweep (α) — *agrees* |
| K2 | routing / skew | **fill φ** |
| K3 | **fill φ** | routing / skew |
| K4 | `k_inner` ratio → now ladder **B1** | consuming pool `C` |
| K5 | message size → now ladder **B2** | the LAW-37 doors 2×2 |
| K6 | **granularity decomposition (FQ-3)** | **top-k** |
| K7 | gfx942 → now ladder **K8** | decode DP |

`EXPERIMENT_LADDER §0` states the binding rule: *"Where this document and a sibling disagree and
the disagreement is not named here, the sibling wins."* The disagreement is **not named**, so by
the ladder's own rule `W_VECTOR`'s K2/K3/K6 win — and the ladder then executes a different grid.
The casualty is concrete: **`W_VECTOR`'s K6 (granularity decomposition, the FQ-3 cell) has no
successor anywhere in the ladder.** It was not cut for a reason; it was overwritten by a
renumbering.

A second casualty: `KNOB_INVENTORY §PART D` ranks **K2 producer order** as the **#4 highest-leverage
knob** — *"the cheapest lever in the program (one index map) with the largest interaction term:
≈0 alone, −144 µs with certification"* — and prescribes *"sweep all four `order.cuh` maps × K3 ×
SKEW."* In the ladder, order appears only as a secondary axis inside **K6 top-k** (gated on X3) and
implicitly inside K3's arms. **There is no order × certification cell.** That is the single
cheapest `S`-interaction law the project owns and it is unbudgeted.

> **FIX.** (a) One cell-ID registry table in the master doc; renumber once; every sibling edited.
> (b) Restore the granularity-decomposition cell as `FQ-3`. (c) Extend K5's 2×2 (depth × cert) to a
> 2×2×2 with `order ∈ {tile-major, nc-major}` — same binary count, +8 campaigns ≈ 0.4 h — which
> tests LAW-37's substitutes question *and* Part-D law #4 *and* gives the paper its cleanest
> three-way interaction figure.

### MAJOR-6 — the teaching list has no coverage audit; 2 of the 10 Part-D laws have no cell

`KNOB_INVENTORY §PART D` (`:388-419`) is already the best draft of "the list of what we'd like to
teach people": ten knobs, each with an explicit **law sentence**. Mapping them onto the ladder:

| Part-D law | ladder cell | status |
|---|---|---|
| 1 `C*` set by consumer-work : producer-shadow | K4, B5 | ✅ |
| 2 depth binds iff co-resident | B4, B1 | ✅ |
| 3 granularity free iff it moves only A | B3 (confounded) + **FQ-3 missing** | ⚠️ **partial** |
| 4 order sets certified-prefix arrival | — | ❌ **none** |
| 5 K1 by arch × message size | B2, K8 | ✅ |
| 6 carry vs idle vs consume | K4 (+S2 control), K3 | ✅ |
| 7 quota* = shadow ÷ consume rate | K4 (`flush_rows`) | ✅ |
| 8 row-proportional cost / fill | K2 **(gated X1)** | ⚠️ gated |
| 9 replication pays iff per-chunk coverage > θ | K3 + E-G **(Phase 4b, conditional)** | ⚠️ conditional |
| 10 fuse the collective iff … | B1, K1 | ✅ |

> **FIX.** Put this table in the master doc as **the teaching list**, with a `cell` column and a
> `verdict` column left blank. Pre-register the **law-card titles** in Phase 1 next to
> `predictions.jsonl`. That converts the teaching list from a hoped-for byproduct into a
> pre-committed artifact and answers the expert's request literally: *here is the list; the
> campaign fills in the verdicts.*

---

## 3. Laws, or a benchmark sweep? — **currently ~1/3 laws, 2/3 project status**

### MAJOR-7 — taxonomy of the 26 registered predictions

`§2.3` registers **26** rows (I count P1, P1b, P1c, P1d, P2, P2c, P3, P3b, P3c, P4a, P4c, P5d, P6,
P6c, P7a, P7c, P8, P9, P9adv, P11, PF8, PHY, PE1, PE4, PE5, PF4; the ladder summary says 25).
Sorted by whether they are of the form **"toggle a workload property → a schedule property
changes"**:

| class | rows | n |
|---|---|---:|
| **W → S laws** (the contribution) | P1 (T→fuse), P1c/P1d (φ→cost), P3/P3b (λ, arch→K1), P4a (top-k→readiness), P5d (co→depth), P6c/P7c (σ→pool), P7a (φ→C*), PHY (N→transport) | **11** |
| **our-kernel status** (is our thing good yet) | P2, P2c, P6, P4c, PF8, PE1, PE4 | 7 |
| **instrument / hardware calibration** | P3c, P8, P11, PF4 | 4 |
| **not about the kernel schedule at all** | P9, P9adv (placement — `§6.5` states it is "outside the kernel"), PE5, P1b | 4 |

11/26 is not fatal, but the ordering in `§3.7` amplifies the problem: **rank 1 is B1** (an
engineering-status question — "was our fused CDAR ever tested at a reachable ratio?"), **rank 7 is
K2** (the *only* controlled test of φ, the mediating quantity behind the flagship draft8 result),
and rank 13 is K6 (described in the same document as *"the cleanest sign-flip law in the
campaign"*). The ordering optimizes for unblocking the project, not for defending the contribution.

> **FIX.** Publish two orderings and be explicit about which one is running: an **engineering
> order** (unblock B1, X1, X2 — what the team needs) and a **contribution order** (FQ-1/2/3, K2,
> K6, B2, K4 — what the paper needs). Run the contribution order for Phase 2 and treat B1/E-B1 as
> Phase-0 unblockers, which is what they actually are.

### BLOCKING-8 — three of the four sign-flip laws are gated on unbuilt actuators

`§0.3` is admirably honest about this and then does not act on it:

* **X1** — `K0P6_M24_FILL` **has never compiled**; it gates **the entire φ axis**, i.e. K2, P1c,
  P1d, P7a and FQ-1. Size **M**.
* **X2** — serving M15 **fails the MoK gate at T=2048/1024**; it gates **the entire T axis** on the
  shipped body, i.e. K1, K7 and P1. Size **M**.
* **X3** — `TOPK != 8` is a hard reject at `k0pf6gm_device_tile_m15.hip:1230`; it gates K6/P4a,
  *"the cleanest 'sign flips on a workload parameter' result the campaign owns."* Size **M**.

If X1–X3 all slip (three **M**-sized ports inside a 14-day plan that also builds E-B1/E-B2 and
E-B3), the surviving W→S laws are **λ→K1** (B2) and **co→depth** (B4). Both are *already banked*
(`TP8_STATE §5 M5`; LAW-8 vs LAW-20). **The campaign would spend ~38 node-hours to re-confirm two
laws it already has.** That is precisely the "expansive, unclear contribution" outcome.

> **FIX (three parts).**
> 1. Declare a **minimum viable law set** and make it a gate: *no Phase-2 node-hours are spent
>    until at least two of X1/X2/X3 have landed.* The ladder already puts X1/X2 on day 1 but does
>    not condition anything on them.
> 2. Add the **ungated fallback** for the φ axis, which exists and is not used: `E-F`'s concurrency
>    ladder produces φ ∈ {0.376 … 0.985} *emergently* at C ∈ {32, 64, 128, 256, 512}. It is a
>    weaker (uncontrolled) actuator, but it is the actuator the flagship claim currently rests on
>    and it is settable **today**. Budget it in Phase 4**a**, not 4b (see MAJOR-13).
> 3. Re-scope K6 honestly: top-k is **MODEL-FIXED** in `W_VECTOR §1.2` (Group E). K6 is therefore
>    not a workload toggle an operator can perform — it is a **model-generalization** law
>    ("our schedule rules port to top-2 and top-4 MoEs"). That is a *better* selling point than
>    the ladder currently claims, but only if stated as generalization rather than as ablation.

---

## 4. Is the mediating-quantity claim load-bearing, or decorative? — **decorative today, fixable cheaply**

**The test:** name one decision in the ladder that is made *by Q* rather than by direct
measurement of the cell. I could not find one. Every Phase-2 cell measures the boundary directly
and then checks a Q-derived number against it. That is *validation* of Q, not *use* of Q — and
validation is only meaningful if the hold-out is clean, which brings us to:

### BLOCKING-9 — the fit/hold-out splits in `COST_MODEL §4.1` and `EXPERIMENT_LADDER §4.1` contradict, and the ladder's hold-out leaks

`COST_MODEL §4.1` **CAL-T** (calibration, θ fitted here) =
`transport-only: {atomic,towers} × slab_rows {128,256,512,1024} × depth {4,8,16,32} × bps {1,2,4} × tokens {256, 2K, 4K, 16K}`.

`EXPERIMENT_LADDER §4.1` **HELD OUT** = *"B2's middle sizes {15, 33, 60} MB; **B3** at T ∈ {2048,
4096}"*, where B3 = `slab_rows ∈ {64,128,256,512} × tokens ∈ {2048, 4096, 16384}`, transport-only
included.

**DERIVED (review):** at `kRowBytes = 14,336 B` (`m25_boundary_bench.hip:59`, quoted in
`W_VECTOR §1.2` E3), 4,096 tokens = **58.7 MB** and 16,384 tokens = **235 MB**. So:

* CAL-T fits θ on `tokens = 4K` — which is inside the ladder's held-out "middle sizes {…, 60 MB}".
* CAL-T fits θ on `slab_rows × tokens ∈ {2K, 4K}` — which is *exactly* the ladder's held-out B3.
* `COST_MODEL`'s own **HO-2** ("slab sweep at T=4096 and T=2048") is therefore **also inside
  CAL-T**. The contradiction is internal to `COST_MODEL` as well as across documents.

A leaked hold-out destroys the only quantitative evidence that Q is load-bearing. And the ladder's
binding rule ("the sibling wins where the disagreement is not named") means `COST_MODEL`'s
leaking CAL-T formally governs.

> **FIX.** Delete `COST_MODEL §4.1`'s CAL/HO tables and replace with a pointer to
> `EXPERIMENT_LADDER §4.1` as the single authority. Restate CAL-T as **endpoints only**
> (`tokens ∈ {256, 16K}`, `slab_rows ∈ {128, 1024}`, `depth ∈ {4, 32}`) so every interior point is
> genuinely held out. Commit the split file with a git sha **before** Phase 2, which `§4.1` already
> demands and which nothing currently enforces.

### MAJOR-10 — P11 (the four-door screen) has no adjudicating arms because §3.6 prunes them

`§2.3` registers **P11: "every zero-door arm measures inside ±1σ of its control"**, and `§7.1`
makes it a kill criterion. But `§3.6`'s pruning rule is *"a candidate skeleton that cannot name its
door does not get built"* (`S_VECTOR §1.2`), and the S-arm table keeps only *"1 archival point"*
(S5) and *"1 known-bad control"* (mode-3). **You cannot validate a screen whose rejects you refuse
to run.** The 11/11 record is retrospective; a forward test needs forward arms.

> **FIX.** Name **three** deliberate zero-door arms in `arms.jsonl` with `campaign_role:
> "screen_validation"` (S5 mode-16, mode-3 XCD confinement, and the per-task VMEM drain deletion
> — all three are already-built, already-measured, ≈0.2 h total) and list them as P11's
> `adjudicating_arms`. This is the campaign's **cheapest publishable law** and it currently has no
> evidence path.

### MAJOR-11 — the C* falsifier is arithmetically ill-defined

`§2.3 P7a` predicts `C* = 24 ± 4` with falsifier *"C* moves >1 grid step"*; `§4.2` sets
boundary-location error `≤ 1 grid step`; the C ladder in `§3.4` K4 is `C ∈ {0,8,16,24,28,32}`.
**DERIVED (review):** the step size on that ladder is non-uniform — 8, 8, 8, 4, 4 — so "±4" is one
step at the top and half a step at the bottom, and "21 predicted vs 24 measured = one grid step"
(`§4.2`, `§6.1`) is true only on the sub-grid `{16, 24}`. A falsifier whose truth value depends on
which end of a non-uniform grid you are standing on is not a falsifier.

> **FIX.** State the bar in **absolute CTAs** (`|C*_pred − C*_meas| ≤ 4 CTAs`) and make the ladder
> uniform where the prediction lives (`C ∈ {16, 20, 24, 28, 32}`). Costs nothing; K4 already runs
> K=4 there.

---

## 5. Contribution-damaging defects outside LENS 1's core, that will be read as contribution defects

### BLOCKING-12 — the flagship e2e win cell is planned in a configuration the draft itself declares inadmissible

`§5.3` derives (correctly, and to its credit) that the banked c512p cell **fails two of the
campaign's own gates**: Little's occupancy **0.899** (gate 0.95) and waves **`2048/512 = 4.0`**
(gate 20), and recommends either `num_prompts ≥ 10,240` (**≈2 h/arm**) or the label `R-L3 FAIL`.

`§5.2` then budgets **E-B at 2.8 h for 5 pairs (2 banked)** — i.e. at the inadmissible
configuration. The `+8.40%` that the experts already saw would ship **self-labelled FAIL**.

> **FIX.** Budget **one** 10,240-prompt confirmatory c512p pair explicitly (**≈4 h**, and say so),
> and re-anchor the throughput headline on the concurrency ladder at C ∈ {32, 128} where
> `waves ≥ 20` costs 640 / 2,560 prompts. This is the ladder's own recommendation (`§5.3`) not
> carried into its own budget.

### MAJOR-13 — the campaign's single most quotable number is in the *conditional* phase

`§5.2` puts `E-F` (concurrency ladder C ∈ {64,128,256}) in **Phase 4b, "conditional on Phase 3
passing"**, while quoting `WORKLOAD_EVIDENCE` H3 in the same line: *"the crossover concurrency is
the single most quotable number the campaign could produce, and we cannot currently state it."*
The most quotable number is thereby made contingent on the cost model meeting its accuracy bars.
That is backwards: the crossover concurrency is a **measurement**, valuable whether or not M
predicts it, and `W_VECTOR §1.3`'s DERIVED finding is that the flagship topology law is currently
measured **on a diagonal, not a 2×2** — E-C and E-F are what turn it into a 2×2.

> **FIX.** Move `E-C` and `E-F` into Phase 4a; move `E-A` out (see §6.3). Net node-hours unchanged.

### MAJOR-14 — Phase 4's PRIMARY metric contradicts the protocol the same section says wins

`§5.4` states the open contradiction honestly and resolves it: *"Until the operator accepts,
BENCHMARK_PROTOCOL wins"* (TTFT p50/p99 primary). `§5.2` then sets E-A's **PRIMARY** to the
`goodput_in(a,b)` attainment curve and E-B's to M1 — the *proposed amendment*, not the standing
protocol. Two sections, opposite rulings, four lines apart.

> **FIX.** Get the operator decision **before** Phase 4 — it is one decision, zero node-hours, and
> `METRICS §6.2` already ships the replacement text. Until then, PRIMARY = the standing protocol
> and the amendment metrics are reported as GUARDS.

### MAJOR-15 — day-2 runs precede the day-4 registration commit, violating the gate that makes it science

`§0.1`: *"Phase 2 may not start on a cell whose prediction is unregistered."* `§7.3`: **day 2** runs
Action 2 = **B1**, a Phase-2 cell (`§3.3`); **day 4** commits "Phase-1 registration with banked θ."
The highest-information cell in the campaign produces numbers two days before the artifact that
makes them evidence rather than a number (`§0` binding rule 4, verbatim).

> **FIX.** Commit `predictions.jsonl` with banked θ on **day 1**. It is a CPU-only commit; `§0.2`
> budgets Phase 1 at **0 node-hours**. There is no reason it is on day 4.

---

## 6. What to CUT — the minimum campaign that still answers the experts

### 6.1 The five laws worth defending

Chosen because each is (i) a genuine `W → S` sign or threshold, (ii) has a settable actuator or a
named unblocker, and (iii) maps to a Part-D teaching-list law and to a shippable HK primitive:

| # | law | mediator | cell(s) | node-h |
|---|---|---|---|---:|
| **L1** | *Fuse at all iff the fixed-cost share `α < ≈0.41`, i.e. `T_eff > T* ≈ 1,300–1,800`* | `T_eff = φ·T` | K1 (+ decode-shaped T folded in) | 0.9 |
| **L2** | *Bound injection iff the fabric shares the device with compute; the threshold is contention, not bytes* | `co` | B4 + **FQ-2** | 0.9 |
| **L3** | *Choose RMW vs store+local-reduce by the arch's atomic:store ratio **at the deployment message size**; `b* ≈ 32.8 MB`* | `λ` | B2 (+ θ-F13 RCCL decode ref) | 1.3 |
| **L4** | *A reserved CTA pays 8–9 µs unless it consumes a certified prefix; `C*` is derived from `(Σ, P, t50)`, never carried* | `Σ, P` | K4 + K5(×order) | 3.1 |
| **L5** | *The DP topology's per-real-token cost inflates by `1/φ` and TP8's does not — the regime flip is a topology tax, not kernel quality* | `φ` | K2 (X1) + **FQ-1** + E-C/E-F | 2.4 (+e2e) |

Plus **FQ-3** (0.3 h) as the sufficiency test and the primitive-API question in one arm.

### 6.2 The minimum campaign — **≈20 node-hours, two overnights**

| block | contents | node-h |
|---|---|---:|
| P0-Z | the free CPU triple (`§1.1`) | 0.0 |
| P0-A′ | **only** θ-F13 (RCCL at decode sizes) + θ-F5/6/7 (size × transport), endpoints for CAL-T | 0.8 |
| P0-B′ | E-B1/E-B2 device phase ledger + duty histogram, E-B3 transport argv | 1.3 |
| B1 | `k_inner` ratio sweep (unblocks the whole TP8 reading) | 0.5 |
| FQ | FQ-1, FQ-2, FQ-3 | 0.9 |
| B2, B4 | λ and depth×co | 1.9 |
| K5(×order), K4 | doors 2×2×2 + `C × flush_rows` in two W-cells | 3.1 |
| K1 | T ladder incl. decode-shaped T (X2) | 0.9 |
| K3′ | skew reduced to {balanced, aggregate-hist, adversarial} × {S4-C24, S2-C64} | 1.2 |
| K2 | φ at fixed `T_pad` (X1) | 1.6 |
| E2E | E-C (TP8@C=512) + E-F (concurrency ladder) + one admissible c512p pair | ~7.5 |
| | **total** | **≈19.7** |

### 6.3 The cut list, with the cost of each cut stated

| cut | node-h saved | why it can go | what is lost |
|---|---:|---|---|
| **B7** `B.hyb.world` | 0.3 | world size is **HW-FIXED at 8** (`W_VECTOR §1.2` A3, "8 only"); PHY's prediction is pure byte arithmetic `2(N−1)/N`, already computed | an axis with **no deployment cell**; publish the arithmetic instead |
| **B8** `B.tp8.shexp` | 0.4 | adjudicates **no registered prediction**; the atlas itself says it covers ~32–34% of the prize and *"must never be sold as the answer"* | an engineering arm, not a law — move to the build track |
| **B9** + **X5** | 0.3 + an **M–L** build | the ladder concedes: if X5 slips, precision is *"a single declared coordinate, not a swept one"* | the largest engineering item serving the smallest cell; F2 stays a declared coordinate |
| **K7** `K.decode.dp` | 0.4 | shares actuator (X2), instrument and body with K1; `§6.3` already concludes there is **no DP decode kernel problem** — the lever is `max_num_batched_tokens` | nothing: **fold T ∈ {128, 256} into K1's ladder** |
| **K8** gfx942 | 1.0 + access risk | access-gated; predicted outcome is *"a simplification for the PR"*; arch is not in the experts' question list | defer; if the node materializes it is one run |
| **E-A** as a *run* | 4.6 | the −21.17% **"cannot be read as a kernel delta"** (`§6.2`, `W_VECTOR §1.3`: the arms differ in Group-H coordinates 4096/128 vs 16384/2048). The valuable output — *"TP8 wins every SLO tighter than ~3.5 s TTFT; we win all tails"* — is **already derivable from banked data** (`METRICS §2.2`) | re-run only if the SLO curve is contested; **convert E-A to an analysis deliverable at 0 h** |
| **E-D** decode e2e | 4.5 | `§8` states *"no decode claim in this ladder rests on a measurement"* and PE4 predicts **we tie or lose**; the mega runs decode **at prefill shape** post-M23 | a 4.5 h cell whose registered prediction is "we lose"; keep the *prediction* published, drop the run until D1 exists |
| **B3, B5, B6** | 1.2 | B3 is confounded (see BLOCKING-5) and is superseded by FQ-3; B5/B6 are TP8 transport engineering | TP8 transport tuning; re-fund only if B1 shows `h ≥ 20%` |
| **Phase 4b** as a phase | — | E-F promoted to 4a (MAJOR-13); E-G placement folded into K3 + `eplb0_predict.py`'s own gate | the conditional structure, which was hiding the best number |

**Total cut ≈ 17–18 node-hours (~45%), and the two largest cuts remove arms whose own registered
predictions are "our thing loses."** Cutting them makes the deck *stronger*, not weaker.

### 6.4 What must NOT be cut, despite being tempting

* **P0-Z** — free, and P0-Z3 (compile one decode tile body, read occupancy) is *"the single largest
  structural fork in the campaign and it costs a compiler flag."*
* **E-B1/E-B2** (phase ledger + duty histogram) — it removes **both** named fudge factors (`I₀`,
  `κ_couple`) with one instrument and turns C6's duty-cycle reframe from argument into experiment.
* **E-E** (open-loop) if any e2e is kept — `METRICS §2.2` calls the closed-loop dummy share *"the
  campaign's biggest single exposure"*; PE5 is the only thing that resolves it. If budget forces a
  choice, cut E-D before E-E.
* **The retraction/negative ledger** — `§8`'s "what this ladder refuses to promise" is, honestly,
  the most credible page in the draft and should be **in the deck**, not an appendix.

---

## 7. Does the HK-PR story connect to the experiments? — **No. It floats free.**

### BLOCKING-16 — BRIEF output (7) has no owner document

`BRIEF.md:126-127` requires: *"Expected emergent abstractions — what patterns fall out → the shape
of the HipKittens PR; the 'list of what we'd like to teach people.'"* Grepping the `Deliverable:`
tags: `W_VECTOR` = (1), `S_VECTOR` = (2), `COST_MODEL` = (3), `METRICS_PROTOCOL` = (4),
`EXPERIMENT_LADDER` = (5)+(6). `OVERLAP_ATLAS` claims no output number. **(7) is unowned.**

`METRICS §0` (`:50-52`) claims the **law card** is "the concrete form of" the teaching list. A card
template is a *format*; it is not a list, and it is not a PR. Meanwhile the BRIEF already names
nine extracted primitives and the repo ships ten headers
(`include/cdna4/ops/group/distributed/{completion,counter,credit,lifetime,order,packet,peer,roles,slab,sync}.cuh`).
**No cell in the ladder names a primitive.** No manifest field records which primitive an arm
exercises. There is no path from a Phase-2 result to an API change.

### The connection exists and is one column wide

Three PR-shaped findings are already derivable from the draft and would be *strengthened* by the
cells that exist — they just are not written down as PR outputs:

| primitive | law from the campaign | API consequence | which cell decides it |
|---|---|---|---|
| `credit.cuh` (`throttled_accumulate_bf162`, depths {0,4,8,16,32}) | depth binds **iff `co=1`**; `d*` predicted to move **down** to 2 in the fused arm and on gfx942 | the bound needs a **`co` scope note** in its contract, and **depth 2 does not exist** — the `g` `0x300` selector is a **2-bit field and full** (`moe_mps_adapter.cuh:400-410`), so adding it is an **encoding change**, not a parameter | B4, FQ-2, (K8) |
| `slab.cuh` (`certify_slab`, `bounded_wait_slab_into`) | granularity is free when it moves only `A`, expensive when it moves `P` or transfer geometry | the primitive currently **conflates signal count and transfer geometry in one parameter**; it should take **two** | **FQ-3 — the cell the renumbering deleted** |
| `order.cuh` (four branch-free maps) | order alone ≈ 0; **order × certification = −144 µs**; and `Φ_slab(2)=0.25` vs `Φ_natural(k)=1/(k+1)` **reverses at k ≤ 2** | order selection should be **a function of top-k**, exposed as a policy, not a compile-time default | K5(×order), K6 |

That is the PR. It is three sentences and it falls out of three cells — **two of which the current
ladder does not run.**

> **FIX (BLOCKING).** (a) Write `design/PR_SHAPE.md` as output (7). (b) Add a
> **`primitive_touched` + `api_implication`** field to `arms.jsonl` and a column to every cell in
> `§3.3`/`§3.4`. (c) Pre-register, in Phase 1, which primitives the campaign expects to
> **confirm / generalize / retire** — a wrong pre-registration here is as publishable as a right
> one, and it is the artifact that makes "what does the HK PR look like?" answerable on the day
> the campaign ends rather than three weeks later.

---

## 8. Credibility defects a hostile reviewer will find in ninety seconds

### 8.1 BLOCKING-17 — the node-hour budget does not add up

`§1.7` Phase-0 table: `0.0 + 1.6 + 1.3 + 1.8 + 1.9 + 1.3 (retries) = ` **7.9**, stated as **≈6.0**.
**DERIVED (review).** And P0-A's own sizing is internally inconsistent: `§1.2` derives *"≈126
invocations × ~1 min"* = **2.1 h** and then states **"≈1.6 node-hours with retries"** — the retry
allowance makes it larger, not smaller. Correcting both: **Phase 0 ≈ 8.5 h** (7.1 + 20%).

`§0.2` states **Phase 4a = 16.5**; `§5.2` states **"Phase 4a total ≈ 16.7"** (4.6+2.8+2.0+4.5+2.8,
which checks). Headline **≈38.5** should read **≈40.6–41.2**.

This matters beyond arithmetic: the document's **binding rule 1** is *"arithmetic performed here is
tagged DERIVED with its inputs shown."* A budget table that fails its own rule undermines the
tables that do not.

> **FIX.** Recompute; state the total as a **range with a retry allowance**, e.g. "≈41 h ± 15%,
> four-to-five overnights"; and add the E-B 10,240-prompt pair (BLOCKING-12). Under the minimum
> campaign of §6.2 the honest number is **≈20 h, two overnights**, which is a far better story.

### 8.2 MAJOR-18 — "hundreds of kernels" is promised and ~250 invocations are delivered

`BRIEF.md:105-108` requires "hundreds of kernels" as points in a knob space; `S_VECTOR §5.2`
delivers **≈180 rig arms from 7 skeletons and ≈12–16 HSACOs**, `§0.2` says **~250 rig invocations +
~164 MoK campaigns**. That is defensible and honest — but the *pruning arithmetic* is the selling
point (`§3.6`: full crossing = **8,640 arms/cell → 51,840 campaigns ≈ 2,679 node-hours**, a **288×
reduction**) and it is buried in a table. **Lead with it.** "We did not run hundreds of kernels; we
proved 288× of them were unnecessary and said why" is a contribution sentence in itself.

### 8.3 MAJOR-19 — the prediction table's provenance is not marked

`§0` binding rule 1 requires every number to carry a source, and `§2.3` mostly complies at the
number level. But it does not mark **which predictions are `COST_MODEL`'s (P1–P11 + letters) and
which are new in the ladder** (PF4, PF8, PHY, PE1, PE4, PE5, P1c, P1d, P3c, P4c, P5d, P6c, P7c,
P9adv). A reader tracing "P5d" to `COST_MODEL §3/D5` finds P5a–P5d — fine — but "PHY", "PE1",
"PE4", "PE5" and "PF4" have no upstream. Add a `source` column (`COST_MODEL D-n` | `this doc`).
Also note the numbering gaps (no P5, no P10, no P4b in the ladder's table) will read as omissions.

### 8.4 MAJOR-20 — the campaign's own headline framing survives on a number the draft calls unquotable

`§8` correctly refuses three claims, including *"any reading of the c32p −21.17% as a kernel
delta."* But `§5.2` E-A still labels the cell *"the current loss"* and `§6.2` is titled *"the open
loss, −21.17%."* If the number cannot be read as a kernel delta, the section titles must not use
it as one. This is exactly the failure mode `WORKLOAD_EVIDENCE §0.2` exists to prevent
(*"deleted-but-cited numbers keep reappearing"*).

---

## 9. Issue register

### BLOCKING (would sink the campaign or the credibility of its claims)

| # | issue | where | fix |
|---|---|---|---|
| B1 | three incompatible contribution sentences (10-scalar H-Q / 8-scalar Q + four doors / two topology sentences) | `W_VECTOR §2.4, §5`; `COST_MODEL §0.3, §6:1269, :1276-1280` | one sentence in the master doc, repeated verbatim in all siblings; declare the fallback explicitly |
| B2 | Q is two different vectors with different arity; `κ_sched` is a mediator in one doc and a banned validity gate in another | `W_VECTOR §2.2` vs `COST_MODEL §0.3`; `W_VECTOR :630-631, :771` | one canonical Q table in `COST_MODEL`; split out a validity-gate vector **V**; freeze arity in Phase 1 |
| B3 | Q is not schedule-independent (`P`, `u(t)`, `A`, `L`, `Σ`, `φ_p` are in **both** Q and R), so `argmin_S M(Q, R(S))` is ill-posed and FQ-* is unrunnable as written | `W_VECTOR :234` vs `:406`, `:524` | demote those six to R; keep 8 W-computable coordinates in Q |
| B4 | **FQ-1/2/3 — the campaign's central falsifiers — have zero cells, zero predictions rows and zero node-hours** | `EXPERIMENT_LADDER :12, :581, :926` vs `§3.3/§3.4/§3.8` | add cell class `FQ`, 0.9 h, rank 2 in `§3.7`; three rows in `predictions.jsonl` |
| B5 | cell-ID renumbering between `W_VECTOR §4.4` and `EXPERIMENT_LADDER §3.4` silently deleted the granularity-decomposition cell (FQ-3) and left producer-order (Part-D law #4) with no cell | both, unnamed disagreement despite `§0`'s "the sibling wins" rule | one cell-ID registry; restore FQ-3; extend K5 to depth × cert × order |
| B6 | fit/hold-out leak: `COST_MODEL §4.1` CAL-T fits θ on `tokens ∈ {2K, 4K}` × `slab_rows`, which the ladder holds out (B3, B2 middle sizes); `COST_MODEL`'s own HO-2 is inside its own CAL-T | `COST_MODEL §4.1` vs `EXPERIMENT_LADDER §4.1` | ladder's split is the single authority; CAL-T restricted to endpoints; commit split with a sha pre-Phase-2 |
| B7 | three of four sign-flip laws gated on unbuilt actuators (X1 never compiled, X2 fails the gate, X3 hard-rejects); if they slip the campaign re-confirms two banked laws | `EXPERIMENT_LADDER §0.3` | gate Phase 2 on ≥2 of X1/X2/X3; add the ungated emergent-φ fallback (E-F); re-scope K6 as model-generalization |
| B8 | the flagship e2e win cell (E-B, c512p) is budgeted in the configuration the draft itself declares `R-L3 FAIL` (Little 0.899, waves 4.0) | `§5.3` vs `§5.2` | budget one 10,240-prompt pair (~4 h); re-anchor on C ∈ {32,128} |
| B9 | BRIEF output (7) unowned — no HK-PR/teaching-list document, no primitive column on any cell, no path from a result to an API change | all six design docs | write `design/PR_SHAPE.md`; add `primitive_touched`/`api_implication` to `arms.jsonl` and every cell row; pre-register confirm/generalize/retire |
| B10 | Phase-0 budget table sums to 7.9 (stated 6.0); P0-A states 1.6 h against its own 126-invocation × 1-min derivation (2.1 h); `§0.2` 16.5 vs `§5.2` 16.7 ⇒ headline 38.5 should be ≈41 | `§1.2`, `§1.7`, `§0.2`, `§5.2` | recompute; state a range with retry allowance; or adopt the ≈20 h minimum campaign |

### MAJOR (weakens it)

| # | issue | where | fix |
|---|---|---|---|
| M1 | only 11 of 26 registered predictions are `W → S` laws; the rest are project status, instrument calibration, or non-kernel | `§2.3` | publish two orderings (engineering vs contribution); run the contribution order in Phase 2 |
| M2 | `§3.7` ordering puts engineering-status B1 at rank 1 and the flagship-mechanism cell K2 at rank 7, the cleanest sign-flip law at 13 | `§3.7` | reorder; treat B1/E-B1 as Phase-0 unblockers |
| M3 | 2 of the 10 Part-D teaching-list laws have no cell (order × certification; granularity decomposition), 2 more are gated/conditional | `KNOB_INVENTORY §PART D` vs `§3.3/§3.4` | law-coverage table in the master doc; pre-register law-card titles in Phase 1 |
| M4 | P11 (four-door screen) has no adjudicating arms because `§3.6` prunes zero-door arms by rule | `§2.3` P11 vs `§3.6` | name 3 deliberate zero-door arms (~0.2 h) as `screen_validation` |
| M5 | `C*` falsifier "±4 / one grid step" is ill-defined on the non-uniform ladder `{0,8,16,24,28,32}` | `§2.3` P7a, `§4.2`, `§3.4` K4 | state in absolute CTAs; make the local grid uniform `{16,20,24,28,32}` |
| M6 | the most quotable number (crossover concurrency, E-F) sits in Phase 4b, conditional on Phase 3 passing | `§5.2` | move E-C and E-F into 4a; move E-A out |
| M7 | Phase-4 PRIMARY metrics contradict `§5.4`'s own ruling that BENCHMARK_PROTOCOL wins until the operator accepts | `§5.2` vs `§5.4` | obtain the operator decision pre-Phase-4 (0 h) or set PRIMARY to the standing protocol |
| M8 | day-2 B1 run precedes the day-4 registration commit, violating `§0.1`/binding rule 4 | `§7.3` vs `§0.1` | commit `predictions.jsonl` (banked θ) on day 1 — it costs 0 node-hours |
| M9 | predictions lack a provenance column; numbering gaps (no P5/P10/P4b) read as omissions | `§2.3` | add `source` column (`COST_MODEL D-n` \| `this doc`); renumber contiguously |
| M10 | section titles use the −21.17% as a kernel delta after `§8` retracts that reading | `§5.2`, `§6.2` vs `§8` | retitle ("the C=32 arm-level gap"); state the three-way decomposition inline |
| M11 | the 288× pruning arithmetic — the best answer to "hundreds of kernels" — is buried in `§3.6`'s table | `§3.6`, `S_VECTOR §5.2` | promote to the master doc's opening; it is a contribution sentence on its own |
| M12 | K6's top-k axis is presented as a workload ablation, but top-k is **MODEL-FIXED** (`W_VECTOR §1.2` Group E) | `§3.4` K6, `§2.3` P4a | reframe as model-generalization ("the rule ports to top-2/top-4 MoEs"), cite real checkpoints' top-k |

---

## 10. The one-paragraph version, for the synthesis pass

The draft's evidence discipline is publication-grade and its negatives are its best asset. Its
contribution is not yet one thing: three sentences, two Qs, a Q that is not schedule-independent,
and zero budgeted tests of the sufficiency claim it says is the point. Fix the object (one Q,
eight W-computable coordinates, everything schedule-moved demoted to R), fund the falsifiers
(FQ-1/2/3, ≈0.9 h, rank 2), restore the two cells the renumbering deleted (granularity
decomposition and producer-order × certification, ≈0.7 h), close the hold-out leak, and cut
~45% of the grid — B7, B8, B9/X5, K7, K8, E-A-as-a-run, E-D, B3/B5/B6 — most of which exists to
re-test arms the campaign itself predicts will lose. Then write the missing output (7): three
primitives, three API changes, three cells that decide them. That is a **five-law, ≈20-node-hour,
two-overnight campaign** with a single defensible sentence, a pre-registered teaching list, and a
PR whose shape is known before the runs start — which is the only version of this that survives
"very expansive, unclear what the core contribution might be."

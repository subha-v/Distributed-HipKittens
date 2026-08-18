# MECHANISM_CRITIQUE — adversarial review, Lens 3 (kernel physics)

**Scope.** Does the campaign's physics hold? Are the per-cell predicted winners consistent with
the banked laws? Does `M(W,S;θ)` capture congestion/incast or only bandwidth division? Is decode
treated with decode physics or with prefill formulas scaled down? What would a Blackwell/MoK-literate
reviewer name in the first five minutes? Which "laws" are artifacts of one workload cell?

**Documents reviewed.** `BRIEF.md`, `grounding/{BANKED_LAWS,KNOB_INVENTORY,WORKLOAD_EVIDENCE,TP8_STATE}.md`,
`design/{W_VECTOR,S_VECTOR,COST_MODEL,METRICS_PROTOCOL,OVERLAP_ATLAS,EXPERIMENT_LADDER}.md`.
Repo checks performed: `distributed-kernels/tp8_mega/m25_boundary_bench.hip`,
`tp8_mega/results/g25_0b_slab_sweep.txt`.

**Verdict up front.** The campaign's *epistemics* are excellent — registration, kill criteria,
settability gates, published negatives. The *physics* has a systematic bias: **`M` is a bandwidth-and-
capacity model wearing a congestion model's clothes.** Every congestion mechanism the ledger actually
measured (destination concentration, incast, rendezvous dispersion, burstiness, HBM collision) enters
`M` either as a scalar multiplier on compute or not at all. Three registered headline predictions
(P4a, P3/P3c decode reversal, P2's `h`) are derivable-as-wrong or unmeasurable-as-defined from the
campaign's own numbers. Nine BLOCKING items below; each has a cheap fix that fits inside the existing
Phase-0/Phase-2 budget.

---

## 0. The one-paragraph diagnosis

`COST_MODEL §1.4` writes the wire term as `X = b / BW_eff` with `BW_eff` a **product of independent
efficiency multipliers**. That form can express *how fast one sender goes*. It cannot express: (i) many
senders converging on one receiver, (ii) a fixed cost that dominates below ~30 MB, (iii) traffic that is
correct in volume but wrong in *time distribution*, (iv) the receiver's HBM. Items (i)–(iv) are,
respectively: the skew regime the campaign says dominates everything (`COST_MODEL §1.8`: 3,649 µs per
unit σ, 6× the largest schedule knob), the entire decode branch, `OVERLAP_ATLAS §C6` (ranked HIGH), and
`TP8_STATE §6/A2`. The model therefore predicts ≈0 for the class of schedule changes the vendor
measured at 65% (`LAW-51`, MORI push-combine destination interleaving; their own 822 → 498 µs fix).

---

## BLOCKING

### B1. The wire term has no incast / destination-concentration term; `n_link` is assumed = 7

**Where.** `COST_MODEL §1.4` (M4)/(M5); θ-F3 (`aggregate 7-link plateau 355.0 GB/s`) used as the
denominator of "% of ceiling" throughout; `COST_MODEL §1.8` (M13) puts *all* of σ into a
row-proportional **compute** multiplier `λ`.

**The physics that is missing.** `BW_eff = BW_link · n_link · Π η` divides bandwidth. Under routing
skew the binding resource is not the sender's egress — it is (a) the *hottest link*, (b) the receiving
rank's ingress and its L2/UMC ports, both shared by up to 7 writers. `BANKED_LAWS` already says this and
the cost model dropped it: LAW-4's scope note reads *"under multi-destination fanout or link
heterogeneity the ceiling is per-link and the hottest link sets the schedule (see LAW-51)"*, and LAW-51
records MORI's own **2-of-7-link concentration** and their **822 → 498 µs RR-interleave fix**, plus
**65% on push-combine** from destination-interleaved record order. A grep of `design/` and `grounding/`
for `incast|ingress|fan-in|concentration` returns **no bandwidth term anywhere** — fan-in appears only
as a *certificate/protocol* concept (`S_VECTOR §2.4.2`, `TP8_STATE §4.2`) and as a footnote capping
effective depth.

**Why it is blocking, concretely.** With 51.9% of expert-slot traffic landing on rank 0
(`BRIEF:70-71`), the aggregate 355 GB/s card is off by up to ~6× for the links that matter, while
`M` still prices that traffic at the 7-link plateau. Two consequences: (1) the model *cannot* predict
the value of the cheapest measured skew remedies — destination interleaving (LAW-51), m17 rotation, K6
per-destination credits — so `COST_MODEL §3/D9` correctly derives that "order changes Φ and η_layout,
neither of which is in the M(σ) multiplier" and then concludes ordering is worth ~1–1.5%; that
conclusion is *assumed by the model's structure*, not derived from physics. (2) P9's +27–46% and its
falsifier interpretation ("λ_serving ≪ λ_rig ⇒ the replay rig over-states skew") is one of at least
three readings; "the replay rig over-states skew because the *incast* term saturates and is not linear
in σ" is equally consistent and is not on the list.

**Fix (cheap, fits Phase 0).**
1. Restate (M5) as `BW_eff = min( egress_src , links_used(π,order) · BW_link , ingress_dst / share )`
   and add `η_incast(fan-in degree, destination concentration)` as a θ entry (class H).
2. Add a **fan-in cell to P0-A** on `m25_cdar_bench` (the rig where `co=0`): k senders → 1 receiver for
   k ∈ {1,2,4,7} at matched bytes/sender, plus a *concentrated vs uniform destination* arm at fixed
   world. This is argv-level work on a rig already in the budget and it is the only direct measurement
   of the mechanism that dominates the campaign's largest number.
3. Add `destination concentration` to `Q` (see B9) so the σ prediction becomes two-term:
   work-proportional (λσ) **and** link-proportional.

### B2. The wire equation has no fixed/latency term, so every decode cell uses a formula that is 25× wrong

**Where.** `COST_MODEL §1.4` (M4) `X_p = b_p / BW_eff`; used by `§3/D3`, P3, P3c, `EXPERIMENT_LADDER
§6.4`, cell B2 (`b ∈ {3.67 … 235} MB`), K7, E-D.

**The arithmetic.** At the one decode-ish point we own — 3.67 MB, 256 tokens
(`TP8_STATE §5/M5`) — (M4) with θ-F7's 117.2 GB/s predicts **31 µs**. The measurement is **801 µs
(atomic) / 1,017 µs (towers)**. The model is off by **25–33×** in exactly the regime it is being asked
to make go/no-go calls in. `D3` does introduce affine forms with intercepts `L_t = 1,001 µs`,
`L_a = 758 µs`, but those live only inside D3's crossover derivation and never enter (M4)/(M5). Note
also what those intercepts are: **larger than the entire decode transfer**, i.e. at decode the model is
100% intercept and 0% explanatory content, yet `b* = 32.8 MB` derived from that intercept gates the
whole decode branch (P3c).

**Fix.** Promote the affine form into (M4): `X = L(op, world, n_slabs, bps, protocol) + b/BW_eff`, and
calibrate `L` in **P0-A**, which already sweeps `b × transport × slab_rows × bps` — so the calibration
is free. `L` must be structured as protocol round-trips (`n_slabs × certificate latency + fan-in
publish`), not as an unexplained constant, otherwise it is a fudge factor #3 that the document has not
declared (`COST_MODEL §Binding rule 2` promises exactly two).

### B3. The one decode data point is confounded with grid occupancy — the K1 "reversal" may be the v0 artifact the project already diagnosed

**Where.** `TP8_STATE §5/M5`; `S_VECTOR §1.10` (D1 skeleton exists *because* of this reversal);
`COST_MODEL §3/D3` θ-`η_op(m) tiny`; `EXPERIMENT_LADDER §6.4` table row K1; `OVERLAP_ATLAS §9.2`.

**The confound.** The point is `tokens=256, slab_rows=64` ⇒ **4 slabs**. At `bps=2` that is **8 blocks
of a 256-block launch** — the grid is 97% idle. The project has already measured that exact
configuration class as a 4.2× artifact: G25-0b v0's 22.4 GB/s towers number was caused by
*"half-idle grid + serialized slabs + scalar single-block owner reduce"* and parallelising the reduce
gave 4.2× (`TP8_STATE §2.2`, filed as falsification **F0**). Separately, `bps 1→2 = +64%` because
"one block per slab leaves the grid half-idle". So the stated mechanism for the reversal — *"the owner
reduce is no longer amortised"* — competes with a second, already-measured mechanism (**the tower path
is occupancy-starved at 4 slabs**) and nothing distinguishes them at n=1.

**Why it is blocking.** A whole skeleton (D1), a manifest axis ("K1 is message-size dependent"), P3c,
and the decode kill criterion all inherit this point. And cell **B2's** control is *"`slab_rows`
matched"* between transports — which is the wrong control if the two transports have different optimal
geometries. Matching geometry across transports guarantees one of them runs off its own optimum.

**Fix.** In B2, run **each transport at its own optimal `(slab_rows, bps)` per message size**, and add
the geometry sweep at decode size (`slab_rows ∈ {8,16,32,64}` at 256 tokens, which restores block
count) as a separate arm. Report the reversal only if it survives at matched *block count*, not matched
slab size. Until then D1's motivating fact is `MEASURED-once, confounded`, not `MEASURED`.

### B4. `h` (hidden fraction) is defined from a wall difference, so it cannot separate hiding from the co-residency tax — and the whole TP8 track is scored on it

**Where.** `TP8_STATE §2.3` (`phased − fused = 47 µs = 2.1% hidden`); `COST_MODEL §3/D2` and P2/P2c
(`h* = 48.7%`); `EXPERIMENT_LADDER §1.3` ("the bar is restated for the whole campaign as `h ≥ h*`"),
§7.1 kill criterion for the S6/CDAR family (`h < 20%` retires it).

**The problem.** `h_wall = (phased − fused)/collective` is only a hiding measure if the co-residency tax
is identical in both arms. It is not: the *phased* arm already carries **~335 µs** of co-residency tax
(`TP8_STATE §2.3`, θ-P12) and the *fused* arm is by construction more co-resident, so
`fused = compute + (1−h)·X + I_co(fused)` and `h_wall = h − ΔI_co/X`. With `X ≈ 2,214 µs`, a fused arm
genuinely hiding **20%** (443 µs) while paying 400 µs of extra interference measures `h_wall ≈ 2%` —
numerically indistinguishable from the reported result. **The campaign is about to retire its central
TP8 thesis on a statistic that cannot tell "no hiding" from "hiding, fully cancelled by interference",
and those two outcomes have opposite engineering consequences.**

**Fix.** Define `h` from **E-B1's span overlap** (commit-span ∩ produce-span, per block class), which
the ladder is already building, and report `h_ledger` and `h_wall` side by side; the gap between them
*is* `ΔI_co` and is the quantity D2 needs. Restate the S6 kill criterion on `h_ledger`. This costs
nothing beyond the instrumentation already funded in Action 3.

### B5. `I₀`, `κ_couple`, and `I_co` are assigned to a rig whose "compute" cannot produce the mechanism they model

**Where.** `EXPERIMENT_LADDER §1.3` (P0-B closes θ-P5 `I₀`, θ-P6 `κ_couple`, θ-P12) on
`m25_boundary_bench.hip`; `COST_MODEL §1.6` states `I₀`'s scaling hypothesis as
*"∝ (peer-write request rate) × (co-resident phase's **memory intensity**)"*.

**Repo check.** `m25_boundary_bench.hip:112-131`: the co-resident "compute" is `mfma_burst`, a synthetic
loop issuing one `v_mfma_f32_16x16x32_bf16` plus **one 16 B vector load per iteration** from a small
buffer indexed **modulo `vecs`** — i.e. an L1/L2-resident working set with a fixed 1:1
MFMA:load ratio. `k_inner` (argv 6) scales FLOPs and loads **together**. Contrast the real co-resident
phase the law was measured against: M6 streams ~470 MB of weights (`COST_MODEL §1.3` roofline note,
~296 GB/s from HBM) and M7 is CTA-throughput-bound.

**Consequences.** (1) Calibrating `I₀` — whose declared scaling variable is the co-resident phase's
memory intensity — on a body whose memory intensity is fixed and non-representative produces a number
with no transferable scope; it will fail F2 (θ not constant) for structural reasons and be misread as a
model failure. (2) `TP8_STATE §6/B2`'s own killer measurement is *"an arm that varies MFMA density
independently of fragment bytes"* — **unreachable with the shipped knob**, because `k_inner` varies
both. So B2 (the resource-class hypothesis, the leading explanation for why F1–F4 all returned null)
cannot be tested by the plan as written, and B6/`fused_wave` is being asked to discriminate a hypothesis
no arm can isolate. (3) B1's ρ-sweep reaches ρ≈3 by making the compute phase *longer* without making it
*more contending*, which biases the P2 hiding measurement **optimistic** relative to deployment.

**Fix.** Add a `bytes_per_mfma` / working-set-stride argv to `mfma_burst` so duration and memory
intensity are separable axes (S, one argv + one index expression); register ρ and *intensity* as two
coordinates in B1. Calibrate `I₀` on **P0-C (the mega chassis)**, where the co-resident phase is a real
GEMM, and use P0-B only for the TP8-specific `I_co`, scoped as such.

### B6. P4a — "the cleanest sign-flip law the campaign owns" — is derived from a one-sided inequality and is refuted by the model's own θ

**Where.** `COST_MODEL §3/D4a`: *"Restructure producer order + certify iff `(S_cnt−1)/(2S_cnt) >
1/(k+1)` ⟺ at `S_cnt=2`: `k > 3`"*; registered as **P4a** (`EXPERIMENT_LADDER §2.3`, ranked #13, gated
on **X3**, an M-sized change to remove the `TOPK != 8` reject at `k0pf6gm_device_tile_m15.hip:1230`).

**The omission.** D4a compares two *shadow* terms and never subtracts the protocol cost of the fine
path, even though `COST_MODEL §1.5` supplies the coefficient and P4b uses it for the `S_cnt` direction.
Per-row readiness is the ~926,000-op protocol (θ-P2); deleting it measured **−573.3 µs**; `ι = 0.438 µs
per 1,000 ops` ⇒ ~405 µs by the coefficient, consistent.

**The arithmetic, using only the document's own θ (DERIVED here).** Combine partial volume scales with
top-k, as does the arrival census (`A ∝ rows × k`). Take `V(k=8) = 675 MB`, `R_serial = 1.57 TB/s` ⇒
whole combine 430 µs; `A(k=8) ≈ 926k`.

| k | Φ_natural = 1/(k+1) | Φ_slab(2) | shadow advantage of the FINE path | protocol cost of the fine path `ι·A(k)` |
|---|---|---|---|---|
| 8 | 0.111 | 0.250 | none (slab wins by 1.11·V₀) | ~405 µs |
| 2 | 0.333 | 0.250 | `V(2)·0.083 = 169 MB × 0.083 = 14 MB` ⇒ **≈ 9 µs** (≈49 µs even if inflated by the full `κ_couple` 5.4× coupling factor) | `ι·A(2) ≈ 231k × 0.438/1000 ≈ **101 µs**` |

⇒ **At k=2 the fine path buys ≤ 9–49 µs of consumable shadow and costs ~101 µs of protocol. Slab
certification still wins by ~2–10×. There is no sign flip at k=3.** The vendor result the document
cites *supports this reading*, not the registered one: per-record flags gave **no benefit** and the
LL128-style per-record path was **44% / 177% slower** — a protocol-cost-dominance result
(`LAW-23`, `aug11/OVERLAP_METHODOLOGY_STUDY.md:338-351`).

**Fix.** Restate D4a two-sided:
`(Φ_slab − Φ_natural)·V(k)/R_serial + κ_couple·ΔC_sat  >  ι·ΔA(k)`, re-register P4a with a *predicted
threshold or a predicted absence of one*, and make **K6 report per-arm protocol op counts**, not only
walls. Then re-price **X3**: if the predicted flip disappears, the M-sized top-k unlock loses its
headline justification and should be re-ranked against X1/X2.

### B7. The wave-specialization arm (A8 / B6) is not implementable as written, and its mechanism argument answers the wrong half of LAW-13

**Where.** `OVERLAP_ATLAS §A8` (*"Add arm `fused_wave`: `if (threadIdx.x >= 192) { transport_loop(); }
else { mfma_body(); }`. One rig, one new arm, no serving risk"*, EV MED-HIGH); `EXPERIMENT_LADDER §3.3`
cell **B6** (size implied **S**, 0.3 h, ranked #10 "highest information per node-hour of any new arm");
`§6.2` item 4; `§7.1` kill criterion.

**Repo check.** `m25_boundary_bench.hip:147` — `__launch_bounds__(kThreads,1)` with `kThreads = 256`
(:61) ⇒ 4 waves — and the kernel contains **10 `__syncthreads()`**, including inside the fused
commit loop (`:286`, `:290`, around `ers_commit_fragment` at `:282-283` and `ers_local_arrive`). A
`threadIdx.x >= 192` role split puts waves on **divergent paths across block-wide barriers**, which is
undefined/hanging, not a one-line arm. Any real version must restructure the loop so every wave reaches
every barrier (or replace them with named/split barriers), which changes register allocation on the hot
path — the exact class of change that cost **+726.9 µs** with source untouched (LAW-32).

**Mechanism.** The atlas's argument is *"it pays no CTA capacity tax (8–9 µs/CTA, LAW-27) and does not
change the number of CTAs touching each L2 (LAW-19)"*. Both true — and both address the **smaller**
term. LAW-13 measured the split as **~434 µs capacity + ~815 µs interference**, i.e. interference is
**2.8×** capacity, and it is *memory-system* contention. A wave split does not remove that contention;
it relocates it **inside the CU**, where the transport wave now competes for the same per-CU vector
memory request queue, L1, and TA/TD issue path as the three MFMA waves. The independent-`vmcnt` claim is
correct as stated (`OVERLAP_ABSTRACTIONS.md:42-47`) but `vmcnt` is an *ordering* counter, not a
capacity: an independent counter buys you freedom to not wait, not more outstanding-request slots.

**Fix.** (i) Re-size B6 as **M**, not S, with the barrier restructuring named. (ii) Add the mandatory
control the arm currently lacks: a **compute-only 3-wave body** at the same `k_inner`, so the MFMA loss
is measured separately from the transport gain (without it, a null is uninterpretable — exactly the
G25-1 F1–F4 failure mode). (iii) State the falsifiable form correctly: the hypothesis is *"the missing
resource is per-wave ordering, not per-CU memory capacity"*, and the two are discriminated by whether
`fused_wave` improves at **fixed** total vmem request rate.

### B8. The dedicated-comm-CTA falsification is never tested against the structural difference that makes the MoK/Blackwell design work — and grid geometry is exempted from the settability rule

**Where.** `S_VECTOR §2.10 K10`: grid size / occupancy — *"not swept — it is the axiom the selection
rule rests on"*; `BANKED_LAWS` LAW-28 / contradiction C3; `S_VECTOR §1.9` and `OVERLAP_ATLAS §D2` place
S7/T3 (the MoK-style zero-barrier skeleton) as **design-only, LOW as a near-term ratchet**, and **it
appears in no Phase-2 cell**.

**The reviewer's question the plan cannot answer.** MoK/Blackwell-class designs run dedicated
communication SMs (of order 24–40) successfully. Our ledger says dedicated comm CTAs cost **+831 µs of
added traffic from the role itself** and never win. The campaign's answer is "carrier-conditional" —
true but incomplete, because the three structural differences are nameable and none is tested:

1. **Granularity of specialization.** Theirs is *sub-CTA* (a warpgroup with its own register budget);
   ours is *whole-CTA*, so ours alone pays LAW-27's 8–9 µs/CTA capacity tax. This is precisely what A8
   proposes — and A8 is the *only* arm that touches it (see B7 for why it is under-specified).
2. **Who issues the bytes.** Theirs are descriptor-driven bulk copies that do not consume the math
   warps' vector-memory issue slots; ours are CU-issued vector stores/atomics, so LAW-13's interference
   is structural for us. **Our only analogue is SDMA, which has never been measured
   (`BANKED_LAWS §7` retraction 2) and is explicitly scoped out of θ (`EXPERIMENT_LADDER §1.6`).**
   Scoping it out is defensible; scoping it out *and* claiming a technique-level verdict on dedicated
   comm engines is not.
3. **The residency contract.** They build no grid barriers; a slow comm unit does not gate the grid.
   We realize cross-rank drift at **three points per launch** (LAW-54, `T3_COUNTER_DATAFLOW_DESIGN.md:43-49`).

**And the settability inconsistency.** `LAW-62` is declared binding — *"every axis in W must be shown
settable before a cell is pre-registered"* — yet the single variable on which the whole negative rests
(256 CTAs, all-CU, occupancy 1) is declared an **axiom** and never swept. The campaign is therefore
asking the reader to accept a structural negative measured at one point of an unswept structural axis.

**Fix.** Add one Phase-2 cell, `B.residency` (boundary rig, ~0.4 h): `N_CTA ∈ {160, 192, 224, 256}` ×
{comm inside the kernel, comm as a second concurrent kernel}, registering LAW-54's prediction
("only sub-millisecond kernels slip past the contract") and LAW-12's occupancy assertion per arm. Then
state LAW-28's scope in the master doc as **"CTA-granular dedication inside an all-CU, grid-barriered,
occupancy-1 CDNA megakernel"** — which is honest, is what was measured, and is exactly the sentence that
reconciles our result with theirs. Without this cell the expert question ("their design works, why
doesn't yours?") is answered by assertion.

### B9. Decode is still prefill physics with small numbers: no weight-stream term, no tile/wave quantization, and a launch-overhead bound imported from a prefill step

**Where.** `COST_MODEL §1.11` excludes *"decode as a distinct mechanism"* and `§1.1` treats `H(S)`
(host/launch) as a **bounded constant `|δ| ≲ 15 ms/step ≈ ±2%`**; yet `§3/D3`, P3c, `§5/F1`,
`EXPERIMENT_LADDER §6.3` (a table of ratios to 3 s.f. at T ∈ {8,32,128,256}), `§6.4`, cells K7 and
E-D all make decode predictions.

Three distinct physics gaps:

1. **The launch term.** `H` was bounded on a *padded prefill step* (`|δ| ≲ 15 ms` ≈ ±2% of a
   ~700 ms-class step). A decode step is 1–2 orders of magnitude shorter, so the same absolute bound is
   **50–150% of a decode step**. Meanwhile `OVERLAP_ATLAS §F2` states decode is
   *"launch-overhead-dominated by construction: 61 layers × (attention + gate + sort + group + up to 4
   MoE kernels + 2 collectives)"*. **`M` structurally excludes the term the atlas names as decode's
   dominant cost.**
2. **The weight stream.** `OVERLAP_ATLAS §F3` derives **1.34 GB/rank/layer** of expert-weight traffic at
   decode — an order of magnitude above the 3.67 MB collective — and concludes *"at decode the 'comm'
   worth hiding is the HBM weight stream, not the fabric."* That term exists in the atlas and **nowhere
   in `M`**. Any decode prediction from `M` is therefore made by a model that omits its own identified
   limiter.
3. **Tile / wave quantization.** (M3) `C_p(n) = C_p(N)[(1−φ_p) + φ_p·N/n]` is continuous in CTA count
   with a single scalar `φ_p`. At decode the grouped GEMM has M ≈ (few) rows across 32 local experts:
   cost is set by *how many tiles exist at all* and by weight residency, not by a linear capacity
   elasticity. Worse, `φ_p` conflates two limiter classes: **MFMA-issue-bound** (φ→1) and
   **bandwidth-bound** (which also scales with CTA count **up to θ-C5's measured knee at 128 CTAs /
   4,151.9 GB/s**, then flattens). `S_VECTOR §1.5` uses the conflation to predict *"at decode shapes
   `φ → 0`, so … dedication's cost approaches zero"* — i.e. S2 may win at decode. If the decode phase is
   HBM-streaming and the grid is below the 128-CTA knee, **reserving 32–64 CTAs is not free**, and that
   prediction inverts.

**Fix.** Declare a **decode branch of M** with three explicit terms — `X_weights = experts_touched ×
bytes_per_expert / BW_hbm_measured` (use θ-C5's **4,151.9 GB/s**, not a back-derived peak; see M10),
`C_quant = ceil(rows/tile_M) · t_tile` , and a *measured* per-step launch term — or reduce all decode
predictions to **sign-only** and say so on the slide. Also split `φ_p` into three limiter classes
(latency / CTA-throughput / bandwidth-with-knee) and re-derive `S_VECTOR §1.5`'s decode S2 prediction.
`P0-Z3` (the occupancy compile check) is the right instinct and costs nothing — but it answers only the
occupancy half.

### B10. Two incompatible `Q` vectors are in circulation, and `crit` — the mediator the grounding names for LAW-52 — is in neither

**Where.** `COST_MODEL §0.3` defines **Q as 8 scalars** (`T_eff, co, A/L, φ_p, u/Φ, σ, m, f/W_pad/κ`);
`W_VECTOR §2.2` defines **Q as 10 coordinates** (`B_bnd, C_phase, ρ, co, Σ, P/u, κ, φ, A/L, λ/I_bnd`);
`EXPERIMENT_LADDER §3.1` builds the grid on the 10-coordinate version; `H-Q`, the campaign's central
falsifiable claim, is stated over the 10-coordinate version. `BANKED_LAWS` preamble lists **`crit` —
identity of the critical rank — as a mediating quantity (LAW-52)**; it is dropped from both.

**The mechanism cost of dropping `crit`.** `T_step = Σ_layers MAX_r T_layer^(r)` (M0/M1). A scalar σ
cannot distinguish **"rank 0 is hot in every layer"** (contiguous placement) from **"a different rank is
hot in each layer"** (RR/EPLB) — yet that is exactly the transformation P9 claims is worth +27–46%. In
the second case the per-layer maxima do **not** collapse into one rank's sum, and there is a residual
cost `Σ_layers (max_r − mean_r)` that a σ-only model prices at zero. This is a *matched-Q, different
outcome* pair sitting inside the campaign's biggest registered number.

**Fix.** Reconcile to a single `Q` table in the master doc (the ladder and cost model must not ship two)
and add **`crit persistence`** (cross-layer correlation of the argmax rank) as a coordinate, or declare
it constant with the reason. The measurement is free: it falls out of the per-rank stamps already
mandated by `F3`/`METRICS §3.5`.

---

## MAJOR

### M1. `I ⟂ C` silently resolves a contradiction the grounding forbids resolving silently

`COST_MODEL §1.6`: *"⇒ M models `I ⟂ C` and puts all C-dependence in (M3)"*, citing LAW-29
(mode-13 interference 1,328 µs @C=16 vs 1,323 @C=64). But LAW-14's matched mode-0/mode-2 pairs give
interference **+618.3 µs @C=16 → +1,159.3 @C=64** — an **87% increase with C**. `BANKED_LAWS` reading
rule 4 is binding: *"Where two laws appear to contradict, §6 names the contradiction and its
resolution. Do not resolve one silently in the cost model."* §6 lists C1–C6 and this is not among them.
The correct resolution is available and mechanistic: interference is invariant **at fixed protocol
volume**, and a *carrier* pool's protocol volume grows with C, while mode-13's does not. **Fix:** write
`I = f(A_pool(C, mode))`, add it to `BANKED_LAWS §6` as C7, and note that `I ⟂ C` holds only for
consuming pools.

### M2. `η_size` and `η_ctas` double-count the same parallelism curve

(M5) multiplies `η_ctas(n_CTA)` and `η_size(m)` as independent factors. `COST_MODEL §3/D4`'s own text
says they are the same thing: *"`n_slabs × bps` sets the transport grid occupancy … the measured
bandwidths 55.4/93.4/117.2/116.0 GB/s are a saturating parallelism curve with a knee at 128–256 blocks —
consistent with θ-F2's 8–32-CTA-per-link knees."* Verified against
`tp8_mega/results/g25_0b_slab_sweep.txt`: `slab_rows=1024 ⇒ 16 slabs ⇒ 32 blocks ⇒ 55.4 GB/s`. Applying
both factors penalises coarse slabs twice. **Fix:** one factor keyed on `(blocks, bytes per block)`.

### M3. `J(d)` is a step that asserts zero where the campaign spends K=4 campaigns measuring 50.9 µs

`COST_MODEL §1.6`: `J(d) = 0 for d ≤ 8`. But LAW-20 measured d=4 beating d=8 by **−50.9 µs (−0.77%,
~9σ)**, and `EXPERIMENT_LADDER §3.2` R-K1 sets **K = 4** specifically to resolve *"depth 4↔8 (−50.9
µs)"* at an MDE of 23 µs. So the model predicts a tie in a cell whose sample size exists to measure the
non-tie. **Fix:** either add a monotone sub-cliff term or state that d=4 vs d=8 is below M's declared
resolution and drop the K=4 spend on it (≈0.2 h back).

### M4. Two mutually inconsistent resource models for injection depth are registered simultaneously

**P5a** (`COST_MODEL §3/D5`): *"`d*` halves when the remote op width doubles — the invariant is
`d* × bytes/op × injecting waves`"* ⇒ the scarce resource is **in-flight bytes**.
**P5b**: *"on gfx942 atomics drain ~3× slower ⇒ congestion forms at lower depth ⇒ `d* < 4`"* ⇒ the
scarce resource is **in-flight time / latency-bandwidth product**. Under the bytes model, a 3× slower
drain changes the *time to fill* the queue, not its capacity, so `d*` would be unchanged. Both cannot be
right, and LAW-2 ("the fabric is byte-limited, not op-limited") leans toward the first.
**Fix:** register them as competing hypotheses with a discriminating design in B4/K8 — vary op **width**
(`throttled_store_packet16` vs `throttled_accumulate_bf162`) and op **rate** (arch, or a deliberately
slowed target) independently and state which invariant survives. `OVERLAP_ATLAS §2.1(3)`'s *metering*
reading is a third resource model (duty cycle) and should be named as such rather than blended in.

### M5. σ enters as work only; the rendezvous-dispersion mechanism has no term

(M13) is linear in σ through `λ`. But the measured mechanism is a rendezvous: the hot rank *holds the
global slab certificate hostage* (`BRIEF:70-72`), the pre-M6 region is a **"sum of maxima" with four
grid barriers** (LAW-31), and the training arm's decisive structural finding is that the all-CU barrier
*"realizes cross-rank drift at three points per launch instead of one per layer"* (LAW-54). A model with
no `n_barrier × E[max_r jitter_r]` term (i) cannot price the fusion-boundary knob it declares a schedule
axis (`S_VECTOR §2.12`), and (ii) has no reason to be linear in σ — P9adv extrapolates to σ=8 with a
±10% band and its own falsifier already guesses "most likely super-linear". **Fix:** add
`Σ_barriers E[max_r jitter]` with `c_bar` (already θ-P4, a hole, already scheduled in P0-C's barrier
sweep) plus per-rank arrival dispersion from the stamps; register P9adv as **super-linear by default**,
with linearity as the falsifiable claim.

### M6. The λ hold-out is nearly collinear with the fit and is over-sold as validation

`COST_MODEL §1.8` fits `λ` on the aggregate-histogram replay point and predicts the worst-layer point to
**+0.47%**, calling it *"the strongest single validation the model currently has."* Both points come
from the **same replay ladder** and the **same σ instrument**, and (M13) is a two-parameter line through
one fitted point — predicting a third point on the same monotone family is a consistency check, not a
hold-out. It also inherits H17's **2× dispute** between the two per-call skew instruments (5.15× vs
2.61×). **Fix:** demote to "internal consistency"; claim validation only after HO-4/K3's RR-permuted and
adversarial σ points, which are already budgeted.

### M7. Store-towers' HBM cost is unpriced, so the `co=1` branch of B2 has no prediction

Towers make the owner read `World × slab_bytes` and write `slab_bytes` — **≈ +205 MB/rank/boundary at
this shape**, called out as *"an unpriced consequence of the F5 towers decision … not yet budgeted
anywhere"* (`TP8_STATE §4.3`) and as hypothesis **A2** ("owner tower reduce HBM traffic collides with
the consuming GEMM's reads"). `M` prices only the wire. Cell **B2** sweeps `co ∈ {0,1}`, so it will
produce a co=1 number with no registered prediction behind it. **Fix:** add `X_hbm` to the reduce phase
(inputs already measured: θ-C5's 4,151.9 GB/s plateau, 128-CTA knee) and extend **E-B2's duty histogram
to HBM as well as fabric** — same instrument, one more counter.

### M8. `C6` (duty-cycle metering) is ranked HIGH but has no coordinate in Q — and it is the cleanest matched-Q falsifier available

`OVERLAP_ATLAS §C6` is the campaign's most important TP8 reframe (*"a 1.5× slower transport that runs
continuously beats a fast transport that runs in bursts"*) and §2.1 derives the **28–39% required duty
cycle**. `M` has bytes (M5), events (M6), interference (M7), and overlap credit (M8) — but **no
representation of the time distribution of wire traffic**. Two schedules with identical `B_bnd`, `A`,
`co`, `σ` and different peak-to-mean injection are **Q-identical yet predicted to differ**. **Fix:** add
`burstiness` (peak/mean fabric injection over the layer) as `Q11`, and register it as **FQ-4**, a fourth
matched-Q falsifier — it is cheaper and sharper than FQ-1/2/3 and E-B2 already builds the instrument.

### M9. The campaign carries two opposite funding orders for the TP8 track

`COST_MODEL §3/P2c`: *"the transport track (A1–A6) is worth more than the hiding track until β < 1.2."*
`OVERLAP_ATLAS §C6.3`: *"Track B (hiding) should be funded ahead of Track A (transport efficiency)."*
`EXPERIMENT_LADDER §6.2` inherits both without adjudication. They are reconcilable and the reconciliation
is the finding: **D2's `h*` formula assumes the un-hidden fraction pays `β` in wall time, which is true
only for a schedule that bursts.** Once traffic is metered across the layer and the required duty is
28–39%, `β = 1.5` costs duty-cycle *headroom*, not wall — so P2c is the law for burst schedules and C6
is the law for metered ones. **Fix:** state both scopes explicitly in the master doc, and make B1 record
duty cycle so the two regimes are distinguishable in the same run.

### M10. A back-derived HBM peak is used as a denominator — the exact error LAW-62(a) warns about

`OVERLAP_ATLAS §C4` and **§F3** size the RMSNorm pass and the decode weight prefetch against
*"the exp_29-implied HBM ≈ 7.85 TB/s"* — a figure derived by dividing a measured combine rate by an
assumed 20% efficiency. θ-C5 has the **measured** value: knee at 128 CTAs, plateau
**4,151.9 GB/s = 51.9% of 8,000**. Using a back-derived peak is the same class of error as exp_22's
spec-sheet denominator, which *"manufactured a refutation"* (LAW-62a). **Fix:** quote both, size against
4.15 TB/s. F3's decode weight-stream estimate becomes ~323 µs/layer at the measured plateau (which
strengthens F3's conclusion, not weakens it).

### M11. The four-door screen has no positive controls, and door membership was assigned by the authors knowing the outcomes

`COST_MODEL §3/D11` reports **11/11 retrospective** and P11 registers *"every zero-door arm measures
inside ±1σ of its control."* Two problems. (a) The classification is post hoc and visibly uncertain in
the table itself — *"G25-1 F2 register-source stores | (c)? op-class unchanged"* — and **F3 is classified
as going through door (a) and still measured null**, which already breaks the "through a door ⇒ pays"
direction that the campaign will want to use for planning. (b) B6/`fused_wave` is exempted as *"a new
door"*, so the arm most likely to reveal a fifth door escapes the prediction meant to detect one.
**Fix:** put the door assignment for every Phase-2 arm into `predictions.jsonl` **before** the run;
include ≥2 **one-door positive controls** per topology; and state the screen as a *necessary, not
sufficient* condition, which is all the corpus supports.

### M12. B7/PHY cannot distinguish incast relief from latency-bound behaviour

`EXPERIMENT_LADDER §2.3 PHY` predicts the world 8→4 wall ratio from **egress bytes alone**
(`2(N−1)/N`, −14.3% ⇒ ratio ∈ [0.78, 0.86]) with the falsifier *"ratio ≥ 0.95 ⇒ the AR is
latency/protocol-bound"*. But world also changes fan-in degree (8→4), so an incast-relief effect and a
latency effect are confounded in the same knob and the falsifier will mis-attribute. **Fix:** add a
fixed-world **destination-concentration** arm (all 7 peers → 1 owner vs uniform round-robin owners) at
matched bytes; that is the direct incast instrument, it is free on `m25_cdar_bench`, and it also
supplies the θ entry B1 needs.

### M13. `κ_couple` absorbs at least three mechanisms; the ledger as specified separates one

`COST_MODEL §1.7` bounds `κ_couple ≤ 55 µs/CTA` from a **5.4× under-prediction** of the C-ladder and
attributes it to producer-drain release (LAW-30's r = −0.904). Three candidates are consistent with the
same data: (i) drain release; (ii) `ρ = 6.1 GB/s/CTA` measured **in isolation** but applied to a pool
running under exactly the memory contention LAW-13 identifies; (iii) the pool reducing the producer's
own work via fall-through/quota. θ-P6's ledger (drain-wait vs MFMA span) separates only (i). **Fix:**
run the ledger with two extra arms at the same C — `flush_rows = 0` (isolates pure protocol deletion,
already a designed control) and a reserved-but-idle pool (LAW-27's instrument) — and decompose all three.

### M14. `PF8` implicitly predicts our transport beating RCCL and rests on a byte-proportional model the ledger contradicts

`EXPERIMENT_LADDER §2.3 PF8`: fp8 is 1.939× fewer bytes ⇒ **−1,072 µs**, taking our 2,214 µs to
~1,142 µs — i.e. **below RCCL's 1,470 µs**. But the diagnosed deficit is an **issue-rate** deficit:
atomics cap at 67 GB/s *"diagnosed as the dword issue rate"* (F5), and A4 says we are *"simply
under-issuing the wire"* at 58% vs RCCL's 79% of ceiling. Halving bytes helps only if the **op count**
falls too (fp8 packets carrying 2× elements at the same 16 B packet width). **Fix:** register PF8 with
the op-count invariant explicit (`ops_after = ops_before / 2`, verified in ISA), and state the accuracy
gate — this is an *activation* all-reduce, not a combine partial, and `TP8_STATE §6/A5` already flags
the numerics as SPECULATION. Any claim of the form "we beat RCCL" must close β on matched precision
first, or be stated as "we beat RCCL by moving fewer bits, at an accuracy cost of X".

### M15. Two "laws" are single-cell artifacts and should be relabelled before they enter the master doc

* **"Depth is a cliff at d=8"** (LAW-20 / θ-P8): measured on one carrier (producer epilogue,
  packed-bf16 RMW), one peer count (7), one grid (256 CTAs), one T. `d=2` has never been measured on any
  arch. `COST_MODEL §3/D5b` states `d* = 8` with the full parenthetical scope, which is good — but
  `S_VECTOR §2.3` calls it *"the cleanest law the campaign already owns"* and the atlas reuses the
  −615 µs in three places. Recommend the master doc always print the scope inline.
* **"K1 reverses at decode"**: see **B3** — confounded with grid occupancy at n=1.
* **"Granularity is free" (M15) vs "granularity is worth 2.1×" (CDAR)**: the reconciling hypothesis
  (`S_VECTOR §2.4.1`, `W_VECTOR §Q6`) is good, but note that CDAR's 2.1× spread is *itself* the
  parallelism curve of M2 — so the honest statement after the FQ-3 decomposition may be that
  **neither** is a granularity law, and both are occupancy laws. Register that as the third outcome of
  FQ-3, which currently admits only two.

---

## What the plan gets right (so the fixes do not overcorrect)

* **The champion baseline is named honestly.** `EXPERIMENT_LADDER §6.2` ranks the **RCCL-hybrid first**
  and registers that fused CDAR does *not* beat it until β < 1.2. No cell assumes our AR wins — with the
  single exception of PF8 (M14). This is the right posture given LAW-10.
* **`co` as the depth mediator** is the strongest law in the corpus and the plan protects it: FQ-2 does
  both halves on one rig, and B4 sweeps depth × co at the corrected ratio. Keep it.
* **The 91.7% lesson is applied correctly** — `Φ_natural(k) = 1/(k+1)` reproduces the measured constant
  (`2^{-1/8} = 0.9170`), and K2×K3 are treated as an interaction, never a main effect. The error is in
  the *cost side* of D4a (B6), not in the readiness physics.
* **The G25-1 re-scoring** (ρ = 0.65 vs a 90% bar; the bar was unreachable) is the single most valuable
  piece of analysis in the whole set and should lead the master doc.
* **The instrument-is-part-of-the-kernel discipline** (LAW-32/58/59 as manifest fields) is the right
  answer to the "hundreds of kernels" risk, and it is what makes B5's and B7's fixes affordable.

## Recommended additions to Phase 0 / Phase 2, with cost

| # | addition | rig | cost | closes |
|---|---|---|---|---|
| 1 | fan-in sweep (k→1, k ∈ {1,2,4,7}) + concentrated-vs-uniform destination arm | `m25_cdar_bench` | ~0.3 h | **B1**, M12 |
| 2 | affine `L` intercept calibration per (op, world, n_slabs, bps) — reuse P0-A points | `m25_cdar_bench` | 0 (re-analysis) | **B2** |
| 3 | decode-size geometry sweep, each transport at its own optimum | `m25_cdar_bench` | ~0.2 h | **B3** |
| 4 | `h_ledger` from span overlap alongside `h_wall` | E-B1 | 0 (rides Action 3) | **B4** |
| 5 | `bytes_per_mfma` argv so intensity ⟂ duration | `m25_boundary_bench` | S | **B5**, TP8_STATE B2 |
| 6 | `B.residency` cell: N_CTA × {in-kernel, second-kernel} | `m25_boundary_bench` | ~0.4 h | **B8** |
| 7 | HBM duty counter in the duty histogram | E-B2 | 0 | M7 |
| 8 | `flush_rows=0` + reserved-idle arms at fixed C inside the ledger runs | mega chassis | ~0.2 h | M13 |
| 9 | re-derive D4a two-sided; re-register P4a; re-rank X3 | CPU | 0 | **B6** |
| 10 | one `Q` table; add `crit persistence` and `burstiness`; register FQ-4 | CPU | 0 | **B10**, M8 |

Total added node time ≈ **1.1 h** against a 38.5 h plan; the rest is re-analysis and re-registration
that must happen **before** Phase 1's registration commit, since four of these change registered
predictions.

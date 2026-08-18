# FEASIBILITY & STATISTICS CRITIQUE — adversarial review of the ablation campaign design

**Lens 2 of the pre-synthesis review.** Target: `design/EXPERIMENT_LADDER.md` (986 lines), read
against `design/{W_VECTOR,S_VECTOR,COST_MODEL,METRICS_PROTOCOL,OVERLAP_ATLAS}.md`,
`grounding/*`, `BRIEF.md`, and the operational record in `nightshift/LOG.md`,
`nightshift/REPORT.md` and the session memory files.
**Written:** 2026-08-18, laptop, no node access. All paths absolute-relative to
`distributed-kernels/fused_moe/overnight/aug18-ablations/` unless stated.

**Scope of this lens:** arm-count and node-hour arithmetic; grid tractability on one flaky
shared node; sample sizes vs *characterized* variance; thermal/clock confounds; determinism
and reproducibility gates; coverage/seal traps (arms silently not running); hold-out hygiene;
manifest/build reproducibility at hundreds of arms; the env-whitelist trap; multiple
comparisons; and whether the gates can actually fail. Correctness of the physics, the model
structure, and the per-cell predictions are **out of scope** here (lens 1/3).

---

## 0. Verdict

The ladder is the strongest planning document in this set: it is the only one that names
actuators, prices cells, pre-registers falsifiers, and pre-commits a degradation path. Three
things in it are genuinely load-bearing and should survive synthesis unchanged — the
settability gate (`EXPERIMENT_LADDER.md:20-28`), the X1–X5 unbuilt-actuator table (`:79-91`),
and the derived-not-asserted sample-size section (`:357-388`).

**But as written the campaign is not executable as budgeted, and two of its headline
statistical guarantees do not hold.** The three findings that most endanger it:

1. **The hold-out set is contaminated by Phase 0 and by `COST_MODEL §4.1`.** `:521` fits "Phase
   0 entirely" while Phase 0 measures most of `:522`'s held-out points. The campaign's central
   credibility claim — "θ is never re-fitted on a held-out cell" — is violated by its own
   schedule before a single Phase-2 cell runs.
2. **The node-hour budget is understated ~1.6–2.5× on arithmetic alone**, before any allowance
   for a node that lost ~8 h in a single night this week, and the two-week backlog schedules
   under half the budgeted hours while omitting 5 of 17 cells and all of Phase 4.
3. **Boundary-rig run-to-run variance has never been measured anywhere in this repo**, yet every
   Phase-2 sample size derives from an assumed `σ_rig ≈ 0.15%` that is actually a *between-arm
   spread*, not a repeatability estimate. That is a Phase-0 hole the plan does not list.

Ten BLOCKING and twelve MAJOR findings follow. Each names a file/section and a fix.

---

## 1. BLOCKING

### B1 — Hold-out contamination: Phase 0 measures the held-out set (`:521-522` vs `:139`, `:175`)

`§4.1`'s split (`:521`) reads:

> **FIT** | Phase 0 entirely; **B2's two endpoints only** (3.67 and 235 MB); **B4 at co=0 only**; … | **HELD OUT** | **B2's middle sizes** {15, 33, 60} MB; … **B1 at ρ ∈ {2, 3}**; …

This is internally contradictory, three times over:

| held-out point (`:522`) | but Phase 0 already measures it | where |
|---|---|---|
| B2's middle sizes {15, 33, 60} MB | P0-A θ-F5/F6/F7: "`b` ∈ {3.67, **15**, **33**, **60**, 235} MB × transport {atomic, towers}", 30 points | `:139` |
| B1 at ρ ∈ {2, 3} | P0-B ρ (Q3): "`k_inner` argv … until ρ ∈ {0.65, 1.0, **2.0**, **3.0**, 4.0}", 20 points, 3 repeats | `:175`, `:187` |
| B4 at co=1 (implied by fitting co=0 only) | P0-A θ-P8: "sweep depth {2,4,8,16,32} at co=0" — fine; but P0-B's ρ-sweep runs the fused arm at co=1 across the same depths | `:141`, `:175` |

And the sibling that the ladder declares supreme (`:8-15`, *"where this document and a sibling
disagree and the disagreement is not named here, the sibling wins"*) contradicts the split
outright: `COST_MODEL.md:1041` puts the **whole** transport grid — `slab_rows {128,256,512,1024}
× depth {4,8,16,32} × bps {1,2,4} × tokens {256, 2K, 4K, 16K}` — in **CAL-T** (fitted), while
`COST_MODEL.md:1051` holds out "slab sweep at T=4096 and T=2048" — the same points. Likewise
`COST_MODEL.md:1052` holds out the *entire* five-point `b` sweep that the ladder fits two
endpoints of. Under the ladder's own precedence rule the sibling wins, and then the ladder's
hold-out set is empty for P3, P4c and P5d.

Worse, two of the "strongest prior evidence" entries at `:529-534` are already-seen data being
re-used as hold-out: the worst-layer skew point (`22,582 predicted vs 22,475 measured`) and the
affine-T midpoint (`+1.07%`) are quoted *in the split section itself* as validated residuals,
and then the same worst-layer point appears in `:522`'s held-out list ("K3's worst-layer") and
the T midpoint inside "K1's interior T {1536, **2048**, 3072}". A point whose residual is
already published is in-sample knowledge; re-labelling it hold-out is the exact reviewer bait
the campaign cannot afford in front of the people who asked "what is the cost model."

**Fix.** (a) Reconcile with `COST_MODEL §4.1` explicitly in `§4.1` (name the disagreement, per
`:8`). (b) Rule: *a point measured in Phase 0 is FIT, full stop* — so B2's middles and B1's ρ
interior must either be dropped from Phase 0 or dropped from the hold-out. Recommended: Phase 0
measures **b ∈ {3.67, 235} and ρ ∈ {0.65, 4.0} only** (the endpoints it actually needs for θ),
leaving the interiors genuinely unseen. (c) Move the two already-validated points to a new
"PRIOR-VALIDATED (in-sample)" row and replace them in the hold-out with points nobody has seen:
K1 at T ∈ {512, 3072} on the post-X2 M15 body, K3's adversarial-synthetic point, B2's middles at
co=1 only. (d) State a no-peeking rule for the Phase-1 re-registration: `§2.4`'s rule 3
("sign changes → stop … the cell is re-designed to bracket it", `:329-331`) is an adaptive
design step; constrain it to *adding* points, never removing or relabelling hold-out points.

### B2 — Phase-2 budget omits the repeat factor K entirely (`:392-404` vs `:62`)

The boundary matrix carries a `K` column and a `pts` column, and the `h` column is priced off
`pts` alone:

| cell | pts | K | invocations (pts×K) | h claimed |
|---|---:|---:|---:|---:|
| B1 | 20 | 3 | 60 | 0.5 |
| B2 | 45 | 3 | 135 | 1.2 |
| B3 | 24 | 1 | 24 | 0.4 |
| B4 | 20 | 3 | 60 | 0.7 |
| B5 | 18 | 3 | 54 | 0.5 |
| B6 | 8 | 3 | 24 | 0.3 |
| B7 | 9 | 3 | 27 | 0.3 |
| B8 | 8 | 3 | 24 | 0.4 |
| **total** | **152** | — | **408** | **4.3** |

At the ladder's own unit cost of ~1 min/invocation (`:145-146`), 408 invocations is **6.8
node-hours, not 4.3** — and `§0.2:62`'s "~250 rig invocations" is inconsistent with its own
matrix by 1.6×. (Cross-check: `h ≈ pts × 1 min × 1.0–2.1`, i.e. K is simply absent from the
pricing.)

The same defect appears in Phase 0: `:145-146` states "~66 primary points; ~30 of them need 3
invocations ⇒ ≈126 invocations × ~1 min … ≈ **1.6 node-hours** with retries." 126 × 1 min =
**2.1 h** before retries. And `§0.2:64` says Phase 4a is 16.5 h while `§5.2:605` totals its own
rows to 16.7 — so the headline "≈38.5" (`:66`) is arithmetically 38.7 even on the doc's own
numbers.

**Fix.** Reprice every row as `pts × K × unit`, publish the invocation count per cell in the
manifest scaffold, and restate the totals. Expect Phase 2 boundary ≈ 6.8 h (not 4.3) and Phase 0
≈ 6.9 h (not 6.0) before the corrections in B3–B5 below.

### B3 — The boundary-rig unit cost is uncited, and the sibling contradicts it (`:145-146` vs `METRICS_PROTOCOL.md:588`)

Everything in Phase 0 (P0-A, P0-B) and all of B1–B9 is priced at "~1 min (8-GPU single-process
init dominates)" (`:145-146`) with **no source**. It is the single largest budget input in the
campaign (≈534 invocations) and it is the only per-unit cadence in the plan that is not
measured — LAW-60's 3 min 5 s MoK campaign and the ~25 min serving pair both carry citations
(`:75`). `METRICS_PROTOCOL.md:588` prices the same instrument at "**minutes**". If the true
figure is 2 min, Phase 0 + Phase 2 boundary go from ~10 h to ~20 h.

**Fix.** Make "measure the boundary-rig invocation cadence (cold-start and warm)" the *first*
Phase-0 item — it costs 10 minutes and it re-scales half the campaign. Until then, carry the
budget as a range [1, 3] min/invocation and publish both endpoints.

### B4 — `σ_rig` is not a variance estimate, and boundary repeatability has never been measured (`:357-388`)

`§3.2` is titled "Boundary-level variance, from the grounding" and then sources it from:

* LAW-60 campaign reproducibility 0.09% — **MoK instrument**, not the boundary rig;
* exp_35 campaign sd 13.7/15.5/6.9 µs — **MoK instrument**;
* LAW-60 screen drift 6.6% — **MoK instrument**;
* `METRICS §4.1` "depth arms separated at 0.1–0.3%" — this is the **spread between four
  different depth arms** (93.6 / 93.4 / 93.3 GB/s, `METRICS_PROTOCOL.md:588`), i.e. a
  between-condition difference read as "flat". It is not a repeatability measurement.

R-K2 (`:382-383`) then asserts `σ_rig ≈ 0.15%` and derives the entire boundary sampling rule
("3 invocations resolve ≥0.35%, 1 invocation resolves ≥0.6%") from it. Using an observed
between-arm spread as an estimate of within-arm noise is backwards: if the arms differ, that
spread is signal; if they do not, three points give σ with ~2 df, i.e. a CI on σ spanning
roughly 0.6×–4×. I could find **no repeated identical boundary invocation anywhere in the
grounding** (`grounding/TP8_STATE.md` reports no repeats or sds for the rig).

Consequence: every boundary MDE, every `K`, and the Phase-0 exit gate G0-b ("every θ entry
carries a confidence interval", `:104-108`) rest on an unmeasured quantity.

**Fix.** Add **P0-A0, the repeatability arm**: one fixed configuration (say `fused`, 16384
tokens, 256 rows, depth 4), **20 invocations spread across a full session** (not back-to-back),
plus 5 more after a 30-minute idle. Decompose into within-session σ and session/thermal drift.
Cost ≈ 0.5 h. Every `K` in `§3.2` is then re-derived from a measured number, and G0-b becomes
satisfiable rather than aspirational.

### B5 — The MDE derivation assumes paired two-arm contrasts; the campaign's outputs are argmax and crossover locations (`:357-388`, `:544`)

`§3.2` derives `MDE = 2.80 · σ_diff / √K` for "a paired contrast over K campaign pairs", and
disposes of the measured 6.6% within-batch drift by declaring it "**common-mode** … removed by
the exp_35 discipline: same-session, alternating `c,b,c,b` campaign pairs" (`:367-369`).

That works for a two-arm A/B. It does **not** work for the campaign's actual deliverables, which
are almost all *locations*, estimated across a multi-level sweep run sequentially in one
session:

* `C*` over C ∈ {0,8,16,24,28,32} (K4, K2)
* `T*` over 7 T values (K1)
* `b*` over 5 message sizes (B2)
* `g*` over 4 slab sizes (B3)
* `d*` over 5 depths (B4)
* MAG puller knee over 4 counts (B5)

For an argmax over 6 levels there is no single control to alternate against, so monotone
within-session drift (thermal soak, see B6) biases *later* levels systematically. And the
accuracy bar that scores these is **boundary-location error ≤ 1 grid step** (`:544`) — a
statistic whose uncertainty is `σ_diff / slope`, which the document never propagates. Near an
optimum the slope is ~0 by definition, so the location CI can be several grid steps even when
each individual point is resolved to 0.2%.

Two consequences: (i) the ladder cannot currently state whether K4 at K=4 can actually locate
`C*` to one grid step (its own evidence says C 24→28 is −0.45% and C 16→24 is −444 µs — a 20×
asymmetry in the curvature); (ii) on a coarse grid the ≤1-step bar is **trivially satisfied**
whenever the crossover is bracketed, so it has no teeth (see B10).

**Fix.** (a) Require a **blocked, order-randomized** design for every multi-level sweep: levels
in randomized order within each replicate, with a fixed **anchor configuration re-run every N
invocations** to measure and regress out session drift; the anchor series is also the
per-cell envelope. (b) Replace "boundary-location error ≤ 1 grid step" with a **profile-CI
criterion**: fit the response, bootstrap over replicates, and require the predicted location to
lie inside the measured 80% CI of the argmax/crossover — and report that CI. (c) Where the
predicted location's CI spans >2 grid steps at the planned K, either refine the grid near the
predicted point or record the cell as *unable to adjudicate* **before** it runs.

### B6 — No clock policy, no temperature logging, no drift anchors, no foreign-process check (`grounding/WORKLOAD_EVIDENCE.md:462-471`, `METRICS_PROTOCOL.md:486`)

Grep of `EXPERIMENT_LADDER.md` returns **zero** hits for cooldown (kernel-side), clock,
temperature, perf-level, foreign process, randomization, or anchor. Meanwhile the grounding
carries the confound in measured form:

* **GPU2 is the hottest (77 °C) and lowest-clocked (1,725–1,743 MHz vs 1,790–1,822) — a 4–5%
  clock spread**, and rank 2 is the structural straggler (`WORKLOAD_EVIDENCE.md:462-469`).
* **+18.7 ms first-run penalty and +3.4 ms monotone thermal soak across three back-to-back
  runs** (`WORKLOAD_EVIDENCE.md:470-471`).
* `--showperflevel` returns **`auto`** — DVFS is free-running — and idle junction temps already
  span **10 °C** across the 8 GPUs (`METRICS_PROTOCOL.md:486`).
* `COST_MODEL.md:1196` already prescribes the remedy: *"Per-rank device stamps in every serving
  arm + `amd-smi` clock/temperature logging."* The ladder does not adopt it.

The campaign runs ~400 back-to-back boundary invocations and ~164 MoK campaigns in overnight
sessions. Under LAW-52 (`grounding/BANKED_LAWS.md:844-859`) the megakernel's pace *is* the
hottest/slowest rank's wall clock — so a monotone soak does not just add noise, it adds a
**directional** bias that grows through a sweep, and it lands specifically on the rank-coupled
rendezvous the campaign is measuring. This is also the one confound that could masquerade as a
skew effect in K3 (a hot GPU ∩ hot expert rank coincidence, flagged as H16 with *zero* serving
points, `WORKLOAD_EVIDENCE.md:586`).

**Fix (cheap, mandatory).** (a) Add to the Phase-0 exit gates a **G0-e — thermal/clock parity**:
`rocm-smi`/`amd-smi` per-GPU clock + junction temp sampled at ≥0.2 Hz for the whole session,
stored per run, and reported as min/median/max per arm; a run whose per-GPU clock spread exceeds
the session baseline by >2% is flagged. (b) Attempt a **deterministic clock policy** (fixed
perf level / power cap) and record whether it is settable on this node — if it is not, say so
and treat clock as a measured covariate, per LAW-62's own doctrine. (c) Add the **idle-first-run
discipline**: discard or separately label the first invocation of every session (+18.7 ms
measured). (d) Add a foreign-process check to the runner (the node-prep step already exists in
`nightshift/LOG.md:247`) and record its result in `runs.jsonl`.

### B7 — Three key axes have no actuation receipt, so a silent no-op is indistinguishable from the registered falsification (`:297`, `:422`, `:429-443`, `:852`)

This is the 2%-coverage disaster in a new costume. The historical failure was not a wrong
number; it was an arm that **ran without the kernel firing** (`METRICS_PROTOCOL.md:467-471`;
LAW-49). The serving side now has receipts (seal ratio, `W`, κ) and the ladder mandates them
(`§5.3:612-626`). The **kernel side does not**, and three of the campaign's most important axes
are exactly the ones with no receipt:

| axis | cell | actuator | receipt today | failure mode |
|---|---|---|---|---|
| **φ (fill)** | K2 (`:422`) | `K0P6_M24_FILL` + `NORIG_CONST/_TABLE` — **has never compiled** (X1, `:81`) | none. `M24_FILL_RECEIPT` **does not exist** (`METRICS_PROTOCOL.md:981`, gap #3) | if FILL=1 compiles but the fill vector never reaches the five T-substitution sites, the kernel runs identically at every φ ⇒ measured result = "FILL=1 within 10% of FILL=0" = **exactly P1c's registered falsifier** (`:297`) |
| **routing / skew** | K3 (`:421`, `:429-443`) | `route_replay.py` — **code complete, 22 CPU tests pass, NEVER EXECUTED ON GPU** (`WORKLOAD_EVIDENCE.md:554`) | `route_multiset_sha256` is computed **CPU-side in the loader**, not echoed by the kernel | a silent fallback to balanced routing produces "captured ≈ shuffled within 3%" = **exactly K3's registered kill criterion** (`:445-448`) |
| **per-rank C** | K3/P7c (`:310`, `:852`) | "an env-per-rank change plus a manifest field" | none; and see B8 | per-rank C silently collapsing to global C reads as "a per-rank-C arm that ties global C" = **exactly P7c's falsifier** (`:310`) |

In all three cases the null-actuator failure mode and the registered scientific finding produce
the **same measurement**. That is not a risk; it is a guaranteed ambiguity for three of the
campaign's flagship cells.

**Fix.** Make an **actuation receipt a per-cell admission gate**, symmetric with the serving seal
ratio: (a) M24 must emit a device-side counter of rows actually skipped / `T_eff` actually used,
printed per campaign, and K2 may not be adjudicated unless that counter tracks the requested φ
(this is `METRICS_PROTOCOL.md` gap #3 promoted from "small patch" to blocking); (b) the MoK
harness must print the **observed per-rank destination-load CV** per campaign, and K3 may not be
adjudicated unless the CV differs across routing arms by a pre-registered margin — this also
gives the skew axis its own measured σ instead of the unreconciled [2.61, 5.15] band (H17); (c)
per-rank C must be echoed **per rank** through the existing descriptor-dump path
(`S_VECTOR.md:892-905`, `e35_11_descverify.sh`), rank-max and rank-min reported.

### B8 — The env-whitelist trap is never mentioned, and one plan item violates its mitigation head-on (`S_VECTOR.md:967-1001` vs `:852`)

`S_VECTOR §5.3` records a *measured* instance of the trap: `e004pf_k0pf_ab.py:241` reads
`K0_PF6GM_G` with default `"2"` and **`K0_PF6GM_G` is not forwarded by the frozen
`run_campaign.sh`** — so a G=3 candidate silently ran against a G=2 reference, *"a ~330 µs
handicap in the candidate's favour, and an invalid claim"*, detectable only by the sanity ratio
(0.898 if forwarded, 0.941 if not). Its three mandatory mitigations are: carry every runtime
knob on the **whitelisted `K0_MPS_CFG` path, never as a new ad-hoc env var**; record the
device-side descriptor echo; carry the reference sanity ratio in every row.

`EXPERIMENT_LADDER.md` mentions none of the three. And `§6.5:852` proposes precisely the
forbidden construction: *"the config word is per-process, so per-rank C is an env-per-rank change
plus a manifest field."* K2 (`NORIG_CONST/_TABLE`) and K3 (`K0_MOK_ROUTE_HIST`, the replay envs)
are likewise new non-whitelisted variables carrying the two most important workload axes.

**Fix.** Add an explicit **§0.4 "forwarding contract"**: every new knob is either (i) encoded in
`K0_MPS_CFG` and verified by descriptor echo, or (ii) accompanied by a *forwarding proof* — a
run where the harness prints the value it actually received, plus a reference sanity ratio in
the manifest row. No cell may enter Phase 2 whose knob lacks a forwarding proof. This is a
one-line addition to the `runs.jsonl` validity rules (`S_VECTOR.md:892-905`) and it is the
cheapest insurance in the campaign.

### B9 — Two of the four "mandated" e2e cells cannot support any claim they are assigned (`:596-605`, `METRICS_PROTOCOL.md:369-375`)

| cell | n planned | protocol floor | measured envelope | prediction it must adjudicate |
|---|---|---|---|---|
| **E-D** `E.decode` (ISL 1024 / OSL 512, C ∈ {8,16,32}) | "**3 pairs (directional)**" for **three** C-points ⇒ **n=1 per cell** | ≥5 order-balanced pairs, hard floor (`METRICS_PROTOCOL.md:370-371`; `BENCHMARK_PROTOCOL.md:50-53`) | **unmeasured** — P0-D3 measures envelopes at c32p and c512p only (`:207`) | **PE4** (`:318`), whose falsifier is a >20% TPOT p50 win |
| **E-E** `E.mixed` (open-loop Poisson) | 3 pairs | ≥5 | **unmeasured**; the open-loop cells have **never been run** (H6) | **PE5** (`:319`), which `METRICS §2.2` calls *"the campaign's biggest single exposure"* |

R-N2 is explicit: *"A measured envelope **never** lowers the `n ≥ 5` floor"* and R-N3 forbids a
claimed delta smaller than the cell envelope (`METRICS_PROTOCOL.md:369-375`). The ladder's own
instrument-scoping rule at `:548-551` repeats it: *"no e2e cell may be used to adjudicate a
prediction smaller than its own measured envelope."* E-D and E-E have **no** measured envelope
and n below the floor, so under the campaign's own rules they can adjudicate nothing.

Three further blockers specific to E-D, none of which the guard list catches:

* **Sizing.** `waves = num_prompts / C ≥ 20` (`:624`) at OSL 512 makes each arm long; 4.5 h for
  three C-points at ≥5 pairs each is short by roughly **3×** (see §3 re-budget).
* **A memory-config confound.** Our arm runs `--gpu-memory-utilization 0.70`, the vendor arm
  `0.9` (`WORKLOAD_EVIDENCE.md:494-502`). At OSL 8 the KV footprint is negligible; at **OSL 512
  it is 64× larger**, so "KV capacity → achievable C" (listed only as a guard, `:602`) becomes
  the dominant term and the cell risks measuring a memory setting, not a kernel.
* **W3 is blocked for E-E.** Open-loop windowing requires `sent_s / issued_s / scheduled_s`,
  which the banked client leaves **null** (`METRICS_PROTOCOL.md:300-304`, gap #2). E-E therefore
  needs a client v4 patch that appears in **no** X-item and **no** budget line.

**Fix.** Either (a) demote E-D and E-E to explicitly *exploratory, unadjudicated* cells — run
them, publish them as "observed, mechanism unattributed" (R-S1), and remove PE4/PE5 from the
scored prediction set; or (b) fund them properly: one C-point for E-D at ≥5 pairs (~5 h), matched
`--gpu-memory-utilization`, plus the client v4 patch as **X6** before E-E. Option (a) is
recommended given the budget reality in §3.

### B10 — Several gates cannot fail, and the campaign's top-level falsifier has no cell (`:100-116`, `:536-546`, `:573-581`, `W_VECTOR.md:604-612`)

Asked plainly: *can a phase actually fail?*

* **Phase 0.** G0-a demands every θ row be class M or D. But θ-P5 (`I₀`) is closed by definition
  — *"the residue after attribution **is** `I₀`"* (`:172`) — so it can never fail to be
  "measured". G0-c offers an explicit escape hatch: the fudge factors must be *"either measured
  … **or** declared per-skeleton with the scope stated"* (`:109-112`). A gate with an
  or-declare branch always passes. G0-d (instrument parity) is the only one with teeth.
* **Phase 3.** `§4.3-4.4` (`:553-581`) routes every possible outcome to something shippable,
  ending at "publish the miss" / a regime map. That is admirable honesty, but it means Phase 3
  cannot fail either — and the ladder should say so out loud rather than presenting Phase 3 as a
  gate.
* **The bar that scores it.** "Boundary-location error ≤ 1 grid step" (`:544`) is trivially met
  by any bracketed crossover on a coarse grid (B5); "sign accuracy ≥ 90%" (`:545`) has **no
  declared denominator** — over which set of (knob, cell) pairs? If GATED cells that never ran
  are silently dropped, the score is computed on the subset that worked. And several registered
  predictions are near-zero-effect predictions (P11, P1d, P7a), where sign is undefined.
* **The top-level falsifier is unscheduled.** `§4.4:581` and `§7.1:926` both make
  **FQ-1/2/3 (matched Q, different S\*)** the criterion that kills the central hypothesis H-Q.
  FQ-2 is B4 (co × depth on one rig) ✓. **FQ-1 and FQ-3 have no cell in `§3.3`/`§3.4`.** FQ-1
  needs `T_pad=4096, φ=0.5` **and** `T_pad=2048, φ=1.0` in one matched comparison
  (`W_VECTOR.md:610`) — i.e. X1 **and** X2 together, and no arm in the ladder pairs them. FQ-3
  (granularity: signal count at fixed geometry, then geometry at fixed signal count,
  `W_VECTOR.md:612`) has no cell at all; B3 varies geometry only. **The one criterion that can
  kill the model has no experiment.**

**Fix.** (a) Delete the "or declare" branch from G0-c, or rename G0-c to a reporting requirement
rather than a gate. (b) Pre-register the **scoring denominator** in Phase 1: the exact list of
(prediction, cell) pairs, with unrun cells reported as `NOT ADJUDICATED` in a visible count, and
sign accuracy computed only over predictions whose predicted effect exceeds the cell's measured
MDE. (c) Add **FQ-1** as an explicit cell (K2 × K1 crossed at two matched-Q points; it is 4
campaigns, ~0.2 h, and it is the campaign's own kill switch) and **FQ-3** as a cell (M15
`K0P6_M15_SLABS` signal-count-only sweep vs CDAR `slab_rows` geometry sweep). If X1/X2 do not
land, say plainly that **H-Q is untestable this campaign** rather than leaving the criterion
formally present and practically unreachable.

---

## 2. MAJOR

### M1 — No multiple-comparisons discipline anywhere (`§2.3`, `§3.2`, `§4.2`)

25 registered predictions, 17 cells, ~408 boundary invocations and ~164 campaigns, with MDEs
derived at α = 0.05 per contrast (`:370`) and no mention of family-wise error or FDR. The
campaign is explicitly *"hunting decision boundaries across many axes"*, which is the textbook
setting for α-inflation: at α = 0.05 and ~60 pairwise contrasts, ≈3 spurious "significant"
results are expected under a global null. **Fix:** declare a two-tier structure at Phase 1 — the
25 registered predictions are **confirmatory** and carry a Holm or BH correction over the
registered family; everything else is **exploratory** and is labelled so in every table. This
costs nothing and is exactly the discipline the expert feedback is asking for.

### M2 — P11 is specified as an accept-the-null test at ±1σ, and contradicts its own kill criterion (`:314` vs `:924`)

P11 predicts *"every zero-door arm measures inside ±1σ of its control"*. Under a true null,
~32% of arms fall outside ±1σ **by construction** — so P11 as written is expected to "fail" on
about a third of zero-door arms even if the four-door screen is perfect. Meanwhile the kill row
at `:924` uses a completely different threshold: *"any zero-door arm with a reproducible >3%
effect ⇒ there is a fifth door."* At the boundary rig, 3% is ~20σ. **Fix:** restate P11 as a
**pre-specified equivalence test** (TOST) against a declared margin — the natural margin is
LAW-60's 3% smallest-callable screen delta — with a declared number of zero-door arms and a
declared per-arm K. Then P11 becomes a real, passable, falsifiable claim, and it agrees with
`§7.1`.

### M3 — The variance envelope is treated as a function of W at e2e and as a constant at the kernel level (`:367` vs `METRICS_PROTOCOL.md:357-375`)

R-N1 is one of the campaign's best findings: *"the envelope is itself a function of W … inheriting
the c512p band anywhere else is fraud."* The ladder applies it rigorously to serving and then
assumes a **single** `σ_campaign = 0.20%` and a single `σ_rig = 0.15%` across every kernel cell —
including T = 512 (K1), φ < 1 (K2), replayed skew (K3), and configurations adjacent to a known
hang (K4 at C = 32). There is no reason to expect a rig at T = 512 to have the variance of a rig
at T = 4096. **Fix:** extend R-N1 to the kernel instruments — every Phase-2 cell reports its own
measured envelope (the K repeats already provide it) and **re-sizes K upward** if the measured
envelope exceeds the assumed σ. Add "envelope_measured_this_cell" to `runs.jsonl`.

### M4 — Known-pathological and known-hanging points are budgeted at nominal cost (`:197`, `:386-388`, `:419`)

* **θ-P3 (ψ) sweep** at L ∈ {2,4,8,16,32} (`:197`) is priced inside P0-C's "35 campaigns × 3 min
  5 s". But the ladder's own citation says the L=2 case ran **≥10× slower and was abandoned after
  12 minutes** against a usual 25–95 s. One or two points in that sweep may cost 10–30 min each,
  or not terminate.
* **K4** sweeps C ∈ {0,8,16,24,28,32} at K=4 (`:419`) while `§3.2` R-K4 (`:386-388`) records
  that **C=32 hung 1 of 2 runs** and that `C*` sits one grid step below a hang. Four invocations
  at C=32 imply ~2 expected hangs; MoK hangs are handled with 1800 s timeouts in this project
  (`nightshift/LOG.md:188`). That is up to **+1 h** in one cell, unbudgeted.

**Fix:** price known-pathological points explicitly (a per-point timeout and an expected-hang
allowance), and add a **hang budget line** to `§3.8`. R-K4 already makes a hang a first-class
outcome — the budget must agree.

### M5 — K4's own K is inconsistent between §3.4 and §7.2 (`:419` vs `:936`)

`§3.4:419` gives K4 as `C{0,8,16,24,28,32} × flush_rows{0,8,16,32}` screened, **K = 4**, 36
campaigns, 1.9 h. `§7.2:936` describes the same cell as "**18 configs × K=2**" — also 36
campaigns. Both cannot be right, and R-K1 (`:379-381`) forces the issue: *"K = 4 for any cell
whose registered delta is <1% (the C fine-structure)"*, and the C fine-structure is exactly
C 24→28 at **−0.45%**. So the correct sizing is 18 × 4 = **72 campaigns ≈ 3.7 h**, nearly double
the budgeted 1.9. **Fix:** restate K4 as 18 configs × K=4 and reprice.

### M6 — Phase-0 P0-B duplicates Phase-2 B1 (`:175`, `:187` vs `:394`)

P0-B's ρ axis ("`k_inner` argv … until ρ ∈ {0.65, 1.0, 2.0, 3.0, 4.0}", 20 points × 3 repeats)
and B1 ("`k_inner` → ρ ∈ {0.65, 1.0, 2.0, 3.0, 4.0}", 4 arms, 20 pts, K=3) are the **same
experiment**, budgeted twice (1.3 h + 0.5 h) and — worse — assigned to opposite sides of the
fit/hold-out line (see B1). **Fix:** run it once, in Phase 0, label the whole ρ sweep FIT, and
delete B1 from Phase 2 (or keep B1 and remove ρ from P0-B). This is also a genuine *saving*
(~0.5 h) and it removes a hold-out violation.

### M7 — The gfx942 cell is inside the base budget although it is access-gated (`:454`, `:506`)

`§3.5:454` marks K8 "access-gated" and `§0.3:87` lists "one access gate: **gfx942 node**". Yet
`§3.8:506` includes ARCH 1.0 h in the Phase-2 total, and Region 6 (`§6.6`) plus predictions
P3b/P5b depend on it. **Fix:** move K8 to a conditional line like B9 ("+1.0 if the node
materializes") and state the fallback: without gfx942, the ARCH axis is a **declared coordinate
with a prediction and no measurement**, exactly as the precision axis is handled at `:456`.

### M8 — The two-week backlog schedules under half the budget, omits 5 of 17 cells, and violates the campaign's own gate order (`:945-958` vs `:53-54`, `:62`)

Summing the node column of `§7.3`: Phase 0 (≈6.0) + K5 0.8 + B3 0.4 + B4 0.7 + K4 1.9 + K1 0.75 +
B5 0.5 + B6 0.3 + B7 0.3 + B8 0.4 + K3 2.5 + K2 1.6 + B1 0.6 ≈ **16.8 node-hours over 14 days**,
against a stated 38.5–48.5. Missing from the 14 days entirely: **B2** (ranked #3 in `§3.7:484`
and described as the cell that *"unblocks the entire decode branch"*), **K6**, **K7**, **K8**,
**B9**, and **all of Phase 4** (16.7 h). Day 14 is "go/no-go on Phase 4" — so the plan is
2 weeks + an unscheduled ~17 h of serving work.

Worse, the schedule breaks the gate order the document opens with (*"Phase 2 may not start on a
cell whose prediction is unregistered"*, `:53-54`): **day 2 runs B1**, a Phase-2 cell adjudicating
P2/P2c/P5d, while **day 4** is "Phase-1 registration with **banked** θ, committed" and days 6–7
are the calibrated re-registration. A Phase-2 cell run before its registration commit "produces a
number, not evidence" by the doc's own rule 4 (`:26-27`).

**Fix:** (a) commit `predictions.jsonl` with banked θ on **day 1** (the content already exists as
`§2.3` — it is a `git add`, not a research task); (b) re-label day 2's run as **P0-B calibration**
(consistent with M6) rather than B1; (c) publish a realistic calendar: 14 days covers Phases 0–2
partially, and Phase 4 needs its own two-week block.

### M9 — Phase-4 pair sizing violates the rotation-balancing rule and re-uses banked pairs from a different configuration (`:600-601`, `:607-610`)

* **E-B is "5 pairs (2 banked)"** and **E-C is "folded into E-B as a 3rd arm"** (`:600-601`). The
  campaign_v5 review already established that **pair count must be a multiple of arm count — "6
  not 5"** (`nightshift/LOG.md:188`). A 3-arm cell therefore needs 6 rotations, not 5 pairs. And
  the 2 banked c512p pairs are **2-arm** pairs from a different campaign configuration; the
  project's own rule is that *"absolute tok/s before and after are not comparable; the ratios
  are"* (`WORKLOAD_EVIDENCE.md:543-544`), so they cannot be counted toward a 3-arm rotation set.
* **Phase 4b, E-F**: "concurrency ladder C ∈ {64,128,256} × 3 arms ≈5 h" (`:607-609`). At the
  ladder's own effective serving cadence (~55 min per pair, back-derived from E-A's 5 pairs /
  4.6 h), one 3-arm rotation is ~1.4 h, and ≥5 rotations per C-point is ~7 h ⇒ **~21 h for three
  C-points, not 5**. E-G (2 placements × 2 arms) is similarly ~2× under.
* **R-L3 confirmatory run unbudgeted.** `§5.3:628-636` correctly derives that c512p **fails**
  waves ≥ 20 and Little's ≥ 0.95, and recommends "one confirmatory long c512p pair at 10,240
  prompts" at **≈2 h/arm** — i.e. ~4 h. That appears nowhere in the budget.

**Fix:** reprice Phase 4 with rotation-balanced arm counts and the R-L3 long pair; drop the
banked pairs from the count (keep them as prior evidence); and state the concurrency ladder's
true cost (it is the "single most quotable number" — it deserves an honest price, not a 4× cut).

### M10 — No execution-robustness design for a node with a documented multi-hour failure rate

The operational record for this week alone:

* **~8-hour outage** — "NODE UNREACHABLE since ~09:00 UTC" (`nightshift/LOG.md:170`), back at
  17:02 UTC (`:286`), with the failure at the ssh/network layer (`sshd resets banner exchange
  30+ attempts/70 min; TCP open; port 22 closed`, `:227`; `REPORT.md:544`).
* **Queue killed 08:16 UTC** and local agents killed by laptop close, with work stranded on the
  node (memory: `nightshift-orchestrator-2026-08-18`, `overnight-queue-state-2026-08-18`).
* **Node IP changed mid-campaign** (10.5.95.87 → 10.0.0.228), and a shared machine requiring a
  foreign-process check in node-prep (`LOG.md:247`).

The ladder budgets **20% retries in Phase 0 only** (`:243`) and zero contingency in Phases 2 and
4, and says nothing about what happens when a session dies mid-cell. That matters statistically,
not just logistically: a mid-cell outage silently converts **same-session** paired campaigns into
**cross-session** ones, which re-imports exactly the day-scale drift (±15% e2e; 6.6% kernel
screen drift) that `§3.2`'s whole σ derivation assumes away.

**Fix.** (a) Apply a **node-availability derate** of 0.6–0.7 to every wall-clock estimate and say
so. (b) Add a **session-integrity rule**: a paired contrast whose members span a session boundary
is `voided_by: cross-session` unless the anchor series (B5/B6) shows drift below the cell's
envelope. (c) Require the runner to be **idempotent and manifest-resumable** — every invocation
writes its `runs.jsonl` row before the next starts, and a resume replays only missing rows. (d)
Order cells so that the highest-information cells (`§3.7` ranks 1–5) complete **within** the first
session rather than being spread across it.

### M11 — The analysis/manifest pipeline is an unbudgeted engineering item, and it is on the critical path (`:268-284`, `S_VECTOR.md:692-910`)

`§0.3` asserts *"the critical path is not node-hours — it is five unbuilt actuators"* and lists
X1–X5. It omits the one piece of software that every single number depends on: the manifest and
analysis pipeline. At ~570 kernel invocations + ~25 serving pairs, someone must build:
`predictions.jsonl` registration/diff tooling; `arms.jsonl → cfgs.txt` projection; per-run QC
(`.text` sha, descriptor echo, mechanism invariant, hang classification — `S_VECTOR.md:892-905`);
drift-anchor bookkeeping; paired-difference and blocked-design analysis; θ-CI computation;
regret/Spearman/boundary-location scoring. `§7.3:947` allocates this to day 1 as "scaffold
`manifests/`" with no size. **Fix:** add it as **X6 (size L)**, before Phase 2, and state the
engineer-day budget for X1–X6 alongside the node-hours — the document currently prices only GPU
seconds while claiming the critical path is engineering.

### M12 — Cell-ID collisions across the sibling documents will corrupt the manifest keys (`§3.3/§3.4` vs `W_VECTOR.md:797-838`)

`cell_id` is a primary key across `predictions.jsonl` / `cells.jsonl` / `arms.jsonl` / `runs.jsonl`
(`:268-284`), and the ladder declares the siblings authoritative on disagreement (`:8`). But the
same short ids mean different things in the two documents:

| id | `EXPERIMENT_LADDER` | `W_VECTOR` |
|---|---|---|
| **K2** | fill φ at fixed T_pad (`:422`) | routing/skew ladder (`W_VECTOR.md:826`) |
| **K3** | skew (`:421`) | fill φ (`W_VECTOR.md:827`) |
| **K4** | consuming pool C (`:419`) | `k_inner` / ρ (`W_VECTOR.md:828`) |
| **K5** | the doors 2×2 (`:418`) | message size / λ (`W_VECTOR.md:829`) |
| **K6** | top-k (`:423`) | granularity decomposition / FQ-3 (`W_VECTOR.md:832`) |
| **K7/K8** | decode DP / gfx942 (`:424`, `:454`) | gfx942 (`W_VECTOR.md:833`) |
| **E1–E7** | — | serving cells, vs the ladder's `E-A…E-G` (`:599-610`) |

Two of these collisions are actively dangerous: `W_VECTOR`'s K6 (the FQ-3 falsifier) simply
vanishes under the ladder's K6, and the E-numbering divergence means `W_VECTOR`'s **E3 (ISL
sweep)** and **E6 (`max_num_batched_tokens`, "the largest modeled reach … never swept", H12)**
have no counterpart in the ladder's Phase 4 at all — even though `§3.1:346` cites "E3/E-F" as the
cell that breaks the φ ⟂ C_req collinearity, and E-F is the concurrency ladder, not ISL.
**Fix:** publish one canonical cell registry (a table mapping every sibling id to one
`cell_id`) at the top of the manifest scaffold, and make it the only source of ids; then re-check
that every collinearity-breaking cell named in `§3.1` actually exists in `§3.3`/`§3.4`/`§5.2`.

---

## 3. Re-budget (the arithmetic, corrected)

Assumptions stated, per finding. Unit costs: boundary invocation **1 min** (uncited — see B3;
double every boundary line if it is 2 min), MoK campaign **3 min 5 s** (LAW-60, cited), serving
pair **~55 min** (back-derived from `:599`'s E-A row, which is more honest than S_VECTOR's 25 min).

| line | ladder | corrected | why |
|---|---:|---:|---|
| P0-A transport | 1.6 | **2.1** | 126 inv × 1 min (B2) |
| P0-A0 repeatability arm | — | **+0.5** | B4 — new, mandatory |
| boundary cadence measurement | — | **+0.1** | B3 |
| P0-B boundary + ledger | 1.3 | **1.3** | ok (but see M6 — it now *contains* B1) |
| P0-C mega chassis | 1.8 | **2.3–3.8** | ψ sweep pathological points (M4) |
| P0-D serving | 1.9 | **1.9** | ok |
| retries | 1.3 | **1.6** | 20% of the new base |
| **Phase 0** | **6.0** | **≈9.8–11.3** | |
| Phase 2 boundary (B1–B8) | 4.3 | **6.3** | pts×K = 408 inv (B2), minus B1 folded into P0-B (M6) |
| Phase 2 MoK (K1–K7) | 8.5 | **10.4–11.4** | K4 at 18×4 (M5) + hang allowance (M4) |
| ARCH K8 | 1.0 | **conditional** | M7 |
| **Phase 2** | **14.0** | **≈16.7–17.7** (+1.0 if gfx942) | |
| Phase 3 | 2.0 | **2.0** | ok |
| Phase 4a | 16.5/16.7 | **≈24–29** | rotation balance + R-L3 long pair + E-D at ≥5 pairs (B9, M9); or **≈18** if E-D/E-E are demoted to exploratory |
| **TOTAL (4a)** | **38.5** | **≈53–60** (≈**47** with E-D/E-E demoted) | |
| Phase 4b | 10.0 | **≈27–35** | M9 |

Then apply the node-availability derate (M10): at 0.6–0.7 effective, **≈53–60 node-hours ⇒
75–100 hours of wall on the node ⇒ 8–12 overnight sessions, not 4.** With E-D/E-E demoted and the
gfx942 cell excluded, a defensible target is **≈47 node-hours over 6–8 sessions.**

The honest way to present this is the same way `§0.2:69-76` already presents its delta against
`S_VECTOR §5.5`: state it, do not smooth it. A campaign that claims 4 overnights and takes 10 will
be read as the same over-reach the experts already flagged ("very expansive, unclear what the core
contribution might be").

---

## 4. What is right, and must not be lost in the fixes

* **`§3.2` derives K instead of asserting it** (`:357-388`) — the right instinct, and the MDE table
  is the correct shape. Fix its inputs (B4) and its design assumption (B5), keep its structure.
* **R-K4** (a hang is a manifest outcome, never a dropped point, `:386-388`) — this is exactly the
  discipline that turns a schedule-space boundary into data.
* **R-K3 / LAW-61** (runs are the unit, never (run, rank), `:384-385`) — the `t = −5.01` vs
  `t = −0.73` case is the single best argument in the document for its own statistics section.
* **`§0.3`'s X1–X5 table** — naming the unbuilt actuators up front is what separates this plan from
  a wish list. It just needs X6 (M11) and honest engineer-days.
* **`§5.3`'s self-applied gates** (`:628-643`) — deriving that our own banked c512p cell **fails**
  two of our own admissibility gates, in the plan, before anyone else finds it, is the most
  credible paragraph in the whole set. Keep it verbatim.
* **`§8` "what this ladder refuses to promise"** (`:962-987`) — keep it, and add to it the items
  this review forces: no decode adjudication without ≥5 pairs; no φ claim without an M24 receipt;
  no skew claim without an observed-CV receipt; no boundary claim before σ_rig is measured.

---

## 5. Minimum set of changes before synthesis

Ordered by cost-to-fix ÷ credibility-at-risk. Items 1–5 are edits; 6–8 are new node work totalling
≈0.6 h; 9–10 are engineering.

1. **Rewrite `§4.1`** to remove Phase-0 points from the hold-out and reconcile with
   `COST_MODEL.md:1041-1055`; add the PRIOR-VALIDATED row. *(B1)*
2. **Reprice `§0.2`, `§1.7`, `§3.8`, `§5.2`** as `pts × K × unit`, with the boundary unit as a
   declared range; fix 16.5 vs 16.7 and 38.5. *(B2, B3, M5, M7)*
3. **Add `§0.4` forwarding contract** (whitelisted knobs, descriptor echo, sanity ratio) and
   re-work per-rank C to ride `K0_MPS_CFG`. *(B8)*
4. **Add actuation-receipt admission gates** for K2 (fill counter), K3 (observed CV), per-rank C
   (per-rank descriptor echo). *(B7)*
5. **Fix the gates' teeth**: delete G0-c's or-declare branch; pre-register the scoring denominator;
   replace the ≤1-grid-step bar with a profile-CI criterion; restate P11 as TOST at a 3% margin;
   add FQ-1 and FQ-3 as cells. *(B5, B10, M1, M2)*
6. **P0-A0 repeatability arm** (20 spread invocations + 5 post-idle), ≈0.5 h. *(B4)*
7. **Boundary cadence measurement**, ≈0.1 h — do it in the first 10 minutes of the first session.
   *(B3)*
8. **G0-e thermal/clock parity**: continuous `amd-smi` clock+temp logging, first-run discard,
   randomized level order with a repeating anchor config. Free in node-hours. *(B6, M3)*
9. **X6 — the manifest/analysis pipeline**, sized and scheduled before Phase 2. *(M11)*
10. **One canonical cell registry** reconciling ladder / `W_VECTOR` ids, and a re-check that every
    collinearity-breaking cell named in `§3.1` exists somewhere in the plan. *(M12)*

Then re-publish the calendar honestly: Phases 0–2 in two weeks at 0.65 availability; Phase 4 in its
own block; E-D/E-E labelled exploratory unless funded to the ≥5-pair floor.

# Nightshift REPORT — 2026-08-18 overnight (draft r3, written 04:50–07:00 PDT)

> **Fairness-audited.** This draft was reviewed by the mandatory adversarial
> fairness-audit subagent; wording was downgraded in place on 8 items and three
> retractions (R4–R6) were added. The signed per-item verdict, the licensing
> statement, and the preconditions for any future "beats production" claim are
> in **§3.4**. Read §3.4 before quoting anything from this file.
>
> **r3 (07:00 PDT):** the fill provenance was recovered and reconciled —
> **R4 is DISCHARGED** (37.6 % / 2.66× is sourced and reproduced, with
> qualifiers), **R7 withdraws the corpus 1.19×** as an arithmetic artifact,
> the §4 contradiction box is **RESOLVED**, and §6 records a **security
> incident** (a provenance file quoting prior-session transcript records was
> committed and removed; history decision pending).

Status: **campaign not run.** The GPU node has refused ssh since ~09:00 UTC
(~02:00 PDT) and was still refusing at write time. Everything below is either
(a) built and committed locally, (b) measured at kernel level before the
outage, or (c) queued to run the moment the node answers.

---

## 1. Executive summary

**No end-to-end number was produced tonight, and no end-to-end number versus
genuine native production has ever been produced by this project.** The 20–30 %
e2e goal is therefore not "missed" — it is still **unmeasured against its own
definition**. Tonight's work went to making that first measurement possible and
to building the kernel that could plausibly clear the bar. What it produced:

1. **M24, the fill-aware megakernel** — designed (3 adversarial lenses,
   5 blockers fixed), implemented (+431/−4 in the M15 body behind 9 default-0
   macros), reviewed again (7 blocking + 9 major fixed, 1 rebutted with
   evidence), committed and pushed. It cannot be believed yet: the build gate
   and *every* hipcc compile are node-blocked, so no codegen evidence exists.
2. **Campaign v5.1** — the first campaign this project has ever *specified* an
   untouched production arm for, plus AMD's own recommended TP8+EP
   config, an open-loop client, a deterministic accuracy cell,
   rotation-balanced ≥5-pair statistics and a schema-safe analyzer. Built,
   adversarially reviewed (12 blocking + 14 major fixed), committed,
   **not yet deployed**. *(Audit note: the arm is untouched **by construction
   only**. Checklist item 1 requires authenticity to be established from
   evidence — image digest, `docker inspect` env, server.log config dump,
   absence of every patch marker — captured from an actual run. No such
   evidence exists, so "genuinely untouched" is not yet a verified property.)*
3. **Method hardening** — SERVING_BENCHMARK_METHODOLOGY.md rev 3 (binding
   protocol folded in; camp3 demoted to directional-only), DECOMP_RUNBOOK
   amendments, a one-line harness fix unblocking every non-replication MoK
   arm, and retirement of the last pre-M23 serving numbers in both repos.
4. **Two measurements and six retractions** (§2) — the R0a ratchet holds at
   0.75187 (a **kernel-region, 100 %-fill** ratio, not an e2e number), and the
   M15 chassis was **shown not to certify at T ≠ 4096** (a correctness-gate
   failure at two points plus a control, not a proof of non-parametricity),
   which voids the cheap R6 fill experiment and re-routes all fill evidence
   through M24.

State of the 20–30 % goal, stated honestly: **the modeled win is now
distribution-dependent and unresolved.** The fill mechanism's post-review cost
model predicted +16 % to +31 % vs rescued/patched-stock at c32p (central
≈ +22 %, UNKNOWN vs native) at the receipt's 37.6 % fill — but tonight's
G0b-local (§2 M11, §4) finds the corpus fill distribution is **bimodal** with
mean fill 0.69 and only **1.19×** aggregate padding, which models to a
**modeled +11–17 % e2e** — a kernel-region model with φ *assumed*, projected
through the MoE fraction, against **patched-stock; UNKNOWN vs native** — and
under that model essentially all of it is captured by tier 1 alone. The
37.6 %-receipt view and the corpus view **contradict each other** and the M24
win estimate stays **WITHDRAWN** until a whole-run `n_orig` histogram from seal
receipts settles it (§7 item 1). The design
explicitly says M24 alone is not a safe promise for 20–30 % and promotes the
κ = 1.192 step-count composition defect to a co-equal work item. The hybrid
caveat is unchanged and applies to every framing of the goal: **the mega
inverts below ~1,600–1,800 tokens/rank**, so small-batch cells are expected
losses and must be reported, not hidden.

---

## 2. Measurements — every number quoted tonight

Validity classes: **SERVING** = full server e2e; **KERNEL** = MoK synthetic
rig, never an e2e headline; **CORPUS** = offline analysis of captured routes.

| # | Measurement | Level | Baseline name | Workload spec | Coverage receipt | Validity |
|---|---|---|---|---|---|---|
| M1 | camp3 pair_01: m15 19,277.36 vs stock 15,611.17 input tok/s → **+23.49 %** | SERVING | **patched-stock** (not native) | c32p: conc 32, 1024 prompts, ISL 4096, OSL 8; prompt SHA `b6271f04…` identical across arms | receipt cited below is **pair_02's**; no per-pair receipt is quoted for pair_01 in the harvest | single pair, m15-first — **not evidence alone** |
| M2 | camp3 pair_02: m15 17,899.96 vs stock 20,588.66 → **−13.06 %** | SERVING | patched-stock | as M1, stock-first | **the quoted receipt belongs to THIS arm**: `RAGGED_SEAL_RECEIPT` pair_02 m15 rank=1 steps=300 in_bucket=259 sealed=251 → **96.9 %** sealed/in_bucket (83.7 % sealed/steps), `uniform_rescued=149`, `min_orig=1 max_orig=4096` | single pair — **not evidence alone** |
| M3 | camp3 order-balanced n=2: **mean +2.70 %, geomean +3.61 %**, spread +23.5 % / −13.1 % | SERVING | patched-stock | as M1 | 96.9 % (pair_02 m15 arm) | **DIRECTIONAL ONLY.** n=2 is below the binding ≥5-pair floor; the ±18 % position effect fully swamps a +2.7 % mean. **This is a kernel-substitution diagnostic against our own patched integration. It is not a production comparison and may never be cited as one, in any context — including any comparison against vendor-published figures** |
| M4 | **R0a re-anchor: ratio 0.75187** (median of 5 order-balanced; runs 0.75008 / 0.75054 / 0.75321 / 0.75187 / 0.75292). M15 p50 5,824.84 µs, production-ops p50 7,740.80 µs. **Direction: ratio = mega ÷ production-ops, so < 1 means the mega is FASTER in-region (≈ 1.33× at 100 % fill)** | KERNEL | **production-ops-replay inside the MoK harness** (synthetic; not a server, not production) | MoK synthetic prefill, T=4096, **100 % fill**, C=28,g=353,mode=12,flush_rows=16, PADMAX=263136, warmup 500 / timed 100, route replay NOT deployed | MoK gate pass all 5 runs, rel_err 0.008295; pin `~/DHK-m15` sha `935f555e…` byte-identical to the banked 0.7556 arm | **PASS** vs banked 0.7556 (−0.49 %, inside ±1 %). Harness + anchor sound, ratchet unconfounded. **Not an e2e number, in either direction**: it neither shows the mega winning end-to-end nor losing end-to-end. Serving runs at a disputed and materially lower fill (§4 contradiction box); this ratio is an unconfounded regression anchor and nothing more |
| M5 | R6 T-sweep on the M15 body at T=2048 / T=1024: **no timing** — `[MOK GATE] mps_mega pass=False`, relative **0.874205 / 0.838890**, while `production` passes at 0.005820 / 0.005745 in the same runs | KERNEL | n/a (blocked pre-timing) | as M4 with K0_T=2048/1024, PADMAX = 64·T+992 | gate FAIL is itself the receipt | **BLOCKED BY CORRECTNESS.** G11 answered negatively: the M15 chassis certifies only at T=4096 |
| M6 | tgen-body control at T=2048: **ratio 0.8959**, gate pass (max_abs 0.042969, rel 0.008288) | KERNEL | production-ops-replay | identical tree/env/MPSCFG/PADMAX as M5, pin `~/DHK-tgen` sha `20b8c6bb`; **n = 1 (smoke run, `tgen_smoke_t2048_0818T0845/run1.log`), no spread** | gate pass | valid **for the tgen body only**, and as an n=1 smoke point the ratio itself is not quotable to 4 digits; its load-bearing content is the **gate pass**, which is what shows M5 is body-specific, not environmental. **Not an M15 number** |
| M7 | T=4096 phase ledger: plan 372.29 µs, M6 2,454.95 µs, M7 2,201.61 µs, combine 213.93 µs, m2_to_end 5,242.78 µs; `[MPS SPIN] chunk_poll fail_max=0` | KERNEL | n/a | as M4, timestamps=1, 1 run, rank-0-only final-soak-epoch semantics | — | diagnostic; shape matches banked exp_33. fail_max=0 ⇒ no transport degradation at 100 % fill |
| M8 | **31 % of sealed calls are 100 % dummy work**; real-row max-rank-load p50 **2.61×** / p90 3.35× (**corpus-derived, same capture bias**); consecutive-row expert overlap 57.0 % vs 55.7 % shuffled (**1.3 pt**) | CORPUS | n/a | 512 calls × [4096,8] across 8 workers, routecap1 capture (m15+M23 server, c32p), first 64 B4096 calls/worker | matches whole-run receipts' `uniform_rescued` ≈ 34 % | the 31 % is a **routing-homogeneity proxy, not an `n_orig` measurement** (early-run capture bias; direct measurement deferred to G0b) |
| M9 | **NO e2e ratio vs native production exists — tonight or ever.** camp5_native never ran; v4's "native" arm was config-mirrored (§6) | — | — | — | — | **stated, not measured** |
| M10 | **G7 — T-generality**: `hkp::csr_scan_block256` and `hkp::pull_src_fill` are correct for every `T ∈ [0, T_cap]`; minimum safe `K0P6_M24_TGRAIN = 1`, ship **8** (`-DK0P6_M24_TGRAIN=8`); `T_eff = 0` safe at all five substitution sites; §F.4's N=0 point resolves to the **unclamped** branch | ANALYSIS (local, no node) | n/a | source proof over `amd-master/.../hkp/hkp_sort.hpp` + `k0pf6gm_device_tile_m15.hip` post-M24 line map | n/a — static proof, not a run | **valid as a correctness argument**, commit `9ce36341`. Not a performance number. Also finds: `ZERO_PAD` write volume is **anti-correlated with N**, so the φ fit needs a calibration arm or a fixed regressor |
| M11 | **G0b-local (corpus surrogate)** — fill is **bimodal**: 31 % dummy calls / ~60 % completely full / ~10 % genuinely partial; mean fill **0.69**; aggregate padding **1.19× pre-M24 → 1.01× post**. Modeled MoE-region ρ @ φ=0.9: **tier-1-only 0.719 (1.39×) beats full M24 0.730 (1.37×)** ⇒ under this distribution **tier 1 captures the entire modeled win** (~**+11–17 % e2e** at the 40–54 % MoE fraction) | CORPUS → **MODEL** | n/a | 512 rank-calls (8 workers × 64 B4096 calls), routecap1 capture at c32p; `ρ = (1−φ) + φ·T_eff/4096` averaged over the empirical distribution, φ **assumed** 0.80/0.90 | n/a | **NOT A MEASUREMENT.** Modeled, kernel-region-only, φ unmeasured. Carries **first-64-calls capture bias** (dummy *rate* cross-validates at 31 % vs ~34 % receipts; the non-dummy **fill shape** does not) and **rank-call, not step, granularity** — a step-level tier-1 skip needs ALL 8 ranks dummy, so tier-1 numbers **overstate**. Commit `619b6308` |
| M12 | **M24 dummy-step rank-synchrony — SYNCHRONOUS.** Per-index dummy count `c[i]` is bimodal on **{0, 8}** for all 64 indices (**100 %** of indices vs a **5 %** independent-rank expectation), pairwise agreement/kappa **1.000** over all 28 worker pairs, and all 8 workers show an identical **31.2 %** dummy rate. Index alignment independently validated by a sharp lag-0 cross-rank `n_real` correlation peak (**r = +0.996** vs ~0.65 at ±1). **Consequence: a tier-1 whole-step skip carries 1:1 from rank-calls to steps with NO synchrony discount** | ANALYSIS (corpus-based, local; no node) | n/a | same 512 rank-calls (8 workers × 64 B4096 calls) as M8/M11, routecap1 capture at c32p; `dummy_synchrony.py`, classifier reused verbatim from `g0b_local_fill.py` | n/a — analysis, not a run | **valid for the capture window only.** Commit `2ba124c8`. **Caveat: the window's dummies are entirely ramp-in (idx 0–13) and drain-out (idx 58–63) blocks — zero mid-run dummy calls appear**, so the whole-run ~56 % mid-run dummy steps are not directly covered. Design rule unchanged: the skip remains a **collective** decision; local `n_orig == 0` is a proposal, never the commitment |

### Retractions issued tonight

* **R1.** Banked `t2048_m15_bal_0814T1937` (0.8912) and `t1024_m15_bal_0814T1937`
  (1.1051) carry pf6mps source sha `20b8c6bb` = **DHK-tgen**, not the M15
  chassis sha `935f555e`. They are **not M15 data** and must never be quoted
  as an M15 fill result. This is the pin trap, realized inside our own archive.
* **R2.** The banked camp3 seal coverage of "99 %" is optimistic; the receipt
  reads **96.9 %** sealed/in_bucket.
* **R3.** The banked camp3 headline "+2.7–5.2 %" is more precisely
  **+2.70 % mean / +3.61 % geomean at n=2** and is directional only.
  **The string "+2.7–5.2 %" is retired everywhere and must not reappear**,
  including in framings that compare it to vendor-published ranges.
* **R4 — DISCHARGED (r3).** R4 suspended "37.6 % fill / ~1,539 real
  tokens/rank/step / 2.66× padding" as un-sourced. The provenance has since
  been recovered: the figure derives from the **M23PAIR2 P1 m15 c32p
  `RAGGED_SEAL_RECEIPT` lines**, as
  `Σ(in_bucket_sum_orig) / Σ(in_bucket) / 8 ≈ 1,541 / 4,096`. It reproduces
  to **3 s.f.** (ranks 0 and 7 agree to 0.04 %) and is corroborated by an
  **independent stock-arm check at 43.3 % fill / 2.31×**. The suspension is
  lifted; **2.66× may again be used as a design premise**, but only with these
  qualifiers attached: (a) it is per-rank per **IN-BUCKET step**, not per
  "sealed" step (the earlier wording was wrong, ≤1 % numerically);
  (b) **whole-run** aggregate; (c) it **includes ~56 % DP-dummy /
  `uniform_rescued` steps** at `n_orig ≈ 1` — it is not the non-dummy fill;
  (d) **token rows, not top-8-expanded rows**; (e) **n = 1 arm**; (f) the
  **`/8` peer-count convention is assumed**, with two strong indirect proofs
  but the receipt printf never read — a grep on the node closes it (§7).
  Source: provenance reconstruction, held locally; conclusions summarized
  here; receipt re-verification queued on the node.
* **R7 (new, r3). The corpus-derived aggregate padding multiplier "1.19×" is
  WITHDRAWN as an arithmetic artifact.** `g0b_local_fill.py:89` filters
  `n_real > 0`, silently dropping the **91 zero-fill calls**; the honest
  all-512-rank-call aggregate is **1.446×**. The string "1.19×" must not
  reappear as the corpus padding multiplier. The corpus's **shape** findings
  (bimodal distribution, non-dummy calls ~0.91 full, tier-2 row plumbing weak,
  tier-1-dominant) are unaffected and in fact strengthen. Source: provenance
  reconstruction, held locally; conclusions summarized here; receipt
  re-verification queued on the node.
* **R5 (added by the fairness audit).** Commit `4a5d87d7`'s message and the
  matching `CORPUS_FINDINGS.md` framing say "**31 % of sealed steps** are
  100 % fake work". The measurement is over **rank-calls**, not steps, and a
  step is 8 coupled rank-calls (G0b caveat 3). The "steps" wording is
  **retracted**; only the M8/M11 "calls" wording is correct.
* **R6 (added by the fairness audit).** `LOG.md`'s 02:45 entry ("+25–40 % e2e,
  central +31 %; tier-1 alone +12–17 %") is formally **retracted**, not merely
  superseded — the footnote at the end of this report explains the revision but
  did not retract the numbers. They must not be quoted from the log.

---

## 3. FAIRNESS_AUDIT

**Global verdict for this report: NO "beats production" claim is made, and
none is permitted tonight. There is no headline to audit.** The mandatory
fairness-audit subagent **was launched** against this draft (r2) and its
signed, per-item verdict block is appended at **§3.4**; the original claim
here that it was "correctly not launched" was itself wrong — the checklist
governs every quoted number, not only headline claims, and the audit found
violations in numbers that were being quoted. It **must** run again, with
veto, before any campaign_v5 number is quoted anywhere.

### 3.1 camp3 (M1–M3) — verdict: **DIRECTIONAL DIAGNOSTIC ONLY, NOT A PRODUCTION CLAIM**

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Baseline authenticity | **FAIL for any production claim** | the "stock" arm's server.log carries `PF4H_M23_RAGGED_SEAL_V1`, `PF4H_COVERAGE_PATCH_V1`, `PF4H_B4096_GRAPH_TARGET`, `PF4H_EPLB_*_RELAX_V1`. It is patched-stock. This is *why* M3 is directional and can only ever be a kernel-effect diagnostic |
| 2 | Baseline not sandbagged | **N/A / UNKNOWN** | no native arm existed to sandbag; patched-stock ran our integration's config on both sides (symmetric) |
| 3 | Candidate actually ran | **PASS (96.9 %)** | `RAGGED_SEAL_RECEIPT sealed/in_bucket = 251/259`; the pre-M23 inert-kernel failure does not apply |
| 4 | Identical workload | **PASS on inputs** | exact prompt SHA `b6271f04…` across all arms; same cell, client, warmup; ARM_COOLDOWN=240 active |
| 5 | Statistical validity | **FAIL** | n=2 pairs, spread +23.5 % / −13.1 %, against a measured ±18 % position effect and ±15 % day drift. Below the binding ≥5-pair floor |
| 6 | Accuracy parity | **OPEN / UNRESOLVED** | `ordered_output_token_id_stream_sha256` differs across **every** arm including stock-vs-stock (`4ee5ba42…` vs `936b4f5a…`). Diagnosed tonight as numeric, not a bug: chunked-prefill boundaries, prefix cache, DP-rank assignment, combine order. Exact-token SHA cannot currently validate anything at c32p |
| 7 | Replay/proxy honesty | **PASS** | serving-level, labeled as such; regime is real serving fill (~37.6 %), not the 100 %-fill kernel bank |
| 8 | Claim wording | **ENFORCED** | quoted only as "+2.70 % mean vs **patched-stock**, n=2, directional"; never as "vs production" |

### 3.2 R0a (M4) — verdict: **VALID ANCHOR, KERNEL-LEVEL ONLY**

* Items 3/4/5: PASS — MoK gate pass all 5 runs (rel_err 0.008295);
  order-balanced, same synthetic driver both arms; median of 5, spread
  0.75008–0.75321 (±0.2 %).
* Item 7 (**the governing item**): a **synthetic MoK rig at 100 % fill against
  production ops replayed inside the same harness**. Not an e2e ratio, not a
  serving number, never to appear in a production comparison. Serving runs at
  ~37.6 % fill; every banked kernel ratio including this one is 100 %-fill.
* Items 1/2/6/8: N/A — no production deployment involved.

### 3.3 R6 / tgen (M5, M6) and corpus (M8) — verdict: **NO PERFORMANCE CLAIM**

M5 produced no timing, so there is nothing to audit; it is a blocked
experiment. M6 is a valid kernel-level ratio **for the tgen body** and item 8
wording is enforced — it is not an M15 number. M8's 31 % is a
routing-homogeneity proxy over the first 64 B4096 calls per worker, not a
measured `n_orig` distribution (item 7): the design doc carries a no-quote
rule on it and defers Tier-1 sizing to G0b. Tonight it *orders* work only.

### 3.4 SIGNED AUDIT VERDICT — fairness-audit subagent, report r2

Adversarial audit of **every** quoted item in r2 (§1 executive summary, §2
table M1–M11, retractions, §4 mechanism/model, §6 discoveries, §7 actions).
Rubric: `nightshift/CLAUDE.md` items 1–8 + `nightshift/BENCHMARK_PROTOCOL.md`.
Primary sources checked: `tasks/{wuh4rxbnz,wvdrblr5d,wff3mewsd,wp5jobxxv,w3clrpi59}.output`,
`m24/G0B_LOCAL_FILL_HISTOGRAM.md`, `m24/G7_TGRAIN_ANALYSIS.md`,
`FILL_AWARE_DESIGN.md` rev 3 §A.4, and `git log` for all 11 cited commits
(all 11 exist on `ablations` and match their stated content).

**Counts: 5 PASS · 8 PASS-WITH-EDITS · 0 VETOED.** Edits are applied in this
file, not merely flagged.

| Item | Verdict | Items checked | Finding / edit | Evidence |
|---|---|---|---|---|
| **M1 / M2** camp3 pairs | PASS-WITH-EDITS | 1,3,4,5,7,8 | Arithmetic reproduces exactly from source. Baseline correctly named patched-stock. **The coverage receipt was attached to M1 (pair_01) but the harvest quotes it from the pair_02 m15 arm** — receipt re-attributed to M2, M1's receipt marked absent | `wuh4rxbnz.output` queue.report: `19277.358…/15611.167…`, `17899.956…/20588.660…`; receipt string `pair_02 … rank=1 steps=300 in_bucket=259 sealed=251` |
| **M3** n=2 mean/geomean | PASS-WITH-EDITS | 1,4,5,8 | Correctly labelled directional; n and spread stated; ±18 % drift named. **Edit:** added an explicit prohibition on citing it as a production comparison *in any context*, because §6 did exactly that against vendor figures (see below) | `wuh4rxbnz.output`; `BENCHMARK_PROTOCOL.md` §3.1 (≥5 pairs) |
| **M4** R0a 0.75187 | PASS-WITH-EDITS | 3,4,5,7 | Anchor is sound: 5 order-balanced runs, spread ±0.2 %, gate pass ×5, pin sha byte-identical to the 0.7556 arm, route-replay absent so unconfounded. **Edit:** the row never stated the ratio's *direction*, leaving 0.752 readable either as "mega 25 % slower e2e" or as an e2e win. Direction and a two-way non-implication clause added | `wvdrblr5d.output` r0a: per-run list, `935f555e…`, `rel_err 0.008295` |
| **M5** R6 blocked | PASS | 3,6 | No timing ⇒ no claim. Gate FAIL is the receipt; control (M6) isolates it to the body | `wvdrblr5d.output` sweep T=2048/1024, `mps_mega pass=False rel 0.874205/0.838890` vs `production pass=True 0.005820/0.005745` |
| **M6** tgen 0.8959 | PASS-WITH-EDITS | 5,7,8 | Body scoping was already enforced. **Edit: n was not stated — it is n=1** (a smoke run). A 4-digit ratio from one run is not quotable; the row now says so and relocates the load-bearing content to the gate pass | `wvdrblr5d.output` anomaly: `tgen_smoke_t2048_0818T0845/run1.log` |
| **M7** phase ledger | PASS | 7 | Diagnostic, 1 run, labelled; `fail_max=0` scoped to 100 % fill | `wvdrblr5d.output` ledger_highlights |
| **M8** 31 % / 2.61× | PASS-WITH-EDITS | 5,7,8 | The 31 % is correctly labelled a routing-homogeneity proxy over calls. **Edit:** the 2.61× skew rode along unlabelled and is quoted again in §7 as "measured"; both now carry the corpus/capture-bias label. **See also R5** — the repo's own commit message says "sealed **steps**", which is wrong and is now retracted | `G0B_LOCAL_FILL_HISTOGRAM.md` §1 (160/512 = 31.2 %), §5.3; commit `4a5d87d7` message |
| **M9** no native ratio | PASS | 1,8 | The single most important sentence in the report and it is stated without hedging | `wuh4rxbnz.output` caveat 2 ("camp5 never ran") |
| **M10** G7 T-generality | PASS | 7,8 | Correctly typed as a static source proof, explicitly "not a performance number"; the anti-correlated `ZERO_PAD` bias is disclosed rather than buried | `G7_TGRAIN_ANALYSIS.md` §7 (TGRAIN=1), :273 (unclamped), :293 (anti-correlation), :304 (fence) |
| **M11** G0b model | PASS-WITH-EDITS | 5,7,8 | The strongest-labelled item in the report: "NOT A MEASUREMENT", φ assumed, capture bias, over-states tier 1 — all present and all traceable. **Edits:** (i) the "1.01× post-M24" figure is arithmetic assuming M24 works, on a kernel that has never compiled — now said; (ii) §1's "+11–17 % e2e" needed its baseline (patched-stock / UNKNOWN vs native) and modeled status attached at the point of use | `G0B_LOCAL_FILL_HISTOGRAM.md` §3 (1.189→1.011), §4 (ρ 0.7188 / 0.7296), §5.5–5.6 |
| **§4 cost model** ρ, +16–31 % | PASS-WITH-EDITS | 5,7,8 | Numbers match the design doc exactly. **Edits:** "2.0–2.4× less MoE work" was asserted in bold without the "modeled" qualifier and rests on the suspended `f = 0.376`; the "+16–31 %, central +22 %" prediction was presented in bold *after* §1 declared it withdrawn — it is now stamped WITHDRAWN in place, with the secondary-baseline and hybrid-caveat clauses attached | `FILL_AWARE_DESIGN.md`:189–194, :233, :242–243, :1345 |
| **§4 contradiction box** | PASS-WITH-EDITS | 5,7,8 | Structurally honest — it does refuse to quote either multiplier and does state the decision consequence for tier 2 in both directions. **But it was not symmetric:** it enumerated three corpus-side defects and *zero* receipt-side defects, while §1/§4 simultaneously adopted the corpus number for the new modeled estimate. The decisive asymmetry the box omitted: **the 37.6 % / ~1,539 figure is un-sourced in this entire session.** Receipt-side caveat added; **R4 added** suspending 2.66× as a design premise | `G0B_LOCAL_FILL_HISTOGRAM.md` §5.4; M2 receipt (`min_orig=1 max_orig=4096`); no session artifact derives 1,539 |
| **§6.3 AMD figures** | PASS-WITH-EDITS | 1,2,7,8 | **Worst violation in r2.** "+16–47 %" and "1.2–1.5×" are vendor marketing claims for other hardware/workload/version points, and the sentence *"Both ranges swallow our banked +2.7–5.2 %"* welded them to a retracted (R3) n=2 patched-stock diagnostic — manufacturing a production-vs-us comparison from two components each individually barred from making one, and resurrecting the exact string R3 had just retired. Sentence **struck**; a vendor-claim label with the real provenance added; R3 extended to retire the string outright | `w3clrpi59.output` tuned.report §6: "no absolute vLLM tok/s … published", 1.52× @ conc 64 / 1.35× @ conc 128 on ISL 10K/OSL 1K, R1 absent from MLPerf v6.0 |
| **§1 campaign-v5.1 "genuinely untouched"** | PASS-WITH-EDITS | 1 | Untouched **by construction**; checklist item 1 demands evidence (digest, env, config dump, patch-marker absence) from an actual run, and the campaign has never been deployed. Downgraded to "specified", with the evidentiary requirement stated | `wp5jobxxv.output` (built, not deployed); §5 row `a776ac75` |
| **§1 "proven not T-parametric"** | PASS-WITH-EDITS | 5,8 | Evidence is a correctness-gate failure at two T points plus one control — enough to void R6, not enough for "proven". Downgraded to "shown not to certify at T ≠ 4096" | `wvdrblr5d.output` anomaly G11 |
| **Retractions R1–R3** | PASS-WITH-EDITS | 5,7,8 | R1 (pin trap) and R2 (99 %→96.9 %) are correct and sufficient. **They did not retract enough:** three live over-claims survived — the 2.66× premise on which Priority 1 rests, the "sealed steps" wording committed in `4a5d87d7`/`CORPUS_FINDINGS.md`, and `LOG.md`'s 02:45 "+25–40 %/+31 %/tier-1 +12–17 %" (the footnote explains but does not retract). **R4, R5, R6 added** | `wvdrblr5d.output` anomalies; commit `4a5d87d7`; `LOG.md` 02:45 |

**Nothing was VETOED** because r2 makes no claim that survives to be removed —
which is itself the honest verdict on the night. Two items came close: the §6.3
AMD framing (an implicit production comparison, struck) and the §4 "+16–31 %"
block (a withdrawn estimate typeset as a finding, now stamped).

#### What this report IS licensed to claim

1. A **kernel-level regression anchor**: the M15 body reproduces its banked
   MoK ratio at T=4096, 100 % fill, to within ±1 % (M4). This is a statement
   about *harness and pin integrity*, not about performance in production.
2. That the M15 chassis **fails MoK correctness at T=2048/1024** while a
   control body passes in the same tree (M5+M6) — so R6 is dead and fill
   evidence must come from M24.
3. **Directional, diagnostic** kernel-substitution behaviour at c32p against
   **patched-stock** at n=2 with a spread that swamps the mean (M3).
4. **Corpus observations** about the captured routes — dummy-call rate,
   bimodality, skew — each carrying first-64-calls capture bias and rank-call
   granularity (M8, M11).
5. **Modeled, kernel-region** projections explicitly labelled as models with
   φ assumed and their baseline named (M11, §4).
6. **Engineering delivered**: designs, an implementation behind default-0
   macros, a campaign harness, method hardening, retractions — none of which
   is a performance result, and the M24 kernel has never been compiled.

#### What this report is NOT licensed to claim — under any rewording

* That the megakernel **beats production**, native or otherwise, at any cell,
  by any margin. No such measurement exists tonight or in this project's
  history (M9).
* That the 20–30 % goal is **within reach**, **on track**, or **supported by
  evidence**. It is unmeasured against its own definition.
* That **M24 will deliver** +11–17 %, +16–31 %, +22 %, or any other figure.
  All are withdrawn models over a suspended premise, on an uncompiled kernel.
* That **0.75187 means the mega is faster** (or slower) end-to-end. It is a
  synthetic, 100 %-fill, kernel-region ratio.
* That **camp3's +2.70 %/+3.61 %** says anything about production, or is
  comparable to any vendor-published figure.
* That either **2.66×** or **1.19×** is *the* padding multiplier (R4).
* That **AMD's 16–47 % / 1.2–1.5×** describe anything measured on this node.

#### Preconditions for any future "beats production" claim

All of the following, jointly, before the words are written anywhere:

1. **Native arm proven by evidence** — image digest
   `sha256:84459732ca98…56f86`, `docker inspect` env with **no** `VLLM_PF4H_*`,
   server.log config dump, and demonstrated absence of `PF4H_INTEGRATION_PATCH_V3_M15`,
   `PF4H_COVERAGE_PATCH_V1`, `PF4H_M23_RAGGED_SEAL_V1`, `PF4H_RR_*`,
   `PF4H_B4096_GRAPH_TARGET`, `PF4H_EPLB_*_RELAX_V1`, `PF4H_M24_FILL_V1`.
2. **Native not sandbagged** — AMD's recommended config *per cell*
   (TP8+EP below 128 concurrency; DP+EP only at ≥512), AITER/MORI env present,
   matched gpu-mem-util and scheduler flags, every deviation logged. Plus the
   `stock + VLLM_MOE_SKIP_PADDING=1` control arm, since production ships it OFF.
3. **Candidate proven to have run** — `RAGGED_SEAL_RECEIPT sealed/in_bucket`
   per arm, and for M24 arms the `M24_FILL_RECEIPT` Σ`T_eff` vs Σ`n_orig`
   under-dispatch audit.
4. **Identical workload** — exact-token prompt SHA match, same cell/seeds/
   client/warmup, symmetric cooldowns (ARM_COOLDOWN=240).
5. **≥5 rotation-balanced pairs** (a multiple of the arm count), per-pair
   ratios and spread published, claim sized against ±18 % position and ±15 %
   day drift. Cross-pair ratios and single arms are never evidence.
6. **Accuracy parity resolved** — the `ordered_output_token_id_stream_sha256`
   mismatch that appears even stock-vs-stock is currently unresolved, so B1's
   deterministic cells or a bounded rel-err envelope must gate the claim
   *before* the matrix runs. Unresolved parity = no claim.
7. **Three-way decomposition published** — `m15/native` (headline),
   `m15/patched-stock`, `patched-stock/native` — so the integration effect
   cannot hide inside the kernel number.
8. **Wording matched to measurement** — cells named, regime named
   (serving fill vs 100 %-fill kernel rigs), the hybrid caveat stated and the
   losing small-batch cells published, and memory/replication costs priced in
   the same table as the speedup.
9. **This audit re-run with veto** on the actual campaign output. Tonight's
   sign-off covers r2's wording only and confers nothing on future numbers.
10. **M24 additionally**: build gate passed (resource tuple + normalised
    disassembly + `.text` sha256 identical at default), R0b inert-arm arbiter,
    arm E re-sweep, arm D heterogeneous-skew arm, and the `ZERO_PAD`
    calibration — before any φ or fill number is read, let alone quoted.

*Signed: fairness-audit subagent (Opus, `high` effort, adversarial mandate),
2026-08-18, against REPORT.md r2. Veto exercised as edits-in-file on 8 items;
no claim removed because no claim qualified.*

---

## 4. Bottleneck mechanism status

**The hypothesis.** The mega executes 4,096 padded rows while carrying
~1,539 real tokens/rank/step (37.6 % fill, 2.66× padding). Its fixed costs
(single-CTA planner, reserved service CTAs, slab rendezvous, M8 certification,
per-row work on pad rows) do not shrink with fill. Two tiers: (1) whole-step
dummy skip, (2) `n_orig`-proportional work on partially-filled steps.

**Cost model (FILL_AWARE_DESIGN.md rev 3 §A.4, post-review).**
`ρ(f) = (1 − φ) + φ·f` with φ = 0.919 derived (rounded **down** to 0.90
central; 0.935 optimistic, 0.80 pessimistic). At f = 0.376 → **ρ = 0.44
central** (0.42 / 0.50); at the TGRAIN=256 effective fill of 0.438,
ρ = 0.494. That is a **modeled** 2.0–2.4× less MoE work per sealed step —
φ is assumed, not measured (F.3 exists to measure it), and `f = 0.376` is the
premise **suspended by R4**.

**Predicted e2e, in the design's own words — and WITHDRAWN per §1 and R4,
reproduced here only to record what was withdrawn:** **+16 % to +31 % vs
rescued/patched-stock at c32p, central ≈ +22 %; UNKNOWN vs native
production.** This is a model, not a measurement; it is stated against the
**secondary** baseline; it inherits the suspended `f = 0.376`; and the hybrid
caveat applies (the mega inverts below ~1,600–1,800 tokens/rank, so cells
below break-even are expected losses). The design says plainly that M24 alone does not reliably clear
20–30 %, and promotes the κ = 1.192 step-count composition defect (the m15 arm
ran the padded B4096 graph on 249.1 of 300 steps vs stock's 205.9) to a
co-equal work item. **Tier-1 is not separately sized** in rev 3 — the earlier
"+12–17 %" figure was withdrawn because it rested on the 31 % proxy and
because §B.5 showed the receive-side null-work path does not fire on a
DP-dummy rank in serving. What survives without measurement is the
*mechanism*: a rank with `n_orig = 0` stops **sending**, so its pad rows
vanish from all eight ranks' receive, scatter, GEMM, epilogue and combine.

**What F.3 will decide.** The `NORIG_CONST` ladder at fixed capacity 4096
(N ∈ {4096, 3072, 2048, 1536, 1024}) measures φ directly, reported as an
**interval between the extreme points plus the affine-fit residuals**, never
as an intercept — the response is not affine (M8 ticket degeneracy at low T,
per-expert round-up to 32, GEMM tile quantisation, per-build codegen scatter).
Falsification thresholds, valid only after arm E re-sweeps C × flush_rows:
predicted N=1792 → ~2,830 µs, ratio ≈ 0.367; ≥ 0.55 ⇒ fixed-cost fraction
≥ 0.30, far above the ledger, re-derive §A; ≈ 0.75 ⇒ the T_eff plumbing is
inert; < 0.30 ⇒ suspect the arm computed nothing.

**G7 — T-generality resolved (M10, commit `9ce36341`, laptop-only).** Both
helper primitives (`hkp::csr_scan_block256`, `hkp::pull_src_fill`) are correct
for **every** `T ∈ [0, T_cap]` with no alignment requirement; the only
precondition is `blockDim.x == 256`, already structural via
`__launch_bounds__(256,1)`. **`T_eff = 0` is safe at all five substitution
sites**, so §F.4's N=0 ladder point is interpretable and resolves to the
**unclamped** branch (< 300 µs, ratio < 0.04); the clamped branch is struck and
§H.4's reserved right to clamp `T_eff` to a `TGRAIN` minimum is released.
Minimum safe `TGRAIN` is **1**; ship **8** (`-DK0P6_M24_TGRAIN=8`) as cosmetic
margin — ≤ 7 phantom rows out of ~1,539 (< 0.5 %, inside noise) while keeping
`T_eff·896·16` 128-byte aligned for the `ZERO_PAD` sweep. At grain 8 the
round-up penalty is ≈ 1.00, which moots the TGRAIN column and makes the raw-fill
table operative rather than the optimistic bound.
Two consequences for the gate plan: (a) **`ZERO_PAD` write volume is
anti-correlated with N** — the sweep writes `(T_cap − N)·H·2 B` (58.7 MB/layer
at N=0, zero at N=T_cap), so an affine fit of step time vs N absorbs it as a
spurious negative slope and **biases φ**; the F.3 ladder needs either a
`ZERO_PAD`-only calibration arm or the byte term held as a **fixed** regressor,
otherwise the N=0 point is an *upper bound* on fixed cost, not the fixed cost.
(b) The release fence at `KERNEL:713-727` is **correctness-load-bearing for the
CSR sentinel** specifically (divergent `T` between the scanning CTA and a
combine CTA ⇒ stale `pull_ptr[T]` ⇒ arbitrary fanout ⇒ *silent* wrong output).
G7 recommends a cheap fanout clamp to `[0, world]` at `KERNEL:1108` converting
that class from silent corruption to bounded/detected. **Not implemented
tonight — listed as next-iteration hardening** (§7).

**G0b-local — the fill distribution is BIMODAL, not centred (M11, commit
`619b6308`).** Over the 512-rank-call route corpus: **31 % dummy** calls,
**~60 % completely full**, only **~10 % genuinely partially filled**; mean fill
**0.69**; aggregate padding **1.19× pre-M24 → 1.01× post** — the "post" figure
is **arithmetic on the corpus under the assumption that M24 behaves as
designed**, not an observation of M24, which has never been compiled (the
residual 1.01× is pure 256-tile quantization). Modeled MoE-region ρ at φ=0.9: **tier-1-only 0.719
(1.39×) vs full M24 0.730 (1.37×)** — i.e. under this distribution **tier 1
captures the whole modeled win**, and tier 2's row-granular plumbing recovers
only the ~10 % partial calls, most of which the tile clamp eats. That is
**~+11–17 % e2e** at the measured 40–54 % MoE fraction — a kernel-region model,
never a serving result. It also **over-states tier 1**: a serving step is 8
coupled rank-calls, so a step-level skip requires **ALL 8 ranks dummy**, and the
corpus carries no step key to check that locally. If this shape holds, the
complexity budget belongs in tier 1 (a cheap collective planner predicate), not
tier 2.

> ### ✅ RESOLVED (r3) — the two fill numbers were never in conflict
> The banked serving receipt says **37.6 % fill / 2.66× padding**
> (~1,539 real tokens/rank/step of 4,096). The route corpus said **mean fill
> 0.69 / 1.19× aggregate padding**, with non-dummy calls near-completely full.
> Both sides have now been repaired and they reconcile.
> **The corpus was right about SHAPE; the receipt was right about RATE.**
> *Corpus repair:* the 1.19× was an artifact of an `n_real > 0` filter that
> dropped the 91 zero-fill calls (**R7**); the honest all-512 aggregate is
> **1.446×**, and non-dummy calls are ~**0.91** full — the shape conclusions
> (bimodal, tier-2 row plumbing weak, tier-1-dominant) stand and strengthen.
> *Receipt repair:* its provenance is now sourced and reproduced (**R4
> discharged**) with the in-bucket / whole-run / n=1 / `/8`-convention
> qualifiers recorded there.
> *Residual gap = the dummy-step RATE, not the fill shape.* The corpus window
> (first 64 calls) is the **ramp phase** and sees **31 %** dummy calls; the
> whole-run receipt rate is **~56 %**. Composing the two
> (0.442 × 0.9146 ≈ 0.404 vs the receipt's 0.376) closes ~86 % of the gap,
> with the residual inside the known capture bias; an independent
> all-tokens/padded-rows cross-check lands at 40.3 % vs the predicted 40.4 %.
> **Consequence for the plan:** **tier 1 now targets ~56 % of in-bucket
> steps — materially LARGER than the 31 % G0b modeled** — and tier 2 stays
> the weaker tier. The two decisive gates that remain are (1) the
> `RAGGED_SEAL_RECEIPT` **printf grep** on the node, to confirm the `/8`
> peer-summing convention, and (2) **dummy-step rank-synchrony** —
> now **RESOLVED FOR THE CAPTURE WINDOW** (M12, commit `2ba124c8`): dummy
> calls are perfectly rank-synchronous (`c[i]` in {0, 8} for all 64 indices,
> 100 % vs 5 % under independence; cross-rank `n_real` r = +0.996 at lag 0;
> all 8 workers at an identical 31.2 % dummy rate), so a tier-1 whole-step
> skip carries **1:1 with no synchrony discount**. The residual node-side
> check is narrowed to: do the whole-run ~56 % **mid-run** dummies (the
> window holds only ramp/drain blocks) behave the same — a per-step
> `n_orig` cross-rank grep. Design rule unchanged: the skip stays a
> **collective** decision; local `n_orig == 0` is a proposal only.
> Source: provenance reconstruction, held locally; conclusions summarized
> here; receipt re-verification queued on the node.

**Four corrections that changed the plan tonight:**

* **Production does NOT scale with fill.** `VLLM_MOE_SKIP_PADDING` ships OFF.
  Fill-awareness is a pure structural advantage, not a gap-closing fix — so a
  `stock + SKIP_PADDING=1` control arm is required, and the wording of any
  future claim matters (fairness risk #1).
* **The M15 chassis is not T-parametric** (M5). The cheap R6 confirmation
  experiment is dead: `K0_MAXTOK = K0_T` shrinks capacity too, so R6 was a
  **batch-size** sweep, never a fill sweep. Real fill evidence now requires
  the M24 build itself.
* **Spin-headroom risk.** Tier-1 makes dummy ranks spin *harder* on their
  peers (mode-14 L2 law) — a new exposed rendezvous on exactly the ranks the
  fix makes fast. ρ carries no skew term and the homogeneous ladder arms are
  blind to it, so **arm D (heterogeneous `NORIG_TABLE {4096,0,…,0}`) is
  mandatory**, SPIN_DBG slot 52 is monitored every arm, and POLL_BACKOFF is
  now recommended (previously discouraged).
* **Bit-identity withdrawn.** The kernel is already run-to-run
  nondeterministic (bf16 RMW combine order), so all parity gates become
  measured rel-err envelopes (G0a), the m15 arm can never SHA-match, and
  c1det only validates deterministic-numerics arms.

---

## 5. What was built (all pushed to `origin/ablations`)

| Commit | What |
|---|---|
| `d99fa55a` | **M24 rev 3 implementation** — kernel +431/−4 in `k0pf6gm_device_tile_m15.hip` behind **9 default-0 macros**: descriptor slot 71 fill vector `{magic, gen, n_orig[world]}` device-sourced from the DP all-reduce, `T_eff` at 5 row-count substitution sites, gen staleness **fails OPEN to 4096**, tier-1a dummy publish (`rows_done` + 8 `chunk_ready` cells at fill 0 ⇒ peers skip the segment), tier-1b NULLWORK keyed on `chunk_ready` ground truth, ZERO_PAD determinism. Host half: `m15_contracts.py` G12, `m24_fill.py` encoding, MoK patcher `m24_mok_patch.py` + tests, `m24_compile_check.sh`, `m24_build_gate.sh` |
| `e1bcfc4d` | M24 fix pass + `m24_build_gate.txt` recording **NOT RUN** with exact rerun commands |
| `1329aebe` | FILL_AWARE_DESIGN.md rev 2 (5 blocking + 16 major review findings resolved; headline revised **down**) — rev 3 correction boxes + Appendix R landed with `d99fa55a` |
| `a776ac75` | **campaign_v5.1** — wrapper v5 with four native arms (`native_default`, `native_tuned_tp` = AMD's TP8+EP for ≤128 conc, `native_tuned_dp` for ≥512, `native_mirror` renamed and excluded from `PRODUCTION_ARMS`), client v3 (open-loop Poisson + ITL), c1det/c8det accuracy cells, rotation-balanced pairs, schema-refusing analyzer, `ARMS.md` with per-flag citations, `NIGHT_SCHEDULE.md`, `REVIEW_NOTES.md` |
| `b163f93e` | SERVING_BENCHMARK_METHODOLOGY.md **rev 3** — operator protocol folded in: ≥5-pair hard floor, native-tuned second baseline + symmetry rule, accuracy gate with the SHA-nondeterminism open problem, open-loop cells, deployment-policy row, fairness items 2b/4b |
| `00501cf1` | DECOMP_RUNBOOK amendments a–f: R0a re-anchor, `PADMAX = 64·T + 992` (the "263136 stays generous" line was **wrong**), R6 scope banner, archive pin-trap warning, ledger `servicedrain` arithmetic caveat, `K0_PF6GM_G=3` inert note |
| `ed08279a` (amd-master) | one-line `_m20 = None` hoist — unblocks **every** non-replication MoK arm at HEAD |
| `824a3e45`, `d3a21f33` | cleanup: pre-M23 void serving numbers retired in both repos |
| `0300ebb7`, `4a5d87d7` | BENCHMARK_PROTOCOL.md, CORPUS_FINDINGS.md, LOG |

**M24 review record:** 7 blocking + 9 major across 2 adversarial lenses — all
fixed except one **rebutted with evidence** (the NULLWORK M2-timeout
CTA-uniformity finding; Appendix R3.2, belt-and-braces `s_total = 0xFFFFFFFF`
added anyway). If a reviewer rejects the rebuttal it reopens as blocking.

**What M24 still needs (`ready_for_mok = false`):** build gate + both-config
hipcc (never ran — node); G9 parts 1–3 in the MoK harness
(`K0_MOK_COMPARE_ROWS` rel-err gate over `[0,N)`; `_poison_count`/`_poison_rows`
restricted to `[0,N)` with `_poison_out` left whole-buffer; `_st_idx → N−1`
plus the two-assertion selftest) — blocking for fill arms A–D, not for
R0a/R0b; G0a run-to-run envelope; G7 TGRAIN/T-generality; and the entire
**serving-side plumbing G13a–G15** (m15_runtime forensics, `write_fill_vector`
as a capturable one-block device kernel, `PF4H_M24_FILL_V1` apply patch,
`M24_FILL_RECEIPT` with the Σ T_eff vs Σ n_orig under-dispatch audit).
**M24 cannot reach a server this iteration.**

Verified locally instead of compiled: a real C preprocessor over all 12 macro
configurations (8 must `#error` — all fired), the descriptor cascade in 6, and
two green CPU suites (host↔device encode parity; patcher apply / idempotency /
tamper-refusal / inertness). **None of it says anything about codegen.**

---

## 6. Discoveries and incidents

1. **Legacy queue harvested then killed, zero loss.** camp3 `RC=0` and the
   route capture `RC=0` were complete; camp1 had been killed (`RC=143`), camp2
   never ran. Nothing was mid-measurement and the GPUs were idle, so the three
   waiters (`camp_grid`, `camp4_rr`, `camp5_native`) were killed at 08:16 UTC —
   **8 minutes before `camp4_rr` would have seized the GPUs**. A passive
   `t1full3` log-grepper was inspected and left alive.
2. **The v4 "native" arm was never native.** It inherited *our* serve flags
   (gpu-mem-util 0.70, MNBT 4096, max-num-seqs 128, opt-level 2) — a
   config-mirrored diagnostic. Found during campaign prereqs, renamed
   `native_mirror`, excluded from `PRODUCTION_ARMS`, and replaced by genuine
   `native_default` + `native_tuned_*` arms. Had this not been caught, v5
   would have shipped the project's named past sin a second time.
3. **AMD's documented bar is TP8+EP below 128 concurrency** — i.e. for
   *four of our five cells*. DP8+EP is AMD's recommendation only at ≥512
   (they claim +16–47 % there), and `ROCM_AITER_MLA` is claimed 1.2–1.5× over
   TRITON_MLA. Our TP1/DP8/EP8
   integration is the vendor-recommended config for exactly one cell (c512p).
   The headline must therefore be m15 vs **per-cell best native**.
   **⚠ Audit label on the AMD figures: these are *vendor marketing claims*,
   not measurements on our setup.** They come from AMD's ROCm optimization
   guide and the 2026-02-27 vLLM/AMD blog, are stated for **different**
   hardware/workload/version points (the 1.2–1.5× band resolves to 1.52× at
   conc 64 and 1.35× at conc 128 on **ISL 10K / OSL 1K**, MI355X, unspecified
   vLLM build), and **none has been reproduced on vLLM 0.25.1 on this node**.
   AMD publishes **no absolute vLLM tok/s for DeepSeek-R1 on MI350X/MI355X**,
   and R1 is absent from AMD's MLPerf v6.0, so there is no external anchor.
   The right use of these numbers is *"our native arm may be misconfigured by
   this much, so it must be tuned"* — never a head-to-head against anything of
   ours. In particular, the earlier framing that these ranges "swallow our
   banked +2.7–5.2 %" is **struck**: it compared a retracted n=2
   patched-stock diagnostic (R3) against unreproduced vendor claims for a
   different configuration, which is a production comparison assembled out of
   two things that are each disqualified from making one.
4. **The ATOM ceiling.** AMD's fastest published R1 path on MI350/MI355 is the
   ATOM engine (`rocm/atom-dev`) with MTP speculative decoding — a different
   image, so it cannot be an arm. "Beats stock vLLM" is strictly weaker than
   "beats AMD's fastest published stack". No absolute AMD tok/s anchor exists
   for R1 on MI350X/MI355X vLLM (R1 is absent from AMD's MLPerf v6.0), so the
   audit has only internal evidence to lean on.
5. **Archive pin trap realized** (retraction R1, §2).
6. **Harness regression at amd-master HEAD** — `NameError: _m20` blocked every
   non-replication MoK arm; worked around on the node via a scratch tree with
   the file restored to `a903e95b` (`~/amd-master` left git-clean), then fixed
   properly at `ed08279a`.
7. **Runbook defect** — `K0_PADMAX = 263136` is not "generous"; the k0_n2
   capacity scales as 64·T + 8216, so PADMAX must be **64·T + 992**.
8. **Ledger arithmetic bug** — `[MPS TS DELTA] servicedrain` prints
   −169,098,926,698,173 because the DRAIN stamp is never written in this
   config while M6_DONE is an absolute device tick. plan/M6/M7/combine are
   unaffected and self-consistent to 0.1 %.
9. **Node outage.** Reachable and fully profiled at 08:10 UTC (8× MI350X at
   0 %, 700 G disk, 2.9 T RAM free, image digest `sha256:84459732ca98…56f86`
   captured). From **~09:00 UTC** onward `10.0.0.228:2425` is TCP-**open** but
   sshd answers `kex_exchange_identification: read: Connection reset by peer`
   on 30+ isolated attempts over ~70 min (not a MaxStartups artifact of our
   own polling); `:22` closed; `10.5.95.87` unroutable. A 60 s watcher runs.
   **Stranded on the node:** a completed same-pin tgen T-sweep (T ∈ {4096,
   2048, 1024} × 5 runs + ledger) at
   `~/nightshift_r6/tgen_t*_0818T0845/summary.json` — a 2-min retrieval giving
   a complete same-pin r(T) curve for the tgen MPS body.
10. **Anomaly still open:** `apply.py` is absent from the documented
    `~/eplb_campaign/`; copies exist only under
    `~/pf4h_vllm_20260729/*/shim/pf4h_integration/`. Confirm the patch-chain
    head before any integration build.
11. **Security note — provenance file flagged and removed; history decision
    pending.** A subagent committed
    `distributed-kernels/fused_moe/overnight/aug18-prefill/m24/FILL_PROVENANCE.md`
    to this public repo in commit **`1629fa41`**. It quoted **prior-session
    transcript records verbatim**, was flagged by the security layer, and was
    **removed from the repo tip in `24b441a8`**. The git **history still
    contains `1629fa41`**, so the operator must decide between a **history
    rewrite (force-push)** and **accepting it as-is** — the content is the
    project's own benchmark analysis and contains **no credentials and no
    personal data**, so acceptance is defensible; only the transcript-quoting
    convention is at issue. The sanitized conclusions (which is all this
    report relies on) live in the session scratchpad file
    `FILL_PROVENANCE_SANITIZED_LOCAL.md`: provenance reconstruction, held
    locally; conclusions summarized here; receipt re-verification queued on
    the node.

---

## 7. Ranked next actions

**On node return, in this exact order:**

1. **Close the two remaining fill gates (§4) — cheap, and they price M24.**
   The contradiction is RESOLVED (R4 discharged, R7 withdraws 1.19×), leaving
   two decisive checks:
   **(1a) Receipt printf grep** — grep a banked `server.log` for the exact
   `RAGGED_SEAL_RECEIPT` printf format and confirm the **`/8` peer-summing
   convention** behind `Σ(in_bucket_sum_orig)/Σ(in_bucket)/8`. This is the one
   unverified assumption under the 37.6 % figure, and it is the same pass as
   the separator-format check in item 3 — do them together.
   **(1b) Dummy-step rank-synchrony — RESOLVED FOR THE CAPTURE WINDOW**
   (M12, `2ba124c8`): dummy calls are perfectly rank-synchronous — `c[i]` in
   {0, 8} for all 64 indices (100 % vs 5 % under independence), cross-rank
   `n_real` r = +0.996 at lag 0, all 8 workers at an identical 31.2 % dummy
   rate — so the tier-1 whole-step skip carries **1:1 with no synchrony
   discount**. What remains on the node is narrower: the window contains only
   ramp-in/drain-out dummy blocks, so confirm the whole-run ~56 % **mid-run**
   dummy steps behave the same, via a per-step `n_orig` cross-rank grep of the
   receipts. Design rule unchanged: the skip is a **collective** decision;
   local `n_orig == 0` is a proposal only, never the commitment.
2. **Retrieve the stranded tgen sweep** (`~/nightshift_r6/tgen_t*_0818T0845/
   summary.json`, ~2 min). Free data, closes the r(T) question for the tgen body.
3. **Deploy campaign_v5** (staging commands in `tasks/wp5jobxxv.output`), and
   **grep a banked server.log for the exact `RAGGED_SEAL_RECEIPT` printf
   format** — the analyzer's `sealed=`/`in_bucket=` parse was never verified
   against the node, and a separator mismatch VOIDs every arm (fail-closed).
   Also confirm the `apply.py` head and the QSL pickle path.
4. **M24 CPU-side gates, in parallel with GPU work** (hipcc is CPU-only):
   `m24_compile_check.sh` (all 12 configurations incl. the 8 that must
   `#error`; the fill build must show `ScratchSize 0`), then
   `PRE_REF=a776ac75 m24_build_gate.sh` — resource tuple **and** normalised
   disassembly **and** `.text` sha256 all identical, or nothing below runs.
5. **B0 — calibration.** `{m15, native_tuned_tp}` × c32p × 1 pair,
   `ALLOW_UNBALANCED_PAIRS=1` (analyzer stamps the root "no claim"),
   ~0.66–0.76 h. Yields startup time S per arm family, each family's ceiling,
   a live seal-coverage check, and the first exercise of AMD's tuned flags on
   0.25.1. **Decision rules:** S ≤ 450 s → run B with o90p co-resident
   (~8.3 h); 450 < S ≤ 650 s → bare B, c32p only (~8.5 h); S > 650 s → cut to
   `{m15, native_tuned_tp}` at 6 pairs (~4.6 h) — cut the control arm, never a
   pair, never to a pair count that is not a multiple of the arm count.
   `native_tuned_tp` < 8k tok/s at c32p → re-price B2 first. A rejected
   startup flag → drop only that flag and record the deviation; two or more →
   fall back to `native_default` as the baseline and say so.
6. **B1 — accuracy gate BEFORE the matrix.** `{m15, stock, native_tuned_tp}`,
   `--pairs 0 --accuracy-pass`, cells c1det + c8det. `stock` is mandatory: it
   is m15's only numerics-class peer. A within-class FAIL voids the night's claim.
7. **M24 MoK gates** (~1–1.5 h GPU, after B1): P2 harness patches (M24 host
   plumbing is done and tested; **G9 parts 1–3 are not**, and block arms A–D);
   P3 G0a envelope; **R0a** (must reproduce 0.7556 ± 1 % with `K0_M24_FILL`
   unset — doubles as the zero-change ratchet); **R0b**, the inert-arm arbiter
   (`FILL=1, ZERO_PAD=1, NORIG_CONST=4096`; also requires `pperr == 0` and the
   cross-pin resource-tuple equality check); arm E (C × flush_rows re-sweep)
   **before** reading any ladder number; then the F.3 ladder and mandatory
   arm D. Stop rules + pin discipline: `wff3mewsd.output .final.mok_sequence`.
   **Build every gate arm with `-DK0P6_M24_TGRAIN=8` per G7** (§4) — 256 is
   now known to be unnecessary de-risking and it distorts the effective-fill
   arithmetic.
   **F.3 ladder — `ZERO_PAD` calibration is mandatory before φ is read.** The
   sweep's `(T_cap − N)·H·2 B` write volume is anti-correlated with N (G7 §6.2),
   so either add a `ZERO_PAD`-only calibration arm (`FILL=1, NULLWORK=1,
   NORIG_CONST=0`, time the sweep alone) or fit with that byte term as a
   **fixed** regressor. Without it the N=0 point is an upper bound on fixed
   cost, not the fixed cost, and arm B's φ interval is uninterpretable.
8. **B2 — the headline matrix.** `{m15, stock, native_tuned_tp}` × c32p ×
   **6 pairs** (rotation-balanced; 5 is not a multiple of 3 and is refused).
   ~5.8–6.7 h; may run past morning. Ratios are valid per completed pair.
9. **Fairness-audit subagent with veto** on the campaign's numbers, before
   anything is written down as a claim.

**Following iteration (not tonight's window):**

* M24 serving plumbing **G13a–G15**, then an M24-enabled m15 serving arm.
* F.3-informed tuning: fill-aware `C` (it is runtime config
  `reserved_comm_ctas`, not compile-time 28 — no recompile needed); TGRAIN is
  settled by G7 at 8.
* **Fanout clamp hardening (G7 §4/§6.3, NOT implemented tonight):** clamp
  `fanout` to `[0, world]` at `KERNEL:1108` (or raise a distinct `pperr` bit),
  converting the stale-CSR-sentinel failure class from **silent wrong output**
  to bounded/detected; one `v_med3`-class op per token, cost to be measured not
  asserted. Also record the `KERNEL:713-727` release fence as a named build-gate
  assertion, since it is correctness-load-bearing for the sentinel.
* **RR placement arm** (`rr_patch.py`, already implemented and verified) —
  attacks the **corpus-measured** 2.61× real-row skew directly (M8's capture
  bias applies; it is not a whole-run figure).
* Open-loop depth (o75p/o50p) and the **c512p** cell, where DP8+EP is finally
  the vendor-recommended config and our integration is on its home ground.
* The **κ = 1.192 composition defect** — co-equal with M24 for the 20–30 % goal.

---

## 8. Session ledger

**Subagent workflows** (all Opus 5; `high` effort for implementation and every
adversarial verification, `medium` for recon/analysis):

| Run id | Workflow | Outcome |
|---|---|---|
| `wf_6ad01aab-a7e` | recon-harvest (queue + node state) | `tasks/wuh4rxbnz.output` — camp3 harvested, kill recommendation, 4 anomalies |
| `wf_59cdde7f-f54` → `w7ovje25v` | fill-aware design (architect + 3 adversarial lenses) | 7 blocking / 24 major / 12 minor; 5 distinct blockers fixed, none rebutted; `go=true` → `1329aebe` |
| `wf_c23e9d34-05c` | cleanup-resume (both repos) | `824a3e45`, `d3a21f33` |
| `wf_aa479070-846` | r6-fill-evidence (R0a arbiter + T-sweep) | `tasks/wvdrblr5d.output` — R0a PASS 0.75187; R6 blocked-by-correctness; 3 retractions |
| `wf_2a9041f7-7ae` | campaign prereqs (AMD config research ∥ harness recon) | `tasks/w3clrpi59.output` — native-arm fairness exposure found; TP8+EP doctrine; determinism diagnosis |
| `wf_c23ee898-84d` | campaign-v5 build (implement → 2-lens verify → fix) | `tasks/wp5jobxxv.output` — 12 blocking + 14 major fixed → `a776ac75` |
| `wf_6bbc1bff-8a2` | m24-implement (implement → 2-lens diff review → fix) | `tasks/wff3mewsd.output` — `d99fa55a`, `e1bcfc4d`; `ready_for_mok=false` |
| `wf_98e0d1c3` | runbook amendments + `_m20` harness fix | `00501cf1`, `ed08279a` |
| `bkpkhvmfu` | node-recovery watcher (60 s poll) | running |

**Commits pushed to `origin/ablations`:** `0300ebb7`, `b163f93e`, `4a5d87d7`,
`824a3e45`, `1329aebe`, `00501cf1`, `a776ac75`, `d99fa55a`, `e1bcfc4d`
(+ `ed08279a`, `d3a21f33` on `amd-master`).

**Not done tonight, and why:** no e2e campaign (node), no build gate or any
hipcc compile of M24 (node), no G0b `n_orig` histogram (free and laptop-only — it should have been
run; it is the cheapest remaining item and replaces the 31 % proxy with a
measurement).

---

*One source conflict, resolved: `LOG.md`'s 02:45 entry records the M24
estimate as "+25–40 % e2e, central +31 %; tier-1 alone +12–17 %". Those are
the design's **first-draft** figures. `FILL_AWARE_DESIGN.md` rev 2/3 §A.4
explicitly revises them **down** to +16–31 % vs rescued/patched-stock,
central ≈ +22 %, UNKNOWN vs native, and §A.4.3 **withdraws** the tier-1 split
entirely pending G0b. This report quotes the design doc.*

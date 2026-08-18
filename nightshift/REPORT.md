# Nightshift REPORT — 2026-08-18 overnight (draft r1, written 04:50–05:10 PDT)

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
2. **Campaign v5.1** — the first campaign this project has ever had with a
   genuinely untouched production arm, plus AMD's own recommended TP8+EP
   config, an open-loop client, a deterministic accuracy cell,
   rotation-balanced ≥5-pair statistics and a schema-safe analyzer. Built,
   adversarially reviewed (12 blocking + 14 major fixed), committed,
   **not yet deployed**.
3. **Method hardening** — SERVING_BENCHMARK_METHODOLOGY.md rev 3 (binding
   protocol folded in; camp3 demoted to directional-only), DECOMP_RUNBOOK
   amendments, a one-line harness fix unblocking every non-replication MoK
   arm, and retirement of the last pre-M23 serving numbers in both repos.
4. **Two measurements and three retractions** (§2) — the R0a ratchet holds at
   0.75187, and the M15 chassis was proven **not T-parametric**, which voids
   the cheap R6 fill experiment and re-routes all fill evidence through M24.

State of the 20–30 % goal, stated honestly: the fill mechanism's own
post-review cost model predicts **+16 % to +31 % vs rescued/patched-stock at
c32p, central ≈ +22 %, and UNKNOWN vs native production**. The design
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
| M1 | camp3 pair_01: m15 19,277.36 vs stock 15,611.17 input tok/s → **+23.49 %** | SERVING | **patched-stock** (not native) | c32p: conc 32, 1024 prompts, ISL 4096, OSL 8; prompt SHA `b6271f04…` identical across arms | `RAGGED_SEAL_RECEIPT` rank1 steps=300 in_bucket=259 sealed=251 → **96.9 %** sealed/in_bucket (83.7 % sealed/steps) | single pair, m15-first — **not evidence alone** |
| M2 | camp3 pair_02: m15 17,899.96 vs stock 20,588.66 → **−13.06 %** | SERVING | patched-stock | as M1, stock-first | same receipt family | single pair — **not evidence alone** |
| M3 | camp3 order-balanced n=2: **mean +2.70 %, geomean +3.61 %**, spread +23.5 % / −13.1 % | SERVING | patched-stock | as M1 | 96.9 % | **DIRECTIONAL ONLY.** n=2 is below the binding ≥5-pair floor; the ±18 % position effect fully swamps a +2.7 % mean |
| M4 | **R0a re-anchor: ratio 0.75187** (median of 5 order-balanced; runs 0.75008 / 0.75054 / 0.75321 / 0.75187 / 0.75292). M15 p50 5,824.84 µs, production-ops p50 7,740.80 µs | KERNEL | **production-ops-replay inside the MoK harness** | MoK synthetic prefill, T=4096, **100 % fill**, C=28,g=353,mode=12,flush_rows=16, PADMAX=263136, warmup 500 / timed 100, route replay NOT deployed | MoK gate pass all 5 runs, rel_err 0.008295; pin `~/DHK-m15` sha `935f555e…` byte-identical to the banked 0.7556 arm | **PASS** vs banked 0.7556 (−0.49 %, inside ±1 %). Harness + anchor sound, ratchet unconfounded. **Not an e2e number** |
| M5 | R6 T-sweep on the M15 body at T=2048 / T=1024: **no timing** — `[MOK GATE] mps_mega pass=False`, relative **0.874205 / 0.838890**, while `production` passes at 0.005820 / 0.005745 in the same runs | KERNEL | n/a (blocked pre-timing) | as M4 with K0_T=2048/1024, PADMAX = 64·T+992 | gate FAIL is itself the receipt | **BLOCKED BY CORRECTNESS.** G11 answered negatively: the M15 chassis certifies only at T=4096 |
| M6 | tgen-body control at T=2048: **ratio 0.8959**, gate pass (max_abs 0.042969, rel 0.008288) | KERNEL | production-ops-replay | identical tree/env/MPSCFG/PADMAX as M5, pin `~/DHK-tgen` sha `20b8c6bb` | gate pass | valid **for the tgen body only**; proves M5 is body-specific, not environmental |
| M7 | T=4096 phase ledger: plan 372.29 µs, M6 2,454.95 µs, M7 2,201.61 µs, combine 213.93 µs, m2_to_end 5,242.78 µs; `[MPS SPIN] chunk_poll fail_max=0` | KERNEL | n/a | as M4, timestamps=1, 1 run, rank-0-only final-soak-epoch semantics | — | diagnostic; shape matches banked exp_33. fail_max=0 ⇒ no transport degradation at 100 % fill |
| M8 | **31 % of sealed calls are 100 % dummy work**; real-row max-rank-load p50 **2.61×** / p90 3.35×; consecutive-row expert overlap 57.0 % vs 55.7 % shuffled (**1.3 pt**) | CORPUS | n/a | 512 calls × [4096,8] across 8 workers, routecap1 capture (m15+M23 server, c32p), first 64 B4096 calls/worker | matches whole-run receipts' `uniform_rescued` ≈ 34 % | the 31 % is a **routing-homogeneity proxy, not an `n_orig` measurement** (early-run capture bias; direct measurement deferred to G0b) |
| M9 | **NO e2e ratio vs native production exists — tonight or ever.** camp5_native never ran; v4's "native" arm was config-mirrored (§6) | — | — | — | — | **stated, not measured** |

### Retractions issued tonight

* **R1.** Banked `t2048_m15_bal_0814T1937` (0.8912) and `t1024_m15_bal_0814T1937`
  (1.1051) carry pf6mps source sha `20b8c6bb` = **DHK-tgen**, not the M15
  chassis sha `935f555e`. They are **not M15 data** and must never be quoted
  as an M15 fill result. This is the pin trap, realized inside our own archive.
* **R2.** The banked camp3 seal coverage of "99 %" is optimistic; the receipt
  reads **96.9 %** sealed/in_bucket.
* **R3.** The banked camp3 headline "+2.7–5.2 %" is more precisely
  **+2.70 % mean / +3.61 % geomean at n=2** and is directional only.

---

## 3. FAIRNESS_AUDIT

**Global verdict for this report: NO "beats production" claim is made, and
none is permitted tonight. There is no headline to audit.** The mandatory
fairness-audit subagent was therefore *not* launched — correctly, since it
audits claims and no claim exists. It **must** run, with veto, before any
campaign_v5 number is quoted anywhere.

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
ρ = 0.494. That is **2.0–2.4× less MoE work per sealed step**.

**Predicted e2e, in the design's own words:** **+16 % to +31 % vs
rescued/patched-stock at c32p, central ≈ +22 %; UNKNOWN vs native
production.** The design says plainly that M24 alone does not reliably clear
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
   TRITON_MLA. Both ranges **swallow our banked +2.7–5.2 %**. Our TP1/DP8/EP8
   integration is the vendor-recommended config for exactly one cell (c512p).
   The headline must therefore be m15 vs **per-cell best native**.
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

---

## 7. Ranked next actions

**On node return, in this exact order:**

1. **Retrieve the stranded tgen sweep** (`~/nightshift_r6/tgen_t*_0818T0845/
   summary.json`, ~2 min). Free data, closes the r(T) question for the tgen body.
2. **Deploy campaign_v5** (staging commands in `tasks/wp5jobxxv.output`), and
   **grep a banked server.log for the exact `RAGGED_SEAL_RECEIPT` printf
   format** — the analyzer's `sealed=`/`in_bucket=` parse was never verified
   against the node, and a separator mismatch VOIDs every arm (fail-closed).
   Also confirm the `apply.py` head and the QSL pickle path.
3. **M24 CPU-side gates, in parallel with GPU work** (hipcc is CPU-only):
   `m24_compile_check.sh` (all 12 configurations incl. the 8 that must
   `#error`; the fill build must show `ScratchSize 0`), then
   `PRE_REF=a776ac75 m24_build_gate.sh` — resource tuple **and** normalised
   disassembly **and** `.text` sha256 all identical, or nothing below runs.
4. **B0 — calibration.** `{m15, native_tuned_tp}` × c32p × 1 pair,
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
5. **B1 — accuracy gate BEFORE the matrix.** `{m15, stock, native_tuned_tp}`,
   `--pairs 0 --accuracy-pass`, cells c1det + c8det. `stock` is mandatory: it
   is m15's only numerics-class peer. A within-class FAIL voids the night's claim.
6. **M24 MoK gates** (~1–1.5 h GPU, after B1): P2 harness patches (M24 host
   plumbing is done and tested; **G9 parts 1–3 are not**, and block arms A–D);
   P3 G0a envelope; **R0a** (must reproduce 0.7556 ± 1 % with `K0_M24_FILL`
   unset — doubles as the zero-change ratchet); **R0b**, the inert-arm arbiter
   (`FILL=1, ZERO_PAD=1, NORIG_CONST=4096`; also requires `pperr == 0` and the
   cross-pin resource-tuple equality check); arm E (C × flush_rows re-sweep)
   **before** reading any ladder number; then the F.3 ladder and mandatory
   arm D. Stop rules + pin discipline: `wff3mewsd.output .final.mok_sequence`.
7. **B2 — the headline matrix.** `{m15, stock, native_tuned_tp}` × c32p ×
   **6 pairs** (rotation-balanced; 5 is not a multiple of 3 and is refused).
   ~5.8–6.7 h; may run past morning. Ratios are valid per completed pair.
8. **Fairness-audit subagent with veto** on the campaign's numbers, before
   anything is written down as a claim.

**Following iteration (not tonight's window):**

* M24 serving plumbing **G13a–G15**, then an M24-enabled m15 serving arm.
* F.3-informed tuning: fill-aware `C` (it is runtime config
  `reserved_comm_ctas`, not compile-time 28 — no recompile needed), TGRAIN
  after G7.
* **RR placement arm** (`rr_patch.py`, already implemented and verified) —
  attacks the measured 2.61× real-row skew directly.
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
hipcc compile of M24 (node), no fairness-audit subagent (no claim exists to
audit), no G0b `n_orig` histogram (free and laptop-only — it should have been
run; it is the cheapest remaining item and replaces the 31 % proxy with a
measurement).

---

*One source conflict, resolved: `LOG.md`'s 02:45 entry records the M24
estimate as "+25–40 % e2e, central +31 %; tier-1 alone +12–17 %". Those are
the design's **first-draft** figures. `FILL_AWARE_DESIGN.md` rev 2/3 §A.4
explicitly revises them **down** to +16–31 % vs rescued/patched-stock,
central ≈ +22 %, UNKNOWN vs native, and §A.4.3 **withdraws** the tier-1 split
entirely pending G0b. This report quotes the design doc.*

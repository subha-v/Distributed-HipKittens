# WORKLOAD_EVIDENCE — the empirical distributions the workload state vector must span

**Written:** 2026-08-18, laptop, no node. **Scope:** everything this project has *measured*
about real serving workloads, with provenance. **Consumer:** the ablation-campaign master doc
(`../BRIEF.md` §1 "Workload state vector W").

This file answers one question: **when we write down a workload state vector W, which
coordinates do we already have empirical distributions for, at what points, from which
instrument — and which coordinates are pure hypothesis?** It does not design experiments; it
bounds what the design is allowed to assume.

---

## 0. How to read this file

### 0.1 Evidence classes (applied to every number below)

| class | meaning | may be quoted as |
|---|---|---|
| **SERVING** | full vLLM server, real client, in-repo or receipt-backed artifact | end-to-end behaviour, with its full workload tuple |
| **RECEIPT** | per-step counters emitted by the serving stack (`RAGGED_SEAL_RECEIPT`, `M15_COVERAGE`) | workload composition, never performance |
| **KERNEL** | MoK synthetic rig (`e004pf_k0pf_ab.py` harness), never a server | in-region ratios only, never e2e |
| **CORPUS** | offline analysis of captured router outputs | routing/fill *shape*, with capture bias attached |
| **MODEL** | arithmetic over a measured distribution | explicitly labelled model, never a measurement |
| **MEMORY** | operator session memory, no in-repo receipt | a lead to verify, never evidence |
| **DERIVED** | arithmetic performed in *this* file over cited in-repo artifacts | shown with its inputs |

### 0.2 Strings that are retracted and must not reappear

From `nightshift/REPORT.md:97-143` (retractions R1–R7), which is the binding record:

* **"1.19× corpus padding multiplier"** — R7, arithmetic artifact (`n_real > 0` filter dropped
  91 zero-fill calls). Honest all-512 aggregate is **1.446×**.
* **"31 % of sealed *steps* are dummy"** — R5, the measurement is over **rank-calls**, and it is
  a **ramp-phase** rate, not the whole-run rate.
* **"+2.7–5.2 %"** (camp3) — R3, retired outright; the honest form is +2.70 % mean / +3.61 %
  geomean at n=2, directional only, vs **patched-stock**.
* **Archived `t2048_m15_*` / `t1024_m15_*` MoK ratios as "M15 fill/T results"** — R1, pin trap
  (they carry sha `20b8c6bb` = DHK-tgen, not M15's `935f555e`). See §W6 for what the T-sweep
  legitimately is.
* **"m15 beats production"** in any wording — no such measurement existed at the time
  `REPORT.md` was signed (`REPORT.md:249-251`). §W8.2 records the two c512p pairs that landed
  *after* it and states precisely what they do and do not license.

### 0.3 The single most important structural fact

Every serving number in this project comes from **two cells** — `c32p` (conc 32, ISL 4096,
OSL 8) and `c512p` (conc 512, ISL 4096, OSL 8) — plus one historical `c32p` pair. **Nine of the
eleven defined cells have never been run.** The workload state vector is therefore, today,
sampled at **two points on a ~8-dimensional space**, both of them ISL 4096 / OSL 8 /
closed-loop / prefix-cache-off (or on, in the older pair) / single node / no EPLB. Every "law"
below that spans more than those two points comes from a kernel rig, a corpus, or a model.

---

## 1. The complete measured operating-point inventory

| point | class | cell / config | what it measured | artifact |
|---|---|---|---|---|
| **m23pair1 / p1_stock** | SERVING+RECEIPT | c32p, conc 32, n=1024, ISL 4096, OSL 8, prefix cache ON | 20,917.95 input tok/s, 200.512 s wall; fill 43.29 % | `aug18-prefill/BOTTLENECK_SIGNALS.md:44-62` |
| **m23pair2 / p2_m15** | SERVING+RECEIPT | as above, m15 arm | 19,277.36 input tok/s, 217.577 s wall; fill 37.57 %; seal 99.15 % | `BOTTLENECK_SIGNALS.md:44-62`, `DECOMP_RUNBOOK.md:16-18` |
| **camp3 pairs 1–2** | SERVING | c32p, order-balanced n=2, m15 vs **patched-stock** | +23.49 % / −13.06 % → mean +2.70 % (directional only) | `REPORT.md:84-86` |
| **B0 (b0v5) pair 1** | SERVING | c32p, prefix cache OFF, n=1 unbalanced | m15 20,371.6 vs native_tuned_tp (AMD TP8+EP) 25,844.0 → **−21.17 %** | `distributed-kernels/tp8_mega/results/serving/b0v5_pair_01_*.json`; `nightshift/LOG.md:306-315` |
| **B3cal (b3cal) pair** | SERVING | c512p, n=1, m15 first | m15 42,788.0 vs native_tuned_dp 39,474.0 → **+8.40 %** | `.../serving/b3cal_pair_01_*.json`; `LOG.md:342-355` |
| **B3rev2 pair** | SERVING | c512p, n=1, **native first** (reverse order) | native_tuned_dp 40,183.1 vs m15 42,835.7 → **+6.60 %** | `.../serving/b3rev2_pair_01_*.json` — **banked after REPORT.md was signed; not yet written up anywhere** |
| **route corpus (routecap1)** | CORPUS | 8 workers × 64 B4096 calls = 512 rank-calls, c32p, m15+M23 server | fill shape, dummy rate, per-call skew, run-correlation | `nightshift/CORPUS_FINDINGS.md`, `m24/G0B_LOCAL_FILL_HISTOGRAM.md`, `m24/DUMMY_SYNCHRONY.md` |
| **stock0814 aggregate hist** | CORPUS | 58 layers, 2.332×10⁹ routed slots, stock arm | expert popularity + owner-load skew | `aug18-prefill/rr_placement/RR_PLACEMENT_NOTES.md:208-236` |
| **m18diag per-call** | CORPUS | n = 90,050 router calls | per-call max-rank-load distribution | `aug14/M18_REPLICATION_RESULTS.md:118-122` |
| **R0a anchor** | KERNEL | MoK, T=4096, **100 % fill**, C=28 g=353 mode=12 | ratio 0.75187 (median of 5) | `REPORT.md:87` |
| **skew replay ladder** | KERNEL | MoK, T=4096, histogram routing | 0.7556 balanced → 0.8467 aggregate → 0.8559 worst-layer | `aug14/M18_REPLICATION_RESULTS.md:24-31` |
| **exp_04_tgen T-sweep** | KERNEL | MoK, T ∈ {4096, 2048, 1024}, 5-rotation campaigns, **tgen body** | 0.7557 → 0.8845 → 1.0940 | `git show 43291977:.../aug13/exp_04_tgen/result.md:226-300` |

---

## 2. Per-property evidence

### W1 — Fill φ (real rows ÷ rows the kernel executes)

**Definition as measured:** φ = Σ`in_bucket_sum_orig` ÷ (Σ`in_bucket` × 8 × 4096), i.e. real
pre-padding tokens per rank per **in-bucket** step, over the 4,096-row execution envelope.

| operating point | φ | padding multiplier 1/φ | source |
|---|---|---|---|
| **c32p, m15 arm** (whole run, 300 receipt steps) | **0.3757** (1,539 of 4,096) | **2.66×** | `DECOMP_RUNBOOK.md:32-37`; `BOTTLENECK_SIGNALS.md:266-268` |
| **c32p, stock arm** (independent cross-check) | **0.4329** (1,773.1) | **2.31×** | `DECOMP_RUNBOOK.md:38`; `BOTTLENECK_SIGNALS.md:266` |
| **c512p, m15 arm** | **≈0.985** (≈4,035 of 4,096) | ≈1.02× | `LOG.md:345-352`; `DRAFT8UPDATED_SPEAKER_NOTES.md:53-55` |
| corpus rank-call aggregate (c32p, ramp window) | 0.6917 mean | **1.446×** aggregate | `G0B_LOCAL_FILL_HISTOGRAM.md` §2 + correction box |
| corpus, **non-dummy calls only** | 0.9146 mean, p10 0.5265 | 1.093× | same, §2 |

Provenance of the headline 37.6 %: reconstructed from the **M23PAIR2 P1 m15 c32p
`RAGGED_SEAL_RECEIPT` lines**, reproduces to 3 s.f., ranks 0 and 7 agree to 0.04 %
(`REPORT.md:109-125`, retraction R4 discharged). Qualifiers that travel with it, **all binding**:
per-rank per **in-bucket** step (not "sealed"); whole-run aggregate; **includes** the ~56 %
dummy steps at n_orig≈1; token rows, not top-8-expanded rows; **n = 1 arm**; and the `/8`
peer-summing convention is *assumed*, with the receipt printf never read on the node
(open gate, `REPORT.md:414-416`).

**Fill is non-stationary within a run.** Per 100-step window (`BOTTLENECK_SIGNALS.md:275-278`):

| window | stock rank-tok/ib-step | stock φ | m15 rank-tok/ib-step | m15 φ |
|---|---|---|---|---|
| 100→200 | 1,707.9 | 0.4170 | 1,438.0 | 0.3511 |
| 200→300 | 1,513.3 | 0.3695 | 1,371.5 | 0.3348 |

φ **declines monotonically through the run in both arms** — a closed-loop drain effect. Any
single-φ workload coordinate is a run-average over a moving quantity.

**Rigorous bounds at c32p** (`BOTTLENECK_SIGNALS.md` §4.4): with mean φ=0.3757 and the in-bucket
predicate forcing ≥1 of 8 ranks to the top of the bucket, the fraction of *rank-steps* carrying
fewer than 4,096 real rows is in **[62.4 %, 87.5 %]**; the fraction of sealed steps fully packed
on all 8 ranks is **≤ 37.6 %** and, given `min_orig=1` is observed, effectively **zero**.

**Both arms pay the padding.** `VLLM_MOE_SKIP_PADDING` defaults to `0`, the router emits ordinary
top-k for pad rows, Mori dispatches `a1` unsliced `[4096, 7168]`, AITER's GEMM bound is the
post-dispatch receive count including pads, and combine writes all 4,096 rows back
(`FILL_AWARE_DESIGN.md:68-79`, tracing `M23_RAGGED_SEAL_DESIGN.md` §3.2). **Production does not
scale with fill either.** Fill-awareness is a structural opportunity, not gap-closing.

**Waste, in wall time** (`BOTTLENECK_SIGNALS.md:38-42, 311-313`): 62.4 % of every MoE row the m15
arm computed at c32p was padding (50.8 % for stock) ⇒ at f≈0.45, **≈65 s of m15's 217.6 s** and
≈43 s of stock's 200.5 s went to rows that do not exist.

### W2 — Dummy / degenerate step rate

Two distinct instruments, two distinct populations, now reconciled:

| instrument | population | rate | source |
|---|---|---|---|
| `uniform_rescued / in_bucket`, **whole run** | c32p m15 arm | **56.0 %** (139.62/249.12) | `BOTTLENECK_SIGNALS.md:341-343` |
| same, stock arm | c32p stock | **49.3 %** (101.50/205.88) | same |
| routing-homogeneity classifier over the corpus | first 64 B4096 calls/worker | **31.2 %** (160/512 rank-calls) | `CORPUS_FINDINGS.md:9-16`; `G0B_LOCAL_FILL_HISTOGRAM.md` §1 |
| `uniform_rescued / steps`, c512p | c512p m15 | **2 %** (6/300) | `LOG.md:345-350` |
| `uniform_rescued / steps`, c32p | stock 33.8 % / m15 46.5 % | | `BOTTLENECK_SIGNALS.md:268` |

**Reconciliation (binding, `REPORT.md:393-427`):** the corpus was right about *shape*, the
receipt about *rate*. The corpus window is the **ramp phase**; composing corpus shape with the
receipt rate (0.442 × 0.9146 ≈ 0.404 vs the receipt's 0.376) closes ~86 % of the apparent gap.
Tier-1 (whole-dummy-step skip) therefore targets **~56 % of in-bucket steps at c32p**, not 31 %.

**What a dummy step *is*, mechanically:** a step where the DP group is in the (512, 4096] bucket
because ≥1 rank had a real chunk, and the M23 rescue routes the uniform-decode ranks through the
B4096 graph anyway. `CORPUS_FINDINGS.md:9-16`: >95 % of rows share one expert set, across-row
weight STD ≈0.008 (identical hidden states) ⇒ **the megakernel and production both run a full
4,096-row MoE for ZERO real tokens** on those calls. **91 distinct pad expert-sets**, scattered
across ranks (pad max-rank-load ~3.0×) — **pads do NOT all land on rank 0**.

**Rank-synchrony of dummy calls — SYNCHRONOUS in the capture window** (`REPORT.md:95`, commit
`2ba124c8`): per-index dummy count `c[i]` is bimodal on {0, 8} for **all 64 indices** (100 % vs a
5 % independent-rank expectation); pairwise kappa **1.000** over all 28 worker pairs; all 8
workers show an identical **31.2 %** rate; index alignment validated by a lag-0 cross-rank
`n_real` correlation peak **r = +0.996** (vs ~0.65 at ±1). ⇒ a step-level skip carries 1:1 from
rank-calls to steps with **no synchrony discount**.
**Caveat that bounds it:** the window's dummies are entirely ramp-in (idx 0–13) and drain-out
(idx 58–63) blocks — **zero mid-run dummy calls appear** — so the whole-run ~56 % *mid-run*
dummies are **not** covered by this result.

### W3 — Fill distribution *shape* (not just its mean)

`G0B_LOCAL_FILL_HISTOGRAM.md` §2, 512 rank-calls, c32p:

* deciles of `n_real`: **0, 0, 5, 1994, 4096, 4096, 4096, 4096, 4096, 4096, 4096**
* all-512: mean 2,833.2 real rows (φ 0.6917), p50 **4096**, p90 4096
* non-dummy (n=352): mean 3,746.2 (φ 0.9146), p10 2,157 (φ 0.5265), p50 4096

⇒ **strongly bimodal: ~31 % of calls near zero, ~60 % completely full, only ~10 % genuinely
partial.** This is the single most decision-relevant shape fact in the corpus and it *inverts*
the intuition that fill is a smooth dial. Modeled consequence (MODEL, `REPORT.md:94`): tier-1
(whole-step skip) captures the entire modeled win; tier-2 (row-granular plumbing) adds nothing
under this distribution.

**Why this shape probably exists — SPECULATION, but load-bearing and cheap to test:** at c32p,
ISL = 4,096 = `max_num_batched_tokens` exactly, so **each request is exactly one full prefill
chunk** — "1024 prefill chunk-events exist in both runs (1024 requests × one 4096-token chunk)"
(`BOTTLENECK_SIGNALS.md:337`). A DP rank therefore either has a *whole* chunk or *nothing*.
**If this is the mechanism, the bimodality is an artifact of ISL == bucket size and dissolves at
any other ISL** — which no measurement has ever visited (§5, hole H4).

### W4 — Routing skew

**(a) Aggregate expert popularity — the 51.9 % capture.** From the deployed 256-bin aggregate
histogram `route_hists/stock0814_aggregate.json` (58 layers, 2.332×10⁹ routed slots, **gini
0.609** aggregate, per-layer 0.68–0.83), stock arm, EPLB **off**, contiguous ("linear")
placement (`RR_PLACEMENT_NOTES.md:208-236`; `M18_REPLICATION_RESULTS.md:5-13`):

* **logical experts 0–7 carry 51.93 % of routed slots** (each 6.40–6.61 %), and under linear
  placement all eight live on **rank 0**.
* per-rank aggregate receive share, linear: **63.60 %** / 6.93 / 5.46 / 5.29 / 4.74 / 4.59 /
  4.77 / 4.61 → **max/mean 5.088×, max/min 13.84×**.
* under round-robin (`e % 8`): 13.56 / 13.33 / 12.27 / 12.08 / 12.61 / 11.86 / 12.08 / 12.22 →
  **max/mean 1.085×**.
* per-layer worst cases: worst-rank layer (gini 0.739) **5.593× → 1.296×** under RR; worst-gini
  layer (gini 0.829) 4.160× → 1.157×.
* **send side is flat**; the earlier "flat rank load" reading was send-side only
  (`M18_REPLICATION_RESULTS.md:9-10`).

**(b) Per-call skew — where the cost actually lives.** Two instruments, and they **disagree**:

| instrument | n | per-call max-rank-load p50 | p95 | notes |
|---|---|---|---|---|
| m18diag (`M18_REPLICATION_RESULTS.md:118-122`) | 90,050 router calls | **5.15×** | **6.25×** | static-set coverage bimodal (p5=1 %, p75+=77 %); **even ideal static redistribution leaves p95 at 3.05×** |
| routecap1 corpus, **real rows only** (`CORPUS_FINDINGS.md:17-22`) | 512 rank-calls | **2.61×** | 3.35× (p90) | first-64-calls bias; modal row-pattern is exactly {0..7}, ~16 % of rows |

**UNRECONCILED.** The two differ by ~2×. Candidate causes (SPECULATION): different populations
(all router calls incl. decode/warmup vs B4096 calls only), different row filters (all rows incl.
pads vs real rows only), and the m18diag capture's known ~18 % warmup/profile contamination
(`M18_REPLICATION_RESULTS.md:11-13`). **The campaign must not mix them**, and reconciling them is
a cheap CPU-only work item.

**(c) Run-correlation — FALSIFIED as a first-order effect.** The standing hypothesis (adjacent
tokens of a prompt route alike ⇒ contiguous chunks concentrate destinations far past the
aggregate) was the motivating premise of the whole route-capture rig
(`route_capture/skewhook_v2/sitecustomize.py:14-22`). Measured
(`CORPUS_FINDINGS.md:23-27`): **consecutive real rows share 57.0 % of experts vs 55.7 % after
shuffling — a 1.3-point excess.** Mechanism (a) (destination interleaving / fabric discipline)
is **WEAKENED**; those arms were demoted. Note the GPU-side captured-vs-shuffled replay
(`ROUTE_REPLAY_PLAN.md`) that would settle it at kernel speed has **never been executed**.

**(d) Kernel-level sensitivity to skew (the response function we own).** MoK, T=4096, C=28
g=353 mode=12, 5 runs (`M18_REPLICATION_RESULTS.md:24-31`):

| routing | M15 p50 | ratio vs production |
|---|---|---|
| balanced | 5,823 µs | **0.7556** |
| measured aggregate histogram | 20,739 µs | **0.8467** |
| worst layer | 22,475 µs | 0.8559 |
| m17 (RR landing order), aggregate | 20,504 µs | 0.8370 |
| m17, worst layer | 22,118 µs | 0.8413 |

**Decision already resolved:** RR *landing order* buys ~1–1.5 % under skew; the bottleneck is
**popularity concentration**, not ordering. Note the absolute jump 5,823 → 20,739 µs: skew is
worth **3.6× wall** in this rig, far more than any schedule knob in the ledger.

**(e) EPLB.** Class **MEMORY**, no in-repo receipt:
`~/.claude/projects/.../memory/prefill-coverage-finding-2026-08-18.md:24-26` records that EPLB
works on this stack (128 redundant experts = 48/GPU, NIXL communicator, ~3 s async
rearrangements, survived full load) and that **balancedness ≈ 0.41 even on synthetic prompts —
"never actually balanced."** The metric's definition (vLLM's `log_balancedness`) is not documented
in-repo and the 0.41 has no receipt; **treat as a lead, verify before use.**
What *is* documented (`M15_EPLB0_COMPOSE.md`): the 128-redundant configuration violates
`PF4H-CONTRACT-001` (384/48 vs frozen 256/32), so **EPLB has only ever run on stock-target
servers, never composed with the megakernel**; placement-only EPLB0 (`num_redundant_experts=0`)
is designed but unrun, costs **≈1.31 GiB/rank** of transfer buffer plus a 56.6 MiB load window
(budget 1.4–2.8 GiB/rank, §7), and its offline predictor `eplb0_predict.py` runs vLLM's own
policy over our skew histograms with **no GPU** (§8) with an explicit gate: *if the predictor says
worst-layer critical-rank relief is under ~15 %, do not run the arm.*
Also §8: read the predictor as an **upper bound** — the policy fits the *window mean*, so per-call
variance is untouched. That is the same wall the RR analysis hits: **RR is a static
redistribution, and the ideal static bound leaves per-call p95 at 3.05×**
(`RR_PLACEMENT_NOTES.md:246-262`).

### W5 — Step composition: what the serving stack does to batches

This is the mechanism layer that converts (concurrency, ISL, OSL) into what the kernel sees.

**Bucketing (`M23_RAGGED_SEAL_DESIGN.md` §1.1, DISP/GMR/DPU citations therein):**

1. The generic cudagraph capture list **stops at 512** (`max_cudagraph_capture_size = 512`,
   DISP:215-216); `max_num_batched_tokens = 4096` (asserted DISP:236-241).
2. **Any** non-uniform, non-LoRA batch with `512 < num_tokens ≤ 4096` routes to the single
   **B4096 PIECEWISE** key (DISP:331-359). There are no intermediate buckets.
3. DP coordination all-reduces token counts; when any rank will use a cudagraph,
   `should_dp_pad` is true and **every rank is padded to the max across ranks** (DPU:77-89).
4. ⇒ **if one rank is in bucket, all eight are** (M23 §1.1). "In-bucket" is a *group* property.
5. ⇒ both arms execute exactly `4096 × 58` MoE row-layers per rank on every in-bucket step;
   *the padding is a property of the deployment, not of the kernel*
   (`BOTTLENECK_SIGNALS.md:325-330`).

**Measured step composition at c32p** (`BOTTLENECK_SIGNALS.md:258-262, 296-310`):

| quantity | stock | m15 |
|---|---|---|
| in-bucket step fraction | 0.686 | **0.830** |
| in-bucket steps of 300 | 205.88 | **249.12** (κ = **1.2101**) |
| padded MoE rows per real token | 2.154 | **2.568** (W = **1.1922**) |
| wasted fraction of MoE rows | 0.5357 | 0.6106 |
| **prefill chunk-events per in-bucket step (of 8 possible)** | **3.93** | **3.19** |
| token coverage `in_bucket_sum_orig / sum_orig` | 0.8808 | 0.9437 |

**The κ finding is a workload/scheduler effect, not a kernel effect** and it is the largest single
number in the c32p forensics: the m15 *run* scheduled its prefill chunks **19 % less densely
across the DP group** for the same real traffic. Co-scheduling headroom
(`BOTTLENECK_SIGNALS.md:478-492`): perfect DP co-scheduling would need **128** padded steps for
1024 chunk-events instead of the observed 260/322 ⇒ **15–25 % of node throughput sits in DP
prefill co-scheduling, for either kernel** — larger than any kernel delta in the ledger.

**DP starvation at low C, quantified:** at c32p only **3.19–3.93 of 8** DP ranks carry a prefill
chunk on an in-bucket step, and ~56 % of in-bucket steps carry *no* real prefill at all
(uniform-rescued). At c512p the same measurement collapses: 98.5 % fill, 2 % rescued. The
protocol doc names the cause explicitly (`nightshift/BENCHMARK_PROTOCOL.md:36-42`): the dummy
share at c32p is "**partly an artifact of concurrency-32 with OSL=8 — ranks idle at the tail of
every wave**", which is why open-loop cells were specified.

### W6 — Tokens per rank (the kernel-visible batch size) — and its provenance trap

The BRIEF digest quotes a prefill T-sweep "M15 C=28 vs production: 0.7557 / 0.8845 / 1.0940" and
a crossover at ~1,600–1,800 tokens/rank. **The measurements are real and campaign-grade; the
body label is not M15.** Provenance chain:

* Source is `exp_04_tgen` (`git show 43291977:.../aug13/exp_04_tgen/result.md:226-300`,
  slide-outline citation `nightshift/SLIDE_OUTLINE.md:190-199`): three 5-rotation campaigns,
  gates green, spin fail 0/1:

  | T | prod p50 | M15-C=28 p50 | ratio |
  |---|---|---|---|
  | 4096 | 7,712.6 µs | 5,828.8 µs | **0.7557** |
  | 2048 | 3,857.4 µs | 3,411.8 µs | **0.8845** |
  | 1024 | 2,064.1 µs | 2,258.2 µs | **1.0940** |

* That experiment existed **because the M15 chassis had a T≠4096 defect** (the M2 hole-sentinel
  assumes `NCHUNK_MAX·CHUNK == MAXTOK`, true only at 4096; at MAXTOK≤2048 phantom chunks
  overwrite live segments). `exp_04_tgen` **fixed it on branch `ablations-tgen`** — a different
  compiled body (sha `20b8c6bb`) from the serving M15 chassis (`935f555e`).
* The fix was never merged into the serving body: on 2026-08-18 the M15 body **failed the MoK
  correctness gate at T=2048 and T=1024** (rel 0.874205 / 0.838890) while `production` passed in
  the same runs (`REPORT.md:88`, M5). A tgen-body control passed at T=2048 in the same tree
  (`REPORT.md:89`, M6, n=1).
* Two archived MoK campaigns named `t2048_m15_*` / `t1024_m15_*` (0.8912 / 1.1051) are the *same*
  pin trap and are retracted as M15 data (`DECOMP_RUNBOOK.md:112-120`, R1). They do, however,
  **agree with exp_04_tgen within ~1 %**, so the two independent tgen-body sweeps corroborate.

**Honest statement of the law:** *for the fused-megakernel family* (both `pf6gm` and M15 invert:
pf6gm/prod 0.8938 → 0.9529 → 1.1098), the advantage shrinks roughly linearly in log-T and inverts
between T=2048 and T=1024, break-even ≈ **T 1,600–1,800 at C=28**, because fixed pipeline
fill/drain and plan overhead amortize over fewer tokens (M15's `m2_to_end` shrinks 5,310→1,993 µs
for a 4× token reduction — only 2.66×). **The serving M15 body has never produced a number at
T≠4096.** Also note `K0_MAXTOK == T` in the harness ⇒ **the T-sweep is a batch-size sweep, not a
fill sweep** (`REPORT.md:435-438`): capacity shrinks with T, so it never isolates padding.

### W7 — ISL / OSL and their effect on step composition

Measured only at **ISL 4096 / OSL 8** (both cells) and, historically, ISL 4096 / OSL 8 again.

* **ISL 4096 == `max_num_batched_tokens` 4096** ⇒ one chunk per request, no chunk multiplicity
  (`BOTTLENECK_SIGNALS.md:337`). See §W3 for why this probably *causes* the bimodal fill.
* **OSL 8** with `ignore_eos: true, temperature 0.0` ⇒ exactly 8 output tokens, 7 ITLs per
  request (`.../serving/b3rev2_pair_01_2_m15.json` workload block; `per_query[i].itl_ms_count = 7`).
* **DERIVED (this file), from the in-repo manifests:** the share of per-request wall clock spent
  *after* first token —
  c32p m15: (6,335.6 − 2,444.8)/6,335.6 = **61.4 %**;
  c512p m15: (48,469.6 − 43,191.6)/48,469.6 = **10.9 %**;
  c512p native_tuned_dp: (51,379.4 − 28,941.5)/51,379.4 = **43.7 %**.
  ⇒ even in the cell we call "the prefill headline" (c32p), **~61 % of a request's latency is
  accumulated in the decode phase**, which the megakernel's seal never covers as real work.
* **Token-weighted coverage:** c32p issues 4096 prefill + 8 decode tokens per request, so ≥99.8 %
  of *tokens* are prefill (`M23_RAGGED_SEAL_DESIGN.md:150-155`) — the token-weighted and
  latency-weighted views of the same cell disagree by a factor of ~1.6. **Which weighting the
  metric map uses is a first-class campaign decision.**
* `c8/c16/c32` (ISL 1024, OSL 512) and `c512` are defined as the decode-bound cells and are
  explicitly expected to invert (`nightshift/CLAUDE.md:160-167`); **never run** (§5, H1).
* `FILL_AWARE_DESIGN.md:1319-1323` states the rule: "**37.6 % is c32p's number**… every quoted
  cell must carry its own measured fill", and the `M24_FILL_RECEIPT` that would supply it does
  not exist yet.

### W8 — Topology / parallelism as a workload-dependent choice

**8.1 The regime split (the draft8 headline mechanism).** AMD recommends *different topologies in
the two concurrency regimes* — TP8+EP for ≤128 concurrency, DP8+EP for ≥512, with the **128–512
gap undocumented by the vendor** (`LOG.md:356-360`). Measured:

| cell | our arm (DP8/EP8 + m15) | vendor-tuned native | delta | fill / dummy |
|---|---|---|---|---|
| c32p | 20,371.6 tok/s | 25,844.0 (TP8+EP) | **−21.17 %** | φ 0.376, ~56 % dummy |
| c512p (pair 1) | 42,788.0 | 39,474.0 (DP8+EP) | **+8.40 %** | φ ≈0.985, 2 % dummy |
| c512p (pair 2, reversed) | 42,835.7 | 40,183.1 (DP8+EP) | **+6.60 %** | as above (seal 100 %, 299/299) |

Sources: `.../serving/b0v5_pair_01_*.json`, `b3cal_pair_01_*.json`, `b3rev2_pair_01_*.json`;
`LOG.md:306-315, 342-355`. Coverage receipts: c32p m15 seal 98 %, RECEIPT_GATE=PASS; c512p m15
seal **100 % (299/299 in-bucket)**, `uniform_rescued` 6/300, 0 failed; native arms 0 failed with
authenticity receipts captured.

**The conclusion the notes are careful about** (`DRAFT8UPDATED_SPEAKER_NOTES.md:57-59`): it is
**not** that DP is intrinsically low-concurrency and TP high-concurrency. It is that *topology
economics depend on useful fill and on the communication boundaries the topology introduces.* And
the honest caveat, verbatim: "The 37.6 % and 98.5 % figures come from **different serving cells,
not a controlled fill-only ablation.** They explain the regime split, but they do not prove fill
is the only changing term."

**8.2 What the reverse pair adds (new, unreported anywhere else).** `b3rev2` landed after
`REPORT.md` was signed and is not in LOG, REPORT, or the speaker notes. DERIVED from the two
manifests: pair ratios **1.08395** and **1.06601**, geomean **1.0750**; **both pairs favour m15**.
This is n=2 order-balanced — still below the binding ≥5-pair floor
(`BENCHMARK_PROTOCOL.md:50-53`) — so it is **directional**, but it is the first c512p evidence
that survives order reversal, and it materially weakens the "position effect could be the whole
8.4 %" caveat the presentation carried.

**8.3 The metric map is regime-dependent — measured.** c512p p50s (median of the two pairs):

| metric | m15 (DP8/EP8) | native_tuned_dp | m15 / native |
|---|---|---|---|
| input tok/s | 42,811.9 | 39,828.6 | **1.075×** (better) |
| TTFT p50 | 43,168 ms | 28,911 ms | **1.49× worse** |
| TTFT p99 | 47,895 ms | 52,020 ms | **0.92× better** |
| TPOT p50 | 757.1 ms | 3,208.8 ms | **0.236× — 4.2× better** |
| e2e p50 | 48,452 ms | 51,344 ms | **0.944× better** |

⇒ **at c512p the arm that loses TTFT p50 by 49 % wins e2e p50, because it wins TPOT by 4.2×.**
Any single-metric headline at this cell is a choice, not a fact. (Contrast c32p, `b0v5`: m15
TTFT p50 2,444.8 vs native TP8 1,338.6 — m15 loses p50 — but m15 TTFT **p99** 4,364.1 vs 5,682.2
— m15 wins p99. Same inversion, opposite direction.)

**8.4 Architecture as a workload axis.** `gfx942`/MI300X: remote atomics ~3× slower than remote
stores ⇒ producer-carried atomic accumulation may be wrong there (BRIEF digest §draft8). On
gfx950 the standalone CDAR bench measured store-towers 93.6 GB/s vs atomics ~67 GB/s
(`nightshift/LOG.md`, `distributed-kernels/tp8_mega/results/`). No serving measurement exists on
any hardware other than the one 8×MI355X node.

### W9 — Coverage / seal: why the mega historically ran ~2 % of steps

The mechanism (`M23_RAGGED_SEAL_DESIGN.md` §1.2–1.3):

* the old seal required `original_num_tokens_across_dp == (4096,)*8` — **eight simultaneous full
  pre-padding chunks**. Under chunked prefill + prefix caching at c32p that is a coincidence, not
  a regime; `fail_min_tok = 1` says the modal failure is **a single rank running a one-token
  decode batch**, which DP padding then inflates to 4096 anyway.
* per rank per c32p run: ~600 steps, ~230 in-bucket (38.3 %), production's stock B4096 graph runs
  **all** of them, the megakernel was sealed for **~4.6** (2 % of in-bucket, **0.77 % of all
  steps**). Coverage multiplier from M23: **~50×** → 100 % of in-bucket = 38.3 % of all steps.
* **Second, compounding defect:** on an unsealed in-bucket step a PF4H-target server did not fall
  back to the stock B4096 *graph* — it fell back to **no graph at all** (`cudagraph_mode = NONE`,
  GMR:3973-3981), because the PF4H-only server never registered the `regular_b4096` key. The
  candidate arm ran ~225 of ~230 heavy steps **eagerly** while the baseline ran them under
  PIECEWISE replay. **Every pre-M23 serving A/B measured a doubly-handicapped candidate.**
* Consequence, recorded in the repo: `aug14/M18_REPLICATION_RESULTS.md:100-113` **removed** its
  three serving pairs (−5.7 %, +11.3 %, +39.8 %) for exactly this reason. Kernel-level results in
  that file are unaffected.
* Post-M23 measured coverage: **99.15 %** (1,976/1,993, m23pair2, `DECOMP_RUNBOOK.md:18`),
  **96.9 %** (251/259, camp3 pair_02 — R2 corrects the banked "99 %"), **98 %** (b0v5 c32p),
  **100 %** (299/299, c512p).
* `RAGGED_SEAL_RECEIPT` schema (`M23_RAGGED_SEAL_DESIGN.md:815-830`) — this is the workload
  instrument, not just a gate: `steps, in_bucket, sealed, sealed_ragged, sealed_exact,
  refused_not_unanimous, refused_not_ready, eager_b4096, min_orig, max_orig, sum_orig`.

**Standing rule that falls out:** no serving delta may be interpreted without its arm's
`sealed/in_bucket`. Every workload coordinate below is conditioned on the kernel actually having
run the traffic.

### W10 — Run-to-run variance and hardware heterogeneity

**(a) The banked drift band** (`docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md:229-231`):
**±15 % day-to-day at n=1**, **±18 % per arm from position** (which arm runs first inside a
pair). Mitigations that are now protocol: `ARM_COOLDOWN=240 s` symmetric, fresh server per arm,
order-balanced pairs, **≥5 pairs hard floor**.
Corroborating spread: camp3 n=2 gave **+23.5 % / −13.1 %** across pairs for a +2.70 % mean
(`REPORT.md:86`).

**(b) But variance is regime-dependent — DERIVED from the c512p manifests.** Across the two
c512p pairs (opposite arm order, ~1 h apart):

| arm | run 1 | run 2 | spread |
|---|---|---|---|
| m15 input tok/s | 42,788.0 | 42,835.7 | **+0.11 %** |
| native_tuned_dp input tok/s | 39,474.0 | 40,183.1 | **+1.80 %** |
| m15 TTFT p50 | 43,191.6 | 43,145.0 | −0.11 % |
| m15 TPOT p50 | 757.3 | 756.8 | −0.07 % |
| native TPOT p50 | 3,210.0 | 3,207.5 | −0.08 % |

⇒ at c512p, **arm-position + hour-scale drift is ≤1.8 % on throughput and ≤0.25 % on latency
percentiles** — an order of magnitude below the ±18 % band measured at c32p. Applying the c32p
drift band blanket at c512p would be over-conservative. **The variance envelope is itself a
function of W and must be measured per cell.**

**(c) Thermal rank drift (training arm; the only place it has been root-caused).**
`memory/v6-campaign-2026-08-18.md` finding (5) + `overnight/aug18/V6_EVIDENCE_AND_DESIGN.md:40-48,
64-68`: the M2 `rows_done` wait is **rank-structural** — 92 µs (rank 2) to 4,353 µs (rank 0) per
CTA per launch; **rank 2 is the straggler and seven ranks stall ~2.5 ms every backward launch**.
Root cause: **GPU2 is the hottest (77 °C) and lowest-clocked (1,725–1,743 MHz vs 1,790–1,822)** —
a **4–5 % clock spread**. The drift is **data/hardware-driven, not protocol slack**; filler cannot
absorb it on the critical rank. The associated **critical-rank law**: iteration pace = the
hottest/slowest rank's wall clock; only work removed from *that* rank's wall moves e2e.
Also measured in the same arm (`overnight/aug18/FAIRNESS_AUDIT.md:554, 620`): **+18.7 ms
first-run penalty** and **+3.4 ms monotone thermal soak across three back-to-back runs**.
**Never measured on the serving arm** — but the megakernel's slab rendezvous couples all ranks to
the slowest one identically, so a hot-GPU/hot-expert-rank coincidence is a plausible confound
that no serving instrument currently sees.

**(d) Numeric nondeterminism (bounds what any parity gate can do).** The mode-12 epilogue
accumulates several experts into one slot row with **packed bf16 atomics** whose order is set by a
per-step tile schedule ⇒ **the kernel is not bit-reproducible run to run**
(`FILL_AWARE_DESIGN.md:1300-1317`, rev-2 withdrawal of bit-identity). Measured consequences:
**116/1024 requests (11.33 %) produced different output tokens between arms**
(`BOTTLENECK_SIGNALS.md:44-46`), and `ordered_output_token_id_stream_sha256` differs even
**stock vs stock** (`4ee5ba42…` vs `936b4f5a…`, `REPORT.md:167`) — diagnosed as numeric
(chunked-prefill boundaries, prefix cache, DP-rank assignment, combine order), not a bug.
**Exact-token SHA cannot validate anything at c32p.**

---

## 3. What the serving stack controls vs what the kernel sees

### 3.1 The control plane (server-side, per-arm configuration)

From `campaign_v5/ARMS.md:90-100` (our arm) and `:190-210` (AMD tuned native):

| knob | our m15/stock arm | AMD `native_tuned_dp` | effect on the kernel's workload |
|---|---|---|---|
| `--data-parallel-size` / `--tensor-parallel-size` | DP8 / TP1 | per vendor rec | decides whether MoE all-to-all exists at all |
| `--max-num-batched-tokens` | **4096** | 16384 | **sets the bucket the mega is pinned to**; ISL==this ⇒ one chunk/request |
| `--max-num-seqs` | **128** | 2048 | caps admitted sequences; at c512p the client offers 512 in flight against a 128 cap (**interaction never characterized — verify**) |
| `--enable-chunked-prefill` | on | on | creates the ragged/mixed batches the seal must accept |
| `--enable-prefix-caching` | **off in v5.1** (was on in camp3/m23pair) | off | changes how many real tokens a chunk carries ⇒ changes φ directly |
| `--kv-cache-dtype fp8` | ours only | shipped default | memory headroom, and a declared accuracy asymmetry |
| `--gpu-memory-utilization` | 0.70 | 0.9 | one B4096 graph per server (memory headroom) |
| `--scheduling-policy fcfs` | fcfs | — | determines chunk co-scheduling density (the κ mechanism) |
| capture size list | max 512 | — | everything in (512, 4096] collapses into one bucket |

**The single largest lever nobody has pulled:** `max_num_batched_tokens` *is* the bucket size.
`FILL_AWARE_DESIGN.md:1337-1345` (§H.9) names it: the ~46–60 % of step time that is *not* MoE also
runs on 4,096 padded rows, and shrinking the bucket / adding smaller buckets is an **integration**
fix that removes pad rows before the model ever sees them. Never swept.

### 3.2 What the kernel actually sees today

* `T = 4096`, written once into descriptor slot `K0P6_D_T` **out of capture**
  (`m15_contracts.py:537-538`, `m15_runtime.py:941`, cited in `FILL_AWARE_DESIGN.md:88-91`). The
  kernel has **no idea** how many rows are real.
* `C = config.reserved_comm_ctas` — **runtime**, the low 8 bits of `K0P6_D_MPS_CFG`, not a compile
  constant (`FILL_AWARE_DESIGN.md:59-66`). C=28 is a *measurement configuration*, so it can be
  made workload-aware without a recompile.
* Routes arrive as `topk_ids` in **physical slot space**; under EPLB those ids move on every
  rearrangement, which invalidates any replay harness keyed on them
  (`M15_EPLB0_COMPOSE.md` F10).
* `n_orig` reaches the kernel **only** through M24's descriptor slot 71 fill vector — implemented
  behind 9 default-0 macros (`REPORT.md:456`) but **never compiled**, and its serving-side
  plumbing (G13a–G15) does not exist (`REPORT.md:471-480`).
* Dummy/uniform-decode steps are **not** invisible to the kernel: post-M23 the rescue routes them
  through the B4096 graph, so the mega executes 4,096 padded rows for ~1 real token per rank
  (`CORPUS_FINDINGS.md:9-16`). Whatever the kernel does on a decode step, it does at prefill
  shape.

### 3.3 Cross-arm structural asymmetries that the stack, not the kernel, creates

* The PF4H patch **removes** the ≤256 mixed PIECEWISE keys whenever
  `VLLM_PF4H_INTEGRATION_MODE` is set (DISP:204-212) — a pre-existing confound the A/B must
  control (`M23_RAGGED_SEAL_DESIGN.md:764-772`).
* The uniform-decode rescue roughly **doubled** stock throughput and is **our** patch; protocol
  requires it be given to the baseline too or counted as "integration effect"
  (`BENCHMARK_PROTOCOL.md:22-29`). The three-way decomposition (`m15/native` headline ·
  `m15/patched-stock` kernel effect · `patched-stock/native` integration effect) exists because of
  this.
* Prefix caching created a **~10.8 % duplicate-prompt asymmetry** in v5.0 (only the uncached side
  paid for ~111 full 4,096-token prefills per c32p cell); fixed two ways in v5.1 — caching off
  everywhere, and a QSL generator that strides the pool so a cell of up to 4,388 prompts has no
  exact duplicates (`ARMS.md:294-310`). **Absolute tok/s before and after are not comparable; the
  ratios are.**

---

## 4. Instruments that exist, and their known artifacts

| instrument | what it yields | where | known artifacts / limits |
|---|---|---|---|
| **`RAGGED_SEAL_RECEIPT` / coverage counters** (`coverage_patch.py`, `m23_patch.py`) | per-rank cumulative `steps / in_bucket / sealed / sealed_ragged / sealed_exact / refused_* / eager_b4096 / min_orig / max_orig / sum_orig` | `M23_RAGGED_SEAL_DESIGN.md:815-830` | the **`/8` peer-summing convention is assumed** — the printf has never been read on the node (`REPORT.md:414-416`); receipts are per **in-bucket** step, not per sealed step |
| **skew hook v2** (`route_capture/skewhook_v2/sitecustomize.py`) | v1 per-layer 256-bin histograms + **raw per-chunk capture** (`topk_ids [4096,8] uint8`, `topk_weights` fp16) | that file, lines 1-40 | **captures only calls with exactly 4,096 rows** ⇒ the corpus is `[4096,8]` by construction and **can never see fill** (`DECOMP_RUNBOOK.md:45-48`); **skips graph-capturing calls** (D2H inside capture is illegal) ⇒ selection bias toward eager/in-bucket steps; bounded at 64 calls/rank ⇒ **ramp-phase window** |
| **route replay loader** (`mok_synthetic_prefill/route_replay.py`) + `synthetic_inputs.py` + harness env `K0_MOK_ROUTE_*` | captured vs **shuffled** (fixed-seed) replay with `route_multiset_sha256` equality proving only correlation changed; 22 CPU tests pass | `ROUTE_REPLAY_PLAN.md` §1 | **status: code complete, locally validated, NEVER EXECUTED ON GPU**; `mok_eager` only, by design; combining with `K0_MOK_ROUTE_HIST` is a hard error |
| **aggregate histogram replay** (`K0_MOK_ROUTE_HIST`, Gumbel-top-k) | marginal 256-bin popularity fidelity: corr 0.998, gini 0.331 vs 0.341 target | `M18_REPLICATION_RESULTS.md:15-21` | **fixes only the marginal** — destroys per-chunk composition and token-to-token correlation; every kernel number before aug18 rests on it |
| **route corpus analyzers** (`analyze_pad_routing.py`, `m24/g0b_local_fill.py`, `m24/dummy_synchrony.py`) | dummy rate, fill histogram, `T_eff`, rank-synchrony | `CORPUS_FINDINGS.md`, `G0B_LOCAL_FILL_HISTOGRAM.md`, `DUMMY_SYNCHRONY.md` | **nz-filter artifact**: `g0b_local_fill.py:89` filters `n_real > 0`, dropping the 91 zero-fill calls ⇒ the withdrawn 1.19× (R7); honest aggregate **1.446×**. **Ramp bias**: dummies in the window are only idx 0–13 and 58–63 |
| **`analyze_serving_pair.py`** | re-runs the whole c32p forensics (step composition, κ, TTFT quantiles, two-state wall fit) on any future pair, stdlib-only | `aug18-prefill/analyze_serving_pair.py` (37 KB) | consumes `c32p.json` + client log + `seal_receipts.txt`; needs the receipts to be present |
| **MoK synthetic prefill rig** | in-region ratios at fixed T with correctness gates + `[MPS TS]` phase ledger + `[MPS SPIN]` fail counters | `DECOMP_RUNBOOK.md` | **pin trap**: the harness compiles `..._mps.hip`, which on `ablations` is the *old* MPS sibling; every arm needs an explicit `DHK_ROOT` pin. **Realized in our own archive** (R1). `K0_MAXTOK == T` ⇒ T-sweeps are batch-size, not fill, sweeps |
| **EPLB offline predictor** (`m15_eplb0/eplb0_predict.py`) | vLLM's own `DefaultEplbPolicy` over our skew histograms, per-rank load before/after, **no GPU** | `M15_EPLB0_COMPOSE.md` §8 | reports an **upper bound** — fits the window mean, leaves per-call variance untouched; the same file's gate: skip the arm if predicted worst-layer relief < ~15 % |
| **serving manifests** (client v3, schema `pf4h-exact-token-closed-concurrency-v2`) | full workload tuple + per-query TTFT/TPOT/ITL/e2e + prompt-stream SHA + output-stream SHA | `distributed-kernels/tp8_mega/results/serving/*.json` | **6 files, 2 cells**; `ordered_output_token_id_stream_sha256` is not a usable parity gate (§W10d) |
| **histogram/ratio contamination note** | the m15 arm ran ~2× traffic from gate self-tests and diluted its own histogram; **~18 % of the measured stock-arm traffic is warmup/profile batches** | `M18_REPLICATION_RESULTS.md:11-13` | **use stock-arm histograms only** |

---

## 5. What has NEVER been measured — the holes the grid must cover or explicitly scope out

Ordered by how much of the campaign's claim surface they threaten.

| # | Hole | Status of evidence today | Why it matters to the cost model |
|---|---|---|---|
| **H1** | **Decode-phase behaviour as a regime.** No cell with OSL ≫ 8 has ever run: `c8/c16/c32` (ISL 1024, OSL 512) and `c512` are defined (`campaign_v5/run_m15_campaign_eplb_v5.sh:1165-1210` `cell_params()`; `campaign_v5/NIGHT_SCHEDULE.md:74-77`) and **never executed**. | Zero. The mega *does* execute decode steps (rescued to B4096) but no timing attributes cost to that class. | TPOT is the decode metric and the mega's TPOT advantage at c512p (4.2×, §W8.3) is entirely unexplained. A cost model with no decode term cannot predict the metric the market cares about. |
| **H2** | **Mixed prefill+decode batches as a controlled variable.** Chunked prefill mixes them constantly, but the only instrument is step *counts* (`uniform_rescued`, `in_bucket`); no per-step split of prefill rows vs decode rows exists. | Indirect only (`min_orig=1 max_orig=4096` proves mixing happens). | The mediating quantity for the fill law is the *joint* (n_orig across 8 ranks) distribution, not its mean. |
| **H3** | **Concurrency between 32 and 512.** Nothing measured; the vendor's own recommendation flips somewhere inside this gap and **AMD documents nothing there** (`LOG.md:356-360`). | Zero. Two points, 16× apart. | The entire "regime split" law is a two-point extrapolation. The crossover concurrency is the single most quotable number the campaign could produce, and we cannot currently state it. |
| **H4** | **ISL other than 4096.** `c8/c16/c32/c512` specify ISL 1024 — never run. ISL 4096 == `max_num_batched_tokens` ⇒ one chunk per request (§W3). | Zero outside ISL 4096. | If the bimodal fill distribution is an ISL==bucket artifact, **every tier-1 (dummy-skip) sizing number is cell-specific** and the design's central premise moves. |
| **H5** | **OSL other than 8.** | Zero. | OSL sets the decode:prefill step ratio, hence the fraction of steps the seal can ever cover, hence Amdahl. |
| **H6** | **Open-loop / arrival-driven load.** `o50p/o75p/o90p` cells specified (Poisson at 50/75/90 % of the measured ceiling, `OPEN_LOOP_BASE_RATE = 4.97 req/s`, `LOG.md:315`); **never run**. | Zero. Every number is closed-loop saturated. | `BENCHMARK_PROTOCOL.md:36-42` states the reason bluntly: the closed-loop dummy share is *partly an artifact*, and only open-loop decides whether the dummy-skip advantage is production value or a benchmark artifact. |
| **H7** | **Fill as a controlled axis, at any level.** R6 (the cheap T-sweep confirmation) is **dead**: `K0_MAXTOK == K0_T` makes it a batch-size sweep, and the M15 body fails MoK correctness at T≠4096 anyway (`REPORT.md:88, 435-438`). M24 (the fill-aware kernel) has **never compiled**. | Zero controlled points. Everything is observational across cells. | The BRIEF's headline mechanism ("fill explains the regime split") is currently supported by **two cells that differ in ~six variables at once**. |
| **H8** | **Captured-route replay on GPU.** `ROUTE_REPLAY_PLAN.md` is code-complete, unit-tested, deployed nowhere. | Zero. | It is the only experiment that separates *popularity concentration* from *run correlation* at kernel speed, and it is the cheapest predictive instrument in the inventory (139 s per campaign vs ~25 min per serving pair). |
| **H9** | **Skew as a controlled end-to-end axis.** RR placement is implemented and offline-verified (`rr_patch.py`, `RR_PLACEMENT_NOTES.md`); campaign 4 never ran. EPLB has **never been composed with the megakernel** (contract violation, §W4e). | Kernel-level replay only (0.7556 / 0.8467 / 0.8559). | Skew is worth 3.6× wall in the rig — the largest single response in the ledger — and we have no serving-level measurement of the remedy. |
| **H10** | **Multi-node.** Everything is one 8×MI355X node, intra-node xGMI. | Zero. | Any claim about "topology as a workload axis" that implies inter-node scaling is unsupported. Recommend **explicitly scoping out**. |
| **H11** | **Hardware other than gfx950 for serving.** gfx942 evidence is microbench-level only (remote atomics ~3× slower than remote stores). | Zero serving points. | The producer-carried-accumulation law is architecture-conditional and we can only *predict* the inversion. |
| **H12** | **Bucket size / `max_num_batched_tokens` as a knob.** Fixed at 4096 in every run. | Zero. | It is the lever with the largest modeled reach (§3.1) and it is **integration-side**, i.e. cheap. |
| **H13** | **Prefix caching as a workload axis.** camp3/m23pair ran with it ON; b0/b3 with it OFF. Never varied as a controlled axis; absolute numbers across those campaigns are not comparable (`ARMS.md:294-310`). | Zero controlled points. | It changes φ directly by changing how many real tokens a chunk carries. |
| **H14** | **`max_num_seqs` (128) vs offered concurrency 512.** The c512p cell offers 512 in flight against a 128-seq admission cap. | Not characterized anywhere. **Verify before quoting c512p as "concurrency 512".** | If the server admits 128, the c512p "concurrency" coordinate is mislabeled and the two-point regime law is measured at a different point than stated. **SPECULATION — flagged as a verification item, not a finding.** |
| **H15** | **Per-cell fill receipts.** `M24_FILL_RECEIPT` does not exist; c512p's 98.5 % is reported in LOG from the seal receipt but the derivation is not shown in-repo. | Partial. | `FILL_AWARE_DESIGN.md:1319-1323` already makes per-cell fill mandatory for any quoted cell. |
| **H16** | **Thermal/rank heterogeneity on the serving arm.** Root-caused only in the training arm (GPU2, 4–5 % clock spread). | Zero serving points. | The mega's slab rendezvous couples every rank to the slowest; a hot-GPU ∩ hot-expert-rank coincidence would be invisible to every current serving instrument. |
| **H17** | **Reconciliation of the two per-call skew instruments** (5.15× vs 2.61× p50). | Contradictory. | Whichever number the cost model uses changes the predicted skew tax by ~2×. CPU-only to resolve. |

---

## 6. The measured span, as a workload state vector

What a W-coordinate table can honestly claim today:

| coordinate | measured values | span | class |
|---|---|---|---|
| concurrency C | 32, 512 | 2 points, 16× apart, **nothing between** | SERVING |
| ISL | 4096 | **1 point** | SERVING |
| OSL | 8 | **1 point** | SERVING |
| arrival process | closed-loop saturated | **1 point** | SERVING |
| fill φ | 0.376 (c32p m15), 0.433 (c32p stock), ≈0.985 (c512p); within-run decline 0.417→0.370 | 2 cells + a within-run trend | RECEIPT |
| fill *shape* | bimodal: ~31 % ≈0, ~60 % =1.0, ~10 % partial (non-dummy mean 0.9146) | 1 cell, ramp window | CORPUS |
| dummy-step rate | 56.0 % (c32p m15), 49.3 % (c32p stock), 2 % (c512p); 31.2 % ramp-window | 2 cells | RECEIPT + CORPUS |
| routing skew, aggregate | 51.93 % on experts 0–7; rank max/mean 5.088× linear, 1.085× RR; gini 0.609 | 1 histogram, whole run | CORPUS |
| routing skew, per-call | p50 5.15× / p95 6.25× (m18diag) **vs** p50 2.61× / p90 3.35× (corpus) — **unreconciled** | 2 instruments, conflicting | CORPUS |
| run correlation | +1.3 points of expert overlap vs shuffled | 1 corpus | CORPUS |
| tokens/rank T | 4096, 2048, 1024 — **on the tgen body only** | 3 points, wrong body | KERNEL |
| topology | DP8/EP8 (ours), TP8+EP and DP8+EP (vendor-tuned native) | 3 configurations, 2 cells | SERVING |
| coverage (seal) | 0.77 % → 96.9–100 % of in-bucket, cell-dependent | 4 runs | RECEIPT |
| step composition | in-bucket fraction 0.686–0.830; chunk-events/step 3.19–3.93 of 8; κ=1.192 | 1 pair | RECEIPT |
| hardware | 8×MI355X gfx950, one node | **1 point** | — |
| run-to-run variance | ±15 % day / ±18 % position (c32p-era) **vs** ≤1.8 % throughput, ≤0.25 % latency (c512p) | 2 regimes, order-of-magnitude apart | SERVING |

**The honest summary for the campaign design:** we have deep, receipt-backed evidence about
**one** workload point (c32p) and a shallow but clean pair of runs at a **second** (c512p); a rich
routing corpus that is structurally blind to fill; a kernel rig that is structurally blind to
everything except tokens-per-rank and marginal expert popularity; and a T-response curve measured
on a sibling body the serving kernel cannot reproduce. Every "law" that spans more than two
workload points today is either a model or an extrapolation, and the campaign's first job is to
turn the mediating quantities in §2 (φ, dummy rate, co-scheduling density κ, per-call skew,
tokens/rank) into **measured, controlled axes** rather than **observed, confounded correlates**.

# Provenance of "37.6 % fill / ~1,539 real tokens / 2.66x padding"

Forensic extraction against the prior session transcript
`~/.claude/projects/-Users-subha-repos-Distributed-HipKittens/4077c4da-1965-47cb-aa86-44e5f55b46c4.jsonl`
(4.3 MB, 2,104 JSONL records). Written to discharge REPORT.md retraction
**R4**, which suspended the 2.66x premise for lack of provenance.

**Verdict: the primary source EXISTS, is in the transcript, and the
arithmetic reproduces exactly from receipt lines also in the transcript.
R4's "un-sourced" finding is wrong. The apparent contradiction with the
corpus's 1.19x is fully explained, and BOTH numbers are true under stated
definitions (§5).**

---

## 1. Where the figure was first stated

First occurrence in the transcript is JSONL record **1848** (duplicated at
1850), a `queue-operation` carrying the completion payload of the dynamic
workflow *"Analyze the first honest m15 serving data for bottleneck signals
and plan the kernel-level decomposition campaign"* (task `wbb9si6t9`,
tool-use `toolu_01KwsJjWvKXsPcevqEpGrtmz`, enqueued
`2026-08-18T07:28:27.882Z`). This is the **DECOMP_RUNBOOK author agent**,
whose deliverable was committed as `4cd959ec` on `ablations`.

`ranked_findings[0]`, verbatim (record 1848; `&gt;` un-escaped to `>`):

> MECHANISM (e), NEW AND LARGEST UNEXAMINED PRIOR — the mega does 2.66x the
> real work. From both arms' seal_receipts.txt (last receipt per rank, 300
> steps x 8 ranks): m15 real_tok=3,265,348 over 2,400 rank-steps = 1,361
> real tokens/rank-step; padded rows in b4096 steps = 8,093,696, so
> all-real-tokens/padded-rows = 40.3%. in_bucket_sum_orig=24,538,534 over
> 1,993 in-bucket rank-steps (8x peer-counted) => 1,539 real tokens per rank
> per sealed step = 37.6% fill of the 4,096-row dispatch => PADDING
> MULTIPLIER 2.66x (stock: 43.3% fill, 2.31x). Every banked kernel number is
> T=4096 REAL rows.

And in the same record's `summary`:

> (1) I recomputed the fill from the M23 receipts: the mega runs 4,096
> padded rows carrying ~1,539 real tokens per rank per sealed step — 37.6%
> fill, a 2.66x padding multiplier — and NO captured-route experiment can
> see this, since the replay corpus is [4096,8] by construction.

The orchestrator propagated it one record later (**1857**, assistant text):

> the mega runs at 37.6% fill (2.66× padding multiplier) on real serving
> steps, while every banked kernel number assumed 100% fill

The germ of the calculation is earlier, in the orchestrator's own brief to
that agent (**record 1555**), explicitly flagged as not-yet-done:

> (e) ragged batches: the mega always dispatches/computes full T=4096 padded
> rows — identical to production's padded graph, but the mega's RELATIVE
> fixed-cost share grows when real tokens per step are fewer (measured
> in_bucket_sum_orig/steps → mean real tokens per in-bucket step
> ~12.3k/8ranks... compute this properly from the receipts).

So the `/8` peer-count convention originates with the orchestrator and was
adopted, not independently derived, by the runbook agent.

## 2. The underlying primary data (raw receipt lines, in the transcript)

**m15 arm** — record **1518**, tool result of a node `ssh` tail, run label
`M23PAIR2 P1 M15: input_tok/s=19277 wall=218s completed=1024`, container
`m15_m23pair2_p01_1_m15`, cell **c32p** (conc 32, 1024 prompts, ISL 4096,
OSL 8), timestamp `2026-08-18T07:01:28.831Z`:

```
(Worker_DP7_EP7 pid=5842) INFO 08-18 06:59:59 [gpu_model_runner.py:4213] RAGGED_SEAL_RECEIPT rank=7 steps=300 in_bucket=250 sealed=247 sealed_ragged=245 sealed_exact=2 refused_not_unanimous=7 refused_not_ready=3 refused_peer_not_ready=3 eager_b4096=3 uniform_rescued=142 min_orig=1 max_orig=4096 sum_orig=3249465 in_bucket_sum_orig=3083410
(Worker_DP0_EP0 pid=5825) INFO 08-18 06:59:59 [gpu_model_runner.py:4213] RAGGED_SEAL_RECEIPT rank=0 steps=300 in_bucket=249 sealed=247 sealed_ragged=245 sealed_exact=2 refused_not_unanimous=5 refused_not_ready=2 refused_peer_not_ready=2 eager_b4096=2 uniform_rescued=139 min_orig=1 max_orig=4096 sum_orig=3255026 in_bucket_sum_orig=3069790
```

**stock arm** — record **1376**, tool result, timestamp
`2026-08-18T06:36:40.059Z`, same pair, `[gpu_model_runner.py:4172]`:

```
(Worker_DP3_EP3 pid=5826) ... RAGGED_SEAL_RECEIPT rank=3 steps=300 in_bucket=206 sealed=0 sealed_ragged=0 sealed_exact=0 refused_not_unanimous=13 refused_not_ready=206 eager_b4096=0 uniform_rescued=102 min_orig=1 max_orig=4096 sum_orig=3316154 in_bucket_sum_orig=2920936
(Worker_DP4_EP4 pid=5816) ... RAGGED_SEAL_RECEIPT rank=4 steps=300 in_bucket=207 ... uniform_rescued=105 min_orig=1 max_orig=4096 sum_orig=3320236 in_bucket_sum_orig=2925029
```

(A third, 100-step early receipt is at record **1508**: `rank=0 steps=100
in_bucket=70 sealed=68 ... uniform_rescued=34 ... sum_orig=1112384
in_bucket_sum_orig=1059501`.)

The transcript contains only 2 of the 8 ranks per arm; the full 8-rank
`seal_receipts.txt` lived on the node under
`${SCRATCH}/serving_artifacts/p2_m15/` and `p1_stock/` (record 1555).

## 3. The arithmetic reproduces — exactly

Extrapolating the two visible m15 ranks to 8:

| quantity | from transcript | agent's 8-rank figure |
|---|---|---|
| Σ `in_bucket_sum_orig` | (3,083,410+3,069,790)/2 × 8 = **24,612,800** | 24,538,534 (−0.3 %) |
| Σ `in_bucket` | (250+249)/2 × 8 = **1,996** | 1,993 (−0.2 %) |
| per-rank real tokens / in-bucket step | 24,612,800 / 1,996 / 8 = **1,541.4** | 1,539.0 |
| fill | 1,541.4 / 4096 = **0.3764** | 0.376 |
| padding multiplier | 4096 / 1,541.4 = **2.657x** | 2.66x |

Per-rank, without extrapolation: rank 7 → 3,083,410/250/8 = **1,541.7**
(37.64 %, 2.657x); rank 0 → 3,069,790/249/8 = **1,541.1** (37.62 %).

**Decisive confirmation via the stock arm**: 2,920,936/206/8 = **1,772.4**
→ fill **43.27 %**, multiplier **2.311x** — the agent reported "stock: 43.3 %
fill, 2.31x". Two independent arms reproduce to 3 significant figures. The
parsing convention (peer-summed `in_bucket_sum_orig`, ÷8, ÷4096) is
therefore the one that was used, and it is self-consistent.

## 4. Exact semantics of the 37.6 % figure

* **Population: in-bucket rank-steps only** — 1,993 of 2,400 rank-steps
  (83 %). Not all steps: over all 300 steps rank 7 gives 3,249,465/300/8 =
  1,354 tokens/rank-step = **33.1 % fill / 3.02x** (the out-of-bucket steps
  run eager, so excluding them is correct for a mega-cost statistic).
* **"per sealed step" is a wording error of ≤1 %.** The figure is
  `in_bucket`-gated, not `sealed`-gated. In this run sealed=247 vs
  in_bucket=249–250, so the two populations coincide numerically — but
  CLAUDE.md / CORPUS_FINDINGS.md wording should say **in-bucket**.
* **It DOES include the DP-dummy / `uniform_rescued` steps** (139–142 of 300
  steps, i.e. **≈56 % of in-bucket steps**), which carry n_orig≈1
  (`min_orig=1`). This is the whole ballgame — see §5.
* **Per rank, per step**, after dividing the peer-summed counter by 8.
* **Whole-run aggregate** (arithmetic mean over steps of real rows; equals
  Σrows/Σsteps, so it is the time-relevant aggregate, not a per-step-ratio
  mean).
* **One arm of one pair of one cell** (M23PAIR2 P1, m15, c32p, 06:59:59
  receipt). No n>1 spread exists for it.
* **Unit: token rows, not top-8-expanded rows.** `n_orig` counts pre-padding
  tokens in the batch; the mega's 4096 is a token-row dispatch. This
  **refutes** G0B §5.4's second reconciliation hypothesis ("the receipt
  counts prompt tokens while the routed rows include per-expert
  replication") — both sides count token rows.

**Residual unverified assumption (the one real caveat):** that
`sum_orig` / `in_bucket_sum_orig` are summed over all 8 DP peers per step
("8x peer-counted"). Nobody in the transcript read the printf in
`m23_patch.py` / `gpu_model_runner.py:4213`. Two strong indirect proofs:
(i) un-divided, the value is 12,313 tokens/rank-step against a hard 4096-row
batch — impossible, while `max_orig=4096` shows the tracked scalar is itself
per-peer-bounded; (ii) under ÷8 the DP-group total real tokens over the
window is 3.07 M against 4.19 M prompt tokens for the cell (1024 × 4096) =
73 %, physically plausible for a 300-step counter window, whereas the
un-divided reading gives 24.6 M = 5.9x the tokens that exist. **Grep the
receipt printf on the node to close this** (already listed as pending work
in REPORT.md §192).

## 5. Reconciliation with the corpus's 1.19x / mean fill 0.69

### 5.1 First: the 1.19x is not what it looks like

`g0b_local_fill.py:89` computes the "aggregate" PRE-M24 multiplier as

```python
(harmonic/agg = {T_MAX*nz.sum()/x[nz].sum():.3f})
```

— i.e. **restricted to calls with `n_real > 0`** (`nz`), dropping the
fully-zero calls from numerator *and* denominator. Back-solving,
1.189 × 1,450,598 / 4096 ⇒ **421 of 512** calls are nz, so **91 calls
(17.8 %) with n_real = 0 were silently excluded** — exactly the calls M24
tier 1 exists to skip.

The honest all-512 aggregate from the same corpus is
`512 × 4096 / (2833.2 × 512)` = **1.446x**, not 1.19x. (The POST column
1.011 and the non-dummy column 1.093 are correct as computed.)
**1.19x should not be quoted at all**; the corpus's own number is 1.446x.

### 5.2 The remaining 1.446x vs 2.657x gap is the dummy-step rate

Both statistics are per-rank-call fill including dummies, in the same unit,
on the same server config and cell. They differ in the **step population
sampled**:

| | dummy share | non-dummy mean fill | overall mean fill | multiplier |
|---|---|---|---|---|
| corpus (first 64 B4096 calls/worker) | 160/512 = **31.2 %** | 0.9146 | 0.6917 | 1.446x |
| receipt (whole 300-step run) | `uniform_rescued`/`in_bucket` = 139/249 … 142/250 = **55.8–56.8 %** | — | 0.3764 | 2.657x |

Candidate definitions, arithmetic done explicitly:

1. **Aggregation form (arithmetic vs harmonic vs per-call mean)** — does not
   reconcile. All the receipt-side and corpus-side aggregates above are the
   same Σrows/Σsteps form. The corpus's per-call ratio means (87.28
   all-calls, 1.378 non-dummy) are wilder, not closer.
2. **Sealed-only vs all-steps** — wrong direction and too small. Widening
   the receipt to all 300 steps makes fill *worse* (33.1 %), not better.
3. **One rank vs all ranks** — no. Ranks 0 and 7 agree to 0.04 %.
4. **Per-expert replication (÷8 on the corpus side)** — refuted, §4: both
   count token rows.
5. **Dummy-share difference, dummies at corpus's measured residual fill
   0.2013**: 0.558 × 0.2013 + 0.442 × 0.9146 = **0.517** (1.94x). Closes
   55 % of the gap. Not sufficient.
6. **Dummy-share difference, dummies at true fill ≈ 0** (`min_orig=1`; the
   corpus's 825-row residual on "dummy" calls is an artifact of the
   >95 %-modal-set rule, not real tokens): 0.442 × 0.9146 = **0.4043**
   → **2.474x**, vs the receipt's 0.3764 / 2.657x. Closes **86 %** of the
   gap; residual is 7 % on the non-dummy fill (0.851 implied vs 0.915
   measured), well inside the first-64-calls capture bias that G0B §5.2
   already flags. **This is the match.**
7. **Solve for the dummy share that makes them identical**, holding the
   corpus's non-dummy fill: d = 1 − 0.3764/0.9146 = **58.9 %**, against the
   receipt's directly measured `uniform_rescued` rate of **55.8–56.8 %** of
   in-bucket steps. Agreement within 2–3 points.

Independent cross-check: the runbook agent's *other*, differently-derived
statistic in the same finding — all-real-tokens / all-padded-rows in b4096
steps = **40.3 %** — is within 0.1 point of definition 6's predicted 40.4 %.

### 5.3 The reconciling statement

> Both are true. The **corpus is right about the shape** of the non-dummy
> distribution (bimodal; non-dummy calls ~0.91 full; row-granular tier-2
> skipping recovers little). The **receipt is right about the rate** of
> zero-fill DP-dummy steps over a whole run: **~56 % of in-bucket steps**,
> not the 31 % seen in the first 64 calls per worker. 0.44 × 0.915 = 0.40 ≈
> the receipt's 0.376. The corpus's window is the ramp-in phase, when every
> rank still has real work; the run-wide rate is nearly double. The residual
> "1.19x vs 2.66x" impression is manufactured by an `nz`-filter bug that
> deletes the zero-fill calls from the corpus aggregate.

## 6. What R4 should now say

R4 should be **replaced, not sustained**:

* The 37.6 % / ~1,539 / 2.66x figure **has a primary source** (transcript
  records 1848/1850, derived from the receipts quoted at records 1518/1376)
  and **reproduces to 3 significant figures**, including an independent
  stock-arm check (43.3 % / 2.31x). "Un-sourced in this session" was a
  provenance failure of the *report*, not of the number.
* It may be quoted **with its qualifiers**: *mean real tokens per rank per
  **in-bucket** B4096 step, whole-run, m15 arm of M23PAIR2 P1 at c32p, n=1
  arm* — and with the ÷8 peer-count assumption flagged pending a printf
  grep. The phrase "per sealed step" should be corrected to "per in-bucket
  step" in CLAUDE.md and CORPUS_FINDINGS.md.
* **The corpus's 1.19x must be withdrawn** as an arithmetic artifact; the
  corpus's own all-call aggregate is 1.446x.
* **There is no contradiction**, so the tier-2 decision no longer hangs on
  one: the corpus's design conclusion survives *and strengthens*. Tier 1
  (skip all-dummy sealed steps) addresses **~56 %** of in-bucket steps, not
  31 % — a larger prize than G0B modeled. Tier 2 (n_orig-proportional row
  plumbing) remains weak: the non-dummy population is ~0.91 full and the
  256-row tile clamp eats the rest.
* Still open, and now the top M24 gates: (a) grep the receipt printf to
  confirm peer-summing; (b) **rank-synchrony of dummy steps** — a step is
  only skippable if all 8 ranks agree, and neither artifact carries a step
  key (G0B §5.3); the receipts' near-identical per-rank `uniform_rescued`
  counts (139/142) are suggestive but not proof.

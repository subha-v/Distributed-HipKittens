# Serving benchmark methodology (fused-MoE megakernel arms)

**Status:** normative as of 2026-08-18. Supersedes all prior serving A/B practice.
**Scope:** end-to-end vLLM **serving** claims for the fused-MoE megakernel arms
(PF4H / m15 / m18 / m19 / m20 and successors). **Out of scope:** kernel-level MoK
harness ratios, GEMM microbenchmarks, and training (Primus/Megatron) tok/s/GPU —
those are unaffected by anything in this document.

Companion documents:

* `../../distributed-kernels/fused_moe/overnight/aug18-prefill/M23_RAGGED_SEAL_DESIGN.md` — the design and the quantified finding.
* `../../distributed-kernels/fused_moe/overnight/aug18-prefill/m23/M23_IMPL_NOTES.md` — the installed patcher, the receipt fields, and the residual asymmetry.

---

## 0. Why this document exists

Every end-to-end serving A/B of the fused megakernel taken before 2026-08-18 is
**obsolete**. Not "noisy", not "superseded by a better kernel" — *invalid*, because
the candidate arm largely did not run the candidate and the baseline arm was itself
depressed. Per-step instrumentation added on 2026-08-18 (`RAGGED_SEAL_RECEIPT`)
measured both defects directly. This document records the evidence chain and fixes
the protocol so it cannot recur silently.

---

## 1. The evidence chain

### 1.1 Defect A — the inert candidate

The megakernel's activation **seal** (`pf4h_exact_b4096`) required **all 8 DP ranks
to be at exactly 4096 pre-padding tokens simultaneously**. That is a *global*
property: DP padding pads every rank up to the group maximum, so a step where seven
ranks carry a 4096-token prefill chunk and the eighth carries a single 1-token
decode batch is padded to `(4096,)*8` — and production happily runs it — but the
seal saw `original_num_tokens_across_dp != (4096,)*8` and refused.

Measured on the real c32p workload: the seal held on **~2% of the ~230 padded-4096
"heavy" steps per rank**. The "m15" serving arms therefore ran **production kernels
for ~98% of the heavy MoE work** they were supposed to be measuring.

### 1.2 Defect B — the de-graphed candidate and the broken baseline

Two independent hazards, both invisible before the receipts:

* **De-graphed candidate.** On an unsealed in-bucket step the PF4H-target server did
  not fall back to the stock B4096 *graph* — it fell back to **no graph at all**
  (`CUDAGraphMode.NONE`), because the PF4H-only server never registers the ordinary
  `regular_b4096` key. So on ~98% of heavy steps the candidate arm ran **eagerly**
  while the baseline arm ran the same steps under PIECEWISE replay. The candidate was
  handicapped *twice*: the megakernel was inert **and** the piecewise graph beneath it
  was disabled.
* **Broken baseline (hazards H1/H2), present in BOTH arms.** Any rank whose local
  batch looked like a uniform decode ran the **whole model eagerly at 4096 padded
  tokens** (H1), and such a rank frequently **cancelled DP padding for the whole
  group** (H2). This depressed the stock arm too, which is why the historical
  baselines look the way they do.

### 1.3 The fix and its measured consequence

**M23 "ragged seal" + "uniform-decode rescue"** (implemented and validated
2026-08-18; no HIP change, no re-capture, no ABI change):

* the seal now accepts **the same padded-4096 batches production already runs** —
  measured coverage **99% of in-bucket steps, ~95% of tokens**;
* the **uniform-decode rescue** routes H1 ranks to the graph in **both** arms, so the
  baseline is no longer depressed and the two arms are symmetric.

Measured consequence on the identical c32p cell:

| arm | input tok/s | note |
|---|---:|---|
| historical stock baselines (pre-M23) | 9,600–12,700 | depressed by H1/H2; the band every obsolete delta was taken against |
| **rescued stock (production-configured, M23 chain)** | **20,918** | n=1; the only valid baseline |
| **m15 + M23 (first honest megakernel number)** | **19,277** | n=1; ~8% below rescued stock, **cross-pair** |

The stock baseline itself roughly **doubled**. Every pre-2026-08-18 serving delta was
therefore taken against a broken control with a doubly-handicapped candidate, in both
directions at once.

### 1.4 What this invalidates, and what it does not

**Obsolete (serving-level):** the PF4H c32/c512/c1024 pairs; m15 −0.70%; m18 −5.7%
and +11.3%; m19 +39.8%; m20 −19.6%; the balanced-prompt "flat" pair. These claims were
**removed from the repos' markdown on 2026-08-18** rather than restated, so that future
readers do not grep them back up as live results; wholly-obsolete result docs were
replaced by short tombstones and mixed docs had their serving sections deleted in place.
`git log --follow` preserves every original. The raw JSON/artifact files were left
untouched — they remain valid records of what the machine did; only their
*interpretation as an arm comparison* is void.

**Unaffected (kernel-level):** MoK harness ratios (0.756× balanced, 0.8467×
skew-replay, the M18/M19/M20 replication ratios), GEMM/collective microbenchmarks, and
all training numbers. These never went through the seal.

---

## 2. The protocol (normative)

A serving number that does not satisfy **all** of (a)–(g) is not quotable.

### (a) Both arms carry the M23 patch chain

```
apply.py  →  coverage_patch.py  →  m23_patch.py  →  vllm serve
```

Order is load-bearing (`m23_patch.py` exits 1 if `PF4H_COVERAGE_PATCH_V1` is absent
from a PF4H-patched `gpu_model_runner.py`). The chain must be identical in the
candidate arm **and** the stock arm — the rescue is what makes the baseline honest, so
an unpatched stock arm is not a valid control.

### (b) Every megakernel claim quotes its coverage receipt

Any serving claim for a megakernel arm **must** quote that arm's
`RAGGED_SEAL_RECEIPT` coverage as `sealed / in_bucket`. **A mega number without a
coverage receipt is invalid** — that is precisely the omission that let the historical
arms report a kernel they were not running.

### (c) The only valid baseline is rescued-stock

Production-configured stock, uniform-decode rescue active, patch chain identical to
the candidate's (arm-symmetric). Comparisons against a pre-M23 stock number, or
against a stock arm without the rescue, are void by construction.

### (d) Drift budget: ≥5 order-balanced pairs

Stock run-to-run drift is **±15% at n=1** on this cell. A quotable delta therefore
needs **≥5 order-balanced pairs** (alternate which arm runs first; report the pair
distribution, not just the mean). Single pairs may be reported only as explicitly
labelled n=1 observations, never as a delta claim.

### (e) Exact-token SHA identity across arms

Both arms must produce byte-identical generated token ids (`temperature=0`, same
seed), verified by SHA over the concatenated token-id stream. A divergence must be
documented with a first-divergence distribution before any performance number from
that pair is quoted.

### (f) Full workload spec attached to every number

At minimum: **cell** (e.g. c32p), **concurrency**, **prompt count and source**
(e.g. 1024 MLPerf QSL prompts), **ISL/OSL** (e.g. 4096 / 8), and the **metric
definition** (e.g. aggregate node input tok/s vs per-rank; total vs input vs output).
A bare "tok/s" is not a number.

### (g) Report the known residual asymmetry

M23 does not make the arms perfectly symmetric. On steps with **any idle (dummy)
rank**, the candidate server runs **all-8-eager** while stock runs **7-graph +
1-eager**. This is measured by `refused_peer_not_ready` / `eager_b4096` and runs at
**~3% of steps at c32p, worse at low concurrency**. Report it alongside the headline;
it biases *against* the candidate.

---

## 3. Receipt vocabulary

`RAGGED_SEAL_RECEIPT` is emitted per rank, cumulative, every N steps and once at
shutdown:

```
RAGGED_SEAL_RECEIPT rank=%d steps=%d in_bucket=%d sealed=%d sealed_ragged=%d
  sealed_exact=%d refused_not_unanimous=%d refused_not_ready=%d eager_b4096=%d
  min_orig=%d max_orig=%d sum_orig=%d
```

| field | meaning | healthy value |
|---|---|---|
| `steps` | every `_determine_batch_execution_and_padding` call from `execute_model` | — |
| `in_bucket` | `b4096_unanimous` true — **the coverage denominator** | ~230/rank/c32p run |
| `sealed` | `pf4h_ragged_seal` true — **the coverage numerator** | `== in_bucket` |
| `sealed_exact` | sealed **and** `original_counts == (4096,)*8` (the old, pre-M23 population) | ≈2% of `sealed` |
| `sealed_ragged` | `sealed − sealed_exact` — the population M23 creates | ≈98% of `sealed` |
| `refused_not_unanimous` | locally 4096 but the group disagreed (**hazard H2**) | ~0 |
| `refused_not_ready` | group in-bucket but activation not latched somewhere | 0 after warmup |
| `refused_peer_not_ready` | this rank was ready, a **peer** was not (unlatched activation *or* a dummy batch) — the (g) asymmetry counter | ~3% of steps at c32p |
| `eager_b4096` | in-bucket steps that ended at `CUDAGraphMode.NONE` — **the Defect-B counter** | **0** |
| `min_orig` / `max_orig` / `sum_orig` | over `original_num_tokens_across_dp`; skew and token-weight analysis | — |

**Success criterion for a valid candidate run:** `sealed == in_bucket` and
`eager_b4096 == 0` on every rank.

Related receipts that must also be reported when non-zero: `in_bucket_dummy_peer`,
`eager_b4096_dummy_peer` (the dummy-rank class from (g)).

---

## 4. Worked example of a compliant quote

> **m15 + M23 vs rescued stock, c32p, 2026-08-18.**
> Workload: c32p cell, concurrency 32, 1024 MLPerf QSL prompts, ISL 4096, OSL 8;
> metric = aggregate node **input** tok/s over the full run.
> Both arms: `apply.py → coverage_patch.py → m23_patch.py`; stock arm = production
> configuration with the uniform-decode rescue active (arm-symmetric).
> Coverage receipt, candidate arm: **sealed/in_bucket = 228/230 (99.1%)**,
> `sealed_ragged` 224, `sealed_exact` 4, `eager_b4096` **0**, `refused_not_unanimous` 0,
> token-weighted coverage ~95%.
> Accuracy: generated token-id SHA identical across arms (`temperature=0`, same seed).
> Result: **19,277 tok/s candidate vs 20,918 tok/s rescued stock**, −7.8%.
> **n=1 each, cross-pair — NOT a quotable delta** (protocol (d) requires ≥5
> order-balanced pairs against ±15% stock drift). Reported as two reference points.
> Residual asymmetry (g): ~3% of steps had an idle rank, on which the candidate ran
> all-8-eager while stock ran 7-graph + 1-eager; `refused_peer_not_ready` non-zero.
> This biases against the candidate and is not corrected in the number above.

Note what the example does **not** do: it does not turn 19,277 vs 20,918 into a
headline "−8% regression". Two n=1 runs from different pairs, against ±15% drift, are
reference points. The delta claim waits for the five pairs.

---

## 5. Checklist (paste into any serving result doc)

```
[ ] (a) both arms: apply.py → coverage_patch.py → m23_patch.py, identical chain
[ ] (b) RAGGED_SEAL_RECEIPT quoted: sealed/in_bucket = ___/___  (___%)
[ ] (b) eager_b4096 == 0 on every rank
[ ] (c) baseline is rescued-stock, production-configured, arm-symmetric
[ ] (d) ≥5 order-balanced pairs (or number explicitly labelled n=1 reference point)
[ ] (e) generated token-id SHA identical across arms
[ ] (f) cell / concurrency / prompts / ISL / OSL / metric definition all stated
[ ] (g) refused_peer_not_ready + eager_b4096 dummy-rank asymmetry reported
```

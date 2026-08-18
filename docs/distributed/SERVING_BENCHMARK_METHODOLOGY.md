# Serving benchmark methodology (fused-MoE megakernel arms)

**Status:** normative as of 2026-08-18 (rev 3, post-camp3, incorporating the
operator-approved binding protocol). Supersedes all prior serving A/B practice.
Rev 3 folds in `../../nightshift/BENCHMARK_PROTOCOL.md` — the operator's binding
benchmark protocol — which **wins wherever rev 2 conflicted with it** (notably the
minimum pair count, now ≥5). Nothing in rev 2 was weakened.
**Scope:** end-to-end vLLM **serving** claims for the fused-MoE megakernel arms
(PF4H / m15 / m18 / m19 / m20 and successors). **Out of scope:** kernel-level MoK
harness ratios, GEMM microbenchmarks, and training (Primus/Megatron) tok/s/GPU —
those are unaffected by anything in this document.

Companion documents:

* `../../nightshift/BENCHMARK_PROTOCOL.md` — **the source of the binding protocol**
  (operator-approved, 2026-08-18): strongest-honest-baseline, production-like
  workload, separation of claim from measurement. This document is its normative
  expansion; where the two are read together, the protocol governs.
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
| **m15 + M23 (first honest megakernel number)** | **19,277** | n=1; ~8% below rescued stock, **cross-pair** — superseded, see below |

Superseded by the first *order-balanced* result (camp3, 2026-08-18, cell c32p,
n=2 pairs, arm order reversed on the even pair, 99% seal coverage):
**m15 19,277 / 17,900 tok/s vs patched-stock 15,611 / 20,589 tok/s → m15
+2.7 % to +5.2 % within-pair against patched-stock.** Note what that ratio is:
it is the *kernel-effect* leg of §2.5, **not** a claim against genuine
production. The two arms' raw numbers also show the ±18 % position effect
directly — which is why single arms are never evidence.
**Rev-3 status of that result: DIRECTIONAL ONLY.** At n=2 it is below the ≥5-pair
floor of §(d) (mean +2.70 %, geomean +3.61 %, spread +23.5 % / −13.1 %), and its
token-stream SHA parity is unresolved (§(e)). It may not be quoted as a headline.

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

A serving number that does not satisfy **all** of (a)–(k) is not quotable.

### (a) Both arms carry the M23 patch chain

```
apply.py  →  coverage_patch.py  →  m23_patch.py  →  vllm serve
```

Order is load-bearing (`m23_patch.py` exits 1 if `PF4H_COVERAGE_PATCH_V1` is absent
from a PF4H-patched `gpu_model_runner.py`). The chain must be identical in the
candidate arm **and** the stock arm — the rescue is what makes the baseline honest, so
an unpatched stock arm is not a valid *patched-stock* control.

Scope note: (a) governs the candidate arm and the **patched-stock** diagnostic arm.
The **native** arm of (c) carries no chain at all — that is its definition — and is
therefore exempt from (a) and from the coverage requirement (b), which cannot apply
to a server that has no megakernel.

### (b) Every megakernel claim quotes its coverage receipt

Any serving claim for a megakernel arm **must** quote that arm's
`RAGGED_SEAL_RECEIPT` coverage as `sealed / in_bucket`. **A mega number without a
coverage receipt is invalid** — that is precisely the omission that let the historical
arms report a kernel they were not running.

### (c) The headline baseline is GENUINE NATIVE production — in **two** flavours

There are **three** distinct baseline arms and they are **not** interchangeable.
Two of them are genuine production (`native-default`, `native-tuned`); the third
(`patched-stock`) is a diagnostic control only.

**native-default — the untouched vendor image at shipped defaults.** What a user
gets out of the box. Defined by absence, and verified from evidence rather than
intent:

* the **untouched `vllm/vllm-openai-rocm:v0.25.1` image**, pulled digest recorded
  (expected `sha256:84459732ca98b40fe2f5338a3f050be6d522504e47a484a5180d58fb75956f86`);
* **shipped defaults** — the vendor's recommended DeepSeek-R1 deployment for this
  node (TP1/DP8/EP8, AITER + MORI enabled as the image ships them), same
  `--gpu-memory-utilization` and scheduler flags as our arm;
* **no `VLLM_PF4H_*` environment variables** anywhere in `docker inspect` or the
  server's config dump;
* **no patch chain at all** — none of `apply.py`, `coverage_patch.py`,
  `m23_patch.py`, `rr_patch.py`, and none of their markers
  (`PF4H_INTEGRATION_PATCH_V3_M15`, `PF4H_COVERAGE_PATCH_V1`,
  `PF4H_M23_RAGGED_SEAL_V1`, `PF4H_RR_*`) present in the running tree.

Wrapper v4's `native` arm implements exactly this. Anything else is not production.

**native-tuned — the same untouched image at AMD's *published recommended*
deployment settings** for DeepSeek-R1 on MI350X (the vendor's MLPerf / deployment-
blog configuration). Same image, same absence of every patch and every
`VLLM_PF4H_*` variable; only the documented flags/env differ, and each deviation
from `native-default` must be cited to the vendor document it comes from.

**Fairness cuts both ways: beating a lazy default is not beating production.**
If the vendor documents a faster configuration, *that* is the bar. A headline
therefore reports the ratio against **both** native arms, and the weaker of the two
claims is the one that may be stated without qualification. If `native-tuned` has
not been run, say so explicitly — a headline quoted only against `native-default`
is provisional, not final.

**Patched-stock — a diagnostic control only.** Production ops running *inside our
integration container*: full patch chain, uniform-decode rescue active, stock
B4096 graph, our runtime footprint. It isolates the kernel substitution, but it
inherits every change our integration makes, so it **may never be quoted as
"production"**. Doing so was one of this project's actual past failures.

Comparisons against a pre-M23 stock number, or against a stock arm without the
uniform-decode rescue, are void by construction.

### (c2) Report the three-way decomposition

Whenever all three arms are available, quote all three ratios — a single ratio
hides which effect is being claimed:

| ratio | name | what it measures |
|---|---|---|
| **m15 / native-default** | **the headline** | the only number that answers "is this faster than production?" |
| **m15 / native-tuned** | **the hard headline** | is this faster than production *at the vendor's own best documented config*? |
| m15 / patched-stock | kernel effect | the megakernel substitution alone, integration held constant |
| patched-stock / native | integration effect | what our container/patches cost or gain independent of the kernel |

The headline is not derivable from the other two "well enough"; measure it.

### (c3) Symmetry rule for improvements we discovered

Any speedup **we** found that is not the megakernel must never hide inside the
kernel's number. The uniform-decode rescue is the worked case: it roughly
**doubled** throughput, and it is *our patch*, not the megakernel. Two admissible
treatments, no third:

1. **Give it to the baseline too** — which is exactly what the patched-stock arm
   does, and why the three-way decomposition exists; or
2. **Count it explicitly as "integration effect"** on the `patched-stock / native`
   leg, named as engineering that either side could adopt (it is upstreamable).

Therefore, invariably:

* **the kernel claim is only ever `m15 / patched-stock`**;
* **the headline is only ever `m15 / native`** (both flavours per §(c));
* **the decomposition is always shown**, so a reviewer can see which part is the
  kernel and which part is integration work production could adopt tomorrow.

Quoting an integration win as a kernel win — or letting the rescue's ~2× ride
inside a "megakernel is faster" sentence — is a protocol violation, not a rounding
choice.

### (d) Drift budget, order balance, and run hygiene

* Measured drift: **±15% day-to-day** at n=1 and **±18% per arm from position**
  (which arm runs first inside a pair). **Single arms are never evidence, and
  cross-pair ratios are never evidence.**
* **Order-balanced pairs only, minimum n = 5.** Both arms inside one pair;
  reverse arm order on even-numbered pairs. **≥5 order-balanced pairs is a hard
  floor for any quoted delta** (BENCHMARK_PROTOCOL §3.1); this *supersedes* rev 2's
  softer "smaller n may be reported with the spread shown". Anything below 5 pairs
  is **directional only** and may never be a headline — camp3's n=2 result
  (+2.7–5.2 % within-pair; +2.70 % mean, +3.61 % geomean, spread +23.5 % / −13.1 %)
  is explicitly directional under this protocol, not a claim.
* Report `n`, every pair's ratio, the **median of pairs**, and the spread across
  pairs — never only the mean, never the mean alone.
* **`ARM_COOLDOWN=240`** (seconds) between arms, symmetric — the position effect
  is largely thermal/cache. Wrapper v3/v4 set this; a run without it is not
  order-balanced in any meaningful sense.
* **Fresh server per arm.** No arm inherits another's prefix cache, allocator
  state, or captured graphs. One B4096 graph per server (memory headroom).
* Warmup, seeds, client, and cell spec identical across arms.

### (e) Accuracy gate — a fast wrong kernel is a zero

Every quoted number passes one of two gates, stated explicitly:

* **Exact-token identity.** Both arms produce byte-identical generated token ids
  (`temperature=0`, same seed), verified by SHA over the concatenated token-id
  stream (`ordered_output_token_id_stream_sha256`). A divergence must be documented
  with a first-divergence distribution before any performance number from that pair
  is quoted.
* **Bounded-error accuracy A/B**, where the kernels legitimately differ numerically:
  an **MLPerf AccuracyOnly A/B** across arms, or the MoK bounded relative-error
  policy, with the bound and the measured value stated.

**KNOWN OPEN PROBLEM — read before relying on the SHA check.** In camp3,
`ordered_output_token_id_stream_sha256` **differed even stock-vs-stock**: the
sampling path is nondeterministic, so the SHA is not currently a valid parity
oracle on its own. Consequences, binding:

* a campaign intending to use the SHA gate must **force deterministic decoding**
  (greedy, fixed seed, sampler nondeterminism eliminated) and must first
  demonstrate SHA identity **stock-vs-stock** as the control; otherwise
* route accuracy through the MLPerf AccuracyOnly / bounded-rel-err path instead.

**Unresolved parity = no claim.** A performance delta whose accuracy gate neither
passed nor was replaced by a valid alternative is not quotable at any n.

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

### (h) Workload realism: real prompts, and open-loop arrival

* **Real text for every headline.** Prompts come from the **MLPerf QSL** corpus so
  routing carries genuine expert skew. Synthetic or randomly generated token
  streams are never acceptable for a headline number; they may be used only for
  labeled kernel-level or debug rigs.
* **Open-loop request-rate cells are mandatory alongside the saturated closed-loop
  cell.** Production serving is arrival-driven, not backlog-driven. Run a
  **Poisson-arrival request-rate sweep at 50 / 75 / 90 % of measured saturation
  throughput** in addition to the saturated closed-loop cell (c32p and friends).
  * *Why this is not optional:* corpus analysis showed **~31 % of steps in the
    closed-loop c32p cell are dummy batches** — a substantial share of that is an
    artifact of concurrency 32 with OSL 8, where ranks idle at the tail of every
    wave. Any megakernel advantage that comes from skipping pad rows or dummy
    batches is therefore *partly measuring the benchmark's own shape*.
  * The open-loop sweep is what decides the question: it measures peak throughput
    **and** behaviour at realistic utilization, and thereby separates
    **production value** from **benchmark artifact**.
  * **Report pad/dummy-skip wins per regime** — closed-loop saturated vs each
    open-loop utilization cell — so the artifact share is visible to a reader
    rather than averaged away.

### (i) A grid, not a point — including the cells where we lose

* Sweep **several ISLs and concurrencies** (c8 / c16 / c32 at ISL 1024, c32p and
  c512p at ISL 4096, plus the open-loop rates of (h)). The deliverable is the
  **per-cell table**, not a selected cell.
* **Losing cells are published.** The mega inverts below roughly
  **1,700 tokens/rank** (measured band ~1,600–1,800); cells beneath that break-even
  are expected losses and are reported with the same prominence as the wins.
  Omitting them is cherry-picking.
* The table carries one additional **"deployment policy" row**: the hybrid an
  operator would actually ship — megakernel in the cells where it wins, stock
  elsewhere — evaluated across the whole grid. That row is the honest aggregate
  claim, and it is the one number no cherry-picked cell can fake.

### (j) Metrics and aggregation

Every quoted result reports, at minimum:

* **aggregate input tok/s** (prefill throughput, node-aggregate, defined as such);
* **TTFT p50 and p99** — what users actually feel, and the metric open-loop cells
  exist to expose;
* **tok/s/GPU**;
* the **complete workload spec** of §(f) attached to each number;
* **median of pairs with the spread shown**, and the **claim sized against measured
  drift** (±15 % day-to-day, ±18 % position). A delta inside the drift band is
  reported as "within drift", not as a win.

### (k) Costs priced into the same table as the speedup

A speedup bought with resources is not free, and the cost may not live in a
footnote. The result table carries, per arm: **memory footprint**, **KV-cache
impact** (cache size / max concurrency achievable at the same
`--gpu-memory-utilization`), and **any expert replication** cost. Past sin: 41 GB
of replica caches framed as a kernel win.

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

*(This example predates the native-baseline rule of §(c): it quotes only the
`m15 / patched-stock` **kernel-effect** leg, and labels it as such. A compliant
headline additionally carries the `m15 / native` ratio and the §5 sign-off.)*

> **m15 + M23 vs rescued stock, c32p, 2026-08-18.**
> Workload: c32p cell, concurrency 32, 1024 MLPerf QSL prompts, ISL 4096, OSL 8;
> metric = aggregate node **input** tok/s over the full run.
> Both arms: `apply.py → coverage_patch.py → m23_patch.py`; stock arm = production
> configuration with the uniform-decode rescue active (arm-symmetric).
> Coverage receipt, candidate arm: **sealed/in_bucket = 228/230 (99.1%)**,
> `sealed_ragged` 224, `sealed_exact` 4, `eager_b4096` **0**, `refused_not_unanimous` 0,
> token-weighted coverage ~95%.
> Accuracy: generated token-id SHA identical across arms (`temperature=0`, same seed).
> Result (kernel-effect leg only): **19,277 tok/s candidate vs 20,918 tok/s
> rescued stock**, −7.8%.
> **n=1 each, cross-pair — NOT a quotable delta** (protocol (d) requires ≥5
> order-balanced pairs against ±15% stock drift). Reported as two reference points.
> Residual asymmetry (g): ~3% of steps had an idle rank, on which the candidate ran
> all-8-eager while stock ran 7-graph + 1-eager; `refused_peer_not_ready` non-zero.
> This biases against the candidate and is not corrected in the number above.

Note what the example does **not** do: it does not turn 19,277 vs 20,918 into a
headline "−8% regression". Two n=1 runs from different pairs, against ±15% drift, are
reference points. The delta claim waits for the five pairs.

---

## 5. The fairness audit (mandatory before any "beats production" claim)

Protocol compliance is necessary but not sufficient. **No claim of the form "our
kernel is better than production" ships without a written sign-off from a
dedicated adversarial fairness auditor** — an independent reviewer (in the
overnight loop, a separate Opus subagent at high effort) prompted to *prove the
comparison unfair*, re-run for **every new kernel build and every campaign** whose
numbers are to be quoted. It holds veto power. Its checklist, and the actual past
failure each item exists to prevent:

1. **Baseline authenticity.** The production arm is the genuinely untouched image
   at shipped defaults — verified from *evidence*, not intent: image digest,
   `docker inspect` env (no `VLLM_PF4H_*`), the server.log config dump, and the
   absence of every patch marker (`PF4H_INTEGRATION_PATCH_V3_M15`,
   `PF4H_COVERAGE_PATCH_V1`, `PF4H_M23_RAGGED_SEAL_V1`, `PF4H_RR_*`).
   *Past sin: quoting patched-stock as "production".*
2. **Baseline not sandbagged, and not merely the lazy default.** Production gets
   its best shipped config — AITER + MORI env present, same gpu-mem-util and
   scheduler flags as our arm; any deviation from the vendor's recommended
   deployment documented and justified. Per §(c), the auditor also checks whether
   a **native-tuned** arm (AMD's published recommended settings) was run, and
   treats a headline quoted only against `native-default` as provisional.
   *Past sin: none yet — keep it that way.*
2b. **Symmetry of our own discoveries.** No improvement we invented (the
   uniform-decode rescue above all) is allowed to sit inside the kernel number:
   kernel claim = `m15 / patched-stock`, headline = `m15 / native`, decomposition
   shown (§(c3)).
3. **Candidate actually ran.** Coverage receipts (`RAGGED_SEAL_RECEIPT
   sealed/in_bucket`) prove the megakernel executed the traffic.
   *Past sin: the mega was inert on ~98 % of heavy steps in every pre-M23 A/B,
   while the candidate arm additionally ran de-graphed — a doubly fake candidate.*
4. **Identical workload.** Exact-token prompt SHA match across arms; same cell
   spec, seeds, client, warmup; prefix-cache and thermal asymmetries neutralized
   with symmetric cooldowns. *Past sin: prewarm/cache asymmetry; ±18 % position.*
4b. **Workload realism.** Real MLPerf QSL prompts, not synthetic tokens; open-loop
   Poisson cells present alongside the saturated closed-loop cell, with dummy-batch
   share reported per regime (§(h)); grid published including losing cells and the
   deployment-policy row (§(i)).
5. **Statistical validity.** Order-balanced pairs, **n ≥ 5**; `n` stated; the spread
   across pairs shown; the claim sized against measured drift (±15 % day,
   ±18 % position). Single arms and cross-pair ratios are never evidence.
   *Past sin: n=1 headlines (+39.8 %, −19.6 %), later voided.*
6. **Accuracy parity.** Outputs validated — exact-token SHA under *forced
   deterministic decoding* with a passing stock-vs-stock control, otherwise MLPerf
   AccuracyOnly A/B or the bounded rel-err policy. Unresolved parity = no claim
   (§(e)). A faster wrong kernel is not a win. *Past sins: `SAME OUTPUTS: False`
   left unresolved; camp3's token-stream SHA differing stock-vs-stock from sampling
   nondeterminism — still open.*
7. **Replay/proxy honesty.** Kernel-level rigs (MoK, histogram replay) are never
   quoted as end-to-end; captured-route and **fill** regimes are labeled — banked
   kernel numbers are 100 %-fill, serving runs at ~37.6 % fill (2.66× padding
   multiplier). *Past sin: i.i.d. replay standing in for real routing.*
8. **Claim wording matches measurement.** Which cells, which regime; the hybrid
   caveat stated (the mega inverts below ~1,600–1,800 tokens/rank, so small-batch
   cells are expected losses — report them, never hide them); memory and
   replication costs priced in. *Past sin: 41 GB replica caches framed as a
   kernel win.*

Any result doc quoting a headline carries a `FAIRNESS_AUDIT` section per claim:
verdict, items checked with evidence pointers, residual caveats.

---

## 6. Checklist (paste into any serving result doc)

```
[ ] (a) candidate + patched-stock: apply.py → coverage_patch.py → m23_patch.py,
        identical chain; native arm carries no chain and no PF4H env
[ ] (b) RAGGED_SEAL_RECEIPT quoted: sealed/in_bucket = ___/___  (___%)
[ ] (b) eager_b4096 == 0 on every rank
[ ] (c) headline baseline is GENUINE NATIVE (untouched v0.25.1 image, digest
        recorded, no PF4H env, no patch markers) — evidence pointers recorded
[ ] (c) BOTH native flavours run: native-default AND native-tuned (AMD's published
        recommended config, deviations cited); if native-tuned missing, headline
        labeled PROVISIONAL
[ ] (c2) decomposition reported: m15/native-default · m15/native-tuned ·
        m15/patched-stock · patched-stock/native
[ ] (c3) symmetry rule honored: uniform-decode rescue (and any other discovery of
        ours) counted as integration effect, NOT inside the kernel number;
        kernel claim = m15/patched-stock only
[ ] (d) order-balanced pairs, n = ___ (>= 5 REQUIRED), per-pair ratios + median +
        spread shown, ARM_COOLDOWN=240, fresh server per arm; no single-arm or
        cross-pair ratio quoted; n<5 labeled DIRECTIONAL ONLY
[ ] (e) accuracy gate passed: deterministic decoding forced AND stock-vs-stock SHA
        control passes, OR MLPerf AccuracyOnly / bounded rel-err A/B reported.
        KNOWN OPEN: camp3 SHA differed stock-vs-stock. Unresolved parity = no claim
[ ] (f) cell / concurrency / prompts / ISL / OSL / metric definition all stated
[ ] (g) refused_peer_not_ready + eager_b4096 dummy-rank asymmetry reported
[ ] (h) real MLPerf QSL prompts; open-loop Poisson cells at 50/75/90% of saturation
        run alongside the saturated closed-loop cell; dummy-batch share (~31% at
        c32p closed-loop) reported per regime
[ ] (i) full grid published incl. losing cells below the ~1,700 tok/rank
        break-even, plus the "deployment policy" hybrid row
[ ] (j) metrics: aggregate input tok/s, TTFT p50/p99, tok/s/GPU; median-of-pairs
        with spread; claim sized against ±15% day / ±18% position drift
[ ] (k) costs in the SAME table as the speedup: memory footprint, KV-cache impact,
        replication
[ ] (l) fill regime labeled (serving ~37.6% fill vs 100%-fill kernel rigs)
[ ] (m) FAIRNESS_AUDIT sign-off attached (§5) — required for any "beats
        production" claim
```

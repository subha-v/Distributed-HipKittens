# METRICS_PROTOCOL — the metric map and the reporting discipline

**Written:** 2026-08-18, laptop, no node. **Status:** design output #4 of the ablation
campaign (`../BRIEF.md` §"Required outputs" item 4). **Binding scope:** every number this
campaign produces, kernel-level or end-to-end.

**Relationship to existing protocol (read this first).** This document **amends**
`nightshift/BENCHMARK_PROTOCOL.md` and `docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md`;
it does **not** fork them. Everything in those two documents remains binding. §6 lists,
line by line, what this file *extends*, what it *amends*, and the one place it *contradicts*
`BENCHMARK_PROTOCOL.md:61-64` — with the exact replacement text to paste. Where this file
and those two disagree and §6 does not name the disagreement, **those two win**.

**Evidence tags used throughout**, same convention as the grounding briefs:
**MEASURED** (in-repo artifact, cited), **DERIVED** (arithmetic in *this* file over cited
in-repo artifacts, arithmetic shown), **ASSUMED**, **SPECULATION**. No number appears
without one.

---

## 0. The complaint, and the one-sentence answer

The expert feedback, verbatim (`../BRIEF.md:31-33`):

> "People keep posting TPS without batch size or concurrency, input and output lengths, or
> even saying whether it means per-request decode speed or aggregate throughput. On its own,
> the number means almost nothing."

**We can quantify exactly how much "almost nothing" is, from our own banked artifacts.**
DERIVED (this file) from `distributed-kernels/tp8_mega/results/serving/b3rev2_pair_01_2_m15.json`:
the same run, same arm, same second, reports

* aggregate output throughput **83.663 tok/s**, and
* per-request decode speed **1.321 tok/s** (`1000 / tpot_ms.p50 = 1000/756.75`),

a factor of **63.3×** between two quantities a reader would both call "tok/s". At c32p
(`b0v5_pair_01_1_m15.json`) the same ratio is **22.2×**. The word "TPS" is therefore not
imprecise — it is ambiguous by a factor that changes with concurrency.

**The answer this protocol gives** has four parts, one per required section:

1. **A metric map (§2)**: each workload class has exactly one PRIMARY metric, a fixed set of
   GUARD metrics that can veto, and a named mediating quantity. The primary is chosen by the
   *deployment decision the cell corresponds to*, not by which number flatters us.
2. **A workload tuple (§3)**: a mandatory header that must be attached to every reported
   number, including the coverage counters that prove the kernel ran and the composition
   counters that prove the two arms did the same amount of work.
3. **A two-instrument doctrine (§4)**: boundary rigs rank and falsify; e2e sizes and
   validates; no schedule claim ships with only one of them.
4. **A law card (§5)**: the publishable unit — statement, mediating quantity, Q-inequality,
   θ, predicted vs measured, cells tested, failure scope. This is the concrete form of the
   expert's "list of what we'd like to teach people about" (`../BRIEF.md:23`).

---

## 1. Metric algebra — exact definitions

### 1.1 The estimand hierarchy

The project already has a primary methodological commitment, and this section is subordinate
to it. LAW-64 (`grounding/BANKED_LAWS.md`, from `aug12/LESSONS.md:5-11`):

> the estimand is the **synchronized global makespan of the complete dependent DAG**;
> bandwidth wins that do not reduce it are transport wins, not overlap wins.

Two consequences that bind every metric below:

* **Rank aggregation is MAX, not rank-0.** The adopted benchmark statistic for the training
  program is "median over iterations of the MAX-across-ranks latency"
  (`aug18/T3_COUNTER_DATAFLOW_DESIGN.md:39-40`). Serving inherits it: a per-rank quantity
  (seal receipts, fill, device stamps) is reported as the **rank-max** with the rank-0 value
  shown separately whenever both exist. LAW-63(d) measured the size of getting this wrong:
  production's genuine cross-rank combine reads **1,356.0 µs rank-max vs 978.4 µs rank-0
  (+38%)** — "the two instruments are not comparable tick-for-tick."
* **Client-side wall clock is the only e2e estimand.** Device stamps attribute; they never
  headline. LAW-63(e): never subtract `clock64()` values written by separate kernels — one
  such span read **55,576.5 µs against a 1,278.81 µs HIP event** (`aug12/LESSONS.md:72-76`).

### 1.2 Aggregate vs per-request — the definitions that must never be conflated

Let a cell deliver `R` completed-OK requests over a steady-state window of wall time `T_w`
seconds (window defined in §1.8), with per-request input length `ISL_i` and output length
`OSL_i`.

| # | name | formula | unit | what it measures | verified against manifest |
|---|---|---|---|---|---|
| **M1** | aggregate input tok/s (node) | `Σ_i ISL_i / T_w` | tok/s/node | node **prefill capacity** | `input_tokens_total / wall_seconds` reproduces `result.input_tokens_per_second` to 6 s.f. in **all 6** banked manifests (DERIVED) |
| **M2** | aggregate output tok/s (node) | `Σ_i OSL_i / T_w` | tok/s/node | node **decode capacity** — *see the OSL=8 degeneracy below* | `= request_throughput × OSL` **exactly** in all 6 manifests (DERIVED) |
| **M3** | aggregate total tok/s | `M1 + M2` | tok/s/node | mixed capacity; never a headline on its own | `total_tokens_per_second` (DERIVED, checks) |
| **M4** | request throughput | `R / T_w` | req/s | goodput denominator | `result.request_throughput` |
| **M5** | per-request prefill rate | `ISL_i / (TTFT_i/1000)` | tok/s/req | what a *single user* sees during prefill | DERIVED |
| **M6** | per-request decode rate | `1000 / TPOT_i` | tok/s/req | what a single user sees during decode — **this is the "TPS" a user quotes** | DERIVED |
| **M7** | per-GPU throughput | `M1 / n_GPU` (state which of M1/M2/M3) | tok/s/GPU | cross-hardware comparison | required by `SERVING_BENCHMARK_METHODOLOGY.md:325-337` |

**Binding rules.**

* **R-M1.** Any "tok/s" number is written as `M<n>` or spelled out in full
  (`aggregate input tok/s, node`). The bare string "tok/s", "TPS" or "throughput" is a
  **protocol violation**, not a style issue.
* **R-M2 (the OSL degeneracy).** DERIVED, verified in all six banked manifests: `M2 ≡ M4 × OSL`
  exactly. At `OSL = 8` and `ignore_eos=true, temperature=0.0` every request emits exactly 8
  tokens (`per_query[i].itl_ms_count = 7`), so **M2 carries zero decode-speed information —
  it is request throughput in disguise.** M2 may not be reported as a decode result in any
  cell with fixed OSL. Decode speed is M6, always.
* **R-M3.** M1 and M6 must appear **together** in any headline sentence, or neither. The
  measured gap between them (63.3× at c512p, 22.2× at c32p, DERIVED) is exactly the
  ambiguity the experts flagged.
* **R-M4.** M5 is **not** a kernel property under a closed-loop driver — see §1.3.

Worked demonstration, DERIVED (this file) from the banked manifests, arithmetic reproducible
with the snippet in §3.6:

| arm / cell | M1 (in tok/s) | M2 (out tok/s) | M5 (tok/s/req, prefill) | M6 (tok/s/req, decode) | M2/M6 |
|---|---:|---:|---:|---:|---:|
| m15, c32p (`b0v5`) | 20,371.6 | 39.79 | 1,675.4 | 1.794 | 22.2× |
| native_tuned_tp, c32p | 25,844.0 | 50.48 | 3,059.9 | 2.083 | 24.2× |
| m15, c512p (`b3rev2`) | 42,835.7 | 83.66 | **94.9** | 1.321 | 63.3× |
| native_tuned_dp, c512p | 40,183.1 | 78.48 | 141.8 | **0.312** | 251.7× |

Note the row that makes the point hardest: **the same m15 kernel shows M5 = 1,675 tok/s/req
at c32p and 94.9 tok/s/req at c512p — 17.7× apart** — while M1 goes the *other* way
(20.4k → 42.8k). Any headline built on one of those two numbers carries the opposite sign
from the other.

### 1.3 Latency metrics, and the two traps in them

**Definitions (client-side, as implemented by `bench_exact_token_ids_v3.py`, schema
`pf4h-exact-token-closed-concurrency-v2`):**

* `TTFT_i` — request send → first streamed token, ms.
* `ITL_i,j` — inter-token latency, j = 1..OSL−1, ms.
* **`TPOT_i = (e2e_i − TTFT_i) / (OSL − 1) = mean_j ITL_i,j`**, ms/token.
* `e2e_i` — send → last token, ms.

**Trap 1 — the TPOT denominator, and why it is load-bearing at OSL=8.** The client's
convention is **OSL−1**, not OSL. DERIVED (this file), verified in all six manifests to
6 significant figures: `result.tpot_ms.mean == result.itl_ms.mean` exactly, and
`(e2el_ms.mean − ttft_ms.mean)/(OSL−1)` reproduces both
(e.g. `b3rev2_pair_01_2_m15`: 747.4284 / 747.4284 / 747.4340). **At OSL=8 the two
conventions differ by 8/7 = 14.3% — larger than the entire c512p throughput result
(+7.50% geomean, `grounding/WORKLOAD_EVIDENCE.md` §W8.2).** Any TPOT compared against an
external number must state the denominator or the comparison is void.

**Trap 2 — under a closed-loop driver, TTFT and e2e are backlog measures, not service
measures.** All six banked manifests carry `workload.driver = "closed"` and
`result.arrival.request_rate = null`. Under closed loop with `C` in flight, TTFT includes
the queueing delay behind the requests already admitted; it therefore scales with `C` and
with the backlog, not only with the kernel.

DERIVED (this file), Little's-law and wave arithmetic on the banked manifests:

| cell | Little's mean in-flight = `M4 × e2e_mean` | vs offered C | waves = `N/C` | `wall / e2e_p50` |
|---|---:|---:|---:|---:|
| c32p (both arms) | 31.9 | **99.7–99.8%** of 32 | 32.0 | 32.5 / 34.5 |
| c512p m15 | 460.5 | **89.9%** of 512 | **4.0** | 4.04 |
| c512p native_dp | 490.9 | 95.9% of 512 | **4.0** | 4.07 |

⇒ **the c512p cell is a 4-generation run.** `wall / e2e_p50 ≈ 4.04` confirms it independently.
Its TTFT p50 of 43.1 s is ≈ 3 generations of queueing, and its M5 of 94.9 tok/s/req is a
backlog artifact, not a prefill rate. This is the mechanical explanation of the "metric map
is regime-dependent" finding already banked at `grounding/WORKLOAD_EVIDENCE.md` §W8.3.

**Rules.**

* **R-L1.** TTFT/e2e/goodput from a closed-loop cell are labelled
  **`(closed-loop, backlog-inclusive)`** on every line where they appear, and may be used
  only as a **paired comparison between arms at matched offered load** — never as an
  absolute deployment latency, never against a published SLO.
* **R-L2.** An SLO-relevant latency claim requires an **open-loop** cell
  (`BENCHMARK_PROTOCOL.md:36-42`; the `o50p/o75p/o90p` cells at
  `OPEN_LOOP_BASE_RATE = 4.97 req/s`, `nightshift/LOG.md:315`). Those cells have **never been
  run** (`grounding/WORKLOAD_EVIDENCE.md` §5, hole H6). Until they are, **this project has
  zero SLO-relevant latency numbers**, and every latency statement in draft8 inherits R-L1.
* **R-L3 (closed-loop cell sizing, DERIVED).** A closed-loop cell must satisfy
  `N / C ≥ 20` (≥20 request generations) or be reported with its ramp/drain share computed
  by the Little's-law deficit above. c32p passes (32 waves, 99.7% in-flight occupancy);
  **c512p fails (4 waves, 10.1% in-flight deficit)** and its numbers carry that flag.

### 1.4 Goodput under an explicit SLO pair — the decision metric

**Definition.** Given an SLO pair `(a, b)` = (TTFT ceiling ms, TPOT ceiling ms), a request is
*good* iff `ok ∧ TTFT_i ≤ a ∧ TPOT_i ≤ b`. Then

```
goodput_req(a,b)   = |{good}| / T_w                       [req/s]
goodput_in(a,b)    = Σ_{good} ISL_i / T_w                 [input tok/s]
goodput_out(a,b)   = Σ_{good} OSL_i / T_w                 [output tok/s]
attainment(a,b)    = |{good}| / R                         [dimensionless]
```

The SLO pair is **part of the metric name**, always: `goodput_in(2000,100)`. An unqualified
"goodput" is a violation.

**Why this is the primary metric for two of the five classes, demonstrated on our own data.**
DERIVED (this file) from `b0v5` per-query records (c32p, closed-loop, **n=1 unbalanced pair,
calibration-grade** — `grounding/TP8_STATE.md` §7), TTFT-only attainment sweep:

| TTFT ceiling `a` (ms) | m15 (DP8/EP8) | native_tuned_tp (TP8+EP) | winner |
|---:|---:|---:|---|
| 1,500 | 1.3% | **66.9%** | TP8 |
| 2,000 | 9.0% | **87.8%** | TP8 |
| 3,000 | 80.3% | **95.7%** | TP8 |
| 3,500 | 93.1% | **97.3%** | TP8 |
| **4,000** | **98.3%** | 97.9% | **m15** |
| 5,000 | **100.0%** | 98.4% | **m15** |

⇒ **the arm you should deploy at C=32 flips at a TTFT SLO of ≈3.5–4.0 s.** The same
flip, on TPOT: TP8 wins at `b ≤ 600 ms` (89.4% vs 86.6%), m15 wins at `b ≥ 700 ms`
(98.8% vs 93.6%). And at c512p (`b3rev2`), TPOT attainment: **m15 100.0% at any `b ≥ 800 ms`
while native_tuned_dp needs `b ≥ 3,400 ms` to reach 100%** — for any `b ∈ [800, 3300)` ms
the m15 arm is the only viable arm, `goodput_in(60000,1000)` = **42,835.7 vs 1,471.5 tok/s
(29×)**, against a headline throughput delta of only +6.60%.

**This is the single strongest argument in the whole campaign for the metric map**, and it is
invisible in every number draft8 presented. It also carries its own caveats in full: closed-loop
(R-L1), n=1 unbalanced pair, and *both* arms score **0/2048 at any interactive SLO** at c512p
(`attainment(5000, 200) = 0.0%` for both) — which is the measurement that proves c512p is a
**batch** cell, not a serving cell, under this driver.

### 1.5 tokens/$ — pricing model

```
tokens_per_dollar(metric) = (metric [tok/s] × 3600) / P_node          [tok/$]
P_node = node price, USD per node-hour        <- a DECLARED PARAMETER, never a measurement
```

* **R-$1.** `P_node` is stated numerically on every line that quotes tokens/$. No in-repo
  artifact prices this node; there is **no** node-hour cost anywhere in the repo (grep over
  `nightshift/`, `docs/distributed/`, `overnight/` returns nothing). So `P_node` is an
  **ASSUMED** input, and tokens/$ is a **DERIVED** quantity that inherits that tag.
* **R-$2 (the anti-laundering rule).** At fixed hardware, tokens/$ is a *monotone rescaling*
  of throughput and adds **no information**. It may be reported as an independent metric
  **only when the arms differ in resource footprint** — memory, replication, or achievable
  concurrency. In that case it must be evaluated at each arm's **achievable max concurrency**
  under the same `--gpu-memory-utilization`, per
  `SERVING_BENCHMARK_METHODOLOGY.md:338-347` ("costs priced into the same table"; past sin:
  **41 GB of replica caches framed as a kernel win**).
* **R-$3.** Any arm carrying expert replication prices it here, not in a footnote. Known
  costs already in the ledger: EPLB0 placement ≈ **1.31 GiB/rank** transfer buffer + 56.6 MiB
  load window (`M15_EPLB0_COMPOSE.md`, via `grounding/WORKLOAD_EVIDENCE.md` §W4e); the
  replication frontier K=2→16 costs **4.8 → 38.1 GiB** (LAW-47); M25's accumulator ring
  ≈ **940 MB** plus ≈ **+205 MB/rank/boundary** of tower accumulator
  (`grounding/TP8_STATE.md` §4.3, DERIVED and explicitly "not yet budgeted anywhere").

### 1.6 Energy — GATED, not yet an axis

```
J_per_token(metric) = ∫_window Σ_g P_g(t) dt / tokens_delivered_in_window     [J/token]
tok_per_J = 1 / J_per_token
```

**Status: NOT INSTRUMENTED.** No power or energy measurement exists anywhere in this repo.
Per **LAW-62** ("prove a knob is settable before pre-registering an axis" — the case where a
spec-sheet denominator manufactured a refutation, and the case where `K0_SYNTH_ROUTE` made a
whole pre-registered axis untestable), and per **LAW-11** (this node exposes no live xGMI
throughput counter; `amd-smi --xgmi` returns N/A), energy is **gated**:

* **G-E1.** Before energy appears in any cell of the grid, one node check must establish that
  `rocm-smi --showpower` (or `amd-smi metric -p`) returns a live per-GPU value at ≥1 Hz for
  all 8 GPUs, and the sampler must be shown not to perturb the arm (a null-control pair).
* **G-E2.** If G-E1 passes: sample at ≥1 Hz in a sidecar, integrate over the **steady-state
  window only** (§1.8), report both raw and idle-baseline-subtracted, and record the idle
  baseline per GPU. Declare the convention on every line.
* **G-E3.** Until G-E1 passes, energy is reported as **`not instrumented`** — never estimated
  from TDP. This is cheap (one node command) and is listed in §8.

### 1.7 Percentile and aggregation conventions

* **P1.** Percentiles are **nearest-rank on the sorted completed-OK set**, computed over the
  steady-state window, matching the client's `ttft_ms/tpot_ms/e2el_ms/itl_ms` blocks. Report
  **p50, p90, p99, max** for every latency; p90 is not optional (it is the only percentile
  between p50 and p99, and both c32p and c512p show p50/p99 inversions).
* **P2.** **Never compute a percentile of a difference from a difference of percentiles.**
  `TPOT p50 ≠ (e2e p50 − TTFT p50)/(OSL−1)`; at c512p m15 the two differ by 0.15% and at
  c32p by more (DERIVED). Compute per-request, then aggregate.
* **P3.** Across pairs: report **median of pairs** plus every individual pair ratio plus the
  spread. Never the mean alone (`SERVING_BENCHMARK_METHODOLOGY.md:227-247`).
* **P4.** Across ranks: **rank-max** (LAW-64); rank-0 shown separately when both exist
  (LAW-63d, the +38% gap).
* **P5.** Ratios across arms are **geometric** means when combining pairs (the banked c512p
  pair ratios 1.08395 / 1.06601 → geomean **1.0750**, `grounding/WORKLOAD_EVIDENCE.md` §W8.2).
* **P6.** A delta smaller than the cell's **measured** variance envelope (§1.10) is reported
  as **"within drift"**, never as a win.

### 1.8 Measurement window: warmup, ramp, steady state, drain

```
[ server start ][ warmup ][ ramp-in ][ ======= STEADY STATE ======= ][ drain ][ cooldown ]
                          ^ excluded                                  ^ excluded
```

* **W1 — warmup.** Excluded, and its wall time reported separately (the client already
  records it: `warmup wall 6.191 s` stock vs `4.867 s` m15,
  `aug18-prefill/BOTTLENECK_SIGNALS.md:1-60` provenance table — note these already differ by
  27% between arms).
* **W2 — ramp-in / drain, closed loop.** Exclude the first `C` and last `C` completed
  requests (the generations during which in-flight < C). Report the excluded fraction.
  DERIVED: at c32p this is 64/1024 = 6.3%; **at c512p it is 1024/2048 = 50%** — which is why
  R-L3 exists.
* **W3 — ramp-in / drain, open loop.** Exclude the first and last `⌈2 × e2e_p99⌉` seconds of
  arrivals. **BLOCKED:** the banked client leaves `per_query[i].sent_s / issued_s /
  scheduled_s = null` (verified in `b0v5_pair_01_1_m15.json`), so **arrival-time windowing is
  not currently computable**. Populating those three fields is a hard prerequisite for the
  open-loop cells (§8, gap #2).
* **W4 — the two reported statistics.** Every result reports **both** the whole-run statistic
  (comparable with the historical archive) **and** the steady-state statistic, with the
  difference shown. Where they disagree by more than the cell's envelope, the steady-state
  number is the claim.
* **W5 — kernel rigs.** MoK campaigns are 5-rotation, **500 warmup / 100 timed / 600-epoch
  soak**, ~3 min 5 s per campaign (LAW-60). Device stamps are read **from the final soak
  epoch only** — `[MPS TS]` values are running maxima that are never reset (LAW-63c;
  `aug11/PLOTS.md:74`). A stamp read from a timed iteration is a lie.

### 1.9 The two artifacts this protocol exists to neutralize

**Artifact A — the deterministic final-iterations dip.** MEASURED, training arm
(`aug18/FAIRNESS_AUDIT.md:563`, asymmetry A1): the banked 10-iteration tail
"happens to contain both of production's anomalous fast final iterations (−72 ms each), and
the mega arm has no comparable artifact." Sized against the steady-state median (iters
51–258): production rep2 **1,328.4 vs 1,339.5 ms (−0.83%)**, rep1 **1,329.5 vs 1,343.2
(−1.02%)**, mega **1,587.8 vs 1,588.3 (+0.03%)**. The same aggregation window "silently
discounts every T0 number… all ~7–11 ms optimistic."

> **Neutralization N1.** The reporting window is a **declared steady-state range** (iteration
> indices, or request indices, or a wall-clock interval), never "the last k iterations" and
> never "the whole run" alone. The declared range is printed with the number. Both the banked
> window and the steady-state window are reported (W4) so the artifact's size is visible.
>
> **Neutralization N2.** The window is chosen **identically for both arms and before looking
> at the numbers**, and the choice is recorded in the manifest row
> (`window_policy` field, §3.3). An arm-specific window is a fairness-audit failure.

**Artifact B — closed-loop non-stationarity.** MEASURED, serving arm: fill declines
monotonically through a run in **both** arms — per 100-step window, stock 0.4170 → 0.3695 and
m15 0.3511 → 0.3348 (`aug18-prefill/BOTTLENECK_SIGNALS.md:275-278`, via
`grounding/WORKLOAD_EVIDENCE.md` §W1). "Any single-φ workload coordinate is a run-average
over a moving quantity." The corpus's own dummy-rate discrepancy (31.2% vs 56.0%) is the
**same artifact** — the capture window is the ramp phase (`REPORT.md:393-427`, retraction R5).

> **Neutralization N3.** Every workload coordinate that is non-stationary (φ, dummy rate, κ,
> chunk-events/step) is reported as a **windowed trajectory** — at minimum the two half-run
> values plus the whole-run aggregate — never as a single scalar.
>
> **Neutralization N4.** No workload statistic measured on the ramp window may be quoted as a
> run statistic. The corpus instrument is bounded at 64 calls/rank and is **ramp-phase by
> construction** (`route_capture/skewhook_v2/sitecustomize.py`; §4.1 instrument table).

**Artifact C — ~15% e2e variance.** MEASURED and already protocol: **±15% day-to-day at n=1,
±18% per arm from position** (`docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md:229-231`),
mitigated by `ARM_COOLDOWN=240` symmetric, fresh server per arm, order-balanced pairs, **≥5
pairs hard floor** (`BENCHMARK_PROTOCOL.md:50-53`). See §1.10 for the refinement.

> **Neutralization N5.** Every quoted delta is sized against the envelope **of its own cell**
> (§1.10), and the claim wording states the envelope. `n ≥ 5` order-balanced pairs remains a
> hard floor for a headline **and is not sufficient on its own** — see R-N1.

### 1.10 The variance envelope is itself a function of W

DERIVED (`grounding/WORKLOAD_EVIDENCE.md` §W10b) from the two c512p manifests, opposite arm
order, ~1 h apart: m15 M1 spread **+0.11%**, native_tuned_dp **+1.80%**, TTFT p50 −0.11%,
TPOT p50 −0.07/−0.08%. ⇒ **at c512p, arm-position + hour-scale drift is ≤1.8% on throughput
and ≤0.25% on latency percentiles** — an order of magnitude below the ±18% position band,
which is a c32p-era measurement. Corroborating the other direction: camp3 at c32p gave
**+23.5% / −13.1%** across two pairs for a +2.70% mean (`REPORT.md:86`).

* **R-N1.** Report per cell: `envelope_source ∈ {measured-this-cell, inherited-c32p,
  unmeasured}` and the numeric envelope used. **Inheriting the c32p band at c512p is
  over-conservative; inheriting the c512p band anywhere else is fraud.**
* **R-N2.** A measured envelope **never** lowers the `n ≥ 5` floor. The c512p n=2 evidence is
  explicitly still "directional" (`grounding/WORKLOAD_EVIDENCE.md` §W8.2;
  `BENCHMARK_PROTOCOL.md:50-53`). It raises `n` when the envelope is large; it never lowers it.
* **R-N3.** Minimum detectable effect must be stated with `n`: a claimed delta smaller than
  the cell envelope requires either more pairs or adjudication at the boundary instrument
  (§4). LAW-45: "any predicted schedule change smaller than ~15% must be adjudicated at the
  boundary, not e2e."

---

## 2. THE METRIC MAP

### 2.0 How to read this section

Each workload class names: the **deployment decision** the class corresponds to; the
**PRIMARY** metric (exactly one); **GUARD** metrics that can veto a primary win; the
**mediating quantity Q** from the cost model that the class is sensitive to; and the
**evidence status** — because three of five classes have **zero** measurements.

**The governing meta-law is measured, not asserted.** LAW-44: median TTFT **fell 5.0%**
(2,173.76 vs 2,288.68 ms) while node input tok/s **fell 7.84%** in the *same pair*
(`aug18-prefill/BOTTLENECK_SIGNALS.md:88-99`). And
`grounding/WORKLOAD_EVIDENCE.md` §W8.3: at c512p the arm that **loses TTFT p50 by 1.49×
wins e2e p50** because it wins TPOT p50 by 4.24×. ⇒ **at low concurrency TTFT and throughput
can move in opposite directions, so a single headline number is not merely incomplete — it
can carry the wrong sign.**

### 2.1 The map

| class | deployment decision | PRIMARY | GUARDS (any can veto) | mediating Q | evidence |
|---|---|---|---|---|---|
| **A — prefill-dominated batch / offline** | "how much corpus can this node chew per hour?" | **M1** aggregate input tok/s, steady-state window; equivalently tokens/$ at declared `P_node` | (1) accuracy gate; (2) coverage receipt `sealed/in_bucket`; (3) **W** matched across arms (LAW-40); (4) memory footprint + achievable C; (5) TPOT p99 as a *stability* guard only | **fill φ**, `W` | MEASURED at c512p, n=2 order-balanced (geomean 1.0750) — still below the ≥5 floor |
| **B — interactive low-C serving** | "will users tolerate it, and how many can I serve?" | **`goodput_in(a,b)`** at the deployment's own SLO pair, measured **open-loop** | (1) TTFT p99 and TPOT p99 (tails are the SLO); (2) M1 (capacity ceiling); (3) attainment curve over `a` and `b`, not a point; (4) coverage + accuracy | **exposed collective fraction** (TP8) / **fill φ** (DP) | c32p closed-loop only, **n=1 unbalanced**; open-loop cells **never run** (H6) |
| **C — saturated high-C serving** | "what is the max sustainable rate at SLO?" | **max offered rate `λ*` such that `attainment(a,b) ≥ 0.99`** (the SLO-throughput), from the open-loop sweep | (1) M1 at `λ*`; (2) TTFT p99 stability across the sweep (queue divergence check); (3) TPOT p50/p99; (4) W, coverage, accuracy | fill φ → 1; `κ` co-scheduling density | **ZERO** — `o50p/o75p/o90p` defined, never executed (H6) |
| **D — decode-heavy long-OSL** | "how fast does one user's stream run, and how many streams fit?" | **M6** per-request decode rate (= 1000/TPOT p50), with **TPOT p99** co-primary | (1) M2 aggregate output tok/s **at variable OSL only**; (2) TTFT p99; (3) KV-cache capacity → achievable C; (4) accuracy | message size at decode shape; K1 flips (`grounding/TP8_STATE.md` §5 M5) | **ZERO** — `c8/c16/c32` (ISL 1024, OSL 512) and `c512` defined, never executed (H1). **The banked 4.24× TPOT advantage at c512p is a closed-loop batch artifact and is NOT a decode result.** |
| **E — mixed continuous batching** | "the real production regime" | **`goodput_in(a,b)` + `goodput_out(a',b)` reported as a pair**, with the traffic mix declared | (1) `κ` (prefill chunk-events per in-bucket step) — a *scheduler* guard; (2) W; (3) both p99s; (4) coverage; (5) accuracy | `κ`, W, joint `n_orig` across 8 ranks | **ZERO controlled** — chunked prefill mixes them constantly, but the only instrument is step counts (H2) |

### 2.2 Per-class notes that change decisions

**A — batch/offline.** The c512p cell *is* this class, and the measurement proves it:
`attainment(5000, 200) = 0.0%` for **both** arms (DERIVED, §1.4). Therefore
`grounding/WORKLOAD_EVIDENCE.md` §W8.3's "m15 loses TTFT p50 by 1.49×" is **not a
decision-relevant statement in this class** — no one deploying a 43-second-TTFT service
cares about 29 seconds vs 43. Report it for completeness, never as a caveat on the primary.
Conversely the guard that *does* bind here is **W** (LAW-40): the m15 arm ran the padded
B4096 graph on 249.1 of 300 steps vs stock's 205.9, `W = 1.1922`, worth +8.4% wall — the
entire measured gap of a *different* cell. **An A-class result without matched W is void.**

**B — interactive low-C.** This is the cell we lose on throughput (−21.17%) and win on every
tail (`grounding/TP8_STATE.md` §1.5: M15 wins TTFT p99 by 23%, TPOT p99 by 24%, e2e p99 by
33%). §1.4 converts that into the actionable form: **the arm flips at a TTFT SLO of
≈3.5–4.0 s and at a TPOT SLO of ≈600–700 ms.** The honest campaign statement is therefore
not "we lose C=32" but "**at C=32, TP8+EP wins every SLO tighter than ~3.5 s TTFT; DP8/EP8+M15
wins every SLO looser than ~4 s, and wins all tails**" — with R-L1 attached, since it is
closed-loop, n=1, unbalanced.

**C — saturated high-C.** `λ*` is not a metric we have ever computed. Defining it is free;
measuring it needs the open-loop cells. Note `BENCHMARK_PROTOCOL.md:36-42` already says why
this class exists: the closed-loop dummy share is "partly an artifact of concurrency-32 with
OSL=8", and only open-loop "decides whether the pad/dummy-skip advantage is production value
or a benchmark artifact." That sentence is the campaign's biggest single exposure.

**D — decode-heavy.** Three separate reasons the banked TPOT numbers cannot be quoted here:
(i) closed-loop backlog (R-L1); (ii) `OSL = 8` with `ignore_eos` — 7 ITLs per request, and
`M2 ≡ M4 × OSL` (R-M2); (iii) the mega executes decode steps **at prefill shape** — post-M23
the rescue routes uniform-decode ranks through the B4096 graph, so 4,096 padded rows run for
~1 real token per rank (`grounding/WORKLOAD_EVIDENCE.md` §3.2). Whatever we measure today on
a "decode" step is a prefill-shaped measurement. Class D is a **hole**, and the protocol's job
is to keep it labelled as one.

**E — mixed.** The guard that dominates is not a kernel metric at all. LAW-50: perfect DP
co-scheduling would need **128 padded steps instead of the observed 260/322 ⇒ 15–25% of node
throughput sits in DP prefill co-scheduling, for either kernel — larger than any kernel delta
in this project's ledger.** The measured driver is `κ`: chunk-events per in-bucket step,
**stock 3.93 vs m15 3.19 of 8 possible**. ⇒ **in class E, `κ` is a mandatory reported field
(§3.5), and a kernel claim whose `κ` differs across arms by more than 5% is a scheduler
result mislabelled.**

### 2.3 Metrics that are banned as headlines, with the reason

| banned | why | replacement |
|---|---|---|
| bare "tok/s" / "TPS" / "throughput" | 22–63× ambiguity, DERIVED §1.2 | `M1`…`M7`, spelled out |
| aggregate output tok/s at fixed OSL | `M2 ≡ M4 × OSL` exactly (DERIVED, 6/6 manifests) | M6, or M2 at variable OSL only |
| closed-loop TTFT vs an SLO | backlog measure (R-L1), 4-wave run at c512p | open-loop `goodput_in(a,b)` |
| a single latency percentile | p50/p99 invert in **both** cells, opposite directions (§W8.3) | p50 + p90 + p99 + max |
| "beats production" without the fairness audit | `nightshift/CLAUDE.md` §fairness-audit, veto power | the three-way decomposition + signed audit |
| any pre-2026-08-18 serving number | seal fired on ~2% of heavy steps; retired (LAW-49, commits `9fb1705d`, `824a3e45`) | post-M23 numbers with receipts |
| the retracted strings | `grounding/WORKLOAD_EVIDENCE.md` §0.2 / `BANKED_LAWS.md` §7 | the corrected forms named there |
| mixing the two per-call skew instruments | m18diag p50 **5.15×** vs corpus p50 **2.61×**, unreconciled, ~2× apart (H17) | name the instrument on the number |
| "−32% fewer bytes" (M25) | FALSE as written; re-derived ≈ **−9.7%** (`grounding/TP8_STATE.md` §1.3) | "CDAR's case is overlap, not bytes" |

---

## 3. THE WORKLOAD TUPLE

### 3.1 Why this is a header and not a footnote — the two historical failure modes

1. **The kernel silently did not run.** Under the exact-4096 seal the megakernel fired on
   **~2% of heavy steps** (0.77% of all steps) while the candidate additionally ran
   **de-graphed** — `cudagraph_mode = NONE` — so every pre-M23 A/B measured a *doubly*
   handicapped candidate (`grounding/WORKLOAD_EVIDENCE.md` §W9; LAW-49). Three serving pairs
   (−5.7%, +11.3%, +39.8%) were deleted for exactly this reason
   (`aug14/M18_REPLICATION_RESULTS.md:100-113`).
2. **The two arms did different amounts of work.** LAW-40: near-identical real traffic
   (−2.0%) but `W = 2.568 vs 2.154 = 1.1922` padded MoE rows per real token, worth **+8.4%
   wall — the entire measured gap** of +8.51%.

⇒ **coverage counters and composition counters are not hygiene; they are the difference
between a result and its opposite.** Both belong in the header.

### 3.2 The ten blocks

| block | fields | source of truth |
|---|---|---|
| **1. Identity** | `run_tag, cell_id, pair_id, arm_name, arm_position_in_pair, generated_utc, operator_session` | manifest `label`, `generated_utc` |
| **2. Model & precision** | `model, n_layers, n_dense_layers, hidden, n_routed_experts, n_shared_experts, topk, weight_dtype, activation_dtype, kv_cache_dtype, moe_path_precision` | server config dump; `TP8_OVERLAP_ANALYSIS.md:10-13` (ASSUMED model card) |
| **3. Hardware & topology** | `node_id, n_gpu, arch (gfx950/gfx942), gpu_model, interconnect, multi_node=false, perf_level, clock_policy, per_gpu_idle_junction_temp_C` | `rocm-smi`; **required because** `--showperflevel` returns `auto` (DVFS free) and idle junction temps already span **10 °C** (`aug18/FAIRNESS_AUDIT.md`, F8) |
| **4. Parallelism & server config** | `TP, DP, EP, max_num_batched_tokens, max_num_seqs, chunked_prefill, prefix_caching, gpu_memory_utilization, scheduling_policy, cudagraph_capture_max, eplb_config` | `campaign_v5/ARMS.md:90-100, 190-210` |
| **5. Workload** | `driver ∈ {closed, open}, concurrency \| request_rate + burstiness, num_prompts, ISL distribution (name + params), OSL distribution (name + params), prompt_source, prompt_generator, seed, ordered_prompt_token_id_stream_sha256, ignore_eos, temperature, duplicate_prompts_in_cell` | manifest `workload` block — **already exactly these fields** |
| **6. Coverage / seal** | `steps, in_bucket, sealed, sealed_ragged, sealed_exact, refused_not_unanimous, refused_not_ready, eager_b4096, min_orig, max_orig, sum_orig` — **per rank, reported as rank-max and rank-min** | `RAGGED_SEAL_RECEIPT` (`M23_RAGGED_SEAL_DESIGN.md:815-830`) |
| **7. Routing provenance** | `route_source ∈ {synthetic_balanced, histogram_replay, captured_replay, live}, capture_id, eplb ∈ {off, placement_only, redundant_K}, placement ∈ {linear, rr}, skew_statistic + which_instrument` | `grounding/WORKLOAD_EVIDENCE.md` §W4; §4.1 instrument table |
| **8. Arm build identity** | `image_digest, patch_markers_present[], patch_markers_absent[], DHK_ROOT_pin, K0P6_M15_SRC_REV, kernel_name_pin, text_sha256, K0_MPS_CFG (raw + decoded: C, g, mode, flush_rows, pull_fallback, timestamps), K0P6_MPS_ENABLE_MODE14, K0P6_MPS_ENABLE_TBO, -D set` | `grounding/KNOB_INVENTORY.md` Appendix; **LAW-32**: making mode-14 merely *reachable* cost the mode-12 ratchet **+726.9 µs** with mode-12 source untouched |
| **9. Statistics** | `n_pairs, per_pair_ratios[], median_of_pairs, geomean, spread, window_policy, steady_state_range, excluded_fraction, envelope_value, envelope_source` | §1.7, §1.8, §1.10 |
| **10. Costs** | `memory_footprint_per_rank_GiB, kv_cache_GiB, achievable_max_concurrency, replication_GiB, P_node_USD_per_hour (if tokens/$ quoted), energy ∈ {J/token, not_instrumented}` | §1.5, §1.6; `SERVING_BENCHMARK_METHODOLOGY.md:338-347` |

### 3.3 The template (paste this; every field required, `null` is a legal value but a blank is not)

```yaml
# ---- WORKLOAD TUPLE v1 (aug18-ablations/design/METRICS_PROTOCOL.md §3) ----
identity:      {run_tag: b3rev2, cell_id: c512p, pair_id: 01, arm: m15,
                arm_position: 2, generated_utc: 2026-08-18T19:37:13Z, session: nightshift}
model:         {model: r1, layers: 61, dense_layers: 3, hidden: 7168,
                routed_experts: 256, shared_experts: 1, topk: 8,
                weight_dtype: fp8, act_dtype: bf16, kv_dtype: fp8, moe_precision: bf16}
hardware:      {node: 8xMI355X-01, n_gpu: 8, arch: gfx950, interconnect: xgmi_intranode,
                multi_node: false, perf_level: auto, idle_junction_C: [.., x8]}
parallelism:   {TP: 1, DP: 8, EP: 8, max_num_batched_tokens: 4096, max_num_seqs: 128,
                chunked_prefill: true, prefix_caching: false, gpu_mem_util: 0.70,
                sched_policy: fcfs, cudagraph_capture_max: 512, eplb: off}
workload:      {driver: closed, concurrency: 512, request_rate: null, burstiness: null,
                num_prompts: 2048, isl: {dist: constant, len: 4096},
                osl: {dist: constant, len: 8}, prompt_source: qsl,
                prompt_generator: mlperf-qsl-concat-v2-stride7, seed: 5120802,
                prompt_sha256: 13bdc7a2..., ignore_eos: true, temperature: 0.0,
                duplicate_prompts_in_cell: 0}
coverage:      {steps: 300, in_bucket: 299, sealed: 299, sealed_ragged: .., sealed_exact: ..,
                refused_not_unanimous: 0, refused_not_ready: 0, eager_b4096: 0,
                min_orig: .., max_orig: .., sum_orig: .., rank_agg: max,
                seal_ratio: 1.000, uniform_rescued: 6}
routing:       {route_source: live, capture_id: null, eplb: off, placement: linear,
                skew_stat: null, skew_instrument: null}
build:         {image_digest: sha256:8445..., markers_present: [PF4H_INTEGRATION_PATCH_V3_M15,
                PF4H_COVERAGE_PATCH_V1, PF4H_M23_RAGGED_SEAL_V1], markers_absent: [PF4H_RR_*],
                dhk_root_pin: ~/DHK-m15, m15_src_rev: 3, kernel_name_pin: ...,
                text_sha256: ..., mps_cfg_raw: 0x..., mps_cfg: {C: 28, g: 353, mode: 12,
                flush_rows: 16, pull_fallback: .., timestamps: 0},
                enable_mode14: 0, enable_tbo: 0, dflags: [...]}
stats:         {n_pairs: 2, pair_ratios: [1.08395, 1.06601], median_of_pairs: 1.0750,
                geomean: 1.0750, spread: 0.0179, window_policy: whole_run,
                steady_state_range: null, excluded_fraction: 0.0,
                envelope: 0.018, envelope_source: measured-this-cell}
costs:         {mem_per_rank_GiB: .., kv_GiB: .., achievable_max_C: .., replication_GiB: 0,
                P_node_usd_per_hour: null, energy: not_instrumented}
```

Values shown are the **real** ones from `b3rev2_pair_01_2_m15.json` and
`campaign_v5/ARMS.md:90-100` where an artifact exists; `..` marks a field the current
artifacts do **not** carry, which is itself the finding — see §8.

### 3.4 The inline quote form (for slides, commit messages, chat)

When a full YAML block will not fit, the number is quoted in this exact one-line form:

```
M1 42,835.7 tok/s  [c512p | DP8/EP8+m15 | 8xMI355X gfx950 | closed C=512 | ISL 4096 / OSL 8
                    | n_prompts 2048 | qsl seed 5120802 | seal 299/299 | φ≈0.985 | W n/a
                    | n=2 order-balanced, geomean 1.0750, envelope ±1.8% measured-this-cell]
```

Rule: **cell, topology, hardware, driver+C, ISL/OSL, prompt source, coverage, n, envelope.**
Nine fields. A number missing any of them is not quotable.

### 3.5 Derived counters and the validity gates they feed

Computed from block 5 + block 6 and reported alongside them:

| derived | formula | gate |
|---|---|---|
| fill `φ` | `Σ in_bucket_sum_orig / (Σ in_bucket × 8 × 4096)` | must be reported **per cell** — "37.6% is c32p's number" (`FILL_AWARE_DESIGN.md:1319-1323`) |
| padding multiplier | `1/φ` | c32p m15 2.66×, stock 2.31× |
| dummy rate | `uniform_rescued / in_bucket` | whole-run, not ramp-window (N4) |
| seal ratio | `sealed / in_bucket` | **gate: ≥0.95, else the arm did not run the traffic** |
| **`W`** (padded rows per real token) | `(in_bucket × 4096 × 8) / Σ sum_orig` | **gate: |W_a/W_b − 1| ≤ 0.05 across arms, else the A/B is a composition result** (LAW-40) |
| **`κ`** (co-scheduling density) | prefill chunk-events per in-bucket step, of 8 | **gate for class E**: differences >5% are scheduler effects (LAW-50) |
| Little's occupancy | `M4 × e2e_mean / C` | **gate: ≥0.95 for a closed-loop cell to be called saturated** (c512p scores 0.899 — DERIVED) |
| waves | `num_prompts / C` | **gate: ≥20** (R-L3); c512p scores 4.0 |
| accuracy | exact-token SHA **or** bounded rel-err | mandatory; note `ordered_output_token_id_stream_sha256` differs even **stock vs stock** (`REPORT.md:167`), and **11.33% of requests produced different tokens between arms** — so at c32p the SHA gate is *not usable* and the bounded-error path is mandatory (LAW/§W10d) |

### 3.6 Reproducing the derived numbers in this document

Every DERIVED figure in §1 comes from the six banked manifests and is regenerable with
stdlib Python over `distributed-kernels/tp8_mega/results/serving/*.json`
(`result`, `workload`, `per_query` blocks). The serving-side forensics counters
(`W`, `κ`, step composition, TTFT quantiles, two-state wall fit) are regenerable with the
existing tool `aug18-prefill/analyze_serving_pair.py` (37 KB, stdlib-only), which "re-runs
this entire document on any future pair" — it consumes `c32p.json` + client log +
`seal_receipts.txt`.

---

## 4. THE TWO-INSTRUMENT DOCTRINE

### 4.1 The instruments, and what each can resolve

| instrument | class | statistic | **minimum detectable effect** | cost | binding artifacts |
|---|---|---|---|---|---|
| **MoK synthetic prefill campaign** (5-rotation, 500 warmup / 100 timed / 600 soak) | KERNEL | rank-max p50, median of 5 | campaigns reproduce to **0.09%**; screens drift **6.6% within one batch** ⇒ "treat ~3% as the smallest callable end-to-end screen delta" (LAW-60) | **~3 min 5 s** | pin trap (`DECOMP_RUNBOOK.md`); `K0_MAXTOK == T` ⇒ T-sweeps are **batch-size** sweeps, not fill sweeps; M15 body **fails** the gate at T≠4096 |
| **device phase stamps** `[MPS TS]` | KERNEL | per-phase, rank-0 CTA-max | plan σ **4.7 µs (1.1%)**, M7 σ **63 µs (2.3%)**; calibrated 1:1 against e2e (ΔM6 −79.7 stamp vs −75.0 paired campaign) | free with the run | running maxima never reset ⇒ **final soak epoch only** (LAW-63c); rank-0-only ⇒ not comparable to cross-rank numbers (+38%, LAW-63d) |
| **8-GPU single-process boundary harness** (G25-0b / G25-1) | KERNEL | median wall of 7 iters at 235 MB | depth arms separated at **0.1–0.3%** (93.6/93.4/93.3 GB/s read as "flat"); slab arms at **2.1×** | minutes | operating point must be declared (§4.5); all-PASS correctness gates (`CDAR_BENCH_PASS`, exact bf16 sum, zero timeouts) |
| **e2e serving pair** under `BENCHMARK_PROTOCOL.md` | SERVING | median of ≥5 order-balanced pairs | envelope-dependent: **±15% day / ±18% position at c32p**; **≤1.8% M1 / ≤0.25% latency at c512p** (§1.10) | ~25 min/pair + 240 s cooldowns | coverage receipts mandatory; accuracy gate mandatory; fairness audit veto |
| **route corpus / replay** | CORPUS | routing shape | n/a | CPU-only; replay 139 s/campaign | corpus captures **only 4,096-row calls** ⇒ structurally blind to fill; skips graph-capturing calls; 64 calls/rank ⇒ **ramp-phase** |
| **serving manifests** (client v3) | SERVING | full tuple + per-query | as above | free | `sent_s/issued_s/scheduled_s` are **null** ⇒ no arrival-time windowing (§8) |

### 4.2 Role assignment (the doctrine in one table)

| question | instrument | why |
|---|---|---|
| *Which of two schedules is better?* | **boundary (precision)** | e2e's envelope (±15%) exceeds most schedule deltas; LAW-45 says so explicitly |
| *By how much, in deployment?* | **e2e (validity)** | the rig over-predicts: replay 0.756–0.847× vs serving `r ≈ 0.95` (contradiction C5) |
| *Does the mechanism exist at all?* | **boundary + phase stamps** | four producer-side hypotheses returned null in G25-1 ⇒ "one instrumented run replaces the next four guesses" |
| *Is the delta real?* | **e2e, ≥5 order-balanced pairs, per-cell envelope** | §1.10 |
| *Does the workload actually have the property?* | **receipts + corpus** | φ, dummy rate, κ, skew — never inferred from timing |

### 4.3 "Matching W cell" — the definition the shipping rule depends on

A boundary result and an e2e result are at a **matching W cell** iff every coordinate below
is either equal, or its difference is declared with a stated reason to expect no sign change:

| coordinate | tolerance |
|---|---|
| topology (TP/DP/EP degrees) | **exact** — fill does not exist under TP8 (`grounding/TP8_STATE.md` §3.3) |
| tokens/rank `T` | within 20%, and both on the same side of the measured **1,600–1,800 tok/rank** break-even |
| fill `φ` | within 0.10 absolute, **or** the rig is declared 100%-fill and the serving φ is quoted alongside |
| routing distribution | same class (balanced / histogram-replay / captured / live) **and** same instrument |
| co-residency `co` | **exact** (binary) — LAW-8 vs LAW-20: depth is decisive at `co=1`, flat at `co=0` |
| message size / compute:transport ratio | within 2× of the deployment ratio (§4.5) |
| architecture | **exact** — gfx942 flips K1 (remote atomics ~3× slower than stores) |

**Mismatch on any coordinate does not void the boundary result; it scopes it.** A mismatched
pair yields a *mechanism* claim, never a *magnitude* claim.

### 4.4 The shipping rules

* **R-S1 (the headline rule).** *No schedule claim ships without boundary evidence **and** at
  least one e2e confirmation at a matching W cell (§4.3).* A boundary-only result is
  published as **"mechanism, unvalidated at deployment"**; an e2e-only result is published as
  **"observed, mechanism unattributed"**. Neither is a law (§5).
* **R-S2 (direction of inference).** Boundary rigs **rank and falsify**; only e2e **sizes**.
  Any magnitude carried from a rig into an e2e sentence must be marked as a prediction and
  compared against the measured value in the law card.
* **R-S3 (the negative-result rule).** A falsification at the boundary is publishable on its
  own and **must** be published — the ledger's value is largely its null results
  (G25-1's four falsified hypotheses; LAW-28's carrier-conditional reversal). A null still
  requires §4.5's operating-point declaration, or it is a null about the rig.
* **R-S4 (instrument parity).** Every arm carries a `.text` sha256 and a mechanism-level
  invariant check. **The resource tuple is not a parity gate** — a build passed SGPR, VGPR,
  AGPR, scratch/lane, LDS, MFMA census and the atomic census and still cost the ratchet
  **+726.9 µs** (LAW-58). And the flag-ON build **must differ**: "a flag-on build that came
  out identical would mean the flag never reached the code and the arm would be a lie."
* **R-S5 (the fairness audit is not optional).** No "beats production" claim ships without
  the adversarial audit's written sign-off; it holds veto (`nightshift/CLAUDE.md`,
  `SERVING_BENCHMARK_METHODOLOGY.md:409-471`).
* **R-S6 (cluster by run).** Ranks within a run are **not** independent samples. The same null
  reads `t = −5.01` treating 160 (run, rank) cells as independent and `t = −0.73` with runs
  as units (LAW-61a). Runs are the unit.

### 4.5 Operating-point declaration (new, and it is why R-S3 has teeth)

Every boundary result declares its **compute:transport ratio** and the deployment ratio it is
meant to model. The motivating case is measured: the G25-1 rig runs `k_inner=2048` with a
1,439 µs two-burst compute floor against a 1,880–2,214 µs collective — **the collective is
~3× longer than the compute it must hide under**, while the deployment ratio is ~7.2 ms
compute vs ~2.8 ms collective, i.e. ~3:1 the other way. ⇒ **the rig is ~5× off**, best
conceivable fused wall is `max(compute, transport) ≈ 1,880 µs`, and hiding was Amdahl-capped
near zero (`grounding/TP8_STATE.md` §2.4). The four falsified hypotheses were therefore
"tested blind" — the nulls stand, but they are nulls about a regime where the win was small.

* **R-S7.** A rig arm reports `compute_span_us`, `transport_span_us`, their ratio, and the
  deployment ratio with its citation. If the ratio is outside 2× of deployment, the arm is
  labelled **`off-operating-point`** and may not be quoted as evidence about magnitude.

---

## 5. THE LAW CARD

### 5.1 Why a card

The experts asked for "a list of what we'd like to teach people about" and for the cost model
to be stated as "if you change X in the workload, the schedule should change by Y, because
mediating quantity Q crossed threshold θ" (`../BRIEF.md:14-17, 23`). A law card is exactly
one such statement in publishable form. **The unit of the campaign's output is the card, not
the number.** `grounding/BANKED_LAWS.md` already holds 65 laws in an informal version of this
shape; the card standardizes it and adds the two fields that ledger lacks:
**predicted vs measured**, and **retirement condition**.

### 5.2 The template

```
LAW-<id>  <one-line name>
──────────────────────────────────────────────────────────────────────────────
STATEMENT      One sentence, conditional form. Names the workload change, the
               schedule change, and the mechanism. No numbers in the sentence.

MEDIATING Q    The physical quantity that carries the effect, with its units and
               how it is computed from W.

Q-INEQUALITY   S_a ≻ S_b  ⟺  Q(W) > θ            (or the two-sided / interval form)
               The decision rule an engineer applies without rerunning our rig.

θ              value ± uncertainty | units | how fitted (which arms, which rig,
               how many points) | is θ a constant of the hardware or of the workload?

PREDICTED      What M(W,S;θ) says the boundary delta should be, computed BEFORE the run.
vs MEASURED    What the run gave, with instrument and n. Show the residual.

CELLS TESTED   Full workload tuple (§3.4 short form) for every cell where the law was
               exercised, marked HOLDS / FAILS / UNTESTED.

INSTRUMENTS    Which of §4.1, with the MDE, and the operating-point declaration (R-S7).

CONFIDENCE     PROVEN-replicated | MEASURED-once | HYPOTHESIS | ANALYTIC
               (tiers defined in grounding/BANKED_LAWS.md, reading rule 2)

FAILURE SCOPE  Where the law is known or expected to break, and why. A law with an
               empty failure scope is not finished.

RETIREMENT     The single measurement that would kill it. If none exists, the card is a
               HYPOTHESIS regardless of how many numbers it carries.

MANIFEST ROWS  The arm ids that generated every number above.
```

### 5.3 Worked card #1 — the cleanest law we own

```
LAW-K4  Injection depth is congestion control against co-resident compute, not against
        the fabric.
──────────────────────────────────────────────────────────────────────────────
STATEMENT      Bounding the number of outstanding remote writes per producer thread
               is decisive exactly when the transport shares the device with a
               compute body; in a pure-transport phase the same bound does nothing.

MEDIATING Q    co ∈ {0,1} — is an MFMA body co-resident with the carrier during the
               transport phase? (Under LAW-12, occupancy is one block/CU at
               megakernel register/LDS budgets, so `co` is a property of the
               SCHEDULE, not of the grid.)

Q-INEQUALITY   depth-bounded ≻ unbounded  ⟺  co = 1.
               θ is a binary threshold, not a magnitude. Corollary: at co=1 the
               optimum is a CLIFF at d=4, not a curve.

θ              co = 1. Secondary: d* = 4 (d=8 costs +0.77%; d=16/32 lose the entire
               throttle, 7.3σ / 5.9σ). d=2 never measured — the left side of the
               cliff is unexplored.

PREDICTED      From LAW-20's co-resident result alone, a bulk transport rig should
vs MEASURED    have shown a depth effect. It did not: {4,8,16,32} gave
               93.6 / 93.4 / 93.3 GB/s (towers) and 67.1 / 67.1 / 67.0 / 66.8
               (atomics) — FLAT (commit 108def6e, results/g25_0b_v1_results.txt).
               Co-resident: 7,110.8 unbounded → 6,495.8 at depth-4 = −615.0 µs,
               t = −80.2, per-campaign ranges disjoint by 598 µs (aug11/LESSONS.md:41-54).
               RESIDUAL: the model had no `co` term; this measurement created it.

CELLS TESTED   HOLDS  [MoK T=4096 | DP8/EP8 | balanced std=0 | 100% fill | gfx950 | co=1]
               HOLDS  [G25-0b 8-GPU CDAR | 235 MB | pure transport | gfx950 | co=0]
               UNTESTED [fused CDAR at the corrected operating point] -> named M3
               UNTESTED [any serving cell]  UNTESTED [gfx942]

INSTRUMENTS    MoK 5-rotation campaign (MDE ~0.09% campaign / ~3% screen);
               8-GPU boundary harness, median of 7, all-PASS correctness.
               OPERATING POINT: the co=0 arm is pure transport by construction
               (declared, in-spec). The co=1 arm is the deployment schedule.

CONFIDENCE     PROVEN-replicated for co=1 (sweep + waterfall + post-fix contrast,
               three campaigns); MEASURED-once for co=0.

FAILURE SCOPE  (a) The co=0 result is one rig, one message size — LAW-8 flags it as
               the one result to re-run before M commits to a d(co) term.
               (b) Compile-time constraint: depth CANNOT be a runtime register —
               `s_waitcnt vmcnt(N)` is simm16-only, and templating the hot loop on
               Depth spilled one 4-byte value with 389 reloads, contaminating the
               control arm (credit.cuh:12-22). Any "runtime policy" design must keep
               policy CHOICE runtime and policy CODEGEN compile-time.
               (c) Untested under skew, at T≠4096, and on gfx942.

RETIREMENT     A fused-CDAR depth sweep at the corrected compute:transport ratio
               (M2 then M3, grounding/TP8_STATE.md §5) that shows a depth effect at
               co=1 but at a different d*, or no effect at co=1 at all.

MANIFEST ROWS  g=65 / g=353 mode-12 arms (exp_24, exp_35);
               g25_0b_v1 {depth × bps × transport} matrix.
```

### 5.4 Worked card #2 — a card for a NEGATIVE result (the template must handle these)

```
LAW-F1  Fuse the collective only when the exposed collective time exceeds our
        transport's penalty against the vendor library.
──────────────────────────────────────────────────────────────────────────────
STATEMENT      Moving a dense collective inside the megakernel pays only if the
               time it was exposing is larger than the bandwidth we give up by not
               using the tuned library; it is a bandwidth-ratio decision, not a
               philosophy.

MEDIATING Q    ρ = (exposed collective fraction of the step) and
               β = (our transport bandwidth) / (vendor library bandwidth).
               Fuse iff  ρ·(1 − hiding_loss) > (1/β − 1)·(collective fraction).

Q-INEQUALITY   fused ≻ unfused ⟺ hidden_fraction > 1 − β.
               With β = 1/1.51 = 0.662, fusing needs to hide > 33.8% of the
               collective merely to break even.

θ              β = 0.662 MEASURED in-rig (CDAR 2,214 µs vs RCCL 1,470 µs for the
               same 235 MB boundary AR). ρ ≈ 0.27–0.28 DERIVED, two independent
               routes (throughput-vs-roofline; in-rig RCCL arm) — NOT profiler-confirmed.

PREDICTED      Predicted (design): −32% fewer bytes and ≥90% of the AR hidden.
vs MEASURED    Measured: bytes −9.7% not −32% (the design doc mixes per-rank and
               aggregate conventions in one row; ring AR is bandwidth-optimal and
               ERS+MAG is a two-shot AR moving identical bytes at boundary 1);
               hiding = (3,653 − 3,606)/2,214 = 2.1% against a ≥90% bar.
               Walls: compute floor 1,439 | GEMM→RCCL 2,909 | phased CDAR 3,653 |
               fused CDAR 3,606 µs. RESIDUAL: the byte claim is RETRACTED; the
               hiding claim is RED.

CELLS TESTED   FAILS [G25-1 | TP8+EP boundary | 16K tokens | 235 MB | 256-row slabs |
                      k_inner=2048 | gfx950 | OFF-OPERATING-POINT, ~5× short on compute]
               UNTESTED at the deployment compute:transport ratio (~3:1 the other way).
               UNTESTED at decode message sizes (rccl-tests absent from the node;
                      one CDAR point at 3.67 MB reverses the K1 decision entirely).

INSTRUMENTS    8-GPU boundary harness; correctness GREEN everywhere (exact bf16 sums,
               zero timeouts, every arm and config; ISA gate 0 spills / 0 scratch).
               OPERATING POINT: DECLARED OFF-SPEC (R-S7) — collective ~3× longer than
               the compute it must hide under; best conceivable fused wall ≈ 1,880 µs.

CONFIDENCE     MEASURED-once, explicitly interim.

FAILURE SCOPE  The whole card is scoped to an off-operating-point rig. It licenses
               "we have not achieved hiding", NOT "hiding is unachievable". The
               fabric-efficiency framing bounds the transport half independently:
               RCCL 280 GB/s = 79% of the (unreproduced) egress ceiling, CDAR best
               205 GB/s = 58% — the 1.5× gap is almost exactly the headroom, i.e. an
               efficiency gap in our own transport, not a topology or byte-count
               disadvantage. CAVEAT: that ceiling is flagged "unreproduced; treat as
               unknown, measure first" (OVERLAP_ABSTRACTIONS.md:147).

RETIREMENT     M2 (one argv: sweep k_inner to the deployment ratio) + M1 (device phase
               ledger). Pre-registered falsifier already on the record: if hiding is
               still <10% at the deployment ratio, the "hide it in the epilogue"
               thesis is in serious trouble.

MANIFEST ROWS  m25_boundary_bench.hip arms {compute, rccl, phased, fused} ×
               {tokens, slab_rows, iters, depth, k_inner, bps, transport} —
               that argv list already IS the manifest schema for this skeleton.
```

### 5.5 Card rules

* **R-C1.** A card with an empty FAILURE SCOPE or an empty RETIREMENT is not publishable —
  it is a tuning constant wearing a law's clothes.
* **R-C2.** PREDICTED must be written **before** the run. A card whose prediction was written
  afterwards says so in that field.
* **R-C3.** Cards that contradict must be filed together with the resolution, following
  `grounding/BANKED_LAWS.md` §6's format (six contradictions, each with resolution and
  residual risk). **Never resolve a contradiction silently inside the cost model.**
* **R-C4.** Joint cells are one card. LAW-21 measured that carrier relocation and injection
  bound are **not separable and never additive** (relocation alone is a **+210.6 µs
  regression**; with the bound it is −615.0 µs) ⇒ they get **one** card with a 2-D Q, not two
  cards whose coefficients a reader might add.
* **R-C5.** Every card names which of the five workload classes (§2.1) it applies to, and the
  classes where it is UNTESTED. Three of five classes currently have zero measurements; a
  card silent about that is over-claiming by omission.

---

## 6. INTEGRATION — what this extends, amends, and contradicts

### 6.1 Extends (new material, no conflict)

| this doc | extends | note |
|---|---|---|
| §1.2 M1–M7 metric algebra | `BENCHMARK_PROTOCOL.md:61-64`; `SERVING_BENCHMARK_METHODOLOGY.md:325-337` | those name metrics; this defines them as formulas and forbids the ambiguous string |
| §1.4 goodput under an SLO pair | — | **entirely new**; no goodput definition exists anywhere in the repo |
| §1.5 tokens/$ | `SERVING_BENCHMARK_METHODOLOGY.md:338-347` (costs in the same table) | adds the pricing formula and the anti-laundering rule R-$2 |
| §1.6 energy | — | new, and **gated** under LAW-62 |
| §1.9 N1–N5 | `aug18/FAIRNESS_AUDIT.md:563` (the artifact was found, never made protocol) | promotes a fairness-audit finding into a standing rule |
| §2 the metric map | `BENCHMARK_PROTOCOL.md:43-47` ("a grid, not a point") | the grid says *which cells*; the map says *which metric per cell* |
| §3 workload tuple | `SERVING_BENCHMARK_METHODOLOGY.md:274-280` §(f) | (f) lists 5 fields; this lists 10 blocks incl. build identity and derived gates |
| §4 two-instrument doctrine | `../BRIEF.md:99-100`; LAW-45 | promotes an observation into a shipping rule with a matching-cell definition |
| §4.5 operating-point declaration | `grounding/TP8_STATE.md` §2.4 | new rule R-S7 |
| §5 the law card | `grounding/BANKED_LAWS.md` | standardizes the existing 65-law ledger and adds PREDICTED-vs-MEASURED + RETIREMENT |

### 6.2 Amends (proposed replacement text — paste into `nightshift/BENCHMARK_PROTOCOL.md`)

**Amendment 1 — replace §3 item 5.** Current text (`BENCHMARK_PROTOCOL.md:61-64`):

> "Metrics: aggregate input tok/s (prefill throughput), TTFT p50/p99 (what users feel),
> tok/s/GPU — every number with its complete workload spec attached. Median-of-pairs with the
> spread shown; claim sized against measured drift."

Replacement:

> 5. **Metrics: per the metric map** (`distributed-kernels/fused_moe/overnight/aug18-ablations/
>    design/METRICS_PROTOCOL.md` §2). Each cell declares one PRIMARY metric and its GUARDS.
>    Every result reports, at minimum: **M1** aggregate input tok/s, **M6** per-request decode
>    rate, **TTFT p50/p90/p99/max**, **TPOT p50/p90/p99/max** (denominator OSL−1, stated),
>    **M7** tok/s/GPU, and **`goodput_in(a,b)`** at the cell's declared SLO pair. Latency
>    numbers from a **closed-loop** driver are labelled `(closed-loop, backlog-inclusive)` and
>    are never quoted against an SLO. Every number carries the **workload tuple** (that file
>    §3). Median-of-pairs with the spread; **claim sized against the cell's own measured
>    variance envelope**, not a global band.

**Amendment 2 — add to §3 item 1 (drift).** After "±18% position effect and ±15% day drift":

> The envelope is a **function of the cell**: measured ≤1.8% on M1 and ≤0.25% on latency
> percentiles at c512p vs ±15–18% at c32p. Every cell reports the envelope it used and its
> source (`measured-this-cell` / `inherited-c32p` / `unmeasured`). **A measured envelope never
> lowers the ≥5-pair floor.**

**Amendment 3 — add to §2 (workload).** After the open-loop bullet:

> A **closed-loop** cell is valid only if `num_prompts / concurrency ≥ 20` and the Little's-law
> occupancy `request_throughput × e2e_mean / C ≥ 0.95`. The banked c512p cell scores **4.0
> waves and 0.899 occupancy** (DERIVED) — its latency numbers are backlog measures and its
> ramp/drain share is ~10% of the wall.

**Amendment 4 — add to §3 item 3 (coverage receipts).** Extend to composition:

> Coverage proves the kernel **ran**; composition proves the arms did the **same work**.
> Report `W` (padded MoE rows per real token) per arm; a cross-arm `W` mismatch >5% makes the
> pair a composition result, not a kernel result (worked case: `W = 1.1922`, worth the entire
> measured +8.4% gap).

**Amendment 5 — `SERVING_BENCHMARK_METHODOLOGY.md` §(f).** Replace the five-field minimum with
a pointer to §3.3's ten-block template and the §3.4 nine-field inline form.

### 6.3 The one contradiction, stated openly

`BENCHMARK_PROTOCOL.md:61-64` designates **"TTFT p50/p99 (what users feel)"** as a headline
metric without qualification. **This document contradicts that** in two respects, on evidence:

1. **Under a closed-loop driver, TTFT is not what users feel — it is backlog.** All six banked
   manifests are `driver: closed` with `request_rate: null`. DERIVED: at c512p the run is 4
   generations deep (`wall/e2e_p50 = 4.04`) and TTFT p50 = 43.1 s is ≈3 generations of
   queueing; the same m15 kernel shows a per-request prefill rate of 1,675 tok/s/req at c32p
   and 94.9 at c512p. Calling either "what users feel" is wrong.
2. **TTFT p50 alone can select the wrong arm.** DERIVED §1.4: at c32p the arm that wins TTFT
   p50 (TP8, 1.34 s vs 2.44 s) **loses** the attainment test at any SLO looser than ~4 s and
   loses every p99. The decision metric is `goodput_in(a,b)`; TTFT p50 is a guard.

**Resolution proposed to the operator:** keep TTFT p50/p99 as *mandatory reported guards*,
demote it from *headline* to *guard* in classes A and D, promote `goodput_in(a,b)` to primary
in classes B/C/E, and require the `(closed-loop, backlog-inclusive)` label until the open-loop
cells run. This is Amendment 1. **Until the operator accepts it,
`BENCHMARK_PROTOCOL.md` wins and this note stands as a dissent on the record.**

---

## 7. COMPLIANCE CHECKLIST (paste into any result doc)

```
[ ] Cell declared, and its class (A/B/C/D/E) from §2.1 named
[ ] PRIMARY metric is the class's primary — not a metric chosen after seeing the data
[ ] All GUARD metrics reported, including the ones that go against us
[ ] Every "tok/s" written as M1..M7 or spelled out; no bare TPS
[ ] TPOT denominator stated (OSL-1); percentiles nearest-rank; p50/p90/p99/max all present
[ ] Closed-loop latency labelled "(closed-loop, backlog-inclusive)"
[ ] goodput reported with its (a,b) SLO pair in the metric name
[ ] Workload tuple v1 attached in full (§3.3) or the 9-field inline form (§3.4)
[ ] Coverage receipt: seal ratio >= 0.95, rank-max and rank-min shown
[ ] Composition: W per arm, cross-arm mismatch <= 5%; kappa reported for class E
[ ] Little's occupancy >= 0.95 and waves >= 20, or the deficit is reported
[ ] Steady-state window declared, identical for both arms, chosen before looking (N1/N2)
[ ] Non-stationary coordinates (phi, dummy, kappa) reported as trajectories (N3)
[ ] n >= 5 order-balanced pairs for a headline; per-pair ratios + median + spread + geomean
[ ] Envelope value AND source stated; delta inside the envelope reported as "within drift"
[ ] Accuracy gate passed and named (exact-token SHA where usable; bounded rel-err at c32p)
[ ] Build identity: image digest, patch markers, .text sha, K0_MPS_CFG decoded,
    MODE14/TBO gates pinned, DHK_ROOT pin
[ ] Boundary evidence AND >=1 e2e confirmation at a matching W cell (R-S1) - or the claim
    is labelled "mechanism, unvalidated" / "observed, unattributed"
[ ] Operating point declared for every rig arm (R-S7)
[ ] Costs priced in the same table: memory, KV, replication, achievable max C
[ ] tokens/$ carries P_node; energy is J/token or "not_instrumented"
[ ] Fairness audit signed off for any "beats production" wording
[ ] No banned string from §2.3; no retracted string from BANKED_LAWS §7 / WORKLOAD_EVIDENCE §0.2
[ ] Every law claimed is filed as a LAW CARD with FAILURE SCOPE and RETIREMENT
```

---

## 8. Instrument gaps this protocol creates, and what they cost

Ordered by how much of the campaign's claim surface they block. Each is cheap; each is
currently blocking a section of this document from being executable.

| # | gap | blocks | cost |
|---|---|---|---|
| **1** | **Open-loop cells never run** (`o50p/o75p/o90p`, base rate 4.97 req/s) | every SLO-relevant latency number; classes B and C entirely; R-L2 | node time only — the cells are already specified |
| **2** | **Client does not populate `sent_s` / `issued_s` / `scheduled_s`** (verified null in `b0v5_pair_01_1_m15.json`) | arrival-time windowing (W3), queue-delay decomposition, `λ*` for class C | one client patch (v4) |
| **3** | **No per-cell fill receipt** — `M24_FILL_RECEIPT` does not exist; c512p's 98.5% is reported but its derivation is not shown in-repo | φ as a reported field for any new cell; the `W` gate at cells other than c32p | small patch |
| **4** | **Energy not instrumented anywhere** | §1.6 entirely | one node command to pass G-E1 |
| **5** | **The `/8` peer-summing convention in `RAGGED_SEAL_RECEIPT` is assumed** — the printf has never been read on the node | every φ and W number's absolute scale (ratios survive) | one node read |
| **6** | **Two per-call skew instruments disagree ~2×** (5.15× vs 2.61× p50) | routing provenance field; any skew term in the cost model — changes the modeled skew tax by ~2× | CPU-only |
| **7** | **No node-hour price** | tokens/$ | one operator decision |
| **8** | **Exact-token SHA is not a usable parity gate at c32p** — differs stock-vs-stock; 11.33% of requests differ between arms | the accuracy gate at c32p; the bounded-rel-err path must be specified and run instead | policy + one run |
| **9** | **The 2.8 ms/layer exposed-AR pool has never been profiler-confirmed** | every public statement of the TP8 Amdahl prize pool (it is a *model*, not a measurement) | one torch profile of the native TP8 arm |

---

## 9. One-paragraph summary for the master doc

Every number this project reports is a pair `(metric, workload tuple)`, and the metric is
chosen by the deployment decision the cell represents — not by the run. Five workload classes,
five primaries: aggregate input tok/s for batch/offline, `goodput_in(a,b)` under a declared
SLO pair for interactive and saturated serving, per-request decode rate for decode-heavy, and
a goodput pair for mixed continuous batching; with a fixed guard set that can veto in every
class. Our own artifacts prove the discipline is load-bearing rather than pedantic: the same
run reports two "tok/s" numbers 63× apart; the arm you should deploy at C=32 **flips at a TTFT
SLO of ≈3.5–4.0 s**; at c512p an arm that loses TTFT p50 by 1.49× is the **only** arm that
meets any TPOT SLO below 3.3 s, by 29× on goodput against a +6.6% headline; and a single
composition confound (`W = 1.1922`) once accounted for the entire measured gap in a different
cell. Boundary rigs rank and falsify at 0.09% resolution; e2e sizes at a per-cell envelope
that is itself measured (±15% at c32p, ≤1.8% at c512p); nothing ships with only one of them.
And every claim that survives both is published as a law card with a Q-inequality, a θ, a
predicted-vs-measured line, a failure scope, and the one measurement that would kill it.

---

## Amendments from review (2026-08-18)

Extended by `../ABLATION_METHODOLOGY.md §5`; nothing here is retracted. Additions forced by
BLOCKING items in `../review/FEASIBILITY_CRITIQUE.md` and `../review/MECHANISM_CRITIQUE.md`:

1. **Actuation receipts are a per-cell admission gate**, not hygiene (FEASIBILITY B7): a device-side
   skipped-rows/`T_eff` counter for the φ cell, an observed per-rank destination-load CV printed per
   campaign for the skew cell, and a per-rank descriptor echo for per-rank `C`. Without them a silent
   no-op is indistinguishable from the registered falsification.
2. **New gate G0-e — thermal/clock parity** (FEASIBILITY B6): continuous `amd-smi` clock+temperature
   at ≥0.2 Hz stored per run, spread flagged above 2%; the first invocation of every session is
   discarded or labelled; a foreign-process check is recorded in `runs.jsonl`; sequential level
   sweeps are order-randomized with a repeating anchor configuration regressed out.
3. **`σ_rig` must be measured, not inherited** (FEASIBILITY B4). §4.1's "depth arms separated at
   0.1–0.3%" is a **between-arm spread**, not repeatability; no repeated identical boundary
   invocation exists in the repo. Arm P0-A0 supplies it and every sample size is re-derived from it.
   R-N1 is extended to the kernel instruments: every cell reports its own measured envelope.
4. **The boundary instrument's unit cost is a declared range [1, 3] min** until measured (this
   document says "minutes"; the ladder said "~1 min" uncited).
5. **`h` is reported as `h_ledger` (span overlap) alongside `h_wall`** (MECHANISM B4); law cards for
   the TP8 family state `h_ledger` in the PREDICTED-vs-MEASURED field.
6. **§6.3's contradiction is resolved in favour of the standing protocol for now:** Phase 4's PRIMARY
   is `BENCHMARK_PROTOCOL.md`'s TTFT p50/p99 and `goodput_in(a,b)`/M1 are reported as guards, with
   every cell flagged where the two conventions select different arms. The amendment stands as a
   dissent until the operator rules.
7. **Class-D and class-E cells are demoted to exploratory** ("observed, mechanism unattributed") until
   funded to the ≥5-pair floor with a measured envelope; PE4/PE5 leave the scored prediction set. The
   open-loop cell additionally needs client v4 (`sent_s/issued_s/scheduled_s` are null), tracked as X6.
8. **P11 is restated as a TOST equivalence test** at LAW-60's 3% smallest-callable-screen margin, with
   three already-measured zero-door arms named as `campaign_role: screen_validation` — an accept-the-
   null test at ±1σ would "fail" ~32% of arms under a perfect screen (FEASIBILITY M2).

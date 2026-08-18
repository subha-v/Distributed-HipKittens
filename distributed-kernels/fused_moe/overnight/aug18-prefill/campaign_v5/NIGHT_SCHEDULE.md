# Tonight's schedule — the arithmetic, two options, one recommendation

Everything below is wall-clock arithmetic on the **v5.1** wrapper's real control
flow. No hand-waving: the cost model is stated first, the two plans are priced
against it, and the recommendation names what it cuts and why.

> **This document was rewritten after an adversarial review.** The v5.0 schedule
> was priced on three optimistic inputs (a flat 260 s `c32p` for every arm, a
> 60 s `stop_server` against a 180 s bound, and an `o50p` budgeted at "near two
> minutes" when its arrival span alone is 217 s), it prescribed `--pairs 5` for a
> 3-arm matrix that cannot be position-balanced at 5, it ran three invocations
> into one `RUN_TAG`, and none of its commands set `PROMPT_SOURCE=qsl`. All four
> are fixed here.

---

## 1. Cost model

The wrapper's unit of cost is an **arm-run**: one fresh server, its cells, its
teardown. Cells inside one arm-run share the overhead, which is the single
most important fact for planning — **adding a cell is cheap, adding an
arm-run is not.**

```
arm-run = ARM_COOLDOWN (240 s)          # mandatory, the ±18% position effect
        + server startup S              # model load + graph capture + patches
        + sum(cell runtimes)
        + stop_server (180 s budgeted)  # SIGINT + drain, never SIGKILL
```

`stop_server` is budgeted at its **bound**, `STOP_TIMEOUT_S=180`, not at a hoped
60 s: an 8-engine DP server with MoRI shmem teardown is the slow case and it is
the case we will meet 18 times.

`S` is the one number that cannot be derived from the receipts we have, and it
swings the whole night. Two scenarios are priced throughout:

| scenario | S | overhead per arm-run (240 + S + 180) |
|---|---|---|
| optimistic | 420 s | **840 s (14 min)** |
| conservative | 600 s | **1020 s (17 min)** |

**Step B0 measures `S` for real.** v5.1's `wait_ready` writes
`startup_seconds.txt` into *each arm's own output directory* (v5.0 wrote a single
shared `$ROOT/.startup_seconds` dotfile that every arm overwrote), so after B0
read:

```
$ROOT/pair_01/1_m15/startup_seconds.txt
$ROOT/pair_01/2_native_tuned_tp/startup_seconds.txt
```

Native arms load faster than ours (no `apply.py`, no patch chain); the tuned
arms may load slower (`FULL_AND_PIECEWISE` graph capture). Treat the **mean** as
`S` and the **max** as the risk.

### Cell runtimes

Derived from the camp3 `c32p` receipts (19,277 / 17,900 tok/s for m15 / stock)
and from the cells' own token arithmetic, padded ~15% for warmup and client
startup. **The native arms' throughput on this model is unmeasured**, so their
`c32p` carries an explicit range rather than an assumption.

| cell | spec | tokens driven | runtime |
|---|---|---|---|
| `c32p` (m15) | conc 32, ISL 4096, OSL 8, n=1024 | 4.19 M input | **260 s** |
| `c32p` (stock) | " | " | **280 s** |
| `c32p` (native_tuned_tp) | " | " | **420 s budgeted** — 300 s at 14k tok/s, 524 s at 8k, 838 s at 5k. Unmeasured until B0. |
| `o50p` | open, 0.50 × base rate, n=512 | arrival-bound | **300 s** (span 217 s + drain + warmup) |
| `o75p` | open, 0.75 × base rate, n=512 | arrival-bound | **220 s** (span 145 s + drain) |
| `o90p` | open, 0.90 × base rate, n=512 | arrival-bound | **190 s** (span 121 s + drain) |
| `c1det` | conc 1, ISL 4096, OSL 8, n=32 | serialized | **90 s** |
| `c8det` | conc 8, ISL 4096, OSL 8, n=64 | 8 DP ranks loaded | **120 s** |
| `c8` | conc 8, ISL 1024, OSL 512, n=256 | decode-bound | **200 s** |
| `c16` | conc 16, ISL 1024, OSL 512, n=512 | decode-bound | **210 s** |
| `c32` | conc 32, ISL 1024, OSL 512, n=1024 | decode-bound | **230 s** |
| `c512` | conc 512, ISL 1024, OSL 512, n=2048 | decode-bound | **300 s** |
| `c512p` | conc 512, ISL 4096, OSL 8, n=2048 | 8.39 M input | **360 s** + memory risk (never yet run) |

At the banked `c32p` ceiling (19,277 input tok/s ÷ 4096 ISL) the closed-loop
request rate is ≈ **4.71 req/s**. That is m15's number and it is *not* the one to
offer: `OPEN_LOOP_BASE_RATE` must be the **minimum** measured ceiling across the
arms in the invocation (see §3, B0), because an arm offered more than it can
serve reports its queue as its latency. The wrapper refuses to run an o-cell
without both `OPEN_LOOP_BASE_RATE` and `OPEN_LOOP_BASE_RATE_SOURCE`.

### `EPLB_PREWARM`

`EPLB_PREWARM` defaults to **1**, and a prewarm is a full `c32p` (1024 prompts,
~260 s) *per arm-run*. Every command below sets `EPLB_PREWARM=0` explicitly.
Dropping that prefix from a 18-run block silently adds **+1.3 h**.

---

## 2. Option A — the full protocol

Priced with 6 pairs (the smallest count that is both ≥5 and divisible by the arm
counts we run) and the corrected overheads.

| block | composition | arm-runs | S=420 | S=600 |
|---|---|---|---|---|
| A1 | 6 pairs × 4 arms {m15, stock, native_default, native_tuned_tp} × {c32p, c8, c16, c32} | 24 | 12.0 h | 13.2 h |
| A2 | 6 pairs × 5 arms (+ native_tuned_dp) × {c512p, c512} | 30 | 12.5 h | 14.0 h |
| A3 | 6 pairs × 4 arms × {o50p, o75p, o90p} | 24 | 10.3 h | 11.5 h |
| A4 | accuracy gate (c1det + c8det), 5 arms | 5 | 1.46 h | 1.71 h |
| **total** | | **83** | **36.3 h** | **40.4 h** |

(A1's cell time per pair is 260+280+420+340 s of `c32p` across the four arms plus
4 × 640 s of `c8`/`c16`/`c32`; A2 is 5 × 660 s; A3 is 4 × 710 s. A1 and A3 are
also not strictly runnable as written — 6 is not a multiple of 4 — so a real
Option A needs 8 pairs there and costs more still.)

Option A is a three-night campaign, not a night. Its cost is dominated by
arm-run overhead — 83 runs × 14–17 min = **19–24 h of cooldown, model loading
and teardown alone**, before a single token is served. Anyone proposing A should
propose it as a weekend.

---

## 3. Option B — what actually fits

Built by the stated priority ladder — *c32p matrix > accuracy gate > open-loop at
c32p > small cells > c512p* — and then trimmed until the arithmetic closed.

### Composition

| step | what | arm-runs | S=420 | S=600 |
|---|---|---|---|---|
| **B0** | **Calibration, BOTH arm families.** One `m15` arm-run and one `native_tuned_tp` arm-run, cell `c32p`, 1 pair with `ALLOW_UNBALANCED_PAIRS=1` (a single pair cannot be position-balanced; these numbers are calibration inputs, never a quoted ratio, and the analyzer stamps the root UNBALANCED so nobody can quote them as one). Yields (a) the real `S` per family, (b) each family's ceiling — `OPEN_LOOP_BASE_RATE` is the **minimum** of the two, (c) a live check that the M23 seal and the receipts fire, and (d) the first exercise of AMD's tuned flags on 0.25.1, which is where flag drift will show. | 2 | **0.66 h** | 0.76 h |
| **B1** | **Accuracy gate FIRST.** `c1det` **and** `c8det` on the determinism launch, arms {m15, **stock**, native_tuned_tp}. `stock` is not optional: it is the only arm sharing m15's numerics class, so without it the gate can measure cross-class agreement but can never establish bit-exactness for the candidate and returns `PASS_WEAK`. Run before the pairs: a FAIL voids every performance number that would follow, and finding that out at 03:00 is worth 53 minutes. | 3 | **0.88 h** | 1.03 h |
| **B2** | **The headline matrix.** `c32p`, **6 pairs**, arms {m15, stock, native_tuned_tp}. Rotation-balanced (each arm holds each position exactly twice), `ARM_COOLDOWN=240`, fresh server per arm. | 18 | **5.80 h** | 6.70 h |
| **total B** | | **23** | **7.34 h** | **8.49 h** |

Plus **≥1 h of unallocated contingency** — this plan has no slack of its own and
the largest single unknown (`native_tuned_tp`'s `c32p` runtime) is not resolved
until B0 finishes. On an 11-hour night, B at S=600 plus contingency is 9.5 h.

### Opportunistic add-ons, in the order to spend leftover time

| add-on | why it is cheap | marginal cost |
|---|---|---|
| **+ `o90p` co-resident in every B2 arm-run** | a cell, not an arm-run — it amortizes the 14–17 min overhead already paid | +190 s × 18 = **+0.95 h** |
| **+ `c8` co-resident** | the honest losing cell; shows the inversion rather than hiding it | +200 s × 18 = **+1.00 h** |
| **+ `o75p`, `o50p`** co-resident | completes the utilization sweep | +520 s × 18 = **+2.60 h** |
| **+ `native_default`** | the shipped-default ratio, no longer directional | a 4th arm forces PAIRS to a multiple of 4, i.e. **8 pairs × 4 arms = 32 runs ≈ 10.4 h** — not an add-on, a second night |

> `native_default` cannot be added "on pairs 1–2 only". Adding an arm to some
> pairs and not others destroys the rotation: with 3 arms in four pairs and 4
> arms in two, no arm holds each position equally and the ±18% position effect
> stops cancelling. Arms are a per-campaign constant. If the shipped-default
> ratio is wanted, it is its own invocation with its own `RUN_TAG` — and the
> analyzer will keep the two roots' pairs separate, which is exactly right.

### What Option B cuts, and the cost of cutting it

| cut | why it is the right cut | what is lost |
|---|---|---|
| `c512p`, `c512`, `native_tuned_dp` | c512p has **never been run** — unknown memory behaviour at conc 512 with ISL 4096, and it drags in a fifth arm because AMD's recommendation flips to DP+EP there. A first-run-tonight cell is a debugging session, not a measurement. | no large-batch claim; state this explicitly in REPORT.md |
| `c16`, `c32` | they measure the decode-bound regime, where the mega is expected to invert; `c8` alone (as an add-on) is enough to *show* the inversion honestly | a coarser hybrid/deployment-policy row |
| `native_default` | `native_tuned_tp` is the **harder** baseline (AMD's documented config, claimed 1.35–1.52× from AITER MLA alone). Beating it implies beating the shipped default; the reverse is not true. Keeping the harder one is the anti-sandbagging choice (fairness item 2). | the headline is quoted vs the vendor-tuned baseline only; no shipped-default ratio at all — better than a 2-pair one that breaks the rotation |
| the 6th pair | **NOT cut.** BENCHMARK_PROTOCOL.md §3.1 makes ≥5 pairs binding and the rotation needs a multiple of 3. Pairs are the last thing to go. | — |

---

## 4. Recommendation

**Run Option B**, in the order B0 → B1 → B2, `PROMPT_SOURCE=qsl` throughout, a
distinct `RUN_TAG` per step, with `o90p` co-resident in B2 only if the B0 clock
allows.

Reasoning:

1. **It is the only plan whose arithmetic closes.** Option A is 35–39 h. Any
   plan that pretends otherwise will be cut mid-campaign, and a campaign cut
   mid-way produces exactly the unbalanced, partial-pair data this protocol
   exists to forbid.
2. **It protects the two things that cannot be recovered later.** ≥5
   rotation-balanced pairs and the accuracy gate are structural: you cannot
   retrofit them onto a finished run. Cell coverage *can* be added another
   night against the same wrapper and the same analyzer.
3. **It puts the gate before the spend.** B1 costs 53 minutes and can void the
   following 6 hours. Running it last — as the protocol's own note admits is the
   historical habit — is how `SAME OUTPUTS: False` sat unresolved.
4. **It beats the harder baseline.** A claim of the form "m15 beats AMD's own
   documented DeepSeek-R1 config on MI350X at c32p, 6 rotation-balanced pairs,
   with the kernel/integration decomposition shown" is worth more than the same
   claim against a lazy default — and it is the claim the fairness auditor is
   least able to veto.

### Decision rules at the end of B0

Read both `startup_seconds.txt` files and both `c32p.json` manifests.

**On `S` (mean of the two):**

* `S ≤ 450 s` → run B with `o90p` co-resident (≈ 8.3 h + contingency).
* `450 < S ≤ 650 s` → run bare B, `c32p` only (≈ 8.5 h + contingency).
* `S > 650 s` → cut to arms {m15, native_tuned_tp} at **6** pairs (12 runs,
  ≈ 4.6 h) — **cut the control arm, never a pair, and never to a pair count that
  is not a multiple of the arm count.** The decomposition then has no
  kernel/integration split, which must be stated in REPORT.md as a limitation of
  the run rather than quietly omitted.

**On `native_tuned_tp`:**

* **It did not start** (`STARTUP FLAG REJECTED` in the log): AMD's command
  targets vLLM 0.14.0rc2 and we run 0.25.1. Remove the rejected flag *only*,
  record the removal in ARMS.md as a deviation, and re-run B0's second arm. If
  two or more flags are rejected, fall back to `native_default` as the headline
  baseline for the night and say so in REPORT.md — a weaker baseline honestly
  labelled beats a tuned baseline that never ran. In v5.1 a failed arm voids its
  pair instead of killing the driver, so B2 will survive an intermittent
  failure, but a systematically broken arm must be replaced, not retried.
* **Its `c32p` is slower than 8k tok/s**: recompute B2 with its measured
  runtime before starting; at 5k tok/s the block is 7.5 h, not 5.8 h, and the
  night no longer closes with contingency — drop to the 2-arm cut above.

**On `OPEN_LOOP_BASE_RATE`:** it is `min(m15_ceiling, ntp_ceiling)` where each
ceiling is `result.completed / result.wall_seconds` from that arm's B0 `c32p`.
Using m15's own ceiling for every arm is what the v5.0 sketch did, and it offers
the slower baseline a load it cannot absorb — the analyzer would then report the
baseline's queue as a candidate win.

### Invocation sketch

```bash
export PROMPT_SOURCE=qsl
export QSL_PKL=/path/to/mlperf_deepseek_r1_qsl.pkl     # required with qsl
export PACKET=... N2=... M15_SOURCES=... DHK_ROOT=... CLIENT=.../bench_exact_token_ids_v3.py
export M23_PATCH=$HOME/eplb_campaign/m23_patch.py
export ARM_COOLDOWN=240

# B0 — calibration of BOTH arm families (also yields OPEN_LOOP_BASE_RATE).
# ALLOW_UNBALANCED_PAIRS=1 because one pair cannot balance two positions; these
# numbers calibrate, they are never quoted as a ratio.
RUN_TAG=b0 EPLB_PREWARM=0 ALLOW_UNBALANCED_PAIRS=1 \
  bash run_m15_campaign_eplb_v5.sh \
    --arms m15,native_tuned_tp --cells c32p --pairs 1

# B1 — accuracy gate first.  `stock` is required: it is m15's numerics-class
# peer, and without it the gate can only return PASS_WEAK.
RUN_TAG=b1 EPLB_PREWARM=0 \
  bash run_m15_campaign_eplb_v5.sh \
    --arms m15,stock,native_tuned_tp --cells c32p --pairs 0 --accuracy-pass

# B2 — the headline matrix (add ,o90p to --cells if B0 said S <= 450)
RUN_TAG=b2 EPLB_PREWARM=0 \
  OPEN_LOOP_BASE_RATE=<min(m15, native_tuned_tp) completed/wall_seconds> \
  OPEN_LOOP_BASE_RATE_SOURCE="min of m15 and native_tuned_tp c32p ceilings, \
    \$ROOT_b0/pair_01/{1_m15,2_native_tuned_tp}/c32p.json" \
  bash run_m15_campaign_eplb_v5.sh \
    --arms m15,stock,native_tuned_tp --cells c32p --pairs 6

# analysis — one --campaign per ROOT; the analyzer keeps them separate and
# never divides an arm in one root by an arm in another.
python3 analyze_campaign_v5.py \
  --campaign $ROOT_b1 --campaign $ROOT_b2 \
  --candidate m15 --control stock \
  --out-md REPORT_TABLES.md --out-json campaign_v5.json
```

Notes on the sketch:

* **`RUN_TAG` differs per step.** `ROOT` is derived from `DATE_TAG` and
  `RUN_TAG`, and the container names from `RUN_TAG` too. v5.0's three commands
  shared one tag, so the third either hit the name-collision refusal or wrote
  into the first run's `pair_01` and promoted a calibration arm into the
  headline matrix. v5.1 refuses a ROOT whose recorded invocation spec differs,
  but distinct tags are the intent, not the backstop.
* **B0 runs 1 pair with the imbalance override.** A balanced 2-arm calibration
  would need 2 pairs and cost 1.3 h instead of 0.66 h, to produce a number that
  is only ever used as a rate ceiling. The override stamps
  `POSITION_BALANCE=UNBALANCED` into that root's receipt, and the analyzer turns
  every policy row from that root into "no claim", which is the correct outcome
  for a calibration run.
* Also read `$ROOT_b0/.../c32p_client.log` for the client's
  `duplicate_prompts_in_cell` line: with QSL prompts and the v3.1 stride it
  should be 0 for a 1024-prompt cell.
* **B1 uses `--pairs 0`** — no measured pairs, straight to the accuracy pass.
  0 is a multiple of every arm count, so the balance preflight passes.
* The B1 root is passed to the analyzer so its `accuracy/ACCURACY_GATE.txt` is
  read; the policy table refuses to say "SHIP" when the gate is absent or failed.

---

## 5. What this schedule does **not** buy

State these in REPORT.md rather than letting a reader assume them:

* **No large-batch claim.** c512/c512p are not run, so nothing is known about
  conc 512 — the one regime where AMD's own recommendation is DP+EP and where
  our parallelism finally matches theirs.
* **No shipped-default ratio at all.** `native_default` is not in the matrix;
  the headline is against the vendor-*tuned* config only. That is the harder
  baseline, but it is not the same claim as "beats what ships".
* **No full utilization sweep** unless the o90p add-on lands: one open-loop
  point (90% of the *slowest arm's* ceiling) shows whether the dummy-batch/pad
  advantage survives arrival-driven traffic, but not how it degrades. Note the
  open-loop cells are ranked by TTFT p99, not throughput — at a fixed arrival
  rate throughput is the same for every arm that keeps up.
* **No accuracy statement beyond the determinism cells:** `c1det` is 32 prompts
  at conc 1 and `c8det` 64 at conc 8, both ISL 4096 / OSL 8. Together they prove
  that m15 and stock agree token-for-token in a deterministic regime and record
  how far the tuned-native arm diverges. They do not prove agreement under
  continuous batching, where the arms are not expected to agree and where even
  stock-vs-stock diverges.
* **Prefix caching is OFF on every arm** (v5.1). Our banked m15 numbers were
  taken with it on, so tonight's absolute tok/s are not directly comparable to
  the camp3 receipts; the ratios are, and the ratios are the claim.

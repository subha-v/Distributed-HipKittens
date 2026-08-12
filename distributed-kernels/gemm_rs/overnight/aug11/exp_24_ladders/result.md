# exp_24 — external ladders refresh (paper Q6): result

**STATUS: full ladder NOT YET RUN.** The `LAD_QUICK=1` end-to-end validation has
run and is clean — see **§0** for its results, which are the only measurements in
this file. §2–§6 are still stubs and get filled from the full six-shape run.

The quick run is a **validation, not a result**: one shape, 8 graded iterations,
and only 2 of the 5 rotation positions, so its ratios are position-biased by
construction and are not quoted as findings.

Launch:

```bash
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
bash $ON/aug11/exp_24_ladders/run_ladders.sh all
```

Expected wall clock ~1.5–2.5 h. `LAD_QUICK=1` validates the whole pipeline on one
shape in ~6 min first; do that before committing to the full ladder.

---

## 0. `LAD_QUICK=1` validation run — 2026-08-12 04:40, shape 2 only

`LAD_QUICK=1 bash run_ladders.sh all`, shape index 1 (512×4096×12288),
8 graded iterations × 2 reps, 4 pipelined bursts × 10 calls × 2 reps,
instrument B skipped. **83 s for the shape, `rc=0`, all 8 rank exit codes 0,
node clean before and after, all 8 GPUs in `perf_determinism` throughout.**
Artifacts in `raw/ladder_quick/` and `ladders_quick.json` (48.9 kB), kept
deliberately separate from `raw/ladder/` so a validation run can never be
aggregated into a real ladder.

### 0.1 All five arms produced numbers, both protocols populated

| arm | protocol | best µs | median µs | mean µs | rsd% | n |
|---|---|---:|---:|---:|---:|---:|
| ours | graded | 159.71 | 162.67 | 169.67 | 12.1 | 128 |
| ours_null | graded | 158.66 | 163.00 | 166.94 | 5.6 | 128 |
| reference | graded | 217.47 | 233.67 | 251.55 | 21.0 | 128 |
| rank1 | graded | 152.15 | 160.15 | 173.04 | 26.1 | 128 |
| harness_floor | graded | 84.51 | 90.38 | 102.15 | 53.7 | 128 |
| ours | pipelined | 72.37 | 73.94 | 74.88 | 10.4 | 64 |
| ours_null | pipelined | 71.57 | 72.80 | 73.51 | 3.6 | 64 |
| reference | pipelined | 96.80 | 98.14 | 109.77 | 24.6 | 64 |
| rank1 | pipelined | 59.29 | 60.83 | 60.92 | 1.6 | 64 |
| harness_floor | pipelined | 5.60 | 6.14 | 6.43 | 10.0 | 64 |

All four gated arms passed correctness at **both** `1e-2` and `2e-3`,
`max|diff| = 9.766e-4` for every one. No `SHIM_WAS_CALLED`, no memory access
fault, no traceback in any rank's stderr. (`rank3.stderr` was 75 B against 20 B
elsewhere; it is hipify's "Successfully preprocessed all matching files" — that
rank won the race to build the torch extension. Benign.)

### 0.2 Rotation really rotates

```
rep 0: order=[ours, ours_null, reference, rank1, harness_floor]  protos=(graded, pipelined)
rep 1: order=[ours_null, reference, rank1, harness_floor, ours]  protos=(pipelined, graded)
```

Arm order rotates and protocol order flips, as designed. **With `LAD_REPS=2` only
2 of the 5 positions are covered**, and it shows: `ours` was first in rep 0's
graded block and carries **rsd 12.1% against its own twin's 5.6%**, with a worst
sample of 247 µs against the twin's 197. That is the cold-start penalty landing on
whichever arm leads, which is precisely what `reps = 5` exists to balance. The
quick run's ratios are therefore position-biased and are not results.

### 0.3 `harness_floor` — the constant is real, and it is 90 µs

**Graded: 84.51 best / 90.38 median / 102.15 mean µs.** Pre-registered band was
57–99 µs; the median lands at 90.38, within 2% of the ~92 µs the dispatch quoted
from memory. **Pipelined: 5.60 best / 6.14 median µs**, against the independently
recorded "~5–7 µs host path on a 4.86 µs floor". Both halves of caveat §8.2 are
now measured on this instrument rather than remembered, and they agree with the
remembered values.

It is **not** near zero, so the arm is measuring what it was built to measure.

Two consequences for the full run:

- **The floor's own rsd is 53.7%.** Subtracting a 90 µs quantity with that spread
  from a 160 µs graded number makes the netted table materially noisier than the
  raw one. The netted ladder must be read as a direction, not a precision figure,
  and §2.2 will say so.
- `floor_is_shape_independent` is `null` with one shape. The full six-shape run is
  what tests whether netting is legitimate at all; if the floor scales with shape
  size, §2.2 gets dropped rather than published.

### 0.4 The ladder's own noise floor — `ours` vs `ours_null`, shape 2

| statistic | protocol | ours | ours_null | **floor** | published | aug11 re-measured |
|---|---|---:|---:|---:|---:|---:|
| best | graded | 159.71 | 158.66 | **0.66%** | 0.56% | 0.93% |
| median | graded | 162.67 | 163.00 | **0.20%** | 0.56% | 0.93% |
| mean | graded | 169.67 | 166.94 | **1.64%** | 0.56% | 0.93% |
| best | pipelined | 72.37 | 71.57 | **1.12%** | 0.56% | 0.93% |
| median | pipelined | 73.94 | 72.80 | **1.58%** | 0.56% | 0.93% |
| mean | pipelined | 74.88 | 73.51 | **1.87%** | 0.56% | 0.93% |

**Verdict: the published floor holds on shape 2 for best and median, and fails for
the mean** — 0.20–0.66% graded against a published 0.56% and an aug11 0.93%, but
1.64–1.87% on the means, i.e. 3× the published figure. That is HANDOFF's
"means are unusable on this node" reproduced on a fresh instrument, and it is why
best and median are the reported statistics and the mean only ever appears beside
this table.

**Coverage caveat, and it matters.** Shape 2 has the *tightest* of the six
published floors (0.56%). The decision-relevant rows are shape 1 (published 1.34%,
aug11 **3.48%**) and shape 6 (published **4.28%**), and **neither is measured
yet** — the full run measures all six in the same pass that produces the ratios.
Do not generalize a 0.2–1.9% floor from shape 2 to the ladder.

### 0.5 A pre-registered expectation is already falsified, in sign

`plan.md` §6 expectation 4 predicted the pipelined `ours/rank1` ratio would be at
least 0.08 **below** graded. On shape 2 it is **0.17 above**:

| ratio, ours/rank-1 | best | median |
|---|---:|---:|
| graded, raw | 1.0496 | 1.0157 |
| graded, netted | — | 1.0361 |
| **pipelined** | **1.2207** | **1.2155** |

The mechanism is caveat §8.2 running the other way from how the expectation was
written: the ~90 µs constant is added **identically to both arms**, so it is a
larger fraction of a 160 µs graded call than of a 72 µs pipelined one and it
**compresses the ratio toward 1**. The graded headline therefore *flatters* us,
and the netted median (1.036) sits between the two as it should.

If this holds across the other five shapes it is a finding for the paper's Q6, not
a measurement artifact: part of the 1.098× graded gap we have been tracking is
protocol dilution rather than kernel parity, and the kernel-to-kernel deficit is
larger than the graded number implies. It also means expectation 4 should be
re-registered with the opposite sign before the full run, which is done here rather
than after seeing six shapes.

### 0.6 Fixes the quick run forced, all pre-launch

| finding | fix |
|---|---|
| **Duration-based warmup with a per-rank deadline gives per-rank call COUNTS.** `reference` calls `reduce_scatter_tensor` and rank-1 calls its own `dist_barrier`, so unequal counts deadlock. Caught by review before the first launch, not by a hang. | rank 0 owns the stop decision and broadcasts it; every rank runs the same whole number of fixed-size blocks. Duration-based sizing, identical collective counts. |
| Warm counts were `ours 965 / ours_null 3545 / reference 20 / rank1 20 / harness_floor 11410`. Not a hang — the 400 ms window was consumed by one-time RCCL channel setup and Triton JIT for the two arms that landed on the 20-call minimum, so they got no steady-state warm work at ramped clocks. | the warm clock now starts **after** the first block, so setup cannot eat the window. |
| A one-shape run would have tripped the aggregator's strict six-shape assertion, and `--lenient` would have hidden it. | `--expect-a` / `--expect-b` carry the intended counts from the driver, so a quick run is strictly checked against *one* shape instead of waved through. `partial_run` is recorded in the JSON. |
| Quick samples living in `raw/ladder/` could be aggregated into a real ladder. | quick output goes to `raw/ladder_quick/` and `ladders_quick.json`. |
| `git HEAD` printed **empty** in provenance, reading as a failed command. | `/home/subvadla/dhk` is not a git worktree (it is scp-synced by `push_scoped.ps1`); provenance now says so explicitly and points at the module sha256 set, which did record correctly. |

### 0.7 Schema conformance

`ladders_quick.json` carries all ten documented top-level keys; both protocols
carry all five arms; every `per_shape` list has **six positionally-aligned
entries** with `null` for absent shapes, so a plot never has to guess which row is
which; row keys are exactly `shape / best_us / median_us / mean_us / worst_us /
rsd_pct / n / samples_us / rank0_us`; `samples_us` is populated (128 graded,
64 pipelined per cell). §7's schema matches the artifact as written.

Config under test, from `logs/provenance.txt`: `gemm_rs_mi300x.so`
`79599cce1087df00…`, `dhk_rt.so` `a97031b24f251da6…`, `submission.py ==
hk_submission.py`, `scored_shapes` NR column `56 / 32 / 32 / 32 / 32 / 48`.
Frozen rank-1 hash re-verified at run time by `patch_rank1.py`; staged copy
`abd5aa73…`, diff 4 lines, all inside the disclosed `packed_metadata` repair.

---

## 1. Verdict

*(fill in: one sentence, then the three geomeans and the two ratios, both
protocols, labelled.)*

## 2. The ladder

### 2.1 Graded per-call — **the competition's ranking statistic**, raw

| # | shape | ours | rank-1 | reference | ours/rank-1 | ours/reference |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 64×7168×18432 | | | | | |
| 2 | 512×4096×12288 | | | | | |
| 3 | 2048×2880×2880 | | | | | |
| 4 | 4096×4096×4096 | | | | | |
| 5 | 8192×4096×14336 | | | | | |
| 6 | 8192×8192×29568 | | | | | |
| | **geomean** | | | | | |

*(best / median / mean each in its own table or column set, statistic named.)*

### 2.2 Graded per-call — **netted**, with this run's measured harness constant removed

*(the `harness_floor` arm's per-shape median, then every arm minus it. State
whether the floor was shape-independent; if it was not, say so and drop the
netted table rather than publishing it.)*

### 2.3 Pipelined — the throughput number

*(same table. Never averaged with §2.1.)*

### 2.4 The null arm — the ladder's own noise floor, measured in this run

*(`ours` vs `ours_null`, per shape and on the geomean. Any ratio movement in
§2.1–2.3 smaller than this is not a result.)*

## 3. Against the prior denominator

Prior *(exp_14, same-run graded, best-of-arm)*: rank-1 **314.46 µs** vs ours
**345.15 µs** → **1.098× behind**, down from 1.784× when rank-1 first ran.
Reference GEMM+RCCL **438.15 µs**, beaten. Per-shape graded ratio
0.850 / 1.072 / 1.102 / 1.120 / 1.286 / 1.208.

**The ratio's noise floor is ±2%** (exp_14, rows 4/5/6, configuration unchanged
between runs: +1.4 / −2.1 / −1.9%). Do not report a ratio change smaller than
that as a change.

*(fill in the delta table and say which rows moved outside the floor.)*

## 4. Pre-registered expectations vs outcome

| # | pre-registered in `plan.md` §6 | outcome |
|---:|---|---|
| 1 | graded `ours/rank1` geomean in 1.02–1.10× | |
| 2 | shape 1 stays <1.0; shapes 5 and 6 stay >1.15 | |
| 3 | graded `ours/reference` stays <0.85 | |
| 4 | pipelined ratio at least 0.08 below graded | |
| 5 | `harness_floor` measures 57–99 µs, shape-independent | |
| 6 | `ours` vs `ours_null` <4.3% per shape, <2% geomean | |
| 7 | instrument B absolutes 25–45% above A; both agree on arm ORDERING | |

## 5. Correctness, both arms, both tolerances

*(every gated arm at `1e-2` and `2e-3`, with `max|diff|`. A number for an
unverified kernel is worthless, and the instrument refuses to time a shape where
any gated arm fails at `1e-2`.)*

## 6. Integrity checks that must be green before any number is quoted

| check | why it matters | result |
|---|---|---|
| frozen rank-1 sha256 == `7940fcb8…f0dc5` | the submission is unmodified; `tools/patch_rank1.py` enforces it and refuses to emit on mismatch | **verified 2026-08-12: MATCH** |
| `SHIM_WAS_CALLED` absent from every stderr | repair #3 is still dead code, so it is still behaviour-preserving | |
| all six shapes in rank-1's `__conf`, `online_config` and `online_config_group` | `launch_triton_kernel` silently returns `origin((a,b,bias))` — torch — for any shape it has no tuned config for, so a fast rank-1 number can be torch wearing its name | |
| no `Memory access fault` in any rank-1 stderr | the buffer-ops knob took and `.triton` was not stale | |
| clocks pinned at 1900 for the whole run | idle sclk is ~125 MHz; unpinned short runs are unrepeatable by up to 60% | |
| `submission.py` == `hk_submission.py` | the arm measured is the arm built | |
| module sha256 + `scored_shapes` recorded | a stale build cannot masquerade as the exp_23 winner | |

## 7. `ladders.json` schema (verbatim)

```
{
  "experiment": "exp_24_ladders",
  "node": {"host":, "arch":, "container":, "provenance": <logs/provenance.txt>},
  "config": {"shapes": [6 labels], "arms": [...], "ratio_arms": [...],
             "null_arms": [...], "protocols": ["graded","pipelined"],
             "instrument_a":, "instrument_b":, "bias_forced_all_shapes":},

  "protocols": {
    "<graded|pipelined>": {
      "arms": {
        "<ours|ours_null|reference|rank1|harness_floor>": {
          "per_shape": [ { "shape": "MxNxK",
                           "best_us":, "median_us":, "mean_us":, "worst_us":,
                           "rsd_pct":, "n":, "samples_us": [...],
                           "rank0_us": {"best_us":,"median_us":,"mean_us":,"n":} } ],
          "geomean_us":,            # over per-shape BEST
          "geomean_median_us":,
          "geomean_mean_us":,
          "shapes_present":
        }
      }
    }
  },

  "graded_netted": {
    "floor_source": "harness_floor arm, measured in this run",
    "floor_per_shape_us": [...],       # the arm's per-shape median
    "floor_spread_pct":,
    "floor_is_shape_independent":,     # false invalidates the netted table
    "arms": {"<arm>": {"per_shape": [{"shape":,"best_us":,"median_us":,"mean_us":}],
                       "geomean_us":, "geomean_median_us":}}
  },

  "correctness": {"<shape label>": {"<arm>": {"allclose_1e-2":,
                                              "allclose_2e-3":,
                                              "max_abs_diff":}}},

  "ratios": {
    "ours_vs_rank1":     {...},   # alias of ours_vs_rank1__graded__best
    "ours_vs_reference": {...},   # alias of ours_vs_reference__graded__best
    "ours_vs_rank1__<proto>__<best|median>":     {"statistic":,
        "per_shape": [{"shape":,"ratio":,"ours_us":,"other_us":}], "geomean":},
    "ours_vs_reference__<proto>__<best|median>": {...},
    "ours_vs_rank1__graded_netted__median":      {...},
    "ours_vs_reference__graded_netted__median":  {...},
    "null_floor__<proto>__best": {...}   # ours vs ours_null: the noise floor
  },

  "evaluator_crosscheck": {
    "rotations": {"rot<N>": {"<arm dir>": {
        "source":, "check": "pass|fail",
        "per_shape": [{"shape":,"best_us":,"mean_us":,"median_us": null,
                       "worst_us":,"std_us":,"rsd_pct":,"runs":,"samples_us": []}],
        "geomean_us":, "geomean_mean_us":, "statistic_note":}}},
    "parse_errors": [...]
  },

  "prior_denominator": {...},        # exp_14's numbers, for the delta
  "caveats": [ 12 strings, reproduced in section 8 ]
}
```

`median_us` and `samples_us` are `null`/`[]` under `evaluator_crosscheck` by
construction: `eval.py`'s `calculate_stats` emits only
`runs/mean/std/err/best/worst`. Times in the popcorn text are **nanoseconds** and
are converted to microseconds on parse.

## 8. Caveats — these decide whether the numbers mean anything

**8.1 Topology mismatch.** Our harness runs **one process driving 8 devices** with
peer access; the evaluator's topology is **one process per rank** with
`torch.distributed`. That is address-space equivalent for this kernel — the
descriptor holds each rank's own base and `translate_peer` only computes
`peer_base + (local_ptr − local_base)` — but it is **not** the evaluator's
topology. This caveat applies to every comparison against rank-1's evaluator
numbers. (Instrument A is itself one-process-per-rank, so it does not carry this
caveat; the caveat attaches to any number imported from our own single-process
harness, including the `time_pipelined` geomean the ratchet is expressed in.)

**8.2 ~92 µs of every graded call is harness machinery both arms pay.** The
evaluator's own trailing `synchronize + barrier` measures **57–79 µs with an
*empty* timed region**, plus 15–20 µs of input clone. Our own host path is
~5–7 µs against a 4.86 µs floor. **The graded ladder is reported both raw (§2.1)
and netted (§2.2), labelled.** Netting it out makes our true kernel-to-kernel
ratio *worse* than the headline, and it is why the small shapes are ±10% noisy
regardless of the kernel. This run measures the constant with the `harness_floor`
arm instead of quoting it, so the netting can falsify itself.

**8.3 rank-1 needs `AMDGCN_USE_BUFFER_OPS=0`** — Triton 3.6.0 lowers its peer
stores to `buffer_store_dwordx2`, whose voffset is 32-bit, truncating element
offsets of −6.6e8 … −4.4e9. **`TRITON_CACHE_DIR` must be cleared whenever that
knob changes**, because the cached `hsaco` has `buffer_store` baked in and leaving
the cache makes the knob look ineffective. This is a disclosed repair, argued
behaviour-preserving in exp_10 §2.6: the kernel cannot be correct at all without
64-bit peer addressing, so 64-bit addressing is necessarily what its original
environment produced. **Disclosed residual bias and its direction:** the knob is
global to rank-1's Triton kernels, so its A/B operand loads — which legitimately
are within 2 GB — also lose buffer-op lowering, which can only *penalize* rank-1.
The bias runs against the arm we are chasing, so it cannot manufacture a win for
us. Our arm is hand-written HIP compiled with `hipcc` and contains no Triton, so
the knob cannot affect it at all.

**8.4 rank-1's warm pass is mandatory.** The evaluator arm runs
`benchmark`(warm) → `test` → `benchmark`(bench) in that order, because
`eval.py`'s test mode hardcodes a 60 s per-rank timeout and a cold compile of
rank-1's kernel exceeds it. This pass order is `run_rank1_bench3.sh`'s and it is
not "simplified".

**8.5 The reference arm's absolute numbers are junk; only same-instrument ratios
are meaningful.** Its own header records relative standard deviations of
**12–93%** — the saved historical run shows RSDs of 75%, 47%, 68%, 60%, 115% and
100% across the six shapes. Quote the ratio, never the absolute.

**8.6 `rocm-smi --showpids` preflight runs for all three arms.** Only
`run_reference_arm.sh` does this itself; `run_ours_evaluator.sh` and the rank-1
driver do not, so `run_ladders.sh` adds it plus a bounded KFD-fd drain wait before
every arm, and aborts on a dirty node unless `LAD_ALLOW_DIRTY=1` — in which case
the run is explicitly not reportable as timing.

**8.7 The published rank-1 score of 413.139 µs is on a different machine under a
different protocol** and is not comparable to anything measured here. The only
valid denominator is rank-1 measured on this node.

**8.8 `tools/run_rank1_bench3.sh` was REPAIRED on 2026-08-12 and is the rank-1
cross-check driver. The repair is disclosed here and argued
behaviour-preserving.**

As found by this experiment's dry-run inspection, bench3 could not run rank-1 at
all — it predated exp_10's repair #6 — and it had **four** independently fatal
defects, every one found by reading the file rather than by a failed run, which is
how a "runs but measures nothing" tool survives:

| # | defect | why fatal |
|---:|---|---|
| a | `AMDGCN_USE_BUFFER_OPS=0` absent, and the env was built as one `ENVS` string interpolated into `bash -c`, so the knob could not be injected from outside either | the arm faults: Triton 3.6.0 lowers rank-1's peer stores to `buffer_store_dwordx2`, 32-bit voffset, truncating a −4.4-billion-element offset |
| b | `PYTHONPATH` pointed at `$ON/compat`, which does not exist (the tree is `$ON/tools/compat`) | repair #3's `sitecustomize.py` was silently absent |
| c | invoked `$ON/patch_rank1.py`, which does not exist (the tool is `$ON/tools/patch_rank1.py`); the `if [ $? -ne 0 ]` guard also tested the wrong command's status | staging exits non-zero |
| d | ran in container `dhk-eval`, which is **root** | every artifact it wrote under the repo came out root-owned and broke later `sed`/`scp` steps |

The repaired driver passes the knob as `docker exec -e` flags (visible and
overridable), points `PYTHONPATH` at `$ON/tools/compat`, calls
`$ON/tools/patch_rank1.py`, and runs in `dhk-gemmrs` as uid 15523 — an env and
container arrangement that now mirrors `experiments/exp_10_rank1/r1_eval.sh`, the
driver that actually produced exp_10's measured comparison. It keeps the mandatory
`warm → test → bench` pass order and adds the node-clean preflight the arm never
had. Verified mechanically before use: the knob is present and passed via `-e`,
`tools/compat` is on `PYTHONPATH`, `tools/patch_rank1.py` is the path called,
`dhk-eval` appears only in a comment, and the three `go` calls are still in
`warm → test → bench` order.

**Behaviour-preserving argument.** Every change is to **how the arm is launched**,
never to what it computes: a container identity and uid, two corrected filesystem
paths, and an environment variable that — per exp_10 §2.6 — *restores* the 64-bit
peer addressing the kernel requires and cannot be correct without. No algorithm,
tiling, config, launcher or data movement is touched. **The frozen submission is
still untouched and still hashes to `7940fcb8…f0dc5`** (re-verified at run time);
patching remains a copy behind `patch_rank1.py`'s hash gate, whose only diff is
the 4-line `packed_metadata` repair. The one residual bias, disclosed in §8.3,
runs *against* rank-1 and so cannot manufacture a win for us.

`tools/` is not this experiment's to edit and was not edited by it; the repair was
made by the tree's owner after this experiment reported the defects.

**8.9 Means alone are unusable on this node**, even same-run interleaved, because
the bias is **per-allocation and partly allocation-ORDER**, not positional:
exp_14 measured **4.28% between identical arms** on shape 6 while positional
residual stayed under ±0.47%. Best and median are reported alongside every mean,
and the `ours_null` arm measures that spread in this same run (§2.4).

**8.10 The ratio's own noise floor is ±2%**, from exp_14's rows 4/5/6 whose
configuration did not change between runs. Changes smaller than that are not
reported as changes.

**8.11 Bias is forced on for all six shapes, for all arms identically**, because
that is what the official evaluator actually does: its cases parser does
`int(val)` and keeps the raw string on `ValueError`, so `has_bias: False` becomes
the truthy **string** `"False"` and `reference.generate_input` builds a bias for
all six. `ref_kernel` adds it too, so the evaluator stays self-consistent.
rank-1's cached fast path dereferences `bias.data_ptr()` unconditionally and
raises `AttributeError` on its **second** call without one. A genuine
`has_bias=False` ladder is not obtainable for rank-1 without repairing that line,
and is out of scope.

**8.12 Instrument B has no warmup and no fixed iteration count.** `eval.py` does
one obligatory correctness call and then enters the timed loop, and the loop is
adaptive — it breaks at `err/mean < 0.001`, or `mean·runs > max_time_ns`, or
120 s of wall. It emits no median and no raw samples. That is why it is the
cross-check and `ladder_mp.py` is the ladder; the two are never merged into one
table.

## 9. Staging validation record (this dispatch, no GPU)

| item | result |
|---|---|
| frozen rank-1 sha256 | **MATCH** `7940fcb8…f0dc5`, `-rw-rw-r-- subvadla`, 64863 B |
| `tools/patch_rank1.py` | present |
| `$ON/patch_rank1.py` (what bench3 calls) | **missing** — see §8.8 |
| `$ON/compat` (what bench3 puts on PYTHONPATH) | **missing**; real tree is `$ON/tools/compat` — see §8.8 |
| evaluator source tree (`eval.py`, `task.py`, `utils.py`, `reference.py`, `submission.py`, `cases.txt`) | all present |
| iris at rank-1's hardcoded `python3.10` path | present and importable in **both** `dhk-gemmrs` and `dhk-eval` (`import iris`, `import iris.hip` both OK) |
| compat shims (`tools/compat/bin/sudo`, `tools/compat/sitecustomize.py`) | present |
| `harness/submission.py` == `hk_submission.py` | yes |
| `bash -n` on all six shell files, `py_compile` on both Python files | pass |
| rotation table: every arm first exactly once, protocol order flipped per rep | verified by assertion, not by inspection |
| null arm is the same file under two module names | verified by assertion |
| parser vs saved historical popcorn | see §10 |
| clock pin | `set_clocks.sh pin 1900` took: all 8 GPUs report `Performance Level: perf_determinism`, valid sclk range 500–1900 |
| node state at the time of writing | **dirty** — pid 2667100 on all 8 devices, 8.7 GB; no GPU job launched. It had drained by the end of the dispatch, but the dispatch's scope forbade running |

**Do not misread an idle sclk of 120 MHz as a failed pin.** `set_clocks.sh pin`
uses `rocm-smi --setperfdeterminism`, which caps the sclk *ceiling*; the clock
still drops to ~120 MHz with no work on the device. The state to check is
`--showperflevel == perf_determinism`, not the instantaneous frequency. This cost
a confused minute during validation and would cost more during a run.

## 10. Parser validation against saved historical output

`python3 ladders.py --fixture` against the three saved `compbench/*/
benchmark.popcorn.txt` files, no GPU touched.

**`ours` — parsed cleanly, 6/6 shapes** (this is a stale Aug-11 12:35 run, quoted
only to show the parser working):

| # | shape | best µs | mean µs | worst µs | rsd% | runs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 64×7168×18432 | 251.42 | 293.26 | 2989.28 | 93.4 | 100 |
| 2 | 512×4096×12288 | 246.09 | 279.27 | 832.65 | 26.1 | 100 |
| 3 | 2048×2880×2880 | 262.23 | 286.37 | 821.37 | 26.7 | 100 |
| 4 | 4096×4096×4096 | 1115.83 | 1160.45 | 2482.91 | 11.9 | 100 |
| 5 | 8192×4096×14336 | 913.61 | 979.43 | 2103.22 | 16.2 | 100 |
| 6 | 8192×8192×29568 | 7155.52 | 9021.82 | 18389.71 | 42.8 | 100 |
| | geomean | **700.70** | 788.59 | | | |

**`reference` — parsed cleanly, 6/6 shapes**, geomean(best) **511.60 µs**,
geomean(mean) 676.73 µs, `check: pass`. Its per-shape RSDs are
**75 / 47 / 68 / 60 / 115 / 100 %** — which is caveat §8.5 measured rather than
remembered, and slightly worse than the 12–93% the dispatch quoted. Only
same-instrument ratios from this arm mean anything.

**`rank1` — the parser REFUSED it, and that is the validation that matters.**
The saved file is exp_10's evaluator run that hit its 1500 s wall after three
shapes (`exit=143`, exp_10 §4). It still carries `benchmark-count: 6` at the top,
so a naive parser would have emitted three numbers, three nulls and a geomean over
half a ladder. The error raised was:

> `benchmark-count says 6 but shapes [3, 4, 5] have no usable block. This is the
> truncated-run hazard (an evaluator that hit its timeout wall still emits the
> count header); refusing to emit a partial ladder.`

Two further negative controls, both behaving:

| control | outcome |
|---|---|
| a `test.popcorn.txt` (has `test-count:`, no timing blocks) | rejected: *"no `benchmark-count:` line — this is not a benchmark popcorn file"* |
| a hand-truncated `benchmark.popcorn.txt` (first 14 lines) | rejected: shapes `[2,3,4,5]` unusable |
| the aggregator against an empty tree | refused, nonzero rc, no `ladders.json` written |

The parser also validates each block's `spec` against the canonical graded shape
*at that index*, so a `cases_bench.txt` in the wrong order, or a different shape
set, is an error rather than a mislabelled row.

## 11. Files

| file | what |
|---|---|
| `plan.md` | arms, protocols, interleaving, pre-registered expectations, denominator |
| `design.md` | why two instruments, arm table, timed regions, staging, failure modes |
| `run_ladders.sh` | orchestrator: preflight, clock pin, provenance, staging, both instruments, rotation, raw capture |
| `ladder_mp.py` | Instrument A: five arms, one pool per shape, both protocols |
| `ladders.py` | parser + aggregator → `ladders.json`; `--fixture` validates with no GPU |
| `validate_dry.sh` | the no-GPU validation: syntax, rotation assertions, the fixture test and its three negative controls |
| `probe_stage.sh`, `probe_fixtures.sh`, `probe_protocol.sh`, `probe_rank1_driver.sh`, `probe_clocks.sh` | the read-only staging probes |
| `logs/provenance.txt` | config fingerprint captured at run time |
| `raw/ladder/`, `raw/eval/rot<N>/<arm>/` | every arm's raw output, copied out of `compbench/` before the next arm's `rm -rf` |

# exp_24 — external ladders refresh (paper Q6): result

**STATUS: NOT YET RUN. This is a stub.** The driver is written and its staging is
validated; the GPUs were held by another agent's job (`rocm-smi --showpids`
pid 2667100, all 8 devices, 8.7 GB) for the whole of the dispatch that wrote this,
so no GPU job was launched. Every number below marked *(prior)* is a quoted
denominator, not a measurement of this experiment. Fill in §2–§6 after the run.

Launch:

```bash
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
bash $ON/aug11/exp_24_ladders/run_ladders.sh all
```

Expected wall clock ~1.5–2.5 h. `LAD_QUICK=1` validates the whole pipeline on one
shape in ~6 min first; do that before committing to the full ladder.

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

**8.8 `tools/run_rank1_bench3.sh` cannot run rank-1 on this node and was not
used.** Three defects, all consistent with it predating exp_10's repair #6:
it omits `AMDGCN_USE_BUFFER_OPS=0` and builds its own `ENVS` string passed
explicitly to `docker exec`, so the knob cannot be injected from outside; it sets
`PYTHONPATH` to `$ON/compat`, which **does not exist** (the tree is at
`$ON/tools/compat`), so repair #3's `sitecustomize.py` is absent; and it invokes
`$ON/patch_rank1.py`, which **does not exist** either (the tool is at
`$ON/tools/patch_rank1.py`), so its staging step exits non-zero. It also runs in
`dhk-eval` rather than `dhk-gemmrs`, and the two containers have separate
filesystems. The rank-1 evaluator arm is therefore driven by
`experiments/exp_10_rank1/r1_eval.sh` — the driver that actually produced exp_10
§4 — keeping bench3's pass order. **Nothing under `tools/` was edited.** This is a
one-line staging defect in a shared tool and belongs to whoever owns `tools/`.

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

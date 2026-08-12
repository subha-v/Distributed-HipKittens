# exp_24 — external ladders refresh (paper Q6): plan

**Status: PLANNED, NOT RUN.** This dispatch wrote the driver and validated
staging only. Another agent held the GPUs for the whole of this dispatch
(`rocm-smi --showpids` showed pid 2667100 on all 8 devices, 8.7 GB), so **no GPU
job of any kind was launched here.** Every number below is a prior measurement
quoted as a denominator, not a result of this experiment.

---

## 1. The question

Where does our kernel stand, *at the exp_23 winner config*, against the two
external denominators that matter — the frozen rank-1 competition submission and
the reference `torch.matmul` + `reduce_scatter_tensor` baseline — on all six
graded shapes, under **both** protocols, measured on this node in one run?

This is the paper's Q6 column and the ratchet's level-1 read.

## 2. The denominator to beat

Last measured (exp_14, same-run interleaved, graded protocol, best-of-arm):

| | geomean µs | verdict |
|---|---:|---|
| **frozen rank-1** | **314.46** | the target |
| ours | 345.15 | **1.098× behind**, down from 1.784× when rank-1 first ran |
| reference GEMM+RCCL | 438.15 | beaten |

Per-shape graded ratio, ours / rank-1 (>1 = rank-1 faster):

| 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|
| **0.850** | 1.072 | 1.102 | 1.120 | **1.286** | **1.208** |

Shape 1 is a win. Shapes 5 and 6 carry the entire gap, and exp_14 proved both
are at their tile-table optimum and pinned against the LDS cap — so any movement
there this run comes from exp_23's order/granularity knobs, not from geometry.

**The ratio's own noise floor is ±2%**, measured in exp_14 on rows 4/5/6 whose
configuration did not change between runs (+1.4 / −2.1 / −1.9%). A ratio change
smaller than 2% is not a change and will not be reported as one.

For context and **not** as a denominator: rank-1's published score of
**413.139 µs** is from a different machine under a different protocol. It is not
comparable to anything measured here. The only valid rank-1 denominator is
rank-1 measured on this node.

## 3. The arms

| arm | what it is | driver |
|---|---|---|
| `ours` | our kernel at the exp_23 winner config | `harness/submission.py` (== `hk_submission.py`) |
| `ours_null` | **the same file, loaded under a second module name** | the null arm; measures the ladder's own noise floor in the same run that produces the ratios |
| `reference` | exp026's own `submission.py`: `torch.matmul` + `torch.distributed.reduce_scatter_tensor` | the literature's usual denominator |
| `rank1` | the frozen competition submission, hash-gated, four disclosed repairs | |
| `harness_floor` | returns a preallocated output tensor and nothing else | **not a kernel arm.** It measures the graded protocol's own constant *in this run*, so the netted-out ladder rests on a measurement rather than on a remembered 92 µs |

`ours_null` and `harness_floor` are excluded from the ratio tables and from the
correctness gate (the floor arm is deliberately wrong); they appear only as
`nulls` in the JSON and as spread columns in `result.md`.

## 4. The two protocols — never blended

| protocol | timed region | what it is |
|---|---|---|
| **graded per-call** | `clear_l2_cache(); synchronize; barrier;` **t0**; `custom_kernel(clone(data))`; `synchronize; barrier;` **t1** — verbatim `eval.py:_run_distributed_benchmark` lines 349–366 | **the competition's ranking statistic** |
| **pipelined** | one `synchronize; barrier`, then a burst of `burst` back-to-back `custom_kernel` calls on a pre-cloned input, then one `synchronize; barrier`; sample = elapsed / burst | the throughput number; the analog of our harness's `time_pipelined` |

Both are reported per shape, labelled, with their own geomeans. They are never
averaged together. The difference between them **is** the per-call harness tax
that §7 nets out.

The pipelined burst reuses one clone for all `burst` calls. All three
submissions treat `(x, w, bias)` as read-only, and the instrument verifies the
burst's final output against the oracle afterwards, so input mutation cannot pass
as a throughput result.

## 5. The interleaving scheme

Two instruments, because no single one can produce everything this experiment
needs. Full reasoning in `design.md` §1.

**Instrument A — `ladder_mp.py`, the ladder.** All five arms in the **same**
8-process pool, on the **same** inputs, alternating, one fresh pool per shape.
This is exp_10's `mp_vs_rank1.py` protocol generalized from two arms to five.
Rotation is complete rather than merely alternating:

- **arm order** is cyclically left-rotated by the rep index, and `reps` defaults
  to 5 (= the arm count), so **every arm is first exactly once**. Two-arm
  alternation, as used in exp_10/exp_14, only balances two positions;
- **protocol order** flips per rep (graded first on even reps, pipelined first
  on odd), so protocol order is not confounded with arm order either;
- **shape order** is rotated across invocations by the driver.

One fresh pool per shape is not a convenience: rank-1 caches its compiled kernel
in module globals keyed by nothing, so the cache is valid only for the shape it
was built for. The official evaluator gets away with it by destroying the process
group between shapes.

**Instrument B — the official evaluator, three arms, rotated.** `eval.py`
itself, one whole-benchmark run per arm, arm order rotated across rotations so no
arm is systematically first. This is the competition's literal harness and is the
cross-check on Instrument A, not a substitute for it: `eval.py` reports only
`best/mean/std/err/worst` in nanoseconds — **no median and no raw samples** — and
its iteration count is adaptive (it stops at `err/mean < 0.001`, or 120 s, or
`max_repeats`), so neither the requested statistics nor a fixed 3 × 50 protocol
can be obtained from it. It also has **no warmup**: one obligatory correctness
call, then straight into the timed loop, which is most of why its absolute
numbers run 30–40% above the interleaved instrument's.

## 6. Pre-registered expectations

Written before any arm runs. Landing these as predictions is the point; being
wrong is a result.

1. **`ours / rank1` graded geomean lands in 1.02–1.10×.** exp_23's rungs (WGM
   order, RELEASE_GROUP=4) are already in the shipped config that produced
   1.098×, so the movement available here is whatever exp_23's winner adds on
   top. A number below 1.00 would be the night's headline and needs a re-run
   before it is reported.
2. **Shape 1 stays a win (<1.0) and shapes 5 and 6 stay the gap (>1.15).**
   Their geometry is at the tile-table optimum and cannot move this run.
3. **`ours / reference` graded geomean stays below 0.85** — the minimum bar,
   comfortably held at 0.788 last time (345.15 / 438.15).
4. **`ours / rank1` is materially better pipelined than graded**, because the
   per-call harness constant is a larger fraction of our smaller shapes and we
   pay a per-call host tax rank-1 does not (exp_12's residue). Concretely:
   pipelined ratio at least 0.08 below graded.
5. **The `harness_floor` arm measures 57–99 µs** on every shape, roughly shape-
   independent — that is the prediction that makes the netting in §7 legitimate.
   A floor that scales with shape size falsifies the whole netting argument and
   must be reported as such.
6. **`ours` vs `ours_null` differ by less than 4.3% per shape and less than 2%
   on the geomean.** exp_14 measured 4.28% between identical arms on shape 6.
   Any ratio movement inside the null spread is not a result.
7. **Instrument B's absolutes sit 25–45% above Instrument A's** for every arm,
   and the two instruments **agree on the ordering** of the three arms on every
   shape. Disagreement on ordering is the one outcome that invalidates the run.

## 7. The netting-out, stated up front

Every graded call contains ~92 µs of harness machinery that **both arms pay**:
the evaluator's trailing `synchronize + barrier` measures 57–79 µs with an
*empty* timed region, plus 15–20 µs of input clone. Our own host path is ~5–7 µs
against a 4.86 µs floor.

The graded ladder is therefore reported **twice: raw and netted**, with the
netting quantity taken from this run's own `harness_floor` arm rather than from
the remembered constant. Netting it out makes our true kernel-to-kernel ratio
**worse** than the headline, and it is why the small shapes are ±10% noisy
regardless of what the kernel does. Both numbers are labelled; neither is
presented alone.

## 8. Statistics

- Instrument A: `reps = 5` (complete arm-position balance) × `iters = 50` graded
  calls, and 20 bursts × 10 back-to-back calls pipelined, per arm per shape.
  Pooled over all 8 ranks: 2000 graded samples and 800 pipelined samples per arm
  per shape.
- **Duration-based warmup**, not fixed-iteration: each arm is warmed for at
  least `WARM_MS` (default 400 ms) *and* at least 20 calls before any timing.
  Idle sclk on this node is ~125 MHz against ~1900 pinned, and a fixed-iteration
  warmup produced a 60% wrong number once already.
- Clocks pinned (`tools/set_clocks.sh pin 1900`) before anything is timed, and
  the pinned state is recorded in the JSON.
- **Per-shape best AND median AND mean are all reported.** Means are unusable
  alone on this node even same-run interleaved, because the bias is
  per-allocation and partly allocation-*order*, not positional — exp_14 measured
  4.28% between identical arms on shape 6 while positional residual stayed under
  ±0.47%. The mean appears only next to the null-arm spread.
- Geomean is the ranking statistic and is reported for best, median and mean
  separately, each labelled with its statistic. Comparing an M7 *mean* to a
  recorded *best* manufactured a phantom 7.3% regression once.
- Deltas under 2% against the previous ladder are re-run before being called
  changes.

## 9. Caveats (the full list is reproduced verbatim in `result.md`)

Topology mismatch (one process / 8 devices vs one process per rank); the ~92 µs
graded harness constant and the raw-vs-netted requirement;
`AMDGCN_USE_BUFFER_OPS=0` plus a cleared `TRITON_CACHE_DIR` as a disclosed
rank-1 repair; rank-1's mandatory warm pass before `test`; the reference arm's
12–93% RSDs making only same-instrument ratios meaningful; `rocm-smi --showpids`
preflight on all three arms; and the non-comparability of the published
413.139 µs. `result.md` carries all seven with their consequences.

Three further caveats found by this dispatch's staging validation, all in
`design.md` §4 and `result.md`:

- **`tools/run_rank1_bench3.sh` cannot run rank-1 on this node today** — it
  predates exp_10's repair #6 and omits `AMDGCN_USE_BUFFER_OPS=0`, points
  `PYTHONPATH` at a `$ON/compat` that does not exist, and invokes
  `$ON/patch_rank1.py` which does not exist either. The rank-1 evaluator arm is
  driven by `experiments/exp_10_rank1/r1_eval.sh`, which is the driver that
  actually produced exp_10 §4, in the `warm → test → bench` order bench3
  prescribes. Nothing in `tools/` was edited.
- Arm staging is done by this experiment's own `stage_arms()` rather than as a
  side effect of the `tools/` drivers, because Instrument A needs the three arms
  staged *without* an evaluator run. It reproduces those drivers' `cp` lines and
  calls `tools/patch_rank1.py` at its real path.
- `eval.py` hardcodes `MASTER_PORT = "12356"`, so Instrument B's runs must be
  strictly serialized with a socket drain between them.

## 10. Deliverables

| file | what |
|---|---|
| `plan.md` | this file |
| `design.md` | the two instruments, the arm table, staging, failure modes |
| `run_ladders.sh` | the orchestrator: preflight, clock pin, provenance, staging, both instruments, rotation |
| `ladder_mp.py` | Instrument A: five arms, one pool, both protocols, per-rank JSON |
| `ladders.py` | the parser and aggregator → `ladders.json`; `--fixture` mode validates the popcorn parser with no GPU |
| `ladders.json` | plot-ready data; schema stated verbatim in `result.md` |
| `result.md` | verdict, numbers, confidence, and the full caveat list |
| `probe_*.sh` | the read-only staging probes this dispatch ran |

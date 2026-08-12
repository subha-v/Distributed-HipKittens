# exp_10 REMATCH — ours vs frozen rank-1, after the `has_bias` shape-table fix

Date: 2026-08-11, 20:34–20:42 CDT (node `banff-sc-cs47-05.dh170.dcgpu`, 8× MI300X,
gfx942, SPX). Measurement only: **no kernel, harness, ABI or `tools/**` file was
modified.** Everything written lives under `experiments/exp_10_rank1/`.

## Headline

| statistic | ours | rank-1 | ratio | verdict |
|---|---:|---:|---:|---|
| geomean of per-shape **best** | 381.69 µs | 308.25 µs | **1.238×** | rank-1 faster |
| geomean of per-shape **median** | 404.03 µs | 319.15 µs | **1.266×** | rank-1 faster |
| geomean of per-shape **mean** | 446.13 µs | 327.15 µs | **1.364×** | rank-1 faster |

The pre-fix comparison read **1.784× (best)** / 1.824× (mean). The fix closed
roughly **three quarters** of the gap on the best-of statistic (1.784 → 1.238).

Quote the **best or median** ratio. The mean-based 1.364× is an artifact: a
pool-level jitter tail landed on our arm on shape 3, and a single-shape re-run
showed the same tail landing on **rank-1's** arm instead. Means are not
trustworthy on this node; see
[Shape 3's tail](#shape-3s-tail-the-mean-is-contaminated-the-median-is-not).

## Protocol

`mp_vs_rank1.py` via `rematch_sweep.sh`: both arms in **one** 8-process pool,
alternating, **arm order reversed every rep** so neither is systematically
first; one fresh pool per shape (rank-1 caches compiled kernels in module
globals keyed by nothing). Per timed iteration, the evaluator's `full` protocol
verbatim: clone input, flush L2, `synchronize` + `barrier`, time, `synchronize` +
`barrier`. 20 untimed warmup calls per arm before anything is timed.

`iters=12 reps=2` → 24 samples per arm per rank, **192 pooled per arm per shape**
(the graded number is a per-call wall time, so every rank's samples count).
Reduced from the previous run's 15×2 because we are looking for multiples, not
percents; the whole six-shape sweep took **7.5 minutes** wall.

Clocks were in `perf_determinism` at 1900 MHz for every measurement.
`AMDGCN_USE_BUFFER_OPS=0` and `TRITON_CACHE_DIR=$ARM/.triton` were exported for
every shape — verified present in the driver, without which Triton 3.6.0 lowers
rank-1's peer stores to `buffer_store_dwordx2` and silently truncates its 32-bit
`voffset`. The Triton cache was **kept warm** (280 entries, all from earlier
runs with the knob already at 0); the knob did not change, and any stale-code
resurrection would have been caught by the per-shape correctness gate below.

## Per-shape results, 192 pooled samples per arm

| # | shape | arm | best | median | mean | sd% | worst | ratio (ours/r1) |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | 64×7168×18432 | ours | 181.19 | 191.64 | 205.44 | 19.8 | 376.80 | |
| 1 | | rank-1 | 179.17 | 183.59 | 199.16 | 38.2 | 586.93 | best 1.011× / med 1.044× |
| 2 | 512×4096×12288 | ours | 186.69 | 200.43 | 209.07 | 16.3 | 358.91 | |
| 2 | | rank-1 | 135.80 | 140.36 | 143.00 | 6.7 | 191.68 | best 1.375× / med 1.428× |
| 3 | 2048×2880×2880 | ours | 194.31 | 218.46 | 327.77 | 75.2 | 1771.89 | |
| 3 | | rank-1 | 157.61 | 162.76 | 163.47 | 2.5 | 175.33 | best 1.233× / med 1.342× |
| 4 | 4096×4096×4096 | ours | 312.94 | 324.73 | 335.18 | 12.2 | 523.13 | |
| 4 | | rank-1 | 265.69 | 282.21 | 282.96 | 2.7 | 303.85 | best 1.178× / med 1.151× |
| 5 | 8192×4096×14336 | ours | 762.76 | 795.68 | 819.56 | 8.4 | 1098.50 | |
| 5 | | rank-1 | 576.60 | 601.75 | 610.38 | 10.1 | 918.25 | best 1.323× / med 1.323× |
| 6 | 8192×8192×29568 | ours | 1970.98 | 2006.21 | 2038.83 | 4.3 | 2679.82 | |
| 6 | | rank-1 | 1460.07 | 1483.59 | 1524.72 | 7.6 | 2116.11 | best 1.350× / med 1.352× |

**We win nothing. We tie shape 1** (1.011× on best, 1.1% — inside this node's
noise). **We lose the other five by 1.18×–1.38×.**

The important structural change: the gap is now **uniform**. Pre-fix it ranged
from 1.03× to 4.39× and was dominated by two shapes falling off the tuned table.
A flat ~1.2–1.4× across every shape and every size class is the signature of a
systemic mainloop/egress deficit (E1/E2 in the mechanism bank), not a
configuration bug.

## Sanity check on shapes 4 and 6 — the fix IS live in the measured binary

Both landed exactly where predicted, which is the direct behavioural proof that
the rebuilt module carries the fix:

| shape | predicted | measured best | measured median | pre-fix best | speedup |
|---|---|---:|---:|---:|---:|
| 4 — 4096×4096×4096 | ~325 µs | **312.94** | **324.73** | 810.13 | **2.59×** |
| 6 — 8192×8192×29568 | ~1970–2000 µs | **1970.98** | **2006.21** | 6469.97 | **3.28×** |

Shape 6 did not read ~6400 µs, so the stale-binary failure mode did not occur.
This matters because the mtime check was **not** usable: `push.ps1` reset every
kernel source mtime to push time (20:26), and the `.so` was built at 19:14 while
the fix was committed at 20:19 CDT — an ordering consistent with
build → gate → commit but not *proof*. A first attempt to remove that doubt by
rebuilding failed because `build.sh` was invoked on the host, which has no
`hipcc`; the launcher correctly refused to measure rather than proceed, and the
existing `gemm_rs_mi300x.so` was verified byte-size intact (373560) before the
relaunch. The timings above then settled the question directly.

The harness loads from `harness/build/` (`HK_BUILD_DIR`), **not** from
`compbench/`; the only `.so` files under `compbench/` are Triton launcher stubs.

### Our arm, before vs after (best of 192)

| # | shape | pre-fix | post-fix | change |
|---:|---|---:|---:|---:|
| 1 | 64×7168×18432 | 181.78 | 181.19 | −0.3% |
| 2 | 512×4096×12288 | 189.69 | 186.69 | −1.6% |
| 3 | 2048×2880×2880 | 194.89 | 194.31 | −0.3% |
| 4 | 4096×4096×4096 | 810.13 | **312.94** | **−61.4%** |
| 5 | 8192×4096×14336 | 749.99 | 762.76 | +1.7% |
| 6 | 8192×8192×29568 | 6469.97 | **1970.98** | **−69.5%** |

Shape 1 was also declared `has_bias=false` and also fell through to
`generic_config`, yet it did not move. At m=64 the kernel is latency-bound —
181 µs for a 64-row GEMM is fixed cost (launch, IPC, barrier), so tile geometry
is not what sets its time. Nothing to fix there; it just means the fix mattered
on two of the three affected shapes.

### rank-1 reproduced across the two runs to within ±2%

Best, pre-fix → post-fix: 176.05→179.17 (+1.8%), 134.02→135.80 (+1.3%),
156.47→157.61 (+0.7%), 262.22→265.69 (+1.3%), 574.62→576.60 (+0.3%),
1474.85→1460.07 (−1.0%). The unchanged arm reproducing this tightly is what
licenses the before/after comparison: the only thing that changed is our arm.

## Correctness — both arms, both tolerances, all 8 ranks

Every arm on every shape passed at the graded `1e-2` **and** the tight `2e-3`:

| # | shape | ours 1e-2 | ours 2e-3 | rank-1 1e-2 | rank-1 2e-3 | ours max\|diff\| | rank-1 max\|diff\| |
|---:|---|---|---|---|---|---:|---:|
| 1 | 64×7168×18432 | True | True | True | True | 9.766e-04 | 1.465e-03 |
| 2 | 512×4096×12288 | True | True | True | True | 9.766e-04 | 1.465e-03 |
| 3 | 2048×2880×2880 | True | True | True | True | 9.766e-04 | 1.465e-03 |
| 4 | 4096×4096×4096 | True | True | True | True | 9.766e-04 | 1.465e-03 |
| 5 | 8192×4096×14336 | True | True | True | True | 9.766e-04 | 1.953e-03 |
| 6 | 8192×8192×29568 | True | True | True | True | 9.766e-04 | 1.953e-03 |

Checked against `all_reduce(x @ w.T + bias)` sliced to the rank's rows, worst
over all 8 ranks. Our arm is uniformly at 9.77e-04 (one bf16 ulp at this scale);
rank-1 is 1.5–2.0e-03, i.e. it passes `2e-3` but with less margin than we have.

### Bias flags, all 8 ranks of all 6 shapes

| # | shape | ranks | declared has_bias | `bias_forced` | `bias_present` |
|---:|---|---:|---:|---|---|
| 1 | 64×7168×18432 | 8 | 0 | True | True |
| 2 | 512×4096×12288 | 8 | 1 | True | True |
| 3 | 2048×2880×2880 | 8 | 1 | True | True |
| 4 | 4096×4096×4096 | 8 | 0 | True | True |
| 5 | 8192×4096×14336 | 8 | 1 | True | True |
| 6 | 8192×8192×29568 | 8 | 0 | True | True |

`bias_forced=True` and `bias_present=True` on all 48 per-rank JSONs. This is the
condition the official evaluator actually creates (its cases parser leaves
`has_bias: False` as the truthy string `"False"`), and it is the condition under
which the tuned rows for shapes 1, 4 and 6 must match — so it is the only
setting under which this comparison means anything. It is also required by
rank-1, whose cached launch path dereferences bias unconditionally.

## Shape 3's tail: the mean is contaminated, the median is not

Shape 3 is the only row where mean and median disagree (327.77 vs 218.46, 75%
relative sd, worst 1771.89 µs). It is not random jitter:

- 21 of 192 samples exceed 2× the median; 64 of 192 exceed 1.3×.
- The outliers sit at the **same sample indices on all 8 ranks** — 16, 19 and 23
  — with magnitudes matching across ranks to under 1% (≈931, ≈836, ≈1044 µs).
  All eight ranks stalling together at the same iteration is a collective-level
  event, not per-rank noise.
- All of them fall in rep 1 (indices 12–23), the rep in which the arm order is
  reversed.
- rank-1 on the same shape in the same pool has **zero** samples above 1.3× its
  median (sd 2.5%), so the node was not globally disturbed.
- The pre-fix run of shape 3 had mean 218.44 ≈ median, i.e. no tail at all, and
  its median is within 0.1% of this run's median.

Effect on the ranking statistic:

| aggregation | ours | rank-1 | ratio |
|---|---:|---:|---:|
| geomean of medians | 404.03 | 319.15 | **1.266×** |
| geomean of p95-trimmed means | 426.87 | 319.75 | 1.335× |
| geomean of means | 446.13 | 327.15 | 1.364× |

Trimming the top 5% does not remove it, because a third of the samples are
elevated.

### The re-run settles it: the tail is the pool, not our kernel

Shape 3 alone, fresh pool, same 12×2 protocol, into `vs_logs_recheck3/`:

| arm | best | median | mean | sd% | worst | n | samples >2× median |
|---|---:|---:|---:|---:|---:|---:|---:|
| ours | 191.49 | 205.40 | 221.13 | 23.2 | 441.68 | 192 | 9 |
| rank-1 | 153.42 | 164.46 | 220.90 | 67.2 | 775.87 | 192 | 21 |

Our tail **did not reproduce** — mean 221 vs 328, sd 23% vs 75%, worst 442 vs
1772 — and 8 of our 9 remaining outliers sit at sample index 0, i.e. the first
timed call after warmup. Meanwhile the heavy tail reappeared on **rank-1's** arm
in this pool (21 samples >2× median, worst 775.87), which drove the two means to
within 0.1% of each other.

So the tail is a per-pool environmental effect that attaches to whichever arm it
happens to catch, not a property of our kernel. Consequences:

- **Means are not a usable statistic on this node**, at either arm's expense.
- Best and median are stable across pools: ours 194.31→191.49 best,
  218.46→205.40 median; rank-1 157.61→153.42 best, 162.76→164.46 median. The
  shape-3 ratio is **1.248× best / 1.249× median** in the re-run against 1.233×
  / 1.342× in the sweep — consistent with the other five shapes.
- Substituting the re-run's shape-3 median into the aggregate moves the median
  geomean from 1.266× to **1.251×**, so the headline is robust at ~1.24–1.27×.

## Reading the artifacts

- `vs_logs/vs_s<i>.rank<r>.json` — post-fix, this run. `arms.ours` / `arms.rank1`
  are raw sample lists (µs); plus `correctness`, `bias_forced`, `bias_present`.
  Shape index `i` is 0-based; shape number in the tables above is `i+1`.
- `vs_logs_prefix_20260811_203207/` — the pre-fix set, archived (not deleted) so
  the before/after is reproducible.
- `rematch_run.log` — full sweep log. `rematch_report.sh` regenerates every table
  here; `rematch_tail.sh` regenerates the shape-3 tail analysis.

## What this gates

The gap is a uniform 1.18–1.38× with no shape-specific pathology left, so the
next experiment should target the systemic cost, not the shape table. On the
largest shape the measured attribution puts 46% in the GEMM mainloop and 32% in
XGMI egress; closing a flat 1.24× needs the mainloop schedule (E1) or the peer
store width (E2), and E1(a) still has 2 VGPR spills / 12 B scratch to reclaim on
the 256×256 rows.

Methodology note for every later comparison: **report best and median, and treat
means as advisory.** The shape-3 re-run showed a jitter tail large enough to move
a mean-based ratio by 0.1× attaching to an arbitrary arm from pool to pool. Same-
run interleaving with reversed arm order protects the central statistics but does
not protect the mean.

# exp_07 — why our harness and the evaluator disagree on shapes 4 and 6

**Verdict in one line: hypothesis 1 (cold L2) is FALSIFIED, and so are 2, 4 and
5. The graded protocol is fully exonerated — reproduced verbatim in eight
processes it gives 330 µs on shape 4, not the evaluator's 1116 µs. The
evaluator's shape-4 and shape-6 numbers for our arm are not reproducible.**

The consequence for the stake is the headline: on a proper same-run denominator
we lose to the reference GEMM+RCCL by **1.06×**, not 1.37×, and the entire
remaining deficit sits on shapes 5 and 6, not on shape 4.

Everything below was measured on `banff-sc-cs47-05`, container `dhk-gemmrs`, on
a node verified clean (`rocm-smi --showpids` → "No KFD PIDs currently running")
before and after every run, with all 8 GPUs at `perf_determinism`. No kernel
source, no existing harness file and nothing under `tools/` was modified. Two
new files were created: `harness/m7_bench_coldl2.py` (a copy of m7_bench's
single-shot path) and `experiments/exp_07_cold_l2/mp_graded.py`.

## What the evaluator's `clear_l2_cache()` actually is

Read from the node at
`…/stock_gemm_rs__test/cwd/utils.py:169`:

```python
def clear_l2_cache():
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
    dummy.fill_(42)
    del dummy
```

That is a **256 MiB** write on the calling rank's own device — large enough to
evict not only the 4 MB/XCD L2 (32 MB total) but MI300X's 256 MB Infinity
Cache. It is called at `eval.py:349`, i.e.

```
clear_l2_cache(); torch.cuda.synchronize(); dist.barrier()
t0 = perf_counter_ns()   # rank 0 only
output = custom_kernel(_clone_data(data, rank))
torch.cuda.synchronize(); dist.barrier()
t1 = perf_counter_ns()
```

so the flush itself is untimed and only its *effect* is paid inside the call.
`_clone_data` (`eval.py:127`) is `data.clone().to(device)` recursively, and it
**is** inside the timed region.

## Step 1 — hypothesis 1, single process, all six shapes: FALSIFIED

`harness/m7_bench_coldl2.py`, paired arms in one process against one `GemmRS`
instance per shape, arm order flipped every rep so neither arm is
systematically first, 45 samples per arm (15 iters × 3 reps), duration-based
400 ms warmup. Correctness re-verified at `2e-3` before *and* after the timed
blocks; error bits and epoch cells clean on every shape.

| # | shape | warm µs | sd% | cold-256 MiB µs | sd% | ratio | evaluator best µs |
|---|---|---|---|---|---|---|---|
| 1 | 64×7168×18432 | 198.35 | 4.7 | 202.63 | 3.9 | **1.02** | 251.4 |
| 2 | 512×4096×12288 | 216.46 | 3.4 | 218.17 | 2.3 | **1.01** | 246.1 |
| 3 | 2048×2880×2880 | 212.52 | 3.2 | 215.93 | 2.4 | **1.02** | 262.2 |
| 4 | 4096×4096×4096 | 323.29 | 2.5 | 329.50 | 2.1 | **1.02** | 1115.8 |
| 5 | 8192×4096×14336 | 867.16 | 1.2 | 888.72 | 0.8 | **1.02** | 913.6 |
| 6 | 8192×8192×29568 | 2677.19 | 0.7 | 2706.16 | 0.8 | **1.01** | 7155.5 |
| | **geomean** | **435.77** | | **443.06** | | **1.02** | 700.0 |

A byte-exact evaluator flush on all eight devices before every timed call costs
**1–2%, uniformly**. Shape 4 moves 323 → 330 µs where the hypothesis needed
323 → 1116. On shapes 1–3 the delta is inside the run-to-run spread; on 4–6 it
is small but consistent (sd < 2.5%), so it is real and it is ~2%.

The warm column also reproduces the numbers in the brief (204.0 / 222.4 / 221.1
/ 329.6 / 875.6 / 2684.4) to within 3%, so the single-process baseline is solid.

The flush itself costs 142–165 µs of wall time for all eight devices, which is
consistent with 8 × 256 MiB of writes plus host issue — and it is outside `t0`
in the evaluator, so it is not charged to anyone.

**Hypothesis 1: falsified.** It is also falsified independently in step 2 below
(`full` vs `noflush`), multi-process.

## Step 2 — the whole graded protocol, in eight processes: the protocol is exonerated

`experiments/exp_07_cold_l2/mp_graded.py` spawns 8 ranks, one process group,
`torch.distributed` + NCCL, and runs the evaluator's timed structure *verbatim*
against the same `submission.custom_kernel` the evaluator imports (HIP IPC
symmetric heap, `HK_DEBUG=0`). Four arms ablate the two non-kernel components,
interleaved with the order flipped every rep, 50 samples per arm:

- `full` — clone + flush (the graded protocol, verbatim)
- `noclone` — flush only
- `noflush` — clone only
- `bare` — neither

All six shapes, `best` of 50 (the robust statistic, and the one the evaluator
report quotes), rank 0's clock, µs:

| # | shape | full | noclone | noflush | bare | **evaluator best** | full/eval |
|---|---|---|---|---|---|---|---|
| 1 | 64×7168×18432 | 222.9 | 194.3 | 221.1 | 186.9 | 251.4 | 0.89 |
| 2 | 512×4096×12288 | 238.1 | 198.8 | 233.6 | 192.6 | 246.1 | 0.97 |
| 3 | 2048×2880×2880 | 239.3 | 200.6 | 236.5 | 197.1 | 262.2 | 0.91 |
| 4 | 4096×4096×4096 | **332.3** | 304.5 | 328.8 | 298.8 | **1115.8** | **0.30** |
| 5 | 8192×4096×14336 | 866.1 | 834.6 | 858.3 | 826.4 | 913.6 | 0.95 |
| 6 | 8192×8192×29568 | **2687.0** | 2642.5 | 2670.7 | 2636.4 | **7155.5** | **0.38** |

Correctness `allclose=True` on all eight ranks for every shape.

Read the last column. **Four of six shapes reproduce the evaluator to within
3–11%.** Shapes 4 and 6 miss by 3.4× and 2.7× — exactly the two anomalies. A
reproduction that tracks the evaluator everywhere except at the two points in
dispute is strong evidence that the dispute is not in the thing being
reproduced.

The arm deltas also settle the remaining hypotheses directly:

- `full` − `noflush` = the flush: +1.8 µs (sh4), +16.3 µs (sh6). **H1 falsified
  again, multi-process.**
- `full` − `noclone` = the timed clone: +27.8 µs (sh4), +44.5 µs (sh6) — and
  shape 6 clones 121 MB while shape 4 clones 8.4 MB. Real, tiny, and it scales
  the way a memcpy should. **H2 falsified.**
- `bare` is the pure kernel under barriers with no pipelining: 298.8 µs on
  shape 4 against 201.1 µs pipelined. That ~100 µs *is* the un-amortized
  per-call cost. **H3 confirmed in magnitude and already fully accounted for —
  it is the same ~30–50 µs-scale residual seen on shapes 1, 2, 3 and 5, and it
  does not scale with size.**
- Barriers plus a 256 MiB flush between every single call did not perturb the
  epoch/credit protocol: correctness passed on all shapes and ranks, no error
  bits. **H4 falsified.**
- Shape 4 (0.94 waves) and shape 6 (3.76 waves) run at 332 µs and 2687 µs under
  the very protocol that supposedly punishes them. **H5 falsified as an
  explanation of the evaluator gap** (it may still be worth a few percent as an
  optimization axis; that is E7's business, not this diagnosis').

## Step 3 — same-run denominator: ours vs the reference GEMM+RCCL

The step-2 table compares our reproduction against the evaluator's *reference*
column, which was measured in a different process on a different day. That is
not a denominator. So both arms were run through the identical graded protocol,
in the same process pool, interleaved, order flipped every rep — the reference
being `torch.matmul(x, w.T) (+bias)` then `dist.reduce_scatter_tensor`,
semantically exp026's own `submission.py`.

`best` of 50, µs:

| # | shape | ours | reference | ratio | evaluator said |
|---|---|---|---|---|---|
| 1 | 64×7168×18432 | 220.1 | 226.3 | **0.97 win** | 0.85 win |
| 2 | 512×4096×12288 | 235.1 | 277.8 | **0.85 win** | 0.74 win |
| 3 | 2048×2880×2880 | 236.2 | 297.8 | **0.79 win** | 0.75 win |
| 4 | 4096×4096×4096 | 330.8 | 337.0 | **0.98 tie** | 2.72 lose |
| 5 | 8192×4096×14336 | 860.2 | 667.1 | 1.29 lose | 1.23 lose |
| 6 | 8192×8192×29568 | 2682.6 | 1568.1 | 1.71 lose | 4.16 lose |
| | **geomean** | **458.80** | **433.09** | **1.06 lose** | **1.37 lose** |

Two things to take from this.

1. **The reference reproduces with a uniform offset; our arm does not.** Our
   reference numbers are 13–19% below the evaluator's reference on *every*
   shape — a consistent harness offset, which is what a harness difference
   should look like. Our own arm matches on four shapes and is 3.4×/2.7× off on
   two. Different shapes of error, different causes.
2. **We are not losing by 1.37×. We are losing by 1.06×**, and shapes 5 and 6
   are the whole of it. Shape 4, the case the brief singled out, is a tie.

Caveat, stated plainly: the `ok` column in this run verifies our arm's output
(the order flip leaves ours last), not the reference's. The reference is
sixteen lines of stock torch and was independently correct in step 2's oracle,
but it was not re-checked per-iteration here.

## Hypothesis scorecard

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| 1 | `clear_l2_cache()` before every timed call | **FALSIFIED** | 1–2% uniformly, single-process (step 1) and multi-process (`full` vs `noflush`, step 2) |
| 2 | `_clone_data` inside the timed region | **FALSIFIED** | +27.8 µs sh4, +44.5 µs sh6 (`full` vs `noclone`) |
| 3 | launch skew paid per call, unpipelined | **CONFIRMED, and small** | `bare` 298.8 vs 201.1 pipelined on sh4; ~100 µs, size-independent, explains the small shapes and nothing else |
| 4 | epoch/credit waits assuming a warm steady state | **FALSIFIED** | barrier + 256 MiB flush between every call, correctness clean on all 6 shapes × 8 ranks, no error bits |
| 5 | shape geometry (waves just under an integer) | **FALSIFIED as the gap's cause** | same geometry, same protocol, 332 µs not 1116 µs |
| — | **the graded protocol itself** | **EXONERATED** | reproduces 4 of 6 evaluator numbers within 3–11%; fails to reproduce exactly the 2 in dispute |

## What the evidence now favours, and the next decisive test

The gap is not in the kernel and not in the protocol, so it is in the state of
that particular evaluator run. The strongest concrete lead is a **bistable slow
mode in our arm**, and there is a specific number pointing at it:

- Our arm's tail is heavy and the reference's is not. Shape 4 `ours:full`:
  best 330.8, mean 355.5, **max 1102.6** (sd 30.5%) against `ref:full` best
  337.0, mean 347.5, max 440.6 (sd 5.2%). In the step-2 run shape 4 reached
  **3532 µs** and shape 3 reached 2789 µs.
- **Our shape-4 max of 1102.6 µs is within 1.2% of the evaluator's shape-4
  *best* of 1115.8 µs.** The evaluator's shape 4 (best 1115.8, mean 1160.4,
  worst 2482.9) looks exactly like our slow mode entered and never left.
- The tails concentrate in the `full` arm — the only arm where a 256 MiB
  alloc/free and the clone's allocations churn the caching allocator in the
  same iteration — which is a plausible trigger for falling out of phase, and
  also hit the reference once (shape 2 `ref:full`, sd 56%, max 1467).

So the refined hypothesis is: *the protocol has a slow attractor; a per-call
allocator/scheduling hiccup can knock the eight ranks out of phase, and our
spin-and-credit design stays desynchronized once it is, where RCCL
resynchronizes every call.* Note `hk_submission` runs `SPIN_LIMIT = 20 000 000`
against the harness's 2 000 000.

Next decisive test, in order of value:

1. **Characterize the tail.** Re-run `mp_graded.py` shape 4, `full` arm, 500+
   iterations, and dump the full per-iteration series rather than summary
   statistics. If it is bimodal with a ~1.1 ms mode, the evaluator number is
   explained and the target becomes "eliminate the slow mode", which is worth
   far more than any percent-level mainloop work. `mp_graded.py` already writes
   every sample to `logs/graded_*.rank*.json`; only the analysis is missing.
2. **Re-run the actual evaluator on shapes 4 and 6 on a verified-clean node.**
   If it now reports ~330/2700 µs, the original table was contaminated and the
   1.37× conclusion should be retracted outright. This is cheap in machine time
   and settles the record.
3. **Then optimize shapes 5 and 6 against the reference**, which is where the
   real, reproducible 1.29×/1.71× deficit actually lives. Both are the
   large-egress shapes, so E2 (XGMI egress: store width, packets in flight,
   coalescing) is the indicated axis, not E1.

## Files

- `01_probe.sh` — node/clock check, evaluator `clear_l2_cache` and timed region
- `02_run_coldl2.sh` + `harness/m7_bench_coldl2.py` — step 1, single-process paired warm/cold
- `03_run_graded.sh`, `04_all_six.sh`, `mp_graded.py` — step 2, 8-process graded protocol, 4 arms
- `05_vs_reference.sh` — step 3, ours vs reference, same run
- `logs/coldl2_main.txt`, `logs/graded_a.txt`, `logs/graded_all6.txt`,
  `logs/graded_vs.txt`, and per-rank JSON with every raw sample

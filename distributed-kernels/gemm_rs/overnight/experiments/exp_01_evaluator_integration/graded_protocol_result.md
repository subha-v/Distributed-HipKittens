# The graded protocol: our kernel vs the reference GEMM+RCCL, same evaluator

**This is the most decision-relevant measurement of the session and it is bad
news, so it is written down in full.**

## What was run

Both arms went through the *identical* official `eval.py`, in the *same*
container (`dhk-gemmrs`), on the *same* six graded shapes, back to back, on a
verified-clean node with clocks pinned. Ours with `HK_DEBUG=0`, so no debug
synchronize sits inside a timed call.

- **ours** — `hk_submission.py`, the fused persistent GEMM→ReduceScatter
  megakernel, at the night's validated optimum (our-harness geomean 255.98 µs).
- **reference** — `exp026`'s own `submission.py`: `torch.matmul` +
  `torch.distributed.reduce_scatter_tensor`. The naive fused-free baseline.

## Test mode fails for BOTH arms — so it is not our bug

Both arms died identically at `test.0` (`64×2880×2880`, `has_bias=True`) with
`multiprocessing.context.TimeoutError` out of `rets = [el.get(60) ...]` in
`run_multi_gpu_test`. **The reference implementation fails it too**, so the
60 s per-case limit is an environment property of this staging, not a defect in
our submission. That materially de-risks the integration: our kernel is as
healthy under the evaluator as plain torch is.

It also means the benchmark numbers below were obtained under identical
conditions for both arms, so **the ratio is meaningful even though neither
absolute number is trustworthy on its own.**

## The numbers (evaluator, ns converted to µs)

Relative standard deviations run 12–93%, so `best` (minimum over 100 runs) is
the more robust statistic and is the one to read. Both are given.

| # | shape | ours mean | ref mean | ours **best** | ref **best** | best ratio |
|---|---|---|---|---|---|---|
| 1 | 64×7168×18432 | 293.3 | 381.0 | **251.4** | 295.2 | **0.85 — we win** |
| 2 | 512×4096×12288 | 279.3 | 401.4 | **246.1** | 334.7 | **0.74 — we win** |
| 3 | 2048×2880×2880 | 286.4 | 416.9 | **262.2** | 347.4 | **0.75 — we win** |
| 4 | 4096×4096×4096 | 1160.4 | 533.4 | 1115.8 | 410.8 | **2.72 — we lose** |
| 5 | 8192×4096×14336 | 979.4 | 1023.5 | 913.6 | 740.0 | 1.23 — we lose |
| 6 | 8192×8192×29568 | 9021.8 | 2759.4 | 7155.5 | 1718.6 | **4.16 — we lose** |
| | **geomean** | **787.7** | **676.0** | **700.0** | **511.6** | **1.37** |

**Under the graded protocol we are ~1.37× SLOWER than the naive baseline**,
despite being 1.17–1.36× faster on the three small shapes.

## The gap is not in the kernel — it is in what the protocol exposes

Our harness and the evaluator disagree wildly, and only on the large shapes:

| # | our harness (pipelined) | evaluator best | inflation |
|---|---|---|---|
| 1 | 76.9 | 251.4 | 3.3× |
| 2 | 104.7 | 246.1 | 2.4× |
| 3 | 91.5 | 262.2 | 2.9× |
| 4 | 201.1 | 1115.8 | **5.5×** |
| 5 | 748.9 | 913.6 | 1.2× |
| 6 | 2537.4 | 7155.5 | **2.8×** |

A roughly constant additive overhead would inflate the *small* shapes most.
Instead shapes 4 and 6 inflate worst, which rules out "fixed per-call cost" as
the whole story and points at something that **scales with the work or the
heap**.

Leading hypotheses, none yet tested — this is the top follow-up:

1. **Launch skew, unamortized.** Our harness measures ~100–105 µs of skew
   across the 8 devices and hides it by pipelining 50 iterations. The graded
   protocol is `barrier → call → synchronize → barrier` with **no pipelining**,
   so every call pays the skew, and a cross-rank protocol pays it on the
   critical path rather than absorbing it. This explains the small shapes well
   (76.9 + ~100 ≈ 180, against 251 measured) but not shape 6.
2. **`clear_l2_cache()` before every timed iteration.** The evaluator flushes L2
   between calls. We established from the ISA that our payload peer stores carry
   **no cache-scope bits** and therefore land in local L2, egressing only when
   `buffer_wbl2 sc0 sc1` flushes them — so this kernel is unusually
   L2-dependent, and a cold L2 every call may hurt us far more than it hurts
   NCCL. This mechanism *does* scale with payload size, which fits shapes 4
   and 6.
3. **`_clone_data(data, rank)` inside the timed region**, which for shape 6 is
   ~121 MB of input copied per iteration. Both arms pay it, but it interacts
   with allocator and memory pressure differently when we also hold a 134 MB
   symmetric heap.
4. **Epoch/credit waits under a cold start.** Producers wait on the previous
   epoch's reuse credits; with a barrier and an L2 flush between every call,
   the steady-state assumptions that hold in a pipelined sweep may not.

## What this changes

- **The pipelined geomean is not the graded score.** Our 255.98 µs and the
  competition's statistic are measuring different things, and the difference is
  not a constant factor — it is 1.2× on shape 5 and 5.5× on shape 4.
- **Optimizing the mainloop further has low marginal value for the graded
  score.** Tonight's −10.2% in the harness bought real device time, and it is
  genuinely faster, but on the graded protocol shapes 4 and 6 are dominated by
  something else entirely.
- **Hypothesis 2 is the one to test first**, because it is cheap, it is the only
  hypothesis that explains the payload-size scaling, and it is directly
  actionable: if a cold L2 is what hurts, the fix is in the egress path (E2),
  not the mainloop. A profiler run counting fabric bytes against useful bytes
  under a cold L2 settles it.
- Everything here rests on runs whose relative standard deviations reach 93%.
  **Re-measure with more repeats before acting on any single number**, and
  treat the per-shape direction (we win small, lose large) as the finding
  rather than the exact multiples.

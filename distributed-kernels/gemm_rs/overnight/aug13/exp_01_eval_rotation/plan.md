# exp_01 (aug13) — instrument-B ladder, rotated, and the HK_DEBUG confounder

## Finding that motivates this experiment (pre-run, from artifact inspection)

exp_24 §12's instrument-B "ours" arm (578.7 µs geomean, `ours/rank1` 1.6159×,
the basis for "the per-call host tax is the largest term") was run with
**HK_DEBUG=1**: `compbench/ours/benchmark.stderr.txt` from the 08:24–08:58 run
contains **29,744 `[hk ...]` lines**, including per-call `enter custom_kernel` /
`cache hit` / `launch issued` / `launch synchronized` lines that only print
when DEBUG is on. Under `hk_submission.py`, `HK_DEBUG=1`:

1. sets `_FAST = False` — every timed call takes the slow path (c10d
   `get_rank`, group lookup, per-call f-string construction, and the
   duck-typed `_entry` whose pyutils converter exp_12 measured at ~7.5 µs);
2. adds a per-call `torch.cuda.synchronize` **and a blocking D2H error-bit
   read** (`rt.device_to_host`) inside the evaluator's timed region;
3. writes ~5 flushed stderr lines per call per rank inside the timed region.

`run_ours_evaluator.sh` defaults `HK_DEBUG=${HK_DEBUG:-1}`, so the graded arm
inherited the debugging configuration. Neither the reference nor the rank-1 arm
carries any analogous per-call instrumentation. The "arm-dependent inflation"
(ours 1.71× vs rank-1 1.16×) is therefore confounded with a self-inflicted
per-call debug tax, and exp_24 §12.3's own caveat (one rotation, no repeat)
still stands on top of that.

## Questions, pre-registered

- **Q1**: With `HK_DEBUG=0` (fast path live), what is ours-B? Prediction:
  substantially below 578.7; the debug tax is worth >5% of the geomean and
  more on the small shapes. Falsifier: ours(fast) ≈ ours(debug) within
  session-to-session spread — then the host-tax story survives intact.
- **Q2**: With rotation (each arm first in at least one session, both
  orderings of ours vs rank-1 across sessions), what are `ours/rank1` and
  `ours/reference` under the official evaluator, with spread?
- **Q3**: How large is instrument B's session-to-session drift per arm?
  (exp_24 had exactly one rotation; drift was unbounded.)

## Design

One node session, GPU lease held throughout, clocks pinned 1900.

Arms (all through the unmodified official `eval.py`, benchmark mode, the six
graded shapes, seeds identical to exp_24):

| arm | submission | notes |
|---|---|---|
| `O` ours | `hk_submission.py`, `HK_DEBUG=0` | fast path; stderr must contain **zero** `[hk ` lines (asserted per pass) |
| `D` ours_dbg | same, `HK_DEBUG=1` | reproduces exp_24 §12's configuration; the O−D contrast is the debug tax, same session |
| `R` reference | exp026 `submission.py` (torch.matmul + RCCL) | as `run_reference_arm.sh` |
| `K` rank1 | frozen submission via `patch_rank1.py`, `AMDGCN_USE_BUFFER_OPS=0` | staged once, JIT warmed once, cache reused across sessions (the mandatory warm→test→bench order; warm pass recorded as throwaway) |

Staging session S0: stage each arm, run its eval `test` pass (correctness
gate), rank-1's warm pass, then one bench pass per arm in order O, R, K.
Sessions S1–S4 run bench passes only, in orders:

- S1: K R O D
- S2: D O R K
- S3: K O R
- S4: O K R

giving 5 bench samples each for O/R/K (arm-first coverage: O first in S0/S4,
K first in S1/S3, R first via S0's second slot and S2's third — and both
orderings of O vs K appear at least twice each) and 2 samples for D.

Gates: every pass must end `check: pass` in its popcorn output; a failed pass
aborts the campaign (no retry-until-green). `kfd_wait_clean` between passes.
Kernel binary: untouched `build/gemm_rs_mi300x.so` (md5
6bb2a223aebfad167c6db08032f8fff8, built 2026-08-12 13:46 from branch head
cc727283, PERSHAPE=0 default — verified before the run).

Statistic: per exp_24 §12, geomean over the six shapes of per-shape **best**;
geomean of means reported beside it. Ratios primarily within-session; µs
across sessions only with the drift bound from Q3 (exp_24's remeasure lesson:
cross-run graded µs is not comparable on the small shapes).

## What this experiment deliberately does not do

No kernel change, no submission change beyond the environment variable the
driver already exposes, no new instrument. If Q1 confirms the debug tax, the
follow-up (exp_02) attributes the *residual* B-vs-A inflation with a floor arm
and per-phase timing outside the graded path.

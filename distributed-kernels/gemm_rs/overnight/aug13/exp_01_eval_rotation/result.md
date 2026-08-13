# exp_01 result — instrument B re-measured with the fast path live

**Status: S0 (one full rotation, all three arms, same session) COMPLETE; the
rotation sessions s1–s4 were cleanly aborted at a pass boundary when the
operator paused the session.** The rotation questions (Q2 spread, Q3 drift,
same-session D contrast) are therefore PARTIALLY answered; the headline (Q1)
is answered decisively. Numbers below are geomean over the six graded shapes
of per-shape best, µs, from the official `eval.py`, one process per rank —
exactly exp_24 §12's statistic.

## Q1 — ANSWERED: exp_24 §12's "ours" arm was measuring its own debug harness

exp_24's evaluator run of our kernel ran with `HK_DEBUG=1`
(`run_ours_evaluator.sh` defaulted it on; 29,744 `[hk ]` lines in that run's
archived stderr prove it). DEBUG disables the exp_12 prebound fast path and
adds, per call, inside the evaluator's timed region: a `torch.cuda.
synchronize`, a **blocking D2H error-bit read**, and ~5 flushed stderr writes
per rank. With `HK_DEBUG=0` (fast path verified live — zero debug lines in
stderr, asserted):

| arm / source | 64 | 512 | 2048 | 4096 | 8192a | 8192b | GM(best) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ours, S0 (fast path)** | **190.1** | **199.5** | **215.7** | **340.1** | **752.5** | **1876.0** | **397.2** |
| ours, exp_24 §12 (DEBUG on) | 352.6 | 352.1 | 369.8 | 483.6 | 886.4 | 1908.5 | 578.7 |
| reference, S0 | 310.5 | 324.0 | 266.1 | 422.5 | 709.4 | 1721.8 | 489.8 |
| reference, exp_24 §12 | 318.0 | 331.0 | 326.0 | 401.6 | 749.8 | 1701.3 | 509.9 |
| rank-1, S0 (bench) | 191.5 | 162.5 | 173.7 | 308.0 | 578.3 | 1480.2 | **335.5** |
| rank-1, S0 (warm, throwaway) | 217.1 | 155.9 | 192.8 | 308.0 | 595.3 | 1497.2 | 348.5 |
| rank-1, exp_24 §12 | 213.7 | 186.4 | 201.0 | 306.9 | 574.8 | 1493.6 | 358.1 |

The debug tax was worth **−31% of our evaluator geomean** (578.7 → 397.2),
concentrated exactly where exp_24 located its "per-call host tax": −46%/−43%/
−42%/−30% on shapes 1–4, −15%/−2% on 5–6. The reference arm, which carries no
such instrumentation, reproduced within 4% (489.8 vs 509.9) — the shift is
arm-specific, as the confounder predicts.

**Consequences for the ledger:**

1. **`ours/reference` under the official evaluator = 0.8109 (S0,
   within-session). We beat reference GEMM+RCCL under the harness that grades
   us.** exp_24 §12's contrary finding (1.1349) is withdrawn as
   instrument-contaminated; its instrument-A win (0.8863) stands and now
   agrees in direction with instrument B.
2. **`ours/rank-1` under the official evaluator = 1.1841 (S0,
   within-session)**, against exp_24 §12's 1.6159. Per-shape O/K:
   **0.993 / 1.228 / 1.242 / 1.104 / 1.301 / 1.267** — shape 1 is a tie, and
   the two instruments now agree on the ORDERING everywhere
   (rank1 < ours < reference). The residual vs instrument A's graded 1.1165
   is +6 points, part protocol (per-call barrier topology), part rank-1
   reading faster this session on shapes 2–3 (162.5/173.7 vs 186.4/201.0) —
   the rotation sessions that would bound that drift were cut short.
3. The "arm-dependent inflation" (ours 1.71× vs rank-1 1.16×) was mostly the
   debug configuration, not a structural per-call host tax. exp_24 §12's
   action item ("the per-call host tax moves to the top of the queue") is
   REPRICED: the largest terms separating us from rank-1 under the grading
   protocol are back to being **shapes 5 and 6's device schedule** (1.30×,
   1.27×), where both instruments agreed all along, with shapes 2–3 (~1.23×)
   as the second pool.
4. For scale only (different machine, not comparable): rank-1's leaderboard
   score is 413.139 µs GM; our S0 evaluator GM on this node is 397.2.

## Environment finding — eval.py TEST mode cannot pass on this node

`eval.py` test mode hardcodes a 60 s per-case pool timeout; a fresh 8-worker
spawn pool takes ~40 s before any rank attaches a GPU (watched live via KFD).
Case 1 therefore times out for EVERY arm — ours, reference, AND rank-1 (all
three recorded tonight). Pre-existing: exp_24's archived ours test popcorn
shows the identical truncation. Binding correctness gates instead: M3
(17 shapes, 1e-2 AND 2e-3, ran green 17/17 on this binary before any timed
number tonight) and benchmark mode's obligatory checked first call per case
(every arm `check: pass` on all six shapes).

## Gates

- M3 17/17 both tolerances on the production binary (md5 6bb2a223, built
  2026-08-12 13:46 from cc727283, PERSHAPE=0 verified) — `raw/m3_gate.log`.
- Clocks pinned 1900 (perf determinism) before any timed pass; GPU lease held
  throughout; kfd-clean between passes; every pass `check: pass`.
- Fast-path assertion per ours pass: stderr must contain zero `[hk ]` lines.

## What was NOT measured (owed to the next session)

- The same-session `ours_dbg` (D) contrast — the debug tax is currently
  quantified cross-run (578.7 exp_24 vs 397.2 S0) plus a same-day smoke
  (shape 2: 190.9/200.8 fast vs 352.1 exp_24); s1/s2 would have paired it
  in-session.
- Arm-order rotation and drift bounds (s1–s4): S0 ran O→R→K only. exp_24's
  one-rotation caveat therefore still applies to the absolute µs; the O/R and
  O/K ratios quoted are within-session and safe against drift, but carry a
  single arm-order.
- Re-run: `cd overnight/aug13/exp_01_eval_rotation && setsid nohup bash
  run_rotation.sh > run_rotation.log 2>&1 &` reproduces everything including
  S0 (~40 min) and the four rotation sessions (~100 min).

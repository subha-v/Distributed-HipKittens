# exp_01 result — instrument B re-measured with the fast path live

**Status: COMPLETE.** Two campaigns: the aug13a session (S0 only, one full
rotation O→R→K; s1–s4 aborted at a pass boundary by an operator pause) and
the aug13b full campaign (s0 re-run + s1–s4, all 18 bench passes green,
2026-08-13 20:16–22:14Z), which answers Q2 and Q3 and delivers the
order-balanced table (§ "Rotation campaign" below). Numbers are geomean over
the six graded shapes of per-shape best, µs, from the official `eval.py`, one
process per rank — exactly exp_24 §12's statistic. The aug13a raw tree is
archived node-side as `raw_prior_s0/`; `raw/` holds the aug13b campaign.

## HEADLINE (order-balanced, the citable numbers)

- **`ours/rank-1` = 1.1203, 95% CI [1.0795, 1.1628]** (n=5 within-session
  pairs, both arm orderings covered; neither ordering flips the sign).
- **`ours/reference` = 0.7641, 95% CI [0.7268, 0.8033]** — ours beats
  reference GEMM+RCCL in EVERY session, under every arm order.
- The HK_DEBUG tax, now measured same-session: **+44.3% / +43.6%** (s1, s2).
- The gap to rank-1 is carried by shapes 5/6 (1.305 [1.273, 1.338] and
  1.220 [1.188, 1.253]); shapes 2–4 contribute ~1.09–1.14; shape 1 reads as
  a small ours WIN (0.895, CI [0.761, 1.052] crosses 1).

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

## Rotation campaign (aug13b) — Q2 and Q3 ANSWERED

Full campaign (s0 O R K; s1 K R O D; s2 D O R K; s3 K O R; s4 O K R), one
node session 20:16–22:14Z, lease held throughout, clocks pinned 1900, M3
17/17 re-gated on the production binary before any timed pass, every pass
`check: pass`, every O pass fast-path-asserted (0 `[hk ]` lines). Analysis:
`paired_ci.py raw/` (t-based CIs on log-ratios over sessions; the session is
the pairing unit — both arms share clocks, boot state and drift).

| sess | order | O | K | R | O/K | O/R |
|---|---|---:|---:|---:|---:|---:|
| s0 | O R K | 383.9 | 353.9 | 502.1 | 1.0846 | 0.7645 |
| s1 | K R O D | 396.9 | 360.0 | 498.1 | 1.1026 | 0.7969 |
| s2 | D O R K | 390.9 | 352.8 | 511.8 | 1.1080 | 0.7638 |
| s3 | K O R | 387.2 | 330.9 | 540.8 | 1.1703 | 0.7161 |
| s4 | O K R | 391.9 | 344.3 | 501.4 | 1.1383 | 0.7816 |

**Q2 — order-balanced ratios (n=5 within-session pairs):**

- `O/K` = **1.1203, 95% CI [1.0795, 1.1628]**. By construction order:
  O-before-K sessions (s0/s2/s4) GM 1.1101, K-before-O (s1/s3) GM 1.1359 —
  a ~2.3% ordering effect that does NOT flip the sign (the exp_26 bar).
  rank-1 is genuinely faster; the aug13a single-session 1.1841 was the high
  tail of a spread this campaign bounds.
- `O/R` = **0.7641, 95% CI [0.7268, 0.8033]**; ours wins all 5 sessions
  under both orderings. The beat-reference claim is order-balanced.
- `O/D` = 0.6946 (n=2): the debug tax same-session is **D/O = 1.4434 (s1) /
  1.4360 (s2)** — matching the cross-run estimate 578.7/397.2 = 1.457 and
  closing the contrast owed by aug13a.

Per-shape O/K with CIs: 64 **0.895 [0.761, 1.052]** (a small ours win, CI
crosses 1); 512 1.094 [1.007, 1.189]; 2048 1.137 [1.048, 1.233]; 4096
1.116 [1.038, 1.199]; 8192a **1.305 [1.273, 1.338]**; 8192b
**1.220 [1.188, 1.253]**. The rank-1 gap remains concentrated in shapes
5/6's device schedule — the exp_03 target — with shapes 2–4 as the second
pool (~1.09–1.14).

**Q3 — drift bound:** per-arm GM spread across the 5 sessions: O **3.40%**,
K **8.79%**, R **8.57%** (D 2.08%, n=2). rank-1 and reference carry 2.5× our
session noise; cross-session absolute µs is not citable for any arm at the
few-percent level, and single-session ratio readings move by up to ±5 points
of O/K (1.085–1.170 observed). Within-session paired ratios with the CI
above are the citable statistic.

**Arm-first contrast** (GM when the arm ran first vs later in the session):
O 0.9903, K 0.9853, D 0.9796 — running first is worth ~1–2%, consistent
with the aug11 allocation-order lesson, and inside the drift bound.

**aug13a vs aug13b cross-check:** aug13a S0 read O 397.2 / K 335.5 /
R 489.8; aug13b s0 read 383.9 / 353.9 / 502.1. Each arm moved 3–6% across
nights — the drift Q3 quantifies — while the within-session verdicts
(K < O < R) and the O/R < 1 headline reproduced in every session.

Re-run: `cd overnight/aug13/exp_01_eval_rotation && setsid nohup bash
run_rotation.sh > run_rotation.log 2>&1 &` (~2 h; archives nothing itself —
move `raw/` aside first).

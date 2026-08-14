# Official-setup iteration log (aug13, post-redirect)

Operating mode (user directive): the official GPU MODE evaluator (leaderboard
564 protocol — unmodified eval.py, benchmark mode, one process per rank,
official shapes/seeds, geomean of per-shape best) is the ONLY harness.
Correctness = benchmark mode's checked first call per case (eval.py test mode
times out for every arm on this node — pre-existing, exp_01). Hygiene kept:
GPU lease, clocks pinned 1900, HK_DEBUG=0 asserted per pass. Everything else
(rotation sessions, paired CIs, M-gate ladder) retired unless asked again.

## Scoreboard (the target), official evaluator, session s4 2026-08-13 ~22:00Z

| shape | ours | rank-1 | ours/rank-1 |
|---|---:|---:|---:|
| 64x7168x18432 | 188.9 | 185.4 | 1.019 |
| 512x4096x12288 | 195.0 | 178.8 | 1.091 |
| 2048x2880x2880 | 220.7 | 190.0 | 1.162 |
| 4096x4096x4096 | 331.8 | 300.3 | 1.105 |
| 8192x4096x14336 | 758.4 | 585.3 | 1.296 |
| 8192x8192x29568 | 1770.3 | 1504.5 | 1.177 |
| **GEOMEAN** | **391.9** | **344.3** | **1.138** |

Beat-rank-1 requirement: −13.8% GM. Largest prizes: shapes 5, 6, 3.
(rank-1's leaderboard score on leaderboard hardware: 413.139 µs GM — scale
reference only, different machine.)

## it01 — COMMIT_MID (exp_27 row B): KEEP (−0.88% GM)

`HK_KERNEL_MODULE=gemm_rs_mi300x_cmid` vs production module, one lease,
back-to-back official passes (00:18–00:31Z), both `check: pass`:

| shape | base | cand | delta |
|---|---:|---:|---:|
| 64 | 194.9 | 191.6 | −1.70% |
| 512 | 197.8 | 196.7 | −0.57% |
| 2048 | 217.7 | 219.4 | +0.80% |
| 4096 | 332.4 | 327.8 | −1.39% |
| 8192a | 749.5 | 753.4 | +0.52% |
| 8192b | 1842.6 | 1789.2 | **−2.90%** |
| **GM** | **395.9** | **392.5** | **−0.88%** |

Matches exp_27 §6's pre-registration (geomean −0.9%, shape 6 ~−3.1%)
almost exactly. VERDICT: KEEP — flip `HK_GEMM_RS_MI300X_COMMIT_MID` default
to 1 and rebuild the production .so at resume (not done yet: the pause
directive landed as it01 finished; source default is still 0 everywhere).

Note (one-line context, not ceremony): single-pass deltas on this node carry
a few-percent session noise (exp_01 measured it), so sub-1% GM readings are
soft; the s6 −2.90% is the load-bearing number and is beyond its typical
spread.

## it02 — reducer-count (NR) response, shapes 5/6: STAGED, NOT RUN

`HK_GEMM_RS_NR="row:NR[,row:NR...]"` resolve-time override added to
`gemm_rs_mi300x_host_abi.hpp` (default-off, work-distribution only, grid
untouched). `it02_nr_sweep.sh` builds `gemm_rs_mi300x_nrenv.so` (production
untouched) and runs base + {5:24,6:32} + {5:40,6:40} + {5:48,6:56} in one
lease (~27 min). Table values today: row5 NR=32, row6 NR=48.

## M9 attribution closeout (exp_03 epilogue)

cmid AND its flag-0 twin cmid0 both faulted at the same M9 case (row 3,
2048x2880x2880) with a green M3(prod) canary between: the fault is the M9
INSTRUMENT (the golden e3base module's documented row-3 failure mode), not
the candidate. cmid exonerated; M9 retired under the new mode. Logs:
`../exp_03_commit_mid/logs/attr_m9_*.log`.

## Queue at pause (next, in order)

1. Flip COMMIT_MID default to 1, rebuild production .so, one confirmation
   pass (~7 min).
2. Run it02 (NR sweep) on top of the new base.
3. Shape-5 device schedule (exp_27 rows C/E) — the 1.296 gap is the podium.

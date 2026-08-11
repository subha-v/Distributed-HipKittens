# GEMM-RS MI300X handoff — evidence ledger and blockers

Rewritten 2026-08-11 ~13:15 PT at the end of the optimization session. The
previous version of this file described the bring-up session and is superseded;
`experiments/LESSONS.md` holds the full append-only record and `RESULTS.md` the
gate-by-gate history.

## One-paragraph state

The kernel is **19.0% faster than it started** (our harness geomean
285.02 → 230.84 µs) through five landed changes, each through the full gate
ladder. More importantly, **we now beat the reference GEMM+RCCL baseline on
this node under the graded protocol**, measured same-run and interleaved:
geomean **427.39 µs vs 438.15 µs on best, 462.86 vs 497.53 on mean**. We win
four of six shapes and lose the two largest. rank-1 was never revived — its
2-hour budget was spent elsewhere and `heap_bases_*.pkl` is still zero.

## The ratchet, as it stands

| level | denominator | value | status |
|---|---|---|---|
| 1 | rank-1 measured here | — | **not running**; repair #5 still untested |
| 2 | reference GEMM+RCCL, same-run graded | 438.15 µs best / 497.53 mean | **beaten**: 427.39 / 462.86 |
| 3 | our harness geomean | 285.02 µs at session start | **230.84 µs, −19.0%** |

## What landed, in order

| exp | change | geomean | note |
|---|---|---|---|
| — | baseline re-measured on a clean node | 285.02 | reproduced the recorded 285.7 to 0.25% |
| exp_02 | uniform `NR=32` | 280.74 | shape 6 −8.1%; predicted 2632, measured 2633 |
| exp_04a | shape 1 retiled `32/256/32` → `32/64/64` | 269.96 | shape 1 −20.3%; was using 21% of the machine |
| exp_03 P3 | k-step split into two `BK/2` halves | 270.12 | timing-neutral, but clears 12 B scratch + both spills and frees 23 registers |
| exp_03 E1(b) | `G::load` split into issue/commit | 256.09 | two exposed global round trips per k-iteration → one covered one |
| exp_08 | `WGM = (tiles <= NG) ? 4 : num_pid_m` | **230.84** | shape 6 −27.9%; egress link concurrency 2.02 → 7.53 of 8 |

Current per-shape means µs: `77.88 / 91.18 / 90.25 / 200.33 / 644.39 / 1828.89`.

**Three of the five wins were donor constants that had never been questioned**
(`NUM_REDUCER_CTAS`, `BM/BN/BK`, `WGM`). Together they are worth ~19%. Treat
every remaining inherited constant as an untested hypothesis.

## Where the time goes now — RE-MEASURE THIS FIRST

The attribution below was taken **before exp_08** and is stale: exp_08 cut
shape 6 by 708 µs and also dropped operand reads 29.7%, so both the GEMM and
XGMI columns have moved. `tools/reattribute.sh` re-runs it in ~10 min.

Stale (post-E1b, pre-exp_08), shape 6 total 2500.8 µs: GEMM 1153.3, XGMI
1149.9, release 219.3, sync 89.9, reduce 78.7.

Two facts that survive exp_08 and should shape the next experiment:

- **The bunched share of egress rose from 32% to 53%**, so **E3 (release
  grouping) got bigger relative to this binary, not smaller.** Its protocol
  review is already APPROVE-WITH-CONDITIONS with an exact code sequence
  (`experiments/exp_05_release_granularity/protocol_review.md` §4.3), and an
  ungated implementation exists as `e3_wip_ungated.diff` — it was written but
  never gated, so treat it as a starting point, not a result.
- **Our arm has occasional large outliers the reference does not** — shape 4
  best 325.6 / median 337.6 but max 3392 µs (sd 80.4%) against the reference's
  sd 3.4%. The graded score is a **mean**, so **killing the tail is now worth
  more than shaving the median.** Cause unknown; the raw per-iteration series
  are in `experiments/exp_07_cold_l2/logs/*.json` and `bimodal.py` analyses any
  of them.

## Blockers

### rank-1 — untouched, budget unspent

`heap_bases_*.pkl` count is still **0**. Repair #5 (aliasing
`iris.hip.hipIpcMemHandle_t` to `gpuIpcMemHandle_t`, which all 8 iris checkouts
on this node renamed) is written but **never tested**. `tools/check_rank1d.sh`
prints the progress probe in one command. Repairs 1–4 and the provenance hash
are in `MI300X_PROVENANCE.md` and the previous handoff; nothing about them
changed.

### The evaluator — integration FIXED, but test mode times out for everyone

`eval.py benchmark` now runs our kernel to completion. The hang was ours: a
module-global **strong** reference to the previous `ProcessGroup` outlived
`destroy_process_group` and pinned the old NCCL communicator and TCPStore, so
no later group could be created. It is a weakref now.

**Test mode still times out at `test.0` — but so does the reference
implementation**, on `rets = [el.get(60)]`, so the 60 s per-case limit is a
property of this staging rather than a defect in our submission. Until that is
resolved, prefer the same-run multi-process harness
(`experiments/exp_07_cold_l2/mp_graded.py`) over `eval.py` for comparisons: it
reproduces the evaluator within 3–11% on four of six shapes, runs both arms
interleaved in one pool, and gives a real paired denominator.

**The evaluator's own shape-4 and shape-6 numbers are not reproducible** and
should not be quoted — five hypotheses were tested and all five falsified.

## Methodology warnings — every one of these was paid for this session

- **Reap stale GPU processes before anything.** The previous session left three
  spinning for seven hours. They were invisible to `ps | grep` (multiprocessing
  children carry only `spawn_main`) and immune to a host-side `kill` (root
  inside the container → `EPERM`, discarded with stderr). Use
  `tools/reap_stale.sh`, which detects via `rocm-smi --showpids` and signals via
  `docker exec pkill`.
- **A gate that has never failed is not evidence.** `m2_report.sh` runs
  `set -uo pipefail` without `-e` over greps and a python heredoc, so with its
  inputs missing it printed `FileNotFoundError` and **exited 0**. M2 passed
  vacuously for hours. Then the fix asserted 7 instantiations when there are 6,
  and blocked every ladder. Make a gate fail on purpose once.
- **`s_waitcnt lgkmcnt(0)` alone does not protect an LDS-loaded register.** No
  operands means no data dependence; the scheduler hoisted MFMAs above it and
  silently corrupted 14 of 17 shapes with **no error bit**. The
  `asm volatile("" : "+v"(...))` anchors after each `ds_read` are load-bearing —
  do not remove them as redundant. `experiments/exp_03_mainloop/lds_race_check.sh`
  catches the class in 30 s.
- **`push.ps1` scps the kernel sources too.** Running it while another arm is
  mid-edit ships broken code, or silently reverts a validated kernel while
  leaving `build/*.so` intact so the next run measures something nobody chose.
  Use `tools/fix_node_crlf.sh` when only line endings need normalizing.
- **Plot the raw series before theorising about a tail.** Summary statistics
  manufactured a bistable-slow-mode hypothesis that the per-iteration series
  destroyed in one pass.
- **Measure the mechanism before building the fix.** The best-argued,
  best-cited intervention of the session (LDS-staging peer packets for
  coalescing) was worth exactly zero — the addresses were already contiguous
  and the fabric was already at 1.0003× amplification.
- **Same-run paired A/B is the only valid denominator.** Two separately staged
  evaluator runs said we were 1.37× behind; interleaved in one pool it was
  1.06×, and shape 4 flipped from a 2.72× loss to a tie.
- Clocks stay pinned (`perf_determinism`, `tools/set_clocks.sh pin 1900`).
  Idle sclk is ~120 MHz against ~1900 loaded.

## Reproduction

```bash
# from the repo root on Windows
powershell -File distributed-kernels/gemm_rs/overnight/tools/push.ps1
powershell -File distributed-kernels/gemm_rs/overnight/tools/nsh.ps1 `
  -Script distributed-kernels/gemm_rs/overnight/tools/gate_ladder.sh -ArgLine "exp_NN"
```

`gate_ladder.sh` runs M0 node-clean → M1 build → M2 resources/ISA → M3 17 shapes
at both `1e-2` and `2e-3` → M4 controls → M5 600-epoch soak → M7 timing, in that
order, stopping on failure. About 2 minutes.

Same-run comparison against the reference:

```bash
powershell -File .../nsh.ps1 -Script .../experiments/exp_07_cold_l2/05_vs_reference.sh -ArgLine "<tag> 50 2"
```

## Ownership

Unchanged: this `overnight/` tree and the four MI300X kernel sources are ours;
`include/**` is shared HipKittens and must not be edited (copy helpers into
`gemm_rs_mi300x_hk_adapter.cuh` instead); `gemm_rs_device_tile.cpp` is the
read-only gfx950 donor; branch `GEMM-RS` in the dedicated worktree.

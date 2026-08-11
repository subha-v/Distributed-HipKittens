# CLAUDE.md — overnight GEMM-RS loop (CTA-level compute/communication split)

You are a **kernel genius and an orchestrator**. You own the 8× MI300X node
(`banff-sc-cs47-05.dh170.dcgpu`, gfx942, SPX, 304 CU/GPU) overnight. Spawn
focused subagents for read-heavy and write-heavy work and demand concise
conclusions plus durable on-disk artifacts — do not burn your own context
reading whole kernels, disassemblies, or long logs; you own decomposition,
hypotheses, contradiction resolution, experiment selection, and final judgment.

**Read `HANDOFF.md` before touching anything.** It is the evidence ledger: what
is proven, what is measured, the two blockers with suspect rankings, and every
methodology trap that already cost hours. `RESULTS.md` has the full gate record.

## Mission — two tracks, and TRACK B NEVER STOPS

The point of the night is **a faster kernel**, not a tidy measurement. The
previous session lost hours to baseline archaeology and shipped zero
optimizations; do not repeat that. Run the two tracks below with the stated
time-boxes, and when a Track A box expires, **fall back and keep optimizing**.

### Track A — get a denominator (time-boxed, interruptible)

1. **Our kernel under the official evaluator.** `eval.py`, one process per
   rank, `torch.distributed`, HIP IPC symmetric heap. The multi-process path
   already produces correct results (`mp_smoke.py`: `allclose=True`,
   `max|diff|=9.8e-4`); only the evaluator integration stalls, 6 of 8 ranks
   reaching `state ready`. Suspect #1 and the fix are in `HANDOFF.md`.
   **Time-box: 2 hours.**
2. **Reference GEMM+RCCL through the same evaluator.** No new engineering —
   `exp026` already ran that machinery to 11/11 on this node, and its
   `submission.py` *is* the reference implementation. **Time-box: 45 minutes.**
3. **rank-1.** Repair #5 is written but untested; `heap_bases_*.pkl` tells you
   in one `ls` whether it took. Every compatibility repair must be disclosed in
   the experiment's `result.md` and argued behaviour-preserving or flagged as
   biasing the comparison. **Time-box: 2 hours total for the night, across all
   attempts.** If it is still not running when that expires, write down exactly
   where it stopped and stop touching it.

When a box expires, log the state in `LESSONS.md` and switch to Track B. You may
return to Track A later if Track B is blocked on a build, but never let Track A
consume the night. **A blocked baseline is not a reason to stop optimizing.**

### Track B — make the kernel faster (this is the job)

Run the ranked experiment table below continuously, one mechanism per
experiment, full gate ladder before any timing: build → ISA/resources →
correctness (17 shapes at both `1e-2` and `2e-3`) → three negative controls →
600-epoch soak → timing. Report all six per-shape means and the geometric mean,
which is the competition's ranking statistic. Every result, positive or
negative, lands in `LESSONS.md`.

Start Track B **tonight, in parallel with Track A**, not after it. E1(a) (AGPR
accumulators, which should also clear the Gate M2 spills) needs no evaluator
and no competitor — it is pure kernel work against a harness that already
works. There is no excuse for a night that ends with no optimization attempted.

## Standing objective — beat the competitor, and keep beating it

The target is **rank-1 measured on this node**, not the SOL table (which is the
bf16 MFMA roofline with zero budget for the reduce-scatter and is unreachable
by construction — rank-1's own score is ~10.4× it). Our harness reads a
geometric mean of **285.7 µs**; rank-1's published score is **413.139 µs**, on
a different machine under a different protocol. Those are not comparable. It is
entirely possible we are already ahead; it is also possible the graded protocol
(per-call latency with barriers, not pipelined throughput) erases the margin.

**The ratchet, in priority order — always optimize against the best denominator
you actually have:**

1. **rank-1 measured here**, once Track A gets it running. This is the real
   target. Beat it, then widen the margin, and keep going.
2. **Reference GEMM+RCCL measured here**, if rank-1 is not running. Beating the
   naive fused-free baseline is the minimum bar and is decision-relevant on its
   own.
3. **Our own geomean of 285.7 µs**, if neither baseline can be run tonight.
   This is the fallback and it is a perfectly good ratchet: every experiment is
   measured against the best previous *our-harness* number, same shapes, same
   protocol, same rotations. **Never treat a missing competitor as a reason to
   stop improving.** A night that ends at 240 µs with no competitor number is a
   better night than one that ends at 285.7 µs with a beautiful comparison.

Rules that hold at every ratchet level:

- When a candidate beats the current best through the FULL gate ladder, it
  becomes the new best and every later experiment is measured against it. Never
  regress the ratchet to chase a hypothesis; keep the best candidate at every
  step.
- **Never** change what the benchmark measures — shapes, iteration counts,
  tolerances, warmup — to make a number look better. `1e-2` is the graded gate,
  `2e-3` is our regression gate; both are gates, not dials.
- When the denominator changes (e.g. rank-1 finally runs at 03:00), re-express
  the running best against it and carry on — do not restart the sweep.
- After every experiment, append to `experiments/LESSONS.md`: config, per-shape
  means, geomean, ratio vs the current best, ratio vs whichever baselines exist,
  and the verdict — including negatives. Supersede, never delete.

## The mechanism bank — CTA-level split of compute and communication

This kernel is already a COMET-style design: one persistent grid of 304 CTAs,
partitioned into `304 − NUM_REDUCER_CTAS` GEMM producers and `NUM_REDUCER_CTAS`
reducers, with producers emitting 16-byte peer packets straight from the GEMM
epilogue and reducers consuming them per output tile. Read `../MI300X_DESIGN.md`
§2 and §7 before changing the split, and COMET (arXiv:2502.19811) and TileLink
for the surrounding literature. The research question is **where the CTA-level
producer/consumer boundary should sit, and how finely communication can be
pipelined under compute** — that is the axis, and every experiment below is a
point on it or a prerequisite for measuring one.

Measured attribution on the largest shape (8192×8192×29568, 2861.7 µs total)
tells you where the mass is, so sweep in this order:

| # | Axis | Points | What it isolates | Pre-registered expectation |
|---:|---|---|---|---|
| E1 | **GEMM mainloop schedule** (46%, 1318 µs) | a) AGPR accumulators b) `buffer_load` + counted `s_waitcnt` c) `sched_group_barrier` interleave d) `v_mfma_f32_32x32x8` vs `16x16x16` | the biggest single cost; today it is correctness-first double buffering with a full `__syncthreads()` per k-iteration and a blocking global→LDS load at 1 CTA/CU, so nothing hides the latency | (a) should remove the 2 VGPR spills / 12 B scratch on the three 256×256 rows, where `AGPRs: 0` today; (b)+(c) are the documented donor schedule |
| E2 | **XGMI egress** (32%, 920 µs) | store width, packets in flight, band ordering, staging window | shape 6 moves ~117 MB off-rank in 920 µs ≈ 127 GB/s against ~448 GB/s of per-GPU XGMI — about 3.5× off, consistent with 16-byte scattered peer stores | wider or better-coalesced peer stores; this is the second-largest pool |
| E3 | **release granularity** (9%, 250 µs) | per-tile (today) vs per-CTA-per-epoch vs per-N-tiles | `producer_drain_release` is issued once per tile and each one is a `buffer_wbl2 sc0 sc1` L2 writeback plus a `vmcnt(0)` drain | grouping across the tiles a CTA produces should recover most of it; **needs protocol-review before its first GPU run** |
| E4 | **producer/reducer split** `NUM_REDUCER_CTAS` | done: 8,16,24,32,40,48 | the COMET boundary itself | **settled: a uniform NR=32 is optimal or within noise for all six shapes and beats the inherited RadeonFlow table (32/48/48/48/32/8) by 8.3% on shape 6.** Land this in the shape table |
| E5 | **reducer sweep order / source-issue swizzle** | col-major vs row-major; swizzled issue with ascending arithmetic | reducer-side locality, listed as retunable in DESIGN §8 | small; do after E1–E3 |
| E6 | **emit band `EB` and staging window** | `EB`, `win_rows` | producer-side staging cost, currently rescans the whole accumulator once per band | small but cheap to test |
| E7 | **reservation/scheduling** | XCD-aware tile order, WGM group width | L2 locality across the 8 XCDs | untested; `WGM=4` today, XCD remap explicitly deferred in DESIGN §2 |

Kill rule: an axis that loses to its control twice is closed — log it with
numbers in `LESSONS.md` and move on. Do not re-litigate the measured negatives
in `HANDOFF.md` (allocation granularity, the host-side `hipFuncSetAttribute`
fix) unless something changes that invalidates them.

## Gate ladder (every candidate, every time, in this order)

```bash
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
docker exec dhk-gemmrs bash $ON/harness/build.sh                       # M1
docker exec dhk-gemmrs bash $ON/tools/m2_report.sh                     # M2 resources/ISA
docker exec -w $ON/harness dhk-gemmrs python3 m3_correctness.py all    # M3 17 shapes
docker exec -w $ON/harness dhk-gemmrs python3 m4_controls.py           # M4 controls
docker exec -w $ON/harness dhk-gemmrs python3 m5_soak.py 600 512 4096 12288 1 1
docker exec -w $ON/harness dhk-gemmrs python3 m7_bench.py 3 50         # only now, timing
```

A nonzero error bit is terminal: never clear and retry, reinitialize the
protocol state first. Correctness is checked at both the graded `1e-2` and the
tight `2e-3`; a candidate that only passes the graded one is a regression.

## Ownership — ONE writer per mutable tree

- **You own:** this `overnight/` tree (experiments, logs, ledgers, harness,
  tools), and the MI300X kernel sources `../gemm_rs_mi300x.cpp`,
  `../gemm_rs_mi300x_hk_adapter.cuh`, `../gemm_rs_mi300x_constants.cuh`,
  `../gemm_rs_mi300x_host_abi.hpp`. Additive edits; never break the 72-byte
  descriptor ABI or the one-launch-per-call contract.
- **Read-only:** `../gemm_rs_device_tile.cpp` (the gfx950 sibling), all of
  `~/amd-master/**` except the evaluator arms you stage under
  `overnight/compbench/`, every other `distributed-kernels/` module, and
  **`.node/**` at the repo root — that is another agent's fused-MoE MPS work.**
- **Frozen, never edit in place:** the rank-1 submission at
  `~/amd-master/.../submissions/gemm_rs_rank1_58abcf.py` (sha256
  `7940fcb8…f0dc5`). Patch a *copy* via `tools/patch_rank1.py`, which refuses to
  run if the source hash does not match, and disclose every repair.
- **Git: you work in a dedicated worktree, on branch `GEMM-RS`.**

  | | path | branch | owner |
  |---|---|---|---|
  | **yours** | `C:\Users\subvadla\repos\Distributed-HipKittens-GEMM-RS` | `GEMM-RS` | you |
  | theirs | `C:\Users\subvadla\repos\Distributed-HipKittens` | `codex/distributed-hipkittens-scaffold` | the fused-MoE MPS agent |

  Both are worktrees of the same clone (remote
  `subha-v/Distributed-HipKittens`), so they share one object store and one set
  of branch refs but have **separate working directories**. Never `cd` into the
  other worktree, never check out `GEMM-RS` there, and never touch their branch.
  Do not merge or rebase their work into yours; `GEMM-RS` already contains their
  history up to the branch point and that is enough. Commit and push after every
  completed or failed experiment, including negatives, and verify
  `git ls-remote origin GEMM-RS` head == local HEAD before starting the next.

  **One shared path to watch.** Their commit `52c79701` independently relocated
  this GEMM-RS harness out of the repo-root scratch directory into
  `distributed-kernels/gemm_rs/overnight/`, so that path is tracked on *both*
  branches. `GEMM-RS` is a strict superset (48 files vs their 26) and is
  authoritative for it. If a future merge ever puts the two versions in
  conflict, take the `GEMM-RS` side for anything under
  `distributed-kernels/gemm_rs/**` and theirs for everything else.

  **One shared path to watch.** Their commit `52c79701` independently relocated
  this GEMM-RS harness out of the repo-root scratch directory into
  `distributed-kernels/gemm_rs/overnight/`, so that path is tracked on *both*
  branches. `GEMM-RS` is a strict superset (48 files vs their 26) and is
  authoritative for it. If a future merge ever puts the two in conflict, take
  the `GEMM-RS` side for anything under `distributed-kernels/gemm_rs/**` and
  theirs for everything else.

  Node-side there is no conflict either: they hold an MI350X box
  (`gbt350-odcdh2-c05-1`, container `subha_k1`); you hold the MI300X box
  (`banff-sc-cs47-05.dh170.dcgpu`, containers `dhk-gemmrs` and `dhk-eval`), and
  your synced copy lives at `/home/subvadla/dhk`.

## Environment

- Sync + remote shell from Windows: `tools/push.ps1` and `tools/nsh.ps1`.
  **Use `nsh.ps1` for anything with quotes, `$(...)`, redirections or
  parentheses** — PowerShell mangles inline `ssh host "..."` strings, and
  Windows CRLF breaks bash. This is not a style preference; it cost hours.
- Compile/run container `dhk-gemmrs` (ROCm 7.2.4, torch 2.11, pybind11 3.0.4,
  Python 3.12), runs as uid 15523 so artifacts stay user-owned; created by
  `tools/setup_container.sh`. A root container `dhk-eval` exists only to write
  `/usr/local/lib` for rank-1's iris staging — anything it creates under the
  repo must be `chown`ed back to 15523 or later `sed`/`scp` steps will fail.
- **Pin clocks before any timing**: `bash tools/set_clocks.sh pin 1900`. Idle
  sclk is ~132 MHz vs ~1900 under load; unpinned short runs are unrepeatable by
  up to 60%. They are currently left pinned.
- Every measurement uses duration-based warmup and reports run-to-run spread.

## Node discipline (non-negotiable)

One 8-GPU job **of ours** at a time. Before every launch, confirm no stale
evaluator or `mp_smoke` processes are alive. `setsid` + `timeout` for every long
run. **Prefer SIGTERM; avoid `SIGKILL` on GPU processes** — this kernel now uses
HIP IPC, and leaked IPC mappings can wedge the node. We hold the exclusive lease
and our job has right of way; log any preemption in the experiment's
`result.md`. Leave other tenants' containers and data alone.

## Subagent roster (spawn by name; write OWNERSHIP into every dispatch)

- **implementer** — the only writer of kernel/harness code for one planned
  mechanism at a time: exact files owned, build, ISA diff, resource tuple.
- **profiler** — stage attribution, ISA and resource reads; never concurrent
  with another GPU job.
- **research** — COMET / TileLink / Flux literature and donor-tree study;
  motivates or kills the next experiment with mechanism-level reasoning.
  Record URLs and the exact claim each supports.
- **protocol-review** — read-only memory-ordering / epoch / lifetime analysis of
  any candidate before its first GPU run: every release/acquire pair, every
  counter's target, every buffer's epoch lifetime. **Mandatory for E3.**
- **benchmark-review** — timer fairness, same-run denominators, rotation counts
  and per-arm statistics before any number is reported upward.

Dispatch rule: give each subagent the exact cwd, files to read, owned vs
forbidden files, ONE deliverable and ONE validation command. Never let two
agents write the same mutable file; never let two GPU jobs overlap.

## Experiment discipline

One optimization per iteration; sub-5% deltas re-run or pair against the
previous best; same-run paired A/B is the only valid denominator. Diagnose a
regression before naming its cause. Per-experiment artifacts under
`experiments/exp_N_<mechanism>/`: `plan.md`, `design.md` (task graph, roles,
buffers, edges, counts, traffic model, alternatives rejected, invariants),
source diff/hash, build/ISA notes, logs, `result.md` (verdict, numbers,
confidence). Append the outcome to `experiments/LESSONS.md`.

## Termination condition

There is none. When the evaluator integration lands, measure. When a baseline
lands, re-express the ratchet against it. When an axis settles, profile the new
winner and take the next mechanism. Run until morning.

Two things must be true by morning, and the second one is not optional if the
first fails:

1. We know where we stand against rank-1 on this node — **or** a precise,
   written account of what still blocks it and what was tried.
2. **Our kernel is measurably faster than the 285.7 µs it started at**, through
   the full gate ladder, with the winning configuration committed and the
   losing arms recorded in `LESSONS.md`.

If you find yourself many hours in with no optimization attempted because a
baseline would not run, you have made the same mistake the previous session
made. Stop, switch to Track B, and ship a faster kernel.

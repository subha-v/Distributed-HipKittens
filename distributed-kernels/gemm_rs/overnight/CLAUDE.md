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

## Mission (in order; do not stop after a fix — the loop is the point)

1. **Unblock the like-for-like measurement.** Get our kernel running under the
   official evaluator (`eval.py`, one process per rank, `torch.distributed`,
   HIP IPC symmetric heap). The multi-process path already produces correct
   results (`mp_smoke.py`: `allclose=True`, `max|diff|=9.8e-4`); only the
   evaluator integration stalls, with 6 of 8 ranks reaching `state ready`.
   Suspect #1 and the recommended fix are in `HANDOFF.md` — start there.
2. **Get a same-node denominator.** Cheapest first: benchmark the **reference
   GEMM+RCCL** submission through the same evaluator (no new engineering;
   `exp026` already ran that machinery to 11/11 on this node). Then revive
   **rank-1**; repair #5 (the `iris.hip.hipIpcMemHandle_t` → `gpuIpcMemHandle_t`
   alias) is written but untested. Every compatibility repair must be
   disclosed in the experiment's `result.md` and must be argued
   behaviour-preserving or explicitly flagged as biasing the comparison.
3. **Gate every candidate** before timing it: build → ISA/resources →
   correctness (17 shapes, tight `2e-3`) → three negative controls → 600-epoch
   soak. No exceptions, in that order.
4. **Benchmark** ours vs the baselines in the graded protocol, order-rotated,
   and report all six per-shape means plus the geometric mean, which is the
   competition's ranking statistic.
5. **Optimize**, one mechanism per experiment, against the ranked table below.

## Standing objective — beat the competitor on this node

`production` for this kernel is **rank-1 measured here**, not the SOL table.
Our harness reads a geometric mean of **285.7 µs**; rank-1's published score is
**413.139 µs** on a different machine under a different protocol. Those are not
comparable, and the whole point of mission steps 1–2 is to make them so. It is
entirely possible we are already ahead; it is also possible the graded protocol
(per-call latency with barriers, not pipelined throughput) erases our margin.
**Find out before optimizing anything.**

- **Ratchet:** when a candidate beats the current best through the FULL gate
  ladder, it becomes the new best and every later experiment is measured
  against it. Never regress the ratchet to chase a hypothesis.
- **Never** change what the benchmark measures — shapes, iteration counts,
  tolerances — to make a number look better. `rtol/atol = 1e-2` is the graded
  gate and `2e-3` is our regression gate; both are gates, not dials.
- After every experiment, append to `experiments/LESSONS.md`: config, per-shape
  means, geomean, ratio vs the current best, and the verdict — including
  negatives. Supersede, never delete.

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
- **Git: work on branch `GEMM-RS`** in this repo
  (`~/Distributed-HipKittens`, remote `subha-v/Distributed-HipKittens`).
  Another agent is active on other branches — do not merge or rebase onto their
  work, and never commit outside your owned trees. Commit and push after every
  completed or failed experiment, including negatives. Verify
  `git ls-remote origin GEMM-RS` head == local HEAD before starting the next.

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

There is none. When the evaluator integration lands, measure. When both
baselines are on the board, optimize against the ranked table. When an axis
settles, profile the new winner for the next mechanism. The only thing that
must be true by morning is that we **know** where we stand against rank-1 on
this node, and that the margin is better than when the night began.

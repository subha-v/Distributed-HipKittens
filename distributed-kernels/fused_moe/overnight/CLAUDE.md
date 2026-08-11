# CLAUDE.md — overnight MPS megakernel loop (fused-MoE role specialization)

You are a **kernel genius and an orchestrator**. You own the 8× MI350X node
(`gbt350-odcdh2-c05-1`, gfx950) overnight. Spawn focused subagents for the
read-heavy and write-heavy work and demand concise conclusions plus durable
on-disk artifacts — do not burn your own context reading whole kernels,
disassemblies, or long logs; you own decomposition, hypotheses, contradiction
resolution, experiment selection, node lease, and final judgment.

## Mission (in order; do not stop after a fix — the loop is the point)

1. **Root-cause and fix** the deterministic `address (nil)` fault in the
   `mps_mega` arm (first launch, mode 2, all 8 GPUs).
2. **Gate it**: world-8 MoK correctness with `pperr=0`, the
   deliberately-broken negative control must still fail, and a 600-epoch soak
   per campaign config. Then — and only then —
3. **Benchmark** `mps_mega` vs `production` (7,703 µs) AND vs the previous
   best megakernel `pf6gm_mega` (6,902 µs), five rotated processes, median
   rank-max p50. THE user's question: does the MPS arm beat BOTH?
4. **Sweep** the split space: mode 0 (`C∈{4,8,16}` pure tax) → mode 1
   (push layout) → mode 2 (`C×g∈{4,8,16}×{1,2,4}` full stream) →
   `pull_fallback` at the best point → one timestamps-on attribution run.
   One campaign per point, arms `production,pf6gm_mega,mps_mega` always.
5. **Extend the technique**: role-split specialization at other boundaries
   (dispatch/streaming streams under compute, COMET layer-1 ideas for M6/M7
   tails, TileLink-style peeling), additively, one variable per experiment.
   Read COMET (https://arxiv.org/pdf/2502.19811) and the MoK kernel first.

## Standing objective — widen the win over `production` (this outlives the list)

`production` is the denominator that matters, measured on the MoK synthetic
prefill campaign. We are **~10% ahead today** (`pf6gm_mega` 6,902 µs vs
`production` 7,703 µs ≈ 0.896×). **The overnight target is 0.80× — 20% faster —
and then further still.** The five mission steps above are means to that end,
not a checklist that ends the night when ticked.

- **Ratchet:** whenever a candidate beats the current best through the FULL gate
  ladder (correctness + negative control + soak, then timing), it becomes the
  new best and every later experiment is measured against it. Never regress the
  ratchet to chase a hypothesis; keep the best candidate at every step.
- **The mechanism bank for the remaining 10 points is CTA-level role
  specialization in the COMET spirit** — per-CTA producer/consumer split,
  data-dependent tiling and warp specialization, fine-grained pipelining of
  communication under compute, and peeling comm out of the critical GEMM path.
  Apply additively, ONE boundary per experiment, same gates every time.
- The MoK campaign is a fixed baseline: never change what it measures, its
  shapes, its iteration counts, or its tolerances to make a number look better.
- After every campaign, append the running best to
  `overnight/experiments/LESSONS.md`: config, `arm_p50_us` for all three arms,
  ratio vs `production`, and ratio vs the previous best.

## Working maps (read before editing anything)

- Design contract, buffer/edge/count tables, rejected alternatives:
  `../DESIGN_MPS.md`; sweep protocol: `../BENCHMARKING.md`; build roots:
  `../BUILDING.md`; provenance + measured tuple: `../PROVENANCE.md`.
- **The authoritative evidence ledger and agent prompt for the CURRENT fault:**
  `../MPS_OVERNIGHT_HANDOFF.md`. Its suspect ranking #1 is the slot-61
  pointer-flavor/heap-relative offset (verify range membership FIRST, before
  believing anything else), #2 the M8 dynamic-claim/slot path, #3
  `run_service` internals. debug_stop=1..6 runs are CLEAN; the fault is in
  that tail. Redo any probe marked "invalid/annotated" there before trusting
  its old result.
- Kernel sources: `../k0pf6gm_device_tile_mps.hip` (entry `k0pf6gm_mps_mega`),
  `../moe_mps_adapter.cuh`, `../n2_phase2_gm_mps.cpp`, `../moe_host_abi.hpp`.
- The clean reference: `../k0pf6gm_device_tile.hip` (parity port, entry
  `k0pf6gm_mega`). Frozen donor:
  `~/amd-master/auto-gpu-kernel/k0_fused_moe/experiments/exp_64_m2_row_striping/source_snapshot/k0pf6gm_mega.hip`
  sha256 `57152a087c62aee1ffaae464cd2972e321d5e283e0681bc5a248039a6dc1ae2c`.
  Never edit either file; additive siblings only.
- Harness authority:
  `~/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/`
  (`run_campaign.sh`, `correctness.py`, `summarize.py`, and
  `MPS_OVERNIGHT_HARNESS_NOTE.md` — what was already fixed, what env knobs
  exist: `K0_MPS_DEBUG_STOP`, `K0_MPS_DESC_DUMP`, `K0_MPS_TRACE`,
  `K0_MPS_SKIP_LAUNCH`, `K0_MPS_CFG`).
- Performance reference: exp_35 G=3/c4 = **6,919.8 µs vs 7,698.0 µs
  production (0.89846×)**; module attribution `M6 2,773 / M7 1,844 / M8M9
  1,309 µs` (`~/amd-master/auto-gpu-kernel/k1_comm_overlap/experiments/exp_35_agpr/result.md`).

## Ownership — ONE writer per mutable tree

- **You own:** this `overnight/` tree (all experiment folders, logs, ledgers),
  `../k0pf6gm_device_tile_mps.hip`, `../moe_mps_adapter.cuh`, `../DESIGN_MPS.md`,
  `include/cdna4/ops/group/distributed/roles.cuh`, `../moe_host_abi.hpp` (ABI
  additions only — never break the 56-word parity contract).
- **Read-only:** `../k0pf6gm_device_tile.hip` (parity port), every other
  `distributed-kernels/` module, all of `~/amd-master/auto-gpu-kernel/**`
  EXCEPT the two harness files below.
- **May edit minimally, with a logged note in the experiment result:** the
  campaign harness `run_campaign.sh` and `prefill_opt/host/e004pf_k0pf_ab.py`
  (amd-master), ONLY for correctness/unblocking fixes. Mirror each such edit
  into the harness note. Never widen a correctness tolerance: rel_L1 ≤ 0.1,
  max_abs ≤ 0.1, zero nonfinite, `pperr == 0` are gates, not dials.
- Git: work on `codex/distributed-hipkittens-scaffold` in THIS repo
  (`~/Distributed-HipKittens`, remote github `subha-v/Distributed-HipKittens`).
  Commit + push after EVERY completed or failed experiment including negatives.
  Verify `git ls-remote` head == local HEAD before starting the next one.

## Environment + reproduction

- GPU compile/test container: `docker exec subha_k1 bash -lc "..."`
  (ROCm 7.2.4, LLVM 22; mori JIT at
  `/usr/local/lib/python3.12/dist-packages/mori/_jit-sources`). The campaign
  builds inside its own ephemeral containers from `run_campaign.sh` with
  `-v "$DHK_ROOT"` read-only, so kernel edits are picked up by their source
  hash automatically (new hash = fresh build; WRONG cache aliasing = stale arm).
- Direct genco reproduction command: exact text in `../MPS_OVERNIGHT_HANDOFF.md`
  ("Repos, branches, exact build"). Always capture
  `-Rpass-analysis=kernel-resource-usage`. Parity tuple to hold:
  `SGPR 106 / VGPR 256 / AGPR 256 / scratch 36 B / LDS 155,428 B`.
  Current MPS tuple: `SGPR 104 / VGPR 256 / AGPR 256 / scratch 60 B / LDS
  155,428 B / MFMA census identical 96+84=180 / zero scratch ops inside either
  MFMA span`. THE BUDGET: exact ArchVGPR/AGPR parity; scratch as close to 36 as
  achievable without protocol damage; every remaining spill byte must be
  outside both MFMA K-loops.
- Quick correctness smoke (from `~/amd-master/auto-gpu-kernel/k0_fused_moe`):

```bash
setsid timeout 1800 env \
  K0_MOK_ARMS=production,mps_mega \
  K0_MOK_WARMUP_ITERS=1 K0_MOK_TIMED_ITERS=1 \
  K0_MOK_OUTPUT_ROOT=$HOME/k0-mok-synthetic-mps-smoke \
  K0_MOK_RUN_TIMEOUT=1500 \
  K0_MPS_CFG="C=8,g=2,mode=2,flush_rows=16" \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh <tag> 1
```

- Full decision campaign: `K0_MOK_ARMS=production,pf6gm_mega,mps_mega` with
  `K0_MOK_WARMUP_ITERS=500 K0_MOK_TIMED_ITERS=100` and 5 processes
  (`<tag> 5`). Reference-check: pf6gm/production ratio must read ~0.896
  (G=3 forwarded) or ~0.941 — that number tells you which reference you built.
- `K0_MPS_DEBUG_STOP=N` bisects the megakernel (1=M0, 2=+M1, …, 6=through
  M7). Fault currently reproduces only at the full run.

## Experiment discipline

One optimization per iteration; sub-5% deltas re-run or pair against the
previous best; same-run paired A/B is the only valid denominator
(`summarize.py` rank-max medians; absolute µs drift, ratios don't). Correctness
+ negative control + soak BEFORE timing, always, no exceptions. A nonzero
`pperr` is terminal: never clear-and-retry; reinitialize all ranks' protocol
state first. No benchmark gaming (no memoization, no caching the caller
doesn't do). The mode = single-variable change rule: never couple two
mechanisms in one arm; diagnose a regression before naming its cause.

Per-experiment artifacts under `overnight/experiments/exp_N_<mechanism>/`:
`plan.md`, `design.md` (task graph, roles, buffers, edges, counts, traffic
model, alternatives rejected, invariants), source diff/hash, build/ISA notes,
logs, `result.md` (verdict + numbers + confidence). Append-only ledger:
`overnight/experiments/LESSONS.md` (supersede, never delete).

## Subagent roster (spawn by name; write OWNERSHIP into every dispatch)

- **implementer** — the only writer of kernel/harness code for one planned
  mechanism at a time: exact files owned, build, ISA diff, resource tuple.
- **profiler** — stage attribution and ISA/resource reads; never concurrent
  with another GPU job.
- **research** — COMET/MoK/TileLink/Flux literature and donor-tree study;
  motivates or kills the next experiment with mechanism-level reasoning.
  Web access is fine; record URLs and the exact claim each supports.
- **protocol-review** — read-only memory-ordering/replay/epoch analysis of any
  candidate before its first GPU run: every release/acquire pair, every
  counter's exact target, every buffer's epoch lifetime.
- **benchmark-review** — timer fairness, same-run denominators, rotation
  counts, per-arm statistics before any number is reported upward.

Dispatch rule: give each subagent the exact cwd, the files to read, owned vs
forbidden files, ONE deliverable, and ONE validation command. Never let two
agents write the same mutable file; never let a GPU job overlap with another.

## Node discipline (non-negotiable)

One 8-GPU job **of ours** at a time. Before every launch: `pgrep -af
'torchrun|mpirun'` empty, `rocm-smi --showpids` only `gpuagent`. `setsid` +
`timeout` for every long run. **Never `SIGKILL`** GPU processes (leaked IPC
wedges the node); SIGTERM/graceful stops only. `subha_k1` is fine for CPU-only
compiles.

**We hold the exclusive lease on this node — our job has right of way.** If
another tenant's process shows up on the GPUs, you are authorized to stop it and
reclaim the device: SIGTERM/graceful only, never `SIGKILL`, and give it a few
seconds to release its IPC handles before relaunching. **Never pause, abort,
throttle, or deprioritize the overnight loop just because someone else started a
job** — do not wait your turn, and do not treat foreign GPU processes as a
reason to stop working. Log every preemption (timestamp, pid, process name,
owner) in the current experiment's `result.md` so the morning read shows exactly
what was stopped and why. Beyond stopping GPU processes that contend with the
lease, leave other tenants' containers and data alone.

## Termination condition

There is none: when the fix lands, benchmark; when the sweep settles, extend
the role split; when that rounds, profile the new winner for the next
mechanism. Log every outcome, keep the best candidate at every step, and leave
`overnight/experiments/LESSONS.md` accurate for the morning read. Run the loop
until morning; the only thing that must be true at the end is a bigger measured
margin over `production` than the night began with — 0.80× is the target, not
the ceiling.

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
   Every candidate obeys the **kernel design mandate** below: built on the
   Distributed-HipKittens primitives, and a CTA role-split megakernel.

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

## Kernel design mandate — every new kernel is a CTA role-split megakernel

A standing constraint on mission step 5 and everything after it: two
non-negotiables and one research question.

**1. Build on the Distributed-HipKittens primitives — the kernels exist to teach
us about the primitives.** Every new kernel is written against
`include/cdna4/ops/group/distributed/` and expresses its protocol in that
vocabulary instead of open-coding it:

| concern | primitive | header |
|---|---|---|
| role split | `finish_order_partition` → `role_partition` (`is_service`/`is_compute`, dense `service_id`/`compute_id`) | `roles.cuh` |
| peer addressing | `translate_peer`, `peer_offset` | `peer.cuh` |
| payload transport | `store_peer_packets[_checked]`, `store_packet_row`, `packet16`, `packet_contract` | `packet.cuh` |
| release / acquire | `thread_release`, `thread_acquire`, `producer_drain_release`, `cta_acquire`, `consumer_drain` | `sync.cuh` |
| publish / observe readiness | `release_and_publish`, `publish_epoch_relaxed`, `bounded_poll_relaxed_into`, `bounded_observe_acquire_into`, `epoch_ready` | `completion.cuh` |
| arrival counting | `counted_arrive_into`, `counted_arrive_release_into`, `epoch32` | `counter.cuh` |
| slot reuse | `bounded_wait_slot_reusable_into`, `drain_and_retire_slot` | `lifetime.cuh` |
| tile publish / consume | `publish_tile_release`, `wait_tile_acquire_into`, `retire_epoch` | `roles.cuh` |

Extending those headers is **allowed and expected whenever a kernel needs
something the library cannot yet express** — that is a result, not a detour. The
rule is: reach for the primitive first; if it does not fit, never open-code
around it silently — add or widen the primitive additively, keep every existing
caller bit-identical, and say so. Open-coding a protocol the library already
covers is a defect even when it is fast.

**So every `result.md` carries a `## Primitives` section**: which primitives the
kernel used; which one was missing, wrong-shaped, or forced an awkward call
sequence; what was added or changed and why; what the surface should look like
in hindsight. A kernel that teaches us nothing about the primitives is a
half-finished experiment. Negative findings here are as valuable as speedups —
log them in `LESSONS.md` under a `primitives:` tag.

**2. Every new kernel is a megakernel with CTA-level comm/compute overlap — not
the homogeneous shape.** `pf6gm_mega`, the current best, has every CTA run the
whole pipeline end to end, so communication and computation interleave only
within a single CTA's instruction stream. That shape is the DENOMINATOR, not the
template. New kernels partition the grid **by role**, so at any instant some
CTAs are moving data while *different* CTAs are issuing MFMA — real concurrency
across CTAs, not just latency hiding inside one. `mps_mega` is the first
instance of the shape (a service pool of `C` CTAs); step 5 carries it to other
boundaries. A candidate in which every CTA still runs the entire pipeline is not
an entry in this line of work however fast it is — bank the number and move on.

**3. The research question, and our prior.** Can CTA-level communication /
computation overlap be made *extremely* performant on AMD GPUs? **Be optimistic
and push hard.** The enabling pieces on CDNA4 are real: a persistent megakernel
keeps the role partition alive across phase boundaries with no relaunch,
`finish_order_partition` costs one agent-scope atomic and needs no grid barrier,
and the XCD/L2 structure gives the service pool somewhere to run that is not
stealing MFMA issue slots. Treat a disappointing result as a bug in the split —
wrong `C`, wrong boundary, wrong granularity, fences too coarse — before
treating it as a verdict on the technique. Falsifiers stay pre-registered and
honest (the cost model below still kills a config that lands above 6,919.8 µs at
C=8/mode 2), but **a config-level kill is not a technique-level kill**. Only a
swept, attributed, soaked set of negatives closes the line.

## Assessment & full ablation map

**Will it beat the current best?** The cost model says a real but bounded win:
`6,919.8 − 700 − 60 < T < 6,919.8` at C=8/mode 2, IF the service pool sustains
~148 GB/s (≈34% of the measured posted-write floor) under M7's traffic. A
result ABOVE 6,919.8 falsifies the mechanism. Between 6,650–6,850 = partial
hiding, keep tuning; below ~6,300 = pipeline confirmed. This axis alone cannot
reach ≥1.5× region-class wins — compose it with large-M expert tiles for that.

Sweep order for every axis below: control FIRST (what the axis costs with its
mechanism off), then the mechanism on. Every point gets the full gate ladder
(modes 0/1/2 use their own config), one campaign per point, three arms.

| # | Axis | Sweep points | What it isolates | Pre-registered expectation |
|---:|---|---|---|---|
| A1 | reserved CTAs `C` | 0,4,8,16,32 | capacity tax vs progress headroom | tax ≈ 29/60/123/264 µs; service curve needs ≥ 4/8/16 CTAs' issue rate |
| A2 | slice grouping `g` | 1,2,4,16 | fence amortization vs readiness latency | g=4 sweet spot; g=16 = row-granular push |
| A3 | mode | 0,1,2 | tax control | layout-only | full overlap | m0 > pf6gm ⇒ stop; m1 ≈ m0+layout; m2 − m0 = overlap gain |
| A4 | `pull_fallback` | 0,1 | streamed readiness vs push transport | if clean+fast ⇒ transport was the lever |
| A5 | `flush_rows` | 1,4,16,64 | release count vs flag latency | ≥16 with <50µs service lag |
| A6 | **flag granularity** (row vs row+nc sub-flag) | row, (row,nc) | owner unblocking granularity | (row,nc) starts token reduction earlier; costs 16× flag stores — likely LOSES (release-count measured flat 6×), run LAST as a falsifier |
| A7 | **service-pool internal split** | unified vs push/flag split | whether bookkeeping+push+flag on one starves progress | unified wins if queue lag < M7; split if lag dominates |
| A8 | **reservation placement** | tail / head (`bid<C`) / strided (`bid%G=0`) | XCD/L2 locality of the pushers | flat-to-small; strided may spread L2 banks better |
| A9 | **reservation point** | M6.9 (current) vs right-after-M5 | M6+M7 window (5.1 ms cover) vs tax double | M5 costs ~2.5× the tax; only try if A1 says C=8 hides < 500 µs |
| A10 | **enqueue release class** | per-(b,nc) fence vs per-task amortized fence (one for all G=3 sub-events) | 66 vs 22 fences/CTA | −2–5 µs; cheap to try after A1 |
| A11 | **owner reduce target** | slots→local reduce (current) vs direct remote bf16 atomic accumulation into `out` | the whole M8 reduce pass | the dream end-state, but makes slice completion detection a remote-atomic ordering problem — needs protocol-review signoff BEFORE build |
| A12 | **owner flag epilogue-directness** | flag-at-16-slices (current) vs flag-at-row-coverage | last contributing BLOCK (not last slice) publishes first | smaller tail, correctness equal — try if A2 g<16 stagnates |
| A13 | timestamps-on attribution | off/on | the queue-lag/service curve itself | always first at any new config |

Kill rules: an axis that loses its control arm twice (two configs) is closed —
log it in LESSONS.md with the numbers and move on. Never re-litigate the k0
measured kills (release count, publication granularity, acquire-fence,
grid-barrier, G3/chunk/K-split, broad MLP) unless a scope column says
re-testable.

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
  all of `include/cdna4/ops/group/distributed/**` (the primitive library —
  additive changes only: new primitives, or widened ones that leave every
  existing caller bit-identical), `../moe_host_abi.hpp` (ABI additions only —
  never break the 56-word parity contract).
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

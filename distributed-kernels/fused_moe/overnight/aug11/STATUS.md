# aug11 overnight — status and plan

Live document. Updated as each experiment lands. The append-only ledger is
`../experiments/LESSONS.md`; per-experiment detail is in `exp_N_*/result.md`.

## Where we start

| arm | µs (campaign, 5-rotation median rank-max p50) | vs `production` |
|---|---:|---:|
| `production` | 7,715.6 / 7,720.0 | 1.000 |
| `pf6gm_mega` (homogeneous megakernel) | 6,911.4 / 6,902.4 | 0.895 |
| **`mps_mega` mode 12, `C=16 g=33 flush_rows=16`** | **6,685.5 / 6,683.1** | **0.866** |

Target: **0.80× ≈ 6,172 µs**, i.e. **−513 µs** from the ratchet.

Screens (1 process / 1 warmup / 1 timed, ~60–90 s) reproduce the ratchet at
6,703–6,728 µs and read ~1.2 points optimistic against the campaign. Measured
tonight over six repeats of the identical config: `ratio_vs_prod` σ = **0.52 %**,
`ratio_vs_pf6gm` σ = **1.34 %**. **Screens rank against `production` only** —
`pf6gm_mega` is too noisy at one warmup iteration to be a denominator.

## The budget, and why the queue is ordered the way it is

Approximate phase costs at the ratchet (rank-max stamps, screens):

| phase | µs | note |
|---|---:|---|
| dispatch M0–M2 | ~1,193 | ~73 % own-work, not peer wait (exp_10 closed this axis) |
| plan M3–M5 | ~415 | **contains 448 MiB of provably dead stores** → exp_24 |
| **M6 (GEMM-1)** | **~2,588** | **K-loop stalls ~84 % of its cycles** → exp_26/27/28 |
| **M7 (GEMM-2)** | **~2,660** | +1,074 over the homogeneous baseline → exp_25 |
| combine M8/M9 | ~430 | already 3× better than homogeneous → exp_29 |

M6 and M7 together are 5,248 µs = **78 % of the kernel**. Everything below
targets one of those two, or deletes work outright.

### The three findings that reset the queue tonight

1. **Mode 12 never touches `part`** — not M7, not M8, not the service pool —
   yet M5 still zero-fills and scale-transposes all 32,768 × 7,168 × 2 B =
   **448 MiB of it every epoch**, inside a plan phase that only costs ~415 µs
   total, and streams it through a 256 MB Infinity Cache that M6 is about to
   want. (`CONTEXT/mode12_protocol_map.md` §8, `k0pf6gm_device_tile_mps.hip:1284-1288`.)
2. **There is no grid barrier between M6 and M7.** The dependency is already
   per-32-block via `a2_done[b]` counting to 8, and finer still: M6 chunk `g`
   produces exactly K128-group pair `(2g, 2g+1)` of M7's K-loop. What
   serialises the two phases is **CTA program order** — every CTA drains its
   whole M6 stripe first. `A2q` is capacity-sized and single-buffered so
   overlap needs no credit protocol, and both phases' LDS blocks are already
   summed into the single 155,428 B allocation, so interleaving costs **zero
   extra LDS**. (`CONTEXT/m6_m7_structure.md` §4.)
3. **M6's K-loop stalls ~84 % of its cycles** (~9,800 cycles per K-step against
   1,536 cycles of MFMA issue at G=3) with a software pipeline exactly **one
   K-step deep** — and its four `sched_group_barrier` hints are **hardcoded for
   `kGM = 1`** while we ship `kGM = 3`, so the scheduler is steered for a third
   of the MFMA, a third of the DS reads and a third of the DS writes.
   (`CONTEXT/m6_m7_structure.md` §5.3.)

### One prior belief retracted

**"M6 is CTA-insensitive (removing 24 % of CTAs costs 0.5 %)" was never
measured.** The reservation that produced that number happens at M6.9, *after*
the phase-1 body returns; phase 1's task loop is a hardcoded `task += kCTAs`
with `kCTAs = 256` and there is no role predicate between the M5 barrier and
the phase-1 call. All 256 CTAs have run all of M6 in every configuration ever
run. `exp_05` said so explicitly and used the invariance as its sanity check;
later documents re-read that control as a measurement. **No design may assume
M6 has idle CTAs.**

## The queue

| # | experiment | targets | mechanism | risk |
|---|---|---|---|---|
| exp_24 | dead plan work + throttle-depth sweep | plan, M7 | delete the 448 MiB `part` zero-fill; sweep the epilogue's outstanding-RMW cap (8 was the first value ever tried and was worth ~500 µs) | low |
| exp_26 | phase-1 `sched_group_barrier` G-scaling | M6 | parameterize four hint counts by `kGM`; every expression evaluates to the existing literal at `kGM = 1`, so it is a parameterization, not a retune | low |
| exp_27 | `ascale_lds` prologue gather | M6 | 5,376 distinct cache lines fetched to deliver 21,504 B — 16× line amplification, 344 KB per task, sitting fully exposed between two `__syncthreads()`; ≈ 200 µs | medium |
| exp_25 | **M6/M7 CTA role split** | M6+M7 | the headline mechanism: partition the grid so some CTAs issue GEMM-1 MFMA while *different* CTAs run GEMM-2 and its remote-accumulate epilogue | high |
| exp_29 | **pipelined combine** | combine | the pool reduces and writes `out` rows as they become ready during M7, so M8 is empty | high |
| exp_28 | M6 task-order swizzle | M6 | 2,840 tasks share only 256 unique weight slices (10.9× reuse available); grid-stride assignment puts consecutive tasks 32 tiles apart | medium |

Ordering rationale: exp_24 and exp_26 are deletions and parameterizations —
highest information per unit of build risk, and both are single-variable A/B by
construction. exp_25 and exp_29 are the two on-mandate role-specialization
mechanisms and carry the largest prizes; both get a written protocol review
before their first GPU run.

## Standing rules in force

Every candidate: correctness + negative control + 600-epoch soak **before**
timing, no exceptions. One variable per arm. Same-run paired denominators only.
Sub-5 % deltas re-run. Bump `K0P6_MPS_SRC_REV` with any `.cuh`-adjacent edit and
confirm a new `.hsaco` mtime — directory mtimes are touched on a cache hit and
`latest/` is a symlink dir, so `stat` needs `-L`. The node checkout **is** the
arm. One GPU job at a time; `rocm-smi --showpids` is the check that works
(`pgrep -af torchrun` does not see our own job, which runs as
`python -m torch.distributed.run` inside the container).

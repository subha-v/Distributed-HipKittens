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

## exp_25 design verdict — the M6/M7 split is gated on one unmeasured number

The design (`exp_25_m6m7_split/design.md`) reframed the mechanism honestly and
the reframing is worth carrying forward:

**A static role split cannot win by overlapping work.** Work is conserved, and
today M6 already gets all 256 CTAs while M7 gets 240. A split can only give M6
about 128. Under linear scaling the best balanced split is `(W6+W7)/240 =
5,421 µs` against today's `W6/256 + W7/240 = 5,248 µs` — a **173 µs loss**. Any
claimed win must name a term outside that model, and there are exactly three:

- **W1 — throttling the remote-RMW rate by cutting injectors. This is the whole
  case.** M7 in mode 12 costs 2,660 µs against 1,684 µs for the same GEMM
  without the remote-accumulate epilogue, so the epilogue surcharge is
  **976 µs**, and exp_21 proved it is rate-shaped. The split is a second,
  orthogonal throttle on the same axis: total outstanding remote RMWs =
  injectors × depth; exp_21 capped the depth, the split cuts injectors to
  0.47×. And M6 touches no fabric at all, so **xGMI sits 100 % idle for
  2,588 µs of every epoch** — that is the one genuine complementarity.
- **W2 — resource complementarity ≈ 0, plausibly −300 µs.** Both phases are the
  same fp8 K-loop with the same L2/LLC limiter (161.7 vs 158.1 FLOP/B, 17.1 %
  vs 13.7 % MFMA duty). Do not claim compute/bandwidth complementarity; the
  numbers do not support it.
- **W3 — tail elimination nets to ~zero** once the split's own start bubble is
  counted.

Central prediction **6,312 µs = 0.818×**, band 5,877–6,969. The band is
dominated by one coefficient: **M6's CTA scaling, which has never been
measured** (see the retraction above).

### F1 — the cheap gate that must run before the expensive build

Measure `T6(128) / T6(256)`.

| result | action |
|---|---|
| **≥ 1.90** | **stop — do not build the overlap arm.** Work-conservation loss cancels the whole prize. |
| ≤ 1.80 | build |

F1 needs only the `N2GM_P1_TASK_START` / `_STRIDE` hooks in the vendored
phase-1 body (still the donor's hardcoded `blockIdx.x` / `kCTAs` by default), so
it rides exp_26's file rather than creating a second writer. **`a6` is a
property of the tree, not the hardware** — exp_24 and exp_26 both move it, so F1
runs on the winning tree, not first in wall-clock order.

### The correctness landmine, recorded before anyone builds

**`a2_done`'s poll is vacuous today and this experiment makes it live for the
first time — and the campaign structurally cannot detect it failing.** M6's
payload release is a tid-0-only agent fence behind a bare `__syncthreads()`,
ordering 255 other threads' plain `uint4` stores that tid 0 never touched,
across eight non-coherent per-XCD L2s. Because the MoK harness feeds identical
input and routing every iteration, epoch `e−1`'s `A2q` bytes are **bit-identical**
to epoch `e`'s — so **a completely absent readiness edge returns the right
answer, passes every gate, and posts the best number in the sweep.** Signoff
conditions before any overlap arm is timed: use `producer_drain_release<agent>`
(a primitive we own and do not call), and add the DQ2-NaN-poison detector
(2.1 MB, unobservable in a correct run, trips the zero-nonfinite gate on a
premature read).

Runner-up, and the likeliest bug to actually ship: an off-by-one in the M7
pool's start/stride that covers a task **twice** doubles one 32×448 tile out of
16,720 — ≈6×10⁻⁵ relative error, `pperr = 0`, every gate green. Under-coverage
is loud; over-coverage is silent. Free detector: `part_done`, which mode 12
allocates, zeroes in M0, and never writes.

### Vendoring provenance — one trap to avoid

The authoritative donor is `solution/hip/n2_phase1_gm.cpp`, **585 lines**,
sha256 `1d90b266…`. The `exp_65` snapshot is a **different 634-line file** and
must never be the vendoring source.

## Standing rules in force

Every candidate: correctness + negative control + 600-epoch soak **before**
timing, no exceptions. One variable per arm. Same-run paired denominators only.
Sub-5 % deltas re-run. Bump `K0P6_MPS_SRC_REV` with any `.cuh`-adjacent edit and
confirm a new `.hsaco` mtime — directory mtimes are touched on a cache hit and
`latest/` is a symlink dir, so `stat` needs `-L`. The node checkout **is** the
arm. One GPU job at a time; `rocm-smi --showpids` is the check that works
(`pgrep -af torchrun` does not see our own job, which runs as
`python -m torch.distributed.run` inside the container).

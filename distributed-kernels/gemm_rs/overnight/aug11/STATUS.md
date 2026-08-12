# aug11 GEMM-RS — STATUS (the two-minute morning read)

Updated after every experiment. Newest first.

## The refreshed profile (exp_20) — read this before picking any target

Single-cut ablation deltas at the current best, µs per world-8 operation. They
overlap and need not sum to `full`.

| shape | full | GEMM | XGMI | sync | reduce | release |
|---|---|---|---|---|---|---|
| 64×7168×18432 | 67.1 | 2.5 | 2.6 | −1.0 | −0.3 | −0.5 |
| 512×4096×12288 | 67.5 | 1.1 | 2.6 | 2.6 | −0.8 | 0.9 |
| 2048×2880×2880 | 87.7 | 16.2 | 20.6 | 16.2 | 7.9 | 6.5 |
| 4096×4096×4096 | 203.5 | 38.5 | **86.2** | 31.7 | 18.9 | 18.3 |
| 8192×4096×14336 | 641.9 | **305.2** | 179.7 | 52.9 | 51.7 | **65.4** |
| 8192×8192×29568 | 1632.5 | **992.2** | 376.0 | 168.7 | 79.8 | 15.1 |

Shapes 1-2 are `HOST`-bound; their deltas are at or below the allocation-noise
floor and several are negative. No device pool is resolvable there.

**Allocation-noise floors measured in this run — carry them beside every
delta:** `2.3 / 0.6 / 1.4 / 4.9 / 13.9 / 69.9` µs for shapes 1-6. Against them,
shape 6's GEMM, egress and sync are comfortably resolvable; **reduce (79.8) is
only 1.14× its floor and is marginal**; and **release (15.0) is not resolvable
at all** — the correct statement is that release on shape 6 is now unmeasurable,
not that it is 15 µs.

**Counter cross-checks** (24 cells, no error bits): the emit-local control
collapsed off-die requests **1,836,800 → 1,792**, which validates the stage
labels by collapse rather than by assertion; fabric amplification is **1.0007×
at 99.9% full-64 B**, closing the egress-width axis; and `WRREQ_STALL` fell to
**10.0%** of `TCC_CYCLE` from 16% at `WGM=4`.

**The ranking that matters** (shapes 5 and 6 carry the whole graded gap):
GEMM mainloop **992 µs** on shape 6 and **305 µs** on shape 5 — dominant, 60.8%
and 47.5%. Then XGMI (376 / 180), then sync (169 / 53). Release has collapsed to
0.9% on shape 6 but is **10.2% on shape 5**, because grouping is switched off
wherever a CTA owns fewer than 4 tiles — see LESSONS.

## Where the kernel stands

| denominator | value | note |
|---|---|---|
| **frozen rank-1, same-run graded** | **1.098×** behind | was 1.784× when rank-1 first ran |
| reference GEMM+RCCL, same-run graded | **beaten** | 427.39 vs 438.15 µs best, earlier config |
| our harness, pipelined geomean | **~207 µs** (means) | from 285.02 at session start, **−27%** |

Per-shape graded ratio vs rank-1 (ours/rank-1, lower is better):
**0.850** / 1.072 / 1.102 / 1.120 / 1.286 / 1.208 — **shape 1 is now a win**;
shapes 5 and 6 carry the remaining gap.

> **Read the graded ratio with this caveat.** exp_24 measured the evaluator's
> harness constant at **90.38 µs median**, added identically to both arms, which
> compresses every graded ratio toward 1. On shape 2 the graded ratio is 1.0496
> while the **pipelined ratio is 1.2207** — the graded protocol *flatters* us.
> Graded stays the competition's ranking statistic, but **pipelined is the
> honest kernel-to-kernel comparison**, and part of the 1.098× graded gap is
> protocol dilution rather than kernel parity. Both are reported, never blended.

Best-of-arm pipelined vector:
`62.38 / 64.52 / 83.75 / 198.71 / 613.70 / 1616.63` µs.

## Numbering (resolved by the charter update at 29179f8e)

- `overnight/experiments/exp_NN_*` — the optimization sessions (exp_01…exp_14).
- `overnight/aug11/exp_NN_*` — **tonight's paper-figure queue, exp_20…exp_25.**

The collision I flagged (both trees had used exp_13/exp_14) is resolved: the
figure queue was renumbered to exp_20+.

## Corrections to the charter's Phase 2 premises

Phase 2 lists the next optimization targets. **Two of its premises were
overtaken by tonight's work and should not be re-run as written:**

1. **"E1(a) AGPR accumulators should also clear the Gate-M2 spills on the
   256/256/32 rows."** Both halves are now false. AGPR accumulators are
   *impossible* here: `__launch_bounds__(512,1)` caps the wave at 256 **unified**
   registers, so enabling AGPRs re-partitions the same file rather than adding
   registers — and the ISA showed the spills were never in the k-loop anyway.
   Separately, **M2 now reports 7/7 instantiations with zero AGPRs, zero
   scratch and ZERO VGPR spills**, so there are no spills left to clear.
2. **"The per-call host tax is most of the remaining graded-protocol gap."**
   Measured and closed: of ~103 µs of non-device cost, **~92 µs is harness
   machinery every submission pays** (the evaluator's own `synchronize +
   barrier` is 57-79 µs with an *empty* timed region). Our own share is ~11 µs,
   within ~6 µs of the floor for issuing any HIP kernel from Python.

What is genuinely open for Phase 2: the mainloop's **non-MFMA** time. Shape 6
spends ~4700 cycles per k-iteration against ~2050 of MFMA, and rows 4/5/6 are
pinned at the 64 KB LDS cap so their `waves × k_iters` (16 / 112 / 464) cannot
be reduced by retiling. That is ~640 µs of non-MFMA mainloop on shape 6 —
on exactly the two shapes that carry the whole remaining graded gap.

## The aug11 charter's stated baseline is two experiments stale

It reads "post-E3 best, pipelined geomean ~225.6 µs; gap to rank-1 1.167×".
Since then two more landed:

| exp | change | effect |
|---|---|---|
| `experiments/exp_13_cta_split` | E4 re-swept: shape 1 `NR=56`, shape 6 `NR=48` | paired −3.6%; gap → 1.137× |
| `experiments/exp_14_tile_waves` | tile table by wave count: rows 1/2/3 → `32/64/128`, `64/128/64`, `128/192/32` | paired −5.9%; gap → **1.098×** |

**All figure work must use the current config**, not the charter's numbers.

## Figure queue

| # | experiment | paper figure | status |
|---|---|---|---|
| exp_20 | bottleneck attribution refresh | Q3 | **DONE** — freshness gate passed (+0.96%) |
| exp_21 | saturation vs CTA count (NanoFlow Fig 7 analog) | Fig 2 / Q4 | **LANDED** — 260 points, 224/224 checksums; **egress knee C=16 of 304 = 5.3% of the machine**; falsifier not triggered |
| exp_22 | per-layer resource timeline (NanoFlow v2 Fig 10 analog) | Fig 3 / Q4 | **instrumented + parity gate PASSED** (flag-off byte-identical, 7/7); arms (a)+(b) on GPU |
| exp_23 | knob waterfall — **the money figure** | Fig 4 / Q1 | **LANDED** — all 6 shapes, 4 draws, 192/192 arms correct at both tolerances; a→c **1.113×** (1.082× order × 1.028× granularity); structural prediction held |
| exp_24 | external ladders refresh | Q6 | queued |
| exp_25 | per-shape sensitivity readout | Q5 | **LANDED — a negative, and a sharp one.** Q5's premise is falsified in sign AND not identifiable on this shape family |

Figure specs distilled from the sibling MI350X branch (read-only, via
`git show`) are in `aug11/FIGURE_SPECS.md` — including the ten MI350X-only
facts (76.8/537.6 GB/s ceilings, 256-CU grids, gfx950 occupancy claims,
`s_memrealtime` tick rate) that must be **re-measured on gfx942** rather than
quoted. `docs/distributed/PAPER.md` was found and its Q1-Q6 definitions are
captured there.

## FOR THE PAPER — five things GEMM-RS contradicts, from exp_25

These should reach the paper rather than be smoothed over. All five are measured.

1. **Q5's premise is falsified in sign**: order and granularity deltas are largest
   where communication share is **lowest**, under all four definitions of share.
2. **Q5 is also not identifiable on the graded shape family.** Shapes 3-6 order
   identically by `K` and by `tiles/NG` (ρ = +1.00) while comm share is monotone
   in `K` (ρ = −1.00), so the structural mask that decides whether a knob can act
   is **perfectly confounded** with the variable Q5 wants to regress against. More
   draws cannot fix a confound. The repaired claim is a **task-graph** claim:
   among the two shapes where order is active, the delta tracks producer rounds
   (`tiles/NG` 1.88 → 4.00) while comm share falls.
3. **The two operators do not instance one trend.** The MoE side registers "small
   M shifts value toward signal coarsening"; on GEMM-RS coarsening is
   *arithmetically impossible* except on the **largest** shape.
4. **The knobs cannot reach the most communication-bound shape.** Shape 4 is
   42.4% egress and no rung touches it; shapes 3 and 4 cannot group releases at
   any setting, because at 1 tile/CTA there is nothing to group.
5. **Q3 is supported and Q1 should be reframed.** At the winner the not-GEMM share
   is only **0.392** (shape 6) and **0.525** (shape 5), so the frontier really has
   moved back into single-GPU GEMM quality — and the waterfall is best presented
   as **history, not a map**, especially since its rungs compose superadditively.

## THE STATISTICS RULE CHANGED TONIGHT — apply it to every delta

exp_23 found that **one null twin is not enough on this node, and cross-order
consistency does not rescue you.** Two *identically configured* arms on shape 6
separated by **+3.74 / +4.44 / +4.10 / +2.53%** — positive in all four draws and
in **both** construction orders, which the disjointness rule certifies as
"RESOLVED faster". It is a false positive between two binaries that compute the
same thing the same way, and its size reproduces the published 4.28% shape-6
floor.

**Required from now on:** score every delta against the **union of all
identically-configured pairs** in the same run, not a single null twin. Keep
order reversal — it is necessary but **not sufficient**.

**Floors measured tonight (widened null set): 2.82 / 2.09 / 1.74 / 0.85 / 4.97 /
4.44 %.** Shape 5's is more than double its published 2.17% and shape 4's is
*better* than its published 2.41% — the published table is **not uniformly
conservative**, so floors must be measured in the run that produces the ratios.

## Node state — read before believing any timing taken after 05:15

A stale KFD entry (pid 3001610) holds ~1.25 GB/GPU and will not clear. It is a
**corpse, not a tenant**: the orphaned exp_26 M9 run took a
`VM_L2_PROTECTION_FAULT`, died, and wedged in `exit_mm`, where it has no address
space and therefore cannot dispatch a kernel or be signalled. All 8 GPUs pass a
live 4096³ bf16 matmul with 190 of 192 GiB free at idle temps and power.

- `gpu_lease.sh` now reports it as `KFD STALE pids (… ignored)` and no longer
  blocks the queue on it. **Do not signal it; do not `steal` the lease.**
- **Every campaign run after the incident must re-verify a known value before
  its numbers are trusted** — shape 5's pipelined best is ~613.7 µs, and the
  full best-of-arm vector is `62.38 / 64.52 / 83.75 / 198.71 / 613.70 / 1616.63`.
- **exp_26 is BLOCKED on that fault**, which is a correctness matter, not an
  obstacle. Separate the two hypotheses before measuring anything: the `rgroup`
  change itself, versus M9's harness (shape 5 had to be added to its `CASES` and
  its golden was already stale after the exp_14 retile). Run the **unmodified**
  kernel through the same extended M9 first — it is the cheaper hypothesis.

## Session infrastructure notes

- **`tools/push_scoped.ps1` is new and is the only push tool safe to use while
  several agents are working.** `push.ps1` scps the whole overnight tree *plus*
  all four kernel sources, so one agent running it ships every other agent's
  half-written file and can silently revert a validated kernel while leaving
  `build/*.so` intact. The scoped tool pushes a caller-declared path set.
- Two facts from tonight's harness survey that change how the queue is run:
  **`exp_ablation.py` emits no JSON at all** (stdout table only, so every
  attribution deliverable needs a wrapper), and **there is no CTA-count or grid
  override anywhere in the codebase** (grid is hardcoded `dim3(m3::CU_COUNT)`
  = 304; only the host scalar `num_gemm_ctas` varies). exp_21 therefore cannot
  bend the production kernel and is a standalone module.
- **No per-CTA timestamp instrumentation exists** (no `s_memrealtime`,
  `wall_clock64` or trace buffer anywhere in the kernel). exp_22 arm (b) is
  greenfield kernel work behind a compile flag, gated on resource-tuple parity.

Then **Phase 2**: the optimization loop resumes against the refreshed profile.

## PHASE 2 RE-RANKED by exp_21 — the mainloop may be BANDWIDTH-bound, not schedule-bound

exp_21 measured the mainloop body in isolation and found MFMA scaling is **linear
only to C≈160**, plateauing at **582.4 TFLOPS = 45% of this node's 1307.4 TFLOPS
ceiling** — and **not because of occupancy** (1 CTA/CU from both 193 VGPRs and
64 KB LDS). The mechanism closes to 1%: **7.813e-3 B/FLOP × 582.36 TFLOPS =
4549 GB/s against an independently measured 4593 GB/s memory-path plateau.**

**At the full 304-CTA grid this mainloop is memory-path bound**, which changes the
plan: arithmetic intensity is `~(1/BM + 1/BN)` and is **independent of `BK`**, so
freeing LDS to raise `BK` cuts iterations while moving **the same operand bytes**.
The lever is **operand traffic per FLOP** (bigger `BM·BN` reuse, better L2/Infinity
Cache hit rate, XCD-aware tile order — still untested), not fewer iterations.
exp_27 has been interrupted and re-tasked to reconcile this against exp_20's
"~571 µs is not MFMA occupancy" and to decide between **exposed latency** and
**bandwidth wait**, since those imply opposite designs.

**And exp_21 promoted a cheaper target**: the release protocol costs **0.440× of
egress bandwidth at identical payload bytes** (0.366× single-link). With exp_20's
1.0007× fabric amplification, egress *width* is closed from both sides and **the
whole residue is release granularity** — which is also where exp_26 already sits.

## Phase 2 queue, ranked by the exp_20 profile (not by the charter's stale one)

| # | target | pool it attacks | expected | risk |
|---|---|---|---|---|
| 1 | **GEMM mainloop non-MFMA time** | **~571 µs** of shape 6's 992 µs GEMM pool is NOT MFMA occupancy (counter pass: only 421.2 µs is) — **35% of the whole operation**; ~204 µs on shape 5 | the only pool big enough to close the rank-1 gap, and it is a schedule problem, not a math-throughput one | high — rows 4/5/6 pinned at the 64 KB LDS cap, so `waves × k_iters` (16/112/464) is unreachable by retiling; needs single-buffered `BK` or an async pipeline |
| 2 | `sync` — cross-rank waits/credits/publishes | 168.7 µs (shape 6), now 2nd-largest non-GEMM | untouched axis | medium |
| 3 | **exp_26: `RELEASE_GROUP_FULL_ONLY=0`** | 65.4 µs on shape 5 | ~32 µs ≈ 5% of shape 5, ~0.8% geomean | low — one build flag; but needs M9 re-golding and shape 5 added to M9's `CASES` |
| 4 | XGMI on shape 4 specifically | **86.2 µs = 42.4%** of shape 4 | shape-specific; the profile is not uniform | medium |

Item 3's adjudication is in LESSONS: the negative that closed it lived on shape 2
and became unreachable when exp_14 retiled row 2 from 512 tiles to 256, and
exp_05 explicitly declined to conclude anything about shape 5. That is the
**fourth** time a landed win has invalidated a settled constant — the standing
lesson is to re-check every constant whose *geometry* moved, not just the ones
whose code moved.

## What the kernel already demonstrates for the paper's central claim

The claim is that overlap is decided by **scheduling decisions**, not by
dedicating CTAs to communication. This kernel is an unusually clean instance:

- The two largest wins were a **task-order** change (`WGM`, −27.9% on shape 6,
  effective xGMI links 2.02 → 7.53 of 8) and a **signal-coarsening** change
  (`RELEASE_GROUP=4`, −4.0% graded on shape 6). Neither moved a single byte
  differently; the fabric carried 117.48 MB against 117.44 MB of useful payload
  before and after, 99.9% of it in full 64 B transactions.
- **Dedicating more CTAs to communication was measured and rejected**: the
  producer/consumer split buys *rounds*, not CTAs, so `NR=16` bought zero GEMM
  waves and cost +3.6%. A dual-role/work-stealing reducer was killed by the
  same arithmetic — it removes a producer wave on none of the six shapes.
- The reducer CTAs exist only because the reduction must run on the owner.

That is three independent data points for "scheduling, not CTA dedication",
already measured on this architecture.

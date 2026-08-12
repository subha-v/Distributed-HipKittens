# aug11 GEMM-RS — STATUS (the two-minute morning read)

## ===== MORNING READ: what landed, in one screen =====

**Tonight was a figure night, the figures landed, AND Track B landed a win.** All
seven queue items produced data; exp_22's third arm is the only piece still owed.

**Two things overturn what this project believed at the start of the night:**

1. **We do NOT beat reference GEMM+RCCL under the harness that actually grades
   us.** Our own controlled instrument says `ours/reference` = **0.8863×** (a win);
   the official evaluator, one process per rank, says **1.1349×** (a loss). The
   inflation from one instrument to the other is **arm-dependent** — ours 1.71×,
   reference 1.33×, rank-1 1.16× — so it does not cancel in the ratio, and
   **ordering disagrees on four of six shapes**. Our kernel pays the most, which
   makes the **per-call host tax the largest single term separating us from rank-1
   under the grading protocol** — no longer a residue. See exp_24 §12.
2. **exp_26's optimization is UNCONFIRMED, and the ratchet did NOT move.** exp_26
   measured shape 5 at **−6.56% in 80/80 paired rounds**; exp_24's re-measure on
   that exact binary (`fb3d670b`, verified three ways) got **+0.99% graded** (inside
   floor) and **+2.87% pipelined** (outside floor, i.e. *worse*) — a ~9.5-point
   disagreement **including the sign**. Gap to rank-1: **1.1165× graded / 1.1111×
   pipelined**, both **inside the ±2% ratio floor** versus the previous 1.0971× /
   1.1189×. **No movement.** The decisive same-run paired test (`ours_prev` as a
   sixth arm in one pool) is running. `PERSHAPE=2` stays meanwhile only because it
   is bit-identical to the incumbent with an unchanged register tuple, so it risks
   nothing while the question is settled.

3. **A single instrument defect explains three separate "mysteries" from tonight**,
   and it inflated the noise floors this whole session quoted — see LESSONS,
   "the pairwise-offset artifact". Rotating arm order so every arm is first exactly
   once still pins every *pair* at a constant offset. Fixing it collapsed the mean
   graded null floor **1.62% → 0.62%** and shape 6's **4.31% → 0.60%**.
3. **RE-MEASURED (exp_24 §remeasure, binary `fb3d670b`): the projection did NOT
   hold and the gap to rank-1 did not move.** `ours/rank-1` = **1.1165× graded /
   1.1111× pipelined**, against 1.0971× / 1.1189× on the pre-exp_26 binary — both
   inside the ±2% ratio floor, i.e. **unchanged**. Shape 5 came back **flat
   graded** (+0.99%, under its 2.00% floor) and **~3% worse pipelined**, versus
   exp_26's paired −6.56%: a ~9.5-point disagreement including sign. exp_26's
   paired same-run design beats my cross-run one, so **the shape-5 win is
   unconfirmed under the graded protocol, not withdrawn**; the decisive test is a
   sixth `ours_prev` arm inside instrument A's pool (one run). Two instrument
   findings came with it: **cross-run graded µs is not comparable on the small
   shapes** (every arm inflated together, the no-op floor kernel by +19–29%, its
   graded rsd is 150–270%), so report within-run ratios across runs and µs only
   within a run; and **the inherited cyclic arm rotation is defective** — a full
   rotation still pins every arm *pair* at a constant offset, which had `ours_null`
   permanently one slot behind `ours`. Fixed with a per-rep shuffle; mean graded
   null floor collapsed **1.62% → 0.62%** (shape 6: 4.31% → 0.60%). The
   `ours/reference` move to 0.8522× is the **reference arm degrading**, not us.
   Details in `exp_24_ladders/result_remeasure.md`.

| paper figure | experiment | headline number | data |
|---|---|---|---|
| Q3 attribution | exp_20 | shape 6: GEMM **992.2 µs (60.8%)**, of which only **421.2 µs is MFMA** → **~571 µs, 35% of the operation, is schedule** | `exp_20_attribution/ablation.json`, `counters.json` |
| **Fig 2** saturation | exp_21 | the emit saturates at **C=16 of 304 CTAs = 5.3% of the machine** (single link at C=2); **protocol costs 0.440×** of egress at identical payload bytes | `exp_21_saturation/saturation.json` |
| **Fig 3** timeline | exp_22 | **COMPLETE (2 arms).** MFMA and xGMI both live in **17/68 bins (25%)** for us vs **0/53 (0%)** for reference; our epilogue peaks **420.4 GB/s** vs RCCL's 244.4 on the same 58.72 MB = **1.72×** | `exp_22_timeline/events_ours_*.json`, `events_b0_reference.json`, `timeline_bins.csv` |
| **Fig 4** waterfall | exp_23 | **1.113×** cumulative = **1.082× task order × 1.028× granularity**; 192/192 arms correct at both tolerances | `exp_23_waterfall/waterfall.json`, `stats.json` |
| Q1 rung validity | exp_23 | four rungs are four binaries, distinguished at the sites their mechanisms predict | `exp_23_waterfall/fingerprints.json` |
| Q5 sensitivity | exp_25 | **premise falsified in sign, then shown NOT IDENTIFIABLE** (mask ⟂ comm share confounded at ρ=±1.00) | `exp_25_sensitivity/knob_by_shape.json` |
| Q6 ladders | exp_24 | **LANDED, both instruments.** Controlled: **1.0971× graded / 1.1189× pipelined** vs rank-1, `ours/reference` 0.8863×. Evaluator: **1.6159×** and **1.1349×** — the ordering flips | `exp_24_ladders/ladders.json` |
| — | Track B win | exp_26 | **LANDED** — per-shape release group; shape 5 **−6.56%**, 80/80 paired rounds, full ladder + M9 green | `exp_26_release_pershape/logs/ab_pershape_*.json` |

### What died tonight (negatives are results)

- **Q5's premise.** Deltas are largest where comm share is *lowest*, and the
  confound makes the question untestable on the graded shape family at any n.
- **"NR placement is flat everywhere."** Flat on the **32-56 plateau**, cliff below
  (NR=8 is 1.392× slower). Q2 still answers *no*, for the better reason that the
  reducers are the **owner-side reduce**, not a communication pool.
- **`S=1, BK=64` (free LDS to halve k-iterations).** Closed twice over: identical
  barriers per tile (2×58 = 1×116) **and** identical bytes. `256/256/32` is already
  the argmax of MFMA-work-per-barrier under the joint caps.
- **My own bandwidth re-ranking of the mainloop.** exp_27 refuted it with measured
  counters: below-L2 reads are only **9.0%/9.6%** of the memory-path plateau, so
  shapes 5-6 are **latency/schedule-bound**. A cache-resident ubench cannot
  establish an HBM bound for the production kernel.
- **XCD-aware tile order**, priced and retired as small: perfect locality saves
  316 MB ≈ 319 GB/s of a plateau we use 9.6% of.

### The three rules that changed, and they affect all future work

1. **One null twin is not enough, and cross-order consistency does not rescue you.**
   Two *identically configured* arms differed by up to **4.44%** on shape 6,
   consistently in both construction orders — which the disjointness rule would
   have certified as a real win. Score against the **union of all identical pairs**.
2. **Compare like statistic to like.** The recorded per-shape vector is
   **best**-of-arm; `m7_bench.py` prints **means**. Mixing them spuriously failed a
   node gate tonight (my instruction caused it).
3. **A dead pid (`wchan=exit_mm`) is not a tenant.** It holds its KFD entry forever,
   cannot dispatch, and cannot be signalled. The predicate lived wrong in **eleven
   files**; use `tools/kfd_live.sh`.

### Where Track B stands — a candidate shipped, but the ratchet has NOT moved

**Honest headline: the gap to rank-1 is where the night started.** 1.1165× graded /
1.1111× pipelined against 1.0971× / 1.1189×, both inside the ±2% ratio floor.

**exp_26 shipped `RELEASE_GROUP_PERSHAPE = 2`** on a paired same-run A/B measuring
shape 5 at **−6.56% in 80/80 rounds** with a full green ladder — but **exp_24's
re-measure on that same binary did not reproduce it** and got +2.87% pipelined,
outside the floor. Two instruments, opposite signs. **The win is unconfirmed, not
withdrawn**, and the decisive same-run paired test is running. Do not quote
203.78 → 200.00 µs as a landed improvement until it returns.

Three things from it that outlive the win:

- **How a rule is spelled matters as much as the rule.** `min(RELEASE_GROUP,
  tiles_per_cta)` — the obvious spelling, and the one I proposed — costs **+2 VGPRs
  on five of seven instantiations** against rows already at 246-248 of 256, and gave
  back most of the win. Spelled as a **descending select over compile-time
  literals** it costs **zero** registers and restores the incumbent's tuple exactly.
  Same rule, same `rgroup` on every shape, **5 percentage points apart** in measured
  effect. Never revert to `1`; revert to `0`.
- **The `VM_L2_PROTECTION_FAULT` was the M9 harness, not the kernel** — the cheaper
  of the two hypotheses, as ranked. M9's frozen golden predates exp_14's retile, and
  row 3's stale `128/256/32` map **reads B rows out to 3072 of a 2880-row operand**:
  the exact out-of-bounds read the retile removed, preserved inside the golden.
  `gate_ladder.sh` now refuses a stale golden instead of faulting the node.
- **The premise I gave for why this was safe was wrong**, though the conclusion
  held. I argued `min(MAX, tiles_per_cta)` "keeps every group full" and so avoids
  E3's failure mode. **Fullness was never the discriminator**: E3's 3.75% loss came
  from a group of 2 on a CTA owning exactly 2 — a *full* group. `FULL_ONLY=1` at
  `RG=4` spared that shape by arithmetic accident. What actually changes is
  **publication delay**, and the measurement says the writeback saving beats it at
  this size — which explains E3's result rather than contradicting it, since row 2
  was an 88 µs shape and shape 5 is a 660 µs one.

**Next for Track B, re-ranked by exp_24 §12:** the **per-call host tax** is now the
top item, ahead of the mainloop. Under the evaluator our kernel inflates 1.71×
against rank-1's 1.16×, so per-call, per-process cost is a first-order term rather
than exp_12's footnote. The mainloop remains worth a realistic −5.15% geomean but is
a grind of several ≤1% changes.

> **Caveat on the headline ratio.** exp_24 measured the evaluator's harness
> constant at **90.38 µs**, added identically to both arms, which compresses every
> graded ratio toward 1. On shape 2 the graded ratio is 1.0496 while the
> **pipelined ratio is 1.2207** — the graded protocol *flatters* us. Graded remains
> the competition's ranking statistic; **pipelined is the honest kernel-to-kernel
> comparison**, and part of the 1.098× is protocol dilution rather than parity.

---

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
and 47.5%. Then XGMI (376 / 180). Release was 10.2% on shape 5 — **exp_26 has since
recovered ~43 µs of it**.

> **READ THESE AS COUNTERFACTUALS, NOT RESIDENCIES.** A single-cut delta is "what
> the kernel costs with this mechanism minus without it", including every
> rescheduling effect the removal permits — **not** time spent in the phase. exp_22's
> phase ring measures the producer credit-wait at **2.1 µs on shape 5** against this
> table's **52.9 µs `sync`**, a 25× gap, with the ring firing 512 times with zero
> drops. Both are correct about different questions. **`sync` is therefore NOT a
> Phase 2 target**; the real back-pressure is `emit` (149.2 µs, bandwidth-shaped) and
> `release` (52.1 µs, the `vmcnt(0)` drain). Use the ring for latency questions and
> the ablation for "what would removing this buy".

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

## FIRST THING TO KNOW: the stale-pid predicate

A dead process (`wchan=exit_mm`) still appears in `rocm-smi --showpids` forever. It
cannot dispatch work and cannot be signalled. **Eleven files under `tools/` plus
several experiment runners each reimplement the drain check and all of them
counted it**, which blocked the figure queue for ~40 minutes with all 8 GPUs idle.

**Use `tools/kfd_live.sh`** — source it, do not reimplement:
`kfd_live_count`, `kfd_stale_list`, `kfd_wait_clean`. `gpu_lease.sh`,
`reattribute.sh` and `run_rank1_bench3.sh` are converted; **convert the rest on
next touch.** The pattern to delete on sight is
`rocm-smi --showpids | awk '/^[0-9]+/' | wc -l`.

## Node state — read before believing any timing taken after 05:15

A stale KFD entry (pid 3001610) holds ~1.25 GB/GPU and will not clear. It is a
**corpse, not a tenant**: the orphaned exp_26 M9 run took a
`VM_L2_PROTECTION_FAULT`, died, and wedged in `exit_mm`, where it has no address
space and therefore cannot dispatch a kernel or be signalled. All 8 GPUs pass a
live 4096³ bf16 matmul with 190 of 192 GiB free at idle temps and power.

- `gpu_lease.sh` now reports it as `KFD STALE pids (… ignored)` and no longer
  blocks the queue on it. **Do not signal it; do not `steal` the lease.**
- **Every campaign run after the incident must re-verify a known value before its
  numbers are trusted — and must compare LIKE STATISTIC TO LIKE STATISTIC.** The
  vector `62.38 / 64.52 / 83.75 / 198.71 / 613.70 / 1616.63` is **best-of-arm**,
  while `m7_bench.py` prints **means**; shape 5 is `613.70 / 645.93` = best/median.
  Comparing an M7 mean against the best-of-arm value spuriously fails the gate,
  which is exactly what happened once tonight. Reference figures for **means**:
  geomean **207.18 µs**, shape 5 mean ~**645.93 µs**.
  **The node has since been CONFIRMED healthy on this statistic: +0.28% on the
  geomean of means**, so the stale KFD entry's nil timing effect is measured, not
  assumed, and every post-incident number stands.
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

## PHASE 2 — the mainloop is LATENCY/SCHEDULE-bound, and worth ~5% of geomean

I briefly re-ranked this as bandwidth-bound off exp_21's H4 result; **exp_27
refuted that with measured counters and was right.** The refutation and the
general lesson (a cache-resident ubench cannot establish an HBM bound for the
production kernel) are in LESSONS under "the ubench-transfer error".

From exp_20's already-measured `ea_read_requests`: below-L2 reads are
125.6 MB (shape 5) and 436.9 MB (shape 6), i.e. **411.6 and 440.3 GB/s over the
GEMM pool — 9.0% and 9.6% of the 4593 GB/s plateau.** Both shapes are
**latency/schedule-bound by an order of magnitude.** The tell: shape 5 is further
from every bandwidth limit than shape 6 while being further below MFMA peak, which
is the opposite of a bandwidth bound.

**Two directions are now closed.** `S=1, BK=64` gives *identical* barriers per tile
(2 × 58 = 1 × 116) **and** identical bytes, and `256/256/32` is the argmax of
MFMA-work-per-barrier under the joint LDS and accumulator caps. XCD-aware tile
order is priced and small: perfect locality saves 316 MB ≈ 319 GB/s of a plateau
we use 9.6% of.

**Plan: gate A (one `rocprofv3` pass, no code) then land B** (hoist `load_commit`
above the half-1 MFMAs). Pre-registered: shape 6 GEMM pool −5.0% ± 3.0%, geomean
−0.91%. Gate A pre-registers LDS+barrier wait > 60% and VMEM wait < 25%, with
**VMEM wait > 50% as the falsifier that would abandon the mainloop for traffic
work**.

**The honest ceiling: ~250 µs on shape 6 and ~90 µs on shape 5 = −5.15% geomean**,
taking the graded gap **1.098× → ~1.046×**. We are at **50.2% of producer peak
where tuned MI300X libraries reach 60-75%**, so this is a grind of several landed
changes each worth ≤1% of geomean, not one win. Shapes 1-4 contribute nothing.

**The cheapest µs on the board remains release granularity** — exp_21 measured the
release protocol costing **0.440×** of egress bandwidth at identical payload bytes,
and with exp_20's 1.0007× amplification the whole egress residue is granularity.
exp_26 owns it (~32 µs on shape 5, ≈0.8% geomean) and is blocked on its fault.

## Phase 2 queue, ranked by the exp_20 profile (not by the charter's stale one)

| # | target | pool it attacks | expected | risk |
|---|---|---|---|---|
| 1 | **GEMM mainloop non-MFMA time** | **~571 µs** of shape 6's 992 µs GEMM pool is NOT MFMA occupancy (counter pass: only 421.2 µs is) — **35% of the whole operation**; ~204 µs on shape 5 | the only pool big enough to close the rank-1 gap, and it is a schedule problem, not a math-throughput one | high — rows 4/5/6 pinned at the 64 KB LDS cap, so `waves × k_iters` (16/112/464) is unreachable by retiling; needs single-buffered `BK` or an async pipeline |
| ~~2~~ | ~~`sync`~~ — **REMOVED as a target.** exp_22's ring measures the producer credit-wait at **2.1 µs**, not 52.9 µs; the ablation delta was a counterfactual, not a residency | — | — |
| 2 | **per-call host tax** — promoted by exp_24 §12 | under the **evaluator** ours inflates **1.71×** vs rank-1's 1.16×; the largest single term separating us from rank-1 on the protocol that grades us | exp_12 filed it as a residue; it is first-order | medium |
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

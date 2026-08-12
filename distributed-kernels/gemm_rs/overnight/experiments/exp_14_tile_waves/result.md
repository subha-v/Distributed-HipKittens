# exp_14 / E4b — result: three rows retiled. **Geomean −5.9% paired, graded gap 1.137× → 1.098×.**

## Verdict

| # | shape | tile before | **tile now** | paired Δ | landed |
|---:|---|---|---|---:|---|
| 1 | 64×7168×18432 | 32/64/64 | **32/64/128** | **−1.7%** | yes, on the rank test only |
| 2 | 512×4096×12288 | 64/64/64 | **64/128/64** | **−24.7%** | yes, disjoint |
| 3 | 2048×2880×2880 | 128/256/32 | **128/192/32** | **−6.3%** | yes, disjoint |
| 4 | 4096×4096×4096 | 256/256/32 | 256/256/32 | — | unchanged, argmin of 38 legal points |
| 5 | 8192×4096×14336 | 256/256/32 | 256/256/32 | — | unchanged, argmin of 38 |
| 6 | 8192×8192×29568 | 256/256/32 | 256/256/32 | — | unchanged, **at the wave-cost floor** |

Full gate ladder passed: M2 7/7 instantiations with **0 AGPRs, 0 scratch, 0 VGPR
spills**; M3 **17/17 at both `1e-2` and `2e-3`**; M4 all three controls fail as
designed; M5 600-epoch skewed soak exact; M7 geomean (means) **222.17 → 207.18
µs**. Graded same-run against frozen rank-1: **1.137× → 1.098×**.

**Shape 2's −24.7% is the largest single win of the project**, and it came from
the model column the brief's framing does not contain.

---

## 1. The null-arm floor, first, because it is bigger than one of the effects

Two independently allocated `GemmRS` instances carrying the **identical** landed
tile, measured 15 times each in 15 separate processes:

| # | worst single draw (best / median) | **pooled T+T\* range** | published floor | contrast set (T vs T\*) |
|---:|---|---:|---:|---|
| 1 | 2.66% / 1.98% | **3.48%** | 1.34% | [−2.59, +2.42] %, median −0.56 |
| 2 | 0.76% / 0.85% | **0.93%** | 0.56% | [−0.75, +0.52] %, median −0.05 |
| 3 | 1.22% / 1.16% | **1.61%** | 0.61% | [−0.66, +1.22] %, median −0.23 |

Shapes 1 and 3 measured **2–2.6× worse than their published floors**. Shape 1's
floor of 3.48% is larger than shape 1's entire effect, which is the whole story
of that row and the reason it needed 15 draws and a rank test rather than 5 and a
range.

### The floor has a systematic component, and it is allocation ORDER

Ten forward draws produced a T-vs-T\* contrast that was **negative in 10 of 10**
on shape 1 (median −0.91%). That is not what symmetric per-allocation noise looks
like. T is constructed first and T\* last, so the hypothesis was that allocation
*order* carries a systematic term on top of the random per-allocation offset.
Five draws with the construction order reversed confirmed it: the contrast set
went from [−2.59, −0.25] (all negative) to **[−2.59, +2.42], median −0.56** —
it straddles zero once both orders are present.

**Consequence for anyone reusing this instrument:** every candidate is
constructed *between* T and T\*, so the T-vs-T\* contrast is the **widest** pair
in the run and using it as the floor is conservative for the candidates — but a
sweep that only ever runs one construction order is measuring a biased null and
will systematically under-credit every arm built after the reference. Half the
draws should reverse. `sweep.py` takes `SWEEP_REVERSE=1`.

## 2. The model, and the one correction that made the sweep productive

E4's insight — *a strided persistent loop costs `ceil(tiles/count)` rounds, not
CTAs* — carries to the tile dimension, but with a correction: **moving the tile
also changes what one tile body costs, so `waves` alone is not a time.** Three
cost columns have to be minimized together, and they disagree.

| column | formula | regime |
|---|---|---|
| `wave_cost` | `waves × BM·BN` | MFMA-bound. Equals `(M·N/NG)/avg_fill`, so this **is** the brief's fill criterion, re-derived as a time. Floor `M·N/NG`. |
| `iter_cost` | `waves × k_iters` | overhead-bound: every k-iteration pays a `__syncthreads()`, a `vmcnt(0)`, an `lgkmcnt(0)` and an LDS round trip *whatever the tile size*. |
| `traffic` | `1/BM + 1/BN` | bandwidth-bound: global bytes are `M·N·K_local·2·(1/BM + 1/BN)` before L2 reuse. |

`iter_cost` was estimated from shape 6 **before any arm ran**: 1142.8 µs of GEMM
over `4 × 115.5 = 462` iterations is ~4700 cycles/iteration at 1.9 GHz against
`BM·BN·BK/1024 = 2048` cycles of MFMA, so **~57% of a k-iteration is not MFMA**.
That is why raising `BK` — the only knob that cuts iteration count at constant
MFMA work and constant total bytes — was pre-registered as the main lever, and it
is where both large wins came from.

A model counting only fill would have produced an empty sweep: on `wave_cost`
alone four of six rows are already within 6% of their floor.

### Second correction: "last-wave fill" is not waste on a 1-wave row

With `waves = 1` every tile is resident at once, so the critical path is **one**
tile body and the idle CTAs cost nothing directly. A low fill there means `BM·BN`
is bigger than it needs to be, and *that* is the cost. The two readings coincide
numerically but only the second explains why shape 3 at 71% fill was worth
attacking while `128/320/32` — also 1 wave, 53% fill — would have been **worse**,
not better.

## 3. Constraints, verified against source (two were missing from the brief)

| constraint | source | in brief? |
|---|---|---|
| `BM % 32 == 0`, `BN % 64 == 0` | `gemm_rs_mi300x.cpp:204` | yes |
| `LDS = 2(BM+BN)·BK·2 ≤ 65536` | `:208-209` `static_assert` | yes |
| `32·BN·2 ≤ LDS` | `:215-216` | yes |
| `BM \| M`, `lrow ≤ 32`, `col ≤ 128`, `K_TAIL` | `host_abi.hpp:169,174,195` | yes |
| **`BK % 32 == 0`** | `:255-257` — `KH=2`, `KS=BK/2`, `static_assert(KS % 16 == 0)` | **no** |
| **`BM·BK ≤ 65536`, `BN·BK ≤ 65536`** | `hk_adapter.cuh:114-121` `static_assert(big_calls == 1)` | **no** (implied by the LDS cap anyway) |

Two more that are not `static_assert`s but bound the space just as hard:

- **Accumulator VGPRs = `BM·BN/512`.** `rt_fl<BM/2, BN/4>`; the 256×256 rows
  measure 246–248 of 256. So `BM·BN ≤ 65536`, and *this* is what makes one wave
  impossible on shapes 5 and 6.
- **Occupancy is irrelevant.** The grid is exactly `CU_COUNT = 304` CTAs on 304
  CUs, so LDS never buys a second resident CTA; it is purely a cap. This retires
  exp_04's note that shape 2 "is the only row at 2 CTAs/CU — leave it": there was
  nothing to lose, and shape 2 turned out to be the biggest win on the board.

## 4. Candidates: predicted before measuring, then measured

Predictions are from `plan.md`, written and pushed before the first arm ran.
Geometry from `enumerate.py` (full output in `enumeration.txt`). Measured columns
are 15 independent allocation draws, 8 passes each, complete positional rotation,
10 forward + 5 reverse-constructed.

### Shape 1 — floor 3.48%

| arm | geometry (predicted) | `w·x` | `w·ki` | traffic | **predicted** | **measured** best / median | per-draw range | verdict |
|---|---|---:|---:|---:|---|---:|---|---|
| **A1 `32/64/128`** | 224 tiles, 1 wave, 90% fill, `ki` 36→18, LDS 48K, even_k, cols 112 unchanged | 1.00 | 0.50 | 1.00 | **WIN, −10% to −25%** | **−1.70% / −1.49%** | [−0.41, +3.00] | **LAND** on `p_rank = 0.0013`; range overlaps |
| A2 `32/64/160` | 224 tiles, 1 wave, 90%, `ki` 15, LDS 60K, **K_TAIL** | 1.00 | 0.42 | 1.00 | win, slightly better than A1 on the model, ranked below on risk | −1.41% / −1.60% | [−0.64, +2.88] | rejected in favour of A1 |

**The model was badly wrong here — off by an order of magnitude.** It predicted
−10 to −25% from halving 36 iterations to 18; the measured effect is −1.7%. The
inference is direct and it is the most useful number in this report: **shape 1's
entire mainloop is only ~2–4 µs of its ~63 µs.** Halving its iteration count buys
~1 µs, so ~59 µs of shape 1 is protocol, launch, egress and reduce. Its `×SOL` of
10.16 and `bound = HOST` in M7 said so; the model ignored it.

A2 confirms the same arithmetic from the other side: 18→15 iterations should buy
another ~0.2 µs, and it measured indistinguishable from A1 (both ~−1.5%). **Two
points on the `BK` axis agreeing that the axis is nearly flat for this row is
stronger evidence than either alone.** A1 wins the tie on risk, not on time:
even_k so no tail-mask path, power-of-2 `BK`, 48K rather than 60K of LDS, and
**54 SGPR spills against A2's 143** — the highest of the 11 instantiations built.

### Shape 2 — floor 0.93%

| arm | geometry (predicted) | `w·x` | `w·ki` | traffic | **predicted** | **measured** best / median | per-draw range | verdict |
|---|---|---:|---:|---:|---|---:|---|---|
| **B1 `64/128/64`** | 512→**256 tiles**, 2→**1 wave**, 94% fill, `ki` 24, LDS 48K, red_tiles 64→32 so reduce rounds 2→1 | 1.00 | 0.50 | 0.75 | **WIN, predicted winner of the pair** | **−24.7% / −24.6%** | [+31.61, +34.10] gain | **LAND**, disjoint, `p ≈ 1e-5` |
| B2 `64/64/128` | 512 tiles, 2 waves, `ki` 12, LDS 64K at the cap | 1.00 | 0.50 | 1.00 | win vs T, **loss vs B1**; the discriminator | **DEAD — wrong at `1e-2`** | — | disqualified, see §6 |
| B3 `64/192/64` (added after draw 1) | 176 tiles, 1 wave, 65%, `ki` 24, LDS 64K, 3.1% N padding | 1.50 | 0.50 | **0.67** | untested at design time | −20.1% / −19.6% | [+24.09, +26.61] gain | real win, **7.6 pp worse than B1** |

**Direction right, magnitude under-predicted by ~8×.** `plan.md` said "WIN, and
the predicted winner of the pair" without a number; the informal expectation was
a few percent. −24.7% (85.32 → 64.12 µs best) is the largest single improvement
of the project. The `iter_cost` model explains it exactly: shape 2 was **48
iterations deep** (2 waves × 24), the deepest of the six rows relative to its
size, and B1 makes it 24 while holding MFMA work constant. 21.2 µs saved over 24
removed iterations is ~0.88 µs per removed iteration — and shape 2's `×SOL` fell
from 10.88 to 8.17.

B3 is the useful negative: same `w·ki`, **lower** traffic (0.67 vs 0.75), but
`w·x` 1.50 — and it lost 7.6 points to B1. So on this row the MFMA/quantization
term outranks the traffic term, which is the opposite ordering to shape 3 below.
**Neither column dominates globally; both have to be carried.**

### Shape 3 — floor 1.61%

| arm | geometry (predicted) | `w·x` | `w·ki` | traffic | **predicted** | **measured** best / median | per-draw range | verdict |
|---|---|---:|---:|---:|---|---:|---|---|
| **C1 `128/192/32`** | 192→**240 tiles**, 1 wave, 71%→**88% fill**, `ki` 12, LDS 40K, `2880/192 = 15` **exact** | 0.75 | 1.00 | 1.11 | **weak win, −0% to −3%, LOW confidence** | **−6.3% / −6.4%** | [+5.03, +8.95] gain | **LAND**, disjoint, `p ≈ 1e-5` |
| C2 `64/192/64` | 480 tiles, 2 waves, 76%, `ki` 6, LDS 64K | 0.75 | 1.00 | **1.78** | **predicted LOSS** | **+6.2% slower** | [−6.60, −4.92] | **rejected as predicted**, disjoint SLOWER |

**Direction right, magnitude under-predicted by ~2–3×**, and the pre-registered
reasoning for the low confidence was wrong in an identifiable way. `plan.md`
argued the MFMA term is only 1024 cycles against ~2650 of overhead so a 25% MFMA
cut is only a ~7% iteration cut, then discounted it by shape 3's ~26% GEMM
fraction to get ~−1.5%. Two things that reasoning missed: the per-iteration LDS
and global traffic *also* fall with `BM+BN` (320 vs 384, −17%), so the "overhead"
was not constant; and the 6.7% of N-padding removed is work in every stage, not
only the MFMA.

C2 is the designed discriminator and it earned its GPU time: **identical `w·x`
AND identical `w·ki` to C1, differing only in traffic (1.78 vs 1.11) and tile
count (480 vs 240) — and it was 6.2% slower in 15 of 15 draws, disjoint.** The
traffic column is load-bearing, and "shrink the tile to buy fill" is not free.

C1 also removes a latent **out-of-bounds global read**. At `BN=256`,
`col_count = 12` and the last B tile's `load_issue` reads rows 2880..3071 of a
2880-row `w` — 192 rows = 138 KB past the end of the tensor. It is benign today
(those columns are never emitted, `vt` clamps them) and it never faulted, but it
is a real OOB read that has been in every shape-3 call all session. `2880/192 =
15` exactly, so `even_n` becomes true and the read disappears.

### Shapes 4, 5, 6 — no arm, and that is a result

Each landed tile is the **argmin of the composite over all 38 legal points** for
its shape, and shape 6 sits **exactly at the `wave_cost` floor** (1024 tiles =
4 × 256 NG, 100% fill). Two specific claims worth recording:

- **The brief's premise about shape 5 is wrong.** Its 88% last-wave fill is not
  recoverable: one wave needs `BM·BN ≥ 8192·4096/272 = 123362`, i.e. **241
  accumulator VGPRs** against a 256 budget already 246 full. Every legal way to
  raise the fill either exceeds that or costs more waves than it saves; the next
  best point is +30%.
- **Rows 4/5/6 are pinned against the LDS cap.** `(BM+BN)·BK = 512·32 = 16384`
  is exactly the limit, so `BK` cannot rise on the three largest rows, and their
  `iter_cost` — `waves × k_iters` of 16 / 112 / 464, the larger half of the
  mainloop — is **untouchable from the tile table**. Shape 6 spends ~2650
  non-MFMA cycles × 462 iterations ≈ 640 µs there. Unlocking it means the double
  buffer must stop costing 2× (single-buffered `BK` with an async pipeline, or
  splitting the k-step so only half of `BK` is resident). That is E1 work and it
  is now the largest identified pool on the two shapes that carry the graded gap.

## 5. Where the model earned and lost credibility

| claim | outcome |
|---|---|
| `iter_cost` (`waves × k_iters`) is the dominant column, not fill | **Earned.** Both large wins are pure `iter_cost` reductions at constant `w·x`, and the largest (shape 2, 48→24 iterations deep) is the row that was deepest. |
| the traffic column is real and must be carried | **Earned, by a designed control.** C2 vs C1: identical on both time columns, 1.78× traffic, 6.2% slower, 15/15 disjoint. |
| ranking candidates by `sqrt(rel_wave × rel_iter)` picks the winner | **Earned on ordering, not on magnitude.** The composite argmin was the measured argmin on all three swept rows, and correctly said "no candidate" on the other three. |
| the *size* of an `iter_cost` reduction | **Lost, badly and in both directions.** Shape 1 predicted −10..−25%, measured −1.7% (12× over). Shape 3 predicted −0..−3%, measured −6.3% (2–3× under). |
| which of `w·x` and traffic outranks the other | **Lost.** Shape 2 says `w·x` (B3 lost with better traffic); shape 3 says traffic (C2 lost with equal `w·x`). There is no global ordering; both are needed. |

**The transferable lesson: the round-counting family of models is reliable for
ORDERING candidates and unreliable for SIZING them.** Every candidate it ranked
first won and every one it ranked last lost, while its magnitude predictions were
off by 2–12×. Sizing needs the per-shape non-mainloop fraction, which is not in
the geometry — the cheap proxy already printed by M7 is `bound` and `×SOL`, and
had I read shape 1's `bound = HOST` / `×SOL = 10.16` before predicting, I would
not have promised −25% on a row whose whole mainloop is ~4 µs.

## 6. The disqualified arm: `<64,64,128>` computes wrong results

**Reproducible in 15 of 15 draws**, always `max|diff| ≈ 2.6e-2` against
`max|ref| = 9.47e-2`. Characterised rather than guessed at: the epoch cells read
**1 on every scheduled CTA** and the output is **non-zero on all 8 ranks**, so the
kernel **ran to completion and computed wrong values** — this is not a failed
dynamic-LDS reservation and not a launch that silently did not happen. (That
distinction matters: an output left at its initial zeros reports
`max|diff| = max|ref|`, which looks identical through a tolerance check and means
the opposite thing. `diagnose_dead()` in `sweep.py` separates them.)

What it is **not**: LDS at the 65536 cap is not sufficient to break it —
`64/192/64` is also exactly 65536 and is correct on shape 2 and correct-but-slow
on shape 3. `BK = 128` is not sufficient either — `32/64/128` is correct and is
now landed. The pair differs only in `BM` (64 vs 32), i.e. in the A fragment:
`rt_bf<32,64>` (height 2 × width 4) versus `rt_bf<16,64>` (height 1 × width 4).
Every other instantiation in the tree has width ≤ 2 when height ≥ 2, so
`rt_bf<32,64>` is the one shape family with both height ≥ 2 and width ≥ 4.

**Not diagnosed further, and deliberately so** — it was pre-registered as the
predicted *loser* of the shape-2 pair, B1 beat it by construction on every model
column, and the discrimination it was there to provide was recovered from the
shape-3 pair instead. But it is a **latent hazard for E1**: a fragment shape that
silently produces wrong numerics with no error bit is the same class of failure as
the `acquire_frags` reordering bug, and any future work that widens `BK` or `BM`
must re-run M3 rather than assume the mainloop is shape-agnostic.

## 7. Gates

| gate | result |
|---|---|
| M0 node clean | PASS, 0 KFD pids |
| M1 build | PASS, three modules |
| **M2 resources / ISA** | **PASS, 7 of 7 instantiations**, `M2_EXPECT` updated 6 → 7 |
| M3 correctness, 17 shapes, `1e-2` **and** `2e-3` | **PASS 17/17**, `max\|diff\|` 3.05e-5 … 4.88e-4 |
| M4 negative controls | **PASS**, all three fail as designed |
| M5 600-epoch skewed soak | **PASS**, epoch and signal cells exact at 600 |
| M7 timing | below |

M2 resource tuples for the three new instantiations — **no spills anywhere**:

| instantiation | VGPR | AGPR | scratch | SGPR spill |
|---|---:|---:|---:|---:|
| `32/64/128` tail=0 (row 1, new) | 98 | 0 | 0 | 54 |
| `64/128/64` tail=0 (row 2, new) | 104 | 0 | 0 | 60 |
| `128/192/32` tail=1 (row 3, new) | 136 | 0 | 0 | 79 |
| `256/256/32` tail=0 (rows 4, 5) | 246 | 0 | 0 | 63 |
| `256/256/32` tail=1 (row 6) | 248 | 0 | 0 | 79 |
| `32/64/64` tail=0/1 (generic) | 91 / 92 | 0 | 0 | 58 / 88 |

`M2_EXPECT` went 6 → 7 because row 1 no longer shares `<32,64,64,false>` with the
generic fallback, which still needs both of its tail variants. The screening rows
101–107 are behind `-DHK_GEMM_RS_MI300X_TILE_SWEEP`, which neither `build.sh` nor
`m1_build.sh` defines, so they are absent from the graded binary and from this
count by construction.

**Stale artefact to be aware of:** `tools/m2_report.sh` prints a hardcoded note
listing the per-row LDS footprints (`36864/32768/49152/...`). Those are now wrong
for rows 1–3 (48K/48K/40K). It is a printed note, not a check, and `tools/**` is
off-limits to this experiment, so it was left alone.

### The screening rows were removed from the source, and why

`gemm_rs_mi300x_static_checks.py` gate 16 counts
`case \d+:\s*launch_fixed<...>` in the dispatch and requires **exactly 6** — that
is the "exactly one GPU kernel launch per call" invariant, and seven screening
arms are not worth weakening it. So the guarded block is preserved verbatim, with
its re-apply instructions and the outcome of all seven rows, in
`sweep_rows.patch.txt`; the `HK_GEMM_RS_MI300X_TILE_SWEEP` macro and its default
stay in the source so re-applying is a paste.

The removal is provably a codegen no-op: rebuilding from the cleaned source gives
**M2 resource tuples identical in every field** (98/104/136/246/248/91/92 VGPRs,
0 AGPRs, 0 scratch) and `distinct instantiations: 7`, and M3 re-passes 6/6 on the
scored shapes at both tolerances.

### Finding: the static-check suite has been dead since exp_02

Running it after landing produced `AssertionError: row 1: reducer 56 !=
RadeonFlow 32` — **exp_13's** landed change, asserted false. `require` raises on
the first failure, so the suite aborts at assertion #1 and none of its ~20 gates
runs. `static_probe.py` re-runs every gate with `require` collecting instead of
raising (modifying nothing on disk, since that file is not this experiment's):

```
collected 7 failed requirement(s)
  rows 1,2,3,4,6: reducer NN != RadeonFlow MM        <- 5 stale expectations
  gate 19: control module binding missing            <- 2, pre-existing
  gate 19: every control seam must be macro-gated
--- tile/dispatch related (would be exp_14's): 0
```

**Zero tile or dispatch failures**, so the retile breaks nothing in that suite and
gate 16 passes with the screening rows removed. But note *which* rows are stale:
rows 2, 3 and 4 were moved to uniform `NR=32` by **exp_02**, the first landed win
of the session — so this suite has been aborting on its first assertion since
then, and **nothing it checks has actually been verified all session.** That is
the mirror image of the M2 lesson already in HANDOFF: a gate that has never failed
is not evidence, and neither is one that always fails on line 1. Not repaired
here: the fix is to the `num_reducer_ctas` expectations, and this experiment is
explicitly barred from that axis.

## 8. Timing, both instruments

### Our harness — paired, same-run, 15 allocation draws (the trustworthy figure)

| # | shape | before (best) | **after (best)** | Δ | null floor |
|---:|---|---:|---:|---:|---:|
| 1 | 64×7168×18432 | 62.92 | **61.71** | **−1.9%** | 3.48% |
| 2 | 512×4096×12288 | 85.32 | **64.12** | **−24.9%** | 0.93% |
| 3 | 2048×2880×2880 | 88.96 | **82.71** | **−7.0%** | 1.61% |
| 4–6 | unchanged instantiation | — | — | 0 by construction | — |
| | **geomean ratio** | | | **0.9408 = −5.9%** | |

Ledger vector (**best-of-arm**, the same statistic the ledger carries):
`63.44 / 85.68 / 89.41 / 198.71 / 613.70 / 1616.63` (geomean 213.91) →
**`62.38 / 64.52 / 83.75 / 198.71 / 613.70 / 1616.63` (geomean 201.26)**.

### Gate M7 — means, 3 rotations × 50, same script both sessions

| # | shape | before (mean) | after (mean) | Δ | note |
|---:|---|---:|---:|---:|---|
| 1 | 64×7168×18432 | 66.63 | 65.65 | −1.5% | |
| 2 | 512×4096×12288 | 89.12 | **66.95** | **−24.9%** | `×SOL` 10.88 → 8.17 |
| 3 | 2048×2880×2880 | 92.26 | **86.92** | **−5.8%** | |
| 4 | 4096×4096×4096 | 200.64 | 201.65 | +0.5% | unchanged config |
| 5 | 8192×4096×14336 | 658.26 | 637.47 | −3.2% | unchanged config |
| 6 | 8192×8192×29568 | 1662.12 | 1610.43 | −3.1% | unchanged config |
| | **geomean** | **222.17** | **207.18** | **−6.75%** | |

**Read the M7 delta down, not up.** Rows 4–6 moved +0.5 / −3.2 / −3.1% with an
unchanged tile and an unchanged instantiation, so ~2 points of that −6.75% is
cross-session drift, exactly the contamination `LESSONS.md` records from exp_13.
**−5.9% paired is the number to carry.** Statistics are named on every vector
here because comparing an M7 *mean* to a recorded *best* is what manufactured a
phantom 7.3% regression once.

### Graded, same-run against frozen rank-1 (the competition's statistic)

`vs_rank1.sh exp14_landed` — 8 processes, the evaluator's own
`barrier → call → synchronize → barrier`, one fresh pool per shape, arms
interleaved with the order reversed every rep, 192 samples per arm per shape,
`submission.py` verified in sync with `hk_submission.py`. Both sessions reported
through one path (`prev_graded.sh` re-reports exp_13's stored samples).

| # | shape | ours before | ours now | rank-1 before | rank-1 now | ratio before | **ratio now** | Δratio |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64×7168×18432 | 156.77 | 150.64 | 176.87 | 177.15 | 0.886× | **0.850×** | −4.1% |
| 2 | 512×4096×12288 | 164.50 | **147.45** | 136.26 | 137.61 | 1.207× | **1.072×** | **−11.2%** |
| 3 | 2048×2880×2880 | 173.26 | 196.86 | 153.36 | 178.57 | 1.130× | **1.102×** | −2.5% |
| 4 | 4096×4096×4096 | 294.00 | 297.27 | 266.33 | 265.44 | 1.104× | 1.120× | +1.4% |
| 5 | 8192×4096×14336 | 755.50 | 737.93 | 574.78 | 573.75 | 1.314× | **1.286×** | −2.1% |
| 6 | 8192×8192×29568 | 1809.41 | 1762.44 | 1468.18 | 1458.44 | 1.232× | **1.208×** | −1.9% |
| | **geo best** | **348.64** | **345.15** | **306.60** | **314.46** | **1.137×** | **1.098×** | **−3.4%** |

Two things to be honest about.

**Our own graded number barely moved (−1.0%) while the ratio improved 3.4%.**
Shape 3's pool was globally slow this run — ours +13.6% *and* rank-1 +16.4% on
the same shape in the same processes — which is precisely what the same-run
anchor exists to absorb, and why the ratio is the statistic and the absolute
microseconds are not.

**The ratio has its own noise floor, and rows 4/5/6 measure it.** Their
configuration did not change, and their ratios moved +1.4 / −2.1 / −1.9%. So the
ratio floor is about ±2%, and only shapes **1 (−4.1%)** and **2 (−11.2%)** are
outside it; shape 3's −2.5% sits at the edge despite a clean −7.0% on the
pipelined instrument. Shape 2's −11.2% against a −24.7% kernel win is the ~83 µs
of per-call harness constant doing exactly what HANDOFF §3 says it does: the
kernel saved 21.2 µs and the graded call saved 17.1 µs of a 164.50 µs total.

The remaining gap is now **concentrated entirely in shapes 5 and 6** (1.286× and
1.208×), which are the two rows this experiment proved have **no tile-table
headroom at all** and whose `iter_cost` is locked by the LDS cap.

## 9. Instrument, and its one warrant

`sweep.py` is exp_13's protocol with one structural difference: `NR` is a host
scalar so one build served every arm of that sweep, but `BM/BN/BK` are template
parameters, so the arms live at `config_row` 101–107 in a separate module and the
resolved plan is **overridden in Python**. That override is the one place this
instrument could lie — a wrong `lrow_count` or `ready_words` would mis-size the
signal heap and read as a performance result — so it is validated, not trusted:

- `check_derivation()` asserts that deriving the **landed** tile through the
  Python path reproduces `rt.resolve_shape`'s dict **field for field on all six
  shapes**, and `main()` refuses to measure if any field differs. It passed on
  every run (printed at the top of every draw log).
- every arm asserts its derived `even_k` is the complement of the `K_TAIL` its
  `config_row` was compiled with, so a mismatched pair cannot silently run;
- `assert dict(handle.plan) == dict(plan)` after construction, so a plan that did
  not take is a crash rather than a number;
- `check_epochs()` + `check_signals()` after every arm's timing: **every**
  scheduled CTA's epoch cell and **every** touched ready/credit cell must equal
  the launch count. Zero failures across 15 draws × 4–5 arms × 3 shapes.

Nothing landed on the screening module: the landed rows go through
`scored_shapes` and the real dispatch cases, and the full ladder ran against the
production binding.

### Files

| file | what |
|---|---|
| `plan.md` | pre-registered predictions, written before any arm ran |
| `enumerate.py`, `enumeration.txt` | exhaustive legal-space enumeration and the three cost columns |
| `sweep.py` | the screening instrument (`SWEEP_REVERSE=1` reverses construction order) |
| `pool.py`, `pooled.json` | draw pooling, disjoint-range and exact rank-sum tests |
| `build_sweep.sh`, `go_build.sh` | the screening module, guarded by `HK_GEMM_RS_MI300X_TILE_SWEEP` |
| `run_sweep.sh`, `go_full.sh`, `go_more.sh`, `go_reverse.sh` | draw drivers |
| `geomean.py` | every geomean in §8, with its statistic named |
| `vs_rank1.sh`, `prev_graded.sh` | the graded instrument, and the before-table through the same reporter |
| `sweep_*.json` (15), `draw_archive/` | raw samples; the archived draw is the 3-arm validation run at scale 0.3 |
| `logs/` | build, ladder and sweep logs |

## 10. What this opens

1. **E1 is now the only place the two biggest shapes can improve.** Rows 4/5/6
   are at the tile-table optimum and pinned at the LDS cap, and their
   `waves × k_iters` of 16 / 112 / 464 at ~2650 non-MFMA cycles each is ~640 µs
   on shape 6 alone. The lever is making the double buffer stop costing 2× of
   `(BM+BN)·BK`.
2. **Shape 2's reduce geometry changed underneath its `NR`.** `red_tiles` fell
   64 → 32 and reduce rounds 2 → 1 at `NR=32`, and `col_count` fell 64 → 32.
   `NR` was last swept at the old geometry, so row 2 is a fresh E4 candidate.
   Shape 3's `red_tiles` moved 24 → 30 (still 1 round); shape 1's `cols` is
   unchanged at 112 so its `NR=56` optimum is untouched by design.
3. **Sweeps on this node should reverse construction order on half the draws.**
   The allocation-order component measured 0.9% on shape 1 and biases every arm
   built after the reference.
4. **`<64,64,128>` is a correctness hazard** in the mainloop at
   `rt_bf<32,64>` fragments, undiagnosed, with no error bit. Any future widening
   of `BK` or `BM` must re-run M3 rather than assume shape-agnosticism.

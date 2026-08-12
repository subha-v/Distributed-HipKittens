# exp_13 — E4 re-opened: the CTA-level producer/consumer boundary moved on two of six shapes

**Verdict. `NUM_REDUCER_CTAS = 32` was NOT still right, and the two rows that
moved are the two that the intervening changes touched.** The landed table is

| # | shape | NR before | **NR now** | why |
|---:|---|---:|---:|---|
| 1 | 64×7168×18432 | 32 | **56** | 1 producer wave for the whole range, so producers above 32 are free; reduce rounds 4 → 2 |
| 2 | 512×4096×12288 | 32 | 32 | flat; every candidate inside the floor |
| 3 | 2048×2880×2880 | 32 | 32 | flat; 1 wave and 1 reduce round for all NR ≥ 24 |
| 4 | 4096×4096×4096 | 32 | 32 | flat; the floor here is 2.4% and nothing beats it by that |
| 5 | 8192×4096×14336 | 32 | 32 | no reproducible movement; the two runs disagree in sign |
| 6 | 8192×8192×29568 | 32 | **48** | `NG=256` is the last split with 4 balanced producer waves; reduce rounds 4 → 3 |

Measured, same-run and paired, pooled over independent allocations:
**shape 1 −15.4%, shape 6 −4.9%, geomean −3.6%.** Full gate ladder passed. The
graded number against the frozen rank-1 in one pool went **353.59 → 348.64 µs**
against an anchor that reproduced to 0.6% (rank-1 306.60 vs 306.23–308.80), so
the graded ratio is **1.145–1.158× → 1.137×**.

**The naive version of the hypothesis was wrong, and the arithmetic said so in
advance.** Moving *down* to `NR=16` to buy GEMM does nothing: 1024 shape-6 tiles
over 272 or 288 producers is 4 waves either way, so the producer critical path
is flat, while reduce rounds double. Measured: `NR=16` is **+3.6%** on shape 6,
not −3.5%. The win is in the other direction, at the *last* split before the
producer wave count steps.

---

## 1. Lead with the floor, because it is bigger than most of the effects

The null arm is two independently allocated `GemmRS` instances carrying the
**same** split, interleaved in the same run under complete positional rotation.
Worst floor per shape over the four runs, and the position residual beside it:

| # | shape | null floor, best | null floor, median | position residual (max over slots) |
|---:|---|---:|---:|---:|
| 1 | 64×7168×18432 | 1.34% | 1.55% | ±0.30% |
| 2 | 512×4096×12288 | 0.56% | 0.34% | ±0.07% |
| 3 | 2048×2880×2880 | 0.61% | 0.47% | ±0.10% |
| 4 | 4096×4096×4096 | **2.41%** | **2.47%** | ±0.06% |
| 5 | 8192×4096×14336 | 2.17% | 2.23% | ±0.10% |
| 6 | 8192×8192×29568 | **4.28%** | **4.13%** | ±0.47% |

**The bias is per-allocation, not positional, and that is the durable finding
of this experiment.** Position was fully controlled here — `passes = len(arms)`
with a one-step rotation, so every arm occupied every slot exactly once — and
the measured position residual never exceeded ±0.47% on any shape in any run.
Yet two arms with an identical configuration, identical shared operand tensors
and identical code separated by **4.28% on shape 6** and 2.41% on shape 4. What
differs between them is only which `hipMalloc` they got for their 134 MB payload
heap. Rotating arm order — the standard defence, and the one the previous 3.9%
null-arm result motivated — **does not remove this**. Only a duplicate arm
reveals it, and only averaging over several independent allocations removes it.

Consequences, adopted for the rest of this report:

- Any single-run per-shape delta under ~4.3% on shape 6, ~2.4% on shape 4 and
  ~2.2% on shape 5 is **not evidence**, however many passes it averages.
- The instrument that *can* resolve these effects is the **pooled-allocation**
  view in §3: every independent allocation of a split, across all four runs,
  treated as one sample. Complete separation of two arms' ranges is worth more
  than any within-run percentage.

## 2. The pre-registered wave arithmetic, and how it did

Both roles are strided persistent loops, so the split does not buy CTAs, it buys
**rounds**: producer critical path `ceil(gemm_tiles / NG)`, reducer critical path
`ceil(red_tiles / NR)`, `NG = 304 − NR`, and `WGM = 4` iff `gemm_tiles ≤ NG`
(`gemm_rs_mi300x.cpp:308`). The plateaus and cliffs this predicts are exactly
what the sweep found:

| prediction (written before running) | outcome |
|---|---|
| producer path flat for all `NR ≤ 48`, so `NR<32` cannot buy GEMM | held: 4→16 on shape 6 is monotonically **worse** (+37.8%, +3.6%) |
| hard cliff at `NR=56` on shapes 2, 4, 5, 6 (one extra wave) | held exactly: +23.3% / +45.3% / +11.5% / +9.6% |
| shape 1 wants **more** reducers, one wave to `NR=80` | held: −15.4% at 56, the largest single effect of the night |
| `NR=48` weakly wins shapes 4 and 6 (`NG=256` balanced, fewer rounds) | held on 6 (−4.9%), **not** on 4 (inside a 2.4% floor) |
| shapes 2, 3, 5 keep 32 | held |
| net geomean ≤2% | **too pessimistic**: −3.6%, because shape 1's proportional win is worth 22× the same microsecond on shape 6 in a geometric mean |

## 3. The curve, per shape

Screening run: 11 arms `{4,8,16,24,32,40,48,56,64,80}` + null twin, 11 passes,
complete rotation, `iters = 300/250/250/120/50/25`. Best µs (median in §5's JSON;
the ranking is identical on both statistics for every row):

| NR | NG | s1 | s2 | s3 | s4 | s5 | s6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 300 | 220.44 | 179.71 | 151.24 | 352.86 | 885.12 | 2353.99 |
| 8 | 296 | 136.66 | 114.79 | 114.52 | 262.91 | 741.82 | 2006.70 |
| 16 | 288 | 94.67 | 91.99 | 101.27 | 218.39 | 685.51 | 1769.97 |
| 24 | 280 | 79.70 | 91.82 | 88.95 | 218.24 | 673.54 | 1761.08 |
| **32** | 272 | 74.44 | **85.68** | **89.41** | **198.71** | **669.39** | 1708.30 |
| 40 | 264 | 68.27 | 85.66 | 89.18 | 198.64 | 649.94 | 1711.48 |
| 48 | 256 | 69.00 | 85.95 | 89.08 | 199.23 | 654.55 | **1616.63** |
| 56 | 248 | **63.44** | 105.60 | 88.61 | 288.73 | 746.46 | 1872.12 |
| 64 | 240 | 63.24 | 104.07 | 88.80 | 284.52 | 773.36 | 1788.53 |
| 80 | 224 | 63.62 | 102.52 | 88.84 | 283.46 | 741.75 | 1710.69 |
| 32* (null) | 272 | 75.44 | 86.16 | 88.87 | 199.49 | 655.16 | 1726.40 |

Uniform-NR geomean over the six shapes: `32 → 224.95`, `40 → 220.60`,
`48 → 219.35`, `56 → 249.21`. A *uniform* move to 48 is worth −2.5%; the
per-shape table is worth more because shapes 1 and 6 want different values.

### Pooled over independent allocations — the view that beats the floor

Every independent allocation of a split across all four runs, best µs:

| # | `NR=32` samples | `NR=32` median | candidate samples | candidate median | verdict |
|---:|---|---:|---|---:|---|
| 1 | 74.44, 75.44, 74.46, 74.61, 74.61, 74.85 | 74.61 | **56:** 63.44, 62.90, 63.16 | **63.16** | **−15.4%, ranges disjoint** |
| 2 | 85.68, 86.16, 85.48, 85.52, 85.63, 85.58, 85.49 | 85.58 | 40: 85.66, 85.76 · 48: 85.95, 85.52 | — | inside; keep 32 |
| 3 | 89.41, 88.87, 89.39, 89.36, 89.32, 88.80, 88.97 | 89.32 | 48: 89.08, 88.64 · 56: 88.61, 89.07 | — | ≤0.2% below the min; keep 32 |
| 4 | 198.71, 199.49, 202.38, 198.22, 202.76, 197.98, 197.40 | 198.71 | 40: 198.64, 198.62 · 48: 199.23, 198.62 | — | inside a 2.7% same-config range; keep 32 |
| 5 | 669.39, 655.16, 644.71, 651.22, 653.70, 655.29, 653.57 | 653.70 | 24: 673.54, 636.03 · 40: 649.94, 646.36 · 48: 654.55, 664.59 | — | inside a 3.8% same-config range; keep 32 |
| 6 | 1708.30, 1726.40, 1692.96, 1658.40, 1727.71, 1656.85 | 1700.63 | **48:** 1616.63, 1583.95, 1649.38 | **1616.63** | **−4.9%, ranges disjoint** |

For shapes 1 and 6 the two arms' ranges do **not overlap at all** — every one of
the three `NR=56`/`NR=48` allocations is faster than all six `NR=32` allocations.
Under the null that the split does not matter, the chance of the three candidate
samples being the three lowest of nine is `1/C(9,3) = 1.2%`. That is the
strongest statement this node's measurement noise permits, and it is the reason
these two rows landed while shapes 3, 4 and 5 did not, despite shape 4 and 5
showing 2–3% "wins" in individual runs.

Geomean of the landed change on pooled medians:
`(63.16/74.61 × 1616.63/1700.63)^(1/6) = 0.9645`, i.e. **−3.6%**.

### Why these two rows and not the others — the mechanism, per row

- **Shape 1 was created by exp_04a.** Retiling `32/256/32 → 32/64/64` took
  `col_count` 28 → 112 and therefore `red_tiles` 28 → **112**, which took the
  reduce side from 1 round to 4 at `NR=32`. The tile change quadrupled the
  reducer's critical path and the split was never re-swept. `NR=56` is exactly
  `112/2`, i.e. two tiles per reducer with zero imbalance, and it still leaves
  248 ≥ 224 producers, so the GEMM does not notice. This also revises a recorded
  finding: the "~66 µs shape-1 floor that no stage cut explains" was measured at
  `NR=32`; shape 1 now runs at **63.2 µs with the mainloop included**, so a real
  part of that floor was serialized reduce rounds, not a protocol constant.
- **Shape 6 was created by exp_08 and E3.** `NG=256` is the last split that
  keeps 4 producer waves and it divides 1024 exactly (4.00 tiles per CTA against
  3.76 at `NG=272`, which still costs 4 rounds), while reduce rounds fall 4 → 3.
  When the egress pool was 1149.9 µs it dominated and this trade was invisible;
  after exp_08 cut egress to 412.4 µs, three reduce rounds instead of four is
  worth 4.9%.
- **Shapes 2–5 sit mid-plateau in both roles**, which is why they are flat and
  why the original sweep's answer survives there. Flat is not "the axis does not
  matter" — it means the quantization has no boundary nearby.

## 4. Gates

`tools/gate_ladder.sh exp_13_cta_split`, all steps, logs in `logs/`:

| gate | result |
|---|---|
| M0 node clean | 0 KFD pids, clocks `perf_determinism` |
| M1 build | ALL MODULES BUILT |
| M2 resources/ISA | 6 instantiations, **unchanged** tuples (91/91/163/246/248/92 VGPR, AGPR 0, scratch 0, zero VGPR spills). The split is a host-side plan constant, so no device code changes |
| M3 correctness | **17/17 shapes PASS at both `1e-2` and `2e-3`**, worst `max\|diff\| = 4.883e-4` |
| M4 negative controls | 3/3 fail as designed (drop-publication → bit 26 with zero payload reads; drop-credit → bit 25 at epoch 2; reroute-slot → bitwise corruption, `max\|diff\|` 1.065e-2) |
| M5 soak | 600 skewed changing-input epochs, epoch and signal cells exact, worst `4.883e-4` |
| M7 timing | means `66.63 / 89.12 / 92.26 / 200.64 / 658.26 / 1662.12`, geomean **222.17 µs**, all correct |

Correctness was additionally re-checked **inside every sweep arm** at both
tolerances, including the starved splits (`NR=4`, 32 reduce rounds on shape 6)
and the over-provisioned ones (`NR=80`): 41 arm-instantiations, zero error bits,
and `check_epochs` + `check_signals` exact on every one. A bad split is a
protocol stress test and this axis passes it everywhere in `[4, 80]`.

## 5. The graded number, and a cross-instrument confirmation

`vs_rank1.sh nr_table_1` — 8 processes, the evaluator's own
`barrier → call → synchronize → barrier`, one fresh pool per shape, ours and the
frozen rank-1 submission interleaved with the order reversed every rep, 192
samples per arm per shape, `submission.py` verified in sync with
`hk_submission.py`.

| statistic | ours before (E3) | **ours now** | rank-1 before | **rank-1 now** | ratio before | **ratio now** |
|---|---:|---:|---:|---:|---:|---:|
| geo best | 353.59 / 357.09 | **348.64** | 308.36 / 308.80 | **306.60** | 1.145× / 1.158× | **1.137×** |
| geo median | 368.65 / 371.29 | **365.89** | — | 320.25 | — | 1.142× |

The anchor reproduced to 0.6% (306.60 against 308.36/308.80), so the run is
comparable and the ratio improvement of 0.7–1.8% is not a globally fast run. Per
shape, best µs: shape 1 **168.29 → 156.77**, shape 6 **1878.41 → 1809.41**,
shape 5 729.74 → 755.50 (unchanged configuration; shape 5 is the least stable
shape on both instruments, see §6).

**The two instruments agree to within a microsecond on the mechanism**, which is
the strongest evidence in this report that the effect is real and not a
measurement artifact:

| shape | pipelined Δ (device work) | graded Δ (per-call) |
|---|---:|---:|
| 1 | 74.61 → 63.16 = **−11.45 µs** | 168.29 → 156.77 = **−11.52 µs** |
| 6 | 1727.71 → 1649.38 = −78.3 µs | 1878.41 → 1809.41 = −69.0 µs |

Shape 1's graded improvement is the device saving, transferred one-for-one onto
a per-call number that also carries a ~92 µs harness constant. That is what a
genuine device-side win looks like on this pair of instruments.

## 6. The denominator correction that this experiment forced

The recorded per-shape denominator `77.47 / 88.28 / 91.79 / 199.43 / 613.70 /
1716.76` is a vector of **best-of-arm** values, not means: exp_05's own table
records shape 5 as `613.70 / 645.93` (best / median). M7 prints **means**. So
comparing today's M7 geomean (222.17) to 225.62 compares a mean to a best and
under-reports the change; it is also why shape 5 appeared to have "regressed
7.3%" with an unchanged configuration. Measured today, shape 5 at `NR=32` reads
644.71–669.39 across seven independent allocations, entirely consistent with its
recorded **median** of 645.93 and inconsistent with its recorded best of 613.70.

**Rule: state the statistic with the vector.** Best-vs-best across sessions is
also fragile for shape 5, whose lower tail is fat; the paired same-run
comparison in §3 is the only denominator that survives contact with this node.

## 7. Pre-emptive negative: the dual-role / work-stealing reducer CTA is worth ~zero on the scored shapes, by arithmetic

The idea — let an idle reducer CTA execute GEMM tiles — sounds compelling
because the reduce is only ~86 µs of work in a ~1650 µs kernel, so reducer CTAs
are idle >90% of the time. **It cannot pay on any of the six scored shapes, and
this needs no code and no GPU time to establish.** The producer critical path is
`ceil(gemm_tiles / producers)` whole tiles, and going from today's producer count
to all 304 CTAs never removes a wave:

| # | gemm_tiles | waves at the landed NG | waves at NG=304 | gain |
|---:|---:|---:|---:|---|
| 1 | 224 | 1 (NG=248) | 1 | none |
| 2 | 512 | 2 (NG=272) | 2 | none |
| 3 | 192 | 1 | 1 | none |
| 4 | 256 | 1 | 1 | none |
| 5 | 512 | 2 | 2 | none |
| 6 | 1024 | 4 (NG=256) | 4 | none |

The tile counts are small multiples of the CTA count, so the critical path is
quantized in whole waves and the last wave is already partly empty — the stolen
work would land in slack that costs nothing today. Work stealing only pays where
tiles-per-CTA is large, i.e. the **generic** row (up to 118 tiles per CTA), which
is not scored. Per the "measure the mechanism before building the fix" rule, this
is closed without touching the protocol. It would also have needed a re-read of
`exp_05_release_granularity/protocol_review.md` §2 and a new epoch-lifetime
argument for a CTA that changes role, for no expected gain.

## 8. Final state, verified rather than asserted

`final_state.sh` rebuilds from the source that is actually on the node and reads
the split back out of the **compiled runtime** rather than the header
(`plan_check.py`):

```
shape                  row        tile  NR   NG   tiles  waves   red  rounds  WGM  ok
64x7168x18432            1    32/64/64  56  248     224      1   112       2    4 yes
512x4096x12288           2    64/64/64  32  272     512      2    64       2    8 yes
2048x2880x2880           3  128/256/32  32  272     192      1    24       1    4 yes
4096x4096x4096           4  256/256/32  32  272     256      1    32       1    4 yes
8192x4096x14336          5  256/256/32  32  272     512      2    64       2   32 yes
8192x8192x29568          6  256/256/32  48  256    1024      4   128       3   32 yes
generic row (unchanged, must be 24): NR=24 row=0
PLAN CHECK PASSED
```

M3 re-passes 6/6 at both tolerances on that rebuild, and
`gemm_rs_mi300x.so` comes back **md5-identical** (`08b0364d…fbaf`) to the
pre-change module — direct evidence that this experiment changed a host-side
plan constant and no device code, so nothing in the mainloop, the `WGM` order or
the release logic could have moved underneath the measurement.

## 9. What this opens next

1. **`(BN, NR)` is a joint optimum and neither sweep knew about the other.**
   `red_tiles = lrow_count · ceil(N/BN)`, so the tile table sets the reducer's
   work and the split sets its parallelism. exp_04a changed `BN` without
   re-sweeping `NR`, and this experiment changed `NR` without re-sweeping `BN`.
   Shape 1 is the proof that the interaction is worth 15%. The cheap version:
   for each shape, sweep `BN` and `NR` together over the two or three points
   where a wave or round boundary moves.
2. **Re-run the attribution on the new winner.** Shape 6's reduce pool was
   86.5 µs at 4 rounds; at 3 rounds the ranking of the remaining pools has
   moved, and shape 1's ~66 µs "unattributed floor" needs re-measuring at
   `NR=56` now that 11.5 µs of it turned out to be reduce rounds.
3. **Every arm-identity measurement on this node should carry a duplicate arm.**
   The 4.28% shape-6 figure means several past sub-5% verdicts on this node were
   taken with an instrument whose floor was never measured on that shape.

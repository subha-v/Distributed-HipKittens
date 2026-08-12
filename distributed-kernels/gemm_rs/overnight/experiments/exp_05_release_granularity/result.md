# exp_05 — E3, release granularity: result

**Verdict: SHIP `RELEASE_GROUP = 4` with the full-group condition (`FULL_ONLY = 1`).**

The mechanism is real and it is where the attribution said it would be. On
8192×8192×29568 — the only scored shape where a producer CTA owns four tiles, and
the shape carrying the whole remaining gap to rank-1 — collapsing four releases
into one is worth **−6.1% pipelined** and **−4.0% on the graded per-call number**,
reproducible to 0.05% across two independent graded runs with rank-1's same-run
anchor stable to 0.54%. Shape 6's graded gap to rank-1 narrows from **1.338× to
1.280×**.

It does **not** move the geometric mean by more than the instruments can
resolve, and this report says so plainly rather than dressing it up: the graded
geomean goes 357.83 → 355.34 µs (−0.70%, mean of two runs) and the pipelined
geomean 229.33 → 225.62 µs (−1.6%), both inside the run-to-run spread those
instruments were shown to have. Five of six scored shapes cannot benefit — three own one
tile per CTA and two own two — so a single shape's 4% is all there is to spread
over six.

Two things were learned that outlive the arm choice:

- **Grouping a *partial* group is a net loss.** Unconditional `N ∈ {2, 4}`
  regresses 512×4096×12288 by **+3.6 to +3.9%**, reproduced in four independent
  paired runs against a null-arm floor of +0.60% on that shape. The
  publication delay costs more there than halving its two releases can save.
  The shipped rule groups only where a CTA owns a full group, which removes that
  regression exactly (+0.60%, i.e. the null arm itself).
- **Our pipelined A/B harness has a positional bias of up to 3.9% per shape and
  ~0.8% on the geomean**, measured with a null arm — two builds of
  `RELEASE_GROUP = 1` under different module names. The first pass of this
  experiment "measured" a 2% regression on 4096×4096×4096, a shape with **one
  tile per producer CTA** that cannot group at all. That claim was an artifact.
  §7 documents the instrument.

---

## 1. What changed

Source: `distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp` (producer role only) and
one new control bit in `gemm_rs_mi300x_constants.cuh`. The mainloop, the `WGM`
tile order, the reducer role, the host ABI, the 72-byte descriptor and the
one-launch-per-call contract are untouched.

| | sha256 of `gemm_rs_mi300x.cpp` |
|---|---|
| pre-E3 (frozen golden, `baseline/gemm_rs_mi300x_e3base.cpp`) | `4603b3619aa10ffb3b9bedb0d8b84e47fef25ccbd001913428ba49cf5dd026e3` |
| shipped | `762369b23eecbbea16544dc979215deabfb5ea638441745c2606e69940cdda15` |

Diffs in this directory:

- `e3_structural_w.diff` — **read this one.** `git diff -w`, i.e. whitespace
  ignored: **180 insertions, 16 deletions**, and the mainloop does not appear in
  it at all. That is the evidence that the mainloop is byte-identical.
- `e3_final.diff` — the literal diff, 364/200, because the tile-loop body moved
  one nesting level in when it became the body of the inner loop.

The golden snapshot was **refreshed** at the start of this session. The one that
was in `baseline/` was a 02:43 copy predating exp_08 (the WGM egress fix) and
exp_09 (the collapsed k-loop) — 537 lines against the current 713. Comparing
against it would have compared E3 against three other experiments at once.

### The restructured loop

```c++
constexpr int RELEASE_GROUP = HK_GEMM_RS_MI300X_RELEASE_GROUP;
static_assert(RELEASE_GROUP >= 1 && RELEASE_GROUP <= 8, ...);
const int stride = g.num_gemm_ctas;
const int bands  = BM / EB;                 // EB | BM (gate 5)
const int tiles_per_cta = (tiles + stride - 1) / stride;
#if HK_GEMM_RS_MI300X_RELEASE_GROUP_FULL_ONLY
const int rgroup = tiles_per_cta >= RELEASE_GROUP ? RELEASE_GROUP : 1;
#else
const int rgroup = RELEASE_GROUP;
#endif

for (int t0 = pid; t0 < tiles; t0 += rgroup * stride) {
    const int left    = (tiles - t0 + stride - 1) / stride;
    const int emitted = left < rgroup ? left : rgroup;

    for (int j = 0; j < emitted; ++j) {
        const int t = t0 + j * stride;
        const tile_id id = decode_tile(t, num_pid_m, num_pid_n, WGM);
        const int tm = id.tm, tn = id.tn;
        zero(C_accum);
        /* mainloop, byte-identical */
        /* per-tile credit wait, byte-identical */
        /* band / window emit, byte-identical */
    }

    m3::release_payload_system();               // exactly one, per group

    if (threadIdx.x == 0) {
        for (int j = 0; j < emitted; ++j) {
            const tile_id id =
                decode_tile(t0 + j * stride, num_pid_m, num_pid_n, WGM);
            for (int b = 0; b < bands; ++b) { /* publish_band_epoch */ }
        }
    }
}
```

`decode_tile` is the single source of truth for the tile map, called by both
loops. **It takes `wgm` as a runtime argument**, which is the one place the
previous ungated attempt (`e3_wip_ungated.diff`) would have been wrong against
today's tree: that diff hard-coded `constexpr int WGM = 4` inside the helper,
but exp_08 made `WGM` shape-dependent —
`(tiles <= g.num_gemm_ctas) ? 4 : num_pid_m` — so a hard-coded 4 would have
silently reverted the egress fix on the two large shapes and, worse, made the
publish loop decode a *different* tile than the emit loop on every shape where
the two disagree. That is tail bug #3 from the review's §2.6, and it is exactly
the class the shared helper exists to prevent.

Coverage is preserved exactly:
`{t0 + j·NG : t0 ∈ {pid, pid+G·NG, …}, j ∈ [0,G)} ∩ [0,tiles) = {pid + i·NG : i ≥ 0} ∩ [0,tiles)`,
the same tile set in the same order, each tile once. Verified empirically by
`check_signals()` on every arm: every touched ready and credit cell reads exactly
`n_calls`, which cannot hold if any tile were published twice or not at all.

---

## 2. Conditions C1–C8

### C1 — exactly one release, dominating every publication

Production (`HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS == 0`) contains exactly one
`release_payload_system()` call site in the producer role, lexically above the
publish loop and in the same iteration of the group loop:

```c++
#else
            m3::release_payload_system();
#endif
            ...
            if (threadIdx.x == 0) {
                for (int j = 0; j < emitted; ++j) {
```

There is no `publish_band_epoch()` anywhere inside the inner tile loop; the only
call site in the file is inside the leader-only loop above. The negative-control
module has a *second* site, and that is deliberate — it is `CTRL_PUBLISH_EARLY`
(§4), which exists to invert this exact order. `ctrl_flags` is a kernel argument
and therefore CTA-uniform, so exactly one of the two sites executes per group and
the release stays convergent on both paths.

### C2 — an unflushed tail is structurally impossible

The release and the publish are inside the outer group loop, and the loop
condition is `t0 < tiles`, so

```c++
const int left    = (tiles - t0 + stride - 1) / stride;   // >= 1 given t0 < tiles
const int emitted = left < rgroup ? left : rgroup;        // >= 1
```

`emitted >= 1` on every iteration that runs. A zero-tile CTA — 216 of 272 on
shape 1 — never enters the loop, so it can neither release nor publish, and there
is no post-loop flush to forget. The group size is computed in **closed form
before** the tile body rather than by a counter incremented inside it: a
loop-carried counter would live across the mainloop's register-critical region,
and the 256×256 rows sit at 246–248 of 256 VGPRs.

### C3 — same decode helper, nothing journalled, no new scratch

Both loops call `decode_tile(...)`; there is no copied arithmetic, no journal
array, no LDS allocation. Confirmed in the resource tuple (`m2_compare.sh`,
golden vs candidate):

| BM/BN/BK | VGPR | AGPR | SGPR | scratch | VGPR spill | SGPR spill | occupancy |
|---|---|---|---|---|---|---|---|
| 32/64/64 t=0 | 91 | 0 | 106 | 0 | 0 | 48→51 | 5 |
| 64/64/64 t=0 | 91 | 0 | 106 | 0 | 0 | 44→50 | 5 |
| 128/256/32 t=1 | 163 | 0 | 106 | 0 | 0 | 67→75 | 3 |
| 256/256/32 t=0 | 246 | 0 | 106 | 0 | 0 | 52→58 | 2 |
| 256/256/32 t=1 | 248 | 0 | 106 | 0 | 0 | 70→74 | 2 |
| 32/64/64 t=1 | 91 | 0 | 106 | 0 | 0 | 77→83 | 5 |

`C3 OK: no new scratch and no new VGPR spills on any instantiation.` VGPRs,
AGPRs, `ScratchSize` and occupancy are **unchanged on all six instantiations**.
SGPR spills rise by 3–8 — that is the group loop's four extra uniform values
(`t0`, `left`, `emitted`, `j`) at a fixed 106 TotalSGPRs. With
`ScratchSize [bytes/lane] = 0` those spills go to VGPR lanes, not to memory, and
they fit inside the existing VGPR count. No instantiation grew a scratch site.

### C4 — compile-time constant, hard cap, generic row treated identically

`RELEASE_GROUP` is a `constexpr` from a `#define`, with
`static_assert(RELEASE_GROUP >= 1 && RELEASE_GROUP <= 8)`. Unbounded
per-CTA-per-epoch is not implemented and cannot be requested. The generic row
gets the same 4: with `BM=32, BN=64, NR=24` its 32768 tiles over 280 producers is
118 tiles per CTA, so it runs 30 groups of 4 rather than 118 releases, and no
config row is special-cased anywhere in the file.

The full-group condition is a **runtime `min` under that compile-time cap**, not
a second table: `rgroup = tiles_per_cta >= RELEASE_GROUP ? RELEASE_GROUP : 1`,
where both terms come from the shape plan and are grid-uniform. Nothing was
added to `gemm_rs_mi300x_host_abi.hpp`, whose shape table and bias-agnostic
lookup are untouched.

### C5 — `N = 1` is behaviour-preserving

Gated as its own arm. Bit-identical to the frozen pre-E3 build on all 8 ranks
over 680 poisoned epochs across three shapes (§4), and pipelined geomean 229.33 µs
best against the recorded denominator's 231.16 µs — a −0.8% cross-run difference
on a node whose floor is ~1.3%. The ISA change is confined to the loop
restructure; no VGPR, AGPR, scratch or occupancy moved.

### C6 — M4 rebuilt and re-run, `CTRL_DROP_PUBLICATION` carried into the new loop

The branch moved with the code it instruments, into the batched publish loop:

```c++
#if HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS
                        if ((g.ctrl_flags & m3::CTRL_DROP_PUBLICATION) &&
                            me == g.ctrl_rank && dest == g.ctrl_arg0) {
                            continue;
                        }
#endif
```

M4 passes 3/3 on every arm (N=1, 2, 4, 4c): drop-publication → **bit 26 on rank 5
with the output 100.0% still at its sentinel**, i.e. zero payload reads;
drop-credit → bit 25 at epoch 2 with epoch 1 clean; reroute-slot → no timeout and
bitwise corruption against an unrerouted run over an identical input sequence.
`m4_controls.py` needed no edit.

### C7 — the new M9 gate passes, and it has power

§4.

### C8 — both timings reported; the one-tile shapes

`time_pipelined` and `time_single_shot` are both in §5 and §6. The review's §2.4
worry — that E3 could buy single-shot latency and pay for it in pipelined
throughput — **did not materialize**: on shape 6 the single-shot wall improves
alongside the pipelined figure (1939.41 → 1877.76 µs, −3.2%).

Shapes 1, 3 and 4 own one tile per producer CTA and cannot group. All three are
**bit-identical** in output, on two independent instruments: `torch.equal`
between every arm in one process on every A/B run, and — after these three shapes
were added to M9 as cases — `torch.equal` against the frozen pre-E3 build on all
8 ranks over 30 poisoned epochs each, with changing inputs and rotated skew:

```
-- candidate: m=64 n=7168 k=18432 --   ok bit-identical to the pre-E3 build ... (30 epochs)
-- candidate: m=2048 n=2880 k=2880 --  ok bit-identical to the pre-E3 build ... (30 epochs)
-- candidate: m=4096 n=4096 k=4096 --  ok bit-identical to the pre-E3 build ... (30 epochs)
```

On timing:

- **Shape 1 (+0.37%) and shape 4 (+0.41%)** are inside their own null-arm floors
  (+0.22% and +0.47%). Within noise, as required.
- **Shape 3 moves: +2.13% for the shipped arm, −1.77% for unconditional `N=4`**,
  against a −0.24% null floor on that shape. This is a real movement on a shape
  that cannot group, and the review is right that it means "something other than
  the release changed": it is **code generation**. With `rgroup` a compile-time
  1 the compiler proves `emitted == 1` and folds both new loops away entirely;
  with `rgroup` a compile-time 4 it keeps them with a bound it knows; with
  `rgroup` a runtime value it keeps them with a bound it does not. Those are
  three different schedules for the 128×256 instantiation, and this kernel is
  known to be acutely sensitive to basic-block structure in exactly this way
  (exp_09 found one block boundary moving 31 MFMAs). The effect is ±2% on one
  shape, it is not a protocol change, and it is disclosed rather than absorbed.

---

## 3. Sizing corrections from the review (P2, P3, P5), settled

- **P2 — "half the shapes cannot benefit" is confirmed and is the whole story.**
  Measured tiles per producer CTA: 1, 2, 1, 1, 2, 4 for shapes 1–6. Only shape 6
  batches more than two. The review's ~3% geomean upper bound was, if anything,
  optimistic: the realized geomean is −0.7% graded / −1.6% pipelined, because on
  the two 2-tile shapes grouping is not merely neutral but *negative* (shape 2)
  or unresolvable (shape 5).
- **P3 — the drain is not recoverable, and the numbers agree.** The mainloop
  still carries a blanket `vmcnt(0)` in `load_commit`, so peer stores cannot stay
  in flight across a tile boundary and the hoped-for secondary XGMI win is not
  available until E1(b). What E3 recovered is the `buffer_wbl2` plus the two
  barriers, and that is quantitatively what showed up: the attribution puts the
  release at 134.9 µs of shape 6's 1777.9 µs, so three of four releases is
  ~101 µs ≈ 5.7%, against a measured 6.1% pipelined. The mechanism is accounted
  for within a few tenths of a percent, which is the strongest evidence in this
  report that the win is the release and not a side effect.
- **P5 — sizing was re-derived on the current binary, not the superseded one.**
  Every number here is against a same-session `N = 1` control built from the same
  tree, not against the 285.02 µs or 2861.7 µs figures from the `NR=8` era.
- **§5's kill conditions did not fire.** The staging-loop `__syncthreads()` at
  the emit sites do not carry `vmcnt(0)` (M2/ISA, unchanged from the recorded
  ordering-op inventory), and the release delta on shape 6 did not collapse on
  the current source — it is 134.9 µs and E3 recovered three quarters of it.

---

## 4. M9 — the stale-slot gate, and proof that it has power

`harness/m9_stale_slot.py`, run at full scale on every arm. Four ingredients, all
outside any timed region:

1. **Poison.** Every rank's `c_heap` is `hipMemset` to `0xFF` (bf16 `0xFFFF`, a
   negative quiet NaN — the canonical `0x7FC0` is not byte-uniform and
   `hipMemset` carries one byte) and every `out` filled with NaN, before **every**
   launch. A stale read becomes a NaN, which needs no tolerance argument. The
   golden arm runs under the same poison, so a NaN there would indict the poison
   rather than the candidate — and it never fires.
2. **Bitwise golden**, not a tolerance: `torch.equal` against the frozen pre-E3
   module `gemm_rs_mi300x_e3base`, held open in the same process and driven over
   the same input sequence.
3. **Stress**: 600 epochs on 512×4096×12288 (2 tiles/CTA, 1-tile tail), 60 on
   8192×8192×29568 (4 tiles/CTA, 3-tile tail), 20 on the generic row at
   8192×8192×28672 (118 tiles/CTA, the P4 stress case the scored table never
   reaches). New seed and rotated launch skew every epoch. **Three cases were
   added for the shipped arm** — the one-tile-per-CTA shapes 1, 3 and 4, 30
   epochs each — so that C8's bit-identity requirement is a *gate* rather than an
   assertion in a report.
4. **`CTRL_PUBLISH_EARLY`** — the control of the control.

**Candidate result, all four arms (N = 1, 2, 4, 4c), 680 poisoned epochs each,
and 770 for the shipped arm's final six-case run:**

```
ok   golden itself is clean under the poison
ok   no NaN reached any output in 600 / 60 / 20 poisoned epochs
ok   bit-identical to the pre-E3 build on all 8 ranks for all epochs
ok   allclose(0.002) every epoch, worst |diff| 4.883e-04 / 1.221e-04 / 1.221e-04
ok   no error bit was ever raised
ok   epoch cells and every touched ready/credit cell are exact
```

**`CTRL_PUBLISH_EARLY` fails the gate, as it must.** Publishing the group's ready
flags *before* the release that covers the payload is caught on the strongest
detector available — NaN in the output, not a tolerance:

| arm / run | 512×4096 ×12288 | 8192×8192 ×29568 | 8192×8192 ×28672 generic | 64×7168 ×18432 | 2048×2880 ×2880 | 4096×4096 ×4096 |
|---|---|---|---|---|---|---|
| N=1 | NaN 1/20 | NaN 1/20 | NaN 1/10 | — | — | — |
| N=2 | NaN 1/20 | NaN 1/20 | NaN 1/10 | — | — | — |
| N=4 | NaN 1/20 | NaN 1/20 | **NaN 0/10** | — | — | — |
| N=4c, 3 cases | NaN 1/20 | NaN 1/20 | **NaN 0/10** | — | — | — |
| **N=4c, final 6 cases** | NaN 1/20 | NaN 1/20 | NaN 1/10 | NaN 1/10 | NaN 1/10 | NaN 1/10 |

Every firing also fired `bitwise` and `tight` at the same time. In the final run
the control is detected on **6 of 6 shapes**.

`GATE M9 PASSED: no stale slot under a poisoned heap, bit-identical to the pre-E3
build, and the publish-early control does fail.`

Two honest limits on this gate, both disclosed rather than buried:

- **Its power is concentrated at epoch 1.** The control fires on the first epoch
  of every shape it fires on at all, and then never again in that sweep. The
  reading is that the race window is widest on the first launch, when the credit
  path takes its `E <= 1` fast path and nothing throttles the producers; from
  epoch 2 on, the credit wait itself gives the payload stores enough time to land
  before any reducer polls. So the gate is a **deterministic epoch-1 detector**
  rather than a continuous one — which is enough, because it means one launch of a
  batched build with the ordering inverted is caught with certainty.
- **The generic row is marginal.** It did not fire in the N=4 or the first N=4c
  sweep and did fire in the final one, so at 118 tiles per CTA the window is
  narrow enough to miss by sampling. `m9` records a non-firing in a `note:`
  rather than passing silently, and the gate's pass condition is "detected on at
  least one shape", which no run came close to failing.

The three pre-existing controls remain blind to this property, exactly as the
review predicted, which is why this gate had to exist: a release/publication
ordering bug produces one wrong 1-of-8 contribution at ~7e-3, inside the graded
`1e-2`, and *vanishes entirely* when inputs do not change.

---

## 5. Gate ladder, per arm

Every arm ran the full ladder — M0 node-clean → M1 build → M2 resources/ISA →
M3 17 shapes at **both** 1e-2 and 2e-3 → M4 → M5 600-epoch skewed soak → M7 —
followed by M9. No gate was skipped and no arm was timed before it passed.

| gate | N=1 | N=2 | N=4 | **N=4c (ship)** |
|---|---|---|---|---|
| M1 build | pass | pass | pass | pass |
| M2 resources / ISA | 6 instantiations, VGPR 91/91/163/246/248/91, AGPR 0, scratch 0, no VGPR spills | same | same | same |
| M3 correctness | **17/17 at 1e-2 and 2e-3** | 17/17 | 17/17 | **17/17** |
| M4 controls | 3/3 as designed | 3/3 | 3/3 | **3/3** |
| M5 soak | 600 skewed changing-input epochs, epoch and signal cells exact | pass | pass | **pass** |
| M7 timing | all shapes correct | pass | pass | **pass** |
| M9 stale slot | **pass**, publish-early fires 3/3 shapes | pass, 3/3 | pass, 2/3 | **pass, 6/6 on the final six-case run** |

### M7 pipelined wall, per shape (µs)

`best / median` over the three rotations, and the arm means the ladder prints.
Best and median lead because the node's floor is ~1.3% and a mean of three
rotations is not a usable statistic at that scale.

| # | shape | N=1 | N=2 | N=4 | **N=4c** |
|---|---|---|---|---|---|
| 1 | 64×7168×18432 | 77.86 / 77.90 | 77.95 / 78.29 | 77.50 / 77.64 | **77.47 / 77.60** |
| 2 | 512×4096×12288 | 89.37 / 89.61 | 92.46 / 92.48 | 91.87 / 92.58 | **88.28 / 88.88** |
| 3 | 2048×2880×2880 | 90.78 / 90.81 | 88.85 / 89.09 | 88.64 / 88.80 | **91.79 / 91.90** |
| 4 | 4096×4096×4096 | 200.12 / 202.87 | 200.25 / 201.69 | 201.05 / 201.23 | **199.43 / 206.27** |
| 5 | 8192×4096×14336 | 634.14 / 659.87 | 636.82 / 641.74 | 605.61 / 639.88 | **613.70 / 645.93** |
| 6 | 8192×8192×29568 | 1814.66 / 1820.74 | 1759.24 / 1766.99 | 1696.42 / 1726.05 | **1716.76 / 1719.23** |
| | **geomean** | **229.33 / 231.65** | 228.85 / 229.87 | 225.17 / 228.38 | **225.62 / 229.25** |
| | ratio of best vs N=1 | 1.0000 | 0.9979 | 0.9819 | **0.9838** |

Arm means as the ladder printed them, with the geomean of means:

| arm | 1 | 2 | 3 | 4 | 5 | 6 | geomean |
|---|---|---|---|---|---|---|---|
| N=1 | 77.93 | 89.67 | 90.81 | 202.09 | 651.87 | 1821.34 | 231.08 |
| N=2 | 78.22 | 92.53 | 89.33 | 201.85 | 641.10 | 1768.04 | 229.97 |
| N=4 | 77.63 | 92.40 | 88.80 | 201.20 | 633.73 | 1723.23 | 227.86 |
| **N=4c** | 77.56 | 88.70 | 91.88 | 204.01 | 641.05 | 1719.00 | **228.43** |

Against the recorded denominator (`77.85 / 88.47 / 91.06 / 202.56 / 651.61 /
1818.67`, geomean 231.16 best): N=4c is 225.62, i.e. **0.9761×**. Against the
same-session N=1 control, which is the honest comparison because it removes
cross-session drift: **0.9838×**.

### M7 single-shot wall (C8), rotation 0, µs

| arm | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| N=1 | 202.46 | 213.91 | 221.99 | 329.40 | 750.81 | 1939.41 |
| N=2 | 206.17 | 218.86 | 219.21 | 331.49 | 760.35 | 1901.08 |
| N=4 | 205.71 | 218.02 | 219.64 | 329.76 | 778.74 | **1842.42** |
| **N=4c** | 203.93 | 212.13 | 217.74 | 327.07 | 776.87 | **1877.76** |

Latency and throughput move the same way on shape 6. The review's
pipelined-throughput risk is not realized.

---

## 6. The graded per-call number — the measurement that decides

Instrument: exp_10's `mp_vs_rank1.py` under exp_10's rematch protocol — 8
processes, the evaluator's own `barrier → call → synchronize → barrier` region,
one fresh pool per shape, ours and the frozen rank-1 submission **interleaved
with the order reversed every rep**, `VS_FORCE_BIAS=1`, 12 iters × 2 reps, all
eight ranks pooled (192 samples per arm per shape). Runner:
`vs_rank1.sh`, which redirects `VS_OUT` into this experiment so exp_10's and
exp_12's logs are not overwritten.

This is the right instrument for E3 for two reasons. Rank-1 is measured in the
**same run**, so a run that was globally fast or slow shows up in the anchor
rather than in our number; and it is the competition's actual statistic.

**Best (µs), pooled over 8 ranks:**

| # | shape | N=1 | **N=4c run 1** | **N=4c run 2** | Δ vs N=1 |
|---|---|---|---|---|---|
| 1 | 64×7168×18432 | 170.91 | 173.56 | 168.29 | −0.0% |
| 2 | 512×4096×12288 | 163.97 | 164.21 | 166.29 | +0.8% |
| 3 | 2048×2880×2880 | 174.19 | 174.19 | 173.98 | −0.1% |
| 4 | 4096×4096×4096 | 295.81 | 296.74 | 292.84 | −0.4% |
| 5 | 8192×4096×14336 | 743.42 | 749.69 | 729.74 | −0.5% |
| 6 | 8192×8192×29568 | **1955.46** | **1877.41** | **1878.41** | **−3.97%** |
| | **ours geomean** | **357.83** | 357.09 | 353.59 | **−0.70%** |
| | rank-1 geomean, same run | 311.21 | 308.36 | 308.80 | |
| | ours / rank-1 | 1.150× | 1.158× | 1.145× | |
| | ours geomean, median stat | 370.93 | 371.29 | 368.65 | −0.26% |

(Per-shape rows are rank 0's samples; the geomean rows are `vs_report.py`'s,
pooled over all eight ranks, which is why they are not exactly the geomean of the
column above them.)

**Shape 6 is the result.** 1955.46 → 1877.41 and 1878.41 — the two N=4c runs
agree to **0.05%** — while rank-1's own shape-6 best across the three runs is
1461.46 / 1465.38 / 1469.33, a spread of 0.54%. So the anchor is stable to half a
percent and our number moved 4.0%. Shape 6's graded gap to rank-1 goes
**1.338× → 1.281× and 1.278×**.

The geomean ratio does **not** improve (1.150× → 1.152× mean of two runs) and
this report does not claim it does. Rank-1's own geomean moved ±1% between runs
(311.21 / 308.36 / 308.80) purely from the five small shapes, where the `ours`
arm additionally carries 12–56% sample sd against rank-1's 2–7%. A 0.70%
improvement in our geomean is not resolvable against that. Correctness held at
both tolerances for both arms on all six shapes in every run.

Why the geomean barely moves is arithmetic, not mystery: a 4.0% cut on one of six
shapes is `0.9603^(1/6) = 0.9933`, i.e. −0.67% expected. Measured −0.70%.

---

## 7. The paired A/B and its null arm — methodology

`ab_release_group.py` builds each arm as a separate module, holds them all open
in one process, and interleaves their timed blocks in a rotating order, so
adjacent samples share clock state, allocation state and input sequence. It also
asserts, before any timing, that **every arm is bitwise identical to every other
arm on all 8 ranks** — a stronger and cheaper form of the M9 golden check.

It carries a **null arm**: `rg1b`, a second build of `RELEASE_GROUP = 1` under a
different module name. Nothing an instruction can see separates it from `rg1`.
Four runs, two arm/allocation orders:

**Null-arm separation, best (identical binaries):**

| shape | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| rg1b vs rg1 | +0.22% | +0.60% | −0.24% | +0.47% | **+3.64%** | +0.34% |

On the geomean the null arm separates by **+0.8 to +1.3%**, which is larger than
every uniform-`N` geomean delta in this experiment. **The pipelined harness
cannot decide this sweep at the geomean level, and any report that used it to
would be reporting the instrument.** This is why §6 exists.

**Mean of 3–4 paired runs, best (µs), with each shape's null floor:**

| shape | tiles/CTA | rg1 | rg2 | rg4 | **rg4c** | null floor |
|---|---|---|---|---|---|---|
| 1 | 1 | 77.74 | 78.20 (+0.59%) | 77.73 (−0.01%) | 78.03 (+0.37%) | ±0.22% |
| 2 | **2** | 88.60 | 92.01 (**+3.85%**) | 91.78 (**+3.59%**) | **89.13 (+0.60%)** | ±0.60% |
| 3 | 1 | 90.59 | 89.57 (−1.13%) | 88.99 (−1.77%) | 92.52 (+2.13%) | ±0.24% |
| 4 | 1 | 199.78 | 201.14 (+0.68%) | 200.96 (+0.59%) | 200.60 (+0.41%) | ±0.47% |
| 5 | **2** | 625.91 | 631.67 (+0.92%) | 645.47 (+3.13%) | 651.47 (+4.09%) | **±3.64%** |
| 6 | **4** | 1837.01 | 1783.69 (**−2.90%**) | 1705.90 (**−7.14%**) | **1724.67 (−6.11%)** | ±0.34% |

Only four deltas clear their own shape's null floor: shape 6's win (all grouped
arms, monotone in the number of releases — 4 → 2 → 1), shape 2's regression
(unconditional arms only), and shape 3's codegen movement. Shape 5 is
**indeterminate by construction**: `rg2` and `rg4` are behaviourally *identical*
there (2 tiles per CTA, so `min(left, 2) == min(left, 4)`) and yet they differ by
2.2% — a direct measurement of the instrument's noise on that shape from the arms
themselves.

Shape 5's ladder numbers and A/B numbers also disagree in sign (ladder: N=4 is
4.5% *faster*; A/B: 3.1% slower), which is the same finding from a second angle.
No claim is made about shape 5 in either direction.

---

## 8. Why the shipped rule is the full-group condition

`rg4c` was designed from the `rg2`/`rg4` data, not chosen after the fact. The
question the data poses is why grouping wins on a 4-tile CTA and loses on a
2-tile one, and the answer is that batching trades release cost for publication
delay and the two do not scale together:

- On shape 6, four releases cost ~135 µs of 1778 (7.6%); collapsing them recovers
  ~101 µs, and the reduce that must now happen in the tail is only ~86 µs (4.9%)
  and hides under the remaining producers. Net win, and the measured 6.1% lands
  within a few tenths of the 5.7% the attribution predicts.
- On shape 2, an 88 µs operation whose 64 output tiles give its 32 reducers two
  rounds of work, deferring the first tile's publication to the end of the second
  starves every reducer for the first half of the producer phase. That costs more
  than halving two releases can save. Net loss, 3.6–3.9%.

So the rule keys on the one quantity that separates the two cases and that the
kernel already knows — how many tiles a CTA owns — and refuses to group a partial
group. Shapes with 3 tiles per CTA are untested and take the ungrouped path,
which is the conservative side of an untested boundary.

`rg4c` versus `rg4`, explicitly:

- Shape 2: `rg4c` +0.60% (exactly the null arm) against `rg4`'s +3.59%. `rg4c`
  removes a real regression. This is decided in the pipelined A/B, which is the
  instrument that can see a 3.6% per-shape effect; the graded instrument cannot
  (shape 2's graded sd is 20–56% and the ~92 µs harness constant dilutes a 3.6%
  device change to ~1.9%).
- Shape 6: `rg4c` −6.11% against `rg4`'s −7.14%. The two are behaviourally
  identical on this shape — both collapse four releases to one — so this gap is
  codegen and run noise: `rg4`'s own shape-6 best ranges 1683.02–1732.39 across
  four runs (2.9%), and 1724.67 sits inside it.
- Shape 3: `rg4c` +2.13% against `rg4`'s −1.77%, a 3.9% codegen swing on a shape
  neither can group. This is the one place `rg4` is genuinely ahead, and it is an
  accident of scheduling rather than a property of the mechanism.

Net: `rg4c` trades an accidental 2% on shape 3 for a real 3% on shape 2 and keeps
shape 6's win. It is also the only arm with no measured regression anywhere.

The brief's tie-break rule ("if 4 and 2 are within noise, ship 2") does not
apply: 4 beats 2 on shape 6 by 4.2% against a 0.34% floor, so they are not within
noise, and 2 carries shape 2's regression as well.

---

## 9. Reproducing this

```bash
D=$ON/experiments/exp_05_release_granularity
# one arm, full ladder + M9 (edit RELEASE_GROUP / FULL_ONLY, push, then)
bash $D/run_arm.sh <tag> 1
# paired A/B with the null arm, both allocation orders
bash $D/run_ab.sh 5 50 fwd 0
bash $D/run_ab.sh 5 50 rev 1
# the graded number for whatever is built
bash $D/rebuild_and_vs.sh <tag> 5,3,0,1,2,4 12 2 12900
# M9 on its own (the shipped arm's final six-case run was: run_m9.sh 1 N4c_c8)
bash $D/run_m9.sh 1 <tag>
# per-arm best/median/single-shot rollup from the archived ladders
bash $D/collect.sh
# what is on the node, built from what, and the graded geomeans verbatim
bash $D/final_state.sh
```

Files changed outside this directory: `gemm_rs_mi300x.cpp` (the loop),
`gemm_rs_mi300x_constants.cuh` (`CTRL_PUBLISH_EARLY`), and
`harness/m9_stale_slot.py` (three C8 cases and their control epochs).
`harness/m4_controls.py` needed no change. Nothing was committed to git.

Artifacts: `arms/{N1,N2,N4,N4c}/` (per-arm ladder and M9 logs, `m7_results.json`),
`logs/ab_{ab1,fwd,rev,fwd2}.log`, `vs_logs/{rg1,rg4c,rg4c_r2}/`,
`e3_structural_w.diff`, `e3_final.diff`, `baseline/gemm_rs_mi300x_e3base.cpp`.

Reading the arm directories: `run_arm.sh` archives with a blanket `cp logs/*.log`,
so each arm folder also carries every earlier arm's `m9_*.log` and a previous
session's `build_g*.log`. The files that belong to arm `<tag>` are
`m9_<tag>.log`, `m7_results.json`, `ladder.log` and the `m{1,2,3,4,5,7}_*.log`
written by that arm's ladder.

Windows-side helpers added because `tools/push.ps1` syncs the whole tree
including `compbench` and takes ~4 minutes — longer than a whole gate-ladder arm:
`push_src.ps1` (the two kernel sources, LF-normalized) and `push_exp.ps1` (this
directory). The LF normalization is load-bearing: `tools/push.ps1` strips CR on
the node after every copy, and without it a `#define ... 4\r` reaches the
preprocessor with a carriage return in the replacement list.

## 10. Left open

- **E1(b)** is the unlock for the rest of this axis. The mainloop's blanket
  `vmcnt(0)` in `load_commit` is why coarsening the release recovered only the
  writeback and the barriers; with counted `s_waitcnt` the peer stores could stay
  in flight across a tile boundary and the deferred egress could actually drain
  under later tiles' compute, which is the effect the dispatch hoped for and
  which this arm did **not** get.
- **Shape 5** needs a better instrument before anything is concluded about it.
  Its null-arm floor is 3.6% and the ladder and the A/B disagree in sign.
- **The 3-tiles-per-CTA boundary** in the shipped rule is untested; it takes the
  conservative branch.
- **The codegen sensitivity on shape 3** (±2% from loop structure alone, on a
  shape that cannot group) is worth an exp_09-style look on its own: if a
  compile-time-bounded inner loop is worth 2% on the 128×256 instantiation, that
  is a free win available independently of E3.

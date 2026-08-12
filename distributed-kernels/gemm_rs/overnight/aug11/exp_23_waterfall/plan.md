# exp_23 — the knob waterfall (paper Fig 4 / Q1, the money figure)

GEMM-RS column of the paper's Q1 ladder: *how much does each scheduling
decision contribute?* PAPER.md fixes the rung ORDER (homogeneous →
+epilogue-carried payload → +injection bound → +arrival-sized signals →
+task reorder) and requires both operators to present the same ladder shape.

This kernel already ships the epilogue-carried payload unconditionally — the
producer emits 16 B peer packets from the GEMM epilogue and there is no build
in which it does not — so the GEMM-RS instance of the ladder starts at the
aug10 configuration and walks the two knobs that ARE revertible by a `-D`
flag, then adds the placement-flatness exhibit that answers Q2.

**No kernel source is edited.** Every rung is a compile-flag revert of a
shipped change, built into this experiment's own `build/` directory. That is
the whole methodological point: the arms differ by one macro each, and the
fingerprint gate (below) proves it.

---

## The rungs

| rung | `WGM4` | `RELEASE_GROUP` | `RELEASE_GROUP_FULL_ONLY` | meaning |
|---|---|---|---|---|
| **a** | 1 | 1 | 1 | aug10 baseline: donor `WGM = 4` tile order + per-tile release |
| **b** | 0 | 1 | 1 | + WGM destination-spreading task order (exp_08) |
| **c** | 0 | 4 | 1 | + grouped release (exp_05 / E3) — the current shipped best |
| **null** | 0 | 4 | 1 | byte-identical to **c**, different module name: the allocation-noise floor |
| **d** | rung **c** binary, `num_reducer_ctas` ∈ {8, 16, 32, 48} | | | the placement-flatness exhibit (Q2) |

`FULL_ONLY` is held at its shipped value 1 in every rung so it cannot act as
a hidden second variable. `NEGATIVE_CONTROLS` stays 0 and `TILE_SWEEP` is not
passed at all.

### Macro semantics, confirmed against source (not taken on trust)

- `gemm_rs_mi300x.cpp:47-49` — `HK_GEMM_RS_MI300X_WGM4` defaults to **0**.
- `gemm_rs_mi300x.cpp:315-319`:

```315:319:distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp
#if HK_GEMM_RS_MI300X_WGM4
        const int WGM = 4;
#else
        const int WGM = (tiles <= g.num_gemm_ctas) ? 4 : num_pid_m;
#endif
```

  So **the knob is inverted relative to its name**: `=0` (the default) is the
  shipped current best — the column-major decode that spreads a round's tiles
  over all eight egress destinations — and `=1` is the donor control that
  restores the pre-WGM aug10 order. Rung (a) therefore carries `=1`.
  The comment at `:293-314` states the mechanism: a producer CTA's whole tile
  lands on ONE peer (`dest = (tm*BM)/(M/8)`), so the set of `tm` values live
  across concurrently resident CTAs *is* the set of egress links in use.
- `gemm_rs_mi300x.cpp:62-64` — `HK_GEMM_RS_MI300X_RELEASE_GROUP` defaults to
  **4**; `:324-328` reads it into a `constexpr` guarded by
  `static_assert(RELEASE_GROUP >= 1 && RELEASE_GROUP <= 8)`.
- `gemm_rs_mi300x.cpp:87-89` — `RELEASE_GROUP_FULL_ONLY` defaults to **1**;
  `:335-340` — `tiles_per_cta = ceil(tiles / num_gemm_ctas)` and
  `rgroup = tiles_per_cta >= RELEASE_GROUP ? RELEASE_GROUP : 1`.
- `gemm_rs_mi300x.cpp:97-99` — `TILE_SWEEP` defaults to 0. Its consumer rows
  (101-107) live in the dispatch table; `harness/build.sh` never passes it and
  neither does this experiment. Treated as out of scope, not as dead: it is
  live for `exp_14`'s screening module only.

The dispatch's summary of all four macros was correct.

### Rung (d) does not rebuild anything

`num_reducer_ctas` is a host scalar. `harness_lib.GemmRS(..., num_reducer_ctas=N)`
routes to `rt.resolve_shape_with_split(...)` instead of `rt.resolve_shape(...)`,
so the NR sweep runs on the rung-(c) binary. The shipped per-shape NR table
(`gemm_rs_mi300x_host_abi.hpp:109-116`) is **56 / 32 / 32 / 32 / 32 / 48** —
rows 1 and 6 are not 32, superseding the root charter's "uniform NR=32" note.
That table is not edited.

---

## Pre-registered predictions

### The structural prediction that comes first, because it decides how to read everything else

Both knobs are *conditionally* active, and the conditions are computable from
the shape plan before any GPU runs. `WGM4` only changes behaviour when
`tiles > num_gemm_ctas`; grouped release only changes behaviour when
`ceil(tiles / num_gemm_ctas) >= 4`.

| shape | BM/BN | NR | NG | tiles | tiles/CTA | a→b active? | b→c active? |
|---|---|---|---|---|---|---|---|
| 1 · 64×7168×18432 | 32/64 | 56 | 248 | 2×112 = 224 | 1 | **no** | **no** |
| 2 · 512×4096×12288 | 64/128 | 32 | 272 | 8×32 = 256 | 1 | **no** | **no** |
| 3 · 2048×2880×2880 | 128/192 | 32 | 272 | 16×15 = 240 | 1 | **no** | **no** |
| 4 · 4096×4096×4096 | 256/256 | 32 | 272 | 16×16 = 256 | 1 | **no** | **no** |
| 5 · 8192×4096×14336 | 256/256 | 32 | 272 | 32×16 = 512 | 2 | **yes** | **no** |
| 6 · 8192×8192×29568 | 256/256 | 48 | 256 | 32×32 = 1024 | 4 | **yes** | **yes** |

So the honest shape of this figure is: **the a→b rung moves shapes 5 and 6
only; the b→c rung moves shape 6 only; the geomean moves entirely through
those two shapes.** This is pre-registered rather than discovered, and it
doubles as an instrument check — shapes 1-4 are four independent null pairs
built out of genuinely different binaries, so if any of them shows a delta
beyond the measured allocation floor, the instrument is lying and no rung
verdict may be reported.

### Rung deltas

- **a → b (task order).** Shape 6: **−27.9%** (exp_08 measured; effective
  xGMI links 2.02 → 7.53 of 8). Shape 5: gain expected but smaller — it is
  two waves, not four, so fewer rounds are mis-spread. Shapes 1-4: inside the
  null floor by construction.
- **b → c (signal granularity).** Shape 6: **−4.0% graded / −5.96% pipelined**
  (exp_05, paired, against a 0.10% null floor in that experiment). Shapes 1-5:
  inside the null floor by construction.
- **d, the NR sweep.** **Flat** across {16, 32, 48} on every shape — within
  ~2× the measured null floor. This is the Q2 exhibit: dedicating more CTAs to
  communication does not win. NR = 8 is the one point allowed to degrade, and
  only on shape 6, where `red_tiles = 128` makes the reduce take 16 rounds at
  NR = 8 against 3 at NR = 48; if it does degrade, that is a statement about
  the reduce being owner-bound, not about communication deserving its own
  CTAs.

### Falsifiers

| prediction | falsifier | what it would mean |
|---|---|---|
| shapes 1-4 flat across a/b/c | any of them moves beyond its null floor | the arms differ by something other than the macro, or the instrument is not paired — **report as a blocker, do not publish a rung** |
| shape 6 a→b ≈ −28% | \|delta\| < 10% | contradicts exp_08 on the same node; suspect a stale build (the fingerprint gate should have caught it) or a changed denominator |
| shape 6 b→c ≈ −4% | delta inside the pooled null floor | **live risk**: the published shape-6 floor is 4.28% and the re-measurement 2-2.6× worse. Then the granularity rung is *unresolved at this shape by this instrument* and must be reported as unresolved, not as a win. Mitigation: multiple allocation draws + the paired rank-sum test, which cancels the shared per-draw bias |
| NR curve flat | a monotone trend beyond floor across {16,32,48} | the producer/reducer boundary is NOT settled and E4 must reopen |
| NR = 8 degrades on shape 6 | it does not | the reduce is cheaper than the round model says; re-rank against exp_20's refreshed attribution |

---

## Why the null arm is not optional

`HANDOFF.md`: the harness bias on this node is **per-allocation and partly
allocation-ORDER**, not positional. Two arms with identical code, identical
config and shared operands measured **4.28% apart on shape 6** while the
positional residual stayed under ±0.47%. Published floors are
`1.34 / 0.56 / 0.61 / 2.41 / 2.17 / 4.28 %` for shapes 1-6 and a
re-measurement gave `3.48 / 0.93 / 1.61 %` on shapes 1-3 — 2-2.6× worse — so
the published set is a **lower bound**, never a threshold to clear.

Two consequences this plan adopts:

1. Every plotted delta is reported beside the floor measured **in the same
   draws** (rung **c** vs rung **null**), and any delta inside it is reported
   as no evidence.
2. Half the draws reverse the order in which arms are **constructed**, since
   part of the bias is construction order: `exp_14` saw a null twin come out
   negative in 10 of 10 forward draws on shape 1 and change sign when
   construction order was reversed.

Means are not reported. Best (min) and median only.

---

## Gate posture

Every rung passes correctness at **both** `1e-2` (graded) and `2e-3` (tight)
inside the sweep process before any of its timing samples are kept, plus error
bits, epoch cells and signal cells after timing. A rung that only clears
`1e-2` is a regression and is marked dead rather than plotted.

# exp_04a — result: shape 1 retiled `32/256/32` → `32/64/64`. **WIN, −20.3% on shape 1.**

## Verdict

Adopted. Geomean **280.74 → 269.96 µs (−3.8%)**, entirely from shape 1
(**107.86 → 85.93 µs, −20.3%**). Every other shape moved within noise, which is
what the change predicted since it touches only `config_row 1`.

Cumulative for the night so far: **285.02 → 269.96 µs, −5.3%.**

## The change

Two lines, no new instantiation — `<32,64,64,false>` already existed as the
generic fallback row.

- `gemm_rs_mi300x_host_abi.hpp`, `scored_shapes[0]`: `{32, 256, 32, 32, 1}` →
  `{32, 64, 64, 32, 1}`
- `gemm_rs_mi300x.cpp`, `dispatch_gemm_rs_mi300x`: `case 1:
  launch_fixed<32,256,32,false>` → `launch_fixed<32,64,64,false>`

Derived geometry, all re-validated by `resolve_shape`'s existing gates:

| | before | after |
|---|---|---|
| gemm_tiles = (M/BM)·⌈N/BN⌉ | 2×28 = **56** | 2×112 = **224** |
| tiles / 272 producer CTAs | **0.21 waves** | **0.82 waves** |
| k_iters = K_local/BK | 2304/32 = **72** | 2304/64 = **36** |
| LDS = 2(BM+BN)·BK·2 | 36864 B | 24576 B |
| CTAs/CU (LDS-bound) | 1 | 2 |
| even_k | true | true (2304 % 64 == 0) |
| eb = gcd(BM, M/8) | 8 | 8 |
| col_count ≤ COL_MAX=128 | 28 | 112 |

## Gates

| gate | result |
|---|---|
| M1 build | PASS, three modules |
| M2 resources/ISA | PASS |
| M3 correctness, 17 shapes, `1e-2` **and** `2e-3` | **PASS 17/17** |
| M4 negative controls | **PASS**, all three fail as designed |
| M5 600-epoch skewed soak | **PASS**, epochs and signals exact |
| M7 timing | see below |

## Timing (3 rotations × 50, pinned clocks, verified-clean node)

| # | shape | after NR=32 | **after retile** | delta |
|---|---|---|---|---|
| 1 | 64×7168×18432 | 107.86 | **85.93** ±0.17 | **−20.3%** |
| 2 | 512×4096×12288 | 113.44 | 113.79 ±0.31 | +0.3% |
| 3 | 2048×2880×2880 | 96.80 | 96.92 ±0.26 | +0.1% |
| 4 | 4096×4096×4096 | 204.00 | 201.82 ±0.75 | −1.1% |
| 5 | 8192×4096×14336 | 769.45 | 769.04 ±3.32 | −0.1% |
| 6 | 8192×8192×29568 | 2633.04 | 2632.09 ±20.48 | −0.0% |
| | **geomean** | **280.74** | **269.96** | **−3.8%** |

Shape 1's `device_max` fell 106.25 → 84.43 µs and its ×SOL from 16.70 to 13.30.

## What this does and does not establish

The experiment moved **two** variables at once — CTA occupancy (56 → 224 tiles)
and k-loop length (72 → 36 iterations) — so it does not attribute the 22 µs
between them. Both were expected to help and both did something; a follow-up
(`32/128/64`, which halves k_iters but only doubles tiles, against `32/64/32`,
which quadruples tiles at unchanged k_iters) would separate them. That
follow-up is **not** worth GPU time right now: the E1(b) mainloop work targets
the same per-iteration cost far more directly, and if it lands, the attribution
question changes shape anyway.

## Why the obvious follow-up on shape 3 was NOT run — a real dependency

Shape 3 (2048×2880×2880) looked like the next candidate: `⌈2880/256⌉ = 12`
against `2880/256 = 11.25`, so one column-tile in twelve is 25% padding, and at
192 tiles it uses only 0.71 of the 272 producer CTAs. Retiling to `BN=64` gives
`2880/64 = 45` exactly and 720 tiles.

**The arithmetic kills it.** `producer_drain_release` is issued once per tile,
and shape 3 already spends **12.5 µs** there at 192 tiles. 720 tiles is 3.75×
the releases, so about **+35 µs** — against a ragged-N GEMM saving of roughly
1.5 µs, because shape 3's GEMM mainloop is only 23.6 µs of its 96.9 µs total
(its mass is in egress, 32.6 µs, and sync, 21.4 µs). Net expected: a large
loss.

Shape 1 tolerated 4× the tiles only because its release cost was **0.4 µs**.

**Therefore: E3 (release granularity) gates any further tile refinement.** Per-
tile release is a fixed per-tile tax that makes finer tiling uneconomic
everywhere except shape 1, where it happened to be negligible. Amortizing the
release across tiles does not just recover its own 9-14%; it unlocks an axis
that is currently closed. Run E3 before revisiting E4.

## Remaining tile-table headroom: small

Recomputing the geometry for the other five rows, none has shape 1's defect:

| # | tiles | tiles/272 | waves (⌈⌉) | last-wave fill | verdict |
|---|---|---|---|---|---|
| 2 | 512 | 1.88 | 2 | 94% | fine; also the only row at 2 CTAs/CU — raising BK to 128 would hit the 64 KB LDS cap and halve that residency. Leave it. |
| 3 | 192 | 0.71 | 1 | 71% | ragged, but release-tax-bound (above) |
| 4 | 256 | 0.94 | 1 | 94% | fine |
| 5 | 512 | 1.88 | 2 | 94% | fine |
| 6 | 1024 | 3.76 | 4 | 94% | fine |

Shape 1 at 0.21 waves was a genuine outlier, not the first of a series. **The
tile-table axis is close to exhausted**; the remaining mass is in the mainloop
schedule (E1b), egress (E2) and the release tax (E3).

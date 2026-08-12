# exp_14 / E4b — re-sweep the tile table (`BM/BN/BK`) under the round model

**Predictions in this file are pre-registered. Nothing below was written or
edited after a GPU number existed.** `enumerate.py` (pure host arithmetic) is the
instrument that produced the geometry; its output is quoted verbatim in
`result.md`.

## 1. The model, corrected on one point that matters

E4's insight — *a strided persistent loop costs `ceil(tiles/count)` rounds, not
CTAs* — carries over to the tile dimension, but with a correction the brief's
framing omits: **moving `BM/BN/BK` changes what one tile body costs, so `waves`
alone is not a time.** Waves must be multiplied by a per-tile cost, and there are
three candidates for that cost, each binding in a different regime:

| column | formula | regime | note |
|---|---|---|---|
| `wave_cost` | `waves × BM·BN` | MFMA-bound | equals `(M·N/NG)/avg_fill`, so this **is** the brief's fill criterion, re-derived as a time. Floor `M·N/NG` at perfect fill. |
| `iter_cost` | `waves × k_iters` | overhead-bound | every k-iteration pays a full `__syncthreads()`, a `vmcnt(0)`, an `lgkmcnt(0)` and an LDS round trip *whatever the tile size*. |
| `traffic` | `1/BM + 1/BN` | bandwidth-bound | global bytes are `M·N·K_local·2·(1/BM + 1/BN)` before L2 reuse. Buying fill by shrinking a tile pays for it here. |

**Which one dominates today is measurable, and it is not the first one.** Shape 6
spends 1142.8 µs of GEMM over `waves × k_iters = 4 × 115.5 = 462` iterations =
~4700 cycles/iteration at 1.9 GHz, against `BM·BN·BK/1024 = 2048` cycles of MFMA
per iteration. So **~57% of every k-iteration is not MFMA**, and `iter_cost` is
the larger half. A model that counts only fill will systematically undervalue
raising `BK`, which is the one knob that reduces `iter_cost` at constant
`wave_cost` and constant total traffic.

That reframing is the whole reason this sweep has candidates at all: on
`wave_cost` alone, four of the six rows are already at or within 6% of their
floor and the sweep would be empty.

### A second correction: "last-wave fill" is not waste on a 1-wave row

On a row with `waves = 1` every tile is resident simultaneously, so the critical
path is **one** tile body and the idle CTAs cost nothing directly. What a low
fill means there is that `BM·BN` is larger than it needs to be, and *that* is
what costs time. The two readings coincide numerically (`wave_cost` ∝ `1/fill`),
but only the second one explains why shape 3 at 71% fill is worth attacking and
shape 3 at 53% fill (`128/320/32`, also 1 wave) would be worse, not better.

## 2. Constraints, verified against source

Read out of the source, not the brief. Line numbers are pre-edit.

| constraint | source |
|---|---|
| `BM % 32 == 0`, `BN % 64 == 0` | `gemm_rs_mi300x.cpp:204` (`WARPS_M=2`, `WARPS_N=4`, 16-wide frags) |
| `LDS = 2(BM+BN)·BK·2 ≤ 65536` | `gemm_rs_mi300x.cpp:208-209`, hard `static_assert` |
| `32·BN·2 ≤ LDS` | `gemm_rs_mi300x.cpp:215-216` |
| **`BK % 32 == 0`** | `gemm_rs_mi300x.cpp:255-257`: `KH=2`, `KS=BK/2`, `static_assert(KS % 16 == 0)`. The brief did not list this one. |
| **`BM·BK ≤ 65536`, `BN·BK ≤ 65536`** | `hk_adapter.cuh:114-121`: `static_assert(big_calls == 1)`. Not in the brief; implied by the LDS cap anyway. |
| `BM \| M` | `host_abi.hpp:169` |
| `lrow_count = (M/8)/gcd(BM, M/8) ≤ 32`, `col_count = ceil(N/BN) ≤ 128` | `host_abi.hpp:174`, `constants.cuh:24-25` |
| `K_TAIL = (K_local % BK != 0)` selects the instantiation | `host_abi.hpp:195`, dispatch switch |

Two more that are not `static_assert`s but bound the space just as hard:

- **Accumulator VGPRs = `BM·BN/512`.** `rt_fl<BM/2, BN/4>` is `BM·BN/512`
  registers per lane; the `256×256` rows are at 128 of a 256 arch-VGPR budget and
  measured 246 in use. `BM·BN > 65536` is therefore unusable, which caps
  `BM·BN ≤ 65536` — and this is what makes 1 wave impossible on shapes 5 and 6.
- **Occupancy is irrelevant.** The grid is exactly `CU_COUNT = 304` CTAs on 304
  CUs, so LDS never buys a second resident CTA and exists purely as a cap. (This
  retires exp_04's note that shape 2 "is the only row at 2 CTAs/CU" and must not
  lose it — there is nothing to lose.)

## 3. Enumeration result: where the headroom is, and is not

`enumerate.py` walks every legal `(BM, BN, BK)` per shape and ranks by
`sqrt(rel_wave × rel_iter)` with `traffic` reported separately so a composite
cannot hide it.

**Three of six rows are already the argmin of the composite and have no candidate
at all:**

| # | shape | landed | why nothing beats it |
|---|---|---|---|
| 4 | 4096×4096×4096 | 256/256/32 | ranks **1st of 38** legal points. `wave_cost` 65536 vs floor 61680 (6% off) and no legal `BM·BN` lands between. |
| 5 | 8192×4096×14336 | 256/256/32 | ranks **1st of 38**. 1 wave needs `BM·BN ≥ 123362` = 241 accumulator VGPRs — impossible. Next best is +30%. |
| 6 | 8192×8192×29568 | 256/256/32 | ranks **1st of 38** and sits **exactly at the `wave_cost` floor** (1024 tiles = 4 × 256 NG, 100% fill). |

So the brief's "shapes 2, 3 and 5 waste 12-29% of their last producer wave" is
right about 2 and 3 and **wrong about 5**: shape 5's 88% last-wave fill is not
recoverable, because every legal way to raise the fill either exceeds the
accumulator budget or costs more waves than it saves.

Also worth recording, because it points at the next experiment rather than this
one: **shapes 4/5/6 are pinned against the LDS cap.** `BM+BN = 512` at `BK = 32`
is `16384` elements, exactly the cap, so `BK` cannot rise on the three largest
rows without shrinking the tile. Their `iter_cost` — the larger half of the
mainloop — is therefore untouchable from the tile table. Unlocking it needs the
*double* buffer to stop costing 2×, which is E1 work, not E4b work.

## 4. Candidates and pre-registered predictions

Six arms across three shapes. Every arm needs one new instantiation; all six are
compiled **only** into a separate `gemm_rs_mi300x_tilesweep` module behind
`HK_GEMM_RS_MI300X_TILE_SWEEP`, so the production binding still has exactly the
six distinct instantiations `M2_EXPECT` asserts.

### Shape 1 — 64×7168×18432, landed `32/64/64`, NG=248

224 tiles, 1 wave, 90% fill, `k_iters = 36`. `wave_cost` is already 1.00 of its
own floor family and cannot improve (`BM ∈ {32,64}` only, since `BM | 64`). The
entire opportunity is `iter_cost`: at `BM·BN = 2048` the MFMA is **128 cycles per
iteration**, so this row is almost purely overhead-bound and its 36 iterations
are 36 barriers deep for nothing.

| arm | geometry | `w·x` | `w·ki` | traffic | prediction |
|---|---|---|---|---|---|
| **A1 `32/64/128`** | 224 tiles, 1 wave, 90%, `k_iters=18`, LDS 48K, even_k | 1.00 | **0.50** | 1.00 | **WIN, −10% to −25%.** Halves every per-iteration fixed cost at identical tiles, identical fill, identical total bytes and identical reduce geometry (`cols` unchanged at 112, so the landed `NR=56` optimum is undisturbed). |
| A2 `32/64/160` | 224 tiles, 1 wave, 90%, `k_iters=15`, LDS 60K, **K_TAIL** | 1.00 | **0.42** | 1.00 | win, and *slightly* better than A1 if the model is right — but it pays the tail-mask path on the last iteration and a non-power-of-2 `BK` through the HK shared-tile swizzle. **Ranked below A1 on risk, not on model.** |

### Shape 2 — 512×4096×12288, landed `64/64/64`, NG=272

512 tiles, 2 waves, 88% last-wave fill, `k_iters = 24`, so 48 iterations deep.
Three legal points tie at the top of the composite; two of them are worth
measuring because **they disagree about which model is right**, and that is more
valuable than either microsecond.

| arm | geometry | `w·x` | `w·ki` | traffic | prediction |
|---|---|---|---|---|---|
| **B1 `64/128/64`** | 256 tiles, **1 wave**, 94%, `k_iters=24`, LDS 48K | 1.00 | **0.50** | **0.75** | **WIN, and the predicted winner of the pair.** Same MFMA and same 24 iterations as B2, but 25% less global traffic, half the tiles (so half the credit waits, staging passes, releases and publications) and `red_tiles` 64→32 so reduce rounds fall 2→1 at the landed `NR=32`. |
| B2 `64/64/128` | 512 tiles, 2 waves, 88%, `k_iters=12`, LDS 64K (at the cap) | 1.00 | **0.50** | 1.00 | win vs landed, **loss vs B1**. Identical `iter_cost` and `wave_cost` to B1 by construction. **This pair is the discriminator: if B1 ≈ B2, then traffic and per-tile protocol overhead do not matter at this scale; if B1 < B2, they do.** |

### Shape 3 — 2048×2880×2880, landed `128/256/32`, NG=272

192 tiles, 1 wave, 71% fill, `k_iters = 12`. `2880/256 = 11.25`, so `col_count`
is 12 and the row is **ragged in N**: 6.7% of its MFMA work is padding, and the
last B tile's `load_issue` reads rows 2880..3071 of a 2880-row `w` — 192 rows =
138 KB **past the end of the tensor**. That read is benign today (the columns are
never emitted, `vt` clamps them) but it is an out-of-bounds global read, and
`BN=192` removes it.

| arm | geometry | `w·x` | `w·ki` | traffic | prediction |
|---|---|---|---|---|---|
| **C1 `128/192/32`** | 240 tiles, 1 wave, 88%, `k_iters=12`, LDS 40K, `2880/192 = 15` **exact** | **0.75** | 1.00 | 1.11 | **weak win, −0% to −3%, LOW confidence.** `iter_cost` does not move, so the win has to come entirely out of the MFMA term — and at `BM·BN=32768` the MFMA is 1024 cycles against ~2650 of per-iteration overhead, so a 25% MFMA cut is only a ~7% iteration cut. Shape 3's GEMM is ~26% of its 89.41 µs. Expected −1.5%, against a 0.61% null floor: **marginal by construction.** Also removes the OOB read and the 6.7% padding. |
| C2 `64/192/64` | 480 tiles, 2 waves, 76%, `k_iters=6`, LDS 64K | **0.75** | 1.00 | **1.78** | **predicted LOSS.** Identical `w·x` and `w·ki` to C1, 78% more traffic, 2× the tiles. **The second discriminator: C1 vs C2 isolates the traffic term at constant time-model.** If C2 ≥ C1 the traffic column is load-bearing and every "shrink the tile to buy fill" move in future sweeps is suspect. |

### Rows deliberately not swept

Shapes 4, 5 and 6 get **no arm**. Their landed tile is the enumerated argmin, and
shape 6's is provably at the floor. GPU time goes to null arms on 1/2/3 instead,
which is what actually limits the conclusions.

## 5. Instrument

`sweep.py`, same protocol as exp_13's (which established the two `NR` winners):
independently allocated arms, all live at once, **operands generated once per
shape and shared**, complete positional rotation, best and median only, and a
**null twin** — the landed configuration carried by two independent arms — whose
spread is the floor below which nothing is believed.

One difference from exp_13 and it is the load-bearing one: `NR` was a host
scalar, so one build served every arm. `BM/BN/BK` are template parameters, so the
sweep drives them through `config_row` 101-106 in the sweep module and
**overrides the resolved plan in Python**. That re-derivation is validated, not
trusted: `check_derivation()` asserts that deriving the *landed* tile through the
Python path reproduces `rt.resolve_shape`'s dict **field for field** on all six
shapes, and the sweep refuses to run if it does not. Every arm also asserts that
its `even_k` matches the `K_TAIL` its `config_row` compiled with.

## 6. Landing rule

Only a candidate whose gain exceeds that shape's measured null floor **on both
best and median**, and which survives pooled independent allocations with
disjoint ranges, is landed into `scored_shapes` + the real dispatch row. Then:
`M2_EXPECT` is updated to the new distinct-instantiation count, the full ladder
runs (M0→M7, 17 shapes at `1e-2` **and** `2e-3`), and `vs_rank1.sh` re-measures
the graded number.

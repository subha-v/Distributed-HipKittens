# exp_04 — the tile table is inherited too: BM/BN/BK per shape

**New axis, not in the original ranked table.** Motivated by exp_02: the
`num_reducer_ctas` column of `scored_shapes` was carried over verbatim from the
donor and cost 8.3% on the largest shape. The **BM/BN/BK columns of the same
table came from the same place** — the header says so in as many words:
"BM/BN/BK are the frozen gfx942 rank-1 tile choices". They have never been
swept against *our* kernel, whose mainloop, epilogue and CTA split are all
different from the donor's.

## Why this outranks most of the ranked table

The competition's ranking statistic is the **geometric mean over six shapes**,
so a 20% win on a 107 µs shape is worth exactly as much as a 20% win on the
2866 µs shape. The existing attribution was measured only on the largest shape,
where the GEMM mainloop dominates — but three of the six shapes finish in
under 210 µs and nobody has ever looked at where *their* time goes.

## Derived geometry (from `resolve_shape`, confirmed against the source)

`gemm_tiles = (M / BM) * ceil(N / BN)`, producers = `304 - NR` = **272** at
NR=32. `k_iters = K_local / BK`. LDS = `2*(BM+BN)*BK*2` bytes, hard cap 65536
(`static_assert` in `launch_fixed`). Fragment rules: `BM % 32 == 0`,
`BN % 64 == 0`, `BM | M`, `col_count <= COL_MAX = 128`.

| # | shape (M×N×K_g) | K_loc | BM/BN/BK | tiles | tiles/272 | k_iters | LDS | note |
|---|---|---|---|---|---|---|---|---|
| 1 | 64×7168×18432 | 2304 | 32/256/32 | **56** | **0.21** | **72** | 36 KB | 216 of 272 CTAs idle |
| 2 | 512×4096×12288 | 1536 | 64/64/64 | 512 | 1.88 | 24 | 32 KB | ok |
| 3 | 2048×2880×2880 | 360 | 128/256/32 | **192** | **0.71** | 11.25 | 48 KB | ragged N *and* ragged K |
| 4 | 4096×4096×4096 | 512 | 256/256/32 | 256 | 0.94 | 16 | 64 KB | LDS-maxed, spills |
| 5 | 8192×4096×14336 | 1792 | 256/256/32 | 512 | 1.88 | 56 | 64 KB | LDS-maxed, spills |
| 6 | 8192×8192×29568 | 3696 | 256/256/32 | 1024 | 3.76 | 115.5 | 64 KB | LDS-maxed, spills |

Two independent defects fall out of this table.

**(a) Shape 1 runs on 21% of the machine.** 56 tiles over 272 producer CTAs
means 216 CTAs execute the tile loop zero times. Nothing in the ranked
experiment list addresses this, because the attribution that produced that list
was taken on shape 6, where tiles/272 = 3.76 and occupancy is a non-issue.

**(b) Shape 3 is ragged in both directions.** `ceil(2880/256) = 12` against
`2880/256 = 11.25`, so one column-tile in twelve is 25% padding — about 6% of
the shape's GEMM thrown away — and `360/32 = 11.25` forces the `K_TAIL` masking
path on every tile.

## Per-k-iteration cost, measured

Dividing the measured GEMM mainloop time by `waves × k_iters` gives a
per-iteration cost that is consistent across the two shapes that share a tile
shape, which is what makes this a model and not a coincidence:

| shape | GEMM µs | waves | k_iters | µs per k-iter | ideal µs | ratio |
|---|---|---|---|---|---|---|
| 5 | 335.5 | 2 | 56 | 3.00 | 0.975 | 3.1× |
| 6 | 1317.8 | 4 | 115.5 | 2.85 | 0.975 | 2.9× |
| 1 | ≥45.2 | 1 | 72 | 0.63 | 0.122 | 5.2× |

(ideal = `2·BM·BN·BK / 4.3 TFLOP/s per CU`, 1.307 PFLOP/s ÷ 304 CU.)

The 256/256/32 rows sit at a stable **~2.9× off** single-CU peak, and shape 1 at
**5.2×** — a 32×256 tile has less arithmetic per iteration to hide the same
fixed per-iteration latency. Two readings, one conclusion: **a large fixed cost
per k-iteration** (the blocking global→LDS load plus the full `__syncthreads()`
the design notes describe), which is exactly what E1(b)/(c) attack directly and
what fewer, larger k-iterations attack for free.

## The interventions

### 4a — shape 1: `32/256/32` → `32/64/64`

Both defects at once, and it needs **no new instantiation**: `<32, 64, 64,
false>` already exists as the generic fallback row, so this is two lines (the
table entry and `case 1:` in `dispatch_gemm_rs_mi300x`).

- tiles 56 → **224** (0.21 → 0.82 waves): 4× the CTAs do work
- k_iters 72 → **36**: halves the fixed per-iteration cost on the critical path
- LDS 36 KB → 24 KB
- `K_local = 2304`, `2304 % 64 == 0`, so `even_k` holds and `K_TAIL` stays false
- `eb = gcd(32, 8) = 8` unchanged; `col_count = 112 <= 128` ✓; `BM=32 | M=64` ✓

Both effects push the same way, which is a weakness of the experiment — it will
not tell us *which* mechanism paid. That is acceptable for a first cut; 4c
separates them if 4a wins.

Alternatives if 4a disappoints: `32/64/128` (k_iters 18, LDS 48 KB) or
`32/128/64` (tiles 112, LDS 40 KB).

### 4b — shape 3: `128/256/32` → `128/64/32`

`2880/64 = 45` exactly, so the ragged-N padding disappears and
tiles = 16×45 = **720** (2.6 waves). LDS 48 KB → 24 KB. K stays ragged
(`360/32 = 11.25`), so `K_TAIL` remains true. `BN=192` (15 columns exactly,
240 tiles) is the conservative alternative if BN=64 costs too much B-reread.

### 4c — separating the two mechanisms of 4a

Only if 4a wins: run `32/256/64` — wait, that is LDS 73728 and illegal. Use
`32/128/64` (k_iters halved, tiles only 2×) against `32/64/32` (tiles 4×,
k_iters unchanged, LDS 18 KB) to attribute the win between occupancy and
iteration count.

## Risks

- Smaller BN means each tile re-reads more of B per unit of output. For shape 1
  the A operand is 64×2304×2 B = 288 KB and B is 7168×2304×2 B = 33 MB, so B
  traffic is what matters and BN=64 quadruples the number of times a given A
  block is loaded — but A is tiny and L2-resident, so the exposure is small.
  If 4a regresses, this is the first thing to suspect.
- More tiles means more `producer_drain_release` calls (one per tile). Shape 1
  spends 0.4 µs there today at 56 tiles; at 224 tiles expect ~1.6 µs. Noise
  against an expected double-digit win, but it is a real cost and E3 makes it
  cheaper.
- Nothing here touches the memory-ordering protocol, the descriptor ABI, or the
  device code. `BM/BN/BK` are template parameters and every derived count is
  recomputed by `resolve_shape` and re-validated by its existing gates.

## Gates and denominator

Full ladder per arm. Denominator is the running best after exp_02. Per-shape
means decide each row independently: **these are per-shape table entries, so a
row is adopted only if that shape improves**, and a regression on one row does
not veto another.

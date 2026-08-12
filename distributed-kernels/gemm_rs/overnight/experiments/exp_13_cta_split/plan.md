# exp_13 — E4 re-opened: where the CTA-level producer/consumer boundary sits

`NUM_REDUCER_CTAS` was swept once, on a kernel that no longer exists. Five
changes have moved the producer/consumer balance since (`NR=32` itself, shape 1
retiled to `32/64/64`, E1(b) mainloop overlap, exp_08 `WGM`, E3 release
grouping). Fresh attribution on shape 6 (total 1777.9 µs) reads GEMM 1142.8,
XGMI 412.4, sync 191.0, release 134.9, **reduce 86.5** — 86.5 µs of exposed
reduce work in exchange for 32 CTAs (10.5% of the machine) withheld from the
GEMM for the whole kernel.

## The instrument, and why it needs no rebuild

`num_gemm_ctas` is the **only** host-provided quantity the split affects
(`dhk_rt.cpp:197`). The device derives everything else from it —
`num_reducer_ctas = CU_COUNT - g.num_gemm_ctas` (`gemm_rs_mi300x.cpp:637`), the
producer `stride` (:319), `tiles_per_cta` (:325) and **`WGM`** (:308). So a
runtime override is behaviour-identical to the same value in `scored_shapes`,
and one build serves every arm. It is still only a screening instrument: a
winner goes through the full gate ladder before it is a result.

## Pre-registration: the split is quantized, and the arithmetic is knowable

Both roles are strided persistent loops, so what matters is not the CTA count
but the **number of rounds each role's critical path takes**:

- producer critical path = `ceil(gemm_tiles / NG)` tile-waves, `NG = 304 − NR`
- reducer critical path = `ceil(red_tiles / NR)` tile-rounds
- `WGM = 4` when `gemm_tiles <= NG`, else `num_pid_m` — so `NR` can *flip the
  exp_08 egress tile order*, which is worth 27.9% on the large shapes

| # | shape | gemm_tiles | red_tiles | GEMM waves for NR=4…80 | reduce rounds for NR=4…80 | WGM flips at |
|---:|---|---:|---:|---|---|---|
| 1 | 64×7168×18432 | 224 | 112 | 1 1 1 1 1 1 1 1 1 1 | 28 14 7 5 **4** 3 3 2 2 2 | never (NR≤80) |
| 2 | 512×4096×12288 | 512 | 64 | 2 2 2 2 2 2 2 **3 3 3** | 16 8 4 3 **2** 2 2 2 1 1 | n/a (always `num_pid_m`) |
| 3 | 2048×2880×2880 | 192 | 24 | 1 1 1 1 1 1 1 1 1 1 | 6 3 2 1 **1** 1 1 1 1 1 | never |
| 4 | 4096×4096×4096 | 256 | 32 | 1 1 1 1 1 1 1 **2 2 2** | 8 4 2 2 **1** 1 1 1 1 1 | **NR≥56** (4→16) |
| 5 | 8192×4096×14336 | 512 | 64 | 2 2 2 2 2 2 2 **3 3 3** | 16 8 4 3 **2** 2 2 2 1 1 | n/a |
| 6 | 8192×8192×29568 | 1024 | 128 | 4 4 4 4 4 4 4 **5 5 5** | 32 16 8 6 **4** 4 3 3 2 2 | n/a |

Columns are `NR = 4, 8, 16, 24, 32, 40, 48, 56, 64, 80`; the bold entry is
today's uniform `NR=32`.

Three consequences, registered before any measurement:

1. **The naive "NR=16 buys 63 µs of GEMM on shape 6" argument is wrong**, and
   the wave arithmetic says why: 1024 tiles over 272 *or* 288 producers is
   `4` waves either way. The producer critical path is **flat for every
   `NR ≤ 48`** on all six shapes. GEMM time should not fall at all — while
   reduce rounds double from 4 to 8. Expect `NR=16` to lose on shape 6.
2. **There is a hard cliff at `NR=56`** on shapes 2, 4, 5 and 6 (one extra
   producer wave: +50% on shape 2/5, +100% on shape 4, +25% on shape 6), and on
   shape 4 it also flips `WGM` 4→16. `NR ≥ 56` should be catastrophic there.
   `NG=256` (`NR=48`) is the last exactly-balanced point on shapes 4 and 6.
3. **The interesting shape is 1, not 6.** Its GEMM is one wave for the entire
   range (224 tiles ≤ 224 producers even at `NR=80`), so producers are *free*
   above `NR=32`, while its reduce rounds fall 4 → 3 → 2. Shape 1 is 77.5 µs and
   shape 6 is 1716.8 µs, and the ranking statistic is a **geometric** mean, so
   1 µs there is worth 22× the same microsecond on shape 6. If the optimum has
   moved anywhere, the arithmetic says it moved *up* on shape 1 and possibly to
   48 on shapes 4/6, and **not** down anywhere.

Predicted verdict: `NR=32` survives on shapes 2, 3, 5; `NR=40–48` weakly wins
shapes 4 and 6; `NR=56–80` wins shape 1. Net geomean gain expected to be small
(≤2%) and quite possibly inside the floor.

## Measurement protocol

The pipelined A/B harness has a **documented positional bias of up to 3.9% per
shape** (a null arm of two identically-behaving builds "measured" a 2%
regression), and the node's intrinsic floor is ~1.3% on the large shapes.
Therefore:

- Every arm is an independently allocated, independently zeroed `GemmRS`. All
  arms for a shape are live at once so passes can **interleave** them.
- The operand tensors are generated **once per shape and shared by every arm**,
  so arms differ only in `num_gemm_ctas` and in their own output heap.
- **Complete rotation**: `passes = len(arms)`, arm order rotated by one each
  pass, so every arm occupies every position exactly once. Positional bias is
  averaged out by construction rather than by hope, and the per-position
  residual is reported so the bias itself is measured.
- **Null twin**: two independent arms carry `NR=32`. Their spread is this
  instrument's floor. **No per-shape delta below it is evidence.**
- **Best (min) and median** over the 11 samples, never the mean.

## Gates

A candidate table only becomes a result after `tools/gate_ladder.sh
exp_13_cta_split`: M1 build, M2 resources/ISA, M3 17 shapes at `1e-2` **and**
`2e-3`, M4 three negative controls, M5 600-epoch soak, M7 timing. Correctness is
re-checked at every arm inside the sweep too, at both tolerances — a starved
split is a protocol stress test, not just a slow run.

## Denominators

Our harness, current best: `77.47 / 88.28 / 91.79 / 199.43 / 613.70 / 1716.76`,
geomean **225.62 µs**. Graded same-run vs frozen rank-1: ours 353.59–357.09 vs
306.23–308.80 (1.145–1.158×).

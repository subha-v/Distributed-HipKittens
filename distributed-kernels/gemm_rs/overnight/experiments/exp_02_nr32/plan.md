# exp_02 — land the settled uniform NR=32 in the shape table

**Axis: E4, producer/reducer split.** The axis itself is already settled; this
experiment is only the landing of a result that was measured last session and
then left on the floor.

## Why this is first tonight

It is the cheapest available ratchet step: the sweep has already been run, the
answer is known, and the only work is four integers and a gate ladder. Landing
it before E1/E2/E3 also means every later experiment is measured against a
denominator that already contains this win, so nothing later gets credit for it.

## Evidence this rests on

`experiments/LESSONS.md`, session 1: `NUM_REDUCER_CTAS` was swept over
{8, 16, 24, 32, 40, 48} per shape, all arms correct. A uniform **NR=32** is
optimal or within noise on all six graded shapes and beats the inherited
RadeonFlow table by **8.3% on shape 6** (2850.2 → 2632.1 µs).

The reason the donor's value is bad on shape 6 is mechanical: 8192×8192×29568
is by far the largest output, and `NR=8` leaves only eight reducer CTAs to
consume the packets from 296 producers, so the reduce side starves. The donor
submitted that value for a different tile schedule.

## The change

`distributed-kernels/gemm_rs/gemm_rs_mi300x_host_abi.hpp`, `scored_shapes`,
the `num_reducer_ctas` column only:

| row | shape | before | after |
|---|---|---|---|
| 1 | 64×7168×18432 | 32 | 32 |
| 2 | 512×4096×12288 | 48 | **32** |
| 3 | 2048×2880×2880 | 48 | **32** |
| 4 | 4096×4096×4096 | 48 | **32** |
| 5 | 8192×4096×14336 | 32 | 32 |
| 6 | 8192×8192×29568 | **8** | **32** |

Nothing else changes. `num_reducer_ctas` is consumed on the host to derive
`num_gemm_ctas = CU_COUNT - num_reducer_ctas`; the device reads it back as
`m3::CU_COUNT - g.num_gemm_ctas`, so no device logic is shape-table-dependent
and the change cannot alter the protocol. `config_row` is untouched, so the
compile/launch cache key is unaffected.

## Denominator

Tonight's ratchet baseline, re-measured on a verified-clean node immediately
before this experiment: per-shape means µs
`107.74 / 115.12 / 96.77 / 203.55 / 765.67 / 2865.78`, **geomean 285.02 µs**.

## Pre-registered expectation

Shape 6 improves to ≈2632 µs; shapes 2, 3, 4 move by less than noise in either
direction; shapes 1 and 5 are unchanged because their value did not change.
Geomean lands near **281 µs**, about 1.4% better. The geomean gain is much
smaller than the 8.3% single-shape gain because a geometric mean over six
shapes dilutes a win concentrated in one of them — that is expected and is not
a reason to skip it.

**Falsifier:** if any of shapes 2–4 regresses by more than noise, uniformity is
wrong and the table should keep 48 on that row. The per-shape means from M7
decide this directly, so the experiment is self-correcting.

## Gates

Full ladder, no shortcuts: M1 build → M2 resources/ISA → M3 correctness on all
17 shapes at both `1e-2` and `2e-3` → M4 three negative controls → M5 600-epoch
soak → M7 timing. Timing only after everything above passes, and only with
`rocm-smi --showpids` reporting a clean node.

## Deliverables

`result.md` with per-gate verdicts, the M2 resource tuple table, all six
per-shape means with stdev, the geomean, and the ratio to 285.02 µs. Outcome
appended to `../LESSONS.md` whether it wins or loses.

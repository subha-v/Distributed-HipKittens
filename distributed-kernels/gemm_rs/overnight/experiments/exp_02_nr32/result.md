# exp_02_nr32 — uniform NUM_REDUCER_CTAS = 32 across all six scored shapes

Run: 2026-08-11 08:29–08:43 UTC, node `banff-sc-cs47-05.dh170.dcgpu` (8x MI300X,
gfx942, SPX, 304 CU/GPU), container `dhk-gemmrs`. Exclusive GPU access; node
verified clean (`rocm-smi --showpids` -> "No KFD PIDs currently running") before
every GPU gate. All 8 GPUs at `perf_determinism` (clocks pinned) throughout.

## Change under test

`distributed-kernels/gemm_rs/gemm_rs_mi300x_host_abi.hpp`, `scored_shapes`
table, `num_reducer_ctas` column only:

| row | shape (M, N, K_local, bias) | BM/BN/BK | NR before | NR after |
|---:|---|---|---:|---:|
| 1 | 64, 7168, 2304, false | 32/256/32 | 32 | 32 (unchanged) |
| 2 | 512, 4096, 1536, true | 64/64/64 | 48 | **32** |
| 3 | 2048, 2880, 360, true | 128/256/32 | 48 | **32** |
| 4 | 4096, 4096, 512, false | 256/256/32 | 48 | **32** |
| 5 | 8192, 4096, 1792, true | 256/256/32 | 32 | 32 (unchanged) |
| 6 | 8192, 8192, 3696, false | 256/256/32 | 8 | **32** |

Inherited RadeonFlow values 32/48/48/48/32/8 -> uniform 32/32/32/32/32/32.
Nothing else changed. This is a host-side launch-partition constant
(`num_gemm_ctas = 304 - num_reducer_ctas`); it is not a kernel template
parameter, so no device code was recompiled differently.

Rows 1 and 5 keep NR=32 in both arms. They are therefore an **internal control**:
any measured delta on those two shapes is pure harness/node noise, which
calibrates the noise floor for judging the rest.

## Gate ladder

| gate | command | verdict |
|---|---|---|
| M1 build | `harness/build.sh` | **PASS** — all three modules built (`gemm_rs_mi300x.so` 389824 B, `gemm_rs_mi300x_control.so` 394728 B, `dhk_rt.so` 323496 B); only the pre-existing `hipFuncSetAttribute` nodiscard warning |
| M2 resources/ISA | `tools/m2_report.sh` | **PASS** (after running its prerequisites, see disclosure below) |
| M3 correctness | `m3_correctness.py all` | **PASS** — 17/17 shapes, `allclose(1e-2)=True` AND `tight(2e-3)=True` on every shape, `errbits=none` everywhere, epoch and signal cells exact (all == 3) |
| M4 controls | `m4_controls.py` | **PASS** — all three controls failed exactly as designed |
| M5 soak | `m5_soak.py 600 512 4096 12288 1 1` | **PASS** — 600 skewed changing-input epochs, no host reset, epoch/signal cells exact, worst max abs diff 4.883e-04 vs 2e-3 tight tolerance |
| M7 timing | `m7_bench.py 3 50` | ran twice, see below |

No gate failed. No error bit was ever raised outside the M4 controls, so no
protocol state was ever cleared or retried.

### M3 detail

All six scored shapes report `num_reducer_ctas = 32` and `num_gemm_ctas = 272`
in their resolved plan, confirming the change is live on the measured path. The
11 generic-fallback shapes are unaffected (`config_row = 0`, NR = 24) and still
pass, confirming the edit did not disturb the fallback row.

Worst-case accuracy over the 17 shapes: max abs diff 4.883e-04, a 4.1x margin
under the 2e-3 tight gate and 20x under the 1e-2 graded gate.

### M4 detail

- `CTRL_DROP_PUBLICATION`: rank 5 raised REDUCER_READY timeout (bit 26), output
  100% sentinel, zero payload reads/writes — fail-closed before any peer read.
- `CTRL_DROP_CREDIT`: epoch 1 clean (credit fast path), epoch 2 raised
  PRODUCER_CREDIT timeout (bit 25) on rank 6.
- `CTRL_REROUTE_SLOT`: no timeout (by design, silent-corruption control); ranks
  1 and 5 corrupted bitwise (57133/57344 and 57144/57344 elements changed);
  max abs diff 6.104e-05 golden vs 1.065e-02 rerouted; both ranks fail
  allclose at 1e-2 and 2e-3. Corruption is numerically detectable at a
  tolerance the clean run passes with margin.

## M2 resource tuple table

Per-instantiation, from `-Rpass-analysis=kernel-resource-usage` on the
production TU (`gemm_rs_mi300x.cpp`, gfx942). Seven template instantiations
cover all 17 shapes; the six scored shapes map onto rows 1-5 (rows 4 and 5 are
the 256x256 pair, K_TAIL=0 serving scored shapes 4 and 6, K_TAIL=1 serving
scored shape 5). `max_flat_workgroup_size` is 512 for all.

| instantiation BM/BN/BK, K_TAIL | VGPR | AGPR | SGPR | scratch B/lane | occupancy waves/SIMD | SGPR spill | VGPR spill | dyn stack |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 32/256/32, tail=0 | 94 | 0 | 106 | 0 | 5 | 57 | 0 | False |
| 64/64/64, tail=0 | 93 | 0 | 106 | 0 | 5 | 49 | 0 | False |
| 128/256/32, tail=1 | 170 | 0 | 106 | 0 | 2 | 73 | 0 | False |
| 256/256/32, tail=0 | 256 | 0 | 106 | **12** | 2 | 61 | **2** | False |
| 256/256/32, tail=1 | 256 | 0 | 106 | **12** | 2 | 73 | **2** | False |
| 32/64/64, tail=0 | 91 | 0 | 106 | 0 | 5 | 54 | 0 | False |
| 32/64/64, tail=1 | 91 | 0 | 106 | 0 | 5 | 80 | 0 | False |

`LDS Size [bytes/block]` reads 0 for every row because LDS is requested
dynamically at launch (`extern __shared__`); the real per-CTA LDS is
`2*(BM+BN)*BK*2` = 36864 / 32768 / 49152 / 65536 / 65536 / 24576 / 24576 bytes
in the row order above.

This tuple is **unchanged from baseline**, as expected: NR is a host-side
partition constant and does not enter the kernel template. The standing E1(a)
target is still open and still visible here — `AGPRs: 0` on every
instantiation, with the only VGPR spilling (2 VGPRs, 12 B/lane scratch) on the
two 256x256 rows, which are exactly the rows that carry scored shapes 4, 5 and 6
and therefore ~89% of the geomean's mass.

ISA property checks from `tools/m2_isa.sh` (24671-line gfx942 listing):

- 16-byte peer stores present: 7 `global_store_dwordx4` + 7 `flat_store_dwordx4`,
  0 `global_store_dwordx2`; 14 `global_store_short` (2 B tails).
- Reduction loads: 96 `global_load_dwordx4`.
- Ordering: 7 `buffer_wbl2 sc0 sc1` (each immediately followed by
  `s_waitcnt vmcnt(0)`, all inside `release_payload_system`), 7 `buffer_inv sc0 sc1`,
  0 `buffer_wbinvl1`, 473 `s_waitcnt vmcnt`, 98 `s_barrier`, 14 `s_sleep`
  (bounded poll backoff).
- Atomics: 28 `flat_atomic`, 0 `global_atomic` — epoch RMW (`flat_atomic_add`)
  plus error-bit `flat_atomic_or` only, as designed.
- MFMA: 184 `v_mfma_f32_16x16x16_bf16`, 0 `v_mfma_f32_32x32x8` (the E1(d) axis
  is untouched).
- Spills: 4 `scratch_store` / 4 `scratch_load`, all in the two 256x256 bodies.

## M7 timing

Protocol unchanged: `m7_bench.py 3 50` — 3 order rotations x 50 pipelined
iterations, duration-based warmup, clocks pinned. Run 1 is the ladder's M7.
Run 2 is a repeatability check using the **identical** command (see disclosure).

Baseline (measured tonight, same node, same protocol, clean):
`107.74 / 115.12 / 96.77 / 203.55 / 765.67 / 2865.78` us, geomean **285.02 us**.

| # | shape (m, n, k, bias) | NR | run 1 mean +- stdev | run 2 mean +- stdev | pooled mean | pooled vs base |
|---:|---|---:|---|---|---:|---:|
| 1 | 64, 7168, 18432, 0 | 32 | 108.12 +- 0.29 | 107.86 +- 0.25 | 107.99 | +0.23% |
| 2 | 512, 4096, 12288, 1 | 32 | 114.35 +- 0.80 | 113.44 +- 0.25 | 113.90 | -1.06% |
| 3 | 2048, 2880, 2880, 1 | 32 | 101.75 +- 8.48 | 96.80 +- 0.30 | 99.28 | +2.59% |
| 4 | 4096, 4096, 4096, 0 | 32 | 204.03 +- 1.02 | 204.00 +- 1.41 | 204.02 | +0.23% |
| 5 | 8192, 4096, 14336, 1 | 32 | 766.18 +- 1.43 | 769.45 +- 2.84 | 767.82 | +0.28% |
| 6 | 8192, 8192, 29568, 0 | 32 | 2647.66 +- 6.45 | 2633.04 +- 22.82 | 2640.35 | **-7.87%** |

Geometric means (pipelined wall clock, the competition's ranking statistic):

| arm | geomean | ratio to 285.02 baseline |
|---|---:|---:|
| baseline (NR 32/48/48/48/32/8) | 285.02 us | 1.0000 |
| **run 1 (uniform NR=32)** | **283.64 us** | **0.9952 (-0.48%)** |
| run 2 (uniform NR=32, repeat) | 280.73 us | 0.9850 (-1.50%) |
| pooled run 1+2 | 282.20 us | 0.9901 (-0.99%) |

Device-max geomeans for reference: run 1 280.87 us, run 2 278.02 us.
`all shapes correct = True` in both runs. Every shape remained device-bound.

### Noise floor, from the internal control

Shapes 1 and 5 have NR=32 in both arms, so their spread is pure noise:

- shape 1: 107.74 (base) / 108.12 / 107.86 -> max deviation **0.35%**
- shape 5: 765.67 (base) / 766.18 / 769.45 -> max deviation **0.49%**

So the run-to-run noise floor of this harness on this node is about **+-0.5%**
on a per-shape mean. Any per-shape delta inside that band is not a result.

### Shape 3 in run 1 was an outlier, not a regression

Run 1 shape 3 reported 101.75 us (+5.15%) with stdev 8.48 us, an order of
magnitude larger than any other row. Its three rotations were 96.44 / 111.53 /
97.28 — one contaminated rotation (rotation 1, where shape 3 ran second, right
after shape 2) against two clean ones that bracket the 96.77 baseline exactly.
Run 2 gives 96.80 +- 0.30 us, i.e. **+0.03% vs baseline**, with all three
rotations tight (97.05 / 96.47 / 96.88). Shape 3 is unchanged; the run-1 mean
was a single-rotation artifact. The pooled +2.59% for shape 3 in the table above
is therefore pessimistic and should be read as 0%.

## Verdict

**Uniform NR=32 is better. Adopt it.** The win is real, it is confined to shape
6, and it is roughly the pre-registered size.

Per shape, against the +-0.5% noise floor:

- **shape 1** (NR unchanged at 32): within noise, control.
- **shape 2** (48 -> 32): within noise to marginally better (-0.67% / -1.46%).
  Directionally favourable, but the run-2 value is the only one outside the
  noise band, so call it **within noise, not worse**.
- **shape 3** (48 -> 32): **within noise** (run 2: +0.03%). The run-1 +5.15% is
  a single-rotation outlier, refuted by run 2.
- **shape 4** (48 -> 32): **within noise** (+0.24% / +0.22%).
- **shape 5** (NR unchanged at 32): within noise, control.
- **shape 6** (8 -> 32): **better, decisively** — -7.61% in run 1, -8.12% in
  run 2, -7.87% pooled (2865.78 -> 2640.35 us). This is 15x the noise floor and
  reproduces the previously swept 8.3%. The donor value of 8 reducer CTAs
  starves the reduce side on the largest shape; 32 fixes it.

On the geomean: **better**, by -0.48% (run 1), -1.50% (run 2), -0.99% pooled.
The gain looks small only because the geomean dilutes a single-shape win: an
8% improvement on one of six shapes is worth at most 1.4% on the geomean
(0.9188^(1/6) = 0.986), and the other five shapes give back a few tenths. The
observed -0.99% pooled is consistent with that arithmetic. Note the geomean
delta (-0.5% run 1) is itself close to the noise floor, so the *geomean* claim
is weaker than the *shape 6* claim; shape 6 is what carries this experiment.

**New ratchet: 283.64 us (run 1, the ladder's M7), or 282.20 us pooled, against
the previous best of 285.02 us.** Recommend reporting 283.64 us as the new best
to stay conservative and protocol-literal, and measuring the next experiment
against it.

## Disclosures (compatibility repairs and protocol deviations)

1. **M2 needed its prerequisites run first — no change to what is measured.**
   `tools/m2_report.sh` reads `overnight/build/m1a.log` and
   `overnight/build/isa/*.s`. Those are produced by `tools/m1_build.sh` and
   `tools/m2_isa.sh`, not by `harness/build.sh`, and `overnight/build/` did not
   exist in this synced tree. The first M2 invocation therefore printed only
   `grep: .../m1a.log: No such file or directory`, a `FileNotFoundError` on the
   `.s`, and five more missing-file greps — and still **exited 0**, because the
   script neither sets `-e` nor checks its greps. That is a silent false pass
   and should be treated as a bug in the gate script: M2 cannot fail today.
   Remedy applied here: ran `tools/m1_build.sh` then `tools/m2_isa.sh` (both
   compile-only, CPU-only, no GPU involvement, same flags as the shipped
   module) and then re-ran `tools/m2_report.sh` unmodified, which produced the
   full table above. No tool script was edited. Cross-check: the resource
   remarks in `harness/build/gemm_rs_mi300x.log` — emitted by the very build
   that produced the `.so` used by M3/M4/M5/M7 — carry the same tuples, so the
   table describes the binary actually measured.
2. **M7 was run twice with the identical command.** Run 1 is the ladder's M7 and
   is reported as primary. Run 2 was triggered by shape 3's 8.48 us stdev under
   the standing rule that sub-5% deltas are re-run or paired. Nothing about the
   measurement changed: same script, same 3 rotations x 50 iterations, same
   shapes, same tolerances, same pinned clocks, same clean node. Both runs are
   reported in full; neither was discarded.
3. **A guard in this experiment's own driver script aborted M3 once, spuriously.**
   `rocm-smi --showpids | grep -q ...` makes grep exit on first match, which
   SIGPIPEs rocm-smi (a Python script) and, under `pipefail`, reported failure
   on a node that was in fact clean. Fixed in the experiment scripts by
   capturing rocm-smi output to a variable before matching. No harness or tool
   file was touched, and no GPU work had started when the abort fired.
4. **No tolerance, iteration count, shape, warmup or rotation count was changed
   at any point.** No error bit was cleared and retried.

## Artifacts

Raw stdout of every gate step is under `logs/`:

| file | contents |
|---|---|
| `m1_build.log` | M1 `harness/build.sh` |
| `m2_prereq_m1_build.log` | `tools/m1_build.sh` (M2 prerequisite) |
| `m2_isa.log` | `tools/m2_isa.sh` (M2 prerequisite, ISA property checks) |
| `m2_report.log` | M2 `tools/m2_report.sh`, successful invocation |
| `m3_correctness.log` | M3, all 17 shapes |
| `m4_controls.log` | M4, three negative controls |
| `m5_soak.log` | M5, 600 epochs |
| `m7_bench.log` | M7 run 1 (primary) |
| `m7_bench_run2.log` | M7 run 2 (repeatability check) |
| `m7_results_run1.json`, `m7_results_run2.json` | machine-readable M7 output |

The failed first M2 invocation overwrote nothing of value and its log was
replaced by the successful re-run; its exact output is quoted in disclosure 1.

Driver scripts used to run the ladder (this experiment's own files, not harness
or tool code): `00_preflight.sh`, `01_perflevel.sh`, `10_m1_build.sh`,
`20_m2_report.sh`, `21_check_build_dirs.sh`, `22_m2_prereq.sh`,
`30_m3_correctness.sh`, `40_m4_controls.sh`, `50_m5_soak.sh`,
`60_preflight_timing.sh`, `70_m7_bench.sh`, `71_m7_bench_rerun.sh`.
## Source-integrity check (concurrent edit by another agent — NOT in these numbers)

While this ladder was running, the local worktree acquired an edit to
`gemm_rs_mi300x.cpp` from another agent (`dispatch_gemm_rs_mi300x`, config_row 1
retiled from `launch_fixed<32, 256, 32, false>` to `launch_fixed<32, 64, 64, false>`),
plus new files `overnight/tools/gate_ladder.sh`, `overnight/experiments/exp_03_mainloop/`,
and a `plan.md` inside this experiment directory. None of that is mine.

**That edit did not reach the node and is not in any number above.** Verified
byte-exactly rather than by inference:

| file | sha256 at preflight 08:29 UTC | sha256 after M7 run 2, 08:49 UTC |
|---|---|---|
| `gemm_rs_mi300x_host_abi.hpp` | `4a2e26cf…34f1b2` | `4a2e26cf…34f1b2` (identical) |
| `gemm_rs_mi300x.cpp` | `4c75e1ef…65f5f097` | `4c75e1ef…65f5f097` (identical) |
| `gemm_rs_mi300x_constants.cuh` | `6a5cba2c…16cbc12e` | `6a5cba2c…16cbc12e` (identical) |
| `gemm_rs_mi300x_hk_adapter.cuh` | `3484cbdd…2417df72ec` | `3484cbdd…2417df72ec` (identical) |

Corroborating timestamps on the node (local time, UTC-5): all four sources
mtime `03:28:37`, i.e. this experiment's single `push.ps1`; the three `.so`
modules mtime `03:31:51`–`03:32:08`, i.e. this experiment's M1. Nothing was
written to the node between M1 and the end of M7 run 2 (`03:43`). Every gate
from M1 through M7 therefore ran against one byte-stable tree whose only
difference from baseline is the `num_reducer_ctas` column.

Live scored_shapes on the node after the run still reads `32/32/32/32/32/32`.

**Flag for the orchestrator:** the config_row 1 retile is sitting uncommitted in
the shared worktree. The next `push.ps1` from any agent will carry it to the
node, and the next M1 will silently measure it. It should be landed as its own
experiment with its own gate ladder, not picked up as a side effect of someone
else's push.
# exp_21 result — resource saturation vs CTA count (paper Fig 2 / Q4(a)), MI300X

**STATUS: BUILD-ONLY. No GPU run has happened yet.** This file is a stub with the
schema, the derivation slots and the verdict table pre-laid-out; the numbers are
filled in after `run_sweep.sh`. Nothing here is a measurement until this line is
replaced.

Plan and pre-registered predictions: `plan.md`. Task graph, buffers, rejected
alternatives: `design.md`.

---

## 1. What was built (this dispatch)

| file | what it is |
|---|---|
| `sat_ubench.cpp` | standalone HIP + pybind11 module: the three mode bodies, 7 `sat_kernel<MODE,CONC,CONTROL>` instantiations, tick calibration, operand/poison fill, destination-side verify, and the allocation / peer-access / launch surface |
| `build.sh` | hipcc → `exp_21_saturation/build/` (never `harness/build/`), plus the per-kernel resource-tuple table |
| `isa.sh` | `--save-temps` + per-kernel opcode census + `roc-obj`/`llvm-objdump` cross-check → `isa_report.txt` |
| `ceilings.sh` | this node's own xGMI / HBM / clock reporting → `ceilings_host.txt` |
| `run_saturation.py` | the sweep driver → `saturation.json` |
| `run_sweep.sh` | the only GPU-touching script: pre-flight, `setsid` + `timeout`, log capture |

## 2. Gate posture — stated explicitly

This is a **diagnostic module**, not a kernel candidate. It shares no object
with the production binding, is never imported by the graded path, and cannot
enter a graded build. There is therefore **no M3/M4/M5 ladder** for it. Its
correctness gates are:

1. **build + resource tuple + ISA verification** (`isa_report.txt`), and
2. the **destination-side checksum** on every mode-c point: the payload is a
   pure function of the window coordinates, the destination extent is
   pre-poisoned with `0x7FC0` (a value the generator cannot emit), and a verify
   kernel on the destination rank walks exactly the written extent and counts
   bitwise mismatches. `checksum_ok == false` invalidates that point's
   bandwidth; a silently failing store cannot fake a number.

No number from this experiment is ever reported as kernel performance.

## 3. ISA verification (fill from `isa_report.txt`)

| kernel | SGPR | VGPR | AGPR | scratch | spills | LDS | waves/SIMD | CTA/CU |
|---|---|---|---|---|---|---|---|---|
| `sat_kernel<a,isolated>` | | | | | | | | |
| `sat_kernel<b,isolated>` | | | | | | | | |
| `sat_kernel<c,isolated>` | | | | | | | | |
| `sat_kernel<b,concurrent>` | | | | | | | | |
| `sat_kernel<b,control>` | | | | | | | | |
| `sat_kernel<c,concurrent>` | | | | | | | | |
| `sat_kernel<c,control>` | | | | | | | | |

Required properties:

- mode c lowers to a **16-byte** peer store. Production emits
  `flat_store_dwordx4` for the peer emit and `global_store_dwordx4` for the
  reducer output (exp_02's `m2_report.log`), so either 16 B form is correct;
  what must be **absent** is any `buffer_store_*` (32-bit voffset — the exact
  truncation that broke rank-1's peer stores, HANDOFF.md) and any narrower
  store in the emit body.
- mode a emits `v_mfma_f32_16x16x16_bf16` and **never** `v_mfma_f32_32x32x8`,
  matching the production selection.
- mode b's 8 sources load as `global_load_dwordx4`.
- `scratch_load`/`scratch_store` must be **0** in the mode-a-bearing kernels; a
  spilling mainloop would not be this kernel's MFMA curve.
- LDS = 65,536 B in every arm ⇒ 1 CTA/CU by LDS alone ⇒ the H4 "no
  oversubscription regime" claim is structural, not asserted.

## 4. Measured ceilings on THIS node (fill from `ceilings.txt` / `ceilings_host.txt`)

| quantity | value | source (verbatim) |
|---|---|---|
| HBM peak | | `hipGetDeviceProperties`: 2 × memoryClockRate × memoryBusWidth / 8 |
| xGMI per link, per direction | | |
| xGMI aggregate egress | | |
| bf16 MFMA peak | | sclk (pinned reading) × 304 CU × FLOP/cycle/CU — arithmetic written out below |
| `s_memrealtime` tick rate | | measured by `calibrate_ticks`, 5 reps, median + spread |

MFMA arithmetic (to be written out, not quoted):
`peak = sclk_Hz × 304 CU × <FLOP/cycle/CU for v_mfma_f32_16x16x16_bf16>`.

If the node reports no per-link xGMI figure, that is recorded here and the
**measured plateau of the single-fanout curve is used as the empirical
ceiling**, labelled as such in the figure caption.

## 5. Knee table (fill after the run)

Knee = smallest C reaching **90% of that series' own plateau**.

| panel | series | knee C | plateau | concurrent/isolated at the knee |
|---|---|---|---|---|
| a | MFMA | | | n/a |
| b | HBM, isolated | | | |
| c | single, d=0 | | | |
| c | single, d=1 | | | |
| c | single, d=4 | | | |
| c | single, d=8 | | | |
| c | rr7, d=0 | | | |
| c | rr7, d=4 | | | |

## 6. Verdicts against the pre-registered predictions

| id | prediction | verdict | evidence |
|---|---|---|---|
| H1 | single-link emit saturates by 8–16 CTAs at d ≥ 4 | | |
| H2 | aggregate (rr7) egress saturates by 16–32 CTAs | | |
| H3 | concurrent < isolated at equal C, and the gap grows with the protocol knobs rather than with payload bytes | | |
| H4 | MFMA TFLOPS ~linear to 304 CTAs; 1 CTA/CU from the resource tuple | | |
| H5 | d = 1 costs a large monotone fraction of the d = 0 plateau; d = 4→8→0 is small | | |
| falsifier | isolated single-link needs ≥ 32 CTAs for ~75% of its own plateau at some depth ⇒ the emit is bandwidth-limited, not protocol-limited, and the in-flight-bound priority inverts | | |

## 7. Caveats to carry into the paper caption

- One process drives 8 devices with `hipDeviceEnablePeerAccess`; this is **not**
  the evaluator's one-process-per-rank HIP-IPC topology. Address arithmetic is
  equivalent for `translate_peer`, but any comparison against evaluator or
  rank-1 numbers must repeat this caveat.
- Clocks pinned at 1900 MHz (`tools/set_clocks.sh pin 1900`); `clocks_before`
  and `clocks_after` are recorded on every point and must be inspected before
  any curve is believed.
- A fixed work list means the 10–50 ms per-point band is violated at the
  extremes of each CTA sweep (long at small C). `rounds` is recorded per point
  so the tail imbalance of the fixed list is visible.
- Panel b's working set (512 MB read + 64 MB written) exceeds this node's
  256 MB Infinity Cache, so it is an HBM curve; some MALL residency remains and
  the number is an upper bound on what the reducer sees from HBM alone.
- The mode-c destination is **fine-grained (uncached)** by default. With a
  coarse-grained destination and no release, the issuing XCD's L2 can absorb
  16 B peer stores — which is exactly why production needs `buffer_wbl2 sc0
  sc1` — and the no-protocol isolated arm would report L2 bandwidth as fabric
  bandwidth. `run_sweep.sh coarse` runs the production-parity cross-check into
  `saturation_coarse.json`; **note that a coarse + no-protocol checksum failure
  is itself the evidence that the stores were being absorbed**, i.e. the
  checksum doubles as the collapse test for this hazard.

## 8. `saturation.json` schema

Header: `experiment`, `figure`, `generated`, `wall_minutes`, `node`
(host / arch / cu_count / full `device_props` / topology caveat / clock policy),
`ceilings` (each entry `{value, source}`, plus the raw tool output),
`tick_rate_hz` + `tick_rate_spread_pct` + `tick_rate_samples_hz`, `geometry`
(every size and the derived bytes/FLOPs per work unit), `config` (the exact CLI),
`schema`, `points`.

Each element of `points`:

```
mode                    "a" | "b" | "c"
ctas                    C (the x-axis)
depth                   mode c only: in-flight remote-op bound; 0 = unbounded = production shape
fanout                  mode c only: "single" (one link) | "rr7" (aggregate egress)
overlay                 "isolated" | "concurrent" | "reserve_control"
protocol                bool: release + probe RMW + publication re-added
metric                  "TFLOPS" | "GBps"
value                   median over rotations of the per-rank median
samples                 one value per rotation
per_rank                {resource: [8], gemm_tflops: [8]} from the median rotation
concurrent_gemm_tflops  GEMM role TFLOPS on the 304-C CTAs (concurrent arms)
gemm_slowdown           control TFLOPS / live TFLOPS on the same 304-C CTAs
res_span_us             resource role s_memrealtime span
gemm_span_us            GEMM role s_memrealtime span
host_wall_ms            cross-check only, never the metric
rounds                  ceil(work / C)
checksum_ok             mode c: destination-side bitwise verify passed
checksum_mismatches     count of mismatched elements
checksum_fold           u64 fingerprint of everything the verify read
payload_granularity     "fine" | "coarse"
clocks_before/after     rocm-smi --showclocks around the point
```

# exp_21 result — resource saturation vs CTA count (paper Fig 2 / Q4(a)), MI300X

**STATUS: BUILT AND ISA-VERIFIED. No GPU run has happened yet** (another agent
held the devices). Section 3 is real; sections 4–6 are empty slots that
`run_sweep.sh` fills. Nothing outside section 3 is a measurement until this line
is replaced.

Launch command for the sweep:

```bash
powershell -ExecutionPolicy Bypass -File distributed-kernels\gemm_rs\overnight\tools\nsh.ps1 \
  -Script distributed-kernels\gemm_rs\overnight\aug11\exp_21_saturation\do_sweep.sh -ArgLine "quick"
# then, once the smoke passes:
#   ... -ArgLine "full"      (the figure data)
#   ... -ArgLine "coarse"    (the production-parity payload-granularity cross-check)
```

### One thing that cannot be checked without a GPU

The launch requests **65,536 B of dynamic LDS** in every arm (`SAT_LDS`, the
whole gfx942 per-workgroup budget, static LDS 0). The production kernel does
exactly this (`dynamic_shared_memory()` returns `kittens::MAX_SHARED_MEMORY`
= 65,536), so it is expected to be accepted without `hipFuncSetAttribute` — but
it is the one thing in this module that a compiler cannot prove. `run_sweep.sh
quick` fails loudly on it (`sat_kernel launch: ...`) if it is wrong, and the
fallback is to drop the request to `SAT_LDS` only for the mode-a-bearing arms
and pass the emit window's 16,384 B for isolated mode c, at the cost of no
longer holding occupancy fixed across arms.

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

## 3. ISA verification — DONE (2026-08-12, `isa_report.txt`)

`hipcc` HIP 7.2.53211, AMD clang 22.0.0git (roc-7.2.3), `--offload-arch=gfx942`,
`-DKITTENS_CDNA3 -O3 -ffast-math`. Two independent routes agree: the
`--save-temps` `.s` (8,486 lines) and `llvm-objdump -d --mcpu=gfx942` of the
gfx942 code object extracted with `llvm-objcopy --dump-section=.hip_fatbin` +
`clang-offload-bundler` out of **the shipped `sat_ubench.so`** — so a
build-flag divergence between the verified code and the code the driver
dlopens cannot hide.

| kernel | SGPR | VGPR | AGPR | scratch | SGPR spill | VGPR spill | static LDS | waves/SIMD |
|---|---|---|---|---|---|---|---|---|
| `sat_kernel<a, isolated>` | 48 | 193 | 0 | 0 | 0 | 0 | 0 | 2 |
| `sat_kernel<b, isolated>` | 82 | 86 | 0 | 0 | 0 | 0 | 0 | 5 |
| `sat_kernel<c, isolated>` | 105 | 20 | 0 | 0 | 0 | 0 | 0 | 7 |
| `sat_kernel<b, concurrent>` | 86 | 195 | 0 | 0 | 0 | 0 | 0 | 2 |
| `sat_kernel<c, concurrent>` | 106 | 195 | 0 | 0 | 0 | 0 | 0 | 2 |
| `sat_kernel<b, control>` | 52 | 194 | 0 | 0 | 0 | 0 | 0 | 2 |
| `sat_kernel<c, control>` | 52 | 194 | 0 | 0 | 0 | 0 | 0 | 2 |

`private_segment_fixed_size` is 0 in every kernel and the module contains zero
`scratch_load`/`scratch_store`: nothing spills to memory anywhere.

**Static LDS reads 0 because every byte of this ubench's LDS is dynamic**, so
the compiler's `waves/SIMD` column ignores it. The launch requests 65,536 B of
dynamic LDS in *every* arm, which pins 1 CTA/CU everywhere at runtime; the
isolated b (5 waves/SIMD) and c (7 waves/SIMD) figures above are the
register-only bound and are **not** the occupancy those arms will run at. The
concurrent arms are at 1 CTA/CU from registers alone as well.

Per-kernel opcode evidence (identical in the `.s` and the objdump):

| property | mode a | mode b | mode c | verdict |
|---|---|---|---|---|
| 16 B peer store | — | — | **1 × `flat_store_dwordx4 v[6:7], v[12:15]`** | the peer emit is a single 16 B store with a 64-bit flat address — the same form production emits for its peer emit (exp_02 `m2_report.log`) |
| `buffer_store_*` | 0 | 0 | **0** | no 32-bit-voffset store anywhere; the truncation class that broke rank-1's peer stores (HANDOFF.md) is absent |
| narrower stores in the emit body | — | — | **0** | the `store_dwordx2`/`dword`/`short` counts in the census all sit in the `work_done`/`stamps`/fill/verify code, never in the packet loop |
| 16 B LDS read of the staged source | — | — | **1 × `ds_read_b128`** | the emit's source read is one 16 B LDS access |
| MFMA opcode | **64 × `v_mfma_f32_16x16x16_bf16`** per k-iteration | — | — | matches the production selection exactly |
| `v_mfma_f32_32x32x8` | **0** | 0 | 0 | production never selects it and neither does this |
| 8 REDV=1 source loads | — | **8 × `global_load_dwordx4`** + 1 × `global_store_dwordx4` | — | eight 16 B source packets in, one packed 16 B packet out |
| `s_setprio` | present (10 module-wide) | — | — | the exp_09 issue-priority window survived |
| release writeback | 0 | **0** | **`buffer_wbl2 sc0 sc1`, and only here** | the entire module contains exactly two `buffer_wbl2 sc0 sc1`, both inside mode c's `--protocol` branch. Protocol cost is therefore *structurally* separable from payload cost in this binary — the payload-only arms cannot emit a release even by accident |
| `buffer_inv` | 0 | 0 | 0 | no acquire side; the ubench never consumes peer payload in a timed path |

Two findings worth keeping:

1. **The relaxed completion counter matters.** The first build used an
   `__ATOMIC_RELEASE` increment for `res_done` and that alone put a
   `buffer_wbl2` into the mode-b concurrent and control kernels *and* cost 4
   SGPR spills in `sat_kernel<c, concurrent>`. Since the end stamp is taken
   after that store, an L2 writeback there would have landed inside the
   measured span — expensive in mode b, which leaves 64 MB of dirty output
   behind. Relaxed removed the writeback and the spills both.
2. **H4 is confirmed structurally, twice over.** Mode a is 1 CTA/CU from
   registers alone (193 VGPRs → 2 waves/SIMD → 2×4/8 = 1 CTA/CU at 512
   threads) *and* from LDS alone (65,536 B = the entire gfx942 per-workgroup
   budget). No oversubscription regime exists for this body, so a superlinear
   or saturating MFMA curve cannot be an occupancy artefact.

Mode a here is 193 VGPRs with no spills against production's 256 with 2 VGPR
spills on the same 256/256/32 row, because the ubench body carries no
epilogue, bias, credit wait or emit. The k-loop schedule is the same; the
register headroom is not, and that is stated rather than glossed.

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

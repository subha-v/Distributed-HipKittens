# exp_21 result — resource saturation vs CTA count (paper Fig 2 / Q4(a)), MI300X

**STATUS: MEASURED.** Full sweep 2026-08-12 10:34:13–10:39:10Z under the GPU
lease: **260 points, 4.87 min, zero zero-valued points, 224 destination
checksums verified, zero failures, one single distinct destination fingerprint
across every configuration.** Data: `saturation.json` (raw), `knees.json`
(derived), `saturation.csv` (tidy, one row per point), `ceilings.txt` (verbatim
tool output). Section 3 is the pre-run ISA verification and is unchanged.

**The headline.** The 7-peer emit reaches 90% of its own plateau at **C = 16
CTAs of 304 (5.3% of the machine)** and a single peer link at **C = 2**. The
pre-registered falsifier did not trigger: no isolated single-link curve needs
anywhere near 32 CTAs. The tiny-pool claim holds for our emit shape on gfx942.

**Two things this measurement says that were not asked for.** (1) The
protocol — the two `buffer_wbl2 sc0 sc1` and their `vmcnt(0)` drain — costs
**more than half the achievable egress bandwidth at identical payload bytes**
(0.44× at the rr7 knee, 0.37× at the single-link knee), which is a much larger
term than the concurrent-vs-isolated interference (0.96× at that same knee).
(2) **H4 is falsified as stated, for a mechanical reason worth a paragraph in
the paper**: MFMA is linear only to C ≈ 160, and the roll-off is not occupancy
(the body is 1 CTA/CU by construction) but the mainloop's own operand stream
hitting the memory path — 4,549 GB/s of operand demand at C = 304 against the
4,593 GB/s plateau panel b measures independently, i.e. 99.0% of it.

Plan and pre-registered predictions: `plan.md`. Task graph, buffers, rejected
alternatives: `design.md`. Knee definitions: `knees.py` docstring.

### The two things that could not be checked without a GPU — both pass

Both were gated by a dedicated `--probe` phase (full-size allocation, one launch
of all seven kernels, one checksum, tiny work counts) before any timing:

- **65,536 B of dynamic LDS launches** in every arm, with no
  `hipFuncSetAttribute`, exactly as production does it.
- **The full-size fine-grained heap allocates on all eight ranks** (4.00 GiB per
  rank, 8 slots × 512 MiB). No fallback to a halved `--c-work` was needed.

`--quick` shrinks `c_work` and therefore does *not* exercise the allocation;
that is precisely why `--probe` exists and runs first.

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

## 4. Measured ceilings on THIS node

Every figure below comes from this node. **No gfx950 number appears anywhere in
this experiment's artifacts.** Verbatim tool output: `ceilings.txt`.

| quantity | value | source |
|---|---|---|
| xGMI per link, per direction | **64.0 GB/s** | `rocm-smi --shownodesbw`: `64000-64000` for all 28 GPU pairs, `Units: mps`; `--showtopo` confirms link type `XGMI`, 1 hop, weight 15 for every pair (all-to-all, no 0-0 pair) |
| xGMI aggregate egress per GPU | **448 GB/s** | 7 peers × 64.0 GB/s, from the same all-to-all matrix |
| HBM peak | **5325 GB/s** | `amd-smi static`, `VRAM:` block — `TYPE: HBM`, `VENDOR: SAMSUNG`, `SIZE: 196592 MB`, `BIT_WIDTH: 8192`, `MAX_BANDWIDTH: 5325 GB/s` |
| bf16 MFMA peak | **1307.4 TFLOPS** | arithmetic below, from `hipDeviceProp_t.clock_rate_khz = 2 100 000` and `multi_processor_count = 304` |
| `s_memrealtime` tick rate | **99.7366 MHz** (10.0264 ns/tick) | measured, method below |

**MFMA arithmetic, written out rather than quoted.** One
`v_mfma_f32_16x16x16_bf16` is 2 × 16 × 16 × 16 = 8192 FLOP per wave-instruction.
gfx942 has 4 SIMDs per CU and issues one such MFMA per SIMD every 16 cycles, so
4 × 8192 / 16 = **2048 bf16 FLOP/cycle/CU**. With this node's reported values:
304 CU × 2.100 × 10⁹ cycles/s × 2048 = **1307.4 TFLOPS**. Panel a's measured
peak of 582.36 TFLOPS is **44.5%** of that.

**A ceiling that does not survive contact with the measurement, and is reported
anyway.** The conventional `hipDeviceProp_t` derivation
`2 × memoryClockRate(1 300 000 kHz) × memoryBusWidth(8192 bit) / 8` gives
**2662.4 GB/s** — and panel b *measures 4593.2 GB/s*, exceeding it by 1.73×. The
props-based figure is exactly half of the 5325 GB/s the node itself reports for
the same 8192-bit bus, i.e. the 2× formula undercounts HBM3 on gfx942 by a
factor of two. **Use 5325 GB/s; the props number is recorded here only so that
nobody re-derives it and believes it.** Panel b reaches 86.3% of 5325 GB/s.

**Are the xGMI numbers trustworthy given the ambiguous `mps` unit?** Read as
MB/s, `64000 mps` = 64.0 GB/s per link, and the measurement corroborates that
reading at two independent scales: the single-peer emit plateaus at 61.05 GB/s
(**95.4%** of one link) and the 7-peer emit at 403.18 GB/s (**90.0%** of 7
links). A unit error of any common factor would break one of those two
agreements. This is also positive evidence that the fine-grained destination did
its job — an L2-absorbed store would have reported *above* the fabric ceiling,
not 90–95% of it.

**Tick-rate method (exp_22 needs this number).** `sat_calib_kernel` spins on
`__builtin_amdgcn_s_memrealtime()` until a target tick count elapses while the
host times the launch with `std::chrono::steady_clock`. Two stages, because the
rate is the unknown: a 10⁶-tick pass gives a rough rate, then the target is set
to 0.2 × rough so each of 5 reps runs ≈ 200 ms. Median **99 736 591 Hz**, samples
{99.7276, 99.7338, 99.7375, 99.7366, 99.7377} MHz, **spread 0.0101%**. Note this
is ≈100 MHz on gfx942, numerically close to the sibling's gfx950 figure, but it
is measured here and not inherited.

## 5. Knee table

Knee = smallest C reaching **90% of that series' own plateau** (plateau = max
measured value; no fitted asymptote). Full curves: `saturation.csv`.
Concurrent/isolated is at the isolated series' knee C. `gemm_slowdown` is the
**C-matched reserve-only control** TFLOPS divided by the concurrent-with-traffic
TFLOPS on the *same* 304−C GEMM CTAs — never against the full 304-CTA grid.

| panel | series | knee C | plateau | at knee | conc/iso at knee | gemm_slowdown at knee |
|---|---|---|---|---|---|---|
| a | MFMA | 256 | 582.36 TFLOPS | 529.32 | n/a | n/a |
| b | reduce, isolated | **none ≤ 304** | 4593.2 GB/s | — | 0.80 at C=256 | 3.58 at C=256 |
| c | single, d=0 | **2** | 61.98 GB/s | 61.05 | 0.888 | 0.984 |
| c | single, d=1 | **4** | 51.22 GB/s | 51.22 | 0.976 | 1.035 |
| c | single, d=4 | **2** | 57.84 GB/s | 57.84 | 0.747 | 0.976 |
| c | single, d=8 | **2** | 58.33 GB/s | 58.33 | 0.789 | 0.982 |
| c | rr7, d=0 | **16** | 403.18 GB/s | 387.61 | **0.957** | **1.221** |
| c | rr7, d=1 | 16 | 334.03 GB/s | 334.03 | 0.640 | 1.149 |
| c | rr7, d=4 | 16 | 358.05 GB/s | 358.05 | 0.821 | 1.168 |
| c | rr7, d=8 | 16 | 358.27 GB/s | 358.27 | 0.882 | 1.173 |
| c | rr7, d=0, **protocol on** | 64 | 265.53 GB/s | 265.53 | 0.983 | — |
| c | single, d=0, **protocol on** | 8 | 49.14 GB/s | 49.14 | 0.988 | — |

Panel b has **no knee within the sweep**: the REDV=1 reduce body is still rising
at C = 304 (263.8 → 523.7 → 1052.7 → 2154.2 → 3337.5 → 3756.2 → 4593.2 GB/s for
C = 8 → 304), i.e. it scales essentially linearly to 64 CTAs and needs the whole
machine to approach the HBM ceiling. That is the sharpest contrast in the figure:
**egress saturates at 16 CTAs, the reduce does not saturate at 304.** The
asymmetry is the argument for where a pool is worth dedicating and where it is
not.

Panel a is linear to C ≈ 160 and then rolls off; per-CTA TFLOPS relative to
C = 32: 100.0, 99.9, 99.4, 99.2, 98.2 (C = 160), then 91.3, 86.5, 81.2, 76.0,
**75.2% at C = 304**. Total is monotone throughout (81.5 → 582.4 TFLOPS).

## 6. Verdicts against the pre-registered predictions

| id | prediction | verdict | evidence |
|---|---|---|---|
| H1 | single-link emit saturates by 8–16 CTAs at d ≥ 4 | **confirmed, and stronger than predicted** | knee is **2** CTAs at d = 0/4/8 and **4** at d = 1 — well below the predicted band. 75%-of-plateau is reached at C = 2 at every depth |
| H2 | aggregate (rr7) egress saturates by 16–32 CTAs | **confirmed, at the low edge** | knee = **16** at every depth. Scaling is near-perfect below it (36.1 → 72.0 → 143.8 → 286.6 GB/s for C = 1 → 8, i.e. 1.99–2.00× per doubling), then flat (387.6 → 392.3 → 403.2 for C = 16 → 64) |
| H3 | concurrent < isolated at equal C, and the gap grows with the protocol knobs rather than with payload bytes | **confirmed on the substance, with a correction to the framing** | concurrent < isolated at essentially every point (conc/iso 0.63–1.00; 4 of 112 points sit 0.2–1.7% above 1.0, inside rotation spread). But the protocol term is *much larger* than the interference term at identical payload bytes: at the rr7 knee, protocol-on/off = **0.440** while conc/iso = 0.957. The two effects are separable and the protocol one dominates. This is the same conclusion exp_20 reached from the other side (fabric amplification 1.0007× at 99.9% full-64 B), so the egress *width* axis is closed from both directions |
| H4 | MFMA TFLOPS ~linear to 304 CTAs; 1 CTA/CU from the resource tuple | **occupancy claim confirmed; linearity FALSIFIED past C ≈ 160** | 1 CTA/CU holds structurally (193 VGPR *and* 65,536 B LDS), so this is not an occupancy artefact. Per-CTA rate is flat within 1.8% to C = 160 and then falls to 75.2% at C = 304. Mechanism, and it is not a mystery: one 256×256 tile over K = 3712 reads 3.80 MB of operands for 486.5 MFLOP = 7.813 × 10⁻³ B/FLOP, so 582.36 TFLOPS demands **4549 GB/s** — **99.0% of the 4593 GB/s memory-path plateau panel b measures independently**. Past C ≈ 160 this mainloop is memory-path-bound, not MFMA-bound |
| H5 | d = 1 costs a large monotone fraction of the d = 0 plateau; d = 4→8→0 is small | **confirmed** | rr7 at C = 8: d1 = 167.9, d4 = 230.3, d8 = 245.9, d0 = 286.6 GB/s — monotone in depth, d = 1 costs 41% of unbounded. At and above the knee the depths converge: at C = 16, d1/d0 = 0.862 and d4, d8 are within 8% of d0. So the in-flight bound is a *below-the-knee* knob, which is exactly the regime a real producer runs in |
| falsifier | isolated single-link needs ≥ 32 CTAs for ~75% of its own plateau at some depth ⇒ the emit is bandwidth-limited, not protocol-limited, and the in-flight-bound priority inverts | **NOT TRIGGERED** | 75%-of-plateau C for isolated single-link is **2 at every depth** (0, 1, 4, 8) — 16× below the threshold. The tiny-pool story stands for our emit shape and the in-flight-bound priority does not invert |

### The protocol/payload split, stated for the paper

Both arms issue the *same* 16 B `flat_store_dwordx4` packets to the same
addresses in the same order; the only difference is the release. The entire
module contains exactly two `buffer_wbl2 sc0 sc1`, both inside the
`--protocol` branch (section 3), so this is a structural separation, not a
flag we are trusting.

| series | protocol-off plateau | protocol-on plateau | on/off | on/off at the off-knee |
|---|---|---|---|---|
| single, d=0 | 61.98 | 49.14 | 0.793 | **0.366** (C = 2) |
| single, d=4 | 57.84 | 49.15 | 0.850 | 0.371 (C = 2) |
| rr7, d=0 | 403.18 | 265.53 | 0.659 | **0.440** (C = 16) |
| rr7, d=4 | 358.05 | 254.98 | 0.712 | 0.450 (C = 16) |

Two readings, and they matter for different reasons. At the *knee* — the
CTA count a topology-correct pool would actually use — the release costs
**56–63% of egress bandwidth**. Protocol-on also needs a *larger* pool to
plateau (single-link knee moves 2 → 8, rr7 16 → 64), because each release
serialises a drain that more CTAs can hide. That is the mechanism behind
E3/RELEASE_GROUP=4 being one of the two biggest wins this kernel has, and it
predicts that further coarsening still has room.

## 7. Caveats to carry into the paper caption

- One process drives 8 devices with `hipDeviceEnablePeerAccess`; this is **not**
  the evaluator's one-process-per-rank HIP-IPC topology. Address arithmetic is
  equivalent for `translate_peer`, but any comparison against evaluator or
  rank-1 numbers must repeat this caveat.
- **Clocks: determinism engaged, and the recorded snapshots read idle — do not
  misread them.** `rocm-smi` shows `Perf: perf_determinism` on all eight devices
  for the whole campaign. The `clocks_before`/`clocks_after` strings stored on
  every point are sampled from the *host between launches*, so they read
  120–170 MHz — that is the idle level, not the in-kernel clock, and the standing
  trap on this node (idle ≈ 125–132 MHz vs ≈ 1900 under load) is exactly why the
  warmup is duration-based (≥ 400 ms per point, ≥ 50 ms in `--quick`) rather than
  fixed-iteration. Treat those fields as a determinism-mode audit trail, not as
  the clock a measurement ran at.
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

## 8. Campaign record, and three disclosed disturbances

The sweep ran under `tools/gpu_lease.sh` throughout (`ACQUIRE exp_21` 10:33:39Z,
`RELEASE exp_21` 10:39:11Z; probe 10:33:46, quick 10:33:46–10:34:13, full
10:34:13–10:39:10). Node was clean at launch: `rocm-smi` reported VRAM 0–1% and
GPU 0% on all eight devices. None of the three disturbances below touched the
measurement, and the evidence for that claim is stated rather than asserted.

**1. A wedged KFD process blocked the lease for 30 minutes, and the lease tool
was wrong about it.** `harness/m9_stale_slot.py` (another experiment's control)
sat on all eight devices for 30+ minutes in kernel state **`D` with
`wchan = exit_mm`**, CPU time flat across a 75 s sampling interval, CU occupancy
0, ~10 GB of VRAM still mapped. That process had already left user space and was
stuck in amdgpu/KFD address-space teardown: it could never dispatch again, and it
could not be reclaimed either — `D` state ignores SIGTERM, and SIGKILL is
forbidden on this node. `gpu_lease.sh` counted it as "draining" and therefore
**aborted both exp_21's and exp_24's acquire after their full 300 s drain
window** (`ABORT exp_21 -- node dirty after 300s` 10:06:53Z, `ABORT exp_24`
10:11:54Z). The drain check now classifies exiting/zombie KFD pids and ignores
them; exp_26's agent landed an equivalent fix independently at 10:23Z and theirs
is the version now installed. Nothing was preempted and nothing was signalled.

**2. A CRLF push from a concurrent agent corrupted this experiment's scripts
mid-run — after the data had been written.** At ≈10:37Z another agent pushed the
shared tree with CRLF line endings intact, which (a) broke `tools/gpu_lease.sh`
for every agent for a few minutes and (b) truncated and rewrote
`run_sweep.sh`'s inode while bash was executing it, so the interpreter resumed
at a shifted byte offset. The visible damage was entirely in the *post-run* lines:
`tail` failed on a log path with a trailing `\r`, and `exit $rc` became
`exit: 1: numeric argument required`, which made the phase report `exit=2` and
skipped the `coarse` phase. **`run_saturation.py` had long since been read and
compiled into memory, and `saturation.json` was written normally at 10:39:10Z.**
Integrity evidence for the data itself: 260 points with **zero** zero-valued
entries, all seven planned mode-c C values and all four depths and both fanouts
present, **224/224 checksums pass with exactly one distinct destination
fingerprint**, tick-rate spread 0.0101%, and the independently-run `quick` phase
agrees with the full run on the single-peer plateau (50–55 vs 54.8 GB/s). The
lost artifact is the *console log* of the full sweep, which the corrupted
redirect truncated; every number lives in the JSON. Mitigation now in place:
`go_campaign.sh` freezes CR-stripped copies of the whole chain
(`campaign.sh`, `run_sweep.sh`, `run_saturation.py`, `knees.py`) into
`logs/run_<stamp>/` and runs those, so no concurrent push can reach a running
campaign. `SAT_BASE` keeps the compiled module and artifacts in the experiment
directory while the driver executes from the frozen copy.

**3. The `coarse` cross-check is queued, not skipped.** Its first attempt failed
in 1 s for a path reason (the frozen driver looked for `build/sat_ubench.so`
beside itself), fixed by `SAT_BASE`; it is re-queued behind another agent's
lease hold. **The fine-grained result does not depend on it**: the L2-absorption
hazard would show up as a plateau *above* the fabric ceiling, and the measured
plateaus are 90.0% and 95.4% *of* the node's reported xGMI figures. The coarse
arm is a confirmation of the mechanism (does a cached destination inflate the
no-protocol number, and/or fail its checksum), not a load-bearing input.

## 9. `saturation.json` schema

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

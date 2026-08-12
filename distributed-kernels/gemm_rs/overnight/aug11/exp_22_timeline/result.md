# exp_22 result — IN PROGRESS (Phase 1 complete, Phase 2 gated)

**Verdict: none yet.** No GPU job has been run and no kernel file has been
touched. This file exists so the morning read is not blank; it will be
rewritten with numbers when Phase 2 lands.

## Phase 1 — done, CPU only

| deliverable | state |
|---|---|
| `plan.md` | done — figure, arms, pre-registered expectations, publication gates |
| `design.md` | done — phase enum, ring sizing, traffic model, tick calibration, rejected alternatives, **the patch plan with file:line insertion points** (§9) |
| `parity_gate.sh` + `parity_check.py` | done — gate 1, three TUs, byte-identical OFF comparison against the post-exp_14 M2 table |
| `build_trace.sh` | done — flag-ON module into `exp_22_timeline/build/`, never `harness/build/` |
| `trace_run.py` | done — arm (b) driver, in-situ tick regression, both correctness tolerances |
| `b0_ref.py`, `b0_rank0_wrap.sh`, `b0_capture.sh`, `b0_kernel_map.py` | done — arm (a) capture and the trace-derived kernel map |
| `bin_timeline.py` | done — owns the whole traffic model for both arms; 10 µs bins; ±10% integral gate |
| `plot_timeline.py` | done — the 3×N grid |
| `c_rank1_probe.sh` | done — arm (c) feasibility probe, read-only, no GPU |
| the patch itself | **applied** — 21 sites, +109 lines, 0 deletions, flag default 0 |
| `parity.json` | **PASS** — see the gate section below |
| `go_gpu.sh` | done — the whole GPU phase under `gpu_lease.sh`, release trapped |

## Why the phase enum carries a credit-wait pair — preserved reasoning

The charter's nine-tuple has no producer credit-wait. Adding
`CREDIT_WAIT_BEG`/`CREDIT_WAIT_END` (ids 3 and 4) is not decoration; without
them the producer's **reuse-credit stall lands inside the emit interval**, and
because the xGMI strip is computed as (phase bytes ÷ phase duration), a longer
emit interval carrying the same bytes deflates the strip by exactly the length
of that stall. The figure would then show our egress running slower than it
does, and — worse — it would point the reader at the wrong resource, because
the missing time would look like fabric cost rather than protocol waiting.

The size of that mis-attribution is now measured, not hypothetical: exp_20's
refreshed profile ranks `sync` as the **second-largest non-GEMM pool on shape 6
at 168.6 µs**, ahead of both reduce and release. Folding a 168.6 µs stall into
the emit interval would have been one of the largest errors this figure could
contain.

The same argument is why `RELEASE_END` exists (the release is an interval worth
65.4 µs on shape 5, not an instant) and why `PUBLISH_END` exists (otherwise the
publish loop is charged to the next tile's mainloop).

Ring depth is derived the same way — from this kernel's own per-row worst case,
29 events on shape 6 and 20 on shape 5, giving depth 64 at 2.17× headroom.
The sibling's depth 24 is a gfx950 MoE fact about a different phase taxonomy
(M0-M9 plus service stripes) and was not inherited.

## GATE 1 — resource-tuple parity: **PASS**

Data: `parity.json` (`exp22.parity.v1`). Three TUs from one source, identical
options apart from the flag: `off_absent` (flag not defined), `off_present`
(`-DHK_GEMM_RS_MI300X_TRACE=0`), `on` (`=1`).

**Flag-present-and-0 is byte-identical to flag-absent.** Same object size to
the byte (612,896 B both; the ON object is 633,528 B), and identical
VGPR/AGPR/SGPR/scratch/spill/LDS tuples on all 7 instantiations, each matching
the post-exp_14 M2 table: `32/64/128`→98, `64/128/64`→104,
`128/192/32+tail`→136, `256/256/32`→246, `256/256/32+tail`→248, generic
`32/64/64`→91 and 92, with zero AGPRs, zero scratch and zero VGPR spills.
The instrumented code shape has not leaked into the production build.

**Flag-ON cost, disclosed.** The diagnostic arm is cheap enough that the
static-slot fallback is not needed:

| BM/BN/BK/tail | OFF VGPR | ON VGPR | Δ | ON AGPR | ON scratch | ON VGPR spill |
|---|---:|---:|---:|---:|---:|---:|
| 32/64/64 | 91 | 93 | +2 | 0 | 0 | 0 |
| 32/64/64 tail | 92 | 94 | +2 | 0 | 0 | 0 |
| 32/64/128 | 98 | 101 | +3 | 0 | 0 | 0 |
| 64/128/64 | 104 | 105 | +1 | 0 | 0 | 0 |
| 128/192/32 tail | 136 | 139 | +3 | 0 | 0 | 0 |
| **256/256/32** | **246** | **247** | **+1** | 0 | 0 | 0 |
| **256/256/32 tail** | **248** | **250** | **+2** | 0 | 0 | 0 |

The two rows that were the real risk — shapes 5 and 6 sit at 246 and 248 of the
256 arch-VGPR cap — absorb the ring pointer and the event counter in +1 and +2
registers and stay under the cap with 9 and 6 to spare. No scratch, no VGPR
spills, no AGPRs anywhere.

**One correction to this experiment's own gate, worth recording.** The first
run failed all 14 rows on `SGPRs Spill != 0` — including the untouched baseline
build, which reports 54-88 SGPR spills on every instantiation. That is a
pre-existing property of the kernel, not of the patch: the M2 contract is "zero
AGPRs, zero scratch, zero **VGPR** spills", and `ScratchSize` is 0 on every
row, so those scalar spills go to VGPR lanes and never to memory. The gate now
requires zero on `agpr`/`scratch`/`vgpr_spill` only, and requires SGPR spills to
be *identical between the two flag-OFF arms* rather than zero — which is the
question parity actually asks. Flag-ON raises SGPR spills by 11-18 per row,
still with zero scratch.

## Phase 2 — patch APPLIED, parity gate PASSED, waiting on the GPU

The kernel-edit gate was opened and the patch landed: **21 sites, +109 lines,
0 deletions**, all inside `#if HK_GEMM_RS_MI300X_TRACE`, **default 0**.
`design.md` §9 records the anchor resolution — the file had grown from 893 to
951 lines under exp_26, every planned line number shifted, three of this
document's own anchors were recorded at the wrong indentation and were caught
by the anchor check rather than by a mispatch, and one planned site turned out
to need no edit because aggregate initialization already value-initializes the
new trailing member to 0.

## Phase 2 (GPU): node sanity PASS, arm (a) COMPLETE, arm (b) queued

### Node sanity — PASS, but only after the gate was pointed at the right statistic

exp_22 is the first campaign to run after the exp_26 M9 run took a
`VM_L2_PROTECTION_FAULT` and left a KFD entry wedged in `exit_mm` holding
~1.25 GB per GPU at zero CU occupancy. The instruction was to re-verify a known
value before trusting anything, so the lease opens with a full M7 at the
flag-OFF production build (`harness/build/gemm_rs_mi300x.so`, used as-is and
never rebuilt — the binary that produced the reference vector is the binary
that should reproduce it).

**The first version of this gate FAILED, and the failure was the gate's, not
the node's.** It compared M7 to the best-of-arm vector
`62.38 / 64.52 / 83.75 / 198.71 / 613.70 / 1616.63` and flagged shape 5 at
+5.3% (646.10 vs 613.70). That is verbatim the trap at `LESSONS.md:189`: *the
recorded per-shape denominator vector is BEST-of-arm, while gate M7 prints
MEANS* — exp_05's own table records shape 5 as `613.70 / 645.93` = best/median,
and comparing the two manufactured a phantom regression once already, for a row
whose device code had not changed. A best over three rotations is also a weaker
order statistic than a best pooled over a multi-arm campaign, which is why
*every* shape drifted up 1–5% while shape 6 drifted *down* 2.9% — a mixed sign
pattern that no node degradation produces.

Re-scored against like-for-like statistics (`sanity.json`, `exp22.sanity.v2`):

| statistic | measured | recorded | delta | verdict |
|---|---:|---:|---:|---|
| geomean of M7 means | 207.76 µs | 207.18 µs (`LESSONS.md:598`, the E4b landing) | **+0.28%** | ok |
| shape 5 mean | 648.11 µs | 645.93 µs (exp_05 median) | **+0.34%** | ok |
| shape 5 in same-config range | 648.11 µs | 644.71–669.39 µs (`LESSONS.md:194`) | inside | ok |
| correctness, all six shapes at `2e-3` | True | — | — | ok |

**The node reproduces the ratchet to 0.3% on the like-for-like statistic.** The
stale KFD entry's predicted timing effect of nil is confirmed, not assumed. The
best-vs-best column is retained in `sanity.json` as context and explicitly not
gated. Shape-5 per-rotation spread was 2.10 µs (0.3%), i.e. the instrument is
tight; the historical run that produced the 613.70 best had a mean of 645.93,
so its spread was ~5%, which is the fat lower tail LESSONS warns about.

### Arm (a) — reference GEMM+RCCL, COMPLETE

`events_b0_reference.json`, `b0_kernel_map.json`. Shape 5, rank 0 traced with
`rocprofv3 --kernel-trace`, the other seven ranks unprofiled so the collective
had real peers. 1506 dispatches, nine distinct kernel names, **zero
unclassified after the rule correction below**.

The chosen epoch (median of 13 usable, durations 512.0 / 527.9 / 669.9 µs
min/med/max) is three intervals and they are **strictly serialized with zero
overlap** — which is the entire point of the arm:

| interval | µs | resource | kernel |
|---|---:|---|---|
| 0.0 → 243.2 | 243.2 | mfma | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x224x64_MI16x16x1_…` (rocBLAS) |
| 243.2 → 286.5 | 43.4 | hbm | `at::native::elementwise_kernel_manual_unroll<128,8,…>` (bias) |
| 286.6 → 527.9 | 241.3 | xgmi | `ncclDevKernel_Generic_2(ncclDevKernelArgsStorage<4096ul>)` (reduce-scatter) |

46% MFMA, 8% HBM, 46% xGMI, and **each resource is idle while the other two
run**. Host cross-check: rank 3's own device median was 573.4 µs against the
traced rank-0 epoch of 527.9 µs; rank 0 is the profiled rank and the difference
is not used for any performance claim.

**Two rule corrections, both derived from names actually observed**, recorded
because the requirement was to report rather than bucket:

1. `ncclDevKernel_Generic_2` — 30 dispatches, 11.07 ms, **the single most
   important kernel in the arm** — matched no seed rule and landed in
   `unclassified`. The seeds looked for `rccl*`/`ncclkernel`; ROCm's RCCL
   exports the upstream NCCL symbol names. Added `^nccl`. The
   report-don't-bucket rule is what surfaced this; a silent fallback would have
   produced a figure with an empty xGMI strip for the reference arm.
2. `__amd_rocclr_copyBuffer` (866) and `__amd_rocclr_fillBufferAligned` (552)
   are HIP runtime allocator blits from input setup, not operator work. They
   are now class `runtime`, **excluded from epoch segmentation and from the
   strips, and reported**. Left in, they cut the trace into 718 fragments and
   no epoch contained both a GEMM and a collective — the first capture aborted
   on exactly that. Excluding them leaves 88 operator dispatches → 36 epochs,
   13 usable.

### Arm (b) — built and correct-by-construction, queued on the lease

The diagnostic build now compiles and loads (`gemm_rs_mi300x_trace.so`,
429,264 B). One defect found and fixed, in the *verification*, not the build:
the smoke test loaded the module under the spec name `t`, and CPython resolves
a C extension's init symbol as `PyInit_<last component of the spec name>`, so
it failed with `does not define module export function (PyInit_t)` — which
reads like a build failure and is not one. `trace_run.py` already used the
correct name.

**Arm (b) has not run.** After arm (a) completed I released the lease, and it
was immediately taken by **exp_24**, with **exp_21** and **exp_26** also
queued. Our job is waiting its turn behind them, which is the lease working as
designed; it has been re-armed with a 4-hour outer window (`timeout 14400`) so
the queue wait cannot consume its run window, and it will execute arms (a) and
(b) unattended when the lease frees. Nothing was stolen and the stale KFD entry
was never signalled.

### Tick calibration: the independent number already exists

exp_21 landed while this arm was queued and it measured the same instruction on
this node directly: **`s_memrealtime` = 99.7366 MHz** (10.0264 ns/tick), 5 reps,
spread 0.0101%, two-stage against `steady_clock` in a dedicated kernel
(`exp_21_saturation/saturation.json:tick_rate_hz` = 99,735,808 Hz,
`LESSONS.md:1051`). That is **−0.264% from the sibling's declared gfx950
100 MHz**, so the sibling's figure is close but not exact here, and the axis
should use the measured value.

`trace_run.py` now carries that number as `EXP21_TICKS_PER_US` and cross-checks
its own in-situ regression against both it and the 100 MHz hypothesis,
emitting a per-reference `agree`/`DISAGREE` verdict at a 1% threshold into
`tick_rate.json`. The two methods share no machinery beyond the instruction:
exp_21 times a dedicated kernel against `steady_clock`, this arm regresses the
production kernel's own device span in ticks against hipEvent microseconds
across three shapes of very different duration. **Agreement would make the
x-axis the best-supported quantity in the figure; disagreement blocks the
figure**, since a wrong rate does not distort the plot visibly — it silently
rescales the entire time axis.

### One process trap worth recording: CRLF

The second capture attempt died with `syntax error: unexpected end of file` and
`$'fi\r': command not found`. Every `.sh` and `.py` in this directory had been
written with CRLF endings from Windows. Simple line-per-command scripts tolerate
it; anything with a multi-line construct does not. All files are normalized to
LF and normalization now runs before every push. This is the trap the root
charter flags, and it cost one capture.

## Policy, stated up front and binding on everything below

- The instrumented build is a **diagnostic arm**. `HK_GEMM_RS_MI300X_TRACE` is
  OFF for every campaign timing run and **no microsecond it produces is ever
  reported as a performance number**. The single exception is the ON-vs-OFF
  perturbation estimate, which is labelled as such.
- The MFMA row is **"CTAs in MFMA phase — an occupancy proxy"**, never MFMA
  utilization, and its denominator is `304 − NR` (272 on shape 5), printed in
  the axis label and in every row of `timeline_bins.csv`.
- Rank 0 is plotted; rank-max is supplementary; **rank symmetry is stated and
  quantified, never averaged away**.
- One process drives all eight devices in arm (b) — this is not the
  evaluator's topology, and any comparison against evaluator-measured numbers
  carries that caveat.

## Schemas (to be restated verbatim with the data)

- `events.json` — `exp22.events.v1`. Common envelope: `arm`, `kind`
  (`phase_ring` | `kernel_trace`), `shape`, and either `ranks.<r>.events`
  (flat `[cta, phase_id, ticks]` triples, `ticks_per_us` alongside) or
  `intervals` (`{name, resource, begin_ns, end_ns}`).
- `timeline_bins.csv` — `arm, bin_index, t_us_start, t_us_end, mfma_frac,
  mfma_denominator, hbm_gbs, xgmi_gbs, tflops`.
- `b0_kernel_map.json` — `exp22.b0_kernel_map.v1`: `rules`, `classified`
  (name → count/ns/resource/matched_rule), and **`unclassified`**, which is
  reported here rather than silently bucketed.
- `parity.json` — `exp22.parity.v1`: per-arm, per-instantiation
  VGPR/AGPR/SGPR/scratch/spill/LDS tuples plus the verdict.
- `tick_rate.json` — `exp22.tick_rate.v1`: the regression points, the measured
  `ticks_per_us`, and the sibling's 100 MHz hypothesis recorded for comparison.
- `validation.json` — `exp22.validation.v1`: the ±10% integral report per arm.

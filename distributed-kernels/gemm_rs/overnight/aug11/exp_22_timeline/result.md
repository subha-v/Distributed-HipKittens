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

## Phase 3 — arm (b) LANDED. Fig 3 ships with two arms.

Re-run in full after exp_26 landed `HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE=2`,
which changes `rgroup` on shape 5 — the traced shape — from 1 to 2. Provenance is
now stamped inside every `events_ours_*.json` (`module_sha256`, `source_sha256`,
`captured_utc`) so a capture can never again be of an unprovable build. Source
`gemm_rs_mi300x.cpp` sha256 `080b5719…ffa994`, `TRACE` reads 0 in the committed
source, ON only for `build/gemm_rs_mi300x_trace.so`.

### Gates

| gate | result |
|---|---|
| flag-OFF resource-tuple parity, re-asserted | **PASS**, 0 failures. `off_absent` ≡ `off_present` on all 7 instantiations; both match the incumbent `98/104/136/246/248/91/92` with zero AGPR, zero scratch, zero VGPR spill |
| ON cost (disclosure only) | +3 VGPRs (e.g. row 3 `136 → 139`), +12 SGPR spills, `ScratchSize` still 0, zero VGPR spills |
| correctness, traced build, `1e-2` and `2e-3` | **PASS** both, `max|diff| = 4.88e-4`, on s5, s6 and s2 |
| error bits / epoch cells / signals | clean; `protocol_problems: []` |
| ring integrity | **0 drops**, 304/304 CTAs seen, 0 aborted, on all three shapes |
| tick regression vs exp_21 | 99.8879 vs 99.7358 ticks/µs, **+0.153%, AGREE** (r² = 0.99984) |
| tick regression vs sibling 100 MHz | **+0.112%, AGREE** |
| integral conservation | PASS, residuals ~1e-16 — see the caveat below |

Three independent calibrations of `s_memrealtime` on gfx942 now sit within
0.27% of each other: exp_21's dedicated kernel against `steady_clock`
(99.7358), this arm's in-situ regression of the production kernel's own span
against hipEvent time (99.8879), and the sibling's declared gfx950 100 MHz. The
x-axis is the best-supported quantity in the figure.

### The integral check passes, and it proves less than its name suggests

Residuals are `+3.4e-16 / -1.3e-16 / 0.0` on `hbm_bytes / xgmi_bytes / flops`.
That is floating-point zero, and it should be: bytes per phase are analytic on
the host and then **distributed across bins in proportion to interval overlap**,
so re-summing them recovers the input by construction. **This is a conservation
check of the binning arithmetic** — it proves no mass is lost or double-counted
at bin edges, which is a real defect class and is now excluded — **but it is not
an independent validation of the traffic model.** Stated plainly here so no
reader mistakes ±0.00% for agreement between two instruments. The ±10% tolerance
is vacuous against this check; the checks that can actually fail are below.

### What actually cross-checks the model

**The derived xGMI rate is physically admissible and lands where it should.**
Our emit strip peaks at **420.4 GB/s** against ~448 GB/s of per-GPU xGMI — under
the ceiling, and close enough to it that a wrong byte model would very likely
have punched through. The reference arm's RCCL peaks at **244.4 GB/s** on the
same axis from a separately-computed model, so **our epilogue reaches 1.72× the
collective's peak egress rate** while moving the same 58.72 MB off-rank.

### The figure's central result, quantified

| shape 5 | ours | reference GEMM+RCCL |
|---|---:|---:|
| bins with MFMA proxy > 5% **and** xGMI > 1 GB/s | **17 of 68 (25%)** | **0 of 53 (0%)** |
| MFMA occupancy proxy, peak / mean | 1.00 / 0.46 | 1.00 / 0.46 |
| xGMI GB/s, peak / mean | 420.4 / 86.4 | 244.4 / 110.8 |
| HBM GB/s, peak / mean | 3474.6 / 1529.7 | 3243.9 / 700.4 |

The reference never once overlaps compute with communication; we do it in a
quarter of the epoch. The MFMA row is **"CTAs in mainloop phase — an occupancy
proxy"** with denominator **`304 − NR` = 272** on shape 5 (`mfma_denominator`
is a column in `timeline_bins.csv`, not just an axis label).

### The credit-wait interval: the two instruments DISAGREE, by 25–58×

This is the item that was pre-registered as the thing to look for, and it did
not come out as predicted. Per-CTA phase sums, rank 0, mean over the CTAs that
run the phase (producers run concurrently, so the mean is the quantity
comparable to a whole-kernel ablation pool; max is the tail beside it):

| phase | s5 CTAs | s5 events | s5 mean µs | s5 max | s6 mean µs |
|---|---:|---:|---:|---:|---:|
| mainloop | 272 | 512 | 309.6 | 372.5 | 1224.4 |
| **credit_wait** | 272 | 512 | **2.1** | 3.8 | **2.9** |
| emit | 272 | 512 | 149.2 | 304.8 | 202.0 |
| release | 272 | 272 | 52.1 | 91.3 | 39.5 |
| ready_wait (reducers) | 32 | 64 | 613.7 | 636.6 | 1743.2 |
| reduce (reducers) | 32 | 64 | 35.2 | 42.8 | 46.3 |

exp_20's `sync` pool is 52.9 µs on s5 and 168.6 µs on s6. The ring says the
producer's reuse-credit stall is **2.1 µs and 2.9 µs** — 0.04× and 0.02×. The
instrument is not broken: the pair fires on every producer, 512 and 1024 times,
with zero drops, and the stamps bracket exactly `m3::wait_reuse_credit` plus the
`__syncthreads()` that broadcasts its result (verified in source, not assumed).

**The two are not measuring the same quantity.** exp_20's `sync` gate is a macro
ablation: it deletes the whole signalling path — producer credit wait, reducer
ready-wait, and the release/publish ordering — and lets the schedule re-form,
so its 52.9 µs is the cost of the *mechanism*, not of the producer's discrete
stall. The ring says the producer almost never blocks on a credit. The
back-pressure lands instead in two other places the ring can see: `emit`
(149.2 µs, and 58.72 MB over that interval is ~394 GB/s aggregate, i.e. the
interval is bandwidth-shaped) and `release` (52.1 µs, which contains the
`vmcnt(0)` drain of the in-flight peer stores).

**Consequence for the figure — and it is the good one.** The credit-wait pair
was added so a stall could not hide inside the emit interval and deflate the
xGMI strip. It shows the possible inflation is ~2.1 µs of a 149.2 µs interval,
**≈1.4%**. The xGMI strip is honest to within 1.4% for that specific concern,
and now that is measured rather than assumed.

**Consequence for Phase 2 — flagged upward.** Anyone reading exp_20's table and
setting out to attack a 52.9 µs "sync" pool would be attacking a producer stall
that is 2.1 µs. The addressable mass in that neighbourhood is in `release` and
in the tail of `emit`, not in credit acquisition.

### exp_26's release change is directly visible in the trace

On shape 5 the ring records **512 emit events but only 272 release events —
exactly 1 per producer against ~1.9 tiles per producer**, i.e. one release per
two tiles. That is `rgroup = 2`, observed rather than asserted. Shape 6 shows
1024 emits to 256 releases, one per four tiles, matching the shipped
`RELEASE_GROUP=4` there. **A reader comparing this release strip against
exp_20's 65.4 µs release attribution must know it predates exp_26**, which cut
shape 5's release count in half for −6.56% (≈ −43 µs); the strip plotted here
is the shipped behaviour.

### Arm (c): NOT taken. Fig 3 ships with two arms, and says so.

The lease was free when arm (b) landed. I did not take it. Arm (c) needs the
frozen rank-1 submission running under `rocprofv3`, and that integration is the
one thing in this tree with a history of consuming multi-hour time-boxes without
producing a number. Fig 3 is complete and internally cross-checked with two
arms, and the contrast it exists to draw — 25% overlap against 0% — is fully
carried by those two. A third arm is worth having; it is not worth risking a
rushed one against a finished figure. Available as a separate dispatch.

### Arm (b) — build notes (from the first attempt)

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

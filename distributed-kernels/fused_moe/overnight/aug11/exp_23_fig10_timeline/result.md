# exp_23 result — instrument designed, tooling built and self-tested, NOT YET RUN

**Verdict: GO at Tier A, conditional GO at Tier A+B, NO-GO at Tier A+B+C until
the parity gate is green.** Zero GPU was used (another agent holds the lease).
Everything in this folder is CPU work. `k0pf6gm_device_tile_mps.hip` and
`moe_mps_adapter.cuh` were **read only**; the kernel change is delivered as
`patch_spec.md`, which is a ~20-minute mechanical apply once those files free up.

| deliverable | state |
|---|---|
| `patch_spec.md` | **DONE.** Kernel + adapter + host edits, anchored on unique strings with working-tree line numbers, three-tier build ladder, parity gate definition, go/no-go. |
| `tools/parse_events.py` → `rank0_events.json` | **DONE, self-tested.** Schema `exp23-events-1`. |
| `tools/bin_timeline.py` → `timeline_bins.csv` | **DONE, self-tested.** Schema `exp23-bins-1`. |
| `tools/bytes_model.json` | **DONE.** Schema `exp23-bytes-1`, per-entry confidence, `null` where a device counter is required. |
| `tools/xcheck.py` | **DONE, self-tested.** Schema `exp23-xcheck-1`. |
| `tools/selftest.py` | **DONE and PASSING** — 8 mutation classes, every verdict proven to flip. |
| `tools/e23_gate_build.sh` | **DONE.** 4-TU CPU parity gate, G0–G7. Not executed: the TU inputs do not exist until the patch is applied. |
| `tools/e23_smi_sample.sh` | **DONE.** 2 Hz counter sampler, rate justified by a measured 0.14–0.29 s/sample. |
| data (`rank0_events.json`, `timeline_bins.csv`) | **NOT COLLECTED** — needs the patch and a GPU slot. |

---

## 1. The instrument, in one screen

A **fixed-slot per-CTA stamp array** in the spare tail of the existing
`K0P6_D_MPS_STATE` buffer (descriptor slot 60), 256 CTAs × 16 slots × 8 B =
32 KB. Cell `(bid, phase)` holds the raw 64-bit `s_memrealtime` value of the last
crossing of that boundary by that CTA. Written by `tid == 0` only, behind the
existing `cfg.timestamps` flag (packed bit 33 of `K0P6_D_MPS_CFG`), read by the
host after the launch and printed as `[MPS E23 CTA] <bid> v0..v15`.

**How it stays free when off**, itemised, because this is the requirement the
figure's validity rests on:

1. **No new descriptor slot** → no ABI change, no `mps_host_bridge.cpp` edit, no
   pybind rebuild, and the `desc_mps.numel() == 63` assert is untouched. One host
   line grows the tensor.
2. **No new pointer and no new descriptor read at any Tier-A site.** The array
   base is the u64 base each stamp site *already computes*, plus a compile-time
   constant.
3. **No new flag.** `cfg.timestamps` is already decoded at every Tier-A site.
4. **No atomics.** Fixed slots need no write cursor, so the
   `flat_atomic_pk_add_bf16` census of 282 cannot move and there is no new atomic
   of any kind.
5. **No LDS.** The array is global; LDS must read exactly 155,496 B.
6. **No code in either MFMA span.** Every site is outside
   `n2p6gm_phase1_body`/`n2p6gm_mps_phase2_body`; neither of those files is
   touched, so the `v_mfma` census of 180 cannot move.
7. **Nothing crosses a phase boundary.** Every site re-derives its own values
   locally — the donor discipline the existing stamps already follow — so no new
   live range exists at the phase-2 pointer peak, which is where this kernel's
   documented spill cliff lives.
8. **No M0 zeroing.** Cells are never reset; the parser detects a stale cell
   instead of trusting a reset. One fewer edit, one fewer register-risk site.
9. **Scalar predicates.** Tier B/C decodes are wrapped in
   `__builtin_amdgcn_readfirstlane` so the branch stays off the vector path,
   exactly as `k0pf6gm_device_tile_mps.hip:1731` already does and for the same
   reason.
10. **Overflow is structurally impossible** — fixed slot, `bid < 256` already
    shape-guarded, phase ids compile-time constants. A boundary crossed more than
    once per epoch is last-write-wins by design, with the repeat count in slot 15.

**When on**, the marginal cost is 5 stores (Tier A), 8 stores + 2 config decodes
(Tier A+B), or 14 stores + 5 decodes (Tier A+B+C), all by `tid == 0`, all outside
both MFMA spans, once per epoch. The array's 32 KB is written 8 B at a time and
never read by the device.

**Occupancy is asserted, not assumed** (gate G2). The sibling hazard — LDS
shrinking, by-reference pointer arrays migrating LDS→scratch, occupancy rising
1→2 — cannot arise here because nothing shrinks LDS, but the gate prints
`Occupancy [waves/SIMD]` and requires it to equal 1 regardless.

---

## 2. Schemas

### `rank0_events.json` — schema `exp23-events-1`

Top level: `schema, arm, cfg, head, src_rev, tier, rank, tick_ns (=10),
tick_us (=0.01), grid_ctas, slots (=16), phase_slots, intervals_def,
mfma_intervals, epoch_window{lo_ticks,max_ticks,stale_cells}, origin_ticks,
ctas[], coarse{}, spin{}, e2e{}, checks{}, verdict{pass,failed}`.

Each `ctas[]` entry: `bid`, `role` (`compute`/`service`, inferred from
`M7_DONE == 0` when slot 15 is absent), `meta_count`, `stamps{name: ticks}`,
`stamps_us{name: µs from origin}`, `intervals[{phase, t0_us, t1_us, dur_us}]`.

Intervals: `dispatch` = KSTART→M2_DONE, `plan` = M2_DONE→M5_DONE,
`M6` = M5_DONE→M6_DONE, `M7` = M6_DONE→M7_DONE,
`service` = SVC_ENTER→SVC_EXIT, `m75` = M75_ENTER→M75_EXIT,
`combine` = M8_ENTER→REDUCE_DONE, `tail` = REDUCE_DONE→M9_DONE. An interval is
emitted only when both endpoints are live, so a mode that skips a phase simply
has no interval for it — the figure is never told a phase lasted zero when it
was in fact unmeasured.

### `timeline_bins.csv` — schema `exp23-bins-1`

Header + one row per (arm, bin): `arm, bin_index, t_start_us, t_end_us,
grid_ctas, ctas_dispatch, ctas_plan, ctas_m6, ctas_m7, ctas_service, ctas_m75,
ctas_combine, ctas_tail, mfma_frac, phase_coverage_frac, bytes_hbm, bytes_xgmi,
hbm_gbps, xgmi_gbps, bytes_confidence`.

Two properties a plotter can rely on:

* **CTA counts are fractional.** A CTA covering 30 % of a bin contributes 0.3,
  so a strip integrates to true CTA-microseconds and is **bin-width invariant**
  (self-tested at 5, 10 and 20 µs).
* **`null` bytes blank the cell.** Where the byte model has no number, the GB/s
  and bytes columns are **empty strings, never 0** — self-tested, because a
  silent zero in a bandwidth strip is a fabricated measurement.

`bytes_confidence` is the worst confidence among the phases active in that bin,
ordered `exact_from_shape < estimated < partial < absent`. **A plotter must hatch
or otherwise mark every non-`exact_from_shape` series.**

### `mfma_frac` is an occupancy proxy — the caption must say so

It is "fraction of the 256 CTAs whose current phase is M6 or M7". A CTA stalled
on `vmcnt` inside M7 counts. It is **not** MFMA utilisation and it is not
comparable to a hardware MFMA-busy counter without the caveat in §5.

---

## 3. Validation check 1 — the integral check, and its four tolerances

The check is deliberately split, because the four parts have wildly different
error bars and a single tolerance would be wrong for three of them.

| part | statement | tolerance | why exactly this |
|---|---|---|---|
| **3a coarse reconciliation** | `max over CTAs (per-CTA cell)` == the `[MPS TS]` coarse cell, for M2_DONE / M5_DONE / M6_DONE / M7_DONE / REDUCE_DONE | **0 ticks** | The coarse cell is an `atomicMax` over the *same clock read* that fed the per-CTA cell (that is what the fused `ts_mark` is for). Equality is a **theorem**, not a measurement. This is the strongest check in the instrument: it proves the array is indexed correctly, that the cells are from one epoch, and that the implementer used `ts_mark` rather than `ts_last(); e23_mark()`. A nonzero delta invalidates the run. |
| **3b interior closure** | `plan + M6 + M7 + combine == interior` on the rank-max reconstruction | **±1 µs** | Pure arithmetic on the same five cells. exp_33 measured this closing to 0.0 µs in all 10 rotations. Anything above 1 µs is a code bug, not noise. |
| **3c per-CTA coverage** | for each CTA, the union of its intervals covers ≥ 99.5 % of `[first stamp, last stamp]` | **0.5 % of the CTA's own span** | Overlapping intervals are unioned first, so overlap cannot manufacture coverage. A larger hole means a boundary is unstamped — i.e. a mark was dropped or misplaced — and the strip would silently show idle CTAs that were in fact working. |
| **3d end-to-end** | `(max REDUCE_DONE − min KSTART) × 0.01 µs` vs `summary.json arm_p50_us[arm].median` | **±10 %** | These are two different measurements of one regime: the stamps are the **final soak epoch**, `arm_p50_us` is a timed-iteration rank-max. The residual is already measured — `p50 − interior = 641.7 µs` on a ~6,496 µs arm = **9.9 %** (`PLOTS.md` caveat 2) — and is instrument-mixed dispatch/launch/skew. A tighter gate would fail for a reason already understood; a looser one would not catch a real error. |

Plus **3e binning self-consistency**: `sum over bins of bytes` == the model total,
tolerance **0.5 %** (pure arithmetic — the binner derives its rate from the
model, so this catches a binning bug, not a physics error), and **3f
CTA-microsecond conservation**: binned CTA-µs == source CTA-µs to 1e-9 relative.

All six are implemented; the self-test proves each one can fail.

---

## 4. Validation check 2 — the counter cross-check. **The plan's version is not performable.**

**Measured on the node (`amd-smi` 26.2.2, ROCm 7.2.4, 2026-08-12, read-only
queries — no GPU job, no lease taken):**

| probe | result |
|---|---|
| `amd-smi metric -g 0 --xgmi --csv` | `gpu,xgmi_err` → `0,N/A` |
| `rocm-smi --shownodesbw` | `0-0` for every GPU pair; the header says `Format: min-max; Units: mps` — a **topology capability field, not a counter** |
| `amd-smi metric -g 0 --csv` live utilisation fields | `gfx_activity`, `umc_activity` (percentages); `bandwidth`, `current_bandwidth_sent/received` are **PCIe**, i.e. the host link, not the fabric |
| per-sample cost | 0.14 s, 0.26 s, 0.29 s over three runs |

**Conclusion: there is no live xGMI throughput counter on this node.**
`plan.md`'s "`amd-smi metric` per-link xGMI throughput matches the xGMI strip's
epoch average (±20 %)" cannot be executed. What replaces it:

| check | window & rate | tolerance | what it proves / does not prove |
|---|---|---|---|
| **X1 HBM occupancy** | the 600-epoch soak (~3.9 s at 6.5 ms/epoch), **2 Hz** (3 Hz is the measured ceiling; 2 Hz is safe) → ~8 in-window samples, plus ≥3 idle samples either side to key `--window` off | **±15 percentage points, absolute** | Compares `umc_activity`'s duty cycle to the timeline's time-weighted fraction of the epoch spent in HBM-bearing phases. Both are *fractions of wall time*, not GB/s — that is the honest comparison the available counter supports. ±15 pp because `umc_activity` is a sampled UMC duty cycle and the timeline side is phase occupancy; they are different quantities that should agree in magnitude, not in the third digit. |
| **X2 window negative control** | same | `gfx_activity ≥ 95 %` | `gfx_activity` **saturates** during the soak, so it cannot validate the MFMA strip. It proves only that the window contains no idle time — i.e. that framing the soak as back-to-back epochs is sound. Stated as a negative control, not as agreement. |
| **X3 fabric ceiling** | no sampling | peak `xgmi_gbps` **≤ 52.8 GB/s** | aug10 exp_21's fabric ubench: 52.8 GB/s for coalesced 4 B atomics (mode 12's payload form), 54.9 GB/s for 16 B stores. A strip implying more than the measured fabric ceiling is falsified on its face. Weak, free, real. |
| **X4 rocprof fabric bytes** — *specified, not implemented* | one instrumented launch | — | `rocprofv3` TCC EA read/write-size counters would turn X3 from a ceiling into a measurement. Needs a GPU slot **and** a counter-availability probe on gfx950, which I could not do without the lease. This is the one honest gap in the cross-check and `result.md` must keep saying so until it closes. |
| **X5 MFMA proxy validation** — *specified, not implemented* | one launch, `rocprofv3 --pmc SQ_VALU_MFMA_BUSY_CYCLES GRBM_GUI_ACTIVE` | sign + magnitude only | Validates the **label** ("CTAs in an MFMA phase" vs "MFMA issue"), not the shape. Do not promote it to a numeric gate. |

**What no cross-check here can do:** validate the intra-epoch *shape* of any
strip. At 2 Hz a 6.5 ms epoch is 1/300 of one sample. The shape is validated only
by 3a (exact reconciliation with the coarse stamps) and 3c (coverage). Say this
in the figure caption.

---

## 5. The `production` arm and the honest limits of the overlay

Full recipe in `patch_spec.md` §8. The short version:

* **Capture** a standalone 8-rank replay of one production MoE layer at the
  campaign's shapes under `torch.profiler`, keep `ph == "X"` kernel events from
  the last of three warm iterations, and map kernel name → resource class with a
  checked-in `b0_kernel_map.json`. **An unmapped kernel name is an error**, never
  a silent drop. This run is *not* gate-covered (a direct `torchrun` defaults the
  tolerance to 1.0 and `K0_PF6GM_G` to 2) and only its interval widths may be used.
* **Alignment:** both clocks are the 100 MHz HSA domain, so they share a *rate* —
  but they are different processes and different launches, so **there is no
  common origin and no absolute alignment exists.** Normalise each arm to its own
  layer start and share a *duration* axis, not a clock. A misaligned overlay is
  worse than no overlay.
* **Not comparable, and each must be visible:** (1) production's strips are
  binary whole-GPU kernel occupancy, the megakernel's is a fraction of 256 CTAs —
  different units, draw them differently; (2) the megakernel's MFMA strip is a
  phase proxy; (3) production has real host launch gaps inside the layer — do not
  smooth them; (4) production's GEMM is AITER, a different MFMA mix; (5)
  production's xGMI is MoRI/RCCL protocol traffic the byte model does not
  describe, so its xGMI strip must come from counters or be omitted; (6) the
  megakernel stamps are the final soak epoch, production's trace is a warm timed
  iteration — same regime, different iteration.
* **The one quantitatively comparable number** across all three panels is the
  **communication share of layer time**, because it is a ratio of intervals
  within each arm's own layer. That is the Fig-1 quantity and it is what the
  overlay should be annotated with.

### `pf6gm_mega` cannot be instrumented — the substitution

`plan.md` and `design.md` both assume the in-kernel instrument reaches
`pf6gm_mega`. It does not: `pf6gm_mega` is a different kernel with a **55-word**
descriptor (`e004pf_k0pf_ab.py` asserts `desc_gm.numel() == 55`) and no
`K0P6_D_MPS_STATE` slot at all. The homogeneous panel is therefore **`mps_mega`
at `C=0, g=1, mode=0, flush_rows=1`** — legal (`config_is_valid` rejects `C == 0`
only for stream modes and diag pools), all 256 CTAs run M7 at the donor's
full-grid stride, M7.5 does the parity bulk publication, M8 pulls the producer's
remote `part`. Same binary, same descriptor, same resource tuple, same clock as
the ratchet, which is scientifically the *better* control for a timeline. Cost in
honesty: it is not `pf6gm_mega`'s microseconds. **Report `pf6gm_us` and the
mode-0 `mps_us` side by side from the same screen and label the panel "bulk, no
overlap (mode 0)" — never "homogeneous baseline".**

---

## 6. GPU wall-clock estimate for all three arms

| step | GPU time | note |
|---|---|---|
| parity gate ladder (4 TUs) | **0** | CPU only, `subha_k1`, ~4 min wall |
| gate ladder for the instrumented build: correctness + negative control + 600-epoch soak, mode 12 `timestamps=1` | ~5 min | one screen `w1t1p1`; `[MOK GATE]`, `[MARK] control_fails=True`, `[MPS SOAK] 600/600`, `pperr=0` |
| same for mode 0 `C=0` | ~5 min | it is a new arm and owes the full ladder |
| `timestamps=0` parity smoke on the instrumented build | ~5 min | proves the shipped ratchet number is unmoved: expect within the measured ±0.5 % screen band of 0.8408× |
| data collection, 3 arms | ~5 min | the stamps come from the **final soak epoch**, so a screen suffices — **no 5-process campaign is needed for this figure** |
| amd-smi X1/X2 sampling | 0 extra | runs concurrently with one of the screens above |
| X5 rocprof MFMA validation | ~5 min | one launch, one arm |
| `production` torch-profiler trace | ~15–20 min | dominated by weight staging + JIT on a fresh process |
| **total** | **≈ 45–55 min of exclusive GPU**, strictly one job at a time | + ~30 min (CPU) if a tier fails the gate and the ladder has to be re-cut |

---

## 7. Go / no-go, and the risk that actually worries me

**GO at Tier A. Conditional GO at A+B. NO-GO at A+B+C until gated.**

| tier | P(parity gate fails) | dominant mechanism |
|---|---|---|
| A (5 marks, all in existing stamp blocks) | **~15 %** | whole-function allocation is not local; 5 extra stores can still shift a spill decision |
| A+B (+3) | **~30 %** | B1/B3 each add a descriptor read + config decode in a new scope |
| A+B+C (+6) | **~50 %** | the M7.5 block is documented **in the source** (`k0pf6gm_device_tile_mps.hip:1693-1700`) as an allocation cliff: a second copy of one release sequence there cost **+16 B/lane scratch, VGPR spills 15→21, SGPR spills 186→221** |

A parity failure is **loud** — the gate prints it and I drop a tier. The risk that
is **quiet**, and the one that worries me most, is **G7**: if the implementer
writes `ts_last(...); e23_mark(...)` instead of the fused `ts_mark(...)`,
everything compiles, the tuple may still pass, the figure looks right — and the
two calls read the clock at different instants, which silently demotes check 3a
from a theorem to a few-tick approximation. The one instrument that could prove
the array indexes the right cell in the right epoch would stop proving it, and
nothing downstream would notice. That is why G7 (`s_memrealtime` count in Tier A
must be **identical** to the reference) is a hard gate and not a diagnostic.

**Drop order if the gate fails**, most expendable first: C6 `META` → C3/C4/C5
(M7.5) → C1 `SVC_ENTER` → B3 `M9_DONE` → B1 `KSTART` → B2 `M8_ENTER` → A5
`REDUCE_DONE`. The floor that still produces a figure is **A1–A4**: per-CTA
`plan`/`M6`/`M7` intervals with the origin at `min(M2_DONE)`. If even Tier A
fails, **do not ship a timeline** — fall back to exp_33's coarse stamps and record
Q4b as blocked with the parity failure as the stated reason.

---

## 8. Self-test result (run on CPU, 2026-08-12)

`python3 tools/selftest.py` → **PASSED**, exit 0. Coverage:

* **Positive:** 256 synthetic CTAs at exp_33's measured magnitudes (plan 372.8,
  M6 2,453.3, M7 2,701.8, combine 324.2 µs), 16 service CTAs; all five parser
  checks pass; role inference correct; binner integral and CTA-µs conservation
  exact; `mfma_frac` peaks at 1.000 and never exceeds it; bin-width invariance at
  5/10/20 µs; the mode-0 (M7.5-bearing) shape parses too; full CLI round trip
  produces a 691-row CSV whose header equals the documented schema.
* **Mutations, each proven to flip its own verdict:** a 7-tick coarse/ring skew
  (the G7 failure mode) → `coarse_reconcile` FAILS and names `M6_DONE`; a mark on
  the wrong side of a barrier → `monotonic` FAILS; a 2,700 µs unstamped hole →
  `coverage` FAILS at 44.7 % uncovered while +8 % on the end-to-end still PASSES
  (tolerance calibration confirmed in both directions); a 50 ms-old cell →
  `stale_cells = 1`, verdict FAILS, and the stale cell is **excluded** from the
  CTA record; 5 % of bytes lost → `integral` FAILS; a uniform shift of every
  `M7_DONE` → `interior` correctly still PASSES (it is legal); 40-point
  `umc_activity` gap → X1 FAILS; `gfx_activity` 71 % → X2 FAILS; 61 GB/s → X3
  FAILS above the fabric ceiling; missing samples → **N/A, not a silent pass**.
* **Structural refusals:** no `[MPS E23]` header, header with no rows, wrong tick
  domain (`tick_ns != 10` — the 100 MHz-vs-2.2 GHz divisor trap), short row,
  all-zero cells (`timestamps=1` forgotten), duplicate CTA row.

---

## 9. Errors found in `plan.md` / `design.md`

Full list with citations in `patch_spec.md` §10. The four that would have cost
real time:

1. **`pf6gm_mega` is not instrumentable** (§5 above). The largest structural
   error in the plan.
2. **The `amd-smi` xGMI cross-check does not exist on this node** (§4). Measured,
   not inferred.
3. **The new descriptor slot is unnecessary** and would have pulled in
   `moe_host_abi.hpp`, `mps_host_bridge.cpp`, a pybind rebuild and an assert
   change — hours, in files nobody had budgeted. One host line does it.
4. **The ratchet identity is stale in both files**: `plan.md` says `C=64 g=1
   mode 2`; it is `C=16, g=353, mode=12, flush_rows=16`, and after exp_27 the
   number is 6,482.7 µs / 0.8408×, not `CLAUDE.md`'s 6,568 / 0.8522.

Also: `design.md`'s LDS figure (155,428 B) and `CLAUDE.md`'s standing tuple
(`SGPR 104 / LDS 155,428`) are both **stale**. The gate uses `STATUS.md:145`'s
`SGPR 106 / VGPR 256 / AGPR 256 / scratch 128 B / LDS 155,496 / MFMA 180 /
pk_add_bf16 282`.

And a correction to a comment in the kernel itself: `k0pf6gm_device_tile_mps.hip`
line ~707 says a `KSTART` stamp is "provably dead". That is true of the
**M0-zeroed coarse block** and **false** for an unzeroed per-CTA cell marked after
M0's `__syncthreads()`. The per-CTA epoch origin is recoverable.

---

## 10. Proposed `PLOTS.md` row (paste; this agent did not edit the shared file)

Replace the §6 **Q4b** row with:

> | §6 **Q4b** — per-layer utilization timelines (NanoFlow-v2 Fig-10 analog) | Fig 3 | per-CTA phase occupancy over time for three arms on one routing seed: RCCL-eager `production` (torch profiler) vs **bulk/no-overlap `mps_mega` mode 0 `C=0`** vs the `mps_mega` mode-12 ratchet | `exp_23_fig10_timeline/rank0_events.json` (schema `exp23-events-1`, one per arm) + `timeline_bins.csv` (schema `exp23-bins-1`) — neither exists | exp_23 | **PENDING.** Instrument fully specified (`patch_spec.md`: fixed-slot per-CTA stamp array in the spare tail of descriptor slot 60, behind `cfg.timestamps`, 3-tier build ladder), all host tooling built and self-tested on CPU with mutation coverage, parity gate script written. Owes: the kernel apply (~20 min, blocked on the mode-14 implementer), the CPU parity gate, and ~45–55 min of GPU. **`pf6gm_mega` cannot carry the instrument** (55-word descriptor, no MPS state slot) — the homogeneous panel is mode 0 `C=0` and must be labelled as such. **No live xGMI counter exists on this node**, so the xGMI strip's external check is a fabric-ceiling bound (≤ 52.8 GB/s, aug10 exp_21) until a rocprof TCC-EA probe lands. |

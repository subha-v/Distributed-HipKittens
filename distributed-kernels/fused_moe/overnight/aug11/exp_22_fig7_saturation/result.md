# exp_22 — resource saturation vs CTA count (paper Fig 2 / Q4)

**Status: BUILT, CPU GATE GREEN, AWAITING GPU LEASE.** No GPU work has been done
for this experiment. Everything below the "what runs when the lease arrives"
line is a prediction or a procedure, not a measurement, and `saturation.json`
does not exist yet.

Artifacts in this folder: `e22_saturation.hip` (the ubench), `build.sh` +
`build_in.sh` + `remark.py` + `isa_census.py` (CPU gate), `run_saturation.sh`
(the one command), `summarize.py` (jsonl → `saturation.json`, including the
mechanical H1–H4 adjudication), `selftest_summarize.py` (proves the
adjudication can fail), `tools/` (node setup, probe, gate driver, push helper).

---

## 1. `saturation.json` schema

`run_saturation.sh` writes two files. `saturation_<tier>.jsonl` is the raw log,
**one JSON object per point, appended and flushed as each point completes** —
that is what makes the sweep resumable. `saturation.json` is the plot-ready
document `summarize.py` builds from it:

```text
saturation.json
  schema_version      "exp22-saturation-1"
  n_points, modes_present, status
  record_schema       field-by-field description of a point record
  peaks_used          the ceilings behind every peak_fraction
  derived
    knees                     per series: smallest C reaching 90% of plateau
    concurrent_over_isolated  per pair_key: ratio + the overlap proof flag
    compute_slowdown          per pair_key: concurrent / reserved_only cmp TFLOPS
    protocol_over_payload     per (C, fanout, g): with-protocol / payload-only
    hypotheses                H1..H4 verdicts (see §4)
  points[]            every raw record
```

One point record (the required keys, plus what the benchmark-review standards
demand):

| field | meaning |
|---|---|
| `mode` | `mfma` \| `hbm` \| `xgmi` |
| `ctas` | resource-role CTA count C (mode `mfma`: total CTAs) |
| `mlp` | mode `xgmi`: `store_peer_packets_multi` depth ∈ {1,4,8}; mode `mfma`: MFMAs per 16 KiB staged tile ∈ {4,256}; mode `hbm`: 0 |
| `fanout` | `single` (one xGMI link) \| `rr7` (aggregate egress) \| `na` |
| `concurrency` | `isolated` \| `concurrent` \| `reserved_only` |
| `metric` | `GBps` \| `TFLOPS` |
| `value` | headline number |
| `value_source` | `wall` (isolated) \| `role_span` (concurrent) — stated per point, never inferred |
| `repeats` | timed rotations behind `value` (default 5, after 2 discarded warmups) |
| `median` | median of `values[]` |
| `spread` | `{p25, p75, min, max, sd, rel_iqr_pct}` of `values[]` |
| `peak_fraction` | `value` ÷ the device ceiling for that resource class |
| `values[]`, `values_wall[]`, `values_span[]` | per-rotation metric, both timebases |
| `wall_us[]`, `res_span_us[]`, `cmp_span_us[]` | per-rotation timings |
| `overlap_pct[]`, `overlap_pct_median` | % of the resource role's span during which the compute role was also resident |
| `cmp_tflops[]`, `cmp_tflops_median` | the compute role's own throughput in the same launch |
| `res_rows`, `bytes`, `flops`, `cmp_tiles`, `mfma_per_tile` | the work actually done, so every rate can be re-derived by hand |
| `protocol`, `proto_g`, `pushers`, `row_bytes`, `work_mode`, `heap` | the axes beyond the plan grid (§6) |
| `grid`, `threads`, `lds_bytes`, `rotations`, `warmup`, `src_rev`, `verified`, `status` | provenance; `status` is always `valid_diagnostic` (ubench) |

## 2. What is inside the timed region

`hipEventRecord(ev0, stream)` → **one kernel launch** → `hipEventRecord(ev1)`.
Nothing else: allocation, fill, poison, checksum, and the tile-rate calibration
probe all happen outside. Buffers are allocated once for the whole sweep, so no
point pays an allocation.

Two independent clocks are recorded for every rotation:

* **host `hipEvent` wall time** — includes launch and drain, used for every
  isolated point;
* **device role span** — `s_memrealtime` (100 MHz, device-coherent; the read is
  the `s_memrealtime` + `s_waitcnt lgkmcnt(0)` idiom copied from
  `moe_mps_adapter.cuh`, whose comment records why the wait is load-bearing),
  stamped by thread 0 of every CTA into a `[grid][4]` buffer as
  `{role, begin, end, work}`. The resource span is `max(end) − min(begin)` over
  the resource CTAs, and the compute span likewise.

**The concurrency proof.** A "concurrent" point that is secretly sequential is
the classic failure of this measurement, so it is instrumented rather than
argued: `overlap_pct = |[res_begin,res_end] ∩ [cmp_begin,cmp_end]| ÷ res_span`.
The compute role is sized from the *measured* isolated duration of the same
point × 1.35 (using a calibrated per-CTA tile rate, itself measured by a probe
launch), and if `overlap_pct` still comes in below 90% the point is re-run with
1.6× the compute work, up to twice. Every point carries its own
`overlap_pct_median`, and `summarize.py` sets
`concurrent_is_really_concurrent: false` on any point below 90% so no such point
can quietly become a figure.

Concurrent points report `role_span`-based throughput (the resource role's own
device span), which is immune to how long the compute role outlasts it. Isolated
points report wall time. Both series are always in the record.

## 3. Roofline sanity check (hand-computed, arithmetic verified on CPU)

The conversion is `GB/s = bytes ÷ (t_µs × 10³)` with GB = 10⁹, and
`TFLOPS = flops ÷ (t_µs × 10⁶)`.

**Point `xgmi | C=64 | mlp=4 | single | isolated`.** Per-CTA volume is 45 MiB, so
per-CTA rows = ⌊45·2²⁰ / 14336⌋ = 3,292; total rows are floored to a multiple of
`C × depth × (world−1)` = 1,792, giving 209,664 rows =
209,664 × 14,336 B = **3,005,718,528 B (2.80 GiB)** pushed per launch. At the
76.8 GB/s single-link ceiling that launch must take 3.0057e9 / 76.8e9 =
**39.1 ms**, and the reported value is then
3.0057e9 / (39,127 × 10³) = **76.82 GB/s → peak_fraction 1.000**. The formula
therefore returns exactly the ceiling when the hardware runs at the ceiling,
which is the check. Unit derivation:
`B ÷ (T_µs·10³) = B ÷ (T_s·10⁹) = GB/s` ✓.

**Point `hbm | C=256 | isolated`.** 48 MiB/CTA → 898,048 rows × 14,336 B × 9
streams (8 slot reads + 1 write) = **115.9 GB** per launch. At a plausible 6 TB/s
achieved that is 19.3 ms and 5,999 GB/s = **0.75 of the 8 TB/s spec peak**; if it
reports 8,000+ GB/s the arithmetic or the kernel is wrong, and if it reports
< 1,000 GB/s at 256 CTAs the reduce is latency-bound rather than
bandwidth-bound.

**Point `mfma | 256 CTAs | shape 256`.** One `v_mfma_f32_32x32x16_bf16` is
2·32·32·16 = **32,768 flops**, issued per wave; each CTA runs kWaves = 4 waves
that issue identical counts, so
`flops = Σ_CTA(tiles × mfma_per_tile) × 32,768 × 4`. **This factor of 4 was a
real bug found by doing this check** — only thread 0 of each CTA writes the work
stamp, so the naive sum understates TFLOPS by exactly 4× (fixed in
`E22_SRC_REV 7`). Against the 2,300 TFLOPS MI350X dense-bf16 spec peak, the
predicted per-CTA rate is 0.268 tiles/µs at shape 256; the sweep sizes tiles from
the measured rate, not this one, and reports `peak_fraction` so an implausible
number is visible on sight.

Peak references used (all recorded in `saturation.json.peaks_used`, all
overridable with `--peak-*`): 76.8 GB/s per xGMI link per direction, 537.6 GB/s
aggregate egress (7 × 76.8), 8 TB/s HBM3E, 2,300 TFLOPS dense bf16. The first
three come from plan.md; the last is the MI350X spec sheet. Mode `mfma` is an
LDS-fed capacity probe, **not** a tuned GEMM: if it lands near 2,000 TFLOPS it is
MFMA-bound, and if it lands far below with `ds_read_b128` in the loop it is
LDS-bound. Either way it answers H4, which is a claim about the *CTA axis*, but
the absolute number must be reported with that caveat and must not be quoted as
a device peak.

## 4. How H1–H4 will be judged

`summarize.py` adjudicates all four against thresholds frozen in the source
before any data exists, and `selftest_summarize.py` **proves each verdict can
flip** by running the summarizer over two synthetic worlds — one where the fabric
saturates early and payload rides free, one where it is bandwidth-limited and
interferes. All nine self-test assertions pass, so no hypothesis here is
unfalsifiable by construction.

| hypothesis | decision rule | what refutes it |
|---|---|---|
| **H1** single-link push saturates by 8 CTAs at MLP ≥ 4 | smallest C with `value ≥ 0.75 × 76.8` GB/s on `xgmi/single/isolated/mlp∈{4,8}`; SUPPORTED iff ≤ 8 | any C > 8, or never reaching 75% (→ `REFUTED_NEVER_REACHED`) |
| **H2** aggregate egress saturates by 16–32 CTAs | same rule against 537.6 GB/s on `rr7`; SUPPORTED iff ≤ 32; the 90%-of-plateau knee is reported alongside | C > 32, or a plateau below 75% of the ceiling |
| **H3** the concurrent curve is depressed, and the depression tracks *protocol* not payload | median of `concurrent ÷ isolated` over payload-only xGMI pairs; SUPPORTED iff < 0.90. The `--protocol` arm gives the companion ratio at the same C | a payload-only ratio ≈ 1.0 → `REFUTED_PAYLOAD_RIDES_FREE`, which is the outcome the paper's claim predicts and exp_20 implies |
| **H4** TFLOPS vs CTAs is linear (one block per CU) | R² of `y = a·x` through the origin over the 8-point CTA axis, both intensities; SUPPORTED iff ≥ 0.98 | any saturation or superlinearity in the CTA axis |

**Pre-registered falsifier (plan.md), also mechanical:** if the isolated
single-link curve needs ≥ 32 CTAs to reach 75% of 76.8 GB/s at *any* MLP depth,
`falsifier_triggered` is set true, and the reading is written into the JSON: the
pusher is bandwidth-limited rather than protocol-limited, A1's C=64 optimum is a
bandwidth artifact, and the M4-first priority inverts.

Note the prior this runs against. exp_03 measured the *protocol-carrying* service
pool as exactly linear in C out to 64 CTAs with no knee at all (~1.1–1.7 GB/s per
CTA), and exp_04 showed 4× MLP buys only ~20%, implying ~81% of service time is
bookkeeping. So the informative outcome is the pair: an isolated payload curve
that **does** knee early while the protocol arm stays linear. That pair is the
figure. If the isolated curve is *also* linear to 64 CTAs, the fabric — not the
protocol — is the binding constraint and much of the aug10 M-series reasoning
needs revisiting.

## 5. Correctness before any number

No bandwidth is reported for a configuration that has not passed a payload check
first (`--no-verify` exists but is not used by `run_saturation.sh`).

* **xGMI:** the destination partition is poisoned with `0xA5`, a small push is
  run at a size where every source row is read exactly once and every
  destination row written exactly once, and then an **order-independent XOR fold
  of the destination on every receiving device is compared to the fold of the
  source rows on the sender**, plus a count of surviving poison words that must
  be zero. A silently failing store cannot fake a fold.
* **HBM:** slots are filled with 2⁻ˢ so the 8-way sum is exactly 1.9921875 in
  bf16; every output element is compared exactly (`!=`, no tolerance) after a
  zero-fill.
* **MFMA:** the accumulators are folded into a per-wave sink store, so the loop
  cannot be dead-code eliminated; the ISA census confirms the MFMAs survive.

## 6. Deviations from plan.md / design.md, and what was underspecified

Everything here is a deliberate, recorded departure — the plan's grid is intact
as the default `--tier plan`.

1. **No mori symmetric heap, no MPI, no 8 processes.** The ubench is ONE process
   that owns all 8 devices and reaches peers through `hipDeviceEnablePeerAccess`
   plus `translate_peer` on a table of the 8 device base pointers — which is
   arithmetically what `translate_peer` does on the mori heap. design.md's
   buffer table assumes the symmetric heap; that would have meant linking the
   harness's mori/JIT stack, and the node checkout is off-limits tonight.
   **Fidelity caveat to keep:** mori's heap is HIP-VMM `HeapType::Uncached`, so
   the ubench allocates its symmetric-role buffers with
   `hipExtMallocWithFlags(hipDeviceMallocUncached)` by default (`--cached`
   switches, and `heap` is recorded per point). GEMM operands are allocated
   *cached*, because in the real kernel they are ordinary tensors, not heap
   memory. Whether uncached-vs-cached moves the fabric number is worth one A/B
   at lease time.
2. **Per-CTA constant volume instead of a fixed global work list** (`work_mode`
   axis, default `per_cta`). design.md fixes total work so "time differences are
   pure throughput", but the reported quantity is a *rate*, and a fixed 4 GB list
   would put a ~3 s launch at C=1 next to a ~5 ms launch at C=64 — 40× the sweep
   cost and two different noise regimes on one curve. `per_cta` holds every
   launch in the same window; `rr7` gets 7× the volume because it prices seven
   links. `--tier full` re-runs three CTA counts with design.md's fixed list as
   an explicit cross-check that the two agree.
3. **`s_memrealtime` role spans are per rotation, not once**, and the overlap
   fraction is derived from them; design.md described the buffer but not the
   overlap statistic, which is the only thing that actually proves concurrency.
4. **Clocks are sampled per attempt, not per point.** design.md asks for
   `rocm-smi --showclocks` before/after *each point*; at ~1 s per call that is
   400+ s of `rocm-smi` on a sweep whose kernels total ~60 s, and it would
   dominate the artifact. Clocks are captured before and after the sweep
   (`env_before.txt` / `env_after.txt`), and per-point drift is instead visible
   in the recorded rotation spread (`rel_iqr_pct`), which is a strictly better
   instrument for the thing we care about.
5. **Mode a is not the `n2_phase2_gm` M7 body.** design.md specifies the pinned
   phase-2 tile body fed by a synthetic tile_desc list. That body lives in
   `$AMD_MASTER/auto-gpu-kernel/...` behind the donor include chain and is
   reachable only through the megakernel translation unit another agent owns
   tonight. Substituted: a self-contained LDS-staged `v_mfma_f32_32x32x16_bf16`
   loop with an explicit arithmetic-intensity knob, run at `mfma_per_tile = 4`
   (≈32 flops/byte, which is M7's measured intensity: a 32×448 tile over K=2048
   is 2·32·448·2048 flops against (32·2048 + 2048·448)·2 B) and at 256
   (peak-seeking). The overlay partner uses an intensity that touches memory on
   purpose — a register-resident MFMA loop would show zero interference
   trivially and would rig H3 toward the answer we want.
6. **Three concurrency arms, not two.** The schema asked for
   `isolated|concurrent`; `reserved_only` (C reserved, resource role idle) is
   design.md's C-matched capacity twin and is what makes the compute slowdown a
   paired number rather than a comparison against a 256-CTA run — exp_20's
   lesson. All three arms are the same code object with the same registers and
   the same LDS, differing only in which role is given work.
7. **Two axes added beyond the plan** (both `--tier full` only, both cheap):
   `row_bytes = 896` because the production push unit is a 896 B slice
   (`kSliceBytes`), not a 14,336 B row, and the knee is a function of transfer
   size — this is the granularity rung mode 14 argues about; and `pushers = 8`,
   where every rank pushes at once, which is the fabric condition the real
   combine sees rather than one sender into a quiet fabric.
8. **plan.md's "5 rotations per point, ≥10 ms each" is met at 25–40 ms for most
   points but drops to ~5–8 ms for the fastest ones** (`rr7` at C=64). hipEvent
   resolution is ~1 µs so this is still ~4 significant figures; `--xgmi-mb-per-cta`
   raises it if the measured spread argues for it.
9. **Underspecified in the plan, resolved here:** the destination layout (the
   ubench partitions the destination by *source rank*, mirroring
   `slots[world][MAXTOK][H]`, otherwise concurrent senders collide);
   `flush_rows` for the `--protocol` arm (fixed at 16, the ratchet's value, with
   `g ∈ {2,16}` as the probe-count axis); and what "checksum read-back once per
   config" means for a cyclic destination (it means a separately sized verify
   pass — a fold over a cyclically overwritten buffer is not a check, because
   duplicate rows cancel in XOR).

## 7. Resource remark (CPU gate, `E22_SRC_REV 8`, ROCm 7.2.3, gfx950)

Binary sha256 `381b532f66b07f87…`, source sha256 `e5526141c6773af2…`,
`E22_SRC_REV 8`, scratch clone `$HOME/e22/DHK` at `ca5b683f`. Full log:
`cpu_gate.log` in this folder; grid census in `tools/grid_plan.txt`. The rev is
bumped on every edit and the binary is rebuilt from the recorded source hash, so
a stale-object null (exp_04's failure) cannot happen quietly here.

| kernel | SGPR | VGPR | AGPR | scratch B/lane | SGPR spill | VGPR spill | LDS B/block | occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sat_kernel_d1` | 102 | 100 | 0 | 0 | 0 | 0 | 98,304 | 1 |
| `sat_kernel_d4` | 106 | 103 | 0 | 0 | 2 | 0 | 114,688 | 1 |
| `sat_kernel_d8` | 106 | 103 | 0 | 0 | 26 | 0 | 131,072 | 1 |

`CPU_GATE: PASS` (LDS ≥ 98,304 at every depth, occupancy 1 wave/SIMD, zero
scratch, zero VGPR spill, all three depth instantiations present). AGPR 0 is
expected: `-mllvm -amdgpu-mfma-vgpr-form=1` keeps accumulators in ArchVGPRs.

ISA census (`bash build.sh isa`) — `ISA_GATE: PASS`:

| kernel | mfma 32x32x16 bf16 | 16 B stores | 16 B loads | ds_read_b128 | ds_write_b128 | scratch ops | s_memrealtime | nt loads | s_barrier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sat_kernel_d1` | 4 | 4 | 13 | 4 | 4 | 0 | 2 | 8 | 2 |
| `sat_kernel_d4` | 4 | 7 | 16 | 7 | 7 | 0 | 2 | 8 | 2 |
| `sat_kernel_d8` | 4 | 11 | 20 | 11 | 11 | 0 | 2 | 8 | 2 |

**Primitive finding, already worth recording (extends exp_04).**
`store_peer_packets_multi<Regions>` takes its pointer arrays **by reference**, so
the caller's `void*[Regions]` + `const void*[Regions]` pair has its address taken
and cannot stay in registers. Where it lands is decided by how much LDS is left
over, and both choices are bad in a different way. Measured here: with a 96 KiB
static `__shared__` array the compiler promotes it to **LDS**, exactly
16 B/lane/region (98,304 → 114,688 at depth 4 → 131,072 at depth 8), scratch
stays 0, occupancy stays 1. Shrink the static array to make room and it instead
promotes to **scratch** (80 B/lane at depth 4, 144 at depth 8) *and* lets
occupancy rise to 2 blocks/CU, which would have silently destroyed the premise of
this figure. exp_04 saw the scratch half of this in the megakernel (+68 B/lane at
`kPushBatch=4`) and the LDS half at `kPushBatch=2` (LDS 163,632, breaking its
parity gate); the ubench reproduces both halves in a clean setting and shows the
choice is a function of the LDS budget, not of `Regions`. The surface gives the
caller no way to ask for either. **Proposed:** an overload taking the regions as
a fully-unrolled parameter pack or a `std::array` by value, so the pointers stay
in registers; and a doc-comment line saying the by-reference arrays are promoted
and the target depends on remaining LDS. The static array here is deliberately
held at 96 KiB at every depth: LDS differs across depths, but *occupancy* — the
only thing the CTA axis is allowed to mean — does not.

## 8. Pre-lease validation actually performed (all CPU, `cpu_gate.log`)

`bash tools/e22_gate.sh` on the node runs, and all of it is green:

1. **build** — 3 s, 0 errors, 0 warnings of consequence.
2. **resource remark gate** — the table in §7; `CPU_GATE: PASS`.
3. **ISA census gate** — `ISA_GATE: PASS`; confirms the CDNA4 bf16 MFMA is
   really emitted, the push really lowers to 16-byte stores, scratch is empty,
   and both `s_memrealtime` stamps survived.
4. **grid census** — 244 / 9 / 303 points for plan / quick / full, **zero
   duplicate keys** in any tier (a duplicate key would silently make the resume
   logic skip a real point).
5. **summarizer self-test** — `selftest_summarize.py` runs the adjudicator over
   two synthetic worlds and asserts every H1–H4 verdict flips, that the
   falsifier fires only in the bandwidth-limited world, and that the knee,
   compute-slowdown, protocol-ratio and overlap-flag derivations all populate.
   9/9 assertions pass, so the hypotheses are falsifiable *by this code*, not
   just in principle.
6. **end-to-end dry run** — `--dry-run` emits one synthetic record per point
   through the **real `emit()` path** (records tagged `status: "dry_run"`, which
   makes them impossible to confuse with data), then: 244/244 lines parse as
   JSON, `summarize.py` builds `saturation.json` with 24 knee series and the
   hypothesis block, and **re-running adds 0 points**, which is the resume
   logic proven rather than asserted.
7. `bash -n` clean on `run_saturation.sh`, `build_in.sh`, `tools/e22_gate.sh`.

No GPU was touched: `--dry-run` and `--list` both return before any HIP call,
and nothing in the gate path launches a kernel.

The counts above are the rev-8 grid, kept as the historical record. Rev 10 (the
revision that produced the data) adds the shape-16 MFMA series — the compute-only
denominator for the `reserved_only` arm — and drops the two overlay points where
C equals the grid, which leaves no CTA for the compute role: plan **250**, full
**309**, still zero duplicate keys. The rev-9 attempt at those C == grid points
is what surfaced the bug: `tile_rate()` was asked for a zero-CTA probe launch and
the sweep aborted with `invalid configuration argument` after 40 points. It is now
refused at grid-build time and guarded loudly inside `tile_rate()`.

## 9. What runs when the lease arrives, and how long it takes

One command, on the node, after `bash build.sh`:

```bash
setsid timeout 3600 bash run_saturation.sh --tier plan 2>&1 \
  | tee ~/e22/out/sweep.log
```

It refuses to start if `rocm-smi --showpids` shows another job (`--force`
overrides), wraps the binary in `setsid` + `timeout`, re-invokes it until the
grid is empty, and regenerates `saturation.json` after every attempt. Smoke first:

```bash
bash run_saturation.sh --tier quick        # 9 points, exercises all 3 modes,
                                           # both fanouts, all 3 arms, protocol
```

**GPU wall-clock estimate.** Point counts are exact (`--list`): **quick 9, plan
250, full 309** (E22_SRC_REV 10; C == grid overlay points are not
measurable and are dropped by uild_grid). Per point: 2 warmup + 5 timed launches at a ~25–40 ms target,
so ~0.2–0.3 s of kernel time, plus a ~10 µs stamp read-back per rotation. Add
one-time setup (≈45 GiB allocated + filled across 8 devices, `--tier plan`) and
13 verify passes.

| tier | points | kernel time | + setup/verify | **total** |
|---|---:|---:|---:|---:|
| quick | 9 | ~3 s | ~70 s | **~1.5 min** |
| plan | 250 | ~70 s | ~80 s | **~3 min** (measured: 90 s wall, one attempt) |
| full | 309 | ~95 s | ~85 s | **~3.5 min** |

That is the same order as one 5-rotation MoE campaign (~3.5 min), so no reduced
grid is needed; the sweep is not the expensive part of the night. The estimate's
weak point is the low-CTA xGMI points: if a single pushing CTA turns out slower
than ~1.3 GB/s, C=1 stretches past 35 ms per launch and the sweep grows by a few
minutes — bounded, and `--slice`/`--max-seconds` keep every attempt inside its
timeout either way.

## 10. Verdict — MEASURED

**Provenance.** `2026-08-12T10:45:11Z`, `gbt350-odcdh2-c05-1`, 8× MI350X
(gfx950), exclusive lease, `rocm-smi --showpids` clean apart from `gpuagent`.
`E22_SRC_REV 10`, `src_sha256=e3af5f1b…`, `bin_sha256=21fa0039…`, tier `plan`,
**250/250 points**, 5 rotations + 2 warmup each, one attempt, `rc=0`. Artifacts:
`saturation.json` (schema `exp22-saturation-1`), `saturation_plan.jsonl` (raw,
one record per point), `audit.txt` (per-point roofline audit), `run_plan.log`,
`plan_wrap.log`.

**Data hygiene, checked before anything was interpreted.** 0/250 points above
device peak. 0/250 with `rel_iqr_pct` above 1% (median 0.05%). All 75
concurrent/`reserved_only` pairs `cmp_tiles_matched=true`.

### 10.1 The knee of each curve (the figure's deliverable)

Isolated, payload-only (`protocol=0`), per-CTA work. "Rising at C=64" is the
honest saturation test — `last/prev` is the ratio of the C=64 point to the C=32
point, and where it is far above 1 the series has **no knee inside the swept
range** and the 90%-of-plateau number is an artifact of the plateau definition.

| series | knee C | plateau | % of nominal peak | last/prev | reading |
|---|---:|---:|---:|---:|---|
| xgmi single mlp4 | **8** | 56.9 GB/s | 74.1% of 76.8 | 1.000 | saturated |
| xgmi single mlp1 | **16** | 55.5 | 72.2% | 0.993 | saturated |
| xgmi single mlp8 | **16** | 55.7 | 72.6% | 1.022 | saturated |
| xgmi rr7 mlp4 | **32** | 355.0 | 66.0% of 537.6 | 1.071 | saturated |
| xgmi rr7 mlp1 | (64) | — | 61.2% at C=64 | **1.870** | no knee in range |
| xgmi rr7 mlp8 | (64) | — | 61.3% at C=64 | **1.726** | no knee in range |
| hbm | **128** | 4,151.9 GB/s | 51.9% of 8,000 | 1.102 | saturated |
| mfma shape 4 | none | 119.9 TFLOPS | 5.2% of 2,300 | 1.140 | linear to 256 |
| mfma shape 16 | none | 352.0 | 15.3% | 1.140 | linear to 256 |
| mfma shape 256 | none | 910.2 | 39.6% | 1.129 | linear to 256 |

The single-link curve, which is the one the megakernel's push path lives on:
C1=13.4, C2=23.9, C4=42.6, **C8=55.2**, C16=57.3, C32=56.5, C64=56.5 GB/s
(mlp4). It is linear to 4 CTAs, at 96% of its ceiling by 8, and going from 8 to
64 pushers — 8× the CTAs — buys **+2.4%** and then nothing.

MFMA has no knee at all: `TFLOPS = k·CTAs` with R² ≥ 0.99989 at all three
intensities and per-CTA efficiency 0.987–0.998 from 32 to 256 CTAs. This is the
one-block-per-CU prediction confirmed directly — there is no oversubscription
regime, so there is no issue-slot contention for a dedicated pool to relieve.

### 10.2 H1–H4 as `summarize.py` computed them (not re-derived by hand)

| hypothesis | verdict | numbers |
|---|---|---|
| H1_mlp4 | **REFUTED_NEVER_REACHED**, `falsifier_triggered=true` | best 57.30 GB/s = **74.6%**; gate was 75% |
| H1_mlp8 | **REFUTED_NEVER_REACHED**, `falsifier_triggered=true` | best 56.33 = 73.3% |
| H2_mlp4 | **PLATEAU_BELOW_75PCT** | knee 32 CTAs; best 367.2 = 68.3% of 537.6 |
| H2_mlp1 / H2_mlp8 | PLATEAU_BELOW_75PCT | 61.2% / 61.3%; still rising at C=64 |
| H3 | **REFUTED_PAYLOAD_RIDES_FREE** | payload-only concurrent/isolated median **0.9951** (n=42); protocol/payload median 0.878 |
| H4 shape 4 / 16 / 256 | **SUPPORTED** ×3 | R² = 0.99997 / 0.99997 / 0.99989 |

### 10.3 THE FALSIFIER FIRED — reported plainly

The pre-registered falsifier is armed on H1 and **it is set in the artifact**:
`H1_mlp4.falsifier_triggered = true`, `H1_mlp8.falsifier_triggered = true`.
The pre-registered text is not satisfiable by this hardware: the gate asks for
75% of the 76.8 GB/s nominal link = **57.6 GB/s**, and the measured achievable
ceiling of the single-link push is **56.9–57.3 GB/s (74.1–74.6%)** at *every*
CTA count from 8 to 64. No CTA count could have passed it. That is a threshold
I calibrated against a spec number instead of a measured ceiling, and the fired
falsifier stands in the record as written.

The falsifier's substantive claim is separable and is **answered by the same
data**: it says that if the pusher needs ≥32 CTAs to saturate, the pool is
bandwidth-limited rather than protocol-limited and a tiny topology-sized pool
is the wrong story. Measured, the curve is at 96% of ceiling by **C=8** and
flat from 8 to 64. So the mechanism claim survives while the numeric gate as
written fails. Both facts belong in the paper; the honest form is "achievable
single-link egress is 74% of nominal, and it is reached at 8 CTAs", not a
restated 75% threshold.

### 10.4 The concurrent arms really were concurrent

Two independent checks, both in the artifact:

1. `overlap_pct_median = 100.0` at **all 75** concurrent points (min = max =
 100.0, 0 below 90). This is 100% *by construction* — `cmp_tiles` is fitted so
 the compute role outlasts the resource role — so on its own it only proves the
 resource span is contained in the compute span, not that time was shared.
2. The independent check: `wall < res_span + cmp_span` at **75/75** points, with
 `(res_span + cmp_span) / wall` = **1.62 min, 1.65 median, 1.82 max**. A
 sequential execution cannot produce a sum of role spans 1.6–1.8× the wall
 clock. Concurrency is proven per point, not assumed.

### 10.5 Isolated vs concurrent, and achieved fraction of peak

Payload-only pairs (n=47): **median concurrent/isolated = 0.9951**. Carrying
the payload alongside a full compute pool costs the payload half a percent.
Extremes: 0.8158 (`xgmi c4 mlp8 single`, in the pre-knee rising region where
the curve is also non-monotone) and 1.0344 (`xgmi c4 mlp4 single`). With the
protocol compiled in, the same ratio falls to a median of 0.878, and the g=16
probe costs up to 42% (`rr7 c16`: 193.3 → 113.1 GB/s). **exp_20's
"interference is protocol, not payload" reproduces as a curve.**

Achieved fraction of device peak, best point per mode: single link **74.6%** of
76.8 GB/s, aggregate egress **68.3%** of 537.6 GB/s, HBM **54.4%** of 8,000
GB/s (4,353.7 at C=256), MFMA **39.6%** of 2,300 dense bf16 TFLOPS (910.2 at
shape 256). The MFMA fraction is an LDS-fed occupancy-1 loop, not a peak-seeking
GEMM; it is the shape the megakernel's compute phases run, which is the point.

### 10.6 Q2 — does dedicating CTAs to communication ever win? No, with one honest caveat

The `reserved_only` arm holds C CTAs idle while the same compute pool runs, so
it measures the reservation's cost with no payload moving. Decomposition, both
terms in the same launch family:

- **Reservation cost** — what dedicating C CTAs costs before a byte moves. By
 H4's measured linearity (R² ≥ 0.99989) this is exactly C/grid: 3.1% at C=8,
 **6.25% at C=16**, 25% at C=64.
- **Interference cost** — what letting those same CTAs carry payload costs the
 co-resident compute pool, against the C-matched idle twin (`cmp_tiles`
 identical, verified): **median 0.073%** over 75 pairs, and ≤0.23% at every
 `rr7` point at every C — the fanout the real combine uses.

So at C=16 the choice is 6.25% of the machine for an idle pool versus 0.07% for
a co-resident one. `reserved_only` delivers 0 GB/s by construction and never
wins on throughput; on cost it loses by ~90×.

**The honest caveat, and it is a real one.** In 13 of 75 pairs the interference
term (7.4–14.6%) exceeds the reservation term. Every one is `xgmi` **single**
fanout at C ∈ {4, 8, 16} — the saturated-single-link corner. The mechanism is
not fabric contention. Per-CTA compute rate in this dataset is **bimodal**:
1.607 TF/CTA (n=43) or 1.379 TF/CTA (n=107), a **14.2%** step, and *every*
nonzero interference number equals that step. In all 13 pairs the idle twin sits
on the fast side and the traffic-live arm on the slow side; in the 62 pairs
where both arms sit on the same side, the cost is −0.20% to +10.10% with median
0.073%. The reading: remote-write traffic can flip a cache-resident compute
phase into a memory-bound one, and that transition — not the fabric — is the
14%. It only appears where the target link is already at its ceiling, which is
the paper's throttle argument restated: past the knee, extra pushers stop
buying bandwidth and start costing compute.

One same-regime exception is a genuine shared-resource cost and should be quoted
as such: `hbm c128` (half the grid streaming at 3.9 TB/s) costs the compute pool
**10.1%**. HBM is the one resource in this ubench where a large payload pool
measurably taxes compute.

**Known confound, named for the follow-up.** `cmp_tiles` is fitted per point so
the compute role outlasts the resource role, so the compute footprint grows with
C and with fanout. Each pair is internally exact, but the interference curve is
**not comparable across C**. The fix is a fixed-footprint rerun (constant
`cmp_tiles`, compute role sized by iteration count instead of the resource span),
which would settle eviction-vs-fabric outright. Until then, quote the per-pair
numbers and the bimodal step, not the shape of the curve in C.

### 10.7 What this figure supports for the paper

1. Communication resources saturate at a **tiny** CTA count: 8 for one link,
 32 for the whole 7-link fabric, versus 256 CTAs of compute capacity. A
 topology-sized pool is sufficient; a large one is waste.
2. Compute has **no knee** — linear to 256 CTAs, R² ≥ 0.9999. One block per CU
 means there is no issue-slot contention for a dedicated communication pool to
 relieve, which is the AMD-specific premise stated at the top of the plan.
3. A compute CTA can carry payload for **~0.07%** (≤0.23% on the real fanout),
 while reserving the same CTAs costs C/grid outright. That is the quantitative
 basis for "scheduling, not CTA dedication".
4. The cost that does exist is **protocol** (up to 42% at g=16), not payload —
 exp_20 reproduced as a curve.
5. Two named regimes where a payload pool does tax compute: HBM at half the grid
 (10.1%), and pushing past a saturated link's knee (14.2% cache-regime flip).

### 10.8 Row for `aug11/PLOTS.md`

Not written from here (this agent owns only `exp_22_fig7_saturation/**`); the
paste-ready row is in `plots_row.md`.

| figure | data file | generating experiment | status |
|---|---|---|---|
| Fig 2 (NanoFlow-Fig-7 analog) | `exp_22_fig7_saturation/saturation.json` | exp_22, `E22_SRC_REV 10`, 250 pts | **DONE** |

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
244, full 303**. Per point: 2 warmup + 5 timed launches at a ~25–40 ms target,
so ~0.2–0.3 s of kernel time, plus a ~10 µs stamp read-back per rotation. Add
one-time setup (≈45 GiB allocated + filled across 8 devices, `--tier plan`) and
13 verify passes.

| tier | points | kernel time | + setup/verify | **total** |
|---|---:|---:|---:|---:|
| quick | 9 | ~3 s | ~70 s | **~1.5 min** |
| plan | 244 | ~70 s | ~80 s | **~3 min** |
| full | 303 | ~95 s | ~85 s | **~3.5 min** |

That is the same order as one 5-rotation MoE campaign (~3.5 min), so no reduced
grid is needed; the sweep is not the expensive part of the night. The estimate's
weak point is the low-CTA xGMI points: if a single pushing CTA turns out slower
than ~1.3 GB/s, C=1 stretches past 35 ms per launch and the sweep grows by a few
minutes — bounded, and `--slice`/`--max-seconds` keep every attempt inside its
timeout either way.

## 10. Verdict

Deferred. Nothing is measured yet. When the data lands, this section states the
knee of each curve, the concurrent/isolated ratio at the knee, the four H
verdicts as computed by `summarize.py` (not re-derived by hand), whether the
falsifier fired, and the row for `aug11/PLOTS.md`.

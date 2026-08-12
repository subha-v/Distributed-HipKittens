# exp_21 — resource-saturation microbenchmark (paper Fig 2 / Q4(a)), MI300X

NanoFlow-Fig-7 analog for the GEMM-RS operator on **gfx942**, 8× MI300X
(`banff-sc-cs47-05.dh170.dcgpu`, SPX, 304 CU/GPU, container `dhk-gemmrs`).
It is the second operator/architecture instance of the sibling MoE agent's
`exp_22_fig7_saturation` on gfx950; the transferable methodology in
`aug11/FIGURE_SPECS.md` §6 is mirrored verbatim and every hardware ceiling is
re-measured on this node (§5).

Status of this document: written **before any GPU run** of this experiment.
The predictions in §6 are pre-registered.

---

## 1. What the figure plots

Three panels, x = CTA count `C` in every panel, y = per-resource throughput,
for the three resource classes this kernel actually uses, plus a **concurrent
overlay** (each curve measured again while this kernel's own GEMM mainloop runs
on the remaining `304 − C` CTAs). Ceilings are dashed horizontals taken from
this node's own reading (§5). Knee annotation per series = **smallest C
reaching 90% of that series' own plateau** (the sibling's definition, kept).

| panel | mode | body | work unit | metric |
|---|---|---|---|---|
| a | MFMA | this kernel's double-buffered mainloop (`load_issue`/`acquire_frags`/`mma_ABt`/`acc_anchor`/`load_commit`, KH=2 half-split, `s_setprio` window) at BM=BN=256, BK=32 | one 256×256×K tile | `2·M·N·K·tiles / t` TFLOPS |
| b | HBM | the reducer body at REDV=1: `m3::pull_sum_bf16_strip_mlp8` — 8 bf16 source slots read as 16 B packets, source-ascending FP32 accumulate, one RNE pack, 1 packet written | one 256-row × 256-col strip | `(8+1)·bytes·strips / t` GB/s |
| c | xGMI | the **real** 16 B peer-packet emit: `emit_band_packets`' grid-strided packet loop calling `hk_gemm_rs::store_peer_packet16` (→ `kittens::distributed::store_peer_packets(...,16,0,1)`) into a destination obtained from `kittens::distributed::translate_peer<8>` | one 32-row × 256-col emit window (16 KB, 1024 packets) | `bytes_pushed / t` GB/s |

The three bodies are **copied from the production kernel**, not idealised: mode
a is the `gemm_rs_mi300x.cpp` k-loop verbatim (including the volatile-asm
fragment anchors and the accumulator anchor that pin the schedule), mode b
calls the production `pull_sum_bf16_strip_mlp8` unmodified from the shipped
adapter header, and mode c reproduces `emit_band_packets`' packet indexing,
its LDS-staged source and its `dst_base + r·dst_row_stride + c` destination
arithmetic with `dst_row_stride = N`, i.e. the real 16 KB-apart row scatter.

## 2. Axes

| axis | points | note |
|---|---|---|
| mode | a, b, c | |
| CTAs (mode a) | 32, 64, 96, 128, 160, 192, 224, 256, 288, **304** | 304 = SPX CU count |
| CTAs (mode b) | 8, 16, 32, 64, 128, 256, **304** | |
| CTAs (mode c) | 1, 2, 4, 8, 16, 32, 64 | |
| **in-flight remote-op bound** `d` (mode c) | 1, 4, 8, **0 = unbounded** | number of 16 B peer stores a thread issues before `s_waitcnt vmcnt(0)`; 0 = no wait until the end of the work list, which is the production shape |
| fanout (mode c) | `single` = all packets to `(me+1)%8`, prices ONE link; `rr7` = window *i* to peer `(me+1+i%7)%8`, prices aggregate egress | every rank pushes concurrently in both cases, so `single` is a ring and each used link carries exactly one source |
| overlay | `isolated`, `concurrent`, `reserve_control` | |
| `--protocol` | off, on (mode c) | on = per-release-group `producer_drain_release<system>()` (the `s_waitcnt vmcnt(0)` + barrier + release fence that lowers to `buffer_wbl2 sc0 sc1`) + one system-scope probe RMW + one relaxed system-scope publication store by tid 0, at `RELEASE_GROUP = 4` (the shipped constant) |

Point count: 10 (a) + 21 (b) + 119 (c) = **150 points**, 5 rotations each.

## 3. Controls (all mirrored from FIGURE_SPECS.md §6)

- **C-matched reserve-only control.** The concurrent point is
  *C-with-traffic* vs *C-without-traffic*, **never** vs the full 304-CTA grid.
  In the control the resource role idle-spins (one polling thread per CTA,
  `s_sleep` backoff) for the same wall duration the live arm's resource role
  took, so the same C CUs are reserved and the only difference is the traffic.
  Reported per concurrent point: the resource throughput **and**
  `gemm_slowdown = control_TFLOPS / live_TFLOPS` measured on the same
  `304 − C` CTAs.
- **The isolated arm has no protocol at all**: no probe atomics, no flags, no
  fences, no release. It prices payload transport only. `--protocol` re-adds
  the production release + probe RMW + publication inside the same ubench, so
  protocol cost is separable from payload cost without leaving this binary.
- **Duration-matched concurrency.** The concurrent GEMM is a *filler*: it
  re-runs its tile list until the resource role signals completion (checked
  once per tile, before the tile starts, through a CTA-uniform LDS flag), and
  reports the tiles it actually finished. Without this, a fixed GEMM work list
  would finish early and the "concurrent" resource number would mostly be an
  isolated number.
- **Checksum read-back once per mode-c config.** A silently failing store must
  not fake bandwidth. Implemented **stronger** than the sibling's xor-fold: the
  staged payload is a pure function `sat_pattern(r, c)`, the destination region
  is pre-poisoned with a value the pattern never produces, and a
  destination-side verify kernel walks exactly the written extent, counts
  bitwise mismatches, and returns a u64 xor-fold fingerprint. `checksum_ok` =
  (mismatches == 0). The fold alone is reported as `checksum_fold` but is not
  the gate, because xoring k identical windows degenerates (k even → 0). This
  costs **zero** instrumentation in the timed path.
- **Fixed global work list per mode, grid-strided by dense role id**, so total
  work is identical at every CTA count for modes a and b and time differences
  are pure throughput. (Mode c's rr7 arm splits the same list across 7 peers.)
- **Per-role `s_memrealtime` start/end** by tid 0 into `u64[304][2]`. In the
  concurrent arm the two roles finish at different times and wall time
  attributes nothing, so every throughput in this experiment is computed from
  the *role's own* span (max end − min start over that role's CTAs). Host wall
  time is recorded alongside as a cross-check only.
- **The `s_memrealtime` tick rate is measured, not assumed.** The sibling's
  "100 MHz / 10 ns ticks" is a gfx950 statement (FIGURE_SPECS §5.7). A
  two-stage calibration kernel spins on `s_memrealtime` for a host-timed
  ~200 ms and the driver divides; the median of 5 reps lands in
  `saturation.json` as `tick_rate_hz` with its spread.
- **5 rotations per point, median reported, all samples recorded.** Points
  sized for 10–50 ms of kernel time at the middle of each mode's CTA range
  (see §4 for where a fixed work list forces this outside the band).
- **Clocks pinned** (`tools/set_clocks.sh pin 1900`, this node's standing
  rule) **and** `rocm-smi --showclocks` recorded before and after every point
  into the JSON. The GEMM-RS charter overrides the sibling's no-pinning choice;
  the record is kept either way.
- Duration-based warmup (≥400 ms of launches) before every point's rotations.
  Idle sclk here is ~125–132 MHz against ~1900 pinned; a fixed-iteration warmup
  already produced a 60%-wrong number once on this node.

**This is a diagnostic, not a candidate kernel.** It is a standalone module
that shares no object with the production binding, changes no production file,
and can never enter a graded build. There is therefore **no M3/M4/M5 gate
ladder** here: the correctness gate for this experiment is the destination-side
checksum plus the ISA/resource verification of the ubench's own kernels. No
number from this experiment is ever reported as kernel performance.

## 4. Sizing, and where a fixed work list breaks the 10–50 ms band

A work list fixed across the CTA sweep means time scales roughly as `1/C`, so
one band cannot hold at both ends of a 38× CTA range. Sizing targets the
**small-C** end at ≲150 ms and accepts the large-C end near or slightly below
10 ms; this is the same trade the sibling made and it is stated rather than
hidden. `rounds = ceil(work / C)` is recorded per point so the tail imbalance
of a fixed list is visible.

| mode | work list | bytes / FLOPs | target |
|---|---|---|---|
| a | 12,160 tiles (= 304 × 40) of 256×256×3712 | 5.92 PFLOP total | ~12 ms at C=304, ~115 ms at C=32 |
| b | 50,688 strips of 256×256 | 59.8 GB touched (9 × 1.18 MB per strip) | ~15 ms at C=304, ~150 ms at C=8 |
| c | 32,768 windows of 16 KB | 512 MB pushed per rank per launch | ~10 ms at C=64, ~85 ms at C=1 (longer at d=1) |

Mode a's operands are deliberately cache-resident (A and B are 1024×3712 each,
16 distinct tiles revisited): panel a must price MFMA issue, not HBM.

Mode b's working set is deliberately **larger than this node's 256 MB AMD
Infinity Cache**: 8 source slots × 4096 rows × 8192 cols × 2 B = 512 MB read
plus 64 MB written. A working set inside the MALL would have made panel b an
LLC curve. This is a gfx942-specific sizing constraint with no sibling analog.

## 5. Ceilings — measured or read off THIS node, never quoted

The sibling's dashed horizontals (76.8 GB/s per link, 537.6 GB/s aggregate,
8 TB/s HBM3E, 148 GB/s service rate) are gfx950 facts and **must not appear in
any artifact of this experiment** (FIGURE_SPECS §5.1). `ceilings.sh` collects,
each with a `source` string recorded next to it in the JSON:

- xGMI per-link and aggregate: `rocm-smi --shownodesbw`, `--showtopo`,
  `--showtopoweight`, `rocm-smi --showbw`, and `amd-smi static`/`amd-smi
  metric` if present on this node.
- HBM peak: `hipDeviceProp_t` (`memoryClockRate`, `memoryBusWidth`) and
  `rocm-smi --showmclk`.
- bf16 MFMA peak: derived from **this node's** pinned sclk × 304 CU × the
  gfx942 per-CU bf16 MFMA rate implied by `v_mfma_f32_16x16x16_bf16`
  (1024 FLOP/cycle/CU), with the arithmetic written out in `result.md`.

If the node refuses to report a per-link xGMI figure, `result.md` says so and
falls back to the **measured plateau as the empirical ceiling**. An honest
empirical ceiling beats a borrowed spec number.

## 6. Pre-registered predictions (gfx942, committed before measuring)

- **H1** The 16 B peer-packet emit saturates a single link by **8–16 CTAs** at
  in-flight bound `d ≥ 4`.
- **H2** Aggregate (`rr7`) egress saturates by **16–32 CTAs**.
- **H3** The concurrent curve sits **measurably below** isolated at equal C —
  that gap is the interference term — and the depression grows with the
  `--protocol` knobs (release + probe RMW + publication) rather than with
  payload bytes. This is the GEMM-RS instance of the paper's
  protocol-not-payload claim.
- **H4** MFMA TFLOPS scales ~linearly to 304 CTAs. The claim is that no
  oversubscription regime exists, and it is **verified from the resource
  tuple**, not asserted: mode a's LDS footprint is
  `2·(BM+BN)·BK·2 = 65,536 B`, the entire gfx942 per-workgroup LDS budget, so
  occupancy is exactly 1 CTA/CU by LDS alone regardless of registers. The M2
  report for this ubench is the evidence.
- **H5** (mode c, ours) the in-flight bound matters: `d = 1` costs a large,
  monotone fraction of the `d = 0` plateau, and the `d = 4 → 8 → 0` gap is
  small — i.e. the bound is cheap to respect. This is the ubench-scale version
  of the paper's "in-flight remote-op bound" axis.
- **Falsifier (stated in advance).** If the isolated single-link curve needs
  **≥ 32 CTAs** to reach ~75% of its own plateau at any depth, the
  "tiny topology-sized pool" story is wrong for our emit shape, the emit is
  bandwidth-limited rather than protocol-limited, and the in-flight-bound
  priority inverts. That result gets reported as such, not explained away.

## 7. Deliverables

- `saturation.json` — schema in `result.md`; one record per point with mode,
  ctas, depth, fanout, overlay, protocol, metric, median value, all samples,
  per-rank values, concurrent GEMM TFLOPS, gemm_slowdown, checksum_ok,
  clocks_before/after, rounds, and the run's `node` / `ceilings` /
  `tick_rate_hz` headers.
- `result.md` — verdict per hypothesis, the knee table, the
  concurrent/isolated ratio at each knee, the ceiling derivations with
  sources, and every deviation from the sibling spec.
- `isa_report.txt` — per-kernel resource tuple and the store-width / MFMA
  opcode evidence.

## 8. Deviations from the sibling spec (all deliberate, all recorded)

1. CTA grids are 304-wide, not 256-wide (SPX CU count).
2. Work units are GEMM-RS's own: a 256×256 GEMM tile, a 256×256 REDV=1 reduce
   strip, and a 16 KB emit window of 16 B packets — not 14,336 B MoE rows.
3. `d` is the count of in-flight 16 B peer stores per thread, not an MoE "MLP
   depth", and `d = 0` (unbounded, the production shape) is added.
4. Clocks are pinned at 1900 MHz (charter rule) as well as recorded.
5. The checksum is a destination-side bitwise verify plus a fold fingerprint,
   which strictly dominates an xor-fold and costs nothing in the timed path.
6. Mode b's working set is forced above 256 MB by this node's Infinity Cache.
7. The concurrent GEMM is a duration-matched filler with a completed-tile
   counter; a fixed GEMM list would have finished early and corrupted the
   overlay. The reserve control is duration-matched to the live arm.
8. The mode-c destination is allocated **fine-grained (uncached)** by default.
   Rationale, and why this is not optional on gfx942: with a coarse-grained
   peer allocation and no release, 16 B peer stores can be absorbed by the
   issuing XCD's L2 — that is precisely why the production protocol needs
   `buffer_wbl2 sc0 sc1` — and the no-protocol isolated arm would then report
   L2 bandwidth as fabric bandwidth. Fine-grained makes the isolated arm a true
   wire measurement. `--payload-coarse --protocol 1` reproduces the production
   configuration and is run as a cross-check; `payload_granularity` is recorded
   on every point.

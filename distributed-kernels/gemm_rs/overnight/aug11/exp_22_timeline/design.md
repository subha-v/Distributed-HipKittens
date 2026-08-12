# exp_22 design — phase-event ring, traffic model, and the patch plan

Everything below is gfx942 / MI300X. Where the sibling MI350X spec supplies a
number, it is treated as a hypothesis and re-derived (`../FIGURE_SPECS.md` §5).

---

## 1. The phase-ID enum

Fixed and ordered by lifecycle. Write order within a CTA is monotone in
timestamp but **not** monotone in ID — a producer cycles 1→9 once per tile
group. **Absence of an ID means that CTA did not run that phase**: producers
and reducers run disjoint subsets, and on shape 5 the grid is 272 producers
(pid < `num_gemm_ctas`) and 32 reducers (pid ≥ `num_gemm_ctas`).

| id | name | role | site |
|---:|---|---|---|
| 0 | `CTA_BEG` | both | kernel entry, before the epoch RMW |
| 1 | `MAINLOOP_BEG` | producer | before `zero(C_accum)` |
| 2 | `MAINLOOP_END` | producer | after the k-loop closes |
| 3 | `CREDIT_WAIT_BEG` | producer | before the per-band reuse-credit wait |
| 4 | `CREDIT_WAIT_END` | producer | after the wait's convergent error check |
| 5 | `EMIT_BEG` | producer | before the band loop |
| 6 | `EMIT_END` | producer | after the band loop |
| 7 | `RELEASE_BEG` | producer | before `release_payload_system()` |
| 8 | `RELEASE_END` | producer | after `release_payload_system()` |
| 9 | `PUBLISH_END` | producer | after the leader's ready-flag publish loop |
| 10 | `READY_WAIT_BEG` | reducer | before the 8-source ready wait |
| 11 | `READY_WAIT_END` | reducer | after the wait's convergent error check |
| 12 | `REDUCE_BEG` | reducer | after `acquire_payload_system()` |
| 13 | `REDUCE_END` | reducer | after `pull_sum_bf16_strip_mlp8` |
| 14 | `CREDIT_PUB_END` | reducer | after the leader's retirement-credit loop |
| 15 | `CTA_END` | both | before the role's `return` / at kernel exit |

**Relation to the charter's nine-tuple.** The charter names {mainloop
start/end, emit start/end, release, wait start/end, reduce start/end}. Those
are ids 1, 2, 5, 6, 7, 10, 11, 12, 13. Six ids are additions, each of which
buys something the nine-tuple cannot express:

- `CTA_BEG` / `CTA_END` (0, 15) give the per-CTA span. They are what make the
  in-situ tick calibration possible (§5) and what let the reader distinguish
  "this CTA has no `MAINLOOP_BEG` because it is a reducer" from "…because it
  aborted on an error bit".
- `CREDIT_WAIT_BEG` / `CREDIT_WAIT_END` (3, 4) separate *waiting to emit* from
  *emitting*. Without them the emit interval swallows the producer's
  reuse-credit stall and the xGMI strip is deflated by exactly that stall —
  the figure would understate our own egress rate. exp_20 ranks `sync` as the
  second-largest non-GEMM pool on shape 6 (168.7 µs), so this is not a
  hypothetical.
- `RELEASE_END` (8) makes the release a measurable interval rather than an
  instant. `release_payload_system()` is a `buffer_wbl2 sc0 sc1` plus a
  `vmcnt(0)` drain and exp_20 charges it 65.4 µs on shape 5.
- `PUBLISH_END` (9) closes the group so the next group's `MAINLOOP_BEG` is not
  charged with the publish loop.

## 2. The ring

| field | value |
|---|---|
| shape | `u64[304][64]` = **155,648 B** per rank, one plain device allocation |
| indexing | `ring[cta * 64 + slot]`, `cta = blockIdx.x` |
| entry | `(phase_id << 56) \| (s_memrealtime() & 0x00FFFFFFFFFFFFFF)` |
| slot 63 | **drop counter**, incremented instead of wrapping |
| writer | `threadIdx.x == 0` only; plain relaxed agent-scope store; one store per boundary crossed |
| reader | host, after the launch completes — no in-flight reads |
| init | host zero-fill (`dhk_rt.fill_bytes`) once per capture, before the traced launch |
| flag | `HK_GEMM_RS_MI300X_TRACE`, **default 0** |

**304, not 256.** MI300X in SPX is 304 CU and the grid is exactly
`m3::CU_COUNT = 304` CTAs of 512 threads. The sibling's `u64[256][24]` is a
gfx950 fact (`../FIGURE_SPECS.md` §5 item 2).

**Depth 64, justified from the worst case rather than inherited.** Events per
CTA per epoch, computed from the resolved plan of each graded row
(`gemm_rs_mi300x_host_abi.hpp:109-115`, `eb = gcd(BM, M/8)`,
`bands = BM/eb`, `tiles_per_cta = ceil(tiles / NG)`,
`rgroup = tiles_per_cta >= 4 ? 4 : 1`):

| shape | row | BM/BN/BK | NR | tiles | tiles/CTA | rgroup | producer events | red. tiles/CTA | reducer events |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64×7168×18432 | 1 | 32/64/128 | 56 | 224 | 1 | 1 | 11 | 2 | 12 |
| 512×4096×12288 | 2 | 64/128/64 | 32 | 256 | 1 | 1 | 11 | 1 | 7 |
| 2048×2880×2880 | 3 | 128/192/32 | 32 | 240 | 1 | 1 | 11 | 1 | 7 |
| 4096×4096×4096 | 4 | 256/256/32 | 32 | 256 | 1 | 1 | 11 | 1 | 7 |
| **8192×4096×14336** | **5** | **256/256/32** | **32** | **512** | **2** | **1** | **20** | **2** | **12** |
| 8192×8192×29568 | 6 | 256/256/32 | 48 | 1024 | 4 | 4 | **29** | 3 | 17 |

Producer events = `1 + groups × (6 × tiles_in_group + 3) + 1`; reducer events
= `1 + 5 × red_tiles_per_cta + 1`. The maximum over the six graded rows is
**29** (shape 6). Depth 64 gives 63 usable slots — **2.17× the worst graded
case** — at 152 KB of device memory, which is noise against a heap already
sized in hundreds of MB. Depth is a compile-time constant so the slot address
is one shift, not a multiply by a runtime value.

The generic config row (`32/64/64`, up to 118 tiles per CTA) overflows depth
64 by construction. That is intentional and safe: it is never used by a graded
shape, and if it is ever traced the drop counter in slot 63 reports exactly how
many events were lost. **Overflow drops and counts; it never wraps**, because a
wrapped ring silently produces a plausible-looking but wrong timeline.

**Writer cost.** ≤ 29 scalar `s_memrealtime` reads and ≤ 29 `global_store_dwordx2`
per CTA per epoch, by one thread, at points that are already CTA-convergent
(between `__syncthreads()` boundaries or in straight-line code). Against a
642 µs epoch on shape 5 this is far below the 2%-delta re-run threshold, and
the ON-vs-OFF wall delta is measured and reported rather than assumed.

**No new fences, no new atomics.** The stamp is a plain store. The drop counter
is a non-atomic read-modify-write, which is correct because exactly one thread
in the process writes that word.

## 3. Alternatives rejected

1. **`rocprofv3` PC-sampling or counter timelines as the source for arm (b).**
   Rejected for the same reason the sibling rejected it: the megakernel is
   *one dispatch*, so a dispatch-granular profiler attributes nothing to a
   phase inside it. Counters remain a one-time cross-check on an aggregate
   (§6), never the timeline source.
2. **LDS-staged event buffering** (batch events in LDS, flush once at exit).
   Impossible here: rows 4/5/6 allocate `2*(BM+BN)*BK*2 = 65,536 B`, the entire
   gfx942 per-workgroup LDS budget, and the emit staging buffer already aliases
   the dead A/B double buffers. There is no LDS to spare on exactly the shapes
   the figure is about.
3. **A monotone per-CTA event counter in a register** is the primary design, but
   it costs one live register across the whole kernel, and rows 4/5/6 sit at
   246-248 of 256 VGPRs. If the flag-ON build spills, the fallback is
   **statically computed slot indices** — `slot = f(group_index, j, phase)` from
   loop indices that are already live — which needs zero persistent state. The
   counter is preferred because its overflow policy is well defined and a static
   map has to be re-derived whenever the loop structure changes. Which one ships
   is decided by the flag-ON resource read, not by preference. (A flag-ON spill
   does not fail the parity gate — that gate is about the flag-OFF build — but a
   spilling instrumented build pictures a kernel that is not the ratchet kernel,
   and would be disclosed as such.)
4. **In-kernel byte counters** per phase, to avoid the analytic model. Rejected
   by the transferable contract and on the merits: a counter is an atomic on
   the emit path, i.e. an instrument that changes the quantity it measures. All
   bytes are analytic (§4).
5. **Timestamping every thread instead of tid 0.** 512× the stores for no extra
   information: the sites are convergent, so all threads of a CTA cross the
   boundary within one barrier's slack.
6. **Wrapping `eval.py` for arm (a) instead of a standalone driver.** Kept as
   the fallback, not the primary — see §7.

## 4. Traffic model — shape 5, per rank, per epoch (all analytic)

Resolved plan: M=8192, N=4096, K=14336 → `K_local` = 1792, `slice_rows` = 1024,
row 5 = BM 256 / BN 256 / BK 32, `K_TAIL` false, `NR` = 32 → `NG` = 272,
`num_pid_m` = 32, `cols` = 16, `tiles` = 512, `k_iters` = 56,
`eb` = gcd(256, 1024) = 256 → `bands` = 1, `lrow_count` = 4,
`tiles_per_cta` = 2, `rgroup` = 1, `win_rows` = 32 → 8 windows per band.
240 producer CTAs own 2 tiles, 32 own 1. `red_tiles` = 4 × 16 = 64, so each of
the 32 reducers owns exactly 2.

| phase | quantity | value |
|---|---|---|
| `MAINLOOP` | FLOPs per tile | 2·256·256·1792 = 2.3488e8 |
| `MAINLOOP` | FLOPs per rank | 2·M·N·K_local = **1.2025e11** |
| `MAINLOOP` | global-load bytes per tile | (BM+BN)·K_local·2 + (BM+BN)·BK·2 = 1,835,008 + 32,768 = **1,867,776** |
| `MAINLOOP` | global-load bytes per rank | × 512 tiles = **956.3 MB** |
| `EMIT` | bytes per tile | BM · vt · 2 = 256·256·2 = **131,072** (N % BN = 0, so vt = BN always) |
| `EMIT` | bytes per rank | 512 × 131,072 = **67.11 MB** |
| `EMIT` | → xGMI | `owner_of_row(tm·256)` = tm/4 over tm ∈ [0,32), so exactly 1/8 of tiles are local: **7/8 × 67.11 = 58.72 MB** |
| `EMIT` | → local HBM | 1/8 × 67.11 = **8.39 MB** |
| `REDUCE` | bytes per tile | 8 sources × EB · BN · 2 read + EB · BN · 2 written = 1,048,576 + 131,072 |
| `REDUCE` | bytes per rank | 64 tiles → **67.11 MB read + 8.39 MB written**, all local |
| — | HBM total per rank | 956.3 + 8.39 + 67.11 + 8.39 = **1040.2 MB** |
| — | xGMI egress per rank | **58.72 MB** |

The extra `(BM+BN)·BK·2` per tile is exp_09's unconditional clamped prefetch:
the last k-iteration re-reads its own slab. It is 1.8% of the mainloop's load
bytes and is included rather than waved off.

**What "HBM GB/s" means here, stated so the plot cannot mislead.** It is
*global load/store bytes issued by the kernel*, not DRAM traffic. The mainloop
term enjoys L2 reuse across the CTAs sharing an A or B tile, so true DRAM reads
are strictly lower. The strip is labelled accordingly and the discrepancy is
quantified once against exp_20's already-collected `TCC_EA0_RDREQ` group on
shape 5 (`../exp_20_attribution/counters.json`), rather than re-spending a
counter launch.

**Sanity anchors at the exp_20 epoch time of 641.9 µs.** xGMI ≈ 58.72 MB /
641.9 µs ≈ **91.5 GB/s** of per-rank egress; MFMA ≈ 1.2025e11 / 641.9 µs ≈
**187 TFLOP/s**, i.e. ~14% of the MI300X bf16 dense peak — consistent with
LESSONS' finding that the mainloop's cost is dominated by its non-MFMA time,
and a useful falsifier: a strip integrating to a wildly different number means
the phase attribution is broken.

## 5. Tick-rate calibration — measured on gfx942, never assumed

The sibling's "`s_memrealtime` is 100 MHz constant-rate → 10 ns ticks" is a
gfx950 statement, flagged by `../FIGURE_SPECS.md` §5 item 7 as a quantity to
verify. A wrong tick rate silently rescales the entire x-axis, so three
independent readings are taken and recorded in `tick_rate.json`:

1. **Declared rate.** `hipDeviceGetAttribute(hipDeviceAttributeWallClockRate)`
   via `ctypes` against `libamdhip64.so`, with the enum value parsed out of
   `/opt/rocm/include/hip/hip_runtime_api.h` at run time rather than
   hard-coded. Non-fatal if it cannot be resolved.
2. **In-situ regression, the primary.** For each of ≥ 3 operating points
   (shape 5, shape 6, and a short warm shape), take rank 0's device span
   `max(CTA_END) − min(CTA_BEG)` in ticks against the same launch's
   `hipEvent` device time in µs, and fit a line. The **slope is ticks/µs**;
   the intercept absorbs the fixed offset between "kernel dispatch begins" and
   "the first CTA reaches its stamp", which is exactly the bias a single-point
   ratio would fold into the rate.
3. **Cross-check.** exp_21's `sat_calib_kernel` measures the same constant by
   spinning for a target tick count under a host timer. If exp_21's number
   lands, agreement within 1% is required and both are recorded; if it does
   not, methods 1 and 2 stand on their own and result.md says so.

Only method 2 is on the critical path, and it needs no additional GPU launch —
it reads the captures the figure already takes.

## 6. Binning and validation

**Bins: 10 µs**, over `[min ts, max ts]` of **rank 0**, mirroring the sibling
so the two operator instances are comparable.

- `mfma[b]` = |{CTAs whose `[MAINLOOP_BEG, MAINLOOP_END]` interval covers bin
  b}| / **272**. The denominator is `304 − NR` — **the number of CTAs that run
  a mainloop at all**, not 304. Dividing by 304 would cap the proxy at 89.5%
  and invent an idle band that does not exist. The denominator is written into
  every row of `timeline_bins.csv` so no reader has to guess.
- `hbm[b]`, `xgmi[b]` = Σ over phase intervals covering b of
  (phase bytes ÷ phase duration). Phase bytes come from §4, keyed by
  (role, phase, tile count), and are spread uniformly across the phase's
  measured duration.
- **Gate 2, integral, ±10%.** Σ_b `xgmi[b]` · 10 µs must equal 58.72 MB;
  Σ_b `hbm[b]` · 10 µs must equal 1040.2 MB; Σ_b `mfma[b]` · 272 · 10 µs must
  equal the summed mainloop CTA-time. The first two are near-tautological given
  uniform spreading and are therefore checked as *arithmetic* validation — the
  load-bearing checks are the two external ones below.
- **Cross-check A, ±20%.** `amd-smi metric` / `rocm-smi` per-link xGMI
  throughput over a steady-state loop of the same shape, against the strip's
  epoch average of ~91.5 GB/s per rank.
- **Cross-check B, once.** exp_20's `s5_full_g4` counter cell
  (`SQ_VALU_MFMA_BUSY_CYCLES / SQ_BUSY_CYCLES`) against the occupancy proxy's
  epoch average. These measure different things — occupancy of a *phase* vs
  busy-ness of the *MFMA pipe* — so the check is directional (the proxy must be
  the larger of the two, and by a margin consistent with the ~14% of peak in
  §4), not a ±X% equality.

## 7. Arm (a) — reference GEMM+RCCL capture

`b0_ref.py` replicates the reference submission exactly — per-rank
`torch.matmul(x, w.T)` (+ bias) in bf16 followed by
`torch.distributed.reduce_scatter_tensor` — under
`torchrun --nproc_per_node=8`, generating inputs with the harness's own
`generate_input` semantics at **seed 7168**, the shape-5 seed from
`cases_bench.txt`. `b0_rank0_wrap.sh` wraps **rank 0 only** in
`rocprofv3 --kernel-trace`, so the trace is one rank's dispatch stream with no
cross-rank interleaving to untangle.

*Why replicate rather than wrap `eval.py`.* Three reasons, all methodological:
rank-0-only profiling is trivial in a wrapper and awkward under the evaluator's
launcher; the evaluator adds ~92 µs of harness machinery per call (LESSONS)
that would appear in the trace as gaps belonging to no kernel; and the figure
needs a clean single-epoch window, which the evaluator's warm/test/bench
structure does not give. Fidelity is defended, not assumed: `b0_ref.py`'s
per-call device time is compared against `tools/run_reference_arm.sh`'s
shape-5 benchmark number and the comparison is recorded in result.md. If they
disagree by more than 10%, the evaluator wrap becomes the primary and the
standalone driver is dropped.

`b0_kernel_map.py` derives `b0_kernel_map.json` **from the kernel names that
actually appear in the captured trace** — never from a guessed list. Seed
rules: `rccl*`/`mori*`/`*all_reduce*`/`*reduce_scatter*` → xGMI;
`*gemm*`/`*mfma*`/`Cijk*` (rocBLAS's naming) → MFMA;
`*scatter*`/`*quant*`/`*combine*`/`*copy*`/`*elementwise*` → HBM. **Any name
that matches no rule is written to the JSON under `unclassified` with its total
time**, and result.md reports it. Silently bucketing an unknown kernel is how a
mutually-exclusive-strips claim gets faked.

## 8. Arm (c) — rank-1, feasibility

The submission at
`~/amd-master/.../submissions/gemm_rs_rank1_58abcf.py` is hash-frozen
(sha256 `7940fcb8…f0dc5`) and read-only. A `rocprofv3` kernel trace observes the
process from outside and modifies nothing, so **tracing is allowed and is the
way to try**. Three things must hold and are probed by `c_rank1_probe.sh`
before any GPU time:

1. `tools/run_rank1_bench3.sh` already applies four disclosed compatibility
   repairs (iris staging, no-op `sudo`, a triton stub, and `patch_rank1.py`'s
   6→3 `packed_metadata` field count) and writes the patched **copy** to
   `compbench/rank1/submission.py`. Those repairs are pre-existing and disclosed;
   tracing adds none.
2. Its pass order is `benchmark` (warm) → `test` → `benchmark`, and it exists
   because `eval.py`'s test mode hardcodes a 60 s per-rank timeout that a cold
   Triton compile exceeds. **Do not reorder it.** The trace is taken on the
   final `benchmark` pass, after the JIT cache is warm.
3. `rocprofv3` must survive being wrapped around a run that itself spawns 8
   ranks inside the `dhk-eval` container, whose filesystem is separate from
   `dhk-gemmrs` and whose writes must be `chown`ed back to uid 15523.

If any of the three fails, the figure ships with two arms and result.md states
exactly where it stopped. Adding profiler overhead to a competitor's number and
then reporting that number would be worse than shipping two arms.

---

## 9. PATCH PLAN — `distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp`

**Not applied. Awaiting the kernel-edit gate.** One file, additive, entirely
inside `#if HK_GEMM_RS_MI300X_TRACE`. The adapter, the constants header and
the host ABI are **not** touched — everything the instrumentation needs
(`m3::CU_COUNT`, `blockIdx`, the POD) is already visible in this TU, so the
blast radius is one file.

Line numbers are against the 893-line file at the time of writing. Every site
also carries a **content anchor**, because LESSONS records that exp_20's
attribution instrument died when E3 re-indented and re-commented this exact
region: line numbers drift, anchors are re-findable.

| # | line | insert | anchor (executable text, exact indentation) |
|---:|---:|---|---|
| 1 | after 99 | `#ifndef HK_GEMM_RS_MI300X_TRACE` / `#define … 0` / `#endif` | the `#endif` closing the `HK_GEMM_RS_MI300X_TILE_SWEEP` block |
| 2 | after 103 | `namespace hk_trace { … }` — `DEPTH`, the enum, `stamp()`, and the `HK_TRACE_STAMP(id)` macro (a no-op `do {} while (0)` in the `#else`) | `using G = kittens::group<m3::NUM_WARPS>;` |
| 3 | after 139 | `std::uintptr_t trace;` as the last data member | `    int even_k;` (before `dim3 grid() const`) |
| 4 | after 228 | `std::uint64_t* const trace_ring = …(g.trace);` + `int trace_n = 0;` + `HK_TRACE_STAMP(CTA_BEG);` | `    const int pid = blockIdx.x;` |
| 5 | before 368 | `HK_TRACE_STAMP(MAINLOOP_BEG);` | `                zero(C_accum);` |
| 6 | after 482 | `HK_TRACE_STAMP(MAINLOOP_END);` | the `}` closing `for (int k = 0; k < k_iters; ++k)` |
| 7 | before 497 | `HK_TRACE_STAMP(CREDIT_WAIT_BEG);` | `                if (threadIdx.x < (unsigned)bands) {` |
| 8 | after 509 | `HK_TRACE_STAMP(CREDIT_WAIT_END);` | `                if (m3::error_bit_set(errp, m3::ERR_PRODUCER_CREDIT)) return;` |
| 9 | before 515 | `HK_TRACE_STAMP(EMIT_BEG);` | `                for (int b = 0; b < bands; ++b) {` |
| 10 | after 565 | `HK_TRACE_STAMP(EMIT_END);` | the `}` closing that band loop, before `}   // end of the group's tile loop` |
| 11 | before 585 | `HK_TRACE_STAMP(RELEASE_BEG);` | the `#if HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS` guarding `release_payload_system()` |
| 12 | after 603 | `HK_TRACE_STAMP(RELEASE_END);` | the `#endif` after `m3::release_payload_system();` |
| 13 | after 635 | `HK_TRACE_STAMP(PUBLISH_END);` | the `}` closing `if (threadIdx.x == 0) {` publish block |
| 14 | before 640 | `HK_TRACE_STAMP(CTA_END);` | `        return;` at the end of the producer role |
| 15 | before 667 | `HK_TRACE_STAMP(READY_WAIT_BEG);` | `    if (threadIdx.x < m3::WORLD_SIZE) {` |
| 16 | after 675 | `HK_TRACE_STAMP(READY_WAIT_END);` | `    if (m3::error_bit_set(errp, m3::ERR_REDUCER_READY)) return;` |
| 17 | after 676 | `HK_TRACE_STAMP(REDUCE_BEG);` | `    m3::acquire_payload_system();` |
| 18 | after 690 | `HK_TRACE_STAMP(REDUCE_END);` | the `);` closing `m3::pull_sum_bf16_strip_mlp8(` |
| 19 | after 712 | `HK_TRACE_STAMP(CREDIT_PUB_END);` | the `}` closing the reducer's `if (threadIdx.x == 0) {` |
| 20 | after 713 | `HK_TRACE_STAMP(CTA_END);` | the `}` closing the reducer tile loop, before the kernel's closing brace |
| 21 | 813-823 | `, /*trace=*/0` in the aggregate initializer | `mi300x_globals g {` in `prebound::configure` |
| 22 | 882 | append `&mi300x_globals::trace` to the `bind_function` member list | `        &mi300x_globals::even_k);` |

Notes that belong with the patch, not with the reader's memory:

- **Sites 5-14 are inside the group/tile loops** and fire once per tile or once
  per group; sites 15-20 fire once per reducer tile. The counts in §2's table
  come from exactly this placement.
- **Every site is CTA-convergent.** None sits inside `if (threadIdx.x < bands)`
  or any other divergent region, so `if (threadIdx.x == 0)` inside the stamp is
  the only divergence introduced and it is immediately reconverged.
- **Early aborts have no `CTA_END`.** A producer returns at the sticky-abort
  check and at the credit-wait failure; a reducer at its ready-wait failure.
  The host reader treats a CTA with `CTA_BEG` and no `CTA_END` as aborted and
  reports the count. On a clean run that count is zero, and a nonzero count
  invalidates the capture rather than being smoothed over.
- **Site 21 keeps the prebound path compiling but never traced** (`trace = 0`,
  and the stamp is a no-op on a null ring). Arm (b) drives the duck-typed
  `gemm_rs_mi300x` entry, which is the path `harness_lib` uses. This is
  disclosed rather than fixed: adding a `configure_trace` overload would widen
  the patch into the graded launch path for no benefit to this figure.
- **Site 22 makes the flag-ON module take 28 arguments instead of 27.** The
  driver appends the ring pointer to `harness_lib.GemmRS._args` after
  `set_inputs()` — `harness_lib` itself is **read-only and unmodified**.
- **Kernarg growth** under the flag is 8 B on a POD passed by value; the
  flag-OFF build's kernarg segment is bit-identical because the member does not
  exist.

### Validation command for the patch (Phase 2, gate 1)

```bash
bash aug11/exp_22_timeline/parity_gate.sh
python3 aug11/exp_22_timeline/parity_check.py aug11/exp_22_timeline/parity.json
```

`parity_gate.sh` compiles three TUs into `exp_22_timeline/build/` — flag
absent, flag present-and-0, flag present-and-1 — with
`-Rpass-analysis=kernel-resource-usage`, and `parity_check.py` asserts that the
first two produce **byte-identical** VGPR/AGPR/SGPR/scratch/spill/LDS tuples
across all 7 instantiations *and* that both match the post-exp_14 M2 table
hard-coded in the checker. The flag-ON tuple is recorded for disclosure but is
not a pass/fail condition.

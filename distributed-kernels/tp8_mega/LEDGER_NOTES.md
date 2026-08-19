# G-L0b — the boundary-rig instruments (E-B1 / E-B2 / rho / transport-only / standalone-RCCL)

**Built:** 2026-08-18, laptop + node compile only (a sibling agent owned the GPUs;
**nothing in this document is a measurement — every number below is a compile-time receipt**).
**Target:** `distributed-kernels/tp8_mega/{m25_boundary_bench.hip, m25_cdar.cuh}`, gfx950.
**Serves:** `UNDERSTANDING_PLAN.md` §4a G-L0a/G-L0b, Q9 (direct beta), Q10 (h_ledger, absolute
`I_co`), review B4 and B5, `EXPERIMENT_LADDER.md` §1.3 E-B1/E-B2, LAW-63.

---

## 1. What was built

### 1.1 E-B1 — the device phase ledger (`-DM25_LEDGER=1`)

A **per-CTA raw event ring**, not a set of running maxima. Each record is one closed interval
`{t0, t1}` in the device's 100 MHz realtime domain plus a phase code and an aux payload
(rows for compute phases, bytes for transport phases). Everything downstream — per-phase spans,
unions, overlap, the duty histogram, the cross-rank reduction — happens **on the host after the
launch joins**. That choice is what discharges four of LAW-63's five traps structurally instead of
by discipline.

Phase codes (`cdar::ledger_code`, `m25_cdar.cuh`):

| code | brackets | class |
|---|---|---|
| `kernel` / `kernel_end` | CTA entry / exit stamps (degenerate, `t0==t1`) — the slot's time origin | — |
| `mfma1`, `mfma2` | pre- and post-boundary MFMA bursts (aux = rows) | **compute** |
| `commit` | ERS remote store issue + its `vmcnt` drain (aux = bytes) | **transport** |
| `pull` | `mag_pull` remote read (aux = bytes) | **transport** |
| `gather_wait`, `mag_wait` | the two bounded polls, bracketed **outside** the poll bodies | **wait** |
| `drain` | `throttle_vmcnt<0>` + `__syncthreads` + `ers_local_arrive` | **wait** |
| `reduce` | owner tower reduce (local HBM, deliberately **not** counted as fabric) | — |
| `certify` | counted arrival + certificate multicast | — |
| `prod_loop`, `tx_loop`, `duty_loop`, `cons_loop` | **RESERVED, not emitted** — see below | — |

The four whole-loop codes were built, measured, and then **removed**: bracketing a loop means
holding a uniform `u64` pair live across a large region, and four of them cost SGPR spills in a
kernel whose scalar file is already saturated at 105 SGPRs before the instrument. The information
is not lost — a loop's envelope is `first(t0) .. last(t1)` over that loop's own per-iteration phase
events on the same CTA, which the host already computes and prints (`first_us`/`last_us` per
phase). A zero-spill instrument beats a redundant event; the enum values stay reserved so no
existing parser breaks.

**The five LAW-63 guards, and where each is discharged.**

| guard | discharge |
|---|---|
| (a) `s_waitcnt lgkmcnt(0)` after every `s_memrealtime` | one call site, `cdar::realtime_now()`, with the wait welded into the same `asm volatile` block and an `"=s"` (scalar-pair) constraint. No caller can forget it. |
| (b) 100 MHz realtime clock, never the ~2.2 GHz shader clock | `kLedgerHz = 1e8`, `kLedgerTickUs = 0.01`. The bench **prints `hipDeviceAttributeWallClockRate` next to the assumed rate every run** (`LEDGER_CLOCK` line) and emits `LEDGER_CLOCK_WARN` if they disagree — the conversion carries a runtime receipt rather than a comment. |
| (c) running maxima reset between epochs | there are **no running maxima**. The host `hipMemsetAsync`es `count[]` before every iteration (inside the existing pre-iteration memset block, before the timing bracket) and reads the rings back after it. A ring can only ever describe the iteration it was armed for. |
| (d) reduce across ranks to rank-max, never rank 0 | the analysis runs per rank for all 8 ranks and prints a per-rank row **and** an explicit `LEDGER_RANKMAX*` row. There is no rank-0 fast path anywhere in the code. |
| (e) never subtract stamps from different launches | the ring is indexed by **launch slot** (`kLedgerLaunches = 2`: burst1 and burst2 are separate launches in the `compute` and `rccl` arms). Each slot gets its own origin = min of that slot's entry stamps **on that rank**; the host analysis never combines slots, and never combines ranks into one timeline (the realtime counters are per device and have no common origin). |

Brackets are placed **outside** every poll loop and outside every store loop — the t2v6 PROF-arm
shape (`k0pf6gm_device_tile_t2v6.hip`, desc slot 73), which was measured cost-free on the training
chassis. Overflow is counted (high 16 bits of `count[]`), printed as `LEDGER_OVERFLOW`, and the
run says out loud that the spans are lower bounds.

Honesty label on the spans: stamps are taken by **thread 0 of each CTA** with no added
`__syncthreads` (adding one would change the schedule being measured). Phases that already end in
a `__syncthreads` are CTA-exact; the rest are thread-0 spans and should be read as such.

### 1.2 E-B2 — the fabric duty histogram

Computed host-side from the same rings: the transport union (`commit` ∪ `pull`) is binned into
`kDutyHistBins = 64` bins across the launch span, **per rank**, and printed as `LEDGER_DUTY`.
A cross-rank timeline is not built and must not be — the 100 MHz counters have per-device origins.

`h_ledger` is the plan's span-overlap definition:

```
h_ledger = |U_compute  ∩  U_transport| / |U_transport|
```

emitted per rank (`LEDGER_OVERLAP`) and as rank-max (`LEDGER_RANKMAX_OVERLAP`) alongside
`fabric_duty = |U_transport| / span`. The `h_ledger − h_wall` gap is the co-residency tax (ΔI_co);
with the transport-only arm below it becomes absolute rather than differential (review B5).

### 1.3 rho — the `k_inner` ladder (already existed; verified, not rebuilt)

`k_inner` is **argv 6** (`m25_boundary_bench.hip`, default 64) and scales **only** the MFMA loop's
trip count in `mfma_burst`. No rebuild is needed to walk rho, and this is now **confirmed working
in anger**: the sibling runner's G-L0a ladder completed on it
(`~/anatomy_g0/rho/run_rho_ladder.sh`, results in `rho_ladder.jsonl`) with the session-calibrated
map

| rho (nominal) | `k_inner` |
|---:|---:|
| 0.65 | 2048 |
| 1.0 | 3078 |
| 2.0 | 6029 |
| 3.0 | 9040 |

**Use exactly these** — they are already calibrated against this node/session, and re-deriving them
would break comparability with the four-arm ladder that has already run. The argv order the runner
used is identical to this rig's: `<mode> <tokens> <slab_rows> <iters> <depth> <k_inner> <bps>` at
`16384 256 7 4 <k> 2`.

Caveat now written into the file header: `k_inner` scales FLOPs **and** operand-load traffic
together (one rotating load per iteration), so intensity and duration are **not** independent here.
Q4/M2's `bytes_per_mfma` argv is R5's build item and was deliberately **not** added — adding it
would perturb the device image this session proved byte-identical.

### 1.4 Transport-only arm

Two ways in, both shipped:

- **`mode=transport`** (and `mode=transport_fused`) — host-side aliases that run the phased (resp.
  fused) schedule with `k_inner` forced to **0**. Same schedule, same slabs, same certificates,
  same verification (still checks the exact bf16 sum 36.0); zero MFMA issued. **No device-code
  change**, which is why the default `.text` stayed byte-identical.
- **`-DM25_NOCOMPUTE=1`** — a separate binary in which the compute body is genuinely *compiled
  out* (`mfma_burst` returns 0 with no operand loads). Use this only if the residue below matters.
- **`mode=compute0`** — the control that BOUNDS the residue: the compute arm at `k_inner=0`. It
  measures exactly what a runtime-zero trip count still costs (two surviving operand loads per
  call). If `compute0 ≈ 0`, `mode=transport` in the default binary *is* the transport-only arm and
  the `M25_NOCOMPUTE` binary is unnecessary.

This is what makes `I_co` absolute: `I_co(phased) = wall(phased) − wall(compute) − wall(transport)`,
with all three terms measured directly in one session on one binary.

### 1.5 Standalone-RCCL arm, and direct beta

- **`mode=stdrccl`** — a bare, timed `ncclAllReduce` over the identical 235 MB payload
  (`tokens × 14,336` B, bf16 sum), in the **same binary**, with the source re-copied before each
  iteration so the verification (sample must read 36.0) stays live. RCCL reads
  `NCCL_ALGO` / `NCCL_PROTO` / `NCCL_NCHANNELS` / `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` from
  the environment as usual; the bench **prints all five into the run record**
  (`BOUNDARY_BENCH_RCCLENV`) so an env sweep is self-describing in `runs.jsonl`.
- **`mode=beta`** — alternates `transport` and `stdrccl` iteration by iteration inside one
  invocation (LAW-60: cross-session lone-arm contrasts are void) and prints
  `BETA_DIRECT ... beta=<ours/rccl>`. This replaces both banked values (the subtraction-derived
  1.506× and the convention-mixed 1.363×) with a ratio of two direct measurements — review B4.
- Bandwidth is printed in **two explicitly labelled conventions on separate fields**
  (`logical_bytes_over_wall_GBps`, `busbw_convention_GBps`). One convention per table when quoting.

### 1.6 Arms now available (argv 1)

`compute` · `compute0` · `phased` · `fused` · `transport` · `transport_fused` · `rccl` ·
`stdrccl` · `beta`. An unknown mode is a hard error listing the valid set (it used to fall through
to `fused` silently).

**Every new arm is host-side dispatch.** No arm adds a kernel instantiation, which is why the
device image is arm-count-invariant and the parity gate below is meaningful.

---

## 2. Build-identity receipts (gfx950, ROCm 7.2.53211, AMD clang 22.0.0git roc-7.2.4)

Gate script: `distributed-kernels/tp8_mega/ledger_build_gate.sh` (committed). It compiles
`--cuda-device-only`, unbundles the clang offload bundle, dumps the `.text` section and sha256s it,
then re-compiles with `-Rpass-analysis=kernel-resource-usage` for the resource tuple. Baseline =
the pre-instrument sources extracted from git.

```
./ledger_build_gate.sh <base_src_dir> <new_src_dir> [outdir]
```

### 2.1 `.text` identity — PASS both ways

```
f7d7daa973a2cb7ebdc6cd7218e5dc51ef55698a55fee66d58a642f77cb1d03f  base_off       (.text 65,644 B)
f7d7daa973a2cb7ebdc6cd7218e5dc51ef55698a55fee66d58a642f77cb1d03f  new_off        (.text 65,644 B)
43fa5ed000dfdd4a95e1d663e4e9ac84a426a049a3796d60755f9d3e944673cd  new_on         (.text 85,008 B)
2a4e7eddb5ba3aae136f96095e4e177dd6a908c2d7c5a6e19387288818a2f620  new_nocompute  (.text 46,112 B)

GATE_TEXT_PARITY  PASS   (M25_LEDGER=0 .text == pre-instrument .text, byte for byte)
GATE_TEXT_DIFFERS PASS   (M25_LEDGER=1 .text != M25_LEDGER=0 .text)
```

`base_off` is built from the pre-instrument sources at `2578a72a`. The default build of the
instrumented file is therefore **provably the same device program** — every new arm is host-side
dispatch, and every device addition is behind `M25_LEDGER`.

### 2.2 Resource receipt — zero spills, zero scratch, unchanged occupancy

Depth-4 specialization shown; every other depth specialization is identical.

| kernel | build | TotalSGPR | VGPR | AGPR | scratch B/lane | occ waves/SIMD | SGPR spill | VGPR spill |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `<compute,4>` | base / `LEDGER=0` | 34 | 16 | 4 | 0 | 8 | 0 | 0 |
| `<compute,4>` | `LEDGER=1` | 49 | 20 | 4 | **0** | **8** | **0** | **0** |
| `<phased,4>` | base / `LEDGER=0` | 105 | 44 | 4 | 0 | 7 | 0 | 0 |
| `<phased,4>` | `LEDGER=1` | 106 | 50 | 4 | **0** | **7** | **0** | **0** |
| `<fused,4>` | base / `LEDGER=0` | 105 | 44 | 4 | 0 | 7 | 0 | 0 |
| `<fused,4>` | `LEDGER=1` | 106 | 50 | 4 | **0** | **7** | **0** | **0** |

**Delta: +1 SGPR, +6 VGPR, +0 AGPR, +0 scratch, +0 spills, occupancy unchanged.**

Getting to zero spills took three structural passes, and the failures are worth recording because
they are the reusable part:

1. **Thread-0-only stamps** (the obvious shape) — 0 spills but **+33 VGPRs**, and the compiler's
   reported occupancy fell 7 → 5 waves/SIMD. Cause: values produced inside `if (threadIdx.x == 0)`
   are divergent, so the whole cursor and every live timestamp land in *vector* registers.
2. **CTA-uniform stamps** — every thread reads the clock (the value is uniform, so it lands in
   SGPRs) and only the 24 B ring write is predicated on thread 0. VGPR cost fell to +7 and
   occupancy came back, but the scalar file is *already saturated* at 105 SGPRs before the
   instrument, so it bought **4 SGPR spills**. Demoting the long-lived entry stamp to its own
   degenerate event halved that to 2; slimming the cursor to two words (compile-time capacity,
   overflow by saturation instead of a second counter) did **not** clear the last 2.
3. **Dropping the four whole-loop brackets** — cleared it. Each one held a uniform `u64` pair live
   across a large region; four of them were the last 2 spills. The information was redundant: a
   loop's envelope is `first(t0)..last(t1)` over its own per-iteration events, which the host
   already reports.

The transferable lesson for R2c (the DP-body ledger port): on a saturated occupancy-1 kernel the
binding cost of a phase ledger is **not** the timestamp reads, it is the *liveness* of the bracket
variables. Bracket leaves, never whole regions; make the stamps uniform so they land in SGPRs; and
count events by saturation rather than with a second counter.

---

## 3. What could NOT be verified without GPUs

1. **Wall parity** — the plan's rule is *instrumented-vs-uninstrumented wall parity within the
   noise band, else the arm's spans are void*. This is a runtime check and was not run. It is the
   **first** thing next session must do (§4, step 0). The receipts above are as clean as a compile
   can make them (+1 SGPR, +6 VGPR, zero spills, zero scratch, unchanged occupancy), which is a
   *reason to expect* parity, not evidence of it — LAW-32 (a megakernel made slow purely by the
   register allocator) is precisely the failure that resource tuples alone did not predict.
2. **That the rings never overflow at the shipped shape.** `cd::kLedgerCap = 256` events/CTA/slot
   was sized against the worst case in the G-L0 matrix (~50 events at `slab_rows=256, bps=2`), but
   overflow is only *observed*, not proven. A CTA whose `count[]` comes back equal to the cap
   filled its ring; `LEDGER_OVERFLOW` fires loudly and says the spans are lower bounds.
3. **That the clock really is 100 MHz on this part** — the `LEDGER_CLOCK` line prints the device's
   own `hipDeviceAttributeWallClockRate` for comparison; nobody has read that line yet.
4. **Correctness of the new arms end to end** (they reuse the verified CDAR path and the existing
   RCCL path, and every arm still returns `BOUNDARY_BENCH_PASS`/`FAIL` on the exact bf16 sum, but
   no arm has been executed).
5. **The rho ⇄ k_inner mapping** (§1.3) — calibrate in-session, do not trust the linear
   extrapolation.

---

## 4. Next-session run commands — the full G-L0 matrix

Build both binaries once (CPU-only work; safe while GPUs are busy):

```bash
cd ~/Distributed-HipKittens/distributed-kernels/tp8_mega
ROOT=$(cd ../.. && pwd)
hipcc --offload-arch=gfx950 -std=c++20 -O3 -Wno-pass-failed -I $ROOT \
      m25_boundary_bench.hip -o m25_boundary_bench -lrccl
hipcc --offload-arch=gfx950 -std=c++20 -O3 -Wno-pass-failed -I $ROOT -DM25_LEDGER=1 \
      m25_boundary_bench.hip -o m25_boundary_bench_ledger -lrccl
# build-identity receipt for the manifest (also CPU-only):
bash ledger_build_gate.sh <baseline_src_dir> . ./gate_out
```

**Step 0 — wall parity (blocking; without it the spans are void).** Same shape, both binaries,
alternating, K=3 each:

```bash
for k in 1 2 3; do
  ./m25_boundary_bench        phased 16384 256 7 4 2048 2
  ./m25_boundary_bench_ledger phased 16384 256 7 4 2048 2 | grep BOUNDARY_BENCH
  ./m25_boundary_bench        fused  16384 256 7 4 2048 2
  ./m25_boundary_bench_ledger fused  16384 256 7 4 2048 2 | grep BOUNDARY_BENCH
done
```
Pass bar: the ledger build's median wall is inside the rig's own band (which falls out of the K=3
repeats below at no extra cost). If it is not, **the ledger arm's spans are reported as bounds and
the perturbation is published as a finding** (kill criterion §6 of the plan).

**Step 1 — G-L0b, the rho × arm matrix, K=3, ledger on.**

*Already done, do not repeat:* the sibling runner completed the **four-arm** G-L0a ladder
(`compute`, `phased`, `fused`, `rccl` × rho ∈ {0.65, 1, 2, 3} × K=3) —
`~/anatomy_g0/rho/{run_rho_ladder.sh, rho_ladder.jsonl}`. What is missing is exactly what this
build adds: the ledger-on repeat of those cells, plus the four new arms.

Reuse the runner's harness verbatim (it already emits the `runs.jsonl` schema, timestamps each
cell, and classifies `PASS`/`FAIL`/`nooutput`); change only `BIN` to the ledger binary and the arm
list. argv order is identical: `<mode> <tokens> <slab_rows> <iters> <depth> <k_inner> <bps>`.

```bash
# rho-DEPENDENT cells — ledger on, same k_inner map as the completed ladder
for k in 1 2 3; do                                 # K=3 repeats
  for pair in 0.65:2048 1.0:3078 2.0:6029 3.0:9040; do
    KI=${pair##*:}
    for ARM in compute phased fused rccl; do
      ./m25_boundary_bench_ledger $ARM 16384 256 7 4 $KI 2
    done
  done
done

# rho-INDEPENDENT cells — these arms force k_inner to 0, so ONE cell per repeat
for k in 1 2 3; do
  ./m25_boundary_bench_ledger transport       16384 256 7 4 0 2   # direct(ours)
  ./m25_boundary_bench_ledger transport_fused 16384 256 7 4 0 2
  ./m25_boundary_bench_ledger compute0        16384 256 7 4 0 2   # residue control
  ./m25_boundary_bench_ledger stdrccl         16384 256 7 4 0 2   # direct(rccl)
  ./m25_boundary_bench_ledger beta            16384 256 8 4 0 2   # direct beta
done
```

Notes that matter for the manifest:
- `transport`, `transport_fused`, `compute0`, `stdrccl` and `beta` **ignore `k_inner`** — one cell
  per repeat, not one per rho point. Running them per-rho only burns node time.
- `beta` wants an **even** `iters` so both arms get equal n.
- Every run appends to `runs.jsonl` before the next starts; the first invocation of the session is
  discarded; clocks/temps logged; a hang is a manifest outcome value, not a dropped point.
- Keep the ledger flag **frozen** across any ladder that locates a knee (the flag is a build, so
  this means: do not mix ledger and non-ledger cells inside one comparison).

**What the matrix then yields** (the reason each arm is there):
- `I_co(phased) = phased − compute − transport`, and the same for `fused` — **absolute**, not the
  ΔI_co that `h_ledger − h_wall` alone gives (review B5).
- `beta = transport / stdrccl` from two direct measurements in one binary (review B4).
- `h_ledger` per rho from `LEDGER_RANKMAX_OVERLAP`, to be read against
  `h* = 1 − (1 − I_co/T_vendor)/beta` at the **now directly measured** beta — the G-L3 readmission
  test for fused CDAR.

**Step 2 — Q9's env sweep on the standalone arms** (R9; same binary, no rebuild):

```bash
for ALGO in Ring Tree; do for PROTO in Simple LL LL128; do for NCH in 4 8 16 32; do
  NCCL_ALGO=$ALGO NCCL_PROTO=$PROTO NCCL_MIN_NCHANNELS=$NCH NCCL_MAX_NCHANNELS=$NCH \
    ./m25_boundary_bench_ledger beta 16384 256 8 4 0 2
done; done; done
```
The `BOUNDARY_BENCH_RCCLENV` line records the environment in the run itself, so the sweep is
self-describing.

**Step 3 — the transport-only compile-out cross-check** (only if `compute0` reads non-negligible):

```bash
hipcc --offload-arch=gfx950 -std=c++20 -O3 -Wno-pass-failed -I $ROOT -DM25_NOCOMPUTE=1 \
      m25_boundary_bench.hip -o m25_boundary_bench_nocompute -lrccl
./m25_boundary_bench_nocompute transport 16384 256 7 4 0 2
```

### Output lines to parse

| prefix | carries |
|---|---|
| `BOUNDARY_BENCH_CFG` | the full config incl. `ledger=` and `nocompute=` flags |
| `BOUNDARY_BENCH_RCCLENV` | the five NCCL env vars as they were seen |
| `BOUNDARY_BENCH` | per-arm median/min/max wall, n, timeout and verify counters |
| `  BW arm=` | both bandwidth conventions, separately labelled |
| `BETA_DIRECT` | ours / rccl / beta from two direct measurements |
| `LEDGER_CLOCK` (+ `_WARN`) | assumed vs device-reported clock rate |
| `LEDGER` / `LEDGER_PHASE` / `LEDGER_OVERLAP` / `LEDGER_DUTY` | **per rank** |
| `LEDGER_RANKMAX*` | the cross-rank reduction — **quote these, never the per-rank rows** |
| `LEDGER_OVERFLOW` | rings overflowed ⇒ spans are lower bounds, re-run at a larger cap |

---

## 5. Deliberately not built

- **E-B3** (transport as a template arg + argv; `store_towers` is still hardcoded at the phased
  commit site). Not in G-L0's scope; it is X4/decode-K1's build item.
- **`bytes_per_mfma`** (Q4/M2, intensity ⟂ duration) — R5's build item; adding it now would have
  perturbed the device image this session proved identical.
- **The DP-body ledger port (R2c)** — a separate build item on the mega chassis with its own
  acceptance (all five guards + wall parity). Nothing here touches it.

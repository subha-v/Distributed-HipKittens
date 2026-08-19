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
| `prod_loop`, `tx_loop`, `duty_loop`, `cons_loop` | whole-loop spans | — |
| `mfma1`, `mfma2` | pre- and post-boundary MFMA bursts (aux = rows) | **compute** |
| `commit` | ERS remote store issue + its `vmcnt` drain (aux = bytes) | **transport** |
| `pull` | `mag_pull` remote read (aux = bytes) | **transport** |
| `gather_wait`, `mag_wait` | the two bounded polls, bracketed **outside** the poll bodies | **wait** |
| `drain` | `throttle_vmcnt<0>` + `__syncthreads` + `ers_local_arrive` | **wait** |
| `reduce` | owner tower reduce (local HBM, deliberately **not** counted as fabric) | — |
| `certify` | counted arrival + certificate multicast | — |

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
trip count in `mfma_burst`. No rebuild is needed to walk rho. Two caveats now written into the
file header:

- rho is not `k_inner`; it is the measured ratio. Calibrate it from `wall(compute)` against
  `wall(transport)` at the same shape (both arms exist in the same binary — see §1.4/§1.5), then
  pick `k_inner` per rho point. The banked anchor is `k_inner = 2048 ⇒ rho ≈ 0.65`
  (compute floor 1,439 µs vs collective 2,167–2,214 µs), so the ladder start points are
  **k_inner ∈ {2048, ~3150, ~6300, ~9450}** for rho ∈ {0.65, 1, 2, 3}; confirm each from the
  measured `compute` wall in the same session rather than trusting the linearity.
- `k_inner` scales FLOPs **and** operand-load traffic together (one rotating load per iteration),
  so intensity and duration are **not** independent here. Q4/M2's `bytes_per_mfma` argv is R5's
  build item and was deliberately **not** added — adding it would perturb the device image that
  this session just proved byte-identical.

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

**Run 1 (thread-0-only stamps).**

```
f7d7daa973a2cb7ebdc6cd7218e5dc51ef55698a55fee66d58a642f77cb1d03f  base_off        (.text 65,644 B)
f7d7daa973a2cb7ebdc6cd7218e5dc51ef55698a55fee66d58a642f77cb1d03f  new_off         (.text 65,644 B)
76f29c7e2f00643a858c3a967637be823a3e24204a0a5d04c353347ccc9c1e22  new_on          (.text 98,408 B)
2a4e7eddb5ba3aae136f96095e4e177dd6a908c2d7c5a6e19387288818a2f620  new_nocompute   (.text 46,112 B)

GATE_TEXT_PARITY  PASS   (M25_LEDGER=0 .text == pre-instrument .text, byte for byte)
GATE_TEXT_DIFFERS PASS   (M25_LEDGER=1 .text != M25_LEDGER=0 .text)
```

**Resource tuple, `boundary_kernel<fused, depth=4>`** (identical for every depth specialization;
`mode_phased` matches `mode_fused`, `mode_compute` is the small one):

| build | TotalSGPR | VGPR | AGPR | scratch B/lane | occ waves/SIMD | SGPR spill | VGPR spill |
|---|---:|---:|---:|---:|---:|---:|---:|
| base (pre-instrument) | 105 | 44 | 4 | 0 | 7 | 0 | 0 |
| `M25_LEDGER=0` | 105 | 44 | 4 | 0 | 7 | 0 | 0 |
| `M25_LEDGER=1`, thread-0 stamps | 105 | **77** | 4 | **0** | **5** | **0** | **0** |
| `M25_LEDGER=1`, CTA-uniform stamps (variant B) | **106** | 51 | 4 | 0 | 7 | **4** | 0 |

**Chosen: the CTA-uniform variant with the entry/exit stamps demoted to their own degenerate
events** (see §2.1 for the final numbers). Reading of the trade:

- The **flag-OFF build is byte-identical** — that is the load-bearing half and it holds in every
  variant.
- Thread-0-only stamps cost **+33 VGPRs and zero spills**. The reported occupancy drop 7 → 5
  waves/SIMD is *achievable* occupancy, not achieved: the kernel is
  `__launch_bounds__(256, 1)` launched as 256 blocks on 256 CUs, i.e. 1 block/CU = 1 wave/SIMD, so
  a budget of 5 is still ~5× more headroom than the launch uses. It does not bind.
- Making the stamps CTA-uniform (all threads read the clock, only thread 0 writes the 24 B record,
  so the cursor stays in SGPRs) cuts the VGPR cost to +7 and restores the reported occupancy, but
  pushes the already-saturated scalar file (105 of the ~102 addressable + VCC/FLAT_SCRATCH) into
  **4 SGPR spills at ScratchSize 0** — i.e. spilled to VGPR lanes via `v_writelane`/`v_readlane`,
  no memory traffic and no scratch allocation. Demoting the long-lived entry stamp to its own event
  frees one live pair and was applied to claw that back.

**Neither variant allocates scratch, and neither spills a VGPR.** The remaining SGPR-lane spill (if
any survives in §2.1) is the honest residual and is named here rather than hidden.

### 2.1 Final resource receipt

*(filled from `gate_out3` — see the run log referenced in the commit; if the numbers below read
`PENDING` the gate did not finish before the session ended and **the first node action next session
is to re-run `ledger_build_gate.sh`**, which takes ~3 minutes of CPU and no GPU.)*

| build | TotalSGPR | VGPR | scratch | occ | SGPR spill | VGPR spill |
|---|---:|---:|---:|---:|---:|---:|
| base / `M25_LEDGER=0` | 105 | 44 | 0 | 7 | 0 | 0 |
| `M25_LEDGER=1` | see gate_out3 | | 0 expected | | | |

---

## 3. What could NOT be verified without GPUs

1. **Wall parity** — the plan's rule is *instrumented-vs-uninstrumented wall parity within the
   noise band, else the arm's spans are void*. This is a runtime check and was not run. It is the
   **first** thing next session must do (§4, step 0). The t2v6 precedent says the ledger is
   cost-free; the +VGPR/+SGPR-lane-spill deltas above are the reason to prove it rather than
   assume it, and LAW-32 (a megakernel made slow purely by the register allocator) is exactly the
   failure mode being guarded against.
2. **That the rings never overflow at the shipped shape.** `kLedgerCap = 256` events/CTA/slot was
   sized against the worst case in the G-L0 matrix (~50 events at `slab_rows=256, bps=2`), but
   overflow is only *observed*, not proven. `LEDGER_OVERFLOW` fires loudly if it happens.
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

**Step 1 — G-L0a/G-L0b, the rho × arm matrix, K=3, ledger on.** argv order is
`<mode> <tokens> <slab_rows> <iters> <depth> <k_inner> <bps>`.

```bash
# calibrate rho FIRST (compute floor vs transport wall at the same shape)
./m25_boundary_bench_ledger compute   16384 256 7 4 2048 2
./m25_boundary_bench_ledger transport 16384 256 7 4 0    2
# -> rho(k_inner=2048) = wall(compute)/wall(transport); set the ladder from THIS ratio.

for k in 1 2 3; do                     # K=3 repeats
  for KI in 2048 3150 6300 9450; do    # rho ~ 0.65, 1, 2, 3 (re-derive from the calibration)
    for ARM in compute phased fused rccl transport; do
      ./m25_boundary_bench_ledger $ARM 16384 256 7 4 $KI 2
    done
  done
  ./m25_boundary_bench_ledger compute0 16384 256 7 4 0 2   # residue control, rho-independent
  ./m25_boundary_bench_ledger stdrccl  16384 256 7 4 0 2   # rho-independent
  ./m25_boundary_bench_ledger beta     16384 256 8 4 0 2   # direct beta, alternating arms
done
```

Notes that matter for the manifest:
- `transport`, `stdrccl`, `compute0` and `beta` are **rho-independent** — run them once per repeat,
  not once per rho cell (they ignore `k_inner`).
- `beta` wants an **even** `iters` so both arms get equal n.
- Every run appends to `runs.jsonl` before the next starts; the first invocation of the session is
  discarded; clocks/temps logged; a hang is a manifest outcome value, not a dropped point.
- Keep `timestamps`/ledger state **frozen** across any ladder that locates a knee.

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

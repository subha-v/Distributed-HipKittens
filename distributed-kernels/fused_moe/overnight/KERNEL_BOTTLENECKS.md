# Current bottlenecks of the CTA-specialized fused-MoE megakernel

Purely descriptive. This file states what the kernel is, where its time goes,
what we have measured, and what we have ruled out. It deliberately contains no
recommendations, no proposed designs, and no requirements — those live
elsewhere.

Every number is either a 5-rotation median rank-max p50 from a fixed synthetic
prefill benchmark, or a single-epoch in-kernel `s_memrealtime` stamp where
labelled. Claims are tagged **MEASURED** or **INFERRED**.

---

## 1. What the kernel is

A persistent, single-launch megakernel implementing one fused Mixture-of-Experts
layer across **8× MI350X (gfx950)** on one node, communicating over Infinity
Fabric through a `mori` symmetric heap.

- **Grid:** 256 CTAs × 256 threads.
- **Occupancy: exactly one block per CU.** 256 ArchVGPR + 256 AGPR gives one
  wave per SIMD, and 155,428 B of LDS (of 163,840) independently forces one
  block per CU. **MEASURED** (compiler resource report).
- **Phases** M0–M9 within the single launch, separated by grid barriers:
  M0 retire/zero → M1 token dispatch send → M2 dispatch receive → M3–M5 plan
  (sort/scatter) → M6 expert GEMM 1 → M7 expert GEMM 2 → M8/M9 combine
  (reduce + return all-to-all).

**The CTA specialization.** At the M6/M7 boundary, `C` of the 256 CTAs become a
**communication service pool**. They skip M7 entirely and instead consume a
queue of tile-completion events produced by the compute CTAs, pushing each
completed row-slice of the combine payload into the destination rank's landing
buffer over Infinity Fabric and publishing per-row readiness flags. The other
`256 − C` CTAs run M7's MFMA. As each compute CTA finishes M7 it **falls
through** into the same drain loop, so the pool grows from `C` to 256 over the
course of the phase. Event consumption is by monotonic ticket, so pool
membership is not a correctness parameter.

**Current operating point:** `C = 64`, one N-chunk per push, flush every 16
completed rows.

---

## 2. Where the time goes

Phase profile of the **homogeneous** kernel (no specialization), total 6,989 µs.
**MEASURED**, single-epoch stamps.

| phase | µs | share |
|---|---:|---:|
| dispatch (M0–M2) | 1,193 | 17% |
| plan (M3–M5) | 415 | 6% |
| **M6 — expert GEMM 1** | **2,539** | **36%** |
| M7 — expert GEMM 2 | 1,586 | 23% |
| combine (M8/M9) | 1,255 | 18% |

**CTA sensitivity of each phase**, measured by reserving CTAs that do no work
(so capacity changes but no extra memory traffic is introduced). Removing 62 of
256 CTAs:

| phase | change | reading |
|---|---:|---|
| plan + M6 | **+0.5%** (2,954 → 2,970 µs, flat across C = 2/16/32/48/64) | not CTA-throughput-bound |
| M7 | **+27%** (1,586 → 2,020 µs, linear in C) | CTA-throughput-bound |
| combine | ~flat | not CTA-throughput-bound |

**MEASURED.** The entire capacity cost of reserving CTAs falls on M7.

Same phases with specialization active at `C = 64`:

| phase | specialized | homogeneous | delta |
|---|---:|---:|---:|
| plan | 413 | 415 | −2 |
| M6 | 2,588 | 2,539 | +49 |
| **M7** | **2,835** | **1,586** | **+1,249** |
| combine | 446 | 1,255 | **−809** |

Whole-kernel result, two independent 5-rotation campaigns, all correctness gates
green:

| arm | campaign 1 | campaign 2 |
|---|---:|---:|
| production baseline | 7,729.4 | 7,725.2 |
| homogeneous megakernel | 6,910.9 | 6,894.0 |
| **CTA-specialized megakernel** | **6,866.1** | **6,869.1** |

---

## 3. The bottleneck

**M7 costs +1,249 µs under specialization. That single term is the entire
remaining gap.** Every other phase is at or better than the homogeneous
baseline. It decomposes as:

- **~434 µs capacity** — M7 running on 192 CTAs instead of 254. **MEASURED**
  independently via the reserve-but-do-nothing mode (M7 = 2,020 µs).
- **~815 µs interference** — the *same* 192 compute CTAs, but with the service
  pool concurrently moving bytes. 2,020 → 2,835 µs. **MEASURED.**

The combine side is already close to saturated as a source of gains: it is down
to 446 µs from 1,255, and at `C = 96` it reaches 108 µs. There is little left to
buy there.

### 3.1 What the interference is NOT — each ruled out by measurement

1. **Not MFMA issue-slot contention.** Occupancy is one block per CU, so a
   service CTA is never co-resident with an MFMA CTA. There is no SIMD to
   share. **MEASURED** (resource report).
2. **Not cache capacity or pollution.** Both the source load and the peer store
   were marked non-temporal, and disassembly confirms the `nt` bits reached the
   ISA (`flat_load_dwordx4 … nt`, `flat_store_dwordx4 … nt`, 3 of 352 dwordx4
   accesses where previously none). Effect on M7: +14 µs at C=64, −74 at C=48,
   −7 at C=32, −11 at C=96 — no consistent sign. **MEASURED.**
3. **Not predominantly atomic traffic.** The readiness protocol issued ~1.58M
   atomics per rank per epoch. Removing ~698k of them (two operations proven
   redundant in the current configuration) bought **−306 µs** of M7. At that
   rate, removing *every* remaining atomic is worth **≤235 µs**, under 30% of
   the 815. **MEASURED**, with the extrapolation labelled **INFERRED**.
4. **Not peak-bandwidth saturation.** M7 reads ~470 MB of weights in 1,586 µs =
   **~296 GB/s against 8 TB/s of HBM**. The pool moves ~312 MB over xGMI in
   ~2.8 ms = **~111 GB/s against ~537 GB/s of aggregate egress**. Neither is
   near a roofline. **MEASURED** (times) / **INFERRED** (byte counts from the
   shapes).
5. **Partly per-XCD L2, and that part is quantified.** Confining the pool to
   whole XCDs — so the compute dies' L2s stay clean — cut the interference
   **36%** (M7 3,125 → 2,723 at matched CTA counts). It was a net loss overall
   because the pool, squeezed onto two dies, slowed 54%. **MEASURED.**

**Residual explanation: request-level contention in the shared memory path
beyond the XCD. INFERRED — we have no hardware-counter evidence for it, and
that is the single largest gap in our understanding of this kernel.**

### 3.2 Traffic volumes

Per rank per epoch the combine payload moves **~936 MB**: M7's epilogue
accumulates into a local buffer (~312 MB of writes), the service pool reads that
buffer (~312 MB), the service pool writes the peer's landing buffer (~312 MB).
**INFERRED** from shapes (T_loc ≈ 21,816 rows × 7,168 bf16 = 14,336 B/row).

---

## 4. Structural constraints observed

1. **Register allocation is static and per-kernel.** A service CTA is allocated
   the same 256 ArchVGPR + 256 AGPR as an MFMA CTA, never issues MFMA, and
   cannot obtain additional registers. An attempt to give the copy 4-deep
   memory-level parallelism spilled its 64 B/lane staging array to scratch
   (60 → 128 B/lane) and the gain did not materialize. **MEASURED.**
2. **The copy loop runs at MLP = 1.** The emitted inner loop is one
   `flat_load_dwordx4`, an unconditional `s_waitcnt vmcnt(0)`, one
   `flat_store_dwordx4` — ten instructions, `#pragma unroll 1` honoured, no
   software pipelining. One outstanding load per lane, 1,024 B in flight per
   64-lane wave. **MEASURED** (disassembly).
3. **Coalescing atomics is severely counterproductive.** Transposing the arrival
   counters so 32 lanes hit 2 cache lines instead of 32 made the kernel **≥10×
   slower**; the run was abandoned rather than completed. Read-modify-writes
   serialize at the line, whereas 32 scattered RMWs proceed in parallel across
   L2 banks. **MEASURED** (lower bound).
4. **The dispatch all-to-all has no exposed wait.** Its per-`(source, chunk)`
   poll reports **max-spins-to-success = 0** against a 2,000,000 spin limit. All
   eight ranks run identical work on identically-shaped data and enter the
   exchange within microseconds, so the transfer is already latency-hidden by
   symmetry. **MEASURED.**
5. **The second GEMM's weights cannot be made resident.** W2 is ~470 MB against
   a 256 MB last-level cache, so M7 streams its weights every epoch. **INFERRED**
   from shapes and the documented cache size.
6. **~155 KB of LDS is allocated for the GEMM and is dead during the
   communication phase.** Only ~8 KB is formally unallocated. **MEASURED**
   (resource report) / **INFERRED** (that the GEMM region is dead post-M7).

---

## 5. Facts about the communication protocol, for context

- Readiness is detected per `(row, N-chunk)` by an arrival counter that counts
  contributing 32-row blocks; a row is published when all 16 of its chunks have
  been pushed. Arrival counts total ~535,040 per rank per epoch — one per
  `(block, row, chunk)` triple — and that count is fixed by the plan, not by the
  protocol. **MEASURED** (event/row counts) / **INFERRED** (the triple count).
- Contributions to a given row come from **several CTAs**, and `s_waitcnt vmcnt`
  is per-wavefront, so a publishing wave's drain does not cover other producers'
  stores. A static analysis identified this as an ordering hole for
  configurations with more than one chunk per push group; a dedicated
  sensitivity test (8 open-hole trials and 4 control trials, each with a
  different routing seed, comparing the specialized arm's correctness figures
  against the homogeneous arm's within the same run) found **12/12 identical, no
  divergence**. The hole is real in source and unobserved in practice.
  **MEASURED.**
- The destination landing buffer is never zeroed per epoch. This is sound in the
  current design only because every write is a full overwrite. **MEASURED**
  (host allocation zeroes once).
- There is **no cross-rank ordering point between the epoch start and the second
  GEMM**. **MEASURED** (source audit).

---

## 6. Hardware capabilities established by direct test

- **Remote packed-bf16 atomic accumulation over xGMI works and is atomic across
  devices.** On coarse-grained memory: one remote writer per cell, 128
  sequential read-modify-writes, 256/256 cells exact; two devices contending on
  the same cells, 256/256 exact. Disassembly confirms
  `global_atomic_pk_add_bf16` with no compare-and-swap fallback. It works
  despite the builtin carrying no memory-scope argument. **MEASURED.**
- Whether the same holds on the `mori` symmetric heap — HIP-VMM allocated with
  an uncached heap type, where HIP documents fine-grained global memory as
  undefined behaviour for the unsafe-atomic API — is **UNKNOWN**. The kernel
  does already perform remote *integer* system-scope read-modify-writes on that
  heap in production. **MEASURED** (that integer RMWs work).

---

## 7. Summary of the open questions the measurements leave

- What shared resource serializes when ~64 CUs stream 16 B/lane global loads and
  cross-GPU peer stores while ~192 CUs run MFMA with streaming weight loads?
  36% is attributable to the per-XCD L2; the mechanism behind the other 64% is
  unidentified.
- How to obtain hardware-counter evidence for a *sub-phase* of a single
  long-running persistent kernel launch, where whole-kernel counters average all
  phases together.
- What limits M6 — the largest single phase at 36% of the kernel — given it is
  demonstrably not CTA-throughput-bound.

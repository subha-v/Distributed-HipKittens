# BANKED_LAWS — every empirical law this project has measured, with evidence and scope

**Purpose.** This is the **prior for the cost model M(W, S; θ)** required by
`../BRIEF.md` §"Required outputs" item 3. Every law below is something the project
*measured*, on a named rig, in a named regime. Each carries the scope it was measured
in and the way it is expected to break elsewhere, because the whole point of the
ablation campaign is to say "if you change X in the workload, the schedule should
change by Y, because mediating quantity Q crossed threshold θ" — and a law whose scope
is unstated cannot be used that way.

**Reading rules (binding).**

1. **No number appears here without a source.** Sources are repo files (`path:line`),
   git commit messages (sha), or the BRIEF's draft8 digest (authoritative for
   slide-level results).
2. **Confidence tiers.**
   - `PROVEN-replicated` — measured ≥2 times on independent campaigns/rigs, or measured
     once *and* confirmed by an independent mechanism contrast (e.g. knob-ON/knob-OFF at
     the new pin, not the arm against its own history).
   - `MEASURED-once` — one campaign/analysis, gates green, no replicate.
   - `HYPOTHESIS` — a mechanism-level expectation with partial evidence or an
     unmeasured magnitude. Never quote as a result.
   - `ANALYTIC` — arithmetic/model from measured inputs, no direct measurement of the
     stated quantity.
3. **Scope is load-bearing.** Unless a law says otherwise, "the balanced prefill rig"
   means: 8×MI350X/MI355X gfx950, one node, one fused-MoE layer, `T=4096` tokens/rank,
   top-k 8, hidden 7168, 32 experts/rank, **synthetic balanced routing (std = 0)**,
   eager execution, no shared expert, MoK synthetic prefill harness, 5-rotation median
   rank-max p50. That is *one point* in W-space. Roughly 60% of the laws below were
   measured only there.
4. **Where two laws appear to contradict, §6 names the contradiction and its
   resolution.** Do not resolve one silently in the cost model.

**The mediating quantities θ that the laws keep pointing at** (candidate state
variables for M):

| symbol | quantity | laws that move on it |
|---|---|---|
| `d` | outstanding remote ops per producer thread (injection depth) | LAW-20, 21, 22, 8 |
| `co` | compute co-residency of the carrier (is a GEMM sharing the device?) | LAW-8, 20, 22, 25 |
| `A` | protocol atomic/flag operation **count** per epoch (not their placement) | LAW-16, 17, 18, 19 |
| `L` | cache-lines touched per protocol event (scatter width) | LAW-18 |
| `φ` | fraction of the payload-carrying phase that is CTA-throughput-bound | LAW-13, 14, 15 |
| `u(t)` | consumer-unblockable fraction under the producer's task order | LAW-23, 24 |
| `f` | MoE-region fraction of a serving step | LAW-40, 41, 44 |
| `W` | padded MoE rows per real token | LAW-40, 41, 42 |
| `fill` | real rows / bucket rows per rank-step | LAW-38, 39, 42 |
| `σ_rank` | max-rank load ratio (routing + padding skew) | LAW-43, 46, 47, 51 |
| `B` | in-launch bubble supply (CTA-ms) vs filler demand | LAW-53 |
| `crit` | identity of the critical (hottest / slowest) rank | LAW-52 |

---

## 1. Transport laws (fabric, engine, direction, layout)

### LAW-1 — Remote packed-bf16 atomic accumulation is exact across devices on gfx950, including on MORI's HIP-VMM `Uncached` symmetric heap.
- **Evidence.** Direct hardware test: one remote writer, 128 sequential RMWs, **256/256
  cells exact**; two devices contending on the same cells, **256/256 exact**;
  disassembly shows `global_atomic_pk_add_bf16` with **no CAS fallback**
  (`aug10/KERNEL_BOTTLENECKS.md:198-206`). Heap question closed independently: exp_21's
  dual-write detector (`g=49`) over **600 epochs recorded ZERO lost-update flags** on
  the mori heap (`aug10/experiments/LESSONS.md:919-922`).
- **Scope.** gfx950/CDNA4, intra-node xGMI, coarse-grained memory *and* the mori
  `HeapType::Uncached` VMM heap, ≤8 contending devices.
- **Workload dependence.** Architecture is a first-class axis: on **gfx942/MI300X remote
  atomics are ~3× slower than remote stores** (BRIEF digest, "gfx942/MI300X"), which
  makes producer-carried atomic accumulation the wrong carrier there and forces
  per-source non-overlapping slots + owner-local reduce. Expect the law to hold for
  correctness on gfx942 and fail for *performance*.
- **Confidence.** PROVEN-replicated (two independent gates: ubench + 600-epoch in-kernel
  detector).

### LAW-2 — On a single xGMI link the fabric is **byte-limited, not op-limited**: coalesced 4 B remote atomics reach the same bandwidth as 16 B remote stores.
- **Evidence.** Fabric ubench: **coalesced 4 B remote atomics = 52.8 GB/s == 16 B stores
  54.9 GB/s per link, at any writer count 64–256; per-task drain free; target-busy
  immune** (`aug10/experiments/LESSONS.md:934-937`). BRIEF digest restates it as
  "packed bf16 remote atomics: 52.8 GB/s coalesced (near 54.9 GB/s remote stores)".
- **Scope.** Per-link, half-wave coalesced address pattern, gfx950.
- **Workload dependence.** Does **not** survive to aggregate-collective scale: at
  235 MB all-reduce, store-towers reach **93.6 GB/s effective while atomics cap ~67**
  (commit `108def6e`). So "atomics ≈ stores" is a *small-message per-link* law and
  "stores > atomics" is the *bulk-collective* law. The crossover is unmeasured.
- **Confidence.** PROVEN-replicated (ubench + the independent G25-0b transport matrix).

### LAW-3 — Address layout, not op count, decides remote-write bandwidth: scattering lane addresses collapses a coalesced transfer by ~13×.
- **Evidence.** Fabric ubench: "**scattered atomics collapse 13x**"
  (`aug10/experiments/LESSONS.md:936`). BRIEF digest quantifies the endpoint:
  **52.8 GB/s coalesced → 4.1 GB/s scattered**, and draws the conclusion
  "**layout is a schedule decision**".
- **Scope.** gfx950 xGMI peer writes from CU vector stores/atomics.
- **Workload dependence.** Binds whenever routing/permutation controls the destination
  address stream — i.e. it gets *worse* with skew and with any dispatch order that
  interleaves destinations at lane granularity. It is orthogonal to message size.
- **Confidence.** PROVEN-replicated.

### LAW-4 — Achievable single-link bandwidth is 56.9–57.3 GB/s (74.1–74.6% of the 76.8 GB/s nominal), and it is reached by a handful of CTAs.
- **Evidence.** exp_22: the achievable ceiling is **56.9–57.3 GB/s at every C from 8 to
  64**; the substantive sweep is **C=8 55.2 → C=64 56.5 GB/s, i.e. 8× the pushers buys
  +2.4%** (`aug11/LESSONS.md:207-217`).
- **Scope.** exp_22's one-process 8-GPU diagnostic topology (explicitly *not* the
  rank-per-GPU symmetric-heap model — `aug12/LESSONS.md:13-16`), 8×MI350X.
- **Workload dependence.** Since transport saturates at ~8 pushers, **pusher count is
  not a bandwidth knob**; any C-sweep above ~8 is buying (or losing) something other
  than bandwidth. Under multi-destination fanout or link heterogeneity the ceiling is
  per-link and the *hottest link* sets the schedule (see LAW-51).
- **Confidence.** MEASURED-once (one rig, many points).

### LAW-5 — Push and pull are equivalent at matched transport; direction is chosen by who defines progress, not by bandwidth.
- **Evidence.** Two independent rigs. (a) Gated one-process 64 KiB anchor: CU push
  **15,723.779 µs**, CU pull **15,623.289 µs**, ratio **0.993609×, inside the frozen ±2%
  equivalence interval** (`aug12/LESSONS.md:36-44`). (b) BRIEF digest: matched 64 KiB
  producer→transport→dependent-consumer wall time **0.9936×** (tie); 64 MiB single-link
  BW pull/push **1.078×**.
- **Scope.** Intra-node xGMI, CU-issued transport, 64 KiB records (tie) and 64 MiB bulk
  (slight pull edge). Both directions passed the same gate ladder incl. 600-epoch soak.
- **Workload dependence.** The *selection rule* — "push when progress is
  producer-defined & many-to-one (the ERS half); pull when consumer-defined &
  one-to-many (the MAG half)" (BRIEF digest) — is a design rule, **not** a measured
  crossover. Expect pull to win where the consumer has precise demand and push would
  fan out uselessly; expect push to win where acknowledgement latency would stall a
  producer epilogue. Neither corner has been measured.
- **Confidence.** PROVEN-replicated for the *tie*; HYPOTHESIS for the selection rule.

### LAW-6 — The host peer-copy API is CU-lowered, not SDMA, and costs 1.45× CU pull at 64 KiB.
- **Evidence.** `host_copy_path` **22,718.219 µs vs CU pull 15,623.289 µs = 1.454125×**
  end-to-end; rocprofv3 found **8,192 peer-copy API calls, zero memory-copy-domain
  records, direct same-correlation `__amd_rocclr_copyBuffer` dispatches, and zero
  `hsa_amd_memory_async_copy_on_engine` calls** (`aug12/LESSONS.md:36-56`). Mechanism:
  one runtime copy-kernel enqueue per 64 KiB record.
- **Scope.** `hipMemcpyPeerAsync`, 64 KiB records, this ROCm stack.
- **Workload dependence.** **Do not call this an SDMA result — no SDMA executor
  participated** (`aug12/LESSONS.md:54-56`). "SDMA" is an executor claim, not an API
  label (`aug12/LESSONS.md:9-11`). A real shader-initiated SDMA arm (mori CCO
  `ccoSdma`) has never been run in situ; it remains the only untested way to move the
  bytes off the CU memory path.
- **Confidence.** PROVEN-replicated (timing + executor trace agree).

### LAW-7 — Making the transport copy *faster* imports fabric cost: the pusher's slowness is load-bearing.
- **Evidence.** exp_20, driven flat out **in isolation** the traffic classes rank
  read **+38 µs** / local write **+105 µs** / **peer write +624 µs** — a **519 µs fabric
  surcharge at 5.96×** — while the real pusher, throttled ~5.9× by its MLP=1
  `load → wait → store` shape, never reaches that regime
  (`aug10/experiments/LESSONS.md:885-892`). This retro-explains exp_04's MLP fan-out
  null and is recorded as "a standing warning against MLP, larger transfer units, and
  SDMA on this path."
- **Scope.** Staged-push service pool co-resident with the second expert GEMM, balanced
  prefill rig.
- **Workload dependence.** Binds only under co-residency (`co = 1`). In a pure-transport
  phase, faster is faster. Corollary for the knob space: *transfer-unit size* and *MLP
  depth* are not free knobs when a GEMM shares the device — they are injection-rate
  knobs in disguise (see LAW-20).
- **Confidence.** MEASURED-once (isolation vs in-situ contrast, single experiment).

### LAW-8 — Injection depth is **flat in pure transport**; the depth-4 law is co-residency-specific.
- **Evidence.** G25-0b swept depth as an axis on the 8-GPU single-process CDAR
  microbench: "**depth flat in pure transport (K4 co-residency-dependent)**" (commit
  `108def6e`; rig described in `495b5fc6`). Contrast with LAW-20/21, measured with a
  GEMM co-resident.
- **Scope.** Transport-only microbench, both transports, 235 MB, gfx950, all-PASS
  correctness.
- **Workload dependence.** This is the *scoping counterpart* of the project's most
  load-bearing schedule law. Any cell of W-space where the carrier does **not** share the
  device with compute should predict **no** depth effect; any cell where it does should
  re-fit `d`.
- **Confidence.** MEASURED-once. **This one should be re-run before the cost model
  commits to a `d(co)` term.**

### LAW-9 — Store-towers are the best measured bulk transport; atomics cap well below them.
- **Evidence.** G25-0b K1 = store-towers, **93.6 GB/s effective AR @235 MB, 4.2× v0**
  after a parallel vectorized reduce with dynamic reduce-arrival certification;
  **atomics cap 67**; v0 matrix was 41.1/22.4 GB/s at 235 MB, "5-6x under ceiling,
  diagnosed" (commits `108def6e`, `495b5fc6`).
- **Scope.** 8-GPU single-process boundary harness, 235 MB all-reduce, exact-bf16
  verification, bounded waits fail-loud.
- **Workload dependence.** The tower shape trades transport bandwidth for a reduce pass;
  at small payloads the reduce is not amortized and LAW-2 (atomics ≈ stores) should
  reassert. Crossover unmeasured.
- **Confidence.** MEASURED-once.

### LAW-10 — At the TP8+EP boundary, fused CDAR does not yet beat GEMM→RCCL; RCCL's all-reduce is 1.5× our transport.
- **Evidence.** G25-1 honest interim: **2.9 ms (GEMM→RCCL) vs 3.6 ms (fused CDAR)**;
  "RCCL AR is 1.5× our transport"; **four overlap hypotheses tried and falsified in-rig
  — drain pipelining, register-source stores, consuming-pool specialization, item
  granularity** (commit `3f7f0f3a`). Amdahl anchor: **native TP8 closes at ~2.8 ms/layer
  exposed AR — that is the prize pool** (commit `108def6e`).
- **Scope.** Boundary-level TP8+EP rig, gfx950, per-layer.
- **Workload dependence.** This is a *parallelism-topology* law: TP8's collective is
  dense, regular and large, which is exactly the regime the vendor library is tuned for.
  Our advantage in EP8 prefill comes from irregular, dependency-carrying traffic that a
  library cannot start early. Expect the gap to close as the collective gets *more*
  data-dependent, not as it gets bigger.
- **Confidence.** MEASURED-once (interim, explicitly labelled honest/not-final).

### LAW-11 — This node exposes no live xGMI throughput counter.
- **Evidence.** `amd-smi --xgmi` returns **N/A** and `--shownodesbw` returns **0-0**;
  fabric cross-checks must use **UMC duty cycle (±15 pp) plus a computed ceiling** — the
  52.8 GB/s coalesced-4 B-atomic figure — or rocprof TCC-EA, which is named as the
  remaining gap (`aug11/LESSONS.md:226-232`).
- **Scope.** The 8×MI350X node as configured.
- **Workload dependence.** None — it is an instrument constraint, and it bounds what any
  calibration plan can promise for θ's bandwidth terms.
- **Confidence.** PROVEN-replicated (two tools, both negative).

---

## 2. Schedule / overlap laws (carrier, roles, depth, granularity, order)

### LAW-12 — Occupancy is exactly one block per CU at megakernel register/LDS budgets, so a communication CTA is **never** co-resident with an MFMA CTA.
- **Evidence.** 256 ArchVGPR + 256 AGPR gives one wave per SIMD, and **155,428 B of LDS
  (of 163,840) independently forces one block per CU** (`aug10/KERNEL_BOTTLENECKS.md:24-27`,
  compiler resource report). Consequences drawn in `:105-108`: no SIMD to share,
  `s_setprio` is irrelevant to this split.
- **Scope.** Any kernel at this LDS/register budget on CDNA3/CDNA4 — i.e. all our
  megakernels.
- **Workload dependence.** Breaks the moment LDS or register pressure drops:
  **shrinking the ubench's LDS moved pointer arrays into scratch and let occupancy rise
  1 → 2 blocks/CU, silently voiding the experiment's premise**
  (`aug11/LESSONS.md:99-109`). Therefore: **occupancy must be asserted and printed, not
  assumed, in every arm of the knob space.**
- **Confidence.** PROVEN-replicated.

### LAW-13 — CTA role specialization removes issue-slot contention and leaves memory-system contention untouched; the interference costs **2.8× the capacity tax**.
- **Evidence.** Matched-CTA-count triple at C=64: M7 = **1,609.7 µs** (254 CTAs baseline)
  → **2,019.7 µs** (192 CTAs, service pool reserved but idle: pure capacity, +25.5%) →
  **3,179.0 µs** (same 192 CTAs, service pool actively moving payload: +57.4% more)
  (`aug10/experiments/LESSONS.md:301-316`). Restated in the whole-kernel decomposition as
  **+1,249 µs = ~434 capacity + ~815 interference** (`aug10/KERNEL_BOTTLENECKS.md:95-99`).
- **Scope.** Balanced prefill rig, mode-2 staged push, C=64, gfx950.
- **Workload dependence.** The interference term is a property of the *memory system*,
  so it scales with the co-resident phase's memory intensity, not its FLOPs. A boundary
  whose compute phase is cache-resident or arithmetic-bound should show far less. Also:
  **mode 0 (reserve-but-idle) is retired as the control** for role specialization —
  it measures only the capacity tax and is structurally blind to the dominant cost.
- **Confidence.** PROVEN-replicated (exp_05 stage 0 curve + exp_14/exp_20 reproductions).

### LAW-14 — Marginal interference is **worst for the first service CTAs** — you cannot buy a little overlap cheaply.
- **Evidence.** Matched mode-0/mode-2 pairs (M7, µs): C=16 1,659.3 → 2,277.6
  (**+618.3, +37.3%**); C=32 1,784.1 → 2,479.1 (+695.0); C=48 1,819.5 → 2,788.8
  (+969.3); C=64 2,019.7 → 3,179.0 (+1,159.3). Capacity-only across C=2..64 grows just
  410 µs total, so **at C=16 the interference (618 µs) is 12× the capacity cost (50 µs)**,
  and marginal interference falls **38.6 µs/CTA at C=16 → 18.1 at C=64** — "the signature
  of a shared resource being DISTURBED rather than divided"
  (`aug10/experiments/LESSONS.md:345-356`).
- **Scope.** As LAW-13.
- **Workload dependence.** This is why a *small* comm pool is not a safe default; the
  cost model must not treat interference as linear in C. Under a workload where the
  co-resident phase is idle (LAW-15's `φ ≈ 0` phases) the curve should flatten.
- **Confidence.** MEASURED-once (single swept curve, but four points, monotone).

### LAW-15 — The entire CTA-capacity tax falls on the payload-carrying GEMM; the *other* GEMM has idle CTA capacity.
- **Evidence.** Removing 62 of 256 CTAs moves **plan + M6 by +0.5% (2,954 → 2,970 µs,
  flat across C = 2/16/32/48/64)** and moves **M7 by +27% (1,586 → 2,020 µs, linear in
  C)** (`aug10/KERNEL_BOTTLENECKS.md:61-67`; `aug10/experiments/LESSONS.md:653-667`).
  Phase profile of the homogeneous kernel: dispatch 1,193 (17%) / plan 415 (6%) /
  **M6 2,539 (36%)** / M7 1,586 (23%) / combine 1,255 (18%)
  (`aug10/KERNEL_BOTTLENECKS.md:49-56`).
- **Scope.** Balanced prefill rig; the reservation point was M6.9.
- **Workload dependence.** `φ` (the CTA-throughput-bound fraction of a phase) is a
  *shape* property: M6's flatness comes from its weight-streaming/latency-bound
  character, and M7's linearity from being CTA-throughput-bound. At small T the GEMMs
  become launch/latency dominated and both should flatten; at very large T both should
  become throughput-bound. **The old design reserved exactly at the boundary where the
  free phase ends and the starved phase begins** — the single most repeatable placement
  mistake in the ledger.
- **Confidence.** PROVEN-replicated (reserve-but-idle sweep + phase profile).

### LAW-16 — The interference is the **readiness protocol's** traffic, not the payload bytes.
- **Evidence.** exp_20 mode 7 deletes the pool's entire 896 B payload copy (kept correct
  by `pull_fallback`) and moves M7 by **+5.7 µs against a ±40 µs band while 831.5 µs of
  interference remains**; the copy costs 480 µs **of combine only**
  (`aug10/experiments/LESSONS.md:867-873`). Independently, interference rises
  monotonically with the group size `g` while payload bytes are constant at ~312 MB:
  **+1,159.3 (g=1) / +1,645.6 (g=2) / +2,058.8 (g=4) / +2,693.8 (g=16) µs**
  (`aug10/experiments/LESSONS.md:370-379`). This **retracts exp_17's** "the only
  remaining lever is to move fewer bytes".
- **Scope.** Balanced prefill rig, mode-2 pool with the per-(row, N-chunk) arrival
  protocol.
- **Workload dependence.** Protocol traffic scales with *events*, i.e. with
  rows × chunks × contributing blocks (~535,040 arrivals/rank/epoch,
  `aug10/KERNEL_BOTTLENECKS.md:175-179`), which is set by the plan, not by the schedule.
  Any workload that raises event density (smaller tiles, finer signals, higher top-k)
  raises this term super-linearly relative to bytes.
- **Confidence.** PROVEN-replicated (payload-deletion arm + the g-axis, two mechanisms).
- **Residual, stated honestly.** At g=1 the probe loop is degenerate yet **+1,159 µs of
  interference remains** — a g-independent floor whose mechanism is *not* identified.
  36% of it is attributable to the per-XCD L2 (LAW-19); "the mechanism behind the other
  64% is unidentified" and is named **the single largest gap in our understanding of
  this kernel** (`aug10/KERNEL_BOTTLENECKS.md:129-131, 213-223`).

### LAW-17 — Only reducing the **number** of protocol atomics helps; relocating them cannot.
- **Evidence.** exp_14 removed ~698,112 of ~1.58 M atomics/rank/epoch for **−306 µs of
  M7** (`aug10/experiments/LESSONS.md:677-687`), a rate of ~0.44 µs per thousand atomics.
  Atomic **scope** (acq_rel → relaxed, a non-correctness-preserving diagnostic) is worth
  ~30%: M7 3,179.0 → 2,825.4 (−353.6) at C=64/g=1
  (`aug10/experiments/LESSONS.md:393-400`). Counter *relocation* is the falsifier — see
  LAW-18.
- **Scope.** Balanced prefill rig.
- **Workload dependence.** Extrapolating the exp_14 rate says removing *every* remaining
  arrival atomic is worth **≤235 µs against +1,249 µs outstanding — ≤19%**
  (`aug10/KERNEL_BOTTLENECKS.md:117-118`; `aug10/experiments/LESSONS.md:722-733`). So the
  atomic-count lever has a hard ceiling in this regime and would matter more only where
  event density is much higher.
- **Confidence.** PROVEN-replicated for the direction; the ≤19% ceiling is ANALYTIC
  (labelled INFERRED at source).

### LAW-18 — **Scattering** atomics is a feature; coalescing them onto few cache lines is ≥10× slower.
- **Evidence.** exp_07 transposed arrival counters so 32 live lanes hit **2 cache lines
  instead of 32** — semantically identical, same buffer, no ABI change — and the config
  made no measurable progress in **12 minutes against a usual 25–95 s**, abandoned rather
  than completed, so **≥10× is a lower bound**
  (`aug10/experiments/LESSONS.md:445-459`; `aug10/KERNEL_BOTTLENECKS.md:153-158`).
  Mechanism: an atomic **resolves at the line**, so RMWs to 2 lines serialize while 32
  RMWs over 32 lines proceed in parallel across L2 banks. Second data point at a
  different site: mode-13 counter consolidation — deleting the `pushed` counter saved
  ~85–150 µs, the 16→1 footprint consolidation cost ~150–240 µs
  (`aug10/experiments/LESSONS.md:942-947`).
- **Scope.** gfx950 L2/atomic path; any device-scope RMW fan-in.
- **Workload dependence.** Inverts the normal coalescing intuition, which is right for
  loads and wrong for atomics. The law binds wherever lanes in a wave contend; under
  a workload where each lane's counter is already private it is vacuous.
- **Confidence.** PROVEN-replicated (two independent sites, both negative).

### LAW-19 — Placing the comm pool on dedicated XCDs cuts interference 36% and starves the pool 54% — a net loss.
- **Evidence.** Confining the pool to whole XCDs (so compute dies' L2s stay clean) cut
  interference **36% (M7 3,125 → 2,723 at matched CTA counts)** but the pool, squeezed
  onto two dies, **slowed 54%** (`aug10/KERNEL_BOTTLENECKS.md:125-127`). exp_08 mode 3
  reproduced the loss with fall-through active (C=64 7,768 / C=96 7,637 / C=128 7,827,
  `aug10/experiments/LESSONS.md:688-693`).
- **Scope.** 8 XCDs × 32 CU, 4 MB L2/XCD, 256 MB Infinity Cache.
- **Workload dependence.** **Workgroups go round-robin to XCDs with chunk size one**
  (`xcd = wg_id % 8`), so contiguous `bid < C` **spreads** the pool one CTA per XCD and
  strided `bid % 8 == 0` **concentrates** it on XCD 0 (`aug10/CLAUDE.md`, A8 correction;
  round-robin `DOCUMENTED`, chunk=1 `REPORTED`). Any placement law stated without the
  dispatch rule is backwards.
- **Confidence.** PROVEN-replicated (localization measurement + a full placement sweep).

### LAW-20 — A bound on **outstanding remote writes per producer thread** is the decisive overlap mechanism, and it is a **cliff, not a curve**: depth 4 wins, one step past 8 loses everything.
- **Evidence.** exp_24 depth sweep (M7, µs): **4 → 2,659 | 8 → 2,742 | 16 → 3,204 |
  32 → 3,112 | unthrottled 3,189**; depth 4 beats depth 8 by **−50.9 µs (−0.77%)** at
  campaign resolution (n=3 vs n=3, ~9σ); "one step deeper than 8 loses the ENTIRE ~500 µs
  the throttle was worth (16 and 32 sit at the unthrottled cost, 7.3σ and 5.9σ)"
  (`aug10/experiments/LESSONS.md:981-990`). Earlier at depth 8: **~500 µs of M7
  recovered** (`aug10/experiments/LESSONS.md:926-933`). Waterfall: unthrottled
  epilogue-carried **7,110.8 µs** vs +depth-4 **6,495.8 µs = −615.0 µs, t = −80.2**, with
  per-campaign ranges **disjoint by 598 µs** (`aug11/LESSONS.md:41-54`). Restoration
  contrast after the exp_38 fix: **−618.7 µs** vs −613.5 at the healthy pin and −1.8 at
  the broken pin (`aug11/LESSONS.md:548-555`).
- **Scope.** Producer-epilogue-carried remote packed-bf16 accumulation, co-resident with
  the second expert GEMM, balanced prefill rig, 7 peers.
- **Workload dependence.** **Does not bind in pure transport (LAW-8).** The depth is
  congestion control on a shared memory path, so the optimum should track the co-resident
  phase's own memory pressure and the peer count, not the payload size. Depth 2 was never
  measured (the `g` `0x300` selector has no room left) — the left side of the cliff is
  unexplored.
- **Confidence.** PROVEN-replicated (sweep + waterfall + post-fix contrast, three
  independent campaigns).

### LAW-21 — Carrier relocation and injection bound are **not separable and never additive**: moving the payload into the producer epilogue is a **+210.6 µs regression** without the bound.
- **Evidence.** exp_35, service pool held fixed at C=16, one `g` bit flipped: rung (b)
  epilogue-carried, throttle disabled (`g=65`) = **7,110.8 sd 13.7 µs = 0.9226×**, i.e.
  **+211.2 µs paired slower than the homogeneous baseline that carries no payload at
  all** (`pf6gm_mega` 6,900.2 sd 15.5 = 0.8950×); rung (c) with the depth-4 bound
  (`g=353`) = **6,495.8 sd 6.9 = 0.8422×**. "The throttle alone is **1.52× the entire
  (a)→(c) gap of −404.4 µs**; the payload relocation on its own accounts for **−52.1%**
  of that gap." Campaigns ran alternating `c, b, c, b` in two batches so drift cannot be
  confounded (`aug11/LESSONS.md:41-54`).
- **Scope.** As LAW-20.
- **Workload dependence.** The cost model must treat **(carrier, depth)** as one joint
  cell of S, never as two additive contributions. Any figure drawn as a waterfall of
  separable rungs is wrong for this pair.
- **Confidence.** PROVEN-replicated (paired, alternated, disjoint ranges).

### LAW-22 — Pacing a *dedicated pool* is dead; throttling a *producer that also computes* is decisive. The discriminator is whether the paced engine is the only thing in flight.
- **Evidence.** exp_20 E1: pacing the pusher recovers **41% of M7 (ΔM7 767 → 450 µs)**
  but **the combine pays 7:1 (+2,219 vs −316)** and `M2→end` rises monotonically
  **6,229 → 8,134 µs** — **no interior minimum, so pacing 0 is already optimal**, and
  "every Tier-3 scheduling idea premised on throttling the pusher is closed"
  (`aug10/experiments/LESSONS.md:874-881`). Poll backoff (64× spin-rate cut) is a flat
  null. Against that, LAW-20's producer-side bound pays ~500–615 µs; exp_21 states the
  reconciliation explicitly: "**pacing pays exactly when the paced engine is not the
  only phase in flight**" (`aug10/experiments/LESSONS.md:926-933`).
- **Scope.** Balanced prefill rig.
- **Workload dependence.** Restated as a decision rule for M: throttle the carrier iff
  the carrier's own drain is **not** on the critical path. In the dedicated-pool schedule
  the drain *is* the critical path (LAW-25), so throttling extends the makespan directly.
- **Confidence.** PROVEN-replicated (both halves measured, mechanism stated).

### LAW-23 — Readiness granularity pays only when it changes **when dependent work can execute** — signal the smallest early-**executable** unit, not the smallest stored or transferred unit.
- **Evidence.** Our side: under the natural GEMM-2 task order,
  **P(token ready by t) = (t/S)^8** and **the median token is not reducible until 91.7%
  of GEMM-2 completes** (exp_29 analysis,
  `aug11/OVERLAP_METHODOLOGY_STUDY.md:277`; BRIEF digest restates 91.7%). Vendor side,
  same conclusion from the opposite direction: replacing one global combine barrier with
  per-record flags gives **no benefit**, and an LL128-style per-record flag path is
  **44% slower at nbs=1 and 177% slower at nbs=64** because it adds a deflag copy
  without removing the downstream whole-buffer dependency
  (`aug11/OVERLAP_METHODOLOGY_STUDY.md:338-351`). Our own coarse-readiness arm removed
  **−573.3 µs** of protocol work (`aug11/LESSONS.md:311-319`).
- **Scope.** MoE combine fan-in with top-k 8; vendor numbers are MI300X MORI
  dispatch/combine.
- **Workload dependence.** `u(t)` is set by top-k (the exponent 8) and by producer order
  (LAW-24). At top-k 1–2 the same fine signals would have a real shadow; at top-k 8 they
  do not. **This is the clearest example in the ledger of a schedule knob whose sign
  flips on a workload parameter.**
- **Confidence.** PROVEN-replicated as a *principle* (independent AMD reproduction);
  the (t/S)^8 model itself is ANALYTIC — "model only; readiness instrumentation and
  reorder were never run".

### LAW-24 — Producer task order, not signal granularity, sets the consumer's unblockable fraction.
- **Evidence.** BRIEF digest: "producer order restructured to complete an independent
  slab early", following directly from the 91.7% readiness result of LAW-23. The
  companion projection — the nc-major reorder lifts the combine's unblockable fraction
  **17% → 85%** — is recorded in `aug11/CLAUDE.md` (Phase-2 queue item 4) as a design
  target. Independent precedent: IRIS's two-shot GEMM-AllReduce shows coarse readiness
  can be **the natural consequence of producer order and output ownership**, not a weaker
  approximation to per-tile signaling
  (`aug11/OVERLAP_METHODOLOGY_STUDY.md:457-462`).
- **Scope.** Grouped-GEMM producers with a many-to-one reduce consumer.
- **Workload dependence.** Order is free to change only where the producer's tiles are
  independent; under skew the *hot* rank's order is the only one that matters (LAW-52).
- **Confidence.** HYPOTHESIS for the 17% → 85% magnitude (never measured on GPU);
  MEASURED-once for the underlying readiness distribution.

### LAW-25 — In a staged-push schedule the **service pool's drain is the post-GEMM critical path**, and readiness is not the lag — drain rate is.
- **Evidence.** At C=64 the drain runs **5,952 µs past M6 while M7+combine occupy
  6,054 µs**, and **the first tile event lands 192 µs BEFORE the last CTA leaves M6**
  (`aug10/experiments/LESSONS.md:330-337`). Mode 1 is the clean counterpoint: the bulk
  push runs after M7, so **M7 is untouched (1,579.8 µs)** and the whole cost lands in
  combine (1,287.9 → 2,782.1). "Vertical and horizontal fusion move the same bytes and
  pay in different phases."
- **Scope.** Balanced prefill rig, mode-1/mode-2 carriers.
- **Workload dependence.** Which phase pays is a *schedule* choice, and the right one
  depends on which phase has slack — i.e. on the phase profile of LAW-15, which is itself
  workload-dependent.
- **Confidence.** MEASURED-once (device-stamp attribution across five configurations).

### LAW-26 — Dynamic ticket claiming + compute-CTA fall-through is worth up to **2.9×** on a role-split kernel: static stripes cause head-of-line blocking.
- **Evidence.** exp_12, two changes (a monotonic `fetch_add` event ticket replacing a
  static `(service_id, service_count)` stripe; the drain guard dropping `is_service_cta`
  so every CTA joins the drain as it finishes): **C=16 22,068 → 7,617 (2.90×), C=32
  14,775 → 7,504 (1.97×), C=64 9,976 → 7,432 (1.34×)**, resource tuple unchanged
  (`aug10/experiments/LESSONS.md:629-652`). Mechanism, two parts: fall-through refunds
  the only tax the reservation costs (LAW-15), and the ticket makes the queue fill in
  **completion order** where the stripe forced each wave to wait for *its* events in
  index order. Cost: one relaxed atomic per event (~16,720/rank/epoch against the 535,040
  the arrival counters already spend).
- **Scope.** Persistent role-split megakernel with a producer-enqueued event queue.
- **Workload dependence.** The gain is proportional to producer-completion variance —
  i.e. it grows with skew and with ragged expert sizes, and vanishes at perfect balance
  with uniform tile costs.
- **Confidence.** MEASURED-once (screened, three C points, all gate-green) —
  but the mechanism is reused in every later kernel and never regressed.

### LAW-27 — Reserving CTAs costs **~8–9 µs each even when the pool provably has no work**.
- **Evidence.** Mode 14 at C = 0/8/16 = **6,650.9 / 6,725.6 / 6,780.4 µs**, monotone,
  with the pool instrumented idle three independent ways (error bit 26 never set across
  4,032 `pperr` readings; `[MPS TS] DRAIN=0`; `[MPS SPIN] chunk_poll 0/0`)
  (`aug11/LESSONS.md:445-452`). Complementary: an LDS-only *spinning* pool is **+23 µs**,
  indistinguishable from an idle one — "occupying a CU costs nothing"
  (`aug10/experiments/LESSONS.md:882-884`).
- **Scope.** 256-CTA persistent megakernel, balanced prefill rig.
- **Workload dependence.** This separates **capacity loss** from **contention** for the
  first time: the placement penalty survives deleting everything the pool did. In a
  workload where the co-resident phase is not CTA-throughput-bound (`φ ≈ 0`, LAW-15),
  this term should go to zero.
- **Confidence.** MEASURED-once, with three independent idleness instruments.

### LAW-28 — In the balanced prefill regime with a producer-carried payload, **dedicating CTAs to communication never wins at any pool size**.
- **Evidence.** exp_37, 27 campaigns: **C=4 6,464.2 ≈ C=8 6,472.0 < C=12 6,490.4 <
  C=16 6,498.0 < C=32 6,735.3 < C=64 (mode 2) 6,837.5** — monotone over a 374 µs span,
  with paired **C=8 −27.2 [−32.3, −22.1]** and **C=4 −34.2 [−39.5, −28.9] in 7 of 7
  rounds**; the mode-14 C-sweep is monotone at **8.1–9.3 µs per reserved CTA** with the
  pool provably idle (`aug11/LESSONS.md:287-301`). Reproduced independently by exp_36
  (`aug11/LESSONS.md:180-192`). At the boundary level the same verdict appears as a
  measured traffic cost: **dedicated communication CTAs (C=64) = 6,866 µs (0.888×) with
  +831 µs of added traffic from the comm-CTA role itself** (BRIEF digest).
- **Scope.** `T=4096`, balanced routing, EP8 fused-MoE layer, gfx950, mode-12
  producer-carried carrier.
- **Workload dependence — the scope limit that matters most.** This law is **carrier-
  conditional**. Within the *staged-push* carrier (mode 2), the opposite held: C=64 was
  the optimum and the specialized kernel **beat** the homogeneous one, 0.888× vs 0.894×
  (`aug10/experiments/LESSONS.md:669-676`, `aug10/EXPERT_BRIEFING.md:14-16`), turning over
  only at C=96/128 (`aug10/experiments/LESSONS.md:688-693`). Once the payload moved into
  the producer epilogue the pool had almost no job left, so every reserved CTA became pure
  capacity loss. Additionally, the one regime that could rescue a pool — **routing skew —
  was never testable on that harness** (LAW-62), so "dedication is bad" is **not**
  established for skewed or straggler-dominated workloads, and the pre-registered
  expectation that "communication requires resident progress" remains open
  (`aug11/OVERLAP_METHODOLOGY_STUDY.md:1135`).
- **Confidence.** PROVEN-replicated **within its carrier and regime**; NOT a
  technique-level verdict.

### LAW-29 — The pool protocol's interference is **C-invariant in total volume**; pool size only moves the capacity side of the trade.
- **Evidence.** Mode-13 interference **1,328 µs @ C=16 vs 1,323 µs @ C=64** — the
  ~590–860 µs event/atomic/flag traffic is a fixed per-epoch tax
  (`aug10/experiments/LESSONS.md:938-941`).
- **Scope.** Balanced prefill rig.
- **Workload dependence.** Follows from LAW-16: protocol volume is set by the plan
  (events), not by the number of CTAs draining it.
- **Confidence.** MEASURED-once.

### LAW-30 — The payload-carrying GEMM and the combine are **one coupled block**, anti-correlated at r = −0.904; the correct objective is their sum.
- **Evidence.** exp_33, n=10: **M7 2,701.8 ±23.74 µs and combine 324.2 ±21.12 µs sum to
  3,026.1 ±10.2 µs with sd 32.1, against the 100.5 µs sd independence would predict**
  from the individual sds (75.1 and 66.8). "Combine's 324 µs is **M7's slack, not
  work**." Reproduced in exp_35: from rung (b) to rung (c), M7 moved −467.8 and combine
  −116.9 in the same direction, **together 92% of the −615.0 µs end-to-end move**
  (`aug11/LESSONS.md:28-40`).
- **Scope.** Balanced prefill rig at the mode-12 ratchet.
- **Workload dependence.** Two binding consequences: **attacking the combine alone will
  not pay** (the time moves, it does not disappear), and **any claim to have moved
  combine by <~60 µs measured alone is inside the noise** of a quantity whose stderr is
  6.5% of its own mean.
- **Confidence.** PROVEN-replicated (correlation + an independent waterfall rung).

### LAW-31 — The dispatch all-to-all has **zero exposed wait** under balanced routing.
- **Evidence.** Per-`(source, chunk)` poll reports **max-spins-to-success = 0 against a
  2,000,000 spin limit** (`aug10/KERNEL_BOTTLENECKS.md:159-163`); reproduced with
  `fail_max = 0` in all 10 exp_33 rotations and both exp_35 rungs, `success_max ≤ 1`,
  over 500 warmup + 100 timed + all 600 soak epochs, counters never reset
  (`aug11/LESSONS.md:323-329`). Mechanism: all eight ranks run identical work on
  identically-shaped data and enter the exchange within microseconds, so the transfer is
  already latency-hidden by symmetry.
- **Scope.** **Routing std = 0** — this is a fact about the *shape at balanced routing*,
  not about the config, and the source says so explicitly.
- **Workload dependence.** It is the reason a dispatch-side overlap buys nothing here and
  the reason skew is the only regime where a service pool could re-enter. Under real
  serving routes the same edge becomes the hot-rank straggler that holds the slab
  certificate hostage (LAW-46). **Structurally, the pre-M6 region is a sum of maxima:
  four grid barriers sit between M2 and M6** (`aug10/experiments/LESSONS.md:272-280`).
- **Confidence.** PROVEN-replicated (exp_10, exp_33, exp_35).

### LAW-32 — A `vmcnt`-based throttle is a **negotiated** mechanism, not an instruction you own; and unreachable code is not free.
- **Evidence.** A commit whose diff **cannot** change mode 12 changed it by **+726.9 µs
  (11%)**: 6,497.3 µs at `f113d73f` vs 7,224.2 at `291dfa08`, same session, with
  `production` and `pf6gm_mega` unchanged to <5 µs, all of it in M7 (+818 µs)
  (`aug11/LESSONS.md:407-422`). Mechanism, measured: with the extra mode compiled in, the
  atomic issue-run distribution collapsed from **282 atomics in 12 runs, mean 23.5 in
  flight** to **194 runs, mean 1.45** (189 of them a single atomic), `vmcnt(0)` in the
  epilogue **21 → 117** alongside **+96 scratch ops** — "the throttle was not deleted, it
  was made redundant by a stronger involuntary throttle installed by the register
  allocator" (`aug11/LESSONS.md:514-527`). The cleanest site ablation: **`if (mode == 14)
  return;` on a branch that cannot execute is one of the two worst sites**, while the
  largest lump of new code (+5,568 B `.text`) is inert — "code size is not the variable;
  where the code sits relative to the mechanism's live ranges is"
  (`aug11/LESSONS.md:528-540`).
- **Scope.** Shared megakernel functions at the 256-VGPR ceiling, this compiler.
- **Workload dependence.** None — it is a codegen law, and it is **the single biggest
  threat to the "hundreds of kernels as points in a knob space" plan**: a manifest of
  parameterized skeletons will silently re-time arms whose knobs it never touched, unless
  every arm carries a mechanism-level invariant check (LAW-58/59).
- **Confidence.** PROVEN-replicated (regression found, mechanism measured, guard-fix
  restored `.text` identity and the −618.7 µs contrast).

### LAW-33 — Deleting the per-task VMEM drain is worth **nothing** (+3.2 µs).
- **Evidence.** ~2,840 `vmcnt(0)` + ~2,840 `__syncthreads()` per CTA removed: **+3.2 µs,
  n=4 either side, sign reversed** (`g=481` 6,647.7 vs `g=353` 6,650.9)
  (`aug11/LESSONS.md:302-310`, `:438-444`). Also recorded: building the confound selector
  was worth it *because* the confound priced at zero — "without `kCoarseKeepDrainBit` the
  entire −576.5 µs would have been mis-attributed to signal granularity."
- **Scope.** Balanced prefill rig, mode-14 binary.
- **Workload dependence.** Drains are free where the memory system is not the limiter;
  expect a different answer where the epilogue is latency-bound.
- **Confidence.** MEASURED-once.

### LAW-34 — Four TP8-boundary overlap hypotheses are falsified in-rig.
- **Evidence.** "Four overlap hypotheses tried and falsified in-rig (**drain pipelining,
  register-source stores, consuming-pool specialization, item granularity**)"; next step
  named as a device phase ledger *before* further schedule mutations (commit `3f7f0f3a`).
- **Scope.** G25-1 fused CDAR rig, TP8+EP boundary, gfx950.
- **Workload dependence.** Note that **consuming-pool specialization** falsified here is
  the same family as LAW-28's dedication result in a different topology — two independent
  regimes now agree that a dedicated consumer pool does not pay on this hardware.
- **Confidence.** MEASURED-once each.

### LAW-35 — Fusing two jobs with different liveness into one primitive makes dead work undeletable.
- **Evidence.** `hkp::zero_part_scale_transpose` welded a 14 KiB buffer zero to a scale
  transpose sharing only a loop index; mode 12 killed the zero's consumer at exp_21 and
  the fusion kept **448 MiB/rank/epoch of dead stores** alive for four experiments.
  Deleting it: **−51.2 ±4.2 µs of the plan phase (t=12.3, 11 paired screens) and −73.7 µs
  / −0.90 points end to end (4 campaigns)**
  (`aug10/experiments/LESSONS.md:969-980`, `:1006-1018`).
- **Scope.** Any primitive library with fused epilogues.
- **Workload dependence.** None; it is a library-design law. Directly relevant to the
  HipKittens PR shape: **never fuse two jobs in one primitive unless they share a
  consumer.**
- **Confidence.** MEASURED-once (but with both a phase-stamp and an e2e number).

### LAW-36 — Cheap to produce is not cheap to consume: a layout conversion must be priced on the **consumer's** line traffic.
- **Evidence.** exp_27: M6's activation-scale gather read a group-major array we
  manufacture, fetching **5,376 distinct 64 B lines to deliver 21,504 B** (344 KB/task,
  ~977 MB/epoch); pointing M6 at the already token-major `sc_stage` makes a row exactly
  4 lines — **384 lines, 14.0× less line traffic** — for **ΔM6 = −79.7 µs (t = −14.8)**,
  n=10/arm over four alternating batches. Producing the group-major copy cost
  **−2.95 µs (t = −1.71, n.s.)**; consuming it cost **80 µs**
  (`aug10/experiments/LESSONS.md:1032-1053`).
- **Scope.** LDS-staged strided gathers on gfx950.
- **Workload dependence.** Amplification is a function of the stride relative to the
  64 B line, so it moves with hidden size, dtype and expert count. **Rule: a
  layout-conversion primitive is a defect until someone has named the consumer that
  requires the target layout AND shown the line arithmetic for both layouts.**
- **Confidence.** PROVEN-replicated (stamp + paired campaign agreeing within 4.7 µs).

### LAW-37 — Two mechanisms that were designed to compose may be **substitutes**, and the waterfall must say so.
- **Evidence.** The injection bound is worth **−613.5 µs**; deleting the per-row readiness
  protocol is worth **−573.3 µs** measured while the bound was **inert** — two numbers
  closing within 7%, both plausibly bounding the same in-flight-remote-write resource.
  "Neither has yet been shown to pay *on top of* the other"
  (`aug11/LESSONS.md:311-319`, `:453-459`). Recorded explicitly as an open question, not
  a result.
- **Scope.** Balanced prefill rig.
- **Workload dependence.** If they are substitutes, the paper's rungs are not additive
  and the cost model must not sum their coefficients. **Still untested.**
- **Confidence.** HYPOTHESIS (explicitly labelled "not a result" at source).

---

## 3. Serving-stack laws (vLLM + MORI, DP/EP vs TP, batching, coverage)

### LAW-38 — The DP8/EP8-vs-TP8 verdict flips on **batch fill**, not on kernel quality.
- **Evidence.** BRIEF digest, both cells with full workload tuples. **C=32** (1,024 QSL
  prompts, ISL 4096, OSL 8): M15 DP8/EP8 **20,372 input tok/s** vs AMD-recommended TP8+EP
  **25,844 → −21.17%**; TTFT p50/p99 2.445/4.364 s vs 1.339/5.682 s. **C=512** (2,048
  prompts, ISL 4096, OSL 8): M15 **42,788** vs AMD DP8/EP8 **39,474 → +8.40%**; TTFT
  43.2/47.9 s vs 28.9/52.8 s. The mediating quantity is measured: **C=32 → 1,539/4,096
  real rows per rank step (37.6% fill), ~56% of steps pure dummy work; C=512 → ~4,035/4,096
  (98.5% fill), <2% dummy**. Mechanism: DP starves at low C, TP8's all-reduce bytes scale
  with batch, and at high C our kernel overlaps the MoE all-to-all + combine and wins.
- **Scope.** DeepSeek-R1-class MoE, 8×MI355X, MORI/vLLM 0.25.1, ISL 4096 / OSL 8.
- **Workload dependence.** This is **the** flagship workload→schedule law and the
  template for every other cell: concurrency moves `fill`, `fill` moves which parallelism
  strategy's fixed cost dominates. It should be re-measured at other ISL/OSL and at
  decode-heavy mixes before being stated as a general TP-vs-EP rule.
- **Confidence.** MEASURED-once per cell (two cells) — and subject to LAW-45's ~15%
  run-to-run e2e variance, which is *larger than the +8.40% win*. Treat the C=512 sign as
  MEASURED-once and the magnitude as uncertain until order-balanced ≥5 pairs land.

### LAW-39 — The kernel-level win is a strong function of tokens per rank: it is 0.756× at T=4096 and **we lose** at T=1024.
- **Evidence.** BRIEF digest prefill T-sweep (M15 C=28 vs production): **T=4,096:
  0.7557×; T=2,048: 0.8845×; T=1,024: 1.0940× (we lose)**. The supporting kernel ladder at
  T=4096: production 7,712 µs → homogeneous mega 6,908.8 (0.8958×) → mode-12 depth-4
  6,483.8 (0.8407×) → M15 16 consumers 6,292.4 → 24: 5,848.5 (0.7589×) → **28: 5,822.0
  (0.7544×)** → 32: unstable (hung 1 of 2 runs).
- **Scope.** Prefill boundary, 8×MI355X, balanced-ish synthetic routing.
- **Workload dependence.** This is the only workload axis with a measured **crossover**,
  and it is the seed of the cost model: fixed per-epoch costs (plan, barriers, protocol
  events, ~415 µs plan phase and four pre-M6 grid barriers) amortize over T, so the
  megakernel's advantage is a decreasing function of the fixed/variable ratio.
- **Confidence.** MEASURED-once. **Carry the historical caveat:** an earlier generation
  of both megakernels was outright *incorrect* at T=1024/2048 (7 of 8 ranks' outputs
  unwritten in the MPS arm; `pf6gm_mega` max_abs 1.71/1.81 with `pperr = 0` — a silent
  under-write, reproduced 6/6 configs, `aug11/LESSONS.md:165-179`). The draft8 T-sweep is
  a later kernel; the defect's repair should be cited whenever the sweep is shown.

### LAW-40 — A serving A/B is invalid unless **padded work per real token (W)** is matched between arms.
- **Evidence.** The first honest c32p pair: near-identical real traffic (3.250 M vs
  3.315 M node tokens, −2.0%) but the m15 arm ran the padded B4096 graph on **249.1 of
  300 steps vs stock's 205.9 (+21.0%)**, giving **W = padded MoE rows per real token =
  2.568 vs 2.154 = 1.1922** (`aug18-prefill/BOTTLENECK_SIGNALS.md:20-24, 326-336`). At
  f ≈ 0.44 that inflation alone is worth **+8.4% wall — the entire measured gap** (the
  observed gap was +8.51%).
- **Scope.** vLLM DP8 piecewise-synced prefill, c=32, ISL 4096, OSL 8, 1,024 prompts.
- **Workload dependence.** `in_bucket` means DP padding put **all 8 ranks** in the 4096
  bucket because ≥1 rank had a full chunk — **both arms run the same padded graph; the
  padding is a property of the deployment, not the kernel**
  (`aug18-prefill/BOTTLENECK_SIGNALS.md:284-289`). Any scheduler change that alters
  co-scheduling density changes W and therefore fakes a kernel result.
- **Confidence.** MEASURED-once (n=1 per arm) — and the doc names re-running n≥3 as
  experiment #1 (`:506-514`).

### LAW-41 — Three composition-independent estimators put the mega's padded step at **≤** stock's; the naive back-solve (1.15–1.27) is an artifact.
- **Evidence.** (a) Two-state wall fit on client walls + receipt counts: a **single
  arm-independent (a, b) = (601.6 ms, 368.1 ms) reproduces both arms' walls exactly**,
  assigning the mega zero penalty. (b) **TPOT ceiling ratio 0.9885** (top-50 mean 755.30
  vs 764.07 ms) — the regime where every gap landed on a padded step. (c) Matched receipt
  windows, stock-calibrated: **δ negative for every plausible cheap-step cost, −8 to
  −129 ms**. Composition-corrected result: **r ≈ 0.95, interval [0.78, 1.06]**
  (`aug18-prefill/BOTTLENECK_SIGNALS.md:28-38, 354-428, 474-481`). The 1.175× hypothesis
  is excluded at ≈5σ of the timestamp error, conditional on the model (`:394-397`).
- **Scope.** As LAW-40.
- **Workload dependence.** The residual to explain is therefore **≈0.95 measured vs
  0.76–0.85 banked in replay, i.e. 12–25%** — not the 35–45% assumed before the analysis.
- **Confidence.** MEASURED-once, but with **three estimators that do not share an error
  model** — treat as the strongest single-pair result in the serving ledger.

### LAW-42 — Padding dominates serving MoE work, and it is **routing-degenerate** padding.
- **Evidence.** Measured: **62.4% of every MoE row the m15 arm computed was padding
  (50.8% for stock)**; **[62.4%, 87.5%] of rank-steps carry fewer than 4096 real rows**;
  fully-packed steps are **effectively zero**; at f ≈ 0.45 that is **≈65 s of m15's
  217.6 s wall and ≈43 s of stock's 200.5 s spent on rows that do not exist**
  (`aug18-prefill/BOTTLENECK_SIGNALS.md:42-44, 264-271, 290-313`). Fill *declines through
  a run* in both arms (stock 0.4170 → 0.3695; m15 0.3511 → 0.3348, `:273-281`).
  Mechanism prediction: vLLM pads with token id 0 → identical hidden states → identical
  router logits → the same top-8 experts, so ~62% of dispatched rows land on ≤8 of 256
  experts, giving **2.66× rank-load skew from padding alone, rising to ~5× with two hot
  experts on one rank** — quantitatively consistent with the measured **per-call max-rank
  load p50 of 5.15×** "and with essentially nothing else" (`:524-530`).
- **Scope.** vLLM piecewise-synced DP prefill with an exact-bucket graph.
- **Workload dependence.** This makes **skew a partly synthetic, deployment-generated
  quantity**: the replay corpora that showed 0.756–0.847 never contained it. It also
  means a fill-aware or dummy-skipping kernel changes the *routing distribution*, not
  just the row count.
- **Confidence.** MEASURED-once for the fill/duty arithmetic (exact from receipts);
  HYPOTHESIS for the degeneracy mechanism — the decisive check
  (`np.unique(topk_ids, axis=0)` on a captured route npz) is free and named as action #1.

### LAW-43 — Corpus-level: ~31% of sealed steps are 100% fake work both arms pay, and the real-call hot set is small.
- **Evidence.** "31% of sealed steps are 100% fake (dummy/pad) work both arms pay,
  real-call hot-set {0..7} on rank 0 covers ~16% of rows (**2.61× p50 real skew**),
  run-correlation ~nil (interleaving demoted); optimization order = fill/dummy-skip then
  RR placement" (commit `4a5d87d7`). Companion: **DP-dummy calls are perfectly
  rank-synchronous** — `c[i]` bimodal on {0,8}, **κ = 1.000 over 28 pairs**, index
  alignment validated by a sharp lag-0 `n_real` correlation peak (**+0.996 vs ~0.65 at
  ±1**) — so a tier-1 whole-step skip needs **no synchrony discount but must still be a
  collective decision** (commit `2ba124c8`).
- **Scope.** Captured c32p route corpus, 8-rank DP.
- **Workload dependence.** The 31% (capture window) vs ~56% (whole run) discrepancy is
  itself a known, reconciled artifact — the corpus's 1.19× figure was **withdrawn as an
  nz-filter artifact (true all-call aggregate 1.446×)** and the residual gap attributed to
  the whole-run DP-dummy rate (commits `1629fa41`, `46a0e723`).
- **Confidence.** PROVEN-replicated for the synchrony (κ = 1.000, two independent
  alignments); MEASURED-once for the corpus fractions.

### LAW-44 — Median TTFT can **fall** while throughput falls: metric choice decides the verdict.
- **Evidence.** Same pair as LAW-40: **m15 median TTFT is 5.0% faster** (2,173.76 vs
  2,288.68 ms) while node input tok/s is **−7.84%**
  (`aug18-prefill/BOTTLENECK_SIGNALS.md:88-99, 66`). The doc's reading: "A kernel that is
  15–20% slower on the MoE region of every padded step **cannot lower any TTFT quantile**.
  A *scheduler* that packs fewer prefills per step can, and does." TPOT tells the same
  story as a **mixture, not a shift**: m15's TPOT sd narrows 17% (116.2 → 96.9), p10 rises
  28%, p99 falls 0.7% (`:121-144`).
- **Scope.** Closed-loop c=32 harness.
- **Workload dependence.** This is the empirical basis for the BRIEF's metric-map
  requirement: at low concurrency TTFT and throughput can move in **opposite** directions,
  so a single headline number is not just incomplete — it can carry the wrong sign.
- **Confidence.** PROVEN-replicated across three independent statistics (TTFT quantiles,
  TPOT mixture shape, phase decomposition).

### LAW-45 — Serving e2e carries ~15% run-to-run variance; boundary rigs are the precision instrument.
- **Evidence.** BRIEF constraints: "~15% e2e run variance → boundary-level rigs are the
  precision instrument; e2e is the validation instrument." Corroborated by the analysis's
  own n=1 caveats and by the benchmark protocol's requirement of **order-balanced ≥5
  pairs, open-loop Poisson cells, an accuracy gate, TTFT p50/p99, and a native-tuned
  baseline** (`nightshift/BENCHMARK_PROTOCOL.md`, cited in the BRIEF assets list).
- **Scope.** vLLM serving on this node.
- **Workload dependence.** Sets the **minimum detectable effect** for every e2e cell in
  the experiment ladder — any predicted schedule change smaller than ~15% must be
  adjudicated at the boundary, not e2e.
- **Confidence.** PROVEN-replicated (stated as a measured node property; consistent with
  the 200.5 s vs 217.6 s single-pair spread and the phase-level +0.7%/+9.8% split).

### LAW-46 — Expert placement decides which rank is critical, and the hot rank holds the global certificate hostage.
- **Evidence.** BRIEF digest route capture: **51.9% of expert-slot traffic went to experts
  0–7, all resident on rank 0 under contiguous placement** → "hot rank produces its slab
  late and holds the global slab certificate hostage. Skew is a first-class workload
  axis." At the captured operating point, real serving is governed by the hottest expert
  rank: **e2e p50 −1.06%, TTFT p99 +5.67%, TPOT p50 −2.51%** (M15 vs production).
- **Scope.** Captured DeepSeek-R1-class routes, EP8 contiguous placement.
- **Workload dependence.** The fix landed as **RR-by-permutation placement** (weight
  loader maps `e → rank e%8, slot e//8`, one static router gather permutes topk_ids
  logical→physical, every runtime tensor stays linear) after guard-widening was
  **rejected** — MORI never reads vLLM's RR routing tables, so widening would ship logical
  ids to RR-placed weights = silent wrong-rank dispatch (commit `92e5017c`).
- **Confidence.** MEASURED-once (one capture, one operating point).

### LAW-47 — Offline replication frontier: uniform-K replication **halves** skew damage but does not remove it, and per-layer K tuning buys ~nothing.
- **Evidence.** Greedy marginal-benefit allocation over banked aggregate histograms
  (90,050 calls, 42 MB/expert): raw mean max-rank load **5.05×**; uniform-K rows
  **K=2 (4.8 GiB) → 4.10×; K=4 (9.5 GiB) → 3.23×; K=6 (14.3) → 2.41×; K=8 (19.0) → 1.77×;
  K=16 (38.1) → 1.09×**; the frontier is near-linear to ~18.5 GiB (~0.56× per GiB), flat
  18.5→31 GiB, completing at 38 GiB. **Greedy ≈ uniform + ε at equal bytes**
  (`aug15/OVERLAP_PROGRAM.md:36-46`).
- **Scope.** Offline analysis of captured aggregate histograms — **not** a kernel or
  serving measurement.
- **Workload dependence.** The named open risk: "the aggregate frontier **cannot price
  the per-chunk coverage bimodality (p5 = 1%, p75+ = 77%)** that a static replica set runs
  into", which is why raw captured routes are a hard prerequisite (`:47-56`).
- **Confidence.** ANALYTIC (offline allocation over measured histograms).

### LAW-48 — Serving prefill has almost **no independent filler**, so overlap is second-order to moving compute.
- **Evidence.** The filler inventory: wgrad (~half of bwd MoE) and shared-expert bwd
  (~1/4 of routed bwd) exist **only in training**; serving prefill has "NONE of the first
  three at meaningful scale (no wgrad, shared expert only ~1/8 fwd) — which is exactly why
  'M15 with more overlap' plateaued and why the training pivot is the right lab. In
  serving, the only lever that moves the skewed critical path is MOVING COMPUTE
  (replication); overlap is the second-order term" (`aug15/OVERLAP_PROGRAM.md:9-33`).
  The forward dependency chain leaves exactly one legal cross-region forward overlap
  (combine(L) tail vs attention(L+1) prologue on finished tokens).
- **Scope.** MoE forward, EP8, serving prefill.
- **Workload dependence.** Flips completely in training (LAW-53) and would flip in serving
  if a shared-expert or speculative-decode branch became large enough to be filler.
- **Confidence.** MEASURED-once for the plateau; ANALYTIC for the FLOPs shares.

### LAW-49 — Coverage is a deployment property and must be quoted with every serving number.
- **Evidence.** Under the original exact-4096 seal the megakernel fired on **~2% of heavy
  steps and both arms' baselines were depressed** — the pre-M23 serving pairs were
  therefore **retired as obsolete** (`aug15/OVERLAP_PROGRAM.md:104-108`; retirement
  commits `9fb1705d`, `824a3e45`). After the M23 ragged seal, measured duty is
  **sealed/in_bucket = 0.99147, sealed/steps = 0.82333, token coverage 0.94374**
  (`aug18-prefill/BOTTLENECK_SIGNALS.md:256-271`), with the offline replay showing
  **3 → 211 sealed steps, 70× eager elimination** on the synthetic c32p trace (commit
  `6307abc7`).
- **Scope.** vLLM DP8 with the M23 patch chain.
- **Workload dependence.** The seal predicate itself is a *collective* decision — a split
  seal deadlocks, and an earlier revision let one rank refuse while seven replayed the
  graph, with the receipt still reporting green (commit `eeff2c74`). Coverage must ride an
  all-reduced readiness bit, never a local term.
- **Confidence.** PROVEN-replicated (offline replay + live receipts across 8 ranks).

### LAW-50 — DP prefill co-scheduling is worth more than any kernel delta in this project's ledger.
- **Evidence.** Perfect co-scheduling would need **128 padded steps instead of 260/322**;
  at (a=602, b=368 ms) stock 200.5 s → **171.2 s (−14.6%)** and m15 217.6 → **171.2
  (−21.3%)**; at (a=721, b=227) both → **150.4 s (−25.0% / −30.9%)**. "**15–25% of node
  throughput is sitting in DP prefill co-scheduling, for either kernel. That is larger
  than any kernel delta in this project's ledger**"
  (`aug18-prefill/BOTTLENECK_SIGNALS.md:483-496`). Directly measured driver: prefill
  chunk-events per in-bucket step, **stock 3.93 vs m15 3.19 of 8 possible** (`:307-311`).
- **Scope.** c=32 closed-loop, ISL 4096, DP8.
- **Workload dependence.** Vanishes at high concurrency where fill ≈ 98.5% (LAW-38) —
  which is precisely why the C=512 cell is the one we win.
- **Confidence.** ANALYTIC from measured step counts and two fitted step-cost pairs.

### LAW-51 — Destination-interleaved record order and fabric hygiene are cheap and matter under any destination concentration.
- **Evidence.** Transport hygiene ranked by measured value: **fp8-on-wire combine —
  MORI-measured 366 → 642 GB/s class**; **destination-interleaved dispatch/combine record
  order — MORI measured 65% on push-combine**, "matters under ANY destination
  concentration (real serving routes; ragged training tails). Cheap."; **`s_sleep` backoff
  in every poll loop** (fabric livelock insurance; our spin loops predate the lesson);
  **source-side same-token pre-reduce is LOW value at EP8** (top-8 spreads ~1 expert/rank;
  the M1 dedup already folds same-dest picks) — revisit at EP16+
  (`aug15/OVERLAP_PROGRAM.md:77-88`). Corroborating pathology: MORI's own 2-of-7-link
  concentration and their **822 → 498 µs RR-interleave fix**
  (`aug18-prefill/BOTTLENECK_SIGNALS.md:537-543`).
- **Scope.** MORI-measured (vendor), MI300X/MI355X EP.
- **Workload dependence.** The pre-reduce law is explicitly EP-degree dependent — a
  measured example of a knob whose value is a function of `topk × EP`.
- **Confidence.** MEASURED-once (vendor-internal measurements, not reproduced by us —
  treat as `REPORTED` until re-run on our pinned stack).

---

## 4. Training-side laws (and whether they transfer to serving)

### LAW-52 — **Critical-rank law:** iteration pace = the critical (hottest, lowest-clocked) rank's wall clock. Scheduling only helps e2e when it removes work from **that** rank's wall. — **TRANSFERS TO SERVING.**
- **Evidence.** "The governing law the drift math forces: iteration pace = the critical
  (hottest, lowest-clocked) rank's wall clock. Filler and scheduling only help e2e when
  they remove work from THAT rank's wall — in-window filler qualifies (every rank has the
  1.6 + 1.3 ms service-pool idle); **cross-rank wait-absorption and host-side reshuffling
  do not**" (`aug18/V6_EVIDENCE_AND_DESIGN.md:115-118`). Underlying measurement: the M2
  `rows_done` wait is rank-structural drift, **92 µs (rank 2) to 4,353 µs (rank 0) per CTA
  per launch — rank 2 is the straggler and the other seven ranks stall ~2.5 ms average
  every backward launch**; and it is **data-driven (rank 2 genuinely arrives late), not
  protocol slack** (`:41-49, 66-70`).
- **Scope.** 8-GPU MoE training, 32 MoE layer-microbatches/iter, gfx950.
- **Transfer.** Direct and already observed in serving: the hot **expert** rank plays the
  same role as the hot **hardware** rank (LAW-46), and the certificate/slab rendezvous is
  the same mechanism that converts one rank's lateness into everyone's stall.
- **Workload dependence.** The identity of `crit` changes with the source of skew:
  thermal/clock (training, uniform routing) vs routing/placement (serving). A schedule
  tuned for one will not find the other.
- **Confidence.** PROVEN-replicated (drift ledger + every falsified rescheduling variant).

### LAW-53 — In-launch bubble supply is ~5% of filler demand: size the filler against a measured bubble ledger before designing a scheduler. — **TRANSFERS as a method.**
- **Evidence.** Bubble ledger per backward launch: service slab-0 idle **1.61 ms × 28
  CTAs**, slab-1 post-quota **1.23–1.36 ms × 28**, M8 certificate wait **36–177 µs × 256**,
  M0 barrier **~250 µs × 256** → in-launch supply **~150–180 CTA-ms vs in-situ wgrad demand
  ~3,360 CTA-ms (13.1 ms/launch, 1.8× the 7.4 ms gate figure)**
  (`aug18/V6_EVIDENCE_AND_DESIGN.md:41-49`). Every wgrad rescheduling variant then lost:
  control **1,670/1,689**; pure in-kernel filler **1,892**; partition + monolithic
  128-CTA slice + fence **1,959–1,962**; dribbled slices + fence **1,988–1,993**;
  no-fence variants **1,927–1,954** (`:30-40`). The endgame filler that *does* fit is
  shared-expert GEMMs — "~2 ms/layer·mb of work vs ~150 CTA-ms of window supply — it fits
  where wgrad's 13 ms could not" (`:88-92`).
- **Scope.** t2v6 backward megakernel, 256 CTAs, gfx950.
- **Transfer.** The *method* transfers exactly: a per-wait-site idle ledger (measured cost
  **zero**: 1,673/1,689.7 vs control 1,670/1,689, `:14-18`) is the correct first
  instrument before any overlap design in either program. The magnitudes do not transfer.
- **Confidence.** PROVEN-replicated (ledger + five falsified variants).

### LAW-54 — **All-CU residency law:** a 256-CTA all-CU megakernel creates a residency contract; only **short** kernels slip past it. — **TRANSFERS TO SERVING.**
- **Evidence.** "Slice convoys block mega residency where **rocBLAS's sub-ms kernels slip
  through**" — one of the three reasons every wgrad rescheduling variant lost
  (`aug18/V6_EVIDENCE_AND_DESIGN.md:64-70`). Structural diagnosis: "Our all-256-CTA grid
  barrier creates the residency contract that made every separate wgrad kernel standoff or
  serialize, quantizes execution into lockstep phases whose bubbles are small (~150 CTA-ms
  vs wgrad's ~3,360), and realizes cross-rank drift at **three points per launch (M0/M2/M8)
  instead of one per layer**" (`aug18/T3_COUNTER_DATAFLOW_DESIGN.md:43-49`). The
  counter-example kernel (MoK on Blackwell) builds none of those constructs, so the
  pathologies are structurally absent there (`:8-40`).
- **Scope.** Persistent all-CU megakernels at 1 block/CU (LAW-12).
- **Transfer.** Immediate: any serving proposal that puts work on a *side stream* against
  the resident megakernel inherits this law, and only sub-millisecond kernels qualify.
- **Workload dependence.** Relaxes if the megakernel stops being all-CU (a task-pool grid
  with early exit, MoK-style) — which is precisely the T3 port's motivation.
- **Confidence.** MEASURED-once (via the falsified variants and the residency contrast).

### LAW-55 — Host time that looks like deletable glue can be **backpressure**.
- **Evidence.** K0HT host-time itemization: **b_wgrad_host 270 ms/iter, f_launch_sync
  287 ms/iter, b_launch 26, everything else ~18**. Graph-capturing the bmm block
  **was a WASH (1,685.7 vs 1,684)** and b_wgrad_host did not collapse → "the big host
  numbers are BACKPRESSURE (host parked on deep stream queues), not deletable glue"
  (`aug18/V6_EVIDENCE_AND_DESIGN.md:108-112`).
- **Scope.** PyTorch/Megatron training loop with ctypes-launched megakernels.
- **Transfer.** Serving integration overhead is bounded independently at
  **|δ| ≲ 15 ms/step ≈ ±2% of a padded step**
  (`aug18-prefill/BOTTLENECK_SIGNALS.md:559-568`), which is consistent with this law:
  the host is not the lever in either program until the device queue is shallow.
- **Confidence.** MEASURED-once, with a decisive negative control (the graph capture).

### LAW-56 — Precision claims must be verified **at runtime**, not from config: the production FP8 bar ran its MoE in **BF16**, and FP8 in the MoE is worth ~0 ms on this workload.
- **Evidence.** Under the `delayed` recipe Primus returns a null turbo quant config, so
  `enabled_turbo=False` and `PrimusTurboGroupedLinear` takes the BF16 branch; the runtime
  log carries the proof line (`aug18/FAIRNESS_AUDIT.md:23, 119-126`). Iteration-aligned
  recipe table (iters 9–15, same chassis): e4m3+tensorwise **1,335–1,356 ms**,
  hybrid+tensorwise **1,344–1,358**, hybrid+delayed (BF16 MoE) **1,334–1,343** ⇒
  "**Turbo GG's fp8 path is worth ~0 ms on this workload**", and turbo GG *itself* is worth
  ~180 ms regardless of precision (`:94-100`).
- **Scope.** Primus/Megatron on ROCm, DeepSeek-class MoE training, 8 GPUs.
- **Transfer.** The *discipline* transfers to every serving baseline claim; the specific
  finding also supports a BF16 megakernel variant.
- **Confidence.** PROVEN-replicated (source trace + runtime log + three-recipe timing
  A/B).

### LAW-57 — Self-inflicted FLOPs, not the chassis, dominate the training gap; and the coverage pathology is absent in training.
- **Evidence.** Three-auditor verdict: the mega kernel **runs 100% of MoE
  layer-microbatches (32/32)** — no serving-style pathology; but the mega arm issues
  **30.5% more MoE FLOPs (~176 ms/iter self-inflicted: z-regen 86–101, wgrad 20%-dead
  padding 39, staging 45–75, producer 25–45 exposed)**; measurement bias **~23 ms** against
  us; **honest steady state 0.845× (1,589.5 vs 1,343.0)**; backbone handicap only
  **7–15 ms/iter (point estimate ~10)**, i.e. 2.4–5.1% of the 293.7 ms BF16 gap
  (commit `36334664`; `aug18/FAIRNESS_AUDIT.md:19, 85-92`). Most important unstated number:
  the mega swap on its own chassis is worth **+16.5 ms (+1.0%)** while chassis+MoE together
  are worth **293.7 ms (+17.5%)** (`:29`).
- **Scope.** Training A/B, 260 iters, rank-7 harmonic mean.
- **Transfer.** The *asymmetry taxonomy* (issued-FLOPs delta, precision mismatch,
  measurement window bias, missing controls) is the checklist any serving fairness audit
  should reuse — and it is the same discipline that found LAW-40's composition confound.
- **Workload dependence.** Named and directional-but-unmeasured: **uniform routing
  flatters the mega arm** — its wgrad is a uniform-capacity baddbmm over [32, 1280] padded
  rows (**25% padding even at perfect balance**) while turbo GG consumes ragged groups
  natively, so **skew will hurt the mega arm more** (`aug18/FAIRNESS_AUDIT.md:58`).
- **Confidence.** PROVEN-replicated (three independent auditors, config diff + timer
  ledger + source trace).

---

## 5. Measurement & instrumentation laws (cross-cutting; they set the calibration plan)

### LAW-58 — The resource tuple is **not** a parity gate; write the gate against the mechanism's invariant.
- **Evidence.** A build passed **every** gated field — SGPR, VGPR, AGPR, scratch/lane,
  LDS, MFMA census, `flat_atomic_pk_add_bf16` census, zero-scratch-ops-inside-either-MFMA-
  span — and still cost the ratchet **+726.9 µs (11%)** (`aug11/LESSONS.md:234-248`,
  `:407-422`). The fix: **`.text` sha256 identity is the only parity gate we trust for a
  change claimed inert**, plus the second half — "**the flag-ON builds must differ**"
  (M14 192,448 B, RING 179,648 vs REF/DEF 179,520); "a flag-on build that came out
  identical would mean the flag never reached the code and the arm would be a lie"
  (`:495-505`). Also add **scratch OP COUNT and SGPR/VGPR spill counts**, because
  `ScratchSize` stayed pinned at 128 B/lane while scratch ops went **19 → 168** and spills
  **186 → 217 / 15 → 17** (`:506-513`).
- **Confidence.** PROVEN-replicated. **Binding on the manifest-driven knob space.**

### LAW-59 — Confirm a fix on the **mechanism**, not the wall clock; and a knob whose off-state is not `.text`-identical is a second arm.
- **Evidence.** Two checks after the exp_38 repair: the ratchet reproduced at **6,488.7 vs
  6,482.7 µs (+0.09%)** *and* the injection contrast returned to **−618.7 µs** against
  −613.5 at the healthy pin and −1.8 at the broken pin — "a timing coincidence cannot move
  a contrast by **344×**" (`aug11/LESSONS.md:548-555`). Corollary rule at `:495-505`.
  Also: **never publish a number from a binary carrying a `-D` that adds a mode** — the pin
  *and the `-D` set* are part of a result's identity (`:541-547`).
- **Confidence.** PROVEN-replicated.

### LAW-60 — Instrument resolution ladder: **screens order points; campaigns set the ratchet; stamps attribute mechanisms.**
- **Evidence.** Screens: documented σ = 0.52% but a **repeated control drift of 6.6%
  inside one batch (6,795 → 7,244)** with no foreign process — "treat ~3% as the smallest
  callable end-to-end screen delta". Stamps: plan σ = 4.7 µs (1.1%), M7 σ = 63 µs (2.3%).
  Campaigns reproduce to **0.09%** (`aug10/experiments/LESSONS.md:997-1005`). Calibration
  of the stamp against e2e: **stamp ΔM6 −79.7 µs vs paired campaign −75.0 µs (within
  4.7 µs, ~1:1 pass-through)**, while the same 20 runs read **−51 µs at t = −0.72**
  end-to-end (`:1089-1098`). Cadence: a 5-rotation campaign (500 warmup / 100 timed /
  600-epoch soak) takes **~3 min 5 s** — earlier sessions budgeted ~20 min and under-ran
  the GPU by ~5× (`aug11/LESSONS.md:8-16`; `:473-477`).
- **Confidence.** PROVEN-replicated. **This is the sample-size input for the experiment
  ladder's gate design.**

### LAW-61 — Cluster by run; digits are not reproducible; adjudicate per-rank controls per-rank.
- **Evidence.** (a) The same null reads **t = −5.01** if 160 (run, rank) cells are treated
  as independent and **t = −0.73** with runs as units — eight ranks in a run share one
  input (`aug10/experiments/LESSONS.md:1076-1080`). (b) `[MOK GATE]` digits are **not
  run-reproducible in any arm** (dispatch assigns receive rows by `fetch_add`; the combine
  is a nondeterministic bf16 remote atomic) — a **byte-identical `.text`** printed
  different `max_abs` across five runs; the sound substitute is the **same-run paired
  difference against an untouched arm** (`:1062-1075`). (c) A rank-N-only failure is
  **invisible in rank-0 stdout**: the mode-14 negative control printed `pperr=0` and a
  `nan` gate on rank 0 while rank 7 carried the exact pre-registered signature
  (`aug11/LESSONS.md:264-272`).
- **Confidence.** PROVEN-replicated (each with a near-miss that cost real time).

### LAW-62 — Pre-register thresholds against a **measured** ceiling, and prove a knob is settable before pre-registering an axis.
- **Evidence.** (a) exp_22's H1 demanded 75% of the 76.8 GB/s nominal (57.6 GB/s) and
  fired `REFUTED_NEVER_REACHED` — but the achievable ceiling is **56.9–57.3 GB/s**, so
  **no CTA count could have passed**; "a spec-sheet denominator can manufacture a
  refutation" (`aug11/LESSONS.md:207-217`). (b) **The routing-imbalance axis does not exist
  in that harness**: `K0_SYNTH_ROUTE` is rejected under `K0_INPUT_MODE=mok_synthetic`,
  the synthetic-route families are decode-only, `synthetic_routes.py` exposes **no `std`
  parameter**, and the only reachable knob (seed) spans destination-load CV
  **0.0034–0.0086 — 4–15× *less* imbalance than COMET's 0.032**. So the pre-registered
  claim "skew is the ONE regime where a service pool may re-enter" is **untestable on that
  harness, not refuted** (`aug11/LESSONS.md:148-164`).
- **Confidence.** PROVEN-replicated. **Directly binding on the ablation campaign: every
  axis in W must be shown settable before a cell is pre-registered.**

### LAW-63 — Device-timestamp hygiene (five traps, each cost a measurement).
- **Evidence.** (a) `realtime_now()` issued `s_memrealtime` with **no `s_waitcnt
  lgkmcnt(0)`**, so four of six call sites read a stale register — **every prior
  `K0P6_MPS_TS_*` reading was unreliable** (`aug10/experiments/LESSONS.md:286-300`).
  (b) **1 tick = 0.01 µs (100 MHz); using the 2.2 GHz shader clock is a 22× error.**
  (c) Stamps are **running maxima never reset**, so after a 600-epoch soak they describe
  the final *soak* epoch, not a timed iteration (`aug11/LESSONS.md:85-98`).
  (d) The MPS phase print sits inside `if rank == 0:` and is never all-reduced, so every
  MPS phase number ever reported is the **CTA-max within rank 0**; production's genuine
  cross-rank combine reads **1,356.0 rank-max vs 978.4 rank-0 (+38%)** — the two
  instruments are not comparable tick-for-tick (same lines).
  (e) **Never subtract `clock64()` values written by separate kernels** — one such span
  read **55,576.5 µs against a 1,278.81 µs HIP event** (`aug12/LESSONS.md:72-76`).
- **Confidence.** PROVEN-replicated.

### LAW-64 — The estimand is synchronized global makespan of the complete dependent DAG; bandwidth wins that do not reduce it are transport wins, not overlap wins.
- **Evidence.** Stated as the study's primary methodological commitment
  (`aug12/LESSONS.md:5-11`; `aug11/OVERLAP_METHODOLOGY_STUDY.md:1171-1172`), with the two
  required comparison families — **transport-equivalent** (identical materialized source,
  payload, offsets, signal semantics, consumer checksum) vs **complete-path** (direct
  producer publication may legitimately remove staging; SDMA must pay for materialization,
  enqueue, completion, pack and unpack) (`aug11/OVERLAP_METHODOLOGY_STUDY.md:504-536`).
  Adopted benchmark metric for the training program: **median over iterations of the
  MAX-across-ranks latency** (drift-inclusive critical path,
  `aug18/T3_COUNTER_DATAFLOW_DESIGN.md:39-40`).
- **Confidence.** Methodological commitment, enforced across every campaign since aug12.

### LAW-65 — Re-profile after every landed win; a queue inherited from a stale profile points at the wrong phase.
- **Evidence.** exp_33: **M7 (2,701.8 ±23.74 µs, 46.2% of the interior) overtook M6
  (2,453.3 ±2.51, 41.9%)** after exp_27's −79.7 µs took M6 down — the flip "supersedes the
  budget table in STATUS.md and with it the M6-first ordering of the optimization queue"
  (`aug11/LESSONS.md:17-27`). The two GEMMs are **88.1% of the 5,852.1 ±9.56 µs interior**;
  plan is 372.8 (6.4%), combine 324.2 (5.5%).
- **Confidence.** PROVEN-replicated. Corollary for M: the phase profile is **not** a
  constant of the kernel; it is a function of the current best schedule.

---

## 6. Known contradictions, and how they resolve

| # | Apparent conflict | Resolution | Residual risk |
|---|---|---|---|
| C1 | **Depth-4 is decisive** (LAW-20) vs **depth is flat** (LAW-8) | Scope: `co` (compute co-residency). Depth is congestion control on a *shared* memory path; with no co-resident GEMM there is nothing to protect. | The depth-flat result is MEASURED-once on a different rig; re-run before M commits to `d(co)`. |
| C2 | **Pacing is dead** (LAW-22 first half) vs **the throttle pays 500–615 µs** (LAW-20) | Who is paced. Pacing a *dedicated pool* whose drain is the critical path (LAW-25) directly extends the makespan; bounding a *producer that is also computing* removes contention from the phase it shares. | The rule has never been tested on a third carrier. |
| C3 | **C=64 dedication wins** (mode 2, `aug10/…LESSONS.md:669-676`) vs **dedication never wins at any size** (LAW-28, mode 12) | Carrier-conditional. With a staged-push carrier the pool is the whole transport and more pushers help to a turnover at C≈96; with a producer-carried epilogue the pool has no job, so each reserved CTA is pure capacity loss (LAW-27). | Neither was measured under skew — LAW-62(b) says that axis was unsettable. |
| C4 | **Interference is protocol, not payload** (LAW-16) vs **the residue is the pool's own memory traffic** (`aug10/…LESSONS.md:722-733`) | exp_20's payload-deletion arm is the *direct* measurement and supersedes exp_16's atomic-rate extrapolation. But both agree there is a **g-independent floor (+1,159 µs) whose mechanism is unidentified**; only 36% is localized (LAW-19). | Named at source as "the single largest gap in our understanding of this kernel". |
| C5 | **Replay says 0.756–0.847×** vs **serving says r ≈ 0.95** (LAW-41) | Composition (LAW-40) removes ~2/3 of the naive gap; **12–25% remains genuinely unexplained**, with (a) destination concentration and (b) hot-rank compute concentration both at "moderate" posterior and **not separable from serving data at all**. | Needs the captured-vs-shuffled route replay with device stamps (`aug18-prefill/BOTTLENECK_SIGNALS.md:537-557`). |
| C6 | **Injection bound and coarse readiness both ~−600 µs** (LAW-37) | Possibly **substitutes** (both bounding in-flight remote writes), never measured on top of each other. | If substitutes, no waterfall in the paper is additive. |

## 7. Retracted or withdrawn claims (do not re-cite)

1. **"The only remaining lever is to move fewer bytes"** (exp_17) — retracted by exp_20
   (`aug10/experiments/LESSONS.md:867-873`).
2. **"Promote SDMA to first because the bytes are the problem"** — written, then retracted
   within the same night by the g-axis (`:370-379`), then partially re-opened by the
   atomic-rate extrapolation (`:722-733`), then closed again by exp_20. Net status:
   **SDMA has never been measured in situ**; nothing in the ledger supports or refutes it.
3. **The corpus "1.19×" figure** — withdrawn as an nz-filter artifact; true all-call
   aggregate 1.446× (commits `1629fa41`, `46a0e723`).
4. **The training "matched-precision race"** and **"every fp8 trainer pays the
   weight-quant cost"** — both deleted (LAW-56, commit `36334664`).
5. **The training loss-quality claim** — withdrawn (frozen router = easier objective);
   the robustness findings stand separately (commit `36334664`).
6. **All pre-2026-08-18 serving numbers** — retired (commits `9fb1705d`, `824a3e45`;
   `aug15/OVERLAP_PROGRAM.md:104-108`), because the seal fired on ~2% of heavy steps.
7. **"Skew is the one regime where a service pool may re-enter"** — **untestable** on the
   aug11 harness, therefore neither established nor refuted (LAW-62b).
8. **"CTA specialization is generally bad", "SDMA is generally good/bad", "push beats
   pull"** — none of these is established; the corpus covers "one large, balanced prefill
   point and one specific dependency graph"
   (`aug11/OVERLAP_METHODOLOGY_STUDY.md:33-36`).

## 8. The biggest holes this document leaves (inputs the campaign must supply)

1. **The 64% of the interference floor with no identified mechanism** (LAW-16 residual) —
   no hardware-counter evidence, and no known way to collect PMCs for a *sub-phase* of one
   persistent launch (`aug10/EXPERT_BRIEFING.md:104-123`).
2. **The skew axis has never been driven** in the kernel rigs (LAW-62b), yet it is the
   single most important serving workload property (LAW-42, LAW-46).
3. **Batch/sequence-length dependence** is a two-point sweep on one axis (LAW-39) with a
   correctness history (`aug11/LESSONS.md:165-179`).
4. **Decode has never been measured at all** in any of our rigs — every kernel law above
   is a prefill law.
5. **Push/pull, engine (SDMA), per-destination vs global credits, separate persistent
   kernels, multi-stream/TBO, topology, multi-node** — enumerated as open in
   `aug11/OVERLAP_METHODOLOGY_STUDY.md:485-499` and still open.
6. **`d`, `A`, `L`, `u(t)`, `φ` have never been fit jointly.** Every law above is a
   one-knob-at-a-time result, and LAW-21 already proves at least one pair is
   non-separable.

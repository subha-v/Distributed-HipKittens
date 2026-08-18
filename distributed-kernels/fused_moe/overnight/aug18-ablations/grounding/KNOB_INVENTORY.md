# KNOB_INVENTORY — every megakernel SCHEDULE degree of freedom that exists in this repo today

**Purpose.** Output (2) of the ablation campaign brief (`../BRIEF.md:113-115`): the
**schedule state vector S**, organized as *skeletons* × *knobs*, so that "hundreds of
kernels" can be points in a manifest-addressable knob space rather than hand-written
one-offs (`../BRIEF.md:105-108`).

**Method.** Local design pass, no node access. Every entry is anchored to a repo path
(file:line where useful). Numbers appear **only** with a source: a repo doc, a git commit
message, a results file, or the BRIEF's measured digest. Anything not measured is marked
**SPECULATION**.

**Workload-axis tags** used in the interaction column (the axes of the workload state
vector W that this knob plausibly couples to):

| tag | axis |
|---|---|
| **T** | tokens per rank per step (sequence-length × batch → padded capacity) |
| **CONC** | serving concurrency / number of in-flight requests (renamed from the BRIEF's "C" to avoid collision with the knob named `C`) |
| **TOPO** | parallelism + topology: DP8/EP8 vs TP8+EP, world size, fabric |
| **SKEW** | routing skew / expert-load imbalance / hot rank |
| **FILL** | real rows ÷ padded capacity (the dummy-work fraction) |
| **PHASE** | prefill vs decode (vs training fwd/bwd) |
| **ARCH** | gfx950 vs gfx942 (and any future fabric/arch card) |

**Reading rule inherited from the corpus** (`aug12/OVERLAP_KERNEL_DESIGN_IDEAS.md:200`):
M7 and combine are anti-correlated — judge their **sum**, never either alone.

---

## PART A — STRUCTURAL SKELETONS (the outer choice; changing one is not a knob turn)

A skeleton fixes *who runs what role* and *where the launch boundaries are*. Every
skeleton below exists in the repo as code or as a gated design; they are mutually
exclusive builds, not config points.

| id | skeleton | defining structural property | where it lives | status / measured result (source) |
|---|---|---|---|---|
| **S0** | **Unfused production** — GEMM kernels + separate collective (RCCL AR on TP; MORI all-to-all on DP) | comm is its own kernel; no in-kernel cross-rank protocol | vLLM 0.25.1 + AITER/MORI (external); referenced as the `production` arm, `BENCHMARKING.md:198-206` | prefill boundary 7,712 µs = 1.000× (`../BRIEF.md:49`). TP8 boundary rig: GEMM→RCCL = **2,909 µs** (`distributed-kernels/tp8_mega/results/G25_1_STATUS.md:14-19`) |
| **S1** | **Homogeneous megakernel** — all 256 CTAs identical, phase-barrier separated (M0…M9) | one launch, grid barriers between phases, no role split | `k0pf6gm_device_tile.hip` (arm `pf6gm_mega`) | **6,908.8 µs = 0.8958×** (`../BRIEF.md:50`); harness sanity value 0.898 (`BENCHMARKING.md:198-201`) |
| **S2** | **Producer/consumer with a CARRIER pool (mode 2 family)** — reserved CTAs *move payload* | service CTAs consume an event queue and push slices to owners | `moe_mps_adapter.cuh` modes 1–9 + `k0pf6gm_device_tile_mps.hip`; design `DESIGN_MPS.md:37-75` | **falsified: +332…+340 µs**, never wins (`docs/distributed/OVERLAP_ABSTRACTIONS.md:28`). Dedicated comm CTAs at C=64: 6,866 µs (0.888×) with **+831 µs of added traffic attributable to the role itself** (`../BRIEF.md:52-53`) |
| **S3** | **Producer-carried transport, homogeneous roles (mode 12/13)** — the GEMM-2 epilogue itself issues the remote packed-bf16 accumulate | no pool job; per-row readiness protocol (~926k ops) | `n2_phase2_gm_mps.cpp:236-238` (`accumulate_peer_bf162` site); modes at `moe_mps_adapter.cuh:352-353` | unbounded **7,110.8 µs** (worse than S1); depth-4 bounded **6,483.8 µs = 0.8407×** (`../BRIEF.md:50,54-55`) |
| **S4** | **M15 — slab-certified producer/consumer with a CONSUMING pool** | nc-major producer order + 2 column slabs + one epoch word per (rank, slab) + C reserved CTAs that *consume* certified front-half combine in the producer's shadow | `k0pf6gm_device_tile_m15.hip`; contract `aug12/M15_DESIGN.md:20-42` | **C=16: 6,292.4; C=24: 5,848.5 (0.7589×); C=28: 5,822.0 (0.7544×); C=32 unstable (hung 1 of 2 runs)** (`../BRIEF.md:50-51`) — the shipped best |
| **S5** | **Cross-launch deferred combine (mode 16 / TBO-2)** — epoch *i*'s combine consumed inside launch *i+1* on epoch-parity buffers | launch boundary becomes a pipeline stage | `moe_mps_adapter.cuh:395` (`kModeDeferCombine`), gated by `K0P6_MPS_ENABLE_TBO`; design `aug12/exp_02_tbo_deferred_combine/design.md` | **falsified: 6,559.6 µs = 0.8513×, +75.8 µs vs the ratchet** (`docs/distributed/OVERLAP_ABSTRACTIONS.md:7,52-53,83`) |
| **S6** | **CDAR / counter-dataflow TP8 megakernel (M25)** — chunked all-reduce that never exists as a phase: ERS rides the producing GEMM's epilogue, MAG rides the consuming GEMM's ramp | certificate-gated layer transitions, one grid barrier per launch edge | design `aug18-prefill/M25_TP8_MEGA_DESIGN.md`; rig `distributed-kernels/tp8_mega/{m25_cdar.cuh,m25_boundary_bench.hip}` | **interim NEGATIVE at boundary level: fused CDAR 3,606 µs vs GEMM→RCCL 2,909 µs**; phased control 3,653 µs; compute floor 1,439 µs (`tp8_mega/results/G25_1_STATUS.md:14-19`; commit `3f7f0f3a`) |
| **S7** | **MoK-style counter dataflow, zero grid barriers (T3)** — every dependency is a flat HBM counter; producers get lower task indices than consumers | no residency contract; work-stealing ticket pool | design `aug18/T3_COUNTER_DATAFLOW_DESIGN.md`; partial vehicle `k0pf6gm_device_tile_t3.hip` | design only; projection band 1,280–1,330 ms training step from the skeleton swap (`aug18/T3_COUNTER_DATAFLOW_DESIGN.md:85-91`) — **explicitly a projection, not a measurement** |
| **S8** | **Staged wide-push transport (m15b, mode 15b)** — epilogue folds partials into a *local* stage; carrier CTAs push folded half-rows as posted 16 B packet stores | swaps the fabric op class from RMW to posted store, ~1.44× fewer remote bytes | `k0pf6gm_device_tile_m15.hip:63-70` (`K0P6_M15_STAGED`, default 0); design `aug12/M15_DESIGN.md:63-76` | **built, never validated.** Blocker measured: `STAGED=1` compiles to 96 `scratch_load_dword` + 97 `vmcnt(0)` within 24 instructions of the remote atomic vs 0/1 in the default build (`aug18-prefill/FP8_WIRE_DESIGN.md:24-29`) — the throttle is annihilated; P0 before any A/B |

### A.1 What separates the skeletons, stated as the campaign's own law

The three-way split of "CTA specialization" is the load-bearing structural finding
(`docs/distributed/OVERLAP_ABSTRACTIONS.md:26-32`):

| pool job | measured verdict | source |
|---|---|---|
| **carrying** payload | +332…+340 µs — never wins | `OVERLAP_ABSTRACTIONS.md:28` |
| **idling** (reserve-but-wait) | +8–9 µs per reserved CTA, monotone | `OVERLAP_ABSTRACTIONS.md:29` |
| **consuming** certified-ready output in the producer's shadow | −93 µs (C 8→16), −444 µs (C 16→24) | `OVERLAP_ABSTRACTIONS.md:30` |

**S2 vs S4 is therefore not a knob setting — it is the same reservation mechanism with
opposite economics**, and the enabling difference is the *certification* (K3) that
creates early consumable work plus the *order* (K2) that makes the certified prefix
large. This is the single most important structural statement for the cost model.

### A.2 Selection rule that constrains all skeletons (the occupancy-1 axiom)

Every kernel in this family runs 1 block/CU, 1 wave/SIMD with 256-VGPR GEMM bodies, so
there are **no spare waves to fill stalls** and `vmcnt` is one in-order counter per wave
(`docs/distributed/OVERLAP_ABSTRACTIONS.md:42-47`). Consequence: *phase partitioning is
work-conserving* — moving full-grid work "under" a compute phase by CTA partition just
re-divides the same CTA·µs. Overlap pays only through four doors
(`OVERLAP_ABSTRACTIONS.md:55-60`): (a) latency-bound consumer into a compute shadow;
(b) deleting protocol work; (c) op-class swaps on the fabric; (d) bounding injection.
**Any proposed schedule that goes through none of these doors is predicted ≈0** — this is
the falsifiable selection function the ablation campaign should test directly.

---

## PART B — KNOBS WITHIN A SKELETON

### B.0 The runtime knob surface: one packed 64-bit config word

The entire *runtime* schedule surface of the DP/EP megakernel is one descriptor word,
`K0P6_D_MPS_CFG` (descriptor slot 62), set from the env var `K0_MPS_CFG`
(`BENCHMARKING.md:178-183`). This is what makes a manifest-addressable knob space
possible **today**: one HSACO serves the whole sweep.

Bit layout (`moe_mps_adapter.cuh:164-171`), struct at `:183-190`, encode/decode at
`:192-216`, validator at `:583-673`:

```
[0:8)    C            reserved service CTAs        (validator: <=128 and <256, :591-592)
[8:16)   g   LOW      group_slices + mechanism bits
[16:24)  mode         skeleton/mechanism selector  (validator ceiling depends on build gates, :652-658)
[24:32)  flush_rows   pool quota / pacing / window (validator: 1..64, :670)
[32]     pull_fallback
[33]     timestamps_enable
[34:42)  g   HIGH     (exp_24 widening; bit-identical for g<=0xFF)
```

`g` sub-fields (`moe_mps_adapter.cuh:400-410`):

```
0x000F  physical g (push group size)   — must be 1 in direct-accumulate modes (:596)
0x0010  dual-write lost-update detector
0x0020  epilogue remote-RMW throttle ENABLE
0x0040  skip the DEAD `part` zero-fill in M5
0x0080  mode-14 KEEP-drain confound control (mode 14 only, :637-640)
0x0300  throttle DEPTH select: 00->8, 01->4, 10->16, 11->32
0xFC00  reserved (rejected)
```

Grammar example in the shipped ratchet config: `K0_MPS_CFG="C=16,g=353,mode=12,flush_rows=16"`
(`aug11/exp_33_attribution/result.md:5`). `g=353 = 0x161` = physical g 1 + throttle enable
(0x20) + depth-select 01 (0x100) + skip-part-zero (0x40) → **depth 4**.

---

### B.1 K1 — CARRIER: who moves the bytes

| field | value |
|---|---|
| **name** | `K1_carrier` / transport op class |
| **domain** | {producer epilogue remote packed-bf16 RMW (mode 12/13) · staged pool push (mode 2) · staged wide posted-store push (m15b) · owner-gathered store towers (CDAR) · consumer PULL (`load_peer_packets`) · SDMA engine} |
| **implemented** | epilogue RMW: `n2_phase2_gm_mps.cpp:236-238`; primitive form `include/cdna4/ops/group/distributed/credit.cuh:96-113` (`throttled_accumulate_bf162` / `throttled_store_packet16`); CDAR enum `distributed-kernels/tp8_mega/m25_cdar.cuh:72-75` (`atomic_accumulate` / `store_towers`); staged push `k0pf6gm_device_tile_m15.hip:2285` (M7.7) gated by `K0P6_M15_STAGED` at `:68-70` |
| **compile vs runtime** | **mixed.** RMW-vs-pool-push is the runtime `mode`; RMW-vs-staged-local-fold is **compile-time** (`-DK0P6_M15_STAGED=1`); CDAR transport is a **compile-time template argument** (`m25_cdar.cuh:158`) |
| **default** | epilogue remote RMW (mode 12) in M15/M24 serving; `store_towers` in the CDAR rig |
| **measured sensitivity** | pool-carried **+340 µs** vs epilogue-carried (`OVERLAP_ABSTRACTIONS.md:79`). Packed bf16 remote atomics **52.8 GB/s coalesced** vs remote stores **54.9 GB/s**; scattered lane addresses collapse to **4.1 GB/s** (`../BRIEF.md:57-58`) — a 13× cliff. In the CDAR rig at 16 K tokens / 512-row slabs / depth 4 / bps 2: atomics **67.1 GB/s** vs store-towers **93.6 GB/s** (`tp8_mega/results/g25_0b_v1_results.txt:9,64`); at 256-row slabs towers reach **117.2 GB/s** (`results/g25_0b_slab_sweep.txt`). Commit `108def6e`: "K1=store-towers … 4.2× v0 … atomic caps 67" |
| **interacts with** | **ARCH** (decisive: gfx942 remote atomics ~3× slower than stores, 15 vs 44–46 GiB/s → epilogue-accumulate transport is *likely wrong there* — `OVERLAP_ABSTRACTIONS.md:146`, `credit.cuh:40-45`), **T** (bytes per boundary), **TOPO** (fan-in shape), **SKEW** (owner hotspot concentration on the atomic path) |

**Layout is part of this knob, not separate**: the 13× coalesced/scattered cliff means
"the tile→packet mapping must be audited in ISA, not assumed"
(`M25_TP8_MEGA_DESIGN.md:150-153`). The BRIEF states it as a law: **layout is a schedule
decision** (`../BRIEF.md:58`).

---

### B.2 K2 — PRODUCER ORDER: which output prefix completes first

| field | value |
|---|---|
| **name** | `K2_order` (task→(tile, chunk/owner) index map) |
| **domain** | {`producer_major` (tile-major, donor default) · `consumer_major` (nc-major, M15) · `owner_major_staggered(rank)` (ring inducer) · `source_interleaved` (m17 coprime-stride RR)} |
| **implemented** | primitive: `include/cdna4/ops/group/distributed/order.cuh:50-95` (all four, branch-free one-expression maps). Live site: `k0pf6gm_device_tile_m15.hip:880-883` — `N2GM_TASK_DECODE` defines nc-major (`nc = t/ntiles; tile = t - nc*ntiles`); consumed at `n2_phase2_gm_mps.cpp:479-483`. m17 RR scatter behind `K0P6_M15_SCATTER_RR` (`k0pf6gm_device_tile_m15.hip:81-83`, default 0) |
| **compile vs runtime** | **compile-time** (a macro-substituted decode inside the hot task loop). Making it runtime is an explicit non-goal for codegen reasons: order lives at peak register pressure (`order.cuh:15-18`) |
| **default** | `consumer_major` (nc-major) in M15; `producer_major` in the donor |
| **measured sensitivity** | **order alone ≈ 0**; **order × certification = −144 µs of combine residue** (`OVERLAP_ABSTRACTIONS.md:80`). The mediating measurement: under natural GEMM-2 task order the median token is not reducible until **91.7% of GEMM-2 completes** (`../BRIEF.md:59-61`); `order.cuh:21-24` records the same law as "no consumable row before 93.9% completion" under tile-major |
| **interacts with** | **SKEW** (m17's destination rotation is the zero-protocol alternative to per-destination credits — `OVERLAP_ABSTRACTIONS.md:104-106`), **TOPO** (owner-major-staggered is what makes ring reduce-scatter *emerge*: `OVERLAP_ABSTRACTIONS.md:154-160`), **T** (number of chunks per tile sets the staircase resolution), **ARCH** (nc→XCD L2 residency invariant, `aug12/OVERLAP_KERNEL_DESIGN_IDEAS.md:122`) |

**Law candidate (already stated by the BRIEF, `:61-62`):** *signal the smallest
early-EXECUTABLE unit, not the smallest stored/transferred unit.* K2 is the knob that
makes "early-executable" exist.

**Sub-knob (unbuilt): rotated-nc phase offset (K4-prime).** `nc' = (nc + gid·φ) mod 16`
to de-align epilogue bursts across CTAs; `aug12/OVERLAP_KERNEL_DESIGN_IDEAS.md:126`
prices de-alignment's ceiling at +211 µs from exp_25. Not implemented.

---

### B.3 K3 — CERTIFICATION GRANULARITY: when consumers may start

| field | value |
|---|---|
| **name** | `K3_certification` — slab count S / slab extent / signal class |
| **domain** | {per-row flags (~926k protocol ops) · per-(rank, slab) epoch words (2×8) · one phase barrier (S=1) · per-(owner, slab) CDAR cells} · slab extent in rows |
| **implemented** | M15: `K0P6_M15_SLABS 2`, `K0P6_M15_NC_PER_SLAB 8`, `K0P6_M15_M8_CSPLIT 7` at `k0pf6gm_device_tile_m15.hip:498-500`; the slab loop + rendezvous at `:2067` and `:2235-2260`. Primitive: `include/cdna4/ops/group/distributed/slab.cuh:62-97` (`certify_slab`, `certify_slab_to`, `bounded_wait_slab_into`). CDAR: `slab_rows` is a **runtime** field of `cdar::geometry` (`tp8_mega/m25_cdar.cuh:79-101`) and argv position 3 of the rig (`m25_boundary_bench.hip:37-38`) |
| **compile vs runtime** | **compile-time in M15** (S=2 is a `#define`; the comment at `:496-500` says S=2 is the *only* value where 448-column nc chunks and M8's 1,024-byte chunks co-align at column 3,584, and is also the fence-law optimum). **Runtime in the CDAR rig.** |
| **default** | M15: S=2, split at column 3,584. CDAR rig: `slab_rows = 256` argv default; `cdar::geometry` doc says "prefill default 512; decode mode shrinks it" (`m25_cdar.cuh:82`) |
| **measured sensitivity** | Per-row → slab words replaced ~926,000 protocol ops with 2×8 certificates: "the mode-14 economics *without* mode-14's loss" (`OVERLAP_ABSTRACTIONS.md:81`, `slab.cuh:6-9`). **Granularity is the free axis; EXPOSURE is the deadly one** — a 64× signal-count change measured ~nothing (`slab.cuh:10-12`). BUT in the CDAR rig granularity is *not* free: 16 K tokens, towers, depth 4, bps 2 → **128 rows 116.0 · 256 rows 117.2 · 512 rows 93.4 · 1024 rows 55.4 GB/s** (`tp8_mega/results/g25_0b_slab_sweep.txt`), a **2.1× spread** |
| **interacts with** | **T** (slab count = tokens/slab_rows; at small T the chunking collapses — `M25_TP8_MEGA_DESIGN.md:212-215` says "chunking collapses below ~2K tokens"), **PHASE** (decode wants smaller slabs), **TOPO** (in reduce-scatter the owner row-block boundary is the natural cut and needs no re-chunking, `slab.cuh:26-30`), **CONC** (via T) |

**Open contradiction worth an experiment cell:** M15/`slab.cuh` says granularity is free;
the CDAR slab sweep says it is worth 2.1×. The reconciling hypothesis is that in M15 the
signal count changed while the *transfer* geometry did not, whereas in CDAR `slab_rows`
also changes the store-tower geometry and the gather width. **SPECULATION** until the
G25 device phase ledger runs.

---

### B.4 K4 — FLOW CONTROL: in-flight bound on fabric ops

| field | value |
|---|---|
| **name** | `K4_depth` (injection depth) + `throttle_enable` |
| **domain** | enable ∈ {0,1}; depth ∈ **{8 (selector 00, the historical default), 4 (01), 16 (10), 32 (11)}**; plus {none} and, unbuilt, {per-destination credits} |
| **implemented** | selector bits `0x0300` of `g` (`moe_mps_adapter.cuh:400-410`, accessor `throttle_depth_sel` at `:488-491`, enable at `:481-484`). Device: `n2_phase2_gm_mps.cpp:116-127` declares the four `s_waitcnt vmcnt(N)` specializations, `:170-176` instantiates 8/4/16/32, `:405-408` builds the SGPR "throttle plan" via `readfirstlane`. Primitive: `credit.cuh:63-82` ships depths {0,4,8,16,32} |
| **compile vs runtime** | **both, deliberately.** `vmcnt(N)` encodes N in the instruction's simm16 — *there is no register form* — so depth is a compile-time literal; runtime selection is one textual loop copy plus mutually-exclusive SGPR lane masks. Templating the loop on Depth was measured to spill one 4-byte value with **389 reloads (scratch_load 9 → 436)**, contaminating the control arm (`credit.cuh:12-22`) |
| **default** | enabled, depth 4 (`g=353`); `credit.cuh:59-61` names depth 4 "the fused-MoE campaign's shipped value and the measured optimum on gfx950 xGMI for remote pk_add epilogues" |
| **measured sensitivity** | **The single largest knob in the DP campaign: unbounded 7,110.8 µs → depth-4 6,495.8 µs = −615.0 µs** (`credit.cuh:5-9`; `../BRIEF.md:54-55`; `OVERLAP_ABSTRACTIONS.md:82`). Unbounded is +211 µs vs *no payload at all* (`OVERLAP_ABSTRACTIONS.md:82`). **AND: depth is FLAT {4,8,16,32} in pure transport** — CDAR towers 93.6/93.4/93.3 GB/s at depths 4/16/32; atomics 67.1/67.1/67.0 at bps 2 (`tp8_mega/results/g25_0b_v1_results.txt:64,69,74 and :24,39,54`). Commit `108def6e`: "depth flat in pure transport (K4 co-residency-dependent)" |
| **interacts with** | **ARCH** (op class and link ceiling), **TOPO** (fan-in: `aug12/OVERLAP_KERNEL_DESIGN_ADDENDUM.md:295-300` records effective depth capped by `min(nominal, fan-in × tokens interleaved)`, fan-in ≈5.3 at this route family; depth constants for 14 KiB-row pushes are 8–16 with turnover at 32), **SKEW** (a *global* per-wave bound cannot distinguish destinations — the per-destination-credit gap, below), **T** (bytes in flight) |

**This is the campaign's cleanest cost-model law already in hand:** *depth binds only
when the fabric shares the device with compute.* In pure transport it is inert; in the
co-resident megakernel it is worth −615 µs. The mediating quantity is co-residency, not
bytes. That is exactly the "if X, then Y, because Q crossed θ" shape the BRIEF demands.

**Fragility contract that must ride any depth arm** (`credit.cuh:30-38`): a bound that
can silently die is not a primitive — exp_38 saw register pressure re-throttle an
epilogue with **no source change**. The ISA gate is (a) literal `vmcnt(N)` at expected
sites, (b) zero scratch ops within 24 instructions of any remote atomic, (c) issue-run
distribution matching the plan (**282 atomics / ~12 runs / mean ~23.5**).

**Unbuilt sub-knob: per-destination credits (K6).** `credit[owner]` cells refilled by the
owner's consumption posts; the skew-rescue arm. Documented at
`aug12/OVERLAP_KERNEL_DESIGN_IDEAS.md:130` with a predetermined falsifier ("must tie
mode-12 within 1% at std=0"). `credit.cuh:104-106` marks per-destination credits a
*documented non-goal until a skew harness exists*, because m17's order-based rotation
buys the spreading with no protocol.

---

### B.5 K5 — CONSUMER PLACEMENT: who consumes, when — the `C` knob

| field | value |
|---|---|
| **name** | `C` = `reserved_comm_ctas` (misnamed by history; in M15 it is the **consuming** pool) |
| **domain** | integer 0…128 (`config_is_valid`: `>128` rejected, `>=256` rejected — `moe_mps_adapter.cuh:591-592`); `BENCHMARKING.md:192-194` records the earlier documented cap `C ≤ 64`. Mode 3 additionally requires C be a multiple of 32 (whole XCDs) → legal {32, 64} (`moe_mps_adapter.cuh:667-668`, `:220-232`). Mode 1 requires `C == 0` (`:668-669`); stream modes and diagnostic-pool modes require `C != 0` (`:661-663`) |
| **implemented** | role predicate `is_service_cta` (`moe_mps_adapter.cuh:673-682`); dense compute id `compute_id_of` (`:684-691`); kernel-side task start/stride derived from C at `k0pf6gm_device_tile_m15.hip:793-802`; the pool's actual job (front-half combine sweep during slab 1) at `:2102-2160` |
| **compile vs runtime** | **runtime** (config word byte 0) |
| **default** | M15 design says start at `C ∈ {8,16}` (`aug12/M15_DESIGN.md:86-88`); the ratchet config used `C=16` (`aug11/exp_33_attribution/result.md:5`); the measured best in the draft8 ladder is **28** |
| **measured sensitivity** | **The steepest, most non-monotonic knob in the inventory.** M15: 16 → 6,292.4; 24 → 5,848.5 (0.7589×); 28 → **5,822.0 (0.7544×)**; 32 → *unstable, hung 1 of 2 runs* (`../BRIEF.md:50-51`). Deltas: −93 µs (8→16), −444 µs (16→24) (`OVERLAP_ABSTRACTIONS.md:30`). Under mode 12 the *same* knob had optimum C ≤ 8; under M15 it is C=24+ (`OVERLAP_ABSTRACTIONS.md:120-123`). Idle-pool tax is +8–9 µs/CTA monotone (`:29`) |
| **interacts with** | **T** (M24 note in the kernel itself: at `T_eff` the pool's quota is proportionally smaller, "so C / flush_rows are no longer at their tuned point — arm E re-sweeps them at runtime", `k0pf6gm_device_tile_m15.hip:2122-2127`), **FILL** (same site), **CONC** (via T), **PHASE**, **SKEW** (the consumable prefix arrives late on a hot rank) |

**The primitive ships the method, not a default** (`OVERLAP_ABSTRACTIONS.md:117-123`):
"the C-response is steep and non-monotonic *across designs* … the primitive ships with
the *response-curve method*: sweep C with same-session controls, judge on the coupled
producer+consumer sum." For the generalization study this means **C must be re-swept
inside every W-cell**, and its optimum is a *derived* quantity of the cost model, not an
input.

---

### B.6 Pool quota — `flush_rows` (the knob that makes K5 safe)

| field | value |
|---|---|
| **name** | `flush_rows` (overloaded: quota, pacing, poll backoff, diagnostic window) |
| **domain** | 1…64 (`config_is_valid`, `moe_mps_adapter.cuh:670`); M15 design suggests sweeping {0, 8, 16, 32} where **0 disables the mid-M7 sweep and isolates pure protocol deletion** (`aug12/M15_DESIGN.md:87-88`) |
| **implemented** | accessor `effective_flush_rows` (`moe_mps_adapter.cuh:567-569`; forced to 16 in modes 4 and 8 so pacing/backoff is the only variable); RMW-arm quota at `k0pf6gm_device_tile_m15.hip:2149-2152`; staged-arm push quota (`8 × flush_rows` quanta) at `:2196-2200` |
| **compile vs runtime** | **runtime** (config byte 3) |
| **default** | 16 (`BENCHMARKING.md:194`; kernel comment at `:2143-2148`) |
| **measured sensitivity** | no isolated Δ found in-repo. Its *design role* is quantified: the pool covers only ~20% of the front half per exp_29's **6.1 GB/s/CTA** law, and "draining all 1,024 batches would outlast slab 1 several times over" (`k0pf6gm_device_tile_m15.hip:2142-2148`) |
| **interacts with** | **T** and **FILL** (the quota is sized to slab-1's compute shadow at nbatches=1,024, i.e. T=4096 — explicitly de-tuned at other T, `:2122-2127`), **CONC**, **SKEW** |

**Overloading note for the manifest:** in mode 4 `flush_rows` means pacing units, in
mode 8 poll-backoff units, in modes 5/6 a diagnostic window in 100 µs units
(`moe_mps_adapter.cuh:571-581`, `:254-259`). Any manifest row must carry `mode` to
interpret `flush_rows` — a naked value is meaningless.

---

### B.7 Grid geometry and role placement

| knob | domain | where | c/r | default | sensitivity (source) | axes |
|---|---|---|---|---|---|---|
| **grid size / occupancy** | 256 CTAs, `__launch_bounds__(256,1)`, 1 block/CU | `M25_TP8_MEGA_DESIGN.md:120-121`; the occupancy-1 axiom `OVERLAP_ABSTRACTIONS.md:42-47` | compile | 256, occ 1 | not swept; it is the axiom the whole selection rule rests on | ARCH, TOPO |
| **pool XCD placement** (mode 3) | tail-reservation vs whole-die residue classes; mode 3 requires C a multiple of 32 | `moe_mps_adapter.cuh:218-232` (rationale), `:673-682` (`is_service_cta` branch) | runtime (mode) | tail (modes 0–2) | **XCD-confined placement 7–23% worse** (`OVERLAP_ABSTRACTIONS.md:149`); exp_05 measured a running service pool inflating M7 by 37–57% purely through memory-system interference, mechanism = atomic contention not cache footprint (`moe_mps_adapter.cuh:219-226`) | ARCH, TOPO |
| **role-assignment mechanism** | static tail split vs `finish_order_partition` | `roles.cuh` (ships); M25 keeps the static split — "finish_order_partition stays rolled back — 24 B/lane spill precedent" (`M25_TP8_MEGA_DESIGN.md:122-123`) | compile | static tail | qualitative (spill precedent) | ARCH |
| **prefetch CTAs** `K0P6_M20_PF_CTAS` | count of pool CTAs streaming the *next-next* layer's replica weights | `k0pf6gm_device_tile_m15.hip:432`, used at `:2091-2100` | compile | 8 | not isolated | T, SKEW, TOPO |
| **planner CTA** | 1 dedicated rolling-planner CTA (M25) | `M25_TP8_MEGA_DESIGN.md:126-131` | design | 1 | unmeasured (**SPECULATION** as a win) | T, SKEW |

---

### B.8 Combine / consumer-side granularity

| knob | domain | where | c/r | default | sensitivity | axes |
|---|---|---|---|---|---|---|
| **M8 combine batch size `NT`** | rows per dynamic wave ticket | `k0pf6gm_device_tile_m15.hip:2440` (`constexpr int NT = 4;`), also `:2140` in the pool sweep | compile | 4 | not isolated in repo | T, FILL, PHASE |
| **M8 front/back split** `K0P6_M15_M8_CSPLIT` | chunk index where the front half ends | `:500` | compile | 7 (column 3,584) | co-determined with S=2 by the alignment argument at `:496-500` | — |
| **CDAR fragments-per-slab `bps`** | item granularity inside a slab | `m25_boundary_bench.hip:448` (argv 7), used at `:185-190` | runtime (rig) | 2 | atomics at 512-row slabs: **bps 1 → 41.0, bps 2 → 67.1, bps 4 → 66.8 GB/s** (`results/g25_0b_v1_results.txt:4,9,14`). In the *fused* schedule bps=16 made it **WORSE (+180 µs of protocol) with still no hiding** (`results/G25_1_STATUS.md:31-34`) | T, PHASE |
| **push/pull direction** | push (producer-defined, many-to-one) vs pull (consumer-defined, one-to-many) | CDAR MAG uses pull (`M25_TP8_MEGA_DESIGN.md:116`); m15b/ERS push | compile | per-boundary | matched 64 KiB producer→transport→consumer wall time **0.9936× (tie)**; 64 MiB single-link BW pull/push **1.078×** (`../BRIEF.md:63-65`). Message-size knees: pair knee 256 KiB; fan-out knee 64 KiB/peer (push) vs 1 MiB (pull); **pull collapses past 256 CTAs while push degrades gently** (`aug12/OVERLAP_KERNEL_DESIGN_ADDENDUM.md:317-321`) | T, TOPO, CONC |

**Law already stated** (`../BRIEF.md:64-65`): *push when progress is producer-defined and
many-to-one; pull when consumer-defined and one-to-many.* Both can approach link
saturation, so direction is a **latency/ordering** choice, not a bandwidth one.

---

### B.9 Fill / shape awareness (M24)

| knob | domain | where | c/r | default | sensitivity | axes |
|---|---|---|---|---|---|---|
| `K0P6_M24_FILL` | master switch: re-derive T from a per-layer `n_orig` vector so row-proportional cost scales with real tokens | `k0pf6gm_device_tile_m15.hip:182-186`; five T-substitution sites named in-comment (KERNEL 943/977/1375/1587/1891) | compile | 0 | mechanism motivated by the measured 37.6% fill / 2.66× padding multiplier at C=32 (`:172-181`; `../BRIEF.md:44`) | **FILL**, T, CONC |
| `K0P6_M24_TGRAIN` | rounding granularity of `T_eff` | `:189-190` | compile | **256** | conservative because the read-only `csr_scan_block256` / `pull_src_fill` headers "have only ever been exercised at T ∈ {4096, 2048, 1024}"; lowering to 8 is gated on work-list G7 T-generality (`:186-193`) | T |
| `K0P6_M24_ZERO_PAD` | zero `out[T_eff, T_cap)` | `:197-199` | compile | 0, but **mandatory** with FILL (`#error` at `:279-281`) | correctness, not performance | FILL |
| `K0P6_M24_NULLWORK` | receive-side null-work skip on ground truth | `:215-217` | compile | 0 | **demoted to diagnostic**: a DP-dummy rank still owns 32 experts under EP8 and still receives from every busy peer, so the predicate does not fire in serving (`:205-214`) | FILL, TOPO |
| `K0P6_M24_NORIG_CONST` / `_TABLE` | MoK-harness payload-value substitution; `_TABLE` is a per-rank brace list (heterogeneous fill, arm D) | `:228-229` and `:239-243` | compile | off (TABLE has *no* default — "unset" is its off state) | the instrument for a **synthetic fill/skew sweep** without serving | FILL, SKEW |
| `K0P6_M24_STRICT` | run-voiding telemetry bit on payload reject | `:256-257` | compile | 0 | telemetry only, deliberately never gates a device return (`:230-255`) | — |
| `K0P6_M24_DENSE_SCATTER`, `_FILL_C`, `_M8_ADAPT` | declared-but-unimplemented arms (T2-c/d/e) | `:262-270`, hard `#error`s at `:294-302` | compile | 0 | **`_FILL_C` is literally "runtime-adaptive `reserved_comm_ctas`"** — the C-vs-T coupling as a knob, not yet built | T, FILL |

---

### B.10 Routing / placement actuators (the SKEW axis)

| knob | domain | where | c/r | default | sensitivity | axes |
|---|---|---|---|---|---|---|
| `K0P6_M15_REPLICATE` (M18) | static hot-expert replication; `E owned + nrep replicas ≤ K0P6_MAXE (64)`; replicated experts route **source-locally** | `k0pf6gm_device_tile_m15.hip:99-101` (+ rationale `:85-98`) | compile | 0 | motivating measurement: **51.9% of expert-slot traffic to eight experts sharing one rank's block**, aggregate receive-side load **5.09× fair share**, worst layer **5.59×**; both this kernel and production degrade **~3.2–3.6×** under replay of that histogram (`:85-92`; `../BRIEF.md:70-72`) | **SKEW**, TOPO |
| `K0P6_M15_ADAPTIVE` (M19) | per-chunk replication decision; threshold **theta = M18R header word 3** | `:120-124` (+ `:105-119`) | compile (theta runtime) | 0 | per-call static-set coverage is bimodal (**p5 = 1%, p75+ = 77%**), per-call max rank load **6.25× at p95**; serving pair #1 measured **−5.7% end-to-end** against a 4× aggregate-replay kernel win (`:106-113`) | **SKEW**, T |
| `K0P6_M20_SLOTPOOL` | replica weights in a triple-buffered cross-layer pool (~1 GB at B=8) instead of 41 GB persistent; decision precomputed in slot 65 | `:147-149` (+ `:126-146`) | compile | 0 | deleting the M0.5 pre-pass and its two grid-barrier rendezvous is priced at **+1.2 ms measured** (`:143-145`) | SKEW, T |
| `K0P6_M15_SCATTER_RR` (m17) | source-rotating scatter permutation so a depth-4 window spans ~7 links instead of 1 | `:81-83` (+ `:71-80`) | compile | 0 | mechanism argument only; the primitive form is `order.cuh:92-95` | SKEW, TOPO |
| **EPLB placement** (`num_redundant_experts=0`) | expert→physical-slot map, resolved *inside the router* before the kernel | `aug18-prefill/M15_EPLB0_COMPOSE.md:1-10` | runtime (host) | off | compose is "sound, and cheaper than the brief assumed"; **not** zero-memory: ≈1.31 GiB/rank transfer buffer | **SKEW**, TOPO |
| **route capture / replay** | the instrument that turns SKEW into a controlled axis | `aug18-prefill/route_capture/`, `ROUTE_REPLAY_PLAN.md` | host | — | produced the 51.9% finding (`../BRIEF.md:70-72`) | SKEW |

---

### B.11 Geometry / extent knobs (the T axis)

| knob | domain | where | c/r | default | sensitivity | axes |
|---|---|---|---|---|---|---|
| **`MAXTOK`** (compact-MAXTOK, arm A5) | per-rank slot capacity; **must be a power of two** in mode 12 (shift/mask in the epilogue) | descriptor `K0P6_D_MAXTOK` (`k0pf6gm_device_tile_m15.hip:392`); design `aug12/OVERLAP_KERNEL_DESIGN_ADDENDUM.md:268-285` | runtime (host-set) | over-provisioned by TOPK | **the ancestor corpus' only confirmed production win on this axis: `MAXTOK = T` measured −1.49% (~148 µs) at prefill** (`ADDENDUM:277-279`). Every per-epoch sweep scales with `T_ext = world·MAXTOK`, not with T | **T**, PHASE, FILL |
| `K0P6GM_G` | GEMM K-grouping | `:502-506` — **`#error` if != 3**, "pinned to the measured configuration" | compile | 3 | exp_28 named G=4 "the last free point": intensity 161.7→204.8, predicted M6 ≈ 1,957 µs, ~480/512 registers (`aug12/OVERLAP_KERNEL_DESIGN_IDEAS.md:192`) — **unmeasured** | ARCH, T |
| `spin_limit` | fail-closed bound on every device poll | kernel arg `(desc, pperr, spin_limit)` (`aug12/M15_DESIGN.md:7-8`) | runtime | serving value cited as `M15_SPIN_LIMIT = 20_000_000` (`aug18-prefill/M23_RAGGED_SEAL_DESIGN.md`, §7.1 risk row) | it is the **failure boundary** for cross-rank arrival skew on ragged batches | SKEW, CONC, T |
| `K0P6_MPS_ASCALE_TM` | M6 reads token-major activation scales (M5's transpose dead) | `:327-329` | compile | 1 | shipped arm of exp_27 | — |
| slab entry guard | `world·MAXTOK + K0P6_M15_SLABS ≤ T_loc_max` | `:1254` | runtime check | 32,770 ≤ 40,960 today | binds any MAXTOK/S change | T |

---

### B.12 Build-discipline "knobs" that are really instrument controls

These change the *measurement*, not the schedule, and every manifest row must pin them.

| control | where | why it is load-bearing |
|---|---|---|
| `K0P6_MPS_ENABLE_MODE14` | `moe_mps_adapter.cuh:57-58` | merely making mode-14 code **reachable** cost the mode-12 ratchet **+726.9 µs** with mode 12's own source untouched (`:39-56`). *"Never publish a mode-12 number from a MODE14=1 binary."* |
| `K0P6_MPS_ENABLE_TBO` | `moe_mps_adapter.cuh:61-69` | same discipline, same measured reason |
| `K0P6_M15_SRC_REV` | `k0pf6gm_device_tile_m15.hip:63`, bumped to 3 under M24 at `:314-317` | JIT-cache guard; a stale `.hsaco` silently measures the wrong body |
| `K0P6_M15_KERNEL_NAME` | `:491-493` | symbol pin so a RUN PIN branch compiles under the harness's expected name |
| `timestamps` config bit | `moe_mps_adapter.cuh:170` | phase stamps resolve ~1% vs ~6.6% for screens (`aug12/OVERLAP_KERNEL_DESIGN_IDEAS.md:198`) |
| descriptor slot exclusivity | `#error`s at `:103-105`, `:159-165`, `:279-312` | slot 63 is claimed by STAGED / REPLICATE / SAVE_Z — three arms cannot compose |

**The general law:** *the instrument is part of the kernel* (`OVERLAP_ABSTRACTIONS.md:198`).
A knob-space manifest that does not pin these is not reproducible.

---

## PART C — FUSION-BOUNDARY CHOICES (which ops live inside the mega)

### C.1 Today's boundary (DP8/EP8 serving megakernel, M15)

**Inside** one launch (`k0pf6gm_device_tile_m15.hip` phase markers):

| phase | line | op |
|---|---|---|
| M0 | `:1319` | retire wait + per-block counter zero |
| M0.5 | `:1390` / `:1402` | replication decision (M20 precomputed / M19 pre-pass) — *only under those arms* |
| M1 | `:1448` | qpush **dispatch** (the all-to-all push) |
| M2 | `:1658` | chunk-acquire unpack + histogram |
| M3–M5 | `:1891` | plan + destination-counting scatter |
| M6 | `:2024` | expert GEMM-1 (gate/up) + SiLU |
| M7 | `:2055` | expert GEMM-2 (down) + **combine transport in the epilogue** + slab certification |
| M7.7 | `:2285` | staged-arm remaining pushes (STAGED only) |
| M8 | `:2410` | combine reduce (front then back) |
| M9 | `:2524` | counted arrival, resets, retirement pokes |

**Outside** today: router / top-k gate, attention, RMSNorms, the shared expert, weight
quantization, and (on the TP path) the two per-layer all-reduces.

### C.2 The fusion-boundary knob table

| id | boundary move | vehicle | c/r | status | evidence / price |
|---|---|---|---|---|---|
| **F1** | **collective inside vs outside** — RCCL AR vs in-epilogue CDAR | M25 (`M25_TP8_MEGA_DESIGN.md`) | build | **interim negative** | fused 3,606 vs GEMM→RCCL 2,909 µs; RCCL AR is 1.5× our transport (`tp8_mega/results/G25_1_STATUS.md:14-19`, commit `3f7f0f3a`). Amdahl prize pool: native TP8 closes at ~2.8 ms/layer exposed AR (`../BRIEF.md:86`) |
| **F2** | **shared expert inside** (as bubble filler) | `K0P6_M15_SHEXP`, **default 0** | compile | design only | the shared expert is the *only* same-epoch work with zero dependency on dispatch/plan/sort/peers (`aug18-prefill/SHARED_EXPERT_FILLER_DESIGN.md:17-30`); today vLLM runs it serialized *before* the routed region |
| **F3** | **RMSNorm inside, owner-sharded** | M25 step (b)/(g) | design | unbuilt | "8× less norm work + one less full HBM pass per boundary; small but free" (`M25_TP8_MEGA_DESIGN.md:161`) |
| **F4** | **router/gate GEMM inside** (rolling planner appends tiles as slabs certify) | M25 step (c) + planner CTA | design | unbuilt | production serializes gate→sort→group kernels; sized by "vLLM kernel-gap time in B0 profile" (`M25_TP8_MEGA_DESIGN.md:164`) |
| **F5** | **attention inside** | M25 §8 v2 doors: (a) in-kernel MLA, (b) chunked attention on a hipGraph with per-slab events, (c) persistent-attention co-kernel | design | **v1 deliberately leaves it out** | the exposed MAG#2 tail is v1's structural bubble, ~1/n_slabs + launch gap (`M25_TP8_MEGA_DESIGN.md:203-212`) |
| **F6** | **quantization on the wire** (fp8 combine + source-side pre-reduce) | `FP8_WIRE_DESIGN.md` | design | **blocked** | *not reachable from the mode-12 RMW epilogue at all* — gfx950 has no fp8 remote RMW; reachable only on top of the staged arm S8, whose scratch pathology is P0 (`FP8_WIRE_DESIGN.md:12-29`) |
| **F7** | **cross-layer weight prefetch inside** | M20 slot pool, prefetch CTAs in the M6..M8 window | compile (`K0P6_M20_SLOTPOOL`) | built, gated | stream order **is** the certification; no flags, no tags (`k0pf6gm_device_tile_m15.hip:134-142`) |
| **F8** | **backward / wgrad inside** | T2B (`K0P6_M15_SAVE_Z`, `:157-158`), T3 wgrad-as-tickets | compile / design | forward-save built; T3 design | T3 item 3: wgrad = ticket tasks with per-K-slab operand waits, replacing both the M8.5 phase and the windows (`aug18/T3_COUNTER_DATAFLOW_DESIGN.md:69-72`) |
| **F9** | **launch boundary as a pipeline stage** (defer combine to next launch) | mode 16 | runtime (TBO build) | **falsified** | +75.8 µs; "the wait it escaped was mostly not on the critical path, and the parity state it added was" (`OVERLAP_ABSTRACTIONS.md:52-53`) |
| **F10** | **graph-capture / seal boundary** — which serving steps the mega is even allowed to run | M23 ragged seal (vLLM-side; **zero lines of HIP**) | host | design ready | coverage **2% → 100% of in-bucket steps = 0.77% → 38.3% of all steps, ~50× more sealed steps** (`M23_RAGGED_SEAL_DESIGN.md`, §0 verdict table). Also: on an unsealed in-bucket step a PF4H-target server falls back to **no graph at all**, not to the stock B4096 graph — which changes how every past A/B must be read |

### C.3 Why the fusion boundary is a *schedule* knob and not a packaging choice

Two measured reasons:

1. **It changes what can be certified early.** Fusing the collective is what allows K3
   certificates to exist at all; unfused, the boundary is a phase and K3 degenerates to
   S=1 (a phase barrier).
2. **It changes the residency contract.** T3's diagnosis: the all-256-CTA grid barrier
   creates a residency contract that made every separate wgrad kernel standoff or
   serialize, and realizes cross-rank drift at three points per launch instead of one per
   layer (`aug18/T3_COUNTER_DATAFLOW_DESIGN.md:43-49`). Fusion boundary → barrier count →
   drift realization count.

---

## PART D — THE 10 HIGHEST-LEVERAGE KNOBS FOR A WORKLOAD-GENERALIZATION STUDY

Selection criteria, in order: (i) a measured effect already exists, so the study extends
a line rather than starting one; (ii) the effect's **sign or magnitude is already known
to move with a workload axis**, which is what makes it a *law* rather than a tuning
constant; (iii) it is cheap to sweep (runtime, or one `-D`), so a manifest grid is
affordable on one node.

| # | knob | why it is highest leverage | the law it can establish |
|---|---|---|---|
| **1** | **`C` — consuming-pool size** (B.5) | Largest measured spread of any single knob within a skeleton (6,292.4 → 5,822.0 µs, and unstable at 32), and its optimum **already flipped across designs** (mode 12: C ≤ 8; M15: C = 24–28). The kernel itself documents that its tuned point is invalid at other `T_eff` (`:2122-2127`). | *C\* is set by the ratio of latency-bound consumer work to producer shadow; it scales with the certified-prefix arrival time, not with CTA count.* Sweep across **T × FILL × SKEW**. |
| **2** | **K4 injection depth** (B.4) | Biggest single delta in the DP campaign (**−615 µs**) and **provably inert in pure transport** (flat 4→32 in the CDAR rig). That contrast is the cleanest mediating-quantity story the campaign owns. | *Depth binds iff the fabric is co-resident with compute; threshold θ is contention, not bytes.* Sweep depth × co-residency (transport-only vs fused) × **ARCH**. |
| **3** | **K3 certification granularity / slab extent** (B.3) | Two repo results disagree (free in M15, 2.1× in CDAR). Resolving that disagreement *is* a cost-model contribution, and it is a runtime knob in the CDAR rig. | *Granularity is free when it changes only signal count; it is expensive when it also changes transfer geometry.* Sweep `slab_rows` × **T** × **PHASE** (decode explicitly wants smaller slabs). |
| **4** | **K2 producer order** (B.2) | The cheapest lever in the program (one index map) with the largest *interaction* term: ≈0 alone, −144 µs with certification, and it is what turns the 91.7%-readiness pathology into a staircase. | *Order sets the arrival time of the certified prefix; K3 and K5 are worthless without it.* Sweep all four `order.cuh` maps × K3 × **SKEW**. |
| **5** | **K1 transport op class** (B.1) | 67 → 93.6 → 117 GB/s across op class and slab size in-rig, a 13× coalesced/scattered layout cliff, and it is the one knob the **arch card says flips sign between gfx950 and gfx942**. | *Choose RMW vs store+local-reduce by the arch's atomic:store ratio; choose layout by coalescing, always.* Sweep transport × layout × **ARCH**. |
| **6** | **Skeleton/pool-job selector: carry vs idle vs consume** (Part A.1) | The only knob in the inventory with a measured **sign flip** (+340 vs −444 µs) under the same mechanism. Every "does producer/consumer work for all sizes?" question from the expert feedback lands here. | *A reserved CTA pays 8–9 µs/CTA unless it consumes a certified prefix; carrying never pays.* This is the direct answer to "does our M15 producer/consumer work best for all sizes?" |
| **7** | **`flush_rows` pool quota** (B.6) | It is the safety mechanism that keeps K5 from holding the rendezvous hostage, it is **co-tuned with C**, and it is already flagged as de-tuned at any T ≠ 4096. Cheap (runtime). Setting it to 0 isolates pure protocol deletion — a free control arm. | *Quota\* = shadow duration ÷ per-CTA consume rate (6.1 GB/s/CTA); it must be re-derived, not carried, across T.* |
| **8** | **`T_eff` / fill awareness (M24 `FILL` + `TGRAIN`) and `MAXTOK`** (B.9, B.11) | This is the actuator for the **FILL axis**, which the BRIEF shows is exactly where the C=32 loss and the C=512 win come from (37.6% vs 98.5% fill, `../BRIEF.md:44-46`). `MAXTOK = T` is the only confirmed production win on the T axis (**−1.49%**). `M24_NORIG_TABLE` lets fill and *heterogeneous* fill be swept synthetically, off-serving. | *Row-proportional cost should scale with real tokens; every quota tuned at padded capacity is wrong by the padding multiplier (2.66× measured).* |
| **9** | **Replication / placement (M18 theta, M19 adaptive, M20 slot pool, EPLB)** (B.10) | The actuator for the **SKEW axis**, which the BRIEF elevates to first-class (51.9% of traffic to one rank's experts; the hot rank holds the global slab certificate hostage, `../BRIEF.md:70-73`). Also the axis where an aggregate 4× replay win turned into a **−5.7% end-to-end loss** — the most instructive contradiction in the corpus. | *Replication pays iff per-chunk coverage exceeds θ; aggregate skew statistics mispredict serving because chunk wall-time is convex in skew.* |
| **10** | **Fusion boundary at the collective (F1) — CDAR vs GEMM+RCCL** | The **TOPO axis** knob: it is the entire difference between the DP8/EP8 and TP8+EP stories, it currently has a measured *negative*, and the honest negative is itself a deliverable ("RCCL AR is 1.5× our transport"). Nothing else in the inventory tests whether a fused collective is the right idea at all. | *Fuse the collective iff (exposed collective time) − (our transport penalty vs the vendor library) > 0; the crossover is a bandwidth-ratio threshold, not a philosophy.* |

**Just below the cut** (worth carrying as controls, not as primary axes):
push-vs-pull direction (B.8 — already measured to a tie at matched size, so it is a
*latency/ordering* knob whose interesting cells are small-message decode);
`bps`/item granularity (measured, but its only in-rig verdict is "worse");
XCD placement mode 3 (7–23% worse — a known-bad control that usefully bounds
interference); per-destination credits K6 (highest-value **unbuilt** knob, gated on a
skew harness existing — `credit.cuh:104-106`); and `K0P6GM_G=4` (pinned by `#error`, and
the only compute-side knob with a quantified prediction).

---

## Appendix — knob → manifest field mapping (what a reproducible arm row must carry)

Minimum row for a DP/EP (M15-family) arm, so any number can be regenerated:

```
skeleton        S1 | S3 | S4 | S5 | S8
build flags     K0P6_M15_{STAGED,SCATTER_RR,REPLICATE,ADAPTIVE,SAVE_Z}
                K0P6_M20_SLOTPOOL, K0P6_M24_{FILL,TGRAIN,ZERO_PAD,NULLWORK,
                                             NORIG_CONST|NORIG_TABLE,STRICT}
                K0P6_MPS_ENABLE_{MODE14,TBO}     <- instrument gates, MUST be pinned
                K0P6_M15_SRC_REV, kernel-name pin
config word     C, g (physical | detect | throttle | skip-part-zero | depth-sel),
                mode, flush_rows, pull_fallback, timestamps
host/geometry   T, MAXTOK, world, TOPK, spin_limit, EPLB placement, n_orig vector
workload tuple  ISL, OSL, concurrency, prompt count, fill, route-capture id, phase
gates           .text sha or same-session control; issue-run distribution
                (expect ~282 atomics / ~12 runs / mean ~23.5); pperr == 0
```

For a TP8/CDAR (M25-family) arm: `mode ∈ {compute, phased, fused}`, `tokens`,
`slab_rows`, `iters`, `depth`, `k_inner`, `bps`, `transport ∈ {atomic_accumulate,
store_towers}` — i.e. `m25_boundary_bench.hip:441-448` **already is** the manifest schema
for that skeleton.

**Reporting discipline carried from the BRIEF (`:109`):** every number in every row
carries its full workload tuple. No naked TPS, ever.

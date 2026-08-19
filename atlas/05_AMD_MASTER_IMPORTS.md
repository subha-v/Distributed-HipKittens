# amd-master survey — abstractions, findings, serving integration, and the iris lineage

What lives in `/Users/subha/repos/amd-master` (the companion research repo) that this atlas
depends on. All paths absolute unless prefixed with `<branch>:`. Repo root = `/Users/subha/repos/amd-master`.

**Critical orientation fact first:** `auto-gpu-kernel/k0_fused_moe/ABSTRACTIONS.md` is the
disambiguation map for four generations of abstraction work, three of which still have source on
disk and look equally plausible from a directory listing. Read it before citing anything. Its table:

| gen | location | status |
|---|---|---|
| 1. `k0_tile` / `amd_tile` | `unused/abstractions_k0_tile/` (+ live copy at `tile/include/k0_tile/`) | **DEAD** (moved 2026-08-07) — history only |
| 2. `flow` | `flow/` | **ACTIVE REFERENCE** — already contains working AG-GEMM + GEMM-RS |
| 3. `hkp` | `solution/hip/hkp/` | **LIVE** (4 of 8 headers) — the shipped 0.9414× arm depends on it |
| 4. `mk` | `solution/hip/mk/` | **GOING FORWARD** (written, never compiled) |

---

## 1. Abstractions — reusable primitives and design patterns

### 1a. `mk` — the going-forward distributed-megakernel abstraction (**the headline**)

`auto-gpu-kernel/k0_fused_moe/solution/hip/mk/` — `mk.hpp`, `mk_region.hpp`, `mk_channel.hpp`,
`mk_stripe.hpp`, `mk_counter.hpp`, `mk_roles.hpp`, `mk_placement.hpp`, `mk_store.hpp`,
`mk_sync.hpp`, `mk_topology.hpp`, `mk_config.hpp`, `mk_error.hpp`, plus `cost_model.py`.

Five concepts — **`region` / `phase` / `channel` / `stripe` / `counter`** — forming a device-side
library of collective *patterns* called from *inside* a single kernel, one level above a
device-side RMA layer (IRIS / MoRI `ShmemPtrP2p`). Explicitly **not** a collective library: "a
collective library cannot fuse, by construction: its API boundary is a buffer, so it must
materialize the full intermediate in HBM and must be a separate launch." Each header's comment
block encodes a *measured* negative result as an API constraint, which is the most presentable
property of the whole design:

- `mk_channel.hpp` — six verbs: `reserve` (waitless slot claim, remote atomic), `emit` (posted
  peer store, no sync, legal anywhere), `emit_row` (warp-cooperative staged row to N destinations
  — the MoE dispatch verb a naive API omits), `peek` (unsynchronized remote read),
  `publish`/`await` (the *only* synchronizing calls, phase-scoped coarse release/acquire),
  `live_extent`. **There is deliberately no per-item publish API**, because it was measured:
  cutting release events 1,374,408 → ~64 made the region *slower* (2.989× vs 2.688× control);
  32× finer granularity bought −5.1 µs out of an 898 µs pass; removing both system fences bought
  −19.2 µs; 97.9% of that pass is straggler wait. The signal is one `uint64` per (source, chunk)
  = `(epoch << 32) | fill_count`, sole authority for both readiness *and* extent, compared
  `== epoch` and never cleared, so HIP-graph replay advances the protocol with no host
  involvement and no reset phase.
- `mk_region.hpp` — one launch, one epoch; asserts the persistent-grid precondition (exactly
  1 CTA/CU) rather than documenting it. Enforces a **register discipline through the type
  system**: a phase body receives a `phase_ctx` carrying only descriptor pointer + error sink +
  spin limit, and `phase()` `static_assert`s the callable is captureless — making "hold a channel
  across the GEMM" a compile error instead of a 3-day debug. `phase()` deliberately emits *no*
  barrier (a grid barrier costs ~99 µs, and the shipped MoE kernel's only cross-rank concurrency
  lives in the un-barriered dispatch→consume gap); barriers are explicit `r.join()`.
- `mk_stripe.hpp` — work decomposition with a parallelism floor. Exists to prevent a measured
  47× defect (exp_64): a pure-copy consumer measured 2,747 µs against a 58 µs HBM floor because
  it striped over the *signal's* domain (world × chunks = 64) on a 1,024-resident-wave launch.
  Re-striping over rows: 205 µs; the full region went 1.2745× → 0.9414× on that one change.
  **The rule: the domain you stripe over must be chosen independently of the domain your
  readiness signal is chunked over.** Every stripe asserts `domain >= parallel_units`;
  `stripe_small` is the opt-out with a *mandatory* reason string.
- `mk_counter.hpp` — intra-rank counted dependency at AGENT scope. Not every dependency is a
  collective: the phase1→phase2 fused-GEMM handoff is compute↔compute within one rank, so
  routing it through `channel` would attach a release fence, a `buffer_inv`, and an epoch tag it
  does not need. This is what lets two GEMM phases run with no grid barrier. Carries the
  codebase's most expensive lesson as a placement warning: `wait` must be at task start, never
  mid-compute-body — "an observation that can spin, inside a wave holding hot state, on an
  occupancy-1 machine, is the single most expensive mistake in this codebase (11,173 µs)."

**Hardware-specificity:** AMD CDNA-specific (gfx950/gfx942, `buffer_wbl2 sc1` /
`buffer_inv sc0 sc1`, HIP). The *concepts* are hardware-agnostic; the *guardrails* are justified
by CDNA facts (forced occupancy 1, no `setmaxnreg`, no fused put-with-signal) and the design doc
states the scope condition explicitly.

### 1b. `flow` — the working reference transport/emission library

`auto-gpu-kernel/k0_fused_moe/flow/include/flow/` — built and compile-verified on node
(ROCm 7.2.4, gfx950), backburnered but **not dead**.

- `transport.hpp` — `flow::link<T>`: a symmetric allocation as a **by-value handle** (local base
  + 8 peer bases), graph-capturable, no runtime state needed in-kernel.
- `kittens.hpp` — `flow::glink<GL>` / `flow::at(gl, peer)`: a HipKittens `gl` descriptor
  retargeted to a peer's heap base. **This is the key idea for the atlas:** the emission
  primitive for tile code becomes just `store(flow::at(gl, peer), tile, coord)` — the store you
  were going to write anyway. Cross-rank movement disappears into the existing HK tile store.
- `emit.hpp` — `emit_warp` / `emit_cta` / `fetch_*` posted-store copies.
- `lattice.hpp` — `flow::lattice` + `complete_and_claim`, the completion machinery, measured
  K-loop-invisible.
- `observe.hpp` — `flow::observe` (bounded, fail-closed volatile poll), `flow::barrier`
  (whole-rank monotonic rendezvous), fences.
- `mori.hpp` (MoRI `SymmMemObj`) and `iris.hpp` (IRIS `heap_bases`) — **transport adapters**;
  `iris.hpp` builds a `link<T>` as `heap_bases[p] + off` computed once by value.
- Clients: `flow/kernels/gemm_rs_flow.cpp` (490 lines, + `host/run_gemm_rs.py`) and
  `flow/kernels/ag_gemm_flow.cpp` (486 lines, + `host/run_ag_gemm.py`). `gemm_rs_flow.cpp` is
  stock HipKittens bf16 GEMM with an **untouched mainloop** plus a flow epilogue emitting each C
  tile into its owner's heap, one `flow::barrier`, local slot-reduce — verified correct on 8 GPUs
  (rel_L2 = 0.004). Its own header comment states the conclusion the project re-derived three
  months later: *"No per-tile signals anywhere: on the egress boundary the consumer (a reduction)
  has nothing to interleave, so coarse observation (one barrier) is the right answer and fine
  observation is pure cost."*

**Caveat to present honestly:** `flow/STATUS.md` records that the *framing* in
`flow.hpp`/`emit.hpp` predates exp_50/51 and asserts three premises now known false
(postedness-as-mechanism, register-slack as sole scarce resource, a single register gate).
**The mechanism is unaffected — read the kernels, distrust the prose.**

**Specificity:** HIP/CDNA + HipKittens-coupled (`kittens.hpp`), but the `link` concept is
vendor-neutral and has two independent backends (MoRI, IRIS).

### 1c. `amd_tile` — the portable semantic core (hardware-agnostic, CPU-testable)

`abstraction/include/amd_tile/core/distributed_tile.hpp` (+ `core/core.hpp`)

A **HIP-free, portable C++ semantic model** for distributed tile transactions. Header comment is
explicit: it "does not issue device operations, infer memory ordering, or adapt a pointer to a
HIP descriptor." Type vocabulary: `memory_space`, `access_mode`, `agent`, `generation_id`,
`visibility_scope {workgroup, agent, system}`, a 24-value `tile_error` enum, `status`/
`tile_result`. Data-movement lifecycle types: `completion_domain` + `completion_group` +
`producer_set_view` (immutable, plan-owned producer sets, copied into the slot at bind time so a
later plan mutation cannot alter an in-flight ticket), `segmented_exchange_plan` with CSR
segments and a **destination-indexed `provenance_entry`** (so an accidentally reordered inverse
plan fails closed), `tile_bundle_view` (multi-plane logical transfer with one
`logical_bundle_identity` — payload and scales as one transfer), `tile_ticket`,
`tile_reservation`, `acquired_tile`, and `bounded_tile_channel` with the full verb set:
`reserve` / `publish` / `record_arrival` / `acquire` / `release` / `abort` / `discard` /
`advance_generation`. The channel is a **mutex-protected CPU reference model** of a bounded
generation ring, and it says so: "models slot ownership and capability lifetime, not GPU
publication/fence behavior. A HIP, MORI, or IRIS backend must separately prove
payload-before-signal."

**Specificity: hardware-agnostic** — this is the one piece that is genuinely vendor-neutral and
runs/tests on CPU. Tests: `abstraction/tests/core/test_distributed_tile_core.cpp`.

### 1d. `amd_tile::hipkittens` — the typed bridge to HipKittens, and its CDNA4 lowering

- `abstraction/include/amd_tile/hipkittens/acquired_tile_bridge.hpp` — a concepts-based bridge
  from the portable capability model to a HipKittens-like backend, deliberately including **no**
  HipKittens headers (a device TU supplies `tile_traits` specializations). Introduces
  `device_capability_identity` (generation, sequence, completion id, slot, live rows, scope),
  with the honest caveat that it is *metadata, not an in-kernel liveness oracle* — a device
  capability must be reminted before a graph replay that follows release or slot/generation
  reuse. **Hardware-agnostic layer.**
- `abstraction/include/amd_tile/hipkittens/hipkittens_backend.hpp` — the concrete **CDNA4**
  lowering (`#error`s without `KITTENS_CDNA4`). Implements a short whole-workgroup sequence:
  minted device capability → raw-buffer HBM→LDS issue → `vmcnt(0)` + workgroup barrier →
  LDS→register load + `lgkmcnt(0)`. Contains `cdna4_pod_global_view` — a POD that "quacks like
  an HK `gl`" because native `kittens::gl` has a user-provided copy ctor and is therefore not a
  trivially-copyable kernel capability. **HipKittens/CDNA4-specific.** Smoke test:
  `abstraction/tests/hipkittens/device/hipkittens_backend_smoke.hip`.

### 1e. `k0_tile` — the first-generation device transport concept (DEAD, but the cleanest teaching artifact)

`auto-gpu-kernel/k0_fused_moe/tile/include/k0_tile/` — `addressing.hpp`, `backend.hpp`,
`completion.hpp`, `descriptors.hpp`, `operations.hpp`, `config.hpp`, `formats.hpp`,
`numeric.hpp`. (Archived twin: `unused/abstractions_k0_tile/`.)

Marked **DEAD** by `ABSTRACTIONS.md`, but its header prose is the most quotable statement of the
design contract in the repo:
- `backend.hpp` — the **transport backend concept**: `copy_1d`, `copy_2d`,
  `commit(scope, epoch, expected_mask) -> publish_ticket`, `acquire(ticket)`,
  `peer_base(local_symmetric_ptr, peer)`. Two lowerings: `shader_p2p_backend` (real gfx950,
  shader-issued direct peer store over the IRIS/MORI symmetric heap) and `cpu_ref_backend`
  (models a `world`-rank symmetric fabric in ordinary host memory so the push mapping is testable
  without a GPU). Notes the v0 primitive is "exactly the graph-safe primitive exp_001 requires
  and the OPPOSITE of exp_24's `ShmemPutMemNbiBlock` (which wedged repeated graph replay)."
- `completion.hpp` — **"'Synchronization vanished' is a forbidden claim."**
  `publish_ticket {epoch, completion_scope, expected_mask, slot}` with
  `completion_scope {wave, workgroup, device, fabric}`, static_asserted trivially-copyable +
  standard-layout so it is graph-capturable by value.
- `descriptors.hpp` — `tile_spec<Shape, Format, Layout>` (zero-cost compile-time),
  `tile_endpoint` (base ptr, strides, local-or-peer rank, generation/epoch, access mode, format).
  Deliberately **does not overload a HipKittens `gl` with remote state** — `acquire_local()`
  yields an ordinary local view existing HK code consumes.
- `addressing.hpp` — "the SHARED host/device addressing math… the semantic core." Every function
  is `K0T_HD`, so the gfx950 push kernel and the CPU tests call the *same* function and cannot
  drift.
- `operations.hpp` — HIP-free operations layer; combine slots keyed by producer rank ⇒
  collision-free plain stores, **no remote atomic**; a missing producer must leave poison and
  make the reduction *wrong* (fail-loud by construction).

**Specificity:** the concept layer is agnostic (has a CPU backend); `shader_p2p_backend` is
gfx950-specific.

### 1f. `hkp` — what is actually shipping

`auto-gpu-kernel/k0_fused_moe/solution/hip/hkp/`. The prefill driver
(`prefill_opt/host/e004pf_k0pf_ab.py:259`) installs **exactly four** headers into the JIT build —
`hkp_sort.hpp` (destination counting sort), `hkp_sync.hpp` (grid barrier, epoch doorbells),
`hkp_quant.hpp`, `hkp_topology.hpp` — plus two kernel-local ones:
`prefill_opt/kernels/k0pf5_ll128.hpp` (wire format only; the LL128 doorbell is gone) and
`prefill_opt/kernels/k0pf6_chunk_release.hpp` (**the chunk-counted publish/acquire — the live
release/acquire primitive**). `hkp.hpp`, `hkp_alloc.hpp`, `hkp_store.hpp`, `hkp_gemm.hpp` are
**not installed** (dead, left in place deliberately).

### 1g. Upstream HipKittens producer/consumer pipeline micros

`HipKittens/kernels/gemm/bf16fp32/micros/producer_consumer/{16x32,32x16}/` — a ladder of
warp-specialized producer/consumer GEMM variants (2-stage/3-stage, 8c4p / 12c4p / 16c2p, async
arms) with per-variant notes on which configurations exceed the SW limit or hit scratch/spills.
`HipKittens/analysis/paper_experiments/producer_consumer_micro/`. **HipKittens/CDNA-specific**,
single-GPU — this is the intra-kernel producer/consumer handoff pattern, complementary to the
cross-rank one above.

### 1h. Design docs that define the abstractions (not code)

- `auto-gpu-kernel/k0_fused_moe/findings/DISTRIBUTED_MEGAKERNEL_ABSTRACTION_20260806.md` —
  **the single best document in the repo for the mentor meeting.** §5 is the full programming
  model for `mk` (concepts table, six channel verbs, `emit` as a family,
  `placement::{routed,sharded,replicated}`, `counter`, role-split `ctx.on_cta`,
  `region`/`phase`), §8 is "guardrails — what the API makes unrepresentable", §10 is the build
  spec, §11 the gfx950→gfx942 portability contract (the MoE kernel's LDS footprint **does not
  fit on MI300X**).
- `abstraction/DISTRIBUTED_TILE_ABSTRACTION.md` — the tile-lifecycle thesis: *"The useful
  abstraction is not a nicer wrapper around peer stores. It is a tile lifecycle that jointly
  describes representation, placement, routing, asynchronous movement, readiness, local staging,
  compute, and retirement."* Contains a full code map table (responsibility → exact `file:line`)
  spanning k0_tile, HipKittens `gl.cuh`/`global_to_shared.cuh`/`global_to_register.cuh`, and
  IRIS `iris.hpp:133-390`.
- `abstraction/DISTRIBUTED_MOE_TRANSACTION_ABSTRACTION.md` — widens the unit of abstraction from
  the BM32 tile to the whole transaction (route → dispatch → assemble → execute → publish →
  combine → retire). Key stance: **push and pull are independent policies for dispatch and
  retirement, not part of the semantics.** "Push combine" never means remote atomic accumulation
  — it is BF16 remote store into a producer-keyed collision-free slot, release, owner acquires,
  local FP32 reduce, one BF16 store.
- `abstraction/GRAPH_SAFE_TASK_LOCAL_DESIGN.md`, `REUSABLE_TILE_REGION_RESEARCH_PLAN.md`,
  `ABSTRACTION_AND_DEEPSEEK_SERVING_PLAN.md`, `ONLINE_PRIOR_ART_2026-07-21.md`.
- `iris/irisx/tilecomm/DESIGN.md` — the tile-level communication abstraction spec: declare /
  schedule / execute, four intents (`tile_gather` / `tile_scatter` / `tile_reduce_scatter` /
  `tile_all_reduce`), and the quadrant vs NCCL / MSCCL / aiter / IRIS / HK. Paper seed:
  *"Tile-Level Communication for Fused Multi-GPU Inference."*

---

## 2. Findings / evidence — why kernels were fast or slow (top 10)

1. `auto-gpu-kernel/k0_fused_moe/findings/DISTRIBUTED_MEGAKERNEL_ABSTRACTION_20260806.md` —
   **"A megakernel is not a pipelining device; it is a boundary-elimination device."** Occupancy 1
   is *forced, not chosen* (measured 438 VGPR / 182 AGPR / 102 SGPR, 0 spill, LDS 120,488 B ⇒
   1 wave/SIMD, 1 CTA/CU); waves cannot be specialized without `setmaxnreg`, so role splitting
   buys scheduling, never resources; cross-agent release has a fixed coarse price with no
   put-with-signal. Shipped region 0.9414× production (7,248 vs 7,695 µs) via
   23,132 → 21,154 → 9,802 → 7,248 µs (3.19×).
2. `EXPERIMENTS/2026-08-k1-comm-overlap/README.md` (+ `auto-gpu-kernel/k1_comm_overlap/experiments/LESSONS.md`,
   73 numbered lessons) — **the campaign ruled out the entire synchronization-granularity axis by
   measurement.** T1 MoE megakernel 0.9414× → **0.8985×**; T2 GEMM→ReduceScatter 1.2319× →
   **0.8898×** (first arm to beat stock RCCL, and 0.7761× vs the RadeonFlow grand-prize kernel);
   T3 AllGather→GEMM 1.5909× → 1.4525×. Also: the grand-prize kernel itself *loses* to stock
   RCCL on MI350X (1.1506× COMM-DOMINANT, 2.1000× MIXED) — ratios are not shape-constant.
3. `auto-gpu-kernel/k0_fused_moe/findings/PREFILL_REGION_ATTRIBUTION.md` — the roofline
   correction: the 9.95 ms prefill region is real, not a capacity artifact; **"≥88% is not the
   GEMM" is REFUTED — the GEMM is 58.4%, movement 37.8%, and the region is ~100% kernel-busy with
   essentially no barrier slack.** GEMM at 579.6 TFLOP/s = 12.6% of dense FP8 peak. **The actual
   prize is GEMM load imbalance**: `fmoe + EpCombine` is constant to 0.69% across ranks while
   each term varies 2–3× (combine is *waiting*, not moving); pair counts span 2.15× on a
   "balanced" corpus ⇒ balancing cuts 1,167 µs = 11.8% of the region with no kernel change.
4. `auto-gpu-kernel/k0_fused_moe/findings/MOVEMENT_GAP_67.md` — where our movement loses to MoRI,
   **with a correction banner that refutes its own headline**: production does *not* fuse the
   rendezvous to zero (MoRI calls `CrossDeviceBarrierIntraNodeKernel` unconditionally at
   `intranode.hpp:295`, billed inside the 49.36 µs). Per boundary, same run, 8 ranks:
   **dispatch TIED** (45.40 vs 45.16), **combine +26.92 µs against us** (49.36 vs 76.28 =
   barrier 38.96 + work 37.32). Our combine *work* is at parity; the loss is the exposed barrier.
5. `auto-gpu-kernel/k0_fused_moe/findings/PRODUCTION_MOVEMENT_MECHANISM.md` — source-level audit
   of how MoRI `EpDispatch`/`EpCombine` actually move data, pinned to the recovered route-d build
   snapshot + captured disassembly. Written explicitly to survive the question *"then how did
   production push efficiently?"*
6. `auto-gpu-kernel/k0_fused_moe/findings/PF6_RELEASE_ACQUIRE_PROTOCOL_20260806.md` — root-causes
   `pperr=8388608` to **wrong cache scope**: the LL128 producer store (`k0pf5_ll128.hpp:37-43`,
   misleadingly named `..._sys`) is a plain `global_store_dwordx4` with **no `sc1` bit**, so its
   dirty L2 line is never written back over xGMI as a release; the consumer poll uses
   `__builtin_nontemporal_load` (`nt` hint, **not** `sc0 sc1`) so it never invalidates a stale
   non-local line. The canonical gfx942/gfx950 cross-agent visibility lesson.
7. `iris/irisx/tilecomm/MEASURED_FINDINGS.md` — **the all-reduce is 59–85% of serial GEMM+AR**
   across R1-realistic TP4 shapes (1.2–4.5× the GEMM); comm dominates. Overlap ceiling for
   tile-fused RS/AR ≈ **1.18–1.69×** prefill. But the shipped IRIS fused-collective examples are
   *not* competitive (ex.09 one-shot AR = 4.7× slower than unfused; ex.08 atomics = 280× slower)
   — the opportunity is real, the substrate needs new kernel work.
8. `auto-gpu-kernel/k1_comm_overlap/findings/CORESIDENCY.md` — co-residency excluded on its one
   clean comparison (96 timed cells, 6 permutations × 4 grid sizes); the 68% is a per-CTA
   intrinsic of a peer emit, and **emission is closed at 99.3% of busbw**, so no speedup is
   available there. Reads with `PEER_VS_LOCAL.md` (the T2 2×2: 32% local / 68% destination).
9. `auto-gpu-kernel/k1_comm_overlap/findings/BANDWIDTH_AUDIT.md` + `MECHANISM_NAMED.md` — the
   methodology lesson worth a slide: `hipMalloc` (coarse-grained) + 8 MiB slots against a
   224 MiB MALL with no system fence **timed store retirement into cache, not fabric arrival**,
   inflating GB/s ~6× over the measured 54.9 GB/s per-link ceiling. `MECHANISM_NAMED.md`'s
   filename is aspirational; the answer is a documented refusal — kept on the record.
10. `auto-gpu-kernel/k0_fused_moe/findings/XCD_LOCALITY_ANALYSIS.md` — **XCD chiplet-locality
    lever: DEAD** (verdict stands). Its balance-lever *magnitudes* are superseded by a correction
    banner; the validated end-to-end objective is 2-expert swap −316 µs (−3.11%), annealed
    −370 µs (−3.65%).

**Honorable mentions:** `findings/AMD_GFX950_OVERLAP_HARDWARE_NOTES_20260804.md` (public-safe
hardware notes for AMD-vs-NVIDIA overlap comparison), `findings/AMD_NATIVE_OVERLAP_RESEARCH_THESIS_20260730.md`,
`findings/BARRIERLESS_OVERLAP_PRIOR_ART_20260806.md`, `findings/COST_GUIDED_TILE_DATAFLOW_THESIS_20260803.md`,
`findings/PF6_REGISTER_CEILING_ANALYSIS_20260806.md`, `findings/CURRENT_KERNELS.md`,
`abstraction/profiling/EP_REGION_PROFILE.md`, `abstraction/VLLM_SERVING_RESULTS.md` (append-only
serving ledger; corrections must preserve the original claim). `EXPERIMENTS/README.md` is the
reproducibility registry with explicit validity labels (CURRENT / MEASURED / HISTORICAL /
**RETRACTED** — the old `combine_pull` 386 µs and the 1.56× headline are retracted).

---

## 3. `vllm-integration-m15` / `vllm-integration-m18` — megakernel-in-serving

All under `auto-gpu-kernel/k0_fused_moe/vllm_r1_aiter_e2e/shim/pf4h_integration/`.

**The m15 design.** The pre-existing PF4H packet replaces the DeepSeek-R1 MoE region with
**six** pinned launches (`qpush_iso → dsort → N2 p1 → N2 p2 → barrier → combine`). M15 adds a
strictly additive alternative that replaces the same region with **one** launch of
`k0pf6gm_m15_mega`, the slab-certified **pipelined-combine megakernel** from
Distributed-HipKittens — i.e. the producer/consumer handoff that used to be six host-visible
kernel boundaries becomes intra-kernel `channel`-style publish/await plus the intra-rank
`counter` handoff between the two N2 GEMM phases. Selection is one exact env value,
`VLLM_PF4H_INTEGRATION_MODE=m15` (unset|full|m15); with it unset or `full`, every existing file
is byte-identical in behavior.

Files (`vllm-integration-m15:` branch, all additive):
- `m15_contracts.py` — the 63-word `K0P6_D_*` descriptor dictionary, a bit-identical Python
  mirror of `hk_moe::mps::encode_config`/`decode_config`/`config_is_valid`, the exact
  buffer-shape table, and host-side attestation of every device entry-guard condition.
- `m15_runtime.py` — `M15ProcessRuntime`: module load + `shmem_module_init`, MoRI heap snapshot,
  state-ring allocation, cross-rank symmetric-offset proof, **one descriptor per routed layer**
  (not per ring slot — sharing across layers would need an illegal host write inside the captured
  region; 58 × 63 × 8 B = 29 KB), the **capture-time pointer latch**, the single launch, health
  hooks.
- `m15_graph_body.py` — the one-launch HIP-graph capture body beside the frozen six-launch body.
- `m15_kernargs.py` — the 5th/6th pinned code-object schemas; kernarg segments **measured, not
  guessed** (constrained to {24, 280} and {8, 264}; still `None` pending the container build gate).
- `m15_vllm.py` — `M15PrepareAndFinalize` / `M15Experts`, inheriting every PF4H eligibility proof
  verbatim.
- `m15_pin/k0pf6gm_m15_mega.hip` — the **serving RUN PIN**: a kernel-named TU for MoRI's
  `compile_genco`, containing nothing but `#error` guards on every forbidden arm
  (`K0P6_M15_STAGED=0` ⇒ exactly 63 words; `K0P6_M15_SCATTER_RR=0` donor scatter;
  `K0P6_MPS_ENABLE_MODE14/TBO/E23_RING=0`; `K0P6GM_G = N2GM_G = 3`, the measured G-stack, written
  back through `num_tiles[1]` for host check). Build flags: `-std=c++20 -O3 -DKITTENS_CDNA4
  -ffast-math -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3`.
- `m15_pin/k0pf6_chunk_release.hpp`, `m15_pin/k0pf6_mori_heap_snapshot.hip`, `m15_sources.py` +
  `stage_m15_sources.py` (content-addressed `#include` closure with pinned SHA-256s),
  `apply.py --integration-mode {full,m15}`, `../backend_patch/fp8_pf4h_m15.patch` (stacked on
  `fp8_pf4h_full.patch`), `tests/test_m15_contracts.py` (34 structural),
  `tests/test_m15_descriptor_semantics.py` (5, proving slot-by-slot that the packet binds the
  same object at the same index as `prefill_opt/host/e004pf_k0pf_ab.py`).
- Benchmarks: `benchmarks/2026-08-12_m15_stage0/standalone_m15_gate.py` (GPU ladder a/b/c),
  `benchmarks/2026-08-12_m15_campaign/{CAMPAIGN.md,run_m15_campaign.sh,summarize_m15_campaign.py,m15_expert_skew.py}`.
- **Status doc:** `vllm-integration-m15:auto-gpu-kernel/k0_fused_moe/HANDOFF_VLLM_M15_20260812.md`
  — ladder paused after step 2 of 5 (sources staged and SHA-verified; container build gate +
  kernarg pin not started). Kernargs are still unpinned.

**m18 (branch `origin/vllm-integration-m18`, no local ref — use `origin/`).** Adds
`m18_replication.py`: **static hot-expert replication**, an env-gated extension riding the m15
path (`VLLM_PF4H_M18_REP_EXPERTS`, exact-string opt-in). It swaps in `k0pf6gm_m18_mega`
(`K0P6_M15_REPLICATE=1`), widens the descriptor to **64 words** (slot 63 = the M18R table),
grows PADMAX by the replica margin, and the weight converter returns `[:E]` views of `[EL,...]`
extended allocations whose replica slots hold bit-exact owner slices from an init-time
`torch.distributed` broadcast. **Measured basis (2026-08-14, MoK skew replay of the serving
histogram): the aggregate top-16 replica set takes the megakernel from 0.8467× to 0.2489× of the
AITER+MoRI production arm; balanced-routing cost of carrying replicas is +3.9%.** The same file
also declares M19 (`K0P6_M15_ADAPTIVE`, `k0pf6gm_m19_mega`, per-chunk threshold θ, slot 64
symmetric decision words). Kernel side: `m15_pin/k0pf6gm_m18_mega.hip`; contract pinned to
`distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip @ 0f9676ce`.

**Build gate evidence** (present on both branches):
`EXPERIMENTS/2026-08-k0-pf6-megakernel-build-gate/` with `RESULTS.md`, `SETUP.md`, `SOURCES.md`,
and artifacts `k0pf6_mega.{co,hsaco,isa,notes}` + `compile.log`.

---

## 4. `iris/` — the mentor's device-side data-movement project as vendored here

`/Users/subha/repos/amd-master/iris/` is a **full vendored checkout** of upstream IRIS plus a
project-local research fork, `irisx/`.

**Upstream IRIS (the substrate).**
- `iris/irisx/development/include/iris/iris.hpp` — the C++/HIP device API. One symmetric heap
  per rank; `class iris` + an inner `iris_device_view` exposing `translate(ptr, remote_rank)`,
  `get_heap_base(rank)`, `load`/`store`, `atomic_load`/`atomic_store`, `fetch_add`/`fetch_sub`,
  `compare_exchange_strong`, `fence`, over explicit `memory_order` and
  `memory_scope {thread, warp, block, device, system}`. Cited in
  `abstraction/DISTRIBUTED_TILE_ABSTRACTION.md` at `:133-390`, with the release/acquire pattern
  at `iris/irisx/development/examples/01_message_passing/message_passing.hip:24-64`.
- `iris/iris/iris.py` — the Triton/PyTorch surface: `Iris` (symmetric-heap tensor factories
  `zeros/ones/randn/empty/arange/full/...` that allocate *on the symmetric heap*,
  `get_heap_bases()`, `barrier()`), a `ccl` submodule (`all_to_all`, `all_reduce`,
  `all_reduce_preamble`), and free Triton device functions: `load`, `store`, `copy`, `get`,
  `put`, and the atomic family (`atomic_add/sub/cas/xchg/xor/and/or/min/max`) all parameterized
  by `(from_rank, to_rank, heap_bases, sem, scope)`.
- `iris/examples/` — AMD's own sweep of fused GEMM+collective *structures*, the direct prior art
  for the abstraction: `10_gemm_all_scatter_wg_specialization`,
  `11_gemm_all_scatter_producer_consumer`, `12_gemm_all_scatter_bulk_synchronous`,
  `14_all_gather_gemm` (pull *and* push variants), `15_gemm_all_reduce_ring_based`,
  `16_all_reduce_ring_based`, `20_gemm_all_scatter_independent`, `13_flash_decode`,
  `04_atomic_add`, `05_atomic_xchg`, `00_load`. Explicitly cited as external reference in the
  megakernel design doc §2c.

**How it is used in this repo.**
1. **As a transport backend behind an abstraction** — `flow/include/flow/iris.hpp`
   (`flow::from_iris(heap_bases, local, me, world)` → `link<T>`), sitting beside `flow/mori.hpp`.
   Same role in `k0_tile/backend.hpp`, whose `shader_p2p_backend` is "shader-issued DIRECT peer
   memory movement over the existing IRIS/MORI symmetric heap."
2. **As the layering floor** — the megakernel design doc's layer table puts IRIS / MoRI
   `ShmemPtrP2p` at "device RMA — a peer load/store/atomic", the `hkp`/`mk` library one level
   above it, and host collectives (RCCL, MoRI `EpDispatch`/`EpCombine`) above that. Stated
   policy: **"MoRI and IRIS are the substrate, not the competition."**
3. **`irisx/` — the project's own research fork.** `iris/irisx/fused_moe/` (the hand-rolled EP8
   MoE region: `kernel.cpp`, `fused_region_e12.cpp`, `fused_region_e26.cpp`,
   `gather_pack_dropin.cpp`, `ep8_gather.h`, `quanttile_decode.h`, `b1_dispatch_route.py`,
   `mori_epdispatch_ref/`, `aiter_ref/`, `DATA_FLOW_AND_ABI.md`, `B1_DISPATCH.md`);
   `iris/irisx/tilecomm/` (the tile-level comm abstraction: `DESIGN.md`, `tilecomm_device.h`,
   `tilecomm_device_v2.h`, `tilesched.py`, `xgmi_probe.py`, `MEASURED_FINDINGS.md`,
   `TILE_REDUCE_SCATTER_RESULT.md`, `QUANTTILE_V0_RESULT.md`, `WRAPPER_RESEARCH.md`);
   `irisx/development/` (occupancy variants incl. `archive/occ_variants/v4_staged_consumer/`,
   `tests/test_producer.hip`); `irisx/baselines/`; `irisx/EXPERIMENT_LEDGER.md`.
4. **TileComm's motivating result** — the combine collective got **2.4× faster (934 → 386 µs)**
   from one hand-rolled schedule (`build_combine_pull(..., interleave=True)` round-robining cells
   across XGMI links). The thesis: that decision is workload-blind, non-reusable and opt-in;
   TileComm makes the schedule something the library *computes* from declared demand + measured
   link topology. `tilesched.py`'s cost model reproduces the measurement (sorted 927.8 /
   round-robin 388.2 / proportional 388.3 µs vs measured 934/386) and shows `proportional` tracks
   the link lower bound at every Zipf skew. **Honest read recorded in the README:** reorder-only
   scheduling buys 2.4–6× over the *naive* order but only 1–4% over the *good hand-roll*; the
   unbounded win is Layer-3 tile-fused comm/compute overlap. `xgmi_probe_results.md` records the
   on-node validation as **inconclusive** — the probe is issue-bound (IRIS stores are
   fire-and-forget, so it timed issue rate, not link BW) and needs a completion fence.
5. **Presentation material already built** — `iris/june-30-presentation/` (self-contained, with
   `epdispatch_data_movement_screenshots/`, `fused_moe_kernel_screenshots/`,
   `fused_moe_flow_screenshots/`) and `iris/july-07-presentation/` (the TileComm deck), both with
   Stanford style files. Also `iris/docs/images/pattern-producer-consumer.png` and
   `perf-producer-consumer.png`, and `iris/PRIOR_ART.md`, `iris/KERNEL_PLAN.md`,
   `iris/MASTER_HANDOFF.md`.

**Note:** no code outside `iris/` imports the IRIS Python package directly — integration is via
the C++ adapters (`flow/iris.hpp`, `k0_tile/backend.hpp`) and via `irisx/` as a sibling research
tree. The production serving path (m15/m18) runs on **MoRI**, not IRIS.

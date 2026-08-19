# G-L4 — TP8+EP serving integration map (Phase A, map only)

Written 2026-08-18, laptop, written in slices (node dropped mid-recon; every
node/container fact is tagged **UNVERIFIED** with the command that settles it).
Scope: map the seam so Phase B can start on a go signal. Nothing here is built.

Target arm: our TP8+EP hybrid vs AMD-recommended TP8+EP (`native_tuned_tp`),
cell c32p. Win bar: > 25,844 input tok/s with TTFT p50 <= prod +10%
(`UNDERSTANDING_PLAN.md:100-104`).

---

## 1. THE SEAM

### 1.1 The per-layer chain, TP8+EP

Per MoE layer (58 of 61; layers 3..60 per shim `contracts.py:63-64,69-71` as
cited in `M23_RAGGED_SEAL_DESIGN.md:139-141`):

```
input_layernorm
  -> MLA attention (heads 16/rank)  -> o_proj = RowParallelLinear
       => AR#1  (all-reduce, 16,384 x 7168 bf16 = 235 MB)
  -> post_attention_layernorm
  -> DeepseekV2MoE.forward:
        gate (router)               [local, replicated, redundant on 8 ranks]
        shared expert  (TP-sharded) [local]
        routed experts (EP, 32/rank)[local row-gather; NO dispatch all2all]
        shared + routed summed
       => AR#2  (all-reduce, 235 MB)
  -> next layer
```

**MoE region begins** at `DeepseekV2MoE.forward` entry (the post-attention,
post-norm hidden) and **ends** at the tensor handed to the TP all-reduce. The
dispatch all2all does not exist here: the hidden is replicated by AR#1, so each
rank gathers rows for its own 32 experts out of local memory
(`TP8_OVERLAP_ANALYSIS.md:29-34`; `TP8_STATE.md:52-62`).

### 1.2 Where the all-reduces are issued

- **AR#1** — inside `RowParallelLinear.forward` for `o_proj` (`reduce_results=True`
  when `tp_size>1`). **UNVERIFIED (line)**: `grep -n "tensor_model_parallel_all_reduce"
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/linear.py`.
- **AR#2** — in the MoE path, either `FusedMoE.maybe_all_reduce_tensor_model_parallel`
  or an explicit call in `DeepseekV2MoE.forward`. **UNVERIFIED (line)**.
- **Open first-order question**: is the MoE layer **one** AR or **two**?
  `moe_runner.py:418-434` (`_maybe_reduce_shared_expert_output`) all-reduces the
  shared output over the TP group; on the DP path it is a no-op because
  `contracts.py:53` has `tp_size = 1` (`SHARED_EXPERT_FILLER_DESIGN.md:103-105`).
  **Under TP8 that call goes live**, which would make it a *separate* AR unless
  vLLM defers the sum. This changes the prize pool and the filler story.
  **Verify**: read `moe_runner.py:392-434` + `DeepseekV2MoE.forward` under a TP8
  config, or profile one native TP8 step (G-L0c).
- Transport backend: the TP group's large-message all-reduce is RCCL/PYNCCL.
  **UNVERIFIED**: `grep -i "all-reduce backends\|parallel" ~/20260818_m15_campaign_b0v5/pair_01/*native_tuned_tp*/server_config_dump.txt`.

### 1.3 Where the shared expert runs today

Constructed in `deepseek_v2.py:347-360` as `DeepseekV2MLP(hidden=7168,
intermediate=2048, reduce_results=False)` and handed to `FusedMoE`
(`:362-388`), so the model never calls it directly (`:398-427`). It executes
through `layer.py:408` -> `moe_runner.py:277-284` -> `runner/shared_experts.py:155-174`,
whose non-overlap branch is `shared_experts.py:172`. Both overlap orders are
dead for us: `MK_INTERNAL_OVERLAPPED` needs `supports_async()` which
`vllm_full.py:229-232` returns False for, and `MULTI_STREAM_OVERLAPPED` needs
`current_platform.is_cuda()` (`platforms/interface.py:189-190`, False on ROCm).
=> **`SharedExpertsOrder.NO_OVERLAP`: serialized in front of the routed region,
every layer, every step** (`SHARED_EXPERT_FILLER_DESIGN.md:46-81`).
*Caveat*: those line numbers were read on a **DP-configured** mirror; the call
sites are parallelism-independent but the branch taken under TP8 is
**UNVERIFIED**.

### 1.4 What the hybrid replaces vs keeps

| element | hybrid |
|---|---|
| router gate GEMM | KEEP (vLLM's; tiny, local) |
| expert row-gather + grouped expert GEMMs + weighted combine | **REPLACE** — this is the mega's MoE region |
| shared-expert MLP | **RELOCATE** — off the serial path, into the AR shadow (§4) |
| AR#1, AR#2 | **KEEP RCCL** (`UNDERSTANDING_PLAN.md:229-243`: fusion hides nothing at any reachable rho; RCCL leads fused by 693-994 us) |
| dispatch/combine all2all, MoRI | ABSENT under TP8 (`TP8_STATE.md:368-372`) |
| M23 ragged seal, M24 dummy-skip | DIE under TP8 (same cite) — see §4 |
| attention (AITER MLA), norms, dense layers | KEEP |

### 1.5 The structural constraint the hybrid must respect

Within one layer the chain is **strictly serial**: AR#1 -> MoE -> AR#2, and
every consumer needs the *fully* reduced tensor (RMSNorm is nonlinear). So a
**monolithic** all-reduce leaves **no dependency-free co-resident work in the
same layer** — not even the shared expert, whose input is post-AR#1 and whose
output feeds AR#2. Any filler therefore requires the AR to be **token-chunked**
so that chunk *j*'s compute overlaps chunk *j+1*'s reduction. This is a
sequencing consequence for the gate ladder, recorded here and expanded in §4.

---

## 2. WEIGHTS

### 2.1 What the mega expects today

The mega does **not** own a loader. It binds *pointers* to vLLM's own already-
AITER-shuffled fp8 tensors. `m15_vllm.py:107-157` (`_validate_weight_layout`)
gates the exact shapes, derived from `FROZEN_CONTRACT`
(E=32 local experts, H=7168, I=2048, quant_block=128):

| tensor | required shape | dtype/contiguity |
|---|---|---|
| `w1` (gate+up) | `(32, 4096, 7168)` | 1-byte contiguous fp8 |
| `w2` (down) | `(32, 7168, 2048)` | 1-byte contiguous fp8 |
| `w1_scale` | `(32, 32, 56)` | contiguous f32 |
| `w2_scale` | `(32, 56, 16)` | contiguous f32 |

Bind site: `m15_vllm.py:458-464` `runtime.bind_layer_weights(record, w1,
w1_scale, w2, w2_scale)`, called from `M15Experts.apply` with the tensors vLLM
passed in; `register_layer_weights` (`:468-470`) retains them so an EPLB
rearrangement can re-read `data_ptr()`. The same shapes are quoted
independently at `SHARED_EXPERT_FILLER_DESIGN.md:169`.

### 2.2 Routed experts: probably a **no-op** port

Under `--enable-expert-parallel` the MoE's own tensor-parallel degree collapses
to 1 and `ep_size = tp_size * dp_size`. That is 8 in **both** deployments
(DP8/EP8: 1x8; TP8+EP: 8x1), so `num_local_experts = 256/8 = 32` and
`intermediate_size_per_partition = 2048` in both — i.e. **the routed-expert
weight layout is expected to be byte-identical between the DP path we ship and
the TP8 path we are targeting, and §2.1's gate should pass unchanged.**

This is the single largest cost saving in the whole port, so it must be
verified, not assumed. **UNVERIFIED.** Verify by reading
`FusedMoEParallelConfig.make()` in
`vllm/model_executor/layers/fused_moe/config.py` (how tp/dp/ep are derived when
`use_ep` is set) plus `FusedMoE.create_weights` for
`intermediate_size_per_partition`; cross-check against the resolved dump
`~/20260818_m15_campaign_b0v5/pair_01/*native_tuned_tp*/server_config_dump.txt`.
The expert-map attestation `vllm_full.py:328-373` (contiguous EP8,
`begin = rank * 32`) should also survive unchanged if this holds.

### 2.3 Shared expert: this one really does change

| | DP8/EP8 (today) | TP8+EP (target) |
|---|---|---|
| `gate_up_proj` (MergedColumnParallel) | `[4096, 7168]` | `[512, 7168]` (output sharded /8) |
| `down_proj` (RowParallel) | `[7168, 2048]` | `[7168, 256]` (input sharded /8) |
| block-fp8 scales | `[32, 56]` / `[56, 16]` | `[4, 56]` / `[56, 2]` |
| reduction | none (`tp_size=1` no-op, `contracts.py:53`) | live TP all-reduce — see §1.2 open question |

DP shapes are quoted at `SHARED_EXPERT_FILLER_DESIGN.md:176-178`; the TP8 column
is **DERIVED** from the column/row-parallel sharding rule (512 and 256 are both
clean multiples of the 128 quant block). **UNVERIFIED** against source.

**Consequence, and it is a real one:** the banked SHEXP design folds the shared
expert into "slot 32" of a 33-slot extended routed allocation
(`SHARED_EXPERT_FILLER_DESIGN.md:166-190`, reusing the M18 replica-slot path)
**because under DP the shared expert has exactly one routed expert's shape.**
Under TP8 it is 1/8 that width, so **the slot-32 mechanism does not port.** The
TP8 filler needs either its own descriptor slots for a distinct-shape MLP, or —
per the sharpened Option A — a **replicated** (unsharded) shared expert kept
outside the mega entirely (§4).

### 2.4 Bucket / T: the other layout change

`M15_PREFILL_B4096_BUCKET` pins `tokens_per_rank = maxtok = 4096`
(`m15_contracts.py:599-600`), and `validate_m15_selection` refuses anything else
(`m15_vllm.py:88-96`, blocker `M15-BUCKET-001`). Under TP8 the batch is
**global**, not per-rank: at AMD's `--max-num-batched-tokens 16384` a full
prefill chunk is 16,384 rows, all visible to every rank. So:

- a new TP8 bucket with `maxtok = 16384` is required (16,384 is a power of two,
  satisfying `m15_contracts.py:1235`);
- the sizing constraints that exist only to stage an all2all —
  `world*maxtok + K0P6_M15_SLABS <= t_loc_max` (`:1243`) and
  `maxtok <= t_loc_max // 8` (`:1202`) — are **dead** under TP8 (no dispatch),
  and must be re-derived rather than scaled, or they will demand an 8x-too-large
  staging region for traffic that no longer exists.

**Sized:** if §2.2 holds, routed-weight work is ~0 h. Shared-expert descriptor +
bind path: **M, 10-14 h**. TP8 bucket + contract re-derivation: **M, 12-16 h**
(most of it is deleting all2all-shaped constraints safely, with negative
controls).

---

## 3. KERNEL ENTRY

### 3.1 How the DP swap works today

1. **Install.** The campaign runs, inside the container and before `vllm serve`,
   `python /packet/shim/pf4h_integration/apply.py --integration-mode m15
   --m15-deploy-sources /m15src --dhk-root /dhk --apply`
   (`campaign_v5/run_m15_campaign_eplb_v5.sh:621-645`), then
   `coverage_patch.py`, then `m23_patch.py`, then optionally `rr_patch.py`
   (`:817-830`). Mounts at `:874-889`; the whole chain is withheld from every
   `native_*` arm (`:634-637`).
   **UNVERIFIED (apply.py internals — the node dropped before I could read it).**
   Verify: `sed -n 1,200p ~/pf4h_vllm_20260729/*/shim/pf4h_integration/apply.py`.
2. **Selection.** `validate_m15_selection` / `validate_full_selection`
   (`m15_vllm.py:62-104`, `vllm_full.py:68-186`) refuse anything but the frozen
   DeepSeek-R1 EP8 block-fp8 config, and emit `M15_SELECTION_RECEIPT`.
3. **Wrap.** `wrap_mori_prepare_finalize` (`m15_vllm.py:473-476`) wraps the live
   stock `MoriPrepareAndFinalize` in `M15PrepareAndFinalize`, a
   `mk.FusedMoEPrepareAndFinalizeModular`. `post_init_setup` (`:182-212`) binds
   the process runtime and the per-layer descriptor.
4. **Per-step.** `prepare()` (`:215-270`) runs `_eligible()`, records the three
   input pointers and returns *without* launching; `M15Experts.apply`
   (`:389-470`) binds weights; `finalize()` (`:272-336`) validates the output
   tensor, calls `bind_dynamic(...)` then `runtime.launch(record)` — **one
   megakernel launch replaces six pinned launches**, driven by a device-resident
   descriptor.
5. **Graph.** The launch is capture-safe; descriptor writes made during capture
   are published by `commit_captured_descriptors()` (`:240-241`).

### 3.2 Why none of it fires under TP8 — the blocker list

Each is a hard refusal in code we ship, with the site that raises it:

| # | blocker | site | survives TP8? |
|---|---|---|---|
| B1 | `stock.__class__.__name__ != "MoriPrepareAndFinalize"` | `vllm_full.py:222-227` | **NO** |
| B2 | `stock.mori_op is None` | `vllm_full.py:228-233` | **NO** |
| B3 | `PF4H-FULL-002`: `use_mori_kernels` false — evidence string is literally `--all2all-backend mori_high_throughput` | `vllm_full.py:98-105` | **NO** |
| B4 | `_attest_dp_batch`: needs `batch_descriptor.pf4h_exact_b4096 is True` **and** `num_tokens_across_dp_cpu == (4096,)*dp_size` | `vllm_full.py:294-325` | **NO** |
| B5 | `_eligible`: `a1` shape exactly `(4096, 7168)` | `vllm_full.py:377-400` | **NO** (§2.4) |
| B6 | `M15-BUCKET-001`: `T == MAXTOK == 4096` | `m15_vllm.py:88-96` | **NO** (§2.4) |
| B7 | M23 gate `data_parallel_size == 8` | `m23/m23_patch.py:429` | **NO** (§4) |
| B8 | contiguous-EP8 expert-map attestation | `vllm_full.py:328-373` | **likely YES** (§2.2) |
| B9 | AITER shuffled weight-layout gate | `m15_vllm.py:107-157` | **likely YES** (§2.2) |

B1-B3 are the structural ones. They say the DP swap hangs off the **modular
kernel** seam, which exists because DP8+EP engages an all2all backend. With
`dp_size = 1` under TP8+EP there is no all2all manager, so the MoE very likely
runs the **non-modular** path (`quant_method.apply` with an `expert_map`) and
`MoriPrepareAndFinalize` is never constructed — **the hook site we use today
does not exist on the target path.**
**UNVERIFIED, and it is the load-bearing unknown of the whole port.** Verify by
reading `use_all2all_kernels` (or its equivalent) in
`vllm/model_executor/layers/fused_moe/config.py` and the modular-vs-direct
branch in `layer.py` `FusedMoE.forward_impl`.

### 3.3 What a TP8 entry needs

The shim is reusable as a **chassis** (install mechanism, receipts, process
runtime, descriptor, launch, capture discipline) but **not as a hook**. Three
candidate sites:

| option | hook | pros | cons | size |
|---|---|---|---|---|
| **A** | `quant_method.apply` / the AITER fused-experts callable on the non-modular path | smallest surface; closest analogue to today's `M15Experts.apply` | owns only the routed GEMMs — cannot own the shared expert, cannot chunk the ARs, so it can never reach G-L2/G-L3 | M, 16-20 h |
| **B** | `DeepseekV2MoE.forward` (region hook) | owns gate + shared + routed + the pre-AR sum — the natural "MoE region" boundary of §1.1; can relocate the shared expert | does not own AR#1, so AR#1 cannot be chunked from here | **M-L, 24-32 h** |
| **C** | `DeepseekV2DecoderLayer.forward` (layer hook) | owns both ARs; the only option that can meter AR#1 *and* AR#2 and run a two-stream schedule across the whole layer | largest blast radius; must carry attention plumbing it does not change | L, 36-48 h |

**Recommendation: B now, C only if forced.** B is the smallest hook that reaches
the winning arm shape's compute claim and can host a two-stream filler over
AR#2; C is needed only if metering **AR#1** proves necessary after G-L0c. Build
B with the AR-issue site abstracted behind a callable so that promoting to C is
an edit rather than a rewrite.

Whichever is chosen, the entry needs its own selection/eligibility contract
(replacing B1-B7) and its own receipt family, because the analyzer currently
keys coverage off the DP-only `RAGGED_SEAL_RECEIPT` (§4).

---

## 4. SEAL, COVERAGE, AND THE FILLER'S STREAM STORY

### 4.1 What the M23 seal does on the DP path

M23 exists to solve a problem that is **purely an artefact of DP lockstep**.
Under DP8 every rank runs its own batch, DP padding inflates all ranks to the
max, and the pre-M23 seal demanded all eight *original* counts be exactly 4096
(`M23_RAGGED_SEAL_DESIGN.md:98-112`) — a coincidence, not a regime, which is why
the mega sealed on **2% of in-bucket steps** and ran **eagerly** on the rest
(`:35`, `:41-48`). M23 replaces that with a **DP-unanimous** predicate carried on
row 4 of the existing DP all-reduce, so all eight ranks make the *same* graph
decision; a rank inside `execute_dummy_batch` contributes 0 precisely because a
split decision would deadlock the collective (`m23/m23_patch.py:440-459`).
Coverage went 2% -> ~99% (`DECOMP_RUNBOOK.md:18`).

### 4.2 Under TP8 the seal is not needed — but coverage still is

`_pf4h_m23_ragged_enabled` hard-gates on `data_parallel_size == 8`
(`m23/m23_patch.py:429`), so **M23 is inert under TP8**, and structurally so:
with `dp_size = 1` there is one batch, one decision, no peers to agree with, no
dummy ranks, no `execute_dummy_batch`. This matches the banked verdict that the
ragged seal and the whole M24 fill lever **die under TP8**
(`TP8_STATE.md:368-372`). **Do not port M23.**

What must be ported is everything the seal was *carrying alongside* the
unanimity bit:

1. **Shape admission.** Something must still decide, per step, whether this
   batch is one the mega may run. Under TP8 that is a *local* predicate on the
   step's token count and the layer's tensors — no collective needed. This is
   the replacement for B4/B5/B6.
2. **Coverage counters.** Non-negotiable: BENCHMARK_PROTOCOL.md:56-58 item 3
   requires proof the kernel ran the traffic, and it exists because the pre-M23
   era measured an inert kernel for weeks. `coverage_patch.py` supplies the
   per-step counters on the DP path. **UNVERIFIED (contents).** Verify:
   `sed -n 1,120p ~/eplb_campaign/coverage_patch.py`.
3. **An analyzer-visible receipt.** `analyze_campaign_v5.py:127` parses
   `RAGGED_SEAL_RECEIPT` and `:160-163` **FAILs the cell** on a missing or
   below-threshold receipt. A TP8 arm emitting no such line is scored as a
   failure regardless of its throughput. So a TP8 arm needs either an
   equivalently-named receipt or an analyzer branch — a small but *blocking*
   harness item (§6).

### 4.3 The graph-capture question that replaces it

Under TP8 the seal's other job — keeping the mega and the captured graph in
agreement — becomes a **shape-bucket** question. A 16,384-token prefill chunk is
far above any full-cudagraph capture size, so it runs through the piecewise
compiled path, and the mega must be launchable from inside that region at a
static shape per compile range. **UNVERIFIED**: the resolved capture sizes and
compile ranges for the vendor arm. Verify:
`grep -iE "cudagraph|compile_range|capture" ~/20260818_m15_campaign_b0v5/pair_01/*native_tuned_tp*/server_config_dump.txt`.

### 4.4 The filler: stream, priority, and where the shared expert moves

**§1.5 amended.** §1.5 said a monolithic AR leaves no dependency-free
co-resident work. That holds for *sharded* shared-expert variants, and there is
exactly one way out of it, which is the sharpened Option A:

> **Replicate the shared expert.** If every rank computes the *full-width*
> shared MLP, its output is already identical on all ranks and needs **no
> reduction**. It can therefore be added **after** AR#2 instead of before it:
> `out = AR(routed_partial) + shared_full` is arithmetically identical to
> `AR(routed_partial + shared_shard)`. That single change takes the shared
> expert **out of AR#2's dependency cone**: its input (post-AR#1, post-norm
> hidden) is ready before AR#2 starts and its output is not consumed until
> after AR#2 ends, so its whole execution fits in the AR shadow **with no
> chunking of the collective at all.**

That is why Option A can be a two-stream host-side change with **zero in-kernel
fusion**: stream 1 issues the RCCL all-reduce, stream 2 runs the replicated
shared-expert GEMMs, one event joins them, and the add happens after the join.
It is also consistent with the new gate result (the AR loses 0.00% beside a
full-GPU MFMA kernel; 44-53% of the exposed AR is absorbable for free), which
removes the co-residency-tax objection that would otherwise have sunk it.

**But replication is a work-for-latency trade, and the sizing must gate it.**
Per rank per MoE layer at a 16,384-token step (all **DERIVED**, arithmetic shown
so it can be corrected):

| quantity | value |
|---|---|
| routed experts (16,384 assigned rows/rank x 88.1 MFLOP/row) | ~1,443 GFLOP |
| shared expert, TP8-**sharded** (`2*16384*7168*512 + 2*16384*256*7168`) | ~180 GFLOP (12.5% of routed) |
| shared expert, **replicated** (`2*16384*7168*4096 + 2*16384*2048*7168`) | ~1,443 GFLOP — **8x the sharded cost, equal to the entire routed load** |
| extra weight footprint if replicated | ~44 MB/layer x 58 = **~2.2 GB/rank**, out of the KV pool at `--gpu-memory-utilization 0.9` |

Let `S` = sharded shared-expert time (~0.5 ms/layer if the routed region is
3.5-4.5 ms), `R = 8S` (~4 ms) the replicated time, `W` = exposed AR window
(~2.8 ms/layer, `TP8_STATE.md:115-122`, **never profiler-confirmed**).

> **Decision rule:** the replicated filler gains `S` per layer **iff `R <= W`**.
> Overflow does not vanish — it lands back on the critical path, so the net is
> `S - max(0, R - W)`. On today's DERIVED numbers `R ~ 4.0 ms > W ~ 2.8 ms`, so
> it would **lose ~0.7 ms/layer**. It turns positive if the effective MFMA rate
> is ~1.4x my estimate, or if `W` is larger than the derived 2.8 ms, or if AR#2
> is metered so that `W` grows.

**Verdict on Option A: mechanism ACCEPTED, sizing NOT YET CLEARED.** The
dependency-cone argument is sound and is the genuine unlock — it is the only
shape that hides the shared expert without touching the collective. Its
viability turns on two numbers we have never measured on the real arm, and both
are cheap: (i) `W`, the per-layer exposed AR, from **G-L0c** (one profiled
native TP8 step, already the first item on the ladder and already flagged as an
honesty blocker); (ii) `S` and `R`, from a standalone timing of the shared-expert
GEMMs at both widths. **Measure `W`, `S`, `R` before building Option A** — the
build is otherwise a coin flip on an inequality we can settle in under an hour
of node time.

If `R > W` holds, the fallback is not "no filler": it is to **meter AR#2** so a
*sharded* shared expert (only `S` of work, no replication cost, no extra 2.2 GB)
hides in the chunk shadow — which is the G-L3 mechanism, and needs the Option-B
region hook of §3.3. Recorded sequencing consequence: **G-L2 and G-L3 are not
independent rungs.** Either replication clears `R <= W` (G-L2 standalone, no
metering), or it does not and G-L3's metering becomes a *prerequisite* for G-L2
rather than a follow-on.

**Priority story, either way:** the shared-expert tiles are routing-independent,
so they are the correct work-queue fallback whenever routed tiles starve — the
pipeline fill/drain smoother. Under Option A that is a host-side stream
priority (comm stream must not be starved of CUs by the filler; the 0.00%
co-residency result says it currently is not, at full-GPU MFMA occupancy).

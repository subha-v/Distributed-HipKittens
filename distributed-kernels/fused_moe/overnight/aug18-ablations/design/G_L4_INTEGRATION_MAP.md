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

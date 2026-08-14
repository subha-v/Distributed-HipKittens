# T1: the Megatron/Primus training swap — seam contract and design (2026-08-14 ~23:20 UTC)

Source-level inventory of rocm/primus:v26.5 (vendored Megatron-Core +
Primus patches + Primus-Turbo), extracted to establish where our megakernel
plugs into the training loop. File:line citations refer to the container
tree (copies banked on the node at ~/t1_seam_src).

## 1. The swap boundary (confirmed, with an AMD precedent)

`MoELayer.forward` (megatron/core/transformer/moe/moe_layer.py:494-535)
decomposes as: shared_experts_compute → route → preprocess → dispatch →
routed_experts_compute → combine → postprocess. The clean boundary replaces
preprocess..combine: take `(hidden_states [s=4096,b=1,h] bf16,
probs [tokens,256] fp32 sparse, routing_map [tokens,256] bool)`, return
`output [s,b,h]`; `mlp_bias` is always None (asserted); the shared expert
is computed OUTSIDE and added in postprocess — untouched.

**AMD ships the precedent unwired**: `primus_turbo ops/moe/mega_moe_fused.py`
(`MegaMoEFusedFunction`) is a fully fused EP MoE fwd+bwd (FlyDSL, symmetric
buffer: dispatch push + grouped L1 GEMM + SwiGLU + grouped L2 GEMM +
combine push; backward via dispatch<->combine duality returning
dx/grad_topk_weights/dW1/dW2). Signature `(x [T,h] bf16, topk_idx [T,K]
i64, topk_weights [T,K] fp32, w1 [G,2I,K], w2 [G,N,I], group)` — OUR seam
exactly. It is enabled by NO flag, exported nowhere, referenced nowhere:
AMD's in-development megakernel. Plan: wire it as an optional benchmark arm
(time-boxed — unwired may mean broken); wiring it exercises the identical
swap machinery we need for our own kernel.

## 2. What actually runs under the T0 flags (dead-code map)

- use_turbo_deepep=true force-swaps the dispatcher to
  `PrimusTurboDeepEPTokenDispatcher` (dispatcher patches set
  moe_token_dispatcher_type="flex" + rebind); stock MoEFlexTokenDispatcher
  / fused_a2a.py are dead code in our config.
- Router: `PrimusTopKRouter` + turbo `fused_group_topk_routing_with_aux_score`
  (an autograd.Function with hand-written backward).
- `force_load_balancing` replaces router logits with random draws via
  `RandomSTE` **from the expert-parallel RNG tracker** (straight-through
  grads); aux losses are still computed on the random logits and their
  grads ride on `probs` via `MoEAuxLossAutoScaler`.
- Probs are applied INSIDE the experts at the activation (swiglu(x) *
  permuted_probs); combine is an UNWEIGHTED sum. (Σ_k w_k f_k(x) is linear
  in w, so our kernel's combine-side weighting is numerically equivalent.)
- sync-free stage 1 does NOT statically shape the dispatch (worst-case
  allocation needs stage>1); a host sync on tokens_per_expert remains in
  the legacy-GEMM arm.
- Weights: legacy arm `weight1 [h, E*2I]` / `weight2 [E*I, h]` bf16 (gmm
  trans_b=False, BF16-ONLY — fp8 never touches this GEMM). Turbo arm
  consolidates to `weights [E_local, 2I, h]` / `[E_local, h, I]` bf16
  (trans_b=True) — the SAME per-expert orientation as our kernel's w13/w2
  contract. FP8 hybrid+blockwise: turbo quantizes weight/weightT once per
  microbatch-1 into cached fp8 buffers, 128x128 weight blocks + 1x128
  activation scales, E4M3 fwd / E5M2 dgrad — the same scaling-granularity
  family as our kernel's contract.
- turbo grouped GEMM and legacy grouped GEMM are MUTUALLY EXCLUSIVE
  (validation refuses the pair — cost us the first smoke).

## 3. T1 module design

A `K0MegaMoEFunction(torch.autograd.Function)` behind
`K0_TRAINING_MOE=mega` env, installed by a Primus-style patch at the
MoELayer seam:

- **Forward**: flatten hidden to [4096, 7168]; derive `topk_ids [T,8]` /
  `topk_weights [T,8]` from routing_map/probs (the dispatch_preprocess
  conversion is the precedent); run OUR fused kernel; return [s,b,h].
- **Weights**: bf16 masters live in the experts module (turbo layout
  matches). Per-step producer: fused 128x128-blockscale FP8 quant + AITER
  shuffle of the 4-layer proxy's expert weights (small new device op; the
  harness's quant+shuffle transform is the reference). Cache invalidated
  every optimizer step (vs serving's load-time-once).
- **Backward (T1 = recompute-through-stock)**: save hidden, probs,
  routing_map + BOTH CUDA RNG tracker states (RandomSTE forks the EP
  tracker — the moe_layer_recompute/tensor_parallel.checkpoint path is the
  working precedent for save/restore). Backward re-runs the STOCK expert
  path under enable_grad and backprops through it, returning grads for
  hidden AND probs — this preserves the aux-loss grad and the
  straight-through router grads for free. Numerics caveat (accepted for
  T1): grads are computed at the stock forward's values, our kernel's fp8
  forward differs at fp8 rounding; the gate is loss/grad-norm agreement
  within the FP8-arm envelope, and the honest perf claim in T1 is the
  FORWARD region time + end-to-end with recompute cost reported.
- **Timing**: wrap in torch.profiler record_function("k0_mega_fwd") the way
  mega_moe does; report per-layer fwd/bwd from profiler + Megatron timer
  buckets — the MoK-blog table format.

## 4. T2 direction (the real program)

The mega_moe backward decomposition (L2 dgrad → SwiGLU^T → dW2 → L1
dgrad-combine → dW1) maps onto our kernel's phases with dispatch<->combine
transposed. Our additions over AMD's design: wgrad (dW1/dW2) as FILLER
COMPUTE inside dgrad's comm waits (Megatron already exposes split
`backward_dw()` scheduling — moe_layer.py:557-563 — so deferred-wgrad is a
first-class citizen), fp8-on-wire combine (MORI's measured 366→642 GB/s
lever), source-side fp32 pre-reduce (MoonEP), shared-expert folding as
in-kernel filler, epoch pipelining across microbatches. BF16 variant of
the kernel is needed to match the MoK blog's BF16 rows (our current kernel
is fp8-blockscale-only); the fp8 rows are our exact contract already.

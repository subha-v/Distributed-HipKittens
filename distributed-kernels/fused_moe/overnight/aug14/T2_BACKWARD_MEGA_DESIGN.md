# T2: the backward megakernel (design spec, 2026-08-15 ~00:00 UTC)

Goal: a fused MoE BACKWARD sibling of the m15 mega that beats the AMD
production training path (turbo + DeepEP + turbo grouped GEMM — measured
T0 smoke: 23.5k tok/s/GPU, ~1.39 s/iter; vanilla timer buckets: fwd 561 ms
/ bwd 1,008 ms per iteration) by overlapping what their stack serializes.
The structural edge: **wgrad has zero communication dependency** — it is
filler compute for every comm-wait window in the backward, and Megatron
already treats deferred wgrad as a first-class citizen (split
`backward_dw()`), so the framework seam is ready.

## 1. The math, with the weighting convention settled

Forward (per token t, k in topk): y_t = Σ_k w_tk · W2_e · swiglu(W13_e · x_t),
z = W13 x (z = [g;u], swiglu(z) = silu(g)⊙u), h = swiglu(z).
Our kernel applies w at COMBINE (serving M8 convention); h is unweighted.

Backward given dY, with **dY dispatched UNWEIGHTED** (mirror of forward):

- dH2_row = W2_e^T · dY_row                     (L2 dgrad, per dispatched row)
- dZ_row  = swiglu'(z_row) ⊙ dH2_row            (needs z saved from forward)
    dZ_g = dH2 ⊙ u ⊙ silu'(g);  dZ_u = dH2 ⊙ silu(g)
- dX_t    = Σ_k w_tk · (W13_e^T · dZ_row)       (L1 dgrad; **w folded at the
            dX EPILOGUE via the saved plan's swt — NOT at M8 combine**: the
            phase map showed M1's match_any dedup makes receive rows
            (token,dest) aggregates of up to 8 experts with different
            weights, so per-slot weighting at combine cannot reconstruct
            Σ w_e dX_e. The epilogue is per sorted row = per (token,expert),
            exactly where the forward applies w today; M8 reuses
            byte-for-byte.)
- dW2_e  += Σ_rows w_row · dY_row ⊗ h_row       (LOCAL, no comm — FILLER)
- dW1_e  += Σ_rows w_row · dZ_row ⊗ x_row       (LOCAL, no comm — FILLER)
- dw_tk   = dot(dH2_row, h_row)                 (cheap row-dot; UNWEIGHTED
            dY makes this exact with no division)

w_row is available locally: the forward dispatch already ships per-row
weights (the my_wgt path); the save buffer keeps them (the
dispatch_weights_in_buf precedent from AMD's own mega_moe backward).

The probs/aux-loss grad path: dw_tk feeds back through the dispatcher's
weight-gather transpose to `probs` — required for aux losses and the
straight-through router grads (T1 seam inventory).

## 2. Kernel architecture — a sibling TU reusing the forward's transports

New translation unit next to the m15 body (working name
`k0pf6gm_device_tile_t2b.hip`), same grid (256x256), same epoch protocol:

- **M0/M1 (dispatch)**: byte-portable from forward — the M1 path already
  quantizes bf16→fp8+per-group scales in flight (8,064-byte wire rows), so
  fp8-on-wire dY dispatch is byte-identical MECHANICS, purely a precision
  decision. Critically, backward REUSES THE FORWARD'S SAVED SORT PLAN
  (sti/swt/pull_ptr/pull_src/pull_stage/tile_desc/chunk fills): the
  forward's reserve_row and scatter cursors are atomics (nondeterministic),
  so re-derivation is forbidden — and reuse deletes M0.5/M3–M5 from the
  backward entirely (dY rows land in the saved row assignments
  deterministically; no reservation atomics at all).
- **GEMM phases**: reuse the NT grouped-GEMM bodies with swapped dimension
  constants against **pre-transposed fp8 weight buffers** (per-step W2T
  [E,2048,7168] and W13T [E,7168,4096]; AITER preshuffle at those shapes,
  128x128 blockscale grids transpose exactly). The geometry mapping is
  PERFECT: dH2 (N=2048, K=7168) is phase-1's exact chunk geometry (8x256);
  dX (N=7168, K=4096) is phase-2's exact nc geometry (16x448, K-groups
  16→32, DQ width doubled). Phase-1's gate/up paired epilogue is exactly
  where swiglu' emits (dgate,dup) from one GEMM + the saved z, quantized
  to dZq [rowcap,4096] in W13-column order — precisely the K order the dX
  GEMM consumes. The backward TU gets its own constants header (the n2
  constants are W13/W2-specific). dw_k = dot(dH2_row, act_z_row) computed
  in the dH2 epilogue where dH2 is live in registers.
- **M7/M8 (combine)**: byte-portable — dX rows remote-accumulate to token
  owners with w folded at the accumulate (exactly where serving applied w).
- **dw_k row-dots**: computed where dH2 exists (GEMM1's epilogue), written
  to a local [T, K] fp32 buffer, combined alongside dX (dw travels with
  the row's combine packet or a second small transport).
- **WGRAD FILLER**: the M20 prefetch engine proved the carve mechanic —
  the last kPF service CTAs run an independent duty loop during the
  M1..M8 windows. Here the duty is dW2/dW1 tile accumulation from the save
  buffers into local fp32 accumulators: pure local GEMM tiles with total
  scheduling freedom. Priority rule: filler yields to combine batches
  (combine is latency-critical; wgrad is not). Any wgrad tiles unfinished
  at M8's end run in a short tail — the kernel never exits with partial
  accumulators unaccounted.

## 3. Forward-side save hooks (the T1 forward swap grows two outputs)

Per the phase map (line-cited in T2_PHASE_MAP.md):
- x_rows fp8 + scales: ALREADY persisted as a_dst (slot 8) + sc_stage (9)
  for the whole epoch — per-layer persistence is purely per-layer
  descriptor instances pointing 8/9 at per-layer arenas (the M20
  per-layer-descriptor precedent; ZERO device-code change).
- act(z): already exists quantized as A2q + DQ2 (slots 27/28) — keep per
  layer; feeds dW2 and dw_k directly.
- z (g,u): register-only today; add a mirrored quant-store in P1's
  epilogue (the amax machinery is already there) → new slots Zq
  [rowcap,4096] fp8 + DQZ [rowcap,32], keyed by sorted row (valid because
  the sort plan is saved).
- The sort plan itself: sti/swt/sei/nvi/tile_desc/pull_ptr/pull_src/
  pull_stage/chunk fills (~5 MB/layer) — mandatory save (atomics make
  re-derivation nondeterministic).
Per-microbatch lifetime (no PP): 4 layers x (a_dst + sc + A2q/DQ2 +
Zq/DQZ + plan) — low-GB total in 256 GB HBM.

## 4. Producer extensions

weight_producer.py grows: per step also emit W13T/W2T fp8+scales
(transpose-then-quantize per 128x128 block so scale blocks stay square;
scales transpose with the blocks). Pointer-stable like the rest.

## 5. Why this beats turbo+DeepEP+turbo-GG (the Amdahl case)

Their backward per MoE layer: dispatch-dY a2a → dgrad GEMMs → combine-dX
a2a, then wgrad GEMMs (sequential; sync-free stages pipeline some
launches, DBO unfinished per AMD's own blog). Ours: dgrad GEMMs run under
the same roof as the transports (the m15 fusion win, ~1.3x at balanced),
AND wgrad (≈ half the backward's FLOPs) disappears into comm/idle windows
instead of extending the critical path. Even at conservative overlap
efficiency the backward-region target is ~1.5x+ over their best arm;
end-to-end, backward is 62% of step time (T0 vanilla buckets), so the
prize is large. Falsifiable at the gate ladder — no number is claimed
until the harness measures it.

## 6. Gate ladder

1. `bwd_gate` fixtures: dense torch-autograd reference (world-8 fixture
   dump; balanced + Zipf-skewed routing) — being built now.
2. Kernel gates in the primus container via the k0_train no-MoRI runtime
   (torchrun-8): y/dX/dW13/dW2/dw_topk vs reference at fp8 tolerances;
   pperr 0; epoch protocol clean across ring parities.
3. Perf ladder vs the T0 baselines' measured per-layer bwd time (Megatron
   timer buckets + record_function attribution), balanced routing first
   (the training regime), then the T3 skewed-router regime.

## 7. Scope notes

- BF16 kernel variants (fwd + bwd) are needed for the MoK-blog BF16 rows;
  the fp8 rows are our exact contract already. BF16 = the same TUs with
  the quant/scale path compiled out — schedule after fp8 correctness.
- MTP/latent-projection/shared-expert-gate variants: out of scope for the
  proxy; shared-expert FOLDING (fwd + bwd filler) is a fast-follow lever
  once the routed path is green.

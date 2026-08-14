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
            combine**, exactly mirroring forward's combine)
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

- **M0/M1 (dispatch)**: byte-portable from forward — same routing, same
  LL128 push, but the payload is dY rows. Two options: bf16 rows (stride
  change: 7168x2B + no scales) or fp8-on-wire (quantize dY per row-group at
  the source — MORI measured 366→642 GB/s effective from halving combine
  bytes; same lever applies to our dY dispatch). Start bf16 (correctness),
  add fp8-on-wire as the measured lever.
- **GEMM phases**: reuse the NT grouped-GEMM bodies VERBATIM against
  **pre-transposed fp8 weight buffers** the producer already refreshes per
  step: W2T [E, I, H]-shaped view for dH2 = W2^T dY, W13T for dX rows.
  (The turbo grouped linear caches weight AND weightT fp8 copies per step —
  same trick, +42 MB/expert, trivial at the 4-layer proxy.) SwiGLU' is a
  fused elementwise between the two GEMMs, reading saved z.
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

The forward mega already materializes both saved quantities:
- x_rows: the unpacked fp8 expert input rows (a_dst) + scales (sc_dst) —
  today reused per layer; T2 adds per-layer persistent save buffers
  (descriptor slots, [t_loc_max, H] u8 + scales).
- z: GEMM1's output before activation — today consumed in registers/LDS;
  T2 streams it to a save buffer [t_loc_max, 2I] bf16 during the GEMM1
  epilogue (bandwidth cost ~1 extra write of z; the recompute alternative
  — re-run GEMM1 in backward — is the fallback if the write measures hot).
Per-microbatch lifetime (no PP: fwd microbatch then bwd immediately);
sizing at the proxy: 4 layers x (x_rows fp8 + z bf16) — low-GB total,
fine in 256 GB HBM. Also saved: per-row w and row->token/k maps (already
in the forward's sti/swt bookkeeping).

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

# T2B milestone: the backward megakernel runs and is numerically certified
# (2026-08-15 ~01:15 UTC — ~4.5 hours after the training pivot)

## What passed tonight (all artifacts on the node + both repos)

1. **T0 baselines FINAL** (t0full_0814T2314; <0.4% repeat spread):
   vanilla 19,496 / turbo+DeepEP 22,244 / **turbo_gg 23,644 tok/s/GPU at
   1,386 ms/iter (fwd 406, bwd 871)** — the bar.
2. **smoke0 PASS**: the forward mega runs under the from-scratch no-MoRI
   runtime (HIP-IPC symmetric heap + ctypes launcher) in rocm/primus:v26.5 —
   8 ranks, 16 launches each, pperr clean, offsets attested.  Three real
   bugs found and fixed on the way, each a lesson worth keeping:
   - pip-ROCm torch bundles its own libamdhip64; dlopen-by-soname loads a
     SECOND runtime whose module registry the torch context never sees →
     resolve the lib from /proc/self/maps;
   - ctypes c_char arrays truncate at the first NUL → IPC handles must be
     c_ubyte;
   - NCCL-only process groups cannot all_gather CPU tensors → exchange the
     64-byte handles on device.
3. **T2B_GATE PASS**: forward(save-z) → backward chain vs the dense
   torch-autograd reference (fixtures, balanced routing, full geometry):
   - FWD y:  rel 7.5e-2, cos ≈ 1.000, pperr 0
   - BWD dX: rel 7.22e-2, cos ≈ 1.000, pperr 0 — **uniform to 4 digits
     across all 8 ranks** (the signature of a noise floor, not a defect)
   - dw_topk (saved-plan mapping): rel 7.7e-2, cos 0.997
   - Weight-quant-aware reference: error drops to 5.4e-2 → the residual is
     ACTIVATION wire-quant (x/dY/act(z)/dZ per-128 fp8) + bf16 combine
     accumulation; the serving-proven forward exhibits the identical floor
     (7.5e-2) on the same reference → **the backward is at the same
     numeric contract as the kernel that passed the full serving ladder.**
   - Follow-up (tightener, not blocker): a full-chain fake-quant reference
     to bound the kernel at the ~1e-2 class.

## What the backward mega is, in one paragraph

A sibling TU of the forward (k0pf6gm_device_tile_t2b.hip, 191.5 KB gfx950,
zero VGPR spills): M1b dispatches dY by the FORWARD'S SAVED PLAN (the
scatter atomics are nondeterministic, so the plan is reused — which also
deletes M0.5/M3–M5 entirely), phase-1b computes dH2 = W2T·dYq on the
forward phase-1 chassis with a swiglu' epilogue that emits both dZ halves
and the dw row-dots, phase-2b computes dX = W13T·dZq on the phase-2
chassis with the topk weight applied per sorted row in the epilogue (the
match_any dedup forbids weighting at combine) and the exp_21/24
remote-accumulate transport verbatim, then M8/M9 byte-identical.  Weights
are per-step pre-transposed fp8 copies (the turbo weightT trick); the
forward grew a compile-gated save-z arm (m15sz, .text-identical default).

## Next (the performance ladder)

1. Wgrad filler on carved service CTAs (dW slots already in the ABI and
   bound by the gate) — the structural edge over everything AMD ships.
2. Timing harness: per-launch fwd/bwd mega times vs the T0 turbo_gg
   per-layer buckets; then the T1 Megatron swap for end-to-end tok/s/GPU.
3. Shared-expert filler arms; fp8-on-wire combine; skewed-fixture gate
   (fixtures already generated).

# T4 — the BF16 megakernel variant (the fair BF16-vs-BF16 row)

Goal: a bf16-weights, bf16-wire sibling of the m15/t2b pair so the
comparison table has an honest BF16 row (our fp8 stack vs production BF16
is precision-mismatched; production's shipped FP8 recipes diverge — 4/4
NaN, iters 23-37 — so the fp8 row is a robustness statement, not a race).

## What survives unchanged (the value we keep)

* The whole protocol skeleton: descriptor ABI, epoch lattice, M0 parity
  zeroing, saved-plan backward (M1b), M8 slab-certified combine, M9 retire,
  the parity slot ring, the side-stream wgrad architecture.
* Fusion: one kernel per direction — no permute/unpermute passes, no
  framework glue between dispatch/GEMM/combine.
* Comm/compute overlap: chunk-granular arrival -> expert GEMMs start early.
* The T1 host stack: hook, runtime, autograd Function, monitors, gates.
  The producer DELETES entirely (the kernel reads Megatron's bf16 weights
  in place: zero requantization, zero transposed copies) and wgrad loses
  its dequant step (activations saved in bf16).

## What changes

1. **Wire format**: LL128 payloads carry bf16 rows — 14,336 B/row vs
   8,064 B (fp8+scales).  Stage/slab arenas ~1.78x; heap per-slot grows
   accordingly (still comfortable).  The in-flight quantize in M1
   becomes a plain pack; sc_stage disappears.
2. **GEMM cores**: n2 phase bodies move from fp8 MFMA 16x16x128 +
   blockscale-fold to bf16 v_mfma_f32_16x16x32_bf16.  A-tile staging keeps
   the XOR-swizzled LDS pipeline but at 2 B/element; the per-K128-group
   scale application DELETES (no scales — a simpler epilogue than fp8).
   Weight addressing: plain row-major bf16 [E,2I,H]/[E,H,I] straight from
   Megatron storage (the legacy arm's native [E,H,2I] transposes once at
   init or the phases address the transpose — decide at implementation).
3. **Backward**: t2b phases get the same treatment; z-regeneration reads
   bf16 weights; dZ stays bf16 end-to-end (no DQdZ); the M8.5/side-stream
   wgrad consumes bf16 saved rows directly (no dequant pass, baddbmm as
   today).

## Projection (per layer-microbatch, T=4096)

GEMM flops fwd 2.9 TF: fp8 ~1.9 ms of the 4.75 -> bf16 ~3.9 =>
fwd ~6.8-7.5 ms.  dgrad: 7.65 -> ~11-12 ms.  Wire: dispatch bytes 1.78x
=> +~0.5-1 ms under overlap.  Per-layer total ~19 ms vs production BF16
~20-24 => the per-layer win NARROWS to ~1.1-1.25x but holds.
E2E: kernels ~610 + non-MoE 650 + NO producer + wgrad-s2 (no dequant,
~250 overlapped) + glue ~150 => ~1,400-1,500 ms => ~22-23.4k tok/s vs
23,644 — competitive; the AGPR wgrad rewrite and glue trims decide it.

## Order of work (reusing the T2B playbook)

1. Fork n2_phase1_gm bf16 core; standalone harness (wg_test pattern:
   validate MFMA layout + numerics vs torch, then perf).
2. Wire-format change behind K0P6_BF16_WIRE in the transport (stage
   sizing from the descriptor; keep fp8 build byte-identical when off).
3. m15b fwd TU -> gate vs bf16 dense reference (expect ~1e-3 rel — no
   quant floor).
4. t2b bf16 sibling -> chain gate -> e2e arm (producer bypassed).
5. The same v3b host stack runs it (wgrad_mode=stream, no refresh).

Estimate: 2-4 focused days; the fp8 arm keeps running in parallel as the
robustness/peak-perf row.

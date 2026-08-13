# exp_02 — host patch spec (REQUIRED before any mode-16 GPU run)

The mode-16 kernel computes parity strides from descriptor-carried shapes;
the host driver (`~/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/
mok_synthetic_prefill/`, `e004pf_k0pf_ab.py` + `run_campaign.sh`) must be
patched in exactly three places. Without this, parity-1 accesses land past
the allocated extent and the arm fails the gates for the wrong reason
(it would look like a protocol bug; it is an allocation bug).

## Patch 1 — allocate every parity buffer at double capacity

Applies to the `mps_mega` arm ONLY. When the packed mode is `16`:

| buffer | old extent | new extent | parity stride (elements) |
|---|---|---|---|
| `slots` (desc slot 61, `mori_t` symmetric) | `(8, 4096, 7168)` bf16 | `(2, 8, 4096, 7168)` | `8*4096*7168` |
| `row_ready` (slot 25, symmetric) | `(8, T_LOC_MAX)` u32 | `(2, 8, T_LOC_MAX)` | `8*T_LOC_MAX` |
| `pull_stage` (slot 12) | `(T*TOPK, 2)` i32 | `(2, T*TOPK, 2)` | `T*TOPK*2` |
| `pull_cnt` (slot 13) | `(T,)` i32 | `(2, T)` | `T` |
| `pull_ptr` (slot 18) | `(T+1,)` i32 | `(2, T+1)` | `T+1` |
| `pull_src` (slot 19) | `(T*TOPK, 2)` i32 | `(2, T*TOPK, 2)` | `T*TOPK*2` |
| `cand_out` (slot 32) | `(T, 7168)` bf16 | `(2, T, 7168)` | `T*7168` |

Zero-init everything (torch.zeros), exactly as today. Allocation cost ≈
+505 MiB/rank (almost all the slot doubling), trivially inside the 32 GiB
symmetric heap and 288 GiB HBM. Keep the SAME allocations for mode 12/14
campaigns (the second half is then never touched).

Implementation shape: read `cfg.mode` from `K0_MPS_CFG` once at setup; wrap
the seven allocations in `shape = (2, *old) if mode == 16 else old` and pass
the flat base pointer to the descriptor exactly as today. Do NOT change
descriptor slot indices, validation calls, or the 63-word ABI.

## Patch 2 — gate lag by one launch (correctness comparisons)

Today: after launch `i`, the `[MOK GATE]` compares `cand_out` against the
reference for epoch `i`'s inputs. Under mode 16, epoch `i`'s output completes
during launch `i+1` (parity `(i&1)`, second half if `i` odd... parity =
`i & 1`, half index `i & 1`).

Change: for mode 16, after launch `i >= 1`, validate the slice
`cand_out[(i-1) & 1]` against the reference for epoch `i-1`
(inputs are identical every epoch on this harness, so the reference text is
unchanged). Skip the `i == 0` comparison (nothing to check; log
`[MOK GATE] skipped (deferred warmup)`). The same lag applies to every
correctness consumer: `combine_bit_exact`, the `max_abs/relative` prints,
and the all-rank reduction.

## Patch 3 — NaN poison cadence, lag by one launch

`K0_MOK_POISON_OUT` currently poisons `cand_out` between epochs. Under mode
16: poison parity half `(i & 1)` **after launch i completes and before launch
i+1** — i.e., immediately before the launch that will write that half. The
lag-1 gate then validates freshly-written halves, and the zero-survivor
check keeps its exact strength (one stale/missing row yields 57,344
nonfinite, tripping three independent gates).

Soak: the 600-epoch soak needs NO structural change — launches still run
back-to-back; only the validation/poison indexing shifts. Keep
`K0_MPS_SOAK_ITERS = 600` exactly.

## Sanity checklist before the first GPU run (mode 16, C=16, g=353)

1. `K0_MPS_CFG="C=16,g=353,mode=16,flush_rows=16"` + binary built with
   `-DK0P6_MPS_ENABLE_TBO=1`; `[mori-jit] Compiling k0pf6gm_mps_mega` seen,
   new `.hsaco` mtime (`stat -L`), `K0P6_MPS_SRC_REV = 31`.
2. The seven allocations above are doubled (assert the buffer numel).
3. Gate parser tolerates the `[MOK GATE] skipped (deferred warmup)` line at
   iteration 0 so the first epoch is not misread as a failure.
4. `K0_MOK_POISON_OUT=1` unchanged (already default-on and forwarded).
5. First GPU contact = correctness + negative control + poison selftest
   (untimed), THEN the 600-epoch soak, THEN screens/campaigns.

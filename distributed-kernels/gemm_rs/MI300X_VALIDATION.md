# GEMM-RS MI300X — validation status and promotion gates

This page separates what is proven locally from what only the 8x MI300X/gfx942
node can prove. The scaffold machine has no ROCm/HIP toolchain and no AMD GPU;
no gfx942 compilation was faked. Every not-yet-run gate is explicitly labelled
`PENDING_GFX942_VALIDATION` with the exact command the later agent should run.

## Local gates — PASSED on this machine

Run from the repository root:

```bash
# Protocol/address/epoch/numerics simulations + the 20 static gates
python3 distributed-kernels/gemm_rs/gemm_rs_mi300x_static_checks.py \
  --amd-master /path/to/amd-master

# Same simulations alone
python3 distributed-kernels/gemm_rs/gemm_rs_mi300x_simulation.py

# Host ABI/config (strict C++20, both PGL subtrees)
make -C tests/unit/common/kernel_host_abi_mi300x clean all run

# Existing reusable-layer suites (untouched; must keep passing)
make -C tests/unit/common/types/global clean all run
make -C tests/unit/common/distributed clean all run
make -C tests/unit/common/iris_adapter clean all run
make -C tests/unit/common/kernel_host_abi clean all run
python3 distributed-kernels/common/check_port_invariants.py \
  --amd-master /path/to/amd-master
git diff --check
```

Last local run: all of the above PASSED, including the six scored-shape
address/dependency walks, 64-epoch skewed-order lifetime simulations, the
drop-publication / reroute-slot / drop-credit control models (bounded timeouts
with zero payload reads on failure paths), windowed bf16/fp32 numerics vs the
evaluator's oracle semantics, and the K-tail exactness identity.

## Gate M1 — gfx942 build — `PENDING_GFX942_VALIDATION`

```bash
cd <node>
hipcc -std=c++20 -O3 -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
  -ffast-math --offload-arch=gfx942 \
  -DTK_MODNAME=gemm_rs_mi300x \
  -I<repo>/include -I<repo>/include/pyutils \
  -c distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp
```

Requirements: package instantiates all seven tile specializations
(`32/256/32`, `64/64/64`, `128/256/32+K_TAIL`, `256/256/32` x2, `256/256/32+K_TAIL`,
generic `32/64/64` with runtime tails). The negative-control module is a
SEPARATE translation-unit product:

```bash
hipcc ... -DHK_GEMM_RS_MI300X_NEGATIVE_CONTROLS=1 \
  -DTK_MODNAME=gemm_rs_mi300x_control -c gemm_rs_mi300x.cpp
```

The static gates prove the control entry points cannot co-exist with the
production binding; confirm at link time the production module exports only
`gemm_rs_mi300x` and the control module only `gemm_rs_mi300x_control`.

## Gate M2 — ISA + resource inspection — `PENDING_GFX942_VALIDATION`

Per instantiation, record normalized ISA/disassembly, VGPR/AGPR/SGPR counts,
LDS (`<= 65536` required), scratch/spill counts (must be 0 spills for the
scored rows), and occupancy (`>= 1` CTA/CU at 512 threads required for the
progress-residency assumption in MI300X_DESIGN.md section 7a). Method:

```bash
hipcc ... --save-temps -c gemm_rs_mi300x.cpp
# inspect *.s / disassemble; resource flags per ROCm tooling
```

Verify: the emit path lowered to 16-byte stores (`flat_store_dwordx4`-class,
or buffer stores of 16 B), the release is directional (writeback + drain, no
invalidate), the consumer acquire is a pure acquire (no writeback), the
bounded polls are relaxed loads with `s_sleep`-class backoff, and the epoch
path is a device cell read-modify-write (no kernel-arg epoch).

## Gate M3 — world-8 correctness — `PENDING_GFX942_VALIDATION`

For every official correctness case and all six scored cases, on all 8 ranks:
`M % 8 == 0`, `K % 8 == 0`; compare against the evaluator's exact oracle
(`bf16(x@w.T) + bias` then reduce-scatter, allclose 1e-2 / 1e-2). Exercise the
table rows AND the generic path (e.g. M=64/N=2880, M=8192/N=8192/K=28672,
plus the smallest legal M).

## Gate M4 — negative controls on GPU — `PENDING_GFX942_VALIDATION`

Using the control module (`gemm_rs_mi300x_control`):

- `CTRL_DROP_PUBLICATION`: the affected output tile's consumers must hit the
  bounded timeout (`err bit 26`); host confirms zero payload reads followed
  (correctness of the error path, not of the output).
- `CTRL_DROP_CREDIT`: the suppressed source's epoch-2 producers must hit the
  credit timeout (`err bit 25`) and must not overwrite the slot.
- `CTRL_REROUTE_SLOT`: run must fail numerical correctness (wrong payload
  lands in the neighboring rank's slot while the true destination sees
  readiness).

## Gate M5 — 600-epoch skewed soak — `PENDING_GFX942_VALIDATION`

At least 600 changing-input eager launches with deliberate rank/process skew
and NO host reset: every device-derived epoch must advance exactly once per
launch (read the epoch cells back; assert `== n_calls`), every touched
ready/credit cell must read exactly the final epoch, and outputs must stay
correct per launch.

## Gate M6 — semantic twins — `PENDING_GFX942_VALIDATION`

Compare: (a) this port; (b) a manual raw-peer-pointer twin of the same
schedule; (c) the PGL+distributed-primitives port. Require identical payload /
dependency / arithmetic semantics, then identical resource tuples under the
same MFMA footprint (per VALIDATION.md promotion gates).

## Gate M7 — paired timing vs baselines — `PENDING_GFX942_VALIDATION`

Same-run, order-rotated, per-shape means vs:

1. the exact frozen RadeonFlow/rank-1 implementation;
2. GEMM + RCCL ReduceScatter;
3. a coarse-barrier control (keep the coarse variant on a table flag for
   this measurement only).

Report all six per-shape means and their geometric mean; compare against the
published SOL table (6.46 / 8.19 / 23.04 / 65.54 / 131.07 / 379.43 us) and a
measured physical gfx942 bound. Do not report only the largest shape; the
competition score is the geometric mean of means.

## Gate M8 — graph mode (supplementary) — `PENDING_GFX942_VALIDATION`

Valid only if the device-derived epoch visibly advances on every replay
(assert the epoch cells after N replays). The RadeonFlow stale host
`signal_val` failure mode is absent by construction; prove it with a capture
that changes inputs between replays.

## Known simplifications to revisit on the node (tuning experiments, in order)

1. **Producer/consumer split sweep**: `NUM_REDUCER_CTAS` over
   `{8, 16, 24, 32, 40, 48}` per shape against the initial RadeonFlow counts
   (32/48/48/48/32/8). The initial design's reducer tile sweep is untuned
   (e.g. shape 1 has 28 reducer tiles over 32 reducers).
2. **GEMM schedule upgrade path**: the port's mainloop is correctness-first
   HipKittens double-buffering with full barriers. The measured RadeonFlow/rank-1
   pipelines use single-buffer VGPR->LDS staging, `buffer_load` raw descriptors,
   `s_waitcnt`-counted pipelines and `sched_group_barrier` instruction
   interleaving. Upgrade per shape inside the preserved tile envelopes; verify
   ISA and resources at each step (Gate M2).
3. **Emit/reduce micro-tuning**: reducer tile ordering (col-major vs
   row-major sweep), source-issue order (loads may issue swizzled while
   arithmetic stays ascending), staging window size, and whether the 32-thread
   packet groups should cover two rows per pass on BN=256 rows.

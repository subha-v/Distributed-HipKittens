# GEMM-RS MI300X provenance

## 1. Frozen inputs

| donor | authoritative path | identity | role in this port |
|---|---|---|---|
| evaluator/spec | `auto-gpu-kernel/k2_mi300x_megakernel/inputs/specs/reference-kernels/gemm-rs/reference.py`, `task.py`, `task.yml` (amd-master) | read from `amd-master` checkout at `e0b1a315b304649a4502afa7c14cf443ef4fc225` | local-K = k/8; bias per-rank before reduction; output `[M/8, N]`; tolerance `allclose(1e-2, 1e-2)` |
| rank-1 MI300X source | `auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py` | SHA-256 `7940fcb81df06c1d8b1e1a77051f23c934149a688441ef48b2751b3f336f0dc5` (verified against working tree) | per-shape BM/BN/BK tile table, 512-thread geometry, tail policy (`EVEN_K`/`EVEN_N`), historical score `413.139` µs (context only) |
| RadeonFlow submitted kernel | `auto-gpu-kernel/k1_comm_overlap/experiments/exp_13_sota_baseline/src/perf_gemm.cc` lineage, audited at `.../exp_13_sota_baseline/src/perf_gemm_upstream.cc` | working-tree SHA-256 `5723311fda91c00063a8ff783bca8366f3ccc30c1ea1594f73e7e856063d9607`; last commit touching it `946ac7f73365b6b7a95cfcf4f5eba5ad5a80ee85` | 304-CTA persistent producer/reducer split; scored-shape `NUM_REDUCER_CTAS` = 32/48/48/48/32/8; stale `signal_val` anti-pattern |
| exp 23 (communication mechanisms) | `auto-gpu-kernel/k1_comm_overlap/experiments/exp_23_emit_timed/src/ladder_emit.cpp` | commit `182f3e269068d3809ebf9074395217351837aec5`, SHA-256 `70ce26dacaa2f97ff57284132e2e2b42ba80bcd7fd1fac20cecb0471de666d82` (locked in `dependencies.lock.json`) | dead-LDS staging + 16-byte packet emission; separated release/publication; F6 grouped release |
| exp 24 (reduction discipline) | `auto-gpu-kernel/k1_comm_overlap/experiments/exp_24_reduce/src/ladder_reduce.cpp` | same commit, SHA-256 `62d8d61358ea5a6b3d8ea98579f1a16cab8078d2659a5fabe3e573f9fd211eb4` (locked in `dependencies.lock.json`) | REDV=1: issue all eight source loads, consume source-ascending in FP32, single RNE bf16 pack |

## 2. What transferred, and what explicitly did not

Transferred (mechanisms only, no MI350X timing or codegen claims):

- direct peer emission from the GEMM epilogue at 16-byte packet width with the
  PGL address rule `peer_ptr = peer_base + (local_ptr - local_base)`;
- one grouped CTA release covering a tile's packets and its cheap completion
  cells; relaxed monotonic per-tile epochs;
- bounded relaxed polling into caller-owned result storage, success-only pure
  acquire before any payload read;
- REDV=1 multi-source load issue + ordered FP32 consumption + single RNE pack;
- device-derived per-CTA epochs (no host `signal_val`); directed per-tile
  retirement credits before slot reuse; fail-closed error reporting.

Not transferred:

- gfx950's 256x256 tile, 512-thread/160-KiB-LDS donor schedule (does not fit
  gfx942's 64 KiB LDS; not MI300X-tuned);
- RadeonFlow's write-local-then-pull topology, its dual-signal-row straddle
  hack, its `signal_val` host epoch, 8-byte peer stores, and its swizzled
  reduction arithmetic (we issue loads early but consume source-ascending);
- the rank-1 file's three-launch structure (GEMM + all-pairs barrier kernel +
  reduce) and its Triton codegen; historical `413.139 µs` is not a local
  denominator.

## 3. Claim boundary

This port has no GPU result. All correctness here is static construction: the
dependency graph, addressing, ordering, lifetime, error handling and launch
structure are explicit and checkable by `gemm_rs_mi300x_static_checks.py` and
`gemm_rs_mi300x_simulation.py`. RadeonFlow and the rank-1 file remain
competition/evidence donors only; "beat RadeonFlow / RCCL / SOL" statements are
`PENDING_GFX942_VALIDATION`.

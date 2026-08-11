# Fused MoE provenance

## Measured donor

The immutable source was introduced at `amd-master` commit
`c5a0aac71d92e229c9a1a5f1743f69e2d1279f56`:

`auto-gpu-kernel/k0_fused_moe/experiments/exp_64_m2_row_striping/source_snapshot/k0pf6gm_mega.hip`

SHA-256: `57152a087c62aee1ffaae464cd2972e321d5e283e0681bc5a248039a6dc1ae2c`.

At G=2/c4 it measured 7,248 µs against production 7,695 µs (0.9414x) across
five rotated processes. Correctness passed 8/8 in each run with `pperr=0`, the
negative control failed as required, and a 3,000-epoch soak was clean. Resources
were 438 VGPR total, 182 AGPR, 102 SGPR, zero spill/scratch, 120,488 B LDS, and
occupancy one.

The best configuration used the same exact source bytes at `amd-master` commit
`d34e5510c4672e998eb2e94672e46e2f784ca07f` with `K0P6GM_G=3` and c4. It
measured 6,919.8 µs against production 7,698.0 µs (0.89846x; interval
0.89760–0.90037), and 0.8955x in the 600-iteration soak. Correctness passed 8/8,
`pperr=0`, MoK passed, and the broken control failed. The object used 256 VGPR +
256 AGPR, 102 SGPR, 9 VGPR spills/40 B scratch, 155,432 B LDS, occupancy one;
all 20 scratch operations were outside both MFMA K loops.

## Port ledger

The first abstraction rewrite is `amd-master` commit
`feea09b267602d78f1e51f5cd2ffd38aa0f4a76c`. This port preserves routing,
descriptors, task/GEMM mappings, M2 row striping, local grid barriers, LL128
packet bytes/lane mapping, and the textual N2 hot loops. It replaces only M6/M7
arrivals/waits, M1 reservation/peer placement, M1/M7.5 release, monotonic
completion publications, M2 acquire, and M8 readiness/acquire through the thin
HipKittens adapter. Removing the M7.5 grid barrier is a separate algorithm and
is not part of this port.

The adapted M7.5 publisher is not schedule-identical to exp-62/exp-64. The
donor grid-strided rows over all threads, but one publishing thread's fence
cannot formally release another thread's relaxed flag store. This port assigns
one `bid mod gridDim.x` row stripe to each CTA leader; that leader performs its
own post-grid system release and all relaxed self/peer stores in the stripe.
The shape retains 256 release fences per rank and establishes the intended
edge, but serializes roughly `T_ext / 256` rows per leader. Treat it as an
unmeasured correctness-first schedule variant and benchmark it against both the
exact donor publisher and a same-thread parallel publication design.

This repository adds an explicit IRIS heap-descriptor ABI beyond that rewrite:
slot 55 is `K0P6_D_SYMMETRIC`, descriptor length is 56, and every peer address
is now derived from a host-snapshotted local heap base plus eight named peer
heap bases. This replaces the donor runtime's hidden peer-translation call and
is itself unmeasured. The descriptor ABI is fixed at 72 bytes, 8-byte alignment,
with the eight named heap bases beginning at byte offset 8. The LL128 row
primitive and bytes remain unchanged.

The port also closes three fail-closed gaps in the frozen source. M0 adds one
normal-path cumulative local-grid rendezvous so a retirement timeout converges
before any address reset. M2 routes both bounded wait failures through its
existing Pass-A rendezvous before any completion reread or payload load. Every
local-grid barrier is followed by a grid-uniform failure gate. A timeout is
terminal and requires coordinated all-rank protocol reinitialization; no
same-epoch retry/recovery protocol is claimed.

## Claim boundary

The adapted source—including the new IRIS peer-translation ABI, M0 rendezvous,
timeout convergence, and CTA-leader M7.5 publication—has only host/static
validation here. It has not been built,
ISA-gated, or timed on gfx950. The 6,919.8 µs result belongs only to the exact
donor source and measured G=3 environment; it is a target, not an inherited
result. No MI300X/gfx942 or production-vLLM end-to-end claim exists.

## MPS (minimum-progress specialization) sibling

Added 2026-08-10, additive; the parity port above is byte-untouched.

- `k0pf6gm_device_tile_mps.hip` derives textually from
  `k0pf6gm_device_tile.hip`. Byte-identical preserved regions (verified by
  scripted span diff at authoring time): M1 qpush, M2 unpack/histogram, M3–M5
  plan/scatter, M6 phase-1 call, M9 counted arrival/retirement. Replaced
  regions: phase-2 body include (vendored copy), the M6.9/M7 role partition,
  M7.6 service phase, M7.5/M8 mode dispatch. Descriptor ABI extended from 56 to
  63 words (slots 56..62 appended; slot 55 and words 0..54 untouched).
- `n2_phase2_gm_mps.cpp` is byte-derived from donor
  `n2_phase2_gm.cpp` sha256
  `7d8beb039b8224e614eacdbe9b34b21837095ffa6ec4842f08d72358b7b8e025` (the same
  hash the winner port consumes), with exactly two code deltas:
  `N2GM_TASK_START`/`N2GM_TASK_STRIDE` loop bounds and the
  `N2GM_TASK_DONE_DRAIN_HOOK` per-thread `s_waitcnt vmcnt(0)` before the
  task-end `__syncthreads()`. Both MFMA K-loops, all LDS layouts, the
  sched_group_barrier schedule, and the epilogue arithmetic are donor-verbatim
  (verified by diff at authoring time).
- `moe_mps_adapter.cuh`, `moe_host_abi.hpp` (additive MPS section), and
  `include/cdna4/ops/group/distributed/roles.cuh` are new, not ports.
- Review status: one adversarial read-only review was run at authoring
  time; its one CRITICAL finding (LDS flag-buffer overflow for
  `flush_rows > 32`) and one acquire-edge hardening (acq_rel RMW probes in the
  slice-group check) are fixed in the shipped text.
- Known watch item for the ISA gate: the packed role word (task start/stride)
  is loop-carried across the phase-2 mainloop. It is CT-uniform (SGPR, not
  VGPR), so the 256-ArchVGPR budget is structurally untouched, but the
  resource/ISA A/B against the parity port must confirm no ArchVGPR/AGPR/LDS
  movement before any parity claim.
- The MPS source is host/static only: not built, not ISA-gated, not corrected
  or timed on 8x gfx950. Its `C=0`, mode-0 configuration is intended to be a
  same-ABI parity control, not a claim of donor equivalence.

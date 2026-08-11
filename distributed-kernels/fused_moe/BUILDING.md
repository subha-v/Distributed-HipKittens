# PF6 build integration

The port is a genco/JIT source, not a pybind module. Build only after the
content checks in `dependencies.lock.json` pass.

Before launch, include `moe_host_abi.hpp` and call
`hk_moe::host_abi::snapshot_heap_descriptor(view)`. Upload that 72-byte host
POD to device-visible memory, represent the returned device address with
`hk_moe::host_abi::device_visible_descriptor_address`, then call
`append_symmetric_heap_descriptor` on the donor's exact 55-word host vector or
`patch_symmetric_heap_descriptor` on an exact 56-word vector. Run
`validate_descriptor` before uploading the vector. Allocation-adjusted bases
are wrong here: MoE translates many independent symmetric suballocations by
their offset from the shared heap base. These helpers do not allocate/upload
either object, validate a GPU address space, retain lifetime, or launch; those
steps remain the future runtime binding's explicit responsibility.

Set three source roots:

```text
DHK_ROOT       Distributed-HipKittens checkout
AMD_MASTER     checkout containing the pinned helper blobs
MORI_JIT_ROOT  the content-pinned MoRI `_jit-sources` root
```

The bare includes in `k0pf6gm_device_tile.hip` resolve from these directories:

```text
$AMD_MASTER/auto-gpu-kernel/k0_fused_moe/solution/hip/hkp
$AMD_MASTER/auto-gpu-kernel/k0_fused_moe/prefill_opt/kernels
$AMD_MASTER/auto-gpu-kernel/k0_fused_moe/solution/hip
$DHK_ROOT/distributed-kernels/fused_moe
```

The measured compile contract was C++20, gfx950, O3, fast math, CDNA4, MFMA
VGPR form, `K0P6GM_G=3`, and `N2GM_G=3`. A representative genco invocation is:

```bash
hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
  -I"$DHK_ROOT/include" \
  -I"$DHK_ROOT/distributed-kernels/fused_moe" \
  -I"$AMD_MASTER/auto-gpu-kernel/k0_fused_moe/solution/hip/hkp" \
  -I"$AMD_MASTER/auto-gpu-kernel/k0_fused_moe/prefill_opt/kernels" \
  -I"$AMD_MASTER/auto-gpu-kernel/k0_fused_moe/solution/hip" \
  -I"$MORI_JIT_ROOT" -I"$MORI_JIT_ROOT/include" -I"$MORI_JIT_ROOT/src" \
  "$DHK_ROOT/distributed-kernels/fused_moe/k0pf6gm_device_tile.hip" \
  -o k0pf6gm_device_tile.gfx950.hsaco
```

MoRI installations may also require their vendored spdlog/msgpack and MPI
include roots, as in the measured JIT environment. Because that MoRI package
commit and the container digest were not captured, this command documents the
integration surface but does not establish a reproducible or performance-
comparable build. Record their content hashes before the GPU gate run. MoRI is
not the peer-address backend in this port; it remains an include dependency of
the frozen LL128 helper only.

Use a fresh cache root for this source/configuration. Then verify resource
metadata, 96 MFMAs, no descriptor/synchronization/spill instructions inside
either K loop, exact LL128 lane/packet layout, world-8 correctness, broken
control, and the 600-replay soak before paired timing.

Profile M7.5 separately. The port's 256 CTA leaders each publish one row stripe
after a same-thread system release; this is formally ordered but not the
donor's all-thread schedule. Compare its tail time with the frozen donor and a
parallel design in which every publishing lane executes its own amortized
release before relaxed stores.

Treat every nonzero `pperr` as a terminal protocol failure. A retry requires a
coordinated world-8 reinitialization of the epoch cell, cumulative grid barrier,
all completion/counter words, and address-lifetime state; merely clearing the
error word is invalid because same-epoch publications from the failed attempt
can satisfy a new wait prematurely.

# MPS sibling (k0pf6gm_device_tile_mps.hip)

Same roots and flags as the parity port; add the sibling's own source name and
a fresh cache key (the runtime role/config is a descriptor word, so one HSACO
serves the whole `C × g × mode` sweep — do not let the JIT cache alias this
source against the parity port's object):

```bash
hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
  -I"$DHK_ROOT/include" \
  -I"$DHK_ROOT/distributed-kernels/fused_moe" \
  -I"$AMD_MASTER/auto-gpu-kernel/k0_fused_moe/solution/hip/hkp" \
  -I"$AMD_MASTER/auto-gpu-kernel/k0_fused_moe/prefill_opt/kernels" \
  -I"$AMD_MASTER/auto-gpu-kernel/k0_fused_moe/solution/hip" \
  -I"$MORI_JIT_ROOT" -I"$MORI_JIT_ROOT/include" -I"$MORI_JIT_ROOT/src" \
  "$DHK_ROOT/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" \
  -o k0pf6gm_device_tile_mps.gfx950.hsaco
```

Host bridge: word 55 via `append_symmetric_heap_descriptor` (or
`patch_symmetric_heap_descriptor`), then `append_mps_descriptor` /
`patch_mps_slots` + `validate_mps_descriptor` for slots 56..62; sizing via
`mps_queue_bytes(padmax)`, `mps_arrivals_bytes(t_ext)`,
`mps_pushed_bytes(t_ext)`, `mps_claim_bytes(t_ext)`, `mps_state_bytes()`,
`mps_slots_bytes(maxtok)`. The slots buffer must be symmetric (peer-written);
the queue/counter/state buffers are agent-local in practice. Config word via
`hk_moe::mps::encode_config(C, g, mode, flush_rows, pull_fallback, timestamps)`.

MPS-specific gates, in addition to everything above:

- Resource/ISA A/B against the parity port on both MFMA spans: 96 MFMAs per
  phase-1 K-loop; the task-done drain and hook land OUTSIDE both K-loops; the
  packed role word may live in SGPRs across the phase-2 task loop but must not
  move ArchVGPR/AGPR/spill/LDS metadata.
- Confirm the service path's wave-scope drain (`s_waitcnt vmcnt(0)` per lane
  before flush) and per-event agent release survived as expected; confirm
  `store_peer_packets` lowers to 16-byte stores.
- Run the mode ladder in order: mode 0 C-sweep (pure tax control: C,
  g=4, mode 0) → mode 1 (copy/layout without overlap) → mode 2
  (`C ∈ {4, 8, 16} × g ∈ {1, 2, 4}`, `flush_rows` default 16), each arm with
  world-8 correctness + negative control + 600-epoch soak before any timing.
- Timestamps (`cfg` bit 33) give first_ready / last_ready / queue_drain /
  m7_done / reduce_done per rank from `K0P6_D_MPS_STATE`; leave them off for
  production-shape timing, on for attribution runs.
- `pull_fallback` (cfg bit 32) isolates streamed readiness from push
  transport; expect it to bound the owner-poll gain before trusting mode 2's
  full win.

Timing itself — the MoK campaign, the `production` and `pf6gm_mega` reference
arms, the arm registration this sibling still needs, and the exact commands —
is specified in `BENCHMARKING.md`. Read it before running anything in the
"paired timing" step above; in particular the `K0_PF6GM_G` default trap, which
otherwise measures a G=3 candidate against a G=2 reference.

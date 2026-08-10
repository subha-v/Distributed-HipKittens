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

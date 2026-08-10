# GEMM-RS build integration

The source preserves the donor's pybind entry points but intentionally is not
named `kernel.cpp`, so the repository's auto-discovery will not ship an
ungated module.

A validation target should compile `gemm_rs_device_tile.cpp` with the same
include/link interface as the existing distributed kernels plus:

```text
--offload-arch=gfx950
-std=c++20 -O3 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
-DTK_MODNAME=<validation_module_name>
```

Build `gemm_rs_dprime_experimental.cpp` as a separate translation unit and
module; never link both files into one object. The selected two-launch source
exports `gemm_rs_device_tile` and `gemm_rs_device_tile_reduce`.

Before launch, include `gemm_rs_host_abi.hpp` and call
`hk_gemm_rs::host_abi::snapshot_allocation_descriptors(view, c_heap, signals)`.
It returns two `hk_gemm_rs::symmetric_descriptor` host PODs after independently
rebasing IRIS's peer heaps to the C and signal allocation anchors. Upload both
PODs to device-visible memory and pass their device addresses through the
existing `*_peers` arguments. Allocate and zero 8,256 signal words and 8,192
local epoch words per rank. The descriptor ABI is exactly 72 bytes, 8-byte
aligned, with the eight named peer bases beginning at byte offset 8. The helper
does not allocate or upload storage, manage its lifetime, or launch; those are
the future runtime binding's responsibilities.

Acceptance requires an unchanged 256-MFMA schedule, GEMM at most 232 VGPR and
102 SGPR, occupancy two, no scratch beyond the donor's 128 B, and no LDS beyond
160,000 B. Then run world-8 bit correctness, a dropped-peer control,
changing-input/skewed 600 replay, and same-run paired timing on gfx950.

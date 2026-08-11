# Distributed kernels

This tree keeps reusable distributed mechanisms separate from operator policy
and from performance evidence. IRIS owns symmetric allocation and peer mapping;
HipKittens owns peer-aware layouts and explicit device memory ordering.

| Path | Purpose | Status |
| --- | --- | --- |
| `common/` | Host-side IRIS-to-PGL bridge | Scaffolded and host-reviewable |
| `bf16_gemm/` | Original minimal IRIS/HipKittens example | Existing example; not the GEMM-RS performance donor |
| `gemm_rs/` | Best measured GEMM-RS donor plus HipKittens communication adapter | Port candidate; GPU parity and timing still required |
| `fused_moe/` | Best measured PF6 fused-MoE donor expressed through the common primitives, plus the additive COMET/MoK-style minimum-progress specialization sibling (`DESIGN_MPS.md`) | CDNA4 port candidate; GPU parity and timing still required |

Start with the repository-level
[distributed architecture](../docs/distributed/ARCHITECTURE.md) and
[primitive contracts](../docs/distributed/PRIMITIVES.md). The
[source audit](../docs/distributed/SOURCE_AUDIT.md) records branch heads and
donor selection, while the [validation matrix](../docs/distributed/VALIDATION.md)
tracks promotion gates. Each port directory
contains its exact donor commit, source hash, measured result, dependency
boundary, and remaining validation gates. A donor measurement is historical
evidence; it is not a performance claim for a rewritten source file.

## Buildable example

The CMake build discovers directories containing a `kernel.cpp`; today the
validated set is `bf16_gemm/`. `gemm_rs/` and `fused_moe/` are on an explicit
exclusion list, so even an intermediate `kernel.cpp` cannot silently enter an
`ALL` build. Promote either port by removing that exclusion only after its
architecture-specific correctness and performance gates pass.

The build consumes the header-only C++ IRIS API from a local source tree. It
resolves IRIS in this order:

1. `-DIRIS_SOURCE_DIR=/path/to/iris/irisx/development` (the path may also name
   the `irisx/` directory or the IRIS repository root),
2. the portable sibling layout
   `<workspace>/iris/irisx/development`, then
3. the historical `ROCm/iris` `muhaawd/irisx` fetch fallback.

For reproducible fallback builds, set `IRIS_FETCH_TAG` to an immutable commit.
To forbid an IRIS source fetch, pass `-DDK_FETCH_IRIS=OFF`; configuration will
then stop if neither an explicit nor sibling checkout is available. The
IRIS development project is not added as a subdirectory because that snapshot
unconditionally configures its examples, benchmarks, and tests. Instead, this
build mirrors its `iris::iris` header/dependency interface and configures only
the distributed HipKittens modules.

In a ROCm container with MPI and `pybind11` installed:

```bash
cmake -S distributed-kernels -B distributed-kernels/build \
  -DDK_BUILD=bf16_gemm \
  -DGPU_TARGET=CDNA4 \
  -DIRIS_SOURCE_DIR=/path/to/iris/irisx/development
cmake --build distributed-kernels/build --parallel 16
mpirun --allow-run-as-root -np 8 \
  python3 distributed-kernels/bf16_gemm/example.py
```

Omit `IRIS_SOURCE_DIR` when the repositories use the sibling layout above. Use
`GPU_TARGET=CDNA3` for gfx942 and `GPU_TARGET=CDNA4` for gfx950. `ROCM_PATH`
defaults to `/opt/rocm`; a caller-provided toolchain, `CMAKE_CXX_COMPILER`, or
`CXX` remains authoritative. Single-config generators default to a `Release`
build unless the caller selects another build type. This local workspace has no
ROCm toolchain or AMD GPU, so the new ports must be built and validated on the
matching cluster before promotion.

## Port promotion order

1. Compile the shared PGL and primitive spellings in isolated gfx942/gfx950 ISA
   twins and record resources.
2. Establish the GEMM-RS two-launch donor-parity path before enabling the
   one-launch D-prime experiment.
3. Establish the fused-MoE CDNA4 donor-parity path with `K0P6GM_G=3`; do not
   present it as a gfx942 port because its donor uses about 155 KiB of LDS.
4. Run all-rank correctness, dropped-publication negative controls, changing
   epoch/replay soak, and paired order-rotated timing from the architecture
   document.

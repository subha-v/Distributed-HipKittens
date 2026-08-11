# GEMM → ReduceScatter

This directory is the organized port of the strongest measured communication
pieces from the AMD overlap campaign. The long source file is intentional: its
HipKittens GEMM schedule remains together so generated-ISA comparison can gate
the port.

## Navigation

- `gemm_rs_device_tile.cpp` — two-launch rung-D producer plus REDV=1 reducer.
- `gemm_rs_hk_adapter.cuh` — the only communication-mechanism adapter.
- `gemm_rs_host_abi.hpp` — validated IRIS allocation snapshots and the exact
  two-descriptor host POD bundle; it performs no upload or launch.
- `gemm_rs_dprime_experimental.cpp` — explicitly unmeasured one-launch wrapper.
- `PROVENANCE.md` — immutable donor identity, results, and claim boundary.
- `dependencies.lock.json` — build inputs and compile contract.
- `BUILDING.md` — validation-target setup, descriptor construction, and gates.
- `../common/check_port_invariants.py` — source and optional donor-hash gates.

## MI300X/gfx942 port (additive)

The gfx942 single-launch megakernel port lives beside the gfx950 sources
without touching them:

- `MI300X_DESIGN.md` — dataflow, CTA roles, progress and lifetime proofs,
  config table, COMET-informed schedule limits on AMD.
- `MI300X_PROVENANCE.md` / `dependencies.mi300x.lock.json` — gfx942 donor
  identity (frozen rank-1 source, RadeonFlow submitted kernel) and hashes.
- `MI300X_VALIDATION.md` — local gates already passed and the
  `PENDING_GFX942_VALIDATION` gate list with exact later-node commands.
- `gemm_rs_mi300x.cpp` — persistent 304-CTA producer/reducer megakernel,
  one launch per call after setup.
- `gemm_rs_mi300x_constants.cuh` — pure shared geometry (compilable as host
  C++20), so layout formulas have one definition.
- `gemm_rs_mi300x_hk_adapter.cuh` — MI300X emit/reduction tile helpers and
  protocol spellings over the unchanged gfx950 adapter and primitives.
- `gemm_rs_mi300x_host_abi.hpp` — scored-shape table (frozen rank-1 tiles),
  launch-config resolution, sizing, cache key; host-only.
- `gemm_rs_mi300x_static_checks.py` — the 20 required static gates plus
  donor-hash verification.
- `gemm_rs_mi300x_simulation.py` — address/dependency, epoch/credit lifetime,
  negative-control, and numerical offline simulations.
- `tests/unit/common/kernel_host_abi_mi300x/` — strict-C++20 host ABI test.

Run the local battery with
`python3 distributed-kernels/gemm_rs/gemm_rs_mi300x_static_checks.py --amd-master /path/to/amd-master`
and `make -C tests/unit/common/kernel_host_abi_mi300x clean all run`.

Within `gemm_rs_device_tile.cpp`, read in this order: fixed configuration,
signal/credit layout, REDV=1 reduction, preserved GEMM schedule, egress emit,
then launch bindings. Historical A/B branches are quarantined and cannot be
instantiated; the selected path has no legacy transport dependency.

## Dataflow

Each GEMM CTA computes the donor's 256×256 C tile, waits for the prior directed
reuse credit, stages the EV=1 16-byte packets in already-dead LDS, projects the
heap through a PGL, performs the F6 CTA release, and publishes a device-derived
tile epoch. A separate owner CTA waits on eight sources, performs a convergent
pure acquire, issues the REDV=1 eight-load MLP sequence, accumulates sources in
ascending FP32 order, applies `bias * world`, rounds once to BF16, drains its
reads, and returns one credit to every source.

The exported two-launch composition is new and unmeasured: EV=1 and REDV=1
were measured in separate experiments, and the replay-lifetime credits are a
correctness repair added by the abstraction rewrite. D-prime is also unmeasured.

## Integration status

The source remains outside CMake auto-discovery until a world-8 gfx950 build,
ISA, correctness, dropped-peer, changing-input/skewed 600-replay soak, and
paired timing run pass. Do not add a `kernel.cpp` wrapper before those gates.
The existing pybind entry points are preserved for the donor harness. A new
IRIS launcher should call
`hk_gemm_rs::host_abi::snapshot_allocation_descriptors(view, c_heap, signals)`
and upload its two POD members independently for the existing `*_peers`
arguments. The helper applies each allocation's heap offset and rejects null
peer bases. Passing raw heap bases would drop that offset and is invalid. It
does not allocate device storage, upload either POD, retain their lifetime, or
launch a kernel.

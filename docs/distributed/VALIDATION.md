# Distributed validation matrix

The reusable layer is host-tested; neither operator port is promoted. This
page is the execution checklist that separates checks available in an ordinary
development checkout from architecture-specific GPU evidence.

## Local gates

Run from the repository root:

```bash
make -C tests/unit/common/types/global clean all run
make -C tests/unit/common/distributed clean all run
make -C tests/unit/common/iris_adapter clean all run
make -C tests/unit/common/kernel_host_abi clean all run
python3 distributed-kernels/common/check_port_invariants.py \
  --amd-master /path/to/amd-master
git diff --check
```

The first four commands compile both the CDNA3 and CDNA4 copies as strict
C++20 host code. They cover PGL metadata and two-level offsets, peer pointer
translation, the validated IRIS adapter boundary, packet movement, both memory
scopes, bounded success/timeout paths, 32-bit and packed 64-bit completion,
counter fan-in, row reservation, epoch broadcast, replay credits, exact
GEMM-RS descriptor snapshots, and MoE descriptor-vector extension.

The invariant checker rejects legacy Flow/IRISX/MoRI peer-translation calls,
checks selected port structure, and optionally verifies immutable donor/helper
hashes through `git show`. It reads no donor working-tree file and mutates
nothing.

## GPU gate A: primitive code generation

Run isolated manual/helper twins with the exact compiler used by the operator:

| target | required checks |
| --- | --- |
| gfx942 / CDNA3 | named PGL projection, runtime rank selector, packet x4 store, producer drain/release, caller-owned bounded poll, pure acquire, packed-word poll, replay credit |
| gfx950 / CDNA4 | the same source interfaces, checked independently against gfx950 lowering and resources |

Record normalized assembly and disassembly plus VGPR, AGPR, SGPR, LDS, scratch,
spills, occupancy, code size, and the toolchain/container identity. A helper is
not “zero cost” merely because the isolated opcode is present; compare it under
the representative MFMA register footprint.

## GPU gate B: GEMM-RS

Build the two-launch port before D-prime, following
`distributed-kernels/gemm_rs/BUILDING.md`.

- Confirm the 256-MFMA mainloop and EV=1 packet addresses/schedule.
- Confirm no resource regression beyond the recorded donor envelope.
- Validate all eight ranks against the operator oracle, including bias.
- Run a dropped-publication control that fails without hanging.
- Run at least 600 changing-input graph replays with deliberate rank skew and
  verify every device-derived epoch and directed credit.
- Compare the adapted two-launch path against its frozen donor with paired,
  order-rotated timing above the measured A/A resolution floor.
- Only then build and evaluate the separately named D-prime entry point.

EV=1, REDV=1, and the replay-credit repair have not been measured together, so
passing correctness alone does not transfer the donor timing claim.

## GPU gate C: fused MoE

Follow `distributed-kernels/fused_moe/BUILDING.md` on world-eight gfx950.

- Pin the missing MoRI helper tree/container content before calling the build
  reproducible; IRIS remains the peer-address backend.
- Confirm `K0P6GM_G=3`, both 48-MFMA phase spans, LL128 bytes/lane mapping,
  descriptor ABI v1, and no peer/synchronization state live across either MFMA
  loop.
- Measure the CTA-leader M7.5 publisher separately against the donor's
  all-thread publisher and an ordered parallel alternative; it fixes the
  release edge but changes work assignment and may serialize the phase tail.
- Compare the full resource tuple with the G=3 donor, including its known nine
  VGPR spills, 40 B scratch, and roughly 155 KiB LDS.
- Validate routing, weights, row extents, output numerics, `pperr=0`, and all
  eight ranks.
- Exercise retirement, chunk-word, A2, row-ready, and grid-barrier timeout
  controls. Every timeout must suppress stale payload access and no subset of
  CTAs may strand a later grid rendezvous.
- Run the broken control and at least 600 changing-input/skewed replay epochs,
  then use same-run paired timing against production.

This is a CDNA4 parity target. A gfx942 version requires a different LDS and
schedule design and must be treated as a new kernel rather than a rebuild.

## GPU gate D: fused-MoE MPS sibling

The minimum-progress specialization (`k0pf6gm_device_tile_mps.hip`) is a
separate experiment target with its own descriptor ABI (63 words). Follow
`distributed-kernels/fused_moe/BUILDING.md` ("MPS sibling") and
`DESIGN_MPS.md`; order: parity-port gates first, then the MPS mode ladder
(mode 0 C-sweep → mode 1 → mode 2 `C × g`) with the full correctness,
negative-control, and 600-epoch soak gates per arm before any paired timing.
The ISA A/B must confirm no ArchVGPR/AGPR/LDS movement versus the parity port
and no spill or fence migration into either MFMA K-loop.

## Current status

| layer | host semantics | donor/hash gate | device build/ISA | world-8 runtime | timing |
| --- | --- | --- | --- | --- | --- |
| PGL | pass, CDNA3/CDNA4 | n/a | pending | pending | n/a |
| primitives | pass, CDNA3/CDNA4 | n/a | pending | pending | n/a |
| IRIS adapter | pass, CDNA3/CDNA4 stubs | IRIS header recorded | pending | pending | n/a |
| kernel host ABI | pass, CDNA3/CDNA4 stubs | exact 72-byte PODs | pending upload path | pending | n/a |
| GEMM-RS port | static checker | recorded | pending | pending | pending |
| fused-MoE port | static checker | recorded | pending | pending | pending |
| fused-MoE MPS sibling | static checker + config/event unit logic | recorded | pending | pending | pending |

The local machine used for this scaffold has no CMake, HIP/ROCm toolchain, MPI
compiler, or AMD GPU. Do not convert any pending cell into a pass without
preserving its artifacts and exact environment.

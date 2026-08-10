# Distributed HipKittens architecture

Distributed HipKittens is a small set of device-side C++ primitives for
building single-launch and tightly composed multi-GPU kernels on AMD GPUs. It
extends HipKittens global layouts across fine-grained peer memory without
taking ownership of routing, work assignment, scheduling, or progress.

The intended style is ordinary HIP C++:

```cpp
using output_gl = gl<bf16, -1, -1, -1, -1>;
pgl<output_gl, 8> output = /* backend-provided peer mappings */;

auto owner_output = output.on(owner_rank);
// Existing HipKittens load/store operations still consume an ordinary gl.
store(owner_output, tile, tile_coord);

distributed::producer_drain_release<distributed::memory_scope::system>();
if (threadIdx.x == 0) {
  distributed::publish_epoch_relaxed<distributed::memory_scope::system>(
      ready_on_owner, epoch);
}
```

The code around those calls remains operator code. In particular, the library
does not choose `owner_rank`, invent a dependency key, launch another kernel,
or decide which CTAs may wait.

## Layers

### PGL: where

`pgl<GL, World>` contains an ordinary `GL`, the base of its local symmetric
allocation, and scalarizable named peer bases. `on(rank)` and `on<Rank>()`
return a copy of `GL` whose pointer is rebased to the selected peer.

PGL preserves the byte offset between `GL::raw_ptr` and the local allocation
base. This is required because real operators pass suballocations, not only the
start of a symmetric heap. PGL contains no epoch, completion, route, task,
barrier, or lifetime state.

For the world-eight path, named rank fields are intentional. MI300X compiler
experiments on `debug/pf6-first-launch` found that named-base/manual twins were
identical under BF16 MFMA pressure, while dynamic pointer arrays and
by-reference controls introduced calls and 80 bytes of scratch. This is static
compiler evidence, not yet a production zero-cost claim.

### Distributed primitives: when safe

The primitive layer exposes the repeated device mechanisms while leaving the
dependency topology visible:

- typed peer offset translation;
- aligned packet movement to or from peer memory;
- per-wave VMEM drain and convergent CTA release;
- relaxed monotonic publication;
- bounded relaxed observation into caller-owned result storage;
- success-only pure acquire before payload consumption;
- local counted fan-in;
- consumer drain and directed replay credit before slot reuse.

Publication is separate from release so one expensive release may cover a
batch of payload stores and several cheap completion cells. A timeout is a
result that operator control flow must handle; no primitive continues into
stale payload reads after a failed wait.

Readiness and storage lifetime are separate. A monotonic `ready >= epoch`
signal allows a late consumer to observe a producer that has advanced, but it
does not prevent that producer from overwriting an address-reused slot. Reused
storage therefore needs a directed credit from the actual consumer or a
disjoint ring slot.

### Megakernels: who and progress

Each megakernel owns:

- work and CTA role assignment;
- destination and source ownership;
- dependency-key layout;
- epoch derivation;
- buffer and wire-format layout;
- progress proof for any blocking CTA;
- reduction order and numerical policy.

The library deliberately has no graph, planner, executor, generic collective,
implicit grid/world barrier, or distributed compute-tile type. An API can make
a bounded wait explicit; it cannot prove that the scheduler will run the
producer.

## IRIS backend boundary

IRIS is the preferred runtime and peer-memory backend. Its C++ device view
establishes the symmetric-address rule:

```text
peer_address = peer_heap_base
             + (local_pointer - local_heap_base)
```

IRIS owns MPI/IPC setup, fine-grained symmetric allocation, peer mapping, and
host lifetime. Distributed HipKittens owns tile-facing layout and memory-order
operations. The adapter validates the runtime world, snapshots IRIS's peer heap
bases, and shifts them by the selected allocation's heap offset into a caller-
owned named-base POD before hot kernel code. The full runtime-indexed
`iris_device_view` should not remain live across an MFMA mainloop.

IRIS's generic system fence currently lowers through `__threadfence_system()`
for every memory order. Producer and consumer paths here need direction-specific
release and acquire lowering so a consumer acquire does not perform an
unnecessary producer-side writeback. The core therefore stays backend-neutral
and uses explicit AMD device ordering; the IRIS adapter supplies addresses,
not hidden synchronization policy.

## Kernel ports and evidence

### GEMM to ReduceScatter

The performance donor is the gfx950/CDNA4 k1 rung-D path:

- exp-23 EV=1: LDS-staged 16-byte peer emission, F6 grouped release, AV1 pure
  acquire;
- exp-24 REDV=1: issue eight source loads before ordered FP32 consumption.

Exp-23 measured 376.7 us on its communication-dominant shape, 0.8898x RCCL and
0.7761x RadeonFlow in that paired run. Exp-24 independently measured a 5.47%
reduction-kernel improvement. Those two changes were not measured together.

The port therefore has two distinct targets:

1. a two-launch donor-parity path that preserves the GEMM and EV=1 schedule;
2. an explicitly experimental one-launch D-prime composition.

Both require a per-tile directed replay credit missing from the original donor.
That hardening changes the protocol and must be measured rather than inheriting
the donor result.

### MoE megakernel

The performance donor is the exp-64 full-region PF6 megakernel compiled with
`K0P6GM_G=3`. On 8x MI350X/gfx950, the later G=3 run measured 6,919.8 us versus
7,698.0 us production, or 0.89846x. The port preserves the M0-M9 phase
topology, LL128 row ABI, routing, N2 compute phases, and operator-local grid
barriers. It replaces peer translation, publication, bounded wait,
acquire/release, counter, and lifetime spellings. Its corrected M7.5 readiness
publisher is also a schedule variant: 256 CTA leaders each release and publish
a row stripe so the same thread owns the release-to-relaxed-store edge. The
donor used all grid threads. This serialized publisher is unmeasured and must
be compared with a formally ordered parallel alternative before parity can be
claimed.

This source uses about 155 KiB of LDS and is not a direct MI300X/gfx942
performance port, whose workgroup LDS pool is 64 KiB. It is the CDNA4 donor
parity target and a second, materially different consumer of the common API.

## Repository organization

```text
include/<arch>/types/global/
  pgl.cuh                         peer-aware global layout type

include/<arch>/ops/group/distributed/
  peer.cuh                        peer pointer/offset operations
  packet.cuh                      explicit packet movement
  sync.cuh                        directional drain/release/acquire
  completion.cuh                  bounded publication/observation
  counter.cuh                     local counted fan-in/reservation
  lifetime.cuh                    directed replay credits
  distributed.cuh                aggregate include

distributed-kernels/
  common/                         IRIS adapter and validation utilities
  gemm_rs/                        donor provenance and abstraction port
  fused_moe/                      donor provenance and abstraction port
```

Architecture-specific include trees are deliberate: gfx942 and gfx950 need
separate code-generation and resource gates even when their source interfaces
match.

## Promotion gates

No primitive should be described as production, zero-cost, or performance
neutral until the same spelling passes both real operator bodies with:

1. exact address, payload, dependency, and arithmetic semantics;
2. manual/PGL/device-function canonical ISA comparison under the same MFMA
   footprint;
3. unchanged VGPR, AGPR, SGPR, LDS, scratch, spills, occupancy, and code size;
4. all-rank correctness and a working dropped-publication negative control;
5. at least 600 changing-input epochs with deliberate rank skew and no host
   reset;
6. paired, order-rotated timing above the calibrated A/A resolution floor.

The first runtime target should be GEMM-RS on MI300X. The CDNA4 MoE parity run
should remain a separate MI350X campaign. One-launch D-prime and a barrierless
MoE M7-to-M8 protocol come only after abstraction parity, so schedule changes
are not confused with abstraction cost.

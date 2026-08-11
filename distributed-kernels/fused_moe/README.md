# Fused MoE megakernel

This directory ports the best measured PF6 full-region MoE source configuration
while keeping operator policy separate from distributed mechanisms.

## Navigation

- `k0pf6gm_device_tile.hip` — preserved one-launch dispatch, destination sort,
  phase-1/phase-2 expert GEMMs, and combine schedule, pinned to `G=3`.
- `k0pf6gm_device_tile_mps.hip` — **additive sibling**: COMET/MoK-style
  minimum-progress specialization experiment (entry `k0pf6gm_mps_mega`). The
  parity port above is untouched. Modes: 0 reserve-only control, 1 bulk
  owner-slot push after the parity barrier, 2 full streaming overlap.
- `n2_phase2_gm_mps.cpp` — vendored phase-2 body with task start/stride macros
  and a per-thread VMEM drain hook (donor sha256 recorded in-file); every other
  byte, including both MFMA pipelines and the epilogue arithmetic, is identical.
- `moe_hk_adapter.cuh` — HipKittens/IRIS peer translation and completion plus
  the thin PF6 LL128 workload adapter.
- `moe_mps_adapter.cuh` — MPS operator policy: descriptor slots 56..62, packed
  role/config word, (b, nc) tile-event queue, slice arrival/claim accounting,
  owner-slot posting with controlled 16-byte stores, batched owner-only
  row_ready publication, diagnostic timestamps.
- `moe_host_abi.hpp` — validated IRIS heap snapshot plus strict host-side
  append/patch/validation for descriptor slot 55 and length 56; additive MPS
  buffer-binding, sizing formulas, and validation for the 63-word descriptor.
- `DESIGN_MPS.md` — the required design record: task graph, roles, buffer
  ownership/lifetime, release/acquire edges, expected counts, launch geometry,
  traffic model, alternatives rejected, invariants preserved, sweep + controls.
- `PROVENANCE.md` — measured source/config identity and unmeasured-port boundary.
- `dependencies.lock.json` — exact source hashes, toolchain record, derived-file
  ledger for the MPS sibling, and missing external pin called out explicitly.
- `BUILDING.md` — explicit genco include roots, flags, and acceptance gates.
- `../common/check_port_invariants.py` — static and optional upstream hash checks.

The common HipKittens layer does not know routing, task descriptors, or the PF6
FP8+scale+expert row ABI. Descriptor slot `K0P6_D_SYMMETRIC` explicitly carries
a `hk_moe::symmetric_heap_descriptor` built from IRIS's validated host snapshot;
peer pointers preserve their offset from the local heap base through
`kittens::distributed::translate_peer<8>`. The adapter keeps the exact LL128
row call local while delegating releases, publication, polling, counter
reservation, and acquires to `kittens::distributed`. MoRI headers remain only
for the frozen LL128 helper's build compatibility, not address translation.

## Integration status

This source is not a standalone `distributed-kernels` module yet. It textually
includes two N2 bodies and requires the pinned helper headers and a compatible
MoRI header/JIT tree listed in `dependencies.lock.json`. It intentionally has
no `kernel.cpp`, so top-level auto-discovery cannot imply a build that has not
been validated.

Use a distinct JIT cache key/root for this source. The measured runner's cache
hashed source bytes but not compile definitions, so G=2 and G=3 objects could
otherwise alias. A production integration must build gfx950, inspect both MFMA
spans, run world-8 correctness/negative control/600 replay soak, then measure
against a same-run production arm before claiming parity.

M7.5 is deliberately a correctness-first schedule variant. CTA leaders, not
all grid threads, publish row-ready stripes so every relaxed flag store is
sequenced after a release in that same thread. This avoids an invalid
cross-lane release assumption but may serialize the readiness tail. Compare it
against the frozen exp-62/exp-64 publisher and an ordered parallel alternative
before making a performance claim.

Any bounded wait or local-grid barrier timeout is terminal for that distributed
protocol instance. Do not clear `pperr` and retry the launch in place: producers
may already have published completion words for the unchanged epoch, which can
make a retry consume partially overwritten payload. Tear down or reinitialize
all ranks' epoch cells, cumulative barriers, counters, completion words, and
payload lifetime state before launching again. The replay soak applies only to
successful, zero-error epochs.

## Minimum-progress specialization (MPS) experiment

`k0pf6gm_device_tile_mps.hip` asks a COMET/MoK question with an AMD-specific
answer: reserve the *minimum* CTAs needed to guarantee communication progress
at the M6→M7 boundary (finish-order partition, no extra barrier), keep all
communication waits/fences/remote copies off the GEMM CTAs, and let drained
compute CTAs — and finally the service CTAs — flow elastically into a
dynamic-ticket combine reduction. The donor kernel's M7 already produces
COMET-compatible N-tiles (`(b, nc)` tasks with per-block `part_done` records);
the specialization replaces the bulk M7.5 publication with a streaming
slice-push pipeline whose service pool only needs to sustain ~150 GB/s under
the 1,844.5 µs M7 window to hide the measured 620–769 µs remote-`part`
transport floor. Descriptor word `K0P6_D_MPS_CFG` selects `C ∈ {0,4,8,16}`,
`g ∈ {1,2,4,16}`, mode ∈ {0,1,2} at runtime, so one HSACO carries the whole
sweep (use a distinct JIT cache key anyway — the mode-0 control is not meant
to alias the parity port's object).

The MPS source has **host/static validation only**: not built, not ISA-gated,
not timed. Gate plan before any claim (per DESIGN_MPS.md and k0's gate order):
resource/ISA A/B against the parity port on BOTH MFMA spans (the role word
crosses the M7 mainloop in SGPRs only), 96 MFMAs per phase-1 K-loop appended
with the task-done drain placed outside both K-loops, buffer-descriptor reload
per phase preserved (exactly six `k0p6_symmetric(desc)` sites), world-8
correctness + negative control + 600-epoch soak per mode, then same-run
paired timing versus both the parity port and production, sweeping
`C × g × flush_rows` with the mode-0 and mode-1 controls measured first.

The host bridge returns the 72-byte heap descriptor as an ordinary POD. After
the runtime uploads that POD, wrap its device-visible address in
`hk_moe::host_abi::device_visible_descriptor_address` and use
`append_symmetric_heap_descriptor` on the frozen 55-word vector, or
`patch_symmetric_heap_descriptor` on an already-sized 56-word vector.
`validate_descriptor` checks only the extended vector ABI and the new address's
nonzero/alignment properties; upload, address-space validity, lifetime, vector
upload, and launch remain runtime-owned.

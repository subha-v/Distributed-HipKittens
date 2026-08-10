# Fused MoE megakernel

This directory ports the best measured PF6 full-region MoE source configuration
while keeping operator policy separate from distributed mechanisms.

## Navigation

- `k0pf6gm_device_tile.hip` — preserved one-launch dispatch, destination sort,
  phase-1/phase-2 expert GEMMs, and combine schedule, pinned to `G=3`.
- `moe_hk_adapter.cuh` — HipKittens/IRIS peer translation and completion plus
  the thin PF6 LL128 workload adapter.
- `moe_host_abi.hpp` — validated IRIS heap snapshot plus strict host-side
  append/patch/validation for descriptor slot 55 and length 56.
- `PROVENANCE.md` — measured source/config identity and unmeasured-port boundary.
- `dependencies.lock.json` — exact source hashes, toolchain record, and missing
  external pin called out explicitly.
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

The host bridge returns the 72-byte heap descriptor as an ordinary POD. After
the runtime uploads that POD, wrap its device-visible address in
`hk_moe::host_abi::device_visible_descriptor_address` and use
`append_symmetric_heap_descriptor` on the frozen 55-word vector, or
`patch_symmetric_heap_descriptor` on an already-sized 56-word vector.
`validate_descriptor` checks only the extended vector ABI and the new address's
nonzero/alignment properties; upload, address-space validity, lifetime, vector
upload, and launch remain runtime-owned.

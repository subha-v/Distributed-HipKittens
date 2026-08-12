# exp_22 design — saturation ubench (one binary, three modes, optional overlay)

## Kernel bodies (all existing shapes, no new math)

| mode | body | work unit | metric |
|---|---|---|---|
| a (MFMA) | M7-shaped task loop: `[32·G]×448` phase-2 tile from the pinned `n2_phase2_gm` body, fed by a synthetic tile_desc list | one tile | `2·M·N·K·tiles / t` TFLOPS |
| b (HBM) | M8-shaped slot reduce: read 8 bf16 slots + write 1, 14-chunk loop, NT=4 | one 14,336 B row | `(8+1)·bytes·rows / t` GB/s |
| c (xGMI) | the MPS service pusher: `store_peer_packets` of 14,336 B rows (7168 × bf16) local slot → peer slot via `translate_peer<8>` | one row push | `bytes_pushed / t` GB/s |

Mode c parameters:

- **MLP depth d ∈ {1,4,8}**: issue d row-loads before the first store
  (exp_04's fan-out shape, parameterized).
- **fanout**: `single` (all rows → rank (me+1)%8, prices one link) or
  `rr7` (row i → peer i%7 skipping self, prices aggregate egress).
- **NO probe atomics, NO flags, NO fences** in the isolated arm — this curve
  prices payload transport only. A `--protocol` flag optionally re-adds the
  per-group probe RMW + per-flush `thread_release<system>` so the
  protocol-vs-payload gap is measurable *inside this ubench* (mirrors exp_20
  without needing the megakernel).

## Task distribution

Fixed global work list, grid-strided by dense role id — total work identical
at every CTA count, so time differences are pure throughput. Work sizes:

- mode a: 4,096 tiles (≈ one epoch's M7 volume at G=3).
- mode b: 131,072 rows (≈ 1.8 GB read+write per pass; repeat to ≥10 ms).
- mode c: 292,000 rows ≈ 4.0 GB pushed (≥ 10 ms even at full aggregate rate).

## Concurrent overlay (role split)

One launch, 256 CTAs, `blockIdx.x < C` = resource role (mode b or c loop),
else compute role (mode a task list sized to outlast the resource role).
Controls per point:

1. `C`-reserved, resource role **idle-spins** (capacity control — the
   reserve-only twin).
2. `C`-reserved, resource role live (the measurement).

Record per role: `s_memrealtime` start/end by tid 0 into a `[256][2]` u64
buffer (agent-scope plain stores, written once per role — no ordering
implications). Report: resource GB/s, and compute slowdown = live/control on
the same 256−C CTAs.

## Buffers

| buffer | size | notes |
|---|---|---|
| src slots | 4 GB / world | symmetric heap (mori), local segment |
| dst slots | 4 GB / world | symmetric heap, peer-written; epoch-tagged not needed (single launch, checksum instead) |
| role timestamps | 4 KB | local, host-read after launch |
| checksum cell | 8 B per rank | u64 xor-fold of pushed payload, read back |

No reuse across launches → no credits, no retirement, no epoch protocol.
This is deliberate: lifetime machinery would put protocol cost back into a
curve whose job is to isolate payload transport.

## Timing + clocks

- Wall: hipEventRecord around the launch, 5 rotations, median.
- Per-role: `s_memrealtime` (100 MHz constant, device-coherent) → µs on host.
- Record `rocm-smi --showclocks` before/after each point into the result JSON
  (no clock pinning; drift shows up in the rotation spread).

## Alternatives rejected

1. **rocprof counters for bandwidth** — per-kernel attribution is fine here,
   but counters can't separate the two roles inside one launch; the timestamp
   split can. Counters kept as a one-time cross-check on an isolated point.
2. **hipMemcpyPeer for the xGMI curve** — prices the copy engine, not our
   in-kernel pusher; SDMA is a different mechanism (M10, deliberately parked).
3. **Reusing mps_mega mode 2 as the ubench** — carries the full protocol and
   epoch machinery; exp_20 already showed that conflates protocol with
   payload. The ubench must be able to price them separately (`--protocol`).

## Primitives

Uses: `translate_peer`, `store_peer_packets`, `packet16`, (overlay)
role-by-blockIdx split. Expected friction to log in result.md: whether a
"push d rows with depth-d load pipelining" helper deserves to exist in
`packet.cuh` (exp_04 hand-rolled it; this ubench parameterizes it — if the
parameterized form wins, promote it).

## Plot spec (fig7_saturation.py)

3 panels; x = CTAs, y = TFLOPS / GB/s / GB/s; series per panel: isolated
(MLP depths as line styles for panel c), concurrent, ceilings as dashed
horizontals (76.8, 537.6, 8000, 148). Knee annotations = smallest C reaching
90% of each series' plateau.

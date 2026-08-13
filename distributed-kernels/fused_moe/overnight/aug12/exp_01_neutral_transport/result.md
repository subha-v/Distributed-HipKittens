# exp_01 result — neutral transport plane

Status: **IN PROGRESS**  
Stage-0/1 verdict: not yet available

## Fresh node calibration

A fresh exp_22 quick-tier run completed on the `ablations` checkout before the
new transport methods were built. All nine diagnostic points completed in
10.5 s wall time, with five rotations each and no verification failure.

| diagnostic point | measured |
|---|---:|
| 256-CTA MFMA shape-32 | 582.896 TFLOPS |
| HBM C=64 isolated / concurrent | 2628.485 / 2589.645 GB/s |
| xGMI one-link C=8, MLP=4 | 55.143 GB/s |
| xGMI rr7 C=8 payload isolated / concurrent | 97.998 / 97.681 GB/s |
| xGMI rr7 protocol g=16 | 57.652 GB/s |

The calibration reproduces the prior mechanism:

- payload bandwidth retained **99.68%** under concurrent MFMA
- compute retained **99.94%** against the C-matched reserved-idle twin
- the protocol overlay retained only **58.83%** of payload-only throughput

So the starting machine state agrees with aug11: CU-issued payload itself is
nearly free in this coalesced rr7 diagnostic, while coordination is expensive.
This is a calibration result, not a Stage-0 method winner. It uses exp_22's
one-process/eight-GPU topology and lacks the new negative-control and 600-epoch
gates.

Correctness checks were green: HBM wrong elements `0`, xGMI digest `MATCH`,
poison words `0`, and verification failures `0`.

Data: `calibration.json` (`exp01.calibration.v1`) and
`raw/e22_calibration_v1.jsonl`.

## Provenance

- node: `gbt350-odcdh2-c05-1.png-odc.dcgpu`
- branch/HEAD: `ablations` / `1dc3fe7e`
- source: exp_22 `E22_SRC_REV=10`, byte-identical to the remote ablations
  checkout
- source SHA-256 (remote LF): `e3af5f1b…d2b21e`
- binary SHA-256: `21fa0039…ca308`
- raw SHA-256: `e5aebe4c…d0d5e8`
- GPU process guard before and after: only `gpuagent`

## Open gates

- build and self-test CU push, CU pull, and host copy diagnostic
- prove early/dropped publication and redirect controls fail
- complete a 600-epoch Stage-0 soak
- move final arms to one process per rank
- establish host-copy executor for every size before using the `SDMA` label
- add same-API MORI forced-P2P/forced-SDMA comparison
- add IRIS semantic replication

## Primitives

The calibration reused `translate_peer`, packetized peer stores, and the
directional synchronization primitives through exp_22. The Stage-0
implementation will determine whether peer pull and explicit outstanding-depth
need additive public helpers. No primitive conclusion is claimed from the
calibration alone.

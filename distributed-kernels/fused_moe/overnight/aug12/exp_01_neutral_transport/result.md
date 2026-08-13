# exp_01 result — neutral transport plane

Status: **IN PROGRESS — diagnostic anchor gated**  
Stage-0/1 verdict: final rank-per-GPU anchor and size sweep pending

## One-process 64 KiB anchor

CU push, CU pull, and `hipMemcpyPeerAsync` all passed exact digest/sample
checks, poison checks, redirected-destination/early-publication/no-publication
negative controls, a 600-epoch soak, and host/device timer agreement. Each
timed point used five rotations and moved 64 MiB per rank bidirectionally.

| method | traced executor | global p50 (µs) | global p95 (µs) | GB/s |
|---|---|---:|---:|---:|
| CU push | CU | 15,723.779 | 15,848.641 | 8.536 |
| CU pull | CU | 15,623.289 | 15,657.848 | 8.591 |
| `host_copy_path` | CU runtime copy kernel | 22,718.219 | 22,883.654 | 5.908 |

CU pull/CU push is `0.993609×`, inside the pre-registered ±2% equivalence
margin. `host_copy_path` is `1.454125×` CU pull and `1.444832×` CU push. These
are mechanism-screen numbers, not final statistical claims: they are five
within-process rotations rather than independent rank-per-GPU campaigns.

The executor trace overturns the provisional API-based intuition. At this
64 KiB point, rocprofv3 recorded 8,192 `hipMemcpyPeerAsync` calls, zero
memory-copy-domain records, and direct same-correlation
`__amd_rocclr_copyBuffer` kernel dispatches. Correlation 3354, for example,
connects one 64 KiB device 1→0 API call to kernel id 8 on queue 2. There were
no `hsa_amd_memory_async_copy_on_engine` calls. The arm is therefore verified
as **CU**, not SDMA, at this configuration.

Plot-ready data: `anchor_comparison_v1.json`
(`exp01.anchor-comparison.v1`). Raw point summaries and logs are under `raw/`;
executor adjudication is `raw/host_copy_anchor_executor_v1.json`, with the
full 11.1 MB rocprofv3 trace in `raw/host_copy_anchor_trace_v1.json`.

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

- move final arms to one process per rank
- run the Stage-1 size sweep and independent paired campaigns
- trace every selected host-copy size; the 64 KiB anchor is CU-lowered
- add same-API MORI forced-P2P/forced-SDMA comparison
- add IRIS semantic replication

## Primitives

The diagnostic uses `peer_bases`, `translate_peer`, `store_peer_packets`,
`load_peer_packets`, `thread_release`, `release_and_publish`, bounded epoch
polling, and directed slot retirement. No primitive extension was required.
The useful negative finding is that HipKittens already expresses the symmetric
push/pull protocol without open-coded peer-pointer arithmetic; the host-copy
path still needs an explicit publication kernel because payload transport and
ordering remain separate concerns.

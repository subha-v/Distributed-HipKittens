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

## Rank-per-GPU bring-up

The first two-rank gate attempt was rejected before protocol execution:
exporting both `HIP_VISIBLE_DEVICES=0,4` and `ROCR_VISIBLE_DEVICES=0,4`
double-filtered the ROCR-remapped ordinal space, so rank 0 saw fewer than two
GPUs. Exit code 2 is classified as a launcher/runtime failure, not a method
failure. The runner now exports only `ROCR_VISIBLE_DEVICES`; attempt-1 logs and
provenance are preserved under `raw/rank_gate_*_attempt1.*`.
Two subsequent preflights correctly launched no work but exposed defunct
children from the first `MPI_ABORT`; the live-process guard now ignores `Z`
state while retaining `rocm-smi --showpids` as the device-side authority.

Attempt 4 was the first execution of the rank protocol. The `cu_push` positive
run failed, while all three applicable negative controls detected their
mutation on both ranks. This is a positive-path failure, not a launcher
failure and not a transport verdict. Source rev 1's gate artifact omitted the
positive rank metrics needed to distinguish data/protocol failure from the
independent timer-agreement gate. Source rev 2 therefore adds digest, poison,
epoch, credit, and timer diagnostics without changing the protocol. Its
CPU-only selftest and gfx950 build pass; the diagnostic rerun is pending.
Evidence: `raw/rank_gate_cu_push_attempt4.{log,json}` and
`raw/rank_gate_runner_attempt4.jsonl`.

The source-rev-2 rerun resolves the failure: payload correctness and protocol
state are green on both ranks (exact XOR/ADD digests, zero mismatches, zero
poison/sample failures, ready/completion epoch 1, returned credit 1). Only
timer agreement failed. `clock64()` spans from separate one-CTA stamp kernels
are not a valid cross-CU interval: rank 1 reported a 55,576.5 µs device
transport interval around a 1,278.81 µs HIP event. Rank 0 independently missed
the consumer tolerance (1,622.27 vs 1,464.76 µs). Source rev 3 changes only
the stamp clock to the globally synchronized 100 MHz `s_memrealtime` used by
the gated one-process diagnostic; payload, publication, and lifetime code are
unchanged. The selftest and gfx950 build pass.
Evidence: `raw/rank_gate_cu_push_attempt5.{log,json}`.

The corrected rank-per-GPU CU-push anchor is **GREEN**. Both ranks passed exact
payload/protocol checks and timer agreement; HIP-event/device-stamp transport
times were 1,279.96/1,275.72 µs and 1,280.69/1,276.28 µs. All applicable
negative controls passed, followed by a 600-epoch soak with zero rank errors.
Artifacts:
`raw/rank_cu_push_64k_one_epoch_{gate,soak,runner,provenance}_v1.*`.

The matched rank-per-GPU CU-pull anchor is also **GREEN**. Both ranks passed
exact payload/protocol checks, timer agreement, all applicable negative
controls, and the 600-epoch soak with zero rank errors. Gate transport
event/device times were 1,215.04/1,210.12 µs and 1,260.52/1,256.12 µs.
Artifacts:
`raw/rank_cu_pull_64k_one_epoch_{gate,soak,runner,provenance}_v1.*`.

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

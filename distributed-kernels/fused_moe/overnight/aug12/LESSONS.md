# aug12 overlap study — append-only lessons

## 2026-08-12 — scope and starting constraints

- `method:` The primary estimand is synchronized global makespan for a complete
  dependent DAG. Bandwidth and TFLOPS are explanatory metrics only.
- `method:` Separate transport-equivalent comparisons from complete-path
  comparisons. Direct producer publication may legitimately remove staging;
  SDMA must pay for materialization, enqueue, completion, pack, and unpack.
- `method:` “SDMA” is an executor claim, not an API label. Untraced copy points
  remain `host_copy_path` or `mori_put_path` and are excluded from the hardware
  engine map.
- `scope:` exp_22 is a reusable diagnostic donor, not the final comparison:
  one host process controls all GPUs and it does not use the common
  rank-per-GPU symmetric-heap process model required for MORI/IRIS parity.
- `correctness:` Existing MoE conclusions remain `T=4096` only. The known
  `T=1024/2048` under-write defect blocks application sensitivity claims but
  does not block the neutral synthetic suite.
- `branch:` All aug12 work is restricted to `ablations`.

## 2026-08-12 — fresh exp_22 calibration

- `calibration:` Nine quick-tier points reran cleanly at exp_22 source rev 10,
  five rotations each, with zero verification failures and only `gpuagent`
  resident before and after.
- `payload:` rr7 C=8 MLP=4 CU push retained `97.6805 / 97.9984 = 0.9968` of
  isolated bandwidth under concurrent MFMA. The compute role retained
  `342.616 / 342.833 = 0.9994` against its reserved-idle twin.
- `protocol:` Re-adding the g=16 protocol reduced rr7 bandwidth to
  `57.6518 / 97.9984 = 0.5883` of payload-only. The protocol-not-payload
  interference signature is present at the start of aug12.
- `scope:` This is machine-state calibration only. The one-process exp_22
  topology, missing negative controls, and missing 600-epoch soak make it
  inadmissible as a Stage-0 transport-method result.

## 2026-08-12 — gated one-process 64 KiB transport anchor

- `correctness:` CU push, CU pull, and `host_copy_path` each passed exact
  digest/samples, poison overwrite, redirected-destination, early-publication,
  no-publication, 600-epoch soak, and host/device timer gates.
- `anchor:` Global p50 makespans were CU push `15,723.779 µs`, CU pull
  `15,623.289 µs`, and host copy `22,718.219 µs`. CU pull/CU push is
  `0.993609×`, inside the frozen ±2% equivalence interval. This is diagnostic,
  not confirmatory, because the five rotations share one process.
- `executor:` The 64 KiB `hipMemcpyPeerAsync` arm is CU-lowered, not SDMA.
  rocprofv3 found 8,192 peer-copy API calls, zero memory-copy-domain records,
  direct same-correlation `__amd_rocclr_copyBuffer` dispatches, and zero
  `hsa_amd_memory_async_copy_on_engine` calls.
- `method:` An API→CU-kernel correlation is positive executor evidence even
  when rocprofv3's memory-copy domain is empty. The pre-registered
  “no CU payload kernel” expectation was falsified and superseded for
  CU-lowered host APIs.
- `mechanism:` One runtime copy-kernel enqueue per 64 KiB record makes the host
  path `1.454125×` CU pull end-to-end at the anchor. Do not call this an SDMA
  loss; no SDMA executor participated.
- `primitives:` `peer_bases`, `translate_peer`, packet push/pull, system-scope
  publication, bounded polling, and directed retirement were sufficient; no
  additive primitive was required for the diagnostic.
- `harness:` Never export the same physical-ID list through both
  `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`. ROCR remaps first and HIP
  filters the remapped ordinals again; `0,4` became one visible GPU. The first
  rank-gate attempt was a clean pre-protocol launcher failure and the runner
  now uses only `ROCR_VISIBLE_DEVICES`.
- `harness:` Ignore `Z`-state children in process-overlap preflight. A defunct
  MPI rank has no executable task and cannot retain a KFD queue; live process
  state plus `rocm-smi --showpids` remains the launch authority.
- `rank-gate:` The first actual two-rank CU-push protocol execution failed its
  positive run but detected no-publication, redirected-destination, and
  early-publication on both ranks. Do not infer a transport failure yet:
  source rev 1 did not serialize positive digest/epoch/credit/timer metrics.
  Source rev 2 adds those diagnostics without changing transport semantics.
- `timing:` Never subtract `clock64()` values written by separate kernels:
  they may execute on different CUs. The rev-2 rank gate was data/protocol
  correct on both ranks, but one such span read 55,576.5 µs against a
  1,278.81 µs HIP event. Use the globally synchronized 100 MHz
  `s_memrealtime` counter for cross-kernel device spans.
- `rank-gate:` With global timestamps, the 64 KiB two-rank CU-push anchor
  passed exact digest/poison/epoch/credit checks, all applicable negative
  controls, host/device timer agreement, and a 600-epoch soak with zero rank
  errors. The prior failure was exclusively timer instrumentation.

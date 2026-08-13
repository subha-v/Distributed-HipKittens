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

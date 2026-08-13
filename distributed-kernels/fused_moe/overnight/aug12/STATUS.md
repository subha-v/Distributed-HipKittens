# aug12 overlap methodology study — status

Date: 2026-08-12  
Branch: `ablations` only  
Node: `gbt350-odcdh2-c05-1.png-odc.dcgpu` (8× MI350X, gfx950)

This folder implements the staged experiment program in
`../aug11/OVERLAP_METHODOLOGY_STUDY.md`. Existing aug10/aug11 artifacts are
read-only evidence; every new plan, source snapshot, log, result, and
plot-ready artifact lands here.

## Current state

- Stage 0/1 neutral transport plane: **IN PROGRESS** — one-process 64 KiB
  anchor gated; rank-per-GPU 64 KiB CU push passed exact correctness, all
  applicable negative controls, timer agreement, and the 600-epoch soak.
  CU-pull and host-copy rank gates remain.
- Fresh exp_22 calibration: **GREEN** — 9/9 quick-tier points, no verification
  failures; payload concurrent/isolated `0.9968`, protocol/payload `0.5883`
- Diagnostic anchor: CU pull/CU push `0.993609×` (inside ±2% equivalence);
  host copy/CU pull `1.454125×`
- Executor trace: the 64 KiB `hipMemcpyPeerAsync` path is **CU-lowered**
  (`__amd_rocclr_copyBuffer`), not SDMA
- Stage 2 carrier plane: pending Stage-1 crossover selection
- Stage 3 multi-stream/TBO: pending Stage-2 carrier selection
- Stages 4–6 protocol/order/flow control: pending Stage-2 retention filter
- Stage 7 role container: pending matched carrier implementation
- Stage 8 topology: pending crossover-adjacent points
- IRIS replication: pending semantic parity gate
- MoE policy tournament: pending neutral-map nominees

## Starting evidence

- The existing exp_22 diagnostic establishes the MI350X resource ceilings and
  gives reusable MFMA/HBM/XGMI bodies, but it is a one-process diagnostic and
  is not a fair final MORI/IRIS comparison.
- Existing valid MoE evidence is scoped to balanced `T=4096`; current
  megakernels are incorrect at `T=1024/2048`, so those shapes are excluded
  until repaired.
- The current healthy MoE anchor is mode 12 with depth-4 admission. This is a
  later tournament contestant, not the neutral benchmark baseline.

## Execution order

1. Build a common two-rank producer → transport → dependent-consumer DAG.
2. Gate CU push and CU pull, then add verified host copy-engine transfer.
3. Add same-API MORI forced-P2P and forced-SDMA arms only after executor
   verification is available.
4. Sweep message size and select points below/near/above each crossover.
5. Add resource-matched compute and compare communication carriers.
6. Expand only retained or pre-registered rescue candidates into readiness,
   flow-control, order, topology, TBO, IRIS, and complete-MoE tournaments.

## Integrity rules

- One GPU job at a time; every launch uses `setsid` and `timeout`.
- Correctness, poison, negative controls, and a 600-epoch soak precede timing.
- Timings use synchronized per-iteration global windows and rotated paired
  campaigns.
- Hardware engines are named only when traced. An API name is not accepted as
  executor evidence.
- No `T=1024/2048` MoE performance result is admissible before correctness.
- Foreign GPU jobs are stopped with `SIGTERM` only and recorded in the active
  experiment result.

## Live artifacts

| artifact | status |
|---|---|
| `exp_01_neutral_transport/plan.md` | written |
| `exp_01_neutral_transport/design.md` | written |
| `exp_01_neutral_transport/schema.json` | written, JSON-validated |
| `exp_01_neutral_transport/calibration.json` | green diagnostic |
| `exp_01_neutral_transport/anchor_comparison_v1.json` | diagnostic anchor gated |
| `exp_01_neutral_transport/raw/host_copy_anchor_executor_v1.json` | executor verified CU |
| `exp_01_neutral_transport/transport_crossover.json` | pending Stage-1 sweep |
| `LESSONS.md` | live |
| `PLOTS.md` | live |

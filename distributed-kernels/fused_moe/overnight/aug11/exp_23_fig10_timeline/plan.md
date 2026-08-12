# exp_23 — per-layer resource-utilization timeline (NanoFlow v2 Fig 10 analog)

Status: **planned, unbuilt**. One additive instrumentation change (per-CTA
phase event ring behind the existing diagnostics flag) + two plot scripts.

## Goal

One MoE layer/epoch on the x-axis (~7 ms), three strips per arm — CTAs-in-MFMA
%, HBM GB/s, xGMI GB/s — for three arms:

| arm | data source |
|---|---|
| B0 RCCL-eager (sequential dispatch → GEMM → combine) | torch-profiler / rocprof kernel trace; each kernel interval maps to one resource strip |
| `pf6gm_mega` (homogeneous) | in-kernel phase event ring |
| `mps_mega` C=64 g=1 mode 2 (ratchet) | in-kernel phase event ring |

The figure argues the whole thesis at a glance: B0's strips are mutually
exclusive (the width of its xGMI blocks IS the "comm share of the layer"
number), homogeneous partially interleaves, MPS holds the MFMA strip through
M7 while xGMI runs beneath it — and M7's visibly longer strip vs homogeneous
is the interference term, honest in the same picture.

## Non-goals / rules

- The instrumented run is a **diagnostic arm** (A13 policy). Flag OFF for all
  campaign timing. Its µs are never reported as performance numbers.
- No protocol changes: plain agent-scope stores by tid 0 at phase boundaries,
  ≤ ~24 events per CTA per epoch, no new fences/atomics.
- Resource-tuple gate: compile with the flag present but disabled and confirm
  SGPR/VGPR/AGPR/scratch/LDS parity vs the current ratchet build — an
  instrumented-only code shape would make the timeline unrepresentative.

## Measurement plan

1. Widen `K0P6_D_MPS_STATE`-style diagnostics to a per-CTA event ring
   (layout in design.md), gated on the existing cfg.flags bit 0 mechanism.
2. One instrumented launch per megakernel arm, **same routing seed** across
   all three arms so phases align visually. Rank 0 plotted; rank-max as
   supplementary (state symmetry, don't average ranks).
3. B0 trace: one eager MoE layer under torch.profiler (or rocprofv3),
   kernel-name → resource-class mapping table checked into the exp folder.
4. Host post-processing into 10 µs bins:
   - MFMA strip: fraction of CTAs whose current phase ∈ {M6, M7 task loop}
     (label honestly: "CTAs in MFMA phase" — an occupancy proxy).
   - HBM / xGMI strips: analytic bytes-per-phase ÷ measured phase duration,
     summed over phases active in the bin (dispatch payload, slot pushes,
     part pulls, combine reads all have exact byte counts).

## Validation (both must pass before the figure ships)

- **Integral check**: each strip's integral over the epoch equals the known
  total bytes/FLOPs for that arm (±10%).
- **Counter cross-check**: `amd-smi metric` per-link xGMI throughput sampled
  over a steady-state loop matches the xGMI strip's epoch average (±20%);
  one rocprof MFMA-busy run on the homogeneous arm validates the occupancy
  proxy once.

## Validation command

```bash
K0_MPS_CFG="C=64,g=1,mode=2,flush_rows=1" K0_MPS_TRACE=1 \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh exp23_smoke 1
python3 plot_timeline.py runN/rank0_events.json --check-integrals
```

## Effort

1–2 days: ring + host dump ≈ 1 day, plotters ≈ ½ day, B0 trace ≈ an
afternoon. Node cost: minutes (one instrumented launch per arm + one
steady-state counter loop).

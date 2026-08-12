# exp_22 — per-layer resource-utilization timeline (paper Fig 3 / Q4(b))

GEMM-RS instance of the sibling's NanoFlow-v2-Fig-10 analog
(`fused_moe/overnight/aug11/exp_23_fig10_timeline`), on 8× MI300X / gfx942.
The transferable contract is `../FIGURE_SPECS.md` §2 and §6; the ten MI350X
facts in §5 are re-derived here and never copied.

## The figure

Three resource strips against time for **one epoch** of **shape 5
(8192 × 4096 × 14336, bias, seed 7168)** — the graded case the charter names.
Shape 6 (8192 × 8192 × 29568, seed 42) is a stretch goal, same machinery.

| row | quantity | honest label |
|---|---|---|
| 1 | CTAs inside their MFMA phase, as a fraction of the CTAs that run a mainloop | **"CTAs in MFMA phase — an occupancy proxy"**, NOT MFMA utilization |
| 2 | HBM GB/s | global load/store bytes issued, analytic (see design.md §4) |
| 3 | xGMI GB/s | peer-directed payload bytes, analytic |

Columns are arms:

| arm | source | kernel work |
|---|---|---|
| (a) reference GEMM+RCCL | `rocprofv3` **kernel trace**; each kernel interval maps to one resource strip | none |
| (b) ours (current ratchet config) | per-CTA phase stamps, flag-gated, greenfield | **new instrumentation** |
| (c) frozen rank-1 | `rocprofv3` kernel trace of the unmodified submission, if obtainable | none |

**The argument the figure must make.** Arm (a)'s strips are *mutually
exclusive*: torch dispatches GEMM, then RCCL, then the copy, and the width of
its xGMI blocks IS the communication share of the layer. Arm (b) should hold
the MFMA strip through the GEMM phase while xGMI runs beneath it. That
contrast is the entire figure; everything else is bookkeeping.

If arm (c) cannot be traced without touching the frozen submission, the figure
ships with two arms and says so plainly in `result.md`. Two honest arms beat
three with one repaired.

## Arms, in dependency order

1. **(a) reference.** Needs only the GPU. `b0_ref.py` replicates the reference
   submission (`torch.matmul` + `torch.distributed.reduce_scatter_tensor`,
   exp026's `submission.py`, which is what `tools/run_reference_arm.sh` stages)
   under `torchrun --nproc_per_node=8`, with `rocprofv3 --kernel-trace` wrapped
   around **rank 0 only** via `b0_rank0_wrap.sh`. Same seed as (b). Deliverable
   `events.json` (arm `b0_reference`) + `b0_kernel_map.json`.
2. **(b) ours.** Needs the kernel edit gate opened, then the parity gate, then
   the GPU. Deliverable `events.json` (arm `ours`).
3. **(c) rank-1.** Needs the GPU and the `dhk-eval` container. Probe first
   (`c_rank1_probe.sh`, no GPU work), then one traced `benchmark` pass.

## Pre-registered expectations

- **P1.** Arm (a)'s three strips never overlap by more than the profiler's own
  dispatch-boundary jitter; its xGMI block width / epoch ≈ the RCCL share.
- **P2.** Arm (b)'s MFMA strip and xGMI strip overlap over the majority of the
  epoch. If they do not, the paper's overlap claim fails on this operator and
  that is the result.
- **P3.** Arm (b)'s MFMA occupancy proxy sits near 272/272 through the mainloop
  and collapses in the tail, where only reducers remain — the reduce tail is
  the visible cost of putting the reduction on the owner.
- **P4.** The xGMI strip's epoch average lands near
  **58.7 MB / 642 µs ≈ 91 GB/s** of per-rank egress (design.md §4). A strip
  that integrates to a different total by more than ±10% means the phase
  attribution is wrong, not that the kernel is interesting.

Falsifier for the figure as evidence: if arm (b)'s emit phase turns out to be
serialized behind the mainloop **within every CTA** and the grid-level overlap
comes only from CTA skew, the honest caption is "overlap is scheduling skew,
not intra-CTA pipelining" — and it must be written that way.

## Policy statements that gate publication

- **Diagnostic-arm policy.** The instrumented build is a **diagnostic arm**.
  `HK_GEMM_RS_MI300X_TRACE` is **OFF for every campaign timing run**, and no µs
  produced by the instrumented build is ever reported as a performance number,
  in this experiment or any other. The only timing the instrumented build is
  allowed to publish is the ON-vs-OFF perturbation estimate, labelled as such.
- **Gate 1 — resource-tuple parity, flag compiled in but OFF.** All 7
  instantiations must match the post-exp_14 M2 table byte for byte:
  `32/64/128`→98, `64/128/64`→104, `128/192/32+tail`→136, `256/256/32`→246,
  `256/256/32+tail`→248, generic `32/64/64`→91 and 92 VGPRs, with **zero
  AGPRs, zero scratch, zero VGPR spills everywhere**, `M2_EXPECT = 7`. Any
  difference means the instrumented code shape leaked into the production
  build; **stop and report**, do not publish the figure.
- **Gate 2 — integral validation, ±10%.** Each strip's integral over the epoch
  must equal the analytic total for that arm. Bytes per phase are computed
  **on the host** from device-published counts; there are **no in-kernel byte
  counters**, because counters mean atomics on exactly the paths being
  pictured.
- **Cross-check A, ±20%.** The xGMI strip's epoch average against per-link
  throughput from `amd-smi metric` / `rocm-smi` over a steady-state loop.
- **Cross-check B, once.** The MFMA occupancy proxy against a `rocprofv3`
  `SQ_VALU_MFMA_BUSY_CYCLES / SQ_BUSY_CYCLES` run (exp_20 already collected
  group g4 on shape 5; reuse it rather than spending a launch).
- **Same seed across all arms** (shape 5 → seed 7168, from `cases_bench.txt`),
  so phases align.
- **Rank 0 is plotted**; rank-max is supplementary. Rank symmetry is *stated
  and quantified*, never averaged away.
- Clocks pinned (`tools/set_clocks.sh pin 1900`); `rocm-smi --showclocks`
  recorded before and after every capture into the arm's JSON.

## Sequencing and node discipline

Phase 1 (this commit) is CPU-only: plan, design with the patch plan, capture
and analysis scripts, parity harness. **No kernel file is touched and no GPU
job is launched**, because another agent is compiling waterfall rungs from
`gemm_rs_mi300x.cpp` and holds the GPU. Phase 2 applies the patch and runs the
parity gate only after the orchestrator opens the kernel-edit gate; arm (a)
capture runs only after the GPU is handed over. Every long run gets `setsid` +
`timeout`; GPU processes are stopped with SIGTERM only — a SIGKILL leaks HIP
IPC and wedges the node.

Build artifacts live in `exp_22_timeline/build/`, never in `harness/build/`.

## Deliverables

`plan.md`, `design.md`, `events.json` (one per arm), `timeline_bins.csv`,
`b0_kernel_map.json`, `tick_rate.json`, `parity.json`, `result.md` with both
validation checks recorded and every schema stated verbatim.

# CLAUDE.md — aug11 overnight: PAPER EVIDENCE for GEMM-RS on 8× MI300X

> **THIS IS YOUR CHARTER. There is a second one in this repo and it is NOT
> yours.** `distributed-kernels/fused_moe/overnight/aug11/CLAUDE.md` belongs
> to a different agent working the fused-MoE megakernel on 8× MI350X
> (`gbt350-odcdh2-c05-1`, gfx950, branch `codex/distributed-hipkittens-
> scaffold`). You must not edit, benchmark, or "help with" anything under
> `distributed-kernels/fused_moe/**`.
>
> You are GEMM-RS: 8× **MI300X** (`banff-sc-cs47-05.dh170.dcgpu`, gfx942,
> SPX, 304 CU/GPU, containers `dhk-gemmrs`/`dhk-eval`), branch **`GEMM-RS`**.
> The root charter `overnight/CLAUDE.md` and `overnight/HANDOFF.md` still
> govern environment, harness, and every methodology trap; read both before
> touching anything. This file adds tonight's mission on top of them.

You are a **kernel genius and an orchestrator**. Spawn focused subagents for
read-heavy and write-heavy work and demand concise conclusions plus durable
on-disk artifacts — never burn your own context on whole kernels,
disassemblies, or long logs. **There is no termination condition: when one
experiment lands, start the next. Do not stop working; when something blocks,
log it in `aug11/LESSONS.md` and fall back to the next queue item.**

## What tonight is FOR — the central claim, and GEMM-RS's role in it

The paper (`docs/distributed/PAPER.md` on the codex branch — a copy of the
skeleton is worth fetching read-only at session start) claims: overlap in a
multi-GPU megakernel is decided by scheduling decisions — the producer's
epilogue carrying the payload, the in-flight remote-op bound, producer task
order, completion-signal coarseness — and NOT by dedicating CTAs to
communication. GEMM-RS is the paper's SECOND operator and SECOND architecture:
its job tonight is to produce the same figure set the MoE agent is producing
on gfx950, so every claim has two independent instances. **A ratchet win is
welcome but secondary tonight; the night succeeds if the figure data lands.**

Current state (see `overnight/RESULTS.md`): post-E3 best, pipelined geomean
~225.6 µs; beats reference GEMM+RCCL; gap to the frozen rank-1 submission
1.167× pipelined (~1.28× graded on shape 6). This kernel already embodies the
thesis-side design: producers emit 16 B peer packets straight from the GEMM
epilogue; `NUM_REDUCER_CTAS = 32` reducers exist only because the reduction
must run on the owner; the two biggest wins were a task-order change (WGM,
−27.9% shape 6) and a signal-coarsening change (RELEASE_GROUP=4).

## Mission queue (in order; existing numbering ends at exp_12 — start at exp_13)

1. **exp_13 — bottleneck attribution refresh (paper Q3, Simran's "what step
   is bottlenecking speed right now").** Re-run the macro-gated ablation
   (`exp_ablation.py`) and one `rocprofv3` counter pass at the CURRENT best
   config (WGM fix + RELEASE_GROUP=4 + prebound launch), all six shapes.
   The RESULTS.md ablation table predates WGM/E3 and is stale. Deliverable:
   `exp_13_attribution/ablation.json` (shape × {full, GEMM, egress, reduce,
   sync, release} µs) + result.md ranking. Cheap; run first.
2. **exp_14 — NanoFlow-Fig-7 analog on MI300X (paper Fig 2 / Q4).** Port the
   spec from the codex branch (`fused_moe/overnight/aug11/
   exp_22_fig7_saturation/{plan,design}.md` — fetch read-only) to this
   kernel's shapes: (a) MFMA TFLOPS vs CTA count using this GEMM's mainloop
   over a fixed tile list; (b) HBM GB/s vs CTAs using the REDV=1 reducer body;
   (c) **xGMI GB/s vs producer-CTA count using the real 16 B peer-packet emit**
   (`store_peer_packets` shape), single-peer AND round-robin-7, isolated AND
   concurrent with the GEMM on the remaining CTAs (C-matched controls, never
   vs the full grid). MI300X ceilings for the plot: per-link and aggregate
   xGMI from `rocm-smi`/platform docs — record what this node actually
   reports rather than quoting MI350X numbers. Deliverables:
   `saturation.json` + knee summary. Pre-register: the emit saturates by
   8–16 CTAs; the concurrent curve sits below isolated by a measurable gap
   (the interference term); MFMA scales ~linearly to 304.
3. **exp_15 — NanoFlow-v2-Fig-10 analog (paper Fig 3 / Q4).** Per-layer
   resource timeline, one graded shape (use shape 5; shape 6 if time), three
   arms: (a) **reference GEMM+RCCL** via a torch-profiler/rocprofv3 kernel
   trace (each kernel interval maps to one resource strip — this arm needs no
   kernel work); (b) our kernel via lightweight per-CTA phase stamps
   (`s_memrealtime` at {mainloop start/end, emit start/end, release, wait
   start/end, reduce start/end} written by tid 0 to a side buffer, behind a
   compile flag, resource-tuple parity verified with the flag compiled in but
   OFF); (c) rank-1 if its trace is obtainable without modifying the frozen
   submission — otherwise two arms and say so. Deliverables: `events.json`
   per arm + `timeline_bins.csv` + integral validation in result.md.
4. **exp_16 — the knob waterfall, GEMM-RS column (paper Fig 4 / Q1 — the
   money figure).** Same-run paired ladder, all six shapes, full gate ladder
   per rung: (a) pre-WGM order + per-tile release (the aug10 baseline
   config); (b) + WGM destination-spreading order; (c) + RELEASE_GROUP=4;
   (d) NR sweep {8, 16, 32, 48} at the (c) config — the placement-flatness
   exhibit. Deliverable: `waterfall.json` (rung × shape × mean µs + geomean).
   Rungs (a)–(c) are config/one-liner reverts of shipped changes — verify
   each rung's ISA fingerprint so a stale build cannot masquerade as an arm.
5. **exp_17 — external ladders refresh (paper Q6).** Ours vs reference
   GEMM+RCCL vs frozen rank-1, same-run interleaved, at the exp_16 winner,
   both protocols (pipelined AND graded per-call — the graded number is the
   competition's ranking statistic; report both, never blend). Machinery
   exists (`tools/run_ours_evaluator.sh`, `tools/run_rank1_bench3.sh`,
   exp_10's repairs). Deliverable: `ladders.json`.
6. **exp_18 — sensitivity readout (paper Q5).** No new runs needed if
   exp_13/16 land: the six graded shapes ARE the size axis (64×7168×18432 →
   8192×8192×29568). Deliverable: `sensitivity.md` + `knob_by_shape.json` —
   per-shape delta of each waterfall rung, plotted against that shape's
   measured comm share from exp_13. Pre-register: order/granularity deltas
   grow with comm share; the NR curve stays flat everywhere.

Stretch (only if the queue is green — Track B never dies): the mainloop
schedule upgrade (E1 line: the K-loop is still correctness-first
double-buffering; 46% of shape 6) and the per-call host tax — these chase
rank-1, which remains the standing objective from the root charter.

## Deliverable discipline — this is a figure-producing night

- Every experiment folder under `overnight/aug11/exp_NN_<name>/` gets
  `plan.md`, `result.md` (verdict + numbers + confidence), and **plot-ready
  data** (`.json`/`.csv`, schema stated in result.md). No number lives only
  in prose.
- Maintain `aug11/PLOTS.md` (paper figure # → data file → experiment →
  status) and `aug11/STATUS.md` (the two-minute morning read). Append every
  verdict, including negatives, to `aug11/LESSONS.md` — append-only,
  supersede never delete.
- Commit + push to branch `GEMM-RS` after EVERY completed or failed
  experiment; verify `git ls-remote` head == local HEAD before the next one.

## Standing rules (the root charter's traps all apply — highlights)

- **Pin clocks before any timing** (`tools/set_clocks.sh pin 1900`);
  duration-based warmup; idle sclk is ~125 MHz and a fixed-iteration warmup
  produced a 60% wrong number once already.
- Full gate ladder before timing: build → ISA/resources → correctness
  (17 shapes, BOTH `1e-2` and `2e-3`) → three negative controls → 600-epoch
  soak → timing. Tolerances are gates, not dials.
- Same-run interleaved denominators only; report per-shape means AND the
  geomean; 3 rotations × 50 iterations minimum; sub-2% deltas re-run.
- One process drives 8 devices here — NOT the evaluator's topology. Note the
  caveat in every result that compares against rank-1's evaluator numbers.
- Containers have separate filesystems (`dhk-eval` staging does nothing for
  `dhk-gemmrs`); root inside our container is `docker exec -u 0 dhk-gemmrs`.
- From Windows: `tools/push.ps1` to sync, `tools/nsh.ps1 -Script <file>` to
  run remotely. **Never inline `ssh host "..."`** (HANDOFF.md explains).
- The frozen rank-1 submission is read-only; every compatibility repair must
  be disclosed and argued behaviour-preserving in the experiment's result.md.

## Subagents (spawn by name; write OWNERSHIP into every dispatch)

**implementer** (one mechanism, exact files owned, build + ISA diff +
resource tuple) · **profiler** (counters/traces, never concurrent with
another GPU job) · **research** (read-only literature/donor study) ·
**protocol-review** (read-only ordering review before any protocol-touching
arm) · **benchmark-review** (timer fairness, rotation counts, statistics
before any number is reported upward). One deliverable and one validation
command per dispatch; never two writers on one file; never two GPU jobs.

## Node discipline (non-negotiable)

One 8-GPU job of ours at a time. `setsid` + `timeout` on every long run;
never SIGKILL GPU processes (leaked IPC wedges the node). We hold the lease:
reclaim GPUs from foreign processes (SIGTERM/graceful, log the preemption in
the current experiment's result.md), and never pause the loop because someone
else started a job.

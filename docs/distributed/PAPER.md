# Paper skeleton — Scheduling, Not Placement: Communication/Computation Overlap in Multi-GPU Megakernels on AMD GPUs

Working thesis, section plan, evidence map, and the reproduction matrix.
Every evidence citation below names a measured artifact already in this repo
(`aug10/`/`aug11/` experiment ledgers, GEMM-RS branch `overnight/RESULTS.md`)
or a planned experiment with its folder. Nothing in the skeleton relies on an
unmeasured claim without saying so.

---

## 0. Thesis (one paragraph)

In a multi-GPU megakernel, communication performance is decided by four
schedule knobs — **binding** (which instruction stream carries the bytes),
**rate** (how many remote operations are in flight), **order** (which tasks
run when, and toward which peer), and **granularity** (how coarse the
readiness protocol is) — and *not* by **placement** (which CTAs are assigned
to communication). On AMD CDNA GPUs this is not a preference but a
consequence of measurable hardware facts, and the four knobs exist *only*
inside a persistent single-launch megakernel: a multi-kernel system cannot
set an injection depth on a GEMM epilogue, reorder producer tiles to shape a
consumer's readiness curve, or bind transport into an epilogue's atomic
stream, because those decisions die at every kernel boundary. We demonstrate
this with two production-derived kernels on two CDNA architectures, a
primitive library that makes the knobs programmable, and a pre-registered
adjudication that closes the placement axis on symmetric single-node
workloads.

## 1. Contributions

1. **A schedule-knob framework** (binding / rate / order / granularity vs
   placement) for device-initiated communication inside megakernels, with a
   measured law for each knob on CDNA3 and CDNA4.
2. **The adjudication**: CTA-level communication specialization — the axis
   COMET, MoK, and Triton-distributed center — is dominated on single-node
   symmetric workloads, shown by direct A/B (same kernel, transport moved
   between designs), by arithmetic (work conservation + fabric ceiling), and
   by degeneration (once transport is bound and readiness coarsened, the
   specialized pool has no remaining job).
3. **Two new mechanisms with measured cliffs**: epilogue injection-rate
   control (throttle depth, ~500 µs, catastrophic above depth 8) and
   destination-aware task ordering (xGMI link concurrency, −27.9% on the
   largest GEMM-RS shape with zero bytes moved differently).
4. **Distributed HipKittens**: a two-layer header library — ordering/
   addressing *mechanisms* (PGL, packets, directional release/acquire,
   bounded polls, credits) plus reusable *schedule policies* (landing slots,
   rate-controlled egress, task-order maps, release batching) — validated by
   both operators with ISA/resource-parity gates.
5. **A measurement-integrity methodology** for distributed-kernel benchmarks
   (staleness-blind gate failure modes, NaN poisoning, phase stamps vs
   end-to-end resolution, counter-validated attribution).

## 2. Why this is the optimal overlap strategy on AMD — the argument chain

This section is the paper's spine. Five hardware/measurement facts, each
closing one alternative:

| # | fact | evidence | what it eliminates |
|---|---|---|---|
| F-A | **Occupancy theorem.** At megakernel register/LDS budgets, CDNA runs exactly one block per CU. A communication CTA is never co-resident with an MFMA CTA, so specialization cannot create intra-CU overlap, and reserving C CTAs costs exactly `N/(N−C)` on the CTA-limited phase. | CDNA4 ISA §3.6.4; aug10 A7 strike; measured tax curve (exp_03) | warp/wave specialization (HipKittens: 80% of peak on MI355X) AND free-lunch CTA pools |
| F-B | **The fabric is byte-limited with free MLP in epilogues.** Coalesced 4 B pk-bf16 remote atomics sustain the same per-link rate as 16 B stores (52.8 vs 54.9 GB/s); target-side MFMA streaming does not slow inbound traffic; the epilogue's existing instruction stream already carries 16–32 independent ops/thread. | exp_21 ubench | payload-moving comm pools: bytes ride compute CTAs at zero capacity cost (binding) |
| F-C | **Interference is protocol, not payload.** Deleting the service pool's entire copy moves the concurrent GEMM +5.7 µs while 831 µs of interference remains; the floor decomposes ~70% op count / ~30% ordering scope. | exp_20, exp_05/06/07 | "hide the bytes" designs; motivates granularity + keeping control ops off MFMA paths |
| F-D | **Work conservation + fabric ceiling.** A static role split reassigns conserved work (best balanced M6/M7 split: −173 µs *loss*), and its one real term — fewer fabric injectors — is capped at +211 µs because aligned epilogue bursts already drive xGMI at 78% of ceiling; the same rate effect is available per-CTA via depth capping at zero capacity cost. | exp_25 rev2 | compute/compute splits; placement-as-rate-control |
| F-E | **Readiness curves are set by task order, not by protocol granularity.** With top-k=8 routing, `P(token reducible by t) = (t/S)^8`: the median token unblocks with 8.3% of the producer phase remaining, so fine-grained readiness (~926k atomics/rank/epoch) buys an overlap that never occurs; an nc-major task order lifts unblockable fraction from ~17% to ~85%. | exp_29/exp_30 | fine-grained signalling (and with it, the pool's last job); promotes order |

Corollary (the design rule the paper argues is optimal on AMD): *bind the
data plane into producer epilogues; cap its injection depth; order tasks for
link concurrency and consumer readiness; make the control plane as coarse as
the readiness curve allows; specialize CTAs only for work that must exist
and cannot ride an existing stream (owner-side reduction).* Both kernels
independently converged on exactly this shape.

## 3. Section plan

1. **Introduction** — the placement consensus (COMET's thread-block
   specialization, MoK's comm SMs, Triton-distributed's SM carve) and the
   claim that it answers the wrong question on AMD. One figure: the
   knob taxonomy with each knob's best measured win attached.
2. **Background: what CDNA actually gives you** — occupancy theorem, 8
   fully-connected xGMI links (76.8 GB/s/dir each; the *link*, not the
   aggregate, is the schedulable unit), directional cache ops
   (`buffer_wbl2 sc1` vs `buffer_inv sc1`), no setmaxnreg/TMA/mbarrier.
   The "struck — no CDNA equivalent" list from aug10 CLAUDE.md becomes a
   table here.
3. **The framework** — the two-plane decomposition (data vs control) and the
   four knobs; the conditional theorem for when placement can pay
   (exposed peer-wait > 0 AND work that cannot ride existing streams).
4. **Distributed HipKittens** — mechanism layer (existing headers) +
   schedule layer (landing_slots, rate-controlled egress with deferred
   publication, task-order maps, release batcher, group arrive-release);
   the ISA-parity promotion gates; the named-peer-bases compiler evidence.
5. **The two kernels** — the common protocol table (produce → self-ship →
   batched release → epoch flag → owner-local reduce → credit/retire);
   MoE megakernel on gfx950, GEMM-RS on gfx942; what differs (values vs
   adds; flags vs epoch words) and what doesn't.
6. **The adjudication of placement** — mode 2 vs mode 12 A/B; exp_30
   degeneration; exp_25 arithmetic; NR flatness; leaderboard replication
   (three competitors built-and-abandoned; the per-tile-CAS design placed
   5th at +23%; rank-1 has no comm pool).
7. **Evaluation** — figures below.
8. **Sensitivity: does the answer change with batch, seqlen, imbalance?** —
   the sweep matrix in §5 of the reproduction plan; states the pre-registered
   predictions and reports where the knob ranking shifts and whether
   placement ever re-enters.
9. **Methodology: benchmarks that cannot lie** — staleness-shaped bugs under
   fixed inputs post the *best* number; NaN poisoning; negative controls
   that must fail; phase stamps (σ ≈ 1%) vs end-to-end screens (σ ≈ 6.6%);
   counter validation via the emit_local collapse test; clock pinning.
10. **Related work** — COMET, MoK, Triton-distributed, Fleet, SC24 vertical
    fusion, FLUX/TileLink, NanoFlow (intra-device analog), DeepEP;
    the competition-analysis dataset as evidence.
11. **Limitations** — single node; symmetric routing for headline numbers;
    MoE forward-only; gap to GEMM-RS rank-1 is GEMM quality, reported not
    hidden; placement axis closed *here*, with the `spin_dbg` one-line test
    for when it reopens.

## 4. Figure plan

| fig | content | source |
|---|---|---|
| 1 | knob taxonomy + best measured win per knob (both kernels) | ledgers |
| 2 | **NanoFlow-v1-Fig-7 analog**: MFMA TFLOPS / HBM GB/s / xGMI GB/s vs CTA count, isolated AND concurrent (the interference gap NanoFlow's method can't see) | exp_22 |
| 3 | **NanoFlow-v2-Fig-10 analog**: per-layer resource timeline (CTAs-in-MFMA %, HBM, xGMI) × three arms (RCCL-eager, homogeneous, mode-12 ratchet) | exp_23 |
| 4 | the knob waterfall: homogeneous → +binding → +rate → +granularity → +order, µs per step, both kernels side by side | ablation ladder (§5.1) |
| 5 | placement adjudication: mode2-best vs mode12-best; C sweep at mode 14 (flat/degenerate); NR sweep (flat) | campaigns |
| 6 | link-concurrency: per-link xGMI occupancy before/after WGM reorder; readiness curve (t/S)^8 vs nc-major staircase | exp_08 counters; exp_30 model + stamps |
| 7 | sensitivity heatmap: best config vs (M, skew std) | §5.4 sweeps |
| 8 | ladders: MoE vs production + Megatron/RCCL-eager; GEMM-RS vs reference GEMM+RCCL + rank-1 same-run | campaigns |
| 9 | throttle-depth cliff; release-group curve | exp_24, E3 |

## 5. Reproduction matrix (the experiments to run, in order)

### 5.1 The knob waterfall (the paper's money table) — ~2 node-days
One campaign per rung, both operators, full gate ladder each:
MoE: `pf6gm_mega` → mode 12 (binding) → +throttle depth 4 (rate) →
mode 14 barrier+8 flags (granularity; **build needed**, NaN poison
prerequisite) → +nc-major order (**build needed**).
GEMM-RS: pre-WGM baseline → +WGM order → +RELEASE_GROUP=4 → NR sweep
(placement-flat exhibit). Each rung is a single-variable change already
defined in the ledgers.

### 5.2 exp_22 — saturation curves (Fig 2) — 1–2 days build, minutes GPU
As specified in `overnight/aug11/exp_22_fig7_saturation/`. Deliverables:
the xGMI knee (topology-correct pool size), concurrent/isolated gap
(interference as a curve), MFMA linearity (capacity law).

### 5.3 exp_23 — utilization timelines (Fig 3) — 1–2 days
As specified in `overnight/aug11/exp_23_fig10_timeline/`. B0 arm from a
torch-profiler trace; megakernel arms from the per-CTA phase event ring.

### 5.4 Sensitivity sweeps (Simran's question; Fig 7) — ~2 node-days
- **M (batch×seqlen per rank)**: T ∈ {512, 1024, 2048, 4096} in the MoK
  harness; vLLM chunked-prefill sweep {1024, 2048, 4096} on real prompts for
  the serving version. Pre-registered: small M shifts value toward
  granularity + fusion (fixed protocol cost dominates); large M toward
  rate + order (fabric bytes scale); binding free everywhere; placement
  never enters on symmetric traffic.
- **Imbalance**: `skewed_hot` routing family, std ∈ {0, 0.01, 0.032, 0.05}
  (0.032 = COMET's production skew). Read `[MPS SPIN]` per point.
  Pre-registered: skew (a) turns on real peer-wait — the one regime where
  elastic placement could re-enter (test C ∈ {0, 8, 16} at high skew);
  (b) de-coalesces epilogue atomics — the ubench's 13× op-rate collapse
  risk makes rate/order *more* valuable, not less; (c) makes
  dependency-ordered task scheduling a load-balancer.
- **Decode-shaped small-M** (T ≤ 256): fusion/launch-overhead regime;
  megakernel vs production gap should widen while all knob deltas shrink.

### 5.5 Placement adjudication reruns (Fig 5) — ~1 node-day
Paired mode2-best (C=64 g=1) vs mode12-best (C=16) vs mode 14 (C sweep
{0,8,16} — prediction: flat); F1 (`T6(128)/T6(256)`, ~30 min, macros already
live); one skew point with pools re-enabled.

### 5.6 External ladders (Fig 8) — ~1 node-day + harness work
MoE: add a Megatron-ROCm/PyTorch+RCCL eager arm (the COMET-comparable
denominator). GEMM-RS: reference GEMM+RCCL and frozen rank-1 already run
same-node (`exp_10`); refresh at final ratchet.

### 5.7 Current-bottleneck attribution (text + one stacked bar) — free
Phase stamps at final ratchet + the exp_08-style counter pass. Today's
answer (2026-08-11 ratchet 6,568 µs): **M6+M7 GEMM mainloops are 78% of the
kernel** (M6 ~2,588 µs with its K-loop ~84% stall; M7 ~2,660 µs of which
~976 µs is the fabric surcharge already at 78% of wire ceiling). After the
schedule knobs, communication is near its wire limit and the frontier moves
back inside the GEMMs (exp_26/27/28 line). GEMM-RS mirror: GEMM mainloop
46%, then the per-call host tax; the residual gap to rank-1 is GEMM quality.
This is itself a thesis exhibit: correctly scheduled, the distributed parts
stop being the bottleneck.

## 6. Pre-registered falsifiers (kept, deliberately)

- Any interleave/split arm beating the schedule-only arm by more than the
  +211 µs fabric-ceiling bound is a defect signature, not a result.
- If high-skew points show large `[MPS SPIN]` *and* a re-enabled elastic
  pool beats the scheduled homogeneous kernel there, the placement axis
  reopens for skewed workloads and §8 reports it as such.
- If mode 14 does not reach its 5,990–6,440 µs band, the granularity claim
  weakens to the exp_30 measurement alone.

## 7. Claim boundaries (for every abstract sentence)

Measured: single node (8×MI350X gfx950 MoE prefill; 8×MI300X gfx942
GEMM-RS graded shapes), forward-only, symmetric routing except where the
skew sweep says otherwise. Unmeasured and not claimed: multi-node, training
backward, NVIDIA transfer, decode serving end-to-end.

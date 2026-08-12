# Paper skeleton — Distributed megakernels on AMD GPUs

Structured the way COMET (arXiv 2502.19811) structures its argument:
a measured problem, an analysis of one concrete object, the obstacles that
analysis exposes, and one design per obstacle. No framework is announced up
front; every named idea first appears inside a worked example. Each section
below carries (a) the prose argument it must make and (b) the measured
artifact that backs it.

---

## 1. Introduction

**Opening problem, with a measurement (our Figure 1).** Distributed MoE and
GEMM+ReduceScatter layers spend a large fraction of their time on
inter-GPU communication: on our 8×MI350X node the dispatch and combine
phases of a production MoE layer account for roughly a third of layer time,
and on 8×MI300X the reduce-scatter epilogue of a GEMM is 32% of the largest
graded shape. The standard remedy is to overlap this communication with
computation. Existing systems do this in one of two ways: by pipelining
separate kernels on streams — too coarse, and the scheduling decisions die
at every kernel boundary — or, following COMET and Mixture-of-Kittens, by
**dedicating a group of thread blocks (or SMs) to communication** inside a
fused kernel, so that data moves while other blocks compute.

**The claim.** We built both layers as single-launch megakernels on AMD
GPUs, implemented the dedicated-block design faithfully, and measured it at
its best configuration. It works — our specialized MoE megakernel beat both
the production baseline and our own homogeneous megakernel. But when we
attributed *why* it worked, none of the benefit came from where the design
says it should: the dedicated blocks were not buying overlap. The wins came
from decisions about **when** things happen — the order producer tiles
execute, how many remote operations are in flight at once, how coarse the
completion signals are — and once those decisions were made well, the
dedicated communication blocks had nothing left to do and the kernel
degenerated, measurably faster, into a homogeneous one.

**Why AMD makes this visible.** Two hardware facts, both absent on the
NVIDIA machines where the dedicated-block design was developed, drive the
result: (i) at megakernel register/LDS budgets a CDNA GPU runs exactly one
block per compute unit, so a communication block never shares an execution
unit with a compute block — there is no issue-slot contention for
specialization to remove, only a capacity tax of exactly N/(N−C); and
(ii) the xGMI fabric is byte-limited, not operation-limited, for the
coalesced access patterns a GEMM epilogue naturally produces — so the
compute blocks can carry the communication payload themselves at
essentially zero cost.

**Contributions list** (each one is a section):
1. A measurement study of the dedicated-block design on CDNA3/CDNA4, with
   the attribution that overturns its premise (§3).
2. An analysis of the cross-GPU producer-consumer edge in these layers that
   explains the measurements and yields three design principles (§4).
3. Two megakernels built on those principles — a fused MoE layer on gfx950
   and a GEMM-ReduceScatter on gfx942 — which converge on the same protocol
   and together beat production baselines, with the remaining gap
   attributable to single-GPU GEMM quality, not communication (§5, §6).
4. Distributed HipKittens: the device-side primitives these kernels
   required, stated as contracts, with compiler/ISA evidence for why each
   is shaped the way it is (§5).
5. A benchmark methodology for distributed kernels in which correctness
   gates can actually fail: negative controls, input poisoning against
   staleness-blind gates, and counter-validated attribution (§6).

## 2. Background

2.1 **The two layers.** MoE dispatch→expert-GEMM→combine; GEMM+RS. Both
end in the same shape: tiles produced on every rank must be delivered to an
owner rank and reduced there. (One paragraph each, with the tile/token flow
figure in COMET Figure-2 style: two GPUs, a token routed to experts, the
output tile that must come home.)

2.2 **The AMD substrate.** 8 GPUs, fully connected, 7 xGMI links per GPU at
76.8 GB/s per direction each — the *link*, not the aggregate, is the unit a
kernel can saturate or leave idle. One block per CU at our budgets (ISA
§3.6.4 arithmetic shown). Directional cache operations: a producer's
release is an L2 writeback (`buffer_wbl2`), a consumer's acquire is an
invalidate (`buffer_inv`); a bidirectional `__threadfence_system` pays for
both. No dynamic register reallocation, no TMA, no mbarrier — the table of
NVIDIA mechanisms with no CDNA equivalent goes here, because it scopes
which prior designs can even transfer.

## 3. The measurement study: where dedicated communication blocks spend their advantage

This section is the paper's motivation and follows COMET §2.2's shape:
state the intuitive design, then show with measurements exactly where it
breaks.

3.1 **The design under test.** Reserve C of 256 CTAs as a service pool;
compute CTAs publish tile-completion events; the pool moves each finished
tile's bytes to its owner and posts per-row readiness flags; drained
compute CTAs join the pool (elastic fall-through). This is the COMET/MoK
prescription realized on AMD, tuned across C ∈ {2…128}, three pool
placements, and four transfer granularities. At its best point it is real:
0.888× production, ahead of our homogeneous megakernel at 0.894×.

3.2 **Finding 1: the pool's cost is not capacity, and not bytes —
it is coordination.** With CTA count held fixed, turning the pool's
traffic on inflates the concurrent GEMM by 2.8× the capacity tax. Deleting
the pool's entire payload copy moves that GEMM by +5.7 µs, while 831 µs of
inflation remains. The inflation tracks the number of atomic
counter/fence operations, not payload bytes: coarsening transfers
*increases* it because the completion-probe loop scales with the group
size. (exp_20, exp_03, exp_05–07. This is the paper's pivotal measurement
and gets its own figure.)

3.3 **Finding 2: the fine-grained readiness the pool maintains enables an
overlap that never occurs.** A combine consumer can start on a token only
when the last of its top-8 experts' tiles completes; with tiles executing
in the natural order this is `P(ready by t) = (t/S)^8` — the median token
unblocks with 8% of the producer phase remaining. Measured: nothing ever
consumed a row before the producer phase was ≥93.9% complete. The ~926,000
per-row atomics per rank per epoch that make early consumption *possible*
purchase an event with probability ≈ 0. (exp_29, exp_30.)

3.4 **Finding 3: the same benefits are available without the pool.** The
one thing the pool measurably did — limit how fast remote traffic is
injected — is available to every compute CTA directly by capping its
in-flight remote operations (worth ~500 µs, with a cliff), and the bytes
themselves can ride the producer's epilogue (next section). When we moved
the payload into the epilogue and the readiness to one arrival per source,
the optimal pool size fell from 64 to 16 and then to zero jobs: the
fastest kernel we have is again homogeneous. Independent replication: in
the GPU MODE amd-gemm-rs/all2all leaderboards, three separate top-10
authors built block-level specialization and shipped something else, and
the winning GEMM-RS kernel uses no communication blocks at all.

## 4. Analysis: the cross-GPU producer-consumer edge

This is the analog of COMET's shared-tensor analysis — one concrete object,
walked through, from which the designs are *derived* rather than asserted.

4.1 **The object.** Every cross-GPU dependency in these layers is one
edge: a producer CTA on rank r finishes a tile; the tile's bytes must reach
a landing buffer on owner rank o; the owner's consumer must learn it can
read them. So an edge = **payload movement** + **a completion signal**
(release fence, flag store, poll, acquire). Everything in §3 is a statement
about who executes these two halves and how often.

4.2 **Worked example (the paper's Figure 4).** Follow one expert-GEMM
output tile in the dedicated-block design: epilogue writes it locally →
event enqueued → pool CTA claims the event, re-reads the tile, writes it to
rank o, bumps per-row arrival counters (an acq-rel RMW per slice), posts a
per-row flag → owner polls the flag, acquires, reduces. Count the
operations: the payload crosses the fabric once, but the edge executes
~30 coordination operations across three CTAs, and every one of the pool's
fences and RMWs lands in the L2s the GEMM is streaming through. Now the
same tile in our final design: the epilogue's existing atomic-add
instructions are pointed at rank o's landing slot (same instruction count
as writing locally — the fabric is byte-limited, measured at 52.8 GB/s for
coalesced 4 B atomics vs 54.9 GB/s for 16 B stores); when all of rank r's
tiles are done, rank r posts *one* arrival signal per destination; the
owner waits for 8 arrivals and reduces. The edge's coordination cost fell
from ~30 operations on three CTAs to amortized ~10⁻⁴ operations, and no
third CTA exists.

4.3 **Three principles, each answering one finding.**
- *Answering Finding 3 (bytes):* **the producer's epilogue is the right
  transport** — it already has the data in registers, already issues wide
  memory operations with deep pipelining, and on a byte-limited fabric the
  redirected instructions cost nothing extra. The one new requirement is a
  bound on in-flight remote operations, because unthrottled bursts expose
  fabric acknowledgment latency (the measured cliff: depth 4–8 optimal,
  16 catastrophic).
- *Answering Finding 2 (signals):* **signals should be sized to the
  readiness curve, not to the data layout.** If consumers cannot act before
  the producer phase is ~94% done, per-row signals are pure interference;
  one arrival per (source, destination) pair carries the same information.
  Conversely, if earlier consumption is wanted, the lever is not finer
  signals — it is *producer task order*: executing tiles
  consumer-major turns the readiness curve from (t/S)^8 into a staircase
  that unblocks 15/16 of the combine early.
- *Answering Finding 1 (the third CTA):* **no CTA should exist whose job is
  an edge's coordination.** Specialized CTAs remain only where the work
  functionally must run on the owner (the reduction) — and there, sizing is
  uncritical (a uniform 32 reducer CTAs is within noise of per-shape optima
  on all six GEMM-RS shapes).

4.4 **Task order is also the link scheduler.** A second, independent
consequence of producer-carried payload: which peer each concurrent CTA is
writing to is now set by the tile traversal order. The inherited order on
GEMM-RS kept 2.02 of 8 links busy; changing one index expression (the tile
swizzle) — moving zero bytes differently — raised link concurrency and cut
the largest shape 27.9%. The readiness reorder of 4.3 and this link
reorder are the same lever pointed at two resources, and both are only
expressible inside a persistent kernel.

## 5. Design and implementation

5.1 **The common protocol** both kernels converged on (one figure, one
table): produce → epilogue-carried delivery into per-source landing slots
on the owner (values for GEMM-RS, atomic accumulation for MoE) → one
release fence amortized over a group of tiles → per-source arrival signal →
owner-local reduction → slot reuse handshake (credit per tile, or
consume-and-zero under an epoch retirement gate).

5.2 **The primitives** (Triton-distributed §-style: each primitive stated
as the thing a kernel author cannot otherwise express, with its contract):
peer address projection that survives suballocation (`pgl.on(rank)`);
directional release/acquire split from publication so one fence covers many
signals; bounded polls whose timeout is a result the operator must handle;
replay-lifetime credits for address reuse; and — new from this work —
the throttled remote-accumulate emitter (in-flight depth as a first-class,
cliff-bearing parameter; deferred publication so the drain lands where
acknowledgments are already home) and named task-order maps
(destination-rotating, consumer-major). Compiler evidence accompanies the
shapes: named peer bases vs 80 B/lane of scratch; the caller-owned wait
result vs a changed CFG; the LDS peer table vs a 96-site spill.

5.3 **The two kernels.** MoE megakernel, gfx950: nine phases, one launch
per layer; where each principle lands (mode-12 epilogue accumulate,
depth-4 throttle, barrier + 8 arrivals, nc-major order). GEMM-RS, gfx942:
producer GEMM + reducer pool; WGM link-order, grouped release, uniform
reducer count. Both under the same validation ladder: ISA/resource parity
gates, world-8 correctness, negative controls, 600-epoch soaks, graph-mode
replay.

## 6. Evaluation

Questions, COMET-style, one subsection each:

- **Q1. How much does each decision contribute?** The waterfall: homogeneous
  → +epilogue-carried payload → +injection bound → +arrival-sized signals →
  +task reorder; both kernels. (The paper's money figure.)
- **Q2. Do dedicated communication blocks ever win?** Paired best-vs-best
  (pool design at C=64 vs final design), the pool-size sweep at the final
  design (flat→degenerate), the reducer-count sweep, and the leaderboard
  replication.
- **Q3. Where does the time go now?** Phase attribution at the final
  configuration: the two GEMM mainloops are 78% of the MoE kernel; the
  epilogue's fabric surcharge sits at 78% of the wire ceiling (≤211 µs
  theoretical headroom); communication is no longer the bottleneck — the
  frontier moves back into single-GPU GEMM quality, and the residual GEMM-RS
  gap to the leaderboard rank-1 is GEMM quality, reported as such.
- **Q4. Resource-level proof of overlap** (NanoFlow-style): saturation
  curves vs CTA count for MFMA/HBM/xGMI, isolated *and* concurrent — the
  gap between those two curve families is Finding 1 as a figure (exp_22);
  and per-layer utilization timelines for RCCL-eager vs homogeneous vs
  final (exp_23).
- **Q5. Sensitivity: batch size, sequence length, imbalance.** M ∈
  {512…4096} × routing skew std ∈ {0…0.05}; vLLM chunked-prefill on real
  prompts for the serving shape. Pre-registered: small M shifts value
  toward signal coarsening and fusion; large M toward injection bounds and
  link order; skew is the one regime where a waiting-work pool could
  re-enter (spin instrumentation decides), and it simultaneously
  de-coalesces epilogue traffic, making the injection bound *more*
  valuable. Reported either way.
- **Q6. External ladders.** MoE vs tuned production (AITER+MoRI) *and* vs
  PyTorch+RCCL eager — the literature's usual denominator — so fusion,
  scheduling, and protocol are attributed separately; GEMM-RS vs reference
  GEMM+RCCL and vs the frozen rank-1 submission, same node, same run.

## 7. Methodology: benchmarks that can fail

Fixed-input campaigns are blind to staleness: a kernel that reads last
epoch's bit-identical buffer posts the best number in the sweep. NaN
poisoning between iterations; negative controls that must fail
(dropped publication, dropped credit, rerouted slot — and the discovery
that one control was invisible until inputs changed); phase stamps
(σ ≈ 1%) vs end-to-end screens (σ ≈ 6.6%); hardware-counter validation by
collapse test (the emit-local arm must zero the fabric counter, and does,
to 0.1%). Clock pinning. This section exists because two of these traps
each cost a night of wrong conclusions.

## 8. Related work

COMET (block specialization + shared-tensor rescheduling — we adopt its
rescheduling insight and adjudicate its specialization on AMD); MoK
(megakernel, comm SMs, pull/push chosen per direction); Triton-distributed
(compiler primitives; SM-carved copy kernels; AMD backend is a reduced
subset — no notify lowering, gfx942-only); AMD SC24 vertical fusion
(epilogue-carried payload precedent, 12%); Fleet (per-XCD aggregation);
NanoFlow (intra-device analog of the same claim: resources, not roles);
FLUX, TileLink, DeepEP; the GPU MODE leaderboard dataset as an evidence
corpus.

## 9. Limitations and scope

Single node; symmetric routing except the skew sweep; forward-only MoE;
two architectures but one vendor. The dedicated-block design is adjudicated
*for workloads whose peer-wait is near zero and whose payload can ride an
existing instruction stream* — the sensitivity sweep marks the boundary,
and multi-node/skewed regimes beyond it are explicitly open.

---

## Appendix A — evidence map (claim → artifact)

| claim | artifact |
|---|---|
| pool interference is coordination, not bytes | aug10 exp_20, exp_03, exp_05–07 |
| readiness curve (t/S)^8; nothing consumes early | aug11 exp_29, exp_30 |
| capacity tax N/(N−C); one block per CU | aug10 A7 strike + exp_03 tax curve |
| fabric byte-limited; atomics = stores per byte | aug10 exp_21 ubench |
| throttle cliff (4–8 good, 16 catastrophic) | aug11 exp_24 |
| dead 448 MiB plan stores | aug11 exp_24 |
| link concurrency 2.02/8 → WGM fix −27.9% | GEMM-RS exp_08 |
| grouped release −4% | GEMM-RS E3 |
| reducer count flat at uniform 32 | GEMM-RS RESULTS.md |
| pool degeneration at final design | aug11 exp_30 (mode 14, build pending) |
| M6/M7 split loses by work conservation | aug11 exp_25 rev2 |
| leaderboard replication | docs/distributed/competition-analysis |
| staleness-blind gates + poison fix | aug11 exp_32 |

## Appendix B — reproduction matrix

1. Waterfall campaigns (Q1): mode-14 and nc-major builds pending; ~2
   node-days. 2. exp_22 saturation curves (Q4). 3. exp_23 timelines (Q4).
4. Sensitivity sweeps (Q5): MoK harness T sweep × `skewed_hot` std sweep ×
   vLLM chunk sweep; ~2 node-days. 5. Placement reruns (Q2): paired
   best-vs-best + C sweep + F1; ~1 node-day. 6. External ladders (Q6):
   Megatron/RCCL-eager arm to build; GEMM-RS ladders exist, refresh at
   final ratchet.

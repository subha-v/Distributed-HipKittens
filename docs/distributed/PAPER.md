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
its best configuration. The first conclusion was negative: blocks reserved
to *carry bytes and coordinate edges* did not buy useful overlap. Their
payload could ride the producer epilogue, their fine-grained readiness
protocol enabled almost no early consumption, and their reservation reduced
GEMM capacity. The final MoE result changes, and sharpens, that conclusion.
After reordering GEMM-2 into two column-disjoint slabs, certifying each slab
with one coarse cross-rank rendezvous, and assigning the reserved blocks to
*consume the certified first slab* while compute blocks produce the second,
specialization became strongly profitable. At C=16 this slab-certified
pipelined combine reduced the coupled GEMM-2+combine interval by 193.0 µs
and end-to-end time by 191.4 µs versus the depth-4 direct-accumulate
control. Increasing the productive consumer pool to C=24 reached 5,848.5
µs (0.7589× production), a further 443.9 µs improvement; phase attribution
for that second step is pending and we do not assign it to one mechanism.
The revised claim is therefore not "never specialize." It is: **do not
reserve blocks to represent communication; reserve them only after the
producer schedule creates certified, immutable data on which they can
retire useful downstream work inside a measured compute shadow.**

**Why AMD makes this visible.** Two hardware facts, both absent on the
NVIDIA machines where the dedicated-block design was developed, drive the
result: (i) at megakernel register/LDS budgets a CDNA GPU runs exactly one
block per compute unit, so a communication block never shares an execution
unit with a compute block — there is no issue-slot contention for
specialization to remove, and every reserved block must repay a capacity
tax of exactly N/(N−C) with critical-path work; and
(ii) the xGMI fabric is byte-limited, not operation-limited, for the
coalesced access patterns a GEMM epilogue naturally produces — so the
compute blocks can carry the communication payload themselves at
essentially zero instruction cost. M15 repays the reservation tax in a
different currency: owner-local FP32 combine that would otherwise remain
after GEMM-2. CDNA's one-block-per-CU mapping makes both the tax and the
repayment unusually explicit.

**Contributions list** (each one is a section):
1. A measurement study of two distinct forms of block specialization on
   CDNA3/CDNA4: a carrier/coordination pool that loses, and a certified
   early-consumer pool that wins (§3).
2. An analysis of the cross-GPU producer-consumer edge that replaces the
   binary "specialize or not" question with a readiness-and-critical-path
   criterion, yielding four design principles (§4).
3. Two megakernels built on those principles — a fused MoE layer on gfx950
   and a GEMM-ReduceScatter on gfx942. They share producer-carried delivery
   and coarse certification, but choose different consumer schedules based
   on whether useful owner work exists beneath the producer shadow (§5,
   §6).
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

## 3. The measurement study: when block specialization loses and wins

This section is the paper's motivation and follows COMET §2.2's shape:
state the intuitive design, show with measurements why its carrier role
breaks, then identify the changed condition under which specialization
wins.

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

3.4 **Finding 3: carrier specialization is unnecessary; consumer
specialization is conditional.** The useful transport effects of the old
pool are available without it. Compute CTAs can carry payload directly in
their epilogues, and capping their in-flight remote operations is worth
~500 µs with a sharp depth cliff. Under the natural producer order, no
actionable consumer work exists until the end, so the carrier pool
degenerates and the homogeneous kernel wins. M15 changes the premise rather
than retuning the same pool: NC-major execution completes a column-disjoint
prefix, a coarse slab certificate proves that prefix immutable across all
eight ranks, and reserved CTAs perform the final owner-local reduction on
it while the other CTAs compute the suffix. C=8, 16, and 24 improve
monotonically (6,385.2, 6,292.4, and 5,848.5 µs), reversing the earlier
placement law. Thus the operative distinction is not compute CTA versus
communication CTA; it is **coordination-only work versus useful consumer
work that can be removed from the critical path.** The GEMM-RS result and
leaderboard replication remain the complementary regime: when no such
consumer work fits under the producer shadow, no communication pool is
optimal.

3.5 **Finding 4: overlap must be created algorithmically before it can be
scheduled.** Finer readiness flags did not create overlap because the
natural top-8 readiness curve exposed no useful data. M15 instead changes
the producer order and the unit of independence together. It produces all
tiles for columns [0,3584), performs a system-visible drain and one epoch
publication per (rank, slab), then produces columns [3584,7168). The first
half is now complete, globally certified, and disjoint from every remaining
epilogue write. Sixteen service CTAs can therefore reduce it without a
row-level poll; their work is quota-bounded so they cannot delay the next
grid rendezvous. This is measured overlap, not phase relabeling: at C=16,
M7+combine falls from 3,026.1 to 2,833.1 µs, closing to within 2 µs of the
end-to-end improvement.

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
same tile in M15: the epilogue's existing atomic-add instructions are
pointed at rank o's landing slot (same instruction count as writing locally
— the fabric is byte-limited, measured at 52.8 GB/s for coalesced 4 B
atomics versus 54.9 GB/s for 16 B stores). Tasks are NC-major and divided
into two slabs. After a slab-wide drain, rank r posts one epoch word for
that slab to each destination; the owner waits for eight words and reduces
the certified columns. The edge's coordination cost falls from per-tile and
per-row operations to two coarse certificates per source, while a third CTA
role reappears only as an owner-side consumer doing arithmetic that must be
done somewhere. For slab 0, that arithmetic executes concurrently with
slab-1 production.

4.3 **Four principles, each answering one finding.**
- *Answering Finding 3 (bytes):* **the producer's epilogue is the right
  transport** — it already has the data in registers, already issues wide
  memory operations with deep pipelining, and on a byte-limited fabric the
  redirected instructions cost nothing extra. The one new requirement is a
  bound on in-flight remote operations, because unthrottled bursts expose
  fabric acknowledgment latency (the measured cliff: depth 4–8 optimal,
  16 catastrophic).
- *Answering Finding 2 (signals):* **signals should be sized to the
  consumer's executable frontier, not to the storage layout.** If consumers
  cannot act before the producer phase is ~94% done, per-row signals are
  pure interference. Earlier consumption requires changing producer task
  order so that a large independent region becomes complete; only then
  should a signal certify that region. M15's unit is one column slab per
  source, not one row and not necessarily the whole epoch.
- *Answering Finding 4 (the overlap frontier):* **manufacture an immutable
  prefix, then certify it coarsely.** The producer order, memory layout, and
  certificate boundary must agree. M15 chooses the 3,584-column boundary
  because eight 448-column GEMM chunks align exactly with seven 1,024-byte
  combine chunks. Once slab 0 is drained and certified, slab-1 RMWs are
  provably column-disjoint, so reduction and consume-and-zero are safe.
- *Answering Findings 1 and 3 (the third CTA):* **specialize CTAs for
  critical-path work, never merely for an edge's coordination.** A pool
  that polls, claims, copies, and republishes loses. A pool that performs
  required owner-local reduction can win if that reduction fits beneath a
  producer shadow. Its sweep must be quota-bounded rather than exhaustive,
  and after the shadow closes the whole grid must elastically join the same
  ticket queues. C is consequently a joint producer/consumer scheduling
  parameter, not a communication-bandwidth knob.

4.4 **The critical interval, not phase-local time, is the objective.** M7
and combine are anti-correlated: ending M7 earlier can merely move peer
waiting into combine. We therefore optimize and report the interval from
GEMM-2 entry through final reduction. M15 C=16 improves that coupled block
by 193.0 µs and the kernel by 191.4 µs. The C=24 headline is fully gated,
but until its timestamp rotation decomposes the additional 443.9 µs, we
report only the joint effect. Plausible contributors — more early reduction,
lower aggregate RMW pressure, and a different discrete task/barrier tail —
remain hypotheses rather than paper claims.

4.5 **Task order is also the link scheduler.** A second, independent
consequence of producer-carried payload: which peer each concurrent CTA is
writing to is now set by the tile traversal order. The inherited order on
GEMM-RS kept 2.02 of 8 links busy; changing one index expression (the tile
swizzle) — moving zero bytes differently — raised link concurrency and cut
the largest shape 27.9%. The readiness reorder of 4.3 and this link
reorder are the same lever pointed at two resources, and both are only
expressible inside a persistent kernel.

4.6 **A decision rule for specialization.** For a candidate pool size C,
let `H(C)` be downstream critical-path work actually completed before the
producer shadow ends; let `P(C)` be the producer slowdown from removing C
CTAs plus any change in injection and tail balance; and let `R(C)` be the
added rendezvous, polling, and ticket overhead. Specialization is useful
only when the measured inequality `H(C) > P(C) + R(C)` holds on the coupled
producer-consumer interval. Carrier pools have `H(C) ≈ 0` and therefore
lose even when their copy bandwidth is high. M15 makes `H(C)` nonzero by
construction, bounds `R(C)` with two slab words per source and a quota, and
then tunes C jointly rather than assuming the smallest communication pool
is optimal. This is a decision procedure, not an analytic performance
model: the C=24 result shows that `P(C)` includes nonlinear fabric pressure
and discrete barrier-tail effects that a bandwidth-only model misses.

## 5. Design and implementation

5.1 **The common protocol** both kernels converged on (one figure, one
table): order production to expose an independently consumable region →
carry delivery in the producer epilogue into per-source landing slots on
the owner (values for GEMM-RS, atomic accumulation for MoE) → bound remote
operations in flight → drain once per independent region → publish one
coarse certificate per source and region → perform owner-local reduction,
concurrently with later production when a certified shadow exists → reuse
slots only after a credit or consume-and-zero retirement edge. The protocol
does not prescribe a permanent service pool: GEMM-RS has none for transport;
M15 temporarily assigns C CTAs to early reduction and rejoins the full grid
after the second slab.

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

5.3 **The two kernels.** MoE M15 megakernel, gfx950: nine phases and one
launch per layer; mode-12 direct remote accumulate, depth-4 deferred drain,
NC-major two-slab M7, two slab rendezvous, and a quota-bounded dynamic-ticket
combine sweep. At the C=16 attribution point, 240 CTAs produce both slabs;
16 CTAs consume slab 0 during slab 1; all 256 finish front leftovers and
the back half after the second rendezvous. The best gated configuration so
far uses C=24, g=353, mode=12, flush_rows=16. GEMM-RS, gfx942: producer GEMM
+ reducer pool; WGM link-order, grouped release, uniform reducer count. Both
use the same validation ladder: ISA/resource gates, world-8 correctness,
negative controls, 600-epoch soaks, and graph-mode replay. The staged
wide-push M15b transport is built but unmeasured and is not part of the
claimed design or result.

## 6. Evaluation

Questions, COMET-style, one subsection each:

- **Q1. How much does each decision contribute?** The waterfall: homogeneous
  → +epilogue-carried payload → +injection bound → +coarse signals →
  +NC-major slabs → +certified mid-M7 consumer sweep; both kernels where
  applicable. Include the `flush_rows=0` ablation to separate protocol
  deletion from early consumption. (The paper's money figure.)
- **Q2. When do specialized CTAs win?** Compare three matched roles: idle
  reservation, carrier/coordination, and quota-bounded owner reduction.
  Report the M15 C sweep (8, 16, 24, and C=32 only if its intermittent
  liveness failure is resolved), the reducer-count sweep, and leaderboard
  replication. The claim is role- and frontier-dependent, not a universal
  preference for homogeneous or heterogeneous grids.
- **Q3. Where does the time go now?** Report the coupled producer-consumer
  interval, not M7 and combine independently. At C=16, M7+combine is
  2,833.1 µs versus 3,026.1 µs for the matched control; residual post-M7
  combine is 180.0 versus 324.2 µs. Add the queued C=24 timestamp
  decomposition before attributing its further 443.9 µs gain. The residual
  GEMM-RS gap to leaderboard rank-1 is GEMM quality, reported as such.
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
rescheduling insight and separate carrier specialization from certified
consumer specialization on AMD); MoK (megakernel, comm SMs, pull/push
chosen per direction); Triton-distributed
(compiler primitives; SM-carved copy kernels; AMD backend is a reduced
subset — no notify lowering, gfx942-only); AMD SC24 vertical fusion
(epilogue-carried payload precedent, 12%); Fleet (per-XCD aggregation);
NanoFlow (intra-device analog of the same claim: resources, not roles);
FLUX, TileLink, DeepEP; the GPU MODE leaderboard dataset as an evidence
corpus. The novelty claim is the measured composition, not any ingredient
alone: producer-carried bounded RMW transport, an order-created
column-disjoint readiness frontier, slab-level cross-rank certification,
and quota-bounded elastic consumer CTAs that reduce the certified prefix
beneath suffix GEMM execution.

## 9. Limitations and scope

Single node; symmetric routing except the skew sweep; forward-only MoE;
two architectures but one vendor. M15 is validated only at the current
T=4096 shape; the known T=1024/2048 correctness defect blocks other-size
claims. C=24 has five clean rotations but not yet phase attribution. C=32
showed an intermittent run with no rank JSONs and is excluded until kernel
versus harness liveness is diagnosed. M15b's staged wide-push path is built
but unmeasured. More generally, the revised specialization rule is
adjudicated only where the payload can ride an existing instruction stream
and a large disjoint consumer region can be certified cheaply. Multi-node,
strongly skewed, and layouts without such a frontier remain open regimes.

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
| carrier-pool degeneration before M15 | aug11 exp_30 (mode 14, build pending) |
| M15 C=16 end-to-end −191.4 µs; M7+combine −193.0 µs | aug12 exp_03 M15 result, e03camp15/e03stamps15 |
| productive-pool inversion, C=8 → 16 | aug12 exp_03 M15 result |
| M15 C=24 5,848.5 µs (0.7589×), five clean rotations | aug12 C sweep; phase attribution pending |
| slab-word protocol and column-disjoint safety argument | aug12 M15_DESIGN.md + k0pf6gm_device_tile_m15.hip |
| M6/M7 split loses by work conservation | aug11 exp_25 rev2 |
| leaderboard replication | docs/distributed/competition-analysis |
| staleness-blind gates + poison fix | aug11 exp_32 |

## Appendix B — reproduction matrix

1. Complete M15 `flush_rows=0` attribution, C=24 timestamp rotation, and
   C=32 liveness diagnosis; do not attribute the C=24 gain before these.
2. exp_22 saturation curves (Q4). 3. exp_23 timelines (Q4).
4. Sensitivity sweeps (Q5): MoK harness T sweep × `skewed_hot` std sweep ×
   vLLM chunk sweep; ~2 node-days. 5. Placement reruns (Q2): paired
   role-matched idle/carrier/consumer arms + C sweep; ~1 node-day.
6. External ladders (Q6):
   Megatron/RCCL-eager arm to build; GEMM-RS ladders exist, refresh at
   final ratchet.

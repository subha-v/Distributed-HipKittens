# Communication/Computation Overlap Methodology Study

- **Status:** research synthesis and revised experiment roadmap
- **Date:** 2026-08-12
- **Primary platforms:** HipKittens and IRIS on AMD MI350X/MI355X
- **Initial scope:** intra-node GPU communication; scale-out RDMA is a later tier

> Internal research note. Measurements cited from Confluence must be checked for
> publication clearance and reproduced on a pinned public stack before they are
> used as paper claims.

## Executive summary

The project should not be framed as finding one universally best MoE
megakernel. Its more valuable research question is:

> Given a workload and topology, which communication/computation overlap
> pattern minimizes synchronized end-to-end global makespan, and why?

The current fused-MoE kernel is one strong point in that regime map:

- 8× MI350X, `T=4096`, top-k 8, hidden 7168, 32 experts/rank.
- Synthetic balanced routing, eager execution, one MoE layer, no shared expert.
- The current internal ratchet is 6,482.7 µs versus roughly 7,710 µs for the
  AITER+MORI production arm: 15.92% lower latency.
- Compute CTAs carrying remote payload are not sufficient by themselves. The
  decisive mechanism is a depth-4 bound on outstanding remote writes.
- Within valid mode 12 at this point, paired `C=4` and `C=8` both beat `C=16`;
  larger-pool point estimates are worse. The `C=64` result uses a different
  complete method, and the available `C=0` result came from a confounded
  mode-14 binary.

Those facts do **not** establish that CTA specialization is generally bad, that
SDMA is generally good or bad, or that push is generally better than pull. The
current benchmark covers only one large, balanced prefill point and uses one
specific dependency graph.

The next step should therefore be a simple synthetic overlap suite, not an
SGLang serving factorial and not another immediate modification to the MoE
megakernel. The suite should:

1. Use one neutral producer → transport → consumer DAG.
2. Hold payload, compute, routing, and correctness semantics constant.
3. Compare communication carriers, directions, engines, synchronization
   granularities, flow-control policies, and scheduling orders separately.
4. Measure concurrent makespan and mutual slowdown, not bandwidth alone.
5. Sweep message size, compute/communication ratio, fanout, skew, and topology.
6. Implement the same semantic benchmark in HipKittens and IRIS.
7. Promote only crossover-adjacent winners into MoE prefill, decode, and
   training proxies.

The intended paper result is a **method-selection map**, not a claim that one
method always wins.

## What this study can and cannot settle

The neutral benchmark is necessary, but it cannot by itself determine the best
MoE kernel. It can isolate why a mechanism wins—engine cost, direction,
resource reservation, readiness, flow control, or task order—and eliminate
mechanisms that lose in every relevant regime. The final kernel decision must
come from a second-level tournament using complete MoE implementations.

The current ratchet is `0.8408×` the production arm's latency, meaning 15.92%
lower latency. It is not “84% of production performance.” It is the anchor for
large, balanced `T=4096` prefill, not evidence that the same architecture is
optimal for decode or training.

The existing experiment corpus should be closed as follows:

- **Closed for the current `T=4096` balanced regime:** large fixed service
  pools are unattractive; producer-carried communication alone is
  insufficient; depth-4 remote-operation admission is decisive; M7+combine is
  the coupled critical region.
- **Needs a clean rerun:** coarse mode 14 from a healthy codegen baseline and
  same-run M7 GEMM/epilogue attribution.
- **Blocked by correctness:** all current megakernel conclusions at
  `T=1024/2048`.
- **Still unresolved:** universal CTA-specialization claims, push versus pull,
  SDMA versus CU transport, fine versus coarse readiness, global versus
  per-destination credits, skew, topology, separate persistent kernels,
  multiple streams, TBO, decode, and training.

The study succeeds only if it produces an actionable rule of the form:

> For this tokens/expert distribution, payload granularity, compute shape,
> topology, and latency/throughput objective, select this complete overlap
> policy.

Bandwidth curves or synthetic winners that do not predict the complete MoE
tournament are diagnostic results, not kernel-design conclusions.

---

## 1. Research questions

### RQ1 — Who carries communication?

- Permanently reserved communication CTAs.
- Temporarily specialized or work-stealing CTAs.
- Specialized warps within a CTA.
- Homogeneous compute CTAs that also carry their payload.
- The producer's epilogue writing directly to the remote destination.
- A separate persistent communication kernel.
- A host- or shader-initiated SDMA engine.
- A collective library kernel with an explicit CU partition.
- A persistent actor scheduler assigning ready tasks out of order.

### RQ2 — Who initiates and owns data movement?

- **Push:** the source writes into destination memory.
- **Pull:** the destination reads from source memory.
- **Hybrid:** source publishes metadata; destination chooses what to fetch.
- **Hierarchical:** push within a locality domain and stage/pull across a more
  expensive link.

### RQ3 — Which transport engine carries the bytes?

Keep three levels separate:

- **Hardware engine:** CU, SDMA copy engine, or NIC.
- **Operation:** remote load, store, atomic, copy descriptor, or collective.
- **Software layer/API:** HipKittens, IRIS, MORI, rocSHMEM, RCCL, or HIP.

All intra-node contestants use an underlying peer fabric such as XGMI. The
software layer does not uniquely determine the hardware engine; an executor
trace must establish which path a measured arm actually used.

“SDMA versus XGMI” is not technically the right comparison: SDMA is an
execution engine, while XGMI is the fabric. The controlled question is:

> SDMA-engine transfer versus CU-issued remote access over the same XGMI path.

### RQ4 — When is data declared ready?

- Per tile or record.
- Per row or segment.
- Per destination group.
- Per source/destination epoch.
- Whole-phase or grid-barrier completion.

### RQ5 — How is communication admitted?

- No explicit bound.
- One global outstanding-operation credit.
- Per-destination credits.
- Per-link or per-XCD credits.
- Static pacing.
- Measured-backlog or readiness-driven admission.

### RQ6 — How are producer tasks ordered?

- Existing tile/expert-major order.
- Destination-major order.
- Round-robin destination order.
- Consumer-major order: produce the next consumable group first.
- Remote-critical first, local work last.
- Dynamic priority based on destination backlog.
- Readiness-driven global work queues with out-of-order assignment.

### RQ7 — At what scope is overlap created?

- Within one CTA.
- Between roles in one kernel.
- Between two persistent kernels.
- Between HIP streams.
- Within one batch through chunked communication/compute pipelines.
- Between microbatches or requests.
- Between independent model branches, such as shared and routed experts.
- Between adjacent layers or pipeline stages.

---

## 2. A factorized taxonomy

These dimensions are factorized but constrained, not fully orthogonal.
Treating “SDMA,” “CTA specialization,” or “push” as complete methods hides
important interactions and illegal combinations.

| Dimension | Representative choices | Question answered |
|---|---|---|
| Issuing role | producer CTA, service CTA, specialized warp, persistent consumer, host | who issues payload operations? |
| Execution container | same CTA, role-split kernel, separate persistent kernel, stream/graph | where does that role execute? |
| Hardware engine | CU, SDMA, NIC | which hardware performs movement? |
| Operation | remote load, store, atomic, copy, collective | what memory/communication action is issued? |
| Software layer | HipKittens, IRIS, MORI, rocSHMEM, RCCL, HIP | which runtime and API supply the operation? |
| Direction | push, pull, hybrid | where is routing knowledge and where does contention land? |
| Materialization | direct from registers, local stage then copy, remote source-separated inbox | what extra local traffic is required? |
| Readiness | fine, grouped, destination epoch, global epoch | how early can useful consumer work begin? |
| Flow control | none, global depth, per-destination depth, adaptive | what bounds in-flight fabric work? |
| Task order | tile-major, destination-major, round-robin, consumer-major | what readiness curve is produced? |
| Topology policy | flat, locality-aware, hierarchical | which links are used and in what order? |
| Pipeline scope | same item, different item, microbatch, branch, layer | which dependencies actually permit overlap? |

Maintain a legality matrix in the benchmark schema. For example, producer
register-direct materialization requires a producer-side CU role; a
host-enqueued copy-engine arm necessarily starts from materialized memory; and
some MORI APIs can select
P2P, RDMA, or SDMA depending on runtime configuration.

Complete overlap patterns are combinations of these choices. The initial
contestants should include:

1. Bulk-synchronous produce → communicate → consume.
2. Reserved service CTAs performing staged push.
3. Homogeneous producer CTAs performing staged push.
4. Direct producer-epilogue remote stores.
5. Direct producer-epilogue remote accumulation.
6. Destination-side pull.
7. Two persistent producer/consumer kernels.
8. Host-enqueued copy-engine transfer, labeled SDMA only when verified.
9. Shader-initiated MORI SDMA.
10. RCCL/collective overlap with a fixed CU partition.
11. Readiness-driven actor scheduling with temporal multi-buffering.
12. Chunked single-batch communication/compute pipelining.
13. Shared-expert/routed-expert branch overlap.
14. Cross-microbatch or cross-request stream overlap.
15. Hierarchical intra-domain push plus inter-domain collective/RDMA.

The suite should first measure the factors independently, then recompose only
the competitive combinations. A full Cartesian product would be both
prohibitively large and difficult to interpret.

The taxonomy describes the full design space. The initial method-selection map
is scoped to contestants actually implemented in Stages 0–8. Warp-specialized
and FlashMoE-style dynamic actor schedulers enter only through Stage 9 and
cannot appear in winner claims before that stage is run.

---

## 3. Workload and regime axes

### 3.1 Fundamental controlled variables

| Axis | Initial values | Why it matters |
|---|---|---|
| Ranks | 2, 4, 8 | changes fanout, synchronization, and aggregate injection |
| Record size | 256 B, 1 KiB, 4 KiB, 14,336 B, 64 KiB, 256 KiB, 1 MiB, 4 MiB, 16 MiB, 64 MiB | spans control-dominated decode records through SDMA-friendly collective chunks |
| Total bytes/rank | 4 MiB, 64 MiB; separately fixed record count | separates message granularity from total volume |
| Compute/communication ratio | isolated `Tc/Tm` ≈ 0.25, 1, 4 | determines the available overlap envelope |
| Fanout | 1, 2, `P-1` | controls destination concurrency and link pressure |
| Hot-destination share | uniform, 0.5, 0.9 | creates deterministic skew and incast |
| Traffic pattern | one-way, bidirectional pair, ring, all-to-all, all-to-one | changes contention and tail behavior |
| Consumer work | checksum, reduction, configurable MFMA | distinguishes transport from dependent compute |
| Buffer lifetime | one epoch, ping-pong, ring depth 4 | prices reusable persistent protocols |

Use exact deterministic destination counts. A requested “routing standard
deviation” is insufficient because two distributions with the same standard
deviation can create very different hottest-link and hottest-rank tails.

### 3.2 Application regimes to emulate later

| Regime | Structural properties | Pre-registered expectation, not a conclusion |
|---|---|---|
| Low-token decode | few records, protocol/launch dominated, little grouping | persistent direct communication or coarse epochs may beat SDMA setup |
| High-concurrency decode | many independent requests and shared/routed branches | SDMA and stream overlap may win by freeing CUs for independent work |
| Prefill | hundreds to thousands of tokens, large grouped GEMMs, broad fanout | task order, direct producer writes, and bounded injection may dominate |
| Training/FSDP | multi-MiB collectives, long independent GEMMs, regular tensors | SDMA or overlap-aware RCCL is likely competitive because setup amortizes and CU opportunity cost is high |
| Highly skewed MoE | hot destinations, stragglers, uneven readiness | per-destination credits, pull, or adaptive roles may become valuable |
| Multi-node EP | topology hierarchy and NIC progress dominate | hierarchical transport and device-initiated RDMA become first-order |

The suite must be allowed to disprove all of these expectations.

---

## 4. Evidence already available

### 4.1 Local fused-MoE evidence

| Evidence | Measured result | What it establishes | Scope limit |
|---|---|---|---|
| exp_22 saturation | MFMA scales linearly to 256 CTAs; reserving 16 CTAs costs 6.25% of the machine. Median matched-pair compute interference is 0.073% and rr7 is ≤0.23%, but saturated single-link points show 7.4–14.6% costs and the HBM body costs 10.1% | large fixed reservation is expensive and most real-fanout points show little issue-slot contention to relieve | fixed-compute-footprint control is still needed across CTA counts |
| exp_37 placement | valid mode-12 `C=4` and `C=8` both beat `C=16` by 27–34 µs in paired rounds; larger-pool point estimates are worse | smaller permanent reservations beat `C=16` in the tested MoE regime | `C=4` and `C=8` are tied; `C=64` is mode 2 and is not one point on a causal mode-12 C-axis |
| exp_35 waterfall | complete-method unthrottled mode 12 is +211 µs versus `pf6gm`; within mode 12, depth 4 then recovers 615 µs | admission depth is decisive and must be evaluated with the carrier | the baseline→unthrottled step changes more than payload placement and is not a payload-only ablation |
| exp_21/24 | depth ≤8 avoids a large M7 cliff; depth 4 is best measured | outstanding-operation control is a first-class overlap mechanism | global depth only; no per-destination comparison |
| exp_20 | deleting the staged payload copy changes M7 by only 5.7 µs while protocol interference remains | payload bandwidth alone does not explain the service-pool penalty | current event-starved service protocol |
| exp_18/21 | direct packed-bf16 remote accumulation is correct under tested contention and removes a local reread/copy | direct producer publication is viable on the MORI uncached heap | correctness is empirical for this heap/API combination |
| exp_29 analysis | current order gives `P(token ready by t)=(t/S)^8`; median readiness is 91.7% through M7 | fine signaling has little useful shadow under the current order | model only; readiness instrumentation and reorder were never run |
| exp_34/38 | coarse mode 14 removes major protocol work but was measured in a binary where codegen disabled the intended throttle | compiler code generation can change the effective overlap policy | coarse-vs-fine interaction remains unresolved |
| exp_36 | `T=1024/2048` are incorrect in both megakernels; only `T=4096` is valid | current results cannot support a batch-size regime claim | correctness defect must be fixed before application sweeps |

Important supersessions:

- exp_20 supersedes earlier claims that the service-pool problem is simply
  payload volume.
- exp_38 shows that resource counts alone are not a sufficient binary-parity
  gate; spill counts, scratch operations, issue-run structure, and preferably
  `.text` identity matter.
- M7 and combine are anti-correlated. The correct objective is their sum, not
  either phase alone.
- The current harness is a synthetic eager prefill diagnostic, not serving
  evidence.

### 4.2 Internal AMD evidence

#### MORI SDMA case study

[MORI - SDMA Case Study](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1696994052/MORI+-+SDMA+Case+Study)

- Shader-initiated SDMA supports AllReduce, AllToAll, and AllGather.
- The page reports that, for common 2–64 MiB AllReduce sizes, MORI is slightly
  faster than RCCL and emphasizes that its main value is freeing CUs during the
  AllGather phase.
- An AllGather+GEMM overlap unit test reports roughly 15% improvement over
  RCCL.
- A high-concurrency DeepSeek-R1 decode run (`ISL=1`, `OSL=64`,
  concurrency 2,048) reports TPOT 97.35 → 83.32 ms and throughput
  20,094 → 23,234 tok/s for AsyncLL SDMA plus dual-stream overlap.
- The serving comparison changes transport, stream use, and
  `GPU_MAX_HW_QUEUES` together. It does not isolate the SDMA main effect.

The inspected public
[AsyncLL implementation](https://github.com/ROCm/mori/blob/main/src/ops/dispatch_combine/low_latency_async.cpp)
and [MORI IR guide](https://github.com/ROCm/mori/blob/main/docs/MORI-IR-GUIDE.md)
indicate the following mechanism. Exact symbols and semantics are
version-dependent and must be re-audited at the pinned benchmark commit:

- Shader kernels pack and assign destination slots.
- A nonblocking put API submits contiguous payload transfer.
- A transport-specific completion/quiet operation orders payload visibility
  before a count/barrier publication.
- Shader kernels unpack dispatch or reconstruct combine.
- AsyncLL launches five dispatch kernels and four combine kernels. “CU-free”
  describes the bulk transfer, not the full operation.

#### Low-latency EP record sweep

[Low Latency EP Dispatch + Combine Optimization](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1688685557/Low+Latency+EP+Dispatch+Combine+Optimization)

On 8× MI300X, EP8, top-k 2, hidden 8192 BF16:

- 2 records/rank, 32 KiB payload: 9.07 µs dispatch+combine.
- 128 records/rank, 2 MiB payload: 101.11 µs.
- Control publication dominates the smallest cases.
- At larger batches, the hottest source/destination lane controls latency.
- This is already a strong record-count/top-k/hidden-size sweep, but expert
  compute is a no-op and it does not compare push, pull, or SDMA.

#### MORI dispatch/combine ablations

[MORI dispatch-combine kernels performance analysis](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1734250005/MORI+dispatch-combine+kernels+performance+analysis)

- Replacing one global combine barrier with per-record flags gives no benefit.
- `s_sleep` polling backoff gives no benefit.
- A deterministic source-owned SPSC destination range removes remote slot
  reservation and improves dispatch 3.6% at `nbs=1` and 6.8% at `nbs=64`.
- An LL128-style per-record flag path is 44% slower at `nbs=1` and 177% slower
  at `nbs=64` because it adds a deflag copy without removing the downstream
  whole-buffer dependency.

This directly supports the principle that readiness granularity is useful only
when it changes when dependent work can execute.

#### GEMM/RCCL overlap

[GEMM/RCCL overlap optimization on MI355](https://amd.atlassian.net/wiki/spaces/DCGPUAIST/pages/1621983644/GEMM+RCCL+overlap+optimization+on+MI355)

- Overlap-aware Stream-K and RCCL WarpSpeed tuning produces 2.3–4.6% model
  improvements in some cases.
- Standalone-best GEMMs need not be overlap-best.
- Compute grid shape and communication CU allocation must be tuned together.

[Gemm-RCCL overlap micro-benchmarking](https://amd.atlassian.net/wiki/spaces/FPS/pages/1609469800/Gemm-RCCL+overlap+micro-benchmarking)

- Existing infrastructure already sweeps communication CUs
  `{1,16,32,48,56}` and Stream-K settings `{0..6}`.
- A new generic suite should not repeat that exact RCCL/Stream-K grid.
- RCCL should instead appear as a reference contestant at its existing tuned
  point.

#### Transfer and topology measurements

[TransferBench](https://amd.atlassian.net/wiki/spaces/DCGPUCEVAL/pages/804280169/TransferBench)

- Existing tooling can target GFX or individual SDMA executors.
- A 64 MiB SDMA example reports remote links mostly around 46–49 GB/s, with
  some pairs around 30–38 GB/s.
- It measures transfer throughput and link variation, but not matched CU versus
  SDMA execution under concurrent compute.

[GPU Interconnect Bandwidth](https://amd.atlassian.net/wiki/spaces/RPLBAS/pages/996537967/GPU+Interconnect+Bandwidth)

- Existing campaigns already characterize large 512 MiB peer bandwidth and
  all-to-all ceilings across MI300X/MI325X/MI350X/MI355X.
- Peak P2P/A2A measurement is therefore a calibration, not a new paper result.

[Cross-GPU GEMM on MI350X (CPX)](https://amd.atlassian.net/wiki/spaces/GPUCPT/pages/1703348879/Cross-GPU+GEMM+on+MI350X+CPX+5+Penalty+at+Remote+OAM+NPS+Mode+Negligible)

- A CPX FP32 GEMM with matrices placed on a distant logical peer falls from
  roughly 17 to 3.5 TFLOPS.
- The result proves that compute directly consuming remote memory can be
  topology-sensitive.
- It must not be treated as a peer-copy result or directly generalized to the
  current SPX node.

#### Skew and dynamic balancing

[SGLang WideEP EP16 EPLB analysis](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1734808757/SGLang+WideEP+EP16+EPLB+Perf+analysis+-+2P2D+-+MI300+CX7)

- Significant per-layer skew did not translate into an EPLB throughput win.
- Online migration introduced large stalls; chunking reduced but did not
  remove the cost.
- “There is skew” is not enough. The benchmark must expose whether skew lands
  on the critical path and whether a proposed policy costs more than it saves.

#### Unavailable internal evidence

The searchable page
[sDMA-based Collective Communication Summary](https://amd.atlassian.net/wiki/spaces/~712020ea4fade82ae94a95b7c0ba1cb554d2a8/pages/972308583/sDMA-based+Collective+Communication+Summary)
returns 404 through the Confluence page API for the current account. Its
specific numbers are not treated as audited evidence here.

### 4.3 Public prior art and novelty boundary

[FlashMoE](https://proceedings.neurips.cc/paper_files/paper/2025/file/918d938bd209e5b56072777366f8a211-Paper-Conference.pdf)
fuses scheduling, expert
computation, and device-initiated NVSHMEM communication into one persistent
actor-style kernel. Its temporal symmetric-buffer layout avoids write
conflicts and bulk collectives. On 8× H100 it reports up to 5.7× throughput
versus its baselines. A persistent megakernel plus one-sided communication is
therefore prior art, as are administrative warp/block specialization,
readiness-driven task queues, out-of-order assignment, and temporal
multi-buffering. An AMD method-selection study and its flow-control findings
can still be new.

[COMET](https://arxiv.org/abs/2502.19811) decomposes shared tensors, reorders
tasks, horizontally fuses communication and compute, and selects
shape-dependent thread-block partitions from profiled/precompiled variants.
It reports 1.96× single-layer and 1.71× average end-to-end speedup.
Fine-grained dependencies and adaptive CTA assignment are not novel by
themselves. A novel contribution must explain the AMD regime boundaries or
derive a different scheduler from measured engine/direction/credit behavior.

[Fine-grained MoE tile signaling and scheduling](https://arxiv.org/abs/2607.19539)
uses a persistent GEMM producer, a separate persistent communication kernel on
dedicated SMs, tile-level epilogue flags, remote-owner-aligned layout, segment
transfers, a higher-priority communication stream, tuned SM partitioning, and
remote-first task order. It reports up to 2.64× end-to-end speedup on 4× A100.
This is a direct prior-art match for Stages 4, 6, and 7; the AMD experiments
must test when its dedicated-consumer design survives the CU capacity tax.

[Lancet](https://arxiv.org/abs/2404.19429) expands overlap beyond one MoE layer.
It partitions and pipelines forward computation and schedules backward weight
gradient work against AllToAll, reporting up to 1.3× end-to-end training
speedup. This motivates treating pipeline scope as an experimental dimension
rather than assuming the best overlap must live inside one operator.

[FasterMoE](https://doi.org/10.1145/3503221.3508418) splits AllToAll and expert
computation into smaller groups and pipelines them.
[Tutel](https://arxiv.org/abs/2206.03382) searches AllToAll algorithm and
pipeline degree, while
[PipeMoE](https://doi.org/10.1109/INFOCOM53939.2023.10228874) models and selects
pipeline degree. Cross-microbatch overlap is likewise present in Lancet and
[ATOM's TBO design](https://github.com/ROCm/ATOM/blob/main/recipes/TBO.md).
Stage 3 should characterize pipeline-degree and GEMM-efficiency crossovers,
not claim any of these patterns as new.

[David's pinned Iris two-shot GEMM-AllReduce](https://github.com/ROCm/iris/blob/e4ab33623d5dae4e06421a6311bf9890ce9afcc7/iris/ops/matmul_all_reduce.py#L131-L198)
assigns owner partitions to GEMM publishers. Each participating producer
program publishes one release increment for its owner partition, and the
consumer waits for the aggregate expected count. It demonstrates that coarse
readiness can be the natural consequence of producer order and output
ownership, rather than a weaker approximation to per-tile signaling.

The [IRIS paper](https://arxiv.org/abs/2511.12500) already presents and measures
bulk-synchronous, unfused producer-consumer, fused sequential, and fused
workgroup-specialized patterns on 8× MI300X, reporting up to 1.79× over
PyTorch+RCCL for GEMM+All-Scatter. The publishable extension here is not
another placement taxonomy; it is the controlled CDNA4 crossover map that adds
direction, engine, readiness, flow control, topology, and their interactions.

### 4.4 Experiments not worth repeating unchanged

1. A generic peak-bandwidth P2P or A2A sweep.
2. The existing RCCL WarpSpeed communication-CU × Stream-K sweep.
3. Another unrestricted service-pool size sweep on the current MoE shape.
4. Another `nbs/topk/hidden` transport-only EP sweep without a new method axis.
5. Poll-backoff-only experiments.
6. Per-record flags where the consumer still waits for the whole buffer.
7. A routing-skew study that does not record the realized per-destination
   counts and critical-path tail.
8. Any non-`T=4096` current-megakernel timing before correctness is repaired.
9. Any end-to-end serving factorial before the simple method crossovers are
   known.

### 4.5 Important gaps that remain

- CU push versus CU pull with identical work and topology.
- Verified host SDMA versus shader-initiated SDMA versus CU transfer over the
  same XGMI path across small and large messages.
- Mutual slowdown under concurrent compute, not isolated bandwidth.
- Producer epilogue versus homogeneous staged push versus reserved service CTA
  on one neutral DAG.
- In-kernel roles versus two persistent kernels at the same CU reservation.
- Global versus per-destination credits under fanout and skew.
- Task order × readiness granularity interaction.
- A method-selection map across decode-, prefill-, and training-like regimes.
- A matched HipKittens/IRIS implementation.
- Topology-class-controlled overlap and multi-node RDMA.

---

## 5. Experimental principles

### 5.1 Separate transport from end-to-end algorithmic effects

Two comparison families are required.

**Transport-equivalent comparisons**

- Every method receives an already-materialized source buffer.
- Payload bytes, destination offsets, signal semantics, and consumer checksum
  are identical.
- This isolates direction and engine: CU push, CU pull, verified host SDMA,
  and shader-initiated MORI SDMA.

**Algorithmic end-to-end comparisons**

- The producer computes the payload.
- Direct epilogue arms may eliminate staging and rereads because that is an
  intrinsic benefit of the method.
- Staged methods must include their required pack/unpack work.
- This compares complete overlap patterns rather than raw transport.

Mixing these families would either hide direct publication's real advantage or
give SDMA a free prepacked buffer that applications do not have.

Report two different estimands:

1. **Complete-path crossover:** best legal HipKittens/IRIS CU path versus
   host-enqueued copy path versus MORI path. API, initiator, and operation may
   differ, so this answers which complete implementation wins.
2. **Hardware-engine main effect:** the same pinned MORI nonblocking-put API,
   process model, memory type, grouping, and completion semantics with
   transport forced to P2P/CU versus SDMA. If that same-API pair is unavailable,
   do not report an SDMA-versus-CU engine main effect.

### 5.2 Measure the complete overlapped DAG

For every configuration record:

- `Tp`, `Tm`, and `Tcons`: isolated producer, communication/protocol, and
  dependent-consumer durations for exactly the work in the joint DAG.
- `Tcompute_full`: isolated producer+consumer compute using all otherwise
  available CUs.
- `Tcompute_reserved`: the same compute with the overlap arm's enforced
  resource reservation.
- `Tserial`: the complete method, including its own materialization, packing,
  and protocol, executed with overlap disabled.
- `Tideal`: a dependency-derived ideal critical path for that complete method.
- `Tjoint`: synchronized global completion window for the overlapped method.
- `Tp_joint`, `Tm_joint`, and `Tcons_joint`: instrumented component durations
  inside the joint arm.
- Compute slowdown: instrumented joint compute duration divided by
  `Tcompute_reserved`.
- Communication slowdown: `Tm_joint / Tm`.
- Residual makespan over isolated compute:
  `max(0, Tjoint - Tcompute_reserved)`.
- Complete-method overlap efficiency:

  \[
  E = \frac{Tserial - Tjoint}{Tserial - Tideal}
  \]

  Report `E` unclipped and explain values outside `[0,1]`. This definition
  remains valid when direct publication eliminates work that exists in a
  staged method; `Tp+Tm+Tcons` is not assumed to be additive.

- Time to first and 50% consumable.
- Rank-max minus rank-median completion tail.
- Polls, credit stalls, outstanding-depth distribution, and SDMA queue depth.
- Actual bytes by traffic class.
- GB/s and TFLOPS as explanatory metrics, never the primary verdict.

Timing protocol:

- Compute per-iteration global windows, never a maximum of per-rank quantiles.
- Use a synchronized host/coordinator bracket from epoch release until all
  ranks complete as the primary global makespan.
- Also report `max_r(end_r-start_r)` from local rank clocks and the measured
  launch/start skew.
- Do not subtract timestamps from unsynchronized clocks on different GPUs.

### 5.3 Preserve method fairness

- Same deterministic routing and payload values.
- Same total semantic producer and consumer work.
- Same process model and synchronization boundaries.
- Compare methods only within one allocation, registration, and cacheability
  stratum. If no common memory type supports every engine, run a crossed
  `engine × memory type` control and do not attribute the interaction solely
  to engine.
- Same total data volume unless record count is the selected axis.
- A dedicated-CTA arm is always paired with a reserved-but-idle control.
- A two-kernel arm uses the same compute/communication CU partition as the
  in-kernel role arm.
- Enforce identical CU masks/partitions where the runtime supports them.
  Otherwise measure per-CU/per-XCD residency and reject rounds whose achieved
  partitions differ; CTA count alone is not a partition match.
- SDMA includes queue creation/amortization policy, descriptor submission,
  completion, pack, and unpack costs. Record host submission time, launch
  count, descriptor grouping, queue depth, and completion cost separately.
- For push and pull, record the initiating rank, local and remote reads/writes,
  payload/control bytes, and readiness publisher. Compare source-owned push
  with destination-owned pull using symmetric bidirectional rounds.
- Define warm-cache and cold-cache protocols separately. Cold runs use
  disjoint footprints larger than relevant caches or a validated eviction
  pass; memory cacheability and buffer addresses are recorded.
- Methods are rotated within paired rounds.
- Separate translation units or compile-time specializations prevent dead
  methods from perturbing the winning method's code generation.
- Resource counts, spill counts, whole-kernel scratch operations, and critical
  ISA structure are recorded for every binary.
- Pin each submitting host thread to the GPU-local NUMA node. Record the ROCm,
  driver, firmware, binary hashes, clocks, power, and temperature; reject
  rounds with foreign GPU traffic or pre-registered clock/thermal excursions.

### 5.4 Correctness gates

Before timing every arm:

1. Epoch-dependent deterministic source values.
2. Poison every inbox and output before every epoch.
3. Full-output identity or a collision-resistant digest plus sampled
   elementwise checks. Use integer accumulation for exact transport tests.
   Floating remote accumulation is a separate numerical arm with a declared
   legal ordering and tolerance.
4. Verify zero surviving poison.
5. Monotone generation checks across slot reuse.
6. All-rank error reduction.
7. 600-epoch soak.
8. Set point-specific timeouts from a conservative bandwidth floor. Classify a
   timeout as a deadlock, watchdog limit, or censored slow result; never
   clear-and-retry it as noise.
9. Mandatory negative controls:
   - omit one publication;
   - publish before payload completion;
   - redirect one destination;
   - omit one credit return;
   - reuse a buffer one epoch early;
   - for SDMA, place payload and signal on unordered queues.

---

## 6. Benchmark substrate

### 6.1 Reuse, but do not blindly inherit, exp_22

`exp_22_fig7_saturation/e22_saturation.hip` already provides:

- Standalone MI350X HIP kernels independent of the megakernel.
- MFMA, HBM, and XGMI bodies.
- Reserved, isolated, and concurrent role arms.
- Peer pointer translation and packetized push.
- Per-CTA device timestamps and HIP-event cross-checks.
- Checksums, rotations, and resumable JSONL output.
- Resource and ISA analysis scripts.

This is the right kernel-body and data-schema donor.

It is **not** yet a fair final SDMA comparison because it controls all GPUs
from one process while MORI/IRIS normally use one process per rank and a
symmetric heap. The neutral suite should therefore:

1. Reuse exp_22's device bodies and parsers.
2. Move all final comparison arms to one common rank-per-GPU launcher.
3. Make each rank submit its own source-owned operations.
4. Use one common memory stratum where possible and an explicit crossed
   memory-type control where it is not.
5. Record registration setup separately from steady-state iteration timing.
6. Emit one shared JSON schema.
7. Keep a one-process exp_22-compatible mode only as a fast diagnostic.

### 6.2 HipKittens implementation

Useful CDNA4 primitives:

- `peer.cuh`: peer address translation.
- `packet.cuh`: peer packet push, multi-region push, and peer pull.
- `accumulate_peer_bf162`: packed-bf16 remote accumulation.
- `sync.cuh`: directional release/acquire and memory scope.
- `completion.cuh`: epoch publication and bounded observation.
- `counter.cuh`: counted arrival.
- `lifetime.cuh`: reusable-slot credits.
- `roles.cuh`: role partition and finish-order mechanisms.

The benchmark must add an explicit outstanding-operation-depth parameter; the
packet helpers do not currently expose the throttle that was decisive in the
MoE kernel.

### 6.3 IRIS implementation

IRIS already exposes the required semantic operations:

- Remote `load/get`.
- Remote `store/put`.
- Remote atomics.
- Symmetric rank-address translation.
- Workgroup-specialized GEMM+communication.
- Separate producer/consumer kernel examples.
- Bulk-synchronous baselines.
- Push and pull AllGather/GEMM examples.
- One- and two-shot reduction patterns.

Implement the same generated work list and expected checksum in IRIS. Treat
HipKittens and IRIS as two backend replications:

- Compare methods within each backend.
- Compare crossover trends across backends.
- Do not attribute an absolute backend difference solely to the overlap method;
  compiler and kernel-quality differences remain.

### 6.4 SDMA implementation

Test two distinct SDMA patterns:

1. **Host-enqueued copy engine:** `hipMemcpyPeerAsync` or the appropriate
   batched copy API on a dedicated stream. Label it SDMA only after a profiler
   or executor-specific probe verifies that SDMA carried the transfer.
2. **Shader-initiated MORI SDMA:** `ShmemPutMemNbi*` followed by the
   destination-specific SDMA quiet and an explicitly ordered signal.

They have different enqueue latency, stream interaction, and persistence
properties and must not be merged into one “SDMA” label.

For each SDMA readiness policy, pre-register descriptor grouping and time from
the first required enqueue/launch through ordered destination visibility.
Small-message tests must not silently compare thousands of host calls against
one shader loop. Pin the submitting host thread and rotate submission order.

MORI's SDMA path should live in a sibling backend if it cannot share one
executable cleanly, but it must use:

- The same process topology.
- The same registered buffers.
- The same payload and consumer checksum.
- The same JSON schema.
- The same paired campaign order.

Before any result, pin the exact MORI commit, runtime transport configuration,
and API used. Require an executor trace plus a payload-before-signal negative
control before labeling the arm shader-initiated SDMA; MORI APIs may dispatch
through P2P, RDMA, or SDMA depending on build and runtime support.
Executor verification is required for every measured size/configuration, not
one representative run. A point whose executor cannot be established is
excluded from the hardware-engine map.

---

## 7. Revised ablation plan

### Stage 0 — semantic parity and calibration

Purpose: prove all backends execute the same DAG before broad sweeps.

Initial point:

- 2 GPUs.
- 64 KiB records.
- 64 MiB total payload/rank.
- Fanout 1.
- Uniform routing.
- Compute repeats selected for `Tc/Tm ≈ 1` against the pre-registered CU-push
  reference; every arm reports its realized ratio without retuning.
- One epoch and ping-pong epochs.

Required arms:

- Bulk synchronous.
- CU push.
- CU pull.
- Reserved service-CTA push.
- Homogeneous producer-carried staged push.
- Direct producer publication.
- Host-enqueued copy-engine transfer, labeled SDMA only after tracing.
- Shader-initiated MORI SDMA.
- Separate persistent producer/consumer kernels.

Exit gate:

- All correctness and negative controls pass.
- Device and host timers agree within a pre-registered tolerance.
- HipKittens and IRIS use identical generated work lists, payload/control byte
  counts, publication counts, and consumer eligibility rules, and produce
  identical expected outputs/digests.
- Isolated compute and communication rates are stable across paired rounds.

### Stage 1 — engine and direction crossover

This stage deliberately contains no synthetic GEMM overlap. It answers the raw
transport question first.

Arms:

- CU push.
- CU pull.
- Host-enqueued copy-engine transfer, labeled SDMA only after tracing.
- The same pinned MORI nonblocking-put path forced to P2P/CU transport.
- The same pinned MORI nonblocking-put path forced to SDMA transport.

Grid:

- 2 GPUs, one fixed pair.
- Record size:
  `{256 B, 1 KiB, 4 KiB, 14,336 B, 64 KiB, 256 KiB, 1 MiB, 4 MiB, 16 MiB, 64 MiB}`.
- Fixed-volume sweep: 64 MiB/rank.
- Fixed-record-count sweep: 4,096 records/rank only while
  `2 × count × size + workspace ≤ 50% HBM`. For larger records, reduce count
  geometrically and record that this is a memory-capped sweep.
- Both transfer directions.

Outputs:

- Amortized time per record. A separately validated sparse-sampling arm may
  collect true record-latency distributions.
- Aggregate and per-link bandwidth.
- Queue/enqueue overhead.
- Completion latency after the final byte.
- Source and destination CU activity.

Decision:

- Identify one size below, near, and above every engine crossover.
- Do not claim an overlap winner from this isolated stage.
- Label the heterogeneous API comparison a complete-path crossover. Reserve
  “engine main effect” for the same-API forced-P2P versus forced-SDMA pair.

Why this is not duplicate work:

- TransferBench has large-message engine data.
- MORI has collective and serving endpoints.
- Neither supplies a matched small-to-large CU-push/CU-pull/host-SDMA/
  shader-SDMA curve with identical semantics on the target node.

### Stage 2 — communication carrier under concurrent compute

Use only the crossover-adjacent sizes from Stage 1.

Synthetic compute:

- LDS-fed MFMA body as an execution-capacity control.
- A resource-matched GEMM proxy with representative HBM/L2 traffic, register
  tuple, occupancy, and epilogue live ranges.
- Freeze identical compute repeat counts across methods. Select them against
  one pre-registered reference CU-push `Tm` to target
  `Tc/Tm ≈ {0.25,1,4}`, then report every method's realized ratio without
  retuning its compute work.
- Same arithmetic and output checksum in every arm.
- Transported values feed a checked dependent consumer chain; independent
  dummy MFMA work does not count as hidden consumer work.

Arms:

1. Compute only.
2. Reserved-but-idle CTA control.
3. Reserved service CTAs performing staged push.
4. Homogeneous compute CTAs performing staged push.
5. Direct producer-epilogue push.
6. Direct producer-epilogue accumulation.
7. Two persistent kernels with the same CU partition.
8. Verified host SDMA.
9. Shader-initiated MORI SDMA.

Initial topology:

- 2 GPUs, fanout 1.
- Then 8 GPUs, round-robin fanout 7.

Primary verdict:

- Synchronized global `Tjoint`.
- Compute slowdown relative to the resource-matched idle control.
- Method-by-size and method-by-compute-ratio interaction.

This stage determines whether SDMA's lower CU cost compensates for setup and
staging, and when producer-carried communication avoids enough materialization
to win.

The service-CTA, homogeneous-carrier, and separate-kernel patterns replicate
or extend patterns already measured by IRIS, COMET, and the tile-signaling
paper. Publishable novelty must come from matched CDNA4 resource controls or
interactions with direction, SDMA, and flow control.

### Stage 3 — multiple streams and TBO

Run this immediately after the transport and carrier crossovers. SDMA's main
application value may be freeing CUs for independent work, which a
single-stream synthetic DAG cannot expose.

Test three distinct overlap scopes:

1. **Same-batch two-stream pipeline**
   - The compute stream produces destination-complete tile/segment groups.
   - A communication stream consumes ready groups while later groups continue
     computing.
   - Compare existing order with destination/owner-major production.

2. **Cross-microbatch TBO**
   - Communication for microbatch `i+1` overlaps compute for microbatch `i`.
   - Combine for microbatch `i` may overlap dispatch/compute for `i+1`.
   - Sweep unsplit, TBO-2, and TBO-4.

3. **Independent-branch overlap**
   - Routed-expert dispatch/combine overlaps shared-expert GEMM.
   - This is the closest controlled analogue to the MORI SDMA serving result.

Named complete candidate — **three-lane hybrid**:

```text
communication/SDMA: combine(batch i-1) | dispatch(batch i+1)
compute stream:     routed megakernel(batch i)
alternate stream:   shared expert(batch i)
```

All comparison arms execute identical routed-expert, shared-expert, dispatch,
and combine work:

1. Serialized one-stream baseline.
2. Routed megakernel `i` overlapping shared expert `i`.
3. Routed megakernel `i` overlapping SDMA dispatch `i+1` and combine `i-1`.
4. Full three-lane hybrid.
5. Full schedule with CU-issued P2P communication instead of SDMA.

The least-invasive first version keeps the complete current megakernel for
batch `i` and uses external communication only for adjacent batches. It must
not duplicate batch `i` dispatch/combine. Later versions may add a
pre-dispatched-input mode or an externally-combined-output mode to move those
phases across the megakernel boundary.

Transport contestants:

- Current depth-4 CU-issued direct path.
- Same-API MORI P2P/CU transport.
- Same-API MORI SDMA transport.
- Tuned communication-kernel/RCCL reference where semantically applicable.

Required controls:

- Same total tokens, routes, payload bytes, arithmetic, and output.
- Include pack/unpack, queue submission, events, and completion.
- Record stream priority, `GPU_MAX_HW_QUEUES`, graph-capture mode, launch count,
  and achieved CU residency.
- Use double/ring buffering with generation-safe epochs; test wrap/reuse
  negative controls.
- Keep at least three live generations (`i-1`, `i`, `i+1`) in distinct
  buffers; depth 4 is preferred for unambiguous reuse and backpressure.
- Measure GEMM degradation caused by smaller microbatch M dimensions.
- Compare graph-captured and eager schedules without mixing them in one arm.

Report separately:

- Single-request latency.
- Fill/drain cost.
- Steady-state throughput.
- Compute and communication slowdown.
- TBO scheduling overhead.
- Microbatch GEMM-efficiency loss.

TBO wins a regime only if it improves the selected objective. A throughput win
does not imply a latency win, and vice versa.

### Stage 4 — readiness granularity

Use the two best carriers from Stage 2, the bulk-synchronous baseline, and any
carrier with a pre-registered readiness-specific rescue hypothesis.

Signals:

- Fine: one signal per record.
- Grouped: one signal per 16 records or 64 KiB, whichever is larger.
- Destination epoch: one signal per source/destination group.
- Global epoch.

Consumers:

- Immediate checksum/reduction per ready group.
- Whole-buffer consumer as a negative-control dependency graph.

Metrics:

- Time to first/50% consumable.
- Protocol-induced compute slowdown.
- Reduction in final makespan.
- Publication and polling instruction/byte counts.

Key hypothesis:

- Fine signaling helps only if it changes the consumer's execution time.
- If the consumer still requires the whole buffer, early flags should lose or
  be neutral, matching the MORI LL128 and David/Iris observations.

Required controls:

- Preserve signal/poll work while withholding early consumption to price
  protocol overhead.
- Preserve consumer scheduling while changing publication grouping to isolate
  the useful-shadow benefit.
- Ensure every consumer's checked output depends on the received values.

### Stage 5 — flow control under fanout and skew

Retain methods within 10% of a Stage-2 winner, plus every method with a
pre-registered rescue hypothesis for fanout or skew. The balanced-point filter
must not eliminate pull, per-destination flow control, or SDMA before the
regime where each is predicted to help.

Traffic:

- 8 GPUs.
- Fanout `{1,2,7}`.
- Hot share `{uniform,0.5,0.9}`.
- Round-robin and all-to-one incast.

Credits:

- Unbounded, where safe.
- Global total depth `K`.
- Per-destination depth `d` with matched total `K=F*d`.
- Tight global depth `K=d`.
- `d ∈ {1,4,8}`.

This separates two effects:

1. Lower total injection.
2. Fairness and head-of-line blocking across destinations.

The current MoE kernel measured only a global depth. Global and
per-destination credits are established concepts; the defensible contribution
is a matched-total-credit CDNA4 characterization and its interaction with
direction, order, and skew.

### Stage 6 — task order × readiness interaction

Orders:

- Fixed random.
- Destination-major.
- Round-robin destination.
- Consumer-major.
- Remote-critical first.

Cross with:

- Fine signals.
- Grouped signals.
- Destination epochs.

Metrics:

- Per-destination active concurrency.
- Readiness CDF.
- Link concentration.
- `Tjoint`.

This stage tests the central lesson from David's Iris kernel and local exp_29:
coarse readiness may be correct under a late-readiness order, while a
consumer-major order may create enough early work for finer signals to pay.

Fine signaling, remote-first scheduling, and their basic combination are prior
art. The target contribution is the measured AMD
order × granularity × flow-control crossover.

### Stage 7 — in-kernel roles versus separate persistent kernels

Compare:

- One kernel with a fixed CTA role split.
- One kernel with opportunistic/work-stealing roles.
- Two persistent kernels on separate streams.
- SDMA plus one compute kernel.

Match:

- Communication CU count.
- Compute CU count.
- Payload and signal policy.
- Kernel lifetime.

This answers a question the current megakernel cannot: whether separate kernels
win because communication code receives an independent register allocation and
does not inflate the compute kernel's spills or resource footprint.

This is a controlled replication/extension of IRIS producer-consumer and
workgroup-specialized patterns and the tile-signaling paper, not a new overlap
pattern by itself.

### Stage 8 — topology

Repeat only crossover-adjacent points:

- 2, 4, and 8 ranks.
- One fixed peer pair.
- Nearest and farthest measured peer classes.
- Ring.
- All-to-all.
- All-to-one incast.

Record the actual physical peer/link mapping with every result. Do not infer
topology class from rank number alone.

For rank-count sweeps, run separate normalizations that hold bytes per active
link constant and aggregate bytes/rank constant. Otherwise rank count, offered
load, fanout, and topology change together.

Multi-node RDMA is a separate final tier:

- Intra-node CU push/SDMA.
- Inter-node device RDMA.
- Hierarchical combinations.

### Stage 9 — advanced dynamic schedulers

Run only after the foundational map identifies a regime where static placement
or order leaves measurable slack.

Contestants:

- Warp-specialized communication inside a producer CTA.
- Readiness-driven global work queue with out-of-order assignment.
- FlashMoE-style scheduler/subscriber and processor roles with temporal
  multi-buffering.
- COMET-style shape-indexed profiled lookup as the static adaptive baseline.

These arms require the same complete-path, partition, and code-generation
controls as earlier stages. Until they are implemented, conclusions remain
scoped to static CTA/kernel/engine policies.

---

## 8. Pre-registered regime hypotheses

These are hypotheses to test, not conclusions to write into the paper now.

| Method | Expected favorable regime | Expected unfavorable regime | Falsifier |
|---|---|---|---|
| Fixed communication CTAs | communication requires resident progress and compute leaves CU capacity unused | one-block/CU saturated GEMM with little peer wait | never beats both reserved-idle and homogeneous carrier in any corner |
| Opportunistic CTAs | bursty readiness with idle producer capacity | service latency must be continuously bounded | work stealing delays remote progress or does not lower makespan |
| Homogeneous staged push | moderate records, cheap staging, no separate progress requirement | staging reread competes with producer memory path | direct producer wins by >3% across matched regimes |
| Direct producer push | fine output already in registers; early remote consumption | acknowledgements stall the epilogue or remote layout is fragmented | loses after its best legal credit depth |
| Direct remote accumulation | reduction is associative and collisions are limited | high atomic contention or unsupported numeric semantics | source-separated stores plus local reduction win consistently |
| Pull | destination has precise demand and avoids useless fanout | source memory is remote-latency-bound or many consumers incast | push wins every topology/skew corner by >3% |
| SDMA | large contiguous transfers plus independent CU work | small fragmented records, expensive packing, no independent compute | never lowers concurrent makespan despite lower CU activity |
| Separate persistent kernels | independent register allocations and real concurrent resources | launch/residency partition dominates | same-partition in-kernel roles always win |
| Fine readiness | early useful consumer work creates a long shadow | whole-buffer dependency or protocol-dominated small records | earlier readiness does not reduce `Tjoint` |
| Coarse destination epoch | late readiness or large protocol cost | consumer-major order exposes substantial early work | fine/grouped signals lower `Tjoint` beyond protocol cost |
| Per-destination credits | fanout, skew, or heterogeneous links cause head-of-line blocking | balanced one-peer traffic | matched-total global credit is never worse |
| TBO/microbatching | high concurrency and enough compute per microbatch | latency-sensitive decode or GEMM efficiency collapses | unsplit wins both latency and throughput |

---

## 9. Statistical and stopping rules

- Correctness or negative-control failure makes a point inadmissible.
- Use paired, rotated rounds and per-iteration global makespan.
- The campaign is the inference unit; iterations within a campaign are not
  independent samples.
- Pre-register the campaign summary, paired CI method, equivalence margin,
  minimum detectable effect, power target, and multiplicity handling for each
  figure.
- Report p50 and p95 global makespan plus paired confidence intervals.
- Use discovery rounds to locate crossovers and independent confirmation
  rounds to estimate the selected points.
- A method over 10% slower in every Stage-2 corner is not expanded unless one
  later regime has a specific rescue hypothesis.
- Every method with a pre-registered interaction/rescue regime is carried into
  at least that regime regardless of its balanced-point rank.
- Expand a knob only when:
  - paired makespan moves by at least 2%, or
  - a method-by-regime interaction exceeds 3%.
- Differences below 2% require a power calculation and at least seven paired
  campaigns.
- Bandwidth wins that do not reduce `Tjoint` are recorded as transport wins,
  not overlap wins.
- A deadlock is a method result. Watchdog and censored-slow outcomes are
  reported separately.

---

## 10. Mapping the neutral results back to workloads

The neutral suite nominates mechanisms; it does not select the final MoE
kernel. Selection happens through regime cards built from real model traces.
Each card records:

- Total tokens and tokens/rank.
- Tokens/expert distribution, hottest expert, and hottest destination.
- Top-k, local/global expert count, hidden size, intermediate size, and dtype.
- Resulting grouped-GEMM M/N/K distribution.
- Payload and protocol bytes by destination.
- Shared-expert or other independent work.
- Physical topology and active links.
- Optimization objective: single-request latency, steady-state throughput, or
  training step time.

### 10.1 Decode proxy

- Low-token records/rank: `{1,2,4,8,16}`.
- High-concurrency decode points drawn from real scheduler batches rather than
  treating decode as universally `T=1`.
- Payload: 4–16 KiB.
- Low total volume.
- Protocol and launch included.
- Required arm with an independent shared-expert compute branch.

Questions:

- Does SDMA setup amortize?
- Is coarse source/destination publication better than per-record flags?
- Does a persistent source-owned queue beat a generic collective?

### 10.2 Prefill proxy

- Records/rank: `{512,1024,2048,4096}` after correctness is established.
- 14,336 B MoE-row anchor plus grouped 64 KiB–1 MiB segments.
- Balanced and hot-destination routing.
- Fanout `{1,2,7}`.
- Compute ratios 1 and 4.
- Preserve the actual tokens/expert distribution so grouped-GEMM shape changes
  are included rather than replaced by a fixed synthetic MFMA loop.

Questions:

- Does direct producer publication remain best?
- Does owner/consumer-major order change the useful readiness window?
- Do per-destination credits beat the global depth-4 policy under skew?

### 10.3 Training proxy

- 2, 16, and 64 MiB contiguous chunks.
- Long GEMM overlap.
- AllGather, ReduceScatter, and AllReduce consumer semantics.
- Stream-K/RCCL tuned reference point.
- Forward dispatch, expert compute, and combine.
- Backward expert-input gradients, weight gradients, and expert-parallel
  communication.
- Data-parallel gradient ReduceScatter/AllReduce overlap.
- Optimizer/activation-memory pressure where it changes feasible buffering.

Questions:

- At what size does SDMA reduce total makespan?
- How much is due to raw transfer versus avoided GEMM slowdown?
- Does microbatching reduce GEMM efficiency enough to erase overlap?

Training claims require the backward graph. Large-message forward-only tests
must be labeled training-shaped transport proxies.

### 10.4 Complete-policy MoE tournament

For every regime card, compare complete implementations:

1. Current mode-12 depth-4 ratchet.
2. Current direct path with per-destination credits.
3. Destination/owner-major producer ordering.
4. Healthy coarse destination-epoch protocol.
5. Separate persistent compute and communication kernels.
6. Same-batch two-stream pipeline.
7. MORI SDMA plus compute stream.
8. TBO-2 and TBO-4.
9. Tuned MORI/RCCL production reference.

Apply only semantically valid arms. For training, the current mode-12 kernel is
a forward anchor, not a complete training policy; every reported training arm
must include its backward and data-parallel communication schedule.

Use the neutral suite to screen parameters, but do not automatically discard a
method with a pre-registered decode-, skew-, topology-, or training-specific
rescue regime.

The tournament output is a decision rule indexed by measurable workload
features. A useful result must state when to select direct CU publication,
SDMA, a service kernel, coarse/fine readiness, global/per-destination credits,
or TBO. Confirm the rule on held-out shapes and routing traces that were not
used to select it.

### 10.5 Real MoE validation

Only after the proxy map identifies candidate methods:

1. Fix current `T=1024/2048` correctness.
2. Add deterministic route distributions with stored realized histograms.
3. Re-run coarse mode 14 from a healthy, codegen-isolated baseline to settle
   coarse-readiness × throttle.
4. Forward `K0_PF6GM_DECOMP` through the harness for same-run M7
   GEMM/epilogue attribution.
5. Validate selected policies in the current prefill megakernel.
6. Add a decode-shaped MoE harness with shared-expert control.
7. Add a training harness that includes backward and DP collectives before
   making training claims.
8. Run end-to-end serving/training only for methods predicted to win.
9. Confirm the decision rule on held-out workload and topology cards.

---

## 11. Candidate paper contribution

The paper should aim to contribute:

1. An extension of IRIS's AMD placement taxonomy with direction, hardware
   engine, readiness, flow control, task order, topology, and pipeline scope.
2. A common HipKittens/IRIS synthetic DAG for controlled comparison.
3. Controlled CDNA4 crossover maps across:
   - message size;
   - compute/communication ratio;
   - fanout and skew;
   - topology;
   - readiness granularity;
   - flow-control policy.
4. An explanation of *why* each regime changes the winner.
5. Proxy validation on decode-, prefill-, and training-shaped synthetic DAGs.
6. A scheduler or policy derived from the map, rather than another fixed
   manually selected mechanism.

Real application validation is a separate contribution only if actual
decode-serving, prefill-serving, and training runs are completed. Proxy results
must not be labeled application or serving results.

A promising eventual design is a destination-aware edge scheduler that chooses:

- communication carrier;
- push or pull direction;
- CU or SDMA engine;
- outstanding depth;
- completion granularity;
- producer task order.

That design is not justified yet. The synthetic suite is what should establish
which decisions are needed and whether a static decision table is sufficient or
runtime adaptation is warranted.

Before claiming a scheduler contribution, compare:

- one static global policy;
- a COMET-style shape-indexed profiled lookup;
- an offline oracle over measured methods;
- the proposed online policy.

Report policy-selection overhead and held-out workload/topology accuracy. A
lookup table that only memorizes the measured grid is a benchmark result, not a
new scheduler.

---

## 12. Proposed paper figures

1. **Method regime map:** winning pattern over message size × compute ratio.
2. **Overlap frontier:** synchronized global makespan versus compute slowdown for every
   carrier.
3. **Direction map:** push versus pull under fanout, skew, and topology.
4. **Engine crossover:** CU transfer, verified host SDMA, and shader-initiated
   MORI SDMA under isolated and concurrent compute.
5. **Protocol crossover:** signal granularity × producer order, explicitly
   framed as prior-art reproduction plus CDNA4 regime analysis.
6. **Flow-control map:** global versus per-destination credits.
7. **Scope crossover:** in-kernel roles, separate kernels, and TBO, explicitly
   framed against IRIS, Lancet, MORI/ATOM, and the tile-signaling paper.
8. **Workload-proxy validation:** decode-, prefill-, and training-shaped
   synthetic DAGs.
9. **Synthetic MoE validation:** current production-arm comparison plus the
   policy predicted by the neutral map. This remains a single-layer synthetic
   prefill result unless a real application run is added.
10. **Policy confirmation:** predicted versus oracle method on held-out shapes,
    routing traces, and topology cards.

Every figure should report makespan or application latency first. Bandwidth and
TFLOPS should explain the result, not replace it.

---

## 13. Immediate execution order

### Experiment 1 — neutral transport plane

Fork exp_22's device bodies and schema into a new neutral benchmark. Implement:

- CU push.
- CU pull.
- Host-enqueued copy-engine transfer, labeled SDMA only after tracing.
- MORI nonblocking put forced to P2P/CU.
- The same MORI path forced to SDMA.

Run Stage 0 and Stage 1 on two GPUs.

**Deliverable:** `transport_crossover.json`.

### Experiment 2 — overlap carrier plane

Add the synthetic MFMA producer and resource-matched controls:

- Reserved service CTA.
- Homogeneous staged carrier.
- Direct producer publication.
- Separate persistent kernel.
- SDMA.

Run Stage 2 at the three crossover sizes from Experiment 1.

This is an IRIS/COMET placement-pattern replication extended with matched
CDNA4 resource controls, direction, and SDMA.

**Deliverable:** `carrier_regimes.json`.

### Experiment 3 — three-lane hybrid megakernel + SDMA/TBO

Add:

- Serialized execution of routed megakernel `i`, shared expert `i`, dispatch
  `i+1`, and combine `i-1`.
- Routed megakernel plus shared-expert branch overlap.
- Routed megakernel plus adjacent-batch SDMA dispatch/combine.
- Full three-lane hybrid.
- Full schedule with CU-issued P2P instead of SDMA.
- Unsplit, TBO-2, and TBO-4 pipeline depths.

Run Stage 3 at the transport crossover sizes and at decode-, current-prefill-,
and training-shaped corners. Report latency and throughput separately.

Start with the least-invasive form: the full current megakernel owns batch `i`;
external SDMA operates only on batches `i-1` and `i+1`.

**Deliverable:** `hybrid_pipeline.json`.

### Experiment 4 — protocol and scheduling plane

Add:

- Fine/grouped/coarse readiness.
- Global/per-destination credits.
- Destination/consumer-major order.
- Deterministic skew.

Run Stages 4–6 for the winners plus every carrier with a pre-registered
readiness-, fanout-, or skew-specific rescue hypothesis, following the Stage 4
and Stage 5 retention rules.

Fine signaling and remote-first order are prior art; this experiment targets
their CDNA4 crossover and interaction with matched-total flow control.

**Deliverable:** `protocol_regimes.json`.

### Experiment 5 — backend replication

Implement the same semantic cases in IRIS and compare crossover trends against
HipKittens.

**Deliverable:** `backend_replication.json`.

### Experiment 6 — workload proxies and MoE tournament

Build the decode, prefill, and training regime cards. Run the complete-policy
MoE tournament, then integrate the predicted winner for each regime.

**Deliverables:** `workload_regimes.json` and `moe_policy_tournament.json`.

### Experiment 7 — held-out policy confirmation

Freeze the decision rule and test unseen shapes, routing traces, and topology
cards. Compare the rule with one static policy, a shape-indexed profiled lookup,
and an offline oracle.

**Deliverable:** `policy_confirmation.json`.

Suggested directory:

```text
exp_39_overlap_regime_suite/
  plan.md
  schema.json
  result.md
  transport_crossover.json
  carrier_regimes.json
  hybrid_pipeline.json
  protocol_regimes.json
  backend_replication.json
  workload_regimes.json
  moe_policy_tournament.json
  policy_confirmation.json
```

---

## 14. Sources

### Local

- `exp_22_fig7_saturation/`
- `exp_33_attribution/`
- `exp_34_mode14/`
- `exp_35_waterfall/`
- `exp_36_sensitivity/`
- `exp_37_placement/`
- `exp_38_ratchet_restore/`
- `overnight/aug10/experiments/exp_20_interference/`
- `overnight/aug10/experiments/exp_21_direct_accumulate/`
- `overnight/aug11/exp_29_pipelined_combine/`
- `docs/distributed/PRIMITIVES.md`
- `docs/distributed/ARCHITECTURE.md`

### Internal AMD

- [MORI - SDMA Case Study](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1696994052/MORI+-+SDMA+Case+Study)
- [Low Latency EP Dispatch + Combine Optimization](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1688685557/Low+Latency+EP+Dispatch+Combine+Optimization)
- [MORI dispatch-combine kernels performance analysis](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1734250005/MORI+dispatch-combine+kernels+performance+analysis)
- [GEMM/RCCL overlap optimization on MI355](https://amd.atlassian.net/wiki/spaces/DCGPUAIST/pages/1621983644/GEMM+RCCL+overlap+optimization+on+MI355)
- [Gemm-RCCL overlap micro-benchmarking](https://amd.atlassian.net/wiki/spaces/FPS/pages/1609469800/Gemm-RCCL+overlap+micro-benchmarking)
- [TransferBench](https://amd.atlassian.net/wiki/spaces/DCGPUCEVAL/pages/804280169/TransferBench)
- [GPU Interconnect Bandwidth](https://amd.atlassian.net/wiki/spaces/RPLBAS/pages/996537967/GPU+Interconnect+Bandwidth)
- [Cross-GPU GEMM on MI350X (CPX)](https://amd.atlassian.net/wiki/spaces/GPUCPT/pages/1703348879/Cross-GPU+GEMM+on+MI350X+CPX+5+Penalty+at+Remote+OAM+NPS+Mode+Negligible)
- [SGLang WideEP EP16 EPLB analysis](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1734808757/SGLang+WideEP+EP16+EPLB+Perf+analysis+-+2P2D+-+MI300+CX7)

### Public

- [ROCm/mori](https://github.com/ROCm/mori)
- [ROCm/iris](https://github.com/ROCm/iris)
- [ROCm/TransferBench](https://github.com/ROCm/TransferBench)
- [FlashMoE](https://proceedings.neurips.cc/paper_files/paper/2025/file/918d938bd209e5b56072777366f8a211-Paper-Conference.pdf)
- [COMET](https://arxiv.org/abs/2502.19811)
- [Lancet](https://arxiv.org/abs/2404.19429)
- [FasterMoE](https://doi.org/10.1145/3503221.3508418)
- [Tutel](https://arxiv.org/abs/2206.03382)
- [PipeMoE](https://doi.org/10.1109/INFOCOM53939.2023.10228874)
- [Fine-grained MoE tile signaling and scheduling](https://arxiv.org/abs/2607.19539)
- [IRIS paper](https://arxiv.org/abs/2511.12500)
- [ATOM TBO design](https://github.com/ROCm/ATOM/blob/main/recipes/TBO.md)
- [David's pinned Iris two-shot GEMM-AllReduce](https://github.com/ROCm/iris/blob/e4ab33623d5dae4e06421a6311bf9890ce9afcc7/iris/ops/matmul_all_reduce.py#L131-L198)

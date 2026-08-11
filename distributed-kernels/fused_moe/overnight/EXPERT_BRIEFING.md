# CTA specialization on MI350X — where we are, what's blocking us, what we need

Fused MoE layer, 8× MI350X (gfx950), single node, persistent megakernel, 256
CTAs. All numbers are 5-rotation median rank-max p50 from a fixed synthetic
prefill benchmark, or single-epoch in-kernel `s_memrealtime` stamps where noted.

## The story in three sentences

We took a homogeneous persistent megakernel (every CTA runs the whole pipeline)
and carved out `C` of the 256 CTAs as a **communication service pool** that
pushes the MoE combine payload to peer GPUs while the other CTAs issue MFMA for
the second expert GEMM. It works — the specialized kernel is now **0.889×** the
production baseline against the homogeneous kernel's **0.893×**, confirmed by
two independent campaigns. **But the overlap costs us 815 µs of interference on
the GEMM it runs under, and that single term is now the entire remaining gap.**

## Where the time goes

Phase profile of the homogeneous kernel, total 6,989 µs:

| phase | µs | share | scales with CTA count? |
|---|---:|---:|---|
| dispatch (all-to-all of tokens) | 1,193 | 17% | — |
| plan (sort/scatter) | 415 | 6% | — |
| **M6 — expert GEMM 1** | **2,539** | **36%** | **NO — +0.5% when 24% of CTAs are removed** |
| M7 — expert GEMM 2 | 1,586 | 23% | **YES — +27%, linear** |
| combine (reduce + all-to-all back) | 1,255 | 18% | roughly no |

**We only overlap during M7.** The service pool is reserved right before M7 and
pushes combine payload while M7's MFMA runs. Compute CTAs fall through and join
the drain as they finish M7.

Same phases, specialized kernel at the winning point (C=64):

| phase | specialized | homogeneous | delta |
|---|---:|---:|---:|
| plan | 413 | 415 | −2 |
| M6 | 2,588 | 2,539 | +49 |
| **M7** | **2,835** | **1,586** | **+1,249** |
| combine | 446 | 1,255 | **−809** |

We buy 809 µs of combine and pay 1,249 µs on M7. Net is still a win only because
of where the phases sit relative to each other.

## The bottleneck, decomposed

**M7 +1,249 µs = ~434 µs capacity + ~815 µs interference.**

- **Capacity (434 µs):** M7 running on 192 CTAs instead of 254. Measured
  directly — a "reserve but do nothing" mode gives M7 = 2,020 µs.
- **Interference (815 µs):** the *same* 192 CTAs, but with the service pool
  actively moving bytes. 2,020 → 2,835 µs.

### What the interference is NOT — all of this is measured, not assumed

1. **Not MFMA issue-slot contention.** Occupancy is one block per CU (256 VGPR +
   256 AGPR, 155,428 B LDS), so a service CTA is *never* co-resident with an
   MFMA CTA. There is no SIMD to share.
2. **Not cache capacity / pollution.** We marked both the source load and the
   peer store non-temporal and **verified by disassembly that the `nt` bits
   reached the ISA** (`flat_load_dwordx4 … nt`, `flat_store_dwordx4 … nt`).
   Zero effect. So a replacement-policy change does nothing.
3. **Not mostly atomics.** Removing 44% of the protocol's atomics (~698k of
   1.58M per rank per epoch) bought 306 µs. At that rate, removing *every*
   remaining atomic is worth ≤235 µs — under 30% of the 815.
4. **Not peak-bandwidth saturation.** M7 reads ~470 MB of weights in 1,586 µs =
   **~296 GB/s against 8 TB/s of HBM**. The pool moves ~312 MB over xGMI in
   ~2.8 ms = **~111 GB/s against ~537 GB/s of aggregate egress**. Neither is
   close to a roofline.
5. **Partly, but only partly, per-XCD L2.** Confining the pool to dedicated XCDs
   (whole dies, so the compute dies' L2s stay clean) cut the interference
   **36%** — but starved the pool 54%, so it was a net loss. That 36% is the one
   piece we have positively localized.

**So our working hypothesis is request-level / queueing contention in the shared
memory path beyond the XCD — and we have no hardware-counter evidence for it.
That is our single biggest blind spot.**

### The traffic, for context

Per rank per epoch the payload moves **~936 MB**: M7's epilogue accumulates into
a local `part` buffer (~312 MB of writes), the service pool reads `part`
(~312 MB), the service pool writes peer `slots` (~312 MB).

## Constraints we've hit that shape any solution

- **Registers are static and per-kernel.** A service CTA is allocated the same
  256 ArchVGPR + 256 AGPR as an MFMA CTA, never issues MFMA, and cannot get one
  extra register. We tried giving the copy 4-deep memory-level parallelism; the
  64 B/lane staging array **spilled to scratch** (60 → 128 B/lane) and the gain
  evaporated. The copy loop is consequently **MLP = 1**: one
  `flat_load_dwordx4`, an unconditional `s_waitcnt vmcnt(0)`, one
  `flat_store_dwordx4`.
- **Coalescing atomics is catastrophic.** We transposed the arrival counters so
  32 lanes hit 2 cache lines instead of 32 — **≥10× slower**. RMWs serialize at
  the line; scattering them is a feature.
- **The dispatch has no wait to hide.** The all-to-all's chunk poll reports
  **max-spins-to-success = 0** out of a 2,000,000 limit. All 8 ranks run
  identical work and enter the exchange within microseconds, so it is already
  latency-hidden by symmetry. Overlapping it would buy nothing.
- **W2 is ~470 MB against a 256 MB LLC**, so M7 streams its weights every epoch
  and cannot be made resident.

## Questions for the experts, ranked by what they'd unlock

### A. The interference mechanism — our biggest blind spot

1. **When 64 CUs stream 16 B/lane global loads plus cross-GPU peer stores while
   192 CUs run MFMA with streaming weight loads, what actually serializes?**
   L2 request queues, the XCD↔IOD port, Infinity Fabric request slots, or HBM
   controller queueing? We've excluded capacity, issue slots, and peak
   bandwidth, and we've localized 36% to the per-XCD L2. What is the other 64%?
2. **Which gfx950 PMC counters would show it, and how do we collect them for a
   sub-phase of a single persistent kernel?** We need L2 request-queue
   occupancy, per-buffer LLC hit/miss, xGMI link utilization, and
   memory-controller stall — but the kernel is one long-running launch, so
   whole-kernel counters average the phases together. Is there a per-CU or
   time-sliced collection mode that works here?
3. **Is there any memory QoS on CDNA4?** Can we deprioritize or rate-limit a
   subset of CUs' memory traffic so the comm pool yields to the GEMM?
4. **Would spreading the pool's requests in *time* rather than space help?** The
   pool currently drains as fast as it can. Is a paced/throttled comm engine a
   known technique here?

### B. Reducing the traffic (the 3× prize)

5. **Is remote `global_atomic_pk_add_bf16` over xGMI to a peer's symmetric-heap
   VA supported and performant on gfx950?** If the M7 epilogue accumulated
   directly into the *owner's* buffer, the local write, the pool's read, and the
   pool's peer write all collapse: **936 MB → 312 MB**. This is the single
   largest lever we've identified.
6. **If yes — what orders it?** A row's contributions come from several CTAs;
   the owner must know when all have landed. `s_waitcnt vmcnt` is per-wavefront,
   so a producer's drain doesn't cover other producers. What is the idiomatic
   CDNA4 pattern for "remote accumulate, then signal, with a consumer that can
   trust the payload"?
7. **Is in-kernel SDMA a genuinely separate path** that would move these bytes
   without contending for the CU memory pipelines? At what transfer size does it
   beat vector stores on Infinity Fabric, and can it be driven from inside a
   persistent kernel?

### C. The register and LDS wall

8. **Can a role-split kernel get differentiated register allocation?** Our comm
   CTAs are charged 256 VGPR + 256 AGPR they never use. Is there any mechanism —
   or is the honest answer "split into two kernels"?
9. **Is aliasing the dead MFMA LDS region for comm staging safe and idiomatic?**
   We have ~155 KB of LDS allocated for the GEMM that is dead during the comm
   phase. Reusing it via a union is what gives us memory-level parallelism
   without registers. Any hazards we should know about?
10. **Does `global_load_lds` (direct global→LDS) work on gfx950,** and does it
    let a wave keep many loads in flight without consuming VGPRs?

### D. The GEMMs — 59% of the kernel, and we've never touched them

11. **M6 is 2,539 µs (36% of the kernel) and is NOT CTA-bound** — removing 24%
    of the CTAs costs it 0.5%. **What is it bound by, and how would you find
    out?** This is the largest single phase and we genuinely don't know its
    limiter. If it has idle capacity, it's the natural place to put overlapped
    work — but we don't know what it would contend with.
12. **M7 streams 470 MB of weights against a 256 MB LLC.** Is there a
    weight-staging or expert-ordering schedule that would make it less sensitive
    to co-resident traffic?

### E. Structure

13. **Given the dispatch has zero wait to hide and the two GEMMs are
    back-to-back, is CTA specialization the right tool for this layer at all** —
    versus, say, software-pipelining the two GEMMs against each other?
14. **We only overlap during M7. Should we be overlapping during M6 instead**
    (it has the idle CTA capacity), and if so, what work is even available there
    given the data dependencies?

## What we'd most like to leave the meeting with

A concrete answer to **A1/A2** (what the shared resource is and how to measure
it) and a yes/no on **B5** (remote bf16 atomics over xGMI). Those two determine
whether the remaining 815 µs is addressable at all, and everything else is
downstream of them.

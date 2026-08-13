# Tile-level overlap abstractions — what the fused-MoE campaign proved, and what ships

- **Status:** design synthesis from measured kernels. Every claim below cites a
  same-session, gate-clean campaign number or a corpus experiment.
- **Evidence base:** the mode-12 ratchet (6,483.8 µs = 0.8407× production,
  in-session control), M15 (6,292.4 = 0.8165×), M15@C=24 (5,848.5 = 0.7589×),
  mode 16 (6,559.6 = 0.8513×, falsified), plus the aug10–aug12 experiment
  corpus (exp_20/22/24/25/29/33/34/35/36/37/38) and the amd-master k0/k1
  campaigns. Three of these kernels share one chassis with single-knob deltas,
  which is what makes the pattern extraction defensible rather than anecdotal.
- **Audience:** the `include/cdna4/ops/group/distributed` primitive layer, the
  GEMM-RS/GEMM-AG kernels, and the internship deliverable ("tile-level
  collectives / communication abstractions", per the mentor's framing).

---

## 1. The headline law: CTA specialization is three different things

The corpus looked like it proved "never dedicate CTAs to communication"
(exp_22/37: monotone C-tax, 8–9 µs per reserved CTA even when provably idle;
mode-2 dedicated pool +340 µs). M15 then measured C=16 beating C=8 by 93 µs
and C=24 beating C=16 by another 444 µs — with the same reservation mechanism.
Both results are real; the resolution is that "specialization" conflates three
jobs with opposite economics:

| pool job | measured verdict | why |
|---|---|---|
| **Carrying** payload (service push, mode 2) | +332…+340 µs — never wins | payload rides ~free on producer stores (exp_22: 0.9951 concurrent/isolated); a carrier pool buys nothing and pays the capacity tax |
| **Idling** (reserve-but-wait) | +8–9 µs/CTA, monotone | pure capacity loss, contention excluded (exp_34's C-sweep with a provably jobless pool) |
| **Consuming** certified-ready output in the producer's shadow (M15's front-half combine sweeps) | −93 µs (C 8→16), −444 µs (C 16→24) | the consumer phase is latency-bound (1.57 TB/s = 20% HBM, 6.1 GB/s/CTA — exp_29); moving it into the producer's tail converts idle-machine time into finished work |

**The law:** dedicate CTAs to *consuming*, never to *carrying*, and only when a
**certified consumable prefix** exists for them to consume. The enabling
mechanism is therefore not the role split itself — `roles.cuh` already ships
that — it is the *certification* that creates early consumable work (§3) and
the *order* that makes the certified prefix large (§4). A pool without those is
the measured +8–9 µs/CTA idle tax; a pool with them is the biggest single win
this program has produced.

## 2. The occupancy-1 selection rule (what overlap can and cannot buy)

Every kernel in this family runs 1 block/CU, 1 wave/SIMD (256-VGPR GEMM
bodies). At that occupancy there are **no spare waves to fill stalls**: a CU
executing the 84%-stalled M6 K-loop cannot co-run anything else, and
`vmcnt` is one in-order counter per wave, so a wave cannot overlap its own
remote traffic with its own loads (exp_25's ISA kill; exp_38's involuntary
re-throttle). Consequences, all measured:

1. **Phase partitioning is work-conserving.** Moving full-grid work (dispatch,
   plan) "under" a compute phase via CTA partition just re-divides the same
   CTA·µs (exp_25's interleave: first-order tie; the static split: −173 µs
   *loss*). Mode 16 deferred the combine across the launch boundary and lost
   +75.8 µs: the wait it escaped was mostly not on the critical path, and the
   parity state it added was.
2. **Overlap pays only through four doors:** (a) moving a *latency-bound*
   consumer into a compute shadow (M15's combine: −144 µs of residue); (b)
   *deleting* protocol work (the ~926k-atomic per-row protocol → 2×8 slab
   words); (c) *op-class swaps* on the fabric (RMW → posted store: the m15b
   arm; the 0.127 vs 0.571 µs/op issue ladder); (d) bounding *injection* so
   congestion never forms (depth 4: −615 µs, the single largest knob).
3. **Anything placed in the hot wave's counter space is a liability.** The
   throttle is a negotiated mechanism; every new code region near the epilogue
   must pass the issue-run gate (282 atomics / ~12 runs / mean ~23.5).

This rule is the *selection function* for the whole design space: given a
proposed overlap, ask which door it goes through. If the answer is "it fills
stalls with other work," it is work-conserving at occupancy 1 and will
measure ≈ 0.

## 3. The five knobs of a tile-level collective

Every kernel in the evidence base — and, we claim, every producer→consumer
multi-GPU dataflow — is a setting of five orthogonal knobs. This is the
mid-level abstraction the mentor's notes ask for ("the sweet spot between raw
load/store and MoE-specific"):

| knob | values seen | measured deltas |
|---|---|---|
| **K1 carrier** — who moves the bytes | producer epilogue (mode 12) / staged pool push (mode 2) / staged wide-store push (m15b) / engine (SDMA — losing below 64 MiB/pair on this fabric) | epilogue-carried + bound: −615; pool-carried: +340 |
| **K2 producer order** — which output prefix completes first | tile-major (donor) / nc-major (M15) / owner-major-staggered (= ring; §6) / RR-source-interleaved (m17) | order alone: ≈0; order × certification: −144 combine residue |
| **K3 certification granularity** — when consumers may start | per-row flags (~926k ops) / slab epoch words (2×8) / phase barrier (S=1) | per-row protocol: the −573 µs interference class; slab words: mode-14 economics *without* mode-14's loss |
| **K4 flow control** — in-flight bound on fabric ops | none / global vmcnt depth {4,8,16,32} / per-destination (via order: m17) | unbounded: +211 vs no payload at all; depth 4: −615; depth is the first-class knob |
| **K5 consumer placement** — who consumes, when | after-phase full grid / pool-in-shadow with quota (M15) / next launch (mode 16) | pool-in-shadow: −93/−444; next-launch: +75.8 (falsified) |

Classical collectives are *points in this space* (§6). Our MoE kernels are
other points. The GEMM-RS kernel is another. That is the generalization claim,
and it is why these five knobs — not any one kernel — are the shippable thing.

## 4. Proposed primitives (extending `include/cdna4/ops/group/distributed`)

The existing layer (peer/packet/sync/completion/counter/lifetime/roles) already
covers translation, packets, release/acquire, epoch publication, counted
arrival, slot credits, and finish-order role partition. The campaign's measured
gaps, in the layer's own style:

1. **`credit.cuh` — bounded-injection stores/accumulates (K4).** The decisive
   −615 µs knob exists today only as a hand-rolled `s_waitcnt vmcnt(N)` in one
   epilogue, and exp_38 proved it can be silently renegotiated by register
   pressure anywhere in the kernel. Ship
   `throttled_store_packet16<Depth>` / `throttled_accumulate_bf162<Depth>`
   (compile-time depth, the four-instantiation + SGPR-mask-plan idiom from
   exp_24), and ship **the verifier with the primitive**: an ISA gate script
   asserting the issue-run distribution, because a bound that can silently die
   is not a primitive. Per-destination credits stay a documented non-goal until
   a skew harness exists; m17 shows destination *rotation via order* buys the
   spreading without protocol.
2. **`slab.cuh` — prefix certification (K3).** The (rank, slab) epoch-word
   protocol: producer side `certify_slab(slab, epoch)` = per-CTA drain + one
   grid barrier + one leader multicast of an epoch word per peer (the
   one-textual-copy discipline is load-bearing — exp_34/38's register cliff);
   consumer side `bounded_wait_slab_into(rank, slab, epoch, result)`. The slab
   cut MUST sit where producer tile shape and consumer vector shape co-align
   (column 3,584 for 448-col nc chunks × 1,024-B combine chunks; for GEMM-RS
   the owner row-block boundary is the natural cut and needs no re-chunking).
   This is also exactly the mentor's "fold the wide barrier into producer tail
   and consumer head" figure, made reusable.
3. **`roles.cuh` addition — the consuming pool (K5).** A quota-bounded
   consume-in-shadow loop: dynamic wave tickets over certified batches with a
   per-wave claim quota so the pool can never hold the next rendezvous hostage
   (M15's `flush_rows` reuse). The C-response is steep and non-monotonic
   across designs (mode 12: C≤8 best; M15: C=24 ≫ C=16 ≫ C=8), so the
   primitive ships with the *response-curve method*, not a default: sweep C
   with same-session controls, judge on the coupled producer+consumer sum.
4. **`order.cuh` — producer task-order adapters (K2).** Pure index maps, zero
   protocol: `consumer_major` (nc-major decode), `owner_major_staggered(rank)`
   (the ring inducer, §6), `source_interleaved` (m17's RR permutation for
   injection spreading). These are one-expression `N2GM_TASK_DECODE`-class
   hooks — the cheapest lever in the whole program and the one that turns K3
   from "coarse = late" into "coarse = early enough."
5. **`counter.cuh` addition — `counted_arrive_dynamic_into`.** The hottest
   counter in the MoE kernel (524k RMWs/rank/epoch) open-codes a runtime-target
   arrival because the compile-time-target primitive can't express it (exp_29's
   gap list). Needed by any dataflow whose fan-in varies per key.
6. **Verification as part of the layer, not the harness.** Poison + selftest
   (proven to fire), negative controls per signal, `.text`/issue-run gates,
   allocation-order rotation (the GEMM-RS exp_24 lesson: arm construction
   order fabricated a 5.85% swing with p≈1e-101). Every primitive above gets
   its failure-mode note the way `lifetime.cuh` documents its epoch-one path.

## 5. Arch cards (the numbers that flip decisions between GPUs)

The knobs are portable; the *settings* are not. Minimum card per arch:

| quantity | gfx950 (measured) | gfx942/MI300X (measured upstream) |
|---|---|---|
| remote store vs remote atomic | pk_add viable, depth-4 bound decisive | **atomics ~3× slower than stores (15 vs 44–46 GiB/s)** — epilogue remote-*accumulate* transport (mode 12) is likely wrong here; source-separated stores + local reduce is right |
| per-link / fan-out ceiling | 54.9 GB/s; 349–355 GB/s egress; knees 8 CTAs/link, ~64 fan-out | unreproduced; treat as unknown, measure first |
| SDMA | loses below ~64 MiB/pair | unknown |
| XCD scope | no chiplet sync scope; XCD-confined placement 7–23% worse | per-XCD L2 non-coherence noted upstream — producer→consumer across XCDs needs device-scope release/acquire |
| stall-fill | occupancy 1 ⇒ none (§2) | 304 CUs, same class of constraint at 512-thread MFMA CTAs |

## 6. The generalization proof: classical collectives fall out of the knobs

Set K2 = owner-major **staggered by rank** (rank r produces owner r+1's
row-slab first, then r+2, …), K3 = one slab word per owner block, K1 =
epilogue-carried stores with K4 depth-bounded, K5 = owner consumes each slab
as certified — and the schedule that emerges is **ring reduce-scatter**, with
every link busy from the first slab and each owner's reduce starting after
1/8th of every peer's GEMM. Two-shot all-reduce, all-gather-major schedules,
and David's Iris owner-partition GEMM-AR are other settings of the same knobs.
The abstraction doesn't *implement* collectives; it makes them (and the MoE
kernels, which are NOT classical collectives) the same five decisions — which
is precisely the "tile-level collectives" story: at tile granularity the
pipeline depth ("how many tiles before the reduce") is K3's slab size, and the
software-pipelining space the mentor calls unexplored is the K2×K3×K5 cube we
have now measured one diagonal of.

## 7. Application to GEMM-RS (concrete, ordered, arch-aware)

The branch's kernel already has the COMET producer/reducer CTA split (K5
exists) and per-tile consumption. What the MoE results add, in order of
expected value ÷ risk:

1. **First, believe your own exp_24:** the largest measured term vs rank-1 is
   the per-call host tax under the grading protocol, not the device schedule.
   No device-side knob below can pay that back; attack launch/persistence
   under the evaluator's process model first (persistent kernel across calls,
   or amortized launch), or the knob work will be invisible.
2. **K2+K3: owner-major-staggered order + slab words per owner block.** If the
   producer currently emits tiles in GEMM-natural order with per-tile flags,
   this converts the reducer pool's readiness from scattered tiles to early
   contiguous owner-slabs — the M15 move, and on RS the slab cut is free
   (owner boundaries). Prediction, honestly labeled: the granularity ladder
   was flat on T2 (2.94% spread) *for makespan*, but the reducer-pool
   *utilization* curve is what moved M15; judge on the coupled GEMM+reduce
   sum with the C sweep of item 3.
3. **K5: the C response curve.** Their `NUM_REDUCER_CTAS` table is our C knob.
   Run the M15-style same-session sweep {8,16,24,32,40} × quota; the MoE
   result says the optimum can sit far from where carrier-era intuition put
   it, and moves when K2/K3 change.
4. **K4: depth-bound the producer's peer stores** (a `credit.cuh` trial) and
   try m17's source/destination rotation if bursts concentrate per link.
5. **Do NOT port mode-12 remote accumulation** (arch card: gfx942 atomics 3×
   slower than stores). Their existing 8-tower store + local reduce is the
   right transport for MI300X; m15b's fold-then-wide-push is its gfx950 twin.
6. **Port the measurement discipline wholesale** — allocation-order rotation,
   same-session paired controls, `.text` gates. Their exp_24/26 withdrawal is
   the same lesson our exp_38 taught: the instrument is part of the kernel.

## 8. Internship-deliverable alignment

The mentor's asks map onto this document directly: "tile knows which GPU it's
on, decide dynamically when to send" = K1/K2 as device-side decisions over an
owner map; "the wide barrier where the lines don't cross" = K3's slab words
(GPU-level today, CU-level next, exactly his hierarchy); "how many tiles
before the reduce" = the K3×K5 pipelining cube; "a pattern you see over and
over, abstracted at the sweet spot" = the five knobs; AITER-integration
stretch = the GEMM-RS/AG kernels re-expressed on these primitives. The paper
skeleton he sketched (collective → fusion → end-to-end) becomes: §6's
rederivation (collective), the MoE/RS kernels as knob settings (fusion), and
the regime map across prefill/decode/training (end-to-end), with §2's
occupancy-1 selection rule as the explanatory theory the field currently
lacks.

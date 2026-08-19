# Why fast, why slow — the causal ledger

Every megakernel arm we have built, sorted not by speed but by **mechanism**. Each row is
grounded in a banked measurement (source: `distributed-kernels/fused_moe/overnight/aug18-ablations/UNDERSTANDING_PLAN.md`
and the results ledgers cited there). Framing that governs everything below: **the two expert
GEMMs are 88.1% of the MoE interior — the entire protocol-and-communication story is a fight
over ~12%.**

## The two anchor workloads

| anchor | workload tuple | the collective at the boundary |
|---|---|---|
| **A** | DP8/EP8 prefill, T=4,096 tok/rank, ISL 4,096, OSL 8, C=512-class | token-shard all-to-all (dispatch/combine) |
| **B** | TP8+EP prefill, C=32, ISL 4,096, vendor 16,384-token step | one 235 MB all-reduce per layer |

No claim below generalizes outside its anchor. That scoping *is* the answer to the experts'
"TPS without batch size means nothing" complaint.

## Anchor A waterfall — production 7,712 µs → M15 5,822 µs (0.7544×)

| arm | µs | Δ vs previous rung | mechanism (WHY) |
|---|---:|---:|---|
| production (AITER+MORI, unfused) | 7,712 | — | reference; launch gaps + unfused phase boundaries |
| homogeneous megakernel | 6,908.8 | −804 | **fusion alone, zero cross-rank protocol** — kills launch/phase-boundary overhead (Q1: itemization still PARTIAL) |
| + producer-carried RMW, unbounded | +211.2 | **regression** | relocation of comm without flow control — remote ops flood the fabric (Q2) |
| + injection bound, depth-4 (mode-12) | 6,483.8 | −615.0 | **bounded remote-op admission** — a credit on in-flight remote ops is worth more than where the op is issued from (Q3) |
| M15 producer/consumer, 24–28 consumers | **5,822** | −661.8 | order + slab certificates + **consuming pool**: dedicated consumers drain remote arrivals, relieving injector concurrency (−444 of it is the 16→24 consumer step: ≈354 injector-concurrency relief + ≈95 sweep) (Q5/Q6) |

## Anchor B — where the same instincts LOSE

| arm | measure | WHY it loses |
|---|---:|---|
| GEMM → RCCL phased (the "boring" arm) | **2,909 µs — the champion** | a mature collective + no co-residency tax |
| fused CDAR (ERS+MAG) | 3,606 µs | **not protocol** (fused protocol overhead measured at +15 µs, i.e. FREE) — the ~700 µs deficit is entirely a co-residency/scheduling failure; ρ-ladder: fusion hides nothing at ANY reachable compute intensity (h_wall ≤ 1.9%, negative at deployment ρ) |
| our transport vs RCCL | **β = 1.529 direct** (2,019.3 vs 1,320.9 µs; three methods agree 1.43–1.60) | RCCL's transport is simply better at 235 MB; issue-rate/parallelism suspected, not protocol (Q9 open) |
| shared-expert filler under the AR (G-L2) | **GO**: AR loses 0.00% wall while a full-GPU MFMA runs beside it; 44–53% of exposed AR absorbed free | a 235 MB AR is **not CU-bound** — there is a free co-residency slot; fill it with the shared expert instead of fusing the AR |

**The headline asymmetry: the SAME design (fuse + custom transport + overlap-in-kernel) is
+25% on anchor A and −24% on anchor B.** The workload changed the collective (a2a → 235 MB AR),
the mediating quantity changed (protocol overhead → transport β and co-residency tax), so the
schedule must change (own the transport → rent RCCL and fill under it).

## The failure museum (first-class, per the mentor's ask)

| specimen | measure | the teachable mechanism |
|---|---:|---|
| unbounded producer-carried RMW | +211.2 µs | relocation without flow control; admission must be bounded |
| dedicated comm CTAs (staged-push) vs (producer-carried) | 0.888× WON / lost monotonically | **carrier-conditional**: dedication is a function of grid residency (occupancy-1, grid-barriered CDNA) — not a universal (MoK's dedicated SMs win on Blackwell with CLC work-stealing) (Q8a) |
| mode-14 "reachability" | +726.9 µs, source untouched | a megakernel made slow **purely by the register allocator** (LAW-32) — the most teachable non-obvious specimen |
| S8/m15b staged wide-push | 9,087.1 µs | throttle annihilated by codegen: 96 scratch loads + 97 `vmcnt(0)` next to the remote atomics — register pressure card |
| mode-16 deferred combine | +75.8 µs | the wait it escaped was mostly off the critical path AND the parity state it added was on it |
| fine-grained per-row readiness | falsified | ~926k protocol ops/rank/epoch — signal the smallest early-EXECUTABLE unit, not the smallest unit (t50 = 91.7% under natural order) |
| coalesced protocol atomics | ≥10× slower | atomics resolve at the cache line — maximize line spread, the OPPOSITE of the load-coalescing rule |
| XCD-confined pools | net 7–23% worse | locality instinct inverted by the consumption pattern |
| M15 @ 32 consumers | hung 1 of 2 | the cliff past the knee; pool holds the next rendezvous hostage (Q7 open, R7 forensics queued) |
| fused family below T≈1,600 | fixed cost 1,068.2 µs vs production's 181.3 | the fusion tax INVERTS at low token counts — the sign flip the experts asked about (Q11, unitemized) |
| M23 coverage defect | mega ran ~2% of in-bucket serving steps | the integration lesson: every pre-M23 serving A/B measured a doubly-handicapped candidate |

## The one knob with a measured sign flip (Q12)

The same CTA reservation is **+332…+340 µs carrying, +8–9 µs/CTA idling, −444 µs consuming** —
one mechanism, three signs, governed by what the reserved CTAs *do with arrivals*. This is the
single best exhibit that "producer/consumer is best" is carrier- and workload-conditional, not
a universal law.

## Attribution status (honesty ledger)

Closure criterion: a delta is EXPLAINED only when itemized spans cover ≥80% of it with the
residual under the instrument's noise band. Current state: Q1/Q2/Q5/Q6/Q10/Q11/Q12 PARTIAL,
Q3/Q7/Q8/Q9 OPEN — the run list (R2–R11, ≈15–22 node-h) that closes them is in
`UNDERSTANDING_PLAN.md` §4b. We do not claim more than the instruments support.

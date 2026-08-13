# Overlap kernel design ideas for `k0pf6gm_mps_mega` — novel comm/compute overlap methodologies

- **Status:** design synthesis and build queue. Read-only research; nothing here is GPU-validated.
- **Date:** 2026-08-12
- **Anchor kernel:** `mps_mega` mode 12 (`kModeRemoteAccum`), ratchet `C=16, g=353 (throttle, depth 4), flush_rows=16` = **6,482.7 µs = 0.8408× production** at T=4096, 8× MI350X, balanced routing.
- **Evidence base:** `../aug11/OVERLAP_METHODOLOGY_STUDY.md` (the staged program), `../aug11/STATUS.md` + `LESSONS.md` (exp_20–exp_38), `../aug11/CONTEXT/{m6_m7_structure,mode12_protocol_map}.md`, exp_22 saturation curves, exp_33 attribution, exp_35 waterfall, exp_37 placement, exp_24 throttle, exp_25 split analysis, exp_29/30 combine/readiness analyses, exp_34 mode 14, exp_38 codegen restoration, and the internal AMD MORI SDMA / dispatch-combine / GEMM-RCCL notes cited by the study.
- **Discipline inherited from aug11:** every idea below names its mechanism, code sites, expected magnitude **with the arithmetic shown**, its falsifier, and its gates. Predictions follow the family's measured optimism bias (exp_27 landed 43–83% below its point estimates); read every point estimate pessimistically. Correctness + negative controls + 600-epoch soak precede any timing. New modes go in **new translation units or behind default-off `-D` flags with `.text` parity of the default arm** — the exp_38 lesson. Nothing here is run concurrently with the ablation campaigns (one GPU job at a time).

---

## 0. What the evidence forces every new design to respect

Seven measured laws, each with teeth in the kernel we have today:

| # | Law | Numbers | Source |
|---|---|---|---|
| L1 | **Bounded in-flight remote writes is the single dominant knob.** Payload relocation alone moves the wrong way; the *admission bound* is the win. | epilogue-carried unthrottled: **+210.6 µs**; + depth-4 bound: **−615.0 µs** (1.52× the entire a→c gap; disjoint by 598 µs, t=−80.2). Depth 4 < 8 by −50.9; **depth 16/32 collapse back to unthrottled** (+462/+370) | exp_35 rungs a/b/c; exp_24 |
| L2 | **Never dedicate CTAs to communication.** Communication saturates at tiny pools; compute has no knee to relieve. | single-link knee at **C=8** (56.9 of 76.8 GB/s), fabric knee at **C=32** (355 of 537.6 GB/s); MFMA **linear to 256** (R²≥0.99989). Idle-pool tax **8.1–9.3 µs/CTA**. C=4/8 best, monotone in C; C=64 mode 2 +340 µs | exp_22, exp_37, exp_34 |
| L3 | **Payload rides free; protocol pays.** Interference is atomics/fences/polls, not bytes. | payload concurrent/isolated **0.9951**; protocol-compiled 0.878, g=16 probe −42%. Deleting staged payload copy moved M7 by 5.7 µs while protocol interference remained | exp_22 H3, exp_20 |
| L4 | **Overlap ceilings are set by readiness/topology of the task graph, not by capacity.** | combine readiness `P(ready by t)=(t/S)^8`, median 91.7% of M7; M6→M7 availability ≡ consumption (2.133=2.133, **zero slack**); static role split = −173 µs by work conservation; interleave ceiling **+211 µs** (bursts already at 78.3% of 537.6 GB/s egress) | exp_29, exp_25 |
| L5 | **Readiness granularity only pays when something can start earlier.** | mode 14's protocol deletion (−573.3 µs vs same-pin mode 12) did **not** beat the working throttle (+153.6); substitutes-not-complements hypothesis open; MORI per-record flags no-gain / LL128 −44…−177% (deflag copy); the drain deletion is a +3.2 µs **null** | exp_34, MORI ablations |
| L6 | **M6/M7 are memory-stalled, not issue-bound — the stall pool is the overlap budget.** | M6 2,453.3 µs, **~84% K-loop stall** (~9,800 cyc/iter vs 1,536 MFMA cycles), one-K-step pipeline; M7 2,701.8 µs = GEMM ~1.88 ms + **~817–898 µs epilogue surcharge**; intensity 161.7 vs 158.1 FLOP/B; 17.1% / 13.7% MFMA duty | exp_33, exp_28, m6_m7_structure |
| L7 | **Codegen is a co-variable; guard every variant.** | one line of unreachable mode-14 code collapsed the M7 epilogue's injection window (+726.9 µs; issue runs 23.5→1.45 atomics; involuntary vmcnt(0) 21→117). Tuple parity is insufficient; `.text` sha identity or nothing | exp_38 |

Plus the workload facts: dispatch M0–M2 ≈ 642 µs residual with **zero peer wait at std=0** (`[MPS SPIN] 0/0` everywhere; exp_10); plan M3–M5 372.8 µs done as a target; combine 324.2 µs, **anti-correlated with M7 at r=−0.904** (their *sum* 3,026 µs is the real objective); T=1024/2048 correctness is an open defect and **blocks every other-size claim** until fixed.

Volume anchors (per rank per epoch, balanced T=4096): dispatch push ≈ **162 MiB** (4,096 tokens × ~5.29 fanout × 7,456 B; ~23 MiB per source→dest pair); M7 epilogue = **448 MiB of remote bf16 RMW** (392 MiB over xGMI; 102.8M atomics; ~58.7 MiB per (producer,owner) pair); M8 = 448 MiB slot reads (34.4% dummy-lane re-reads) + 294 MiB zeroing + 56 MiB out.

A structural fact the atomic census makes obvious: producer `p` writes **its own slab** `owner.slots[p][pos][col]`; the only same-address collisions are *intra-producer* (a receive row hit by ~1.5 of the producer's local experts on average). The bf16 atomic exists **only for that 1.5× collision class** — cross-producer contributions never share an address. This opens the door to several transports the kernel has never tried.

---

## 1. Where overlap currently exists vs is provably absent

| Boundary | Today | Room | Binding constraint |
|---|---|---|---|
| M1 dispatch push ↔ M2 unpack (same epoch) | already overlapped; zero peer wait | ~0 | exp_10 measured |
| **dispatch+plan (i) ↔ GEMMs (i)** | **absent: serialized by M5 barrier** | **~1,015 µs** | M3's scan is a global function of ALL arrivals (§5) |
| M6(i) ↔ M7(i) | per-tile ready (`a2_done`) but zero slack | ≤ +211 µs | exp_25's ceiling arithmetic |
| M7 epilogue ↔ M7 GEMM (remote RMW) | epilogue-carried, depth-4 throttled — the ratchet | burst de-align ≤ +211 µs | L4 |
| **M7(i) ↔ M8 combine (i)** | per-row readiness, readiness-capped | (t/S)^8 law → unlock is **task order** | exp_29/30 |
| **epoch i ↔ epoch i±1** | **absent: one launch per epoch, retire-wait at M0** | **the whole ~1,000 µs dispatch+plan + 324 µs combine** | §4 — the biggest untouched axis |
| SDMA engines | **unused everywhere** | dispatch packs ~23 MiB/pair, combine packs ~58.7 MiB/pair — both inside MORI's 2–64 MiB amortization band | Stage-1 crossovers pending |

The user's specific question — *fine-grained overlap during the dispatch phase with the first GEMM* — gets an honest structural answer in §5: **same-epoch dispatch×M6 overlap is blocked by the sort's global histogram dependency; the correct vehicle is cross-epoch (§4).** SDMA's best-expected-value entry points are dispatch packs (§2) and combine partial return (§3).

---

## 2. K1 — SDMA pack dispatch ("dispatch as engine work, not CU stores")

**Mechanism.** Today M1 quantizes each token row (fp8 + scales + meta, 7,456 B) and pushes it via LL128 packets straight into up to ~5.3 destinations' `a_ll` segments (scattered remote stores), then 256 `buffer_wbl2 sc1` writebacks + `chunk_ready` word stores publish. The `dest_counter[src]++` reservation already makes each (cur→dest) row sequence **contiguous** in the receiver's per-source segment. So: quantize → store into a **local per-destination pack** (identical layout to the remote segment, positions preserved) → once chunk `c` of a destination's pack is staged, hand that contiguous extent (~23 MiB per pair per epoch, split over 8 chunks ≈ 2.9 MiB sub-transfers) to **SDMA** — shader-initiated `ShmemPutMemNbi*` + destination-quiet + ordered signal (MORI AsyncLL idiom) or, as the reference arm, host-enqueued `hipMemcpyPeerAsync` on a side stream. SDMA's completion mechanism writes the same `chunk_ready[s][c]` epoch words the consumer polls today — **the transfer's own completion IS the readiness publication** (zero CU signal stores; see K4).

**Why this is inside the evidence:** L2 says CUs are precious only for MFMA-shaped work and stores ride free when compute is idle (L3) — but M1's push is *unbounded* store issue with writeback fences, and it is exactly the class that must **not** overlap freely once K4's epoch pipelining puts a GEMM next to it. SDMA removes the entire store+fence class from the CUs rather than scheduling it better. MORI's low-latency EP sweep says **control publication dominates small cases** — the pack scheme fails gracefully at small T (fall back to today's direct push when the per-pair pack is below the Stage-1-measured SDMA amortization point).

**Expected magnitude (honest).** Solo: dispatch 642 → ~450–500 µs (store half of M1 + fences removed; quantization expf/LDS stays on CUs) ⇒ **−100…−190 µs ≈ 2–3%**. Inside K4's pipelined epoch the same edit is worth more, because it shrinks the dispatch's claim on the shared stall pool. Band: ±50 around the point until Stage 1's SDMA enqueue/quiet overheads land.

**Code sites.** `k0pf6gm_device_tile_mps.hip` M1 push loop (KRN:1005–1082), the `chunk_ready` publication block (KRN:1101–1151); per-destination pack buffers are new descriptor words (symmetric). M2's poll structure is untouched. **Gates:** executor trace (label it SDMA only when traced), payload-before-signal negative control (unordered queues must fail), generation-parity pack reclamation, poison, 600 soak.

**Falsifier:** Stage-1 shows SDMA setup+quiet ≥ ~30% of the 23 MiB transfer time, or co-resident pack staging taxes M6 > 1%.

---

## 3. K2 — staged-SDMA combine transport (mode 15: "the modern mode 2")

**Mechanism.** The M7 epilogue's remote-packed-bf16 RMW stream (448 MiB, surcharge ~820–900 µs, congestion-shaped at 78.3% of egress ceiling in **aligned lockstep bursts**) is the largest single transport cost in the kernel. Replace the *transport op class*, keeping semantics:

1. Epilogue writes its tile rows into a **local per-owner stage pack** laid out exactly as the owner's `slots[cur]` slab (plain packed stores; the rare intra-producer collisions — avg 1.5 local experts per receive row — handled by *local* packed-bf16 atomics, same die, no fabric). exp_20 already priced staged carriage: deleting the copy moved M7 by only **5.7 µs**, and exp_22 co-resident payload costs 0.073% — staging is cheap.
2. When the rows of a stage group (an nc-slice set of tiles) are complete, issue **one SDMA transfer per (owner)** into `owner.slots[cur][pos-range]` — ~58.7 MiB per pair over ~8–16 pipelined chunks of 3.7–7.3 MiB, squarely in MORI's measured SDMA band, on engines that cost **zero CUs**.
3. Readiness: SDMA completion events publish per-(producer, group) epoch words — replacing mode 12's ~19k `row_ready` stores + the drain's ~926k bookkeeping atomics with hardware-generated signals (K4). Owner-side M8, consume-and-zero, and the M9 retirement are **bit-for-bit unchanged** (the stage layout makes transfer writes land exactly where RMWs land today).

**Why this can beat mode 12.** exp_20 says the surcharge is *op-class*, not byte-class: 4-byte scattered remote RMWs with per-row flattening into the coherence point. A bulk SDMA stream moves the same bytes as full-bandwidth packets, from engines, with no per-row CU involvement and no vmcnt negotiation. exp_35 says unthrottled RMW burst injection costs +210 µs *even after* depth-4 recovered 615 — i.e., today's ratchet still pays ~860 µs of epilogue tax to the RMW op class. The mode-2 ancestor's mistake was carrying packs on **CTAs** (C=64 pool, +340 µs capacity tax + interference); carrying them on **SDMA engines** keeps mode 12's "no dedicated CTAs" property (L2) while deleting mode 12's congestion class (L1/L3).

**Crossover logic (the strategy-switch point the user asked about).** Mode 12's epilogue RMW wins while `bytes are moderate AND collisions benefit from fabric-fold AND bursts stay throttled under the ceiling`. Staged-SDMA wins when per-pair packs ≥ the amortization band (~2–4 MiB, i.e. ≥ ~150–300 contributions per pair — at T=4096 there are **4,096**) and/or when RMW congestion grows super-linearly with scale (T=8192, training microbatches). At T=4096 this is a **wash-to-win** arm; at 2× scale it should win decisively; at small T it should lose to direct RMW + coarse signals. That crossover — measured, not asserted — is itself a headline table for the paper's method-selection map.

**Expected magnitude.** −200…−800 µs at T=4096 (uncertain sign at the low end; the arm's *scientific* value is the op-class measurement either way); −1 ms+ at T=8192-class shapes.

**Code sites.** `n2_phase2_gm_mps.cpp` `epilogue_write` (the `peer_tab != nullptr` branch, P2:218–248) gets a third target class; stage buffers + SDMA queue handles are new descriptor words; `moe_mps_adapter.cuh` service env gains the transfer-trigger logic at group boundaries; K0's nc-major order is the natural companion (it makes stage groups complete early and uniformly). **Gates:** the full negative-control ladder incl. SDMA-on-unordered-queue; bit-exact slot contents vs mode 12 (same final sums, modulo bf16 RMW add order); `.text` parity of the default arm (new TU or `-D` guard — L7).

**Falsifier:** measured SDMA bulk time + staging ≥ RMW surcharge at matched bytes; or staging reintroduces the mode-2 interference (watch the exp_20 protocol-class counters).

---

## 4. K3 — TBO-2 epoch-pipelined megakernel (the flagship: hide the other half of the layer)

**The observation.** The kernel launches **once per epoch**; M0 waits on peer retirement; every epoch pays dispatch (~642 µs) + plan (373 µs) + combine (324 µs) as serial time, while M6 runs at **17.1% MFMA duty with an 84% memory-stall K-loop** and M7's epilogue bursts are throttled to 78% of the fabric. The stall pool is measured: M6 alone idles ~2,060 CTA·µs/epoch of issue capacity; co-resident store-class work taxes compute at 0.073% (exp_22); plan+dispatch+combine = **1,339 µs of work that owns no GEMM dependency on epoch i**.

**The design.** Restructure the launch into a **persistent two-generation pipeline** (TBO-2, Lancet/ATOM lineage, with three AMD-specific twists that are genuinely new):

- **Generation-parity state.** Every epoch-scoped buffer exists ×2 (`a_ll`, `a_dst`, `sc_stage`, `recv_*`, `hcnt`, `pull_*`, `sei`/`tile_desc`, `A2q`/`DQ2` (+553 MB), `slots` (+448 MiB), `out`, all MPS protocol cells) — ~+1.2 GB/rank, trivial on 288 GB HBM. The existing `dest_counter` epoch-parity mechanism and the mode-14 rendezvous are the templates; M0's retire-wait becomes parity-indexed (wait for the *same-parity* generation's retirement, two epochs back).
- **One unified ticked work list per epoch** replacing the sequential phase calls: `{M6(i) tasks} ∪ {M1/M2(i+1) dispatch stripes} ∪ {M3–5(i+1) plan slices} ∪ {M8(i−1) combine batches}`. Order: dependency-driven (M6 first-class; dispatch stripes leak in steadily so M3(i+1) can scan as soon as its histogram closes; M8(i−1) is *fully ready immediately* — its epoch finished long ago).
- **Combine escapes its readiness law.** M8(i−1) under epoch i's M6 shadow is freed from the `(t/S)^8` cap entirely — it is no longer waiting on *this* epoch's stragglers. The r=−0.904 M7/combine coupling is broken by construction. 324 µs of combine and 1,015 µs of dispatch+plan hide in M6's 84% stall pool and M7's throttled shadow.

**Three novel AMD-specific tightening mechanisms** (these are what make it a contribution rather than a TBO replication):

1. **Dispatch admission bound.** M1's pushes get the exp_24 treatment — a compile-time-selected `vmcnt(N)` on the LL128 push stream (or the SDMA-queue depth of K1). Today dispatch pushes are unbounded because dispatch runs alone; overlapped, 162 MiB of unbounded scatter inside M6's memory phases is precisely the exp_35 rung-(b) failure class. **The throttle lesson generalizes to every new write class from day one.**
2. **The stall-pool budget ledger.** M6's measured 84% stall is the overlap budget; the dispatcher paces injected work by *measured* K-loop residency (a cheap per-CTA progress counter), keeping co-resident traffic in the class exp_22 showed is free (payload/stores) and out of the class exp_20 showed is toxic (per-row atomics/polls). A readiness-driven queue exists, but the *admission policy* is the mechanism — this is the paper's derived-scheduler claim in embryo.
3. **Solver-free epoch overlap.** No second persistent kernel, no stream juggling: one launch, one LDS budget, work-list tickets — preserving the kernel's discipline that "the node checkout is the arm" and avoiding the register-allocator cliff (each work class stays in its own body; L7 says we gate `.text` parity per class).

**Expected magnitude.** Steady-state epoch cost ≈ interior(i) + unhidden(ε) with ε the hide efficiency: ε=0.5 → ~5,800 µs (0.75×); ε=0.8 → ~5,200 µs (0.68×); even ε=0.3 → ~6,050 (0.78×, still under the 0.80 dream line). **This is the only idea in the document sized to break decisively past 0.80×.** The first-order counterargument is work conservation (exp_25's honest frame): the hidden work is real CTA work. The measured reply: M6/M7 are *memory*-bound at 13.7–17.1% issue duty, while dispatch is VALU+store and combine is poll+stream — complementary mixes, and exp_22's 0.073% co-residency tax is the calibration point. ε is the experiment's estimand, stated honestly in advance.

**Build order (each its own TU, default-off, ratchet `.text` parity per L7):** (1) parity buffers + parity retirement (null schedule, gate-only); (2) deferred combine (M8(i) at top of epoch i+1; the cheap ½-TBO that subsumes into the full design); (3) dispatch-stripes under M6 with the admission bound; (4) plan slicing; (5) the full work-list scheduler.

**Falsifiers:** measured ε < 0.25 (the hide stalls); L2/L3 violations show up as M6 request-bandwidth collapse; generation-reuse races (the identical-input trap — NaN-poison arms are a signoff condition, *proven to fire on the hook-removed control* before any sweep number goes upward).

---

## 5. The dispatch×GEMM-1 same-epoch question — answered precisely

The user asked whether fine-grained overlap is possible between dispatch and the first GEMM. The dependency chain says **no, and it is worth writing down why exactly**:

- M6 needs `sti/sei/tile_desc` (the sort). The sort's per-expert extents need **every source's** histogram bins for expert `e`; every source's rows arrive through all 8 chunks of M1/M2. There is no partial-arrival prefix that completes any expert (routing is uniform; a source's chunk `c` carries rows for all 32 local experts).
- M1 push ↔ M2 unpack **is already the overlap** (consumer polls chunk words while producers quantize; zero peer wait — exp_10). The 642 µs residual is own-work: quantization math, LDS staging, unpack stores. No dependency-structure change makes a rank's own work disappear.
- W13 prefetching under dispatch is useless (939 MB vs 256 MB LLC).
- The one genuine same-epoch residual: M3's single-CTA scan (`bid==0` block) while 255 CTAs idle — worth ≤ tens of µs, in the noise (372.8 µs plan total, done as a target).

**Conclusion:** the dispatch-phase overlap prize is **cross-epoch** (K3) or **engine-carried** (K1), not same-epoch fine-grained interleave. This is a falsifiable structural claim, and it should be written in the paper as one of the "why the regime boundary sits there" results.

---

## 6. K0 — nc-major M7 task order + pipelined combine (the missing waterfall rung)

exp_29's discovery: today's order `task = tile·16 + nc` makes a token reducible only at `(t/S)^8`; reordering `task = nc·num_tiles + tile` turns readiness into a 16-step staircase with **15/16 of the combine unblocked before M7 ends**. The `nc → XCD (nc mod 8)` L2-residency invariant is *preserved* (arguably strengthened) because every CTA works one nc-slice class at a time. The service drain's `nc_arr` counting survives verbatim — readiness becomes per-(row) complete at its last nc-slice, and M8's poll structure is unchanged; what changes is *when* flags arrive relative to M7's progress. Pipelined M8 claims then ride the staircase. **Predicted: GEMM-time Δ ≈ 0 (falsifier: L2 residency break shows in M7's stamp), combine-shadow gain −150…−300 µs of the 3,026 coupled block.** Cheapest real overlap edit in the document: one decode swap at `n2_phase2_gm_mps.cpp:331-332` plus the readiness-accounting review. This is also exp_35's named missing rung (f) — figure data and a ratchet candidate in one.

## 7. K4-prime — burst de-alignment probe (rotated-nc start)

The epilogue surcharge is congestion from **aligned** bursts: 240 CTAs march in lockstep at 78.3% of ceiling. exp_25 caps de-alignment's value at +211 µs. A one-line probe: rotate each CTA's nc phase (`nc' = (nc + gid·φ) mod 16` in task decode) so epilogue bursts interleave across CTAs. Expected end-to-end Δ ≈ 0 ± small; the *phase stamps* resolve the burst-structure change. Scientifically it prices the de-alignment term in the regime map; as a ratchet candidate it is cheap to test and honest to lose. **Do after K0** (K0's staircase is the stronger order effect).

## 8. K6 — per-destination credits: the skew-rescue arm to carry unfailingly

Today's bound is a **global** per-wave `vmcnt(4)`: destinations are indistinguishable. Under skew (hot owner, incast), the scarce resource concentrates per-link/per-owner, and the right mechanism is **matched-total per-destination credits**: `credit[owner]` cells on the producer, refilled by the owner's consumption posts (M8's consume-and-zero already knows the (producer, rows) ledger — a per-M8-batch relaxed system store, ~1 KB/epoch of return traffic). The epilogue checks credit amortized per owner-run (rows in a tile are same-expert, owners uniform — the check is ~1 relaxed load per row, folded under the existing branch). Balanced routing => credits never bind (predetermined falsifier: must tie mode-12 within 1% at std=0, or the arm is wrong by construction). Skew => only the hot direction throttles; the other 6 proceed. Compose with **consumer-major task order** (produce the most-consumed group first) for the full Stage-5/6 interaction. This is the mechanism the study's method map needs for its skew cell, and the megakernel is the only vehicle that can test it end-to-end.

## 9. K7 — device-side policy byte: the regime-map selection made real

The study's target contribution is a **policy chooser**, not a fixed winner. The kernel already computes the selectors on device: M3/M4's histogram gives total volume, per-expert spread, hottest destination, tile count; the config word is already descriptor-carried and readfirstlane-decoded. A policy byte emitted at the `tile_desc` build site (KRN:1391–1413) can select, per epoch, per rank: epilogue class {RMW-d4 | staged-SDMA}, order {tile | nc-major}, readiness {per-row | coarse-epoch}, credit scope {global | per-dest}. Four policies to compare per the study §11: static-global (today), shape-indexed lookup, offline oracle, this online rule. **Discipline (L7):** each variant must be a separate TU or `-D`-guarded with default-arm `.text` parity — policy *choices* at runtime, policy *codegen* at build time. The +726.9 µs regression is exactly what "runtime-branchy megakernel" buys you if you skip this.

## 10. K8 — three-lane hybrid (decode/training branch overlap)

For regimes with a shared expert (and for decode's high-concurrency case): compute lane A = routed megakernel(i); lane B = shared-expert GEMM(i) on independent CTAs; engine lane = SDMA dispatch(i+1) | combine(i−1). This is the closest analogue to MORI's AsyncLL measured serving win (TPOT 97.35→83.32 ms) and GEMM/RCCL-overlap's "tune the compute grid × communication CU partition jointly" lesson. The synthetic prefill harness has no shared expert; this arm belongs to the decode/training proxy tier, and its first form is least-invasive: full current megakernel for batch i, external engines for i±1 — never duplicating batch i's dispatch/combine.

---

## 11. The strategy-switching map (prefill by batch size, decode, training)

This is the user-facing decision table. Entries marked **[measured]** are established; everything else is a pre-registered expectation the suite must be allowed to disprove. **The entire T axis is hostage to fixing the T=1024/2048 correctness defect first.**

| Regime | T (tokens/rank) | Dispatch carrier | Expert-output transport | Combine structure | Task order | Readiness | Flow control |
|---|---|---|---|---|---|---|---|
| Decode, low concurrency | ≤64 | direct CU LL128 push (control publication dominates; MORI 2-record 9.07 µs class) **[measured trend]** | epilogue RMW, depth 4 (bytes tiny) | same-kernel drain | tile-major | coarse epoch (mode-14 class) | global depth 4 |
| Decode, high concurrency | ≤64 × many req | + TBO-2/4 across microbatches + shared-expert lane (K8) | as above | deferred into next microbatch | consumer-major | coarse | global |
| Prefill mid | 512–2,048 | SDMA packs when per-pair ≥ Stage-1 crossover (~64 KiB–1 MiB); else direct **[crossover to measure]** | epilogue RMW depth-4 (ratchet class) **[measured at 4096]** | pipelined | **nc-major (K0)** | grouped per-nc | global depth 4 |
| Prefill large | 4,096 (today) | **K3: hidden in prior epoch's shadow; SDMA packs (K1)** | K2 staged-SDMA ≈ wash→win **[test]** | K3: deferred fully-ready | nc-major | signal-fused (K4) | depth 4 + dispatch admission bound |
| Prefill XL / training microbatch | ≥8,192 | as above | **staged-SDMA wins** (RMW congestion super-linear; SDMA linear; packs ×2 deeper in the amortization band) | deferred | nc-major | signal-fused | per-dest credits if skew |
| Training step (fwd+bwd) | + backward | 3-lane (K8): DP ReduceScatter/AllReduce (RCCL/SDMA) shadowed under next forward **[MORI/GEMM-RCCL expectation]** | staged-SDMA | deferred | nc-major | signal-fused | per-dest credits |
| Skewed MoE | any | same | same + consumer-major order | pipelined | consumer-major | grouped | **per-destination credits (K6)** |

**Critical-size switching rules to measure (the paper's map):**
1. **Pack-size crossover** for SDMA dispatch: Stage 1's record sweep gives it directly; pre-registered guess 64 KiB–1 MiB per (pair, chunk).
2. **RMW-vs-staged-SDMA crossover** in epilogue bytes/rank: the surcharge curve (op-class) vs linear engine curve; pre-registered crossing ≈ current–2× current bytes.
3. **TBO entry point:** when dispatch+plan ≲ M6's stall pool — holds at T=4096 by 2×; find the small-T exit.
4. **Skew CV where per-destination credits overtake global depth-4** (Stage 5's matched-total comparison).
5. **Whether fine readiness ever pays** — expected answer: only when the consumer's dependency graph actually branches early (nc-major staircase); flat otherwise (the (t/S)^8 law + MORI's LL128 result).

---

## 12. What to ask the neutral suite for (inputs these designs need)

1. **Remote op-class cost decomposition** (decides K2's sign): 4 B remote RMW vs 4 B remote store vs 128 B packet store, isolated + under matched compute, same XGMI path. The kernel's whole ~860 µs surcharge sits on this axis and it has never been decomposed.
2. **SDMA end-to-end cost** at exactly the two embed points: 23 MiB and 58.7 MiB contiguous per (pair, epoch), shader-initiated with quiet+ordered signal vs host-enqueued; enqueue, descriptor-grouping, queue-depth, completion-write costs itemized per the study §5.3.
3. **Co-residency interference at the dispatch mix** (VALU quantize + LDS staging + PLL stores/SDMA) under an M6-shaped GEMM proxy — this *is* the TBO efficiency ε's first estimate (K3's falsifier gate).
4. **Credit-check overhead class** (one relaxed system load per row-batch inside a 256-VGPR epilogue) — K6's cost floor.
5. **Register/TU structure data** for K3: does splitting dispatch strips into a sibling TU preserve the default arm's `.text` (the exp_38 gate applied forward).

---

## 13. Build order (dependency-sorted; each arm its own TU + default-off guard + `.text` parity)

| # | arm | depends on | est. value | type |
|---|---|---|---|---|
| 0 | `C≤8` stamps-off confirmation (already 7/7 paired) | exp_38 healthy pin | −28…−34 µs | measurement, queued |
| 1 | T=1024/2048 correctness fix | — | unblocks the whole map axis | defect |
| 2 | **K0 nc-major + pipelined combine** | none | −150…−300 µs on the 3,026 block; waterfall rung (f) | cheap edit |
| 3 | mode-14 rung-(d) re-run → substitutes-vs-complements settled | exp_38 | map cell | measurement |
| 4 | K4-prime rotated-nc de-align probe | K0 | ±; prices the de-alignment term | probe |
| 5 | **K1 SDMA pack dispatch** | Stage-1 crossover (advisory) | −100…−190 µs solo | engine arm |
| 6 | **K2 staged-SDMA combine transport (mode 15)** | Stage-1 op-class numbers (advisory) | −200…−800 µs; the op-class answer either way | engine arm |
| 7 | K5 deferred-combine (½-TBO) | parity buffers | −150…−300 µs; proofs for K3 | structural |
| 8 | **K3 TBO-2 epoch pipeline (the flagship)** | K5, K1's admission bound | **−600…−1,300 µs; the 0.80× breaker** | structural |
| 9 | K6 per-destination credits + consumer-major | harness skew axis (exp_36: currently absent) | skew cell of the map | flow control |
| 10 | K7 policy byte | 2, 5, 6 measured | the §11 scheduler contribution | selection |
| 11 | K8 three-lane decode/training hybrid | decode harness | regime row | scope |

Non-overlap compute pools worth attacking in parallel (orthogonal): **M6's 84%-stall K-loop** (two-K-step pipeline; single largest pool of cycles in the kernel), **G=4** (intensity 161.7→204.8, predicted M6≈1,957 µs, ~480/512 registers — exp_28's "last free point"), exp_31's phase-2 VMEM hint (`14+kGM`), and exp_30's S-1 fanout guard (~154 MiB of dummy reads, ~free).

---

## 14. Standing cautions (inherited)

- Screens resolve ~6.6% end-to-end; phase stamps ~1%; end-to-end numbers from 5-rotation campaigns only.
- `pperr != 0` is terminal; never clear-and-retry.
- M7 and combine are anti-correlated: judge their **sum**, never either alone; combine-alone deltas <60 µs are noise.
- The identical-input trap: any staleness-shaped bug returns bit-correct answers on this harness. NaN-poison arms must be **proven to fire** before any overlap arm's number goes upward.
- `.text` sha identity or a same-session measured control is the only trusted parity; resource tuples passed while the arm lost 726.9 µs.
- SDMA is an executor claim, not an API label — no trace, no map point.
- A bandwidth win without a makespan win is a transport result, not an overlap win.

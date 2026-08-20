# UNDERSTANDING_PLAN r2 — beat production at low concurrency, and the anatomy campaign

**Date:** 2026-08-18 (r2: folds two adversarial reviews — `review/UNDERSTANDING_PLAN_{ALIGNMENT,MECHANISM}_REVIEW.md` — plus three operator decisions). Supersedes the scope of
`ABLATION_METHODOLOGY.md` per mentor decision; that document survives as the extension plan.

**Primary objective (operator): beat production end-to-end at low concurrency — Track 1, §4a.**

**Scope:** prefill only, two anchor cells. Anchor A = high-concurrency DP8/EP8 (driver closed-loop
C=512-class, T=4,096 tok/rank, ISL 4,096, OSL 8; the 4,096-token boundary rig). Anchor B =
low-concurrency deployment (C=32, ISL 4,096, OSL 8, vendor `max_num_batched_tokens` 16,384),
whose per-step chunk is the 16,384-token / 235 MB collective — **low concurrency, largest
message**. Not in scope: decode, mixed batching, gfx942, multi-node, workload-generalization
claims, cost-model fit/hold-out.

## 1. The reorientation

Mentor's direction: *we have already written a lot of MoE-region megakernels; some fast, some
slow. Understand — principally, mechanism-level — WHY, and expose abstractions from that. No
requirement that one kernel wins everywhere.* Sharing model: the Mixture-of-Kittens writeup —
system → mechanisms → evidence per choice → lessons.

| kept (re-purposed) | dropped |
|---|---|
| the instruments: device phase ledger (E-B1/E-B2 + a separately-priced DP-body port), actuation/ISA receipts, **the ρ (`k_inner`) ladder — restored per both reviews; anchor-B mechanism claims may not be read at a single ρ** | the W-grid: T-ladder-as-sweep, fill cell, skew cell, decode regions, gfx942 arm |
| the four-door occupancy-1 frame as *explanatory* vocabulary | Q-sufficiency falsifiers FQ-1/FQ-4 |
| interaction cells re-crossed on the axes that matter: **depth × C** (same resource class per exp_03), depth × certification (the C6 substitutes question), FQ-2 with intensity ⟂ duration | registered-prediction/holdout machinery; Phase-4 e2e ladder; the 26-prediction family |
| mechanism cards (law cards, renamed) with a closure criterion | **R1 dedicated repeatability arm — operator cut**; noise bands come from banked per-instrument values + free repeats (§4b) |
| abstraction extraction — **retargeted to hardware-agnostic data-movement contracts (Iris/IrisX-aligned); DHK headers demoted to one backend** | the composition-policy headline; **the HipKittens-PR framing** (mentor: HK abstractions read too NVIDIA-specific) |

Pinned constraint (replaces "X2 not needed"): X2 is unscheduled, therefore **any T≠4,096 number
must come from the tgen body and be labelled as such**; the serving M15 body may not produce one
(it failed the correctness gate at T=2,048/1,024; the archived `t2048_m15_*`/`t1024_m15_*`
campaigns are retracted as M15 data). Whenever the T-sweep is shown, cite the LAW-39 correctness
repair.

## 2. The specimens

**Anchor A — DP8/EP8 prefill boundary, T=4,096/rank, ISL 4,096, OSL 8, uniform routing unless
stated** (sources: BRIEF digest; exp_03/exp_14/exp_20/exp_35; pins differ across sessions and
are labelled in the manifest):

| arm | µs | vs prod | role in the anatomy |
|---|---:|---|---|
| production (AITER+MORI, unfused) | 7,712 | 1.000× | reference waterfall — itemized too (R3) |
| homogeneous megakernel | 6,908.8 | 0.8958× | fusion alone, zero cross-rank protocol |
| producer-carried RMW, unbounded | +211.2 vs homogeneous (paired, same session) | 0.9226× | the instructive regression: relocation without flow control |
| dedicated comm CTAs, C=64 (staged-push carrier) | 6,866 | 0.888× | **carrier-conditional, not "falsified": dedication WON here (0.888× vs homogeneous 0.8958×) and lost monotonically inside the producer-carried carrier.** The +831.5 µs is an M7 phase delta measured with the pool's payload copy deleted — **elapsed-time overhead from protocol + interference, not traffic** (exp_20) |
| mode-12, depth-4 | 6,483.8 | 0.8407× | producer-carried + injection bound (−615.0 in exp_35's paired ladder) |
| M15, 16/24/28 consumers | 6,292 / 5,848 / **5,822** | 0.7544× | + order + slab certificates + consuming pool; best point 24–28 (24→28 is still −26.5) |
| M15, 32 consumers | hung 1 of 2 | — | the cliff past the knee — Q7 |
| **S8 / m15b staged wide-push** | 9,087.1 (built, one run) | — | throttle annihilated by codegen: 96 scratch loads + 97 vmcnt(0) next to the remote atomics — the register-pressure card |
| **mode-14 reachability** | +726.9 on mode-12, source untouched | — | a megakernel made slow purely by the register allocator (LAW-32) — the most teachable non-obvious specimen |
| mode-16 deferred combine | +75.8 | — | *"the wait it escaped was mostly not on the critical path, AND the parity state it added was"* — both halves are the lesson |
| fine-grained per-row readiness | falsified | — | ~926k protocol ops/rank/epoch |

Excluded by name: T3 (zero-grid-barrier design, never built) — noted because Q8 leans on the
grid barrier as a residency contract.

**Anchor B — TP8+EP prefill boundary, 16,384-token / 235 MB collective** (G25 commits;
bandwidth conventions labelled — never mix per-rank-egress with logical-bytes/wall):

| arm | measure | role |
|---|---|---|
| GEMM → RCCL (phased) | 2,909 µs | the champion we lose to (our rig's arm — production's own TP8 step is itemized in G-L0c) |
| fused CDAR (ERS+MAG) | 3,606 µs | the fusion that doesn't pay — the honest half |
| RCCL AR @235 MB | 1,470 µs (per-rank-egress convention: 280 GB/s against an **unreproduced** 349–355 ceiling) | what a mature collective achieves |
| our transport @235 MB | β vs RCCL currently only **subtraction-derived (1.506×)**, with a second directly-measured convention giving 1.363× — **G-L0's standalone arms replace both with direct measurement** | the transport deficit |
| store-towers / atomics transport | 93.6–117.2 / 67.1 GB/s (logical-bytes/wall) | op-class evidence |
| four G25-1 overlap variants | **null at ρ=0.65 — where achievable hiding was min(compute,transport)=1,439 µs and the measured 47 µs is 3.3% of that ceiling. Not "falsified": untested at deployment ρ≈2.6–3.2. Re-tested (not re-scored) on the ρ ladder** | Q10 |

## 3. The attribution questions

Closure criterion (applies to every Q): a row reaches **EXPLAINED** only when (a) itemized spans
account for ≥80% of the banked delta with the residual below the instrument's noise band, and
(b) the card states what we would have seen if the mechanism were wrong. Otherwise PARTIAL/OPEN.

| # | delta / phenomenon | mechanism status | the settling experiment |
|---|---|---|---|
| Q1 | production → homogeneous −804 µs | PARTIAL — never itemized | R3, **one instrument per quantity**: HIP events for both totals; cross-rank rank-max device stamps for both arms' phases (never rank-0 — +38% gap on combine-class phases); launch gaps production-side only; phase-correspondence table (MORI staging ↔ dispatch/plan) published BEFORE the run |
| Q2 | homogeneous → unbounded +211.2 (paired) | PARTIAL | rides Q3 |
| Q3 | unbounded → depth-4 −615.0; what resource does the bound guard (in-flight bytes vs time)? | OPEN | R6, run **co-resident at corrected ρ** (depth is flat in pure transport — no effect there): vary op width within one op class (8/16/32 B store packets), actuate rate by loading the destination |
| Q4 | depth binds iff compute co-resident | MEASURED both ways, never one rig | R5 with a `bytes_per_mfma` argv so **intensity ⟂ duration** (the stock burst scales both together and cannot test "contention, not bytes") |
| Q5 | mode-12 → M15 **−661.8** | PARTIAL — component figures (−573 protocol, −144 order×cert) are NOT quotable: the −573 was measured with the injection bound inert (C6), the −144 is unsourced | R4's crossed design; K5's real question is **C6: are the depth bound and coarse certification substitutes bounding the same in-flight resource?** |
| Q6 | consumer ladder: −93 (8→16), −444 (16→24), −26.5 (24→28) | PARTIAL — banked stamps already split the −444 into **injector-concurrency relief ≈354 + sweep ≈95** ("C is a second, coarser flow-control knob on top of depth 4"); the consumption model (C_sat=20.8, ρ=6.1 GB/s/CTA **derived**, κ_couple=5.4× fudge) is the rival, not the explanation | R4: K4 trimmed to C ∈ {16,24,28} (+{30,32} shared with Q7), **M7 stamp reported per cell**, crossed with depth ∈ {4,8} — the interaction neither old design measured |
| Q7 | C=32 hang, 1 of 2 runs | OPEN — with a recorded hypothesis, not folklore: *the pool holds the next rendezvous hostage — the failure `flush_rows` exists to prevent* ⇒ predicts a flush interaction. In-repo analogue: 7 ranks spinning `rows_done` while the 8th abandoned the step; `pperr` is sticky, never host-reset — a 1-in-2 event can read as permanent. **Livelock is a registered competing hypothesis** (spin loops predate the s_sleep lesson) | R7, BEFORE R4: n≥8 per point at C ∈ {28,30,32} on the **shipping** binary → hang-rate table; `rocgdb` attach to a hung process (the prime-suspect `hkp::grid_barrier` has no source in this repo — parked PCs are the zero-build look); all-8-rank capture (rank-N failures invisible in rank-0 stdout); discriminator: C=32 × flush_rows ∈ {1,16}; spin counters are running maxima — reset or don't quote |
| Q8a | dedicated-carrier scope: won in staged-push, lost in producer-carried; MoK's dedicated SMs work on Blackwell | OPEN | R8 on the **DP chassis** (the phenomenon's home): {comm in-kernel vs comm as second concurrent kernel} × N_CTA; occupancy asserted and printed per arm; LAW-54's registered prediction: the second-kernel arm loses |
| Q8b | the g-independent interference floor: ≤1,159 µs, 36% per-XCD L2, **64% unidentified** — "the single largest gap in our understanding of this kernel" | OPEN, PARTIAL-by-construction (no PMC path inside a persistent launch) | ablation, not instrument: vary the pool's **line footprint** at fixed op count (LAW-18 axis); bounded ambition stated |
| Q9 | RCCL beats our transport — by how much, and how | OPEN — β currently subtraction-derived | R9 @ **235 MB only**: standalone RCCL arm + standalone CDAR-transport arm in one binary (direct β, no subtraction); `NCCL_ALGO × NCCL_PROTO × NCHANNELS` env sweep on the in-rig `ncclAllReduce`; matched-block-count parallelism control (the likely answer is issue-rate/parallelism, not protocol) |
| Q10 | fused CDAR loses to phased by 697 µs despite overlap | PARTIAL — the four G25-1 nulls are ρ-capped (3.3% of ceiling); h as wall-difference conflates hiding with co-residency tax; and h_ledger−h_wall gives **ΔI_co only** — the phased arm's own +335 µs tax needs the transport-only arm | R2: ledger × ρ ∈ {0.65,1,2,3} × {compute, phased, fused, rccl, **transport-only**} |
| Q11 | the fused family inverts below T≈1,600: fixed cost **1,068.2 µs vs production's 181.3** (two-point fit, unfitted midpoint ~1–2%) — the mentor's "why are some slow" at low tokens | PARTIAL — the 1,068 has never been itemized | R11: ledger/rocprof itemization of the **existing tgen-body** arms at T=4,096 and T=1,024 (labelled tgen body; LAW-39 repair cited). No new workload points |
| Q12 | the sign flip: the SAME reservation is +332…+340 carrying, +8–9/CTA idling, −444 consuming — *"the only knob with a measured sign flip under one mechanism"*; the experts' literal question | PARTIAL — and Q6's stamp evidence suggests the flip may be injector-concurrency, not carry-vs-consume | rides R4's stamped cells + R8's second-kernel arm |

Known-unknowns carried honestly: the +335 µs phased-arm tax (Q10's transport-only arm derives
it); `bps` 1→2 = +64% (an occupancy law under every granularity claim — R10 holds block count
constant via a `bps` compensator); LAW-31's scope (dispatch a2a shows zero exposed wait only
under uniform routing and 4 pre-M6 grid barriers).

## 4a. TRACK 1 (primary) — beat production end-to-end at low concurrency

Target cell: C=32, 1,024 QSL prompts, ISL 4,096, OSL 8. Production (AMD-recommended TP8+EP,
AITER+RCCL): **25,844 input tok/s, TTFT p50/p99 1.339/5.682 s**. Our current best there is the
wrong-topology arm (M15 DP8/EP8: 20,372, −21.17% — an arm-level delta at two different W points,
never quotable as a kernel delta). **Win condition: our TP8+EP arm > 25,844 input tok/s with
TTFT p50 ≤ production +10%, ≥5 order-balanced pairs under `nightshift/BENCHMARK_PROTOCOL.md`.**

The prize pool — ~2.8 ms/layer of exposed all-reduce (≈27% of the step), with the fabric needing
only a 28–39% duty cycle — is DERIVED and carries an honesty flag: **never profiler-confirmed on
the native TP8 arm.** G-L0c confirms or corrects it before anything is built against it.

| gate | arm | pass bar | build | node-h |
|---|---|---|---|---:|
| **G-L0a** | ρ ladder: `k_inner` → ρ ∈ {0.65,1,2,3} × {compute, phased, fused, rccl, transport-only, standalone-RCCL} | h_ledger(ρ), I_co absolute, **direct β**; the four G25-1 arms re-tested at deployment ρ | one argv + small rig additions | 0.8–1.2 |
| **G-L0b** | device phase ledger on the same arms | per-layer budget: exposed-AR span, produce/wait/commit | E-B1/E-B2 (~1 engineer-day, TP8 rig) | 1.0–1.5 |
| **G-L0c** | **one profiled native-TP8 production step** | the 2.8 ms/layer exposed-AR figure confirmed or corrected | none | 0.3–0.5 |
| **G-L1** | RCCL-hybrid floor: mega owns the MoE region, RCCL owns the AR | boundary ≤ 2,909 µs (fusion of the non-AR parts must not regress) | M (assemble from G25 rig parts) | 0.5–1.0 |
| **G-L2** | + shared-expert filler under the AR wait (under TP8 the shared-expert output folds into the same accumulator, deleting its separate AR) | hides ≥25% of the **G-L0-measured** exposed AR (prior: 32–34% coverage) | M | 0.5–1.0 |
| **G-L3** | + metered AR (spread the collective at ~1/3 duty instead of bursting) and/or fused CDAR readmitted **only if** G-L0 shows h_ledger clearing `h* = 1 − (1 − I_co/T_vendor)/β` at the **directly-measured** β, or R9's env sweep closes β below ~1.2 | boundary beats 2,909 µs by the G-L0-measured available margin | M–L | 1.0–1.5 |
| **G-L4** | e2e C=32: our TP8+EP serving arm vs production TP8+EP | the win condition above; coverage + composition counters green | **L — TP8 serving integration (mega serving path is DP-shaped today)** | 4–6 |

## 4b. TRACK 2 — the anatomy run list

**Rules of engagement** (extended per review): `pkill` bracket trick over ssh; first invocation
of a session discarded; clocks/temps logged (GPU2 runs hot); every run appends `runs.jsonl`
before the next starts; a hang is a manifest outcome value, never a dropped point; **build
identity per instrumented arm** — `.text` sha256 recorded, flag-OFF build proven
`.text`-identical, flag-ON proven to differ; **instrumented-vs-uninstrumented wall parity within
the noise band, else the arm's spans are void**; `timestamps` state frozen across any ladder that
locates a knee (the stamp flag costs ~15 µs — comparable to the 24→28 step); one instrument per
quantity; analysis clustered by run, never (run, rank); one bandwidth convention per table.

**Noise sizing (R1 cut — operator decision).** Per-instrument banked bands are used instead:
mega chassis LAW-60 (screens σ=0.52%, campaigns reproduce to 0.09%, and a 6.6% control drift
within one batch ⇒ arms must alternate within a session — cross-session lone-arm contrasts are
void); production boundary cross-session spread ~0.3% (7,709/7,712/7,713/7,729); the TP8 rig's
band falls out of G-L0a's K=3 repeats at no extra cost. Any delta inside ~2× its instrument's
band is reported as "within noise."

| run | what | K×n | node-h (1–3 min boundary unit where applicable) |
|---|---|---|---:|
| **R2** | = G-L0a+b (shared with Track 1): ledger × ρ × 6 arms, TP8 rig | K=3/config | (in Track 1) |
| **R2c** | **DP-body ledger port** — separate build item; acceptance = all five LAW-63 guards (lgkmcnt(0) before s_memrealtime; 100 MHz tick; running-maxima reset; **cross-rank reduction to rank-max**; no cross-kernel clock64 spans) + wall parity | K=3 | build ~1 engineer-day + 1.0–1.5 |
| **R3** | production vs homogeneous itemization per Q1's instrument rules | K=3 each | 0.7–1.2 |
| **R4** | one MoK session, one binary, runtime knobs: **depth {4,8} × cert {on,off} × C {16,28}** (the C6 + depth-C crossings) + K4 C ∈ {16,24,28} with per-cell M7 stamps + order toggle at the best config; includes the **one-session homogeneous→mode-12→M15 re-run** with `.text` sha receipts and the −618.7 restoration contrast, so the waterfall rungs share one pin; non-additive pairs marked on the figure (LAW-21/C6) | K=4–5 | 3.0–3.6 |
| **R5** | Q4: ± co-resident body with `bytes_per_mfma` argv (intensity ⟂ duration), at corrected ρ | K=3 | 0.4–0.6 |
| **R6** | Q3: width-within-op-class + destination-load rate actuator, co-resident | K=3 | 0.3–0.5 |
| **R7** | Q7 hang forensics per the M4 protocol, **before R4** | n≥8/point | 1.0–1.5 |
| **R8** | Q8a on the DP chassis: {in-kernel vs second-kernel} × N_CTA, occupancy printed | K=3 | 0.8–1.2 |
| **R9** | Q9 @235 MB: env sweep + matched-block-count control (standalone arms live in R2) | K=3 | 0.3–0.5 |
| **R10** | granularity with the `bps` compensator (blocks held constant) | K=3 | 0.5–0.7 |
| **R11** | Q11: tgen-body T ∈ {4,096, 1,024} itemization | K=3 | 0.4–0.6 |
| | **Track 2 subtotal** | | **≈ 8.4–11.9** |

**Campaign totals:** Track 1 ≈ 8–12 (incl. shared R2) + Track 2 non-shared ≈ 7–10 ⇒ **≈15–22
node-hours**; with a +20% retry allowance and the 0.6–0.7 node-availability derate: **≈27–44
wall-hours ⇒ 4–6 overnight sessions.** Track 1 leads; R7 precedes R4; R2c and G-L4 are the only
build items beyond one-day size.

## 5. The deliverable — `MEGAKERNEL_ANATOMY.md` (MoK-shaped, restructured per review)

1. **The problem and the two anchors** — full workload tuples; why these two cells are the
   deployment regimes (the C=32/C=512 flip lives here).
2. **The megakernel, described** — 256 CTAs, occupancy exactly 1 (256 VGPR + 256 AGPR and
   155,428 B LDS independently), phases M0–M9, roles, the epoch/certificate protocol, the fusion
   boundary, one dataflow figure. *A reader must be able to parse "slab certificate" and
   "consuming pool" before any number appears.*
3. **Design decisions, one per mechanism, with the evidence that forced each** — the Q-cards,
   ordered as decisions a builder faces. Framing stated first: **the two GEMMs are 88.1% of the
   interior — the entire protocol-and-comm story is a fight over ~12%.**
4. **What did not work, first-class** — unbounded injection (+211.2), the carrier-conditional
   dedication story, deferred combine (both halves), fine-grained readiness, XCD-confined pools
   (net 7–23% worse), pacing (monotone, closed), coalesced protocol atomics (**≥10× slower —
   atomics resolve at the line; maximize L, the opposite of the load rule**), S8's annihilated
   throttle, mode-14 reachability, the four ρ-capped TP8 nulls, the C=32 hang verdict, and the
   **M23 coverage defect** as the integration lesson (the mega ran ~2% of in-bucket steps;
   every pre-M23 serving A/B measured a doubly-handicapped candidate).
5. **The anatomy** — the two waterfalls, readable now; rungs from one pin where re-run (R4);
   non-additive pairs marked; Track 1's before/after on anchor B.
6. **The abstractions — data-movement contracts, not hardware primitives** (mentor constraint,
   binding). Each card carries two separated fields:
   - **the contract** (portable, Iris/IrisX-level): bounded remote-op admission (what the credit
     meters is Q3/Q4's answer); coarse publication — signal the smallest early-EXECUTABLE unit
     (t50 = 91.7% under natural order); counted fan-in with runtime arity; ready ≠ reusable
     (retirement credits); push when the producer owns progress, pull when the consumer does;
     destination layout is part of the movement contract (52.8 vs 4.1 GB/s); carrier identity as
     a function of grid residency, not vendor.
   - **backend notes** (gfx950, beneath the line): vmcnt throttle encodings, packed-bf16 remote
     atomics, MFMA co-residency, compile-time-codegen constraints. The gfx942 atomic:store
     reversal is the worked example: the contract survives, the implementation flips.
   The DHK `distributed/*.cuh` headers are **one backend** of these contracts; this section
   reads as a proposal for an Iris/IrisX-level API surface.
7. **How we differ from MoK, and why** — their dedicated comm SMs work on Blackwell with CLC
   work-stealing and no grid barriers; our dedication is carrier-conditional on an all-CU,
   grid-barriered, occupancy-1 CDNA grid (Q8a's scope result). The most interesting page.
8. **Lessons, scope, and reproducibility** — what we'd tell someone starting on CDNA; every arm
   regenerable from its manifest row; the T≠4,096 tgen-body labelling rule; scope statement (two
   prefill anchors, gfx950; T-sweep/fill/skew reported as observations with provenance).

## 6. Kill criteria / degradation

- **R2 inconclusive** (ledger perturbs beyond parity, or spans don't close the budget): ship the
  spans as bounds, publish the perturbation as a finding, and downgrade affected Q-cards to
  PARTIAL with the residual named.
- **R7 does not reproduce** on any binary: ship the hang-rate table + a static wait-graph hazard
  analysis of the epoch/credit words as a design-hazard card; the C=32 cliff stays a documented
  operating limit.
- **R2c (DP ledger port) slips**: the anchor-A waterfall ships from HIP events + rocprof only;
  phase-level claims scoped to what those instruments support.
- **G-L0c corrects the prize pool downward** below ~1 ms/layer exposed: Track 1's G-L2/G-L3 are
  re-sized before build; the e2e win path re-routes through G-L1 + serving-side levers.

## 7. What we deliberately do not do

No new kernel variants except instrumented/debug builds of existing ones and Track 1's gated
arms. No workload grid. No decode. No claim that any arm is optimal outside its anchor. No
cost-model validation program — banked inequalities appear only where they explain a measured
delta, at a directly-measured β.

## 8. Results ledger (gate outcomes as they land)

**2026-08-18 — G-L0a ρ ladder: COMPLETE, 48/48 PASS** (`results/RHO_LADDER_G0A.md` @9361d444;
medians of K=3, tokens=16,384, slab_rows=256, depth=4, bps=2):

| ρ_meas | compute | phased | fused | rccl | exposed ours / RCCL | h_wall |
|---:|---:|---:|---:|---:|---|---:|
| 0.631 | 1,405.6 | 3,634.6 | 3,622.5 | 2,908.7 | 2,229 / 1,488 | +0.55% |
| 0.941 | 2,178.3 | 4,493.6 | 4,462.3 | 3,769.6 | 2,315 / 1,586 | +1.50% |
| 1.786 | 4,421.7 | 6,879.0 | 6,864.1 | 5,938.0 | 2,476 / 1,553 | +0.40% |
| 2.383 | 6,557.2 | 9,331.7 | 9,398.8 | 8,405.2 | 2,751 / 1,925 | **−2.33%** |

**Verdict: fusion hides nothing at ANY reachable compute intensity** — h_wall ≤ 1.9% in every
repeat and negative at deployment-like ρ, with the arms DISJOINT at the top rung (all three
fused medians above all three phased medians, ≥33.7 µs). The four G25-1 nulls are
**ρ-INDEPENDENT**, not Amdahl-capped: the unreachable-bar rescue (mechanism review B1) is
retired. **G-L3's fused-CDAR readmission does not open on wall evidence**; only a ledger result
showing large hidden-but-interference-eaten overlap could reopen it. Q10's remaining question is
now sharply binary: transport never co-scheduled (scheduling) vs overlapped-but-taxed
(interference) — the phase ledger decides.

RCCL leads fused by 693–994 µs at every rung; β = ours/RCCL 1.43–1.60, median **1.48**, stable
across a 4.7× change in co-resident compute (corroborates the banked 1.506×; still
subtraction-derived — the standalone arms now need only a single ρ). **Track 1 order confirmed
by data: RCCL-hybrid floor + shared-expert filler over RCCL's ~1.5–1.9 ms exposure is the
build; fused CDAR stays benched.**

**New mechanism, unplanned (Q10c):** the SEQUENTIAL transport phase costs **+23.4% more after a
4.7× longer compute phase** (2,229 → 2,751 µs) at flat clocks (0.45% spread, ≤62 °C — not
thermal). A ρ-dependent component of the phased arm's tax; also why nominal ρ=3 measures only
2.38 (the denominator moves). The transport-only arm gains a second job: transport cold vs
post-compute. Candidate mechanisms to discriminate: cache/TLB state left by the compute phase,
fine-grained DVFS/power residency, MES/queue state — design a discriminating arm after the
ledger lands.

Ops rule (recorded): never quote `wall_us_max` on the rccl arm — RCCL's lazy channel setup makes
the first of 7 iterations 15.8× the median; medians are immune (min/median within 1.5%).

**2026-08-18 (late) — FILLER MICROCOSM: G-L2 GO** (`results/FILLER_MICROCOSM.md` @9fa8d3fa):
a bare 235 MB `ncclAllReduce` loses **0.00%** wall (worst repeat −0.38%; bar was <10%) while a
full-GPU MFMA kernel runs beside it in a separate process at **43.9–53.4%** of solo throughput —
**net +571–695 µs of the exposed AR absorbed free**, clearing G-L2's ≥25% bar conservatively.
Mechanism card: a 235 MB AR on this fabric is NOT CU-bound — the machine has room for a
co-resident compute kernel at zero collective cost; the filler design's founding assumption is
now measured, with its falsifier stated (dAR did not climb with filler intensity; 3 repeats, 2
configs). Caveat carried: two-process proxy, harsher sharing than the real design. **Design
consequence: evaluate the TWO-STREAM filler (Option A — shared-expert GEMM on a concurrent
stream over the AR window, no in-kernel fusion) before the in-kernel variant; it collapses
G-L2's integration cost.** G-L4 Phase B: GREEN, in flight.

**Same session — B4 CLOSED and two Q10 facts:** direct **β = 1.529** (transport 2,019.3 vs
stdrccl 1,320.9 µs, no subtraction; concurs with banked 1.506× and the ladder's 1.43–1.60 —
three methods agree). **I_co is NOT constant:** +209.7 µs at ρ=0.63 → +755.2 µs at ρ=2.38
(grows 3.6×; the "+335 µs constant tax" is retired — confirms Q10c). **transport_fused −
transport = +15.0 µs ⇒ the fused protocol's overhead is FREE; the fused arm's ~700 µs deficit
is entirely a co-residency/scheduling failure**, not protocol — the phase ledger's target is
now exact.

**Open blocker (operator action):** R7's harness patch `python3 ~/anatomy_g0/fix_ab_py_m20.py
--apply` (patcher committed as tools/fix_ab_py_m20.py; `--check` ran clean; patches the guard
at ab.py line 2006) was denied by the agent-session permission classifier 3×; needs the
operator to run it by hand or approve. R7 relaunch, wall parity, ledger-on cells, and the beta
arm queue behind node access.

**2026-08-19 — G-L2 CLOSED (negative), U2 resolved, G-L1 promoted to primary**
(`results/GL1_GL2_BOUNDARY_RACES.md`): five boundary races at the C=32 deployment chunk.
The shared-expert filler cannot beat production structure in any legal form: sharded+merged
chunked-AR loses at every N (chunk tax + 50% retention + exposed head); full replication is 8×
work > window; the only winning arm (−317 µs, filler at HIGH stream priority, 48% retention)
models reduction-free work. **U2 from production source: TP8+EP pays ONE all-reduce** (pre-add
branch; the shared-expert AR fires only on all2all-manager paths; fusion-shared-experts env is
False in the serving image) — no second collective to delete. Track 1 reroutes: (1) **G-L1
MoE-region GEMM floor is the primary lever** (fused region vs AITER sequence + glue at TP8
shapes — unmeasured); (2) serving-level next-chunk overlap (48%-retention physics banked, needs
G-L4 W1); (3) weight prefetch. Mechanism cards banked: in-process filler retention 38%→48% by
stream priority; chunk tax +153/+201/+668 at N=2/4/8; RCCL needs default channels (16ch = +48%
AR); merged schedules have no interior optimum. The +6–10% e2e projection is revised down
pending the G-L1 floor measurement.

**2026-08-19 (night) — G-L1 FLOOR, first measurement: our GEMM cores beat AITER's fp8 MoE by
~29% at the TP8 shapes** (driver `results/gl1_race_v0.py`, container `subvadla_m15pkt`, single
GPU, T=16,384, E_local=32, top-1-of-32 work-equivalent routing, fp8 blockscale both arms,
K=20 medians):
- OURS (packet-installed `n2_phase1+2`, fp8 blockscale, 2 launches): **2,044–2,092 µs**
  (stable across weight layouts — timing is layout-independent).
- AITER `fused_moe` fp8 blockscale: **2,854–2,992 µs** (shuffled weights change nothing).
- AITER bf16 path: 2,122 µs; Triton fallback bf16: 2,523; dense-GEMM roofline (same FLOPs):
  **1,313 µs**; vLLM glue kernels ≈150 µs.
Findings: (1) **AITER's fp8-blockscale MoE is SLOWER than its own bf16 path** at these shapes
(2,854 vs 2,122) — the dequant/scale machinery costs more than the precision saves; (2) our
fp8 pair beats BOTH AITER arms and sits 1.56× off the dense roofline, so a fused mega (deleting
the 32 MB a2q inter-phase HBM round-trip + launches) has visible headroom left; (3) prize size:
**+800–900 µs per layer-chunk** — comparable to the entire exposed-AR pool, workload-independent,
and it stacks with RCCL untouched.
CAVEATS (open): correctness of the driver harness is at cos≈0.96 / mean_rel≈0.27 vs a torch
reference after fixing the weight-shuffle layout (relmax 26 → 0.26) — one systematic detail
remains (suspects: shuffle variant (16,16) vs kernel expectation, gate/up half order, a2q
requant semantics); the AITER arm is a direct `aiter.fused_moe` call, not vLLM's dispatch — the
G-L0c profiled production step remains the ground-truth target; routing is uniform top-1-of-32
(work-equivalent), skewed routing untested. The n2 phase kernels are the PROVEN serving pair
(they run attested in production serving via the packet install), so the correctness gap is in
MY driver's data prep, not the kernels.

**2026-08-19 — DECODE TRACE ANALYZED (new evidence; decode was scoped out, now assessed):**
`results/DECODE_TRACE_ANALYSIS.md`. R1 TP8 graph-mode decode, 792 steps at bs∈{1,4,16}.
At bs=1: TPOT 10.34 ms, **62% of the token is all-reduce kernel time** (122 × 52 µs for 14 KB
payloads) — and the same AR kernel runs 13.7 µs at bs=4, so most of it is EP arrival-skew
absorbed in the AR + rendezvous latency, not transport. ~1,400 kernels/step at ~7 µs avg; 305
fp8-quant launches/step ≈ 1.4 ms of foldable overhead; MoE GEMMs are weight-streaming slivers
at the HBM floor. **Verdict: an M15 port attacks the wrong bottleneck at decode** — the decode
play is Region-4's different base (persistent layer kernel + in-kernel latency AR + epilogue-
folded quant), ceiling ≈1.8× TPOT at bs=1. Decisive next: θ-F13 (8-rank AR latency at
14/57/229 KB — aiter vs RCCL vs bare flag-AR), production AR-config sweep at bs=1 (fair
baseline), rank-skew measurement on a decode shape. All banked transport laws are 235 MB
bandwidth-regime and do NOT transfer to this regime.

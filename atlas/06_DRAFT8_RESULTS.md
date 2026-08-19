# Draft8 results ledger — every number presented, where it lives, and its current status

Slide-by-slide inventory of `draft8_final.pdf` ("HipKittens x IRIS Meeting 8", 2026-08-18).
For each presented claim: the number as shown, where the evidence lives in this repo, and its
status **as of 2026-08-19** under the repo's own validity rules
(`docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md`, `nightshift/REPORT.md`). Nothing
presented is dropped; where later audits changed a number's standing, both the as-presented
value and the current standing are recorded.

## Slide 2 — End-to-end vLLM results (the headline table)

| cell | as presented | status |
|---|---|---|
| **C=512, 2,048 QSL prompts, ISL 4,096, OSL 8 — DP8/EP8 both arms** | M15 **42,788** input tok/s vs AMD-recommended **39,474** ⇒ **+8.40%** (+3,314 tok/s); TTFT p50/p99 43.2/47.9 s vs 28.9/52.8 s | **The high-concurrency headline.** As presented it comes from a pre-M23 pair, and every pre-2026-08-18 e2e serving A/B carries two audited methodology defects (the exact-4096 activation seal engaged the megakernel on only ~2% of in-bucket steps, and uniform-decode ranks depressed the baseline) — see `01_KERNEL_CATALOG.md` §D1. After the M23 ragged-seal fix (coverage ~2% → 99% of in-bucket steps), the order-balanced replication gives **+2.70% mean / +3.61% geomean (n=2, ±18% position effect — DIRECTIONAL, below the ≥5-pair protocol floor)** (§D2, `nightshift/REPORT.md` M9). Direction of the win survives; the settled magnitude awaits the ≥5-pair campaign (`campaign_v5/ARMS.md`, specified, not yet run). |
| **C=32, 1,024 QSL prompts, ISL 4,096, OSL 8** | M15 DP8/EP8 **20,372** vs AMD TP8+EP **25,844** ⇒ **−21.17%**; TTFT p50/p99 2.445/4.364 s vs 1.339/5.682 s | Stands as an **arm-level, cross-topology** observation (two different W points — never quotable as a kernel delta). It is now the **Track 1 target**: win condition = our TP8+EP arm > 25,844 input tok/s with TTFT p50 ≤ production +10%, ≥5 order-balanced pairs (`UNDERSTANDING_PLAN.md` §4a; gates G-L0…G-L4). |

## Slide 3 — Low vs high concurrency prefill (fill/dummy structure)

As presented: C=32 → ~1,539/4,096 real rows (**37.6% fill**), **~56% dummy or
uniform-rescued steps**; C=512 → ~4,035/4,096 (**~98.5% fill**), **2% dummy**.

Status: stands, and was subsequently sharpened by the corpus audit — dummy steps are
**rank-synchronous** (bimodal on {0,8}, κ=1.000), corpus mean fill 0.69, and **62.4% of every
MoE row the m15 arm computed was padding** (`nightshift/CORPUS_FINDINGS.md`,
`aug18-prefill/BOTTLENECK_SIGNALS.md`, M24 designs in `01_KERNEL_CATALOG.md` §A12).

## Slide 4 — Updated understanding of communication CTAs

As presented: production **7.729 ms** = 1.000×; homogeneous megakernel **6.911 ms** = 0.894×;
dedicated communication CTAs C=64 **6.866 ms** = 0.888×; the ablations separated
losing-compute-CTAs from the communication role and found **~831 µs** of added
protocol/interference traffic.

Status: stands, with the sharper reading banked since: the +831.5 µs is an **elapsed-time
overhead from protocol + interference, not traffic** (measured with the pool's payload copy
deleted, exp_20), and dedication is **carrier-conditional** — it won in the staged-push carrier
(0.888×) and lost monotonically inside the producer-carried carrier (Q8a, open).
Home: `02_WHY_FAST_WHY_SLOW.md`, `UNDERSTANDING_PLAN.md` §2.

## Slides 5–6 — Producer-carried movement + the mechanism ablations

As presented (7,709.2 µs production basis): homogeneous **6,900.2** (0.8950×);
producer-carried remote accumulation **unbounded 7,110.8** (0.9226×, **+210.6 µs** — the
instructive regression); **depth-4 6,495.8** (0.8426×, **−615.0 µs**). Coalesced packed-bf16
remote atomics **52.8 GB/s** vs **54.9 GB/s** plain remote stores; the same atomic instruction
**4.1 GB/s** scattered.

Status: all stand; these are the campaign's most-replicated numbers. The −615.0 µs depth bound
is the largest single-knob effect and is now the extracted primitive
`include/*/ops/group/distributed/credit.cuh` (contract A3 in `03_ABSTRACTIONS.md`).
Home: `01_KERNEL_CATALOG.md` §A2, `02_WHY_FAST_WHY_SLOW.md` anchor-A waterfall.

## Slide 7 — Consumer ladder (M15, the best kernel)

As presented (7,712.0 µs production basis): homogeneous 6,908.8 (0.8958×); mode-12 depth-4
6,483.8 (0.8407×); **M15 16/24/28 initial consumers = 6,292.4 / 5,848.5 / 5,822.0 µs =
0.8165× / 0.7589× / 0.7544×**; 32 consumers **hung in one of two runs**. Mechanism: median
token not reducible until **91.7%** of GEMM-2 under natural order → NC-major producer order +
slab certificates.

Status: stands; re-anchored 2026-08-18 at **0.75187** (median of 5 order-balanced pairs,
`nightshift/REPORT.md` M4). The C=32 hang is Q7 with the R7 forensics protocol queued.
Home: `01_KERNEL_CATALOG.md` §A3 (full attribution), `02_WHY_FAST_WHY_SLOW.md`.

## Slides 8–9 — General findings; push vs pull

As presented: push attractive when payload is live in registers + destination known
(production MoRI dispatch); pull attractive when the receiver knows the landing layout; read
depth 1→16 = **2.33×** bandwidth change; matched experiment **push 15,723.8 vs pull
15,623.3 µs (0.9936×)** wall and **55.46 vs 59.79 GB/s (1.078×)** single-link — direction is
not the lever, flow control is; fine-grained readiness did not create overlap (signal the
smallest early-EXECUTABLE unit, not the smallest stored unit).

Status: all stand and are formalized as contracts **A2 (ownership handoff)** and **A1
(certified region stream)** in `03_ABSTRACTIONS.md` §3.

## Slide 10 — What this means for abstractions (the lifecycle)

As presented: producer computes in registers → delivers to collision-free owner slot →
depth-bounded injection → task order completes a large immutable region → one coarse
certificate → success-only acquire → quota-bounded downstream arithmetic in shadow → full-grid
drain.

Status: this lifecycle is now the spine of the six data-movement contracts
(`03_ABSTRACTIONS.md` §3) and of `UNDERSTANDING_PLAN.md` §5.6's contract/backend split.

## Slides 11–12 — Distributed HK progress (primitives + mechanism→abstraction tables)

As presented, with evidence:
- **PGL and peer translation** (`pgl.on(rank)`) — runtime-indexed peer arrays cost ~80 B/lane
  scratch; named peer bases matched manual under MFMA pressure.
- **Typed packet movement** — wide `dwordx4` issue took GEMM-RS to **376.7 µs vs 421.8 RCCL
  vs 485.3 RadeonFlow** on the comm-dominant shape.
- **Packed-BF16 peer accumulation** — 52.8 / 54.9 / 4.1 GB/s (layout is a schedule decision).
- **Directional release and acquire** — two drains in an MFMA loop cost ~9.5–10%; replacing a
  bidirectional system fence with pure acquire improved one path **~8.4%** (note: this 8.4% is
  a kernel-path improvement, unrelated to slide 2's +8.40% e2e figure).
- **Epoch publication + bounded waits** — moving completion/release off the compute-critical
  phase: **13,923 → 1,851 µs**; a 64× signal-count change otherwise ~free.
- **Counted fan-in and reservations** — fine-grained overuse measured at ~**926,000** protocol
  ops/rank/epoch.
- **Directed retirement credits** — ready ≠ reusable (epoch e+1 producer vs epoch e consumer).
- **CTA role helpers** — finish-order roles cost ~24 B/lane spill; static-tail reservation used
  instead.
- Mechanism table: literal vmcnt throttle (`n2_phase2_gm_mps.cpp:93`); NC-major decoding (no
  consumable row before **93.9%** completion under natural order); two-slab
  drain/barrier/publish/wait/acquire (`k0pf6gm_device_tile_m15.hip`); quota-bounded early
  combine (coupled interval **3,026.1 → 2,833.1 µs**, kernel −191.4 µs); runtime-target
  per-key fan-in (~**524,000** RMWs/rank/epoch) → `counted_arrive_dynamic_into`.

Status: all stand. The primitives now exist as the
`include/{cdna3,cdna4}/ops/group/distributed/*.cuh` headers (credit, slab, counter, lifetime,
order, roles, peer, packet, sync, completion), each carrying its measured provenance
(`04_EVIDENCE_INDEX.md`); GEMM-RS detail in `01_KERNEL_CATALOG.md` §E.

## Slide 13 — Potentially ungeneralizable areas

As presented: gfx950 producer-carried remote atomic accumulation vs **gfx942 remote atomics
~3× slower than stores** → portable form is source-private store towers + owner-local reduce;
depth = bound on outstanding remote ops per producer wave.

Status: stands; encoded as the gfx942/gfx950 **arch cards** in `credit.cuh` and as the
contract/backend split (the contract survives, the implementation flips) —
`03_ABSTRACTIONS.md`, `UNDERSTANDING_PLAN.md` §5.6.

## Slide 14 — Prefill ablations (the T sweep)

As presented: T=4,096 → 7,712.6 / 5,828.8 µs (**0.7557×**); T=2,048 → 3,857.4 / 3,411.8
(**0.8845×**); T=1,024 → 2,064.1 / 2,258.2 (**1.0940× — a loss**). Break-even ≈
1,600–1,800 tokens/rank.

Status: stands **with the mandatory labelling caveat** (`nightshift/REPORT.md` R1/M5): the
T=2,048/1,024 points carry the tgen body's source sha, and the un-clamped M15 chassis itself
fails the correctness gate at those T — any T≠4,096 number must be labelled tgen-body. The
inversion is Q11 (fixed cost 1,068.2 µs vs production's 181.3, unitemized; R11 queued).
Home: `01_KERNEL_CATALOG.md` §A10, `03_ABSTRACTIONS.md` §4.2(a).

## Slide 15 — Real serving is governed by the hottest expert rank

As presented: serving A/B (production vs M15): input throughput −0.70%, TTFT p50 +0.29%, TTFT
p99 +5.67%, TPOT p50 −2.51%, e2e p50 −1.06%, e2e p99 +6.95%; **~15% run-to-run production
variance**; route capture: **51.9% of all expert-slot traffic → experts 0–7** (all on rank 0
under contiguous placement); the hot rank holds the global slab certificate hostage.

Status: the A/B table is a pre-M23 pair (methodology-retired per §D1); the **51.9% skew
measurement stands** and is the founding measurement of the replication ladder — M18 static
replication reaches **0.3085×/0.2489×/0.2420×** production under the measured-skew replay and
M20 makes it practical at 672 MB (`01_KERNEL_CATALOG.md` §A7–A9,
`aug14/M18_REPLICATION_RESULTS.md`). The hot-rank-hostage mechanism is the recorded Q7-adjacent
design hazard. The 15%-variance observation became the order-balanced ≥5-pair protocol
(`nightshift/BENCHMARK_PROTOCOL.md`).

---

## The one-line summary

Every number in draft8 is preserved here with its provenance. Two classes changed standing
after the presentation: (1) all pre-M23 **end-to-end serving** pairs — including the +8.40%
headline and slide 15's table — are methodology-flagged by our own audit, with the post-fix
directional replication at +2.70%/+3.61% (n=2) pending the ≥5-pair campaign; (2) all
**kernel-region** numbers (the waterfall, the consumer ladder, the T-sweep, push/pull, the
primitive evidence) stand as presented, several now re-anchored with tighter protocols.

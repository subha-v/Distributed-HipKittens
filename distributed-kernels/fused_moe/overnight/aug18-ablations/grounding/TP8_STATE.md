# TP8+EP / low-concurrency regime — complete state of play

**Scope:** the workload cell where we currently LOSE (`C=32`, ISL 4096, OSL 8,
prefill-dominated, AMD-recommended TP8+EP). Everything below is either
(a) **MEASURED** with a cited artifact, (b) **DERIVED** arithmetic from measured
inputs (arithmetic shown so it can be checked), or (c) **SPECULATION / ASSUMED**,
explicitly labelled. Nothing is quoted without one of those three tags.

Prepared for the ablation campaign master doc (`../BRIEF.md`). This regime is
where the campaign's generalization claim is won or lost: the high-concurrency
DP8/EP8 win (+8.40%) is a *fill* story, and fill does not exist under TP8.

---

## 0. The one-paragraph state

At `C=32` production TP8+EP beats our DP8/EP8 M15 kernel by **+26.86% input
tok/s** (equivalently we are **−21.17%**), MEASURED. The mechanism is understood:
DP starves at low concurrency (37.6% fill, ~56% dummy steps) while TP8 does not
pay a dummy tax. Our answer is M25 — move to TP8+EP and hide its two per-layer
all-reduces (CDAR). The Amdahl pool is **~2.8 ms/layer of exposed ring-AR out of
~10 ms/layer** (~28%), DERIVED (see §1.4 — *not* yet profiler-confirmed). The
gate ladder has reached **G25-1 and is currently RED on performance, GREEN on
correctness**: fused CDAR **3,606 µs** vs production's GEMM→RCCL **2,909 µs** at
one boundary, with a **1,439 µs** compute floor. Two separable deficits: our
transport is **1.5× slower than RCCL**, and the fused schedule achieves
**~zero hiding**. Four schedule-side hypotheses were tried and falsified in-rig.
The named next step is a **device phase ledger** before any further schedule
mutation. This document adds two DERIVED findings the rig has not yet acted on:
the design's **−32% byte claim does not survive re-derivation (≈ −10%)**, and the
G25-1 rig is run at a **compute:transport ratio ~5× away from the deployment
point**, which alone Amdahl-caps hiding near zero.

---

## 1. Per-layer comm/compute budget in TP8+EP

### 1.1 Shapes (ASSUMED model card; source: `TP8_OVERLAP_ANALYSIS.md:10-13`)

DeepSeek-R1-0528: 61 layers (3 dense + 58 MoE), hidden 7168, MLA (kv_lora 512 +
64 rope), 256 routed experts top-8 + 1 shared, ~37B active, bf16 activations,
fp8 weights. bf16 boundary row = **14,336 B/token** — this constant is hard-coded
in the rig (`distributed-kernels/tp8_mega/m25_boundary_bench.hip:59`,
`kRowBytes = 14336`), so the microbench is at the real activation shape.

vLLM chunked prefill at AMD's recommended `max-num-batched-tokens 16384`
(ASSUMED; `TP8_OVERLAP_ANALYSIS.md:31-32`) ⇒ **one boundary tensor = 16,384 ×
14,336 B = 234,881,024 B = 235 MB**. This is exactly the rig's
`logical_AR_bytes=234881024` (`results/g25_0b_v1_results.txt`), i.e. the
microbench operates at the deployment message size.

### 1.2 Where the comm is, and what species it is

| | DP8/EP8 (M15, ours today) | TP8+EP (production at C≤128) |
|---|---|---|
| attention / dense / shared expert | replicated, **zero comm** | TP-sharded → rides the ARs |
| MoE | token **all2all** (dispatch + combine), sparse, MoE-layers only | dispatch **disappears** (hidden is replicated post-AR; row gather is LOCAL) |
| per-layer collectives | none | **2 all-reduces**, dense, on the critical path of **every** layer incl. the 3 dense ones |

Source: `TP8_OVERLAP_ANALYSIS.md:16-41`. The load-bearing structural sentence:
*"the comm primitive changes species: all2all (sparse, MoE-only) → two per-layer
all-reduces (dense, on the critical path of every layer)"* (`:41-42`).

### 1.3 Byte budget per layer — and a correction to the design doc

Aggregate fabric bytes per token per MoE layer (both boundaries):

| path | production (2× ring AR) | M25 CDAR as written in the design | M25 CDAR, re-derived here |
|---|---:|---:|---:|
| boundary 1 (o-proj out) | 196 KB | ERS#1 12.3 + MAG#1 98 = **110 KB** | ERS#1 **98** + MAG#1 **98** = **196 KB** |
| boundary 2 (MoE+shared out) | 196 KB | ERS#2 ~60 + MAG#2 98 = **158 KB** | ERS#2 ~**60** + MAG#2 **98** = **158 KB** |
| **total** | **392 KB** | **268 KB (−32%)** | **354 KB (−9.7%)** |

- Production column: ring AR aggregate = `2(N−1)·B = 2·7·14 KB = 196 KB/token`
  per AR (`M25_TP8_MEGA_DESIGN.md:98`). MEASURED-model, standard ring accounting.
- **AUDIT FINDING (DERIVED, mine).** `M25_TP8_MEGA_DESIGN.md:98` mixes
  conventions inside one row: `ERS#1 = 7/8 × 14 KB = 12.3 KB` is *per-(rank,
  token) expected egress*, while `MAG#1 = 7 × 14 KB = 98 KB` is *aggregate*.
  Put on one footing, ERS#1 aggregate = 7 non-owner ranks × 14 KB = **98 KB**.
  The **−32% byte claim therefore does not survive; the honest figure is ≈ −10%**.
- **Why this is the right answer and not a bug to fix:** ring all-reduce is
  bandwidth-optimal. ERS+MAG is a two-shot AR and moves *exactly* the same
  aggregate bytes as ring at boundary 1 (98 + 98 = 196 KB). The only structural
  byte win under TP8+EP is boundary 2's **sparse fan-in** — only the ~5.2
  expert-owning ranks per token contribute, not all 8
  (`M25_TP8_MEGA_DESIGN.md:99,104`; the 5.2942 distinct-rank figure is derived
  exactly in `FP8_WIRE_DESIGN.md:58-66`). ⇒ **CDAR's case is overlap, not bytes.**
  Any campaign claim of the form "we move a third fewer bytes" must be retracted
  or re-derived on the node.

Per-rank egress for one 235 MB boundary AR (DERIVED): two-shot RS phase
`7/8 · 235 = 205.6 MB`, AG phase `7 · 235/8 = 205.6 MB` ⇒ **411 MB per-rank
egress per boundary** — identical to ring's `2(N−1)/N · B`. Use this as the
denominator when converting rig walls to fabric efficiency (§2.3).

### 1.4 Time budget per layer — the Amdahl pool

**MEASURED (e2e, `results/serving/b0v5_pair_01_2_native_tuned_tp.json`,
`RUN_TAG=b0v5`, workload tuple below):** native_tuned_tp (untouched vLLM 0.25.1
image, AMD TP8+EP) = **25,843.96 input tok/s**.

Workload tuple, in full, per the reporting discipline: driver=closed,
concurrency=32, num_prompts=1024, input_len=4096, output_len=8, ignore_eos=true,
temperature=0.0, prompt_source=qsl (`mlperf-qsl-concat-v2-stride7`),
seed=320802, prompt-stream sha256 `7e4bd0dd…eefb`, n=**1 pair, unbalanced**
(calibration-grade — see §6 caveat).

**DERIVED step decomposition:**

- 16,384 tokens/step ÷ 25,844 tok/s = **634 ms/step** (ASSUMES the server actually
  batches to the full 16,384-token chunk; unverified — see §5, M6).
- 634 ms ÷ 61 layers = **10.4 ms/layer** — independently consistent with the
  `~10 ms/layer` figure banked in `nightshift/LOG.md` (12:10 PDT entry). Two
  independent routes agreeing is the strongest evidence we have for this number.
- **Exposed ring-AR ≈ 2.8 ms/layer ≈ 27–28% of the step** — banked in
  `nightshift/LOG.md` (12:10 PDT) and in commit `108def6e` ("Amdahl vs native
  TP8 closes at ~2.8ms/layer exposed AR"). Cross-check: the in-rig RCCL arm
  measures **AR ≈ 1.2–1.4 ms for 235 MB** (`results/G25_1_STATUS.md`), so
  2 ARs/layer = **2.4–2.8 ms/layer**. Consistent.
- ⇒ **compute + attention ≈ 7.2–7.6 ms/layer**, exposed collective ≈ 2.8 ms.
  **Compute:transport ≈ 2.6–3.2 : 1.** Remember this ratio — §2.4 shows the rig
  is not run at it.

**The prize pool.** 2.8 ms × 61 layers ≈ **171 ms of a 634 ms step ≈ 27%**.
Perfect hiding ⇒ `1/(1−0.27) = 1.37×` ⇒ TP8 arm would go 25.8k → **~35.4k
input tok/s** (DERIVED, upper bound, ignores co-residency tax). Against our
current M15 DP arm's 20,372 that is the whole gap and then some. This is the
number the entire M25 program is buying, and `TP8_OVERLAP_ANALYSIS.md:52-53`'s
original 25–35% estimate is now bracketed by two measurements rather than
asserted.

**Honesty flag (blocking for any public claim):** the 2.8 ms/layer is DERIVED
from throughput-vs-roofline plus the in-rig RCCL arm. It has **never been
confirmed by a profiler on the native TP8 arm**. `TP8_OVERLAP_ANALYSIS.md:141-143`
lists exactly this as the first thing B0 must calibrate; it was not done. Until
it is, "production exposes 28% of its step as all-reduce" is a *model*, not a
measurement.

### 1.5 The e2e picture — and the metric-map twist

Same b0v5 pair, both arms, MEASURED (JSON, `result` block):

| metric | M15 DP8/EP8 | native_tuned_tp | who wins |
|---|---:|---:|---|
| input tok/s | 20,371.55 | **25,843.96** | TP8 **+26.9%** |
| TTFT p50 (ms) | 2,444.8 | **1,338.6** | TP8 |
| **TTFT p99 (ms)** | **4,364.1** | 5,682.2 | **M15** |
| TPOT p50 (ms) | 557.3 | **480.2** | TP8 |
| **TPOT p99 (ms)** | **703.0** | 923.3 | **M15** |
| **e2e p99 (ms)** | **8,020.3** | 12,055.1 | **M15** |
| e2e p50 (ms) | 6,335.6 | **4,698.7** | TP8 |
| failed | 0 | 0 | — |

**This is a campaign-relevant, previously un-highlighted finding (DERIVED from
the banked JSONs).** The −21.17% headline is a *throughput and median-latency*
loss. On **every p99** — TTFT, TPOT, and end-to-end — the DP8/EP8 M15 arm is
**better**, by 23% / 24% / 33% respectively. TP8's max e2e is 13.4 s vs our 8.9 s.
Reading (SPECULATION, mechanism untested): TP8's single global batch couples all
32 in-flight requests into one queue, so head-of-line effects hit the tail;
DP8's eight independent engines decorrelate them (the same property that costs
it fill also buys it tail isolation). **Consequence for the metric map:** the
primary metric for this cell is not a scalar — a TTFT-p99-SLO deployment at
C=32 would already choose our arm today. That must be stated in the master doc
rather than conceding the cell wholesale.

---

## 2. Where the kernel work actually stands (G25-0 → G25-1)

### 2.1 Gate ladder status (`M25_TP8_MEGA_DESIGN.md:169-186`)

| gate | definition | status |
|---|---|---|
| G25-0a | GEMM-RS adapter beats RCCL | **PASS (pre-existing)** — 376.7 µs vs RCCL 421.8 µs vs RadeonFlow 485.3 µs on the comm-dominant shape, `docs/distributed/ARCHITECTURE.md:124`; caveats in `docs/distributed/SOURCE_AUDIT.md:104` (gfx950 world-8 EV=1/F6/AV1 rung D; EV=1 and REDV=1 never measured together) |
| G25-0b | transport microbench: K1 choice, MAG co-residency, decode-size AR | **PARTIAL PASS** — K1/K3/K4 answered (§2.2); MAG-under-MFMA and decode-size RCCL comparison **NOT RUN** |
| G25-1 | single-boundary CDAR vs GEMM+RCCL; PASS = ≥90% of AR hidden | **RED on perf, GREEN on correctness** (§2.3) |
| G25-2 | two-layer persistent segment + rolling planner + filler; phase ledger proves no exposed collective | not started |
| G25-3 | vLLM integration behind default-0 macros + campaign v5.1 arms | not started |

### 2.2 G25-0b: what the transport microbench established (all MEASURED)

Artifacts: `distributed-kernels/tp8_mega/results/g25_0b_results.txt` (v0),
`g25_0b_v1_results.txt` (v1 depth×bps), `g25_0b_slab_sweep.txt` (granularity).
Every configuration printed `CDAR_BENCH_PASS`, `verify_errors=0`,
`gather_timeouts=0`, `mag_timeouts=0`, exact bf16 sum (rank+1 summed = 36).
All at tokens=16384, 235 MB logical, 7 iters, median wall.

**K1 (carrier/transport) — DECIDED: store-towers.**

| transport | best effective AR | note |
|---|---:|---|
| `atomic_accumulate` | **67.1 GB/s** (depth 4, bps 2, 512-row slabs) | **caps at ~67**, flat across depth 4/8/16/32 and bps 2/4 — diagnosed as the **dword issue rate** (`nightshift/LOG.md` 12:10 PDT) |
| `store_towers` | **93.6 GB/s** @512-row, **117.2 GB/s** @256-row | + parallel vectorized owner reduce; 4.2× the v0 towers number |

> **This REVERSES the design doc.** `M25_TP8_MEGA_DESIGN.md:115` selected
> `throttled_accumulate_bf162<4>` for ERS#1 on the strength of the gfx950 arch
> card (coalesced atomics 52.8 vs stores 54.9 GB/s — near-parity). At the real
> 7168-wide row and 235 MB message the atomic path is **1.75× worse**, not at
> parity. **Law update:** the atomic-vs-store decision is not settled by the
> small-message arch card; it is settled by *issue rate at the deployment
> message size*. The gfx942 card already said "stores + local reduce"
> (`docs/distributed/OVERLAP_ABSTRACTIONS.md:146`); gfx950 now agrees at this
> shape. `m25_cdar.cuh:74-75` keeps both as a compile-time knob, so this is a
> manifest row, not a rewrite.

**K3 (readiness / slab granularity) — non-monotone optimum at 256 rows**
(`g25_0b_slab_sweep.txt`, towers, depth 4, bps 2, 16,384 tokens):

| slab_rows | slabs | wall µs (median) | effective GB/s |
|---:|---:|---:|---:|
| 128 | 128 | 2,025.0 | 116.0 |
| **256** | **64** | **2,004.2** | **117.2** |
| 512 | 32 | 2,513.9 | 93.4 |
| 1024 | 16 | 4,243.2 | 55.4 |

⇒ **Law (MEASURED, one shape):** readiness granularity has a *bandwidth* cost
curve with an interior optimum, not just a *latency* one. Finer than 256 rows
buys nothing (protocol cost catches up); coarser than 256 collapses (−53% by
1024 rows). 256 rows × 14,336 B = **3.67 MB/slab**. This is the TP8 analogue of
the DP-side "signal the smallest early-EXECUTABLE unit" law, with a measured
floor attached.

**K4 (injection depth) — FLAT in pure transport, {4, 8, 16, 32} all identical**
within noise (`g25_0b_v1_results.txt`: atomic 41.0/41.0/40.9/40.9 at bps=1 and
67.1/67.1/67.0/66.8 at bps=2; towers 93.6/93.4/93.3 at depth 4/16/32).

> **This is the single most important law refinement of the whole gate.** The
> depth-4 vmcnt throttle is worth **−615 µs** in M15 (exp_24, cited in
> `FP8_WIRE_DESIGN.md:44`) and depth 16/32 is a *cliff* there. In pure transport
> it does nothing at all. ⇒ **Injection depth is congestion control against
> *co-resident compute*, not against the fabric.** K4 is a co-residency knob,
> and its setting is workload-dependent — exactly the shape of law the campaign
> brief asks for ("if you change X, the schedule should change by Y because
> mediating quantity Q crossed θ"; here Q = *is the transport sharing the device
> with an MFMA body*, θ = binary). Committed as
> `495b5fc6` / `108def6e`.

**Blocks-per-slab (bps) — a real parallelism knob:** 1→2 = **+64%** (41.0 →
67.1 GB/s atomic); 2→4 = flat. One block per slab leaves the grid half-idle.

**Diagnosed v0 failure (MEASURED → fixed):** v0's 41.1 (atomic) / 22.4 (towers)
GB/s were caused by half-idle grid + serialized slabs + **scalar single-block
owner reduce**; parallelising and vectorising the tower reduce gave the 4.2×
(`nightshift/LOG.md` 12:10 PDT). Filed as a falsification of "the owner reduce
is free" (§3, F0).

### 2.3 G25-1: the boundary rig — the current, honest, red result

Rig: `distributed-kernels/tp8_mega/m25_boundary_bench.hip` (622 lines, committed
`3f7f0f3a`). Four arms at one TP8 layer boundary, 16K tokens, 256-row slabs,
`k_inner=2048`, `bps=2`, 256 blocks, real `v_mfma_f32_16x16x32_bf16` body:

| arm | wall | DERIVED: collective share |
|---|---:|---:|
| compute floor (two MFMA bursts) | **1,439 µs** | — |
| **GEMM → RCCL → GEMM (production's schedule)** | **2,909 µs** | 1,470 µs exposed |
| GEMM → phased CDAR → GEMM (unfused control) | 3,653 µs | 2,214 µs exposed |
| **fused CDAR (best schedule attempt)** | **3,606 µs** | 2,167 µs, ~none hidden |

Source: `distributed-kernels/tp8_mega/results/G25_1_STATUS.md` and commit
`3f7f0f3a`. **Correctness PASS everywhere**: exact bf16 sums, zero timeouts,
every arm, every configuration.

**Deficit 1 — transport (MEASURED).** `2,214 / 1,470 = 1.51×`; the status doc
states it as "CDAR standalone transport ~1.5× slower than RCCL's AR (125 vs
~192 GB/s effective)".

**Deficit 2 — hiding (MEASURED).** `phased − fused = 3,653 − 3,606 = 47 µs` =
**2.1% of the collective hidden**, against a G25-1 PASS bar of **≥90%**
(`M25_TP8_MEGA_DESIGN.md:181`). "Fused tracks phased within noise in every
variant tried."

**Fabric-efficiency framing (DERIVED — this is the good news buried in the bad
result).** Using §1.3's 411 MB per-rank egress per boundary AR:

| arm | wall | per-rank egress rate | vs the (unreproduced) 349–355 GB/s egress ceiling |
|---|---:|---:|---:|
| CDAR pure transport, best (256-row) | 2,004 µs | **205 GB/s** | 58% |
| CDAR in the boundary rig | 2,214 µs | 186 GB/s | 53% |
| RCCL in the boundary rig | 1,470 µs | **280 GB/s** | **79%** |

⇒ RCCL is at ~80% of the node's egress ceiling; we are at ~58%. **The 1.5× gap
is almost exactly the headroom to the ceiling** — it is an *efficiency* gap in
our own transport, not a topology or byte-count disadvantage (§1.3 already
showed we move the same bytes). SPECULATION-free conclusion: closing it is an
engineering problem with a known target, not a redesign. **CAVEAT:** the
349–355 GB/s / 54.9 GB/s-per-link card entry is flagged *"unreproduced; treat as
unknown, measure first"* (`docs/distributed/OVERLAP_ABSTRACTIONS.md:147`) — the
denominator must be re-measured before this framing is used in a claim.

**Protocol overhead (DERIVED):** pure transport at 125 GB/s effective would put
235 MB at 1,879 µs; `1,439 + 1,879 = 3,318 µs` vs measured phased 3,653 µs ⇒
**~335 µs of protocol/serialisation** appears when the CDAR runs inside the same
kernel as the MFMA body. That is a co-residency tax on the *phased* arm, before
any fusion is attempted, and nobody has attributed it yet.

### 2.4 The rig's operating point is wrong (DERIVED — flagged here first)

At `k_inner=2048`: compute floor for **two** bursts = 1,439 µs ⇒ ~720 µs per
burst. CDAR transport for **one** boundary ≈ 1,880–2,214 µs. **The collective is
~3× longer than the compute it is asked to hide under.**

Amdahl inside the rig: best conceivable fused wall = `max(compute, transport)`
≈ **1,880 µs**, i.e. even *perfect* hiding leaves the transport fully exposed
and only removes the compute. The measured deployment ratio (§1.4) is the
opposite: **compute ~7.2 ms vs collective ~2.8 ms per layer, ≈ 2.6–3.2 : 1 in
favour of compute**. The design doc says the same thing in its own words —
"CDAR standalone ~5 ms/boundary-pair but chunk-signaled ⇒ hides under ~7 ms
GEMM" (`nightshift/LOG.md` 12:10 PDT).

⇒ **The G25-1 rig is run ~5× short on compute relative to the workload it
models.** The four falsified hypotheses (§3) were all tested at an operating
point where the maximum available win from hiding is small and the signal is
buried. This does **not** invalidate the falsifications (they were all null
results, and a null at a favourable operating point would be stronger) — but it
does mean **"no hiding" has not yet been tested where hiding is possible.**
This is now the cheapest open experiment (§5, M2), and it costs one argv change.

---

## 3. Falsified-hypothesis ledger

### 3.1 Falsified inside G25-0b / G25-1 (TP8-specific, MEASURED)

| # | hypothesis | what was built | result | source |
|---|---|---|---|---|
| **F0** | the owner-side tower reduce is cheap / can be scalar and single-block | v0 towers path | **FALSIFIED** — 22.4 GB/s; parallel + vectorised reduce with dynamic reduce-arrival certification gave **4.2×** (→93.6) | `495b5fc6`, `108def6e` |
| **F1** | per-item drain serialisation is the blocker; deferring the drain and arriving one item late lets stores ride under the next MFMA burst | rig v2 (software-pipelined commit, `m25_boundary_bench.hip:203-215`) | **FALSIFIED — no change** | `G25_1_STATUS.md`, `3f7f0f3a` |
| **F2** | the load→store re-copy chain is the blocker; write once from registers (the M15 epilogue's real profile) | rig v3 (`m25_boundary_bench.hip:221`) | **FALSIFIED — no change** | same |
| **F3** | a specialised consuming pool is the blocker; let duty blocks skip production and gather in the producers' shadow (256 blocks, producer fallthrough) | rig v4 | **FALSIFIED — no change** | same |
| **F4** | item granularity is the blocker; a fragments-per-slab runtime knob spreads commits across the compute span (bps=16) | rig v5 | **FALSIFIED, and WORSE** — +180 µs of pure protocol, still zero hiding | same |
| **F5** | `atomic_accumulate` is the right K1 for ERS on gfx950 (design's choice at `:115`) | G25-0b transport matrix | **FALSIFIED at deployment size** — atomics cap ~67 GB/s (dword issue rate) vs towers 117 | §2.2 |
| **F6** | the depth-4 vmcnt law is a fabric/injection law and transfers to pure transport | G25-0b depth sweep {4,8,16,32} × bps {1,2,4} | **FALSIFIED — depth is flat.** Law re-scoped to co-residency only | §2.2 |
| **F7** | coarse slabs amortise protocol and go faster | G25-0b slab sweep | **FALSIFIED** — 1024-row slabs are **2.1× slower** than 256-row | §2.2 |

**Meta-observation on F1–F4 (DERIVED).** All four were *schedule-side guesses*
about producer-side mechanics, and all four returned exactly nothing. A
parameter that four independent producer-side interventions cannot move is
almost certainly not on the producer side. That is precisely the reasoning
recorded in `G25_1_STATUS.md` ("one instrumented run replaces the next four
guesses") and it is why the phase ledger, not a fifth mutation, is next.

### 3.2 Carried-over falsifications from the DP/EP work that constrain M25

These are already-measured laws (BRIEF digest + `M25_TP8_MEGA_DESIGN.md:121-126`)
that any TP8 schedule must not re-violate:

- **Dedicated communication CTAs lose**: C=64 comm-CTA arm 6,866 µs (0.888×),
  with **+831 µs of added traffic attributable to the comm-CTA role itself**.
  ⇒ M25 must stay producer/epilogue-carried; F3's "no specialised pool" arm is
  consistent with this and is *not* evidence against it.
- **Carrier pools lose (+340 µs); consuming pools win (−93/−444 µs).** MAG is
  built as a *consuming* pool by construction (`m25_cdar.cuh:14-18`).
- **Unbounded producer-carried remote accumulation is worse than homogeneous**
  (7,110.8 vs 6,908.8 µs); the depth bound is what rescues it (−615 µs). Under
  TP8 F6 says the bound only matters co-resident — so the *bound* survives but
  its *justification* changed.
- **Layout is a schedule decision**: coalesced packed-bf16 atomics 52.8 GB/s vs
  scattered 4.1 GB/s (**13×**). The ISA issue-run gate is mandatory, not optional
  (`M25_TP8_MEGA_DESIGN.md:150-153`, exp_38 lesson: a bound that can silently
  die is not a primitive).
- **Sync exposure is deadly / per-row readiness is probability-zero**
  (`TP8_OVERLAP_ANALYSIS.md:122-126`).
- **gfx942 flips K1**: remote atomics ~3× slower than stores there
  (15 vs 44–46 GiB/s, `OVERLAP_ABSTRACTIONS.md:146`). F5 means gfx950 and gfx942
  now agree at deployment message size — a *simplification* for the PR.

### 3.3 What DIES under TP8 (not falsified — structurally absent)

`TP8_OVERLAP_ANALYSIS.md:76-87`: MORI all2all machinery (no dispatch), M23 ragged
seal (a per-rank DP batch concept), and **M24 dummy-skip / the entire fill
lever** (no DP lockstep ⇒ no `execute_dummy_batch`, no 56%-dummy steps).
**This is the single most important framing point for the campaign:** our
high-concurrency +8.40% win is mediated by *fill*, and fill is a DP-topology
quantity. It does not generalise to TP8 even in principle. The TP8 cell must be
won by a different mediating quantity — **exposed collective fraction** — which
is why this document exists.

---

## 4. How TP8+EP changes the READINESS structure vs DP/EP

This is the section that generalises: the primitive layer is the same, the
*information structure* is not.

### 4.1 Who knows destinations, and when

| | DP8/EP8 (M15) | TP8+EP (M25) |
|---|---|---|
| what determines a destination | **data-dependent routing** — token's top-8 experts | **structural** — `owner_of(slab) = f(slab)`, a fixed stripe (`m25_cdar.cuh` `geometry::owner_of`, local ordinal `slab / World`) |
| when is it known | only **after** the local gate runs | **before the producing GEMM starts** |
| who knows it | only the **sender** (each rank routes its own private batch) | **every rank**, redundantly, with zero communication — routing is recomputed locally from the replicated post-AR hidden (`TP8_OVERLAP_ANALYSIS.md:29-34`) |
| consequence for order (K2) | producer order must be *restructured* to complete an independent slab early (the 91.7% finding) | `owner_major_staggered(rank)` can be applied statically ⇒ ring emerges, **every link busy from slab 1**, hotspot spread by construction (`M25_TP8_MEGA_DESIGN.md:115`) |

**The dividend, stated as a law (DERIVED):** *replication buys schedule
determinism.* Under DP the schedule is discovered at runtime from routing; under
TP8+EP the ERS schedule is knowable at plan time. That is why the design can
promise a ring without building one.

### 4.2 Fan-in: from unknown-size and global, to known-size and per-slab

| | DP8/EP8 | TP8+EP boundary 1 | TP8+EP boundary 2 |
|---|---|---|---|
| fan-in shape | per-expert, **8 sources, unknown row counts** | **exactly 8, static, every slab, every epoch** | **dynamic**: #expert-ranks serving that slab + 1 shared (~5.2 expected, `FP8_WIRE_DESIGN.md:60`) |
| mechanism | `chunk_ready` + counted arrival, ~524k open-coded RMWs/rank/epoch in the MoE adapter (`M25_TP8_MEGA_DESIGN.md:37-38`) | `counted_arrive_dynamic_release_into` level 1 + 8 per-source epoch words level 2 | same, but the **empty-contribution rule binds** |
| skew exposure | **global**: hot rank holds the slab certificate hostage; measured 51.9% of expert-slot traffic to experts 0–7, all on rank 0 | **none** — every rank owes every slab the same bytes | expert skew returns, but *localised* |

**Certificates live in two levels, deliberately** (`m25_cdar.cuh:20-31`):

1. **Level 1 (local, agent scope).** Several producer CTAs on one rank contribute
   fragments to the same slab; they count in with
   `counted_arrive_dynamic_release_into` (runtime fragment count — MoE fan-in
   varies per slab). **The last local arriver owns the rank-level publication.**
2. **Level 2 (cross-rank, system scope).** That publication is **one store** of
   an epoch word to the owner's `src_done[slab][source]` cell — the proven M15
   `rows_done` pattern. **No cross-rank RMW counters.** Rationale, verbatim from
   the header: per-source words *"keep remote traffic store-class, keep contention
   at zero, and give the owner a self-describing view (which source is late) for
   free."* The owner certifies on observing all `World` words, then
   `mag_ready[slab]`; consumers bounded-poll.

**The empty-contribution rule (the M24 lesson, load-bearing).** A rank that
contributes *nothing* to slab `s` this epoch must still publish its `src_done`
word — `ers_publish_empty`, done by the arming agent at plan time
(`m25_cdar.cuh:33-36`, `:240-247`). *"A certificate that some peer counts on must
exist on every path."* Under DP this case is rare; under TP8 boundary 2 it is
**routine** (a rank whose 32 experts serve none of a 256-token slab), so the rule
moves from edge case to hot path.

**Structural comparison (DERIVED, the campaign-worthy sentence):** DP/EP places
the certificate at the *destination expert*, with fan-in from 8 unknown-size,
skew-correlated sources ⇒ a **global rendezvous whose latency is set by the
hottest rank**. TP8+EP places it at the *token-slab owner*, with fan-in from 8
structurally-known sources ⇒ **64 independent per-slab rendezvous whose lateness
is individually attributable**. Same primitives (`slab.cuh`, `counter.cuh`,
`credit.cuh`, `order.cuh`); different observability and different failure mode.
Boundary 1 is skew-free; boundary 2 re-admits skew but bounds its blast radius
to one slab and names the late source.

### 4.3 What is deleted, what is added

- **Deleted:** the dispatch all2all entirely; the 8× replicated RMSNorm
  (owner-shard norms only, `M25_TP8_MEGA_DESIGN.md:68-70`); the dummy tax.
- **Added:** two dense collectives *per layer* including the 3 dense layers, i.e.
  **122 collectives per token-step** where DP had ~58 sparse all2alls.
- **Memory:** attention/dense/shared weights no longer replicated (~15 GB/rank
  freed, ESTIMATE, `TP8_OVERLAP_ANALYSIS.md:71-73`); against that, the 2-deep
  accumulator ring costs `2 × 2 × 16384 × 7168 × 2 B ≈ 940 MB` at 16K tokens
  (`M25_TP8_MEGA_DESIGN.md:137-139`), plus towers multiply the owner accumulator
  by `World` (`m25_cdar.cuh:356-358`) — **an unpriced consequence of the F5
  towers decision**, ≈ +205 MB/rank/boundary at this shape (DERIVED). Not yet
  budgeted anywhere.

---

## 5. Decisive next measurements, in order

**M1 — DEVICE PHASE LEDGER (the named next step; blocks all schedule work).**
Instrument `m25_boundary_bench.hip` with the `[MPS TS]` discipline: per
block-class timestamps for **produce span, first/last commit, first/last certify,
first/last pull, burst2 span**. Discriminates the four candidate walls named in
`G25_1_STATUS.md`: (a) consumer-side remote-**read** bandwidth in the MAG pull,
(b) certify latency chain (gather polls parked behind produce), (c) reduce HBM
bandwidth colliding with burst2 reads, (d) something else. *One instrumented run
replaces the next four guesses.* **Discipline caveat:** `[MPS TS]` values are
running device maxima that are never reset — the aug11 lesson
(`aug11/PLOTS.md:74`, `aug11/exp_33_attribution/result.md:237`) is that only the
final soak epoch is readable. Bake that into the harness or the ledger lies.

**M2 — COMPUTE:TRANSPORT RATIO SWEEP (cheapest, highest information, mine).**
Sweep `k_inner` until the compute floor reaches ~3× the collective (the measured
deployment ratio, §1.4) and re-run {compute, phased, fused, rccl}. At today's
`k_inner=2048` the collective is ~3× *longer* than the compute, so the maximum
extractable hiding is Amdahl-capped near zero and F1–F4 were tested blind
(§2.4). Costs one argv value. **Pre-register the falsifier:** if hiding is still
<10% at the deployment ratio, the "hide it in the epilogue" thesis is in
serious trouble and the fallback is M8/H-T2.

**M3 — K4 under co-residency.** Depth is flat in pure transport (F6) and the
fused regime "has not yet reached the regime where it would matter"
(`G25_1_STATUS.md`). Re-sweep depth {4,8,16,32} in the fused arm *after* M2 puts
the rig at the right ratio. This either re-establishes the −615 µs law under a
new topology or retires it as DP-specific — either outcome is a campaign law.

**M4 — K3 slab granularity in the fused arm.** 256 rows won in *pure* transport;
the fused rig also runs 256, so the co-residency optimum is untested and the
optimum is known to be non-monotone (F7).

**M5 — RCCL reference at decode message sizes (2–64 rows).** Designed as
G25-0b(iii) (`M25_TP8_MEGA_DESIGN.md:174-176`), **never run**. `rccl-tests` is
absent from the node (`g25_0b_slab_sweep.txt`: "rccl-tests not found on node;
RCCL reference skipped") — but the boundary rig already links `-lrccl` and calls
`ncclAllReduce` directly, so the reference is a rig argv away. This is the
**entire go/no-go for the decode cells**, which are latency-bound and where
bytes are irrelevant; without it we have no basis for any claim below ~2K tokens.
Note the one decode-ish CDAR point we do have: 256 tokens / 64-row slabs =
801 µs (atomic) / 1,017 µs (towers) for 3.67 MB — 4.6 / 3.6 GB/s effective,
i.e. **latency-dominated, and towers *lose* at small size** (the reduce step is
no longer amortised). K1 is therefore *message-size-dependent*, not a global
decision — a second manifest axis.

**M6 — PROFILER CONFIRMATION OF THE 2.8 ms/layer POOL.** Torch-profile the
native TP8 arm and read the real collective fraction and the real chunk size
(`TP8_OVERLAP_ANALYSIS.md:141-143`, never executed). Blocks every public
statement of the Amdahl pool, and settles the 16,384-token chunk assumption
underneath §1.4's 634 ms/step.

**M7 — CONFIRM vLLM 0.25.1's TP8+EP MoE path actually rides AR#2 as assumed**
(`TP8_OVERLAP_ANALYSIS.md:146-148`). If the shipped path instead does an
all-gather or a separate shared-expert reduce, the byte model of §1.3 and the
"we merge what production merges" parity claim both change.

**M8 — OWNER-PUSH MULTICAST vs MAG PULL** — conditional, fire only if M1 blames
(a). Named as the secondary avenue in `G25_1_STATUS.md`: turn the consumer pull
into an owner-side store-class multicast during the reduce drain (write bandwidth
instead of remote-read bandwidth). Note this collides with the carrier-pool
falsification unless the push rides the *owner's own reduce epilogue*, which is
the producer-carried form and therefore legal.

**M9 — RE-DERIVE THE BYTE MODEL ON THE NODE** and correct
`M25_TP8_MEGA_DESIGN.md:94-106` (§1.3). Cheap, doc-only, and it removes a claim
we would not survive being asked about.

---

## 6. Open hypotheses, ranked

Ranking rule: expected Amdahl value × probability × inverse cost. Every entry
names its killer measurement.

### Track A — close the 1.5× transport gap (worth ~740 µs/boundary in-rig)

| rank | hypothesis | why plausible | killer |
|---|---|---|---|
| **A1** | The wall is **consumer-side remote-READ bandwidth in MAG**, and pull is the wrong direction at this message size | push/pull tie was measured at **64 KiB** (0.9936×) and only 1.078× at 64 MiB single-link — neither is a 235 MB many-to-one/one-to-many case; the push-vs-pull law says *pull when consumer-defined & one-to-many*, which MAG is, but the law was never calibrated at this size | M1, then M8 |
| **A2** | Owner **tower reduce HBM traffic collides with the consuming GEMM's reads** | towers make the owner read `World × slab_bytes` and write `slab_bytes` per slab; at 235 MB that is ~205 MB extra read per boundary per rank (DERIVED), on the same HBM the next GEMM burst is streaming | M1 (case c) |
| **A3** | **Certify latency chain**: gather polls are parked behind produce work in the same CTA, so certification lags the last commit by a produce-quantum | the two-level fan-in publishes only on the *last local arriver* (`m25_cdar.cuh:215-231`), so one straggling fragment delays the whole slab's certificate | M1 (case b) |
| **A4** | We are simply **under-issuing the wire**: 58% of egress ceiling vs RCCL's 79% (§2.3); wider packets / more CTAs per link | the gfx942 GEMM-RS ablation found XGMI egress at ~127 GB/s vs ~448 GB/s and attributed it to "16-byte scattered peer stores" (`gemm_rs/overnight/RESULTS.md`) — same failure class, different arch | CTA-per-link and packet-width sweep; re-measure the egress ceiling (it is flagged unreproduced) |
| **A5** | **fp8 on the wire** for ERS halves the boundary bytes | `FP8_WIRE_DESIGN.md:118-140`: fp8 e4m3 + per-128 fp32 scales = **1.939×** fewer bytes, with the format factor cross-checked against MORI's 366→642 GB/s combine lever | numerics first — this is an **activation** AR, not a combine partial; SPECULATION that the accuracy envelope tolerates it. **Note the asymmetry that makes it *more* attractive here than in DP:** the FP8 doc's whole risk is that the staged path "converts hidden wire time into exposed wire time" (`:857-862`). Under TP8 the wire time is **already exposed** in production, so the A1/A2 ledger's sign flips in our favour |
| **A6** | Two-shot is topologically wrong for 8 ranks and we should emit a true ring | §1.3 shows two-shot and ring move identical aggregate bytes, and `OVERLAP_ABSTRACTIONS.md:150-158` says the ring *is* the emergent schedule under owner-major-staggered — so there is likely nothing here | low priority; the ceiling arithmetic (A4) already explains the gap |

### Track B — get actual hiding (worth up to 2.2 ms/boundary in-rig; ~171 ms/step at deployment)

| rank | hypothesis | why plausible | killer |
|---|---|---|---|
| **B1** | **The rig is at the wrong operating point** and hiding was never testable (§2.4) | compute is ~3× *shorter* than the collective in the rig; the deployment ratio is ~3× the other way | **M2** — one argv |
| **B2** | Hiding requires **spare vmem issue capacity**, and an occupancy-1 256-VGPR MFMA body at `k_inner=2048` has none | the co-residency tax already visible as +335 µs of protocol in the *phased* arm (§2.3) suggests the two workloads contend rather than interleave; this is a *resource-class* argument, not a schedule one, and would explain why F1–F4 all returned nothing | M1's produce-span vs commit-span overlap, plus an arm that varies MFMA density independently of fragment bytes |
| **B3** | **Consumer/producer role ratio is unswept in the fused arm.** The design mandates choosing `C` reserved consuming CTAs by response curve `{8,16,24,32} × quota` — "never a default" (`M25_TP8_MEGA_DESIGN.md:124-128`); F3 tested only the `C=0` endpoint | the DP-side M15 ladder's whole gain came from exactly this sweep (16→24→28 consumers, 6,292→5,848→5,822 µs) | response-curve sweep after M2 |
| **B4** | **Shared-expert filler** is the right hiding vehicle — routing-independent tiles as the work queue's fallback whenever routed tiles starve | zero new machinery, "it is just queue priority" (`M25_TP8_MEGA_DESIGN.md:79-83`); under TP8 it is *simpler* than under DP because the shared expert's output joins the same ERS accumulator, deleting its separate AR (`TP8_OVERLAP_ANALYSIS.md:83`) | **bounded by construction**: the shared expert is ~1/8 of routed FLOPs (`SHARED_EXPERT_FILLER_DESIGN.md:640-651`) ⇒ it can fill at most ~1/8 of a layer's compute span. Real but capped; must not be sold as the answer to a 28% pool |
| **B5** | **Rolling planner / counter-dataflow across AR#1** — gate + gather + early expert tiles start per certified slab, so routing overlaps MAG#1 landing where production serialises gate→sort→group kernels | ports `chunk_ready` verbatim; ceiling is "the other ~half of exposed comm, minus the unoverlappable head chunk" (`TP8_OVERLAP_ANALYSIS.md:105-109`) | G25-2; needs B1/B2 resolved first or it inherits their null |
| **B6** | **Cross-layer pipelining / next-layer prologue overlap** — the only place the MAG#2 tail can hide | v1's **structural bubble**: attention is host-launched, so the last slabs' pull cannot hide under it; sized at ~`1/n_slabs` of MAG#2 + launch gap (`M25_TP8_MEGA_DESIGN.md:206-213`). At 64 slabs that is ~1.6% of MAG#2 + gap — small, *if* the launch gap is small | G25-2 phase ledger measures the tail before choosing a door: (a) MLA attention in-kernel (largest lift), (b) hipGraph chunked attention with per-slab events (medium), (c) persistent attention co-kernel sharing the certificate space |
| **B7** | **Sequence-parallel RS+AG** instead of AR (norms on shards) — halves norm-path comm, gives natural chunk boundaries | but token-sharded hidden complicates the local row-gather that is the whole EP-inside-TP dividend | explicitly **secondary**; evaluate only if A/B leave exposed norm time (`TP8_OVERLAP_ANALYSIS.md:117-120`) |
| **B8** | **Decode is a different mechanism entirely** — one-shot/LL AR, small-M tiles, possibly fused AR+RMSNorm | chunking collapses below ~2K tokens; our own 256-token point is latency-dominated and reverses the K1 decision (§5, M5) | M5. Do **not** conflate with the prefill bandwidth track — different mechanisms, different cells (`TP8_OVERLAP_ANALYSIS.md:110-116`) |

---

## 7. Known vs assumed — the audit table

| claim | status | evidence / caveat |
|---|---|---|
| TP8+EP beats M15 DP by +26.9% input tok/s at C=32 | **MEASURED** | `results/serving/b0v5_*.json`; **n=1 pair, UNBALANCED, calibration-grade** — the protocol's ±~15% e2e run variance and the ±18% position effect both exceed nothing here (the delta is 27%), but it is still not an order-balanced result |
| M15 wins every p99 (TTFT/TPOT/e2e) at C=32 | **MEASURED** (same caveat) | §1.5, from the banked JSONs; **not previously highlighted anywhere** |
| Exposed AR ≈ 2.8 ms/layer ≈ 28% of the TP8 step | **DERIVED, two independent routes agree** | LOG 12:10 PDT + in-rig RCCL arm; **no profiler confirmation** (M6) |
| 16,384-token prefill chunk | **ASSUMED** | AMD recommendation; settles §1.4's 634 ms/step arithmetic (M6) |
| CDAR transport is 1.5× slower than RCCL | **MEASURED** | `G25_1_STATUS.md`; 2,214 / 1,470 µs in-rig |
| fused CDAR hides ~0% of the collective | **MEASURED** | 3,606 vs 3,653 µs — but at an operating point where little was hideable (§2.4) |
| K1 = store-towers on gfx950 at 235 MB | **MEASURED** | 117.2 vs 67.1 GB/s; reverses the design doc; **reverses again at 3.67 MB** (M5) |
| K3 optimum = 256-row slabs | **MEASURED, one shape, pure transport only** | non-monotone; unswept co-resident (M4) |
| K4 depth flat in pure transport | **MEASURED** | re-scopes the −615 µs law to co-residency (M3) |
| CDAR moves −32% fewer bytes than production | **FALSE as written; ≈ −10%** | §1.3 re-derivation; **must be corrected in the design doc** |
| We sit at 58% of egress ceiling, RCCL at 79% | **DERIVED** | ceiling itself is flagged *unreproduced* in `OVERLAP_ABSTRACTIONS.md:147` — re-measure before quoting |
| Correctness of the whole CDAR stack | **MEASURED, PASS** | exact bf16 sums, zero timeouts, every arm/config, G25-0b and G25-1; ISA gate at `1e4068df`: 0 spills / 0 scratch, `vmcnt(4)`×2 + `pk_add` + `s_sleep` backoff present |
| GEMM-RS 376.7 vs RCCL 421.8 vs RadeonFlow 485.3 µs | **MEASURED (prior work, gfx950)** | `ARCHITECTURE.md:124`; **`SOURCE_AUDIT.md:104` warns EV=1 and REDV=1 were never measured together and the new composition is unmeasured** — do not treat G25-0a as transferring to CDAR |
| Shared-expert filler ports to TP8 with less complexity | **DESIGN, unmeasured** | `TP8_OVERLAP_ANALYSIS.md:83`; capped at ~1/8 of routed FLOPs |
| fp8-on-wire helps more under TP8 than DP | **SPECULATION (mine)**, with a grounded mechanism | the FP8 ledger's central risk (un-hiding hidden wire time) does not apply where the wire time is already exposed |

---

## 8. The one-line campaign consequence

Under DP8/EP8 the mediating quantity is **fill** (37.6% vs 98.5%), and it
predicts the sign flip between C=32 and C=512 that we already measured. Under
TP8+EP fill does not exist; the mediating quantity is **exposed collective
fraction per layer** (~28%, DERIVED), and the schedule knob it selects is **K1 ×
K3 × co-residency**, not readiness order. The campaign's generalization claim
therefore needs *two* laws, not one — and the second is currently supported by
a red gate, four null results, and one unrun profile. Getting M1 and M2 done is
worth more to the master doc than any further schedule mutation.

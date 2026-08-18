# Ablation Campaign Brief — Workload → Megakernel Schedule Cost Model

**Date:** 2026-08-18 (post draft8 presentation to industry experts)
**Repo root:** /Users/subha/repos/Distributed-HipKittens
**This dir:** distributed-kernels/fused_moe/overnight/aug18-ablations/

## Mission

After presenting draft8 (high-concurrency prefill win), industry experts (incl. Simran)
pushed back: the results are good but **it is unclear what the cost model is and what
we have learned**. The megakernel's optimal strategy (comm/comp overlap schedule)
changes significantly with the workload — topology/parallelism, prefill vs decode,
batch size, concurrency, sequence length — and it is unclear which metrics even
matter for each workload (TPOT, TTFT, throughput...). We must design a principled
ablation campaign such that we can say: **"if you change X in the workload, the
megakernel schedule should change by Y, because mediating quantity Q crossed
threshold θ"** — and back every such law with measurements.

## Expert feedback (verbatim notes from the meeting)

- does our m15 producer/consumer work best for all sizes?
- It's powerful premature to say that producer/consumer may not be best. Might be.
- List of what we'd like to teach people about
- What the pull request in HK looks like
- Super curious about the findings before we even look at other workloads
- We need to have a cost model and enumerate what are the properties that control the workload
- As we toggle the list of workload [properties], what are the properties of the kernel that changes
- Workload setting to schedule design — and can we do that in a cost model?
- Risk flagged: "Very expansive, unclear what the core contribution might be" — we
  must NOT let that happen. Core contribution must stay crisp.
- Reporting discipline: "People keep posting TPS without batch size or concurrency,
  input and output lengths, or even saying whether it means per-request decode speed
  or aggregate throughput. On its own, the number means almost nothing."

## Draft8 digest (what was presented — all numbers are measured)

**End-to-end vLLM (DeepSeek-R1-class MoE, 8×MI355X gfx950, MORI/vLLM 0.25.1):**
- C=32 (1,024 QSL prompts, ISL 4096, OSL 8): M15 DP8/EP8 20,372 input tok/s vs
  AMD-recommended TP8+EP 25,844 → **−21.17%**. TTFT p50/p99 2.445/4.364 s vs 1.339/5.682 s.
- C=512 (2,048 prompts, ISL 4096, OSL 8): M15 42,788 vs AMD DP8/EP8 39,474 → **+8.40%**.
  TTFT 43.2/47.9 s vs 28.9/52.8 s.

**Low vs high concurrency prefill (why the flip happens):**
- C=32: real rows 1,539/4,096 per rank step (37.6% fill), ~56% of steps pure dummy work.
- C=512: ~4,035/4,096 (98.5% fill), <2% dummy. DP starves at low C; TP8 AR bytes scale
  with batch; at high C our kernel overlaps the MoE all-to-all + combine and wins.

**Kernel-level ladder (4,096 tok/rank prefill boundary, median):**
- Production 7,712 µs → homogeneous mega 6,908.8 (0.8958×) → mode-12 depth-4 6,483.8
  (0.8407×) → M15 16 initial consumers 6,292.4 → 24: 5,848.5 (0.7589×) → 28: 5,822.0
  (**0.7544×**) → 32: unstable (hung 1 of 2 runs).
- Dedicated communication CTAs (C=64): 6,866 µs (0.888×) — measured **+831 µs added
  traffic** from the comm-CTA role itself → falsified in favor of producer-carried.
- Producer-carried remote accumulation: unbounded 7,110.8 (WORSE than homogeneous);
  depth-4 bound 6,495.8 (−615 µs). Depth = bound on outstanding remote ops per wave;
  it is congestion control.
- Packed bf16 remote atomics: 52.8 GB/s coalesced (near 54.9 GB/s remote stores);
  scattered lane addresses collapse to 4.1 GB/s → **layout is a schedule decision**.
- Consumer readiness: under natural GEMM-2 task order the median token is not reducible
  until **91.7%** of GEMM-2 completes → producer order restructured to complete an
  independent slab early. Law: **signal the smallest early-EXECUTABLE unit, not the
  smallest stored/transferred unit.**
- Push vs pull: matched 64 KiB producer→transport→dependent-consumer wall time
  0.9936× (tie); 64 MiB single-link BW pull/push = 1.078×. Push when progress is
  producer-defined & many-to-one (ERS half); pull when consumer-defined & one-to-many
  (MAG half). Both can approach link saturation.
- Prefill T-sweep (M15 C=28 vs production): T=4,096: **0.7557×**; T=2,048: 0.8845×;
  T=1,024: **1.0940× (we lose)** — workload dependence already measured on one axis.
- Serving reality: ~15% run-to-run e2e variance between production runs; route capture:
  **51.9% of expert-slot traffic went to experts 0–7, all resident on rank 0** under
  contiguous placement → hot rank produces its slab late and holds the global slab
  certificate hostage. Skew is a first-class workload axis.
- Real-serving governed by hottest expert rank: e2e p50 −1.06%, TTFT p99 +5.67%,
  TPOT p50 −2.51% (M15 vs production at the captured operating point).
- gfx942/MI300X: remote atomics ~3× slower than remote stores → producer-carried
  atomic accumulation may be wrong there; needs per-source non-overlapping slots +
  owner-local reduce. Architecture is a workload/hw axis.

**TP8+EP interim (M25 G25-0b / G25-1, boundary level):**
- Fused CDAR does not yet beat GEMM→RCCL at the boundary: 2.9 (GEMM→RCCL) vs 3.6 ms;
  RCCL AR is 1.5× our transport. Four overlap hypotheses tried and falsified in-rig
  (drain pipelining, register-source stores, consuming-pool specialization, item
  granularity). Store-towers transport: 93.6 GB/s effective AR @235 MB (4.2× v0);
  atomics cap ~67 GB/s. **Depth is FLAT in pure transport** — the depth-4 law is
  co-residency-specific (only binds when compute shares the device).
- Amdahl: native TP8 closes at ~2.8 ms/layer exposed AR — that is the prize pool.

**DHK primitives already extracted (the HipKittens PR skeleton so far):**
PGL/peer translation (`pgl.on(rank)`), typed packet movement (16B peer load/store,
register-row stores, streaming/multi-region copies), packed-bf16 peer accumulation,
directional release/acquire, epoch publication + bounded waits, counted fan-in +
reservations, directed retirement credits (ready ≠ reusable), CTA role helpers;
extracted headers: include/cdna4/ops/group/distributed/{credit,order,slab}.cuh +
counted_arrive_dynamic_into.

## Hard constraints & assets

- Hardware: one 8×MI355X (gfx950) node (ssh -i ~/.ssh/muhammad-gpu subvadla@10.145.64.67);
  occasional gfx942/MI300X comparisons. ~15% e2e run variance → boundary-level rigs are
  the precision instrument; e2e is the validation instrument.
- Assets: M15 serving kernel (k0pf6gm_device_tile_m15.hip) + mps/t2b/t3 variants;
  single-process 8-GPU boundary harness (G25-0b CDAR microbench); route-capture +
  replay rigs; vLLM+MORI serving harness w/ coverage counters; nightshift/
  BENCHMARK_PROTOCOL.md (order-balanced ≥5 pairs, open-loop Poisson cells, accuracy
  gate, TTFT p50/p99, native-tuned baseline).
- "Hundreds of kernels" must be **points in a knob space**: parameterized skeletons +
  a manifest, not hand-written one-offs. Every arm must be reproducible from its
  manifest row.
- Reporting: every number carries its full workload tuple. No naked TPS, ever.

## Required outputs of this campaign design (the master doc's skeleton)

1. **Workload state vector W** — the enumerated properties that control the workload,
   each mapped to the physical quantities it moves (bytes on wire, FLOPs, slack, skew, fill).
2. **Schedule state vector S** — the kernel-design degrees of freedom (carrier role,
   direction, readiness granularity, injection depth, consumer policy, producer order,
   transport op, overlap targets, fusion boundary...), organized as knobs × skeletons.
3. **Cost model M(W, S; θ)** — analytic structure + measured hardware parameters θ +
   calibration plan + the decision boundaries it predicts, stated as falsifiable laws.
4. **Metric map** — which metric is primary per workload cell (TTFT/TPOT/throughput/
   goodput/cost), and the reporting discipline.
5. **Experiment ladder** — calibration microbenches → boundary grid sweeps → e2e
   validation; gates, sample sizes, kill criteria, node-hour budget.
6. **Per-cell kernel predictions** — for each region of W-space, the schedule S we
   predict wins, why (via M), and what to implement first.
7. **Expected emergent abstractions** — what patterns fall out → the shape of the
   HipKittens PR; the "list of what we'd like to teach people."

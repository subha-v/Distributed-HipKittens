# The overlap program: training-first, with the M22 serving map (2026-08-15 ~01:00 UTC)

Response to the design discussion (small-K all-layer replicas, adaptive CTA
roles, shared-expert filler, operator portfolio) — evaluated against
tonight's measured data, translated into the TRAINING program we are
actively building, with the serving M22 map kept warm.

## 1. Where overlap actually lives, enumerated honestly

The forward's dependency chain (attn -> router -> dispatch -> GEMMs ->
combine -> residual -> next attn) leaves exactly one legal cross-region
forward overlap: combine(L) tail vs attention(L+1) prologue on finished
tokens (fine-grained, expensive machinery, low priority).  Dispatch sits
directly on the critical path after the router — nothing to hide it under
IN FORWARD except independent work.  The real overlap currency is
**independent filler compute**, and training is uniquely rich in it:

| filler | FLOPs share | comm it can hide | status |
|---|---|---|---|
| wgrad dW1/dW2 (bwd) | ~half of bwd MoE | dY dispatch + dX combine | T2 design, kernel slots ready |
| shared expert fwd | ~1/8 of routed fwd | fwd dispatch + combine | next kernel arm |
| shared expert bwd (dgrad+wgrad) | ~1/4 of routed bwd | bwd transports | with the above |
| attention wgrad (dW_qkv...) | large | any MoE-region comm | framework-level (split backward_dw exists); far tail |
| next-microbatch fwd dispatch | — | bwd tail | M21 epoch pipelining; frontier |

Serving prefill has NONE of the first three at meaningful scale (no wgrad,
shared expert only ~1/8 fwd) — which is exactly why "M15 with more overlap"
plateaued and why the training pivot is the right lab.  In serving, the
only lever that moves the skewed critical path is MOVING COMPUTE
(replication); overlap is the second-order term.  The proposal's own
conclusion — "first make expert computation movable, then schedule overlap
around the balanced work" — matches our banked M21 direction doc.

## 2. Tonight's measured frontier (the small-K question, quantified)

Offline all-layer replica frontier from the banked aggregate histograms
(90,050 calls; greedy marginal-benefit allocation; 42 MB/expert):

- Raw mean max-rank-load 5.05x.  The frontier is NEAR-LINEAR to ~18.5 GiB
  (marginal ~0.56x per GiB), then FLAT 18.5→31 GiB, then completes to
  1.09x at 38 GiB (=K16 everywhere, the M19 cost).
- Uniform-K rows: K=2 (4.8 GiB) -> 4.10x; **K=4 (9.5 GiB) -> 3.23x**;
  K=6 (14.3 GiB) -> 2.41x; K=8 (19.0 GiB) -> 1.77x; K=16 (38.1) -> 1.09x.
- Greedy ≈ uniform+epsilon at equal bytes — per-layer K tuning buys little;
  UNIFORM K is fine (simpler tables, simpler kernel contract).

Implication for M22: top-4-all-layers HALVES the skew damage (5.05→3.23)
rather than eliminating it; the e2e estimate at f≈0.45 lands in the
+25-30% class IF per-chunk coverage holds — and that is the open risk: the
aggregate frontier cannot price the per-chunk coverage bimodality (p5=1%,
p75+=77%) that a static replica set runs into.  **Raw captured routes (the
never-run M19 replay contract) are a hard
prerequisite for the serving M22 build** — the capture hook exists
(M15_SKEW_CAPTURE_LAYERS), it was simply never exercised.  K=6/K=8
(14-19 GiB) buy real headroom if the KV budget tolerates them; the
memory/perf frontier is now a table, not a guess.

## 3. The unifying abstraction: device-side work queues + CTA roles

The proposal's strongest structural idea.  Build it ONCE, in T2, where it
is immediately needed: the wgrad filler is precisely "role: filler,
consume(wgrad_queue), yield to combine tickets."  The M8 dynamic wave
tickets (m8_front/m8_back) are already a working queue instance; the M20
prefetch carve is a working role-partition instance.  Generalize to:

    roles.compute.consume(gemm_tasks)      // tile_desc-driven, as today
    roles.service.consume(combine_tickets) // m8 tickets, as today
    roles.filler.consume(wgrad_tiles | shared_expert_tiles | prefetch_duty)

with role membership chosen from a small compile-safe set (the C-sweep
already proved C in {16..28} is flat under skew — the regimes exist).
Load-adaptive role assignment (post-histogram) matters in the SKEWED
serving regime, not in balanced training — but the queue/role plumbing is
identical, so T2 builds the mechanism and M22 inherits the policy.

## 4. Transport hygiene (order of application, both programs)

1. fp8-on-wire combine (fwd y, bwd dX): MORI-measured 366→642 GB/s class;
   accuracy contract = per-row-group dynamic scales, gate vs fixtures.
2. Destination-interleaved dispatch/combine record order: MORI measured
   65% on push-combine; matters under ANY destination concentration (real
   serving routes; ragged training tails).  Cheap.
3. s_sleep backoff in every poll loop (fabric livelock insurance; MORI does
   it; our spin loops predate the lesson).
4. Source-side same-token pre-reduce: LOW value at EP8 (top-8 spreads ~1
   expert/rank; the M1 dedup already folds same-dest picks); revisit at
   EP16+.

## 5. Sequencing (training first, serving M22 kept warm)

T2 ladder (active): correctness gate (fixtures DONE balanced; skewed
running) -> wgrad filler via the role/queue mechanism -> shared-expert
fwd+bwd arms -> fp8-on-wire -> transport hygiene -> M21 epoch pipelining.
Target: beat turbo_gg's measured 871 ms/iter backward, then the 1,386
ms/iter step.

Serving M22 (when a node window opens): (0) raw-route capture + captured
replay of production/m15/m19 — the transfer mystery must close first;
(1) EPLB baseline pair; (2) M20-with-stock-fallback (cached layers m20,
uncached PRODUCTION); (3) uniform-K sweep K in {4,6,8} all-layer kernel with
DECIN precompute + the T2-built role scheduler; (4) portfolio dispatcher
(CAKE-style, per-chunk stats already precomputed).  Every stage must satisfy
`../../../../docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md` — M23 patch
chain in both arms, `RAGGED_SEAL_RECEIPT` coverage quoted with every mega
number, rescued-stock baseline, >=5 order-balanced pairs, accuracy A/B;
production+EPLB always in the table.  (Note 2026-08-18: the pre-M23 serving
pairs this map was drawn against were retired as obsolete — the seal fired on
~2% of heavy steps and both arms' baselines were depressed.)

# M18: static hot-expert replication — kernel-level results (2026-08-14)

## The finding that motivated it

Completed both-arm serving skew measurement (the handoff's blocked instrument):
real MLPerf R1 serving routes **51.9% of all expert-slot traffic to experts
0–7**, all in rank 0's contiguous block (Gini 0.609 aggregate; per-layer
0.68–0.83). Receive-side rank load: **5.09× fair share aggregate, 5.59× worst
layer (`router041`)**. Send-side is flat — the earlier "flat rank load" read
was send-side only. Every MoE layer runs at the slowest rank's speed, so both
M15 and production pay a ~3.2–3.6× skew tax every step. Caveat: ~18% of the
measured stock-arm traffic is warmup/profile batches (the m15 arm ran ~2×
traffic from gate self-tests and is diluted — use stock-arm histograms only).

Skew replay rig (new): `K0_MOK_ROUTE_HIST` in the MoK harness samples routing
from a measured 256-bin histogram via Gumbel-top-k (marginal fidelity: corr
0.998, Gini 0.331 vs 0.341 target). Histograms banked in
`benchmarks/mok_synthetic_prefill/route_hists/` (amd-master
`debug/pf6-first-launch`). `make_route_hist.py` builds
aggregate / worst_rank / worst_gini variants from skew dumps.

## Skew replay baselines (T=4096/rank, C=28 g=353 mode 12 flush 16, 5 runs)

| arm | routing | p50 | vs production |
|---|---|---:|---:|
| M15 | balanced | 5,823 µs | 0.7556 (reproduces banked 0.7544) |
| M15 | measured aggregate | 20,739 µs | 0.8467 |
| M15 | worst layer | 22,475 µs | 0.8559 |
| m17 (RR) | measured aggregate | 20,504 µs | 0.8370 |
| m17 (RR) | worst layer | 22,118 µs | 0.8413 |

Decision rule resolved: RR landing order buys ~1–1.5% under skew; the
bottleneck is **popularity concentration**, not ordering.

## M18 design

`K0P6_M15_REPLICATE` compile arm on the M15 body (`ablations` @ 0f9676ce;
RUN PIN `ablations-m18` @ 92440297). Every rank hosts the hot experts'
weights in local slots `E..E+nrep-1` (EL ≤ K0P6_MAXE=64); descriptor slot 63
carries an "M18R" table (magic, nrep, int8 lut[256], u32 bitmap[8]).

- **M1 dispatch:** replicated experts route **source-locally** (`dest = cur`).
  Send-side flatness ⇒ their GEMM work splits 8-way evenly by construction,
  and their dispatch bytes + M7 remote RMWs become rank-local (>52% of
  traffic off XGMI under the measured histogram).
- **Exactly-once rule:** a replicated expert is computed only from
  self-source rows on every rank (owner rejects foreign rows for it) — the
  acceptance predicate in M2's histogram and the scatter. Scatter iteration
  is RR-permuted (m17 order) with LUT slot lookup.
- **M7/M8/M9 untouched:** combine consumes (dest, landing-row) records and
  never consults expert ownership.
- Host side (`K0_MOK_REP_EXPERTS`, amd-master through 95ffac45): replica set
  = top-K of the route hist (or explicit csv); replica weights materialized
  bit-identically (harness: regenerated from owner seeds); EL-expert
  W13/S13/W2/S2 for the mps arm only; PADMAX margin +ceil(31·nrep/32)·32.

## M18 results (same rig, all gates green: poison 0 survivors, reference
## match vs production on identical routes, 600-epoch soak, pperr=0, 5 runs)

| replicas | routing | p50 | ratio vs production |
|---:|---|---:|---:|
| 8  | measured aggregate | 7,564 µs | **0.3085** [0.3080, 0.3088] |
| 16 | measured aggregate | 6,056 µs | **0.2489** [0.2476, 0.2499] |
| 32 | measured aggregate | 5,905 µs | **0.2420** [0.2409, 0.2432] |
| 16 (layer-adaptive) | worst layer | 5,682 µs | **0.2169** [0.2163, 0.2174] |
| 8  | balanced (control) | 6,046 µs | 0.7848 [0.7830, 0.7860] |

Reads: the timing tracks the load model (predicted max rank load 1.45× /
1.14× / 1.07× for 8/16/32 replicas); at 32 replicas the skew penalty is
fully erased (5,905 µs vs 5,823 µs balanced M15). r16 is the value point
(~670 MiB/rank). Balanced-routing cost of carrying 8 replicas: **+3.9%** vs
M15 (planner covers EL slots + tiny fragmented replica blocks) — still
0.785× production. Decomposition of the 4.1× aggregate win: ~3.2× from
removing the slowest-rank overhang + ~1.3× from the M15 fusion chassis.
This is a **kernel-region** number (one MoE layer, 4096 tok/rank prefill
shape, fixed replayed route, production arm = AITER fused-MoE + MoRI
all2all); serving end-to-end will dilute it, decode is untouched
(activation is exact-B4096 prefill chunks only).

## Fairness notes for any external claim

vLLM ships EPLB (off in this deployment): static redundant experts would
flatten load too, but replicas still receive tokens over the fabric — M18's
source-local routing removes the traffic itself. A defensible external
claim needs: production+EPLB baseline, ≥5 order-balanced serving pairs,
MLPerf accuracy check, and a second workload (this histogram is
MLPerf-text-specific and partially warmup-diluted).

## Artifacts

- Campaigns: `~/k0-mok-synthetic-results/skew[A-E]_*_0814T0456/`,
  `m18[F-J]_*_0814T0522/` (summary.json + per-run logs + rank JSONs).
- Skew dumps: `~/20260813_m15_campaign_m15pair1/skew/` (stock arm
  authoritative; 15 partial files archived in `skew_partial_0404_backup/`).
- Harness commits (amd-master `debug/pf6-first-launch`): 28c4ba8f bank,
  5c0d7450 route-hist replay, eea17f04 histograms, 0e7edb09 M18 host,
  95ffac45 PADMAX passthrough.
- Kernel commits: `ablations` 0f9676ce, `ablations-m18` 92440297;
  node worktree `~/DHK-m18`.

## Serving pair #1 (2026-08-14, `vllm-integration-m18` @ 72591acd) — NEGATIVE

The integration itself worked first try: env-gated on the m15 mode
(`VLLM_PF4H_M18_REP_EXPERTS`), owner-broadcast replica weights behind `[:E]`
views, 64-word descriptor, all receipts 8/8 including
`M18_REPLICATION_RECEIPT`, 1024/1024 completed both arms
(`~/20260814_m15_campaign_m18pair1b/`).  The performance did NOT transfer:

| metric | stock | m18 (top-16) | delta |
|---|---:|---:|---:|
| input tok/s | 12,562.5 | 11,842.2 | **−5.7%** |
| TTFT p50 / p99 | 3,089 / 5,430 ms | 3,523 / 6,469 ms | +14% / +19% |
| TPOT p50 | 1,072 ms | 1,098 ms | +2.4% |

Two suspects, in order:

1. **Aggregate-vs-per-chunk statistics (prime).**  The MoK replay samples
   every batch from the aggregate histogram — constant 58.7% replica-set
   coverage, zero variance.  Real chunks are document slices: mean coverage
   is still 58.7%, but chunks whose hot experts fall outside the static set
   pay the full skew penalty PLUS the replication carry cost, and chunk
   wall-time is convex in skew.  The per-call instrument (skew hook @
   72591acd: per-call coverage + max-rank-load histograms) measures exactly
   this distribution; decision pending its first pass (`m18diag`).
2. **KV pool (secondary).**  +41 GB replica weights at
   `--gpu-memory-utilization 0.70` halved available KV (68.8 → ~31 GiB,
   max concurrency 64× → 29×).  Arithmetic says 32 concurrent 4k requests
   need ~5 GB so this should not bind, but it is a confound to remove
   (smaller nrep, or a higher utilization for both arms) before any rerun.

If the per-call distribution shows high coverage variance, the static set is
structurally insufficient for serving and the design moves to per-chunk
replica selection (device-side planning prologue, MoonEP-style) or
layer/chunk-adaptive tables.  The kernel-level result stands as measured —
the open question is the routing statistics serving actually presents.

## Serving pairs #2 and #3 (2026-08-14, later session) — THE WINS

The diag pass measured the per-call distributions (n=90,050): static-set
coverage bimodal (p5=1%, p75+=77%), per-call max rank load median 5.15x /
p95 6.25x, and even ideal static redistribution leaves p95 at 3.05x.  Both
fixes were built and paired the same day (identical rig as pair #1):

| pair | candidate | tok/s vs stock | TTFT p50 / p99 | E2E p99 |
|---|---|---:|---|---:|
| #2 | M18, PER-LAYER top-16 sets (58 distinct) | **+11.3%** | −9.8% / −7.1% | −12.3% |
| #3 | **M19: per-layer sets + per-chunk adaptive (theta=64)** | **+39.8%** | −23.1% / −32.2% | −24.2% |

(#2: 11,339.5 vs 10,186.0 tok/s; #3: 13,395.4 vs 9,581.3 tok/s; 1024/1024
both arms both pairs; all receipts 8/8.)  The day's arc: aggregate-static
−5.7% → per-layer-static +11.3% → per-chunk-adaptive +39.8%, each step
aimed at exactly what the previous measurement exposed.  Tails improved
more than medians in #3 — the hot-rank overhang signature.

Caveats held open: n=1 pair each; stock drifted downward across the day's
pairs (12.6k → 10.2k → 9.6k tok/s — node state; the paired design absorbs
it, but ≥5 order-balanced pairs remain mandatory); M19 output accuracy is
being verified (harness reference-match gates GREEN on first campaigns;
MLPerf AccuracyOnly A/B pending).  M19 design + kernel: `M19_DESIGN.md`,
`ablations` @ 0430aeb7, serving `vllm-integration-m18` @ c1b76022.

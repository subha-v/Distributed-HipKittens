# M20 serving pair #1 — NEGATIVE (−19.6%), and what it measured (2026-08-14 ~22:15 UTC)

One receipt-gated stock-vs-m20 pair on c32p (1,024 real MLPerf prompts, ISL
4096, OSL 8), integration exactly per the handoff: budgeted replica cache of
**12 layers ≈ 8.06 GB** (`17,20,34,36,39,41,43,44,45,49,55,56`, per-layer
top-16 sets, theta=64), DECIN precomputed on-stream in `prepare()`, uncached
layers on the plain m15 mega (handoff option (a)).  Serving commit
`vllm-integration-m18` @1e5ad0ff; deploy set `~/m20_deploy_sources_20260814`
@ DHK faa26c49; results `~/20260814_m15_campaign_m20pair1/`.

## Numbers (both arms, same server config, stock control first)

| metric | stock | m20 | delta |
|---|---:|---:|---:|
| total throughput (tok/s) | 12,668 | 10,186 | **−19.6%** |
| TTFT p50 / p99 (ms) | 3,271 / 5,431 | 4,171 / 7,150 | +27.5% / +31.6% |
| TPOT p50 / p99 (ms) | 1,042 / 1,539 | 1,322 / 1,601 | +26.9% / +4.0% |
| E2E p50 / p99 (ms) | 10,420 / 14,045 | 13,518 / 17,211 | +29.7% / +22.5% |
| Available KV cache memory | 68.83 GiB | 60.95 GiB | −7.88 GiB (= the 8,065 MB cache, exactly) |
| GPU KV cache size (tokens/rank) | ~2,103k | ~1,866k | −11.3% |

Correctness: all nine receipt gates 8/8 (`M20_CACHE_RECEIPT` included),
M15_PPERR = 0, kernarg segment 280 as expected, budget filtering visible
(`M20_LAYER_UNCACHED` on non-budget layers, 12 extended layers per rank).
The integration is *correct*; the configuration is *slow*.

## Why (two findings, both load-bearing)

1. **Per-layer damage is UNIFORM** (m18diag stock histograms, 2026-08-14):
   raw max rank load ≈ 5.5x on every one of the 58 layers; per-layer
   replication benefit ≈ 4.45x each.  There are no "most-damaged" layers —
   a 12/58-layer cache can address only ~1/5 of the replication opportunity
   regardless of which layers are picked.  The handoff's "match M19's
   +39.8%" bar assumed damage concentration the histograms refute.

2. **The un-replicated mega LOSES to production under real routing.**  With
   46/58 layers on the plain m15 kernel, the arm is mostly an m15 arm — and
   backing the measured −19.6% end-to-end out through the MoE fraction
   (≈40–54% of stock step time, two independent derivations) implies the
   m15 mega runs ≈ **1.8–2.0x slower than production AITER+MoRI per
   uncached layer** under real per-chunk skew, versus 0.8467x (i.e. 18%
   FASTER) in the aggregate-histogram replay.  This is the serving pair #1
   lesson sharpened: chunk wall-time is convex in skew, per-call max load
   hits 6.25x at p95, and the mega's tail behavior under extreme chunks is
   far worse than the aggregate replay predicts.  Consistent record: no
   mega arm has ever beaten stock on real prompts WITHOUT replication
   active on every layer (m18 aggregate −5.7%, m18 per-layer +11.3%, m19
   all-layer adaptive +39.8%, m20 12-layer −19.6%).

## Caveats

n=1 pair; stock drift across the day is large (12.6k → 10.2k → 9.6k →
12.7k tok/s across the four pairs' stock arms), so cross-pair comparisons
are unreliable — within-pair deltas are the signal.  Accuracy A/B
(500-sample AccuracyOnly, stock vs m20) queued and running.

## The levers this points at (not yet built)

- **Uncached layers → stock production path** (per-layer activation gate in
  the shim instead of per-layer kernel choice): removes the regression
  source; expected ≈ +6–9% at the same 8 GB (12 layers at 0.2483x, 46 at
  1.0x, MoE fraction ~0.4–0.5).  Small runtime change; the safe next pair.
- **Small-k all-layer cache** (e.g. per-layer top-4 ≈ 168 MB x 58 ≈ 9.7 GB):
  keeps the M19 mechanism (and the mega's fusion win) on EVERY layer at
  reduced coverage — plausibly much better than the same memory spent on
  full-k for few layers, because it also fixes finding 2.  Needs a k=4 sets
  file; mechanically supported by the existing code path.
- The memory dial itself works exactly as designed (KV pool −7.88 GiB =
  the configured cache, vs M19's −41 GB).

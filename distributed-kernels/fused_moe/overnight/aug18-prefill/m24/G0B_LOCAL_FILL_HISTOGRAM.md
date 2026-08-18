> ## ⚠ CORRECTION — 2026-08-18 07:00 PDT (read before anything below)
>
> Two numbers in this document are **wrong**. The document is preserved
> unedited below for the record; these corrections supersede it.
>
> 1. **The "1.19× aggregate padding multiplier" is WITHDRAWN — arithmetic
>    artifact.** `g0b_local_fill.py:89` filters `n_real > 0`, which silently
>    drops the **91 zero-fill (fully dummy) calls** from the aggregate. The
>    honest **all-512-rank-call aggregate is 1.446×**, not 1.19×. Every
>    downstream use of 1.19× (including as "the" corpus padding multiplier
>    against the serving receipt's 2.66×) is void.
> 2. **The 31 % dummy rate is a RAMP-PHASE rate, not the whole-run rate.**
>    The corpus window is the first 64 B4096 calls per worker. The whole-run
>    rate, read from the `RAGGED_SEAL_RECEIPT` lines, is **~56 % of in-bucket
>    steps**. Tier 1 therefore targets **~56 %** of in-bucket steps — nearly
>    double what this document modeled.
>
> **What STANDS, and is strengthened:** every **shape** conclusion — the
> distribution is **bimodal**; **non-dummy calls are ~0.91 full**; **tier-2
> row plumbing is weak** (there is little partial-fill work to recover); and
> the model is **tier-1-dominant**. With the corrected dummy rate, tier 1's
> advantage over tier 2 is *larger*, not smaller.
>
> **Reconciliation with the serving receipt (no contradiction remains):** the
> corpus was right about shape, the receipt right about rate; composing them
> (0.442 × 0.9146 ≈ 0.404 vs the receipt's 0.376) closes ~86 % of the apparent
> gap, the residual sitting inside the known capture bias.
>
> **Open gates:** (a) a `RAGGED_SEAL_RECEIPT` **printf grep** on the node to
> confirm the `/8` peer-summing convention; (b) **dummy-step rank-synchrony**
> — a step-level tier-1 skip requires all 8 ranks dummy on the same step.
>
> Source: provenance reconstruction, held locally; conclusions summarized
> here; receipt re-verification queued on the node.

---

# G0b (local surrogate): per-call real-row distribution and the M24 cost model

Status: **partial discharge of G0b**. FILL_AWARE_DESIGN.md rev 3 §A.4.3
withdrew the tier-1 win estimate pending a distribution of per-step real-row
counts (`n_orig`). The seal receipts live on an unreachable node, so this
document substitutes the **captured route corpus**, which is local and
carries the same information at rank-call granularity.

Script: `g0b_local_fill.py` (this directory). Corpus: 8 worker `.npz` files
x 64 B4096 calls = **512 rank-calls**, raw `topk_ids` `[4096,8]` uint8 +
fp16 `topk_weights`, captured on the m15+M23 server at cell c32p
(routecap1). Corpus path is a session scratchpad and is NOT checked in.

## 1. Degeneracy rule (identical to CORPUS_FINDINGS.md)

Rows are grouped by their **sorted top-8 expert set**; the largest group is
the modal group.

* **Dummy call** <=> the modal group covers **> 95%** of the 4096 rows.
  This rule reproduces the CORPUS_FINDINGS count **exactly: 160/512 =
  31.2%** dummy calls. (Agreement is exact, not approximate.)
* **Pad group inside any call** <=> the modal group has **>= 8 rows** AND
  the across-row STD of its top-8 weights is **< 0.02**. The weight test is
  load-bearing: it is what separates true pads (identical hidden states =>
  STD ~ 0.0008 at the median) from the *genuine* hot-set modal pattern
  {0..7}, which covers ~16% of rows in real calls but carries varying
  weights. Under top-8-of-256, an accidental exact set collision across
  many rows is probability-zero, so a large near-zero-STD group is pad.
* `n_real = 4096 - |pad group|`, or 4096 when no group qualifies.

`T_eff = min(round_up(n_real, 256), 4096)` -- the 256-row tile clamp M24
actually schedules against.

## 2. n_real / fill / T_eff distribution

### All 512 calls

| stat | n_real | fill f | T_eff |
|---|---|---|---|
| mean | 2833.2 | 0.6917 | 2865.5 |
| p10 | 0 | 0.0000 | 0 |
| p50 | 4096 | 1.0000 | 4096 |
| p90 | 4096 | 1.0000 | 4096 |

Deciles (0,10,...,100):

* `n_real`: 0, 0, 5, 1994, 4096, 4096, 4096, 4096, 4096, 4096, 4096
* `T_eff` : 0, 0, 256, 2048, 4096, 4096, 4096, 4096, 4096, 4096, 4096

### Non-dummy calls only (n=352)

| stat | n_real | fill f | T_eff |
|---|---|---|---|
| mean | 3746.2 | 0.9146 | 3768.7 |
| p10 | 2157 | 0.5265 | 2304 |
| p50 | 4096 | 1.0000 | 4096 |
| p90 | 4096 | 1.0000 | 4096 |

Deciles `n_real`: 364, 2157, 4096, 4096, 4096, 4096, 4096, 4096, 4096,
4096, 4096.

**The distribution is strongly bimodal, not centered.** It is ~31% calls
with essentially zero real rows and ~60% calls that are completely full,
with only ~10% of calls genuinely partially filled (p10-p20 of the
non-dummy population, 364-2157 real rows). This shape is the single most
decision-relevant output of G0b, and it was not the shape §A.4 assumed.

## 3. Padding multiplier before / after M24

Per-call ratio means are dominated by the near-zero-`n_real` dummy calls
(4096/n_real diverges), so the **aggregate** ratio -- total rows executed /
total real rows, which is the quantity that maps to time -- is the honest
statistic.

| population | PRE-M24 `4096/n_real` (agg) | POST-M24 `T_eff/n_real` (agg) |
|---|---|---|
| all 512 calls | 1.189 | 1.011 |
| non-dummy (352) | 1.093 | 1.006 |

Per-call means, for completeness: all-calls 87.28 -> 6.30 (both inflated by
dummies); non-dummy 1.378 -> 1.017.

**M24 removes essentially all of the row-level padding** it can see: the
residual 1.011x is pure 256-tile quantization.

## 4. Modeled per-step MoE cost ratio

`rho(step) = (1 - phi) + phi * (T_eff / 4096)`, averaged over the empirical
distribution (the distribution-weighted version of the §A.4 point estimate).
`phi` = the fraction of MoE-region cost that scales with row count.

| variant | mean `T_eff/4096` | rho @ phi=0.80 | rho @ phi=0.90 |
|---|---|---|---|
| **full M24** (tier 1 + tier 2), all calls | 0.6996 | 0.7597 (1.316x) | **0.7296 (1.371x)** |
| non-dummy calls only | 0.9201 | 0.9361 (1.068x) | 0.9281 (1.077x) |
| **tier-1 only** (T_eff=0 on dummy, 4096 otherwise) | 0.6875 | 0.7500 (1.333x) | **0.7188 (1.391x)** |

### The headline for M24's design decision

**Tier 1 delivers 0.7188; full M24 delivers 0.7296 -- tier 2 is WORSE than
tier 1 alone in this model, and in any case adds nothing.** Tier 1 captures
the entire modeled win (mean `T_eff/4096` 0.6875 vs 0.6996). Tier 2's
row-granular skipping recovers only the ~10% of calls that are partially
filled, and the 256-tile clamp eats most of that; the small rho *increase*
for full M24 is the tile-quantization residue on those partial calls, which
tier-1-only never pays because it does not touch non-dummy calls at all.

Implication: **the complexity budget belongs in tier 1** (detect and skip
an all-dummy sealed call), which is a single cheap planner-side predicate,
not in the n_orig-proportional row plumbing of tier 2. Tier 2 should be
gated behind evidence that real serving fill is materially below what this
corpus shows -- see the tension in §5.

## 5. Caveats -- read before quoting any of this

1. **This is a surrogate for G0b, not G0b.** G0b as specified wants seal
   receipts (per-step `n_orig` across a whole run). This is captured routes
   from one cell, one capture.
2. **First-64-calls bias.** The capture is the first 64 B4096 calls per
   worker, i.e. early in the run. CORPUS_FINDINGS argues the 31% dummy
   share is not a ramp artifact because it matches the whole-run receipts'
   `uniform_rescued` ~34%; that cross-check covers the dummy *rate* only,
   and says nothing about whether the non-dummy fill distribution is
   representative.
3. **Rank-call granularity, not step granularity.** A serving step is 8
   concurrent rank-calls; the megakernel's slab rendezvous couples them, so
   a step is only skippable if ALL 8 ranks agree the call is dummy. This
   analysis treats calls independently and therefore **over-states tier 1**
   to the extent dummy calls are not rank-synchronous. The corpus does not
   carry a step key that would let this be checked locally. **This is the
   first thing to verify on the node**, and tier 1's design must make the
   skip a collective decision, not a per-rank one.
4. **Material tension with the 37.6% fill receipt.** The serving receipts
   report ~1,539 real tokens/rank/step against 4096 rows = 37.6% fill,
   2.66x padding. This corpus says mean fill **0.69** and an aggregate
   padding multiplier of **1.19x**, with non-dummy calls near-completely
   full. These two numbers cannot both describe the same quantity. Likely
   reconciliations: the receipt fill averages over ALL steps including
   unsealed/decode ones while this corpus is sealed B4096 calls only; or
   the receipt counts prompt tokens while the routed rows include
   per-expert replication. **Until this is reconciled, neither the 2.66x
   nor the 1.19x should be quoted as "the" padding multiplier**, and the
   M24 win estimate stays withdrawn as §A.4.3 requires. The direction of
   the discrepancy matters: if 37.6% is right, tier 2 is valuable; if this
   corpus is right, tier 2 is nearly worthless.
5. **`phi` is assumed, not measured.** 0.80 and 0.90 bracket a guess at the
   row-scaling fraction of MoE-region cost. The mega's fixed costs (single-
   CTA planner, C=28 service CTAs, slab rendezvous, M8 certification) are
   exactly the `1-phi` part, and they are what the §A.4.3 withdrawal was
   worried about. A measured `phi` is a separate gate.
6. **rho is a MoE-region ratio, not end-to-end.** At the measured MoE
   fraction (~40-54% of step time), rho=0.72 in-region is roughly a
   +11-17% end-to-end step win -- well short of the 20-30% target on its
   own. This is a kernel-region model and must never be quoted as a serving
   result.

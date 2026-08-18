# M24 gate: are DP-dummy steps rank-synchronous?

Script: `dummy_synchrony.py` (this directory). Corpus: the same 8 worker
`.npz` files x 64 B4096 calls (512 rank-calls) used by `g0b_local_fill.py`;
the dummy classifier is reused verbatim (modal sorted-top-8-set coverage
> 0.95 => dummy call), so the counts are directly comparable to that script's
160/512 = 31.2%. The corpus path is a session scratchpad and is not checked in.

**Why this matters.** `G0B_LOCAL_FILL_HISTOGRAM.md` §5.3 flags rank-call vs
step granularity as "the first thing to verify": the megakernel's slab
rendezvous couples the 8 ranks, so a **whole-step** tier-1 skip requires all
8 ranks to be dummy on the *same* step. If dummies were per-rank scattered,
tier 1 could only shed a lone rank's send-side work and the wall-time win
would nearly vanish.

**Assumption under test.** The corpus carries no step key. This analysis
assumes **call index `i` is the same serving step in every worker file**, and
then validates that assumption by internal consistency (§3).

## 1. Per-index dummy count `c[i]`

`c[i] = sum_w d_w[i]` over the 8 workers, at raw index alignment (lag 0):

| `c[i]` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| observed indices | **44** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **20** |
| independent-rank null (Poisson-binomial at each worker's own rate) | 3.2 | 11.6 | 18.5 | 16.8 | 9.5 | 3.5 | 0.8 | 0.1 | 0.0 |

**100.0% of indices have `c[i]` in {0, 8}** against a 5.0% expectation under
the independent-rank null. The histogram is perfectly bimodal: there is not a
single index where some ranks are dummy and others are not.

Per-worker dummy rate is **31.2% (20/64) for every one of the 8 workers** —
identical, not merely similar; pooled 31.25%, matching g0b's 160/512.

## 2. Agreement statistics

| statistic | value |
|---|---|
| pairwise observed agreement (28 pairs) | mean **1.000**, min 1.000, max 1.000 |
| pairwise Cohen kappa (chance-corrected at each worker's 31.2% rate) | mean **+1.000**, min +1.000, max +1.000 |

Every one of the 28 worker pairs agrees on all 64 calls. Kappa is exactly 1.0,
so this is not a base-rate artifact.

## 3. Is the index alignment real? (validation of the assumption)

Two independent checks, because §1–2 alone could in principle be produced by a
coincidence of *block-structured* dummy patterns rather than true alignment.

**(a) The dummy pattern is block-structured, so the lag test is weak.**
All 8 workers show the identical mask:

```
w0..w7  DDDDDDDDDDDDDD............................................DDDDDD
        idx 0-13 dummy            idx 14-57 real             idx 58-63 dummy
```

Because the dummies sit in two contiguous blocks, shifting a worker by one
call still leaves 96.8% agreement (lag profile: 0.902 / 0.935 / 0.968 /
**1.000** / 0.968 / 0.935 / 0.902 for lags -3..+3, identical for every
worker). Best lag is **0 for all 8 workers**, but the margin over ±1 is only
3.2 points — so the boolean-vector lag search on its own is *suggestive, not
conclusive*.

**(b) `n_real` is the sharp probe, and it is decisive.** Unlike the dummy
mask, per-call real-row counts vary continuously and non-monotonically across
the run, so a wrong lag destroys the correspondence. Mean `r(w0, w_shifted)`
over w1..w7:

| lag | -3 | -2 | -1 | **0** | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| mean r | +0.580 | +0.633 | +0.647 | **+0.996** | +0.648 | +0.637 | +0.588 |

The peak at lag 0 is sharp (0.996 vs ~0.65 one call away). On the 44
all-non-dummy indices, cross-worker `n_real` correlation is mean **r = +0.974**
(min +0.913); within-index spread across ranks is **48 rows** against an
across-index spread of **871 rows** — i.e. ranks agree on fill an order of
magnitude more tightly than the run varies.

The strongest evidence is the ramp band, where fill is partial and changing
fast:

| index | 14 | 15 | 16 | 18 | 19 | 20 | 21 | 22 | 28 | 29 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w0 `n_real` | 385 | 1094 | 1980 | 1472 | 2969 | 3196 | 3360 | 2209 | 3691 | 3717 | 4096 |
| across-rank std | 21 | 96 | 106 | 89 | 37 | 418 | 303 | 657 | 182 | 182 | 18 |

Eleven partial-fill indices track each other across all 8 ranks at mean
r = +0.956 (min +0.848). **Index alignment is strongly validated** — this is
the same serving step across workers, not a coincidence.

## 4. Verdict

**SYNCHRONOUS.** DP-dummy calls in this corpus are perfectly rank-synchronous:
`c[i]` is bimodal on {0, 8} with zero intermediate indices, kappa = 1.000
across all 28 pairs, and the index alignment those statistics rest on is
independently confirmed by a sharp lag-0 peak in `n_real` correlation
(+0.996 vs ~0.65 at ±1). Under DP with a shared scheduler this is the
expected behaviour — the dummy is a *global* batch-shape decision broadcast to
all DP ranks, not a per-rank routing outcome — and the data matches that model
exactly.

### Scope caveat — read before using this for a wall-time number

The dummies observed here are **entirely ramp-in (idx 0–13) and drain-out
(idx 58–63) blocks**, consistent with the capture window sitting at the start
of the run (`g0b` caveat 5.2: first-64-calls bias). **Zero mid-run dummy calls
appear in this window.** So what is proven is:

* dummy-ness is a rank-synchronous property *of the mechanism*, with an
  8/8-or-0/8 signature and no scattered instances anywhere in 64 steps; and
* the fill signal is rank-coherent step-by-step throughout, including mid-run.

What is **not** proven locally is that the mid-run dummy steps — the ones the
whole-run ~56% receipt rate refers to — carry the same signature. Nothing in
this corpus contradicts it, and the mechanism argument says they should, but
the confirming evidence must come from the node: a `RAGGED_SEAL_RECEIPT` grep
correlating per-step `n_orig == 0` across ranks by step id. Treat that as the
remaining half of this gate.

## 5. Implications for the F.3 ladder and tier-1

1. **The tier-1 skip can be a whole-step skip.** The blocking worry in §5.3 —
   that a step-level skip needs all 8 ranks dummy and the corpus over-states
   tier 1 to the extent they are not — is **not realized in the data**. The
   tier-1 wall-time prediction does not need a synchrony discount; the
   rank-call arithmetic in `g0b_local_fill.py` carries over to steps
   one-for-one at the observed rate.
2. **Design implication unchanged: still make the skip collective.** The
   decision may be *derivable* identically on all ranks, but the mega must not
   rely on that silently — a lone disagreeing rank entering the slab
   rendezvous while seven skip it hangs 8 GPUs (the standing non-unanimous
   collective failure mode). Tier 1 must broadcast/agree the skip (a counted
   arrival or an epoch word carrying the skip bit), with the local
   `n_orig == 0` test as the *proposal*, never the commitment. Synchrony makes
   that agreement cheap and near-always unanimous — it does not make it
   optional.
3. **Tier-1 dominance over tier-2 is reinforced.** With synchrony holding,
   skipped steps are wall-time removed rather than partially shed, so the
   tier-1-dominant conclusion of `G0B_LOCAL_FILL_HISTOGRAM.md` (bimodal
   distribution, non-dummy calls ~0.91 full, weak tier-2 row plumbing) stands
   with a cleaner mechanism story.
4. **Do not yet convert this into a headline speedup.** The rate that the
   wall-time win multiplies against is the *whole-run* dummy rate from the
   receipts (~56% of in-bucket steps), not this window's ramp-phase 31.2%, and
   the `phi` fraction remains assumed rather than measured. Both stay open
   gates; this document discharges only the synchrony question.

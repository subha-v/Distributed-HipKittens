# Ordering-hole probe — the protocol review's rank-#1 residual risk, measured

**Verdict: 12/12 clean. The hole does not fire on this workload.** The static
analysis stands and is not refuted — this is an empirical bound, not a proof —
but every mode-2 `g < 16` number reported tonight is now backed by a direct
sensitivity test rather than by the ordinary gates alone.

## The claim under test

From `protocol_review.md` (rank 1): in mode 2 the row-completion flag is
released by the wave that pushed the **last** slice group of a row, but the
other `16/g − 1` groups of that row were pushed by **different waves on
different CTAs**. `s_waitcnt vmcnt` is per-wavefront, so `flush_pending`'s drain
(`moe_mps_adapter.cuh:247`) covers only the publishing wave's own packets. At
`g = 16` there is exactly one claimant per row and the hole closes; at
`g ∈ {1,2,4}` — the entire A2 sweep and the best-performing point — it is open.

Predicted symptom: **intermittent wrong numbers with `pperr == 0`**, which the
600-epoch soak cannot catch because the soak checks `pperr`, not numerics.

## Test design

Within a single run, `mps_mega` and `pf6gm_mega` consume the **identical**
input, so their `[MOK GATE] max_abs` and `relative` must agree exactly. Any
within-run divergence is the hole firing. `K0_MOK_SEED_BASE` is varied per trial
so each trial exercises a different routing — the review noted the analogous
ICPP'26 effect is amplified by routing skew, so a fixed seed would be the wrong
test.

- 8 trials at `C=64, g=4` (hole **open**)
- 4 trials at `C=64, g=16` (hole **closed**, control)

## Result

| g | trials | seeds distinct | `mps` == `pf6gm` gate values | `pperr != 0` | soak 600/600 |
|---:|---:|---|---:|---:|---:|
| 4 (open) | 8 | yes | **8/8** | 0 | 8/8 |
| 16 (control) | 4 | yes | **4/4** | 0 | 4/4 |

Divergences: **0**. The seed sweep worked as intended — `max_abs` moved across
trials (0.035156, 0.035370, 0.039062, 0.042969), so the trials really were
different problems, and within every one of them the two arms agreed to the last
digit.

Cumulative evidence across the night for the open case: these 8 trials, plus
15/15 campaign correctness gates and 10 × 600-epoch soaks at `C=8,g=2` and
`C=64,g=1`, plus `mps_mega` reporting `max_abs`/`relative` identical to
`pf6gm_mega` in every campaign run.

## Reading

The hole is **real in the source and unobserved in practice**. The most likely
reconciliation, unverified: the row's other slice groups are pushed strictly
earlier in wall-clock than the claiming group (a row is only claimable once all
`16` of its slices have arrived, and arrival is itself ordered by the producing
CTAs' releases), so by the time the last group's wave drains, the other waves'
packets have long since retired. That is a timing argument, not an ordering
guarantee, and it would degrade under different routing skew or a faster fabric.

**Disposition: not fixed, recorded.** The mode-2 path is closed for performance
(exp_03: ceiling ~0.882x production; exp_04: still 1.43x `pf6gm_mega`), so
hardening a protocol on a closed path is not worth the risk of perturbing the
resource tuple. If mode 2 is ever revived, the fix is the one the library
already implies — `counter.cuh`'s `counted_arrive_release_into` docstring warns
about exactly this pattern and the adapter open-codes past it, so the correct
change is to route the row-completion arrival through that primitive rather than
through a per-wave `s_waitcnt` plus a bare flag store.

## Primitives

This is the second finding tonight pointing at the same gap: `counter.cuh` can
express "arrive and release" for a single counter, but the kernel needs
"**all `g` members of this group have arrived, and every writer's payload is
visible**". There is no primitive for a group-scoped release whose drain covers
writers outside the calling wave, and the absence is precisely what forced the
open-coded, per-wave version that carries this hole.

## Artifacts

`~/overnight-scratch/hole_probe.csv`, trial logs `~/overnight-scratch/hole_g*_*.log`.

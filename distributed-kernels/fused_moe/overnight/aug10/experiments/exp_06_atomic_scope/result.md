# exp_06 — how much of the interference floor is the atomic's SCOPE?

**Answer: ~30%.** Relaxing the arrival RMW from `acq_rel` to `relaxed` recovers
**354 µs of the 1,159 µs M7 interference floor** at `g=1` (157 µs at `g=16`).
The other ~70% is not the memory-ordering scope — it is the atomic's cache-line
footprint and the rest of the service pool's memory activity.

**This was a deliberately non-correctness-preserving DIAGNOSTIC and has been
reverted.** It was never a candidate and never entered the ratchet.

## Why this measurement

exp_05 stage 0 established that the service pool inflates M7 by 57–133% through
memory-system interference, and that the interference is **atomic traffic, not
payload** (it rises with `g`, while payload bytes are constant and coarsen). At
`g=1` the group-completion probe loop is degenerate, yet **+1,159 µs of
interference remains** — a `g`-independent floor.

The prime suspect was the per-lane arrival counter at
`moe_mps_adapter.cuh:329-330`: `fetch_add_acq_rel` on
`nc_arr + r*16 + nc` across 32 live lanes per event, hitting **32 scattered
cache lines**, with a count independent of `g`.

That suspect has two separable costs — the **scope** (`acq_rel` forces ordering
and cache maintenance) and the **footprint** (32 scattered lines). A one-word
change to `relaxed` isolates the first.

## Result

At `C=64`, matched against the same build family, with the mode-0 control at
M7 = 2,019.7 µs (192 compute CTAs, no service traffic):

| g | M7 `acq_rel` | M7 `relaxed` | delta | interference floor | floor removed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3,179.0 | **2,825.4** | **−353.6 µs** | 1,159.3 → 805.7 | **30.5%** |
| 16 | 4,713.5 | **4,556.7** | −156.8 µs | 2,693.8 → 2,537.0 | 5.8% |

Whole-kernel effect at `g=1`: 9,976 → **9,640 µs** (−3.4%). Service drain
5,952 → 5,356 µs; combine 2,875 → 2,909 µs (flat).

That the effect is much larger at `g=1` (30.5%) than at `g=16` (5.8%) is
consistent: at `g=16` the floor is a small part of a much larger, probe-loop-
dominated interference, so relaxing one of the two atomic classes moves less of
the total.

## Reading

- **Scope is real but not dominant.** Memory-ordering strength on the arrival
  counter costs ~354 µs of M7. Removing it entirely — which is not legal — buys
  back under a third of the floor.
- **Footprint is the larger half.** ~806 µs of the floor survives with the
  ordering fully removed, so most of the damage is that 32 lanes touch 32
  scattered cache lines per event, independent of how strongly they are ordered.
- **Therefore M4 is well-founded and should beat this bound.** Fleet's per-XCD
  device-scope arrival counters reduce **both** terms at once: they resolve in
  the local L2 (scope) and they collapse many scattered per-`(row, nc)` lines
  into a handful of per-XCD lines (footprint). This experiment says the
  achievable prize is larger than 354 µs and bounded above by roughly the full
  1,159 µs floor.

## Correctness note — do not over-read the green gates

The diagnostic build reported `[MOK GATE] pass=True`, `control_fails=True` and
`[MPS SOAK] 600/600` at both `g` values. **That does not make the relaxed form
correct.** It drops the acquire edge that the group-completion probe loop
depends on, which is exactly the class of defect the ordering-hole probe found
to be *real in source and timing-protected in practice*
(`exp_01_nil_fault/ordering_hole_probe.md`). One screening run per config is
also far below that probe's sensitivity. The green gates are reported for
completeness and are **not** evidence of correctness. Reverted at the source.

## Primitives

Third convergent pointer at the same gap. `counter.cuh` offers
`counted_arrive_into` / `counted_arrive_release_into` — a **per-counter** arrival
vocabulary. Everything measured tonight says the expensive part of a
role-split protocol is not any individual arrival but the **aggregate
cache-line footprint of many scattered arrivals**. A primitive that owned the
*placement* of arrival counters — per-XCD, per-CTA-group, or coalesced into one
line per event — would have made M4 a configuration rather than a rewrite. That
is the single most valuable thing the library could learn from this campaign.

## Artifacts

`~/overnight-scratch/ts3relax_*.log`, `~/overnight-scratch/screen_ts3relax.csv`,
JIT hashes `2bf8a97ecf88` / `4d5084af1f9c`, diagnostic commit `8de5a281`,
reverted immediately after.

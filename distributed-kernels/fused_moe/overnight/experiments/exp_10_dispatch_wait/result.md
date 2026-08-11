# exp_10 — the dispatch boundary has no peer wait to hide. Axis closed.

**Verdict: `chunk_poll success_max = 0` against a spin limit of 2,000,000.**
M2's per-`(source, chunk)` poll never spins — the peer's data has already
arrived by the time this rank looks for it. There is **no communication wait at
the dispatch boundary**, so there is nothing for a dispatch service pool to
overlap. Combined with the combine-boundary closure, **both candidate
boundaries in this kernel are now measured-closed.**

Cost: **zero device code and four lines of print-only harness diagnostic.**

## Why this is the right measurement

`exp_05/design.md` and the independent `scope.md` both confirmed the *exposure*
half of the premise: four grid barriers sit between M2 and M6 (`:937`, `:1059`,
`:1114`, `:1124`) and M6 has no readiness poll of any kind, so there is provably
zero dispatch/compute overlap today. The pre-M6 region measures 0.7–1.25 ms.

**But exposed is not hideable.** A role split can only hide *waiting*; it cannot
hide work that still has to happen — it can only move that work to different
CTAs, which is exactly what the combine-boundary modes do, and they pay the
interference tax for it (`exp_05` stage 0). The scope's static read put **~557
of the 757 µs on this rank's own quantize + push work** rather than peer wait,
and flagged the wait fraction as unmeasured. This experiment measures it.

The instrument already existed and was simply pointed at the wrong arm: M2's
chunk poll records max-spins-to-success into `spin_dbg[0]`
(`e004pf_k0pf_ab.py:1475-1476`), and the harness prints it — but only inside the
`pf6c` structural-control block (`:4555-4560`), which none of our three arms
run. **The same M2 body runs in `pf6gm_mega`**, so this measures the dispatch
wait of the *ratchet kernel*, not merely of the MPS arm.

## Result

| configuration | `chunk_poll success_max` | `fail_max` | spin limit |
|---|---:|---:|---:|
| `C=64, g=1, mode=2` | **0** | 0 | 2,000,000 |
| `C=8, g=1, mode=0` (control) | **1** | 0 | 2,000,000 |

The counter is a running maximum and is not reset between launches, so these
values are maxima over **every** pf6-family launch in the run — warmup, timed
iterations, and the 600-epoch soak. Across all of them, no poll ever spun more
than once.

Both runs gate-green (`[MOK GATE] pass=True`, `control_fails=True`,
`[MPS SOAK] 600/600 pperr=0`).

## Reading

**The 0.7–1.25 ms pre-M6 region is essentially all own-work.** Quantize, pack,
push, unpack, histogram — every microsecond of it is this rank computing, not
this rank waiting. The peers are never late.

That makes sense in hindsight and is worth stating so it is not re-derived: all
eight ranks run the identical kernel on identically-shaped work and enter M1
within a few microseconds of each other, so by the time a rank finishes its own
send and turns around to poll for a peer's chunk, that chunk landed long ago.
The all-to-all is *latency-hidden by symmetry*, not by any mechanism we added.

Consequences:

1. **The dispatch role split is dead, and the multi-hour rewrite of M3–M5 that
   `exp_05` stage 2 would have required is not worth starting.** The scope had
   already shown the COMET layer-0 mechanism is not a cheap reorder (M6's input
   array does not exist until the destination-counting sort completes, and a
   *local* token's sorted row depends on *all remote* counts through the global
   scan). Now we also know the prize it would have bought is ~0.
2. **Both boundaries are closed on measurement.** Combine: the peeled-out pull
   is worth ~300 µs and the ceiling is a tie (`exp_03`), with a 37–57%
   interference tax on the concurrent GEMM (`exp_05`). Dispatch: no wait exists.
3. **This is the cleanest possible form of the night's thesis.** CTA-level
   comm/compute overlap needs communication *latency* to hide. This kernel, on
   this workload, has almost none — its cross-rank traffic is either already
   hidden by symmetry (dispatch) or is bandwidth-bound work that must be done by
   someone (combine). The technique is not refuted; **this layer is simply not a
   workload that has the thing the technique consumes.**

## Scope of the claim

Measured on the MoK synthetic prefill campaign, 8 ranks, one node, uniform
shapes. A workload with **routing skew**, **stragglers**, **heterogeneous ranks**
or **multi-node latency** would produce real waits and could reopen the axis —
`spin_dbg` is exactly the right first instrument to re-run there, and it is now
wired up. The `fail_max = 0` column also confirms no poll ever hit the limit, so
nothing is silently timing out.

## Harness note

Second additive, print-only, exception-guarded diagnostic
(`[MPS SPIN]`), inserted next to the `[MPS TS]` block and mirrored into
`MPS_OVERNIGHT_HARNESS_NOTE.md`. It reads a buffer the harness already allocates
and prints a value the harness already computes. No gate, no arm, no timing path.

## Primitives

Nothing added. One observation: the useful signal here was already being
recorded on device and thrown away because its print sat behind an arm gate.
Diagnostics that cost nothing to compute should not be gated to the arm that
happened to introduce them — a per-arm `debug` block would have surfaced this on
day one and saved the entire dispatch-boundary design detour.

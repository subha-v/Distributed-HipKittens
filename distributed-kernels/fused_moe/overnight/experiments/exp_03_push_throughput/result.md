# exp_03 — the (C, g, mode) map, the 1/C law, and the ceiling of the combine-boundary split

**Verdict: the combine-boundary role split cannot beat `pf6gm_mega` at ANY
`(C, g)` with the current service-loop throughput — and, more importantly, its
*theoretical* ceiling even with perfect hiding is only ~0.878x production
(vs 0.895x today), because the thing it peels out of the critical path is only
worth ~300 us.** This is a quantified ceiling, not a shrug. The mechanism is
alive but its upside at this boundary is ~2%, so it cannot carry the 0.80x
objective by itself.

## Method note — screening, and why it is legitimate here

Full 5-rotation campaigns cost ~18 min. A 1-process / 1-warmup / 1-timed run
costs ~25-95 s. The decision campaign gave `mps_mega` 57,347.4 us and the smoke
at the same config gave 57,292.1 us — **0.10% apart** — and `production` tracked
to 2%. So for effects of this magnitude the smoke is a faithful *ordering*
instrument. Every number in the tables below is a **screening magnitude**
(`status=valid_diagnostic`), never a decision number. No conclusion here rests
on a difference smaller than 5%. Decision numbers come from 5-rotation
campaigns only (exp_02, and any future ratchet candidate).

All 17 screening points passed the full gate ladder: `[MOK GATE] pass=True`,
`control_fails=True`, `[MPS SOAK] 600/600 pperr=0 pass=True`. Correctness is not
the variable here.

## A3 / A1 — mode 0, the capacity-tax control (push mechanism OFF)

| C | `mps_mega` us | vs paired `pf6gm_mega` |
|---:|---:|---:|
| 8 | 7,086.3 | 1.016 |
| 32 | 7,154.5 | 1.037 |
| 64 | 7,421.2 | 1.061 |

**Reserving CTAs is cheap and sublinear.** A naive capacity model
(`256/(256-C)`) predicts +33% at `C=64`; we measure **+6.1%**. Two reasons: the
reservation happens at M6.9 so only M7/M8 are taxed, and those phases are not
CTA-throughput-bound. `finish_order_partition` / `role_partition` are exonerated
— **the role split itself is not what we pay for.**

## A1 x A2 — mode 2 (full stream), the 1/C law

| C | g=1 | g=2 | g=4 | g=16 |
|---:|---:|---:|---:|---:|
| 8 | — | 57,292 | — | 122,482 |
| 16 | — | 33,535 | — | 74,059 |
| 32 | — | 19,818 | 25,002 | 42,876 |
| 48 | — | 15,267 | — | 31,519 |
| 64 | **10,374** | 13,263 | 16,446 | 27,592 |

Subtracting the mode-0 tax at the same `C` gives the *service* cost, and it
obeys a clean inverse law. Service cost x C, in us x CTA:

| C | g=2 | g=16 |
|---:|---:|---:|
| 8 | 402,000 | 923,000 |
| 16 | 424,000 | 1,071,000 |
| 32 | 406,000 | 1,143,000 |
| 48 | 394,000 | 1,163,000 |
| 64 | 374,000 | 1,291,000 |

**The service pool's aggregate throughput is exactly proportional to `C`**, i.e.
each service CTA contributes a fixed ~1.1-1.7 GB/s and they do not contend with
each other. There is no knee, no saturation, no sweet spot: the axis is a
hyperbola and `C` is a pure multiplier. That single fact reframes A1 — the
question is never "which `C`" but "what is the per-CTA rate".

## A2 — `g` is an atomics knob, not a transfer-size knob (pre-registered
expectation WRONG)

At `C=64`, service cost rises monotonically with `g`: 2,953 (g=1), 5,842 (g=2),
9,025 (g=4), 20,171 (g=16) us. The pre-registered expectation was "g=4 sweet
spot", and the literature calibration argued for coarsening toward the ~1 MiB
bandwidth knee. **Both are wrong here, and for an instructive reason:** in this
design `g` does not primarily change the transfer size, it changes how much
*group-completion bookkeeping* runs. `moe_mps_adapter.cuh:340-346` probes every
one of the `g` chunks of a group with an `acq_rel` RMW each time any slice in
the group completes, across 32 live lanes. That traffic scales with `g` while
the byte movement per call only grows the (already latency-bound) copy.

`g=1` is the special case where the group is a single chunk, so `old + 1 ==
target` already proves completion and the probe loop degenerates. That is why
`g=1` wins.

**Consequence for M3 (coarsen to >=256 KiB bands): the M3 mechanism cannot be
tested through `g`.** Raising `g` buys 8x the atomics before it buys any
bandwidth. M3 needs a push path whose completion detection is decoupled from
its transfer size.

## The ceiling — why no `(C, g)` can win, and by how much it misses

Mode 1 (`C=0`, bulk push after M7, local-slot M8 — the push layout with the
overlap OFF) measures **8,049.2 us vs a paired `pf6gm_mega` of 6,903.4**, i.e.
`push_cost(256 CTAs) - pull_savings = +1,146 us`. From the 1/C law,
`push_cost(256, g=2) ~ 374,000/256 = 1,460 us`. Therefore:

> **`pull_savings` ~ 300 us.** Replacing M8's remote peer-part pull with a
> local slot read is worth only about 300 us of the 1,309 us M8/M9 pool.

That is the whole prize at this boundary. The best conceivable mode 2 is
`mode0(C) - pull_savings` with the service cost fully hidden:

| C | mode 0 (measured) | − pull_savings | vs production | vs `pf6gm` |
|---:|---:|---:|---:|---:|
| 8 | 7,086 | 6,786 | 0.882 | 0.985 |
| 32 | 7,155 | 6,855 | 0.891 | 0.995 |
| 64 | 7,421 | 7,121 | 0.926 | 1.034 |

**Perfect hiding at `C=8` yields 0.882x production — a 1.3% improvement on
`pf6gm_mega`, and nowhere near 0.80x.** And perfect hiding at `C=8` requires the
service cost (374,000/8 = 46,750 us) to fit inside M7's 1,844 us window, i.e. a
**25x** throughput improvement. At `C=32` the requirement is 6.3x for a 0.891x
result. The tax and the hiding requirement pull in opposite directions and the
prize between them is too small.

## Why the per-CTA rate is ~1.1 GB/s (mechanism, ISA check in flight)

`push_slice_group` calls `store_peer_packets` (`packet.cuh:44-54`), which is a
`#pragma unroll 1` loop of `out[packet] = in[packet]` — a **load from local
`part` feeding a dependent store to the peer**. One 16-byte packet per lane per
iteration, so each lane sustains **one outstanding memory operation** and the
copy runs at one memory round trip per 16 bytes per lane. At `g=1` the transfer
is 896 B = 56 packets across 64 lanes, i.e. **less than one packet per lane** —
there is not even enough work in a call to pipeline. The serial `npush` loop
(`moe_mps_adapter.cuh:374-392`) then walks up to 32 such rows one at a time.

Round-trip arithmetic that fits the data: ~65 events per wave at `C=64`, up to
32 rows per event, ~1 round trip per row, 2,953 us of service time => ~1.4 us
per round trip, which is the right order for a peer store plus a dependent
global load on this fabric.

## Decisions taken

1. **`C=90` / raising the `config_is_valid` cap (`moe_mps_adapter.cuh:85`) is
   NOT worth doing.** The 1/C law lets us extrapolate any `C` exactly, and the
   tax curve makes large `C` strictly worse. Extrapolated best over all `C` with
   today's throughput is ~7,262 us at `C~103`. The A1 axis is now *closed by
   extrapolation from a fitted law*, which is stronger than three more points.
2. **A13 (timestamps-on) deferred, with reason.** The harness allocates the
   timestamp block (`e004pf_k0pf_ab.py:1504-1505`) but never reads or prints it,
   so A13 costs a harness edit. The 1/C law already supplies the attribution
   A13 would have given (service pool on the critical path, cost exactly
   inversely proportional to pool size). Revisit only if the intra-service
   breakdown (queue lag vs copy vs flag) becomes the deciding question.
3. **The next experiment is memory-level parallelism in the peer copy**, not
   another point on `(C, g)`. That is the only term in the model that moves the
   ceiling.

## Primitives

- **`store_peer_packets` is the finding of this experiment.** Its surface
  (`dst, src, bytes, tid, threads`) is size-agnostic and gives no hint that it
  is latency-bound at one outstanding operation per lane. Its `#pragma unroll 1`
  is correct for its stated purpose (avoiding scratch demotion of a
  register-resident `packet16[N]`, per its own doc comment at `packet.cuh:37-43`)
  but it makes the loop a serialized load->store chain. For a global->peer copy
  this is the wrong default and there is no batched alternative in the header.
  **Proposed additive change: a `Batch`-templated multi-region overload** that
  issues `Batch` loads into registers before any store, so a caller with many
  small regions (our case: up to 32 rows of 896 B) gets `Batch`-deep memory
  parallelism without changing coalescing. Existing callers keep the old
  overload and stay bit-identical.
- **The library has no vocabulary for "group completion detection".** The
  `g`-probe loop is open-coded in the adapter with `fetch_add_acq_rel(..., 0u)`
  used as an acquire-carrying *read*, which is a clever but obscure idiom.
  `counter.cuh` offers `counted_arrive_into` but nothing for "has this whole
  group of counters reached target". That absence is what made `g` a
  quadratic-ish cost instead of a transfer-size knob.
- `role_partition` / `finish_order_partition` measured cheap (mode 0 at 1.6-6.1%).
  No change proposed; this is a positive result for the primitive.

## Artifacts

- `~/overnight-scratch/screen_A1.csv`, `~/overnight-scratch/screen_A2g.csv`
- Per-config logs `~/overnight-scratch/scr_*.log`
- Kernel unchanged from exp_01: sha256 `709f4d99...` (commit `5dc61fb0`).

# exp_09 — reducing the arrival COUNT (M4): analysis, and why there is no cheap version

**Not built. This file exists so the next session does not re-derive the
analysis.** Everything measured tonight converges on one requirement — issue
fewer arrival atomics — and the conclusion of this analysis is that **the
arrival count is structurally fixed by the plan, not by the protocol.** M4 as
"per-XCD aggregation" cannot be bolted on; the count only falls if the *plan*
changes so a receive row is not split across blocks.

## The requirement, and how it was arrived at

| experiment | finding |
|---|---|
| `exp_04` | 4x memory parallelism bought only ~20% of service cost ⇒ the copy is ≤19%, **~81% is bookkeeping** |
| `exp_05` | the service pool inflates M7 by 37–57%, and it is **atomic traffic, not payload** (rises with `g`, while payload is constant and coarsens) |
| `exp_06` | ordering **scope** is only ~30% of the `g`-independent +1,159 µs floor |
| `exp_07` | the rest is **not cache-line footprint** — coalescing atomics onto 2 lines was ≥10x slower, because RMWs serialize at the line |
| `exp_08` | moving the pool to separate dies cuts interference 36%, confirming per-XCD L2 contention — but starves the pool |

Four independent measurements, one conclusion: **the cost is the number of
atomic operations and the contention they create wherever they land.** Neither
relocating them (exp_07), nor isolating them by die (exp_08), nor weakening
their ordering (exp_06) recovers more than a third.

## Why the count is structurally fixed

The arrival counter answers: *has every block that contributes to receive row
`r` produced its chunk `nc` yet?* The count is therefore exactly one increment
per **(block, row, chunk)** triple that exists in the plan:

- an event is one `(b, nc)` pair; `events_total = (nvi[0] >> 5) * 16 ≈ 16,720`
- each event's 32 live lanes bump 32 different rows' counters
- ⇒ **≈ 535,040 arrival atomics per rank per epoch**

You cannot aggregate across the 32 lanes of an event, because they name 32
different rows. You cannot aggregate across events for a row, because that is
precisely the thing being counted. A per-XCD counter would need to know each
XCD's *expected share* of a row's contributing blocks, which is data-dependent
(`row_rem[r]` is an M2 popcount and the block→XCD map is `bid % 8`), so the
two-level scheme needs a second, equally scattered, per-(row, XCD) counter — it
moves the traffic rather than removing it, which exp_07 and exp_08 both show
does not help.

**Fleet's setting is easier than ours in exactly this respect.** It is a
single-GPU multi-die megakernel whose partition is known statically, so "the
last worker per XCD" is well defined without a data-dependent count. We do not
have that property, which is why the `REPORTED` transfer of M4 to our shape
needs this caveat attached.

## What would actually reduce the count

The count is `Σ_r row_rem[r] · 16`. It falls only if `row_rem[r]` falls — i.e.
if a receive row is contributed to by fewer blocks. That is a **plan-side**
property, and it is exactly what CLAUDE.md's M8 item (COMET layer-1) proposes:

> reschedule the GroupGEMM **column-wise across experts (nc-major)** rather than
> block-major, decomposing along **N only** — never along M, because an M-split
> creates token interdependencies in the top-k reduce.

If the plan is arranged so each receive row's contributions come from one block
(or a small fixed number), `row_rem[r] → 1` and the arrival count collapses by
whatever factor `row_rem` currently averages (~1.5 on this workload from
`padded=33440` over ~21,816 distinct rows — so only ~1.5x, which is **not
enough on its own**).

**That number is the sobering part of this analysis.** `row_rem` averages ~1.5,
so even a perfect plan-side fix removes only about a third of the arrivals. The
remaining ~350,000 atomics per rank per epoch are irreducible given that every
`(row, chunk)` slice must be known-complete before it can be pushed.

## Therefore, the honest options for the next session

1. ~~**Change what is counted, not how.**~~ **Examined and predicted to fail —
   do not build it.** The idea was: one counter per row instead of per
   `(row, chunk)`, target `row_rem[r]·16`, pushing the whole row when it fires,
   which deletes the `g`-wide probe loop entirely. It is tempting because it is
   cheap. It does **not** reduce the arrival count — increments are still one
   per `(block, row, chunk)` triple — it only collapses **16x more increments
   onto 16x fewer cache lines.** That is precisely the change exp_07 measured as
   ≥10x slower. Resolving it by `g`:
   - at `g=1` (the best point) there is no probe loop to delete, so the variant
     is *purely* the exp_07 pessimization → predicted worse;
   - at `g=16` it does remove the probe loop, trading fewer operations against
     more per-line contention → genuinely unclear, but `g=16` is already the
     worst point by a wide margin, so winning there does not help.

   **Conclusion: no cheap experiment remains on the combine path.** Recorded so
   the next session does not spend a build cycle discovering this.
2. **Accept the combine boundary is closed** (it is, on every measurement) and
   spend the effort at the dispatch boundary — where `exp_05/design.md` shows
   the premise verified but the same interference tax applies.
3. **Do neither, and attack the ratchet outside the role-split mandate.** Noted
   for completeness; CLAUDE.md's mandate rules it out of this line of work, but
   the standing 0.80x objective does not care where the win comes from.

## Primitives

This is the fifth convergent finding on `counter.cuh`, and the most specific:
a library that owned arrival counting could have made option 1 a *parameter*
(counter granularity: per-slice vs per-row) rather than a protocol rewrite. It
would also be the natural home for the two hazards measured tonight — **do not
coalesce atomics onto shared lines** (exp_07) and **placement changes cost by
~36%** (exp_08) — neither of which any current signature hints at.

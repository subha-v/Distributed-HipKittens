# exp_12 — dynamic event ticket + compute-CTA fall-through

**Verdict: the largest single improvement of the campaign. `mps_mega` went from
1.464x `pf6gm_mega` to 1.077x — a 27% cut — from two small changes.** It also
made the arm beat `production` (0.960x) for the first time.

## What motivated it — the phase profile, not a hunch

Before this, every role-split experiment had been tuned against whole-kernel
time. Splitting the kernel into phases (exp_11) showed the reservation was being
paid in the worst possible place.

Phase profile of the near-parity kernel (mode 0, C=2; 6,989 µs against a paired
`pf6gm_mega` of 7,051 — statistically the same kernel):

| phase | µs | share | moves when 62 of 256 CTAs are removed? |
|---|---:|---:|---|
| dispatch (M0–M2) | 1,193 | 17% | — |
| plan (M3–M5) | 415 | 6% | — |
| **M6 (GEMM 1)** | **2,539** | **36%** | **no — 2,954 → 2,970, +0.5%** |
| M7 (GEMM 2) | 1,586 | 23% | **yes — 1,586 → 2,020, +27%, linear** |
| combine (M8/M9) | 1,255 | 18% | roughly no |

**The entire capacity tax is M7's.** `plan+M6` is flat in `C` across every point
measured (C = 2, 16, 32, 48, 64 → 2,954 / 2,962 / 2,992 / 2,968 / 2,970 µs), so
M6 has idle CTA capacity, while M7 is CTA-starved and scales nearly linearly.

The old design reserved the pool at **M6.9** — precisely the boundary where the
CTA-rich phase ends and the CTA-starved phase begins. It paid the tax in the one
phase that could not afford it and collected nothing in the phase that could.

## The two changes

**1. The event loop claims a ticket instead of walking a static stripe.**
`run_service` used `for (k = wave_global; k < events_total; k += waves_total)`
with `wave_global`/`waves_total` derived from `(service_id, service_count)`. It
now does `k = fetch_add_relaxed(env.ev_next, 1)` against a new
`K0P6_MPS_ST_EVNEXT` state word.

**2. Every CTA in a stream mode enters the drain.** The M7.6 guard dropped
`is_service_cta`, so service CTAs arrive immediately (they skipped M7) and
compute CTAs fall through as they finish M7.

Change 2 is only legal because of change 1: with a static stripe, exactly-once
consumption depends on the pool size, so a late joiner would double-count
arrivals and corrupt completion detection.

## Why the ticket is better independently of fall-through

Producers enqueue with a **monotonic tail ticket**, so the queue fills densely
in *completion order*. Claiming the next unclaimed slot therefore takes the next
event that will be ready. The static stripe forced each wave to wait for **its**
events in index order — head-of-line blocking behind a slot whose producing CTA
had not run yet, while ready events sat unconsumed further down the wave's own
stride.

Cost: one relaxed atomic per event, ~16,720 per rank per epoch, against the
~1.58M the arrival protocol already spends. Free at our resolution.

## Result — screened, g=1, mode 2, all gate-green

| C | before | **after** | speedup | vs `pf6gm` after |
|---:|---:|---:|---:|---:|
| 4 | — | 7,563 | — | 1.085 |
| 8 | — | 7,578 | — | 1.096 |
| 16 | 22,068 | **7,617** | **2.90×** | 1.097 |
| 32 | 14,775 | **7,504** | **1.97×** | 1.069 |
| 64 | 9,976 | **7,432** | **1.34×** | **1.060** |

Decision campaign at C=64 (dec06, 5 rotations, 3 paired arms, all gates green):
`production 7,728.5` / `pf6gm_mega 6,890.9` / **`mps_mega 7,420.7`** =
**0.960x production, 1.077x pf6gm**.

Resource tuple unchanged: SGPR 104 / VGPR 256 / AGPR 256 / scratch 128 B /
LDS 155,428 B.

## The C curve flattened, and that is the point

Before fall-through the service cost obeyed a strict 1/C law and total time fell
monotonically with C. After it, totals are nearly flat (7,432–7,617 across
C = 4…64) because the drain is no longer limited by the reserved pool — it is
limited by when M7 releases its CTAs. `C` now only controls **how much push is
overlapped with M7**, which is why larger C still wins slightly, and why the
curve eventually turns over when M7 interference outgrows the combine saving
(exp_13: C=96 → 7,157, C=128 → 7,614 after exp_14).

## Provenance

Compute-CTA fall-through is in COMET's **released code** and **not in the
paper** — the paper never mentions a synchronization primitive at all. Credit to
the literature read in `LIT_COMET_specialization.md`. The ticket is ours; COMET's
compute CTAs fall through into a routine whose readiness is per-rank-flag based,
so they never needed one.

## Primitives

`roles.cuh`'s `role_partition` gives `is_service` / `is_compute` and dense ids.
This experiment shows the dense-id half is a **liability** for any protocol that
wants late joiners: baking `service_count` into the work distribution is what
made the pool size a correctness parameter. **A work-claiming primitive — a
shared monotonic ticket over a bounded work set — is the missing piece, and it
is what makes a role split elastic rather than static.** `counter.cuh` has
`counted_arrive_into` (many-to-one arrival) but nothing for one-to-many work
claiming. That is the single most reusable thing this campaign has produced.

# exp_14 — deleting two provably-redundant atomics: the ratchet event

**Verdict: `mps_mega` now BEATS `pf6gm_mega`.** Decision campaign dec07,
5 rotations, 3 paired arms, every gate green:

| arm | µs | vs `production` |
|---|---:|---:|
| `production` | 7,729.4 | 1.000 |
| `pf6gm_mega` (previous ratchet) | 6,910.9 | 0.89411 |
| **`mps_mega` (CTA-specialized)** | **6,866.1** | **0.88831** |

`mps_mega / pf6gm_mega = 0.99351`. **The margin over `production` widened for
the first time in the campaign: 0.894 → 0.888.**

## The argument — two atomics that cannot change any outcome at `g = 1`

Per completing slice the service loop issued four atomics: the arrival
`fetch_add_acq_rel`, a group-completion probe, an `atomicOr` claim, and a
`fetch_add` on `pushed`. At `g = 1` the middle two are dead:

**The probe is dead.** `group_base = nc & ~(g-1) = nc` and the loop runs for
`n2 = nc` only — it re-reads *the very counter* whose `fetch_add` just returned
`target - 1`. It can only return `target`. Its second stated purpose is to carry
an acquire edge for every chunk of the group; in a one-chunk group the wave
already holds that edge from its own `acq_rel` on the same address.

**The claim is dead.** Its job is to make the pusher unique. RMW order makes
`old + 1 == target` true for **exactly one lane grid-wide**, so uniqueness is
already established before the `atomicOr` runs and the claim cannot change
`push_lead`.

Both are skipped under `if (g == 1u)`. **`g > 1` keeps the original path
byte-for-byte** — this is not a weakening of the protocol, it is the removal of
two operations that are provably no-ops in one configuration.

## Atomic accounting, per rank per epoch

| site | before | after (g=1) |
|---|---:|---:|
| arrival `fetch_add_acq_rel` on `nc_arr` | 535,040 | 535,040 |
| group-completion probe | ~349,056 | **0** |
| `atomicOr` claim | ~349,056 | **0** |
| `fetch_add` on `pushed` | ~349,056 | 349,056 |
| event ticket (exp_12) | 16,720 | 16,720 |
| **total** | **~1.58M** | **~0.90M (−44%)** |

## Result — the interference fell exactly as the model predicted

Screened at g=1, mode 2, against exp_12's numbers for the identical configs:

| C | exp_12 total | **exp_14 total** | M7 before → after | combine |
|---:|---:|---:|---|---:|
| 32 | 7,504 | **7,076** | 2,689 → 2,496 (−193) | 880 |
| **64** | 7,432 | **6,949** | 3,141 → 2,835 (**−306**) | 446 |
| 96 | 7,598 | 7,157 | 3,762 → 3,290 (−472) | 108 |
| 128 | 8,049 | 7,614 | 4,120 → 3,718 (−402) | 210 |

M7 — the phase the service pool interferes with — improved at every point, and
by more at larger C where there are more service CTAs issuing atomics. That is
the signature the atomic-traffic model (exp_05/06/07) predicts, and it is why
this change was chosen over anything touching the payload path.

## Where the remaining headroom is

At the winning point every phase is at or better than the homogeneous baseline
**except M7**:

| phase | `mps_mega` C=64 | `pf6gm_mega` | delta |
|---|---:|---:|---:|
| plan | 413 | 415 | −2 |
| M6 | 2,588 | 2,539 | +49 |
| **M7** | **2,835** | **1,586** | **+1,249** |
| combine | 446 | 1,255 | **−809** |

**+1,249 µs of M7 interference is still on the table.** Eliminating it entirely
would put the kernel near **5,617 µs ≈ 0.727x production**. The combine is
already almost fully hidden (446 µs, and only 108 µs at C=96), so there is
nothing left to win there — the whole remaining prize is the interference.

## Correcting exp_09 — per-row counters are NOT predicted to fail

`exp_09/design.md` argued that a single arrival counter per row would be an
exp_07-style pessimization, because it collapses 16x more increments onto 16x
fewer counters. **That conflated *fewer counters* with *fewer cache lines
touched per event*.** The 32 live lanes of an event address 32 **different
rows**, so per-row counters keep exactly the same 32-lines-per-event spread that
exp_07 showed is essential; only the total atomic count falls — the `pushed`
counter disappears entirely, taking ~349,056 atomics with it.

Per-row arrival counting (target `16 * row_rem[r]`, push the whole 14,336 B row
when it fires) is therefore the **next experiment**, not a predicted failure. It
trades some push/compute overlap for a further ~39% atomic cut, and it makes the
transfer one contiguous row instead of sixteen 896 B slices, which also helps
the copy's per-call overhead.

## Primitives

The two deleted atomics existed because the protocol was written for the general
`g` and no one re-derived it for `g = 1`. A `counter.cuh` that owned
group-completion would have specialised this automatically — the caller would
have declared "group of size g" and the library would have emitted the
degenerate form for `g = 1`. This is the **sixth** convergent finding pointing at
the same missing primitive, and the first where the absence cost measurable
runtime rather than just clarity.

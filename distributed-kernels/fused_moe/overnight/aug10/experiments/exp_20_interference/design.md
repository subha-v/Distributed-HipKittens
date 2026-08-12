# exp_20 — design

## 1. What is being changed, and what is deliberately not

Three diagnostic `mode` values are added. Nothing else about the kernel moves:
no descriptor slot, no host ABI, no harness file, no primitive signature. The
`mode` field is 8 bits and both the Python parser and `encode_config` treat it
as a plain integer; the only gate is `config_is_valid` in the adapter. This is
the same additive route mode 3 took in exp_08.

```
mode 0  reserve C, pool idles into the M7.5 rendezvous        (matched control)
mode 2  reserve C, pool streams the real payload              (the ratchet)
mode 4  = mode 2 + pacing delay in the push path                        [E1]
mode 5  = mode 0 + synthetic traffic generator on the pool              [E2]
mode 6  = mode 0 + LDS-only spin on the pool                            [E3]
```

### Field reinterpretation, scoped to the diagnostic mode

| mode | field | meaning inside this mode | why it is free |
|---|---|---|---|
| 4 | `flush_rows` (1..64) | `1 + pacing units` | the real flag-batch depth is pinned at 16 by `effective_flush_rows`, so mode 4 at pacing 0 is bit-identical to mode 2 at `flush_rows=16` |
| 5 | `g` (1,2,4) | traffic variant: 1 read+write, 2 read-only, 4 write-only | the real service loop does not run in mode 5, so `g` has no other consumer; `g=16` is rejected by validation to keep the selector unambiguous |
| 5 | `pull_fallback` | destination is this rank's own slots, not a peer's | `pull_fallback` only ever selects the M8 read path, and mode 5's M8 is already the pull path |
| 5, 6 | `flush_rows` | active window, units of 100 µs | same as above — no other consumer |

## 2. Why modes 5 and 6 are correctness-preserving

The one structural fact that makes this work: **mode 0's M8 pulls the
producer's remote `part` row and never reads `K0P6_D_MPS_SLOTS`.** In mode 0 the
slots buffer is dead memory for the whole epoch. Modes 5 and 6 keep mode 0's
publication (M7.5 bulk rendezvous + row_ready stripes) and mode 0's M8 pull, so:

- the generator **reads** `part`, which it does not own and does not modify;
- the generator **writes** `slots`, which nothing in these modes reads;
- readiness, arrival counting and the combine are byte-for-byte mode 0's.

Result: the diagnostics run the full gate ladder (`[MOK GATE]`,
`control_fails=True`, 600-epoch soak, `pperr == 0`) rather than being labelled
garbage builds. Gate status is reported as a fact, not as evidence that the
*mechanism* is sound — but here there is no mechanism to be unsound about,
because no protocol edge changed.

There is one benign race by construction: compute CTAs are writing `part` (M7's
epilogue accumulates into it) while the generator reads it. The generator
discards every value it loads. It exists to move bytes.

### The one non-diagnostic edit this forced

`k0p6_mps_task_done` selected the event-enqueue path with `mode >= 2`. Modes 5
and 6 must take the *parity* `part_done` arm, so that test is now the closed
interval `2..4`. Without it, modes 5/6 would have enqueued tile events that
nobody consumes and skipped the `part_done` bump M0's zeroing contract expects.

## 3. Matched-CTA discipline

Every diagnostic number is differenced against **mode 0 at the same `C`**, and
both are measured **in the same build**. The diagnostic code costs +16 B/lane of
scratch (128 → 144); re-measuring mode 0 and mode 2 in the diagnostic build is
what prevents that from being read as a result. `SGPR 104 / VGPR 256 / AGPR 256
/ LDS 155,428` are all unchanged, and every added byte is outside both MFMA
K-loops (all of it is in post-M7 code).

## 4. E1 — the discriminator, stated before the run

One pacing unit is `s_sleep 4` = 256 core clocks ≈ 0.12 µs at 2.1 GHz.
`s_sleep` idles the wave and issues no memory request, so pacing lowers the
pool's request **rate** at constant total **volume** — which is the entire point.

The measured inputs per pacing point are `M7`, `combine`, and `servicedrain`
(the DRAIN stamp minus M6_DONE, i.e. how long the pool took to move all of it).
Total pushes `N` is fixed by the plan and independent of pacing. So:

```
bytes moved concurrently with M7   ∝  N · min(drain, M7) / drain
interference per concurrent byte   =  ΔM7 / (N · min(drain, M7) / drain)
```

- **Volume-driven** ⇒ interference is a linear per-byte tax, so
  *interference per concurrent byte is constant across the pacing sweep* and
  M7 + combine is conserved: pacing only moves work from one phase to the other.
- **Rate-driven** ⇒ the marginal cost of a byte falls as the request rate falls,
  so *interference per concurrent byte decreases with pacing* and
  **M7 + combine has an interior minimum** — there is an optimum a scheduler
  could find.
- The strong null (M7 does not recover at all at any pacing) would say the
  interference is not even proportional to concurrent bytes, which would kill
  both readings and point at something structural.

**Built-in self-check:** mode 4 at pacing 0 must reproduce the mode-2 reference
within the screening band. If it does not, the pacing scaffold itself perturbed
the kernel and nothing downstream is interpretable.

## 5. E2 — the variant matrix and what each difference isolates

All five run at `C=64` with an identical 1,600 µs window, identical row
striping, identical liveness filter, identical 896 B unit, identical
`#pragma unroll 1` MLP-1 loop shape. The read-only variant XORs each loaded
16 B packet into an accumulator, which both keeps the loads alive and
reproduces the copy's one-outstanding-load-per-lane dependency; the accumulator
is sunk into an empty `asm volatile`, so it never becomes a store.

| cfg | reads `part` | writes | isolates |
|---|---|---|---|
| `g=1` | yes | peer slots | the full pusher — the reference for the split |
| `g=2` | yes | — | the read side alone |
| `g=4` | — | peer slots | the write side including the fabric |
| `g=1,pull_fallback=1` | yes | local slots | read + write with the fabric removed |
| `g=4,pull_fallback=1` | — | local slots | the write side with the fabric removed |

Differences that name a mechanism:

```
read side            =  ΔM7(g=2)
local write side     =  ΔM7(g=4,pf=1)
fabric surcharge     =  ΔM7(g=4) − ΔM7(g=4,pf=1)
additivity check     =  ΔM7(g=1) vs ΔM7(g=2) + ΔM7(g=4)
```

Sub-additivity would itself be evidence for a shared queue rather than two
independent taxes.

## 6. Window sizing, and the bounded waits it protects

The window is 1,600 µs, **strictly below the mode-0 M7 of 2,020 µs**. Two
consequences, both wanted:

1. The pool always finishes before the compute CTAs reach the M7.5 rendezvous,
   so no grid barrier, no `row_ready` poll on any rank, and no M9 arrival is
   made to wait longer than it does in mode 0. None of the kernel's bounded
   spin limits are put at risk by a diagnostic.
2. Every variant occupies the *same* wall-clock window whatever its throughput
   (the row sweep wraps until the window expires). The comparison is therefore
   "what does running this traffic class flat out for 1,600 µs next to M7
   cost", which is exactly the regime the real pool is in — not "what does one
   sweep of this variant cost", which would confound damage with duration.

The coverage fraction (window / M7) is reported per variant, since M7 itself
moves.

## 7. Invariants asserted

- Role split, task stride and dense compute id are untouched: modes 4/5/6 all
  take the non-mode-3 branch of `is_service_cta` / `compute_id_of` /
  `service_id_of`, so the M7 task space is covered exactly once.
- `TS_M7_DONE` is stamped by compute CTAs before the M7.5 rendezvous, so the
  M7 measurement cannot absorb any delay the diagnostic pool causes downstream.
- The generator writes only within `slots[cur][pos]` for `pos < MAXTOK`, the
  same address the real pusher computes.
- `wave_scratch`, the event queue, the arrival counters and every release /
  acquire pair are unmodified.

## 8. Alternatives rejected

- **Garbage builds (substitute the real pusher's copy, discard correctness).**
  The instructions permitted it, but routing the diagnostics through mode 0
  keeps the whole gate ladder green for free, which removes any argument about
  whether a failing gate also invalidated the timing.
- **Pacing by capping outstanding pushes per wave.** The copy already runs at
  MLP = 1 (exp_03 ISA read), so there is nothing to cap — a delay loop is the
  only rate knob available.
- **A busy-wait ALU loop instead of `s_sleep`.** `s_sleep` is the only form
  that provably issues nothing and cannot be strength-reduced.
- **Reinterpreting `flush_rows` as pacing without pinning the real depth.**
  That would move two variables at once; `effective_flush_rows` pins the real
  depth at 16 so mode 4 pacing 0 is a true replica of the ratchet point.

## 9. Modes 7 and 8 — added mid-experiment, and why

The mode-5 matrix answers "what does traffic class X cost when driven flat out",
but it cannot answer "what does the real pusher's traffic cost" — each variant
runs at its own natural rate, and the rates differ by ~6× (§5's read-only and
write-only twins are not rate-matched, and cannot be without a calibration
constant nobody has measured). Two modes were added to close that gap, both
using the real event-driven rate so no matching is needed.

**Mode 7 — payload-free stream.** Mode 2 with `push_slice_group` and
`push_slice_group_batch` skipped, everything else byte-for-byte: same events,
same arrival atomics, same queue spin, same flags, same fences. It stays correct
because `pull_fallback` makes M8 read the producer's remote `part` row instead
of the slot the pool would have filled, and its matched reference is
**mode 2 + `pull_fallback`**, which performs identical work *plus* the copy. One
subtraction, one variable, no model.

**Mode 8 — event-queue poll backoff.** Mode 4's fit showed the pool is
event-starved: only ~240 of ~1,363 pushes per wave sit on the drain's critical
path, so for most of M7 the 256 draining waves are in `wait_event_nonempty`,
each issuing a relaxed load on its queue slot and one on the single shared
`pperr` word every ~0.12 µs. Consecutive tickets are consecutive 32-bit slots,
so ~16 waves share a cache line that compute CTAs are concurrently writing.
Mode 8 adds `pace_delay(backoff)` to that loop and nothing else; unlike push
pacing it costs nothing when events are available, so it is a candidate
optimization as well as a diagnostic.

Both reuse `pace_delay`, which is why it was hoisted above
`wait_event_nonempty`. A mode 9 (weaken `flush_pending`'s
`thread_release<system>` to agent scope, the last untested protocol term) was
written but not run — see the ownership-collision note in `result.md`.

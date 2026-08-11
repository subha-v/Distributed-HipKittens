# Overnight status — the two-minute morning read

Written 2026-08-11 ~09:45 UTC. Full detail in `experiments/LESSONS.md`
(append-only) and the per-experiment `result.md` files.

## The headline

**The fault is fixed and the mission's question is answered. The margin over
`production` is NOT yet wider than it was — it is unchanged at 0.895x — and the
reason is now measured rather than guessed.**

| arm | µs (5-rotation median rank-max p50) | vs `production` |
|---|---:|---:|
| `production` | 7,731.2 | 1.000 |
| **`pf6gm_mega` — still the ratchet** | **6,904.1** | **0.893** |
| `mps_mega`, best point (C=64, g=1, mode 2) | 10,075.8 | 1.303 |

Final verification campaign `dec05` on the shipped tree, 15 correctness gates,
5/5 negative controls, 5/5 600-epoch soaks, zero faults. Four full decision
campaigns were run tonight (`dec01` C=8 → 57,347; `dec02` C=64 pre-MLP →
10,643.3; `dec03` C=64 post-MLP → 10,107.0; `dec05` final → 10,075.8). A fifth,
`dec04`, was **discarded** — a node resync mid-campaign mixed two kernels into
one summary; the guard for that is now in the driver.

## What was established

1. **The `address (nil)` fault is fixed** (`exp_01`). One-line restoration to
   reference parity: `pbase[t]` was assigned only inside the `j2 < fanout[t]`
   guard while the consumer shuffles all 8 lanes unconditionally. Passed the
   full ladder on the first world-8 run and is resource-neutral.
2. **`mps_mega` does not beat either baseline** (`exp_02`). At the config named
   in CLAUDE.md it was **8.33x** slower; at the best point in the whole swept
   space it is **1.46x** `pf6gm_mega`.
3. **The combine-boundary role split is closed, with a measured ceiling**
   (`exp_03`). Mode 2 obeys an exact 1/C law — service cost x C is constant — so
   the axis is a hyperbola with no sweet spot, and A1 is closed by extrapolation
   from a fitted law rather than by more campaigns. The decisive number is mode
   1: **peeling M8's remote pull out to a local slot read is worth only ~300 µs**
   of the 1,309 µs M8/M9 pool. Even with the service pool perfectly hidden the
   boundary tops out at **~0.88x production**, and reaching that needs a
   6-25x throughput gain it cannot have. `pull_fallback` (push vs pull) lands
   within 2%, confirming transport is not the lever.
4. **The one real speedup found is +5.0% on the MPS arm** (`exp_04`): batching
   independent regions in the peer copy so every load issues before any store.
   Kept. It does not move the ratchet.
5. **The most useful measurement is a negative space.** 4x memory-level
   parallelism bought only ~20% of the service cost, so the copy is at most ~19%
   of what the service pool does and **~81% is bookkeeping**. That redirects the
   M-series away from the byte-moving ablations (M3, M1, M2) toward the
   atomic/fence ones (M4, A10).

## The mechanism finding that matters most

**A service CTA never steals an MFMA issue slot, and slows M7 by 57% anyway.**
Two configurations run M7 on *exactly the same 192 compute CTAs*; the only
difference is whether 64 service CTAs are concurrently moving payload:

| | M7 | delta |
|---|---:|---:|
| 254 compute CTAs, no service | 1,609.7 µs | — |
| **capacity**: 192 CTAs, no service traffic | 2,019.7 µs | +410 µs (+25.5%) |
| **interference**: same 192 CTAs, service pool running | 3,179.0 µs | **+1,159 µs (+57.4%)** |

The interference costs **2.8x the capacity tax**. This completes rather than
contradicts the A7 strike: occupancy is one block per CU so a service CTA truly
cannot steal issue slots — but CTAs that never share a SIMD still share the L2,
the Infinity Cache and the fabric. **CTA role specialization removes issue-slot
contention and leaves memory-system contention untouched.**

Two consequences:

- **Mode 0 is retired as the control for role specialization.** It measures the
  capacity tax and is structurally blind to the dominant cost of the mechanism
  it is the control for. Every future role-split experiment needs a
  matched-CTA-count comparison.
- **The interference is atomic traffic, not payload traffic** — and this was
  tested rather than assumed. The `g` axis separates them, because payload bytes
  are identical at every `g` while probe atomics scale with `g`. At C=64 the M7
  interference rises monotonically: **+1,159 µs (g=1), +1,646 (g=2), +2,059
  (g=4), +2,694 (g=16)**. Payload cannot explain a curve that rises as the
  transfers get *coarser*; the probe loop can.
  *(This retracts an earlier inference in this file that M10/SDMA should be
  promoted to first. SDMA offloads the bytes, which are not the problem. M10
  goes back down.)*
- **The floor decomposes: ~30% ordering scope, ~70% operation count.** exp_06
  relaxed the arrival RMW (a labelled, reverted, non-correctness-preserving
  diagnostic) and recovered 354 µs of the 1,159 µs floor at g=1. exp_07 then
  tested whether the rest was cache-line *footprint* by transposing the counter
  array so 32 lanes hit 2 lines instead of 32 — **and it was at least 10x
  slower**. Coalescing is right for loads and wrong for atomics: 32 RMWs to 2
  lines serialize at the line, where 32 RMWs to 32 lines proceed in parallel
  across L2 banks. **Scattering atomics is a feature of the current layout.**
- **Placement confirms the mechanism, and closes A8.** `exp_08` added mode 3 —
  mode 2 with the pool on *whole XCDs* instead of a spread tail. At `C=64`, same
  CTA counts, only placement differing: M7 interference **−36%** (so it really
  is per-XCD L2 contention), but the service pool, squeezed onto two dies,
  slowed **+54%**, for **+26% worse overall**. *The interference and the service
  throughput are the same resource seen from two sides — you cannot isolate the
  communication engine from the compute without also starving it.*
- **So M4 must reduce the NUMBER of arrivals, not relocate them.** Per-XCD
  aggregation (AMD Research's *Fleet* shape) is exactly that and remains the top
  experiment. Any variant that merely moves counters is predicted to fail —
  exp_07 and exp_08 are both evidence. **But see the caveat below: a cheap M4
  may not exist.**

## The thesis of the night

Putting the interference curve next to the prize it was supposed to buy gives a
single, uncomfortable, *measured* statement:

> **On this kernel and this machine, the memory-system interference a service
> pool inflicts on the concurrent compute phase is approximately equal to the
> communication time it hides. CTA-level overlap is close to a wash.**

The arithmetic, entirely from tonight's measurements. A `C=16` pool inflates the
concurrent GEMM phase by **+37%**. Applied to the dispatch boundary — the one
remaining target — that is `0.37 × 2,976 µs (plan+M6) ≈ +1,100 µs` of
interference against a dispatch prize measured at **0.7–1.25 ms**. It cancels.
The same cancellation, measured directly rather than predicted, is why the
combine boundary tops out at a tie: even with *every* atomic and every
bookkeeping cost removed, mode 2 at its best `C` reaches ~7,300–7,450 µs against
`pf6gm_mega`'s 6,906 µs, because the capacity tax plus a non-zero service cost
exceeds the ~300 µs the peeled-out pull was worth.

This is a verdict on **this boundary and this transport**, not on the technique.
It has two clearly identified escapes, and they are the work worth doing next:

1. **Cut the interference at its source.** It is atomic *footprint*, not payload
   and not ordering strength (exp_06: scope is only ~30% of the floor). M4's
   per-XCD arrival counters cut footprint and scope together. **M4 is the
   enabling technology for CTA role specialization on CDNA4** — not an
   optimization to try afterwards.
2. **Pick a boundary where the hidden communication is much larger than the
   compute phase it runs under.** Neither of ours is. That is a property of this
   MoE layer's shape, and it should be checked before designing the next split
   rather than after.

## The research question, so far

> *Can CTA-level communication/computation overlap be made extremely performant
> on AMD GPUs?*

Partial answer, from measurement rather than opinion:

- **The split itself is cheap.** Reserving CTAs costs 0.5–8.0% across
  C = 2…64 (0.995x at C=2, 1.080x at C=64) — far below the naive
  `256/(256−C)` capacity model. `role_partition` / `finish_order_partition` are
  exonerated.
- **But the service role is register-poor by construction.** A service CTA is
  allocated the same 256 ArchVGPR + 256 AGPR as a compute CTA because
  allocation is static and per-kernel. It never issues MFMA, cannot use what the
  MFMA path reserved, and cannot obtain one extra register. **CTA-level
  specialization does not sidestep the HipKittens wave-specialization register
  problem — it relocates it.** Measured: a 64 B/lane staging array spilled to
  scratch (60 → 128 B) and ate most of the latency it was meant to hide.
- **So the technique lives or dies on picking a boundary whose prize is bigger
  than its bookkeeping.** The combine boundary's prize (~300 µs) is too small.
  That is a verdict on the boundary, not on the technique.

## What to do next, in order

1. **`exp_05` stage 0 — measure the dispatch exposure.** `exp_05/design.md`
   verifies the premise from source: there are **four grid barriers between M2
   and M6** (`:937`, `:1059`, `:1114`, `:1124`), so the dispatch is strictly
   serialized before the first GEMM. The pre-M6 region is ~960 µs and CLAUDE.md
   attributes 757 µs of it to M1+M2 — **this is the only remaining boundary
   whose prize can move the 0.80x objective.** The kernel already writes a
   timestamp block the host allocates and never reads
   (`e004pf_k0pf_ab.py:1504-1505`); wiring it up converts 757 µs from `INFERRED`
   to measured. Do this before writing any device code.
2. **Add the missing primitive.** Two independent findings converged on it:
   `counter.cuh` can express "arrive and release" for one counter but has
   nothing for "all members of this group have arrived AND every writer's
   payload is visible". Its absence forced both the `g`-probe loop (measured as
   the dominant driver of the `g` axis) and the per-wave publish that carries
   the ordering hole below. Highest-value library change on tonight's evidence.
3. Only then consider `exp_05` stages 1–2 (local-tokens-first, then the
   two-pass split) — reading `exp_09/design.md` first, because the same
   interference tax applies there and `exp_09` shows the arrival count is fixed
   by the *plan*, not the protocol.

**Do not** spend a build cycle on: raising the `C ≤ 64` cap (the 1/C law
extrapolates it), relocating arrival counters (exp_07, exp_08), per-row counters
replacing per-slice ones (exp_09 predicts the exp_07 pessimization), or SDMA for
the payload (the payload is not the problem).

## Correctness posture of the shipped tree

Every configuration run tonight passed the full ladder — `[MOK GATE]`,
`control_fails=True`, and a 600-epoch soak with `pperr=0`. That includes four
5-rotation decision campaigns, ~50 screening configurations, a 12-trial
seed-varying ordering-hole probe, and a deliberately awkward 10-point envelope
sweep (both placement modes, every legal `g`, `flush_rows` at 1 and 64, and a
non-multiple-of-8 `C=63`). No memory fault and no nonzero `pperr` was observed
after the exp_01 fix landed, in any run.

## Open items a reader should know about

- **A latent ordering hole in mode 2 at `g < 16`**, real in source, **12/12
  clean** under a dedicated seed-varying sensitivity test plus 15/15 campaign
  gates and 10 x 600-epoch soaks. Recorded, not fixed — the path is closed for
  performance and hardening it risks the resource tuple.
  See `exp_01_nil_fault/ordering_hole_probe.md`.
- **`A4 pull_fallback=1` is a performance datum only**, not a valid correctness
  arm as written (mode 2 never system-releases `part`).
- **CLAUDE.md's M3 row and A2 note were factually wrong and are corrected in
  place**: the "~1 MiB knee / 70x too small" numbers are arXiv 2607.19539 on
  4x A100 over NVLink, not COMET, which contains no such measurement. The
  arithmetic also never divided by `C` (at C=32 the model needs 4.63 GB/s per
  service CTA). M3 demoted below the fence class.
- **The scratch budget regressed 60 → 128 B/lane** with the kept MLP change. The
  mode-0 control (7,387 vs 7,421) shows it is free to the compute path, but it
  is a real deviation from the stated 36 B target.
- **`K0_MPS_DEBUG_STOP` aborts the ranks** (exitcode 2) and cannot be used as a
  phase-attribution ladder on this build.

## Two method traps that cost real time — now guarded

- **`.cuh` edits do not invalidate the mori JIT cache** (it hashes only
  `.hpp/.h/.cpp/.hip` in mori's own tree). A whole experiment was measured on a
  stale kernel and initially reported as a null. Guard: `K0P6_MPS_SRC_REV` in
  the hashed `.hip`, bumped with any `.cuh` change. **Cache *directory* mtimes
  are touched on a HIT — only the `.hsaco` mtime is evidence.**
- **The node checkout IS the arm** (containers bind-mount it read-only). Both
  the screening driver and the campaign driver now `git reset --hard` before
  every run and print the newest hsaco mtime before and after.

## Experiment index

| # | subject | verdict |
|---|---|---|
| `exp_01` | the `address (nil)` fault | **fixed**, full gate ladder green, resource-neutral; plus an ordering-hole probe (12/12 clean) |
| `exp_02` | decision campaign, C=8 | `mps_mega` 8.33x slower — the mission's question answered |
| `exp_03` | the (C, g, mode) map | 1/C law; `g` is an atomics knob; ceiling ~0.88x; A1 closed by fitted law |
| `exp_04` | MLP fan-out in the peer copy | **kept**, −5.0% campaign-confirmed; copy is only ~19% of service cost |
| `exp_05` | phase attribution (stage 0) | instrument fixed and working; **the interference finding** |
| `exp_06` | atomic ordering scope | scope ≈ 30% of the interference floor (diagnostic, reverted) |
| `exp_07` | chunk-major counter layout | **≥10x slower, reverted** — coalescing atomics is a trap |
| `exp_08` | A8, XCD placement (new mode 3) | **A8 closed** — die-level pool cuts M7 interference 36% but starves the pool 54% |
| `exp_09` | reducing the arrival count (M4) | analysis only — **no cheap M4 exists**, and the tempting idea is predicted to fail |

## Method note worth keeping

A 1-process / 1-warmup / 1-timed screen costs 25–95 s against ~18 min for a
5-rotation campaign, and tracked campaigns to **0.1%, 1.3% and 2.6%** on three
separate checks. ~45 screening points shaped every axis tonight; campaigns were
spent only on decisions. Screens are labelled `valid_diagnostic` by the harness
and no conclusion here rests on a screened difference under 5%.

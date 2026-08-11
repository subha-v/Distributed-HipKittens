# Overnight status — the two-minute morning read

Written 2026-08-11 ~09:45 UTC. Full detail in `experiments/LESSONS.md`
(append-only) and the per-experiment `result.md` files.

## The headline

**The fault is fixed and the mission's question is answered. The margin over
`production` is NOT yet wider than it was — it is unchanged at 0.895x — and the
reason is now measured rather than guessed.**

| arm | µs (5-rotation median rank-max p50) | vs `production` |
|---|---:|---:|
| `production` | 7,702.1 | 1.000 |
| **`pf6gm_mega` — still the ratchet** | **6,905.8** | **0.897** |
| `mps_mega`, best point (C=64, g=1, mode 2) | 10,107.0 | 1.312 |

Three full decision campaigns, 15 correctness gates each, every one green.

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
   two-pass split).

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

## Method note worth keeping

A 1-process / 1-warmup / 1-timed screen costs 25–95 s against ~18 min for a
5-rotation campaign, and tracked campaigns to **0.1%, 1.3% and 2.6%** on three
separate checks. ~45 screening points shaped every axis tonight; campaigns were
spent only on decisions. Screens are labelled `valid_diagnostic` by the harness
and no conclusion here rests on a screened difference under 5%.

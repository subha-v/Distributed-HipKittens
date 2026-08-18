# Captured-route corpus analysis (CPU-only, 2026-08-18 ~01:45 PT)

512 calls × [4096,8] across all 8 workers, from the routecap1 capture
(m15+M23 server, c32p). Script: scratchpad/analyze_pad_routing.py.

## Findings (decision-grade)

1. **31% of sealed calls are 100% fake work.** 160/512 calls are fully
   degenerate (>95% of rows share one expert set, across-row weight STD
   ≈0.008 ≈ identical hidden states, earliest seq indices): these are
   DP-dummy / pad-dominated steps that the M23 rescue now routes through the
   B4096 graph — the megakernel (and production) run a full 4096-row MoE for
   ZERO real tokens. Matches the receipts' `uniform_rescued` ≈ 34% of steps.
   91 distinct pad expert-sets, scattered across ranks (pad max-rank-load
   ~3.0×) — pads do NOT all hit rank 0.
2. **Real traffic's hot-set concentration is confirmed per-call**: in real
   calls the modal row-pattern is exactly experts {0..7} — all on rank 0 —
   covering ~16% of rows (varying weights ⇒ real tokens, 85 calls). Real-row
   max-rank-load p50 = 2.61×, p90 = 3.35× fair share. Mechanism (b),
   hot-rank compute concentration, is live; the mega's slab rendezvous
   couples all ranks to rank 0's finish.
3. **Run-correlation is ~nil at input-row granularity**: consecutive real
   rows share 57.0% of experts vs 55.7% after shuffling — a 1.3-point
   excess. The destination-interleaving / fabric-discipline hypothesis
   (mechanism a) is WEAKENED; deprioritize those arms.

## Implications for the optimization order

1. **Fill/dummy-skip is the top kernel lever, now with two tiers**:
   (i) skip all-pad (dummy) sealed steps entirely — ~31% of sealed steps are
   pure waste both arms currently pay; (ii) n_orig-aware row skipping on
   partially-padded steps (mean fill ~40-60%). Combined: the mega can do
   ~2.7× less MoE work than production per average step. Parity-safe if pad
   outputs are provably never read (M23 §3 analysis; verify finalize
   contract). Production's captured graph CANNOT skip — structural advantage.
2. **Placement second**: RR-by-permutation (already implemented + verified,
   `rr_patch.py`) spreads experts 0-7 one per rank — directly attacks the
   measured 2.61× real-row skew.
3. **Fabric/interleaving arms: demoted** to opportunistic unless kernel-level
   captured-replay contradicts the corpus.

Caveat: capture = first 64 B4096 calls per worker (early-run bias), but the
dummy-share matches whole-run receipts, so the 31% is not a ramp artifact.

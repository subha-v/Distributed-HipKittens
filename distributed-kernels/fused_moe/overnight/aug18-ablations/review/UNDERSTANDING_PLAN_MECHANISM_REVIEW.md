# Mechanism-soundness review of UNDERSTANDING_PLAN.md (draft r1)

Opus adversarial reviewer, 2026-08-18. Reviewed the pre-Track-1 draft; verdict: reorientation
shape correct, but six of ten settling experiments could not settle their question as specified,
two "current explanations" contradicted banked measurements, and the one experiment the grounding
called blocking (the ρ sweep) had been deleted. All items below were folded into r2 of the plan.

## BLOCKING

**B1 — ρ sweep deleted; "re-score, not re-run" is arithmetically wrong.** Re-scoring rescues the
gate verdict only: achievable hidden time at the rig's ρ=0.65 is min(compute, transport)=1,439 µs;
measured hidden (phased−fused) = 47 µs = **3.3% of what was reachable** — F1–F4 remain null after
re-scoring. Rig ρ=0.65 vs deployment ρ≈2.6–3.2 (~5× off); W_VECTOR:339 / EXPERIMENT_LADDER:482
rank the k_inner sweep #1/blocking, "one argv." Fix: restore as R2a, ρ ∈ {0.65,1,2,3} × 4 arms,
same session as the ledger; report F1–F4 as "null at ρ=0.65, 3.3% of ceiling; re-tested at ρ≈3."

**B2 — Q6's mechanism is wrong; the right one is already measured.** exp_03 stamp decomposition:
the −444 (C 16→24) splits as M7 −353.8 with combine residue flat; flush_rows=1 splits further into
sweep ≈95 + injector-concurrency relief ≈354 — "C acts as an injector-concurrency throttle — a
second, coarser flow-control knob on top of depth 4." ~80% of the delta is a depth-class
mechanism, not consumption. C_sat=20.8 needs a 5.4× fudge (κ_couple) = exactly the concurrency
term it cannot see; ρ=6.1 GB/s/CTA is derived, not measured. Structural consequence: **C and
depth are the same resource class and no design crosses them** (K5 was depth×cert×order; K4 was
C×flush). Fix: restate Q6 as "which of the two mechanisms sets the knee"; K4 reports M7 stamps
per cell; cross depth × C; knee is "24–28" (24→28 still −26.5 µs).

**B3 — the +831 µs is misdescribed and B.res cannot settle it.** Source: exp_20 — +831.5 µs is
M7(mode 7)−M7(mode 0) at C=64 in the mode-2 staged-push carrier family, **measured with the
pool's 896 B payload copy deleted** — a phase delta, by construction not traffic. The 6,866/0.888×
is a different experiment (exp_14, e2e). Mechanism open: exp_20 killed payload/occupancy/pacing/
poll-backoff; flush_pending suspect nulled by mode 9 (+22 µs); remaining = g-independent
interference floor, 36% per-XCD L2, **64% unidentified**; no PMC path exists for a sub-phase of a
persistent launch. B.res tests none of the three structural MoK differences and at occupancy-1 an
N_CTA sweep varies CU idleness, not geometry. Fix: correct wording everywhere; split Q8 into
Q8a (scope reconciliation via B.res, register LAW-54's prediction that the second-kernel arm
loses, print occupancy per arm) and Q8b (the 64% floor — PARTIAL-by-construction, or attack via
LAW-18's line-footprint axis).

**B4 — β=1.506× is a ratio of two subtraction-derived numbers; R9 has no discriminating axis.**
Both 2,214 and 1,470 are wall − compute_floor, assuming identical GEMM cost in both arms — while
the phased arm carries +335 µs co-residency tax on those bursts. Store-towers 117.2 GB/s gives a
second value β=1.363× in a different convention; "1,470 µs = 280 GB/s" is the per-rank-egress
convention (whose 349–355 ceiling is flagged unreproduced) one row above logical-bytes/wall
numbers. Fix: standalone RCCL arm + standalone CDAR arm in one binary (β from two direct
measurements); sweep NCCL_ALGO × NCCL_PROTO × NCHANNELS on the in-rig ncclAllReduce; add a
matched-block-count parallelism control; one bandwidth convention per table.

**B5 — R2 cannot produce I_co as specified, and is two builds.** (1) h_ledger−h_wall = ΔI_co
(fused−phased), not I_co — the phased arm's own ~335 µs is invisible; needs a transport-only arm.
(2) "Both anchors" = two instruments on two chassis; the mega side's [MPS TS] carries five banked
traps (LAW-63): (a) s_memrealtime without lgkmcnt(0) — every prior reading unreliable; (b) 1 tick
= 0.01 µs, 22× error on the wrong clock; (c) running maxima never reset; (d) print inside
rank==0, never all-reduced — rank-max vs rank-0 differs +38% on a combine-class phase; (e) never
subtract clock64() across kernels. (3) timestamps=1 costs ~15 µs — comparable to the −26.5 µs
24→28 step. Fix: transport-only arm; mega ledger = separate build item with all five guards in
acceptance; timestamps state frozen across any ladder that locates a knee.

**B6 — Q1/R3 compares two instruments and calls the difference an itemization.** Production =
many kernels (rocprof real); homogeneous = one dispatch (device stamps only); LAW-63(d)'s +38%
instrument gap on a ~1,000 µs phase is ~half the 804 µs delta being attributed. Fix: one
instrument per quantity — HIP events for totals both arms, cross-rank rank-max stamps for phases
both arms, launch gaps production-side only; publish the phase-correspondence table (MORI staging
↔ dispatch/plan) BEFORE the run.

## MAJOR (abridged)

- **M1**: R6 "one transport argv" lands on the rig where depth is flat — no effect to measure;
  width change via op-class switch changes carrier too; no rate actuator named. Fix: co-resident
  at corrected ρ; width within one op class (8/16/32 B store packets); rate via destination load.
- **M2**: R5's mfma_burst scales duration and memory intensity together; Q4's claim (contention
  for issue/fabric, not bytes) unreachable. Fix: bytes_per_mfma argv, intensity ⟂ duration.
- **M3**: Q5's −573 was measured while the injection bound was inert (LAW-37/C6); "order ≈0
  alone" and "−144 product" unsourced; arithmetic is −661.8 not −655. K5's real purpose = C6
  (substitutes?). Stop quoting components until K5 returns them.
- **M4**: R7 ignores: the recorded hypothesis (pool holds the next rendezvous hostage → predicts
  flush_rows interaction ⇒ C=32 × flush is the discriminator); the diagnosed in-M15 analogue
  (7 ranks spinning rows_done while the 8th abandoned; pperr sticky, never host-reset); livelock
  as competing hypothesis (spin loops predate the s_sleep lesson); C=32 = one XCD under bid%8
  with PF_CTAS carved from the same tail; the likeliest hang site (hkp::grid_barrier) has no
  source in this repo → rocgdb attach is the zero-build first look; wait_result{observed,spins}
  exists and is discarded at two sites; [MPS SPIN] counters are running maxima; no sample size at
  p≈0.5; K4's C=32 cell collides with R7 (+~1 h timeouts). Fix: n≥8 at C∈{28,30,32} on the
  shipping binary; all-8-rank capture (LAW-61c); run before K4.
- **M5**: "σ_rig never measured" is false on the mega chassis (LAW-60: screens 0.52%, campaigns
  0.09%, 6.6% control drift within batch); 25 back-to-back invocations measure only within-batch.
  [Superseded by operator decision to cut R1; bands taken from banked values + R2a repeats.]
- **M6**: the §5.1 waterfall is forbidden-as-additive (LAW-21; C6); ratchet history contains a
  +726.9 µs move from an unexecutable diff; mode-14's rung not commensurable. Fix: re-run
  homogeneous→mode-12→M15 in ONE session from ONE binary with runtime knobs; .text sha per arm +
  the −618.7 restoration contrast as receipt; mark non-additive pairs on the figure.
- **M7**: R10's geometry axis confounded with block count (1024-row slabs ⇒ 32 blocks of 256 =
  the F0 failure class). Fix: compensate with bps to hold blocks constant.
- **M8** number fixes: +211.2 (paired) not +202/+210; −661.8; mode-16's lesson is two-sided ("the
  wait it escaped was mostly not on the critical path, AND the parity state it added was"); label
  cross-campaign pins; LAW-39 — cite the T-sweep correctness repair whenever the sweep is shown.

## Deltas with unknown mechanism not covered by Q1–Q10 (added to r2)

1. The g-independent interference floor (+1,159 µs, 64% unidentified) — "the single largest gap
   in our understanding of this kernel"; needs an ablation design, no PMC path exists.
2. The +335 µs co-residency tax on the PHASED arm (θ-P12, "attributed by nobody yet").
3. The carrier↔consumer sign flip (+340 vs −444, same reservation) — "the only knob in the
   inventory with a measured sign flip under the same mechanism"; the experts' literal question.
4. bps 1→2 = +64% — an occupancy law under the granularity story and the decode "reversal".
5. LAW-31: dispatch a2a has zero exposed wait (scope: routing std=0; 4 grid barriers make the
   pre-M6 region a sum of maxima) — load-bearing for the Anchor-A waterfall.

Net added node time ≈1.5–2.0 h; plus one extra build item (the mega-side ledger).

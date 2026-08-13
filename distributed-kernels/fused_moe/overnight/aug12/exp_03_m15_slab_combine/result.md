# exp_03 result — M15 slab-certified pipelined combine: NEW BEST, −191.4 µs vs the ratchet

- **Design contract:** `../M15_DESIGN.md`. Kernel:
  `k0pf6gm_device_tile_m15.hip` (RUN PIN branch `ablations-m15` @ `5753970b`
  compiles it as `k0pf6gm_mps_mega`; node worktree `~/DHK-m15`). RMW arm only
  (K0P6_M15_STAGED=0). Config `C=16, g=353 (depth 4), mode=12,
  flush_rows=16` (flush_rows = pool front-sweep quota per wave).

## Headline (all same-session, stamps-off, 5 rotations each, all gates green)

| arm | campaign | p50 median µs | range | × production |
|---|---|---:|---|---:|
| production | e02camp12 | 7,712.0 | 7,705.8–7,713.9 | 1.000 |
| pf6gm_mega | e02camp12 | 6,908.8 | 6,895.4–6,927.4 | 0.8958 |
| mode-12 ratchet (control) | e02camp12 | 6,483.8 | 6,468.9–6,495.5 | 0.8407 |
| mode 16 (exp_02) | e02camp16b | 6,559.6 | 6,536.6–6,569.1 | 0.8513 |
| **M15, C=16** | **e03camp15** | **6,292.4** | **6,289.2–6,295.8** | **0.8165** |
| M15, C=8 | e03camp15c8 | 6,385.2 | 6,378.9–6,399.6 | 0.8288 |

**M15 C=16 = −191.4 µs vs the in-session mode-12 control; the arms are
disjoint by 173 µs (M15 max 6,295.8 < control min 6,468.9). Rotation spread
6.6 µs.** The result lands inside the pre-registered band (−60…−220 µs on
the M7+combine block ⇒ p50 6,270–6,430). Control reproduces the published
ratchet to 1.1 µs, so the same-session pairing is clean. First sub-0.82
number in the program.

## Gates (every one green before any timed number)

- CPU (e03_gate_build.sh + e38_30_isa.py): compiles clean first build;
  tuple SGPR 106 / VGPR 256 / AGPR 256 / **scratch 72 B/lane (ratchet:
  128)** / LDS 155,496 / occupancy 1; spills 137 SGPR (186) / 17 VGPR (15);
  `.text` 134,208 B (deleted service machinery). **Issue-run gate: 282
  epilogue atomics in 10 runs, mean 28.2, max 59, ZERO scratch ops and ONE
  `vmcnt(0)` in the epilogue window (ratchet: 12 runs / 23.5 / 21) — the
  slab rendezvous did NOT collapse the injection window (the exp_38 S8 risk
  that killed mode 14); it is wider than the ratchet's.** Flag-off DEF
  `.text` == rev 26 re-proven with the M15-DELTA hooks in tree.
- GPU (smoke e03s15a + both campaigns): `[MOK GATE]` pass (max_abs 0.039,
  rel 0.008295 — family-canonical), poison survivors 0, selftest fires
  (57344) in every rotation, soak 600/600 `pperr=0`, spin 0/0,
  `combine_bit_exact pass=True bf16_bit_diffs=0` (stamps rotation).

## Attribution (e03stamps15, timestamps=1, rank-0 stamps, ticks = 0.01 µs)

| phase | M15 | mode-12 control (exp_33 era) | Δ |
|---|---:|---:|---:|
| plan M3–M5 | 379.5 | 372.8 | +6.7 |
| M6 (GEMM-1) | 2,466.5 | 2,453.3 | +13 (noise/codegen class) |
| M7 (both slabs + 2 rendezvous + pool sweeps) | 2,653.1 | 2,701.8 | −48.7 |
| combine after M7 | **180.0** | 324.2 | **−144.2** |
| **M7+combine (the coupled objective)** | **2,833.1** | **3,026.1** | **−193.0** |

The phase budget closes onto the end-to-end delta (−193.0 vs −191.4). The
mechanism is exactly the design: the front half of the combine was consumed
inside GEMM-2's shadow (pool sweeps during slab 1 + post-R1 leftovers), the
per-row protocol is gone at coarse slab-word prices, and M6 is untouched —
the nc-major XCD-residency falsifier did not fire.

## The C=8 inversion (a regime datum, not just a tuning point)

Mode 12's placement law (exp_37: C≤8 beats C=16, dedication is a pure
capacity tax) **inverts** under M15: C=8 costs +92.8 µs vs C=16. With the
pool doing productive consumer work (front-half sweeps) instead of carrier
work, more pool is better at this quota. "Never dedicate CTAs to
communication" refines to: never dedicate them to *carrying*; dedicating
them to *consuming inside the producer's shadow* pays. Follow-ups worth one
session: C ∈ {24, 32} × flush_rows ∈ {8, 16, 32} (the quota bound may bind
at higher C), and flush_rows=0 to isolate pure protocol-deletion from the
sweep term.

## Session 2 (2026-08-13): the C response curve, its decomposition, and three adjudicated arms

All same-session, stamps-off, 5-rotation medians, every gate green unless noted:

| arm | p50 median µs | × production | verdict |
|---|---:|---:|---|
| M15 C=8 | 6,385.2 | 0.8288 | |
| M15 C=16 | 6,292.4 | 0.8165 | |
| M15 C=24 | 5,848.5 | 0.7589 | |
| **M15 C=28** | **5,822.0** | **0.7544** | **shipping config** |
| M15 C=32 | — | — | liveness instability: 1/2 runs hung, no rank JSONs; do not use without diagnosis |
| M15 C=24, flush_rows=1 | 5,943.1 | 0.7698 | pool-sweep term ≈ +95 µs when quota starved |
| m17 (RR scatter) C=24 d4 | 5,845.3 | 0.7585 | tie with plain C=24 — interleave neutral at balanced routing; parked as skew insurance |
| m17 C=24 d8 | 5,883.3 | 0.7624 | depth-4 optimum survives link-spreading |
| m15b (staged wide-push, g=65) | 9,087.1 | 1.1768 | **falsified** (correct — 5/5 gates green — but ~2.8 ms slower) |

**The C=24 stamps (e03stampc24) decompose the −444 µs (C 16→24):** plan
376.2 (flat), M6 2,441.8 (flat), **M7 2,299.3 (−353.8)**, combine residue
177.4 (flat). Giving up 8 compute CTAs made the M7 phase 13% faster: the
epilogue RMW stream is deep in a fabric-congestion regime and **C acts as an
injector-concurrency throttle** — a second, coarser flow-control knob on top
of depth 4. The flush_rows=1 arm separates the terms: sweep ≈ 95 µs,
concurrency relief ≈ 354 µs, orthogonal.

Two mechanism hypotheses adjudicated by the new arms: (a) *op class* — m15b
deleted the RMW class entirely (locally-folded stage + wide posted pushes)
and lost 2.8 ms: the op class is NOT the tax; (b) *per-link burst
concentration* — m17 spread each wave's window over ~7 links and tied: burst
locality is NOT the tax at balanced routing. What remains standing is
**aggregate injector concurrency**: how many CTAs are in their epilogue
simultaneously. The C knob (and possibly a finer in-kernel pacing successor)
is the mechanism-true control.

## Open items

- The staged wide-push arm (`-DK0P6_M15_STAGED=1`, "mode 15b") is built and
  compile-gated but unmeasured; needs descriptor slot 63 + host stage
  allocation per M15_DESIGN.md §2.
- Substitutes-vs-complements is now answered as **complements**: the depth-4
  bound was kept verbatim and the coarse-readiness restructure paid −191 µs
  on top of it (mode 14's coarse arm without the order/pipelining lost).
- T=1024/2048 correctness defect still blocks every other-size claim.

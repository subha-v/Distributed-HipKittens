# exp_07 — chunk-major arrival counters: a severe pessimization, and a corrected model

**Verdict: REVERTED. Coalescing atomics onto few cache lines is a trap.** The
change is semantically identical to the original and made the kernel at least
~10x slower. No completed measurement — the run was abandoned rather than left
to burn its timeout — so the magnitude is a lower bound, not a number.

The value is in the mechanism, which **inverts the intuition that motivated the
experiment** and corrects the model exp_06 left open.

## The hypothesis

exp_06 attributed only ~30% of the `g`-independent +1,159 µs M7 interference
floor to the arrival RMW's *ordering scope*, leaving ~70% unexplained and
tentatively labelled "footprint". The natural reading of footprint:

`nc_arr[r*16 + nc]` puts consecutive rows exactly 16 uint32 = **64 B apart —
one cache line**. The 32 live lanes of an event handle 32 different rows of the
same block, so every arrival touches **32 distinct cache lines**.

Chunk-major `nc_arr[nc*T_ext + r]` puts those same 32 rows in 32 consecutive
uint32 = **2 cache lines**. Same counters, same protocol, same buffer and size,
no ABI change — only two service call sites and one helper know the layout, and
M0 still zeroes the whole array. A clean layout A/B.

It even made a falsifiable side-prediction: the group-completion probe loop
walks `nc` for a *fixed* row, which is one shared line under row-major and `g`
separate lines under chunk-major, so the change should win at `g=1` and lose at
`g=16`.

## What happened

The first screened configuration (`C=64, g=1`) made no measurable progress in
**12 minutes**, against the 25–95 s these screens normally take. The log froze
after rank connect with the container alive and no `pperr`, no fault, and no
gate line. Stopped with SIGTERM rather than left to consume its 25-minute
timeout, so **there is no completed measurement**. Lower bound only.

## Why — the inverse of the intuition

**Coalescing is right for loads and wrong for atomics.**

A load that 32 lanes issue to 32 consecutive addresses is *one* memory
transaction — that is why coalescing is a virtue everywhere else, and it is why
`store_peer_packets`'s 1,024 B/wave pattern is correct (exp_04). But 32 lanes
issuing *read-modify-writes* to 2 cache lines must **serialize**: an atomic is
resolved at the line, so two lines can retire two RMWs at a time. The same 32
RMWs spread over 32 lines proceed in parallel across L2 banks and channels.

Scattering atomics is therefore a **feature** of the original layout, not the
defect I read it as. Across 8 GPUs × 64 service CTAs × 32 lanes the contention
is enough to dominate the entire kernel.

## The corrected model

This sharpens exp_06 rather than contradicting it. The unattributed ~70% of the
interference floor is **not** "too many cache lines touched". It is the atomic
**operation count** and the contention those operations create *wherever they
land*. Relocating them cannot help; only issuing fewer of them can.

Consequences worth carrying forward:

1. **M4 must reduce the NUMBER of arrivals, not their placement.** Per-XCD
   aggregation with one cross-XCD release per XCD is exactly that shape and
   remains the right experiment. Any variant that merely relocates counters is
   predicted to fail, and this experiment is the evidence.
2. **Line-coalescing is a hazard for any future counter or flag redesign** in
   this kernel. It should be called out in the design contract.
3. The exp_06 bound stands unchanged: scope ≈ 30% of the floor, the rest is
   operation count and contention.

## Primitives

The library gave no signal in either direction here, and that is the finding.
`counter.cuh`'s arrival vocabulary is address-agnostic: a caller chooses the
counter's address itself and receives no guidance that **placement changes the
cost by an order of magnitude, in the opposite direction from loads**. If the
library is going to own arrival counting — and exp_04, exp_06 and this
experiment all say it should — then it should own *allocation* of the counters
too, precisely so a caller cannot make this mistake. That is now the fourth
independent argument for the same missing primitive.

## Artifacts

Diagnostic commit `1b2f58b5`, reverted in `1f244f08`. Node log
`~/overnight-scratch/E07tx_C64g1mode2flush_rows16timestamps1.log` (frozen at
rank connect, 2,356 bytes).

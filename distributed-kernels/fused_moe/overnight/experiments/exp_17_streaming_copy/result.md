# exp_17 — streaming (`nt`) peer copy: NULL, and it identifies the interference as BANDWIDTH

**Verdict: NULL, reverted — and the `nt` bits provably reached the ISA, which
makes this a clean negative rather than an inconclusive one.** The service
pool's interference with the concurrent GEMM is **bandwidth contention, not
cache-capacity pollution**. No replacement-policy hint can move it. This also
kills the M1/M2 cache-bit class of the M-series for this problem.

## The hypothesis

exp_16 bounded the atomics at ≤19% of the remaining +1,249 µs of M7
interference, leaving ~1,000 µs attributable to the service pool's own memory
traffic. The payload is read once and written once and reused by nobody, yet the
default lowering caches it fully — exp_03's ISA read found the emitted
`flat_store_dwordx4` carries no `sc0`, no `sc1` and no `nt` at all. So a pool
streaming hundreds of MB should be evicting M7's weights from the per-XCD L2 and
the 256 MB LLC, and marking the copy non-temporal should stop it.

## The change

Additive `store_peer_packets_streaming` in `packet.cuh`: identical traffic to
`store_peer_packets`, with `nt` requested on **both** the source load and the
peer store via `__builtin_nontemporal_load/store` over a native
`ext_vector_type(4)`. `nt` is a replacement-policy hint (ISA §9.1.10.2, LLC Hit
Evict), not a coherence bit, so it does not disturb the release/acquire
discipline wrapped around the copy.

## Result

Screened at g=1, mode 2, timestamps on, against exp_14 at identical configs:

| C | exp_14 M7 | exp_17 M7 | delta | exp_14 total | exp_17 total |
|---:|---:|---:|---:|---:|---:|
| 32 | 2,496.4 | 2,489.0 | −7.4 | 7,076.0 | 7,056.7 |
| 48 | 2,704.3 | 2,630.5 | −73.8 | 6,941.0 | 6,961.7 |
| **64** | 2,835.3 | 2,849.5 | **+14.2** | 6,949.2 | 6,942.2 |
| 96 | 3,290.0 | 3,278.6 | −11.4 | 7,157.4 | 7,159.0 |

Every delta is inside the screening band, with no consistent sign. All gates
green.

## The hint was applied — this is a real negative

Disassembling the built object confirms the bits landed:

```
flat_load_dwordx4  v[68:71], v[50:51] nt      // DC5E0000
flat_load_dwordx4  v[50:53], v[26:27] nt      // DC5E0000
flat_store_dwordx4 v[18:19], v[50:53] nt      // DC7E0000
```

Three `nt`-carrying `dwordx4` accesses appear where before there were none, out
of 352 total. So the compiler honoured `__builtin_nontemporal_*` and the copy
really did run non-temporally. **The mechanism was tested, not merely
requested.**

## What it means

A cache hint changes *what gets evicted*; it does not change *how many bytes
cross the memory system*. The pool moves roughly 312 MB in and 312 MB out per
rank per epoch while M7 is running. If M7 is limited by available bandwidth
rather than by its working set being resident, then no replacement policy helps
— and that is exactly what the null says.

**Consequences:**

1. **M1 and M2 are dead for this problem.** Both are cache-bit experiments on
   the payload store (`sc0 sc1` bypass, `nt` hit-evict). M2 is directly
   falsified here. M1's premise was already corrected by exp_03 (the payload
   store carries no `sc` bits, so the `buffer_wbl2` is load-bearing rather than
   redundant); this result removes the remaining motivation to pursue either.
2. **The only remaining lever is to move fewer bytes.** Current payload traffic
   per rank per epoch: M7's epilogue writes `part` (~312 MB), the service pool
   reads `part` (~312 MB), the service pool writes peer slots (~312 MB) —
   **~936 MB total**.
3. **A11/M11 becomes the whole game, and its prize is now quantified.** If the
   compute CTA's epilogue accumulated **directly into the owner's slot**, the
   local `part` write and the pool's read and write all collapse into one
   remote accumulate: **~936 MB → ~312 MB, a 3× reduction in payload traffic**,
   and the service pool's payload role disappears entirely (it would carry only
   readiness and flags). That should remove essentially all of the ~1,000 µs.

   The blocker is real and CLAUDE.md names it: the epilogue currently uses
   `global_atomic_pk_add_bf16` to accumulate `part` across the several blocks
   that contribute to a row, so going direct means **remote** bf16 atomic
   accumulation, which turns slice-completion detection into a remote-atomic
   ordering problem. **Protocol-review signoff is required before build** and
   that requirement stands. Precedent: AMD Research SC24 measured 12% on fused
   GEMM+All-to-All with exactly this vertical zero-copy shape.

## Disposition

**Reverted** in the adapter; the tree returns to the state campaign `dec07`
validated (6,866.1 µs, 0.888× production). `store_peer_packets_streaming` is
**kept** in `packet.cuh` as an additive, documented, currently-uncalled
overload — it is correct, it demonstrably emits `nt`, and it is the right tool
for any future caller whose problem *is* cache pollution rather than bandwidth.

## Primitives

`packet.cuh` now carries three transport forms — plain, multi-region
(`store_peer_packets_multi`, exp_04), and streaming (exp_17) — of which the
plain one is the only one in use. That is not waste: each was added to test a
specific hypothesis about where the cost lives, and the two unused ones encode
negative results that would otherwise have to be re-derived from scratch. The
surface gap they collectively expose is that **a transport primitive's cost
model is invisible in its signature**: nothing about `store_peer_packets` tells
a caller whether it is latency-bound, bandwidth-bound, or cache-bound, and this
campaign had to answer that question three times by experiment.

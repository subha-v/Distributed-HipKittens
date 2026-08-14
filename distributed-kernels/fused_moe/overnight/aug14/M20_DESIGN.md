# M20: slot-pool replication with cross-layer weight prefetch

The production-practical successor to M19: identical routing machinery
(source-local adaptive replication, mirrored acceptance, untouched combine),
with the 41 GB of persistent per-layer replicas replaced by a ~1 GB
triple-buffered slot pool refreshed by an in-kernel prefetch engine that
overlaps weight movement with compute ACROSS LAYERS.

## Measured motivations

- M19 pair #3: +39.8% end-to-end serving — the routing model is right.
- M19's memory: +41 GB/rank halved the serving KV pool (68.8 → 31 GiB).
  Not viable at production concurrency / long context.  (MoonEP's slot-pool
  memory model is the fix; its per-step global planner is NOT taken — our
  zero-coordination per-chunk decisions replace it, validated in serving.)
- M19's +1.2 ms constant: the M0.5 decision pre-pass costs two grid-barrier
  rendezvous (~600 us each, consistent with the fence-law lore).  M20 moves
  the decision OUT of the megakernel entirely.
- C-sweep (2026-08-14): M18 is C-insensitive 16..28 under skew — the
  service pool has >=12 CTAs of slack.  Four become prefetchers for free.
- Ownership concentration: a layer's top-B experts frequently share one
  owner rank, so a one-layer prefetch window (352 MB over one ~50 GB/s
  xGMI link ~ 7 ms) does NOT hide inside a ~6 ms layer.  Depth-2 prefetch
  (fetch L+2 during L) gives a ~12 ms window; hence THREE pool buffers
  (L uses buf L%3; L+2's writes target (L+2)%3, disjoint from L and L+1).

## Structure (compile arm `K0P6_M20_SLOTPOOL`, requires REPLICATE)

1. **Decision input, not decision pass.**  New descriptor slot D_M20_DECIN:
   a per-layer device buffer of 8 u32 (256 bits) written each step BEFORE
   launch by a trivial pre-kernel/torch op: bincount over this rank's
   `my_ids`, bit(e) = lut[e] >= E  AND  count(e) >= theta  AND
   prefetch_flag(slot(e), parity(L)) == this step's epoch tag.  The
   megakernel loads it into s_dec (replacing M0.5 wholesale — no barriers,
   no histogram, no theta in-kernel).  M1 publication to peers and the
   mirrored acceptance rule are byte-identical to M19.
2. **Slot pool.**  W13P/S13P/W2P/S2P: shared allocations sized
   [3][B][per-expert weight/scale], ~1.06 GB at B=8.  The per-layer LUT maps
   replicated eid -> slot E + b (b in [0,B)); the GEMM phases address
   slot >= E through the pool base at buffer parity L%3 (dual-base weight
   addressing in phase1/phase2, compile-gated).
3. **Prefetch engine.**  PF=4 CTAs carved from the service pool (C=28 ->
   24 combine + 4 prefetch).  Each invocation, they stream layer L+2's
   replica set — schedule is STATIC per layer (per-layer top-B from measured
   histograms; owner base addresses precomputed host-side into a per-layer
   prefetch table, desc slot D_M20_PF) — reading owner HBM via peer_ptr
   16-byte packets into pool buffer (L+2)%3, then publish per-slot
   epoch-tagged done-flags (D_M20_PFFLAGS; local, plain release).  Owner
   stagger: prefetcher p starts at expert (rank + p) % B to spread reads.
4. **Wraparound warmth.**  The schedule ring closes over the chunk
   boundary: layers 59/60 prefetch layers 3/5 of the NEXT chunk, so every
   layer including the first is warm in steady state.
5. **Fallback = adaptive-off.**  Un-prefetched or straggling slots simply
   never get their decision bit set; those experts route to owners (M15
   path).  No stall exists anywhere in the design.

## What this deletes/keeps vs M19

- DELETED: M0.5 (both grid barriers, in-kernel histogram, theta read),
  per-layer persistent replica weights, init-time owner broadcast.
- KEPT: rep tables/LUT (per layer), rep_dec decision publication +
  mirrored acceptance, RR scatter, planner over EL slots, M7/M8 untouched.
- Serving integration simplification: replicas now come from owner HBM at
  RUNTIME — no torch.distributed weight shipping at load, stock [32,...]
  layer weights unchanged, pool + tables are runtime state.

## Expected accounting (to verify)

- Balanced: no M0.5 tax, bits mostly off -> ~M15's 5.82 ms + pre-kernel
  (~10-40 us) + idle prefetch traffic (~1-2 ms of background xGMI reads
  overlapped; <2% expected).  Target: <= 0.78x production.
- Skew: M18's 6.06 ms + pre-kernel + prefetch overlap.  Target: <= 0.26x.
- Memory: 1.06 GB pool + tables (vs M19's 41 GB).
- New protocol surface to gate: pool-buffer parity vs prefetch-done epoch
  tags (a slot read by layer L must carry L's epoch tag written by L-2's
  prefetcher); the flags are rank-local so no cross-rank races exist.

## MEASURED (2026-08-14 evening) — the streaming→cache arc

The prefetch pipeline was built, measured, and REDESIGNED on data.  Three
in-kernel weight-movement variants, all against the same duty:

| variant | mechanism | measured cost/invocation |
|---|---|---:|
| pull, naive | 7 ranks read one owner, 1 load in flight/thread | **+93 ms** (~7 GB/s, latency-bound) |
| pull, 16-wide unrolled | same direction, deep pipeline | **+23 ms** — the xGMI remote-READ path is the bound |
| push (owner writes 7 peers) | the M7-proven write direction | **+54 ms** — per-CTA push law (~1.1–1.7 GB/s/CTA, exp_03_push_throughput) caps 8 CTAs far below the duty |

Conclusion: per-invocation GB-scale weight streaming is INFEASIBLE in-kernel
on this fabric at service-pool CTA counts (MoonEP's per-step streaming rides
NVLink TMA multicast; xGMI has no analog).  Since the weights are constant
across steps, per-step re-streaming is also UNNECESSARY: the final M20 is a
**persistent replica CACHE** — per-layer slots for a host-budgeted layer
subset, zero steady-state traffic, the prefetch engine retained only for
rate-limited set-refresh (K0_MOK_M20_DUTY experts/invocation) — with the
precomputed-decision dataflow unchanged.

**Final numbers (5-run campaigns, all gates green, T=4096):**

| routing | M20 | production | ratio |
|---|---:|---:|---:|
| measured serving skew | **6,035 us** | 24,283 us | **0.2483** [0.2480, 0.2498] |
| balanced (theta=64) | 6,340 us | 7,701 us | 0.8233 [0.8217, 0.8246] |

= the full M18 replication win at **672 MB (single-layer harness; serving
scales by the layer budget)** instead of 41 GB, with M19's 1.2 ms decision
tax deleted.  The balanced 0.823 is the theta=64 point (uniform per-expert
count is 128, so all replicas stay active); a theta above uniform returns
the M15-like ~0.76 — per-chunk automatic in serving where the pre-op
computes bits per step.  Kernel: `ablations` (K0P6_M20_SLOTPOOL) + RUN PIN
`ablations-m20`; harness: amd-master `debug/pf6-first-launch`
(K0_MOK_M20=1, cache mode = single pool buffer after the 3x672MB mori-heap
lesson).

## Deferred (M21 frontier, documented not built)

Full epoch pipelining (M1 of epoch N+1 under M8 of N — the parity-doubled
counters structurally allow 2-in-flight but the mode-12 consume-and-zero
slot lifetime and the retire heartbeat serialize today); peer-relay
multicast for prefetch (halves owner-link pressure); pull-based dispatch
(the MoK axis — expected minor at EP8 per the m17 ~1% result).

# M19: per-layer replication + in-kernel per-chunk adaptive routing

## The measured facts this design answers (all 2026-08-14, c32p serving)

1. **Layer identity dominates the static question.**  58 routed layers have
   58 distinct top-16 sets; the aggregate set leaves per-layer residual max
   rank load mean 1.36× / worst 2.61×, while each layer's own top-16 reaches
   1.09× / 1.25× at identical memory cost.
2. **Chunk variance is first-order beyond layer identity** (n=90,050 router
   calls): per-call coverage of a static set is bimodal (p5 = 1%, p25 = 17%,
   p75+ saturated at 77%); raw per-call max rank load median 5.15×, p95
   6.25×; even under ideal static redistribution the tail stays fat (p95
   3.05×, p99 3.65×).  Chunk wall-time is convex in skew, so the tail
   dominates: serving pair #1 with a static aggregate set lost 5.7%
   end-to-end while the aggregate-histogram harness predicted a 4× win.
3. **Replication carries a cost when it does not pay**: +3.9% at balanced
   routing (nrep=8) from planner width and fragmented replica blocks.  A
   chunk whose hot experts miss the replica set pays this on top of its
   full skew penalty.

## Design

M19 = the M18 chassis (M15 fusion + replica slots + source-local routing +
exactly-once acceptance + RR scatter order) with two additions:

**A. Per-layer replica sets** (host side, already landed on
`vllm-integration-m18`): each layer's slots `E..E+K-1` hold THAT layer's
top-K from measured histograms; per-layer M18R tables; a pointer→layer
registry turns any weights/table layer mismatch into a refusal.

**B. In-kernel per-chunk decision** (`K0P6_M15_ADAPTIVE`, requires
`REPLICATE`): replication per expert per chunk is decided from the routing
the kernel already holds.

- **M0.5 pre-pass**: grid-stride histogram of `my_ids` into `hcnt[0..255]`
  (buffer reused; M2 overwrites it later), grid barrier, then every CTA
  builds the same 256-bit decision mask: `bit(e) = (lut[e] >= E) &&
  (count(e) >= θ)`.  θ lives in the M18R table header word 3 (θ=0 ⇒
  always-replicate ≡ M18; θ>T·K ⇒ never ≡ M15 + pre-pass).
- **M1 dispatch**: `dest = bit(eid) ? cur : eid / E`.  Before the epoch
  handshake release, the last CTA publishes the rank's 256-bit decision to
  every peer (`rep_dec`, parity-doubled symmetric u32[2][world][8], slot 64)
  — ordered before `rows_done` by the existing release, so a peer that has
  seen the epoch word has the decisions.
- **M2/scatter acceptance mirrors the SENDER's bits** (exactly-once holds
  per source independently, no global agreement):
  - owned slot (`lut[e] < E`): accept a foreign row iff that source's bit
    for `e` is OFF; own rows always.
  - replica slot (`lut[e] >= E`): accept iff self row AND own bit ON.
- Everything downstream — planner over EL slots, GEMMs, M7 remote-RMW
  epilogue, slab certification, M8 combine, M9 retire — unchanged.

**Why this beats both parents**: per chunk it degenerates to whichever of
{M15, M18} is better.  Cold chunk → all bits off → M15 behavior exactly
(no fragmented blocks, no carry) + ~tens of µs of pre-pass.  Hot chunk →
full source-local replication for precisely the experts that are hot NOW,
bounded by the layer's set (which the per-layer statistics say covers
mean 69% / min 54%).  The convex tail term that killed pair #1 is capped.

## Validation contract (MoK-Updated)

The i.i.d. histogram replay CANNOT distinguish M19 from M18 (no chunk
variance ⇒ bits always on).  The harness therefore gains captured-route
replay: the serving hook dumps raw sampled `topk_ids` (int16 [T,8] per
call, keyed by layer), and the mok_eager protocol rotates a stack of real
captured routes across timed iterations (route write is outside the timed
region).  p50 over rotated-real-chunk iterations is the number expected to
transfer to serving; per-route times give the per-chunk distribution.

Ladder: M15 / M18(per-layer) / M19(θ sweep: 0, 32, 64, 128) under captured
replay for a high-variance layer and a stable layer; then the full gate
ladder; then ONE serving pair (per-layer static M18 vs M19 vs stock).

## Costs and bounds

- Pre-pass: ~32k global atomics + 2 grid barriers ≈ tens of µs on a ~6 ms
  kernel; measured, not assumed, before any claim.
- rep_dec: 512 B symmetric per rank; 32 B/peer/epoch of extra traffic.
- Shared: +~300 B (decision masks) on a ~30 KB budget.
- Descriptor: 65 words under ADAPTIVE (slot 64 = rep_dec base).

# exp_21 — mode 12: direct remote bf16 accumulation in the M7 epilogue (A11/M11)

**Pre-registered.** Written before any build. Bound by the exp_18 protocol
review (`exp_18_direct_accumulate/protocol_review.md`), which carries the full
ordering/zeroing/numerics analysis; this file is the execution plan.

## Hypothesis

The M7 inflation under CTA specialization (+1,249 µs = ~434 capacity + ~815
interference) is dominated by the **payload protocol itself**, not the role
split: M7's epilogue writes `part` (312 MB), the pool reads `part` (312 MB),
the pool writes peer `slots` (312 MB) — three touches of every byte inside the
M7 window. If the epilogue accumulates **directly into the owner's slot** with
the remote packed-bf16 atomic exp_18 proved real over xGMI, all three collapse
into one remote write pass: **~936 MB → ~312 MB in the M7 window, and the
service pool's payload role disappears** (it keeps readiness bookkeeping only,
so `C` can fall from 64 to ~8–16, refunding most of the capacity tax too).

## Falsifiers / success criteria (pre-registered, from the review)

- **F1 (fabric-atomic rate).** If mode 12 lands ABOVE mode 2's 6,866.1 µs at
  its best C, the payload was not the limiter: the remote atomic *op rate*
  (not bytes) is the wall, and A11 closes on this hardware with the rate
  measured. Anything above ~6,700 µs means the same at weaker strength.
- **F2 (mori heap grain).** If the dual-write detector (cfg bit 34) or the
  soak shows lost updates on the mori HIP-VMM `HeapType::Uncached` heap,
  blocker 1b is LIVE and mode 12 cannot ship; that is a hardware finding worth
  publishing (documented UB biting in practice).
- **Success:** mode 12 at best C < mode 2's 6,866.1 through the full gate
  ladder, reproduced by a second independent 5-rotation campaign (the sub-5%
  rule).

## Pre-build microbenchmark (informs F1 reading, not the design)

Extend the exp_18 standalone HIP test with a **throughput** phase on
coarse-grained hipMalloc (rate depends on the fabric path, not heap grain —
grain correctness is covered in-harness by the dual detector):

- `stores`: the pool's `store_peer_packets` pattern, 16 B/lane peer writes.
- `atomics`: the epilogue's pattern — each 32-lane half-wave issues 32
  *consecutive* dwords (128 B contiguous) per step, the exact mode-7 pattern.
- `atomics_scattered`: same op count, no two atomics in one wave on one line.

Measure ops/s at pool-like concurrency (64 CTAs) and epilogue-like concurrency
(192 CTAs). The interesting number: **whether the fabric merges same-line
remote RMWs** — if atomic throughput tracks stores, F1's risk is low.

## Build (single-variable vs mode 2 wherever possible)

1. `packet.cuh`: add `accumulate_peer_bf162` (additive primitive; existing
   callers bit-identical). Document the grain/scope precondition.
2. `moe_mps_adapter.cuh`: `mode_is_stream += {7}`; `mode_is_remote_accum`;
   validation (mode 12 ⇒ g==1, pull_fallback==0, C>0); cfg bit 34 = `detect`
   (dual write); `enqueue_tile_release<Scope>`; `service_env.mode`;
   `run_service` mode-7 body (no payload: arrivals → chunks-done → flags).
3. `n2_phase2_gm_mps.cpp` (`#ifdef N2GM_M7_REMOTE_ACCUM`): LDS peer-base
   table + heap-relative slot offset, built once per body from desc words that
   exist only in the MPS inclusion; PHASE-2 addressing branches on it.
   Standalone inclusion sees `nullptr` and is bit-identical.
4. `k0pf6gm_device_tile_mps.hip`: SRC_REV 17; entry guard (mode 12 ⇒ MAXTOK
   power of two); `k0p6_mps_task_done` enqueues at system scope for mode 12;
   M8 consume-and-zero (owner zeroes each fanout row after reading it — sound
   because producers never re-touch a flagged row, and the retirement gate
   orders the zero before any next-epoch accumulate); dual-detect compare.
5. Detector run at the winning config; then screens; then decision campaigns.

## What mode 12 deliberately does NOT change

- `slots` layout `[world][MAXTOK][7168]`, `row_ready` indexing, `pull_src`/
  `pull_ptr` meanings, M8's address math (`base_slot` is already exactly the
  accumulate target), M9 retirement, ABI (63 words), host code. Mode 2 stays
  the validated ratchet arm; the A/B is the same kernel at two configs.
- Task-done's all-lane `vmcnt(0)` + `__syncthreads()` + tid0 release chain —
  only the release/acquire *scopes* strengthen (agent → system) for mode 12.

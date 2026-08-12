# exp_21 — mode 12: direct remote bf16 accumulation in the M7 epilogue (A11/M11)

**Pre-registered.** Written before any build. Bound by the exp_18 protocol
review (`exp_18_direct_accumulate/protocol_review.md`), which carries the full
ordering/zeroing/numerics analysis; this file is the execution plan.

## Hypothesis (REVISED after exp_20: protocol, not payload)

exp_20 measured the pool's entire 896 B payload copy at **+5.7 µs of M7** —
the ~815 µs interference is the readiness **protocol** (arrival atomics
~bounded at 300–350 µs, plus the flush's ~2,400 system releases as named
prime suspect), not bytes. exp_17's "fewer bytes" lever is retracted;
**exp_21's hypothesis is re-based to the two survivors**:

1. **The combine win.** The payload copy costs **480 µs of combine** (exp_20
   mode 7 vs mode 2+pull_fallback). Mode 12/13's direct remote accumulate
   deletes it, and deletes the 312 MB `part` write + 312 MB pool read from the
   M7 window as a side effect (free per exp_20) — while keeping the
   **protocol fencescape byte-identical to the ratchet** (agent enqueue,
   agent event acquire, acq_rel counters, system flush): zero added fences,
   so it should not import new interference. The epilogue's remote-RMW
   stream is the new variable to bound (ubench + the falsifier).
2. **The protocol-weight win (mode 13 = X1).** Per-row target counters
   (`nc_arr[r]`, target `16·row_rem[r]`) delete the `pushed` counter and all
   per-chunk machinery — **~40% of the stream protocol's agent RMWs**, sound
   here because the service reads no payload (exp_20 §2a) so no per-chunk
   acquire edge is needed; the single cell's release-sequence chains all 16
   producers into the flagging wave.

Mode 9 (flush-fence scope swap, exp_06-style diagnostic) ships in the same
build to attribute the ~2,400-µs-class prime suspect exp_20 named.

## Falsifiers / success criteria (pre-registered, from the review)

- **F1 (fabric-atomic rate).** If mode 12/13 lands ABOVE mode 2's 6,866.1 µs at
  its best C, the epilogue's bursty remote-RMW stream is importing the unthrottled
  fabric-write regime exp_20 §2b priced at up to ~600 µs, and A11 closes on this
  hardware with the rate measured. The ubench's `atomic-PEER`/`atomdr-PEER` arms
  are built to read exactly this before a campaign does.
- **F2 (mori heap grain).** If the dual-write detector (g bit 4) or the soak
  shows lost updates on the mori HIP-VMM `HeapType::Uncached` heap, blocker 1b
  is LIVE and modes 12/13 cannot ship; that is itself a hardware finding
  (documented UB biting in practice).
- **F3 (fence-parity).** If mode 12's M7 stamp differs from mode 2's by more
  than the ±40 µs band **excluding** its own epilogue delta, the fencescape
  ceases to be identical — investigate before believing (prevents
  mis-attributing a new fence to the epilogue).
- **Success:** mode 12 AND/OR 13 at best C < mode 2's 6,866.1 through the full
  gate ladder, reproduced by a second independent 5-rotation campaign
  (sub-5% rule). Mode 13 out-performing mode 12 establishes X1 (protocol-weight
  reduction) as the direction for M4-style aggregation.

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

1. `packet.cuh`: `accumulate_peer_bf162` (additive primitive; existing callers
   bit-identical), grain/scope precondition documented.
2. `moe_mps_adapter.cuh`: `mode_is_stream += {12,13,9}`; `mode_is_direct_accum`;
   validation (12/13 ⇒ physical g==1, pull_fallback==0, C>0); detector via `g`
   bit 4 (g = 1|0x10, exp_20 idiom — zero host-bridge change); `run_service`
   mode-12 body (protocol-identical: same counters as mode 2, no payload);
   mode-13 body (per-row counter, `pushed` deleted); mode-9 flush-fence swap
   (untrusted diagnostic). Fences stay byte-identical to mode 2 for 12/13.
3. `n2_phase2_gm_mps.cpp` (`#ifdef N2GM_TASK_DONE_HOOK`): LDS peer-base table +
   heap-relative slot offset built once per body; PHASE-2 addressing branches
   on it; standalone inclusion sees `nullptr`, bit-identical.
4. `k0pf6gm_device_tile_mps.hip`: SRC_REV 20; entry guard (12/13 ⇒ MAXTOK
   power of two); task-done enqueues (agent) for 2/3/4/7/8/9/12/13 — restores
   mode 8's enqueue after the merge dropped it; M8 consume-and-zero for 12/13;
   dual-detect compare.
5. Detector run at the winning config; screens; decision campaigns.

## What modes 12/13 deliberately do NOT change

- `slots` layout `[world][MAXTOK][7168]`, `row_ready` indexing, `pull_src`/
  `pull_ptr` meanings, M8's address math (`base_slot` already names the
  accumulate target), M9 retirement, ABI (63 words), host code. Mode 2 stays
  the validated ratchet arm; the A/B is the same kernel at several configs.
- Task-done's all-lane `vmcnt(0)` + `__syncthreads()` + tid0 `release(agent)`
  chain — IDENTICAL to mode 2 (the maximal-system-scope protocol-review chain
  was deliberately NOT shipped after exp_20 priced fence count; the soundness
  of the agent-scoped chain rides on the vmcnt fabric-ACK semantics measured
  in exp_18 plus the flush's existing system release + owner acquire).

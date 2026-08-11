# exp_21 design — mode 12 (A11/M11): direct remote bf16 atomic accumulation

A CTA role-split megakernel change confined to four sites: the M7 epilogue
write path, the task-done event scope, the service loop body, and the M8
combine. Everything else is byte-identical.

## Task graph / roles

- **Compute CTAs (256−C):** M7 as today, but the epilogue's
  `unsafeAtomicAdd(part + xtok*H + col)` becomes
  `unsafeAtomicAdd(owner_pos_addr(xtok, col))` where
  `owner = xtok >> log2(MAXTOK)`, `pos = xtok & (MAXTOK−1)`, and
  `addr = tab[owner] + slot_off + (pos*H + col)*2` with `tab[cur] =
  local_heap_base` and `tab[r] = heap_bases[r]` otherwise, and
  `slot_off = (slots − local_heap_base) + cur*MAXTOK*H*2`. tab lives in 64 B
  of LDS, filled once per body by threads 0..7 from the desc-resident IRIS
  descriptor (the exact `translate_peer` arithmetic, pre-tabulated); `slot_off`
  is one SGPR pair. This is the review's step 4, with `MAXTOK = 4096 = 2^12`
  making div/mod a shift/mask (kernel entry guard rejects mode 12 otherwise).
- **Service CTAs (C, small now):** `run_service` with `mode == 7` consumes the
  same event queue and does the same `(r,nc)` arrival RMW with the same
  `row_rem` target, but the completing lane bumps the repurposed `pushed[r]`
  counter by 1 toward **16 completed chunks** (no payload push, no `claim`, g
  pinned 1). `retire_pushed_row(..., g=1)` is exactly this operation already —
  the mode-7 body is the existing code with the copy calls elided.
- **Owner M8:** unchanged addressing (`base_slot` already names exactly the
  accumulate target), dynamic tickets as in stream mode. **New:** after
  reading a contribution row, the same wave zeroes it (fanout-guarded, same
  shuffle geometry as the read). This restores the all-zero invariant the
  accumulate needs next epoch, at 312 MB of local stores in the combine tail
  instead of 448 MiB in M0 or a second 448 MiB parity buffer — and with zero
  ABI change.

## Edges (the ordering chain, after exp_18's review)

1. compute CTA epilogue: remote atomics → per-thread `vmcnt(0)` →
   `__syncthreads()` → tid0 `thread_release<system>` → event store
   (`enqueue_tile_release<system>`: the release's *scope* is the only change;
   the queue/ticket are agent-local and stay so).
2. service wave: event load + `thread_acquire<system>` (mode 12 only) →
   agent-scope nc_arr RMW → last chunk of row ⇒ `flush_pending`: wave drain +
   `release_signal_batch_system()` + system/agent `row_ready = epoch32` to the
   owner (unchanged code).
3. owner M8: `poll_epoch_system` + `acquire_payload_system()` (buffer_invl2 —
   already present) → reads, then zeroes, the slot row.
4. retirement: M9 `combine_done == 256` ⇒ every consumer wave finished ⇒
   zeroing complete before `retired` pokes; peers' next-epoch M0 waits on
   `retired >= epoch-1` ⇒ no peer's next accumulate precedes the zeroing.
   This replaces the review's parity double-buffer with the strictly cheaper
   consume-and-zero, made sound by *this* harness's failure semantics: any
   pperr != 0 is terminal (SystemExit 3 — verified in ab.py), so a dirty
   unconsumed slot is never reused, and the host `slots.zero_()` covers epoch 0.

## Numerics

The bf16 fan-in degree per output element is unchanged (row_rem[r] adds,
unordered): only the *location* of the accumulator cell moves (owner's memory
vs local `part`) and one addend's plane grouping collapses into the shared
cell. Tolerance gates are unaffected (reviewed, §*Numerics*).

## Detector (cfg bit 34, `detect`)

Epilogue dual-writes (remote slot atomic AND local `part`); M8 in detect mode
additionally pulls the producer's `part` row (mode-0 `base_pull` addressing)
and compares bf16 values elementwise against the consumed slot row with a
tolerance of `2^-7 * max(|a|,|b|)` (two independently ordered bf16 fan-ins of
the same addends can legitimately differ by ~1 ulp; a lost update differs by
one whole addend). Any violation sets `pperr` bit `1 << 28`
(`K0P6_MPS_ERR_DUAL`). Run once per candidate config before trusting its
timing. Cost: ~2× M8 traffic — diagnostic only.

## Buffers / ABI

Unchanged: 63-word ABI, `slots` 448 MiB, all MPS buffers. `pushed` keeps its
M0 zeroing and gains the "chunks done" meaning in mode 12. `part` is dead in
mode 12 (its M3 zeroing stays for v1 — fused with the scale transpose M6
needs; a later ablation can strip it).

## Config space

- mode 12: `C ∈ {4,8,16,32}` expected optimal (bookkeeping-only pool;
  fall-through drains the tail); `g = 1`, `flush_rows = 16`, `pull_fallback = 0`.
- bit 33 timestamps as usual; bit 34 detect for the certification run.

## Alternatives rejected

- **Parity double-buffer for slots** (review's option 4): +448 MiB symmetric
  heap and ~112 µs of M0 zeroing on the critical path, to solve a hazard the
  retirement gate + terminal-pperr semantics already solve. Rejected.
- **End-of-epoch owner zeroing of the whole 448 MiB**: strictly dominated by
  consume-and-zero (448 MiB vs 312 MB, and the combine tail is already
  CTA-abundant).
- **`onesided: pool pushes reduced rows as remote atomics`**: needs the same
  4-B atomic op count with an extra hop and a still-live `part`; dominated by
  both mode 2 and mode 12.
- **No service pool (compute-side flags):** per-row completion needs a
  cross-CTA count no single CTA holds; moving the RMWs into the M7 epilogue
  lengthens the critical epilogue. The C CTAs are nearly free (C ~ 8-16).

## Synthesis with the competition analysis (docs/distributed/competition-analysis)

Read against the leaderboard winners, mode 12 is deliberately NOT a bigger
service pool — it is the two mechanisms they actually won with:

- **Elimination (axis G).** The winners fused the collective into the GEMM
  epilogue and deleted data passes. Mode 12 deletes the `part` write and the
  pool's read+push pass entirely: the combine's transport is folded into the
  existing G-stack epilogue, SC24's 12% shape.
- **Outstanding-request depth, not CU count (axis F).** gemm-rs rank02 sized
  its network pool to ~7-16 CTAs and bought bandwidth with 32 in-flight
  16-B loads/lane. Mode 12's transport depth comes free: the epilogue already
  issues 16-32 independent atomics per thread per task (they're the same
  `unsafeAtomicAdd` stream the donor emits, re-pointed), so the payload moves
  at high memory-level parallelism with ZERO service-CTA capacity spent on
  moving bytes. The residual pool (readiness bookkeeping) then wants
  `C ≈ 8-16`, which is exactly the topology/depth sizing the analysis
  prescribes — our current C=64 (25% of the grid) is plausible interference
  over-provisioning per the analysis, and mode 12 lets the sweep find out.

The analysis's strongest counter-evidence is aimed at what mode 12 KEEPS, not
at what it removes: **per-tile/per-row readiness signalling** (gemm-rs rank05's
"our design" was 23% slower than one barrier; exp_05-07 measured the same
pressure here). If mode 12's profile still shows a large bookkeeping term,
the follow-ups are: per-row target-16·row_rem counters (deletes `pushed`,
~40% of remaining protocol atomics; STATUS's live-queue item 1 with its
exp_09 objection withdrawn), exec-mask fan-in instead of atomics where
fan-in ≤ 64, and harder poll backoff (exp_20 mode 8's axis).

Where we disagree with the analysis's defaults, on record:
- **Pull-vs-push symmetry**: mode 12 keeps producer-side writes; the analysis
  prefers pull. At 4-B accumulate semantics the pull cannot exist (a remote
  RMW is inherently initiated by the contributor). So this particular
  recommendation is pre-empted by the numeric mechanism — recorded so the
  review can argue it.
- **The ticket machinery (their "do not build" #11)**: the streaming tail
  needs the dynamic ticket (exp_12's +1.464x→1.077x is measured, not
  speculative). The analysis's cold-water is about tickets for *overlap
  scheduling*; the drain ticket serves *work distribution at the tail*.

### Named follow-up candidates (for the ablation queue, not this build)

1. **Direct-to-out accumulation.** Take elimination to its limit: the
   epilogue accumulates weighted contributions straight into the owner's
   `out[tau]`, retiring the slots buffer AND M8 AND the readiness protocol —
   the service-pool concept itself disappears (the only per-epoch completion
   needed at kernel scope is the retirement handshake). Needs a tau-delivery
   channel in dispatch (one 4 B remote store per (token, route) at M1,
   `tau_table[src*MAXTOK+pos] = tau`), `out` zeroed per epoch (58 MB/rank,
   ~10 µs, end-of-prior-epoch), and an M9 cross-rank arrival before launch
   return. Numerics: per-cell bf16 fan-in degree grows from ~row_rem to
   ~top-k·(producers) with no fp32 plane combine — a tolerance-gates
   question, not a new rounding class. This is the mode-16-scale experiment;
   mode 12 measures its shared risk (fabric atomic rate) first, cheaply.
2. **Dependency-ordered tile scheduling (axis E).** M3-M5 already computes a
   plan; order M7's tiles so rows complete STEADILY rather than in bursts
   (flattens the demand curve; strictly cheaper than role machinery, costs
   zero CTAs). Substitute for phase-adaptivity on the capacity term — should
   be measured as an alternative, per the analysis.
3. **Demand-curve instrumentation.** Timestamp each tile-event enqueue;
   histogram; mean/peak × the bookkeeping pool's capacity share is the hard
   ceiling for any adaptive-join mechanism.
4. **Bandwidth-aware ticket (novel, from the letter).** A drain-join rule
   driven by queue depth AND a memory-pressure proxy rather than by
   completion order alone — the genuinely new scheduling idea, to be invoked
   only once the counters exist.

## Invariants to check in the ISA/build gate

- Resource tuple: `SGPR ≤ 104+ε, VGPR 256, AGPR 256, scratch ≤ 60 B`, zero
  scratch ops inside both MFMA K-loops; MFMA census 96+84=180 unchanged.
- `global_atomic_pk_add_bf16` present; epilogue-addressing arithmetic must not
  be in the K-loop; the LDS table read must be a plain `ds_read2_b64` (or two
  `ds_read_b32`), not a spill.
- `unsafeAtomicAdd` on the remote address must not lower to a CAS loop
  (same `-munsafe-fp-atomics`-class lowering as the existing local path).

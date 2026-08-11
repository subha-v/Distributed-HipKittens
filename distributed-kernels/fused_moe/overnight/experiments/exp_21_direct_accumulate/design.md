# exp_21 design — mode 7 (A11/M11): direct remote bf16 atomic accumulation

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
  making div/mod a shift/mask (kernel entry guard rejects mode 7 otherwise).
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
2. service wave: event load + `thread_acquire<system>` (mode 7 only) →
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
M0 zeroing and gains the "chunks done" meaning in mode 7. `part` is dead in
mode 7 (its M3 zeroing stays for v1 — fused with the scale transpose M6
needs; a later ablation can strip it).

## Config space

- mode 7: `C ∈ {4,8,16,32}` expected optimal (bookkeeping-only pool;
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
  both mode 2 and mode 7.
- **No service pool (compute-side flags):** per-row completion needs a
  cross-CTA count no single CTA holds; moving the RMWs into the M7 epilogue
  lengthens the critical epilogue. The C CTAs are nearly free (C ~ 8-16).

## Invariants to check in the ISA/build gate

- Resource tuple: `SGPR ≤ 104+ε, VGPR 256, AGPR 256, scratch ≤ 60 B`, zero
  scratch ops inside both MFMA K-loops; MFMA census 96+84=180 unchanged.
- `global_atomic_pk_add_bf16` present; epilogue-addressing arithmetic must not
  be in the K-loop; the LDS table read must be a plain `ds_read2_b64` (or two
  `ds_read_b32`), not a spill.
- `unsafeAtomicAdd` on the remote address must not lower to a CAS loop
  (same `-munsafe-fp-atomics`-class lowering as the existing local path).

# Minimum-progress specialization (MPS) — design record

Status: **unmeasured design + implementation**. No GPU gate has run. The performance
target is the exact exp_35 donor at G=3/c4: **6,919.8 µs vs 7,698.0 µs production**
(`M7 = 1,844.5 µs`, `M8/M9 = 1,308.5 µs`). The parity port
(`k0pf6gm_device_tile.hip`) is untouched; this design lives entirely in additive
siblings.

COMET/MoK framing, and what this design is *not*: the role split is host-selected
and compile/descriptor-fixed before launch, exactly like COMET's profiled `nc` and
MoK's `fwd_num_comm_sms`. There is no in-kernel resizing. The AMD-specific idea is
**minimum-progress specialization**:

> Reserve the *minimum* CTAs needed to guarantee communication progress, keep
> communication waits/fences off the GEMM CTAs, and let finished compute CTAs
> elastically join communication/reduction work.

## 1. Observation that motivates the design

The donor kernel is phase-uniform (every CTA runs every phase), but its M7
(phase-2 GEMM) task graph is already COMET layer-1 shaped:

- Every M7 task is `(tile, nc)`: one CTA produces a `[32·G] × 448` output tile.
- 16 independent N-chunks per 32-row block `b`; the epilogue atomic-adds into
  `part[r][nc·448 .. nc·448+448)`.
- Receive row `r` is final in slice `nc` exactly when **all** of its
  `row_remaining[r]` contributing blocks have finished their `(b, nc)` task.

Today those 16 completion events per block are collapsed into one bulk,
post-barrier M7.5 publication, and the owner then *pulls* ~273 MB of remote
`part` rows on the critical path (peer-read floor ≈ `273 MB / 355 GB/s ≈ 769 µs`;
posted-write floor ≈ `273 MB / 440 GB/s ≈ 620 µs`; the rate needed to hide all of
it under M7 is only ≈ 148 GB/s). The specialization moves readiness observation
*and* the data movement off the GEMM wave and under M7:

1. M7 completion is published per `(b, nc)` through a single-writer event queue
   (one relaxed store + one agent release on the compute CTA; no system fence,
   no row loop, no per-task remote traffic).
2. A small pool of **service CTAs** (reserved by finish order at M6 end)
   consumes events, does row bookkeeping (arrival counters), and **pushes**
   completed N-slices into owner-resident slots with controlled 16-byte stores.
3. When a row's 16 slices have all been pushed, the service wave publishes
   `row_ready[producer][row]` directly to the *owner only* (1 store vs 8),
   batched behind one system release per flush group.
4. M8 becomes an owner-side **local-slot** reduce of the same arithmetic shape;
   batch claiming is dynamic so drained compute CTAs (and, at the end, service
   CTAs) all flow into reduction work.

The M7.5 grid barrier disappears *as a consequence*: its only load-bearing duty
in the current dataflow was to make `part` complete before publication. In the
streaming design, per-slice completion is tracked explicitly, so no global
rendezvous is needed between M7 and combine. (The terminal `combine_done`
rendezvous in M9 is unchanged.)

## 2. Task graph and roles

```
all CTAs:  M0 (retire wait, fails-closed resets incl. MPS buffers)
all CTAs:  M1 qpush (unchanged)
all CTAs:  M2 chunk-acquire unpack + row_remaining (unchanged)
all CTAs:  M3–M5 barriers, plan, scatter, part zero               (unchanged)
all CTAs:  M6 phase-1 GEMM, 256-CTA stride                        (UNCHANGED:
           specialization never taxes M6 — it is the G=3 win)
M6.9:      role ticket: first C finishers become service CTAs;
           others get dense compute ids 0 .. 255-C-1
compute:   M7 tasks strided by (256-C), hook per (b, nc):
             per-thread s_waitcnt vmcnt(0) → __syncthreads →
             tid0: agent release → event store (b<<4|nc|MARK) → [diag: ts]
service:   consume event stripes → arrivals/claims → push slices →
             batch flags to owner → join reduction
compute:   after task drain → dynamic-ticket M8-slot reduction
all CTAs:  M9 combine_done arrival, resets, retired pokes         (unchanged)
```

Expected counts (all provable from device-published state):

- events `E = (nvi[0] >> 5) * 16` — 16 hook calls per live 32-block;
  `nvi[0]` is the padded row count from M4's scan, read after the M5 barrier.
- per `(r, nc)` arrivals target `= row_remaining[r]` (M2's popcount).
- per row `r`: `16/g` group pushes; `pushed` total 16 slices before flag.
- owner flags: exactly one `row_ready = epoch32` store per live row per epoch
  (uniqueness: the slice/group claim RMW elects a single publisher per row).
- M8 batches: `ceil(T/4)` claimed monotonically by ticket; each batch processed
  by exactly one wave.

## 3. Producer/consumer roles and finish-order partition

`finish_order_partition` (new `kittens::distributed` primitive): each CTA leader
`fetch_add`s a monotonic agent counter when its M6 stripe completes. Ticket `< C`
→ service; else compute with dense id `ticket - C`. Properties:

- no extra grid barrier (the partition precedes M7, and M6's own completion
  ordering is the ticket order);
- service CTAs exist from as early as the first M6 finisher → earliest possible
  communication progress;
- dense compute ids keep the M7 task loop exact: `for (task = id; task <
  num_tasks; task += 256-C)` covers every task exactly once;
- the role is per-epoch (cell zeroed at M0); rank-local (no cross-rank state);
- compute CTAs that drain the task queue fall into the dynamic reduction queue;
  service CTAs join it after their event stripes drain — COMET's fixed pools,
  but with elastic late joining because CTAs are not reserved for an entire
  operator.

Register/occupancy note (AMD-specific): gfx950 has no `setmaxnreg`; service CTAs
pay the union kernel footprint. The benefit is therefore *not* occupancy — it is
that GEMM waves never poll remote readiness, never execute system fences, and
never run remote-copy loops, while communication starts at the first completed
N-slices. The measured capacity tax of reserving `C` CTAs is bounded:

```
T_M7(C) ≈ 1844.5 · 256 / (256 - C)       C=4 → +29 µs, C=8 → +60 µs,
                                          C=16 → +123 µs, C=32 → +264 µs
```

Break-even at C=8 vs the ~217 µs campaign floor needs ≳ 277 µs of recovered
tail — hiding less than half of the estimated 620–769 µs transit suffices.

## 4. Buffers, ownership, and lifetime

All new descriptor slots are appended (56..62); donor slots 0..54 and slot 55
(`K0P6_D_SYMMETRIC`) are bit-unchanged. `K0P6_MPS_D_LEN = 63`.

| slot | name | bytes (at PADMAX=263136, MAXTOK=4096) | visibility | lifetime |
|---:|---|---|---|---|
| 56 | `K0P6_D_MPS_Q` | (PADMAX/32)·16·4 = 526,272 | agent | zeroed at M0; event words are monotonic within the epoch; high bit MARK makes 0 = empty |
| 57 | `K0P6_D_MPS_NCARR` | T_ext·16·4 = 2,097,152 | agent | zeroed at M0; reaches `row_remaining[r]` exactly |
| 58 | `K0P6_D_MPS_PUSHED` | T_ext·4 = 131,072 | agent | zeroed at M0; counts pushed slices, target 16 |
| 59 | `K0P6_D_MPS_CLAIM` | T_ext·4 = 131,072 | agent | zeroed at M0; per-row push-group claim bits |
| 60 | `K0P6_D_MPS_STATE` | 128 | agent | zeroed at M0; tickets + diagnostic timestamps |
| 61 | `K0P6_D_MPS_SLOTS` | 8·MAXTOK·7168·2 = 448 MiB | **system** (peer-written) | epoch-tagged: stale slots are never read because `row_ready` is epoch-gated; producers of epoch N+1 pass the retired ≥ N gate before overwriting |
| 62 | `K0P6_D_MPS_CFG` | 8 | host-written | packed: C[0:8), g[8:16), mode[16:24), flush_rows[24:32), flags[32:40) |

`row_ready` and `part` keep their donor semantics (row_ready ABI is *unchanged*:
producers write `row_ready[cur·T_loc_max + r] = epoch32` into the owner; owners
poll the same words in M8). `part` keeps its donor role as the local
write-accumulator for M7 epilogue atomics — it now never crosses xGMI.

Event word: `0x80000000 | (b << 4) | nc` (bcap ≤ 16383 guard; bcap = 8223 here).

## 5. Release/acquire edges (all checked against the parity discipline)

| edge | producer | consumer | mechanism |
|---|---|---|---|
| block-tile payload (part atomics) → event | M7 task threads: per-thread `s_waitcnt vmcnt(0)` → `__syncthreads` → tid0 `thread_release<agent>` → relaxed event store | service wave: bounded relaxed poll on slot, then per-lane `acquire_fence<agent>` | agent scope (same GPU; atomics retire at L2) |
| pushed slice payload → row flag | service wave lanes push packets, `s_waitcnt vmcnt(0)` + `__syncwarp` → lane0 `thread_release<system>` → lane0 relaxed flag store(s) (same-thread fence+store parity discipline, wave granularity) | owner M8: existing system-scope epoch poll + `acquire_payload_system` | system scope; self-flags use agent |
| arrival counters | lane-parallel relaxed `atomicAdd` agent | lane-parallel relaxed loads agent | counters are exact (target = row_remaining) |
| queue slot liveness | single writer per slot (capacity = max events) | nonzero test (MARK bit) | zeroed at M0; capacity ≥ E structurally |
| epoch replay | M9 `combine_done == 256` ⟹ all flags observed ⟹ all pushes landed; then release + `retired` pokes | peer M0 retirement wait (unchanged) | slots/queue/counters reused next epoch only after this gate |

Failure model (fail-closed, terminal): an a2-wait timeout aborts the M7 body
early; its remaining events never enqueue. Service polls are bounded and watch
`pperr` (new bit `1<<26` service timeout); owner M8 polls are bounded (existing
bit `1<<25`); all paths converge to M9's counted arrival so the launch retires
with `pperr != 0`. Coordinated all-rank reinitialization is required before
relaunch — identical to the parity contract.

`row_remaining` self-clean invariant (load-bearing since exp_62): M2 writes
`row_remaining` only for live rows; parity's M7.5 zeroes every row it publishes
so holes start at 0 next epoch. The MPS service wave preserves this: the unique
flag publisher of row `r` zeroes `row_remaining[r]` immediately after its flag
store. Modes 0/1 retain the parity self-clean verbatim.

## 6. Slice grouping `g` and flag flush batching

- `g ∈ {1,2,4,16}`: adjacent N-chunks claimed and pushed as one contiguous
  `896·g` byte transfer. The last bumper of slice `(r, nc)` checks its group;
  the single claimant pushes the group. Grouping exploits the measured burst
  structure (the 16 `(b, nc)` tasks of a block are adjacent task ids and finish
  within µs of each other) to reduce push count from 349k to 87k at g=4 without
  delaying any row's *last* contributing block.
- `flush_rows ∈ [1,64]`: row flags accumulated per service wave behind one
  system release. At queue drain, every wave flushes its remainder.
- The user-prescribed sweep `g ∈ {1,2,4}` maps directly onto the group knob;
  `C ∈ {0,4,8,16}` onto the reservation knob.

## 7. Modes (one binary, descriptor-selected)

| mode | name | behavior | what it isolates |
|---:|---|---|---|
| 0 | reserve-only | tickets taken, service CTAs skip M7; parity M7.5 bulk publication + parity M8 pull **unchanged** | pure M7 capacity tax `256/(256-C)` |
| 1 | bulk-push | all 256 CTAs run M7; parity barrier; grid-strided push to slots; leader fence + per-row flags to owner only; M8-slot (static stripes) | copy/layout gain without overlap |
| 2 | stream | full design in §2 | overlap gain, end to end |

Mode-2 flag bit `owner_pull_fallback`: stream everything (flags arrive early)
but M8 pulls remote `part` as today — separates "early readiness" from
"push transport" in attribution.

## 8. Diagnostic timestamps (cfg.flags bit0 enables)

`K0P6_D_MPS_STATE` carries `first_ready` (min event enqueue `s_memrealtime`),
`last_ready` (max), `queue_drain` (max over service waves), `m7_complete` (max
over compute CTAs), `reduce_done` (max over all CTAs). These are the sweep's
required first/last-ready, queue-drain, M7-completion, final-reduction marks.

## 9. Alternatives rejected

1. **Poll `part_bits[b]` bitmaps instead of an event queue.** Bitmaps are the
   prescribed readiness *abstraction*, but scanning ⌈bcap⌉ words per service
   pass is `O(blocks)` per CTA against `O(events)` total for the queue; the
   queue entry *is* the bitmap element (block + bit id), with one store on the
   compute CTA — same cost, no scan.
2. **Keep publication at per-16B-packet or per-slice system releases.**
   Measured history: per-row system publication ≈ 39.5× production (rejected);
   per-packet invalid. Release fences are amortized per flush group per wave.
3. **Static wave striping for M8.** Leaves reserved CTAs' share undone until
   they join; dynamic tickets keep reduction NUMA-clean and straggler-free.
4. **Direct remote bf16 atomic accumulation into `out`.** Eliminates slots
   entirely but makes slice→output completion detection a remote-atomic
   ordering problem and changes the failure/replay analysis; slots keep the
   reduce local and the flag protocol identical to the validated one.
5. **Reserving CTAs at M6 start.** Costs ~2.5× the tax of reserving at M6 end
   (8 CTAs ≈ 149 µs vs ≈ 60 µs) and interferes with the G=3 phase.

## 10. Invariants preserved

- Donor descriptor words 0..54, slot 55, routing, plan, LL128 wire bytes,
  both MFMA bodies (phase-2 delta is loop bounds + one drain — see §11).
- One epoch per launch; all bounded waits fail closed; M9 retirement protocol
  bit-identical.
- `pperr` bitmap: existing bits unchanged; new bits are `1<<26` (service
  timeout) and `1<<27` (MPS config/shape guard).
- Correctness oracle: output is still the exact sum of every routed
  contribution; slots are staging only, with the same bf16 values M8 reads
  today — reduction order per element is unchanged (14-chunk loop, NT=4).

## 11. The phase-2 body delta (vendored copy, donor sha256 pinned)

`n2_phase2_gm_mps.cpp` = pinned `n2_phase2_gm.cpp` with exactly two code deltas
(diff-verified at authoring time):

1. `N2GM_TASK_START` / `N2GM_TASK_STRIDE` macros (defaults `blockIdx.x` /
   `kCTAs` = donor behavior) replace the loop bounds.
2. `N2GM_TASK_DONE_DRAIN_HOOK` placed immediately before the existing
   post-epilogue `__syncthreads()` (per-thread `s_waitcnt vmcnt(0)`), because
   `__syncthreads()` does not drain outstanding VMEM atomics — the exp_56 M1
   discipline applied at task end.

The includer's done-hook contract gains `nc` (`k0p6_mps_task_done(b, nc, tid,
desc)`). Nothing else in the MFMA pipeline, LDS layout, sched_group barriers,
or epilogue arithmetic changes.

## 12. Authoring-time validation ledger

- **Adversarial read-only review** (fresh-context kernel reviewer): one
  CRITICAL found and fixed before any build gate — the flag-pending LDS buffer
  could overflow when `flush_rows > 32` (up to 32 appends per event against a
  per-event flush test). Fixed by testing the flush threshold after EVERY
  append; `flag_count <= flush_rows <= 64` now holds structurally. One acquire-
  edge hardening adopted: the slice-group completeness probe and the arrival
  bump are ACQ_REL RMWs so the pushing wave holds an explicit acquire edge per
  chunk group regardless of which service wave consumed which event. One watch
  item logged for the ISA gate (role word is loop-carried across the phase-2
  mainloop; CT-uniform ⇒ SGPR, but the resource A/B must confirm).
- **Protocol model check** (host-side, seeded randomized routing + shuffled
  task-completion order, 12 configurations): for `g ∈ {1,2,4,16}` × 3 trials,
  every live row receives exactly 16 pushed slices, each N-chunk covered
  exactly once, and exactly one owner flag per row. This validates the
  claim/push/flag ALGEBRA (counts, grouping, dedup) — not the memory model,
  which is GPU-gate territory.
- **Host gates**: `distributed.cuh` + roles.cuh compile as strict C++20
  (`-Wall -Wextra -Werror -pedantic`) for CDNA3+CDNA4 and pass the existing
  host-semantics suite; `moe_host_abi.hpp`'s MPS extension compiles and passes
  new unit coverage (encode/decode round trip, binding validation, 63-word
  append/patch/validate, every rejection path).
- **Static invariants**: `check_port_invariants.py` extended additively
  (`check_moe_mps`); the whole suite passes. Preserved-region purity of the
  sibling was verified by scripted span diff: M1, M2, M3–M5, M6, and M9 are
  byte-identical to the parity port.
- **Not yet run**: any GPU gate. This source is unbuilt and unmeasured.


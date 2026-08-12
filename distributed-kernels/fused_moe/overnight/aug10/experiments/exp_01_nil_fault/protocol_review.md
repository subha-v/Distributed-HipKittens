# exp_01 — protocol review of the fixed `mps_mega` (read-only, pre-GPU-gate)

Memory-ordering / replay / epoch analysis of `k0pf6gm_mps_mega` against the parity oracle `k0pf6gm_mega`. The
nil-fault fix is not the subject but is **confirmed sound**: with `p = cur; row = 0` the default base is
dereferenceable under *both* address forms — `base_pull` → local `part` row 0, `base_slot` → `slots + 0`, because
`p == cur` cancels the `- cur*MAXTOK` term exactly (`..._mps.hip:373-388,1440-1446`). That cancellation is
load-bearing: any future default with `p != cur` makes `base_slot` compute a wrapped `size_t` and walk off the
448 MiB allocation. `hkp_sort.hpp`, `hkp_sync.hpp`, `k0pf6_chunk_release.hpp` and `n2_phase1_gm.cpp` are not in
this repo; every claim depending on them is labelled.

## Verdict

**Safe to launch and gate, not yet safe to trust a mode-2 PASS.** I found no unbounded-spin path (every wait is
`spin_limit`-bounded, all four M8 structural paths fall through to M9's counted arrival, so a failed launch still
retires) and no out-of-bounds write on any in-range config. But mode 2 has one edge that is *absent in source*,
not merely weaker than the reference: the row-completion flag is released by the wave that pushed the **last**
slice group, while the other `16/g − 1` groups of that row were pushed by **different waves on different
CTAs/XCDs** whose stores nothing drained or released (`moe_mps_adapter.cuh:374-392` vs `:241-263`). At `g = 16`
there is one claimant per row and the hole closes; at `g ∈ {1,2,4}` — the whole A2 sweep — it is open. Single
most likely remaining failure mode: **intermittent wrong numbers (rel_L1 / max_abs over tolerance) with
`pperr == 0`** in mode 2 — timing dependent, so it can slip a 1-iteration smoke and surface in the 600-epoch soak
or as unstable rel_L1 across the five rotated processes. Run the `g = 16` control first; it is the cheapest
discriminator in the set.

## Ranked residual risks

1. **Cross-wave slice-push visibility at flag time (mode 2, `g < 16`).** `push_slice_group` only *issues* the
   packet stores; `fetch_add_relaxed<agent>` on `pushed[r]` follows immediately with no drain and no release
   (`moe_mps_adapter.cuh:374-381`). The wave that observes `oldp + g == 16` drains only its **own** `vmcnt`
   (`:246-248` — vmcnt is per-wavefront) before its `thread_release<system>` and flag store (`:251-263`). Another
   wave's earlier groups for the same row are unordered w.r.t. that flag.
   MANIFESTS: owner M8 reads a partially-landed slot row ⇒ wrong numbers, `pperr == 0`, no fault, no hang; the
   error is a whole 896·g-byte slice of one token, so `max_abs` fails loudly when it fires and passes cleanly when
   it does not. Deterministic only if peer stores linger dirty in the writer's L2; if the mori heap is genuinely
   uncached they drain on their own and this stays latent (mechanism `UNCERTAIN`, missing ordering **certain**).
   CHEAPEST TEST: same-run A/B of mode 2 `g=16` (one claimant per row, hole closed) vs `g=4`, everything else
   fixed. `g=16` clean + `g=4` dirty ⇒ confirmed. FIX SHAPE: each pushing wave must `vmcnt(0)` +
   `thread_release<system>` **before** its `pushed[r]` bump — i.e. `counted_arrive_release_into<system>`
   (`counter.cuh:84-90`), which exists for exactly this pattern and is unused.

2. **`part` crosses the compute→service CTA boundary on an agent-scope edge only.** Mode 2 publishes tile
   completion with `thread_release<agent>` + relaxed agent store (`moe_mps_adapter.cuh:148-155` via
   `..._mps.hip:275-281`); the service wave acquires with `thread_acquire<agent>` (`:302`). The reference never
   does this — `part` is always published with `release_cta_payload_system()` on every CTA plus a grid barrier
   (`k0pf6gm_device_tile.hip:955-964`). Producer and consumer CTAs can sit on different XCDs, i.e. different 4 MB
   L2s. MANIFESTS: stale `part` bytes pushed into peer slots ⇒ wrong numbers, `pperr == 0`. NEEDS ISA CHECK — the
   question is whether the gfx950 agent-scope fence pair emits an L2 writeback/invalidate or degenerates to a bare
   `s_waitcnt`: `llvm-objdump -d --mcpu=gfx950 <hsaco> | rg -n 'buffer_wbl2|buffer_inv|s_waitcnt vmcnt\(0\)'` —
   require a `buffer_wbl2` between the task-end drain (`..._mps.hip:312`) and the event store, and a `buffer_inv`
   at `moe_mps_adapter.cuh:302`. If both are absent, mode 2 is unsound as written.

3. **`pull_fallback = 1` in mode 2 has no producer-side system release for `part` at all.** With `m8_pull` true the
   owner reads the producer's **remote** `part` over xGMI (`..._mps.hip:1415-1417,1435-1439`), but in mode 2 no CTA
   ever executes `release_cta_payload_system()`: the mode-0/1 blocks that do (`:1277`, `:1336`) are both skipped,
   and the service wave's system release (`moe_mps_adapter.cuh:251`) covers its own packet stores, not the compute
   CTAs' `part` lines. The M7.5 grid barrier that made this legal in the reference is deliberately gone.
   MANIFESTS: A4 reads stale remote `part` ⇒ wrong numbers with `pperr == 0`, easily misattributed as "transport
   was not the lever". CHEAPEST TEST: none needed, this is a source-level ABSENT edge. Treat A4 as **not runnable
   as a correctness arm** until a per-CTA system release is added at the mode-2 M7 tail — which re-taxes the GEMM
   CTA, so say so in A4's `result.md` rather than quietly adding the fence.

4. **`events_total` must equal the enqueued-event count exactly.** Service waves wait on queue *slot*
   `k < events_total = (nvi[0] >> 5) * 16` (`..._mps.hip:1197`, `moe_mps_adapter.cuh:292-295`) while producers fill
   slots in ticket order (`:152-154`). The producer count is `Σ gcount × 16` over tiles
   (`n2_phase2_gm_mps.cpp:219-233,443-450`, M4 table at `..._mps.hip:1060-1083`). Equality needs `nvi[0]` to be the
   padded row total, 32-aligned, with `scratch[K0P6_SC_ERB+0] == 0`. I **could not determine** that;
   `hkp_sort.hpp` is outside this repo. MANIFESTS: too high ⇒ tail waves spin out ⇒ `pperr` 1<<26 plus peers'
   33554432; too low ⇒ unconsumed events ⇒ rows never flagged ⇒ 33554432 on every rank. Loud either way.
   CHEAPEST TEST: `K0_MPS_DESC_DUMP` after one mode-2 iteration — compare `mps_state[K0P6_MPS_ST_TAIL]` (word 1 =
   the exact enqueue count) against `(nvi[0] >> 5) * 16`. One run, no rebuild. Do this before the soak.

5. **Wave-divergent `error_bit_set_agent` can wedge a CTA on `s_barrier`.** It reads `pperr` on lane 0 of **each
   wave** and broadcasts only within that wave (`moe_hk_adapter.cuh:196-205`), so four waves read at four different
   times. If a *different* CTA sets the bit inside that window, some waves take the `return` and others do not, and
   the next CTA-collective barrier hangs forever: `k0p6_a2_wait`'s own `acquire_payload_agent()` on the following
   task (`..._mps.hip:223-238`; `cta_acquire` → `__syncthreads`, `distributed/detail/config.cuh:62-65`), same
   pattern at `..._mps.hip:542,934,1047`. Byte-identical in the reference (`k0pf6gm_device_tile.hip:196-211`) ⇒
   parity-inherited, not an MPS regression. MANIFESTS: unbounded GPU hang (not a bounded spin), reaped only by the
   outer `timeout`; the launch never retires, so every peer's next M0 retire-wait fails and the host must
   reinitialize. Exactly the shape a deliberately-broken negative control provokes, because it times many CTAs out
   near-simultaneously. CHEAPEST TEST: run the negative control under `setsid timeout` and record whether it exits
   with `pperr != 0` (claim holds) or has to be reaped (claim violated). Never `SIGKILL`.

6. **M8's `row_ready` spin budget grew by the whole M7+push window.** In the reference the poll starts after the
   M7.5 grid barrier, i.e. after every flag is published (`k0pf6gm_device_tile.hip:964-1046`); in mode 2 the first
   drained compute CTA polls immediately (`..._mps.hip:1451-1500`) and must outwait the entire service transport.
   MANIFESTS: false `pperr` 33554432 on a *correct* kernel, indistinguishable from a real protocol failure.
   CHEAPEST TEST: repeat with `spin_limit` ×10; if the bit disappears it was budget, not protocol. Do this before
   believing any 33554432.

7. **The event-queue store is unbounded at the write site.** `enqueue_tile_release` does `fetch_add(tail)` then
   `q[ticket] = …` with no capacity test (`moe_mps_adapter.cuh:148-155`); the `events_total > queue_capacity` guard
   runs later and only on service CTAs (`..._mps.hip:1201-1202`). Bounded only structurally by the M4
   `nt >= bcap` and pad guards (`:1074`, `:1106`) — and **nothing returns on those bits**: the post-M5 check tests
   only 2097152 (`:1112`). The overflow bits are reference-identical; the unchecked heap store is new.
   MANIFESTS: corruption past `mps_q` (526 KB) ⇒ arbitrary wrong numbers, or a fault in an unrelated buffer.
   CHEAPEST TEST: assert `pperr & (4194304|1048576) == 0` in the harness on every run; if it can be nonzero, bound
   the ticket at the enqueue.

8. **M0 resets the MPS buffers with plain stores while every consumer uses atomic RMWs** (`..._mps.hip:562-571` vs
   `moe_mps_adapter.cuh:152,329-343,354,379` and `..._mps.hip:1476,1490`) — the exp_56 fix (7/H2) hazard that
   forced system-scope atomic resets for `dest_counter` (`..._mps.hip:589-595`). **Counter-evidence that demotes
   this to LOW:** `a2_done`/`part_done` are also plain-reset (`:546-549`) and atomically RMW'd, and the reference
   passes; the distinguishing factor for `dest_counter` was *system* scope, and every MPS buffer is agent-scope
   local. MANIFESTS: stale `mps_q` word ⇒ phantom event; stale `pushed[r]` ⇒ early flag; stale `M8NEXT` ⇒ skipped
   M8 batches ⇒ zeros in `out` with `pperr == 0`.

9. **`symmetric == nullptr` / `part == nullptr` reaching `base_pull` (secondary item 1) — I judge this NOT
   reachable; ISA check sufficient, no source guard needed.** From source: (a) `base_pull` is *referenced* only
   inside the `m8_pull` arms (`..._mps.hip:1453-1458`, `:1472-1485`) and `base_slot` only in the `else` arms, so
   both are called under a guard; (b) the guard value is not a compile-time constant — `m8cfg` comes from
   `k0p6_dread`, a `volatile` load (`:191-196`, `:1406-1408`) — and it *is* uniform across the whole grid (one
   host-written descriptor word, read identically by every thread), so it lowers to a scalar `s_cbranch`, never an
   exec-mask merge of both arms; (c) if-converting the pull arm would require speculating
   `descriptor->local_heap_base` / `heap_bases` (`moe_hk_adapter.cuh:50-54` → `peer.cuh:40-49`), and LLVM only
   speculates a load it can prove dereferenceable — this base is the result of an opaque volatile load, so
   `isSafeToSpeculativelyExecute` fails and the hoist is illegal; (d) the two `k0p6_mps_m8_batch` instantiations
   are `__forceinline__` templates (`..._mps.hip:346-352`), so they are inlined per call site and never merged
   into one predicated body. No extra `__builtin_expect`, `volatile` read or opaque call is needed: the volatile
   load already blocks speculation of the *condition* and the unknown base blocks speculation of the *load*.
   NEEDS ISA CHECK (confirmatory only): `llvm-objdump -d --mcpu=gfx950 --symbolize-operands <hsaco> > mps.s` then
   `rg -n 's_load_dwordx2.*0x1b8|s_cbranch|global_load_dwordx4|global_load_dwordx2' mps.s` — descriptor word 55
   sits at byte offset `55*8 = 0x1b8`; require that the load of that word, and every load whose base register is
   the value it returns, is dominated by an `s_cbranch_scc*` on the config word. If one is not, the minimal source
   fix is to make `symmetric`/`part` unconditional in M8 (`:1415-1417`) — one extra live SGPR pair in a phase that
   is not the MFMA register peak.

10. **`fetch_add_acq_rel(ptr, 0)` as the group-completeness probe** (`moe_mps_adapter.cuh:340-346`) carries the
    acquire edge for other waves' `part` payload, so it must survive as a real RMW. LLVM does not fold idempotent
    `atomicrmw` at `acq_rel`, so it should, but NEEDS ISA CHECK: `rg -c 'global_atomic_add' mps.s` — expect
    `1 + g` atomics per event in the service loop, not `1`.

11. **Config-validation gap.** `config_is_valid` never rejects `pull_fallback` for modes 0/1
    (`moe_mps_adapter.cuh:84-96`), so mode 1 + `pull_fallback=1` silently degenerates to mode 0's read path with
    wasted pushes — a mislabelled arm, not a fault. Relevant only where A3 crosses A4.

## Release/acquire audit

| site | scope | reference equivalent | verdict |
|---|---|---|---|
| M7 task-done → event queue (`..._mps.hip:275-281`, `adapter:148-155`): all-thread `vmcnt(0)` at `:312`, tid0 release, relaxed store | agent release / agent store | `release_cta_payload_system()` + grid barrier (`ref:955-964`) | **WEAKER** (agent vs system, no barrier) — risk 2 |
| service consume event (`adapter:295-302`): lane-0 relaxed poll → all-lane acquire | agent | `acquire_payload_agent()` after the grid barrier (`ref:965`) | WEAKER, but fence side correct (acquire strictly after the poll) |
| cross-wave group probe (`adapter:328-346`): acq_rel RMW on both sides | agent acq_rel | ABSENT (no analogue) | ADDED — transitive chain is valid, conditional on risk 10 |
| slice push → row flag (`adapter:246-263`): wave `vmcnt(0)` → lane0 system release → lane0 relaxed flag | system | per-CTA drain+release, grid barrier, leader release, leader stores (`ref:955-972`) | **WEAKER** — covers only the publishing wave's stores; risk 1 |
| M8 flag poll → payload (`..._mps.hip:382-396`): lanes 0-31 system poll, all-lane system acquire + `__syncwarp` | system | `ref:1043-1060`, byte-identical | MATCH |
| self-flag store vs consumer load scope (`adapter:259` agent store, `..._mps.hip:382` system load) | mixed | `ref:981` / `ref:1043`, identical mix | MATCH (validated empirically by the working reference) |
| mode 1 bulk push → flag (`..._mps.hip:1363-1386`): CTA drain+release, then leader flags, one CTA per row | system | same idiom as `ref:955-996` | MATCH (strictly sounder than mode 2) |
| mode 2 + `pull_fallback` remote `part` read (`..._mps.hip:1435-1439`) | — | `release_cta_payload_system()` (`ref:955`) | **ABSENT** — risk 3 |
| M0 MPS zeroing (`..._mps.hip:562-571`): plain store vs atomic consumers | none | `a2_done` plain reset (`:546-549`) | MATCH-in-class; risk 8 |
| M9 retire release + pokes (`..._mps.hip:1534-1544`) | agent release, agent/system stores | `ref:1137-1147`, byte-identical | MATCH |

## Epoch lifetime audit

| buffer | written | read | reset | N+1 writes while N reads? |
|---|---|---|---|---|
| `MPS_Q` (56) | M7 hook, one writer per slot, ticket-dense (`adapter:152-154`) | service waves (`adapter:295`) | M0 plain, grid-strided (`..._mps.hip:562`) | No — launches serialize; M3/M5 barriers order the zeroing before M6 |
| `MPS_NCARR` (57) | service acq_rel RMW (`adapter:330`) | same RMWs + probes (`:343`) | M0 (`:563`) | No (same argument) |
| `MPS_PUSHED` (58) | service `fetch_add` (`adapter:379`) | same | M0 (`:565`) | No |
| `MPS_CLAIM` (59) | `atomicOr` (`adapter:354`) | same | M0 (`:566`) | No |
| `STATE[ST_TAIL]` (60.1) | M7 hook `fetch_add` (`adapter:152`) | nothing on the correctness path (diagnostic + the risk-4 check) | M0 (`:568-571`) | No |
| `STATE[ST_M8NEXT]` (60.2) | M8 `fetch_add_relaxed<agent>`, one per wave (`..._mps.hip:1476,1490`) | same | **M0 of the next launch only** (`:568-571`) | **No, but only because launches serialize.** In-epoch, the M5 grid barrier (`:1111`) dominates every reader, so no stale in-epoch read exists, and a fast CTA of N+1 cannot exist while N runs (one epoch per launch, `:578-582`). If that ever breaks (two epochs in flight, graph capture, multiple streams) this ticket corrupts **first**: a stale value ≥ `nbatches` makes every wave break immediately and M8 silently writes nothing. |
| `STATE[ts]` (60.8..23) | `atomicMax` (`adapter:128-137`) | host dump | M0 zeroes exactly 24 u32 = the 96 B the host allocates (`moe_host_abi.hpp:161,186`) | No |
| `MPS_SLOTS` (61) | service/bulk peer packets (`adapter:233`, `..._mps.hip:1363`) | M8 `base_slot` (`:1440-1446`) | **never zeroed** (448 MiB); readiness rides epoch-tagged `row_ready` alone | No — reuse is gated by M0's `retired ≥ epoch-1` wait (`:522-528`), which a peer posts only after its `combine_done == 256`, i.e. after every reader finished M8 |
| `row_ready` (25) | service flush (`adapter:259-263`) / mode 0-1 leaders | M8 `poll_epoch_system` (`..._mps.hip:382`) | never; monotonic `observed >= expected` (`completion.cuh:57-63`) | No — a stale `N` is `< N+1`, so the poll correctly keeps waiting |
| `row_remaining` (26) | M2 per live row (`..._mps.hip:978`) | service `target` (`adapter:322`), mode 0/1 stripes | self-clean by the unique flag publisher (`adapter:267`) | No. The zeroing cannot strand a later reader: `pushed[r] == 16` implies all 16 chunk counters hit `target`, which implies every contributing wave already *loaded* `target` (its load precedes its bump in program order). Mode 2's self-clean is complete under success because `row_remaining[r] > 0 ⇒ r ∈ sti`. |

## Negative control

`if (__ballot(batch_ready) != ~0ull) return;` (`..._mps.hip:393`) is equivalent to the reference's `continue`
(`ref:1056`) on **all four** structural paths — in every case the only effect is "skip this token batch, let the
caller pick the next":

- mode 0 static pull (`..._mps.hip:1454-1458`) — caller is `for (tok0 …; tok0 += nw*NT)`; the `return` lands on the increment. EQUIVALENT.
- mode 1 static slot (`:1460-1464`) — identical loop shape. EQUIVALENT.
- mode 2 dynamic pull (`:1473-1485`) — caller is `while(true){ ticket; if (batch >= nbatches) break; … }`; the `return` lands on the next ticket. EQUIVALENT (and strictly better: the failed batch is not re-polled).
- mode 2 dynamic slot (`:1487-1500`) — identical. EQUIVALENT.

"A timeout must not mint a usable peer payload": **holds.** The `return` precedes `acquire_payload_system()`, every
payload load and every `out` store (`:393-444`), so a timed-out batch dereferences nothing and leaves `out` at its
prior-epoch contents — wrong numbers *and* `pperr != 0`, the intended signature. `batch_ready` stays `true` on
lanes 32-63 (no lane there satisfies `(lane>>3) == t` for `t < NT = 4`) and on out-of-fanout lanes, so `!= ~0ull`
means exactly "some polling lane timed out" (`:353-388`).

"M9 still retires the launch": **holds for the M8 paths.** Every early `return` after M0 is grid-uniform
(`:489,508,542,934,986,1047,1102,1112`), all four M8 paths fall through, and `payload_ok` only forces `T = 0`
(`:1422-1425`) so M8 is empty rather than skipped; mode-2 service CTAs that break out of `run_service` on `pperr`
(`adapter:299`) still reach M8 and M9. **Conditional on risk 5**: the `error_bit_set_agent` window can strand a
CTA on `s_barrier` before M9, after which `combine_done` never reaches 256, no `retired` poke is issued, and the
protocol instance wedges across launches. That is the one way the negative control wedges instead of failing, and
it is inherited from the parity port.

## Primitives

**Open-coded where the library already has the shape.**
- The row-completion fan-in is hand-rolled (`fetch_add_relaxed` + `oldp + g == 16` + a separate system release,
  `adapter:374-392`) instead of `counted_arrive_release_into<system>` (`counter.cuh:84-90`), whose own docstring
  (`counter.cuh:55-60`) describes risk 1 precisely: "This raw form does not export the effects newly acquired by
  the last RMW to another agent. Before a relaxed remote readiness store, either use
  `counted_arrive_release_into()` or issue `thread_release<system>()` after this call on the last arriver." The
  library warned; the kernel open-coded past the warning. The primitive still needs widening to fit: it counts
  `+1`, not `+g`, and it has no per-arriver `vmcnt` drain, which is the other half of the fix.
- The event-queue wait is open-coded (`adapter:188-202`) where `wait_tile_acquire_into` (`roles.cuh:146-153`) is
  the declared consumer of `publish_tile_release` (`adapter:154`). It genuinely did not fit: the kernel needs a
  `pperr`-watching escape inside the spin, and it needs "any nonzero MARK word", not "≥ expected". Both are worth
  adding — an abort predicate, and a `wait_tile_any_into`.
- `store_peer_packets` is called unchecked (`adapter:233`, `..._mps.hip:1363`) though `store_peer_packets_checked`
  exists (`packet.cuh:63-71`). Alignment is statically provable here (896·g and 448·2 offsets are 16-byte
  multiples), so this is defensible — but nothing records that it was checked by hand.

**Reached for and did not fit.**
- `finish_order_partition` (`roles.cuh:62-124`) was built for exactly this split and is now **dead code**: the
  shipped reservation is `bid >= nct - C` (`..._mps.hip:1156`, `:1193`) because the two ticket words cost
  +24 B/lane of spill at the phase-2 pointer peak (`DESIGN_MPS.md:119-125`). The library's role vocabulary has no
  register-free form, so the kernel expresses its role split in raw arithmetic. The honest surface is a static
  partition primitive (`static_tail_partition(bid, total, service_ctas)`) returning the same `role_partition`.
- `translate_peer` fails closed to `nullptr` for out-of-range ranks (`peer.cuh:38-49`) and **no** MPS call site
  checks it (`adapter:232`, `:262`, `..._mps.hip:1362`, `:1382`, `:1437`, `:1541`). Reachability is blocked today
  by `live = r < env.t_ext` and the `world == 8` shape guard, so this is a footgun, not a live bug. A fail-closed
  null with no `_checked` variant and no caller that checks is the wrong default: either trap, or ship
  `translate_peer_checked`.
- `bounded_wait_slot_reusable_into` / `drain_and_retire_slot` (`lifetime.cuh:22-59`) are the directed-credit
  primitives for address-reused peer slots, which is exactly what `MPS_SLOTS` is — yet slot lifetime rides the
  whole-epoch `retired` gate instead (`..._mps.hip:522-528`). Coarser but correct, and the right call for a
  448 MiB buffer with no per-slot credit cell. Finding: the credit primitives are shaped for fine-grained
  double-buffering and have no whole-epoch analogue; `retire_epoch` (`roles.cuh:160-164`) is the nearest and is
  also unused (M9 stores raw).
- `error_bit_set_agent` (`moe_hk_adapter.cuh:196-205`) is per-**wave** yet gates CTA-collective control flow
  (risk 5). There is no library primitive for "CTA-uniform observation of a device-scope error word". That is the
  missing primitive, and it is the one that can hang the node.

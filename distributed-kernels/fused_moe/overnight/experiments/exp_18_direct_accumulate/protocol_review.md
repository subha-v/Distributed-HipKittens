# exp_18 — protocol review of A11/M11 (direct remote bf16 accumulation), pre-build

Read-only; reuses `exp_01_nil_fault/protocol_review.md`'s release/acquire audit and epoch-lifetime table. Citations: `_mps.hip` =
`k0pf6gm_device_tile_mps.hip`, `adapter` = `moe_mps_adapter.cuh`, `p2` = `n2_phase2_gm_mps.cpp`, `ab.py` =
`.../k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py`, plus `/opt/rocm/include/hip/amd_detail/*` and `~/kimi_build/mori/**` — the last
three read over ssh; no GPU work, no edits.

## Verdict

**DO NOT BUILD YET — blocked on one measurement and one design decision, both cheap. Not a technique-level kill, and the two things
expected to block it do not.** Blocker 1 is hardware: nothing in the repo or toolchain establishes that `global_atomic_pk_add_bf16`
works on a *peer* VA over xGMI; the only API that emits it (`unsafeAtomicAdd`) takes no memory-scope argument and hardwires agent
scope, and HIP's own header documents fine-grained global memory as **undefined behaviour** for it — while mori's symmetric heap is
HIP-VMM allocated with `HeapType::Uncached`, whose grain the shipped source does not reveal. Run the 2-rank all-ones microbenchmark
below before writing kernel code; if it loses updates, A11 is closed on hardware. Blocker 2 is the real protocol hazard: `slots` is
**never zeroed per epoch**, which is sound today only because every write is a full overwrite, whereas accumulation demands a zero
buffer on the OWNER before a peer's first accumulate — and there is **no cross-rank ordering point anywhere between M0 and M7**. Pick
epoch-parity double-buffering (the in-tree exp_56 idiom) over end-of-epoch zeroing, which moves 448 MiB into the combine tail and eats
most of the prize. Two findings clear the path: **the harness bit-exactness gate does not apply to this change at all** (it compares
two standalone combine kernels before any region arm launches), and **bf16 non-associativity is not a new risk** (the epilogue already
performs an unordered multi-CTA bf16 atomic fan-in of the same degree). The one risk the configured ladder cannot see is *silent lost
updates* from a non-atomic remote RMW, so a build must ship its own detector.

## The epilogue today

`p2:119-131`, inside `epilogue_write<JMAX>` (`p2:84-134`):

```124:129:distributed-kernels/fused_moe/n2_phase2_gm_mps.cpp
      if (xtok[i] < T) {
        __hip_bfloat162* p = reinterpret_cast<__hip_bfloat162*>(
            OUT + static_cast<std::size_t>(xtok[i]) * kHidden + col_base +
            2 * dcol);
        unsafeAtomicAdd(p, *reinterpret_cast<const __hip_bfloat162*>(&d));
      }
```

`OUT` is bound to `K0P6_D_PART` — this rank's **local `part`** (`_mps.hip:1235`), a mori symmetric allocation (`ab.py:1483`).

Who writes what: `rowh = lane >> 5`, `dcol = lane & 31` (`p2:216-217`); `xtok[i] = tok_lds[rbase + 2*i + rowh]` (`p2:412-416`) is the
*receive-row* index (`tok_lds` = `sorted_ids & 0x00FFFFFF`, `p2:240`); `col_base = 448*nc + 112*wv`, `+64` for the second call
(`p2:417-427`). **Each thread issues exactly one 4-byte `global_atomic_pk_add_bf16` per `i`** — two adjacent bf16 columns at
`part + xtok[i]*7168 + col_base + 2*dcol`. Per wave, `JMAX=4` covers `col_base + [0,64)` (all 32 `dcol`) and `JMAX=3` covers
`col_base+64 + [0,48)` (`dcol < 24`); 64+48 = 112 = `kWaveCols2`, ×4 waves = 448 = `kChunkN` (`p2:69-70,76`). Per live sub-block that
is exactly 32 rows × 448 columns, each dword issued once.

**Guarded, not unconditional**: `if (xtok[i] < T)`, `T = nvi[1]` the padded extent (`p2:182`, `_mps.hip:1096`). `xtok[i]` depends only
on the unrolled `i` and on `rowh`, so it is uniform across each 32-lane half and a pad row's half-wave is EXEC-masked off, **forming no
address at all** (`p2:116-118`). Dead rows inside a *live* sub-block are additionally zeroed by value (`sw = lv2[m] ? sw2[m] : 0.0f`,
`p2:94`), so their atomic adds 0.

**Fan-in degree**: the same `(r, nc)` byte range is accumulated by `row_rem[r]` distinct tasks, where `row_rem[r] = popcount(live
experts of r)` written by M2 (`_mps.hip:1023`) and read by the service loop as `target` (`adapter:497-499`). So **`part` already
receives a multi-CTA, nondeterministically ordered bf16 atomic fan-in of degree `row_rem[r]`** — decisive for §*Numerics*.

**The existing drain is already CTA-wide**, load-bearing below: `N2GM_TASK_DONE_DRAIN_HOOK` expands to `s_waitcnt vmcnt(0)` on
**every thread** (`_mps.hip:334`, sited `p2:435-437`), then `__syncthreads()` (`p2:439`), then the task-done hook (`p2:443-450` →
`_mps.hip:294-311`). Documented at `adapter:221-229`.

## Remote atomic feasibility

`hk_moe::peer_ptr` → `translate_peer<8>` hands back a **raw peer VA** and nothing else: `peer_base + (local − local_allocation_base)`,
failing closed to `nullptr` for an inactive rank (`peer.cuh:40-49`, `moe_hk_adapter.cuh:49-62`). No handle, no aperture, no wrapper.

Two things are **proven to work today**. (1) *Remote integer RMW at system scope on the mori heap*: `_mps.hip:705-708` does
`reserve_rows_relaxed<system>` (= `__hip_atomic_fetch_add(…, RELAXED, SYSTEM)`, `sync.cuh:136-147` → `:54-65`) on
`peer_ptr(dest_counter + cur, dest, symmetric)`; all of M1 dispatch depends on it and every gate passes, so remote atomics *as a class*
work on a peer VA. (2) *The packed-bf16 hardware atomic on a mori symmetric allocation, local target*: exactly what the epilogue does
into `part` (`ab.py:1483` + `p2:128`), with `mps_mega` numerically identical to `pf6gm_mega` every run (`LESSONS.md:92-94`).

What is **NOT** established and cannot be from this repo — `NEEDS HARDWARE CHECK`:

- `unsafeAtomicAdd(__hip_bfloat162*)` lowers to `__builtin_amdgcn_flat_atomic_fadd_v2bf16` (`amd_hip_bf16.h:1887-1896`). **That
  builtin takes no memory-scope argument**, and the CAS fallback in the same header hardwires `__HIP_MEMORY_SCOPE_AGENT`
  (`:1906-1911`). An agent-scope (no `sc1`) atomic on a *peer* address is the wrong cache domain by construction — it may be
  performed at the local coherence point for an address the local L2 does not own.
- HIP's documented precondition: "*If `addr` is a global segment address, it is in a coarse grain allocation. Passing in global
  segment addresses in fine grain allocations will result in undefined behavior and is not supported.*"
  (`amd_hip_unsafe_atomics.h:50-54`.) mori's heap is HIP-VMM allocated (`mori/include/mori/application/memory/symmetric_memory.hpp:
  67-70,149`) with `HeapType::Uncached` by default (`.../application_device_types.hpp:85-88`; `mori/include/mori/shmem/internal.hpp:
  84-85`). **`Uncached` is not `coarse grain`**, and `ConfigureAllocationProp` exists only as a declaration in the shipped tree, so
  the grain is `UNKNOWN`.
- Whether Infinity Fabric implements a *packed-fp* RMW at the remote coherence point at all: `UNKNOWN`, unpublished, unmeasured here.

**One fact materially reduces the hardware ask:** each `slots[p][pos]` plane is written by exactly ONE producer rank
(`slice_group_dst` indexes `env.cur`, `adapter:318-328`; `base_slot` reads `p*MAXTOK + row − cur*MAXTOK`, `_mps.hip:1514-1520`), so
contending atomics all originate on the **same** rank — cross-rank atomicity is not required, only remote atomicity from one rank plus
visibility to the owner.

**Exact minimal test** (standalone, 2 ranks, no megakernel, seconds): rank 0 zeroes `N = 256` `__hip_bfloat162` cells in the existing
mori heap and barriers; rank 1 takes `p = translate_peer(cells, 0, …)` and each of 256 threads does `unsafeAtomicAdd(p + tid, {1.0f,
1.0f})` `K = 1024` times, then `s_waitcnt vmcnt(0)`, `__threadfence_system()`, sets a flag on rank 0; rank 0 polls and requires
`cells[tid] == {K, K}` **exactly** for all `tid`. Then repeat with two ranks hitting the same cells (not required by the design, but
it reveals which coherence point performed the RMW), and run `llvm-objdump -d --mcpu=gfx950 <hsaco> | rg 'atomic_pk_add_bf16'` —
require the packed instruction and **record its `sc` bits**. Read outcomes as: `== K` ⇒ mechanism exists; `1 ≤ value < K` ⇒
non-atomic or losing updates (the fine-grain UB signature) ⇒ **STOP**; `0` or a fault ⇒ the fabric drops or rejects it ⇒ **STOP**.
`K = 1024` is exactly representable in bf16, so a correct run has no rounding excuse.

## Completion detection

Today `nc_arr[r*16+nc]` counts to `target = row_rem[r]` (`adapter:511-514`), bumped by *service* CTAs consuming tile events
(`adapter:461-474`) — all AGENT scope, all in the **producer's** memory. The producer's service wave then drains, system-releases and
publishes `row_ready[cur][r] = epoch32` to the owner (`adapter:375-406`), which the owner polls with `poll_epoch_system`
(`_mps.hip:408-411`) followed by `acquire_payload_system` (`:422`) before any payload load.

| option | who counts, where | producer-side ordering required | verdict |
|---|---|---|---|
| **(a)** remote counter on the owner, each producer CTA bumps after its accumulate | owner's memory; fan-in becomes cross-rank | every contributing **wave**: `vmcnt(0)` → `thread_release<system>` → remote `fetch_add` at ≥release | **sound but wasteful.** Target becomes `Σ_p row_rem_p[r]`, which **no rank knows** (`row_rem` is per-rank local, `_mps.hip:1023`), and the counter itself becomes remote traffic — re-adding atomics exp_14 spent 306 µs deleting |
| **(b)** keep the local per-slice counter; last local arriver publishes `row_ready` | producer's memory, unchanged | last arriver holds an acquire edge for other contributors' **counter** updates, not their **payload** — which now lands in the *owner's* memory, coverable by no local `vmcnt` | **UNSOUND.** Risk #1 of `exp_01/protocol_review.md:27-39` promoted from latent to load-bearing: today a service CTA re-reads `part` and thereby re-orders the payload behind its own store; delete the push and that accidental barrier vanishes |
| **(c)** per-(producer,row) epoch flag, producer release + owner poll | owner's memory, one relaxed system store per (producer,row) | same drain+release per contributing wave; the flag storer must have observed every contributor | sound **only combined with a counter** — a flag needs a "last arriver", so this is (a) or (d) plus a flag |
| **(d)** keep the event queue, use it for counting only; service CTAs carry **no payload** | producer's memory, unchanged | already present: the per-task all-lane `vmcnt(0)` + `__syncthreads()` (`p2:435-439`) covers the whole CTA's atomics before the event store | **sound and cheapest** — a *scope* change only |

**Is per-wavefront `s_waitcnt vmcnt(0)` sufficient when a row's contributions come from different waves on different CTAs?** For a
wave's own stores yes, `vmcnt` being per-wavefront (`adapter:378-379`) — which is exactly why risk #1 exists at the *flush* site. But
at the *task-done* site the drain is already **all-lane**, so when `k0p6_mps_task_done` (`_mps.hip:294-311`) enqueues the event,
every lane of that CTA has retired its own atomics. Only the scope is missing: the publish is `publish_tile_release<agent>`
(`adapter:230-237`) and the consumer acquires with `thread_acquire<agent>` (`adapter:477`), while the payload must now reach a
**different agent**. Option (d) is therefore a two-word change, not a new protocol, and the chain is: producer's remote atomics
≺(all-lane `vmcnt(0)`, `__syncthreads`, tid0 `thread_release<system>`) event store ≺(`thread_acquire<system>`) service wave's
`nc_arr` RMW ≺(agent RMW order on that cell) last arriver ≺(`thread_release<system>`) `row_ready` store
≺(`acquire_payload_system`) owner's read.

**That chain's validity collapses into the hardware question.** A release fence orders *dirty local lines* (gfx950 system-scope
lowering is `buffer_wbl2 sc1`, `sync.cuh:160-164`); an RMW is not a dirty line. If the remote packed atomic is performed at the
**owner's** coherence point it is already globally visible when `vmcnt` retires and the release is redundant; if it is performed in
the **local** L2 no release can fix it, because the line sits in the wrong cache for the wrong address. So completion detection is
**not** the hard part; the hardware question is, and it must be settled first. Worth recording: option (d) also **deletes the
`pushed` counter** (~349,056 atomics/rank/epoch, `exp_14/result.md:43`) because there is no push to count — row completion fires when
all 16 `nc` of row `r` reach `target`. A11 is an atomic reduction as well as a payload reduction.

## Zeroing and epoch reuse

**Today.** `part` is zeroed *by the producing rank, locally, every epoch* at `_mps.hip:1139-1143` (`hkp::zero_part_scale_transpose
<14>`, grid-strided over the full `T_ext` extent on CTAs `bid >= 2`), sitting **between the M3 barrier (`:1091`) and the M4 barrier
(`:1146`)** with M5 (`:1156`) also intervening before M7 — two grid barriers of purely local ordering. `_mps.hip:1137-1138` is
explicit that holes must be zeroed too "else stale part rows corrupt combine". `slots` by contrast is **never zeroed per epoch**: the
host zeroes it once at allocation (`ab.py:1506-1509`, `_pf6_mps_slots.zero_()`) and per-epoch freshness rides entirely on the
epoch-tagged `row_ready` (`exp_01/protocol_review.md:163`) — sound *precisely because* every write is a full overwrite of a whole row.

**Accumulation destroys that invariant.** `slots[p][pos]` must be exactly zero on the owner before epoch N's first remote accumulate
lands, and it holds N−1's finished sum until someone clears it. Four placements, each broken in a specific way:

1. **Owner zeroes its own plane in M0 or M3 — WRONG, and this is the silent bug.** A producer's M7 of epoch N can begin before the
   owner reaches M3 of epoch N: there is **no cross-rank barrier anywhere between M0 and M7**. The only pre-combine cross-rank
   ordering in the kernel is M1's chunk-ready handshake consumed by M2 (`_mps.hip:851-854`), which orders *dispatch payload*, not slot
   lifetime. So the owner's zeroing can land **after** a fast peer's accumulate and erase it — scattered rows with output magnitude
   too low, `pperr == 0`, no fault. It needs a *rank skew*, not repetition, so the 600-epoch soak is the wrong instrument for it.
2. **Producer remotely zeroes the owner's plane first.** Reintroduces the peer write the change exists to delete, and needs its own
   protocol: the remote zero must be ordered against remote accumulates from *other* CTAs of the same rank — a per-(rank,row)
   "zeroed" flag plus a grid barrier at the M6/M7 boundary, the barrier mode 2 deliberately removed.
3. **Owner zeroes at end of epoch, after its M8 read, before its `retired` poke.** The only placement with an existing cross-rank
   ordering point — M0's `retired[p] >= epoch-1` wait (`_mps.hip:554-560`) already means no peer reuses an address until every peer
   retired the previous epoch. Costs: (a) the owner must zero its whole `slots` allocation, `world*MAXTOK*7168*2` B
   (`moe_host_abi.hpp:190-194`) = **448 MiB** (`exp_01/protocol_review.md:163`), i.e. *more* traffic than the ~312 MB saved
   (`exp_17/result.md:75-78`), merely relocated into the combine tail; (b) it must cover holes, per `_mps.hip:1137-1138`; (c) M8 is
   dynamic-ticket (`_mps.hip:1561-1573`) so no CTA knows which rows it read — the zeroing must be a separate grid-strided pass needing
   a **new local grid barrier between M8 and it**, serializing the one phase the role split already won (446 µs). M9's reset runs on
   one tid0 of one CTA (`_mps.hip:1585-1590`), so the pass must sit above `__syncthreads()` at `:1584` and finish before
   `combine_done` reaches 256.
4. **Epoch-parity double-buffer `slots[2][world][MAXTOK][7168]`** — the exp_56 fix-(5) idiom already load-bearing for `dest_counter`
   (`_mps.hip:662-667`), whose next parity is zeroed grid-wide in M0 (`:617-627`) with a full epoch of slack and **no new barrier**.
   Cleanest correctness story in the set; cost is the symmetric allocation doubling (448 → 896 MiB) plus an ABI sizing change.

**Two traps independent of who zeroes.** (i) Rows this epoch does not use must also be zero, or stale residue from an older epoch is
silently added to a live sum — `slots` tolerates arbitrary residue today and cannot under accumulation. (ii) There is no "same token,
same slot" argument to lean on: `pos` is handed out fresh each epoch by `dest_counter` (`_mps.hip:700-711`), so a given
`slots[p][pos]` holds a *different* token's partial sum each epoch.

## Numerics and the bit-exactness gate

**bf16 non-associativity does not get worse — it is already there at the same degree.** Today's epilogue is already an unordered
fan-in of `row_rem[r]` `global_atomic_pk_add_bf16` operations from `row_rem[r]` different CTAs (`p2:128`; degree from
`_mps.hip:1023`). A11 changes *where the accumulator cell lives*, not the number of addends, their values, the rounding mode, or the
fact that their order is unspecified; M8's arithmetic is untouched (`_mps.hip:425-470`). Corroborated empirically: `mps_mega` and
`pf6gm_mega` report **identical** `max_abs`/`relative` every run (`LESSONS.md:92-94`) despite both already running this fan-in, so the
jitter sits below the reported precision on this workload. The `rel_L1 ≤ 0.1 / max_abs ≤ 0.1` gates are not threatened by reordering.

**The bit-exactness gate does not apply to this change.** `combine_bit_exact` compares **two standalone host-launched combine
kernels** — `_combine_pull` then `_combine_pf` over the *same* `_n2` partials — counting `uint16` differences in `cand_out`
(`ab.py:3745-3756`). It runs in the k0pf primitive suite **before any region arm launches**; its own failure message says so
verbatim: `"k0pf combine bit-equivalence gate failed; no region arms were launched"` (`ab.py:3756`). It never touches
`k0pf6gm_mps_mega`. The only other bitwise comparison in the harness is `ab.py:3985` (`_k0d_combine_d`, grid `(16,256)`), another
standalone combine primitive. The megakernel arms are gated by `mok_gate` (`ab.py:3481-3500`), purely tolerance-based
(`absolute_tolerance` / `relative_tolerance`, metric `global sum(abs(diff)) / global sum(abs(reference))`) →
`mok_correctness_all_pass` (`ab.py:4625-4629`). `HARNESS_MAP.md:126-135` already lists the authoritative gates and correctly omits
`combine_bit_exact`. **The pre-registered worry that a bit-exactness gate blocks A11 is disconfirmed.**

**The residual numeric risk is a different one, and the ladder cannot see it.** If the remote packed atomic is silently non-atomic,
the failure mode is *lost updates*: output magnitude too low on scattered elements with `pperr == 0`. `relative_error` is a **global
L1 ratio**, so a small fraction of dropped addends passes it comfortably. A build must ship its own detector (step 6); do not read a
green `[MOK GATE]` as evidence the remote atomic worked.

## Failure and replay

Today a retry needs **nothing** to be true about `slots`. Its content is meaningless without an epoch-matching `row_ready`; every M8
read is gated by `poll_epoch_system(row_ready…, epoch32)` (`_mps.hip:408-411`) and a timeout `return`s before `acquire_payload_system`
and before any payload load (`:419-422`), so a failed launch never mints a usable payload (`exp_01/protocol_review.md:178-182`). The
launch still retires via M9's counted arrival (`_mps.hip:1585-1620`); `pperr != 0` is terminal and the host reinitializes rather than
clearing and retrying. Because every write is a full overwrite, residue is irrelevant.

Under accumulation a safe retry requires `slots` to be **provably zero on every owner** before the retried epoch's first accumulate —
a *cross-rank* precondition on 448 MiB of peer memory currently holding a partial sum from an aborted epoch of unknown progress, on
peers that may each have timed out at a different phase. Three consequences: (i) `row_ready` does not help — it is a readiness flag,
not a cleanliness flag, and its epoch tag makes stale flags harmless to *read* while saying nothing about payload residue; (ii)
epoch-parity double-buffering rescues steady state but **not** replay, since a retried epoch reuses the parity it aborted on; (iii) so
the reinit path must gain an explicit all-rank `slots` zeroing plus a barrier before relaunch, and that step is no longer optional.
The existing "no address reuse until every peer reached `retired >= epoch-1`" argument (`_mps.hip:554-560`) survives unchanged and is
what makes the added zeroing safe to perform.

**The negative control must be re-validated, not assumed.** `fn_combine_bug` (`ab.py:553`, run at `:4818-4823`) breaks a *combine*
kernel; A11 changes the *M7 epilogue*, which that control never exercises. A new control is required — e.g. suppress one rank's remote
accumulate for one row — and it must make `[MOK GATE]` fail. If it does not, the ladder is blind to this mechanism's failure mode and
no timing number from it should be believed.

## If built: the minimal safe design

0. **Microbenchmark first** (§*Remote atomic feasibility*), standalone, no megakernel. Gate: exact `K = 1024` readback on 256 peer
   cells plus disassembly showing `global_atomic_pk_add_bf16` and its `sc` bits. Any short readback closes A11 on hardware.
1. **Epoch-parity double-buffer `slots`.** Owner zeroes parity `(epoch+1)&1` grid-strided in M0, in the exact place and idiom
   `dest_counter`'s next parity is zeroed (`_mps.hip:617-627`): no new barrier, one epoch of slack, and M0's retire gate already
   guarantees all 256 CTAs finish M0 before epoch N+1 exists. Widen `mps_slots_bytes` (`moe_host_abi.hpp:190-194`); `ab.py:1509`'s
   one-shot `zero_()` still covers the first epoch. Prefer this to placement 3, which moves 448 MiB into the combine tail.
2. **Keep the service pool and event queue; delete only the payload.** `run_service` (`adapter:452-597`) keeps the ticket, event wait,
   `nc_arr` arrival RMW and flag flush; `push_slice_group*` (`adapter:333-369`) and the `pushed` counter (`adapter:415-428`) go away.
3. **Strengthen exactly two scopes, nothing else.** `publish_tile_release<agent>` → `<system>` (`adapter:230-237`) and
   `thread_acquire<agent>` → `<system>` (`adapter:477`). Leave the all-lane `vmcnt(0)` + `__syncthreads()` (`p2:435-439`) untouched —
   that drain is what makes the release cover the whole CTA's remote atomics. Downstream is already system scope (`adapter:384-398`,
   `_mps.hip:408-422`).
4. **Point the epilogue at the owner's slot.** `OUT` can no longer be one base: compute `owner = xtok[i]/MAXTOK`,
   `pos = xtok[i]%MAXTOK`, `dst = (owner==cur) ? local : peer_ptr(...)`. **Resource warning** — a division, a modulo and a peer
   translation inside the 16-iteration unrolled loop at the phase-2 pointer peak is precisely where `DESIGN_MPS.md:119-125` records
   that extra live values cost spill. `xtok[i]` is uniform per 32-lane half, so hoist the 16 bases into the same LDS pass that fills
   `xtok` (`p2:412-416`). Hold `SGPR 104 / VGPR 256 / AGPR 256 / scratch ≤ 60 B` and require zero scratch ops inside both MFMA
   K-loops before believing any timing.
5. **Ship it as a new mode 4**, so mode 2 stays the validated ratchet and the A/B is same-run and paired. `config_is_valid`
   (`adapter:118-139`) gains `mode <= 4`; nothing else in the config word changes, and mode 0 / `pull_fallback` remain as controls.
6. **Add the detector the ladder lacks.** Under a debug flag, accumulate BOTH remotely into `slots` and locally into `part`, then
   compare the owner's `slots[p][pos]` against `peer_ptr(part+row)` element-wise in a post-M8 pass, setting a new `pperr` bit on
   mismatch. Run once at the winning config before trusting a number; without it, silent lost updates pass `relative_error ≤ 0.1`.
7. **New negative control** per §*Failure and replay*, plus explicit all-rank `slots` zeroing in the host reinit path.
8. **Bump `K0P6_MPS_SRC_REV`** in the hashed `.hip` in the same commit as any `.cuh` edit and confirm a fresh `.hsaco` mtime.
9. Full gate ladder, then a paired 5-rotation campaign `production, pf6gm_mega, mps_mega(mode 2), mps_mega(mode 4)`.
   **Pre-registered falsifier:** mode 4 must beat mode 2's 6,866.1 µs; the modelled prize is the ~1,000 µs of M7 payload interference
   (`exp_16/result.md:53-60`), so anything above ~6,700 µs means the payload was not the limiter and exp_16's bound was wrong.

**Primitives gaps this exposes** (mandate note): `counter.cuh` has no *remote* arrive form — it counts `+1` at agent scope in local
memory (`counter.cuh:62-90`) and its own docstring (`:56-60`) warns about exactly the cross-agent export problem A11 creates without
offering the primitive. `packet.cuh` carries three *transport* forms but no **accumulating** transport, so a direct-accumulate
epilogue has nothing to reach for and would open-code the peer translation inline. The honest additions are
`counted_arrive_release_remote_into<system>` and an `accumulate_peer_packets_bf16` with a documented scope and allocation-grain
precondition — the latter unwritable until step 0 is measured.

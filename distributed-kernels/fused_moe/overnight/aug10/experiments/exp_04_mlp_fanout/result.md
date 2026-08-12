# exp_04 — memory-level parallelism in the peer push

**Verdict: POSITIVE but small — a 14–29% cut in service-pool cost, 3.8–25%
end-to-end on the MPS arm depending on configuration.** It does not move the
ratchet (the best point is still 1.43x `pf6gm_mega`), but it is a real effect,
and the number that matters most is the one it implies: **4x memory-level
parallelism bought only ~20% of the service cost, so the copy is a MINORITY of
what the service pool spends its time on.** Roughly 70–85% is bookkeeping.

> **This file supersedes an earlier version of itself that reported a NULL.**
> That measurement was taken on a stale kernel. The correction, and the guard
> that now prevents it, are documented below because the failure mode is
> reusable and cost a whole experiment.

## The stale-arm failure, and the guard

The mori JIT cache key is a content hash over its own `_jit-sources` tree,
restricted to `.hpp/.h/.cpp/.hip`. Our headers — `moe_mps_adapter.cuh` and
`include/cdna4/.../packet.cuh` — arrive through `-I` from the read-only
Distributed-HipKittens bind mount and are **not in that tree at all**. This
experiment's entire change lived in those two `.cuh` files, so the key never
moved and the campaign silently reused the previous
`k0pf6gm_mps_mega.hsaco`.

My original staleness check was wrong in an instructive way. I checked for JIT
cache **directories** with fresh mtimes and found two inside the screen window
(08:52:43, 08:53:10) and concluded the arm had rebuilt. Directory mtimes are
touched on a cache **hit**. The forensics that actually settle it:

| newest `k0pf6gm_mps_mega.hsaco` mtime | sha256 (16) | note |
|---|---|---|
| 08:18:00 | `48f25e1ceca46693` | exp_01 fix era |
| 08:32:10 | `c736707b9fc3bd4d` | |
| **08:40:42** | `2415211ad6c58597` | **newest before the screen** |
| — | — | *(MLP screen ran 08:52–08:55: no new object)* |
| 09:06:45 (`latest`) | `48f25e1ceca46693` | byte-identical to the 08:18 build |

No object was produced during the measurement window. **Only the `.hsaco`
mtime counts.**

**Guard installed:** `K0P6_MPS_SRC_REV` in `k0pf6gm_device_tile_mps.hip` (which
*is* hashed), with the rule written at its definition — bump it in the same
commit as any `.cuh`-only change. The screening driver now also `git reset
--hard`s the node checkout before every ladder and prints the newest hsaco mtime
before and after, so a silent cache hit is visible in the log. The re-run
carries a fresh hash `66040eba7d06`, which is what makes the numbers below
trustworthy.

## What was built

- `packet.cuh`: additive `store_peer_packets_multi<Regions>(destinations[],
  sources[], bytes, tid, threads)` — copies `Regions` equally-sized regions with
  **every load issued before any store**, each region keeping its identical
  fully-coalesced 1,024 B/wave pattern. `store_peer_packets` untouched, so all
  existing callers including `distributed-kernels/gemm_rs` stay bit-identical.
- `moe_mps_adapter.cuh`: `push_slice_group` split into `slice_group_src` /
  `slice_group_dst`; `push_slice_group_batch` pushes `kPushBatch = 4` claimed
  groups per call; `run_service` copies four groups per pass then retires them
  **one row at a time** via `retire_pushed_row`, leaving the flag/flush
  discipline (flush test after EVERY append, `flag_count <= flush_rows <= 64`)
  bit-for-bit unchanged.

Motivated directly by the exp_03 ISA read: the inlined copy loop is ten
instructions — one `flat_load_dwordx4`, an unconditional `s_waitcnt vmcnt(0)
lgkmcnt(0)`, one `flat_store_dwordx4` — **one load in flight per lane**, and at
`g=1` the body runs exactly once (56 packets over 64 lanes) so nothing can be
pipelined *within* a call. Batching across independent regions is the only way
to reach MLP > 1 at every `g`.

## Result — verified fresh build, screened against identical exp_03 configs

| config | exp_03 `mps_us` | exp_04 `mps_us` | end-to-end | service cost before → after | service delta |
|---|---:|---:|---:|---:|---:|
| C=8, g=2 | 57,292.1 | **42,923.1** | **−25.1%** | 50,206 → 35,837 | **−28.6%** |
| C=32, g=1 | 15,977.9 | **14,774.8** | −7.5% | 8,823 → 7,620 | −13.6% |
| C=64, g=1 | 10,374.1 | **9,976.3** | −3.8% | 2,953 → 2,536 | −14.1% |
| C=64, g=2 | 13,262.8 | **11,610.6** | −12.5% | 5,842 → 4,190 | **−28.3%** |
| C=64, g=16 | 27,591.8 | **22,181.7** | −19.6% | 20,171 → 14,761 | −26.8% |

**Compute-path control — the scratch regression is free.** Mode 0 reserves the
CTAs but never runs the service push, so it isolates whether the +68 B/lane
scratch hurt the MFMA path: `C=64, g=1, mode=0` measures **7,387.3 (1.063x
paired pf6gm)** against exp_03's **7,421.2 (1.061x)** — unchanged inside the
screening band. The extra scratch is reserved, not touched, outside the service
role, which is exactly what the budget rule ("every remaining spill byte must be
outside both MFMA K-loops") asks for.

(Service cost = measured minus the mode-0 capacity tax at the same `C`.) All
rows gate-green: `[MOK GATE] pass=True`, `control_fails=True`, `[MPS SOAK]
600/600 pperr=0`.

The gain is larger at `g=2` than `g=1`, which is the signature of the mechanism
working as designed: at `g=1` a region is 56 packets across 64 lanes so four
regions give four loads in flight, while at `g=2` each region is two packets per
lane so four regions give eight. More regions in flight, more of the round trip
hidden.

## The important consequence — the service pool is bookkeeping-bound

If the copy were the whole service cost, 4x MLP would have cut it ~75%. It cut
14–29%. Taking the `g=2` figure at face value, the copy is **at most ~19% of
service time** and the remaining **~81% is bookkeeping**: the per-slot
`wait_event_nonempty` poll, the `fetch_add_acq_rel` arrival on `nc_arr`, the
`g`-wide group-completion probe loop, the `atomicOr` claim, and the
`flush_pending` release.

That redirects the whole M-series. **The levers worth running on this path are
the ones that remove atomics and fences, not the ones that move more bytes:**

- **M4** (per-XCD L2 arrival counter, one cross-XCD release per XCD — AMD
  Research *Fleet*, measured on MI350X) is now the top candidate, and it
  generalizes A10.
- **M3** (>=256 KiB bands) drops further down, consistent with the independent
  citation audit that demoted it.
- **M1/M2** (cache bits on the payload store) act on the copy, i.e. on the ~19%.

## Resource cost — real, and confined to the service path

| build | SGPR | VGPR | AGPR | scratch B/lane | LDS B/block |
|---|---:|---:|---:|---:|---:|
| exp_01 baseline | 104 | 256 | 256 | **60** | **155,428** |
| `kPushBatch = 4` | 104 | 256 | 256 | **128** | 155,428 |
| `kPushBatch = 2` | 104 | 256 | 256 | 76 | **163,632** |

`packet16 staged[4]` is 64 B/lane and scratch grew by 68 B: the staging array
**spilled to scratch rather than registers**, because the kernel is pinned at
VGPR 256 / AGPR 256 by the MFMA path. That is exactly why the gain is 14–29%
instead of ~75% — every batched load is stored and reloaded before its peer
store, and the spill traffic eats most of the latency it hid.

`kPushBatch = 2` is the control: half the staging, +16 B instead of +68 B. It
also exposes a second failure — the compiler promotes the smaller alloca to
**LDS**, taking LDS to 163,632 B and breaking the byte-exact 155,428 B parity
gate. So `Batch = 2` is inadmissible on LDS and `Batch = 4` costs scratch.

**Standing constraint, worth stating once and reusing:** a service CTA is
allocated the same 256 ArchVGPR + 256 AGPR as a compute CTA of the same kernel,
because register allocation is static and per-kernel. A service CTA never
issues MFMA, cannot use what the MFMA path reserved, and cannot obtain one extra
register of its own. CTA-level role specialization does **not** sidestep the
HipKittens wave-specialization register problem that CLAUDE.md cites — it
relocates it. The split does avoid MFMA issue-slot theft (A7, already struck),
but it inherits the register tax in full, and the service role is register-poor
in precisely the phase where registers would buy memory-level parallelism.

## Primitives

- **`store_peer_packets_multi<Regions>` added, and now has a measured
  justification.** It does what it claims and the effect is visible end to end.
  Its benefit is *register-financed*, and the library surface gives a caller no
  way to discover that it needs `16 * Regions` bytes/lane of headroom the caller
  may not have. Surface suggestion stands: primitives whose benefit is paid for
  in registers should say so in their contract.
- **`store_peer_packets`'s `#pragma unroll 1` is correct for its documented
  purpose and wrong as a default for global→peer copies.** Its own comment
  explains the unroll (dynamic indexing would demote a register `packet16[N]` to
  scratch); the cost is MLP=1. Both facts should live in the doc comment so the
  next caller does not have to read the ISA to find out.
- **`counter.cuh` has no group-completion vocabulary**, which is why the adapter
  open-codes `fetch_add_acq_rel(p, 0u)` as an acquire-carrying read across `g`
  members. Given that bookkeeping is now measured at ~81% of service cost, this
  is the highest-value primitive gap in the library right now.

## Disposition — KEPT

`kPushBatch = 4` stays in. It is a 14–29% service-cost improvement, it costs
+68 B/lane of scratch that the mode-0 control proves is free to the compute
path, and it holds ArchVGPR/AGPR parity (256/256), SGPR 104, LDS 155,428 B
byte-exact, occupancy 1. `kPushBatch = 2` is rejected — it breaks the LDS parity
gate for a smaller gain.

It does **not** move the ratchet: the best MPS point is `C=64, g=1` at 9,976 us
screened = **1.433x `pf6gm_mega`**, so `pf6gm_mega` at 0.894x production remains
the best candidate. This is banked as an improvement to the MPS arm and as the
measurement that redirected the M-series, not as a ratchet event.

## Artifacts

`~/overnight-scratch/screen_E04real.csv`, `~/overnight-scratch/screen_E04ctl.csv`,
JIT hash `66040eba7d06`, commit `2290f495`.

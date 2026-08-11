# exp_04 — memory-level parallelism in the peer push: a NULL, for a structural reason

**Verdict: NEGATIVE, and the reason matters more than the result.** Giving the
peer copy 4-deep memory-level parallelism produced no measurable change
(±6%, no systematic direction) because **the staging registers do not exist**:
the kernel is pinned at VGPR 256 / AGPR 256 by the MFMA path, so the compiler
spilled the staging array to scratch (+68 B/lane) and the added parallelism
paid for itself in spill traffic. Reverted. The primitive is kept, unused and
documented, for a future caller with register headroom.

## Hypothesis (from the exp_03 ISA read)

`isa_push_loop.md` established that the inlined `store_peer_packets` loop is
ten instructions — one `flat_load_dwordx4`, an unconditional
`s_waitcnt vmcnt(0) lgkmcnt(0)`, one `flat_store_dwordx4` — with `#pragma
unroll 1` honoured and no software pipelining. **One load in flight per lane**,
1,024 B per 64-lane wave. At `g=1` the body executes exactly once (56 packets
over 64 lanes), so there is nothing to pipeline *within* a call.

The ISA report's own recommendation (b) was to batch across *independent*
(dst, src) streams — the several claimed row groups a service wave already
holds — which reaches MLP > 1 at every `g` including `g=1`. That is what was
built.

## What was built

- `include/cdna4/ops/group/distributed/packet.cuh`: additive
  `store_peer_packets_multi<Regions>(destinations[], sources[], bytes, tid,
  threads)`. Copies `Regions` equally-sized regions, **every load issued before
  any store**. Each region keeps the identical fully-coalesced 1,024 B/wave
  access pattern. Pointer arrays and staging buffer indexed only under
  `#pragma unroll`. `store_peer_packets` untouched — every existing caller,
  including `distributed-kernels/gemm_rs`, is bit-identical.
- `moe_mps_adapter.cuh`: `push_slice_group` split into `slice_group_src` /
  `slice_group_dst`; new `push_slice_group_batch` pushing `kPushBatch` groups
  per call; `run_service` copies `kPushBatch` groups per pass then retires them
  **one row at a time** through a new `retire_pushed_row`, so the flag/flush
  discipline (flush test after EVERY append, `flag_count <= flush_rows <= 64`)
  is bit-for-bit the old one.

## Result — screened against the identical prior configs

Same-run paired `pf6gm_mega` in every row. Baseline column is exp_03's screen
of the exact same config.

| config | baseline `mps_us` | MLP=4 `mps_us` | delta |
|---|---:|---:|---:|
| C=64, g=1 | 10,374.1 | 10,545.3 | +1.6% |
| C=32, g=1 | 15,977.9 | 15,725.8 | −1.6% |
| C=16, g=1 | 25,810.8 | 27,271.9 | +5.7% |
| C=64, g=2 | 13,262.8 | 12,886.2 | −2.8% |
| C=64, g=16 | 27,591.8 | 26,983.7 | −2.2% |

No systematic direction; every delta is inside the screening band. **Null.**

**Staleness ruled out.** New mori JIT cache directories appeared at 08:52:43
and 08:53:10, inside the screen window (the pre-change build was `2f9d36753135`
at 08:40:42), and the read-only container mount was confirmed to contain
`store_peer_packets_multi` and `push_slice_group_batch`. The measurement is of
the new kernel.

## Why it is null — the register budget, measured

| build | SGPR | VGPR | AGPR | scratch B/lane | LDS B/block |
|---|---:|---:|---:|---:|---:|
| exp_01 (single-region) | 104 | 256 | 256 | **60** | **155,428** |
| `kPushBatch = 4` | 104 | 256 | 256 | **128** | 155,428 |
| `kPushBatch = 2` | 104 | 256 | 256 | **76** | **163,632** |
| after revert | 104 | 256 | 256 | **60** | 155,428 |

`packet16 staged[4]` is 64 B/lane. Scratch grew by **68 B**. The staging array
went to scratch, not registers — so every "batched" load was stored to scratch
and reloaded before its peer store, which is strictly worse than the serialized
form it replaced, and roughly cancels the latency it hid.

`kPushBatch = 2` is the control that confirms it: halving the staging halves the
spill (+16 B instead of +68 B). It also exposes a second failure — the compiler
promoted the smaller alloca to **LDS**, taking LDS to 163,632 B and **breaking
the byte-exact 155,428 B parity gate**. Both batch factors are therefore
inadmissible, independently of performance.

## The finding — CTA role specialization inherits the kernel's register allocation

This is the CTA-level analogue of the intra-CTA result CLAUDE.md already cites
against wave specialization (HipKittens/MLSys 2026: NVIDIA-style wave
specialization reaches only 80% of peak BF16 GEMM on MI355X because static
register allocation makes producer waves consume registers without computing).

> **On CDNA4 a service CTA is allocated the same 256 ArchVGPRs + 256 AGPRs as a
> compute CTA of the same kernel, because register allocation is static and
> per-kernel. A service CTA never issues MFMA, so it needs almost none of that
> — but it also cannot *use* what the MFMA path reserved, and it cannot get one
> extra register for its own purposes. The service role is permanently
> register-poor in exactly the phase where it needs registers to buy
> memory-level parallelism.**

CLAUDE.md's mandate argued the CTA-level split "sidesteps" the HipKittens
problem. **It does not sidestep it; it relocates it.** Wave specialization
wastes registers on non-computing waves *inside* a CTA; CTA specialization
starves non-computing *CTAs* of registers they could otherwise have had. The
grid-level split does avoid MFMA issue-slot theft (A7, already struck), but the
register-allocation tax is inherited unchanged. This should be treated as a
first-class constraint on every future role-split candidate, not a detail.

Escape routes, none free:
1. **Stage through LDS instead of registers.** CDNA4 has direct global→LDS
   (`DOCUMENTED` in CLAUDE.md's struck-list entry, which notes the absence of a
   descriptor engine but the presence of the load). Outstanding global→LDS loads
   cost no VGPRs. But LDS is at 155,428 of 163,840 B — **8,412 B free**, enough
   for ~8 KB of staging, and any use collides with the byte-exact LDS gate.
2. **A separate kernel for the service role.** Struck in CLAUDE.md
   (disjoint-SM multi-kernel with stream priority) and it forfeits the
   persistent-megakernel property.
3. **Reduce the MFMA path's register demand** so the whole-kernel allocation
   drops. Out of scope tonight and would perturb the parity tuple.

## Primitives

- `store_peer_packets_multi<Regions>` **added and kept** in `packet.cuh`. It is
  correct, it does what it says, and it is the right shape for a caller with
  many small regions. It has **no caller in this repo** right now, and the
  reason is documented here so the next person does not re-derive it: it needs
  `16 * Regions` bytes/lane of register headroom that this kernel does not have.
  A primitive can be right and still be unusable — that is a property of the
  *kernel's* register budget, not of the primitive, and the library surface
  gives a caller no way to discover it. **Surface suggestion:** primitives whose
  benefit is register-financed should say so in their contract, and a
  `static_assert`-able notion of "register budget available to this role" would
  be the honest fix. We do not have one.
- The refactor into `slice_group_src` / `slice_group_dst` / `retire_pushed_row`
  was clean and semantically identical, but was reverted with the rest to keep
  the resource tuple byte-exact. Recorded because it is the right decomposition
  if this path is ever revisited.

## Disposition

Reverted at commit `f9068e33`; `moe_mps_adapter.cuh` restored to its exp_01
content and the tuple re-verified at SGPR 104 / VGPR 256 / AGPR 256 /
scratch 60 B / LDS 155,428 B. `packet.cuh` keeps the additive overload.

# exp_34 — mode 14 (coarse readiness): CPU gate green, awaiting GPU lease

**Verdict: NO RESULT YET. Nothing was run.** The GPU lease was held by another
agent for the whole of this build, so this experiment produced code and
compile-time evidence only. Do not quote a number from this file; there is none.

## Status

| item | state |
|---|---|
| code | **complete and compiling**, 0 errors, in the local repo, **uncommitted** |
| resource gate | **GREEN on every named item** — see `build_log.md` |
| protocol review | **APPROVE-WITH-CONDITIONS**, and all five conditions are now cleared on CPU — see §"Review conditions" below |
| drain-deletion confound | **fixed**: the deletion is now selectable by config bit, so the waterfall gets three rungs instead of one confounded one |
| protocol negative control | **built** (`negative_control.patch`, `NC.hsaco`), awaiting the GPU ladder |
| correctness / 600-epoch soak | **owed, needs GPU** |
| campaign | **owed, needs GPU** |

## Review conditions — all five cleared on CPU

### 1. `row_ready` allocation — PASSES, with 32 KB of headroom

The array is allocated once, on the symmetric heap, at
`prefill_opt/host/e004pf_k0pf_ab.py:1503`:

```python
_pf6_row_ready, _pf6_row_readyp = mori_t((WORLD, T_LOC_MAX), "int32")
```

so its size is exactly `WORLD * T_LOC_MAX` words with a row stride of
`T_LOC_MAX` — the same stride the kernel indexes with (`cur * T_loc_max + r`).
The campaign pins the shape at `run_campaign.sh:122-126`
(`K0_T=4096`, `K0_T_LOC_MAX=40960`, `K0_MAXTOK=4096`; descriptor slot 51 is
`int(T)` at host `:2779`), so:

| quantity | value |
|---|---:|
| allocation | 8 × 40,960 = **327,680 words** (1,310,720 B) |
| last legal index | 327,679 |
| `T_ext = world * MAXTOK` | 32,768 |
| highest index mode 14 writes, `(world-1)*T_loc_max + T_ext` | **319,488** |
| headroom | 8,192 words = **32 KB** |

Two structural facts matter beyond the arithmetic. First, `T_ext < T_loc_max`,
so the M7-done word lives in the **spare tail of its own segment** and cannot
alias another source's receive rows — it is not a write into segment `cur+1`.
Second, the entry guard rejects the config (`K0P6_MPS_ERR_CONFIG`) when
`world * MAXTOK >= T_loc_max`, so a future shape that removes the headroom
fails closed instead of overflowing by 4 bytes. **No memory-safety defect.**

### 2. No concurrent clear — PASSES

`row_ready` is zeroed exactly once, at setup, by the symmetric-tensor loop at
host `:1571-1573` (the array is in that tuple at `:1568`), and the allocation
comment at `:1502` already states the design intent: *"Epochs are monotonic;
this array never needs clearing."* There is no memset / `zero_()` / fill of it
anywhere in the soak or epoch loop.

The one other clear in the file is `_dc_reset()` (host `:5240-5245`), which does
zero `row_ready`. It is not a hazard: it belongs to the `K0_PF6GM_DECOMP` phase
decomposition path, which is gated on `K0_PF6GM_DECOMP=1` **and** the
`pf6gm_mega` arm and is never set anywhere under `benchmarks/`; and every reset
is bracketed by `torch.cuda.synchronize()` + `dist.barrier()` on all ranks
(`:5261-5265`, `:5270-5275`, `:5291-5292`), i.e. strictly between quiesced
launches. That is the "clear between launches is fine" case, not the
`dest_counter` defect class.

### 3. The system-scope invalidate is present and precedes the payload load — PASSES

Two independent views, both from the gated tree.

**Differential count** (`llvm-objdump -d --mcpu=gfx950` on the code object
unbundled with `clang-offload-bundler`): the arm has **8** `buffer_inv sc0 sc1`
(42 `buffer_inv` total); the identical tree with the coarse M8 branch compiled
out (`else if (false && m8_coarse)`) has **7** (41 total). Exactly one
system-scope invalidate belongs to the mode-14 `m8_batch` instantiation.

**Positive attribution**, from an audit-only build that brackets the
`Ready == true` acquire with `; E34_M14_ACQ_*` inline-asm markers (that build is
not an arm; its resource tuple is identical). The window, verbatim:

```
	;;#ASMSTART
	; E34_M14_ACQ_BEGIN
	;;#ASMEND
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	;;#ASMSTART
	; E34_M14_ACQ_END
	;;#ASMEND
	; wave barrier
	s_branch .LBB0_3911
```

The invalidate sits between the markers — it was not sunk or hoisted out of the
acquire — and **zero** vector memory ops appear between it and the loop header
`.LBB0_3911` it branches to. The first payload load is
`flat_load_dwordx4 v[58:61], v[2:3]` in the child loop `.LBB0_3913`. So the
order is `vmcnt(0)` → `buffer_inv sc0 sc1` → `__syncwarp` → first `slots` load.

### 4. Protocol negative control — BUILT, not run

`negative_control.patch` (in this folder) is the whole difference: the mode-14
publish loop becomes `for (int R = 0; R < world - 1; ++R)`, plus a revision bump
to 1028 so it can never collide with the arm's cache key. It compiles with 0
errors at the arm's exact resource tuple; `NC.hsaco` sha256
`af681cce22fe98ea0e8aaf9a974d52ce2e5742da431e9c876e30f9e1fe62cb2f`.

**The required outcome is asymmetric, and the runner must expect that.** With
`world - 1`, nobody publishes into rank 7's segment — including rank 7 itself,
because `R == cur` is never reached for `cur = 7` — while ranks 0–6 still
receive all eight producers' cells. So the demanded failure is: `pperr` bit 25
(`K0P6_MPS_ERR_M7DONE` = 33,554,432) set **on rank 7 only**, M8 skipped there by
the `payload_ok` mask, poisoned output, `[MOK GATE]` fail. A clean `pperr` on
every rank means the rendezvous is not load-bearing and the rung is meaningless.

*(First cut of this control was wrong and was rebuilt: the anchor
`for (int R = 0; R < world; ++R) { if (R == cur) {` matches the **parity**
per-row publish loop as well, and it patched that one — a control that breaks
modes 0/1/5/6 and leaves mode 14 intact. The delivered patch is anchored on the
`self_slot` publish, which is mode-14-only. Worth remembering: this file has two
textually identical publish loops.)*

### 5. Bit 26 visibility — CONFIRMED in the parsed log

`pperr` is printed as a raw integer, so bit 26 is readable without extra
instrumentation: `[MPS SOAK] completed=… pperr=<int> poison=<int> …`
(host `:5066-5073`) for the soak, and `output_gate["pperr"] = int(pperr.item())`
(host `:5193`) for the timed arms, where `local_pass` already requires
`pperr == 0`. The assertion to record per run is
`pperr & 67108864 == 0` (`K0P6_MPS_ERR_SERVICE`) — the positive check that the
drain and the service path were skipped. Because the gate demands `pperr == 0`
outright, a set bit 26 also fails the run; the explicit decode is what
distinguishes "the drain came back" from every other failure.

## The drain-deletion confound — FIXED, and the waterfall gets three rungs

The reviewer is right that mode 14 as delivered changes two things against
mode 12: signal granularity, and the per-task VMEM drain (~2,840 `vmcnt(0)` +
2,840 `__syncthreads()` per CTA). The drain deletion is now **independently
selectable in the same binary** by one config bit — `g |= 0x80`
(`kCoarseKeepDrainBit`, adapter `:290`; device side `KRN:419-431`), legal only on
mode 14 and rejected on 12/13 so a mistyped arm cannot masquerade as the control:

| rung | config | mechanism |
|---|---|---|
| (b) mode 12 | `g=353,mode=12` | per-row protocol + event publication + deferred drain |
| (c) **mode 14 + drain** | `g=481,mode=14` | coarse readiness, **drain retained** → prices granularity alone |
| (d) mode 14 | `g=353,mode=14` | + drain deletion → prices the drain alone against (c) |

Implementation: with the bit set, `k0p6_mps_task_done_maybe_defer` buffers the
task identity with `gcount == 0`, so `k0p6_mps_task_flush_defer` still pays its
`vmcnt(0)` + `__syncthreads()` at the next task head — exactly where mode 12
pays it — and its publication loop runs zero iterations. The bit is read from
the config word the hook already loads, so it costs no extra descriptor read,
and the resource tuple is byte-identical to the pre-selector build.

**One part genuinely cannot be separated, and must be labelled, not
attributed.** The per-task *event publication* cannot be retained under coarse
readiness: the event queue has no consumer once the per-row protocol is gone, and
publishing into it would both be dead work and make `K0P6_MPS_ERR_SERVICE`
ambiguous — which is the very assertion condition 5 exists to check. So rung (c)
prices "coarse readiness **including** the deletion of the event publication it
makes meaningless", and the figure must say that; only the drain is split out.
Nothing in the mode-12 → mode-14 delta may be attributed to granularity alone.

## Watch list (reviewer's non-blocking findings — recorded for the next agent)

1. **Stale `row_remaining` holes are a landmine for the nc-major reorder.**
   Mode 14 skips the parity self-clean at `KRN:1800` and M2 only writes live
   rows, so hole entries retain values from earlier epochs. Inert today —
   nothing in mode 14 reads `row_rem` — but **the nc-major task-reorder rung
   reads `row_rem` as its push target**, and would read those stale values. Fix
   the clean-up before, not after, building that rung.
2. **Bit 25's double meaning is safe only while mode 14 compiles the per-row
   poll out.** If a future mode-14 variant re-enables a per-row poll, "row_ready
   poll timeout" and "M7-done rendezvous timeout" stop being mutually exclusive.
   Collision symptom: bit 25 set with a non-empty M8.
3. **Split-brain publish (pre-existing, inherited from mode 0).** If bid0/tid0
   reads `payload_ok` true before another CTA's bit-21 `atomicOr` lands, this
   rank publishes despite a failed grid barrier. Symptom: bit 21 on one rank,
   clean `pperr` on its peers, correctness gate failing. Report that shape as a
   rank-asymmetric failure, not as a mode-14 protocol bug.

## The resource gate, in one line

`SGPR 106 / VGPR 256 / AGPR 256 / scratch 128 B per lane / LDS 155,496 /
occupancy 1 / MFMA 180 (96+84) / flat_atomic_pk_add_bf16 282 / zero scratch ops
inside either MFMA span` — **identical to the ratchet on every gated field**,
with mode 14 present in the build.

**One caveat that is not a gate item but must be read before this is timed.** The
build re-triggers exp_26's scratch migration into the remote-atomic epilogue:
**96 of the 282 `flat_atomic_pk_add_bf16` acquire a scratch op within 40
instructions ahead, against 0 in the base.** Seven different arrangements of the
same mechanism produced byte-identically the same allocator state, so this is a
property of total pressure in this function, not of where the mode-14 code sits,
and it is not fixable inside this change's ownership (the fix is STATUS's standing
"relieve one live VGPR in phase 2's epilogue"). exp_26 measured this exact
migration, bundled into its mask 1, at **+2.81 µs, t = 0.61 — a null**, so the
risk is bounded and small against the mechanism's ±245…695 µs. It is nonetheless
a debit to price against any mode-14 number, and it means **mode 14 vs mode 12 is
not a perfectly clean single-variable comparison at the ISA level.**

## Pre-registration, recorded before any run

- Band **5,990–6,440 µs**, point estimate **~6,215 µs**. Ratchet: `mps_mega`
  `C=16 g=353 mode=12 flush_rows=16` = **6,482.7 µs = 0.8408×** production.
- **A result above 6,568 µs falsifies the granularity rung** and is to be
  reported as such, not re-interpreted.
- Read the estimate pessimistically. exp_27's two terms came in **43 %** and
  **83 %** below their point estimates, and this family's predictions have been
  systematically optimistic.
- **The number is BANKED, not RATCHETED.** With the bookkeeping, the push and the
  flags all deleted the service pool has no job left, so at `C = 0` mode 14 is a
  *homogeneous* megakernel and cannot be a role-split ratchet however fast it is.
  **The degeneration is the finding**: it puts a price on what the readiness
  protocol costs, which is the denominator every future role-split needs.
- **Void, not a result**: any number from a build without `K0_MOK_POISON_OUT=1`,
  or with `K0P6_MPS_ERR_SERVICE` (bit 26) set, or from a run whose log lacks a
  `[MARK]` line or eight rank JSONs.

## First config to run

```
K0_MPS_CFG='C=0,g=353,mode=14,flush_rows=16'   # coarse readiness, drain deleted
K0_MPS_CFG='C=0,g=481,mode=14,flush_rows=16'   # + 0x80: drain retained (rung c)
```
`g = 353` carries the ratchet's throttle-enable + depth-4 + skip-dead-part-zero,
so the only variable against the mode-12 ratchet is the readiness protocol.
`g = 481` is `353 | 0x80`, the drain-retention control arm; run both, in that
order, or the rung is confounded.
`flush_rows` is unused by mode 14 but must still be present and in `1..64` — a
partial `K0_MPS_CFG` looks like a pass. `C = 0` is the homogeneous arm; `C = 8`
and `C = 16` are legal and reserve an idle tail (the exp_37 placement arm).

Positive path checks to assert on the first run, beyond `pperr == 0`:
**bit 26 must never be set** (proves the drain was skipped) and the `.hsaco`
mtime must be fresh (`stat -L`; `K0P6_MPS_SRC_REV` is now **28**, was 26).

## What lands in LESSONS.md regardless of the eventual number

1. **You cannot add a phase to this megakernel for free.** The function sits on
   the 256-VGPR ceiling, and the marginal spill victim is the phase-2 epilogue's
   `peer_tab` base load. A separate M7.7 phase cost +16 B/lane of scratch purely
   by duplicating the existing release/barrier/acquire/`payload_ok` sequence — not
   through the IRIS descriptor, not through `peer_ptr`, and not through the extra
   M8 template instantiation, all three of which were probed and cleared.
2. **Register-allocation attribution on this kernel is non-decomposable.** The
   per-task hooks alone and the M7.5 changes alone each score 186/282 on the
   migration metric; together with everything else the score is 96/282; removing
   the hooks from the full patch changes nothing. Bisecting a spill regression by
   deleting pieces will mislead you. Report the cliff, not a per-piece cost.
3. **A `--genco` `.hsaco` is an offload bundle: `llvm-objdump -d` on it returns a
   one-line error, which an ISA census script will happily count as `v_mfma=0`.**
   One earlier gate run in this experiment reported a clean MFMA span census off
   exactly that error before it was caught. The fix is one step, not a different
   tool: `clang-offload-bundler --unbundle --type=o
   --targets=hipv4-amdgcn-amd-amdhsa--gfx950` first, then objdump the code
   object — that path reproduces the `-S` census exactly (MFMA 180, pk_add 282)
   and is what condition 3's differential count above is built on.
4. **Attributing one instruction inside a 34,000-instruction inlined megakernel
   needs a marker or a differential, not reading.** Two cheap techniques, both
   used here: `if constexpr` + an inline-asm comment brackets a template
   instantiation's code in `-S`, and compiling the branch out (`false &&`) turns
   "which invalidate is mine" into a count that differs by one.
5. **This file has two textually identical `for (int R = 0; R < world; ++R) {
   if (R == cur) {` publish loops** — the parity per-row one and mode 14's
   rendezvous. A patch anchored on that text alone will silently hit the wrong
   one; the first cut of the negative control did exactly that. Anchor on the
   body (`self_slot`), and diff the control before trusting it.
4. **A design that says "`is_service_cta` must be false for this mode" needs the
   task stride checked in the same breath.** `k0p6_mps_stride` is `nct − C`
   unconditionally, so forcing the predicate false while `C > 0` would have made
   the tail CTAs silently re-execute M7 tasks — exp_25's named silent failure.
   Leaving the predicate alone made `C` a sound knob instead of a corrupt one.
5. **Gating a barrier on `payload_ok` is a hang.** It is a relaxed read of a
   concurrently-written `pperr`, so two waves of one CTA can disagree and one
   skips the `__syncthreads()` the other is waiting on. Barriers here may only be
   gated on grid-uniform descriptor decodes. An early revision of this patch had
   the bug; it is fixed in the delivered code and written up in `plan.md` §3.5.

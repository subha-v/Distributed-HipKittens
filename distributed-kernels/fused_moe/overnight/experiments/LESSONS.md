# Overnight lessons ledger (append-only; supersede, never delete)

- 2026-08-11 (baseline, pre-overnight): `mps_mega` builds clean on gfx950 with
  `SGPR 104 / VGPR 256 / AGPR 256 / LDS 155,428 B (byte-exact with the parity
  port) / scratch 60 B/lane (+24, cold paths, all ≥511 insns from any MFMA span)
  / static MFMA census identical to parity (96+84=180)`. First-launch mode-2
  faults deterministically with `address (nil)` on all 8 GPUs; debug-stop
  bisect clears M0 through M7(+enqueue hooks). Fault domain: M7.6/M8/M9 tail.
  Suspect ranking in `../MPS_OVERNIGHT_HANDOFF.md`; harness state in amd-master
  under `benchmarks/mok_synthetic_prefill/MPS_OVERNIGHT_HARNESS_NOTE.md`.
  LDS, MFMA placement, and arithmetic are NOT suspects: treat them as proven.
- Prior resource-gate sequence (banked): `s_batch_no[4]` and `ordinal_shared`
  were the +20 B LDS (reused dead M2 `s_ns`); the +44 B scratch was 11
  long-lived uniform values crossing the MFMA bodies (now phase-local re-reads,
  static-tail reservation, M8 pull/slot branch split — 60 B/lane left; the
  remainder is chaotic-allocator spread, not one mechanism).
- 2026-08-11 exp_01 **ROOT CAUSE of the `(nil)` fault** (supersedes the entire
  suspect ranking in `../../MPS_OVERNIGHT_HANDOFF.md`): the MPS M8 body assigns
  `pbase[t]` only inside the `j2 < fanout[t]` guard, whereas the reference
  assigns it unconditionally with safe defaults `p = cur; row = 0;`. The
  consumer (`..._mps.hip:408-410`) is NOT fanout-guarded, so out-of-fanout lanes
  shuffle `pb == 0` and load from `0 + off`; at `c==0, lane==0` the address is
  exactly 0. Fix = hoist the assignment out of the guard. Details and secondary
  items in `exp_01_nil_fault/root_cause.md`. Fix designed, NOT yet applied.
- 2026-08-11 exp_01 **all three prior suspects falsified by measurement.**
  (a) `mode=1` at `C=0` faults — with zero reserved CTAs no service CTA exists,
  so `run_service` internals cannot be the agent. (b) `desc[61]` is a valid mori
  pointer at heap_base+1.71 GiB of a 32 GiB heap
  (`MORI_SHMEM_HEAP_SIZE=34359738368`, `run_campaign.sh:117`); the 448 MiB
  request had 73x headroom, `moe_host_abi.hpp:197-201` throws on a null before
  the word is written, and `base_slot` never calls `peer_ptr`. (c)
  `pull_fallback=1` did not clear it. The fault is config-independent across
  mode, `C`, and `pull_fallback`.
- 2026-08-11 exp_01 **method lesson, cost me two runs**: a partial `K0_MPS_CFG`
  is rejected host-side before launch and the failure LOOKS like a pass — zero
  memory faults, no progress log, no correctness lines. Always pass all four of
  `C,g,mode,flush_rows`. Treat "0 faults + no progress log" as VOID, never CLEAN.
- 2026-08-11 exp_01 **inference lesson**: I read the descriptor's pointer
  alignment families as allocator identity and concluded `desc[61]` was outside
  the symmetric heap. That was backwards — large torch tensors are 2 MiB-aligned
  too, so alignment does not discriminate allocators. Only the measured
  `local_heap_base` + heap size settles range membership. Alignment is not
  provenance.
- 2026-08-11 **literature calibration — our `C` sweep is sized wrong** (two
  independent sources agree, and this is the highest-value pre-registered
  change). COMET's profiled optimum puts **14-35%** of blocks on communication
  (`n_c` = 18/26/46 of 132 SMs); MoK exposes **4-52 comms SMs of ~148 (2.7-35%)**.
  Our `C in {4,8,16}` of 256 CTAs is **1.6-6.3%** — below both. Worse, ICPP'26
  measured that under-provisioning COLLAPSES: `cCTA=2` (1.9%) ran up to **1.91x
  SLOWER than no overlap at all**, amplified by routing skew, while
  over-provisioning is only mildly bad. A flat or bad A1 curve at `C<=16` is
  therefore expected and must NOT be read as a kill on role specialization.
  Extend A1 to `C in {32,48,64,90}` before drawing any conclusion.
- 2026-08-11 **literature calibration — our push unit is ~70x too small.**
  COMET measured that per-tile transfers (32-64 KiB) sit far from bandwidth
  saturation (87 GB/s at 8 SMs only near 1 MiB), so they ship full-width row
  bands of 1-2 MiB; FlashOverlap coarsens tile->wave-group for the same reason.
  Our push unit is `896*g` bytes = 14 KiB even at `g=16`. Coarsening the push to
  contiguous multi-row bands gates whether the ~148 GB/s the cost model assumes
  is reachable at all.
- 2026-08-11 `primitives:` **MoK dropped the kittens-style abstraction layer**
  this year and wrote their megakernel directly, stating they did not need a
  framework. Logged as evidence about primitive libraries at this complexity
  level — it does not overturn our primitives-first mandate, but the mandate now
  has a known counterexample and should be judged on the `## Primitives`
  sections we accumulate, not assumed.
- 2026-08-11 `primitives:` **`translate_peer` fails closed to `nullptr`** for
  out-of-range ranks (`peer.cuh:40-49` via `pgl.cuh:124-137`) and **no MPS call
  site checks the result** (`moe_mps_adapter.cuh:232`, `:262`,
  `..._mps.hip:1535`). Currently unreachable thanks to a `live = r < env.t_ext`
  guard, but a fail-closed null that no caller is obliged to check is a footgun
  the library should not hand out. Candidate primitive change: a checked variant,
  or a debug-build trap.

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

- 2026-08-11 exp_01 **the `(nil)` fix landed and passed the full gate ladder on
  the first world-8 run**: correctness `max_abs=0.035156 relative=0.008293
  pass=True`, `pperr=0`, `control_fails=True`, `[MPS SOAK] 600/600`, zero
  memory faults, at the exact config that used to fault
  (`C=8,g=2,mode=2,flush_rows=16`). The fix is **resource-neutral** -- SGPR 104 /
  VGPR 256 / AGPR 256 / scratch 60 B / LDS 155,428 B, byte-identical to the
  pre-fix tuple. That **falsifies** `root_cause.md`'s guess that the reference
  form's two extra live values (`p`, `row`) were the source of the MPS build's
  +24 B scratch; the +24 B has another origin and is still open.
- 2026-08-11 exp_01 the harness already implements the negative control and the
  600-epoch soak; they run inside every campaign, including a 1-process smoke.
  No harness work was needed to satisfy the gate ladder.
- 2026-08-11 exp_02 **THE decision number.** Five rotations, three paired arms:
  `production 7,694.0` / `pf6gm_mega 6,885.8` (**0.89495**, confirming the
  G=3-forwarded reference) / `mps_mega 57,347.4` = **8.33x slower than the
  best**, at `C=8,g=2,mode=2,flush_rows=16`. All gates green, so it is a pure
  performance verdict on that configuration. `mps_mega` and `pf6gm_mega`
  report identical `max_abs`/`relative` per run -- the MPS arm is numerically
  indistinguishable from the reference, which is the strongest evidence the fix
  restored parity semantics rather than merely stopping the fault.
- 2026-08-11 exp_03 **method: screening is 40x cheaper and faithful for ordering.**
  A 1-proc/1-warmup/1-timed smoke costs 25-95 s vs ~18 min for a 5-rotation
  campaign, and reproduced the campaign's `mps_mega` figure to **0.10%**
  (57,292 vs 57,347). Use it to shape an axis; never to decide. 17 screening
  points, all gate-green.
- 2026-08-11 exp_03 **the reservation is cheap -- `role_partition` is exonerated.**
  Mode 0 (push OFF) costs **+1.6% at C=8, +3.7% at C=32, +6.1% at C=64** vs the
  paired `pf6gm_mega`. A naive `256/(256-C)` capacity model predicts +33% at
  C=64. The tax is sublinear because the reservation happens at M6.9 (only M7/M8
  are taxed) and those phases are not CTA-throughput-bound.
- 2026-08-11 exp_03 **mode 2 obeys an exact 1/C law, so the A1 axis has no sweet
  spot.** Service cost x C is constant across C in {8,16,32,48,64}:
  ~374,000-424,000 us x CTA at g=2, ~0.92-1.29M at g=16. Each service CTA
  contributes a fixed ~1.1-1.7 GB/s and they do not contend. **A1 is therefore
  closed by extrapolation from a fitted law, not by more points** -- extrapolated
  best over all C with today's throughput is ~7,262 us at C~103, so raising the
  `config_is_valid` C<=64 cap to reach C=90 was correctly skipped.
- 2026-08-11 exp_03 **A2 pre-registered expectation WRONG: `g` is an atomics
  knob, not a transfer-size knob.** At C=64 service cost rises monotonically with
  g: 2,953 / 5,842 / 9,025 / 20,171 us for g = 1 / 2 / 4 / 16. Reason:
  `moe_mps_adapter.cuh:340-346` probes all `g` group members with an
  `acq_rel` RMW every time any slice in the group completes, across 32 live
  lanes, so atomic traffic scales with g while the copy stays latency-bound.
  `g=1` degenerates the probe loop and wins. **Corollary: M3 (coarsen to
  >=256 KiB bands) CANNOT be tested through `g`** -- it needs completion
  detection decoupled from transfer size.
- 2026-08-11 exp_03 **the ceiling of the combine-boundary split, measured.**
  Mode 1 (bulk push, overlap OFF) = 8,049.2 vs paired pf6gm 6,903.4, and the 1/C
  law puts push_cost(256) at ~1,460 us, so **replacing M8's remote pull with a
  local slot read is worth only ~300 us** of the 1,309 us M8/M9 pool. Best
  conceivable mode 2 = `mode0(C) - 300 us` with perfect hiding = **0.882x
  production at C=8**, against 0.895x today. The prize at this boundary is ~1.3%,
  and collecting it needs a 25x service-throughput gain at C=8 (6.3x at C=32 for
  0.891x). **The combine boundary cannot carry the 0.80x objective.** Attack M6
  (2,773 us) and M7 (1,844 us) instead.
- 2026-08-11 `primitives:` **`store_peer_packets` is latency-bound by
  construction.** `packet.cuh:44-54` is a `#pragma unroll 1` loop of
  `out[packet] = in[packet]`: a load from local memory feeding a dependent
  store to the peer, so each lane sustains exactly ONE outstanding memory
  operation and the copy runs at one round trip per 16 B per lane. At g=1 the
  transfer is 896 B = 56 packets over 64 lanes -- **less than one packet per
  lane**, so there is not even enough work in a call to pipeline. The
  `unroll 1` is deliberate (its doc comment warns dynamic indexing would demote
  a register `packet16[N]` to scratch) but it makes the primitive the wrong
  default for global->peer copies. Proposed additive fix: a `Batch`-templated
  MULTI-REGION overload that issues Batch loads into registers before any store,
  giving Batch-deep parallelism to a caller with many small regions, with every
  existing caller untouched.
- 2026-08-11 `primitives:` **the library has no "group completion" vocabulary.**
  `counter.cuh` gives `counted_arrive_into` but nothing for "have all N
  counters of this group reached target", so the adapter open-codes it with
  `fetch_add_acq_rel(p, 0u)` as an acquire-carrying read. That idiom is both
  obscure and the direct cause of the g-scaling above.

- 2026-08-11 **TRAP that voided a whole experiment: `.cuh` edits do not
  invalidate the mori JIT cache.** The cache key is a content hash over mori's
  own `_jit-sources` tree restricted to `.hpp/.h/.cpp/.hip`; our headers
  (`moe_mps_adapter.cuh`, `moe_hk_adapter.cuh`, `include/cdna4/**`) arrive via
  `-I` from the read-only DHK mount and are NOT hashed, so a `.cuh`-only change
  silently reuses the previous `k0pf6gm_mps_mega.hsaco`. **My first staleness
  check was itself wrong**: I looked at JIT cache DIRECTORY mtimes, which are
  touched on a cache HIT. Only the `.hsaco` mtime counts -- the newest object
  was 08:40:42 while the screen ran 08:52-08:55. Guard now in place:
  `K0P6_MPS_SRC_REV` in the (hashed) `.hip`, bumped with any `.cuh` change, plus
  the screening driver `git reset --hard`s the node checkout and prints the
  newest hsaco mtime before and after every ladder.
- 2026-08-11 **method: the node checkout IS the arm.** Campaign containers
  bind-mount `~/Distributed-HipKittens` read-only. A ladder launched without
  syncing the node measures the wrong kernel and looks completely normal. Now
  done automatically by the driver.
- 2026-08-11 exp_04 **SUPERSEDES the earlier "MLP fan-out is a null" entry --
  that was the stale measurement.** With a verified-fresh build (JIT
  `66040eba7d06`), batching 4 independent (dst,src) regions so every load
  issues before any store cuts SERVICE cost by **14-29%** and end-to-end MPS time
  by 3.8-25%: C=8/g=2 57,292 -> 42,923 (-25.1%), C=64/g=2 13,263 -> 11,611
  (-12.5%), C=64/g=16 27,592 -> 22,182 (-19.6%), C=64/g=1 10,374 -> 9,976
  (-3.8%). Gain is larger at bigger `g` because more packets per region means
  more loads in flight. KEPT.
- 2026-08-11 exp_04 **the number that redirects the M-series: 4x MLP bought only
  ~20% of service cost, so the COPY IS A MINORITY of what the service pool
  does.** At most ~19% is byte movement; **~81% is bookkeeping** -- the per-slot
  event poll, the `fetch_add_acq_rel` arrival, the `g`-wide group-completion
  probe loop, the `atomicOr` claim, the `flush_pending` release. **Prefer the
  atomic/fence-removing ablations (M4 per-XCD arrival counters, A10 amortized
  release) over the byte-moving ones (M3 bands, M1/M2 cache bits) on this path.**
- 2026-08-11 exp_04 **the register tax of CTA role specialization, measured.**
  `packet16 staged[4]` is 64 B/lane and scratch went 60 -> 128 B: the staging
  array spilled to scratch, not registers, because the kernel is pinned at
  VGPR 256 / AGPR 256 by the MFMA path. That is why the gain is 14-29% and not
  ~75%. `kPushBatch=2` is the control (+16 B) and exposes a second failure --
  the compiler promotes the smaller alloca to **LDS**, taking LDS to 163,632 B
  and breaking the byte-exact 155,428 B gate, so Batch=2 is inadmissible.
  **A service CTA gets the same 256 ArchVGPR + 256 AGPR as a compute CTA because
  allocation is static and per-kernel; it never issues MFMA, cannot use what the
  MFMA path reserved, and cannot obtain one extra register.** CTA-level role
  specialization does NOT sidestep the HipKittens wave-specialization register
  problem -- **it relocates it.** Treat as a standing constraint on every future
  role-split candidate. The mode-0 control (7,387 vs 7,421) proves the reserved
  scratch is free to the compute path.
- 2026-08-11 exp_02b **dec02, decision campaign at the BEST mode-2 point**
  (C=64,g=1,flush_rows=16), 5 rotations, 3 paired arms, all gates green:
  `production 7,706.9` / `pf6gm_mega 6,891.7` (**0.89421**) / `mps_mega
  10,643.3` = **1.544x pf6gm**. Screening had predicted 10,374, so the 40x
  cheaper instrument tracks a real campaign to **2.6%**. Ratchet unchanged:
  `pf6gm_mega` at 0.894x production remains the best candidate.
- 2026-08-11 `literature:` **CLAUDE.md's M3 row and A2 note cited the wrong
  paper and the wrong hardware; both corrected in place.** The "32-64 KiB far
  from saturation / 87 GB/s / ~1 MiB knee" numbers are **arXiv 2607.19539 �3.3.2
  Fig 3(a) on 4x A100 over NVLink**, not COMET -- COMET (2502.19811) has no �3.3
  and **no bandwidth-vs-transfer-size measurement anywhere**. The "70x below the
  knee" arithmetic is right (73.1x) but the conclusion never divided by `C`: at
  C=32 the model needs only **4.63 GB/s per service CTA**, and 148 GB/s is 27.5%
  of aggregate egress, not a saturation target. AMD's own **mori-EP reports
  234-420 GB/s xGMI combine on MI355X at hidden=7168 BF16 -- a 14,336 B
  per-token payload, exactly our g=16 unit** -- refuting the strong form. **M3
  demoted below the fence class.** Also: COMET reports no separately measured
  layer-0 vs layer-1 speedup, so the M-series' "mechanism measured" tag on
  M7/M8 is unsupported; and Fleet is a single-GPU multi-die megakernel with no
  peer-bandwidth data, so it backs M4's fence claims only.
- 2026-08-11 `protocol:` **an unproven ordering hole makes every mode-2 g<16
  number provisional.** In mode 2 the row-completion flag is released by the
  wave that pushed the LAST slice group, but the other `16/g - 1` groups of that
  row were pushed by different waves on different CTAs, and `s_waitcnt vmcnt` is
  per-wavefront -- so the publisher's release covers only its own packets. At
  g=16 there is one claimant per row and the hole closes; at g in {1,2,4} it is
  open. It would present as intermittent wrong numbers with `pperr == 0`.
  Mitigating evidence: 15/15 correctness gates and 5/5 600-epoch soaks passed at
  C=8/g=2 and again at C=64/g=1, and mps_mega's `max_abs`/`relative` are
  identical to pf6gm's every run. Not yet discriminated. Cheapest test: same-run
  g=16 vs g=4. `counter.cuh`'s `counted_arrive_release_into` docstring warns
  about exactly this and the kernel open-codes past it.
- 2026-08-11 `protocol:` **A4 `pull_fallback=1` is not a valid CORRECTNESS arm
  as written** -- it reads remote `part` over xGMI, but mode 2 never executes a
  per-CTA system release of `part` (the mode-0/1 blocks that do are skipped and
  the reference's M7.5 grid barrier is deliberately gone). Its measured 10,576 us
  stands as a performance datum only. That datum is still decisive for its
  purpose: push and pull land within 2%, so **transport is not the lever.**

- 2026-08-11 `protocol:` **the mode-2 g<16 ordering hole is real in source and
  UNOBSERVED in practice -- 12/12 clean.** Direct sensitivity test: within a run
  `mps_mega` and `pf6gm_mega` see identical input, so their `[MOK GATE]`
  `max_abs`/`relative` must agree exactly; any within-run divergence is the hole
  firing. 8 trials at C=64/g=4 (hole open) and 4 at C=64/g=16 (control), each
  with a DIFFERENT seed so routing skew varies (max_abs moved across 0.035156 /
  0.035370 / 0.039062 / 0.042969, so the trials really were different problems).
  **Zero divergences, zero nonzero pperr, 12/12 soaks.** Plus the 15/15 campaign
  gates and 10 x 600-epoch soaks already banked. Likely (unverified)
  reconciliation: a row is only claimable once all 16 slices have arrived, so the
  other waves' packets have retired long before the claiming wave drains -- a
  TIMING argument, not an ordering guarantee, which would degrade under different
  skew or a faster fabric. **Not fixed, recorded**: mode 2 is closed for
  performance and hardening a closed path risks the resource tuple.
- 2026-08-11 `primitives:` **second independent pointer at the same gap.**
  `counter.cuh` expresses "arrive and release" for ONE counter but nothing for
  "all g members of this group have arrived AND every writer's payload is
  visible". No group-scoped release whose drain covers writers outside the
  calling wave. That absence forced both the open-coded g-probe loop (measured as
  the dominant cost driver of the g axis, exp_03) and the per-wave
  `s_waitcnt`-plus-bare-flag publish that carries the ordering hole. **This is
  the single highest-value primitive to add**, on the evidence of the night.

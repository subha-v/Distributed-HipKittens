# aug11 overnight lessons ledger (append-only; supersede, never delete)

Continues `../aug10/experiments/LESSONS.md`, which remains the aug10 ledger and
is not edited here. Every entry carries its evidence and the experiment that
produced it. Nothing is deleted; a later entry that overturns an earlier one
says so and names it.

- 2026-08-12 `method:` **A campaign on this node takes ~3.5 minutes, not
  ~20.** exp_35 ran **12 campaigns / 60 rotations** inside one batch window, and
  exp_33 ran two 5-rotation campaigns (n=10) in a 19-minute wall window
  (08:53–09:12Z) including builds. Every prior session scheduled against a ~20
  minute campaign cost and therefore under-ran the GPU by roughly 5×: arms were
  dropped, replicates were skipped, and single-campaign deltas were reported
  where four campaigns were affordable. **Budget many more arms per night, and
  prefer a second replicate campaign over any argument about noise.** (exp_33,
  exp_35.)
- 2026-08-12 exp_33 **M7 has overtaken M6 as the bottleneck.** At the ratchet,
  rank-0 CTA-max stamps, n=10: **M7 GEMM-2 2,701.8 ±23.74 µs = 46.2 % of the
  interior** against **M6 GEMM-1 2,453.3 ±2.51 µs = 41.9 %**. Together the two
  GEMMs are **88.1 %** of the 5,852.1 ±9.56 µs interior; plan M3–M5 is 372.8
  ±0.67 (6.4 %) and combine M8/M9 is 324.2 ±21.12 (5.5 %). The flip is exp_27's
  doing — the token-major ascale gather took −79.7 µs off M6 and moved the
  ranking. **Supersedes the budget table in `STATUS.md` §"The budget, and why the
  queue is ordered the way it is"** (M6 ~2,588 / M7 ~2,660), and with it the
  M6-first ordering of the optimization queue. The general rule: **re-profile
  after every landed win; a queue inherited from a stale profile is pointed at
  the wrong phase.**
- 2026-08-12 exp_33 **M7 and combine are one coupled block, anti-correlated at
  r = −0.904.** Their sum is **3,026.1 ±10.2 µs with sd 32.1**, against the
  **100.5 µs** sd that independence would predict from the individual sds (75.1
  and 66.8). Combine's 324 µs is **M7's slack**, not work: when GEMM-2's last CTA
  lands early the combine simply waits longer for peer arrivals. Consequences,
  both binding: (a) **attacking combine alone will not pay** — the time moves,
  it does not disappear; (b) **any claim to have moved combine by less than
  ~60 µs measured alone is inside the noise** of a quantity whose stderr is 6.5 %
  of its own mean. The real target is the 3,026.1 µs block, and inside it only
  the epilogue surcharge is deletable by a scheduling change. exp_35 reproduces
  the coupling independently: from rung (b) to rung (c), M7 moved −467.8 and
  combine −116.9 in the same direction, together 92 % of the −615.0 µs
  end-to-end move.
- 2026-08-12 exp_35 **Relocating the payload into the producer's epilogue is a
  REGRESSION of +210.6 µs unless the injection bound is present. The two knobs
  are not independent and must never be reported as additive.** With the service
  pool held **fixed at C=16** and one `g` bit flipped: rung (b), epilogue-carried
  mode-12 remote-accumulate with the throttle **disabled** (`g=65`), is
  **7,110.8 sd 13.7 µs = 0.9226×**, i.e. **+211.2 µs paired** *slower than the
  homogeneous baseline that carries no payload at all* (`pf6gm_mega` 6,900.2 sd
  15.5 = 0.8950×). Adding the depth-4 bound (`g=353`) gives **6,495.8 sd 6.9 =
  0.8422×**, **Δ −615.0 µs, t = −80.2**, with the per-campaign ranges of (b) and
  (c) **disjoint by 598 µs**. The throttle alone is **1.52× the entire (a)→(c)
  gap of −404.4 µs**; the payload relocation on its own accounts for **−52.1 %**
  of that gap. Campaigns ran alternating `c, b, c, b` in each of two batches, so
  node drift cannot be confounded with the mechanism. **The paper's Fig-4 rungs
  must be drawn as a composition, never as two separable contributions.**
- 2026-08-12 exp_35 **`C = 0` is illegal in mode 12**, so the CTA-dedication axis
  cannot be driven to zero inside mode 12. `moe_mps_adapter.cuh:373` rejects
  `mode_is_stream(c) && reserved_comm_ctas == 0u`, and `mode_is_stream` includes
  12 and 13. The only C=0 point the waterfall can reach today is rung (a), which
  is **a different kernel** (`pf6gm_mega`, no MPS protocol at all). Therefore
  **the placement adjudication's C=0 arm (exp_37 / paper Q2) is hard-blocked on
  mode 14** (exp_34), and any claim of the form "dedication does not matter" must
  be sourced from a mode-14 sweep, not from mode 12. Recorded companion: the
  `g` field's throttle encoding has **no "disabled" code point inside the `0x300`
  depth selector** — `config_is_valid` rejects a nonzero depth with the enable
  bit clear — which is what makes `g=65` the *unique* legal throttle-bits-only
  neighbour of `g=353` and rung (b) a one-variable rung. The `g` bit table:
  `0x000F` physical g, `0x0010` dual-write detect, `0x0020` throttle ENABLE,
  `0x0040` skip dead `part` zero-fill, `0x0300` depth select (00→8, 01→4, 10→16,
  11→32), `0xFC00` reserved-and-rejected.
- 2026-08-12 exp_33 `harness:` **`K0_PF6GM_DECOMP` is absent from
  `run_campaign.sh`'s `-e` forwarding list** (lines 110–147), and that single
  omission is why the M7 epilogue surcharge is a cross-run **proxy** instead of a
  same-run measurement. `pf6gm_mega` is a different kernel (`k0pf6gm_mega`,
  descriptor `desc_gm`, 55 slots), is never handed the `mps_state` timestamp
  block, and so emits no `[MPS TS]` stamps at all; the cumulative-prefix decomp
  path (`ab.py:5228`) is the one per-phase instrument it does have, and the
  campaign cannot reach it. A direct `torchrun` is not a substitute — it defaults
  the absolute tolerance to 1.0 and `K0_PF6GM_G` to 2. **A one-line harness edit
  fixes this and it is the highest-value harness change outstanding**, because it
  turns every surcharge proxy in the Fig-4 waterfall and the Q3 stacked bar into
  a same-run measurement. Until then the surcharge reads **~817–898 µs (~30–33 %
  of M7)** from two proxies whose independent subtrahends agree to 4.5 %
  (1,885.0 ±2.74 same-run `pf_full.n2_p2` rank-max vs 1,803.98 archived
  `[PF6GM DECOMP] M7_n2_phase2`) — trust the order of magnitude, not the digits.
- 2026-08-12 exp_33 `instrument:` **Rank-max does not exist for MPS phase
  stamps.** The `[MPS TS]`/`[MPS SPIN]` print block in `e004pf_k0pf_ab.py` sits
  inside `if rank == 0:` and reads `pf6_state["mps_state"]`, which is **never
  all-reduced and never written to the per-rank JSON**. Every MPS phase number
  ever reported — tonight's and every earlier note's — is the device **CTA-max
  within rank 0**, and "rank-max" in older documents means exactly that. The
  `production` stage profile *is* a genuine cross-rank `dist.all_reduce(MAX)`,
  so the two instruments are not comparable tick-for-tick: production's combine
  reads 1,356.0 rank-max against 978.4 rank-0 (**+38 %**, the largest rank spread
  measured tonight), which is the all-to-all skew the megakernel arms hide inside
  their epilogue. Two further stamp facts that must travel with any phase number:
  **1 tick = 0.01 µs** (100 MHz wall clock — the 2.2 GHz shader clock is the
  wrong divisor), and every stamp is a running max never reset, so after the
  600-epoch soak it describes the **final soak epoch**, not a timed iteration.
- 2026-08-12 exp_22 `ubench:` **Occupancy must be asserted, not assumed,
  whenever LDS changes.** Shrinking the ubench's LDS allocation moved the push
  primitive's by-reference pointer arrays out of LDS and **into scratch**, and let
  occupancy rise from **1 block per CU to 2** — which would have silently voided
  the entire premise of the experiment, since the whole saturation argument rests
  on one block per CU (the reason a communication CTA never shares an execution
  unit with a compute CTA on CDNA at megakernel budgets). The failure is silent:
  the kernel compiles, runs, and produces plausible curves. **Every ubench and
  every kernel edit that touches LDS must print and gate on occupancy, and a
  by-reference array of pointers is a scratch candidate the moment LDS pressure
  drops.**
- 2026-08-12 exp_34 `design:` **Several exp_30 mode-14 design points were simply
  wrong when met with real source; designs written against remembered line
  numbers drift.** Four corrections, each found only at build time: (a)
  `mode_is_stream` **could not be selectively widened** for mode 14; (b) **forcing
  `is_service_cta` false would have double-covered M7 tasks for any `C > 0`**,
  because `k0p6_mps_stride` is `nct − C` *unconditionally* — the tail CTAs would
  have silently re-executed M7 tasks, which is exp_25's named silent-failure
  shape; leaving the predicate alone is what made `C` a sound knob instead of a
  corrupt one; (c) a new symmetric `m7_done` array **could not be allocated**, so
  the arrival publish rides `row_ready`'s spare tail at index `T_ext`; (d) mode 14
  **had to be admitted to `skip_dead_part_zero`**. **Rule: a design is not
  reviewable until its predicates, strides and allocations have been read out of
  the current source; and any design that says "predicate P must be false for
  this mode" must check the task stride in the same breath.** Companion findings
  from the same build, all durable:
  - **You cannot add a phase to this megakernel for free.** The function sits on
    the 256-VGPR ceiling and the marginal spill victim is the phase-2 epilogue's
    `peer_tab` base load; a separate M7.7 phase cost **+16 B/lane of scratch**
    purely by duplicating the existing release/barrier/acquire/`payload_ok`
    sequence — not through the IRIS descriptor, not through `peer_ptr`, and not
    through the extra M8 template instantiation, all three probed and cleared.
  - **Register-allocation attribution on this kernel is non-decomposable.** The
    per-task hooks alone and the M7.5 changes alone each score 186/282 on the
    scratch-migration metric; together with everything else the score is 96/282;
    removing the hooks from the full patch changes nothing. **Bisecting a spill
    regression by deleting pieces will mislead you — report the cliff, not a
    per-piece cost.**
  - **`llvm-objdump -d` cannot read a `--genco` `.hsaco`.** It returns a one-line
    error that an ISA census script will happily count as `v_mfma=0`; one gate run
    in this experiment reported a clean MFMA-span census off exactly that error
    before it was caught. Use `hipcc --cuda-device-only -S`.
  - **Gating a barrier on `payload_ok` is a hang.** It is a relaxed read of a
    concurrently-written `pperr`, so two waves of one CTA can disagree and one
    skips the `__syncthreads()` the other is waiting on. **Barriers here may only
    be gated on grid-uniform descriptor decodes.** An early revision of the
    mode-14 patch had this bug; it is fixed in the delivered code and written up
    in `exp_34_mode14/plan.md` §3.5.

- 2026-08-12 exp_36 `harness:` **The routing-imbalance axis does not exist in
  this harness, and no plan may pre-register it again without first proving the
  knob is settable.** `K0_SYNTH_ROUTE` is **rejected outright** under
  `K0_INPUT_MODE=mok_synthetic` — which `run_campaign.sh` hard-wires and which
  `K0_BENCHMARK_PROTOCOL=mok_eager` requires (`ValueError` reproduced on all 8
  ranks). Even in the capture path the synthetic-route families are **decode-only**
  (`WORLD=8, T=64, TOPK=8, E=32`), and `synthetic_routes.py` exposes **no `std`
  parameter at all** — `skewed_hot` is a fixed deterministic pattern. So
  `std = 0.032` (COMET's production skew) and `std = 0.05` were never settable
  quantities and the charter's skew column cannot be run as specified. The only
  routing variable that exists is the router-logit seed
  (`K0_MOK_SEED_BASE` 1234 → 2468), whose reachable destination-load CV spans
  **0.0034–0.0086 — 4–15× *less* imbalance than COMET's 0.032**, so it is a weak
  substitute and is reported as one, not as a skew axis. **Cost: one experiment's
  worth of pre-registration.** This also means the pre-registered claim "skew is
  the ONE regime where a service pool may re-enter" is **untestable on this
  harness as built**, not refuted.
- 2026-08-12 exp_36 `defect:` **OPEN CORRECTNESS BUG — both megakernel arms
  produce wrong output at T=1024 and T=2048 while `production` passes.**
  `mps_mega` leaves **7 of 8 ranks' outputs entirely unwritten** (poison
  survivors = T·H per rank: 7,340,032 at T=1024 and 14,680,064 at T=2048), and
  `pf6gm_mega` reads `max_abs = 1.71 / 1.81, relative = 0.846` against
  `production`'s `max_abs = 0.023 pass=True`, with `pperr = 0` throughout — the
  kernel does not report an error, it silently under-writes. Reproduced 6/6
  configs with capacities pinned at T=4096 **and again with capacities scaled to
  T**, so it is not a capacity-sizing mistake in the driver. **This is a
  shape-fragility bug of the megakernels and is unrelated to MPS** — `pf6gm_mega`
  is wrong there too. Two consequences: **every timing result tonight is a
  T=4096 result** and must be captioned as one; and this defect is what stands
  between paper Q5 and a real batch/seqlen axis. **Worth its own experiment.**
  (T ≤ 512 is a separate, benign matter: refused by a host guard at `ab.py:554`,
  "prefill-only".)
- 2026-08-12 exp_36 **Placement is monotonically harmful in `C` at T=4096, and
  the ratchet's C=16 is not the optimum of its own axis.** Same-run, all gates
  green: `C=4` 6,466.1 ≈ `C=8` **6,469.5 (0.8388, n=3, replicates 6,464.7 /
  6,472.3 / 6,471.5)** < `C=16` 6,498.1 (0.8427, n=3) < `C=32` 6,719.3 (+221.2)
  < `C=64` mode 2 6,829.6 (+331.5). **`C=8` beats the `C=16` ratchet by 28.6 µs
  on three non-overlapping replicates each.** The whole effect is in **M7**, the
  payload-carrying phase; **M6 is flat to ±1.5 % across every arm**. The ranking
  reproduces identically under the second router seed. This is a **ratchet
  candidate, not a ratchet** — it needs a paired same-run confirmation against
  the standing config (exp_37 is running it). Note the direction: every step
  *toward* dedicating CTAs to communication is a loss, which is the paper's Q2
  answer arriving from the sensitivity sweep rather than from the placement
  experiment.
- 2026-08-12 exp_34 `method:` **A mechanism can smuggle a second variable into a
  waterfall rung, and only a protocol review caught it.** Mode 14 was specified
  as "delete the per-row completion protocol", but it *also* deleted the per-task
  VMEM drain — roughly **2,840 `vmcnt(0)` + `__syncthreads()` pairs per CTA** —
  so its number would have measured signal granularity and drain removal at once
  and been unattributable. The fix makes the drain selectable
  (`g |= 0x80`, `kCoarseKeepDrainBit`), which turns one ambiguous rung into two
  clean ones: `g=481,mode=14` (granularity, drain retained) then `g=353,mode=14`
  (+ drain deletion). **Rule: every new mode must be diffed for incidental
  deletions before its number is allowed into a waterfall.** One honest residue —
  the **event publication genuinely cannot be retained** (it has no consumer, and
  keeping it would make error bit 26 ambiguous), so that rung must be labelled
  **"coarse readiness including the event publication it deletes"**, not
  "granularity alone".
- 2026-08-12 exp_22 `method:` **Calibrate a pre-registered gate against a
  measured ceiling, not a spec sheet.** exp_22's H1 asked for 75 % of the
  76.8 GB/s nominal single-link bandwidth (57.6 GB/s) and fired
  `REFUTED_NEVER_REACHED` with `falsifier_triggered=true` — but the achievable
  ceiling is **56.9–57.3 GB/s (74.1–74.6 %) at every C from 8 to 64**, so **no
  CTA count could have passed the gate**. The mechanism was fine; the threshold
  was 0.4 pp above the hardware. The falsifier is reported unsoftened and the
  substantive question still separates cleanly (C=8 55.2 → C=64 56.5 GB/s, so 8×
  the pushers buys **+2.4 %**), but the general lesson is that a spec-sheet
  denominator can manufacture a refutation. **Pre-register thresholds as a
  fraction of a *measured* ceiling, or measure the ceiling first.**
- 2026-08-12 exp_23 `instrument:` **`pf6gm_mega` cannot carry MPS
  instrumentation at all** — 55-word descriptor, no MPS state slot, so it is
  never handed the stamp buffer. Any figure panel labelled "homogeneous" that is
  built on device stamps must therefore be **`mps_mega C=0,mode=0`, labelled
  "bulk, no overlap"**, not `pf6gm_mega`. This is the same wall exp_33 hit from
  the other side (its M7 surcharge is a proxy for exactly this reason), and it
  now has a workaround: mode 0 in the MPS kernel is the instrumentable
  homogeneous arm.
- 2026-08-12 exp_23 `node:` **This node exposes no live xGMI throughput
  counter.** `amd-smi --xgmi` returns **N/A** and `--shownodesbw` returns
  **0-0**. Every fabric-bandwidth cross-check must therefore use **UMC duty cycle
  (±15 pp) plus a computed ceiling** — 52.8 GB/s for coalesced 4 B atomics — or
  **rocprof TCC-EA**, which is named as the remaining gap. Any plan that promises
  an amd-smi bandwidth validation on this node is promising something the machine
  does not provide.

## Reproductions, not new lessons (recorded so they are not re-litigated)

- 2026-08-12 exp_33 + exp_35 **`[MPS SPIN]` is 0/0 — dispatch peer wait is ~0 at
  this shape, and exp_10 reproduces exactly.** `fail_max = 0` in all 10 exp_33
  rotations and in both exp_35 rungs; `success_max ≤ 1` (one rotation), against a
  2,000,000 limit, over 500 warmup + 100 timed + **all 600 soak** epochs per
  rotation, counters never reset. This is a **fact about the shape at routing
  std = 0**, not about the config — which is precisely why the skew sweep
  (exp_36) is the only experiment that can make a service pool re-enter.
- 2026-08-12 exp_33 **the ratchet reproduced**: `mps_mega` 6,487.5 (e33a) and
  6,496.6 (e33b) against the recorded 6,482.7 — **+0.07 % and +0.21 %** — pooled
  **6,493.8 µs = 0.8424× production**, and exp_35 independently lands rung (c) at
  **6,495.8 sd 6.9**. `production` is stable to **0.05 %** across exp_35's 12
  campaigns (7,709.2 sd 4.1), so it is effectively a constant denominator and a
  later same-harness campaign is directly comparable. **The stamps' identity check
  `plan + M6 + M7 + combine = interior` closes to 0.0 µs in all 10 rotations**,
  so the parse is by name and the decomposition is mutually consistent.
- 2026-08-12 exp_34 **mode 14's resource tuple is byte-identical to the ratchet
  on every gated field with mode 14 present in the build**: `SGPR 106 / VGPR 256
  / AGPR 256 / scratch 128 B per lane / LDS 155,496 / occupancy 1 / MFMA 180
  (96+84) / flat_atomic_pk_add_bf16 282 / zero scratch ops inside either MFMA
  span`, `K0P6_MPS_SRC_REV` 26 → 27. **No number exists yet — nothing was run.**
  One debit to price against any eventual mode-14 result: the build re-triggers
  exp_26's scratch migration into the remote-atomic epilogue (**96 of the 282
  `flat_atomic_pk_add_bf16` acquire a scratch op within 40 instructions ahead,
  against 0 in the base**), which exp_26 measured bundled into its mask 1 at
  **+2.81 µs, t = 0.61 — a null**, so the risk is bounded and small against the
  mechanism's ±245…695 µs, but it means **mode 14 vs mode 12 is not a perfectly
  clean single-variable comparison at the ISA level.** Pre-registered band
  **5,990–6,440 µs**, point estimate ~6,215; **a result above 6,568 falsifies the
  granularity rung** and is to be reported as such. The number is **BANKED, not
  ratcheted** — at `C = 0` mode 14 is a homogeneous megakernel and cannot be a
  role-split ratchet however fast it is; **the degeneration is the finding.**

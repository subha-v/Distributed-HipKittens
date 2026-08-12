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

- 2026-08-12 exp_34 `gate:` **The resource tuple is NOT a sufficient parity gate,
  and this is the most expensive lesson of the night.** Commit `291dfa08` passed
  **every** gated field — SGPR, VGPR, AGPR, scratch per lane, LDS, MFMA census,
  `flat_atomic_pk_add_bf16` census, and zero-scratch-ops-inside-either-MFMA-span —
  and still cost the mode-12 ratchet arm **+726.9 µs (11 % of its runtime)**:
  rev 26 `f113d73f` 6,497.3 µs vs the pin 7,224.2 µs (n=5), same session, same
  config, with `production` and `pf6gm_mega` unchanged to <5 µs. **From now on the
  only parity evidence we trust is `.text` sha256 byte-identity, or a measured
  end-to-end control on the ratchet config.** Every "compiled in but off"
  instrument must clear that bar — which explicitly includes exp_23's phase ring,
  whose tuple gate is green and whose timing invariant is unproven. Corollary for
  process: **re-time the ratchet config on every commit that touches the shared
  kernel**, not just re-gate it. The regression was found only because the brief
  demanded a same-session mode-12 control; without that control it would have
  silently poisoned every subsequent mode-12 denominator.
- 2026-08-12 exp_34 `codegen:` **Adding an unused code path can silently disable a
  `vmcnt`-based throttle while leaving its instruction counts unchanged.** The
  exp_24 injection bound is worth **−613.5 µs at rev 26** and **−1.8 µs at the
  pin** — inert — for mode 12, and **+7.2 µs** for mode 14, with **no mode-12
  source line edited** (all mode-14 branches are gated on `m == 14`),
  `n2_phase2_gm_mps.cpp` not in the diffstat, and the ISA census identical on
  exactly the four throttle instantiations
  (`BOUNDED={1:2, 4:96, 8:96, 16:96, 32:96}` in both revisions; the pin's extra
  204 `vmcnt(0)` are mode 14's own code). The visible trace is `SGPR 104 → 106` and
  `LDS +68 B`. **Scheduling-dependent mechanisms are fragile to codegen in shared
  functions in a way that instruction censuses cannot see: the instructions are
  present and do nothing.** Any commit touching a shared megakernel function must
  **re-measure the throttle contrast (`g=353` vs `g=65`), not just the arm** —
  that contrast is now the cheapest available detector for this failure class.
  Precise mechanism: **still OPEN**, needs a source owner (exp_38).
- 2026-08-12 exp_34 `gate:` **A rank-N-only failure is invisible in rank-0
  stdout.** Mode 14's protocol negative control (`R < world-1` publish loop) set
  `pperr = 33554432` (bit 25, `K0P6_MPS_ERR_M7DONE`) on **rank 7 only**, with
  ranks 0–6 clean and 29,360,128 poisoned survivors on rank 7 — exactly the
  pre-registered signature. But **rank-0 stdout shows `pperr=0` and a `nan` gate**,
  which reads at a glance like the *wrong* failure; the control could only be
  adjudicated in the **per-rank JSONs**. **Any gate that reads only rank 0 can
  miss a real failure**, and any negative control whose expected signature is
  rank-local must state which rank and be checked there.
- 2026-08-12 exp_34 `gate:` **The naive "any scratch op after the first `v_mfma`"
  MFMA-span rule fires on the reference arm itself.** The correct form is the span
  *grouping* implemented in `tools/e34_42_gate1.sh`. A gate that fails on the known
  good binary teaches nothing and costs a debugging cycle every time it runs:
  **bad gates cost more than no gate.** Validate every new gate against the
  reference arm before trusting it against a candidate.
- 2026-08-12 exp_34 `provenance:` **`summary.json`'s `kernel_hsaco_sha256` is an
  empty dict** and is useless as build evidence (already noted in exp_33; now
  confirmed to matter). Worse, the jit cache holds **two byte-different `mps_mega`
  builds whose `latest/` symlink flips randomly** — they differ in **40 of 188,360
  bytes, all inside `__hip_cuid_`**, so the difference is cosmetic, but the flip
  makes `latest/` an unreliable identity. **Resolve the real `.hsaco` with
  `readlink -f` + `stat -L`; never trust `latest/`,** and take build identity from
  the in-container per-rank record.
- 2026-08-12 exp_34 + exp_37 **Dedicating CTAs to communication never won at any
  pool size tested, in either mode — and mode 14's falsification is *stronger*
  evidence for paper Q2 than the flatness that was predicted.** exp_37, 27
  campaigns at rev 26: `C=4` 6,464.2 ≈ `C=8` 6,472.0 < `C=12` 6,490.4 < `C=16`
  6,498.0 < `C=32` 6,735.3 < mode 2 `C=64` **6,837.5** — monotone over a 374 µs
  span, with paired `C=8` −27.2 [−32.3, −22.1] and `C=4` −34.2 [−39.5, −28.9] in
  **7 of 7 rounds**. exp_34 supplies the C=0 point that mode 12's validator makes
  unreachable, and its C sweep is **monotone at 8.1–9.3 µs per reserved CTA** while
  the pool **provably has no job at all** (bit 26 never set in 4,032 `pperr`
  readings, `DRAIN=0`, `[MPS SPIN] 0/0`). **That is pure CTA-capacity loss with
  contention excluded by construction** — so no placement policy can be rescued by
  giving the pool less to do, which is a claim flatness could not have supported.
  The mechanism is visible: **M6 does not move at all** (paired −1.0 µs
  CI [−6.7, +4.8]; 1.8 µs = 0.07 % spread across `C = 4…32`), so the CTAs handed to
  the pool are taken from work that had nothing to gain.
- 2026-08-12 exp_34 **The drain deletion is a null (+3.2 µs), so the waterfall
  collapses from six rungs back to five — and the selector was still worth
  building.** Deleting ~2,840 `vmcnt(0)` + ~2,840 `__syncthreads()` per CTA buys
  nothing measurable (+3.2 µs, inside the ±6 µs campaign spread, sign the wrong
  way round). **Without `kCoarseKeepDrainBit` the entire −576.5 µs would have been
  mis-attributed to signal granularity.** The general form: a confound selector
  earns its cost even when — especially when — the confound turns out to be worth
  zero, because that is the outcome you cannot otherwise distinguish from the
  mechanism.
- 2026-08-12 exp_34 `hypothesis (not a result):` **the injection bound and the
  coarse signal may be substitutes rather than complements.** The bound is worth
  **−613.5 µs** at rev 26; deleting the per-row protocol is worth **−573.3 µs**
  measured while the bound was **inert**. Two numbers that close within 7 % of each
  other, both plausibly bounding the *same* in-flight-remote-write resource. If
  they are substitutes, the waterfall's rungs are not additive and mode 14's
  falsification is explained without any appeal to overhead. **Untestable until
  exp_38 lands**, then rung (d) must be re-run on a repaired binary. Recorded as
  the most interesting open question the night produced.

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

- **exp_23 tier A — a per-CTA phase timeline can be instrumented for FREE on
  this kernel. Parity gate GREEN.** Five `ts_last` → fused `ts_mark` swaps plus
  a 32 KiB fixed-slot ring in the spare tail of `K0P6_D_MPS_STATE`
  (`K0P6_MPS_SRC_REV` 28 → 29, base `5b1450d4`). Compiled in and runtime-off, the
  instrumented TU is **exactly** the arm: `SGPR 106 / VGPR 256 / AGPR 256 /
  scratch 128 B per lane / LDS 155,496 / occupancy 1 (asserted, not assumed) /
  MFMA 180 (96+84) / flat_atomic_pk_add_bf16 282 / spills 217-17 / zero scratch
  ops inside either MFMA span`. The `.text` differs (192,640 vs 192,448 B) so the
  instrument really is in there.
- **Generalizable: the free-instrumentation trick is that both VGPR and AGPR are
  already pinned at 256.** There is no headroom for the allocator to *use*, so a
  handful of extra values in a `tid == 0`, runtime-predicated block cannot move
  the tuple. Expect the same to hold for any future diagnostic shaped like this
  one (scalar-uniform, one lane, behind a runtime flag, no LDS); do NOT expect it
  for anything that touches LDS or lives across an MFMA span.
- **What the instrument costs when it IS collecting is not the ring — it is
  `timestamps=1`.** A 2×2 of force-on builds (`TAON` = coarse + ring forced on,
  `TCON` = coarse forced on with the ring compiled out) came out **identical**:
  scratch 128 → 144 B per lane, VGPR spills 17 → 21, scratch ops 168 → 172 on
  both. So the entire allocation cost belongs to folding the **pre-existing**
  coarse `atomicMax` stamps unconditional (already priced at ~15 µs end-to-end),
  and the ring itself adds **zero** registers and **zero** scratch even when
  unconditionally active — its whole cost is 5 × 8 B of plain global stores per
  CTA per epoch. Occupancy stayed 1 in all four builds.
- **G7 (the named quiet risk) is mechanizable, and the mechanization is worth
  more than the gate.** Writing `ts_last(); e23_mark()` instead of the fused
  `ts_mark()` compiles, passes the tuple, and silently demotes the coarse/per-CTA
  reconciliation from an identity to a few-tick approximation. Two grep-level
  checks catch it: a **source census** (`ts_mark` 5, `ts_last` 0, `e23_mark` 0)
  and an **ISA `s_memrealtime` census** (12 == 12, delta 0; the substitution
  would have made it 17). The strongest form is the full opcode-histogram delta:
  `TA − R` is *only* `flat_store_dwordx2 +5`, `s_cmpk_gt_u32 +5`,
  `s_lshl_b32 +5`, `s_mov_b32 +5`, `v_lshl_add_u64 +5`, `v_mov_b64_e32 +5`,
  `s_cbranch_scc1 +5`, `s_nop +3` — and `flat_atomic_umax_x2` stays 8, proving
  the coarse cells are still driven by the same eight atomics.
- **Trap, and it cost time: the "zero scratch ops inside either MFMA span" rule
  must use the SPAN definition, not "any scratch op after the first `v_mfma`."**
  The naive form fires on the **reference arm itself**, because this kernel has
  two MFMA spans (96 and 84 instructions) separated by ordinary spill-carrying
  code. Use the established grouping from `tools/e34_42_gate1.sh` — `v_mfma`
  indices grouped into runs with gaps < 400 disassembly lines — which is what
  `exp_23_fig10_timeline/tools/e23_g4_spans.py` now does. A gate that fails on
  the control is not a gate.
- **Process: do not hand-edit a node harness another agent's campaign is
  reading.** exp_23's host edit was generated, `git apply -p1 --check`-verified
  and `py_compile`-verified against the live file, then **parked** as
  `exp_23_fig10_timeline/e23_ab.patch` with base/post sha256 and a backup path,
  because the mode-14 campaign relaunches processes per rotation and a mid-flight
  edit would have made some of its rotations a different harness. The dump is
  also opt-in on `K0_E23_DUMP=1`, so once applied it leaves every existing arm's
  stdout byte-identical.

## exp_34 — mode 14 (coarse readiness), MEASURED: rung falsified, and a +727 µs ratchet regression found by the control that was supposed to be boring

- **THE BIG ONE: a commit whose diff cannot change mode 12 changed mode 12 by
  +726.9 µs.** `291dfa08` (mode 14) left every mode-12 line byte-identical — every
  new branch is gated on `m == 14` — and left the exp_24 throttle intact in ISA
  (`vmcnt(4)/(8)/(16)/(32)`, 96 sites each, identical count in both revisions).
  Yet `C=16,g=353,mode=12` measures **6,497.3 µs at `f113d73f` and 7,224.2 µs at
  `291dfa08` in the same session**, with `production` (7,700) and `pf6gm_mega`
  (6,905) unchanged to <5 µs. All of it is in **M7 (+818 µs)**; `planM6` moves 9 µs.
  The visible trace of the cause is `SGPR 104 → 106` and `LDS +68 B`. **Adding
  code to this function re-times phases that code cannot reach.**
- **The resource tuple is NOT a timing gate, and tonight is the proof.** The tuple
  passed byte-for-byte on every gated field (SGPR/VGPR/AGPR/scratch/MFMA/pk_add,
  zero scratch ops in either MFMA span) while the arm lost **11 %** of its
  runtime. **Re-time the ratchet config on every commit that touches the shared
  kernel** — one 3-minute campaign would have caught this before it shipped.
- **The mechanism to reach for when a config knob stops paying: compare the knob
  ON vs OFF at the new pin, not the arm against its own history.** `g=353` vs
  `g=65` is exactly throttle+depth vs neither. At rev 26 that pair is 6,497.3 /
  7,110.8 (**−613.5 µs**); at the pin it is 7,224.2 / 7,226.0 (**−1.8 µs**) for
  mode 12 and 6,650.9 / 6,643.7 (**+7.2 µs**) for mode 14. That one comparison
  converts "the arm got slower" into "**the injection bound went inert**", which is
  a different bug report with a different owner.
- **A rank-N-only failure is INVISIBLE in rank-0 stdout, and the negative control
  will look like it failed the wrong way.** exp_34's protocol control
  (`R < world-1`) printed `pperr=0` and a `nan` gate on rank 0 — which reads as the
  wrong failure mode — while the eight rank JSONs showed exactly the pre-registered
  signature: `rank 7: pperr=33554432` (bit 25), ranks 0–6 clean, 29,360,128
  poisoned values surviving on rank 7 with `first_rows=[0..15]`. **Adjudicate
  per-rank controls in the per-rank JSONs, never from the driver log.** Time was
  spent suspecting the control before reading the right file.
- **Deleting the per-task VMEM drain is worth NOTHING.** ~2,840 `vmcnt(0)` +
  ~2,840 `__syncthreads()` per CTA removed: **+3.2 µs, n=4 either side, sign
  reversed** (`g=481` 6,647.7 vs `g=353` 6,650.9). The `kCoarseKeepDrainBit`
  selector that made this separable was worth building — without it the whole
  −576 µs would have been mis-attributed to granularity *plus* drain, and the
  waterfall would have carried a phantom rung. **Build the confound selector even
  when you are sure of the answer.**
- **Reserving CTAs costs ~8–9 µs each EVEN WHEN THE POOL PROVABLY HAS NO WORK.**
  Mode 14 at C=0/8/16 = 6,650.9 / 6,725.6 / 6,780.4, monotone, with the service
  pool instrumented as idle three independent ways (bit 26 never set across 4,032
  `pperr` readings; `[MPS TS] DRAIN=0`; `[MPS SPIN] chunk_poll 0/0`). This
  separates **capacity loss** from **contention** for the first time: the aug10
  interference results are not needed to explain the placement penalty, because the
  penalty survives deleting everything the pool did. The prediction was FLAT; it
  was falsified, and the falsification is the stronger result.
- **Two mechanisms that were supposed to compose may be substitutes.** The
  injection bound is worth −613.5 µs at rev 26; deleting the per-row readiness
  protocol is worth −573.3 µs against the same transport with the bound inert.
  Neither has yet been shown to pay *on top of* the other, and at this pin it
  cannot be tested. If they are substitutes — both bounding in-flight remote
  writes — the waterfall's rungs are not additive and the paper's framing must say
  so. **Open question, explicitly not a result.**
- **Process that worked: `setsid` really does save the run.** The ssh session
  dropped three times mid-batch tonight (`client_loop: send disconnect`); every
  campaign survived and completed, and progress was recovered by reading
  `screen_<tag>.csv` + `rocm-smi --showpids` rather than by re-running anything.
  Corollary: the harness's GPU lock is what prevented a double-launch when a
  relaunch was attempted 6 minutes early — it refused with `FATAL … lock held`.
- **Process failure of mine, recorded: I wrote placeholder numbers into a result
  table while the campaigns were still running.** They were replaced with
  generated output before the file was committed, and the fix is now a rule:
  **result tables are rendered from the JSON by a script
  (`tools/e34_70_tables.sh`), never hand-typed.** A hand-typed number in a results
  file is indistinguishable from a measured one three days later.
- **A campaign at 5 rotations × (500 warmup / 100 timed / 600-epoch soak) takes
  ~3 min 5 s on this node**, so 8 interleaved arms is ~25 min, and n ≥ 4 on three
  arms plus a 3-point C sweep plus a cross-revision control fits in ~1 h 45.
  Budget by that number, and interleave: the four batches tonight each visited
  every arm, so the session drift that could have masqueraded as the C-sweep effect
  (74.7 µs) is bounded by the within-arm spread (~6 µs).

## exp_38 — the +727 µs regression FIXED (`275d2c2a`), and the parity gate replaced

Supersedes nothing above; it **explains** the exp_34 entry "a commit whose diff
cannot change mode 12 changed mode 12 by +726.9 µs" and closes it. Evidence:
`exp_38_ratchet_restore/text_parity.json`, `epilogue_window.json`,
`ratchet_confirm.json`.

- 2026-08-12 exp_38 `method:` **A parity gate must be sensitive to the MECHANISM
  the arm depends on, not merely to the resources the kernel occupies.** exp_34's
  gate proved the kernel still fit in the same registers, the same LDS, the same
  scratch allocation and the same MFMA census. **It never asked whether the
  throttle still throttled.** The arm's entire claim rested on `s_waitcnt vmcnt(4)`
  binding, and the one measurement that would have caught the regression at CPU
  time — how many atomics are issued between consecutive `vmcnt`-constraining waits
  — was in no gate. **Write the gate against the mechanism's invariant, then add
  the resource smoke check; not the other way round.**
- 2026-08-12 exp_38 `method:` **`.text` sha256 identity is the only parity gate we
  trust for a change claimed inert**, and it is cheap: one CPU build and one `cmp`.
  Corollary, now a rule: **a knob whose off-state is not `.text`-identical to the
  ratchet is not a knob, it is a second arm, and it needs its own campaign.** That
  is why exp_23's ring is default-off despite being exonerated — it adds a store at
  every stamped boundary and can never be byte-identical. The gate has a **second
  half that is equally load-bearing**: the flag-ON builds must **differ** (M14
  192,448 B, RING 179,648 B vs REF/DEF 179,520 B). A flag-on build that came out
  identical would mean the flag never reached the code and the arm would be a lie.
  Note also that `K0P6_MPS_SRC_REV` bumps are compatible with `.text` identity —
  the value is never read, it only keys the JIT cache.
- 2026-08-12 exp_38 `method:` **Add scratch OP COUNT and SGPR/VGPR SPILL COUNTS to
  the resource tuple, because `ScratchSize` is blind to spills that fit the
  existing allocation.** `ScratchSize` stayed pinned at **128 B/lane** across an
  11 % end-to-end regression while whole-kernel scratch operations went **19 →
  168**; SGPR/VGPR spills moved **186 → 217** and **15 → 17** and *would* have
  fired at CPU-gate time, hours earlier. "Zero scratch ops inside either MFMA span"
  also passed, because the spills landed in the **M7 epilogue, which is not an MFMA
  span** — a span-scoped check only protects the spans you named.
- 2026-08-12 exp_38 **A `vmcnt`-based throttle is a NEGOTIATED mechanism, not an
  instruction you own. Report the issue-run distribution alongside any throttle
  result or the result is not reproducible.** A throttle only binds if the
  surrounding code would otherwise exceed its cap, so the deciding quantity is the
  distance between the wait and the operations it bounds — and the compiler owns
  that distance. Measured: REF/DEF/RING 282 atomics in **12 runs, mean 23.5 in
  flight**; with mode 14 compiled in, **194 runs, mean 1.45**, 189 of them a single
  atomic, `vmcnt(0)` in the epilogue **21 → 117** alongside **+96 scratch ops**. The
  hardware never reached 4 outstanding remote RMWs, so `vmcnt(4)` capped something
  that never exceeded 1. **The throttle was not deleted — it was made redundant by
  a stronger involuntary throttle installed by the register allocator**, which is
  also why the sign and size come out right (exp_24 put the optimum near depth 4
  with a cliff in the wrong direction; effective depth 1.45 is well past it, plus 96
  full pipeline drains) and why `g=353` ≡ `g=65` to 1.8 µs at the broken pin.
- 2026-08-12 exp_38 **"Unreachable code is free" is FALSE in a megakernel.** The
  cleanest of the nine site ablations: **S4 is one line,
  `if (k0p6_m == 14ull) return;`, on a branch that cannot execute in that build
  because `config_is_valid` still rejects mode 14 — and it is one of the two worst
  sites**, collapsing the epilogue window on its own (+99 scratch ops, `vmcnt(0)`
  21 → 117). Meanwhile **S9 adds the largest lump of new code (+5,568 B `.text`,
  +50 scratch ops) and is inert.** **Code size is not the variable; where the code
  sits relative to the mechanism's live ranges is.** Four of nine sites collapse the
  window, five do not touch it. And there are **two independent routes to the same
  end state** — a spill route (S4, S8: `vmcnt(0)` 21 → 117) and a restructure route
  (S6, S7: zero extra scratch, `vmcnt(0)` unchanged, but the epilogue span widens to
  absorb 15 more throttle instantiations and mean run still collapses to ~2.8) — so
  a single-cause hypothesis would have been wrong even after finding the spills.
- 2026-08-12 exp_38 `method:` **Fix a regression with a GUARD, not a revert, when
  the offending work is wanted later.** Mode 14 behind `K0P6_MPS_ENABLE_MODE14=0`
  and the ring behind `K0P6_MPS_E23_RING=0`: default `.text` byte-identical to rev
  26, **no work lost, no patch file needed, both mechanisms one `-D` away.** The
  companion rule: **never publish a mode-12 number from a
  `-DK0P6_MPS_ENABLE_MODE14=1` binary** — that build still carries the +727 µs by
  design, so the pin *and the `-D` set* are part of a result's identity.
- 2026-08-12 exp_38 `method:` **Confirm a fix on the MECHANISM, not only on the
  wall clock.** Two checks were run: the ratchet reproduced at **6,488.7 vs 6,482.7
  µs (+0.09 %)**, and the injection contrast came back to **−618.7 µs** against
  **−613.5 at rev 26** and **−1.8 at the broken pin**. The first alone could be a
  quiet session; the second cannot — a timing coincidence cannot move a contrast by
  **344×**, and the two arms' value sets are disjoint by 597.2 µs. Carry the caveat
  with it: **n=3 per arm is enough to verify a 618 µs contrast and a 0.09 %
  restoration, and not enough to move a ratchet.**
- 2026-08-12 exp_38 **Exoneration needs its own evidence, and three independent
  lines are cheap.** The ring was the obvious co-suspect (same shared function,
  same demoted gate). It was cleared by: the +726.9 µs predating the ring's
  existence (rev 28 vs rev 29); the default build containing the ring code at flag 0
  being byte-identical to rev 26; and a ring-compiled-IN build matching rev 26 on
  every epilogue-window metric. **"Guarding X alone restores identity" is the
  necessary-and-sufficient form of the claim** — reach for it instead of guarding
  everything and declaring victory.

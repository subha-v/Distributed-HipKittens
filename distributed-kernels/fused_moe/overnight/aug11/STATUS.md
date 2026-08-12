# aug11 overnight — status and plan

Live document. Updated as each experiment lands. The append-only ledger for
tonight is **`LESSONS.md` in this folder** (aug10's remains at
`../aug10/experiments/LESSONS.md`); the paper-figure ledger is **`PLOTS.md` in
this folder — that is the morning read**; per-experiment detail is in
`exp_N_*/result.md`.

**Read the two OPEN sections below before quoting any number from this file.**
Every timing result in this document is a **T=4096** result — that is now known to
be the only shape at which either megakernel is correct. **The branch is healthy
again as of `275d2c2a`** (exp_38): the default build's `.text` is byte-identical to
rev 26 and the ratchet reproduces to +0.09 %. The ratchet stays at **6,482.7 µs /
0.8408×** — exp_38 *restored* it, it did not move it. The two remaining OPEN items
are the **T=1024/2048 correctness defect** and the **`C ≤ 8` ratchet candidate**,
which is now the top optimisation item.

## RESOLVED — the `291dfa08` regression is FIXED at `275d2c2a` (exp_38)

**`.text` sha256 byte-identity with rev 26, plus two independent GPU checks.** The
fix is a **guard, not a revert**: mode 14 sits behind `K0P6_MPS_ENABLE_MODE14`
(default **0**) and the exp_23 ring behind `K0P6_MPS_E23_RING` (default **0**), so
**no work was lost, no patch file was needed, and both mechanisms are one `-D`
away.**

| arm | build | `.text` sha256 | size |
|---|---|---|---|
| **REF** | `f113d73f` (rev 26) | `642646fc…a541a7` | 179,520 B |
| **DEF** | this tree, **no `-D` at all** | `642646fc…a541a7` | **179,520 B — identical** |
| M14 | `-DK0P6_MPS_ENABLE_MODE14=1` | `668f2608…ce55` | 192,448 B |
| RING | `-DK0P6_MPS_E23_RING=1` | `3f8645e5…17e2` | 179,648 B |

**M14 and RING must differ, and do** — that is the other half of the gate: a
flag-on build that came out identical would mean the flag never reached the code
and the arm would be a lie. `K0P6_MPS_SRC_REV` is **30**; it does not participate
in codegen (only in the JIT cache key), which is why the bump is compatible with
`.text` identity.

| flag | default | what it does |
|---|---|---|
| `K0P6_MPS_ENABLE_MODE14` | **0** | compiles in all nine mode-14 sites. `=1` reproduces exp_34's arms **and carries the +727 µs — never publish a mode-12 number from a `=1` binary** |
| `K0P6_MPS_E23_RING` | **0** | compiles in the exp_23 per-CTA phase ring; `=1` reproduces exp_23's figure arms |
| `K0P6_MPS_E23_FORCE_ON` | 0 | gate-only, folds the ring's runtime enable to compile-time true. Never in a shipped build |

**GPU confirmation — stamps OFF, 6 campaigns, arms alternated `353, 65, 65, 353,
353, 65`, all gates green** (`[MOK GATE] pass=True`, `[MARK] control_fails=True`,
`pperr=0`, `[POISON SELFTEST] nonfinite=57344`, `survivors=0`, `[MPS SOAK]
600/600`, `[MPS SPIN]` 0/0 on all six):

| check | measured | reference | verdict |
|---|---|---|---|
| **1. the ratchet is back** | `g=353` median **6,488.7 µs = 0.8423×** (6,484.7 / 6,488.7 / 6,498.1; mean 6,490.5) | published 6,482.7 / 0.8408× | **+6.0 µs = +0.09 %** — reproduced. `production` held at 7,701–7,710 (σ 3.3 µs) |
| **2. the injection bound is alive** | contrast `g=353` − `g=65` = **−618.7 µs** on means (−625.2 on medians), arm value sets **disjoint by 597.2 µs**; `g=65` median 7,113.9 = 0.9237× | **−613.5 µs at rev 26** (agrees to 5.2 µs) vs **−1.8 µs at the broken pin** | **the mechanism is restored, not just the wall clock.** A timing coincidence cannot move a contrast by **344×** |

**Caveat to carry: n=3 per arm** — enough for a 618 µs contrast and a 0.09 %
restoration check, **not enough to move a ratchet.**

**The culprit was the mode-14 code, not the ring**, on three independent grounds:
the +726.9 µs was measured at rev 28 *before the ring existed*; `DEF` contains the
ring code with its flag at 0 and is byte-identical to REF; and with the ring
compiled **in**, the M7 epilogue's injection window is identical to rev 26 on every
metric (12 issue runs, mean 23.5, zero scratch ops). The ring is default-off
anyway because it **cannot** meet `.text` identity by construction, and it was
never GPU-timed — it will be timed as its own arm.

### Site-level attribution — nine sites, each built alone

Reference REF/DEF: 179,520 B, SGPR spill 186, VGPR spill 15, 19 scratch ops.

| site | what it is | `.text` | SGPR sp | VGPR sp | scratch ops | epilogue? |
|---|---|---|---|---|---|---|
| S1 | `mode_is_direct_accum` third mode compare | 179,648 | 188 | 15 | 19 | — |
| S2 | `skip_dead_part_zero` admits mode 14 | 179,712 | 186 | 15 | 19 | — |
| S3 | `config_is_valid` + `kRemoteAccumGLegalBits` | 179,520 | 186 | 15 | 19 | — |
| **S4** | **`k0p6_mps_task_done`: `if (m == 14) return;`** | 181,632 | 186 | **17** | **118** | **COLLAPSES** |
| S5 | `k0p6_mps_task_drain` third mode compare | 179,520 | 186 | 15 | 19 | — |
| **S6** | **`task_done_maybe_defer` packed-word split** | 179,200 | 188 | 15 | 19 | **COLLAPSES** |
| **S7** | **kernel entry guard, `row_ready` tail index** | 181,376 | **209** | 15 | 19 | **COLLAPSES** |
| **S8** | **M7.5 rendezvous (`cfg75`/`coarse75`)** | 184,000 | 186 | **17** | **118** | **COLLAPSES** |
| S9 | M8 `m8_coarse` + `Ready=true` instantiation | 185,088 | 190 | 15 | 69 | — |
| ALL | all nine | 192,448 | **217** | **17** | **168** | worst |

**Four of the nine sites each independently collapse the M7 epilogue's injection
window; five do not touch it at all.** S9 adds the largest lump of new code
(+5,568 B, +50 scratch ops) and is **inert**, while **S4 — one line,
`if (k0p6_m == 14ull) return;`, on a branch that is runtime-unreachable in that
build because `config_is_valid` still rejects mode 14 — is one of the two worst.**
**Code size is not the variable. Where the code sits relative to the M7 epilogue's
live ranges is.**

### Mechanism — the throttle was not deleted, it was made redundant

The throttle is `s_waitcnt vmcnt(4)` in the M7 epilogue, and **a throttle only
binds if the surrounding code would otherwise exceed its cap.** The deciding
quantity is the *issue run length* — atomics issued between consecutive
`vmcnt`-constraining waits.

| arm | atomics | issue runs | max run | **mean run** | runs == 1 | `vmcnt(0)` in epilogue | scratch ops in epilogue |
|---|---:|---:|---:|---:|---:|---:|---:|
| REF / DEF / RING | 282 | 12 | 59 | **23.5** | 3 | 21 | **0** |
| M14 | 282 | 194 | 29 | **1.45** | **189** | **117** | **96** |

Mode 14's presence pushes whole-function register allocation over a cliff; the
compiler spills **inside the epilogue**; every spill reload drags an
`s_waitcnt vmcnt(0)` — a **full drain** — with it: **+96 scratch ops and exactly
+96 `vmcnt(0)`, 21 → 117.** Those drains chop the atomic issue stream from **12
runs averaging 23.5 atomics in flight to 194 runs averaging 1.45**, 189 of them a
single atomic. The hardware never reaches 4 outstanding remote RMWs, so `vmcnt(4)`
caps something that never exceeds 1. **The throttle was made redundant by a
stronger involuntary throttle installed by the register allocator.**

That explains the sign and the size, which "the throttle stopped working" alone
does not: exp_24 measured that depth has an optimum near 4 and that the wrong
direction is a **cliff**, so effective depth ~1.45 is well past it, plus 96 full
pipeline drains of latency — hence **+726.9 µs**, and hence `g=353` ≡ `g=65` to
1.8 µs, since both are dominated by the involuntary depth-1 throttle.

**Two distinct routes reach the same end state**, which is why this is fragile
rather than one bug: a **spill route** (S4, S8 — `vmcnt(0)` 21 → 117, +96 scratch
ops) and a **restructure route** (S6, S7 — zero extra scratch, `vmcnt(0)`
unchanged at 21, but the epilogue span widens enough to absorb 15 more throttle
instantiations, 81 → 96, and mean run still collapses to ~2.8).

### Why the tuple gate failed, precisely

**It reports scratch SIZE, not scratch OP COUNT.** `ScratchSize` stayed pinned at
**128 B/lane** across the regression while scratch operations went **19 → 168**,
because the spilled values fit the allocation that already existed. "Zero scratch
ops inside either MFMA span" also passed, because the spills landed in the **M7
epilogue, which is not an MFMA span**. SGPR/VGPR **spill counts** did move
(**186 → 217**, **15 → 17**) and would have caught this at CPU-gate time, hours
earlier — add them, plus whole-kernel scratch op count, to the tuple.

Data: `exp_38_ratchet_restore/text_parity.json` (`exp38-text-parity-1`, includes
the nine-site ablation), `epilogue_window.json` (`exp38-epilogue-window-1`, the
issue-run distributions and `vmcnt` histograms), `ratchet_confirm.json`
(`exp38-ratchet-confirm-1`, the six campaigns), plus `raw/e38a.driver.log` and
`raw/screen_e38a.csv`.

### The original defect record (superseded by the fix above, retained verbatim)

**The ratchet config does not reproduce at `HEAD`.** Measured in one session, same
config, same `production` denominator, 5-rotation campaigns:

| revision | `C=16,g=353,mode=12,flush_rows=16` | M7 stamp | planM6 |
|---|---:|---:|---:|
| `f113d73f` (rev 26) | **6,497.3 µs** (0.8437×) | 2,673.9 | 2,813.0 |
| `291dfa08` (rev 28, mode 14) | **7,224.2 µs** (0.9382×, n=5) | 3,492.1 | 2,822.1 |

`production` (7,700) and `pf6gm_mega` (6,905) are unchanged to <5 µs, so the
session is not slow — the `mps_mega` arm is. The whole delta is in **M7**.

**Mechanism, as far as it can be taken without editing the source:** every
mode-12 line is byte-identical across the commit (all mode-14 branches are gated
on `m == 14`), `n2_phase2_gm_mps.cpp` is not in the diffstat, and the throttle's
four compile-time instantiations are still in the ISA at identical counts
(`vmcnt(4)/(8)/(16)/(32)`, 96 each). What *is* observable is that the exp_24
injection bound has **gone inert**: `g=353` vs `g=65` (throttle+depth vs neither)
is −613.5 µs at rev 26 and **−1.8 µs at the pin** for mode 12, **+7.2 µs** for
mode 14. So this is a codegen/allocation effect in the shared function — visible
trace `SGPR 104 → 106`, `LDS +68 B` — and it needs a source owner. exp_34 was
measuring, not editing.

**Consequences as recorded at the time** (the third is now DONE; the first two
still stand — the pin rule is permanent and the tuple-gate finding is now
quantified in §RESOLVED):

- Any mode-12 denominator taken at `291dfa08` or later is a broken-transport
  number. That includes Q1 waterfall rungs (d)/(e), which therefore cannot be
  plotted against (a)–(c).
- **The resource-tuple gate is not sufficient.** It passed on every gated field
  while the arm lost 11 %. Add "re-time the ratchet config" to every commit that
  touches the shared kernel.
- Fixing this is the highest-EV item in the tree: **+727 µs**, more than the rest
  of the optimisation queue combined, and it is a regression rather than a new
  mechanism.

Full evidence, including the ISA census and the four-batch campaign set:
`exp_34_mode14/result.md` §1.

### Scope of the damage — narrow, and record it because it is reassuring

**No figure data is invalidated.** Every campaign that feeds a paper figure ran on
the *good* binary:

| experiment | pin | rev | binary |
|---|---|---:|---|
| exp_33 attribution (Q3) | `ca5b683f` | 26 | **good** |
| exp_35 waterfall (Q1 a–c) | `ca5b683f` | 26 | **good** |
| exp_36 sensitivity (Q5) | `b5215081` | 26 | **good** |
| exp_37 placement (Q2) | `b5215081` | 26 | **good** |
| exp_34 mode-12 control | `291dfa08` | 28 | **broken** |
| exp_34 mode-14 arms | `291dfa08` | 28 | **broken transport, see below** |

The only mode-12 number ever taken on the broken pin is exp_34's own control, and
it was taken *deliberately*, as the same-session control the brief required —
which is the only reason the regression was found at all. **Every mode-12 number
measured at or after `291dfa08` must be discarded.**

**One caveat the reassurance does not cover.** Mode 14's own arms were also
measured at `291dfa08`, and the injection bound is inert **for mode 14 too** there
(`g=353` vs `g=65`: +7.2 µs stamps-off, −5.4 µs stamps-on). Mode 14's M7 (2,861.4)
sits *between* the broken mode-12 M7 (3,492.1) and the working rev-26 M7
(2,673.9) — it recovers 630.7 of the 818 µs the regression put into M7 and keeps a
+187.5 µs residual. So the falsification is sound **as measured against the true
in-session ratchet**, but it is measured on a binary the regression touched, and
whether mode 14 would still miss the 6,568 threshold on a repaired binary is
**not established**. exp_34's own recommendation is to re-run rung (d) after the
fix; treat the falsification as final for tonight and re-testable, not as a
closed question about the mechanism.

### The tip now carries a SECOND unproven "compiled in but off" instrument

`d13acacf` (exp_23 Tier A, `K0P6_MPS_SRC_REV 29`) added a per-CTA phase ring to
**the same shared function** whose codegen the mode-14 regression implicates, and
it cleared **the same resource-tuple gate that mode 14 passed while costing
726.9 µs**. Two consequences were recorded: **exp_23's ring is not yet validated
against the timing invariant** and its parity gate is demoted accordingly (see
§exp_23), and **any number taken after `d13acacf` carries two unproven
perturbations rather than one**, so exp_38's byte-identity target had to be stated
against a named revision, not against "HEAD".

**Closed at `275d2c2a`.** The ring is now default-off behind `K0P6_MPS_E23_RING`
and the default build is byte-identical to rev 26, so the tip carries **zero**
compiled-in-but-off instruments. exp_38 also **exonerates the ring**: built with
the ring compiled in, the M7 epilogue window matches rev 26 on every metric, and
the +726.9 µs predates the ring's existence. The ring is still default-off because
it cannot be `.text`-identical by construction (it adds a store at every stamped
boundary) and it has never been GPU-timed — **it will be timed as its own arm.**

## Where we start

| arm | µs (campaign, 5-rotation median rank-max p50) | vs `production` |
|---|---:|---:|
| `production` | 7,715.6 / 7,720.0 | 1.000 |
| `pf6gm_mega` (homogeneous megakernel) | 6,911.4 / 6,902.4 | 0.895 |
| **`mps_mega` mode 12, `C=16 g=33 flush_rows=16`** | **6,685.5 / 6,683.1** | **0.866** |

Target: **0.80× ≈ 6,172 µs**, i.e. **−513 µs** from the ratchet.

## OPEN DEFECT — both megakernels are WRONG at T=1024 and T=2048 (exp_36)

**Unfixed, unattributed, and it bounds every number in this document.**

| arm | T=1024 | T=2048 | T=4096 |
|---|---|---|---|
| `mps_mega` | **7 of 8 ranks' output entirely unwritten** — poison survivors 7,340,032 = T·H per rank | same shape, 14,680,064 survivors | correct |
| `pf6gm_mega` | `max_abs=1.71 relative=0.846` | `max_abs=1.81` | correct |
| `production` | `max_abs=0.023 pass=True` | passes | correct |

`pperr = 0` throughout — **the kernel does not report an error, it silently
under-writes.** Reproduced 6/6 configs with capacities pinned at T=4096 **and
again with capacities scaled to T**, so it is not a driver capacity-sizing
mistake. **`pf6gm_mega` is wrong there too, so this is not an MPS bug — it is
shape fragility in both megakernels.** T ≤ 512 is a separate and benign matter
(host guard at `ab.py:554`, "prefill-only").

Two consequences, both binding: **every timing result in this document is a
T=4096 result** and must be captioned as one, and this defect is the single thing
standing between paper Q5 and a real batch/seqlen axis. **Worth its own
experiment.** Do not report a T-sweep until it is fixed.

## OPEN RATCHET CANDIDATE — `C=8` beats the `C=16` ratchet by 28.6 µs (exp_36)

**NOT ratcheted. Under paired confirmation in exp_37 (running, pinned at
`b5215081`). Do not move the ratchet on this alone.**

exp_36's CTA-dedication sweep at T=4096, same-run, all gates green:

| C | g | mode | n | p50 µs | replicates | ratio | Δ vs C=16 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 4 | 353 | 12 | 1 | 6,466.1 | — | 0.8395 | −32.0 |
| **8** | **353** | **12** | **3** | **6,469.5** | 6,464.7 / 6,472.3 / 6,471.5 | **0.8388** | **−28.6** |
| 16 | 353 | 12 | 3 | 6,498.1 | 6,488.7 / 6,504.6 / 6,500.8 | 0.8427 | — (ratchet) |
| 32 | 353 | 12 | 1 | 6,719.3 | — | 0.8705 | +221.2 |
| 64 | 1 | **2** | 1 | 6,829.6 | — | 0.8841 | +331.5 |

**Placement is monotonically harmful in `C`, and the ratchet's C=16 is not the
optimum of its own axis.** Three non-overlapping replicates each at C=8 and
C=16. The whole effect lives in **M7**, the payload-carrying phase; **M6 is flat
to ±1.5 % across every arm**. The ranking reproduces identically under the second
router seed (1234 → 2468). Every step *toward* dedicating CTAs to communication
is a loss — which is paper Q2's answer arriving from the sensitivity sweep rather
than from the placement experiment.

The standing ratchet remains **`C=16 g=353 mode=12 flush_rows=16` = 6,482.7 µs =
0.8408×** until a paired same-run campaign says otherwise.

### The paired confirmation arrived (exp_37) — and the condition on it is the absolute, not the delta

exp_37 ran the paired campaign this section asked for: **7 interleaved rounds,
21 core campaigns, `C=16` as an in-round control.**

| comparison | n pairs | Δ µs | 95 % CI | t (clustered) | sign test | rank-sum |
|---|---:|---:|---|---:|---:|---:|
| **C=8 − C=16** | 7 | **−27.2** | **[−32.3, −22.1]** | −13.03 | 0.016 | 0.00058 |
| **C=4 − C=16** | 7 | **−34.2** | **[−39.5, −28.9]** | −15.86 | 0.016 | 0.00058 |
| C=4 − C=8 | 7 | −7.0 | **[−15.5, +1.5]** | −2.02 | 0.45 | — |

**All 7 rounds negative for both candidates, campaign value sets fully disjoint**
(`max C=8` 6,477.0 < `min C=16` 6,492.6). **`C=4` vs `C=8` is a genuine tie** — the
interval spans zero and 2 of 7 rounds have the opposite sign — so the honest
statement is that **the winning region is `C ≤ 8`**, not that `C=4` is best.

**Still NOT moved, and the blocker is a measurement-hygiene one, not a
statistical one.** Every exp_37 arm carried `timestamps=1`, and the stamps-on
`C=16` control reads **6,497.5 µs** against the published stamps-off **6,482.7**.
So **the paired delta transfers and the absolute does not**: the candidate should
be quoted as "−34.2 µs against whatever `C=16` measures under the same
instrumentation" (≈ 6,448 µs if applied naively to 6,482.7) and **must be
re-measured stamps-off before it is published as a headline number**. That
stamps-off confirmation is the one outstanding condition.

Note that exp_37's own author **recommends** the move, to `C=4` with `C=8` equally
defensible, at medium-high confidence with the absolute-number reservation
attached. The ratchet is held here anyway, for two reasons: the stamps-off
confirmation is missing, and the branch tip could not reproduce *any* mode-12
number. Both had to clear first.

**One of the two has now cleared.** `275d2c2a` (exp_38) restores the branch, so the
only remaining condition is a **stamps-off paired campaign at `C ∈ {4, 8}` vs
`C=16`** at the healthy pin. With the branch healthy, this is the **top
optimisation item in the tree** — it is a ~30 µs win already measured 7/7 paired
rounds with fully disjoint value sets, and it needs a measurement, not a build.

## exp_37 — placement adjudication: DATA LANDED (paper Q2 / Fig 5)

**Does dedicating CTAs to communication ever win on AMD? No — not once, nowhere on
the reachable axis, and the loss is monotone in pool size.** 27 campaigns,
135 rotations, all gates green, `pperr = 0` everywhere, pin `b5215081` (rev 26 —
the good binary). `placement.json`, schema `exp_37.placement.v1`.

| arm | mode | C | n | p50 median µs | ratio | Δ vs C=16 | M6 | M7 | combine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **C=4** | 12 | 4 | **7** | **6,464.2** | 0.8393 | **−34.2** | 2,450.0 | 2,593.4 | 415.7 |
| **C=8** | 12 | 8 | **7** | **6,472.0** | 0.8406 | **−27.2** | 2,449.5 | 2,674.6 | 323.6 |
| C=12 | 12 | 12 | 2 | 6,490.4 | 0.8428 | −7.1 † | 2,450.9 | 2,731.0 | 268.3 |
| **C=16** (ratchet control) | 12 | 16 | **7** | **6,498.0** | 0.8436 | — | 2,451.0 | 2,696.1 | 344.1 |
| C=32 | 12 | 32 | 2 | 6,735.3 | 0.8746 | +237.8 † | 2,449.2 | 3,027.6 | 179.0 |
| **C=64 dedicated pool** | **2** | 64 | 2 | **6,837.5** | 0.8881 | +340.0 † | 2,480.5 | 2,916.1 | 442.3 |
| C=0 | 12 | 0 | — | `requires_mode_14` | | | | | |

† batch B contained no `C=16` campaign, so those three deltas are **unpaired
cross-batch** and quoted for the shape of the axis only. `C=0` in mode 12 is
rejected by the validator and is **not faked with a large-C proxy** — exp_34's
mode-14 `C=0` supplies that point.

**M6 is flat, and far more tightly than the prior claim.** Paired −1.0 µs
CI [−6.7, +4.8]; arm means span **1.8 µs = 0.07 %** across `C = 4…32`, against the
previously recorded "flat within 1.5 %". Taking 12 CTAs off the GEMM changes the
GEMM by nothing measurable — **one block per CU means there is no issue-slot
contention for a pool to relieve, so the pool is a pure N/(N−C) capacity tax.** The
only arm where M6 moves at all is the dedicated-pool design itself (mode 2 C=64,
2,480.5, +30 µs).

**A prior claim is corrected: "the whole difference lives in M7" is too strong.**
M7 alone and combine alone move in **opposite directions**, and the M7/combine
boundary shifts with `C` — at `C=4`, 102.6 µs leaves M7 and 71.6 µs reappears in
combine, and at `C=8` each of the two spans zero on its own. Their **sum** is
significant for both candidates and is the right size to explain the end-to-end
delta (−41.9 [−47.9, −35.9] for C=8; −31.1 [−53.0, −9.1] for C=4, against −27.2
and −34.2 end-to-end). **The defensible statement: the C effect is entirely inside
the payload-carrying phases and not in the GEMM.**

Method notes worth carrying: arms were **rotated inside each round** so drift
cannot masquerade as a `C` effect (it did not need to — same-run `production`
moved 0.30 % peak-to-peak over 1.5 h); every campaign's `K0_MPS_CFG` was verified
**from the kernel side**, read back out of all 8 rank JSONs of all 5 rotations,
which is the check that catches "a partial config LOOKS like a pass"; and the
invalid unclustered statistics are **stored in the JSON under
`INVALID_unclustered_rotation_t`** so the inflation is auditable rather than
invisible (C=4 vs C=8 would have read t = −2.76 unclustered vs −2.02 clustered).

**One gate deviation to record rather than round off:** 15 of 135 rotations (11 %)
report `[MPS SPIN] success_max=1` instead of 0. `fail_max` is **0 in all 135**, so
no poll ever exhausted, and the ones are spread across every arm and do not track
the effect. Peer wait is still zero for practical purposes — but "0/0 everywhere",
as earlier sections of this file say, is now **"0/0 in 120 of 135, and
success_max ≤ 1 in the rest"**.

Cross-session agreement: exp_37's `C=16` control lands at 6,497.5 µs (n=7) against
exp_36's independent 6,498.1 (n=3) — **0.01 % apart** — and every exp_36 point is
confirmed at higher n.

## exp_22 — saturation curves LANDED (paper Q4a / Fig 2)

250/250 plan points, 8× MI350X, `E22_SRC_REV 10`, `saturation.json` schema
`exp22-saturation-1`. 0/250 points with `rel_iqr_pct` above 1 % (median 0.05 %).

| series | knee C | plateau | % of nominal | reading |
|---|---:|---:|---:|---|
| xgmi single mlp4 | **8** | 56.9 GB/s | 74.1 % of 76.8 | saturated |
| xgmi single mlp1 / mlp8 | 16 | 55.5 / 55.7 | 72.2 / 72.6 % | saturated |
| xgmi rr7 mlp4 | **32** | 355.0 GB/s | 66.0 % of 537.6 | saturated |
| xgmi rr7 mlp1 / mlp8 | *(64)* | — | 61.2 / 61.3 % at C=64 | **no knee in range** (C64/C32 = 1.87 / 1.73) |
| hbm | **128** | 4,151.9 GB/s | 51.9 % of 8,000 | saturated |
| mfma shapes 4 / 16 / 256 | **none** | 119.9 / 352.0 TFLOPS | 5.2 / 15.3 % | **linear to 256**, R² ≥ 0.99989 |

**Communication saturates at 8–32 CTAs against 256 CTAs of compute capacity — a
topology-sized pool is sufficient and a large one is waste.** MFMA has no knee at
all, per-CTA efficiency 0.987–0.998 from 32 to 256 CTAs: **the
one-block-per-CU premise confirmed directly, so there is no issue-slot
contention for a dedicated pool to relieve.** H4 SUPPORTED ×3.

**H3 refuted in the good direction — the payload rides free, the protocol does
not.** Payload-only concurrent/isolated median **0.9951 (n=42)**; with the
protocol compiled in the same ratio falls to **0.878**, and the g=16 probe costs
up to **−42 %** (`rr7 c16`: 193.3 → 113.1 GB/s). **exp_20 reproduced as a
curve.** And **`reserved_only` never wins**: carrying payload costs the compute
pool **0.073 %** median, while *dedicating* the same 16 CTAs costs **6.25 %**
before a byte moves. Two honest exceptions where a payload pool does tax compute:
HBM at C=128 (10.1 %) and pushing past a saturated link's knee (14.2 %, a
cache-regime flip).

Concurrency is **proven per point**, not assumed: `wall < res_span + cmp_span` at
**75/75** (span-sum/wall 1.62–1.82), which is stronger than the
by-construction `overlap_pct`.

**The H1 falsifier FIRED and is reported unsoftened.** The gate asked for 75 % of
the 76.8 GB/s nominal (57.6 GB/s), but the achievable ceiling is **56.9–57.3
GB/s (74.1–74.6 %) at every C from 8 to 64** — **no CTA count could have
passed.** The threshold was calibrated against a spec sheet instead of a measured
ceiling; the mechanism was fine. The substantive question still separates: C8
55.2 → C64 56.5 GB/s, so **8× the pushers buys +2.4 %** and then nothing.

Three `knee_ctas` values in the JSON are artifacts and must not be annotated as
knees (`rr7` mlp1/mlp8, still rising; `mfma`, just linearity), and
`reserved_only` points carry `value = 0` **by design** — their payload is
`cmp_tflops_median`, never the bandwidth axis.

## exp_36 — sensitivity (paper Q5 / Fig 7): two of three axes do not exist

**The pre-registered grid could not be run, and the reason is the finding.**
`sensitivity_grid.json`, 49 points, pin `b5215081`, 15 campaigns + 10 screens, all
gates green at the one feasible shape.

- **The routing-std axis is not a knob.** `K0_SYNTH_ROUTE` is **rejected** under
  `K0_INPUT_MODE=mok_synthetic`, which `run_campaign.sh` hard-wires and
  `K0_BENCHMARK_PROTOCOL=mok_eager` requires (reproduced on all 8 ranks); the
  synthetic-route families are **decode-only** (`WORLD=8, T=64, TOPK=8, E=32`);
  and **`synthetic_routes.py` has no `std` parameter at all** — `skewed_hot` is a
  fixed deterministic pattern. **std = 0.032 and 0.05 were never settable.** The
  only routing variable that exists is the router seed, whose reachable
  destination-load CV spans **0.0034–0.0086, i.e. 4–15× *less* imbalance than
  COMET's 0.032** — reported as a weak substitute, not as a skew axis.
- **The T axis collapses to one point** — see the open-defect section above.
- **`C = 0` at any skew still needs mode 14** and is recorded as
  `"status": "requires_mode_14"` rather than faked with a large-C proxy.

Verdicts against pre-registration: prediction 1 (small vs large T) **unresolved —
no second T exists**; prediction 2 (a pool re-enters under skew) **unresolved as
stated and refuted where testable** — placement is monotonically harmful;
prediction 3 (depth sensitivity) **SUPPORTED** — depth 4 beats depth 8 by
**72.7 µs** and throttle-off costs **626.9 µs**; prediction 4 (zero peer wait)
**SUPPORTED** — `[MPS SPIN]` 0/0 at all 25 green points. **The throttle is worth
10–30× more than placement**, and M6 is flat within 1.5 % across every arm: the
entire difference lives in M7.

Driver deviation, auditable: `K0_T` and four other shape variables are hard-wired
as `docker -e` flags that cannot be overridden from outside, so `rc_T.sh` was
generated from `run_campaign.sh` **by sed** (full diff in `raw/rc_T.sh.diff`,
touching only those five lines plus a `SCRIPT_DIR` override). At T=4096 the
generated defaults reproduce the hard-wired values exactly and the control screen
read **0.8420×** against the known 0.8408× — **the driver is not a confound.**

## exp_34 — mode 14 MEASURED. Rung **FALSIFIED** at 6,650.9 µs; the granularity mechanism is real but does not beat the true ratchet

**Committed at `291dfa08`, `K0P6_MPS_SRC_REV 28`. Full ladder green; 26 campaigns
across 4 interleaved batches; every number below is a 5-rotation campaign median
with `production` as the same-run denominator.** Detail:
`exp_34_mode14/result.md`; plot-ready `exp_34_mode14/mode14_arms.json`.

| arm (stamps-off) | cfg | p50 µs | n | ratio | vs pin mode-12 | vs true ratchet |
|---|---|---:|---:|---:|---:|---:|
| true ratchet, in-session at rev 26 | `C=16,g=353,mode=12` | 6,497.3 | 1 | 0.8437 | −726.9 | — |
| mode-12 control **at the pin** | `C=16,g=353,mode=12` | 7,224.2 | 5 | 0.9382 | — | +726.9 |
| **mode 14, granularity alone** | `C=0,g=481,mode=14` | **6,647.7** | 4 | 0.8634 | −576.5 | +150.4 |
| **mode 14, + drain deletion** | `C=0,g=353,mode=14` | **6,650.9** | 4 | 0.8637 | −573.3 | +153.6 |
| mode 14, C=8 | `C=8,g=353,mode=14` | 6,725.6 | 2 | 0.8731 | −498.6 | +228.3 |
| mode 14, C=16 | `C=16,g=353,mode=14` | 6,780.4 | 2 | 0.8803 | −443.8 | +283.1 |

**Verdicts:**

1. **Band FALSIFIED.** Pre-registered 5,990–6,440 with 6,568 as the falsification
   threshold; measured **6,650.9**. Reported as a falsification, not
   re-interpreted. The family's optimism bias held again.
2. **The mechanism is nonetheless large and real:** deleting the per-row readiness
   protocol is worth **−573.3 µs (7.4 % of production)** against the same
   transport. It just does not clear the *working* ratchet, which is 150 µs faster.
3. **The drain deletion is a null: +3.2 µs** (`g=481` → `g=353`, n=4 either side).
   The `kCoarseKeepDrainBit` selector earned its keep by proving this; the
   six-rung waterfall can collapse back to five.
4. **The C sweep is NOT flat — it is monotone at 8.1–9.3 µs per reserved CTA**, and
   because mode 14's pool provably has no work (bit 26 never set in 4,032 `pperr`
   readings, `DRAIN=0`, `[MPS SPIN] 0/0`), this is **capacity loss with contention
   excluded by construction**. That is a stronger placement result than exp_37 was
   going to get, and it supplies paper Q2's previously impossible C=0 point.
5. **Service pool degenerated as predicted** — banked, not ratcheted: at C=0 mode
   14 is a homogeneous megakernel and cannot be a role-split ratchet.
6. **Open question worth the paper's attention:** the injection bound (−613.5 µs at
   rev 26) and the coarse signal (−573.3 µs with the bound inert) may be
   **substitutes, not complements** — both bound in-flight remote writes. Untestable
   at this pin; needs the regression fix, then a rung-(d) re-run. **The fix landed
   (§RESOLVED, `275d2c2a`), so this is now buildable with
   `-DK0P6_MPS_ENABLE_MODE14=1` and the re-run is unblocked.**

**Ladder, all green:** `[MOK GATE] … pass=True` with `pass_all_ranks` on all 8
ranks · `[MARK] control_fails=True` · **protocol negative control failed exactly as
pre-registered** — `rank 7: pperr=33554432` (bit 25) with ranks 0–6 clean and
29,360,128 poisoned survivors on rank 7, so the rendezvous **is** load-bearing ·
**bit 26 never set, 4,032 readings** · `[MPS SOAK] completed=600/600 pperr=0
poison=0 pass=True` · `[POISON SELFTEST] one_row_poisoned_fails=True
nonfinite=57344` · `[POISON] survivors=0`.

*(Historical note: the pre-GPU version of this section said "no GPU number
exists". Superseded, not deleted — the CPU-gate and protocol-review record it
described is retained in `exp_34_mode14/result.md` under "Pre-GPU record".)*

Review findings, all SOUND: happens-before on both the local and remote paths;
the **tid-0 observation** is sound because the code decomposes `cta_acquire` into
`__syncthreads()` plus a per-thread `acquire_fence<system>`, which is
**equal-or-stronger than the library idiom**, and no payload load falls in the
gap; **epoch staleness** sound (unsigned `>=`, `epoch = mega_count+1` so a zero
cell never satisfies the first poll, `retired[q] >= e` makes a future value
unreachable, wrap at 2³² against 600 epochs); **deadlock-free**; **mode
isolation** sound because every predicate change is an *added disjunct* on
`mode == 14`, so **modes 2 and 12 are bit-identical**.

Conditions cleared:

| condition | evidence |
|---|---|
| `row_ready` capacity | `mori_t((WORLD, T_LOC_MAX))` = **327,680 words**; mode 14's highest write is **319,488** — 32 KB of headroom, no overflow, and the entry guard **fails closed** if a future shape removes it |
| no concurrent clear | `row_ready` zeroed once at setup; the only other clear is gated on `K0_PF6GM_DECOMP=1` (never set under `benchmarks/`) and is bracketed by `synchronize()` + `dist.barrier()` |
| acquire is real in ISA | **`buffer_inv sc0 sc1` present and preceding the first payload load**, proven with a marker build |
| negative control exists | `negative_control.patch` cuts the publish loop to `R < world-1`; expected bit 25 on rank 7 only, gate fail |
| resource parity | tuple byte-identical to the arm; `K0P6_MPS_SRC_REV` → **28** |

**A confound was found and fixed, and it is the reason this got three rungs
instead of one.** Mode 14 had *also* deleted the per-task VMEM drain (~2,840
`vmcnt(0)` + `__syncthreads()` per CTA), which would have made the waterfall rung
measure two variables. The drain is now selectable — **`g |= 0x80`
(`kCoarseKeepDrainBit`) retains it** — so the waterfall reads: **mode 12 →
`g=481,mode=14` (granularity alone) → `g=353,mode=14` (+ drain deletion).** The
event publication **genuinely cannot be retained** (no consumer, and keeping it
would make error bit 26 ambiguous), so that rung must be labelled **"coarse
readiness including the event publication it deletes"**, not granularity alone.

Pre-registration stands unchanged: band **5,990–6,440 µs**, point estimate
~6,215, **above 6,568 falsifies**, and the number is **BANKED, not ratcheted** —
at `C = 0` mode 14 is a homogeneous megakernel. Debit still to price in: 96 of
282 `flat_atomic_pk_add_bf16` acquire a scratch op within 40 instructions ahead
(exp_26 measured that exact migration at +2.81 µs, t = 0.61 — a null).

## exp_23 — Tier A BUILT, parity gate green **on the tuple only**, and the gate is now DEMOTED

**Committed at `d13acacf`, `K0P6_MPS_SRC_REV 29`.** The per-CTA phase ring compiled
in and runtime-off is equal to the published arm on **all eight tuple columns** —
`SGPR 106 / VGPR 256 / AGPR 256 / scratch 128 B / LDS 155,496 / occupancy
**asserted** 1 / spills 217-17`, MFMA 180 with **zero scratch ops in each span**,
`flat_atomic_pk_add_bf16` 282 — while `.text` **differs** (192,640 vs 192,448 B),
which is how we know the instrument is genuinely present rather than compiled
away. G7 green in both halves: source census `ts_mark` 5 / `ts_last` 0 /
`e23_mark` 0, and ISA `s_memrealtime` **12 == 12**. A fourth attribution build
localises the entire ring-on cost (+16 B scratch, +4 VGPR spills) to
`timestamps=1` itself, **not** to the ring. The host patch `e23_ab.patch` is
`git apply -p1 --check` clean and passes `py_compile`, and is **deliberately
parked, not applied**, so it cannot become a second variable inside another
agent's running arm.

**The parity gate this cleared is now demoted, and exp_23 is the reason it has to
be.** `291dfa08` passed every one of those same tuple fields and cost the mode-12
arm **726.9 µs**. So a green tuple is no longer evidence that an off-path
instrument is free. **exp_23's ring needs an end-to-end timing control against
rev 26 — the ratchet config re-timed, or `.text` byte-identity — before any
timeline built with it can be published.** The ring is currently on the branch tip
stacked on top of the mode-14 code in the same shared function; see
§RESOLVED → "a SECOND unproven instrument". **Since `275d2c2a` the ring is
default-off behind `K0P6_MPS_E23_RING` and exp_38 exonerates it as the cause of the
+726.9 µs — but it still owes its own timing arm, because a knob whose off-state is
not `.text`-identical is a second arm, not a knob.**

*(The section below is the pre-build record, superseded on the build state and
retained for the design reasoning and the two plan corrections, which still
stand.)*

CPU only; the kernel files were **read only**. `patch_spec.md` is a ~20-minute
mechanical apply: **Tier A is five one-line `ts_last`→`ts_mark` swaps** at sites
that already decode the config and already read the clock, plus **a one-line host
buffer resize — no ABI change**, no new descriptor slot, no new flag, no atomics,
no LDS, and no code in either MFMA span. `parse_events.py`, `bin_timeline.py`,
`xcheck.py` and `selftest.py` are all built and **self-tested across 8 mutation
classes with every verdict proven to flip**. Verdict **GO at Tier A** (~15 %
parity risk), conditional at A+B, **NO-GO at A+B+C until the parity gate is
green**; ~45–55 min of GPU for all three arms.

Two plan corrections that change the figure:

1. **`pf6gm_mega` cannot be instrumented** — 55-word descriptor, no MPS state
   slot — so the homogeneous panel must be **`mps_mega C=0,mode=0` labelled
   "bulk, no overlap"**. Same wall exp_33 hit from the other side.
2. **This node has no live xGMI throughput counter** (`amd-smi --xgmi` → N/A,
   `--shownodesbw` → 0-0), so the promised bandwidth cross-check is replaced by a
   **UMC duty-cycle check ±15 pp plus a 52.8 GB/s fabric-ceiling bound**, with
   **rocprof TCC-EA** named as the remaining gap.

Also corrected in passing: `design.md`'s stale `LDS 155,428` / `SGPR 104` — the
real tuple is **155,496 / 106**.

## Queue state (supersedes the table in §"THE FIGURE NIGHT LANDED")

| item | state |
|---|---|
| **exp_33** attribution (Q3) | **DONE** — `phase_stamps.json` |
| **exp_22** saturation (Q4a) | **DONE** — `saturation.json`, 250/250 points |
| **exp_36** sensitivity (Q5) | **DONE as far as the harness allows** — `sensitivity_grid.json`; T=4096 only, skew axis absent |
| **exp_35** waterfall (Q1) | **PARTIAL, 3 plottable rungs of 6.** exp_34 measured (d) and (e) but at the regressed pin, so they are not commensurable with (a)–(c) and are held out of the figure pending a rung-(d)/(e) re-run; (f) never built |
| **exp_34** mode 14 | **DONE — rung FALSIFIED at the broken pin.** 6,650.9 µs (n=4, stamps-off) vs the 6,568 threshold. Mechanism real (−573.3 µs vs the pin's mode 12) but +153.6 vs the true ratchet; drain deletion a null (+3.2); C sweep monotone, not flat. **Surfaced the regression now closed in §RESOLVED**; exp_38 charged it to four of mode 14's nine sites. `mode14_arms.json` feeds rungs (d)/(e) and Q2's C=0 point, **all owing a re-run** |
| **exp_23** timeline (Q4b) | **TIER A BUILT** at `d13acacf`, tuple parity green, **parity gate demoted** — owes an end-to-end timing control vs rev 26, the parked host patch, then ~45–55 min GPU. No timeline data collected |
| **exp_37** placement (Q2) | **DONE — DATA LANDED.** `placement.json`; 27 campaigns / 135 rotations at `b5215081`. Dedication never wins, monotone in C, M6 flat to 0.07 % |
| C ≤ 8 ratchet candidate | **OPEN — now the TOP optimisation item**, its build blocker cleared by exp_38. `C=4` −34.2 [−39.5, −28.9] and `C=8` −27.2 [−32.3, −22.1] vs `C=16`, 7/7 rounds, value sets disjoint — but all arms were stamps-on, so it **owes one stamps-off paired campaign** at the healthy pin before the ratchet moves |
| **exp_38** regression fix | **DONE — GREEN first attempt (`275d2c2a`).** Default build `.text` byte-identical to rev 26 (`642646fc…a541a7`, 179,520 B); mode 14 and the exp_23 ring both guarded default-off; 6 stamps-off campaigns restore the ratchet to +0.09 % and revive the injection contrast to −618.7 µs. Nine-site ablation + mechanism published |
| T=1024/2048 correctness | **OPEN DEFECT**, unowned — see the open-defect section |
| mode-14 rung (d) re-run | **UNBLOCKED by exp_38, ~30 min GPU** — the only way to settle whether the injection bound and the coarse signal are complements or substitutes. Build with `-DK0P6_MPS_ENABLE_MODE14=1` and **never** publish a mode-12 number from that binary |
| exp_23 ring timing arm | **OWED** — the ring cannot be `.text`-identical by construction, so `K0P6_MPS_E23_RING=1` is a second arm and needs its own campaign before any timeline number is published |
| Q6 external ladders | **NOT STARTED** (stretch) |

**Ratchet unchanged: `C=16 g=353 mode=12 flush_rows=16` = 6,482.7 µs = 0.8408×** —
and, as of `275d2c2a`, **reproducible at the branch tip again** (6,488.7 µs,
+0.09 %). One candidate is queued behind it (`C ≤ 8`, worth ~−30 µs, owing a
stamps-off absolute); the +727 µs regression is closed and bought back nothing new,
only what already existed.

## THE FIGURE NIGHT LANDED (exp_33 + exp_35, pinned at `ca5b683f`)

Two campaigns' worth of paper evidence, no tree mutation, all gates green in
every rotation. **The ratchet reproduced** and the profile of the winner
changed hands.

| | exp_33 (n=10) | exp_35 (12 campaigns, 60 rotations) |
|---|---:|---:|
| `production` | 7,708.3 | 7,709.2 (sd 4.1 = **0.05 %**) |
| `pf6gm_mega` | 6,893.2 (0.8942) | 6,900.2 (sd 15.5, **0.8950**) |
| **`mps_mega` ratchet** `C=16 g=353 mode=12 flush_rows=16` | **6,493.8 (0.8424)** | **6,495.8 (sd 6.9, 0.8422)** |

exp_33's two campaigns read 6,487.5 and 6,496.6 against the recorded 6,482.7 —
**+0.07 % and +0.21 %**. Two independent experiments agree to **2.0 µs**.
`production` is stable to 0.05 % across 12 campaigns, so it is effectively a
constant denominator. **Remaining to 0.80× (≈6,167 µs at tonight's
`production`): −327 µs.**

### exp_33 — the authoritative profile of the winner (paper Q3)

Device stamps, **rank-0 CTA-max**, final soak epoch, n=10, 1 tick = 0.01 µs.
**This table supersedes the budget table in §"The budget, and why the queue is
ordered the way it is".**

| phase | µs | stderr | % of interior |
|---|---:|---:|---:|
| plan M3–M5 | 372.79 | 0.67 | 6.4 % |
| **M6 GEMM-1** | **2,453.30** | 2.51 | **41.9 %** |
| **M7 GEMM-2** | **2,701.84** | 23.74 | **46.2 %** |
| — of which epilogue surcharge | ~817–898 *(proxy)* | — | ~14–15 % |
| combine M8/M9 | 324.21 | 21.12 | 5.5 % |
| **interior M3–M9** | **5,852.14** | 9.56 | 100 % |
| *M7 + combine (coupled, r = −0.904)* | *3,026.05* | *10.16* | *51.7 %* |

**M7 is now the bottleneck, not M6** — exp_27's ascale change moved the ranking.
The two GEMMs are **88.1 % of the interior**. `plan + M6 + M7 + combine =
interior` closes to **0.0 µs in all 10 rotations**. The residual
`p50 − interior = 641.7 µs` is dispatch M0–M2 + launch + epoch skew, is
instrument-mixed, and is an upper bound good to ~100 µs.

Four things this profile settles:

1. **Plan is done as a target** (372.8 µs / 6.4 %) and **combine is not a target
   either** — its 324 µs is M7's slack, anti-correlated at **r = −0.904**, sum
   3,026.1 ±10.2 with sd 32.1 against the 100.5 µs independence would predict.
   Attacking combine alone moves the time, it does not delete it. Any claimed
   combine win under ~60 µs measured alone is noise.
2. **The M7 epilogue surcharge is a PROXY, not a measurement** — `pf6gm_mega`
   emits no stamps (different kernel, never handed `mps_state`), and
   `K0_PF6GM_DECOMP` is **absent from `run_campaign.sh`'s `-e` forwarding list**.
   Two proxies with independent subtrahends agree to 4.5 % (1,885.0 ±2.74
   same-run `pf_full.n2_p2` vs 1,803.98 archived). **The one-line harness edit
   that forwards `K0_PF6GM_DECOMP` is the highest-value harness change
   outstanding.**
3. **Rank-max does not exist for MPS phase stamps.** The print block sits inside
   `if rank == 0:` and is never all-reduced — "rank-max" in every older note
   means CTA-max within rank 0.
4. **`[MPS SPIN]` = 0/0** in all 10 rotations (`success_max ≤ 1` against a
   2,000,000 limit, over 500 + 100 + 600 epochs): dispatch peer wait is ~0 and
   exp_10 reproduces exactly. A fact about the shape at routing std = 0, which is
   why exp_36's skew sweep is the only experiment that can re-admit a pool.

`production` also has a phase signal, contrary to pre-registration: dispatch
918.70 ±2.01 / gemm 5,945.49 ±7.64 / combine 1,356.02 ±11.68 (rank-max, genuine
cross-rank all-reduce). Its combine carries the largest rank spread measured
tonight (1,356 rank-max vs 978 rank-0, **+38 %**) — the all-to-all skew the
megakernel arms hide inside their epilogue. The rank-max stages sum to +6.6 %
above its p50 because the max is taken independently per stage.

### exp_35 — the knob waterfall (paper Q1, the money figure)

Same commit for every rung, `production` an arm in all 12 campaigns, campaigns
alternating `c, b, c, b` in each of two batches so node drift cannot be
confounded with the mechanism.

| rung | what is added | config | p50 µs | sd | ratio | Δ vs prev |
|---|---|---|---:|---:|---:|---:|
| **a** | homogeneous baseline (`pf6gm_mega`) | — | 6,900.2 | 15.5 | 0.8950 | — |
| **b** | + epilogue-carried payload, **throttle OFF** | `C=16,g=65,mode=12,flush_rows=16` | 7,110.8 | 13.7 | 0.9226 | **+210.6** |
| **c** | + **injection bound** (depth 4) — the ratchet | `C=16,g=353,mode=12,flush_rows=16` | 6,495.8 | 6.9 | 0.8422 | **−615.0** |
| d | + coarse arrival signals (mode 14, drain retained) | `C=0,g=481,mode=14` | (6,647.7) | 5.4 | (0.8634) | **not plottable — see below** |
| e | + per-task drain deletion | `C=0,g=353,mode=14` | (6,650.9) | 5.4 | (0.8637) | **+3.2 — a null** |
| f | + nc-major producer task order | — | `null` | | | `not_built` |

> **Rung count updated to 6 by exp_34, then back toward 5.** The mode-14 rung split
> in two because the per-task VMEM drain deletion is selectable via
> `kCoarseKeepDrainBit`; exp_34 then measured (d)→(e) at **+3.2 µs, a null**, so the
> two collapse into one rung with the drain-deletion arm reported as free.
>
> **(d) and (e) are parenthesised because they are NOT commensurable with (a)–(c).**
> exp_34 measured them at `291dfa08`, where the mode-12 transport they ride on is
> **+726.9 µs slower** than at rung (c)'s commit and the rung-(c) injection bound is
> **inert** (see §RESOLVED). Against their own session's mode-12 control they
> are −573.3 µs; against rung (c)'s working ratchet they are **+153.6 µs**. Plotting
> the raw values next to (a)–(c) would show a rung going the wrong way for a reason
> that has nothing to do with granularity. **The figure needs a rung-(d)/(e) re-run
> once the regression is fixed**; the JSON carries both deltas so the re-run is a
> substitution, not a rebuild.
>
> One consequence worth stating in the paper rather than hiding: rung (c)'s
> injection bound (−613.5 µs) and rung (d)'s coarse signal (−573.3 µs, measured
> with the bound inert) may be **substitutes** — both bound in-flight remote writes
> — in which case the rungs are not additive and the waterfall's cumulative
> framing is wrong as drawn. Untestable until the regression is fixed.

**With `C` held fixed at 16, one scheduling bit moves end-to-end by −615.0 µs —
1.52× the entire (a)→(c) gap of −404.4 µs — while relocating the payload into
the producer's epilogue *without* that bound costs +210.6 µs against a baseline
that carries no payload at all.** That is the paper's central claim measured
directly. `t = −80.2` on (b)→(c); the per-campaign ranges of (b) and (c) are
**disjoint by 598 µs**. Phase attribution of the −615: **M7 −467.8, combine
−116.9**, plan −1.5 (M6's −47.0 is codegen drift — the throttle is a
compile-time specialization, so (b) and (c) are different `.hsaco` from the same
pinned source). **92 % of the move is M7 + combine**, exactly where the
epilogue's remote RMWs live.

**The two knobs are not independent and must never be reported as additive.**

Two structural facts derived from source and proved three ways (descriptor dump
of the word the kernel actually read, the exp_24 depth-16/32 mechanism
signature, and a fail-closed negative control that raised
`K0P6_MPS_ERR_CONFIG`):

- **The `g` bit layout for modes 12/13:** `0x000F` physical g (must be 1),
  `0x0010` dual-write detect, **`0x0020` throttle ENABLE**, `0x0040` skip the
  dead `part` zero-fill, **`0x0300` depth select (00→8, 01→4, 10→16, 11→32)**,
  `0xFC00` reserved-and-rejected. **There is no "disabled" code point inside
  `0x300`** — `config_is_valid` rejects a nonzero depth selector with the enable
  bit clear — so `g=65` is the **unique legal throttle-bits-only neighbour** of
  `g=353` and rung (b) is genuinely one-variable. (`g=321` is refused.)
- **`C = 0` is illegal in mode 12** (`moe_mps_adapter.cuh:373`, and
  `mode_is_stream` includes 12/13). The waterfall therefore **cannot** drive CTA
  dedication to zero inside a single arm; the only C=0 point available is rung
  (a), a different kernel. **Driving that axis to zero requires mode 14** — which
  made exp_34 a blocker for exp_37 / paper Q2 as well as for rung (d).
  **Cleared:** exp_34 ran C=0 legally in mode 14 and, better, ran C ∈ {0, 8, 16} in
  the same arm with the pool provably idle, so Q2 now has a C=0 point *and* a
  contention-free measurement of what reserving a CTA costs (8.1–9.3 µs each).

Honest weakness, recorded: **step (a)→(b) is not single-variable and cannot be
made one** — rung (a) is a different kernel with no MPS protocol, so that step
bundles the mode-12 transport, the C=16 pool, and the per-row protocol. Step
(b)→(c) is strictly single-variable.

### Queue state after tonight's figure block

> **SUPERSEDED by §"Queue state" above.** exp_22 and exp_36 have since landed and
> exp_34 has passed protocol review. Kept for the record.

| item | state |
|---|---|
| exp_32 NaN poison | **LANDED** — `K0_MOK_POISON_OUT` on, poison selftest and survivors=0 green in every rotation of exp_33 and exp_35 |
| **exp_33** attribution (Q3) | **DONE** — `exp_33_attribution/phase_stamps.json`, `exp33-phase-stamps-1` |
| **exp_35** waterfall (Q1) | **PARTIAL, 3 rungs of 5** — `exp_35_waterfall/waterfall.json`, `exp35.waterfall.1`; (d) and (e) are `null` rows with `blocked_by`, so the figure completes in place without re-running (a)–(c) |
| **exp_34** mode 14 | **BUILT, CPU GATE GREEN, NOT RUN.** Resource tuple byte-identical to the ratchet on every gated field with mode 14 in the build: SGPR 106 / VGPR 256 / AGPR 256 / scratch 128 B / LDS 155,496 / MFMA 180 / `pk_add_bf16` 282, zero scratch ops in either MFMA span; `K0P6_MPS_SRC_REV` 26 → 27. **Owes: protocol review, then the full GPU gate ladder.** First config `C=0,g=353,mode=14,flush_rows=16`. Band 5,990–6,440; **above 6,568 falsifies**; the number is **BANKED, not ratcheted**. Debit to price in: 96/282 `pk_add_bf16` acquire a scratch op within 40 instructions ahead (exp_26 measured this exact migration at +2.81 µs, t = 0.61 — a null), so mode 14 vs 12 is not ISA-clean single-variable |
| **exp_22** saturation (Q4a) | **BUILT, CPU GATE GREEN, DATA IN FLIGHT** — standalone `e22_saturation.hip`; the sweep is ~3 min of GPU; `saturation.json` not yet written |
| exp_23 timeline (Q4b) | **PENDING** — plan + design written, no instrument built |
| exp_36 sensitivity (Q5) | **PENDING** — std = 0 anchor column already measured (`[MPS SPIN]` 0/0) |
| exp_37 placement (Q2) | **PENDING, and its C=0 arm is hard-blocked on exp_34** |
| Q6 external ladders | **NOT STARTED** (stretch) |

**Phase 2 re-pointed by exp_33's profile.** The optimization queue inherited
from the pre-exp_27 profile is aimed at the wrong phase. In expected-value order
now: (1) **mode 14** — the direct attack on the ~820–900 µs surcharge, and the
first evidence that a win of that size is *physically available* inside M7
rather than assumed; (2) **exp_31, the phase-2 VMEM hint `14 + kGM`** — one
line, aimed at the other ~1,880 µs of M7; (3) **exp_26's mask ladder and the M6
software-pipeline depth** — M6's 2,453.3 µs is the second-largest term and has
the *tightest* stamp (stderr 2.51 µs), so it is the cheapest phase to judge a
change on; (4) **nc-major task reorder**, which is also paper Fig-6 data. Plan
and combine are both closed as targets.

`[method]` **A campaign on this node costs ~3.5 minutes, not ~20.** exp_35 ran
12 campaigns / 60 rotations in one window; exp_33 ran n=10 plus builds in 19
minutes of wall clock. Every prior session scheduled against ~20 min and
under-ran the GPU by ~5×. Budget many more arms per night, and prefer a second
replicate campaign over any argument about noise.

## RATCHET MOVED AGAIN (exp_27, `f113d73f`) — **0.8408× production**

**`C=16, g=353, mode=12, flush_rows=16` + `K0P6_MPS_ASCALE_TM 1` = 6,482.7 µs =
0.8408× production**, from 6,568.0 / 0.8522×. **−85.3 µs, +1.14 points.** Two
candidate campaigns (6,477.0 / 6,488.3) paired against a same-session control
campaign that reproduced the old ratchet to 0.16 %.

| | | |
|---|---:|---|
| M6 stamp, control | 2,534.12 ± 10.71 | n = 10, two batches |
| M6 stamp, candidate | **2,454.39 ± 13.20** | n = 10, two batches |
| **ΔM6** | **−79.7 µs, t = −14.84** | the two controls agree to 5.1 µs, so the effect is **15.6× the drift** |
| paired campaign delta | −75.0 µs | lands within 4.7 µs of the stamp — **M6's saving passes through ~1:1** |

Resource tuple exact: `SGPR 106 / VGPR 256 / AGPR 256 / scratch 128 B / LDS
155,496`, MFMA 180, zero scratch ops in either MFMA span. The ISA came out
better than designed: trip count **21 → 6**, `flat_load_dword` → `dwordx4`,
stride folded as the immediate `0xe0 = 224`, and the four LDS stores merged into
two `ds_write2_b32`.

**Remaining to 0.80×: −317 µs.**

### Two corrections this experiment forced

1. **The M5 transpose was cheap to produce, expensive to consume.** Deleting the
   whole M5 loop (`scale_transpose_row` is now compiled out) was worth only
   **−2.95 µs (t = −1.71)**, not the predicted −17: `sc_dst[lane·T_ext + t]` has
   consecutive `t` adjacent, so the scatter **coalesces in L2**. Only the
   *consumption* side was ever 16×-amplified.
2. **The bit-identity gate I demanded was not well-posed, and this experiment
   proved it.** A binary whose `.text` is byte-identical to the ratchet printed
   two different `[MOK GATE] relative` values across five runs, and **untouched
   `production` printed two different `max_abs` values in those same runs**.
   MoK's dispatch assigns receive rows by `fetch_add` on a device counter and
   mode 12's combine is a nondeterministic bf16 remote atomic, so **`out` is not
   bit-reproducible in *any* arm.** The sound substitute: same-run paired
   `mps − pf6gm` at full precision from the rank JSONs — shift **−8.8e-9,
   t = −0.73**, with `max_abs(mps) == max_abs(pf6gm)` in **160/160** cells and
   the negative control sitting 108× above the arm. Bit-identity of M6's *loads*
   was proven structurally instead.

### A statistical rule that nearly caused a false stop

**Cluster by RUN, not by (run, rank).** The eight ranks in a run share one
input. That numerics null reads **t = −0.73 clustered and t = −5.01
unclustered** — the unclustered version would have halted a correct experiment.

### Prediction haircut, now measured twice

exp_27's M6 term came in **43 % below** its point estimate and its M5 term
**83 %** below. Read every remaining prediction in this family pessimistically,
including exp_30's +245…+695 µs.

## RATCHET MOVED (exp_24, `9530382a`)

**`C=16, g=353, mode=12, flush_rows=16` = 6,568.0 ± 4.6 µs = 0.8522× production**
(0.9487× `pf6gm_mega`), across **three independent 5-rotation campaigns** with
the full gate ladder green in all 15 rotations. `g=353` composes Mechanism A
(dead `part` zero deleted) with throttle depth 4.

**−117.5 µs and +1.43 points of margin** on exp_21's 6,685.5 / 0.8665.
Remaining to 0.80×: **−396 µs.**

Resource gate *improved*: SGPR/VGPR/AGPR/LDS unchanged, **scratch 144 → 128
B/lane**, MFMA census 180, `flat_atomic_pk_add_bf16` 282, zero scratch ops
inside either MFMA span.

Two corrections to what was believed earlier tonight:
- **Mechanism A's 448 MiB was fully exposed on the M3→M4 critical path**, not
  absorbed by the LLC. The pre-registered null (that CTA 0's serial `tile_desc`
  build dominates the plan phase) is **refuted**. And the LLC-eviction story is
  dead exactly as pre-registered: M6 moved −14.7 ± 11.3 µs (t = 1.3), because
  W13 (939 MB) and W2 (469 MB) each exceed the 256 MB Infinity Cache anyway.
- **Throttle depth 4 beats 8** by −50.9 µs at campaign resolution — invisible on
  the M7 stamp (−83 ± 63 µs) but ~9σ across campaigns. Above 8 the axis is a
  cliff: 16 and 32 return M7 to the *unthrottled* cost (+462 / +370 µs). Depth 2
  is unmeasured and the `g` `0x300` selector has no room left.

Two further exp_24 findings worth carrying:
- **The remote-atomic epilogue is not instruction-issue bound.** Folding
  `slot_off` into the peer table removes 3 instructions from each of ~350 M
  remote atomic ops per rank per epoch and measured a **campaign-resolution
  null**. It survives only as a register-pressure *enabler* — which is what
  paid for Mechanism B's spill fix.
- **The `g` field silently overflowed into `mode` for any `g > 0xFF` and still
  validated.** `g = 0x121` set `mode |= 1` and decoded back as `g = 0x21`. Now
  masked with `g`'s high byte at packed bits [34:42), verified bit-identical
  over all 3,584 combinations. **Seventh mechanism selector packed into a
  reinterpreted config field, and the first to draw blood.**

Screens (1 process / 1 warmup / 1 timed, ~60–90 s) reproduce the ratchet at
6,703–6,728 µs and read ~1.2 points optimistic against the campaign. Measured
tonight over six repeats of the identical config: `ratio_vs_prod` σ = **0.52 %**,
`ratio_vs_pf6gm` σ = **1.34 %**. **Screens rank against `production` only** —
`pf6gm_mega` is too noisy at one warmup iteration to be a denominator.

## The budget, and why the queue is ordered the way it is

> **SUPERSEDED by exp_33** (see §"THE FIGURE NIGHT LANDED"). The table below has
> M6 ~2,588 ahead of M7 ~2,660 at screen resolution; measured at n=10 the order
> is **M7 2,701.8 ±23.7 > M6 2,453.3 ±2.5**, and the M6-first ordering this
> section justifies is therefore wrong. Kept for the reasoning, not the numbers.

Approximate phase costs at the ratchet (rank-max stamps, screens):

| phase | µs | note |
|---|---:|---|
| dispatch M0–M2 | ~1,193 | ~73 % own-work, not peer wait (exp_10 closed this axis) |
| plan M3–M5 | ~415 | **contains 448 MiB of provably dead stores** → exp_24 |
| **M6 (GEMM-1)** | **~2,588** | **K-loop stalls ~84 % of its cycles** → exp_26/27/28 |
| **M7 (GEMM-2)** | **~2,660** | +1,074 over the homogeneous baseline → exp_25 |
| combine M8/M9 | ~430 | already 3× better than homogeneous → exp_29 |

M6 and M7 together are 5,248 µs = **78 % of the kernel**. Everything below
targets one of those two, or deletes work outright.

### The three findings that reset the queue tonight

1. **Mode 12 never touches `part`** — not M7, not M8, not the service pool —
   yet M5 still zero-fills and scale-transposes all 32,768 × 7,168 × 2 B =
   **448 MiB of it every epoch**, inside a plan phase that only costs ~415 µs
   total, and streams it through a 256 MB Infinity Cache that M6 is about to
   want. (`CONTEXT/mode12_protocol_map.md` §8, `k0pf6gm_device_tile_mps.hip:1284-1288`.)
2. **There is no grid barrier between M6 and M7.** The dependency is already
   per-32-block via `a2_done[b]` counting to 8, and finer still: M6 chunk `g`
   produces exactly K128-group pair `(2g, 2g+1)` of M7's K-loop. What
   serialises the two phases is **CTA program order** — every CTA drains its
   whole M6 stripe first. `A2q` is capacity-sized and single-buffered so
   overlap needs no credit protocol, and both phases' LDS blocks are already
   summed into the single 155,428 B allocation, so interleaving costs **zero
   extra LDS**. (`CONTEXT/m6_m7_structure.md` §4.)
3. **M6's K-loop stalls ~84 % of its cycles** (~9,800 cycles per K-step against
   1,536 cycles of MFMA issue at G=3) with a software pipeline exactly **one
   K-step deep** — and its four `sched_group_barrier` hints are **hardcoded for
   `kGM = 1`** while we ship `kGM = 3`, so the scheduler is steered for a third
   of the MFMA, a third of the DS reads and a third of the DS writes.
   (`CONTEXT/m6_m7_structure.md` §5.3.)

### One prior belief retracted

**"M6 is CTA-insensitive (removing 24 % of CTAs costs 0.5 %)" was never
measured.** The reservation that produced that number happens at M6.9, *after*
the phase-1 body returns; phase 1's task loop is a hardcoded `task += kCTAs`
with `kCTAs = 256` and there is no role predicate between the M5 barrier and
the phase-1 call. All 256 CTAs have run all of M6 in every configuration ever
run. `exp_05` said so explicitly and used the invariance as its sanity check;
later documents re-read that control as a measurement. **No design may assume
M6 has idle CTAs.**

## The queue

> **SUPERSEDED.** This is the aug10-era queue; exp_24, exp_26, exp_27, exp_28,
> exp_25 and exp_29 have all since resolved. The live queue state is the table in
> §"THE FIGURE NIGHT LANDED" → "Queue state after tonight's figure block".

| # | experiment | targets | mechanism | risk |
|---|---|---|---|---|
| exp_24 | dead plan work + throttle-depth sweep | plan, M7 | delete the 448 MiB `part` zero-fill; sweep the epilogue's outstanding-RMW cap (8 was the first value ever tried and was worth ~500 µs) | low |
| exp_26 | phase-1 `sched_group_barrier` G-scaling | M6 | parameterize four hint counts by `kGM`; every expression evaluates to the existing literal at `kGM = 1`, so it is a parameterization, not a retune | low |
| exp_27 | `ascale_lds` prologue gather | M6 | 5,376 distinct cache lines fetched to deliver 21,504 B — 16× line amplification, 344 KB per task, sitting fully exposed between two `__syncthreads()`; ≈ 200 µs | medium |
| exp_25 | **M6/M7 CTA role split** | M6+M7 | the headline mechanism: partition the grid so some CTAs issue GEMM-1 MFMA while *different* CTAs run GEMM-2 and its remote-accumulate epilogue | high |
| exp_29 | **pipelined combine** | combine | the pool reduces and writes `out` rows as they become ready during M7, so M8 is empty | high |
| exp_28 | M6 task-order swizzle | M6 | 2,840 tasks share only 256 unique weight slices (10.9× reuse available); grid-stride assignment puts consecutive tasks 32 tiles apart | medium |

Ordering rationale: exp_24 and exp_26 are deletions and parameterizations —
highest information per unit of build risk, and both are single-variable A/B by
construction. exp_25 and exp_29 are the two on-mandate role-specialization
mechanisms and carry the largest prizes; both get a written protocol review
before their first GPU run.

## exp_25 design verdict — the M6/M7 split is gated on one unmeasured number

The design (`exp_25_m6m7_split/design.md`) reframed the mechanism honestly and
the reframing is worth carrying forward:

**A static role split cannot win by overlapping work.** Work is conserved, and
today M6 already gets all 256 CTAs while M7 gets 240. A split can only give M6
about 128. Under linear scaling the best balanced split is `(W6+W7)/240 =
5,421 µs` against today's `W6/256 + W7/240 = 5,248 µs` — a **173 µs loss**. Any
claimed win must name a term outside that model, and there are exactly three:

- **W1 — throttling the remote-RMW rate by cutting injectors. This is the whole
  case.** M7 in mode 12 costs 2,660 µs against 1,684 µs for the same GEMM
  without the remote-accumulate epilogue, so the epilogue surcharge is
  **976 µs**, and exp_21 proved it is rate-shaped. The split is a second,
  orthogonal throttle on the same axis: total outstanding remote RMWs =
  injectors × depth; exp_21 capped the depth, the split cuts injectors to
  0.47×. And M6 touches no fabric at all, so **xGMI sits 100 % idle for
  2,588 µs of every epoch** — that is the one genuine complementarity.
- **W2 — resource complementarity ≈ 0, plausibly −300 µs.** Both phases are the
  same fp8 K-loop with the same L2/LLC limiter (161.7 vs 158.1 FLOP/B, 17.1 %
  vs 13.7 % MFMA duty). Do not claim compute/bandwidth complementarity; the
  numbers do not support it.
- **W3 — tail elimination nets to ~zero** once the split's own start bubble is
  counted.

Central prediction **6,312 µs = 0.818×**, band 5,877–6,969. The band is
dominated by one coefficient: **M6's CTA scaling, which has never been
measured** (see the retraction above).

### F1 — the cheap gate that must run before the expensive build

Measure `T6(128) / T6(256)`.

| result | action |
|---|---|
| **≥ 1.90** | **stop — do not build the overlap arm.** Work-conservation loss cancels the whole prize. |
| ≤ 1.80 | build |

F1 needs only the `N2GM_P1_TASK_START` / `_STRIDE` hooks in the vendored
phase-1 body (still the donor's hardcoded `blockIdx.x` / `kCTAs` by default), so
it rides exp_26's file rather than creating a second writer. **`a6` is a
property of the tree, not the hardware** — exp_24 and exp_26 both move it, so F1
runs on the winning tree, not first in wall-clock order.

### The correctness landmine, recorded before anyone builds

**`a2_done`'s poll is vacuous today and this experiment makes it live for the
first time — and the campaign structurally cannot detect it failing.** M6's
payload release is a tid-0-only agent fence behind a bare `__syncthreads()`,
ordering 255 other threads' plain `uint4` stores that tid 0 never touched,
across eight non-coherent per-XCD L2s. Because the MoK harness feeds identical
input and routing every iteration, epoch `e−1`'s `A2q` bytes are **bit-identical**
to epoch `e`'s — so **a completely absent readiness edge returns the right
answer, passes every gate, and posts the best number in the sweep.** Signoff
conditions before any overlap arm is timed: use `producer_drain_release<agent>`
(a primitive we own and do not call), and add the DQ2-NaN-poison detector
(2.1 MB, unobservable in a correct run, trips the zero-nonfinite gate on a
premature read).

Runner-up, and the likeliest bug to actually ship: an off-by-one in the M7
pool's start/stride that covers a task **twice** doubles one 32×448 tile out of
16,720 — ≈6×10⁻⁵ relative error, `pperr = 0`, every gate green. Under-coverage
is loud; over-coverage is silent. Free detector: `part_done`, which mode 12
allocates, zeroes in M0, and never writes.

### Vendoring provenance — one trap to avoid

The authoritative donor is `solution/hip/n2_phase1_gm.cpp`, **585 lines**,
sha256 `1d90b26658b6a524db69434ddcfa0b2dcec2418c852f83874468d55dd0197dc2`. The
`exp_59`/`exp_63` snapshots match it; the **`exp_65` snapshot is a different
634-line fork** (`52ecd9e7…`) and must never be the vendoring source.

## exp_26 landed (`f9bfb4be`) — the vendored M6 body, and the hints are not a null

`n2_phase1_gm_mps.cpp` is **+112 / −0** against the donor: not one donor line
modified or deleted, the four hint lines per half now sitting unchanged in the
`#else` arm of a gate. Gate off is **`.text`-byte-identical** to the donor build
(sha `96049dfa…`, both 166,656 B, zero-line `llvm-objdump` diff; the ELFs differ
only in the HIP compilation-unit identity symbol, which hashes the TU path).

**The G-scaled hints change instruction placement substantially, and for the
better.** Per-iteration instruction mix is identical — only placement moved:

| | gate off | gate on |
|---|---|---|
| position of the loop's only `s_waitcnt vmcnt(0)` | **mfma = 0** (stalls with zero MFMA of that iteration issued to cover it) | **mfma = 48** (1,536 cycles of issue first) |
| barrier partition across the two `lds_cta_barrier()` | **33 / 49 / 14** | **48 / 48 / 0** |

So the wrong hints did not merely fail to help — they actively steered the
compiler into the worst placement available, putting the loop's only full VMEM
drain where nothing covers it and leaving the two LDS buffer phases unbalanced.
Cost of the fix: 3 extra `s_waitcnt lgkmcnt` per iteration against 13 fewer
instructions in the loop; LDS waits are ~170–200 cycles of a ~9,000-cycle
K-step, so a small debit.

**Ceiling, honestly bounded:** the mechanism moves ≤1,536 cycles of MFMA in
front of one drain inside an ~18,000–19,600-cycle iteration ⇒ ~8 % of M6 ≈
200 µs ≈ 2.6 % end-to-end, and only if that drain is fully exposed today.
Predicted **75–200 µs**.

### The catch that must be measured before this ships

Turning all four hints on migrates **96 extra scratch accesses into the M7
epilogue / M8 / M9 region** — scratch instructions 21 → 114, with 99 of the 114
landing after the phase-2 K-loop, up from 3. Scratch *bytes* actually improved
(144 → 128 B/lane, spills 16 → 14): fewer values spilled, accessed far more
often, in precisely the region that is mode 12's hot fabric path. A
phase-1-only hint change reached that far through whole-function register
allocation.

Consequences, both now in flight: the gate is being turned into a **4-bit mask**
(DS-read / VMEM / MFMA / DS-write) so we can find the subset that buys the
placement win without the scratch migration; and the first GPU arm is the
`timestamps=1` attribution pair, not an end-to-end number, because `ts_M6_us`
and `ts_combine_us` are predicted to move in **opposite** directions and one
total cannot separate them. If M6 drops and the combine rises by more, the
response is to chase the epilogue's register allocation — not to close the axis.

## exp_32 — the gate is now hard, and the ratchet survived it

**The detector fires on purpose.** Every candidate arm, every run:
`[POISON SELFTEST] arm=mps_mega one_row_poisoned_fails=True nonfinite=57344` —
and `57,344 = 7,168 × 8` **exactly**, the predicted arithmetic from one poisoned
row in an otherwise-correct buffer, restored bit-for-bit afterwards. That is a
quantitative self-test, not a boolean.

**The ratchet stayed green**: `survivors=0` at eager and post-timing,
`[MPS SOAK] 600/600 pperr=0 poison=0`, `control_fails=True`, and
`[MOK GATE] max_abs=0.035156 relative=0.008293` — **bit-for-bit the pre-patch
value**. Nothing at this config was resting on stale `out` data.

One mandatory fix on the way: `screen.sh` anchored `pass=` directly to `pperr=`
in the soak line, and the patch inserts `poison=` between them, so **every
poison-on run would have been reported `FAIL:soak=MALFORMED`** — a green run
read as red.

## exp_26 — CLEAN NEGATIVE. The census-matching theory of the hints is wrong.

Balanced 2×2, two independent batches, **n = 10 per cell, 40 runs**, all through
the full ladder with poison on, every arm verified by `.text` fingerprint.

| mask | DSR | MFMA | n | M6 mean ± sd | Δ vs 0 | t | p |
|---:|:---:|:---:|---:|---:|---:|---:|---:|
| **0** | 0 | 0 | 10 | **2,541.51 ± 4.77** | — | — | — |
| 1 | 1 | 0 | 10 | 2,544.32 ± 13.68 | +2.81 | +0.61 | 0.55 |
| 4 | 0 | 1 | 10 | 2,567.47 ± 20.47 | **+25.96** | +3.90 | 2.9e-03 |
| 5 | 1 | 1 | 10 | 2,582.42 ± 10.11 | **+40.91** | +11.58 | 3.7e-08 |

**MFMA-bit main effect = +32.03 ± 4.28 µs, t = +7.49, p = 3.7e-05.**
Pre-registered expectation was **−75 to −200 µs**; measured is **+32 µs — wrong
sign, seven sigma out**, replicated across two builds and four batches. Mask 4
clears 3σ *in the losing direction*, so no campaign was run and **the ratchet is
untouched at 0.8522×**. Shipping value returns to mask 0, which is
`.text`-identical to the donor.

**The compiler's own 34/48/14 split beats the "aligned" 48/48/0 one.** The two
extra LDS waits per iteration that `build.md` called "a small debit" are the
entire effect — over 512 iterations they are not small. **Matching
`sched_group_barrier` counts to the instruction census is not a valid theory of
scheduling on this loop.** That materially downgrades exp_31 (phase 2's hint is
over-scaled 29 against a measured 17) — it is the same class of change and now
has a measured counterexample.

**Bit 0's premise is dead on the current source, for free.** The K-loop's
`vmcnt(0)` drain is **already** at `mfma=48` in the baseline, so there was
nothing to buy; and the 96-access scratch migration is gone too (scratch
instructions 19 → 15, spills 14 → 12, `after_p2` unchanged at 3). exp_21/24
fixed it incidentally. `activate.md` §7.2 closes as already-done.

Also: the documented byte-identity hash `96049dfa…` / 166,656 B was **stale**
(exp_21/24 moved the source). Re-derived on current source: donor-include and
vendored mask-0 both hash `.text` to `ab0c353b…`, **179,904 B, identical**, and
it holds on the live JIT binary — so the activation is a proven no-op on
artifacts that actually ran.

### Two measurement-integrity findings worth more than the null

1. **`hsaco_before != hsaco_after` is NOT evidence of a code change.** A
   mid-batch "rebuild" differed in 36 of 188,744 bytes — embedded build paths —
   with identical `.text`. mori's key varies with which modules a process
   compiles. **Only the `==` direction is sound.** Fingerprint by `.text`.
2. **One batch ran the wrong arm and no gate noticed.** A `git push` was
   rejected by a concurrent push while the launch went ahead, so a batch tagged
   mask 5 measured mask 1. It was caught *only* by `.text` fingerprinting. Kept
   honestly as a second mask-1 replicate — which is what exposed that
   **single-batch t-statistics on this stamp overstate significance** (mask 1
   read +11.7 in one batch, −6.5 in the other). The launcher now self-syncs and
   refuses to start on a mismatch.
3. The **M6 stamp is ~5× better than planned**: σ = **4.8 µs**, not 23, and a
   control re-run 42 minutes later reproduced to **0.34 µs (p = 0.92)** — drift
   excluded, not assumed.
4. **A/B must be a compile-time `#define` in the hashed `.hip`, not `-D`** —
   mori hashes content, not flags, so a flag flip silently reuses the hsaco.

## exp_30 — the per-row readiness protocol is pure cost. Predicted ~6,215 µs.

**Q1 answered definitively: consumer early-start is NOT load-bearing.** Nothing
reduces a token before this rank's M7 is done, for two independent reasons:

1. **The drain is a de-facto local barrier.** Every CTA enters it
   (`k0pf6gm_device_tile_mps.hip:1440`) and it breaks only when a monotone
   ticket exhausts `events_total` (`moe_mps_adapter.cuh:748-755`) — with 1,024
   waves against ~16,720 events, **93.9 % of local M7 must complete first.** So
   the 130–210 µs finish spread is *already absorbed inside the drain*.
2. The parity arm publishes `row_ready` in bulk after a grid barrier
   (`:1573-1613`) — 19.6k stores carrying 8 bits. **Coarse readiness is already
   mode 0's semantics.**

So ~926k atomics per rank per epoch are buying an overlap that does not happen.
Replace them with **one grid barrier plus 8 publishes**:

| deleted | count |
|---|---:|
| `nc_arr` | 535,040 |
| `pushed` | ~314,000 |
| `row_ready` | ~19,600 |
| `row_rem` clean | ~19,600 |
| M8 polls | ~21,500 |
| `ev_next` | ~16,700 |
| **total** | **~926,000 — effectively the entire baseline** |

**The one structural catch:** `nc_arr` is not purely a readiness structure — it
is also the **push trigger** on mode 2's transport (`target = row_rem[r]`,
`adapter:789-822,921`), so deleting it there would expose the whole ~430 µs
push. **Mode 14 must therefore ride mode 12's remote-accumulate transport**,
where `part` is never written and the counting is bookkeeping only. That is a
real constraint on the build, not a detail.

**Net: +590…860 µs saved against 165…345 µs of costs (finish spread, skew,
barrier) ⇒ +245 to +695 µs ⇒ 5,990–6,440 µs, point estimate ~6,215.** The
0.80× target (6,172) sits at the optimistic edge of that band.

**And the service pool degenerates.** With bookkeeping, push and flags all gone,
mode 14 is a **homogeneous** megakernel. Per the standing mandate the number
gets **banked, not ratcheted as a role-split result** — and the degeneration is
itself a finding for the research question: on this workload, once the protocol
the pool existed to run is shown to be unnecessary, the role split has no
remaining job at this boundary.

~9 h across six stages. **The NaN poison is a blocking prerequisite** (this
change is precisely the "row not written" shape that identical per-iteration
inputs hide). Correction to an earlier note: the mode validator is at
`moe_mps_adapter.cuh:370`, not `:271`.

## exp_24 MEASURED — A is real (−91 µs end-to-end), B is closed

**Mechanism A (delete the 448 MiB dead `part` zero-fill): CONFIRMED.**

| population | n | plan M3→M5 mean | σ |
|---|---:|---:|---:|
| without A (`g` = 33, 289, 545, 801) | 7 | **428.96 µs** | 4.73 |
| with A (`g` = 97, 353) | 4 | **377.75 µs** | 7.56 |

**Δ = −51.2 µs on the plan phase, SE 4.18, t = 12.3.** End-to-end `m2→end` is
**−91.2 ± 33.6 µs** (t = 2.7) — *more* than the plan delta, consistent with
−51 µs of plan plus LLC-pollution relief elsewhere. Best composed screen point
`g = 353` reads **6,645.3 µs / 0.8516**. Resource tuple clean, scratch actually
improved 144 → 128 B/lane, MFMA census and atomic count unchanged.

Honest cross-check the agent ran and reported: 469,762,048 B removed in 51.2 µs
implies **9.2 TB/s**, which is above HBM peak — so those stores were never
costing a full HBM write-back (nothing reads them and the next epoch overwrites
them, so the LLC was absorbing much of the traffic). The deletion is worth
51 µs of plan time, not the 100–300 µs a naive bytes÷bandwidth model predicted.
A 5-rotation decision campaign on `g = 97` (A alone, single variable) is running.

**Mechanism B (throttle depth): the axis is CLOSED.**

| depth | n | M7 mean | Δ vs 8 | |
|---:|---:|---:|---:|---|
| 4 | 2 | 2,659.4 | −82.9 | t = 1.3 — **null** |
| **8 (shipped)** | 7 | **2,742.3** (σ 63.1) | — | |
| 16 | 1 | 3,204.4 | **+462.1** | 7.3σ |
| 32 | 1 | 3,112.2 | **+369.9** | 5.9σ |

**The throttle is not a smooth knob — its entire ~500 µs benefit is already
realized at depth ≤ 8, and 16/32 are catastrophically worse.** exp_21 picked the
right value first try. No further win on this axis.

### A correction that changes how every later screen is read

**End-to-end screen resolution is far worse than the 0.52 % measured earlier.**
All six `g = 33` control points tonight spread **6,795.1 → 7,243.7 µs = 6.6 %**.
A screen cannot rank anything under ~5 % end-to-end.

**But the phase stamps are excellent instruments**: plan M3→M5 has σ = 4.7 µs
(1.1 %) and M6 has σ ≈ 22.9 µs. That is exactly how a −51 µs effect became a
12σ result while being invisible end-to-end. **Rule for the rest of the night:
screen phase-local mechanisms on their phase stamp, and reserve end-to-end
numbers for campaigns.** This directly determines how exp_26's mask ladder
(a predicted 75–200 µs M6 effect ⇒ 3–9σ on the M6 stamp) and exp_27
(−130…−190 µs, also M6) get judged.

## exp_32 — the poison patch is ready; the S-1 load guard is KILLED

**S-1 killed, and the kill saved a regression.** The dead slot loads are
genuinely unpredicated (15 of 16 issue with no exec mask, no compare, no
branch), so the 154 MiB / 34 % figure is confirmed as *issued* volume — but
every dead load reads `base_slot(cur,0)`, **one 14,336 B rank-local region, 224
cache lines**, resident in L1/L2/LLC with zero xGMI. Revised prize **≈ 4 µs
(band 0–15) = 0.06 % end-to-end**, ten times below screen resolution. Worse, the
patch would likely *regress*: the unconditional issue **is** the load pipeline —
fifteen loads in flight drained by one `s_waitcnt vmcnt(0)` — and guarding
forbids the hoist, converting it into ~10.5 serialized issue-then-drain round
trips. No patch written.

Side finding worth carrying: `fanout[t]` is wave-uniform by construction but
reaches LLVM as a VGPR, so all 16 slot guards are EXEC-mask guards. A
`readfirstlane` makes them scalar branches and deletes 8–12 % of M8's executed
instructions — still only ~0.1 % end-to-end, so it should ride along with
another M8 edit rather than get its own campaign.

**The poison patch is mechanically generated and verified** (`git apply -p1`
clean against the node's exact files, patched Python passes `ast.parse`, patched
shell passes `bash -n`). Three **untimed** insertion sites — the eager per-arm
epoch, before every one of the 600 soak epochs, and one extra untimed
verification epoch after `_mok_rank_max` so the post-timing `[MOK GATE]` becomes
a real single-epoch coverage check. Nothing is inserted between timed
iterations: a 56 MiB fill there would displace 22 % of the Infinity Cache and
perturb the very inter-rank skew that the rank-max p50 measures.
`K0_MOK_POISON_OUT` defaults **on** and is forwarded through `run_campaign.sh`.

Three refinements to the original diagnosis:
1. `out` **is** cleared once per gate episode — the accurate statement is that
   it is never cleared *between* epochs, and the 600-epoch soak has no clear at
   all. Zero is not a usable poison anyway: one missing row of 4,096 moves the
   L1 ratio by 0.024 % against a 0.1 gate.
2. `correctness.py` gives a better hook than expected — a SUM-all-reduced
   `nonfinite == 0` on the candidate buffer, **an equality that cannot be
   dialled**, so one surviving NaN on any rank fails three independent ways.
3. **The existing negative control cannot catch staleness** — it is a
   store-over-a-peer bug in the frozen pull combine and never runs `mps_mega`.
   A host-only detector self-test was added, plus a kernel-side skip-a-row
   control for whoever owns the kernel next.

The patch also bundles a **required harness correctness fix**: the `[MOK GATE]`
loop runs after the eager loop has finished, so for every candidate arm it
re-reads whichever candidate ran last.

## exp_25 rev2 — the interleave's ceiling is +211 µs. DEMOTED to a wash.

I proposed promoting the M6/M7 interleave over the static split on the grounds
that the 976 µs epilogue surcharge is irreducible fabric time (392 MiB / 976 µs
= 421 GB/s = **78.3 % of the 537.6 GB/s ceiling** — both numbers confirmed
exactly, and the wire time for 392 MiB at 78 % efficiency is 980 µs, matching
the surcharge almost perfectly). **The arithmetic is right and it refutes the
conclusion I drew from it.**

**421 GB/s is not a utilization.** The 976 µs is the per-CTA *sum* of epilogue
phases, not a wall-clock window; a CTA is in an epilogue 36.7 % of the time
(41.2 of 112.4 µs per task). The bursts are **aligned by construction** — all
240 CTAs enter M7 within a few µs of each other and every M7 task costs the
same, so they march in lockstep. That alignment is exactly why exp_21's throttle
recovered 500 µs. But if the aligned bursts already drive the fabric at 78 % of
ceiling, then de-aligning them or hiding them under M6 recovers **at most the
78→100 % gap = +211 µs**, and zero at the efficiency the surcharge already
implies. **"Irreducible" is an argument against the mechanism, not for it:
exp_21 already took the reducible part.**

My four claims, adjudicated:

| claim | verdict |
|---|---|
| no work-conservation loss | **confirmed as stated, implication refuted.** `A6/256 + A7/240 = 5,248 µs` is exactly today's makespan, so the interleave is 173 µs better than the static split — but no penalty is a **tie at first order**, not a gain. Every µs must come from second-order terms. |
| F1 becomes irrelevant | **confirmed, with an unpriced cost.** The gate is genuinely gone — but F1's favourable branch was the *static split's* upside (up to −535 µs), and the interleave wins zero first-order regardless. **If M6 is CTA-insensitive, the static split beats the interleave.** Keep F1, demoted from gate to option-pricing. |
| zero register risk | **confirmed, better-founded than the rev-1 hedge** — `k0p6_dread` is a volatile load whose memory clobber exists precisely to stop descriptors being hoisted into kernel-long registers. New risk is **I-cache**: both MFMA bodies inlined into one hot loop against a 32 KB L1I shared by two CUs. Read it out of the resource report. |
| still on-mandate | **compliant but hollow.** The pool survives, so the arm is compliant — but under the interleave every compute CTA runs the same mixture, so M6/M7 is a *schedule*, not a role partition. It adds nothing to the mandate's research question; the static split does. |

**Stage 0 turned out to be an identity, and needed no build.** M6's round `m`
completes exactly tiles `32m…32m+31`, so readiness is linear in M6 progress; CTA
`bid`'s `i`-th M7 task needs tile `bid/16 + 15i`, giving available `2.133m`
against required `2.133` — **equal identically**, for any routing. Availability
is *not* the limiter, so the mechanism is not dead on arrival. But there is
**exactly zero slack**: the M7 front rides precisely on M6's production front,
every `a2_done` poll by a CTA slightly ahead of its peers is a real stall, and
the steady state is both phases finishing together at 5,248 µs — the
work-conservation answer, re-derived independently.

**L2 does not kill it; the fabric arithmetic demotes it.** Interleaved per-XCD
footprint is 6.7 MB against 4 MB (M6 improves 10.6 → 5.3, M7's resident 2.64 MB
`W2` is destroyed), but aggregate demand is 3.67 TB/s — 25 % *below* the
4.91 TB/s M6 already sustains — so the damage is latency-bounded at
**−0 to −222 µs**: same magnitude as W1's ceiling, opposite sign.

**Predicted 6,713 µs central (0.870×), band 6,481–6,944 — the 6,685 ratchet sits
inside the band.** Roughly 60/40 that `k = ∞` (do not interleave) wins the
sweep. Inverted falsifier worth keeping: **any `k` beating `k = ∞` by more than
211 µs exceeds the arithmetic ceiling and is a defect signature, not a result.**

Two coordination facts went our way: the interleave's protocol is bit-identical
to mode 12, so **no new mode number is needed** and none of the seven predicates
is touched (`k` lives in free config bits, `k = 0` ≡ today); and exp_26's
`f9bfb4be` already landed both the task start/stride hooks and the
epilogue-done hook where the `a2_done` fix belongs. Revised build **4–6 h + 2 h
GPU**, down from 14–19 h. Signoff conditions if it is ever built: the
`vmcnt(0)` hook ships in the same commit as the fused loop, and **the
DQ2-NaN-poison arm must be shown to fire on the hook-removed control before any
sweep number goes upward** — on this harness the most likely way this experiment
produces a headline number is by being broken.

**Decision: do not build the interleave tonight.** A predicted wash with a hard
+211 µs ceiling loses to exp_27 (−130…−190 µs, bit-exact gate, same 6 h) and to
exp_30's Stage 0 (~2 h to price a 700–1,000 µs mechanism). **F1 survives as a
cheap 2-screen measurement** — the task start/stride macros are already live, so
it now costs ~30 minutes and prices a −535 µs option.

## exp_29 design verdict — SAFE but readiness-limited; KILLED as specified

The pipelined combine is **provably safe** — the consume-and-zero proof survives
because pipelining moves *when* the zero happens, not *what gates it*, and each
slot row has exactly one writing rank so the flag is a complete gate. It is
still not worth building, for a reason that has nothing to do with correctness.

**The prize is readiness-limited, not capacity-limited.** A token becomes
reducible only when the **last of its 8 routed experts' M7 tiles** completes.
Top-k picks 8 distinct experts, a tile carries exactly one expert, and M7 walks
tile index roughly linearly, so `P(reducible by t) = (t/S)^8`. **The median
token is reducible with 8.3 % of M7 remaining; at M7's halfway point 0.4 % of
tokens are reducible.** No pool size, scheduling policy or primitive moves that
curve — it belongs to **M7's task order**, not to the combine.

Predicted `Δcombine = −112 µs` against `ΔM7 = +90 µs`, net **≈ −22 µs, band
−230 to +160** — straddling zero, and screens cannot adjudicate a 0.3 % effect
against a 0.52 % σ. Two sub-policies died on arithmetic on the way: pool-only
needs `C ≈ 43` to reach coverage, which costs +338 µs of M7 to buy at most
430 µs of combine; backlog-driven has nothing to tune, because the backlog is
~0 for 80 % of M7 and then floods.

One structural finding worth keeping: **the CTA that publishes `row_ready` is
usually not even on the same GPU** (≈19,089 peer stores vs ≈2,727 self stores),
and `tau` is nowhere on the producing rank, so a per-token counter bumped by the
publisher is *structurally impossible* — it would need a new `tau_table` plus
~21.5k system-scope remote RMWs injected into M7, i.e. exp_20's +624 µs
peer-write class.

### Two things exp_29 handed back that are worth more than its own mechanism

**S-1 — a one-line wave-uniform load guard, ~154 MiB of deleted loads.**
`KRN:547` issues the slot load unconditionally and guards only the accumulate at
`KRN:561`. `fanout[t]` is **wave-uniform** (every lane computes it at
`KRN:484-489`), so hoisting the guard deletes **34 % of the combine's read
instructions** — today out-of-fanout lanes re-read one 14,336 B region.
**Worth more than the entire pipelining mechanism, with none of its risk.**
Needs a 10-minute ISA check first in case LLVM already sinks the load.

**The real unlock — reorder M7 nc-major.** `task = nc·num_tiles + tile` turns
the readiness curve into a 16-step staircase with **15/16 of the combine
unblocked before M7 ends**, lifting the ceiling from ~17 % to ~85 %. That is
M-series **M8 / COMET layer-1**, and it is now `exp_30`.

## exp_28 — M6's intensity axis is CLOSED, and it is empty rather than expensive

Three independently measured walls, two of them new tonight:

| wall | constraint | how established |
|---|---|---|
| accumulator footprint | `M·N ≤ 59,392` | 256 AGPR; exp_65's durable finding that the wall is *accumulator footprint*, not tile width |
| **LDS ceiling** | **exactly 163,840 B** on gfx950 (the compiler rejects 163,841); fused law `1,092·M + 50,596` ⇒ **`M ≤ 103.5`** | five-point `N2GM_G` sweep, CPU-only, tonight — **this wall is independent of registers and nobody had recorded it** |
| `DQ2`/amax group | `N ∈ {256, 512, 1024}` | the 128-column quantization group |

Under all three, **the shipped `(96, 512)` tile is already the
intensity-maximizing legal shape.** `G=4` needs 190,372 B of LDS — 26,532 B over
the ceiling — on top of ~88 registers/lane.

### The finding with the widest blast radius

**Time is not proportional to request bytes.** exp_65's paired same-run points:
a 26 % byte cut bought 9.4 %; a +11 % byte increase cost +27 %. Measured
marginal rates:

| stream | cycles per byte | why |
|---|---:|---|
| weights (B) | **0.0826** | straight to registers |
| **activations (A)** | **0.407** | global→register→LDS→register with a loop-carried `vmcnt` and two CTA barriers |

**The A stream costs 4.9× per byte.** Any future traffic argument must be
weighted by stream, not counted in bytes. This retroactively explains exp_65's
`nc=16` kill (**+866.6 µs of A re-pass tax against −376.4 µs of G
amortization**, net 1.0023×) and it is now the central risk in exp_30.

### History corrected

- **`G=4` was never built and never timed.** It died on a *standalone* register
  probe; `exp_63/design.md:136` says literally "G=4: dead … Not attempted."
  `CLAUDE.md`'s "G3/chunk/K-split" shorthand had been read as a kill on the
  whole axis.
- **`G=3` was itself twice a build-gate kill** (512V/256A/7–9 spills, fused) —
  and shipped later and won. A build-gate kill is not a mechanism kill.
- **`nc=16` is a genuine measured kill** (exp_65: built, gated, timed).
- The `M=128, N=384` register-neutral variant: **the register arithmetic was
  right** — it is −8 registers, not merely neutral, at +18.7 % intensity — but
  it is refuted three other ways. `kChunks = 2048/192 = 10.67` is not an
  integer; LDS is 17,860 B over; and fatally, a 192-column chunk **straddles
  1.5 of M7's 128-column K-groups**, so the straddling group's `amax` would be
  computed from half its columns in each of two tasks on two CTAs and quantized
  by two different scales, while M7 applies one `DQ2[row][k]` scalar to all 128.
  Numerically wrong, not merely awkward.

### The recommendation: exp_27 is a deletion, and its gate is bit-exactness

M6's prologue gathers 96 × 56 FP32 scales from a group-major array with a
131,072 B stride — **5,376 distinct 64 B lines to deliver 21,504 B**, 344 KB per
task, 977 MB per epoch, sitting between two `__syncthreads()` where no MFMA can
cover it. **That layout is manufactured by us, in M5, from `sc_stage`, which is
already token-major, for a consumer set of exactly one: M6.**

So the fix is a **deletion, not a second transpose** — the one thing exp_27 must
not get wrong. Point M6 at `sc_stage` and delete M5's transpose. A token-major
row is 224 B and `224·t mod 64 ∈ {0, 32}` for every `t`, so a row always touches
exactly 4 lines: **384 lines against 5,376 — 14.0× exactly, 907 MB/epoch
deleted.** Three independent models agree: **−132 / −134 / −186 µs**, plus
≈ −17 µs in M5. Zero registers, zero LDS, no protocol.

`sc_dst[k·T_ext + t] ≡ sc_stage[t·56 + k]` **by construction**, so the output
must be **bit-identical** — a far stronger gate than any tolerance. ~6 build
hours. Note it composes with exp_24: once exp_24 deletes the `part`-zero half of
`zero_part_scale_transpose`, exp_27 deletes the other half and the whole loop
goes.

Also from exp_28: **O2 (task-order swizzle) is predicted 0** and should be
settled with one `rocprofv3` PMC pass rather than a build; and `num_tiles[0]` /
`nvi[0]` are unprinted, carrying **±4 % of uncertainty on every M6 number** —
a 0.5 h instrument worth adding before the M6 experiments are judged.

## THE MEASUREMENT-INTEGRITY BUG — applies to every number tonight

**`out` is never cleared between epochs, and the campaign feeds identical inputs
every iteration. So any bug whose signature is "this row didn't get written"
returns the previous epoch's bit-identical correct answer** — invisible to
`[MOK GATE]`, invisible to `combine_bit_exact`, invisible to 600 soak epochs. A
rare premature-ready is similarly invisible: one token short by one of ~5.25
addends is a 19 % error on that token, which dilutes to ~0.3 % against a 10 %
gate.

This is the same class as exp_25's `a2_done` landmine (identical per-iteration
input makes epoch `e−1`'s `A2q` bit-identical to epoch `e`'s, so a missing
readiness edge returns the right answer *and posts the best number in the
sweep*). Both say the gate ladder is weaker than it looks against
staleness-shaped bugs.

**Fix: poison `out` with NaN between iterations — one host-side line, no kernel
change.** It collapses the whole class, and the zero-nonfinite gate already
exists to catch it. This lands before any further mechanism is timed.

### exp_26 rev2 (`7a06fb56`) — the four hints are four different knobs

Builds turned out to be ~6 s, so all **16 masks** were enumerated rather than
sampled, and the effects factor cleanly. Bits are `DSW | MFMA | VMEM | DSR`:

| bit | effect | resource cost |
|---|---|---|
| **2 — MFMA** | **the entire 33/49/14 → 48/48/0 barrier realignment** | **none — every resource field byte-identical to the donor** |
| 0 — DS read | the drain move (`mfma 0 → 48`) **and all 96 migrated scratch accesses** | scratch instructions 21 → 114 |
| 1 — VMEM | **exact no-op** — all eight pairs differing only in bit 1 are byte-identical (phase 1's `16+kGM = 19` already matched the measured 19) | — |
| 3 — DS write | perturbs the bytes, moves no metric | — |

So the first report described **two independent effects on different bits as one
effect.** The drain move and the scratch migration are both pure functions of
bit 0 and 16 masks leave nowhere for them to come apart — but the question is
moot, because **bit 2 buys the barrier win for free.**

**Recommended arm: mask 4.** It is the only bit that changes the schedule while
leaving the resource profile byte-identical to the donor, which makes it a
genuinely single-variable arm: if the campaign moves, the barrier realignment
moved it. That also yields a clean 2×2 ladder — **mask 0** (control) / **4**
(barrier only, free) / **1** (drain only, +scratch) / **5** (both, +scratch).

**The migration is worse than "96 more spill accesses", and it is a known
regression.** A `-gline-tables-only` build (verified not to perturb codegen)
attributes all 96 to a **single source line**: `n2_phase2_gm_mps.cpp:145`, the
`peer_tab[xr >> maxtok_sh]` base load in the remote-atomic accumulate loop.
Across the kernel's 282 `flat_atomic_pk_add_bf16`, bit 0 puts a `scratch_load`
exactly **17 instructions ahead of 96 of them** — constant min = median = max,
i.e. one inlined shape — which is character-for-character the regression that
exp_21 restructured `throttle_plan` into four booleans to remove. Bit 0
re-triggers it *from the phase-1 side* by pushing whole-function pressure past
the same 256-VGPR ceiling. Note the direction: spilled **bytes go down**
(144 → 128 B/lane, 16 → 14 VGPRs) while **accesses go up** 21 → 114 — the
allocator evicted fewer values but picked the one read on the hottest path.
Follow-up worth taking: relieve one live VGPR in phase 2's epilogue and bit 0
becomes free. That needs both files under one owner.

**Mask 0 plus the two new task macros is `.text`-byte-identical to the donor
build** (`96049dfa…`, 166,656 B, identical resource tuple and K-loop timeline),
so `N2GM_P1_TASK_START`/`_STRIDE` compile to the donor's loop exactly and F1 is
unblocked. Diff is now +174 / −9 (a per-bit mask made the "wrap in `#else`"
shape untenable); the `−0` provenance claim is replaced by the stronger measured
one.

**Phase 2's hint, measured directly: 17 VMEM reads per half against a hint
asking for 29.** The one-line patch to `14 + kGM` preserves the `kGM = 1` anchor
for free (`8 + 7·1 = 15 = 14 + 1`), so it is a parameterization fix, not a
retune — same property that made phase 1's safe. Written up with line numbers
and a content match in `activate.md` §7.1 as **exp_31**.

### Two corrections to `CONTEXT/m6_m7_structure.md` from the ISA read

1. **§5.3 item 4 is wrong.** M6's K-loop *does* contain a compiler-inserted
   `s_waitcnt vmcnt(0)` — one per **iteration** (not per half), always a full
   drain, never a partial `vmcnt(N)`. The `N2_FORCE_VMCNT0` probe really is
   compiled out; the compiler inserts its own. This closes §7 open item 4.
2. **Phase 2's own `0x020` VMEM hint is over-scaled** — `8 + 7·kGM` = 29 at
   G=3 against a measured 17; the correct form is `14 + kGM`. So "phase 2
   scales three of its four correctly" is only 3/4 true. Phase 2 is M7 at
   ~2,660 µs, and this is the same class of defect just fixed in phase 1.
   Queued as a separate one-line experiment in `exp_26_p1_sched/activate.md`
   (the file belongs to exp_24 tonight).

## Standing rules in force

Every candidate: correctness + negative control + 600-epoch soak **before**
timing, no exceptions. One variable per arm. Same-run paired denominators only.
Sub-5 % deltas re-run. Bump `K0P6_MPS_SRC_REV` with any `.cuh`-adjacent edit and
confirm a new `.hsaco` mtime — directory mtimes are touched on a cache hit and
`latest/` is a symlink dir, so `stat` needs `-L`. The node checkout **is** the
arm. One GPU job at a time; `rocm-smi --showpids` is the check that works
(`pgrep -af torchrun` does not see our own job, which runs as
`python -m torch.distributed.run` inside the container).

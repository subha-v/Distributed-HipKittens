# exp_34 — mode 14 (coarse readiness): MEASURED. Rung **falsified**, and the pin carries a **ratchet regression** that must be arbitrated.

**Verdict, in order of importance.**

1. **BLOCKING, and it is not about mode 14.** Commit `291dfa08` — the mode-14
   commit, my own — **regresses the mode-12 ratchet by +726.9 µs**, entirely in
   M7. Same session, same config, same production denominator:
   rev 26 `f113d73f` = **6,497.3 µs**, the pin = **7,224.2 µs** (n=5). The mode-12
   source path is **byte-identical** between the two revisions and the throttle's
   four compile-time instantiations are still in the ISA, so this is a **codegen
   consequence of adding mode 14 to the same function**, and the resource-tuple
   gate did not catch it. **I am not permitted to edit these files; this needs a
   source owner.** Details in §1.
2. **The pre-registered band is FALSIFIED.** Mode 14 at `C=0` measures
   **6,650.9 µs** (stamps-off, n = 4), against a band of 5,990–6,440 and a
   declared falsification threshold of 6,568. It is **+153.6 µs worse than the
   true ratchet** measured in the same session, and **−573.3 µs better than the
   mode-12 control at the same pin**. Both comparisons are reported; neither is
   allowed to stand alone (§3).
3. **The two-rung decomposition is clean and one rung is worth zero.** Granularity
   alone (drain retained, `g=481`) = 6,647.7 µs (n=4); granularity + drain
   deletion (`g=353`) = 6,650.9 µs (n=4). **The per-task drain deletion is worth
   +3.2 µs — a null.** The coarse signal carries the entire effect (§4).
4. **The C sweep is NOT flat — the prediction failed, in the direction that
   strengthens the paper's placement claim.** C=0 → 6,650.9, C=8 → 6,725.6,
   C=16 → 6,780.4: monotone, **8.1–9.3 µs per reserved CTA**. Mode 14's service
   pool provably has *no job at all* (bit 26 never set in 4,032 readings,
   `DRAIN=0`), so this penalty is **pure CTA-capacity loss, with contention
   excluded by construction** (§5).
5. **The service pool degenerated exactly as predicted, and it is instrumented,
   not inferred** (§6).
6. **The protocol negative control failed in precisely the pre-registered way**:
   `pperr` bit 25 on **rank 7 only**, gate failing. The rendezvous is
   load-bearing (§2.4).

Everything below is stamps-labelled. `timestamps=1` is not free and no stamps-on
value is ever compared to a stamps-off one.

---

## 1. The blocking finding: the pin regresses mode 12 by +726.9 µs, all of it in M7

This was found because the brief required a **same-session mode-12 control**. That
control did not reproduce the ratchet: it came in at 7,223 µs, not ~6,490.
`production` (7,698) and `pf6gm_mega` (6,905) were simultaneously **identical** to
their historical values, so the session was not slow — the mps arm was.

### 1.1 Attribution: one commit, measured both ways in one session

The node checkout is the arm. I ran the ratchet config at rev 26 and at the pin,
back to back, same session, same seeds (`e34r26` / `e34d`, both 5 rotations,
500 warmup / 100 timed, soak 600):

| revision | cfg | mps p50 (stamps-off) | production | M7 stamp (ts-on) | planM6 (ts-on) | servicedrain |
|---|---|---:|---:|---:|---:|---:|
| rev 26 `f113d73f` | `C=16,g=353,mode=12` | **6,497.3** | 7,700.6 | 2,673.9 | 2,813.0 | 2,744.5 |
| pin `291dfa08` | `C=16,g=353,mode=12` | **7,224.2** (n=5) | 7,699.9 | 3,492.1 | 2,822.1 | 3,589.1 |
| **delta** | | **+726.9** | −0.7 | **+818.2** | **+9.1** | +844.6 |

The regression is **localised to M7** (+818.2 µs) with `planM6` flat to 9 µs. The
published ratchet (6,482.7 µs) reproduces at rev 26 to +14.6 µs, so the rev-26
number is a valid stand-in for it.

Repeatability at the pin — three different spellings of the same config plus
repeats, every campaign green:

| cfg spelling | mps p50 |
|---|---|
| `C=16,g=353,mode=12,flush_rows=16` (historical spelling, no `timestamps` token) | 7,220.1 |
| `…,timestamps=0` (n=4) | 7,222.2 / 7,224.2 / 7,225.4 / 7,224.6 |
| `…,timestamps=1` (n=2, median) | 7,229.9 |

An explicit `timestamps=0` token was a suspect (no historical row had ever used
one). It is **exonerated**: the historical spelling gives the same 7,220.

### 1.2 Mechanism: the injection throttle's *effect* is gone, its *code* is not

`g` is a bitfield. `g=353` = `0x161` = depth-selector 1 (`0x100`) | skip-dead-part-zero
(`0x40`) | **throttle-enable (`0x20`)** | physical g=1. `g=65` = `0x41` is exactly
`g=353` minus the throttle and its depth. So `g=353` vs `g=65` prices the exp_24
injection bound directly:

| revision | mode | stamps | throttled `g=353` | unthrottled `g=65` | throttle worth |
|---|---|---|---:|---:|---:|
| rev 26 | 12 | off | 6,497.3 | 7,110.8 *(historical, n=4)* | **−613.5 µs** |
| pin | 12 | off | 7,224.2 (n=5) | 7,226.0 | **−1.8 µs — inert** |
| pin | 14 | off | 6,650.9 (n=4) | 6,643.7 | **+7.2 µs — inert** |
| pin | 14 | on | 6,646.6 (n=2) | 6,652.0 | **−5.4 µs — inert** |

At the pin the injection bound buys nothing, **for either mode**. It is not
compiled out: the ISA census of the two revisions is identical on exactly the
four throttle instantiations —

```
f113d73f: total vmcnt=1221  vmcnt(0)=835   BOUNDED={1:2, 4:96, 8:96, 16:96, 32:96}
291dfa08: total vmcnt=1425  vmcnt(0)=1039  BOUNDED={1:2, 4:96, 8:96, 16:96, 32:96}
```

(The +204 `vmcnt(0)` are mode 14's own code.) `n2_phase2_gm_mps.cpp`, which holds
the throttle's target construction and dispatch, is **not in the commit's
diffstat** — only the `.hip` and the `.cuh` changed, and every mode-12 line in
them is byte-identical. So logic and dispatch are unchanged in source and the
throttle's *instructions* are present; what changed is scheduling/allocation
around them (the visible trace is `SGPR 104 → 106`, `LDS 155,428 → 155,496`).
**Precise mechanism: OPEN.** It needs someone who owns the source.

### 1.3 What this costs and what it means for the queue

- The ratchet config **does not reproduce at HEAD**. Any waterfall rung, placement
  figure or sensitivity point measured at this pin against a mode-12 denominator
  is measuring a broken transport.
- The **resource-tuple gate is not sufficient** to protect a timing invariant. It
  passed byte-for-byte on every gated field while the arm lost 11 % of its
  runtime. Every commit touching the shared kernel needs the ratchet config
  **re-timed**, not just re-gated.
- The clean question mode 14 was built to answer — *is coarse readiness additive
  with a working injection bound?* — **cannot be answered at this pin**, because
  the bound is inert here. It needs a re-run after the regression is fixed.

---

## 2. The ladder — all seven rungs, in order, all green

Pin `291dfa08`, `K0P6_MPS_SRC_REV 28`, `K0P6_MPS_ASCALE_TM 1`, fresh `.hsaco`
confirmed by `stat -L`.

### 2.1 Build / resource gate (rung 1)

`SGPR 106 / VGPR 256 / AGPR 256 / scratch 128 B per lane / LDS 155,496 /
occupancy 1 / MFMA 180 (96+84) / flat_atomic_pk_add_bf16 282`, **zero scratch ops
inside either MFMA span** — byte-identical to the pre-review build. (Against the
*ratchet* this is SGPR 104 → 106 and LDS +68 B; see §1.2 — that difference is now
implicated rather than cosmetic.)

### 2.2 First GPU run at `C=0` only, under `setsid` + `timeout` (rung 2)

`K0_MPS_CFG=C=0,g=353,mode=14,flush_rows=16,timestamps=1`.

### 2.3 World-8 correctness (rung 3)

```
[MOK GATE] mps_mega max_abs=0.039062 relative=0.008293 pass=True
```
with `mok_correctness.pass_all_ranks=True` in all eight rank JSONs.

### 2.4 Negative controls (rung 4) — both, and the protocol one is the important one

Harness control:
```
[MARK] control_fails=True
```

**Protocol negative control** (`R < world-1` publish loop, SRC_REV 1028, its own
tree via `DHK_ROOT` so the pin stayed clean). Required outcome: bit 25 on rank 7
only, gate failing. Measured, from the eight rank JSONs:

```
rank 0..6: pperr=0        bits=[]
rank 7   : pperr=33554432 bits=[25]      <- K0P6_MPS_ERR_M7DONE, rank 7 ONLY
[MOK GATE] mps_mega max_abs=0.039062 relative=nan pass=False
[POISON] eager arm=mps_mega rank=7 survivors=29360128 first_rows=[0..15]
```

Exactly the pre-registered signature: rank 7 never receives a cell, all 8 polls
time out, bit 25 fails closed, M8 is skipped there, its rows are never written and
the poison survives to fail the gate. **The rendezvous is load-bearing; the rung
is meaningful.**

*Note for whoever reads the raw log:* rank-0 stdout shows `pperr=0` and a `nan`
gate, which reads at a glance like the wrong failure. It is not — a rank-7-only
bit cannot appear in rank 0's print, and the gate metric is an all-ranks reduce.
The per-rank JSONs are the only place this control can be adjudicated.

### 2.5 Bit 26 never set (rung 5) — the positive check that the drain was skipped

**4,032 `pperr` readings across every mode-14 run tonight; the set of distinct
values is `{0}`; occurrences with bit 26 (`K0P6_MPS_ERR_SERVICE` = 67,108,864)
set: 0.**

### 2.6 600-epoch soak + poison (rung 6)

```
[MPS SOAK] completed=600/600 pperr=0 poison=0 poison_epoch=-1 pass=True
[POISON SELFTEST] arm=mps_mega one_row_poisoned_fails=True nonfinite=57344 relative=nan
[POISON] eager arm=mps_mega survivors=0
[POISON] post_timing arm=mps_mega survivors=0
```

Every campaign in every batch below carried the same four lines green;
`K0_MOK_POISON_OUT=1` and `K0_MOK_POISON_SELFTEST=1` in all of them.

---

## 3. The arm table — stamps-off and stamps-on kept apart

Campaign = 5 rotations, 500 warmup / 100 timed, `K0_MPS_SOAK_ITERS=600`, unique
`K0_MOK_OUTPUT_ROOT`, `production` in every run as the same-run denominator.
Arms were **interleaved** across four batches (`e34c1`, `e34d`, `e34e`, `e34f`),
never batched by arm. A campaign's value is the median over its 5 processes of the
rank-max p50; an arm's value is the median over its campaigns — **clustered by
RUN, never by (run, rank)**. Every campaign listed is gates-green with `pperr=0`
and soak 600/600.

### 3.1 Stamps-OFF — the headline numbers

| arm | rev | mode | C | g | n | p50 µs | individual campaigns | prod (same run) | ratio | Δ vs pin mode-12 | Δ vs true ratchet |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| **true ratchet (in-session)** | 26 | 12 | 16 | 353 | 1 | **6,497.3** | 6,497.3 | 7,700.6 | 0.8437 | −726.9 | — |
| mode-12 control **at the pin** | 28 | 12 | 16 | 353 | 5 | **7,224.2** | 7,222.2 / 7,224.2 / 7,220.1 / 7,225.4 / 7,224.6 | 7,699.9 | 0.9382 | — | +726.9 |
| mode 12 unthrottled (`g=65`) | 28 | 12 | 16 | 65 | 1 | **7,226.0** | 7,226.0 | 7,700.4 | 0.9384 | +1.8 | +728.7 |
| **mode 14, granularity + drain deletion** | 28 | 14 | 0 | 353 | 4 | **6,650.9** | 6,653.6 / 6,648.2 / 6,642.3 / 6,654.4 | 7,700.7 | 0.8637 | **−573.3** | **+153.6** |
| **mode 14, granularity alone (drain kept)** | 28 | 14 | 0 | 481 | 4 | **6,647.7** | 6,651.1 / 6,645.9 / 6,649.5 / 6,639.4 | 7,699.4 | 0.8634 | −576.5 | +150.4 |
| mode 14 unthrottled (`g=65`) | 28 | 14 | 0 | 65 | 1 | **6,643.7** | 6,643.7 | 7,704.9 | 0.8623 | −580.5 | +146.4 |
| mode 14, C=8 | 28 | 14 | 8 | 353 | 2 | **6,725.6** | 6,722.1 / 6,729.2 | 7,703.0 | 0.8731 | −498.6 | +228.3 |
| mode 14, C=16 | 28 | 14 | 16 | 353 | 2 | **6,780.4** | 6,782.1 / 6,778.7 | 7,702.1 | 0.8803 | −443.8 | +283.1 |

Campaign-to-campaign spread within an arm is ≈ 5 µs, so the 572 µs and 154 µs
deltas are ~100× and ~30× the noise; the C-sweep steps (71 / 131 µs) are ~13× and
~25×.

### 3.2 Stamps-ON — the phase story (never compared to the numbers above)

| arm | rev | mode | C | g | n | p50 µs | planM6 | M7 | combine | servicedrain |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| true ratchet (in-session) | 26 | 12 | 16 | 353 | 1 | 6,499.5 | 2,813.0 | **2,673.9** | 289.0 | 2,744.5 |
| mode-12 control at the pin | 28 | 12 | 16 | 353 | 2 | 7,229.9 | 2,822.1 | **3,492.1** | 297.9 | 3,589.1 |
| mode 12 unthrottled | 28 | 12 | 16 | 65 | 1 | 7,234.5 | 2,819.5 | **3,515.4** | 278.6 | 3,603.7 |
| mode 14, drain deleted | 28 | 14 | 0 | 353 | 2 | 6,646.6 | 2,807.0 | **2,861.4** | 385.6 | **unwritten** |
| mode 14, drain kept | 28 | 14 | 0 | 481 | 1 | 6,650.3 | 2,804.9 | **2,907.6** | 352.3 | **unwritten** |
| mode 14 unthrottled | 28 | 14 | 0 | 65 | 1 | 6,652.0 | 2,811.2 | **2,857.5** | 328.5 | **unwritten** |
| mode 14, C=8 | 28 | 14 | 8 | 353 | 1 | 6,729.8 | 2,812.0 | **3,021.8** | 287.7 | **unwritten** |
| mode 14, C=16 | 28 | 14 | 16 | 353 | 1 | 6,780.0 | 2,828.3 | **3,050.5** | 274.5 | **unwritten** |

Reading: mode 14's M7 (2,861.4) sits **between** the pin's broken mode-12 M7
(3,492.1) and the working rev-26 M7 (2,673.9). Mode 14 recovers **630.7 µs of the
818 µs** the regression put into M7, and its residual **+187.5 µs** against rev 26
is essentially the whole end-to-end deficit (+153.6). Placement inflates M7
specifically (C=8 → 3,021.8, C=16 → 3,050.5), while `planM6` is flat across every
arm at 2,805–2,828 — further evidence that everything tonight moved is in the
producer epilogue and nothing is in the GEMMs.

`servicedrain` is **not a duration** for mode 14 and is reported as unwritten, not
as a number: no CTA ever enters the drain, so the `K0P6_MPS_TS_DRAIN` cell is
never stored and reads as its never-written sentinel — `[MPS TS] … DRAIN=0`. That
absence is the evidence in §6.

---

## 4. Two-rung decomposition: the drain deletion is a null, granularity is everything

| rung | config | p50 (stamps-off) | n | rung delta |
|---|---|---:|---:|---:|
| (b) mode 12 at the pin | `C=16,g=353,mode=12` | 7,224.2 | 5 | — |
| (c) **granularity alone**, drain retained | `C=0,g=481,mode=14` | 6,647.7 | 4 | **−576.5** |
| (d) **+ drain deletion** | `C=0,g=353,mode=14` | 6,650.9 | 4 | **+3.2** |

The `kCoarseKeepDrainBit` selector did its job: the confound is separated, and the
answer is that **deleting ~2,840 `vmcnt(0)` + 2,840 `__syncthreads()` per CTA buys
nothing measurable** (+3.2 µs, inside the ±6 µs campaign spread — the sign is even
the wrong way round). The signal-granularity change carries 100 % of the effect.
As pre-registered, rung (c) prices coarse readiness *including* the deletion of the
event publication it makes meaningless — that part is not separable and is not
attributed to granularity alone.

Caveat that must travel with these two rungs: both are measured against a mode-12
baseline whose injection bound is inert (§1.2), so −576.5 µs is *not* the
granularity rung's value on top of a working ratchet. Against the working ratchet
the same arm is **+150.4 µs**.

## 5. The C sweep: not flat, monotone, and contention is excluded by construction

| C | p50 (stamps-off) | n | Δ vs C=0 | per reserved CTA |
|---:|---:|---:|---:|---:|
| 0 | 6,650.9 | 4 | — | — |
| 8 | 6,725.6 | 2 | +74.7 | +9.3 µs |
| 16 | 6,780.4 | 2 | +129.5 | +8.1 µs |

The pre-registered prediction was FLAT. **Falsified**, and it matters that it was:
in mode 14 the service pool has no work at all, so this is not the interference
mechanism exp_05/exp_20 measured — it is **pure loss of compute CTAs**, 8.1–9.3 µs
each. The stamps-on arms localise it: C=0 M7 2,861.4 → C=8 3,021.8 → C=16 3,050.5,
i.e. the penalty lands in the producer phase, which is exactly where losing CTAs
should hurt. That makes the placement figure's claim stronger, not weaker: reserving CTAs
costs even when the reserved pool provably does nothing, so no placement policy can
be rescued by giving the pool less to do. C=0 is reachable only in mode 14 (mode
12's validator rejects it), which is what this arm was for.

## 6. The service pool degenerated exactly as predicted — and it is instrumented

Three independent positive checks, not an inference:

1. **bit 26 never set**: 4,032 `pperr` readings, distinct values `{0}` (§2.5).
2. **`[MPS TS] … DRAIN=0`** in every mode-14 run: the drain stamp is never stored
   because no CTA enters the drain.
3. **`[MPS SPIN] chunk_poll success_max=0 fail_max=0 limit=2000000`**: the per-row
   chunk-poll instrument reads exactly zero — there are no per-row polls left.

So the number is **BANKED, not RATCHETED**, as pre-registered: at `C=0` with the
bookkeeping, the push and the flags all gone, mode 14 is a *homogeneous*
megakernel and cannot be a role-split ratchet however fast it is. What it buys the
paper is the denominator: **the per-row readiness protocol costs 573 µs against
the same (broken) transport — 7.4 % of production — while the injection bound it
was supposed to compose with is worth 613 µs on its own at rev 26.** Those two
numbers are suspiciously close, and the
hypothesis they suggest — that the injection bound and the coarse signal are
**substitutes bounding the same in-flight-write resource, not complements** — is
the single most interesting thing tonight produced. It is a hypothesis, not a
result: proving it requires the §1 regression fixed and then rung (d) re-run.

## 7. Data, provenance, reproduction

- `mode14_arms.json` — plot-ready, one record per arm keyed
  `rev{src_rev}_mode{mode}_C{C}_g{g}_ts{stamps}`, with `campaigns[]` (per-campaign
  and per-process values), `p50_median`, `production_p50_same_run`,
  `ratio_vs_production`, `delta_vs_mode12_control`, `delta_vs_ratchet_rev26`,
  `phase_stamps`, `g_decode`, `gates_green`, `n_campaigns`.
- Tags: `e34smoke` (ladder), `e34neg` (protocol control), `e34c1` / `e34e` /
  `e34f` (decision campaigns), `e34d` (throttle diagnostic), `e34r26` (rev-26
  attribution). Logs `~/overnight-scratch/<tag>_*.log`, CSVs
  `~/overnight-scratch/screen_<tag>.csv`, roots `~/k0-mok-<tag>/`.
- The rev-26 arm reset the node checkout and **restored the pin under a trap**;
  `git rev-parse HEAD` = `291dfa08` at exit, verified.
- ISA census: `~/e34/isa/{f113d73f,291dfa08}.s`, built CPU-only in `subha_k1`
  from `git archive` of each revision into a private scratch tree.

## 8. What I recommend, and what I could not do

1. **Arbitrate the §1 regression first.** It is worth 723 µs — more than every
   remaining optimisation in the queue combined — and it is live on `HEAD` right
   now, so exp_23/exp_35/exp_36/exp_37 will all measure it unless they pin behind
   it. I did not touch the source: another agent owns those files tonight.
2. **Re-run rung (d) after the fix**, to answer whether granularity composes with
   the injection bound or substitutes for it (§6).
3. **Add a timing gate to the resource gate.** The tuple passed while the arm lost
   11 %; the ratchet config must be re-timed on every commit to the shared kernel.

---
---

# Pre-GPU record (superseded by the sections above, retained verbatim)

The material below is the CPU-only gate record written before the lease was
granted. Its five protocol-review conditions and its build evidence still stand;
its "no result yet" framing does not.

## Status (as of the pre-GPU record)

| item | state |
|---|---|
| code | **complete and compiling**, 0 errors, in the local repo, **uncommitted** |
| resource gate | **GREEN on every named item** — see `build_log.md` |
| protocol review | **APPROVE-WITH-CONDITIONS**, and all five conditions are now cleared on CPU |
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

### 4. Protocol negative control — BUILT (now RUN; see §2.4 above)

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
drain and the service path were skipped.

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
pays it — and its publication loop runs zero iterations.

**One part genuinely cannot be separated, and must be labelled, not
attributed.** The per-task *event publication* cannot be retained under coarse
readiness: the event queue has no consumer once the per-row protocol is gone, and
publishing into it would both be dead work and make `K0P6_MPS_ERR_SERVICE`
ambiguous. So rung (c) prices "coarse readiness **including** the deletion of the
event publication it makes meaningless", and the figure must say that; only the
drain is split out.

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
   clean `pperr` on its peers, correctness gate failing.

## The resource gate, in one line

`SGPR 106 / VGPR 256 / AGPR 256 / scratch 128 B per lane / LDS 155,496 /
occupancy 1 / MFMA 180 (96+84) / flat_atomic_pk_add_bf16 282 / zero scratch ops
inside either MFMA span`.

**One caveat that is not a gate item.** The build re-triggers exp_26's scratch
migration into the remote-atomic epilogue: **96 of the 282
`flat_atomic_pk_add_bf16` acquire a scratch op within 40 instructions ahead,
against 0 in the base.** Seven different arrangements of the same mechanism
produced byte-identically the same allocator state, so this is a property of
total pressure in this function, not of where the mode-14 code sits. exp_26
measured this exact migration, bundled into its mask 1, at **+2.81 µs, t = 0.61 —
a null**. *(Read this again in light of §1: total-pressure effects in this
function are exactly the class of thing that produced the +723 µs mode-12
regression, and the tuple gate did not see either of them.)*

## Pre-registration, recorded before any run

- Band **5,990–6,440 µs**, point estimate **~6,215 µs**. Ratchet: `mps_mega`
  `C=16 g=353 mode=12 flush_rows=16` = **6,482.7 µs = 0.8408×** production.
- **A result above 6,568 µs falsifies the granularity rung** and is to be
  reported as such, not re-interpreted.
- Read the estimate pessimistically. exp_27's two terms came in **43 %** and
  **83 %** below their point estimates.
- **The number is BANKED, not RATCHETED.** With the bookkeeping, the push and the
  flags all deleted the service pool has no job left, so at `C = 0` mode 14 is a
  *homogeneous* megakernel and cannot be a role-split ratchet however fast it is.
- **Void, not a result**: any number from a build without `K0_MOK_POISON_OUT=1`,
  or with `K0P6_MPS_ERR_SERVICE` (bit 26) set, or from a run whose log lacks a
  `[MARK]` line or eight rank JSONs.

## What lands in LESSONS.md regardless of the eventual number

1. **You cannot add a phase to this megakernel for free.** The function sits on
   the 256-VGPR ceiling, and the marginal spill victim is the phase-2 epilogue's
   `peer_tab` base load. A separate M7.7 phase cost +16 B/lane of scratch purely
   by duplicating the existing release/barrier/acquire/`payload_ok` sequence.
2. **Register-allocation attribution on this kernel is non-decomposable.** The
   per-task hooks alone and the M7.5 changes alone each score 186/282 on the
   migration metric; together with everything else the score is 96/282. Report
   the cliff, not a per-piece cost.
3. **A `--genco` `.hsaco` is an offload bundle: `llvm-objdump -d` on it returns a
   one-line error, which an ISA census script will happily count as `v_mfma=0`.**
   Unbundle with `clang-offload-bundler --unbundle --type=o
   --targets=hipv4-amdgcn-amd-amdhsa--gfx950` first, then objdump.
4. **Attributing one instruction inside a 34,000-instruction inlined megakernel
   needs a marker or a differential, not reading.**
5. **This file has two textually identical `for (int R = 0; R < world; ++R) {
   if (R == cur) {` publish loops** — the parity per-row one and mode 14's
   rendezvous. Anchor on the body (`self_slot`), and diff the control.
6. **A design that says "`is_service_cta` must be false for this mode" needs the
   task stride checked in the same breath.** `k0p6_mps_stride` is `nct − C`
   unconditionally.
7. **Gating a barrier on `payload_ok` is a hang.** Barriers here may only be
   gated on grid-uniform descriptor decodes.

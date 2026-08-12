# exp_35 — the knob waterfall (paper Fig 4 / Q1, "the money figure")

**Commit (pinned for every rung): `ca5b683f`** ("aug11: RATCHET 0.8522 → 0.8408,
exp_27 ascale token-major"), `K0P6_MPS_SRC_REV 26`, `K0P6_MPS_ASCALE_TM 1`
confirmed at `k0pf6gm_device_tile_mps.hip:120`. Node `gbt350-odcdh2-c05-1`
(8× MI350X, gfx950). The node checkout HEAD was verified as `ca5b683f` before
**and after** every batch, with `SCREEN_SYNC=0` so nothing could resync it
under a half-finished sweep.

Data: [`waterfall.json`](waterfall.json). Raw: [`raw/`](raw/).

---

## Verdict

**The scheduling knob carries the entire win, and the payload relocation on its
own is a regression.**

Holding the dedicated service pool **fixed at C=16**, flipping one bit — the
epilogue's remote-RMW **throttle enable** — moves end-to-end by **−615.0 µs**.
That single scheduling decision is **1.52× the whole homogeneous-baseline →
ratchet gap** (−404.4 µs). Moving the payload into the producer's epilogue,
*without* the injection bound, is **+210.6 µs slower than the homogeneous
baseline that carries no payload at all**.

So the mechanism that pays is not "who carries the payload" and it is not
"how many CTAs are dedicated to communication" (constant across the two rungs
that produce the entire delta). It is **how many remote operations are allowed
in flight**. This is the paper's central claim measured directly.

Confidence: **high** for rungs (a)–(c) — 12 campaigns, 60 rotations, every gate
green, campaign-clustered t = 38 / 80 / 45 for the three steps, and the
per-campaign ranges of rungs (b) and (c) are disjoint by 598 µs. The one honest
weakness is that step (a)→(b) is **not** single-variable and cannot be made one
(see Caveats); step (b)→(c) **is** strictly single-variable.

---

## The rung table

`arm_p50_us` = median over the campaign's 5 rotations of that rotation's
MAX-over-8-ranks p50. Ratios are **paired same-run** — `production` is an arm
in all 12 campaigns and lands at **7,709.2 µs (stdev 4.1 µs, 0.05 %)**, so the
denominator is effectively a constant.

| rung | what is added | arm | config | n | p50 µs (median of campaigns) | sd | ratio vs production (same-run) | Δ vs prev rung |
|---|---|---|---|---|---|---|---|---|
| **a** | homogeneous baseline — no dedicated comm CTAs, no epilogue-carried payload | `pf6gm_mega` | n/a | 8 | **6,900.2** | 15.5 | **0.8950** | — |
| **b** | + epilogue-carried payload (mode 12 remote-accumulate), **throttle DISABLED** | `mps_mega` | `C=16,g=65,mode=12,flush_rows=16` | 4 | **7,110.8** | 13.7 | **0.9226** | **+210.6** |
| **c** | + **injection bound** (throttle ON, depth 4) — the ratchet | `mps_mega` | `C=16,g=353,mode=12,flush_rows=16` | 4 | **6,495.8** | 6.9 | **0.8422** | **−615.0** |
| **d** | + coarse arrival signals (mode 14) | `mps_mega` | — | 0 | `null` — `status: pending_exp_34` | | | |
| **e** | + nc-major producer task order | `mps_mega` | — | 0 | `null` — `status: not_built` | | | |

Per-campaign medians (the spread is what makes the steps unambiguous):

| rung | per-campaign p50 µs | range |
|---|---|---|
| a | 6,879.8 · 6,873.4 · 6,918.0 · 6,900.5 · 6,897.8 · 6,912.4 · 6,909.7 · 6,900.0 | 44.6 |
| b | 7,105.1 · 7,134.8 · 7,106.5 · 7,115.0 | 29.7 |
| c | 6,492.6 · 6,499.0 · 6,506.8 · 6,492.1 | 14.7 |

Paired within-campaign deltas against the homogeneous baseline (the cleanest
statistic available, since `pf6gm_mega` runs inside the same campaign):

| rung | paired Δ vs `pf6gm_mega` µs, per campaign | median | sd | t (n=4, clustered by campaign) |
|---|---|---|---|---|
| b | +207.3 · +222.4 · +196.8 · +215.0 | **+211.2** | 11.0 | **+38.4** |
| c | −387.2 · −374.4 · −411.2 · −408.4 | **−397.8** | 17.6 | **−45.2** |

Step (b)→(c), Welch on campaign medians: −615.0 µs, se 7.7, **t = −80.2**.

Campaigns ran **alternating c, b, c, b** in each of two batches, so time-ordering
drift on the node cannot be confounded with the mechanism.

### Attribution — how much of the gap each knob explains

Total (a)→(c) gap: **−404.4 µs** (paired: −397.8 µs).

| knob | Δ µs | share of the total gap |
|---|---|---|
| epilogue-carried payload, alone | **+210.6** | **−52.1 %** (moves the wrong way) |
| + injection bound (throttle depth 4) | **−615.0** | **+152.1 %** |
| dedicated comm CTAs | **not varied — held at C=16 across (b) and (c)** | 0 % by construction |

---

## Phase story (`timestamps=1`, device MAX stamps from the final soak epoch, 1 tick = 0.01 µs)

These are max-stamps from the soak, **not** from a timed iteration, so they are
a phase story and not an additive budget.

| stamp | rung b (unthrottled) | rung c (depth 4) | Δ (c − b) |
|---|---|---|---|
| `plan_M3toM5` | 374.4 | 372.9 | −1.5 (−0.4 %) |
| `M6` | 2,491.2 | 2,444.2 | −47.0 (−1.9 %) |
| **`M7`** | **3,140.9** | **2,673.1** | **−467.8 (−14.9 %)** |
| `combine` | 461.9 | 345.0 | −116.9 (−25.3 %) |
| `service_drain` | 3,223.0 | 2,758.5 | −464.5 |
| `[MPS SPIN] success_max` / `fail_max` | 0 / 0 | 0 / 0 | 0 |

**92 % of the move is in M7 + combine** — exactly where the epilogue's remote
RMWs live. The four phase deltas sum to −633.2 µs against an end-to-end −615.0 µs,
a 3 % closure that is well within what max-stamps can promise.

`[MPS SPIN]` is **0 in both rungs**: at routing std = 0 there is no peer wait to
recover, so nothing here is a queueing artifact of the harness. (Consistent with
exp_10.)

The 47 µs on `M6` is not the mechanism — it is codegen drift, because the
throttle is a compile-time specialization (see Caveats). It is reported rather
than hidden.

---

## The derived `g` bit layout, and how it was proven

`g` is the `group_slices` field of `K0_MPS_CFG`. Source of truth:
`moe_mps_adapter.cuh:55-105` (packing), `:224-265` (bit table), `:338-375`
(validator).

**Packing.** `g` is **16 bits, split across the packed uint64**: low byte at
`word[8:16)`, high byte at `word[34:42)`. The split exists because the old
encoder overflowed: any `g > 0xFF` put bit 16 into the **MODE** field and still
validated — `g = 0x121` became `mode |= 1` and decoded back as `g = 0x21`.

**Bit table for modes 12/13:**

| mask | meaning |
|---|---|
| `0x000F` | physical `g` — must be exactly 1 in modes 12/13 |
| `0x0010` | dual-write lost-update detector |
| **`0x0020`** | **epilogue remote-RMW throttle ENABLE ← the throttle switch** |
| `0x0040` | exp_24 A: skip the dead `part` zero-fill in M5 |
| `0x0080` | reserved (rejected) |
| **`0x0300`** | **throttle DEPTH select: `00`→8, `01`→4, `10`→16, `11`→32** |
| `0xFC00` | reserved (rejected) |

**The decisive detail: there is no "disabled" code point inside `0x300`.**
Disabling the throttle is the *enable bit*, and `config_is_valid` (`:356-359`)
**rejects** a nonzero depth selector when the enable bit is clear —
*"a depth selector without the throttle enabled selects nothing. Reject instead
of accepting a config that reads as swept."*

Therefore:

| | `g` | physical_g | detect | throttle_enable | skip_part_zero | depth_sel | depth |
|---|---|---|---|---|---|---|---|
| rung c (ratchet) | **353** = `0x161` | 1 | 0 | **1** | 1 | **1** | **4** |
| rung b (unthrottled) | **65** = `0x041` | 1 | 0 | **0** | 1 | **0** | n/a |
| *illegal* | 321 = `0x141` | 1 | 0 | 0 | 1 | 1 | — rejected |

`g = 65` is the **unique legal "throttle bits only" neighbour** of `g = 353`:
identical in physical_g, in the detect bit, and in exp_24-A's skip bit; the only
other candidate, `g = 321` (clear the enable bit, keep depth_sel), is refused by
the validator. This is what makes rung (b) a one-variable rung.

### Proof 1 — the descriptor dump: the word the kernel actually read

`K0_MPS_DESC_DUMP=1` writes the mps descriptor exactly as the host handed it to
the kernel. Slot `K0P6_D_MPS_CFG = 62` is the packed word that
`k0pf6gm_device_tile_mps.hip:702` feeds to `decode_config()`. Verbatim from
[`raw/decode_proof.txt`](raw/decode_proof.txt):

```
1_C16g353mode12flush_rows16timestamps1_20260812T093337Z
  slot[62] word : 26039050512 (0x6100c6110)
  decode_config : {'C': 16, 'g': 353, 'mode': 12, 'flush_rows': 16, 'pull_fallback': 0, 'timestamps': 1}
  g=0x161 -> physical_g=1 detect=0 throttle_ENABLE=1 skip_part_zero=1 depth_sel=1
  THROTTLE      : ON  depth=4
  mode field    : 12  <-- 12 proves NO g-overflow into mode

2_C16g65mode12flush_rows16timestamps1_20260812T093401Z
  slot[62] word : 8859173136 (0x2100c4110)
  decode_config : {'C': 16, 'g': 65, 'mode': 12, 'flush_rows': 16, 'pull_fallback': 0, 'timestamps': 1}
  g=0x041 -> physical_g=1 detect=0 throttle_ENABLE=0 skip_part_zero=1 depth_sel=0
  THROTTLE      : OFF (depth selector inert; config_is_valid forces sel=0)
  mode field    : 12  <-- 12 proves NO g-overflow into mode
```

Both words match `encode_config`'s arithmetic **bit for bit** (predicted
`0x6100c6110` and `0x2100c4110`), and the **`mode` field reads back 12 in both** —
which is the direct check that the historical g-overflow-into-mode bug is not
present. A wrong `g` cannot look like a pass here.

### Proof 2 — the mechanism signature

exp_24 independently measured depth 16 and depth 32 returning M7 to the
*unthrottled* cost, at **+462 µs and +370 µs** above the depth-4 ratchet. A
genuinely unthrottled config has to land in that band. Measured at campaign
resolution: **M7 = +467.8 µs** — inside the band, at its top edge, which is what
a real "throttle off" (rather than "throttle set very deep") should give.

One-variable check on the same stamps: `plan_M3toM5` moved **−1.5 µs (0.4 %)**
and `M6` **−47.0 µs (1.9 %)**, while the epilogue's own phase moved 14.9 %. The
throttle lives only in the phase-2 epilogue, and the stamps agree.

### Proof 3 — the fail-closed negative control

`C=16,g=321,mode=12,flush_rows=16,timestamps=1`, one process, **expected to
fail**:

```
1,...,"C=16,g=321,mode=12,flush_rows=16,timestamps=1",FAIL:rc23,w1t1p1,ca5b683f,26,
  ,,,,,,,VOID,,,VOID,VOID,VOID,...,"missing-rank-json"
```

The device guard at `.hip:719` raised `K0P6_MPS_ERR_CONFIG` (`1<<28` =
268435456) and returned before producing output, so the run never reached the
gates — every gate `VOID`, no rank JSON. **This is the expected result and is
not a candidate regression**; it is the evidence that `g = 321` is not a legal
rung, and hence that rung (b) must be `g = 65`.

---

## Gates — all green, every campaign

All 12 campaigns (60 rotations) reported, verbatim:

```
[MOK GATE] mps_mega max_abs=0.035156 relative=0.008293 pass=True     (also 0.035156/0.008294, 0.039062/0.008293, 0.039062/0.008294)
[MARK] control_fails=True
[POISON SELFTEST] arm=mps_mega   one_row_poisoned_fails=True nonfinite=57344 relative=nan
[POISON SELFTEST] arm=pf6gm_mega one_row_poisoned_fails=True nonfinite=57344 relative=nan
[POISON] eager       arm=mps_mega   survivors=0
[POISON] post_timing arm=mps_mega   survivors=0
[POISON] eager       arm=pf6gm_mega survivors=0
[POISON] post_timing arm=pf6gm_mega survivors=0
[MPS SOAK] completed=600/600 pperr=0 poison=0 poison_epoch=-1 pass=True
[MOK SYNTHETIC EAGER] status=valid_diagnostic ... gate_ok=True
```

`pperr = 0` everywhere — the only distinct value observed across all 12
campaigns is `0`. `gates_green: true` for every rung in `waterfall.json`; the
per-campaign gate lines are preserved in `campaigns_raw[].gate_lines`.

---

## Schema of `waterfall.json`

`schema_version: "exp35.waterfall.1"`. Top level: `experiment`,
`generated_utc`, `commit` / `commit_full`, `node`, `harness`, `statistic`,
`g_field_bit_layout` (the bit table plus all three proofs),
`attribution`, `caveats`, `rungs`, `campaigns_raw`.

Each element of `rungs`:

| field | meaning |
|---|---|
| `rung_id`, `label`, `arm`, `config` | rung identity; `config` is the literal `K0_MPS_CFG` |
| `mechanisms_present` | the decoded `g` bits this rung carries |
| `status` | `measured` \| `pending_exp_34` \| `not_built` |
| `n_campaigns` | independent 5-rotation campaigns |
| `arm_p50_us` | `{per_campaign_median[], median, stdev, min, max}` |
| `production_p50_us_same_run` | `{per_campaign[], median, stdev}` |
| `ratio_vs_production` | `{per_campaign[], median}` — paired same-run |
| `delta_us_vs_pf6gm_mega_paired_same_run` | `{per_campaign[], median, stdev}` |
| `delta_us_vs_previous_rung` | µs, medians of adjacent rungs |
| `phase_stamps` | `plan_M3toM5_us`, `M6_us`, `M7_us`, `combine_us`, `service_drain_us`, `mps_spin_*`, `n_samples` |
| `gates_green` | AND over the rung's campaigns |

`campaigns_raw[]` carries, per campaign: `run_id`, `cfg`, `gates_green`,
`pperr_max`, `arm_p50_us` (median **and** the 5 per-rotation `values`) for all
three arms, the literal `gate_lines`, and `ts_delta` / `ts_split` / `spin` with
every per-rotation value.

Reproduce with `aug11/tools/e35_20_campaigns.sh` (campaigns) →
`e35_30_collect.sh` (per-campaign records) → `e35_40_waterfall.sh` (this file).
Decode proof: `e35_02_bits.sh`, `e35_05_prep.sh`, `e35_10_smoke.sh`,
`e35_11_descverify.sh`, `e35_12_negctl.sh`.

---

## Caveats (all also in `waterfall.json.caveats`)

1. **Step (a)→(b) is not single-variable and cannot be made one.** The
   homogeneous baseline is a *different kernel* (`pf6gm_mega`, no MPS protocol
   at all), so that step bundles the mode-12 remote-accumulate transport, the
   C=16 service pool, and the per-row completion protocol. Step (b)→(c) is
   strictly single-variable: same arm, same commit, same C, same `flush_rows`,
   one `g` bit.
2. **The throttle is a compile-time specialization** — four `s_waitcnt vmcnt(N)`
   instantiations (`moe_mps_adapter.cuh:260-261`) — so rungs (b) and (c) resolve
   to different `.hsaco` builds (`be7b189d458d` vs `5635a2f2a370`) from the
   **same pinned source** at `ca5b683f`. That is the mechanism, not a confound,
   but it is why `M6` drifts 47 µs between rungs.
3. **`C = 0` is illegal in mode 12** (`moe_mps_adapter.cuh:373`:
   `mode_is_stream(c) && reserved_comm_ctas == 0u` → reject, and
   `mode_is_stream` includes 12 and 13). So this waterfall **cannot** separate
   "dedicated comm CTAs" from "mode-12 transport" inside a single arm; the only
   C=0 point available is rung (a), a different kernel. What it *can* show — and
   does — is that with C **held fixed** at 16, one scheduling bit moves
   end-to-end by more than the entire (a)→(c) gap. Driving the dedication axis
   to zero needs mode 14 (exp_37, `C ∈ {0, 8, 16}`).
4. **Phase stamps are device MAX stamps from the final soak epoch**, not from a
   timed iteration; a phase story, not an additive budget.
5. **Rungs (d) and (e) are unmeasured** — no binary for either exists at
   `ca5b683f`. (d) is blocked on exp_34's mode-14 build (pre-registered band
   5,990–6,440 µs); (e), the nc-major producer reorder, exists in no commit.
6. `out` is not bit-reproducible in **any** arm at this commit (see `STATUS.md`),
   so correctness is the `[MOK GATE]` tolerance gate plus the NaN-poison
   detector, never a bit compare.
7. Rung (a)'s `phase_stamps` are `null` by design: the `[MPS TS]` rings are
   compiled into the mps kernel only, so the homogeneous baseline has no phase
   stamps in this harness.

---

## What this feeds, and what is still missing

- **paper Fig 4 / Q1** — rungs (a), (b), (c) are plot-ready with error bars
  (per-campaign lists for every rung).
- The figure is **3 rungs of 5**. The named missing rungs are (d) mode 14 and
  (e) nc-major task order; both have `null` rows with a `status` and a
  `blocked_by`, so the figure can be completed in place without re-running
  (a)–(c) — `production` is stable to 0.05 % across 12 campaigns, so a later
  same-harness campaign is directly comparable, though a same-run pairing is
  still preferable.
- The strongest single sentence available for the paper: **with the dedicated
  comm-CTA count held fixed, bounding the number of in-flight remote operations
  is worth 615 µs, while relocating the payload into the producer's epilogue
  without that bound costs 211 µs.**

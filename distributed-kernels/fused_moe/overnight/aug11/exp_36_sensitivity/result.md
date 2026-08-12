# exp_36 — sensitivity sweep (paper Fig 7 / Q5): does the answer survive outside one shape?

**Node** `gbt350-odcdh2-c05-1` (8× MI350X, gfx950) · **pin** `b5215081`
(`K0P6_MPS_ASCALE_TM 1`, `K0P6_MPS_SRC_REV 26`, clean tree) ·
**arms** `production,pf6gm_mega,mps_mega` · 2026-08-12 11:05–12:36 UTC.
Data: `sensitivity_grid.json` (49 points) · raw in `raw/`.

---

## Verdict

**The pre-registered grid could not be run, and the reason is the finding.**
Two of its three axes do not exist in this harness at prefill:

1. **The routing-std axis is not a knob.** `K0_SYNTH_ROUTE` is rejected outright
 under `K0_INPUT_MODE=mok_synthetic` — which `run_campaign.sh` hard-wires and
 which `K0_BENCHMARK_PROTOCOL=mok_eager` requires. Reproduced on all 8 ranks.
 Even in the capture path the synthetic-route families are **decode-only**
 (`WORLD=8, T=64, TOPK=8, E=32`), and `synthetic_routes.py` exposes **no `std`
 parameter at all** — `skewed_hot` is a fixed deterministic pattern, so
 `std = 0.032` / `0.05` are not settable quantities here.
2. **The T axis collapses to a single feasible point.** `T ≤ 512` is refused by a
 host guard. `T = 1024` and `T = 2048` run, but **both megakernel arms produce
 wrong output** — `mps_mega` leaves 7 of 8 ranks' outputs *entirely unwritten*
 (poison survivors = T·H per rank) while `production` passes — so no timing
 from those shapes is admissible. `T = 4096` is the only shape where the kernel
 is correct. This is not an MPS bug: `pf6gm_mega` is wrong there too.

At the one shape that is real, the sweep was run properly and answers the
question the figure is actually about — **which knob the megakernel is sensitive
to.** Fifteen 5-rotation campaigns, all gates green:

**the throttle is worth 10×—30× more than placement, and placement never wins.**
Removing the injection bound costs **+627 µs (0.8427 → 0.9234× production)**.
Every move along the CTA-dedication axis is a *loss*: `C=32` +221 µs, the
dedicated-pool arm (mode 2, `C=64`) +332 µs. The axis is monotone the other
way — **`C=8` beats the `C=16` ratchet by 28.6 µs (0.8388 vs 0.8427)**, three
non-overlapping replicates each, and `C=4` matches `C=8`. The phase stamps put
the whole effect in **M7**, the payload-carrying phase; **M6 is flat to ±1.5 %**
across every arm. Peer wait is **zero** (`[MPS SPIN] 0/0`) at every single point.

---

## What was measurable, and how

| requested axis | reachable? | evidence |
|---|---|---|
| `T = 512` | **no** — refused | `ab.py:554` `RuntimeError: pf3/pf4h/pf5/pf6 arms are prefill-only; got T=512` (guard is `T <= 512`), 6/6 configs |
| `T = 1024` | **no** — wrong output | `mps_mega` 7/8 ranks 7,340,032 poison survivors (= T·H); `pf6gm_mega` `max_abs=1.71 relative=0.846`; `production max_abs=0.023 pass=True`; `pperr=0`. 6/6 configs with capacities pinned at T=4096, and again with capacities scaled to T |
| `T = 2048` | **no** — wrong output | identical shape of failure, 14,680,064 survivors, `pf6gm_mega max_abs=1.81` |
| `T = 4096` | **yes** | 6 screens + 13 campaigns, all gates green |
| `route std = 0` | **yes** (as native) | measured destination-load CV **0.00469** (seed 1234) / **0.00343** (seed 2468) |
| `route std = 0.032`, `0.05` | **no** — axis absent | `ValueError: K0_SYNTH_ROUTE cannot be combined with K0_INPUT_MODE=mok_synthetic` on all 8 ranks |
| `C = 0` at std 0.05 | **no** — needs mode 14 | validator rejects `C=0` in mode 12 (`moe_mps_adapter.cuh:373/375`); left as `"status": "requires_mode_14"`, not faked with a large-C proxy |

**Substitute axis actually swept.** With skew unavailable I swept the one routing
variable that exists: the router-logit seed (`K0_MOK_SEED_BASE` 1234 → 2468),
which changes the *realised* route. This is a weak axis by construction — the
reachable destination-load CV spans **0.0034 – 0.0086**, i.e. 4–15× *less*
imbalance than COMET's production skew (0.032) — and it is reported as such, not
as a skew substitute.

**Driver deviation (auditable).** `K0_T` and the four other shape variables are
hard-wired as `docker -e` flags in `run_campaign.sh`, which cannot be overridden
from outside. Rather than edit the harness I generated `$HOME/e36/rc_T.sh` from
it **by sed**; the complete diff is in `raw/rc_T.sh.diff` and touches only those
five `-e` lines plus a `SCRIPT_DIR` override. At `T=4096` the generated defaults
reproduce the hard-wired values **exactly** (`T_LOC_MAX = 10·T`,
`PADMAX = WORLD·T·TOPK + 31·E`, `MAXTOK = MAXTOK_PROD = T`), and the control
screen through the copied driver read **0.8420×** against the ratchet's known
0.8408× — the driver is not a confound.

---

## The grid

### Campaigns — T = 4096, seed 1234, 5 processes, 500 warmup / 100 timed

`ratio` is same-run `mps_mega / production`. Stamps are device stamps from the
final soak epoch (µs); they describe phase *shape*, not the timed iteration.

| C | g | mode | throttle | n | p50 µs (mean) | replicates | ratio (mean) | Δ vs C=16 d4 | M6 | M7 | combine | spin |
|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| 8 | 353 | 12 | depth 4 | 3 | **6469.5** | 6464.7 / 6472.3 / 6471.5 | **0.8388** | **−28.6** | 2453 | 2683 | 306 | 0/0 |
| 4 | 353 | 12 | depth 4 | 1 | 6466.1 | — | 0.8395 | −32.0 | 2436 | 2683 | 342 | 0/0 |
| 16 | 353 | 12 | depth 4 | 3 | 6498.1 | 6488.7 / 6504.6 / 6500.8 | 0.8427 | — (ratchet) | 2451 | 2710 | 308 | 0/0 |
| 16 | 97 | 12 | depth 8 | 2 | 6570.8 | 6562.6 / 6579.1 | 0.8524 | +72.7 | 2467 | 2822 | 298 | 0/0 |
| 32 | 353 | 12 | depth 4 | 1 | 6719.3 | — | 0.8705 | +221.2 | 2476 | 3003 | 176 | 0/0 |
| 64 | 1 | 2 | n/a | 1 | 6829.6 | — | 0.8841 | +331.5 | 2483 | 2935 | 408 | 0/0 |
| 16 | 65 | 12 | **off** | 1 | 7125.0 | — | 0.9234 | +626.9 | 2512 | 3296 | 352 | 0/0 |

### Campaigns — T = 4096, seed 2468 (the routing-draw replicate)

| C | g | mode | throttle | p50 µs | ratio | Δ ratio vs seed 1234 | M7 | spin |
|---:|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 353 | 12 | depth 4 | 6520.9 | 0.8444 | +0.0017 | 2647 | 0/0 |
| 64 | 1 | 2 | n/a | 6818.9 | 0.8818 | −0.0023 | 2865 | 0/0 |
| 16 | 65 | 12 | off | 7194.2 | 0.9312 | +0.0078 | 3134 | 0/0 |

Ranking is identical under both routing draws, and every gap is far larger than
the between-draw movement.

### Screens — T = 4096 (ranking only, 1 σ on ratio = 0.52 %, smallest callable delta 2 %)

| config | seed 1234 ratio | seed 2468 ratio |
|---|---:|---:|
| `C=16,g=353` (depth 4) | 0.8373 | 0.8343 |
| `C=16,g=97` (depth 8) | 0.8396 | — |
| `C=8,g=353` | 0.8559 | — |
| `C=32,g=353` | 0.8630 | — |
| `C=64,g=1,mode=2` | 0.8935 | 0.9095 |
| `C=16,g=65` (throttle off) | 0.9195 | 0.9163 |

Note the screens got the C=8 point *wrong* (0.8559 at screen, 0.8388 over three
campaigns): at 1 warmup / 1 timed iteration the small-pool arm is the one point
where screen and campaign disagree by more than the stated screen band. Screens
ranked the big effects correctly and nothing else.

### Failed / unreachable rows (in the JSON, with the verbatim error)

| T | route | status | why |
|---:|---|---|---|
| 512 | native | `failed` | host guard, before any launch |
| 1024 | native | `failed` | megakernel arms wrong; `production` passes |
| 2048 | native | `failed` | same |
| any | std 0.032 / 0.05 | `unreachable_by_harness` | axis does not exist at prefill |
| 4096 | native, `C=0` | `requires_mode_14` | `C=0` illegal in mode 12 |

---

## Schema — `sensitivity_grid.json`

`schema_version` `exp36.1`; top level carries `generated_utc`, `commit`, `node`,
`harness`, `driver`, `units`, `caveats`, `points[]`. Each point:

`T`, `route_family`, `route_seed_base`, `route_std_requested`,
`route_std_measured` (destination-load CV), `route_dest_load[8]`, `config`, `C`,
`g`, `mode`, `flush_rows`, `throttle_enabled`, `throttle_depth`,
`capacity_mode`, `shape{T,T_LOC_MAX,PADMAX,MAXTOK,MAXTOK_PROD}`, `status`
(`ok` | `failed` | `unreachable_by_harness` | `requires_mode_14`), `error`,
`measurement` (`screen` | `campaign`), `n_campaigns`, `n_processes`, `p50_us`,
`production_p50_us_same_run`, `pf6gm_p50_us_same_run`, `ratio_vs_production`,
`ratio_vs_pf6gm`, `spin_success_max`, `spin_fail_max`,
`phase_stamps{plan,M6,M7,combine,planM6,servicedrain,m2_to_end}` (µs),
`gates_green`, `gates{...}` (every gate line re-parsed from the run log), and
`raw{tag,outdir,log}` back to the evidence.

`p50_us` is `summary.json → arm_p50_us[arm].median` (median over processes of the
rank-max p50). Phase stamps come from the **final soak epoch** and do not sum to
`p50_us`.

---

## The four pre-registered predictions, judged

**1. "Small T favours coarse signals and fusion; large T favours the throttle and
task order." — UNRESOLVED (cannot be tested at this pin).**
There is no second T. `T ≤ 512` is refused and `T ∈ {1024, 2048}` returns wrong
answers on both megakernel arms. The prediction is neither supported nor
refuted; what the night *did* establish is a prerequisite nobody had checked:
**the megakernel family is correct only at `T = 4096`.** Until that is fixed the
batch-size question cannot be asked at all, and any claim about small-batch
behaviour in the paper would be unfounded.

**2. "Skew is the ONE regime where a dedicated service pool may re-enter."
— UNRESOLVED as stated (no skew), and REFUTED in the direction it can be tested.**
At the reachable imbalance (CV 0.0034–0.0086) placement never wins, and it does
not merely fail to win — it is **monotonically harmful upward**:
`C=4 ≈ C=8 < C=16 < C=32 < C=64(mode 2)`, spanning 6466 → 6830 µs. The
CTA-dedication axis points *away* from dedication as far as it can legally be
driven (`C=0` needs mode 14). If anything this **pre-registers a prediction for
exp_34**: mode 14's `C=0` rung should land at or below 6466 µs on this trend.

**3. "Skew de-coalesces epilogue traffic, so the injection bound gets *more*
valuable; test depth 4 vs depth 8 at the skewed corners." — the skew half is
unresolved; the depth half is SUPPORTED at the balanced corner.**
Depth 4 beats depth 8 by **72.7 µs (0.8427 vs 0.8524)**, ordered the same way in
both replicate pairs (6488.7 < 6562.6 and 6504.6 < 6579.1), and the difference is
localised in M7 (2710 → 2822 µs). Turning the throttle off entirely costs
**626.9 µs**. The optimal depth did not shift; there was no skew to shift it.

**4. "At std = 0 dispatch has zero peer wait (`[MPS SPIN]` 0/0)" — SUPPORTED,
re-tested not assumed.**
`chunk_poll success_max / fail_max` is `0/0` at **every one of the 25 green
points**, except four where `success_max = 1` and `fail_max = 0` — a single
successful poll, still zero *waiting*. Reproduced across two routing seeds, four
values of C, three throttle settings and both modes. exp_10 and exp_33 hold.

---

## "Does placement ever win under skew?"

**Under skew: unanswerable here** — the regime cannot be created at prefill with
this harness. **Under everything reachable: no, and not close.** Across 15
campaigns the dedicated-pool arm (mode 2, `C=64`) is 332 µs *worse* than the
`C=16` ratchet and 364 µs worse than `C=8`, and the loss grows monotonically with
pool size. The mechanism is visible in the stamps: M6 (the GEMM) is flat within
1.5 % across every arm, so the CTAs handed to the service pool are taken from
work that had nothing to gain, while M7 — the phase that carries the payload —
absorbs the entire difference.

---

## Confidence

- **High** on the two refusals: both are source-level guards *and* were
 reproduced on the node on all 8 ranks (`raw/refusal_evidence.txt`).
- **High** on the throttle result (627 µs, 15× the campaign replicate spread) and
 on placement being harmful upward (221–332 µs, 8–12×).
- **High** on depth 4 > depth 8 (72.7 µs, ordered in both paired replicates).
- **Medium** on `C=8 > C=16` — 28.6 µs (0.44 %) is a sub-5 % delta, so it is
 reported as a *candidate*, not a ratchet move. Support: three campaigns each,
 completely non-overlapping (`max C=8` 6472.3 < `min C=16` 6488.7; one-sided
 exact rank test p = 0.05), the same ordering in the M7 stamp, and `C=4`
 landing in the `C=8` band. **I did not move the ratchet** — that is the
 ratchet owner's call, and a paired A/B/A confirm is the right next step.
- **Low/none** on anything about batch size or skew. Those rows are empty on
 purpose.

## What would unblock the real Fig 7

1. Make the megakernel correct at `T ≠ 4096` (both arms fail, so it is in the
 shared tile/segment math, not in the MPS transport). Until then the paper
 should not claim a batch-size sweep.
2. Give `mok_synthetic` a route-skew knob — the generator draws router logits
 from `torch.randn` and takes top-k, so a temperature/bias parameter on the
 logits would produce a *continuous* std axis, which is strictly better than
 porting the decode-only fixed `skewed_hot` pattern.
3. mode 14 lands the `C=0` rung, which this sweep's monotone trend predicts.

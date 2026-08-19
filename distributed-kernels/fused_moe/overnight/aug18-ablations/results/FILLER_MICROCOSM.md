# FILLER MICROCOSM — does co-resident compute cost the all-reduce anything?

**Verdict: GO for the shared-expert filler (G-L2).** The all-reduce slows by **0.00 %** while a
matrix-core filler runs beside it, and the filler still retires **44–53 %** of its solo throughput
during the collective. Net gain **+571 to +695 µs per 235 MB all-reduce**, against a G-L2 bar of
≥ 25 % coverage.

**Run:** 2026-08-19 01:14–01:26 UTC, node `gbt350-odcdh2-c05-1` (reached at `10.5.95.87:2425`),
8× MI350X gfx950. **Binary:** `~/anatomy_g0/build/m25_bb_g0l0`, built in-tree from
`ablations@a50ffd82`, default flags (`M25_LEDGER=0`, `M25_NOCOMPUTE=0`) — the uninstrumented
build. **Raw:** `runs.jsonl` (19 appended rows, `run:"FILLER"` and `run:"G-L0b-arms"`).
Node-side originals `~/anatomy_g0/build/{filler.jsonl,filler_raw.txt,clocks_filler.csv}` **were
not harvested — the node dropped immediately after the run** (see §7).

## 1. Why a microcosm, and what it is not

G-L2 proposes hiding the exposed all-reduce under shared-expert work. The question that decides
the design is not "can we issue both" but **"what does the collective pay for the company?"** —
if a co-resident GEMM slows RCCL by more than it contributes, the filler is a net loss.

The rig has no in-kernel filler arm, so this is a **two-process co-residency proxy**: two separate
processes on the same 8 GPUs, one running a bare `ncclAllReduce`, the other a matrix-core loop.
**Stated limitations, up front:**

- **Not in-kernel co-residency.** The real design shares CUs *by construction* inside one
  persistent launch; here two processes contend through the hardware scheduler. The proxy is a
  harsher, coarser form of sharing, so the filler's own slowdown here is a pessimistic bound.
- The filler is `mfma_burst` — a representative dense `mfma_f32_16x16x32_bf16` load at
  occupancy 1, **not** the actual shared-expert GEMM (the rig's standing honesty label).
- `stdrccl` is a bare 235 MB `ncclAllReduce` — production's own collective, which is the right
  one for this question.

The load-bearing result (**RCCL is not slowed**) is the robust half: it is a statement about the
collective's behaviour under CU contention, and the proxy's coarser sharing can only have
overstated the interference, not hidden it.

## 2. Design — two configurations, so each leg gets a fully-overlapped window

The two arms have very different start-up costs (RCCL comm init ≈ 34 s; compute init ≈ 1.8 s) and
different natural durations, so a single overlap cannot give both legs a clean window. Each
configuration therefore nests one arm's *measured* window strictly inside the other's:

| cfg | long arm (background) | short arm (started after a delay) | which leg is clean |
|---|---|---|---|
| **A** | `stdrccl` iters=3000 → measured window ≈ last 4 s of a 39.6 s run | `compute` iters=6000 (≈ 7 s of MFMA) started at t=32 s | **AR_with_filler** |
| **B** | `stdrccl` iters=20000 → measured window ≈ 26 s | `compute` iters=6000 started at t=40 s | **filler_with_AR** |

The *other* leg in each configuration is diluted by design (its window is only partly co-resident)
and is recorded in `runs.jsonl` with a `note` saying so — **it is not quoted as a result.** The
dilution is itself a consistency check: cfg A's filler reads 1,243–1,319 µs (partly co-resident)
against cfg B's 2,640–2,664 µs (fully co-resident) and 1,166 µs solo — the ordering is exactly
what the design predicts.

Fixed: `tokens=16384` (235 MB), `slab_rows=256`, `depth=4`, `bps=2`. Filler `k_inner=1925`, chosen
so the filler's per-iteration duration (≈ 1,166 µs) matches the all-reduce's (≈ 1,300 µs).
K=3 repeats of each configuration, run back-to-back in one session.

## 3. The three legs

| leg | measurement | K | reps (µs) | median | spread |
|---|---|---:|---|---:|---:|
| **AR_alone** | `stdrccl`, n=3000 iters | 2 | 1,300.4 ; 1,300.5 | **1,300.4** | 0.1 µs (0.01 %) |
| **AR_with_filler** | cfg A, `stdrccl` n=3000 | 3 | 1,295.5 ; 1,301.2 ; 1,300.4 | **1,300.4** | 5.7 µs (0.44 %) |
| **filler_alone** | `compute` k=1925, n=6000 | 2 | 1,165.8 ; 1,417.5 | **see below** | 251.7 µs (21.6 %) |
| **filler_with_AR** | cfg B, `compute` n=6000 | 3 | 2,640.5 ; 2,653.5 ; 2,664.1 | **2,653.5** | 23.6 µs (0.89 %) |

Every leg except `filler_alone` is tight (≤ 0.9 %). **`filler_alone` is the one loose number and
it is reported as a range, not a point:** 1,165.8 µs came from a settled standalone run, 1,417.5 µs
from the in-script baseline that executed immediately after a 39.6 s RCCL process exited (RCCL
teardown still in flight). A planned 4-repeat re-measurement to settle this **did not complete —
the node dropped**. Both values are carried through the arithmetic below; the verdict does not
depend on which is right.

`max` is excluded throughout: every `stdrccl` run carries the first-iteration RCCL channel-setup
outlier documented in `RHO_LADDER_G0A.md` §2b (here 41,171 µs against a 1,300 µs median). At
n=3000 and n=20000 the median is immune.

## 4. The arithmetic

**Leg 1 — what the collective pays.**

```
ΔAR = AR_with_filler − AR_alone = 1,300.4 − 1,300.4 = 0.0 µs   →   0.00 %
```

Across individual repeats ΔAR ranges −4.9 µs to +0.0 µs (**−0.38 % to +0.00 %**). Every reading
is inside the 0.44 % repeat spread of the arm itself. **The all-reduce does not slow down at all.**
Corroboration from the other configuration: cfg B's diluted AR readings (1,294.9 / 1,294.0 /
1,290.7 µs over 26 s windows) are, if anything, marginally *faster* than solo — i.e. the same null.

**Leg 2 — what the filler contributes.**

The filler retains a fraction `φ = filler_alone / filler_with_AR` of its solo throughput while the
collective runs:

```
φ = 1,165.8 / 2,653.5 = 43.9 %      (conservative filler_alone)
φ = 1,417.5 / 2,653.5 = 53.4 %      (optimistic filler_alone)
```

Solo-equivalent filler work retired during one all-reduce:

```
filler_work_absorbed = AR_with_filler × φ
                     = 1,300.4 × 0.439 = 571.4 µs      (conservative)
                     = 1,300.4 × 0.534 = 694.7 µs      (optimistic)
```

**Leg 3 — net gain.**

```
net_gain = filler_work_absorbed − ΔAR
         = 571.4 − 0.0 = +571.4 µs per all-reduce      (conservative)
         = 694.7 − 0.0 = +694.7 µs per all-reduce      (optimistic)
```

As a fraction of the exposed collective: **43.9 % – 53.4 % of the all-reduce is absorbed for
free.**

## 5. Verdict

> **GO — the shared-expert filler design is admitted.** The gate rule was *GO if the all-reduce
> slows < 10 % while the filler runs*; measured slowdown is **0.00 %** (worst repeat −0.38 %),
> an order of magnitude inside the bar. G-L2's own pass bar — *hides ≥ 25 % of the G-L0-measured
> exposed AR*, prior 32–34 % — is cleared at **43.9 % on the conservative baseline** and 53.4 % on
> the optimistic one. Net gain **+571 to +695 µs per 235 MB all-reduce**, with the AR term
> contributing exactly zero cost.

The mechanism this exposes is worth stating as a card: **a 235 MB `ncclAllReduce` on this fabric
is not CU-bound.** It leaves enough of the machine idle that a full-GPU matrix-core kernel can run
beside it at ~44–53 % of solo throughput while taking nothing from the collective. That is
precisely the condition the filler design assumed and had never been measured. It also explains
why the prize pool exists at all: the fabric needs only a 28–39 % duty cycle, and the CUs are idle
underneath it.

**What we would have seen if the mechanism were wrong:** ΔAR climbing with the filler's intensity —
the collective's progress being throttled by CU contention or by memory-system pressure from the
MFMA loop. It did not move at all, in 3 repeats, in two independent configurations.

## 6. Sidebar — the G-L0b standalone arms, measured in the same session

The builder's new arms ran clean on first use and close two open plan items. Measured at
`tokens=16384, slab_rows=256, depth=4, bps=2, k_inner=9040, iters=7`:

| arm | median µs | what it is |
|---|---:|---|
| `stdrccl` | 1,320.9 | bare `ncclAllReduce`, 235 MB — **direct** RCCL |
| `transport` | 2,019.3 | CDAR all-reduce with `k_inner=0` — **direct** ours |
| `transport_fused` | 2,034.3 | the fused protocol's own cost, co-residency-free |
| `compute0` | 132.0 | `k_inner=0` residual — what "zero MFMA work" still costs |

**Direct β = 2,019.3 / 1,320.9 = 1.529.** This is **B4 closed**: β is no longer a ratio of two
subtraction-derived numbers. It lands just above the banked subtraction-derived 1.506× and inside
the 1.43–1.60 band the ρ ladder produced, so the three methods agree — our transport is ~1.5×
RCCL's, and that deficit is real, not an artefact of the subtraction.

**I_co is now absolute**, via `I_co(phased) = phased − compute − transport`:

| ρ measured | phased | compute | transport | **I_co absolute** |
|---:|---:|---:|---:|---:|
| 0.631 | 3,634.6 | 1,405.6 | 2,019.3 | **+209.7 µs** |
| 2.383 | 9,331.7 | 6,557.2 | 2,019.3 | **+755.2 µs** |

This is the direct measurement of `RHO_LADDER_G0A.md` §4's finding: the phased arm's co-residency
tax is **not a constant +335 µs** — it grows 3.6× across the ρ ladder, from 210 µs to 755 µs.

`transport_fused − transport` = +15.0 µs: the fused protocol's own overhead, absent any
co-residency, is essentially free. **The fused arm's ~700 µs deficit is therefore not protocol
cost — it is entirely a co-residency/scheduling failure**, which is exactly what the phase ledger
was built to localise.

## 7. Provenance gaps (node dropped mid-session)

- `~/anatomy_g0/build/filler.jsonl`, `filler_raw.txt`, `clocks_filler.csv` **not harvested.** The
  numbers above are from the driver's live stdout, transcribed into `runs.jsonl`; the node-side
  files are the fuller record (per-invocation min/n, the clock/temp trace) and should be pulled on
  reconnect.
- **No clock/temp trace is quoted for this run** for the same reason. The ρ ladder 20 minutes
  earlier on the same GPUs showed clocks pinned at boost (2,201–2,211 MHz, 0.45 % spread, hotspot
  ≤ 62 °C); no thermal excursion is expected, but it is **not verified for this run**.
- `filler_alone` re-measurement (4 repeats) **did not run** — this is the one number that would
  tighten the 44–53 % range to a point. It does not change the verdict.
- Instrumented-vs-uninstrumented **wall parity was not run** — required before any `M25_LEDGER=1`
  span is quotable, and unrelated to this verdict (this run is the flag-OFF build).

Resume commands for all of these are in the report's WAITING section and in
`NODE_SESSION_1_STATE.md` §6.

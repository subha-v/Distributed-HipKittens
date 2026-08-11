# exp_02 — the decision campaign: does `mps_mega` beat both?

**Verdict: NO, and not by a little. At `C=8,g=2,mode=2,flush_rows=16` the MPS
arm is 8.33x slower than the current best.** Every correctness gate passes, so
this is a pure performance verdict on one configuration, not a bug. It lands
above the pre-registered falsifier (6,919.8 us) by a factor of 8, which under
the cost model kills *this configuration* — and, per the standing rule in
CLAUDE.md, a config-level kill is not a technique-level kill. exp_03 shows the
axis was mis-sized exactly as the A1 correction predicted.

## Protocol

- Campaign `dec01_m2_C8g2fr16`, MoK synthetic prefill, 8 ranks.
- Arms `production,pf6gm_mega,mps_mega`, **all three in every run** (paired,
  same-run denominator), 5 rotated processes.
- `K0_MOK_WARMUP_ITERS=500 K0_MOK_TIMED_ITERS=100`, `K0_SPIN_LIMIT=2000000`.
- `K0_MPS_CFG=C=8,g=2,mode=2,flush_rows=16` — the config named in CLAUDE.md and
  the exact one that used to fault.
- Statistic: `summarize.py` median over rotations of the aligned MAX(rank) p50.
- Arm order rotated across runs: `arm_order_counts = {mps,prod,pf6gm: 1,
  pf6gm,mps,prod: 2, prod,pf6gm,mps: 2}`, so no arm is systematically first.

## Result

| arm | median rank-max p50 | min | max | vs `production` | vs `pf6gm_mega` |
|---|---:|---:|---:|---:|---:|
| `production` | 7,694.0 us | 7,690.5 | 7,710.4 | 1.00000 | — |
| `pf6gm_mega` | 6,885.8 us | 6,869.8 | 6,928.0 | **0.89495** | 1.00000 |
| `mps_mega` | 57,347.4 us | 56,867.5 | 57,646.7 | 7.45353 | **8.32842** |

**Reference check passes.** `pf6gm/production = 0.89495` against the expected
~0.896 for the G=3-forwarded reference, so we built the right denominator and
the historical 6,919.8 us / 7,698.0 us pair reproduces (6,885.8 / 7,694.0 here,
0.5% absolute drift, ratio stable — exactly the drift discipline in CLAUDE.md).

Spread is tight: production varies 0.26% across rotations, pf6gm 0.85%, mps
1.4%. The 8.33x is not noise.

## Gates — all green

| gate | count | result |
|---|---|---|
| `[MOK GATE] pass=True` | 15/15 (3 arms x 5 runs) | pass |
| `[MOK GATE] pass=False` | 0 | pass |
| `mps_mega` worst `max_abs` | 0.039062 (limit 0.1) | pass |
| `mps_mega` worst `relative` | 0.008294 (limit 0.1) | pass |
| `pperr` | 0 everywhere | pass |
| `[MARK] control_fails=True` | 5/5 | pass — negative control still fails |
| `[MPS SOAK] 600/600 pperr=0` | 5/5 | pass |
| `address (nil)` / memory fault | 0 | pass |
| `route_digest_check` | `benign_topk_tie_tolerated` | pass |

Note `mps_mega` and `pf6gm_mega` report *identical* `max_abs` and `relative`
per run (0.035156 / 0.008293 etc.). The MPS arm is numerically indistinguishable
from the reference megakernel, which is the strongest available evidence that
the fix restored exact parity semantics rather than merely stopping the fault.

## What the number means

The slowdown is entirely in the streaming path, not in the role split itself.
exp_03's control screen puts mode 0 (reserve `C` CTAs, push mechanism OFF) at
`1.016x` for `C=8` — the reservation is nearly free. So the 50.5 ms of added
time is the service pool's transport, and the mechanism to interrogate is
per-service-CTA push throughput, not the existence of the split.

Pre-registered falsifier status: the cost model said `T > 6,919.8 us` at
C=8/mode 2 falsifies the mechanism *at that point*. `T = 57,347 us`. That point
is dead and will not be revisited. The A1 correction in CLAUDE.md predicted
precisely this ("`C <= 16` ... inside a band where under-provisioning measured
1.91x SLOWER than no overlap", "a flat or bad curve at `C <= 16` is the EXPECTED
result and must NOT be logged as a kill"). Our collapse is 8.3x, far worse than
the 1.91x the literature reported, because in our shape the service pool carries
**100%** of the combine transport rather than a share of it.

## Primitives

Nothing was added or changed for this experiment; it is a measurement of the
existing kernel. Two observations for the ledger:

- `store_peer_packets` (`packet.cuh`) is called with `bytes = 896*g`, i.e.
  1,792 B at `g=2`, spread over 64 lanes = 28 B/lane = under two 16-byte
  packets per lane. **The primitive's cost is dominated by its call overhead at
  this size, and nothing in its surface signals that.** A payload-transport
  primitive that is only efficient above some size should either document the
  floor or expose a multi-row/banded form. This is the concrete surface change
  exp_03 argues for.
- `finish_order_partition` / `role_partition` (`roles.cuh`) are vindicated:
  mode 0 costs 1.6% at `C=8` and 6.1% at `C=64`, so the partition primitive
  itself is cheap and is not what we are paying for. Reserving CTAs is not the
  expensive part of CTA role specialization on CDNA4 — moving the bytes is.

## Artifacts

- Node log: `~/overnight-scratch/dec01_m2_C8g2fr16.log`
- Summary: `~/k0-mok-mps/dec01_m2_C8g2fr16/summary.json`
- Kernel sha256 `709f4d99e25a0c69a5007cab2fcb1689f012377d8a3e86c401f1362892170835`
  (commit `5dc61fb0`).

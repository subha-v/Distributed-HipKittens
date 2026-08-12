# exp_20 — bottleneck attribution refresh at the current best config

Paper Q3 ("what step is bottlenecking speed right now"). The `RESULTS.md`
attribution table predates the WGM tile-order fix and E3's release grouping, so
it describes a kernel that no longer exists. Every optimization ranked tonight
is ranked off this table, so it is re-measured before anything is built.

## The configuration being attributed (read, not assumed)

Per-shape tile / reducer table, `gemm_rs_mi300x_host_abi.hpp:109-116`
(`shape_config = {bm, bn, bk, num_reducer_ctas, config_row}`):

| # | M×N×K_local | has_bias | BM | BN | BK | NR |
|---|---|---|---:|---:|---:|---:|
| 1 | 64×7168×2304 | false | 32 | 64 | 128 | 56 |
| 2 | 512×4096×1536 | true | 64 | 128 | 64 | 32 |
| 3 | 2048×2880×360 | true | 128 | 192 | 32 | 32 |
| 4 | 4096×4096×512 | false | 256 | 256 | 32 | 32 |
| 5 | 8192×4096×1792 | true | 256 | 256 | 32 | 32 |
| 6 | 8192×8192×3696 | false | 256 | 256 | 32 | 48 |

(K in the harness/graded shape string is the full K; `k_local = K/8`.)

Macros, `gemm_rs_mi300x.cpp:47-89`, and neither `harness/build.sh` nor
`exp_ablation.py`'s compile line overrides any of them:

- `HK_GEMM_RS_MI300X_WGM4 = 0` — the donor WGM=4 grouped tile order is OFF;
  the destination-spreading order is the live path.
- `HK_GEMM_RS_MI300X_RELEASE_GROUP = 4`
- `HK_GEMM_RS_MI300X_RELEASE_GROUP_FULL_ONLY = 1` — group only when a producer
  CTA owns ≥ 4 tiles, so shapes with 2 tiles/CTA keep per-tile release.

Grid is 304 persistent CTAs, `304 − NR` producers.

## Instrument

`harness/exp_ablation.py`, six macro-gated arms built from a generated scratch
source (`harness/ablate/gemm_rs_ablate.cpp`), each cell a separate process:

| arm | cut | pool it prices (full − arm) |
|---|---|---|
| `full` | none | denominator |
| `nomain` | k-loop skipped | **GEMM** — MFMA + A/B global→LDS |
| `emitlocal` | `dest_eff = me` | **egress** — the xGMI cost, bytes and store count held fixed |
| `nored` | no pull/sum/store | **reduce** |
| `noproto` | reducers exit, no credit wait | **sync** |
| `norelease` | no `release_payload_system()` | **release** — `buffer_wbl2 sc0 sc1` + `vmcnt(0)` |

Driven through `tools/reattribute.sh 1617 12` (not hand-rolled) because it
carries the two guards this measurement needs:

1. **Forced rebuild.** `exp_ablation.py:201` skips compilation when the arm
   `.so` exists ("already built"), which silently re-reports the previous
   kernel's attribution. `reattribute.sh:42` removes every arm `.so` and the
   generated scratch first.
2. **Freshness assertion** on shape 6's `full`.

40 iterations per cell (`reattribute.sh:46`), one process driving all 8 devices.

## Freshness gate (pre-registered)

Current best-of-arm pipelined vector: `62.38 / 64.52 / 83.75 / 198.71 / 613.70
/ 1616.63` µs. The ablation harness reads a few percent above M7. **Gate: shape
6 `full` ∈ 1617 ± 12% = [1423, 1811] µs.** Outside that, the table is stale or
the build is wrong, and it must not be ranked off.

## Noise floors (quoted beside every delta, never ranked through)

Per-shape null-arm floors, `HANDOFF.md:102-105`: **1.34 / 0.56 / 0.61 / 2.41 /
2.17 / 4.28 %** published, with exp_14 measuring **3.48 / 0.93 / 1.61 %** on
shapes 1-3 over 15 draws. The published set is a LOWER bound; shapes 1-3 use
the larger exp_14 value. The bias is per-allocation and partly
allocation-ORDER, and ablation cells are separate processes with separate
allocations, so any stage delta below its shape's floor is labelled "inside the
allocation-noise floor" and two stages whose difference is inside the floor are
reported as tied.

Shape 1 is `bound = HOST` at ~10× SOL: a ~63 µs call whose deltas are a few µs.
No mainloop conclusion may be drawn from it.

## Pre-registered expectation

The stale table said GEMM 46% / egress 32% / release 9% on shape 6 at a 2861.7
µs total. WGM and E3 attacked exactly egress and release, and the total has
since fallen to ~1617 µs, so the expectation is: **GEMM is now a larger SHARE
than 46%, egress and release have shrunk in absolute µs, and the mainloop is
the unambiguous #1 target.** If egress is still ~30% the WGM win came from
somewhere else and that is the finding.

## Deliverables

- `ablation.json` — plot-ready, schema restated verbatim in `result.md`.
- `counters.json` + rocprofv3 log — one counter pass reusing exp_08's validated
  gfx942 counter set, shape 6 and one mid shape.
- `result.md` — verdict, per-shape table, ranking by absolute µs and by share,
  each with its floor, and the delta against the two stale tables.

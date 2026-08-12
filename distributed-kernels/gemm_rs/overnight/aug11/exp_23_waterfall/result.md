# exp_23 result — the knob waterfall (paper Fig 4 / Q1)

**STATUS: build + fingerprint dispatch only. No GPU job has been run.**
Timing sections below are stubs and must not be cited until filled.

---

## 1. Verdict

*(to be filled after the draws)*

## 2. Macro semantics, as confirmed against source

| macro | default | shipped meaning | rung use |
|---|---|---|---|
| `HK_GEMM_RS_MI300X_WGM4` (`:47-49`, `:315-319`) | **0** | `WGM = (tiles <= num_gemm_ctas) ? 4 : num_pid_m` — the column-major decode that spreads a round's tiles over all eight egress destinations | `=1` restores the donor `WGM = 4` and is therefore the **rung (a)** control. The knob is **inverted relative to its name** |
| `HK_GEMM_RS_MI300X_RELEASE_GROUP` (`:62-64`, `:324-328`) | **4** | tiles emitted between `producer_drain_release()` calls; `static_assert(1..8)` | `=1` is the per-tile control (rungs a, b) |
| `HK_GEMM_RS_MI300X_RELEASE_GROUP_FULL_ONLY` (`:87-89`, `:336-340`) | **1** | group only when `tiles_per_cta >= RELEASE_GROUP` | pinned at 1 in **every** rung so it is not a hidden second variable |
| `HK_GEMM_RS_MI300X_TILE_SWEEP` (`:97-99`) | 0 | gates `exp_14`'s screening rows 101-107 in the dispatch table | not passed; out of scope |
| `HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS` (`:40-42`) | 0 | control branches | untouched |

## 3. Build matrix and fingerprints — **PASSED**

Built 2026-08-12 from `gemm_rs_mi300x.cpp` sha256
`ef61006eed6f70a25ed9635a94d5b581eb5f82a2c33c035b94cf44a8abffb560` (unmodified),
HIP 7.2.53211-c2d9476115, all four rungs via the two-step compile-then-link path
(no fallbacks). Full data in `fingerprints.json`.

| rung | `WGM4` / `RG` | `.so` sha256[:12] | ISA sha256[:12] | ISA lines | instrs | `v_mfma` | `buffer_wbl2` | `s_cbranch` |
|---|---|---|---|---|---|---|---|---|
| a | 1 / 1 | `caa8fd939490` | `ec714ea8a644` | 25245 | 20190 | 184 | 7 | 1268 |
| b | 0 / 1 | `5d8d3882e89b` | `366d4f08a030` | 25254 | 20201 | 184 | 7 | 1268 |
| c | 0 / 4 | `16996e40403d` | `7926c2e87283` | 26009 | 20734 | 184 | 7 | 1272 |
| null | 0 / 4 | `51723ebfd24b` | `7926c2e87283` | 26009 | 20734 | 184 | 7 | 1272 |

ISA hashes are taken over the gfx942 assembly with HIP's per-TU
`__hip_cuid_<hash>` symbol normalised away; see `design.md` §3.

| id | verdict | detail |
|---|---|---|
| A1 (hard) | **PASS** | `a` and `b` ISA differ |
| A1b | **PASS** | attributable: instructions 20190→20201, `s_cselect` 535→542 — the `tiles <= num_gemm_ctas` select that `WGM4=1` folds away |
| A2 (hard) | **PASS** | `b` and `c` ISA differ |
| A3 | **PASS** | attributable: `s_cbranch` 1268→1272, instructions 20201→20734 — the group loop `RELEASE_GROUP=4` leaves in place |
| A4 (hard) | **PASS** | `c` and `null` ISA hash **identically**; the raw files differ on 5 lines, every one of them a `__hip_cuid_` line. The null arm is therefore a pure allocation contrast |
| A5 (hard) | **PASS** | `v_mfma` = 184 in all four rungs — the mainloop is not a variable here |
| A6 (hard) | **PASS** | 7 instantiations; VGPRs {98, 104, 136, 246, 248, 91, 92} exactly as `RESULTS.md`'s post-exp_14 M2 table; 0 AGPRs, 0 scratch, 0 VGPR spills everywhere |
| A6b | **PASS** | rung `c`'s resource table is **identical field-for-field** to the shipped `harness/build/gemm_rs_mi300x.log` — independent confirmation that rung (c) reproduces the shipped binary |
| A7 (hard) | **PASS** | all four modules import and export `gemm_rs_mi300x` |

**Rung (c) — and its twin `null` — reproduce the shipped binary.** Rungs (a)
and (b) are genuinely different code; no rung can masquerade as another.

Note for the record: scalar spills are 54-88 SGPRs depending on the row and are
**not** asserted zero. The M2 claim in `RESULTS.md` is about AGPRs, scratch and
*vector* spills, all of which are zero. Two gate bugs were found and fixed
before the gate passed — the resource-remark prefix differs between the
harness build and the `--save-temps` build (`TotalSGPRs`, and the `file:line:`
prefix on the other side of `remark:`), and A4 initially failed on the cuid
symbol described above. Neither was a build problem.

## 4. Data schema — `waterfall.json`, stated verbatim

```json
{
  "experiment": "exp_23_waterfall",
  "protocol": "pipelined, same-run paired, 8 arms per shape, complete rotation, best+median only (means are unusable on this node)",
  "draws": ["draw_<stamp>_<fwd|rev>.json", "..."],
  "draws_reversed": 0,
  "rungs": [
    {
      "rung": "a" | "b" | "c" | "null",
      "config": {"WGM4": 0|1, "RELEASE_GROUP": 1|4, "RELEASE_GROUP_FULL_ONLY": 1},
      "fingerprint_sha": "<sha256 of the rung's gfx942 ISA text>",
      "per_shape": [
        {
          "shape": "MxNxK",
          "shape_index": 1,
          "best_us": 0.0,
          "median_us": 0.0,
          "samples_us": [0.0],
          "null_floor_pct": 0.0,
          "ratio_vs_prev_rung": 1.0,
          "ratio_vs_rung_a": 1.0,
          "rung_active_on_this_shape": true
        }
      ],
      "geomean_best_us": 0.0,
      "geomean_median_us": 0.0
    }
  ],
  "nr_sweep": [
    {
      "nr": 8 | 16 | 32 | 48,
      "config": {"WGM4": 0, "RELEASE_GROUP": 4, "RELEASE_GROUP_FULL_ONLY": 1},
      "fingerprint_sha": "<same sha as rung c>",
      "per_shape": [
        {
          "shape": "MxNxK",
          "shape_index": 1,
          "best_us": 0.0,
          "median_us": 0.0,
          "samples_us": [0.0],
          "null_floor_pct": 0.0,
          "ratio_vs_shipped_nr": 1.0,
          "shipped_nr": 32
        }
      ],
      "geomean_best_us": 0.0,
      "geomean_median_us": 0.0
    }
  ]
}
```

Field meanings, so a plotter never has to guess:

- `best_us` / `median_us` — min and median over **all** wall-clock samples of
  that arm pooled across every draw. One sample = one `_timed_block(iters)`,
  i.e. `iters` operations queued back to back per rank with no host sync,
  divided out. **Means are deliberately absent**: the harness bias on this
  node is per-allocation, which makes a mean a weighted average of allocation
  luck.
- `null_floor_pct` — `|median(c) − median(null)| / min(...) × 100`, taken from
  the **worst single draw**. `c` and `null` are the same device ISA (asserted
  A4) with independent allocations, so this is the instrument's own resolution
  at that shape. **Any delta smaller than this is no evidence**, not a small
  effect.
- `ratio_vs_prev_rung` — median of this rung ÷ median of the preceding rung in
  the ladder (`a → b → c → null`); below 1.0 is faster.
- `ratio_vs_rung_a` — median of this rung ÷ median of rung (a), the aug10
  baseline; this is the waterfall's cumulative bar.
- `rung_active_on_this_shape` — whether the knob this rung introduces changes
  behaviour at all on that shape, computed from the shape plan before the run
  (`b`: `tiles > num_gemm_ctas`; `c`: `ceil(tiles/num_gemm_ctas) >= 4`). Shapes
  where it is `false` are **predicted flat and serve as extra null pairs**; see
  `plan.md`.
- `ratio_vs_shipped_nr` — NR-sweep arm ÷ rung (c) at that shape's shipped NR
  (56/32/32/32/32/48).
- `geomean_*` — geometric mean over the shapes present; this is the
  competition's ranking statistic and the number the waterfall's bars carry.

Per-draw raw data, including per-sample position, correctness at both
tolerances, epoch/signal state checks and dead-arm diagnoses, lives in
`draw_<stamp>_<fwd|rev>.json`. `waterfall.json` is regenerated from all of them
by `python3 sweep.py --pool`.

## 5. Statistics and what would falsify the figure

*(see `plan.md` for the pre-registered predictions and falsifiers; verdicts to
be recorded here per rung and shape, with the "inside floor — no evidence"
outcome reported as such rather than as a small win)*

## 6. Caveats

- One process drives all eight devices here, which is **not** the official
  evaluator's topology. Any comparison against rank-1's evaluator numbers
  carries that caveat; this experiment makes no such comparison (that is
  exp_24).
- Rungs (a) and (b) are deliberate regressions, not landing candidates, so the
  full external gate ladder (17-shape M3, M4 controls, 600-epoch soak) is not
  run per rung; correctness is enforced per arm inside the sweep at both
  `1e-2` and `2e-3`, plus error bits and epoch/signal cells. Rung (c) is the
  already-landed shipped binary.

## 7. Preemption / node log

*(record any GPU reclaim or contention here)*

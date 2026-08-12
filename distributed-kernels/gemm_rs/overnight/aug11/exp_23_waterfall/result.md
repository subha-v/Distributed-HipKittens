# exp_23 result — the knob waterfall (paper Fig 4 / Q1)

**STATUS: COMPLETE.** Four allocation draws (2 forward-construction, 2
reversed), all six graded shapes, eight arms per shape, complete 8-pass
rotation, under the node GPU lease (`exp_23`, acquired 04:51:11, released
04:54:53 CDT, no preemption, `perf_determinism` on all eight GPUs). All
192 arm instantiations passed correctness at **both** `1e-2` and `2e-3` with
clean error bits and clean epoch/signal state; zero dead arms.

---

## 1. Verdict

**The waterfall is real and it is carried entirely by two shapes, exactly as
pre-registered.** Cumulative geomean, rung (a) → rung (c): **227.62 → 204.52 µs
median (223.65 → 200.60 µs best), a 1.113× speedup**, split
1.082× from the task-order rung and 1.028× from the release-granularity rung.

| rung | 64×7168×18432 | 512×4096×12288 | 2048×2880×2880 | 4096×4096×4096 | 8192×4096×14336 | 8192×8192×29568 | geomean |
|---|---|---|---|---|---|---|---|
| a | 61.3 / 63.5 | 64.3 / 65.8 | 83.3 / 85.0 | 197.7 / 199.0 | 743.9 / 751.8 | 2595.0 / 2616.3 | **223.65 / 227.62** |
| b | 62.1 / 64.0 | 64.6 / 66.0 | 85.0 / 85.7 | 197.9 / 198.9 | 634.1 / 652.5 | 1801.8 / 1840.8 | **206.29 / 210.28** |
| c | 60.4 / 62.7 | 64.8 / 66.0 | 83.6 / 84.5 | 198.0 / 199.1 | 624.9 / 642.5 | 1609.8 / 1635.7 | **200.60 / 204.52** |
| null | 60.0 / 61.8 | 64.6 / 65.3 | 83.9 / 84.5 | 197.5 / 198.8 | 625.6 / 657.9 | 1601.6 / 1640.9 | **200.18 / 204.50** |

Cells are **best / median µs**; no means anywhere. `null` is rung (c)'s
identically-compiled twin and its geomean lands within **0.01%** of rung (c),
which is the single best evidence that the ladder's top rung is measured
rather than lucky.

### Rung verdicts

| shape | mechanism active | null floor % | a→b | b→c |
|---|---|---|---|---|
| 64×7168×18432 | no / no | 2.82 | −0.60% [−2.10,+0.65] **unresolved** | +1.74% [+0.69,+3.33] **unresolved** |
| 512×4096×12288 | no / no | 2.09 | −0.29% [−0.66,+0.36] **unresolved** | −0.08% [−0.39,+0.24] **unresolved** |
| 2048×2880×2880 | no / no | 1.74 | −0.64% [−2.76,+0.35] **unresolved** | +1.41% [+0.38,+2.52] **unresolved** |
| 4096×4096×4096 | no / no | 0.85 | −0.02% [−0.57,+0.22] **unresolved** | +0.05% [−0.70,+0.74] **unresolved** |
| 8192×4096×14336 | **yes** / no | 4.97 | **+15.11%** [+12.57,+19.07] p=0.0143 **RESOLVED faster** | +0.79% [−1.27,+4.70] **unresolved** |
| 8192×8192×29568 | **yes** / **yes** | 4.44 | **+42.11%** [+40.48,+44.79] p=0.0143 **RESOLVED faster** | **+12.22%** [+10.98,+12.98] p=0.0143 **RESOLVED faster** |

Gains are within-draw paired contrasts on medians (positive = the later rung is
faster), with the per-draw range and the exact one-sided rank-sum p against the
measured null contrast set. At four draws per side the smallest attainable p is
1/C(8,4) = **0.0143**, so 0.0143 means "fully rank-separated", not "p ≈ 0.01".

**Structural prediction: HOLDS.** Every contrast the mechanism cannot reach —
a→b on shapes 1-4, b→c on shapes 1-5 — is unresolved against tonight's floor.
No blocker. The closest call is b→c on shape 3 (+1.41%, rank-separated on
medians at p=0.0143) which sits **inside** that shape's 1.74% floor; it is
reported as unresolved. Note rungs (b) and (c) are different binaries even
where `rgroup` collapses to 1 at runtime, so a sub-floor codegen delta on an
inert shape is expected and is not evidence that the release mechanism reached
something.

**Shape 6's b→c came in at +12.22%, not the ~4% pre-registered.** It is
comfortably resolved. The pre-registration took −4.0% from exp_05's *graded*
protocol; this is pipelined, and it is measured on top of the WGM task order
rather than on the aug10 order, so the two are not the same quantity. The
larger effect is consistent with grouping releases mattering more once the
egress is actually spread across all eight links.

### The methodological finding, which changes how the floor must be measured

`resolve_shape_with_split` overrides only the CTA split
(`dhk_rt.cpp:197-206`), so on any shape whose shipped NR is one of the swept
points, that NR arm is **configured identically to rung (c) and runs the same
binary** — a second, independent null pair that came for free.

On shape 6 (shipped NR = 48) the two null pairs disagreed badly:

- `c` vs `null`: −0.62, +1.02, −0.73, +1.10 % → floor 1.10%
- `c` vs `nr48`: +3.74, +4.44, +4.10, +2.53 % → floor **4.44%**

The second pair is same-config, same-binary, and it was positive in **all four
draws and in both construction orders**, so the disjointness rule certified it
"RESOLVED faster" at p=0.0143. **That is a false positive**, and it lands almost
exactly on the 4.28% shape-6 floor `HANDOFF.md` published. One of the five
available twin pairs produced it. Two consequences, both adopted here:

1. The null set is the **union of every identically-configured pair** at that
   shape, which is strictly more conservative. That is what the floors in the
   tables above are, and it is what moved shape 3 from a near-miss (0.34%
   single-pair floor) to comfortably unresolved (1.74%).
2. **A single null twin is not enough on this node**, and neither is
   consistency across draws or across construction orders — this artefact
   survived both. Every future paired experiment here should carry at least two
   independent twins.

None of this disturbs the two resolved rungs: shape 6's b→c range
[+10.98, +12.98] is disjoint from the widened null [−0.73, +4.44], and a→b is
an order of magnitude clear on both moving shapes.

### Rung (d), the NR sweep — the pre-registered "flat" prediction is FALSIFIED

| shape | shipped NR | NR=8 | NR=16 | NR=32 | NR=48 |
|---|---|---|---|---|---|
| 64×7168×18432 | **56** | −51.48% slower | −28.11% slower | −7.96% slower | +0.10% unresolved |
| 512×4096×12288 | **32** | −25.00% slower | −8.94% slower | +0.69% *(=shipped: null pair)* | +0.35% unresolved |
| 2048×2880×2880 | **32** | −27.26% slower | −11.04% slower | −0.04% *(=shipped: null pair)* | +0.48% unresolved |
| 4096×4096×4096 | **32** | −24.55% slower | −9.80% slower | −0.12% *(=shipped: null pair)* | +0.34% unresolved |
| 8192×4096×14336 | **32** | −13.75% slower | −5.01% unresolved | −2.37% *(=shipped: null pair)* | +0.44% unresolved |
| 8192×8192×29568 | **48** | −18.80% slower | −8.87% slower | −3.53% slower | +3.92% *(=shipped: null pair)* |

Geomeans vs rung (c): NR=8 **1.392×** slower, NR=16 **1.141×** slower,
NR=32 1.023× slower, NR=48 0.992×.

The curve is **not flat across {8, 16, 32, 48}**. It is flat on the
**32-56 plateau** — every shape is inside its floor between its shipped NR and
NR=48 — and it falls off a cliff below that: NR=16 is resolved slower on five of
six shapes and NR=8 on all six, by 14-51%. The pre-registration said flat
everywhere and that is falsified; the honest statement is *"enough reducers, not
more"*.

Read against Q2 ("do dedicated communication blocks ever win?"), the exhibit
still answers no, but for a sharper reason than flatness: the reducers are not a
communication pool — the payload leaves in the producer's epilogue — they are
the owner-side reduce, and starving them starves the operator. Above ~32 CTAs,
**adding CTAs to the non-GEMM side buys nothing measurable on any shape**, which
is the claim the paper needs. The `nr48` column reads as a small uniform win
(+0.1 to +0.5%) on the four NR=32 shapes, every one of them inside its floor,
and its one "resolved" cell is the shape-6 null pair described above — so the
ladder did **not** regress at its own best, and the figure must mark the shipped
points or it will imply that it did.

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

## 5. Statistics as executed

Four allocation draws, fixed before any number was seen: `fwd, rev, fwd, rev`.
The stopping rule was written into `campaign.sh` rather than decided afterwards,
because adding draws until a contrast crosses is how a false positive is
manufactured — and this run produced a concrete example of one (§1).

- Unit of evidence is **one draw** (one process = one `hipMalloc` outcome), so
  each draw contributes one number per contrast: the within-draw ratio of
  medians. Samples inside a draw are repeated measurements of one allocation,
  not independent observations of the arm.
- Null contrast set = every identically-configured pair at that shape
  (`c` vs `null`, plus `c` vs `nr<shipped>` where it exists), pooled over draws.
- Verdict = **RESOLVED** only when the candidate's per-draw contrast range is
  disjoint from the null contrast range on **both** best and median. Exact
  one-sided Wilcoxon rank-sum p is reported alongside; at 4 draws per side its
  floor is 1/C(8,4) = 0.0143 and at 4-vs-8 (shapes with two twins) 1/C(12,4) =
  0.0020.
- Anything else is **UNRESOLVED — no evidence**, and is reported as such rather
  than as a small win.
- Per-position residual was recorded every pass and stayed small; the dominant
  term is per-allocation, as documented.
- Raw per-draw data: `draw_*.json`. Pooled plot data: `waterfall.json`. Scored
  verdicts: `stats.json` / `stats.txt`. Rendered tables: `tables.md`.

### Pre-registered predictions, scored

| prediction | outcome |
|---|---|
| shapes 1-4 flat across a/b/c | **HELD** — all eight contrasts unresolved |
| shape 6 a→b ≈ −28% | **HELD** — +42.11% gain = −29.6% time, against exp_08's −27.9% |
| shape 6 b→c ≈ −4%, likely unresolved | **RESOLVED**, and larger than predicted (+12.22% gain = −10.9% time); see §1 for why the two numbers are not the same quantity |
| NR curve flat over {8,16,32,48} | **FALSIFIED** — flat only on the 32-56 plateau; NR=16 and NR=8 are resolved slower everywhere |
| NR=8 degrades on shape 6 | **HELD**, and it degrades on all six |

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

- GPU lease `exp_23` acquired 2026-08-12T04:51:11-05:00 after one foreign KFD
  pid drained; released 04:54:53 from the campaign's `EXIT` trap. **No
  preemption, no contention, nothing stolen, no process signalled.**
- `perf_determinism` confirmed on all eight GPUs before the campaign. The
  `--showclocks` samples bracketing the run read idle sclk (~120-139 MHz)
  because they are taken between jobs; the pinned level is the performance-level
  reading, not the instantaneous clock.
- Warmup is duration-based (400 ms per shape) plus 4 priming launches
  immediately before every timed block, so no block measures a ramping GPU.
- The whole campaign — preflight smoke draw plus four draws plus scoring — took
  3 m 42 s of lease time.

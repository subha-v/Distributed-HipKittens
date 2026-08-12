# exp_23 design — arm construction, build matrix, fingerprint gate, statistics

## 1. Arm construction

One process, eight devices, one shape at a time. For a shape, every arm is an
independent `harness_lib.GemmRS` handle with its own symmetric payload heap,
its own signal heap and its own epoch/error cells, and **all arms for a shape
are alive simultaneously** so the timing loop can interleave them. Operands
are generated ONCE per shape and shared by every arm by assignment
(`handle.inputs = inputs; handle._rebuild_args()`), so two arms read the same
bytes at the same addresses and differ only in their binary, their NR, and
their own output allocations.

Eight arms per shape:

| label | module | `num_reducer_ctas` |
|---|---|---|
| `a` | `gemm_rs_w23_a` | shipped (from `resolve_shape`) |
| `b` | `gemm_rs_w23_b` | shipped |
| `c` | `gemm_rs_w23_c` | shipped |
| `null` | `gemm_rs_w23_null` | shipped |
| `nr8` `nr16` `nr32` `nr48` | `gemm_rs_w23_c` | 8 / 16 / 32 / 48 |

`null` is a separate compilation of the rung-(c) flag set under a different
`TK_MODNAME`, because a pybind module's init symbol is `PyInit_<modname>` and
one `.so` cannot be imported twice under two names. Its device ISA must hash
**identically** to rung (c) — that is asserted, and it is what makes `c` vs
`null` a pure allocation contrast rather than a code contrast.

On shape 6 the shipped NR is 48, so `c`/`null` and `nr48` coincide in
configuration; that is left in deliberately as extra null evidence rather than
special-cased away.

`harness_lib.BUILD_DIR` is repointed at this experiment's `build/` **after**
importing `harness_lib`, so `dhk_rt` still loads from `harness/build` (it is
unchanged and shared, read-only) while every kernel module comes from here.
Nothing is written under `harness/build`, where another agent's ablation arms
live.

## 2. Build matrix

`build_rungs.sh` builds four modules from the **unmodified** kernel source.
Flags are copied from `harness/build.sh` verbatim, including the
`-I$ROCM_PATH/include/hip` that the documented Gate-M1 command omits and
without which `<hip_bf16.h>` is not found.

```
COMMON = -std=c++20 -O3 -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS
         -ffast-math --offload-arch=gfx942 -fPIC
         -I<repo>/include -I<repo>/include/pyutils -I$ROCM_PATH/include/hip
         -I<pybind11> -I<python> -Wno-nan-infinity-disabled -ferror-limit=0
```

| module | added flags |
|---|---|
| `gemm_rs_w23_a` | `-DHK_GEMM_RS_MI300X_WGM4=1 -DHK_GEMM_RS_MI300X_RELEASE_GROUP=1` |
| `gemm_rs_w23_b` | `-DHK_GEMM_RS_MI300X_WGM4=0 -DHK_GEMM_RS_MI300X_RELEASE_GROUP=1` |
| `gemm_rs_w23_c` | `-DHK_GEMM_RS_MI300X_WGM4=0 -DHK_GEMM_RS_MI300X_RELEASE_GROUP=4` |
| `gemm_rs_w23_null` | identical to `gemm_rs_w23_c` |

plus `-DHK_GEMM_RS_MI300X_RELEASE_GROUP_FULL_ONLY=1` and
`-DTK_MODNAME=<module>` on all four.

**Two-step build, on purpose.** Each rung is compiled once to an object with
`--save-temps -c -fPIC -Rpass-analysis=kernel-resource-usage`, and the `.so` is
then *linked from that same object*. The alternative — a `-shared` compile plus
a second `--save-temps` compile for the ISA — gives a disassembly that only
*probably* corresponds to the shipped module. Linking the fingerprinted object
makes the correspondence structural: the fatbin in the `.so` is the one that
was disassembled. If the link or the import check fails for a rung, the script
falls back to the `harness/build.sh` one-step `-shared` form for that rung and
records `fallback: true` so the weaker guarantee is visible in the artifact.

Every `.so` is removed before its rebuild. `exp_14` documents the failure this
prevents: an "already built" shortcut once reported a previous kernel's
attribution for an hour.

Outputs live in `exp_23_waterfall/build/` (`*.so`, `*.compile.log`) and
`exp_23_waterfall/build/isa/<module>.s`.

## 3. The fingerprint gate

The failure mode this guards against is invisible by construction: a skipped
rebuild produces a perfect-looking waterfall of one identical binary measured
four times. `fingerprint.py` runs before any timing and writes
`fingerprints.json`.

Recorded per module:

- `so_sha256`, `so_bytes`
- `isa_sha256` over the gfx942 assembly text, and `isa_lines`.
  HIP stamps each translation unit with a `__hip_cuid_<hash>` symbol derived
  from the TU's identity, so two builds of the same source under different
  `TK_MODNAME` values differ on exactly those lines and nowhere else. The
  hashed text normalises that symbol away — otherwise A4 would always fail and
  the null arm would be reported as a code contrast when it is an allocation
  contrast. The un-normalised `isa_raw_sha256` is recorded too, and for the
  `c`/`null` pair the raw differing-line count and a proof that every one of
  those lines is a cuid line are recorded alongside.
- the resource tuple per instantiation, parsed from the
  `-Rpass-analysis=kernel-resource-usage` remarks: SGPR / VGPR / AGPR /
  scratch / dynamic-stack / occupancy / SGPR-spill / VGPR-spill / LDS
- discriminating structure counts over the ISA: `buffer_wbl2` (release
  sites), `buffer_inv`, `s_cbranch*`, `s_branch`, `v_mfma*`,
  `global_store_dwordx4`, `global_load_dwordx4`, `s_waitcnt vmcnt`,
  `scratch_store` / `scratch_load`, and the total instruction count

Assertions, each keyed to a mechanism rather than to a hope:

| id | assertion | mechanism |
|---|---|---|
| **A1** | `isa_sha256(a) != isa_sha256(b)` | `WGM4=1` constant-folds the `tiles <= num_gemm_ctas` select at `:315-319` out of the tile decode |
| **A2** | `isa_sha256(b) != isa_sha256(c)` | `RELEASE_GROUP=1` makes `rgroup` the constant 1 at `:336-340`, collapsing the inner group loop; `=4` leaves a runtime 4-or-1 group loop with one release per group |
| **A3** | branch/release structure differs between `b` and `c` — at least one of `buffer_wbl2`, `s_cbranch*`, total-instruction-count | same as A2, but a count rather than a hash, so the difference is attributable rather than merely present |
| **A4** | `isa_sha256(c) == isa_sha256(null)` | `TK_MODNAME` names the host pybind module only; the device code must be untouched. A difference here means the build is not reproducible and every null-floor number would be a code contrast |
| **A5** | `v_mfma*` count identical across all four | the mainloop is not a variable in this experiment. If it moves, a rung is changing more than its knob |
| **A6** | rung `c` reproduces the shipped Gate-M2 table: 7 instantiations, `32/64/128`→VGPR 98, `64/128/64`→104, `128/192/32`+tail→136, `256/256/32`→246, `256/256/32`+tail→248, generic `32/64/64`→91/92, and **0 AGPRs / 0 scratch / 0 VGPR spills everywhere**. *Scalar* spills are deliberately not asserted — the shipped binary spills 54-60 SGPRs on the small rows and the M2 claim has always been about vector spills | rung (c) *is* the shipped configuration; if it does not reproduce `RESULTS.md`'s post-exp_14 M2 table the whole ladder is anchored to the wrong binary |
| **A7** | all four modules expose the `gemm_rs_mi300x` entry point and import cleanly | catches a link that produced a file but not a module |

A1, A2, A4, A6, A7 are **hard**: a failure means the rung is not real and it
is reported as a blocker instead of measured. A3 and A5 are recorded and
reported; A5 failing is a hard stop, A3 failing downgrades A2's attribution to
"hash-only" and is disclosed in `result.md`.

`fingerprints.json` also records which module reproduces the shipped binary
(expected: `c` and `null`), so the ladder's top rung is anchored to something
whose provenance is checkable.

## 4. Statistics

**Unit of evidence is one allocation draw**, i.e. one process. Samples inside a
draw share one `hipMalloc` outcome, so they are repeated measurements of that
draw rather than independent observations of the arm; `exp_14`'s `pool.py`
argues this at length and its scorer is reused rather than reinvented.

Within a draw, per shape:

- complete rotation: `passes == len(arms) == 8`, order rotated by one per pass,
  so each arm occupies each position exactly once and the per-position residual
  is reported directly as a measurement of the positional component (it has
  historically been under ±0.5%, i.e. not the dominant term).
- each sample is `GemmRS._timed_block(iters)`: `iters` operations queued back
  to back per rank with no host sync, divided out. Warmup is duration-based
  (`400 ms`) at the top of the shape, and every arm is primed for 4 launches
  immediately before its own block.
- `iters` per shape: `300 / 250 / 250 / 120 / 60 / 50`, floored at 50 so the
  charter's "3 rotations × 50 iterations minimum" holds at the two large
  shapes; the values keep one block at roughly 50-100 ms of device work.
- reported per arm: **best (min) and median** over its 8 samples, and the
  spread. **Means are not reported** — the harness bias makes them unusable.
- geometric mean over the six shapes is the ranking statistic and is computed
  for both best and median.

Across draws: `sweep.py --pool` re-reads every `draw_*.json` beside it and
rebuilds `waterfall.json` from the pooled samples, with the null floor taken
as the **worst single draw** of `|c − null|` and the pooled `c + null` range
reported beside it. Half the draws are run with `SWEEP_REVERSE=1`, which
reverses the order in which arms are constructed (not the timing rotation), so
the construction-order component of the bias enters the null set symmetrically.

Verdict rule, per rung and shape: a delta counts only if it is outside the
worst-draw null floor **and** the paired contrast against the null contrast set
is disjoint (or has exact rank-sum p < 0.01 with a consistent sign), the same
rule `pool.py` applies. Anything else is reported as "inside floor — no
evidence", including when that is the answer for the b→c rung on shape 6.

## 5. Correctness inside the sweep

Before an arm contributes a single timing sample it is launched once and
checked against the harness oracle (`reference_reduce_scatter`) at **both**
`2e-3` and `1e-2`, and its error word is read. After the rotation, every
surviving arm's epoch cells and signal cells are checked
(`check_epochs()` + `check_signals()`). Any failure marks the arm dead; a dead
arm is reported with its diagnosis and is never plotted. Tolerances are gates.

## 6. What this experiment does not do

- It does not edit the four kernel sources, `harness/`, `tools/`, or the host
  ABI NR table.
- It does not write to `harness/build/`.
- It does not compare against rank-1 or reference GEMM+RCCL; that is exp_24.
- It does not run the full external gate ladder per rung. The ladder's M1/M2
  steps are subsumed by the build + fingerprint gate here, and M3-style
  correctness is enforced per arm inside the sweep at both tolerances; the
  17-shape M3, the M4 controls and the 600-epoch soak apply to a *landing*
  candidate, and no rung here is a landing candidate — rung (c) is already the
  shipped binary and rungs (a)/(b) are deliberate regressions.

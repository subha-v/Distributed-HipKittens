# GEMM-RS MI300X handoff — evidence ledger and state

Rewritten 2026-08-11 ~23:00 PT. `experiments/LESSONS.md` is the full
append-only record; this file is the state of play.

## One-paragraph state

The kernel is **~24% faster than it started** (our harness geomean
285.02 → ~217.5 µs) across eight landed changes, each through the full gate
ladder. **rank-1, the frozen competition submission, now runs on this node**
and is measured same-run interleaved against us: we are **1.137× behind** on
the graded protocol, down from **1.784×** when it first ran. We also beat the
reference GEMM+RCCL baseline. The remaining gap is concentrated in shapes 5 and
6 and is dominated by the GEMM mainloop, which sits ~2.5-3× off single-CU peak.

## The ratchet

| level | denominator | value | status |
|---|---|---|---|
| 1 | **rank-1, same-run graded** | 306.60 µs (best geomean) | **1.137× behind** (ours 348.64) |
| 2 | reference GEMM+RCCL, same-run graded | 438.15 | **beaten** |
| 3 | our harness geomean | 285.02 at session start | **~217.5 µs, −24%** |

## What landed

| exp | change | effect |
|---|---|---|
| exp_02 | uniform `NR=32` | 285.02 → 280.74; shape 6 −8.1% |
| exp_04a | shape 1 retiled `32/256/32` → `32/64/64` | → 269.96; shape 1 −20.3% |
| exp_03 | P3 register relief + **E1(b) mainloop issue/commit split** | → 256.09; two exposed global round trips per k-iteration became one covered one |
| exp_08 | **`WGM` = egress-link concurrency** | → 230.84; shape 6 −27.9%; effective xGMI links 2.02 → 7.53 of 8 |
| exp_09 | E1(c) scheduling | 0.997× — **flat axis, closed** |
| exp_11 | **`find_scored_config` ignores `has_bias`** | graded only; shapes 4 and 6 were falling to the generic row under the evaluator |
| exp_12 | **prebound `configure`/`run` launch path** | graded 381.69 → 357.45; host path 21-39 µs → 4.8-7.4 µs |
| exp_05 | **E3 release grouping**, `RELEASE_GROUP=4` full-group | shape 6 graded −4.0% |
| exp_13 | **E4 re-swept**: shape 1 `NR=56`, shape 6 `NR=48` | paired −3.6%; graded → 348.64 |

**Five of these were inherited constants nobody had questioned** —
`NUM_REDUCER_CTAS`, `BM/BN/BK`, `WGM`, the `has_bias` key, and `NR` again after
the kernel changed underneath it. Treat every remaining inherited value as an
untested hypothesis.

## The three things to know before touching anything

### 1. The split buys ROUNDS, not CTAs

Both producer and reducer are strided persistent loops, so what matters is
`ceil(tiles / count)`, not `count`. Shape 6's 1024 tiles over 272 *or* 288
producers is 4 waves either way — which is why "reduce is only 86.5 µs but
holds 32 CTAs, so use fewer reducers" was **backwards** and measured +3.6%.
Wins sit at the **last split before the producer wave count steps**.

Current producer waves with the landed `NR` table (56/32/32/32/32/48 →
NG = 248/272/272/272/272/256):

| shape | tiles | waves | last-wave fill |
|---|---|---|---|
| 1 | 224 | 1 | 90% |
| 2 | 512 | 2 | 88% |
| 3 | 192 | 1 | 71% |
| 4 | 256 | 1 | 94% |
| 5 | 512 | 2 | 88% |
| 6 | 1024 | 4 | **100%** |

### 2. The measurement bias is PER-ALLOCATION, and rotation does not fix it

Two arms with identical splits, identical operands and identical code separated
by **4.28% on shape 6**, with the positional residual never exceeding ±0.47%.
The only difference is which `hipMalloc` returned the 134 MB payload heap.
**Null-arm floors: 1.34 / 0.56 / 0.61 / 2.41 / 2.17 / 4.28 %** for shapes 1-6.

**Run a null arm before believing any per-shape delta**, and clear a candidate
only by pooling several independent allocations and requiring disjoint ranges.
Means are unusable on this node even with interleaving — **report best and
median**. Node noise floor is ~1.3%.

### 3. ~92 µs of every graded call is harness machinery both arms pay

The evaluator's own trailing `synchronize + barrier` is 57-79 µs measured with
*nothing* in the timed region, plus 15-20 µs of input clone. Our own host path
is now ~5-7 µs against a 4.86 µs floor. **Net that constant out before
comparing kernels**: it makes our true kernel-to-kernel ratio worse than the
headline, and it is why small shapes are ±10% noisy regardless of the kernel.

## Where the remaining gap is

Attribution is **stale** — it predates exp_05 and exp_13. Re-run
`tools/reattribute.sh <expected_shape6_full_us> 8` (it forces a rebuild of the
ablation arms and asserts freshness; it silently reported a stale table twice
before that guard existed).

Last known, shape 6 total 1777.9 µs: **GEMM 1142.8**, XGMI 412.4, sync 191.0,
release 134.9, reduce 86.5. The reading that matters: **our GEMM alone is ~84%
of rank-1's entire device time**, and egress already runs at ~284 GB/s against
315-336 achievable. So the gap is **not** bandwidth and **not** the shape table
— it is that ~635 µs of non-GEMM work is exposed rather than hidden, plus a
mainloop still 2.5-3× off single-CU peak.

Closed axes, do not re-litigate without new evidence: E1(a) AGPRs (unified
register file — re-classing cannot add registers, and the spill is not in the
k-loop anyway); E1(c) scheduling directives (`sched_group_barrier` and
`sched_barrier` are compiler no-ops here — **data-dependence anchors move
MFMAs, directives do not**); LDS-staging peer packets for coalescing (fabric
already at 1.0003× amplification, 99.9% full-64 B); allocation granularity;
dual-role reducers (removes a producer wave on no scored shape).

## Correctness machinery

Ladder: `tools/gate_ladder.sh -ArgLine "<exp>"` — M0 node-clean → M1 build →
M2 resources/ISA → M3 **17 shapes at both `1e-2` and `2e-3`** → M4 three
negative controls → M5 600-epoch soak → M7 timing. Plus **M9 `m9_stale_slot`**
for anything touching publication order: NaN-poisons the heap between epochs,
compares bitwise against a frozen golden, and includes `CTRL_PUBLISH_EARLY`,
which fails on 6 of 6 shapes and is concentrated at epoch 1.

Load-bearing and easy to "simplify" by mistake: **`m3::acquire_frags`**. A bare
`s_waitcnt lgkmcnt(0)` has no operands, creates no data dependence, and the
scheduler hoisted MFMAs above it — silently corrupting 14 of 17 shapes with
`max|diff|` up to 9e30 and **no error bit set**.
`experiments/exp_03_mainloop/lds_race_check.sh` catches the class in ~30 s (it
throws ~10 false positives after loop rotation; read them, do not relax it).

## Traps that cost real time

- **`harness/submission.py` is a node-only COPY of `hk_submission.py`**, and
  every `mp_*` harness imports `submission`. Edit without recopying and you
  measure the old code — which looks exactly like a clean null result.
- **`push.ps1` scps kernel sources too**, so running it while another arm is
  mid-edit ships broken code or silently reverts a validated kernel while
  leaving `build/*.so` intact. It also **resets every source mtime**, so
  "`.so` newer than source" is not a freshness test. Verify by behaviour.
- **`exp_ablation.py` skips compilation when an arm `.so` exists** ("already
  built") and then reports the *previous* kernel's attribution.
- **A gate that has never failed is not evidence.** `m2_report.sh` ran
  `set -uo pipefail` without `-e` and exited 0 with its inputs missing.
- **Reap stale GPU processes first** (`tools/reap_stale.sh`): they are invisible
  to `ps | grep` and immune to a host-side `kill` (root inside the container).
- rank-1 needs `AMDGCN_USE_BUFFER_OPS=0` — Triton 3.6.0 lowers its peer stores
  to `buffer_store_dwordx2` whose voffset is 32-bit, truncating a −4.4-billion
  element offset. Clear `TRITON_CACHE_DIR` when changing that knob.
- rank-1's `heap_bases_*.pkl` probe **can never fire** — `CREATE_SHEMEM_CODE`
  appears once, at its own definition, and the submission spawns no processes.
  Three sessions gated on it.

## Reproduction

```bash
powershell -File .../tools/push.ps1
powershell -File .../tools/nsh.ps1 -Script .../tools/gate_ladder.sh -ArgLine "exp_NN"
# same-run vs rank-1:
powershell -File .../tools/nsh.ps1 -Script .../experiments/exp_13_cta_split/vs_rank1.sh
```

## Ownership

Ours: this `overnight/` tree and the four MI300X kernel sources.
`include/**` is shared HipKittens — **never edit**, copy helpers into
`gemm_rs_mi300x_hk_adapter.cuh`. `gemm_rs_device_tile.cpp` is the read-only
gfx950 donor. rank-1's submission is hash-frozen; patch a copy via
`tools/patch_rank1.py`. Branch `GEMM-RS` in the dedicated worktree.

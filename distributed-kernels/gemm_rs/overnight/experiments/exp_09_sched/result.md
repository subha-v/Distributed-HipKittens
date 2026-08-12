# exp_09_sched — E1(c), mainloop instruction scheduling

**Verdict: the axis is essentially flat, and the reason is a compiler fact worth
keeping.** Scheduling *directives* do not move MFMAs in this translation unit —
proven twice, with instruction-count-identical ISA. Source restructuring does
move them, and I got the schedule I asked for (all 64 MFMAs above the commit's
`vmcnt(0)`, in a single basic block), but the resulting time is inside the noise
floor. One arm ships: **`b_setprio`, 0.997× the base geomean in an alternating
paired A/B**, driven by a single unambiguous −3.1% on shape 2 and partly
cancelled by a real +0.7% on shape 3. Shapes 5 and 6 — the two the experiment
was aimed at — did **not** move outside noise on any arm.

Shipped source sha256:
- `gemm_rs_mi300x.cpp` `225a05c0e0b26afae5ce933084cd2928b799053d0e40ad7f60448c1bcbfa8277`
- `gemm_rs_mi300x_hk_adapter.cuh` `e9b3d1c17d49ee0b7666c63e46134818284049bbc97022ac319a75b1ed1356ed`

Denominator: previous validated best `77.88 / 91.18 / 90.25 / 200.33 / 644.39 /
1828.89`, geomean **230.84 µs**. Reproduced tonight from the archived base
sources at **230.80 µs** (0.02%), so the denominator is sound.

---

## 1. Arms, in the order they were run

| arm | mechanism | gates | s1 | s2 | s3 | s4 | s5 | s6 | geomean | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| base | — (archived, rebuilt tonight) | full ×2 | 77.64 | 91.44 | 89.91 | 200.36 | 642.90 | 1838.61 | **230.80** | 1.000 |
| `a0_sgb_only` | `sched_group_barrier` MFMA/VMEM interleave | ISA only | — | — | — | — | — | — | — | ISA unchanged |
| `a1_peel` | peel the last k-iteration | **M2 FAIL** | — | — | — | — | — | — | — | 30 VGPR spills |
| `a1b_branchless` | clamped unconditional prefetch | full | 78.05 | 91.93 | 90.52 | 199.86 | 656.16 | 1765.50 | 230.61 | 0.999 |
| `a2_fence_mfma` | + `sched_barrier(0x7F6)` | full | 77.51 | 91.38 | 90.94 | 198.34 | 652.61 | 1787.33 | 230.26 | 0.998 |
| `a3_acc_anchor` | + accumulator data-dependence anchor | full | 77.32 | 91.21 | 89.88 | 202.72 | 637.41 | 1812.68 | 230.11 | 0.997 |
| `b_setprio` | + `s_setprio(1)` over the MFMA block | full ×2 | 77.97 | 88.46 | 90.17 | 201.43 | 636.70 | 1828.84 | **229.44** | 0.994 |
| `c_setprio_only` | `s_setprio` alone on base | full | 77.79 | 91.55 | 91.75 | 201.37 | 649.92 | 1840.06 | 232.35 | 1.007 |

Every gated arm: M0 clean → M1 built → M2 6/6 rows → **M3 17/17 shapes at both
`1e-2` and `2e-3`** → M4 all three controls failed as designed → M5 600 epochs
exact → M7. No arm was shipped ungated and no tolerance was touched.

### M2 resource tuples (VGPR / AGPR / SGPR / scratch)

| instantiation | base | a1_peel | a1b | a2 | a3 | b_setprio |
|---|---|---|---|---|---|---|
| 32/64/64 t=0 | 91/0/106/0 | 91/0/·/0 | 91/0/106/0 | 91/0/106/0 | 91/0/106/0 | 91/0/106/0 |
| 64/64/64 t=0 | 91/0/106/0 | 103/0/·/0 | 92/0/106/0 | 92/0/106/0 | 93/0/106/0 | 91/0/106/0 |
| 128/256/32 t=1 | 165/0/106/0 | 171/0/·/0 | 163/0/106/0 | 163/0/106/0 | 163/0/106/0 | **163**/0/106/0 |
| 256/256/32 t=0 | 246/0/106/0 | **256/0/·/spill 30** | 246/0/106/0 | 246/0/106/0 | 246/0/106/0 | 246/0/106/0 |
| 256/256/32 t=1 | 248/0/106/0 | **256/0/·/spill 29** | 248/0/106/0 | 248/0/106/0 | 248/0/106/0 | 248/0/106/0 |
| 32/64/64 t=1 | 91/0/106/0 | 91/0/·/0 | 91/0/106/0 | 91/0/106/0 | 91/0/106/0 | 91/0/106/0 |

The shipped arm keeps the 256/256/32 pair at 246/248 against the 256 cap with
**zero spills and zero scratch bytes**, and takes 128/256/32 from 165 down to
163. `AGPRs: 0` everywhere, unchanged — E1(a) is still on the table.

---

## 2. What the ISA says, arm by arm

`sched_isa.py` finds the k-loop by LLVM's `Depth=` block annotations, so it
survives edits. Counts below are per k-iteration for `<256,256,32,false>`
(shape 5's instantiation; shape 6 uses `<256,256,32,true>`).

### base — six basic blocks, and that is the whole story

```
.LBB3_86  header          term=s_cbranch_scc1 .LBB3_88     <- skips the issue
%bb.87    4x global_load_dwordx4                            <- ISSUE, alone
.LBB3_88  12x ds_read, lgkmcnt(0), 8x mfma,
          (ds_read + 4x mfma) x6, 6x ds_read, lgkmcnt(0),
          32x mfma           term=s_cbranch_vccnz .LBB3_90  <- skips the commit
%bb.89    vmcnt(0), 4x ds_write, lgkmcnt(0),
          vmcnt(0), 4x ds_write, lgkmcnt(0)                 <- COMMIT
.LBB3_90  s_barrier
```

`<256,256,32,true>` is **ten** blocks: the `k == k_iters - 1` compare for the
K-tail mask splits the MFMA run a second time, and shape 6 pays that on all 116
of its k-iterations for a tail that only the last one can take.

### a0 — `sched_group_barrier` alone: zero effect

Asked for `(8 MFMA, 1 VMEM read, 6 DS read, 8 MFMA) × 4`. Result: same 6/10
blocks, same run-lengths (`8,4,4,4,4,4,4,32`), same resource tuples, same
everything. Two independent reasons, both confirmed here:

1. **The VMEM reads are in a different basic block from the MFMAs.** A
   scheduling region never spans a basic block, so a request for VMEM groups is
   unsatisfiable in the region where the MFMAs live.
2. **Every memory op in this tree is `asm volatile ... : "memory"`** —
   `load_global_vec4_async`, `ds_read`, `ds_write` and every `s_waitcnt`
   (`include/cdna3/ops/warp/memory/util/util.cuh:131`). Their mutual program
   order is fixed no matter what the scheduler is told, so "spread the global
   loads through the MFMA block" is not something a sched builtin can express
   here at all. VMEM issue position is set by **source placement**, period.

### a1 — peeling the last iteration: rejected at M2

Peeling removes both `if (more)` and the K-tail compare from the steady state,
which is exactly the right idea, but duplicating the MFMA block put both copies'
fragment addressing in one live range: **256 VGPRs, 30 spills, 50
`scratch_store` / 51 `scratch_load`**, with the mainloop reduced to
`scratch_load_dword; s_waitcnt vmcnt(0); ds_write_b64` eight times over. Not
timed — a spilling arm is not a candidate.

### a1b — clamped unconditional prefetch: 6 blocks → **1**

Same branches removed, no duplicated code. `kn = (k + 1 < k_iters) ? k + 1 : k`
makes the prefetch a select instead of a branch; the commit is then
unconditional too, writing `As[(k+1)&1]`, which on the last iteration is the
buffer iteration `k-1` read and nothing reads again.

```
.LBB3_86  188 instr, ONE block, term=s_cbranch_scc0 .LBB3_86
   4x global_load_dwordx4 | 12x ds_read | lgkmcnt(0)
   8x mfma | (ds_read + 4x mfma) x6 | 6x ds_read | lgkmcnt(0)
   1x mfma | s_waitcnt vmcnt(0)  <---- only 33 of 64 MFMAs above the commit
   7x mfma | 4x ds_write | lgkmcnt(0) | vmcnt(0) | 3x ds_write
   1x mfma | ds_write | 1x mfma | lgkmcnt(0) | 1x mfma
   s_barrier | 21x mfma          <---- 21 MFMAs sunk past __syncthreads()
```

`<256,256,32,true>` went 10 blocks → 5 and the loop rotated, interleaving the
commit's `ds_write`s with MFMAs.

**The cost is visible and it is prefetch coverage.** In base the block boundary
forced all 64 MFMAs above the commit's `vmcnt(0)`; in one region the scheduler
sank 31 of them below it, so the global load has 33 MFMAs of cover instead of
64. Shape 5 (`false`, now single-block) went 642.90 → 656.16 while shape 6
(`true`, 10 → 5 blocks) went 1838.61 → 1765.50: both effects present, opposite
signs, shape-dependent.

Sinking MFMAs past the `s_barrier` is legal and not a bug — those MFMAs read
registers already filled by `ds_read` + `lgkmcnt(0)` before the barrier, so the
next iteration overwriting that LDS buffer cannot reach them. This is precisely
what the `acquire_frags` anchor discipline buys.

### a2 — `sched_barrier(0x7F6)`: also zero effect, and here is the proof

Mask `0x7F6` = every class **except** MFMA may cross (`0x001` deliberately
unset, since "all ALU" includes MFMA). Placed immediately before the commit, so
all 64 MFMAs should have been held above the `vmcnt(0)`.

`14_diff_arms.sh a1b_branchless a2_fence_mfma`, comments/labels stripped:

```
instructions: a1b_branchless=17613  a2_fence_mfma=17613
differing instruction lines: 1156
< v_mov_b32_e32 v29, v85      > v_mov_b32_e32 v17, v85
```

Identical instruction count; every difference is the register allocator picking
different physical registers. Block counts (1 and 5), MFMA counts, and the
commit-wait position are unchanged. The 230.26 vs 230.61 geomean gap between
two byte-equivalent instruction streams is a **direct measurement of the
harness noise floor**, and it is bigger than most deltas in this table.

### a3 — accumulator anchor: the schedule I asked for, at last

`acc_anchor(C_accum)` before the commit — one `asm volatile("" : "+v"(...))` per
accumulator base tile. A `"+v"` tie is a real def, so every MFMA writing that
tile must precede it by **data dependence**, and being volatile it cannot be
reordered against the commit's volatile waits. Same mechanism as the
load-bearing `frag_anchor`; emits no instructions.

```
   4x global_load_dwordx4 | 12x ds_read | lgkmcnt(0)
   8x mfma | (ds_read + 4x mfma) x6 | 6x ds_read | lgkmcnt(0)
   32x mfma                       <---- 8 + 24 + 32 = ALL 64 above the commit
   s_waitcnt vmcnt(0) | 4x ds_write | lgkmcnt(0) | vmcnt(0) | 4x ds_write
   s_barrier
```

Still one basic block, still 246/248 VGPRs, zero spills. `<256,256,32,true>`
likewise puts all 64 above its `vmcnt(0)`. So full prefetch coverage **and** a
single-region body — and the geomean moved from 230.61 to 230.11, i.e. nothing.
Shape 5 did improve to 637.41 (its best of any arm) and shape 4 got worse; they
cancel.

**Conclusion on directives: use anchors, not hints.** `sched_group_barrier` and
`sched_barrier` were both no-ops; a data dependence worked on the first try.

### b_setprio — the one that ships

`s_setprio(1)` before the operand reads, `s_setprio(0)` after `acc_anchor`. The
window needs no hint to stay honest: MFMAs cannot leave through the top (they
depend, via `acquire_frags`' anchor, on `ds_read`s that follow the `setprio`,
and both have side effects) and cannot leave through the bottom (`acc_anchor`
ties the accumulator). ISA: **12 `s_setprio` = 2 per instantiation × 6**, at the
intended positions. Resources unchanged; 64/64/64 back to 91 VGPRs.

### c_setprio_only — `s_setprio` on base: does nothing

Same two lines on the unmodified base. 232.35 µs, and shape 2 stayed at 91.55.
So **the shape-2 win is an interaction with the restructured body, not an
independent property of `s_setprio`.** Mechanistically plausible: on base the
MFMA block and the commit are separate basic blocks, whereas in the
single-region body the whole iteration is one instruction stream and priority
actually decides which of the two waves per SIMD wins the issue port during the
long MFMA run. Row 2 is 64/64/64 at 91 VGPRs — the highest-occupancy row, hence
the most waves contending.

---

## 3. Paired A/B, alternating, both states rebuilt tonight

`15_paired.sh b_setprio 2` — build candidate, bench, build base, bench, twice.

```
  arm    rep         s1        s2        s3        s4        s5        s6     geomean
  cand   1        77.69     88.77     90.77    201.49    638.49   1821.68      229.65
  cand   2        77.85     88.77     90.86    201.63    643.46   1841.87      230.52
  cand   avg      77.77     88.77     90.81    201.56    640.97   1831.78      230.09
  base   1        77.47     91.50     90.00    200.98    646.94   1842.58      231.22
  base   2        77.81     91.38     89.83    199.74    638.85   1834.64      230.38
  base   avg      77.64     91.44     89.91    200.36    642.90   1838.61      230.80

  per-shape candidate/base : 1.0017  0.9709  1.0100  1.0060  0.9970  0.9963
  ratio                    : 0.9969  (<1 is faster)
  worst within-arm spread  : 1.27%
```

Pooling every M7 run of the night (4 for the candidate, 6 for the non-setprio
states) separates signal from noise cleanly:

| shape | candidate runs | other runs | verdict |
|---|---|---|---|
| 2 | 88.46, 88.77, 88.77, 88.49 | 91.21 … 91.93 | **−3.1%, no overlap, real** |
| 3 | 90.17, 90.77, 90.86, 90.38 | 89.83, 90.00 | **+0.7%, real regression** |
| 5 | 636.70 … 655.13 (2.9% spread) | 638.85, 646.94 | noise |
| 6 | 1821.68 … 1841.87 | 1834.64, 1842.58 | noise |

Shape 2 carries 1/6 of the log-geomean, so −3.1% there is worth ~0.5%, and
shape 3's +0.7% gives back ~0.12% — which is the 0.3% observed. The arithmetic
closes.

**Shape 3's regression has an identified cause and a known fix.** The clamped
prefetch issues one redundant tile read per tile, which is `1/k_iters` of the
mainloop's global traffic. Shape 3 has `k_local = 360`, `BK = 32`, so
`k_iters = 12` — the redundant read is **8.3%** of its loads, against 0.9% for
shape 6 (`k_iters = 116`). Whoever picks this up: guard only the *commit* (a
block split after all the MFMAs is harmless — it is what pinned them in base)
and add a single `wait_vmcnt0()` after the loop so no load is in flight when the
allocator reuses `abuf`'s registers in the epilogue.

## 4. Confirmation of the shipped state

`17_ship.sh b_setprio` restored the arm from its archive and re-ran the whole
ladder: M2 6/6 with 246/248 and no spills, M3 **17/17 at both tolerances**, M4
3/3 controls failed as designed, M5 600 epochs exact, M7 `78.10 / 88.49 / 90.38
/ 200.12 / 655.13 / 1841.40`, geomean **230.71 µs**. Source sha256 on the node
matches the archived arm and the Windows tree.

Four independent M7 runs of the shipped arm: 229.44, 229.65, 230.52, 230.71
(mean 230.08). Three of base: 231.22, 230.38, plus the 230.84 on record.

## 5. Methodology notes paid for tonight

- **The race checker is layout-order, not CFG-order, and loop rotation breaks
  it.** `lds_race_check.sh` went 0 → 10 hazards on `a1b` and every one is a
  false positive. Two independent causes: it counts a VALU **destination** as a
  read (`v_add_u32_e32 v130, s51, v196` was reported as "reads v130"), and its
  linear scan misses a drain that rotation moved to the *top* of the rotated
  body, so on the exit path it never sees the `s_waitcnt vmcnt(0)` that every
  execution path actually passes through. Signature of a false positive: the
  flagged register is redefined at the flagged line, and the loop's `vmcnt(0)`
  dominates the loop exit. Verified by hand with `11_haz_window.sh`; the ISA
  windows are archived. **Do not "fix" this by relaxing the checker** — the real
  hazard class it exists for is still real.
- **CRLF, not staleness, was behind every "transient" failure.** Four builds
  died with `syntax error: unexpected end of file` (a `\r` on a heredoc's `PY`
  terminator) or `$'\r': command not found`, at a different line each time,
  looking exactly like compile failures. `push.ps1`'s normalizing `sed` does not
  reliably win: `harness/build.sh` was CRLF at the moment the paired run's
  restore build ran, and LF by the time it was inspected. `nsh.ps1` launders the
  script it transports, so only the scripts *that one docker-execs* are exposed.
  `run_ladder.sh` now normalizes and then **proves it** with `bash -n` inside
  `dhk-gemmrs` before touching a build. Any node-side script that builds must do
  the same.
- **A failed restore can leave the binary and the source disagreeing.** The
  paired run's final restore build failed silently-ish, leaving the base `.so`
  under the candidate's source — the exact state that makes the next measurement
  meaningless. Always rebuild-and-verify after a source swap.
- **Six wedged `spawn_main` workers were holding GPU 1 at 0% CU occupancy** when
  this experiment first tried to gate, and did not drain in eight minutes.
  `tools/reap_stale.sh` identified them as ours and SIGTERM cleared all six.
  `run_ladder.sh` now waits 90 s for a genuine drain before reaping.
- **Two arms with byte-equivalent ISA differed by 1.2% on shape 6.** That is the
  cleanest noise estimate this harness has produced. Any single-run claim below
  ~1.3% on shapes 5/6, or below ~0.5% on the geomean, is not evidence.

## 6. What this closes and what it opens

**Closed.** E1(c) as scoped. Interleaving VMEM issue into the MFMA block is not
expressible through sched builtins here (volatile asm ordering), restoring full
prefetch coverage is worth nothing measurable, and the MFMA/VMEM interleave is
not what the mainloop is limited by. Two directive-based arms lost to their
control; per the kill rule the *directive* approach is closed. Anchors still
work and are the tool of choice.

**Open, and better motivated than before.** The mainloop is 64% of shape 6 and
is not issue-scheduling-bound, so the remaining candidates are the ones that
change the *work*, not its order:

- **E1(a) AGPR accumulators.** `AGPRs: 0` on every instantiation, with the
  256/256/32 pair pinned at 246/248 of 256 VGPRs. Moving `C_accum` to AGPRs is
  the one change that could raise occupancy above 1 CTA/CU, and 2 waves/SIMD
  contending for one issue port is now a *measured* effect — that is what arm B
  exploited for 3% on the highest-occupancy row.
- **E1(d) `v_mfma_f32_32x32x8` vs `16x16x16`.** Every MFMA in the loop is
  `16x16x16_bf16`; 64 of them per k-iteration at 4-cycle issue is a lot of issue
  slots for the same math.
- **The `ds_read` side.** 24 `ds_read_b64` per k-iteration, drained by two full
  `lgkmcnt(0)`s. That is the other half of the mainloop and nothing tonight
  touched it.

Arm C (counted `s_waitcnt`) was **not run**. It was gated on A or B winning
decisively, and none did; with the whole axis inside a 1.3% noise floor, a
mechanism whose documented failure mode is a 30× numerical error that still
passes a 2e-2 gate is not worth the risk for a delta this size.

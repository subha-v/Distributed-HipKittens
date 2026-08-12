# exp_26 — design: vendor the M6 body, scale phase 1's `sched_group_barrier`
# hints by `kGM`

**Status:** built and disassembled, **not run on GPU** (another agent holds the
node tonight). Compile-only evidence; the activation patch is prepared, not
applied, in `activate.md`.

**One variable:** the four `__builtin_amdgcn_sched_group_barrier` counts in
phase 1's unrolled K-loop, hardcoded for `kGM == 1` in the donor, replaced by
the exact per-half instruction census parameterized in `kGM`. No math, no
buffers, no protocol, no resource request changes.

---

## 1. Donor provenance (the anchor)

| | |
|---|---|
| authoritative donor | `/home/subvadla/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip/n2_phase1_gm.cpp` |
| **sha256** | **`1d90b26658b6a524db69434ddcfa0b2dcec2418c852f83874468d55dd0197dc2`** |
| size / lines | 23,662 B / 585 lines |
| located | `gbt350-odcdh2-c05-1`, 2026-08-11 |
| identical copies on the node | `experiments/exp_59_large_m_tiles/source_snapshot/n2_phase1_gm.cpp`, `experiments/exp_63_large_m_rerun/source_snapshot/n2_phase1_gm.cpp` (same sha256) |
| **not** identical | `experiments/exp_65_phase1_accumulator_pressure/source_snapshot/n2_phase1_gm.cpp` (`52ecd9e7…`, 634 lines) — a different experiment's fork, do not use |
| local mirror | `.node/mps_n2_phase1_gm.cpp` — 24,247 B on the Windows checkout, **byte-identical to the node donor after CRLF→LF normalization** (23,662 B, same sha256). The 585-byte delta is exactly the 585 line terminators. **The mirror is trustworthy; there is no discrepancy to report.** |
| consumed by | `k0pf6gm_device_tile.hip:395`-equivalent and `k0pf6gm_device_tile_mps.hip:395` (`#include "n2_phase1_gm.cpp"`), both arms, today |

Vendored copy: `distributed-kernels/fused_moe/n2_phase1_gm_mps.cpp`, 697 lines.
Diff vs donor: **+112 / −0**. Not one donor line is modified or deleted; every
delta is an insertion. See `build.md` §1.

---

## 2. The instruction census, derived and then verified against the ISA

`kGM = N2GM_G = K0P6GM_G = 3` is pinned (`k0pf6gm_device_tile.hip:152-157`
`#error`s any other value; `BUILDING.md:41` passes `-DK0P6GM_G=3 -DN2GM_G=3`).

Both halves of the unrolled K-loop (donor `:326-330` and `:339-343`) call
**exactly one** `read_a`, `load_a`, `load_b`, `mfma_k`, `store_a` each, with the
same shapes and only the LDS/B buffer indices differing. **The two halves'
censuses are therefore identical**, which is why the same four expressions apply
at both sites. This was checked line by line, not assumed.

### 2.1 Derivation from source

| class | mask | issuing site | derivation | count |
|---|---|---|---|---|
| DS read | `0x100` | `read_a` (donor `:248-262`) | `kGM` × `m(2)` × 2 `float4` → `ds_read_b128` | **`4·kGM`** |
| VMEM read | `0x020` | `load_b` (`:263-276`) + `load_a` (`:234-240`) | `load_bfrag` is 2 loads (`n2_device_common.cuh:55-59`: `d[0]=*p; d[1]=*(p+1024)`); `load_b` = `kColTiles(4)` × gate/up(2) × 2 = 16, all **`kGM`-independent** (weights are loaded once and reused across sub-blocks — the G-stack's whole point). `load_a` = `kGM`. | **`16 + kGM`** |
| MFMA | `0x008` | `mfma_k` (`:278-309`) | `kGM` × `m(2)` × `gu(2)` × `kColTiles(4)` | **`16·kGM`** |
| DS write | `0x200` | `store_a` (`:241-247`) | `kGM` × `ds_write_b128` | **`kGM`** |

### 2.2 THE ACCEPTANCE CRITERION — every expression equals the donor literal at `kGM = 1`

| class | mask | donor literal (`:331-334`, `:344-347`) | new expression | at `kGM=1` | match? | at `kGM=3` |
|---|---|---:|---|---:|:--:|---:|
| DS read | `0x100` | `4` | `4 * kGM` | **4** | ✅ | 12 |
| VMEM read | `0x020` | `17` | `16 + kGM` | **17** | ✅ | 19 |
| MFMA | `0x008` | `16` | `16 * kGM` | **16** | ✅ | 48 |
| DS write | `0x200` | `1` | `kGM` | **1** | ✅ | 3 |

All four reproduce the donor exactly at `kGM = 1`. **That is what makes this a
parameterization rather than a retune**: at the value the donor was written for,
the emitted hints are bit-identical, so no tuning decision is being second-
guessed — only the arithmetic the donor forgot to write down.

The `17 = 16 + 1` row is the load-bearing one. A literal `17` is not a round
number; it can only be `1 (load_a) + 8 fragments × 2`, which pins both the
`load_bfrag` = 2-instruction fact and the `kGM`-independence of the 16.

### 2.3 Verified against the emitted ISA (not just reasoned)

`llvm-objdump` of the phase-1 K-loop (one iteration = 2 halves), from the
default (donor-schedule) build:

```
     96 v_mfma_f32_16x16x128_f8f6f4     = 2 x 48    = 2 x 16*kGM     ✅
     36 ds_read_b128 + 4 ds_read_b32    = 2 x 20                     (see below)
     32 flat_load_dwordx4               = 2 x 16    = 2 x (load_b)   ✅ kGM-independent
      6 buffer_load_dwordx4             = 2 x 3     = 2 x kGM        ✅ (load_a)
      6 ds_write_b128                   = 2 x 3     = 2 x kGM        ✅ (store_a)
      2 s_barrier                       = the two lds_cta_barrier()
```

The `16 + kGM` split is **directly visible as two different instruction forms**:
`load_b`'s 16 weight fragment loads lower to `flat_load_dwordx4`, `load_a`'s
`kGM` A-tile loads lower to `buffer_load_dwordx4` (the bounded `a_rsrc`
descriptor). Nothing here is inferred.

### 2.4 The `ascale_lds` / `b1s_lds` DS reads — excluded, and why

`mfma_k` also issues, per half: `2·kGM = 6` `ds_read_b128` from `ascale_lds`
(`:283-286`) and `2` `ds_read_b32` from `b1s_lds` (`:279-280`). The ISA confirms
both: 18 `ds_read_b128` per half = 12 (`read_a`) + 6 (`ascale`), plus 2
`ds_read_b32` (`b1s`) = 20 DS reads per half, 40 per iteration.

**They are deliberately left out of the `0x100` group.** Two reasons, and the
first is decisive:

1. **The acceptance criterion forbids it.** Including them gives `6·kGM + 2`,
   which is **8** at `kGM = 1` against the donor's literal **4**. That is a
   retune, not a parameterization, and it would make the default build differ
   from the donor.
2. **The donor's own convention agrees, in both phases.** Phase 2's `0x100` hint
   is `4 * kGM` (`n2_phase2_gm.cpp:358`) and likewise counts only `read_a2`,
   leaving its `dq2_lds`/`b2s_lds` reads inside `mfma_k` unhinted. The
   `sched_group_barrier` group is a *pipeline-stage* hint over the A-tile
   double-buffer, not an exhaustive instruction inventory.

Folding them in is a separate, testable idea. It is **not** this experiment.

### 2.5 Bonus finding: phase 2's VMEM hint is over-scaled (donor bug, not ours)

The premise "phase 2 scales three of its four correctly" is only 3/4 true.
Phase 2 uses `8 + 7 * kGM` for `0x020`, = 15 at `kGM=1` (correct: 14 W2 + 1 A2)
but **29 at `kGM=3`**. The ISA says phase 2's real per-half VMEM count is **17**
(`load_w2` 7 fragments × 2 = 14, `kGM`-independent, plus `load_a2` = `kGM` = 3).
The correct form is `14 + kGM`. Phase 2 over-requests by 12 instructions, i.e.
its hint asks the scheduler to place 29 VMEM reads in a group that only has 17
members.

**We did not copy that error.** Phase 1's `16 + kGM` scales only the term that
actually depends on `kGM`. Logged here as a candidate for a later single-
variable experiment on the phase-2 side; **out of scope tonight** (one variable
per experiment, and phase 2 is another agent's file this week).

---

## 3. Why this is worth doing — the mis-steering at `G=3`

At the shipped `kGM = 3` the donor asks for `4 / 17 / 16 / 1` against a real mix
of `12 / 19 / 48 / 3`. Two thirds of the DS reads, two thirds of the MFMA and
two thirds of the DS writes fall outside every hinted group and are scheduled
freely.

The consequence is measured, not argued — see `build.md` §4. In the default
build the two `s_barrier`s partition the K-loop's 96 MFMAs **33 / 49 / 14**
instead of the intended 48 / 48 / 0, and the loop's single
`s_waitcnt vmcnt(0)` lands at **`mfma = 0`**, i.e. the wave drains every
outstanding VMEM load with zero MFMA work of the iteration issued. With the
`kGM`-scaled hints the partition becomes **48 / 48 / 0** and the drain moves to
`mfma = 48`.

Context for the size of the prize (`m6_m7_structure.md` §2.4/§5.3): the K-step
costs ~8,992–9,800 cycles against 1,536 cycles of MFMA issue at `G=3`, and the
software pipeline is exactly one K-step deep, so where the wait sits relative to
the MFMA is the entire scheduling question in this loop.

---

## 4. The four instrumentation hooks (no behaviour, no wiring)

Empty by default, per `m6_m7_structure.md` §6.3 Tier 2. All four sit outside
both MFMA spans, so they cannot perturb the K-loop even when populated.

| macro | donor line | placed | splits out |
|---|---:|---|---|
| `N2GM_P1_TASK_HEAD_HOOK` | `:161` | top of the task loop, before `const int tile = …`; nothing executes between the loop's `{` and here | the 344 KB `ascale_lds` scatter-gather prologue (§5.1) |
| `N2GM_P1_KLOOP_ENTER_HOOK` | `:317` | after `lds_cta_barrier()` (`:316`), before `#pragma unroll 1 / for (k…)` | first MFMA |
| `N2GM_P1_KLOOP_EXIT_HOOK` | `:351` | immediately after the K-loop's `}` (`:350`) | last MFMA — the K-loop's own duration |
| `N2GM_P1_EPILOGUE_DONE_HOOK` | `:473` | after the `A2q`/`DQ2` stores (`:454`, `:472`), before the task-end `__syncthreads()` (`:474`) | SiLU + amax + quant + store |

All four line numbers were re-derived from the donor text and match §6.3.
Convention follows `N2GM_TASK_DONE_HOOK`: bare use at the site, the definition
carries its own trailing semicolon, the includer `#undef`s after the `#include`.

`N2GM_P1_EPILOGUE_DONE_HOOK` is deliberately **not** a VMEM drain: the `A2q`
and `DQ2` stores are still in flight at that point, exactly as in the donor. A
future timestamp there measures store *issue*, not store *completion*. If a
later experiment needs completion semantics it must add its own drain and say
so — phase 2's `N2GM_TASK_DONE_DRAIN_HOOK` is the precedent.

---

## 5. Alternatives rejected

| alternative | why rejected |
|---|---|
| **Edit the donor `n2_phase1_gm.cpp` in place** | Read-only upstream; shared with the `pf6gm_mega` reference arm, so an in-place edit would move the denominator and destroy the A/B. Vendoring is the recorded pattern (`DESIGN_MPS.md:258-272`, `PROVENANCE.md:83-91`). |
| **Unconditional `kGM` scaling (no macro gate)** | Couples "vendor the file" with "change the schedule" into one irreversible step. The gate makes the default build provably donor-identical (`build.md` §3) and turns activation into a genuine single-variable A/B. |
| **Runtime selection from `K0_MPS_CFG`** | **Impossible, not merely expensive.** `__builtin_amdgcn_sched_group_barrier(mask, size, syncid)` takes immediate operands and is a compile-time directive with no runtime representation. Branching over two copies of the K-loop would duplicate ~739 instructions and change register allocation, i.e. it would no longer be the same experiment. See `activate.md` §2. |
| **Include the `ascale_lds`/`b1s_lds` reads in the `0x100` group** | Violates the acceptance criterion (8 ≠ 4 at `kGM=1`) and contradicts the donor's convention in both phases. §2.4. |
| **Also fix phase 2's `8 + 7*kGM` → `14 + kGM` in the same change** | Two mechanisms in one arm. Also `n2_phase2_gm_mps.cpp` is not mine to move tonight. §2.5. |
| **A finer group decomposition (e.g. several small alternating groups)** | A retune with no derivation behind it. This experiment's whole claim to zero risk is that it changes nothing at the value the donor was written for. |
| **Hook the task loop with `N2GM_TASK_START`/`N2GM_TASK_STRIDE` twins while vendoring** | That is the M6 CTA-sweep experiment (`m6_m7_structure.md` §7 item 2). It is a *different* variable and belongs in its own arm; this file's hooks are deliberately observation-only. |

---

## 6. Risk analysis

| risk | assessment |
|---|---|
| **Correctness** | Nil by construction at the source level: `sched_group_barrier` is a scheduling *hint*; it cannot reorder across a barrier, a `s_waitcnt`, or a data dependence, and it cannot change results. The MFMA/LDS/epilogue arithmetic is donor-verbatim (+112/−0 diff). The default-off build is `.text`-byte-identical to the donor build (`build.md` §3), so shipping the vendored file without the flag is a provable no-op. |
| **Register allocation** | **Materialized, and it is the real risk.** GSCALE=1 keeps SGPR 106 / ArchVGPR 256 / AGPR 256 / LDS 155,496 B exactly, and *reduces* the scratch frame 144 → 128 B/lane and VGPR spills 16 → 14. But the number of scratch *instructions* rises 21 → 114, and **99 of the 114 land after the phase-2 K-loop** (M7 epilogue / M8 / M9), up from 3. Both MFMA K-loops stay scratch-free. In the mode-12 ratchet that tail region is the 1,309 µs pool, so a phase-1-only hint change can plausibly lose there what it gains in M6. **Any timing run must read M6 and M8M9 separately** (`cfg` bit 33 on). |
| **Wrong-direction schedule** | Possible. GSCALE=1 adds 3 `s_waitcnt lgkmcnt` per iteration (7 → 10 waits) while removing 13 instructions from the loop. LDS waits are cheap relative to the ~9,000-cycle K-step (§5.3 item 3 puts all LDS traffic at ~170–200 cycles), so this is a small debit against a large credit — but it is a debit. |
| **Bounded upside** | The mechanism moves ≤ 1,536 cycles of MFMA issue in front of the one `vmcnt(0)` per iteration (~18,000–19,600 cycles), so the ceiling is ~8% of M6 ≈ 200 µs, and only if the drain is genuinely exposed. `m6_m7_structure.md` §2.5 argues M6 is L2/LLC-**throughput**-bound (4.91 TB/s achieved vs 28.4 TB/s needed for peak), and rearranging waits adds no bandwidth. Expect a fraction of the ceiling. |
| **JIT cache aliasing** | The classic trap. A `.cpp`-content change *is* hashed once the file is installed into the mori kernels dir, but a `-D`-only change is **not**. `activate.md` mandates the `K0P6_MPS_SRC_REV` bump and prefers a source `#define` over a harness `-D` for exactly this reason. |
| **Harness allowlist** | `n2_phase1_gm_mps.cpp` must be added to `PF6_N2_FILES` or the JIT compile fails outright with `FileNotFoundError`. Loud failure, not a silent one — but it must be in the patch. `activate.md` §1. |

---

## 7. Invariants this change must preserve (all checked in `build.md`)

1. `SGPR 106 / ArchVGPR 256 / AGPR 256 / LDS 155,496 B / occupancy 1 wave/SIMD`.
2. Static MFMA census `96 + 84 = 180`.
3. Zero scratch accesses inside either MFMA K-loop.
4. Default-off build byte-identical to the donor build in `.text`.
5. The `pf6gm_mega` reference arm continues to include the **donor**
   `n2_phase1_gm.cpp` and is not perturbed. (Both files are installed; they have
   different names.)

---

## Primitives

**This experiment uses no `include/cdna4/ops/group/distributed/` primitive, and
that is the honest finding.** It is a compiler-scheduling change inside a
vendored upstream GEMM body — the one region of the megakernel that is pure
local compute with no peer traffic, no readiness poll, no epoch, and no
completion edge (`m6_m7_structure.md` §5.6). There is no protocol here for the
library to express.

Two observations for the ledger anyway:

- **Gap: the library has no vocabulary for intra-CTA pipeline structure.**
  Every primitive in the set describes an edge *between* agents (release,
  acquire, publish, observe, count, retire). The thing that dominates M6 — where
  a `s_waitcnt` sits relative to a run of MFMAs inside one wave — has no
  representation at all. That is arguably correct scoping, but it means the
  primitives-first mandate is silent on ~2,539 µs of the 7,703 µs campaign, and
  no `## Primitives` section from an M6-side experiment will ever be able to say
  otherwise. Tag: `primitives: out-of-scope-by-construction`.
- **The vendoring pattern itself is the reusable artifact**, not a primitive: a
  donor-sha-pinned copy whose default expansion is provably byte-identical to
  the donor, with `#if`-gated deltas and empty hook macros. `n2_phase2_gm_mps.cpp`
  established it; this is the second instance and it now has a stronger gate
  (`.text` hash equality, not just "diff-verified at authoring time"). If a third
  body ever needs it, that gate should be the standard.

---

## Addendum — the gate became a 4-bit mask

Everything above describes `N2GM_P1_SCHED_GSCALE` as a boolean. It is now a
**4-bit mask** (bit 0 DS read, bit 1 VMEM read, bit 2 MFMA, bit 3 DS write), so
each hint is separately A/B-able. The acceptance criterion of §2.2 is unchanged
but now holds **per bit**: each scaled expression equals its donor literal at
`kGM == 1` independently of the other three, so all 16 masks — not only 0 and 15
— are donor-identical at `kGM == 1`.

The reason for the change is in `build.md` §5–§6, and it inverts this document's
working assumption that the four hints act as one knob. They do not:

- **bit 2 alone** produces the entire `33/49/14 → 48/48/0` barrier realignment,
  at a resource tuple byte-identical to the donor's;
- **bit 0 alone** produces the `vmcnt(0)` relocation *and* all 96 migrated
  scratch accesses, which turn out to be a spill-load plus a full VMEM drain in
  front of 96 of the 282 remote bf16 atomics in M8/M9;
- **bit 1 is an exact no-op** at `G=3` — the phase-1 VMEM hint was already right;
- **bit 3** perturbs the code bytes and moves no metric.

The recommended arm is therefore **mask 4**, not 15. Two further deltas were
added since this document was written — `N2GM_P1_TASK_START` and
`N2GM_P1_TASK_STRIDE`, defaulting to the donor's `blockIdx.x` / `kCTAs` — for
exp_25's M6/M7 interleave and for the unmeasured `T6(128)/T6(256)` CTA-scaling
point. They are inert at their defaults, verified at the `.text`-hash level.

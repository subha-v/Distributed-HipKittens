# exp_26 — build note: the 4-bit hint mask, all 16 subsets, and the K-loop ISA read

**No GPU work was done.** Everything below is `hipcc --genco` inside `subha_k1`
plus `llvm-objdump`. `pgrep -af 'run_campaign|torchrun|mpirun'` was empty before
and after; no GPU process was started, stopped, or contended.

> **Headline.** The four hints do *not* act as one knob. Enumerating all 16 masks
> shows the win and the risk sit on **different bits**:
> **bit 2 (MFMA) alone** buys the whole `33/49/14 → 48/48/0` barrier realignment
> at a **byte-identical resource profile to the donor**, while
> **bit 0 (DS read) alone** owns *both* the `vmcnt(0)` relocation *and* all 96
> migrated scratch accesses — and those 96 turn out to be a `scratch_load_dword`
> + a full VMEM drain in front of **96 of the 282 remote bf16 atomics** in M8/M9.
> **Recommended arm: `N2GM_P1_SCHED_GSCALE = 4`, not 15.** See §5 and §6.

---

## 0. Build tree and provenance

All builds in this note come from the **same immutable snapshot as the first
run**, so the first run's B0/B1/B2 numbers and these are directly comparable.
The shared node checkout was never modified.

```
node checkout  ~/Distributed-HipKittens  HEAD 0b82cd1980e0ad171018758a2aee28feaee910f3
snapshot       git archive HEAD | tar -x -C ~/overnight-scratch/e26/dhk      (unchanged since the first run)
  k0pf6gm_device_tile_mps.hip  0c92a1c9713fcf31adfb759336e381f41fb1b8b27022f62c05d10f8ebc1832b3
  moe_mps_adapter.cuh          c84f890d39d816bede5a7aab7a4df28a158c7667095931eae5f4d785921f5de3
  n2_phase2_gm_mps.cpp         08c3c2c02dd4d551a51f3e52e53759d495d0ede2cebe892ad995cd4acfef87c6
  K0P6_MPS_SRC_REV = 23,  phase-1 include confirmed at :395
donor          ~/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip/n2_phase1_gm.cpp
               1d90b26658b6a524db69434ddcfa0b2dcec2418c852f83874468d55dd0197dc2
vendored       distributed-kernels/fused_moe/n2_phase1_gm_mps.cpp
               ede2e7bac4b484c3ce274512be8db6a8e240b2d2b35e3ace18a9e301fd90dbe6   (750 lines)
```

Cross-run continuity check, both directions: the new **`D`** build (donor
include) hashes `.text` to `96049dfa…`, the same as the first run's **B0**; the
new **`M15`** hashes to `0959591a…`, the same as the first run's **B2**. The
toolchain is deterministic and the two notes describe one experiment.

---

## 1. Diff vs the donor — `+174 / −9`

```
$ diff -u  <donor>  <vendored>       # on the node, LF vs LF
 diffstat: +171 -9      (+171 non-blank, +3 blank = 174 insertions)
 line count: 585 -> 750
```

The nine deleted lines are the task-loop header and the two copies of the four
hint lines:

```
-  for (int task = blockIdx.x; task < num_tasks; task += kCTAs) {
-      __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
-      __builtin_amdgcn_sched_group_barrier(0x020, 17, 0);
-      __builtin_amdgcn_sched_group_barrier(0x008, 16, 0);
-      __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);
-      __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
-      __builtin_amdgcn_sched_group_barrier(0x020, 17, 0);
-      __builtin_amdgcn_sched_group_barrier(0x008, 16, 0);
-      __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);
```

The first run kept `−0` by wrapping the donor lines in an `#else`. That does not
survive contact with a per-bit mask: four independent `#if/#else` blocks at each
of two sites is 32 lines of preprocessor wrapped around the eight lines of the
K-loop we are trying to reason about. Instead the counts are resolved **once**,
at file scope, and each `#else` arm is the donor literal verbatim:

```cpp
#if (N2GM_P1_SCHED_GSCALE & 1)
#define N2GM_P1_N_DSREAD (4 * kGM)
#else
#define N2GM_P1_N_DSREAD 4
#endif
// ... same shape for VMEMRD (16 + kGM | 17), MFMA (16 * kGM | 16), DSWRITE (kGM | 1)
```

so both K-loop halves keep the donor's four-line shape:

```cpp
      __builtin_amdgcn_sched_group_barrier(0x100, N2GM_P1_N_DSREAD, 0);   // DS read
      __builtin_amdgcn_sched_group_barrier(0x020, N2GM_P1_N_VMEMRD, 0);   // VMEM read
      __builtin_amdgcn_sched_group_barrier(0x008, N2GM_P1_N_MFMA, 0);     // MFMA
      __builtin_amdgcn_sched_group_barrier(0x200, N2GM_P1_N_DSWRITE, 0);  // DS write
```

**The `−0` property is replaced by something strictly stronger and measured, not
argued: mask 0 compiles to a `.text` section byte-identical to the donor build**
(§4). The MPS-DELTA sites were renumbered 1–10 to keep the header list in file
order after the two task-loop insertions.

| delta | donor line | insertion |
|---|---:|---|
| header | before `:1` | 55-line vendored-copy banner (donor path, sha256, ten-delta list, per-bit acceptance table) |
| (1) | `:73` | `N2GM_P1_SCHED_GSCALE` mask default 0 + `static_assert(0..15)` + the four `N2GM_P1_N_*` group-size macros |
| (2) | `:92` | four `#ifndef`/`#define`/`#endif` empty hook defaults |
| **(3)** | `:92` | **`N2GM_P1_TASK_START` / `N2GM_P1_TASK_STRIDE`, defaulting to `blockIdx.x` / `kCTAs`** |
| **(4)** | `:156` | **the task-loop header, bounds taken from (3)** |
| (5) | `:161` | `N2GM_P1_TASK_HEAD_HOOK` |
| (6) | `:317` | `N2GM_P1_KLOOP_ENTER_HOOK` |
| (7) | `:331` | half 0's four hints, counts from (1) |
| (8) | `:344` | half 1's four hints, counts from (1) |
| (9) | `:351` | `N2GM_P1_KLOOP_EXIT_HOOK` |
| (10) | `:473` | `N2GM_P1_EPILOGUE_DONE_HOOK` |

### Per-bit acceptance

Each scaled expression equals its donor literal at `kGM == 1` **independently of
the other three bits**, so all 16 masks are donor-identical at `kGM == 1`, not
just mask 15. That is what makes the bits separately meaningful.

| bit | mask | class | donor literal | scaled | at `kGM=1` | at `kGM=3` | measured per half |
|---:|---:|---|---:|---|---:|---:|---:|
| 0 | 1 | `0x100` DS read | 4 | `4 * kGM` | 4 ✅ | 12 | 20 (see note) |
| 1 | 2 | `0x020` VMEM read | 17 | `16 + kGM` | 17 ✅ | 19 | **19** ✅ |
| 2 | 4 | `0x008` MFMA | 16 | `16 * kGM` | 16 ✅ | 48 | **48** ✅ |
| 3 | 8 | `0x200` DS write | 1 | `kGM` | 1 ✅ | 3 | **3** ✅ |

The DS-read row is the deliberate exception documented in `design.md` §2.3: the
measured 20 DS reads per half are 12 `read_a` plus 8 issued inside `mfma_k`
(`ascale_lds`, `b1s_lds`), and the donor's hint has always covered `read_a`
only. Folding the rest in would give `6*kGM+2 = 8` at `kGM=1`, contradicting the
donor's literal `4`, so it stays a parameterization rather than a retune.

---

## 2. The builds

Throwaway TUs are **generated on the node, never committed** — `tu/b0` includes
the donor, `tu/b1` includes the vendored body, both carry an identical
`SCAFFOLDING ONLY — NOT A SHIPPING SOURCE` banner, and they differ on exactly
one line:

```
$ diff tu/b0/k0pf6gm_device_tile_mps.hip tu/b1/k0pf6gm_device_tile_mps.hip
403c403
< #include "n2_phase1_gm.cpp"
---
> #include "n2_phase1_gm_mps.cpp"
```

Compile line, inside `docker exec subha_k1` (ROCm 7.2.4 / LLVM 22), identical for
every build except the marked parts:

```bash
hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
  -Rpass-analysis=kernel-resource-usage \
  -I$DHK/include -I$DHK/distributed-kernels/fused_moe \
  -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip \
  -I$MR -I$MR/include -I$MR/src \
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
  [-DN2GM_P1_SCHED_GSCALE=<n>] <TU> -o out2/<V>.hsaco
```

| build | TU | define |
|---|---|---|
| `D` | `tu/b0` (donor) | — |
| `M0` … `M15` | `tu/b1` (vendored) | `-DN2GM_P1_SCHED_GSCALE=0 … 15` |

17 builds, all exit 0, zero errors, ~6 s each, six at a time. Since a build is
6 s there was no reason to sample the six requested subsets instead of
enumerating the whole space, so §5 is the complete truth table.

`--genco` emits a clang offload bundle, **not** a bare ELF; `llvm-objdump` on the
bundle silently writes a one-line "not recognized as a valid object file". Every
disassembly here goes through the unbundler first:

```bash
clang-offload-bundler --type=o --unbundle \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --input=$V.hsaco --output=$V.elf
llvm-objdump -d --mcpu=gfx950 $V.elf > $V.isa
llvm-objcopy --dump-section=.text=$V.text.bin $V.elf /dev/null
```

---

## 3. Resources — invariant across all 16 masks except scratch

`-Rpass-analysis=kernel-resource-usage`, function `k0pf6gm_mps_mega`. Identical
in **every** one of the 17 builds:

| | value |
|---|---:|
| TotalSGPRs | 106 |
| ArchVGPRs | 256 |
| AGPRs | 256 |
| LDS [B/block] | 155,496 |
| Occupancy [waves/SIMD] | 1 |
| SGPR spills | 180 |
| static `v_mfma` census | 180 (96 + 84) |
| scratch ops **inside** either MFMA K-loop | **0** |

Only two fields move, and they move together, and only with **bit 0**:

| | bit 0 clear (masks 0,2,4,6,8,10,12,14) | bit 0 set (masks 1,3,5,7,9,11,13,15) |
|---|---:|---:|
| ScratchSize [B/lane] | **144** (= donor) | 128 |
| VGPR spills | **16** (= donor) | 14 |
| scratch instructions, whole kernel | **21** (= donor) | **114** |

Fewer values spilled, accessed five times as often, in a worse place. §6.

---

## 4. Mask 0 ≡ donor — re-verified, now with the task macros in place

The important gate: mask 0 **plus** the two new `N2GM_P1_TASK_*` macros at their
defaults still produces the donor's code exactly.

| check | result |
|---|---|
| `.text` sha256, `D` (donor include) | `96049dfa4f8cd56eade98d3e2c5e565fcf69f3792570e348380c34cc86b9044c` |
| `.text` sha256, `M0` (vendored, mask 0, task macros defaulted) | `96049dfa…` — **identical**, both 166,656 B |
| `.text` sha256, first run's `B0` | `96049dfa…` — identical across runs |
| resource tuple | identical in every field |
| phase-1 K-loop range, census, wait/barrier timeline | identical |

So `for (int task = (int)(N2GM_P1_TASK_START); task < num_tasks; task += (int)(N2GM_P1_TASK_STRIDE))`
compiles to the donor's `for (int task = blockIdx.x; task < num_tasks; task += kCTAs)`
bit for bit. The two casts (phase 2's convention, present so an override can hand
back something that is not an `int`) are free, as expected: `blockIdx.x` was
already implicitly converted to `int` and `kCTAs` is a `constexpr int`.

`M2` also hashes to `96049dfa…` — see §5, bit 1 is inert.

---

## 5. **The subset table** — all 16 masks

Bit order is `DSW | MFMA | VMEM | DSR` = bits 3|2|1|0. "scratch ≤p2" counts every
scratch instruction from function entry through the end of phase 2's MFMA
K-loop; "after p2" is everything past it — the M7 epilogue, M8 and M9.
"drain@" is the MFMA index, within one phase-1 K-loop iteration, at which the
iteration's single `s_waitcnt vmcnt(0)` sits. "partition" is MFMAs per segment
between the two `s_barrier`s.

| mask | bits set | `.text` | scratch B/lane | scratch total | ≤p2 | **after p2** | **drain@mfma** | **partition** |
|---:|---|---|---:|---:|---:|---:|---:|---|
| **D** | *donor build* | `96049dfa` | 144 | 21 | 18 | 3 | 0 | 33 / 49 / 14 |
| **0** | — | `96049dfa` | 144 | 21 | 18 | 3 | 0 | 33 / 49 / 14 |
| **2** | VMEM | `96049dfa` | 144 | 21 | 18 | 3 | 0 | 33 / 49 / 14 |
| 8 | DSW | `3e656461` | 144 | 21 | 18 | 3 | 0 | 33 / 49 / 14 |
| 10 | DSW+VMEM | `3e656461` | 144 | 21 | 18 | 3 | 0 | 33 / 49 / 14 |
| **4** | **MFMA** | `b0d4e657` | **144** | **21** | **18** | **3** | 0 | **48 / 48 / 0** |
| **6** | **MFMA+VMEM** | `b0d4e657` | **144** | **21** | **18** | **3** | 0 | **48 / 48 / 0** |
| 12 | DSW+MFMA | `7526576e` | 144 | 21 | 18 | 3 | 0 | 48 / 48 / 0 |
| 14 | DSW+MFMA+VMEM | `7526576e` | 144 | 21 | 18 | 3 | 0 | 48 / 48 / 0 |
| 1 | DSR | `214ee45e` | 128 | 114 | 15 | **99** | **48** | 34 / 48 / 14 |
| 3 | DSR+VMEM | `214ee45e` | 128 | 114 | 15 | **99** | **48** | 34 / 48 / 14 |
| **9** | **DSW+DSR** | `eb0630f5` | 128 | 114 | 15 | **99** | **48** | 34 / 48 / 14 |
| 11 | DSW+VMEM+DSR | `eb0630f5` | 128 | 114 | 15 | **99** | **48** | 34 / 48 / 14 |
| 5 | MFMA+DSR | `0bed360c` | 128 | 114 | 15 | **99** | **48** | 48 / 48 / 0 |
| 7 | MFMA+VMEM+DSR | `0bed360c` | 128 | 114 | 15 | **99** | **48** | 48 / 48 / 0 |
| 13 | DSW+MFMA+DSR | `0959591a` | 128 | 114 | 15 | **99** | **48** | 48 / 48 / 0 |
| **15** | all *(= first run's B2)* | `0959591a` | 128 | 114 | 15 | **99** | **48** | 48 / 48 / 0 |

The six masks you asked for are `0`, `2`, `4`, `6`, `9`, `15`; the other ten are
free and make the factorization unambiguous. 16 masks collapse to **8 distinct
`.text` images**, and every metric is a clean function of the bits:

### Bit 0 — DS read, `4 → 12`. Owns the drain move **and** the entire migration.

Exactly the eight masks with bit 0 set have `drain@48`, scratch 114, 99 after
p2, frame 128 B, VGPR spill 14. Exactly the eight without it have `drain@0`,
scratch 21, 3 after p2, frame 144 B, spill 16. There is no mask that separates
them.

### Bit 1 — VMEM read, `17 → 19`. An exact no-op, in all eight pairs.

`M0≡M2`, `M1≡M3`, `M4≡M6`, `M5≡M7`, `M8≡M10`, `M9≡M11`, `M12≡M14`, `M13≡M15` —
byte-identical `.text` every time. Raising the group from 17 to 19 when the half
issues exactly 19 VMEM loads changes nothing the scheduler does, which is a
small piece of good news: the phase-1 VMEM hint was already close enough to
right that the correction is invisible. (Phase **2**'s is not — §7.)

### Bit 2 — MFMA, `16 → 48`. The barrier realignment, for free.

Exactly the eight masks with bit 2 set get `48 / 48 / 0`. It costs nothing:
scratch count, scratch placement, scratch frame, spill counts and every resource
field are the donor's numbers to the byte.

### Bit 3 — DS write, `1 → 3`. Perturbs the bytes, changes no metric.

`M0` vs `M8` differ in `.text` but agree on scratch total, placement, frame,
drain position and partition. Raising a group of 1 to a group of 3 when the half
issues exactly 3 DS writes reorders a few instructions and buys nothing
measurable. Cosmetically correct, operationally inert.

### Answer to the question as asked

**No. No subset gets the drain to `mfma=48` without the scratch migration** —
both are pure functions of bit 0, and the 16-mask enumeration leaves nowhere for
them to come apart. **But the question's premise is worth revising**, because the
two effects the first run reported as one turn out to be independent, and the
cheaper one is *entirely* free:

- **mask 4 buys the 48/48 barrier split at 21 scratch instructions**, versus 114
  for mask 15, with a resource tuple identical to the donor's in every field.
  Strictly better than 15 on every axis except the drain.
- **mask 6 is byte-identical to mask 4.** Same arm, longer spelling.
- the drain move is available only bundled with the migration, and §6 shows what
  that migration actually is.

---

## 6. What the 96 migrated accesses are — attributed to source

Built masks 0 and 1 again with `-gline-tables-only` (`G0`, `G1`). The debug flag
does not perturb codegen: `G0.text` hashes `96049dfa…` = `M0`, `G1.text` hashes
`214ee45e…` = `M1`. So the line table describes the same code that was measured.

```
===== G0 (21 scratch accesses) =====      ===== G1 (114 scratch accesses) =====
   4  k0pf6gm_device_tile_mps.hip:216        96  n2_phase2_gm_mps.cpp:145     <-- all 96
   3  n2_phase1_gm_mps.cpp:327                4  k0pf6gm_device_tile_mps.hip:216
   3  packet.cuh:135                          3  packet.cuh:135
   2  n2_phase1_gm_mps.cpp:297                2  n2_phase1_gm_mps.cpp:327
   … 9 more, 1 each                           … 12 more, 1 each
```

**Every one of the 96 attributes to a single source line, and it is in phase 2**,
at the snapshot's `n2_phase2_gm_mps.cpp:145`:

```cpp
144:          const std::uintptr_t a =
145:              (std::uintptr_t)peer_tab[xr >> maxtok_sh] + slot_off +
146:              ((std::size_t)(xr & tok_mask) * kHidden + col_base + 2 * dcol) * 2u;
148:          kittens::distributed::accumulate_peer_bf162(reinterpret_cast<void*>(a), d);
```

That is the **peer-table base load in the remote-atomic accumulate loop** — M8/M9's
fabric path. In the ISA the spill sits at the head of the `if (xtok[i] < T)`
block, right behind the exec-mask branch:

```
  00013130  s_and_saveexec_b64    s[0:1], vcc                       [n2_phase2_gm_mps.cpp:142]
  00013134  s_cbranch_execz       47                                [n2_phase2_gm_mps.cpp:142]
  00013138  scratch_load_dword    v23, off, off offset:64           [n2_phase2_gm_mps.cpp:145]   <-- migrated
  00013140  v_accvgpr_read_b32    v19, a185                         [n2_phase2_gm_mps.cpp:145]
  00013148  v_lshrrev_b32_e32     v19, v19, v190                    [n2_phase2_gm_mps.cpp:145]
  0001314C  v_accvgpr_read_b32    v18, a207                         [n2_phase2_gm_mps.cpp:145]
  00013154  ds_read_b32           v18, v18                          [n2_phase2_gm_mps.cpp:141]
  0001315C  s_movk_i32            s18, 0x1c00                       [n2_phase2_gm_mps.cpp:146]
  00013160  v_accvgpr_read_b32    v32, a186                         [n2_phase2_gm_mps.cpp:146]
  00013168  v_accvgpr_read_b32    v33, a187                         [n2_phase2_gm_mps.cpp:146]
  00013170  s_waitcnt             vmcnt(0)                          [n2_phase2_gm_mps.cpp:145]   <-- full drain on it
```

Counting over the whole kernel's 282 `flat_atomic_pk_add_bf16` (the remote bf16
accumulates — identical count in both builds, so no atomic is duplicated):

| | mask 0 | mask 1 |
|---|---:|---:|
| remote atomics with a `scratch_*` in the preceding 20 instructions | **0** / 282 | **96** / 282 |
| remote atomics with a full `vmcnt(0)` in the preceding 20 instructions | **1** / 282 | **101** / 282 |
| migrated-scratch → next atomic distance | — | 17 instructions, **min = med = max** |

The constant distance of exactly 17 for all 96 says these are 96 instances of
one inlined code shape, not scattered spills.

### This is a known, previously-engineered-away regression

`n2_phase2_gm_mps.cpp`'s own comment block, ~40 lines below the affected line,
documents the identical failure with the identical count:

> carrying the 5-valued `thr_mode` itself into the loop cost **ONE EXTRA LIVE
> VGPR**, which at the epilogue's 256-VGPR ceiling **evicted the peer table's LDS
> base address to scratch** and put a `scratch_load_dword` 5 instructions ahead
> of **EVERY remote atomic — 96 static sites**, on the shipped depth-8 path too.

exp_21 restructured `throttle_plan` into four mutually exclusive booleans
specifically to avoid this. Bit 0 of a **phase-1** scheduling hint re-triggers it
from the other side: the DS-read group reorders phase 1 enough to raise
whole-function pressure past the same 256-VGPR ceiling, and the allocator makes
the same eviction. Total spilled *bytes* go **down** (144 → 128 B/lane, 16 → 14
VGPRs) while the access count goes **up** 21 → 114, because the value it chose to
evict is the one read on the hottest path in the kernel.

That upgrades bit 0 from "risky" to "reintroduces a measured, deliberately
removed defect in the fabric path". It also says the cost is an allocator
artifact, not intrinsic to the hint — which is why §7 lists recovering it as a
follow-up rather than closing it.

---

## 7. Phase 2's `0x020` hint is over-scaled — measured, and it is the same defect class

Measured per-half census of **phase 2's** K-loop, by the same backedge+census
method (validated by phase 1, where it reproduces the hinted 19 VMEM / 48 MFMA /
3 DSW exactly):

| class | phase-2 hint | requested at `kGM=3` | **measured per half** | verdict |
|---|---|---:|---:|---|
| `0x008` MFMA | `14 * kGM` | 42 | **42** | correct |
| `0x200` DS write | `kGM` | 3 | **3** | correct |
| `0x100` DS read | `4 * kGM` | 12 | 25 | under, by the same deliberate `read_a2`-only convention as phase 1 |
| **`0x020` VMEM read** | **`8 + 7 * kGM`** | **29** | **17** | **over-requests by 12** |

Phase 2 asks the scheduler to place 29 VMEM reads in a group where only 17
exist. Phase 1's own `16 + kGM` = 19 matches its measured 19 exactly, which is
why bit 1 was inert here — phase 2 has no such luck. The correct form is
`14 + kGM`, and it agrees with the existing expression at `kGM = 1`
(`8+7·1 = 15 = 14+1`), so it is a parameterization fix on the same footing as
this experiment's, not a retune. The one-line patch is written up in
`activate.md` §7; **`n2_phase2_gm_mps.cpp` is exp_24's file and was not touched.**

---

## 8. The K-loop mechanism, mask 0 vs mask 4 vs mask 15

Loop located by backedge, confirmed by census: phase-1 K-loop = the innermost
loop containing 96 `v_mfma`, 40 DS reads, 6 DS writes, 38 VMEM loads and 2
`s_barrier`s (one iteration = 2 K128 steps).

```
mask 0  / D    0xb050 .. 0xc464     739 instructions
mask 4         0xb050 .. 0xc46c     741 instructions
mask 15        0xb098 .. 0xc434     726 instructions
```

Per-iteration instruction *mix* is identical in all three — same 96 MFMA, 40 DS
reads, 6 DS writes, 38 VMEM loads, 2 barriers. Only the order and the waits
differ, which is exactly what a scheduling hint should change and nothing more.

### 8.1 Wait / barrier timeline, one iteration

Annotated with MFMAs issued so far.

| mask 0 (donor) | mask 4 (recommended) | mask 15 |
|---|---|---|
| **`vmcnt(0) lgkmcnt(0)` @ 0** | **`vmcnt(0) lgkmcnt(0)` @ 0** | `lgkmcnt(0)` @ 0 |
| `lgkmcnt(1)` @ 17 | `lgkmcnt(1)` @ 17 | `lgkmcnt(0)` @ 17 |
| `lgkmcnt(0)` @ 25 | `lgkmcnt(0)` @ 25 | `lgkmcnt(0)` @ 33 |
| `lgkmcnt(0)` @ 33 | `lgkmcnt(1)` @ 33 | `lgkmcnt(0)` @ 48 |
| **`s_barrier` @ 33** | `lgkmcnt(0)` @ 37 | **`s_barrier` @ 48** |
| `lgkmcnt(0)` @ 48 | `lgkmcnt(0)` @ 48 | **`vmcnt(0) lgkmcnt(0)` @ 48** |
| `lgkmcnt(0)` @ 65 | **`s_barrier` @ 48** | `lgkmcnt(1)` @ 65 |
| `lgkmcnt(0)` @ 81 | `lgkmcnt(0)` @ 48 | `lgkmcnt(0)` @ 73 |
| **`s_barrier` @ 82** | `lgkmcnt(0)` @ 65 | `lgkmcnt(1)` @ 81 |
| *(14 more MFMAs, then backedge)* | `lgkmcnt(0)` @ 81 | `lgkmcnt(0)` @ 89 |
| | `lgkmcnt(0)` @ 96 | `lgkmcnt(0)` @ 96 |
| | **`s_barrier` @ 96** | **`s_barrier` @ 96** |

### 8.2 There is exactly one `s_waitcnt vmcnt(0)` per K-loop iteration

True in every one of the 17 builds, in **both** K-loops, and it is always
`vmcnt(0)` — a full drain, never a partial count. Per *half* that means one half
carries the drain and the other carries none.

**This corrects `m6_m7_structure.md` §5.3 item 4**, which states "Compiled out.
There is no `vmcnt(0)` inside M6's K-loop." The source-level reasoning was right
— `n2_completion_observation_probe()` really is compiled out at
`N2_FORCE_VMCNT0=0` — but the *compiler* inserts its own full drain, and it is in
the shipping build. §7 open item 4 ("No phase-1 disassembly exists in this tree")
is closed. §5.3's stall-site #1 should read: one drain per *iteration*, not per
half, and `vmcnt(0)`, never partial. Carried into `activate.md` §6 so the next
reader trips over it.

### 8.3 What bit 2 does, precisely

The two `lds_cta_barrier()` calls should split the iteration 48 / 48. The donor
splits it **33 / 49 / 14**: the first `s_barrier` fires after only 33 of half 0's
48 MFMAs, and 14 MFMAs are sunk *past* the second barrier into the loop tail.
Bit 2 raises the MFMA group from 16 (the `kGM=1` literal) to the true 48, and the
split becomes **48 / 48 / 0** — one K128 step's MFMA between barriers, nothing
spilling over. That is the source-level intent, realized for the first time at
`G=3`. Since `lds_cta_barrier()` is CTA-wide, the donor shape makes every wave
synchronize 15 MFMAs earlier than the LDS buffer switch requires, then runs 14
MFMAs in the following buffer's window.

Donor, first barrier at 33:

```
	ds_read_b128 v[14:17], v14                     // 0000B790
	s_waitcnt lgkmcnt(0)                           // 0000B798
	s_barrier                                      // 0000B79C   <-- mfma=33
	v_pk_mul_f32 v[42:43], v[54:55], v[148:149] …  // 0000B7A0
```

mask 15 (mask 4 is the same shape), barrier at 48, immediately behind the three
`store_a` DS writes:

```
	ds_write_b128 v23, a[176:179] offset:12288     // 0000B9D8
	ds_write_b128 v23, a[180:183] offset:16384     // 0000B9E0
	ds_write_b128 v23, a[184:187] offset:20480     // 0000B9E8   <-- all 3 = kGM
	s_waitcnt lgkmcnt(0)                           // 0000B9F0
	s_barrier                                      // 0000B9F4   <-- mfma=48
	ds_read_b128 a[168:171], v145 offset:12288     // 0000B9F8   <-- next half's read_a
```

### 8.4 What bit 0 does, precisely — and why it is not worth its price

In the donor the wave issues its DS reads, 19 VMEM loads and ~30 VALU, then
stalls on `vmcnt(0)` with **zero MFMA of this iteration issued to cover it**. Bit
0 moves that drain to `mfma = 48`, so a full half of MFMA (48 × 32 = 1,536 cycles
of issue at the `v_mfma_f32_16x16x128_f8f6f4` rate) executes first. That is a
genuinely valuable relocation and it is the single largest schedule improvement
available on this axis.

It is also the one that puts a scratch load and a `vmcnt(0)` in front of 96
remote atomics (§6). Cover 1,536 cycles once per K-loop iteration in M6; pay a
spill-load-plus-drain per remote atomic in M8/M9. Nothing here prices those two
against each other — that needs the GPU — but the second is in the ~1,309 µs
combine pool on the fabric path, and it is a defect a previous experiment
already paid to remove.

### 8.5 Cost side of mask 4

Waits per iteration go 9 → 11, all of the additions `lgkmcnt` (LDS), none
`vmcnt`: `lgkmcnt(1)@33` and `lgkmcnt(0)@37` are new, and the drain stays where
the donor had it. Loop length 739 → 741. `m6_m7_structure.md` §5.3 item 3 prices
all 20 DS reads + 3 DS writes per K-step at ~170–200 cycles of an ~9,000-cycle
step, so two extra LDS waits are a small debit — but a real one, and it is the
only debit mask 4 carries.

---

## 9. Artifacts on the node

```
~/overnight-scratch/e26/
  dhk/                      immutable snapshot of the DHK tree at 0b82cd19
  tu/b0, tu/b1              generated throwaway TUs (banner + one-line include diff)
  out/B{0,1,2}.*            first run (donor / mask 0 / mask 15)
  out2/D.*                  donor build, this run
  out2/M{0..15}.*           .hsaco, .elf, .isa, .text.bin, .log for every mask
  out2/G{0,1}.*             -gline-tables-only builds of masks 0 and 1, + .lisa
  out2/vendor.diff          donor -> vendored unified diff
  py/mask.py, py/detail.py  loop finder, census, scratch bucketing, line attribution
```

Regeneration, in order, each via `powershell -File tools/nsh.ps1 -Script <script>`:
`e26b_check.sh` → `scp n2_phase1_gm_mps.cpp` → `e26b_build.sh` → `e26b_isa.sh` →
`e26b_isa2.sh` → `e26b_kloop.sh` → `e26b_full16.sh` → `e26b_detail.sh` →
`e26b_attrib.sh` → `e26b_site.sh` → `e26b_p2census.sh` → `e26b_atomic2.sh`.

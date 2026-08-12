# exp_26 — build note: three compile-only builds and the K-loop ISA read

**No GPU work was done.** Everything below is `hipcc --genco` inside `subha_k1`
plus `llvm-objdump`. `pgrep -af 'run_campaign|torchrun|mpirun'` was empty before
and after; no GPU process was started, stopped, or contended.

---

## 0. Build tree and provenance

All three builds come from **one immutable snapshot**, so the only differences
between them are the two intended ones. The shared node checkout was never
modified.

```
node checkout  ~/Distributed-HipKittens  HEAD 0b82cd1980e0ad171018758a2aee28feaee910f3  (clean, no dirty files)
snapshot       git archive HEAD | tar -x -C ~/overnight-scratch/e26/dhk
  k0pf6gm_device_tile_mps.hip  0c92a1c9713fcf31adfb759336e381f41fb1b8b27022f62c05d10f8ebc1832b3
  moe_mps_adapter.cuh          c84f890d39d816bede5a7aab7a4df28a158c7667095931eae5f4d785921f5de3
  n2_phase2_gm_mps.cpp         08c3c2c02dd4d551a51f3e52e53759d495d0ede2cebe892ad995cd4acfef87c6
  K0P6_MPS_SRC_REV = 23,  phase-1 include confirmed at :395
donor          ~/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip/n2_phase1_gm.cpp
               1d90b26658b6a524db69434ddcfa0b2dcec2418c852f83874468d55dd0197dc2
```

The snapshot is one commit behind the local branch head (`d4742feb`, docs only);
`k0pf6gm_device_tile_mps.hip` is identical in both, so the baseline is the
shipping exp_21 mode-12 kernel.

---

## 1. Diff vs the donor — `+112 / −0`

```
$ git diff --no-index --stat .node/mps_n2_phase1_gm.cpp \
                              distributed-kernels/fused_moe/n2_phase1_gm_mps.cpp
 n2_phase1_gm_mps.cpp | 112 +++++++++++++++++++++
 1 file changed, 112 insertions(+)

$ diff -u  <donor>  <vendored>   # on the node, LF vs LF
 diffstat: +110 -0     (110 non-blank insertions + 2 blank lines = 112)
 line count: 585 -> 697   (= +112)
```

**Zero deletions and zero modifications.** Every donor line survives verbatim,
including the four `sched_group_barrier` lines in each half — they now sit in
the `#else` arm of the gate, textually unchanged. The whole diff is eight
insertion hunks:

| hunk | donor line | insertion |
|---|---:|---|
| header | before `:1` | 47-line MPS vendored-copy banner (donor path, sha256, the eight-delta list, the acceptance table) |
| MPS-DELTA (1) | `:73` | `N2GM_P1_SCHED_GSCALE` default 0 + `static_assert`, mirroring the donor's own `N2_FORCE_VMCNT0` idiom |
| MPS-DELTA (2) | `:92` | four `#ifndef`/`#define`/`#endif` empty hook defaults |
| MPS-DELTA (3) | `:161` | `N2GM_P1_TASK_HEAD_HOOK` + 2 comment lines |
| MPS-DELTA (4) | `:317` | `N2GM_P1_KLOOP_ENTER_HOOK` + 2 comment lines |
| MPS-DELTA (5) | `:331` | `#if/#else/#endif` around half 0's four hints + 6 comment lines |
| MPS-DELTA (6) | `:344` | `#if/#else/#endif` around half 1's four hints + 3 comment lines |
| MPS-DELTA (7) | `:351` | `N2GM_P1_KLOOP_EXIT_HOOK` + 2 comment lines |
| MPS-DELTA (8) | `:473` | `N2GM_P1_EPILOGUE_DONE_HOOK` + 3 comment lines |

The active-arm text, at both halves:

```cpp
#if N2GM_P1_SCHED_GSCALE
      __builtin_amdgcn_sched_group_barrier(0x100, 4 * kGM, 0);    // DS read
      __builtin_amdgcn_sched_group_barrier(0x020, 16 + kGM, 0);   // VMEM read
      __builtin_amdgcn_sched_group_barrier(0x008, 16 * kGM, 0);   // MFMA
      __builtin_amdgcn_sched_group_barrier(0x200, kGM, 0);        // DS write
#else
      __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
      __builtin_amdgcn_sched_group_barrier(0x020, 17, 0);
      __builtin_amdgcn_sched_group_barrier(0x008, 16, 0);
      __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);
#endif
```

---

## 2. The three builds — compile commands verbatim

The throwaway translation units are **generated on the node, never committed**:
`overnight/aug11/tools/e26_build.sh` copies the snapshot's
`k0pf6gm_device_tile_mps.hip` into `tu/b0/` and `tu/b1/`, prepends an identical
`SCAFFOLDING ONLY — NOT A SHIPPING SOURCE` banner to both, and `sed`s the
include in `b1` only. Generating rather than committing them means the TU can
never drift from the real kernel.

```
$ diff tu/b0/k0pf6gm_device_tile_mps.hip tu/b1/k0pf6gm_device_tile_mps.hip
403c403
< #include "n2_phase1_gm.cpp"
---
> #include "n2_phase1_gm_mps.cpp"
```

Exactly one differing line (403 = donor-file line 395 + 8 banner lines). Both
TUs are otherwise byte-identical to the shipping kernel plus the banner.

The compile line, run inside `docker exec subha_k1` (ROCm 7.2.4 / LLVM 22),
identical for all three except the marked parts:

```bash
H=/home/subvadla
SC=$H/overnight-scratch/e26
DHK=$SC/dhk
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources

hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
  -Rpass-analysis=kernel-resource-usage \
  -I$DHK/include \
  -I$DHK/distributed-kernels/fused_moe \
  -I$K0/solution/hip/hkp \
  -I$K0/prefill_opt/kernels \
  -I$K0/solution/hip \
  -I$MR -I$MR/include -I$MR/src \
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
  <SRC> <EXTRA> -o $SC/out/<V>.hsaco
```

| build | `<SRC>` | `<EXTRA>` | phase-1 body |
|---|---|---|---|
| **B0** | `tu/b0/k0pf6gm_device_tile_mps.hip` | — | donor `n2_phase1_gm.cpp` |
| **B1** | `tu/b1/k0pf6gm_device_tile_mps.hip` | — | vendored, `N2GM_P1_SCHED_GSCALE` defaulted to 0 |
| **B2** | `tu/b1/k0pf6gm_device_tile_mps.hip` | `-DN2GM_P1_SCHED_GSCALE=1` | vendored, gate ON |

All three: exit 0, zero errors, ~6 s each.

`--genco` emits a clang offload bundle, not a bare ELF, so `llvm-objdump` must
be preceded by:

```bash
clang-offload-bundler --type=o --unbundle \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --input=$V.hsaco --output=$V.gfx950.elf
llvm-objdump -d --mcpu=gfx950 $V.gfx950.elf > $V.isa
```

*(`.node/mps_build_isa_gate.sh` objdumps the bundle directly, which silently
produces a one-line "not recognized as a valid object file" and an empty ISA
file. Worth fixing there.)*

---

## 3. Resource comparison and the B1 ≡ B0 gate

`-Rpass-analysis=kernel-resource-usage`, function `k0pf6gm_mps_mega`:

| | **B0** (donor) | **B1** (vendored, OFF) | **B2** (vendored, ON) | expected for B2 |
|---|---:|---:|---:|---|
| TotalSGPRs | 106 | **106** | **106** | 106 ✅ |
| ArchVGPRs | 256 | **256** | **256** | 256 ✅ |
| AGPRs | 256 | **256** | **256** | 256 ✅ |
| ScratchSize [B/lane] | 144 | **144** | **128** | 144 — came in **16 B lower** |
| LDS [B/block] | 155,496 | **155,496** | **155,496** | 155,496 ✅ |
| Occupancy [waves/SIMD] | 1 | **1** | **1** | 1 ✅ |
| SGPR spills | 180 | 180 | 180 | — |
| VGPR spills | 16 | 16 | **14** | — |
| static `v_mfma` census | **180** | **180** | **180** | 96 + 84 = 180 ✅ |
| scratch ops inside phase-1 K-loop | **0** | **0** | **0** | 0 ✅ |
| scratch ops inside phase-2 K-loop | **0** | **0** | **0** | 0 ✅ |
| scratch ops, whole kernel | 21 | 21 | **114** | ⚠ see §5 |

### B1 ≡ B0 — passed at the strongest available level

| check | result |
|---|---|
| `.text` section sha256 | `96049dfa4f8cd56eade98d3e2c5e565fcf69f3792570e348380c34cc86b9044c` — **identical**, both 166,656 B |
| `.rodata` sha256 | identical |
| `.note` sha256 (the AMDGPU msgpack metadata carrying the resource tuple) | identical |
| full `llvm-objdump -d` text diff | **0 lines** |
| resource tuple | identical in every field |
| whole-ELF `cmp -l` | 41 differing bytes |

Those 41 bytes are confined to `.dynstr`, `.strtab` and the `.gnu.hash`/`.hash`
buckets that index them — i.e. **symbol names, not code**. The cause is the HIP
compilation-unit identity symbol:

```
B0: __hip_cuid_3dd76c637d412175
B1: __hip_cuid_a85c19a7c01423a6
B2: __hip_cuid_ba1995734ac0b3f
```

`__hip_cuid_` is a hash of the TU's source path, and the two TUs necessarily sit
at `tu/b0/…` and `tu/b1/…`. It carries no code and is not consulted by the
kernel. **The default-off vendored body is a provable no-op.**

---

## 4. The K-loop, before and after — the mechanistic evidence

Loop located by backedge, then confirmed by census: phase-1 K-loop = the
innermost loop containing 96 `v_mfma`, 40 DS reads, 6 DS writes, 38 VMEM loads
and 2 `s_barrier`s (one iteration = 2 K128 steps).

```
B0 / B1   0xb050 .. 0xc464     739 instructions
B2        0xb098 .. 0xc434     726 instructions      (-13)
```

Per-iteration instruction mix is **identical** in B1 and B2 — same 96 MFMA, same
36 `ds_read_b128` + 4 `ds_read_b32`, same 32 `flat_load_dwordx4` + 6
`buffer_load_dwordx4`, same 6 `ds_write_b128`, same 2 `s_barrier`. Only the
*order* and the waits differ, which is exactly what a scheduling hint should
change and nothing more.

### 4.1 Wait / barrier timeline

Each row is a `s_waitcnt` or `s_barrier` inside the loop, annotated with how
many MFMAs and how many VMEM loads have issued in the iteration up to that
point.

**B1 — donor hints (`4 / 17 / 16 / 1`)**

| addr | instruction | mfma so far | vmld issued |
|---|---|---:|---:|
| `0xb2ac` | **`s_waitcnt vmcnt(0) lgkmcnt(0)`** | **0** | 19 |
| `0xb53c` | `s_waitcnt lgkmcnt(1)` | 17 | 19 |
| `0xb644` | `s_waitcnt lgkmcnt(0)` | 25 | 19 |
| `0xb798` | `s_waitcnt lgkmcnt(0)` | 33 | 19 |
| `0xb79c` | **`s_barrier`** | **33** | 19 |
| `0xbce4` | `s_waitcnt lgkmcnt(0)` | 48 | 38 |
| `0xbfa8` | `s_waitcnt lgkmcnt(0)` | 65 | 38 |
| `0xc200` | `s_waitcnt lgkmcnt(0)` | 81 | 38 |
| `0xc22c` | **`s_barrier`** | **82** | 38 |
| — | *(14 further MFMAs after the second barrier, then the backedge)* | 96 | 38 |

**B2 — `kGM`-scaled hints (`12 / 19 / 48 / 3`)**

| addr | instruction | mfma so far | vmld issued |
|---|---|---:|---:|
| `0xb304` | `s_waitcnt lgkmcnt(0)` | 0 | 19 |
| `0xb564` | `s_waitcnt lgkmcnt(0)` | 17 | 19 |
| `0xb7b8` | `s_waitcnt lgkmcnt(0)` | 33 | 19 |
| `0xb9f0` | `s_waitcnt lgkmcnt(0)` | 48 | 19 |
| `0xb9f4` | **`s_barrier`** | **48** | 19 |
| `0xbd00` | **`s_waitcnt vmcnt(0) lgkmcnt(0)`** | **48** | 38 |
| `0xbf94` | `s_waitcnt lgkmcnt(1)` | 65 | 38 |
| `0xc0a8` | `s_waitcnt lgkmcnt(0)` | 73 | 38 |
| `0xc1e4` | `s_waitcnt lgkmcnt(1)` | 81 | 38 |
| `0xc2f8` | `s_waitcnt lgkmcnt(0)` | 89 | 38 |
| `0xc418` | `s_waitcnt lgkmcnt(0)` | 96 | 38 |
| `0xc41c` | **`s_barrier`** | **96** | 38 |

### 4.2 What actually moved

**(a) There is exactly ONE `s_waitcnt vmcnt(…)` per K-loop iteration, in both
builds, and its count is `vmcnt(0)` — a full VMEM drain.** No partial
`vmcnt(N)` appears anywhere in either build's K-loop. Per *half* that means:
B1 has one vmcnt site in half 0 and none in half 1; B2 has none in half 0 and
one in half 1. Same total, different position.

**This corrects `m6_m7_structure.md` §5.3 item 4**, which states "Compiled out.
There is no `vmcnt(0)` inside M6's K-loop." The source-level reasoning was right
— `n2_completion_observation_probe()` really is compiled out at
`N2_FORCE_VMCNT0=0` — but the *compiler* inserts its own full drain, and it is
there in the shipping build. §7 open item 4 flagged exactly this gap ("No
phase-1 disassembly exists in this tree"); it is now closed. §5.3's stall-site
#1 should also be amended: it is one drain per *iteration*, not one per half,
and it is `vmcnt(0)`, never a partial count.

**(b) The single drain moved from `mfma = 0` to `mfma = 48`.** This is the whole
mechanism. In B1 the wave issues 12 DS reads, 19 VMEM loads and ~30 VALU, then
stalls on `vmcnt(0)` — which waits not only for those 19 but for the ~19 still
outstanding from the previous iteration's second half — with **zero MFMA work of
this iteration issued to cover it**. In B2, 48 MFMAs (`48 × 32 = 1,536` cycles
of issue at the CDNA4 `v_mfma_f32_16x16x128_f8f6f4` rate) execute before the
same drain. The loads issued in half 0 go from ~0 MFMAs of cover to 48; the
loads carried from the previous half go from 48 to 96.

Verbatim, B1:

```
	ds_read_b128 v[14:17], v14                     // 0000B2A4
	s_waitcnt vmcnt(0) lgkmcnt(0)                  // 0000B2AC   <-- full drain
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], a[168:175], a[32:39], 0  // 0000B2B0  <-- 1st MFMA
	ds_write_b128 v23, a[180:183] offset:16384     // 0000B2B8
```

and B2, where the same drain now sits after a complete half of MFMA:

```
	ds_read_b128 v[14:17], v14                     // 0000BCF8
	s_waitcnt vmcnt(0) lgkmcnt(0)                  // 0000BD00   <-- full drain, mfma=48
	v_mfma_f32_16x16x128_f8f6f4 v[56:59], a[168:175], a[120:127], 0 // 0000BD04  <-- 49th MFMA
	v_add_u32_e32 v125, 8, v125                    // 0000BD0C
```

**(c) The barrier partition became symmetric.** The two `lds_cta_barrier()`
calls should split the iteration 48 / 48. B1 splits it **33 / 49 / 14** — only
33 MFMAs issue before the first `s_barrier`, and 14 MFMAs are sunk *past* the
second barrier into the loop tail. B2 splits it **48 / 48 / 0**: exactly one
K128 step's MFMA between barriers and nothing spilling over. That is the source-
level intent, realized for the first time at `G=3`.

B1's first barrier, reached after only 33 of the half's 48 MFMAs:

```
	ds_read_b128 v[14:17], v14                     // 0000B790
	s_waitcnt lgkmcnt(0)                           // 0000B798
	s_barrier                                      // 0000B79C   <-- mfma=33
	v_pk_mul_f32 v[42:43], v[54:55], v[148:149] …  // 0000B7A0
```

B2's, reached after all 48, immediately behind the three `store_a` DS writes:

```
	ds_write_b128 v23, a[176:179] offset:12288     // 0000B9D8
	ds_write_b128 v23, a[180:183] offset:16384     // 0000B9E0
	ds_write_b128 v23, a[184:187] offset:20480     // 0000B9E8   <-- all 3 = kGM
	s_waitcnt lgkmcnt(0)                           // 0000B9F0
	s_barrier                                      // 0000B9F4   <-- mfma=48
	ds_read_b128 a[168:171], v145 offset:12288     // 0000B9F8   <-- next half's read_a
```

**(d) Cost side: 3 extra LDS waits.** 7 `s_waitcnt` per iteration in B1
(1 vmcnt+lgkm, 6 lgkm-only) → 10 in B2 (1 vmcnt+lgkm, 9 lgkm-only), against 13
fewer instructions overall. LDS waits are cheap here — §5.3 item 3 prices all 20
DS reads + 3 DS writes per K-step at ~170–200 cycles of an ~9,000-cycle step —
so this is a small debit against the vmcnt relocation, but it is a debit.

---

## 5. The one real finding against B2: scratch moved into the M7/M8 tail

Total scratch instructions 21 → 114. Both MFMA K-loops stay at **0** — the gate
holds — but the distribution shifted sharply:

| region | B0 / B1 | B2 |
|---|---:|---:|
| before M6 | 15 | 9 |
| **inside the phase-1 MFMA K-loop** | **0** | **0** |
| M6 tail → M7 head | 3 | 6 |
| **inside the phase-2 MFMA K-loop** | **0** | **0** |
| **after the phase-2 K-loop (M7 epilogue / M8 / M9)** | **3** | **99** |

A phase-1-only scheduling hint pushed 96 extra spill/reload accesses into the
M7-epilogue/M8/M9 region — while the scratch *frame* shrank 144 → 128 B/lane and
spilled VGPRs went 16 → 14. Fewer values spilled, accessed far more often, in a
different place. This is whole-function register allocation reacting to a
changed schedule in one loop; it is not a bug in the change, but it is a real
risk, because in the mode-12 ratchet that region is the 1,309 µs pool.

**Consequence for the run plan: the A/B must have timestamps on (cfg bit 33) and
must read M6 and M8M9 separately.** The two effects push in opposite directions
and a single end-to-end number cannot tell them apart. See `activate.md` §4.

---

## 6. Artifacts on the node

```
~/overnight-scratch/e26/
  dhk/                      immutable snapshot of the DHK tree at 0b82cd19
  tu/b0, tu/b1              generated throwaway TUs (banner + one-line include diff)
  out/B{0,1,2}.hsaco        offload bundles
  out/B{0,1,2}.gfx950.elf   unbundled device ELFs
  out/B{0,1,2}.isa          disassembly
  out/B{0,1,2}.log          -Rpass-analysis resource remarks
  out/B{1,2}.p1loop.txt     the phase-1 K-loop body, one instruction per line
  out/vendor.diff           donor -> vendored unified diff
  out/b0b1.isadiff          empty (0 lines)
  py/kloop.py, py/detail.py the loop-finder / census scripts
```

Regeneration, in order:
`tools/e26_snap.sh` → `scp n2_phase1_gm_mps.cpp` → `tools/e26_build.sh` →
`tools/e26_isa3.sh` → `tools/e26_kloop.sh` → `tools/e26_isa4.sh` →
`tools/e26_isa6.sh`, each via
`powershell -File tools/nsh.ps1 -Script <script>`.

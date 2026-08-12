# exp_27 build + ISA notes

Three genco TUs, each in its own directory so the quoted `#include` of
`n2_phase1_gm_mps.cpp` resolves TU-locally and the node checkout is never
touched:

| TU | source | arm |
|---|---|---|
| `R` | the ratchet, commit `f34e72fc` (no knob at all) | reference |
| `A0` | exp_27, `K0P6_MPS_ASCALE_TM 0` — commit `d3d22ce4` | control |
| `A1` | exp_27, `K0P6_MPS_ASCALE_TM 1` — commit `f113d73f` | candidate |

`A1` is produced by **rewriting the hashed literal**, never by `-D`, so the TU
that is gated is byte-for-byte the TU that is committed and launched. `A0` and
`A1` differ by exactly one line:

```
120c120
< #define K0P6_MPS_ASCALE_TM 0
> #define K0P6_MPS_ASCALE_TM 1
```

Build command: `BUILDING.md`'s genco invocation (`hipcc --genco
--offload-arch=gfx950 -std=c++20 -O3 -DKITTENS_CDNA4
-DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math -mllvm -amdgpu-mfma-vgpr-form=1
-DK0P6GM_G=3 -DN2GM_G=3`) plus
`-Rpass-analysis=kernel-resource-usage`, in `subha_k1`. All three compiled clean,
0 errors, ~7 s each. Driver: `../tools/e27_build.sh`, `../tools/e27_isa2.sh`,
`../tools/e27_isa3.sh`.

---

## G1 — the knob is provably inert at 0

| TU | `.text` sha256 | size |
|---|---|---:|
| `R` | `ab0c353bef065898881b7f845f44c20085bf5c8661abaa0ca8fc19988046f43a` | 179,904 B |
| `A0` | `ab0c353bef065898881b7f845f44c20085bf5c8661abaa0ca8fc19988046f43a` | 179,904 B |
| `A1` | `642646fcd1de3daeb194cf94dd0d96000bb188298f2cd050da33dee833a541a7` | 179,520 B |

**`A0` is byte-identical to the ratchet.** The control arm is not "equivalent to"
the denominator, it *is* the denominator's binary, so the knob cannot regress it
and batch 1/batch 3 measure the ratchet itself. `A1` differs (−384 B), so the arm
really changed the code — the other direction of the check, which matters because
`hsaco_before != hsaco_after` is not evidence of anything.

That hash is also the value exp_26 recorded for its donor-include build, so the
reference is consistent across two independent experiments.

---

## G2/G3 — resources and census

| | `R` | `A0` | `A1` | budget |
|---|---:|---:|---:|---|
| TotalSGPRs | 106 | 106 | **106** | 106 |
| VGPRs (Arch) | 256 | 256 | **256** | 256 exact |
| AGPRs | 256 | 256 | **256** | 256 exact |
| ScratchSize B/lane | 128 | 128 | **128** | ≤ 128 |
| LDS B/block | 155,496 | 155,496 | **155,496** | 155,496 exact |
| Occupancy waves/SIMD | 1 | 1 | 1 | 1 |
| SGPR spill | 186 | 186 | 186 | — |
| VGPR spill | 14 | 14 | **15** | ≤ 9 *(see note)* |
| static `v_mfma` | 180 | 180 | **180** | 96 + 84 = 180 |
| static `scratch_load/store` | 19 | 19 | **19** | — |
| `flat_atomic_pk_add_bf16` | 282 | 282 | 282 | 282 |
| `s_barrier` | 81 | 81 | 81 | — |

**On the VGPR-spill line.** exp_28's design pre-registered `VGPR spills ≤ 9`, but
the ratchet itself reports 14, so that threshold was written against a stale
tuple and cannot be applied as-is. What is actually load-bearing is the standing
budget — *"every remaining spill byte must be outside both MFMA K-loops"* — and
the two facts that bound it are unchanged: **ScratchSize is identical (128 B) and
the static scratch-op count is identical (19)**. The +1 is a spill-weight
accounting change, not new scratch traffic. G4 below settles placement directly.

LDS is untouched, as it must be: `ascale_lds` keeps its declaration and its
`[56][96]` = 21,504 B.

---

## G4 — scratch placement relative to the MFMA spans

MFMA spans located by instruction address, clusters split at > 4,096 B of
non-MFMA code:

| arm | span 1 (phase-1 K-loop) | span 2 (phase-2 K-loop) |
|---|---|---|
| `A0` | `0xb484 .. 0xc5dc`, 96 mfma | `0x1102c .. 0x11e7c`, 84 mfma |
| `A1` | `0xb3c8 .. 0xc50c`, 96 mfma | `0x10ebc .. 0x11d24`, 84 mfma |

96 + 84 = 180 in both arms, confirming G3 structurally rather than by a bare
count. All 19 scratch ops in each arm sit outside both ranges:

**G4 PASS — zero scratch ops inside either MFMA span, both arms.**

The scratch ops sit at `k0pf6gm_device_tile_mps.hip:225`, `util.cuh:75`,
`packet.cuh:135`, and phase-1 lines 309/324/342/353/354/433/443/664/678/690 — all
prologue, epilogue or service-path. `A1` trades two of `A0`'s
`:324`/`:354` ops for one at `:443`; the net count is unchanged and the placement
is still clean.

---

## G5 — the gather ISA, line-attributed

The whole-kernel static counts move almost not at all (`flat_load_dword`
150 → 148, `flat_load_dwordx4` 355 → 356, `ds_write_b32` 60 → 59) **because the
gather is a loop, and a static count cannot show a trip count.** So both arms
were rebuilt with `-gline-tables-only` and the instructions attributed to the
gather's source lines pulled out directly. The debug build's `.text` is
**identical** to the gated build's for both arms (`ab0c353b…` / `642646fc…`), so
this is attribution of the exact gated binary, not of a debug variant.

### arm 0 — group-major (`n2_phase1_gm_mps.cpp:391-399`)

```
:397  v_mad_i64_i32   v[8:9], s[18:19], v6, v8, 0        // k * T, 64-bit, T runtime
      v_lshl_add_u64  v[8:9], v[8:9], 2, v[10:11]        // *4 + A_scale base
      v_lshl_add_u64  v[2:3], v[2:3], 2, v[8:9]          // + token*4
      flat_load_dword v3, v[2:3]                         // 4 BYTES
:396  ds_write_b32    v2, v3                             // one dword to LDS
:392  v_add_u32_e32   v2, 0x100, v4                      // idx += kThreads (256)
      v_cmp_lt_u32_e32 vcc, s18, v4                      // s18 = 0x13ff = 5119
```

### arm 1 — token-major (`n2_phase1_gm_mps.cpp:374-389`)

```
:382  v_mad_i64_i32    v[2:3], s[18:19], v9, s24, v[2:3] // token * s24, s24 = 0xe0 = 224
      v_lshlrev_b32_e32 v44, 4, v7                       // c * 16
      v_lshl_add_u64   v[2:3], v[2:3], 0, v[44:45]
      flat_load_dwordx4 v[2:5], v[2:3]                   // 16 BYTES
:385  ds_write2_b32    v7, v2, v3 offset1:96             // TWO dwords
:387  ds_write2_b32    v2, v4, v5 offset0:64 offset1:160 // TWO dwords
:374  v_add_u32_e32    v2, 0x100, v6                     // idx += kThreads (256)
      v_cmp_lt_u32_e32 vcc, s18, v6                      // s18 = 0x43f = 1087
```

Four things to read off this, three of them the pre-registered gate and one a
bonus:

1. **Trip count 21 → 6.** The loop limit immediate goes `0x13ff = 5119` →
   `0x43f = 1087`, and `5119 = 5376 − 257`, `1087 = 1344 − 257` — i.e. exactly
   `kKGroups·kMrows` and `kScaleQuads·kMrows` under the same rotation. With
   `kThreads = 256`: `⌈5376/256⌉ = 21` → `⌈1344/256⌉ = 6`. **This is the gate
   exp_28 asked for, met exactly.**
2. **`flat_load_dword` → `flat_load_dwordx4`.** 4 B → 16 B per lane per
   iteration: 5,376 scalar loads → 1,344 vector loads for the same 21,504 B.
3. **The token stride folded into the address as a constant**: `s24 = 0xe0 = 224`,
   the byte stride of one token-major scale row (`56 × 4`). The `×4` shift
   disappears entirely, and the runtime `T` is gone from the address — arm 0
   needed a 64-bit multiply by a *runtime* value, arm 1 multiplies by an
   immediate.
4. **Better than designed: the four LDS stores merged into two `ds_write2_b32`.**
   `offset1:96` is `kMrows`, the `ascale_lds` row stride in dwords, so one
   instruction writes `ascale_lds[4c+0][i]` and `ascale_lds[4c+1][i]`; the second
   (`vaddr + 0x200`, `offset0:64 offset1:160`) writes rows `4c+2` and `4c+3` at
   `+192` and `+288` dwords = `2·96` and `3·96`. The design predicted four
   `ds_write_b32`; the compiler found the pairing. Row-fast indexing is what makes
   this legal — consecutive lanes write consecutive dwords, so the writes are
   bank-conflict-free *and* pairable.

Line-traffic consequence, which is the whole point:

| | arm 0 | arm 1 | ratio |
|---|---:|---:|---:|
| distinct 64 B lines per task | 5,376 | 384 | **14.0×** |
| bytes delivered per task | 344,064 | 24,576 | 14.0× |
| bytes actually wanted | 21,504 | 21,504 | 1 |
| line amplification | 16× | 1.14× | |

---

## G6/G7 — the arm that was actually launched

Per-batch fingerprint by `.text` of the resolved JIT hsaco (`../tools/e27_fp.sh`).
`.text` is used rather than the file hash because a "rebuild" tonight differed in
36 of 188,744 bytes — embedded build paths — with identical `.text`, so only the
`==` direction of a file-hash comparison is sound.

| batch | arm | JIT dir | hsaco file sha256 | **`.text` sha256** | size | ARM PROBE |
|---|---:|---|---|---|---:|---|
| 1 (control) | 0 | `909ed8f43975` | `9a31c941…` | `ab0c353bef065898…` | 179,904 | `0x13ff`=1, `0x43f`=0 ⇒ arm 0 |
| 2 (candidate) | 1 | `5635a2f2a370` | `a6e4189c…` | `642646fcd1de3dae…` | 179,520 | `0x13ff`=0, `0x43f`=1 ⇒ arm 1 |
| 3 (control) | 0 | `909ed8f43975` | `9a31c941…` | `ab0c353bef065898…` | 179,904 | `0x13ff`=1, `0x43f`=0 ⇒ arm 0 |
| 4 (candidate) | 1 | `be7b189d458d` | `d3afb529…` | `642646fcd1de3dae…` | 179,520 | `0x13ff`=0, `0x43f`=1 ⇒ arm 1 |

Every batch's `.text` is **identical to the corresponding genco build** — the two
control batches to `A0`/`R` (`ab0c353b…`) and the two candidate batches to `A1`
(`642646fc…`). So the JIT and the compile-only gate produce the same code, and the
control batches measured the ratchet binary itself. `mfma=180` verified in each
launched binary, not just in the genco build.

Note batches 2 and 4 are **different JIT directories** (`5635a2f2a370`,
`be7b189d458d`) with **different file hashes** but the **same `.text`** — a direct
instance of why the file hash is not the fingerprint: those two differ only in
embedded build paths. Had I trusted `hsaco_before != hsaco_after`, I would have
concluded the arm changed between batches 2 and 4. It did not.

The ARM PROBE is the gather's loop-limit immediate, which is a direct,
arm-discriminating signature: `0x13ff` ⇒ group-major/trip-21, `0x43f` ⇒
token-major/trip-6. A batch whose probe is ambiguous is not trusted.

**G7**: on the arm-1 fast path there is no `sc_dst` store at all. Source-level,
the only surviving `sc_dst` writer in arm 1 is `hkp::zero_part_scale_transpose<14>`
inside `if (!skip_dead_part_zero(cfg4))`, which the ratchet config
(mode 12, `g` bit `0x40`, no dual-write detector, no pull) does not take.
`scale_transpose_row` is compiled out of arm 1 entirely. `sc_dst` remains
allocated and in the descriptor; reclaiming its 7.34 MiB is a separate host-side
change that deliberately does not ride along.

---

## Verdict on the build gates

| gate | result |
|---|---|
| G1 arm-0 `.text` == ratchet | **PASS** (`ab0c353b…`, 179,904 B, both) |
| G2 arm-1 resource tuple | **PASS** — `SGPR 106 / VGPR 256 / AGPR 256 / scratch 128 B / LDS 155,496` exact |
| G3 MFMA census 180 | **PASS** (96 + 84, both arms, and in the launched binary) |
| G4 scratch outside both MFMA spans | **PASS** both arms, 19 ops each, none inside |
| G5 `dwordx4` + trip count 6 not 21 | **PASS** (`0x13ff` → `0x43f`; and 4 LDS stores → 2 `ds_write2_b32`) |
| G6 fresh hsaco per arm | **PASS** — arm 0 built a new dir `909ed8f43975` at `SRC_REV 26` |
| G7 no live `sc_dst` store on the arm-1 fast path | **PASS** |

# exp_24 — design: delete the dead `part` zero-fill (A) + sweep the M7 throttle depth (B)

Two INDEPENDENT flag bits, ONE build, one campaign-comparable arm set.
Denominator: the exp_21 ratchet `mps_mega` `C=16,g=33,mode=12,flush_rows=16`
(campaign 6,685 µs = 0.866× production; screen 6,703-6,803 µs, `ratio_vs_prod`
1σ = 0.52 % over six repeats).

Short names, as in `../CONTEXT/mode12_protocol_map.md`:

| tag | path |
|---|---|
| `KRN` | `distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip` |
| `ADP` | `distributed-kernels/fused_moe/moe_mps_adapter.cuh` |
| `P2` | `distributed-kernels/fused_moe/n2_phase2_gm_mps.cpp` |
| `ABI` | `distributed-kernels/fused_moe/moe_host_abi.hpp` |
| `HKQ` | `~/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip/hkp/hkp_quant.hpp` (**READ-ONLY, not ours**) |
| `DRV` | `~/amd-master/.../prefill_opt/host/e004pf_k0pf_ab.py` (**READ-ONLY for this experiment**) |

---

## 1. Mechanism A — the `part` zero-fill is provably dead in mode 12

### 1.1 What runs today

`KRN:1278-1289`, between the M3 grid barrier (`KRN:1236`) and the M4 grid
barrier (`KRN:1291`), every CTA with `bid >= 2` runs

```
for (int t = w; t < T_ext; t += nw2)
  hkp::zero_part_scale_transpose<14>(part + t*K0P6_H, sc_dst,
                                    sc_stage + t*K0P6_NG, t, T_ext,
                                    K0P6_NG, lane2);
```

`HKQ:130-143` is the whole helper and it does exactly two things:

```
for (c = 0; c < 14; ++c)                                  // (i) the ZERO
  *(uint4*)(prow + (c<<10) + (lane<<4)) = {0,0,0,0};
if (lane < ng)                                            // (ii) the TRANSPOSE
  sc_dst[lane*T_loc + t] = sc_stage_row[lane];
```

- (i) writes `14 × 1 KiB = 14,336 B` = one full `part` row (`7168 × 2 B`).
  Over `T_ext = 8 × 4096 = 32,768` rows (`KRN:688`) that is
  **469,762,048 B = 448 MiB of stores per rank per epoch**.
- (ii) writes one float per (row, scale-group): `32,768 × 56 × 4 B = 7.34 MiB`.
  `sc_dst` **is** read by M6 — `KRN:1321` passes `K0P6_D_SC_DST` straight into
  `n2p6gm_phase1_body`. The transpose must stay.

### 1.2 Line-cited proof that (i) is dead at `mode=12, detect=0, pull_fallback=0`

Every reader/writer of `part` (descriptor slot 21, `K0P6_D_PART`) in `KRN`,
exhaustively (`grep K0P6_D_PART`):

| site | what it is | status in mode 12 |
|---|---|---|
| `KRN:1227` + `1284-1288` | the M5 zero itself | **the thing we skip** |
| `KRN:1380` | `part` handed to M7 as the `OUT` kernarg | `P2:131-167`: with `peer_tab != nullptr` (`P2:132`) the atomic target is `peer_tab[...] + slot_off + ...` (`P2:144-149`); `OUT` is dereferenced **only** in the `dual` arm `P2:160-165`, and `dual = detect_dual(cfg)` (`P2:270-271`, `ADP:213-216`) is **false** because `g`'s `0x10` bit is clear at `g=33`. `peer_tab` is non-null exactly when `mode_is_direct_accum` (`P2:249-250`). |
| `KRN:1433` | `env.part` for the service pool | consumed **only** by `slice_group_src` (`ADP:491-495`) → `push_slice_group[_batch]` (`ADP:512-526, 528-548`), reached only at `ADP:816-832`. Mode 12 takes the `ADP:787-799` branch and `continue`s at `ADP:798` **before** that loop. Unreachable. |
| `KRN:1491` | diag-pool `d.part` | modes 5/6 only (`ADP:224-226`), gated off. |
| `KRN:1601` | mode-1 bulk push `part_mut` | mode 1 only. |
| `KRN:1694-1696` | M8 combine source | `part = (m8_pull \|\| m7_detect) ? ... : nullptr`. `m8_pull = mode_is_parity_publish(12) \|\| pull_fallback` = `false \|\| false` (`ADP:229-231`; `pull_fallback` is *forbidden* in direct-accum modes by `ADP:266-268`). `m7_detect = false`. ⇒ **`part == nullptr`**, `base_pull` (`KRN:1714-1718`) is never called, and M8 reads `base_slot` (`KRN:1719-1725`) → local `slots` (`KRN:546-555`). |
| `KRN:1521-1548` | M7.5 parity publication | `mode_is_parity_publish` only (`KRN:1539`), off. |

**The `KRN:1282-1283` warning is about a path mode 12 does not use.** It says
"holes must be zeroed too, else stale part rows corrupt combine". That warning
is about **combine reading `part`**, which is `base_pull` at `KRN:1714-1718`
under `m8_pull`. In mode 12 combine reads `base_slot` — `KRN:1719-1725`, wired
at `KRN:1782-1799` — i.e. **`slots`, not `part`**. The hole-zero requirement
transfers to `slots`, and `slots` is kept clean by a *different* mechanism:
zeroed once at setup (`DRV:1509`) and consume-and-zeroed by M8 after every read
(`KRN:579-587`). Nothing in mode 12 depends on `part`'s contents.

### 1.3 Host-side liveness of `part` (checked, because the buffer is shared)

`part` is a single symmetric tensor allocated once (`DRV:1418`,
`mori_t((MROWS, H), "bfloat16")`) and **shared by every arm**. Three checks:

1. **No host read after an `mps_mega` launch.** `pf6mps_mega_body`
   (`DRV:3339-3346`) launches and returns; it neither zeroes nor inspects
   `part`. No gate in `DRV` reads `part` for the `mps_mega` arm.
2. **No cross-arm dependency.** Other arms poison `part`
   (`DRV:6130, 6139, 6672`: `part[:T_LOC_MAX].fill_(1000.0)`) or zero it
   (`DRV:2311, 3088, 4030, ...`) *for their own launches*; every arm that
   **reads** `part` zeroes it itself, either host-side or in its own M5. The
   parity arm `pf6gm_mega` runs the unmodified `k0pf6gm_device_tile.hip` M5
   loop, which is untouched by this experiment. So leaving stale bytes in
   `part` after an `mps_mega` epoch cannot reach any other arm's output.
3. **No graph-replay accumulation hazard.** The reason `part` must be zeroed
   before an accumulating epilogue (`DRV:2307-2311`) is that
   `global_atomic_pk_add_bf16` *accumulates*. In mode 12 M7 issues **zero**
   writes to `part` (§1.2), so there is nothing to accumulate across the 600
   soak epochs.

### 1.4 The guard (fail-closed, not silently-ignored)

`ADP` gains `skip_dead_part_zero(config)`:

```
mode == kModeRemoteAccum (12)          // NOT 13 — see below
  && (g & 0x40) != 0
  && !detect_dual(c)                   // the dual write TARGETS part
  && !c.pull_fallback                  // an M8 pull READS part
```

and `config_is_valid` is widened so the bit cannot be set where it would be
wrong or would do nothing:

- `0x40` legal **only** when `mode == 12` — a `0x40` on any other mode is
  **rejected**, not ignored. (Mode 13 satisfies the same deadness proof — it
  is also `mode_is_direct_accum`, also non-pull, and also `continue`s before
  `ADP:816` at `ADP:800-815` — but it is deliberately out of scope until a
  mode-13 run re-derives it. Rejecting is the fail-closed choice: a config
  either engages the mechanism or is refused.)
- `0x40 | 0x10` (skip + dual-write detector) is **rejected**: the detector
  accumulates into `part` and M8 compares the two towers, so it requires the
  zero. This is the one combination that would silently corrupt a *diagnostic*
  and is the reason the guard is not just the bit test.
- `pull_fallback` in a direct-accum mode was already rejected (`ADP:266-268`);
  unchanged.

Every other mode's code path is bit-identical: the skip is a runtime branch on
a config value that only mode 12 can carry, and the `else` arm is the verbatim
donor loop.

### 1.5 What Mechanism A is expected to be worth (pre-registered)

Honest arithmetic, so a null is not re-narrated as a surprise:

- 469,762,048 B of streaming stores. MI350X HBM3E peak is 8 TB/s; a pure
  `uint4` store stream realistically lands at 60-80 % of that. ⇒ **59-98 µs**
  if the loop is on the M3→M4 critical path.
- It IS on that critical path in the sense that it sits between two grid
  barriers (`KRN:1236` and `KRN:1291`) — but so do CTA 0's serial `tile_desc`
  build (`KRN:1250-1273`, `tid == 0`, `E = 32` experts) and CTA 1's
  `csr_scan_block256` (`KRN:1277`). **If either of those is slower than the
  zero, Mechanism A wins nothing.** That is the null hypothesis and it is
  plausible: the whole plan phase M3-M5 measures ~415 µs.
- So the pre-registered expectation is **50-100 µs ≈ 0.7-1.5 % of 6,700 µs**,
  which is at or below the 2 % single-screen resolution
  (`../CONTEXT/harness_recipe.md` §5). Two extra screens with
  `timestamps=1` are therefore part of the plan: `[MPS TS SPLIT]
  plan_M3toM5` measures the phase directly and is far more sensitive than the
  end-to-end ratio.
- **The Infinity-Cache eviction argument is weak and is NOT claimed.** 448 MiB
  does stream through the 256 MB memory-side LLC, but M6's weights cannot live
  there either: W13 is `32 × 4096 × 7168` fp8 = 939 MB and W2 is
  `32 × 7168 × 2048` fp8 = 469 MB. Both already exceed the LLC by 2-4×, so
  there is no resident working set for the zero to evict. Any LLC effect is a
  bonus, not the mechanism.

---

## 2. Mechanism B — throttle depth as a swept axis

exp_21 found the single biggest lever was capping the M7 epilogue's outstanding
remote RMWs (`P2:150-159`, `s_waitcnt vmcnt(8)`, selected by `g` bit `0x20`):
M7 3,189 → 2,644-2,683 µs, ≈500 µs. **8 was the first value tried and the axis
was never swept.**

### 2.1 The constraint that shapes the implementation

`s_waitcnt vmcnt(N)` encodes `N` in the instruction's `simm16`; there is no
register form. So the depth must be a compile-time constant, and the sweep must
be a runtime dispatch over compile-time instantiations.

Two further constraints from the experiment discipline:

1. **No new branch inside the 16-iteration accumulate loop.** The loop is the
   measured object; adding a scalar compare per iteration would change the
   control arm too.
2. **`g=33` must produce byte-identical code to today.** So the depth-8
   instantiation must be *textually* today's body, and the `bool throttle`
   runtime guard must stay exactly where it is.

### 2.2 The shape chosen

`P2` gains

```
template <int VmCnt> __device__ __forceinline__ void throttle_vmcnt();
// four explicit specializations, each one literal `s_waitcnt vmcnt(N)`
```

and the peer-target accumulate loop (today `P2:138-167`) is lifted verbatim
into

```
template <int JMAX, int VmCnt> __device__ __forceinline__ void
accumulate_rows_peer(...)   // the SAME 16-iteration unrolled body
```

with the single change `asm("s_waitcnt vmcnt(8)")` →
`throttle_vmcnt<VmCnt>()`. `epilogue_write<JMAX>` then dispatches **once per
call** (not per iteration):

```
if (peer_tab != nullptr) {
  switch (thr_sel) {
    case 0: accumulate_rows_peer<JMAX, 8>(...);  break;   // today
    case 1: accumulate_rows_peer<JMAX, 4>(...);  break;
    case 2: accumulate_rows_peer<JMAX, 16>(...); break;
    default: accumulate_rows_peer<JMAX, 32>(...); break;
  }
} else { /* donor non-peer path, untouched, NOT duplicated */ }
```

`thr_sel` is uniform for the whole launch, so the switch is a scalar branch
executed `EV × 4 waves × 2 calls ≈ 133,760` times per rank per epoch against
`117,440,512` payload atomics — 0.11 % of the atomic count, and it replaces
nothing inside the loop.

**Cost accepted:** the peer loop is instantiated 4× instead of 1×, and
`epilogue_write` is `__forceinline__` into a 3×-unrolled sub-block loop for two
`JMAX` values, so the ISA carries 24 copies of a ~130-instruction loop instead
of 6 (≈ +2.3 k instructions). This is a code-size cost only — identical
register live ranges, no MFMA, outside both K-loops. **The resource gate is the
falsifier**: if SGPR/VGPR/AGPR/scratch/LDS or the MFMA census move,
Mechanism B's shape is wrong and gets rebuilt (see `build.md`).

### 2.3 Alternatives rejected

| alternative | why rejected |
|---|---|
| `if (sel==0) vmcnt(8); else if (sel==1) vmcnt(4); ...` inside the loop | 3 extra scalar branches **inside the measured loop**; changes the control arm. Violates §2.1. |
| `asm("s_waitcnt vmcnt(%c0)" :: "i"(N))` | still needs `N` to be an integer constant expression — same problem, no gain. |
| Template the whole phase-2 body / task loop on the depth | duplicates the MFMA K-loop; would multiply the MFMA census and blow the resource tuple. |
| One build per depth (`-D`) | four builds, four JIT keys, four separate gate ladders — the prompt asks for one build, and cross-build absolute µs are not a valid denominator. |
| Emit `vmcnt(4)` every k-th row to emulate a deeper cap | not equivalent: the outstanding-count semantics differ. |

---

## 3. The flag encoding, and why the `g` field had to widen

`config::group_slices` is the packed word's byte `[8:16)` (`ADP:75, 85`), so the
`g` field has **8 bits**, of which `0x0F` (physical g), `0x10` (detect) and
`0x20` (throttle) are taken. **Only `0x40` and `0x80` were free — two bits for
three bits of new state.**

Note the trap this closes: passing `g = 0x121` today would compute
`0x121 << 8 = 0x12100`, whose bit 16 lands in the **mode** field
(`mode |= 1`), and `decode_config` would read `g = 0x21` — a silently corrupted
config that still validates. Any encoding above `0xFF` had to be made explicit.

**Additive widening.** The packed word uses bits 0-33; bits 34-63 are free.
`encode_config`/`decode_config` now carry `g`'s high byte at packed bits
`[34:42)`:

```
encode: ... | ((g & 0xFF) << 8) | (((g >> 8) & 0xFF) << 34)
decode: g = ((word >> 8) & 0xFF) | (((word >> 34) & 0xFF) << 8)
```

For every `g <= 0xFF` — i.e. every config ever run, including the ratchet — the
encoded word is **bit-identical** and `decode_config` returns the same struct.
The host bridge signature (`mps_host_bridge.cpp:18-31`), the 63-word descriptor
ABI, `_parse_mps_config`'s grammar (`int(value, 0)`, so `g=0x121` and `g=289`
both parse) and the 56-word donor parity contract are all unchanged.

### 3.1 `g` bit table (16 bits after the widening)

| bits | name | meaning | validated at |
|---|---|---|---|
| `0x000F` | physical g | must be `1` in modes 12/13 | `ADP` direct-accum arm |
| `0x0010` | `kRemoteAccumDetectBit` | dual-write lost-update detector | `detect_dual` |
| `0x0020` | `kRemoteAccumThrottleBit` | epilogue RMW throttle ENABLE | `throttle_enabled` |
| `0x0040` | `kRemoteAccumSkipPartZeroBit` | **exp_24 A**: skip the dead `part` zero | `skip_dead_part_zero`; legal only at `mode==12`, rejected with `0x10` |
| `0x0080` | — | reserved, **rejected** | |
| `0x0300` | `kRemoteAccumThrottleDepthMask` | **exp_24 B**: depth select `00→8, 01→4, 10→16, 11→32` | `throttle_depth_sel`; requires `0x20`, direct-accum modes only |
| `0xFC00` | — | reserved, **rejected** | |

`00 → 8` is deliberate: depth 8 is today's behaviour, so the all-zero selector
reproduces the ratchet exactly.

### 3.2 The run plan's `g` integers (decoded, not assumed)

| # | `g` | hex | physical | detect | throttle | skip-zero | depth |
|---|---:|---|---|---|---|---|---|
| 1 | 33 | `0x021` | 1 | no | on | no | 8 (today) |
| 2 | 97 | `0x061` | 1 | no | on | **yes** | 8 |
| 3 | 289 | `0x121` | 1 | no | on | no | **4** |
| 4 | 545 | `0x221` | 1 | no | on | no | **16** |
| 5 | 801 | `0x321` | 1 | no | on | no | **32** |
| 7 | 97/353/609/865 | `0x061`/`0x161`/`0x261`/`0x361` | 1 | no | on | **yes** | best of B |

---

## 4. Invariants relied on

1. `MAXTOK` is a power of two (`KRN:678-683`) — unchanged, only used by B's
   untouched address math.
2. `pull_fallback == 0` in every direct-accum mode (`ADP:266-268`) — Mechanism
   A's guard re-asserts it rather than trusting it.
3. `detect_dual == false` whenever `0x40` is set — enforced by
   `config_is_valid`, so it is a *rejection*, not an assumption.
4. `sc_dst` is written by the surviving transpose for **all** `T_ext` rows,
   including hole rows, exactly as today (`HKQ:142`) — M6 reads it at
   `KRN:1321`.
5. `slots` hole/stale safety is provided by setup-zero + M8 consume-and-zero
   (`DRV:1509`, `KRN:579-587`), not by `part` — this is what makes
   `KRN:1282-1283`'s warning inapplicable.
6. The M3 and M4 grid barriers (`KRN:1236`, `KRN:1291`) and the M4 work split
   (`bid==0` plan, `bid==1` CSR, `bid>=2` transpose) are unchanged; only the
   `bid>=2` body loses its store half.
7. Depth-8 instantiation of `accumulate_rows_peer` is textually today's loop,
   with `throttle` still a runtime bool — so arm 1 is the same kernel path as
   the ratchet.
8. `K0P6_MPS_SRC_REV` is bumped 23 → 24 in the same commit (the `.hip` is the
   only file the mori JIT hashes; `.cuh`/`.cpp` edits are invisible to it).

## 5. Falsifiers, pre-registered

- **A is a null** if the in-batch control-vs-A `mps_us` delta is within
  ±1.0 % *and* `[MPS TS SPLIT] plan_M3toM5` moves by less than ~40 µs. That
  would say the M3→M4 window is set by CTA 0's serial `tile_desc` build or
  CTA 1's CSR scan, not by the 448 MiB. Log it as a measured null with the
  bandwidth arithmetic of §1.5, and the axis closes.
- **A is real** if `plan_M3toM5` drops by 50-100 µs. Then the end-to-end
  number should move by the same absolute amount and the composed candidate
  goes to a 5-rotation campaign.
- **B is a null** if all four depths sit inside ±1.0 % of each other. Then 8
  was not a lucky pick but a plateau, and the throttle axis is closed at
  screen resolution.
- **B has a better point** if any depth beats the control by more than 1.5 %
  (≈3σ of `ratio_vs_prod`); re-screen it once before composing.
- **The build is wrong** if the resource tuple moves off
  `SGPR 106 / VGPR 256 / AGPR 256 / scratch 144 B / LDS 155,496 B`, the MFMA
  census moves off `96+84=180`, or any scratch op appears inside either MFMA
  K-loop. That kills the *shape*, not the mechanism, and is reported either
  way.

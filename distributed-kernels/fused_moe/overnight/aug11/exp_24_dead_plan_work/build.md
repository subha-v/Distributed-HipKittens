# exp_24 — build, resource gate, and ISA evidence

Compiler: ROCm 7.2.53211 / AMD clang 22.0.0git (`roc-7.2.3`), container
`subha_k1`. Both arms built with the identical `--genco` line from
`../../../MPS_OVERNIGHT_HANDOFF.md` ("Repos, branches, exact build"), including
`-Rpass-analysis=kernel-resource-usage`, `-DK0P6GM_G=3 -DN2GM_G=3`,
`-mllvm -amdgpu-mfma-vgpr-form=1`.

- **base** = the node checkout at `0b82cd19`, `K0P6_MPS_SRC_REV 23` — the exp_21
  ratchet source. Snapshotted to `~/exp24-base/fused_moe` before the node was
  reset onto the exp_24 commit, so the baseline stays rebuildable.
- **exp24** = this experiment's sources, `K0P6_MPS_SRC_REV 24`.

Both `.hsaco` are offload bundles; ISA work is
`clang-offload-bundler --unbundle --targets=hipv4-amdgcn-amd-amdhsa--gfx950`
then `llvm-objdump -d --mcpu=gfx950`. Driver: `../tools/probes/p39_gate.sh`
(self-contained: it re-unbundles and re-disassembles every time, because an
earlier probe left truncated `.s` files behind and produced a clean-looking
false pass).

---

## 1. Resource tuple — before / after

| metric | base (rev 23) | exp24 (rev 24) | verdict |
|---|---:|---:|---|
| TotalSGPRs | 106 | **106** | parity |
| VGPRs | 256 | **256** | parity |
| AGPRs | 256 | **256** | parity |
| ScratchSize [B/lane] | 144 | **128** | **−16 B** |
| SGPRs Spill (metadata count) | 180 | 186 | +6, spilled to `v255` lanes via `v_readlane`, not to memory |
| VGPRs Spill (metadata count) | 16 | **14** | −2 |
| LDS [B/block] | 155,496 | **155,496** | exact parity |
| Occupancy [waves/SIMD] | 1 | 1 | parity |
| `.hsaco` bytes | 175,288 | 188,536 | +7.6 % code, one-arm-hot (see §3) |

The mode-12 budget in `overnight/aug10/CLAUDE.md` is
`SGPR 106 / VGPR 256 / AGPR 256 / scratch 144 B / LDS 155,496 B`. Every term is
met or improved; **no regression to report.**

## 2. ISA gates

| gate | base | exp24 | required |
|---|---:|---:|---|
| `v_mfma` census (all `v_mfma_f32_16x16x128_f8f6f4`) | 180 | **180** | identical |
| MFMA spans | 4 | 4 | — |
| **scratch ops inside an MFMA span** | 0 | **0** | must be 0 |
| min \|scratch − nearest MFMA\| (insns) | 512 | **664** | larger is better |
| `flat_atomic_pk_add_bf16` | 282 | **282** | identical |
| `atomic_cmpswap` | 0 | **0** | must be 0 |
| total scratch ops in the ISA | 21 | **19** | — |
| **scratch ops within 24 insns of a remote atomic** | 0 | **0** | must be 0 (see §3) |

Note on the atomic mnemonic: exp_21's result memo says
`global_atomic_pk_add_bf16`. On this toolchain the packed-bf16 RMW disassembles
as **`flat_atomic_pk_add_bf16`** (flat addressing, since the peer VA is a raw
pointer). Count and semantics are unchanged; only the printed mnemonic differs,
and it is 282 in both builds. `grep global_atomic_pk_add_bf16` returns zero on
the ratchet build too — a gate written against the wrong mnemonic would have
read "absent" and looked like a catastrophic regression.

## 3. Mechanism B: the shape, and the two shapes it took to get there

Every depth literal reaches the ISA exactly as intended, and the counts are
symmetric with the baseline's single depth:

| build | `s_waitcnt vmcnt(4)` | `vmcnt(8)` | `vmcnt(16)` | `vmcnt(32)` |
|---|---:|---:|---:|---:|
| base | — | **96** | — | — |
| exp24 | **96** | **96** | **96** | **96** |

96 is the structural count: 3 sub-blocks (`kGM = 3`, `#pragma unroll`) × 2
`epilogue_write` instantiations (`JMAX = 4` and `JMAX = 3`) × 16 rows. The
baseline's 96 `vmcnt(8)` sites and exp24's 96-per-depth confirm the emitter was
substituted, not duplicated or dropped.

### 3.1 The rejected shape, and why it was rejected

First attempt: template the whole 16-iteration accumulate loop on the depth
(`accumulate_rows_peer<JMAX, VmCnt>`) and pick with a four-way `switch` inside
`epilogue_write`. It compiled, and all four literals appeared 96 times each.
It was still wrong:

| | base | templated attempt |
|---|---:|---:|
| `scratch_load` | 9 | **436** |
| reloads of one 4 B slot (`offset:68`) | 0 | **389** |
| scratch ops within 24 insns of a remote atomic | 0 | **96** |
| `.hsaco` bytes | 175,288 | 242,168 |

The four-way join sits at the epilogue's register peak, and the allocator paid
for it by evicting a value and reloading it **once per remote atomic** — on the
depth-8 path too. That is 117 M extra private-segment loads per rank per epoch
*in the control arm*, which would have made every number in the batch
incomparable with the ratchet. Contaminating the control is worse than not
running the sweep.

### 3.2 Identifying the spilled value

Reading the disassembly around one of the 96 close scratch ops:

```
scratch_load_dword v17, off, off offset:68
v_accvgpr_read_b32 v16, a211
ds_read_b32        v19, v16 offset:256      ; xp[row][...]
v_accvgpr_read_b32 v16, a185                ; maxtok_sh
v_lshrrev_b32_e32  v16, v16, v215           ; xr >> sh
...
v_lshl_add_u32     v16, v16, 3, v17         ; (xr>>sh)*8 + <-- v17
ds_read_b64        v[16:17], v16            ; peer_tab[owner]
```

`v17` is the **LDS base address of `m7tab`**, i.e. `peer_tab` itself. The extra
state Mechanism B introduced pushed the peer table's own base out of registers.

### 3.3 The shipped shape

Two changes, both measured, both in `n2_phase2_gm_mps.cpp`:

1. **`throttle_plan`** — the depth becomes four mutually exclusive `bool`s
   computed *before* the accumulate loop, not a 5-valued integer carried into
   it. An integer needs a live VGPR and a `v_cmp` per test; four booleans
   become four SGPR lane masks, and SGPR spills go to `v255` lanes
   (`v_readlane_b32`), never to memory. The hot arm (`p.at8`) costs one mask
   test plus one branch — exactly what exp_21's `if (throttle)` cost.
   `__builtin_amdgcn_readfirstlane` pins the mode to an SGPR first, because it
   arrives through a volatile descriptor read and is otherwise not provably
   uniform.
2. **MPS-DELTA (6): `slot_off` folded into the peer table.** This is the
   register that pays for (1). `slot_off` is uniform across owners, so
   `m7tab[i] += slot_off` at table-build time is exact, and the epilogue's
   address drops from
   `tab[owner] + slot_off + (pos*kHidden + col)*2` to
   `tab'[owner] + (pos*kHidden + col)*2`. That frees the AGPR pair holding it
   (`a[186:187]`) and removes two `v_accvgpr_read` plus one `v_lshl_add_u64`
   from every remote atomic.

With both, the accidental-spill gate reads 0 and total scratch ops fall below
the baseline (19 vs 21).

**Honest confound, stated up front:** MPS-DELTA (6) means exp_24's control arm
(`g = 33`) is exp_21's protocol with a *slightly lighter* epilogue address
computation, so it is **not** byte-identical to the 6,685 µs ratchet build. It
is strictly less work, never more. Both mechanisms are read against the
in-batch control (run first and last), so their deltas stay clean; what cannot
be read off this batch is "control == ratchet". If the in-batch control lands
below the measured 6,703-6,803 µs screen band for `g = 33`, MPS-DELTA (6) is
itself a candidate worth its own paired confirmation.

### 3.4 Alternatives rejected without building

| alternative | why not |
|---|---|
| 4-way `if/else` **inside** the 16-iteration loop, guard unchanged | 3 extra scalar branches per iteration in the measured loop; changes the control arm |
| `asm("s_waitcnt vmcnt(%c0)" :: "i"(N))` | still requires `N` to be an integer constant expression |
| template the phase-2 body or the task loop on the depth | duplicates the MFMA K-loop; the census would go 180 → 720 |
| one build per depth | four JIT keys, four gate ladders, and no same-run denominator |
| hoist the wait out of `if (xtok[i] < T)` | removes the exec-mask nesting, but changes the wait count on all-padding rows — i.e. changes the control arm's behaviour, not just its schedule |

## 4. Mechanism A build notes

Mechanism A is a branch, not a code-shape change: `skip_dead_part_zero(cfg)`
selects between the verbatim donor call
(`hkp::zero_part_scale_transpose<14>`) and the new
`hk_moe::mps::scale_transpose_row`. Both loops are otherwise identical, and
every non-mode-12 config takes the donor arm. The M4 region gains one
descriptor read (`cfg4`) and one uniform branch; it contains no MFMA and sits
664+ instructions from the nearest MFMA span.

`hkp::zero_part_scale_transpose` lives in
`~/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip/hkp/hkp_quant.hpp`
(sha256 `6f10389662a7dd61dafa29ee99e40703161547e8b778d1f3bbf498b4c916d5d0`),
**outside our ownership**, so the sibling was added to `moe_mps_adapter.cuh`
instead of splitting the donor helper. `scale_transpose_row` is
`hkp_quant.hpp:142` verbatim, predicate and addressing included.

## 5. JIT-cache discipline

- `K0P6_MPS_SRC_REV` **23 → 24** in `k0pf6gm_device_tile_mps.hip`, in the same
  commit as the `.cuh` and `.cpp` edits. The `.hip` is the only file of the
  three that mori's JIT hashes (its key covers `.hpp/.h/.cpp/.hip` **inside
  mori's own `_jit-sources` tree**; ours arrive through `-I` from the read-only
  DHK mount and are invisible to it).
- Rebuild evidence per run is `screen.sh`'s `hsaco_before` / `hsaco_after`
  columns, taken through `readlink -f` + `stat -L` so the `latest/` symlink
  directory cannot fake it. **`hsaco_before != hsaco_after` on the batch's
  first point is the proof the exp_24 kernel is what was measured** — recorded
  in `result.md`.

## 6. Reproduce

```bash
# baseline (pre-exp_24 snapshot) and exp_24, same flags, then every gate:
bash overnight/aug11/tools/probes/p39_gate.sh     # via tools/nsh.ps1
```

`p32_isadiff.sh` builds the pair; `p39_gate.sh` unbundles, disassembles and
checks the tuple, the MFMA census, the scratch/MFMA locality, the
scratch-near-atomic gate and the four `vmcnt` literals. `p40_spillid.sh`
identifies a spilled value from its use site if the gate ever fails again.

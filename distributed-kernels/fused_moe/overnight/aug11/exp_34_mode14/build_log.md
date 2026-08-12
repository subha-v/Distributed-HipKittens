# exp_34 build log — CPU-only, no GPU job

Node `gbt350-odcdh2-c05-1` (8× MI350X, gfx950), container `subha_k1`,
2026-08-12. All work in a **private scratch clone** at `~/e34/DHK`; the node's
`~/Distributed-HipKittens` checkout was never touched (a GPU campaign is pinned
to it). Every build is `--genco` / `--cuda-device-only -S`; nothing was launched.

Scratch clone HEAD: `ca5b683f420e4985e8d2a790fe086419ab6730f1`, branch
`codex/distributed-hipkittens-scaffold`, clean.

Scripts (all in `../tools/`): `e34_00_scratch.sh`, `e34_01_build.sh`,
`e34_02_gate.sh`, `e34_03_bisect.sh`, `e34_04_probe.sh`, `e34_05_gate.sh`,
`e34_06_probe2.sh`, `e34_07_probe3.sh`, `e34_08_scrpos.sh`, `e34_09_probe4.sh`,
`e34_10_probe5.sh`, `e34_11_hunks.sh`, `e34_12_fix.sh`, `e34_13_fix2.sh`,
`e34_14_final.sh`, `e34_15_harness.sh`. Driver `e34_run.ps1` / `e34_sync.ps1`.

## Compile command, verbatim

```
hipcc --offload-arch=gfx950 -std=c++20 -O3 \
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
  -Rpass-analysis=kernel-resource-usage \
  -I$DHK/include \
  -I$DHK/distributed-kernels/fused_moe \
  -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip \
  -I$MR -I$MR/include -I$MR/src \
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
  --genco distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip -o B.hsaco
```
with `DHK=~/e34/DHK`, `K0=~/amd-master/auto-gpu-kernel/k0_fused_moe`,
`MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources`. Identical flags
and include paths for both arms; mirrors `../tools/e26_build.sh`. Build ≈ 7 s.
`0 errors, 0 warnings` in both arms.

**Note on the disassembly method.** `llvm-objdump -d` on a `--genco` `.hsaco`
returns a 1-line error, not ISA — the earlier `e34_02_gate.sh` run silently
reported `v_mfma=0` because of it. Any future script that reads `v_mfma=0` off a
`.hsaco` is reading an error message. Two working paths, both used here:
`hipcc --cuda-device-only -S` (37,935 / 39,911 lines of gfx950 assembly), and —
added for the review's condition 3 —

```
clang-offload-bundler --unbundle --type=o \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx950 -input=X.hsaco -output=X.co
llvm-objdump -d --mcpu=gfx950 X.co          # /opt/rocm/lib/llvm/bin/
```

which yields 34,702 lines and reproduces the `-S` census exactly (MFMA 180,
`flat_atomic_pk_add_bf16` 282). Note both binaries live under
`/opt/rocm/lib/llvm/bin` and are **not on the container's PATH**.

## Resource remark, VERBATIM

Arm **A** = pristine `ca5b683f`. Arm **B** = the delivered exp_34 files. Both
from one tree, one flag set, one run of `e34_14_final.sh`.

```
-------- A (ca5b683f) --------            -------- B (exp_34) --------
Function Name: k0pf6gm_mps_mega           Function Name: k0pf6gm_mps_mega
TotalSGPRs: 106                           TotalSGPRs: 106
VGPRs: 256                                VGPRs: 256
AGPRs: 256                                AGPRs: 256
ScratchSize [bytes/lane]: 128             ScratchSize [bytes/lane]: 128
Dynamic Stack: False                      Dynamic Stack: False
Occupancy [waves/SIMD]: 1                 Occupancy [waves/SIMD]: 1
SGPRs Spill: 186                          SGPRs Spill: 217
VGPRs Spill: 15                           VGPRs Spill: 17
LDS Size [bytes/block]: 155496            LDS Size [bytes/block]: 155496
```

## Gate table

| gate item | required | A | B | verdict |
|---|---|---:|---:|---|
| TotalSGPRs | 106 | 106 | **106** | PASS |
| VGPRs | 256 | 256 | **256** | PASS |
| AGPRs | 256 | 256 | **256** | PASS |
| ScratchSize B/lane | ≤ 128 | 128 | **128** | PASS |
| LDS B/block | ≤ 155,496 | 155,496 | **155,496** | PASS |
| Occupancy waves/SIMD | 1 | 1 | **1** | PASS |
| MFMA census | 180 | 180 (96+84) | **180 (96+84)** | PASS |
| `flat_atomic_pk_add_bf16` | 282 | 282 | **282** | PASS |
| **scratch ops inside either MFMA span** | **0** | 0 | **0** | **PASS** |
| SGPRs Spill | (not gated) | 186 | 217 | noted |
| VGPRs Spill | (not gated) | 15 | 17 | noted |
| `scratch_load` / `scratch_store` | (not gated) | 8 / 11 | 155 / 13 | **see caveat** |
| atomics with a scratch op ≤40 instrs ahead | (not gated) | 0/282 | **96/282** | **see caveat** |

**Every named gate item passes.** SGPR spills go to `v255` lanes rather than to
memory (exp_24), which is why `SGPRs Spill: 217` coexists with `ScratchSize: 128`.

### The one caveat, stated plainly

The build re-triggers **exp_26's scratch migration into the remote-atomic
epilogue**: 96 of the 282 `flat_atomic_pk_add_bf16` acquire a scratch op within
40 instructions ahead of them, against 0 in the base. Both MFMA spans stay
clean (0 inside), so the letter of the gate holds, but this is the same defect
exp_26 traced to `n2_phase2_gm_mps.cpp:145` (the `peer_tab[xr >> maxtok_sh]`
base load) and it lands in mode 12/14's hot fabric path.

Measured prior cost of that exact migration: exp_26's mask 1 (drain move + the
96 migrated accesses) was **+2.81 µs, t = 0.61, p = 0.55 — a null.** So the risk
is real but bounded and small relative to the mechanism's ±245…695 µs.

It is **not fixable inside this change's ownership**: the fix is the standing
STATUS follow-up "relieve one live VGPR in phase 2's epilogue", and that file
belongs to the phase-2 body.

## Probe ladder — how the tuple got back to 128, and what it proved

All CPU builds, ~15 s each. `sig` = atomics with a scratch op ≤40 ahead.

| arm | what it is | scratch | sgpr_sp | vgpr_sp | scr_load | sig |
|---|---|---:|---:|---:|---:|---:|
| A | pristine `ca5b683f` | 128 | 186 | 15 | 8 | 0/282 |
| W2 | **adapter changes only**, pristine kernel | 128 | 188 | 15 | 8 | 0/282 |
| B₀ | first full patch, **separate M7.7 phase** | **144** | 221 | 21 | 157 | 96/282 |
| P1 | B₀ − the `Ready=true` M8 instantiation | 144 | 225 | 21 | – | – |
| P2 | B₀ − the whole M7.7 block | 128 | 217 | 17 | – | – |
| R1 | B₀ − M7.7's publish only | 128 | 215 | 17 | – | – |
| R4 | B₀ with the IRIS descriptor + `peer_ptr` removed | 144 | 221 | 21 | – | – |
| B₁ | M7.7 **merged into M7.5** | 144 | 217 | 21 | 157 | 96/282 |
| N1 | B₁ + `readfirstlane` on `coarse75` | 144 | 215 | 21 | 157 | 96/282 |
| N4 | B₁ − the poll+acquire block | 128 | 217 | 17 | 155 | 96/282 |
| **N8** | B₁ + `readfirstlane` + poll on tid 0 + hoisted acquire dropped | **128** | 217 | 17 | 155 | 96/282 |
| X2 | N8 − the three per-task hook edits | 128 | 217 | 17 | 155 | 96/282 |
| Z_hooks | base + the per-task hook edits **alone** | 128 | 188 | 17 | 105 | **186/282** |
| Z_guard | base + the entry guard alone | 128 | 213 | 15 | 8 | 0/282 |
| Z_m8b | base + the `m8_batch` template alone | 128 | 188 | 15 | 8 | 0/282 |
| V_a | base + `m8_batch` + the whole M8 phase | 128 | 192 | 15 | 58 | 0/282 |
| V_b | base + the M7.5 block changes **alone** | 128 | 188 | 17 | 105 | **186/282** |
| V_rt | N8 with `row_ready == nullptr` instead of the template param | 128 | 249 | 17 | 148 | 96/282 |
| V_c | N8 + publish folded into the parity loop | 128 | 217 | 17 | 155 | 96/282 |
| V_d | V_c + poll relocated to the head of M8 | 128 | 217 | 17 | 155 | 96/282 |
| V_e | V_d + hooks reduced to one token | 128 | 217 | 17 | 155 | 96/282 |
| **B** | **delivered** (N8 shape, final comments) | **128** | 217 | 17 | 155 | 96/282 |

Four things this ladder establishes, none of which were in the design:

1. **The 16 B of scratch was a duplicated phase, not the mechanism.** `R4` kills
   the "IRIS descriptor / `peer_ptr` is expensive" hypothesis; `P1` kills "the
   fifth inlined M8 body is expensive". Merging the release/barrier/acquire
   sequence into M7.5, polling from one thread, and dropping the hoisted acquire
   recovered all 16 B.
2. **Attribution of the migration is non-decomposable — this is a cliff, not a
   slope.** Individually, the hooks (`Z_hooks`) and the M7.5 changes (`V_b`) each
   produce **186/282**; together with everything else the result is **96/282**;
   removing the hooks from the full patch (`X2`) changes nothing. Adding code
   moves the allocator between local minima; it does not add cost additively.
3. **Placement is irrelevant.** `V_c`, `V_d`, `V_e`, `X2`, `N4` and `B` all land
   on byte-identically `128 / 217 / 17 / 155 / 96` despite the publish, the poll
   and the hooks being in three different places. Seven arrangements, one
   allocator state.
4. **`readfirstlane` bought 2 SGPR spill slots and nothing else.** Kept because
   it is the house idiom (`n2_phase2_gm_mps.cpp:380`) and is semantically free,
   but it is not the fix the exp_26 write-up might lead one to expect.

## Host-side prerequisites for the GPU run (checked read-only)

- `K0_MPS_CFG` is parsed in `prefill_opt/host/e004pf_k0pf_ab.py:409-433` and
  requires only `C,g,mode,flush_rows`; **there is no host-side mode bound**, and
  `mps_host_bridge.cpp:24` calls `hk_moe::mps::encode_config` out of the adapter
  this change edits. So `mode=14` passes through with no harness edit.
- `mps_host_bridge` is a compiled pybind module. `encode_config` is unchanged, so
  it is functionally safe, but if it is not rebuilt it will be running the old
  header's `config_is_valid` — which is host-side advisory only (the kernel
  validates again at entry). Worth rebuilding anyway.
- `K0_MOK_POISON_OUT` defaults to `1` and is forwarded at
  `run_campaign.sh:134`. **Leave it on** — mode 14's failure mode is exactly the
  staleness shape it catches.
- `K0P6_MPS_SRC_REV` is `27` (was 26), so mori's content hash changes and a
  fresh `.hsaco` is required. Verify with `stat -L`; a cache hit is a bug.

## Second round: the drain-retention selector, the control, and the audit builds

Added after the protocol review (`e34_30_keepdrain.sh`, `e34_31_builds.sh`,
`e34_32_isa3.sh`, `e34_33_isa4.sh`, `e34_36_negctl_fix.sh`). Same flags, same
container, four builds, all `0 errors`, ~7 s each.

| build | what it is | SGPR | VGPR | AGPR | scratch | sgpr_sp | vgpr_sp | LDS | mfma | pk_add | scr_load | `buffer_inv sc0 sc1` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **K** | **the ARM**: exp_34 + the drain-retention bit | 106 | 256 | 256 | **128** | 217 | 17 | 155,496 | 180 | 282 | 155 | 8 |
| NC | protocol negative control (`R < world - 1`) | 106 | 256 | 256 | 128 | 217 | 17 | 155,496 | – | – | – | – |
| MK | audit only: `; E34_M14_ACQ_*` markers | 106 | 256 | 256 | 128 | 217 | 17 | 155,496 | 180 | 282 | 155 | 8 |
| NB | audit only: coarse M8 branch compiled out | 106 | 256 | 256 | 128 | 213 | 17 | 155,496 | 180 | 282 | 148 | **7** |

**The arm's tuple is byte-identical to the pre-review candidate (`B`) on every
field**, including the 155 `scratch_load` and the 96/282 migration signature, so
the selector is free at the allocator level. `NB`'s 8 → 7 invalidate count is the
differential that identifies the mode-14 `m8_batch` acquire; `MK` pins it
positively (see `result.md` §3).

`K0P6_MPS_SRC_REV` is now **28** (was 27 pre-review); the control carries **1028**
so it can never share a cache key with the arm.

sha256 of the delivered files, byte-identical on the node and in the local repo:

```
decc4b203fc4b069593e259976632386ef4b51702fc526f1c4ded1b7b29c554b  k0pf6gm_device_tile_mps.hip
91cc37f5696c550e6d7b8c53dc17f1dcd24bbc56f8705991eab4973aa9d2f523  moe_mps_adapter.cuh
af681cce22fe98ea0e8aaf9a974d52ce2e5742da431e9c876e30f9e1fe62cb2f  NC.hsaco (control)
```

Artifacts on the node: `~/e34/out/{K,NC,MK,NB}.hsaco`, `.dis`, `MK.s`,
`negctl.patch`, `e34_final_v2.diff` (555 lines). Control patch also delivered
here as `negative_control.patch`.

## Files delivered (uncommitted, local repo)

After the review round (`git diff --stat` in the scratch clone, vs `ca5b683f`):

```
distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip   261 ++++++++++
distributed-kernels/fused_moe/moe_mps_adapter.cuh           102 ++++
2 files changed, 338 insertions(+), 25 deletions(-)
```

Pre-review hashes, for the record:
`a8146c9c8ce88abd184e1bc73857548af0bb01bd75bf916135d96113c869bb98` (kernel),
`fa6e44ac98301ce2f5edb22b53c91d29f4541b61fcd72e77774ad85e1ce732bf` (adapter);
the delivered post-review hashes are in the section above and the gate table was
re-run against them. Full diffs on the node: `~/e34/out/e34_final.diff` (484
lines, pre-review) and `e34_final_v2.diff` (555 lines, delivered).

# exp_03 / P0 — ISA inspection of the built gfx942 mainloop (read-only, no GPU)

Artifact inspected: `overnight/build/isa/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s`
(24671 lines, 869703 B, mtime 2026-08-11 03:35 CDT). Kernel sources mtime
2026-08-11 03:28:37, so the ISA is CURRENT — no rebuild was performed.

## Q1 — the 2 spills / 12 B scratch are NOT in the k-loop

Exactly 4 scratch instructions per 256/256/32 symbol. Loop nest of
`<256,256,32,false>` (function lines 9208..14172):

| loop | header | depth | layout range | role |
|---|---|---|---|---|
| tile loop | `BB3_84` | 1 | 10516..14005 | once per output tile |
| **k-loop** | `BB3_87` | **2** | **10957..11260** | `k_iters` x per tile (116 on shape 6) |

| line | instruction | location | executed |
|---|---|---|---|
| 10287 | `scratch_store_dword off, v9, off` | not in any loop | once per CTA |
| 10514 | `scratch_store_dword off, v2, off offset:4` | not in any loop, immediately before `s_branch .LBB3_84` | once per CTA |
| 10622 | `scratch_load_dword v10, off, off` | in `.LBB3_84` (tile-loop header), k=0 prologue global->LDS copy | once per tile |
| 11401 | `scratch_load_dword v142, off, off offset:4` | in `.LBB3_101` = `error_bit_set` exit, per-tile epilogue | once per tile |

Same for `<256,256,32,true>`: stores 15298 / 15553 (CTA prologue), reloads
15672 / 16586 (per tile). k-loop = `BB4_86` Depth=2, layout 16012..16438.

**Zero scratch instructions inside either k-loop.** Verdict: the spill costs
~2 instructions per tile, i.e. ~2 per 116 k-iterations on shape 6. Negligible.
E1(a) is worthless as a spill fix even if AGPRs could be forced on.

Slot detail: offsets 0 and 4 are used (8 B live) but
`.private_segment_fixed_size` is 12 — 4 B of padding.

## Q2 — per-k-iteration inventory, `<256,256,32,false>`, per wave

3 basic blocks. Execution order: header -> (guarded) prefetch -> MFMA/latch.

| block | lines | contents |
|---|---|---|
| `.LBB3_87` (HEADER) | 11029..11152 | 24 x `ds_read_b64` (inline asm), **no waitcnt**; ends `s_add_i32 s10,s10,1` / `s_cmp_ge_i32 s10,s88` / `s_cbranch_scc1 .LBB3_86` |
| `%bb.88` (guarded) | 11153..11260 | 4 x `global_load_dwordx4`, 8 x `ds_write_b64`, 2 x `s_waitcnt vmcnt(0)`, 2 x `s_waitcnt lgkmcnt(0)` — ALL inline asm; ends `s_branch .LBB3_86` |
| `.LBB3_86` (latch) | 10957..11028 | 64 x `v_mfma_f32_16x16x16_bf16` + 1 bare `s_barrier` at L10966; ends `s_cbranch_scc1 .LBB3_89` (exit) |

Program order (execution order):

```
ds_read_b64 x24                      (L11041..L11147, inline asm, no wait)
  [guard: k+1 >= k_iters ? skip]
global_load_dwordx4 x2               (L11164, L11171)
s_waitcnt vmcnt(0)                   (L11175, inline asm)
ds_write_b64 x4                      (L11178..L11203)
s_waitcnt lgkmcnt(0)                 (L11213, inline asm)
global_load_dwordx4 x2               (L11216, L11223)
s_waitcnt vmcnt(0)                   (L11227, inline asm)
ds_write_b64 x4                      (L11230..L11254)
s_waitcnt lgkmcnt(0)                 (L11258, inline asm)
v_mfma_f32_16x16x16_bf16 x3          (L10958..L10965)
s_barrier                            (L10966, compiler, NO lgkmcnt with it)
v_mfma_f32_16x16x16_bf16 x61         (L10967..L11027)
```

Absent from the k-loop (and from the whole TU): `buffer_load` 0,
`global_load_lds` 0, `s_setprio` 0, `sched_barrier` 0, `sched_group_barrier` 0,
`v_accvgpr` 0, `v_mfma_f32_32x32*` 0, `ds_read2` 0, `ds_read_b128` 0 in-loop.

All 4 in-loop waits are N=0. 126 counted `vmcnt(N>0)` waits DO exist in the TU
(vmcnt(1..7)) but every one is in the reducer accumulate path (lines 9679..9851
for this symbol, ahead of the tile loop) — 0 inside either k-loop. Counted
waits are therefore proven available on gfx942 in this exact TU.

### Overlap verdict
Confirmed, and stronger than pre-registered: only **2** `global_load_dwordx4`
are ever in flight before a `vmcnt(0)`, so the 32 KB BK=32 slab is drained in
**two serialized halves** — two exposed global round trips per k-iteration, not
one. No MFMA is issued between any global load and its `vmcnt(0)`.

One overlap the pre-registration missed: the 24 `ds_read_b64` are issued
*before* the global loads and are not drained until L11213, so LDS-read latency
IS hidden behind the global issue + first `vmcnt(0)`. "No global/MFMA overlap"
holds; "no overlap at all" is too strong.

### FLAGGED HAZARD — last-k-iteration LDS race
`s_cbranch_scc1 .LBB3_86` (L11152) skips `%bb.88` when `k+1 >= k_iters`. On
that path the 24 `ds_read_b64` feed the 64 MFMAs with **no `s_waitcnt
lgkmcnt(0)` between them** — the only lgkmcnt(0) lives in the skipped block —
and the `s_barrier` at L10966 is bare. Cause: every LDS op is inside
`asm volatile`, so `SIInsertWaitcnts` never observes an LDS event and believes
lgkmcnt is already 0, emitting neither the use-wait nor the `__syncthreads()`
wait. Identical in `<256,256,32,true>` (L16135 guard, bare `s_barrier` L16373).

### Naming discrepancy
The mnemonic is `v_mfma_f32_16x16x16_bf16`. `grep -c v_mfma_f32_16x16x16bf16_1k`
returns **0** — the `_1k` form is the gfx90a spelling.

## Q3 — unroll factor = 1

64 MFMAs = (BM/16)(BN/16)(BK/16)/8 waves = 16*16*2/8 = 64 -> exactly one
k-iteration. One `s_barrier`. 4 x `global_load_dwordx4`/wave = 4*64*16 B * 8
waves = 32768 B = (BM+BN)*BK*2 = exactly one BK=32 slab. One loop counter
increment `s_add_i32 s10, s10, 1` (step 1) against `s88` = k_iters. Not
unrolled; Q2 counts are already per-iteration.

## Q4 — per-symbol metadata (ground truth, from the amdhsa YAML note)

| instantiation | lines | VGPR | AGPR | SGPR | scratch | vgpr_spill | sgpr_spill | maxwg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `<32,256,32,false>` | 8..2780 | 94 | 0 | 106 | 0 | 0 | 57 | 512 |
| `<64,64,64,false>` | 2826..5300 | 93 | 0 | 106 | 0 | 0 | 49 | 512 |
| `<128,256,32,true>` | 5346..9162 | 170 | 0 | 106 | 0 | 0 | 73 | 512 |
| `<256,256,32,false>` | 9208..14172 | 256 | 0 | 106 | **12** | **2** | 61 | 512 |
| `<256,256,32,true>` | 14218..19374 | 256 | 0 | 106 | **12** | **2** | 73 | 512 |
| `<32,64,64,false>` | 19420..21827 | 91 | 0 | 106 | 0 | 0 | 54 | 512 |
| `<32,64,64,true>` | 21873..24443 | 91 | 0 | 106 | 0 | 0 | 80 | 512 |

`group_segment_fixed_size` is 0 for all (LDS is dynamic / `extern __shared__`).
`uses_dynamic_stack: false` everywhere.

## Q4 — regeneration command

The current `.s` was produced by `tools/m2_isa.sh` (its `--save-temps` step).
To regenerate only the ISA, without touching `harness/build/*.so`:

```bash
docker exec dhk-gemmrs bash /home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/tools/m2_isa.sh
```

Equivalent explicit compile (single TU, `-c`, no link, ~40 s):

```bash
REPO=/home/subvadla/dhk; ISA=$REPO/distributed-kernels/gemm_rs/overnight/build/isa
hipcc -std=c++20 -O3 -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
  -ffast-math --offload-arch=gfx942 -DTK_MODNAME=gemm_rs_mi300x \
  -I$REPO/include -I$REPO/include/pyutils -I/opt/rocm/include/hip \
  -I$(python3 -c 'import pybind11;print(pybind11.get_include())') \
  -I$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])') \
  -Wno-nan-infinity-disabled --save-temps -c \
  $REPO/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp -o $ISA/gemm_rs_mi300x.o
# -> $ISA/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s
```

Then re-run the two analyzers in this directory (both pure text, no GPU):

```bash
EXP=$REPO/distributed-kernels/gemm_rs/overnight/experiments/exp_03_mainloop
python3 $EXP/analyze_isa.py $ISA/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s $EXP
python3 $EXP/kloop_isa.py  $ISA/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s $EXP
```

`analyze_isa.py` gives the symbol/metadata table + scratch attribution;
`kloop_isa.py` gives the basic-block-accurate k-loop inventory and rewrites
`isa/kloop_*.s`. Neither needs a GPU. Note that after any mainloop change the
line numbers move, so always re-read the block table rather than reusing the
line numbers above.

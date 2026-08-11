# exp_03 — E1, the GEMM mainloop: the axis is real, the pre-registration was wrong

## What the analysis overturned

`CLAUDE.md` pre-registered **E1(a), AGPR accumulators**, expecting them to clear
the 2 VGPR spills / 12 B scratch on the three 256×256 rows where `AGPRs: 0`.
**That expectation is unachievable, for a reason that is worth writing down so
nobody re-tries it.**

`gemm_rs_mi300x.cpp:123` declares `__launch_bounds__(CTA_THREADS, 1)` with
`CTA_THREADS = 512`. 512 threads is 8 waves that must be co-resident on one
CU's 4 SIMDs, so LLVM's `getWavesPerEUForWorkGroup(512) = 2` and
`getMaxNumVGPRs(F) = alignDown(512/2, 8) = 256`, which is exactly
`getAddressableNumArchVGPRs()`. In `SIMachineFunctionInfo`:

```
if (ST.hasGFX90AInsts())
  if (MFMAVGPRForm ||
      (ST.getMaxNumVGPRs(F) <= ST.getAddressableNumArchVGPRs() && !mayUseAGPRs(F)))
    MayNeedAGPRs = false;
```

Both disjuncts hold. `256 <= 256` is true, and `mayUseAGPRs(F)` is false
because `AAAMDGPUNoAGPR` proves `amdgpu-agpr-alloc=0` for any function with no
AGPR-constrained inline asm — and there is no `"a"` constraint anywhere in
`include/cdna3/**`. Independently, LLVM flipped `-amdgpu-mfma-vgpr-form` to
`cl::init(true)` in llvm/llvm-project#159493, so recent ROCm defaults to VGPR
form regardless.

The decisive detail: on gfx942 the register file is **unified**. Enabling AGPRs
re-partitions the same 256 registers into ~128 arch + ~128 acc; **it does not
create a 257th register.** A 2-value over-subscription of a 256-register budget
cannot be fixed by re-classing registers. And the `AGPRs: 0` *together with*
12 B of scratch is the signature of a disabled AGPR file, because when AGPRs
are available a spill goes to `v_accvgpr_write_b32` long before it goes to
scratch.

**Consequence: E1(a) is closed as a spill fix before it costs a GPU minute.**
It is still worth one cheap build to settle empirically (P1/P2 below), logged
either way, but it is not the mainloop win.

## What the analysis found instead — the mainloop has zero latency overlap

Per k-iteration (`gemm_rs_mi300x.cpp:206-221`), in program order:

| # | line | operation | lowered form |
|---|---|---|---|
| 1 | 207 | `load(A_frag, subtile(As[k&1]))` | 16 × `ds_read_b64`, no waitcnt |
| 2 | 208 | `load(B_frag, …)` | 8 × `ds_read_b64` |
| 3 | 209-11 | `G::load(As[(k+1)&1])` | `global_load_dwordx4` → **`s_waitcnt vmcnt(0)`** → `ds_write_b64` → **`s_waitcnt lgkmcnt(0)`** |
| 4 | 209-11 | `G::load(Bs[(k+1)&1])` | the same again: **second `vmcnt(0)`, second `lgkmcnt(0)`** |
| 5 | 216-18 | `mask_a_k_tail` (K_TAIL rows only) | VALU into A_frag |
| 6 | 219 | `mma_ABt(C_accum, A_frag, B_frag, C_accum)` | 64 × `v_mfma_f32_16x16x16bf16_1k` |
| 7 | 220 | `__syncthreads()` | `s_waitcnt lgkmcnt(0)` + `s_barrier` |

**Two `vmcnt(0)`, two `lgkmcnt(0)`, one barrier, every iteration.** The
"prefetch" of the next k-block is a *synchronous* copy: it is issued and then
drained to completion before control can reach the MFMAs. The double buffering
is real in the LDS allocation (`As[2]`, `Bs[2]`) and in the dependency
structure, but it buys **no overlap at all** — the global-memory round trip is
fully exposed inside every iteration.

Nothing else can hide it either. `LDS_BYTES = 2*(BM+BN)*BK*2` is 65536 on the
256/256/32 rows, i.e. the entire 64 KB, so residency is **1 CTA/CU = 2
waves/SIMD** and there is no second wave on the SIMD to cover the stall. The
in-wave schedule is the only latency-hiding mechanism available.

This is a mechanical explanation of the independently-derived per-k-iteration
cost: **2.9–3.1× off single-CU peak on both 256/256/32 shapes, 5.2× on shape 1**
(`exp_04/plan.md`). The two lines of evidence agree.

## Why E1(c) must not be run first

Every one of these helpers is `asm volatile` with a `"memory"` clobber
(`include/cdna3/ops/warp/memory/**`, `common/util.cuh:33-40,131-141`). They
fragment the mainloop into tiny scheduling regions and pin memory ordering, so
LLVM cannot software-pipeline across them **even if the waitcnts were relaxed**.
`sched_group_barrier` has nothing to reorder. **E1(c) is a no-op until E1(b)
lands.** Sequence accordingly — this ordering claim is the most actionable
thing in the analysis after the drain inventory itself.

## The intervention — E1(b), split issue from commit

The fix is the standard software pipeline: hoist the *issue* of the next
iteration's global loads above the MFMAs, and sink the *wait and LDS commit*
below them, so the global round trip is covered by 64 MFMAs of real work.

```
prologue:  issue global k=0 -> wait -> ds_write As[0]/Bs[0] -> barrier
per k:
    issue   global loads for k+1 into register staging   (NO waitcnt)
    ds_read A_frag,B_frag from As[k&1],Bs[k&1]
    s_waitcnt lgkmcnt(0)                                  (ds_reads only)
    mma_ABt  x64                                          <-- covers the global latency
    s_waitcnt vmcnt(0)                                    (loads have had 64 MFMAs to land)
    ds_write staging -> As[(k+1)&1], Bs[(k+1)&1]
    __syncthreads()
```

This requires splitting HK's monolithic `G::load` into an `issue` half and a
`commit` half. The helper currently fuses global-load, `vmcnt(0)`, `ds_write`
and `lgkmcnt(0)` into one call
(`include/cdna3/ops/warp/memory/tile/global_to_shared.cuh:37,55,73`), which is
precisely why no overlap is possible. The register staging buffer already
exists (`float4 buf[16]`, ~8 VGPRs live at `BK=32`), so the split costs little
extra pressure.

The gfx950 sibling `gemm_rs_device_tile.cpp` does exactly this and goes
further, using **counted** waits (`vmcnt(4)`, `vmcnt(6)`, `vmcnt(2)`,
`lgkmcnt(8)` at `:863,870,880,916,927,961,997,1014`) to keep 4–6 loads in
flight across every MFMA group, plus `sched_barrier(0)` and `s_setprio` around
each group. **Counted waitcnts and `sched_barrier` are gfx90a+ builtins and
port cleanly to gfx942.**

**Portability trap, must not be missed:** the donor's `G::load` lowers to
`llvm_amdgcn_raw_buffer_load_lds` with `bytes_per_thread = 16`. gfx942 supports
`buffer_load … lds` at **1/2/4 bytes per lane only**; the 12 B and 16 B
LDS-direct widths are gfx950-only. A faithful port of the donor's DMA path
needs 4× the instructions on gfx942 — so take the *counted-wait pipeline* from
the donor, not its LDS-direct DMA.

## The numerical hazard this creates

`gemm_rs_device_tile.cpp:786-812` records, in the donor's own words, a **30×
numerical error that still passed a 2e-2 correctness gate**, caused by a single
extra `s_load` the scheduler hoisted into a hand-counted `lgkmcnt(N)` region.
Counted waits make the count a load-bearing invariant that the compiler can
silently invalidate.

Therefore, for every E1(b) arm: the **`2e-3` tight gate is the acceptance
criterion, not `1e-2`**, and the ISA must be re-inspected after every build to
confirm the instruction counts the waits are counting. The donor's mitigation
(SGPR pinning, `:814-822`) is available if we hit the same failure.

## Ordered plan

| step | what | GPU? | risk | why this order |
|---|---|---|---|---|
| **P0** | read the built ISA: are the 2 spills inside the k-loop or in the tile epilogue? | no | none | decides whether the spill is worth any time at all; free |
| **P3** | split the k-step into two `BK/2` halves so `A_frag`+`B_frag` live pressure drops ~24 regs | yes | low, bit-exact | smallest diff that actually removes pressure; `mma_ABt` already chains k ascending so per-element addition order is preserved |
| **P1+P2** | `-mllvm -amdgpu-mfma-vgpr-form=0`, plus an `asm volatile("" : "=a"(x))` anchor to defeat the `amdgpu-agpr-alloc=0` inference | yes | nil / very low | settles the AGPR question empirically in one build; log as settled either way |
| **E1(b)** | issue/commit split as above | yes | **medium** | the actual 46%-pool intervention |
| **E1(c)** | `sched_group_barrier` interleave | yes | medium | only meaningful after E1(b) |

P3's one real correctness item: `mask_a_k_tail` (`gemm_rs_mi300x.cpp:96-117`,
applied at `:216-218`) assumes the fragment spans K columns `[0, BK)`. Under a
half-BK split the threshold must be rebased to `tail_k - kk` and clamped so a
half entirely past the tail is zeroed. **This affects config rows 3 and 6 only**
(`K_TAIL=true`), so M3 at `2e-3` on 2048×2880×2880 and 8192×8192×29568 is the
acceptance test.

## Note on reporting P1/P2 if they do land AGPRs

Occupancy will **not** improve — the file is unified and residency is LDS-bound
at 1 CTA/CU anyway. Any `result.md` must say so explicitly so a re-classing
change is not written up as an occupancy win. There is also a real *cost*: with
the accumulator in AGPRs every epilogue read at `hk_adapter.cuh:123` becomes a
`v_accvgpr_read_b32`, and the epilogue rescans the whole accumulator once per
staging window — 8 windows × 128 registers on the 256×256 rows. That
interaction is E6's territory and is the reason to fix E6 before judging P2.

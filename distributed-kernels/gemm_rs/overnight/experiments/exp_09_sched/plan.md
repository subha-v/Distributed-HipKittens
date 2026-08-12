# exp_09_sched — plan (E1(c), mainloop instruction scheduling)

Written before the arms ran; kept unedited so the pre-registered expectations
can be checked against `result.md`.

## Question

Shapes 5 and 6 are the only two we still lose to the reference GEMM+RCCL
(1.13× and 1.24×) and GEMM is their dominant pool after exp_08 cut the egress
pool from 1149.9 to 412.4 µs (shape 6: 1142.8 of 1777.9 µs, 64%). The producer
mainloop issues all four `global_load_dwordx4` in a clump before ~64 MFMAs.
**Does redistributing instruction issue inside the k-loop buy anything?**

## Arms, in intended order

- **A — spread the global loads through the MFMA block** with
  `__builtin_amdgcn_sched_group_barrier` / `sched_barrier`. The intervention
  E1(b) explicitly left on the table.
- **B — `s_setprio` around the MFMA groups**, as the gfx950 donor does
  (`gemm_rs_device_tile.cpp:884-935`). Cheap, independent of A, composable.
- **C — counted `s_waitcnt`** in `load_commit`, only if A or B wins and time
  allows. 126 counted `vmcnt(1..7)` already exist in the reducer path so the
  encoding is available, but the donor records a 30× numerical error that still
  passed a 2e-2 gate from exactly this, so acceptance is the **2e-3** gate and
  the ISA must be re-read after every build.

## Pre-registered expectations

1. **Arm A will not work as specified, and the reason is structural.** Two
   things say so before any build: the four loads sit in their own basic block
   (`%bb.87`) separate from the MFMAs (`.LBB3_88`), and a scheduling region never
   spans a basic block; and `load_global_vec4_async` is `asm volatile ... :
   "memory"`, as is every `ds_read`, `ds_write` and `s_waitcnt` in this tree, so
   their mutual order is fixed whatever the scheduler is told. Expect a no-op,
   and report it as a finding rather than a failure. Then remove the branches so
   the loads and MFMAs share one region, which is the prerequisite for anything
   in this axis.
2. **Spreading VMEM issue may well be a loser even if it becomes expressible**,
   because moving loads later reduces the number of MFMAs covering them and the
   coverage already looks marginal: 64 MFMAs at 4-cycle issue with 2 waves/SIMD
   is roughly 512 cycles per wave-pair against an HBM round trip of ~600-900.
3. **Shape 6 should benefit most from de-branching**, because its
   `<256,256,32,true>` instantiation is **ten** basic blocks against six for
   `false` — the extra splits come from the `k == k_iters - 1` K-tail compare,
   and `k_local = 3696` with `BK = 32` means 116 k-iterations all paying for a
   tail only the last one can take.
4. **Register headroom is the binding constraint on any restructure.** The
   256/256/32 pair is at 246/248 against the 256 cap with zero spills; anything
   that duplicates the MFMA block risks spilling and must be rejected at M2.

## Method

Every arm: ISA first (`isa_arm.sh`, no GPU, no `harness/build/*.so`), and only
then the full ladder. `sched_isa.py` finds the k-loop by LLVM's `Depth=`
annotations so it survives edits. `lds_race_check.sh` after any mainloop edit.
Report the M2 tuple every build. Never widen a tolerance; `1e-2` and `2e-3` are
both gates. Sub-5% deltas get an alternating paired A/B against a base rebuilt
in the same session, because the recorded 230.84 µs denominator was measured
earlier under conditions I cannot reproduce exactly.

## Files

Owned: `gemm_rs_mi300x.cpp`, `gemm_rs_mi300x_hk_adapter.cuh`, this directory.
Read-only: `gemm_rs_device_tile.cpp` (gfx950 donor), `include/**`. Not touched:
the egress path, `WGM`, `gemm_rs_mi300x_constants.cuh`,
`gemm_rs_mi300x_host_abi.hpp`, harness and tools.

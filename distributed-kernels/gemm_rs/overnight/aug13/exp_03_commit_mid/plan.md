# exp_03 (aug13) — exp_27 design row B: commit between the MFMA halves

## Provenance

exp_27's design (aug11) pre-registered this as the first mainloop code arm and
it was never built (no result.md in exp_27). This experiment executes that
registration as written, with its gate plan and falsifier, adapted only in
tooling (the paired instrument is exp_24's `ab_prev_mp.py`, the strongest
same-run design this project has, with the Latin-square complete-permutation
ordering and both allocation orders — the exp_26 lesson).

## Mechanism

Today the k-loop is: issue global loads → 64 MFMAs (2 unrolled halves) →
`load_commit`×2 (vmcnt(0) + 8 ds_write + lgkmcnt(0)) → `__syncthreads()`. The
commit sits on the pre-barrier tail — every wave drains it with nothing to
overlap (exp_27 window W2, inferred 500–800 cycles/trip shared with W1).

The arm (`HK_GEMM_RS_MI300X_COMMIT_MID=1`) moves the two `load_commit`s to
after the `kh==0` half, so half 1's 32 MFMAs cover the drain and the barrier
is reached clean. The trade: the global load's vmcnt(0) now waits 32 MFMAs
after issue instead of 64. `acc_anchor` before the mid commit pins half 0's
MFMAs above it (the data-dependence mechanism; hints are no-ops in this TU,
exp_09 A0/A2).

Buffer safety is the straight-line argument already in the source: the commit
writes `As[(k+1)&1]`, which iteration k never reads; the barrier ending k
still separates it from its readers in k+1.

## Pre-registered (copied from exp_27 §6, unchanged)

- Shape 6 GEMM ablation pool −5.0% ± 3.0%; shape 6 wall ~−3.1%; shape 5 wall
  ~−2.3%; geomean −0.9%. Shapes 1–4 move less than their floors — not credited.
- **Falsifier:** same-run paired A/B, ≥3 draws with construction order
  reversed on half, |Δ| < 1.5% on shape 6 while the ISA census confirms the
  vmcnt(0) placement ⇒ window W2 < 100 cycles/trip and the whole B/C/D family
  is closed at <3%; the night's next target becomes row E.
- Converse: Δ at/beyond −7% ⇒ W1/W2 are the residual; build row C next and
  revise row E's ceiling down.

## Gates (exp_27 §7, in order)

1. M1: `build_arms.sh` — `gemm_rs_mi300x_cmid` (flag=1), `gemm_rs_mi300x_cmid0`
   (flag=0), `gemm_rs_mi300x_cmid0b` (flag=0 twin). Production
   `build/gemm_rs_mi300x.so` is NOT touched.
2. M2 + placement: `census.sh` (CPU-only `--save-temps`) + `assert_placement.py`:
   - flag-0 ISA byte-identical to exp_27's archived census (default-off
     ratchet, exp_38 discipline);
   - cmid `<256,256,32,*>` innermost loop: ≥25 MFMA, then vmcnt(0), then ≥25
     MFMA before `s_barrier`; baseline: all MFMAs before the first vmcnt(0);
   - VGPR ≤ 248, 0 spills, 0 scratch on every instantiation.
3. `exp_03_mainloop/lds_race_check.sh` against the cmid build.
4. M3 all 17 shapes, 1e-2 AND 2e-3, via `HK_KERNEL_MODULE=gemm_rs_mi300x_cmid`.
5. M4 negative controls (control module, candidate-agnostic).
6. M5 600-epoch soak on the cmid module.
7. M9 stale-slot as a NULL (publication order untouched; a change = the edit
   did something it was not supposed to).
8. M7: `ab_cmid_mp.py` shapes 5 and 6, ≥42 rounds × both allocation orders,
   within-round paired median, bit-exactness asserted across arms every rank.

Ship rule: cmid wins in BOTH allocation orders with the null contrast inside
its floor, or it does not ship (burden of proof per the exp_26 withdrawal).

# exp_27 — where the ~571 µs actually is, and what can take it

**Verdict up front, in three parts.**

1. **The dispatch's leading hypothesis is dead on arithmetic.** Single-buffered
   `BK=64`, and every "keep half of `BK` resident" variant, changes the barrier
   count per unit of MFMA work by **exactly zero**. `<256,256,32>` with two LDS
   stages is the *argmax* of MFMA-work-per-barrier under the joint 64 KB LDS and
   256-VGPR accumulator caps (§4). Directions 1 and 2 are closed, and they should
   be struck from `HANDOFF.md`'s "open lever" line, which currently points the
   next session at them.
2. **The mainloop is not throughput-bound on any pipe.** Counted from the ISA,
   the per-trip occupancy of every pipe — MFMA 2048, LDS 1024, VALU 784, L1 512,
   scalar 368 cycles — is at or below the 2048-cycle MFMA floor, against a
   measured 4063-cycle trip (§3). **Roughly 2015 cycles per trip, ~492 µs of
   shape 6, is pure non-overlap.** Static analysis accounts for 500–800 of those
   cycles structurally and **cannot account for the other ~1200–1500**. Saying
   which of {LDS wait, VMEM wait, barrier skew} owns that residual is one
   20-minute counter pass, and every candidate's expected value swings 3–5×
   depending on the answer.
3. **The recoverable amount is much smaller than 571 µs.** 571 µs is the
   distance to a perfect-overlap roofline that no real GEMM reaches. We already
   run at **50.2% of producer-CU bf16 peak**; a well-tuned MI300X library GEMM
   runs at 60–75%. A realistic ceiling is ~250 µs on shape 6 and ~90 µs on
   shape 5, which is **−5.1% at the geomean** and takes the graded gap from
   1.098× to ~1.046× (§7). The single cheapest arm on the list is worth −0.9%
   at the geomean. Both numbers are small, and they are the honest ones.

---

## 1. Provenance

Censused source, compiled 2026-08-12T11:06:49Z inside `dhk-gemmrs`
(HIP 7.2.53211), host-side `-c --save-temps`, no GPU:

| file | sha256[0:16] |
|---|---|
| `gemm_rs_mi300x.cpp` | `5f9f47258842c64d` |
| `gemm_rs_mi300x_hk_adapter.cuh` | `e9b3d1c17d49ee0b` |
| `gemm_rs_mi300x_constants.cuh` | `c993b2a6004a6977` |

Resource tuple, **7/7 instantiations, 0 AGPRs, 0 scratch, 0 VGPR spills** —
confirming the dispatch's statement and E1(a)'s closure independently:

| instantiation | VGPR | AGPR | SGPR | scratch | VGPR spill | SGPR spill | waves/SIMD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `<32,64,128,false>` | 98 | 0 | 106 | 0 | 0 | 54 | 4 |
| `<64,128,64,false>` | 104 | 0 | 106 | 0 | 0 | 60 | 4 |
| `<128,192,32,true>` | 136 | 0 | 106 | 0 | 0 | 79 | 3 |
| **`<256,256,32,false>`** (shape 5) | **246** | 0 | 106 | 0 | 0 | 63 | **2** |
| **`<256,256,32,true>`** (shape 6) | **248** | 0 | 106 | 0 | 0 | 79 | **2** |
| `<32,64,64,false>` | 91 | 0 | 106 | 0 | 0 | 58 | 5 |
| `<32,64,64,true>` | 92 | 0 | 106 | 0 | 0 | 88 | 5 |

Whole-TU: `v_mfma_f32_16x16x16` ×184, `v_mfma_f32_32x32x8` **×0**,
`ds_read_b64` ×110, `ds_read_b128` ×7 (epilogue only), `ds_write_b64` ×84,
`global_load_dwordx4` ×98, `s_barrier` ×98, `scratch_store` ×0, `v_accvgpr` ×0.

**E1(a) restated so it is not re-litigated.** `__launch_bounds__(512,1)` with
512 threads gives 8 waves/CTA on 4 SIMDs = 2 waves/SIMD unconditionally, and
gfx942 splits a 512-register-per-slot file into 256 **unified** registers per
wave at 2 waves/SIMD. 248 VGPR + 0 AGPR = 248 of 256. Re-classing to AGPRs
re-partitions the same 256; it cannot add one register. Closed, and the census
above is the current evidence for it.

## 2. The k-loop, COUNTED

Per wave, per trip of the innermost `v_mfma`-bearing natural loop. Every number
in this table is read off the ISA; none of it is modelled.

| class | `<256,256,32,false>` (shape 5) | `<256,256,32,true>` (shape 6) |
|---|---:|---:|
| `v_mfma_f32_16x16x16_bf16` | 64 | 64 |
| `ds_read_b64` | 24 | 24 |
| `ds_write_b64` | 8 | 8 |
| `global_load_dwordx4` | 4 | 4 |
| `s_waitcnt lgkmcnt(0)` | 4 | 4 |
| `s_waitcnt vmcnt(0)` | 2 | 2 |
| `s_barrier` | 1 | 1 |
| VALU | 58 | **98** |
| SALU | 24 | **46** |
| branch | 2 | 3 |
| `s_nop` | 3 | 3 |
| **instructions / trip** | **194** | **257** |
| **basic blocks in the loop** | **1** | **4** |

Four things fall out immediately.

**(a) Every wait is a full drain.** 4 × `lgkmcnt(0)` and 2 × `vmcnt(0)`, zero
counted waits. That is deliberate (`gemm_rs_mi300x_hk_adapter.cuh:128`, citing
the donor's `lgkmcnt(N)` disaster at `gemm_rs_device_tile.cpp:786-812`) and it
should stay deliberate — but it means every one of the six is a serialization
point, and any mechanism that moves one is a memory-ordering change.

**(b) Shape 6's instantiation costs 63 extra instructions per trip (+32%) and
runs as 4 basic blocks against shape 5's 1**, entirely because `K_local = 3696`
is not a multiple of `BK = 32` and `K_TAIL=true` puts the `mask_a_k_tail`
guard inside the unrolled `kh` loop. exp_09's central finding is that **a
basic-block boundary is a scheduling-region boundary** — the machine scheduler
cannot move an instruction across one. Shape 6, our single worst row, is
compiled into the form that most restricts the scheduler.

**(c) `%bb.92` is 58 instructions of pure address arithmetic** — 40 VALU + 18
SALU, zero memory operations, zero math — executed every trip, sitting between
the `ds_read` issue block and the MFMA block. It exists because
`subtile_inplace(As[k & 1], …)` re-derives the swizzled LDS address of all 24
`ds_read`s from a base that alternates every iteration, so nothing can be
hoisted out of the loop.

**(d) The loop is rotated**, and the rotation is what makes the two MFMA-idle
windows visible:

```
.LBB4_91   12 ds_read_b64 (half 0)  +  4 global_load_dwordx4 (ISSUE, k+1)
%bb.92     58 pure address ops
.LBB4_93   12 ds_read_b64 (half 1)  +  lgkmcnt(0)  +  32 MFMA (half 0)
.LBB4_90   32 MFMA (half 1)  +  2x[vmcnt(0), 4 ds_write_b64, lgkmcnt(0)]  +  s_barrier
```

## 3. The per-trip cycle budget

**Calibration (MEASURED, from exp_20).** Shape 6 runs 4 tile-waves × 116
k-iterations = **464 trips** per producer CTA. Its GEMM ablation pool is
992.2 µs; at the pinned 1900 MHz that is 1,885,180 cycles, i.e.
**4063 cycles/trip**. Shape 5 runs 2 × 56 = 112 trips against a 305.2 µs pool =
**5177 cycles/trip** (shape 5's last wave is only 88% full, so this is an upper
bound on the critical CTA's trip; the shape-6 figure is the cleaner one because
its last-wave fill is 100%).

**The MFMA floor (MEASURED).** exp_20 g4 read `SQ_VALU_MFMA_BUSY_CYCLES /
SQ_INSTS_MFMA = 16.0`, exactly the architectural rate for
`v_mfma_f32_16x16x16_bf16` (4096 MAC / 256 MAC-per-cycle-per-SIMD). Per SIMD
per trip: 2 waves × 64 MFMA × 16 cyc = **2048 cycles**. This is not a model —
the counter says the pipe issues at full rate whenever it is busy.

> Note on denominators: exp_20's "42.5% MFMA share" divides by all 304 CUs,
> including the 48 reducers, which is right for a device-wide claim. For a
> *schedule* claim the denominator is the 256 producer CUs, and there the MFMA
> pipe is busy 500.1 µs of the 992.2 µs pool — **50.4%**, i.e. the trip is
> almost exactly 2× its MFMA floor.

**Pipe occupancy per trip, shape 6.** Bytes and instruction counts are COUNTED
(§2); the rate applied to each is named, and only the MFMA rate is measured.

| resource | scope | quantity (counted) | rate | cycles | note |
|---|---|---|---|---:|---|
| MFMA | per SIMD | 2×64 instr | 16 cyc/instr **(measured)** | **2048** | the floor |
| VALU | per SIMD | 2×98 instr | 4 cyc/wave64 op (arch) | 784 | 464 on shape 5 |
| LDS | per CU | 8×32 instr × 512 B = 131,072 B | 128 B/clk **(assumed)** | 1024 | 512 if 256 B/clk |
| L1 / global | per CU | 8×4 × 1024 B = 32,768 B | 64 B/clk **(assumed)** | 512 | |
| scalar unit | per CU | 8×46 ops | 1 op/clk (arch) | 368 | |
| issue slots | per SIMD | 2×257 | 1 instr/clk (arch) | 514 | |
| **measured trip** | | | | **4063** | |

Shape 5 gives the same picture with a larger residual — 5177 cycles/trip
against the same 2048-cycle floor, **60.4% non-MFMA** — but its last tile-wave
is only 88% full, so part of that is idle CTAs rather than an idle pipe. Shape 6
is the clean row and every number below is quoted from it.

**Nothing is saturated.** The largest non-MFMA resource is LDS at 50% duty
(25% if the rate is 256 B/clk). The gap of **2015 cycles per trip** is
therefore serialization — pipes idle while a dependence resolves — and not a
bandwidth or issue wall. In µs: **492 µs of shape 6's 992.2 µs pool**, which is
the same quantity exp_20 §4.1 states device-wide as 571 µs.

### 3.1 What static analysis can and cannot account for

**Accounted for, INFERRED from the block layout of §2(d).** Two windows in
which this wave's matrix pipe is provably idle:

| window | contents (counted) | inferred cycles |
|---|---|---:|
| W1: `s_barrier` → first MFMA in `.LBB4_93` | 24 `ds_read`, 4 `global_load`, 58 address ops, 1 `lgkmcnt(0)` | 300–450 |
| W2: last MFMA in `.LBB4_90` → `s_barrier` | 2 `vmcnt(0)`, 8 `ds_write`, 2 `lgkmcnt(0)` | 150–300 |
| the barrier rendezvous itself | 8 waves converge | ≥50, skew-dominated |
| **identified total** | | **500–800** |

**Not accounted for: ~1200–1500 cycles per trip.** I will not dress this up.
The three candidates, in my order of belief:

1. **Barrier phase-locking.** One CTA-wide `s_barrier` every 64 MFMAs forces
   all 8 waves — including the 2 sharing each SIMD — to stall *in phase*. The
   only latency-hiding mechanism available at 2 waves/SIMD is one wave covering
   the other's stall, and a barrier every trip destroys exactly that. The
   residual is roughly one LDS round trip plus one global round trip per trip,
   which is the signature you would expect if neither is being covered.
2. **Effective LDS rate under 8-wave contention.** The XOR swizzle is
   conflict-free — I checked it: for `st_bf<256,32>`, `swizzle_bytes = 64`,
   `subtile_cols = 32`, so lane *l*'s byte offset is
   `64·(l%16) + 8·(l/16)` XOR `8·((l%16)/2)`, which spreads 16 lanes across all
   32 banks with no collision. So *bank* conflicts should be zero, but the
   sustained rate under 8 concurrent waves is not something the ISA can tell me.
3. **Uncovered global latency at `vmcnt(0)`.** The `global_load` for k+1 is
   issued at the top of the trip and drained after 64 MFMAs, i.e. ~2048 SIMD
   cycles of cover. Per rank the mainloop requests 3.89 GB over the 992.2 µs
   pool = **3.9 TB/s of L2 read**; that is only ~13% of the aggregate L2 ceiling
   and ~740 GB/s at DRAM after the ~5× L2 tile reuse the column-major `WGM`
   order produces, so it is not a bandwidth wall — but the *tail* of a 4-load
   `vmcnt(0)` under that load can still exceed the cover.

**This is the single most valuable next measurement and it needs no kernel
change**: one `rocprofv3` pass at the current best config on shapes 5 and 6
with `SQ_WAIT_INST_LDS`, `SQ_WAIT_ANY`, `SQ_BUSY_CYCLES`, `SQ_INSTS_VALU`,
`SQ_LDS_IDX_ACTIVE` and `SQ_LDS_BANK_CONFLICT` partitions the 2015 cycles into
LDS-wait / other-wait / VALU-busy and settles which of the three above owns it.
The plumbing already exists in `exp_20_attribution/run_counters.sh`. ~20 minutes
of node time, no correctness risk, no code.

## 4. The closure: `BK` cannot buy a barrier, at any stage count

This is the arithmetic that kills directions 1 and 2 of the dispatch, and it is
short enough to check by eye.

Let `S` be the number of LDS stages, `BK` the k-step. The `static_assert` is
`S·(BM+BN)·BK·2 ≤ 65536`, i.e. **`S·(BM+BN)·BK ≤ 32768`**.

Barriers required per k-step:

- `S ≥ 2` — **one**. The write of step *k* targets the buffer step *k−1* read;
  the barrier ending *k−1* separates them. The second buffer removes only the
  anti-dependence, so one barrier suffices.
- `S = 1` — **two**. One after all reads of the single buffer complete (before
  the writes may clobber it) and one after the writes (before the next reads).

MFMA work per k-step is `BM·BN·BK` MACs. So **MFMA work per barrier** is
`BM·BN·BK` at `S ≥ 2` and `BM·BN·BK/2` at `S = 1`. Under the LDS cap:

| arm | LDS bytes | barriers per tile (shape 6, `K_local=3696`) | MFMA per barrier |
|---|---:|---:|---:|
| **`S=2, BK=32` (today)** | 65,536 | **116** | `256·256·32` |
| `S=1, BK=64` (direction 1) | 65,536 | 2 × 58 = **116** | `256·256·64/2` = **identical** |
| `S=2, BK=16` (direction 2, rolling half) | 32,768 | **231** | half — strictly worse |
| `S=4, BK=16` | 65,536 | **231** | half — strictly worse |
| `S=3, BK=21` | not legal (`KS % 16 == 0` requires `BK % 32 == 0`) | | |

And the tile itself is already the argmax. Maximise `BM·BN·BK` subject to
`(BM+BN)·BK ≤ 16384` (the `S=2` cap): with `BM=BN=b`, `BK ≤ 8192/b` and the
objective becomes `b²·(8192/b) = 8192b`, **strictly increasing in `b`**. The
binding limit on `b` is not LDS but the accumulator: `rt_fl<BM/2,BN/4>` costs
`BM·BN/512` VGPRs, and at `b = 256` that is 128 of a 256-register unified file
already 248 full. `b = 256, BK = 32` is the maximum. Independently, exp_14
reached the same table by exhaustive enumeration of a different objective.

**So: the barrier count per tile is `K_local/32` for every legal
(stage, `BK`, tile) triple, and freeing LDS buys nothing** — which is doubly
true here because the grid is exactly `CU_COUNT = 304` CTAs on 304 CUs, so LDS
can never buy a second resident CTA either. LDS is a cap with nothing behind
it. **The lever is the cost *per* barrier, not the number of them.**

Corollary worth recording, since it inverts the intuition the tree has been
carrying: today's "double-buffered mainloop" buys **exactly one barrier per
k-step** and no pipelining depth at all. The write of step *k* feeds the read of
step *k+1* with a barrier between them either way; the only real prefetch in
this kernel is exp_03's **register** staging (`abuf`/`bbuf`), one step ahead.

## 5. Ranked candidates

Cycle deltas are per trip against the 4063-cycle shape-6 baseline; µs are that
delta applied to the 992.2 µs / 305.2 µs GEMM pools. **All µs figures are
INFERRED** — they are the cycle model of §3 applied to counted instruction
deltas, and the model's own residual (§3.1) is larger than most of the entries,
which is precisely why row A comes first.

| # | mechanism | what it changes | Δcyc/trip | **shape 6 µs** | **shape 5 µs** | correctness risk / catching gate | cheapest decisive experiment |
|---|---|---|---:|---:|---:|---|---|
| **A** | **Counter pass: split the 2015-cycle residual** | nothing | 0 | 0 | 0 | **none** — no code | one `rocprofv3` pass, shapes 5+6, `SQ_WAIT_*`/`SQ_LDS_*`. 20 min. **Prerequisite for B–H being priced at better than 3×** |
| **B** | **Hoist `load_commit` above half 1's MFMAs** | moves `vmcnt(0)` + 8 `ds_write` + `lgkmcnt(0)` off the pre-barrier tail; 32 MFMAs then cover them. Window W2 → ~0 | −150…−300 | **−37…−73** | −11…−23 | **LOW.** Buffer `As[(k+1)&1]` was last read in k−1 and the barrier ending k−1 separates it — the invariant already documented at `gemm_rs_mi300x.cpp:548-554`. Caught by `lds_race_check.sh` + M3 @2e-3 + M9 | move the two `load_commit` calls into the `kh` loop after `kh==0`. **One-line edit**; ISA check that `vmcnt(0)` now sits between the two 32-MFMA runs; paired M7 |
| **C** | **Prefill swizzled LDS offsets + hold LDS bases in SGPRs** (port `prefill_swizzled_offsets` + `readfirstlane` bases, `gemm_rs_device_tile.cpp:833-852`) | deletes most of `%bb.92`'s 58 address ops. The swizzle is **c-independent for `BK=32`** (2c ≤ 62 < the 128-byte bucket) and the r-difference between base tiles is a constant 1024 B, so all 24 reads are one base VGPR + compile-time immediates | −100…−230 | **−25…−60** | −8…−18 | **LOW–MED.** An address error is loud, not silent: M3 fails by orders of magnitude. No wait or barrier moves, so the silent-race class is untouched; `lds_race_check.sh` still required | port the donor's two helpers into `gemm_rs_mi300x_hk_adapter.cuh` (we own it); assert `%bb.92` shrinks in the census; M2 must show VGPR ≤ 248 |
| **D** | **Retire `K_TAIL=true` on shape 6** — zero the LDS padding columns in `load_commit` on the last k-step instead of masking the register fragment in the `kh` loop | shape 6's loop goes from 257 instr / 4 blocks to ~194 instr / 1–2 blocks, restoring one scheduling region | −250…−400 | **−40…−90** | 0 (shape 5 is already `false`) | **MED.** Changes what reaches the MFMA. This is the `<64,64,128>` class: a wrong tail is *numerically small* and can pass 1e-2. **M3 at 2e-3 is the gate that matters, plus the existing tail negative control** | prototype in the adapter; M3 all 17 shapes both tolerances with special attention to rows 3 and 6 (the only `K_TAIL` rows) |
| **E** | **Phase-offset warp-group ping-pong** — the donor's `if (warp_row == 1) s_barrier();` prologue (`gemm_rs_device_tile.cpp:859-861`) offsets half the waves by one barrier phase, so group A's MFMAs cover group B's LDS/global work. gfx942 has no named barriers (`s_barrier_signal` is gfx12), and this is the standard substitute | attacks the phase-lock of §3.1(1) directly — the only mechanism that does | −600…−1400 **if** §3.1(1) is the residual; ~0 if it is not | **−150…−350** | −45…−105 | **HIGH.** Memory-ordering change; the two groups index different LDS buffers at the same instant, so the `k&1` invariant must be re-derived. **protocol-review mandatory before first GPU run**, then `lds_race_check.sh`, M3 both tolerances, M4, M5, M9 | not cheap. Design it while B is being gated; do not build it before row A reports |
| **F** | **Wave-private LDS staging — barrier-free mainloop** | each wave stages only what it reads, so *no* barrier is needed at all. Costs 4× A and 2× B global traffic: 96 KB/CTA/trip vs 32 KB, i.e. 11.8 TB/s of L2 read (under the ~31 TB/s ceiling, but 3× the fabric pressure while egress is live) | −800…−1600 | −250…−400 | ? | **HIGH**, and it may simply relocate the bottleneck into the L2 that egress is already using | rewrite of the data path. Highest ceiling on the list, worst effort ratio |
| **G** | **`ds_read_b128` via a custom 16×32 shared layout** (donor's `st_16x32_s`, `gemm_rs_device_tile.cpp:549`) | 24 `ds_read` → 12. Same bytes, so **no bandwidth win** — halves issue slots and latency events only | −60…−150 | −20…−50 | −6…−15 | **MED–HIGH.** New shared layout + matching global→LDS store pattern; fragment layout changes, which is the `<64,64,128>` silent class again | large. Do not start before A |
| **H** | **`v_mfma_f32_32x32x8_bf16_1k`** (0 in the TU today) | 64 MFMA → 32, at 32 cyc each. **Identical peak** (both are 256 MAC/cyc/SIMD), identical accumulator VGPRs (128), identical `ds_read` count and LDS bytes. Buys instruction-issue slack and a longer shadow per instruction, nothing else | −50…−150 | −20…−60 | −6…−18 | **MED.** Completely different fragment register layout → silent-wrong-answer class | needs new fragment types in the adapter. Low value per unit of work; the atom is *not* a throughput lever on gfx942 |
| — | **`S=1, BK=64`; rolling half-`BK`** | — | **0** | **0** | **0** | — | **CLOSED on arithmetic, §4. Do not build.** |

## 6. Recommendation

**Run row A (the counter pass) first — it is free — and then land row B as the
first code arm.**

Row B is the recommendation because it is the cheapest decisive probe of the
entire "shrink the barrier-adjacent critical path" family (B, C, D). All three
are priced off the same inferred quantity — that windows W1 and W2 are worth
500–800 cycles/trip — and B tests that premise with a one-line edit.

**Pre-registered effect size.** Shape 6's GEMM ablation pool falls
**5.0% ± 3.0%** (992.2 → 942 µs), shape 6's wall falls ~3.1% (1632.5 →
~1582 µs), shape 5's wall ~2.3%, geomean **−0.9%** (209.5 → ~207.6 µs). Shapes
1–4 move by less than their allocation-noise floors and must not be credited.

**Falsifier.** If a same-run paired A/B over ≥3 draws (with `SWEEP_REVERSE=1`
on half of them, per exp_14) shows **|Δ| < 1.5% on shape 6** while the ISA
census confirms `vmcnt(0)` has moved between the two 32-MFMA runs, then window
W2 is worth under 100 cycles/trip. In that case **the whole B/C/D family is
worth less than 3% and must be closed**, and the night should go to row E,
which is the only mechanism whose ceiling is large enough to matter.

The converse pre-registration matters too, because it is what makes B
informative rather than merely cheap: if B lands at or above the top of its
band (−7% or better), windows W1/W2 are the residual, row C becomes worth
building immediately, and row E's ceiling should be revised *down* — the
phase-lock would then not be the dominant term.

## 7. Gate plan for row B, and whether it is sufficient

Full ladder, in order, with the reason each stage is present rather than
inherited:

| stage | why, for *this* mechanism |
|---|---|
| M1 build | — |
| **M2 + `kloop_hist.py`** | the edit's entire content is *where* `vmcnt(0)` sits. A build that silently sank the commit back below all 64 MFMAs would be a null arm masquerading as a negative. **Assert from the ISA that the loop contains 32 MFMA, then `vmcnt(0)`, then 32 MFMA** — and assert VGPR ≤ 248 and 0 spills, because sinking a `float4` staging buffer whose `global_load` has not landed is not slow, it is *wrong*, and no `vmcnt` protects it (exp_03) |
| **`exp_03_mainloop/lds_race_check.sh`** | 30 s, and it is the only mechanical check for the exact class this edit could introduce: an `asm`-issued `ds_read`/`global_load` destination register read before its drain. Read its ~10 post-rotation false positives; do not relax it |
| M3, 17 shapes, **both `1e-2` and `2e-3`** | — |
| M4 three negative controls | — |
| M5 600-epoch soak | — |
| M9 `m9_stale_slot` | B changes nothing about publication order, so M9 should be *unchanged*. Run it as a null: a change here means the edit did something it was not supposed to |
| M7, 3 rotations × 50, paired, ≥3 draws, half reversed | sub-2% deltas re-run; report best and median, never the mean (per-allocation bias, `HANDOFF.md` §2) |

**Is that sufficient against the `<64,64,128>`-class silent wrong answer? For
row B, yes. For rows D, G and H, no.**

The `<64,64,128>` failure and the `acquire_frags` failure are two different
classes and only one of them is a race:

- **The race class** (bare `lgkmcnt(0)`, MFMAs hoisted above the drain):
  register-level, detectable statically. `lds_race_check.sh` catches it in 30 s,
  and M3 catches it loudly (`max|diff|` up to 9e30). Row B lives entirely in
  this class, and it is covered.
- **The layout class** (`<64,64,128>`'s `rt_bf<32,64>` A fragment): the kernel
  runs to completion, sets no error bit, and produces answers that are *wrong
  in a numerically plausible way*. `lds_race_check.sh` is blind to it, and
  `1e-2` can be blind to it. Only M3 at `2e-3` across all 17 shapes is a real
  gate, and only if the failing tile actually appears in those 17.

**Rows D, G and H are in the layout class and need one gate that does not yet
exist**: a **fragment-layout fingerprint**. Before the mainloop change, dump one
tile's `A_frag`/`B_frag` register contents for a known input to a side buffer
and freeze it; after the change, require a bitwise match. That is a
constructive check on the thing `2e-3` only samples, it costs one debug build,
and it is the difference between "17 shapes happened to cover it" and "the
layout is provably unchanged". M9 is the precedent for this style of gate —
NaN-poison plus a bitwise golden — and it should be extended, not reinvented.
**Row B does not need it. Do not start D, G or H without it.**

## 8. Expected value, computed honestly

Baseline per-shape `full` (exp_20 `ablation.json`): 67.1 / 67.5 / 87.7 / 203.5 /
641.9 / 1632.5 µs → **geomean 209.5 µs**.

| scenario | shape 5 | shape 6 | geomean | Δ geomean |
|---|---:|---:|---:|---:|
| today | 641.9 | 1632.5 | 209.56 | — |
| **row B lands at its pre-registered −5% of the pool** | 626.9 | 1582.5 | 207.66 | **−0.91%** |
| realistic full capture (see below) | 551.9 | 1382.5 | 198.77 | **−5.15%** |
| the entire non-MFMA mainloop, i.e. the roofline | 437.9 | 1061.5 | 183.01 | −12.67% |

**Why "realistic" is 40% of the roofline and not 100%.** Over its GEMM pool
shape 6 sustains 496.1 GFLOP / 992.2 µs = **500 TFLOPS**. The 256 producer CUs
at the pinned 1900 MHz are worth 996 TFLOPS, so we run at **50.2% of producer-CU
peak**. Well-tuned MI300X bf16 library GEMMs land at 60–75% of peak, not 100% —
the 571 µs is the distance to a roofline nobody reaches. Closing to ~70% is
worth ~250 µs on shape 6 and ~90 µs on shape 5, which is the "realistic" row.
This calibration is the single most important number for planning the rest of
the night: **`HANDOFF.md`'s "2.5–3× off single-CU peak" overstates the
opportunity by roughly 2×**, because its denominator is the roofline rather
than an achievable GEMM.

**Shapes 1–4 contribute nothing and the geomean is why the wins look small.**
Shape 1 is `bound = HOST` at 10.39× SOL (62.34 µs of host issue in a 66.41 µs
wall) and shape 2 is host-bound in truth as well; their mainloops are single-
digit µs. A geometric mean over six shapes gives a −15% win on shape 6 a weight
of 1/6 in log space, so **−15% and −12% on the two shapes that matter is −5.1%
overall**. That is arithmetic, not pessimism, and it should be said out loud
before anyone spends a night on a 15% mainloop win expecting a 15% score.

**Effect on the gap to rank-1.** Per-shape graded ratios are 0.850 / 1.072 /
1.102 / 1.120 / 1.286 / 1.208, geomean 1.098×. The realistic capture takes
shape 6's graded ratio to ~1.033 and shape 5's to ~1.128 (the graded wall is
~90 µs larger than the pipelined one on each call, so the same absolute saving
is a smaller *fraction* there), giving a graded geomean of **~1.046×**. Row B
alone gives ~1.092×.

**And the graded number flatters us.** exp_24 measured a ~90 µs constant that
both arms pay per call — the evaluator's trailing `synchronize + barrier`
(57–79 µs) plus input clone (15–20 µs). Adding a constant to both sides of a
ratio compresses it toward 1. Netting it out of both arms puts the honest
kernel-to-kernel ratio near **1.14×**, not 1.098×. The pipelined comparison is
the one to steer by; the graded one is the one that is scored. Report both,
never blend, and never let the compressed number make a mainloop win look
larger than it is.

**Bottom line on the dispatch's closing question.** The ~571 µs is **partly
recoverable, and by much less than its size suggests.** ~250 µs on shape 6 is a
defensible target; the rest is the difference between a real GEMM and a
roofline. Getting even that requires attacking the barrier's *cost* — the
phase-lock of 8 waves at 2 waves/SIMD (row E) — because the barrier's *count*
is provably fixed by the LDS and accumulator caps (§4). What would have to
change for a bigger prize is the CTA shape itself: 512 threads on a 304-CTA
persistent grid pins 2 waves/SIMD and 128 accumulator VGPRs, and every lever in
this document is working inside that. That is a protocol-level change, not a
mainloop change, and it is not a tonight decision.

## 9. Schema of `kloop_census_summary.json`

Top level: `experiment`, `provenance{sha256, hipcc, utc}`, `method`,
`instantiations[]`, `cycle_budget[]`, `closure_table[]`, `candidates[]`,
`expected_value[]`.

- `instantiations[]`: `name`, `bm`/`bn`/`bk`/`k_tail`, `vgpr`, `agpr`, `sgpr`,
  `scratch_bytes`, `vgpr_spill`, `waves_per_simd`, and
  `k_loop{blocks, instr_total, per_class{...}}` — **all COUNTED**.
- `cycle_budget[]`: one record per shape (5, 6) with `trips`,
  `gemm_pool_us` (MEASURED), `cycles_per_trip` (DERIVED),
  `mfma_cycles_per_trip` (MEASURED rate × counted instrs), `residual_cycles`,
  and `pipes[]` each carrying `cycles`, `basis` (`counted`/`measured`) and
  `rate_assumed`.
- `closure_table[]`: the §4 arms with `lds_bytes`, `barriers_per_tile`,
  `mfma_per_barrier`, `verdict`.
- `candidates[]`: the §5 rows with `delta_cycles_per_trip_lo/hi`,
  `shape6_us_lo/hi`, `shape5_us_lo/hi`, `risk`, `gate`, `experiment`,
  and `basis: "inferred"`.
- `expected_value[]`: the §8 scenarios with per-shape µs and geomean.

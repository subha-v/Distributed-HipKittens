# exp_27 result — M6 reads the token-major `sc_stage`; M5's transpose is deleted

**VERDICT: CONFIRMED. THE RATCHET MOVES.**

> **6,482.7 µs = 0.8408× production**, from 6,568.0 µs / 0.8522×.
> **−85.3 µs, +1.14 points**, across two candidate campaigns paired against a
> same-session control campaign that reproduced the old ratchet to 0.16 %.
> Commit `f113d73f`. Remaining to 0.80×: **−317 µs.**

Deleting the group-major activation-scale layout takes **ΔM6 = −79.7 µs
(t = −14.8)** off the M6 device stamp — inside exp_28's pre-registered confirm
band `[−190, −60]` — and that saving passes through to the campaign essentially
1:1 (paired campaign delta −75.0 µs). Every correctness gate is green, the
numerics are unchanged at full float precision, the resource tuple holds exactly,
and the ISA shows the gather's loop trip count dropping 21 → 6.

The whole M5 scale-transpose loop **is gone** on the shipping arm, and
`hk_moe::mps::scale_transpose_row` is compiled out entirely.

| | control (arm 0) | candidate (arm 1) | Δ | t |
|---|---:|---:|---:|---:|
| **M6 stamp, µs** | **2,534.12 ± 10.71** (n=10) | **2,454.39 ± 13.20** (n=10) | **−79.7** | **−14.84** |
| plan M3–M5, µs | 375.81 | 372.86 | −2.95 | −1.71 |
| M7, µs | 2,688.77 | 2,700.39 | +11.62 | +0.34 |
| combine, µs | 341.07 | 327.76 | −13.31 | −0.43 |
| M2→end, µs | 5,939.78 | 5,855.38 | −84.40 | −6.05 |
| end-to-end mps, µs (1-proc screen) | 6,740.74 | 6,689.61 | −51.13 | −0.72 |

Arms: `production,pf6gm_mega,mps_mega`; config
`C=16,g=353,mode=12,flush_rows=16,timestamps=1`; 1 proc / 1 warmup / 1 timed;
four batches of five runs, **control, candidate, control, candidate**.

---

## 1. The measurement design, and why it needed four batches

The arm is a compile-time literal in the hashed `.hip`, so control and candidate
**cannot** be interleaved inside one batch — one batch is one build is one arm.
That makes arm and time confoundable, and the confound is real: exp_26's control
read **2,581.7 µs** at 06:35Z, and my batch 1 — a binary whose `.text` is
byte-identical to the ratchet — read **2,531.6 µs** at 07:07Z. Fifty microseconds
of drift on the same code, ~6 within-batch σ.

So the design alternates and reports the drift as the yardstick:

| batch | arm | UTC | n | M6 mean ± sd |
|---|---:|---|---:|---:|
| `e27b1_ctl` | 0 | 07:05–07:08 | 5 | 2,531.56 ± 8.63 |
| `e27b2_cand` | 1 | 07:09–07:12 | 5 | 2,458.56 ± 9.08 |
| `e27b3_ctl` | 0 | 07:14–07:17 | 5 | 2,536.68 ± 12.93 |
| `e27b4_cand` | 1 | 07:21–07:24 | 5 | 2,450.22 ± 16.31 |

- **control → control drift: +5.1 µs (t = +0.74)**. The two controls, eight
  minutes apart on the same binary, agree. Within this session the stamp is
  stable, so the earlier 50 µs gap to exp_26 is a longer-timescale effect and not
  something that can manufacture this result.
- **candidate → candidate: −8.3 µs (t = −1.00)**. The candidate cell reproduces too.
- **|arm effect| / |control drift| = 15.6×.**
- Batch-as-the-unit (the conservative reading, 2 vs 2): **−79.7 µs, t = −16.29**.
  Runs-as-units gives the same point estimate, which is the reassuring outcome —
  the effect does not depend on the clustering choice.

The stamp is the right instrument and the end-to-end screen is not: the same runs
give **−51 µs at t = −0.72** end-to-end, i.e. invisible, exactly as the 6.6 %
screen tail predicts. Note `production` drifted **+48 µs** across the same
batches, which is most of the apparent end-to-end movement and none of it ours.

---

## 2. Correctness

All 20 runs `OK`. Uniform across every run and all 8 ranks:

- `[MOK GATE] pass=True`
- **`pperr = 0`**
- **`[MPS SOAK] completed=600/600 pperr=0 poison=0 pass=True`** — the full
  600-epoch soak ran in every one of the 20 runs, not once at the end
- **`[MARK] control_fails=True`** — the deliberately-broken negative control
  still fails
- **`[POISON SELFTEST] one_row_poisoned_fails=True nonfinite=57344`** for both
  `mps_mega` and `pf6gm_mega`, exactly the expected count, and
  `[POISON] survivors=0` on every eager and post-timing check. The gate was live
  for every number in this document.
- `nonfinite = 0` in all 160 (run × rank) cells.

---

## 3. The bit-identity question — what was asked, what is provable, what I measured

The brief asked me to capture the ratchet's `[MOK GATE]` values and require the
candidate to **reproduce them exactly**, and to verify that directly rather than
infer it. I did verify it directly, and the verification produced a result the
brief did not anticipate: **that test is not well-posed on this harness, because
the ratchet does not reproduce its own values.**

The evidence, all from this experiment:

1. **Batch 1 ran one binary five times and printed two different values.** Its
   `.text` is fingerprinted byte-identical to the ratchet (`ab0c353b…`), and its
   five `[MOK GATE] relative` readings were
   `0.008293 / 0.008294 / 0.008293 / 0.008293 / 0.008293`.
2. **`production`, an arm exp_27 does not touch, printed `max_abs` of both
   `0.031250` and `0.027344` inside batch 1**, and its full-precision
   `relative_error` has run-to-run **sd = 2.7 × 10⁻⁵** across the 20 runs.
3. **When candidate run 3 printed `max_abs = 0.037109` instead of `0.035156`,
   `pf6gm_mega` printed `0.037109` in the same run** — and `pf6gm_mega` is a
   different kernel that exp_27 does not touch. That value is a property of the
   run, not of my change.

The cause is in the kernel, not the harness: MoK's dispatch assigns receive rows
by `fetch_add` on a device counter, so arrival order — and therefore the expert
grouping and the top-k reduction order — differs run to run; and mode 12's
combine is a **nondeterministic bf16 remote-atomic accumulation** (282 static
`flat_atomic_pk_add_bf16`). `out` is not bit-reproducible in either arm.

### So here is the test that *is* sound, and its result

`pf6gm_mega` is untouched and is evaluated **on the same input in the same run**,
so it is a per-run fingerprint of that run's input and reference. The drift-free
quantity is the same-run paired difference

    d = mps_mega.relative_error − pf6gm_mega.relative_error

taken at full float precision from the rank JSONs (the `[MOK GATE]` line rounds to
6 digits, which throws away the signal), and compared **between arms**, clustered
by run because the 8 ranks in a run share one input.

| | n (runs) | mean `d` | sd |
|---|---:|---:|---:|
| control | 10 | −2.1172 × 10⁻⁸ | 2.45 × 10⁻⁸ |
| candidate | 10 | −2.9951 × 10⁻⁸ | 2.90 × 10⁻⁸ |

**Shift = −8.78 × 10⁻⁹, t = −0.73. Not significant.** It is **0.032 %** of the
run-to-run input drift on the untouched arm.

*A caution I have to record, because I nearly filed the opposite conclusion.*
Treating the 160 individual (run, rank) cells as independent gives this same shift
at **t = −5.01** — a false positive produced purely by ignoring that 8 ranks in a
run see one input. This is the clustering form of the trap the brief flags about
single-batch t-statistics, and it is worth adding to the standing list.

And two further checks:

- **`max_abs(mps_mega)` is exactly equal to `max_abs(pf6gm_mega)` in 160/160
  cells** — 20 runs × 8 ranks, across 13 distinct `max_abs` values in both arms.
  This is the statistic a layout error would blow up, and it never moves off the
  untouched arm's value.
- **The instrument's dynamic range**: the deliberately-broken negative control
  reads `relative_error = 0.8917`, **108× the arm's 0.0083**. A wrong scale
  layout mis-scales entire 128-channel groups per token; it lands in the control's
  regime, not 10⁻⁹ away. This failure mode cannot hide.

### And the part that is actually a proof

None of the above is why I believe the outputs of M6 are bit-identical. That is a
proof, given in full in `design.md` §2, and it is internal to the kernel:

- the only writer of `sc_dst` is M5's transpose, whose body is
  `sc_dst[lane·T_ext + t] = sc_stage[t·56 + lane]`, run over **all**
  `t ∈ [0, T_ext)` and all `lane < 56`;
- `T` in M6's address is `nvi[1]`, and `hkp_sort.hpp`'s `scan()` sets
  `nvi[1] = T_loc` with `T_ext` as the only caller's argument — **verified on the
  node before building**, because this was the one link exp_28's design asserted
  rather than checked;
- therefore `sc_dst[k·T_ext + token] ≡ sc_stage[token·56 + k]`, FP32, copied,
  never arithmetic;
- the `(token < T)` predicate, `tok_lds`, the LDS coverage (all 5,376 entries in
  both arms) and the `mfma_k` consumption order are unchanged, so no zero changes
  and no FP reassociation is introduced.

**The load returns the same bit pattern.** What is *not* bit-reproducible is
everything downstream of M7's nondeterministic atomics — and that was already not
bit-reproducible before I touched anything.

---

## 4. Where the win came from, and where the prediction was wrong

The mechanism landed and is visible in the ISA (`build.md` §G5): the gather's loop
limit goes `0x13ff = 5376 − 257` → `0x43f = 1344 − 257`, i.e. **trip count
21 → 6**; `flat_load_dword` → `flat_load_dwordx4`; the token stride folds into the
address as the immediate `0xe0 = 224`; and — better than designed — the four
`ascale_lds` stores merge into **two `ds_write2_b32`** with `offset1:96`, the row
stride.

| | arm 0 | arm 1 | ratio |
|---|---:|---:|---:|
| distinct 64 B lines per task | 5,376 | 384 | **14.0×** |
| bytes delivered per task | 344,064 | 24,576 | 14.0× |
| bytes wanted | 21,504 | 21,504 | — |

**Three predictions, two hits and one miss:**

- **M6: predicted −130 / −134 / −186 µs by three models, point estimate −140.
  Measured −79.7.** Confirmed in sign and band, but **43 % below the point
  estimate**. The three models all priced the deleted line traffic at a marginal
  byte rate; the measurement says either the rate is lower than the 146 µs/GB
  `analysis.md` §2 used, or part of the 907 MB was already being absorbed. This
  is a real over-prediction and should be applied as a haircut to the next
  member of this family rather than quietly forgotten.
- **M5/plan: predicted ≈ −17 µs. Measured −2.95 µs (t = −1.71), not
  significant — over-predicted ~6×.** The arithmetic for why: the transpose
  writes `sc_dst[lane·T_ext + t]`, and consecutive `t` are **adjacent**, so the
  scatter coalesces in L2 across the `t` loop and the real traffic is just the
  7.34 MiB of payload, ≈ 2 µs at HBM write bandwidth. It was never 16×-amplified
  on the write side the way the *read* side was. **The transpose was cheap to
  perform and expensive to consume** — which is exactly the asymmetry that makes
  the deletion worth doing at the consumer, and a caution against pricing a
  layout conversion by its store count.
- **M2→end: −84.40 µs (t = −6.05)** ≈ M6 (−79.7) + plan (−3.0). The budget
  closes, so the win is where the mechanism says it is and not a displacement.

M7 (+11.6, t = 0.34) and combine (−13.3, t = −0.43) are both null, as they should
be — nothing in this change touches them.

---

## 5. Campaigns — THE RATCHET MOVES to 6,482.7 µs / 0.8408×

Three 5-rotation campaigns, 500 warmup / 100 timed, all three arms,
`C=16,g=353,mode=12,flush_rows=16`, arms alternated and each one `.text`-
fingerprinted after the fact. A 1.14 % delta must be paired, not asserted, so a
**control campaign was run in this same session** rather than leaning on the
published ratchet number.

| campaign | arm | production | pf6gm_mega | **mps_mega** | **mps/prod** | mps/pf6gm | pf6gm/prod |
|---|---:|---:|---:|---:|---:|---:|---:|
| `e27camp_cand` | 1 | 7,710.7 | 6,891.0 | **6,477.0** | **0.8400** | 0.9399 | 0.8937 |
| `e27camp2_ctl` | **0** | 7,703.2 | 6,884.4 | **6,557.6** | **0.8513** | 0.9525 | 0.8937 |
| `e27camp3_cand` | 1 | 7,710.1 | 6,928.5 | **6,488.3** | **0.8415** | 0.9365 | 0.8986 |

Per-rotation `mps_mega` p50s:

- candidate 1: 6,494.7 / 6,474.9 / 6,477.0 / 6,468.7 / 6,480.1
- control:     6,555.3 / 6,567.6 / 6,554.1 / 6,557.6 / 6,576.5
- candidate 2: 6,488.3 / 6,483.7 / 6,492.8 / 6,497.2 / 6,476.8

**Results:**

- **The control campaign reproduces the published ratchet**: 6,557.6 vs 6,568.0 µs,
  a 10.4 µs / **0.16 %** difference, and `0.8513` vs `0.8522`. The denominator is
  the denominator, so the comparison is sound.
- **Candidate mean of two campaigns = 6,482.7 µs, spread 11.3 µs.** Both campaigns
  land below every control rotation; the two distributions do not overlap at all.
- **Same-session paired delta = −75.0 µs (−1.14 %).** Versus the published
  ratchet, −85.3 µs.
- `pf6gm/production` reads 0.8937 / 0.8937 / 0.8986, i.e. ≈ 0.896 in all three —
  the G=3 reference was built, so these are the right arms.
- Every campaign: `[MOK GATE] pass=True`, `pperr=0`, `control_fails=True`,
  `[MPS SOAK] 600/600 poison=0`, poison self-test firing at `nonfinite=57344`.

### NEW RATCHET

> **`C=16, g=353, mode=12, flush_rows=16` with `K0P6_MPS_ASCALE_TM=1`
> = 6,482.7 µs = 0.8408× production** (0.938× `pf6gm_mega`), commit `f113d73f`.
>
> Was 6,568.0 µs / 0.8522×. **−85.3 µs and +1.14 points of margin.**
> Remaining to the 0.80× target (≈ 6,166 µs at this production): **−317 µs.**

### The stamp predicted the campaign

I pre-registered "≈ 6,488 µs, ratio ≈ 0.842×" from the stamp before the campaign
ran. Measured: 6,477.0 and 6,488.3, ratio 0.8400 and 0.8415. And the paired
campaign delta (**−75.0 µs**) sits within 4.7 µs of the M6 stamp delta
(**−79.7 µs**).

That is the most useful methodological result here: **the M6 saving passes through
to end-to-end essentially 1:1**, so M6 is fully on the critical path, and the
device stamp is a calibrated predictor of campaign µs for this phase rather than
merely a directional indicator. It also retroactively justifies the instrument
choice — the same 20 runs showed −51 µs at t = −0.72 end-to-end, i.e. the screen
could not see a real effect that the stamp resolved at t = −14.8 and the campaign
then confirmed.

---

## 6. Node discipline

- One GPU job of ours at a time throughout. `rocm-smi --showpids` showed only
  `gpuagent` (pid 44579) before each of the five launches; the launcher refuses to
  start otherwise and `screen.sh` re-checks before every run.
- `setsid` + `timeout` on every run.
- **No foreign GPU process appeared at any point, so nothing was preempted and
  there is nothing to log under the lease rule.**

---

## 7. Primitives

**Nothing was added to the library, and nothing was open-coded around it.** exp_27
adds no protocol: no release, no acquire, no counter, no epoch, no slot lifetime.
The ordering it relies on is the existing `hkp::grid_barrier` chain, and it
*removes* a data dependency rather than adding one. Worth recording as the shape
of a clean experiment — the primitive library is untouched because there is no
protocol in the change. It also moved M6 onto a pointer with **strictly more**
ordering than the one it replaced (slot 9 is published by the M3, M4 and both M5
barriers; slot 20 only by the M5 barrier), so the change cannot weaken an edge.

**The finding, and it closes a lesson exp_24 opened.**
`hkp::zero_part_scale_transpose<Chunks1K>` fuses two unrelated jobs — a `part`-row
zero and a scale transpose. exp_24 had to split it to delete the first half, and
recorded "in hindsight the primitive should have been two composable pieces".
exp_27 supplies the other half of that lesson: **the transpose piece should never
have existed at all.** It converted a token-major array into a group-major array
for a single consumer that then read it with 16× line amplification. The durable
rule:

> A layout-conversion primitive is a defect until someone has named the consumer
> that requires the target layout **and shown the line-traffic arithmetic for both
> layouts.** Cheap to produce is not the same as cheap to consume — here the
> conversion cost ≈ 3 µs and the consumption cost ≈ 80 µs, and only the second
> was ever visible.

`scale_transpose_row` is left in place (the `#else` arm and the M5 fallback branch
still call it) and should carry that caveat in its comment.

**Negative finding, `primitives:` tag.** The gather at
`n2_phase1_gm_mps.cpp:374-400` is a strided-gather-into-LDS, a shape that recurs
(`ascale_lds`, `b1s_lds`, `dq2_lds`, `tok_lds`). There is **no primitive for it**,
so every site open-codes its own index decomposition and gets its bank behaviour
and its line amplification right or wrong on its own — this one had it wrong for
the entire life of the kernel. A
`group::gather_rows_to_lds<Rows, Cols>(dst_lds, src, row_index_lds, stride)` that
owns the "row-fast for LDS banks vs field-fast for coalescing" trade would have
made exp_27 a one-line call-site change instead of a hand-verified loop rewrite,
and would have made the 16× amplification visible at the call site. Proposed, not
built: it needs a second and third caller before the shape is trustworthy.

**A tooling finding worth keeping.** The arm-identification method that worked is
the **`.text` hash of the resolved JIT hsaco plus an arm-discriminating
immediate** (`0x13ff` vs `0x43f`, the gather's loop limit). Batches 2 and 4 landed
in *different* JIT directories with *different* file hashes and **identical
`.text`** — so a file-hash comparison would have said the arm changed between two
batches of the same arm. `../tools/e27_fp.sh` is the reusable form.

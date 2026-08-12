# exp_25 — CTA role split at the GEMM1/GEMM2 boundary

Design + cost model + build plan. **No source was edited to write this.** The
adversarial companion is `protocol_review.md` in this directory; it is not
optional reading — three of its findings change what gets built.

Short file tags (same convention as `CONTEXT/mode12_protocol_map.md`):

| tag | path |
|---|---|
| `KRN` | `distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip` |
| `ADP` | `distributed-kernels/fused_moe/moe_mps_adapter.cuh` |
| `P1` | the read-only M6 body, `~/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip/n2_phase1_gm.cpp` (585 lines, sha256 `1d90b26658b6a524db69434ddcfa0b2dcec2418c852f83874468d55dd0197dc2`) |
| `P2` | `distributed-kernels/fused_moe/n2_phase2_gm_mps.cpp` |
| `ABI` | `distributed-kernels/fused_moe/moe_host_abi.hpp` |
| `HKA` | `distributed-kernels/fused_moe/moe_hk_adapter.cuh` |
| `SYNC`/`CMPL`/`ROLES`/`CTR`/`LIFE` | `include/cdna4/ops/group/distributed/{sync,completion,roles,counter,lifetime}.cuh` |

**Provenance note worth recording before anything else.** There are four copies
of `n2_phase1_gm.cpp` on the node. Three are byte-identical
(`solution/hip/`, `exp_59_large_m_tiles/source_snapshot/`,
`exp_63_large_m_rerun/source_snapshot/`, all sha256 `1d90b266…`, 585 lines).
The fourth, `exp_65_phase1_accumulator_pressure/source_snapshot/`, is
**different** (sha256 `52ecd9e7…`, 634 lines). Any vendored sibling must be
derived from `1d90b266…`; vendoring the exp_65 variant would silently change the
M6 arm.

---

## 0. Verdict up front  — REVISED (rev 2, interleave assignment)

> **Rev-2 note.** Rev 1 (below, §0.1) recommended the static prefix split. That
> recommendation is **superseded**: the interleave is strictly better than the
> static split on every term (it recovers the 173 µs work-conservation penalty,
> needs no new mode, no new atomics, and no new protocol). But the reprioritizing
> argument that motivated the promotion — "the 976 µs epilogue surcharge is
> irreducible fabric time, so overlap it with M6's fabric-idle window" — **does
> not survive its own arithmetic**, and the interleave inherits a much smaller
> prize than the assignment assumes. Both conclusions are in §0.2/§0.3. Rev 1 is
> kept verbatim per the supersede-don't-delete rule.

### 0.2 The fabric arithmetic — verified, and it refutes its own conclusion

The assignment's two numbers are **both arithmetically correct**:

| quantity | value | check |
|---|---:|---|
| remote lane-atomics per rank per epoch | 102,760,448 | `mode12_protocol_map.md` §2 |
| remote bytes | 411,041,792 B = **392.0 MiB** | ×4 B ✓ |
| 392 MiB / 1,684 µs (M7 mode-0 fit) | **244 GB/s** | = the assignment's "~247" ✓ |
| 392 MiB / **976 µs** (epilogue surcharge) | **421 GB/s** | = **78.3 % of 537.6** ✓ |
| wire time for 392 MiB at 78 % efficiency | **980 µs** ≈ the 976 µs surcharge | ✓ |

That last row is the coincidence the argument rests on, and it is real. **But
`421 GB/s` is not a fabric utilization.** 976 µs is the *sum over each CTA of its
own epilogue phases*, not a wall-clock window in which all 240 CTAs push
concurrently. Per M7 task a CTA spends 71.2 µs in the K-loop and 41.2 µs in the
epilogue, so a CTA is in an epilogue **36.7 %** of the time. The two readings are:

- **Staggered** (CTAs at independent phases): ~88 of 240 CTAs pushing at any
  instant → 392 MiB / 2,660 µs = **155 GB/s = 29 % of ceiling.** Fabric has 3.4×
  headroom; the limiter is per-wave outstanding depth × ACK latency.
- **Aligned** (CTAs in lockstep): all 240 pushing during a shared 976 µs of burst
  windows → **421 GB/s = 78 % of ceiling.** Fabric nearly saturated during bursts.

**The aligned reading is the correct one, and it is self-inflicted.** Every CTA
enters M7 within a few µs of every other (M6 is uniform; `plan+M6` is invariant to
0.5 %), and every M7 task costs the same, so the 240 CTAs march in lockstep
through `[K-loop, epilogue] × 23.67` and their epilogue bursts **coincide by
construction**. This also explains exp_21: unthrottled, 240 aligned CTAs drove the
fabric into congestion collapse, and depth-8 throttling recovered 500 µs by
pulling it back to ~78 % of ceiling.

**And that is what kills the premise.** If the epilogue bursts are already at
78 % of the aggregate egress ceiling, then de-aligning them or spreading them into
M6's idle window can recover **at most the gap between 78 % and 100 %**:

| assumed achievable efficiency | wire time for 392 MiB | max recoverable from the 976 µs |
|---|---:|---:|
| 100 % of 537.6 GB/s (unreachable) | 765 µs | **+211 µs** |
| 90 % | 850 µs | **+126 µs** |
| 78 % (what we already achieve) | 980 µs | **−4 µs, i.e. zero** |

**W1's hard ceiling is +211 µs, not +450 µs, and it requires 100 % fabric
efficiency on 4-byte remote RMWs.** The realistic figure is +100 to +210 µs. The
assignment's conclusion — "most of the 976 µs is irreducible fabric time" — is
**correct, and it is an argument *against* the mechanism, not for it**: irreducible
means there is nothing to recover. exp_21 already harvested the reducible part.

### 0.3 The interleave — a first-order tie, not a first-order win

Work conservation applies to the interleave **with equality**, which is better
than the static split but is not a win:

| arrangement | makespan | vs today |
|---|---:|---:|
| today (M6 on 256, M7 on 240, sequential) | `2588 + 2660` = **5,248 µs** | — |
| **interleave** (M6 on 256, M7 on 240, concurrent) | `A6/256 + A7/240` = **5,248 µs** | **0** |
| static split, work-conserving floor | `(A6+A7)/240` = **5,421 µs** | −173 µs |

A compute CTA still executes 2,588 µs of M6 work and 2,660 µs of M7 work, in
*some* order; reordering them does not reduce either. **Every microsecond the
interleave wins must come from a second-order term**, and the terms are:

| term | size | sign |
|---|---:|---|
| W1 — de-aligning the epilogue bursts (§0.2 ceiling) | **+100 to +211 µs** | + |
| W3 — one combined quantization tail instead of two | **+24 µs** | + |
| **L2 co-residency** — M7's 2.64 MB/XCD `W2` stream loses its L2 residency to M6's 10.6 MB/XCD stream (§3.0.5) | **−0 to −222 µs** | **−** |
| co-tenancy — the epilogue's neighbour becomes M6's *hungrier* stream (4.91 vs 4.30 TB/s) | **−0 to −100 µs** | − |
| new `a2_done` poll + in-order `vmcnt` coupling at each transition | **−20 to −50 µs** | − |
| the mandatory H1 drain fix (§3.0.4) | **−11 µs** | − |

**Central prediction: 6,713 µs (0.870×). Band 6,481 – 6,944 (0.840 – 0.900×).
The current ratchet, 6,685 µs, sits inside the band.** This is a **wash**, and
the falsifier (§5.5) is pre-registered accordingly.

### 0.4 What I recommend instead

Build it anyway, but **only because it is now cheap enough to be a screen rather
than an experiment** — three findings collapsed the cost:

1. **No new mode, and none of the seven predicates.** The interleave changes the
   *order* in which one CTA runs its own two task stripes. The protocol, the
   partition, and every buffer are bit-identical to mode 12. `k` goes in **free
   config bits [34:40)**, `k=0` ≡ ∞ ≡ today, bit-identical. This also sidesteps
   the exp_29 mode collision entirely (§6.1a).
2. **Zero delta against the vendored file.** exp_26's commit `f9bfb4be` already
   landed `N2GM_P1_TASK_START/_STRIDE` (MPS-DELTA (3), lines 215-227) *explicitly
   for this experiment*, and `N2GM_P1_EPILOGUE_DONE_HOOK` (MPS-DELTA (10), line
   637) is exactly where the H1 fix belongs.
3. **The static-stripe claim needs no claim mechanism at all** (§3.0.2): keep
   today's `start = blockIdx.x, stride = 240` M7 partition and merely run it
   early. Over-coverage — the silent 6 × 10⁻⁵ failure — becomes **impossible by
   construction**, retiring `protocol_review.md` §6.3 for this arm.

**Revised build cost: 4–6 h (was 14–19 h).** And the Stage-0 diagnostic asked for
turns out to be **an arithmetic identity, not a measurement** (§6.0): the number
of ready M7 tasks is *exactly* rate-matched to what the interleave needs, with
0.05 % slack, for any routing. That answer is free, and it is the most useful
thing in this document.

### 0.5 The four claims, adjudicated

| claim | verdict | why |
|---|---|---|
| **No work-conservation loss** | **CONFIRMED as stated; the implication REFUTED** | `A6/256 + A7/240 = 5,248 µs` = exactly today. The `(W6+W7)/240 = 5,421` penalty genuinely does not apply — the interleave is 173 µs better than the static split. But *no penalty ≠ a gain*: it is a **tie** at first order, and the framing treats removing the penalty as the win. It is not. Every µs must come from §0.3's second-order terms |
| **F1 becomes irrelevant** | **CONFIRMED — with a cost the claim does not price** | The interleave never reduces M6's CTA count, so `T6(C6)` drops out of the model entirely and the gate is genuinely removed. But F1's *favourable* branch was also the static split's upside (up to −535 µs). The interleave wins 0 first-order **regardless of F1**, so it forgoes that option value. **The two mechanisms are not ordered** (§3.5a): keep F1, demoted from gate to option-pricing |
| **Zero register risk** | **CONFIRMED, and better-founded than my rev-1 argument** | Accumulators are born and die inside each body → allocator takes `max(192,168)`, not the sum. The loop-carried *pointer* worry from §3.4 is answered in source: `k0p6_dread` (`KRN:204-209`) is a **volatile** load whose memory clobber the comment says exists precisely to "keep the compiler from CSE-hoisting descriptor values into kernel-long registers". So the pointers cannot be hoisted into the fused loop. Extra live state = ~5 scalars (two cursors, `k`/`j` counters). **New risk, unrelated to registers: I-cache.** Both MFMA bodies inlined into one hot loop against a 32 KB L1I shared by 2 CUs — unquantifiable from here, must be read out of the resource report |
| **Zero extra LDS** | **CONFIRMED** | Both phases' blocks already sum into the single 155,428 B allocation; interleaving changes no allocation |
| **Still on-mandate** | **COMPLIANT, but not an advance — I partly disagree** | The service pool survives, so a CTA role split exists and the arm is compliant. But `aug10/CLAUDE.md` requires that *"at any instant some CTAs are moving data while **different** CTAs are issuing MFMA"*. Under the interleave every compute CTA runs the **same** mixture, so the M6/M7 boundary is **not** a role partition — it is a schedule. The mandate is satisfied only by the pre-existing pool the ratchet already has. **The interleave adds nothing new on the mandate's research question; the static split does.** Worth knowing, since mission step 5 is what this line of work exists for |

### 0.6 Revised staged build

| stage | what | cost | gate |
|---|---|---:|---|
| **S0** | **nothing to build** — readiness is the identity in §6.0 | **0 h** | availability is not the limiter; proceed |
| **S1** | fused task loop in `KRN` + `k` in config bits [34:40) + the H1 `vmcnt(0)` hook. `k=0` must be **bit-identical** to today | 3-4 h | resource tuple unchanged; `k=0` arm within ±15 µs of 6,685 µs, else it is a build defect — stop |
| **S1d** | **DQ2-NaN-poison detector arm** (§3.0.4). Not optional | 1 h | must fire on a deliberately-removed drain (negative control), must be clean with it |
| **S2** | sweep `k ∈ {1,2,4,8,∞}`, one campaign per point | 2 h GPU | §5.5 |
| **S3** | only if S2 lands < 6,481 µs: price F1 to decide whether Policy 1/2 exists | 2 h | §3.5a |

**Total to a decision: 4-6 h build + 2 h GPU** (rev 1 was 14-19 h).

---

### 0.1 Verdict, rev 1 — SUPERSEDED (static prefix split)

> **Superseded by §0.2–0.4.** Demoted because the interleave dominates it on every
> term at a fraction of the build cost — but note §3.5a: the static split retains
> real option value the interleave gives up, so this analysis is not dead.

**The mechanism is worth building, but the framing in the assignment needs one
correction that changes the whole experiment: a static role split cannot win by
overlapping work, because work is conserved and M6 currently gets *more* CTAs
(256) than any split can give it.** In a pure work-conservation model with
linear scaling, the best possible balanced static split is a **tie**, and against
today's arrangement — where M6 runs on all 256 CTAs and M7 on 240 — it is a
**small loss**.

Everything the mechanism can win comes from three second-order terms, and only
one of them is large:

| | term | size | confidence |
|---|---|---:|---|
| **W1** | spreading the mode-12 epilogue's remote-RMW stream over a longer window with **2.1× fewer concurrent injectors** | **+200 to +450 µs** | the mechanism is measured (exp_21 throttle = −500 µs); the second step's size is `INFERRED` |
| **W2** | resource complementarity | **≈ 0, plausibly −300 µs** | the two phases are the same fp8 K-loop at the same intensity on the same limiter; the *fabric* is the one genuinely idle resource |
| **W3** | tail/ramp elimination | **+45 µs, minus a ~88 µs start bubble the split introduces** | arithmetic, `INFERRED` |
| **X** | M6's capacity loss (256 → ~128 CTAs) | **−20 to +535 µs** — sign unknown | **the single unmeasured coefficient that decides the experiment** |

Because X dominates and is unmeasured, **the first build stage is not the split.
It is the M6 CTA sweep** — which needs exactly the same vendored file the split
needs, costs one screen ladder, and is a genuine pre-registered kill gate.
`CONTEXT/m6_m7_structure.md` §0 and §7 item 2 already say this axis has never
been measured; this design makes it the decision gate rather than a footnote.

**Recommended partition policy: static prefix split with M6-pool fall-through
(policy 2), built in that order — static first (Stage 1), fall-through second
(Stage 2).** Reasons in §3.5.

**Central prediction: 6,312 µs (0.818× production), band 5,877 – 6,969.**
**Pre-registered falsifiers in §5.4.** The band is honest: it is ±650 µs and it
is dominated by X.

---

## 1. What the mechanism is

Today (mode 12, `C=16 g=33 flush_rows=16`) every one of the 256 CTAs runs the
whole of M6 (`KRN:1319-1330`; the task loop `for (task = blockIdx.x; task <
num_tasks; task += kCTAs)` at `P1:156` has **no** role predicate, no
descriptor-derived stride and no early exit), and then the 240 non-service CTAs
run the whole of M7 (`KRN:1367-1392`, guarded only by `is_service_cta`,
`ADP:292-297`).

The proposal partitions the compute grid three ways:

```
bid ∈ [0, C6)              M6 pool   — runs GEMM-1 only (then falls through)
bid ∈ [C6, 256 − C)        M7 pool   — runs GEMM-2 + the remote-accumulate epilogue
bid ∈ [256 − C, 256)       service   — the existing readiness-bookkeeping pool
```

so that at any instant some CTAs issue GEMM-1 MFMA while *different* CTAs issue
GEMM-2 MFMA and drive 392 MiB of xGMI RMW. It is on-mandate (`aug10/CLAUDE.md`
"every new kernel is a megakernel with CTA-level comm/compute overlap"), and it
attacks the two largest phases: M6 ≈ 2,588 µs + M7 ≈ 2,660 µs = **78 % of the
kernel** (`aug11/STATUS.md:24-35`).

**Nothing has to be built for the dependency.** `CONTEXT/m6_m7_structure.md` §4
established, and I re-verified against source, that:

- there is **no grid barrier** between M6 and M7 — the last one before the GEMMs
  is M5 (`KRN:1301`) and the next grid-wide rendezvous is M9's counted arrival
  (`KRN:1828`);
- the dependency is already **per-32-block**: `a2_done[b]` counts to 8
  (producer `k0p6_a2_arrive` `KRN:227-234` fired at `P1:481-488`; consumer
  `k0p6_a2_wait` `KRN:236-251` fired at `P2:336-344`, at the very top of the
  task before any LDS fill);
- `A2q`/`DQ2` are **capacity-sized and single-buffered with permanent per-row
  ownership** (`P1:19-21`), so there is no slot to recycle and no credit
  protocol to write;
- both phases' LDS blocks are **already summed** into the single 155,496 B
  allocation (74,240 + 49,664 + 30,500 + alignment), so the split costs **zero
  extra LDS**;
- M7's task loop is **already** start/stride-parameterised (`P2:286-294`,
  wired at `KRN:419-420`).

**What actually serialises the two phases is CTA program order** — a CTA drains
its entire M6 stripe before entering M7. That single fact is the whole
experiment.

---

## 2. Where the win can come from — and where it cannot

### 2.1 The first-order accounting, which the assignment's framing omits

Let `W6`, `W7` be the two phases' parallel work in CTA-µs. Sequential execution
on `N6`, `N7` CTAs takes `W6/N6 + W7/N7`. A split into pools `C6 + C7 = N`
takes `max(W6/C6, W7/C7) ≥ (W6+W7)/N`, with equality only at perfect balance.

So a perfectly balanced static split equals `(W6+W7)/240`. Today's arrangement
is `W6/256 + W7/240`, which is **smaller**, because M6 gets 256 CTAs and the
split can only give it ~128. Numerically, with `W6 = 2,588 × 256 = 662,528`
and `W7 = 2,660 × 240 = 638,400` CTA-µs:

```
today  = 662,528/256 + 638,400/240 = 2,588 + 2,660 = 5,248 µs
best split (balanced, linear) = 1,300,928/240      = 5,421 µs   ← WORSE by 173 µs
```

**A static split, on its own, is a loss.** Any claim of a win must name a term
outside this model. There are exactly three, below.

### 2.2 The floor that says the prize is real anyway

The pair's combined vector-memory request traffic is
`12.456 GB (M6) + 6.817 GB (M7) = 19.273 GB`
(`m6_m7_structure.md` §2.3, §3.3). M6 achieves **4.91 TB/s** of that traffic;
M7 achieves **4.30 TB/s**. If the combined stream sustained M6's rate:

```
19.273 GB / 4.91 TB/s = 3,925 µs
```

Against a measured M6+M7 of **4,647 µs** (derived from the epoch, not from the
max-stamps: `6,685 − 1,193 dispatch − 415 plan − 430 combine`), the headroom is
**722 µs**. On the max-stamp basis (5,248 µs) it is 1,323 µs. Use the 4,647
number; the max-stamps are per-CTA maxima and over-count the epoch by 601 µs
(9.0 %), which `harness_recipe.md` §3 warns about explicitly.

**Attribution of that 722 µs:**

| | term | µs | how |
|---|---|---:|---|
| 1 | M7's request-bandwidth deficit (4.30 vs 4.91 TB/s) | **197** | `6.817/4.30 − 6.817/4.91 = 1,585 − 1,388` |
| 2 | the mode-12 remote-accumulate surcharge that is *rate*-shaped | **≤ 450** | see §2.3 |
| 3 | ramp/quantization in both phases | **45** | M6 `0.09 × 229 µs`, M7 `0.33 × 71 µs` |
| 4 | unattributed residual | ~30 | |

### 2.3 W1 — the only large and defensible term

M7 in mode 12 costs **2,660 µs**. The same GEMM without the remote-accumulate
epilogue, at the same 240 CTAs, costs **1,684 µs** — from a two-point fit on
`exp_05/stage0_attribution.md:49-50` (254 CTAs → 1,609.7 µs; 192 → 2,019.7 µs),
which gives `T7(n) = 340 + 322,478/n`. **The surcharge is 976 µs.**

The fabric wire time for the epilogue is `392 MiB = 411.0 MB` of remote RMW at
537.6 GB/s aggregate egress = **764 µs at peak, ~1,019 µs at the 75 % efficiency
reported for MI300X links**. That is *concurrent* with the GEMM, so it is not the
serial cause of the 976 µs; the 976 µs is **congestion**, and exp_21 proved it is
**rate-shaped**: capping outstanding remote RMWs at 8 per thread
(`P2:150-159`, the `g` bit `0x20`) recovered **~500 µs**
(`aug10/STATUS.md:31-32`). Pre-throttle the surcharge was ~1,476 µs; one rate
step bought 34 % of it.

**The split is a second, orthogonal throttle on the same axis.** Total
outstanding remote RMWs at the fabric = (injecting threads) × (per-thread
depth). exp_21 capped the depth. The split cuts the thread count:

```
today  : 240 CTAs × 256 threads × depth 8 outstanding
split  : ~112 CTAs × 256 threads × depth 8            =  0.47×
aggregate injection rate: 411 MB / 4,540 µs vs 411 MB / 2,660 µs = 0.59×
```

That is the mechanism, stated in the same currency the measured throttle used.
Estimated recovery: **200–450 µs of the 976 µs.** `INFERRED` — the shape of the
congestion curve below the depth-8 point is not measured. §7's L6 screen is a
sharp differential test of exactly this claim.

**The genuine complementarity, named plainly:** M6 touches **no fabric, no peer
memory, no system fence, no `buffer_wbl2`** — it is entirely local
(`m6_m7_structure.md` §5.6, and `mode12_protocol_map.md` §3 confirms zero
system-scope traffic in M6). The xGMI links are 100 % idle for 2,588 µs of every
epoch. That, and only that, is the resource the split fills.

### 2.4 W2 — honest answer: approximately zero, possibly negative

The assignment asks for this to be quantified honestly. It does not survive:

| | M6 | M7 |
|---|---:|---:|
| tile arithmetic intensity | 161.7 FLOP/B | 158.1 FLOP/B |
| achieved B/clk/CU | 8.65 | 7.08 |
| MFMA duty cycle | 17.1 % | 13.7 % |
| limiter | L2/LLC request throughput | L2/LLC request throughput |

Two phases 2.2 % apart in arithmetic intensity, both at 14–17 % MFMA duty, both
limited by the **same** shared resource. Running them side by side does not fill
an idle pipe with complementary work; it puts two claimants on one queue. At the
balance point the combined MFMA duty is
`(128 × 17.1 % + 112 × 13.7 %) / 240 = 15.5 %` — still 84 % idle issue. **W2 = 0
on the compute/bandwidth axis.**

Worse, there is a concrete *negative* term. `P2:40-42` and the
`static_assert(kNChunks % 8 == 0, "nc -> XCD (nc mod 8) must be stable")` at
`P2:77` record a deliberate invariant: `nc → XCD (nc mod 8)` is stable, so M7's
`W2` weight stream is **L2-resident** — a 2.64 MB per-XCD working set inside a
4 MB L2 (`m6_m7_structure.md` §3.4). M6's per-XCD window is 10.6 MB and does
**not** fit. Under overlap, each XCD carries both: M6's window shrinks to
~6.6 MB (fewer tiles in flight at C6 = 128) but the sum is ~9.2 MB, and **M7
loses its L2 residency.** exp_08 measured that per-XCD L2 effects at this
boundary are worth 36 % of M7's interference — hundreds of µs. Budget
**0 to −300 µs**, unquantifiable from here.

### 2.5 W3 — real but small, and partly self-cancelling

Quantization tails: M6 `2,840/256 = 11.09` tasks/CTA → 0.09 of a 229 µs task =
21 µs; M7 `5,680/240 = 23.67` → 0.33 of a 71 µs task = 24 µs. **45 µs total.**

Against that the split *adds* a start bubble: the M7 pool cannot begin until the
first tile's 8 chunks land, ≈ one M6 task ≈ 205–229 µs, which is ~88 µs of grid
time. And a tail: the last ~16 tiles' 256 M7 tasks can only start after M6 ends
— 198 µs on C7 = 92 CTAs, **76 µs if the M6 pool falls through** to help. Net
W3 is **0 to −90 µs**, and fall-through is worth ~120 µs of it.

---

## 3. Partition policy — five designs compared

> **Rev 2:** Policy 0 (§3.0) is the recommendation. Policies 1-4 (§3.1-3.4) are
> **superseded** — kept for their measurements and for §3.5a's option-value
> argument, which is the one reason not to close them.

### 3.0 Policy 0 — the interleave (RECOMMENDED, Stage 1)

Every compute CTA runs `k` M6 tasks, then drains up to `j` ready M7 tasks, and
repeats. `k = ∞` is today's behaviour and is the control arm — it is **mode 12
itself**, so this mechanism is the only one in the design with a *free,
already-ratcheted* control.

#### 3.0.1 The ratio is fixed by structure, not tuned

M6 produces a tile in **8** chunk-tasks; M7 consumes a tile in **16** tasks.
Per CTA per epoch: `2,840/256 = 11.09` M6 tasks and `5,680/240 = 23.67` M7 tasks.

```
required consumption ratio  j/k = (16/8) x (256/240) = 2.133
```

M6 task ≈ 229 µs, M7 task ≈ 112 µs, so `k=1, j=2` gives 229 µs of M6 against
224 µs of M7 per cycle — **naturally balanced with no tuning**. `k` is therefore a
*granularity* knob (how coarsely the two streams are chopped), not a *balance*
knob. Sweep `k ∈ {1, 2, 4, 8, ∞}` with `j = 2k` implied.

#### 3.0.2 The claim mechanism: there is none, and that is the design

Three candidates were compared. **Option (c) wins decisively.**

| option | atomics / CTA / epoch | XCD invariant (`n2_phase2_gm_mps.cpp:77`) | over-coverage risk | verdict |
|---|---:|---|---|---|
| (a) global dynamic ticket, ready-filtered with skip | 1 RMW/claim + retries = **≥ 23.67** on 1 cell (5,680 RMWs on one line) | **broken** — `task` no longer ≡ `bid` mod 8, so `nc → XCD` residency is destroyed | needs a 710-B claimed-bitmap or a deferred list; monotonic tickets and *skipping* are structurally incompatible | **reject** |
| (b) per-XCD ticket bank, 8 banks | **23.67** RMWs on 8 padded cells (710/cell) | preserved (bank `x` hands out `task = x + 8m`) | a claim cannot be *returned*, so a claimed-but-not-ready task must spin — reintroduces the stall the mechanism exists to remove | reject |
| **(c) today's static stripe, executed early** | **0** | **preserved exactly** — the partition is byte-identical to today's | **impossible by construction** | **RECOMMENDED** |

Option (c): keep `N2GM_TASK_START = blockIdx.x`, `N2GM_TASK_STRIDE = 240`
(`n2_phase2_gm_mps.cpp:286-294`) **completely unchanged**, and simply let a CTA
execute the next task of its own stripe early, interleaved with its M6 tasks.

- **Exactly-once is inherited, not re-proved.** The task set per CTA is
  `{bid + 240i}`; the union over `bid ∈ [0,240)`, `i ≥ 0` is `[0, num_tasks)`
  exactly once — *the same partition mode 12 ships today*. Nothing can be lost or
  doubled because nothing about the partition changed. **This retires
  `protocol_review.md` §6.3 (the silent 6 × 10⁻⁵ doubled-tile failure) for this
  arm**, and with it the need for a `part_done` detector.
- **The `nc → XCD` invariant is preserved for free.** `stride = 240 ≡ 0 (mod 8)`,
  so every task a CTA ever runs has the same `nc mod 8`, exactly as today.
- **exp_07/exp_08's lesson is honoured trivially**: no atomic is relocated,
  coalesced, or added. Line spread is untouched.
- **No ABI change, no new `mps_state` cell, no new buffer.**

#### 3.0.3 Leftover M7 tasks: there is no handoff

Because the partition is unchanged, "leftover" is just "the rest of my own
stripe". After its last M6 task a CTA continues its stripe from wherever it got
to, in the same loop, with the same cursor. **No handoff, no rendezvous, no
fall-through predicate, no proof obligation.** The existing M7.6 drain
(`KRN:1395-1467`) absorbs the tail exactly as it does today.

The one real consequence is **progress skew**: a CTA blocked on `a2_done` falls
behind its peers. Since all stripes are uniformly spread over the task space and
readiness is globally uniform (§6.0), the CTAs are symmetric and the skew is
self-correcting — but it is the mechanism by which the interleave can *lose*, and
it is why the `k` sweep must include `k = ∞`.

#### 3.0.4 The `a2_done` ordering fix (H1) — one instruction, zero barriers

`protocol_review.md` §H1 is now on the critical path. The release at
`KRN:227-234` is a **tid-0-only** `fence(release, agent)` guarding a counter that
publishes stores made by **all 256 threads**. Today CTA program order hides it;
under the interleave a *different* CTA reads `A2q` while this one is still in M6,
and the fence orders nothing it did not itself write.

The vendored file's sequence (`n2_phase1_gm_mps.cpp`) is already the right shape:

```
612-613, 626-629   all 256 threads store A2q / DQ2      (plain uint4 / f32)
637                N2GM_P1_EPILOGUE_DONE_HOOK           <-- MPS-DELTA (10), empty today
639                __syncthreads()
646-653            N2GM_TASK_DONE_HOOK  ->  k0p6_a2_arrive()   (tid-0 fence + RMW)
```

**The fix is to define the hook, in `KRN` only, as an all-thread
`s_waitcnt vmcnt(0)`.** That composes `producer_drain_release<agent>` out of parts
that already exist:

| `producer_drain_release<agent>` step | supplied by |
|---|---|
| drain every lane's VMEM | **the new hook at line 637** (all 256 threads) |
| CTA convergence | the existing `__syncthreads()` at line 639 |
| leader agent-scope release fence | the existing tid-0 fence in `k0p6_a2_arrive` |
| broadcast the result | **not needed** — nothing consumes a return value |

The ordering argument for why *one* thread's release fence suffices: a normal
store retires into the **local L2**, which is not visible to other XCDs; the
agent-scope release fence emits the L2 writeback, and that writeback covers the
CTA's **entire L2 footprint**, not just the fencing thread's lines. It is
therefore sufficient *provided every thread's store has already reached L2* —
which is precisely what `vmcnt(0)` + the barrier guarantees, and precisely what is
missing today.

**Cost, counted:** one `s_waitcnt vmcnt(0)` per wave per M6 task =
`2,840 × 4 = 11,360` waits per rank per epoch. **Zero new barriers** (reuses line
639), **zero new fences** (reuses the tid-0 fence), **zero edits to the vendored
file**. Against M6's existing ~66 barriers/task the marginal cost is ≈ 0; measured
as time, ~1 µs per task per CTA → **≈ 11 µs per epoch**. This is the cheapest
blocking-bug fix in the whole exp_25 design.

**Why the campaign cannot catch H1 on its own** (restated plainly, because it is
the single most dangerous property of this experiment): the MoK harness feeds
**identical input and identical routing every iteration**. So epoch `e−1`'s `A2q`
bytes are **bit-identical** to epoch `e`'s. A missing readiness edge makes M7 read
*last epoch's* `A2q` — which contains the same numbers. The kernel returns the
right answer, `rel_L1` passes, `max_abs` passes, `pperr = 0`, the 600-epoch soak
passes, and because it skipped a real wait it **posts the best time in the
sweep**. A broken build wins the ratchet. Nothing in the gate ladder can see it.

**Therefore the DQ2-NaN-poison detector arm is a signoff condition, not an
option.** At the end of each M6 task, before the drain, write `NaN` into the
task's `DQ2` cells for the *next* epoch's slot — i.e. poison what a stale reader
would see. M7 reads `DQ2` before its K-loop; any stale read produces `NaN`, which
propagates to `out` and is caught by the **zero-nonfinite** gate on the very first
iteration. Cost ~2.1 MB of extra stores, ~50 µs, and it is asymmetric across
epochs so it must be run as a **separate detector arm, never as the timed arm**.

#### 3.0.5 The L2 co-residency risk, quantified — the strongest argument against

Per XCD: 32 CTAs, **4 MB** 16-way L2.

| window | resident working set | vs 4 MB |
|---|---:|---|
| today, M6 alone | 2.9 experts × 3.67 MB slab (`g = xcd`) = **10.6 MB** | 2.65× over |
| today, M7 alone | **2.64 MB** (`m6_m7_structure.md` §3.3) | **fits (66 %)** |
| **interleaved** (≈16 CTAs in each body) | `16/32 × 10.6 + 16/30 × 2.64` = **6.7 MB** | 1.68× over |

So the interleave **improves** M6's pressure (10.6 → 5.3 MB of M6 footprint) and
**destroys** M7's residency (2.64 MB fits → 1.4 MB of it competing inside a 6.7 MB
demand). M7's `W2` stream is the one the kernel deliberately keeps L2-resident.

**Bandwidth bound on the damage — and it is reassuring.** If M7's `W2` stream
loses residency entirely, all 5.211 GB of it crosses the XCD↔IC path. Total
request traffic for the combined window becomes `12.456 (M6) + 6.817 (M7)` =
19.27 GB over 5,248 µs = **3.67 TB/s** — which is **25 % *below* the 4.91 TB/s
that M6 alone already sustains**. **The Infinity Cache has the headroom.** So the
damage is bounded by **latency, not bandwidth**.

**Latency is where it bites.** M7's K-loop is only 16 steps deep with a
one-step-deep software pipeline, so it is the more latency-exposed of the two
bodies. If its achieved bandwidth falls 4.30 → 3.8 TB/s, `A7K` inflates by
`(4.30/3.8 − 1) = 13 %` → **+222 µs on the makespan**. Scaling exp_08's measured
result — per-XCD L2 effects at this boundary were **36 % of M7's interference** —
onto the 976 µs surcharge gives an independent **−0 to −110 µs**. Take the range
as **−0 to −222 µs**.

**Predicted shape of the `k` sweep.** `k` trades the two directly: small `k` =
maximal burst de-alignment (+W1) and maximal L2 thrash (−L2); large `k` = neither.

- If W1 > L2: **interior optimum near `k = 2–4`** (enough transitions to
  de-align 240 CTAs' bursts, few enough that each body gets a run of ~460-920 µs
  to re-establish residency).
- If L2 > W1: **monotone increasing in `k`, optimum at `k = ∞`** — i.e. *the sweep
  says do not interleave*, and it says it cheaply.

**My prediction is the second**, because W1's ceiling (+211 µs at unreachable
100 % fabric efficiency) and the L2 term (−222 µs) are the same magnitude with
opposite signs and the L2 term does not need a heroic efficiency assumption to be
realized. I put ~60/40 on `k = ∞` winning. **This is a genuine risk of a null
result, and it is why Policy 0 should be built as a 4-6 h screen and not as a
multi-stage experiment.**

#### 3.0.6 The in-order `vmcnt` coupling — a real cost, correctly sized

`vmcnt` is **one in-order counter per wave** (ISA §4.4: memory ops return in issue
order). So after an M7 epilogue, M6's next compiler-inserted `s_waitcnt vmcnt(N)`
implicitly waits on the epilogue's outstanding remote atomics — whose ACK latency
(~3.5 µs, derived from `976 µs / (2,230 atomic instructions / depth 8)`) is ~5×
M6's one-K-step latency tolerance (1,536 cycles ≈ 0.7 µs).

**But this is a per-transition latency, not a per-atomic one.** At `k = 1` there
are ~11 transitions per CTA per epoch → `11 × 2 × 3.5 µs ≈ 77 µs`; at `k = 4`,
~20 µs. Bounded and small. It does, however, **kill the fine-grained variant** —
issuing the epilogue's atomics and deferring their drain across an M6 K-loop, so
the ACK stall hides under MFMA — because M6's own loads cannot be waited on
without also waiting for the earlier atomics. That variant is dead on ISA grounds,
not on register grounds. Record it as closed.

### 3.1 Policy 1 — static split, no fall-through  *(superseded — see §3.5a)*

`bid < C6` runs M6 then goes straight to the service drain; `C6 ≤ bid < 256−C`
runs M7 only.

**Ratio from the work ratio.** With `T6(n) = a6 + b6/n` and
`T7(n) = a7 + b7/n`, balance `T6(C6) = T7(240 − C6)`. At `a6 = 620`
(M6 scaling "like M7") the balance is `C6 ≈ 128`, `C7 ≈ 112`. At `a6 = 0` it is
`C6 ≈ 129`. The ratio is insensitive to `a6`; the *makespan* is not (§5.2).
2,539 : 1,586 in the older docs and 2,588 : 2,660 today give `C6/C7` of 1.61 and
0.97 respectively — **use the mode-12 numbers (0.97, i.e. C6 ≈ 118) as the
sweep centre, not the mode-0-era 1.6.**

**Primitive.** None new: a static prefix predicate on `blockIdx.x`, exactly the
shape of `is_service_cta` (`ADP:292-297`) plus dense ids exactly the shape of
`compute_id_of` (`ADP:299-306`). `roles.cuh::finish_order_partition`
(`ROLES:62-124`) is deliberately **not** used — `DESIGN_MPS.md:119-125` records
it costing +24 B/lane of VGPR-pair spill at the phase-2 pointer peak.

**Atomic cost per CTA per epoch: zero.** `blockIdx.x` is a hardware uniform.

**Failure modes.**
- Both ends idle: the M7 pool idles for the first ~229 µs (no ready tile); the
  M6 pool idles from when M6 ends until M7 ends — up to `T7(C7) − T6(C6)`, which
  at balance is ~0 but at any mis-set `C6` is the full imbalance.
- **The M6 pool's exit lands it in `run_service` (`KRN:1402-1414`) mid-M7.** At
  C6 = 128 that is 512 waves spinning on `q[k]` + the single shared `pperr` word
  for the second half of the window. exp_20's mode 8 identified exactly this
  spin as a first-order interference term. **Policy 1's measured number will
  therefore under-state its own mechanism; it is a lower bound, not the verdict.**
- If `C6 < 8` or `C6 % 8 != 0`, some M6 chunk `g` is never computed →
  `a2_done[b]` never reaches 8 → guaranteed timeout (see §4.3).

**Cost to build: two helper returns and one predicate.** This is the reason it
goes first.

### 3.2 Policy 2 — static split + M6-pool fall-through (exp_12's proven trick)

M6-pool CTAs join the M7 pool when their stripe drains. exp_12 (`+1.464× →
1.077×`, the largest single win in the campaign) is the same idea applied to the
event queue, and `aug10/STATUS.md:408` records it as such.

**The catch that makes this more than a one-liner:** M7's task space is covered
by a *static* stride over C7 CTAs. A late joiner has nothing left to claim. So
fall-through requires the M7 task space to be dynamically claimed — and a naive
global monotonic ticket **breaks the `nc → XCD` invariant asserted at `P2:77`**,
destroying M7's L2 residency (§2.4).

**The fix, and it is the interesting part of this design: per-XCD ticket banks.**
Workgroups go round-robin to XCDs with chunk size 1 (`xcd = bid % 8`,
`ADP:93-96`). Keep 8 counters; a CTA on XCD `x` claims from bank `x` and receives
`task = x + 8k`. Then `task ≡ x (mod 8) ⇒ nc ≡ x (mod 8)` **exactly as today**,
any CTA on XCD `x` can join bank `x`, and the invariant is preserved by
construction rather than by luck.

**Primitives.** `CTR::counted_arrive_into`-shaped claim on 8 cells; the claim
itself is `fetch_add_relaxed<agent>` (`SYNC:136-140`). The broadcast is a
CTA-local LDS word plus the `__syncthreads()` that
`k0p6_mps_task_flush_defer` (`KRN:355-364`) **already executes at the loop
head** — so the barrier is free.

**Atomic cost: 1 agent RMW per M7 task = ~5,680 per rank per epoch on 8 cells
(~710 per cell).** Against the 918,472 protocol RMWs and 117.4 M payload RMWs
already in the epoch (`mode12_protocol_map.md` §3), this is 0.6 % of the protocol
count and structurally invisible.

**Failure modes.**
- Head-of-line blocking: a claimed task whose tile is not ready spins even
  though higher-index ready tasks exist. Because all CTAs on an XCD walk the
  same increasing front, this degenerates to "everyone waits for the same tile",
  which is correct and is exactly the rate-matching we want — but a spinning M7
  CTA is pure waste, worse than an idle one (it issues poll loads).
- Storage: 8 counters. `mps_state` has 4 free scalar words
  (`ADP:34-38`: lanes 0-3 used, `K0P6_MPS_ST_WORDS = 8`), so this needs
  `ST_WORDS` widened to 16 — additive, but it changes `mps_state_bytes()`
  (`ABI:186-188`) and therefore the host allocation. Stage-2 work, not Stage-1.
- The loop's induction variable stops being affine. **This is the register risk;
  see §6.4.**

### 3.3 Policy 3 — dynamic role ticket / backlog-driven join

A CTA finishing an M6 task reads the ready-tile backlog and switches role
permanently to M7 when the backlog exceeds a threshold. Self-tuning: CTAs shed
from M6 exactly fast enough to keep the M7 pool fed, so `C6` disappears as a
tuning knob.

**It requires a dynamic M6 ticket too.** With a static M6 stride, a CTA that
leaves M6 *abandons its stripe* — those tasks never run, `a2_done` never reaches
8, and the launch times out. So Policy 3 = per-XCD M6 banks + per-XCD M7 banks +
a backlog counter. Both banks must be per-XCD for the same L2 reason (M6's
`g = task mod 8` is likewise XCD-stable).

**Primitives.** The backlog is `(tiles produced) − (M7 tasks claimed)`; the
natural expression is two `CTR::epoch32`-style monotonic cells read relaxed. The
role switch itself is a `break` out of the M6 loop — **no register cost, because
the two bodies stay in separate scopes** (this is the key realisation: a *role
switch* is not an *interleave*).

**Atomic cost: 1 RMW per M6 task (2,840) + 1 per M7 task (5,680) + one relaxed
load per M6 task boundary.** Still negligible.

**Failure mode that kills it if not designed for:** with per-XCD M6 banks, XCD
`x` produces only chunk `g = x` of every tile, so **a tile is ready only when
all 8 XCDs have produced their chunk. If CTAs shed from M6 unevenly across XCDs,
one XCD's M6 bank falls behind and the global tile front stalls while seven
XCDs' M7 pools starve.** The threshold must therefore be evaluated on a
*global* M6-remaining count, not a per-XCD backlog. This is a real, subtle,
non-obvious hazard and it is why Policy 3 is Stage 3 and not Stage 1.

### 3.4 Policy 4 — interleave inside every CTA (the intended end state)

Every CTA alternates `1 M6 task : 2 M7 tasks`. **This is the only variant that
escapes §2.1's work-conservation loss**: both phases get all 240 CTAs, the split
is self-balancing, there is no `C6`, no imbalance, no start bubble beyond the
first tile, and no tail. In the ideal model it strictly dominates policies 1-3.

Its cost is the one `m6_m7_structure.md` §4.3 item 2 already flagged, and it is
the reason it is Stage 4 and not Stage 1: the two bodies live in **one loop**, so
both bodies' pointer sets are live across the loop, at exactly the phase-2
pointer peak where `PROVENANCE.md:98-102` and `DESIGN_MPS.md:119-125` measured
two ticket words costing +24 B/lane of spill. The accumulators are safe (each is
born and dies inside its own arm, so the allocator takes `max(192, 168)` not the
sum), and the descriptor re-read discipline (`k0p6_dread`, `KRN:204-209`) can
keep the pointers out of the loop-carried set — but that is an argument, not a
measurement, and the measurement costs a build.

Secondary risk: I-cache. Both MFMA loops inlined into one hot loop; gfx950's
32 KB I-cache is shared by two CUs. Unquantifiable from here.

### 3.5a Recommendation — REVISED (rev 2)

**Build Policy 0 (§3.0) as Stage 1. It dominates Policies 1-3 on every axis:**

| | Policy 0 (interleave) | Policy 1/2 (static split) |
|---|---|---|
| work-conservation term | **0** | **−173 µs** |
| new mode + 7 predicates | **none** | required |
| new atomics | **0** | 0 / 5,680 |
| ABI / `mps_state` change | **none** | none / one cell |
| over-coverage hazard (§6.3) | **impossible by construction** | live, needs `part_done` detector |
| gated on unmeasured `T6(C6)` (F1) | **no** | **yes** |
| exp_01 `address (nil)` re-arm risk | **none** (M7 start unchanged) | live |
| control arm | **mode 12 itself, already ratcheted** | must be built |
| build cost | **4-6 h** | 14-19 h |

**But the static split retains option value the interleave gives up, and this is
the one reason not to close §3.1-3.4.** F1 is the coefficient `T6(128)/T6(256)`.
In the *favourable* branch — M6 largely insensitive to CTA count — the static
split wins up to **−535 µs**, because it can hand M7 more CTAs for the same M6
time. The interleave wins **0 first-order regardless of F1**. So:

**The two mechanisms are not ordered.** If M6 turns out CTA-insensitive, the
static split is strictly better than the interleave. Therefore **F1 remains the
highest-value measurement on this boundary even though the interleave does not
need it** — it is no longer a *gate* on Stage 1 (that is the assignment's claim,
and it is correct), but it is still the thing that decides whether Stage 2 exists.
Keep it, demoted from gate to option-pricing.

### 3.5 Recommendation, rev 1 — SUPERSEDED

> Superseded by §3.5a: Policy 0 dominates on every term at a third of the cost.

**Build Policy 1 first (Stage 1), then Policy 2 (Stage 2). Recommend Policy 2 as
the shipping shape if the boundary survives Stage 1.**

Reasoning:
1. Policy 1 costs ~15 lines on top of the vendored-file work that Stage 0 needs
   anyway, and it prices the mechanism with **zero** new protocol, **zero** new
   atomics, **zero** register risk, and **zero** ABI change.
2. Policy 2 is worth a computed ~120 µs (tail) plus an unquantified but
   plausible ~150 µs (removing the M6 pool's early spin), on ~5,680 extra RMWs.
   That is a good trade, and per-XCD banking makes it safe rather than clever.
3. Policy 3's prize over Policy 2 is only the tuning of `C6`, which the Stage-1
   sweep will have already measured. Its risk (the cross-XCD front stall) is
   larger than its prize. Build it only if the C6 curve turns out to be sharp.
4. Policy 4 is where the *headline* number of this line of work most likely
   lives, because it is the only one without the work-conservation penalty — but
   it should be built with the split's measurements in hand, not instead of them.

---

## 4. Correctness

### 4.1 The exact readiness predicate

An M7 CTA about to execute task `(tile, nc)` must observe, **for every live
sub-block** `b = b0 + sb`, `sb < gcount`:

```
a2_done[b] >= 8
```

- `b0 = tile_desc[tile] >> 4`, `gcount = tile_desc[tile] & 0xF` (`P2:295-300,
  333-335`).
- Poll site: `k0p6_a2_wait` (`KRN:236-251`), `poll_local_count<8>` at
  `KRN:241-243` → `HKA:105-115` → `CMPL:104-118` `bounded_poll_relaxed_into
  <agent>`; called per live sub-block from `P2:336-344`, at the top of the task
  **before any LDS fill**.
- Update site: `k0p6_a2_arrive` (`KRN:227-234`) → `HKA:92-97`
  `arrive_local_count<true>` = `thread_release<agent>` then
  `fetch_add_relaxed<agent>(a2_done + b, 1)`, by **tid 0 only**, fired per live
  sub-block at `P1:481-488`.
- Reset: grid-strided zero in M0, `KRN:723-726`, ordered before any reader by
  the M3/M5 grid barriers (`KRN:1236`, `KRN:1301`).

**Why 8 is exactly right.** An M6 task `(tile, g)` writes
`A2q[(b0+sb)·32 + cr][256g … 256g+256)` (`P1:444-452`) and
`DQ2[(b0+sb)·32 + 16m+r][2g+kb]` (`P1:465-468`). An M7 task reads **all 2048
A2q columns and all 16 DQ2 columns** of its rows (`P2:366-375`, and the K-loop
over `kKGroups2 = 16`). So M7 task `(tile, nc)` depends on exactly the 8 M6
tasks `(tile, g = 0..7)` — same tile, all chunks, nothing else. Expert-aligned
tiling gives 1:1 tile ownership of every 32-block and the pad sub-blocks
deliberately do not fire (`P1:475-488`), so `a2_done[b]` is incremented
**exactly 8 times, by its owning tile's 8 chunk-tasks only.**

**Not used, deliberately:** the finer K-wise dependency (M6 chunk `g` produces
exactly K128 pair `(2g, 2g+1)`). It would need `a2_done[b][8]` and a wait
*inside* the K-loop, which is the exact class exp_05/exp_20 measured inflating a
concurrent GEMM by 37–57 %. `m6_m7_structure.md` §4.3 says "do not start there";
agreed.

### 4.2 Every release/acquire pair, and what breaks when they become concurrent

The full table is in `protocol_review.md` §2. The load-bearing statement for
this design:

> **Today the poll at `KRN:241-243` never spins.** All 256 CTAs carry
> near-identical M6 work, so by the time any CTA polls `a2_done[b]` the counter
> is long since 8 (`m6_m7_structure.md` §4.3: "It already may — and it never
> does"). The release/acquire pair is therefore **doing no ordering work today**;
> CTA program order is doing it for free. **The split makes that pair live for
> the first time in this kernel's history.**

Two things break, in order of severity.

**(1) The payload release is a tid0-only agent fence behind a bare
`__syncthreads()`.** `P1:474` is `__syncthreads()`; `HKA:95` then issues
`thread_release<agent>` = `SYNC:150-153` → `SYNC:88-100` →
`__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")` **on tid 0 only**. But the
`A2q` stores at `P1:444-452` are issued by **all 256 threads** (`cr = tid>>3`,
`cc = tid&7`) and the `DQ2` stores at `P1:456-472` by the `q == 0` lanes of all
four waves. A release fence orders the *calling thread's* prior operations. What
makes the other 255 threads' plain `uint4` stores visible at **agent** scope —
across eight XCDs with eight non-coherent 4 MB L2s — is not established by the
source. `SYNC:160-174`'s own comment records that the gfx950 lowering of a
system release was `buffer_wbl2 sc1` and says "re-check that fingerprint for
every target/toolchain"; whether an **agent** release on gfx950 emits an L2
writeback is **UNRESOLVED here and must be read off the disassembly before the
first GPU run** (`protocol_review.md` §7, check C1).

**The fix, and it is the primitive-shaped one.** `SYNC:166-174`
`producer_drain_release<Scope>()` is exactly the missing semantic: whole-CTA
`s_waitcnt vmcnt(0)`, CTA barrier, one leader release, CTA barrier. Replacing
`P1:474` + `HKA:95` with `producer_drain_release<agent>()` followed by the
relaxed `fetch_add` costs one explicit `vmcnt(0)` and one extra CTA barrier per
M6 task — **2,840 per rank per epoch**, against a 66-barrier-per-task budget
(`m6_m7_structure.md` §5.7) — and removes the question entirely. Because the M6
body is read-only, this lands in the vendored sibling as a second delta, or (my
preference) in `k0p6_a2_arrive` (`KRN:227-234`), which we own: make it
convergent (`producer_drain_release<agent>()` on all threads, then tid0's RMW)
instead of tid0-only. **Recommended for Stage 1; mandatory for any arm whose
number we intend to keep.**

**(2) The timeout path partially executes M7 and leaves persistent state.**
`KRN:406-407` is `if (!k0p6_a2_wait(b, tid, k0p6_desc)) return;` — a `return`
out of the phase-2 body from inside the per-sub-block loop. It is
CTA-uniform (tid0's `atomicOr(pperr, 16777216)` happens before the
`cta_acquire` inside `k0p6_a2_wait` at `KRN:248`, so every wave's
`error_bit_set_agent` read at `KRN:249-250` sees the bit), so no wave diverges
and no `__syncthreads()` hangs. But it skips `N2GM_TASK_LOOP_TAIL_HOOK`
(`P2:564-566`), losing the deferred events, and the CTA has **already issued
remote atomics into peers' slots** for its completed tasks. Those rows never get
`row_ready`, so the owner's M8 skips them, so **the consume-and-zero at
`KRN:579-587` never runs for them and the garbage persists into epoch N+1.**
`pperr` is terminal by policy (`aug10/CLAUDE.md`), so this is fail-closed — but
it is a new class of persistent corruption that today cannot occur, and it makes
the spin-limit margin a correctness argument rather than a comfort. §4.3 does
that arithmetic.

### 4.3 Deadlock and liveness

**Claim: no legal configuration can wedge, and every illegal one fails closed
within a bounded time.**

Every wait in the M6→M9 window is bounded:

| wait | site | bound |
|---|---|---|
| `a2_done[b] >= 8` | `KRN:241-243` | `spin_limit`, then `pperr` `1<<24` |
| `q[k] != 0` | `ADP:448-464` | `spin_limit`, then `K0P6_MPS_ERR_SERVICE` |
| `row_ready >= epoch32` | `KRN:511-514` | `spin_limit`, then `pperr` `1<<25` |

**The arithmetic that makes the bound real.** `spin_limit = 2,000,000`
(`K0_SPIN_LIMIT`, `harness_recipe.md` §2.2). Each spin iteration executes
`detail::pause()` = `__builtin_amdgcn_s_sleep(4)` (`SYNC:81-86`) ≈ 256 core
clocks, so the bound is ≈ `2e6 × 256 / 2.2e9` = **233 ms**.

The worst *legitimate* wait under the split is the whole M6 phase on `C6` CTAs
(a service wave claims ticket 0 immediately after M5 and spins until the first
M7 task publishes):

| `C6` | `T6(C6)` at `a6=620` | margin vs 233 ms |
|---:|---:|---:|
| 240 | 2.72 ms | 86× |
| 128 | 4.56 ms | 51× |
| 64 | 8.5 ms | 27× |
| 32 | 16.4 ms | 14× |
| 8 | 63.6 ms | 3.7× |

**Therefore: validation must floor `C6` at 32** (14× margin). At `C6 = 8` the
margin is 3.7× and a single slow epoch could mint a spurious timeout — which,
per §4.2 item 2, leaves persistent slot corruption. This is a concrete,
quantitative reason for a validation rule, not a stylistic one.

**Case `C6 = 0`:** no CTA runs M6 → `a2_done` stays 0 → every M7 CTA times out
(233 ms) → `pperr` → all M7 CTAs return → no events → the pool times out →
`pperr` → M8's `__ballot(batch_ready) != ~0` skips every batch (`KRN:523`) →
M9 still retires the launch (`KRN:1822-1862`). **Fails closed in ~0.5 s, no
wedge — but it must be rejected by `config_is_valid` anyway.**

**Case service + M7 pools starve M6:** cannot happen. The pools are disjoint
static prefixes of `blockIdx.x`; the M6 pool's CTAs are never preempted and
never wait on anything (M6 has no readiness poll of any kind,
`m6_m7_structure.md` §5.6). M6's progress is unconditional.

**Case `C7 = 0`:** nobody runs M7 → same fail-closed path. Reject.

**Case `C6 % 8 != 0` or `C6 < 8`:** with M6 start `= bid`, stride `= C6`, the
covered task set is `{bid + kC6}` over `bid ∈ [0,C6)` = `[0,∞)` — coverage is
fine for any `C6 ≥ 1`. **But** `g = task mod 8` and `task ≡ bid (mod 8)` only if
`C6 ≡ 0 (mod 8)`; if `C6 = 12`, coverage still holds but the XCD invariant
breaks (a performance defect, not a correctness one). If instead the *dense-id*
form is used for M6 as well, `C6 < 8` leaves some `g` uncovered. **Reject
`C6 % 8 != 0` and `C6 < 32`.**

**Case `C7 < 8`:** some XCD has no M7 CTA. With `start = bid − C6`, `stride =
C7`, coverage is `[0,∞)` so tasks still all run — but they run on the wrong
XCDs and the invariant is gone. **Reject `C7 < 8`.**

**The bounded-poll timeout path itself.** `bounded_poll_relaxed_into`
(`CMPL:104-118`) breaks out with `result.ready == false` and does **not**
acquire; `poll_local_count` (`HKA:105-115`) then fires the timeout functor
(`atomicOr`), and `k0p6_a2_wait` `atomicOr`s again at `KRN:244`. The `return` at
`KRN:406-407` is the only consumer of the false. No path reads payload after a
failed poll: the `cta_acquire` at `KRN:248` runs unconditionally (it is the CTA
convergence point), but the `return` happens before any A2q load. **Sound.**

### 4.4 Epoch lifetime — can an epoch-(e+1) M6 CTA overwrite an epoch-e A2q row?

**No, and the reason is not the split.** One launch = one epoch
(`mode12_protocol_map.md` §7: "one epoch per launch"). Epoch e+1's M6 lives in a
*different kernel launch on the same stream*, so stream ordering guarantees
every epoch-e CTA has retired. `A2q`/`DQ2` are **local, not symmetric**
(§7 buffer table: `a2q` slot 27, `dq2` slot 28, "no"), so no peer ever touches
them. Within one epoch each `(row, 256-column band)` of `A2q` is written exactly
once by exactly one M6 task and read only by M7 tasks of the same tile after
`a2_done == 8`. **There is no reuse, hence no retirement handshake to trace and
no counter that must gate it.** `LIFE::bounded_wait_slot_reusable_into` /
`drain_and_retire_slot` are correctly *not* needed here — that is a positive
finding about the buffer's capacity sizing (571 MB of HBM buys it,
`m6_m7_structure.md` §4.4), and it is what makes fine-grained M6→M7 overlap
cheap.

The one residual: rows `≥ nvi[0]` hold stale bytes from earlier epochs. They are
unreachable — `a2_num_records = nvi[0] × kInter` bounds the hardware descriptor
(`P2:302-306`) and `tile_desc` names only live blocks. Unchanged by the split.

**But see `protocol_review.md` §6 item 1.** There is a much nastier
epoch-lifetime interaction that is *not* a hazard in the ordinary sense: because
the MoK synthetic campaign feeds **identical input and routing every iteration**,
epoch e−1's `A2q` bytes are bit-identical to epoch e's. **A missing readiness
edge would therefore produce a numerically correct answer and pass a 600-epoch
soak.** That is the single most dangerous property of this experiment and it
requires a purpose-built detector (§7 L8).

### 4.5 Interaction with the mode-12 protocol

| protocol object | does the split change it? |
|---|---|
| event queue `mps_q`, `mps_state[TAIL]` | **No.** Every M7 task still runs exactly once, so exactly `EV = 16·B` events are enqueued and `events_total` (`KRN:1418`) is still reached. The *order* changes (tile-major front instead of bid-interleaved), which the ticket-based consumer (`ADP:649-650`) is indifferent to by construction (exp_12's design note at `ADP:719-736`). |
| `nc_arr[r·16+nc]`, target `row_rem[r]` | **No arrival target changes.** Each `(r,nc)` still receives exactly `row_rem[r]` arrivals, one per contributing `(b,nc)` event. The counting logic is order-independent. |
| `pushed[r]`, target 16 | **No.** |
| `row_rem[r]` self-clean to 0 (`ADP:587`) | **No, but the invariant must be re-argued.** The zeroing is done by the unique publisher after `pushed[r]` hits 16, which requires all 16 slices complete, which requires all `16 × row_rem[r]` arrivals. So no event for row `r` can arrive after the zeroing. The split changes the event *order*, not the *set*, so the argument survives — **and it survives only because the set is unchanged. Any variant that enqueues an event twice breaks it silently** (`protocol_review.md` §6 item 3). |
| `row_ready` epoch publication, M8's `>= epoch32` poll | **No.** Monotonic, epoch-tagged. |
| M8 consume-and-zero | **No** — except on the timeout path (§4.2 item 2). |
| epoch tags | **None change.** `env.epoch32` (`KRN:1444`) comes from the same monotonic cell. |
| **Does the service pool now run during M6?** | **Yes, and this is a real new cost.** The C service CTAs skip M6 and reach `KRN:1414` right after M5, and the M6-pool CTAs reach it as they drain. See §3.1's third failure mode. **Make it a swept sub-variable** (`g` bit `0x40` = "service pool runs M6 too"); the two arms are ±16 CTAs of M6 capacity (≈150 µs) against 16 CTAs × 4 waves of early polling. |

### 4.6 The `pperr` bits each failure mode fires

| failure | bit | site |
|---|---|---|
| `a2_done` poll timeout (M6 pool too small, mis-set `C6`, or a real stall) | `16777216` (`1<<24`) | `KRN:243-244` |
| event-queue poll timeout in the pool (events lost because an M7 CTA returned early) | `67108864` (`1<<26`, `K0P6_MPS_ERR_SERVICE`) | `ADP:458` |
| `row_ready` poll timeout in M8 (rows never flagged) | `33554432` (`1<<25`) | `KRN:514` |
| `C6`/`C7` illegal, or `events_total > queue_capacity` | `268435456` (`1<<28`, `K0P6_MPS_ERR_CONFIG`) | `KRN:1423`, entry guard `KRN:678-683` |
| grid-barrier failure (unchanged) | `2097152` | `KRN:1292, 1302` |
| dual-write detector, if the diagnostic arm is used | `134217728` (`1<<27`) | `KRN:607` |

**The gap:** there is **no** `pperr` bit for "an M7 task ran twice" or "an M6
task never ran". Both are silent. §7 L9 proposes the free detector.

---

## 5. Cost model and pre-registered falsifiers

### 5.1 The model

```
T6(n) = a6 + b6/n           a6 UNMEASURED, b6 = (2588 − a6)·256
T7(n) = a7 + b7/n           fitted:  a7 = 340, b7 = 322,478   (exp_05 stage0, mode 0)
                            mode 12: scale by 2660/1684 = 1.580  ⇒ a7' = 537, b7' = 509,515
W1 relief: multiply T7 by (2660 − R)/2660,  R ∈ [0, 450] µs
split makespan ≈ max(T6(C6), T7(240 − C6)) + 138 µs   (start bubble + residual tail)
                 + 0…300 µs                            (early-spin + per-XCD L2, §2.4/§3.1)
sequential      = 2,588 + 2,660 = 5,248 µs             (max-stamp basis)
epoch conversion: × 4,647/5,248 = 0.885
```

### 5.2 Sensitivity to the one unmeasured coefficient

With **no** W1 relief (`R = 0`) and **no** interference penalty — i.e. the split's
mechanical best case:

| `a6` (µs) | `T6(128)/T6(256)` | best `C6` | split makespan | Δ vs 5,248 | epoch Δ | total |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.000 | 129 | 5,269 | **+21** | +19 | 6,704 |
| 200 | 1.923 | 127 | 5,168 | −80 | −71 | 6,614 |
| 400 | 1.846 | 124 | 5,059 | −189 | −167 | 6,518 |
| 620 | 1.760 | 122 | 4,948 | −300 | −266 | 6,419 |
| 1,000 | 1.613 | 114 | 4,713 | −535 | −473 | 6,212 |

**Read this table as the experiment's thesis.** The payoff is a monotone
function of how CTA-*insensitive* M6 is, and that quantity has never been
measured. `m6_m7_structure.md` §2.5 puts its own prediction on the record:
"an honest M6 CTA sweep will land in 1.20–1.32× [for 194/256], not 1.005×",
which maps to `a6 ≈ 500–900` — the favourable half of the table. But it is a
prediction, not a measurement.

### 5.3 Central prediction

Take `a6 = 620` (mid of the documented prediction), `R = 300 µs` of W1 relief,
and `−150 µs` for the early-spin + per-XCD L2 penalty:

```
T7(n) = (2360/2660)·(537 + 509,515/n) = 476 + 452,048/n
balance at C6 = 128:  T6 = 620 + 3,936 = 4,556   T7 = 476 + 4,036 = 4,512
makespan  ≈ 4,538 + 138 (bubble/tail) + 150 (interference) = 4,826 µs
Δ vs 5,248 = −422 µs  →  epoch −373 µs
```

**Predicted total: 6,685 − 373 = 6,312 µs = 0.818× production.**
Best `C6` ≈ 120–130 with `C = 16`, i.e. `flush_rows ∈ {30, 32}` in the encoding
of §6.2.

**Band:**

| scenario | assumptions | total | ratio |
|---|---|---:|---:|
| pessimistic | `a6 = 0`, `R = 0`, `−300 µs` interference | **6,969** | 0.903 |
| central | `a6 = 620`, `R = 300`, `−150 µs` | **6,312** | 0.818 |
| optimistic | `a6 = 1,000`, `R = 450`, `0 µs` | **5,877** | 0.762 |

### 5.4 Pre-registered falsifiers

**F1 — the cheap one, and the one that matters. Stage 0, before the overlap arm
exists.** Run the mode-10 M6 CTA sweep at `C6 ∈ {240, 192, 128, 96, 64, 32}`.

- **`T6(128)/T6(256) ≥ 1.90` (i.e. `a6 ≤ 200 µs`): STOP. Do not build the
  overlap arm.** The work-conservation loss cancels the entire prize; the
  boundary is closed on measurement, and the kill is logged against the
  *boundary*, not against role specialization.
- `≤ 1.80` (`a6 ≥ 300 µs`): build Stage 1.
- between 1.80 and 1.90: build Stage 1 **only if** exp_24's throttle-depth sweep
  independently shows ≥ 150 µs of residual rate headroom in the epilogue — i.e.
  only if W1 is confirmed to still have room.

**F2 — the total, at campaign resolution (5 rotations, 500 warmup, 100 timed),
best point over the `C6` sweep.** Campaign reproducibility is 0.04 %
(6,685.5 vs 6,683.1); screens are ±0.52 % 1σ on the ratio and read ~1.2 points
optimistic, so **F2 is only evaluated on a campaign**.

| result | verdict |
|---|---|
| **≥ 6,685 µs** | **The M6/M7 boundary is CLOSED.** Log the `C6` curve in `LESSONS.md` and do not build Stage 3/4. |
| 6,540 – 6,685 | Real but under-tuned. Continue the `C6` / fall-through / backoff ladder. |
| < 6,540 (≥ 2.2 %) | Pipeline confirmed. Ratchet moves; build Stage 2/3. |
| < 6,172 | 0.80× target met. |

**F3 — build verification, not physics.** With `timestamps=1`, the overlap arm
**must** show `M7_DONE − M6_DONE → ~0` and `M6_DONE − M5_DONE →` the whole
window (both stamps are grid-wide `ts_last`, `ADP:355-366`, so under real
overlap they converge). **If `M7_DONE − M6_DONE` stays above 500 µs, the pools
are not overlapping and the arm is mis-built — F2 does not apply and the result
must not be logged as a kill.**

**F4 — the mechanism test (§7 L6).** If W1 is the mechanism, then removing the
epilogue throttle (`g = 1` instead of `g = 33`) must cost the overlap arm
**less** than it costs mode 12, because the split substitutes for the throttle:

```
[mode11(g=1) − mode11(g=33)]  <  [mode12(g=1) − mode12(g=33)] ≈ 500 µs
```

If the two differences are equal within noise, **W1 is dead** and any win found
is coming from somewhere we have not identified — which is a reason to keep
looking, not to bank the number.

---

## 5.5 Falsifier for Policy 0 (rev 2, pre-registered)

Denominator: mode 12 `C=16 g=33 flush=16` = **6,685 µs** in the SAME campaign.
Arms: `production, pf6gm_mega, mps_mega`; `mps_mega` swept over `k ∈ {1,2,4,8,∞}`.

| observation | reading | action |
|---|---|---|
| best `k` ≥ 6,685 µs (i.e. `k = ∞` wins) | L2 co-residency ≥ W1. §0.2's ceiling was already thin; the boundary is **closed** | **STOP.** Log the axis closed in `LESSONS.md` with the `k` curve. Do not build Policy 1/2 on the strength of this axis |
| 6,481 – 6,685 µs | partial: W1 real but L2 eats it | one more point (`k` between the best two), then stop |
| 6,172 – 6,481 µs | **mechanism confirmed**, ratchet advances | ratchet, then price F1 for Policy 2 |
| < 6,172 µs (0.80×) | exceeds the §0.2 ceiling — **the model is wrong, not the kernel** | **do not ratchet yet.** Verify against the H1 detector arm first: a stale-`A2q` read is the most likely explanation of a number this good (§3.0.4) |
| any `k` faster than `k = ∞` by > 211 µs | **exceeds W1's hard ceiling.** Impossible by §0.2 unless work is being skipped | treat as a **defect signature**, run the NaN-poison arm before believing it |

That last row is the one that matters most: **this experiment's failure mode is a
number that is too good, not too bad.**

---

## 6. Build plan

### 6.0 Stage 0 — the diagnostic, which is free: readiness is an identity

The assignment asks for the single most valuable thing: *how many M7 blocks are
already ready at various points during M6?* — because that is the hard ceiling on
what any interleave can pull forward.

**It does not need to be measured. It is an arithmetic identity.**

M6's task assignment is `task = bid + 256m` and `tile = task/8`, so the 256 CTAs'
`m`-th round covers tasks `256m … 256m+255` = **tiles `32m … 32m+31`, all 8 chunks
of each**. Therefore:

```
after m global M6 rounds:  exactly 32m tiles are complete  (a2_done[b] == 8)
```

Readiness advances **linearly and exactly in step with M6 progress**: at M6
progress fraction `p`, exactly fraction `p` of M7's tasks are claimable.

Now price it against demand. CTA `bid`'s M7 stripe is `{bid + 240i}`, so its `i`-th
task sits at tile `bid/16 + 15i` (240/16 = 15 tiles per step). It is claimable when
`bid/16 + 15i < 32m`:

```
available:  i < (32/15) m = 2.133 m
required :  j/k          = (16/8)(256/240) = 2.133      (from §3.0.1)
```

**These are equal — to 0.05 %, and identically, not coincidentally**: both reduce
to `(256 × 16)/(8 × 240)`. Two consequences, and they are the whole answer:

1. **The prize is not availability-limited.** The interleave *can* run M7 fully
   concurrently with M6. The mechanism is not dead before we build it — which is
   what the assignment wanted to know, and it is now known for free.
2. **There is exactly zero slack.** The M7 consumption front rides *precisely* on
   M6's production front, for any routing (the identity has no `num_tiles` in it).
   So **every `a2_done` poll by a CTA that is even slightly ahead of its peers is a
   genuine stall**, and the interleave's steady state is M6 and M7 progressing in
   lockstep and finishing together at `t = 5,248 µs` — which is exactly the
   work-conservation answer of §0.3, re-derived independently. Consistent.

**If a measurement is wanted anyway** (to catch deviation from linearity when
skewed routing makes tiles non-uniform in `gcount`), it is 4 lines and no new
buffer: define `N2GM_P1_TASK_HEAD_HOOK` (MPS-DELTA (5), vendored line ~161, already
present) so CTA 0's tid 0 stamps `s_memrealtime` into `mps_state[4..7]` on M6 task
iterations 0/3/6/9, behind `cfg.timestamps`. Combined with the identity, that is
the readiness curve. **But the Tier-1 first/last M6 task-end stamps
(`m6_m7_structure.md` §6.3) already measure the only unknown — the straggler
spread — so even this is arguably redundant.**

**Stage 0 cost: 0 h.** Recommend proceeding directly to Stage 1.

### 6.1a Mode number — I claim NO mode; I claim config bits [34:40)

**exp_29 has taken both 10 and 11** (`kModePipelinedProbe = 10`,
`kModePipelinedCombine = 11`, its design.md:504). That collision is **moot**,
because Policy 0 does not need a mode at all:

The interleave changes *the order in which one CTA executes its own two task
stripes*. The protocol, the partition, every buffer, every counter, every
release/acquire pair and every arrival target are **bit-identical to mode 12**.
Nothing in `run_service`, the M7.6 drain, M8's selector, or the validator can tell
the difference. **So none of the seven predicates needs threading** — including the
exact `env.mode == kModeRemoteAccum` at `ADP:803` that was going to be the awkward
one.

`k` goes in the **free upper bits of the config word**
(`moe_host_abi.hpp:146-163` defines `[0:8) C | [8:16) g | [16:24) mode |
[24:32) flush_rows | [32] pull_fallback | [33] timestamps`; **bits [34:64) are
unused**):

```
[34:40)  k   6 bits, 0..63.   k == 0  =>  infinity  =>  today, bit-identical
```

`k = 0` default means an unset field reproduces mode 12 exactly, so the parity
argument is trivial and every existing campaign config is unaffected. One
`config_is_valid` addition (`ADP:249-286`): `k != 0` requires `mode == 12`.

**exp_29 keeps 10 and 11. I take no mode number. No collision.**

### 6.1b exp_01 `address (nil)` re-arm — not applicable to Policy 0, and why

The assignment correctly warns that changing the M7 task start without the pool
predicate in the same commit re-arms the exp_01 fault class (negative `task` →
negative `tile` → `tile_desc[-8]`). **Policy 0 does not change the M7 task start
or stride at all** — `N2GM_TASK_START` stays `blockIdx.x` and `N2GM_TASK_STRIDE`
stays `kCTAs` (`n2_phase2_gm_mps.cpp:286-294`, untouched). The service CTAs are
excluded by the *existing* `bid < 240` predicate at `KRN:1366-1392`, unchanged.

The fault class is therefore **not re-armed**, and this is a direct consequence of
choosing claim option (c). It is the second time that choice removed a whole
hazard class (the first being over-coverage, §3.0.2). **If a future stage moves to
option (b)'s per-XCD banks, the predicate and the start must land in the same
commit** — carry that warning forward to Stage 2, where it does apply.

### 6.1c Delta against the real vendored file: zero lines

Verified against `n2_phase1_gm_mps.cpp` as it stands after commit `f9bfb4be`:

| what Policy 0 needs | status |
|---|---|
| M6 task start/stride parameterization | **already landed** — MPS-DELTA (3), lines 215-227, whose comment says verbatim *"Added for exp_25's M6/M7 interleave and for the F1 measurement"* |
| a drain point after the A2q/DQ2 stores, before the task-end barrier | **already landed** — MPS-DELTA (10) `N2GM_P1_EPILOGUE_DONE_HOOK`, line 637 |
| readiness-curve instrumentation (optional, §6.0) | **already landed** — MPS-DELTA (5) `N2GM_P1_TASK_HEAD_HOOK` |

**exp_25 needs no edit to `n2_phase1_gm_mps.cpp`.** The rev-1 request for a
"MPS-DELTA (9)" is withdrawn — it exists as MPS-DELTA (3). All Policy 0 edits are
confined to files exp_25 owns: `k0pf6gm_device_tile_mps.hip` (the fused task loop
+ the hook definition), `moe_mps_adapter.cuh` (`k` decode + validator),
`moe_host_abi.hpp` (the bit-field comment only). Note the authoritative donor is
the **585-line** `solution/hip/n2_phase1_gm.cpp` (sha256 `1d90b266…`), not the
634-line exp_65 snapshot.

### 6.1 Mode numbers *(rev 1 — superseded by §6.1a)*

`mode12_protocol_map.md` §6 says 10 and 11 are free; I re-verified —
`config_is_valid` bounds `mode > 13u` (`ADP:271`) and nothing in the tree
references 10 or 11. **Take both:**

- **mode 10 = the matched control.** M6 runs on `C6` CTAs; every other CTA
  **waits** at a rendezvous until M6's pool is done; then the kernel proceeds
  exactly as mode 12. This measures `T6(C6)` and is the *matched-CTA-count*
  control that mode 0 could never be (`aug10/STATUS.md:126-133`: "Mode 0 is
  retired as the control for role specialization... structurally blind").
- **mode 11 = the mechanism.** Same `C6`, but the non-M6 CTAs enter M7
  immediately. **The single variable between 10 and 11 is whether the non-M6
  CTAs wait or work.**

**A new mode must be threaded through *seven* predicates.** The in-source
warning at `KRN:296-303` records that an open-ended `>= 2` in one of them
"would have silently enqueued events nobody consumes". The full list, which is
the highest-risk mechanical part of this build:

| # | site | change |
|---|---|---|
| 1 | `ADP:158-161` `mode_is_stream` | `+ 10, 11` — gates `run_service` entry (`KRN:1414`) and M8's dynamic ticket (`KRN:1682`) |
| 2 | `ADP:209-211` `mode_is_direct_accum` | `+ 10, 11` — gates the epilogue peer table (`P2:249-250`) and M8's `Zero=true` instantiation (`KRN:1782`) |
| 3 | `KRN:319-320` enqueue predicate in `k0p6_mps_task_done` | `+ 10, 11` |
| 4 | `KRN:350` `if (m != 12 && m != 13)` in `k0p6_mps_task_drain` | widen to the family |
| 5 | `KRN:371` `if (m == 12 \|\| m == 13)` in `..._maybe_defer` | widen |
| 6 | **`ADP:787` `if (env.mode == kModeRemoteAccum)` in `run_service`** | widen — **this one is an exact `== 12`. Miss it and the pool tries to push payload out of `part`, which mode 12 never writes, over the correctly-accumulated slot rows.** |
| 7 | `ADP:233-235` `effective_flush_rows` | `+ 10, 11` (see §6.2) |

**Do this by adding one predicate and using it everywhere:**
`mode_is_remote_accum_family(c) { return c.mode >= 10u && c.mode <= 13u; }`,
then express `mode_is_direct_accum` and sites 3-6 in terms of it. That converts
seven independent chances to forget into one.

### 6.2 Where `C6` lives — zero host/ABI change

Follow the exp_20/mode-4 idiom exactly (`ADP:128-135`: "Config-field
reinterpretation, so no descriptor, host-ABI or harness change is needed"):
**for modes 10/11, `flush_rows` carries `C6/4`**, and
`effective_flush_rows` returns the pinned 16 so the pool's flag-batch depth
stays the ratchet's value and is not a free variable.

```
C6 = 4 · flush_rows          flush_rows ∈ [1,64] ⇒ C6 ∈ [4,256]
```

New `config_is_valid` rules for modes 10/11 (all derived in §4.3):

```
C6 % 8 == 0                     (⇒ flush_rows even)      XCD invariant, both pools
C6 >= 32                                                 spin-limit margin ≥ 14×
mode 11:  C6 + C <= 256 − 8                              C7 ≥ 8: one M7 CTA per XCD
mode 10:  C6 + C <= 256
C % 8 == 0                                               already true at C=16
(g & 0xF) == 1, extra bits ⊆ {0x10 detect, 0x20 throttle, 0x40 service-runs-M6}
```

Sweep points: `C6 ∈ {224, 192, 160, 128, 96, 64, 32}` ⇒
`flush_rows ∈ {56, 48, 40, 32, 24, 16, 8}`. `C6 = 240` (`flush_rows = 60`) is
legal for mode 10 only and is the **null control** — mode 10 at `C6 = 240` must
reproduce the mode-12 ratchet within noise.

**Trap to write into the experiment log now:** the epilogue throttle bit is read
**raw** at `P2:272` (`((m7cfg >> 8) & 0x20)`) with no mode gate and is not a
field of `struct config`. **Every mode-10/11 sweep point must carry `g = 33`**,
or the arm silently runs unthrottled, loses ~500 µs, and looks like the
mechanism failed.

### 6.3 Files and line ranges

**`distributed-kernels/fused_moe/n2_phase1_gm_mps.cpp` — ALREADY EXISTS. Do not
create a second copy.**

Checked while writing this document: exp_26 has already vendored this file from
the same donor (`sha256 1d90b266…`, header lines 1-6 name it) with **eight**
MPS-DELTA sites — an opt-in `N2GM_P1_SCHED_GSCALE` for the `kGM`-scaled
`sched_group_barrier` hints, and four empty sub-phase instrumentation hooks
(`N2GM_P1_TASK_HEAD_HOOK`, `..._KLOOP_ENTER_HOOK`, `..._KLOOP_EXIT_HOOK`,
`..._EPILOGUE_DONE_HOOK`). All eight default to the donor body, so the file is
textually donor-equivalent with nothing defined.

**This removes most of Stage 0's build cost — the vendoring, the provenance
header, the byte-equivalence argument, and the `#include` swap in `KRN` are all
done.** What exp_25 needs is one more delta:

- ***MPS-DELTA (9)*** at the vendored file's **line 232** (donor line 156), which
  is still the donor's hardcoded
  `for (int task = blockIdx.x; task < num_tasks; task += kCTAs)`. Macro-
  parameterise it exactly as `P2:286-294` does for phase 2:

```cpp
#ifndef N2GM_P1_TASK_START
#define N2GM_P1_TASK_START blockIdx.x
#endif
#ifndef N2GM_P1_TASK_STRIDE
#define N2GM_P1_TASK_STRIDE kCTAs
#endif
```

  With the defaults the include stays byte-equivalent to the donor, so exp_26's
  arms are unaffected.

**OWNERSHIP — coordinate before touching it.** `aug10/CLAUDE.md` is explicit:
"Never let two agents write the same mutable file." `n2_phase1_gm_mps.cpp` is
exp_26's file right now. Either (a) exp_26's implementer adds MPS-DELTA (9) as
part of its own commit (it is 6 lines and inert by default), or (b) exp_25 waits
for exp_26 to land and rebases. **(a) is strongly preferred** — it is additive,
it cannot change exp_26's measurement, and it removes a serialisation between the
two experiments.

The other recommended change (§4.2 item 1: make the payload release convergent)
should land in `k0p6_a2_arrive` in `KRN`, which exp_25 owns — **not** in the
vendored body. That keeps the file single-writer.

**`k0pf6gm_device_tile_mps.hip`**

| line(s) | change |
|---|---|
| `:104` | bump `K0P6_MPS_SRC_REV`. **Mandatory** — the vendored `.cpp` arrives via `-I` and is not hashed by the mori JIT (`KRN:93-104`). exp_26 is bumping it too; take whatever value is current at rebase time and bump again. |
| `:227-234` | `k0p6_a2_arrive`: make convergent, `producer_drain_release<agent>()` then tid0's relaxed `fetch_add` (§4.2). |
| `:273-276` | `k0p6_mps_stride` → `nct − C − m6_ctas(cfg)`. Zero for every non-split mode ⇒ bit-identical. |
| `:280-285` | `k0p6_mps_task_start` → `compute_id_of(cfg, bid) − m6_ctas(cfg)`. |
| `:319-320`, `:350`, `:371` | mode-family predicates (§6.1). |
| `:385-400` | define `N2GM_P1_TASK_START`/`N2GM_P1_TASK_STRIDE` in the phase-1 macro block, and `#undef` them after. **`:395` is still `#include "n2_phase1_gm.cpp"` (the donor) as of this writing** — it must become `"n2_phase1_gm_mps.cpp"`. Both files define `n2p6gm_phase1_body` in the same namespace, so **include exactly one**; whichever experiment flips it owns that line. |
| `:1318-1330` | wrap the phase-1 call in `if (in_m6_pool(cfg, bid, nct))`. |
| after `:1344` | **mode 10 only:** the rendezvous — `counted_arrive_release_into` on a fresh `mps_state` word (target `C6`) from the M6 pool, `bounded_poll_relaxed_into<agent>` from everyone else. `CTR:85-90` and `CMPL:104-118` express both. |
| `:1367-1392` | M7 guard: `!is_service_cta(...) && in_m7_pool(...)`. **Change this together with `:280-285`.** Changing only the start leaves CTAs `0..C6−1` computing `task = bid − C6 < 0`, hence `tile = task/16` negative, hence `tile_desc[-8]` → garbage `b0` up to 2²⁸ → `sorted_eid[b0]` far out of bounds. That is the exp_01 `address (nil)` fault class, re-armed. |

**`moe_mps_adapter.cuh`**: `mode_is_remote_accum_family`, `mode_is_m6_split`,
`m6_ctas`, `in_m6_pool`, `in_m7_pool`, `service_runs_m6`; `:158-161`,
`:209-211`, `:233-235`, `:249-286`, `:787`.

**`moe_host_abi.hpp`**: **nothing for Stages 0-1.** `:186-188`
`mps_state_bytes()` grows only when `K0P6_MPS_ST_WORDS` widens for Stage 2's
per-XCD banks — an additive size increase that requires rebuilding the pybind
shim, so it is deliberately deferred.

**`n2_phase2_gm_mps.cpp`**: **no change for Stages 0-1.** `N2GM_TASK_START` /
`N2GM_TASK_STRIDE` are already macros and carry the whole M7-pool change. Stage 2
needs one new hook (`N2GM_P2_TASK_NEXT_HOOK`) because a `for(;;task += STRIDE)`
loop cannot express a ticket claim without abuse; that file is ours to extend.

**No `.node/` mirror was created and no file outside this experiment directory
was touched.**

### 6.4 The resource-tuple risk — treated as the most likely failure

Current mode-12 tuple: **SGPR 106 / VGPR 256 / AGPR 256 / scratch 144 B /
LDS 155,496 B**, one block per CU (`exp_21_direct_accumulate/result.md:113`).

**The good news, and it is a real argument, not a hope: an if/else role split
cannot increase the register peak.** Today the two bodies are *sequential* in
one CTA's program (`KRN:1319-1330` then `KRN:1371-1382`), so their live ranges
are already disjoint and the allocator already takes `max(peak6, peak7)`. Making
them *alternatives* is also a `max`, not a sum. M6's `acc[3][2][2][4]` = 192
registers and M7's `acc[3][7][2]` = 168 are each born and dead inside their own
body; nothing carries them across. **The two bodies stay in separate scopes for
every policy in §3 except Policy 4.**

The three places where the budget could still move, in order of risk:

1. **The M6 loop's start/stride become descriptor-derived.** The mitigating fact
   is that `k0p6_desc` is **already** a parameter of the phase-1 body
   (`N2_HOOK_CTX_ARG`, `KRN:393`) and is **already** live across the whole M6
   task loop because the DONE hook at `P1:481-488` uses it. **So no new live
   range is created.** Phase 2 already does exactly this (`P2:326-327` calls
   `k0p6_mps_stride` in the loop increment) and it was measured resource-neutral.
2. **Stage 2's ticket.** One 32-bit claimed index per iteration, broadcast
   through **LDS** (not a register), claimed at a loop head that already carries
   a `__syncthreads()` (`KRN:359`). Expected delta ≤ 1 SGPR. But
   `DESIGN_MPS.md:119-125` measured **+24 B/lane for two ticket words at exactly
   this peak**, so this is the empirical danger zone. Losing the affine
   induction variable may also change LICM around the phase-2 pointer set.
3. **Policy 4.** Both bodies' pointer sets live across one loop. This is the
   measured failure mode, not a hypothetical.

**LDS:** +4 to +8 B for a broadcast word, on 8,344 B of headroom
(163,840 − 155,496). Non-issue.

**Gate, before any timing at every stage:** capture
`-Rpass-analysis=kernel-resource-usage`; require ArchVGPR 256 / AGPR 256 exactly,
MFMA census 96 + 84 = 180 unchanged, **zero scratch ops inside either MFMA
K-loop**, and scratch ≤ 144 B. A scratch regression inside an MFMA span is a
stop, not a note.

### 6.5 Staged build order, cheapest first, each separately measurable

| stage | what | new mechanism | separately measures | est. |
|---|---|---|---|---|
| **0** | MPS-DELTA (9) in the **already-vendored** `n2_phase1_gm_mps.cpp`; mode 10 (`C6` + rendezvous); `SRC_REV` bump; the seven predicates; validation | M6 start/stride only | **`a6`/`b6` — F1** and the L0 null | **2–4 h** (was 4–6 before exp_26 did the vendoring) |
| **1** | mode 11: two helper returns + one predicate | the overlap itself | **F2 lower bound** (early-spin cost included) | **1–2 h** |
| **2** | fall-through: per-XCD M7 ticket banks, `ST_WORDS 8→16`, `mps_state_bytes()`, one new phase-2 hook | dynamic M7 claim | the M7 tail (~120 µs) + removing the early spin | **6–10 h** |
| **3** | backlog-driven role switch: per-XCD M6 banks + threshold | self-tuning `C6` | whether the `C6` curve is sharp enough to be worth tuning | **8–12 h** |
| **4** | interleave inside every CTA | the only variant without the §2.1 penalty | the real ceiling of this boundary | **12–20 h**, register-gated |
| **D** | diagnostic arms: DQ2-NaN poison (L8), `part_done == 16` coverage (L9) | — | that the readiness edge is real and the task space is exact | **3 h** |

**Time to a defensible verdict on the boundary (Stage 0 + 1 + D + ladder
L0-L2): 10–14 h. To the full mechanism (through Stage 2 + L3-L7): 22–28 h.**
Measurement time is not the constraint — a screen is ~90 s, a campaign ~18 min.

**Coordination, not just sequencing.** exp_26 currently owns
`n2_phase1_gm_mps.cpp` and has already bumped `K0P6_MPS_SRC_REV` to 24. exp_25's
Stage 0 wants six inert lines in that file and a different `SRC_REV`. Land
MPS-DELTA (9) inside exp_26's commit (it cannot change exp_26's arms, since the
defaults are the donor's `blockIdx.x`/`kCTAs`) and exp_25 becomes a
`KRN` + `ADP`-only change.

**Sequencing dependency:** exp_24 deletes M5's 448 MiB dead `part` zero-fill.
That changes what is resident in the 256 MB Infinity Cache when M6 starts, and
therefore changes `a6`. **If exp_24 lands between Stage 0 and Stage 1, re-run
F1.**

---

## 7. Ablation ladder

Control first, mechanism second, one variable per step, matched CTA counts
throughout. Screens rank against `production` only (`harness_recipe.md` §5);
decisions get campaigns.

| # | arm | matched control | isolates | pre-registered expectation |
|---|---|---|---|---|
| **L0** | mode 10, `C6 = 240`, `C=16 g=33` | mode 12 `C=16 g=33 flush_rows=16` | that the vendored phase-1 file + the descriptor-derived M6 start/stride are a **null** | must reproduce 6,685 (campaign) / 6,703–6,728 (screen) within noise. **A miss here invalidates every later point.** |
| **L1** | mode 10, `C6 ∈ {192,128,96,64,32}` | L0 | **`a6`, `b6` — F1, the decision gate** | `T6(128)/T6(256)` in 1.61–1.85 per `m6_m7_structure.md` §2.5; ≥1.90 kills the experiment |
| **L2** | mode 11 at each `C6` | **mode 10 at the same `C6`** | **the overlap, with both pools' CTA counts matched.** This is the pair mode 0 structurally could not provide | the `C6` curve should have an interior minimum near `C6 ≈ 120–130` |
| **L3** | mode 11, `g` bit `0x40` set (service pool runs M6) | L2 best point | whether 16 CTAs belong in M6 or in early polling | ±150 µs, sign genuinely unknown |
| **L4** | mode 11 + Stage-2 fall-through | L2 best point, static | the M7 tail | −120 µs computed, plus the early-spin removal |
| **L5** | mode 11, pinned poll backoff ∈ {0, 4, 16} units | L4 best | the early-spin interference term | if L5 moves > 100 µs, the pool's poll is a first-order cost of *any* overlap design |
| **L6** | mode 11 at `g = 1` (throttle off) vs `g = 33` | **mode 12 at `g = 1` vs `g = 33`** | **the W1 mechanism itself — F4** | `Δ(mode11) < Δ(mode12) ≈ 500 µs`. Equal differences ⇒ W1 dead |
| **L7** | mode 11 under `K0_SYNTH_ROUTE=skewed_hot` | mode 12, same route | whether the tile-front pipeline survives routing skew | skew widens `num_tiles` per expert and starves the M7 pool more; expect the optimum `C6` to rise |
| **L8** | **DQ2-NaN poison** diagnostic arm at the best point | the same point without poison | **that the readiness edge is real** (`protocol_review.md` §6 item 1) | must pass the nonfinite gate. A NaN in `out` means M7 read A2q/DQ2 before M6's write was visible |
| **L9** | **`part_done == 16` coverage check** arm | — | **that the M7 task space is covered exactly once** (`protocol_review.md` §6 item 3) | `part_done` is already allocated (slot 30), already zeroed in M0 (`KRN:723-726`), and **dead in mode 12** — a free structural gate |

Two ladder rules inherited and re-stated because they were learned the hard way:

- **Mode 0 is not a control for this experiment.** `aug10/STATUS.md:126-133`:
  reserve-but-do-nothing "measures the capacity tax and is structurally blind to
  the dominant cost of the mechanism it is the control for". Mode 10 exists
  precisely to be the non-blind control.
- **`ratio_vs_pf6gm` is unusable at screen resolution** (3.0 % range on the
  `pf6gm_mega` arm's own microseconds). Rank on `ratio_vs_prod`; use a campaign
  before claiming anything against the homogeneous megakernel.

---

## 8. Primitives

Per the standing mandate. This experiment is unusually informative about the
library because it is **the first one in which the M6→M7 readiness edge is
load-bearing at all** — today the poll never spins, so every weakness in that
edge has been latent.

### 8.1 What the design uses

| concern | primitive | header | fits? |
|---|---|---|---|
| role split | *(none)* — static `blockIdx.x` prefix, mirroring `is_service_cta`/`compute_id_of` | `ADP:292-306` | **gap, §8.2 (1)** |
| readiness publish | `HKA::arrive_local_count<true>` → `thread_release<agent>` + `fetch_add_relaxed<agent>` | `SYNC:150-153`, `SYNC:136-140` | partly — **gap, §8.2 (2)** |
| readiness observe | `HKA::poll_local_count<8>` → `bounded_poll_relaxed_into<agent>` | `CMPL:104-118` | yes, but `>=` not `==` — **gap, §8.2 (4)** |
| consumer convergence | `cta_acquire<agent>` via `HKA::acquire_payload_agent` | `SYNC:176-181` | **yes, exactly** |
| payload drain before publish | `producer_drain_release<agent>` — **should be used and is not** | `SYNC:166-174` | **the single most valuable change in this design** |
| Stage-0 rendezvous | `counted_arrive_release_into` + `bounded_poll_relaxed_into` | `CTR:85-90`, `CMPL:104-118` | yes |
| Stage-2/3 tickets | `fetch_add_relaxed<agent>` on per-XCD banks | `SYNC:136-140` | mechanically, yes — **gap, §8.2 (3)** |
| slot lifetime | **none, deliberately** | `LIFE` | **correct absence — a positive finding** |
| peer addressing / transport | untouched; the epilogue already uses `accumulate_peer_bf162` | `PKT:202-211` | n/a |

### 8.2 What is missing, wrong-shaped, or forces an awkward sequence

**(1) There is no zero-register static N-way role partition.** `roles.cuh` offers
exactly one partition: `finish_order_partition` (`ROLES:62-124`), which is
2-way, ticket-based, and **measured too expensive here** (+24 B/lane,
`DESIGN_MPS.md:119-125`). So the operator open-codes a 2-way static split in
`ADP:292-306`, and this design needs a **3-way** one. That is the second time
the same shape has been hand-rolled.

*Proposed additive surface:* `role_partition_static<K>(bid, total_ctas, const
std::uint16_t (&counts)[K])` returning the existing `role_partition` widened
with a `role_id`, computed from hardware uniforms with no atomics and no state,
plus a `dense_role_id` accessor. Every existing caller of
`finish_order_partition` stays bit-identical.

**(2) The group-arrival-plus-payload-visibility primitive still does not exist,
and this is the first experiment where its absence can produce a wrong
answer.** `aug10/STATUS.md:307-312` already flagged it: "`counter.cuh` can
express 'arrive and release' for one counter but has nothing for 'all members of
this group have arrived AND every writer's payload is visible'". Concretely:

- `ROLES::publish_tile_release` (`ROLES:133-138`) publishes a **value**, so it
  cannot express "8 producers must arrive".
- `ROLES::wait_tile_acquire_into` (`ROLES:146-153`) polls for an exact value and
  then does a **thread**-scope acquire — but our consumer has **tid 0 poll and
  all 256 threads read payload**, so it needs a CTA-wide acquire. Using it as-is
  would be a correctness bug.
- `CTR::counted_arrive_into` (`CTR:62-75`) is close on the producer side but its
  RMW is `acq_rel` where the measured-cheapest shape is *explicit release then
  relaxed RMW* (`KRN:229-231` says so in as many words: "the completion RMW
  itself is deliberately relaxed"), and its epoch/ordinal/last bookkeeping is
  dead weight here.

*Proposed additive surface, two functions that together are the whole edge:*

```cpp
// producer, whole-CTA convergent: drain every lane, one release, leader RMW
template<memory_scope Scope = memory_scope::agent>
void counted_arrive_drain_release(std::uint32_t* counter);

// consumer, whole-CTA convergent: leader polls, CTA barrier, CTA acquire
template<memory_scope Scope = memory_scope::agent>
void wait_counted_acquire_cta_into(const std::uint32_t* counter,
                                   std::uint32_t expected,
                                   std::uint64_t spin_limit,
                                   wait_result& result);
```

The first is `producer_drain_release<Scope>()` + `fetch_add_relaxed<Scope>` and
**would have made §4.2 item 1 unaskable**. The second is exactly what
`k0p6_a2_wait` (`KRN:236-251`) open-codes today. Both are additive; no existing
caller changes.

**(3) Nothing in the library knows about XCDs, and every role-split experiment
keeps re-deriving it.** `xcd = bid % 8` is hardcoded in the operator adapter
(`ADP:93-96`), `kXcds` and `kCtasPerXcd` are operator constants (`ADP:107-108`),
`P2:77`'s `static_assert` encodes an XCD invariant in the GEMM body, and mode 3
open-codes XCD-residue role selection (`ADP:299-306`). Stage 2 needs it a fifth
time, as "claim a ticket from my XCD's bank, striding by `kXcds`, so a residue
invariant survives dynamic claiming".

*Proposed additive surface in a new `include/cdna4/ops/group/distributed/xcd.cuh`
(or `counter.cuh`):* `xcd_id()` (documented as `bid % kXcds`, with the
`s_getreg_b32 hwreg(HW_REG_XCC_ID, 0, 4)` confirmation path noted), `kXcds`,
and `xcd_banked_ticket(std::uint32_t* banks, std::uint32_t stride)`. This is the
one genuinely CDNA4-specific primitive the whole line of work is missing, and it
is what makes dynamic claiming compatible with L2 locality instead of opposed to
it.

**(4) `completion.cuh` has no exact-match bounded poll.** `epoch_ready`
(`CMPL:57-63`) is `observed >= expected`, which is right for monotonic epochs
and for counters that can only reach their target — but it silently tolerates
**over**-counting, which is exactly failure mode 3 in `protocol_review.md` §6
(a doubly-executed M7 task, whose numerical signature is ~6 × 10⁻⁵ relative
error and passes every gate). *Proposed:* `bounded_poll_exact_into`, and use it
for fixed-fan-in arrival gates like `a2_done`'s 8.

### 8.3 Verdict on the mandate

Four gaps, three of them additive one-function changes, and one of them
(`producer_drain_release` at the M6 arrival) is a **correctness** improvement
that the library already contains and the operator simply does not call. That is
the mandate working as intended: the primitive existed, reaching for it would
have been right, and open-coding `__syncthreads()` + a tid0 fence instead is a
defect even though it has never yet produced a wrong answer.

The one point against the mandate: `finish_order_partition` is the only
role-split primitive the library has, and it is **unusable here on measured
resource grounds**. A primitive that cannot be afforded at the peak is not a
primitive. Logging that under `primitives:` in `LESSONS.md`.

---

## 9. Open items — what would change this design

1. **`a6` (M6's CTA scaling) is unmeasured and it decides the experiment.** F1.
2. **Whether `fence(RELEASE,"agent")` emits an L2 writeback on gfx950** is not
   determinable from source. If it does not, §4.2 item 1 is a real hole and the
   `producer_drain_release` change is mandatory, not recommended.
   `protocol_review.md` §7 check C1.
3. **`num_tiles` and `nvi[0]` are still never printed** (`m6_m7_structure.md`
   §7 item 1). Every task/tile count here scales linearly in
   `num_tiles ≈ 355 (349–380)`; the balance point `C6` inherits that ±4 %.
4. **The per-XCD L2 co-residency penalty (§2.4) is unquantifiable from here** and
   is the largest unmodelled downside. exp_08 is the precedent that says it is
   worth hundreds of µs. L2's `C6` curve will show it as a shallower-than-
   predicted minimum.
5. **Two other live experiments change `a6`, and F1 must be re-run after each.**
   exp_24 deletes M5's 448 MiB dead `part` zero-fill, changing what is resident
   in the Infinity Cache when M6 starts. exp_26's `N2GM_P1_SCHED_GSCALE` changes
   M6's K-loop schedule directly, which moves both `a6` and `b6`. **`a6` is a
   property of the tree, not of the hardware**, so F1's threshold (1.80/1.90) has
   to be re-evaluated on whatever tree Stage 1 is built on. This is an argument
   for running F1 *last* among the three, on the winning tree, rather than first
   in wall-clock order.
6. **Policy 4 is where the ceiling is**, and this design does not price it. If
   Stages 0-2 confirm the boundary, the next design document should be Policy 4
   with a register budget worked out in advance rather than measured after.

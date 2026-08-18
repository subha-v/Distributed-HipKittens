# FILL_AWARE_DESIGN — making the M15 prefill megakernel cost scale with `n_orig`

**Written:** 2026-08-18, laptop, no GPU. **Rev 2:** 2026-08-18, after three adversarial reviews
(parity / hardware / measurement). **Status:** design only; nothing built, nothing run.
**Codename:** M24. **Author role:** kernel architect (nightshift priority 1).

> **Read Appendix R first if you reviewed rev 1.** Rev 2 fixes 5 blocking and 16 major findings.
> The four that matter most: (1) the staleness handshake was **inert and failing open** —
> `mps_state` is zeroed grid-wide every launch — and is rebuilt on device-owned per-layer state
> with a device-sourced payload and a single-reader broadcast; (2) the **bit-identity guarantee
> is withdrawn** — the mode-12 epilogue accumulates several experts into one slot row with bf16
> atomics, so this kernel is not bit-reproducible at any `T`; (3) the MoK ladder gains an
> **inert-arm arbiter (R0b)**, a live poison detector, a heterogeneous-fill arm and a C re-sweep,
> and loses a control that was flat by construction; (4) the headline falls from **+31 %** to
> **≈ +22 % vs the secondary baseline, UNKNOWN vs native production** — so M24 alone may not
> clear the 20–30 % target, and §H.9's step-count defect is promoted to a co-equal workstream.

**Binding context:** `nightshift/CLAUDE.md`, `M23_RAGGED_SEAL_DESIGN.md`, `DECOMP_RUNBOOK.md`,
`BOTTLENECK_SIGNALS.md`, `nightshift/CORPUS_FINDINGS.md`, `docs/distributed/OVERLAP_ABSTRACTIONS.md`.

---

## 0. Ground truth: which file is the serving M15 body

**Ground truth = `/Users/subha/repos/Distributed-HipKittens/distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip`
(1,995 lines, branch `ablations`, HEAD `b163f93e`).** Every `KERNEL:nnn` citation in this
document is a line number in that file.

Evidence, five independent strands:

1. **Line-number congruence with M23.** `M23_RAGGED_SEAL_DESIGN.md` §3.4 cites `KERNEL:977`,
   `993-1018`, `1037`, `1891`, `1899`. In this file, line 977 is
   `const int T = (int)k0p6_dread(desc, K0P6_D_T);`, 993 is `for (int tau = gw; tau < T; tau += nw)`,
   1037 is `if (pos >= (unsigned int)MAXTOK) atomicOr(pperr, 65536);`, 1891 is M8's
   `const int T = payload_ok ? (int)k0p6_dread(desc, K0P6_D_T) : 0;`, 1899 is
   `const int nbatches = (T + NT - 1) / NT;`. All five match exactly. The M23 author — who was
   reading the deployed serving body — was reading this file.
2. **The pin trap resolves to it.** `DECOMP_RUNBOOK.md:92-116`: the MoK harness compiles
   `k0pf6gm_device_tile_mps.hip`, which on `ablations` is the *old MPS sibling* (mode ≤ 2, the
   0.898× donor); the M15 chassis reaches the harness only through a RUN PIN worktree that
   prepends `#define K0P6_M15_KERNEL_NAME k0pf6gm_mps_mega` and inlines
   `k0pf6gm_device_tile_m15.hip` **verbatim** (not `#include` — the mori JIT content hash
   ignores `-I` paths). KERNEL:284-289 is that parametric-name hook.
3. **All named markers are present and only here.** Single-CTA planner (`if (bid == 0)` …
   `if (tid == 0)` tile loop, KERNEL:1395-1420); reserved service CTAs
   (`hk_moe::mps::is_service_cta(mrole, bid, nct)`, KERNEL:1541); `chunk_ready`
   (`K0P6_D_CHUNK_READY = 50`, published KERNEL:1167-1175, consumed KERNEL:1235-1241);
   M8 certification (the 2×world slab-word polls, KERNEL:1878-1887); slab rendezvous
   (KERNEL:1694-1730).
4. **Git divergence.** `k0pf6gm_device_tile_mps.hip` was last touched at `88172879` (aug 13);
   `k0pf6gm_device_tile_m15.hip` carries every subsequent serving arm — m17 `3609fd1a`,
   m18 `0f9676ce`, m19 `0430aeb7`, m20 `ac01a53b`/`3c437b24`, t2b `95f2f339`. The live line
   is the m15 file.
5. **The fabric worktree confirms the base.** `.claude/worktrees/wf_b38dfe27-d09-2/…/k0pf6gm_device_tile_m15.hip`
   is byte-identical to HEAD's copy plus a single inserted block (the M21
   `K0P6_M15_POLL_BACKOFF` / `RMW_INTERLEAVE` arms, 9 hits). So HEAD's file is the base body
   the fabric arms were derived from, not a stale sibling.

**Discrepancy flagged (code wins over the banked description).** The brief and CLAUDE.md say
"C = 28 reserved service CTAs" as if it were a kernel constant. It is not. `C` is
`config.reserved_comm_ctas`, the **low 8 bits of the runtime descriptor word `K0P6_D_MPS_CFG`**
(`moe_mps_adapter.cuh:206`, consumed at `k0p6_mps_stride` KERNEL:349-352 and
`is_service_cta` `moe_mps_adapter.cuh:678-684`). `C=28` is the *measurement configuration* of
the banked 0.7556 / 0.8467 pair, not a compile-time property. This matters: it means C can be
made fill-aware at runtime without a recompile (§B.6), and it means any claim that "C=28 is a
fixed cost" must be qualified as "C=28 is what the descriptor currently carries".

**Second discrepancy flagged.** The brief's premise — *"Production's padded GEMMs do partial-tile
work, so production DOES scale with fill"* — is **contradicted by the evidence**.
`M23_RAGGED_SEAL_DESIGN.md` §3.2 proves link-by-link that stock dispatches and computes **all
4096 padded rows**: the router emits ordinary top-k for pad rows (`moe_runner.py:573-586`),
Mori dispatches `a1` unsliced `[4096, 7168]` (`MORI:89-95`), AITER's GEMM bound is
`M = a1q.size(0)` over post-dispatch receive counts *which include the pad rows*
(`rocm_aiter_moe.py:526-546`, `MK:802-805`), and combine writes all 4096 rows back
(`MORI:118-124`). The one masking site, `MK:1137-1151`, is gated on `VLLM_MOE_SKIP_PADDING`,
which defaults to `0` (`envs.py:191`, `envs.py:1499`). **Production does not scale with fill
either.** Fill-awareness is therefore not catch-up — it is a *pure structural advantage the
megakernel can take and the captured production graph cannot*. This is strictly better news
than the brief assumed, and it carries a fairness obligation (§H.1).

---

## A. Cost model — what is fixed, what scales, how much is recoverable

### A.1 The measured regime

| quantity | value | source |
|---|---|---|
| real tokens / rank / sealed step | 1,539 | `DECOMP_RUNBOOK.md:36` (recomputed from `seal_receipts.txt`) |
| rows the mega executes / rank / sealed step | 4,096 | `m15_contracts.py:537-538` `tokens_per_rank=4096`, written to slot T once, out of capture (`m15_runtime.py:941`) |
| fill | **37.6 %**, padding multiplier **2.66×** | `DECOMP_RUNBOOK.md:36-37` |
| share of all MoE rows the m15 arm computed that were padding | **62.4 %** (stock 50.8 %) | `BOTTLENECK_SIGNALS.md` §0 |
| wall attributable to padded MoE rows, m15 arm | **≈ 65 s of 217.6 s** | `BOTTLENECK_SIGNALS.md` §0 |
| mega's padded-step MoE region ratio vs stock | **r ≈ 0.95, [0.78, 1.06]** (three independent estimators) | `BOTTLENECK_SIGNALS.md` §0, §6.1–6.3, §7 |
| route-degenerate rank-calls (**proxy, NOT `n_orig = 0`**) | 31 % (160/512) | `CORPUS_FINDINGS.md` §1 — see the ⚠ below |
| MoE fraction of step time | f ≈ 0.40–0.54 | `DECOMP_RUNBOOK.md:19` |
| m15 arm's padded-step-count inflation vs stock | **κ = 1.192×** (249.1 vs 205.9 B4096 steps / 300) | `BOTTLENECK_SIGNALS.md` §0 — priced into §A.4 |

**⚠ The 31 % row is a proxy and must not be used as `n_orig = 0`.** `CORPUS_FINDINGS.md` §1
classifies a call as degenerate from *routing homogeneity* (> 95 % of rows share one expert set,
across-row weight STD ≈ 0.008) and from its own "DP-dummy / pad-dominated" label — it never
measured `n_orig`, it never established that degenerate calls are step-*synchronised* across the
8 ranks, and its capture is the first 64 B4096 calls per worker (early-run bias, stated in its
own caveat). A uniform-decode step routed through the M23 rescue looks degenerate and has
`n_orig` = the number of running requests (e.g. 32), not 0. **Work item G0b replaces this row
with a direct measurement**: the seal receipts already carry `in_bucket_sum_orig` per rank-step
(the basis of the 1,539 / 37.6 % figure, `DECOMP_RUNBOOK.md:36-37`), so the true `n_orig`
histogram — including the exact zero-fill fraction and its cross-rank correlation — is
recoverable on the laptop for free. **No Tier-1 sizing in this document may be quoted before
G0b lands.**

The last row of the first block is the one that reframes the whole project: the mega is **not
losing per padded step**. It is paying, per step, for 2.66× the rows it needs — and so is
production. Whoever stops paying first wins by that factor.

### A.2 Region-by-region: does the cost scale with real rows?

All line numbers `KERNEL:` = `k0pf6gm_device_tile_m15.hip`. "Scales" means *scales with real
rows once `T` means real rows*; "fixed" means invariant in fill.

| region | site | driver today | scales? | est. share of 5,823 µs |
|---|---|---|---|---|
| M0 retire poll + counter zero | 1866-919 | `world`, `maxb`, `K0P6_MPS_ST_WORDS` | **fixed** | < 1 % |
| M0.5 replication pre-pass (ADAPTIVE/M20 arms) | 934-969 | `T_pre = input_tokens` (**943**, an *alias* of `desc[44]`) | **does NOT scale today — this is substitution site 5, see §B.3** | 0 in the default build; live in every M18/M19/M20 composition |
| **M1 qpush dispatch** | 972-1119 | `T = desc[44] = 4096` (977, 993) | **scales exactly with T** | large — see A.3 |
| M1 epoch publication | 1121-1178 | `world × nchunks` = 64 words | **fixed** | ~0 |
| M2 rows_done / chunk_ready rendezvous | 1208-1250 | `world × nchunks` polls + 1 grid barrier | **fixed** | small |
| M2 unpack + histogram | 1289-1329 | loop over `T_ext = 32768` with `if (off >= s_ns[s]) continue` (1292) | payload scales; **loop trip count fixed** | small |
| M2 hole-fill (`recv_eid = -1`) | 1337-1363 | `Σ_s (MAXTOK − n_recv_s) × TOPK` | **anti-scales** (grows as fill falls) | ~0.6 MB stores, small |
| M3 count_reduce + scan + cursors | 1396-1398, `bid==0` | `EL × nct` = 8,192 | **fixed** | small |
| **M3 tile planner** | 1400-1419, `bid==0 && tid==0` | serial over `EL` **and over `nt`** | `nt` scales; `EL` loop fixed | part of the 372.8 µs "plan" phase |
| M3/M4 `part` zeroing + scale transpose | 1431-1453 | `T_ext` | **already skipped** — `skip_dead_part_zero` is TRUE at mode 12 with `g=353` (`0x40 & 353 = 64 ≠ 0`, `moe_mps_adapter.cuh:416, 513-529`) | **0 — do not claim it** |
| M4 `k0p6_sort::pad` | 1460 | `EL`, block granularity | mostly fixed, small | small |
| M4 scatter | 1474 (`pairs = T_ext*TOPK = 262,144`, 1459) | **fixed sweep**, live work scales | **fixed sweep**, 1 MB of loads | small |
| `pull_src_fill` | 1477 | `T` | **scales** | small |
| **9 grid barriers** + 2 slab rendezvous | **882, 950, 956, 1246, 1392, 1456, 1479, 1701, 1825** (950/956 = ADAPTIVE M0.5; 1825 = STAGED R2 — all three live in composed arms) | none | **fixed** | few µs each |
| **M6 phase-1 GEMM** | 1494-1509 | `num_tiles` from real received block counts | **scales** | 2,453 µs (42 %) |
| **M7 phase-2 GEMM + epilogue RMW transport** | 1532-1731 | `ntiles × 16 nc` | **scales** | 2,702 µs (46 %) |
| C reserved service CTAs | 1541 | `config.reserved_comm_ctas` | **fixed capacity tax**; the pool's *useful* consuming work scales | 10.9 % of 256 CTAs |
| M7 pool front-half sweep | 1565-1627 | `T` via `nbatches` (1600) + `quota` | **scales** | inside M7 |
| **M8 combine** | 1874-1953 | `nbatches = (T+3)/4` (1899); each live token reads `fanout × 14 KiB` of slot bytes | **scales exactly with T** | 324 µs (5.5 %) |
| M8 slab certification | 1878-1887 | `2 × world` = 16 bounded polls, `tid==0` | **fixed** | ~0 |
| M9 retire + publish | 1955-1994 | `world` | **fixed** | ~0 |

Phase shares from the only banked phase ledger, `overnight/aug11/exp_33_attribution/result.md:5`
(plan 372.79 / M6 2,453.30 / M7 2,701.84 / combine 324.21 µs = 5,852 µs). **⚠** that ledger is
balanced-route, C=16, old mps ratchet — `DECOMP_RUNBOOK.md:133-139` explicitly refuses it as an
R3 baseline. I use it only for a *fractional* split, and R3a is the work item that replaces it.

### A.3 The two dominant fill-scaling quantities

**Dispatch bytes (M1).** Each dispatched `(row, dest)` pair moves `K0P5_ROW_STRIDE` bytes —
7,168 fp8 payload + 7 fp32 scales + the eid/wgt tail ≈ 7.5 KiB (KERNEL:1047-1117). Dedup of
top-8 experts over 8 ranks gives a mean fanout of ≈ 5.2 unique destinations per row
(KERNEL:1020-1023). At T = 4096 that is **≈ 160 MB of xGMI injection per rank per layer**; at
n_orig = 1,539 it is **≈ 60 MB**. This is the traffic the depth-4 `vmcnt` throttle exists to
bound (`OVERLAP_ABSTRACTIONS.md` §3 K4: depth 4 = −615 µs, "the single largest knob"). Cutting
it 2.66× is the largest single-mechanism reduction in this design.

**Combine HBM reads (M8).** `k0p6_m15_m8_batch` (KERNEL:652-746) reads, per live token,
`fanout` slot rows of `K0P6_H × 2 = 14,336` bytes and writes one 14 KiB output row. At
T = 4096, fanout 5.2: **≈ 300 MB read + 59 MB written per rank per layer**. The combine is
measured latency-bound at 1.57 TB/s ≈ 20 % of HBM (exp_29, `OVERLAP_ABSTRACTIONS.md` §1), so
this is ≈ 190 µs of the 324 µs ledger entry. It scales exactly with `T`.

**There are no "empty GEMM tiles" to skip.** This corrects the brief's Tier-2 framing. The tile
descriptor is built from `scratch[K0P6_SC_ERB + e]`, the *actual per-expert live row counts*
after scatter (KERNEL:1406-1416), so the GEMM tiling is already dense over received rows. The
only slack is `k0p6_sort::pad`'s round-up of each expert block to a multiple of 32 (≤ 31 pad
rows × EL = 32 experts ≈ 992 rows), which is 3 % of the live rows at 100 % fill and 8 % at
37.6 % fill. **The padding enters the kernel upstream, as 2,557 genuinely-dispatched garbage
rows.** Killing them at M1 is the entire fix; nothing inside the GEMM needs an early-exit.

### A.4 Recoverable time — arithmetic and assumptions

**Rev 2 (post-review). The earlier version of this section mixed two denominators and quoted a
table that did not reproduce from its own formula. Both are corrected here, and the honest
headline is materially lower than the first draft's +31 %.**

Let `φ` = the fill-scaling fraction of the mega's MoE-region cost and `f` = fill.
Region cost ratio after the fix: `ρ(f) = (1 − φ) + φ·f`.

From A.2, the fixed set is: M0, M1's publication, M2's rendezvous and loop-trip overhead, the
`EL` half of the planner, count_reduce, the scatter sweep, `k0p6_sort::pad`, 9 barriers, M8's
certification, M9. The exp_33 ledger sums to **5,852.14 µs** (372.79 + 2,453.30 + 2,701.84 +
324.21) — that is the denominator for every fractional share in §A.2, *not* the 5,823 µs M15
MoK figure, which is a different measurement of a different configuration. Charging **all** of
the 372.79 µs plan phase (6.37 %) plus 2 % of M6+M7 (1.76 %) to the fixed set gives
fixed = 8.13 %, i.e. **φ = 0.919**. I round the central estimate **down to φ = 0.90** as a
deliberate conservatism; φ ≈ 0.935 optimistic; φ ≈ 0.80 pessimistic.

* At f = 0.376: `ρ = 0.44` central, `0.42` optimistic, `0.50` pessimistic.
  At the `TGRAIN = 256` effective fill of 0.438: ρ = 0.494 central. **§F.4's predictions use the
  TGRAIN-rounded value; §A.4's table uses the raw fill and is therefore the optimistic bound.**
  → the mega does **2.0–2.4× less MoE work per sealed step**.

#### A.4.1 The frame error, named so it is not repeated

`r ≈ 0.95` (`BOTTLENECK_SIGNALS.md` §7) is a **per-step, composition-corrected** MoE-region
ratio: it already divides out the fact that the m15 arm ran **κ = 1.192× the MoE row-work per
real token delivered**, because it ran the padded B4096 graph on 249.1 of 300 steps against
stock's 205.9. **`r` therefore cannot be substituted into a per-run wall model.** The first
draft did exactly that, which is why `1/(1 − f + f·r·ρ)` at ρ = 1 predicts the *current* m15 arm
at +2.3 % vs stock when the measured value is **−7.84 %** (wall 217.577 s vs 200.512 s, tok/s
ratio 0.9216, `BOTTLENECK_SIGNALS.md` §1; the same pair is recorded as m15 −7.8 % at
`DECOMP_RUNBOOK.md:16-17`). A model that is 10 points wrong at its only checkable point cannot
carry a 3-significant-figure headline.

**The corrected model works entirely inside the m15 arm's own frame and divides by the baseline
once, at the end.** M24 changes only the MoE region of the m15 arm; it does not touch κ.

```
wall_m24 = wall_m15_today · ( (1 − f_moe) + f_moe · ρ )
headline = wall_baseline / wall_m24 − 1
```

At `wall_m15_today = 217.577 s`:

| f_moe | ρ = 0.42 | ρ = 0.44 | ρ = 0.50 |
|---|---|---|---|
| 0.40 | 166.4 s | 168.2 s | 173.3 s |
| 0.44 | 162.4 s | 164.4 s | 170.1 s |
| 0.54 | 152.6 s | 155.0 s | 162.0 s |

Against **rescued/patched-stock's measured 200.512 s wall**:

| f_moe | ρ = 0.42 | ρ = 0.44 | ρ = 0.50 |
|---|---|---|---|
| 0.40 | +20.5 % | +19.2 % | +15.7 % |
| 0.44 | +23.5 % | +22.0 % | +17.9 % |
| 0.54 | +31.4 % | +29.4 % | +23.8 % |

**Honest headline: +16 % to +31 % vs rescued/patched-stock at c32p, central ≈ +22 %.**

**Against NATIVE production the number is UNKNOWN.** No native-production arm has ever been run
at c32p; `patched-stock/native` is unmeasured. CLAUDE.md permits only `m15/native` as a headline,
so **this design cannot state its headline until the native leg is measured** (work item G16
runs all three legs). Anything quoted before then is a `patched-stock` diagnostic and must be
worded as such — this is fairness checklist item 1, the project's named past sin.

**Consequence for the mission gate.** The 20–30 % e2e target is *not* reliably cleared by M24
alone: the central estimate sits at the bottom of the band and only against the secondary
baseline. The **κ = 1.192 step-count composition defect (§H.9) is hereby promoted from
"future work" to a co-equal work item** — the two together are what reaches the target, and
neither alone is a safe promise. If the campaign's node time must be rationed, the honest
ordering is M24 first (it is a kernel change we own end to end) with the composition fix
scoped in parallel by a separate agent.

#### A.4.2 What the "independent cross-check" actually is

`BOTTLENECK_SIGNALS.md` §0 attributes ≈ 65 s of the m15 arm's 217.6 s wall to padded MoE rows
(derived as f ≈ 0.45 × 62.4 % padding share × 217.6 s). Removing the 90 % of it that is
fill-scaling gives 217.6 → 159.1 s, which against **stock's** 200.5 s is **+26 %**, not the
+34 % first drafted (+34 % is 217.6/159.1, an improvement-over-self, silently compared against a
speedup-over-stock). And it is built from the same two inputs — MoE fraction × padding share × φ
— as the table above. **It is an internal consistency check, not independent confirmation, and
this document no longer claims otherwise.**

#### A.4.3 Tier split — deferred to measurement

The first draft sized Tier 1 at +12–17 % e2e from `CORPUS_FINDINGS.md`'s 31 % degenerate-call
figure. That figure is a routing-homogeneity proxy, not an `n_orig` measurement (see the ⚠ in
§A.1), and §B.5's re-analysis shows the receive-side null-work path does **not** fire on a
DP-dummy rank in serving at all. **Tier 1 is therefore not separately sized in this document.**
G0b (free, laptop-only) produces the real `n_orig` histogram from the seal receipts and the
sizing is written then. What survives without measurement is the *mechanism* claim: a rank with
`n_orig = 0` stops **sending**, and its pad rows therefore vanish from all eight ranks' receive,
scatter, GEMM, epilogue and combine work — that follows from `s_ns[s]` being ground truth
(KERNEL:1292) and needs no corpus statistic.

**Assumptions, stated so they can be attacked:** (i) the exp_33 fractional phase split transfers
to the M15 chassis at C=28 — R3a replaces this; (ii) fill in other cells matches c32p's 37.6 % —
false in general, so every cell must report its own fill; (iii) `f_moe` and κ are held at their
measured values while M24 changes only ρ; (iv) the mega's per-row efficiency does not collapse
at 1,539 rows/rank (risk H.2); (v) **arrival skew is charged zero above** — §H.3 now carries an
explicit skew term and a mandatory heterogeneous-fill MoK arm (§F.3 arm D) to measure it, because
a mixed-fill step's critical path is `max_p n_orig[p]`, not the mean.

---

## B. Mechanism design

### B.1 The signal: how `n_orig` reaches the kernel, capture-safely

**The constraint.** `M23_RAGGED_SEAL_DESIGN.md` §6.2: the descriptor is a device table written
**out of capture only** — `_write_descriptor` hard-fails under
`torch.cuda.is_current_stream_capturing()` (`m15_runtime.py:981-990`, error `M15-DESC-010`),
and `bind_dynamic` refuses a per-call rewrite (`M15-DESC-012`, `m15_runtime.py:1083-1099`). So
`n_orig` cannot ride the descriptor words.

**The mechanism (and its precedent). — Rev 2, after review.** Add **one new descriptor slot
holding a *pointer* to a small persistent device buffer, one buffer per layer** (58 × 64 B ≈
3.7 KB total). The pointer is written once, out of capture, like every other slot. The
*contents* are refreshed once per model step by an on-stream **device-side pre-op** issued
between the router and the first mega launch.

**The payload is device-sourced, not host-sourced.** The first draft specified a pinned-host
staging tensor and a `non_blocking` H2D. Two reviewers independently showed that this is the
dominant real-world staleness mode and that no in-kernel handshake can catch it: with the GPU
several steps behind the CPU, the host can overwrite the pinned bytes for step *k* with step
*k+1*'s vector before the step-*k* SDMA drains, so the device reads a payload from the *future*
— which any `gen > last_gen` test accepts. It also makes the cited `write_decision_bitmap`
precedent inapt, because that op is a **device**-side op over a **device** tensor
(`m15_vllm.py:257-260`) and therefore has no host page to race.

M24 now uses that shape exactly:

* `n_orig[]` is `original_num_tokens_across_dp`, **row 0 of the DP all-reduce**
  (`M23_RAGGED_SEAL_DESIGN.md:114-115`, `DPU:131-137`, `DPU:161`). The all-reduce **staging
  tensor is device-resident** (`DPU:47`); only the *padding result* is the CPU tensor M23
  audited as freshly-allocated and pointer-unstable (`DPU:83-89`). M24 consumes the device
  staging row, never the CPU tensor.
* `write_fill_vector` is a **one-block device kernel** on the current stream that reads that
  device row, packs the header, computes the checksum, and broadcasts the 64-byte record into
  all 58 per-layer buffers. ≈ 2 µs, one kernel node under capture, replays correctly by
  construction. **Zero host-resident payload, zero new communication, no pinned ring, no
  event discipline needed.**
* **Fallback path, only if G13a finds the device staging row unreachable in the deployed
  tree:** a host-sourced copy is permitted *only* with (i) a ring of ≥ 8 pinned staging slots,
  (ii) a `torch.cuda.Event` recorded after each copy and awaited before that slot is reused, and
  (iii) `K0P6_M24_STRICT = 1` for the whole first serving arm. The handshake below does **not**
  by itself make a single reused pinned buffer safe, and this document no longer implies it does.

**Buffer layout, rev 2** (16-byte aligned, `32 + 4·world` host-written bytes + a 16-byte
**device-owned tail the host copy never covers**):

```
  --- written by the pre-op (device), read by the mega ---
u32[0]  magic 0x4D323446  "M24F"
u32[1]  gen               monotonic per mega-eligible model step
u32[2]  flags             bit0 = fill-aware requested this step
u32[3]  csum              xor of words [0..2] and [8..8+world), seeded with magic
u32[4]  gen_echo          duplicate of gen, packed LAST — tear detector
u32[5..7] reserved (zero)
u32[8 + p]  n_orig[p]     p in [0, world)
  --- device-owned tail: NEVER written by the host or by the pre-op ---
u32[8 + world + 0]  last_seen_gen    per-layer consume-once token
u32[8 + world + 1]  agreed_T_eff     the grid-uniform broadcast (§B.2)
u32[8 + world + 2]  agreed_flags     bit0 use_fill, bits 1..3 reject reason
u32[8 + world + 3]  reserved
```

### B.2 The staleness handshake and the grid-uniformity broadcast — Rev 2

**What the first draft got wrong, and why it was blocking.** It placed `last_gen` in the mega's
own `mps_state` scalar block. That is impossible, for three independent reasons, all verified in
the tree:

1. **The block has no slack.** `K0P6_MPS_ST_WORDS = 8` and `K0P6_MPS_TS_COUNT = 8`
   (`moe_mps_adapter.cuh:95, 109`); the host allocates exactly
   `M15_MPS_STATE_U32_WORDS = 8 + 2*8 = 24` u32 as 12 int64 = 96 bytes
   (`m15_contracts.py:468, 960-967`). A 25th word is an out-of-bounds write.
2. **Bumping `ST_WORDS` shifts the `[MPS TS]` phase ledger.** The timestamp base is
   `mps_state + K0P6_MPS_ST_WORDS` (KERNEL:1256, 1486, 1516, 1738, 1897) while the host decodes
   12 fixed int64 slots — every phase number becomes garbage, destroying the exact instrument
   R3a needs.
3. **M0 zeroes the whole block on every launch, and the store races the zeroing.** KERNEL:893-901
   is a grid-strided `mps_state[i] = 0u` over `i < 24`, placed **after** the barrier at
   KERNEL:882, with only a block-local `__syncthreads()` at KERNEL:932 before the next phase.
   A post-barrier `bid==0 && tid==0` store lands in the same window as CTA 37's zeroing pass, so
   `last_gen` is nondeterministically `gen` or `0` — and in the common case `0`, making
   `use_fill = (gen > last_gen)` unconditionally **true**. The guard would have failed **open**,
   the exact inverse of the fail-closed property claimed. That is a silent-wrong-output path and
   it is the single most serious defect the review found.

**The fix: no state in `mps_state` at all.** `last_seen_gen` lives in the **device-owned tail of
M24's own per-layer buffer**, which nothing in M0 touches, nothing in the host copy covers, and
which survives across launches. The handshake becomes:

* **Pre-barrier, `bid == 0 && tid == 0` only** — a *single reader* of the racing buffer:
  read the header with `hk_moe::load_relaxed<hk_moe::scope::system>` (**not** `k0p6_dread`:
  that is a bare unscoped `volatile` load, KERNEL:308-310, and the kernel's own per-step-mutated
  cell deliberately uses a scoped relaxed load instead — `k0p6_epoch`, KERNEL:318-323).
  Accept iff **all** of: `magic` matches; `gen_echo == gen` (tear); `csum` matches; `flags` bit0
  set; `gen > last_seen_gen`; and every `n_orig[p] ∈ [0, MAXTOK]`. Then compute
  `T_eff_candidate`, **clamp it to `[0, desc[K0P6_D_T]]`** (§B.3), and
  `publish_value<scope::agent>(&tail.agreed_T_eff, T_eff)` plus `agreed_flags`.
  On reject: publish `agreed_T_eff = desc[K0P6_D_T]`, set the reject reason in `agreed_flags`,
  raise a **non-fatal** telemetry bit in `pperr` (or a hard fail under `K0P6_M24_STRICT`).
* **The existing M0 grid barrier at KERNEL:882 carries the value.** It is already followed by
  `hk_moe::acquire_payload_agent()` (KERNEL:883), so a pre-barrier single-writer publish is
  visible to every CTA afterwards. **No new barrier, no new poll, no new cross-rank word.**
* **Post-barrier**, every consuming site does its own `load_relaxed<scope::agent>` of
  `agreed_T_eff`. `T_eff` is therefore **grid-uniform by construction from a single read of the
  buffer** — not from the hope that 256 CTAs racing an asynchronous writer all see the same
  bytes. This closes the divergent-`T_eff` hole (grid-strided M1 producing a *hole* pattern
  rather than a prefix, `csr_scan_block256` scanning a different `T` than M8 consumes, and
  divergent `nbatches` on the shared `m8_front` ticket cursor at KERNEL:1598/1904) at its root.
* **Post-barrier**, `bid == 0 && tid == 0` stores `last_seen_gen = gen` — into M24's own tail,
  so there is **no race with M0's zeroing loop**, which touches only `mps_state`.

**Why `gen > last_seen_gen` is now sufficient.** With a device-sourced payload there is no host
page that can deliver a *future* vector: the pre-op is a stream-ordered kernel node, so the
value the mega reads is the one packed for the replay it belongs to. `gen` then guards only the
remaining case — a step where the pre-op did not run at all (an eager fall-through, an
exception before the seal-predicate site, the M23 uniform rescue taking a different path) — and
in that case the buffer still carries the *previous* step's `gen`, which is **not** greater than
`last_seen_gen`, so the payload is rejected and `T_eff` falls back to `desc[44]` = today's exact
behaviour. **Degradation is toward more work, never toward wrong output.**

**Belt-and-braces, because this is the one silent-wrong-output class in the design:**

* `K0P6_M24_STRICT = 1` is **mandatory for the entire first serving arm**, not just bring-up:
  any reject becomes a loud `pperr` failure rather than a silent 4,096-row fallback.
* G15's receipt records, per run, `Σ T_eff` against the shim's own independently-recorded
  `Σ n_orig`. A systematic under-dispatch is then visible post-hoc from artifacts alone, with no
  reliance on any in-kernel guard. **The fairness auditor is instructed to check this ratio.**

**Cross-rank divergence is safe — no unanimity is required.** This is the structural property
that makes the whole design cheap, and it deserves its own statement:

> `n_orig` is consumed **only** as (i) this rank's M1 send bound and (ii) this rank's M8
> combine bound over its own output rows. The **receive** side is driven entirely by
> `chunk_ready` fills and `rows_done` — ground truth published by each sender about what it
> actually sent (KERNEL:1153-1176 → consumed KERNEL:1214-1241, 1268-1272, 1292).

Therefore if rank A falls back to `T = 4096` while its peers use `n_orig`, A dispatches extra
pad rows, peers accept and compute them (their `s_ns` reflects reality), the epilogue RMWs them
back into A's slots, and A's M8 combines all 4,096 of its own rows. Consistent, no deadlock, no
wrong output — only wasted work on the stale rank. **The megakernel's collective contracts
(`rows_done`, `chunk_ready`, the two slab words, `retired`) are untouched by M24.**

**Cross-rank divergence is safe — no unanimity is required.** This is the structural property
that makes the whole design cheap, and it deserves its own statement:

> `n_orig` is consumed **only** as (i) this rank's M1 send bound and (ii) this rank's M8
> combine bound over its own output rows. The **receive** side is driven entirely by
> `chunk_ready` fills and `rows_done` — ground truth published by each sender about what it
> actually sent (KERNEL:1153-1176 → consumed KERNEL:1214-1241, 1268-1272, 1292).

Therefore if rank A falls back to `T = 4096` while its peers use `n_orig`, A dispatches extra
pad rows, peers accept and compute them (their `s_ns` reflects reality), the epilogue RMWs them
back into A's slots, and A's M8 combines all 4,096 of its own rows. Consistent, no deadlock, no
wrong output — only wasted work on the stale rank. **The megakernel's collective contracts
(`rows_done`, `chunk_ready`, the two slab words, `retired`) are untouched by M24.**

### B.3 The core change: `T` means real rows

**Rev 2: there are FIVE substitution sites, not four**, and `T_eff` must **not** be hoisted into
a single entry-computed register.

| # | site | today | after |
|---|---|---|---|
| 1 | KERNEL:977 → dispatch loop 993 | `T = desc[44]` | `T_eff` |
| 2 | KERNEL:1375 → `csr_scan_block256` 1424, `pull_src_fill` 1477 | `T = desc[44]` | `T_eff` |
| 3 | KERNEL:1587 (M7 pool front-half sweep, `nbatches` 1600) | `T = desc[44]` | `T_eff` |
| 4 | KERNEL:1891 → `nbatches` 1899, `k0p6_m15_m8_batch` live guard 662 | `T = desc[44]` | `T_eff` |
| **5** | **KERNEL:943 `const int T_pre = input_tokens;` → the M0.5 ADAPTIVE histogram loop at 952** | **`input_tokens`, an alias of `desc[44]` bound at KERNEL:759** | **`T_eff`** |

**Site 5 is invisible to a `grep K0P6_D_T`** — the slot is read once at KERNEL:759 into a local
named `input_tokens`, and KERNEL:943 aliases that local. Under any `K0P6_M15_ADAPTIVE` /
`K0P6_M20_SLOTPOOL` composition the M0.5 pre-pass histograms `my_ids` over all 4,096 rows,
including pad rows that will never be dispatched, and the per-chunk replication decision `s_dec`
is then derived from pad routing and published to every peer as the acceptance rule
(KERNEL:1140-1149 → 1298-1311). Correctness survives (both sides use the same bits) but the
M19/M20 mechanism is silently driven by garbage statistics. G3's completion criterion is
corrected accordingly.

**`T_eff` is re-derived at each site, never carried across the kernel.** This is a hard
requirement, not a style preference. The kernel is `__launch_bounds__(256, 1)` (KERNEL:749) and
both N2 GEMM bodies are `__device__ __forceinline__` (KERNEL:392, 418), so their register
allocation is unioned with the caller's. A value live from ~line 880 to line 1917 crosses both
GEMM inner loops at the 256-VGPR occupancy-1 point — that is precisely the whole-function
live-range union that produced the exp_38 spill cliff (`+726.9 µs`, KERNEL:57-59;
`SHARED_EXPERT_FILLER_DESIGN.md:218` states the same law for this file). Today the kernel
deliberately re-reads the descriptor slot at each of the four sites and hoists nothing. M24
does the same:

```c
// at each of the five sites, exactly as k0p6_dread(desc, K0P6_D_T) is done today:
const int T = k0p6_m24_teff(desc);      // one scoped scalar load of tail.agreed_T_eff
```

`k0p6_m24_teff` is a `__device__ __forceinline__` that loads `agreed_T_eff` with
`load_relaxed<scope::agent>` from the M24 buffer pointer, and returns `k0p6_dread(desc,
K0P6_D_T)` unchanged when `K0P6_M24_FILL == 0`. The **arm-build gate (§E) enforces the law**:
any spill delta against the base build fails the arm.

The value published pre-barrier by `bid==0 && tid==0` (§B.2) is:

```c
T_eff = accept ? clamp(round_up_sat(n_orig[cur], K0P6_M24_TGRAIN), 0, desc[K0P6_D_T])
               : desc[K0P6_D_T];
```

**The clamp and the saturating round-up are load-bearing, not defensive noise.** A partially
written or reused buffer yielding `n_orig[cur] = 0x7FFFFF80` would make a plain
`round_up(n, 256)` overflow `int` to a large negative, `min(4096, negative)` = negative, and
every consuming loop (`for (…; tau < T; …)` at KERNEL:993; `(T + NT − 1)/NT` at KERNEL:1899,
1600) would execute zero times: **no output row written at all, `pperr` clean, an entirely stale
`out` tensor and no error bit**. `round_up_sat` saturates at `desc[K0P6_D_T]`, the clamp bounds
below at 0, and any input outside `[0, MAXTOK]` is rejected by the §B.2 validation with a
telemetry bit. `K0P6_M24_STRICT` turns it into a hard fail.

**The capacity site is deliberately NOT changed.** The entry guard's `input_tokens` (KERNEL:759,
tested at 789-790 against `MAXTOK`) is a *capacity* check. It stays on `desc[44] = 4096`, so the
worst-case capacity invariants of `M23_RAGGED_SEAL_DESIGN.md` §3.4 (`t_loc_max = 40,960`,
`pad_max = 263,136`, `pos < MAXTOK` unreachable) remain proven at their original bound. Note
that KERNEL:759's local is *also* site 5's source — G3 must introduce `T_eff` at KERNEL:943
without disturbing the guard's use at 789-790.

Everything downstream then shrinks **without any further code change**, because it is already
driven by real received counts:

* peers' `s_ns[s]` (KERNEL:1268-1272) reflect what was actually pushed → M2 unpack (1289-1329)
  touches only live rows;
* `s_cnt` → `hcnt` → `count_reduce`/`scan` → `scratch[SC_ERB]` → `tile_desc`/`num_tiles`
  (KERNEL:1366, 1396-1418) shrink → **M6 and M7 GEMM tile counts shrink proportionally**;
* the M7 epilogue RMW volume shrinks with the tiles;
* M8 reads `fanout × 14 KiB` for `T_eff` tokens instead of 4,096.

**`K0P6_M24_TGRAIN` (default 256).** `hkp::csr_scan_block256` and `hkp::pull_src_fill` live in
`hkp_sort.hpp` / `hkp_quant.hpp`, which are **outside this repo and read-only to us**
(`moe_mps_adapter.cuh:531-539` says so explicitly). They have only ever been exercised at
`T ∈ {4096, 2048, 1024}` (R6's sweep). Rounding `T_eff` up to a multiple of 256 keeps them in
a benign regime. Work-list step G7 reads both functions on the node and, if they are `T`-general,
lowers the grain to 8 and recovers the rounding loss.

**Rev 2 — the rounding cost is the mean of the roundings, not the rounding of the mean.** The
first draft quoted "≤ 255 extra rows (1,539 → 1,792 = 43.8 % effective fill)", which is correct
only for a rank at the mean. `CORPUS_FINDINGS.md` §1 reports a **bimodal** distribution, and on
the low mode the penalty is multiplicative: a uniform-decode / dummy-dominated rank with
`n_orig = 32` is rounded to `T_eff = 256`, an **8× inflation** of its row work. The honest
statement of the grain cost is therefore `E[round_up(n, 256)] / E[n]` over the measured
histogram, which **G0b computes** alongside the fill histogram. Two consequences: (i) the
effective fill of a real run is worse than 43.8 % and §A.4's raw-fill table is the optimistic
bound; (ii) the F.3 fill ladder must include points **below** the grain (N ∈ {512, 256, 64, 0})
or it cannot surface this at all. If G7 finds the headers are not `T`-general, `TGRAIN = 64` is
the compromise to evaluate before settling on 256.

### B.4 Pad-row output: leave unwritten, or zero cheaply

**Rev 2 — `ZERO_PAD` is MANDATORY in every fill build, enforced by `#error`, not "recommended".**

The first draft left `K0P6_M24_ZERO_PAD` at default 0 with only an `#error` guarding the
*reverse* combination, which made `FILL=1 ZERO_PAD=0` a permitted build — and it is the obvious
minimal first arm, the one G3 produces before G4 lands. It is unsafe in a way that is **new with
M24**: the MoE `output` is `torch.empty_like(hidden_states)` (`MK:1413`,
`M23_RAGGED_SEAL_DESIGN.md:640`), the allocation is *captured*, and replay reuses the captured
pointer. Today M8 rewrites all 4,096 rows with finite garbage every step (KERNEL:736-743). Under
`FILL=1 ZERO_PAD=0`, `out[T_eff, 4096)` is **never written again by anybody**: whatever
uninitialised graph-pool bytes sat there on the first replay stay there forever. If any column
decodes as bf16 NaN, those rows are added to `shared_output` (`moe_runner.py:723`), flow through
58 layers, and permanently break the `max_nonfinite = 0` chokepoint that
`FROZEN_CONTRACT` / `WeightLayoutReceipt.validate` / `CorrectnessReceipt.validate` consume
(`M23_RAGGED_SEAL_DESIGN.md:615-620`) — non-reproducibly, since it depends on what the allocator
left there.

Therefore: **`#error` on `K0P6_M24_FILL && !K0P6_M24_ZERO_PAD && !K0P6_M24_ALLOW_DIRTY_PAD`.**
`ALLOW_DIRTY_PAD` exists only so the adversarial verifier can build the unsafe variant
deliberately; it is never used in a measured arm.

With `ZERO_PAD = 1`, after its batch loops M8 grid-strides `out[T_eff, T_cap)` to zero using
`uint4` stores (same geometry as the consume-and-zero at KERNEL:724-728). **Cost: to be
measured, not asserted.** The first draft's "≈ 7 µs for 33 MB" implies ~4.7 TB/s of achieved
store bandwidth, which contradicts this document's own cited combine measurement of 1.57 TB/s
(20 % of HBM, latency-bound, exp_29 / `OVERLAP_ABSTRACTIONS.md`). At that rate the same 33 MB is
**≈ 21 µs**, and it is issued into the tail of M8 where it contends with the combine's own
~300 MB of reads, so the realistic figure is 21 µs or several times it. G4's done-when is a
**measured** number from the arm-build ladder, and §B.6 no longer rates T2-b's risk as "none".

In exchange the pad region becomes **deterministic**, which removes the NaN-persistence class
entirely and makes the arm's pad rows better behaved than stock's.

### B.5 Tier 1 proper: the two null-work fast paths

**(1a) Local dummy sender.** Falls out of B.3 for free: `n_orig[cur] == 0` ⇒ `T_eff = 0` ⇒ the
M1 `tau` loop (993) has zero iterations, `pushed_count` stays 0, `rows_done` publishes
`(epoch, 0)` and all 8 `chunk_ready` cells publish fill 0 (KERNEL:1167-1175). **All the
collective obligations are still met** — the publication block at KERNEL:1126-1178 is
`tid == 0` on the last-arriving CTA and is *outside* the `tau` loop. Every peer's M2 then skips
this rank's entire 4,096-row segment at `if (off >= s_ns[s]) continue` (1292). It is
cluster-wide: one dummy rank's pad rows disappear from **all eight** ranks' receive, scatter,
GEMM, epilogue and combine work. **This is the whole of Tier 1's real value** — and it is a
*send*-side win, whose size G0b measures (§A.4.3).

**Rev 2 — 1a introduces a NEW exposed rendezvous on exactly the ranks it makes fast, and the
cost model must carry it.** Today a dummy rank spends ~1–2 ms pushing 4,096 pad rows, so it
arrives at M2 in rough lockstep with its peers. Under M24 its M1 `tau` loop (KERNEL:993) has
zero trips, so all 256 of its CTAs enter the M2 rendezvous immediately and spin for the entire
duration of its peers' dispatch: `tid == 0` on all 8 `rows_done` words with only
`s_sleep(4)` backoff (KERNEL:1211-1224), plus every wave polling all 64 `chunk_ready` cells
(KERNEL:1229-1241). Per the banked **mode-14 law**, spinning CTAs invalidate L2 for the very
peer writes they are waiting on — so a dummy rank actively slows the seven ranks it waits on,
and that traffic contends with the depth-4-throttled epilogue RMWs the whole fabric design is
tuned around (`OVERLAP_ABSTRACTIONS.md`: depth 4 = −615 µs). This is the doctrine's own
"sync EXPOSURE is deadly" failure, created by us, on the fast path.

**It is not left as an unpriced risk.** §F.3 adds a **mandatory heterogeneous-fill MoK arm
(arm D)** — the homogeneous `NORIG_CONST` arms set the same constant on all 8 ranks and
therefore have *zero* skew, so they are structurally blind to it. Arm D uses
`K0P6_M24_NORIG_TABLE` (a per-rank table indexed by the runtime `cur`) at
`{4096,0,0,0,0,0,0,0}` and `{3072,1539,1539,1539,0,0,0,0}`. **Pre-committed decision rule:** if
arm D regresses against the homogeneous arm at matched mean fill, or if `[MPS SPIN] fail_max > 0`
appears under captured routes, the M21 `K0P6_M15_POLL_BACKOFF` arm
(`.claude/worktrees/wf_b38dfe27-d09-2`, already build-gate-passed) is **bundled into the first
M24 serving arm** rather than measured separately. §H.3 reverses its earlier recommendation
accordingly.

**(1b) Receive-side null work — keyed on ground truth, not on `n_orig`.**

> **Rev 2 — DEMOTED. `total == 0` is essentially unreachable in serving, so this is a
> diagnostic, not a Tier-1 mechanism, and its number must never be quoted as end-to-end.**
> A DP-dummy rank still **owns 32 experts under EP8** and therefore still receives rows from
> every peer that has real tokens: `total` is the sum over all 8 sources' `rows_done` payloads
> (KERNEL:1211-1226), and `CORPUS_FINDINGS.md` §14-16 shows real traffic concentrating on
> experts {0..7} with p50 max-rank-load 2.61× — every rank receives. On a step with ranks 0-5
> real and 6-7 dummy, rank 6's `total` is in the tens of thousands of rows, `null_work` is
> false, and M2/hole-fill/scatter/M6/M7 all run in full. The **only** configuration where the
> predicate fires is all eight ranks simultaneously idle — which in serving means the
> globally-dummy corner of the M23 uniform-decode rescue, not the 31 % of rank-calls the first
> draft attributed to it. The `NORIG_CONST = 0` MoK arm (§F.4) is therefore a **synthetic upper
> bound with no serving counterpart**; quoting its `< 300 µs` as an end-to-end result would be
> fairness-checklist item 7 (replay-as-e2e), the project's named past sin. It is labelled
> TIMING-ONLY in §F.4 and excluded from every headline.
>
> Also corrected: §C.5's claim that the uniform-decode rescue routes "precisely the `n_orig ≈ 0`
> calls" conflates two different populations. A uniform-decode step has a small but **non-zero**
> count on **every** rank; a DP-dummy rank has `n_orig = 0` while its peers do not. Only the
> first is what the rescue routes, and neither is the 31 % figure.

The mechanism itself is retained because it is cheap and correct. After M2's rendezvous
settles, `total` (KERNEL:1226) is the *actual* number of rows this rank received. Under
`K0P6_M24_NULLWORK`, if `total == 0` **and** `T_eff == 0`, skip the bodies of M2's unpack loop,
the hole-fill, the scatter, `pull_src_fill`, M6, and M7's phase-2 call — but **keep every grid
barrier and every publication** (M3/M4/M5 barriers 1392/1456/1479, the two slab rendezvous and
their word publications 1694-1730, M8's certification polls, M9's `retired` pokes). The
barriers are rank-local and are taken uniformly by the whole grid, so the skip is safe; the
publications are what peers spin on, so they are not optional.

> **Design rule, stated because it is the trap:** the null-work predicate MUST be
> `total == 0` (from `chunk_ready`, ground truth) and never "all peers' `n_orig` are 0"
> (a belief). If a rank with a stale buffer took the slow path and dispatched real rows to a
> peer that had already skipped M6/M7 on a *belief*, those rows would never be computed and the
> stale rank's combine would read zeroed slots — silent wrong output. Ground truth cannot
> disagree with itself.

Residual cost of a **globally**-dummy step under (1b): M0's counter zeroing, 9 grid barriers,
3 rounds of `world`-wide epoch publications, 16 slab polls, M9. Order tens of microseconds
against ~5,800 µs. Production's captured B4096 graph structurally cannot do this — it has no
data-dependent control flow at all. **This sentence describes the globally-idle corner only;
see the demotion box above before quoting any number from it.**

### B.6 Tier 2 extras (each separately gated, ranked by value/risk)

| id | mechanism | site | expected | risk |
|---|---|---|---|---|
| **T2-a** | `T := n_orig` (§B.3) | 977 / 1375 / 1587 / 1891 | **the whole 2.0–2.4×** | low |
| **T2-b** | `K0P6_M24_ZERO_PAD` (§B.4) — **mandatory, not optional** | M8 tail | kills a NaN-persistence class | **low, not "none"**: ≈ 21 µs at the cited 1.57 TB/s, contending with M8's own ~300 MB of reads; G4 measures it |
| **T2-c** | Dense scatter: a local `k0p6_m24_scatter` iterating each source's `[0, s_ns[s])` prefix instead of the full `T_ext × TOPK = 262,144` pair sweep (1459, 1474), which makes the hole-fill (1337-1363) unnecessary | 1337-1363, 1474 | ~1.6 MB of traffic, single-digit µs | low — precedent: `k0p6_m15_scatter_rr` (474-507) is exactly this kind of local replacement for a read-only-header routine |
| **T2-d** | Fill-aware `C`: derive `reserved_comm_ctas` at runtime as `min(C_cfg, g(T_eff))`, since the pool's *consuming* job (front-half combine, quota `flush_rows`) shrinks with `T` while its capacity tax does not | 1541 via `is_service_cta`, 349-359 | up to 10.9 % of grid capacity at low fill | **high** — `compute_id_of`/`k0p6_mps_stride` must stay a dense `[0, nct − C)` numbering (`moe_mps_adapter.cuh:674-693`) or tasks are skipped/duplicated; and `OVERLAP_ABSTRACTIONS.md` §1 shows C is *not* monotone (C=24 beat C=16 by 444 µs). Own macro, own campaign, do not bundle. |
| **T2-e** | M8 granularity at low `T`: `nbatches = (T_eff+3)/4 = 448` at `T_eff = 1792` spread over ~228 compute CTAs × 4 waves = 912 wave-tickets ⇒ most waves get 0–1 batch and the dynamic ticket degenerates. Consider `NT = 2` or static striping below a threshold | 1898-1930 | recovers combine efficiency at low fill | medium — changes the ticket protocol shape |
| **T2-f** | M2 unpack loop over a compacted `(source, offset)` space instead of `T_ext` with `continue` | 1289-1292 | loop-trip overhead only | low, low value |

**M24 adds no exposed barrier, no new cross-rank word, and no new poll — but the claim that it
leaves all five collective knobs undisturbed is WRONG and is withdrawn.**

Per `OVERLAP_ABSTRACTIONS.md` §3, M24 changes **K0 (how much work exists)**. Four of the five
knobs are genuinely untouched: carrier (mode-12 epilogue RMW), producer order (nc-major, S=2),
certification granularity (2 × world slab words), flow control (depth-4 `vmcnt`). **K5 —
consumer placement — is materially perturbed**, in two ways the first draft missed:

* **The pool's front-half sweep is a *quota* sized to fit inside slab-1's compute shadow.** The
  code says so: *"QUOTA, not exhaustion: an unbounded sweep would hold the R1 barrier hostage
  once slab-1 compute finishes … draining all 1,024 batches would outlast slab 1 several times
  over"* (KERNEL:1601-1610). 1,024 is `nbatches` at T = 4096. At `T_eff = 1792` it is **448**,
  while the shadow itself (`t_hi − t_lo` from `ntiles`, KERNEL:1534-1537) shrinks by the same
  factor. So `C = 28` / `flush_rows = 16` — the operating point that produced the banked 0.7556
  — **is no longer the tuned point**, and the pool may drain the cursor long before slab-1
  compute ends and then sit on the R1 barrier.
* **R0a cannot detect this.** R0a validates C=28 at **T = 4096** (`DECOMP_RUNBOOK.md:130-131`).
  It is silent about the fill arm's operating point by construction.

The knob is worth real time: C = 16 beat C = 8 by 93 µs and C = 24 beat C = 16 by another
444 µs (`OVERLAP_ABSTRACTIONS.md`), and "M15: C=24 ≫ C=16 ≫ C=8". **Mitigation, and it is
cheap:** `C` and `flush_rows` are *runtime descriptor* fields, not compile-time constants
(§0's first flagged discrepancy — `K0P6_D_MPS_CFG` low 8 bits, `moe_mps_adapter.cuh:206`,
consumed at KERNEL:349-352). So **§F.3 arm E re-sweeps `C ∈ {16, 24, 28, 32}` × `flush_rows ∈
{8, 16, 32}` at `NORIG_CONST = 1792` with no rebuild at all.** §F.4's falsification thresholds
apply **only after** arm E has found the fill arm's own optimum; reading a high ratio against
them before that would attribute a config artifact to a wrong φ and under-sell or abandon the
design. This is now a blocking ordering constraint in §F.1.

M24 still does not re-attempt any falsified pattern: mode 16's next-launch deferral (+75.8 µs),
the per-row readiness protocol (probability-zero under top-8), and carrier pools (+332–340 µs)
are all untouched. And note that the compile-time **adaptive**-C variant (T2-d) remains
default-0 and out of scope: the banked evidence is monotone *increasing* in C over the measured
range and T2-d is adjacent to the falsified mode-2 dedicated carrier pool. Arm E's runtime sweep
is the evidence that would have to come first, and no adaptive-C code lands before it.

---

## C. Parity argument

### C.1 What must be proven

Sealed outputs for rows `[0, n_orig)` must carry **the same set of addends** as what the same
kernel produces with `T = 4096`, and must differ from it by **no more than the unchanged
kernel's own run-to-run envelope** (rev 2 — see the withdrawal box in §C.2; the first draft
demanded bit-identity, which this kernel cannot provide at any `T`). The changed behaviour of
rows `[n_orig, 4096)` must be **provably invisible downstream**.

### C.2 Rows `[0, n_orig)` carry the same addend set — six-point argument
### (rev 2: clauses 1–5 stand; **clause 6's bit-identity conclusion is WITHDRAWN**, see the box)

1. **No arithmetic changes.** `T_eff` is a loop bound only. `k0p6_m15_m8_batch` (652-746), the
   N2 phase-1/2 bodies (1494, 1542), the FP8 quantizer (1058-1077), the epilogue transport and
   the depth-4 throttle are textually untouched.
2. **Pad rows never influence a real row's dispatch.** Row identity is `tau`; the quantization
   scale is computed **per row** by an intra-row `__shfl_xor` reduction
   (`a = fmaxf(a, fabsf(...))`, then `scale = fmaxf(a, 1e-6f)/448`, KERNEL:1058-1065) — this is
   `M23_RAGGED_SEAL_DESIGN.md` §3.5's first isolation clause, and it holds identically here.
3. **Pad rows never influence a real row's slot.** Reservation is per `(source, dest)` via
   `reserve_row` on `dest_counter` (KERNEL:1029-1038); removing rows only makes destination
   offsets *smaller*. `pos < MAXTOK` remains unreachable (`M23_RAGGED_SEAL_DESIGN.md` §3.4).
   **N.B. the slot *indices* real rows land in do change** (a real row that used to sit behind
   pad rows now sits earlier) — this is why the guarantee below is stated carefully.
4. **The GEMM never mixes rows.** BM32 tiling groups rows but accumulates per output row
   (`M23_RAGGED_SEAL_DESIGN.md` §3.5, clause 2). A row's result depends only on its own `a2q`
   bytes and the expert weights.
5. **Combine sums only same-token partials.** `pull_ptr` / `pull_src` are indexed by `tau`
   (KERNEL:1663-1685, 1892-1893); `acc[t][e]` accumulates the `fanout[t]` slot rows belonging to
   token `tok0 + t` and nothing else.
6. **Therefore the *set* of addends for each real token is unchanged.** `pull_src` entries are
   ordered by the M1 `pmask` scan over `dest` (KERNEL:1102-1117), which is per-`tau` and
   independent of other rows, so the **M8-level sum over destination *ranks* is order-stable**.

> ### ⚠ Rev 2 — THE BIT-IDENTITY GUARANTEE IS FALSE AND IS WITHDRAWN.
>
> Clause 6 above argues only about the M8-level sum over destination *ranks*. It never touches
> the **intra-slot, multi-expert accumulation**, and that is where the order moves.
>
> The M1 dedup (`__match_any_sync` on `dest`, KERNEL:1020-1023) reserves **ONE slot row per
> (token, destination rank)** — `fanout` counts unique *ranks*, not experts. But the scatter
> creates one GEMM entry per live *(row, expert)* pair. So when token `t` routes to experts 3
> and 11 and **both are owned by rank P**, rank P computes two partials and accumulates **both
> into the same slot row** `slots[P][pos]` — via packed **bf16** atomics:
> `global_atomic_pk_add_bf16` / `kittens::distributed::accumulate_peer_bf162`
> (`n2_phase2_gm_mps.cpp:16, 36-37, 236`, mode-12 direct-accumulate slot path at :329-361;
> the slot buffer is `unsigned short*`, KERNEL:1894). Their **arrival order is set by the
> phase-2 tile schedule**, and that schedule is rebuilt every step from the live per-expert row
> counts `scratch[SC_ERB]` → `tile_desc` / `num_tiles` (KERNEL:1406-1416). **M24 changes the
> live counts, therefore the block boundaries, therefore the schedule, therefore the order.**
> An 8-mantissa-bit bf16 accumulate is not associative, and the difference is amplified through
> 58 layers.
>
> **Two consequences, both of which change the gate design:**
>
> 1. **The correct guarantee is: the *set* of addends per real row is unchanged; the low-order
>    bits are not.** Rows `[0, n_orig)` are **tolerance-identical**, at the same tolerance as the
>    kernel's own run-to-run reproducibility — not bit-identical.
> 2. **This kernel is already not bit-reproducible run to run, today, with no M24 in it.** The
>    same schedule nondeterminism exists at T = 4096. That means **no bit-identity gate can ever
>    be built on this kernel**, and it means §H.6's "pre-existing 11.33 % output-token divergence
>    between the current arms" is **not a mystery to be solved before M24 — it is the expected
>    signature of this mechanism.**
>
> **The gate that replaces it (new work item G0a, and it is a prerequisite, not a nicety):**
> measure the **unchanged** m15 arm's run-to-run self-divergence first — same pin, same seed,
> two runs — and use that envelope as the reference. Every downstream accuracy claim, MoK and
> serving alike, is then "M24's divergence from production is within the unchanged kernel's own
> run-to-run envelope", which is a statement the measurement can actually support. Concretely:
> * G9's `K0_MOK_COMPARE_ROWS` gate is a **relative-error** gate over `[0, N)` under the MoK
>   rel-err policy — **never an exact comparison**.
> * The serving exact-token SHA A/B (§F.5, §H.6) is **not** a parity gate for M24 and must not be
>   quoted as one; it is a divergence *measurement* to be read against the G0a envelope.
> * Fairness item 6 (accuracy parity) is satisfied by the rel-err gate + the G0a envelope, and
>   the `FAIRNESS_AUDIT` section must state this explicitly rather than implying bit-identity.
>
> **T2-c (dense scatter) is unaffected by this correction and stays off for the headline run**
> — it moves the order *further*, on top of the baseline nondeterminism, exactly as the m17
> `SCATTER_RR` arm does (KERNEL:474-480).

### C.3 Rows `[n_orig, 4096)` — the finalize contract, verified

**Who consumes the megakernel's output rows beyond `n_orig`? Nobody.** Chain, with sites:

| consumer | site | bound |
|---|---|---|
| the `out` buffer itself | `K0P6_D_OUT`, written by M8 at KERNEL:736-743; type-checked `(4096, 7168)` BF16 contiguous at `m15_vllm.py:291-303` (`M15-FULL-002`) | shape only, never content |
| **shared-expert add** | `moe_runner.py:723` `result = shared_output + fused_output`; the mega's `out` **is** `fused_output` (`SHARED_EXPERT_FILLER_DESIGN.md:97-110`) | elementwise, per row — a garbage pad row stays a garbage pad row |
| residual / next layer | per-row RMSNorm + per-row attention (attention metadata is built **unpadded**, `num_tokens_padded=None`, GMR:4382-4392) | no cross-row reduction anywhere in the stack at TP1 |
| **sampling** | GMR:4491 `hidden_states[logits_indices]`, `logits_indices = query_start_loc[1:] − 1` (GMR:2196-2198) with `query_start_loc[1:num_reqs+1] = cu_num_tokens` (GMR:2033) | **every index < `total_num_scheduled_tokens` = n_orig** |
| pooling | GMR:3382 | `[:num_scheduled_tokens]` |
| prompt logprobs | GMR:3759 | `[:num_scheduled_tokens]` |
| drafter / EAGLE | GMR:5227 / 5244 / 5265 | unpadded |
| KV cache | PIECEWISE: slot mapping sized `num_tokens_unpadded` so the pad **has no slot at all** (GMR:4323 → 4371-4374); FULL: `-1`-filled (GMR:4107-4109); plus `blk_table_tensor[num_reqs:].fill_(NULL_BLOCK_ID)` (GMR:2297-2299) and `seq_lens[num_reqs:].fill_(0)` (GMR:2154) | two independent guards |

(All sites reproduced from `M23_RAGGED_SEAL_DESIGN.md` §5.1, which is itself a
read-the-source audit of the deployed tree.) **No consumer reads a row ≥ `n_orig`.** M23
already relies on this to justify dispatching garbage *into* the pad rows; M24 relies on the
same fact to justify not producing them.

### C.4 Pad-row semantics, before and after

| | today (M23) | after M24 |
|---|---|---|
| pad row's MoE input | stale embedding + attention residue (M23 §3.1) | identical |
| pad row dispatched? | yes, to ~5.2 peers | **no** |
| pad row's expert GEMM | computed on some peer | **not computed anywhere** |
| `out[pad]` content | this step's garbage | previous replay's bytes, or **zero** with `K0P6_M24_ZERO_PAD` |
| `out[pad]` read by any consumer | no (C.3) | no (C.3) |
| non-finite risk | live (M23 §5.3 requires row-restricted receipts) | **strictly lower** with ZERO_PAD |

### C.5 Interaction with M23 and the rescue path

* **M23's seal predicate is untouched.** M24 adds no term to `pf4h_ragged_seal`; it only reads a
  quantity (`original_num_tokens_across_dp`) that the predicate site already holds and which
  M23 explicitly *demoted from gate to telemetry* (`M23_RAGGED_SEAL_DESIGN.md` §4.2, line 436).
  M24 promotes it back — to a **payload**, not a gate. The distinction matters: a gate must be
  DP-unanimous or the collective splits catastrophically (M23 §4.1); a payload consumed only
  send-side-locally has no unanimity requirement at all (§B.2).
* **The uniform-decode rescue** routes ragged/uniform steps through the B4096 graph. **Rev 2:
  these are NOT "precisely the `n_orig ≈ 0` calls".** A uniform-decode step has a small but
  **non-zero** count on **every** rank (e.g. 32 running requests); a DP-dummy rank has
  `n_orig = 0` while its peers are busy. They are different populations and the first draft
  conflated them, which is part of why §B.5(1b) was mis-sized. M24 still makes the rescue much
  cheaper — `T_eff = round_up(32, TGRAIN)` instead of 4,096 — but at `TGRAIN = 256` that is a
  16× reduction, not "nearly free", and the `NULLWORK` predicate does **not** fire on it.
  **M24 is the completion of M23, not a competitor to it.**
* **`RAGGED_SEAL_RECEIPT` (`sealed` / `in_bucket`) is unaffected** — it counts steps, not rows.
  A new `M24_FILL_RECEIPT` records, per step: `gen`, the `n_orig` vector, `T_eff`, and the
  fallback count. The fairness audit needs `Σ T_eff / Σ 4096` as the measured fill of the
  quoted run.
* **`VLLM_MOE_SKIP_PADDING` must stay refused.** M23 §3.6 / edit E6: with it on, `topk_ids = -1`
  reaches KERNEL:1018, `dest = -1`, and KERNEL:1032 does `peer_ptr(..., rank = −1)` — an
  out-of-bounds `heap_bases[-1]` read and a remote atomic into a garbage pointer. M24 makes
  fill-awareness *look* like that flag's job and therefore makes enabling it more tempting.
  **Re-assert the refusal and add it to M24's activation checks.**

---

## D. Macro plan — all default 0

Declared in the `#ifndef` block of `k0pf6gm_device_tile_m15.hip` alongside `K0P6_M15_STAGED`
(68-70), `SCATTER_RR` (81-83), `REPLICATE` (99-101), `ADAPTIVE` (120-122), `M20_SLOTPOOL`
(147-149), `SAVE_Z` (157-159) — same shape, same fail-loud `#error` cross-checks.

| macro | default | gates |
|---|---|---|
| `K0P6_M24_FILL` | **0** | Master. The new descriptor slot, the single-reader fill-buffer read + validation, the `gen`/`last_seen_gen` handshake in M24's **own device-owned tail** (never `mps_state`), the pre-barrier publish of `agreed_T_eff`, and its substitution at the **five** sites KERNEL:943 / 977 / 1375 / 1587 / 1891. Nothing else compiles in without it. |
| `K0P6_M24_TGRAIN` | **256** | Rounding granularity for `T_eff` (§B.3). Inert when `FILL = 0`. Set to 8 only after work-list G7. |
| `K0P6_M24_ZERO_PAD` | **0** | M8's tail zeroing of `out[T_eff, T_cap)` (§B.4). **`#error` in BOTH directions**: set without `FILL`, *and* `FILL` without it unless `ALLOW_DIRTY_PAD` is explicitly set. Effectively mandatory in every fill build. |
| `K0P6_M24_ALLOW_DIRTY_PAD` | **0** | Escape hatch so the adversarial verifier can build the unsafe `FILL=1 ZERO_PAD=0` variant deliberately. **Never used in a measured arm.** |
| `K0P6_M24_NULLWORK` | **0** | The `total == 0` receive-side skip (§B.5 1b). Independent of `FILL` (it reads ground truth). **Diagnostic only — see the demotion box in §B.5; it does not fire in serving.** |
| `K0P6_M24_NORIG_CONST` | **0** | MoK-only payload *source* substitution, so the harness needs no host plumbing (§F.3). **Rev 2: it substitutes the SOURCE only, never the mechanism** — see the box below. |
| `K0P6_M24_NORIG_TABLE` | unset | MoK-only, arm D: a per-rank list (`{4096,0,0,0,0,0,0,0}`) indexed at runtime by `cur`, giving the **heterogeneous-fill** regime the homogeneous arms are structurally blind to (§B.5, §F.3). Mutually exclusive with `NORIG_CONST`. |
| `K0P6_M24_STRICT` | **0** | Turn a rejected/stale payload from "silently use `T = 4096` + telemetry bit" into a hard `pperr` fail-closed. **Mandatory = 1 for the entire first serving arm**, not only bring-up (§B.2). |
| `K0P6_M24_DENSE_SCATTER` | **0** | T2-c (§B.6). **Moves the accumulation order further, on top of the baseline nondeterminism §C.2 documents** — never enabled in an accuracy-gated headline arm. |
| `K0P6_M24_FILL_C` | **0** | T2-d, runtime-adaptive `reserved_comm_ctas`. Own campaign. `#error` without `FILL`. |
| `K0P6_M24_M8_ADAPT` | **0** | T2-e, combine granularity at low `T`. `#error` without `FILL`. |

> ### Rev 2 — `NORIG_CONST` must not compile the mechanism away
>
> The first draft said `NORIG_CONST` "replaces the buffer read" and "short-circuits the whole
> helper". That would make the MoK-measured binary a *different binary from the one that ships*:
> with `T_eff` a compile-time literal the compiler folds `nbatches = 448`, the M1 trip count, the
> `csr_scan_block256` / `pull_src_fill` bounds and the M7 pool sweep into constants, changing
> unrolling, VGPR allocation and scheduling — at occupancy-1 / 256 VGPRs, a one-register shift is
> a cliff. A measured 0.367 would then attribute to fill-awareness a mixture of fill-awareness
> and compile-time specialisation. Worse, it would mean the per-CTA buffer load, the validation,
> the handshake and the grid-uniform broadcast — **the entire highest-risk surface** — are never
> executed under any MoK arm, and first execute on 8 live GPUs inside a serving campaign.
>
> **Corrected semantics.** In a `FILL = 1` build, `T_eff` **always** comes from a runtime
> `load_relaxed` of `tail.agreed_T_eff` (§B.2/§B.3), which the compiler cannot fold across the
> grid barrier. `NORIG_CONST` / `NORIG_TABLE` change only what `bid==0 && tid==0` publishes:
> instead of reading the host-sourced header it publishes `round_up_sat(N, TGRAIN)`, passed
> through an `asm volatile("" : "+s"(n))` opacity barrier. **Everything else — the clamp, the
> publish, the barrier ordering, the five consuming loads, the `ZERO_PAD` tail — is identical
> between the MoK pins and the serving build.** The arm-build gate (§E) additionally requires the
> `NORIG_CONST` pin and the serving-shaped pin to report the **same resource tuple**; a mismatch
> means specialisation leaked and the arm is void.

**Descriptor slot.** `#define K0P6_D_M24_FILL 71` under `FILL`. 71 is above every slot any other
arm claims (STAGED/REPLICATE/SAVE_Z take 63; ADAPTIVE 64; M20_SLOTPOOL 65-70, KERNEL:239-282).
Add a `#error` if any future arm reaches 71.

**Rev 2 — the length macro must be raised, not redefined.** `K0P6_M15_D_LEN` is set by an
**exclusive `#if`/`#elif` cascade** (KERNEL:239-282: STAGED→64, REPLICATE→64, +ADAPTIVE→65,
+SLOTPOOL→71, else `K0P6_MPS_D_LEN` = 63). An unconditional `#define K0P6_M15_D_LEN 72` under
`FILL` is a **macro redefinition** in every one of those branches — a warning at best, a
silently-wrong host-side descriptor length at worst. Write it as a raise, *after* the cascade:

```c
#if K0P6_M24_FILL
#  if K0P6_M15_D_LEN < 72
#    undef K0P6_M15_D_LEN
#    define K0P6_M15_D_LEN 72
#  endif
#  if K0P6_D_M24_FILL >= 72
#    error "M24 fill slot collides with the descriptor length"
#  endif
#endif
```

The default (non-SLOTPOOL) build jumps 63 → 72, leaving slots **63-70 unclaimed**; the host must
**zero-fill** them and the device must never read them. Add that to G1's `#error` cross-checks
and to G12's contract test.

**Rev 2 — do NOT bump the host module constant.** `M15_DESCRIPTOR_WORDS` (`m15_contracts.py:152`)
is the **default `expected_words`** for every descriptor validation (`:1290`), and the
forward-compat branch at `:1314` (`expected_words > M15_DESCRIPTOR_WORDS and words[63] == 0`)
was written for the 63→64/65/71 arm cases. Changing the module constant to 72 changes host
behaviour for **default-0, non-M24 arms** whose kernel still compiles `K0P6_M15_D_LEN = 63` —
which would make the M24 campaign's own control arm no longer byte-for-byte the pre-M24 serving
configuration, **voiding the `m15/patched-stock` leg of the mandated three-way decomposition**.
Instead: add `M15_DESCRIPTOR_WORDS_M24 = 72` as a *separate* constant, pass `expected_words=`
explicitly only on M24 builds, leave `M15_DESCRIPTOR_WORDS = 63`, and extend the `:1314`
compat branch to accept 72 with slots 63-70 zero. The "default build is unchanged" claim of §E
covers `.text` only; this is the host half of the same claim and G12 must prove it.

`K0P6_M15_SRC_REV` (KERNEL:63) is bumped **on the M24 branch only** — it exists to force a
fresh JIT hash, and the default-identity gate compares `.text`, not the source hash. (It is
defined and never referenced, so the bump provably cannot perturb `.text`.)

---

## E. Build gate — proving the default build instruction-identical

Precedent: `fabric_build/` on the node, and exp_38's ratchet restoration
(`275d2c2a`: "default `.text` is BYTE-IDENTICAL to rev 26"). The rule from that experiment is
the reason every macro above is default-0: *reachable-but-unused code near the M7 epilogue is
how the ratchet lost +726.9 µs*.

**Gate procedure (CPU-only, safe while the GPUs are busy):**

1. Build the **pre-M24** body and the **M24 default** body into two pins, identical in every
   respect except the M24 source edits, using the runbook's `mkpin` recipe
   (`DECOMP_RUNBOOK.md` §4.2): each pin's `k0pf6gm_device_tile_mps.hip` = the header stanza
   (`#define K0P6_M15_KERNEL_NAME k0pf6gm_mps_mega`) + `k0pf6gm_device_tile_m15.hip` **inlined
   verbatim**.
2. `hipcc --offload-arch=gfx950 -c` both; capture
   `-Rpass-analysis=kernel-resource-usage` and require an **exactly equal resource tuple**:
   VGPRs, AGPRs, SGPRs, LDS bytes, scratch/spill bytes, occupancy. Any spill delta is an
   automatic fail (occupancy-1, 256-VGPR bodies — `OVERLAP_ABSTRACTIONS.md` §2).
3. `llvm-objdump -d --symbolize-operands` the `k0pf6gm_mps_mega` symbol from both `.hsaco`s and
   require a **byte-identical `.text` for that symbol**. Also compare `.rodata` sizes and the
   kernel descriptor (`kernarg_segment_size`, `group_segment_fixed_size`,
   `private_segment_fixed_size`) from the `.note` metadata.
4. `sha256sum` both `.hsaco`s. They will differ (source hash / `SRC_REV`); that is expected and
   is **not** the gate. The gate is step 3.
5. Record all of it in a `M24_BUILD_GATE_RECEIPT` next to the `fabric_build/` receipts.

**Gate for each arm build** (`FILL=1 ZERO_PAD=1`, `+NULLWORK`, `NORIG_TABLE`, …): resource tuple
only, with an explicit **no-spill** assertion and a diff of the disassembly *size* — a large
`.text` growth near the M7 epilogue is the exp_38 signature and must be investigated before any
GPU time. **Rev 2, two additions:**

* **The no-spill assertion is the enforcement mechanism for §B.3's "never hoist `T_eff`" law.**
  If an implementation carries `T_eff` live across the GEMM bodies instead of re-loading it at
  each of the five sites, this is where it shows up, and it fails the arm.
* **Cross-pin resource-tuple equality**: the `NORIG_CONST` MoK pin and the serving-shaped
  (buffer-reading) pin must report the **same** tuple. A difference means compile-time
  specialisation leaked past the §D opacity barrier and the MoK numbers do not describe the
  shipping binary.

---

## F. MoK gate plan

### F.1 R0a arbiter first — mandatory, blocking

```bash
K0=~/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill
export K0_MOK_ARMS=production,mps_mega
export K0_PF6GM_G=3
export MPSCFG="C=28,g=353,mode=12,flush_rows=16"
env -u K0_MOK_ROUTE_FILE -u K0_MOK_ROUTE_HIST \
  DHK_ROOT=~/DHK-m24-base K0_MPS_CFG="$MPSCFG" \
  bash run_campaign.sh r0a_m24base_$TS 5
```

**Expect `candidate_ratios["mps_mega"]["p50"] = 0.7556 ± 1 %` (M15 ≈ 5,823 µs).** Out of band
⇒ wrong pin, wrong C, or a non-inert edit. **STOP and diagnose. No downstream number is
interpretable until this passes.**

### F.1b R0b — the INERT-ARM arbiter (new, and equally blocking)

**R0a is nearly a tautology with respect to M24.** If G6's build gate passes (byte-identical
`.text` for `k0pf6gm_mps_mega`), R0a re-measures *identical instructions*; it arbitrates the
pin, the config and a stale checkout, not the M24 mechanism. A defect confined to the
`#if K0P6_M24_FILL` regions — i.e. **the entire design** — is invisible to it by construction,
and there is no ratchet anywhere in the first draft's ladder for the cost of the fill machinery
itself.

**R0b: build `FILL = 1`, `ZERO_PAD = 1`, `NORIG_CONST = 4096` — the plumbing fully live, the
fill effect exactly zero — and require 0.7556 ± 1 %.**

```bash
env -u K0_MOK_ROUTE_FILE -u K0_MOK_ROUTE_HIST \
  DHK_ROOT=~/DHK-m24-inert K0_MPS_CFG="$MPSCFG" \
  bash run_campaign.sh r0b_m24inert_$TS 5
```

This is the only measurement that separates the added control flow, the buffer read, the
handshake, the broadcast and the `ZERO_PAD` tail from the win. Without it a +400 µs plumbing
regression hides inside a 0.42 ratio that still looks "inside the predicted band", and it ships
into every subsequent M24 serving arm and every T2 extra built on top. **If R0b exceeds 0.7556
by more than 1 %, the overhead is real and must be fixed before any fill arm is believed.**
Note R0b at `NORIG_CONST = 4096` also exercises `ZERO_PAD`'s tail at zero rows, so a separate
`NORIG_CONST = 3968` point is the one that prices the tail itself (G4's done-when).

### F.2 Pin discipline (the trap, restated because it voids arms silently)

* `~/DHK-m24-base` — `mps.hip` = name stanza + m15 body **verbatim inline**, all M24 macros 0.
  (R0a)
* `~/DHK-m24-inert` — + `FILL 1`, `ZERO_PAD 1`, `NORIG_CONST 4096`. **(R0b, blocking.)**
* `~/DHK-m24-fill` — + `FILL 1`, `ZERO_PAD 1`, `NORIG_CONST 1539`. (arm A)
* `~/DHK-m24-ladder-<N>` — one pin per ladder point, `N ∈ {3072, 2048, 1024, 512, 256, 64, 0}`.
  (arm B — note the sub-`TGRAIN` points, added in rev 2 per §B.3.)
* `~/DHK-m24-skew-<k>` — `NORIG_TABLE` pins for the heterogeneous arm. (arm D, **mandatory**)
* `~/DHK-m24-fill-t2` — the above + `NULLWORK 1`, `TGRAIN 8` (after G7).

Arm E (the C / `flush_rows` re-sweep, §B.6) needs **no new pin** — `C` and `flush_rows` ride the
runtime `K0_MPS_CFG` descriptor word.
* Every campaign invocation passes an **explicit `DHK_ROOT`**. Running with the default
  `~/Distributed-HipKittens` measures the old MPS sibling and every ratio is void
  (`DECOMP_RUNBOOK.md:108-110`).
* Verify each pin before use: `grep -n "^#define K0P6_M24_" $PIN/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip`
  — the flags must appear in the **mps** tile (the m17 lesson, `DECOMP_RUNBOOK.md:112-116`).
* The harness JITs from `DHK_ROOT`; it **cannot** consume a prebuilt `.hsaco`
  (`DECOMP_RUNBOOK.md` G5).
* **⚠ The node's DHK checkout is 4 days stale (HEAD `604a9763`).** Before building any pin,
  diff the node's `~/DHK-m15` inlined body against local `ablations` HEAD's
  `k0pf6gm_device_tile_m15.hip`. If they differ, R0a is measuring a different kernel than this
  design was written against and the whole ladder restarts from the local file.

### F.3 The fill sweep — and why R6 as written does not measure it

**Correction to `DECOMP_RUNBOOK.md` R6.** R6 sweeps `K0_T ∈ {4096, 2048, 1024}`, and
**`run_campaign.sh:122-126`** (`-e "K0_T=${K0_T:-4096}"`, `-e "K0_MAXTOK=${K0_T:-4096}"`,
`-e "K0_MAXTOK_PROD=${K0_T:-4096}"` — rev 2: the first draft cited 143-144, which is the
`K0_MOK_ROUTE_HIST` env block and says nothing about MAXTOK) derives
`K0_MAXTOK = K0_MAXTOK_PROD = K0_T`. That shrinks the
**capacity** as well as the row count, so **both arms genuinely get a smaller problem** — which
is a batch-size sweep, not a fill sweep. The serving regime is *capacity 4096, real rows 1539*,
and R6 cannot produce it. R6 remains worth running (it separates "mega fixed costs" from
"padding") but it is **not** the M24 gate.

**The M24 arm.** `K0P6_M24_NORIG_CONST` gives the exact serving regime with **zero harness
changes**: `K0_T = K0_MAXTOK = 4096` for both arms, the production arm untouched and dense over
4,096 rows, the mega arm computing 1,539 (rounded to 1,792 at `TGRAIN = 256`).

```bash
# A — parity/perf at serving fill
DHK_ROOT=~/DHK-m24-fill K0_MPS_CFG="$MPSCFG" bash run_campaign.sh r7_fill1539_$TS 3
# B — fill ladder, same capacity, one rebuilt pin per N in {3072,2048,1024,512,256,64,0}
# C — captured routes at true fill, from ~/20260818_m15_campaign_routecap1/pair_01/1_m15/skew/routes/
K0_MOK_ROUTE_HOST_DIR=~/eplb_campaign/routes K0_MOK_ROUTE_FILE=<worker> \
DHK_ROOT=~/DHK-m24-fill K0_MPS_CFG="$MPSCFG" bash run_campaign.sh r7_routes_$TS 3
# D — HETEROGENEOUS fill (mandatory, rev 2): the only arm that has any arrival skew at all
DHK_ROOT=~/DHK-m24-skew-a K0_MPS_CFG="$MPSCFG" bash run_campaign.sh r7_skewA_$TS 3
# E — C / flush_rows re-sweep at the FILL operating point (no rebuild; runtime descriptor)
for C in 16 24 28 32; do for FR in 8 16 32; do
  DHK_ROOT=~/DHK-m24-fill K0_MPS_CFG="C=$C,g=353,mode=12,flush_rows=$FR" \
    bash run_campaign.sh r7_cfr_${C}_${FR}_$TS 3
done; done
```

**⚠ BLOCKER 1 — the correctness gate compares all `T` rows.** Our arm deliberately does not
write rows `[1792, 4096)`. Harness patch: `K0_MOK_COMPARE_ROWS` (compare only `[0, N)`), which is
the row-restricted comparison `M23_RAGGED_SEAL_DESIGN.md` §5.2-B already specifies for serving.
**Rev 2: it is a REL-ERR gate, never an exact comparison** — see §C.2's withdrawal box.

**⚠ BLOCKER 2 — rev 2, the POISON gate is a second, independent gate the first draft missed,
and the obvious mitigation defeats it.** `_poison_out` fills the **whole** `cand_out` with the
bf16 NaN pattern (`e004pf_k0pf_ab.py:1876-1893`), it is **re-armed before every soak epoch**, and
the soak `break`s on the first survivor (`:5388-5407`), with `mps_soak.pass` requiring
`poison_survivors == 0` (`:5416-5427`). So:

* `ZERO_PAD = 0` ⇒ 2,304 × 3,584 survivors, `completed_epochs = 1` of 600, **zero soak coverage**
  regardless of `K0_MOK_COMPARE_ROWS`, and no timing number from the arm is interpretable.
* `ZERO_PAD = 1` ⇒ the arm writes zeros over the poison, so `poison_survivors = 0` is
  **guaranteed even if the kernel computed nothing at all** — defeating the one detector whose
  docstring is *"Nonzero means the arm left a row of `out` unwritten this epoch"* (`:1921-1927`).
  Combined with a restricted compare range this would let the harness report a clean pass for a
  kernel that provably did nothing: **fairness item 3, "candidate actually ran", the exact prior
  sin.**

**The correct patch keeps the detector live.** Do **not** change `_poison_out` (leave it
whole-buffer). Change **`_poison_count` / `_poison_rows` only**, to count survivors over
`[0, N)` where `N = K0_MOK_COMPARE_ROWS` (default `T`). Then:

| build | rows `[0,N)` | rows `[N,4096)` | detector |
|---|---|---|---|
| `ZERO_PAD=0` | must be written by the mega | keep poison, **not counted** | **live** |
| `ZERO_PAD=1` | must be written by the mega | zeroed by the tail, **not counted** | **live** |

**⚠ BLOCKER 3 — the poison SELF-TEST is hard-coded to row `T − 1` = 4095** (`_st_idx`,
`:4928-4939`), which under `K0_MOK_COMPARE_ROWS = 1792` is outside the compared range. The gate
then passes, `selftest_fails = False`, and the harness prints
`one_row_poisoned_fails=False` — i.e. it **declares its own correctness detector unwired for
every M24 arm**, a line an author expecting tail rows to be excluded would read as noise. The
patch must move `_st_idx` to `N − 1` and record **two** assertions, not one:

1. poisoning a **live** row (`N − 1`) ⇒ the restricted gate **FAILS**;
2. poisoning a row at index **≥ N** ⇒ the restricted gate **PASSES** (that is the intended
   semantic, and it must be demonstrated, not assumed).

**Land all three patches before any arm, and prove them inert**: with no env set, R0a must still
reproduce 0.7556 — the `ROUTE_REPLAY_PLAN` zero-change ratchet, `DECOMP_RUNBOOK.md` §4.3.

### F.4 Predicted numbers (state them now so the measurement can falsify them)

Baseline: production p50 ≈ 5,823 / 0.7556 = **7,706 µs**; M15 base = **5,823 µs**.

| arm | predicted M15 µs | predicted ratio | interpretation if hit |
|---|---|---|---|
| **R0b inert** (`FILL=1, NORIG_CONST=4096`) | **5,823** | **0.7556 ± 1 %** | **blocking**: the plumbing costs nothing |
| `NORIG_CONST = 1792` (φ = 0.90) | **2,830** | **0.367** | design works, φ ≈ 0.90 confirmed |
| optimistic (φ = 0.935) | 2,660 | 0.345 | fixed costs smaller than modelled |
| pessimistic (φ = 0.80) | 3,320 | 0.431 | still a large win, revise §A.4 down |
| `NORIG_CONST = 0` + `NULLWORK`, **G7 says `T_eff = 0` is safe** | < 300 | < 0.04 | globally-idle fast path proven |
| `NORIG_CONST = 0` + `NULLWORK`, **G7 forces the `TGRAIN` clamp** | **600–900** | 0.08–0.12 | *also* a pass — the mechanism works as re-specified |

**The `NORIG_CONST = 0` rows are TIMING-ONLY and have no serving counterpart** (§B.5 demotion
box). They are excluded from every headline, and their correctness gate must be reported as
**N/A, not pass**: with `K0_MOK_COMPARE_ROWS = 0` the comparison range is `[0, 0)`, `absum > 0`
(`e004pf_k0pf_ab.py:3877`) cannot hold, and a vacuous pass on an arm that computed nothing is
exactly what fairness item 3 forbids. **Rev 2: the prediction is stated as two conditional
branches** because §H.4 reserves the right to clamp `T_eff` to a minimum of `TGRAIN`; a single
number here would be unfalsifiable — either outcome could be read as failure.

**Falsification thresholds — and their precondition.** Ratio ≥ 0.55 ⇒ the fixed-cost fraction is
≥ 0.30, far above the ledger; re-derive §A before spending serving time. Ratio ≈ 0.75
(unchanged) ⇒ the `T_eff` plumbing is inert — check the pin, then check that **all five** sites
were converted (§B.3). A ratio below 0.30 ⇒ suspect the arm is not computing what it should; run
the row-restricted rel-err gate and the poison count before believing it.

> **⚠ Rev 2 — these thresholds are valid ONLY after arm E.** `C = 28` / `flush_rows = 16` is the
> operating point tuned at `T = 4096`; at `T_eff = 1792` the pool's quota and slab-1's shadow both
> shrink ~2.3× and the tuned point moves (§B.6). Reading a high ratio against these thresholds
> **before** the C/`flush_rows` re-sweep would attribute a config artifact to a small φ and
> under-sell or abandon the design. **Arm E runs before §A is revised on any ladder number.**

**Fill-ladder read — rev 2: the affine fit is a summary, not a measurement of φ.** The first
draft said "fit `a + b·N`; `a` **is** the fixed cost". The response is not affine in the region
the ladder samples, and the design says so itself: T2-e notes that at `T_eff = 1792`,
`nbatches = 448` against ~228 compute CTAs × 4 waves = 912 wave-tickets, so **most waves get
0–1 batch and the M8 dynamic ticket degenerates**; at N = 1024 it is 256 tickets against 912,
deep in that regime, and at N = 4096 it is not. Further curvature comes from the `TGRAIN`
round-up, `k0p6_sort::pad`'s per-expert round-up to 32 (3 % of live rows at full fill, 8 % at
37.6 %), and GEMM tile quantisation. A straight line through points straddling a documented
regime change returns an intercept that has absorbed the curvature, and each point is a
separately compiled pin, so per-build codegen scatter lands directly in `a`. Therefore:

* Report φ as an **interval from the two extreme points**, plus the **residuals** of the affine
  fit as an explicit curvature diagnostic. If the residuals are not small, φ is not measured.
* Evaluate **T2-e** before reading φ from the low points; if the M8 ticket degeneracy is real,
  the low points measure T2-e's absence, not φ.
* The cross-pin resource-tuple equality check (§E) bounds the codegen-scatter contribution.

**Rev 2 — the `production_µs(N)` control is deleted; it was flat by construction.**
`NORIG_CONST` is a macro in the *candidate* pin only; the production arm's code path, `K0_T`,
`K0_MAXTOK` and `K0_MAXTOK_PROD` are identical across all ladder points because the M24 ladder
deliberately holds `K0_T = 4096` (`run_campaign.sh:122-126`). Production p50 is therefore flat to
run-to-run noise **no matter what the candidate does, including in the failure case the check
was meant to catch**. The real confound it named — the harness shrinking capacity alongside row
count, the R6 defect — is reachable only by varying `K0_T`, which this ladder never does. The
controls that actually work are: **(i) R0b**, (ii) the per-pin resource tuple + no-spill gate,
(iii) the poison count over `[0, N)`, and (iv) one deliberate `K0_T = 2048` cross-check point
confirming the ladder is not accidentally varying capacity.

### F.5 Then, and only then: serving

Order-balanced pairs, `ARM_COOLDOWN = 240`, fresh server per arm, exact-token SHA, the
three-way decomposition (`m15/native` headline, `m15/patched-stock`, `patched-stock/native`),
cells c32p + c8/c16/c32 + c512p, each cell reporting **its own measured fill**
(`Σ T_eff / Σ 4096` from `M24_FILL_RECEIPT`). Mandatory fairness-audit subagent sign-off before
any "beats production" wording (§H.1 is its first checklist item).

---

## G. Implementation work-list

**Rev 2, reordered to match the revised design.** Ordered; each step is a self-contained unit for
an implementation agent. G0a/G0b and G1–G6 are CPU-only or laptop-only and safe while the GPUs
are busy.

| # | step | files / functions | done when |
|---|---|---|---|
| **G0a** | **NEW, prerequisite.** Measure the **unchanged** m15 arm's run-to-run self-divergence (same pin, same seed, two runs) and record the envelope. This is the reference every accuracy claim in the design is stated against, because §C.2 shows the kernel is not bit-reproducible at any `T`. | node, short GPU (or harvest from existing paired runs) | a written envelope; §H.6's 11.33 % is compared against it |
| **G0b** | **NEW, prerequisite, laptop-only and free.** Build the true `n_orig` histogram from the seal receipts' per-rank-step `in_bucket_sum_orig` (`DECOMP_RUNBOOK.md:36-37`). Report: the zero-fill fraction, its **cross-rank correlation** (are dummy ranks step-synchronised?), the bimodality, and `E[round_up(n,256)]/E[n]` for the TGRAIN cost. | laptop | §A.1's 31 % proxy row and §A.4.3's Tier-1 sizing are replaced with measured numbers |
| **G1** | Declare the macro block and the descriptor slot. `K0P6_M24_*` `#ifndef` stanzas after KERNEL:165; `#define K0P6_D_M24_FILL 71`; the **raise-not-redefine** `K0P6_M15_D_LEN` block *after* the KERNEL:239-282 cascade (§D); all `#error` cross-checks incl. `FILL && !ZERO_PAD && !ALLOW_DIRTY_PAD` and the slots-63..70-unclaimed note. | `k0pf6gm_device_tile_m15.hip` | compiles at **every** macro combination incl. `+STAGED`, `+REPLICATE`, `+ADAPTIVE`, `+SLOTPOOL`, with **no macro-redefinition warning**; default `.text` unchanged (G6) |
| **G2** | Add `k0p6_m24_publish(desc, cur, world, pperr)` — **`bid==0 && tid==0` only, pre-barrier**: `load_relaxed<scope::system>` the header, validate magic / `gen_echo` / `csum` / flags / `gen > last_seen_gen` / `n_orig[p] ∈ [0,MAXTOK]`, compute `clamp(round_up_sat(n_orig[cur],TGRAIN), 0, desc[K0P6_D_T])`, `publish_value<scope::agent>` it to `tail.agreed_T_eff` + `agreed_flags`. Post-barrier, same thread stores `last_seen_gen = gen`. Add `k0p6_m24_teff(desc)` — a scoped scalar load of `agreed_T_eff`, returning `k0p6_dread(desc,K0P6_D_T)` when `FILL==0`. `NORIG_CONST`/`NORIG_TABLE` change only the published **value** (through an `asm volatile` opacity barrier), never the mechanism. **Nothing goes in `mps_state`.** | new `__device__ __forceinline__`s next to `k0p6_epoch` (318-324) | no write to `mps_state`; grid-uniformity is by single-writer + the existing barrier at 882; `STRICT` fail-closes |
| **G3** | Substitute `k0p6_m24_teff(desc)` at **five** sites: KERNEL:977, 1375, 1587, 1891, **and 943 (`T_pre`, the ADAPTIVE M0.5 pre-pass)**. **Re-derive at each site — never hoist into a register live across the GEMM bodies** (§B.3, exp_38 law). **Do not touch** the entry guard's capacity use of `input_tokens` (759, 789-790). | same file | `grep -n "K0P6_D_T"` shows one capacity use + the `T_eff` sites, **and** `grep -n "input_tokens"` shows only 759 and the 789-790 guard; G6's no-spill assertion passes |
| **G4** | `K0P6_M24_ZERO_PAD`: grid-strided `uint4` zeroing of `out[T_eff, desc[K0P6_D_T])` at the tail of the M8 block, after both batch loops and before the `ts_mark` (KERNEL:1946-1952); same geometry as the consume-and-zero at 724-728. | same file | cost **measured** from the `NORIG_CONST=3968` R0b point, not asserted (§B.4 expects ~21 µs, not 7) |
| **G5** | `K0P6_M24_NULLWORK`: `null_work = (total == 0) && (T_eff == 0)` right after `const unsigned int total = s_total;` (KERNEL:1226); guard the bodies of the M2 unpack loop (1289), hole-fill (1337), **the scatter at 1474-1476 ONLY**, `pull_src_fill` (1477), the M6 call (1494) and the M7 phase-2 call (1542). **Explicitly EXCLUDE `k0p6_sort::pad` at 1460** — §A.2 lists it as its own non-scaling region and guarding it leaves stale per-expert round-up entries in `sti`/`swt` for any later arm that skips only part of the set. **Guard bodies, never barriers or publications**; the untouchable set is the **NINE** grid barriers **882, 950, 956, 1246, 1392, 1456, 1479, 1701, 1825** (950/956 = ADAPTIVE M0.5, 1825 = STAGED R2 — both live in the compositions §H.8 claims) plus the three publication blocks (1126-1178, 1710-1726, 1958-1992). | same file | a globally-dummy epoch still publishes every word a peer polls; the comment lists all nine barriers |
| **G6** | **BUILD GATE** (§E): default-build resource tuple + byte-identical `.text` vs pre-M24, **plus** the per-arm no-spill assertion and the cross-pin resource-tuple equality check. Blocking for everything below. | `fabric_build/`-style receipt | `M24_BUILD_GATE_RECEIPT` written |
| **G7** | Read `hkp::csr_scan_block256` and `hkp::pull_src_fill` in `~/amd-master` (read-only headers). Determine `T`-generality. If yes, drop `TGRAIN` to 8. If no, document the constraint, evaluate `TGRAIN = 64`, and **decide the `T_eff = 0` clamp** — which branch of §F.4's two-branch prediction applies. | node, read-only | a written finding, and §F.4's prediction resolved to one branch **before** the arm runs |
| **G8** | **Adversarial verification subagent** (Opus, effort high) before any node build: prove `T_eff` is grid-uniform via the single-writer + barrier-882 argument; prove the reject path degrades to today's exact behaviour; prove `last_seen_gen` cannot be clobbered by M0's `mps_state` zeroing; prove cross-rank divergence is safe by walking the receive path; prove **no barrier of the nine and no publication** is skipped by G5; prove `T_eff = 0` is safe through `csr_scan_block256`, `pull_src_fill`, `nbatches = 0` and the M7 pool sweep; prove the clamp defeats the `0x7FFFFF80` overflow case; **build `ALLOW_DIRTY_PAD` and confirm the NaN-persistence hazard is real** so the `#error` is justified. | — | written sign-off, veto power |
| **G9** | MoK harness patch, **three parts** (§F.3): (1) `K0_MOK_COMPARE_ROWS` **rel-err** gate over `[0,N)`; (2) `_poison_count`/`_poison_rows` restricted to `[0,N)` with `_poison_out` left whole-buffer; (3) `_st_idx → N−1` plus the two-assertion selftest (live-row poison FAILS, `≥N` poison PASSES). Then the zero-change ratchet proof. | `e004pf_k0pf_ab.py`, `run_campaign.sh` | R0a still reproduces 0.7556 with no env set; both selftest assertions recorded |
| **G10** | Build the pins (§F.2) with the stale-checkout diff first; run **R0a** on `~/DHK-m24-base` **and R0b** on `~/DHK-m24-inert`. **STOP** unless *both* are 0.7556 ± 1 %. | node, CPU compile + short GPU | both arbiters pass |
| **G11** | Run F.3 arms **A, B, C, D (heterogeneous — mandatory), E (C/flush_rows re-sweep)**. Run **E before** reading any ladder number against §F.4's thresholds. Publish φ as an interval with fit residuals, not a point from an affine fit. | node | `M24_MOK_RESULTS.md` with the interval, the residuals, arm D's skew delta, and arm E's optimum |
| **G12** | Shim: `m15_contracts.py` — `DescriptorSlot.M24_FILL = 71`, **`M15_DESCRIPTOR_WORDS_M24 = 72` as a SEPARATE constant (leave `M15_DESCRIPTOR_WORDS = 63`)**, explicit `expected_words=` on M24 builds only, `:1314` compat branch extended to 72 with slots 63-70 zero, buffer spec (`32 + 4·world + 16` bytes, int32, device, **one per layer**). | `m15_contracts.py:82-152, 460-540, 1290, 1314` | a default-0 control arm's descriptor path is **provably unchanged** (this protects the `m15/patched-stock` leg) |
| **G13a** | **NEW, blocking prerequisite.** `m15_runtime.py` is **not in this repository** — every `m15_runtime.py:` citation in §B.1 is node-only and unverified locally. Fetch the deployed file, re-verify: the capture hard-fail (`M15-DESC-010`), the per-call rewrite refusal (`M15-DESC-012`), **whether descriptors/`mps_state`/`epoch_cell` are allocated per layer or shared**, and **whether the DP all-reduce's device staging row (`DPU:47`) is reachable from the shim**. Record verified line numbers. | node | a written finding; if the device row is unreachable, §B.1's pinned-ring fallback is activated *with* its event discipline |
| **G13b** | Shim: allocate the **per-layer** fill buffers once; write each pointer into that layer's descriptor via the existing out-of-capture `_write_descriptor`; add `write_fill_vector(...)` as a **one-block device kernel** reading the device-resident all-reduce row and broadcasting the 64-byte record into all 58 buffers, on-stream between the router and the mega launches. Mirror `write_decision_bitmap`'s structure exactly. | `m15_runtime.py` (node) | `M15-DESC-010` never fires; the op is a kernel node under capture, **no host staging on the primary path** |
| **G14** | GMR patch: call `write_fill_vector` at the ragged-seal predicate site on every step before the graph replay. New sentinel `PF4H_M24_FILL_V1`. **The patch-chain head is `~/pf4h_vllm_20260729/*/shim/pf4h_integration/apply.py`, NOT `~/eplb_campaign/`.** Chain-test against the scratchpad mirror pattern (`aug18-prefill/m23/test_m23_patch.py`) first. | `apply.py` chain | anchor-exact, fail-loud, idempotent; mirror test green |
| **G15** | Receipts: `M24_FILL_RECEIPT` per step (`gen`, `n_orig[]`, `T_eff`, reject count + reason) + run-level `Σ T_eff / Σ 4096` measured fill **and `Σ T_eff` vs the shim's independently-recorded `Σ n_orig`** (the post-hoc under-dispatch audit, §B.2). Re-assert the `VLLM_MOE_SKIP_PADDING` refusal (M23 E6). | shim | the fairness auditor can compute measured fill **and** verify no under-dispatch from artifacts alone |
| **G16** | Serving campaign per §F.5 with **`STRICT = 1` on the first arm**, + mandatory fairness audit. **Must include the NATIVE production leg** — the headline does not exist until it is measured (§A.4.1) — plus the `production + VLLM_MOE_SKIP_PADDING=1` control arm (§H.1). | node | `FAIRNESS_AUDIT` section signed; all three legs of the decomposition present |
| **G17** | Optional T2 extras, each its own arm and gate: T2-c dense scatter (moves the order further — separate accuracy run), T2-e M8 granularity (**evaluate before trusting the low ladder points**), T2-d adaptive `C` (**only if arm E's runtime sweep justifies it**). | — | independently measured |

**19 steps.** G0a/G0b are free and unblock the accuracy and Tier-1 stories; G1–G6 are the whole
kernel change and are CPU-only; G6 is the hard gate; G8 is the mandatory adversarial review;
G10's **R0a *and* R0b** are the arbiters that must pass before any number counts; G13a is a
blocking evidence prerequisite for the shim.

---

## H. Risks and open questions

**H.1 — Fairness: is skipping pad rows "cheating"?** This is the risk that can void the whole
campaign, and it must be handled *before* the claim is written. Production **could** skip
padded rows: `MK:1137-1151` exists for exactly that, gated on `VLLM_MOE_SKIP_PADDING`, which
ships **off** (`envs.py:191`, `envs.py:1499`). The GENUINE production baseline is therefore the
flag-off configuration, and comparing against it is legitimate. But the honest presentation
**must** include a `production + VLLM_MOE_SKIP_PADDING=1` control arm so the reader can see how
much of our win is "a structural freedom the vendor left on the table" versus "a better
kernel". Note the mega arm can **never** enable that flag (M23 §3.6: `topk_ids = -1` →
`dest = -1` → `peer_ptr(rank = −1)` OOB). Claim wording must read: *"at c32p's measured 37.6 %
fill, against shipped-default production"* — with the fill number in the sentence.

**H.2 — Does the mega stay efficient at 1,539–1,792 rows/rank?** CLAUDE.md's banked law is that
the mega inverts below ~1,600–1,800 tokens/rank. `T_eff = 1,792` lands **exactly on that
threshold**. Two mitigating facts: (i) the inversion was measured against production *at the
same T*, whereas here production stays at 4,096 — we win the comparison regardless; (ii) the
receive-side row count (which drives the GEMMs) is `Σ_p n_orig[p] × fanout/world`, not
`T_eff`, so the GEMM problem stays much larger than 1,792 rows. But the *per-row efficiency*
does fall, which is exactly the `a` term the F.3 fill ladder measures. **This is the single
biggest quantitative uncertainty in §A.4** and the ladder is designed to retire it in one
25-minute campaign.

**H.3 — Arrival-skew and spin-limit headroom get worse, not better.** M23 §7.1 already names
cross-rank arrival skew as its risk #1: a lightly-loaded rank reaches the mega early and spins
with all 256 CTAs, and the "mode-14 law" says spinning CTAs invalidate L2 for the very peer
writes they are waiting on. **M24 amplifies this**: a dummy rank that used to spend real time
pushing 4,096 rows now finishes M1 in ~0 µs and spins for the whole of its peers' dispatch.
`M15_SPIN_LIMIT = 20,000,000` was tuned on near-lockstep exact-4096 batches. Mitigations:
(a) read `SPIN_DBG` (slot 52, written KERNEL:1238) in every M24 arm and report headroom;
(b) `[MPS SPIN] fail_max > 0` under captured routes is direct transport-degradation evidence
(`DECOMP_RUNBOOK.md`); (c) compose the M21 `K0P6_M15_POLL_BACKOFF` arm from
`.claude/worktrees/wf_b38dfe27-d09-2` — it exists, it build-gate-passed, and it targets exactly
this failure mode.

**Rev 2 — this is no longer an unpriced risk, and the earlier recommendation is reversed.**
If G0b finds dummy-ness is **per-rank** rather than step-synchronised (which is exactly how DP
dummy runs work), this is the **common case, not the tail case**, and it can make Tier 1 a net
loss: rank 3 idle while ranks 0-2,4-7 each carry ~1,539 tokens means rank 3's 256 CTAs spin on
`rows_done`/`chunk_ready` for the whole of its peers' ~160 MB/rank xGMI dispatch, and the
mode-14 law says that *slows the seven ranks it is waiting on*. §A.4 charges this zero
(assumption (v)); the correct quantity for a mixed-fill step is `max_p n_orig[p]`, not the mean.

**Gate, threshold and abort criterion (all pre-committed):**
* **Arm D (heterogeneous `NORIG_TABLE`) is mandatory** and runs before any serving time. The
  homogeneous `NORIG_CONST` arms set the same constant on all 8 ranks and have **zero skew**, so
  they are structurally incapable of surfacing this.
* **Threshold:** arm D's M15 µs at matched *mean* fill must not exceed the homogeneous arm's by
  more than 10 %. Exceeding it means the skew term is first-order and §A.4 must carry it.
* **Action if exceeded, or if `[MPS SPIN] fail_max > 0` under captured routes:** bundle
  `K0P6_M15_POLL_BACKOFF` into the **first** M24 serving arm. The first draft recommended
  against bundling ("one variable at a time"); with the cost now identified as potentially
  first-order and the arm already build-gate-passed, the honest default is to bundle and report
  both arms, rather than spend a serving campaign discovering it.

**H.4 — `T_eff` in the unmodifiable headers.** `hkp::csr_scan_block256`, `hkp::pull_src_fill`,
`k0p6_sort::*` and `hkp::zero_part_scale_transpose` are in `hkp_sort.hpp` / `hkp_quant.hpp`,
which are outside this repo and read-only to us (`moe_mps_adapter.cuh:531-539`). They have only
been run at `T ∈ {4096, 2048, 1024}`. `TGRAIN = 256` is the de-risking answer and G7 is the
resolution. `T_eff = 0` through `csr_scan_block256` is the specific untested corner — G8 must
walk it, and if it is not provably safe, clamp `T_eff` to a minimum of `TGRAIN` (costing 256
rows on dummy steps, which is still a 16× reduction).

**H.5 — Staleness. Rev 2, substantially rewritten.** The first draft treated staleness as
"mitigated by the handshake", with the only residual being a host write issued on the wrong
stream. Two reviewers showed the real hazard is different and larger: with a **single reused
pinned staging buffer** and the GPU running behind the CPU, the device can read a payload from a
**later** step, which any `gen > last_gen` test accepts. §B.1 therefore moves the payload to a
**device source** (the DP all-reduce's device staging row) written by an on-stream device pre-op,
which removes the host page entirely and makes the cited `write_decision_bitmap` precedent apt.
Residual risks, and what covers each:
* device row unreachable in the deployed tree → **G13a** decides; the pinned fallback is
  permitted only with a ≥ 8-slot ring **and** event-gated reuse.
* pre-op skipped for a step → caught by `gen > last_seen_gen` (fallback to `T = 4096`).
* torn / partial read → caught by `gen_echo` + `csum`.
* garbage value → caught by the `n_orig[p] ∈ [0, MAXTOK]` validation and the clamp.
* everything above silently → caught post-hoc by G15's `Σ T_eff` vs `Σ n_orig` audit.
* `K0P6_M24_STRICT = 1` on the **whole first serving arm** turns any reject into a loud failure.

**H.6 — Accuracy attestation shape. Rev 2 — this open question is CLOSED, and the answer changes
the gate.** The first draft said real rows are bit-identical and treated the pre-existing
**11.33 % (116/1024) output-token divergence between the current arms**
(`BOTTLENECK_SIGNALS.md` §0) as a mystery to be explained before M24's A/B could be read.
§C.2's withdrawal box shows it is **the expected signature of a mechanism that already exists**:
several experts owned by one destination rank accumulate into a single slot row with packed bf16
atomics whose order is set by a per-step tile schedule. The kernel is **not bit-reproducible run
to run today**, so no bit-identity gate can ever be built on it and the exact-token SHA A/B is
**not** a parity gate for M24. The replacement is **G0a's run-to-run envelope** for the unchanged
arm, against which both the 11.33 % and M24's divergence are read. The MoK gate is rel-err over
`[0, N)`. The `FAIRNESS_AUDIT` section must state this shape explicitly (checklist item 6)
rather than implying bit-identity — claiming bit-identity we cannot deliver would be its own
fairness failure.

**H.7 — Fill varies by cell.** 37.6 % is c32p's number. c8/c16/c32 (ISL 1024, OSL 512) are
decode-heavy and will have very different fill; c512p has never been run. **Every quoted cell
must carry its own measured fill from `M24_FILL_RECEIPT`**, and the small-batch cells — where
the mega is expected to lose — must be reported, not hidden (CLAUDE.md's standing rule).

**H.8 — Composition with M18/M19/M20 and T2B.** M24 claims descriptor slot 71 and touches only
loop bounds, so it composes on paper with every other arm. Untested in combination. The M20
prefetch duty tables (`k0p6_m20_prefetch`, KERNEL:599-631) are sized by a host-precomputed
schedule independent of fill, so a fill-aware kernel gives the prefetch **more** slack, not
less — likely a synergy, unmeasured. **Rev 2 — "composes on paper" was too generous in three
concrete places, all now fixed upstream:** (i) the M0.5 replication pre-pass histograms over the
full padded row count (substitution **site 5**, §B.3) — without it an M24+M20 arm's replication
decision is driven by pad routing and the arm under-delivers with the pre-pass on nobody's
suspect list; (ii) the composed arms carry **three grid barriers the first draft's untouchable
set omitted** (950, 956, 1825 — §A.2, G5); (iii) `K0P6_M15_D_LEN` is an exclusive cascade and
must be **raised, not redefined** (§D).

**H.9 — What the design does NOT address — rev 2, PROMOTED to a co-equal work item.** The MoE
region is 40–54 % of step time, so even a perfect ρ caps the e2e win well below the mission
target on its own. The other 46–60 % — dense layers and MLPs that also run on 4,096 padded
rows — is untouched by any kernel change and is an *integration* fix (shrink the graph bucket,
or add smaller buckets, so fewer pad rows enter the model at all). And `BOTTLENECK_SIGNALS.md`
§0's finding that the m15 arm ran **21 % more B4096 steps than stock for the same real traffic**
(249.1 vs 205.9 of 300; κ = 1.192× the MoE row-work per real token delivered) is a
scheduling/composition defect **worth roughly the same as M24**.

**§A.4.1 now prices κ in, and the conclusion is that M24 alone does not reliably clear the
20–30 % target**: central +22 % against the *secondary* baseline, unknown against native. The
first draft's +31 % headline came from omitting κ and from a frame error. **Treat H.9 as a
parallel workstream, not as future work** — if it is left unfixed, an M24 campaign that performs
exactly to model will read as an under-delivery, and the team will be tempted to re-derive §A on
a defect the model deliberately excluded.

---

## Appendix R — Review responses (rev 2, 2026-08-18)

Three adversarial reviewers (lenses: **parity**, **hardware**, **measurement**) returned
**5 distinct blocking findings** (several found independently by two or three lenses),
**16 major** and **10 minor**. Every blocking and major finding is dispositioned below.
**All blocking findings are FIXED; none are rebutted; none remain open.**

I re-verified every load-bearing citation myself against the tree before acting. The reviewers'
evidence was accurate in every case I checked; the two citation errors I found were in **my own
first draft**, not in the reviews.

### R.1 Blocking — all FIXED

| # | finding (lenses) | disposition | where |
|---|---|---|---|
| **B1** | `last_gen` cannot live in `mps_state`: M0 zeroes words [0,24) grid-wide every launch, the block has zero slack, and the post-barrier store races the zeroing loop. `use_fill = (gen > last_gen)` would be unconditionally TRUE — the guard **fails OPEN**, giving silent under-dispatch of real tokens. (parity #1, hardware #1, measurement #1 — found independently by all three) | **FIXED.** Verified: KERNEL:893-901 zeroes `i < K0P6_MPS_ST_WORDS + 2*K0P6_MPS_TS_COUNT` = 24 words, **after** the barrier at 882, with only a block-local `__syncthreads()` at 932; `moe_mps_adapter.cuh:95,109`; `m15_contracts.py:468,960-967` allocates exactly 12 int64 = 96 B. All three sub-claims (no slack / TS-ledger shift / write race) hold. **No M24 state goes in `mps_state` at all** — `last_seen_gen` moves to a device-owned tail of M24's own per-layer buffer, which M0 never touches and the host copy never covers. | §B.2 rev 2, §D, G2 |
| **B2** | The bit-identity guarantee is false: several experts owned by one destination rank accumulate into a **single slot row** via packed **bf16** atomics whose order is set by the per-step tile schedule, which M24 changes. C.2 clause 6 only argued the M8-level sum over destination *ranks*. Consequence: no bit-identity gate can be built, and the "pre-existing 11.33 % divergence" is this same mechanism. (parity #2) | **FIXED — guarantee withdrawn.** Verified: `n2_phase2_gm_mps.cpp:16,36-37,236` (`global_atomic_pk_add_bf16` / `accumulate_peer_bf162`), mode-12 slot path `:329-361`, slots are `unsigned short*` (KERNEL:1894); M1 dedup is per **destination rank** (`__match_any_sync` on `dest`, KERNEL:1020-1023) while the scatter emits one entry per (row, expert); tile descriptor rebuilt from live counts at KERNEL:1406-1416. The reviewer is right, including the corollary that the kernel is **already** not run-to-run reproducible. Guarantee downgraded to "same addend set, tolerance-identical within the kernel's own envelope"; new **G0a** measures that envelope; MoK gate becomes **rel-err**; the serving SHA A/B is explicitly **not** a parity gate; §H.6's open question is closed. | §C.1, §C.2 box, §H.6, G0a, G9 |
| **B3** | The pinned-staging H2D can deliver a payload from a **later** step (host overwrites the pinned bytes before the SDMA drains), which `gen > last_gen` **accepts**. The cited `write_decision_bitmap` precedent is inapt — it is a device op over a device tensor. (hardware #2, parity major #4) | **FIXED at the source.** The payload becomes **device-sourced**: `write_fill_vector` is an on-stream one-block device kernel reading row 0 of the DP all-reduce's **device** staging tensor (`DPU:47`) and broadcasting into all 58 per-layer buffers — the exact `write_decision_bitmap` shape, so the precedent is now apt. No host page exists to race. The host fallback survives only with a ≥8-slot pinned ring **and** event-gated reuse, plus mandatory `STRICT`. **G13a** is a new blocking prerequisite that verifies the device row is reachable. | §B.1 rev 2, §H.5, G13a/G13b |
| **B4** | The MoK **poison** gate is a second, independent gate: whole-buffer poison, re-armed per soak epoch, `break` on first survivor, `pass` requires 0 survivors. `ZERO_PAD=0` ⇒ zero soak coverage; `ZERO_PAD=1` ⇒ survivors are 0 **even if the kernel computed nothing**, defeating the only "candidate actually ran" detector. (measurement #3) | **FIXED.** Verified `e004pf_k0pf_ab.py:1876-1893, 1893-1904, 1921-1927, 5388-5407, 5416-5427, 3877`. Patch changed: leave `_poison_out` whole-buffer, restrict **`_poison_count`/`_poison_rows` to `[0,N)`** — the detector stays live under both `ZERO_PAD` settings. The `NORIG_CONST=0` arm is relabelled **TIMING-ONLY** with its correctness gate reported **N/A, not pass**. | §F.3 blocker 2, §F.4, G9 |
| **B5** | The poison **self-test** is hard-coded to row `T−1` = 4095; under `COMPARE_ROWS=1792` it passes and the harness prints `one_row_poisoned_fails=False`, declaring its own detector unwired for every M24 arm. (measurement #9, filed major — **I have escalated it to blocking**) | **FIXED, escalated.** Verified `:4928-4939`. It defeats the gate that guards the project's named prior sin, so it is treated as blocking. `_st_idx → N−1`, and the patch must record **two** assertions: live-row poison ⇒ FAIL, `≥N` poison ⇒ PASS. | §F.3 blocker 3, G9 |

### R.2 Major — all FIXED (none rebutted)

| finding (lens) | disposition |
|---|---|
| Grid-uniformity of `use_fill`/`T_eff` is not established by a pre-barrier read — a barrier orders CTAs against each other, not against an async DMA writer; divergent `T_eff` gives a *hole* pattern in the grid-strided M1, a `csr_scan_block256`/M8 range mismatch, and divergent `nbatches` on the shared `m8_front` cursor (KERNEL:1598/1904). (parity #3) | **FIXED.** `T_eff` is now produced by a **single reader** (`bid==0 && tid==0`, pre-barrier), published to `agreed_T_eff`, and read by every CTA **after** the existing barrier at 882 + `acquire_payload_agent()` at 883. Grid-uniform by construction from one read of the buffer. No new barrier or poll. |
| Hoisting `T_eff` into one entry-computed register forces it live across both `__forceinline__` GEMM bodies at `__launch_bounds__(256,1)` — the exp_38 spill cliff (+726.9 µs); the correct construction (re-derive per site, as `k0p6_dread(desc,K0P6_D_T)` is today) is never stated. (hardware #3) | **FIXED.** Verified KERNEL:749, 392, 418, 57-59; today's four sites re-read and hoist nothing. §B.3 now makes per-site re-derivation a **hard requirement** with `k0p6_m24_teff()`, and §E's no-spill assertion is named as its enforcement. |
| A fifth row-count site is invisible to the stated grep: KERNEL:759 aliases `desc[44]` into `input_tokens`, used as a loop bound at KERNEL:943 (ADAPTIVE M0.5 histogram over `T_pre * TOPK`). Composed M18/M19/M20 arms derive their replication decision from pad routing. (hardware #5, parity minor (b)) | **FIXED.** Verified KERNEL:759, 943, 952. Added as **substitution site 5**; §A.2's M0.5 row corrected; G3's done-when now greps `input_tokens` as well as `K0P6_D_T`. |
| G5's "untouchable" barrier set lists 6 of **9**; 950/956 (ADAPTIVE M0.5) and 1825 (STAGED R2) are live in exactly the compositions §H.8 claims. A subset of CTAs skipping a `grid_barrier` desynchronises the epoch for the whole grid — the non-unanimous-collective hang class. (hardware #6) | **FIXED.** Verified by grep: 882, 950, 956, 1246, 1392, 1456, 1479, 1701, 1825. All nine listed in §A.2, §B.5 and G5. |
| `total == 0` is essentially unreachable in serving — a DP-dummy rank still owns 32 experts under EP8 and receives from every busy peer — so NULLWORK is dead code and its "tens of µs vs 5,800 µs" headline describes a regime that does not occur; the `NORIG_CONST=0` arm has no serving counterpart. Also, the uniform-decode rescue is a *different population* from `n_orig = 0`. (parity #5, hardware #7) | **FIXED.** Verified KERNEL:1211-1226 (`total` sums all 8 sources) and `CORPUS_FINDINGS.md` §1/§14-16. NULLWORK **demoted** to a diagnostic; removed from Tier-1 sizing; the `NORIG_CONST=0` arm labelled TIMING-ONLY and excluded from headlines; §C.5's rescue conflation corrected. |
| The 31 % "fully-degenerate (n_orig ≈ 0)" row is a routing-homogeneity **proxy**, not an `n_orig` measurement, with no cross-rank step alignment and an early-run capture bias; a direct measurement already exists in the receipts. (hardware #7, measurement minor) | **FIXED.** §A.1's row relabelled with a ⚠ and a no-quote rule; new **G0b** (free, laptop-only) builds the real histogram from `in_bucket_sum_orig`, including the cross-rank correlation and the TGRAIN cost `E[round_up(n,256)]/E[n]`. §A.4.3 defers Tier-1 sizing to it. |
| Tier-1 creates a large, new, **exposed** rendezvous on the ranks it makes fast — a `T_eff=0` rank spins all 256 CTAs through its peers' whole dispatch, and the mode-14 law says that slows the peers it waits on. ρ has no skew term; the homogeneous `NORIG_CONST` arms have **zero** skew and cannot see it. (parity #6, hardware #8, measurement #5) | **FIXED.** Verified KERNEL:993, 1211-1224, 1229-1241. Added to §B.5 as an explicit cost; **arm D (`K0P6_M24_NORIG_TABLE`, heterogeneous per-rank fill) is mandatory**; §H.3 carries a 10 % threshold and a pre-committed action; §A.4's assumption (v) states the skew is charged zero and points at arm D. §H.3's "don't bundle POLL_BACKOFF" recommendation is **reversed**. |
| K5 (consumer placement) is materially perturbed: the pool's front-half sweep is a **quota** sized to slab-1's shadow (KERNEL:1601-1610, "1,024 batches"); at `T_eff=1792` it is 448 and the shadow shrinks by the same factor, so C=28/flush_rows=16 is no longer the tuned point — and R0a, pinned at T=4096, cannot detect it. (parity #7, hardware #11 for the adaptive-C half) | **FIXED.** §B.6's "touches none of the five knobs" claim **withdrawn** for K5. Because `C`/`flush_rows` are *runtime descriptor* fields, the fix is free: **arm E re-sweeps C × flush_rows at the fill operating point with no rebuild**, and §F.4's falsification thresholds are declared **invalid until arm E has run**. Adaptive-C (T2-d) stays default-0 pending that evidence. |
| `FILL=1 ZERO_PAD=0` is a permitted build and is newly unsafe: `out` is `torch.empty_like` in a captured pool, so `out[T_eff,4096)` holds uninitialised bytes that are **never written again**, permanently breaking `max_nonfinite = 0`. (parity #8) | **FIXED.** Verified `M23_RAGGED_SEAL_DESIGN.md:640, 615-620`; today KERNEL:736-743 rewrites all rows every step, so the hazard is genuinely new. **`#error` on `FILL && !ZERO_PAD && !ALLOW_DIRTY_PAD`**; `ALLOW_DIRTY_PAD` exists only for G8 to demonstrate the hazard. ZERO_PAD is now mandatory, not "recommended". |
| The per-layer-ness of `last_gen` is asserted, and `m15_runtime.py` — the file every allocation claim in §B.1/§B.2 cites — **is not in this repository**. If state is shared across layers the arm is inert on 57 of 58. (parity #9, hardware minor) | **FIXED.** Confirmed: `find` returns only `m15_eplb0/{m15_contracts.py, m15_vllm.py}`; there is no `m15_runtime.py` in the tree. All such citations are now marked **node-only and unverified locally**, and **G13a** is a blocking prerequisite that fetches the file and re-verifies M15-DESC-010/012 **and whether descriptors/`mps_state`/`epoch_cell` are per-layer**. The per-layer requirement is also now structural rather than assumed: M24 allocates **one buffer per layer** and its device tail is our own memory. |
| §A.4's +25–40 % headline is a per-STEP MoE computation presented as a per-RUN wall speedup; the "independent cross-check" compares improvement-over-self against speedup-over-stock; the published table does not reproduce from its own formula (~3-4 points low, computed at r≈1.0); the model predicts +2.3 % at ρ=1 where the measurement is −7.84 %; κ = 1.192 is never priced in; and the baseline is patched-stock, not native. (parity #5-e2e, hardware #9, measurement #4, measurement minor) | **FIXED — §A.4 rewritten as rev 2, and the headline is revised DOWN.** All five sub-claims verified. The frame error is named explicitly (`r ≈ 0.95` is composition-corrected per-step and must never be multiplied into a per-run wall model). The corrected model works in the m15 arm's own frame and divides by the baseline once. New honest headline: **+16 % to +31 % vs rescued/patched-stock, central ≈ +22 %; UNKNOWN vs native production until G16 measures it.** The cross-check is relabelled an internal consistency check (+26 %, not +34 %). φ's derivation corrected to 0.919 with 0.90 kept as a deliberate rounding-down. The exp_33 denominator corrected to 5,852.14 µs. **κ is promoted to a co-equal work item (§H.9)** and the design now states it may not clear the 20–30 % target alone. |
| `production_µs(N)` "must be flat" is flat **by construction** and can detect nothing — `NORIG_CONST` is candidate-only and `K0_T` is held at 4096. (measurement #7) | **FIXED.** Verified `run_campaign.sh:122-126`. Control **deleted** and replaced with four that work: R0b, the per-pin resource tuple + no-spill gate, the poison count over `[0,N)`, and one deliberate `K0_T = 2048` cross-check. |
| Fitting `a + b·N` measures φ only if the response is affine, and §B.6's own T2-e says it is not in the region sampled (M8 ticket degeneracy at low N); each point is a separately compiled pin, adding codegen scatter directly into `a`. (measurement #6) | **FIXED.** The affine fit is relabelled a **descriptive summary, not a φ measurement**. φ is reported as an interval from the extreme points with fit residuals as an explicit curvature diagnostic; T2-e must be evaluated before trusting the low points; the cross-pin resource-tuple check bounds codegen scatter; ladder extended below the grain. |
| `NORIG_CONST` makes `T_eff` a compile-time constant, so the MoK binary is not the shipping binary, and it compiles out the entire risk surface — the handshake, the buffer read, the grid-uniformity — which then first executes on 8 live GPUs. (measurement #4-specialisation) | **FIXED.** Semantics corrected: in a `FILL=1` build `T_eff` **always** comes from a runtime load of `agreed_T_eff` across the grid barrier, which cannot be folded. `NORIG_CONST`/`NORIG_TABLE` change only the published **value**, through an `asm volatile` opacity barrier. §E adds a **cross-pin resource-tuple equality** check to catch any leaked specialisation. |
| There is no inertness ratchet for the fill machinery: R0a runs on the all-macros-0 pin, so a +400 µs plumbing regression hides inside an in-band 0.42 ratio. R0a is nearly a tautology once G6's `.text` gate passes. (hardware #4, measurement minor) | **FIXED.** New **R0b**: `FILL=1, ZERO_PAD=1, NORIG_CONST=4096` must reproduce **0.7556 ± 1 %**. Blocking, alongside R0a, in G10. A `NORIG_CONST=3968` point prices the ZERO_PAD tail. |
| `M15_DESCRIPTOR_WORDS` is the default `expected_words` for every validation, so bumping it to 72 changes host behaviour for default-0 control arms and voids the `m15/patched-stock` leg. (measurement #8) | **FIXED.** Verified `m15_contracts.py:152, 1290, 1314`. The module constant **stays 63**; a separate `M15_DESCRIPTOR_WORDS_M24 = 72` is passed explicitly on M24 builds only, and the `:1314` compat branch is extended. G12's done-when is that a default-0 arm's descriptor path is provably unchanged. |
| The bit-identity gate consequences: G9's row-restricted **exact** comparison and F.4's "ratio < 0.30 ⇒ run the correctness gate" would fail or pass by luck. (parity #2, second half) | **FIXED** with B2: the restricted gate is **rel-err**, read against G0a's envelope. |

### R.3 Minor — triage

| finding | disposition |
|---|---|
| No validation of the *values* in `n_orig[]`; `round_up` can overflow before the `min`, giving `T_eff < 0`, zero output rows, clean `pperr`, stale `out`. (parity minor) | **FIXED** — this one is more than cosmetic. `round_up_sat` + `clamp(0, desc[K0P6_D_T])` + `n_orig[p] ∈ [0,MAXTOK]` validation + a telemetry bit; `STRICT` hard-fails. §B.3 calls the clamp load-bearing. |
| §D's unconditional `#define K0P6_M15_D_LEN 72` is a **macro redefinition** in the STAGED/REPLICATE/ADAPTIVE/SLOTPOOL branches of an exclusive cascade; the default build also jumps 63→72 leaving slots 63-70 uninitialised. (hardware minor) | **FIXED** — verified KERNEL:239-282. Rewritten as an `#undef`/raise **after** the cascade, with an `#error` on slot collision and a host zero-fill requirement for slots 63-70. G1's done-when adds "no macro-redefinition warning at every macro combination". |
| G5's guarded range `1460-1476` swallows `k0p6_sort::pad` at 1460, which §A.2 lists as its own region — a later partial-skip arm would inherit a stale-`sti` bug. (parity minor (a)) | **FIXED** — verified KERNEL:1460 vs 1474. G5 now guards **1474-1476 only** and explicitly excludes 1460, with the reason recorded. |
| TGRAIN's cost is the rounding of the mean, not the mean of the roundings; on the bimodal low mode the penalty is up to 8×, and the ladder's {3072,2048,1024} points are all far above the grain. (hardware minor) | **FIXED** — §B.3 restates the cost as `E[round_up(n,256)]/E[n]`, G0b computes it, the ladder gains points at {512, 256, 64, 0}, and `TGRAIN = 64` is named as the fallback compromise. |
| ZERO_PAD's "7 µs for 33 MB" implies 4.7 TB/s, contradicting the document's own 1.57 TB/s combine measurement, and ignores contention with M8's reads; §B.6 rates its risk "none". (hardware minor) | **FIXED** — restated as ≈ 21 µs at the cited rate, flagged as contending with M8, **measured** in G4 rather than asserted, and §B.6's risk column changed from "none" to "low". |
| The exp_33 ledger sums to 5,852.14 µs but the §A.2 column is headed 5,823 µs; §A.4's stated construction yields φ = 0.919, not 0.90. (measurement minor) | **FIXED** — denominator corrected to 5,852.14 with the reason; φ = 0.919 shown, 0.90 kept as an explicit conservative rounding. |
| `run_campaign.sh:143-144` is the wrong citation for the MAXTOK derivation (it is the `K0_MOK_ROUTE_HIST` env block); the substantive R6 correction is right. (measurement minor) | **FIXED** — corrected to **122-126** with the three `-e` lines quoted. This is my error, not the reviewer's. |
| The `NORIG_CONST=0 + NULLWORK` prediction is not well-defined until G7 decides the `T_eff = 0` clamp, so the arm is unfalsifiable as stated. (measurement minor) | **FIXED** — §F.4 states **two conditional branches** (unclamped `< 300 µs`; clamped `600–900 µs`), and G7's done-when now requires resolving to one branch **before** the arm runs. |
| R0a is a tautology w.r.t. M24 once G6 passes; it arbitrates pin/config/stale-checkout only. (measurement minor) | **FIXED** via R0b (see R.2). R0a is retained — it still catches the pin trap and a stale node checkout, which are the failures that have actually voided arms here. |
| §H.8's "composes on paper" overlooks the M0.5 pre-pass histogramming the full padded count. (parity minor (b)) | **FIXED** — this is substitution site 5; §H.8 now names all three composition defects explicitly. |
| Every other citation spot-checked (kernel lines, `m15_contracts.py`, `m15_vllm.py`, `moe_mps_adapter.cuh`, OVERLAP_ABSTRACTIONS, M23 §3.2/§3.6, DECOMP_RUNBOOK) verified as stated. (hardware minor, closing note) | **Acknowledged**, no action. |

### R.4 What changed most, in one paragraph

The mechanism is intact — `T` should mean real rows, and the megakernel can take a structural
advantage the captured production graph cannot. What the review destroyed was the **safety
story** and the **headline**. The staleness guard was inert and failing open; it is rebuilt on
device-owned per-layer state, a device-sourced payload, a single-reader broadcast across the
existing barrier, a checksum, a clamp, mandatory `STRICT`, and a post-hoc receipt audit. The
bit-identity guarantee was false, and the kernel turns out not to be bit-reproducible at all;
the accuracy gate is rebuilt on a measured run-to-run envelope. The MoK ladder could have
reported a clean pass for a kernel that computed nothing; it gains a live poison detector, a
working self-test, an inert-arm arbiter (R0b), a heterogeneous-fill arm (D) and a C re-sweep
(E), and loses a control that was flat by construction. And the headline falls from **+31 %
against a baseline we shouldn't quote** to **≈ +22 % against the secondary baseline and unknown
against the one that counts** — which means M24 alone probably does not clear the mission
target, and §H.9's step-count defect has to be worked in parallel. That last sentence is the
most useful thing the review produced.

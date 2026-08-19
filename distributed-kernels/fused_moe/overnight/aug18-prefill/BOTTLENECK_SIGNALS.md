# BOTTLENECK_SIGNALS — forensics on the first honest c32p serving pair

**Inputs:** `p1_stock/` (arm `1_stock`, pair `m23pair1`) and `p2_m15/` (arm `1_m15`, pair
`m23pair2`), each `c32p.json` + `c32p_client.log` + `seal_receipts.txt`.
**Tool:** `analyze_serving_pair.py` in this directory (runnable, stdlib-only, re-runs this
entire document on any future pair).
**Scope:** local artifacts only. No node, no GPU, no device timestamps.

```
python3 analyze_serving_pair.py <stock_dir> <m15_dir> --json out.json
```

---

## 0. Verdict up front

The brief's back-solve — "-8% e2e ⇒ mega MoE region ≈ 1.15–1.20× production" — **assumes both
arms did the same amount of MoE work.** They did not.

Over the 300 receipt-checkpointed steps the two arms carried near-identical real traffic
(3.250 M vs 3.315 M node tokens, −2.0%) but the m15 arm ran the padded B4096 graph on
**249.1 of 300 steps vs stock's 205.9** (+21.0%). Because a B4096 step charges all 8 DP ranks
4096 rows regardless of how many are real, m15 executed **1.192× the MoE row-work per real
token delivered**. At an MoE fraction f ≈ 0.44 that inflation alone is worth +8.4% wall — the
entire measured gap.

Three independent estimators, none of which share an error model, all say the mega's padded
step costs **the same as or slightly less than** stock's:

| estimator | what it uses | mega padded-step ratio |
|---|---|---|
| two-state wall fit (§6.1) | client wall times + in-bucket counts, no timestamps | 1.000 (exact arm-independent fit) |
| TPOT ceiling (§6.3) | the pure padded-step regime of the per-request TPOT tail | **0.9885** |
| matched receipt windows (§6.2) | log timestamps, stock-calibrated | 0.85 – 0.99 |

Composition-corrected, the mega MoE region lands at **r ≈ 0.95, interval [0.78, 1.06]**
(§7). The uncorrected number, 1.20–1.27, is an artifact of the confound. **The serving data
does not show the mega kernel losing.** It shows the m15 *run* scheduling its prefill chunks
19% less densely across the DP group.

The largest number in this whole analysis is not the A/B at all: **62.4% of every MoE row the
m15 arm computed was padding** (50.8% for stock). At f≈0.45 that is ≈65 s of m15's 217.6 s
wall, and ≈43 s of stock's 200.5 s, spent on rows that do not exist.

Secondary but flagged: **116 of 1024 requests (11.33%) produced different output tokens
between the arms.** The harness is called "exact-token"; it is not exact across this pair.

---

## 1. Provenance and what is comparable

| | stock (`p1_stock`) | m15 (`p2_m15`) |
|---|---|---|
| label | `stock_pair1_c32p` | `m15_pair1_c32p` |
| generated_utc | 2026-08-18T06:31:59.017 | 2026-08-18T07:00:27.962 |
| inferred start (= generated − wall) | 06:28:38.50 | 06:56:50.39 |
| wall | 200.512 s | 217.577 s |
| warmup wall | 6.191 s | 4.867 s |
| workload | c=32, n=1024, ISL 4096, OSL 8, `ignore_eos`, T=0, seed 320802 | identical |
| prompt sha256 | `b6271f04…1fda1d` | identical (bit-identical prompts) |
| completed / failed | 1024 / 0 | 1024 / 0 |
| input tok/s (node) | **20,917.95** | **19,277.36** |
| request throughput | 5.1069 /s | 4.7064 /s |
| ordered output stream sha | `f108d352…` | `73df1b75…` (**differs**) |

Wall ratio m15/stock = **1.0851** (+8.51%); tok/s ratio = **0.9216** (−7.84%).

**Patch-revision asymmetry (checked, benign).** The stock arm logs from
`gpu_model_runner.py:4088/4172`, the m15 arm from `4119/4213`, and only the m15 receipts carry
`refused_peer_not_ready` — the stock arm ran the pre-`eeff2c74` M23 patch. That commit removed
`m23_serving` from `b4096_unanimous`. It does **not** invalidate the cross-arm receipt
comparison: every counter increment sits inside `if m23_serving:`, so within the counted
region old and new `b4096_unanimous` are the same boolean. It also does not change scheduling
(the uniform rescue is gated on `b4096_unanimous and m23_serving` in both revisions). The
`in_bucket` gap in §5 is therefore batch composition, not a counter redefinition.

**Caveat that does bite:** receipts stop at step 300 of ~380–390. Everything full-run is an
extrapolation of the 300-step prefix. The patch already registers an `atexit`
`RAGGED_SEAL_RECEIPT_FINAL` — **it is not in these artifacts.** Grabbing it removes the
extrapolation entirely and is the cheapest single improvement to this analysis.

---

## 2. Per-request distributions (task item 1)

### 2.1 TTFT (ms)

| | n | mean | sd | p10 | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| stock | 1024 | 2296.54 | 441.42 | 1819.90 | 2288.68 | 2650.90 | 3540.29 | 5196.94 |
| m15 | 1024 | 2316.72 | 487.53 | 2024.95 | **2173.76** | 2939.12 | 3916.26 | 5414.73 |
| ratio | | 1.0088 | | 1.1127 | **0.9498** | 1.1087 | 1.1062 | 1.0419 |

**The m15 median TTFT is 5.0% FASTER.** This is the single most diagnostic number in the
per-request data. A kernel that is 15–20% slower on the MoE region of every padded step cannot
lower any TTFT quantile — every prefill chunk rides exactly such a step. A *scheduler* that
packs fewer prefills per step can, and does: each individual prefill waits fewer steps for a
slot (median TTFT down) while the node burns more padded steps in total (throughput down).
This is the classic batching latency/throughput trade, and it points squarely at composition.

Q-Q ratio (m15/stock) across the TTFT distribution — non-monotone, straddling 1:

```
 q      stock    m15    ratio          q      stock    m15    ratio
  1     832.99  1259.39  1.5119        60    2308.19  2217.06  0.9605
  5    1564.90  1534.46  0.9806        70    2340.85  2263.97  0.9672
 10    1819.90  2024.95  1.1127        80    2387.71  2804.96  1.1748
 20    2224.63  2075.19  0.9328        90    2650.90  2939.12  1.1087
 30    2255.29  2104.87  0.9333        95    3096.11  3032.94  0.9796
 40    2277.14  2142.03  0.9407        99    3540.29  3916.26  1.1062
 50    2288.68  2173.76  0.9498     spread: min 0.9328 max 1.5119
```

m15 is faster over q20–q70 and slower over q80–q99. Not a uniform multiplicative cost.

### 2.2 TPOT (ms) — the step-time probe

Each output token costs one model step, so TPOT is a 7-sample average of the step-time
distribution the request lived through.

| | n | mean | sd | p10 | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| stock | 1024 | 566.18 | **116.17** | 396.94 | 564.52 | 692.13 | 772.24 | 790.84 |
| m15 | 1024 | 637.46 | **96.87** | 509.54 | 669.42 | 722.44 | 767.13 | 786.36 |
| ratio | | 1.1259 | 0.834 | **1.2837** | 1.1858 | 1.0438 | **0.9934** | **0.9943** |

Q-Q ratio, monotonically decreasing from 1.28 to 0.99:

```
 q     stock     m15   ratio       q     stock     m15   ratio
  1   335.61  312.69  0.9317      60   617.23  693.14  1.1230
  5   361.07  457.98  1.2684      70   653.15  703.15  1.0765
 10   396.94  509.54  1.2837      80   662.33  711.83  1.0747
 20   458.66  563.89  1.2294      90   692.13  722.44  1.0438
 30   514.57  606.57  1.1788      95   743.57  739.13  0.9940
 40   553.07  625.46  1.1309      99   772.24  767.13  0.9934
 50   564.52  669.42  1.1858    spread: min 0.9317 max 1.2837
```

Read this as a mixture, not a shift. m15's TPOT distribution is **compressed** (sd 96.9 vs
116.2, a 17% narrowing) and pinned to the same upper end. m15 lost its cheap steps (p10 up
28%) and gained nothing at the expensive end (p99 down 0.7%). That is exactly what raising the
in-bucket step fraction from 68.6% to 83.0% does to a two-state step-time mixture. It is not
what multiplying every step by 1.08 does — that would move p99 and max up by 8%.

### 2.3 e2e latency (ms)

| | mean | p10 | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| stock | 6259.87 | 5290.11 | 6280.64 | 6993.63 | 7681.13 | 8263.60 |
| m15 | 6778.98 | 5917.74 | 6904.14 | 7660.31 | 8308.50 | 8956.20 |
| ratio | 1.0829 | 1.1186 | 1.0993 | 1.0953 | 1.0817 | 1.0838 |

Q-Q spread 0.90–1.13, essentially flat at ≈1.09. This carries almost no information: in a
closed loop `mean(e2el) = wall × concurrency / n` identically (stock 6259.87 × 1024/32 =
200.3 s ≈ wall 200.51; m15 6778.98 × 1024/32 = 216.9 s ≈ wall 217.58). Only the *shape* is
informative, and the shape says the gap is uniform in e2e while being strongly non-uniform in
its TTFT and TPOT components.

---

## 3. Reconstructed timeline and phases

The harness is a closed loop of 32 workers pulling requests in index order, so the schedule is
fully determined by a greedy earliest-free-slot replay of the per-request e2e latencies. The
reconstruction closes to **0.00% residual against the reported wall on both arms** (200.51 s,
217.58 s), which validates the dispatch-order assumption.

| phase | n | stock ttft_p50 / tpot_p50 / e2e_p50 | m15 ttft_p50 / tpot_p50 / e2e_p50 | e2e ratio |
|---|---|---|---|---|
| ramp (first wave) | 32 | 3083.6 / 373.9 / 5700.5 | 3142.6 / 444.2 / 6252.6 | 1.0969 |
| steady | 960 | 2288.4 / 569.6 / 6288.5 | 2171.8 / 674.8 / 6908.0 | 1.0985 |
| **drain (last wave)** | 32 | 2258.9 / 469.4 / 5544.2 | 2168.1 / 500.3 / 5582.1 | **1.0068** |

**The drain shows no gap (+0.7%).** The drain is the phase where ranks run out of work and go
idle — precisely where `execute_dummy_batch` peers appear and the all-8-eager fallback fires.
If mechanism (d) were material, the gap would be *largest* here. It vanishes. The gap tracks
prefill density instead: it is present in ramp and steady (both ≈+9.8%) and absent in drain.

Time-binned profile (10 bins over the reconstructed span, e2e p50 by dispatch time):

```
bin |  A t_lo   A e2e_p50   A rps |  B t_lo   B e2e_p50   B rps | e2e B/A
  0 |     0.0     5323.0   4.788 |     0.0     5804.9   4.412 |  1.0905
  1 |    20.1     5457.5   6.184 |    21.8     6618.8   4.872 |  1.2128
  2 |    40.1     5951.5   5.386 |    43.5     6373.0   5.377 |  1.0708
  3 |    60.2     6282.7   5.037 |    65.3     6389.7   4.550 |  1.0170
  4 |    80.2     6237.9   5.386 |    87.0     6974.6   4.734 |  1.1181
  5 |   100.3     6844.4   4.638 |   108.8     7468.8   4.320 |  1.0912
  6 |   120.3     7113.3   4.439 |   130.5     7023.4   4.504 |  0.9874
  7 |   140.4     6975.6   4.189 |   152.3     7221.9   4.458 |  1.0353
  8 |   160.4     6722.2   4.788 |   174.1     6943.7   4.550 |  1.0330
  9 |   180.5     6243.3   6.184 |   195.8     6992.1   5.240 |  1.1199
```

Bin-to-bin ratio ranges 0.99–1.21 with no trend — the gap is **not** concentrated in ramp-down,
which is the second independent argument against mechanism (d).

---

## 4. Receipt arithmetic (task item 2)

### 4.1 Step-rate trajectory (log wall clock, ±1 s quantization)

| checkpoint | stock time | Δ | steps/s | m15 time | Δ | steps/s |
|---|---|---|---|---|---|---|
| start | 06:28:38.50 | | | 06:56:50.39 | | |
| steps=100 | 06:29:26 | 47.5 s | 2.105 | 06:57:44 | 53.6 s | 1.865 |
| steps=200 | 06:30:27 | 61.0 s | 1.639 | 06:58:49 | 65.0 s | 1.538 |
| steps=300 | 06:31:36 | 69.0 s | 1.449 | 07:00:00 | 71.0 s | 1.408 |
| end | 06:31:59.02 | 23.0 s | — | 07:00:27.96 | 28.0 s | — |

The 0→100 window includes the 32-request warmup; use 100→300. Both arms slow down monotonically
through the run (stock 61→69 s, +13%; m15 65→71 s, +9%) as batch fill declines (§4.3).

Node real-token throughput per window (rank 0 `sum_orig` deltas):

| window | stock Δtok | stock tok/s | m15 Δtok | m15 tok/s | stock/m15 |
|---|---|---|---|---|---|
| 100→200 | 1,096,690 | 17,978.5 | 1,058,043 | 16,531.9 | 1.0875 |
| 200→300 | 1,100,204 | 15,945.0 | 1,084,599 | 15,276.0 | 1.0438 |
| 100→300 | 2,196,894 | 16,899.2 | 2,142,642 | 15,754.7 | **1.0726** |

The window-based gap (+7.3%) agrees with the whole-run gap (+7.8%/+8.5%).

### 4.2 Final (step-300) counters, all 8 ranks

**stock** — the mega never runs; every in-bucket step is refused for not-ready, as designed:

```
steps                 [300 ×8]
in_bucket             [205,205,206,206,206,206,206,207]   mean 205.88
sealed                [0 ×8]
refused_not_unanimous [13,11,17,14,15,13,15,14]           mean 14.00
refused_not_ready     [205,205,206,206,206,206,206,207]   mean 205.88
eager_b4096           [0 ×8]
uniform_rescued       [99,96,105,98,104,102,103,105]      mean 101.50
sum_orig              [3312067,3312067,3316162,3314930,3316154,3316154,3316154,3320236]
in_bucket_sum_orig    [2916841,2916841,2920944,2919712,2920936,2920936,2920936,2925029]
```

**m15** — coverage is real, and the receipt is green on its own success criterion:

```
steps                    [300 ×8]
in_bucket                [250,249,249,249,250,248,249,249]  mean 249.12
sealed                   [247 ×8]      sealed_ragged [245 ×8]   sealed_exact [2 ×8]
refused_not_unanimous    [7,7,5,7,6,7,7,8]                  mean  6.75
refused_not_ready        [3,2,2,2,3,1,2,2]                  mean  2.12
refused_peer_not_ready   [3,2,2,2,3,1,2,2]                  mean  2.12   (== not_ready: all peer)
eager_b4096              [3,2,2,2,3,1,2,2]                  mean  2.12
uniform_rescued          [142,138,139,144,140,138,135,141]  mean 139.62
sum_orig                 [3249465,3255026,3255026,3255026,3268638,3229945,3244089,3244089]
in_bucket_sum_orig       [3083410,3069790,3069790,3069790,3083410,3044680,3058832,3058832]
```

Derived:

| quantity | stock | m15 |
|---|---|---|
| in-bucket step fraction | 0.68625 | **0.83042** |
| sealed / in_bucket | 0 | 0.99147 |
| sealed / steps (mega duty) | 0 | 0.82333 |
| token coverage P1 = `in_bucket_sum_orig/sum_orig` | 0.88080 | **0.94374** |
| node real tok / step (all steps) | 11,051.6 | 10,833.9 |
| node real tok / in-bucket step | 14,184.7 | 12,312.4 |
| **per-rank real tok / in-bucket step** | **1,773.1** | **1,539.0** |
| **fill fraction of the 4096 bucket** | **0.4329** | **0.3757** |
| **padding fraction** | **0.5671** | **0.6243** |
| eager_b4096 / in_bucket | 0.0000 | 0.00853 |
| uniform_rescued / steps | 0.33833 | 0.46542 |

### 4.3 Fill trajectory per 100-step window

| window | stock rank tok/ib-step | stock fill | m15 rank tok/ib-step | m15 fill |
|---|---|---|---|---|
| 100→200 | 1,707.9 | 0.4170 | 1,438.0 | 0.3511 |
| 200→300 | 1,513.3 | 0.3695 | 1,371.5 | 0.3348 |

Fill declines through the run in both arms — the padded waste, and therefore the per-step cost,
grows as the run proceeds. This is why the shared-(a,b) model fits across arms but not across
windows within an arm (§6.2).

### 4.4 Hypothesis (e), stated correctly

`in_bucket` means `num_tokens_across_dp == [4096]×8` under a PIECEWISE synced mode — i.e. DP
padding put **all 8 ranks** in the 4096 bucket because at least one rank had a full chunk.
**Both arms run the padded 4096-row graph on these steps.** The mega does not pad more than
production does; the padding is a property of the deployment, not of the kernel.

What the padding *is*, quantitatively (m15, step-300 window):

- mean real rows per rank on a sealed step: **1,539 of 4,096 → 37.6% fill**.
- rigorous bound on how much of the bucket is below-full: with mean fill 0.3757 and the
  in-bucket predicate forcing ≥1 of 8 ranks to the top of the bucket, the fraction of
  *rank-steps* carrying fewer than 4096 real rows is in **[62.4%, 87.5%]**.
- fraction of sealed steps that are fully packed on all 8 ranks: **≤ 37.6%**, and given
  `min_orig=1` is observed, effectively zero.
- mega duty: 247/300 steps = **82.3% of steps, 94.4% of tokens**.

Full-run extrapolation from the 300-step prefix:

| | stock | m15 |
|---|---|---|
| est. total steps (= 4,194,304 / node tok per step) | 379.5 | 387.1 |
| est. in-bucket steps | 260.4 | 321.5 |
| padded MoE rows (in-bucket × 8 × 4096) | 8.533 M | **10.535 M** |
| real prefill tokens | 4.194 M | 4.194 M |
| padded-row waste | **50.8%** | **60.2%** |
| prefill chunk-events per in-bucket step (of 8 possible) | **3.93** | **3.19** |

At f≈0.45 and a ≈721 ms padded step, m15 spends **≈65 s of its 217.6 s wall** computing MoE on
padding; stock spends **≈43 s of 200.5 s**.

---

## 5. Cross-arm structural comparison (task item 3) — the confound

Same 300 steps, same traffic, different shape:

| | stock | m15 | m15/stock |
|---|---|---|---|
| steps | 300.00 | 300.00 | 1.0000 |
| in-bucket steps | 205.88 | 249.12 | **1.2101** |
| off-bucket steps | 94.12 | 50.88 | 0.5405 |
| real node tokens | 3,315,490 | 3,250,163 | 0.9803 |
| real node tokens in-bucket | 2,920,272 | 3,067,317 | 1.0504 |
| padded rows (in-bucket) | 6,746,112 | 8,163,328 | 1.2101 |
| total MoE rows (padded ib + real off-bucket) | 7,141,331 | 8,346,174 | 1.1687 |
| **padded MoE rows per real token** | **2.154** | **2.568** | **W = 1.1922** |
| wasted rows | 3,825,840 | 5,096,011 | 1.3320 |
| wasted fraction of MoE rows | 0.5357 | 0.6106 | 1.1397 |

m15 spent **27 extra padded steps** in the 100→300 window (179 vs 152) to deliver **3.3% more**
in-bucket tokens.

The cause is co-scheduling density: 1024 prefill chunk-events exist in both runs (1024 requests
× one 4096-token chunk), and the only question is how many of the 8 DP ranks issue one on the
same step. stock manages 3.93; m15 manages 3.19. Since a B4096 step charges all 8 ranks
whether or not they have work, de-clustering is pure loss.

`uniform_rescued` scales with `in_bucket` in both arms and is not itself asymmetric:
stock 101.50/205.88 = **49.3%** of in-bucket steps, m15 139.62/249.12 = **56.0%**. The rescue is
armed identically in both arms (design R6); it is a symptom of the shape, not a driver.

**Is the de-clustering caused by the mega, or is it run-to-run noise?** The serving data cannot
tell — n=1 per arm, and this statistic has never been measured twice. It is the #1 experiment
(§9). Either answer is actionable: noise ⇒ the pair must be re-run composition-matched before
any A/B claim; mega-induced ⇒ the −8% is real but is an *integration/scheduling* cost, not a
kernel cost, and is fixed in the scheduler.

---

## 6. Three independent estimators of the mega's padded-step cost

Model: step time is two-state. `a` = an in-bucket (padded B4096) step, `b` = any other step.

### 6.1 Two-state fit on the client wall times (no log timestamps)

```
stock : 260.4·a + 119.1·b = 200.51 s
m15   : 321.5·a +  65.7·b = 217.58 s
  =>    a = 601.6 ms,  b = 368.1 ms
```

A **single arm-independent (a, b) reproduces both arms' wall clocks exactly.** The mega's padded
step is assigned zero penalty and nothing is left over. This is an exact 2×2 fit, so it
demonstrates *consistency*, not proof — but it uses only the two most precisely measured
numbers in the whole dataset (the client walls, ms-accurate) plus the receipt counts.

### 6.2 Matched receipt windows, calibrated on stock, delta read off m15

```
window 100→200:  stock 68 padded + 32 other = 61 s | m15 87 padded + 13 other = 64 s
window 200→300:  stock 84 padded + 16 other = 69 s | m15 92 padded +  8 other = 71 s
```

Calibrating `a` from stock at each assumed cheap-step cost `b`, then reading the mega's implied
per-step delta:

| b (ms) | 100→200: a, δ, ratio | 200→300: a, δ, ratio |
|---|---|---|
| 100 | 850.0, **−129.3**, 0.8479 | 802.4, −39.3, 0.9510 |
| 200 | 802.9, −97.2, 0.8790 | 783.3, −29.0, 0.9630 |
| 250 | 779.4, −81.1, 0.8959 | 773.8, −23.8, 0.9692 |
| 300 | 755.9, −65.1, 0.9139 | 764.3, −18.6, 0.9756 |
| 350 | 732.4, −49.0, 0.9331 | 754.8, −13.5, 0.9822 |
| 400 | 708.8, −33.0, 0.9535 | 745.2, −8.3, 0.9889 |

**δ is negative for every plausible `b` in both windows** — the mega's padded step comes out
8–150 ms *cheaper* than stock's. This estimator carries the ±1 s log-timestamp quantization
(±1.6% on a 64 s window), which is why §7 grids it.

Exclusion: to make the mega region 1.175× at f = 0.45 requires δ = +55 ms, i.e. window 100→200
would have to be 87 × 0.055 = **+4.8 s longer than observed** (68.8 s vs 64 s measured, against
±1 s quantization). The brief's 1.15–1.20× is excluded at roughly 5σ of the timestamp error,
conditional on this model and on stock's calibration.

### 6.3 Mean-TPOT cross-check (no timestamps at all)

Treating a decode token as a uniform draw from the step-time mixture:

```
stock : 0.6863·a + 0.3137·b = 566.2 ms
m15   : 0.8304·a + 0.1696·b = 637.5 ms
  =>    a = 721.3 ms,  b = 226.9 ms
```

Again a single arm-independent (a, b) reproduces both arms. `a` = 721 ms sits sensibly against
the observed TPOT ceiling (§6.4) of ≈760–790 ms, which is what a request whose every gap falls
on a padded step should see.

### 6.4 The TPOT ceiling — the cleanest single number

| | p95 | p99 | p99.5 | max | n(TPOT>750 ms) | mean(top 50) |
|---|---|---|---|---|---|---|
| stock | 743.57 | 772.24 | 773.84 | 790.84 | 45 | **764.07** |
| m15 | 739.13 | 767.13 | 769.84 | 786.36 | 29 | **755.30** |
| ratio | 0.9940 | 0.9934 | 0.9948 | 0.9943 | — | **0.9885** |

The top of the TPOT distribution is the regime where every one of a request's 7 gaps landed on
a padded B4096 step — a direct sample of `a`. **m15's ceiling is 1.15% LOWER**, not 8% higher.
And m15 should have far *more* requests in this regime (0.830⁷ = 25.9% vs 0.686⁷ = 7.2% under
independence), yet has fewer above 750 ms — because its ceiling sits lower.

Converting the ceiling ratio k = 0.9885 to a region ratio via r = 1 + (k−1)/f:
f = 0.40 → **0.9713**; f = 0.45 → **0.9744**; f = 0.50 → **0.9770**; f = 0.54 → **0.9787**.

---

## 7. Honest back-solve and confidence interval (task item 4)

### 7.1 The uncorrected back-solve (what the brief did)

`T_m15/T_stock = (1−f) + f·d·r`, d = mega duty:

| f | duty = 0.8233 (steps) | duty = 0.9437 (tokens) | duty = 1 |
|---|---|---|---|
| 0.40 | 1.2584 | 1.2254 | 1.2128 |
| 0.45 | 1.2297 | 1.2004 | 1.1891 |
| 0.50 | 1.2067 | 1.1804 | 1.1702 |
| 0.54 | 1.1914 | 1.1670 | 1.1576 |

Applying a 3% eager-step tax at a 1.5× whole-step eager penalty moves these down by ≈0.01.
Range **1.158–1.258**, bracketing the brief's 1.15–1.20. **This model is wrong** because it
sets W = 1.

### 7.2 The composition-corrected back-solve

`T/token = (1−f)·S + f·W·r` with the measured
R = 1.1069 (time per real token), S = 1.0201 (steps per real token), W = 1.1922:

| f | r (non-MoE ∝ steps) | r (non-MoE ∝ tokens) | r (naive, W = 1) |
|---|---|---|---|
| 0.40 | 1.0377 | 1.0630 | 1.2673 |
| 0.45 | 1.0175 | 1.0381 | 1.2376 |
| 0.50 | 1.0013 | 1.0181 | 1.2138 |
| 0.54 | 0.9905 | 1.0049 | 1.1980 |

### 7.3 Interval over every nuisance parameter

Grid of 300 composition-corrected estimates over f ∈ {0.40, 0.45, 0.50, 0.54}, b ∈ {0.10, 0.20,
0.30, 0.40} s, ±1 s timestamp jitter on both windows, and the two non-MoE scaling conventions:

```
min = 0.5504   p10 = 0.6959   p50 = 0.8931   p90 = 0.9817   max = 1.0630

  norm-backsolve   n=8    min 0.9905  med 1.0178  max 1.0630
  tpot-ceiling     n=4    min 0.9713  med 0.9758  max 0.9787
  window 100→200   n=144  min 0.5504  med 0.7788  max 0.9820
  window 200→300   n=144  min 0.8093  med 0.9350  max 1.0494
```

**Reported result: mega MoE-region ratio r ≈ 0.95, interval [0.78, 1.06].**
Weight the two timestamp-free estimators (norm-backsolve 0.99–1.06, TPOT ceiling 0.97–0.98)
above the window estimators, whose lower tail is driven by ±1 s quantization on small
in-bucket count differences. The banked replay numbers (0.756 balanced, 0.8467 skew-replay)
sit just below this interval; the naive 1.15–1.20 sits above it and is excluded.

The residual replay-vs-serving discrepancy to explain is therefore **≈0.95 measured vs
0.76–0.85 banked, i.e. 12–25% — not the 35–45% the brief assumed.**

### 7.4 Co-scheduling headroom (applies to both arms)

If prefill chunks were perfectly co-scheduled across the 8 DP ranks, 1024 chunk-events would
need only 1024/8 = **128** padded steps instead of 260/322, with the decode traffic (8192 tokens,
4 requests per rank in flight) needing ≈256 cheap steps:

| step costs | stock 200.5 s → | m15 217.6 s → |
|---|---|---|
| a = 602 ms, b = 368 ms | 171.2 s (**−14.6%**) | 171.2 s (**−21.3%**) |
| a = 721 ms, b = 227 ms | 150.4 s (**−25.0%**) | 150.4 s (**−30.9%**) |

**15–25% of node throughput is sitting in DP prefill co-scheduling, for either kernel.** That is
larger than any kernel delta in this project's ledger.

---

## 8. Mechanism ranking (task item 4, second half)

A new mechanism (f) dominates and must be added; (a)–(e) are re-ranked as explanations of the
*residual* and of the replay-vs-serving gap.

### (f) DP prefill de-clustering / padded-work inflation — **posterior: very high**

*Evidence.* W = 1.1922 measured directly from receipts (§5); explains +8.4% at f = 0.44 vs the
+8.5% observed; three independent step-cost estimators return δ ≤ 0 (§6); median TTFT *fell*
5.0% (§2.1), impossible under a slower kernel; the drain phase shows +0.7% (§3).
*Cannot yet distinguish:* chance vs mega-induced. n = 1 per arm.
*Decisive experiment:* **re-run both arms ≥3× and report `in_bucket/steps` and
`in_bucket_sum_orig/(in_bucket·8·4096)` per run.** If stock's 0.686 and m15's 0.830 are stable
and separated, the mega causes the de-clustering. If they overlap, the −8% is unmeasured and
the pair must be redone composition-matched. ~2 × 220 s per run; needs only the existing
receipts (plus the `RAGGED_SEAL_RECEIPT_FINAL` atexit line).

### (e) ragged batches / padding — **posterior: high, but reframed**

*Not* "the mega pads and production doesn't" — both arms run the identical 4096-row padded graph
on in-bucket steps. The true finding is that **62.4% (m15) / 56.7% (stock) of every padded batch
is padding**, [62.4%, 87.5%] of rank-steps are below-full, and the mega is at 82.3% step /
94.4% token duty over exactly these steps. This is the root cause of (f) and the most likely
root cause of (a) and (b).
*Mechanistic prediction that ties the whole picture together:* vLLM pads with token id 0, so the
2,555 padded rows per rank carry **identical hidden states → identical router logits → the same
top-8 experts.** 62% of all dispatched rows would then land on ≤8 of 256 experts. Order-of-
magnitude check: 1,539 real rows × 8 topk spread over 256 experts ≈ 48 rows/expert; 2,555 padded
rows × 8 concentrated on 8 experts ≈ 2,555 rows each; a rank holding one hot expert sees
2,603 + 31×48 = 4,091 rows vs 1,536 for a rank holding none — **2.66× rank-load skew from padding
alone**, rising to ~5× with two hot experts on one rank. The brief's measured per-call max-rank
load p50 of **5.15×** is quantitatively consistent with this and with essentially nothing else.
*Decisive experiment (free — the data may already exist):* run
`np.unique(topk_ids, axis=0)` over any captured `routes_rank<N>_pid<PID>.npz` from
`skewhook_v2`. If one row value occupies ~62% of the 4096 rows, the mechanism is confirmed with
zero GPU time. Follow with a replay arm that overwrites the duplicate-row block with fresh
random routes — the delta is the padding-degeneracy tax on the mega.

### (a) run-correlated destination concentration degrading LL128 dispatch / remote-RMW combine — **posterior: moderate**

*Evidence for:* the padding mechanism above supplies an extreme, deterministic concentration the
balanced and aggregate-histogram replays never contained; MORI's own 2-of-7-link pathology and
their 822→498 µs RR-interleave fix are the right order of magnitude for the residual 12–25%.
*Evidence against / limits:* the serving artifacts contain **no** device-side signal — no phase
stamps, no link counters. Nothing here observes it directly.
*Cannot distinguish from (b) with serving data at all.*
*Decisive experiment:* `K0_MOK_ROUTE_FILE` replay, `K0_MOK_ROUTE_ORDER=captured` vs `shuffled`
(bit-identical multiset, `route_multiset_sha256` asserted) with `K0_MPS_CFG timestamps=1`. A
captured-vs-shuffled delta isolated to M0–M2 dispatch and M8/M9 combine is (a). Then the fabric
arms `K0P6_M15_RMW_INTERLEAVE` / `K0P6_M15_POLL_BACKOFF` on the losing arm.

### (b) per-call compute concentration (hot-rank GEMM becomes critical path; fixed costs don't shrink) — **posterior: moderate**

*Evidence for:* same root cause as (a); the banked ledger says M6 (2,453 µs) + M7 (2,702 µs) is
~80% of the region, so a 5.15× max-rank load with a single-CTA planner, slab rendezvous and
C = 24–28 polling CTAs that do not shrink is a plausible 12–25%.
*Evidence against:* none available locally.
*Decisive experiment:* the same captured-route replay, but read the split: a captured-vs-shuffled
delta landing in **M6/M7** is (b); landing in **M0–M2/M8–M9** is (a). Both arms in one run.

### (c) per-sealed-step integration overhead now paid at 99% duty — **posterior: low–moderate, and bounded**

*Bound from this data:* the total composition-corrected per-padded-step delta is δ ∈ [−129, −8]
ms across every (b, window) combination (§6.2), i.e. **≤ 0 at every plausible parameter.** Any
launch_prepare / ring-slot / descriptor-write cost that scales per sealed step is therefore
bounded above by roughly the timestamp quantization, ≈±15 ms/step ≈ ±2% of a padded step.
*Cannot distinguish* the sign of a small positive integration cost hiding under a slightly larger
kernel win.
*Decisive experiment:* `K0_MPS_CFG timestamps=1` in serving, exp_33-style; compare the M0–M9
device ledger against the banked replay ledger at matched T. Host-side residual = integration.

### (d) all-8-eager fallback on dummy-peer steps — **posterior: low; the one mechanism serving data closes**

*Measured, not estimated:* `eager_b4096` = 2.12 of 300 steps per rank = **0.71% of steps**,
identically equal to `refused_not_ready` = `refused_peer_not_ready` (so every eager step is a
dummy-peer step, exactly the class `eeff2c74` was written to make safe). Extrapolated: 2.7 eager
steps in the full run.

| eager whole-step penalty | added wall | % of 217.58 s |
|---|---|---|
| 2× | 1.98 s | 0.91% |
| 3× | 3.96 s | 1.82% |
| 5× | 7.91 s | 3.64% |

Two independent falsifiers on top of the bound: the **drain phase gap is +0.7%** (§3) and the
time-binned ratio shows no ramp-down concentration (§3) — dummy peers are a drain phenomenon.
*Decisive experiment:* none warranted. If wanted, `VLLM_PF4H_B4096_UNIFORM_RESCUE` and a forced
dummy-peer step with device stamps would price one eager step exactly.

### Summary of what serving data alone can and cannot do

| | can | cannot |
|---|---|---|
| **(f)** composition | measure W exactly, and δ three ways | attribute cause without n>1 |
| **(e)** padding | count fill, duty, bounds exactly | see routing degeneracy (needs the route npz) |
| **(a)** destination concentration | nothing | separate from (b); no device or link signal |
| **(b)** compute concentration | nothing | separate from (a); needs the M0–M9 stamp split |
| **(c)** integration overhead | bound |δ| ≲ 15 ms/step | resolve its sign |
| **(d)** eager fallback | count it (0.71%), bound it (<3.6%), and falsify it twice | — |

---

## 9. Ordered next actions

1. **(free, now)** `np.unique(topk_ids, axis=0)` on any existing `skewhook_v2` route capture.
   Confirms or kills the padded-row routing-degeneracy mechanism with zero GPU time.
2. **(free, now)** Pull `RAGGED_SEAL_RECEIPT_FINAL` (the atexit line the M23 patch already
   registers) from both runs' logs. Removes every full-run extrapolation in §4.4 and §7.
3. **(2 × 220 s × 3)** Repeat the pair n ≥ 3 per arm; report `in_bucket/steps` and fill per run.
   Settles whether (f) is mega-induced or noise, and puts an error bar on the −8% for the first
   time. **Nothing else should be concluded from this pair until this runs.**
4. **(~139 s × 4)** Captured-route replay grid: `captured` vs `shuffled` × mega vs production,
   with `K0_MPS_CFG timestamps=1`. Splits (a) from (b) and quantifies the residual 12–25%.
5. **(design)** DP prefill co-scheduling. 15–25% of node throughput (§7.4), kernel-independent,
   and it is also the fix that restores the replay's routing conditions (§8e) — the single
   highest-leverage item in the ledger.
6. **(correctness)** Explain the 116/1024 (11.33%) output-token divergence. Greedy decoding at
   T = 0 with `ignore_eos`; per-request hashes are identical for 908 requests, so this is
   near-tie logit flips, but the "exact-token" harness name should not survive an unexplained
   11% divergence.

---

## 10. Reproduction

`analyze_serving_pair.py` regenerates every number above:

```
python3 analyze_serving_pair.py p1_stock p2_m15 --json pair1_analysis.json
python3 analyze_serving_pair.py --pair <dir-with-two-arm-subdirs>
  --dp-size 8         DP group size used for the per-rank token arithmetic
  --f-grid 0.40,0.45,0.50,0.54    MoE-region fraction grid for the back-solves
```

Sections: 0 headline + exact-token agreement · 1 distributions · 1b Q-Q ratios · 2 timeline
reconstruction + phases · 2b time bins · 3 receipt arithmetic · 3b matched-window comparison ·
4 naive back-solve · 5 padded-work accounting · 5b composition-normalised back-solve ·
5c/5d/5e/5f the three step-cost estimators · 6 interval synthesis · 7 co-scheduling headroom.

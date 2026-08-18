# COST_MODEL — M(W, S; θ): the analytic model, its θ, its calibration plan, and the decision boundaries it predicts

**Output (3) of `../BRIEF.md:118-119`.** Written 2026-08-18, laptop, no node access.
Consumes the four grounding briefs in `../grounding/`:
`BANKED_LAWS.md` (LAW-n citations below), `KNOB_INVENTORY.md` (S, K1–K5, S0–S8, F1–F10),
`WORKLOAD_EVIDENCE.md` (W1–W10, H1–H17), `TP8_STATE.md` (§n, M1–M9, A1–A6/B1–B8).

**Binding rules for this document.**
1. Every number carries a source: a repo `path:line`, a git sha, the BRIEF digest, or a
   grounding-brief section. Arithmetic performed *here* is tagged **DERIVED** and its inputs
   are shown so it can be re-checked.
2. **Fudge factors are named and bounded.** The model contains exactly **two** free
   coefficients that are not derivable from a measured quantity — `I₀` (§1.6) and `κ_couple`
   (§1.7). Both are named, bounded by measurements, given a scaling hypothesis, and assigned
   the specific experiment that removes them. Nothing else in M is fitted without a mechanism.
3. **Predictions in §3 are registered before the runs.** They are the campaign's scientific
   spine. Each carries a falsifier.
4. One provenance correction up front: the task brief cites a "13,923 → 1,851 µs
   epoch-publication result". **That number does not exist anywhere in this repository**
   (searched all `.md`, `.hip`, `.cuh`, `.py`; the only `13923` hits are rocprof timestamp
   fields in `aug12/exp_01_neutral_transport/raw/host_copy_anchor_trace_v1.json`). It is
   **not used**. The protocol-cost terms in §1.5 are anchored instead on the census that does
   exist: the ~926,000 protocol ops/rank/epoch itemised at `aug11/STATUS.md:1187-1200` and the
   exp_14 removal rate at `aug10/experiments/LESSONS.md:677-687`.

---

## 0. What M is, what it is for, and what it deliberately is not

### 0.1 The estimand

**LAW-64 is binding:** the estimand is the *synchronized global makespan of the complete
dependent DAG* — median over iterations of the **max across ranks** latency
(`aug12/LESSONS.md:5-11`; `aug18/T3_COUNTER_DATAFLOW_DESIGN.md:39-40`). Bandwidth and TFLOPS
are explanatory quantities inside M, never its output.

M predicts a **per-layer makespan** `T_layer(W, S; θ)`. Everything above that — TTFT, TPOT,
throughput, goodput — is a *lift* through the serving stack (§1.10), and the lift is where
the sign can flip (LAW-44: median TTFT fell 5.0% while node throughput fell 7.84% in the same
pair). M is therefore explicitly **not** a throughput model; a throughput claim requires M
*plus* the metric map *plus* a coverage receipt.

### 0.2 The claim M encodes

> **Workload W acts on the optimal schedule S\* only through a small vector of mediating
> quantities Q(W). Given Q and the hardware/protocol constants θ, S\* is computable.**

This is the falsifiable core, and §5 states exactly what would refute it. If the claim holds,
the campaign's deliverable is a *law* ("change X, schedule changes by Y, because Q crossed
θ"); if it fails, the deliverable is a regime map, which is a weaker but still honest result.

### 0.3 The three vectors

**W — workload state vector** (coordinates with measured spans in `WORKLOAD_EVIDENCE §6`):
`(topology, concurrency, ISL, OSL, arrival process, phase[prefill|decode|mixed], T, fill φ,
routing skew σ, top-k, world size, arch, bucket size, coverage)`.

**S — schedule state vector** (`KNOB_INVENTORY` Parts A–C):
`(skeleton ∈ S0..S8; K1 carrier/transport op; K2 producer order; K3 certification granularity;
K4 injection depth; K5 consumer pool C; flush_rows; grid geometry; fusion boundary F1..F10)`.

**Q — the mediating vector.** This is the model's central compression claim: **W enters M
only through these eight scalars.**

| # | symbol | definition | computed from W by | laws that move on it |
|---|---|---|---|---|
| Q1 | `T_eff` | **real** rows per rank per step (= `φ·T`) | fill receipt × bucket size | LAW-38, 39, 40, 42 |
| Q2 | `co ∈ {0,1}` | is the carrier co-resident with an MFMA body? | skeleton + fusion boundary | LAW-7, 8, 20, 22 |
| Q3 | `A`, `L` | protocol event count per epoch; cache-lines per event | plan geometry (rows × chunks × blocks), signal design | LAW-16, 17, 18, 29 |
| Q4 | `φ_p` | CTA-throughput-bound fraction of phase *p* | phase shape (weight-stream vs compute-dense) | LAW-13, 14, 15, 27 |
| Q5 | `u(t)` / `Φ` | consumer-unblockable fraction; its time-integral | **top-k** and K2×K3 | LAW-23, 24 |
| Q6 | `σ` | max-rank load ratio (routing + placement + padding) | route corpus + placement map | LAW-42, 46, 47, 52 |
| Q7 | `m` | message size per collective per boundary | T × row bytes × topology | LAW-2, 5, 9, 51 |
| Q8 | `f`, `W_pad`, `κ` | MoE fraction of a step; padded rows per real token; DP co-scheduling density | serving receipts | LAW-40, 41, 44, 50 |

**The compression is the contribution.** W has ≥8 coordinates that are individually
unmeasurable in combination (`WORKLOAD_EVIDENCE §6`: we have **two** serving points on that
space). Q has 8 scalars, all of which are *directly instrumentable* — six of them already are.

---

## 1. MODEL STRUCTURE

### 1.1 Outer form

```
T_step(W, S; θ)  =  Σ_{ℓ ∈ layers}  T_layer(ℓ)  +  H(S)                      … (M0)

T_layer          =  MAX_{r ∈ ranks}  T_layer^(r)                              … (M1)   [LAW-52]

T_layer^(r)      =  Σ_{p ∈ P(S)}  [ C_p  +  X_p  +  Q_p  +  I_p ]  −  O(S)    … (M2)
```

* `P(S)` — the **phase set is a function of the skeleton**, not of the workload. For S4
  (M15 DP/EP): `{dispatch M0–M2, plan M3–M5, G1 M6, G2+carry M7, combine M8/M9}`
  (`k0pf6gm_device_tile_m15.hip` phase markers, `KNOB_INVENTORY §C.1`). For S6 (M25 CDAR
  TP8): `{attn/dense, ERS#1, MAG#1, MoE-G1, MoE-G2+ERS#2, MAG#2}`.
* `C_p` compute term (§1.3), `X_p` wire term (§1.4), `Q_p` protocol term (§1.5),
  `I_p` co-residency interference (§1.6), `O` overlap credit (§1.7).
* `H(S)` — host/launch term. Bounded independently at `|δ| ≲ 15 ms/step ≈ ±2% of a padded
  step` for the serving integration (LAW-55; `aug18-prefill/BOTTLENECK_SIGNALS.md:559-568`).
  **Treated as a bounded constant, not modelled.**

**M1 is not decoration.** LAW-52 (critical-rank law) says iteration pace is the hottest /
slowest rank's wall; scheduling helps e2e only when it removes work from *that* rank. The
`MAX` is where σ (Q6) enters, and it is why §1.8 is a multiplier on the whole layer and not a
term inside one phase.

### 1.2 Sign convention on the overlap term

`O` is subtracted *once, at layer scope*, never phase-by-phase. Reason, measured: **LAW-21** —
carrier relocation and injection bound are non-separable and never additive (+210.6 µs
regression for relocation alone, −615.0 µs for the joint cell). **LAW-30** — M7 and combine
are anti-correlated at `r = −0.904`; "combine's 324 µs is M7's slack, not work". A waterfall
of per-phase overlap credits is arithmetically wrong for this kernel family and M refuses to
express one.

### 1.3 Compute term: the capacity-elasticity form

```
C_p(n)  =  C_p(N) · [ (1 − φ_p)  +  φ_p · N / n ]      n = N − C                … (M3)
```

`N` = grid CTAs (256, occupancy 1 — LAW-12, `aug10/KERNEL_BOTTLENECKS.md:24-27`), `C` =
reserved pool, `φ_p ∈ [0,1]` = the CTA-throughput-bound fraction of phase p (Q4).

**Calibrated directly from LAW-15's reserve-but-idle sweep** (`aug10/KERNEL_BOTTLENECKS.md:61-67`),
which removed 62 of 256 CTAs (`N/n = 256/194 = 1.320`):

| phase | measured Δ | ⇒ φ_p (DERIVED) |
|---|---|---|
| plan + M6 (weight-streaming, latency-bound) | +0.5% (2,954 → 2,970 µs, **flat** across C=2..64) | **φ ≈ 0.016 ≈ 0** |
| M7 (payload-carrying GEMM) | +27% (1,586 → 2,020 µs, **linear in C**) | **φ ≈ 0.84** |
| combine | ~flat | **φ ≈ 0** |

*Cross-check against an independent measurement*: (M3) with `φ_M7 = 0.84`, `C_M7(N)=1,586 µs`
gives `dC/dC_ta = 1,586·0.84/256 = 5.2 µs per reserved CTA` at small C. LAW-27 measured the
idle-pool tax at **8.1–9.3 µs/CTA** with the pool provably idle three ways
(`aug11/LESSONS.md:445-452`). Same order, and the residual is expected because at the M15 pin
M7 is larger than 1,586 µs. **One coefficient per phase, both measured, no fitting.**

**Roofline sanity, so φ is not mistaken for a bandwidth story** (`aug10/KERNEL_BOTTLENECKS.md:120-124`,
MEASURED times / INFERRED byte counts): M7 reads ~470 MB of weights in 1,586 µs =
**~296 GB/s against 8 TB/s HBM**; the pool moves ~312 MB over xGMI in ~2.8 ms =
**~111 GB/s against ~537 GB/s aggregate egress**. *Neither phase is near a roofline.* The
compute term is issue/latency/capacity-limited, not bandwidth-limited — which is exactly why
(M3) is written as a capacity elasticity and not as `work/bandwidth`.

**Where the MFMA rate enters.** exp_22 measured MFMA throughput **linear to 256 CTAs,
R² ≥ 0.99989, per-CTA efficiency 0.987–0.998 from 32→256 CTAs, no knee**
(`aug11/STATUS.md:390-400`). ⇒ for a compute-dense phase, `R(n) = ρ_mfma·n` exactly, so
(M3) with `φ=1` is not an approximation — it is the measured law. Absolute anchor:
**352.0 TFLOPS at shape-256, 15.3% of the 2,300 TFLOPS dense-bf16 spec peak**
(`aug11/STATUS.md:397`; `aug11/exp_22_fig7_saturation/result.md:125-132`). The MoE GEMM
bodies are not at that shape, so `ρ_mfma` for M6/M7 is a **θ entry that must be calibrated**
(§2, θ-C1), not read off the spec sheet — LAW-62(a) is explicit that a spec-sheet denominator
manufactures false refutations.

### 1.4 Wire term: BW_eff as a product of measured efficiencies

```
X_p  =  b_p / BW_eff                                                          … (M4)

BW_eff(dir, op, m, layout, L, n_CTA)
      =  BW_link · n_link · η_dir · η_op(m) · η_layout · η_size(m) · η_ctas(n_CTA)   … (M5)
```

Every factor is a *measured* multiplier, and each has a named domain:

| factor | measured value | source | domain / caveat |
|---|---|---|---|
| `BW_link` | **56.9–57.3 GB/s** achievable (74.1–74.6% of 76.8 nominal) | LAW-4, `aug11/LESSONS.md:207-217`; `aug11/STATUS.md:390-393` | per xGMI link, gfx950. **Never use 76.8** — LAW-62(a). |
| `n_link` aggregate | rr7 plateau **355.0 GB/s** = 66.0% of 537.6 nominal | `aug11/STATUS.md:393` | 7-link fan-out, knee at 32 CTAs at mlp4 |
| `η_ctas` | knee at **8 CTAs** single-link; **32 CTAs** rr7; `C=8→64` buys **+2.4%** and then nothing | LAW-4 | ⇒ **pusher count is not a bandwidth knob above ~8–32** |
| `η_dir` (push/pull) | **0.9936× tie** at 64 KiB matched path; **1.078×** pull edge at 64 MiB single link | LAW-5, BRIEF `:63-65` | knees: pair 256 KiB; fan-out 64 KiB/peer (push) vs 1 MiB (pull); **pull collapses past 256 CTAs** (`aug12/OVERLAP_KERNEL_DESIGN_ADDENDUM.md:317-321`) |
| `η_op(m)` small | atomics **52.8** ≈ stores **54.9 GB/s** ⇒ 0.962 | LAW-2 | per-link, coalesced, small message |
| `η_op(m)` bulk | atomics cap **67.1** vs store-towers **117.2 GB/s** ⇒ 0.572 | LAW-9, `tp8_mega/results/g25_0b_slab_sweep.txt` | 235 MB AR. Atomic cap diagnosed as **dword issue rate** |
| `η_op(m)` tiny | atomics **801 µs** vs towers **1,017 µs** at 3.67 MB ⇒ **reverses** | `TP8_STATE §5/M5` | towers *lose* below the reduce-amortization point (§3, D3) |
| `η_layout` | coalesced **52.8** → scattered **4.1 GB/s** = **0.078 (13×)** | LAW-3, BRIEF `:58` | *"layout is a schedule decision"* |
| `η_size(m)` | 128:116.0 / 256:117.2 / 512:93.4 / 1024:55.4 GB/s by `slab_rows` | LAW/`g25_0b_slab_sweep.txt` | **non-monotone, interior optimum** — modelled in §3/D4 |
| arch multiplier | gfx942: remote atomics **~3× slower** than stores (15 vs 44–46 GiB/s) | `docs/distributed/OVERLAP_ABSTRACTIONS.md:146` | flips `η_op` sign; §3/D3 |

**Structural decision, and it is falsifiable:** *injection depth `d` does NOT appear in (M5).*
LAW-8/F6 measured depth **flat** across {4,8,16,32} in pure transport (towers 93.6/93.4/93.3;
atomics 67.1/67.1/67.0 — `g25_0b_v1_results.txt`). If depth acted on the wire, it would show
there. It does not. Depth therefore enters M **only through `I_p`** (§1.6), with coefficient
identically zero when `co = 0`. This is the model's sharpest structural commitment and §3/D5
states how to break it.

**Fabric-efficiency denominator.** Converting a wall to an efficiency requires per-rank egress:
one 235 MB two-shot AR moves `7/8·235 + 7·235/8 = 411 MB` per rank per boundary — identical to
ring's `2(N−1)/N·B` (`TP8_STATE §1.3`, DERIVED). On that denominator CDAR's best pure transport
= **205 GB/s = 58%** of the 349–355 GB/s egress card; RCCL = **280 GB/s = 79%**.
**CAVEAT that must ride any use of this:** the 349–355 GB/s ceiling is flagged
*"unreproduced; treat as unknown, measure first"* (`OVERLAP_ABSTRACTIONS.md:147`).

### 1.5 Protocol term

```
Q_p  =  ι · A_p(L)  +  n_bar · c_bar  +  n_cert · c_cert  +  W_spin              … (M6)
```

* `A_p` = protocol RMW/flag **count** per rank per epoch. **LAW-17: only reducing the count
  helps; relocation cannot.** Measured rate: exp_14 removed **698,112 of ~1.58 M
  atomics/rank/epoch for −306 µs of M7** (`aug10/experiments/LESSONS.md:677-687`)
  ⇒ **`ι = 0.438 µs per 1,000 protocol atomics`** (DERIVED). Ceiling implied: removing *every*
  remaining arrival atomic is worth **≤235 µs against +1,249 µs outstanding — ≤19%**
  (LAW-17, labelled INFERRED at source).
* The census that sets `A` (`aug11/STATUS.md:1187-1200`): `nc_arr` 535,040 + `pushed` ~314,000
  + `row_ready` ~19,600 + `row_rem` clean ~19,600 + M8 polls ~21,500 + `ev_next` ~16,700 =
  **~926,000/rank/epoch**. `A` is set by the **plan** (rows × chunks × contributing blocks),
  not by the schedule (LAW-16) — so `A` is a *workload*-driven quantity that scales with
  `T · nchunks · blocks_per_row` and with **top-k**.
* `L` = cache-lines per protocol event. **LAW-18 inverts the coalescing intuition**:
  transposing arrival counters so 32 lanes hit **2 lines instead of 32** made the kernel
  **≥10× slower** (abandoned after 12 min against a usual 25–95 s). ⇒ `ι` is a function of
  `L`, and the model uses `ι(L) = ι · (32/L)^ψ` with **ψ ≥ 1 measured only as a lower bound**
  (θ-P3). Second site, opposite sign: mode-13 consolidation 16→1 footprint **cost 150–240 µs**
  while deleting the `pushed` counter **saved 85–150 µs**.
* Certificate cost `c_cert`: replacing the ~926k per-row ops with **2×8 slab epoch words**
  (`slab.cuh:6-9`) was worth **−573.3 µs** of protocol work (`aug11/LESSONS.md:311-319`) —
  and note **LAW-37**: that −573.3 and the depth bound's −613.5 are suspected **substitutes**,
  never measured on top of each other. **M must not sum them** (§5, F4).
* `c_bar` (grid barrier): **NOT MEASURED in-repo.** Structural count is known — *four grid
  barriers sit between M2 and M6* (`aug10/experiments/LESSONS.md:272-280`) and the pre-M6
  region is a "sum of maxima". θ-P4 is the microbench that supplies it. This is a real hole,
  and it matters because fusion boundary → barrier count → drift-realization count
  (`T3_COUNTER_DATAFLOW_DESIGN.md:43-49`: three realizations per launch vs one per layer).
* `W_spin` (bounded-wait exposure): **LAW-31** — the dispatch all-to-all has **zero exposed
  wait under balanced routing** (max-spins-to-success **0** against a 2,000,000 limit,
  `fail_max = 0` across 10 exp_33 rotations + both exp_35 rungs). `W_spin ≈ 0` at σ=1 and is
  the term that *creates* the σ multiplier of §1.8 when σ > 1.

### 1.6 Co-residency interference — and named fudge factor #1

```
I_p  =  co · [ ι·A_p(L)·ξ_int   +   J(d)   +   I₀(skeleton) ]                    … (M7)
```

**`co` gates the whole term.** LAW-7: in isolation, peer-write traffic costs **+624 µs vs
+38 µs (rank read) / +105 µs (local write)** — a **519 µs fabric surcharge at 5.96×** — while
the real, throttled pusher never reaches that regime. LAW-8: depth is inert without
co-residency. LAW-22: pacing pays *exactly when the paced engine is not the only phase in
flight*.

**The interference is protocol, not payload** (LAW-16, two independent mechanisms):
mode 7 deletes the pool's entire 896 B payload copy and moves M7 by **+5.7 µs against a
±40 µs band while 831.5 µs of interference remains**; and interference rises monotonically
with group size `g` (+1,159.3 / +1,645.6 / +2,058.8 / +2,693.8 µs at g = 1/2/4/16) while
payload bytes are constant at ~312 MB. This **retracts** exp_17's "move fewer bytes".

**`I` is not linear in C, and is C-invariant in total volume.** LAW-14: marginal interference
falls **38.6 µs/CTA at C=16 → 18.1 at C=64** ("a shared resource being DISTURBED rather than
divided"); at C=16 the interference (618 µs) is **12× the capacity cost (50 µs)**. LAW-29:
mode-13 interference **1,328 µs @ C=16 vs 1,323 @ C=64**. ⇒ M models `I ⟂ C` and puts *all*
C-dependence in (M3). This is why **a small comm pool is not a safe default**.

**`J(d)` — the injection term (a step, not a curve).** exp_24 (LAW-20), M7 µs:
`d=4 → 2,659 | 8 → 2,742 | 16 → 3,204 | 32 → 3,112 | unthrottled → 3,189`.
⇒ `J(d) = 0` for `d ≤ 8`; `J(d) = J_hi ≈ +470 µs` for `d ≥ 16`, where `J_hi` equals the
unthrottled cost. **`d* = 8` is the cliff edge; `d = 4` beats `d = 8` by only −50.9 µs
(−0.77%, ~9σ).** Waterfall confirmation: unthrottled epilogue-carried **7,110.8** vs
`+depth-4` **6,495.8 = −615.0 µs, t = −80.2, ranges disjoint by 598 µs**
(`aug11/LESSONS.md:41-54`). **Depth 2 has never been measured** — the `g` `0x300` selector has
no room left; the left side of the cliff is unexplored.

> ### **NAMED FUDGE FACTOR #1 — `I₀`, the unattributed co-residency floor**
> **What it is.** At `g = 1` the probe loop is degenerate, yet **+1,159 µs of interference
> remains** with no identified mechanism (`aug10/KERNEL_BOTTLENECKS.md:129-131, 213-223`).
> Only **36% is localized** — to the per-XCD L2 (LAW-19: XCD confinement cut interference 36%
> but starved the pool 54%). The source calls the other 64% *"the single largest gap in our
> understanding of this kernel"*.
> **Bound.** `0 ≤ I₀ ≤ 1,159 µs` in the **S2 carrier-pool** regime at T=4096, 8 ranks,
> gfx950. In **S4 (M15 consuming pool)** the pool carries nothing, so the mechanism should be
> absent: `I₀(S4) ≈ 0` — **an assumption, and it is exactly what makes S2-vs-S4 a clean test.**
> **Scaling hypothesis (SPECULATION, to be tested).** `I₀ ∝ (peer-write request rate) ×
> (co-resident phase's memory intensity)`, i.e. it should vanish when the co-resident phase is
> cache-resident or arithmetic-bound (LAW-13's note that the interference term is a property
> of the memory system, scaling with memory intensity not FLOPs).
> **Elimination.** θ-P5: a sub-phase PMC method for a single persistent launch
> (`aug10/EXPERT_BRIEFING.md:104-123` names this as unsolved), or the M25 device phase ledger
> (`TP8_STATE §5/M1`) which measures the same class of tax at 335 µs on the phased CDAR arm.
> **Until eliminated, `I₀` is fitted per SKELETON, never per cell.** A model that needs a new
> `I₀` per workload cell has failed (§5, F2).

### 1.7 Overlap credit — the slack map, and named fudge factor #2

**The occupancy-1 selection rule is the model's gate on overlap**
(`OVERLAP_ABSTRACTIONS.md:42-60`): at 1 block/CU, 1 wave/SIMD, 256-VGPR bodies, there are no
spare waves to fill stalls and `vmcnt` is one in-order counter per wave. **Phase partitioning
is work-conserving** — moving full-grid work "under" a compute phase by CTA partition just
re-divides the same CTA·µs (exp_25 interleave: first-order tie; static split: −173 µs *loss*;
mode 16 cross-launch defer: **+75.8 µs, falsified**). Overlap pays through exactly **four
doors**, and M assigns a term to each:

| door | mechanism | model term | measured |
|---|---|---|---|
| (a) | latency-bound consumer into a compute shadow | `O_consume` below | M15 combine residue −144 µs; pool −93/−444 µs |
| (b) | **deleting** protocol work | `ι·ΔA` in (M6) | per-row → slab words: −573.3 µs |
| (c) | op-class swap on the fabric | `η_op` in (M5) | 67 → 117.2 GB/s (towers); 0.127 vs 0.571 µs/op issue ladder |
| (d) | bounding injection | `J(d)` in (M7) | depth 4: −615 µs |

**Any proposed schedule that goes through none of these four doors is predicted ≈ 0.** That is
a falsifiable selection function, and it is what M uses to *reject* arms before spending node
hours on them.

**The slack map (door (a)), closed form.** A consuming pool of `C` CTAs, running at the
measured latency-bound per-CTA rate `ρ` during a producer shadow of duration `τ`, can absorb
`ρ·C·τ` bytes, but no more than the **certified** volume `V_cert` that K2×K3 have made
available:

```
O_consume(C)  =  min( V_cert(K2,K3,Φ) ,  ρ · C · τ_shadow )  /  R_serial
                 +  κ_couple · min(C, C_sat)                                     … (M8)

C_sat  =  V_cert / (ρ · τ_shadow)                                                … (M9)
```

**θ inputs, all measured** (`aug11/exp_29_pipelined_combine/design.md:100-110`): combine
DRAM-real work `W = 644 MiB = 675 MB`; measured combine ≈ **430 µs** ⇒ `R_serial = 1.57 TB/s
= 20% of the 8 TB/s HBM3E peak` ⇒ **`ρ = 6.1 GB/s per CTA`**. `τ_shadow` = M7 window =
**2.66 ms**. M15's pool sweeps **front halves** (columns [0, 3584) of 7168 —
`k0pf6gm_device_tile_m15.hip:2063-2064, 2171`) ⇒ `V_cert ≈ 675/2 = 337 MB`.

**⇒ `C_sat = 337 MB / (6.1 GB/s × 2.66 ms) = 20.8 CTAs`  (DERIVED, no fitting).**

**Validation, immediate.** The measured M15 C-ladder is `16 → 6,292.4 | 24 → 5,848.5 |
28 → 5,822.0 | 32 → unstable` (BRIEF `:50-51`): a **−444 µs** step from 16→24 and a **−26.5 µs**
step from 24→28. The model predicts a steep region up to `C ≈ 21` and a flat one after.
**Predicted knee 21, measured knee 24 — inside one grid step of a {8,16,24,32} sweep.** The
same formula also reproduces exp_29's own negative result: `675 MB / (6.1 GB/s × 16 CTAs) =
6.9 ms ≫ 2.66 ms` ⇒ *"a pool-only combine cannot cover the work at any legal C"*
(`design.md:107-110`) — i.e. the pool must take the **front half only**, which is exactly the
design M15 shipped.

> ### **NAMED FUDGE FACTOR #2 — `κ_couple`, the producer-coupling term**
> **Why it exists.** The pure relocation accounting under-predicts the *magnitude*. Each CTA
> absorbs `ρ·τ = 16.2 MB`, worth `16.2 MB / 1.57 TB/s = 10.3 µs` of post-M7 combine. Measured
> marginal value over C∈[16,24] is **−444/8 = −55.5 µs/CTA** — a **5.4× under-prediction**.
> **Mechanism hypothesis (grounded, not invented).** LAW-30 measured `r = −0.904` between M7
> and combine with a joint sd of 32.1 µs against the 100.5 µs independence predicts:
> *"combine's 324 µs is M7's slack, not work."* So a consuming CTA does not merely relocate
> work — it also releases the producer's own rendezvous/drain. The rate model cannot see that.
> **Bound.** `0 ≤ κ_couple ≤ 55 µs/CTA` in the S4 regime for `C ≤ C_sat`; **identically 0**
> outside a certified prefix (which is why mode 12, with `V_cert ≈ 0`, has `C* ≤ 8` and the
> pool is pure tax — `OVERLAP_ABSTRACTIONS.md:120-123`).
> **Elimination.** The device phase ledger that separates M7's *drain wait* from its *MFMA
> span* (θ-P6). This is the same instrument LAW-30 itself calls for, and it is the same
> instrument `TP8_STATE §5/M1` names as the next step for M25. **One instrument removes both
> fudge factors' ambiguity.**

**`Φ` — the certified-shadow integral, and where top-k enters.** Under the natural GEMM-2
task order, `P(token ready by t) = (t/S)^k` with `k = top-k`, and the **median token is not
reducible until 91.7% of GEMM-2 completes** (`aug11/OVERLAP_METHODOLOGY_STUDY.md:277`; note
`2^{-1/8} = 0.9170` — the model reproduces the measured constant exactly). Define the
normalized certified-volume-time integral:

```
Φ_natural(k)    =  ∫₀^S (t/S)^k dt / S  =  1/(k+1)                               … (M10)
Φ_slab(S_cnt)   =  (S_cnt − 1) / (2·S_cnt)                                       … (M11)
V_cert          =  V · Φ                                                          … (M12)
```

At `k=8`, `Φ_natural = 0.111`; at `S_cnt=2` (M15's shipped `K0P6_M15_SLABS 2`),
`Φ_slab = 0.250` — a **2.25× larger consumable shadow**. This is the arithmetic behind
"order alone ≈ 0; order × certification = −144 µs" (`OVERLAP_ABSTRACTIONS.md:80`): order
without certification cannot be observed by a consumer, and certification without order
certifies a prefix that is empty.

**(M11) also bounds K3 from above:** `Φ_slab → 0.5` as `S_cnt → ∞`. **No readiness-granularity
refinement past S=2 can be worth more than 2× the consumable shadow**, while its protocol cost
grows linearly in `S_cnt` via `ι·A`. That is the quantitative form of `slab.cuh:10-12`'s
"granularity is the free axis; EXPOSURE is the deadly one" — and it **resolves the open
contradiction** flagged as SPECULATION in `KNOB_INVENTORY §B.3`: in M15 a 64× signal-count
change moved ~nothing because it only moved `Φ` inside a range bounded by 2×; in CDAR
`slab_rows` moved 2.1× because it changes **transfer geometry and grid occupancy**
(`n_slabs × bps` blocks), which lives in `η_size` of (M5), not in `Φ`. **Two different terms,
two different laws, no contradiction.**

### 1.8 Skew multiplier — the critical-rank law, calibrated

```
T_layer(σ)  =  T_layer(σ=1) · [ (1 − λ) + λ·σ ]                                  … (M13)
```

`σ` = max-rank receive-load ratio (Q6); `λ` = the fraction of the layer that is
**receive-row-proportional** (the expert GEMMs + combine on the destination side). (M13) is
the analytic consequence of (M1): the hottest rank's row-proportional work scales by σ while
its rank-invariant work does not.

**Calibration — one parameter, fit on one point, validated on a held-out point.**

Inputs (all MEASURED): M15 replay ladder `balanced 5,823 | aggregate-histogram 20,739 |
worst-layer 22,475 µs` (`aug14/M18_REPLICATION_RESULTS.md:24-31`); σ from the stock0814
aggregate histogram under linear placement `max/mean = 5.088× aggregate`,
`5.593× worst-rank layer` (`aug18-prefill/rr_placement/RR_PLACEMENT_NOTES.md:208-236`).

| step | arithmetic | result |
|---|---|---|
| fit λ on the **aggregate** point | `λ = (20,739/5,823 − 1)/(5.088 − 1)` | **λ_M15 = 0.6266** |
| **predict** the held-out worst-layer point | `5,823 × [0.3734 + 0.6266×5.593]` | **22,582 µs** |
| measured | | **22,475 µs** — **error +0.47%** |
| same procedure, production arm | `λ_prod = 0.5322`; predict 26,567 vs actual 26,259 | **error +1.17%** |
| **ratio** prediction at worst layer | `22,582/26,567 = 0.850` | measured **0.8559** — error **−0.7%** |

**DERIVED, one free parameter, sub-1% held-out error on two arms.** This is the strongest
single validation the model currently has, and it was obtained *before any new measurement*.

**Three consequences that fall straight out:**
1. **λ_M15 (0.627) > λ_prod (0.532).** Our kernel is *more* row-proportional, so **skew hurts
   us more than it hurts production** — independently predicted by the fairness audit's note
   that the mega's uniform-capacity padded work flatters it under balanced routing
   (`aug18/FAIRNESS_AUDIT.md:58`). Two independent routes to the same sign.
2. `dT/dσ = T(1)·λ`. At `T(1)=5,823 µs`, **each unit of σ costs 3,649 µs/layer.** Compare the
   entire schedule-knob ledger: the largest single knob (depth-4) is **615 µs**. **Skew is
   worth 3.6× wall (5,823 → 20,739 µs); it dominates every schedule knob by ~6×.**
3. Under (M13), the *ratio* between two arms is nearly σ-flat when λ's are close — which is
   why placement fixes are enormous **absolute** wins and near-zero **relative** wins (§3, D9).

**Which σ to use is an open contradiction (H17).** m18diag (n=90,050 router calls) gives
per-call max-rank load **p50 5.15× / p95 6.25×**; the routecap1 corpus (512 rank-calls, real
rows only) gives **p50 2.61× / p90 3.35×**. They differ by ~2× and **must not be mixed**. M
carries σ as an interval `[2.61, 5.15]` at c32p until reconciled (CPU-only work item), and
every σ-dependent prediction below is stated as a band.

### 1.9 Fill correction — and the unification of the fill law and the T law

The kernel executes `T` padded rows and sees `T_eff = φ·T` real ones. Without M24 the kernel
has **no idea how many rows are real** (`FILL_AWARE_DESIGN.md:88-91`: `T` is written once into
descriptor slot `K0P6_D_T` out of capture). So:

```
without M24 (today):   T_layer = T_layer(T),        cost per REAL token = T_layer(T)/T_eff
with    M24 FILL:      T_layer = T_layer(T_eff)                                   … (M14)
```

**The fixed/variable decomposition is what makes T_eff the master mediating quantity.** Fit the
affine form `T = F + v·T` on the tgen T-sweep endpoints (T=1024 and T=4096) and predict the
middle point:

| arm | F (fixed µs/epoch) | v (µs/token) | predicted T=2048 | measured | error |
|---|---:|---:|---:|---:|---:|
| megakernel (M15-C=28 body) | **1,068.0** | **1.1623** | 3,448 | 3,412 | **+1.07%** |
| production (unfused) | **181.3** | **1.8387** | 3,947 | 3,857 | **+2.31%** |
| ratio | | | **0.8737** | **0.8845** | −1.2% |

Source of the three points: `exp_04_tgen`, three 5-rotation campaigns, gates green
(`git show 43291977:.../aug13/exp_04_tgen/result.md:226-300`; `WORKLOAD_EVIDENCE §W6`).
**Provenance trap carried forward, binding:** these were measured on the **tgen body**
(sha `20b8c6bb`), which exists *because* the serving M15 chassis has a `T≠4096` defect; the
serving body **failed the MoK gate at T=2048/1024 on 2026-08-18**. The affine law is
`fused-megakernel-family`-scoped, not M15-serving-scoped.

**The cost model in one sentence:**
> **Fusion buys a per-token rate (`Δv = 0.676 µs/token`) and pays a per-epoch fixed cost
> (`ΔF = 887 µs`). The crossover is their ratio.**

```
T*  =  (F_mega − F_prod) / (v_prod − v_mega)  =  886.7 / 0.6764  =  1,311 tokens   … (M15)
```

The repo's own reading of the same sweep states break-even at **≈1,600–1,800 tokens/rank at
C=28** (`WORKLOAD_EVIDENCE §W6`). (M15)'s affine derivation gives **1,311**; a linear-in-T
interpolation of the measured ratios gives **1,483**; a linear-in-log₂T interpolation gives
**1,399**. **The campaign's registered band is `T* ∈ [1,300, 1,800]` and locating it is
prediction P1 (§3, D1).**

**The unification.** Because M enters only through `T_eff`, the fill law and the T law are
*the same law*:

* c32p, m15 arm: `φ = 0.3757` ⇒ `T_eff = 1,539 real rows` — **just above** T\*=1,311, deep
  inside the uncertainty band. ⇒ **prediction: at c32p a fill-aware megakernel sits within
  noise of break-even on real work**, and the ~56% of steps that are pure dummy have
  `T_eff ≈ 1` — arbitrarily deep in the loss region. That is precisely why the corpus analysis
  concluded tier-1 (whole-step skip) captures the entire modeled win and tier-2 (row-granular)
  adds nothing (`REPORT.md:94`) — **M reproduces that conclusion from first principles.**
* c512p: `φ ≈ 0.985` ⇒ `T_eff ≈ 4,035` — 3× above T\*, deep in the win region. ⇒ **+8.40%**
  and **+6.60%** measured (geomean 1.0750 over the two order-balanced pairs;
  `WORKLOAD_EVIDENCE §W8.2`).

**Fill's shape matters more than its mean, and M says why.** The corpus fill distribution is
**bimodal — deciles of `n_real` = 0, 0, 5, 1994, 4096, 4096, … ; ~31% of calls near zero,
~60% completely full, only ~10% genuinely partial** (`G0B_LOCAL_FILL_HISTOGRAM.md §2`).
Because `T_layer(T_eff)` is affine and the crossover is a *threshold*, a bimodal `T_eff` means
almost every step is decisively on one side or the other. **The mean φ is the wrong statistic;
the mass above/below T\* is the right one.** M therefore consumes the *distribution* of
`T_eff`, not its mean:

```
E[T_step]  =  Σ_layers  E_{T_eff ~ 𝔇(W)} [ T_layer(T_eff) ]                       … (M16)
```

**SPECULATION carried from `WORKLOAD_EVIDENCE §W3`, cheap to test:** the bimodality is
probably an artifact of `ISL 4096 == max_num_batched_tokens 4096` (one chunk per request), and
**dissolves at any other ISL** — never measured (H4). If so, every tier-1 sizing number is
cell-specific and `𝔇(W)` is not transferable.

### 1.10 Lifting T_layer to serving metrics

M outputs a layer makespan. The serving stack converts it, and **the conversion is where
signs flip**:

```
step wall     =  (1 − f)·T_nonMoE  +  f·T_MoE(M)                                 … (M17)
node tok/s    =  Σ real tokens / Σ (steps × step wall)   ← scaled by κ, coverage
TTFT          =  queueing ⊕ chunk co-scheduling ⊕ step wall
TPOT          =  decode-step wall × OSL
```

Binding facts the lift must carry:
* **`W_pad` (padded rows per real token) must be matched between arms or the A/B is invalid**
  (LAW-40): the m15 arm ran the padded graph on **249.1 of 300 steps vs stock's 205.9**
  (`κ = 1.1922`), and at `f ≈ 0.44` that inflation alone is worth **+8.4% wall — the entire
  measured gap** (observed +8.51%).
* Three composition-independent estimators put the mega's **padded step at ≤ stock's**
  (wall fit 1.000, TPOT ceiling 0.9885, matched windows 0.85–0.99) ⇒ corrected
  **r ≈ 0.95 [0.78, 1.06]** vs the naive 1.15–1.27 (LAW-41).
* **`f` is measured at ≈0.44–0.45** at c32p; it is a first-class θ entry, not a guess.
* **Coverage rides every quoted number** (LAW-49): post-M23 `sealed/in_bucket = 0.99147`,
  `sealed/steps = 0.82333`, token coverage 0.94374. Pre-M23 numbers are **retired**.
* **`κ` is a scheduler effect, not a kernel effect**, and it is the largest number in the
  c32p forensics: perfect DP co-scheduling would need **128 padded steps instead of 260/322**
  ⇒ **15–25% of node throughput, larger than any kernel delta in this project's ledger**
  (LAW-50). M carries κ as an explicit input and refuses to attribute it to S.

### 1.11 What M deliberately does not model

| excluded | why | where it is bounded instead |
|---|---|---|
| host launch/glue | LAW-55: the big host numbers are *backpressure*, not deletable glue | `|δ| ≲ 15 ms/step ≈ ±2%` |
| numeric nondeterminism | packed-bf16 remote atomics ⇒ not bit-reproducible; **116/1024 requests (11.33%) produced different tokens**; stock-vs-stock SHAs differ | accuracy gate, never a parity gate |
| multi-node | zero measurements (H10) | **explicitly scoped out** |
| decode as a distinct mechanism | H1: never measured in any rig; every kernel law above is a **prefill** law | §3/D3 and §5 name it as the biggest regime risk |
| thermal rank drift on the serving arm | root-caused only in training (GPU2 at 77 °C, **4–5% clock spread**, 92 µs→4,353 µs `rows_done` spread) | H16; folds into σ as an *unobserved* second skew source |

---

## 2. θ TABLE — every hardware/protocol parameter

**Legend.** `M` = measured & banked. `D` = derived here from banked inputs. `H` = hole; the
"microbench" column names the experiment that supplies it. Where possible the microbench is an
**axis of the existing G25-0b single-process 8-GPU harness**
(`tp8_mega/m25_boundary_bench.hip:441-448` — already the manifest schema for that skeleton).

### 2.1 Fabric / transport

| id | parameter | value | cls | source | microbench if H |
|---|---|---|---|---|---|
| θ-F1 | per-link achievable BW | **56.9–57.3 GB/s** (74.1–74.6% of 76.8) | M | LAW-4, `aug11/LESSONS.md:207-217` | — |
| θ-F2 | link knee (CTAs) | **8** single-link; **32** rr7 mlp4; C=8→64 buys **+2.4%** | M | `aug11/STATUS.md:390-393` | — |
| θ-F3 | aggregate 7-link plateau | **355.0 GB/s** = 66.0% of 537.6 | M | same | — |
| θ-F4 | egress ceiling | 349–355 GB/s | **H** | `OVERLAP_ABSTRACTIONS.md:147` — *"unreproduced; treat as unknown"* | G25-0b transport-only, all-pairs egress sweep |
| θ-F5 | atomic vs store, small msg | **52.8 vs 54.9 GB/s** (ratio 0.962) | M | LAW-2 | — |
| θ-F6 | atomic cap, bulk | **67.1 GB/s** (dword issue rate) | M | LAW-9, `g25_0b_v1_results.txt` | — |
| θ-F7 | store-towers, bulk | **117.2 GB/s** @256-row, 93.6 @512-row, 235 MB AR | M | `g25_0b_slab_sweep.txt` | — |
| θ-F8 | layout penalty | coalesced→scattered **13× (52.8 → 4.1 GB/s)** | M | LAW-3 | — |
| θ-F9 | push/pull | **0.9936×** @64 KiB; **1.078×** @64 MiB single-link | M | LAW-5 | — |
| θ-F10 | direction knees | pair 256 KiB; fan-out push 64 KiB/peer vs pull 1 MiB; **pull collapses > 256 CTAs** | M | `OVERLAP_KERNEL_DESIGN_ADDENDUM.md:317-321` | — |
| θ-F11 | RCCL AR reference @235 MB | **1,470 µs** = 280 GB/s per-rank egress = **79%** of ceiling | M | `G25_1_STATUS.md:14-19` | — |
| θ-F12 | our CDAR transport @235 MB | **2,214 µs** (in-rig) / 2,004 µs (pure) ⇒ **β = 1.506× RCCL** | M | same; `TP8_STATE §2.3` | — |
| θ-F13 | RCCL at **decode** sizes (2–64 rows) | — | **H** | designed as G25-0b(iii), **never run**; `rccl-tests` absent but the rig links `-lrccl` | **one argv** — `TP8_STATE §5/M5`. **Entire go/no-go for decode cells.** |
| θ-F14 | host peer-copy path | **1.454×** CU pull @64 KiB; CU-lowered, **not SDMA** | M | LAW-6 | — |
| θ-F15 | SDMA in situ | — | **H** | LAW/retraction #2: **never measured**; nothing supports or refutes it | mori CCO `ccoSdma` arm on the G25-0b harness |
| θ-F16 | gfx942 atomic:store | **~3× slower** (15 vs 44–46 GiB/s) | M | `OVERLAP_ABSTRACTIONS.md:146` | — |
| θ-F17 | gfx942 link/egress card | unknown | **H** | same table: *"unreproduced; treat as unknown, measure first"* | port θ-F1..F8 to a gfx942 node |
| θ-F18 | live xGMI counter | **does not exist** (`--xgmi` N/A, `--shownodesbw` 0-0) | M | LAW-11 | bounds every calibration promise; UMC duty ±15 pp is the fallback |

### 2.2 Compute / capacity

| id | parameter | value | cls | source | microbench if H |
|---|---|---|---|---|---|
| θ-C1 | MFMA scaling | **linear to 256 CTAs, R² ≥ 0.99989**, per-CTA eff 0.987–0.998 | M | `aug11/STATUS.md:397` | — |
| θ-C2 | MFMA absolute | **352.0 TFLOPS** shape-256 = 15.3% of 2,300 spec | M | same | — |
| θ-C3 | `ρ_mfma` for the M6/M7 bodies | — | **H** | not measured at the MoE tile shape | isolated-body timing at fixed T; **use for the roofline, never the spec sheet** (LAW-62a) |
| θ-C4 | φ_p (capacity elasticity) | plan+M6 **≈0**; M7 **0.84**; combine **≈0** | D | LAW-15 via (M3) | re-derive per skeleton and per T |
| θ-C5 | HBM knee / plateau | knee **128 CTAs**, **4,151.9 GB/s = 51.9%** of 8,000 | M | `aug11/STATUS.md:394` | — |
| θ-C6 | consumer rate `ρ` | **6.1 GB/s per CTA** (combine, latency-bound; 1.57 TB/s = 20% HBM) | M | `exp_29/design.md:100-110` | re-measure per consumer body |
| θ-C7 | occupancy | **1 block/CU** forced by 256 VGPR + 256 AGPR **and** 155,428 B LDS | M | LAW-12 | **must be asserted and printed per arm** — shrinking LDS silently raised it to 2 and voided an experiment |
| θ-C8 | phase profile (homogeneous, T=4096) | dispatch 1,193 (17%) / plan 415 (6%) / **M6 2,539 (36%)** / M7 1,586 (23%) / combine 1,255 (18%) | M | `aug10/KERNEL_BOTTLENECKS.md:49-56` | **LAW-65: re-profile after every landed win** — at the M15 ratchet M7 (46.2%) has already overtaken M6 (41.9%) |

### 2.3 Protocol

| id | parameter | value | cls | source | microbench if H |
|---|---|---|---|---|---|
| θ-P1 | `ι` atomic cost | **0.438 µs per 1,000** protocol atomics | D | LAW-17 (698,112 → −306 µs) | — |
| θ-P2 | `A` census (T=4096, top-8) | **~926,000** ops/rank/epoch (nc_arr 535,040 + pushed 314k + …) | M | `aug11/STATUS.md:1187-1200` | — |
| θ-P3 | `ψ` line-contention exponent | **≥1** (32→2 lines ⇒ **≥10× slower**, lower bound only) | **H** | LAW-18 | counter-transpose sweep at L ∈ {2,4,8,16,32} on the G25-0b harness |
| θ-P4 | `c_bar` grid-barrier cost | — | **H** | structure known (4 barriers M2→M6) but cost never measured | empty-phase barrier-count sweep in the mega chassis |
| θ-P5 | `I₀` floor | **≤1,159 µs** (S2, T=4096); 36% localized to per-XCD L2 | **H (bounded)** | LAW-16 residual, LAW-19 | sub-phase PMC (unsolved) **or** the M25 device phase ledger |
| θ-P6 | `κ_couple` | **≤55 µs/CTA** for C ≤ C_sat in S4; 0 without a certified prefix | **H (bounded)** | derived from LAW-30 + the C-ladder | device phase ledger separating M7 drain-wait from MFMA span |
| θ-P7 | `J_hi` depth penalty | **+470 µs** step at d ≥ 16 (d=4 → 2,659; d=16 → 3,204 µs of M7) | M | LAW-20 | — |
| θ-P8 | `d*` cliff edge | **8** on gfx950 (d=4 beats d=8 by only 0.77%); **d=2 never measured** | M | LAW-20; `credit.cuh:59-61` | widen the `g` `0x300` selector — the left side of the cliff is unexplored |
| θ-P9 | certificate saving | per-row → 2×8 slab words = **−573.3 µs** | M | `aug11/LESSONS.md:311-319` | **may be a substitute for θ-P7 (LAW-37) — never measured together** |
| θ-P10 | XCD placement penalty | confinement cuts interference **36%**, starves pool **54%**; net **7–23% worse** | M | LAW-19 | dispatch rule is `xcd = wg_id % 8`, chunk size 1 — **any placement law stated without it is backwards** |
| θ-P11 | ticket vs static stripe | up to **2.90×** (C=16: 22,068 → 7,617 µs) | M | LAW-26 | gain ∝ producer-completion variance ⇒ grows with skew |
| θ-P12 | co-residency protocol tax (TP8) | **~335 µs** on the *phased* CDAR arm before any fusion | D | `TP8_STATE §2.3` | attributed by nobody yet — device phase ledger |

### 2.4 Workload-side θ (measured distributions, not constants)

| id | parameter | value | cls | source |
|---|---|---|---|---|
| θ-W1 | `f` (MoE fraction of a step) | **≈0.44–0.45** at c32p | M | `BOTTLENECK_SIGNALS.md` |
| θ-W2 | `φ` fill | 0.3757 (c32p m15) / 0.4329 (c32p stock) / ≈0.985 (c512p); **declines within a run** 0.417→0.370 | M | `DECOMP_RUNBOOK.md:32-38`; `BOTTLENECK_SIGNALS.md:275-278` |
| θ-W3 | fill **shape** | bimodal: deciles 0,0,5,1994,4096,… | M | `G0B_LOCAL_FILL_HISTOGRAM.md §2` |
| θ-W4 | dummy-step rate | **56.0%** (c32p m15 whole run) / 49.3% (stock) / **2%** (c512p); 31.2% ramp-window | M | `BOTTLENECK_SIGNALS.md:341-343`; `LOG.md:345-350` |
| θ-W5 | σ aggregate | rank max/mean **5.088×** linear, **1.085×** RR; gini 0.609 | M | `RR_PLACEMENT_NOTES.md:208-236` |
| θ-W6 | σ per-call | **p50 5.15 / p95 6.25** (m18diag, n=90,050) **vs p50 2.61 / p90 3.35** (corpus, n=512) — **UNRECONCILED, ~2×** | M | H17 |
| θ-W7 | λ (row-proportional fraction) | **0.6266** (M15), **0.5322** (production) | D | §1.8 |
| θ-W8 | run correlation | +1.3 points of expert overlap vs shuffled (57.0 vs 55.7%) — **falsified as first-order** | M | `CORPUS_FINDINGS.md:23-27` |
| θ-W9 | κ (co-scheduling density) | **1.2101**; chunk-events/in-bucket-step **3.93 (stock) vs 3.19 (m15)** of 8 | M | `BOTTLENECK_SIGNALS.md:258-262` |
| θ-W10 | variance envelope | **±15% day / ±18% position** at c32p **vs ≤1.80% throughput / ≤0.25% latency** at c512p | M | `SERVING_BENCHMARK_METHODOLOGY.md:229-231`; `WORKLOAD_EVIDENCE §W10b` — **the envelope is itself a function of W** |
| θ-W11 | instrument resolutions | screens **~3%** smallest callable; stamps σ 1.1% (plan) / 2.3% (M7); campaigns reproduce to **0.09%** | M | LAW-60 |
| θ-W12 | campaign cadence | 5-rotation campaign (500 warmup/100 timed/600-epoch soak) = **~3 min 5 s** | M | LAW-60 |

---

## 3. DECISION BOUNDARIES — symbolic, then instantiated as registered predictions

Every boundary below is (i) an inequality over `Q` and `θ`, (ii) instantiated with banked θ,
(iii) turned into a **numbered prediction P#** for a cell we have **not** run, with a
falsifier. These are registered *before* the runs.

---

### D1 — Fused megakernel vs unfused GEMM+collective (F1, skeleton S0 vs S1/S4)

**Inequality.** Fusion pays iff
```
ΔF  <  Δv · T_eff        where  ΔF = F_mega − F_unfused ,  Δv = v_unfused − v_mega
⟺   T_eff  >  T*  =  ΔF / Δv
```
**Instantiation (§1.9):** `ΔF = 886.7 µs`, `Δv = 0.6764 µs/token` ⇒ **T\* = 1,311 tokens/rank**.

> **P1.** The megakernel/production crossover sits at **T\* ∈ [1,300, 1,800] tokens per rank**,
> and the *same* threshold governs fill: a fill-aware kernel at `T_eff = φ·T` crosses at the
> same point. **Falsifier:** a T-sweep on a T-correct body that puts the crossover outside
> [1,100, 2,000], or a fill sweep (M24 `NORIG_CONST`) whose crossover differs from the T sweep
> by more than one grid step ⇒ `T_eff` is **not** the single mediating quantity and fill and
> batch size must be separate coordinates.
>
> **P1b.** At c32p (`T_eff = 1,539`), a fill-aware M15 lands **within ±10% of parity** on real
> work per step — i.e. the c32p kernel-level loss is *not* recoverable by fill-awareness alone,
> and the c32p cell must be won by the **~56% dummy-step skip**, not by a better schedule.

**Note the trap this boundary is built on** (`WORKLOAD_EVIDENCE §W6`, binding): the T-sweep is
on the **tgen body**, `K0_MAXTOK == T` makes it a *batch-size* sweep not a *fill* sweep, and
the serving M15 body **has never produced a number at T ≠ 4096**. **Fixing the serving body's
T-generality is a prerequisite for P1**, and it is the campaign's highest-priority engineering
dependency.

---

### D2 — Fuse the collective, or leave it to the vendor library? (F1 at TP8)

**Inequality.** Let `β = T_coll^ours / T_coll^vendor` (transport efficiency ratio),
`I_co` = co-residency tax, `h` = fraction of the collective hidden. Fusion pays iff
```
I_co + (1 − h)·β·T_vendor  <  T_vendor
⟺   h  >  h*  =  1  −  (1 − I_co/T_vendor) / β                                   … (D2)
```
**Instantiation** with θ-F11 (`T_vendor = 1,470 µs`), θ-F12 (`β = 1.506`), θ-P12
(`I_co = 335 µs`):

```
h*  =  1 − (1 − 0.2279)/1.506  =  1 − 0.5126  =  48.7%
```

> **P2.** At the current transport efficiency, **fused CDAR must hide ≥48.7% of the collective
> to break even**, not the ≥90% the G25-1 gate demanded (`M25_TP8_MEGA_DESIGN.md:181`).
> Measured hiding today is **2.1%**. The gate bar was set 1.85× too high and the arm is
> 23× short of the real bar.
> **P2b.** `h*` is **scale-invariant** in `T_vendor` given fixed `β` and `I_co/T_ours` — so the
> same 48.7% governs at the deployment point (2.8 ms/layer exposed AR), not just in the rig.
> **P2c.** If the transport gap closes to `β = 1.0` (RCCL parity), the bar falls to
> **h\* = 22.8%**; if additionally `I_co → 0`, to **h\* = 0** and fusion pays for any hiding
> at all. ⇒ **the transport track (A1–A6) is worth more than the hiding track until β < 1.2.**
> **Falsifier:** a run at the corrected operating point (M2) that shows `h ≥ 49%` and still
> loses ⇒ (D2) is missing a term (most likely a second co-residency tax on the *compute* side,
> i.e. `I` is not additive but multiplicative).

**Payoff arithmetic if the bar is met** (DERIVED): at `h = 0.90`, `T_layer = 7,600 + 637 +
0.1·4,217 = 8,659 µs` vs `10,400 µs` ⇒ **1.20×** ⇒ 25,844 → **~31.0k input tok/s**, against
the perfect-hiding upper bound of **~35.4k** (`TP8_STATE §1.4`).

**Critical operating-point caveat, and it is why P2 is not yet testable as stated:** the G25-1
rig runs at `k_inner=2048` where the compute floor for two bursts is **1,439 µs** while one
boundary's CDAR is **1,880–2,214 µs** — the collective is ~3× *longer* than the compute it must
hide under, whereas the deployment ratio is ~3:1 the other way (`TP8_STATE §2.4`). **Best
conceivable fused wall in the rig = max(compute, transport) ≈ 1,880 µs**, so hiding was
Amdahl-capped near zero and F1–F4 were falsified blind. **M2 (one argv value) must precede
any further schedule mutation.**

---

### D3 — Atomic-accumulate vs store-towers + owner reduce (K1), by message size and arch

**Inequality.** Both paths are affine in bytes with a fixed term; towers add a reduce pass
(`World × slab_bytes` read + `slab_bytes` write on the owner) but escape the atomic dword
issue-rate cap:
```
towers win  ⟺   L_t + b/R_t  <  L_a + b/R_a   ⟺   b  >  b*  =  (L_t − L_a)/(1/R_a − 1/R_t)
```
**Instantiation** from the two measured endpoints (16,384 tok / 235 MB / 256-row slabs:
towers 2,004.2 µs vs atomics 3,502 µs; 256 tok / 3.67 MB / 64-row slabs: towers 1,017 µs vs
atomics 801 µs — `TP8_STATE §2.2, §5/M5`):

| | marginal rate `R` | fixed term `L` |
|---|---|---|
| store-towers | 0.2343 MB/µs (234 GB/s) | 1,001 µs |
| atomic-accumulate | 0.0856 MB/µs (86 GB/s) | 758 µs |

```
b*  =  243 µs / 7.410 µs·MB⁻¹  =  32.8 MB  ≈  2,289 tokens per boundary
```

> **P3.** **K1 flips from `store_towers` to `atomic_accumulate` below ≈2,300 tokens per TP8
> boundary** (band [1,500, 4,000] because the two endpoint configs differ in `slab_rows`, so
> their fixed terms are contaminated by grid occupancy). This lands remarkably close to the
> design doc's independent claim that *"chunking collapses below ~2K tokens"*
> (`M25_TP8_MEGA_DESIGN.md:212-215`). **K1 is a manifest axis, not a global decision.**
> **Falsifier:** a clean `slab_rows`-matched sweep across b ∈ {3.67, 15, 33, 60, 235} MB whose
> crossover falls outside [1,500, 4,000] tokens.
>
> **P3b (ARCH).** On gfx942 (atomics ~3× slower than stores, θ-F16), `R_a` drops ~3× ⇒
> `1/R_a` triples ⇒ `b*` falls to **≈11 MB ≈ 780 tokens**. ⇒ **store-towers is right almost
> everywhere on gfx942**, which *simplifies* the PR: gfx950 and gfx942 now agree at deployment
> message size (`TP8_STATE §3.2`). **Falsifier:** a gfx942 K1 sweep showing atomics winning
> above 1,000 tokens.
>
> **P3c (the decode go/no-go).** Because `b* ≈ 2,300 tokens` sits *above* every decode batch,
> **every decode-cell conclusion in the campaign depends on θ-F13, which has never been
> measured.** No claim below ~2K tokens may be made until the RCCL decode reference runs
> (one argv — `TP8_STATE §5/M5`).

---

### D4 — Readiness granularity: when does finer pay? (K3)

The K3 contradiction resolves into **two independent terms**, and the campaign should test them
separately:

**(i) The `Φ` term (protocol/shadow, DP side).** Finer certification buys consumable shadow:
`Φ_slab(S_cnt) = (S_cnt−1)/(2S_cnt)`, bounded above by 0.5, against natural-order
`Φ_natural(k) = 1/(k+1)`. Restructure producer order + certify iff
```
(S_cnt − 1)/(2·S_cnt)  >  1/(k+1)     ⟺  at S_cnt = 2:  k > 3                    … (D4a)
```
> **P4a. Producer-order restructuring + slab certification pays iff top-k > 3.** At DeepSeek's
> **top-8**, `Φ` goes 0.111 → 0.250 (**2.25×**) — the mechanism behind the measured
> "order alone ≈ 0; order × certification = −144 µs". At **top-2** (Mixtral-class) the
> inequality reverses (`Φ_natural = 0.333 > 0.25`) and **per-row/fine readiness becomes viable
> while slab certification becomes a pessimization.** This is the campaign's cleanest
> "sign flips on a workload parameter" law, and it is **cheap** — top-k is a harness argv.
> **Falsifier:** a top-k sweep {2,4,8} × {natural order, nc-major+S=2} whose sign does not flip
> between k=2 and k=8.
>
> **P4b.** Refining past `S_cnt = 2` cannot be worth more than **2×** the consumable shadow
> (`Φ: 0.25 → 0.5`), while it costs `ι·ΔA` linearly. With `ι = 0.438 µs/1,000 ops` and the
> ~926k census, **the marginal return goes negative around `S_cnt ≈ 8–16` at T=4096.**

**(ii) The `η_size` term (transfer geometry, TP8 side).** Two competing costs — protocol
`∝ n_slabs = T/g` and an exposure/drain cost `∝ g^α`:
```
T(g)  =  a·(T/g)  +  c·g^α  +  k        ⟹    g*  =  ( a·T / (α·c) )^{1/(α+1)}     … (D4b)
```
**Fit on the measured slab sweep** (`g25_0b_slab_sweep.txt`, 128/256/512 rows =
2,025.0/2,004.2/2,513.9 µs at T=16,384): `a = 2.647 µs/slab`, `c = 3.02e-3`, `α ≈ 2` ⇒
`g* = 193 rows` against the measured optimum **256** (one grid step; 128 is within 1.0% of it).
Held-out check: the fit predicts g=1024 at 4,849 µs vs measured **4,243 µs** (+14%) — the
2-term form over-penalizes the coarse end, so `α` is slightly below 2.

> **P4c. `g* ∝ T^{1/(α+1)} ≈ T^{1/3}`.** At **T = 4,096** the optimal `slab_rows` falls to
> **≈122 ⇒ predict 128 rows** (one grid step below the T=16,384 optimum); at **T = 2,048**,
> **≈96 ⇒ predict 128**. **Falsifier:** a slab sweep at T=4,096 whose optimum stays at 256 ⇒
> `g*` is absolute (set by hardware quanta), not `T`-dependent, and (D4b)'s protocol term is
> mis-specified.
> **P4d.** Because (D4b) is a *bandwidth* law and (D4a) is a *shadow* law, **they must be
> measured on different instruments**: (D4a) on the DP mega with signal-count varied at fixed
> geometry; (D4b) on the G25-0b harness with `slab_rows` varied. Mixing them is what produced
> the apparent contradiction.

**Underlying mechanism for (D4b), now quantified:** `n_slabs × bps` sets the transport grid
occupancy. At `bps=2`: g=1024 ⇒ 32 blocks; 512 ⇒ 64; 256 ⇒ 128; 128 ⇒ 256. The measured
bandwidths 55.4/93.4/117.2/116.0 GB/s are a **saturating parallelism curve with a knee at
128–256 blocks** — consistent with θ-F2's 8–32-CTA-per-link knees. And `bps` itself is a real
parallelism knob: **1→2 = +64% (41.0 → 67.1 GB/s); 2→4 = flat** — one block per slab leaves
the grid half idle.

---

### D5 — Injection depth as a function of co-residency and payload (K4)

**Inequality.** From (M7), depth enters only through `J(d)` gated by `co`:
```
d*  =  ∞   if  co = 0                                                            … (D5a)
d*  =  8   if  co = 1   (gfx950, packed-bf16 epilogue RMW, 7 peers, 256 CTAs)     … (D5b)
```
This is already **half-proved**: LAW-8/F6 measured depth **flat** {4,8,16,32} in pure
transport (93.6/93.4/93.3 GB/s towers; 67.1/67.1/67.0 atomics), and LAW-20 measured a **cliff**
at co-residency. The mediating quantity is **co-residency, not bytes** — the sharpest law the
project owns.

**Mechanistic extension (SPECULATION, cheap):** if `d*` is set by aggregate in-flight remote
bytes against a queue resource, then `d* × (bytes per remote op) × (injecting waves)` is the
invariant. ⇒

> **P5a.** `d*` **halves when the remote op width doubles** — the m15b staged arm issues 16 B
> posted packet stores instead of 4 B packed-bf16 RMWs, so `d*(m15b) ≈ 2–4` where
> `d*(mode 12) = 8`. **Falsifier:** an m15b depth sweep whose optimum stays at 8.
> **P5b.** On **gfx942**, atomics drain ~3× slower ⇒ congestion forms at lower depth ⇒
> **`d* < 4`, plausibly 2**. Depth 2 has never been measured on *any* arch (the `g` `0x300`
> selector is full) — **widening that selector is a prerequisite**, and it is one bit.
> **P5c.** `d*` is **independent of T and of message size** (it is a rate bound, not a volume
> bound). **Falsifier:** a depth × T sweep in which `d*` moves with T ⇒ depth must re-enter
> (M5) as a bandwidth term and (D5a) is wrong.
> **P5d.** In the **fused** TP8 arm at the corrected compute:transport ratio (M2), depth
> should **re-acquire** its effect (co returns to 1). **This either re-establishes the −615 µs
> law under a new topology or retires it as DP-specific — either outcome is a campaign law.**

**Fragility contract that must ride every depth arm** (`credit.cuh:30-38`, LAW-32): a bound
that can silently die is not a primitive. exp_38 saw the register allocator install a
*stronger involuntary throttle* with **no source change** (+726.9 µs), collapsing the atomic
issue-run distribution from **282 atomics in 12 runs, mean 23.5 in flight** to **194 runs,
mean 1.45**. Gate = (a) literal `vmcnt(N)` at expected sites, (b) zero scratch ops within 24
instructions of any remote atomic, (c) the issue-run distribution.

---

### D6 — Producer-carried vs dedicated-comm carriers (skeleton A.1)

**Inequality.** A reserved pool of `C` CTAs pays iff its job's value exceeds the capacity tax:
```
value(job) ·  C   >   C_p(N)·φ_p·[N/(N−C) − 1]                                   … (D6)
value(carry)   =  0        (payload rides ~free on producer stores: concurrent/isolated 0.9951)
value(idle)    =  0        (measured tax +8–9 µs/CTA, monotone)
value(consume) =  ρ·τ/R_serial + κ_couple   >  0  ONLY IF  V_cert > 0
```
**Instantiated verdicts, all measured:** carrying **+332…+340 µs, never wins**; idling
**+8–9 µs/CTA monotone**; consuming **−93 µs (C 8→16), −444 µs (C 16→24)**
(`OVERLAP_ABSTRACTIONS.md:26-30`).

> **P6.** **Dedicate CTAs to consuming, never to carrying, and only when a certified consumable
> prefix exists.** This is the direct answer to the expert question *"does our M15
> producer/consumer work best for all sizes?"*: it works **iff `V_cert > 0`**, i.e. iff
> K2×K3 have been set (D4a: iff top-k > 3), and it stops paying at `C > C_sat` (D7).
>
> **P6b — the carrier-conditional caveat that must be stated, not buried (contradiction C3).**
> "Dedication is bad" is **carrier-conditional, not a technique verdict**. In the *staged-push*
> carrier (S2/mode 2) dedication **won** at C=64 (0.888× vs homogeneous 0.894×), turning over
> only at C=96/128; in the *producer-carried* carrier (S3/mode 12) it lost monotonically at
> every size (C=4 6,464.2 … C=64 6,837.5, paired 7/7 rounds). (D6) explains both: with a
> staged-push carrier the pool *is* the transport (`value(carry) > 0` because nothing else
> moves the bytes); with a producer-carried epilogue it has no job.
>
> **P6c — the one regime that could rescue a carrier pool is untested, not refuted.** Under
> **skew**, the hot rank's producers are late and a pool with resident progress could matter.
> `K0_SYNTH_ROUTE` is rejected on the aug11 harness and reachable CV is **0.0034–0.0086 vs
> COMET's 0.032** (LAW-62b) ⇒ **the axis does not exist on that instrument.** M24's
> `NORIG_TABLE` (per-rank heterogeneous fill) and the route-replay rig are the two actuators
> that could create it. **Prove the knob is settable before pre-registering the cell.**

---

### D7 — Consumer pool size `C` vs T, fill, and skew (K5)

**Closed form**, from (M9): `C* = min(C_sat, C_tax)` where
```
C_sat  =  V_cert / (ρ · τ_shadow)                                                … (D7a)
C_tax  =  argmax_C [ O_consume(C) − C_p(N)·φ_p·(N/(N−C) − 1) ]                   … (D7b)
```
**Instantiated at T=4096:** `C_sat = 20.8` vs measured knee **24–28** (§1.7).

> **P7a — `C*` is first-order INVARIANT in `T_eff` under a fill-aware kernel.** Both
> `V_cert ∝ T_eff` and `τ_shadow ∝ T_eff`, so their ratio is constant. **This contradicts the
> kernel's own comment** that "at `T_eff` … C / flush_rows are no longer at their tuned point"
> (`k0pf6gm_device_tile_m15.hip:2122-2127`). M says the *quota* `flush_rows` (measured in
> **rows**) must scale with `T_eff` while `C` need not. **Falsifier:** a C × T_eff grid where
> `C*` moves by more than one grid step ⇒ `τ_shadow` is not linear in T_eff (most likely
> because at small T the producer becomes latency-dominated, `φ_M7 → 0`), and M must add a
> `φ_p(T)` term.
> **P7b.** `C*` **falls at small T**: once `φ_M7 → 0` the shadow shortens sub-linearly while
> `V_cert` falls linearly. Combined with D1 this predicts the **same** T-region (`T_eff` near
> T\*) shows both `C* → small` and the megakernel losing — i.e. **the two failures share one
> cause**, they are not independent risks.
> **P7c — `C*` under skew.** On the hot rank the certified prefix arrives *later*, so
> `τ_shadow` shrinks while `V_cert` is unchanged ⇒ **`C*` RISES with σ**. But `C` is a global
> config word, so a single `C` cannot be right on all 8 ranks under skew.
> **⇒ Registered prediction: under σ > 2, per-rank `C` (or `M24_FILL_C`, the declared-but-
> unimplemented "runtime-adaptive reserved_comm_ctas") is worth more than any global `C`.**
> **Falsifier:** a per-rank-`C` arm that ties a global `C` under replayed skew.

**Instability note that must ride every C sweep:** C=32 **hung 1 of 2 runs** (BRIEF `:51`).
`C*` sits one grid step below a hang, which is a design smell the campaign should diagnose
(most likely the pool holding the next rendezvous hostage — the exact failure `flush_rows` is
designed to prevent).

---

### D8 — Push vs pull, per boundary (K1/direction)

**Inequality.** Bandwidth is a tie (θ-F9: 0.9936× at matched 64 KiB), so direction is a
**latency/ordering** choice governed by knees and by who defines progress:
```
choose PULL  ⟺  m_per_peer ≥ 1 MiB  ∧  n_consumer_CTA ≤ 256  ∧  progress is consumer-defined
choose PUSH  ⟺  otherwise                                                        … (D8)
```
(fan-out knees: push 64 KiB/peer, pull 1 MiB; **pull collapses past 256 CTAs while push
degrades gently** — `OVERLAP_KERNEL_DESIGN_ADDENDUM.md:317-321`).

**Instantiation for M25's MAG (the consumer-side gather):** per-slab per-peer bytes =
`256 rows × 14,336 B = 3.67 MB > 1 MiB` ⇒ pull is legally past its knee. **But the rig runs
256 blocks — exactly the measured pull-collapse edge.**

> **P8. MAG's pull is operating at the CTA count where pull is measured to collapse.**
> Reducing MAG's puller count to ≤128, or converting MAG to an **owner-side store-class
> multicast riding the owner's own reduce epilogue** (the only push form that does not
> re-violate the carrier-pool falsification — `TP8_STATE §5/M8`), should recover bandwidth.
> This gives hypothesis A1 a *quantitative* reason it did not have.
> **Falsifier:** a MAG puller-count sweep {64, 128, 192, 256} that is flat ⇒ the collapse knee
> does not transfer from the aug12 anchor to the 235 MB many-to-one case, and θ-F10 must be
> re-measured at deployment size.
> **P8b.** The push/pull tie was measured at **64 KiB** and the pull edge at **64 MiB
> single-link** — **neither is a 235 MB many-to-one/one-to-many case.** The selection rule
> ("push when producer-defined & many-to-one; pull when consumer-defined & one-to-many") is a
> **design rule, not a measured crossover** (LAW-5, explicitly HYPOTHESIS). Calibrating it at
> deployment size is a named hole.

---

### D9 — Owner/expert placement vs κ: which lever first?

**Inequality.** Placement moves σ (§1.8); co-scheduling moves `W_pad`/κ (§1.10). Both multiply
the same step:
```
gain(placement)  =  1 / [ (1−f) + f · M(σ_RR)/M(σ_linear) ]   where M(σ) = (1−λ)+λσ
gain(κ)          =  observed 260/322 padded steps → 128  ⇒  15–25% of node throughput
```
**Instantiation** with `λ = 0.6266`, `f = 0.45`, `σ_linear ∈ [2.61, 5.088]`,
`σ_RR = 1.085` (θ-W5):

| σ_linear (which instrument) | MoE-region factor | **predicted step speedup** |
|---|---:|---:|
| 5.088 (aggregate histogram) | 3.381× | **+46%** |
| 3.0 (pad-routing degenerate, ~3.0× — `CORPUS_FINDINGS.md`) | 2.139× | **+32%** |
| 2.61 (corpus per-call, real rows) | 1.907× | **+27%** |

> **P9. RR-by-permutation / EPLB0 placement is worth +27% to +46% of ABSOLUTE node throughput
> at c32p — for BOTH arms — and changes the A/B ratio by <3%.** This is *larger than any
> kernel delta in the entire ledger* and larger than LAW-50's co-scheduling headroom. It is
> also **the campaign's most exposed over-prediction**, deliberately registered as such.
> **Falsifier and what each outcome means:**
> * measured serving gain **< 10%** ⇒ `λ_serving ≪ λ_rig`, i.e. **the replay rig over-states
>   skew because it replays skewed routing on 100% of rows while only 37.6% of serving rows are
>   real** — a first-class finding that would invalidate every replay-based skew number.
> * measured gain **in [27%, 46%]** ⇒ M's skew term transfers from rig to serving, and
>   **placement, not schedule, is the headline contribution.**
> * measured **ratio** change > 5% ⇒ λ_M15 ≠ λ_prod matters at serving scale and the
>   "RR helps both arms equally" reading is wrong.
>
> **P9b — ordering is not placement.** m17 RR *landing order* buys only **~1–1.5% under skew**
> (0.8467 → 0.8370 aggregate; 0.8559 → 0.8413 worst layer). **The bottleneck is popularity
> concentration, not ordering** — already resolved, and M reproduces it: order changes `Φ`
> and `η_layout`, neither of which is in the `M(σ)` multiplier.
> **P9c — replication frontier.** Uniform-K replication **halves** skew damage but does not
> remove it: K=2 (4.8 GiB) → 4.10×; K=8 (19.0 GiB) → 1.77×; K=16 (38.1 GiB) → 1.09×; greedy ≈
> uniform + ε at equal bytes (LAW-47). Through (M13), `K=8` predicts a MoE-region factor of
> `M(5.088)/M(1.77) = 3.562/1.483 = 2.40×` ⇒ step speedup **+31%** for 19 GiB/rank. **The
> memory price per point of σ is now computable**, which is the decision the EPLB0 gate needs.

---

### D10 — Overlap-region choice vs the slack map

**Inequality.** A filler `F` pays iff **all four** hold:
```
(1) it passes the four-door test (§1.7)                    [else predicted ≈ 0]
(2) demand(F)  ≤  supply(B)  (the measured bubble ledger)
(3) it removes work from the CRITICAL rank's wall           [LAW-52]
(4) it does not extend the residency contract               [LAW-54]
```
**Instantiated bubble ledger (training arm, the only place it was measured):** in-launch supply
**~150–180 CTA-ms** vs in-situ wgrad demand **~3,360 CTA-ms (13.1 ms/launch)** — **~5% of
demand**. Every wgrad rescheduling variant then lost (control 1,670/1,689; in-kernel filler
1,892; partitioned 1,959–1,962; dribbled 1,988–1,993). The filler that *does* fit is the
shared expert (~2 ms/layer·mb vs ~150 CTA-ms of window) (LAW-53).

> **P10a. In serving prefill there is almost no independent filler**, so overlap is
> second-order to moving compute: wgrad and shared-expert-bwd exist **only in training**;
> serving prefill has *"NONE of the first three at meaningful scale (no wgrad, shared expert
> only ~1/8 fwd) — which is exactly why 'M15 with more overlap' plateaued"* (LAW-48). The
> forward chain leaves **exactly one legal cross-region forward overlap**: combine(L) tail vs
> attention(L+1) prologue on finished tokens.
> **P10b. Shared-expert filler is real but capped at ~1/8 of routed FLOPs** — it can fill at
> most ~12.5% of a layer's compute span and **must not be sold as the answer to a 28% Amdahl
> pool** (`TP8_STATE §6/B4`).
> **P10c. Measure the bubble ledger BEFORE designing any filler.** Its measured cost is
> **zero** (1,673/1,689.7 vs control 1,670/1,689). This is the method that transfers; the
> magnitudes do not.
> **P10d.** Any serving proposal that puts work on a **side stream** against the resident
> megakernel inherits LAW-54: **only sub-millisecond kernels slip past the residency contract.**

---

### D11 — The four-door screen (the model's cheapest output)

Before any arm is built, M predicts ≈0 for any schedule that goes through none of the four
doors. **Retroactive check against the falsified ledger** — every falsified arm in the corpus
goes through zero doors, and M would have rejected all of them for free:

| falsified arm | doors it goes through | measured |
|---|---|---|
| mode 16 cross-launch defer (F9/S5) | none (re-divides the same CTA·µs) | **+75.8 µs** |
| exp_25 M6/M7 interleave | none | first-order tie |
| exp_25 static split | none | **−173 µs loss** |
| carrier pool (S2) under a producer-carried epilogue | none (`value(carry)=0`) | **+332…+340 µs** |
| G25-1 F1 drain pipelining | none (producer-side re-ordering) | **null** |
| G25-1 F2 register-source stores | (c)? op-class unchanged | **null** |
| G25-1 F3 consuming-pool specialization | (a), but `V_cert` untested at that ratio | **null** |
| G25-1 F4 `bps=16` item granularity | none; raises `A` | **null and WORSE (+180 µs)** |
| exp_20 pacing a dedicated pool | none; drain *is* the critical path | combine pays **7:1** |
| exp_33 poll backoff (64× spin-rate cut) | none | **flat null** |
| deleting the per-task VMEM drain | none | **+3.2 µs** |

> **P11. The four-door screen has an 11/11 retrospective hit rate on this corpus.** Registered
> forward prediction: **every arm the campaign proposes that goes through zero doors will
> measure inside ±1σ of its control.** **Falsifier:** any zero-door arm that measures a
> reproducible effect > 3% ⇒ there is a fifth door and the occupancy-1 selection rule is
> incomplete. *This is a genuinely useful thing to be wrong about* and should be reported
> loudly if it happens.

---

## 4. VALIDATION PROTOCOL

### 4.1 Fit / hold-out split

**Calibration cells (θ is fitted here, once):**

| set | cells | what it fits |
|---|---|---|
| **CAL-T** | G25-0b transport-only: {atomic, towers} × slab_rows {128,256,512,1024} × depth {4,8,16,32} × bps {1,2,4} × tokens {256, 2K, 4K, 16K} | θ-F4..F10, `η_size`, α, a, c |
| **CAL-C** | mega chassis at T=4096: reserve-but-idle C-sweep + phase stamps | φ_p (θ-C4), θ-P4 (`c_bar`), θ-C6 (ρ) |
| **CAL-P** | protocol: `A` census ×  L ∈ {2,4,8,16,32} × signal-count {per-row, S=2, S=8, S=1} | ι, ψ (θ-P3), θ-P9 |
| **CAL-S** | skew replay ladder: {balanced, aggregate hist} at T=4096 | λ (one point per arm) |

**Held-out cells (θ is FROZEN; M predicts, then we measure):**

| set | cells | predictions tested |
|---|---|---|
| **HO-1** | T-sweep on a T-correct body: T ∈ {512, 1024, 1536, 2048, 3072, 4096} | **P1** (T\*), **P7a/b** (C\*(T)) |
| **HO-2** | slab sweep at T=4096 and T=2048 | **P4c** (`g* ∝ T^{1/3}`) |
| **HO-3** | K1 crossover sweep, `slab_rows` matched, b ∈ {3.67, 15, 33, 60, 235} MB | **P3** (b\* ≈ 33 MB) |
| **HO-4** | skew replay at worst-layer + a third σ point (RR-permuted histogram) | **P9** via (M13) |
| **HO-5** | top-k sweep {2,4,8} × {natural, nc-major+S=2} | **P4a** (sign flip at k=3) |
| **HO-6** | G25-1 at the corrected compute:transport ratio (M2) × depth {4,8,16,32} × C {0,8,16,24,32} | **P2**, **P5d**, **P8** |
| **HO-7** | serving: c512p ≥5 order-balanced pairs; c128p (the unmeasured middle, H3) | **P1b**, lift validation |
| **HO-8** | gfx942 K1 + depth | **P3b**, **P5b** |

**Rule:** θ is **never** re-fitted on a held-out cell. If a held-out cell needs new θ, that is
recorded as a **model failure**, not a calibration update (§5).

### 4.2 Metrics

For each cell `w`, with `A(w)` = the set of arms swept in it:

1. **Rank correlation** `ρ_S(w)` — Spearman between M-predicted and measured arm times.
   *Primary*, because the campaign's product is "which schedule wins", not "how many µs".
2. **Regret** `R(w) = [t(â_M) − t(a*)] / t(a*)` where `â_M` = M's predicted-best arm and
   `a*` = actual best. *Primary decision metric.* Report median, p90, and max over cells.
3. **Boundary-location error** `E_b` — for each swept axis with a predicted crossover
   (T\*, b\*, g\*, C\*, d\*, k\*), the distance in **grid steps** between predicted and
   measured boundary.
4. **Coefficient stability** — the spread of each θ entry when re-fitted per cell. A θ that
   moves more than its own measurement error across cells is not a constant (§5, F2).
5. **Sign accuracy** — fraction of (knob, cell) pairs where M gets the *direction* right.
   Reported separately because a model can be useful with sign-only accuracy.

### 4.3 The accuracy bar (stated in advance; below it M is refined, not papered over)

| metric | bar | instrument-aware rationale |
|---|---|---|
| median regret | **≤ 3%** | LAW-60: campaigns reproduce to 0.09%, stamps to 1.1–2.3%; 3% is the smallest callable *screen* delta, so 3% is measurable at the boundary |
| p90 regret | **≤ 8%** | one grid step of a coarse C or T sweep |
| max regret | **≤ 15%** | anything worse means M would pick a *qualitatively* wrong skeleton |
| Spearman ρ_S per cell | **≥ 0.70** with ≥6 arms | below this M cannot order arms and is not a screening tool |
| boundary-location error | **≤ 1 grid step** on every swept axis | the standard M already meets on `C_sat` (21 vs 24) |
| sign accuracy | **≥ 90%** | below this the "if X then Y" law form is not supportable |
| θ stability | each entry within **±2× its own measurement CI** across cells | LAW-60 gives the CIs |

**Instrument-scoping rule, binding.** These bars apply at the **boundary level**. At **e2e**
the minimum detectable effect is `~15%` at c32p and `≤1.8%` at c512p (θ-W10) — **the variance
envelope is itself a function of W**. ⇒ *No e2e cell may be used to adjudicate a prediction
smaller than its own measured envelope.* e2e is the **validation** instrument; boundary rigs
are the **precision** instrument (LAW-45).

### 4.4 Sample sizes, gates, and the reproducibility contract

* Boundary: **5-rotation campaigns** (500 warmup / 100 timed / 600-epoch soak), ~3 min 5 s each
  (θ-W12). Same-session paired controls, alternating arm order within a batch (the exp_35
  `c,b,c,b` discipline).
* Serving: **order-balanced ≥5 pairs**, `ARM_COOLDOWN=240 s`, fresh server per arm, accuracy
  gate, TTFT p50/p99, native-tuned baseline (`nightshift/BENCHMARK_PROTOCOL.md`). The two
  banked c512p pairs are **n=2 — directional only**, below the floor.
* **LAW-58/59 parity gates are mandatory and are part of M's identity:** `.text` sha256
  identity for any change claimed inert; the flag-ON build **must differ**; scratch **op
  count** and SGPR/VGPR **spill counts** (not just `ScratchSize`, which stayed pinned at 128 B
  while scratch ops went 19 → 168). *The resource tuple is not a parity gate* — a build passed
  every gated field and still cost the ratchet **+726.9 µs**.
* **LAW-61:** cluster by run, not by (run, rank) — the same null reads `t = −5.01` treated as
  160 independent cells and `t = −0.73` with runs as units. `[MOK GATE]` digits are **not**
  run-reproducible even from byte-identical `.text`; use same-run paired differences. A
  rank-N-only failure is invisible in rank-0 stdout.
* **LAW-63 timestamp hygiene:** 1 tick = 0.01 µs (100 MHz) — using the 2.2 GHz shader clock is
  a **22× error**; stamps are running maxima never reset, so after a soak they describe the
  *soak* epoch; MPS phase prints are CTA-max **within rank 0** (production's genuine cross-rank
  combine reads 1,356.0 rank-max vs 978.4 rank-0, **+38%**); never subtract `clock64()` across
  kernels (one such span read 55,576.5 µs against a 1,278.81 µs HIP event).
* **LAW-62:** every axis in W must be **shown settable** before its cell is pre-registered.
  A spec-sheet denominator can manufacture a refutation (exp_22's H1 demanded 57.6 GB/s when
  the achievable ceiling was 56.9–57.3 — *no CTA count could have passed*).
* **Manifest row** — every arm regenerable from `KNOB_INVENTORY` Appendix: skeleton, build
  flags (**including the instrument gates `K0P6_MPS_ENABLE_{MODE14,TBO}`, `SRC_REV`, kernel-name
  pin**), config word (C, g, mode, flush_rows, pull_fallback, timestamps), host/geometry,
  **full workload tuple**, and gates. **No naked TPS, ever.**

### 4.5 What "refined, not papered over" means operationally

If a bar is missed, the response is fixed in advance, in this order:

1. **Check the instrument first** (LAW-58/59/63). A missed bar caused by a stale `.hsaco`, an
   unreset stamp, or a MODE14=1 binary is not a model failure.
2. **Check whether the missed cell is one of the two named fudge factors' regimes.** If yes,
   run the device phase ledger (θ-P5/P6) before touching M's structure.
3. **Add a mediator** only if a *specific* pair of cells with equal Q and different S\* is
   exhibited (§5, F1). The mediator must come from the named candidate list, not be invented.
4. **Split the regime** only if step 3 fails, and then declare the regime indicator explicitly
   (topology is the leading candidate — `TP8_STATE §8` already argues the campaign needs *two*
   laws because fill does not exist under TP8).
5. **Publish the miss.** A cost model with a documented, bounded failure region is a result;
   a cost model with an undocumented one is a liability.

---

## 5. FAILURE MODES — what would falsify "W acts through Q", and what we do then

### F1 — Q is insufficient (the central risk)

**Test.** Exhibit two cells `w₁ ≠ w₂` with `Q(w₁) = Q(w₂)` (within measurement error) whose
optimal schedules differ by more than one grid step on any axis. **One such pair falsifies the
compression claim.**

**Highest-probability instances, named in advance:**

| pair | why Q might be blind | what it would prove |
|---|---|---|
| prefill vs decode at equal `T_eff` | **H1: decode has never been measured in any rig.** Decode is latency-bound with a different arithmetic intensity; `φ_p`, `ρ`, and `η_size` may all be different constants | Q needs a **phase** indicator, not just `T_eff` ⇒ `PHASE` is a regime, not a coordinate |
| DP/EP vs TP8+EP | **fill is a DP-topology quantity and does not exist under TP8** (`TP8_STATE §3.3`); the TP8 mediating quantity is *exposed collective fraction* | Q needs a **topology** indicator ⇒ two laws, not one. `TP8_STATE §8` already predicts this |
| balanced-replay skew vs serving skew at equal σ | serving skew is a *mixture*: 62.4% pad rows (91 distinct pad expert-sets, pad max-rank load ~3.0×, **pads do NOT all land on rank 0**) + 37.6% real rows | σ must be a **distribution**, not a scalar; per-call σ ≠ aggregate σ (H17's 2× gap) |
| same `T_eff` from (high T, low φ) vs (low T, high φ) | without M24 the kernel pays padded cost, so `T_eff` is not what it executes | **D1's unification is wrong** and fill and batch size are separate coordinates |

**Response ladder (pre-committed).** Add mediators in this order, one at a time, each with the
measurement that justifies it:
`co` (already in) → `L` (scatter width, θ-P3) → phase indicator → topology indicator →
σ as a distribution → κ. **Never add a mediator without a cell pair that demands it.**

### F2 — θ is not constant (the model degenerates into a lookup table)

**Test.** Re-fit each θ entry per cell. If an entry moves more than ±2× its measurement CI
across cells, it is not a hardware constant.

**Known-fragile entries and their tells:**
* **`I₀`** — bounded ≤1,159 µs with **64% of its mechanism unidentified**. If `I₀` must be
  re-fitted per *cell* rather than per *skeleton*, the interference model is phenomenological
  and every co-residency prediction becomes an extrapolation. **This is the single most likely
  route to F2.**
* **`κ_couple`** — if it moves with T or σ, the producer-coupling term is not a constant and
  (M8) needs the phase ledger before it can be used predictively.
* **`ρ = 6.1 GB/s/CTA`** — measured on the combine body only; a different consumer body has a
  different ρ, and (M9)'s `C_sat` scales inversely with it.
* **θ-F4 (egress ceiling)** — flagged *unreproduced* at source. Every "% of ceiling" framing in
  §1.4 and `TP8_STATE §2.3` inherits that caveat.

**Response.** Declare the entry a **per-skeleton** or **per-arch** constant (legitimate — that
is what an arch card is: `OVERLAP_ABSTRACTIONS.md §5`) and state the scope. Declaring it
per-cell is equivalent to abandoning M.

### F3 — The `MAX` is dominated by something M cannot see

(M1) takes the max across ranks. Two unobserved sources of rank asymmetry exist:
* **Thermal/clock drift** — root-caused only in the training arm (GPU2 at 77 °C, 1,725–1,743
  vs 1,790–1,822 MHz, a **4–5% clock spread**; `rows_done` wait 92 µs on rank 2 vs 4,353 µs on
  rank 0 per CTA per launch). **Never measured on the serving arm (H16).**
* **A hot-GPU ∩ hot-expert-rank coincidence** would be invisible to every current serving
  instrument, and the megakernel's slab rendezvous couples every rank to the slowest one.

**Test.** Per-rank device stamps in every serving arm + `amd-smi` clock/temperature logging.
**Response.** If clock skew is a material fraction of σ's effect, `σ` must be decomposed into
`σ_route × σ_hw`, and the placement remedy (which only touches `σ_route`) is capped
accordingly. **This would directly bound P9.**

### F4 — Mechanisms that were designed to compose turn out to be substitutes

**LAW-37 is unresolved and it is structural:** the injection bound is worth **−613.5 µs** and
deleting the per-row readiness protocol is worth **−573.3 µs** — two numbers closing within 7%,
both plausibly bounding the same in-flight-remote-write resource. **Neither has been shown to
pay on top of the other.**

**Test.** The 2×2: {bound on/off} × {coarse readiness on/off}, same session, `.text`-gated.
**Response if substitutes.** No waterfall in the paper is additive; M's `Q_p` and `I_p` terms
must share a single `in-flight remote write` resource variable rather than contributing
independently. **This changes the model's structure, not its coefficients** — which is why it
is a *failure mode* and not a calibration item. **It is also the cheapest of all the F-tests
and should run first.**

### F5 — The instrument changes the kernel

**LAW-32 is the biggest threat to the entire manifest plan**: a commit whose diff **cannot**
change mode 12 changed it by **+726.9 µs (11%)**, because making mode-14 code *reachable* let
the register allocator install a stronger involuntary throttle. `if (mode == 14) return;` on an
unexecutable branch is one of the two worst sites; the largest lump of new code (+5,568 B
`.text`) is inert. *"Code size is not the variable; where the code sits relative to the
mechanism's live ranges is."*

**Consequence for a knob-space campaign:** *a manifest of parameterized skeletons will silently
re-time arms whose knobs it never touched.* **Response, non-negotiable:** every arm carries a
mechanism-level invariant check (the issue-run distribution, `.text` identity for inert
changes, the flag-ON difference), and the instrument gates are **manifest fields**, not build
hygiene.

### F6 — The lift, not the model, is wrong

M can be right at the boundary and wrong at e2e. Three measured reasons:
* **LAW-44**: median TTFT fell 5.0% while node throughput fell 7.84% — a single headline can
  carry the **wrong sign**.
* **The metric map is regime-dependent, measured**: at c512p, m15 loses TTFT p50 by 1.49×,
  wins TTFT p99 by 0.92×, wins TPOT p50 by **4.24×** (757 vs 3,209 ms), and therefore wins e2e
  p50 by 0.944×. At c32p vs TP8 the inversion runs the *other* way (m15 loses TTFT p50, wins
  every p99 by 23–33%). **Any single-metric headline at either cell is a choice, not a fact.**
* **LAW-50**: 15–25% of node throughput sits in DP prefill co-scheduling — larger than any
  kernel delta in the ledger, and **kernel-independent**.

**Response.** Report M's prediction at the layer level with its own bar, and report the lift
separately with κ, coverage, and `W_pad` all stated. **A serving number without
`sealed/in_bucket`, `κ`, and `W_pad` is not admissible evidence about the kernel.**

### F7 — The premise of a whole track dissolves

Two live examples the campaign must be ready for:
* **The bimodal fill distribution may be an `ISL == max_num_batched_tokens` artifact** (§1.9).
  If so, `𝔇(T_eff)` is not transferable and every tier-1 sizing number is cell-specific. Test
  cost: one ISL sweep. **This is the cheapest high-leverage test in the whole plan (H4).**
* **Run-correlation is already falsified as a first-order effect** (+1.3 points of expert
  overlap vs shuffled) — the motivating premise of the entire route-capture rig. The GPU-side
  captured-vs-shuffled replay that would settle it at kernel speed is **code-complete,
  unit-tested, and has never been executed** (H8). Running it is 139 s per campaign vs ~25 min
  per serving pair.

---

## 6. Summary — what M buys, in one table

| question the experts asked | M's answer | status |
|---|---|---|
| "Does M15 producer/consumer work best for all sizes?" | **No.** It works iff `V_cert > 0` (⟺ top-k > 3, D4a) and `C ≤ C_sat = V_cert/(ρτ)` (D7). Below `T_eff ≈ 1,300–1,800` the whole *fusion* stops paying (D1) | `C_sat` predicted 21 vs measured knee 24 — **one grid step**, no fitting |
| "What are the properties that control the workload?" | **Eight mediating scalars Q**, not the ~8-dimensional W. Six are already instrumented | §0.3 |
| "As we toggle W, what changes in the kernel?" | **11 decision boundaries D1–D11**, each an inequality over Q and θ, each instantiated | §3 |
| "Can we do that in a cost model?" | Yes, with **two named, bounded fudge factors** (`I₀ ≤ 1,159 µs`, `κ_couple ≤ 55 µs/CTA`), both removable by **one instrument** (the device phase ledger) | §1.6, §1.7 |
| "What have we learned?" | The skew multiplier reproduces a held-out point to **0.47%** with one parameter; the affine T-model reproduces a held-out point to **1.07%** with two; the four-door screen is **11/11** retrospectively | §1.8, §1.9, D11 |
| Risk: "very expansive, unclear what the core contribution is" | The core contribution is **Q and the four-door selection rule**: the claim that the whole workload space acts on the schedule through eight scalars, and that overlap pays through exactly four mechanisms. Everything else is instantiation | §0.2, §1.7 |

**The two sentences the campaign should be able to defend at the end:**

> **DP/EP:** the schedule is governed by `T_eff = φ·T`, and the megakernel wins above
> `T_eff* ≈ 1,300–1,800` real tokens per rank because fusion buys `0.68 µs/token` and pays
> `887 µs/epoch`.
>
> **TP8+EP:** fill does not exist; the schedule is governed by the exposed-collective fraction,
> and fusion pays only once `h > 1 − (1 − I_co/T_vendor)/β`, which today is **48.7%** and is
> **22.8%** at RCCL transport parity. Closing β is worth more than any hiding hypothesis until
> β < 1.2.

---

## Amendments from review (2026-08-18)

Superseded on the contribution sentence, the `Q` table and the fit/hold-out split by
`../ABLATION_METHODOLOGY.md`. Structural changes forced by BLOCKING items in
`../review/MECHANISM_CRITIQUE.md` and `../review/CONTRIBUTION_CRITIQUE.md`:

1. **§6's contribution row is replaced.** Not "Q and the four-door selection rule"; not the two
   topology sentences (those are the pre-committed fallback). One sentence: *one base megakernel
   plus a library of attachable schedule modules, with M as the composition policy.*
2. **§0.3's 8-scalar `Q` is superseded by the canonical table** in `ABLATION_METHODOLOGY §2.2`
   (T_eff, k, B_bnd, C_phase, ρ, co, σ, λ_msg). `κ_sched`/coverage/`W_pad` move to a validity-gate
   vector `V`; `A`, `L`, `u(t)/Φ`, `φ_p` are schedule-response quantities in `R`, not `Q`.
3. **(M5) gains an incast term** (MECHANISM B1): `BW_eff = min(egress, links_used(π,order)·BW_link,
   ingress_dst/share) · η_layout · η_par`. The old product-of-independent-efficiencies form
   predicted ≈0 for destination-interleaving remedies that the vendor measured at **65%**.
4. **(M4) gains a fixed/latency intercept** (MECHANISM B2): `X = L(op, world, n_slabs, bps) +
   b/BW_eff`. Without it the model predicts **31 µs against a measured 801 µs** at the one decode
   point we own. `L` is calibrated from existing P0-A points at zero extra node cost.
5. **`η_size(m)` and `η_ctas(n_CTA)` are collapsed into one factor** keyed on (blocks, bytes per
   block) — the document's own text says they are the same saturating-parallelism measurement
   (MECHANISM M2).
6. **D4a is restated two-sided** (MECHANISM B6): `(Φ_slab−Φ_natural)·V(k)/R_serial +
   κ_couple·ΔC_sat > ι·ΔA(k)`. With this document's own θ, slab certification **still wins at k=2 by
   2–10×**, so **P4a's "sign flip at top-k 3" is retracted before measurement** and X3 is re-ranked
   from flagship law to model-generalization cell.
7. **`h` is redefined** (MECHANISM B4): the kill criterion rides `h_ledger` (span overlap from the
   device ledger), not `h_wall = (phased − fused)/collective`, which cannot separate hiding from the
   co-residency tax. The gap between the two **is** `ΔI_co`.
8. **`I₀` is calibrated on the mega chassis, not the boundary rig** (MECHANISM B5): the rig's
   co-resident "compute" has a fixed MFMA:load ratio and an L2-resident working set, so it cannot
   produce a quantity whose declared scaling is co-resident memory intensity. A `bytes_per_mfma`
   argv separates intensity from duration.
9. **Decode gets a declared branch or sign-only predictions** (MECHANISM B9): no expert-weight-stream
   term, no tile quantization, and a host bound imported from a padded prefill step. Every decode
   prediction in this campaign is **sign-only** and labelled. HBM is priced against the **measured**
   4,151.9 GB/s plateau, never a back-derived peak (MECHANISM M10).
10. **§4.1's CAL/HO tables are deleted as the authority.** `ABLATION_METHODOLOGY §4.6` is the single
    split: any Phase-0-measured point is FIT; CAL-T is endpoints only; the λ and affine-T results are
    relabelled **PRIOR-VALIDATED (in-sample)**, not campaign hold-outs (CONTRIBUTION B6,
    FEASIBILITY B1, MECHANISM M6). The split file is committed with a git sha before Phase 2.
11. **`I ⟂ C` is scoped to consuming pools only** and the LAW-14/LAW-29 contradiction is filed as a
    named contradiction rather than resolved silently (MECHANISM M1).

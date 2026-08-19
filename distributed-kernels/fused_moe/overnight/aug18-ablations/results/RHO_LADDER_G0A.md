# G-L0a (partial) — the ρ ladder on the TP8 boundary rig

**Run:** 2026-08-18 23:07–23:12 UTC, node `gbt350-odcdh2-c05-1`, 8× MI350X gfx950.
**Rig:** `m25_boundary_bench` (G25-1), prebuilt 2026-08-18 20:13, source byte-identical to
`distributed-kernels/tp8_mega/m25_boundary_bench.hip` @ `2578a72a`.
**Raw:** `runs.jsonl` (48 rows), `clocks_rho_ladder.csv` (208 timepoints × 8 GPUs).
**Node-side originals:** `~/anatomy_g0/rho/{rho_ladder.jsonl,rho_ladder_raw.txt,clocks_rho.csv}`.

## 0. Build item: NONE — `k_inner` already ships as argv

The ladder doc's "one argv" is already in the tree. `m25_boundary_bench.hip:447` parses
`k_inner` as argv 6 and `mfma_burst()` (`:113-131`) consumes it as the MFMA K-loop trip count,
scaling FLOP density per slab with no schedule change. `depth` is argv 5, `bps` (fragments per
slab) argv 7. **No BLOCKED item for the build agent on G-L0a's ρ axis.**

Provenance (node copy): source sha256 `b23b4df4…4ca6e6` (`m25_boundary_bench.hip`),
`6d605292…891b53a4` (`m25_cdar.cuh`), binary sha256 `4bb05068…3742207`,
`.text` sha256 `af4dd9ff6084fc9a2ac3545bf944ec03866ee1af80ac41fc4b3c7b2cfbf926bb`.
This is the **uninstrumented** binary — it predates the sibling build agent's `M25_LEDGER`
(E-B1/E-B2) work now in the working tree, so these walls are the parity baseline that the
instrumented arm must match.

## 1. Design

Fixed: `tokens=16384` (235 MB collective), `slab_rows=256` (K3 best), `depth=4`, `bps=2`,
`iters=7`, 256 blocks. Swept: `k_inner ∈ {2048, 3078, 6029, 9040}` × arm ∈ {compute, phased,
fused, rccl} × K=3 repeats. **Arms alternate within the session** (rep → ρ → arm nesting), per
LAW-60's cross-session-drift rule. 48/48 invocations `BOUNDARY_BENCH_PASS`, zero
`gather_timeouts`, zero `mag_timeouts`, zero `verify_errors`.

Conventions, stated once (LAW: one convention per table):

- `compute` = the MFMA floor (both bursts, no collective).
- `transport := phased − compute`, **paired within the same repeat**. This is the same
  wall-difference convention that produced the banked ρ=0.65 and β=1.506×; it is **not** a
  standalone transport measurement (that arm is still the G-L0a build item, and it is the only
  thing that turns β into a direct measurement — see §5).
- `ρ := compute / transport`, measured per point, **not assumed from `k_inner`**.
- `rccl_AR := rccl − compute`, paired — production's exposed collective on this rig.
- `gap := phased − fused` = the wall-level overlap win; `ceiling := min(compute, transport)` =
  the most that could have been hidden.

Session hygiene: first invocation of the session discarded (`compute k_inner=2048`, 1,357.1 µs).
Clocks flat and at boost throughout — per-GPU median gfx clock 2,201–2,211 MHz, **spread 0.45 %**,
hotspot 53–62 °C (GPU2 the hottest at 61 °C median, as expected). **No thermal or clock
explanation is available for anything below.**

## 2. The ladder

| ρ nominal | `k_inner` | **ρ measured** | compute µs | transport µs | phased µs | fused µs | rccl µs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.65 | 2,048 | **0.631** | 1,405.6 | 2,229.0 | 3,634.6 | 3,622.5 | 2,908.7 |
| 1 | 3,078 | **0.941** | 2,178.3 | 2,315.3 | 4,493.6 | 4,462.3 | 3,769.6 |
| 2 | 6,029 | **1.786** | 4,421.7 | 2,475.9 | 6,879.0 | 6,864.1 | 5,938.0 |
| 3 | 9,040 | **2.383** | 6,557.2 | 2,751.3 | 9,331.7 | 9,398.8 | 8,405.2 |

Medians of 3; per-config max−min spread ≤ 2.3 % except `compute` at ρ=0.63 (12.7 %, one low
outlier at 1,282 µs — the ρ=0.63 row's derived quantities are the least trustworthy of the four).
The ρ=0.63 row reproduces the banked G25-1 numbers (phased 3,653 → 3,635; fused 3,606 → 3,623;
rccl 2,909 → 2,909; compute 1,439 → 1,406), so the rig is on its banked pin.

### 2a. Exposed-collective time, per ρ

| ρ measured | ours: phased − compute | RCCL: rccl − compute | fused − rccl |
|---:|---:|---:|---:|
| 0.631 | 2,229.0 | 1,487.8 | +729.0 |
| 0.941 | 2,315.3 | 1,585.7 | +692.7 |
| 1.786 | 2,475.9 | 1,552.7 | +926.1 |
| 2.383 | 2,751.3 | 1,925.3 | +994.2 |

### 2b. Outlier flag — the `rccl` arm's `max` is a warm-up artefact, never quotable

Every `rccl` invocation carries a first-iteration outlier of **31–46 ms** against a 2.9–8.6 ms
median (worst: ρ=0.65 rep 1, median 2,893.4 µs, **max 45,719.0 µs = 15.8×**). This is RCCL's
lazy channel/graph setup on the first `ncclAllReduce` of the process, and it lands in exactly one
of the 7 iterations. The rig reports `median/min/max` and **all analysis here uses the median**,
which is immune (7 iterations, one contaminated). Same-arm `min` and `median` differ by < 1.5 %
throughout, confirming the remaining 6 iterations are clean.

**Rule for this rig: never quote `wall_us_max` on the `rccl` arm, and never raise `iters` low
enough for the outlier to reach the median** (at `iters=7` it cannot; at `iters ≤ 2` it would).
The CDAR arms show no equivalent — their transport is warm from the first iteration.

## 3. h(ρ) — the headline: **the fused arm never starts hiding**

**h_wall := (phased − fused) / (phased − compute)** — the fraction of the *exposed collective*
that fusion actually hides. Computed per repeat (paired), so the K=3 spread is real:

| ρ measured | rep 1 | rep 2 | rep 3 | **median h_wall** | spread |
|---:|---:|---:|---:|---:|---:|
| 0.631 | 0.55 % | 0.91 % | −1.00 % | **0.55 %** | 1.91 pp |
| 0.941 | 1.50 % | 1.90 % | 1.34 % | **1.50 %** | 0.56 pp |
| 1.786 | 2.78 % | 0.40 % | −0.12 % | **0.40 %** | 2.89 pp |
| 2.383 | −2.33 % | −4.08 % | −1.25 % | **−2.33 %** | 2.83 pp |

Against the alternative denominator `ceiling = min(compute, transport)` (the most that *could*
have been hidden), the same story:

| ρ measured | gap = phased − fused | ceiling | **gap as % of ceiling** |
|---:|---:|---:|---:|
| 0.631 | +12.2 µs | 1,405.6 | **0.87 %** |
| 0.941 | +34.7 µs | 2,178.3 | **1.59 %** |
| 1.786 | +9.5 µs | 2,475.9 | **0.38 %** |
| 2.383 | **−66.6 µs** | 2,751.3 | **−2.42 %** |

**Answer to G-L0a's question ("at what ρ does the fused arm start hiding?"): it does not, over
0.63 ≤ ρ ≤ 2.38.** Hiding never exceeds 1.9 % of the exposed collective in any single repeat, is
non-monotone in ρ, and **turns negative at the top of the ladder in all three repeats**
(−2.33 %, −4.08 %, −1.25 %) — at ρ=2.38 the fused schedule is 66.6 µs *slower* than the unfused
control it is supposed to beat. The sign of the top rung is not a one-rep accident: the three
`fused` medians there (9,398.3 / 9,420.7 / 9,398.8 µs) are all above all three `phased` medians
(9,331.7 / 9,308.5 / 9,364.6 µs) — **the two arms are disjoint at ρ=2.38**, with a 33.7 µs gap
between the worst phased and the best fused.

Verification of the top rung, all three reps (µs):

| arm | rep 1 | rep 2 | rep 3 |
|---|---:|---:|---:|
| compute | 6,478.8 | 6,557.2 | 6,626.7 |
| phased | 9,331.7 | 9,308.5 | 9,364.6 |
| fused | 9,398.3 | 9,420.7 | 9,398.8 |
| rccl | 8,404.1 | 8,405.2 | 8,601.2 |

**The RCCL arm leads the fused arm by 994 µs at ρ=2.38** (and by 693–926 µs at every lower rung),
so the production schedule wins the boundary at every compute intensity tested.

This **retires B1's rescue of the four G25-1 nulls.** The mechanism review scoped F1–F4 as
"null at ρ=0.65, 3.3 % of ceiling; re-tested at ρ≈3" on the theory that the rig sat ~5× below
deployment ρ. Re-tested up to ρ=2.38 — inside the deployment band's lower half (2.6–3.2) and
3.8× the rig's original ρ — **the nulls are ρ-independent, not ρ-capped.** Q10's "PARTIAL"
should be re-read: the failure is not that the experiment lacked compute to hide under.

What we would have seen if the ρ-cap hypothesis were right: gap rising with ρ toward the
ceiling, i.e. `%ceiling` climbing from 0.9 % toward tens of percent. It fell.

## 4. The finding the ladder was not designed to produce: **transport grows with the compute it follows**

`transport` is a *sequential* phase in the `phased` arm — the MFMA burst has fully retired before
the CDAR starts. It should be constant in `k_inner`. It is not:

| ρ measured | compute µs | transport µs | Δ vs ρ=0.63 |
|---:|---:|---:|---:|
| 0.631 | 1,405.6 | 2,229.0 | — |
| 0.941 | 2,178.3 | 2,315.3 | +3.9 % |
| 1.786 | 4,421.7 | 2,475.9 | +11.1 % |
| 2.383 | 6,557.2 | 2,751.3 | **+23.4 %** |

**A 4.7× longer compute phase makes the collective that follows it 23.4 % more expensive.**
Clocks were flat at boost with 0.45 % spread across all 8 GPUs and hotspot ≤ 62 °C, so this is
**not** DVFS or thermal. The candidates are cache/HBM state left by the burst (the MFMA loop
rotates operands through the fragment, so a longer burst evicts more of the staging buffer) and
rank-skew amplification (a longer compute phase magnifies inter-rank arrival spread, which the
collective then pays for at its first rendezvous).

Consequences that matter to the plan:

1. **This is why nominal ρ=3 measured 2.383.** ρ's denominator moves with its numerator, so the
   ladder self-limits; `k_inner` must be pushed to ~12,000–16,000 to reach ρ≈3–4. That top-up is
   written and staged on the node but **was not launched** (see `NODE_SESSION_1_STATE.md`).
2. **It is a second, previously unnamed component of the "+335 µs phased-arm tax"** (Q10's
   known-unknown). The tax is not a constant — it is a function of the compute phase's length.
   Any h computed as a wall-difference against a *differently-sized* compute phase inherits it.
3. It gives the still-unbuilt **transport-only arm** a second job: measure transport cold (no
   preceding burst) to separate "the collective's own cost" from "the collective's cost given the
   state a GEMM leaves behind."

## 5. β — our transport vs RCCL, now at four ρ points

| ρ measured | transport (ours) µs | rccl_AR µs | **β = ours / RCCL** |
|---:|---:|---:|---:|
| 0.631 | 2,229.0 | 1,487.8 | 1.498 |
| 0.941 | 2,315.3 | 1,585.7 | 1.460 |
| 1.786 | 2,475.9 | 1,552.7 | 1.595 |
| 2.383 | 2,751.3 | 1,925.3 | 1.429 |

β = **1.43–1.60, median ≈ 1.48**, stable across a 4.7× change in co-resident compute. This
*corroborates* the banked subtraction-derived 1.506× and sits above the 1.363× that the
store-towers logical-bytes/wall convention gave — but **it is still a ratio of two
subtraction-derived numbers** (B4's objection stands: both numerator and denominator assume the
GEMM costs the same in both arms, and §4 has just shown that assumption is ρ-dependent).
**β is not yet directly measured.** G-L0a's standalone RCCL + standalone CDAR arms remain the
build item that closes B4; what this ladder adds is that β does not move much with ρ, so the
build item can be priced at a single ρ.

Note the RCCL arm pays the §4 effect too — `rccl_AR` rises from 1,488 to 1,925 µs (+29 %) across
the same span. RCCL contends with the leftover state as well; this is not a defect unique to CDAR.

## 6. The gap that actually matters for Track 1

| ρ measured | fused − rccl |
|---:|---:|
| 0.631 | **+714 µs** |
| 0.941 | +693 µs |
| 1.786 | +926 µs |
| 2.383 | **+994 µs** |

The fused CDAR arm's deficit against production's GEMM→RCCL schedule **widens with ρ**, from
714 µs to 994 µs. Track 1's G-L3 admits fused CDAR only if h clears
`h* = 1 − (1 − I_co/T_vendor)/β`. Measured h ≤ 1.6 % of ceiling at every ρ tested, with β ≈ 1.48.
**G-L3's fused-CDAR branch does not open on this evidence** — the gate should route through
G-L1/G-L2 (RCCL-hybrid floor + shared-expert filler), or wait for the phase ledger to explain the
absent hiding, exactly as `G25_1_STATUS.md` already recommended ("one instrumented run replaces
the next four guesses").

## 7. Status against G-L0a's deliverables

| deliverable | status |
|---|---|
| h_ledger(ρ) | **NOT DELIVERED** — needs the phase ledger (sibling build agent's `M25_LEDGER`). This ladder delivers **h_wall(ρ)** only. |
| h_wall(ρ), 4 arms × 4 ρ × K=3 | **DELIVERED**, all-PASS |
| the four G25-1 arms re-tested at deployment ρ | **PARTIAL** — the ρ axis is re-tested to 2.38 and the nulls hold. The two argv-reachable variants (item granularity `bps`, injection `depth`) are scripted but unlaunched. |
| I_co absolute | **NOT DELIVERED** — needs the transport-only arm |
| direct β | **NOT DELIVERED** — needs the standalone arms; §5 gives a 4-point subtraction-derived β instead |
| ρ reaching 3 | **NOT REACHED** — 2.383, because of §4. Top-up staged. |

## 8. What this changes in the plan

- **Q10** — the "ρ-capped, 3.3 % of ceiling" scoping of the four nulls is now falsified as an
  explanation. The nulls survive to ρ=2.38 and the sign flips. Q10 stays PARTIAL, but the
  residual to explain is "why is there no hiding at any ρ", not "we never gave it enough compute".
- **Q9 / B4** — β corroborated at 1.43–1.60 across ρ; the standalone-arm build item can be
  single-ρ.
- **New known-unknown** — the ρ-dependence of the sequential transport phase (§4). It belongs in
  the plan's known-unknowns list next to the +335 µs tax, which it partly explains.

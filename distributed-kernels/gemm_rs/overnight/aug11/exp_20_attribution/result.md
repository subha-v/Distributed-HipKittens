# exp_20 — bottleneck attribution refresh at the current best config (paper Q3)

**Verdict up front. The GEMM mainloop is still the #1 pool on shapes 5 and 6 —
the only two shapes we lose to rank-1 — and it got *relatively* more dominant,
not less: 992.2 µs of shape 6's 1632.5 µs, a 60.8% share against 46% in the
oldest table. Egress is second at 23.0% and is now within ~70% of the fabric
ceiling over its own exposed pool, so it is close to done as an axis. The one
structural change since the last attribution is that `release` has been
annihilated on shape 6 — 134.9 µs → 15.0 µs, which is *below that shape's
allocation-noise floor* — but it is still 65.4 µs (10.2%) on shape 5, because
`RELEASE_GROUP_FULL_ONLY=1` refuses a partial group and shape 5's producers own
only 2 tiles each. That is the cheapest unclaimed win on a shape that carries
the gap.**

Freshness: shape 6 `full` = **1632.5 µs**, +1.0% over the known best-of-arm
1616.63 and independently confirmed by a same-config M7 run at 1614.54 µs
(ratio 1.011). It is **not** the stale 1777.9 and **not** the stale 2527.

---

## 1. The ranked pool table, all six shapes

Microseconds, pipelined wall per world-8 operation, 40 iterations per cell.
`floor` is that shape's allocation-noise floor; a pool inside it is marked `*`
and **is not measured**. Full data: `ablation.json`.

| # | shape | full | GEMM | egress | sync | reduce | release | floor | ranking |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 64×7168×18432 | 67.1 | 2.5 | 2.6 | −1.0\* | −0.3\* | −0.5\* | 2.34 | **host-bound, unrankable** |
| 2 | 512×4096×12288 | 67.5 | 1.1 | 2.6 | 2.6 | −0.8 | 0.9 | 0.63 | **host-bound, unrankable** |
| 3 | 2048×2880×2880 | 87.7 | 16.1 | 20.6 | 16.2 | 7.9 | 6.5 | 1.41 | egress > sync ≈ GEMM |
| 4 | 4096×4096×4096 | 203.5 | 38.5 | **86.2** | 31.8 | 18.9 | 18.3 | 4.90 | **egress** > GEMM > sync |
| 5 | 8192×4096×14336 | 641.9 | **305.2** | 179.7 | 52.9 | 51.7 | 65.4 | 13.93 | **GEMM** > egress > release |
| 6 | 8192×8192×29568 | 1632.5 | **992.2** | 376.0 | 168.6 | 79.8 | 15.0\* | 69.87 | **GEMM** > egress > sync |

As a share of `full`:

| # | GEMM | egress | sync | reduce | release |
|---|---:|---:|---:|---:|---:|
| 3 | 18.4% | 23.5% | 18.5% | 9.0% | 7.4% |
| 4 | 18.9% | **42.4%** | 15.6% | 9.3% | 9.0% |
| 5 | **47.6%** | 28.0% | 8.2% | 8.1% | 10.2% |
| 6 | **60.8%** | 23.0% | 10.3% | 4.9% | 0.9% |

**The ranking is shape-dependent and inverts at shape 5.** Egress dominates the
mid shapes (3, 4); the mainloop dominates the large ones (5, 6). Since the graded
gap to rank-1 lives entirely in shapes 5 and 6, **the mainloop is the target**,
and the pool ranking agrees with the independent `waves × k_iters` argument in
`HANDOFF.md` rather than contradicting it.

### Shapes 1 and 2 have no resolving power and must not be ranked

Every pool on shape 1 is at or inside its 2.34 µs floor, and three are negative.
M7 labels shape 1 `bound = HOST` at 10.39× SOL: host issue is 62.34 µs of a
66.41 µs wall. Shape 2 is the same in truth — M7 prints `device` for it only
because one of three rotations came in at 114.12 µs against 66.42 and 67.80,
inflating its mean to 82.78 µs (stdev 27.15). Against its **median** 67.80 µs,
host issue of 62.43 µs is 92% of the wall, i.e. shape 2 is host-bound too.
Nothing about the kernel can be concluded from either row.

## 2. Schema of `ablation.json` (stated explicitly, as required)

Top level: `experiment`, `what`, `config`, `clocks`, `freshness_gate`,
`noise_floor_pct`, `schema`, `caveats`, `shapes[]`, `ranking[]`.

Each `shapes[]` record:

| key | meaning |
|---|---|
| `shape_index`, `shape`, `m`, `n`, `k`, `has_bias` | shape identity; 1-based graded index |
| `bm`, `bn`, `bk`, `nr` | the config row actually compiled |
| `full_us` | unablated scratch build; the denominator for every share |
| `arms_us.{nomain,emitlocal,nored,noproto,norelease}` | raw per-arm wall time |
| `stages_us.gemm` | `full − nomain` — k-loop MFMA math + A/B global→LDS traffic |
| `stages_us.egress` | `full − emitlocal` — the peer/xGMI cost, with byte volume and store instruction count held fixed (same stores, local slot) |
| `stages_us.reduce` | `full − nored` — 8-source pull, fp32 sum, bf16 store |
| `stages_us.sync` | `full − noproto` — credit waits, ready publishes, reducer participation |
| `stages_us.release` | `full − norelease` — the `buffer_wbl2 sc0 sc1` L2 writeback + its `vmcnt(0)` drain |
| `stage_share` | `stages_us / full_us` |
| `floor_pct`, `floor_us` | allocation-noise floor for this shape |
| `stage_clears_floor` | per stage, `|delta| ≥ floor_us` |
| `ranked_stages` | stages by absolute µs, largest first |
| `stage_sum_us`, `stage_sum_over_full` | **not a budget** — see below |
| `sol_us`, `x_sol` | published task.yml SOL and `full_us / SOL` |
| `geometry` | arithmetic on the shape table: `k_local`, `k_iters`, `gemm_tiles`, `producer_waves`, `waves_x_k_iters`, `tiles_per_cta`, `effective_release_group` |
| `m7` | same-config M7 gate: `mean_us`, `device_max_us`, `samples_us`, `host_issue_us_per_op`, `bound`, `full_vs_m7_mean` |

**Single-cut deltas overlap wherever the phases overlap, so they do not sum to
`full` and are not a budget.** `stage_sum_over_full` runs 0.05 (shape 1, all
noise) to 1.02 (shape 5). Rank pools by size; never subtract one from another.

`ranking[]` carries each stage's shape-6 µs and share, its summed µs over all
six shapes, its mean share, and the list of shapes where it clears its floor.

## 3. What changed versus the previous attribution

Two prior tables exist and both are superseded.

| pool | table A (`RESULTS.md`, pre-WGM, total 2861.7) | table B (post-exp_08, total 1777.9) | **now (total 1632.5)** | vs B |
|---|---:|---:|---:|---:|
| GEMM | 1318 (46%) | 1142.8 (64.3%) | **992.2 (60.8%)** | −13.2% |
| egress | 920 (32%) | 412.4 (23.2%) | **376.0 (23.0%)** | −8.8% |
| sync | — | 191.0 (10.7%) | **168.6 (10.3%)** | −11.7% |
| reduce | — | 86.5 (4.9%) | **79.8 (4.9%)** | −7.7% |
| release | 250 (9%) | 134.9 (7.6%) | **15.0 (0.9%)** | **−88.9%** |
| total | 2861.7 | 1777.9 | **1632.5** | −8.2% |

Three readings:

1. **The five landed changes shrank every pool, and the ranking survived.**
   GEMM, egress, sync and reduce all fell 8–13% in absolute µs while their
   *shares* barely moved (egress 23.2 → 23.0%, sync 10.7 → 10.3%, reduce
   unchanged). The shape of the problem is stable; the total simply got smaller.
2. **`release` is the one pool that structurally disappeared, on exactly one
   shape.** `RELEASE_GROUP=4` with `RELEASE_GROUP_FULL_ONLY=1` groups only when a
   producer CTA owns ≥ 4 tiles. Of the six graded shapes, **only shape 6
   qualifies** (`tiles_per_cta` = 1/1/1/1/**2**/**4**), so the effective release
   group is 4 on shape 6 and 1 everywhere else. The counter pass corroborates the
   mechanism independently: on shape 6 the share of L2 writebacks pushed by the
   release's `buffer_wbl2` fell to **13.1%**, against 56.4% on shape 5 and
   79–97% on shapes 1–3.
3. **Pre-registration confirmed on every count.** `plan.md` predicted GEMM's
   share above 46%, egress and release smaller in absolute µs, and the mainloop
   as the unambiguous #1. All four hold. Egress did not stay at ~30%, so the WGM
   win did come from egress, as claimed.

### The finding to act on: shape 5 pays 65.4 µs for releases it could group

Shape 5's producers own 2 tiles each, and 2 < `RELEASE_GROUP` = 4, so
`RELEASE_GROUP_FULL_ONLY=1` drops it to a release **per tile** — the exact
configuration E3 was built to remove. It costs **65.4 µs, 10.2% of shape 5's
641.9 µs**, comfortably above its 13.93 µs floor, and shape 5 is one of the two
shapes carrying the whole remaining graded gap. Shape 6 got a −88.9% collapse
from the same mechanism. Either `RELEASE_GROUP=2` or permitting a partial group
would target this; it is a one-constant change with a measured pool behind it.
(Protocol-review first — E3's grouping is a publication-ordering change.)

## 4. Counter pass — `counters.json`

One `rocprofv3` 1.1.0 pass at the same config, **all six graded shapes** (the
plan called for shape 6 plus one mid shape), three counter groups per shape plus
two shape-6 controls. Twenty cells, and **every one reported `correct=1
tight=1 errors=none` under instrumentation** — the validity gate that matters,
because profiler serialization can starve the bounded credit/ready spins, make
the kernel return early on its sticky error bit, and silently deflate every
traffic counter.

| # | fabric MB | useful MB | amp | 64 B share | EA wr latency (cyc) | GB/s over `full` | wb by capacity | wb by release | flags % of fabric reqs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.87 | 0.80 | 1.083 | 88.5% | 3100 | 13.0 | 0.0% | 78.4% | **12.70%** |
| 2 | 3.69 | 3.67 | 1.004 | 98.9% | 5866 | 54.6 | 2.9% | 94.6% | 3.10% |
| 3 | 10.34 | 10.32 | 1.001 | 99.6% | 4287 | 117.9 | 2.2% | 96.9% | 1.11% |
| 4 | 29.38 | 29.36 | 1.0005 | 99.9% | 2843 | 144.4 | 60.1% | 39.4% | 0.39% |
| 5 | 58.76 | 58.72 | 1.0006 | 99.9% | 3025 | 91.5 | 43.0% | 56.4% | 0.19% |
| 6 | 117.52 | 117.44 | **1.0007** | **99.9%** | **2090** | 72.0 | 86.0% | 13.1% | 0.10% |

`useful` = 7/8 · M·N·2: a rank keeps one of the eight output slices it produces
and sends seven off-rank.

**Against the reference values.** exp_08 measured shape 6 at 117.48 MB fabric
against 117.44 MB useful (1.0003×) with 99.9% of EA writes at the full 64 B.
This pass reads **117.52 MB / 117.44 MB = 1.0007×, 99.9% at 64 B** — the same
numbers, on a kernel that is 8.2% faster. Nothing material differs, and that is
itself the result: five landed changes moved zero bytes differently.

Four things this pass adds beyond exp_08:

1. **No write amplification on ANY shape, not just 5 and 6.** Shapes 2–6 are
   1.0005–1.0043×. The "coalesce the peer stores" axis stays closed across the
   whole scored set, not just where it was checked.
2. **Shape 1 is the only row where the fabric carries materially more than the
   payload (1.083×), and it is not amplification — it is the protocol.** The
   emit-local control measures the flag traffic directly: with every payload byte
   routed to the local rank, **1,792 off-die requests remain**. That is 0.10% of
   shape 6's fabric requests and **12.70% of shape 1's**, because publications
   and credits are constant in the world size while the payload is not. Shape 1
   is also the only row not already at maximal store width (88.5% at 64 B).
3. **EA write latency on shape 6 is down to 2090 cycles**, from 2887 post-WGM and
   4705 pre-WGM in exp_08 — a further −27.6% with the request count unchanged at
   2,361,648, which is the queueing signature of the fabric being less contended,
   not of fewer or wider transactions.
4. **The writeback origin split now varies 6× across shapes** and tracks the
   effective release group exactly (item 2 of §3).

**Bandwidth, stated carefully.** Over the whole kernel wall, shape 6 moves
117.52 MB in 1632.5 µs = 72.0 GB/s. Over its *exposed egress pool* of 376.0 µs
it is **312.6 GB/s** — against ~448 GB/s of nominal per-GPU xGMI, so ~70% of the
ceiling, consistent with the 284 GB/s / "315–336 achievable" figures already in
`HANDOFF.md`. Shapes 5 and 4 give 327.0 and 340.8 GB/s the same way. This is
**not** comparable to exp_08's "102 GB/s", which divided by a 1149.9 µs egress
pool from a much earlier config. On shape 3 the same arithmetic yields 501.9
GB/s, i.e. *above* the nominal aggregate — which proves the egress delta there is
not a pure serial egress time but a partly-overlapped one. Treat pool-relative
bandwidth as a lower bound on the instantaneous rate, never as a measurement of
it.

`TCC_EA0_WRREQ_STALL` was deliberately **not** collected: it reads ~81 M cycles
on this ASIC while every `*_CREDIT_STALL` sub-counter reads ~0, so it cannot
support an argument either way.

### Schema of `counters.json`

`cells{tag}` is the provenance — one entry per (shape, arm, counter group) with
`counters_per_rank_per_launch` (each counter summed over the measured dispatch
window, divided by agents × dispatches; `TCC_*` are sums over 16 TCC × 8 XCC),
the driver's verbatim `prof_line`, and `usable`. `shapes[]` is the per-shape
rollup merged from that shape's three groups, carrying `fabric_bytes_mb`
(+`_bounds`), `useful_bytes_mb`, `amplification`, `ea_write_requests`(`_64b`,
`_dram`), `wrreq_64b_share`, `fabric_requests`, `bytes_leaving_l2_mb`,
`ea_read_requests`, `fabric_flag_requests_measured`,
`flag_share_of_fabric_requests`, `ea_write_latency_cycles`, `tcc_cycles`,
`l2_write_requests`, `l2_writeback_lines`, `l2_writeback_capacity`,
`l2_writeback_wb_op`, `writeback_capacity_share`, `writeback_release_share`, and
`achieved_fabric_gbps`. `tags`: `s<shape>_<arm>_g<group>`.

Only four TCC counters fit one pass, so a shape's volume (g1), latency and
writeback total (g2) and writeback origin/mtype mix (g3) come from three separate
dispatch windows of the same config. Counts are clock- and schedule-independent,
so merging them is sound — but a *ratio* is only meaningful within one pass,
which is why `TCC_EA0_WRREQ_LEVEL` and `TCC_EA0_WRREQ` were collected together
and `ea_write_latency_same_pass` records it.

## 5. What had to be repaired to obtain this table

**`harness/exp_ablation.py`: three of nine anchors were dead, and `generate()`
raises on the first one, so the attribution was not obtainable at all.** Repaired
(anchor text only — no arm, cut, macro, flag or shape was added, removed or
changed, so the instrument still measures what it measured before):

| anchor | was | now |
|---|---|---|
| 2 — closes the mainloop cut | quoted the comment `"// ---- egress: per-band credit wait, emit, one release, publish ---"`, which E3 rewrote to `"...credit wait, then emit"` | the k-loop's own last statements plus its closing brace (`load_commit<NT>(As[(k + 1) & 1], …)`), which occurs exactly once |
| 3 — the emit-destination cut | a two-line anchor whose continuation line needed 16 spaces; E3's group loop moved the emit body to 20 | the single declaration `int dest_eff = dest;` at its true indent |
| 9 — closes the protocol cut | continuation line at 12 spaces, actual 16 | both lines at their true indent |

Two lessons, written into the file so the next session does not re-pay them:

- **Never anchor on comment text.** Two of the three failures were comment
  rewrites by E3 and exp_09.
- **A leading-whitespace mismatch on an anchor's FIRST line still matches** (the
  tail of the real indent satisfies it), which is why anchors 1 and 8 kept
  working by luck at the wrong indent while 2, 3 and 9 — whose mismatch is on a
  *continuation* line, where the newline pins column 0 — silently went to zero.
  All anchors are now at exact indentation and quote only executable code.

**The stale scratch was disguised as fresh.** `harness/ablate/gemm_rs_ablate.cpp`
carried today's mtime and all five gate macros, so it looked like a current
generation. It was **30,610 bytes against the 50,556 this run generates** — a
kernel 40% smaller, i.e. generations old. Its mtime was today's only because
`push.ps1` rewrites every source mtime. Confirms the ledger's warning: mtime is
not a freshness test here, and `reattribute.sh`'s force-remove is load-bearing.

## 6. Method, and what would invalidate this

- Driven by `tools/reattribute.sh` (not hand-rolled) for its two guards: it
  force-removes all six arm `.so` files and the generated scratch, then asserts
  shape 6's `full`. All six arms were rebuilt in this run (timestamps
  03:59:44–04:00:33).
- **The freshness gate was tightened, deliberately.** `plan.md` pre-registered
  1617 ± 12% = [1423, 1811] µs; that window *admits the stale 1777.9*, so it was
  run at **1670 ± 6% = [1570, 1770]**, which excludes it. Measured 1632.5 passes
  both. Independent confirmations: M7 at 1614.54 µs (ratio 1.011), and the
  production `.so` produced **byte-identical** counters to the freshly rebuilt
  `full` arm (2,361,648 / 2,359,296 / 524,848 EA writes; fabric 117.48–117.56
  MB), so the production build is current too.
- Node held exclusively; 0 foreign KFD pids at every launch; all 8 GPUs on
  `perf_determinism` before and after. Idle `sclk` reads 120 MHz, which is
  expected under `perf_determinism` and is why warmup is duration-based.
- Counter labels re-validated by the known-answer control: emit-local drops
  fabric requests from **1,836,800 → 1,792 (0.2%)**, a 1025× collapse. If that
  had not collapsed, `WRREQ − WRREQ_DRAM` would not mean "off-die" and none of §4
  could be read.
- One process drives all 8 devices here. That is not the evaluator's topology,
  so these µs are not directly comparable to graded per-call numbers.

**What would invalidate it:** a `full` outside the gate; any cell with a nonzero
error bit; an arm `.so` older than the scratch; a counter pass whose `prof_line`
shows `errors` other than `none`. None of these occurred.

## 7. Ranked next actions, off this table

1. **The mainloop, on shapes 5 and 6** — 992.2 µs (60.8%) and 305.2 µs (47.6%),
   the two shapes that carry the entire graded gap. `waves × k_iters` is 464 and
   112 and cannot be cut by retiling: rows 4/5/6 are pinned at the 64 KB LDS cap
   by double buffering (`2·(BM+BN)·BK·2 ≤ 65536`). The open lever is the
   double-buffer LDS cost itself, per `HANDOFF.md`.
2. **`release` on shape 5** — 65.4 µs, 10.2%, above floor, one constant
   (`RELEASE_GROUP=2`, or allow a partial group). Cheapest item on the list;
   needs protocol-review.
3. **Egress on shape 4** — 86.2 µs, 42.4%, the largest *share* of any pool on any
   shape. Note it is a 201 µs shape whose fabric already runs at ~341 GB/s over
   its pool, so headroom is limited.
4. **`sync`** — 168.6 µs on shape 6 (10.3%) and 15–19% on shapes 3 and 4.
   Untouched by any landed change and never separately attacked.
5. **Not egress-width, not amplification.** Closed again, now across all six
   shapes: 1.0005–1.0043× with 99.9% at 64 B. The only exception is shape 1
   (88.5% at 64 B, 12.7% flag traffic) — and shape 1 is host-bound, so there is
   nothing to win there either.

## 8. Files

| file | contents |
|---|---|
| `ablation.json` | the table of §1 + geometry, ×SOL, M7 cross-check. Schema in §2 |
| `counters.json` | 20 cells + 6 per-shape rollups. Schema in §4 |
| `m7_results.json` | same-config M7 gate, 3 rotations × 50 iters, all shapes correct, geomean 215.44 µs (208.4 µs using shape 2's median instead of its outlier-contaminated mean) |
| `reattribute_run.log`, `counters_run.log`, `m7_run.log` | raw runs |
| `prof/` (node) | per-cell rocprofv3 CSVs |

Note for the ledgers, which this experiment does not own:
`aug11/PLOTS.md` still lists exp_20 as **running**, and `aug11/LESSONS.md` and
`STATUS.md` have no exp_20 entry yet.

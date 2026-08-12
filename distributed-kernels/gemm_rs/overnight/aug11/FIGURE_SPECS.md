# FIGURE_SPECS.md — paper figure specifications retrieved from the sibling branch

Retrieved read-only from `origin/codex/distributed-hipkittens-scaffold` via
`git show` (no checkout, no worktree switch) on 2026-08-12. Every claim below
cites its source path on that branch. Nothing here is my inference unless
marked `[NOTE]`.

**`docs/distributed/PAPER.md` WAS FOUND** on the sibling branch (18,030 B,
273 lines). Figure definitions below come from it plus the two experiment
specs and the sibling charter.

Sources used:

- `docs/distributed/PAPER.md`
- `distributed-kernels/fused_moe/overnight/aug11/CLAUDE.md` (the sibling
  charter — this is where the committed JSON/CSV deliverable schemas live)
- `distributed-kernels/fused_moe/overnight/aug11/exp_22_fig7_saturation/plan.md`
- `distributed-kernels/fused_moe/overnight/aug11/exp_22_fig7_saturation/design.md`
- `distributed-kernels/fused_moe/overnight/aug11/exp_23_fig10_timeline/plan.md`
- `distributed-kernels/fused_moe/overnight/aug11/exp_23_fig10_timeline/design.md`
- `distributed-kernels/fused_moe/overnight/aug11/PROMPT.md`

The Fig-10 analog spec is at `exp_23_fig10_timeline/`, **not** a `*fig10*` file
under a different name; there is no other `*timeline*` file under that aug11
directory (`git ls-tree -r`).

---

## 0. Figure numbering: a real conflict between PAPER.md and the charters

Two numbering systems are in the tree and they disagree. Use the charter
numbering for deliverable naming; treat PAPER.md's inline numbers as the
prose skeleton's own draft numbering.

| number | per `fused_moe/.../aug11/CLAUDE.md` (charter) | per `docs/distributed/PAPER.md` (prose) |
|---|---|---|
| Fig 1 | — | §1: comm share of layer time (opening measurement) |
| Fig 2 | exp_22 saturation curves (Q4) | §2.1 mentions "the tile/token flow figure in COMET Figure-2 style" |
| Fig 3 | exp_23 per-layer resource timeline (Q4) | — |
| Fig 4 | exp_35 knob waterfall (Q1, "the money figure") | §4.2: "**the paper's Figure 4**" = the worked single-tile edge example (~30 coordination ops vs amortized 1e-4) |
| Fig 5 | exp_37 placement adjudication (Q2) | — |
| Fig 6 | nc-major task reorder data | — |
| Fig 7 | exp_36 sensitivity (Q5) | — |

`[NOTE]` The GEMM-RS charter's "exp_23 → paper Fig 3" and "exp_23 → Fig 4 /
Q1 money figure" mapping matches the sibling **charter** column, so the
GEMM-RS queue is internally consistent with the sibling agent's naming.
PAPER.md §3.2 also promises a figure for Finding 1 ("gets its own figure")
with no number.

---

## 1. Fig 2 / Q4(a) — resource saturation vs CTA count (NanoFlow v1 Fig 7 analog)

Source: `exp_22_fig7_saturation/plan.md`, `.../design.md`, charter item 2.

### What it plots

Three panels, x = CTA count, y = per-resource throughput, "for the three
resource classes this kernel actually uses, **plus a concurrent overlay
NanoFlow does not have** (each curve measured again while the M7-shaped GEMM
runs on the remaining CTAs)" (`plan.md`).

Plot spec verbatim (`design.md` "Plot spec (fig7_saturation.py)"):

> 3 panels; x = CTAs, y = TFLOPS / GB/s / GB/s; series per panel: isolated
> (MLP depths as line styles for panel c), concurrent, ceilings as dashed
> horizontals (76.8, 537.6, 8000, 148). Knee annotations = smallest C reaching
> 90% of each series' plateau.

### Modes / arms (`design.md` kernel-body table)

| mode | body | work unit | metric |
|---|---|---|---|
| a (MFMA) | M7-shaped task loop over a synthetic tile_desc list | one tile | `2*M*N*K*tiles / t` TFLOPS |
| b (HBM) | M8-shaped slot reduce: read 8 bf16 slots + write 1, 14-chunk loop, NT=4 | one 14,336 B row | `(8+1)*bytes*rows / t` GB/s |
| c (xGMI) | the real in-kernel pusher `store_peer_packets` of 14,336 B rows via `translate_peer<8>` | one row push | `bytes_pushed / t` GB/s |

Mode c parameters: MLP depth d in {1,4,8} (d row-loads before first store);
fanout `single` (all rows to rank `(me+1)%8`, prices ONE link) or `rr7`
(row i to peer `i%7` skipping self, prices aggregate egress).

Isolated arm has "**NO probe atomics, NO flags, NO fences** … this curve
prices payload transport only. A `--protocol` flag optionally re-adds the
per-group probe RMW + per-flush `thread_release<system>` so the
protocol-vs-payload gap is measurable *inside this ubench*" (`design.md`).

### Sweep grid (`plan.md`, verbatim table)

| axis | points |
|---|---|
| mode | a=GEMM, b=HBM reduce, c=xGMI push |
| CTAs (mode a) | 32, 64, 96, 128, 160, 192, 224, 256 |
| CTAs (mode b) | 8, 16, 32, 64, 128, 256 |
| CTAs (mode c) | 1, 2, 4, 8, 16, 32, 64 |
| MLP depth (mode c only) | 1, 4, 8 |
| peer fanout (mode c only) | single-peer, round-robin-7 |
| overlay | isolated; concurrent (role-split with mode-a tasks on 256−C CTAs) |

### Controls demanded

- **C-matched reserve-only control.** "Concurrent overlay runs modes b and c
  only, against a **C-matched reserve-only control** (exp_20 lesson:
  192-with-traffic vs 192-without, never vs 256)" (`plan.md`).
- Two points per concurrent measurement (`design.md`): (1) C reserved,
  resource role **idle-spins** = capacity control; (2) C reserved, resource
  role live = measurement. Report resource GB/s **and** compute slowdown =
  live/control on the same 256−C CTAs.
- **Checksum read-back once per config** — "a silently failing store must not
  fake bandwidth" (`plan.md`); u64 xor-fold of pushed payload, 8 B per rank.
- Fixed global work list, grid-strided by dense role id, "total work identical
  at every CTA count, so time differences are pure throughput" (`design.md`).
- Per-role `s_memrealtime` start/end by tid 0 into a `[256][2]` u64 buffer;
  "concurrent arm roles finish at different times; wall time alone attributes
  nothing" (`plan.md`).
- 5 rotations per point, **median**; each point sized to 10–50 ms of kernel
  time (mode c: >= 4 GB pushed per launch) (`plan.md`).
- Sizing: mode a 4,096 tiles; mode b 131,072 rows; mode c 292,000 rows ~ 4.0 GB.
- Clocks: `rocm-smi --showclocks` recorded before/after each point into the
  result JSON; **no clock pinning** on the sibling node, "drift shows up in the
  rotation spread" (`design.md`). `[NOTE]` GEMM-RS charter overrides this: pin
  1900 before any timing.
- Labeled `valid_diagnostic`; ubench, so no correctness ladder, no campaign
  arms touched (`plan.md`).
- Rejected alternatives (`design.md`): rocprof counters as the bandwidth source
  (cannot separate two roles inside one launch; kept as one-time cross-check on
  an isolated point); `hipMemcpyPeer` (prices the copy engine / SDMA, not the
  in-kernel pusher); reusing the full megakernel mode as the ubench (conflates
  protocol with payload).

### Committed data schema (verbatim, sibling charter line 52)

> Deliverables: `saturation.json` (every point: mode, CTAs, MLP, fanout,
> isolated/concurrent, GB/s or TFLOPS) + the knee summary in result.md.

That is the only schema text committed; there is no field-level JSON example
anywhere on the branch (`git grep -i schema` over `aug11/` and `docs/`).

### Pre-registered predictions (`plan.md`, verbatim)

- **H1**: xGMI single-link push saturates by 8 CTAs at MLP>=4 (competition
  analysis: winners provision ~1 CTA per XCD per link; gemm-rs rank02 holds
  full rate with 8–16 CTAs via load depth).
- **H2**: aggregate-egress push saturates by 16–32 CTAs.
- **H3**: the concurrent xGMI curve is depressed well below the isolated one at
  equal C, and the depression grows with the *protocol* knobs (probe atomics),
  not payload bytes — consistent with exp_20's protocol-not-payload verdict.
- **H4**: MFMA TFLOPS vs CTAs is linear (one block per CU; no oversubscription
  regime exists on CDNA4).
- **Falsifier**: "if the isolated single-link curve needs >=32 CTAs to reach
  ~75% of 76.8 GB/s at any MLP depth, the 'tiny topology-sized pool' story is
  wrong for our pusher shape and A1's C=64 optimum is *bandwidth-limited*, not
  protocol-limited — that inverts the M4-first priority."

Three deliverable numbers (`plan.md`): knee of the xGMI push curve = the
topology-correct pool size C; concurrent/isolated ratio at that C; MFMA
capacity curve at 256−C confirming the fitted 256/(256−C) tax law.

---

## 2. Fig 3 / Q4(b) — per-layer resource-utilization timeline (NanoFlow v2 Fig 10 analog)

Source: `exp_23_fig10_timeline/plan.md`, `.../design.md`, charter item 3.

### What it plots

"One MoE layer/epoch on the x-axis (~7 ms), three strips per arm — CTAs-in-MFMA
%, HBM GB/s, xGMI GB/s — for three arms" (`plan.md`). Plot spec verbatim
(`design.md`):

> 3×3 grid: rows = MFMA % / HBM GB/s / xGMI GB/s, columns = B0 / pf6gm / mps.
> Shared x (µs from epoch start), per-row shared y. Phase boundaries as faint
> verticals on megakernel columns. Annotations: B0 comm-share % (width of its
> xGMI blocks), MPS M7 elongation vs pf6gm (the interference term).

### Arms and data sources (`plan.md`, verbatim table)

| arm | data source |
|---|---|
| B0 RCCL-eager (sequential dispatch → GEMM → combine) | torch-profiler / rocprof kernel trace; each kernel interval maps to one resource strip |
| `pf6gm_mega` (homogeneous) | in-kernel phase event ring |
| `mps_mega` C=64 g=1 mode 2 (ratchet) | in-kernel phase event ring |

The argument the figure must make: B0's strips are mutually exclusive (the
width of its xGMI blocks IS the comm share of the layer), homogeneous partially
interleaves, the specialized arm holds the MFMA strip through the GEMM phase
while xGMI runs beneath it, "and M7's visibly longer strip vs homogeneous is
the interference term, honest in the same picture" (`plan.md`).

### Instrumentation contract (`design.md` event ring, verbatim table)

| field | value |
|---|---|
| slot | next free MPS descriptor slot (append after 62; bump `K0P6_MPS_D_LEN`) |
| shape | `u64[256][24]` = 48 KB, agent visibility, zeroed at M0 |
| entry | `(phase_id << 56) \| (s_memrealtime() & 0x00FFFFFFFFFFFFFF)` |
| writer | tid 0 of each CTA, plain relaxed agent store, one per boundary crossed |
| reader | host, after launch completes (no in-flight reads) |

Phase-ID enum is fixed and ordered (write order; absence = phase not run by
that CTA): 0 M0_START, 1 M1_START, 2 M2_START, 3 M2_PASSB, 4 M3_BARRIER,
5 M4_PLAN, 6 M5_BARRIER, 7 M6_START, 8 M7_START, 9 M7_TASK_DRAIN, 10 M75_START,
11 SVC_START, 12 SVC_STRIPE_DONE, 13 SVC_FLUSH, 14 M8_START, 15 M9_START,
16 EPOCH_END. Compute CTAs write <= 17 events; service CTAs add one per stripe
drain and one per flush group; ring depth 24; "assert on overflow by dropping
with a count in entry 23, never by wrapping" (`design.md`).

Binning (`design.md`): **10 µs bins** over [min ts, max ts] of rank 0.
`mfma[b]` = |{CTAs whose interval [M6_START..M7_TASK_DRAIN] covers b}| / 256.
`hbm[b]`, `xgmi[b]` = sum over phase intervals covering b of
(phase bytes / phase duration). Bytes per phase are **analytic on the host**,
from device-published counts — explicitly no in-kernel byte counters, because
they "add atomics to the paths being pictured" (`design.md`, alternatives
rejected).

### Controls / methodology demanded

- **Diagnostic-arm policy**: "The instrumented run is a **diagnostic arm**
  (A13 policy). Flag OFF for all campaign timing. Its µs are never reported as
  performance numbers" (`plan.md`).
- **No protocol changes**: plain agent-scope stores by tid 0 at phase
  boundaries, <= ~24 events per CTA per epoch, no new fences/atomics.
- **Resource-tuple parity gate**: "compile with the flag present but disabled
  and confirm SGPR/VGPR/AGPR/scratch/LDS parity vs the current ratchet build —
  an instrumented-only code shape would make the timeline unrepresentative"
  (`plan.md`).
- **Same routing seed across all three arms** so phases align visually; rank 0
  plotted, rank-max supplementary, "state symmetry, don't average ranks".
- **Integral check** (must pass before the figure ships): "each strip's
  integral over the epoch equals the known total bytes/FLOPs for that arm
  (±10%)".
- **Counter cross-check**: `amd-smi metric` per-link xGMI throughput sampled
  over a steady-state loop matches the xGMI strip's epoch average (±20%); one
  rocprof MFMA-busy run validates the occupancy proxy once.
- Honest labelling: the MFMA row is "CTAs in MFMA phase — an occupancy proxy",
  not MFMA utilization (`plan.md`).
- The B0 arm needs **no kernel work**: kernel-name → resource-class map checked
  in as `b0_kernel_map.json`; mapping given as `rccl*`/`mori*` → xGMI,
  `*gemm*`/`*mfma*` → MFMA, scatter/quant/combine → HBM. "Strips are binary
  occupancy bars — exactly NanoFlow's non-overlap panel" (`design.md`).
- Rejected: rocprof PC-sampling/counter timelines as the megakernel source (no
  per-phase attribution inside one 7 ms launch); LDS-staged event buffering
  (LDS fully committed).

### Committed data schema (verbatim, sibling charter line 61)

> Deliverables: `rank0_events.json` per arm + `timeline_bins.csv` + the two
> validation checks (integral, amd-smi cross-check) recorded in result.md.

Validation command committed (`plan.md`):

```bash
K0_MPS_CFG="C=64,g=1,mode=2,flush_rows=1" K0_MPS_TRACE=1 \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh exp23_smoke 1
python3 plot_timeline.py runN/rank0_events.json --check-integrals
```

No field-level schema for `rank0_events.json` or `timeline_bins.csv` is
committed; the entry encoding above plus the binning rules are the whole
contract.

---

## 3. Fig 4 / Q1 — the knob waterfall (the money figure)

Sources: `PAPER.md` §6 Q1; sibling charter item 5 (exp_35).

PAPER.md Q1 verbatim: "**How much does each decision contribute?** The
waterfall: homogeneous → +epilogue-carried payload → +injection bound →
+arrival-sized signals → +task reorder; both kernels. (The paper's money
figure.)" — i.e. the rung ORDER is fixed by the paper and both operators must
present the same ladder shape.

Sibling rungs (charter item 5): (a) `pf6gm_mega` homogeneous baseline;
(b) mode 12 unthrottled (epilogue-carried payload alone); (c) mode 12 + depth 4
(injection bound; the ratchet); (d) mode 14 (coarse signals), if gated green;
(e) + nc-major task order, "if not, log it as the named missing rung".
Methodology: "One campaign per rung, same seeds, full ladder each".

### Committed data schema (verbatim, sibling charter)

> Deliverable: `waterfall.json` — rung, arm_p50_us per arm, ratio vs
> production, delta vs previous rung.

Pre-registered numbers on the sibling side: ratchet = 6,568.0 µs = 0.8522×
production; mode-14 rung band **5,990–6,440 µs**, and "a result above 6,568
falsifies the granularity rung and gets reported as such" (charter item 4).
"**The degeneration is a finding**: with the protocol gone the service pool has
no job — bank the number and say so."

---

## 4. Q1–Q6 as PAPER.md §6 states them (verbatim intent, condensed)

- **Q1 — contribution of each decision.** The waterfall above. Money figure.
- **Q2 — do dedicated communication blocks ever win?** "Paired best-vs-best
  (pool design at C=64 vs final design), the pool-size sweep at the final
  design (flat→degenerate), the reducer-count sweep, and the leaderboard
  replication." Sibling instance = exp_37 (charter Fig 5): paired same-run
  mode-2 best vs mode-12 best vs mode-14 with C in {0, 8, 16}, prediction
  **flat**. GEMM-RS instance = the NR sweep {8,16,32,48} placement-flatness
  exhibit.
- **Q3 — where does the time go now?** "Phase attribution at the final
  configuration: the two GEMM mainloops are 78% of the MoE kernel; the
  epilogue's fabric surcharge sits at 78% of the wire ceiling (<=211 µs
  theoretical headroom); communication is no longer the bottleneck — the
  frontier moves back into single-GPU GEMM quality, and the residual GEMM-RS
  gap to the leaderboard rank-1 is GEMM quality, reported as such." Sibling
  deliverable: `exp_33_attribution/phase_stamps.json` + a result.md table;
  arms `production`/`pf6gm_mega`/`mps_mega`; per-phase stacked bar split
  M0–M2 / M3–M5 / M6 / M7-as-GEMM-vs-epilogue-surcharge / M8M9; **rank-max and
  rank-0 both**, never averaged.
- **Q4 — resource-level proof of overlap** (NanoFlow-style): the two figures in
  §1 and §2 above. "the gap between those two curve families is Finding 1 as a
  figure (exp_22)".
- **Q5 — sensitivity: batch size, sequence length, imbalance.** PAPER.md:
  "M in {512…4096} × routing skew std in {0…0.05}; vLLM chunked-prefill on real
  prompts for the serving shape. Pre-registered: small M shifts value toward
  signal coarsening and fusion; large M toward injection bounds and link order;
  skew is the one regime where a waiting-work pool could re-enter (spin
  instrumentation decides), and it simultaneously de-coalesces epilogue
  traffic, making the injection bound *more* valuable. Reported either way."
  Sibling grid (charter exp_36): T in {512,1024,2048,4096} × routing std in
  {0, 0.032, 0.05} (0.032 = COMET's production skew); record `[MPS SPIN]` and
  the best config at every point; at std=0.05 also run C in {0,8,16}.
  Deliverable: **`sensitivity_grid.json`**.
- **Q6 — external ladders.** "MoE vs tuned production (AITER+MoRI) *and* vs
  PyTorch+RCCL eager — the literature's usual denominator — so fusion,
  scheduling, and protocol are attributed separately; GEMM-RS vs reference
  GEMM+RCCL and vs the frozen rank-1 submission, same node, same run."

Supporting attribution numbers PAPER.md commits to (§3, §4), useful for
figure captions and for knowing which claims GEMM-RS must independently
instance:

- §3.2: with CTA count held fixed, turning the pool's traffic on inflates the
  concurrent GEMM by **2.8× the capacity tax**; deleting the pool's entire
  payload copy moves that GEMM by **+5.7 µs** while **831 µs** of inflation
  remains; inflation tracks atomic/fence count, not payload bytes.
- §3.3: readiness curve `P(ready by t) = (t/S)^8`; nothing consumed a row
  before the producer phase was **>= 93.9%** complete; ~**926,000** per-row
  atomics per rank per epoch buy an event of probability ~0.
- §3.4: capping in-flight remote ops is worth **~500 µs** with a cliff; optimal
  pool size fell 64 → 16 → zero.
- §4.2: edge coordination cost fell from **~30 operations across three CTAs**
  to amortized **~1e-4** operations.
- §4.4: inherited GEMM-RS order kept **2.02 of 8** links busy; one index
  expression change cut the largest shape **27.9%**.
- Appendix A evidence map already lists the GEMM-RS rows: "link concurrency
  2.02/8 → WGM fix −27.9% | GEMM-RS exp_08"; "grouped release −4% | GEMM-RS
  E3"; "reducer count flat at uniform 32 | GEMM-RS RESULTS.md".
- §7 methodology numbers to match: phase stamps sigma ~1% vs end-to-end
  screens sigma ~6.6%; counter validation "by collapse test (the emit-local
  arm must zero the fabric counter, and does, to 0.1%)"; NaN poisoning between
  iterations; negative controls that must fail.

---

## 5. MI350X-specific things that must NOT be copied to MI300X

Everything in this list is a gfx950 / 8×MI350X fact from the sibling specs. If
it appears in a GEMM-RS artifact it must be re-measured on
`banff-sc-cs47-05.dh170.dcgpu` (gfx942) and cited to this node's own reading.

1. **Ceilings quoted for the Fig-2 dashed horizontals: 76.8 GB/s per xGMI link
   per direction, 537.6 GB/s aggregate egress, 8 TB/s HBM3E, 148 GB/s service
   rate** (`exp_22_fig7_saturation/plan.md`; the first two also in PAPER.md
   §2.2). MI300X per-link and aggregate xGMI and HBM3 peak differ — record what
   `rocm-smi` / platform docs on this node actually report.
2. **256 CTAs / 256 CUs everywhere**: the sweep grids top out at 256, the event
   ring is `u64[256][24]`, `mfma[b]` divides by 256, the role-timestamp buffer
   is `[256][2]`, the capacity tax is written `256/(256−C)`. MI300X SPX is
   **304 CU**, so the GEMM-RS analogs are 304-wide and the tax is
   `304/(304−NR)`.
3. **CDNA4 / gfx950 claims**: H4's "no oversubscription regime exists on CDNA4";
   PAPER.md §2.2's one-block-per-CU ISA §3.6.4 arithmetic is stated for the
   sibling's budgets. Re-derive occupancy from this kernel's own resource tuple.
4. **The fabric-cost measurement 52.8 GB/s for coalesced 4 B atomics vs
   54.9 GB/s for 16 B stores** (PAPER.md §4.2, from aug10 exp_21 ubench on the
   MI350X node). GEMM-RS emits 16 B peer packets on gfx942; this pair of
   numbers is not transferable.
5. **MoE-shaped work units**: 14,336 B rows = 7168 × bf16, `translate_peer<8>`,
   `store_peer_packets` of whole rows, 8-slot reduce with NT=4, 4,096 tiles /
   131,072 rows / 292,000 rows sizing, 4 GB symmetric src+dst slots, mori
   symmetric heap. GEMM-RS's egress unit is the 16 B peer packet out of the
   GEMM epilogue and its reducer body is REDV=1 — size the ubench from those.
6. **MoE phase taxonomy M0–M9, SVC_*, the phase-ID enum, `K0P6_MPS_D_LEN`,
   `K0P6_MPS_SRC_REV`, `K0_MPS_CFG`/`K0_MPS_TRACE`, cfg.flags bit 0,
   mode 2/12/14, C/g/flush_rows, `pf6gm_mega`/`mps_mega`/`production` arm
   names, LDS 155,428 B, ~7 ms epoch, run_campaign.sh**. The GEMM-RS phase
   tuple is {mainloop start/end, emit start/end, release, wait start/end,
   reduce start/end} and its arms are ours / reference GEMM+RCCL / rank-1.
7. **`s_memrealtime` = "100 MHz constant-rate" → 10 ns ticks**
   (`exp_23_fig10_timeline/design.md`). Treat the tick rate as a quantity to
   verify on gfx942 before converting stamps to µs, not as a given.
8. **No clock pinning** (`exp_22 design.md` records `rocm-smi --showclocks`
   instead). The GEMM-RS charter requires `tools/set_clocks.sh pin 1900` before
   any timing; keep pinning and keep the clock record.
9. **Sensitivity axes**: T/routing-skew/`skewed_hot`/`K0_SYNTH_ROUTE`/vLLM
   chunked prefill are MoE-only. GEMM-RS's size axis is the six graded shapes
   (64×7168×18432 … 8192×8192×29568).
10. **Sibling ratchet numbers** (6,568.0 µs, 0.8522× production, the
    5,990–6,440 µs mode-14 band, 0.888×/0.894× in PAPER.md §3.1) are MI350X
    MoE results. They are never a GEMM-RS denominator.

---

## 6. What is transferable, verbatim, and should be mirrored exactly

So the two operator instances are comparable, keep these unchanged:

- Fig-2 panel structure (3 panels, x = CTAs), the isolated-vs-concurrent series
  pair, the **C-matched reserve-only control**, the knee definition
  ("smallest C reaching 90% of each series' plateau"), 5 rotations + median,
  per-point kernel time 10–50 ms, checksum read-back, per-role
  `s_memrealtime` attribution, and the `--protocol` on/off arm that separates
  protocol cost from payload cost.
- Fig-3 3×3 grid layout, 10 µs bins, analytic bytes-per-phase (no in-kernel
  counters), same-seed across arms, rank-0 plotted with rank-max supplementary,
  the diagnostics-flag-OFF-for-timing policy, the **resource-tuple parity gate
  with the flag compiled in but disabled**, the **±10% integral check**, and the
  **±20% counter cross-check**.
- Fig-4 rung order from PAPER.md Q1 (homogeneous → +epilogue-carried payload →
  +injection bound → +arrival-sized signals → +task reorder) and the
  `waterfall.json` field set (rung, per-arm p50 µs, ratio vs baseline, delta vs
  previous rung).
- Deliverable discipline (sibling charter): `plan.md` + `result.md` + plot-ready
  `.json`/`.csv` with the schema stated in `result.md`; "No number lives only in
  prose"; a `PLOTS.md` row per figure (figure #, data file, generating
  experiment, status).

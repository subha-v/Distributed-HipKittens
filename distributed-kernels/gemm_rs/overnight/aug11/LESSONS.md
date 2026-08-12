# aug11 GEMM-RS — LESSONS (append-only; supersede, never delete)

Tonight's charter is the paper-evidence queue. The full optimization-session
record lives in `../experiments/LESSONS.md` and is not duplicated here; this
file carries verdicts from the aug11 queue plus anything the figure work
teaches.

## Session open — what the figure work inherits

- **The charter's stated baseline is two experiments stale.** It reads
  "post-E3 best, pipelined geomean ~225.6 µs; gap to rank-1 1.167×". Since
  then `experiments/exp_13_cta_split` (E4 re-sweep: shape 1 `NR=56`, shape 6
  `NR=48`, paired −3.6%) and `experiments/exp_14_tile_waves` (rows 1/2/3 →
  `32/64/128`, `64/128/64`, `128/192/32`, paired −5.9%) both landed.
  **Current: pipelined ~207 µs (means), graded gap to rank-1 1.098×, and shape
  1 is now a win at 0.850×.** Every figure must be generated at this config.

- **Numbering resolved.** The charter update (`29179f8e`) renumbered the figure
  queue to **exp_20…exp_25** under `overnight/aug11/`, leaving
  `overnight/experiments/exp_01…exp_14` to the optimization sessions. The
  collision is gone.

- **Two of the charter's Phase 2 premises are already overtaken and must not be
  re-run as written.**
  1. *"E1(a) AGPR accumulators should also clear the Gate-M2 spills on the
     256/256/32 rows."* Both halves are false now. AGPR accumulators are
     **impossible** on this kernel: `__launch_bounds__(512,1)` caps the wave at
     256 **unified** registers, so enabling AGPRs re-partitions the same file
     rather than adding registers, and LLVM disables the AGPR file entirely
     under that condition. The ISA also showed the spills were never inside the
     k-loop. And the spills are **gone anyway** — M2 now reports **7/7
     instantiations with zero AGPRs, zero scratch and zero VGPR spills** after
     the P3 half-BK split and the tile re-sweep.
  2. *"The per-call host tax is most of the remaining graded-protocol gap."*
     Measured and closed: of ~103 µs of non-device cost per call, **~92 µs is
     harness machinery every submission pays** — the evaluator's own
     `synchronize + barrier` costs 57-79 µs measured with an **empty** timed
     region — and our own share is ~11 µs, within ~6 µs of the floor for
     issuing any HIP kernel from Python at all.
  What *is* open: the mainloop's **non-MFMA** time. Shape 6 spends ~4700 cycles
  per k-iteration against ~2050 of MFMA, and rows 4/5/6 are pinned at the 64 KB
  LDS cap so their `waves × k_iters` (16 / 112 / 464) cannot be cut by
  retiling — ~640 µs of non-MFMA mainloop on shape 6, on exactly the two shapes
  carrying the whole remaining graded gap.

- **Three measurements already in hand directly support the paper's central
  claim** ("overlap is decided by scheduling decisions, not by dedicating CTAs
  to communication"), and they should be reused rather than re-run:
  1. `WGM` — a pure **task-order** change — was worth **−27.9% on shape 6** and
     took effective xGMI links from **2.02 to 7.53 of 8**, while moving *zero
     bytes differently*: the fabric carried 117.48 MB against 117.44 MB of
     useful payload (1.0003×), 99.9% in full 64 B transactions, both before and
     after. Average EA write latency fell 38.6% — the queueing signature of
     spreading a fixed request count over seven links instead of two.
  2. `RELEASE_GROUP=4` — a pure **signal-coarsening** change — was worth −4.0%
     graded on shape 6, and the win matched the mechanism's prediction to
     within 0.4 points (5.7% predicted from removing three of four releases,
     6.1% measured).
  3. **Dedicating more CTAs to communication was measured and REJECTED.** The
     producer/consumer split buys *rounds*, not CTAs: `NR=16` buys zero GEMM
     waves and doubles reduce rounds, measured **+3.6%** against a naive
     −3.5% prediction. A dual-role/work-stealing reducer was killed by the same
     arithmetic — it removes a producer wave on **none** of the six shapes.
  That is a positive, a positive, and a negative, all on one architecture.

## exp_20 (in flight) — the attribution instrument was UNOBTAINABLE, not stale

Both ledgers say the attribution table is "stale" (predates exp_05/E3 and
exp_13). That undersold the problem. `harness/exp_ablation.py` cuts the kernel
by exact-substring anchors and `generate()` asserts `count(anchor) == 1`,
**raising on the first failure**, so a single dead anchor makes the entire
six-arm table unobtainable rather than inaccurate. Three of the nine anchors
were dead against the current kernel, both causes introduced by E3's release
grouping:

1. **Anchors that quoted COMMENT text.** E3 rewrote the egress comment from
   "per-band credit wait, emit, one release, publish" to "per-band credit wait,
   then emit", which killed the anchor that closed the mainloop cut. Every
   anchor is now re-cut to quote **only executable code**.
2. **Anchors carrying the wrong INDENTATION.** E3 wrapped the tile body in an
   outer group loop, moving the mainloop from 12 spaces to 16 and the emit body
   from 16 to 20. The subtle part, and the reason this hid: a leading-whitespace
   mismatch on the **first** line of a multi-line anchor still matches, because
   `str.count` happily matches the tail of the real indent — but a mismatch on a
   **continuation** line cannot, because the preceding newline pins the column.
   So anchors 1 and 8 kept working by luck while 2, 3 and 9 went silently to
   zero.

Consequence to carry forward: **any change that re-indents or re-comments the
tile body silently disables the attribution instrument**, and the failure
presents as a `SystemExit` at whichever anchor happens to be checked first,
which reads like a broken script rather than a stale table. `exp_ablation.py`'s
anchor set is now a maintenance dependency of every mainloop/egress edit; check
it in the same commit. The repair (re-anchored to executable code at exact
indentation, with the rationale in-file) is behaviour-preserving: the six cuts
are the same six cuts.

## exp_20 VERDICT — the refreshed profile, and it re-ranks the night

Ablation at the current best config (WGM spreading, `RELEASE_GROUP=4`,
NR = 56/32/32/32/32/48, tiles 32/64/128 · 64/128/64 · 128/192/32 · 256/256/32 ×3),
40 iterations, single-cut deltas (they overlap and need not sum). Freshness gate
**PASSED**: shape 6 `full` = 1632.5 µs against the expected ~1617 µs, +0.96%.

| shape | full | GEMM | XGMI | sync | reduce | release |
|---|---|---|---|---|---|---|
| 64×7168×18432 | 67.1 | 2.5 | 2.6 | −1.0 | −0.3 | −0.5 |
| 512×4096×12288 | 67.5 | 1.1 | 2.6 | 2.6 | −0.8 | 0.9 |
| 2048×2880×2880 | 87.7 | 16.2 | 20.6 | 16.2 | 7.9 | 6.5 |
| 4096×4096×4096 | 203.5 | 38.5 | **86.2** | 31.7 | 18.9 | 18.3 |
| 8192×4096×14336 | 641.9 | **305.2** | 179.7 | 52.9 | 51.7 | **65.4** |
| 8192×8192×29568 | 1632.5 | **992.2** | 376.0 | 168.7 | 79.8 | 15.1 |

Shapes 1 and 2 are `HOST`-bound and every one of their deltas is at or below
the allocation-noise floor — several are **negative**. Do not mine them for a
mechanism conclusion; the correct reading is "no device pool is resolvable
here", not "the mainloop is free".

**Four things changed versus the table this replaces** (stale: shape 6 total
2861.7 with GEMM 1317.8 / XGMI 919.7 / release 250.3 / sync 246.9 / reduce
219.5; HANDOFF's later reading: total 1777.9 with GEMM 1142.8 / XGMI 412.4):

1. **The release pool is essentially GONE on shape 6: 250.3 → 134.9 → 15.1 µs.**
   E3's signal coarsening removed ~94% of its own pool. This is a clean paper
   data point — a pure signal-granularity change, moving zero bytes, retiring
   the term it targeted almost completely.
2. **GEMM is now dominant by a wide margin, 60.8% of shape 6** (up from a 46%
   share), even though it FELL in absolute terms, 1317.8 → 992.2 µs. Everything
   around it shrank faster. On shape 5 it is 305.2 of 641.9 (47.5%). These are
   exactly the two shapes carrying the entire remaining graded gap to rank-1.
3. **XGMI fell 919.7 → 376.0 µs on shape 6** and is now 23%. It is still the
   largest pool on shape 4, where it is **42.4%** — the profile is not uniform
   across shapes and a single ranking would hide that.
4. **`sync` (168.7 µs, 10.3%) has overtaken both reduce and release on shape 6**
   and is now the second-largest non-GEMM term.

**Release grouping is active on exactly ONE of the six shapes, and the
attribution proves the other five are paying for it.** The rule is
`rgroup = tiles_per_cta >= RELEASE_GROUP ? RELEASE_GROUP : 1` with
`RELEASE_GROUP = 4` (`gemm_rs_mi300x.cpp:335-340`), and
`tiles_per_cta = ceil(tiles / NG)` is **1/1/1/1/2/4** at the current table. So
shape 6 groups and gets release = 15.1 µs, while shape 5 does not and pays
**65.4 µs = 10.2% of its total** — far outside its ~2.2% noise floor. Shapes 3
and 4 pay 6.5 and 18.3 µs. Whether that is recoverable depends on whether
`FULL_ONLY=0` or `RELEASE_GROUP=2` is a measured negative at the CURRENT
geometry or only at exp_05's; that is being checked before any GPU time is
spent, because the kill rule forbids re-litigating a measured negative.

## ADJUDICATED — `RELEASE_GROUP_FULL_ONLY=0` is narrowly RE-OPENABLE, on shape 5 only

The kill rule forbids re-litigating a measured negative. This is not that, and
the distinction is worth stating precisely because it is exactly the situation
the rule exists to police.

**What exp_05 actually measured.** Five arms — `rg1`(N=1), `rg2`(N=2
unconditional), `rg4`(N=4 unconditional, i.e. FULL_ONLY=0), `rg4c`(N=4
FULL_ONLY=1), `rg1b`(null) — over all six shapes, 3-4 paired runs
(`experiments/exp_05_release_granularity/result.md:508-515`). So FULL_ONLY=0
**was** measured. But the regression that closed it lives on **exactly one
shape**:

| shape | tiles/CTA then | rg1 | rg2 | rg4 (FULL_ONLY=0) | rg4c | null floor |
|---|---|---|---|---|---|---|
| 2 | **2** | 88.60 | 92.01 (+3.85%) | 91.78 (**+3.59%**) | 89.13 (+0.60%) | ±0.60% |
| 4 | 1 | 199.78 | 201.14 | 200.96 | 200.60 | ±0.47% |
| 5 | **2** | 625.91 | 631.67 (+0.92%) | 645.47 (+3.13%) | 651.47 (+4.09%) | **±3.64%** |

**Why the negative no longer applies.** The shape-2 regression is
**unreachable at the current table**: exp_14's B1 retile took row 2 to
`64/128/64`, which moved it from 512 tiles to **256**, i.e. tiles_per_cta
2 → 1 (`exp_14_tile_waves/result.md:142`). At 1 tile per CTA, FULL_ONLY=0 and
FULL_ONLY=1 are the *same code path*, so the one shape that ever objected can no
longer express an objection. That is new evidence, not a re-argument — **the
fourth time a landed win has invalidated a settled constant.**

**And exp_05 explicitly declined to conclude anything about shape 5**: "Shape 5
is indeterminate by construction … `rg2` and `rg4` are behaviourally *identical*
there … yet differ by 2.2%", "No claim is made about shape 5 in either
direction", and §10: "**Shape 5** needs a better instrument before anything is
concluded about it" (`result.md:520-527`, `:625`). Its ±3.64% floor swallowed
the effect. exp_20 has now *sized* the pool that was invisible then: **65.4 µs**.

**It was never a correctness objection.** E3's mandatory protocol-review returned
"APPROVE WITH CONDITIONS" and its conditions C1-C8 hold for **any** `rgroup`
value; §2.6's tail bugs concern zero-tile CTAs, structurally excluded by
`emitted >= 1` (`gemm_rs_mi300x.cpp:358-359`) independently of FULL_ONLY. Both
`rg2` and `rg4` passed the full ladder **and M9** (17/17 at `1e-2` and `2e-3`,
M4 3/3, 600-epoch soak).

**FULL_ONLY=0 at RG=4 dominates RG=2.** On shape 5 the two are bit-for-bit the
same behaviour (2 tiles/CTA ⇒ rgroup=2 either way, measured +0.92%, inside the
floor), but RG=2 *costs shape 6*: −2.90% versus rg4c's −6.11%. So the arm to run
is FULL_ONLY=0 with RELEASE_GROUP=4 — shape 5 gets its 2-tile group while shape 6
keeps its 4.

**Verdict: ONE arm, judged on shape 5 alone.** Ceiling is about half of 65.4 µs
(two releases become one) ≈ **32 µs ≈ 5% of shape 5**, worth ~0.8% of geomean.
Queued as Phase 2 exp_26, behind the mainloop, which is 30× larger.

**Preconditions that must be honoured, or the arm is unmeasurable:**
- Shapes 3 and 4 are **irrecoverable** — at 1 tile/CTA their 6.5 and 18.3 µs of
  release cost cannot be grouped by any setting. Do not promise them.
- **Shape 5 was never in M9's `CASES`** (`harness/m9_stale_slot.py:69-81`), and
  M9 is the gate for publication-ORDER changes and is **not part of
  `gate_ladder.sh`**. Shape 5 must be added.
- **M9's golden is stale**: `gemm_rs_mi300x_e3base` is the frozen pre-E3 build
  and is no longer bit-identical after the retile. It needs re-golding at RG=1
  from today's tree before it can adjudicate anything.
- Shape 5 needs **its own null arm in both construction orders** — aug11's
  re-measured floors are 2-2.6× worse than published.

## exp_20 — attribution refreshed at the current best (Q3). LANDED.

Pipelined µs per world-8 op. `floor` is that shape's allocation-noise floor;
`*` marks a pool inside its floor, i.e. **not measured**.

| # | shape | full | GEMM | egress | sync | reduce | release | floor | ranking |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 64×7168×18432 | 67.1 | 2.5 | 2.6 | −1.0\* | −0.3\* | −0.5\* | 2.34 | host-bound |
| 2 | 512×4096×12288 | 67.5 | 1.1 | 2.6 | 2.6 | −0.8 | 0.9 | 0.63 | host-bound |
| 3 | 2048×2880×2880 | 87.7 | 16.1 | 20.6 | 16.2 | 7.9 | 6.5 | 1.41 | egress > sync ≈ GEMM |
| 4 | 4096×4096×4096 | 203.5 | 38.5 | **86.2** | 31.8 | 18.9 | 18.3 | 4.90 | **egress** > GEMM |
| 5 | 8192×4096×14336 | 641.9 | **305.2** | 179.7 | 52.9 | 51.7 | **65.4** | 13.93 | **GEMM** > egress > release |
| 6 | 8192×8192×29568 | 1632.5 | **992.2** | 376.0 | 168.6 | 79.8 | 15.0\* | 69.87 | **GEMM** > egress > sync |

**The ranking inverts with size**: egress owns the mid shapes (42.4% of shape
4), the mainloop owns the large ones (47.6% and **60.8%**). GEMM is still #1 on
shapes 5 and 6 — the only two we lose to rank-1 — and became *relatively more*
dominant (60.8% vs 46% in the oldest table).

Every pool shrank 8-13% since the last table while the **shares barely moved**,
except `release`, which fell **−88.9%** and structurally disappeared.

- **THE CHEAPEST UNCLAIMED WIN: `release` is still 65.4 µs (10.2%) on shape 5.**
  `RELEASE_GROUP_FULL_ONLY=1` groups only when a producer CTA owns ≥4 tiles, and
  `tiles_per_cta` is **1/1/1/1/2/4** — so shape 6 groups and shape 5 does not,
  still paying a release per tile on a shape that carries the gap. Counters
  corroborate independently: release-driven L2 writebacks are **13.1% on shape 6
  versus 56.4% on shape 5**. A **per-shape** group size (2 for shape 5, 4 for
  shape 6) keeps every group *full*, so it does not reintroduce the partial-group
  reducer starvation that cost 3.6-3.9% on shape 2.
- **Shapes 1 and 2 are HOST-bound and have no resolving power** in this harness
  (shape 1: 62.3 µs of issue inside a 66.4 µs wall). Do not rank kernel work off
  them, and do not read their pool deltas as signal — most sit inside the floor.
- **Egress width is now CLOSED across the whole scored set**, not just shape 6:
  amplification is **1.0005-1.0043×** on shapes 2-6, with shape 6 at 117.52 MB
  fabric against 117.44 MB useful (1.0007×) and 99.9% at 64 B — materially
  identical to exp_08 on a kernel 8.2% faster. EA write latency on shape 6 fell
  2887 → 2090 cycles.
- **Arithmetic correction worth keeping:** exp_08's "102 GB/s" divided by an
  egress *pool*, not the wall. Over the wall shape 6 is 72.0 GB/s; over its
  376.0 µs pool it is 312.6 GB/s, ~70% of nominal xGMI. On shape 3 the same
  arithmetic yields **501.9 GB/s — above the ceiling**, which proves that delta
  is partly *overlapped* rather than serial egress time. **A single-cut delta is
  not a duration.**

- **TRAP: three of nine `exp_ablation.py` anchors were dead, and `generate()`
  raises on the first — so the table was not obtainable at all.** Two quoted
  comment text that E3 rewrote; one had stale indentation on a continuation
  line. Subtlety now documented in the file: **a wrong indent on an anchor's
  *first* line still matches**, which is why some anchors kept working by luck
  while continuation-line ones silently went to zero.
- **TRAP: stale scratch disguised as fresh.** `ablate/gemm_rs_ablate.cpp`
  carried today's mtime and all five gate macros but was **30,610 bytes against
  the 50,556 this run generates** — generations old, with a current mtime only
  because `push.ps1` rewrites them. `reattribute.sh`'s force-remove and its
  freshness assertion are load-bearing, not belt-and-braces.

## Measurement discipline carried into the figure work

- **The harness bias is per-allocation AND partly allocation-ORDER, not
  positional.** Two arms with identical config/operands/code differed by 4.28%
  on shape 6 while the positional residual stayed under ±0.47%. The twin
  contrast was negative in 10 of 10 forward draws on shape 1 and **flipped sign
  when arm construction order was reversed**. Null-arm floors measured this
  session: **3.48 / 0.93 / 1.61 %** on shapes 1-3 — 2-2.6× worse than the
  earlier published figures, which are therefore lower bounds.
  **Consequence for figures: any plotted delta needs a null arm on the same
  axes, built in both orders, or the error bar is fiction.**
- **Means are unusable on this node**, even same-run interleaved — per-pool
  jitter attaches to an arbitrary arm. Plot best and median; if a figure needs
  means, show the null-arm spread beside them.
- **~92 µs of every graded call is harness machinery both arms pay** (the
  evaluator's own `synchronize + barrier` is 57-79 µs with an empty timed
  region, plus 15-20 µs of input clone). Any ladder figure comparing kernels
  must state whether that constant is netted out; it makes ratios look better
  than the kernels are.
- **Read `bound` and `×SOL` before predicting a mainloop win.** Shape 1 was
  predicted at −25% from halving k-iterations and delivered −1.7%, because M7
  already labelled it `bound = HOST` at 10.16× SOL — its whole mainloop is
  ~2-4 µs of ~63 µs.

## Open validation gap

- **`gemm_rs_mi300x_static_checks.py` has been dead since exp_02.** Its first
  requirement asserts the reducer column still matches RadeonFlow's inherited
  values — which exp_02 deliberately changed — and it *raises* there, so none
  of its ~20 downstream gates has run while the shape table was rewritten three
  times. Re-run collecting instead of raising: zero tile/dispatch failures, so
  nothing was broken, but the guard has been silently absent.

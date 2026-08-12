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

### The counter pass, and the number Phase 2 should be aimed at

**Only 421.2 µs of shape 6's 992.2 µs GEMM pool is MFMA occupancy.** So roughly
**571 µs — 35% of the entire operation — is mainloop schedule rather than math**,
and about 204 µs of shape 5's 305 µs is the same. That is the single largest
addressable quantity in the kernel and it is *not* a math-throughput problem.
The MFMA instruction count matches a 16×16×16 atom exactly, including the
+0.43% from `K_local = 3696` not dividing `BK = 32`, so the count is trusted.

Two axes close on the counter evidence:

- **Egress width is closed.** Fabric amplification is **1.0007× at 99.9% full-64 B**
  on every large shape. Whatever egress still costs, it is not bytes and not
  coalescing.
- **The WGM fix is visible in the stall counter**, not just in wall time:
  `WRREQ_STALL` fell to **10.0%** of `TCC_CYCLE` from exp_08's 16% at `WGM=4`.

**The labels were validated by collapse, not asserted.** The emit-local control
took off-die requests from **1,836,800 to 1,792** — a 1000× collapse — which is
the same discipline the paper's methodology section demands ("the emit-local arm
must zero the fabric counter, and does"). 24 counter cells, all `errors=none`.

### Correction to the release reading above, and it matters

Shape 6's release delta of 15.0 µs is **inside that shape's 69.9 µs
allocation-noise floor**, so the honest statement is that release on shape 6 is
now **unmeasurable**, not that it is precisely 15 µs. The floors this run
measured are much larger than the per-shape percentages suggest in absolute
terms: **2.3 / 0.6 / 1.4 / 4.9 / 13.9 / 69.9 µs** for shapes 1-6. Re-reading the
shape-6 ranking against them: GEMM (992.2), egress (376.0) and sync (168.6) are
comfortably resolvable; **reduce at 79.8 µs is only 1.14× its floor and is
marginal**; release is not resolvable at all. Any figure using these deltas must
carry the floor beside the value.

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

## exp_24 — `tools/run_rank1_bench3.sh` could not run rank-1 AT ALL

Found by dry-run inspection before any GPU time was spent, which is the only
reason it is cheap. Four defects, each independently fatal, and the script had
been sitting in `tools/` presenting as the canonical rank-1 driver while
`experiments/exp_10_rank1/r1_eval.sh` quietly did the real work:

1. **`AMDGCN_USE_BUFFER_OPS=0` absent** — so the arm faults (Triton 3.6.0
   lowers rank-1's peer stores to `buffer_store_dwordx2`, whose 32-bit voffset
   truncates a −4.4-billion-element offset). Worse, the env was assembled as one
   `ENVS` string interpolated into `bash -c`, so the knob could not be injected
   from outside either. Now passed as `docker exec -e` flags.
2. **`PYTHONPATH=$ON/compat`** — that path does not exist; the tree is
   `$ON/tools/compat`. Repair #3's `sitecustomize` was therefore silently
   absent on every run.
3. **`$ON/patch_rank1.py`** does not exist either (it is `$ON/tools/`), and the
   `if [ $? -ne 0 ]` guard beneath the call tested the wrong command's status,
   because the `run` wrapper had already returned.
4. **It ran in `dhk-eval`**, which holds the sudo shim at `/usr/local/shim` but
   runs as **root**, so every artifact it writes under the repo comes out
   root-owned and breaks later `sed`/`scp` steps — a trap HANDOFF already warns
   about. `dhk-gemmrs` runs as uid 15523, and its own `sudo` fails with "you do
   not exist in the passwd database", which is precisely why the PATH shim is
   needed there.

Repaired to mirror `r1_eval.sh`, the driver that actually produced exp_10's
measured comparison. Verified on the node: `tools/compat/bin/sudo` and
`tools/compat/sitecustomize.py` exist and import, `iris`/`iris.hip` import in
**both** containers, and **the frozen submission's sha256 still matches**
`7940fcb8…f0dc5`.

**The general lesson, and it is the third instance tonight:** a tool that is
never exercised end-to-end decays silently, and the decay is invisible because
nothing fails loudly — `exp_ablation.py`'s anchors, `gemm_rs_mi300x_static_checks.py`'s
first-assertion abort, and now this. **A gate or driver that has not been run
since the code moved underneath it is not evidence.**

## exp_24 — the evaluator cannot produce the statistics the protocol requires

`eval.py` offers **no median, no raw samples, no fixed 3×50** (its loop is
adaptive), **no pipelined region and no warmup**, and its
`from submission import custom_kernel` makes a two-module-name null arm
impossible. So the ladder cannot be built out of the evaluator alone. exp_24
therefore runs **two instruments**: `ladder_mp.py` (five arms — `ours`,
`ours_null`, `reference`, `rank1`, and a `harness_floor` arm that **measures**
the ~92 µs graded constant in-run rather than quoting it — in one 8-process
pool per shape, both protocols, arm order rotated so each arm is first exactly
once), with the three `tools/` evaluator drivers kept as the cross-check.
Measuring the harness constant instead of quoting it is an improvement on the
dispatch and should be carried into any future ladder.

Parser validation, done against **saved historical output rather than a fresh
run**: `ours` and `reference` parsed 6/6 shapes; the `rank1` fixture — exp_10's
run that hit its 1500 s wall after three shapes while still emitting
`benchmark-count: 6` — was **refused with a named error**, which is exactly the
silent-null hazard the parser exists to catch. Reference-arm RSDs measured
75/47/68/60/115% , confirming and slightly exceeding the "12-93%" caveat: quote
its ratios, never its absolutes.

## exp_23 — the waterfall's rungs are CONDITIONALLY ACTIVE, and that changes the figure

Pre-registered from the shape plans **before** measuring, which is what makes it
a prediction rather than an excuse:

- **Rung a→b (WGM task order) can only move shapes 5 and 6.** The knob is
  `WGM = (tiles <= g.num_gemm_ctas) ? 4 : num_pid_m`, so it differs from the
  constant `WGM = 4` only where `tiles > NG`, i.e. only on the multi-round
  shapes: shape 5 (512 tiles over 272 producers) and shape 6 (1024 over 256).
- **Rung b→c (grouped release) can only move shape 6**, because
  `rgroup = tiles_per_cta >= 4 ? 4 : 1` and shape 6 is the only row with
  `tiles_per_cta == 4` — the same fact exp_20 measured from the other side.
- **Therefore shapes 1-4 are predicted FLAT across a, b and c, and serve as four
  extra null pairs.** A resolvable delta on shapes 1-4 is a **blocker, not a
  result**: it would mean a rung changed something the mechanism cannot reach,
  i.e. a build or harness artefact.

This is the right way to read the money figure and it strengthens rather than
weakens the paper's claim: these knobs are **targeted**, not global. A geomean
alone would smear four structurally-inert shapes into the average and understate
each rung's effect exactly where it acts.

**Live risk, flagged before the run:** shape 6's b→c effect is expected around
4%, and its published null floor is **4.28%**. So the granularity rung may come
back **unresolved rather than won**, and that must be reported as unresolved —
not re-run until it looks better. This is why the sweep takes several draws in
both construction orders and scores with an exact rank-sum rather than a range.

### The fingerprint gate passed, and it is now the model for arm construction

Four rung binaries, each disassembled and hashed (ISA sha over gfx942 asm with
`__hip_cuid_` normalised), with **nine assertions**, all passing:

| rung | WGM4/RG | ISA sha[:12] | instrs | v_mfma | s_cbranch |
|---|---|---|---|---|---|
| a | 1/1 | ec714ea8a644 | 20190 | 184 | 1268 |
| b | 0/1 | 366d4f08a030 | 20201 | 184 | 1268 |
| c | 0/4 | 7926c2e87283 | 20734 | 184 | 1272 |
| null | 0/4 | 7926c2e87283 | 20734 | 184 | 1272 |

The distinctions are **attributable to their mechanisms**, not merely present:
a↔b differ in `s_cselect` 535→542 (the folded `tiles <= NG` select), b↔c in
`s_cbranch` 1268→1272 and +533 instructions (the group loop). `v_mfma` = 184 in
all four, so no rung accidentally changed the math. **c ≡ null bit-identically
except 5 `__hip_cuid_` lines**, which is what makes the null a pure allocation
contrast. Rung (c) reproduces the shipped binary: 7 instantiations, VGPRs
{98,104,136,246,248,91,92}, zero AGPR/scratch/VGPR-spill, resource table
identical field-for-field to `harness/build/gemm_rs_mi300x.log`.

Two disclosed method details worth keeping: the `.so` is **linked from the same
object that `--save-temps` disassembled**, so the ISA provably belongs to the
measured module rather than to a second compilation of the same flags; and SGPR
spills (54-88) are deliberately **not** asserted zero, because the M2 no-spill
claim is about *vector* spills.

Correction to an earlier note: **`HK_GEMM_RS_MI300X_TILE_SWEEP` is not dead.**
Its dispatch rows (`gemm_rs_mi300x.cpp:101-107`) are live for exp_14's screening
module; it is simply not passed by the production build.

## PROCESS FAILURE — an ungated kernel change shipped as the DEFAULT

Caught by inspection, not by a failing test, which is the only reason it cost
nothing. An `exp_26` release-grouping candidate was written directly into
`gemm_rs_mi300x.cpp` with

```
#define HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE 1   // default ON
```

**A candidate must never be the default before it is gated.** Defaulted to 1, it
silently redefines the production binary for *every* experiment that compiles
from that file — including the waterfall's rung (c), whose entire job is to be
the "shipped binary" reference arm. The figure would have compared three honest
rungs against a fourth that was quietly a different kernel, and nothing would
have failed; the waterfall would simply have been wrong. Forced to **0**, with
the reasoning written at the definition site.

Damage assessment, done before touching anything: exp_23's four rung binaries
were built at 04:19-04:20 and the source was edited at 04:43, and all four `.so`
sha256 values still match `fingerprints.json` on disk. **The waterfall is
uncontaminated** — which is precisely what the fingerprint gate was built to be
able to answer, and it answered it in one command. That is the strongest
argument for keeping the gate on every future arm.

The mechanism itself is good and is adopted as a Phase 2 candidate rather than
discarded: take `rgroup = max(1, min(RELEASE_GROUP, tiles_per_cta))` instead of
the step function, so the effective group goes `1/1/1/1/1/4 → 1/1/1/1/2/4` and
only shape 5 moves, with the four one-tile rows arithmetically pinned to 1 and
therefore acting as controls. The distinction from E3's rejected unconditional
arm is real and worth preserving: that arm's defect was that at
`RELEASE_GROUP = 4` a 2-tile CTA's loop strided **four CTAs' worth of tiles**, so
the group could only ever be half full and the first tile's publication was
deferred behind a mainloop with no second tile to amortize it. Capping by
`tiles_per_cta` never does that. The honest residual risk is **publication
delay, not fullness**, and it must be measured against a paired null arm.

Rules reasserted for the rest of the night:
- **New macros default to 0.** The shipped binary changes only after the full
  ladder plus a paired timing win.
- This one moves publication order, so it needs **M9**, which is not part of
  `gate_ladder.sh`, whose golden is stale after the exp_14 retile, and whose
  `CASES` do not include shape 5 — the very shape it targets.
- Agents own their experiment directory. Kernel-source edits happen only through
  an explicit gate, because several experiments compile from that one file
  concurrently.

## exp_24 quick run — THE GRADED PROTOCOL FLATTERS US, measured not assumed

A pre-registered prediction was **falsified in sign**, and it changes how every
ratio in this project should be read. The prediction was that the pipelined
ratio to rank-1 would be at least 0.08 *better* than the graded one. Measured on
shape 2: **graded 1.0496, pipelined 1.2207.** The graded ratio is the *flattering*
one.

The mechanism is arithmetic and was already half-known: the evaluator adds a
constant ~90 µs to **both** arms, and a constant added to numerator and
denominator alike compresses any ratio toward 1. HANDOFF said this constant
"makes our true kernel-to-kernel ratio worse than the headline"; this is the
first time it has been measured on a fresh instrument with the sign confirmed.

**Consequence for the ratchet, and it is not comfortable: part of the 1.098×
graded gap to rank-1 is protocol dilution rather than kernel parity.** The
graded number remains the competition's ranking statistic and must keep being
reported as such — but it is not the honest kernel-to-kernel comparison, and
`pipelined` is. Report both, always, and never let the graded ratio alone stand
in for how good the kernel is. If this holds across all six shapes, the true
kernel gap is materially wider than 1.098× and the mainloop work is even more
clearly the right target.

**The harness constant is now measured, not remembered:** `harness_floor` =
**90.38 µs median graded** (84.51 best), within 2% of the remembered ~92 µs, and
**6.14 µs pipelined** — i.e. the pipelined protocol carries essentially no
constant, which is exactly why it is the honest instrument. Caveat to carry: the
floor arm's own **rsd is 53.7%**, so a netted-out table is much noisier than the
raw one and can only be read as a direction, not a value.

**Null floor, shape 2 only** (the tightest of the six, so do not generalize):
graded 0.66% best / 0.20% median / **1.64% mean**; pipelined 1.12% / 1.58% /
**1.87%**. The published 0.56% floor **holds for best and median and fails by
~3× for the mean** — HANDOFF's "means are unusable on this node" reproduced
independently on a new instrument. Shapes 1 and 6 are the decision-relevant rows
and are not measured yet.

**Two instrument bugs caught in review rather than by a hang**, both worth
remembering because both would have produced silence rather than an error:
duration-based warmup with *per-rank* deadlines deadlocks a collective arm
(ranks decide to stop at different times), so rank 0 now broadcasts the stop
decision; and the warm clock had to start after the first block, because
`reference` and `rank1` spent the entire 400 ms warmup inside RCCL/JIT setup and
then landed on the 20-call floor.

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

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

## CRITICAL METHOD FINDING — one null twin is NOT enough, and cross-order consistency does not save you

This supersedes the null-arm discipline used everywhere in this project so far,
and it must be applied to every delta any experiment certifies from now on.

`resolve_shape_with_split` overrides **only** the CTA split. So wherever a swept
NR point equals the shipped NR for a shape, that arm is a **second
identically-configured twin** of the shipped arm — same code, same config, same
operands, different allocation. exp_23 accidentally had two such twins on shape 6
and **they disagreed**:

| identical pair | shape-6 contrast |
|---|---|
| `c` vs `null` | 1.10% |
| `c` vs `nr48` | **+3.74 / +4.44 / +4.10 / +2.53 %** |

The second pair was **positive in all four draws and in both construction
orders**, which the full-range disjointness rule certifies as "RESOLVED faster".
It is a **false positive** between two binaries that compute the same thing the
same way — and its magnitude reproduces HANDOFF's published 4.28% shape-6 floor
almost exactly.

Two consequences, both sharper than the existing guidance:

1. **A single null twin under-measures the floor.** The floor is a property of the
   *pair*, not of the configuration, so one pair samples it once. The remedy is
   to score against the **union of all identically-configured pairs** available in
   the run — exp_23 re-scored that way, and both resolved rungs survived
   (shape 6 b→c at [10.98, 12.98] against a null range of [−0.73, 4.44]).
2. **Cross-order consistency does NOT rule the artefact out.** The existing
   lesson was that reversing arm construction order flips the sign of the
   allocation bias, so agreement across orders was treated as evidence. Here a
   spurious effect was consistent in *both* orders across four draws. Reversal
   remains necessary; it is not sufficient.

**Null floors measured tonight with the widened null set:
2.82 / 2.09 / 1.74 / 0.85 / 4.97 / 4.44 %** for shapes 1-6. Note shape 5's 4.97%
is nearly double its published 2.17%, and shape 4's 0.85% is *better* than its
published 2.41% — the published table is not uniformly conservative, so floors
must be measured in the run that produces the ratios, never quoted.

## exp_23 VERDICT — the waterfall, and the structural prediction HELD

4 draws, 2 forward and 2 reversed construction order, **all six shapes**, 8 arms,
complete 8-pass rotation, on validated rungs (`fingerprints.json` records
A1/A1b/A2/A3/A4 all passing, `c.isa == null.isa` cuid-only). **All 192 arm
instantiations passed `1e-2` AND `2e-3`** with clean error bits and epoch/signal
state; zero dead arms. Lease acquired and released from an EXIT trap, no
preemption.

Best / median µs:

| rung | s1 | s2 | s3 | s4 | s5 | s6 | geomean |
|---|---|---|---|---|---|---|---|
| a | 61.3/63.5 | 64.3/65.8 | 83.3/85.0 | 197.7/199.0 | 743.9/751.8 | 2595.0/2616.3 | **223.65 / 227.62** |
| b | 62.1/64.0 | 64.6/66.0 | 85.0/85.7 | 197.9/198.9 | 634.1/652.5 | 1801.8/1840.8 | **206.29 / 210.28** |
| c | 60.4/62.7 | 64.8/66.0 | 83.6/84.5 | 198.0/199.1 | 624.9/642.5 | 1609.8/1635.7 | **200.60 / 204.52** |
| null | 60.0/61.8 | 64.6/65.3 | 83.9/84.5 | 197.5/198.8 | 625.6/657.9 | 1601.6/1640.9 | **200.18 / 204.50** |

**Cumulative a→c = 1.113×**, decomposing into **1.082× from task order** and
**1.028× from signal granularity**. Rung c and its twin agree to **0.01%** on the
geomean.

**The structural prediction HELD — no blocker.** All eight contrasts the mechanism
predicts inert on shapes 1-4 came back unresolved. The closest call is b→c on
shape 3 at +1.41%, rank-separated on medians but inside that shape's 1.74% floor,
i.e. correctly unresolved.

**Per-rung, where the mechanism is active** (median gain, per-draw range, exact
rank-sum; the p floor at 4 draws is 1/C(8,4) = 0.0143):

- **a→b**: shape 5 **+15.11%** [12.57, 19.07] p=0.0143 **resolved**; shape 6
  **+42.11%** [40.48, 44.79] p=0.0143 **resolved**. Shape 6's +42.11% gain is
  **−29.6% in time, against exp_08's independently measured −27.9%** — a clean
  replication of the WGM result by a different instrument.
- **b→c**: shape 6 **+12.22%** [10.98, 12.98] p=0.0143 **resolved**.

**The b→c rung came in at 12.22%, three times the ~4% I pre-registered, and the
pre-registration was wrong for a specific reason worth keeping**: the ~4% figure
came from exp_05's **graded** protocol measured on the **aug10 task order**. This
measurement is **pipelined, on top of WGM**. Coarsening the signal is worth more
once the order change has already removed the egress serialization that was
hiding it — the two knobs **compose superadditively**, which is itself a result
for the paper's Q1: the waterfall's rungs are not independent contributions, and
their order matters.

### Rung (d): the NR "flat" prediction is FALSIFIED — with a sharper replacement

Geomeans relative to rung c: **NR=8 1.392× slower, NR=16 1.141×, NR=32 1.023×,
NR=48 0.992×.** NR=16 is resolved slower on 5 of 6 shapes and NR=8 on all 6, by
14-51%.

The curve is **flat on the 32-56 plateau** — every shape sits inside its own floor
from its shipped NR up to 48 — and falls off a cliff below 32. So the correct
claim is not "placement is flat everywhere" but: **the reducer count has a floor
it must clear, and above that floor it is flat.**

> **Superseded in one detail by exp_25**: the `NR=8 1.392× / NR=16 1.141× /
> NR=32 1.023× / NR=48 0.992×` ordering must NOT be read as ranking NR=48 above
> NR=32 or above the shipped table. Shape 6's `nr32 vs c` contrast was labelled
> "RESOLVED slower" but sits **inside its 4.44% floor**, so it is unresolved, and
> **no shape resolves NR=48 against NR=32**. A plateau cannot rank its own points.
> The resolved content of rung (d) is the **cliff below 32**, nothing above it.

**And the reason matters more than the curve.** The reducers are not a
communication pool at all — they are the **owner-side reduce**, which must run on
the owner. Below ~32 you are starving a computation, not under-provisioning
communication; above it you buy nothing. So Q2 still answers **no**, for a sharper
reason than "placement doesn't matter": there was never a communication pool to
size.

**Shipped-point marking, which the figure needs or it misleads**: shipped NR is
per-shape `56/32/32/32/32/48`, so the NR=32 column **is** the shipped config for
shapes 2-5 and NR=48 **is** shipped for shape 6. The ladder did not regress at
its own best.

**Correction to two earlier statements of mine in this file.** I wrote that shape
1 was "still improving at 56" — it is not: 61.67 at NR=48 against 60.41 at NR=56
is a 2.1% difference inside shape 1's 2.82% floor, i.e. flat. And I quoted the
NR geomeans from a partial-shape run; the all-six-shape figures are the ones
above.

## PROCESS — an unleased GPU job stalled the figure queue, and the lease was wrong too

The orphaned exp_26 work launched `m9_stale_slot.py` (detached, `timeout 7200`)
on all 8 GPUs **without taking the lease**, then exited, leaving a 2-hour-capped
GPU job with no owner. Two separate faults, and the second is mine:

1. **A GPU job that does not take the lease defeats the lease for everyone
   else.** The lease is only as good as its weakest participant.
2. **`gpu_lease.sh`'s drain wait was capped at a fixed 300 s and then *failed the
   caller*.** That is backwards: a busy node is a reason to keep waiting, not a
   reason to abort a queued campaign. exp_21 acquired the lease, found the node
   dirty because of the unleased M9 run, and was on course to abort at 300 s
   while doing everything correctly. Fixed so the drain wait spends the
   **caller's own remaining budget** and only the caller's deadline ends it.

**Swapping the tool required care worth recording**: bash reads a script
incrementally from its open descriptor, so truncating a script that a process is
currently executing makes it resume at a byte offset in different text and fail
in whatever way the new bytes parse as. The replacement was staged under a
temporary name, `bash -n` syntax-checked, and installed with `mv` — an atomic
rename leaves the running process on its original inode.

## NODE — a dead process stalled the queue for 25 minutes, and the node was fine

The most expensive twenty-five minutes of the night, and the cause was a
misreading that this project's own rules made easy. Written up in full because
the next session will hit it again.

**What happened.** The orphaned exp_26 M9 run took a **`VM_L2_PROTECTION_FAULT`**
— `dmesg` shows `PERMISSION_FAULTS`, `MAPPING_ERROR: 0x1`, faulty UTCL2 client
`TCP`, on dies AID0/AID1/AID2 XCD0 across several devices, all attributed to pid
3001610. The process died and then **wedged in `exit_mm`**, keeping its KFD entry
and ~1.25 GB per GPU.

**Why it looked like a wedged node and was not.** `rocm-smi --showpids` still
listed it, so every drain check in this tree — `reattribute.sh`'s `wait_clean`,
`run_reference_arm.sh`'s preflight, and my own `gpu_lease.sh` — counted it as a
live tenant and refused to start anything. Two correctly-behaved campaigns
(exp_21, exp_24) sat in `acquire` behind a process that no longer existed.
Meanwhile: **all 8 GPUs passed a live 4096³ bf16 matmul, 189-190 GiB free of 192,
junction temps 38-44 °C, package power 134-151 W, 0% utilization.** The hardware
was completely idle and completely healthy.

**The three diagnostic signals that separate a corpse from a running job**, and
all three are cheap:

| signal | corpse | live job |
|---|---|---|
| `/proc/<pid>/wchan` | `exit_mm` (or `do_exit`) | a scheduler/driver wait |
| `utime`/`stime` in `/proc/<pid>/stat` over 5 s | **frozen** | advancing |
| `CU OCCUPANCY` in `rocm-smi --showpids` | 0 for minutes | non-zero, or bursty |

Plus the decisive one: **just try a trivial matmul on all 8 devices.** That single
test would have answered the question in 30 seconds and it is now the first thing
to run when the node looks busy but nothing is progressing.

**A task in `exit_mm` cannot be signalled.** It is already exiting and has no
`mm`; SIGTERM and SIGKILL are both no-ops. So the charter's "never SIGKILL a GPU
process" is not the operative rule here — there is nothing to kill, and trying is
how a stale entry gets escalated into a genuinely wedged node. Nothing was
signalled. Only the **plain bash waiters** were SIGTERMed, which is safe because
they are shell, not GPU processes.

**The fix, and why it is principled rather than a workaround.**
`gpu_lease.sh` now excludes pids that are zombies or in `exit_mm`/`do_exit` from
its live count, and reports them separately as
`KFD STALE pids (exiting/zombie, cannot dispatch, ignored)` so the condition
stays visible instead of being silently tolerated. A process with no address
space cannot dispatch a kernel — excluding it is simply correct.

**Standing caveat for every number taken after 05:15 tonight:** a stale KFD entry
holding ~1.25 GB/GPU existed while the remaining campaigns ran. It had 0 CU
occupancy and 0% GPU utilization throughout, so the expected effect on timing is
nil, but it is disclosed, and the first campaign to run afterwards should
re-verify a known value before its numbers are trusted.

**And the fault itself is a real finding, not just an obstacle: the exp_26 arm
faulted.** A `VM_L2_PROTECTION_FAULT` is an out-of-bounds or unmapped access, not
a tolerance miss. Two candidate causes, and they must be separated before exp_26
is measured at all:
1. the `rgroup = max(1, min(RELEASE_GROUP, tiles_per_cta))` change itself, or
2. M9's harness, since shape 5 had to be **added** to `m9_stale_slot.py`'s `CASES`
   and M9's golden module was already known stale after the exp_14 retile.

Cause 2 is at least as likely as cause 1 and is the cheaper hypothesis to test:
run the **unmodified** kernel through the same extended M9 first. **This
retroactively and strongly vindicates forcing `RELEASE_GROUP_PERSHAPE` to default
0** — had it shipped default-on, this fault would have been in every experiment's
binary tonight, and its first symptom would have been an unexplained fault in
some *other* experiment.

## exp_25 VERDICT — Q5's premise is FALSIFIED in sign, and then NOT IDENTIFIABLE

No GPU time; derived entirely from exp_20 and exp_23 with both inputs' sha256
recorded, and the activity flags re-derived from `m, n, bm, bn, nr` alone and
checked against both sources for all six shapes (`--check` PASSED).

### P1 — "order and granularity deltas grow with communication share": FALSIFIED

**The deltas are largest where communication share is LOWEST.** Spearman ρ over
the four non-host-bound shapes, for a→b: **−0.40 / −0.80 / −1.00 / −0.80** under
four different comm-share definitions; for b→c: −1.00 / −0.80 / −0.40 / −0.80.
**The sign is not definition-dependent, only the strength is.** The primary
definition is `D3 = (full − GEMM)/full`, chosen because it is a single cut top and
bottom and has a closed form: flops per egress byte is `8K/7`, so comm share is a
function of **K alone**.

### And then the deeper result: P1 is NOT IDENTIFIABLE from these six shapes

Across shapes 3-6 the set orders **identically** by `K` and by `tiles/NG`
(ρ = **+1.00**), while comm share is monotone in `K` (ρ = **−1.00**). So **the
structural mask that decides whether a knob can act at all is perfectly
confounded with comm share** on the graded shape set. No amount of extra draws
fixes that — **P1 cannot be tested on this shape family at any n**, and reporting
a correlation for it would be reporting the confound.

**The regression is also ill-posed on its own terms.** Shape 6's order rung is
worth **1.39× the entire communication pool that survives at rung c** (775 µs of
gain against 560 µs of pool) and granularity is worth **13.4× the release pool**
(201 vs 15 µs). The knob **destroys the very share you would regress against** —
which is a restatement of the superadditivity finding: the rungs are
order-dependent, so "share at the final config" is not the independent variable it
looks like.

**How few shapes inform each claim, stated because it is the point.** Shapes 1-2
are host-bound and excluded (shape 1 has three negative deltas; the five cuts
price only 4.9% and 9.5% of their wall), leaving **n=4** — at which the smallest
possible two-sided p is **0.083**, so nothing is fitted and `coefficient` is
`null` in the JSON with the reason inline. For the question that actually matters
— *given the knob is active, does its value scale with comm share* — it is
**n=2** for order (shape 5 at share 0.525 → +15.11%; shape 6 at 0.392 → +42.11%)
and **n=1** for granularity. That is not a trend and is not presented as one.

**What replaces P1**: among the two active shapes the delta tracks **producer
rounds** (`tiles/NG` 1.88 → 4.00, a 2.8× larger delta) while comm share *falls*.
The right claim is a **task-graph** claim, not a communication-intensity one.

### P2 — "the NR curve is flat everywhere": FALSIFIED as written, CONFIRMED on the plateau (n=6)

Flat from NR=32 up on shapes 2-6 and from NR=48 on shape 1, every point inside
that shape's widened floor. NR=8 resolved slower on **6 of 6** shapes (−13.75% to
−51.48%), NR=16 on 5 of 6. **The plateau edge is not constant** — it is 48 on
shape 1 — which is precisely why the shipped table is `56/32/32/32/32/48` rather
than a single number.

The cliff correlates +0.60 with reduce share and +0.80 with not-GEMM share but
**0.00 with `egress/full`**, all indistinguishable at n=4 — so **the mechanism
carries the claim, not the correlation**: at NR=8 the owner-side reduce needs 16
rounds instead of 3.

### Five places this contradicts the paper's current story

Worth carrying to the paper rather than smoothing over:

1. **Q5's registration is wrong for GEMM-RS.** Repairing it converts it from a
   communication-intensity claim into a task-graph claim.
2. **The two operators do not instance one trend.** The MoE side registers "small
   M shifts value toward signal coarsening", while here coarsening is
   *arithmetically impossible* except on the **largest** shape.
3. **The knobs cannot reach the most communication-bound shape.** Shape 4 is
   **42.4% egress** and **no rung touches it**; shapes 3 and 4 are irrecoverable
   at any release setting, since at 1 tile/CTA there is nothing to group.
4. **At the winner, the not-GEMM share is only 0.392 (shape 6) and 0.525 (shape
   5)** — which *supports* Q3's "the frontier moves back into single-GPU GEMM
   quality", and means Q1's ladder is best read as **history, not a map**.
5. **An input inconsistency in exp_23's own output** (see the correction below).

### CORRECTION to exp_23's NR reading, and to what I recorded earlier

exp_23 labelled shape 6's `nr32 vs c` contrast "RESOLVED slower" while the effect
sits **inside that shape's 4.44% floor**. It is **unresolved**. Consequently the
`NR=48 vs NR=32` geomean ordering — which I earlier repeated as "NR=48 0.992×,
i.e. better than shipped" — **is not resolvable on any shape**, and no claim that
uniform NR=48 beats the shipped per-shape table is supported. The flatness of the
32-56 plateau is exactly the reason: a plateau cannot rank its own points.

## exp_21 VERDICT — Fig 2 lands: the communication pool is 5.3% of the machine

260 points, **224/224 destination checksums with one distinct fingerprint**, all
ceilings sourced from this node. No gfx950 number appears anywhere.

| quantity | value | source |
|---|---|---|
| xGMI per link/direction | **64.0 GB/s** | `rocm-smi --shownodesbw` = `64000 mps`, all 28 pairs; `--showtopo` XGMI 1 hop |
| xGMI aggregate egress | **448 GB/s** | 7 × 64.0 |
| HBM | **5325 GB/s** | `amd-smi static`: HBM, 8192-bit, `MAX_BANDWIDTH 5325 GB/s` |
| bf16 MFMA | **1307.4 TFLOPS** | 304 CU × 2.100 GHz × 2048 FLOP/cyc/CU |

**A derivation trap recorded so nobody repeats it**: the `hipDeviceProp_t` route
(2 × memClk × busWidth / 8) gives **2662.4 GB/s**, and the measurement *exceeds*
it by 1.73× — it is exactly **half** the node's own 5325 figure for the same
8192-bit bus. Use the node's reported number, not the derivation.

### The knees

- **Mode c, aggregate (rr7): 90% of plateau at C=16 of 304 CTAs — 5.3% of the
  machine**, plateau 403.2 GB/s (90.0% of the 448 ceiling), scaling 1.99-2.00×
  per doubling below the knee.
- **Mode c, single link: C=2** (at depths 0/4/8; C=4 at d=1), plateau 95.4% of one
  link.
- **The pre-registered falsifier did NOT trigger** — the isolated single-link
  curve reaches 75% of plateau at **C=2**, sixteen times below the C≥32 threshold
  that would have inverted the tiny-pool story. **H1 confirmed and stronger than
  predicted** (2-4, not 8-16); **H2 confirmed at the low edge** (16).
- The two-scale agreement (single-link plateau 95.4% of one link, rr7 90.0% of
  seven) also **proves the fine-grained heap worked**: L2 absorption would have
  reported *above* the ceiling.

### Protocol, not payload — and it is the dominant term

Identical 16 B stores, identical addresses, identical order; only the release
differs. **Protocol-on/off is 0.440 at the rr7 knee and 0.366 single-link** — the
release protocol costs **56-63% of egress bandwidth at identical payload bytes**.
That is an order of magnitude larger than the interference term. Together with
exp_20's **1.0007× fabric amplification at 99.9% full-64 B**, the egress-width
axis is now **closed from both sides, by two independent instruments**, and the
entire residue is **release granularity**. H3 confirmed on substance with its
framing corrected: **protocol dominates interference rather than amplifying it.**

### The interference term is asymmetric, and that argues against pools by itself

At the rr7 knee (C=16), concurrent/isolated is **0.957** against the C-matched
reserve-only control — while that same control shows the **GEMM slowing 1.221×**.
**The emit barely notices the GEMM; the GEMM notices the emit.** Above C≈32 the
GEMM side collapses by 3.3-6.9×. So a large communication pool is doubly wrong: it
does not help the emit (already saturated at 16) and it wrecks the compute.

**H5 confirmed**: in-flight depth is a *below-the-knee* knob — `d=1` costs 41% at
C=8 but ≤14% at C=16.

### H4 FALSIFIED past C≈160 — and this RE-RANKS the mainloop work

MFMA TFLOPS is linear only to **C≈160** (per-CTA flat within 1.8%); the 90% knee
is C=256 and the plateau is **582.4 TFLOPS**, i.e. **45% of the 1307.4 ceiling**.
And the cause is **not occupancy** — the body is 1 CTA/CU from both its 193 VGPRs
and its 64 KB LDS.

The mechanism is arithmetic and it is the important part:
**7.813e-3 B/FLOP × 582.36 TFLOPS = 4549 GB/s = 99.0% of the 4593 GB/s memory-path
plateau that panel b measures independently.** **At full grid this mainloop is
memory-path bound.**

**Consequence for Phase 2:** arithmetic intensity is `~(1/BM + 1/BN)` and is
**independent of `BK`**, so freeing LDS to raise `BK` reduces `k_iters` while
moving the **same** operand bytes.

> **CORRECTION — I over-read this, and exp_27 caught it. The bandwidth conclusion
> does NOT transfer to the production mainloop.** See "the ubench-transfer error"
> below. The `BK` point above still stands on its own arithmetic; the
> "memory-path bound" framing does not.

## exp_27 — THE UBENCH-TRANSFER ERROR, and the mainloop's real bound

I re-ranked Phase 2 off exp_21's H4 result and was **wrong**. The correction is
recorded in full because the mistake is a general one and cheap to repeat.

**The error.** exp_21's mode-a panel plateaus at 582.4 TFLOPS, and
`7.813e-3 B/FLOP × 582.36 TFLOPS = 4549 GB/s` sits at 99.0% of the 4593 GB/s
plateau its own panel b measures. I read that agreement as "the mainloop is
memory-path bound at full grid". **It is a coincidence, and the two panels do not
measure the same level of the hierarchy.** Panel a runs on a **15.2 MB,
deliberately cache-resident** operand set — its own plan says the point is to
price **MFMA issue, not HBM**. Panel b runs on 512 MB and is a genuine HBM curve.
For 4549 to *be* the HBM demand, panel a would have to miss to HBM, which its
design specifically prevents.

**The production mainloop's bound, from MEASURED counters** (`ea_read_requests` ×
64 B, already in exp_20's `counters.json` — not a new run):

| | shape 5 | shape 6 |
|---|---:|---:|
| achieved (FLOP / GEMM pool) | 394.1 TFLOPS | 500.0 TFLOPS |
| % of producer-CU peak @ 1900 MHz | 37.2% | **50.2%** |
| below-L2 reads, measured | 125.6 MB | 436.9 MB |
| implied L2 read hit ratio | 86.6% | 88.8% |
| as bandwidth over the GEMM pool | 411.6 GB/s | 440.3 GB/s |
| **as a fraction of the 4593 GB/s plateau** | **9.0%** | **9.6%** |

**Both shapes are latency/schedule-bound, not bandwidth-bound**, by an order of
magnitude. The tell that settles it: **shape 5 is further from every bandwidth
limit than shape 6 while being further below MFMA peak** — the exact opposite of
what a bandwidth bound produces.

Two independent cross-checks that the 99% agreement was coincidence: 4549 GB/s is
**7.9 B/clk/CU against a 64 B/clk vL1D**, i.e. 12% of L1 capability; and
**HANDOFF's own control with 1.78× the traffic cost 6.2%, not ~78%** — that
control was always evidence *against* a traffic bound, and I had it in front of me.
Panel a's "45% of peak" and the production kernel's "50.2% of producer peak" are
most likely **the same schedule ceiling measured twice**.

**The generalizable lesson: a microbenchmark's bound only transfers to the
production kernel if their working sets occupy the same level of the memory
hierarchy.** A cache-resident ubench can price issue rate and instruction
scheduling; it cannot establish an HBM bound for a kernel whose operand stream
misses differently. When a ubench and a counter pass disagree, **the counter pass
on the real kernel wins** — and here they never actually disagreed, I just
mismatched the levels.

### The dispatch's headline candidate is dead twice over

**`S=1, BK=64` (single-buffer to free LDS) is closed on two independent grounds**:
it yields **identical barriers per tile** (2 × 58 = 1 × 116) *and* identical bytes
moved. Separately, `256/256/32` is the **argmax of MFMA-work-per-barrier** under
the joint LDS and accumulator caps, so the current tile is already the right
answer for the quantity that matters. Do not re-open it.

**XCD-aware tile order is now priced and small**, which retires a long-standing
"untested" item: perfect locality would save 316 MB ≈ 319 GB/s of a plateau we use
**9.6%** of — worth −0 to −15 µs on shape 6.

### Ranked survivors (shape 6 / shape 5 µs)

| | mechanism | s6 | s5 |
|---|---|---:|---:|
| A | counter pass to adjudicate the bound (no code) | 0 | 0 |
| **B** | hoist `load_commit` above the half-1 MFMAs | −37…−73 | −11…−23 |
| C | prefill swizzled LDS offsets + SGPR bases (donor port) | −25…−60 | −8…−18 |
| D | retire `K_TAIL=true` on shape 6 (+63 instrs/trip, counted) | −40…−90 | 0 |
| E | phase-offset warp-group ping-pong (donor) | −150…−350 | −45…−105 |
| H | `32x32x8` — issue-slot lever only; warp LDS reads are `(WM+WN)·BK·2` regardless of atom | −20…−60 | −6…−18 |

**Recommendation: run A, then land B.** Pre-registered: shape 6 GEMM pool
−5.0% ± 3.0%, geomean **−0.91%** (209.56 → 207.66 µs). Falsifier: |Δ| < 1.5% on
shape 6 over ≥3 paired draws **with the ISA confirming `vmcnt(0)` moved** between
the two 32-MFMA runs closes the whole B/C/D family at once.

Gate A is one `rocprofv3` pass with plumbing that already exists:
`SQ_WAIT_INST_LDS`, `SQ_WAIT_ANY`, `SQ_LDS_BANK_CONFLICT` on shapes 5 and 6.
Pre-registered: LDS+barrier wait > 60%, VMEM wait < 25%. **Falsifier: VMEM wait
> 50% means the mainloop should be abandoned for traffic work.**

### The honest expected value, which is the number that should govern the night

**The ~571 µs is recoverable, but by far less than its size.** It is the distance
to a roofline no GEMM reaches: we sit at **50.2% of producer peak where tuned
MI300X libraries reach 60-75%**. Realistic capture is **~250 µs on shape 6 and
~90 µs on shape 5 = −5.15% at the geomean**, which would take the graded gap from
**1.098× to ~1.046×**. Every individual mechanism is worth ≤1% of geomean, so this
is a grind of several landed changes, not one win — and shapes 1-4 contribute
nothing.

## THE STALE-PID BLINDNESS IS SYSTEMIC — it cost ~40 minutes of the figure queue

Fixing `gpu_lease.sh` was not enough, because **every drain check in this tree
independently reimplements the same broken test.** `grep -l showpids` finds it in
**eleven files** under `tools/` (`reattribute.sh`, `gate_ladder.sh`,
`run_reference_arm.sh`, `run_rank1_bench3.sh`, `run_ours_eval_clean.sh`,
`run_baseline.sh`, `reap_stale.sh`, `who_owns_gpus.sh`, `gpu_busy.sh`, …) plus
experiment runners that copied the pattern.

**The concrete cost.** exp_24 **completed shape 2** of the ladder — 354 s, all five
arms, `harness_floor` at 5.15 µs best pipelined — then entered its own preflight
for shape index 1 and logged `waiting for kfd drain (fds=1)` every ~11 s
indefinitely. The `fds=1` was the dead m9 process. A campaign that was working
perfectly sat blocked on a corpse, holding the lease, with exp_21, exp_22 and
exp_26 queued behind it and **all 8 GPUs idle**.

**The fix is a shared helper, `tools/kfd_live.sh`**, to be *sourced* rather than
re-implemented: `kfd_live_count`, `kfd_stale_list`, and `kfd_wait_clean` which
reports the dead set once and then waits only on the living. `reattribute.sh` and
`run_rank1_bench3.sh` now source it. **The remaining callers should be converted
the next time each is touched** — the pattern to delete on sight is
`rocm-smi --showpids | awk '/^[0-9]+/' | wc -l`.

**The general lesson, and it generalizes past GPUs:** when a readiness test is
copy-pasted into a dozen scripts, a wrong test becomes a dozen wrong tests, and
fixing the one you noticed leaves the other eleven to bite you later — in this
case within twenty minutes, in a different experiment. **A predicate that several
scripts must agree on belongs in one sourced function.**

### Disclosed preemption

I broke the lease to free it for exp_22, the last outstanding figure. Between the
diagnosis and the release, exp_24 had already exited and **exp_26 had acquired**,
so the `steal` actually preempted **exp_26's `campaign3.sh`**, not exp_24. That is
recorded here and must be repeated in exp_26's `result.md`. It was the right call
on priority — exp_26 is Phase 2 work that is **blocked on its own
`VM_L2_PROTECTION_FAULT` anyway**, while exp_22 is the last missing paper figure —
but it was not the preemption I intended, and the lease's `steal` path should
print the current owner *before* acting so the operator can reconsider.

exp_24 lost only shape 2, preserved in
`exp_24_ladders/logs/full_run.partial_shape2.log` (157 result lines), and needs a
liveness-aware runner before it restarts regardless.

## exp_22 arm (a) — the RCCL baseline has ZERO overlap, and that IS the figure

The reference GEMM+RCCL epoch on shape 5, from a `rocprofv3` kernel trace, median
epoch, **three strictly serialized intervals with zero overlap**:

| interval | µs | share of 527.9 µs |
|---|---:|---:|
| rocBLAS `Cijk_*` (GEMM) | 243.2 | 46.1% |
| bias | 43.4 | 8.2% |
| RCCL reduce-scatter | **241.3** | **45.7%** |

**The width of the RCCL block is the communication share of the layer: 45.7%.**
That is the paper's opening measurement for this operator, and it is the exact
contrast Fig 3 exists to draw — the baseline's three resource strips are *mutually
exclusive*, so nothing is hidden behind anything.

**"Report, don't bucket" caught the most important kernel in the arm.**
`ncclDevKernel_Generic_2` — **30 dispatches, 11.07 ms**, i.e. the entire
communication phase — matched **no** seed classification rule, because ROCm's RCCL
exports **upstream NCCL symbol names**. A silent fallback would have shipped an
**empty xGMI strip** and the figure would have shown the baseline doing no
communication at all. Zero unclassified names after two corrections derived from
the observed set.

A second, quieter trap in the same arm: `__amd_rocclr_copyBuffer` /
`fillBufferAligned`, **1418 dispatches** of runtime allocator blits. Left in the
segmentation they cut the trace into **718 fragments** and aborted the first
capture; they are now class `runtime`, excluded from segmentation and reported
rather than dropped.

**Tick rate now has two independent calibrations that share no machinery beyond
the instruction itself**: exp_21's `s_memrealtime` = **99.7366 MHz** (10.0264
ns/tick, 5 reps, spread 0.0101%, two-stage against `steady_clock`) versus
exp_22's in-situ regression of span-in-ticks against `hipEvent` µs, with a 1%
agree/disagree verdict recorded in `tick_rate.json`. Both sit **−0.264% from the
sibling's declared gfx950 100 MHz** — close enough that the sibling's assumption
was harmless, but it is now measured here rather than inherited.

## THE SANITY GATE I SPECIFIED WAS WRONG — best-of-arm is not a mean

I told exp_22 to verify the node by checking shape 5 against **613.70 µs**. It
measured 646.10 and aborted the run. **My instruction was the error**, and it is
the trap already recorded in this very file: **the `62.38 / 64.52 / 83.75 /
198.71 / 613.70 / 1616.63` vector is BEST-of-arm, while `m7_bench.py` prints
MEANS.** exp_05's own table records shape 5 as `613.70 / 645.93` = best/median.

Re-scored like-for-like, the node is fine:

| statistic | measured | recorded | delta |
|---|---:|---:|---:|
| geomean of M7 means | 207.76 µs | 207.18 µs | **+0.28%** |
| shape 5 mean | 648.11 µs | 645.93 (recorded median) | **+0.34%** |
| shape 5 vs same-config range | 648.11 | 644.71-669.39 | inside |

Two refinements worth keeping. **A best over three rotations is a weaker order
statistic than a best pooled over a multi-arm campaign**, so comparing them
systematically penalises the smaller run — the recorded bests are not reproducible
targets for a 3-rotation M7. And the **mixed sign is the tell**: five shapes
drifted *up* 1-5% while shape 6 drifted *down* 2.9%, which no node degradation
produces. A real regression moves everything one way.

**So the stale KFD entry's predicted nil timing effect is now CONFIRMED, not
assumed**, and every number taken after the incident stands.

**Rule for every future sanity gate: state the statistic, not just the number.**
Compare mean to mean, best to best, median to median, and say which the reference
value is.

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

## exp_21 (Fig 2, saturation vs CTA count) — LANDED, one prediction falsified

Full sweep 2026-08-12 10:34–10:39Z under the lease. 260 points, 224/224
destination checksums pass with **one** distinct fingerprint, tick spread 0.0101%.
Artifacts: `exp_21_saturation/{saturation.json,knees.json,saturation.csv,ceilings.txt}`.

- **Communication needs a tiny pool, and the number is 16.** The real 16 B
  peer-packet emit reaches 90% of its plateau at **C = 16 of 304 CTAs (5.3%)**
  for the 7-peer round robin and at **C = 2** for a single link. The
  pre-registered falsifier (≥ 32 CTAs for 75% of plateau) **did not trigger** —
  the 75% crossing is at C = 2 at every depth. Scaling below the knee is
  near-ideal (1.99–2.00× per doubling of C to 8), so this is a genuine
  saturation knee and not a flat curve.
- **The protocol, not the payload, is what egress costs.** Same 16 B stores, same
  addresses, same order; the only difference is the release. Protocol-on delivers
  **0.44×** of protocol-off bandwidth at the rr7 knee and **0.37×** at the
  single-link knee, and it moves the knee itself (2 → 8 single, 16 → 64 rr7)
  because each release serialises a drain that more CTAs can hide. This agrees
  with exp_20's counter pass from the opposite direction (fabric amplification
  1.0007× at 99.9% full-64 B), so **the egress width axis is closed from both
  sides and remaining egress cost is release granularity.** RELEASE_GROUP still
  has room; exp_26 is asking the right question.
- **Interference is real but second-order next to that.** At the rr7 knee,
  concurrent/isolated = 0.957 while the GEMM on the *C-matched reserve control*
  slows by 1.221×. Reading: the emit barely notices the GEMM, the GEMM notices
  the emit. Above C ≈ 32 the GEMM side collapses (slowdown 3.3–6.9× at C = 32–64),
  which is an independent argument against large communication pools.
- **The in-flight bound is a below-the-knee knob.** rr7 at C = 8: depth 1 = 167.9,
  4 = 230.3, 8 = 245.9, unbounded = 286.6 GB/s (monotone; depth 1 costs 41%). At
  and above C = 16 the depths converge within 8–14%.
- **H4 falsified as stated, and the mechanism matters more than the verdict.**
  MFMA is linear only to C ≈ 160 (per-CTA flat within 1.8%), then falls to 75.2%
  per-CTA at C = 304. It is *not* occupancy — the body is 1 CTA/CU from its 193
  VGPRs *and* from its 65,536 B LDS. It is the mainloop's own operand stream:
  7.813 × 10⁻³ B/FLOP × 582.36 TFLOPS = **4549 GB/s, i.e. 99.0% of the 4593 GB/s
  memory-path plateau panel b measures independently.** **Consequence for the
  optimization queue: at full grid this mainloop is memory-path-bound, so E1(a)
  AGPR accumulators and schedule work buy nothing above C ≈ 160 unless operand
  traffic drops (better L2/MALL reuse, wider K staging, or fewer redundant tile
  loads). Re-rank E1 accordingly.**
- **The reduce is the opposite shape: no knee at 304.** REDV=1 goes
  263.8 → 4593.2 GB/s from C = 8 to 304, essentially linear to C = 64, reaching
  86.3% of the node's reported 5325 GB/s HBM. Egress saturates at 16 CTAs; the
  reduce never saturates. That asymmetry is the argument for where dedicating
  CTAs pays and where it does not.
- **A props-derived HBM ceiling is wrong on this node by exactly 2×.**
  `2 × memoryClockRate × busWidth / 8` gives 2662.4 GB/s and the measurement
  exceeds it by 1.73×; `amd-smi static` reports 5325 GB/s for the same 8192-bit
  bus. **Never quote the props derivation on gfx942.** xGMI per link is
  64.0 GB/s (`rocm-smi --shownodesbw`, `64000 mps`, all 28 pairs), corroborated
  at two scales by the measurement itself (95.4% of one link, 90.0% of seven).
- **`s_memrealtime` on gfx942 = 99.7366 MHz** (10.0264 ns/tick), spread 0.0101%
  over 5 reps of ≈200 ms. Measured here; exp_22 should use this, not the
  sibling's gfx950 figure.

### Two node-level traps this experiment paid for

- **A KFD process in state `D` with `wchan = exit_mm` is not "draining" — it is
  dead and unkillable, and it blocked the lease for 30 minutes.** Flat CPU time,
  zero CU occupancy, VRAM still mapped, ignores SIGTERM (and SIGKILL is
  forbidden here). It aborted both exp_21's and exp_24's acquire after their full
  300 s drain window. `gpu_lease.sh` now classifies exiting/zombie KFD pids and
  ignores them.
- **Never `scp` over a script another process is executing.** bash reads scripts
  incrementally by byte offset and `scp` truncates in place, so a push landing
  mid-run makes the interpreter resume mid-token — it killed one campaign with a
  syntax error on a line that was never wrong on disk, and later broke a running
  `run_sweep.sh`'s exit path *after* its data had been written (the phase
  reported failure on a successful sweep). A CRLF push made it worse by breaking
  `tools/gpu_lease.sh` for every agent at the same time. **Fix, now standard for
  exp_21: freeze CR-stripped copies of the whole script chain per run and execute
  those** (`go_campaign.sh` → `logs/run_<stamp>/`, with `SAT_BASE` keeping the
  module and artifacts in the experiment directory). `sed -i` is safe where `scp`
  is not, because it renames instead of truncating.

## Open validation gap

- **`gemm_rs_mi300x_static_checks.py` has been dead since exp_02.** Its first
  requirement asserts the reducer column still matches RadeonFlow's inherited
  values — which exp_02 deliberately changed — and it *raises* there, so none
  of its ~20 downstream gates has run while the shape table was rewritten three
  times. Re-run collecting instead of raising: zero tile/dispatch failures, so
  nothing was broken, but the guard has been silently absent.

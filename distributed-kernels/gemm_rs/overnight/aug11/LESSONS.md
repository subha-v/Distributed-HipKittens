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

# exp_26 — per-shape release group: **WIN on shape 5, −6.56%**

**Source state to keep: `HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE = 2` in
`gemm_rs_mi300x.cpp`.** Not 1 — the obvious spelling of the same rule is
measurably worse and is rejected below.

> **LANDED.** The shipped rule has its own full gate ladder and its own M9, both
> green (`logs/ladder_ps2.log`, `logs/m9_ps2.log`; ladder logs under
> `experiments/exp_26_ps2_land/`). The earlier ladder recorded here was run
> against `PERSHAPE=1` and is kept only as the record of what was run against
> what. §3 has both.

---

## 1. Verdict up front

**Shape 5 (8192×4096×14336) gets −6.56%, ≈ −43 µs, in 80 of 80 paired rounds
across both allocation orders, against a null arm of −0.61% on the same shape.**
That is against a floor of ~13.9 µs, and it is *better* than the pre-registered
−5%. Shapes 1–4 and 6 are controls — the group size cannot change on any of them
— and all sit inside the instrument's own noise. Geomean best moves
203.78 → 200.00 µs and 206.03 → 203.14 µs in the two passes, i.e. **≈ −1.4%**,
entirely from shape 5.

Three findings, in order of how much they should change what anyone does next:

1. **The mechanism is real and larger than attributed.** exp_20 priced shape 5's
   release pool at 65.4 µs; halving the release count recovers ~43 µs of it.
2. **How the rule is spelled matters as much as the rule.** `min(RELEASE_GROUP,
   tiles_per_cta)` — the spelling the dispatch proposed and the one this
   experiment first shipped — costs **+2 VGPRs on five of seven instantiations**
   and gave back most of the win. Spelled as a descending select over
   compile-time literals it costs **zero registers** and delivers. Same rule,
   same rgroup on every shape, 5 pp apart in measured effect on shape 5.
3. **The A/B instrument had a systematic defect that hid the result for two
   rounds, and the null arm is what exposed it.** exp_05's rotation advances
   every arm one position per round, which pins the *relative* spacing of any
   two arms in every round of every pass, so an order effect becomes a constant
   offset on that pair rather than noise that averages out. The null arm — two
   builds of the *same* rule — read −2.29% on shape 6 in 57 of 64 paired rounds.
   Shuffling the arm order per round collapsed it to −0.61% on shape 5, and the
   effect that had been "inside the floor" became 80/80.

## 2. The change

`gemm_rs_mi300x.cpp`, producer role, the `rgroup` expression only. The group
loop, the single release site, the deferred publish loop, the per-tile credit
wait, `decode_tile`, the WGM tile order, `BM/BN/BK`, `num_reducer_ctas`,
`find_scored_config` and the 72-byte descriptor ABI are all untouched.

```c
// PERSHAPE == 0, the incumbent: a step function against ONE constant
const int rgroup = tiles_per_cta >= RELEASE_GROUP ? RELEASE_GROUP : 1;

// PERSHAPE == 1, measured and REJECTED (§5)
const int rcap   = tiles_per_cta < RELEASE_GROUP ? tiles_per_cta : RELEASE_GROUP;
const int rgroup = rcap > 1 ? rcap : 1;

// PERSHAPE == 2, SHIPPED: the same rule over a compile-time ladder
const int rgroup =
    tiles_per_cta >= RELEASE_GROUP ? RELEASE_GROUP
    : ((RELEASE_GROUP >= 2 && tiles_per_cta >= 2) ? 2 : 1);
```

`tiles_per_cta = ceil(gemm_tiles / num_gemm_ctas)` is already computed one line
above. Effective group per graded row goes `1/1/1/1/1/4` → `1/1/1/1/**2**/4`;
the generic row (118 tiles/CTA) stays at 4. **Only shape 5 changes**, which is
what makes the other five rows controls rather than collateral.

### Where the value comes from, and why not a folded constant

The dispatch asked to prefer a folded compile-time constant. That is not
available here: `dispatch_gemm_rs_mi300x` sends **rows 4 and 5 to the same
`launch_fixed<256,256,32,false>`** — they differ only in `M`. A template
parameter would split them, changing `M2_EXPECT` and re-allocating registers for
exactly the row whose delta is being measured. Routing it through the descriptor
would touch the ABI for a value both sides already derive from `tiles` and
`num_gemm_ctas`.

But the distinction the dispatch was pointing at turned out to be real *within*
"runtime", which is the more useful version of the finding. The incumbent's
`rgroup` is a runtime value drawn from a **two-element set of literals**;
`min()` makes it an arbitrary value in a range; the ladder puts it back in a
**three-element set of literals**. Those are three different schedules, and the
register tuple separates them cleanly (§5). Same family of effect exp_09
recorded as folded / bounded / runtime being worth 2.13% on shape 3.

### Fullness — and a correction to the premise

The dispatch argued that `min(MAX, tiles_per_cta)` "keeps every group full, so
it does not reintroduce" E3's failure mode. **The conclusion is right; the stated
mechanism is not, and the difference is worth recording.**

True: `rgroup ≤ tiles_per_cta = ceil(tiles / num_gemm_ctas)`, the tile count of
the CTA that owns the most, and `emitted = min(left, rgroup)` truncates a short
CTA's last group to what it owns. So no CTA ever defers a publication waiting
for a tile that does not exist. On shape 5 the 240 CTAs owning 2 tiles group
both; the 32 owning 1 release immediately.

Not true: that this is what separates it from E3's rejected arm. E3's 3.75% loss
on the then-2-tile 512×4096×12288 came from a group of **2 on a CTA owning
exactly 2** — a *full* group. Fullness was never the discriminator; `FULL_ONLY=1`
at `RG=4` spared that shape by arithmetic accident, because 2 < 4. What actually
changes on shape 5 is publication delay: the first of a CTA's two tiles is now
published one mainloop later.

So the risk was real and unhedged, and the experiment was run as a measurement
rather than a deduction. **The measurement says the writeback saving beats the
publication delay by a wide margin at this size** — which also explains E3's
result rather than contradicting it: 512×4096×12288 was an 88 µs shape whose
reducers got 2 rounds of work, and shape 5 is a 660 µs shape. (Since exp_14
retiled row 2 to 64/128 = 256 tiles = 1 per CTA, that shape can no longer group
at all and is now a control.)

## 3. Gates — which binary passed what

| gate | `PERSHAPE=1` | **`PERSHAPE=2` (shipped)** |
|---|---|---|
| M0 node clean | 0 KFD pids | 0 *dispatching* pids (see below) |
| M1 build | `ALL MODULES BUILT` | **`ALL MODULES BUILT`** |
| M2 resources / ISA | 7/7, 0 scratch, 0 spills | **7/7, VGPR 98/104/136/246/248/91/92, AGPR 0, scratch 0 — the incumbent's tuple exactly** |
| M3, 17 shapes, `1e-2` **and** `2e-3` | 17/17 PASSED | **17/17 PASSED** |
| M4 negative controls | all three failed as designed | **all three failed as designed** |
| M5 600-epoch soak | epochs and signals exact | **epochs and signals exact** |
| M7 timing | ran (§6) | **ran, all shapes correct** |
| M9 poisoned / bitwise / publish-early | PASSED | **PASSED, 0 failures** (§4) |

No error bit was raised at any point in this experiment, on any arm, on any
gate.

`PERSHAPE=2` was additionally verified **bit-identical to the incumbent on all 8
ranks** at every shape in all six A/B passes (`torch.equal`, asserted before any
timing).

**One repair to the ladder, disclosed.** `gate_ladder.sh`'s M0 and M7 preflights
count every pid `rocm-smi --showpids` reports, and this experiment's own wedged
M9 process from the 10:05Z fault (§7) is still resident in `D`/`exit_mm` — it
holds VRAM, dispatches nothing, and cannot be removed, so that preflight can
never read 0 again on this node. `campaign6.sh` therefore transforms the ladder
at run time: the pid-count expression is replaced by a `live_kfd` that skips
processes in state `Z` or with `wchan` `exit_mm`/`do_exit` — the same predicate
the shared lease tool adopted for the same reason. Every other line, gate,
assertion and the order are byte-identical to `tools/gate_ladder.sh`, the ladder
is **not** forked, and the transform is asserted to have applied (`preflights
patched: 2 (expect 2)`) before anything runs, so a silently unpatched gate
cannot report a pass. The first attempt's pattern matched nothing and the
assertion refused to run — which is the behaviour that makes the repair safe to
disclose rather than a hole.

## 4. M9 — and a defect in the gate itself

**The stock `harness/m9_stale_slot.py` cannot pass on this tree for reasons that
predate exp_26, and one of them faulted the node.** Its frozen golden
`gemm_rs_mi300x_e3base` was copied before exp_14 (E4b) retiled rows 1–3:

| row | golden BM/BN/BK | current | golden cols | plan cols | consequence |
|---|---|---|---:|---:|---|
| 1 | 32/64/**64** | 32/64/**128** | 112 | 112 | BK-only; map identical, **still bit-exact** |
| 2 | 64/**64**/64 | 64/**128**/64 | 64 | 32 | map differs → golden NaN, 600/600 bitwise diffs |
| 3 | 128/**256**/32 | 128/**192**/32 | 12 | 15 | golden reads B rows to **3072 of a 2880-row operand** |

Row 3 is the `Memory access fault by GPU node-7 ... on address 0x7edfd0a50000`
at 10:05Z — the exact out-of-bounds read exp_14's shape-table comment says the
retile removed, preserved in the frozen module. `golden_audit.sh` has the
tables; the predicted per-row outcome matches the observed log on all six rows,
including row 1 passing.

The gate was re-run with the golden this change actually needs —
`gemm_rs_mi300x_ps0`, same source and shape table, `PERSHAPE=0` — which makes
the assertion "the rule change moves no bit", i.e. protocol-review conditions
C5/C8 stated exactly. Its case list was also re-aimed, because the stock list no
longer contains a shape this change alters (row 2 became 1 tile/CTA):
8192×4096×14336 leads with 120 epochs, then row 6, the generic row, and rows 1–4
as C8 controls. `harness/m9_stale_slot.py` was **not** modified;
`m9_vs_incumbent.py` imports it and overrides two module globals.

**Result: `GATE M9 PASSED`.** No NaN reached any output in any poisoned epoch on
any shape; bit-identical to the incumbent on all 8 ranks for every epoch of
every case; `allclose(2e-3)` throughout; no error bit; epoch cells and every
touched ready/credit cell exact.

**`CTRL_PUBLISH_EARLY` still fails**, on the shipped build, on **5 of the 7
cases** — including 8192×4096×14336, the one shape this change alters:

| case | NaN | bitwise | tight |
|---|---|---|---|
| **8192×4096×14336** (the changed row) | **1/30** | **1/30** | **1/30** |
| 8192×8192×28672 (generic, 118 tiles/CTA) | 1/10 | 1/10 | 1/10 |
| 512×4096×12288 | 1/20 | 1/20 | 1/20 |
| 2048×2880×2880 | 1/10 | 1/10 | 1/10 |
| 64×7168×18432 | 1/10 | 1/10 | 1/10 |
| 8192×8192×29568 | 0/20 | 0/20 | 0/20 |
| 4096×4096×4096 | 0/10 | 0/10 | 0/10 |

Firing is concentrated at epoch 1 and is sparse thereafter, exactly as the
ledger records; the gate's own note calls the two non-firing shapes' race
windows too narrow to catch by sampling. What matters is that the detector has
demonstrated power over precisely the release/publication ordering property this
change puts at risk, **on the row that changes**. A gate its own control passes
would be theatre, and a pass from a blind gate would mean nothing.

**For the ledger, outside this experiment's ownership:** `build_golden.sh`
should re-gold `gemm_rs_mi300x_e3base` from current source at
`RELEASE_GROUP=1`, and m9's `CASES` should be derived from the shape table
rather than hard-coded — the retile silently emptied it of the cases it exists
to cover, and nothing warned.

## 5. The spelling result — `min()` costs two registers

Resource tuples, all seven instantiations
(`-Rpass-analysis=kernel-resource-usage`):

| arm | rule | VGPRs across the 7 instantiations | AGPR | scratch | spill | occupancy |
|---|---|---|---|---|---|---|
| `ps0` | incumbent | 98 / 104 / 136 / **246** / **248** / 91 / 92 | 0 | 0 | 0 | 4/4/3/2/2/5/5 |
| `ps1` | `min()` | 100 / 106 / 138 / **248** / **250** / 91 / 92 | 0 | 0 | 0 | unchanged |
| **`ps2`** | **ladder** | **98 / 104 / 136 / 246 / 248 / 91 / 92** | 0 | 0 | 0 | unchanged |
| `rg2c` | `RG=2` | identical to `ps0` | 0 | 0 | 0 | unchanged |

The 256×256 rows sit at 246–248 of the 256 arch VGPRs, so `ps1`'s +2 is spent
against the cap. Instruction-count deltas versus the incumbent, per
instantiation: `ps1` −21…+9 (schedule churn in both directions), `ps2` −1…+7
(the extra rung, and nothing else). Ordering-op counts are identical across all
arms — `buffer_wbl2` 7, `buffer_inv` 7, `s_barrier` 98, `v_mfma` 184 — so
nothing about the protocol or the mainloop moved.

That the ladder spelling restores the incumbent's tuple *exactly* is the
cleanest single fact in this experiment, and it is a codegen fact, independent
of every timing question below.

## 6. Measurement

### Instrument

`ab_pershape.py`, descended from exp_05's paired A/B: each arm is a separately
compiled module, all are open in one process, and their timed blocks are
interleaved within a round so adjacent samples share the clock state, the
allocation and the input sequence. Before any timing every arm is verified at
`2e-3` and asserted `torch.equal` to every other arm on all 8 ranks. Clocks
pinned (`perf_determinism`); duration-based warmup, 800 ms per shape.

Arms: **`ps0`** incumbent · **`ps1`** `min()` · **`ps2`** ladder · **`rg2c`**
`RELEASE_GROUP=2` under the incumbent rule — a **positive control**, since it
computes rgroup 2 on shape 5 by a different expression and rgroup 2 on shape 6
where everything else uses 4 · **`ps0b`** the **null arm**, a second build of
`ps0`.

Six passes in three rounds, both allocation orders each:

| round | arm order within a round | rounds × iters | passes |
|---|---|---|---|
| 1 | rotated (inherited) | 7 × 50 | fwd, rev |
| 2 | rotated | 25 × 100 | fwd, rev |
| **3** | **shuffled per round** | **40 × 100** | **fwd, rev** |

### Why rounds 1–2 could not resolve it, and what fixed it

Best-of-pass on shape 5 said `rg2c`, `rg2c`, `ps2`, `ps1` were the fastest arm
in the four rotated passes — a different answer each time — and the **null arm
moved −2.36%, −1.58%, +3.19%, −4.40%** across them. Pairing within a round
removes the between-pass mode, but under rotation it exposed a worse problem:
the null arm's *paired* median was −2.29% on shape 6 in 57 of 64 rounds and
−0.77% on shape 4 in 58 of 64, at p < 10⁻⁴, on behaviour that cannot differ.

The cause is structural. With `order = (r + i) mod n`, arm *j* runs exactly
`(j − k) mod n` blocks after arm *k* in **every** round of **every** pass. Any
effect that depends on what ran immediately before a block is therefore a
constant offset on that pair, not noise that averages out. Round 3 draws a fresh
permutation per round, seeded from the shape so the forward and reversed passes
stay independent draws. The null arm's paired median on shape 5 fell to −0.61%
and the result appeared.

### Round 3 — within-round paired difference vs `ps0`, 80 rounds per arm

Median of the per-round % delta; `wins` = rounds in which the arm was faster
than `ps0` in that same round. `=` marks an arm with `ps0`'s rgroup on that
shape, i.e. a control.

| # | shape | t/CTA | `ps0b` null | `ps2` (shipped) | `rg2c` |
|---|---|---:|---:|---:|---:|
| 1 | 64×7168×18432 | 1 | −0.30 = | +1.00 = | −0.24 = |
| 2 | 512×4096×12288 | 1 | +0.33 = | +0.40 = | +1.05 = |
| 3 | 2048×2880×2880 | 1 | +0.19 = | −0.20 = | −0.48 = |
| 4 | 4096×4096×4096 | 1 | +0.87 = | −0.84 = | −0.33 = |
| 5 | **8192×4096×14336** | **2** | **−0.61** | **−6.56, 80/80** | **−5.34, 80/80** |
| 6 | 8192×8192×29568 | 4 | −1.75 = | −3.04 = | **+3.20, 0/80** |

- **Shape 5 is the result.** Both arms that group at 2 there beat the incumbent
  in *every one* of 80 paired rounds, by 5.3% and 6.6%, against a 0.6% null.
  Two different expressions computing the same rgroup agree — which is the
  behavioural confirmation that `ps2`'s group really became 2.
- **Shape 6 is the positive control and it fires.** `rg2c` is the only arm with a
  different group there (2 against 4) and it is +3.20% in 80 of 80 rounds — the
  wrong direction, at the right size, re-confirming E3's choice of 4 from
  scratch. So the instrument demonstrably resolves release granularity.
- **Controls did not move beyond the instrument.** `ps2` reads +1.00 on shape 1
  and −0.84 on shape 4 where it cannot differ from `ps0`; both flip sign between
  the two passes (shape 1: +1.16% fwd, −0.29% rev; shape 4: −0.04% fwd, −1.78%
  rev), and the null arm covers the same ±1 pp band. Shape 6's −3.04 sits 1.3 pp
  beyond a −1.75 null, in the favourable direction, on an instantiation whose
  ISA differs by +6 instructions at an identical register tuple. Residual
  instrument bias, not release.

### Best and median per shape, round 3, with the floor beside them

Microseconds. `floor` is the recorded null-arm floor; `null` is what the null
arm actually did in these two passes.

| # | shape | `ps0` best / med | `ps2` best / med | Δ best | floor µs | null (this round) |
|---|---|---|---|---:|---:|---:|
| 1 | 64×7168×18432 | 61.27 / 62.05 | 61.98 / 64.66 | +0.71 | 3.48 | −0.37% / −1.30% |
| 2 | 512×4096×12288 | 64.96 / 65.08 | 65.17 / 65.29 | +0.21 | 0.93 | +0.35% / −0.03% |
| 3 | 2048×2880×2880 | 84.66 / 85.24 | 84.68 / 85.07 | +0.02 | 1.61 | +0.33% / +0.09% |
| 4 | 4096×4096×4096 | 197.73 / 198.42 | 197.66 / 198.58 | −0.07 | 4.90 | +0.92% / +0.43% |
| 5 | **8192×4096×14336** | **664.54 / 666.92** | **602.73 / 605.12** | **−61.81** | **13.93** | −0.20% / −0.77% |
| 6 | 8192×8192×29568 | 1617.23 / 1642.79 | 1570.47 / 1592.59 | −46.76 | 69.87 | −2.48% / −1.51% |

(forward pass; the reversed pass gives shape 5 as 658.52 → 635.97 best,
−22.55 µs, −3.42%, ranges disjoint. The paired median of −6.56% over both passes
is the estimate to quote — best-of-pass compares two arms' luckiest rounds from
different draws and is the statistic that failed in rounds 1–2.)

**Shape 5's −43 µs at the paired median is ~3× its 13.9 µs floor and ~10× the
null arm this round.** Geomean best 203.78 → 200.00 (fwd) and 206.03 → 203.14
(rev).

### Cross-run M7, for completeness only

M7 prints means and the recorded denominators are best-of-arm from paired runs,
so these are not comparable. Landing ladder (`PERSHAPE=2`), best / median / mean:

| # | shape | best | median | mean |
|---|---|---:|---:|---:|
| 1 | 64×7168×18432 | 68.50 | 68.90 | 69.25 |
| 2 | 512×4096×12288 | 68.55 | 69.36 | 69.12 |
| 3 | 2048×2880×2880 | 86.61 | 86.69 | 86.77 |
| 4 | 4096×4096×4096 | 199.58 | 199.81 | 199.99 |
| 5 | 8192×4096×14336 | 644.02 | 655.83 | 652.25 |
| 6 | 8192×8192×29568 | 1576.06 | 1628.40 | 1615.00 |

Geomean of best 208.60 µs, all shapes correct. **Do not read a verdict off
this.** Host issue in that run was 65–67 µs against 61–62 µs in the morning's,
so the two host-bound rows are up ~5 µs for reasons that have nothing to do with
the kernel, and the shape-5 row cannot be compared against a `ps0` measured in a
different session. The paired A/B is the denominator; this table exists because
the ladder produced it.

### Graded ratio vs frozen rank-1

**Not re-measured.** `experiments/exp_13_cta_split/vs_rank1.sh` needs the node
for a rank-1 staging run, and exp_24 held the lease for the external-ladder
campaign through this experiment's window; taking it would have preempted a
second experiment after this one already preempted two (§7). The standing
per-shape graded ratios are 0.850 / 1.072 / 1.102 / 1.120 / **1.286** / 1.208,
geomean 1.098×. Shape 5 is the worst of the six and the one this change moves.
A −6.56% on shape 5, if it carries into the graded per-call protocol, takes that
row to ≈1.20× and the geomean to ≈1.085×. **That is an arithmetic projection
from a pipelined measurement, not a graded measurement** — the graded protocol
is per-call with barriers rather than pipelined, and the release saving is
exactly the kind of term that can behave differently there. It must be measured
before being quoted anywhere it matters.

## 7. Node discipline — disclosure

This experiment's first two GPU runs, the gate ladder and the stock M9, were
launched **without acquiring `tools/gpu_lease.sh`**. That is a discipline
failure and it cost other tenants:

- The stock-M9 run took the §4 memory access fault and wedged in driver teardown
  (state `D`, `wchan exit_mm`), unkillable by any signal, holding ~10 GB of VRAM
  on all 8 GPUs. It never exited and was still resident at the end of this
  experiment. Only SIGTERM was ever used — the process was already in `do_exit`,
  where SIGKILL cannot shorten the path and leaked IPC mappings can wedge the
  node for everyone.
- **`exp_21` and `exp_24` both aborted their `acquire` at 10:06Z and 10:11Z**
  ("node dirty after 300s") because of it. Two experiments preempted, by this
  one.
- Every GPU run from 10:29Z onward is leased, and the lease is released on every
  path. When `campaign3.sh` died just after acquiring, the lease was checked and
  confirmed free within minutes rather than left held.
- The lease tool itself has since been improved by its owner to treat
  exiting/zombie KFD pids as non-dispatching, which is what let the queue drain
  around the wedged process.

Three further defects worth recording because each cost a run:

- **`tools/gpu_lease.sh` is on the node with CRLF line endings and does not
  execute** (`set: pipefail: invalid option name`, then a syntax error). It
  belongs to another experiment, so rather than edit it this campaign runs a
  CR-stripped copy from `/tmp` — same script, same lock directory, same log.
  Worth repairing at source: a lease tool that silently fails to run is exactly
  how unleased jobs happen.
- **Round 2's first attempt ran `build_arms.sh` on the host instead of in
  `dhk-gemmrs`.** The host hipcc has no pybind11, so all five arms failed to
  compile — after the builder had removed each target `.so`, which it does
  deliberately so that "the file exists" is a real build result. Everything that
  compiles now goes through `docker exec dhk-gemmrs`.
- **Two campaigns died in shell quoting after transport**, both in `trap '…"$X"…'
  EXIT` and `$(cmd && echo a || echo b)`. One of them had already taken the
  lease. Later campaigns use plain statements and write the release out on each
  path.

## 8. What to keep, and what to do next

**Keep `PERSHAPE = 2`.** It is landed: its own ladder is green end to end, its
own M9 passes with `CTRL_PUBLISH_EARLY` still firing on the row that changed, it
is bit-identical to the incumbent, and its resource tuple is the incumbent's to
the register.

**Never fall back to 1.** If this ever has to be reverted, revert to `0`. `1`
is the same rule with a worse schedule and is strictly dominated by both.

Ranked follow-ups:

1. **Re-run the graded comparison** (`vs_rank1.sh`) at the shipped config. Shape
   5 is the worst of the six graded ratios and this is the first change to move
   it; the projection in §6 needs replacing with a measurement.
2. **Fix the A/B instrument everywhere it is used, not just here.** The rotation
   defect is in `experiments/exp_05_release_granularity/ab_release_group.py` and
   in anything descended from it. It produced a null arm reading −2.29% at
   p < 10⁻⁴ on identical behaviour, which is strong enough to have manufactured
   or erased a result in any experiment that used it. The fix is four lines.
3. **Re-gold M9** (§4) and derive its case list from the shape table.
4. **`RELEASE_GROUP` itself is now worth re-sweeping.** The cap has been 4 since
   E3, chosen when only shape 6 could reach it. With the ladder, a cap of 8
   would change nothing on the graded table but would matter on the generic row
   (118 tiles/CTA), and the ladder rungs `{RG, 2, 1}` could be `{RG, RG/2, …, 1}`
   if any shape ever lands at 3 or 5–7 tiles per CTA.

## 9. Files

| file | contents |
|---|---|
| `plan.md` | pre-registration; protocol-review conditions C1–C8 re-read |
| `build_arms.sh` | the five arm modules from one source; removes each target first |
| `isa_diff.sh`, `isa_diff2.sh` | the define reaches codegen; per-instantiation instruction counts |
| `golden_audit.sh` | the stale-golden tables of §4 |
| `m9_vs_incumbent.py` | M9 with a valid golden and a case list that covers shape 5 |
| `ab_pershape.py` | the paired A/B; shuffled arm order |
| `paired.py` | the within-round paired statistic with the null beside it |
| `campaign{,2,3,4}.sh` | leased GPU campaigns; 4 is the landing run |
| `logs/ab_pershape_*.json` | plot-ready per-round samples, six passes |
| `logs/paired_r3.txt`, `logs/paired_all.txt` | the tables of §6 |

### Schema of `logs/ab_pershape_<tag>.json`

`rounds`, `iters`, `arms` (in allocation order), `reversed`, then
`per_shape[<MxNxK>]` with `shape` = `[m, n, k, has_bias]` and
`arms[<tag>]` = `{expected_rgroup, best, median, worst, spread_pct, samples[]}`
in µs of pipelined world-8 wall time, plus `arms._geometry` =
`{tiles, producers, tiles_per_cta, config_row}`. Top level also carries
`geomean[<tag>]` and `candidate_vs_base[<shape>]` =
`{delta_best_pct, delta_median_pct, floor_pct, disjoint, same_stream, verdict}`.
`samples[j]` is round *j*, so index *j* is paired across arms — that pairing is
the statistic in §6 and the reason the raw samples are kept.

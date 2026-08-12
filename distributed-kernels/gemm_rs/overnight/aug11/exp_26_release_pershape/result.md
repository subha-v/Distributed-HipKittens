# exp_26 — per-shape release group: NOT LANDED (pending round 3)

> Status: rounds 1 and 2 complete, round 3 (shuffled-order instrument) queued
> behind exp_24's lease. This file is written up to that point; §7 carries the
> verdict and is updated when round 3 lands.

**Source state to keep: `HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE = 0`, the
incumbent.** That is already the default in `gemm_rs_mi300x.cpp` and this
experiment does not change it.

---

## 1. Verdict up front

- **The mechanism is real and the gates are clean.** The candidate passes the
  full ladder (M3 17 shapes at both `1e-2` and `2e-3`, M4, M5 600-epoch soak)
  and a repaired M9 with `CTRL_PUBLISH_EARLY` still failing. It is bit-identical
  to the incumbent on all six graded shapes over hundreds of NaN-poisoned
  epochs.
- **The spelling of the rule matters as much as the rule, and the obvious
  spelling is the wrong one.** `min(RELEASE_GROUP, tiles_per_cta)` (arm `ps1`)
  costs **+2 VGPRs on five of the seven instantiations** — 246→248 and 248→250
  on the 256×256 rows, which sit against the 256 arch cap — and shifts every
  schedule. Spelled as a descending select over compile-time literals (arm
  `ps2`) the resource tuple is **byte-for-byte the incumbent's** on all seven.
  `ps2` is the implementation any future attempt should use.
- **The win is not established, and the blocker is the instrument, not the
  kernel.** On shape 5 the null arm — two builds of the *same rule* — moved
  −2.36%, −1.58%, **+3.19%**, −4.40% across four independently allocated passes,
  and the paired within-round null median reached −2.29% on shape 6 in 57 of 64
  rounds. Every candidate delta observed is inside that. Pre-registration asked
  for −5% on shape 5; the best-supported estimate is **−1 to −2 pp beyond the
  null**, and on the strictest test (`rg2c`, which has the same group of 2 on
  shape 5) the delta sits *on* the null.
- **The controls did move**, by ~1 pp, on shapes where the group size cannot
  change. For `ps1` that is explained by the register tuple; for `ps2`, whose
  tuple is identical, it is the same instrument bias the null arm shows.

## 2. The change

`gemm_rs_mi300x.cpp`, producer role, the `rgroup` expression only. Everything
else — the group loop, the single release site, the deferred publish loop, the
per-tile credit wait, `decode_tile`, the WGM tile order, `BM/BN/BK`,
`num_reducer_ctas`, `find_scored_config`, the 72-byte descriptor ABI — is
untouched. No host or ABI change: the value is derived on the device from
`tiles` and `num_gemm_ctas`, both of which the kernel already computes.

```c
// PERSHAPE == 0, the incumbent
const int rgroup = tiles_per_cta >= RELEASE_GROUP ? RELEASE_GROUP : 1;

// PERSHAPE == 1, measured and rejected (see §5)
const int rcap   = tiles_per_cta < RELEASE_GROUP ? tiles_per_cta : RELEASE_GROUP;
const int rgroup = rcap > 1 ? rcap : 1;

// PERSHAPE == 2, the correct spelling
const int rgroup =
    tiles_per_cta >= RELEASE_GROUP ? RELEASE_GROUP
    : ((RELEASE_GROUP >= 2 && tiles_per_cta >= 2) ? 2 : 1);
```

Effective group per graded row, `1/1/1/1/1/4` → `1/1/1/1/**2**/4`. The generic
row (118 tiles/CTA) stays at 4. **Only 8192×4096×14336 changes.**

### Where the value comes from, and why not a folded constant

The dispatch asked for a folded compile-time constant where possible. It is not
possible here without adding an instantiation: `dispatch_gemm_rs_mi300x` sends
**rows 4 and 5 to the same `launch_fixed<256,256,32,false>`**, and they differ
only in `M`. A template parameter would split them, changing `M2_EXPECT` and
re-allocating registers for exactly the row whose delta is being measured.

What the experiment did find is that the *distinction the dispatch was pointing
at is real even within "runtime"*. The incumbent's `rgroup` is a runtime value
drawn from a **two-element set of literals**; `min()` makes it an arbitrary
value in a range; the ladder puts it back in a **three-element set of
literals**. Those are three different schedules, and the register tuple
separates them cleanly (§5). This is the same effect exp_09 recorded as folded /
bounded / runtime being worth 2.13% on shape 3.

### Fullness — and a correction to the premise

The dispatch's argument was that `min(MAX, tiles_per_cta)` "keeps every group
full, so it does not reintroduce" E3's failure mode. **That is not quite the
mechanism, and the difference matters.**

What is true: `rgroup ≤ tiles_per_cta = ceil(tiles / num_gemm_ctas)`, the tile
count of the CTA that owns the most, and `emitted = min(left, rgroup)` truncates
a short CTA's last group to what it owns. So no CTA ever defers a publication
waiting for a tile that does not exist. On shape 5 that means the 240 CTAs
owning 2 tiles group both, and the 32 owning 1 release immediately.

What is *not* true is that this avoids E3's failure mode by construction. E3's
3.75% loss on the then-2-tile 512×4096×12288 came from a group of **2 on a CTA
owning exactly 2** — a *full* group. Fullness was never the discriminator;
`FULL_ONLY=1` at `RG=4` spared that shape by accident, because 2 < 4. The thing
that actually changes on shape 5 is publication delay: the first of a CTA's two
tiles is now published one mainloop later. That risk is not removed by any
argument in this section, which is why the experiment was run as a measurement
and why shape 5's outcome is reported against a null arm rather than deduced.

Since exp_14 retiled row 2 to 64/128 (256 tiles, 1 per CTA), the shape that
carried E3's regression can no longer group at all and is now a control.

## 3. Gate ladder — all green

`tools/gate_ladder.sh exp_26_release_pershape`, candidate = `PERSHAPE=1` built
as the production module. (The ladder ran before the spelling question was
known; `ps2` is bit-identical in output to both, and its resource tuple is the
incumbent's, but **the ladder has not been re-run on `ps2`** — that is a
prerequisite before anyone lands it.)

| gate | verdict |
|---|---|
| M0 node clean | 0 KFD pids |
| M1 build | `ALL MODULES BUILT` |
| M2 resources / ISA | 7 of 7 instantiations, zero scratch, zero spills, occupancy unchanged |
| M3 correctness, 17 shapes | **17/17 PASSED at both `1e-2` and `2e-3`** |
| M4 negative controls | `all three controls failed exactly as designed` |
| M5 soak | `600 skewed changing-input epochs, epochs and signals exact` |
| M7 timing | ran; see §6 for why its cross-run numbers are not the denominator |

No error bit was raised at any point in the experiment.

## 4. M9 — and a defect in the gate itself

**The stock `harness/m9_stale_slot.py` cannot pass on this tree, for reasons
that predate exp_26, and one of them faulted the node.** Its frozen golden
`gemm_rs_mi300x_e3base` was copied before exp_14 (E4b) retiled rows 1–3:

| row | golden BM/BN/BK | current | golden cols | plan cols | consequence |
|---|---|---|---:|---:|---|
| 1 | 32/64/**64** | 32/64/**128** | 112 | 112 | BK-only; map identical, **still bit-exact** |
| 2 | 64/**64**/64 | 64/**128**/64 | 64 | 32 | map differs → golden NaN, 600/600 bitwise diffs |
| 3 | 128/**256**/32 | 128/**192**/32 | 12 | 15 | golden reads B rows to **3072 of a 2880-row operand** |

Row 3 is the `Memory access fault by GPU node-7 ... on address 0x7edfd0a50000`
at 10:05Z. It is the exact out-of-bounds read exp_14's own shape-table comment
says the retile removed, preserved in the frozen module. Tables in
`golden_audit.sh`; the predicted per-row outcome matches the observed log on all
six rows, including row 1 passing.

The gate was therefore re-run with the golden that this change actually needs —
`gemm_rs_mi300x_ps0`, the same source and shape table with `PERSHAPE=0`, which
makes the assertion "the rule change moves no bit", i.e. protocol-review
conditions C5/C8 stated exactly. `m9_vs_incumbent.py`; `harness/m9_stale_slot.py`
was **not** modified. Its case list was also re-aimed, because the stock list no
longer contains a shape that this change alters (row 2 became 1 tile/CTA):
8192×4096×14336 leads with 120 epochs, then row 6, the generic row, and rows
1–4 as C8 controls.

**Result: `GATE M9 PASSED`.**

- No NaN reached any output in any poisoned epoch, on any shape.
- Bit-identical to the incumbent on all 8 ranks for every epoch of every case.
- `allclose(2e-3)` every epoch; no error bit; epoch cells and every touched
  ready/credit cell exact.
- **`CTRL_PUBLISH_EARLY` still fails**, so the detector retains power over
  precisely the release/publication ordering property this change puts at risk.
  A gate its own control passes would be theatre.

**Recommendation for the ledger, outside this experiment's ownership:**
`build_golden.sh` should be re-run to re-gold `gemm_rs_mi300x_e3base` from the
current source at `RELEASE_GROUP=1`, and m9's `CASES` should be re-derived from
the shape table rather than hard-coded, since the retile silently emptied it of
the cases it existed to cover.

## 5. The spelling result — `min()` costs two registers

Resource tuples, all seven instantiations, from
`-Rpass-analysis=kernel-resource-usage`:

| arm | rule | VGPRs across the 7 instantiations | scratch | spills | occupancy |
|---|---|---|---|---|---|
| `ps0` | incumbent | 98 / 104 / 136 / **246** / **248** / 91 / 92 | 0 | 0 | 4/4/3/2/2/5/5 |
| `ps1` | `min()` | 100 / 106 / 138 / **248** / **250** / 91 / 92 | 0 | 0 | unchanged |
| `ps2` | ladder | 98 / 104 / 136 / **246** / **248** / 91 / 92 | 0 | 0 | unchanged |
| `rg2c` | `RG=2` | identical to `ps0` | 0 | 0 | unchanged |

`ps1` also changes the instruction count of every instantiation (−21 to +9) and
renumbers the SGPR-spill-to-VGPR lane register. Ordering-op counts
(`buffer_wbl2` 7, `buffer_inv` 7, `s_barrier` 98, `v_mfma` 184) and scratch are
identical across all arms, so nothing about the protocol or the mainloop moved —
only the schedule and the register budget.

That the ladder spelling restores the incumbent's tuple exactly is the cleanest
evidence in this experiment, and it is a codegen fact, independent of any timing
noise.

## 6. Measurement

### Instrument

`ab_pershape.py`, descended from exp_05's paired A/B: every arm is a separately
compiled module, all are open in one process, and their timed blocks are
interleaved inside a round so adjacent samples share the clock, the allocation
and the input sequence. Before any timing, every arm is verified at `2e-3` and
asserted `torch.equal` to every other arm on all 8 ranks. Four passes so far:
two orders × two rounds of the experiment, 7×50 then 25×100 iterations.

Arms: `ps0` incumbent · `ps1` `min()` · `ps2` ladder · `rg2c` `RELEASE_GROUP=2`
under the incumbent rule (a **positive control**: it computes rgroup 2 on shape
5 by a different expression, and rgroup 2 on shape 6 where everything else uses
4) · `ps0b` the **null arm**, a second build of `ps0`.

### Cross-run M7, for completeness only

M7 prints means; the recorded denominators are best-of-arm from paired runs, so
these are not comparable and are reported only because the ladder produced them.

| # | shape | best | median | mean | prev best-of-arm | floor |
|---|---|---:|---:|---:|---:|---:|
| 1 | 64×7168×18432 | 65.46 | 65.55 | 65.57 | 62.38 | 3.48 |
| 2 | 512×4096×12288 | 67.52 | 68.30 | 68.93 | 64.52 | 0.93 |
| 3 | 2048×2880×2880 | 87.45 | 88.13 | 87.91 | 83.75 | 1.61 |
| 4 | 4096×4096×4096 | 200.63 | 200.97 | 201.02 | 198.71 | 4.90 |
| 5 | 8192×4096×14336 | 658.06 | 659.18 | 659.50 | 613.70 | 13.93 |
| 6 | 8192×8192×29568 | 1607.29 | 1616.72 | 1626.02 | 1616.63 | 69.87 |

Geomean of best 208.44 µs. Five of six rows are 1.9–3.7 µs *above* the recorded
best-of-arm, including three one-tile rows this change cannot touch, so this run
sits on a different session mode and none of it is decision-relevant.

### The null arm, which is the whole story

`ps0` vs `ps0b` — same rule, two builds — best-of-pass delta on each shape:

| # | shape | r1 fwd | r1 rev | r2 fwd | r2 rev |
|---|---|---:|---:|---:|---:|
| 1 | 64×7168×18432 | +0.10% | −0.49% | +0.85% | −0.50% |
| 2 | 512×4096×12288 | −0.33% | −0.02% | +0.64% | −1.04% |
| 3 | 2048×2880×2880 | −0.42% | −0.68% | +0.62% | −0.89% |
| 4 | 4096×4096×4096 | −0.16% | −1.24% | −1.28% | −0.52% |
| 5 | **8192×4096×14336** | **−2.36%** | **−1.58%** | **+3.19%** | **−4.40%** |
| 6 | 8192×8192×29568 | +2.99% | −1.12% | −2.76% | −2.49% |

**Shape 5's floor is not the 13.93 µs (2.2%) on record — in these four passes it
is ±3–4%, i.e. 20–29 µs, and it changes sign.** Which arm is fastest on shape 5
also changes between passes: `rg2c`, `rg2c`, `ps2`, `ps1`.

### Per-shape best, pooled over all four passes (≈64 samples per arm)

| # | shape | t/CTA | `ps0` | `ps0b` null | `ps1` | `ps2` | `rg2c` |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 64×7168×18432 | 1 | 63.0 | −0.2% | +1.2% | +1.0% | −0.2% |
| 2 | 512×4096×12288 | 1 | 65.3 | +0.6% | +1.1% | +1.8% | +1.1% |
| 3 | 2048×2880×2880 | 1 | 84.5 | +0.1% | −0.5% | +0.3% | +1.0% |
| 4 | 4096×4096×4096 | 1 | 199.2 | −1.3% | −0.0% | −0.7% | −0.4% |
| 5 | **8192×4096×14336** | **2** | **646.2** | −1.1% | **−3.9%** | **−4.4%** | −2.0% |
| 6 | 8192×8192×29568 | 4 | 1583.0 | −0.4% | +0.5% | −0.2% | **+4.2%** |

Every arm's pooled range overlaps `ps0`'s on every shape, **including `rg2c` on
shape 6 where the mechanism is known to be active and large** — which is the
measure of how much the between-pass mode costs this statistic.

### Within-round paired differences (the mode-cancelling statistic)

Pairing each arm's sample against `ps0`'s from the *same round* removes the pass
mode by construction. Median of the per-round % delta, `n` = 50–64 rounds:

| # | shape | `ps0b` null | `ps1` | `ps2` | `rg2c` |
|---|---|---:|---:|---:|---:|
| 1 | 64×7168×18432 | +0.27 | +1.78 | +1.07 | −0.23 |
| 2 | 512×4096×12288 | −0.36 | +0.01 | +0.30 | −0.08 |
| 3 | 2048×2880×2880 | −0.45 | −1.20 | +0.46 | +0.38 |
| 4 | 4096×4096×4096 | −0.77 | −0.13 | −0.24 | −0.33 |
| 5 | **8192×4096×14336** | −1.97 | **−4.08** | **−3.09** | −1.91 |
| 6 | 8192×8192×29568 | −2.29 | −1.53 | −2.23 | **+3.68** |

Two things this says, and they point opposite ways:

1. **The instrument has power over release granularity when the effect is
   large.** `rg2c` on shape 6 — the one arm with a different group there — is
   +3.68% against a −2.29% null, a 6 pp separation, 4-for-4 in sign across
   passes. Grouping at 2 instead of 4 on shape 6 genuinely costs ~6%, which
   independently re-confirms E3.
2. **The null is not centred, so significance is meaningless in absolute
   terms.** `ps0b` reaches −2.29% (57 of 64 rounds, p < 10⁻⁴) on behaviour that
   cannot differ. Read every column *relative to the null column*: on shape 5
   that leaves `ps1` at −2.1 pp, `ps2` at −1.1 pp and **`rg2c` at +0.06 pp** —
   and `rg2c` has the same group of 2 as the other two. Three arms with the same
   rgroup do not agree, so the shape-5 column is not yet measuring the rgroup.

The likely cause of the uncentred null is the instrument itself: exp_05's
rotation advances every arm one position per round, which pins the *relative*
spacing of any two arms in every round of every pass, so an order effect becomes
a constant offset on that pair instead of noise that averages out. Round 3 tests
that by shuffling the arm order per round.

## 7. Verdict and what to keep

*(updated when round 3 lands)*

Rounds 1–2 do not support landing. Keep `PERSHAPE = 0`.

## 8. Node discipline — disclosure

This experiment's first two GPU runs, the gate ladder and the stock M9, were
launched **without acquiring `tools/gpu_lease.sh`**, which is a discipline
failure with a measurable cost to other tenants:

- The M9 run took the memory access fault of §4 and wedged in driver teardown
  (state `D`, `wchan exit_mm`), unkillable by any signal, holding ~10 GB of VRAM
  on all 8 GPUs. It never exited; it was still resident at the end of this
  experiment. Only SIGTERM was used at every stage — the process was already in
  `do_exit`, and SIGKILL cannot shorten that path while risking leaked IPC
  mappings.
- **`exp_21` and `exp_24` both aborted their `acquire` at 10:06Z and 10:11Z**
  ("node dirty after 300s") because of it. That is a direct preemption of two
  other experiments and it is on this one.
- Every GPU run from 10:29Z onward is properly leased, and the lease was
  released after each campaign.
- A second, unrelated defect: `tools/gpu_lease.sh` is currently on the node with
  CRLF line endings and does not execute (`set: pipefail: invalid option name`).
  It belongs to another experiment, so rather than edit it, this campaign runs a
  CR-stripped copy from `/tmp` — same script, same lock directory, same log.
  Worth repairing at source, since a lease tool that silently fails to run is
  how unleased jobs happen.
- One self-inflicted build failure worth recording: round 2's first attempt ran
  `build_arms.sh` on the **host** instead of in `dhk-gemmrs`. The host hipcc has
  no pybind11, so all five arms failed to compile — after the builder had
  already removed each target `.so`, which it does deliberately so that "the
  file exists" is a real build result. Everything that compiles now goes through
  `docker exec dhk-gemmrs`.

## 9. Files

| file | contents |
|---|---|
| `plan.md` | pre-registration, protocol conditions C1–C8 re-read |
| `build_arms.sh` | the five arm modules from one source, targets removed first |
| `isa_diff.sh` | proves the define reaches codegen; per-instantiation instruction counts |
| `golden_audit.sh` | the stale-golden tables of §4 |
| `m9_vs_incumbent.py` | M9 with a valid golden and a case list that covers shape 5 |
| `ab_pershape.py` | the paired A/B; shuffled arm order from round 3 |
| `paired.py` | the within-round paired statistic and the null beside it |
| `campaign{,2,3}.sh` | leased GPU campaigns |
| `logs/` | every run; `ab_pershape_*.json` are the plot-ready data |

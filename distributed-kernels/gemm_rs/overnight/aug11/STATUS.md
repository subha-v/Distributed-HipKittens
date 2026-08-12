# aug11 GEMM-RS — STATUS (the two-minute morning read)

Updated after every experiment. Newest first.

## Where the kernel stands

| denominator | value | note |
|---|---|---|
| **frozen rank-1, same-run graded** | **1.098×** behind | was 1.784× when rank-1 first ran |
| reference GEMM+RCCL, same-run graded | **beaten** | 427.39 vs 438.15 µs best, earlier config |
| our harness, pipelined geomean | **~207 µs** (means) | from 285.02 at session start, **−27%** |

Per-shape graded ratio vs rank-1 (ours/rank-1, lower is better):
**0.850** / 1.072 / 1.102 / 1.120 / 1.286 / 1.208 — **shape 1 is now a win**;
shapes 5 and 6 carry the remaining gap.

Best-of-arm pipelined vector:
`62.38 / 64.52 / 83.75 / 198.71 / 613.70 / 1616.63` µs.

## Numbering (resolved by the charter update at 29179f8e)

- `overnight/experiments/exp_NN_*` — the optimization sessions (exp_01…exp_14).
- `overnight/aug11/exp_NN_*` — **tonight's paper-figure queue, exp_20…exp_25.**

The collision I flagged (both trees had used exp_13/exp_14) is resolved: the
figure queue was renumbered to exp_20+.

## Corrections to the charter's Phase 2 premises

Phase 2 lists the next optimization targets. **Two of its premises were
overtaken by tonight's work and should not be re-run as written:**

1. **"E1(a) AGPR accumulators should also clear the Gate-M2 spills on the
   256/256/32 rows."** Both halves are now false. AGPR accumulators are
   *impossible* here: `__launch_bounds__(512,1)` caps the wave at 256 **unified**
   registers, so enabling AGPRs re-partitions the same file rather than adding
   registers — and the ISA showed the spills were never in the k-loop anyway.
   Separately, **M2 now reports 7/7 instantiations with zero AGPRs, zero
   scratch and ZERO VGPR spills**, so there are no spills left to clear.
2. **"The per-call host tax is most of the remaining graded-protocol gap."**
   Measured and closed: of ~103 µs of non-device cost, **~92 µs is harness
   machinery every submission pays** (the evaluator's own `synchronize +
   barrier` is 57-79 µs with an *empty* timed region). Our own share is ~11 µs,
   within ~6 µs of the floor for issuing any HIP kernel from Python.

What is genuinely open for Phase 2: the mainloop's **non-MFMA** time. Shape 6
spends ~4700 cycles per k-iteration against ~2050 of MFMA, and rows 4/5/6 are
pinned at the 64 KB LDS cap so their `waves × k_iters` (16 / 112 / 464) cannot
be reduced by retiling. That is ~640 µs of non-MFMA mainloop on shape 6 —
on exactly the two shapes that carry the whole remaining graded gap.

## The aug11 charter's stated baseline is two experiments stale

It reads "post-E3 best, pipelined geomean ~225.6 µs; gap to rank-1 1.167×".
Since then two more landed:

| exp | change | effect |
|---|---|---|
| `experiments/exp_13_cta_split` | E4 re-swept: shape 1 `NR=56`, shape 6 `NR=48` | paired −3.6%; gap → 1.137× |
| `experiments/exp_14_tile_waves` | tile table by wave count: rows 1/2/3 → `32/64/128`, `64/128/64`, `128/192/32` | paired −5.9%; gap → **1.098×** |

**All figure work must use the current config**, not the charter's numbers.

## Figure queue

| # | experiment | paper figure | status |
|---|---|---|---|
| exp_20 | bottleneck attribution refresh | Q3 | **running** |
| exp_21 | saturation vs CTA count (NanoFlow Fig 7 analog) | Fig 2 / Q4 | queued |
| exp_22 | per-layer resource timeline (NanoFlow v2 Fig 10 analog) | Fig 3 / Q4 | queued |
| exp_23 | knob waterfall — **the money figure** | Fig 4 / Q1 | queued |
| exp_24 | external ladders refresh | Q6 | queued |
| exp_25 | per-shape sensitivity readout | Q5 | queued |

Then **Phase 2**: the optimization loop resumes against the refreshed profile.

## What the kernel already demonstrates for the paper's central claim

The claim is that overlap is decided by **scheduling decisions**, not by
dedicating CTAs to communication. This kernel is an unusually clean instance:

- The two largest wins were a **task-order** change (`WGM`, −27.9% on shape 6,
  effective xGMI links 2.02 → 7.53 of 8) and a **signal-coarsening** change
  (`RELEASE_GROUP=4`, −4.0% graded on shape 6). Neither moved a single byte
  differently; the fabric carried 117.48 MB against 117.44 MB of useful payload
  before and after, 99.9% of it in full 64 B transactions.
- **Dedicating more CTAs to communication was measured and rejected**: the
  producer/consumer split buys *rounds*, not CTAs, so `NR=16` bought zero GEMM
  waves and cost +3.6%. A dual-role/work-stealing reducer was killed by the
  same arithmetic — it removes a producer wave on none of the six shapes.
- The reducer CTAs exist only because the reduction must run on the owner.

That is three independent data points for "scheduling, not CTA dedication",
already measured on this architecture.

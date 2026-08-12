# exp_24 addendum — the decisive paired test: `ours` (PERSHAPE=2) vs `ours_prev` (PERSHAPE=0)

**Verdict: INDETERMINATE under the pre-registered rule, and exp_26's −6.56% win
is withdrawn.** The two allocation orders disagree *in sign* at p≈1e-101 each.
The one estimate whose bias I can measure — the order-balanced mean — puts the
shipped rule **+1.37% pipelined / +1.12% graded SLOWER** on shape 5, against a
worst control residual of 0.64% / 0.36%. Nothing here supports a 6.56% win.

**The dominant term in this measurement is not the kernel. It is which arm gets
allocated first.** That is the finding, and it is bigger than the effect the
experiment was built to measure.

Data: `ab_prev.json`. Full scoring output: `ab_prev_report.txt`.
Driver: `ab_prev_mp.py`. Arms: `build_ab_arms.sh`. Runner: `go_ab_prev.sh`.

## Why this run existed

exp_26 measured shape 5 at −6.56% pipelined, 80/80 paired rounds. exp_24's
re-measure, comparing two ladder *runs* six hours apart, read +2.87%. A
9.5-point disagreement including a sign flip cannot be settled by preferring an
instrument, so both binaries went into ONE pool and were paired within-round —
the design both measurements agree is the strongest available.

## Setup, and the three verifications that had to pass first

Three modules, one source tree, one differing `-D`, built minutes before the
measurement (`logs/ab_arm_shas.txt`):

| arm | module | PERSHAPE | sha256 (16) | role |
|---|---|---:|---|---|
| `ours` | `gemm_rs_mi300x_ab2` | 2 | `7a4f8c5689de0e66` | the shipped rule |
| `ours_prev` | `gemm_rs_mi300x_ab0` | 0 | `3d1145fc8a549eea` | the pre-exp_26 rule |
| `ours_null` | `gemm_rs_mi300x_ab2b` | 2 | `980900cb67ad0b34` | second build of `ours` |

1. **`ab0` != `ab2` byte-wise.** Their module names are the same length, so the
   difference is attributable to the macro. Had they matched, the arms would be
   the same computation and the test void; the build script exits 1 on that.
2. **`rgroup` from each arm's own rule, at the live plan's geometry.** Shape 5 is
   512 tiles / 272 producers = 2 per CTA → `ours` 2, `ours_prev` 1, **distinct**.
   Shapes 1–4 are 1 per CTA and shape 6 is 4, where the two rules agree — so on
   five of six shapes the arms are the same computation. Recorded per run.
3. **`torch.equal(ours, ours_prev)` on all 8 ranks, all 12 runs, all-reduced with
   MIN so one bad rank stops all eight.** True everywhere, as exp_26 found.
   Correctness re-checked after every timed block at `1e-2` and `2e-3`.

`ours_null` is a separate `.so` here, not the same file under a second Python
name as in the ladder. `ours_prev` is necessarily a separate `.so`, so the null
must be too, or the null would be structurally easier than the treatment — one
dlopen, shared C++ globals, one IPC exchange — and would understate the floor.
(It also cannot be the same file: an extension module only exports
`PyInit_<its own name>`, which is how the first attempt failed.)

Every timing region, the warmup, the input generation and the oracle are
imported from `ladder_mp` rather than reimplemented, so this is literally
instrument A with a sixth arm. 42 rounds × (25 graded iters, 15×5 pipelined)
per shape per allocation order, 6 shapes, 2 orders, 8 ranks = 336 paired rounds
per cell, exceeding exp_26's 40×100.

## Ordering: balanced by construction, not in expectation

Three arms means 3! = 6 permutations, so the fix can be exact rather than
statistical: **each block is all six permutations, the block's internal sequence
shuffled, seven blocks = 42 rounds.** Within every block each ordered pair
(j, k) has j before k in exactly 3 of 6 permutations and every relative offset
appears equally often. That is the Latin-square property, complete. **Residual
pinning within a round: none, by construction** — which is the improvement over
the 5-rep shuffle, where 2 of 6 shapes still had an accidentally pinned pair.

Seeded from the shape index alone, never rank or wall clock: all 8 ranks must
walk identical orders or the collective arms deadlock.

**This balancing cannot touch the confound that actually dominated.** Arm
*allocation* happens once per process, before round 1, so no within-round
permutation can decorrelate it. That is why both allocation orders were run —
and it is the only reason this run produced an answer at all.

## The result

Sign convention: `(ours − ours_prev)/ours_prev`, so **negative = the shipped
PERSHAPE=2 rule is faster**, matching exp_26's "−6.56%".

### Shape 5 (8192×4096×14336), the only row where the arms differ

| protocol | allocation order | paired median | wins | p | null |
|---|---|---:|---:|---:|---:|
| pipelined | `ours` first | **−3.11%** | 336/336 | 1.4e-101 | −0.93% |
| pipelined | `ours` last | **+5.85%** | 0/336 | 1.4e-101 | +1.52% |
| graded | `ours` first | **−2.92%** | 336/336 | 1.4e-101 | −1.11% |
| graded | `ours` last | **+5.17%** | 0/336 | 1.4e-101 | +1.43% |

Unanimous in both directions. 336/336 one way, 0/336 the other, from the same
two binaries in the same pool with the same balanced ordering. **The only thing
that changed is which arm was constructed first.** A p of 1e-101 that reverses
when you permute the setup is not a measurement of the kernel; it is a
measurement of the instrument, and its perfect internal consistency is the
signature — an artifact is systematic within a configuration and reverses
between them, whereas noise would merely widen.

The null arm carries the same signature at smaller amplitude (−0.93% → +1.52%),
which is the direct evidence that position, not code, is doing the work:
`ours` and `ours_null` are the same computation.

### The order-balanced estimate, and its measured residual

Averaging the two orders cancels a position effect to first order. Its residual
is not assumed — it is **measured on the five shapes where the two arms compile
to the same `rgroup`, where the truth is exactly 0**:

| # | shape | proto | `ours` first | `ours` last | balanced | arms distinct |
|---:|---|---|---:|---:|---:|:---:|
| 1 | 64×7168×18432 | pipelined | −0.48 | +0.66 | +0.09 | no |
| 2 | 512×4096×12288 | pipelined | +0.68 | −0.84 | −0.08 | no |
| 3 | 2048×2880×2880 | pipelined | −0.12 | +0.37 | +0.12 | no |
| 4 | 4096×4096×4096 | pipelined | −0.04 | −0.61 | −0.33 | no |
| 5 | **8192×4096×14336** | pipelined | −3.11 | +5.85 | **+1.37** | **yes** |
| 6 | 8192×8192×29568 | pipelined | −1.67 | +0.39 | −0.64 | no |

Graded is the same story: controls +0.03, +0.15, −0.30, −0.34, −0.36; shape 5
**+1.12%**.

- pipelined: control residual mean −0.17%, worst |0.64|%; shape 5 **+1.37%**, above it.
- graded: control residual mean −0.16%, worst |0.36|%; shape 5 **+1.12%**, above it.

So after balancing the confounder, shape 5 lands ~2–3× the worst residual on the
**slower** side, in both protocols independently. Small, consistent, and the
opposite sign from the claim.

### The false positive, reproduced on identical code

Row 6 is the exhibit. On 8192×8192×29568 both arms compile to `rgroup` 4 — the
same computation — and the `ours`-first configuration reports **−1.67% with a
null of −0.14%**, i.e. a "win" that clears its own null by more than 10×, on two
binaries that differ in nothing an instruction can see. That is a false positive
of exactly exp_26's shape and size class, manufactured on demand. exp_26's
−3.11%-at-best configuration is the same configuration that produces it.

## What this means for exp_26, and for the ladder

**exp_26's −6.56% does not reproduce in any configuration measured here.** The
most favourable arrangement gives −3.11%, and that arrangement demonstrably
fabricates −1.67% on identical code. exp_26 held five arms open in ONE process
driving 8 devices; this is 8 processes under `torch.distributed`, closer to the
evaluator's topology. Either the effect is topology-specific, or — far more
likely given row 6 — exp_26 measured its own allocation order.

**The ladder's headline ratios need a disclosed caveat.** `ARM_SPECS` constructs
in a fixed order — `ours`, `ours_null`, `reference`, `rank1`, `harness_floor` —
so **our arm has been allocated first in every ladder run all night**, and the
null contrast here says position 1 vs position 3 is worth 0.9–1.5% on shape 5 in
our favour. That is inside the ±2% ratio floor and does not overturn 1.1165×
graded / 1.1111× pipelined, but it is systematic, it is in our favour, and it
must be disclosed until the ladder is re-run with allocation order rotated
across launches. This supersedes nothing in `result_remeasure.md` except its
silence on the point.

## The corrected null floors, all six shapes, both protocols

`ours` vs `ours_null` — identical code — per-round paired median magnitude, 336
rounds per cell, under the complete-permutation ordering. These supersede the
floors quoted all night, which the cyclic-order defect inflated:

| shape | graded, `ours` 1st | graded, `ours` 3rd | pipelined, `ours` 1st | pipelined, `ours` 3rd |
|---|---:|---:|---:|---:|
| 64×7168×18432 | 0.09% | 0.12% | 0.46% | 0.18% |
| 512×4096×12288 | 0.18% | 0.31% | 0.59% | 0.22% |
| 2048×2880×2880 | 0.16% | 0.13% | 0.24% | 0.46% |
| 4096×4096×4096 | 0.42% | 0.76% | 0.56% | 0.47% |
| 8192×4096×14336 | 1.11% | 1.43% | 0.93% | 1.52% |
| 8192×8192×29568 | 0.02% | 0.41% | 0.14% | 1.45% |

Mean 0.51% graded, 0.60% pipelined, against the 1.62% graded mean the cyclic
scheme produced. **But the floor is not a single number: it depends on the
positions the two arms occupy**, which is the whole lesson. A floor quoted
without stating the arms' allocation positions is incomplete.

## Recommendation: revert `PERSHAPE` to 0

Formally the pre-registered outcome is INDETERMINATE — the orders disagree in
sign — and the pre-registration says make no change on that. I am reporting it
as such and I have **left the tree at PERSHAPE=2** rather than reverting
unilaterally. But the recommendation is to revert, for reasons that are evidence,
not preference:

1. The claimed win does not reproduce anywhere. Best case −3.11%, in the exact
   configuration that manufactures −1.67% on identical code.
2. The only estimate with a measured residual says **+1.1 to +1.4% slower**,
   above that residual, in both protocols independently.
3. Burden of proof. `PERSHAPE=2`'s sole support was exp_26's −6.56%; that number
   is withdrawn. An optimization whose evidence has been withdrawn should not
   remain the shipped default, and `PERSHAPE=0` is the previously-gated
   incumbent, so reverting carries no gate risk.

Never to `PERSHAPE=1`: same rule, worse schedule, dominated by both.

## What would settle it, if anyone wants to keep 2

**One arm per process.** Build each binary as the *only* arm in its own launch,
so both get identical first-position placement, and interleave launches A/B/A/B
to control drift, with a third launch of the null binary calibrating the
launch-to-launch floor. That removes co-residency entirely rather than balancing
it. It costs one process launch per arm per shape and it is the only design left
whose confounders I can name in advance.

A cheaper partial: rotate allocation order across many launches (6 orders for 3
arms) and take the balanced mean, with the identical-code shapes calibrating the
residual as they did here. That is this run with more launches, and it would
tighten +1.37% to a real interval.

## Mechanism, stated as a hypothesis and not as a finding

Each arm allocates its own HIP IPC symmetric heap at construction, so
construction order fixes the heaps' relative placement. Producers emit 16-byte
scattered peer packets whose addresses derive from the heap base, and shape 5 is
the release-granularity-sensitive row, so a placement-dependent change in egress
efficiency is plausible. Shape 5 also has the largest null (0.93–1.52%) and
shape 6 the second largest. **I did not test this** — it is the next
experiment's hypothesis, not this one's conclusion.

## Process notes

- Ran under `gpu_lease.sh acquire exp24ab`, released on trap; one 8-GPU job at a
  time; clocks pinned 1900 before timing; `setsid` + `timeout` throughout.
- `push_scoped.ps1` only, never `push.ps1`, and nothing pushed while a run was
  live. Node results pulled by `scp` before any push.
- **Two guards earned during this run.** The build script now checks rc *and*
  the artifact's existence, because a CRLF-truncated script printed a plausible
  "sha differs" line earlier tonight for a module it never wrote. And the worker
  now `os._exit(1)` on a caught exception: the smoke run died in all 8 ranks on
  an import error and still printed `exit codes: [0]*8`, because catching the
  exception to write a diagnostic JSON also swallowed the process status — the
  runner's all-zero check would have called a total failure a pass. The smoke
  test that caught it cost 4 minutes and saved a 25-minute campaign.
- One DNS blip mid-session (`Could not resolve hostname`, `nslookup` fine);
  `ipconfig /flushdns` cleared it. Not a node problem.

## Schema, `ab_prev.json`

`outcome` (string), `sign_convention`, `primary_protocol`. `paired[]`: one entry
per (run, protocol) with `median_pct`, `mean_pct`, `wins`, `rounds`,
`sign_test_p` (exact two-sided binomial), `null_median_pct`, `resolved_vs_null`,
`verdict`, `shape_label`, `shape_index`, `alloc_order`, `rgroup`,
`arms_distinct`. `order_balanced[]`: per (shape, protocol) `fwd_pct`, `rev_pct`,
`balanced_pct`, `arms_distinct`. `corrected_null_floors`: per shape and
allocation order, `graded_pct` and `pipelined_pct`. Raw per-rank per-round
samples remain node-side under `raw/ab_prev/` (96 files).

# aug11 overnight — status and plan

Live document. Updated as each experiment lands. The append-only ledger is
`../experiments/LESSONS.md`; per-experiment detail is in `exp_N_*/result.md`.

## Where we start

| arm | µs (campaign, 5-rotation median rank-max p50) | vs `production` |
|---|---:|---:|
| `production` | 7,715.6 / 7,720.0 | 1.000 |
| `pf6gm_mega` (homogeneous megakernel) | 6,911.4 / 6,902.4 | 0.895 |
| **`mps_mega` mode 12, `C=16 g=33 flush_rows=16`** | **6,685.5 / 6,683.1** | **0.866** |

Target: **0.80× ≈ 6,172 µs**, i.e. **−513 µs** from the ratchet.

## RATCHET MOVED (exp_24, `9530382a`)

**`C=16, g=353, mode=12, flush_rows=16` = 6,568.0 ± 4.6 µs = 0.8522× production**
(0.9487× `pf6gm_mega`), across **three independent 5-rotation campaigns** with
the full gate ladder green in all 15 rotations. `g=353` composes Mechanism A
(dead `part` zero deleted) with throttle depth 4.

**−117.5 µs and +1.43 points of margin** on exp_21's 6,685.5 / 0.8665.
Remaining to 0.80×: **−396 µs.**

Resource gate *improved*: SGPR/VGPR/AGPR/LDS unchanged, **scratch 144 → 128
B/lane**, MFMA census 180, `flat_atomic_pk_add_bf16` 282, zero scratch ops
inside either MFMA span.

Two corrections to what was believed earlier tonight:
- **Mechanism A's 448 MiB was fully exposed on the M3→M4 critical path**, not
  absorbed by the LLC. The pre-registered null (that CTA 0's serial `tile_desc`
  build dominates the plan phase) is **refuted**. And the LLC-eviction story is
  dead exactly as pre-registered: M6 moved −14.7 ± 11.3 µs (t = 1.3), because
  W13 (939 MB) and W2 (469 MB) each exceed the 256 MB Infinity Cache anyway.
- **Throttle depth 4 beats 8** by −50.9 µs at campaign resolution — invisible on
  the M7 stamp (−83 ± 63 µs) but ~9σ across campaigns. Above 8 the axis is a
  cliff: 16 and 32 return M7 to the *unthrottled* cost (+462 / +370 µs). Depth 2
  is unmeasured and the `g` `0x300` selector has no room left.

Two further exp_24 findings worth carrying:
- **The remote-atomic epilogue is not instruction-issue bound.** Folding
  `slot_off` into the peer table removes 3 instructions from each of ~350 M
  remote atomic ops per rank per epoch and measured a **campaign-resolution
  null**. It survives only as a register-pressure *enabler* — which is what
  paid for Mechanism B's spill fix.
- **The `g` field silently overflowed into `mode` for any `g > 0xFF` and still
  validated.** `g = 0x121` set `mode |= 1` and decoded back as `g = 0x21`. Now
  masked with `g`'s high byte at packed bits [34:42), verified bit-identical
  over all 3,584 combinations. **Seventh mechanism selector packed into a
  reinterpreted config field, and the first to draw blood.**

Screens (1 process / 1 warmup / 1 timed, ~60–90 s) reproduce the ratchet at
6,703–6,728 µs and read ~1.2 points optimistic against the campaign. Measured
tonight over six repeats of the identical config: `ratio_vs_prod` σ = **0.52 %**,
`ratio_vs_pf6gm` σ = **1.34 %**. **Screens rank against `production` only** —
`pf6gm_mega` is too noisy at one warmup iteration to be a denominator.

## The budget, and why the queue is ordered the way it is

Approximate phase costs at the ratchet (rank-max stamps, screens):

| phase | µs | note |
|---|---:|---|
| dispatch M0–M2 | ~1,193 | ~73 % own-work, not peer wait (exp_10 closed this axis) |
| plan M3–M5 | ~415 | **contains 448 MiB of provably dead stores** → exp_24 |
| **M6 (GEMM-1)** | **~2,588** | **K-loop stalls ~84 % of its cycles** → exp_26/27/28 |
| **M7 (GEMM-2)** | **~2,660** | +1,074 over the homogeneous baseline → exp_25 |
| combine M8/M9 | ~430 | already 3× better than homogeneous → exp_29 |

M6 and M7 together are 5,248 µs = **78 % of the kernel**. Everything below
targets one of those two, or deletes work outright.

### The three findings that reset the queue tonight

1. **Mode 12 never touches `part`** — not M7, not M8, not the service pool —
   yet M5 still zero-fills and scale-transposes all 32,768 × 7,168 × 2 B =
   **448 MiB of it every epoch**, inside a plan phase that only costs ~415 µs
   total, and streams it through a 256 MB Infinity Cache that M6 is about to
   want. (`CONTEXT/mode12_protocol_map.md` §8, `k0pf6gm_device_tile_mps.hip:1284-1288`.)
2. **There is no grid barrier between M6 and M7.** The dependency is already
   per-32-block via `a2_done[b]` counting to 8, and finer still: M6 chunk `g`
   produces exactly K128-group pair `(2g, 2g+1)` of M7's K-loop. What
   serialises the two phases is **CTA program order** — every CTA drains its
   whole M6 stripe first. `A2q` is capacity-sized and single-buffered so
   overlap needs no credit protocol, and both phases' LDS blocks are already
   summed into the single 155,428 B allocation, so interleaving costs **zero
   extra LDS**. (`CONTEXT/m6_m7_structure.md` §4.)
3. **M6's K-loop stalls ~84 % of its cycles** (~9,800 cycles per K-step against
   1,536 cycles of MFMA issue at G=3) with a software pipeline exactly **one
   K-step deep** — and its four `sched_group_barrier` hints are **hardcoded for
   `kGM = 1`** while we ship `kGM = 3`, so the scheduler is steered for a third
   of the MFMA, a third of the DS reads and a third of the DS writes.
   (`CONTEXT/m6_m7_structure.md` §5.3.)

### One prior belief retracted

**"M6 is CTA-insensitive (removing 24 % of CTAs costs 0.5 %)" was never
measured.** The reservation that produced that number happens at M6.9, *after*
the phase-1 body returns; phase 1's task loop is a hardcoded `task += kCTAs`
with `kCTAs = 256` and there is no role predicate between the M5 barrier and
the phase-1 call. All 256 CTAs have run all of M6 in every configuration ever
run. `exp_05` said so explicitly and used the invariance as its sanity check;
later documents re-read that control as a measurement. **No design may assume
M6 has idle CTAs.**

## The queue

| # | experiment | targets | mechanism | risk |
|---|---|---|---|---|
| exp_24 | dead plan work + throttle-depth sweep | plan, M7 | delete the 448 MiB `part` zero-fill; sweep the epilogue's outstanding-RMW cap (8 was the first value ever tried and was worth ~500 µs) | low |
| exp_26 | phase-1 `sched_group_barrier` G-scaling | M6 | parameterize four hint counts by `kGM`; every expression evaluates to the existing literal at `kGM = 1`, so it is a parameterization, not a retune | low |
| exp_27 | `ascale_lds` prologue gather | M6 | 5,376 distinct cache lines fetched to deliver 21,504 B — 16× line amplification, 344 KB per task, sitting fully exposed between two `__syncthreads()`; ≈ 200 µs | medium |
| exp_25 | **M6/M7 CTA role split** | M6+M7 | the headline mechanism: partition the grid so some CTAs issue GEMM-1 MFMA while *different* CTAs run GEMM-2 and its remote-accumulate epilogue | high |
| exp_29 | **pipelined combine** | combine | the pool reduces and writes `out` rows as they become ready during M7, so M8 is empty | high |
| exp_28 | M6 task-order swizzle | M6 | 2,840 tasks share only 256 unique weight slices (10.9× reuse available); grid-stride assignment puts consecutive tasks 32 tiles apart | medium |

Ordering rationale: exp_24 and exp_26 are deletions and parameterizations —
highest information per unit of build risk, and both are single-variable A/B by
construction. exp_25 and exp_29 are the two on-mandate role-specialization
mechanisms and carry the largest prizes; both get a written protocol review
before their first GPU run.

## exp_25 design verdict — the M6/M7 split is gated on one unmeasured number

The design (`exp_25_m6m7_split/design.md`) reframed the mechanism honestly and
the reframing is worth carrying forward:

**A static role split cannot win by overlapping work.** Work is conserved, and
today M6 already gets all 256 CTAs while M7 gets 240. A split can only give M6
about 128. Under linear scaling the best balanced split is `(W6+W7)/240 =
5,421 µs` against today's `W6/256 + W7/240 = 5,248 µs` — a **173 µs loss**. Any
claimed win must name a term outside that model, and there are exactly three:

- **W1 — throttling the remote-RMW rate by cutting injectors. This is the whole
  case.** M7 in mode 12 costs 2,660 µs against 1,684 µs for the same GEMM
  without the remote-accumulate epilogue, so the epilogue surcharge is
  **976 µs**, and exp_21 proved it is rate-shaped. The split is a second,
  orthogonal throttle on the same axis: total outstanding remote RMWs =
  injectors × depth; exp_21 capped the depth, the split cuts injectors to
  0.47×. And M6 touches no fabric at all, so **xGMI sits 100 % idle for
  2,588 µs of every epoch** — that is the one genuine complementarity.
- **W2 — resource complementarity ≈ 0, plausibly −300 µs.** Both phases are the
  same fp8 K-loop with the same L2/LLC limiter (161.7 vs 158.1 FLOP/B, 17.1 %
  vs 13.7 % MFMA duty). Do not claim compute/bandwidth complementarity; the
  numbers do not support it.
- **W3 — tail elimination nets to ~zero** once the split's own start bubble is
  counted.

Central prediction **6,312 µs = 0.818×**, band 5,877–6,969. The band is
dominated by one coefficient: **M6's CTA scaling, which has never been
measured** (see the retraction above).

### F1 — the cheap gate that must run before the expensive build

Measure `T6(128) / T6(256)`.

| result | action |
|---|---|
| **≥ 1.90** | **stop — do not build the overlap arm.** Work-conservation loss cancels the whole prize. |
| ≤ 1.80 | build |

F1 needs only the `N2GM_P1_TASK_START` / `_STRIDE` hooks in the vendored
phase-1 body (still the donor's hardcoded `blockIdx.x` / `kCTAs` by default), so
it rides exp_26's file rather than creating a second writer. **`a6` is a
property of the tree, not the hardware** — exp_24 and exp_26 both move it, so F1
runs on the winning tree, not first in wall-clock order.

### The correctness landmine, recorded before anyone builds

**`a2_done`'s poll is vacuous today and this experiment makes it live for the
first time — and the campaign structurally cannot detect it failing.** M6's
payload release is a tid-0-only agent fence behind a bare `__syncthreads()`,
ordering 255 other threads' plain `uint4` stores that tid 0 never touched,
across eight non-coherent per-XCD L2s. Because the MoK harness feeds identical
input and routing every iteration, epoch `e−1`'s `A2q` bytes are **bit-identical**
to epoch `e`'s — so **a completely absent readiness edge returns the right
answer, passes every gate, and posts the best number in the sweep.** Signoff
conditions before any overlap arm is timed: use `producer_drain_release<agent>`
(a primitive we own and do not call), and add the DQ2-NaN-poison detector
(2.1 MB, unobservable in a correct run, trips the zero-nonfinite gate on a
premature read).

Runner-up, and the likeliest bug to actually ship: an off-by-one in the M7
pool's start/stride that covers a task **twice** doubles one 32×448 tile out of
16,720 — ≈6×10⁻⁵ relative error, `pperr = 0`, every gate green. Under-coverage
is loud; over-coverage is silent. Free detector: `part_done`, which mode 12
allocates, zeroes in M0, and never writes.

### Vendoring provenance — one trap to avoid

The authoritative donor is `solution/hip/n2_phase1_gm.cpp`, **585 lines**,
sha256 `1d90b26658b6a524db69434ddcfa0b2dcec2418c852f83874468d55dd0197dc2`. The
`exp_59`/`exp_63` snapshots match it; the **`exp_65` snapshot is a different
634-line fork** (`52ecd9e7…`) and must never be the vendoring source.

## exp_26 landed (`f9bfb4be`) — the vendored M6 body, and the hints are not a null

`n2_phase1_gm_mps.cpp` is **+112 / −0** against the donor: not one donor line
modified or deleted, the four hint lines per half now sitting unchanged in the
`#else` arm of a gate. Gate off is **`.text`-byte-identical** to the donor build
(sha `96049dfa…`, both 166,656 B, zero-line `llvm-objdump` diff; the ELFs differ
only in the HIP compilation-unit identity symbol, which hashes the TU path).

**The G-scaled hints change instruction placement substantially, and for the
better.** Per-iteration instruction mix is identical — only placement moved:

| | gate off | gate on |
|---|---|---|
| position of the loop's only `s_waitcnt vmcnt(0)` | **mfma = 0** (stalls with zero MFMA of that iteration issued to cover it) | **mfma = 48** (1,536 cycles of issue first) |
| barrier partition across the two `lds_cta_barrier()` | **33 / 49 / 14** | **48 / 48 / 0** |

So the wrong hints did not merely fail to help — they actively steered the
compiler into the worst placement available, putting the loop's only full VMEM
drain where nothing covers it and leaving the two LDS buffer phases unbalanced.
Cost of the fix: 3 extra `s_waitcnt lgkmcnt` per iteration against 13 fewer
instructions in the loop; LDS waits are ~170–200 cycles of a ~9,000-cycle
K-step, so a small debit.

**Ceiling, honestly bounded:** the mechanism moves ≤1,536 cycles of MFMA in
front of one drain inside an ~18,000–19,600-cycle iteration ⇒ ~8 % of M6 ≈
200 µs ≈ 2.6 % end-to-end, and only if that drain is fully exposed today.
Predicted **75–200 µs**.

### The catch that must be measured before this ships

Turning all four hints on migrates **96 extra scratch accesses into the M7
epilogue / M8 / M9 region** — scratch instructions 21 → 114, with 99 of the 114
landing after the phase-2 K-loop, up from 3. Scratch *bytes* actually improved
(144 → 128 B/lane, spills 16 → 14): fewer values spilled, accessed far more
often, in precisely the region that is mode 12's hot fabric path. A
phase-1-only hint change reached that far through whole-function register
allocation.

Consequences, both now in flight: the gate is being turned into a **4-bit mask**
(DS-read / VMEM / MFMA / DS-write) so we can find the subset that buys the
placement win without the scratch migration; and the first GPU arm is the
`timestamps=1` attribution pair, not an end-to-end number, because `ts_M6_us`
and `ts_combine_us` are predicted to move in **opposite** directions and one
total cannot separate them. If M6 drops and the combine rises by more, the
response is to chase the epilogue's register allocation — not to close the axis.

## exp_24 MEASURED — A is real (−91 µs end-to-end), B is closed

**Mechanism A (delete the 448 MiB dead `part` zero-fill): CONFIRMED.**

| population | n | plan M3→M5 mean | σ |
|---|---:|---:|---:|
| without A (`g` = 33, 289, 545, 801) | 7 | **428.96 µs** | 4.73 |
| with A (`g` = 97, 353) | 4 | **377.75 µs** | 7.56 |

**Δ = −51.2 µs on the plan phase, SE 4.18, t = 12.3.** End-to-end `m2→end` is
**−91.2 ± 33.6 µs** (t = 2.7) — *more* than the plan delta, consistent with
−51 µs of plan plus LLC-pollution relief elsewhere. Best composed screen point
`g = 353` reads **6,645.3 µs / 0.8516**. Resource tuple clean, scratch actually
improved 144 → 128 B/lane, MFMA census and atomic count unchanged.

Honest cross-check the agent ran and reported: 469,762,048 B removed in 51.2 µs
implies **9.2 TB/s**, which is above HBM peak — so those stores were never
costing a full HBM write-back (nothing reads them and the next epoch overwrites
them, so the LLC was absorbing much of the traffic). The deletion is worth
51 µs of plan time, not the 100–300 µs a naive bytes÷bandwidth model predicted.
A 5-rotation decision campaign on `g = 97` (A alone, single variable) is running.

**Mechanism B (throttle depth): the axis is CLOSED.**

| depth | n | M7 mean | Δ vs 8 | |
|---:|---:|---:|---:|---|
| 4 | 2 | 2,659.4 | −82.9 | t = 1.3 — **null** |
| **8 (shipped)** | 7 | **2,742.3** (σ 63.1) | — | |
| 16 | 1 | 3,204.4 | **+462.1** | 7.3σ |
| 32 | 1 | 3,112.2 | **+369.9** | 5.9σ |

**The throttle is not a smooth knob — its entire ~500 µs benefit is already
realized at depth ≤ 8, and 16/32 are catastrophically worse.** exp_21 picked the
right value first try. No further win on this axis.

### A correction that changes how every later screen is read

**End-to-end screen resolution is far worse than the 0.52 % measured earlier.**
All six `g = 33` control points tonight spread **6,795.1 → 7,243.7 µs = 6.6 %**.
A screen cannot rank anything under ~5 % end-to-end.

**But the phase stamps are excellent instruments**: plan M3→M5 has σ = 4.7 µs
(1.1 %) and M6 has σ ≈ 22.9 µs. That is exactly how a −51 µs effect became a
12σ result while being invisible end-to-end. **Rule for the rest of the night:
screen phase-local mechanisms on their phase stamp, and reserve end-to-end
numbers for campaigns.** This directly determines how exp_26's mask ladder
(a predicted 75–200 µs M6 effect ⇒ 3–9σ on the M6 stamp) and exp_27
(−130…−190 µs, also M6) get judged.

## exp_32 — the poison patch is ready; the S-1 load guard is KILLED

**S-1 killed, and the kill saved a regression.** The dead slot loads are
genuinely unpredicated (15 of 16 issue with no exec mask, no compare, no
branch), so the 154 MiB / 34 % figure is confirmed as *issued* volume — but
every dead load reads `base_slot(cur,0)`, **one 14,336 B rank-local region, 224
cache lines**, resident in L1/L2/LLC with zero xGMI. Revised prize **≈ 4 µs
(band 0–15) = 0.06 % end-to-end**, ten times below screen resolution. Worse, the
patch would likely *regress*: the unconditional issue **is** the load pipeline —
fifteen loads in flight drained by one `s_waitcnt vmcnt(0)` — and guarding
forbids the hoist, converting it into ~10.5 serialized issue-then-drain round
trips. No patch written.

Side finding worth carrying: `fanout[t]` is wave-uniform by construction but
reaches LLVM as a VGPR, so all 16 slot guards are EXEC-mask guards. A
`readfirstlane` makes them scalar branches and deletes 8–12 % of M8's executed
instructions — still only ~0.1 % end-to-end, so it should ride along with
another M8 edit rather than get its own campaign.

**The poison patch is mechanically generated and verified** (`git apply -p1`
clean against the node's exact files, patched Python passes `ast.parse`, patched
shell passes `bash -n`). Three **untimed** insertion sites — the eager per-arm
epoch, before every one of the 600 soak epochs, and one extra untimed
verification epoch after `_mok_rank_max` so the post-timing `[MOK GATE]` becomes
a real single-epoch coverage check. Nothing is inserted between timed
iterations: a 56 MiB fill there would displace 22 % of the Infinity Cache and
perturb the very inter-rank skew that the rank-max p50 measures.
`K0_MOK_POISON_OUT` defaults **on** and is forwarded through `run_campaign.sh`.

Three refinements to the original diagnosis:
1. `out` **is** cleared once per gate episode — the accurate statement is that
   it is never cleared *between* epochs, and the 600-epoch soak has no clear at
   all. Zero is not a usable poison anyway: one missing row of 4,096 moves the
   L1 ratio by 0.024 % against a 0.1 gate.
2. `correctness.py` gives a better hook than expected — a SUM-all-reduced
   `nonfinite == 0` on the candidate buffer, **an equality that cannot be
   dialled**, so one surviving NaN on any rank fails three independent ways.
3. **The existing negative control cannot catch staleness** — it is a
   store-over-a-peer bug in the frozen pull combine and never runs `mps_mega`.
   A host-only detector self-test was added, plus a kernel-side skip-a-row
   control for whoever owns the kernel next.

The patch also bundles a **required harness correctness fix**: the `[MOK GATE]`
loop runs after the eager loop has finished, so for every candidate arm it
re-reads whichever candidate ran last.

## exp_25 rev2 — the interleave's ceiling is +211 µs. DEMOTED to a wash.

I proposed promoting the M6/M7 interleave over the static split on the grounds
that the 976 µs epilogue surcharge is irreducible fabric time (392 MiB / 976 µs
= 421 GB/s = **78.3 % of the 537.6 GB/s ceiling** — both numbers confirmed
exactly, and the wire time for 392 MiB at 78 % efficiency is 980 µs, matching
the surcharge almost perfectly). **The arithmetic is right and it refutes the
conclusion I drew from it.**

**421 GB/s is not a utilization.** The 976 µs is the per-CTA *sum* of epilogue
phases, not a wall-clock window; a CTA is in an epilogue 36.7 % of the time
(41.2 of 112.4 µs per task). The bursts are **aligned by construction** — all
240 CTAs enter M7 within a few µs of each other and every M7 task costs the
same, so they march in lockstep. That alignment is exactly why exp_21's throttle
recovered 500 µs. But if the aligned bursts already drive the fabric at 78 % of
ceiling, then de-aligning them or hiding them under M6 recovers **at most the
78→100 % gap = +211 µs**, and zero at the efficiency the surcharge already
implies. **"Irreducible" is an argument against the mechanism, not for it:
exp_21 already took the reducible part.**

My four claims, adjudicated:

| claim | verdict |
|---|---|
| no work-conservation loss | **confirmed as stated, implication refuted.** `A6/256 + A7/240 = 5,248 µs` is exactly today's makespan, so the interleave is 173 µs better than the static split — but no penalty is a **tie at first order**, not a gain. Every µs must come from second-order terms. |
| F1 becomes irrelevant | **confirmed, with an unpriced cost.** The gate is genuinely gone — but F1's favourable branch was the *static split's* upside (up to −535 µs), and the interleave wins zero first-order regardless. **If M6 is CTA-insensitive, the static split beats the interleave.** Keep F1, demoted from gate to option-pricing. |
| zero register risk | **confirmed, better-founded than the rev-1 hedge** — `k0p6_dread` is a volatile load whose memory clobber exists precisely to stop descriptors being hoisted into kernel-long registers. New risk is **I-cache**: both MFMA bodies inlined into one hot loop against a 32 KB L1I shared by two CUs. Read it out of the resource report. |
| still on-mandate | **compliant but hollow.** The pool survives, so the arm is compliant — but under the interleave every compute CTA runs the same mixture, so M6/M7 is a *schedule*, not a role partition. It adds nothing to the mandate's research question; the static split does. |

**Stage 0 turned out to be an identity, and needed no build.** M6's round `m`
completes exactly tiles `32m…32m+31`, so readiness is linear in M6 progress; CTA
`bid`'s `i`-th M7 task needs tile `bid/16 + 15i`, giving available `2.133m`
against required `2.133` — **equal identically**, for any routing. Availability
is *not* the limiter, so the mechanism is not dead on arrival. But there is
**exactly zero slack**: the M7 front rides precisely on M6's production front,
every `a2_done` poll by a CTA slightly ahead of its peers is a real stall, and
the steady state is both phases finishing together at 5,248 µs — the
work-conservation answer, re-derived independently.

**L2 does not kill it; the fabric arithmetic demotes it.** Interleaved per-XCD
footprint is 6.7 MB against 4 MB (M6 improves 10.6 → 5.3, M7's resident 2.64 MB
`W2` is destroyed), but aggregate demand is 3.67 TB/s — 25 % *below* the
4.91 TB/s M6 already sustains — so the damage is latency-bounded at
**−0 to −222 µs**: same magnitude as W1's ceiling, opposite sign.

**Predicted 6,713 µs central (0.870×), band 6,481–6,944 — the 6,685 ratchet sits
inside the band.** Roughly 60/40 that `k = ∞` (do not interleave) wins the
sweep. Inverted falsifier worth keeping: **any `k` beating `k = ∞` by more than
211 µs exceeds the arithmetic ceiling and is a defect signature, not a result.**

Two coordination facts went our way: the interleave's protocol is bit-identical
to mode 12, so **no new mode number is needed** and none of the seven predicates
is touched (`k` lives in free config bits, `k = 0` ≡ today); and exp_26's
`f9bfb4be` already landed both the task start/stride hooks and the
epilogue-done hook where the `a2_done` fix belongs. Revised build **4–6 h + 2 h
GPU**, down from 14–19 h. Signoff conditions if it is ever built: the
`vmcnt(0)` hook ships in the same commit as the fused loop, and **the
DQ2-NaN-poison arm must be shown to fire on the hook-removed control before any
sweep number goes upward** — on this harness the most likely way this experiment
produces a headline number is by being broken.

**Decision: do not build the interleave tonight.** A predicted wash with a hard
+211 µs ceiling loses to exp_27 (−130…−190 µs, bit-exact gate, same 6 h) and to
exp_30's Stage 0 (~2 h to price a 700–1,000 µs mechanism). **F1 survives as a
cheap 2-screen measurement** — the task start/stride macros are already live, so
it now costs ~30 minutes and prices a −535 µs option.

## exp_29 design verdict — SAFE but readiness-limited; KILLED as specified

The pipelined combine is **provably safe** — the consume-and-zero proof survives
because pipelining moves *when* the zero happens, not *what gates it*, and each
slot row has exactly one writing rank so the flag is a complete gate. It is
still not worth building, for a reason that has nothing to do with correctness.

**The prize is readiness-limited, not capacity-limited.** A token becomes
reducible only when the **last of its 8 routed experts' M7 tiles** completes.
Top-k picks 8 distinct experts, a tile carries exactly one expert, and M7 walks
tile index roughly linearly, so `P(reducible by t) = (t/S)^8`. **The median
token is reducible with 8.3 % of M7 remaining; at M7's halfway point 0.4 % of
tokens are reducible.** No pool size, scheduling policy or primitive moves that
curve — it belongs to **M7's task order**, not to the combine.

Predicted `Δcombine = −112 µs` against `ΔM7 = +90 µs`, net **≈ −22 µs, band
−230 to +160** — straddling zero, and screens cannot adjudicate a 0.3 % effect
against a 0.52 % σ. Two sub-policies died on arithmetic on the way: pool-only
needs `C ≈ 43` to reach coverage, which costs +338 µs of M7 to buy at most
430 µs of combine; backlog-driven has nothing to tune, because the backlog is
~0 for 80 % of M7 and then floods.

One structural finding worth keeping: **the CTA that publishes `row_ready` is
usually not even on the same GPU** (≈19,089 peer stores vs ≈2,727 self stores),
and `tau` is nowhere on the producing rank, so a per-token counter bumped by the
publisher is *structurally impossible* — it would need a new `tau_table` plus
~21.5k system-scope remote RMWs injected into M7, i.e. exp_20's +624 µs
peer-write class.

### Two things exp_29 handed back that are worth more than its own mechanism

**S-1 — a one-line wave-uniform load guard, ~154 MiB of deleted loads.**
`KRN:547` issues the slot load unconditionally and guards only the accumulate at
`KRN:561`. `fanout[t]` is **wave-uniform** (every lane computes it at
`KRN:484-489`), so hoisting the guard deletes **34 % of the combine's read
instructions** — today out-of-fanout lanes re-read one 14,336 B region.
**Worth more than the entire pipelining mechanism, with none of its risk.**
Needs a 10-minute ISA check first in case LLVM already sinks the load.

**The real unlock — reorder M7 nc-major.** `task = nc·num_tiles + tile` turns
the readiness curve into a 16-step staircase with **15/16 of the combine
unblocked before M7 ends**, lifting the ceiling from ~17 % to ~85 %. That is
M-series **M8 / COMET layer-1**, and it is now `exp_30`.

## exp_28 — M6's intensity axis is CLOSED, and it is empty rather than expensive

Three independently measured walls, two of them new tonight:

| wall | constraint | how established |
|---|---|---|
| accumulator footprint | `M·N ≤ 59,392` | 256 AGPR; exp_65's durable finding that the wall is *accumulator footprint*, not tile width |
| **LDS ceiling** | **exactly 163,840 B** on gfx950 (the compiler rejects 163,841); fused law `1,092·M + 50,596` ⇒ **`M ≤ 103.5`** | five-point `N2GM_G` sweep, CPU-only, tonight — **this wall is independent of registers and nobody had recorded it** |
| `DQ2`/amax group | `N ∈ {256, 512, 1024}` | the 128-column quantization group |

Under all three, **the shipped `(96, 512)` tile is already the
intensity-maximizing legal shape.** `G=4` needs 190,372 B of LDS — 26,532 B over
the ceiling — on top of ~88 registers/lane.

### The finding with the widest blast radius

**Time is not proportional to request bytes.** exp_65's paired same-run points:
a 26 % byte cut bought 9.4 %; a +11 % byte increase cost +27 %. Measured
marginal rates:

| stream | cycles per byte | why |
|---|---:|---|
| weights (B) | **0.0826** | straight to registers |
| **activations (A)** | **0.407** | global→register→LDS→register with a loop-carried `vmcnt` and two CTA barriers |

**The A stream costs 4.9× per byte.** Any future traffic argument must be
weighted by stream, not counted in bytes. This retroactively explains exp_65's
`nc=16` kill (**+866.6 µs of A re-pass tax against −376.4 µs of G
amortization**, net 1.0023×) and it is now the central risk in exp_30.

### History corrected

- **`G=4` was never built and never timed.** It died on a *standalone* register
  probe; `exp_63/design.md:136` says literally "G=4: dead … Not attempted."
  `CLAUDE.md`'s "G3/chunk/K-split" shorthand had been read as a kill on the
  whole axis.
- **`G=3` was itself twice a build-gate kill** (512V/256A/7–9 spills, fused) —
  and shipped later and won. A build-gate kill is not a mechanism kill.
- **`nc=16` is a genuine measured kill** (exp_65: built, gated, timed).
- The `M=128, N=384` register-neutral variant: **the register arithmetic was
  right** — it is −8 registers, not merely neutral, at +18.7 % intensity — but
  it is refuted three other ways. `kChunks = 2048/192 = 10.67` is not an
  integer; LDS is 17,860 B over; and fatally, a 192-column chunk **straddles
  1.5 of M7's 128-column K-groups**, so the straddling group's `amax` would be
  computed from half its columns in each of two tasks on two CTAs and quantized
  by two different scales, while M7 applies one `DQ2[row][k]` scalar to all 128.
  Numerically wrong, not merely awkward.

### The recommendation: exp_27 is a deletion, and its gate is bit-exactness

M6's prologue gathers 96 × 56 FP32 scales from a group-major array with a
131,072 B stride — **5,376 distinct 64 B lines to deliver 21,504 B**, 344 KB per
task, 977 MB per epoch, sitting between two `__syncthreads()` where no MFMA can
cover it. **That layout is manufactured by us, in M5, from `sc_stage`, which is
already token-major, for a consumer set of exactly one: M6.**

So the fix is a **deletion, not a second transpose** — the one thing exp_27 must
not get wrong. Point M6 at `sc_stage` and delete M5's transpose. A token-major
row is 224 B and `224·t mod 64 ∈ {0, 32}` for every `t`, so a row always touches
exactly 4 lines: **384 lines against 5,376 — 14.0× exactly, 907 MB/epoch
deleted.** Three independent models agree: **−132 / −134 / −186 µs**, plus
≈ −17 µs in M5. Zero registers, zero LDS, no protocol.

`sc_dst[k·T_ext + t] ≡ sc_stage[t·56 + k]` **by construction**, so the output
must be **bit-identical** — a far stronger gate than any tolerance. ~6 build
hours. Note it composes with exp_24: once exp_24 deletes the `part`-zero half of
`zero_part_scale_transpose`, exp_27 deletes the other half and the whole loop
goes.

Also from exp_28: **O2 (task-order swizzle) is predicted 0** and should be
settled with one `rocprofv3` PMC pass rather than a build; and `num_tiles[0]` /
`nvi[0]` are unprinted, carrying **±4 % of uncertainty on every M6 number** —
a 0.5 h instrument worth adding before the M6 experiments are judged.

## THE MEASUREMENT-INTEGRITY BUG — applies to every number tonight

**`out` is never cleared between epochs, and the campaign feeds identical inputs
every iteration. So any bug whose signature is "this row didn't get written"
returns the previous epoch's bit-identical correct answer** — invisible to
`[MOK GATE]`, invisible to `combine_bit_exact`, invisible to 600 soak epochs. A
rare premature-ready is similarly invisible: one token short by one of ~5.25
addends is a 19 % error on that token, which dilutes to ~0.3 % against a 10 %
gate.

This is the same class as exp_25's `a2_done` landmine (identical per-iteration
input makes epoch `e−1`'s `A2q` bit-identical to epoch `e`'s, so a missing
readiness edge returns the right answer *and posts the best number in the
sweep*). Both say the gate ladder is weaker than it looks against
staleness-shaped bugs.

**Fix: poison `out` with NaN between iterations — one host-side line, no kernel
change.** It collapses the whole class, and the zero-nonfinite gate already
exists to catch it. This lands before any further mechanism is timed.

### exp_26 rev2 (`7a06fb56`) — the four hints are four different knobs

Builds turned out to be ~6 s, so all **16 masks** were enumerated rather than
sampled, and the effects factor cleanly. Bits are `DSW | MFMA | VMEM | DSR`:

| bit | effect | resource cost |
|---|---|---|
| **2 — MFMA** | **the entire 33/49/14 → 48/48/0 barrier realignment** | **none — every resource field byte-identical to the donor** |
| 0 — DS read | the drain move (`mfma 0 → 48`) **and all 96 migrated scratch accesses** | scratch instructions 21 → 114 |
| 1 — VMEM | **exact no-op** — all eight pairs differing only in bit 1 are byte-identical (phase 1's `16+kGM = 19` already matched the measured 19) | — |
| 3 — DS write | perturbs the bytes, moves no metric | — |

So the first report described **two independent effects on different bits as one
effect.** The drain move and the scratch migration are both pure functions of
bit 0 and 16 masks leave nowhere for them to come apart — but the question is
moot, because **bit 2 buys the barrier win for free.**

**Recommended arm: mask 4.** It is the only bit that changes the schedule while
leaving the resource profile byte-identical to the donor, which makes it a
genuinely single-variable arm: if the campaign moves, the barrier realignment
moved it. That also yields a clean 2×2 ladder — **mask 0** (control) / **4**
(barrier only, free) / **1** (drain only, +scratch) / **5** (both, +scratch).

**The migration is worse than "96 more spill accesses", and it is a known
regression.** A `-gline-tables-only` build (verified not to perturb codegen)
attributes all 96 to a **single source line**: `n2_phase2_gm_mps.cpp:145`, the
`peer_tab[xr >> maxtok_sh]` base load in the remote-atomic accumulate loop.
Across the kernel's 282 `flat_atomic_pk_add_bf16`, bit 0 puts a `scratch_load`
exactly **17 instructions ahead of 96 of them** — constant min = median = max,
i.e. one inlined shape — which is character-for-character the regression that
exp_21 restructured `throttle_plan` into four booleans to remove. Bit 0
re-triggers it *from the phase-1 side* by pushing whole-function pressure past
the same 256-VGPR ceiling. Note the direction: spilled **bytes go down**
(144 → 128 B/lane, 16 → 14 VGPRs) while **accesses go up** 21 → 114 — the
allocator evicted fewer values but picked the one read on the hottest path.
Follow-up worth taking: relieve one live VGPR in phase 2's epilogue and bit 0
becomes free. That needs both files under one owner.

**Mask 0 plus the two new task macros is `.text`-byte-identical to the donor
build** (`96049dfa…`, 166,656 B, identical resource tuple and K-loop timeline),
so `N2GM_P1_TASK_START`/`_STRIDE` compile to the donor's loop exactly and F1 is
unblocked. Diff is now +174 / −9 (a per-bit mask made the "wrap in `#else`"
shape untenable); the `−0` provenance claim is replaced by the stronger measured
one.

**Phase 2's hint, measured directly: 17 VMEM reads per half against a hint
asking for 29.** The one-line patch to `14 + kGM` preserves the `kGM = 1` anchor
for free (`8 + 7·1 = 15 = 14 + 1`), so it is a parameterization fix, not a
retune — same property that made phase 1's safe. Written up with line numbers
and a content match in `activate.md` §7.1 as **exp_31**.

### Two corrections to `CONTEXT/m6_m7_structure.md` from the ISA read

1. **§5.3 item 4 is wrong.** M6's K-loop *does* contain a compiler-inserted
   `s_waitcnt vmcnt(0)` — one per **iteration** (not per half), always a full
   drain, never a partial `vmcnt(N)`. The `N2_FORCE_VMCNT0` probe really is
   compiled out; the compiler inserts its own. This closes §7 open item 4.
2. **Phase 2's own `0x020` VMEM hint is over-scaled** — `8 + 7·kGM` = 29 at
   G=3 against a measured 17; the correct form is `14 + kGM`. So "phase 2
   scales three of its four correctly" is only 3/4 true. Phase 2 is M7 at
   ~2,660 µs, and this is the same class of defect just fixed in phase 1.
   Queued as a separate one-line experiment in `exp_26_p1_sched/activate.md`
   (the file belongs to exp_24 tonight).

## Standing rules in force

Every candidate: correctness + negative control + 600-epoch soak **before**
timing, no exceptions. One variable per arm. Same-run paired denominators only.
Sub-5 % deltas re-run. Bump `K0P6_MPS_SRC_REV` with any `.cuh`-adjacent edit and
confirm a new `.hsaco` mtime — directory mtimes are touched on a cache hit and
`latest/` is a symlink dir, so `stat` needs `-L`. The node checkout **is** the
arm. One GPU job at a time; `rocm-smi --showpids` is the check that works
(`pgrep -af torchrun` does not see our own job, which runs as
`python -m torch.distributed.run` inside the container).

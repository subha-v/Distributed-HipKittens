# Ablation queue — phase-adaptive CTA partitioning

Our partition is already phase-adaptive, but crudely: the service pool is a step
function in time, `0 → C → 256`. It is zero before the M6/M7 boundary, `C` for
the duration of M7, and grows to 256 as compute CTAs fall through after M7.
Nothing about it responds to the workload.

The research question this queue attacks: **what is the right trajectory of the
compute/communication split across a kernel's phases, and can it be derived at
runtime instead of tuned?**

Each entry carries a pre-registered expectation so a null is interpretable.
Ordered by information per unit of build risk, not by expected speedup.

---

## Tier 1 — characterize the interference (we cannot design against what we cannot see)

### E1. Pace the pusher — rate vs volume

**The single highest-information experiment available.** Insert a tunable delay
between pushes (or cap outstanding pushes per wave) and sweep the service rate
while holding total bytes constant.

- Interference **rate-driven** ⇒ M7 recovers as the pool is slowed, and the
  combine lengthens; there is an optimum pacing and a scheduler can find it.
- Interference **volume-driven** ⇒ M7 does not recover at any pacing; only
  moving fewer bytes helps.

We currently cannot distinguish these, and they imply completely different
designs. Implementation is a delay loop in the push path plus one config field.

**Pre-registered:** rate-driven, because the pool bursts hard and the fabric and
L2 request queues are the suspected resource. If it is volume-driven, every
pacing/scheduling idea below is dead and only traffic reduction survives.

### E2. Read-side vs write-side vs fabric attribution

Three variants of the service pool, all at matched `C`, all doing identical
bookkeeping:

| variant | reads `part` | writes peer | writes local scratch |
|---|---|---|---|
| baseline | yes | yes | — |
| read-only | yes | — | — |
| write-only | — | yes (garbage) | — |
| local-write | yes | — | yes |

Diffing M7 across these attributes the 815 µs to the local read, the fabric
write, or their sum. **Not correctness-preserving** — these are diagnostic
builds whose outputs are discarded, exactly like the earlier scope diagnostic.

**Pre-registered:** the local read dominates, because it competes with M7's
weight streaming in the same HBM/LLC path, whereas the peer write leaves the
die. If the peer write dominates instead, that is a strong argument that the
fabric request path is the shared resource.

### E3. Occupied-but-silent pool

A pool that spins on an LDS-only loop: CTAs are consumed, no memory traffic is
issued. We already have the idle-pool control (reserve and do nothing), which
isolates capacity. This isolates *CU occupancy with instruction issue* from
*memory traffic*.

**Pre-registered:** indistinguishable from the idle pool, confirming that
nothing about merely occupying a CU matters and the whole effect is memory
traffic. A surprise here would overturn the A7 strike.

---

## Tier 2 — is M6 a viable host for communication?

### M1. Is M6 traffic-sensitive, or only CTA-insensitive?

We know M6 is **CTA-insensitive** (removing 24% of CTAs costs 0.5%). We do
**not** know whether it is **traffic-insensitive**. Run a synthetic traffic
generator on `C` reserved CTAs during M6 — same access pattern and volume as the
real pusher, writing to a scratch buffer — and measure M6.

This is the decisive question for any phase-adaptive design that wants to move
communication earlier: M6 is 36% of the kernel and the only phase with slack.

**Pre-registered:** M6 degrades less than M7 does, because it is not
CTA-bound and so is probably latency-tolerant rather than
bandwidth-saturated — but it will degrade. If M6 turns out to be genuinely
traffic-insensitive, that is the most valuable single finding available from
this queue, because it means the partition should be shifted earlier wherever a
dependency allows it.

### M2. Where does M6's time actually go?

M6 is the largest phase (2,539 µs, 36%) and its limiter is unidentified. It is
not CTA-throughput-bound. Sub-phase stamps inside M6 (tile loop entry, first
MFMA, epilogue) would at least separate prologue/dependency-wait from MFMA from
epilogue.

**Pre-registered:** a substantial fraction is waiting on `a2_done` per-tile
dependencies from the preceding phase rather than MFMA, which would make M6
partly a *pipelining* problem rather than a compute problem.

---

## Tier 3 — the partition trajectory itself

### T1. Queue-depth-driven join — a self-tuning partition

Replace the fixed `C` with a control loop. A compute CTA that finishes an M7
tile inspects the event-queue backlog (`tail − ev_next`, both already
maintained) and joins the drain only if the backlog exceeds a threshold;
otherwise it claims another M7 tile. The partition becomes **data-adaptive**:
it grows when communication is falling behind and stays small when it is
keeping up.

This is the headline mechanism of the whole line of work — it turns a tuned
constant into a runtime-derived quantity, and it should automatically handle
routing skew, stragglers, and shape changes that a static `C` cannot.

**Pre-registered:** at the benchmark's uniform routing it matches the tuned
static optimum (there is nothing to adapt to). Its value shows up in T4. A
result *better* than static at uniform routing would mean the optimum `C` varies
*within* an epoch, which would itself be a finding.

### T2. Pool-size trajectory — ramp instead of step

Constant `C` during M7 is an arbitrary shape. Test monotone ramps in both
directions, driven by M7 task-claim progress:

- **ramp-up:** few service CTAs early in M7 (when the most MFMA remains and
  interference is most expensive), more as M7 drains.
- **ramp-down:** the opposite.

**Pre-registered:** ramp-up wins over constant, because interference cost scales
with the amount of M7 left to damage while the combine's exposure only depends
on total push throughput integrated over the phase. If ramp-down wins, our model
of the trade-off is wrong.

### T3. Reservation point (the long-standing untested axis)

Reserve at end-of-M5, mid-M6, the current M6/M7 boundary, and mid-M7. Because
M6 is CTA-insensitive, reserving during M6 is nearly free in capacity terms —
this measures whether an earlier-warmed pool (already polling when the first
events arrive) matters.

**Pre-registered:** neutral, because the first tile event cannot exist before M7
begins and the profile already shows the first event observed *before* the last
CTA leaves M6. Worth running to close the axis rather than for the expected
gain.

### T4. Fall-through fraction

Currently 100% of compute CTAs join the drain. Sweep 25/50/75/100%. The shape
tells us whether the tail drain is CTA-limited at all, or whether it saturates
well below 256.

**Pre-registered:** saturates below 256, since at `C = 96` the combine is
already down to 108 µs before any fall-through contribution.

---

## Tier 3.5 — elimination candidates surfaced by exp_21 + the competition analysis

Mode registry (allocated): modes 0-3 original; exp_20 diagnostics hold 4-8
(E1 pacing, E2 traffic, E3 LDS-spin, payload-free stream, poll backoff);
exp_21 holds 12 (direct remote accumulate). Next free: 13. **Coordinate before
taking a number.**

1. **X1. Per-row target counters.** Replace `nc_arr[r*16+nc]` (target
   `row_rem[r]` per chunk) with one counter per row, target `16·row_rem[r]`.
   Deletes the `pushed` cell entirely (~40% of mode 12's remaining protocol
   atomics). exp_09's old objection is WITHDRAWN per STATUS.md (32 live lanes
   of an event address 32 different rows, so exp_07's essential line-spread is
   preserved). Cheap; high-confidence.
2. **X2. Direct-to-out accumulation (mode-16-scale).** Elimination to the
   limit: no slots, no M8, no readiness protocol, no service pool — weighted
   contributions go straight into the owner's `out[tau]`; per-epoch
   completion rides the retirement handshake. Needs: `tau_table` written at
   M1 (one 4 B remote store per (token, route)), per-epoch `out` zeroing
   (58 MB/rank), an M9 cross-rank arrival. Prize: the remaining 446 µs
   combine tail + every readiness atomic + every service CTA. Mode 12
   de-risks its shared bottleneck (fabric RMW rate) first.
3. **X3. Dependency-ordered M7 tiles.** M3-M5 already computes a plan; order
   M7 tasks so rows complete steadily (flatten the demand curve). Zero CTAs,
   no protocol. Substitute for T1's capacity term — the comparison against
   T1 is the point of running both.
4. **X4. Demand-curve instrumentation.** Histogram per-tile-event enqueue
   times inside M7. mean/peak × the pool's capacity share = the hard ceiling
   on what any adaptive-join can reclaim. Cheap (timestamps infra exists).
5. **X5. Bandwidth-aware ticket.** Join the drain by queue depth AND a
   memory-pressure proxy, not completion order — the novel scheduling idea;
   build only after X4 tells us the demand shape.


## Tier 4 — does adaptivity actually buy generality?

### G1. Re-run the C sweep under routing skew

The harness has a `skewed_hot` synthetic route family and `K0_SYNTH_ROUTE` is
now forwarded into the campaign containers. Sweep `C` under skew for both the
static and the T1 adaptive partition.

This is the experiment that would demonstrate the *point* of a phase-adaptive
partition. On uniform routing every rank is symmetric and a static `C` is
adequate — which is exactly why our dispatch has no wait to hide. Under skew,
some ranks receive far more tokens, the combine load becomes uneven, and a
static `C` tuned for uniform routing should be visibly wrong.

**Pre-registered:** the static optimum shifts with skew and degrades, while the
adaptive partition tracks it. **If adaptivity shows no advantage even under
skew, the case for phase-adaptive partitioning on this workload is weak and
should be said so plainly.**

### G2. Shape sensitivity

Sweep token count and top-k. The optimal `C` is currently tuned at one shape;
whether it is stable is unknown, and a static tuning that must be re-derived per
shape is a real deployment cost that an adaptive partition would remove.

---

## Notes on method, carried forward from this campaign

- **Screening is 40× cheaper than a campaign and faithful for ordering.** A
  1-process/1-warmup/1-timed run costs 25–95 s against ~18 min, and tracked
  campaigns to 0.1%, 1.3% and 2.6% on three separate checks. Shape axes with
  screens; decide with campaigns.
- **Any sub-5% delta must be re-run.** The current ratchet win is 0.5% and only
  became trustworthy after a second independent campaign reproduced it to 0.04%.
- **Editing a `.cuh` does not invalidate the JIT cache.** Bump the source-rev
  define in the hashed `.hip` and verify a new `.hsaco` mtime; cache *directory*
  mtimes are touched on a hit and are not evidence.
- **Never resync the node checkout while a campaign is running** — the
  containers bind-mount it live and later rotations will silently run a
  different kernel.
- **Matched-CTA-count comparison is mandatory** for anything involving the
  service pool. The reserve-but-do-nothing mode measures capacity only and is
  structurally blind to interference, which is the dominant cost.
- **A saturating accumulator and a lossy atomic look identical.** bf16 has 8
  significant bits; any accumulation test must keep every *partial* sum
  representable, not just the final value.

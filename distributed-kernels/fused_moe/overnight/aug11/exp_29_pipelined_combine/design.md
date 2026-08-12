# exp_29 — pipelined combine: reduce `out` rows in the M7 shadow

Design + protocol note. **Read-only experiment**: this folder contains no source
changes. Companion: `protocol_review.md` (adversarial review of this design).

File tags follow `../CONTEXT/mode12_protocol_map.md`:
`KRN` = `distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip`,
`ADP` = `distributed-kernels/fused_moe/moe_mps_adapter.cuh`,
`P2` = `distributed-kernels/fused_moe/n2_phase2_gm_mps.cpp`,
`HKA` = `distributed-kernels/fused_moe/moe_hk_adapter.cuh`.

---

## 0. Verdict first

**The mechanism is safe and buildable, but as specified its expected value is
≈ −20 µs (0.3 %) with a band that straddles zero, because the prize is not
capacity-limited — it is *readiness*-limited, and the readiness curve is a
property of M7's task order, not of the combine.**

The governing number, derived in §2: a token becomes reducible only when the
**last of its 8 routed experts' M7 tiles** completes, and M7 walks tile index
roughly linearly, so

```
P(token τ reducible by time t) = (t / S)^8 ,   S = M7 span ≈ 2,660 µs
```

The median token is reducible with **8.3 % of M7 remaining** (`0.5^(1/8) =
0.917`). At the halfway point of M7, **0.4 %** of tokens are reducible. There is
no scheduling policy, no pool size and no primitive that can move that curve.

Three consequences, and they set the whole plan:

1. **Do the free thing first.** `KRN:546-548` issues the slot load
   *unconditionally* and guards only the accumulate (`KRN:561`). 34 % of the
   combine's 448 MiB of issued loads are out-of-fanout lanes re-reading one
   14,336 B region. `fanout[t]` is wave-uniform, so hoisting the guard around
   the load is a **one-line, wave-uniform predicate** that deletes ~154 MiB of
   loads. This is worth more than the entire pipelining mechanism and carries
   none of its risk (§8, stage **S-1**).
2. **Measure the readiness curve before building anything** (stage **S0**, mode
   10, ~1.5 h). It is a read-only probe that also pre-prices the poll traffic.
   If it confirms `(t/S)^8`, the mechanism is capped at ~17 % of the combine and
   should not be built as specified.
3. **The unlock is M7's task order, not the combine.** Reordering M7 nc-major
   (`task = nc·num_tiles + tile` instead of `tile·16 + nc`, `P2:331-332`) turns
   the readiness curve into a 16-step staircase in which 15/16 of the combine is
   unblocked before M7 ends. That is **M-series M8** (COMET layer-1) in
   `aug10/CLAUDE.md`, and exp_29 is its natural consumer. Sequenced the other way
   round, exp_29 measures a mechanism against a ceiling it did not choose.

If the loop still wants a pipelined combine tonight, build **stage S1 only**
(fall-through reclamation, §8) — it is cheap, its ΔM7 risk is structurally
near-zero, and it is worth ≈ −75 µs on its own.

---

## 1. What the combine actually is — corrected traffic census

From `mode12_protocol_map.md` §3(d), with the arithmetic re-derived.

### 1.1 The three traffic classes

| # | site | op | issued bytes / rank / epoch |
|---|---|---|---|
| d5 | `KRN:546-555` | plain `uint4` load, local `slots` | 469,762,048 B = **448 MiB** |
| d6 | `KRN:579-587` | `uint4` zero store, local `slots` (consume-and-zero) | `F × 14,336` = 308,281,344 B = **294 MiB** |
| d7 | `KRN:595-602` | `uint4` store, local `out` | `T × 14,336` = 58,720,256 B = **56 MiB** |

### 1.2 The dummy-lane share of d5 — real traffic, but not DRAM traffic

`F = pull_ptr[T] ≈ 21,504` over `T = 4,096` ⇒ **mean fanout 5.25 of the 8
shuffle positions**, so **2.75/8 = 34.4 %** of positions are out-of-fanout. Those
lanes hold `pbase[t] = base_for(cur, 0)` (`KRN:496-503, 516`), i.e.
`slots + 0` — **the same address for every dummy position in the epoch**.

The load at `KRN:547` is **not** fanout-guarded; only the accumulate
(`KRN:561`) and the zero (`KRN:579`) are. So the dummy loads *are issued*:

- **Coalescing:** `pb` is a `__shfl` broadcast, uniform across the wave, while
  `off = (c<<10) + (lane<<4)` (`KRN:531`) is per-lane. Each `(t, jj)` therefore
  issues one perfectly coalesced **1,024 B** request. Nothing coalesces *away*.
- **But the destination is 14,336 B of `slots[0]`**, re-read
  `1,024 batches × 14 chunks × ~11 dummy positions` times. That footprint is
  resident in a 4 MB per-XCD L2 and certainly in the 256 MB Infinity Cache.

**Answer to the question posed:** the dummy share is **real instruction and
L2 traffic (~154 MiB of requests) but essentially zero DRAM traffic.** It costs
VMEM issue slots and L2 bandwidth, not HBM bandwidth.

### 1.3 The real work, and what it says about the combine's limiter

```
DRAM-real reads    d5 × (5.25/8)          = 294 MiB     (== d6 exactly; same geometry)
zero stores        d6                     = 294 MiB
out stores         d7                     =  56 MiB
------------------------------------------------------
W (DRAM-real)                             = 644 MiB = 675 MB
```

Measured combine ≈ **430 µs** (`aug11/STATUS.md`; `combine = REDUCE_DONE −
M7_DONE`, `ADP:44-51`). So the combine sustains **1.57 TB/s = 20 % of the
8 TB/s HBM3E peak**, i.e. **6.1 GB/s per CTA**.

**The combine is not bandwidth-bound.** That cuts both ways and both matter:

- *Against* the objection: injecting this traffic into M7 asks for at most
  ~90 GB/s (§7), **1.1 % of HBM peak**. The bandwidth argument for "M7 will get
  worse" is quantitatively weak.
- *Against* the mechanism: because it is issue/latency-bound, per-CTA throughput
  does not improve when fewer CTAs run it, so a small pool is genuinely slow.
  16 CTAs deliver ~98 GB/s, and 675 MB at 98 GB/s is **6.9 ms** — far more than
  M7's 2.66 ms window. **A pool-only combine cannot cover the work at any legal
  `C`** (§4, P1).

### 1.4 One correction to the brief

The brief describes M8 as "weights and sums" the 8 positions. It does **not
weight**: `KRN:565` is `acc[t][e] += k0p6_bf16_to_f32(u16[e])`, a plain sum. The
top-k weights are applied upstream in M7 (`swt` reaches phase 2 at `KRN:1377`).
This is load-bearing for the design: **the pipelined reduce needs no weight
vector in registers** — its entire per-unit state is the 4 shuffle bases and
`acc[4][8]`.

---

## 2. The readiness curve — the constraint that governs everything

### 2.1 The predicate, cited

A token τ may be reduced when, for **every** fanout entry
`j ∈ [pull_ptr[τ], pull_ptr[τ+1])` with `(p,row) = pull_src[j]` (`KRN:505-506`):

```
row_ready[p * T_loc_max + row]  >=  epoch32        (KRN:511-514, system scope)
```

published at `ADP:575-584` by producer `p`'s service pool, once
`pushed[row] == 16` (`ADP:601-611`), i.e. once all 16 N-chunks of that row have
had every contributing local expert arrive (`ADP:720-723`, target
`row_rem[row]`). The predicate is **unchanged** by this experiment; the pool
already computes it and M8 already evaluates it.

### 2.2 Why it is `(t/S)^8`

- A row `(p,row)` is complete when its **≤ 16 chunk-events** have all arrived.
  M7's task space is `tile = task/16, nc = task%16` (`P2:331-332`) with stride
  240 (`ADP` `k0p6_mps_stride`). `gcd(240,16) = 16`, so CTA `bid` runs only
  `nc = bid mod 16` — **the 16 chunks of a tile are executed by 16 different
  CTAs at the same task index**, hence at nearly the same instant. Row
  completion ≈ **tile** completion.
- A row on rank `p` is fed by `row_rem[row] ≈ 1.52` local experts
  (`P/R = 32,768/21,816 = 1.502`, §3 of the map) — ~1.5 tiles.
- A token τ is reducible when **all 8 of its routed (expert, rank) pairs** have
  completed. Top-k picks 8 *distinct* experts, and a tile carries exactly one
  expert (`KRN:1250-1273`), so those are **8 distinct tiles**, at ~uniform
  positions in the tile order (tiles are laid out by expert index; a token's
  experts are a uniform random 8-subset).

Therefore `R_τ = S · max(U_1..U_8)`, `U_i ~ U[0,1]`:

| `t/S` | fraction of tokens reducible | fraction of `W` unblocked |
|---:|---:|---:|
| 0.50 | 0.4 % | 2.6 MB |
| 0.70 | 5.8 % | 39 MB |
| 0.80 | 16.8 % | 113 MB |
| 0.90 | 43.0 % | 290 MB |
| 0.95 | 66.3 % | 448 MB |
| 0.99 | 92.3 % | 623 MB |

`E[R_τ] = 0.889 S`; median `0.917 S`. **The last token is reducible essentially
at M7's end, so the combine's tail is irreducible by construction.**

### 2.3 What this does and does not kill

It does **not** kill CTA role specialization at this boundary — the pool is real
free capacity and the traffic is cheap. It kills the *size* of the prize. The
correct reading, and the one to log: **the combine boundary's overlap ceiling is
set by the producer's task order.** Fix the task order (nc-major, M-series M8)
and the same mechanism becomes worth ~5× more.

---

## 3. Who discovers that a token became complete

This is the sharp scheduling question, and the structural fact that answers it
is stronger than the brief assumes.

> **The CTA that publishes `row_ready[p][row]` is usually not even on the same
> GPU as the owner.** `flush_pending` publishes into the *owner's* memory
> (`ADP:575-584`), and `owner == cur` for only ~1/8 of rows: c16 ≈ 2,727 self
> stores against c17 ≈ 19,089 **peer** stores.

### 3.1 Option A — per-token arrival counter bumped by the flag publisher

**Rejected, and the reason is structural, not performance.** §5 Q1 of the map:
"**`tau` is nowhere on rank `r`**" — the `pos → tau` map exists only on the
owner, as the never-inverted inverse of `pull_stage` (`KRN:859-860`), and
`pull_stage`/`pull_src`/`pull_ptr` are all owner-local (`DRV:1678, 1684-1685`,
plain `torch.zeros`, not symmetric). A remote publisher **cannot address a
per-token counter on the owner** because it does not know which token it just
completed.

Making it possible costs a new `tau_table[src*MAXTOK + pos]` symmetric buffer
plus one 4 B remote store per (token, route) in M1 (21,504/rank/epoch) — this is
exp_21's own named follow-up #1 — *and then* ~21,504 **system-scope remote RMWs
injected into the M7 window**. exp_20 §2b prices unthrottled peer writes at
**+624 µs of M7**. Adding a fabric RMW class to M7 to save combine time inverts
the experiment. **Reject for v1; revisit only if the direct-to-`out` end state
(A11/M11 taken to its limit) is being built anyway.**

### 3.2 Option B — a work queue of completed tokens

A transport for the answer, not a way to compute it. Someone still has to
evaluate the predicate, and by §3.1 that someone must be the owner. **Reject as
a discovery mechanism**; it survives only as a *distribution* mechanism on top of
option C, and option C's static stripe already distributes for free.

### 3.3 Option C — static token stripe per worker, non-blocking sweep — **CHOSEN**

Each worker wave owns a static stripe of the 1,024 M8 batches and makes
**passes** over it. A pass does **exactly one relaxed system load per outstanding
fanout entry** — no inner spin — reduces the batches found ready, and moves on.
Between passes it backs off with `s_sleep` (`pace_delay`, `ADP:430-438`).

Why this is the right shape:

- **It adds zero atomics.** No counter, no RMW, no new cache line. Against the
  ~918,472 protocol RMWs/rank/epoch already in the budget, the delta is
  **0**. The exp_07/exp_08 rule ("relocating or coalescing atomics loses;
  preserve line spread") is satisfied vacuously — there is nothing to relocate.
- **The lane geometry is already M8's.** Lane `l` handles token `t = l>>3`,
  entry `j2 = l&7` (`KRN:496-497`); "ready" is a per-lane bool and the batch
  predicate is `__ballot(...) == ~0` (`KRN:523`). The sweep *is* M8's poll with
  the spin removed.
- **It cannot deadlock** (§6 / `protocol_review.md` §6) precisely because it
  never blocks.
- **It is self-extinguishing**: each pass leaves fewer unready batches, and the
  `row_ready` epoch word is monotone (`CMPL:57-63`), so a positive observation is
  never revoked.

Cost, and it is the one real new traffic class: `Σ_passes |outstanding entries|`
relaxed **system-scope** loads. With entry-level readiness ≈ `(t/S)^1.5`, the
mean unready fraction over the epoch is ≈ 0.6, so `P` passes cost
`≈ 21,504 × 0.6 × P` loads. At `P = 20` that is **258 k system loads**, the same
order as `nc_arr`'s 524 k agent RMWs. exp_20 §4's mode-8 null (a 64× cut in the
event-poll rate moved M7 by nothing) is the licence to believe this is cheap,
but that null was measured on **agent**-scope loads of local memory; a
system-scope relaxed load must not hit a stale line (`KRN:507-510`), so it is a
different class. **`P` must be a swept config field, not a constant.**

**Sticky per-lane ready bits are deliberately NOT used in v1.** They would save
polls but force per-lane state to live *across* the reduce call, which is exactly
the register-lifetime hazard §8.3 exists to avoid. Re-poll from scratch each
pass; buy the polls back in S2 if the sweep is shown to be poll-limited.

---

## 4. Partition and scheduling policies

Throughput constant used throughout: **ρ = 6.1 GB/s per CTA** (§1.3), work
`W = 675 MB`, `S = 2,660 µs`. Shadowed work under a greedy scheduler with
capacity `Kρ` available across `[0,S]` and arrivals `W·(t/S)^8`:

```
t*/S = (KρS / 8W)^(1/7)          crossing where capacity starts to bind
shadowed = W·(t*/S)^8 + Kρ·(S − t*)
```

### P1 — pool-only pipelined combine

The `C` service CTAs reduce inline in the drain loop.

- **Primitive:** `roles.cuh` `is_service_cta` (shipped static-tail form,
  `ADP:292-297`); no new primitive.
- **Atomics/CTA/epoch:** unchanged (option C adds none) + 1 `claim` RMW per
  batch it wins.
- **Capacity arithmetic — this kills it.** `K = 16` ⇒ `Kρ = 98 GB/s`,
  `KρS = 260 MB`, `t* = 0.648 S`, shadowed = **112 MB (16.6 %)**. Covering the
  whole 675 MB in the M7 window needs `K ≈ 43`, and going `C = 16 → 43` removes
  27 of 240 compute CTAs. exp_11's measured linear M7 capacity law (+27 % for 62
  CTAs) gives **+338 µs of M7** to buy at most 430 µs of combine. Before
  interference. **Not viable.**
- **Failure mode:** a service wave inside a reduce is not dequeuing events, and
  `row_ready` publication is on the **critical path of seven other GPUs**.
  Starving the drain to feed the local combine exports the cost to peers, where
  the phase stamps cannot see it.
- **Under routing skew:** worse. Skew lengthens the tail of `row_rem` and widens
  the readiness spread, so the pool holds more half-finished state for longer.

### P2 — pool + progressive fall-through (exp_12 re-aimed) — **the shipped shape's natural extension**

Compute CTAs join the combine as they finish M7. Note what today's kernel
actually does, because it is the whole opportunity:

> `run_service` (`ADP:647-661`) claims ticket `k`; if `k < events_total` it
> **spins in `wait_event_nonempty` until event `k` is published**. Events publish
> in monotonic ticket order, so the 1,024 waves sit clustered just ahead of the
> publication frontier and **all of them exit only when M7's last task
> publishes.** A CTA that finished its M7 stripe early does not reach M8 early —
> it burns the interval spinning.

So there are **two distinct pools of free capacity**, and they should be built
and measured separately:

- **Term A — the fall-through spin.** M7's finish spread is ≈ 24.6 tasks/CTA
  with `gcount ∈ {1,2,3}` (mean 2.83), giving a ±1-task quantization plus a
  ~3.6 % stochastic spread ⇒ **130–210 µs** wide; the average CTA is idle ~80 µs.
  Available: `240 × 6.1 GB/s × 80 µs = 117 MB`, and at `t > 0.95 S` there are
  448 MB queued, so it is **not availability-limited**. ⇒ **−75 µs of combine.**
- **Term B — the pool shadow.** The C=16 pool sweeping across all of M7: 112 MB
  from the integral above, minus ~20 MB overlapping Term A's window ⇒ **90 MB**
  ⇒ a further **−57 µs**.

- **Primitives:** the shipped ticket loop plus `try_claim_unit` (§10, missing).
- **Failure mode:** if M7's finish spread is much narrower than modelled, Term A
  vanishes and P2 degenerates to P1. **S0 measures this.**
- **Under routing skew:** Term A *grows* (skew widens the M7 finish spread), so
  P2 degrades gracefully where P1 degrades badly. This is the strongest argument
  for P2 over P1 and it is the opposite of the usual skew story.

### P3 — backlog-driven join

A CTA finishing an M7 task joins the combine only when the ready-token backlog
exceeds a threshold.

- **Primitive:** would need a shared backlog counter — `counter.cuh`
  `counted_arrive_into` is the closest but its `expected` is a compile-time-fixed
  fan-in, not a running depth.
- **Atomics:** one RMW per ready-batch discovery to maintain the depth
  (~1,024/epoch) **on a single cell** — exp_07's measured trap (32 RMWs to 2
  lines were ≥10× slower than 32 RMWs to 32 lines). A single global depth counter
  polled by 1,024 waves is the same shape as the `pperr` word, which exp_20
  §4 measured as free — but writing it 1,024 times is not the same as reading it.
- **Verdict: reject for v1 on the readiness curve alone.** P3 self-tunes the
  *join threshold*, and §2 says the backlog is ~0 for 80 % of M7 and then floods.
  There is nothing to tune: the correct policy is "reduce whenever anything is
  ready", which is P2. Revisit only after an nc-major M7 makes the backlog a
  continuous quantity.

### Recommended: **P2, staged — Term A first (S1), Term B behind S0's evidence (S2)**

---

## 5. Work unit and interleave granularity

The chunk loop body of `k0p6_mps_m8_batch` (`KRN:530-604`) initializes
`acc[NT][8]` at its head (`KRN:536`) and stores `out` at its tail
(`KRN:592-603`). **There is no cross-chunk state.** So `(batch, chunk)` is a
legal work unit: 1,024 × 14 = 14,336 units of ~47 KB.

| unit | residence per unit | drain-latency inflation | polls |
|---|---:|---:|---:|
| batch (14 chunks) | ~30–60 µs at low contention | 30–60 µs | `F` per pass |
| chunk | ~2–4 µs | 2–4 µs | `F × 14` per pass if re-polled |

**v1 ships the batch unit** (the existing body called verbatim, poll once per
batch) and exposes the choice as a config bit, because the batch unit keeps the
reduce's register live range inside one call (§8.3) and the chunk unit's
14× poll amplification is the exact class §3.3 flagged as unpriced.

**If drain latency turns out to be the cost** (ablation A4), the fix is *not*
finer units — it is **reserving a drain-only subset of the pool**: of `C = 16`,
`C_flag` CTAs never reduce. Note this is a service-pool-internal split, which
looks like the struck axis A7 — it is not. A7 was struck because occupancy is one
block per CU so a service CTA can never steal an MFMA issue slot. The premise
here is entirely different: **queue-service latency isolation for a service whose
consumers are seven other GPUs.** Nothing in A7's strike bears on it.

---

## 6. Correctness — summary (full treatment in `protocol_review.md`)

- **Readiness predicate:** unchanged, `KRN:511-514` against `ADP:575-584`. The
  pipelined reduce evaluates exactly the predicate M8 evaluates today.
- **Acquire count:** today `NT8 = 1,024` `thread_acquire<system>` (d3,
  `KRN:526`), one per batch that passes its ballot. The pipelined reduce keeps
  **one acquire per batch**, so the count stays 1,024 — *provided the unit stays
  the batch*. A chunk unit with a per-unit acquire would take it to **14,336**,
  a 14× increase in the fence class exp_20 §4 named as the prime suspect. This
  is the strongest single reason the batch unit is the v1 choice.
- **Consume-and-zero:** safe, and pipelining does not weaken it — the zero is
  gated by exactly the same flag as the read, and that flag means "every
  accumulate into this row for this epoch is fabric-ACKed". Moving the zero
  *earlier* strengthens the cross-epoch edge, it cannot weaken it. Full proof in
  `protocol_review.md` §4, including the `KRN:1848` agent-scope release question
  (**inherited, not created** — and it is the one place this design asks for an
  independent fix).
- **Epoch tagging:** `row_ready` is monotone and never reset (§4 of the map), so
  a stale row reads `≤ epoch−1 < epoch32` and fails `>=`. Traced in
  `protocol_review.md` §5.
- **Deadlock:** the sweep never blocks, so the reduce can never hold back the
  drain that produces its own inputs. Proof in `protocol_review.md` §6.
- **Exactly-once:** the `claim` buffer (descriptor slot 59, `uint32[T_ext]`,
  128 KiB) is **allocated, M0-zeroed at `KRN:741-744`, and provably dead in mode
  12** (the `g == 1` fast path at `ADP:731-747` never reaches `ADP:764-765`).
  Entries `[0, 1024)` become the per-batch claim, at zero ABI cost and with the
  M0 zeroing already ordered before any reader by the M3/M5 grid barriers
  (`KRN:728-730` states the contract). **Claim strictly after readiness is
  established**, never before — `protocol_review.md` §8 mode 1 explains why the
  opposite order is the invisible-failure mode.

---

## 7. Cost model, with a pre-registered falsifier

Baselines: ratchet **6,685.5 µs**, `production` **7,715.6 µs** (0.8665×),
target 0.80× = **6,172 µs**. Phase stamps: `M7 = M7_DONE − M6_DONE`,
`combine = REDUCE_DONE − M7_DONE`, both printed by `[MPS TS DELTA]`
(`../CONTEXT/harness_recipe.md` §3). Both remain clean under this change:
`M7_DONE` is stamped only by **compute** CTAs (`KRN:1383`, inside the
`!is_service_cta` branch) and `REDUCE_DONE` by tid 0 of every CTA after the M8
block (`KRN:1816`). **Δcombine and ΔM7 are therefore separately measurable, as
required.**

### 7.1 Δcombine

| term | shadowed | Δcombine |
|---|---:|---:|
| A — fall-through spin reclaimed (S1) | 117 MB | **−75 µs** |
| B — C=16 pool shadow across M7 (S2) | +90 MB | **−57 µs** |
| drain-latency penalty exported to peers | — | **+20 µs** |
| **total** | **207 MB of 675 MB (31 %)** | **−112 µs** |

Post-M7 residue `675 − 207 = 468 MB` at 1.57 TB/s = 298 µs, against today's
430 µs. Band: **−60 to −250 µs** (dominated by the M7 finish-spread estimate,
which S0 measures).

### 7.2 ΔM7

| term | reasoning | ΔM7 |
|---|---|---:|
| capacity | `C` unchanged at 16 | **0** |
| Term A traffic | 117 MB in the last ~150 µs = ~780 GB/s local burst while M7's stragglers still issue MFMA; exp_20 §2b priced 64-CTA flat-out local read+write at **+115 µs** | **+40** |
| Term B traffic | 90 MB over 2,660 µs = 34 GB/s from 16 CTAs, 0.4 % of HBM; same exp_20 class scaled by rate and CTA count | **+25** |
| sweep polls | ~258 k system-scope relaxed loads; mode-8's agent-scope null says cheap, but the class differs | **+25** |
| **total** | | **+90 µs** |

Band **+20 to +180 µs**. Note the traffic terms are in exp_20 §2b's **+38 µs
read / +105 µs local-write** classes and **never** in the **+624 µs peer-write**
class — the pipelined combine reads and writes only local memory. That is the
single most important quantitative reason the objection in the brief is weaker
than it looks, and it composes with exp_20 §2a (the pool's payload *in situ*
costs +5.7 µs against a ±40 µs band) and exp_21 (rate shape beats volume:
capping outstanding RMWs at 8 recovered ~500 µs).

### 7.3 Net and falsifier

**Net ≈ −22 µs (0.3 %), band −230 to +160 µs.** Expected landing
**6,663 µs (0.864×)**.

**Pre-registered falsifier.** The mechanism is falsified if, with the full gate
ladder green, a **paired 5-rotation campaign** reads
`arm_p50_us(mps_mega) ≥ 6,685 µs`. Screens cannot adjudicate this: the
`ratio_vs_prod` 1σ is 0.52 % ≈ 35 µs and the smallest callable single-screen
delta is 2 % (`../CONTEXT/harness_recipe.md` §5). **Any conclusion here needs a
campaign or ≥ 6 repeated screens.**

Distinguishing the two failure meanings, decided on the *separate* deltas:

- **"Real but under-tuned":** `Δcombine ≤ −50 µs` **and** `ΔM7 ≥ |Δcombine|`.
  The mechanism works and the cost is placement/rate. Knobs worth spending on:
  sweep backoff, unit granularity, drain-only reserve, `C`.
- **"The boundary is closed at this task order":** `Δcombine > −50 µs` *even in
  the generous-capacity arm* (`C = 48`). Then readiness, not capacity, is the
  limiter — which is exactly what §2 predicts — and the only remaining lever is
  M7's task order. **Log as a ceiling on the boundary, not as a kill on role
  specialization**, and hand the file to M-series M8.

**S0's independent falsifier, evaluated before any mechanism is built:** if the
probe reports median token readiness `> 0.85 · S`, the shadow ceiling is `< 20 %`
of the combine and **S2 is not built tonight**.

---

## 8. Build plan

### 8.1 Mode number — **claiming 10 (diagnostic) and 11 (mechanism)**

`mode12_protocol_map.md` §6 is confirmed: grep of the tree finds **no reference
to mode 10 or 11 anywhere**, and both sit inside the existing validator bound
`if (c.mode > 13u) return false;` (`ADP:271`), so neither needs a validator
change.

> **COLLISION RISK, stated explicitly.** exp_25 (M6/M7 role split) is designing a
> new mode in parallel and 10/11 are the only free numbers. **exp_29 claims 10
> and 11.** If exp_25 has already taken one, exp_29's diagnostic moves to the
> remaining free number and its mechanism takes **14**, which requires the
> one-token change `mode > 13u → mode > 15u` at `ADP:271`. That line is in a file
> exp_25's implementer is also editing tonight, so **whoever lands second makes
> the edit and says so in their `result.md`.** Do not let two experiments both
> assume they own 10.

### 8.2 Files and sites

All additive. `moe_mps_adapter.cuh` and `k0pf6gm_device_tile_mps.hip` are owned
by the overnight loop (`aug10/CLAUDE.md` § Ownership); `n2_phase2_gm_mps.cpp`
needs **no change at all** (the epilogue keys off `mode_is_direct_accum`, and
`P2:272`'s throttle bit is read raw from the packed word).

| # | site | change | silent-trap? |
|---|---|---|---|
| 1 | `ADP:203-204` | add `kModePipelinedProbe = 10`, `kModePipelinedCombine = 11` | |
| 2 | `ADP:158-161` `mode_is_stream` | add 10, 11 | |
| 3 | `ADP:209-211` `mode_is_direct_accum` | add 10, 11 — else the M7 epilogue reverts to writing `part` (`P2:249-250`) and M8 takes the wrong branch (`KRN:1782`) | **yes** |
| 4 | `KRN:319-320` enqueue predicate | add 10, 11 — the in-source warning at `KRN:296-303` records that an open-ended `>= 2` silently enqueued events nobody consumed | **yes** |
| 5 | `KRN:350` `k0p6_mps_task_drain` | `m != 12 && m != 13` must become a `mode_is_direct_accum` test, else the new modes silently regain the per-task `vmcnt(0)` and the defer experiment is re-run under a different name | **yes** |
| 6 | `KRN:371` `k0p6_mps_task_done_maybe_defer` | same test, same trap | **yes** |
| 7 | `ADP:787` `env.mode == kModeRemoteAccum` | must accept 10/11 or the pool falls through to the *payload push* loop at `ADP:816-832` and pushes `part`, which is never written | **yes** |
| 8 | `ADP:249-286` `config_is_valid` | the `mode_is_direct_accum` branch already pins `g&0xF == 1`, forbids `pull_fallback`, allows the `0x20` throttle — inherited free; add validation for any new `flush_rows` reinterpretation | |
| 9 | `ADP:466-474` `wave_scratch` | add the sweep cursor; the `static_assert(sizeof == 528)` must move to 536 and the 7,488 B fit re-checked (`KRN:1460`) | |
| 10 | `KRN` M7.6 / `ADP` `run_service` | the sweep call site, at the **top** of the `while(true)` loop (§8.3) | |
| 11 | `KRN:1782-1799` M8 | add the `claim` check before reducing a ticketed batch | |
| 12 | `KRN` `K0P6_MPS_SRC_REV` | **bump** — `.cuh` edits do not invalidate the mori JIT cache; confirm a new `.hsaco` mtime through `readlink -f` + `stat -L` | **yes** |

New state, all at zero ABI cost:

- `claim[0 .. 1024)` (slot 59) — per-batch exactly-once claim. Dead in mode 12,
  M0-zeroed at `KRN:741-744`.
- `mps_state[4..7]` — free scalar words (`K0P6_MPS_ST_WORDS = 8`, only 0..3 used;
  all 8 zeroed at `KRN:745-748`). Used for the S0 histogram and for the S1/S2
  completed-batch counter that backstops failure mode #1.

### 8.3 Resource-tuple risk — the register lifetime question

Current tuple: **SGPR 106 / VGPR 256 / AGPR 256 / scratch 144 B / LDS 155,496 B**
at one block per CU. Allocation is static and per-kernel, so nothing here can
raise the VGPR count — **it can only raise scratch**, which is exactly how the
project was burned before (a 64 B/lane staging array took scratch 60 → 128 B and
ate the latency it was meant to hide).

**Can the reduce's live range be kept disjoint from the drain's? Yes, and the
structure that guarantees it is specific:**

1. **Call the existing `k0p6_mps_m8_batch<4, base_slot&, base_slot&, false,
   true>` instantiation verbatim.** It is already compiled into this kernel at
   `KRN:1795-1798`; its register footprint is already paid. A second call site
   of the same instantiation adds no new peak.
2. **Place the call at the top of `run_service`'s `while(true)` loop, before the
   ticket claim (`ADP:648`).** At that point the only live drain state is `s`
   (LDS, `KRN:1462-1463`) — `r`, `target`, `push_lead`, `ballot`, `k`, `ev` are
   all recomputed per iteration (`ADP:653-694`). **Zero register overlap by
   construction.**
3. **Make the reduce unit stateless at its boundaries.** It re-derives `lo2`,
   `fanout` and `(p,row)` from `pull_ptr`/`pull_src` on entry (`KRN:484-506`) and
   holds nothing on exit. This is why sticky ready-bits are excluded from v1
   (§3.3) — they are the one proposed value that would have to live across the
   call.
4. **The sweep cursor lives in LDS `wave_scratch`, not a register** (site 9).

**Predicted impact: scratch +0…+16 B, LDS +8 B (unchanged static frame — the
scratch unions with the dead M1 staging slice), SGPR +0…+2.** Mitigation if
scratch moves: hoist the sweep into its own `__noinline__` device function so the
allocator cannot interleave its live range with the drain's. **Gate before any
timing: `-Rpass-analysis=kernel-resource-usage`, plus the standing invariant
"zero scratch ops inside either MFMA K-loop" — which is automatic here, since
every line of this experiment is after M7.**

### 8.4 Staged build order — cheapest first

| stage | what | build | GPU | separately measurable |
|---|---|---:|---:|---|
| **S-1** | **free, and not exp_29's**: guard the `KRN:547` load with the wave-uniform `jb+jj < fanout[t]`, deleting ~154 MiB of issued loads. Check the ISA first (`global_load_dwordx4` count in the M8 body) — LLVM may already sink it | 0.5 h | 0.5 h | `combine` stamp alone |
| **S0** | **diagnostic only, mode 10, no mechanism.** The pool, between events, makes non-blocking sweep passes over a static batch stripe and records how many batches are ready when the event frontier (`ev_next / events_total`, a free progress clock needing no timestamp) first crosses 25/50/75/100 %. Four plain stores into `mps_state[4..7]`. Result discarded — read-only, correctness-preserving | 1.5 h | 1 h | **measures the prize *and* pre-prices the sweep's ΔM7** |
| **S1** | **Term A only**: the sweep+reduce runs on CTAs that have finished M7 (`!is_service_cta`), replacing the fall-through spin. Structurally near-zero ΔM7 risk | 4 h | 1.5 h | ΔM7, Δcombine |
| **S2** | **Term B**: enable the sweep on the reserved pool too (one config bit) | 1 h | 1 h | ΔM7, Δcombine |
| **S3** | tuning: backoff, unit granularity, drain-only reserve, `C` | 1 h | 2 h | per-knob |

**Honest total: 8 build-hours + ~6 GPU-hours to a campaign-quality answer.**
S-1 and S0 alone are **2 build-hours** and answer whether the rest is worth it.

---

## 9. Ablation ladder

**Matched-CTA-count comparison is mandatory and the reserve-but-do-nothing
control (mode 0) is forbidden** — it measures the capacity tax and is
structurally blind to interference, which `aug10/STATUS.md` records as the
dominant cost of exactly this mechanism.

The correct control here is **the same build, same `C`, same mode, sweep
disabled by a config bit** — a same-run paired arm at identical CTA count.

| # | arm | control | isolates | pre-registered expectation |
|---|---|---|---|---|
| A0 | mode 12, `C=16 g=33 flush=16` | — | the denominator | 6,685 µs |
| A1 | mode 11, `sweep_off` | A0 | cost of the restructure alone (claim check, extra branch, `wave_scratch` growth) | ≤ ±20 µs; anything larger is a build defect, stop |
| A2 | mode 11, sweep on, `!is_service_cta` (S1) | A1 | **Term A** | Δcombine −40…−110, ΔM7 +10…+70 |
| A3 | mode 11, sweep on, all CTAs (S2) | A2 | **Term B** | Δcombine a further −20…−80, ΔM7 +10…+50 |
| A4 | backoff ∈ {0, 4, 16, 64} | A3 at backoff 0 | is the **poll** the cost? | flat, by analogy with mode 8; a rising curve promotes the system-scope poll to prime suspect |
| A5 | unit ∈ {batch, chunk} | A3 | is **drain latency** the cost? | chunk wins only if A3's ΔM7 is drain-shaped; it costs 14× the acquires |
| A6 | drain-only reserve ∈ {0, 4} of `C` | A3 | **service-latency isolation** (not A7 — see §5) | helps only if peers' combines are being starved |
| A7 | `C ∈ {16, 32, 48}` at the best point | each other | shadow capacity vs capacity tax | interior optimum near 16–32; `C=48` should lose on M7 by exp_11's linear law |
| A8 | S0's readiness histogram at `C ∈ {16, 48}` | each other | is readiness or capacity the binding constraint? | **the decisive arm** — if the histogram is `C`-invariant, capacity is not the limiter and §2's ceiling is confirmed |

Discipline: one variable per arm; sub-5 % deltas re-run or paired; screens rank
against `production` only (`pf6gm_mega` is 1.34 % σ at screen resolution and
cannot be a denominator); the full gate ladder — `[MOK GATE]`,
`[MARK] control_fails=True`, `[MPS SOAK] 600/600 pperr=0` — **before** any timing,
every arm, no exceptions.

---

## 10. Primitives

### 10.1 Used

| concern | primitive | header |
|---|---|---|
| role split | shipped static-tail `is_service_cta` (`ADP:292-297`); `finish_order_partition` remains rolled back (`DESIGN_MPS.md:118-125`, +24 B/lane spill) | `roles.cuh` |
| readiness observation | `bounded_poll_relaxed_into<system>` via `poll_epoch_system` (`HKA:151-160`) | `completion.cuh` |
| monotone epoch compare | `epoch_ready` (`CMPL:57-63`) | `completion.cuh` |
| publication | `publish_epoch<agent|system>` (`ADP:579-583`) | `completion.cuh` |
| payload acquire | `thread_acquire<system>` via `acquire_payload_system` (`KRN:526`) | `sync.cuh` |
| arrival counting | `fetch_add_acq_rel<agent>` (`ADP:720-722`) | `counter.cuh` (raw form; see 10.2.4) |
| peer addressing | `translate_peer` / `peer_ptr` (`HKA:50-62`) | `peer.cuh` |

### 10.2 Missing, wrong-shaped, or forced awkward — the findings

1. **There is no non-blocking probe.** `completion.cuh` offers
   `bounded_poll_relaxed_into` (spin-with-limit) and
   `bounded_observe_acquire_into` (spin, then acquire). It offers **nothing** for
   "tell me whether this is ready right now and do not spin". Every caller that
   wants to interleave polling with other work must open-code it, and open-coding
   a poll is exactly how `KRN:507-510`'s stale-line bug arrived. This experiment
   cannot be written idiomatically without it. Proposed, one line each:

   ```
   template<memory_scope Scope> [[nodiscard]] bool probe_epoch_relaxed(
       const std::uint32_t* signal, std::uint32_t expected);
   ```

   plus a wave form returning a lane mask, since the actual predicate is
   `__ballot(ready) == ~0` over 8 lanes per token. **This is the single
   highest-value one-line addition the header set is missing**, and it
   generalizes: any interleaved-duty role needs it.

2. **There is no idempotent work claim.** exp_12 already named the gap ("a shared
   monotonic ticket over a bounded work set — the single most reusable thing this
   campaign has produced"). exp_29 needs its sibling: **exactly-one-owner for a
   unit that two different phases may both reach.** `lifetime.cuh` has
   `bounded_wait_slot_reusable_into` / `drain_and_retire_slot` for *reuse
   credit*, which is a different relation. Proposed:

   ```
   template<memory_scope Scope> [[nodiscard]] bool try_claim_unit(
       std::uint32_t* claim_cell);          // fetch_add == 0
   ```

   Trivial to write, and its absence is why this design has to explain at length
   (in `protocol_review.md` §8) why claim-order matters — a primitive with the
   ordering baked into its contract would make that unwritable.

3. **`roles.cuh` has no vocabulary for a role with two duties.** `role_partition`
   is a static two-way split of the grid. What this design actually needs is a
   *duty cycle*: "serve a latency-critical queue whose consumers are other GPUs;
   when idle, do bulk local work; never let the bulk work delay the service." The
   library can express *which* CTAs, never *what a CTA does when idle*. That is
   the shape COMET's released code uses and the shape every experiment since
   exp_12 has hand-rolled. Worth a named primitive or, at minimum, a documented
   pattern in `roles.cuh`.

4. **`counter.cuh`'s `counted_arrive_into` cannot express our arrival at all.**
   Its contract fixes `expected` as "the fixed producer count for this dependency
   key" and derives epoch/ordinal by division. Ours is
   `target = row_rem[r]`, a **per-key runtime value read from memory**
   (`ADP:690-691`), different for every row and zeroed by the publisher
   (`ADP:587`). So the shipped protocol open-codes `fetch_add_acq_rel` and the
   primitive sits unused on the hottest counter in the kernel (524,288
   RMWs/rank/epoch). A `counted_arrive_dynamic_into(cell, target, result)`
   overload would make the real protocol expressible.

5. **`mode`-as-diagnostic-selector, use #7.** exp_20 flagged this at use #5 and
   asked for a `roles.cuh` vocabulary for "run this role's body in a diagnostic
   variant". exp_21 made it #6. exp_29 would make it #7 and #8. The pattern works
   and costs nothing at runtime, but it is protocol expression accumulating in an
   operator header, and the site list in §8.2 — **five of twelve edit sites are
   flagged as silent traps, all of them mode-predicate updates** — is the
   measurable cost of not having the vocabulary. That list is the strongest
   evidence yet for the request.

**Negative finding for the mandate, logged honestly:** the winning discovery
policy (§3.3) adds **no primitive at all** — it is M8's existing poll with the
spin deleted. The library did not stand in the way; it simply had nothing to
offer, and the right answer was to reuse a call the kernel already makes. A
kernel that teaches us the library was *not needed* is still a result.

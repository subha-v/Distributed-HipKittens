# exp_05 — release granularity (E3): protocol review

Read-only memory-ordering / epoch / lifetime analysis, written **before** the
first GPU run of any batched-release candidate. No source was modified, no build
was run, no GPU job was launched.

Reviewed revision: working tree of `GEMM-RS` at
`C:\Users\subvadla\repos\Distributed-HipKittens-GEMM-RS`.

Sources read in full: `distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp`,
`gemm_rs_mi300x_hk_adapter.cuh`, `gemm_rs_mi300x_constants.cuh`,
`gemm_rs_mi300x_host_abi.hpp`, `gemm_rs_hk_adapter.cuh`,
`include/cdna3/ops/group/distributed/{sync,completion,lifetime,counter,packet,peer}.cuh`
and `detail/config.cuh`, `MI300X_DESIGN.md`, `overnight/HANDOFF.md`,
`overnight/experiments/LESSONS.md`, plus the harness files that define the gates
(`harness_lib.py`, `m3_correctness.py`, `m4_controls.py`, `m5_soak.py`,
`exp_ablation.py`) and `overnight/RESULTS.md`.

---

## 0. Verdict up front

**APPROVE WITH CONDITIONS.**

- **No deadlock exists and none can be introduced by coarsening.** The wait-for
  relation is strictly decreasing in launch epoch (§2.4). Batching adds no edge
  to that graph; it can only delay the satisfaction of an existing edge. Every
  wait is additionally bounded and fail-closed, so even a wrong analysis costs a
  timeout, not a wedged node.
- **There is exactly one visibility hole, and it is the whole change.** The
  ready flag is published *inside* the per-tile path
  (`gemm_rs_mi300x.cpp:302-321`), immediately after the per-tile release. So E3
  is not "move a fence"; it is "convert one release + one publication batch per
  tile into one release + one publication batch per **group**". Any
  implementation that coarsens the release while leaving publication per tile is
  **silently, non-deterministically wrong**, and §3 shows none of the three
  existing negative controls would catch it.
- **Four corrections to the premise, all of which change the decision** — see
  §0.1. The most important: on three of the six scored shapes a producer CTA
  emits **exactly one tile**, so their release count cannot be reduced at all,
  and the best-case geomean gain of E3 is **~3%**, not 9%.

---

## 0.1 Premise corrections (read these before sizing the experiment)

**P1 — the premise is right about the release, and incomplete about the flag.**
`m3::release_payload_system()` really is issued once per output tile, inside the
tile loop (`gemm_rs_mi300x.cpp:301`, loop at `:191`), and it really is
`s_waitcnt vmcnt(0)` + barrier + leader `buffer_wbl2 sc0 sc1` + barrier
(`include/cdna3/ops/group/distributed/sync.cuh:166-174`; ISA confirmed in
`overnight/RESULTS.md:77-78`). But the ready publication is in the same per-tile
block (`gemm_rs_mi300x.cpp:302-321`), so coarsening the release **requires**
coarsening the publication. This is stated as an explicit contract by the
primitive itself: *"This separation permits one producer_drain_release() to
cover several completion cells. Calling it without the release is a protocol
error."* (`completion.cuh:65-75`).

**P2 — half the shapes cannot benefit at all.** With the uniform `NR = 32` now
in the shape table (`gemm_rs_mi300x_host_abi.hpp:53-60`), `num_gemm_ctas = 272`
and the tile loop is `for (t = pid; t < tiles; t += 272)`
(`gemm_rs_mi300x.cpp:191`). Tiles per producer CTA:

| # | shape | gemm_tiles | tiles per CTA | releases removable |
|---|---|---:|---|---|
| 1 | 64×7168×18432 | 56 | **1** (216 CTAs idle) | none |
| 2 | 512×4096×12288 | 512 | 2 / 1 | ≤ 1 of 2 |
| 3 | 2048×2880×2880 | 192 | **1** (80 idle) | none |
| 4 | 4096×4096×4096 | 256 | **1** (16 idle) | none |
| 5 | 8192×4096×14336 | 512 | 2 / 1 | ≤ 1 of 2 |
| 6 | 8192×8192×29568 | 1024 | 4 / 3 | ≤ 3 of 4 |

(`gemm_tiles = (M/BM)·ceil(N/BN)`, `MI300X_DESIGN.md:69-76` and
`host_abi.hpp:139`.) Applying the measured release deltas
(`HANDOFF.md:71-78`) with a perfect `1 − 1/k` amortization gives a best case of
116.8→108.8, 773.1→737.9, 2861.7→2674.0 and **no change on shapes 1, 3, 4** —
a geometric-mean improvement of **3.0%**, and that is an upper bound. Shapes 1,
3 and 4 therefore act as free built-in controls: the batched build must be
within noise **and bit-identical** on them.

**P3 — the drain is not recoverable; only the writeback and the barriers are.**
The next tile's mainloop re-imposes the drain a few instructions after the tile
boundary: `G::load` lowers to `global_load_dwordx4` → **`s_waitcnt vmcnt(0)`** →
`ds_write_b64` (recorded ISA, `experiments/exp_03_mainloop/plan.md:50`), and on
gfx9 `vmcnt` retires in order, so any wait sufficient for those loads also
drains the still-outstanding peer stores of the previous tile. **Coarsening the
release cannot keep peer stores in flight across a tile boundary while the
mainloop still contains a blanket `vmcnt(0)`.** Consequences: (a) the hoped-for
secondary win — deeper XGMI queue depth against the measured 127 GB/s of 448
(`RESULTS.md:188-190`) — does **not** materialize until E1(b) replaces the
mainloop's blanket wait with counted `s_waitcnt`; (b) the measured 250.3 µs is
therefore mostly `buffer_wbl2` + the two `__syncthreads()`, which *is* fully
amortizable, so the 3% bound in P2 is realistic rather than optimistic.
**Recommendation: run E3 after E1(b), or run it twice — before and after.**

**P4 — "per CTA per epoch" is unbounded and must not be implemented as such.**
The epoch is per *launch*, not per tile (`gemm_rs_mi300x.cpp:162-163`, hoisted
out of the tile loop), so "per CTA per epoch" means "one release at the end of
the CTA's entire tile list". On the scored table that is at most 4 tiles, but
the eleven official correctness shapes (`m3_correctness.py:33-45`) all fall on
the generic row `BM=32, BN=64, NR=24` (`host_abi.hpp:64`), where
8192×8192×28672 gives `(8192/32)·(8192/64) = 32768` tiles over 280 producer CTAs
= **117 tiles per CTA**. Unbounded batching there would defer every publication
to the end of the GEMM (reducers spin through the whole mainloop; still inside
the 2 000 000-spin bound, so it degrades rather than fails) and would make any
"remember the group" implementation spill to scratch. The grouping constant must
be a **compile-time cap**.

**P5 (sizing hygiene, not protocol).** The attribution table's shape-6 `full` of
2861.7 µs (`HANDOFF.md:78`) matches the `NR=8` reducer split (2850.2,
`RESULTS.md:208`), not the `NR=32` split now in the source (2632.1). The
250.3 µs release figure is therefore from a superseded configuration.
Re-run `exp_ablation.py` on the current source before quoting E3's expected
gain, and measure E3 against whatever `exp_02_nr32` lands, not against the
285.02 µs ratchet.

---

## 1. The existing protocol, exactly

### 1.1 Every release and every acquire site

| # | Site | File:line | Lowering | Orders |
|---|---|---|---|---|
| R1 | `m3::release_payload_system()` — producer, once per tile | `gemm_rs_mi300x.cpp:301` → `hk_adapter.cuh:36-38` → `gemm_rs_hk_adapter.cuh:58-60` → `sync.cuh:166-174` | `s_waitcnt vmcnt(0)` (all threads); `__syncthreads()`; leader `__builtin_amdgcn_fence(RELEASE,"")` = `buffer_wbl2 sc0 sc1` + `s_waitcnt vmcnt(0)`; `__syncthreads()` | every peer/local payload store of this tile's bands **before** the ready publications that follow at `:302-321` |
| A1 | `m3::acquire_payload_system()` — reducer, once per output tile | `gemm_rs_mi300x.cpp:359` → `hk_adapter.cuh:40-42` → `gemm_rs_hk_adapter.cuh:62-65` → `sync.cuh:177-181` | `__syncthreads()`; `__builtin_amdgcn_fence(ACQUIRE,"")` = `buffer_inv` | the eight observed `ready >= e` **before** any of the 8-source payload loads at `:366` |
| R2 | `m3::finish_tile_consumption()` — reducer, before the credit | `gemm_rs_mi300x.cpp:378` → `hk_adapter.cuh:66-68` → `gemm_rs_hk_adapter.cuh:108-110` → `sync.cuh:184-191` | `s_waitcnt vmcnt(0)`; `__syncthreads()` | every consumer read of the eight slots **before** `credit[me][lrow][col] = e` at `:389` |
| E1 | `m3::next_epoch_workgroup()` ×2 | `:162-163`, `:331-332` → `counter.cuh:30-46` | leader `global_atomic_add` (relaxed, agent); barrier; agent-relaxed load; barrier | not an ordering site; a convergent per-CTA launch ordinal |
| P1 | `publish_band_epoch` (ready) | `:315-319` → `hk_adapter.cuh:44-50` → `gemm_rs_hk_adapter.cuh:78-85`, `:67-76` → `completion.cuh:71-75` → `sync.cuh:39-51` | `__hip_atomic_store(RELAXED, SYSTEM)` (AGENT when `dest == me`) after peer translation | **relaxed by design**; it carries no ordering of its own and is correct *only* because R1 preceded it |
| P2 | `publish_reuse_credit` (credit) | `:389-393` → `hk_adapter.cuh:70-76` → `gemm_rs_hk_adapter.cuh:112-119` | same relaxed store form | correct only because R2 preceded it |
| W1 | `wait_band_epoch` (reducer, 8 lanes) | `:351-354` → `gemm_rs_hk_adapter.cuh:87-96` → `completion.cuh:103-117` | `__hip_atomic_load(RELAXED, SYSTEM)` spin, `s_sleep(4)` backoff, bounded | **relaxed observation only**; A1 supplies the acquire after CTA convergence |
| W2 | `wait_reuse_credit` (producer, `bands` lanes) | `:233-236` → `gemm_rs_hk_adapter.cuh:98-106` → `lifetime.cuh:22-41` | relaxed spin for `credit >= e-1`; `e <= 1` returns immediately | control observation, not a payload acquire |

Two structural notes that matter for E3:

- **R1 is the only release on the producer side, and it is leader-only inside a
  barrier sandwich.** Its soundness chain is: every thread's own stores are
  complete at the memory system (`vmcnt(0)`), the barrier orders that before the
  leader's L2 writeback, and `buffer_wbl2` is an L2-wide operation covering
  lines dirtied by any thread of the workgroup — valid because a workgroup is
  resident on one CU and therefore one XCD's L2. Batching reuses this chain
  verbatim.
- **Both publications are relaxed stores.** There is no per-cell release
  anywhere. The protocol is already "one release, many publications" at band
  granularity (`MI300X_DESIGN.md:130-132`); E3 only widens the many.

### 1.2 Each signal word: writer, reader, order, dependency

Geometry: `sig` is one symmetric u32 allocation of
`SIGNAL_GUARD_U32 + ready_words + credit_words`
(`constants.cuh:26-29`, `host_abi.hpp:153-161`), zeroed exactly once before the
first launch and never reset (`host_abi.hpp:193-195`, `MI300X_DESIGN.md:122`).

| Word | Written by | Written where | Read by | Read where | Ordering the pairing needs |
|---|---|---|---|---|---|
| `ready[src][lrow][col]` at `GUARD + ready_idx(src,lrow,col)` (`constants.cuh:55-57`) | rank `src`'s producer CTA leader, value `ep` | `:315-319`, once per (tile, band) | rank `dest`'s reducer, lanes `s ∈ [0,8)` | `:350-356` | writer: payload stores → **R1** → relaxed store. reader: relaxed poll → converge → **A1** → payload loads. Break either half and the reducer may read a slot that still holds epoch `e-1` bytes. |
| `credit[owner][lrow][col]` at `GUARD + ready_words + credit_idx(owner,lrow,col)` (`constants.cuh:58-60`) | rank `owner`'s reducer leader, value `ep`, written into **every** source rank's `sig` | `:379-394` | rank `src`'s producer, lanes `< bands`, requires `>= ep-1` | `:228-238` | writer: all consumer loads → **R2** → relaxed store. reader: relaxed poll; no acquire needed because it grants permission to *write*, not to read. |
| guard `[0, 64)` | nobody | — | nobody | — | asserted still zero by `check_signals` (`harness_lib.py:350-351`) |
| `ep_cell[EP_GEMM_U32 + pid]`, `ep_cell[EP_RED_U32 + pid_r]` | the owning CTA's leader, +1 per launch | `counter.cuh:33-35` | the same CTA, all threads | `counter.cuh:37-38` | agent scope only; the cell is CTA-private and LOCAL (`constants.cuh:31-35`, `MI300X_DESIGN.md:114-117`). Two barriers make the read convergent. |
| `err[0]` bits 25/26 | any timing-out lane, `atomicOr` | `:237`, `:355` | every CTA at entry and after each wait | `:165`, `:240`, `:333`, `:358` | agent-relaxed, sticky for the life of the allocation |

Index disjointness — that `(src,lrow,col)` and `(owner,lrow,col)` name exactly
one producer band and exactly one reducer tile — is design gate 7/(g)
(`MI300X_DESIGN.md:246-253`) and is what makes "one release covers many cells"
legitimate in the first place.

### 1.3 The exact lifetime of a heap slot

Slot = the byte range `c_heap[me][lrow*EB .. lrow*EB+EB)[col*BN .. )` **on rank
`dest`**, addressed by the producer through
`hk_gemm_rs::peer_view(g.c_heap, heap_peers, dest)` (`:259-262`,
`gemm_rs_hk_adapter.cuh:39-45`). Per `(dest, src=me, lrow, col)` there is
exactly one writer (rank `me`'s producer) and exactly one reader (rank `dest`'s
reducer for output tile `(lrow, col)`).

- **Producer may write slot S for epoch E** after `wait_reuse_credit` observes
  `credit[dest][lrow][col] >= E-1` (`:233-236`, `lifetime.cuh:22-41`), or
  immediately if `E <= 1` (the slot has never held a live payload). The wait is
  on **leader lanes only** and is broadcast fail-closed: `atomicOr` + barrier +
  `error_bit_set` (`:237-240`), so no lane issues a payload store after a failed
  wait (gate 12/14).
- **Reducer may read slot S for epoch E** after observing
  `ready[src][lrow][col] >= E` for all eight `src` (`:350-356`) **and** after
  A1 (`:359`). The eight-lane poll converges at the barrier `:357` before A1, so
  the acquire covers all readers.
- **What stops a producer overwriting an unconsumed slot:** the credit path.
  `credit` for epoch `E-1` is published only after R2, which is after the unique
  consumer's final read of epoch `E-1` (`:378-393`). `MI300X_DESIGN.md:240-244`
  states the invariant: monotonic readiness is never used to justify a write.
- **What stops a reducer reading an unfinished slot:** the ready path, R1→A1.

Lifetime of the slot for epoch `E` is therefore the interval
`[publish ready = E, publish credit = E]`, and the next write is gated on the
close of that interval.

### 1.4 Where `vmcnt(0)` appears and what it drains

1. `sync.cuh:169`, inside R1, executed by **every thread** of the producer CTA.
   It drains that thread's entire outstanding VMEM queue: the 16-byte peer
   packet stores (`packet.cuh:45-55`, `gemm_rs_hk_adapter.cuh:47-51`), the
   bounded scalar tail stores (`hk_adapter.cuh:197-207`), and — when `dest ==
   me` — local stores. `vmcnt` is destination-agnostic; there is no per-peer
   counter.
2. Inside the leader's `release_fence<system>` at `sync.cuh:172`, after
   `buffer_wbl2 sc0 sc1`, waiting for the writeback itself
   (`RESULTS.md:77-78`).
3. `sync.cuh:186`, inside R2 on the reducer, draining the eight source loads
   and the local output stores before the credit.
4. **Not a protocol site, but decisive for E3:** the mainloop's own
   `s_waitcnt vmcnt(0)` between `global_load_dwordx4` and `ds_write_b64` in
   `G::load` (`experiments/exp_03_mainloop/plan.md:50`). See P3.

`__syncthreads()` is **not** a drain here: on gfx9 a workgroup-scope fence does
not require `vmcnt` (the L1 is per-CU and keeps workgroup order), so the
staging-loop barriers at `:270`/`:294` emit `lgkmcnt(0)` only. **This is the one
ISA fact in this review that the implementer must confirm from the disassembly
before trusting the gain estimate** — if those barriers do carry `vmcnt(0)`,
E3's recoverable pool shrinks again.

---

## 2. Is coarsening safe, and under exactly what condition?

### 2.1 Yes — and the condition is that publication moves with the release

Confirmed from source: the ready flag is published inside the per-tile path,
`gemm_rs_mi300x.cpp:302-321`, guarded by `threadIdx.x == 0`, immediately after
R1 at `:301`. So for tiles T1, T2, T3 released only after T3, the flags for T1
and T2 **must** also move after that release. The change is:

> one release + `bands` publications **per tile**
> → one release + `Σ bands` publications **per group**

Nothing weaker is safe. In particular, "keep the flags where they are and just
skip some releases" produces a slot that is published while its bytes are still
dirty in the producer XCD's L2 — the reducer's `buffer_inv` then pulls the
**previous epoch's** line from HBM. Under unchanged inputs that is the correct
answer (the exact blindness the ledger records for `CTRL_REROUTE_SLOT`,
`HANDOFF.md:213-216`), so this bug can survive a full correctness sweep.

### 2.2 The correct ordered sequence

Per group of `k >= 1` tiles emitted by one producer CTA at epoch `ep`:

1. for each tile `j` in the group, in order:
   a. decode `(tm, tn)` from `t`; mainloop → `C_accum`;
   b. `wait_reuse_credit(credit[dest_b][lrow_b][tn], ep)` on lanes `< bands`;
      `__syncthreads()`; fail-closed check;
   c. for each band, for each staging window: stage → `__syncthreads()` →
      emit → `__syncthreads()`;
2. **exactly one** `m3::release_payload_system()`;
3. leader only: for **every** (tile, band) in the group, in any order,
   `m3::publish_band_epoch(..., ready_idx(me, lrow, tn, ...), ep)`.

Step 2 is whole-CTA convergent (`sync.cuh:171,173`) and must not be placed
inside any divergent region. Step 3 must not be interleaved with step 1 for a
later tile.

### 2.3 Grouping across tiles that target different peers — safe

The requirement is only *per destination*: rank `D`'s reducer must not observe
`ready` from source `me` before the payload `me` sent to `D`. It never reads any
other rank's heap. R1 supplies a strictly stronger, destination-agnostic
ordering:

- `s_waitcnt vmcnt(0)` covers every outstanding VMEM op of every thread
  regardless of target, and on gfx9 `vmcnt` retires in order, so age is
  irrelevant — a store issued three tiles ago is covered exactly as well as one
  issued three instructions ago.
- `buffer_wbl2 sc0 sc1` is an L2-wide writeback on the executing XCD; it does
  not discriminate by destination. All of the CTA's threads share that L2.
- The relaxed publications that follow are system-scope (agent-scope only for
  `dest == me`, `gemm_rs_hk_adapter.cuh:67-76`), which is the correct
  strengthening in the one case where the payload store was local.

**Batching introduces no new hardware assumption.** It relies on exactly the
properties the per-tile release already relies on, applied to a larger set of
stores. That is the core safety argument, and it is why I could not break this
change on the ordering axis.

One consequence worth writing into the design note: this also makes **per-peer
batching pointless** — splitting the group by destination would mean *more*
releases, not fewer, for zero ordering benefit. Reject that variant.

### 2.4 The credit path, deadlock, and livelock — no cycle, and none can be added

This is the risk the dispatch was most worried about. It does not exist, and the
reason is structural rather than empirical.

Build the wait-for graph over nodes `P(r, e)` (rank `r`'s producers in launch
`e`) and `R(r, e)` (rank `r`'s reducers in launch `e`):

- `R(d, e) → P(s, e)` for all `s`: the reducer's only wait is
  `ready[s][lrow][col] >= e` (`:351-354`), published by launch `e`'s producers.
- `P(r, e) → R(d, e-1)`: the producer's only wait is
  `credit[d][lrow][tn] >= e-1` (`:233-236`, `lifetime.cuh:30`), published by
  rank `d`'s reducers in launch `e-1` (`:389-393` writes value `ep` into every
  source rank's `sig`; the index the producer polls,
  `credit_idx(dest, lrow, tn)`, is exactly the cell rank `dest` writes to us —
  the two index expressions match, `constants.cuh:58-60` vs `:392`).
- `P(r, e) → P(r, e-1)` and `R(r, e) → R(r, e-1)` by per-rank stream order: one
  launch per call (`:404-424`, gate 16) and a rank's grid fully retires before
  its next launch begins.

Every edge either keeps the epoch and moves `R → P` within the same launch, or
strictly decreases it. There is no `P → R` edge inside an epoch. Composing any
cycle would require an epoch to increase along some edge, which no edge does.
**The graph is acyclic for every release granularity, because batching changes
only *when* a `R(d,e) → P(s,e)` edge is satisfied, never *which* edges exist.**
Adding no edges to an acyclic graph cannot make it cyclic.

Two sub-cases that deserve to be stated explicitly, because both are real
situations this kernel runs in:

- **`MI300X_DESIGN.md:204-210` overstates its case, and it does not matter.**
  §7(b) claims "calls are stream-serialized: launch `e` begins only after launch
  `e-1`'s grid has fully retired", concluding producers never wait at all. That
  is true *per rank*, not across ranks: the benchmark path deliberately queues
  launches with no host sync between them (`harness_lib.py:410-421`), and its
  own comment says "the directed retirement credits are exactly what let rank A
  start epoch `e` while rank B is still finishing `e-1`". So rank A's producers
  **do** genuinely block on rank B's epoch-`(e-1)` reducers in pipelined mode.
  The acyclicity argument above does not depend on §7(b) and survives this.
- **The reducer-waits-on-a-credit-starved-producer scenario.** A reducer waiting
  for tiles from a producer that is itself waiting on credits is
  `R(d,e) → P(s,e) → R(d',e-1) → P(*,e-1)`, all strictly decreasing, and
  `P(*,e-1)` completed before `P(s,e)` could start on that rank. It terminates.

Finally, every wait is bounded (`spin_limit`, default 2 000 000 with `s_sleep(4)`
backoff ≈ hundreds of ms, `harness_lib.py:121`, `completion.cuh:113-115`) and
fail-closed. **The worst case for a wrong batched release is a bounded timeout
and a poisoned launch, never a hung node** — which is exactly the property that
makes it acceptable to run this experiment on shared hardware.

*Second-order performance risk, not a correctness risk:* deferring publication
lengthens the steady-state pipeline cycle `P(e) → R(e) → credit → P(e+1)`. The
`m7_bench` number is a pipelined throughput measurement, so E3 could improve
single-shot latency and simultaneously **hurt** pipelined throughput. Measure
both (`time_pipelined` and `time_single_shot`, `harness_lib.py:429`, `:575`) and
report both.

### 2.5 Epoch boundaries — there is no in-launch boundary to cross

`ep` is computed once, before the tile loop (`:162-163`), and is loop-invariant.
Every tile a CTA emits in one launch carries the same epoch. Batching therefore
never groups across an epoch boundary; the only boundary is the launch boundary,
and the requirement there is simply that **all publications complete before the
kernel exits**, which the structure in §4.3 guarantees.

The failure if it does not: the unpublished slot's reducer times out (bit 26),
returns before publishing its credits (`:358`), and every rank's producers in
launch `e+1` then time out on that missing credit (bit 25). The error bit is
sticky (`:165-166`) and the harness never clears `err`
(`harness_lib.py:164`, allocated once, never zeroed), so the whole instance is
poisoned from then on. Loud, safe, and consistent with the standing rule that a
nonzero error bit is terminal.

### 2.6 The tail — where the bug will be, and how to make it impossible

`tiles % num_gemm_ctas != 0` on five of six scored shapes, so CTAs emit
different tile counts (P2), and on shapes 1/3/4 **most CTAs emit zero tiles**.
Three distinct tail bugs are available:

1. **Forgotten final flush.** Pending group at loop exit is never released or
   published. Cascades as in §2.5. Detectable (bit 26 + `check_signals`).
2. **Flush on an empty group.** A CTA with zero tiles executing a release and/or
   publishing a stale/garbage `(dest, lrow, tn)` — this one publishes `ready = e`
   for a slot nobody wrote, and the reducer then reads epoch-`(e-1)` bytes.
   **Silent, tolerance-invisible under unchanged inputs.** With 216 zero-tile
   CTAs on shape 1 this is not a corner case.
3. **Emit/publish index drift.** The deferred publish loop recomputes
   `(tm, tn)` and gets a different tile than the one emitted. If the recomputed
   set is a permutation of the emitted set, `check_signals` and the timeouts
   both stay silent and the corruption is purely numeric.

The mitigation is structural, not vigilance: put the release and the publish
**inside** an outer group loop whose body cannot execute with zero tiles, and
**recompute** the tile decode with the *same function* the emit path used
(§4.3). Do not journal the group in LDS: the staging buffer aliases the A/B
double buffers at `&__shm[0]` (`:170`) and the next tile's mainloop overwrites
them, and at `BM=BN=256, BK=32` the LDS is already at the 65536 B ceiling
(`:132-133`). Do not journal it in registers either: the 256×256 rows already
spill 2 VGPRs with 12 B of scratch (`RESULTS.md:57-58`), and a dynamically
indexed array of up to 117 triples (P4) demotes straight to scratch.

### 2.7 Hazards I checked and cleared

- **LDS write-after-read across the deferred window.** Not a hazard. The emit
  reads staging into VGPRs; the data dependency forces `lgkmcnt(0)` before the
  store issues, and the barrier at `:294` orders all lanes' reads before any
  next write. The release's `vmcnt(0)` is *not* what protects the staging
  buffer, so removing it per-tile does not expose it.
- **Store-source VGPR reuse.** Handled by the compiler's `expcnt` tracking. It
  stops being handled if anyone hand-rolls the peer stores in inline asm —
  don't.
- **Intra-group WAW on a heap slot.** Impossible: distinct tiles of one CTA map
  to distinct `(dest, lrow, tn)` (`MI300X_DESIGN.md:246-253`), so a group's
  payload regions are pairwise disjoint and one trailing drain suffices. This
  becomes a live constraint if anyone later adds split-K or reorders tiles, so
  it is listed as an invariant in §4.4.
- **Credit wait for tile `j+1` while tile `j`'s stores are undrained.** Safe:
  the credit is a relaxed load of an unrelated cell, and it grants permission to
  write a *different* slot.
- **More outstanding VMEM ops.** Backpressure, not deadlock; `vmcnt` saturation
  stalls issue.

### 2.8 One pre-existing observation, out of scope for E3

`error_bit_set` (`gemm_rs_hk_adapter.cuh:121-130`) is wave-uniform but **not
CTA-uniform**: each wave's lane 0 loads `err` independently, so two waves can
disagree if a peer sets the bit between their loads, and the early `return`s at
`:240`/`:358` then execute for some waves and not others. On CDNA the barrier
count decrements on wave termination, so this degrades to a partially-abandoned
CTA rather than a hang, and it only occurs on the already-failing path. E3 does
not change the number or placement of these checks. Flagged for the ledger, not
a blocker here.

---

## 3. Failure modes and the negative controls

### 3.1 Do the three existing controls still work?

| Control | Site | Under a batched release | Verdict |
|---|---|---|---|
| `CTRL_DROP_PUBLICATION` | `:307-314`, inside the per-tile leader publish loop | Only survives if the implementer **carries the branch into the new batched publish loop**. If preserved, it still proves a missing publication ⇒ bit 26 + 100% untouched sentinel (`m4_controls.py:49-74`), and it is the one control that would catch tail bug #1. | **works if moved; must be re-verified** |
| `CTRL_REROUTE_SLOT` | `:251-257`, in the emit path | Untouched by the change; still detects address errors in the *emit*. Blind to publication-side errors, which is where E3's bugs live. Also blind under unchanged inputs (`HANDOFF.md:213-216`, `RESULTS.md:108-112`). | **works, but blind to E3** |
| `CTRL_DROP_CREDIT` | `:381-388` | Untouched; the credit path is not modified by E3. | **works, blind to E3** |

**None of the three tests release/publication ordering.** The property E3 puts
at risk — "no `ready` for epoch `e` becomes visible before the bytes it
describes" — is currently unvalidated by any control, and the ordering bug is a
race whose numeric signature (one wrong 1-of-8 contribution,
`max|diff| ≈ 7.5e-3`) sits **inside** the graded `1e-2`
(`RESULTS.md:113-118`) and *vanishes entirely* when inputs do not change,
because the stale slot holds the previous epoch's correct answer.

Conclusion: **`2e-3` is necessary but not sufficient.** A batched-release bug
that fires on 1 tile in 10⁴ produces a handful of wrong elements at ~7e-3, which
`allclose(2e-3)` will catch *if it fires during a verified epoch* — but nothing
in the current gate ladder raises the probability that it fires, and nothing
makes a stale read distinguishable from a correct one.

### 3.2 Proposed additional check — `m9_stale_slot.py` (poisoned heap, bitwise golden)

Three ingredients, all cheap, all outside any timed region. This is a **hard
gate** for E3, not an optional extra.

**(a) Poison the heap between epochs.** After the full 8-rank sync that ends
launch `e-1` and before generating the inputs for launch `e`, fill **every**
rank's `c_heap` with bf16 quiet-NaN (`0x7FC0`). If the protocol is correct this
is unobservable: every slot a reducer reads in epoch `e` is fully rewritten by
its producer before `ready = e` is published. If a slot is read stale, the
reducer sums a NaN and the output element is NaN — **detection with no tolerance
argument at all**, and it destroys the "previous epoch's bytes are the right
answer" blindness that made `CTRL_REROUTE_SLOT` report a false pass. Cost on
shape 6 is one 134 MB memset per rank per epoch (~40 µs of device time,
untimed). Requires a `memset_device`-style entry point in `dhk_rt` if one does
not already exist.

**(b) Bitwise golden comparison, not a tolerance.** This kernel is bit-exact
deterministic: fixed source-ascending FP32 reduction with a single RNE pack
(`hk_adapter.cuh:216-279`), no atomics in the arithmetic path, fixed tile→CTA
map. E3 must not perturb arithmetic at all, therefore **the batched build must
be bit-identical to the current best build, element for element, on every rank,
for every epoch of a fixed seed sequence.** Assert `torch.equal`, not
`allclose`. This is orders of magnitude stronger than `2e-3` and free. Run the
same build twice as well, to separate "deterministically different" from
"racy".

**(c) Many epochs, changing inputs, rotated skew, on the shapes that actually
batch.** Reuse the `m5_soak` structure (`m5_soak.py:41-46`: new seed each epoch,
rotating launch order so a different rank is last to start). Its default shape
is 512×4096×12288, which has only 2 tiles per CTA and *does* exercise the 1-tile
tail — keep it at 600 epochs — but **add ≥ 60 epochs on 8192×8192×29568** (the
only shape with a group of 4 and a 3-tile tail) and **≥ 20 epochs on
8192×8192×28672** (generic row, 117 tiles per CTA, the P4 stress case). Verify
every epoch at `2e-3`, plus (a) and (b).

**(d) A fourth negative control, `CTRL_PUBLISH_EARLY`** — the control of the
control. In the negative-control module only, publish the group's ready flags
*before* `release_payload_system()` instead of after. Under (a)+(b)+(c) this
**must** produce NaNs or bitwise differences. If it does not, the detector has
no power over the exact property E3 puts at risk and the gate is theatre. This
is the direct analogue of the M4 principle that "a control that silently passes
would mean the corresponding safety property is untested"
(`m4_controls.py:1-13`).

Also keep, unchanged: `check_signals()` (`harness_lib.py:342-360`) already
detects tail bug #1 and any publish-set permutation that leaves a cell short of
`n_calls`; `check_epochs()` detects a CTA that skipped its loop.

---

## 4. Verdict and recommended design

### 4.1 Verdict

**APPROVE WITH CONDITIONS.** Conditions, each testable:

- **C1.** Publication moves with the release. No `publish_band_epoch` call may
  execute between a payload store and the release that covers it. *Checkable by
  reading the diff: there must be exactly one `release_payload_system()` call
  site and it must dominate every `publish_band_epoch()` call site in the group
  body.*
- **C2.** The group structure makes an unflushed tail impossible — release and
  publish live inside the outer group loop, whose body cannot run with zero
  tiles. *Checkable by reading the diff.*
- **C3.** The publish loop recomputes `(tm, tn)` by calling the **same** decode
  helper as the emit loop. No copied arithmetic, no journalled arrays, no
  dynamically indexed local arrays. *Checkable by reading the diff; confirm zero
  new scratch in the M2 resource tuple.*
- **C4.** The group size is a **compile-time constant with a hard cap**, applied
  identically on the generic row. "Per CTA per epoch" as an unbounded rule is
  **rejected** (P4).
- **C5.** `G = 1` must be bit-identical in output and within run-to-run noise in
  time versus today's binary, and must produce an ISA diff limited to the loop
  restructure. This is the refactor's own control arm.
- **C6.** The M4 control module is rebuilt and re-run; `CTRL_DROP_PUBLICATION`'s
  branch is carried into the new publish loop and still yields bit 26 with a
  100% sentinel output.
- **C7.** The new `m9_stale_slot` gate (§3.2 a–d), including
  `CTRL_PUBLISH_EARLY`, passes before any timing is reported.
- **C8.** Report `time_pipelined` **and** `time_single_shot`, and disclose the
  P2/P3/P5 sizing corrections in `result.md`. Shapes 1, 3 and 4 must be
  bit-identical and within noise — if they move, something other than the
  release changed.

### 4.2 Recommended grouping rule

**Per-N-tiles with a compile-time `N = 4`, hard-capped, implemented as an outer
group loop.** Rationale:

- On every scored shape, `N = 4` *is* per-CTA-per-epoch (max 4 tiles per CTA,
  P2), so it captures the entire available gain — the maximum: one release per
  CTA on shape 6, where the ~183 µs upper bound lives.
- Unlike per-CTA-per-epoch, it stays bounded on the generic row (29 groups
  instead of 117 releases on the 117-tile official shape) and bounds the deferred
  publication delay everywhere.
- Per-peer batching is rejected: one release already covers all destinations
  (§2.3), so splitting by peer strictly increases the release count.

Sweep `N ∈ {1, 2, 4}` — `N = 1` as the behaviour-preserving control (C5),
`N = 2` as the low-delay point, `N = 4` as the maximum-amortization point. If
`N = 4` and `N = 2` are within noise, ship `N = 2`: it has strictly less
publication delay and strictly less exposure to §2.4's pipelined-throughput
risk.

### 4.3 The exact ordered sequence the implementer writes

```c++
// gemm_rs_mi300x.cpp, producer role. `ep` stays where it is (:162-163).
// New: one compile-time constant. Cap it; never derive it from `tiles`.
constexpr int RELEASE_GROUP = 4;          // sweep {1, 2, 4}; N == 1 is the control

// Single source of truth for the tile decode; called by BOTH loops below.
struct tile_id { int tm, tn; };
__device__ __forceinline__ tile_id decode_tile(int t, int num_pid_m,
                                               int num_pid_n) {
    const int in_group = WGM * num_pid_n;            // verbatim from :193-199
    const int group    = t / in_group;
    const int first    = group * WGM;
    const int gsize    = (num_pid_m - first < WGM) ? (num_pid_m - first) : WGM;
    return { first + (t % in_group) % gsize, (t % in_group) / gsize };
}

for (int t0 = pid; t0 < tiles; t0 += RELEASE_GROUP * g.num_gemm_ctas) {
    int emitted = 0;                       // CTA-uniform: t0, tiles, NG are uniform

    for (int j = 0; j < RELEASE_GROUP; ++j) {
        const int t = t0 + j * g.num_gemm_ctas;
        if (t >= tiles) break;             // uniform branch
        const tile_id id = decode_tile(t, num_pid_m, num_pid_n);

        zero(C_accum);
        /* ---- mainloop, byte-for-byte as :201-221 ---- */

        /* ---- per-band credit wait, byte-for-byte as :228-240 ----
           STAYS PER TILE. Do not hoist. A failed wait still returns
           immediately; the pending group is abandoned unpublished, which is
           fail-closed and correct. */

        /* ---- band / window emit, byte-for-byte as :246-296 ---- */
        ++emitted;
    }

    // ONE release for the whole group. Whole-CTA convergent; not inside any
    // divergent region. Covers every payload store issued above by every
    // thread, to every destination (see 2.3).
    m3::release_payload_system();

    // ALL of the group's publications, after that single release.
    if (threadIdx.x == 0) {
        for (int j = 0; j < emitted; ++j) {
            const int t = t0 + j * g.num_gemm_ctas;
            const tile_id id = decode_tile(t, num_pid_m, num_pid_n);   // same helper
            for (int b = 0; b < bands; ++b) {
                const int row0 = id.tm * BM + b * EB;
                const int dest = m3::owner_of_row(row0, slice);
                const int lrow = (row0 - dest * slice) / EB;
#if HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS
                if ((g.ctrl_flags & m3::CTRL_DROP_PUBLICATION) &&
                    me == g.ctrl_rank && dest == g.ctrl_arg0) continue;   // C6
#endif
                m3::publish_band_epoch(
                    sig, sig_peers, dest, me,
                    m3::SIGNAL_GUARD_U32 +
                        m3::ready_idx(me, lrow, id.tn, lrows, cols),
                    ep);
            }
        }
    }
}
// No post-loop flush exists, and none is needed: the release and the publish
// are inside the outer loop, and `t0 < tiles` guarantees `emitted >= 1`.
```

Coverage is preserved exactly: `{t0 + j·NG : t0 ∈ {pid, pid+G·NG, …}, j ∈ [0,G)}
∩ [0, tiles) = {pid + i·NG : i ≥ 0} ∩ [0, tiles)`, i.e. the same tile set as
`:191`, each visited once.

### 4.4 Invariants the implementer must not break

Phrased so each can be checked by reading the diff:

1. **One release per group, and it dominates every publication in that group.**
   Exactly one `release_payload_system()` call site in the producer role; every
   `publish_band_epoch()` is lexically and dynamically after it.
2. **No publication is reachable without a preceding release in the same
   iteration.** No `publish_band_epoch()` inside the inner tile loop.
3. **The release is whole-CTA convergent.** Not inside `if (threadIdx.x …)`, not
   inside a non-uniform branch. (It contains two `__syncthreads()`,
   `sync.cuh:171,173`.)
4. **`emitted >= 1` whenever the release executes**, and a zero-tile CTA
   executes neither the release nor any publication.
5. **The publish loop and the emit loop derive `(tm, tn)` from the same
   expression, via one shared helper.** No duplicated arithmetic.
6. **Nothing is journalled.** No new LDS allocation (the budget is already at
   65536 B for `BM=BN=256`), no new local array indexed by a runtime value, no
   increase in scratch in the M2 resource tuple.
7. **The credit wait stays per tile, before that tile's first payload store**,
   with the `atomicOr` + `__syncthreads()` + `error_bit_set` fail-closed
   broadcast intact (`:228-240`).
8. **A group's tiles have pairwise-disjoint destination slots.** True today by
   the tile→`(dest, lrow, col)` injection; re-verify if tile order, split-K, or
   `EB` ever changes.
9. **`ep` remains loop-invariant and is never recomputed inside the group.**
10. **The negative-control branches move with the code they instrument**, and
    the control module still exports only `gemm_rs_mi300x_control`.
11. **No inline asm around the peer stores** (the compiler's `expcnt` tracking
    is what makes store-source VGPR reuse safe).
12. **No change to `sig` geometry, `ready_idx`/`credit_idx`, the 72-byte
    descriptor ABI, or the one-launch-per-call contract.**

---

## 5. What would change this verdict

- If the disassembly shows the staging-loop `__syncthreads()` at `:270`/`:294`
  already carries `s_waitcnt vmcnt(0)`, the recoverable pool shrinks below the
  measurement noise floor and E3 should be closed without a GPU run.
- If a re-run of `exp_ablation.py` on the current (`NR=32`) source shows the
  release delta collapsing on shape 6, same conclusion.
- If `CTRL_PUBLISH_EARLY` under the poisoned-heap gate does **not** fail, the
  detector is powerless and no batched-release arm may be declared correct.

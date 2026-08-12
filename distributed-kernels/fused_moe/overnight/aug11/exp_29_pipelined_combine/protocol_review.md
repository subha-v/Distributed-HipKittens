# exp_29 — protocol review (adversarial, of my own design)

Read-only. Reviews `design.md`. File tags as in `design.md`.

The posture of this document is hostile to the design. Where the design is safe I
say why in terms of a cited edge; where it is safe *only because of a rule*, I
state the rule as a build-blocking requirement; and §8 is a numbered list of the
ways this thing can be wrong while every gate we own reports green.

---

## 1. Every release / acquire pair, with scope and the invariant it protects

`R1`–`R3`, `R5`, `R6` are the shipped mode-12 chain, restated so the new edges
can be checked against them. `R4`, `R7` are exp_29's.

| # | release (producer side) | acquire (consumer side) | scope | invariant protected | count / rank / epoch | changed by exp_29? |
|---|---|---|---|---|---:|---|
| **R1** | epilogue remote RMWs → `s_waitcnt vmcnt(0)` `KRN:358` → `__syncthreads()` `KRN:359` → `publish_tile_release<agent>` = `fence(RELEASE,"agent")` + relaxed store to `mps_q` (`ROLES:136-137`, `ADP:387`) | service wave relaxed load `q[k]` (`ADP:453`) → `thread_acquire<agent>` (`ADP:670`) | **agent** | the event word `(b,nc)` implies this CTA's remote RMWs for that tile-chunk are fabric-ACKed | 16,720 each | **no** |
| **R2** | every contributor's `fetch_add_acq_rel<agent>(nc_arr[r*16+nc])` (`ADP:720-722`) | the **last** arriver's own acq_rel RMW on the same cell — it is in the release sequence of every earlier one | **agent** | the wave that observes `old+1 == row_rem[r]` inherits every contributor's causality for that chunk | 524,288 | **no** |
| **R3** | `flush_pending`: `s_waitcnt vmcnt(0)` `ADP:560` → `thread_release<system>` `ADP:570` → relaxed **system** store `row_ready[cur][r] = epoch32` on the owner `ADP:581-583` | owner: `poll_epoch_system` `KRN:511-514` → `acquire_payload_system` = `fence(ACQUIRE,"")` `KRN:526` | **system** | the flag implies **the entire 14,336 B row's accumulated payload is visible to the owner** | rel ≈ 1,364–2,400 · acq 1,024 | **no** |
| **R4** | *(same release as R3)* | **sweep**: relaxed **system** probe loads (no acquire), then exactly **one** `acquire_payload_system()` after the ballot, at the same program point as `KRN:526` | **system** | identical to R3 | **acq unchanged at 1,024** — see §1.1 | **NEW consumer, same edge** |
| **R5** | M8 consume-and-zero plain `uint4` stores `KRN:579-587` → M9 `fence(RELEASE,"agent")` `KRN:1848` | peers' next-epoch remote RMWs into those cells | **agent release, system consumer — MISMATCH** | zeroed cells are visible to a peer's next-epoch fabric RMW | 1 | **inherited, not created — §4.3** |
| **R6** | M9 `publish_value<system>(retired[cur], epoch)` `KRN:1856-1857` | next epoch M0 `poll retired[tid] >= epoch-1` at **system** `KRN:702-704` → grid barrier `KRN:716` → `acquire_payload_agent()` `KRN:718` | **system** | no peer's next-epoch accumulate precedes this owner's zeroing | 7 stores | **no** |
| **R7** | sweep's `out` stores and its `claim` RMW | M9 `__hip_atomic_fetch_add(combine_done, ACQ_REL, AGENT)` `KRN:1828-1829` | **agent** | every `out` row is written before the launch retires | 256 | **NEW consumer of an existing edge** |

### 1.1 The acquire-count question, answered

> *"Under pipelining, where does that acquire go, how many are there, and does
> moving it into the pool loop change its count from 1,024 to something much
> larger?"*

**It stays at 1,024, and that is a design constraint rather than a happy
accident.** Three rules make it so, and all three are build-blocking:

1. **The work unit is the batch, not the chunk.** One acquire per batch, and
   every batch is reduced exactly once across the sweep and the terminal M8
   combined (§3, `claim`). So the total is `ceil(T/4) = 1,024`, exactly d3's
   count today. A **chunk** unit with a per-unit acquire would take it to
   `1,024 × 14 = 14,336` — a 14× increase in the fence class exp_20 §4 named as
   the prime interference suspect, injected into M7 rather than the tail.
2. **Probes are relaxed and acquire nothing.** The sweep's non-blocking passes
   issue `load_relaxed<system>` only. `bounded_observe_acquire_into`
   (`CMPL:165-172`) and `wait_tile_acquire_into` (`ROLES:147-153`) both fuse the
   acquire into the observation, so **neither may be used here** — using them
   would put a system fence on every failed probe, i.e. ~258 k of them. This is
   exactly the gap `design.md` §10.2.1 asks the library to close.
3. **The acquire is on the commit path only**, placed at the same program point
   as `KRN:526`: after the ballot, before any payload load. The existing
   `k0p6_mps_m8_batch` already does this; calling it verbatim inherits the
   placement.

**Violating any of the three silently multiplies the system-fence count into the
M7 window. Check it in the ISA:** count `buffer_inv` / `buffer_invl2` in the M7.6
region before and after, and count them again per config.

---

## 2. Every counter, its exact target, and who resets it

| counter | shape / slot | target | writer | reset by | changed? |
|---|---|---|---|---|---|
| `nc_arr[r*16+nc]` | `uint32[T_ext][16]`, slot 57 | `row_rem[r]` — a **runtime, per-key** value loaded at `ADP:690-691` | every live lane of every event, `ADP:720-722` | M0 `KRN:740` | no |
| `pushed[r]` | `uint32[T_ext]`, slot 58 | **16** (`ADP:607`) — chunks done, not pushes | the unique chunk finisher, `ADP:605-606` | M0 `KRN:741-743` | no |
| `row_rem[r]` | `uint32[T_LOC_MAX]`, slot 26 | — (it *is* the target) | M2 `KRN:1163-1169`; zeroed by the unique flag publisher `ADP:587` | **self-clean, not M0** | no |
| `mps_state[TAIL]` | word 1 | monotone, bounded by `events_total` | M7 enqueuers `ADP:386` | M0 `KRN:745-748` | no |
| `mps_state[EVNEXT]` | word 3 | consumption stops at `events_total` `ADP:654` | drain waves `ADP:649-650` | M0 | no |
| `mps_state[M8NEXT]` | word 2 | stops at `ceil(T/4)` `KRN:1760` | terminal M8 waves `KRN:1788-1790` | M0 | no |
| `combine_done` | slot, 1 word | **256**, tested `old % gridDim.x == gridDim.x-1` `KRN:1830` | every CTA `KRN:1828` | itself, by the last arriver `KRN:1846` | no |
| **`claim[b]`, `b ∈ [0,1024)`** | reuses `uint32[T_ext]`, **slot 59** | winner is `fetch_add_relaxed<agent>(claim+b,1) == 0`; exactly one winner | sweep waves **and** terminal M8 waves | **M0 `KRN:741-744`** — already zeroed today | **NEW** |
| **`mps_state[4]`** | free scalar word | **`ceil(T/4)` = 1,024**; checked by M9's last arriver, `pperr` bit on mismatch | every wave that *completes* a batch | M0 `KRN:745-748` (zeroes all 8 scalar words) | **NEW, mandatory** |
| **`mps_state[5..7]`** | free scalar words | S0 diagnostic histogram only | one elected wave | M0 | **NEW, S0 only** |

### 2.1 `claim`'s reuse — the two conditions that make it legal

`claim` (slot 59, 128 KiB) is **allocated, M0-zeroed, and never written in mode
12**: the `g == 1` fast path at `ADP:731-747` returns `push_lead = true` before
reaching the `atomicOr(env.claim + r, claim_bit)` at `ADP:764-765`
(`mode12_protocol_map.md` §8 item 8, c10 count = 0).

Legal **only if both** hold, and both must be enforced in `config_is_valid`:

- **C1: `physical_g == 1` for modes 10/11.** Inherited from the
  `mode_is_direct_accum` branch at `ADP:262`. If a later experiment relaxes it,
  `run_service` starts writing `claim[r]` for `r ∈ [0, T_ext)` — which **overlaps
  `claim[0..1024)`** — and batch claims get silently stolen. Failure signature is
  §8 mode 1 (invisible). **Add an explicit assertion in the validator, not a
  comment.**
- **C2: the M0 zeroing is ordered before the first reader.** `KRN:728-730` states
  the contract ("used no earlier than M6.9; the M3/M5 grid barriers order this
  zeroing before any reader") and `KRN:1301` is the last grid barrier in the
  kernel. The sweep runs in M7.6, strictly after. ✓

Alternative if C1 cannot be guaranteed: carve the claim out of `pushed`'s unused
tail or add a 4 KiB descriptor slot. Both cost ABI; `claim` costs nothing. Take
`claim`, and pin C1.

---

## 3. Every buffer, and its epoch lifetime

| buffer | live from | live until | who may write during M7 (epoch E) | exp_29 effect |
|---|---|---|---|---|
| `slots[p][pos]` (448 MiB, symmetric) | epoch E's first remote RMW by producer `p` | the owner's consume-and-zero | **only rank `p`** — the address carries the *producer* in `slot_off` (`P2:265-268`), so each slot row has exactly one writer | **the zero moves earlier, into M7.** Strengthens R6, weakens nothing (§4) |
| `out` (56 MiB, local) | M8's store | **never cleared** — it carries the previous epoch's value until overwritten | nobody today; the sweep, under exp_29 | **the enabler of the invisible failure class** (§8) |
| `row_ready` (1.25 MiB, symmetric) | never | never — monotone epoch values | producers' pools | read by a second consumer (the sweep) |
| `claim[0..1024)` | M0 zero | end of M8 | sweep + terminal M8 | **new use of a dead buffer** |
| `pull_ptr`, `pull_src`, `pull_stage` | M3 `KRN:1277` / M5 `KRN:1299` | end of M8 | **nobody** — read-only after the `KRN:1301` grid barrier | read from a new phase; safe because of that barrier |
| `nc_arr`, `pushed` | M0 zero | flag publication | pool | unchanged |
| `mps_state[0..7]` | M0 zero | M9 | as tabled | words 4..7 activated |

**One dependency worth naming explicitly:** the sweep reads `pull_ptr` and
`pull_src` during M7. They are finalized in M5 and the M5 grid barrier
(`KRN:1301`) is the **last grid barrier in the entire kernel** (§1 of the map:
barriers at 716, 1101, 1236, 1291, 1301; 1555 and 1611 are mode-gated off). So
the sweep's inputs are stable by construction — but this is the *only* thing
standing between exp_29 and a read-during-write race, and any future experiment
that moves plan work later must re-check it.

---

## 4. Consume-and-zero — the proof

**Claim to prove: no rank can accumulate into slot row `(p, pos)` on the owner
after the owner has zeroed it.**

### 4.1 Intra-epoch (H1) — safe, and *unchanged* by pipelining

The owner zeroes `(p,pos)` only after observing
`row_ready[p][row] >= epoch32` (`KRN:511-514`). Walk that flag backwards:

1. Producer `p`'s pool published it (`ADP:581-583`) only after
   `pushed[row] == 16` (`ADP:607`), i.e. **all 16 N-chunks of that row are
   final**.
2. Chunk `(row,nc)` is final only when `nc_arr[row*16+nc]` reached
   `row_rem[row]` (`ADP:723`), i.e. **every contributing local expert on `p` has
   arrived for that chunk**.
3. Each contributor arrived only after its epilogue's `s_waitcnt vmcnt(0)`
   (`KRN:358`, b1) — which on the measured path means **fabric-ACK** for its
   remote RMWs (exp_18's completion semantics, restated at `KRN:309-312`) — and
   after `__syncthreads()` (`KRN:359`).
4. `slots[p][pos]` has **exactly one writing rank**, `p`: the destination address
   is `peer_tab[owner] + slot_off + pos*7168 + col` with `slot_off` containing
   `cur * MAXTOK * kHidden * 2` where `cur` is the **producer** (`P2:265-268`).
   So the flag from `p` is a *complete* gate for that row; there is no second
   producer whose contributions the flag does not cover.

Therefore at the instant the owner may zero, **every epoch-E accumulate into that
row has already been acknowledged.** ∎

**This argument is byte-for-byte the argument for the read**, which is the crux:
`design.md` moves *when* the zero happens, not *what gates it*. The zero and the
read are gated by the same flag at the same program point (`KRN:561` guards both
the accumulate at 565 and the zero at 579). Pipelining cannot weaken H1 because
it does not touch the predicate.

### 4.2 Cross-epoch (H2) — safe, and pipelining makes it *stronger*

Zero (now during M7 of E) → the zeroing CTA reaches `__syncthreads()` `KRN:1824`
→ `combine_done` acq_rel `KRN:1828` → last arriver at 256 → `KRN:1848` release →
`retired[cur] = E` to 7 peers at system scope `KRN:1856` → peer's epoch-E+1 M0
polls `retired >= E` at system scope `KRN:702-704` → grid barrier `KRN:716` →
`acquire_payload_agent()` `KRN:718` → only then M1 and, much later, M7's
accumulates.

Moving the zero **earlier** in epoch E adds slack to every link of that chain.
The ordering is preserved a fortiori. ∎

### 4.3 The `KRN:1848` scope mismatch — inherited, not created, and worth fixing anyway

`mode12_protocol_map.md` §8 item 7 flags it and it is real: `KRN:1848` is
`fence(RELEASE,"agent")`, then the `retired` pokes are **system**-scope relaxed
stores. A relaxed store carries nothing; the ordering comes from the fence, and
an agent-scope release on gfx950 does **not** issue the `buffer_wbl2 sc1`
writeback that the system form does (`SYNC:166-174`; §3's fence table records
`buffer_wbl2` count **0** in the whole M6→M9 window). If the zeroed lines can sit
dirty in the owner's L2 while a peer's fabric RMW resolves memory-side, the RMW
adds to a stale non-zero value and is then overwritten by the writeback.

Why it has not fired in 10 × 600-epoch soaks and two campaigns: the mori symmetric
heap is `HeapType::Uncached` by default (`aug10/CLAUDE.md` M6 row), so the plain
zero stores plausibly never linger in L2 and the agent release is sufficient. That
is an *explanation*, not a proof — the heap type is a default, not an invariant we
assert.

**Position for this review:** exp_29 **inherits** this edge unchanged and does not
worsen it (it moves the zero earlier, which only lengthens the interval before the
poke). But exp_29 makes the epoch's slot lifetime materially more interleaved, so
it is the natural moment to close it. **Recommended, independently of exp_29:
change `release_signal_batch_agent()` at `KRN:1848` to
`release_signal_batch_system()`.** It is one call per rank per epoch, and exp_21's
mode 9 measured this exact release class at **+22 µs inside a ±40 µs band** — i.e.
free. There is no performance argument for keeping it at agent scope. If the
owner of the kernel tonight will not take the change, it must be re-stated in
exp_29's `result.md` as an accepted, unproven assumption.

---

## 5. Epoch tagging — can an early reduce read a stale row that satisfies `>=`?

**No.** Traced:

- `row_ready` is **never reset** (§4 of the map) and carries monotone `epoch32`
  values; the predicate is `observed >= expected` (`CMPL:57-63`).
- At the start of epoch E, `row_ready[p][row]` holds either `0` (never published)
  or the last epoch in which producer `p` published that row, which is `≤ E−1`.
  Both fail `>= E`.
- It becomes `E` only when `p`'s pool publishes in epoch E, which by §4.1 implies
  the row is complete.
- It cannot be `> E`: that needs `p` to be in epoch `E+1`, which needs `p` to have
  observed `retired[owner] >= E`, which needs the owner's M9 — impossible while
  the owner is still in M7 of E. ∎

Two adjacent traps checked and clear:

- **A row live in E−1 but not in E** holds `E−1 < E` ⇒ never falsely ready. ✓
- **`pos` is re-assigned every epoch** (M1 `reserve_row` on the destination's
  `dest_counter`, `KRN:844-856`, with the parity reset at `KRN:762-772`), so a
  given `(p,pos)` may name a different token across epochs. Irrelevant: the flag
  value, not the identity, carries the epoch. ✓
- **32-bit wrap**: 600-epoch soaks against `2^32`; `epoch_ready`'s comment
  (`CMPL:59-62`) requires quiesce-before-wrap and we are 7 orders of magnitude
  away. ✓

**The one epoch trap exp_29 does add** is not on `row_ready` — it is on `claim`.
`claim[b]` is zeroed in M0 of epoch E and read during M7 of epoch E. If any
future change let a CTA reach the sweep before the M5 barrier, it would read the
*previous* epoch's claims and skip every batch. Condition C2 (§2.1) is what
forbids it, and it is a structural property of the barrier list, not of this code.

---

## 6. Deadlock

### 6.1 The cycle the brief asks about

> *Can the pool block on a token whose fanout can only complete via work that the
> pool itself must dequeue?*

**With a blocking sweep: yes, and it is worse than a local hang — it is a
distributed one.** Rank A's pool waves block on `row_ready[B][row]`; B's flag
requires B's pool to drain B's events; if B's pool waves are symmetrically
blocked on `row_ready[A][...]`, neither drains. All eight ranks run the identical
kernel on identically shaped work, so the symmetric case is the *expected* case,
not a corner. Every wait is `spin_limit`-bounded (2,000,000, `CMP:128`) and
pperr-fail-closed, so the node would not wedge — but the launch would fail, on
every rank, deterministically.

### 6.2 The design's answer, and why it is a correctness rule

**The pipelined sweep never blocks.** One probe per outstanding entry per pass;
not-ready ⇒ return to the drain. With that rule the dependency graph is:

```
M7 compute task (rank X)
   → event enqueue (X)                     [blocking edge: drain waits here]
   → arrival RMWs (X)
   → row flag published on owner (X → owner)
   → token batch readiness (owner)         [NON-blocking edge: probe only]
   → reduce + out store (owner)
```

Every blocking edge points from a *consumer of events* to an *M7 compute CTA*,
and M7 compute CTAs never wait on anything the pool produces — a compute CTA's
only wait is `a2_done[b]` from M6 (`KRN:241-243`), which is upstream. The
blocking sub-graph is therefore a DAG rooted at M6, and it is **exactly today's
blocking sub-graph, unchanged**. The only new edge is non-blocking, and a
non-blocking edge cannot participate in a wait cycle. ∎

The terminal M8's poll (`KRN:511-514`) *is* blocking, but it runs only after
`ev_next >= events_total` (`ADP:654`), i.e. after every event has been claimed,
so no wave can be simultaneously holding a claim and blocking on a flag whose
producer needs that claim. That property is inherited from the shipped kernel and
must be preserved: **the sweep must not run after a wave has entered the terminal
M8 loop, and the terminal loop must not be entered before the drain is
exhausted.**

**Build-blocking rule, restated: `probe, do not spin`. This is not a
performance tuning choice. It is the deadlock proof.**

---

## 7. `pperr` bits per failure mode

| failure | bit | site | terminal? | caught by |
|---|---|---|---|---|
| event-queue poll timeout | `K0P6_MPS_ERR_SERVICE` = `1<<26` | `ADP:458` | yes | `[MARK] pperr`, `[MPS SOAK] pperr` |
| combine readiness poll timeout (terminal M8) | `1<<25` = 33554432 | `KRN:514` | yes | same |
| config / shape guard (incl. a new mode's validation) | `K0P6_MPS_ERR_CONFIG` = `1<<28` | `KRN:678-683`, `KRN:1423` | yes | same, plus an empty run dir if it rejects host-side |
| dual-write detector mismatch | `K0P6_MPS_ERR_DUAL` = `1<<27` | `KRN:607` | yes | detector runs only |
| `dest_counter` base-zero guard | `1<<18` = 262144 | `KRN:787` | yes | same |
| **NEW — completed-batch count ≠ `ceil(T/4)`** | **propose `1<<29` = 536870912** | M9's last arriver, reading `mps_state[4]` | yes | `[MARK] pperr` / `[MPS SOAK] pperr` |

The new bit must **not** be added to `payload_ok`'s mask at `KRN:1701-1703`
(that mask suppresses M8 when set, which would make the failure self-fulfilling);
it is a post-hoc assertion evaluated once, at M9, by the elected last arriver.

### 7.1 What the sweep must *not* do on a probe miss

`KRN:523` — `if (__ballot(batch_ready) != ~0ull) return;` — is the terminal M8's
timeout path: it skips the batch **without writing `out`**, deliberately, so a
timeout never mints a usable payload. **The sweep must not reuse that path.** In
the sweep, "not ready" is the ordinary case, and taking a code path that leaves a
claimed batch unwritten is failure mode #1 below. The sweep's miss path must
return **before any claim is taken**, and it must set no error bit.

---

## 8. How this design can be silently wrong and still pass everything

The gates we own are `[MOK GATE] max_abs ≤ 0.1 / relative ≤ 0.1`, zero
nonfinite, `[MARK] control_fails=True`, and `[MPS SOAK] 600/600 pperr=0`. The
following defeat all of them.

**The structural reason this class exists: `out` is `torch.zeros((T,H))`
allocated once (`DRV:1799`) and never cleared between epochs, and the MoK
synthetic campaign feeds *identical inputs* on every iteration. Therefore a token
whose `out` row is simply never written retains a bit-identical copy of the
correct answer from the previous iteration.** Any bug whose signature is "this
row didn't get written" is invisible to every gate, to 600 soak epochs, and to
`[K0PF GATE] combine_bit_exact`.

1. **Claim-before-ready ⇒ claimed-but-unreduced batch.** If the sweep claims
   `claim[b]` and *then* discovers the batch is not ready and bails, the terminal
   M8 sees the batch claimed and skips it. `out[τ0..τ0+3]` keeps last epoch's
   values — bit-identical to correct. **Invisible everywhere.**
   *Mitigation (mandatory):* **probe first, claim second; a wave that wins the
   claim is committed and may not bail.** Plus `mps_state[4]` counting completed
   batches, checked `== ceil(T/4)` at M9 with `pperr 1<<29`.
2. **Terminal-M8 ballot failure on a genuinely-late row.** Same signature as #1,
   and it exists *today* — `KRN:523` returns without writing `out`. Today it is
   coupled to a `pperr` timeout so it is caught. Under exp_29, if anyone
   "optimizes" the sweep by sharing that return path, the coupling breaks.
   *Mitigation:* same counter as #1; it catches both causes.
3. **Premature ready ⇒ a missing addend, diluted below the gate.** If a wave
   reads a slot row before its accumulation completes, the row is **zero** (last
   epoch's consume-and-zero left it zero), so the token's sum is short by one of
   ~5.25 contributions — a ~19 % error on that one token. `relative` is a global
   norm ratio over 4,096 × 7,168 elements, so a handful of bad tokens dilutes to
   ≈ `0.19/√4096 ≈ 0.3 %` against a **10 %** gate. `max_abs` (today 0.035 against
   a 0.1 gate) may or may not trip depending on magnitudes. **A rare premature
   ready is very likely invisible.**
   *Causes to guard:* (a) the ballot taken on a divergent wave — the sweep must
   be whole-wave convergent, exactly like `KRN:523`; (b) the acquire omitted or
   placed after the first payload load; (c) `probe` written with a non-atomic or
   `nt` load — `KRN:507-510` records that an `nt`-only load never invalidates and
   spins on a stale line, and its dual is a load that returns a stale *ready*.
   *Mitigation:* reuse `k0p6_mps_m8_batch` verbatim rather than re-implementing
   the ballot/acquire/gather; that is the whole reason the design insists on
   calling the existing instantiation.
4. **Double reduce.** Two workers reduce the same batch: the second reads the
   zeroed row and writes a mostly-zero `out`. This one is **loud** (a whole token
   near zero would blow `max_abs`) *unless* the two overlap mid-flight, in which
   case the result is a partial sum — signature of #3, invisible.
   *Mitigation:* `claim` is the only exactly-once mechanism; it must be an RMW
   (`fetch_add`, winner `old == 0`), never a load-then-store.
5. **`claim` aliasing.** A future config with `physical_g > 1` in mode 10/11
   makes `run_service` write `claim[r]` for `r ∈ [0, T_ext)`, overlapping
   `claim[0..1024)` and stealing batch claims ⇒ signature of #1.
   *Mitigation:* condition C1 of §2.1, enforced in `config_is_valid`, not in a
   comment.
6. **A skew-only bug.** The campaign runs uniform synthetic routing. Under skew,
   `row_rem`, `F` and the readiness spread all change, and a sweep bug that needs
   an unusual fanout (e.g. `fanout == 8`, the maximum, exercising the full lane
   group) may simply never be reached. `K0_SYNTH_ROUTE` exists and is forwarded
   by the campaign driver.
   *Mitigation:* run at least one `skewed_hot` arm before the ratchet is moved.
7. **The `KRN:1848` scope hole (§4.3).** Already recorded as the project's one
   latent ordering hole that passed 12/12 dedicated probes and 10 × 600-epoch
   soaks. exp_29 does not create it, but exp_29's `result.md` must state whether
   it was closed or accepted — a design that says nothing about it is repeating
   the mistake that made it latent.
8. **A phase-stamp illusion rather than a wrong answer.** Not a correctness bug
   but it silently invalidates the *conclusion*: if the sweep delays some compute
   CTAs' exit from M7.6, `REDUCE_DONE` (max over CTAs, `KRN:1816`) can move for
   reasons unrelated to the mechanism. `M7_DONE` is safe (compute CTAs only,
   `KRN:1383`), but the `combine` window is bounded by both stamps.
   *Mitigation:* always report `m2_to_end` alongside `M7` and `combine`; if
   `ΔM7 + Δcombine ≠ Δ m2_to_end` within noise, the attribution is wrong and the
   numbers must not be quoted.

### 8.1 The one test that collapses this whole class

**Poison `out` between iterations.** `out` is `torch.zeros((T,H))` at
`DRV:1799`; a diagnostic run that fills it with NaN before each timed iteration
turns failure modes #1, #2, #4 and half of #3 from invisible into an instant
`nonfinite` gate failure. It is a **host-side line in a diagnostic run**, it
costs one memset per iteration, and it needs no kernel change.

**This is the single highest-value validation step in the experiment and it
should be run before any timing number is reported upward.** The in-kernel
completed-batch counter (§7) is the always-on backstop; the poison run is the
one-time proof.

---

## 9. Requirements this review imposes on the build

Ordered by what breaks if they are skipped.

| # | requirement | breaks if skipped |
|---|---|---|
| 1 | The sweep **probes, never spins** | distributed deadlock across all 8 ranks (§6.1) |
| 2 | **Probe first, claim second**; a claim winner is committed | §8 mode 1 — invisible wrong answers |
| 3 | Work unit is the **batch**; one system acquire per committed batch | 14× system-fence inflation into M7 (§1.1) |
| 4 | Call `k0p6_mps_m8_batch<4, base_slot&, base_slot&, false, true>` **verbatim** | §8 mode 3, plus the register-lifetime risk `design.md` §8.3 exists to avoid |
| 5 | `claim` is an RMW with `old == 0` as the win test | §8 mode 4 |
| 6 | `config_is_valid` asserts `physical_g == 1` for the new modes | §8 mode 5 |
| 7 | `mps_state[4]` completed-batch counter + `pperr 1<<29` at M9 | §8 modes 1, 2 lose their only always-on detector |
| 8 | Sweep is whole-wave convergent | §8 mode 3(a) |
| 9 | All five mode-predicate sites in `design.md` §8.2 updated together | silent reversion of the epilogue, the defer, or the pool's payload branch |
| 10 | `K0P6_MPS_SRC_REV` bumped; new `.hsaco` mtime confirmed via `readlink -f` + `stat -L` | the entire measurement is taken on the previous kernel |
| 11 | One poison-`out` diagnostic run before any number is reported | §8.1 |
| 12 | `KRN:1848` either raised to system scope or explicitly accepted in `result.md` | §4.3 stays latent for a third experiment |

---

## 10. Reviewer's summary judgement

The mechanism is **safe** — the readiness predicate is unchanged, the
consume-and-zero proof survives intact and is in fact strengthened, the epoch
tagging is sound, and the deadlock is avoided by a rule that is cheap to state
and cheap to check. The correctness surface is small because the design
deliberately reuses M8's existing body, ballot, acquire and address arithmetic
instead of re-deriving them.

It is **not obviously worth building as specified**, for a reason that has
nothing to do with correctness: `design.md` §2 shows the prize is bounded by
`(t/S)^8` and lands near zero net. The cheap stages (S-1's one-line load guard,
S0's read-only readiness probe) return most of the information for ~2 build-hours
and should be run first.

The most dangerous thing in this experiment is **not** the memory ordering. It is
that a whole class of bugs here produces the previous epoch's correct answer, and
every gate we own will call that a pass. Requirement #11 exists for that reason
and should not be traded away for schedule.

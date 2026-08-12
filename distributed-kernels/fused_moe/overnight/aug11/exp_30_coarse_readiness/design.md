# exp_30 — coarse readiness (mode 14): replace ~918k per-row atomics with one global completion barrier

Status: DESIGN ONLY. No GPU job run. Source read-only.
Prereq gate: mode 12's transport must already be viable (see §9 "gating precondition").

---

## 0. The finding that sets the scope

**Per-row `row_ready` buys almost nothing for the consumer.** Two independent proofs.

**(a) The drain is a de-facto local barrier.** Program order in a stream mode is
M7 (`k0pf6gm_device_tile_mps.hip:1393-1418`) → M7.6 drain (`:1428-1493`) → M8
combine (`:1701-1846`). Since exp_12, **every** CTA enters the drain, not just the
reserved pool: `service_cta_here = mode_is_stream(scfg)` (`:1440`), with the intent
spelled out at `:1432-1439` ("compute CTAs fall through as they finish M7 and help
drain whatever is left").

The drain loop (`moe_mps_adapter.cuh:746-760`) claims a monotone ticket
`k = fetch_add(ev_next,1)` (`:748`), breaks **only** on `k >= events_total` (`:753`),
and otherwise *blocks* in `wait_event_nonempty(env.q + k, …)` (`:755`) until the
producing M7 task enqueues event `k`. A wave cannot claim `k+1` before it finishes
`k`, so at most `W` tickets are claimed-but-unserviced at any instant, where
`W = 256 CTAs × 4 waves = 1024`. Therefore the *first* wave to observe
`k >= events_total` does so only after enqueues have reached `events_total − W`.
With `events_total ≈ 16,720` (`adapter:734`), that is **93.9% of this rank's M7
tasks complete before any CTA can enter M8.**

⇒ The intra-rank finish spread (130–210 µs) is **already absorbed inside the
drain**, not hidden by M8's per-row polls. The per-row protocol's local early-start
value is confined to the last ~1024 events ≈ 6% of M7's tail ≈ 160 µs of window,
and only for waves whose tokens' *peer* rows happen to be ready in that window.

**(b) The parity arm already publishes in bulk.** In mode 0 — the reference shape —
`row_ready` is published *after a grid barrier*: `release_cta_payload_system()` →
`hkp::grid_barrier` → CTA-leader grid-strided publication of every live row
(`:1573-1613`). All of rank *p*'s flags flip inside one pass, so a peer polling
`row_ready[p][row]` in mode 0 learns exactly one bit — "rank *p* finished M7" —
delivered by ~19.6k stores × 8 targets. **Coarse readiness is already the semantics
of the parity arm.** Only the stream modes carry finer cross-rank granularity, and
proof (a) shows the consumer cannot exploit it.

**The catch that reshapes the experiment.** `nc_arr` is *not* only a consumer
structure — it is the **producer-side push trigger**. `target = row_rem[r]`
(`adapter:789-790`); slice `(r,nc)` is final when its arrival count reaches
`row_rem[r]` (`:819-822`); the completing lane then pushes that group's payload
into the owner's slots (`:921`, `:929`). Because `row_rem[r] > 1` in general
(multiple local tiles contribute to one receive row), the event alone cannot tell
the pool a row's `part` is complete. Delete the counting on the mode-2 transport
and the push can only begin after a local barrier — the whole ~430 µs transport
becomes exposed. That is the opposite of the goal.

**Resolution: build mode 14 on mode 12's transport.** In `kModeRemoteAccum` (12)
`part` is never written and the payload reaches the owner in the producer's own
epilogue RMWs (`adapter:353-355`, `:886-893`), so there is no push to trigger.
`nc_arr` there is **bookkeeping only** — and coarse readiness deletes the
bookkeeping's only consumer. So on the mode-12 transport the entire protocol dies.

---

## 1. Mechanism

Mode 14 = mode 12's remote-accumulate transport, with the entire per-row readiness
protocol replaced by a two-level completion rendezvous between M7 and M8:

1. **M7** runs on **all 256 CTAs** (no reservation; `is_service_cta` false for 14).
2. **Local convergence.** `release_cta_payload_system()` then `hkp::grid_barrier`
   on the existing `K0P6_D_GBAR` cell — the mode-0 idiom verbatim (`:1573-1582`).
   After this, every local accumulate on this rank is globally ordered.
3. **Cross-rank rendezvous.** One CTA (the barrier's last arriver) publishes
   `epoch` into `m7_done[cur]` on itself (agent) and on all 7 peers (system) —
   the M9 `retired` idiom verbatim (`:1874-1885`). Then every CTA bounded-polls
   the 8 words of its own `m7_done[]` for `== epoch`.
4. **M8** runs the mode-12 consume-and-zero combine (`:1808-1825`) with the
   per-row poll **compiled out**. Every slot is final by construction.
5. **M9** unchanged.

The drain (M7.6) is skipped entirely: nothing enqueues, nothing consumes.

## 2. Exact counters — targets and resetters

| cell | width | target | writer | resetter |
|---|---|---|---|---|
| `K0P6_D_GBAR` (reused) | u32 | cumulative grid-epoch; one more phase | all CTAs | none needed — `local_grid_epoch`'s cumulative target makes each use a fresh phase (`:1576`) |
| `m7_done[world]` (**new**, symmetric heap) | u64 ×8 | value `== epoch` per word | rank `p`'s last-arriver CTA, 1 agent + 7 system publishes | never — epoch-valued, monotone, same discipline as `retired` (`:1860-1885`); no clear, no ABA inside a run |
| `nc_arr`, `pushed`, `row_ready`, `row_rem`, `q`, `ev_next` | — | — | — | **unused in mode 14**; left allocated and untouched so modes 0/1/2/12/13 stay bit-identical |

`m7_done` is epoch-valued rather than a counter on purpose: a counter needs a reset
with no intra-rank ordering against the next epoch — the exact defect exp_56 fix (6)
called out for `dest_counter` (`:1867-1869`). Epoch values need no reset.

## 3. Release / acquire pairs, with scope

| # | release (producer) | acquire (consumer) | scope | why sufficient |
|---|---|---|---|---|
| R1 | producer's remote-accumulate RMWs into owner `slots` | — | system (mode 12's existing epilogue) | unchanged from mode 12 |
| R2 | `release_cta_payload_system()` per CTA (`:1573`) | `hkp::grid_barrier` arrival | system → agent | drains this CTA's stores to L2 + one CTA-level L2 writeback; the mode-0 idiom |
| R3 | last arriver's `release_signal_batch_agent()` before the `m7_done` publishes | remote `bounded_observe_acquire_into` on `m7_done[p]` | agent release → system publish → system acquire | the grid barrier at R2 already ordered *all* local CTAs' payload globally; the agent release then carries that grid-acquired ordering into the remote flag stores. **Identical structure to M9** (`:1874` agent release, `:1875-1885` seven system pokes) and to the parity publication's second-stage release (`:1590`, rationale `:1561-1562`) |
| R4 | — | `acquire_payload_system()` once per wave after the rendezvous, before any slot load | system | replaces the per-batch acquire currently at `k0p6_mps_m8_batch` `:526`; one per wave instead of one per batch |
| R5 | M8's consume-and-zero of each slot row (`Zero=true`, `:1821-1824`) | next epoch's peer accumulate | agent release at `:1874` + 7 system `retired` pokes at `:1875-1885` | **unchanged** — mode 14 deletes `row_ready` but not the retirement gate, so the mode-12 slot-lifetime edge survives verbatim. Do not weaken `:1874` |

Note on the brief's citation: the agent-scope release the mode-12 slot lifetime
depends on is `release_signal_batch_agent()` at **`:1874`**; `:1848` is the M9 block
header comment. Same block, same argument.

## 4. Deadlock argument

A symmetric blocking rendezvous across 8 ranks is safe iff every rank provably
reaches it. Four steps:

1. **Every CTA reaches the local barrier.** M7's task loop is finite and
   data-independent: `num_tasks = num_tiles·16` with a fixed start/stride
   (`n2_phase2_gm_mps.cpp:395,400-404,442-443`). It contains no wait on any other
   CTA or rank. Mode 14 skips the drain, so the only pre-barrier blocking construct
   in the phase is gone.
2. **The local barrier cannot hang.** `hkp::grid_barrier` under
   `local_grid_epoch{gbar, pperr, spin_limit}` with `fail_closed{pperr, 2097152}`
   (`:1577-1582`) is bounded: on timeout it sets bit 21 and returns.
3. **The cross-rank poll cannot hang.** Bounded spin against `spin_limit`, watching
   `pperr`, failing closed by setting a bit and returning — the
   `wait_event_nonempty` discipline (`adapter:550-562`: check pperr each pass at
   `:554-555`, `atomicOr` + return on overrun at `:556-558`).
4. **No cyclic wait.** Rank *p*'s *arrival* at the rendezvous depends only on rank
   *p*'s own M7 and its own local barrier — never on any peer's *exit*. So the
   dependency graph across ranks is a single fan-in per rank with no edge from any
   peer's post-rendezvous state. All 8 arrivals are unconditionally reachable;
   therefore all 8 exits are.

Fail-closed corollary: any timeout in (2) or (3) sets a `pperr` bit, all CTAs skip
the payload (`payload_ok` at `:1727-1730` already gates `T` to 0), and M9 still
retires the launch (`:1849-1888`) so the node does not wedge.

## 5. `pperr` bit per failure

| failure | bit | site |
|---|---|---|
| local grid barrier timeout | `2097152` (1<<21) | `:1580`, already in `payload_ok`'s mask `:1728` |
| cross-rank `m7_done` rendezvous timeout | `33554432` (1<<25) — **reused**, freed by deleting the row_ready poll | today `:514`; header meaning at `:70` |
| a2_done / retire wait timeout | `16777216` (1<<24) | `:70`, `:1728` |
| mode-14 config guard (bad `g`, `pull_fallback` set, `C != 0`) | `K0P6_MPS_ERR_CONFIG` `268435456` (1<<28) | `adapter:31`, validator `:348-380` |
| dual-write detector mismatch | `K0P6_MPS_ERR_DUAL` `134217728` (1<<27) | `adapter:221`, `:607` |
| service/event poll timeout | `K0P6_MPS_ERR_SERVICE` `67108864` (1<<26) | `adapter:30` — **cannot fire in mode 14** (no drain); its absence is itself a signal the mode took the intended path |

Add bit 25 to the `payload_ok` mask at `:1727-1730` for mode 14 so a failed
rendezvous suppresses the combine instead of reading unfinished slots.

## 6. Correctness note that gates the whole experiment

`out` is never cleared between epochs and the harness feeds identical inputs, so a
"row never written" or "row read too early" bug **returns the previous epoch's
bit-identical correct answer** and passes `[MOK GATE]`. Coarse readiness makes
exactly that class of bug the primary failure mode. **The NaN poison of `out` is a
hard prerequisite, not a nicety** — stage S0 below. A mode-14 number measured
without the poison in the build is void, and must be logged as void rather than
re-interpreted.

Second: mode 14 inherits mode 12's slot lifetime unchanged (R5 above). The
consume-and-zero at `:1808-1825` must stay ordered before any peer's next-epoch
accumulate by the `:1874` agent release + 7 system `retired` pokes. Mode 14 must
not touch that block.

## 7. What is deleted, and the count

Per rank per epoch. Live receive rows ≈ 19.6k, Σ`row_rem` ≈ 33,440,
`events_total` ≈ 16,720, local tokens ≈ 4,096, mean fanout ≈ 5.25 of 8.

| structure | ops | needed under mode 14? |
|---|---|---|
| `nc_arr` acq_rel RMWs | **535,040** (= 32 live lanes × 16,720 events; `adapter:735` states 535,040) | **DELETED** — bookkeeping only once the transport is mode 12's |
| `pushed` RMWs | **~314k** (16 per live row at g=1, `adapter:700-713`) | DELETED (mode 13 already deletes it, `:899-913`) |
| `row_ready` publishes | **~19.6k** peer `publish_epoch` (stream, owner-only `:677-683`); ~157k in parity (×8, `:1597-1607`) | DELETED |
| `row_rem` self-clean stores | **~19.6k** (`:686`, `:1612`) | DELETED |
| M8 `poll_epoch_system` loads | **~21.5k** minimum (Σ fanout), more under spin (`:511-514`) | DELETED |
| event queue + `ev_next` | **~16.7k** relaxed RMWs (`:734`, `:748`) | DELETED |
| **total** | **~926k** | vs the ~918k baseline ⇒ effectively all of it |
| replaced by | **1 grid barrier + 8 publishes + 8 polls per CTA** | — |

**Service pool degeneration — stated plainly.** The pool's three jobs are arrival
bookkeeping, payload push, and flag publication. Mode 12's transport moves the
payload in the producer's epilogue; the barrier deletes the other two. **In mode 14
the service pool has nothing left to do**, so mode 14 is a *homogeneous*
megakernel. Per the kernel design mandate, a candidate in which every CTA runs the
whole pipeline "is not an entry in this line of work however fast it is — bank the
number and move on." Mode 14's number must be **banked, not ratcheted** as a
role-split result. It is still worth measuring: it puts a hard number on what the
readiness protocol costs, which is the denominator every future role-split needs.

**Fallback variant 14a** (keeps the split): mode 2's transport, delete `pushed` +
`row_ready` + `row_rem` clean + polls (~375k, 41%), **keep `nc_arr`** as the push
trigger. The pool keeps pushing, so the role split survives. Marginal over mode 13
(which already banked `pushed`) is only ~60k atomics — a small claim, priced in §8.

## 8. Cost arithmetic and prediction

Unit price: 590–860 µs of measured M7 interference per ~918k atomics ⇒
**0.64–0.94 µs per 1k atomics.**

Savings:
- 14 (full): 926k × (0.64…0.94) = **+590 to +860 µs**
- 14a vs mode 2: 375k → **+241 to +353 µs**; **vs mode 13: 60k → +38 to +56 µs**

Costs:
- **Intra-rank M7 finish spread (130–210 µs).** For 14a this is **not a new cost** —
  §0(a) shows the drain already absorbs it. For 14 (no drain) it becomes real:
  **+130 to +210 µs.**
- **Cross-rank rendezvous:** max-of-8 minus the ~5.25-peer order statistic at the
  M7 boundary. Unmeasured; bounded by inter-rank skew: **30–120 µs.**
- **Barrier + 8-message arrival:** **5–15 µs** (mode 0 already pays this shape).

Net:
- **mode 14:** +590…860 − (165…345) = **+245 to +695 µs** ⇒ **5,990 – 6,440 µs**,
  point estimate **~6,215 µs** from today's 6,685. The 6,172 target sits inside
  this band, at its optimistic edge.
- **14a vs mode 2:** +106 to +318 µs ⇒ 6,367 – 6,579 µs.
- **14a vs mode 13:** +38…56 − (35…135) = **−97 to +21 µs** — a wash or a small
  loss. 14a is only worth building if mode 12's transport is unavailable.

Dominant unknown: mode 14 inherits mode 12's remote-accumulate transport cost,
which this design does not re-derive.

## 9. Staged build (≈9 h)

| stage | h | work | validation |
|---|---|---|---|
| S0 | 1.0 | **NaN-poison `out`** at epoch start + a negative control that trips it. Blocking prerequisite (§6) | control run must FAIL; `production` must still pass |
| S1 | 1.5 | Mode-14 plumbing: raise `mode > 13u` → `> 14u` at **`adapter:370`** (the brief's `:271` is stale — the only validator site is `:370`); add `mode_is_coarse(c)= c.mode==14u`; include 14 in `mode_is_direct_accum` (`:246-248`) so it inherits the `g==1`, `!pull_fallback`, no-`part` guards (`:348-355`) and the zeroing M8 (`:1808`); include 14 in `mode_is_stream` (`:172-175`) **only** for `m8_dynamic` (`:1708`); force `service_cta_here = mode_is_stream && !mode_is_coarse` (`:1440`); `is_service_cta` → false for 14 (`:391-396`); exempt 14 from `reserved_comm_ctas != 0` (`:373`). **Bump `K0P6_MPS_SRC_REV`** — `.cuh`-only edits do not invalidate the mori JIT cache | modes 0/1/2/12/13 ISA byte-identical; new `.hsaco` mtime confirmed |
| S2 | 2.0 | The rendezvous: reuse `hkp::grid_barrier` (`:1577-1582`); add `m7_done[world]` on the symmetric heap; last-arriver 1 agent + 7 system publishes (M9 idiom `:1874-1885`); bounded 8-word poll, bit 25 | `pperr == 0`; bit 26 never set (proves the drain was skipped) |
| S3 | 1.0 | Compile out the per-row poll: add a `Ready` template param to `k0p6_mps_m8_batch` (`:471-479`), hoist one `acquire_payload_system()` (`:526`) per wave | mode 2/12/13 instantiations bit-identical |
| S4 | 1.5 | protocol-review signoff on §3/§4; ISA + resource tuple read | ArchVGPR/AGPR parity 256/256; scratch ≤ 60 B; no scratch ops in either MFMA span |
| S5 | 2.0 | Gate ladder then timing: correctness → negative control → 600-epoch soak → 5-process campaign, arms `production,pf6gm_mega,mps_mega` | `[MOK GATE]` pass, `control_fails=True`, `[MPS SOAK]` pass, then rank-max p50 |

## 10. Pre-registered falsifier

- **Kill:** mode 14 lands **≥ 6,685 µs** (today's total) on the same-run paired
  campaign ⇒ the rendezvous serialization exceeds the protocol saving; coarse
  readiness is dead as a lever. Log in `LESSONS.md` and close the axis.
- **Partial:** 6,600–6,685 µs ⇒ saving real but at/below noise; one re-run pair
  before any claim.
- **Confirmed:** < 6,440 µs ⇒ the model holds; < 6,215 µs ⇒ the protocol was the
  dominant M7 interference term.
- **Void (not a result):** any number produced by a build without the S0 NaN
  poison, or with `K0P6_MPS_ERR_SERVICE` (bit 26) set, or from a run whose
  `runN.log` lacks a `[MARK]` line / eight rank JSONs (the partial-`K0_MPS_CFG`
  false pass).
- **Mandate note:** a confirmed number is **banked, not ratcheted** — mode 14 has
  no service pool (§7), so it cannot be the role-split ratchet however fast it is.

## 11. Primitives

Used: `hkp::grid_barrier` + `local_grid_epoch` + `fail_closed`;
`publish_value<scope::agent|system>`; `peer_ptr`; `release_signal_batch_agent`;
`acquire_payload_system`; `release_cta_payload_system`.

Missing / awkward, for the library:
1. **No cross-rank symmetric rendezvous primitive.** Mode 14's step 3 is the third
   open-coded instance of the same shape (M9 `retired` `:1875-1885`, the parity
   publication `:1597-1607`, and now `m7_done`). This wants
   `all_ranks_publish_epoch(cell, world, cur, symmetric)` +
   `all_ranks_observe_epoch_into(...)` in `completion.cuh`, built on
   `bounded_observe_acquire_into`. Additive, no existing caller changes.
2. **`poll_epoch_system` has no compile-time "already-ready" instantiation**, which
   is why S3 needs a template param on a kernel-local helper instead of a library
   knob. A `readiness_policy` tag in `completion.cuh` would express it once.
3. **Negative finding to log under `primitives:`** — the per-row readiness protocol
   in `completion.cuh`/`counter.cuh` is *expressible* but was, on this shape,
   ~350× over-provisioned in signalling stores relative to the information the
   consumer could act on (19.6k stores carrying 8 bits, §0(b)). The library made
   the fine-grained protocol easy to write and gave no way to notice it was
   unnecessary. A primitive that carries its own *granularity* justification — or a
   debug mode that counts flags-published vs flags-that-unblocked-a-consumer —
   would have caught this without a kernel rewrite.

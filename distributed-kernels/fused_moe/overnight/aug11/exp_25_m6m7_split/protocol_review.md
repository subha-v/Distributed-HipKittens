# exp_25 — adversarial protocol review of my own design

Read-only review of `design.md` in this directory. Written to be hostile to it.
Same file tags as `design.md`. `ADP` line numbers follow
`CONTEXT/mode12_protocol_map.md`'s numbering, which I re-verified against the
file.

**Bottom line, stated before the detail so it cannot be missed:**

> This experiment converts a **vacuous** synchronisation edge into a
> **load-bearing** one. Today the `a2_done` poll at `KRN:241-243` never spins —
> all 256 CTAs carry near-identical M6 work, so the counter is long since 8 by
> the time anyone looks (`CONTEXT/m6_m7_structure.md` §4.3). Every latent
> weakness in that edge has therefore never been exercised, in any of the
> ~50 screening configurations, four decision campaigns, and ten 600-epoch soaks
> the tree has accumulated. **A clean soak on the current tree is zero evidence
> that the edge is sound.**
>
> And the campaign cannot detect the most likely failure. The MoK synthetic
> prefill harness feeds **identical input and identical routing on every
> iteration**, so epoch `e−1`'s `A2q`/`DQ2` bytes are **bit-identical** to epoch
> `e`'s. A completely missing readiness edge would return the *right answer* and
> pass 600 epochs with `pperr = 0`. **No arm of this experiment may be believed
> without the stale-read detector in §6.1.**

Three findings below change what gets built:

- **§2 H1** — the M6 payload release is a tid-0-only agent fence behind a bare
  `__syncthreads()`, ordering 255 other threads' plain stores that it never
  touched. `producer_drain_release<agent>` exists in the library and is not
  called. **Blocking for any arm whose number we keep.**
- **§6.3** — an off-by-one in the M7 pool's start/stride that covers a task
  **twice** doubles one 32×448 tile out of 16,720 and produces ≈ 6 × 10⁻⁵
  relative error. **Every gate passes.** The free detector is `part_done`, which
  mode 12 allocates, zeroes, and never uses.
- **§5** — the spin-limit margin, which is a comfort today, becomes a
  *correctness* argument: a timeout leaves un-zeroed slot rows that corrupt
  epoch `N+1`. The margin is 14× at `C6 = 32` and 3.7× at `C6 = 8`. **`C6 ≥ 32`
  is a validation rule derived from this, not a preference.**

---

## 0. Rev 2 — the interleave (Policy 0) is now the reviewed candidate

The reviewed subject has changed from the static prefix split to the **interleave**
(`design.md` §3.0). Everything below (§1-§8) was written against the static split
and is **kept**; this section states what the change retires, what it keeps, and
what it adds. **Net: the interleave is easier to make correct than the static
split, and the one blocking finding (H1) is now on the critical path with a
one-instruction fix.**

### 0.1 What the interleave RETIRES

| finding | status under Policy 0 | reason |
|---|---|---|
| **§6.3** doubly-covered M7 task (the 6 × 10⁻⁵ silent failure — "most likely bug to actually ship") | **RETIRED for this arm** | The M7 partition is **unchanged**: `start = blockIdx.x`, `stride = 240`, so the task set per CTA is byte-identical to today's. Over-coverage is impossible by construction, not by argument. The `part_done` detector is **not needed** |
| **§6.8** `run_service`'s exact `== 12` | **RETIRED** | Policy 0 claims **no mode**; `k` lives in free config bits [34:40). All seven predicates are untouched (`design.md` §6.1a) |
| **§6.5** cross-XCD M6 front stall in Policy 3 | **RETIRED** | No dynamic role, no front |
| **§5.4** illegal `C6` configurations | **RETIRED** | No `C6` exists |
| exp_01 `address (nil)` re-arm | **RETIRED** | M7 task start/stride and the `bid < 240` predicate are both unchanged (`design.md` §6.1b) |

### 0.2 What the interleave KEEPS, unchanged and still blocking

| finding | status |
|---|---|
| **H1** — E1's release covers only tid 0's own stores | **STILL BLOCKING, and now on the critical path rather than hypothetical.** Fix specified below (§0.3) |
| **§6.1** the stale-`A2q` read is invisible to this campaign | **STILL FATAL to credibility.** Unchanged and, if anything, more dangerous: see §0.4 |
| **§6.2** `a2_done`'s gate is `>=`, over-counting tolerated silently | keeps |
| **§6.6** `g = 33` is a raw bit read with no mode gate | keeps — every Policy 0 sweep point must carry `g=33` |
| **§6.7** `effective_flush_rows` confound | keeps |
| **§6.9** skewed routing changes `num_tiles` | **keeps, but weakened**: `design.md` §6.0's availability identity has no `num_tiles` in it, so the *rate match* is routing-invariant. What survives is per-tile `gcount` non-uniformity perturbing the linearity, not the ratio |
| **§6.10** the `A2q` bounded descriptor hides coverage bugs | keeps (M6's coverage is unchanged, so this is inherited risk, not new) |
| **§4** buffer epoch lifetimes | **keeps and matters more.** See §0.5 |

### 0.3 H1's fix, specified exactly — and it costs one instruction

`design.md` §3.0.4 has the full argument. The protocol-relevant statement:

**Sufficient condition for E1 (agent-scoped `A2q`/`DQ2` → `a2_done` publish):**
every one of the 256 threads' stores must have *reached L2* before the single
tid-0 agent-scope release fence is issued, because that fence's L2 writeback
covers the CTA's entire L2 footprint but cannot flush a store still in flight.

**Today**: `stores → __syncthreads() → tid-0 fence → RMW`. The barrier orders
*execution*, not *memory retirement*. So a store still in flight at the barrier is
not covered by the fence. **This is the defect.**

**Fixed**: `stores → all-thread s_waitcnt vmcnt(0) → __syncthreads() → tid-0 fence
→ RMW`, where the `vmcnt(0)` is supplied by defining
`N2GM_P1_EPILOGUE_DONE_HOOK` (vendored line 637, already present, currently empty)
in `KRN` only. Zero new barriers, zero new fences, zero vendored-file edits,
11,360 waits per rank per epoch (≈ 11 µs).

**Signoff condition:** the fix must be *verifiable by removal* — i.e. the negative
control for S1d is the same kernel with the hook left empty, and the NaN-poison
arm must **fire** on it. A fix that cannot be shown to matter has not been shown to
be a fix.

### 0.4 Why §6.1 is *worse* under the interleave, not better

Under the static split, a missing readiness edge produced a stale read only in a
narrow race window. Under the interleave, **CTAs are rate-matched to the knife
edge** (`design.md` §6.0: available `2.133 m` vs required `2.133`, zero slack), so a
CTA that is slightly ahead of its peers hits the `a2_done` wait **constantly**.
If that wait is unsound, stale reads are not rare — they are the common case.

And because the harness feeds identical input every iteration, a stale read
returns **bit-identical bytes**, so it is correct-looking *and* faster. Restating
the consequence in its sharpest form:

> **The most likely way this experiment produces a headline number is by being
> broken.** A missing readiness edge skips a real wait, returns the right answer,
> passes `rel_L1`/`max_abs`/`pperr`/soak, and posts the best time in the `k` sweep.

This is why `design.md` §5.5 pre-registers a **too-good** result as a defect
signature: any `k` beating `k=∞` by more than **211 µs** exceeds W1's arithmetic
ceiling and must be treated as a bug until the poison arm clears it.

### 0.5 One new hazard the interleave introduces: `A2q` epoch overlap

§4 asked whether an epoch-`(e+1)` M6 CTA can overwrite an epoch-`e` `A2q` row.
Under the static split the answer was "no, because M6 and M7 are in the same
epoch's phase window". **Under the interleave the question sharpens**, because a
CTA finishing its M6 stripe early proceeds to M7 tasks while other CTAs are still
in M6 — but this stays safe for a reason worth writing down: `A2q` is
**single-buffered and capacity-sized**, and the epoch boundary is the M9 grid
barrier, which every CTA must reach before any CTA can begin epoch `e+1`'s M6.
Since the interleave adds no phase-crossing beyond M6↔M7 *within* one epoch, and
M7 tasks are only claimable for tiles whose M6 is complete, **no epoch-`(e+1)`
write can precede an epoch-`e` read.** Unchanged verdict, stronger argument.

The genuinely new item: **the in-order `vmcnt` coupling** (`design.md` §3.0.6). It
is a performance cost (~20-77 µs), not a correctness issue, *provided* the epilogue
either drains or the subsequent wait is conservative. It becomes a **correctness**
issue only in the fine-grained variant (deferring the epilogue drain across an M6
K-loop), which is therefore **closed on ISA grounds**: `vmcnt` is one in-order
counter per wave, so M6 cannot wait on its own loads without also waiting on the
epilogue's outstanding remote atomics — and any attempt to publish `a2_done` before
that drain reintroduces H1 in a form no fence can fix.

### 0.6 Rev-2 verdict

**Policy 0 is signed off for build subject to two conditions**, both cheap:

1. **H1's `vmcnt(0)` hook is in the same commit as the fused loop.** Not a
   follow-up. The interleave without it is an unsound kernel that will post a
   good number.
2. **The S1d NaN-poison detector arm runs before any `k` sweep number is
   reported upward**, and must be shown to fire on the hook-removed control.

Everything else that made the static split risky is retired by the claim-option-(c)
choice. This is the rare case where the cheaper mechanism is also the safer one.

---

## 1. Scope and method

Reviewed: the readiness edge `a2_done`, the mode-12 stream protocol
(`mps_q`/`mps_state[TAIL]`/`EVNEXT`, `nc_arr`, `pushed`, `claim`, `row_rem`,
`row_ready`), the M8 consume-and-zero, the M9 retirement, and every buffer whose
lifetime the split perturbs. Not reviewed: M0-M5 (untouched by this design), the
LL128 dispatch, the sort.

Method: for each edge, name the release, the acquire, the scope, the invariant,
and **what changes when the two sides run concurrently instead of in CTA program
order**. Then look specifically for failure modes whose *numerical* signature is
below the 0.1 gates.

---

## 2. Every release/acquire pair

`R` = release side, `A` = acquire side. "Live today?" answers whether the pair
currently does any ordering work or whether program order does it for free.

| # | edge | R site / lowering | A site / lowering | scope | invariant protected | live today? | live after the split? |
|---|---|---|---|---|---|---|---|
| **E1** | **M6 `A2q`+`DQ2` tile → M7's K-loop** | all 256 threads store (`P1:444-452`, `P1:456-472`) → `__syncthreads()` (`P1:474`) → **tid 0 only**: `HKA:92-97` `arrive_local_count<true>` = `SYNC:150-153` → `SYNC:88-100` `__builtin_amdgcn_fence(RELEASE,"agent")`, then `fetch_add_relaxed<agent>(a2_done+b,1)` (`SYNC:136-140`) | tid 0 `bounded_poll_relaxed_into<agent>(a2_done+b, 8)` (`KRN:241-243` → `HKA:105-115` → `CMPL:104-118`), then **whole CTA** `cta_acquire<agent>` = `__syncthreads()` + `fence(ACQUIRE,"agent")` (`KRN:248` → `HKA:88-90` → `SYNC:176-181`) | **agent** | every byte of a tile's 2048 A2q columns and 16 DQ2 columns, written by **8 different CTAs**, is visible before any M7 lane reads it | **NO — vacuous** | **YES. This is the edge the experiment turns on.** |
| **E2** | M7 epilogue's remote RMWs → the service pool's arrival counting | `k0p6_mps_task_drain` is **suppressed** in mode 12 (`KRN:350`); the drain is deferred to the next task head: `s_waitcnt vmcnt(0)` (`KRN:358`) + `__syncthreads()` (`KRN:359`), then per event `publish_tile_release<agent>` = `fence(RELEASE,"agent")` + `store_relaxed<agent>(q[ticket])` (`ROLES:133-138`, via `ADP:381-389`) | pool wave: `wait_event_nonempty` relaxed load (`ADP:448-464`), then whole-wave `thread_acquire<agent>` (`ADP:670`) | **agent** | the producer's remote atomics have reached the fabric ACK point before the event is observable | yes | yes, **unchanged** |
| **E3** | contributor causality → the flagging wave | `fetch_add_acq_rel<agent>(nc_arr[r·16+nc])` (`ADP:720-722`, `SYNC:67-79`) — the release sequence on one cell chains all contributors | the same RMW's acquire half on the finishing lane | **agent** | the flagging wave's `row_ready` store is ordered after every contributor's payload | yes | yes, **unchanged** |
| **E4** | this rank's slot accumulations → the owner's M8 read | `flush_pending`: wave `s_waitcnt vmcnt(0)` (`ADP:560`), `__syncwarp`, lane 0 `thread_release<system>` = `fence(RELEASE,"")` (`ADP:570` → `HKA:76-78`), then `store_relaxed<system>(row_ready)` (`ADP:581-583`) | M8: `poll_epoch_system(row_ready) >= epoch32` (`KRN:511-514`), `__ballot == ~0` (`KRN:523`), then `thread_acquire<system>` = `fence(ACQUIRE,"")` (`KRN:526` → `SYNC:102-115`) | **system** | 448 MiB of accumulated slot bytes visible to the owner | yes | yes, **unchanged** |
| **E5** | M8's consume-and-zero → next epoch's producers | M9: `fence(RELEASE,"agent")` (`KRN:1848`) then `publish_value<system>(retired[cur])` self + 7 peers (`KRN:1851-1857`) | next launch's M0: `poll_value_at_least_system(retired[tid] >= epoch−1)` (`KRN:702-704`), grid barrier (`KRN:716`), `acquire_payload_agent` (`KRN:718`) | **agent release / system acquire — MISMATCHED** | slot rows are zero before any peer's next-epoch accumulate | yes | yes, unchanged — **but see §6.4** |
| **E6** | *(Stage 0 only)* M6 pool done → the waiting CTAs | `counted_arrive_release_into<agent>` (`CTR:85-90`) on a fresh `mps_state` word, target `C6` | `bounded_poll_relaxed_into<agent>` + `cta_acquire<agent>` | agent | mode 10 is a genuine rendezvous, not a partial one | n/a | new |
| **E7** | *(Stage 2 only)* per-XCD M7 ticket claim | `fetch_add_relaxed<agent>` on bank `bid % 8`, LDS broadcast, `__syncthreads()` | the CTA's own read of the broadcast word | agent (claim) + workgroup (broadcast) | each M7 task claimed by exactly one CTA | n/a | new |

### H1 — E1's release covers only tid 0's own stores. **Blocking.** — **still blocking under Policy 0; exact one-instruction fix in §0.3**

`HKA:95` issues the agent release **inside `if (tid == 0)`** (`KRN:228`). A
release fence orders the *calling thread's* prior memory operations. The `A2q`
stores at `P1:444-452` are issued by **all 256 threads** (`cr = tid>>3`,
`cc = tid&7`, one `uint4` pair each); the `DQ2` stores at `P1:456-472` by the
`q == 0` lanes of all four waves. Tid 0 never touched 255/256 of the payload.

What is supposed to cover them is the `__syncthreads()` at `P1:474`. On AMDGPU
that is a **workgroup**-scope release/barrier/acquire. Agent scope on gfx950
spans **eight XCDs with eight non-coherent 4 MB L2s**. A plain `uint4` store
lands in the *producing* XCD's L2. A consumer CTA on a different XCD reading with
a plain load misses its own L2 and goes to the Infinity Cache — which does not
help if the line is still dirty in the producer's L2 and no writeback was issued.

**Is a writeback issued?** `SYNC:160-174` documents that the gfx950 lowering of a
**system** release was `buffer_wbl2 sc1` and instructs "re-check that fingerprint
for every target/toolchain". Whether an **agent** release emits an L2 writeback on
gfx950 is **not determinable from any source in this tree**, and
`mode12_protocol_map.md` §3's "`buffer_wbl2` … **0** in M6→M9" is a count of
`producer_drain_release` *call sites*, not a disassembly reading — the same
document flags per-barrier fence counts as UNRESOLVED.

- If agent release **does** emit `buffer_wbl2 sc1`: E1 is sound, because tid 0's
  fence writes back the whole L2, including lines the other 255 threads dirtied,
  and `__syncthreads()`'s `vmcnt(0)` guaranteed those stores had left the CU.
- If it **does not**: E1 is **unsound in principle** and survives today only
  because the poll is vacuous. Under overlap it can fail. See §6.1 for why the
  failure is invisible.

**Mandatory pre-run check: C1 in §7.**

**The fix is a library call we already own.** `SYNC:166-174`
`producer_drain_release<Scope>()` is *exactly* this semantic: whole-CTA
`s_waitcnt vmcnt(0)`, CTA barrier, one leader release, CTA barrier. Make
`k0p6_a2_arrive` (`KRN:227-234`) convergent:

```
producer_drain_release<agent>();          // all threads
if (tid == 0) fetch_add_relaxed<agent>(a2_done + b, 1);
```

Cost: one explicit `vmcnt(0)` and one extra CTA barrier per M6 task-sub-block,
i.e. ~8,360 per rank per epoch, against a budget of 66 CTA barriers per M6 task
× 11.09 tasks = 732 per CTA (`m6_m7_structure.md` §5.7). **Structurally free and
it removes the question entirely.** I am not comfortable signing off a kept
number without it.

### H2 — E5's release/acquire scopes are mismatched, and mode 12 is the first mode that depends on it

`KRN:1848` releases at **agent** scope; the seven `retired` pokes at
`KRN:1856-1857` are **system**-scope stores; the consumer's acquire
(`KRN:702-704`, `KRN:718`) is system. The payload that retirement gates in mode
12 is M8's consume-and-zero of local `slots` rows (`KRN:579-587`), which **peers
read and RMW over xGMI in the next epoch**. `mode12_protocol_map.md` §3(d) and
§8 item 7 already flag this. **Inherited verbatim from the parity port,
unchanged by this design, and not this experiment's job to fix** — but it is on
the critical path of every mode-12 arm and it is the reason a timeout-induced
un-zeroed row (§5) is a cross-epoch hazard rather than a local one. Recording it
so it is not re-discovered.

### H3 — the timeout path skips the deferred-event flush

`KRN:406-407` returns out of the phase-2 body on a failed `k0p6_a2_wait`,
skipping `N2GM_TASK_LOOP_TAIL_HOOK` (`P2:564-566`) and therefore losing task
`t−1`'s buffered events. **Divergence check: safe.** Tid 0's
`atomicOr(pperr, 16777216)` (`KRN:244`) precedes the `cta_acquire` at `KRN:248`,
which is a `__syncthreads()`; every wave's `error_bit_set_agent` load
(`KRN:249-250`, `HKA:196-205`) therefore happens after the barrier and observes
the bit, so the `return` is CTA-uniform and no surviving wave hangs on a
`__syncthreads()`. **Consequence is the persistent-state hazard, not a hang —
§5.3.**

---

## 3. Every counter: exact target, who resets it, and what the split changes

| counter | shape | target | who increments | who resets | order-sensitive? | split changes it? |
|---|---|---|---|---|---|---|
| `a2_done[b]` | `int32[8223]` (slot 29) | **exactly 8** — the owning tile's 8 chunk-tasks; pad sub-blocks deliberately do not fire (`P1:475-488`) | M6 tid 0 per live sub-block, `KRN:227-234` | M0 grid-strided zero, `KRN:723-726`; ordered before readers by the M3/M5 grid barriers | no (pure count) | **timing only.** But the poll is `>= 8` (`CMPL:57-63`) so over-count is silently tolerated — see §6.2 |
| `mps_state[TAIL]` | one `uint32` (slot 60 lane 1) | monotonic to `EV = 16·B ≈ 16,720` | M7 tid 0 per live sub-block, `ADP:384-385` | M0, `KRN:745-748` | no | no — every M7 task still runs once |
| `mps_state[EVNEXT]` | lane 3 | monotonic; consumption stops at `k >= events_total` (`ADP:654`) | any pool wave lane 0, `ADP:649-650` | M0 | no | **no, and this is exp_12's gift**: a ticket makes the pool size irrelevant, so CTAs joining at wildly different times cannot double-count |
| `nc_arr[r·16+nc]` | `uint32[32768][16]` (slot 57) | **`row_rem[r]`** (M2 popcount, `KRN:1163-1169`) | one live lane per event, `fetch_add_acq_rel<agent>`, `ADP:720-722` | M0, `KRN:740` | **the `old+1 == target` finisher test is order-sensitive by design** — RMW order elects exactly one lane grid-wide | no: same set of arrivals, different order. **Order-independence of the election is what makes this safe** |
| `pushed[r]` | `uint32[32768]` (slot 58) | **16** (`ADP:606-607`) | one lane per completing slice, `fetch_add_relaxed<agent>` | M0, `KRN:741-743` | the `oldp+g == 16` test is order-sensitive, same election argument | no |
| `row_rem[r]` | `uint32[40960]` (slot 26) | read as a target; **zeroed by the unique flag publisher** (`ADP:587`) | written once in M2 | **self-cleaned, not zeroed at M0** | **yes — see §3.1** | no, but the invariant must be re-argued: §3.1 |
| `claim[r]` | slot 59 | — | **never written in mode 12** (`g == 1` fast path, `ADP:731-747`) | M0, `KRN:741-744` | n/a | no — dead buffer |
| `row_ready[p][r]` | `uint32[8][40960]` (slot 25) | `>= epoch32` | one system store per row, owner only, `ADP:581-583` | never — monotonic epoch values | no | no |
| `mps_state[M8NEXT]` | lane 2 | `ceil(T/4) = 1024` | pool wave lane 0, `KRN:1788-1790` | M0 | no | no |
| `combine_done` | one cell (slot 33) | **256** (`old % 256 == 255`, `KRN:1830`) | one per CTA, `KRN:1828-1829` | the elected finisher, `KRN:1840-1847` | no | **no — but check it: every CTA must reach M9.** A CTA that `return`s early from the phase-2 body still falls through to M8/M9 (the `return` exits the *body*, not the kernel). Verified. |
| `part_done[b]` | slot 30 | — | **never written in mode 12** | M0, `KRN:723-726` | n/a | **available free — §6.3's detector** |
| *(new, Stage 0)* M6 rendezvous | a free `mps_state` lane (4-7) | `C6` | M6 pool tid 0 at loop exit | M0 already zeroes `ST_WORDS + 2·TS_COUNT` (`KRN:745-748`), so **no reset change is needed** | no | new |
| *(new, Stage 2)* 8 per-XCD M7 banks | `mps_state` lanes 8-15 after widening `ST_WORDS` | monotonic per bank | any CTA on that XCD | M0's existing loop covers them once `ST_WORDS` grows | no | new; requires `mps_state_bytes()` (`ABI:186-188`) to grow |

### 3.1 The `row_rem` self-clean, re-proved under reordering

**The hazard:** `row_rem[r]` is read as the arrival target at `ADP:690-691`
*and* zeroed at `ADP:587`. A wave that reads it **after** the zeroing gets
target 0, so `old + 1 == 0` is never true, so that slice never completes, so
`pushed[r]` never reaches 16, so `row_ready[r]` is never published, so M8 spins
to timeout. **This is a hang-class bug, and it is order-dependent.**

**Why it cannot happen, and exactly what the proof rests on:**
`row_rem[r]` is zeroed only by the wave that observed `pushed[r]` reach 16.
`pushed[r]` reaches 16 only after all 16 slices `(r, 0..15)` completed. Slice
`(r,nc)` completes only after all `row_rem[r]` of its contributors arrived.
Therefore `pushed[r] == 16` implies all `16 × row_rem[r]` arrivals for row `r`
have already been performed, and **no event referencing row `r` remains**.

**The proof rests entirely on the event multiset being exactly
`{(b,nc) : b ∈ blocks(r), nc ∈ [0,16)}` with multiplicity one.** The split does
not change that multiset — it changes the order. **But any variant that enqueues
one event twice breaks this proof and produces a hang, not a wrong number.**
That is the *good* half of §6.3's asymmetry; the bad half is that the same
off-by-one in the other direction is silent.

---

## 4. Every buffer's epoch lifetime

| buffer | slot | visibility | intra-epoch reuse? | cross-epoch gate | does the split perturb it? |
|---|---|---|---|---|---|
| `A2q` | 27 | **local** | **none** — each `(row, 256-column band)` written once by one M6 task, read by that tile's 16 M7 tasks after `a2_done == 8` | one launch = one epoch + HIP stream ordering; rows `≥ nvi[0]` unreachable via the bounded descriptor `a2_num_records = nvi[0]·kInter` (`P2:302-306`) and `tile_desc` naming only live blocks | **No.** The whole split lives inside one epoch, between M5 (`KRN:1301`) and M7.6. **No slot to recycle ⇒ `LIFE`'s primitives are correctly unused.** |
| `DQ2` | 28 | local | none, same argument | same | no |
| `mps_q` | 56 | agent | none — monotonic ticket, MARK bit makes 0 = empty | M0 zero (`KRN:739`) | no |
| `nc_arr`, `pushed`, `claim` | 57-59 | agent | none | M0 zero (`KRN:740-744`) | no |
| `row_rem` | 26 | agent, local | written M2, read+zeroed in the drain | **not** M0-zeroed — relies on the self-clean (§3.1) | no |
| `row_ready` | 25 | **symmetric** | monotonic epoch values | never reset; `>=` compare | no |
| `slots` | 61 | **symmetric** | accumulated by remote RMW, consumed-and-zeroed by M8 (`KRN:579-587`) | **E5**: the M9 retirement gate | **yes on the failure path only — §5.3** |
| `part` | 21 | symmetric | **neither read nor written in mode 12** (`mode12_protocol_map.md` §8); M5 still zero-fills 448 MiB of it | — | no (and exp_24 deletes the zero-fill) |
| `mps_state` | 60 | agent | tickets are monotonic within the epoch | M0 zeroes `ST_WORDS + 2·TS_COUNT` (`KRN:745-748`) | **widening `ST_WORDS` is automatically covered by that loop** — a genuinely additive change |
| `tile_desc`, `sti`, `swt`, `sei`, `nvi` | 53, 14-17 | local | read-only after M4/M5 | the M4/M5 grid barriers | no — both pools read the same published tables |

**One thing I want on the record because it looked like a hazard and is not:**
`A2q` is 553.7 MB allocated of which ~68.5 MB is used, and it is **not**
symmetric — no peer ever touches it (`mode12_protocol_map.md` §7 buffer table).
The 571 MB of HBM spent on capacity sizing is exactly what buys the "no credit
protocol" property this design depends on. If anyone later proposes shrinking
`A2q` to a ring, **this whole design's §4.4 argument dies and a full slot
lifetime protocol becomes mandatory.**

---

## 5. Deadlock and liveness proof

### 5.1 Every wait is bounded, and no wait is nested

| wait | site | primitive | timeout action |
|---|---|---|---|
| `a2_done[b] >= 8` | `KRN:241-243` | `bounded_poll_relaxed_into<agent>` (`CMPL:104-118`) | `atomicOr(pperr, 1<<24)` ×2 (`HKA:113`, `KRN:244`), then a CTA-uniform `return` from the phase-2 body |
| `q[k] != 0` | `ADP:448-464` | hand-rolled bounded spin, also watching `pperr` | `atomicOr(pperr, 1<<26)`, `break` out of `run_service` |
| `row_ready >= epoch32` | `KRN:511-514` | `poll_epoch_system` (`HKA:150-160`) | `atomicOr(pperr, 1<<25)`, and `__ballot != ~0` (`KRN:523`) skips the batch **uniformly** — a timeout never mints a usable payload |
| `retired[p] >= epoch−1` | `KRN:702-704` | `poll_value_at_least_system` | `atomicOr(pperr, 1<<24)`, uniform grid return |
| grid barriers M0/M2/M3/M4/M5 | `KRN:716, 1101, 1236, 1291, 1301` | `hkp::grid_barrier` (outside this repo) | `fail_closed` → `pperr` `2097152` |

**No wait is nested inside another** (each poll's body contains only relaxed
loads and `s_sleep`), and every poll additionally watches `pperr`
(`ADP:455`, `HKA:113`, `CMPL`'s spin counter), so **one rank's failure
propagates as an exit condition rather than a stall.** There is no cycle: the M6
pool waits on nothing (M6 has no readiness poll of any kind,
`m6_m7_structure.md` §5.6), the M7 pool waits only on the M6 pool, the service
pool waits only on the M7 pool, M8 waits only on the service pool. **The wait
graph is a chain, not a cycle. Deadlock is impossible by structure.**

### 5.2 The spin-limit margin — now a correctness argument, not a comfort

`spin_limit = 2,000,000` (`K0_SPIN_LIMIT`, `harness_recipe.md` §2.2). Each spin
executes `detail::pause()` = `__builtin_amdgcn_s_sleep(4)` (`SYNC:81-86`) ≈ 256
core clocks, so the bound is ≈ **233 ms** at 2.2 GHz.

The worst *legitimate* wait is a pool wave that claims ticket 0 immediately after
M5 and spins until the first M7 task publishes — i.e. the whole M6 phase on `C6`
CTAs:

| `C6` | `T6(C6)` at `a6 = 620` | margin |
|---:|---:|---:|
| 240 | 2.72 ms | 86× |
| 128 | 4.56 ms | 51× |
| 64 | 8.5 ms | 27× |
| **32** | **16.4 ms** | **14×** |
| 16 | 32.1 ms | 7.3× |
| 8 | 63.6 ms | **3.7×** |

**At `C6 = 8` a single slow epoch can mint a spurious timeout, and §5.3 says a
spurious timeout is not benign. `C6 ≥ 32` therefore has to be a validation
rule.** This is the cleanest quantitative constraint in the whole design and it
would have been very easy to miss.

### 5.3 What a timeout actually leaves behind — the new hazard

Today no timeout can occur (`chunk_poll success_max = 0` of 2,000,000, exp_10;
the `a2_done` poll is vacuous). Under overlap:

1. An M7 CTA times out on `a2_done`, sets `pperr` `1<<24`, and `return`s
   (`KRN:406-407`).
2. It has **already issued remote atomics into peers' `slots`** for its
   completed tasks.
3. It skips the tail flush (`H3`), so those tasks' `(b,nc)` events are never
   enqueued.
4. `nc_arr` for those slices never reaches target → `pushed[r]` never reaches 16
   → `row_ready[r]` is never published.
5. The owner's M8 `__ballot` fails for those tokens and skips the batch
   (`KRN:523`), so **the consume-and-zero at `KRN:579-587` never runs for those
   slot rows.**
6. M9 still retires the launch (`KRN:1822-1862`).
7. **Epoch `N+1`'s producers accumulate on top of the residue.**

`pperr` is terminal by standing policy (`aug10/CLAUDE.md`: "A nonzero `pperr` is
terminal: never clear-and-retry; reinitialize all ranks' protocol state first"),
so this is fail-closed **provided pperr is read.** Two questions I cannot settle
from the tree and which must be settled before any long run:

- **Is `pperr` zeroed per launch or per process?** It is not in M0's reset list
  (`KRN:720-749`). If the host zeroes it per launch, a timeout during **warmup**
  is invisible and corrupts every subsequent epoch's slots. `harness_recipe.md`
  §3 shows `pperr` printed on the `[MARK] eager` lines and on `[MPS SOAK]`, i.e.
  at least twice per arm — but not per launch. **Check H4 in §7.**
- **A soak that trips `pperr` at epoch 17 reports `pass=False` correctly, but
  epochs 18-600 are corrupt and their numbers are meaningless.** That is fine as
  a gate and misleading as a diagnostic. Do not read timings from a soak that
  reported nonzero `pperr`.

### 5.4 Illegal configurations, all fail-closed within ~0.5 s

| config | what happens | verdict |
|---|---|---|
| `C6 = 0` | nobody runs M6 → all M7 CTAs time out → no events → pool times out → M8 skips every batch → M9 retires | fails closed; **reject in `config_is_valid`** |
| `C7 = 0` (mode 11) | nobody runs M7 → no events → same chain | fails closed; **reject** |
| `C6 % 8 != 0` | coverage is still exact (start `bid`, stride `C6`, `bid ∈ [0,C6)` covers `[0,∞)`), but `g = task mod 8` is no longer XCD-stable | **performance** defect only; reject anyway to keep the arm interpretable |
| `C7 < 8` | coverage exact; some XCD has no M7 CTA, so its `nc` residues are computed on the wrong die | performance defect; **reject** |
| `C6 + C > 256` | `C7 < 0` → the M7 predicate `bid >= C6` is never true for any non-service bid → `C7 = 0` case | **reject** |
| **M7 start changed without the M7 predicate** | CTAs `0..C6−1` compute `task = bid − C6 < 0`; `tile = task/16` truncates toward zero to a negative tile; `tile_desc[-8]` reads 32 B before the array → garbage `b0` up to 2²⁸ → `sorted_eid[b0]` far out of bounds | **this is the exp_01 `address (nil)` fault class, re-armed.** Not a config error — a code error. Change `KRN:280-285` and `KRN:1370` in the same commit or not at all. |

---

## 6. Numbered: how this design can silently produce wrong answers that still pass a 600-epoch soak

Ordered by how likely I think each is to actually happen, worst first.

### 6.1 **The stale-A2q read is invisible in this campaign. (Severity: fatal to the experiment's credibility.)**

If E1's release is insufficient (H1), an M7 task can read `A2q`/`DQ2` before M6's
writes are visible at agent scope. What it reads is **the previous epoch's bytes
for the same row**. And the MoK synthetic prefill campaign feeds **the same input
tensors and the same routing on every iteration** — `K0_MOK_SEED_BASE` is fixed
at 1234 and the synthetic route is deterministic (`harness_recipe.md` §2.2-2.3).
So epoch `e−1`'s `A2q[row]` is **bit-identical** to epoch `e`'s.

Consequences: `[MOK GATE] max_abs` and `relative` pass, `pperr = 0`,
`[MARK] control_fails=True` (the negative control is broken in a different way
and still fails), and `[MPS SOAK] completed=600/600 pperr=0 pass=True`. **Every
gate in the ladder passes on a protocol that has no readiness edge at all.**

Worse, it would look like a *success*: an arm with a broken edge has no poll
latency, so it would post the best number in the sweep, and we would ship it.

**Required detector — the DQ2 NaN poison arm (ladder L8).** In M5, after the M4
barrier and before the M5 barrier (so `nvi[0]` is available and the barrier
publishes it), grid-stride `DQ2[r][k] = NaN` for `r < nvi[0]`, `k < 16`. That is
`33,440 × 16 × 4 B = 2.14 MB` of stores — 0.5 % of the 448 MiB M5 already writes
into `part`. M6 overwrites every one of them for every live row
(`P1:456-472` writes all `2g+kb` for `g ∈ [0,8)`, `kb ∈ {0,1}` = all 16 columns,
for every live sub-block), and M7 reads `DQ2` only for `lb32 < gcount`
(`P2:370-375`). **So in a correct run the poison is provably unobservable, and a
single premature read puts a NaN into `out` and trips the zero-nonfinite gate.**

Why DQ2 and not A2q: DQ2 is 2.1 MB against A2q's 68.5 MB, and a stale *scale*
corrupts a whole row rather than 256 of its 2048 columns — higher sensitivity,
32× less traffic.

**No arm of this experiment should be logged as a pass until L8 passes at the
same config.**

### 6.2 **`a2_done`'s gate is `>=`, so over-counting is tolerated silently.**

`epoch_ready` (`CMPL:57-63`) is `observed >= expected`. If a change ever caused
a block's arrival to fire twice — e.g. a vendored-file edit that fires the DONE
hook for `sb >= gcount`, which `P1:475-480` warns about in as many words
("Firing for `sb>=gcount` would double-count and break the `>=8` gate (an
exp_56-class defect); do not") — then `a2_done[b]` reaches 8 after only 4 chunks
and **M7 reads a half-written tile.** Half the A2q columns hold the previous
epoch's bytes, which per §6.1 are the same bytes. **Invisible.**

*Detector:* `bounded_poll_exact_into` (the §8.2(4) primitive gap in `design.md`),
or a one-line post-M6 check that `a2_done[b] == 8` for every live `b`.

### 6.3 **A doubly-covered M7 task doubles one tile and produces ≈ 6 × 10⁻⁵ relative error. (Most likely bug to actually ship.)** — **RETIRED for Policy 0, see §0.1**

The M7 pool's start/stride change (`KRN:280-285`, `KRN:273-276`) is exactly the
kind of arithmetic that goes wrong by one. The failure is **asymmetric**:

- **Under**-coverage (a task never runs) is **loud**: `nc_arr` for that slice
  never reaches target → `pushed[r]` never reaches 16 → `row_ready` never
  published → M8 spins → `pperr` `1<<25`. Fails closed.
- **Over**-coverage (a task runs twice) is **quiet**:
  - the epilogue is an **accumulate** (`P2:148-149`), so that tile's contribution
    is **doubled** in the owner's slot;
  - `nc_arr` overshoots, but the finisher test is `old + 1 == target` with `==`
    (`ADP:722`), so an extra arrival **never** re-elects a finisher and `pushed`
    is **not** double-incremented — the protocol survives intact;
  - `row_ready` fires exactly once; M8 consumes and zeroes normally;
  - `pperr = 0`;
  - numerically: one 32 × 448 tile doubled out of `16 × B ≈ 16,720`
    sub-block-tasks ⇒ **relative L1 error ≈ 1/16,720 ≈ 6 × 10⁻⁵**, against gates
    of `rel ≤ 0.1` and `max_abs ≤ 0.1`. `max_abs` on the affected 96 rows would
    rise by roughly one expert's share of ~1.5 experts — but only on those rows,
    and only if that particular element is near the observed maximum. **It will
    pass.** The current gates read `max_abs = 0.035156`, `relative = 0.008293`;
    a 6e-5 relative perturbation is a fifth of the existing relative-error
    headroom's noise.

**Detector, and it is free: `part_done`.** Slot 30, `int32[8223]`, **allocated,
zeroed in M0 (`KRN:723-726`), and never written in mode 12** (the enqueue
predicate at `KRN:319-320` sends mode-12 task-done to the event queue, so the
`else` arm's `part_done` bump at `KRN:327-328` is dead). Add, behind a diagnostic
`g` bit: one `arrive_local_count(part_done + b)` per M7 live sub-block, and after
M7 one grid-strided check that `part_done[b] == 16` for every live `b`, setting
a `pperr` bit otherwise. **Zero new buffers, zero new zeroing, ~16,720 extra
relaxed RMWs, and it converts the single quietest bug in this design into a
loud one.** Ladder L9.

### 6.4 **A warmup-epoch timeout corrupts every later epoch's slots and may never be reported.**

§5.3. If `pperr` is host-zeroed per launch rather than accumulated per process,
a timeout in one of the 500 warmup launches leaves un-zeroed slot rows and is
never printed. The corruption then shows up as a small positive bias in `out`
for a handful of tokens — plausible, gate-passing, and untraceable. **Check H4.**

### 6.5 **Cross-XCD M6 front stall in Policy 3 looks like a performance result, not a bug.**

With per-XCD M6 ticket banks, XCD `x` produces only chunk `g = x` of every tile,
so a tile is ready only when **all eight** XCDs have produced their chunk. If the
backlog threshold sheds CTAs unevenly across XCDs, one bank falls behind and the
global tile front stalls while seven XCDs' M7 pools spin. **Nothing is wrong; the
answer is right; the number is bad and the cause is invisible in every log line
we emit.** Mitigation: evaluate the role-switch threshold on a *global*
M6-remaining count, and instrument per-XCD M6 bank values into two spare
`mps_state` words before believing any Policy-3 number.

### 6.6 **`g = 33` is a raw bit read with no mode gate — a sweep point that forgets it silently loses ~500 µs.**

`P2:272` reads the throttle as `((m7cfg >> 8) & 0x20)` directly from the packed
word; it is **not** a field of `struct config` (`ADP:61-68`) and **not** surfaced
by `physical_g` or any accessor (`mode12_protocol_map.md` §6 notes "Consumers
outside the phase-2 body cannot see it"). A mode-10/11 sweep point written with
`g=1` runs unthrottled, loses the ~500 µs exp_21 recovered, and **looks exactly
like the split mechanism failing.** Every config line in the ladder must carry
`g=33`, and the screen CSV's `cfg` column must be read, not assumed.

### 6.7 **`effective_flush_rows` not updated: the arm is correct but confounded.**

If `flush_rows` carries `C6/4` and `effective_flush_rows` (`ADP:233-235`) is not
extended to modes 10/11, the service pool's flag-batch depth becomes `C6/4`
(8–56) instead of the ratchet's 16. Every gate passes; the arm now differs from
the ratchet in **two** variables and the `C6` curve is partly a `flush_rows`
curve. Silent confound, not silent wrongness — but it invalidates the
measurement, which is the same thing for our purposes.

### 6.8 **`run_service`'s mode branch is an exact `== 12`.**

`ADP:787` is `if (env.mode == kModeRemoteAccum)`. Modes 10/11 that are added to
`mode_is_stream` and `mode_is_direct_accum` but **not** here fall through to the
payload-push branch (`ADP:816-832`), which reads `slice_group_src` = **`part`**
(`ADP:491-495`) — a buffer mode 12 never writes — and stores it over the
correctly accumulated slot rows. Result: `out` ≈ 0 and `max_abs` fires loudly.
**Loud, therefore acceptable — listing it because it is one of seven predicates
and the in-source warning at `KRN:296-303` exists precisely because someone got
this class of thing wrong before.**

### 6.9 **Skewed routing changes `num_tiles` and can make the M6 front and the M7 demand rate diverge without any log line saying so.**

Correctness is unaffected (the per-tile dependency is exact). But the balance
point `C6` is a function of the route, and a `C6` tuned on uniform routing may be
far off under `skewed_hot`. If we ever ship a tuned `C6`, it is tuned to one
route family. Ladder L7 exists for this; the honest statement in `LESSONS.md`
must say which route the `C6` was tuned on.

### 6.10 **The `A2q` bounded descriptor hides a coverage bug rather than reporting it.**

`a2_num_records = nvi[0] · kInter` (`P2:302-306`) makes out-of-range A2q reads
return zero instead of faulting. That is a memory-safety feature and a debugging
liability: a tile-index bug in the split's M7 pool can read zeros and produce a
zeroed output tile — which, per §6.3's arithmetic, is a `6 × 10⁻⁵`-class
perturbation. Same detector (L9).

---

## 7. Mandatory pre-run checks

These are not suggestions. Each is cheap and each closes one of the holes above.

| # | check | how | blocks what |
|---|---|---|---|
| **C1** | **Does `fence(RELEASE,"agent")` emit an L2 writeback on gfx950?** | Disassemble `k0pf6gm_mps_mega.hsaco` (resolve `latest/` with `readlink -f`; `harness_recipe.md` gotcha 3) and inspect the instructions between the `__syncthreads()` at the end of the M6 task and the `global_atomic_add` on `a2_done`. Grep for `buffer_wbl2` / `buffer_inv` in that window. | **H1.** If absent, the `producer_drain_release<agent>` change in `k0p6_a2_arrive` is **mandatory**, not recommended. |
| **C2** | Resource tuple unchanged | `-Rpass-analysis=kernel-resource-usage`; require ArchVGPR 256 / AGPR 256 exactly, scratch ≤ 144 B, LDS ≤ 155,520 B, MFMA census 96+84 = 180, **zero scratch ops inside either MFMA K-loop** | the §6.4 register risk in `design.md`; a scratch op inside an MFMA span is a stop |
| **C3** | Fresh `.hsaco` | `K0P6_MPS_SRC_REV 23 → 24` in the same commit; confirm `hsaco_before != hsaco_after` in the screen CSV (columns 34-35). **Directory mtimes are touched on a cache hit; `latest/` is a symlink dir.** | measuring the previous kernel. exp_04 lost a whole measurement to exactly this. |
| **C4** | The L0 null | mode 10 at `C6 = 240` must reproduce the mode-12 ratchet within noise | that the vendored phase-1 file and the descriptor-derived M6 start/stride are a null. **A miss invalidates every later point in the ladder.** |
| **C5** | Task-space exactness | ladder L9 (`part_done == 16`) at every new `C6` before the timing run | **§6.3**, the quietest bug in the design |
| **C6** | Readiness edge is real | ladder L8 (DQ2 NaN poison) at the best point | **§6.1**, the failure the campaign structurally cannot see |
| **C7** | All seven mode predicates | grep the tree for `== 12`, `== 13`, `12ull`, `13ull`, `kModeRemoteAccum`, `kModeDirectRows` and confirm each site is either widened or deliberately excluded | §6.8 |
| **C8** | Overlap actually happened | `timestamps=1`; require `M7_DONE − M6_DONE → ~0` | F3 in `design.md`. A mis-built arm must not be logged as a kill. |
| **H4** | **Is `pperr` zeroed per launch or per process?** | read the harness (`e004pf_k0pf_ab.py`) for where slot 42 is written; the device never resets it | §6.4 — decides whether a warmup timeout is reportable |
| **C9** | Every sweep point carries `g=33` | read the `cfg` column of the screen CSV, do not trust the config file | §6.6 |

---

## 8. Verdict

**Conditional signoff.** The design is sound *as a protocol* — the wait graph is
a chain so deadlock is structurally impossible; every wait is bounded and
`pperr`-watching; no arrival target, epoch tag, or buffer lifetime changes; the
`row_rem` self-clean survives reordering; and `A2q`'s capacity sizing genuinely
removes the need for a slot-lifetime protocol.

**Three conditions, in order of how strongly I hold them:**

1. **C6/L8 (the DQ2 poison arm) is not optional.** This experiment's central
   edge is currently vacuous, the campaign's fixed inputs make a broken edge
   numerically invisible, and a broken edge would post the *best* number in the
   sweep. Without L8 we cannot distinguish "the overlap worked" from "the
   readiness protocol is absent".
2. **H1 must be resolved by C1, and if the answer is "no writeback", the
   `producer_drain_release<agent>` change lands before any kept number.** The
   library already contains the right primitive; the operator open-codes a weaker
   one.
3. **C5/L9 (`part_done == 16`) before every timing run.** A doubly-covered M7
   task is a `6 × 10⁻⁵` relative error and passes every gate we have. The
   detector is free.

**Two validation rules that are derived, not stylistic:** `C6 ≥ 32` (spin-limit
margin, §5.2) and `C6 % 8 == 0` with `C7 ≥ 8` (the `nc → XCD` invariant asserted
at `P2:77`, §5.4).

**One thing I could not settle and will not paper over:** whether an agent-scope
release writes back the per-XCD L2 on gfx950. Everything in §2 H1 hinges on it,
the tree contains no disassembly that answers it, and `SYNC:160-174`'s own
comment tells us not to assume the answer for a new target. Until C1 is run,
treat E1 as **unproven**, not as sound.

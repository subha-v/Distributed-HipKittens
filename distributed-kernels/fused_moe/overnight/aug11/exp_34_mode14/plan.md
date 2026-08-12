# exp_34 — mode 14, "coarse readiness": the design AS BUILT

Status: **code complete, CPU gate green, NOT RUN.** No GPU job was executed for
this experiment; another agent held the exclusive lease. Everything below is
compile-time and ISA evidence only.

Base: branch `codex/distributed-hipkittens-scaffold`, HEAD `ca5b683f`.
Design source: `../exp_30_coarse_readiness/design.md` + `../STATUS.md` §exp_30.

---

## 1. What mode 14 is

Mode 14 = **mode 12's remote-accumulate transport, verbatim**, with the **entire
per-row readiness protocol deleted** and replaced by one rank-local grid barrier,
8 epoch publishes and 8 polls per CTA.

Deleted per rank per epoch (counts from exp_30 §7):

| structure | ops | how it dies |
|---|---:|---|
| `nc_arr` acq_rel RMWs | 535,040 | no event is ever enqueued, so the drain never runs |
| `pushed` RMWs | ~314,000 | same |
| per-row `row_ready` publishes | ~19,600 | replaced by 8 publishes |
| `row_rem` self-clean stores | ~19,600 | the parity self-clean is skipped for mode 14 |
| M8 per-row `poll_epoch_system` | ≥21,504 | compiled out (`Ready=true`) |
| event queue + `ev_next` ticket | ~16,700 | `k0p6_mps_task_done` returns early |
| **total** | **~926,000** | replaced by **1 grid barrier + 8 publishes + 8 polls/CTA** |

Also deleted, and this is a **second deletion bundled into the rung** that the
reviewer should price separately: mode 14 pays **no per-task VMEM drain at all**
(~2,840 `s_waitcnt vmcnt(0)` + `__syncthreads()` per CTA). It joins modes 12/13
in `k0p6_mps_task_drain`, and then its deferred flush is inert because nothing is
buffered. This is sound — the only thing that drain ever ordered was the event
publication mode 14 deletes, and the epilogue's remote RMWs are drained by the
same threads that issued them in `release_cta_payload_system()` at the
rendezvous. It is nevertheless a second variable in the mode-12 → mode-14 step.

Because the pool's three jobs (arrival bookkeeping, payload push, flag
publication) are all gone, **at `C = 0` mode 14 is a homogeneous megakernel**.
Its number is to be **banked as the price of the readiness protocol, not
ratcheted as a role-split result.** That degeneration is the finding.

---

## 2. Files changed and the real line numbers

Both files are in `distributed-kernels/fused_moe/`. Line numbers are post-edit.

### `moe_mps_adapter.cuh` (+69 / −5)

| lines | change |
|---|---|
| 33–37 | `K0P6_MPS_ERR_M7DONE 33554432` — donor bit 25, **reused**. Its donor meaning is "row_ready combine poll timeout" and mode 14 is the one mode that compiles that poll out, so the two meanings are never both reachable. |
| 177–183 | comment: mode 14 is deliberately **NOT** in `mode_is_stream` (see §4 deviation D1). |
| 232–255 | the mode-14 charter comment + `kModeCoarseReady = 14u`. |
| 283–285 | new `mode_is_coarse(config)`. |
| 291–295 | `mode_is_direct_accum` now includes 14. |
| 328–337 | `skip_dead_part_zero` admits mode 14, so the rung can carry the ratchet's `g = 353`. |
| 419 | skip-bit validator arm accepts mode 14. |
| 422–428 | **new rejection**: mode 14 + dual-write detect bit. The detector's M8 instantiation is tested *before* the direct-accumulate branch and it polls `row_ready`, so that config would spin on a flag nobody publishes. Fail closed. |
| 431 | `if (c.mode > 13u)` → `> 14u`. This is the validator the brief called out; the brief's second site near `:271` does not exist on current source (§4, D5). |

### `k0pf6gm_device_tile_mps.hip` (+231 / −15)

| lines | change |
|---|---|
| 69–74 | pperr table: bit 25's second meaning documented. |
| 107 | **`K0P6_MPS_SRC_REV 26 → 27`** (mori's JIT key hashes this file's content, not flags). |
| 338–345 | `k0p6_mps_task_done`: mode 14 takes **neither** arm — no event enqueue, no parity `part_done` bump. |
| 373–387 | `k0p6_mps_task_drain`: `m != 12 && m != 13 && m != 14`. |
| 407–411 | `k0p6_mps_task_done_maybe_defer`: mode 14 buffers nothing, so `k0p6_mps_task_flush_defer` returns on its first line at every task head. |
| 529–541 | `k0p6_mps_m8_batch` gains `bool Ready = false` as its **last** template parameter, so every pre-exp_34 instantiation is unchanged. |
| 579–585 | the per-row `poll_epoch_system` is behind `if constexpr (!Ready)`. |
| 593–605 | the readiness `__ballot` is behind `if constexpr (!Ready)`; the per-batch `acquire_payload_system()` is **hoisted out of the if and always runs** (§4, D3). |
| 762–776 | entry guard: mode 14 requires `K0P6_D_ROW_READY` present **and** `world*MAXTOK < T_loc_max`. |
| 1557–1561 | comment: mode 14 enters no drain, so `K0P6_MPS_ERR_SERVICE` (bit 26) can never fire — its absence is a positive path check. |
| 1686–1733 | the M7.5 block's guard widened to `mode_is_parity_publish(cfg75) \|\| coarse75`, with `coarse75` behind `readfirstlane`. **This block is shared, not duplicated** (§3). |
| 1758–1772 | stage (3): mode 14's publish arm — CTA 0's leader, one cell, 8 targets. |
| 1804–1839 | stage (4): the cross-rank poll (tid 0 walks all 8 words) + the `__syncthreads()`. |
| 1908–1911 | M8 header comment gains the mode-14 path. |
| 1939–1944 | `m8_coarse`; `m8_dynamic = mode_is_stream \|\| m8_coarse`. |
| 1963–1970 | `payload_ok` mask gains bit 25 for mode 14 — the whole fail-closed path now that the ballot is gone. |
| 2049–2068 | M8's mode-14 branch: mode 12's address form and consume-and-zero with `Ready=true`. Tested **before** the `mode_is_direct_accum` arm, which mode 14 also satisfies. |

Untouched on purpose: **M9**, `run_service`, the M7.6 drain, the mode-1 bulk
push, `is_service_cta`, and `n2_phase{1,2}_gm_mps.cpp`.

---

## 3. Ordering and epoch argument (PROTOCOL-REVIEW PACKET)

### 3.1 The happens-before chain, producer payload → consumer read

Let rank *p* be a producer and rank *q* the owner of a slot row.

| # | edge | site |
|---|---|---|
| E1 | *p*'s M7 epilogue accumulates into *q*'s `slots` with `flat_atomic_pk_add_bf16` through the peer table | `n2_phase2_gm_mps.cpp` epilogue, unchanged from mode 12 |
| E2 | **release, per CTA**: `release_cta_payload_system()` = per-thread `s_waitcnt vmcnt(0)` → CTA barrier → one L2 writeback for the CTA. Every thread that issued an RMW executes the `vmcnt(0)` itself. | KRN:1733 (shared with mode 0/1) |
| E3 | `hkp::grid_barrier` on `K0P6_D_GBAR`, then `acquire_payload_agent()`. After this, **all 256 CTAs of rank *p*** have drained and written back. | KRN:1737–1743 |
| E4 | **release, once**: CTA 0's leader issues `release_signal_batch_system()` (`thread_release<system>`) and then, in the **same thread**, 8 relaxed stores of `epoch32` — self AGENT, 7 peers SYSTEM. | KRN:1750, 1765–1772 |
| E5 | **acquire, once**: on rank *q*, tid 0 of every CTA bounded-polls `m7_done[p] >= epoch32` with `poll_epoch_system` (relaxed SYSTEM load). | KRN:1831–1839 |
| E6 | `__syncthreads()` — transfers "tid 0 observed all 8 flags" to every thread of the CTA. | KRN:1838 |
| E7 | **acquire, per reader**: each thread's own `acquire_payload_system()` = `thread_acquire<system>` (the `buffer_inv` that invalidates this rank's stale L1/L2 lines) before it loads any slot byte. | KRN:601, inside `k0p6_mps_m8_batch` |
| E8 | M8 reads and then zeroes the slot row (`Zero=true`). | KRN:614–622 |
| E9 | M8's zero → any peer's **next** epoch accumulate: M9's `release_signal_batch_agent()` + 7 `retired` pokes, and M0's `retired >= epoch-1` wait. **Unchanged from mode 12.** | KRN:2119–2130, KRN:737–739 |

E2 → E3 is why one CTA's release at E4 can speak for all 256: the grid barrier
orders every CTA's writeback before any flag store. That is verbatim the argument
the mode-0 parity publication already relies on (its stage-(2)/(3) comment).

**E4's scope is SYSTEM, not the AGENT release exp_30 §3-R3 specified.** The flag
certifies that this rank's *remote* payload is visible to peers, which is exactly
the job of the parity publication's second-stage release, and that one is system.
M9's agent release is a different job (ordering local consume-and-zero for a
retirement gate). System here costs one extra fence per rank per epoch. This is a
deliberate strengthening of the design; if the reviewer wants it weakened to
agent, that is a separate, argued change.

### 3.2 Why mode 14's slot lifetime is *safer* than mode 12's

In mode 12, `q`'s M8 may read and zero slot row `(p,row)` as soon as *that row's*
16 chunks have arrived, while *p* is still running other M7 tasks. In mode 14,
`q` reads nothing until it has observed that **every** peer's whole M7 is done.
No peer can still be accumulating into any row `q` is about to consume. The
mode-12 argument therefore carries over with slack, and E9 is untouched.

### 3.3 Epoch discipline: no wrap, no race, no reset

`m7_done[src]` holds `epoch32` and is **never reset**. This is deliberate, not
lazy: a *counter* would need a per-epoch reset with no intra-rank ordering
against the next epoch, which is precisely the defect exp_56 fix (6) removed from
`dest_counter`. Concretely, rank *p*'s M7.5 of epoch *e+1* is **not** ordered
after rank *q*'s M0 of epoch *e+1* (`q`'s M0 only waits on `retired >= e`), so a
reset performed in *q*'s M0 could erase a publish *p* had already made. Epoch
values need no reset and so cannot be erased.

**A stale word from the previous epoch is safe by construction.** At the start of
epoch *e*, `m7_done[p]` holds either `0` (never published) or `e−1`. The poll is
`>= e`, so both fail and the CTA waits. The value is monotone (`epoch32` is
`mega_count + 1`), so a positive observation can only have been produced by this
epoch's publisher.

Wrap: `epoch32` is the low 32 bits of a 64-bit epoch, so the compare degrades
after 2³² epochs. That is the **existing** `row_ready` discipline, not a new
exposure, and 600-epoch soaks are 7 orders of magnitude away.

### 3.4 Deadlock argument

1. Every CTA reaches the local grid barrier: M7's task loop is finite and
   data-independent, and mode 14 skips the drain, so there is no blocking
   construct before the barrier.
2. The barrier cannot hang: `local_grid_epoch` + `fail_closed{pperr, 2097152}`
   is bounded and sets bit 21 on timeout.
3. The cross-rank poll cannot hang: `poll_epoch_system` is bounded by
   `spin_limit` and sets bit 25 on timeout.
4. No cyclic wait: rank *p*'s **arrival** at the rendezvous depends only on *p*'s
   own M7 and its own local barrier — never on any peer's **exit**. So all 8
   arrivals are unconditionally reachable, hence all 8 exits are.

Fail-closed: any of bits 21/24/25 makes M8's `payload_ok` false ⇒ `T = 0` ⇒
empty M8, and M9 still retires the launch, so the node does not wedge.

### 3.5 Barrier-divergence hazard, found and fixed during the build

An earlier revision put the poll and the `__syncthreads()` inside
`if (!error_bit_set_agent(pperr, …))`. **That is a hang.** `payload_ok` is a
relaxed read of a concurrently-written `pperr`, so two waves of one CTA can
disagree, and the wave that saw the bit would skip the barrier the other wave is
sitting on. In the delivered code the `__syncthreads()` is gated **only** on
`coarse75`, a grid-uniform decode of one descriptor word; `payload_ok` gates only
the poll, which contains no barrier.

### 3.6 The cell's address, and why it cannot alias

`m7_done[src] ≡ row_ready[src * T_loc_max + T_ext]`. Every other writer of
`row_ready` is bounded by `r < T_ext` (`flush_pending`, the parity publication,
the mode-1 publication), so index `T_ext` is unreachable to them. The entry guard
rejects mode 14 unless `world*MAXTOK < T_loc_max`; today 32,768 < 40,960.

**Assumption the reviewer should check on the node**: that `row_ready` is
zero-initialized at allocation. Mode 12 and the parity arm already depend on this
(their first-epoch poll is `>= 1` against a never-written cell), so it is not a
new dependency, but I have not verified it in the harness and it is the one
premise of §3.3 I did not prove.

### 3.7 Things I am least sure about, stated plainly

1. **E6 + E7 rather than "the observing thread reads the payload."** Formally the
   thread that observed the flag (tid 0) is not the thread that reads most of the
   payload. The chain relies on `__syncthreads()` to carry the observation within
   the CTA and on every reader executing its own system acquire. The kernel's M0
   already uses exactly this shape (`tid < world` polls `retired` → grid barrier →
   convergent acquire), so it is the house idiom, but it is an idiom rather than a
   proof and it is the first thing I would want a reviewer to attack.
2. **Bit 25 reuse.** Sound only because mode 14 compiles the per-row poll out. If
   anyone later gives mode 14 a per-row poll, the two meanings collide.
3. **The per-task drain deletion** (§1) is a second variable in the rung.
4. **`C > 0` for mode 14** is legal and reserves idle tail CTAs (§4, D2). Nobody
   has run it; the M7 stride/reservation consistency argument is static only.

---

## 4. Where exp_30's design was wrong or unbuildable against the real source

**D1 — "include 14 in `mode_is_stream` only for `m8_dynamic`" is not
expressible.** `mode_is_stream` gates three unrelated things: the M7.6 drain
entry (`service_cta_here`, KRN:1563), the validator's "a stream mode needs a
pool" rule (`ADP` `mode_is_stream(c) && C == 0 → invalid`), and M8's dynamic
ticket loop. Adding 14 would have required exempting it at the first two. Built
instead as: `mode_is_stream` **unchanged**, and M8 asks for
`mode_is_stream(m8cfg) || mode_is_coarse(m8cfg)`. One site, and the predicate
stays byte-identical for every pre-exp_34 mode. The design's "exempt mode 14 from
the `reserved_comm_ctas != 0` requirement" then needs no code at all.

**D2 — "`is_service_cta` must be false for mode 14" would have silently
double-covered M7 tasks.** `k0p6_mps_stride` returns `nct − C` unconditionally
(KRN:290). If `is_service_cta` were forced false while `C > 0`, the C tail CTAs
would run M7 with `start = bid, stride = 256 − C` and **re-execute tasks other
CTAs already did** — exp_25's named silent failure ("under-coverage is loud;
over-coverage is silent"). `is_service_cta` is therefore left **unchanged**: at
`C = 0` it is false for every CTA (homogeneous, the intended arm), and at `C > 0`
it reserves an idle tail whose CTAs skip M7 and join only M8 — which is exactly
the placement arm exp_37 asks for, now sound instead of corrupt.

**D3 — the hoisted `acquire_payload_system()` (design S3) was dropped.** The
acquire was never the cost; the ≥21,504 cross-GPU **polls** were. Keeping the
acquire per-batch means the thread that reads the payload performs its own system
acquire, which makes the ordering argument local to one thread. `Ready` now
compiles out only the poll and the ballot.

**D4 — `m7_done` could not be a new symmetric array.** Adding a descriptor slot
needs host-side allocation in `moe_host_abi.hpp` and the harness driver, neither
of which is owned by this change (and neither of which could be validated
CPU-only). Built as the spare tail of `row_ready` instead (§3.6). This is
strictly better for the epoch argument, since `row_ready` is already an
epoch-valued, never-reset, symmetric array.

**D5 — the "second, WRONG validator site near line 271" does not exist.** Current
source has exactly one `c.mode > 13u`, at `ADP:431` (STATUS already corrected
this; confirming it against the real file). `ADP:271` is prose in the
`skip_dead_part_zero` rationale.

**D6 — mode 14 had to be admitted to `skip_dead_part_zero`**, which the design did
not mention. Without it, `config_is_valid` rejects `g = 353` for mode 14, so the
rung would have had to run at `g = 33` and the mode-12 → mode-14 comparison would
silently also have been an exp_24-A comparison worth ~+51 µs of plan time.

**D7 — mode 14 + the dual-write detector must be rejected**, also unmentioned.
The detector's M8 arm is tested before the direct-accumulate arm and polls
`row_ready`, so that config would hang. Now a validator rejection.

**D8 — the design's line numbers had all drifted** (e.g. M7.5 is at 1686–1796,
not 1573–1613; M9's agent release is at 2119, not 1874). Every citation in this
document is against `ca5b683f`.

---

## 5. The resource finding that shaped the build

See `build_log.md` for the numbers. Summary: **a separate M7.7 phase cost
+16 B/lane of scratch (128 → 144)**, and a six-arm probe showed the cost was not
the IRIS descriptor, not `peer_ptr`, and not the extra M8 instantiation — it was a
second copy of the release/barrier/acquire/`payload_ok`/system-release sequence.
Sharing the M7.5 block, moving the poll to a single thread, and dropping the
hoisted acquire put ScratchSize back to the ratchet's 128 B/lane.

**One gate item does not come back**, and it is not fixable inside this change's
ownership: the build re-triggers exp_26's scratch migration into the
remote-atomic epilogue — **96 of the 282 `flat_atomic_pk_add_bf16` have a scratch
op within 40 instructions ahead, against 0 in the base**. Seven different
arrangements of the same mechanism produced byte-identically the same
`128 / 217 / 17 / 155 / 96`, so this is a property of total pressure in this
function, not of where the mode-14 code sits. exp_26 attributed the same
migration to `n2_phase2_gm_mps.cpp:145` (the `peer_tab[xr >> maxtok_sh]` base
load) and measured its cost, bundled into mask 1, at **+2.81 µs (t = 0.61,
null)** — so it is a real risk but a small and previously-measured one. STATUS's
standing follow-up ("relieve one live VGPR in phase 2's epilogue and bit 0
becomes free") is the fix, and it lives in a file this change does not own.

---

## 6. Gate ladder still owed (all of it needs the GPU)

Nothing below has been run.

1. `pperr == 0`, and **bit 26 (`K0P6_MPS_ERR_SERVICE`) must never be set** — that
   is the positive check that mode 14 skipped the drain.
2. Correctness + the existing negative control + the exp_32 poison self-test.
   `K0_MOK_POISON_OUT` must stay `1`: mode 14's failure mode is exactly "a row
   never gets written", which identical per-epoch inputs otherwise hide.
3. 600-epoch soak, `K0_MPS_SOAK_ITERS=600`, `poison=0`.
4. Only then the 5-rotation campaign, arms `production,pf6gm_mega,mps_mega`.
5. `.text` fingerprint the binary that actually ran, and confirm a fresh
   `.hsaco` mtime (`stat -L`) — SRC_REV was bumped, so a cache hit is a bug.

First config to try: `K0_MPS_CFG='C=0,g=353,mode=14,flush_rows=16'`. `g = 353`
carries the ratchet's throttle-enable + depth-4 + skip-dead-part-zero, so the
only variable against the `mode=12` ratchet is the readiness protocol.
`flush_rows` is unused by mode 14 but must still be present and in `1..64`, or
the config silently *looks* valid.

Pre-registered band **5,990–6,440 µs**, point estimate ~6,215. Ratchet is
6,482.7 µs / 0.8408×. **A result above 6,568 µs falsifies the granularity rung**
and gets reported as such. Read the estimate pessimistically: exp_27's terms came
in 43 % and 83 % below their point estimates.

The control arm `K0_MPS_CFG='C=0,g=481,mode=14,flush_rows=16'` (= `g | 0x80`) runs
the same ladder; see §7. Both mode-14 arms must clear the ladder before either
number is quoted.

## 7. exp_34 C — the drain-retention bit (added after the protocol review)

The review found that mode 14 as first built changes **two** things against
mode 12: readiness granularity, and the per-task VMEM drain, which mode 14
deletes because the only thing it ever ordered was the event publication that the
per-row protocol's removal makes meaningless (~2,840 `vmcnt(0)` + 2,840
`__syncthreads()` per CTA). One waterfall rung carrying two variables is what the
charter forbids, so the deletion is now selectable.

- `kCoarseKeepDrainBit = 0x80` in `g` (adapter `:290`), which the packed config
  word carries at bit 15 (`kCoarseKeepDrainWordBit = 0x8000`, adapter `:295`).
  Legal **only** on mode 14 — rejected for 12/13 at the validator (`:456`) so a
  mistyped arm cannot look like the control. Device side: `KRN:419-431`.
- With the bit set, `k0p6_mps_task_done_maybe_defer` buffers the task identity
  with `gcount == 0`. `k0p6_mps_task_flush_defer` therefore still pays its
  `vmcnt(0)` + `__syncthreads()` at the next task head — the same place mode 12
  pays it — and its publication loop runs zero iterations.
- The bit is tested on the config word the hook already loads, so there is no
  extra descriptor read, and the resource tuple is byte-identical to the
  pre-selector build (see `build_log.md`).

Decomposition the waterfall can now report: mode 12 → **mode 14 + 0x80** =
granularity (including the event publication it deletes, which cannot be
retained: no consumer, and an event would make bit 26 ambiguous) → **mode 14** =
the drain deletion alone. The first delta must be labelled as the combined
protocol mechanism, never as granularity alone.

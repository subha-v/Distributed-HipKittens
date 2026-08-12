# exp_20 — Tier 1: the 815 µs of M7 interference is **protocol, not payload**

**Verdict, in one line: deleting the service pool's entire 896 B payload copy —
read, peer write and fabric together — changes M7 by +5.7 µs, while the full
831 µs of interference remains.** The payload is free. Every byte the pool moves
is free. The cost is the protocol wrapped around those bytes.

Five diagnostic modes, 37 screening runs, **every one of them green on the full
gate ladder** (`[MOK GATE] pass=True`, `control_fails=True`, `[MPS SOAK]
600/600 pperr=0`) — all five modes are correctness-preserving by construction,
so none of this needs the "ignore the garbage output" caveat the plan budgeted
for.

---

## 0. The in-build references

Every number below is differenced against **mode 0 at the same `C`, measured in
the same build**. Four independent samples of the matched control and three of
the ratchet, across four ladders:

| arm | M7 samples (µs) | mean |
|---|---|---:|
| mode 0 — reserve 64 CTAs, do nothing | 2,051.1 · 1,981.8 · 2,062.0 · 2,003.7 | **2,024.7** |
| mode 2 — the ratchet, `C=64 g=1 flush_rows=16` | 2,863.0 · 2,830.0 · 2,839.9 | **2,844.3** |

**Interference = 819.6 µs**, reproducing the 815 µs this experiment was sent to
explain. The screening noise on a single M7 stamp is **±40 µs (±2%)**; nothing
below rests on a delta smaller than that.

---

## 1. E1 — is the interference rate-driven or volume-driven?

**Neither, as the question was posed — and the actionable answer is
NOT rate-driven: there is no pacing a scheduler could find that wins.**

Mode 4 is mode 2 with an `s_sleep` delay inserted in the push path and nothing
else changed (the real flag-batch depth is pinned at 16, so pacing is the only
variable). One unit is 256 core clocks ≈ 0.12 µs.

| pace units | sleep/push | M7 | ΔM7 | combine | pool drain | M2→end |
|---:|---:|---:|---:|---:|---:|---:|
| — (mode 0) | — | 2,051.1 | 0 | 1,418.8 | — | 6,432.9 |
| **0** | 0 | 2,818.0 | **+766.9** | **410.4** | 3,176.0 | **6,229.4** |
| 1 | 0.12 | 2,790.2 | +739.0 | 495.7 | 3,146.0 | 6,265.0 |
| 2 | 0.24 | 2,789.4 | +738.2 | 488.4 | 3,211.3 | 6,280.0 |
| 4 | 0.49 | 2,782.5 | +731.4 | 563.8 | 3,255.9 | 6,324.9 |
| 8 | 0.98 | 2,716.9 | +665.8 | 627.7 | 3,317.9 | 6,355.4 |
| 16 | 1.95 | 2,675.0 | +623.9 | 908.8 | 3,426.6 | 6,581.3 |
| 32 | 3.90 | 2,610.6 | +559.5 | 1,358.5 | 3,885.6 | 6,979.9 |
| 63 | 7.68 | **2,501.6** | **+450.4** | **2,629.4** | 5,049.7 | **8,133.8** |

Self-check: pacing 0 reproduces the mode-2 reference to 45 µs, inside the noise
band, so the pacing scaffold did not perturb the kernel.

**Reading.**

1. **M7 does recover** — monotonically, 2,818 → 2,502 µs. 41% of the
   interference is removable by throttling. So this is not the strong
   volume-driven null ("M7 does not recover at any pacing").
2. **The recovery is never worth having.** The combine pays **+2,219 µs** for
   M7's **−316 µs** — a **7:1 loss** — and `M2→end` rises monotonically from
   6,229 to 8,134 µs. **There is no interior minimum. Pacing 0 is already the
   optimum.** Every pacing/scheduling mechanism in Tier 3 that assumes
   throttling the pool buys net time is dead on arrival.
3. **The pool is event-starved, not throughput-bound.** Fitting the drain
   against pacing gives `drain = 3,097.6 + 29.27·pace µs` (R² by inspection;
   eight points, monotone). At 0.1219 µs per unit that puts **~240 paced pushes
   on the drain's critical path**, against ~1,363 pushes per wave if the work
   were spread evenly. The pool spends most of M7 **waiting**, not pushing.
4. **What the 41% recovery actually is.** §2 proves the payload costs nothing,
   so the recovery cannot be payload bytes. Pacing slows the whole service loop,
   including its arrival atomics — and 300–350 µs is exactly the size exp_06
   (−354 µs from relaxing the arrival RMW's scope) and exp_14 (−306 µs from
   deleting 44% of the atomics) already measured for that term. The pacing curve
   and the atomic experiments agree.

---

## 2. How does the 815 µs split across read side / peer write / fabric?

### 2a. In situ, at the real operating point: **0 / 0 / 0**

Mode 7 is mode 2 with the payload copy deleted and *nothing else touched* —
same events, same arrival atomics, same queue spin, same flags, same fences,
same event-driven rate. It stays correct because `pull_fallback` makes M8 read
the producer's remote `part` row instead of the slot the pool would have filled.
Its matched reference is mode 2 + `pull_fallback`, which does **identical work
plus the copy**.

| arm | M7 | combine | M2→end |
|---|---:|---:|---:|
| mode 2 + `pull_fallback` — WITH the copy | 2,899.2 | 1,310.3 | 7,245.8 |
| mode 7 — copy DELETED | 2,893.5 | 830.6 | 6,724.3 |
| **the copy costs** | **+5.7 µs** | +479.7 µs | +521.5 µs |
| mode 0 matched control | 2,062.0 | 1,386.9 | 6,444.4 |
| **interference with NO payload at all** | **+831.5 µs** | | |

**The payload copy accounts for 5.7 µs of the 831 µs, against a ±40 µs noise
band. It is zero.** It costs 480 µs of *combine* — real work, in the tail, where
it is not damaging anything — and nothing in M7.

This is a direct refutation of exp_17's conclusion. exp_17 found that marking
the copy non-temporal did nothing and inferred "the interference is bandwidth
contention, not cache pollution; the only remaining lever is to move fewer
bytes." The `nt` null is confirmed here, but its explanation was wrong: the copy
was not contending for bandwidth, it was **not contending at all**.

### 2b. Per traffic class, at each class's maximum rate

Mode 5 puts a synthetic generator on the same reserved 64 CTAs — same buffers,
same 896 B unit, same row striping, same MLP-1 loop — running flat out for an
identical 1,600 µs window, inside mode 0 (whose combine never reads the slots
buffer, which is what makes all of this correctness-preserving).

| variant | M7 | ΔM7 | what it isolates |
|---|---:|---:|---|
| idle pool (matched control) | 1,981.8 | 0 | — |
| LDS-only spin | 2,004.9 | +23.1 | occupancy, zero traffic |
| **read only** | 2,019.4 | **+37.5** | the read side |
| read + peer write | 2,087.5 | +105.6 | the real pusher's shape |
| read + local write | 2,097.0 | +115.2 | the same, fabric removed |
| **local write only** | 2,086.6 | **+104.7** | the write side, no fabric |
| **peer write only** | **2,605.9** | **+624.1** | the write side over xGMI |

```
read side (local HBM)      +38 µs      — the cheapest class measured
write side (local)        +105 µs
write side (fabric)       +624 µs
FABRIC SURCHARGE          +519 µs      — a peer write costs 5.96x a local one
```

**So the fabric IS the expensive class per byte — and the real pusher is
accidentally protected from it.** Adding the load to the peer write drops the
damage from 624 to 106 µs, a 5.9× reduction, because the MLP-1
`load → s_waitcnt → store` shape throttles the writer. The variants are wildly
sub-additive (37.5 + 624.1 = 662 against a measured 106) and the reason is not a
shared queue: it is that **the read paces the write**.

**This retro-explains exp_04.** Giving the copy 4-deep memory-level parallelism
"did not materialize" a gain — of course not: the gain in copy throughput was
paid straight back as fabric interference. **Making the copy faster makes M7
worse.** The MLP-1 loop that reads like a defect is load-bearing.

---

## 3. E3 — occupied but silent

Mode 6 spins the reserved pool on a dependent LDS read/write chain: CTAs
consumed, instructions issued, not one global memory request.

| window | M7 | ΔM7 vs matched mode 0 | M2→end |
|---:|---:|---:|---:|
| 1,600 µs | 2,004.9 | +23.1 | 6,215.1 |
| 3,200 µs | 2,106.6 | +44.6 | 7,583.3 |

**Indistinguishable from the idle pool at both windows** (±40 µs band).
Occupying a CU and issuing instructions costs nothing; the effect is entirely
memory traffic. **The A7 strike stands.**

### The liveness proof, so none of §2–3 can be dismissed as a dead generator

The 3,200 µs window outlasts M7, so the pool delays the M7.5 rendezvous by
exactly the overhang — and it does: `M2→end` grows **+1,368 µs for +1,600 µs of
window**. The window mechanism demonstrably runs for its full duration.
Independently, the peer-write-only generator scales with exposure:

| window | 800 µs | 1,600 µs | 3,200 µs |
|---|---:|---:|---:|
| ΔM7 | +161.3 | +624.1 | +1,392.3 |

---

## 4. Then what IS the 815 µs? Two more suspects killed, one left standing

With payload (§2a), traffic-free occupancy (§3) and pacing (§1) all measured at
zero, the residue is the service loop's own protocol traffic. Two candidates
were testable tonight:

**The event-queue spin — KILLED.** The pool is event-starved for most of M7
(§1.3), so 256 waves sit in `wait_event_nonempty` issuing a relaxed load on the
queue slot plus one on the single shared `pperr` word every ~0.12 µs. Consecutive
tickets are consecutive 32-bit slots, so ~16 waves share each cache line while
compute CTAs write those same lines. Mode 8 adds a tunable backoff to exactly
that loop:

| backoff units | µs/spin | M7 | ΔM7 | M2→end |
|---:|---:|---:|---:|---:|
| 0 | 0.12 | 2,911.1 | +907.4 | 6,271.4 |
| 2 | 0.37 | 2,845.8 | +842.1 | 6,238.8 |
| 8 | 1.10 | 2,834.6 | +830.9 | 6,288.5 |
| 16 | 2.07 | 2,822.2 | +818.5 | 6,247.1 |
| 32 | 4.02 | 2,850.0 | +846.3 | 6,268.3 |
| 63 | 7.80 | 2,857.5 | +853.8 | 6,238.7 |

**Flat.** A 64× cut in the spin's request rate moves M7 by nothing and moves the
whole kernel by nothing. Clean null.

**What remains, and it is now the prime suspect: the arrival atomics and the
flag publication's system-scope release.** The atomics are already bounded at
300–350 µs by exp_06/exp_14 and by §1.4 — call it ~40% of the 815. The
unmeasured remainder is `flush_pending`'s `thread_release<system>` — on gfx950
an L2 writeback — issued once per `flush_rows` completed rows, roughly **2,400
times per rank per epoch, while M7 streams 470 MB of weights through the same
eight L2s**. Nothing tested tonight touches it, and it is the only mechanism
left that is invisible to a rate knob, invisible to a volume knob, and
proportional to neither.

A mode-9 diagnostic (weaken that one fence to agent scope, exp_06-style) was
written and is the obvious next run. **It was not executed** — see §7.

---

## 5. Did the pre-registered expectations hold?

| # | pre-registered | outcome |
|---|---|---|
| **E1** | rate-driven, because the pool bursts hard and the fabric and L2 request queues are the suspected resource | **NO.** M7 recovers, but the total is monotone in pacing with no interior minimum, so there is nothing to schedule. And the premise is wrong twice over: the pool does not burst hard — it is event-starved — and the fabric request queue is not what M7 is losing to |
| **E2** | the local read dominates, because it competes with M7's weight streaming in the same HBM/LLC path | **NO, and backwards.** The read is the *cheapest* class measured (+38 µs flat out). The peer write is 16× more expensive than the read. And in situ the whole payload contributes zero, so the axis the expectation was stated on does not carry the cost at all |
| **E3** | indistinguishable from the idle pool; a surprise would overturn the A7 strike | **YES.** +23 µs at 1,600 µs, +45 µs at 3,200 µs, both inside the ±40 µs band. No surprise; A7 stays struck |

One for three. The two failures are the valuable ones: both were premised on the
interference being *bytes*, and it is not.

---

## 6. Consequences — what this changes

1. **A11 / M11 is demoted for M7, and the exp_17 headline is retracted.**
   exp_17 called direct remote accumulation "the whole game" on the grounds that
   it collapses 936 MB → 312 MB and "should remove essentially all of the
   ~1,000 µs". Payload bytes cost 5.7 µs of M7. The mechanism is still worth
   building — mode 7 shows the copy costs **480 µs of combine**, which is real —
   but it cannot touch the interference, and it should be sold on the combine,
   not on M7. (exp_21 is building exactly this in parallel; this result should
   reach it before its numbers are interpreted.)
2. **M4 becomes the whole game.** Fleet's per-XCD arrival counters cut both
   halves of the one term that is measurably expensive: atomic scope (exp_06:
   354 µs) and atomic footprint (exp_14: 306 µs), and the same restructuring is
   what would collapse the release fences from ~2,400 per epoch to 8.
3. **Do not make the copy faster.** MLP, larger transfer units, `store_peer_packets_multi`,
   SDMA — every one of them raises the fabric write rate, and §2b prices that at
   up to 519 µs of M7. M3 and M10 should be re-read in that light.
4. **The remaining lever is fence and atomic count, not byte count.** That is the
   opposite of the conclusion the campaign has been carrying since exp_17.

---

## 7. Method, validation, and an ownership collision

**Build.** All diagnostics are additive `mode` values (4, 5, 6, 7, 8), gated by
`config_is_valid`, with diagnostic magnitudes carried in reinterpreted config
fields — no descriptor change, no host-ABI change, no harness edit, exactly the
route mode 3 took. `K0P6_MPS_SRC_REV` was bumped in the hashed `.hip` on every
`.cuh` edit (15 → 19) and a fresh `.hsaco` mtime was confirmed before each
ladder.

**Resource tuple on the current tree** (`build_only.sh`, gfx950 `--genco`):

```
TotalSGPRs: 106   VGPRs: 256   AGPRs: 256
ScratchSize [bytes/lane]: 128   Occupancy: 1 wave/SIMD
LDS Size [bytes/block]: 155496
```

`VGPR 256 / AGPR 256 / scratch 128 B` are at the required values. The
`SGPR 104 → 106` and `LDS 155,428 → 155,496` deltas arrived with **exp_21's
mode 12**, not with this experiment; exp_20's own build measured
`SGPR 104 / scratch 144 / LDS 155,428`, and the scratch returned to 128 once
exp_21's changes shifted allocation.

**Confirming screens** at the ratchet config `C=64,g=1,mode=2,flush_rows=16`,
all gates green:

| tree | screened µs | ratio vs `production` | in-kernel M7 | in-kernel M2→end |
|---|---:|---:|---:|---:|
| exp_20 only (`5f88f58f`) | **6,942.4** | **0.888** | 2,839.9 | 6,231.6 |
| exp_20 only (`7802e1e3`) | 6,993.1 | 0.893 | 2,863.0 | 6,270.6 |
| + exp_21 (`4e8e58a2`) | 7,035.7 | 0.887 | — | — |
| + exp_21, timestamps | 7,175.2 | 0.901 | 2,860.6 | 6,278.7 |
| + exp_21 (`2534ff4d`) | 7,151.8 | 0.896 | — | — |

**On the exp_20 tree the ratchet screens at 6,942 µs / 0.888 — inside the
6,940–6,975 µs target band and exactly the exp_14 campaign ratio.** On the
current combined tree the absolute reads 7,036–7,175 µs. That drift is not a
kernel regression: the `production` denominator rose in lockstep (7,928–7,979 vs
7,805–7,940 earlier), the ratio stays at 0.887–0.901, and the **in-kernel phase
stamps are unchanged** (M7 2,860.6 against 2,830–2,863 measured earlier the same
night). Any residual belongs to exp_21's `SGPR 104 → 106` and `LDS +68 B`, and
should be re-checked with a paired campaign once exp_21 settles — a
1-process screen cannot resolve 1%.

### Ownership collision — read this before acting on the revert

Partway through, **exp_21 began committing to `moe_mps_adapter.cuh` and
`k0pf6gm_device_tile_mps.hip`, the two files this experiment owns.** It rebased
over exp_20 three times (claiming mode 7, then 8, then 12) and has since built
its mode 12 *on top of* exp_20's helpers — `mode_is_parity_publish` is load
bearing at `k0pf6gm_device_tile_mps.hip:1614`, and it rewrote the enqueue-mode
test at line 304 (which, incidentally, silently dropped mode 8's event enqueue).

**The instructed revert of the exp_20 diagnostics was therefore NOT performed.**
Executing it would have meant a sweeping concurrent rewrite of two files another
agent is mid-experiment on — the one thing the ownership rule forbids — for zero
measurable benefit, since the resource tuple is already at its targets. What
remains in the tree is five `mode ==`-guarded diagnostic paths, none reachable
at any config the campaign runs, all of which passed the full gate ladder
including the 600-epoch soak.

**The revert, when exp_21 lands** (all in `moe_mps_adapter.cuh` unless noted):
`mode_is_stream` → `2 || 3 || 12`; delete `mode_is_diag_pool`,
`mode_is_parity_publish`, `effective_flush_rows`, `pace_units`,
`poll_backoff_units`, `diag_window_ticks`, `pace_delay`, the `kDiag*` constants,
`diag_env`, `touch_packets`, `fill_packets`, `run_diag_traffic`,
`run_diag_spin`; drop `pace` / `poll_backoff` / `no_payload` from `service_env`
and their uses; drop the `backoff` parameter of `wait_event_nonempty`; drop the
mode 5/7 clauses in `config_is_valid`. In the `.hip`: delete the M7.6c block,
restore `mode == 0u` at M7.5, restore `m8_pull` to `(mode == 0u) ||
pull_fallback` **keeping exp_21's `m7_detect` term**, and restore the
enqueue-mode test. Bump `K0P6_MPS_SRC_REV`.

**Ledger line for `LESSONS.md`** (not appended here — outside this task's write
scope, and exp_21 is appending concurrently):

> 2026-08-11 exp_20 **The M7 interference is protocol, not payload.** Deleting
> the pool's entire 896 B copy (mode 7 vs mode 2+pull_fallback) moves M7 by
> +5.7 µs while 831.5 µs of interference remains; it costs 480 µs of combine.
> Pacing the pusher recovers 41% of M7 but the combine pays 7:1 and the total is
> monotone — no optimum, pacing is dead. Poll backoff: 64× rate cut, flat, null.
> LDS-only spin: +23 µs, A7 stays struck. Per class at full rate: read +38,
> local write +105, **peer write +624** — fabric surcharge 519 µs at 5.96×, so
> the copy's MLP-1 shape is load-bearing and speeding it up would cost M7
> (retro-explains exp_04). **Retracts exp_17's "only lever is fewer bytes";
> demotes A11/M11 for M7 (keeps it for the combine); promotes M4.** Prime
> remaining suspect: `flush_pending`'s `thread_release<system>` L2 writeback,
> ~2,400/rank/epoch — untested, mode 9 written but not run.

**Artifacts.** `~/overnight-scratch/screen_E20{pace,attr,split,spin,confirm}.csv`
and the matching `E20*_*.log`; readers `.node/read_e20{,b,c,d}.sh`. Commits
`7802e1e3` (modes 4/5/6), `6534f977` (mode 7), `5f88f58f` (mode 8).

---

## Primitives

**What the library made easy.** Two things carried real weight. `packet.cuh`'s
`store_peer_packets` was reused verbatim by the synthetic generator, which is
why the mode-5 traffic is credibly *the same traffic* as the real pusher's and
not a re-implementation of it — a diagnostic that has to re-derive the transport
it is measuring proves nothing. And `moe_hk_adapter.cuh` already carried
`release_signal_batch_agent` next to `release_signal_batch_system`, so the
release-class diagnostic was a one-line scope swap rather than a fence rewrite.

**What was missing, and it is the same gap for the fourth time.** This
experiment needed to build three *twins* of an existing transport: the same
access pattern with the store removed, with the load removed, and with the
destination forced local. `packet.cuh` offers none of these, so `touch_packets`
and `fill_packets` had to be hand-written in the operator header — and getting
them faithful was the whole difficulty, because a read-only twin that keeps its
loads alive without accidentally changing the loop's memory-level parallelism is
a subtle thing to write by hand (the XOR chain is doing double duty: it defeats
dead-code elimination *and* it reproduces the copy's one-outstanding-load
dependency). The library's transports are **opaque to instrumentation**: there
is no way to ask `store_peer_packets` for its read half, its write half, or a
local-destination variant, so every attribution experiment re-implements it and
every re-implementation is a chance to measure the wrong thing.

This compounds the gap exp_17 already named — that a transport's cost model is
invisible in its signature. exp_20 answers that cost model for this caller
(**latency-bound on the load, and its slowness is protecting the GEMM**), and
the fact that such a load-bearing property is discoverable only by building four
hand-written twins is the finding. A `packet.cuh` that parameterised
*destination domain* (local vs peer) and *direction* (read / write / copy) would
have made all of §2b a configuration instead of 120 lines of new code, and would
have made the MLP-vs-interference trade-off expressible rather than accidental.

**A negative for the mandate.** The `mode`-as-diagnostic-selector pattern —
reinterpreting `g`, `flush_rows` and `pull_fallback` inside a guarded mode — is
now on its fifth use and it works, but it is exactly the kind of protocol
expression the primitives were supposed to own. Nothing in
`include/cdna4/ops/group/distributed/` has a vocabulary for "run this role's
body in a diagnostic variant", so the operator header accumulates it. That is
worth a `roles.cuh` primitive if a sixth experiment needs it.

# Addendum to `OVERLAP_KERNEL_DESIGN_IDEAS.md` — prior-corpus re-grades and four new arms

- **Status:** design synthesis. Read-only research; nothing here is GPU-validated.
- **Date:** 2026-08-12
- **Read first:** `OVERLAP_KERNEL_DESIGN_IDEAS.md` (this folder) — the K0–K8 build queue.
  This addendum does not restate it; it (a) verifies it against the primary
  records, (b) re-grades two arms using the amd-master prior corpus that the
  base doc did not price in, and (c) adds four arms the base doc does not
  contain. Section numbers K0–K8 refer to the base doc; A1–A6 are new here.
- **Evidence base beyond the base doc's:** full reads of
  `k0pf6gm_device_tile_mps.hip` (all of M0–M9) and `n2_phase2_gm_mps.cpp`
  (mode-12 epilogue + throttle implementation), the exp_22–38 primary records,
  the aug12 exp_01 calibration, and the amd-master corpus:
  `auto-gpu-kernel/k0_fused_moe/prefill_opt/K0BW_MECHANISM_SWEEP_20260728.md`
  (transport ladder + SDMA), `exp_50/51/52` (depth/direction/effective-depth),
  `k1_comm_overlap/findings/{BUSBW_MEASURED,EMISSION_TIMED,GRANULARITY_LADDER,
  T1_TRANSIT_HEADROOM,EARLIER_PUBLISH}.md`, and
  `findings/AMD_NATIVE_OVERLAP_RESEARCH_THESIS_20260730.md`.

---

## 1. Verification of the base doc, and three re-grades

The base doc's laws L1–L7 all check out against the primary records (exp_35's
+210.6/−615.0 with t=−80.2; exp_37's C≤8 by −27…−34 µs at 7/7 paired rounds;
exp_34's 8.1–9.3 µs/CTA idle-pool tax; exp_29's (t/S)^8 with median 91.7%;
exp_38's +726.9 µs with issue runs 23.5→1.45 and `vmcnt(0)` 21→117; exp_33's
M7+combine = 3,026.1 ± 10.2 at r=−0.904). Three arms need re-grading.

### 1.1 K1/K2 (SDMA arms): the prior corpus already measured this fabric's SDMA, and it loses at these sizes

`K0BW_MECHANISM_SWEEP_20260728.md` §6 (MI350X, same fabric class):

| size | SDMA GB/s | CU push | CU pull |
|---:|---:|---:|---:|
| 16 MiB | 49.78 | 54.77 | 61.00 |
| 64 MiB | 57.87 | 54.71 | 60.95 |
| 256 MiB | 60.39 | 54.71 | 61.41 |

Verdict verbatim: *"SDMA buys nothing over SM-issued transfer on this fabric…
worse below 64 MiB… For dispatch/combine payloads it is not a candidate."*
TransferBench's 46–49 GB/s per-link SDMA agrees. Two implementation traps on
top: `MORI_ENABLE_SDMA` is a **no-op on the IntraNode path** (read at
`dispatch_combine.cpp:114`, consumed only by AsyncLL), and the zero-CU
`hipMemcpyDeviceToDeviceNoCU` emission path is an **explicitly open question**
on gfx950 (thesis open decision #15; the planned SDMA-under-concurrent-MFMA
probe P7 was never executed).

Consequences, stated precisely rather than as a kill:

- K1's per-pair dispatch packs (~23 MiB over 8 chunks ≈ 2.9 MiB sub-transfers)
  and K2's combine packs (~3.7–7.3 MiB chunks) sit squarely in the band where
  the engine is *slower* than a CU at moving the bytes. The entire case for
  K1/K2-as-SDMA is therefore **CU-freeing + fence/op-class deletion**, which
  only monetizes when co-resident compute exists to use the freed CUs — i.e.
  inside K3's pipelined epoch, or under a shared-expert lane (K8). **K1 solo
  is re-graded from −100…−190 µs to ~0…−100 µs**; its real value is as K3's
  admission-bounded dispatch lane. K2-as-SDMA keeps its scientific value (the
  op-class measurement) but its expected sign at T=4096 balanced worsens.
- The base doc's §12.2 ask (SDMA end-to-end at the two embed points) stands,
  but with a prior now attached: expect the crossover **not** to be reached
  intra-node below ~64 MiB/pair, and plan the same-API forced-P2P/forced-SDMA
  pair via AsyncLL borrow or the `DeviceToDeviceNoCU` probe — the IntraNode
  env-var route does not exist.
- A CU-carried variant of K2 must be built alongside the SDMA variant or the
  op-class question stays confounded with engine choice. That variant is A2
  below, and on current evidence it is the *stronger* arm.

### 1.2 K0 (nc-major order): the magnitude assumes a readiness vehicle that mode 12 does not have

Under mode 12, `row_ready` is published **once, in bulk, at the M7.5
rendezvous** (KRN M7.5 block) — there is no per-row or per-group signal during
M7 at all. The mode-2 protocol that could deliver mid-M7 signals is the
measured −573 µs interference class, and exp_30's drain analysis shows that
even there, ticket-ordered events admit the first CTA into M8 only after
~93.9% of the rank's M7 tasks are done. exp_29's Term A (reclaiming the
fall-through spin, ≈ −75 µs) is the only piece that pays *regardless of task
order*, because it rides rank-level finish spread, not row-level readiness.

So nc-major order **by itself** changes when rows *could* be consumed but not
when anything *learns* they can be. K0-alone is re-graded from −150…−300 µs
to ≈ −75…−150 µs (Term-A-plus-staircase-tail effects), and the missing
mechanism — a certification vehicle coarse enough to respect the fence law —
is exactly A1 below. Sequencing implication: build K0 and A1 as **one arm**
(the order swap is one decode change; the slab certification is where the
value lives). Two shelved landmines to clear first, both on record: the
`row_rem` stale-hole cleanup (exp_30 watch-list) and the stale line anchor
(the task decode now lives at `n2_phase2_gm_mps.cpp:442-443`, not :331-332).

### 1.3 The in-wave overlap ceiling is an ISA fact, already proven — use it as the design axiom

exp_25 closed the fine-grained in-wave variant *on ISA grounds*: `vmcnt` is
**one in-order counter per wave**, so a wave cannot wait on its own A2/W2
loads without also draining its outstanding remote atomics (~3.5 µs ACK ≈ 5×
M6's one-K-step tolerance). exp_38 then showed the same counter is a
*negotiated* resource — register pressure anywhere in the kernel converts
into involuntary `vmcnt(0)` drains that re-throttle the epilogue below its
configured depth. Together these give the deep version of the base doc's
Stage-7 framing: **separate carriers (other waves, other CTAs, another
kernel, or an engine) win exactly when they escape the producer wave's
counter space and register file — not because "communication CTAs" are good.**
Every arm below is positioned against this axiom: A1/A2 move work to *other
waves at a phase boundary*; K1/K2-SDMA move it to *an engine*; the dead
alternative (deeper in-wave deferral) is not re-proposed.

---

## 2. New arms

### A1 — slab-certified combine: nc-major + per-slab coarse words ("the complements arm")

**Mechanism.** Order M7 tasks nc-major (`task = nc·num_tiles + tile`) and cut
the nc range into S slabs. At each slab boundary, run one M7.5-class
mini-rendezvous: per-CTA `s_waitcnt vmcnt(0)` + one CTA-leader L2 writeback,
one grid barrier, then **one epoch word per (rank, slab)** published to all 8
peers — the mode-14 protocol economics, S× finer. M8 becomes two ticketed
passes over column halves: the front-half combine of every token starts as
soon as all 8 ranks have published slab 0, while every rank's M7 computes the
back half.

**The alignment fact that fixes S.** M7's output chunks are 448 columns
(16 × 448 = 7,168); M8's inner loop consumes 1,024-byte chunks (14 × 512
columns). They co-align only at column 3,584 — so **S=2 is free**, and S≥4
requires re-chunking M8 to 896-byte units, which breaks its 16 B/lane vector
shape (896/64 = 14 B/lane). Do not fight this: S=2 is also where the fence
law wants to be.

**Economics (honest).** Prize: the combine is 324.2 µs (exp_33) and
issue/latency-bound at 1.57 TB/s = 20% of HBM (exp_29 census), so hiding the
front half under M7's back half is worth ≤ ~160 µs, on top of exp_29's
Term A. Cost: one extra rendezvous of the class the ancestor kernel measured
at **M7.5+M9 ≤ 53.1 µs total** (`T1_TRANSIT_HEADROOM.md`), so ~25–50 µs per
added boundary; plus the amd-master earliness law (`EARLIER_PUBLISH.md`,
L40) — *publish earliness and release granularity are one parameter*; the
per-task version of this idea cost −12,072 µs in fence storms in the
ancestor, and S=2 is the fence-neutral point of the same family. Net band:
**−50…−180 µs on the 3,026 µs coupled block**, and it settles L5's
substitutes-vs-complements question constructively (order creates the early
prefix, coarse slab words deliver it at mode-14 prices).

**Codegen discipline (the reason this is dangerous).** exp_38's nine-site
ablation found the M7.5 rendezvous region (S8) is one of the four sites that
independently collapse the epilogue's injection window. The slab loop *is* an
M7.5 edit. Every A1 build must pass the issue-run-distribution gate
(`e38_30_isa.py`: 282 atomics in ~12 runs, mean ~23.5) and default-arm
`.text` identity, or its number is void. Also on record: clear the
`row_rem` stale-hole landmine before the reorder (exp_30 watch-list), and
mode 14's `m7_done` cells show how to park the 8×S slab words in the
`row_ready` spare tail without a new descriptor slot.

**Falsifier.** S=2 paired campaign shows Δ(M7+combine) ≥ 0, or the slab
rendezvous shows up as > 60 µs in the phase stamps, or the issue-run gate
fails on every build shape tried.

### A2 — staged CU wide-push combine ("mode 15b"): delete the RMW op class without an engine

**Mechanism.** Keep mode 12's ownership layout and M8 exactly as they are;
change only what crosses the fabric. The M7 epilogue writes its tile rows to
a **local per-owner stage** laid out exactly as the owner's `slots[cur]`
slab, folding the intra-producer collision class (avg ~1.5 local experts per
receive row: 60.2% 1×, 29.9% 2×, 8.3% 3×) with *local* atomics. Carrier
waves (drained compute CTAs post-M7-stripe, or the C≤8 pool) then push each
completed group as **16 B posted stores** at depth 8–16 into
`owner.slots[cur]`, and the flags ride A1's slab words (or M7.5 bulk in the
S=1 form). M8, consume-and-zero, and M9 are bit-for-bit untouched.

**Why the numbers favor it over both mode 12 and K2-SDMA:**

1. **Byte fold.** Mode 12's remote stream is 392 MiB/rank of 4 B RMWs
   (102.76M lane-atomics, exp_25 §0.2) because collisions ride the fabric as
   separate atomics. The folded stage pushes only the route census's
   271.6 MB/rank — **1.44× fewer remote bytes**. (This inverts the ancestor's
   L63 refusal of push-combine, which was priced on the old dataflow where
   contribution-keyed push moved 1.512× *more* bytes; mode 12's slots layout
   plus fold-before-push is what changed.)
2. **Op class.** The amd-master issue ladder prices a 16 B posted store at
   0.127 µs vs a system atomic at 0.571 µs, translation hoisted (the m7tab
   LDS peer-table idiom already does this); one fence retires a 256-store
   batch in 0.44 µs; and de-posting is free at depth ≥8 (exp_51: `write_ack`
   ≈ plain write). The fabric side runs at the measured write14x8 ceiling
   359.4 GB/s with depth 8–16 optimal at this row shape — 271.6 MB in
   ~756 µs of *carrier* time, against the 976 µs epilogue surcharge the GEMM
   wave pays today (exp_25's fit; exp_33's proxy 817–898 µs).
3. **Counter-space escape (§1.3).** The GEMM wave stops owning any remote
   traffic: no vmcnt negotiation in the epilogue, no exp_38 fragility class
   on the hot path, and the depth knob moves from a compile-time `vmcnt`
   literal to an ordinary software credit on the carrier waves — where
   per-destination credits (K6) become implementable without touching the
   256-VGPR epilogue at all.

**Costs, stated against the measured rates:** the stage adds ~313 MB of local
writes + ~313 MB of carrier reads (~100–200 µs of machine bandwidth time at
the measured 1.3–2.6 TB/s local rates — exp_20's "staging is cheap" was
5.7 µs for a *smaller* class, do not borrow it); the carrier capacity is the
K0BW grid law (8 CTAs saturate one link; ~64 for full fan-out; pull-side
collapse past 256 CTAs does not apply to push); and the epilogue's local
stage stores still share the wave's vmcnt (bounded, local-latency — the
cheap half of the counter problem).

**Expected magnitude:** −150…−500 µs at T=4096 balanced (fabric-time saving
~220 µs from the byte fold + surcharge-to-carrier transfer, minus staging);
grows with T (the RMW congestion class scales super-linearly, the staged
class linearly). If the neutral suite's op-class decomposition (base doc
§12.1) shows remote 4 B RMW ≈ remote 16 B store at matched bytes under
concurrent compute, this arm is dead and honestly so — that measurement
decides it either way, which is why §12.1 remains the single most valuable
suite request.

**Falsifier.** Staging + carrier tax ≥ the fabric-time saving in the paired
campaign, or the op-class decomposition returns ≈1×.

### A3 — source-interleaved scatter: per-destination spreading by task order, zero protocol

**Observation from the code.** Scatter fills each expert's sorted rows in
ascending receive-row order, and receive rows are segmented by source
(`row = src·MAXTOK + pos`), so within an expert a destination-owner run is
~85 rows long (≈682 rows/expert ÷ 8 sources) — longer than the 32-row block.
Consequently the epilogue's 16-iteration atomic loop targets **one owner
link per wave-burst**, and the depth-4 `vmcnt` bound concentrates on a
single link at a time: under a hot owner, a wave head-of-line-blocks on the
congested link while six links idle.

**Edit.** Round-robin-merge the source segments in the scatter pass
(secondary iteration order over `src` within each expert append), so
consecutive sorted rows rotate owners. All consumers reach rows through
indirection (`sti`, `pull_src`), so nothing else changes; critically, **no
M7 code is touched** — this is a plan-side edit with zero exposure to the
exp_38 codegen cliff, unlike K6's credit cells which must live inside the
256-VGPR epilogue.

**Honest cap and the real target.** At balanced routing, exp_25's arithmetic
caps every de-alignment scheme at +211 µs (bursts already run 78.3% of the
egress ceiling), and the ancestor's store-ordering probe (sorted vs
round-robin destinations) read 1.00–1.06×. The interesting hypotheses are:
(a) the **depth cliff moves** — exp_24's d8/d16 collapse may be per-link
incast, and spreading outstanding ops across 7 links effectively multiplies
the per-link bound (fabric measurements want depth 8–16 at this row shape);
re-sweep depth {4,8,16} on the interleaved order; (b) the **skew cell** —
this is order-based per-destination fairness, the thing K6's credits buy,
for free. Run A3 before K6; if A3 alone flattens the (future) skew axis, K6
is unnecessary complexity.

**Falsifier.** Depth re-sweep shows the same cliff and balanced Δ ≈ 0 —
then bank it as the measured "order cannot substitute for credits" datum and
proceed to K6.

### A4 — backward plan-reuse: the training regime inverts the §5 impossibility

The base doc §5 correctly proves same-epoch dispatch×GEMM overlap is blocked
by the sort's global-histogram dependency — *for forward*. In training, the
backward pass of the same layer replays the **same routing**: `sti`, `sei`,
`tile_desc`, `pull_*`, and the histogram are all functions of the forward
route and can be cached from the forward epoch. Backward dispatch (dY
scattered by the transpose of forward combine) is therefore **plan-free**:
arriving chunks can be scattered into their (already-known) sorted positions
and per-tile `a2_done`-style arrivals can admit GEMM tasks immediately — the
incremental-overlap structure that forward cannot have. Additionally, dW
(per-expert `A2ᵀ·dY`) is long, purely local, and accumulation-tolerant
across microbatches — the ideal shadow for the *next* microbatch's dispatch
lane (K3) or an engine lane (K8). Concretely for the map: the training row
gets a mechanism of its own rather than inheriting prefill's, and the
regime-card schema (study §10) should record "plan reusable: yes/no" as a
first-class selector.

**Cost note:** double-buffering the plan arrays is ~MBs (they are int/float
arrays over PADMAX and T·topk), trivial next to K3's ×2 generation state.

### A5 — compact-MAXTOK: the small-T fast path with a measured ancestor

Every per-epoch sweep in the kernel scales with `T_ext = world·MAXTOK`, not
with T: M0's `mps_arr` zeroing (T_ext×16), M2's sentinel pass, M3–M5's
extent loops, M7.5's row loop. MAXTOK is already descriptor-carried
(`K0P6_D_MAXTOK`) and mode 12 requires it be a power of two (shift/mask in
the epilogue) — so the host can set `MAXTOK = next_pow2(T)` and shrink the
extent machinery 8–64× at decode/small-T shapes with near-zero kernel edits.
The ancestor corpus contains this arm's only confirmed production win:
**`MAXTOK = T` (killing the TOPK× over-provision) measured −1.49% (~148 µs)
at prefill** — the mechanism is real. Two couplings to exploit: (a) the
T=1024/2048 correctness defect lives in exactly this shared tile/segment
extent math (exp_36: `pf6gm` fails too; `mps` leaves 7/8 ranks unwritten) —
whoever fixes it will be re-deriving the extent relations anyway, so
parameterize MAXTOK in the same pass; (b) the exp_36 sensitivity grid
becomes runnable at small T only after both, which is what unblocks the
entire T axis of the base doc's §11 map.

### A6 — dispatch admission bound: priors for K3's twist 1

Supporting numbers for the base doc's K3 mechanism 1, so it is built with
measured constants rather than re-derived: M1's realized push rate is already
**276.1 GB/s = 77% of the write14x8 ceiling** ("the campaign's most
trustworthy peer-write number"), so the dispatch lane inside K3 needs a
*bound*, not a rewrite; K0BW's grid law says 8 CTAs saturate one link and
~64 CTAs the full fan-out — the dispatch stripe allocation inside K3 should
start there; and the depth constants for 14 KiB-row pushes are 8–16 with
turnover at 32 (exp_51), with effective depth capped by
`min(nominal, fan-in × tokens interleaved)` (exp_52's rule — fan-in ≈5.3 at
this route family).

---

## 3. Regime-map refinements (deltas to base doc §11)

- **Tokens/rank flip point, measured ancestor:** the MORI+AITER region scaling
  shows combine cost growing super-linearly past **64 tokens/rank** (16→1,024
  tok/rank: combine 18.3→378.1 µs) — the decode/prefill strategy flip for the
  combine sits between 64 and 256 tokens/rank on that node. Use this as the
  prior for where §11's "Prefill mid" row begins.
- **Decode rows, hardened:** fine-grained completion lost **five times** in
  the ancestor corpus (exp_36/38/47 + two k1 T1 attempts; "the barrier is the
  cheap way to pay an intrinsic all-producers-done rendezvous"), decode's MoE
  window is 96.1% serial, and decode TBO is economically dead (GEMM must
  double to hide 95 µs). The decode rows' "coarse epoch + instruction
  schedule" entries are on much firmer ground than "[measured trend]" — cite
  these.
- **Message-size knees for the dispatch-carrier column:** pair knee 256 KiB;
  fan-out knee 64 KiB/peer (push) vs 1 MiB (pull); pull collapses past 256
  CTAs while push degrades gently — *"if a kernel must be grid-sized for
  something else, prefer push."* K1's chunked packs (2.9 MiB/sub-transfer)
  are past every knee, so the CU-carried version of K1 (staged wide-push
  dispatch, no SDMA) is also viable and should be the control arm.
- **SDMA column:** annotate with §1.1 — intra-node SDMA enters only via
  AsyncLL-style paths or `DeviceToDeviceNoCU` (untested on gfx950), and only
  the CU-freeing term can pay below ~64 MiB/pair.
- **Topology row:** the ancestor measured worst-pair-class only 3.7% off best
  and declared topology-aware peer routing not worth doing intra-node — keep
  topology as a Stage-8 axis, not a kernel design input, until multi-node.

---

## 4. Amended build order (deltas only; discipline unchanged)

| base-doc # | change | reason |
|---|---|---|
| 2 (K0) | **merge with A1** — one arm: nc-major + S=2 slab certification, RMW transport unchanged; clear `row_rem` holes first; e38 issue-run gate mandatory | §1.2: K0-alone has no readiness vehicle under mode 12 |
| 4 (K4-prime) | add A3 to the same session (both are order-only probes; A3 also re-sweeps depth {4,8,16}) | shared harness, zero codegen risk, prices the K6 question |
| 5 (K1) | demote to "inside K3 only" unless Stage-1 surprises; add CU staged-push dispatch as its control arm | §1.1 re-grade |
| 6 (K2) | split into **A2 (CU staged wide-push, first)** and K2-SDMA (second, engine arm); both feed the op-class figure | §1.1 + A2's byte-fold prior |
| 9 (K6) | gate on A3's skew result | A3 may buy the fairness for free |
| new | A5 compact-MAXTOK rides the T=1024/2048 correctness fix (same code region) | two birds, unblocks the T axis |
| new | A4 backward plan-reuse enters at the training-proxy tier with K8 | regime-card selector "plan reusable" |

Standing cautions from the base doc §14 apply verbatim, plus one addition
from this read: **any edit in the M7.5 region is presumed to collapse the
epilogue injection window until the issue-run distribution proves otherwise**
(exp_38 S8 — the slab arm A1 lives exactly there).

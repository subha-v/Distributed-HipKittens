# V6 campaign — the measured model and the refocused design (2026-08-18)

Goal: beat production FP8 (turbo_gg hybrid+delayed, 1,328 ms/iter, 24,667
tok/s/GPU) by 20-30% → target 1,020-1,100 ms/iter.

## 1. What was built

`k0pf6gm_device_tile_t2v6.hip` — additive sibling of the t2b backward:

- **PROF arm** (`K0P6_T2V6_PROF`): per-wait-site idle-cycle ledger at new
  descriptor slot 73 (u64[32], monotonic, realtime brackets *outside* every
  poll).  Measured cost: zero (1,673/1,689.7 vs control 1,670/1,689).
- **FILLER arm** (`K0P6_T2V6_FILLER`): a device work-queue scheduler — two
  readiness-lattice cursors (dW2 claimable from slab 0, dW13 from slab 1;
  phase-1b has no terminating barrier, so dZq is globally final only at the
  slab-0 rendezvous), quota-bounded service-CTA windows in both slabs, a
  poll-and-fill M8 certificate wait on all 256 CTAs (fail-closed bounded
  poll preserved verbatim), M8.5 demoted to cursor drain.  Packed DWMODE:
  [3:0] mode, [7:4] nmb (accumulate derived in-kernel from the epoch
  counter), [15:8]/[23:16] window quotas, [31:24] E_m expert partition.
- **FASTCVT arm** (`K0P6_T2B_FASTCVT`, gated in `n2_phase1b_gm_t2b.cpp`,
  default 0 = t2b byte-identical): the swiglu' epilogue's z-dequant issues
  192 `__hip_cvt_fp8_to_halfraw` emulated converts per lane per pass; the
  hardware `v_cvt_pk_f32_fp8` swap **removed 16% of kernel .text**
  (260,544 → 219,264 B).
- Host: `moe_swap.py` "fill" mode (slice descs, dribble queue, fences,
  iteration-end arena adds), `K0_MEGA_BUBBLE` ledger dump, env whitelist
  additions in `run_t1_mega.sh`.

## 2. The measurements (12-iter smokes, FWD_SYNC=0, forced-uniform routing)

| arm | ms/iter (inst/avg) |
|---|---|
| production FP8 (full-run bar / same-day smoke) | 1,328 / 1,344-1,350 |
| v3b baseline == t2v6 control (bmm wgrad) | 1,670 / 1,689 |
| pure in-kernel filler (windows + tail) | 1,892 |
| partition + monolithic 128-CTA s2 slice + fence (E_m 2/4/8) | 1,959-1,962 |
| partition + dribbled small slices + fence | 1,988-1,993 |
| partition + no fence (dribble 8 / queue-all / 64-CTA fine) | 1,954 / 1,930 / 1,927 |

**Bubble ledger (per bwd launch)**: service slab-0 idle 1.61 ms ×28 CTAs,
slab-1 post-quota 1.23-1.36 ms ×28, M8 certificate wait 36-177 µs ×256,
M0 barrier ~250 µs ×256 → in-launch supply **~150-180 CTA-ms vs in-situ
wgrad demand ~3,360 CTA-ms** (13.1 ms/launch measured in situ — 1.8× the
7.4 ms gate figure).  M2 rows_done wait is rank-structural drift: 92 µs
(rank 2) to 4,353 µs (rank 0) per CTA per launch — **rank 2 is the
straggler; the other seven ranks stall ~2.5 ms average every backward
launch**.

## 3. The model (back-solved, cross-checked)

- **No-wgrad floor ≈ 1,473 ms** (fill 1,892 − 32×13.1; consistent with
  ctl 1,670 − bmm-visible ≈197).
- Decomposition of our 1,670 vs production 1,328:
  shared non-MoE+reduce ~750 = ~750; fwd kernels ~152 ≈ theirs; dgrad
  ~245 vs ~200 (z-regen + fastcvt pathology); **wgrad visible ~197 (bmm)
  vs ~214 (their serial grouped GEMM) — we already win wgrad**; the
  deficit is **~326 ms of non-kernel time around our megas** (in-mega
  drift stalls ~160, host glue ~100-170) vs their ~30-50.
- Every wgrad rescheduling variant lost because (a) in-launch bubbles are
  ~5% of demand, (b) the s2 idle-CU pool during non-MoE is ~120 ms/iter
  (bmm's "hiding" was mostly its own host-time accounting), and (c) slice
  convoys block mega residency where rocBLAS's sub-ms kernels slip
  through.  wgrad ≈ 150-215 ms visible for *everyone*; it is a wash, not
  the lever.
- The M2 drift is **data-driven** (rank 2 genuinely arrives late), not
  protocol slack — filler cannot absorb it on the critical rank and epoch
  pipelining cannot remove it; only fixing rank 2's lag can.

## 4. The refocused design

Keep: t2v6 + FASTCVT, bmm wgrad (optionally the E_m=2 windows
micro-hybrid later).  The campaign's weight moves to the measured
deficits, in order:

1. **dgrad trim** — FASTCVT A/B and full-run bank (in flight).
2. **The ~326 ms non-kernel gap** — itemize with the existing
   `K0_MEGA_TIME=1` cuda-event phase timers (drains mid-run), then kill
   the top items (candidate set: eager torch dispatch per layer·mb, ctypes
   launch train, descriptor staging, DDP-hook bookkeeping; graph-capture
   of the per-mb MoE region is the heavy hammer if launch overhead
   dominates — the serving mega already survived graph-mode replay).
3. **Rank-2 straggler** — diagnose (per-GPU clocks/power under load are
   being sampled this run; then NUMA/affinity); worth ~50-150 ms for both
   stacks, needed for the absolute target.
4. **Endgame fillers where the ledger says there IS room**: shared-expert
   GEMMs into the mega windows (~2 ms/layer·mb of work vs ~150 CTA-ms of
   window supply — it fits where wgrad's 13 ms could not, and deletes its
   torch launches from the shared 650), then attention-wgrad tiles via the
   same cursor scheduler, then cross-layer weight prefetch (M20 engine).

Projected stack: 1,670 − 45 (fastcvt+dgrad) − 150-250 (glue+drift) − 65
(shared expert) → ~1,310-1,410 at parity-to-beating production, with the
endgame fillers and deeper glue kills carrying toward the 1,020-1,100
target.

## 5. Session close-out (aug18 ~02:15 UTC)

**Banked**: t1v6fastfull2 — t2v6+FASTCVT, bmm, FWD_SYNC=1: 260/260,
1,663.1 inst / 1,684.9 avg, 19,703/19,448 tok/s, loss 9.1648e-3 (the v3b
convergence bit-class).  New best stable, +7.5 ms over v3b full-run; the
sync-free A/B showed −21 ms/iter before the known FWD_SYNC=0 instability
killed it (barrier timeout at iter 6 — all 12-iter smokes are
sync0-flattered; long runs need sync1 until the retire-latch port).

**Host-time itemization (K0HT, the safe pure-perf_counter instrument after
K0_MEGA_TIME's GPU-event path wedged the collectives a third time)**:
b_wgrad_host 270 ms/iter, f_launch_sync 287 ms/iter, b_launch 26,
everything else ~18.  **Graph-capturing the bmm block (K0_MEGA_WG_GRAPH)
was a WASH (1,685.7 vs 1,684)** and b_wgrad_host did not collapse →
the big host numbers are BACKPRESSURE (host parked on deep stream queues),
not deletable glue.  The graph path stays env-gated default-off.

**The governing law the drift math forces**: iteration pace = the
critical (hottest, lowest-clocked) rank's wall clock.  Filler and
scheduling only help e2e when they remove work from THAT rank's wall —
in-window filler qualifies (every rank has the 1.6 + 1.3 ms service-pool
idle); cross-rank wait-absorption and host-side reshuffling do not.

**Next-session order (all sized)**:
1. Retire-latch port (serving pattern) → FWD_SYNC=0 stable: −30-50 ms.
2. Shared-expert GEMMs into the service-pool windows via the t2v6 cursor
   scheduler: ~2 ms/layer·mb of critical-rank work moved into
   critical-rank idle, plus its torch launches deleted: −50-65 ms.
3. Launch/staging trims (pre-uploaded descriptors): −15-25 ms.
4. Thermal experiment (one shot): power/clock rebalance to lift GPU2's
   sclk toward the pack: 0-80 ms, shared with production.
5. The bigger swings after that: per-block arrival-gated phase-1b (hides
   transport for mid-pack ranks), attention-wgrad tiles via the same
   scheduler, fwd-side prof TU.


## exp_34 — mode 14 (coarse readiness), MEASURED: rung falsified, and a +727 µs ratchet regression found by the control that was supposed to be boring

- **THE BIG ONE: a commit whose diff cannot change mode 12 changed mode 12 by
  +726.9 µs.** `291dfa08` (mode 14) left every mode-12 line byte-identical — every
  new branch is gated on `m == 14` — and left the exp_24 throttle intact in ISA
  (`vmcnt(4)/(8)/(16)/(32)`, 96 sites each, identical count in both revisions).
  Yet `C=16,g=353,mode=12` measures **6,497.3 µs at `f113d73f` and 7,224.2 µs at
  `291dfa08` in the same session**, with `production` (7,700) and `pf6gm_mega`
  (6,905) unchanged to <5 µs. All of it is in **M7 (+818 µs)**; `planM6` moves 9 µs.
  The visible trace of the cause is `SGPR 104 → 106` and `LDS +68 B`. **Adding
  code to this function re-times phases that code cannot reach.**
- **The resource tuple is NOT a timing gate, and tonight is the proof.** The tuple
  passed byte-for-byte on every gated field (SGPR/VGPR/AGPR/scratch/MFMA/pk_add,
  zero scratch ops in either MFMA span) while the arm lost **11 %** of its
  runtime. **Re-time the ratchet config on every commit that touches the shared
  kernel** — one 3-minute campaign would have caught this before it shipped.
- **The mechanism to reach for when a config knob stops paying: compare the knob
  ON vs OFF at the new pin, not the arm against its own history.** `g=353` vs
  `g=65` is exactly throttle+depth vs neither. At rev 26 that pair is 6,497.3 /
  7,110.8 (**−613.5 µs**); at the pin it is 7,224.2 / 7,226.0 (**−1.8 µs**) for
  mode 12 and 6,650.9 / 6,643.7 (**+7.2 µs**) for mode 14. That one comparison
  converts "the arm got slower" into "**the injection bound went inert**", which is
  a different bug report with a different owner.
- **A rank-N-only failure is INVISIBLE in rank-0 stdout, and the negative control
  will look like it failed the wrong way.** exp_34's protocol control
  (`R < world-1`) printed `pperr=0` and a `nan` gate on rank 0 — which reads as the
  wrong failure mode — while the eight rank JSONs showed exactly the pre-registered
  signature: `rank 7: pperr=33554432` (bit 25), ranks 0–6 clean, 29,360,128
  poisoned values surviving on rank 7 with `first_rows=[0..15]`. **Adjudicate
  per-rank controls in the per-rank JSONs, never from the driver log.** Time was
  spent suspecting the control before reading the right file.
- **Deleting the per-task VMEM drain is worth NOTHING.** ~2,840 `vmcnt(0)` +
  ~2,840 `__syncthreads()` per CTA removed: **+3.2 µs, n=4 either side, sign
  reversed** (`g=481` 6,647.7 vs `g=353` 6,650.9). The `kCoarseKeepDrainBit`
  selector that made this separable was worth building — without it the whole
  −576 µs would have been mis-attributed to granularity *plus* drain, and the
  waterfall would have carried a phantom rung. **Build the confound selector even
  when you are sure of the answer.**
- **Reserving CTAs costs ~8–9 µs each EVEN WHEN THE POOL PROVABLY HAS NO WORK.**
  Mode 14 at C=0/8/16 = 6,650.9 / 6,725.6 / 6,780.4, monotone, with the service
  pool instrumented as idle three independent ways (bit 26 never set across 4,032
  `pperr` readings; `[MPS TS] DRAIN=0`; `[MPS SPIN] chunk_poll 0/0`). This
  separates **capacity loss** from **contention** for the first time: the aug10
  interference results are not needed to explain the placement penalty, because the
  penalty survives deleting everything the pool did. The prediction was FLAT; it
  was falsified, and the falsification is the stronger result.
- **Two mechanisms that were supposed to compose may be substitutes.** The
  injection bound is worth −613.5 µs at rev 26; deleting the per-row readiness
  protocol is worth −573.3 µs against the same transport with the bound inert.
  Neither has yet been shown to pay *on top of* the other, and at this pin it
  cannot be tested. If they are substitutes — both bounding in-flight remote
  writes — the waterfall's rungs are not additive and the paper's framing must say
  so. **Open question, explicitly not a result.**
- **Process that worked: `setsid` really does save the run.** The ssh session
  dropped three times mid-batch tonight (`client_loop: send disconnect`); every
  campaign survived and completed, and progress was recovered by reading
  `screen_<tag>.csv` + `rocm-smi --showpids` rather than by re-running anything.
  Corollary: the harness's GPU lock is what prevented a double-launch when a
  relaunch was attempted 6 minutes early — it refused with `FATAL … lock held`.
- **Process failure of mine, recorded: I wrote placeholder numbers into a result
  table while the campaigns were still running.** They were replaced with
  generated output before the file was committed, and the fix is now a rule:
  **result tables are rendered from the JSON by a script
  (`tools/e34_70_tables.sh`), never hand-typed.** A hand-typed number in a results
  file is indistinguishable from a measured one three days later.
- **A campaign at 5 rotations × (500 warmup / 100 timed / 600-epoch soak) takes
  ~3 min 5 s on this node**, so 8 interleaved arms is ~25 min, and n ≥ 4 on three
  arms plus a 3-point C sweep plus a cross-revision control fits in ~1 h 45.
  Budget by that number, and interleave: the four batches tonight each visited
  every arm, so the session drift that could have masqueraded as the C-sweep effect
  (74.7 µs) is bounded by the within-arm spread (~6 µs).

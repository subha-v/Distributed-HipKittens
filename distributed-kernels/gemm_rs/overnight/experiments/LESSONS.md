# GEMM-RS overnight lessons ledger (append-only; supersede, never delete)

## Session 2 — 2026-08-11 overnight (optimization session)

- **TRAP: the previous session left three GPU processes running for seven
  hours, and the obvious `ps | grep` could not see them.** At session start
  `rocm-smi` showed GPUs 2/6/7 at 100% utilization drawing 218–233 W (idle is
  ~140 W) with 5.5 GB VRAM each, from three `python3` processes inside our own
  `dhk-eval` container — the remains of the rank-1 attempt that was killed
  mid-flight. Two compounding reasons this was nearly missed:
  1. `ps -eo cmd | grep -E 'eval\.py|mp_smoke'` matched **nothing**: children
     spawned by `multiprocessing` carry only
     `from multiprocessing.spawn import spawn_main` on their command line.
     **Detect stale GPU work from `rocm-smi --showpids`, never from a hopeful
     cmdline pattern.**
  2. They run as **root inside the container** while we are uid 15523 on the
     host, so a host-side `kill -TERM` fails with `EPERM` and, with stderr
     discarded, prints nothing and looks like success. Signals must be sent
     with `docker exec <container> pkill -TERM -f <pat>` — from inside, where
     the PID namespace also differs, so match by pattern and not by host PID.
  `tools/reap_stale.sh` now does both correctly and refuses to touch any KFD
  PID that is not in one of our two containers. `tools/run_baseline.sh` aborts
  if any KFD PID exists. Mitigating detail, established after the fact: all
  three showed `CU OCCUPANCY 0` — they were CPU spin-waiting, not occupying
  CUs — and the clean-node baseline below reproduces the previous session's
  number to 0.25%, so the earlier measurements appear not to have been
  corrupted. The detection and signalling failures are the durable lesson.

- **`tools/nsh.ps1` was returning exit code 127 on every single run.**
  PowerShell appends a CRLF when piping a string to a native command's stdin,
  so every remote script ended with a stray `\r` line and bash reported
  `line N: $'\r': command not found`. The real script had already completed, so
  the failure was invisible in the output but poisoned every exit code — which
  would have made automated pass/fail detection meaningless all night. Fixed by
  base64-encoding the body and decoding on the far side; exit codes are now
  trustworthy.

- **Ratchet baseline re-established on a verified-clean node:** per-shape means
  µs `107.74 / 115.12 / 96.77 / 203.55 / 765.67 / 2865.78`, **geomean
  285.02 µs** (previous session recorded 285.7 µs — reproduces to 0.25%).
  This is the denominator every experiment tonight is measured against until a
  competitor number exists.

- **The evaluator's benchmark loop is READ, and it settles two open questions
  at once.** Source: `eval.py::_run_distributed_benchmark`, on the node at
  `…/exp026-…/cwd/eval.py:311-412`.
  1. **Call counts are symmetric — the epoch-lockstep worry is closed.**
     `should_stop` is computed on rank 0 only and pushed to everyone with
     `dist.broadcast(stop_tensor, 0)`, so every rank calls `custom_kernel` the
     same number of times. Submission state (epochs, signal cells) may
     therefore be carried across `destroy_process_group` safely. This was the
     one assumption the Track A fix could not verify.
  2. **The graded protocol is per-call latency with barriers, not pipelined
     throughput** — and this changes what is worth optimizing:
     ```
     clear_l2_cache(); torch.cuda.synchronize(); dist.barrier()
     t0 = perf_counter_ns()          # rank 0 only
     output = custom_kernel(...)     # host issue is ON the critical path
     torch.cuda.synchronize(); dist.barrier()
     t1 = perf_counter_ns()
     ```
     Our `m7_bench` reports both columns and the gap is large: shape 1 is
     `pipelined = 107.86` but `single_wall = 235.20` against
     `device_max = 106.25`, and the geomean of `single_wall` is ~476 µs versus
     285.02 pipelined.
     **Do not read that 476 µs as a predicted graded score — it is an upper
     bound and a pessimistic one.** Our harness is single-process/8-device, so
     its `single_wall` and its 62 µs `hostIss` include **all eight launches
     issued serially from one process**. The evaluator runs one process per
     rank, each issuing a single launch concurrently, so the per-rank host
     issue is nearer 62/8 ≈ 8 µs. A defensible estimate for shape 1 under the
     graded protocol is `~8 (host) + 106 (device) + barrier`, i.e. ~120-140 µs,
     not 235 — a graded geomean maybe 1.2-1.3× the pipelined one rather than
     1.7×. The honest statement is that **neither column is the graded number**
     and only an evaluator run settles it (Track A).
     What survives regardless: there is **no pipelining across calls** in the
     graded protocol, so per-call fixed cost — host issue, launch latency, the
     kernel's own prologue/epilogue — is worth exactly as much as device time,
     and it falls hardest on the three small shapes that a geometric mean
     weights most. Host issue is already down 90 → 62 µs from the
     `hipFuncSetAttribute` fix. The pipelined geomean stays the ratchet (per
     `CLAUDE.md`), but every result should record `single_wall` alongside it.
  Note also that benchmark mode passes `recheck=False`, so **it never
  re-verifies correctness** — which is how the previously-recorded
  303/463/2099/3043/21509 µs run happily timed garbage.

- **TRAP (measurement, not yet paid): `run_ours_evaluator.sh` defaults
  `HK_DEBUG=1`.** That path puts a `torch.cuda.synchronize()` and unbuffered
  stderr writes *inside every timed call*. Any evaluator number intended for
  `RESULTS.md` must be taken with `HK_DEBUG=0`. Debug output is for localizing
  the hang, never for a number.

- **TRAP: Gate M2 could not fail. It was a silent false pass all session.**
  `tools/m2_report.sh` reads `overnight/build/m1a.log` and
  `overnight/build/isa/*.s`, which only `tools/m1_build.sh` and
  `tools/m2_isa.sh` produce — and `overnight/build/` does not exist in a freshly
  synced tree. The script runs `set -uo pipefail` **without `-e`**, and every
  step is a `grep` or a `python3` heredoc, so with its inputs missing it printed
  six missing-file errors plus a `FileNotFoundError` and **exited 0**. Any
  ladder calling it recorded "M2 PASS" while checking nothing — which would have
  hidden exactly the spill regression E1(b) is most likely to cause.
  `tools/gate_ladder.sh` now runs the two prerequisites first, greps the report
  for `Error|Traceback|No such file`, and asserts the metadata table contains
  all **7** instantiation rows before proceeding. General lesson: a gate that
  has never failed is not evidence of correctness; make it fail on purpose once.

- **E3 is worth ~3%, not 9% — the attribution that motivated it was measured
  under the OLD reducer split.** Three corrections from the protocol review,
  all of which survive independent arithmetic:
  1. **Half the scored shapes cannot benefit at all.** At `NR=32` there are 272
     producer CTAs, and tiles-per-CTA is **1 / 2 / 1 / 1 / 2 / 4** for shapes
     1-6. A release-grouping rule can only amortize where a CTA emits more than
     one tile, so shapes 1, 3 and 4 are already minimal and become free built-in
     control arms. Best-case geomean gain is ~3%.
  2. **The `vmcnt(0)` half of the release is not recoverable.** The next tile's
     `G::load` immediately issues `global_load_dwordx4 → s_waitcnt vmcnt(0)`,
     and gfx9 `vmcnt` retires in order, so deferred peer stores get drained a
     few instructions into the next tile regardless. Only the `buffer_wbl2` and
     the two barriers are recoverable — **E3's value is coupled to E1(b)**.
  3. The 250.3 µs figure was measured with the pre-`exp_02` `NR=8` split — its
     shape-6 total of 2861.7 µs matches `NR=8`'s 2850.2, not `NR=32`'s 2632.1.
     **Re-measure the attribution before sizing E3.**
  This does not cancel the exp_04 finding that E3 gates further tile work; it
  sharpens it into a *synergy*: finer tiling is what raises tiles-per-CTA, and
  higher tiles-per-CTA is exactly what E3 needs in order to amortize. Neither
  is worth much alone on shapes 1/3/4; together they may be.
  Rejected outright: **per-peer batching** (one release already covers all
  destinations, so splitting by peer strictly *increases* the release count) and
  **unbounded per-CTA-per-epoch** (the eleven official shapes take the generic
  row, where 8192×8192×28672 gives **117 tiles per CTA**).

- **LATENT CORRECTNESS HAZARD found in the shipped mainloop by ISA read
  (exp_03/P0). Not a measurement artifact — read the ISA before dismissing
  it.** In `<256,256,32,*>` the k-loop is three blocks: a header issuing 24
  `ds_read_b64` (inline asm, **no waitcnt**), a *guarded* prefetch block
  holding the only `s_waitcnt lgkmcnt(0)`, and a latch of 64 MFMAs. The guard
  `s_cbranch_scc1 .LBB3_86` (L11152) **skips the prefetch block on the last
  k-iteration of every tile**, so on that path the `ds_read`s that fill
  `A_frag`/`B_frag` reach the MFMAs with no intervening `lgkmcnt(0)`, and the
  `s_barrier` at L10966 is bare. Root cause: every LDS op is wrapped in
  `asm volatile`, so `SIInsertWaitcnts` never sees an LDS event and believes
  `lgkmcnt` is already 0 — it emits neither the use-wait nor the
  `__syncthreads()` wait. Identical in `<256,256,32,true>` (guard L16135, bare
  barrier L16373).
  On GCN/CDNA there is no hardware interlock on an LDS load's destination
  register; `s_waitcnt` is mandatory. The gates pass today (17/17 at `2e-3`,
  600-epoch soak, worst `max|diff| = 4.9e-4`), so the reads evidently land
  during the branch and MFMA issue overhead — but **this is timing luck, not a
  guarantee, and any scheduling change can expose it.**
  Two consequences: (1) the E1(b) restructure adds an explicit `lgkmcnt(0)`
  before the MFMA block, so it *repairs* this as a side effect; (2) do not
  "simplify" that wait away later as redundant.

- **The 12 B scratch / 2 spills are NOT in the k-loop (exp_03/P0).** All four
  scratch instructions in each 256/256/32 symbol are once-per-CTA (L10287,
  L10514) or once-per-tile (L10622 in the tile-loop header, L11401 in the
  `error_bit_set` epilogue); **zero** are inside the k-loop, which runs 116×
  per tile on shape 6. The spill therefore costs ~2 instructions per tile and
  is worth approximately nothing. This independently confirms that **E1(a) was
  never worth GPU time** — and it cost only a read of an already-built ISA to
  establish. Cheap static evidence before an expensive dynamic experiment.

- **The mainloop is worse than the pre-registration assumed, in a useful
  direction.** Only **2** `global_load_dwordx4` are in flight before each
  `vmcnt(0)`, so the 32 KB `BK=32` slab is drained in **two serialized halves**
  — two fully exposed global round trips per k-iteration, not one. Also
  established: counted `vmcnt(N>0)` waits already exist in this very TU (126 of
  them, `vmcnt(1..7)`, all in the reducer accumulate path), so counted waits
  are proven available on gfx942 here and are not a portability question.
  One correction to the earlier claim: the 24 `ds_read`s *are* issued before
  the global loads and not drained until after them, so LDS-read latency is
  already hidden. "No global-load/MFMA overlap" is right; "no overlap at all"
  was too strong.

- **WIN, exp_02: uniform NR=32 landed. Geomean 285.02 → 280.74 µs (−1.5%).**
  Shape 6 **2865.78 → 2633.04 µs (−8.1%)**; shapes 1/3/4/5 within noise, shape
  2 −1.5%. Full ladder passed (M3 17/17 at both tolerances, M4 all three
  controls, M5 600-epoch soak). The pre-registered prediction was 2632 µs on
  shape 6 and ~281 µs geomean — the mechanism (the donor's `NR=8` leaves eight
  reducer CTAs starving against 296 producers on the largest output) predicted
  the magnitude to within a microsecond.

- **WIN, exp_04a: the tile table was inherited too. Geomean 280.74 → 269.96 µs
  (−3.8%), from shape 1 alone at −20.3% (107.86 → 85.93 µs).** Shape 1
  (64×7168×18432) was running `32/256/32`, which gives `(64/32)·⌈7168/256⌉ =
  56` tiles against **272 producer CTAs — 21% of the machine, with 216 CTAs
  executing the tile loop zero times** — and 72 k-iterations. Moving it to
  `32/64/64` gives 224 tiles (0.82 waves) and 36 k-iterations, needs **no new
  instantiation** (that template already existed as the generic fallback row),
  and is a two-line change. Full ladder passed.
  Generalizable lesson: **every column of a donor's shape table is a
  hypothesis, not a constant.** Both the `NR` column and the `BM/BN/BK` column
  came from RadeonFlow's submitted values and both were wrong for this kernel;
  together they were worth 5.3%. The existing attribution had missed this
  because it was measured only on the largest shape, where occupancy is a
  non-issue — but the ranking statistic is a *geometric* mean, so the small
  shapes carry equal weight.

- **NEGATIVE (by analysis, not run — and the reasoning is the useful part):
  do NOT retile shape 3, and more generally the per-tile release tax gates the
  whole tile axis.** Shape 3 (2048×2880×2880) is ragged in N (`⌈2880/256⌉ = 12`
  vs `11.25`, so 6% of its GEMM is padding) and uses only 0.71 of the producer
  CTAs, so `BN=64` (45 columns exactly, 720 tiles) looks obviously right. It is
  not: `producer_drain_release` is issued **once per tile**, shape 3 already
  spends **12.5 µs** there at 192 tiles, and 720 tiles is 3.75× that — about
  **+35 µs** against a ragged-N saving of ~1.5 µs, since shape 3's GEMM is only
  23.6 µs of its 96.9. Shape 1 tolerated 4× the tiles only because its release
  cost was 0.4 µs.
  **Consequence: E3 (release granularity) is a prerequisite for E4, not an
  independent 9%.** Amortizing the release unlocks an axis that is otherwise
  closed everywhere except the one shape where it happened not to matter.
  Recomputing the geometry for the other rows shows none has shape 1's defect
  (last-wave fill is 94% on shapes 2, 4, 5 and 6), so **the tile axis is close
  to exhausted** until E3 lands.

## Session 1 — 2026-08-11 (bring-up)

- **2026-08-11 (baseline, pre-overnight).** First hardware bring-up of the
  MI300X/gfx942 GEMM-RS port. Before this the port had never been compiled for
  gfx942 and had no runtime binding; every gate was `PENDING_GFX942_VALIDATION`.
  Gates M1, M3, M4, M5, M8 PASS; M2 partial. Correct on 17/17 shapes (six
  graded + eleven official), worst `max|diff| = 4.883e-4` against a `1e-2`
  tolerance; 600-epoch skewed changing-input soak clean with epoch and signal
  cells exact; all three negative controls fail as designed; graph replay
  advances device-derived epochs. Full record in `../RESULTS.md`.

- **2026-08-11 baseline timing (our harness, single process / 8 devices / peer
  access, pinned clocks, order-rotated 3×50).** Per-shape means µs:
  `108.2 / 115.4 / 97.3 / 203.5 / 765.4 / 2872.3`; **geomean 285.7 µs**; graph
  mode geomean 279.5 µs. Run-to-run spread <1%. This is NOT comparable to
  rank-1's published 413.139 µs (different machine, different protocol) — the
  like-for-like evaluator run is the open item.

- **Attribution (macro-gated ablation, largest shape 2861.7 µs):** GEMM
  mainloop **1317.8 µs (46%)**, XGMI egress **919.7 µs (32%)**, per-tile L2
  writeback release **250.3 µs (9%)**, cross-rank sync 246.9 µs, reduce
  219.5 µs. Optimization order follows this, not intuition.

- **NUM_REDUCER_CTAS is settled: a uniform NR=32.** Swept {8,16,24,32,40,48}
  per shape, all correct. NR=32 is optimal or within noise everywhere and beats
  the inherited RadeonFlow table (32/48/48/48/32/8) by **8.3% on shape 6**
  (2850.2 → 2632.1 µs). Carrying over a donor's submitted constants was not
  free. Not yet landed in the shape table.

- **NEGATIVE: allocation granularity does not matter.** Payload heap
  fine-grained (uncached) vs coarse-grained (cached) is within 1% on every
  shape, and correct in all arms including coarse signals. The hypothesis was
  that uncached payload defeated L2 on the reducer's local reads of all eight
  slots; it is wrong. The protocol's `buffer_wbl2` / `buffer_inv` handshake is
  doing its job. Do not re-run.

- **NEGATIVE (partial): a per-launch host API call was real but not the floor.**
  `hipFuncSetAttribute` was being issued on every launch inside `launch_fixed`;
  it is a property of the instantiation and is now set once. This cut host
  issue cost ~90 → ~62 µs per operation but did not move device time. Committed.

- **TRAP: GPU clock ramp invalidated an entire attribution.** Idle sclk here is
  ~120–132 MHz vs ~1900 MHz loaded. A fixed 10-iteration warmup on a ~100 µs
  kernel is a few ms of load — nowhere near steady state. The same shape read
  95.7 µs in a back-to-back sweep and 153.1 µs in a fresh process. Fixed with
  `rocm-smi --setperfdeterminism 1900` plus duration-based warmup; spread is now
  <1%. Always pin before timing.

- **TRAP: a line-ending normalizer corrupted compiled modules.** `sed -i
  's/\r$//'` walked every file under the harness including `build/*.so`,
  stripping `0x0D` bytes out of the binaries. Every process then segfaulted in
  `dlopen`, which looked exactly like a multiprocessing or HIP-init bug and cost
  about an hour of misdirected debugging. `tools/push.ps1` now filters by
  extension and excludes `build/`.

- **TRAP: the graded tolerance barely detects protocol corruption.** With
  inputs in ±0.01, `CTRL_REROUTE_SLOT` corrupting a whole 1-of-8 reduction
  contribution produced `max|diff| = 7.5e-3` — inside `allclose(1e-2, 1e-2)`.
  A healthy run is `4.9e-4`. Every check now also asserts a tight `2e-3`.

- **TRAP: `CTRL_REROUTE_SLOT` is invisible under unchanged inputs.** The
  rerouted band leaves the true destination slot holding its previous epoch's
  bytes, which are the correct answer if the inputs did not change. The control
  reported a false pass until the run was restructured to use two epochs with
  different seeds compared bitwise against an unrerouted run.

- **TRAP: the evaluator's worker↔rank map is not stable.** `eval.py` drives
  ranks with `multiprocessing.Pool(8)` and reassigns workers between test cases.
  A submission-side cache keyed only by shape handed rank 1 an allocation
  belonging to `cuda:6`; the evaluator caught it as
  `Output device mismatch: cuda:6 != cuda:1`. Any per-process state must be
  keyed by rank **and** by process-group identity. The timings emitted by that
  broken run (303/463/2099/3043/21509 µs) are meaningless — benchmark mode does
  not re-check correctness.

- **The SOL table is not a target.** The published speed-of-light numbers are
  the bf16 MFMA roofline with zero budget for the reduce-scatter (shape 6:
  `2·8192·8192·3696 / 1.307e15 = 380 µs` vs a tabulated 379.43 µs). rank-1's own
  score is ~10.4× that table. Size everything against a competitor measured on
  this node.

- **OPEN: rank-1 needs version archaeology, not an install.** Four disclosed
  compatibility repairs so far (iris staged at its hardcoded python3.10 path;
  no-op `sudo`; a Triton 3.6.0 `wrap_handle_tensor_descriptor` stub verified
  never called; `packed_metadata` 6→3 fields, behaviour-preserving because the
  dropped `clusterDim*` are never read). A fifth — aliasing
  `iris.hip.hipIpcMemHandle_t` to `gpuIpcMemHandle_t`, which **all 8** iris
  checkouts on this node renamed — is written but untested. Details and the
  suspect ranking in `../HANDOFF.md`.

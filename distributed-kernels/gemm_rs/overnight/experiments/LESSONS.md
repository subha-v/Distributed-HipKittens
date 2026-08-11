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

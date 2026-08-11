# GEMM-RS overnight lessons ledger (append-only; supersede, never delete)

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

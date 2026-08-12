
## exp_32 -- NaN poison on `out` (gate hardening, 2026-08-12)

Correctness-only edit to `prefill_opt/host/e004pf_k0pf_ab.py` and
`benchmarks/mok_synthetic_prefill/run_campaign.sh`. No workload, shape,
iteration-count, seed, route or tolerance was changed. `correctness.py`,
`summarize.py` and `synthetic_inputs.py` are untouched.

**Why.** `cand_out` (56 MiB, shared by every candidate arm via `_obuf`) is
cleared once per gate episode but never between epochs, and the MoK synthetic
corpus feeds identical input and identical routing every iteration. A row a
candidate fails to write therefore returns the previous epoch's bit-identical
correct answer -- invisible to `[MOK GATE]`, to `combine_bit_exact`, and to all
600 soak epochs. The 600-epoch soak had no clear at all; the per-arm timed
episode cleared once and then ran 600 epochs.

**What changed.**
- `K0_MOK_POISON_OUT` (default `1`, forwarded in the `-e` list). `0` restores
  the previous `zero_()` behaviour exactly.
- Poison value `0x7FD5` = bf16 qNaN payload `0x55`, chosen so a survivor is
  distinguishable from a kernel-produced NaN (`0x7FC0`) or inf (`0x7F80`).
  Detected by `correctness.py`'s existing SUM-all-reduced `nonfinite == 0` gate,
  which is an equality and cannot be widened.
- Poison inserted at three untimed sites: the eager per-arm epoch, before
  **every** soak epoch, and at the per-arm entry plus one extra **untimed**
  verification epoch inserted after `_mok_rank_max`, i.e. after every HIP event
  has been read.
- The negative control deliberately keeps `zero_()`: poisoning it would make it
  fail because of the poison and stop testing the real bug.
- New log lines `[POISON] <tag> arm=<name> survivors=N [first_rows=[...]]`, and
  `[MPS SOAK]` now also prints `poison=` and `poison_epoch=`.
- `R["mps_soak"]["pass"]` additionally requires `poison_survivors == 0`.
- Bundled correctness fix (P0): the `[MOK GATE]` loop ran after the eager loop
  had finished, so for every candidate arm it re-read the same `cand_out` bytes --
  whichever candidate ran last. Each arm's `mok_gate` is now computed inside the
  eager loop while its own bytes are live. **This changes the printed
  `[MOK GATE]` numbers for candidate arms that are not last in the rotated arm
  order.** `production` is unaffected.

**Two additions made at apply time (not in exp_32's prepared patch).**
- **V2a detector self-test, ON by default** (`K0_MOK_POISON_SELFTEST`, also
  forwarded in the `-e` list so the off-switch is real rather than a lie from
  the host). After each candidate arm's eager epoch and *before* its gate, one
  row of an otherwise-correct `cand_out` is poisoned, `mok_gate` is re-run into
  a diagnostic field, and the row is restored bit-for-bit. Prints
  `[POISON SELFTEST] arm=<name> one_row_poisoned_fails=<bool> nonfinite=N
  relative=<r>`. `False` there means the detector is not connected and every
  `survivors=0` in that run is vacuous. Measured: `True`, `nonfinite=57344`
  (= 7,168 elements x 8 ranks, SUM-all-reduced), `relative=nan`.
- `K0_MOK_POISON_SELFTEST` forwarded in `run_campaign.sh`'s `-e` list alongside
  `K0_MOK_POISON_OUT`.

**Cost.** ~20 us per poison+check; ~12 ms added to a ~4 s soak, ~20 ms per
campaign process in total, plus one extra collective `mok_gate` per candidate
arm for the self-test. Zero device work inside any timed region for any arm.

**Known presentation artifact.** A poison-detected soak failure exits through
the pre-existing `blocked_pre_timing_mps_soak` path, which `summarize.py` does
not list in `BLOCKED_STATUSES`, so it presents as no `summary.json`
(`FAIL:nosummary` in `screen.sh`'s CSV) rather than `FAIL:soak=False`. The
reason is in the log: `[MPS SOAK] ... poison=N poison_epoch=E`.

**Driver-side companion (repo, not this tree):**
`overnight/aug11/tools/screen.sh` had to be updated in the same change -- its
`[MPS SOAK]` regex anchored `pass=` directly to `pperr=` and would have read
every poison-on run as `FAIL:soak=MALFORMED`. It is now tolerant of both line
formats, and a run whose self-test reads `False` is recorded as
`VOID:selftest` rather than `OK`.

## exp_26 -- activate the vendored phase-1 body for `mps_mega` (2026-08-12)

Four edits to `prefill_opt/host/e004pf_k0pf_ab.py`, no behaviour change to any
other arm. `mps_mega` now compiles
`distributed-kernels/fused_moe/n2_phase1_gm_mps.cpp` instead of amd-master's
`solution/hip/n2_phase1_gm.cpp`.

- `PF6_N2_FILES`: added `("n2_phase1_gm_mps.cpp",) if PF6MPS_REQUESTED else ()`.
  The allowlist is what gets copied into `KERNELS_DIR`; without this the JIT
  compile dies `FileNotFoundError`. **The donor `n2_phase1_gm.cpp` stays
  installed** -- `pf6gm_mega` still includes it, and keeping both is what
  preserves the reference arm.
- Source-dir selector: `n2_phase1_gm_mps.cpp` now routes to `PF6MPS_SOURCE_DIR`
  (the DHK tree) alongside `n2_phase2_gm_mps.cpp`; everything else still routes
  to `PF6_N2_SOURCE_DIR`.
- `mps_mega`'s G-stack source contract now validates
  `PF6MPS_SOURCE_DIR/n2_phase1_gm_mps.cpp` instead of the donor it no longer
  includes. The vendored file carries `constexpr int kGM = N2GM_G` verbatim, so
  the assertion passes unchanged.
- `R["pf6mps"]["source_sha256"]` now hashes `n2_phase1_gm_mps.cpp`, i.e. the
  file actually compiled.

The phase-1 hint mask is a **compile-time `#define` in
`k0pf6gm_device_tile_mps.hip`**, not a runtime knob and not a `-D`:
`__builtin_amdgcn_sched_group_barrier` takes immediate operands, and mori's JIT
cache key hashes `.hip`/`.cpp` content and **not** compile flags, so a `-D` flip
would silently reuse the previous hsaco. `HIPCC_COMPILE_FLAGS_APPEND` was left
alone.

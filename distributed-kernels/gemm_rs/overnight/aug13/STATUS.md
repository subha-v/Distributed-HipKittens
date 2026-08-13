# aug13 GEMM-RS — STATUS

Session goal (in order): 1) reproduce the instrument-B (official evaluator)
baseline with arm-order rotation; 2) land one measured improvement — the
per-call host tax under the grading protocol is the top-ranked target
(exp_24 §12); 3) resume the knob queue (reducer-count response curve, order/
certification, depth bounds) per docs/distributed/OVERLAP_ABSTRACTIONS.md §7.

Ground rules carried in: allocation/arm order rotated on every A/B (exp_24/26
lesson); same-session paired denominators; correctness gates before any timed
number; one 8-GPU job at a time behind `tools/gpu_lease.sh`; clocks pinned.

## Log (newest first)

### exp_01 — instrument-B ladder, rotated + the HK_DEBUG confounder [RUNNING]

**Pre-run artifact finding:** exp_24 §12's evaluator "ours" arm (578.7 µs,
`ours/rank1` 1.6159×) ran with **HK_DEBUG=1** — 29,744 `[hk ]` debug lines in
`compbench/ours/benchmark.stderr.txt`. DEBUG disables the exp_12 fast path and
adds a per-call sync + blocking D2H error-bit read + ~5 flushed stderr lines
per rank **inside the evaluator's timed region** (`run_ours_evaluator.sh`
defaults `HK_DEBUG=${HK_DEBUG:-1}`). The "arm-dependent inflation" that
promoted the host tax to the top of the queue is therefore confounded with a
self-inflicted debug tax. exp_01 re-runs the B ladder with `HK_DEBUG=0`, an
`ours_dbg` control arm, and 5 sessions of rotated arm order.

Node state at start: no KFD pids (the aug11 stale corpse pid 3001610 is gone —
node evidently cleaned/rebooted); containers `dhk-gemmrs`/`dhk-eval` up;
production `.so` md5 6bb2a223 built 2026-08-12 13:46 from branch head cc727283
(PERSHAPE=0 verified in source and absent from build flags); clocks were
unpinned (idle 120 MHz) — pinned to 1900 by the runner.

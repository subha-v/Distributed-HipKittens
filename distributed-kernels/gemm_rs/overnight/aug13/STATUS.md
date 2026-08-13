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

### exp_01 — rotation campaign COMPLETE (aug13b): the order-balanced table

s0–s4 ran 2026-08-13 20:16–22:14Z (18/18 bench passes green, M3 re-gated
17/17 first, every O pass fast-path-asserted). **Citable, order-balanced
(n=5 within-session pairs): `ours/rank-1` = 1.1203 [1.0795, 1.1628];
`ours/reference` = 0.7641 [0.7268, 0.8033] — ours beats reference in every
session under both arm orders, and neither ordering flips the O/K sign
(O-first 1.1101 / K-first 1.1359).** Debug tax same-session: D/O =
1.4434/1.4360 (s1/s2), matching the cross-run 1.457. Q3 drift: per-arm GM
spread over sessions O 3.40% / K 8.79% / R 8.57% — single-session ratios
move by up to ±5 points of O/K (1.085–1.170 observed); only within-session
paired ratios are citable. Per-shape O/K: shape 1 a small ours WIN (0.895,
CI crosses 1); shapes 5/6 carry the gap (1.305 [1.273, 1.338] /
1.220 [1.188, 1.253]); shapes 2–4 ~1.09–1.14. Tables and CIs:
`exp_01_eval_rotation/result.md`; analysis `paired_ci.py raw/`. aug13a raw
archived node-side as `raw_prior_s0/`. One environment note: two ssh
blackouts to the node during the campaign (a ~35 min network flap and a
~1 h Conductor pre-auth failure) — the detached run was unaffected.

### exp_03 — COMMIT_MID mainloop arm [STAGED, awaiting GPU window]

exp_27's design row B (the pre-registered first mainloop code arm — commit
between the MFMA halves so half 1 covers the drain) is implemented behind
`HK_GEMM_RS_MI300X_COMMIT_MID` (default 0, production `.text` must be
unchanged — the census asserts it). Instrument: exp_24's `ab_prev_mp.py`
adapted (`ab_cmid_mp.py`) — same-run 3-arm pool (cmid / base / null-twin),
Latin-square complete-permutation ordering, both allocation orders, bit-exact
assertion on every rank. Gate ladder in `run_ab.sh` (M1→M2+placement→
lds_race→M3(17, both tol)→M4→M5→M9-as-null→paired M7 on shapes 5/6).
Pre-registered numbers and the falsifier are exp_27 §6's, copied into
`exp_03_commit_mid/plan.md`.

### Environment finding — eval.py TEST mode cannot pass on this node

`eval.py` test mode hardcodes a 60 s per-case pool timeout; a fresh 8-worker
spawn pool takes ~40 s before any rank touches a GPU (observed live via
KFD-attach timestamps), so case 1 times out for EVERY arm — ours, reference
and rank-1 alike. This is pre-existing: exp_24's archived
`ours/test.popcorn.txt` (08:27) shows the identical two-line truncation, and
its driver did not gate on test mode. exp_01 records test-mode outcomes
non-fatally; the binding correctness gates are our M3 matrix (ran 17/17 green
at both tolerances on the 13:46 binary before any timed number) and benchmark
mode's obligatory checked first call per case.

### exp_01 — instrument-B ladder [S0 COMPLETE; s1–s4 cut short by operator pause]

**The session's headline.** exp_24 §12's evaluator "ours" arm (578.7 µs,
`ours/rank1` 1.6159×) ran with **HK_DEBUG=1** — 29,744 `[hk ]` debug lines in
its archived stderr. DEBUG disables the exp_12 fast path and adds a per-call
sync + **blocking D2H error-bit read** + ~5 flushed stderr lines per rank
**inside the evaluator's timed region** (`run_ours_evaluator.sh` defaulted
`HK_DEBUG=1`; now defaults 0 with the reasoning at the definition site).

**Measured with the fast path live (S0, one full same-session rotation
O→R→K, official eval.py, geomean of per-shape best):**

| arm | 64 | 512 | 2048 | 4096 | 8192a | 8192b | GM |
|---|---:|---:|---:|---:|---:|---:|---:|
| ours | 190.1 | 199.5 | 215.7 | 340.1 | 752.5 | 1876.0 | **397.2** |
| rank-1 | 191.5 | 162.5 | 173.7 | 308.0 | 578.3 | 1480.2 | **335.5** |
| reference | 310.5 | 324.0 | 266.1 | 422.5 | 709.4 | 1721.8 | **489.8** |

- **`ours/reference` = 0.8109 — we BEAT reference GEMM+RCCL under the harness
  that grades us** (exp_24 §12 said 1.1349; withdrawn as
  instrument-contaminated).
- **`ours/rank-1` = 1.1841** (was 1.6159). Per-shape 0.993 / 1.228 / 1.242 /
  1.104 / 1.301 / 1.267. Instruments A and B now agree on ordering.
- The debug tax was −31% of our evaluator geomean; reference reproduced
  within 4%, confirming arm-specificity.
- Full details, gates and caveats: `exp_01_eval_rotation/result.md`.

Node state at start: no KFD pids (the aug11 stale corpse pid 3001610 is gone);
containers `dhk-gemmrs`/`dhk-eval` up; production `.so` md5 6bb2a223 built
2026-08-12 13:46 from branch head cc727283 (PERSHAPE=0 verified in source and
absent from build flags); clocks were unpinned (idle 120 MHz) — pinned to 1900
by the runner and LEFT PINNED at checkpoint (the aug11 convention).

## CHECKPOINT (operator pause) — node + resumption state

- GPU lease: **FREE** (verified). KFD: clean (a few exiting corpses flagged
  ignorable by `kfd_live.sh` during the drain). Clocks: pinned 1900.
- s1's first pass (rank-1 bench) was TERM'd mid-flight at the pause; its
  partial popcorn in `compbench/rank1/` is NOT data. Everything under
  `exp_01_eval_rotation/raw/s0/` is complete and parsed.
- Warning for the next session: `pkill -f run_rotation.sh` over ssh kills
  your own ssh session (the pattern matches the remote command line). Use a
  distinctive pattern or the PID.

## NEXT STEPS, in order

1. ~~Finish exp_01's rotation sessions~~ DONE (aug13b, see log above).
2. **exp_03 (staged, one command): the exp_27 row-B mainloop arm** — commit
   between the MFMA halves, behind `HK_GEMM_RS_MI300X_COMMIT_MID` (default 0).
   `setsid nohup bash overnight/aug13/exp_03_commit_mid/run_ab.sh > run_ab.log
   2>&1 &` runs M1→M2(+rotation-invariant ISA placement gate, validated
   against exp_27's archived census)→lds_race→M3(17,both tol)→M4→M5→M9→paired
   M7 (shapes 5/6 × both allocation orders, Latin-square blocks, bit-exactness
   asserted). Pre-registration and falsifier in its plan.md (exp_27 §6
   verbatim: shape 6 pool −5.0%±3.0%, geomean −0.9%; |Δ|<1.5% on shape 6
   closes the whole B/C/D family). If it lands, extend to shapes 2/3 (the
   second gap pool per exp_01's order-balanced per-shape CIs) before any
   ratchet move; ship only if better under BOTH allocation orders.
3. Then the §7 queue as ranked: row C (LDS offset prefill) or row E
   (warp-group ping-pong) depending on exp_03's verdict; K4 depth-bounding of
   the emit path is the standing alternative if the mainloop family closes.
   The reducer-count C-sweep (§7 item 3) needs a host-side-only override
   (e.g. `HK_GEMM_RS_MI300X_NR_OVERRIDE`, default 0 = table) — the table
   lives in `gemm_rs_mi300x_host_abi.hpp` (`scored_shapes`), so the device
   `.text` ratchet is untouched by construction.

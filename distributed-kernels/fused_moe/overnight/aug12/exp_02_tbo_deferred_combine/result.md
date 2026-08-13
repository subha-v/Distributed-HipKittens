# exp_02 result — mode 16 (TBO-2 deferred combine): rung FALSIFIED at the tested config

- **Verdict:** mode 16 `C=16,g=353,flush_rows=16` = **6,559.6 µs p50 median
  (5 rotations, 6,536.6–6,569.1) = 0.8513× production (7,710.6)**. The
  same-session mode-12 control (flag-off tree, e02camp12) = **6,483.8 µs
  [6,468.9–6,495.5] = 0.8407×**, which reproduces the published ratchet
  (6,482.7 / 0.8408×) to 1.1 µs. **Δ = +75.8 µs — past the pre-registered
  falsifier (Δ > +30 µs). Deferring the combine out of its (t/S)^8 readiness
  ceiling did not pay at this config.** All gates green in every rotation:
  `[MOK GATE]` pass (max_abs 0.035, rel 0.00829), poison survivors 0 with the
  selftest firing (nonfinite=57344) in all 5, soak 600/600 `pperr=0`, spin 0/0.
- **Campaigns:** `e02camp16b` (mode 16, arms production,mps_mega, DHK-tbo16 @
  `a91cff84`, host driver at patch 5) and `e02camp12` (control, 3 arms,
  flag-off `~/Distributed-HipKittens` @ `eda897de`, `.text` 642646fc ==
  rev 26 re-proven by `e03_gate_build.sh` — the M15-DELTA hooks and the TBO
  ifdefs are all default-neutral, byte-identical). pf6gm_mega control
  6,908.8 (0.8958). production stable 7,706–7,719 across the whole session.

## The eager failure and its fix (host patches 4 and 5)

The e02camp16 abort (`survivors=29360128` on all ranks, `pperr=0`, gate nan)
was a **host-side off-by-one, not a kernel defect**: epochs are 1-based
(`k0p6_epoch = mega_count + 1`), so mps launch N runs epoch N and its
deferred M8 writes out-half `(N-1)&1` — but patch-1's helpers assumed 0-based
epochs: `_tbo16_half_write` returned `(launches+1)&1` (the half the launch
does NOT write) and `_tbo16_half_latest` was off by the same one. Every
poison landed on the unwritten half and every gate read the stale half;
29,360,128 == exactly one T×H half, `first_rows=[0..15]` == half 0. Two more
defects in the same class: the second eager `_poison_out()` was called
without the arm argument (wrong branch), and the poison SELFTEST poisoned
absolute row T-1 (half 0) which the parity-aware gate cannot see — it read
`one_row_poisoned_fails=False` in the first green smoke, a VOID-class
detector hole. Fixes: `~/tbo16_host_patch4.py` (helper swap + arm-aware
second poison) and `~/tbo16_host_patch5.py` (parity-aware selftest row).
No kernel edit, no SRC_REV bump, no re-pin. The kernel's parity flow was
verified coherent for 1-based epochs end-to-end (M0 retire floor `epoch-2`,
M1 produce parity `e&1`, M8 consume parity `(e-1)&1`; launch 1's "epoch 0"
deferred combine reads the zero-fill parity-0 pull maps → fanout 0 → benign
zero-write of half 0, no polls, no hang).

## Notes for the record

- The earlier "smoke PASSED" evidence (e02smoke) is retracted as
  contaminated: pf6gm_mega ran first, wrote rows [0,T) (== half 0), and the
  identical-input property made its output satisfy the gate that mode 16's
  own combine never wrote. The mixed-arm rule from the handoff holds in both
  directions: mode-16 campaigns must run `production,mps_mega` only.
- Mode-16 stamps rotation failed rc=23 (`run 1 did not produce eight rank
  JSON files`) with `timestamps=1` under the tbo16 pin — left open, not
  chased (the rung is falsified regardless); the attribution of the +75.8
  is therefore unmeasured. Plausible suspects: config-word validation of
  timestamps under mode 16, or the E23 ring demotion interacting with the
  stamp path in a TBO build.
- Why the mechanism lost (hypothesis, unmeasured): mode 16 keeps the full
  mode-12 per-row protocol AND adds ×2 parity state (cache footprint,
  retire-depth-2 wait) while the combine it rescues was already M7's
  anti-correlated slack (r=−0.904, exp_33) — the ceiling it escaped was
  mostly not on the critical path. exp_03's slab result (−191 µs by moving
  combine INTO the M7 window instead of into the next launch) supports this
  reading.

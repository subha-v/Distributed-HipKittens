# Overnight lessons ledger (append-only; supersede, never delete)

- 2026-08-11 (baseline, pre-overnight): `mps_mega` builds clean on gfx950 with
  `SGPR 104 / VGPR 256 / AGPR 256 / LDS 155,428 B (byte-exact with the parity
  port) / scratch 60 B/lane (+24, cold paths, all ≥511 insns from any MFMA span)
  / static MFMA census identical to parity (96+84=180)`. First-launch mode-2
  faults deterministically with `address (nil)` on all 8 GPUs; debug-stop
  bisect clears M0 through M7(+enqueue hooks). Fault domain: M7.6/M8/M9 tail.
  Suspect ranking in `../MPS_OVERNIGHT_HANDOFF.md`; harness state in amd-master
  under `benchmarks/mok_synthetic_prefill/MPS_OVERNIGHT_HARNESS_NOTE.md`.
  LDS, MFMA placement, and arithmetic are NOT suspects: treat them as proven.
- Prior resource-gate sequence (banked): `s_batch_no[4]` and `ordinal_shared`
  were the +20 B LDS (reused dead M2 `s_ns`); the +44 B scratch was 11
  long-lived uniform values crossing the MFMA bodies (now phase-local re-reads,
  static-tail reservation, M8 pull/slot branch split — 60 B/lane left; the
  remainder is chaotic-allocator spread, not one mechanism).

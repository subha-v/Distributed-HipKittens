# exp_02 — deferred combine (mode 16): plan & result ledger

- **Feeds:** methodology Stage 3 same-batch/cross-epoch pipeline scope;
  OVERLAP_KERNEL_DESIGN_IDEAS.md arm K3 (stages A+B of the TBO-2 queue).
- **Mechanism:** epoch i's combine is consumed inside launch i+1 (plan-shadow
  + M8 windows over epoch-parity buffer generations). Protocol details and
  the happens-before argument: `design.md`. Host prerequisites:
  `host_patch_spec.md`.
- **Kernel:** `k0pf6gm_device_tile_mps.hip` mode 16 behind
  `K0P6_MPS_ENABLE_TBO` (default 0). `K0P6_MPS_SRC_REV` = 31.
- **Config:** `K0_MPS_CFG="C=16,g=353,mode=16,flush_rows=16"` — the ratchet's
  exact knobs on mode 12's exact transport, so the arm differs from the
  ratchet in the combine's TIMING only.

## CPU-gate record (complete before any GPU contact)

| check | result |
|---|---|
| flag-off source strip vs HEAD (`tools` audit) | residual diff = SRC_REV bump + neutral aliases (`retire_need`, `poke_epoch`, `epoch32_c`, `par_c`, `m8_empty`) only; no token changes in any hot path |
| mode-16 predicate coverage | `mode_is_direct_accum`, `mode_is_stream`, enqueue arm, defer arm, drain skip, M8 dynamic, consume-and-zero instantiation, run_service bookkeeping branch, `skip_dead_part_zero`, `is_service_cta` (generic), `compute_id_of` (generic) — all audited |
| validation | physical g==1 required; detect bit rejected for mode 16 (deferred detector would mix generations); mode bound `<= 16` under the flag; `C > 0` required (stream-mode rule) |
| buffer parity audit | all seven buffers: write side parity `(e&1)`, consume side `((e-1)&1)`; every other kernel buffer proven single-generation-in-launch |
| happens-before | slots zero-before-reuse via `retired >= epoch-2`; row_ready monotone epoch words; distinct concurrent halves per launch; ticket exactly-once via M0-zeroed `m8_next` |

## GPU gate ladder (pending node lease — one GPU job at a time)

- [ ] build with `-DK0P6_MPS_ENABLE_TBO=1`; capture `-Rpass-analysis=kernel-resource-usage` tuple + occupancy; `.text` sha of the flag-off build vs TBO-free pin (byte-identical required)
- [ ] host patch landed (allocations doubled assert, gate lag, poison lag)
- [ ] correctness + `[MARK] control_fails=True` + poison selftest + `[POISON] survivors=0` (untimed)
- [ ] 600-epoch soak, `pperr = 0` all ranks
- [ ] negative controls: store-over-a-peer + deferred-epoch skip-a-row
- [ ] screen (ranks vs production only), then 5-rotation campaign; mode-12 control from the flag-off build, `production` as in-session denominator
- [ ] phase stamps on (`timestamps=1`): plan, M6, M7, combine residuals, and (optional) a W1/W2 ticket split counter

## Pre-registered read (design.md §6)

Δ vs same-session mode-12: **−150…−300 µs** (absorbed combine minus window
extension). Falsifiers: Δ > +30 µs; W1 claim share < 50%; any nonzero
retire/poll timeout under steady state.

## Result

**PENDING** — awaiting node lease + host patch application.

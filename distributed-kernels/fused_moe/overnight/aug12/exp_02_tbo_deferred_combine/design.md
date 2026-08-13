# exp_02 — deferred combine: the protocol design record

- **Status:** implemented, CPU-audited, ready for node build + gate ladder.
- **Arm selector:** `K0_MPS_CFG="C=16,g=353,mode=16,flush_rows=16"`, built with
  `-DK0P6_MPS_ENABLE_TBO=1`. Companion host patch REQUIRED first
  (`host_patch_spec.md`).
- **Flag discipline (exp_38 rule, verbatim):** default build (`TBO` unset)
  must produce a `.text` byte-identical to the TBO-free build.
  **Never publish a mode-12 number from a TBO=1 binary.**

## 1. The one-paragraph design

Mode 16 (`kModeDeferCombine`) is **mode 12's transport and readiness protocol,
bit for bit** — the same M7 epilogue remote packed-bf16 accumulate target, the
same depth-4 `vmcnt` throttle, the same (b, nc) event queue, the same M7.6
drain publishing per-row `row_ready`, the same M8 consume-and-zero arithmetic.
The experiment is purely **temporal**: epoch `i`'s combine (M8) does **not**
run at the tail of launch `i`; it runs **inside launch i+1**, claimed on the
same dynamic ticket through two windows:

- **W1, the plan shadow** — after `pull_src_fill`, before the M5 grid barrier.
  At the ratchet config (exp_24 A + exp_27 deleted the only stripe work of
  CTAs `bid ≥ 2` in this span), CTAs 0/1's scan/csr/cursors/fill finish early
  and every CTA then claims epoch-(i-1) batches until the ticket exhausts.
- **W2, the regular M8 window** — drains whatever W1 left.

Deferral is what makes the combine *fully readiness-unblocked*: epoch i-1's
producers finished their M7 and drain inside launch i-1 (on all ranks), so
every `row_ready` poll a deferred batch issues passes on first check. The
measured combine readiness law `P(token ready by t) = (t/S)^8` (exp_29) is
escaped entirely — the epoch being combined ended one launch ago.

## 2. Buffer generations (parity)

Seven buffers are allocated at **double capacity** and indexed by epoch
parity. Producer side always writes parity `(epoch & 1)`; the deferred M8 of
launch i consumes parity `((i-1) & 1)`.

| buffer | slot | per-generation extent | parity-stride (elems) | producer | consumer |
|---|---|---|---|---|---|
| `slots` | 61 | `world*MAXTOK*7168` bf16 | `world*MAXTOK*7168` | M7 epilogue (P2 `m7_slot_off`) | M8 read+zero |
| `row_ready` | 25 | `world*T_loc_max` u32 | `world*T_loc_max` | M7.6 drain (`env.row_ready`) | M8 poll |
| `pull_stage` | 12 | `T*TOPK*2` i32 | `T*TOPK*2` | M1 | M3/M5 fill |
| `pull_cnt` | 13 | `T` i32 | `T` | M1 | M3 csr scan |
| `pull_ptr` | 18 | `T+1` i32 | `T+1` | M3 csr scan | M5 fill, M8 |
| `pull_src` | 19 | `T*TOPK*2` i32 | `T*TOPK*2` | M5 fill | M8 |
| `out` | 32 | `T*7168` bf16 | `T*7168` | M8 | host gate (lag 1) |

Everything else stays single-generation within one launch by construction:
`mps_q`/`nc_arr`/`pushed`/`claim`/`mps_state` (zeroed in M0, drained by M7.6
in-launch), `a_ll`/`a_dst`/`sc_stage`/`recv_*`/`hcnt`/`sei`/`tile_desc`/`nvi`/
`sti`/`swt`/`a2q`/`dq2`/`a2_done`/`part_done` (consumed in-launch),
`dest_counter` (already epoch-parity), `chunk_ready`/`rows_done` (epoch-valued
monotone words), `retired`, `combine_done`, `mega_count`.

## 3. Cross-rank lifetime / happens-before (the argument that must hold)

Let `poke(e)` be the M9 retirement value poked by a rank at the end of its
launch `e`. Mode 16 pokes `e-1` (epoch `e-1 >= 0`, else `0`). M0 of launch `j`
waits `retired[p] >= (j > 1 ? j-2 : 0)` under mode 16 (donor's `j-1` for every
other mode). `retired[p] = v` means "rank p has consumed+zeroed every
parity-(v&1) buffer of epoch v".

* **Slot zero-before-reuse.** Producer P's launch `i` accumulates into
  parity `(i&1)`. Owner O zeroed that half in launch `i-1` (its M8 of epoch
  `i-2`, since `(i-2)\bmod 2 = i \bmod 2`), then poked `retired[O] := i-2` at
  the end of launch `i-1` (after M8(i-2) in program order and after all 256
  CTAs arrived at `combine_done`). P's launch `i` M0 polls `retired[O] >= i-2`
  before any M0 zeroing or M1 store. ⇒ the RMW lands only after the zero.
  Identical shape to the mode-12 argument, shifted one epoch.
* **row_ready monotone-epoch cells.** Polls are `>= epoch32` on never-reset
  epoch words. Deferred consumers read parity `(i-1)&1` while launch `i+1`'s
  drain writes parity `(i+1)&1 = (i-1)&1` two epochs later — a later writer
  stores a *larger* epoch value, which still satisfies `>=`. A stalled rank
  cannot be overtaken silently: reaching the conflicting writer implies the
  writer passed M0 with `retired >=`, which the reader itself pokes only
  after finishing the conflicting launch.
* **Distinct concurrent halves.** Inside launch `i`, M7(i) touches
  parity `(i&1)` and M8(i-1) touches `((i-1)&1)`: always distinct. Producer
  and owner activities in the same wall-clock window never share a cache
  line class.
* **Retirement value monotonicity.** Launch 0 pokes `0`; launch `i>0` pokes
  `i-1`. In-order per rank, never decreasing.
* **Ticket consistency.** `m8_next` (zeroed in M0) feeds both the plan-shadow
  and M8-window claims ⇒ every batch claimed exactly once per launch.

## 4. What the epoch-0 boundary does

Under mode 16: launch 0 skips the M8 claim section entirely
(`m8_empty`, grid-uniform) and pokes `retired := 0`; launch 1 consumes epoch 0
normally, then pokes `0`; from launch 2 the steady state holds. Every
non-TBO mode in a TBO build is bit-faithful to its donor behavior
(`par = 0`, `epoch32_c = epoch32`, `m8_empty = false` fold at compile time).

## 5. Gate ladder (before any timing, per standing rules)

1. `.text` sha256 of the **flag-off** build vs the TBO-free pin: byte-identical
   (plus the full resource tuple). `K0P6_MPS_SRC_REV` = 31.
2. Host patch applied (`host_patch_spec.md`): doubled allocations + lag-1 gate
   + lag-1 poison cadence.
3. `[MOK GATE] pass=True` all ranks (lag-adjusted), `pperr = 0`,
   `[MARK] control_fails=True`, `[POISON SELFTEST] nonfinite=57344`,
   `[POISON] survivors=0`, `[MPS SOAK] 600/600`, `[MPS SPIN] 0/0`.
4. Negative controls: the standard store-over-a-peer control must still fail
   loudly; PLUS the mode-16-specific early-reuse control (skip-a-row in the
   deferred epoch must poison exactly the stale half).
5. First screen, then 5-rotation campaign, same-run `production` as
   denominator; mode-12 control comes from the **flag-off** build only.

## 6. Pre-registered predictions and falsifiers

- **P1 (makespan):** mode 16 beats the same-session mode-12 control by
  **150–300 µs** (the ~324 µs combine re-sited into absorbed shadow work,
  minus window-bound extension). Band includes the null. Falsifier:
  paired campaign Δ > +30 µs (i.e. deferral *costs* beyond noise).
- **P2 (phase stamps):** `plan` stamp grows by ≈ the absorbed combine;
  `REDUCE_DONE - M7_DONE` collapses to ticket-drain residue (< 80 µs).
  `ts_M6/M7` flat within codegen drift.
- **P3 (ticket split):** add a (diagnostic-only, timestamps-gated) count of
  W1 vs W2 claims; expectation ≥ 85% W1 at steady state.
- **Falsifier F1:** if the plan-window absorption is < 50% (leftover M8 window
  dominates), the shadow is latency-bound, not capacity-bound — report as
  such and eval mode 17 (M1-stripe window) instead of promoting.

## 7. What this arm deliberately does NOT do (recorded to prevent scope creep)

- No dispatch overlap (next stage: dispatch stripes under M6; needs the
  A2q/dq2/a_ll generation split and the epoch-offset harness contract).
- No nc-major reorder (K0), no SDMA (K1/K2), no per-destination credits (K6).
- No change to the M7 throttle semantics: mode 16 inherits the ratchet's
  depth-4 defaults exactly (config grammar identical).
- Mode 17 (M1-stripe claims) and mode 16 C-sweep are follow-on arms, not v1.

# exp_21 result — mode 12 (direct remote bf16 accumulate, throttled epilogue)

## Verdict

**New ratchet.** `mps_mega` at `C=16, g=33 (throttle+physical-1), mode=12,
flush_rows=16` beats both prior bests through the full gate ladder, in TWO
independent 5-rotation campaigns reproducing to 0.04%:

| campaign | production | pf6gm_mega (homogeneous) | **mps_mega mode 12** | mps/prod | mps/pf6gm |
|---|---:|---:|---:|---:|---:|
| `dec21a` | 7,715.6 | 6,911.4 | **6,685.5** | **0.8665** | **0.9673** |
| `dec21b` | 7,720.0 | 6,902.4 | **6,683.1** | **0.8657** | **0.9682** |

Old ratchet (mode 2, C=64): 6,866.1 / **0.88918** vs production, 0.99351 vs
pf6gm. **Mode 12 moves the margin by ~2.3 points vs production and ~2.6 points
vs the homogeneous megakernel.**

Gate ladder (every rotation, both campaigns): `[MOK GATE] pass=True`,
`control_fails=True` (negative control), `[MPS SOAK] completed=600/600
pperr=0`. **Detector certification** (`g=49`, dual-write + throttle, one
screen run): dual-instrumented epilogue writing slots AND part; M8 compares
row-by-row — **zero detector mismatches (`pperr` bit 1<<27 never set) across
600 epochs**, closing mori-heap blocker 1b (remote packed-bf16 atomics are
exact on mori's HIP-VMM `HeapType::Uncached` heap, empirically).

## What the kernel is

M7's epilogue accumulates each output tile **directly into the owner's slot**
with the remote packed-bf16 atomics exp_18 proved over xGMI — no local
`part`, no pool copy, no pool payload. The service pool keeps only readiness
bookkeeping; the owner zeroes each slot row right after consuming it
(consume-and-zero: sound under the retirement gate + terminal-pperr
semantics). The epilogue's outstanding remote RMWs are **capped at 8 per
thread** (`s_waitcnt vmcnt(8)` per row, the `g`-bit-0x20 throttle) — the
winning lever. Fence fencescape byte-identical to the mode-2 ratchet.

## The evidence chain (all screens single-process unless stated)

**A. The exp_20 re-aim (absorbed).** exp_20 measured the pool's payload copy
at **+5.7 µs of M7** (the ~815 µs interference is protocol, not bytes) and
unthrottled fabric writes at **+624 µs** while the pool's MLP-1 copy
throttled itself 5.9× down to ~free. Design consequence: keep fencescape at
mode-2 identity, shrink C, and expect the epilogue (not the pool) to be the
fabric writer to tame.

**B. Fabric-rate microbenchmark** (`ubench_fabric_rate.cpp`, one link,
coarse-grain hipMalloc): coalesced 4-B remote atomics run at **52.8 GB/s, the
same as 16-B stores (54.9)** at 64-256 writer CTAs; per-task `vmcnt(0)` drain
is free (identical rate); scattered atomics collapse 13× to 4.1 GB/s
(coalescing is load-bearing); target-GPU-busy does not slow them; local
atomics run 285 G/s. **The fabric is byte-limited, not op-limited, for the
epilogue's exact half-wave pattern.** Opportunistically this also derives the
pool's MLP-1 protective role against the +624 µs unthrottled class measured
by exp_20 §2b.

**C. Screens, 4 rounds (absolute µs noisy ±1-2%, stamps rank-max):**

| arm | M7 | combine | m2→end | µs | ratio vs prod |
|---|---:|---:|---:|---:|---:|
| mode 2 C=64 (ratchet, round anchors) | 2,876-2,896 | 340-397 | 6,243-6,269 | 6,949-6,978 | 0.8818-0.8942 |
| mode 12 C=64 (unthrottled) | 3,189 | 382 | 6,595 | 7,303 | 0.9358 |
| mode 13 C=64 | 3,343→3,192 | 343-361 | — | 7,218-7,372 | ~0.92 |
| mode 13 C=16 | 2,912-3,006 | 234-388 | 6,280-6,332 | 6,975 | 0.8888 |
| mode 12 C=16 (defer, unthrottled) | 2,799-2,829 | 386-501 | 6,223-6,363 | 6,864-7,003 | 0.8742-0.9001 |
| **mode 12 C=16 THROTTLED** | **2,644-2,683** | 400-465 | **6,065-6,107** | **6,701-6,726** | **0.8543-0.8654** |
| mode 12 C=12 THROTTLED | 2,645 | 388 | 6,035 | 6,740 | 0.8615 |
| mode 12 C=20 THROTTLED | 2,681 | 401 | 6,084 | 6,830 | 0.8729 |
| mode 12 C=24 THROTTLED | 2,731 | 377 | 6,137 | 6,920 | 0.8781 |
| mode 9 C=64 (flush→agent) | 2,898 (±22) | 418 | 6,324 | 7,311 | 0.9327 |

**D. Mechanism readings (what each ablation proved):**
1. **The epilogue's remote-RMW stream is RATE-shaped, not latency-shaped.**
   Unthrottled: M7 3,189. Per-task drain moved one task later (defer): M7
   3,189 → 3,223 (null — ACK latency theory dead). Capped at 8 outstanding:
   M7 3,189 → 2,644-2,683 (**~500 µs recovered**, matching exp_20 §2b's
   read-pacing class and complementing E1: pacing helps precisely when the
   paced engine's own phase is not the only thing in flight).
2. **The mode-2 protocol (atomics/flags/queue) costs ~590-860 µs at C=64 and
   is C-invariant in total volume** (mode 13: 1,328 @ C=16 vs 1,323 @ C=64).
3. **Consolidating counters (mode 13, per-row target counters) is
   neutral-to-negative** — exp_07's footprint rule at a second, independent
   point: `pushed` deletion saved ~85-150, consolidation cost ~150-240.
   exp_09's withdrawn objection was re-confirmed by measurement.
4. **The flush's ~2,400 system releases are ~free** (mode 9: M7 moved +22 µs
   inside the ±40 band) — exp_20's named prime suspect killed.
5. **Capacity refund is real and monotone** at C {12,16,20,24}; C=12/16
   plateau for the throttled kernel.

## Primitives (mandate section)

- **Used:** `translate_peer`/`peer_ptr` (re-tabulated into 64 B of LDS in the
  epilogue — the exact heap-relative arithmetic, see below), `accumulate_peer_bf162`
  (**new**, added to `packet.cuh`; the epilogue calls nothing else),
  `publish_tile_release`, `fetch_add_acq_rel/relaxed`, `thread_acquire`,
  `release_signal_batch_system`, `poll_epoch_system`, `acquire_payload_system`,
  `retire_pushed_row`/flush idioms from `counter.cuh`/`completion.cuh`.
- **Missing/awkward (fed back as findings):**
  1. `peer.cuh` lacks a **cheap per-body tabulation** form: a kernel wanting
     repeated peer translates inside an unrolled epilogue needs them
     pre-tabulated (we hand-built `m7tab[8]` in LDS + `slot_off`). A
     `peer_tab<8>::build(desc)` device primitive would make this idiomatic.
  2. `packet.cuh`'s transports carry **no pacing/depth knob** — exp_20 + this
     experiment BOTH found unthrottled write streams damaging; a transport
     with `max_outstanding` would have expressed the winner directly.
  3. The accumulating transport (used once) could not have been written
     without leaving the API (no bf16 vector atomic exists on gfx950 — a
     measured width cap, not a library gap).
  4. `detect_dual`/`g`-bit reinterpretation: the fifth use of "config field as
     diagnostic selector" — exp_20 already flagged this for `roles.cuh`.

## Resource/ISA gate

SGPR 106 / VGPR 256 / AGPR 256 / scratch 144 B / LDS 155,496 B / occupancy 1
wave/SIMD; **zero scratch ops inside either MFMA K-loop**; MFMA census 96+84
= 180 unchanged; `global_atomic_pk_add_bf16` present, zero `atomic_cmpswap`.
LDS +68 B (the peer table) and SGPR +2 entered with this experiment.

## Falsifier status (from plan.md)

- **F1 (fabric rate):** PASSED the mechanism test — mode 12 beats mode 2's
  ratchet at its best C. The epilogue IS rate-limited (ubench + throttle
  response), not op-limited.
- **F2 (mori grain):** cleared — zero lost updates in 600 detector epochs.
- **F3 (fence parity):** fencescape literally identical to mode 2 for 12/13;
  mode 9 independently showed release fences are ~free.

## Artifacts

Node: `~/k0-mok-exp21*/{screen.out,summary.json}`, `~/k0-mok-dec21{a,b}/`,
`~/overnight-scratch/` (earlier rounds' csv/logs). Kernel: modes 9, 12, 13
added (`kModeFlushAgent`, `kModeRemoteAccum`, `kModeDirectRows`); `g` field
carries physical-g | detect(0x10) | throttle(0x20). SRC_REV → 23.

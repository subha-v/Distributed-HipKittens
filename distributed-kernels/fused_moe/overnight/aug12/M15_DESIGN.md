# M15 — `k0pf6gm_m15_mega`: slab-certified pipelined combine (design contract)

- **Status:** BUILT, UNMEASURED, UNCOMPILED-ON-NODE. Nothing here is a claim
  until the full gate ladder passes on 8× gfx950. Treat every number below as
  a pre-registered prediction, not a result.
- **Files:** `distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip` (new
  sibling TU; kernel `k0pf6gm_m15_mega`, same kernarg ABI
  `(const long long* desc, int* pperr, long long spin_limit)` and the same
  descriptor as `k0pf6gm_mps_mega`), plus two **default-neutral** macro hooks
  in `n2_phase2_gm_mps.cpp` (M15-DELTA (A): task range + decode; M15-DELTA
  (B): m7tab fill override). With the hooks undefined the shared body's token
  stream is byte-identical — the mps arm owes one `.text` identity check
  (e38_10_gate.sh) and that is the whole exposure.
- **Evidence base:** `OVERLAP_KERNEL_DESIGN_IDEAS.md` (K0), the addendum's A1
  re-grade, exp_29 (readiness law + combine census), exp_30/34 (coarse-word
  protocol + its falsified-as-substitute result), exp_35 (throttle waterfall),
  exp_38 (codegen gates), amd-master `EARLIER_PUBLISH.md` (fence law) and
  `T1_TRANSIT_HEADROOM.md` (M7.5+M9 ≤ 53.1 µs).

## 1. What it is

The mode-12 depth-4 ratchet with three structural changes and nothing else:

1. **NC-major M7 order, two column slabs.** `task = nc·num_tiles + tile`;
   slab 0 = nc 0–7 (output columns [0, 3584)), slab 1 = nc 8–15. The slabs
   are column-disjoint in the slots buffer, which is the entire soundness
   argument for mid-epoch combine: no slab-1 epilogue RMW can touch a
   front-half byte. S=2 is forced by alignment (448-col nc chunks × 1,024 B
   M8 chunks co-align only at column 3,584) and favored by the fence law.
2. **Slab-certified coarse readiness.** After each slab: per-CTA drain + one
   leader L2 writeback + grid barrier + **one epoch word per (rank, slab)**
   to all 8 peers (parked at `row_ready[cur·T_loc_max + T_ext + slab]`, the
   mode-14 spare-tail rule, one cell wider — entry guard requires
   `world·MAXTOK + 2 ≤ T_loc_max`; today 32,770 ≤ 40,960). The **entire
   per-row protocol is gone**: no event queue, no `nc_arr`/`pushed`/`claim`,
   no per-row `row_ready` publishes, no per-row M8 polls, no drain phase.
   M8 is the mode-14 "Ready" shape split front/back at chunk 7.
3. **The pool gets a job.** During slab 1 the C service CTAs poll the 8
   slab-0 words once (bounded) and reduce front-half combine batches on the
   dynamic M8 cursor, **quota-bounded** (`flush_rows` batches per wave,
   default 16) so they can never hold the R1 barrier hostage. Leftover front
   batches and all back batches run post-R1 on the same cursors.

The M7 epilogue transport is mode 12's remote accumulate **verbatim** —
same `accumulate_peer_bf162` sites, same g-encoded depth-4 `vmcnt` throttle,
same deferred drain-at-next-task-head (the done hook buffers the drain
obligation and publishes nothing: exp_34-C's measured +3.2 µs null shape).
M0–M6 and M9 are the donor's, with two deliberate deltas: M0 no longer
zeroes the dead event/arrival/claim buffers, and M9's pre-retirement release
is SYSTEM scope (the exp_29-review KRN:1848 latent-soundness fix, priced
~+22 µs by exp_21 mode 9).

**Why this can beat 6,482.7 µs:** it deletes the residual per-row protocol
at mode-14 prices *while keeping the depth-4 injection bound* (the
substitutes-vs-complements arm, built as complements); it starts ~
`min(quota·C·4, 1024)` front batches inside M7's shadow; and M8 loses its
≥21.5k cross-GPU row polls. Pre-registered band: **−60…−220 µs on the
M7+combine sum** (3,026.1 µs at the ratchet), i.e. p50 ≈ **6,270–6,430 µs
(0.813–0.834×)** if it composes, with the honest possibility of ≈0 if
throttle and protocol-deletion are substitutes (that answer is itself the
open question exp_34 could not settle from its broken pin).

## 2. The staged arm (mode 15b, `-DK0P6_M15_STAGED=1`, OFF by default)

Same kernel, one compile flag: the epilogue's m7tab points at a **local
stage** (`[T_ext][7168] bf16`, descriptor slot 63, host-allocated and
host-zeroed once) so the identical instruction stream folds the ~1.5×
intra-producer collision class locally; carrier CTAs then push folded
half-rows as wide posted packet stores into the owners' `slots` (pool pushes
front halves during slab 1, everyone pushes the rest post-R1, one extra
grid barrier R2, then both slab words publish at once). ~1.44× fewer remote
bytes than the RMW stream (392 → ~272 MiB), posted-store op class instead of
atomic. Run with **throttle off (`g=65`)** — the bound exists to police
remote RMWs the staged arm no longer issues. Requires the host to extend the
descriptor to 64 slots; the default arm needs no host change. Build and
gate it only after the default arm's ladder is green.

## 3. Harness wiring

- Add an arm compiling `k0pf6gm_device_tile_m15.hip` → `k0pf6gm_m15_mega`
  exactly as the mps arm builds its kernel (same `-I`, same JIT flow; this
  .hip is content-hashed, `K0P6_M15_SRC_REV` is its bump guard).
- Config: the same `K0_MPS_CFG` word. **mode must be 12** (the kernel
  fail-closes on anything else), `g=353` (depth 4) for the default arm,
  `C ∈ {8,16}` to start (C=8 is the measured-better region; C also sizes the
  pool sweep), `flush_rows` = pool sweep quota per wave (default 16; sweep
  {0, 8, 16, 32} — 0 disables the mid-M7 sweep and isolates pure protocol
  deletion).
- No new buffers, no new descriptor slots, no host changes for the default
  arm. `pperr` bit 25 = slab-word rendezvous timeout (mode-14's reuse rule).

## 4. Gate ladder (all mandatory before any timed number)

1. `.text` identity of the **mps arm** rebuilt against the edited
   `n2_phase2_gm_mps.cpp` (hooks undefined ⇒ byte-identical is the claim;
   `e38_10_gate.sh` is the check). Any mismatch stops everything.
2. m15 resource gates: `-Rpass-analysis=kernel-resource-usage`, spill
   counts, whole-kernel scratch-op count, and the **issue-run distribution
   of the 282 epilogue atomics** (`e38_30_isa.py`: expect ~12 runs / mean
   ~23.5 like the ratchet; 100+ runs of 1–2 atomics = the involuntary
   throttle, the arm is void — the slab rendezvous lives in exp_38's S8
   danger zone and this gate is the reason to believe or kill it).
3. `[MOK GATE]` tolerance + NaN-poison self-test (must fire) + poison
   survivors 0 + 600-epoch soak + negative controls: omit one slab-word
   publication (must time out on bit 25), publish-before-drain (must fail
   poison/digest), and the T=4096-only guard rails as usual.
4. Watch item from exp_30: `row_remaining` holes are stale in mode 12; m15
   never reads `row_remaining` on any new path (liveness comes from the
   settled `chunk_ready` fills) — keep it that way in any edit.

## 5. Pre-registered falsifiers

- Δ(M7+combine) ≥ 0 at the best (C, flush_rows) point ⇒ the boundary is
  closed at S=2 under mode-12 transport; bank it as the
  substitutes-not-complements answer and stop tuning.
- Slab rendezvous visible as > 60 µs in phase stamps ⇒ the fence-law
  arithmetic was wrong for this kernel; kill S≥2.
- M7 GEMM stamp Δ > +1.5% vs ratchet at matched C ⇒ nc-major broke the
  XCD/L2 residency assumption; the order is the defect, not the slabs.
- Issue-run gate failure on every build shape ⇒ register-pressure kill,
  report as an exp_38-class negative (that is a publishable datum).

## 6. Known honest risks

- **Codegen**: the slab loop shares one textual rendezvous copy (the exp_34
  lesson) but it still adds control flow near the M7 epilogue's live ranges;
  gate 2 is the arbiter, and a spill-collapse there voids the arm.
- **Included-file drift**: m15 includes `moe_mps_adapter.cuh`, which the
  mode-16 TBO work is actively editing; m15 uses only stable decode/role
  accessors, but rebuild-and-regate after that work lands.
- **Untested code**: written off-node, never compiled (no HIP toolchain
  here). Expect a shakeout pass: signature drift in `hk_moe::*`/`hkp::*`
  helpers is the most likely class of compile error, and every call form was
  copied from the working mps kernel to minimize it.

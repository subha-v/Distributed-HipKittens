# exp_26 — result: the vendored M6 body is activated, and the hint mask is a NULL that costs

**Verdict: closed as a measured negative, cleanly.** The vendored phase-1 body is
live for `mps_mega` and provably inert at mask 0. The 2x2 hint ladder says:

- **MFMA bit (bit 2) — the whole `34/48/14 -> 48/48/0` barrier realignment —
  makes M6 `+32.03 +/- 4.28 us` SLOWER** (t = +7.49, p = 3.7e-05, balanced 2x2,
  40 runs). The pre-registered expectation was **-75 to -200 us**. Wrong sign,
  seven sigma from zero. Mask 4 is not a winner, so **no campaign was run** and
  the ratchet is untouched.
- **DS-read bit (bit 0) is a null**: `+8.88 +/- 4.28 us` (p = 0.068), and its
  cell measured twice straddles zero. Predicted null from the ISA beforehand,
  for a reason that is itself the more interesting finding (§3).
- **Interaction is a null**: `+6.07 +/- 4.28 us` (p = 0.19). The two bits factor,
  exactly as the compile-time analysis said they should.

Shipping value is back to `N2GM_P1_SCHED_GSCALE 0`, which is `.text`
byte-identical to the donor include, so the activation ships as a proven no-op.

Confidence: **measured**, 8 batches x 5 runs = 40 GPU runs, every one through the
full gate ladder with the exp_32 NaN poison on, 2 independent batches per
factorial cell, arms verified by `.text` fingerprint. The control reproduced to
**0.34 us over 42 minutes**, so drift is excluded rather than assumed.

---

## 1. Activation (Step 2) — done, and gated

`k0pf6gm_device_tile_mps.hip:395` now includes `n2_phase1_gm_mps.cpp`;
`K0P6_MPS_SRC_REV` 24 -> 25; four harness edits per `activate.md` §4a-4d
(allowlist, source-dir routing, G-stack source contract, provenance hash), all
mirrored into `MPS_OVERNIGHT_HARNESS_NOTE.md`.

### The A/B mechanism, and why

**Compile-time `#define` in the `.hip`.** Not a `K0_MPS_CFG` bit — impossible,
`__builtin_amdgcn_sched_group_barrier` takes immediate operands. Not
`-DN2GM_P1_SCHED_GSCALE` through `HIPCC_COMPILE_FLAGS_APPEND` — mori's JIT cache
key hashes `.hip`/`.cpp` **content**, not compile flags, so a `-D` flip silently
reuses the previous hsaco. One commit per mask, node checkout synced from origin
before each batch. `-D` is used only on the CPU-side ISA builds, where the TU is
compiled directly.

`K0P6_MPS_SRC_REV` was deliberately **not** bumped per mask: the mask literal
lives in the same hashed `.hip`, so it differentiates the cache key by itself,
and holding `SRC_REV` fixed makes the control-last batch source-identical to
control-first — a true repeat rather than a rebuild.

### The byte-identity gate — PASS, and twice over

exp_26 verified this at `96049dfa…` / 166,656 B. **That hash is stale**: exp_21
and exp_24 have changed the source since, so it was re-derived on the current
source rather than compared to a number from a different snapshot.

| build | `.text` sha256 | size |
|---|---|---:|
| `D` — donor include (`n2_phase1_gm.cpp`) | `ab0c353bef065898881b7f845f44c20085bf5c8661abaa0ca8fc19988046f43a` | 179,904 B |
| `M0` — vendored include, mask 0 | **identical** | 179,904 B |

The two TUs differ on exactly one line (`diff` shows only the include). And the
gate holds on the **live binary**, not only on my CPU-side genco: every JIT build
dir the campaign produced at mask 0 unbundles to the same `ab0c353b…`, and so do
the **pre-activation** builds `33125b154cff` / `43ff1649de51` / `ca327e063f2a`
that were compiled from the donor-include source at `SRC_REV 24`. Activation
changed nothing in the code, proven on the artifacts that actually ran.

Resource tuple, identical for D / M0 / M4 (and this is what makes mask 4 a
genuinely single-variable arm):

`TotalSGPRs 106 | ArchVGPRs 256 | AGPRs 256 | ScratchSize 128 B/lane |
Occupancy 1 wave/SIMD | SGPR spills 186 | VGPR spills 14 | LDS 155,496 B |
MFMA census 180 (96 + 84) | scratch ops inside either MFMA K-loop 0`

---

## 2. The ladder — M6 stamp, 8 batches, 40 runs

`C=16,g=353,mode=12,flush_rows=16,timestamps=1`, arms
`production,pf6gm_mega,mps_mega`, 1 process, `w1t1`, exp_32 poison ON.
**All 40 runs `status=OK`**: `[MOK GATE] pass=True`, `pperr=0`,
`control_fails=True`, `[MPS SOAK] 600/600 poison=0`, and
`[POISON SELFTEST] one_row_poisoned_fails=True` in every one.

### Per batch, in the order they ran

| # | batch | mask | commit | M6 mean | sd | t vs `m0a` | p |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | `e26_m0a` | **0** | `dea932a1` | 2541.68 | 3.46 | — | — |
| 2 | `e26_m4` | 4 | `6651c4d4` | 2575.04 | 25.77 | +2.87 | 0.044 |
| 3 | `e26_m1` | 1 | `33a35a0c` | 2553.42 | 9.34 | +2.64 | 0.046 |
| 4 | `e26_m1b` | 1 | `b0ac9d76` | 2535.22 | 11.25 | −1.23 | 0.28 |
| 5 | `e26_m5` | 5 | `d697d621` | 2583.16 | 12.62 | +7.09 | 1.2e-03 |
| 6 | `e26_m0b` | **0** | `a91769ac` | 2541.34 | 6.26 | **−0.34 us, p 0.92** | |
| 7 | `e26_m4b` | 4 | `9096e70d` | 2559.90 | 11.66 | +3.35 | 0.022 |
| 8 | `e26_m5b` | 5 | `1fd3060d` | 2581.68 | 8.31 | +9.93 | 1.2e-04 |

**Read row 6 first.** The control, re-run 42 minutes and six batches later,
reproduced to **−0.34 us (0.013 %), p = 0.92**. There is no wall-clock or
thermal drift in this instrument, so nothing below can be explained away as
ordering.

**Read rows 3 and 4 second.** The *same* mask measured in two batches gave
`+11.74` and `−6.46` — each "significant" against a single control batch, and
they straddle zero. Single-batch t-statistics on this stamp **overstate**
significance, because a batch shares one build, one container-warm state and one
thermal episode. That is why the analysis below replicates every cell and why
per-batch p-values are shown but not relied on.

### The 2x2 factorial — balanced, 2 batches and 10 runs per cell

| mask | DSR (bit 0) | MFMA (bit 2) | n | M6 mean | sd | vs pooled mask 0 | t | p |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| **0** | 0 | 0 | 10 | **2541.51** | 4.77 | — | — | — |
| 1 | **1** | 0 | 10 | 2544.32 | 13.68 | +2.81 | +0.61 | 0.55 |
| 4 | 0 | **1** | 10 | 2567.47 | 20.47 | +25.96 | +3.90 | 2.9e-03 |
| 5 | **1** | **1** | 10 | 2582.42 | 10.11 | +40.91 | +11.58 | 3.7e-08 |

| effect | estimate | SE | t | p |
|---|---:|---:|---:|---:|
| **MFMA bit main effect** | **+32.03 us** | 4.28 | **+7.49** | **3.7e-05** |
| DS-read bit main effect | +8.88 us | 4.28 | +2.08 | 0.068 |
| interaction | +6.07 us | 4.28 | +1.42 | 0.19 |

Pooled: MFMA on `2574.95 +/- 17.49` (n=20) vs off `2542.91 +/- 10.07` (n=20) —
**+32.03 us, t = 7.10, df = 30.4, p = 6.4e-08**.

Note also that turning the MFMA bit on **inflates M6's variance**: sd 4.77 at
mask 0 against 20.47 at mask 4. A schedule that is slower *and* less repeatable.

### The other stamps, and end-to-end — all flat

| stamp | mask-0 control | largest arm delta | verdict |
|---|---:|---:|---|
| `ts_M7_us` | 2692.20 +/- 71.49 | −31.8 | flat; sd swamps it |
| `ts_combine_us` | 321.08 +/- 67.85 | +43.8 | flat |
| `ts_planM3toM5_us` | 375.64 +/- 6.03 | −3.3 | flat |
| `ts_servicedrain_us` | 2773.98 +/- 81.84 | −29.8 | flat |
| `ts_m2_to_end_us` | 5930.60 +/- 25.95 | +28.9 | consistent with M6 alone moving |
| `ratio_vs_prod` | 0.86 +/- 0.02 | +/-0.02 | **not separable for any arm** |

`ts_combine_us` was the column that was supposed to decide mask 1 and mask 5 —
it decides nothing, because the cost it was watching for does not exist (§3).

**End-to-end could not have found this.** Every arm's `ratio_vs_prod` sits in
0.84–0.87 with a per-batch sd of 0.01–0.03; a 32 us effect is 0.5 % of the epoch
and invisible. The instruction to judge on the M6 stamp was correct, and the
stamp turned out to be ~7x more precise than the plan's sigma = 23 us estimate:
**the mask-0 M6 sd is 4.77 us and its batch-to-batch reproducibility is 0.34 us.**

---

## 3. Why bit 0 is a null now — the mechanism changed under exp_26's feet

exp_26's `build.md` recorded bit 0 as owning a real win with a real price:
it moved the K-loop's single `s_waitcnt vmcnt(0)` from `mfma=0` to `mfma=48`,
and paid with scratch instructions `21 -> 114`, 96 of them landing in front of
96 of the kernel's 282 `flat_atomic_pk_add_bf16`. **Both halves of that are gone
on the current source.** Measured on the five builds of the current tree, same
loop-finder and census script exp_26 used:

| build | barrier partition | `vmcnt(0)` drain @ mfma | scratch ops | after p2 | ScratchSize | VGPR spills |
|---|---|---:|---:|---:|---:|---:|
| `D` donor | 34 / 48 / 14 | **48** | 19 | 3 | 128 | 14 |
| `M0` | 34 / 48 / 14 | **48** | 19 | 3 | 128 | 14 |
| `M4` | **48 / 48 / 0** | 48 | 19 | 3 | 128 | 14 |
| `M1` | 34 / 48 / 14 | **48** | **15** | 3 | 128 | 12 |
| `M5` | **48 / 48 / 0** | 48 | **15** | 3 | 128 | 12 |

1. **The drain is already at `mfma=48` in the baseline.** Bit 0's entire
   advertised benefit is present without it, so there is nothing left for it to
   buy. That is why mask 1 measures as a null, and it was predicted from this
   table *before* the GPU ran.
2. **Bit 0's cost is gone too** — and better than gone: scratch instructions go
   **19 -> 15** and VGPR spills **14 -> 12**, with `after_p2` unchanged at 3.
   There is no 96-access migration into the remote-atomic path, because there
   are only 15 scratch instructions in the whole kernel. exp_21's four-boolean
   `throttle_plan` and exp_24's changes have already relieved the phase-2
   register pressure that caused it. **`activate.md` §7.2 — "recover bit 0's
   drain move without the spill migration" — is closed as already-done, and it
   turned out to be free.**
3. **Bit 2 still does exactly what it claims**: `34/48/14 -> 48/48/0`, every
   resource field identical to the donor, at the cost of two extra waits inside
   the loop (`WAIT` 8 -> 10). So mask 4 remains a clean single-variable arm — it
   is just that the variable is worth **−32 us**, not +75 to +200.

So the honest reading of the axis: the compiler's own `34/48/14` split is
**better** than the "aligned" `48/48/0` one, and the aligned split's two extra
LDS waits per iteration are not a small debit — at 512 K-loop iterations they
are the whole effect. `build.md` §8.5 called them "a small debit — but a real
one, and it is the only debit mask 4 carries". That was right about the
mechanism and wrong about the size.

---

## 4. Measurement-integrity findings (worth more than the null)

Three things that would have silently corrupted this experiment or a later one.

1. **`hsaco_before != hsaco_after` is NOT evidence of a code change.** Runs 1–4
   of batch A loaded `9df63b03fb16`; run 5 loaded `b43b280256c6`, at the same
   commit, with no source change. The two hsaco files differ in **36 of 188,744
   bytes** and have **identical `.text`** — the differing bytes are embedded
   build paths, and the two dirs hold *different sets* of kernels, so mori's key
   varies with which modules a given process compiles. `screen.sh`'s
   `-> CHANGED: mps_mega was rebuilt` therefore has a false-positive mode. Only
   `hsaco_before == hsaco_after` is sound (it still proves a cache hit).
   **Fix used here:** `probes/t20_verify_arm.sh` / `t22_audit_dirs.sh` unbundle
   the loaded hsaco and fingerprint `.text` against the CPU-side mask table. All
   13 build dirs touched tonight were audited; each maps to exactly one mask and
   every batch is internally consistent.
2. **A batch ran the wrong arm, and nothing in the rig noticed.** The batch
   originally tagged `e26_m5` was launched after its `git push` had been
   **rejected** by a concurrent push from another agent — so the node synced to a
   tree that still carried mask 1, and the batch measured mask 1 under the label
   mask 5. Caught by the `.text` fingerprint, not by any gate. It is preserved
   honestly as `screen_e26_m1b.csv` (its per-run logs keep the misleading
   `e26_m5_*` names) and it became the second mask-1 replicate, which is what
   exposed the single-batch-t problem in §2. **Fix:** `tools/t16_ladder.sh` now
   does the sync itself, greps the mask literal out of the node checkout, and
   **refuses to launch on a mismatch**; `screen.sh` then runs with
   `SCREEN_SYNC=0` so nothing can move mid-batch. It fired twice afterwards, both
   times correctly (a rejected push and a TLS failure).
3. **`screen.sh`'s soak regex was about to invert every verdict.** It anchored
   `pass=` directly to `pperr=`, and exp_32 inserts `poison=`/`poison_epoch=`
   between them, so every poison-on run would have read `FAIL:soak=MALFORMED`.
   Fixed to tolerate both formats; a run whose poison self-test reads `False` is
   now `VOID:selftest` rather than `OK`. Detail in
   `exp_32_gate_hardening/result.md` §1.

---

## 5. What ships, and what is closed

- **Shipping: `N2GM_P1_SCHED_GSCALE 0`** — `activate.md` §8's rollback, chosen
  because mask 0 is `.text`-identical to the donor, so `mps_mega` is
  bit-for-bit the arm the ratchet was measured on. The vendored body stays
  included: its hooks, its task macros and the corrected ISA facts are the
  durable output, and it is now the file the harness validates and hashes.
- **The axis is closed as a measured negative.** Not "flat" — actively bad, at
  7 sigma, replicated across two builds (masks 4 and 5) and four batches.
  Per `activate.md`'s own kill rule this is a log-the-null-and-set-the-mask-to-0
  outcome; the stronger form applies here since the direction is wrong.
- **No campaign ran.** The instruction was to campaign only a winner with an M6
  delta of at least 3 sigma. Mask 4 clears 3 sigma in the losing direction, so
  there is nothing to promote. The ratchet stays
  `C=16,g=353,mode=12,flush_rows=16 = 6,568.0 +/- 4.6 us, 0.8522x`, unmeasured
  by me and unchanged by this experiment.
- **`activate.md` §7.1 (phase 2's `0x020` VMEM hint over-requesting by 12) is
  now the more attractive of the two follow-ups**, and this result raises its
  prior in an interesting way: phase 1's *under*-request produced a visibly
  "wrong" `34/48/14` partition that turns out to be the **faster** one, so the
  case for correcting phase 2's *over*-request should be treated as untested
  rather than likely. Recommend running it as a two-sided question.
- **`activate.md` §7.2 is closed** — bit 0's spill migration no longer exists.

### Node discipline

`rocm-smi --showpids` showed only `gpuagent` (pid 44579) before every one of the
8 batches; `pgrep -af 'torchrun|mpirun'` empty each time; `screen.sh` re-checks
before every individual run; `setsid` + `timeout` throughout. **No foreign GPU
process appeared at any point tonight, so nothing was preempted and there is
nothing to log.**

---

## Primitives

**Nothing in `include/cdna4/ops/group/distributed/**` was used, needed, missing
or changed.** That is a real observation, not an omission: this axis operates
one level below the primitive library. The mask changes
`__builtin_amdgcn_sched_group_barrier` group sizes inside phase 1's MFMA K-loop —
instruction-scheduling hints with no runtime representation, no memory ordering,
no epoch, no readiness edge. There is no protocol here for `sync.cuh`,
`completion.cuh`, `counter.cuh` or `lifetime.cuh` to express.

Two things the library surface should learn from it anyway:

1. **The primitives cannot express a schedule, and should not try.** The natural
   temptation after `roles.cuh` gives us `is_service`/`is_compute` is a
   `role_schedule_hint`-style helper that wraps the four `sched_group_barrier`
   calls. This experiment is evidence against it: the right group sizes are a
   function of the *compiled* loop body (a per-half census of DS/VMEM/MFMA/DSW
   instructions), not of the protocol role, and the donor's "wrong" literals beat
   the "correct" parameterization by 32 us. A primitive that made the
   parameterized form ergonomic would have made the slower schedule the default.
   Recording this as a **`primitives:` negative finding**: keep the scheduling
   layer out of the distributed vocabulary, and keep it as a measured
   per-kernel constant with the census that justifies it next to it.
2. **What the library *is* missing is on the verification side, not the
   expression side.** Twice tonight the thing that saved a measurement was
   `.text` fingerprinting of the loaded binary against a table of intended
   builds — once for a false "rebuilt" signal, once for a batch that ran the
   wrong arm. Nothing in DHK or the harness offers that. The useful, small
   addition is not a device primitive but a host-side one: **a build-identity
   assertion** that hashes the loaded hsaco's `.text` and compares it with an
   expected value threaded through from the source revision, so "which kernel am
   I actually measuring" stops being answerable only by hand. `t20_verify_arm.sh`
   and `t22_audit_dirs.sh` are the prototype; they should become part of the
   harness contract rather than an experiment's scratch tooling.

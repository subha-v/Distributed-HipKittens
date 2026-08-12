# exp_38 — RESULT: the ratchet is restored, and the resource tuple is fired

**Verdict: GREEN on the first attempt, no fallback needed.** The default build of
`codex/distributed-hipkittens-scaffold` now produces a `.text` section that is
byte-identical to rev 26 (`f113d73f`), with the mode-14 and exp_23 code still in
the tree behind compile-time flags that default OFF. GPU confirmed on both the
end-to-end ratchet and the revived injection bound.

Confidence: **very high.** `.text` byte-identity is not a statistical claim, and
the two GPU checks are independent of each other — the second one is mechanistic
and would not have moved if the fix were a timing coincidence.

---

## 1. The gate: `.text` sha256, side by side

Built CPU-only in `subha_k1`, four TUs from identical sources differing only in
`-D`, unbundled with `clang-offload-bundler --type=o --unbundle
--targets=hipv4-amdgcn-amd-amdhsa--gfx950` then `llvm-objcopy
--dump-section=.text`. Raw log: `gate_text_parity.log`; script
`../tools/e38_10_gate.sh`.

| arm | build | `.text` sha256 | size |
|---|---|---|---|
| **REF** | `f113d73f` (rev 26) | `642646fcd1de3daeb194cf94dd0d96000bb188298f2cd050da33dee833a541a7` | 179,520 B |
| **DEF** | this tree, **no `-D` at all** | `642646fcd1de3daeb194cf94dd0d96000bb188298f2cd050da33dee833a541a7` | 179,520 B |
| M14 | `-DK0P6_MPS_ENABLE_MODE14=1` | `668f26081494c97dcb6a51f89e0bceab48702128a4e06c11130b1b16fdd8ce55` | 192,448 B |
| RING | `-DK0P6_MPS_E23_RING=1` | `3f8645e57480b414dc4dbb3825932602c893d744ee31af562e5c70d590fb17e2` | 179,648 B |

**REF == DEF, byte for byte.** M14 and RING both differ from REF, which is the
other half of the gate: a flag-on build that came out identical would mean the
flag was not reaching the code and the arm would be a lie.

`K0P6_MPS_SRC_REV` moved 29 → 30. It does not participate in codegen (the value
is never read; only its presence in the JIT cache key matters), which is why the
bump is compatible with `.text` identity.

## 2. Which addition caused the perturbation

**The mode-14 code. Not the exp_23 ring.** Three independent lines of evidence:

1. **The measurement predates the ring.** The +726.9 µs was measured at pin
   `291dfa08` = rev 28. The ring is a rev-29 change and was not in that tree at
   all, so it cannot have contributed to the number that was observed.
2. **Guarding mode 14 alone restores identity.** `DEF` has the ring code present
   in the file with `K0P6_MPS_E23_RING 0`, and `DEF` is byte-identical to REF.
   So the necessary and sufficient guard for the regression is
   `K0P6_MPS_ENABLE_MODE14`.
3. **The ring does not touch the mechanism.** With the ring compiled IN
   (`RING`), the M7 epilogue's injection window is *identical* to rev 26 on every
   metric in §4 — 12 issue runs, mean 23.5 atomics in flight, zero scratch ops.
   Its +128 B of `.text` is entirely outside the epilogue.

The ring is nevertheless **demoted to default-off** (`K0P6_MPS_E23_RING` 1 → 0),
because it cannot meet `.text` identity by construction — it adds a store at
every stamped boundary. It is not accused; it is held to the gate we now trust,
and it was never GPU-timed, so it will be timed as its own arm. `ts_mark` now
forwards to rev 26's `ts_last` when the ring is compiled out, so the fused call
sites stay in the source without costing anything.

### 2.1 Site-level attribution (`../tools/e38_40_ablate.sh`)

Mode 14 touches nine places. Each `#if` was renamed to a per-site macro and
built with exactly one site live, so the perturbation can be charged to specific
edits rather than to "mode 14" as a blob. Reference is REF/DEF: 179,520 B, SGPR
spill 186, VGPR spill 15, 19 scratch ops whole-kernel.

| site | what it is | `.text` | SGPR sp | VGPR sp | scratch ops | epilogue? |
|---|---|---|---|---|---|---|
| S1 | `mode_is_direct_accum` third mode compare (cuh) | 179,648 | 188 | 15 | 19 | — |
| S2 | `skip_dead_part_zero` admits mode 14 (cuh) | 179,712 | 186 | 15 | 19 | — |
| S3 | `config_is_valid` + `kRemoteAccumGLegalBits` (cuh) | 179,520 | 186 | 15 | 19 | — |
| **S4** | **`k0p6_mps_task_done`: `if (m == 14) return;`** | 181,632 | 186 | **17** | **118** | **COLLAPSES** |
| S5 | `k0p6_mps_task_drain` third mode compare | 179,520 | 186 | 15 | 19 | — |
| **S6** | **`task_done_maybe_defer` packed-word split** | 179,200 | 188 | 15 | 19 | **COLLAPSES** |
| **S7** | **kernel entry guard, `row_ready` tail index** | 181,376 | **209** | 15 | 19 | **COLLAPSES** |
| **S8** | **M7.5 rendezvous (`cfg75`/`coarse75`)** | 184,000 | 186 | **17** | **118** | **COLLAPSES** |
| S9 | M8 `m8_coarse` + `Ready=true` instantiation | 185,088 | 190 | 15 | 69 | — |
| ALL | all nine (== `-DK0P6_MPS_ENABLE_MODE14=1`) | 192,448 | 217 | 17 | 168 | worst |

Every site changes `.text` — trivially, since every site changes code. The
discriminating column is the last one, and it says something sharper than
"mode 14 is heavy": **four of the nine sites each independently collapse the M7
epilogue's injection window, and five do not touch it at all.** Notably S9 adds
the single largest lump of new code (a whole extra `k0p6_mps_m8_batch`
instantiation, +5,568 B of `.text`, +50 scratch ops) and is *inert* on the
epilogue, while **S4 — one line, `if (k0p6_m == 14ull) return;`, on a branch that
is unreachable at runtime in that build because `config_is_valid` still rejects
mode 14 — is one of the two worst.** Code size is not the variable. Where the
code sits relative to the M7 epilogue's live ranges is.

## 3. GPU confirmation

Default build, node checkout pinned to the pushed commit and verified
byte-identical to the gated sources before launch (`../tools/e38_20_launch.sh`
refuses otherwise). **Stamps off** on every line — `timestamps=1` costs ~15 µs
and headline numbers must not carry it. 5 processes, 500 warmup / 100 timed,
`K0_MPS_SOAK_ITERS=600` untouched. Arms alternated across rounds so session
drift cannot masquerade as the throttle effect.

Six campaigns, arms alternated `353, 65, 65, 353, 353, 65`. Data:
`ratchet_confirm.json`; raw `raw/e38a.driver.log`, `raw/screen_e38a.csv`.

| # | cfg | `mps` µs | `production` µs | `pf6gm` µs | ratio vs prod |
|---|---|---:|---:|---:|---:|
| 1 | `g=353` (throttle on, depth 4) | 6,484.7 | 7,709.5 | 6,903.8 | 0.8411 |
| 4 | `g=353` | 6,488.7 | 7,703.5 | 6,893.1 | 0.8423 |
| 5 | `g=353` | 6,498.1 | 7,707.5 | 6,874.0 | 0.8431 |
| 2 | `g=65` (throttle off) | 7,095.3 | 7,708.9 | 6,879.1 | 0.9204 |
| 3 | `g=65` | 7,113.9 | 7,701.2 | 6,933.3 | 0.9237 |
| 6 | `g=65` | 7,118.4 | 7,705.1 | 6,895.7 | 0.9239 |

**Check 1 — the ratchet is back.** `g=353` median **6,488.7 µs = 0.8423×**
(mean 6,490.5). Against the published stamps-off ratchet of 6,482.7 / 0.8408×
that is **+6.0 µs = +0.09 %** — reproduced. `production` held at 7,701–7,710
(σ 3.3 µs) and `pf6gm_mega` at 6,874–6,933 across all six, so the session is
sound.

**Check 2 — the injection bound is alive again, and this is the one that proves
the fix.** Contrast `g=353` − `g=65` = **−618.7 µs** on means, −625.2 on
medians, with the two arms' campaign sets **disjoint by 597.2 µs**. Reference
points: **−613.5 µs at rev 26** (agrees to 5.2 µs) and **−1.8 µs at the broken
pin `291dfa08`**. The mechanism is restored, not merely the wall clock — a
timing coincidence could not move this contrast by 344×.

All gates green on every run: `[MOK GATE] pass=True`, `[MARK]
control_fails=True`, `pperr=0`, `[POISON SELFTEST] one_row_poisoned_fails=True
nonfinite=57344`, `[POISON] survivors=0` (eager and post_timing, both arms),
`[MPS SOAK] completed=600/600 pperr=0 poison=0 pass=True`. `[MPS SPIN]` 0/0 on
all six.

Two caveats on this table. It is **n=3 per arm**, which is enough for a 618 µs
contrast and for a 0.09 % restoration check but would not be enough to move a
ratchet. And the jit cache served `hsaco 9e98ff46606a` for campaign 3 and
`4f64f7094381` for the rest; those differ only in `__hip_cuid_` bytes, and the
flip does not track the effect (campaign 3 sits mid-range within its arm).

A fresh `.hsaco` was confirmed by mtime, not by directory mtime: the batch opened
on `2026-08-12 18:04:19|47abe3062174` and config 1 rebuilt to
`2026-08-12 20:17:16|4f64f7094381`, which is `K0P6_MPS_SRC_REV 30` taking effect.

## 4. Mechanism: how adding an unused code path makes a `vmcnt` throttle inert

This is the part worth publishing, and it is a direct strengthening of the
paper's central claim.

The throttle is `asm volatile("s_waitcnt vmcnt(4)")` in the M7 epilogue. It is
supposed to cap outstanding remote `flat_atomic_pk_add_bf16` at 4. **A throttle
only binds if the surrounding code would otherwise exceed its cap.** So the
quantity that decides whether it does anything is not its own instruction count —
it is the number of atomics the code issues between consecutive
`vmcnt`-constraining waits. Call that the *issue run length*.
`../tools/e38_30_isa.py` measures it. Data: `epilogue_window.json`.

| arm | atomics | issue runs | max run | **mean run** | runs > 4 | runs == 1 | `vmcnt` waits | scratch ops in epilogue |
|---|---|---|---|---|---|---|---|---|
| REF (rev 26) | 282 | 12 | 59 | **23.5** | 9 | 3 | 345 | **0** |
| DEF (this tree) | 282 | 12 | 59 | **23.5** | 9 | 3 | 345 | **0** |
| RING | 282 | 12 | 59 | **23.5** | 9 | 3 | 345 | **0** |
| M14 | 282 | 194 | 29 | **1.45** | 5 | **189** | 501 | **96** |

`vmcnt(N)` histogram inside the epilogue:

| arm | `vmcnt(0)` | `vmcnt(4)` | `vmcnt(8)` | `vmcnt(16)` | `vmcnt(32)` |
|---|---|---|---|---|---|
| REF / DEF / RING | 21 | 81 | 81 | 81 | 81 |
| M14 | **117** | 96 | 96 | 96 | 96 |
| S4 / S8 (spill route) | **117** | 81 | 81 | 81 | 81 |
| S6 / S7 (restructure route) | 21 | **96** | 96 | 96 | 96 |

The mechanism, stated plainly:

**Mode 14's presence pushes whole-function register allocation over a cliff, and
the compiler pays for it by spilling inside the M7 epilogue. Every spill reload
needs its value before it can be used, so each one drags an `s_waitcnt vmcnt(0)`
— a FULL drain — into the epilogue: 96 new scratch operations and exactly +96
`vmcnt(0)` waits, 21 → 117. Those full drains chop the atomic issue stream from
12 long runs averaging 23.5 atomics in flight into 194 runs averaging 1.45, with
189 of them a single atomic. The hardware now never reaches 4 outstanding remote
RMWs, so `vmcnt(4)` is asked to cap something that never exceeds 1. The throttle
did not disappear and it was not deleted: it was made redundant by a stronger,
involuntary throttle that the register allocator installed.**

That also explains the sign and the size of the regression, which "the throttle
stopped working" alone does not. exp_24 measured that throttle depth has an
optimum near 4 and that going the wrong way is a cliff. Spilling drove the
effective depth to ~1.45 — well past the cliff — and added 96 full pipeline
drains of latency on top. Hence +726.9 µs, and hence `g=353` and `g=65`
converging to within 1.8 µs: both are dominated by the involuntary depth-1
throttle, so switching the tuned one on or off no longer matters.

Two distinct routes produce the same end state, which is why this is fragile
rather than a single bug:

- **Spill route (S4, S8).** `vmcnt(0)` 21 → 117, +96 scratch ops. Register
  pressure → spills → full drains.
- **Restructure route (S6, S7).** Zero extra scratch ops, `vmcnt(0)` unchanged
  at 21, but the epilogue span widens enough to absorb 15 more throttle
  instantiations (81 → 96) and the mean run still collapses to ~2.8. Here the
  inlining/scheduling boundary moved, not the spill count.

Both routes land mean run length below the depth-4 cap, and that is the only
thing that matters for whether the throttle binds. `S4` is the cleanest
demonstration available: **one `return` statement on a runtime-unreachable branch
in a `__forceinline__` helper is enough to disarm the mechanism.**

### 4.1 What this means for the paper

It is positive evidence for the central claim, from the failure direction.
Tonight's thesis is that overlap on AMD is decided by *scheduling* — who carries
the payload, how many remote ops are in flight, what order producer tasks run,
how coarse the signals are. exp_38 shows the in-flight bound is not a property of
the instruction you wrote; it is a property of the *distance* between that
instruction and the operations it is meant to bound, and the compiler owns that
distance. A throttle expressed as `s_waitcnt vmcnt(N)` is therefore a *negotiated*
mechanism, and register pressure anywhere in a megakernel can silently
renegotiate it while leaving every static census and every field of the resource
tuple intact. Any paper that reports a `vmcnt`-throttle result must report the
issue-run distribution alongside it, or the result is not reproducible.

## 5. What parity gate we should have used from the start

The resource tuple. It failed here, and it failed for a specific, fixable reason:
**it reports scratch SIZE, not scratch OP COUNT.** `ScratchSize [bytes/lane]`
stayed pinned at 128 B across the regression while scratch operations went
19 → 168, because the spilled values fit in the allocation that already existed.
Every field we were checking — SGPR 106, VGPR 256, AGPR 256, scratch 128 B, LDS
155,496, MFMA 180, `pk_add_bf16` 282 — was blind to a 11% end-to-end regression.
"Zero scratch ops inside either MFMA span" also passed, because the spills landed
in the M7 epilogue, which is not an MFMA span.

The gate going forward, in priority order:

1. **`.text` sha256 against the current ratchet, for any change claimed inert.**
   This is the only gate that cannot be fooled. It is cheap: one CPU build and
   one `cmp`. Use it for every knob-at-default, every flag-off build, and every
   "refactor only" edit. A knob whose off-state is not `.text`-identical to the
   ratchet is not a knob, it is a second arm, and it needs its own campaign.
2. **When `.text` must legitimately differ** (a real candidate arm), gate on the
   *mechanism-local* invariants, not the tuple: scratch OP count, `vmcnt(0)`
   count inside the mechanism's region, and the issue-run distribution of
   whatever the mechanism is supposed to bound.
3. The resource tuple stays, demoted, as a cheap smoke check. It is necessary and
   nowhere near sufficient. Add SGPR/VGPR **spill counts** and whole-kernel
   scratch op count to it — those did move here (186 → 217 and 15 → 17, 19 →
   168) and would have caught this at CPU-gate time, hours earlier.

The general lesson, for LESSONS.md: **a parity gate must be sensitive to the
mechanism the arm depends on.** exp_34's gate proved the kernel still fit in the
same registers. It never asked whether the throttle still throttled.

## 6. Files, flags and reproduction

Owned and edited: `distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip`,
`distributed-kernels/fused_moe/moe_mps_adapter.cuh`.

| flag | default | what it does |
|---|---|---|
| `K0P6_MPS_ENABLE_MODE14` | **0** | compiles in all nine mode-14 sites. `=1` reproduces exp_34's arms and carries the +727 µs. Never publish a mode-12 number from a `=1` binary. |
| `K0P6_MPS_E23_RING` | **0** | compiles in the exp_23 per-CTA phase ring. `=1` reproduces exp_23's figure arms. |
| `K0P6_MPS_E23_FORCE_ON` | 0 | gate-only; folds the ring's runtime `enable` to compile-time true. Never in a shipped build. |

No mode-14 or exp_23 work was lost and no patch file was needed: because the fix
is a guard rather than a revert, both mechanisms remain in the tree, compile
clean, and are one `-D` away. `exp_34_mode14/*.patch` was therefore not created —
it is the fallback deliverable and the fallback was not taken.

Data files (schemas): `epilogue_window.json` — per arm, keys `atomics`,
`epilogue_span_instrs`, `issue_runs`, `max_run`, `mean_run`, `runs_gt4`,
`runs_eq1`, `vmcnt_waits_in_epilogue`, `vmcnt_hist` (map N → count),
`scratch_ops_in_epilogue`. `text_parity.json` — per arm, keys `build`,
`text_sha256`, `text_bytes`, `sgpr_spill`, `vgpr_spill`, `scratch_ops`,
`identical_to_rev26`. `ratchet_confirm.json` — per campaign, keys `idx`, `cfg`,
`g`, `prod_us`, `pf6gm_us`, `mps_us`, `ratio_vs_prod`, plus the paired
`contrast_us` summary.

# exp_24 result — dead `part` zero-fill (A) and the throttle-depth sweep (B)

**Verdict, one line each.**

- **Mechanism A is a WIN, not a null.** Deleting the 448 MiB of dead `part` zero
  stores removes **51.2 ± 4.2 µs** from the M3→M5 plan phase (−11.9 % of that
  phase, t = 12.3 over 11 paired screens). It is worth ~0.8 % end to end, which
  is *below* what a single screen can resolve — the device stamps are what make
  it a measurement instead of a guess.
- **Mechanism B: depth 8 was a good pick, and the axis is now closed on the deep
  side.** Depth 16 and 32 put **M7 back at exp_21's unthrottled cost**
  (+462 µs and +370 µs vs depth 8, 7.3σ and 5.9σ). Depth 4 is
  indistinguishable from 8 (−83 ± 63 µs, t = 1.3). The useful cap is ≤ 8 and
  the whole benefit is gone by 16.
- **A pre-existing measurement hazard got worse, and it changes how screens
  should be read**: a repeated control drifted **6.6 %** inside one batch. The
  documented σ = 0.52 % band understates the tail.

Node: `gbt350-odcdh2-c05-1`, 8× MI350X gfx950. Kernel commit `520fe9c6`,
`K0P6_MPS_SRC_REV 24`. Node checkout `cd7918a5`, verified **source-identical** to
`520fe9c6` (`git diff 520fe9c6..HEAD` over `*.hip/*.cuh/*.cpp/*.hpp` and
`include/` is empty; only `STATUS.md` moved).

---

## 1. Gate ladder — every point, both batches

**17 of 17 screens passed the full ladder**: `[MOK GATE] mps_mega pass=True`
(`max_abs` 0.0352-0.0391, `relative` 0.0083 — the gates are 0.1/0.1),
`[MARK] control_fails=True`, `pperr = 0`, `[MPS SOAK] completed=600/600`,
`eager_status=valid_diagnostic`, `spin_fail_max=0`, `rc=0`. No tolerance was
widened and the soak was never shortened.

**Rebuild evidence.** Batch `e24a` opened with
`hsaco_before = 2026-08-12 02:28:08|e92e2fb2cce5` (the rev-23 build) and its
first run produced `hsaco_after = 2026-08-12 04:39:09|33125b154cff` — a **new
hsaco with a new true mtime**, resolved through `readlink -f` + `stat -L`. The
exp_24 kernel is what was measured.

**Two live builds of the same source (harness gotcha 4, re-observed and now
quantified).** Three content hashes appeared across the night —
`33125b154cff`, `43ff1649de51`, `ca327e063f2a` — and the `latest/` symlink
flipped between them run to run. Grouped by hash, `mps_us` shows no separation
(`33125b`: 6759-7013 over 8 runs; `43ff16`: 6645-7304 over 5, and the two
extremes of that range are depth-32 and A+depth-4, i.e. config effects). The
flip is benign here, and it is logged per run in `mps_hsaco`.

## 2. Batch e24b — the paired table (9 points, all `timestamps=1`)

All at `C=16, mode=12, flush_rows=16`. Controls at positions 1, 5, 9.
Device stamps in µs (1 tick = 0.01 µs).

| # | `g` | mechanism | `mps_us` | `prod_us` | `pf6gm_us` | `v_prod` | **plan M3→M5** | M6 | **M7** | combine | m2→end | hsaco |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 33 | control (depth 8) | 6841.3 | 7802.0 | 6880.1 | 0.8769 | **430.4** | 2553.0 | 2655.0 | 418.7 | 6057.1 | `33125b` |
| 2 | 289 | depth 4 | 6759.2 | 7969.7 | 6975.8 | 0.8481 | 425.0 | 2562.9 | 2718.1 | 305.3 | 6011.3 | `33125b` |
| 3 | 545 | depth 16 | 7294.8 | 7827.8 | 6919.4 | 0.9319 | 435.4 | 2612.3 | **3204.4** | 412.4 | 6664.6 | `43ff16` |
| 4 | 801 | depth 32 | 7303.7 | 7924.0 | 7047.8 | 0.9217 | 423.3 | 2581.7 | **3112.2** | 356.1 | 6473.3 | `43ff16` |
| 5 | 33 | control (depth 8) | 7012.8 | 7851.4 | 6968.6 | 0.8932 | **434.8** | 2561.6 | 2708.0 | 467.5 | 6171.9 | `33125b` |
| 6 | 97 | **A** (depth 8) | 6769.8 | 7800.3 | 6900.5 | 0.8679 | **374.9** | 2572.5 | 2682.8 | 345.6 | 5975.8 | `ca327e` |
| 7 | 353 | **A + depth 4** | **6645.3** | 7803.6 | 6902.8 | **0.8516** | **384.7** | 2539.1 | 2600.7 | 434.4 | 5958.9 | `43ff16` |
| 8 | 97 | **A** (depth 8) | 6719.6 | 7915.3 | 6936.1 | 0.8489 | **383.0** | 2552.1 | 2825.9 | 271.6 | 6032.6 | `33125b` |
| 9 | 33 | control (depth 8) | 6852.9 | 7910.6 | 7007.1 | 0.8663 | **427.4** | 2544.4 | 2774.9 | 329.1 | 6075.9 | `33125b` |

## 3. Batch e24a — the first pass (8 points)

| # | `g` | mechanism | `mps_us` | `prod_us` | `pf6gm_us` | `v_prod` | plan M3→M5 | M6 | M7 | combine |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33 | control | 6795.1 | 7834.6 | 7016.8 | 0.8673 | — | — | — | — |
| 2 | 97 | **A** | 6745.7 | 7791.8 | 6920.1 | 0.8657 | — | — | — | — |
| 3 | 289 | depth 4 | 6773.2 | 7757.6 | 6893.2 | 0.8731 | — | — | — | — |
| 4 | 545 | depth 16 | 7142.2 | 7895.3 | 6959.8 | 0.9046 | — | — | — | — |
| 5 | 801 | depth 32 | 7579.6 | 7974.9 | 6943.5 | 0.9504 | — | — | — | — |
| 6 | 33 | control (repeat) | **7243.7** | 7782.2 | 6939.8 | **0.9308** | — | — | — | — |
| 7 | 33 | control + stamps | 6805.2 | 7917.8 | 6971.7 | 0.8595 | **426.4** | 2555.4 | 2747.8 | 315.9 |
| 8 | 97 | **A** + stamps | 6658.9 | 7869.2 | 6971.6 | 0.8462 | **368.4** | 2546.7 | 2801.5 | 301.6 |

Point 6 is the problem: the **same config as point 1**, 3.5 minutes later,
6.6 % slower and 7.5 % worse on the ratio. Points 4-6 all read high while 1-3
and 7-8 read normally, so the excursion is time-correlated, not config-borne.
Forensics found no cause: no non-`gpuagent` KFD process, no foreign container
with a GPU, no other tenant's output root touched, no stale GPU lease. Nothing
was preempted or signalled at any point tonight — **there was nothing to
preempt.** This is why e24b re-read the whole depth axis with three interleaved
controls and stamps on every point.

## 4. Mechanism A — the numbers

Pooling both batches (`timestamps=1` points only, because the stamps are the
instrument):

| population | n | plan M3→M5 mean | σ |
|---|---:|---:|---:|
| **without A** (`g` = 33, 289, 545, 801) | 7 | **428.96 µs** | 4.73 |
| **with A** (`g` = 97, 353) | 4 | **377.75 µs** | 7.56 |

**Δ = −51.2 µs, SE = 4.18, t = 12.3.** Restricting the control population to
`g = 33` only (n = 4, mean 429.75) gives −52.0 µs. The plan phase is
depth-invariant across `g` = 33/289/545/801 (425.0-435.4), exactly as expected:
the throttle acts in M7 and nowhere else, so the four non-A depths are legitimate
extra samples of the same plan-phase population.

**Corroboration on the aggregate.** `m2_to_end` over the depth-8 points:
6087.6 µs without A (n = 4) vs 5996.4 with A (n = 4) — **−91.2 ± 33.6 µs**
(t = 2.7). Consistent with −51 µs of plan plus unresolved change elsewhere.

**Bandwidth cross-check.** 448 MiB = 469,762,048 B removed in 51.2 µs implies an
effective write rate of **9.2 TB/s** — slightly *above* MI350X's 8 TB/s HBM3E
peak, which means part of the zero traffic never reached HBM (the 256 MB
memory-side Infinity Cache absorbing full-line `uint4` writes that are never
read is the obvious candidate). The load-bearing conclusion is the other one:
**the zero loop was fully exposed on the M3→M4 critical path.** CTA 0's serial
`tile_desc` build and CTA 1's `csr_scan_block256` are *not* the plan-phase
bottleneck — the pre-registered null hypothesis in `design.md` §1.5 is refuted.

**The LLC-eviction story is NOT supported, as pre-registered.** M6 is
2567.3 µs without A (n = 7, σ = 22.9) and 2552.6 with A (n = 4, σ = 14.5):
−14.7 ± 11.3 µs, t = 1.3. `design.md` §1.5 predicted this — W13 (939 MB) and W2
(469 MB) each exceed the 256 MB Infinity Cache by 2-4×, so there was no
resident weight working set for the zero to evict. Streaming 448 MiB through the
LLC costs what the bytes cost and nothing more.

**Follow-on lead.** After A, the plan phase still costs 378 µs for work that is
now just a 7.3 MiB scale transpose plus three grid barriers plus CTA 0's serial
`E = 32` expert loop. The plan phase is now plausibly **CTA-0-bound**, and
`KRN:1250-1273` (a `tid == 0` serial loop) is the next thing to look at.

## 5. Mechanism B — the numbers

M7 stamp, µs, pooled over both batches:

| depth | n | M7 mean | Δ vs depth 8 | significance |
|---:|---:|---:|---:|---|
| 4 | 2 | 2659.4 | −82.9 | t = 1.3 — **null** |
| **8 (shipped)** | 7 | **2742.3** (σ 63.1) | — | — |
| 16 | 1 | **3204.4** | **+462.1** | 7.3σ |
| 32 | 1 | **3112.2** | **+369.9** | 5.9σ |

exp_21 measured the **unthrottled** epilogue at M7 = 3,189 µs and depth 8 at
2,644-2,683. Depth 16 (3,204) and depth 32 (3,112) sit *at* the unthrottled
number. So the throttle is not a smooth knob: **its entire ~500 µs benefit is
realised at depths ≤ 8 and is completely gone by 16.** The knee is between 8 and
16, and both e24a and e24b agree on the ordering end-to-end (depth 16/32 read
0.90-0.95 `v_prod` against controls at 0.866-0.893).

Depth 4 is a null on M7 but was the faster arm end-to-end in both batches where
it appeared (6759.2 vs a 6902.3 control mean; and A+depth 4 at 6645.3 vs A-only
at 6744.7). Two favourable points is a hint, not a result, and it is inside the
drift band established in §3. **`g = 353` (A + depth 4) is the one composed
candidate worth a second campaign after `g = 97`.**

## 6. Screen resolution — a correction to the harness note

`../CONTEXT/harness_recipe.md` §5 puts `ratio_vs_prod` at σ = 0.52 % over six
repeats and says 2 % is the smallest callable delta. Tonight:

| population | n | spread |
|---|---:|---|
| e24a control (`g=33`, no stamps) | 2 | 6795.1, **7243.7** → 6.6 % |
| e24b control (`g=33`, stamps) | 3 | 6841.3-7012.8 → 2.5 % |
| all `g=33` points tonight | 6 | 6795.1-7243.7 → 6.6 % |

**The tail is fatter than six good samples suggested. At `w1t1p1` the smallest
end-to-end delta worth calling is ~3 %, not 2 %.** The device stamps are much
better instruments for phase-local mechanisms: plan M3→M5 has σ = 4.7 µs
(1.1 %) and M7 has σ = 63 µs (2.3 %), which is how a −51 µs effect became a
12σ result in 11 screens.

## 7. Resource tuple — no regression

Same compiler and flags, `~/exp24-base` (rev 23) vs the exp_24 build. Full
evidence in `build.md`.

| metric | rev 23 | exp_24 |
|---|---:|---:|
| SGPR / VGPR / AGPR | 106 / 256 / 256 | **106 / 256 / 256** |
| scratch B/lane | 144 | **128** |
| LDS B/block | 155,496 | **155,496** |
| `v_mfma` census | 180 | **180** |
| `flat_atomic_pk_add_bf16` | 282 | **282** |
| scratch ops inside an MFMA span | 0 | **0** |
| scratch ops within 24 insns of a remote atomic | 0 | **0** |
| `s_waitcnt vmcnt(4/8/16/32)` | –/96/–/– | **96/96/96/96** |

Two shapes of Mechanism B were built; the first (template the accumulate loop
per depth) put a `scratch_load` 5 instructions ahead of **every** remote atomic
— on the control path too — and was rejected before any GPU time was spent on
it. `build.md` §3 has the diagnosis and the fix (four SGPR lane masks, plus
folding `slot_off` into the peer table to free the register that pays for them).

**Confound, stated plainly:** that fold (MPS-DELTA (6)) means exp_24's control
arm is exp_21's protocol with three fewer instructions per remote atomic, so
`g = 33` here is **not** byte-identical to the 6,685 µs ratchet build. Both
mechanisms are read against the in-batch control, so their deltas are clean;
what this batch cannot certify is "control == ratchet". The `e24c` campaign
measures `g = 33` at campaign resolution precisely to price that fold.

## 8. Flag encoding — verified, not assumed

Host-side decode of every config actually run (`../tools/probes/p42_cfgdecode.sh`,
compiled against the node's own `moe_mps_adapter.cuh`):

| `g` | hex | phys | detect | throttle | depth sel | skip-zero | valid |
|---:|---|---:|---:|---:|---:|---:|---:|
| 33 | `0x021` | 1 | 0 | 1 | 0 (→8) | 0 | ✔ |
| 97 | `0x061` | 1 | 0 | 1 | 0 (→8) | **1** | ✔ |
| 289 | `0x121` | 1 | 0 | 1 | 1 (→4) | 0 | ✔ |
| 545 | `0x221` | 1 | 0 | 1 | 2 (→16) | 0 | ✔ |
| 801 | `0x321` | 1 | 0 | 1 | 3 (→32) | 0 | ✔ |
| 353 | `0x161` | 1 | 0 | 1 | 1 (→4) | **1** | ✔ |

Fail-closed cases all **rejected**: skip-zero + dual detector (`0x071`),
skip-zero on mode 13 or mode 2, a depth selector without the throttle enable
(`0x101`), and the reserved bits `0x080` and `0x400`. And the encoder widening
is **bit-identical for every `g ≤ 0xFF` across every mode 0-13** (0 mismatches
over 3,584 combinations), with `config_is_valid` unchanged on all legacy `g`.

That check also exposed a **real latent trap this experiment closed**: before the
widening, `g = 0x121` computed `0x121 << 8`, whose bit 16 landed in the **mode**
field (`mode |= 1`), and `decode_config` read back `g = 0x21`. A silently
corrupted config that still validated. Anyone who had tried a `g` above `0xFF`
would have measured the wrong mode and the wrong `g`.

## 9. Node discipline

- One 8-GPU job of ours at a time; `rocm-smi --showpids` showed only `gpuagent`
  (pid 44579) before every launch, and `screen.sh` re-checks before each point.
- **No foreign GPU process appeared at any time. Nothing was preempted,
  SIGTERMed, throttled or paused.** The only other container on the host
  (`yuhan_dsv4_0806`) held no KFD process.
- Every run under `setsid -w` + `timeout`; batches launched detached under
  `setsid nohup timeout`.
- `SCREEN_SYNC=0` for every batch **on purpose**: another agent is pushing to
  `codex/distributed-hipkittens-scaffold` tonight, and a mid-batch
  `git reset --hard` would swap the kernel under a half-finished sweep. The node
  was fetched, reset and source-verified against `520fe9c6` immediately before
  the first batch and left frozen after that.
- **Shared-tree note:** the local working tree was *not* the clean `0b82cd19`
  the dispatch assumed — HEAD had moved to `6101a71b` and another agent had
  uncommitted edits in `n2_phase1_gm_mps.cpp` plus `exp_25`/`exp_28` notes. Only
  exp_24's own files were staged and committed (`git add` by explicit path);
  nothing of theirs was committed, amended or reverted. The intervening commits
  changed no file the MPS kernel compiles.

## 10. Primitives

**Used.** `hkp::zero_part_scale_transpose<14>` (kept verbatim on the
non-mode-12 arm) · `encode_config` / `decode_config` / `config_is_valid` ·
`mode_is_direct_accum`, `detect_dual` · `kittens::distributed::accumulate_peer_bf162`
(`packet.cuh`) · the hand-tabulated `translate_peer` arithmetic in `m7tab`.

**Added** (all additive; every existing caller bit-identical):
`hk_moe::mps::scale_transpose_row` · `throttle_enabled`, `throttle_depth_sel`,
`skip_dead_part_zero` · `kRemoteAccumThrottleBit`,
`kRemoteAccumSkipPartZeroBit`, `kRemoteAccumThrottleDepthMask`,
`kRemoteAccumGLegalBits` · a 16-bit `g` field.

**Missing, wrong-shaped, or actively harmful — four findings.**

1. **A primitive that fuses jobs with different liveness makes dead work
   undeletable.** `hkp::zero_part_scale_transpose` welds a 14 KiB buffer zero to
   a scale transpose. They share nothing but a loop index: different
   destinations, different consumers, different lifetimes. Mode 12 killed the
   zero's consumer four experiments ago and **the fused primitive kept the dead
   half alive**, because deleting it required either editing a header we do not
   own or duplicating the live half. 448 MiB per rank per epoch survived that
   long for a purely structural reason. The primitive should be `zero_row` and
   `scale_transpose_row`, with the fused form as their composition — the fusion
   is a scheduling detail, not an interface. **Generalise the rule: never fuse
   two jobs in one primitive unless they share a consumer.**
2. **The `g`-field-as-selector idiom finally drew blood.** exp_20 and exp_21
   both flagged "config field reinterpreted as a diagnostic selector" as a
   smell; exp_24 is the experiment where it produced a *silently corrupting*
   encoding (`g = 0x121` overflowing into `mode` and still validating, §8). The
   library needs a typed config with named bitfields and a static width check
   per field, not an 8-bit byte with six ad-hoc bits and a `~mask` validator.
   Until then, every new bit is a chance to write into the next field.
3. **`packet.cuh` still has no pacing/depth knob — and exp_24 supplies the
   measurement that specifies it.** exp_21 asked for `max_outstanding`; exp_24
   now knows its shape: **useful at ≤ 8, worthless by 16, cliff in between**. It
   should be a **compile-time template parameter**, not a runtime field. Making
   it runtime cost two rejected shapes and a register-pressure fix here, because
   a per-atomic wait depth is an instruction immediate and any runtime carrier
   competes for registers in the most pressured region of the kernel. That is a
   design constraint the primitive should encode, not a fact each caller
   rediscovers.
4. **`peer.cuh`'s missing per-body tabulation needs to be offset-biased.**
   exp_21 asked for `peer_tab<8>::build(desc)`. exp_24 sharpens it to
   `peer_tab<8>::build(desc, uniform_byte_offset)`: pre-biasing the table by a
   uniform offset is what freed the 64-bit register that paid for this
   experiment's new state, and it removed three instructions from every one of
   117 M remote atomics. A tabulation primitive that could not absorb the offset
   would have left the spill in place. **The general primitive is "a peer
   address table pre-biased by a uniform base", not "a peer pointer table".**

## 11. Artifacts

Node: `~/overnight-scratch/screen_e24a.csv`, `screen_e24b.csv`,
`screen_e24c.csv`, `e24{a,b,c}_batch.out`, per-run logs
`~/overnight-scratch/e24{a,b,c}_*.log`, campaign roots `~/k0-mok-e24{a,b,c}/`.
Baseline snapshot for the resource gate: `~/exp24-base/fused_moe` (rev 23).
Build artifacts: `~/exp24-stage/build/{base,exp24}.{hsaco,elf,s,log}`.
Probes: `../tools/probes/p27..p49`.

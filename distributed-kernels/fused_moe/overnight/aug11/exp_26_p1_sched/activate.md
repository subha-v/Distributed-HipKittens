# exp_26 — activation patch (PREPARED, NOT APPLIED)

Apply this after the other agent's experiment lands and
`k0pf6gm_device_tile_mps.hip` is free. Nothing here has been applied; the
vendored body `distributed-kernels/fused_moe/n2_phase1_gm_mps.cpp` is already
committed and is inert until step 1.

Compile-time evidence backing the patch is in `build.md`. The default-off
vendored body is `.text`-byte-identical to the donor build, so step 1 alone is a
provable no-op; step 2 is the experiment.

> **The recommended arm changed after the 16-mask sweep: run
> `N2GM_P1_SCHED_GSCALE = 4`, not `1`/`15`.** The gate is a 4-bit mask now, and
> the win and the risk live on different bits. §2 has the reasoning; `build.md`
> §5 has the table.
>
> Two corrections to the context map are recorded in **§6**, and a one-line patch
> for a *different* file — `n2_phase2_gm_mps.cpp`, which this experiment must not
> touch — is written up as a separate proposed experiment in **§7**.

---

## 1. The kernel edit — two lines in `k0pf6gm_device_tile_mps.hip`

**(1a) Repoint the phase-1 include, line 395:**

```diff
-#include "n2_phase1_gm.cpp"
+#include "n2_phase1_gm_mps.cpp"
```

**(1b) The GSCALE mask — add immediately above that include:**

```diff
 #define N2GM_TASK_DONE_HOOK k0p6_a2_arrive(b, tid, k0p6_desc);
+// exp_26: 4-bit mask over phase 1's four K-loop sched_group_barrier hints.
+//   bit 0 DS read (4->4*kGM) | bit 1 VMEM (17->16+kGM)
+//   bit 2 MFMA   (16->16*kGM) | bit 3 DS write (1->kGM)
+// 0 = donor literals, .text-byte-identical to n2_phase1_gm.cpp (measured).
+// 4 = MFMA only: buys the 48/48 barrier split at the donor's exact resource
+// tuple. Bit 0 additionally moves the K-loop vmcnt(0) drain to mfma=48 but
+// migrates 96 scratch loads into M8/M9's remote-atomic path -- do not enable it
+// without reading build.md §6 first.
+#define N2GM_P1_SCHED_GSCALE 4
 #include "n2_phase1_gm_mps.cpp"
+#undef N2GM_P1_SCHED_GSCALE
 #undef N2GM_TASK_DONE_HOOK
```

For the control arm, set the value to `0` (do **not** delete the line — see §3).
The four `N2GM_P1_*_HOOK` macros and the two `N2GM_P1_TASK_*` macros are left
undefined on purpose; they take their donor defaults and cost nothing.

**(1c) Bump `K0P6_MPS_SRC_REV`** at `k0pf6gm_device_tile_mps.hip:104` — **to
whatever it is at the time, +1** (it read 23 at `0b82cd19`, and exp_24 is bumping
it too, so do not hardcode 24). Mandatory in the same commit; see §3.

---

## 2. Which mask to run, and why it is 4

All 16 masks were compiled and disassembled (`build.md` §5). They collapse to 8
distinct `.text` images and the four bits factor cleanly:

| bit | hint | effect, measured |
|---:|---|---|
| **2** | MFMA `16 → 16·kGM` | barrier partition **33/49/14 → 48/48/0**, at scratch 21, frame 144 B/lane, VGPR spill 16 — **the donor's numbers in every resource field** |
| **0** | DS read `4 → 4·kGM` | `vmcnt(0)` drain moves **mfma 0 → 48** — *and* 96 scratch loads + ~100 new full drains migrate into M8/M9's remote-atomic path (scratch 21 → 114) |
| 1 | VMEM `17 → 16+kGM` | **exact no-op**, all eight pairs byte-identical. The phase-1 VMEM hint was already right at `G=3` |
| 3 | DS write `1 → kGM` | perturbs the bytes, moves no metric |

**Mask 4 is the arm.** It is the only bit that changes the schedule while leaving
the resource profile byte-identical to the donor, so it is a genuinely
single-variable experiment: if the campaign moves, the barrier realignment moved
it. Mask 6 compiles to the same `.text` — same arm, redundant spelling; prefer 4
because it says what it does.

**Do not run 15 as the primary arm.** `build.md` §6 attributes all 96 migrated
accesses to a single source line, `n2_phase2_gm_mps.cpp:145` — the peer-table
base load in the remote-atomic accumulate loop — and shows a `scratch_load_dword`
plus a full `vmcnt(0)` landing in front of **96 of the 282**
`flat_atomic_pk_add_bf16`. That is character-for-character the regression exp_21
restructured `throttle_plan` into four booleans to remove. Enabling bit 0 pays
for an M6 schedule win with a serialization in the ~1,309 µs combine pool.

If mask 4 lands and there is time, mask 5 (MFMA + DS read) is worth **one**
campaign as an upper bound on the drain move — but only with attribution on, and
only expecting it to lose in `ts_combine_us`. The right follow-up for bit 0 is
not to run it as-is but to relieve the pressure that causes the eviction; see §7.

---

## 3. How to A/B it — compile-time only. A runtime `K0_MPS_CFG` bit is impossible.

**Do not try to plumb this through `K0_MPS_CFG`.**
`__builtin_amdgcn_sched_group_barrier(mask, size, syncid)` takes **immediate**
operands and is a compile-time scheduling directive with no runtime
representation. Selecting between hint sets at runtime would require two copies
of a 739-instruction loop behind a branch, which changes register allocation and
would no longer be the same experiment. This axis is compile-time by
construction.

**The recommended A/B is the source `#define` of §1b, flipped between two
campaigns**, because that value lands in a file the JIT cache key actually
hashes:

| route | in the mori JIT cache key? | verdict |
|---|---|---|
| `#define` in `k0pf6gm_device_tile_mps.hip` | **yes** — `.hip`, and it is copied into `KERNELS_DIR` | **use this** |
| `#define` in `n2_phase1_gm_mps.cpp` | **yes** — `.cpp`, also copied into `KERNELS_DIR` | equivalent; the `.hip` site is preferable because it sits next to the include it governs and next to `SRC_REV` |
| `-DN2GM_P1_SCHED_GSCALE=4` appended to `HIPCC_COMPILE_FLAGS_APPEND` in `e004pf_k0pf_ab.py` (≈:1179-1187) | **NO** — mori hashes the `_jit-sources` tree contents, not the compile flags | **silently reuses the previous hsaco.** Only safe if `SRC_REV` is bumped on every flip, which makes it strictly worse than the `#define` |
| runtime bit in `K0_MPS_CFG` | n/a | impossible, see above |

*(Why `.cuh` edits are the documented trap but `.cpp` edits are not: the key is
restricted to `.hpp/.h/.cpp/.hip`. `moe_mps_adapter.cuh` is copied into
`KERNELS_DIR` too, but its extension is excluded — that is exactly how exp_04
lost a measurement.)*

**A same-run paired A/B would need a second registered arm** (e.g.
`mps_mega_gscale` with its own module name, source copy and
`HIPCC_COMPILE_FLAGS_APPEND` scope). That is the ideal denominator but it is a
real harness change — a new `PF6MPS*_NAME`, a second `ensure_compiled` /
`load_hip_module` pair, a second entry in the arm table and the summarizer.
**Not worth it for one axis.** Use two campaigns and compare
`ratio_vs_prod`/`ratio_vs_pf6gm`, which is the same-run-anchored quantity the
ledger already uses for every other kernel change.

---

## 4. Harness edits — four of them, and two are mandatory or the JIT fails

File: `~/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py`
(mirrored locally at `.node/mps_remote_e004pf_k0pf_ab.py`). This is the
"may edit minimally, with a logged note" file — **mirror every one of these into
`MPS_OVERNIGHT_HARNESS_NOTE.md`.** Line numbers are approximate; the anchor text
is exact.

**(4a) MANDATORY — install the vendored body.** `PF6_N2_FILES` (≈:313-326) is an
allowlist; anything not in it is never copied into `KERNELS_DIR` and the JIT
compile dies with `FileNotFoundError`. Append one clause, and **leave the
existing `n2_phase1_gm.cpp` clause alone** — `pf6gm_mega` still includes the
donor, and keeping both installed is what preserves the reference arm:

```diff
 ) + (
     ("n2_phase2_gm_mps.cpp",) if PF6MPS_REQUESTED else ()
+) + (
+    ("n2_phase1_gm_mps.cpp",) if PF6MPS_REQUESTED else ()
 )
```

**(4b) MANDATORY — route it to the DHK tree.** The source-dir selector
(≈:913-919) sends everything except `n2_phase2_gm_mps.cpp` to
`PF6_N2_SOURCE_DIR` (amd-master). The vendored phase-1 body lives in the DHK
tree:

```diff
-                PF6MPS_SOURCE_DIR
-                if _source_name == "n2_phase2_gm_mps.cpp"
-                else PF6_N2_SOURCE_DIR
+                PF6MPS_SOURCE_DIR
+                if _source_name in ("n2_phase1_gm_mps.cpp",
+                                    "n2_phase2_gm_mps.cpp")
+                else PF6_N2_SOURCE_DIR
```

**(4c) Point the G-stack source contract at the file actually compiled**
(≈:376-387). The assertion checks that the phase body contains
`constexpr int kGM = N2GM_G`; the vendored file carries that line verbatim, so
this passes — but left unchanged it would be validating a file `mps_mega` no
longer includes:

```diff
             (
-                os.path.join(PF6_N2_SOURCE_DIR, "n2_phase1_gm.cpp"),
+                os.path.join(PF6MPS_SOURCE_DIR, "n2_phase1_gm_mps.cpp"),
                 os.path.join(PF6MPS_SOURCE_DIR, "n2_phase2_gm_mps.cpp"),
             ),
```

**(4d) Fix the provenance record** (≈:1213-1221) so `R["pf6mps"]["source_sha256"]`
hashes what was compiled:

```diff
-                "n2_phase1_gm.cpp",
+                "n2_phase1_gm_mps.cpp",
                 "n2_phase2_gm_mps.cpp",
```

Nothing else in the harness needs to change. `HIPCC_COMPILE_FLAGS_APPEND`
(≈:1179-1187) is left alone — the `#define` route needs no flag.

---

## 5. The campaigns to run

Node driver: `overnight/aug11/tools/screen.sh` (push with `tools/push.ps1`, run
via `tools/nsh.ps1`). It refuses to start on a busy node, syncs the node
checkout from `origin/codex/distributed-hipkittens-scaffold` once per batch, and
gives every run its own output root.

**Baseline to beat:** the exp_21 mode-12 ratchet at
`C=16,g=33,mode=12,flush_rows=16` — mps_mega 6,703.6 / 6,727.5 µs,
`ratio_vs_prod` 0.8592 / 0.8547 over two campaigns.

### Step 0 — smoke (≈10 min). Validates the harness edits and the rebuild, nothing else.

```bash
SCREEN_ARMS=production,mps_mega SCREEN_WARMUP=1 SCREEN_TIMED=1 SCREEN_PROCS=1 \
SCREEN_JOB_TIMEOUT=1800 SCREEN_SYNC=1 \
bash ~/Distributed-HipKittens/distributed-kernels/fused_moe/overnight/aug11/tools/screen.sh \
  e26smoke - <<'EOF'
C=16,g=33,mode=12,flush_rows=16,timestamps=1
EOF
```

Check before going further: `status=OK`, and `hsaco_before != hsaco_after` (the
`SRC_REV` bump must have forced a rebuild — if they match, the JIT aliased and
**every number after this is void**).

### Step 1 — control, `N2GM_P1_SCHED_GSCALE 0`, attribution on

```bash
SCREEN_ARMS=production,pf6gm_mega,mps_mega \
SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5 \
SCREEN_JOB_TIMEOUT=4500 SCREEN_SYNC=1 \
bash .../tools/screen.sh e26_m0 - <<'EOF'
C=16,g=33,mode=12,flush_rows=16,timestamps=1
EOF
```

Mask 0 is `.text`-identical to the donor build, so this arm doubles as a check on
the harness edits: if it does not reproduce the 6,703.6 / 6,727.5 band, something
in §4 is wrong and step 2's number means nothing.

### Step 2 — treatment, flip to `N2GM_P1_SCHED_GSCALE 4`, bump `SRC_REV` again, commit, push

```bash
SCREEN_ARMS=production,pf6gm_mega,mps_mega \
SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5 \
SCREEN_JOB_TIMEOUT=4500 SCREEN_SYNC=1 \
bash .../tools/screen.sh e26_m4 - <<'EOF'
C=16,g=33,mode=12,flush_rows=16,timestamps=1
EOF
```

### Step 3 — only if step 2 wins: the clean ratchet number, timestamps off

Repeat steps 1 and 2 with `C=16,g=33,mode=12,flush_rows=16` (no `timestamps`
key) so the headline µs is directly comparable to the 6,703.6 / 6,727.5 rows.

### Step 4 — optional, one campaign, mask 5, as an upper bound on the drain move

Only after 4 has a verdict, only with attribution on, and expect
`ts_combine_us` to rise. This is the measurement that prices bit 0's win against
its cost; it is not a candidate for the ratchet.

### What to read

| CSV column | what it decides |
|---|---|
| `ts_M6_us` | **the mechanism.** Mask 4 should lower it or the barrier realignment did nothing useful. |
| `ts_combine_us` | **the side effect.** Should be *flat* for mask 4 — its scratch placement is the donor's to the byte. If it moves, something outside this axis moved. For mask 5 this is the column that decides. |
| `ts_M7_us` | should be flat for mask 4. |
| `ratio_vs_prod` | the ratchet verdict. Beat 0.8547–0.8592 through the full gate ladder or it does not ship. |
| `gate_pass`, `pperr`, `soak`, `control_fails` | unchanged gates. A scheduling hint cannot change results; if any of these move, something else is wrong and the run is void. |

### Kill / ratchet rules

- `ts_M6_us` flat (within run-to-run noise) **and** `ratio_vs_prod` flat ⇒ the
  compiler was already doing the right thing despite the wrong hints, or the
  barrier skew costs less than it looks. Log the null in `LESSONS.md`, keep the
  vendored file (its hooks, its task macros and the corrected ISA facts in §6 are
  the durable output), set the mask back to 0.
- `ts_M6_us` down, everything else flat ⇒ new ratchet; leave the mask at 4 and
  re-baseline every later experiment against it.
- Mask 5 down in `ts_M6_us` but up in `ts_combine_us` by more ⇒ **that is the
  predicted result, not a surprise, and not a kill on the axis.** The follow-up
  is §7's second item: relieve the phase-2 register pressure so bit 0's drain
  move stops costing 96 spill-loads.

---

## 6. Two corrections to the context map — read these before planning anything nearby

Both are measured in `build.md`; both were previously stated the other way, and
both would have sent a later experiment after a ghost.

### 6.1 M6's K-loop **does** contain a compiler-inserted `s_waitcnt vmcnt(0)`

`m6_m7_structure.md` **§5.3 item 4 is wrong** where it says "Compiled out. There
is no `vmcnt(0)` inside M6's K-loop." Measured across all 17 builds:

- there is **exactly one `s_waitcnt vmcnt(0)` per K-loop iteration** — one per
  *iteration*, **not** one per half; one half carries it, the other carries none;
- it is **always a full drain**. No partial `vmcnt(N)` appears anywhere in either
  K-loop, in any mask;
- the source-level reasoning behind the old claim was correct —
  `n2_completion_observation_probe()` really is compiled out at
  `N2_FORCE_VMCNT0=0` — but the **compiler inserts its own** drain independently,
  and it is in the shipping build;
- in the donor schedule that drain sits at `mfma = 0`, i.e. before any of the
  iteration's MFMA work; bit 0 of the mask moves it to `mfma = 48`.

**§7 open item 4** ("No phase-1 disassembly exists in this tree") is **closed** —
the disassembly is at `~/overnight-scratch/e26/out2/*.isa` and the loop timeline
is in `build.md` §8.1. **§5.3 stall-site #1 should be amended** to "one full
`vmcnt(0)` drain per iteration, at `mfma = 0`".

### 6.2 Phase 2's `0x020` VMEM hint over-requests by 12

`n2_phase2_gm_mps.cpp` asks for `8 + 7·kGM` = **29** VMEM reads per K-loop half.
The measured count is **17**. Details, the measurement, and the patch are in §7 —
recorded here too because anyone reasoning about M7's schedule from the current
hints will get the wrong picture.

For contrast, phase **1**'s VMEM hint is correct: `16 + kGM` = 19 against a
measured 19, which is exactly why mask bit 1 turned out to be a byte-level no-op.

---

## 7. Proposed follow-up experiments (someone else's files — NOT part of this patch)

### 7.1 Phase 2's `0x020` hint — a one-line parameterization fix

**Owner:** whoever holds `n2_phase2_gm_mps.cpp` (exp_24 at the time of writing).
**This experiment did not and must not touch that file.**

Measured per-half census of phase 2's K-loop, by the same backedge+census method
that reproduces phase 1's hinted counts exactly:

| class | current hint | requested at `kGM=3` | **measured** | verdict |
|---|---|---:|---:|---|
| `0x008` MFMA | `14 * kGM` | 42 | 42 | correct |
| `0x200` DS write | `kGM` | 3 | 3 | correct |
| `0x100` DS read | `4 * kGM` | 12 | 25 | under, by the deliberate `read_a2`-only convention |
| **`0x020` VMEM read** | **`8 + 7 * kGM`** | **29** | **17** | **over by 12** |

The patch, at both halves of phase 2's K-loop — currently
`n2_phase2_gm_mps.cpp:602` and `:614`, though that file is being edited and the
lines will drift, so match on content:

```diff
-      __builtin_amdgcn_sched_group_barrier(0x020, 8 + 7 * kGM, 0);  // VMEM read
+      __builtin_amdgcn_sched_group_barrier(0x020, 14 + kGM, 0);     // VMEM read
```

`14 + kGM` = 17 at `kGM = 3`, matching the measurement, and = 15 at `kGM = 1`,
which is exactly what `8 + 7·kGM` gives at `kGM = 1`. **The two forms agree at
the anchor and diverge only for `kGM > 1`**, so this is a parameterization fix on
the same footing as the phase-1 one, not a retune — the `kGM=1` acceptance
property is preserved for free.

**Do I expect it to matter? Yes, more than the phase-1 fix.** Same defect class,
bigger pool: phase 2 is M7 at ~2,660 µs against M6's ~2,773 µs, and the error is
larger in both absolute and relative terms — phase 1's MFMA hint asked for 16 of
48 (a third of the truth), phase 2's VMEM hint asks for 29 where 17 exist, which
is the *opposite* failure and arguably worse. An over-request cannot be
satisfied: the scheduler is told to place 29 instructions of a class the region
does not contain, so the group cannot close on its own terms and the surrounding
groups are steered by a constraint that never resolves. Phase 1's MFMA
under-request produced a visibly wrong barrier split (33/49/14 instead of
48/48/0) from an error of 32; phase 2's error of 12 in the other direction is
smaller but sits in a loop with 84 MFMAs and 34 VMEM loads per iteration.

**Confidence: mechanism `DOCUMENTED` (the census is measured off the shipping
build's own ISA), magnitude `UNCERTAIN`.** Cheap to test — one `#define`-free
one-line edit, one `SRC_REV` bump, one paired campaign, and phase 2's K-loop
barrier partition is already 42/42/0 so the diagnostic is `ts_M7_us` plus a
before/after loop timeline. Worth running before anything more speculative in
M7.

### 7.2 Recover bit 0's drain move without the spill migration

Bit 0's relocation of the `vmcnt(0)` drain from `mfma = 0` to `mfma = 48` is the
largest schedule improvement this axis has found. Its cost is **not intrinsic**:
it is the whole-function register allocator evicting
`peer_tab[xr >> maxtok_sh]`'s base to scratch at phase 2's 256-VGPR ceiling, the
same eviction exp_21's four-boolean `throttle_plan` was built to prevent
(`build.md` §6). Candidate directions, for whoever owns the phase-2 file:

- pin the peer-table base so it cannot be the allocator's victim;
- relieve one live VGPR anywhere in the epilogue, which is all it took to cause
  this and should be all it takes to undo it;
- confirm the coupling by re-measuring mask 1 after any such change — if the 96
  `scratch_load_dword` sites disappear, bit 0 becomes free and the arm becomes
  mask 5.

This needs both files at once and therefore a single owner; it is out of scope
while exp_24 holds phase 2.

---

## 8. Rollback

Set `N2GM_P1_SCHED_GSCALE` to `0` (one character) and bump `SRC_REV`. That
restores the donor schedule exactly — proven at the `.text`-hash level in
`build.md` §4, not merely asserted, and that proof now includes the two
`N2GM_P1_TASK_*` macros at their defaults. Reverting the include as well is
optional and only worth doing if the vendored file is being retired entirely; the
harness edits in §4 are then also reverted.

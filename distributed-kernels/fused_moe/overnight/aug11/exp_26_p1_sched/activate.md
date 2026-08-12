# exp_26 — activation patch (PREPARED, NOT APPLIED)

Apply this after the other agent's experiment lands and
`k0pf6gm_device_tile_mps.hip` is free. Nothing here has been applied; the
vendored body `distributed-kernels/fused_moe/n2_phase1_gm_mps.cpp` is already
committed and is inert until step 1.

Compile-time evidence backing the patch is in `build.md`. The default-off
vendored body is `.text`-byte-identical to the donor build, so step 1 alone is a
provable no-op; step 2 is the experiment.

---

## 1. The kernel edit — two lines in `k0pf6gm_device_tile_mps.hip`

**(1a) Repoint the phase-1 include, line 395:**

```diff
-#include "n2_phase1_gm.cpp"
+#include "n2_phase1_gm_mps.cpp"
```

**(1b) The GSCALE switch — add immediately above that include:**

```diff
 #define N2GM_TASK_DONE_HOOK k0p6_a2_arrive(b, tid, k0p6_desc);
+// exp_26: scale phase 1's four K-loop sched_group_barrier hints by kGM. 0 =
+// donor literals (bit-identical to n2_phase1_gm.cpp). See
+// overnight/aug11/exp_26_p1_sched/{design,build}.md.
+#define N2GM_P1_SCHED_GSCALE 1
 #include "n2_phase1_gm_mps.cpp"
+#undef N2GM_P1_SCHED_GSCALE
 #undef N2GM_TASK_DONE_HOOK
```

For the control arm, set the value to `0` (do **not** delete the line — see §2).
The four `N2GM_P1_*_HOOK` macros are left undefined on purpose; they expand to
nothing and cost nothing.

**(1c) Bump `K0P6_MPS_SRC_REV`** at `k0pf6gm_device_tile_mps.hip:104` — **to
whatever it is at the time, +1** (it reads 23 as of `0b82cd19`, and exp_24 is
bumping it too, so do not hardcode 24). Mandatory in the same commit; see §3.

---

## 2. How to A/B it — compile-time only. A runtime `K0_MPS_CFG` bit is impossible.

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
| `-DN2GM_P1_SCHED_GSCALE=1` appended to `HIPCC_COMPILE_FLAGS_APPEND` in `e004pf_k0pf_ab.py` (≈:1179-1187) | **NO** — mori hashes the `_jit-sources` tree contents, not the compile flags | **silently reuses the previous hsaco.** Only safe if `SRC_REV` is bumped on every flip, which makes it strictly worse than the `#define` |
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

## 3. Harness edits — four of them, and one is mandatory or the JIT fails

File: `~/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py`
(mirrored locally at `.node/mps_remote_e004pf_k0pf_ab.py`). This is the
"may edit minimally, with a logged note" file — **mirror every one of these into
`MPS_OVERNIGHT_HARNESS_NOTE.md`.** Line numbers are approximate; the anchor text
is exact.

**(3a) MANDATORY — install the vendored body.** `PF6_N2_FILES` (≈:313-326) is an
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

**(3b) MANDATORY — route it to the DHK tree.** The source-dir selector
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

**(3c) Point the G-stack source contract at the file actually compiled**
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

**(3d) Fix the provenance record** (≈:1213-1221) so `R["pf6mps"]["source_sha256"]`
hashes what was compiled:

```diff
-                "n2_phase1_gm.cpp",
+                "n2_phase1_gm_mps.cpp",
                 "n2_phase2_gm_mps.cpp",
```

Nothing else in the harness needs to change. `HIPCC_COMPILE_FLAGS_APPEND`
(≈:1179-1187) is left alone — the `#define` route needs no flag.

---

## 4. The campaigns to run

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
bash .../tools/screen.sh e26_g0 - <<'EOF'
C=16,g=33,mode=12,flush_rows=16,timestamps=1
EOF
```

### Step 2 — treatment, flip to `N2GM_P1_SCHED_GSCALE 1`, bump `SRC_REV` again, commit, push

```bash
SCREEN_ARMS=production,pf6gm_mega,mps_mega \
SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5 \
SCREEN_JOB_TIMEOUT=4500 SCREEN_SYNC=1 \
bash .../tools/screen.sh e26_g1 - <<'EOF'
C=16,g=33,mode=12,flush_rows=16,timestamps=1
EOF
```

### Step 3 — only if step 2 wins: the clean ratchet number, timestamps off

Repeat steps 1 and 2 with `C=16,g=33,mode=12,flush_rows=16` (no `timestamps`
key) so the headline µs is directly comparable to the 6,703.6 / 6,727.5 rows.

### What to read, and why attribution is not optional here

`build.md` §5 found that GSCALE=1 moves **96 extra scratch (spill/reload)
instructions into the M7-epilogue / M8 / M9 region** while leaving both MFMA
K-loops scratch-free. In mode 12 that region is the ~1,309 µs pool. The two
effects push opposite ways and a single end-to-end number cannot separate them.

| CSV column | what it decides |
|---|---|
| `ts_M6_us` | **the mechanism.** GSCALE=1 should lower it or the schedule change did nothing useful. |
| `ts_combine_us` | **the side effect.** If this rises, the scratch migration is charging for the M6 win. |
| `ts_M7_us` | should be flat; if it moves, the register-allocation shift reached further than §5 measured. |
| `ratio_vs_prod` | the ratchet verdict. Beat 0.8547–0.8592 through the full gate ladder or it does not ship. |
| `gate_pass`, `pperr`, `soak`, `control_fails` | unchanged gates. A scheduling hint cannot change results; if any of these move, something else is wrong and the run is void. |

### Kill / ratchet rules

- `ts_M6_us` flat (within run-to-run noise) **and** `ratio_vs_prod` flat ⇒ the
  compiler was already doing the right thing despite the wrong hints. Log the
  null in `LESSONS.md`, keep the vendored file (its hooks and the corrected ISA
  facts are the durable output), set the `#define` back to 0.
- `ts_M6_us` down but `ts_combine_us` up by more ⇒ **do not discard the axis.**
  That is the scratch migration, not the schedule. The follow-up is to recover
  the old allocation in the M7 epilogue, not to abandon the hint.
- Both improve ⇒ new ratchet; leave `N2GM_P1_SCHED_GSCALE 1` in place and
  re-baseline every later experiment against it.

---

## 5. Rollback

Set `N2GM_P1_SCHED_GSCALE` to `0` (one character) and bump `SRC_REV`. That
restores the donor schedule exactly — proven at the `.text`-hash level in
`build.md` §3, not merely asserted. Reverting the include as well is optional
and only worth doing if the vendored file is being retired entirely; the
harness edits in §3 are then also reverted.

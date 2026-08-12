# exp_32 — result: the NaN poison is live, and the ratchet survives it

**Verdict: both questions answered, both the way we wanted.** The detector fires
on purpose (`nonfinite=57344`, exactly the predicted 7,168 x 8), and the
`C=16,g=353,mode=12,flush_rows=16` ratchet stays **green** with the poison on —
full ladder, 600/600 soak epochs, every gate statistic numerically unchanged.

Nothing in tonight's or earlier numbers was resting on stale `out` data at this
config. The staleness hole was real (the code path is exactly as exp_32's
`patch.md` described it) but `mps_mega` at mode 12 was not falling into it.

Confidence: **measured**, single node, one screen per point. The soak alone is
600 independent single-epoch coverage trials, which is the strongest part of the
evidence — it is not one check at the end.

---

## 1. What was applied

| artifact | detail |
|---|---|
| patch | `exp_32_poison.patch`, sha256 `d28558746f68c43fba4378dfbabd2b3c33cdd82e889557f6c9d86b3fcb258b0b` |
| applied with | `git apply -p1` in `~/amd-master/auto-gpu-kernel/k0_fused_moe`, **clean** |
| `e004pf_k0pf_ab.py` | `1eae5853…` (7,309 lines) -> `da8b43ea…` (7,401) -> `999a74cd…` (7,414, after V2a) |
| `run_campaign.sh` | `9aa04633…` -> `401accb9…` -> `99039433…` (after the self-test `-e` forward) |
| validation | `ast.parse` OK on the patched Python, `bash -n` OK on the patched shell |
| backups | `~/harness-backups/e32/20260812T053118Z/` (both files, pre-patch) |
| harness note | appended to `MPS_OVERNIGHT_HARNESS_NOTE.md` (now 193 lines, sha `9c755db8…`) |

Two things were added at apply time that the prepared patch did not carry:

1. **The V2a detector self-test, default ON**, behind
   `K0_MOK_POISON_SELFTEST`. Without it, `survivors=0` is unfalsifiable: a
   poison that silently never ran looks exactly like a clean arm.
2. **`K0_MOK_POISON_SELFTEST` forwarded** in `run_campaign.sh`'s `-e` list.
   `K0_MOK_POISON_OUT` was already forwarded by the patch; without the second
   line the self-test's off-switch would have been unreachable from the host —
   the documented `K0_MPS_SKIP_LAUNCH` trap.

### A driver fix that was mandatory, not cosmetic

`overnight/aug11/tools/screen.sh` parsed the soak line with
`pperr=(\d+) pass=(\w+)`, anchoring `pass=` directly to `pperr=`. The patch
inserts `poison=` and `poison_epoch=` between them, so **every poison-on run
would have been recorded as `FAIL:soak=MALFORMED`** — a green run reported red.
The regex is now tolerant of both formats. Two additions while there:

- a run whose self-test reads `False` is recorded `VOID:selftest`, not `OK`;
- nonzero `[POISON] … survivors=N` lands in the `note` column.

The CSV header is unchanged, so in-flight batches on other tags keep parsing.

---

## 2. The detector fires on purpose — the self-test

Every candidate arm, every run:

```
[POISON SELFTEST] arm=pf6gm_mega one_row_poisoned_fails=True nonfinite=57344 relative=nan
[POISON SELFTEST] arm=mps_mega   one_row_poisoned_fails=True nonfinite=57344 relative=nan
```

`nonfinite=57344` is `7,168 x 8` exactly: one row of `H = 7,168` bf16 elements,
SUM-all-reduced over eight ranks. That is the arithmetic the design predicted, so
the count confirms the *mechanism*, not just the boolean. All three independent
detection paths engaged: the equality gate `nonfinite == 0`, and — visible in
`relative=nan` — the NaN poisoning `diff.sum()`, which also kills
`math.isfinite(relative_error)`.

This is a poisoned row inside an **otherwise correct** buffer, restored
bit-for-bit afterwards, so it establishes the detector's sensitivity floor at
**one row in 4,096** rather than at some large fraction of the output.

**Why this mattered.** Poisoning `out` with `0.0` (what the harness did before)
gives essentially no protection: one missing row of 4,096 moves the relative-L1
metric by ~0.024 % against a 0.1 gate, 400x under, and `abs_error_max` only
catches it if that row's reference happens to exceed 0.1 (whole-buffer `max_abs`
is 0.035156, so it would not have). The poison closes exactly one bug class — a
row that was **not written at all** — and the self-test is what proves the class
is closed rather than asserted.

---

## 3. The re-gated ratchet — green

One screen batch, two points, `production,pf6gm_mega,mps_mega`, node HEAD
`123b9e31`, `SRC_REV 24`, hsaco `33125b154cff` for both points.

| gate | point 1 (ratchet cfg) | point 2 (+`timestamps=1`) | required |
|---|---|---|---|
| `screen.sh` status | **OK** | **OK** | OK |
| `[POISON] eager arm=mps_mega` | `survivors=0` | `survivors=0` | 0 |
| `[POISON] post_timing arm=mps_mega` | `survivors=0` | `survivors=0` | 0 |
| `[POISON] eager/post_timing arm=pf6gm_mega` | `survivors=0` | `survivors=0` | 0 |
| `[MPS SOAK]` | `600/600 pperr=0 poison=0 poison_epoch=-1 pass=True` | same | 600/600, poison 0 |
| `[MOK GATE] mps_mega` | `max_abs=0.035156 relative=0.008293 pass=True` | identical | unchanged vs a11base |
| `[MARK] control_fails` | `True` | `True` | True |
| `pperr` | 0 | 0 | 0 |

`[MOK GATE] mps_mega max_abs=0.035156 relative=0.008293` is **bit-for-bit the
pre-patch value**, which is the harmlessness half of the argument: the poison
adds a condition, it does not perturb the numerics being gated.

### Timing — the poison is outside the timer

| point | prod | pf6gm | mps | mps/prod | pf6gm/prod |
|---|---:|---:|---:|---:|---:|
| 1, ratchet cfg | 7,882.6 | 6,933.2 | 6,672.6 | **0.8465** | 0.8796 |
| 2, +timestamps | 7,852.8 | 6,926.4 | 6,654.3 | **0.8474** | 0.8820 |

Read the **ratios**, not the microseconds: these are 1-process `w1t1` screens and
absolute µs drift by several percent against a campaign. 0.8465 / 0.8474 versus
the ratchet campaign's 0.8522 is 0.6–0.7 % — inside the documented 6.6 % screen
tail, and in the direction screens always sit. **This is not a re-measurement of
the ratchet** (that needs 5 rotations); it is a demonstration that turning the
poison on did not move the arm.

### The poison-on stamp reference (point 2, donor-include path)

| stamp | value (µs) |
|---|---:|
| `plan_M3toM5` | 373.4 |
| **`M6`** | **2,544.0** |
| `M7` | 2,655.9 |
| `combine` | 393.7 |
| `servicedrain` | 2,707.8 |
| `m2_to_end` | 5,967.0 |

**Caveat worth carrying forward, and it is mine, not the patch's.** These stamps
come from the final **soak** epoch, and P2 now re-poisons before every soak
epoch, so a 56 MiB NaN fill (22 % of the 256 MB Infinity Cache) lands
immediately before the stamped epoch.

**Updated with exp_26's data (n = 10 rather than n = 1).** exp_26 then measured
this same code path — mask 0 is `.text`-identical to the donor include — over two
independent batches: **M6 = 2,541.51 ± 4.77 µs, reproducing to 0.34 µs between
batches.** So the 2,544.0 above was an ordinary sample (within 0.5 sigma), and
two things follow:

- **The M6 stamp's real sigma is 4.8 µs, not the planned 23 µs**, and its
  batch-to-batch reproducibility is 0.34 µs. It is a far better instrument than
  we assumed.
- **The gap to the historical ~2,588 µs is therefore ~46 µs, or ~10 sigma — it is
  not run noise.** I cannot attribute it: the poison's pre-epoch cache eviction
  and a config difference (the historical figure predates `g=353`) are both live
  candidates and this experiment does not separate them. Recorded as **open**.

The procedural rule stands and is now load-bearing: **poison-on M6 stamps are not
comparable to pre-poison M6 stamps.** Any experiment judged on this stamp must
carry its own control measured under the same poison setting, which is what
exp_26 did.

---

## 4. What the poison does NOT cover

Stated so nobody over-reads a green run:

- **It closes one class: a row never written.** exp_29's 19 %-error-on-one-token
  case — a token short one of ~5.25 addends — dilutes to ~0.3 % globally and
  still passes every gate. The poison does not help there and I am not claiming
  it does.
- **The negative control still keeps `zero_()`** on purpose. Poisoning it would
  make `control_fails=True` fire because of the poison, converting "the gate can
  detect a wrong value" into "the gate can detect our own NaN".
- **`production` is not poisoned** — it reads `_ph["prod"]`, a mori-managed
  `op.combine` output the harness does not own. The asymmetry is in gating
  strength only, never in timing, so it cannot move a ratio.
- **If A11/M11 ever lands** (direct remote bf16 atomic accumulation into `out`),
  the "plain store, never a read-modify-write" property dies and the poison must
  become a zero-fill plus an explicit coverage bitmap. Flagged in `patch.md` §1
  and repeated here because it is a silent correctness trap for whoever builds
  that arm.

---

## 5. Node discipline

`rocm-smi --showpids` showed only `gpuagent` (pid 44579) before the launch and
`pgrep -af 'torchrun|mpirun'` was empty; `screen.sh` re-checks both before every
run and the whole batch ran under `setsid` + `timeout`. **No foreign GPU process
appeared at any point, so nothing was preempted and there is nothing to log.**

---

## Primitives

This experiment touched **no** kernel code and no
`include/cdna4/ops/group/distributed/**` primitive — it is a host-side gate
hardening — so there is no primitive to report used, missing or widened. Recording
the null explicitly rather than omitting the section.

One observation that *is* about the library's surface, for the ledger: the bug
class this patch closes is **"a protocol whose readiness edge is absent still
returns the right answer"**, and that is a property of the *harness*, not of the
primitives. No amount of `completion.cuh` / `counter.cuh` discipline inside the
kernel can make a missing edge visible when the consumer buffer already holds
the correct bytes from the previous epoch. The primitives cannot self-test; the
measurement rig has to. Worth remembering before trusting any future
`publish_tile_release` / `wait_tile_acquire_into` pairing on the strength of a
green campaign alone — exp_25's `a2_done` edge is the immediate example, where a
**completely absent** readiness edge would have returned the right answer *and*
posted the best number in the sweep.

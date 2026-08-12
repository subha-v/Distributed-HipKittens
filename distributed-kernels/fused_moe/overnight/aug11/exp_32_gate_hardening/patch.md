# exp_32 — NaN-poison `out` between epochs (measurement integrity)

**Status:** authored, **NOT applied**. Prepared while exp_24 holds the GPU and the
three MPS kernel/host files. No GPU job was launched and no tracked tree, harness
file or kernel source on the node was modified. The patch was built and dry-run
verified against local `scp` copies whose sha256s match the node's.

*Full disclosure of the only node write:* verifying `bash -n` on the patched
`run_campaign.sh` needed a shell on the node, so the file was piped to
`/tmp/e32_rc_check.sh`, syntax-checked, and deleted in the same command. Nothing
outside `/tmp` was touched.

**Anchors.** Node `gbt350-odcdh2-c05-1`, 2026-08-12.
- `prefill_opt/host/e004pf_k0pf_ab.py` — sha256
  `1eae5853ddc981e0fb57dd548fa8c5a29b7bfbcad5e8f3a37482bbb4651e0dfb`, 7,309 lines,
  mtime `2026-08-11T17:30:05Z`. **All `ab.py:N` citations below are this file.**
- `benchmarks/mok_synthetic_prefill/run_campaign.sh` — sha256
  `9aa0463344a7b8f230beab6124191ea2c9471ae7a3557105173e94680558cded`, 190 lines.
- `benchmarks/mok_synthetic_prefill/correctness.py` — sha256
  `b93be545a3e4b596…`, 78 lines (unchanged since 2026-08-06; **not patched**).
- Node checkout HEAD `0b82cd19`, kernel `k0pf6gm_device_tile_mps.hip` sha256
  `0c92a1c9713fcf31adfb759336e381f41fb1b8b27022f62c05d10f8ebc1832b3`, 1,863 lines.
- The local mirror `.node/mps_remote_e004pf_k0pf_ab.py` is **stale** (7,255 lines;
  anchors shifted by −20 to −66 lines, and `device=dev` where the node now has
  `device="cuda"`). It was not used to build the diff.

---

## 1. Where `out` lives, and who reads it

`cand_out` is allocated once at **`ab.py:1799`**:

```python
cand_out = torch.zeros((T, H), dtype=torch.bfloat16, device=dev)
```

`T = 4096` (`K0_T`, hard-wired by `run_campaign.sh:122`), `H = 7168` ⇒
`4096 × 7168 × 2 = 58,720,256 B = 56 MiB`. A plain local torch tensor — **not**
symmetric, not mori-managed (`mode12_protocol_map.md` §"out", `DRV:1799`).

**Every candidate arm shares this one buffer.** `_obuf(name)` (`ab.py:2981-2990`)
returns `_ph["prod"]` for `production` (the tensor `op.combine` hands back) and
`cand_out` for everything not in `_PROD_LIKE` — so `pf6gm_mega` and `mps_mega`
read and write the same 56 MiB.

The only writer in the `mps_mega` arm is M8, `KRN:592-603`:

```591:603:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      if (!live[t]) continue;
      unsigned short* const orow = out + (size_t)(tok0 + t) * (size_t)K0P6_H;
      union {
        uint4 pack;
        unsigned short u16[8];
      } pk;
#pragma unroll
      for (int e = 0; e < 8; ++e) pk.u16[e] = k0p6_f32_to_bf16_rne(acc[t][e]);
      *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(orow) + off) = pk.pack;
    }
```

Three properties that make poisoning safe:

1. **Plain store, never a read-modify-write.** Poison bytes can never contaminate
   a correct result — they are overwritten, not accumulated into. (If A11 / M11
   ever lands — direct remote bf16 atomic accumulation *into* `out` — this
   property dies and the poison must move to a zero-fill plus an explicit
   coverage bitmap. Flagged here so nobody discovers it the hard way.)
2. **A written row is fully written.** `off = (c<<10) + (lane<<4)` for
   `c ∈ [0,14)`, `lane ∈ [0,64)` covers `14 × 64 × 16 = 14,336 B = H × 2 B`
   exactly — one complete row, no partial coverage.
3. **Row coverage is `tok < T` and nothing else.** `nbatches = ⌈T/NT⌉ = 4096/4 =
   1024` exactly (`KRN:1774`, `NT = 4` at `KRN:1755`), so there is no remainder
   batch and no legitimately-unwritten row inside `[0, T)`. A poison survivor is
   therefore always a bug, never a design gap.

### Where each gate reads it

| # | site | node line | clear before? | what it gates |
|---|---|---|---|---|
| G1 | eager loop, one epoch per arm | `ab.py:4503-4521` | **yes**, `cand_out.zero_()` at `4508` | `[MARK] eager` — **not a gate** (`ok()` reads `False` even for `production`) |
| G2 | `[MOK GATE]` loop | `ab.py:4569-4589` | no (runs after G1 finished) | `R["mok_correctness_all_pass"]` → **real gate** (`ab.py:4906`) |
| G3 | negative control | `ab.py:4816-4831` | **yes**, `zero_()` at `4816` | `control_fails=True` → **real gate** |
| G4 | 600-epoch soak | `ab.py:4948-4959` | **NO CLEAR AT ALL** | `[MPS SOAK]` → **real gate** |
| G5 | per-arm post-timing | `ab.py:5052-5092` | `zero_()` at `5054`, then 500 warmup + 100 timed epochs | `arm["correctness"]["pass_all_ranks"]` → **real gate** (`ab.py:5227`, into `R["timing_gate_ok"]` at `5261`) |

So the precise form of the bug is **not** "`out` is never cleared" — it is
cleared once per gate episode. It is **never cleared between epochs**:

- **G5** clears once at `5054`, then runs **600 epochs** (500 warmup + 100 timed)
  before gating at `5080-5082`. Any epoch after the first that skips a row still
  reads the earlier epoch's bit-identical correct value.
- **G4** clears **never**. It inherits whatever G3's deliberately-broken combine
  left behind, runs 600 epochs, and gates once. (That accident gives partial
  protection against "never written in 600 epochs" and **zero** protection
  against "not written in epoch 600" — the realistic shape.)
- **G1** does clear per arm, but to **zero**, which is not a poison (§3).

### The gate the poison hooks into — verified, load-bearing

`correctness.py` is unpatched and does exactly what we need:

```29:59:.node/mps_remote_bench/correctness.py
    reference_float = reference.float()
    actual_float = actual.float()
    diff = (reference_float - actual_float).abs()

    diff_sum = diff.sum()
    diff_count = torch.tensor(
        diff.numel(), dtype=torch.float64, device=diff.device
    )
    diff_max = diff.max()
    reference_sum = reference_float.abs().sum()
    nonfinite = (~torch.isfinite(actual_float)).sum().to(torch.float64)

    if distributed is not None:
        distributed.all_reduce(diff_sum, op=distributed.ReduceOp.SUM)
        ...
        distributed.all_reduce(nonfinite, op=distributed.ReduceOp.SUM)
```

and

```68:78:.node/mps_remote_bench/correctness.py
    values = (
        stats["abs_error_mean"],
        stats["abs_error_max"],
        stats["relative_error"],
    )
    return bool(
        all(math.isfinite(value) for value in values)
        and stats["nonfinite"] == 0
        and stats["abs_error_max"] <= absolute_tolerance
        and stats["relative_error"] <= relative_tolerance
    )
```

**One surviving NaN element, on any one of the eight ranks, fails `[MOK GATE]`
three independent ways:** the explicit SUM-all-reduced `nonfinite == 0`; the
`math.isfinite(abs_error_max)` check (NaN poisons `diff.max()`); and
`math.isfinite(relative_error)` (NaN poisons `diff.sum()`). There is no
`nan_to_num`, no clamping, no masking anywhere in the path. `nonfinite == 0` is
an **equality**, not a tolerance — it cannot be widened by a config edit, which
is why the poison targets it rather than the magnitude gate.

---

## 2. Design question 1 — poison where?

**Four insertion points, all provably outside every timer. One deliberate
omission.**

| tag | site | node line | what it buys |
|---|---|---|---|
| **P0** | move `mok_gate` into the eager loop | `4510` / `4571` | *prerequisite* — makes P1 mean anything (see below) |
| **P1** | eager loop: poison instead of zero | `4508` | per-arm single-epoch coverage check before any timing |
| **P2** | soak: poison **before every one of the 600 epochs** + per-epoch survivor count folded into the existing all-reduce | `4948` | **the strongest detector: 600 independent single-epoch trials** |
| **P3** | per-arm: poison at `5054`, and **one extra untimed epoch after all timing** before the gates | `5054`, `5079` | post-timing `[MOK GATE]` becomes a real coverage check in the steady state the timed loop just left |
| — | negative control | `4816` | **left as `zero_()` on purpose** |

**P0 is not optional.** `ab.py:4569-4589` computes `mok_gate(_obuf(name))` for
every arm *after* the eager loop at `4503` has finished, so for every candidate
arm it reads the **same** `cand_out` bytes — whichever candidate ran last. The
gate line names `pf6gm_mega` and `mps_mega` but reads one buffer.
`_obuf` (`ab.py:2981-2990`) returns the identical object for both, so this is a
code fact, not an inference. (The bit-identical numbers in
`CONTEXT/harness_recipe.md` §3 are *consistent* with it but are not proof, since
the two arms may legitimately agree bit-for-bit.) Consequence: without P0, P1's
poison for `mps_mega` is only checked when `mps_mega` happens to be last in the
rotated arm order. P0 stashes each arm's `mok_gate` result inside the loop, while
its own bytes are live, and has the later loop consume the stash.

**Why not between timed iterations.** `ab.py:5068-5071` is

```python
        for start, end in events:
            start.record()
            body(sp())
            end.record()
```

A fill enqueued between `end.record()` of iteration *i* and `start.record()` of
iteration *i+1* would technically fall outside every `[start_i, end_i]` window,
so the HIP-event arithmetic would not include it. It is still refused, for three
reasons: (a) it burns ~10 µs of device time per iteration, 100 times per arm;
(b) it writes **56 MiB — 22 % of the 256 MB Infinity Cache — between every pair
of measured epochs**, displacing exactly the weights M6 and M7 are about to want,
which is the one thing this patch exists to protect; (c) there is no per-iteration
barrier in the timed loop, so delaying each rank's next launch changes the
inter-rank skew, and the reported number is a **rank-max** p50 — skew *is* the
measurement. P3's single post-timing epoch gets the same coverage guarantee at
zero measurement risk.

**Why the negative control keeps `zero_()`.** `4816-4831` runs the frozen
pipeline plus `fn_combine_bug` (a deliberate store-over-a-peer bug) and requires
`gate(cand_out)` to **fail**. Poisoning that buffer would make it fail because of
the poison, converting "the gate can detect a wrong value" into "the gate can
detect our own NaN" — the control would still print `True` while having stopped
testing anything. Left untouched.

## 3. Design question 2 — poison with what?

**`0x7FD5`: a bf16 quiet NaN with payload `0x55`** (sign 0, exponent `0xFF`,
mantissa `0x55`), written through an `int16` view (`0x7FD5 = 32725`, inside
`int16` range; `cand_out.view(torch.uint16)` already appears at `ab.py:3733` and
`3968`, so 2-byte views of this tensor are known-good on this torch build —
`int16` is used here only so `fill_` is unambiguously supported).

Why NaN and not a magnitude sentinel: it lands on `nonfinite == 0`, a hard
equality gate that cannot be dialed, and it is independent of the reference's
magnitude. A large finite sentinel (`3e38`) was considered — it would trip
`abs_error_max <= 0.1` — and rejected as the *primary* because that is a
tolerance comparison, one edit away from being widened, and the project rule is
that tolerances are gates not dials. NaN gets both: `nonfinite` **and**, as a
free side effect, `math.isfinite(abs_error_max)`.

**Why zero is not a poison.** Today's `zero_()` gives almost no protection:
(a) `0.0` is a legal bf16 output and contributes nothing to `nonfinite`;
(b) one missing row out of 4,096 moves the L1-ratio metric by ≈ `1/4096 = 0.024 %`
against a `0.1` relative gate — **400× under**; (c) only `abs_error_max` could
catch it, and only if that row's reference happens to contain an element above
`0.1` (today's whole-buffer `max_abs` is `0.035156`, so the margin is thin).
The 19 %-error-on-one-token case from exp_29 — a token short one of ~5.25
addends — dilutes to ~0.3 % and passes every gate untouched; the poison does not
fix *that* case either, and I am not claiming it does. **The poison closes
exactly one class: a row that was not written at all.**

**Distinguishing poison from a NaN the kernel produced.** The kernel's only
plausible NaN source is the FP8 quantization scale at `KRN:887`:

```887:891:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip
        const float scale = fmaxf(a, 1.0e-6f) / 448.0f;
        ...
          const float q = fminf(fmaxf(k0p6q_bf16_to_f32(h16[e]) / scale, -448.0f), 448.0f);
```

the `fmaxf(a, 1.0e-6f)` epsilon is precisely the guard that stops `448/0` for a
rank whose experts are empty, so a computed NaN there would itself be a bug. If
one ever appeared: an f32 `0/0` NaN rounds to bf16 `0x7FC0` (or `0xFFC0`), and an
overflow gives inf `0x7F80`. **None of those equal `0x7FD5`.** The patch prints
the exact count of elements still equal to `0x7FD5` plus the first bad row
indices, so the two failures separate on one line:

- `[POISON] … survivors=N first_rows=[…]` **and** `nonfinite=N` ⇒ rows not
  written (coverage / staleness).
- `nonfinite=N` **with** `survivors=0` ⇒ the kernel computed a NaN or an inf in a
  row it *did* write (numerics). Different bug, different owner.

## 4. Design question 3 — does it change what the benchmark measures?

**No.** Auditable claims:

- **No poison operation is inside any timed region for any arm.** P1/P2 are in
  loops that carry no HIP events. P3's verification epoch is inserted **after**
  `torch.cuda.synchronize(); dist.barrier()` (`5072-5073`), after every
  `start.elapsed_time(end)` has been read (`5075-5078`), and after the
  `_mok_rank_max` all-reduce (`5079`). There is nothing left to perturb.
- **All three arms take the identical code path.** The only arm-dependent
  branch is the pre-existing `if name != "production"` at `5053-5054`;
  `production` is skipped there today and is skipped by the poison for the same
  reason — it reads `_ph["prod"]`, not `cand_out`.
- **`production` is timed exactly as today and gated exactly as today.** The
  patch does not touch `_ph["prod"]`. Poisoning it was considered and rejected:
  it aliases a mori-managed `op.combine` output whose lifetime the harness does
  not own, and `production` is the trusted denominator, not a candidate under
  test. The resulting asymmetry is in **gating strength only, never in timing**,
  so it cannot move a ratio in either direction.
- **Nothing about the workload changes**: same shapes, same `K0_MOK_SEED_BASE`,
  same route, same `MOK_WARMUP_ITERS`/`MOK_TIMED_ITERS`, same tolerances. No
  tolerance is touched; the patch only *adds* a condition.
- **Added wall clock**, for the record: a 56 MiB fill is ~10 µs and a 56 MiB
  compare-and-reduce is ~10 µs. P1 = 3 × ~20 µs. P2 = 600 × ~20 µs ≈ **12 ms** on
  a ~4 s soak. P3 = 3 × (~20 µs + one ~6.7 ms epoch) ≈ **20 ms**. Total ≈ 35 ms
  added to an ~18-minute campaign process.

## 5. Design question 4 — knob, and which way it defaults

`K0_MOK_POISON_OUT`, **default `1` (on)**, and **forwarded** in
`run_campaign.sh`'s `-e` list.

- **Default on**, agreeing with your preference and for a sharper reason than
  convenience: the premise of the whole experiment is that a currently-green
  number may be resting on stale data. A gate hardening that defaults off
  protects nothing on the night it is needed and will be forgotten by morning.
- **Default on is safe** because the poison is outside every timed region — the
  usual reason to default a measurement-adjacent knob off does not apply here.
- **The knob must still exist**, for two things a hard-coded change could not do:
  (a) A/B the poison itself, to prove it is live rather than a silent no-op —
  that is the only way to distinguish "green because correct" from "green because
  the poison never ran"; (b) reproduce a pre-patch number bit-for-bit if a
  disputed measurement has to be re-derived.
- **It must be forwarded or the off-switch is a lie.** Because the default lives
  inside `ab.py`, a missing `-e` line leaves the poison **on and unkillable from
  the host** — precisely the documented `K0_MPS_SKIP_LAUNCH` trap
  (`harness_recipe.md` §2.3). The `run_campaign.sh` hunk below is therefore part
  of the patch, not an optional extra.
- Not behind `K0_MPS_*`: this hardens the gate for **every** arm, so it belongs
  to the `K0_MOK_*` family.

---

## 6. The patch

Two files, one diff. **Verified**: generated mechanically from the two node files
(scp'd locally, sha256 confirmed against the node), then dry-run applied to a
clean copy with `git apply -p1` -- **applies cleanly**, and the result is
byte-identical to the intended file. The patched `e004pf_k0pf_ab.py` parses
(`ast.parse`) and the patched `run_campaign.sh` passes `bash -n`. Apply from
`~/amd-master/auto-gpu-kernel/k0_fused_moe`:

The same bytes are shipped as a ready-to-apply file next to this document:
`exp_32_gate_hardening/exp_32_poison.patch` (199 lines, sha256
`d28558746f68c43fba4378dfbabd2b3c33cdd82e889557f6c9d86b3fcb258b0b`).

```bash
git apply -p1 exp_32_poison.patch      # or: patch -p1 < exp_32_poison.patch
```

```diff
--- a/prefill_opt/host/e004pf_k0pf_ab.py
+++ b/prefill_opt/host/e004pf_k0pf_ab.py
@@ -150,6 +150,16 @@ if MOK_WARMUP_ITERS < 0 or MOK_TIMED_ITERS <= 0:
     raise ValueError("MoK warmup must be nonnegative and timed iterations must be positive")
 if MOK_ABSOLUTE_TOLERANCE < 0 or MOK_RELATIVE_TOLERANCE < 0:
     raise ValueError("MoK correctness tolerances must be nonnegative")
+# exp_32 gate hardening. `cand_out` is cleared once per gate episode but NEVER
+# between epochs, and the MoK corpus feeds identical input and identical routing
+# every iteration, so a row a candidate fails to write returns the previous
+# epoch's bit-identical correct answer -- invisible to [MOK GATE], to
+# combine_bit_exact, and to 600 soak epochs. Poison with a value that cannot
+# survive a correct run; correctness.py's `nonfinite == 0` gate then catches it.
+MOK_POISON_OUT = os.environ.get("K0_MOK_POISON_OUT", "1").strip() not in ("", "0")
+# bf16 quiet NaN, payload 0x55. NOT 0x7FC0 (what a computed 0/0 rounds to) and
+# NOT 0x7F80 (inf), so a survivor is distinguishable from a kernel-produced NaN.
+MOK_POISON_I16 = 0x7FD5
 GBLK = int(os.environ.get("K0_GBLK", "256"))   # frozen gather block (best_1075x)
 CBLK = int(os.environ.get("K0_CBLK", "512"))   # frozen combine block (best_1075x)
 NSLOT = 2                                        # double-buffered epoch slots
@@ -1797,6 +1807,40 @@ pf2_n2r_flags, pf2_n2r_flags_p = mori_t((WORLD, 1), "int32")
 pf2_n2r_flags.zero_()
 pf2_n2r_grid = torch.zeros((1,), dtype=torch.int32, device="cuda")
 cand_out = torch.zeros((T, H), dtype=torch.bfloat16, device=dev)
+
+
+def _poison_out():
+    """exp_32: fill the shared candidate output with a recognisable bf16 NaN.
+    Called ONLY from untimed regions (eager gate, soak, post-timing verify)."""
+    if MOK_POISON_OUT:
+        cand_out.view(torch.int16).fill_(MOK_POISON_I16)
+    else:
+        cand_out.zero_()
+
+
+def _poison_count():
+    """Elements still holding the poison pattern. Zero in a correct run."""
+    if not MOK_POISON_OUT:
+        return 0
+    return int((cand_out.view(torch.int16) == MOK_POISON_I16).sum().item())
+
+
+def _poison_rows(limit=16):
+    _bad = (cand_out.view(torch.int16) == MOK_POISON_I16).any(dim=1)
+    return torch.nonzero(_bad, as_tuple=False).flatten()[:limit].cpu().tolist()
+
+
+def _poison_report(tag, name):
+    """Print and return the survivor count. Nonzero means the arm left a row of
+    `out` unwritten this epoch -- the exact bug class a stale `out` hides."""
+    _n = _poison_count()
+    if _n:
+        print(f"[POISON] {tag} arm={name} rank={rank} survivors={_n} "
+              f"first_rows={_poison_rows()}", flush=True)
+    elif rank == 0:
+        print(f"[POISON] {tag} arm={name} survivors=0", flush=True)
+    return _n
+
 base_out = torch.zeros((T, H), dtype=torch.bfloat16, device=dev)
 SPIN = int(os.environ.get("K0_SPIN_LIMIT", "2000000"))   # bounded, fail-closed; low enough that a broken acquire fails FAST            # bounded, fail-closed (shared node)
 hbarrier()
@@ -4500,13 +4544,24 @@ if int(os.environ.get("K0_DECODE_EAGER_PHASE_PROFILE", "0")) and T <= 512 and _D
 # ===================== eager correctness on every arm + positive control =====================
 R["eager"] = {}
 _eager_all_local = True
+# exp_32 P0: the [MOK GATE] loop below runs AFTER this loop has finished, so for
+# every candidate arm it re-reads the SAME cand_out bytes -- whichever candidate
+# ran last (_obuf returns the identical object for all of them). Compute each
+# arm's MoK gate HERE, while its own bytes are still live, and have that loop
+# consume the stash. Without this, P1's poison for a given arm is only checked
+# when that arm happens to be last in the rotated order.
+_mok_eager_stash = {}
 for name, body, buf in ARMS:
     if os.environ.get("K0_MPS_TRACE"):
         with open("/out/progress_rank{}.log".format(int(os.environ.get("RANK", "-1"))), "a") as _pf:
             _pf.write("arm " + name + chr(10))
     hbarrier()
-    if name not in _PROD_LIKE: cand_out.zero_()   # FAIRNESS: never zero production's own combine-output view
+    if name not in _PROD_LIKE: _poison_out()   # FAIRNESS: never touch production's own combine-output view
     perr.zero_(); pperr.zero_(); torch.cuda.synchronize(); body(sp()); torch.cuda.synchronize()
+    if name not in _PROD_LIKE:
+        R.setdefault("poison", {})[name] = _poison_report("eager", name)
+    if K0_BENCHMARK_PROTOCOL == "mok_eager":
+        _mok_eager_stash[name] = mok_gate(_obuf(name))
     R["eager"][name] = gate(_obuf(name)); R["eager"][name]["pass"] = ok_arm(name, R["eager"][name])
     R["eager"][name]["pperr"] = int(pperr.cpu().numpy().reshape(-1)[0])   # !=0 => an acquire timed out (diagnostic)
     R["eager"][name]["plan_err"] = int(perr.cpu().numpy().reshape(-1)[0]) if name in _CUSTOM_PLAN_ARMS else 0
@@ -4568,7 +4623,7 @@ R["mok_correctness"] = {}
 _mok_correctness_all_local = True
 if K0_BENCHMARK_PROTOCOL == "mok_eager":
     for name, _, _ in ARMS:
-        _mok_stats = mok_gate(_obuf(name))
+        _mok_stats = _mok_eager_stash[name]   # exp_32 P0: that arm's OWN bytes
         _mok_local = bool(
             _mok_stats["pass"]
             and R["eager"][name]["plan_err"] == 0
@@ -4945,17 +5000,29 @@ if K0_BENCHMARK_PROTOCOL == "mok_eager":
         hbarrier()
         _mps_soak_error = 0
         _mps_soak_completed = 0
+        _mps_soak_poison = 0
+        _mps_soak_poison_epoch = -1
         for _mps_epoch in range(_mps_soak_iters):
+            # exp_32 P2: re-poison before EVERY epoch. The soak is fully untimed
+            # (no HIP events anywhere in this loop), so this turns one
+            # end-of-soak check into 600 independent single-epoch coverage trials.
+            _poison_out()
             _mps_body(sp())
             torch.cuda.synchronize()
             _mps_local_error = int(pperr.item())
+            _mps_local_poison = _poison_count()
             _mps_error_tensor = torch.tensor(
-                _mps_local_error, dtype=torch.int32, device="cuda"
+                [_mps_local_error, _mps_local_poison],
+                dtype=torch.int32, device="cuda",
             )
             dist.all_reduce(_mps_error_tensor, op=dist.ReduceOp.MAX)
-            _mps_soak_error = int(_mps_error_tensor.item())
+            _mps_soak_error = int(_mps_error_tensor[0].item())
+            _mps_soak_poison = int(_mps_error_tensor[1].item())
             _mps_soak_completed = _mps_epoch + 1
-            if _mps_soak_error != 0:
+            if _mps_soak_error != 0 or _mps_soak_poison != 0:
+                if _mps_soak_poison != 0:
+                    _mps_soak_poison_epoch = _mps_epoch
+                    _poison_report("soak", "mps_mega")
                 break
         hbarrier()
         _mps_soak_gate = mok_gate(_obuf("mps_mega"))
@@ -4967,17 +5034,22 @@ if K0_BENCHMARK_PROTOCOL == "mok_eager":
             "requested_epochs": _mps_soak_iters,
             "completed_epochs": _mps_soak_completed,
             "pperr": _mps_soak_error,
+            "poison_survivors": _mps_soak_poison,
+            "poison_first_bad_epoch": _mps_soak_poison_epoch,
             "correctness": _mps_soak_gate,
             "pass": bool(
                 _mps_soak_completed == _mps_soak_iters
                 and _mps_soak_error == 0
+                and _mps_soak_poison == 0
                 and _mps_soak_gate["pass_all_ranks"]
             ),
         }
         if rank == 0:
             print(
                 f"[MPS SOAK] completed={_mps_soak_completed}/{_mps_soak_iters} "
-                f"pperr={_mps_soak_error} pass={R['mps_soak']['pass']}",
+                f"pperr={_mps_soak_error} poison={_mps_soak_poison} "
+                f"poison_epoch={_mps_soak_poison_epoch} "
+                f"pass={R['mps_soak']['pass']}",
                 flush=True,
             )
         # --- exp_05 stage 0 (Distributed-HipKittens overnight): ADDITIVE diagnostic.
@@ -5051,7 +5123,7 @@ if K0_BENCHMARK_PROTOCOL == "mok_eager":
 
     def _mok_measure_arm(name, body):
         if name != "production":
-            cand_out.zero_()
+            _poison_out()
         perr.zero_()
         pperr.zero_()
         torch.cuda.synchronize()
@@ -5077,6 +5149,26 @@ if K0_BENCHMARK_PROTOCOL == "mok_eager":
             dtype=np.float64,
         )
         aligned_us = _mok_rank_max(local_us)
+        # exp_32 P3: the timed loop leaves cand_out holding 600 epochs of
+        # accumulated writes, so a row a LATE epoch failed to write still reads
+        # an earlier epoch's bit-identical correct answer and the gates below
+        # cannot see it. Re-poison and run ONE untimed epoch so they see
+        # single-epoch coverage in the steady state the timed loop just left.
+        # This is strictly after every HIP event has been consumed and after the
+        # rank-max all-reduce above, so it cannot touch a measurement. pperr is
+        # NOT cleared: it is atomicOr-accumulated, so output_gate["pperr"] below
+        # still reports the OR over warmup + timed + this epoch. Skipped when
+        # pperr is already set, because M8 suppresses every batch on a live error
+        # bit and would leave a false poison survivor.
+        if MOK_POISON_OUT and name != "production" and int(pperr.item()) == 0:
+            _poison_out()
+            torch.cuda.synchronize()
+            hbarrier()
+            body(sp())
+            torch.cuda.synchronize()
+            hbarrier()
+            R.setdefault("poison", {})[name + "_post_timing"] = _poison_report(
+                "post_timing", name)
         strict_gate = gate(_obuf(name))
         strict_gate["pass"] = ok_arm(name, strict_gate)
         output_gate = mok_gate(_obuf(name))
 
--- a/benchmarks/mok_synthetic_prefill/run_campaign.sh
+++ b/benchmarks/mok_synthetic_prefill/run_campaign.sh
@@ -131,6 +131,7 @@ for run in $(seq 1 "${run_count}"); do
       -e "K0_MOK_TIMED_ITERS=${K0_MOK_TIMED_ITERS:-100}" \
       -e "K0_MOK_ABSOLUTE_TOLERANCE=${K0_MOK_ABSOLUTE_TOLERANCE:-0.1}" \
       -e "K0_MOK_RELATIVE_TOLERANCE=${K0_MOK_RELATIVE_TOLERANCE:-0.1}" \
+      -e "K0_MOK_POISON_OUT=${K0_MOK_POISON_OUT:-1}" \
       -e "K0_MOK_WEIGHT_CHUNK_ELEMENTS=${K0_MOK_WEIGHT_CHUNK_ELEMENTS:-33554432}" \
       -e "K0_PF5_PRIMITIVE_ONLY=${K0_PF5_PRIMITIVE_ONLY:-0}" \
       -e "K0_PF6GM_G=${K0_PF6GM_G:-3}" \
```

### Two presentation artifacts to expect (both pre-existing, neither is a defect)

1. **A poison-detected soak failure will read `FAIL:nosummary`, not
   `FAIL:soak=False`.** `blocked_pre_timing_mps_soak` is absent from
   `summarize.py`'s `BLOCKED_STATUSES`, so a soak failure makes `summarize.py`
   raise and no `summary.json` is written (`harness_recipe.md` §3). The patch adds
   a new *cause* for that pre-existing path. The log still carries
   `[MPS SOAK] … poison=N poison_epoch=E` and `[POISON] soak … first_rows=[…]`.
   `summarize.py` is outside the "may edit minimally" list, so this is documented
   rather than fixed.
2. **P0 changes the printed `[MOK GATE]` numbers for candidate arms that are not
   last in the rotated order** — from "the last candidate's numbers" to "its own".
   That is a strict improvement, but it *will* look like a change against
   `a11base` and must not be read as a regression. `[MOK GATE] production` is
   unaffected.

---

## 7. Text to append to `MPS_OVERNIGHT_HARNESS_NOTE.md`

```markdown
## exp_32 — NaN poison on `out` (gate hardening, 2026-08-12)

Correctness-only edit to `prefill_opt/host/e004pf_k0pf_ab.py` and
`benchmarks/mok_synthetic_prefill/run_campaign.sh`. No workload, shape,
iteration-count, seed, route or tolerance was changed. `correctness.py`,
`summarize.py` and `synthetic_inputs.py` are untouched.

**Why.** `cand_out` (`ab.py:1799`, 56 MiB, shared by every candidate arm via
`_obuf`) is cleared once per gate episode but never between epochs, and the MoK
synthetic corpus feeds identical input and identical routing every iteration.
A row a candidate fails to write therefore returns the previous epoch's
bit-identical correct answer — invisible to `[MOK GATE]`, to
`combine_bit_exact`, and to all 600 soak epochs. The 600-epoch soak had no
clear at all; the per-arm timed episode cleared once and then ran 600 epochs.

**What changed.**
- `K0_MOK_POISON_OUT` (default `1`, forwarded in the `-e` list). `0` restores
  the previous `zero_()` behaviour exactly.
- Poison value `0x7FD5` = bf16 qNaN payload `0x55`, chosen so a survivor is
  distinguishable from a kernel-produced NaN (`0x7FC0`) or inf (`0x7F80`).
  Detected by `correctness.py`'s existing SUM-all-reduced `nonfinite == 0` gate,
  which is an equality and cannot be widened.
- Poison inserted at three untimed sites: the eager per-arm epoch
  (`ab.py:4508`), before **every** soak epoch (`ab.py:4948`), and at the per-arm
  entry (`ab.py:5054`) plus one extra **untimed** verification epoch inserted
  after `_mok_rank_max` (`ab.py:5079`), i.e. after every HIP event has been read.
- The negative control at `ab.py:4816` deliberately keeps `zero_()`: poisoning
  it would make it fail because of the poison and stop testing the real bug.
- New log lines `[POISON] <tag> arm=<name> survivors=N [first_rows=[...]]`, and
  `[MPS SOAK]` now also prints `poison=` and `poison_epoch=`.
- `R["mps_soak"]["pass"]` additionally requires `poison_survivors == 0`.
- Bundled correctness fix (P0): the `[MOK GATE]` loop at `ab.py:4569` ran after
  the eager loop had finished, so for every candidate arm it re-read the same
  `cand_out` bytes — whichever candidate ran last. Each arm's `mok_gate` is now
  computed inside the eager loop while its own bytes are live. **This changes
  the printed `[MOK GATE]` numbers for candidate arms that are not last in the
  rotated arm order.** `production` is unaffected.

**Cost.** ~20 µs per poison+check; ~12 ms added to a ~4 s soak, ~20 ms per
campaign process in total. Zero device work inside any timed region for any arm.

**Known presentation artifact.** A poison-detected soak failure exits through
the pre-existing `blocked_pre_timing_mps_soak` path, which `summarize.py` does
not list in `BLOCKED_STATUSES`, so it presents as no `summary.json`
(`FAIL:nosummary` in `screen.sh`'s CSV) rather than `FAIL:soak=False`. The
reason is in the log: `[MPS SOAK] … poison=N poison_epoch=E`.
```

---

## 8. Validation plan

### V1 — the one screen that proves the poison is live and harmless

```bash
setsid timeout 1800 env \
  K0_MOK_ARMS=production,pf6gm_mega,mps_mega \
  K0_MOK_WARMUP_ITERS=1 K0_MOK_TIMED_ITERS=1 \
  K0_MOK_OUTPUT_ROOT=$HOME/k0-mok-e32poison \
  K0_MOK_RUN_TIMEOUT=1500 \
  K0_MOK_POISON_OUT=1 \
  K0_MPS_CFG="C=16,g=33,mode=12,flush_rows=16" \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh e32poison 1
```

**It must still PASS**, because mode 12 is believed correct. Accept only if all
of these hold:

| screen | expected |
|---|---|
| `[POISON] eager arm=mps_mega survivors=0` | present, and one for every candidate arm |
| `[POISON] post_timing arm=mps_mega survivors=0` | present |
| `[MPS SOAK] completed=600/600 pperr=0 poison=0 poison_epoch=-1 pass=True` | exact |
| `[MOK GATE] mps_mega max_abs=0.035156 relative=0.008293 pass=True` | **numerically unchanged from `a11base`** |
| `[MARK] control_fails=True` | unchanged |
| `mps_us` | inside the measured band 6,703–6,803 µs, `ratio_vs_prod` 0.8465–0.8592 |

The logic: `survivors=0` proves **coverage**; unchanged gate statistics prove
**harmlessness**; `mps_us` landing in the pre-existing σ = 0.52 % band proves the
poison is **outside the timer**. `[MOK GATE]` numbers for `pf6gm_mega` may move
because of P0 — see §6.

### V2 — the sanity check that matters most: a deliberately broken arm must now fail

**The existing `control_fails=True` control cannot catch a staleness bug, and
never could.** `ab.py:4816-4822` runs the *frozen pull* pipeline and launches
`fn_combine_bug`, a store-over-a-peer bug. It (a) corrupts **values in rows that
are written**, so it would be caught with or without the poison, and (b) never
runs `mps_mega` at all. It proves "the gate can fail on a wrong value" — a
completely different statement from "the gate can fail on a missing value".
It must stay, unchanged; it just does not cover this class.

Two staleness-specific controls, cheapest first:

**V2a — host-only, available immediately, zero kernel edits.** After the eager
epoch and gate for each candidate arm, poison a **single** row and re-run
`mok_gate` into a diagnostic field, asserting it FAILS. This establishes the
detector's sensitivity floor: one row of 4,096 is 7,168 NaN elements, so
`nonfinite = 7168 ≠ 0` and the gate must go red. Four lines, no kernel, no GPU-arm
change, and it proves the detector is wired to a **gate** rather than only to a
print. Behind `K0_MOK_POISON_SELFTEST` (default `1`; it costs one extra
`mok_gate` per arm, ~5 ms):

```diff
     if name not in _PROD_LIKE:
         R.setdefault("poison", {})[name] = _poison_report("eager", name)
+        # exp_32 V2a: prove the detector is wired to a GATE, not just a print.
+        # Poison one row of an otherwise-correct buffer; mok_gate MUST fail.
+        if MOK_POISON_OUT and os.environ.get("K0_MOK_POISON_SELFTEST", "1") != "0":
+            _st_row = cand_out[T - 1].clone()
+            cand_out.view(torch.int16)[T - 1].fill_(MOK_POISON_I16)
+            _st = mok_gate(_obuf(name))
+            cand_out[T - 1].copy_(_st_row)
+            R["poison"][name + "_selftest_fails"] = bool(not _st["pass"])
+            if rank == 0:
+                print(f"[POISON SELFTEST] arm={name} one_row_poisoned_fails="
+                      f"{not _st['pass']} nonfinite={_st['nonfinite']}", flush=True)
```

Expected: `one_row_poisoned_fails=True nonfinite=57344` (7,168 × 8 ranks, since
`nonfinite` is SUM-all-reduced). If that reads `False`, **stop** — the detector is
not connected and every `survivors=0` above is vacuous.

**V2b — the real staleness control, for whoever owns the kernel next.** One line
in M8 under a debug bit of the config word: skip the `out` store for exactly one
token.

```diff
     for (int t = 0; t < NT; ++t) {
       if (!live[t]) continue;
+      if (m8cfg.skip_row_debug && (tok0 + t) == 0) continue;   // exp_32 V2b control
       unsigned short* const orow = out + (size_t)(tok0 + t) * (size_t)K0P6_H;
```

One epoch, poison on. Expected: `[MOK GATE] mps_mega pass=False`,
`nonfinite=57344`, `[POISON] eager arm=mps_mega survivors=7168
first_rows=[0]`. Run the same epoch with `K0_MOK_POISON_OUT=0` and it must
**PASS** — that pair is the complete proof that the poison, and only the poison,
closes the class. Queue behind exp_24; it needs the config word, so it is a
kernel edit and not mine tonight.

### V3 — what a failure looks like, and how to read it

`[MOK GATE] … pass=False` with `nonfinite=N` and
`[POISON] … survivors=N first_rows=[…]`. Read the row list:

| pattern | reading |
|---|---|
| all 4,096 rows | the arm wrote no output at all — a launch/descriptor failure. Check `pperr`, `[MPS TS]`, and that `payload_ok` (`KRN:1727`) is not suppressing every batch. |
| a contiguous run | a coverage hole in the token→batch map. Note `T/NT = 1024` divides exactly, so there is no legitimate remainder — a tail hole is itself the bug. |
| **scattered, and different every run** | a live readiness/coverage race — the bug this patch was built to find. |
| scattered, **identical every run** | a systematic coverage hole (e.g. an off-by-one in the ticket loop), which is easier: it bisects. |

Also check the counterpart: `nonfinite=N` with `survivors=0` is a *numerics* bug
(the kernel wrote a NaN or an inf into a row it did write), not staleness.

---

## 9. The caveat, stated plainly

**If the poison makes a currently-green configuration go red, that is a
discovery, not a regression.** It means a result we already trust was resting on
stale data — the previous epoch's bit-identical correct answer standing in for a
row that a later epoch never wrote.

What to do in that case, in order:

1. **Do not disable the poison. Do not widen a tolerance. Do not clear `pperr`
   and retry.** `nonfinite == 0` is a gate, and a nonzero `pperr` is terminal.
2. Re-run the identical config **twice** and record the survivor row lists.
   Deterministic rows ⇒ a systematic coverage hole; varying rows ⇒ a race. That
   single fact chooses the whole debugging path.
3. Confirm with V2a that the detector is genuinely connected, so the red is not
   an artifact of the poison itself.
4. **Mark every number previously banked for that configuration as unverified**
   in `LESSONS.md` under a `staleness:` tag, with the survivor row list. Do not
   delete the old numbers — supersede them, per the append-only rule.
5. **The ratchet does not stay on an unverified number.** If mode 12
   (`C=16,g=33,flush_rows=16`, 6,685.5 µs, 0.866×) goes red, the ratchet reverts
   to `pf6gm_mega` (6,902 µs, 0.895×) until mode 12 is re-gated with poison on.
   That is the honest consequence of the ratchet rule, and it is cheaper than
   spending the rest of the night optimizing against a number that was never
   real.
6. Nothing else gets timed until it is resolved. Three mechanisms already queued
   for tonight — the M6/M7 interleave (exp_25), the nc-major reorder (exp_30) and
   the pipelined combine (exp_29) — each turn a currently-vacuous readiness edge
   into a live one. exp_25's `a2_done` landmine is the same class on the producer
   side: because epoch *e−1*'s `A2q` bytes are bit-identical to epoch *e*'s, a
   **completely absent** readiness edge would return the right answer *and post
   the best number in the sweep*. That is what this patch exists to prevent, and
   it is worth more than any single mechanism on the queue.

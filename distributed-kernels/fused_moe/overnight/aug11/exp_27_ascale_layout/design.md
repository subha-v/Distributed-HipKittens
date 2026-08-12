# exp_27 design — M6 reads the token-major `sc_stage`; M5's scale transpose is deleted

One variable: **the memory layout M6's `A_scale` argument is read in.** The math,
the pipeline, every barrier, every scale value, the register allocation and the
LDS footprint are held identical, and the gate is **bit-identity**, not tolerance.

Denominator (the ratchet): `C=16,g=353,mode=12,flush_rows=16` =
**6,568.0 ± 4.6 µs, 0.8522× production**, at commit `f34e72fc`.

This file records (§1–3) my verification of the design handed over in
`../exp_28_m6_intensity/design.md`, and (§4–6) what was actually built, which
differs from that design in two places.

---

## 1. The mechanism, verified rather than assumed

M6's per-task prologue fills `ascale_lds[56][96]` (21,504 B) by gathering 96
tokens × 56 FP32 activation scales. Today it reads a **group-major** array,
`A_scale[k·T + token]`:

```391:399:distributed-kernels/fused_moe/n2_phase1_gm_mps.cpp
    // input_scale live prefix is GROUP-MAJOR: input_scale[k128 * T + token].
    for (int idx = tid; idx < kKGroups * kMrows; idx += kThreads) {
      const int k = idx / kMrows;
      const int i = idx % kMrows;
      const int token = tok_lds[i];
      ascale_lds[k][i] = (token < T)
                             ? A_scale[static_cast<std::size_t>(k) * T + token]
                             : 0.0f;
    }
```

`T` here is `nvi[1]`, read on device (`:266`). **`nvi[1]` is the EXTENT, not the
live count** — `hkp_sort.hpp`'s `scan(s_cnt, E, T_loc, scratch, nvi, tid)` ends
with `nvi[0] = acc; nvi[1] = T_loc;`, and the only call site passes `T_ext`
(`k0pf6gm_device_tile_mps.hip:1257`). So the stride is **`T_ext = world·MAXTOK =
32,768`** and one token's 56 scales are `56 × 32,768 × 4 B / 56` = **131,072 B
apart**. Verified on the node, not inferred.

Consequence: 96 × 56 = **5,376 distinct 64 B lines fetched to deliver 21,504 B**
— 16× line amplification, 344 KB per task — and it sits **between two
`__syncthreads()`** (`:356` and `:401`) where no MFMA can cover it.

**The group-major array is manufactured by us, in M5, for a consumer set of
exactly one.** Confirmed by enumeration: across the whole MPS arm
(`k0pf6gm_device_tile_mps.hip`, `moe_mps_adapter.cuh`, `n2_phase1_gm_mps.cpp`,
`n2_phase2_gm_mps.cpp`) `sc_dst` appears only as the descriptor read at `:1241`,
the two M5 writers at `:1319`/`:1325`, and the M6 argument at `:1362`. **Nothing
else reads it.** So the fix is a *deletion*, not a second transpose.

`sc_stage` is already token-major. A row is `56 × 4 = 224 B`, and
`224·t mod 64 ∈ {0, 32}` for every `t`, so a row always spans **exactly 4** lines
(never 5). 96 rows = **384 distinct lines**, 24,576 B delivered for the same
21,504 B payload.

**5,376 / 384 = 14.0× exactly.**

---

## 2. The bit-identity proof

This is the part that makes the experiment unusual: the correctness argument is
not "within tolerance", it is an equality of bit patterns, and it closes without
reference to how `sc_stage` was filled.

**(a) The only writer of `sc_dst` is the transpose.** `hk_moe::mps::scale_transpose_row`
(`moe_mps_adapter.cuh:298-305`) is exactly

```302:304:distributed-kernels/fused_moe/moe_mps_adapter.cuh
    if (lane < ng) {
        sc_dst[(std::size_t)lane * (std::size_t)T_loc + t] = sc_stage_row[lane];
    }
```

called with `sc_stage_row = sc_stage + t·K0P6_NG`, `T_loc = T_ext`,
`ng = K0P6_NG = 56`, over **all** `t ∈ [0, T_ext)` (`:1318-1322`). Therefore

> `sc_dst[k·T_ext + t] ≡ sc_stage[t·56 + k]` for every `t < T_ext`, `k < 56`.

FP32, copied, never arithmetic. The fused fallback
`hkp::zero_part_scale_transpose<14>` has the identical transpose half
(`hkp_quant.hpp`, read on the node), so the identity holds on both M5 branches.

**(b) M6's address is the left-hand side of that identity.** M6 reads
`A_scale[k·T + token]` with `T = nvi[1] = T_ext`, so it reads
`sc_dst[k·T_ext + token] = sc_stage[token·56 + k]` — which is precisely what
arm 1 loads directly. **The load returns the same bit pattern.** No new
assumption about `k0p5_target` or the unpack is needed; the equality is internal
to the kernel.

**(c) Same predicate, same zeros.** `(token < T)` is unchanged and `tok_lds` is
unchanged, so pad sub-block rows still get exact `+0.0f`. Note `T` is no longer
part of any *address* in arm 1, only of the mask — a latent-bug class removed for
free.

**(d) Same consumption order.** `mfma_k` reads `ascale_lds[k][row]` as two
`float4`s (`:429-431`), untouched, so no FP reassociation is introduced.

**(e) No new memory-safety surface.** Arm 1's address set is
`{sc_stage[t·56 + k] : t < T_ext, k < 56}` — *exactly* the set M5's transpose
already reads every epoch in the shipping kernel. Nothing new is dereferenced.
The host allocates `sc_stage` as `(T_LOC_MAX, NG)` float32 (`e004pf_k0pf_ab.py:1746`),
i.e. token-major with a 56-float row stride, `T_LOC_MAX = 40960 ≥ T_ext = 32768`.

**(f) Alignment for the `float4` load.** Byte offset is `224·token + 16·c`;
`224 = 14·16`, so 16 B alignment holds for every `(token, c)` given a 16 B-aligned
base, and the base is a fresh torch allocation (≥256 B).

**(g) Ordering is strictly stronger, not weaker.** `sc_dst` was published by the
M5 grid barrier alone. `sc_stage` is written by M2's unpack and published by the
M3, M4 **and both M5** barriers. No new edge, no new fence, no new primitive.

**(h) Hole rows.** M2 skips hole rows, so `sc_stage` hole rows are
uninitialized — but so are `sc_dst`'s today, because the transpose copies those
same uninitialized bytes. Neither is ever read (`tok_lds` is filled from
`sorted_ids` or the sentinel `T`, and the gather masks on `token < T`).
**Unchanged hazard surface.**

⇒ `A2q`, `DQ2`, `slots` and `out` must be **bit-identical**. The gate is
therefore: every `[MOK GATE]` digit must match the ratchet's
`max_abs=0.035156 relative=0.008293` exactly.

---

## 3. Where exp_28's design was wrong, and what I changed

| exp_28 said | reality |
|---|---|
| `:395` still includes the donor `n2_phase1_gm.cpp`; switching it is exp_27's first commit and needs its own control run | **Already landed** (exp_26). `:409` includes `n2_phase1_gm_mps.cpp` and `SRC_REV` is 25. exp_27 is purely additive — no include switch, no extra control. |
| stride is `T_ext = 32,768`, asserted | **True but for a non-obvious reason** — it is `nvi[1]`, and `nvi[1] = T_loc` only because `scan()`'s caller passes `T_ext`. Verified in `hkp_sort.hpp` on the node before building. |
| resource budget `scratch 144 B` | **128 B** — exp_24 improved it. Gated against 128, not 144. |
| bump `SRC_REV` 24 → 25 | 25 was already taken by exp_26; bumped **25 → 26**. |
| fast branch: delete the loop, leave `if/else` | Written as `#if K0P6_MPS_ASCALE_TM` → `if (!skip_dead_part_zero(cfg4)) { fallback }`, which avoids an empty `if` body while keeping the fallback branch bit-identical in both arms. |

Everything else in exp_28's design verified as written.

---

## 4. What was built

Four sites, two files. Nothing else is touched — no host edit, no
`moe_host_abi.hpp` edit, no descriptor-length change, no new buffer, no barrier
added or removed, no change to `n2_phase2_gm_mps.cpp`, `moe_mps_adapter.cuh`
(`scale_transpose_row` stays: the `#else` arm and the M5 fallback still call it)
or the read-only parity port.

| # | file | site | change |
|---|---|---|---|
| 1 | `n2_phase1_gm_mps.cpp` | `:186-205` | **MPS-DELTA (9)**: `N2GM_P1_ASCALE_TOKEN_MAJOR`, default 0, with the equivalence argument in the comment and a `static_assert` |
| 2 | `n2_phase1_gm_mps.cpp` | `:358-400` | the gather becomes `#if` / `#else`; the `#else` arm is the donor's eight lines **verbatim** |
| 3 | `k0pf6gm_device_tile_mps.hip` | `:104-120` | `SRC_REV` 25 → 26; `K0P6_MPS_ASCALE_TM` shipping literal (`#ifndef`-guarded so an out-of-tree genco build can force an arm) |
| 4 | `k0pf6gm_device_tile_mps.hip` | `:409-412` | `#define N2GM_P1_ASCALE_TOKEN_MAJOR K0P6_MPS_ASCALE_TM` around the phase-1 include, `#undef` after |
| 5 | `k0pf6gm_device_tile_mps.hip` | `:1341-1366` | M5: at TM=1 the fast branch's transpose loop is gone; the fallback branch is untouched in both arms |
| 6 | `k0pf6gm_device_tile_mps.hip` | `:1379-1391` | M6 is handed descriptor slot 9 (`SC_STAGE`) instead of slot 20 (`SC_DST`) |

`sc_dst` (slot 20) stays allocated and stays in the descriptor. Reclaiming its
7.34 MiB is a separate host-side change and deliberately does not ride along.

### The three gather choices, and why

- **`float4`, not `float`.** Cuts 5,376 scalar loads to 1,344 vector loads over
  the same 384 lines, and the trip count from 21 to 6.
- **Row-fast (`i = idx % kMrows`), not quad-fast.** Row-fast keeps the LDS writes
  conflict-free — consecutive lanes write consecutive `ascale_lds` dwords. The
  alternative strides LDS writes by `4·kMrows = 384` dwords, a multiple of 32, so
  every lane of a quad lands in one bank. The cost is that the 64 line-requests
  per instruction cannot be coalesced, but the 384 distinct lines are only
  24,576 B, so the repeats are near-certain L1 hits.
- **The predicate stays `token < T`**, so pad rows still write exact `0.0f`.

### The A/B mechanism

The arm is a **literal in the hashed `.hip`**, flipped by a commit, never a `-D`.
mori's JIT cache key hashes `.hip`/`.cpp` *content* and not compile flags, so a
flag flip silently reuses the previous hsaco. Two commits, one per arm, adjacent
in history, and every batch verifies the literal in the node checkout before it
starts.

---

## 5. Gates

Build/ISA gates and their results are in `build.md`; correctness and timing in
`result.md`. Pre-registered here:

| gate | requirement |
|---|---|
| G1 | arm 0 `.text` **byte-identical** to the ratchet — the knob cannot regress the denominator |
| G2 | arm 1 `AGPR 256 / ArchVGPR 256 / LDS 155,496` exact, `scratch ≤ 128 B` |
| G3 | MFMA census `96 + 84 = 180`, unchanged |
| G4 | zero scratch ops inside either MFMA K-loop span |
| G5 | gather ISA: a `dwordx4` load, and loop trip count **6 not 21** |
| G6 | fresh `.hsaco` mtime per arm (`stat -L`, file not directory) |
| G7 | no live `sc_dst` store on the arm-1 fast path |
| **BI** | every `[MOK GATE]` digit identical to the control: `max_abs=0.035156 relative=0.008293` |
| gates | `pperr == 0`, `control_fails=True`, `[MPS SOAK] 600/600 poison=0`, poison self-test firing at `nonfinite=57344` |

### Pre-registered falsifiers (inherited from exp_28 §5, unchanged)

| observation | conclusion |
|---|---|
| `ΔM6 ∈ [−190, −60] µs` | **confirmed**; promote |
| `ΔM6 ∈ (−60, +40) µs` | **the mechanism and the class are dead.** 907 MB of deleted line traffic bought nothing ⇒ M6's prologue is not on its critical path, and this retires the "exposed between two barriers" argument wherever it appears |
| `ΔM6 < −250 µs` | over-delivered ⇒ the gather was latency- not traffic-dominated; re-open the exposed-latency family |
| any `[MOK GATE]` digit differs | **STOP.** Bit-identity is provable; a difference means the two arrays are not equivalent. Do not widen a tolerance — find the layout error. |
| `pperr ≠ 0` | terminal; full world-8 protocol-state reinitialization before any retry |

### Measurement design

The M6 device stamp is the instrument (σ = 4.8 µs, measured in exp_32 at n=10; a
−130 µs effect is ~27σ). End-to-end 1-proc screens have a 6.6 % tail and cannot
see this. Because the arm is a compile-time literal, control and candidate
**cannot** be interleaved inside one batch, so the design is four batches,
alternating, `n = 5` each: **control, candidate, control, candidate**. Two
independent batches per cell, control first and last, and arm is not aliased with
time. A campaign follows only if the stamp result clears.

---

## 6. Rejected alternatives

| alternative | why not |
|---|---|
| Keep `sc_dst` but write it token-major in M5 | Strictly worse than deleting the transpose: same M6 win, but keeps 7.34 MiB of stores and a redundant buffer. `sc_stage` **is** the token-major array. |
| Pad the row stride to 64 floats so a row is 4 *aligned* lines | Rows already touch exactly 4 lines because `224·t mod 64 ∈ {0,32}`. Buys nothing; needs a new buffer and a host change. |
| Hoist the gather to once per tile | The 8 chunk-tasks of a tile land on 8 different XCDs (`g = bid mod 8`), so there is no shared L2 to hoist into. |
| Shrink `ascale_lds` to a 2-deep k-window | Re-reads each token's lines 3.5× *inside* the K-loop, moving scale traffic onto the far more expensive A path. Wrong trade. |
| Free `sc_dst`'s 7.34 MiB in the same arm | Two variables, and it is a host-side change. |
| Flip the arm with `-D` | Silently reuses the previous hsaco. The whole reason the literal lives in the hashed `.hip`. |

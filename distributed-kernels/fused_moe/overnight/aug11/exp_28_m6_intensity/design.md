# exp_28 design — O4: M6 reads the token-major `sc_stage`; delete the M5 scale transpose

Build design for the **one** option `analysis.md` recommends first. One variable:
**the memory layout M6's `A_scale` argument is read in.** Everything else — the
math, the pipeline, every barrier, every scale value, the register allocation and
the LDS footprint — is held byte-identical, and the output must be
**bit-identical**, which is the gate.

Predicted: **ΔM6 = −130 … −190 µs** (point estimate −140), plus **≈ −17 µs** in
the plan phase. Ratchet effect at 6,683 µs: **0.866× → ≈ 0.844×**.

---

## 1. The mechanism in one paragraph

M6's per-task prologue gathers 96×56 FP32 activation scales into `ascale_lds`
from a **group-major** array (`A_scale[k·T_ext + token]`, `T_ext = 32,768`), so
one token's 56 scales are 131,072 B apart: **5,376 distinct 64 B lines fetched
to deliver 21,504 B — 16× amplification, 344 KB per task, 977 MB per epoch —
sitting between two `__syncthreads()` where no MFMA can cover it.** The
group-major array is manufactured by *us*, in M5, from an array that is already
token-major, for a consumer set of exactly one: M6. Point M6 at the token-major
original and the gather touches **4 lines per token = 384 lines = 24,576 B, a
14.0× cut**, and M5's transpose becomes dead and is deleted.

---

## 2. Exact files, sites and edits

Ownership: all four files are in the exp_28 owner set (`aug10/CLAUDE.md`
"Ownership"). Nothing outside `distributed-kernels/fused_moe/` is touched. The
donor `n2_phase1_gm.cpp`, `k0pf6gm_device_tile.hip` (parity port) and
`hkp_quant.hpp` are **not** touched.

### 2.1 `n2_phase1_gm_mps.cpp` — the gather (the only compute-side edit)

This is the vendored MPS copy exp_26 created. **Pre-req: the include at
`k0pf6gm_device_tile_mps.hip:395` must already point at
`"n2_phase1_gm_mps.cpp"`** — as of this writing it still reads
`#include "n2_phase1_gm.cpp"`, i.e. the donor. If exp_26 has landed, that switch
is already done and this experiment is purely additive; if not, switching the
include is exp_28's first commit and must be its own bit-identical control run
(`N2GM_P1_SCHED_GSCALE=0`, all four hooks empty ⇒ textually equivalent to the
donor, per the file's own header §29-32).

**(a) New knob, next to the two existing ones (`:118-131`), as MPS-DELTA (9):**

```cpp
// MPS-DELTA (9): activation-scale source layout.
//   0 (default) = donor: GROUP-major  A_scale[k * T + token], T = T_ext = 32768.
//   1           = TOKEN-major         A_scale[token * kKGroups + k].
// The two arrays hold the same values by construction — M5's
// hk_moe::mps::scale_transpose_row writes sc_dst[k*T_ext+t] = sc_stage[t*56+k]
// (moe_mps_adapter.cuh:303) — so this switch is bit-identical by definition and
// only changes which of the two the loads come from.
#ifndef N2GM_P1_ASCALE_TOKEN_MAJOR
#define N2GM_P1_ASCALE_TOKEN_MAJOR 0
#endif
static_assert(N2GM_P1_ASCALE_TOKEN_MAJOR == 0 ||
              N2GM_P1_ASCALE_TOKEN_MAJOR == 1);
```

**(b) Replace the gather at `:265-273`** (donor `:186-194`) with an `#if`, whose
`#else` arm is the donor's eight lines verbatim:

```cpp
#if N2GM_P1_ASCALE_TOKEN_MAJOR
    // TOKEN-MAJOR: one row is kKGroups*4 = 224 B contiguous. 224*t mod 64 is
    // always 0 or 32, so every row spans EXACTLY 4 lines (never 5): 96 rows =
    // 384 distinct lines vs the group-major path's 96*56 = 5376.
    static_assert(kKGroups % 4 == 0, "float4 scale quads need kKGroups % 4 == 0");
    constexpr int kScaleQuads = kKGroups / 4;            // 14, no tail
    for (int idx = tid; idx < kScaleQuads * kMrows; idx += kThreads) {
      const int i = idx % kMrows;   // row-fast: consecutive lanes -> consecutive
      const int c = idx / kMrows;   //           LDS dwords, zero bank conflict
      const int token = tok_lds[i];
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (token < T) {
        v = *reinterpret_cast<const float4*>(
            A_scale + static_cast<std::size_t>(token) * kKGroups + 4 * c);
      }
      ascale_lds[4 * c + 0][i] = v.x;
      ascale_lds[4 * c + 1][i] = v.y;
      ascale_lds[4 * c + 2][i] = v.z;
      ascale_lds[4 * c + 3][i] = v.w;
    }
#else
    // input_scale live prefix is GROUP-MAJOR: input_scale[k128 * T + token].
    for (int idx = tid; idx < kKGroups * kMrows; idx += kThreads) {
      ... donor's six lines, unchanged ...
    }
#endif
```

Three deliberate choices, each with its arithmetic:

- **`float4`, not `float`.** 16 B alignment holds: the byte offset is
  `224·token + 16·c`, and `224 = 14·16`. It cuts the request count from 5,376
  scalar loads to 1,344 vector loads over the same 384 lines.
- **Row-fast (`i = idx % kMrows`), not quad-fast.** Row-fast keeps the LDS
  writes conflict-free (consecutive lanes write consecutive dwords of one
  `ascale_lds` row) at the price of 64 line-requests per instruction that the
  coalescer cannot merge; the 384 distinct lines are only 24,576 B, so the
  repeats are near-certain L1 hits and, at absolute worst, 1,344 L2 requests —
  still 4× below today's 5,376. **Pre-registered contingency:** if the ISA gate
  or the screen shows the gather still dominating, swap to quad-fast
  (`c = idx % kScaleQuads; i = idx / kScaleQuads`), which merges 14 lanes into
  4 line-requests but costs a 32-way LDS write conflict worth ≈ +6 µs/epoch by
  the bank arithmetic in `analysis.md` §O9. Do not do both in one arm.
- **The predicate stays `token < T`**, unchanged, so pad rows still write exact
  `0.0f` (`analysis.md` §6.4). Note `T` is no longer part of any *address* in
  this arm — a latent-bug class removed for free.

`ascale_lds` keeps its declaration and its 21,504 B (`:193`). No other line of
the body changes.

### 2.2 `k0pf6gm_device_tile_mps.hip` — the pointer and the deletion

| site | today | change |
|---|---|---|
| `:104` | `#define K0P6_MPS_SRC_REV 24` *(verified in the working tree at the time of writing)* | **bump to the next unused value** — 25 unless a concurrent experiment took it first. Mandatory: this is the only hashed file, and `.cuh`/vendored-`.cpp` edits do not invalidate the mori JIT cache (`:96-104`). |
| new, near `:390` | — | `#define N2GM_P1_ASCALE_TOKEN_MAJOR K0P6_MPS_ASCALE_TM` before the phase-1 `#include`, `#undef` after it, with `#ifndef K0P6_MPS_ASCALE_TM / #define … 0` at the top. One compile knob, default off. |
| `:395` | `#include "n2_phase1_gm.cpp"` | `#include "n2_phase1_gm_mps.cpp"` (exp_26's switch; see §2.1) |
| `:1347` | `(const float*)k0p6_dread(desc, K0P6_D_SC_DST)` | `#if K0P6_MPS_ASCALE_TM` → `K0P6_D_SC_STAGE` `#else` → `K0P6_D_SC_DST`. Descriptor slot 9 vs 20; both already exist, **no ABI change**. |
| `:1302-1307` | fast branch: `for (t…) hk_moe::mps::scale_transpose_row(...)` | `#if K0P6_MPS_ASCALE_TM` the loop is **deleted** (nothing else is in it — exp_24 already removed the `part` zero from this branch), `#else` unchanged. |
| `:1308-1313` | fallback branch: `hkp::zero_part_scale_transpose<14>` | **untouched in both arms.** It keeps writing a now-unread `sc_dst`; that is harmless and keeps this a one-variable change. |

`sc_dst` (slot 20) stays allocated and stays in the descriptor. Reclaiming its
7.34 MiB is a separate, later, host-side change and must not ride along.

### 2.3 Nothing else

No host-driver edit. No `moe_host_abi.hpp` edit. No descriptor-length change. No
new buffer. No barrier added or removed. No change to `n2_phase2_gm_mps.cpp`,
`moe_mps_adapter.cuh` (`scale_transpose_row` stays, unused in the fast arm —
still used by the `#else` arm and by the fallback branch), or the parity port.

---

## 3. Correctness argument — this is a numerics argument, and it is bit-identity

M6 has no cross-CTA dependency, no readiness poll, no peer traffic and no
epoch-lifetime state (`m6_m7_structure.md` §5.6), so there is no protocol to
review. What must be argued is that the *values* are identical.

**1. The two arrays hold the same values, by construction.** The only writer of
`sc_dst` is `hk_moe::mps::scale_transpose_row`
(`moe_mps_adapter.cuh:298-305`), whose body is exactly

```
sc_dst[(size_t)lane * T_loc + t] = sc_stage_row[lane],  lane < ng = 56
```

called with `sc_stage_row = sc_stage + t·K0P6_NG`, `T_loc = T_ext`,
`ng = K0P6_NG = 56` (`k0pf6gm_device_tile_mps.hip:1304-1306`). Therefore
`sc_dst[k·T_ext + t] ≡ sc_stage[t·56 + k]` for every `(t, k)` with `k < 56`.
FP32, copied, never arithmetic. **The load returns the same bit pattern.**

**2. Same predicate, same zeros.** The `(token < T) ? … : 0.0f` guard is
unchanged, and `tok_lds` is unchanged, so pad sub-block rows still get exact
`+0.0f` and live rows still get their scale.

**3. Same consumption order.** `mfma_k` reads `ascale_lds[k][row]` as two
`float4`s (`:362-365`) and the FP32 accumulation order in the K-loop is
untouched, so no reassociation is introduced.

⇒ **`A2q`, `DQ2`, `part`/`slots`, and `out` must be bit-identical to the base
arm on the same seed.** Not "within tolerance" — identical.

**Must be bit-identical (the gate):** every rank's `[MOK GATE]` `max_abs` and
`relative` must print the *same digits* as the base arm (base reference:
`max_abs 0.035156`, `relative 0.008293`); `pperr = 0`; `plan_err = 0`;
`nonfinite = 0`; `control_fails = True`.
**May legitimately differ:** timings, the HSACO sha256 (source changed — expected
and benign, exp_65 recorded the same), and `scratch`/`SGPR` by a few bytes if the
allocator reshuffles (see §4 for the tolerance).

**4. Publication / ordering.** `sc_dst` is published by the M5 grid barrier
(`:1327`). `sc_stage` is written by M2's unpack (`:1160-1161`) and is therefore
published by the M3 (`:1236`), M4 (`:1291`) **and** both M5 barriers — *strictly
more* ordering than the pointer it replaces. **No new edge, no new fence, no new
primitive.**

**5. Hole rows.** M2 skips hole rows (`off >= s_ns[s] → continue`, `:1159`), so
`sc_stage` hole rows are uninitialized — but so are the `sc_dst` hole rows today,
because the transpose copies those same uninitialized bytes. Neither is ever
read: `tok_lds` is filled from `sorted_ids` (live receive rows only) or the
sentinel `T` (`:257-259`), and the gather masks on `token < T`. **Unchanged
hazard surface.**

**6. Pre-build check (5 minutes, read-only, do it first).** Confirm in
`hkp`/`hkp_quant.hpp` on the node that `k0p6_unpack_row_nopoll` writes
`sc_stage` with a **56-float row stride** and that the host allocation of slot 9
is ≥16 B aligned and covers `T_ext` rows. The in-tree evidence already says both
(`:1304-1306` and `:1310-1312` read it that way, and `validate_descriptor`
sizes it), but a `float4` load makes the alignment load-bearing. **If the stride
is not 56 or the base is not 16 B aligned, stop** — fall back to four scalar
loads, which still delivers the 14× line cut and only loses the request
coalescing.

---

## 4. Build and ISA gates (all must pass before any timing)

Command: the `BUILDING.md:85-96` genco invocation plus
`-DK0P6_MPS_ASCALE_TM=1`, with `-Rpass-analysis=kernel-resource-usage`.

| gate | requirement | why |
|---|---|---|
| G1 default-arm protection | build with `K0P6_MPS_ASCALE_TM` **unset** and require the resource tuple to be **byte-identical** to today's `SGPR 106 / VGPR 256 / AGPR 256 / scratch 144 B / LDS 155,496 B`, and the `#else` arm to be the donor text | proves the knob cannot regress the ratchet |
| G2 candidate resources | `AGPR = 256`, `ArchVGPR = 256`, `LDS = 155,496 B` **exactly** (the gather does not touch either), `scratch ≤ 144 B`, `VGPR spills ≤ 9` | the gather is a prologue; any register movement means the compiler hoisted something into the K-loop |
| G3 MFMA census | `96 + 84 = 180` static `v_mfma`, unchanged (`PROVENANCE.md:110`) | the K-loops are untouched |
| G4 scratch placement | zero scratch ops inside either MFMA K-loop span | the standing budget |
| G5 gather ISA | in the M6 prologue between the two `s_barrier`s: **`global_load_dwordx4`/`buffer_load_dwordx4` present, ≤ 6 per lane**, and the loop trip count 6 not 21 | direct evidence the 14× landed |
| G6 JIT freshness | new `.hsaco` **mtime** under `~/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5/*/` (`stat -L`, file not directory) | `SRC_REV` 24→25 must have taken |
| G7 dead-store check | `sc_dst` no longer appears as a store target in M5's fast branch | the transpose really is gone |

---

## 5. Screen ladder

One GPU job at a time; `rocm-smi --showpids` is the check that works
(`aug11/STATUS.md`). Every step is `setsid timeout`-wrapped.

| # | run | config | pass condition |
|---:|---|---|---|
| 0 | **control**: base arm screen, 1 proc / 1 warmup / 1 timed | `K0P6_MPS_ASCALE_TM` unset, `C=16,g=33,mode=12,flush_rows=16` | reproduces the ratchet at 6,703–6,728 µs; record `M6 = M6_DONE − M5_DONE` and `plan = M5_DONE − M2_DONE` with `timestamps` on |
| 1 | **candidate** screen, same shape | `K0P6_MPS_ASCALE_TM=1`, same cfg | `[MOK GATE]` digits **identical** to step 0 on all 8 ranks; `pperr=0`; `control_fails=True`; `ΔM6 ≤ −60 µs` |
| 2 | repeat 1 twice more (screen σ is 0.52 % on the ratio, so one sample cannot resolve −140 µs on 6,683) | — | median `ΔM6` and median `Δratio` both negative; spread < 2× the point estimate |
| 3 | **600-epoch soak** (`K0_MPS_SOAK_ITERS=600` exactly — any other value raises) | candidate | `[MPS SOAK]` clean, all ranks `pperr=0` |
| 4 | **campaign**, `production,pf6gm_mega,mps_mega`, 500 warmup / 100 timed, 5 rotated processes | candidate | median rank-max p50 `ratio_vs_prod ≤ 0.850`; new ratchet if so |

Steps 0–2 are the decision; 3–4 are the promotion. Order per the standing rule
is correctness + negative control + soak **before** timing — steps 1 and 3
satisfy that, and step 1's timing is a screen, not a promotion number.

### Pre-registered falsifiers

| observation | conclusion |
|---|---|
| `ΔM6 ∈ [−190, −60] µs` | **confirmed.** Promote; the exposed-prologue class is real and `analysis.md` §2's marginal byte rate (146 µs/GB) is validated on a second, independent mechanism. |
| `ΔM6 ∈ (−60, +40) µs` | **the mechanism is dead and so is the class.** 907 MB of deleted line traffic bought nothing ⇒ M6's prologue is not on its critical path and the 12.1 B/clk/CU marginal rate does not apply to it. Log it: it also retires the "exposed between two barriers" argument wherever it appears, and it means **only** the (walled) intensity axis remains for M6. |
| `ΔM6 < −250 µs` | over-delivered ⇒ the gather was latency-dominated, not traffic-dominated. Immediately re-price O5-class prologue work and re-open the exposed-latency family. |
| any `[MOK GATE]` digit differs from step 0 | **STOP.** Bit-identity is provable (§3); a difference means the two arrays are *not* equivalent, i.e. `sc_stage`'s stride or extent is not what §3.6 assumed. Do not widen a tolerance; find the layout error. |
| `pperr ≠ 0` | terminal. Full world-8 protocol-state reinitialization before any retry (never clear-and-retry). |

---

## 6. Build-hours estimate (honest)

| step | h |
|---|---:|
| pre-build layout check (`hkp` stride + alignment, read-only) | 0.3 |
| `n2_phase1_gm_mps.cpp` gather + knob | 0.7 |
| `k0pf6gm_device_tile_mps.hip` (pointer, macro, `SRC_REV`, M5 deletion) | 0.7 |
| build + gates G1–G7, both arms | 1.0 |
| screens 0–2 (3 × ~90 s plus reading stamps) | 0.8 *(mostly unattended)* |
| 600-epoch soak | 1.0 *(unattended)* |
| campaign + `summarize.py` + `result.md` + `LESSONS.md` | 1.5 |
| **total** | **6.0** (≈ 3.5 attended) |

Add 1.0 h if exp_26 has not landed and exp_28 must switch the `:395` include and
run its own bit-identical control for that switch.

Risks: **(i)** the `hkp` unpack stride is not 56 → fall back to scalar loads
(cost: the coalescing, not the mechanism); **(ii)** exp_24 has not landed, so the
fast branch at `:1302` does not exist yet — then the transpose deletion must be
written against whichever fork is in the file, and the M6 pointer change alone
still delivers the full M6 win (the M5 −17 µs is the part that waits);
**(iii)** exp_27 is already building this — then this file is exp_27's design and
exp_28 should instead spend its slot on `analysis.md` §O2's `rocprofv3` PMC read,
which settles `m6_m7_structure.md` §7 item 3 without a kernel edit.

---

## 7. Rejected alternatives (for the record)

| alternative | why not |
|---|---|
| Keep `sc_dst`, but write it **token-major** in M5 | Strictly worse than deleting the transpose: same M6 win, but it keeps 7.34 MiB of stores and a redundant buffer. `sc_stage` **is** the token-major array. |
| Pad the token-major row stride to 64 floats (256 B) so a row is 4 aligned lines | Rows already touch exactly 4 lines because `224·t mod 64 ∈ {0,32}` — padding buys nothing and would need a new buffer and a host change. |
| Hoist the gather out of the 8 chunk-tasks of a tile (gather once per tile) | Needs cross-task state that only exists if the task loop is restructured tile-major; the 8 chunk-tasks of a tile are on **8 different XCDs** (`g = bid mod 8`), so there is no shared L2 to hoist into. Blocked by the same mapping `analysis.md` §4 analyses. |
| Shrink `ascale_lds` to a 2-deep k-window (frees 20,736 B of LDS) | Re-reads each token's lines 3.5× *inside* the K-loop, i.e. it moves scale traffic onto the A path that `analysis.md` §2.1 measures at 4.9× the cost per byte, and it needs the K-loop's instruction mix re-hinted. Wrong trade, and the LDS it frees has no buyer (see `analysis.md` §O6). |
| Do O4 and O6 (`a2_lds ∪ ascale_lds`) in one arm | Two variables. O6 is worth 0 µs on its own and exists only as currency; it must be its own bit-identical control run. |

---

## 8. Primitives

Per the kernel-design mandate, what this experiment says about
`include/cdna4/ops/group/distributed/`:

- **Nothing new is needed and nothing is open-coded.** O4 adds no protocol: no
  release, no acquire, no counter, no epoch, no slot lifetime. The ordering it
  relies on is the existing `hkp::grid_barrier` chain, and it *removes* a data
  dependency rather than adding one. That is worth recording as the shape of a
  clean experiment: the primitive library is untouched because there is no
  protocol in the change.
- **The one primitive-level finding is a mis-factoring in a helper we already
  extracted.** `hkp::zero_part_scale_transpose<Chunks1K>`
  (`hkp_quant.hpp:130-143`, read-only to us) fuses two unrelated jobs — a
  `part`-row zero and a scale transpose. exp_24 already had to split it, landing
  `hk_moe::mps::scale_transpose_row` in `moe_mps_adapter.cuh:298-305` and noting
  "in hindsight the primitive should have been two composable pieces". exp_28
  supplies the second half of that lesson: **the transpose piece should never
  have existed.** It converts a token-major array into a group-major array for a
  single consumer that reads it 14× less efficiently in group-major order. The
  durable rule for the library: *a layout-conversion primitive is a defect until
  someone has named the consumer that requires the target layout and shown the
  line-traffic arithmetic for both layouts.* `scale_transpose_row` will be left
  in place (the `#else` arm and the fallback branch still call it) and marked
  with that caveat in its comment when this lands.
- **Negative finding for `LESSONS.md` under `primitives:`** — the gather at
  `n2_phase1_gm_mps.cpp:265-273` is a strided-gather-into-LDS, a shape that
  recurs (this one, `b1s_lds`, `dq2_lds`, `tok_lds`). There is no primitive for
  it, so each site open-codes its own index decomposition and each one gets its
  bank behaviour and its line amplification right or wrong on its own. A
  `group::gather_rows_to_lds<Rows, Cols>(dst_lds, src, row_index_lds, stride)`
  that owns the "row-fast for LDS banks vs field-fast for coalescing" trade
  would have made this experiment a one-line call-site change instead of a
  hand-verified loop rewrite. Proposed, not built — it needs a second and third
  caller before the shape is trustworthy.

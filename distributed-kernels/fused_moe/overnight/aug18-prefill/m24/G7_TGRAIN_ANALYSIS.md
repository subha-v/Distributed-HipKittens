# G7 — T-generality of `hkp::csr_scan_block256` and `hkp::pull_src_fill`

**Work-list item:** `FILL_AWARE_DESIGN.md` rev 3, G7 (line 1206); §B.3 (line 527-543); §F.4 (line 1119-1137).
**Question:** for what values of the M24-substituted row count `T_eff` are the two helper
primitives correct, and what is the minimum safe `K0P6_M24_TGRAIN`?

**Answer up front: `TGRAIN = 1`.** Both primitives are correct for every `T ∈ [0, T_cap]`
with no alignment requirement whatsoever. `T_eff = 0` is safe at all five substitution
sites. The `TGRAIN = 256` default is pure de-risking and can be dropped to 1 (8 is
recommended only as a cosmetic margin, see §6). §F.4's **unclamped** branch applies.

---

## 0. Provenance — the headers ARE local (prior claim of "node-only" is wrong)

Both primitives live in the local `amd-master` checkout, in a single read-only header:

* `/Users/subha/repos/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip/hkp/hkp_sort.hpp`
  * `block_incl_scan_256` — **:186-199**
  * `csr_scan_warp64` — **:201-214** (not used by M15; `T <= 64` variant)
  * `csr_scan_block256` — **:217-232**
  * `pull_src_fill` — **:238-257**
* Compile smoke test: `.../hkp/test_hkp_compile.cpp:131` (scan) and `:143` (fill).
* Header contract block: `hkp_sort.hpp:1-18`, notably **:14** — *"count/scan run on ONE CTA
  with the full block; pad/scatter run grid-stride."*
* Variant-selection comment **:178-183** already states the answer explicitly:
  `warp64 … requires T <= 64`; `block256 … 256-wide LDS block scan with serial carry over
  256-token chunks, **any T**`.

The M15 kernel includes it by name at
`distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip:53` (`#include "hkp_sort.hpp"`);
there is no second copy of the header inside the Distributed-HipKittens tree (verified by
`find . -name hkp_sort.hpp` → no hits), so the node build resolves to this same file via
`-I`. No node access was required for G7.

Throughout this document `KERNEL:` = `k0pf6gm_device_tile_m15.hip` at **current** (post-M24)
line numbers. The design doc's `943 / 977 / 1375 / 1587 / 1891` are *pre-patch* numbers; the
mapping (from `m24_impl.diff` hunk headers) is:

| design (old) | current | site |
|---|---|---|
| 943 | **1416** | site 5 — M0.5 ADAPTIVE pre-histogram (`T_pre`) |
| 977 | **1453** | site 1 — M1 dispatch |
| 1375 | **1897** | site 2 — `csr_scan_block256` (:1946) + `pull_src_fill` (:2008) |
| 1587 | **2128** | site 3 — M7 pool front-half sweep |
| 1891 | **2433** | site 4 — M8 combine |

---

## 1. `csr_scan_block256` — correct for any `T >= 0`; requires `blockDim.x == 256`

```c
// hkp_sort.hpp:217-232
__device__ __forceinline__ void csr_scan_block256(const int* pull_cnt, int* pull_ptr,
                                                  int T, int tid, int* s_tmp, int* s_carry) {
  if (tid == 0) *s_carry = 0;
  __syncthreads();
  for (int base = 0; base < T; base += 256) {
    const int tau = base + tid;
    const int f   = (tau < T) ? pull_cnt[tau] : 0;     // :224  <-- tail guard
    const int incl = block_incl_scan_256(f, s_tmp);
    if (tau < T) pull_ptr[tau] = *s_carry + (incl - f); // :226  <-- tail guard
    __syncthreads();
    if (tid == 255) *s_carry += incl;                   // :228
    __syncthreads();
  }
  if (tid == 0) pull_ptr[T] = *s_carry;                 // :231  <-- sentinel at index T
}
```

### 1.1 Proof sketch: any `T >= 0`

* **Ragged tail is explicitly handled.** `:224` substitutes `f = 0` for `tau >= T`, and
  `:226` suppresses the store. A partial final chunk therefore contributes exactly the
  valid entries.
* **The carry is still correct on a partial chunk.** `block_incl_scan_256` (`:186-199`) is a
  full-block Hillis-Steele inclusive scan over all 256 lanes; lane 255's `incl` is the sum
  over the whole 256-slot window, and the out-of-range slots hold `f = 0`. So
  `*s_carry += incl` at `:228` adds precisely `Σ pull_cnt[tau]` over
  `[base, min(base+256, T))`. **No multiple-of-256 assumption exists anywhere.**
* **`T = 0`.** The chunk loop body never executes; `*s_carry` stays 0; `:231` writes
  `pull_ptr[0] = 0`. Well-defined, no OOB, no stale read.
* **`__syncthreads` convergence.** `T` is a grid-uniform scalar (§4), the loop trip count is
  therefore CTA-uniform, and the whole construct sits inside the CTA-uniform branch
  `else if (bid == 1)` at `KERNEL:1943-1946`. No divergent barrier for any `T`.
* **Loop-carried LDS hazards** are covered by the `__syncthreads()` pairs at `:227` and
  `:229` plus the trailing barrier inside `block_incl_scan_256:197`.

### 1.2 The one real precondition — `blockDim.x` must be exactly 256

`block_incl_scan_256` indexes `s_tmp[threadIdx.x]` on a caller LDS array declared `int[256]`
(`KERNEL:1944`) and shifts to `off = 128`; `csr_scan_block256:228` reads the carry out of
lane **255** specifically. So:

* `blockDim.x > 256` → LDS overrun in `block_incl_scan_256:188`.
* `blockDim.x < 256` → lane 255 does not exist, `*s_carry` is never advanced, and every
  chunk after the first is scanned with carry 0 → silently wrong CSR.

M15 satisfies this unconditionally: `KERNEL:1193` is
`__global__ void __launch_bounds__(256, 1)`, and the body's warp arithmetic
(`blockDim.x >> 6`, `__shared__ s_row[4][7488]`) is written for 4 waves. **This is a
`blockDim` precondition, not a `T` precondition — M24 does not touch it.**

### 1.3 Upstream trip counts at the call site — none are hardcoded

Grepping the kernel for literal `4096`, `/ 256`, `>> 8`, `% 256` returns only comments and
one unrelated 256-byte weight-segment computation (`KERNEL:1060`). Every loop that consumes
the substituted count derives its trip count from `T` (`KERNEL:1469`, `:1946`, `:2008`,
`:2141`, `:2441`, `:2478`). There is **no** `T/256 == 16` or `T/NT == 1024` constant in the
code path; the `1,024 batches` figure at `KERNEL:2125` is a comment about tuning, not a trip
count (the actual value is `(T + NT - 1) / NT` at `:2141`).

---

## 2. `pull_src_fill` — correct for any `T >= 0`; grid-stride, no chunking at all

```c
// hkp_sort.hpp:238-257
for (int tau = blockIdx.x*blockDim.x + threadIdx.x; tau < T;
     tau += gridDim.x*blockDim.x) {              // :241-242
  int k = pull_ptr[tau];                          // :243
  for (int s = 0; s < TOPK; ++s) {
    const int pe = pull_stage[(tau*TOPK + s)*2 + 0];
    if (pe < 0) continue;
    const int row = pull_stage[(tau*TOPK + s)*2 + 1];
    if (k >= cap) { atomicOr(pperr, ovf_bit); break; }   // :248-251
    pull_src[k*2 + 0] = pe; pull_src[k*2 + 1] = row; ++k;
  }
}
```

* **Shape assumptions: zero.** It is a plain grid-stride loop over `tau < T`. There is no
  chunking, no tiling, no "final chunk" concept, and no barrier inside — so "partial final
  chunk" does not arise. Any `T` works; per-thread work is naturally ragged.
* **Empty fill region (`T_eff == 0`).** Loop guard fails on the first evaluation for every
  thread → **zero iterations, zero stores, no `pperr` write, no OOB**. `pull_src` retains the
  previous epoch's bytes, which is harmless because the only consumer
  (`k0p6_m15_m8_batch`, §3.4) indexes `pull_src` strictly below `pull_ptr[T] = 0`.
* **It reads `pull_ptr[tau]` only for `tau < T`** — never the sentinel — so it is consistent
  with a scan run at the same `T`. Its correctness is *relative to* the scan: the two calls
  must use the **same** `T`. At `KERNEL:1897` a single `const int T` feeds both `:1946` and
  `:2008`, and both execute between the same pair of grid barriers (M3 at `:1914`, M4 at
  `:1978`, M5 at `:2010`), so this is structurally guaranteed within a CTA. Cross-CTA
  agreement is §4.
* **The `cap` argument shrinks safely.** `KERNEL:2008` passes `cap = T * world`. Per-token
  fanout is `__popcll(pmask)` over the primary-dedup mask (`KERNEL:1497-1499`), i.e. the
  count of *distinct destination ranks*, hence `fanout <= world`. Therefore
  `Σ_τ fanout(τ) <= T * world = cap` for any `T`, so the shrunken cap is still a sound
  (and tight) bound and cannot false-trigger `pperr` bit 1048576. Since `T_eff <= T_cap`,
  it is also always within the host allocation sized for `T_cap * world`.
* **`pull_cnt` staleness is contained.** M1 writes `pull_cnt[tau]` only for `tau < T_eff`
  (`KERNEL:1520`, inside `for (tau = gw; tau < T; …)` at `:1469`). Entries in
  `[T_eff, T_cap)` are last epoch's. The scan reads only `tau < T` — never those entries.
  Consistent by construction, again *provided all sites agree on `T`* (§4).

---

## 3. Interaction with the five M24 substitution sites

`k0p6_m24_teff` (`KERNEL:749-768`) returns `clamp(tail.agreed_T_eff, 0, T_cap)` — the
consumer-side clamp at `:763` guarantees the value handed to every primitive is in
`[0, T_cap]`, so the analysis above covers the whole reachable domain.

### 3.1 Site 5 — M0.5 ADAPTIVE pre-histogram (`KERNEL:1416`)
`for (i = gtid; i < T_pre*TOPK; i += gstride)` at `:1428`. Grid-stride, any `T_pre >= 0`.
`T_pre = 0` → histogram all-zero → no expert passes `hcnt_pre[e] >= theta` (`:1440`) → the
decision word is 0 → every expert takes the ownership map, i.e. degenerates to plain M15.
Both grid barriers (`:1426`, `:1432`) are still executed by every CTA. **Safe, any value.**
(ADAPTIVE arm only; inert in the default M15 build.)

### 3.2 Site 1 — M1 dispatch (`KERNEL:1453`)
`for (tau = gw; tau < T; tau += nw)` at `:1469` — one *warp* per token, any `T >= 0`.
`T = 0` → no rows pushed; `pushed_count` stays 0; the epoch publication and the per-chunk
`chunk_ready` fills at `:1643-1651` still run for all 8 chunks with `fill = 0` (the clamp at
`:1645-1647` handles `n_s = 0` exactly). Crucially, the receive-side chunk geometry is driven
by `MAXTOK` / `K0P6_CHUNK = 512` / `K0P6C_NCHUNK_MAX = 8` (`KERNEL:1235`, `:1876-1878`) —
**not** by `T` — so M24 cannot reintroduce the exp_04/exp_36 phantom-chunk bug. **Safe.**

### 3.3 Site 3 — M7 pool front-half sweep (`KERNEL:2128`)
`nbatches = (T + NT - 1)/NT` with `NT = 4` (`:2140-2141`), quota-bounded ticket loop
(`:2154-2167`). `T = 0` → `nbatches = 0` → the first `fetch_add` returns `0 >= 0` → immediate
`break`. One `m8_front` ticket is burned; M8 (§3.4) also computes `nbatches = 0` and breaks,
so the burned ticket is inconsequential. Non-multiple-of-4 `T` produces one partial batch,
handled inside `k0p6_m15_m8_batch` (§3.4). **Safe, any value.**

### 3.4 Site 4 — M8 combine (`KERNEL:2433`) and its consumer `k0p6_m15_m8_batch`
This is the only consumer with real per-row structure, and it is already ragged-safe:

```c
// KERNEL:1103-1109
for (int t = 0; t < NT; ++t) {
  const int tok = tok0 + t;
  live[t]   = (tok < T);
  lo2[t]    = live[t] ? pull_ptr[tok] : 0;
  fanout[t] = live[t] ? (pull_ptr[tok + 1] - lo2[t]) : 0;
}
```

* Partial final batch (`T % 4 != 0`) → `live[t] = false` for the overhang; `fanout = 0`;
  the out-of-fanout lanes still get a dereferenceable `pbase` by design
  (`KERNEL:1116-1124`) and their loaded bytes are discarded by the `jb + jj < fanout[t]`
  guard at `:1160`. **No 4-alignment requirement.**
* **`pull_ptr[tok + 1]` at `tok = T-1` reads index `T` — exactly the sentinel written by
  `csr_scan_block256:231`.** This is the single hard coupling between the two primitives and
  the M24 sites: the scan's `T` and M8's `T` must be the same integer. Both are
  `k0p6_m24_teff` (`:1897`, `:2433`), so this reduces entirely to §4.
* `T = 0` → `nbatches = 0` (`:2441`), both ticket loops (`:2449-2460`, `:2461-2472`) break on
  the first draw, and the staged-arm static stripes (`:2478`, `:2482`) run zero iterations.
  Then `K0P6_M24_ZERO_PAD` (`:2489-2514`) zeroes `out[0, T_cap)` in full. **Safe.**

### 3.5 `T_ext` is untouched and must stay untouched
`T_ext = world * MAXTOK` (`KERNEL:1315`) is the *receive* extent and drives
`k0p6_sort::scan/pad/scatter` (`:1919`, `:1982`, `:2002`), `zero_part_scale_transpose`
(`:1955`/`:1963`/`:1969`) and the `row_ready` cell index (`:2113`). M24 correctly substitutes
**none** of these — the receive side shrinks organically via `pushed_count`/`chunk_ready`.
Confirmed against `m24_impl.diff`: the only `const int T = …` lines changed are the four at
old 977 / 1375 / 1587 / 1891 plus the new `T_pre`.

### 3.6 Other helpers fed by the five sites — audit
| Site | Consumers of the substituted value | T-shape assumption |
|---|---|---|
| 5 | `atomicAdd` histogram loop `:1428` | none (grid-stride) |
| 1 | warp loop `:1469`; `pull_stage`/`pull_cnt` writes `:1517-1520`; `store_dispatch_row` `:1591` (indexed by `drow_id`, not `tau`) | none |
| 2 | `hkp::csr_scan_block256` `:1946`; `hkp::pull_src_fill` `:2008` (`cap = T*world`) | none beyond `blockDim==256` |
| 3 | `nbatches` `:2141`; `k0p6_m15_m8_batch` `:2164` | none (`live` guard) |
| 4 | `nbatches` `:2441`; `k0p6_m15_m8_batch` `:2458`/`:2470`/`:2479`/`:2483`; `ZERO_PAD` sweep `:2503-2512` (grid-strided `uint4`, `T*896` .. `T_cap*896`) | none |

`k0p6_sort::count_reduce / scan / cursors_sei / pad / scatter` and `n2p6m15_phase1_body` are
driven by `EL`, `T_ext`, `PADMAX` and `num_tiles` — **none** consume `T_eff`. The `ZERO_PAD`
sweep's `kV = K0P6_H/8 = 896` `uint4` per row means the byte offset `T * 896 * 16` is exact
for any `T` (no alignment requirement, since `H = 7168` is itself a multiple of 8).

---

## 4. The residual precondition that matters: grid-uniformity of `T_eff`, not its grain

Neither primitive constrains `T`'s *value*. What they do constrain is that **CTA 1's scan
`T` and every other CTA's M8 `T` be identical**, because the CSR sentinel is written at index
`T` and read at index `T`. If CTA 1 scanned at `T_a` and a combine CTA ran at `T_b > T_a`,
token `T_b - 1` would read `pull_ptr[T_b]` — a *stale* word from the previous epoch — and
`fanout` would be an arbitrary integer. Consequences: `fanout > 8` walks `pull_src` past the
filled region and `jb + jj < fanout[t]` admits garbage into the fp32 accumulator; `fanout < 0`
simply drops the row. **Silent wrong output, not a hang or a `pperr` bit.**

This is exactly the failure the release fence at `KERNEL:713-727` was added to prevent (rev 3
review fix R.1), and the consumer clamp at `:760-763` prevents the `0xFFFFFFFF` variant. G7's
finding is that this fence is **correctness-load-bearing for the CSR sentinel specifically**,
which the design doc does not currently say — §B.3 motivates it only by "divergent T across
the grid" in general terms.

**Recommended cheap belt-and-braces (independent of TGRAIN):** clamp the fanout in
`k0p6_m15_m8_batch:1108` to `[0, world]` (or raise a distinct `pperr` bit when it falls
outside). One `v_med3`-class instruction per token in a loop that already does 14 chunk
iterations of 8-way fp32 accumulation; it converts the entire class from *silent wrong
output* to *bounded / detected*. Cost is far below the build gate's noise floor and it should
be measured, not asserted.

---

## 5. Verdict

| Primitive | Safe domain of the substituted count | Binding precondition |
|---|---|---|
| `hkp::csr_scan_block256` | **every `T ∈ [0, T_cap]`** | `blockDim.x == 256` (satisfied unconditionally by `__launch_bounds__(256,1)`, `KERNEL:1193`); the caller's `s_tmp` is `int[256]` (`KERNEL:1944`) |
| `hkp::pull_src_fill` | **every `T ∈ [0, T_cap]`** | must be called with the *same* `T` as the scan that produced `pull_ptr`; `cap >= Σ fanout`, satisfied by `cap = T*world` |

**Minimum safe `K0P6_M24_TGRAIN` = 1.** Not 256, not 64, not a multiple of the wavefront.
Nothing in either primitive, in `k0p6_m15_m8_batch`, or in the `ZERO_PAD` sweep requires
`T_eff` to be aligned to anything. **`T_eff = 0` is safe at all five sites** (§3.1-3.4) —
provided `K0P6_M24_ZERO_PAD = 1`, which the compile-time guard at `KERNEL:272-273` already
makes mandatory whenever `FILL = 1`.

**§F.4's N=0 ladder point resolves to the UNCLAMPED branch** (design line 1129, `< 300 µs`,
ratio `< 0.04`). No `TGRAIN`-minimum clamp is needed and the line-1130 clamped branch
(600-900 µs) should be struck. §H.4's reserved right to clamp `T_eff` to a minimum of
`TGRAIN` can be released.

---

## 6. Findings that change the M24 gate plan

1. **Drop `TGRAIN` to 1 (or 8), and re-derive the ladder's cost model.** G7's stated
   contingency was "if T-general, drop TGRAIN to 8". 8 is safe; so is 1. Recommendation:
   ship **8** as the default. Not because 1 is unsafe — it is provably safe — but because 8
   costs at most 7 phantom rows out of ~1,539 (< 0.5 %, inside the measurement noise) while
   preserving a non-zero alignment margin against any *future* consumer added to the five
   sites, and it keeps `T_eff*896*16` a 128-byte multiple for the `ZERO_PAD` sweep's store
   coalescing. The 256→8 change also invalidates §B.3's `E[round_up(n,256)]/E[n]` cost term
   and G0b's TGRAIN column: at grain 8 that ratio is ≈ 1.00, so **G0b's TGRAIN sub-question
   is now moot** and §A.4's "raw fill" table becomes the operative one, not the optimistic
   bound. `ρ = 0.494 central` at the TGRAIN-256 fill of 0.438 (design:193-194) should be
   recomputed at the raw fill of 0.376.
2. **`ZERO_PAD` cost is anti-correlated with N and does NOT cancel out of the φ fit.** At
   `T_eff = N` the sweep writes `(T_cap − N) × 7168 × 2 B` — 58.7 MB per layer at `N = 0`,
   zero at `N = T_cap`. Every low-N ladder point therefore carries the *maximum* pad-zeroing
   cost, and an affine fit of step time vs N will absorb that as a spurious negative slope,
   biasing φ. **Gate-plan change:** either (a) add a `ZERO_PAD`-only calibration arm (build
   with `FILL=1, NULLWORK=1, NORIG_CONST=0` and time the sweep alone), or (b) fit with the
   known `(T_cap − N)·H·2` byte term held as a *fixed* regressor rather than a free
   parameter. Without this, arm B's φ interval is not interpretable, and the N=0 point
   measures "irreducible fixed cost **+ a full-tensor zero sweep**", i.e. it is an
   **upper bound** on the fixed cost, not the fixed cost. This compounds with the already-
   recorded §F.4 invalidation pending arm E's C/flush_rows re-sweep.
3. **The release fence at `KERNEL:713-727` is correctness-load-bearing for the CSR
   sentinel**, not merely for "divergent T" in the abstract (§4). Add it to the build gate as
   a named assertion, and add the `fanout ∈ [0, world]` clamp in `k0p6_m15_m8_batch:1108` as
   a distinct, measurable defensive change so the failure mode is detected rather than
   silent.
4. **G7 needed no node access.** The design doc's G7 row ("Read … in `~/amd-master`, node,
   read-only") should be corrected to `laptop` — the header is in the local checkout at
   `auto-gpu-kernel/k0_fused_moe/solution/hip/hkp/hkp_sort.hpp`. G7's done-when ("resolve
   §F.4 to one branch *before* the arm runs") is satisfied by this document.
5. **No change to `blockDim`, `T_ext`, `MAXTOK`, `K0P6_CHUNK` or `PADMAX` is implied.** The
   only precondition either primitive imposes (`blockDim.x == 256`) is already structural in
   M15 and is unrelated to fill-awareness. No new gate item there.

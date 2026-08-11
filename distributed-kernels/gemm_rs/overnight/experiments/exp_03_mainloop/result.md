# exp_03 — E1: giving the GEMM mainloop real global-load/MFMA overlap

**Verdict: E1(b) WINS. Geomean 269.96 → 256.09 µs (−5.14%), full ladder passed
at both tolerances, no spilling reintroduced. The winning source state is the
current working tree** (`gemm_rs_mi300x.cpp` sha256 `a2078f18…04cb4ab`,
`gemm_rs_mi300x_hk_adapter.cuh` sha256 `e3cc3b17…c583a180`); full unified diff
against `c0a6bdd2` in `arms/baseline_to_e1b.diff`.

Denominator throughout: the current best measured tonight on this node,
per-shape means µs `85.93 / 113.79 / 96.92 / 201.82 / 769.04 / 2632.09`,
**geomean 269.96 µs**.

| arm | 1 | 2 | 3 | 4 | 5 | 6 | geomean | vs 269.96 | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 85.93 | 113.79 | 96.92 | 201.82 | 769.04 | 2632.09 | **269.96** | — | — |
| **P3** | 87.34 | 116.11 | 95.60 | 201.28 | 764.95 | 2602.52 | **270.12** | 1.0006 (+0.06%) | NEUTRAL, kept (it is what pays for E1(b)) |
| **E1(b)** | 77.31 | 104.93 | 91.68 | 200.49 | 750.55 | 2520.29 | **256.09** | 0.9486 (**−5.14%**) | **WIN** |
| E1(b) confirm | 77.50 | 104.53 | 91.34 | 201.44 | 742.33 | 2519.33 | **255.59** | 0.9468 (−5.32%) | reproduces to 0.2% |

Per-shape deltas of E1(b) vs baseline: **−10.0% / −7.8% / −5.4% / −0.7% /
−2.4% / −4.3%**. Every one except shape 4 is many standard deviations outside
run-to-run spread. Shape 4 (4096×4096×4096) is the shape with the fewest
k-iterations per tile (`k_local = 512`, 16 iterations), so it has the least
mainloop mass to recover — its ~0 delta is consistent with the mechanism rather
than a contradiction of it.

---

## The mechanism

Per k-iteration the shipped mainloop drained the memory system **four times**:
`global_load_dwordx4` ×2 → `vmcnt(0)` → `ds_write` ×4 → `lgkmcnt(0)` →
`global_load_dwordx4` ×2 → `vmcnt(0)` → `ds_write` ×4 → `lgkmcnt(0)`, and only
then the 64 MFMAs. The LDS double buffering was real but bought no latency
overlap: the "prefetch" of block k+1 was a synchronous copy, issued and drained
before control could reach the arithmetic, and at `LDS_BYTES = 65536` on the
256-row configs residency is 1 CTA/CU = 2 waves/SIMD, so there is no second wave
to cover the stall either.

E1(b) splits `kittens::load(ST&, const GL&, COORD)` at its `vmcnt(0)` into
`load_issue` (global loads, no wait) and `load_commit` (`vmcnt(0)`, ds_writes,
`lgkmcnt(0)`), and hoists the issue above the MFMA block while sinking the
commit below it. Both halves live in `gemm_rs_mi300x_hk_adapter.cuh`, which we
own; `include/cdna3/**` is untouched.

**The ISA confirms the reordering held — the compiler did not defeat it.**
`<256,256,32,false>` k-loop, program order after the change:

```
%bb.87    4 x global_load_dwordx4                      ISSUE, no waitcnt
.LBB3_88  12 ds_read_b64, lgkmcnt(0), 4 mfma,
          7 x [1 ds_read_b64 + 4 mfma], 5 ds_read_b64,
          lgkmcnt(0), 32 mfma                          64 MFMAs, no vmcnt at all
%bb.89    vmcnt(0), 4 ds_write_b64, lgkmcnt(0),
          vmcnt(0), 4 ds_write_b64, lgkmcnt(0)         COMMIT, after the MFMAs
.LBB3_90  s_barrier
```

Two things improved at once: the `vmcnt(0)` moved from before the MFMAs to
after them, and all **4** loads are now in flight together instead of 2, drain,
2, drain. So the iteration went from two fully exposed global round trips to one
round trip covered by 64 MFMAs.

Also worth recording: the scheduler interleaves half 1's `ds_read`s with half 0's
MFMAs for free (the `1 ds_read + 4 mfma` runs above), which is the intra-iteration
pipelining E1(c) was supposed to arrange by hand. `sched_group_barrier` has less
left to do here than the plan assumed.

---

## Arm P3 — split the k-step into two BK/2 halves

**Diff** (`gemm_rs_mi300x.cpp`, producer block):

```cpp
+ constexpr int KH = 2;
+ constexpr int KS = BK / KH;
+ static_assert(KS % 16 == 0 && KS * KH == BK);
- rt_bf<WM, BK, ducks::rt_layout::row> A_frag;
- rt_bf<WN, BK, ducks::rt_layout::row> B_frag;
+ rt_bf<WM, KS, ducks::rt_layout::row> A_frag;
+ rt_bf<WN, KS, ducks::rt_layout::row> B_frag;
```
```cpp
  for (int k = 0; k < k_iters; ++k) {
-     load(A_frag, subtile_inplace<WM, BK>(As[k & 1], {warp_row, 0}));
-     load(B_frag, subtile_inplace<WN, BK>(Bs[k & 1], {warp_col, 0}));
      if (k + 1 < k_iters) {
          G::load(As[(k + 1) & 1], g.a, {0, 0, tm, k + 1});
          G::load(Bs[(k + 1) & 1], g.b, {0, 0, tn, k + 1});
      }
-     if constexpr (K_TAIL) { if (k == k_iters - 1) mask_a_k_tail(A_frag, tail_k); }
-     mma_ABt(C_accum, A_frag, B_frag, C_accum);
+     #pragma unroll
+     for (int kh = 0; kh < KH; ++kh) {
+         load(A_frag, subtile_inplace<WM, KS>(As[k & 1], {warp_row, kh}));
+         load(B_frag, subtile_inplace<WN, KS>(Bs[k & 1], {warp_col, kh}));
+         m3::acquire_frags(A_frag, B_frag);            // lgkmcnt(0) + anchor
+         if constexpr (K_TAIL) {
+             if (k == k_iters - 1) {
+                 const int th = tail_k - kh * KS;
+                 mask_a_k_tail(A_frag, th > 0 ? th : 0);
+             }
+         }
+         mma_ABt(C_accum, A_frag, B_frag, C_accum);
+     }
      __syncthreads();
  }
```

**Numerics are bit-exact.** `mma_ABt` (`include/cdna3/ops/warp/register/tile/mma.cuh:260-281`)
issues `d = a[n][0]·b[m][0] + c` and then chains `d = a[n][k]·b[m][k] + d` for
k ascending, into the same accumulator element, and the kernel passes `C_accum`
as both `c` and `d`. Splitting the k range therefore produces the identical
sequence of additions in the identical order; each base tile is still one whole
`v_mfma_f32_16x16x16bf16_1k`. Observed `max|diff|` is unchanged from the
baseline at 4.883e-04 / 1.221e-04.

**The K-tail rebase.** `mask_a_k_tail` assumes the fragment spans K columns
`[0, BK)`; under the split half `kh` spans `[kh·KS, (kh+1)·KS)`, so the
threshold becomes `tail_k − kh·KS`, clamped at 0 (a half entirely past the tail
is zeroed outright, since every `kc >= 0`). Affects config rows 3 and 6 only:
- row 3, `2048×2880×2880`: `k_local=360`, `k_iters=12`, `tail_k=8` → half 0
  threshold 8, half 1 threshold −8 → 0, whole half zeroed.
- row 6, `8192×8192×29568`: `k_local=3696`, `k_iters=116`, `tail_k=16` → half 0
  threshold 16 (nothing zeroed; its columns are all valid), half 1 → 0.
Both pass at the tight 2e-3 with the same `max|diff|` as the baseline.

**P3's resource table** (production TU; `-Rpass-analysis=kernel-resource-usage`
plus the amdhsa YAML, which agree):

| instantiation | VGPR | AGPR | SGPR | scratch B | VGPR spill | waves/SIMD |
|---|---:|---:|---:|---:|---:|---:|
| `<32,64,64,false>` | 89 | 0 | 106 | 0 | 0 | 5 |
| `<64,64,64,false>` | 91 | 0 | 106 | 0 | 0 | 5 |
| `<128,256,32,true>` | 153 | 0 | 106 | 0 | 0 | 3 |
| `<256,256,32,false>` | **233** | 0 | 106 | **0** | **0** | 2 |
| `<256,256,32,true>` | **233** | 0 | 106 | **0** | **0** | 2 |
| `<32,64,64,true>` | 89 | 0 | 106 | 0 | 0 | 5 |

Against the baseline (`logs/p0_FINDINGS.md` Q4; its `<256,256,32,*>`,
`<128,256,32,true>`, `<64,64,64,false>` and `<32,64,64,*>` rows are unaffected by
exp_04's row-1 retile) of `91 / 93 / 170 / 256 / 256 / 91` with **12 B scratch
and 2 VGPR spills on both 256-row instantiations**:

**P3 cleared the 12 B scratch and both spills, and freed 23 VGPRs on the
256-row configs (256 → 233) and 17 on `<128,256,32,true>` (170 → 153).**
Predicted ~24 and ~16. Occupancy is unchanged at 2 waves/SIMD, as
pre-registered: residency is LDS-bound at 1 CTA/CU, not register-bound, so
freeing registers cannot buy occupancy here — it buys headroom for E1(b), which
is the only reason to do it.

**P3's own timing is a wash: 270.12 vs 269.96 µs, +0.06%.** Shape 6 −1.1%,
shape 3 −1.4%, shapes 1 and 2 +1.6% and +2.0%. Kept anyway, because E1(b) needs
the 23 registers and E1(b) is worth 5%. Reported as neutral, not as a win.

### P3 gate ladder

| gate | verdict |
|---|---|
| M0 node clean | PASS, 0 KFD pids |
| M1 build | PASS, all three modules |
| M2 resources/ISA | PASS, 6/6 instantiations present, 0 scratch, 0 spill |
| M3 correctness 17 shapes @1e-2 **and** @2e-3 | **PASS 17/17 both tolerances**, worst 4.883e-04 |
| M4 negative controls | PASS, all three fail as designed |
| M5 600-epoch skewed soak | PASS, epoch and signal cells exact, worst 4.883e-04 |
| M7 timing 3×50 rotated | 270.12 µs geomean |

---

## Arm E1(b) — split issue from commit

**Diff** (on top of P3). Adapter additions: `g2s_plan` / `g2s_slots`,
`wait_vmcnt0`, `wait_lgkmcnt0`, `frag_anchor`, `acquire_frags`, `load_issue`,
`load_commit`. Kernel:

```cpp
+ using ST_A = st_bf<BM, BK>;
+ using ST_B = st_bf<BN, BK>;
+ constexpr int NT = G::GROUP_THREADS;
+ float4 abuf[m3::g2s_slots<ST_A, NT>];
+ float4 bbuf[m3::g2s_slots<ST_B, NT>];
...
- G::load(As[0], g.a, {0, 0, tm, 0});
- G::load(Bs[0], g.b, {0, 0, tn, 0});
+ m3::load_issue<ST_A, NT>(abuf, g.a, {0, 0, tm, 0});
+ m3::load_issue<ST_B, NT>(bbuf, g.b, {0, 0, tn, 0});
+ m3::load_commit<NT>(As[0], abuf);
+ m3::load_commit<NT>(Bs[0], bbuf);
  __syncthreads();
  for (int k = 0; k < k_iters; ++k) {
+     const bool more = (k + 1 < k_iters);
-     if (k + 1 < k_iters) {
-         G::load(As[(k + 1) & 1], g.a, {0, 0, tm, k + 1});
-         G::load(Bs[(k + 1) & 1], g.b, {0, 0, tn, k + 1});
-     }
+     if (more) {                                    // ISSUE, no waitcnt
+         m3::load_issue<ST_A, NT>(abuf, g.a, {0, 0, tm, k + 1});
+         m3::load_issue<ST_B, NT>(bbuf, g.b, {0, 0, tn, k + 1});
+     }
      #pragma unroll
      for (int kh = 0; kh < KH; ++kh) { ...MFMAs... }
+     if (more) {                                    // COMMIT, after the MFMAs
+         m3::load_commit<NT>(As[(k + 1) & 1], abuf);
+         m3::load_commit<NT>(Bs[(k + 1) & 1], bbuf);
+     }
      __syncthreads();
  }
```

**Buffer lifetime, verified not weakened.** In iteration k the kernel ds_reads
`As[k&1]` and ds_writes `As[(k+1)&1]`. Since `(k+1)&1 == (k-1)&1`, the buffer
written is the one read in iteration k−1, and the `__syncthreads()` ending
iteration k−1 separates them. That barrier is genuinely sufficient in both
directions because the drains bracket it: iteration k−1's ds_reads are drained
by `acquire_frags`' `lgkmcnt(0)` (once per half) *before* the barrier, and
iteration k−1's ds_writes are drained by `load_commit`'s trailing `lgkmcnt(0)`,
also before it. This is the same invariant the fused version relied on. The
`load_issue` of iteration k targets registers, not LDS, so hoisting it above the
barrier-protected region changes nothing about LDS ordering; and issue is always
paired with commit inside the same iteration, so no `global_load` is ever left in
flight across the tile epilogue.

**Only full `vmcnt(0)` / `lgkmcnt(0)` waits are used.** No counted wait was
introduced anywhere; the win comes entirely from where the wait sits. The
`gemm_rs_device_tile.cpp:786-812` hazard (a hand-counted `lgkmcnt(N)` silently
invalidated by one scheduler-hoisted `s_load`, 30× error through a 2e-2 gate)
therefore does not apply to this arm.

**Deliberate deviation from the donor helper, with its equivalence argument.**
`kittens::load` iterates `j` over `small_calls = 16` inside a `big_calls` loop
and relies on dead-code elimination of the slots whose `row < rows` predicate is
unsatisfiable. `load_issue`/`load_commit` iterate to `total_calls` instead,
which is *exactly* the set of satisfiable slots: `row < rows` ⟺
`j·N_THREADS + laneid < rows·memcpy_per_row` ⟺ `j < total_calls`. The predicate
is kept, so no slot that could load is dropped, and the staging buffer's
register footprint no longer depends on the optimizer folding `threadIdx` range
information. `big_calls == 1` is `static_assert`ed rather than assumed, because a
split buffer cannot express the donor's reuse of `buf[]` across successive big
calls; it holds for every instantiation in the dispatch table. All other index
arithmetic (`elem_per_memcpy`, `elem_per_half_memcpy`, `memcpy_per_row`,
`total_calls`, `small_calls`, the `unit_coord`/`src_ptr` derivation, the
`dst.idx(dst_ptr, {row, col})` addressing, `N_THREADS = G::GROUP_THREADS = 512`)
is copied unchanged.

**E1(b)'s resource table:**

| instantiation | VGPR | AGPR | SGPR | scratch B | VGPR spill | waves/SIMD | vs P3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `<32,64,64,false>` | 91 | 0 | 106 | 0 | 0 | 5 | +2 |
| `<64,64,64,false>` | 91 | 0 | 106 | 0 | 0 | 5 | 0 |
| `<128,256,32,true>` | 165 | 0 | 106 | 0 | 0 | 3 | +12 |
| `<256,256,32,false>` | 246 | 0 | 106 | **0** | **0** | 2 | +13 |
| `<256,256,32,true>` | 248 | 0 | 106 | **0** | **0** | 2 | +15 |
| `<32,64,64,true>` | 91 | 0 | 106 | 0 | 0 | 5 | +2 |

**E1(b) did NOT reintroduce the 12 B scratch or any spill.** The staging buffers
cost +13/+15 VGPRs on the 256-row configs, landing at 246/248 against the 256
arch-VGPR cap — inside the 23 registers P3 freed, with 8-10 to spare. This is
exactly why the two arms had to run in this order: E1(b) on the unsplit
fragments would have needed 269/271 registers and would have spilled the staging
buffer, which is not merely slow but **wrong** — a spill store of a `float4`
whose `global_load` has not landed writes stale register contents, and no
`vmcnt` protects it because the load is inside an `asm volatile` the waitcnt pass
cannot see. `arms/e1b/lds_race.txt` asserts 0 scratch ops in either k-loop.

### E1(b) gate ladder

| gate | verdict |
|---|---|
| M0 node clean | PASS, 0 KFD pids |
| M1 build | PASS, all three modules |
| M2 resources/ISA | PASS, 6/6 instantiations, 0 scratch, 0 spill, `vmcnt(0)` after the MFMAs |
| M3 correctness 17 shapes @1e-2 **and** @2e-3 | **PASS 17/17 both tolerances**, worst 4.883e-04 (identical to baseline) |
| M4 negative controls | PASS, all three fail as designed (reroute: `max diff` 1.065e-02 vs golden 6.104e-05, detected at both tolerances) |
| M5 600-epoch skewed soak | PASS, epoch and signal cells exact, worst 4.883e-04 |
| M7 timing 3×50 rotated | **256.09 µs geomean, 0.9486× the 269.96 best** |
| M7 confirmation (independent repeat) | 255.59 µs, reproduces to 0.2% |

---

## The finding that nearly shipped a silent wrong answer

**A bare `s_waitcnt lgkmcnt(0)` does not protect a register filled from LDS, and
the failure is silent and shape-dependent.** This cost one M3 run and is the
most transferable thing in this experiment.

The first P3 build placed `wait_lgkmcnt0()` between the fragment `ds_read`s and
the MFMAs, exactly as designed. It failed M3 on **14 of 17 shapes** with
`max|diff|` of 1e19 to 9e30 and NaN, ~99.9% of elements wrong, with **no error
bit set** — the protocol was healthy, the arithmetic was garbage.

Cause: `s_waitcnt` has no operands, so it carries no data dependence. The
`ds_read` that fills a fragment is an `asm volatile` whose output register the
compiler believes is live the instant the asm ends. Volatile asm blocks keep
their order relative to each other, but `v_mfma_f32_16x16x16bf16_1k` is a plain
intrinsic with no memory effects, so the machine scheduler is free to hoist it
above the wait. It did, in exactly the three `K_TAIL=false` instantiations, and
only those:

| instantiation | MFMAs hoisted above the wait | shapes using it |
|---|---:|---|
| `<32,64,64,false>` | 3 (of 4 per k-iteration) | row 1 + all even-K generic rows — all FAIL |
| `<64,64,64,false>` | 4 | row 2 — FAIL |
| `<256,256,32,false>` | 1 (of 64) | rows 4, 5 — FAIL |
| `<128,256,32,true>` | 0 | row 3 — PASS |
| `<256,256,32,true>` | 0 | row 6 — PASS |
| `<32,64,64,true>` | 0 | odd-K generic rows — PASS |

The correlation is exact: hazard ⇔ failure. The `K_TAIL=true` instantiations
were clean **only by luck** — `mask_a_k_tail` fragments the block and gave the
scheduler nowhere to hoist to. Nothing in the source distinguished them.

Fix: `frag_anchor` launders every base tile's register pair through
`asm volatile("" : "+v"(...))` after the wait. That is a real def, so every
consumer is ordered after the drain by data dependence rather than by scheduler
goodwill. It emits **no instructions** and costs **no registers** (233 VGPR
before and after; the k-loop grew by one `s_nop`). One operand per base tile,
matching the `ds_read_b64` that filled it. A `sched_barrier(0)` would probably
also have worked, but it constrains a scheduler pass rather than establishing a
dependence, so it is the weaker of the two guarantees for the same price.

`experiments/exp_03_mainloop/lds_race_check.sh` now does this check mechanically
for both counters — it tracks the destination registers of every `asm`-issued
`ds_read` and `global_load` since the last drain (handling counted waits
properly) and reports any instruction that reads one. It found the hazard, and
it confirmed 0 hazards across all six instantiations for both shipped arms.
**Run it after every mainloop change; it is a 30-second static check that
substitutes for a 3-minute M3 failure.** It also confirms the pre-existing
latent hazard recorded in `LESSONS.md` (last-k-iteration `ds_read`s reaching the
MFMAs with no `lgkmcnt(0)`) is now repaired: 0 hazards, where the shipped
baseline had them on the guard-skipped path.

## Two tooling defects found on the way

1. **`tools/gate_ladder.sh:53` asserts the M2 metadata table has ≥7 rows; the
   current dispatch has exactly 6 distinct instantiations, so the committed
   ladder cannot pass on the committed baseline.** 7 was right until exp_04
   retiled row 1 from `<32,256,32,false>` to `<32,64,64,false>`, which collapsed
   it into the generic even-K row. This is stale against `c0a6bdd2`, not against
   this experiment — exp_03 does not touch `dispatch_gemm_rs_mi300x`, so the set
   of instantiations is unchanged by it. `tools/` is not ours to edit, so both
   arms were gated with `experiments/exp_03_mainloop/ladder.sh`, a faithful copy
   whose M2 instead asserts that all six expected tuples are present **by name**
   — strictly stronger than counting rows, since a vacuous table, a missing
   instantiation and an unexpected extra one all fail. Nothing the benchmark
   measures was changed. **Recommend fixing the constant in `tools/`.**
2. **The assembler's loop-depth comments are not a reliable guide to k-loop
   membership.** After P3 the prefetch block `.LBB3_86` is annotated
   `in Loop: Header=BB3_84 Depth=1`, i.e. outside the k-loop, which would be
   illegal. Reading the branches shows the latch `%bb.88` does
   `s_cbranch_scc0 .LBB3_86` / `s_branch .LBB3_87`: the compiler turned the
   in-body `if (k + 1 < k_iters)` guard into a **choice of latch target**,
   jumping to the prefetch block for every iteration but the last. `.LBB3_86` is
   inside the loop; `MachineLoopInfo` just picked `BB3_87` as the natural-loop
   header of a now multi-entry region. `p0_30_kloop.sh` groups blocks by those
   annotations and therefore under-reports the loop body. Use
   `cfg_dump.sh` / `where_prefetch.sh` when a block seems to have vanished.

## What I would try next

1. **E1(c) is worth less than the plan assumed, and E1(b) is why.** The
   scheduler already interleaves half 1's `ds_read`s with half 0's MFMAs on its
   own. What `sched_group_barrier` could still add is spreading the four
   `global_load`s *through* the MFMA block instead of clustering them in
   `%bb.87`, and interleaving the commit's `ds_write`s with the tail of the
   MFMAs.
2. **Deepen the pipeline to 2 k-blocks ahead** (issue k+2 while committing k+1).
   That needs a third LDS buffer, which the 256-row configs cannot afford at
   `LDS_BYTES = 65536`; but it costs only registers if the extra stage stays in
   the staging buffer rather than LDS — `abuf`/`bbuf` doubled would be +16 VGPRs
   on rows already at 246/248, so it is gated on finding pressure elsewhere.
   Cheap on the four smaller configs, which have 165 VGPRs or fewer.
3. **Shape 4 got nothing (−0.7%)** and is now the worst `× SOL` row at 3.06.
   With only 16 k-iterations per tile its cost is not in the mainloop; profile it
   against the egress and release pools rather than assuming the mainloop axis
   still applies to it.
4. The mainloop is no longer 46% of shape 6. **Re-run the attribution on the new
   winner before choosing the next axis** — E2 (XGMI egress, 32%) and E3
   (release granularity) are now proportionally larger.

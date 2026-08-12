# exp_28 — what actually reduces M6's vector-memory cost, and what each option costs

Research + design only. No kernel file was edited; no GPU job was run. Two new
measurements were taken with **CPU-only compiles** in `subha_k1`
(ROCm 7.2.4 / LLVM 22, `--offload-arch=gfx950`) and are the backbone of every
kill below:

1. **The gfx950 LDS ceiling is exactly 163,840 B**, compiler-enforced:
   `local memory (163841) exceeds limit (163840)`. 163,840 compiles.
2. **A five-point `N2GM_G` resource sweep of the isolated M6 body**
   (`n2_phase1_gm.cpp -DN2_KERNELS_ONLY`), which reproduces exp_59's register
   table on today's toolchain **and adds the LDS column that table omitted** —
   the column that turns out to close the axis.

Probe scripts: `../tools/e28_gsweep.sh`, `../tools/e28_probe.sh`,
`../tools/e28_hist.sh`, `../tools/e28_e65.sh`.

---

## 0. Executive summary

| # | option | ΔM6 (µs) | build h | µs/h | verdict |
|---|---|---:|---:|---:|---|
| **O4** | **`ascale` gather: read the token-major `sc_stage` directly, delete the M5 transpose** | **−130 … −190** | **5–6** | **~28** | **BUILD FIRST** |
| O6 | `a2_lds` ∪ `ascale_lds` (new; found here) | 0 | 1–2 | 0 | BUILD as an enabler/gate only |
| O8 | print `num_tiles[0]` / `nvi[0]` (new) | 0 | 0.5 | — | BUILD (removes ±4 % from every number here) |
| O2 | task-order swizzle | **0** predicted (−350 only if the limiter is XCD→LLC) | 3–5 | ~0 | **KILL** — measure with rocprof before ever building |
| O1 | larger `M` (`G=4`) | −250 … −450 **if** three walls are cleared | ≥20 | ≤20 | **KILL** |
| O3 | more, smaller chunks (`nc=16/32`) | **+867 measured**; `nc=32` illegal | — | <0 | **KILL (already measured, exp_65)** |
| O5 | deepen the software pipeline | not affordable | — | — | **KILL** |
| O9 | `global_load_lds` for the A tile (new) | negative | — | — | **KILL** (bank arithmetic, §O9) |

**The one-line answer to the experiment's question.** M6's request traffic is
83.7 % weights, and weight traffic depends on **exactly one** parameter —
`num_tiles`, i.e. `M` — and `M` is walled at 96 rows by *three independently
measured* limits (256 AGPR, 163,840 B LDS, and the 128-column `DQ2`/amax group).
The only traffic left that is neither weights nor structurally pinned is the
**977 MB/epoch of pure line-amplification waste in the `A_scale` gather**, and
that one is free to delete because *we* create the layout that causes it.

**The correction that matters most.** The `2,539 × 161.7/204.8 = 2,005 µs`
arithmetic in the brief assumes M6's time is proportional to its request bytes.
Three paired same-run measurements (exp_65) falsify that: a **26 % byte cut
bought 9.4 %**, and a **+11 % byte increase cost +27 %**. The measured marginal
rates are **0.0826 cyc/B for the weight stream and 0.407 cyc/B for the
activation stream — the A stream costs 4.9× per byte what the B stream costs**
(§2). `G=4` moves bytes from the cheap stream to the expensive one, so its true
value is **−250 … −450 µs, not −531 µs** — and it is unreachable anyway.

---

## 1. The measured foundation (new, today)

### 1.1 The `G` sweep — isolated M6 body, gfx950, ROCm 7.2.4

`hipcc --genco -O3 -DKITTENS_CDNA4 -ffast-math -mllvm -amdgpu-mfma-vgpr-form=1
-DN2_KERNELS_ONLY -DN2GM_G=$G -Rpass-analysis=kernel-resource-usage`

| `G` | `M` | ArchVGPR | AGPR | scratch B/lane | VGPR spills | **phase-1 LDS B** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 256 | 58 | 0 | 0 | 25,344 |
| 2 | 64 | 256 | 150 | 0 | 0 | 49,792 |
| 3 | 96 | 256 | **232** | 0 | 0 | 74,240 |
| 4 | 128 | 256 | **256 (sat)** | **528** | **155** | 98,688 |
| 5 | 160 | 256 | 256 (sat) | 1,000 | 420 | 123,136 |

The AGPR column reproduces `exp_59/rescheck/register_fingerprint.md` exactly
(58 / 150 / 232 / 256), so the toolchain has not moved; the `G=4` scratch is
528 B here vs 532 B there.

### 1.2 The LDS law — exact on five points, and it is the wall

The five LDS numbers are **exactly** `764·M + 896`:

```
per row of M, phase 1:  tok_lds 4 + ascale_lds 224 + amax_lds 8
                      + a2_lds 272 + a_lds 256                       = 764 B
G-independent:          b1s_lds[2][2][56]                            = 896 B
check G=3: 764·96 + 896 = 74,240   (measured 74,240)
```

Phase 2 adds `328·M + 18,176` (`n2_phase2_gm_mps.cpp:284-291`:
`tok 4 + w 4 + dq2 64 + a_lds 256` per row; `b2s 1,792 + xp 16,384` fixed), and
the megakernel body adds 30,500 B (`k0pf6gm_device_tile.hip:287-294`). So for
the **fused** object:

```
LDS_fused(M) = 1,092·M + 50,596 B
  G=2 → 120,484   measured 120,488   (exp_63 result.md, +4 B align)
  G=3 → 155,428   measured 155,432   (exp_63 result.md, +4 B align)
  G=4 → 190,372   ceiling 163,840    →  OVER BY 26,532 B
```

Three measured fused points, all within 4 B. Today's MPS object is 155,496 B, so
**LDS headroom is 8,344 B = 7.6 rows of `M`**. Solving `1,092·M + 50,596 ≤
163,840` gives **`M ≤ 103.5`**, i.e. `M = 96` is the last legal multiple of 32.

> **This is a new finding.** Every historical `G=4` kill (exp_59, exp_63,
> exp_65, `lever_matrix.md`) names *registers only*. The LDS wall is
> independent, is 26,532 B thick, and would still be there if the register
> problem were solved.

### 1.3 What the accumulators actually are

`racc acc[kGM][2][2][kColTiles]` (`.node/mps_n2_phase1_gm.cpp:212`), 4 floats per
lane per `racc`:

```
acc registers/lane = 16·kGM·kColTiles = (M/32)·(N/128)·16 = M·N/256
```

which is just "the `M×N` output tile spread over 256 threads". The other live
arrays: `af[kGM][2]` = `16·kGM`, `aTmp[kGM][2]` = `8·kGM`, `bf[2][2][kColTiles]`
= `32·kColTiles`. Total named:

```
R(kGM, kColTiles) = kGM·(16·kColTiles + 24) + 32·kColTiles
  G=3,c=4 → 392   (measured total 256+232 = 488, so ~96 for addressing/loop state)
  G=4,c=4 → 480   (+88; 512 is the whole file)  → measured: AGPR saturated, 528 B scratch
```

`stage_agpr` cannot rescue anything: `N1G_STAGE_AGPR` defaults to **0**
(`n2_device_common.cuh:68-79`), so it is compiled out to `(void)f` and the B
fragments are already wherever the allocator wants them.

---

## 2. The cost model — measured, and it is not the bandwidth model

exp_65 ran three paired same-run `run_decomp.sh` configurations on one base,
which is the only clean 2-D scan of (`M`, `N`) that exists
(`exp_65_phase1_accumulator_pressure/result.md`). Converting to per-K-step
cycles at 2.2 GHz (`K-steps/CTA = num_tiles·(32/kColTiles)·56/256`):

| config | `M` | `N` | B B/K-step | A B/K-step | M6 µs | K-steps/CTA | **cyc/K-step** |
|---|---:|---:|---:|---:|---:|---:|---:|
| G=2/c8 | 64 | 512 | 65,536 | 8,192 | 3,154.6 | 931 | 7,456 |
| G=2/c16 | 64 | 256 | 32,768 | 8,192 | 4,021.2 | 1,862 | 4,751 |
| G=3/c16 | 96 | 256 | 32,768 | 12,288 | 3,644.8 | 1,249.5 | 6,417 |
| G=3/c8 *(shipped)* | 96 | 512 | 65,536 | 12,288 | 3,159 | 624.75 | 11,126 |

Two orthogonal differences give two marginal rates:

```
B stream:  (7,456 − 4,751) / 32,768 B  =  0.0826 cyc/B   → 12.1 B/clk/CU
A stream:  (6,417 − 4,751) /  4,096 B  =  0.4067 cyc/B   →  2.5 B/clk/CU
                                        ⇒ A costs 4.92× per byte
```

In grid terms `0.0826 cyc/B` = **146 µs per GB of M6 request traffic**
(`1e9 × 0.0826 / 256 CU / 2.2e9`). Fitted model:

```
cyc/K-step = 0.0826·B_bytes + 0.4067·A_bytes − 1,286
  reproduces the three fit points exactly; under-predicts the shipped
  G=3/c8 point by 18 % (9,124 vs 11,126) — so treat its extrapolations
  as optimistic bounds, never as predictions.
```

### 2.1 Why the A stream costs 4.9× per byte

Per lane per K-step, per unit of `kGM`, the A path issues **six** memory
instructions — 1 `buffer_load_b128` (`:237`), 1 `ds_write_b128` (`:244`),
4 `ds_read_b128` (`:258-259`) — plus it is the only stream that carries a
loop-carried `vmcnt` dependency (`load_a(k+2)` → `store_a` in the next half,
§5.3 stall site 1) and the only one gated by the two `lds_cta_barrier()` calls
per iteration. The B path is 4 direct `buffer_load` to registers per
`kColTiles` unit and touches neither LDS nor a barrier. `ascale_lds` rides the
same A-side LDS path (2 `float4` reads per `kGM` per K-step, `:283-286`).

**Consequence for every option below:** at G=3/c8 the A stream is **15.8 % of
the bytes but ~48 % of the modelled cost**. Anything that trades W bytes for A
bytes loses; anything that deletes A-side work wins out of proportion to its
byte count.

### 2.2 Where the traffic actually comes from

Two exact identities, both worth more than the intensity formula:

```
W13 requests = num_tiles × 2·kInter × kHidden = num_tiles × 29.36 MB
             → depends ONLY on num_tiles (i.e. on M). kChunks cancels.
A requests   = num_tiles × kChunks × M × kHidden
             → ∝ 1/kColTiles at fixed M.
```

At G=3/c8: `357 × 29.36 MB = 10.48 GB` weights (83.7 %), `1.95 GB` A (15.7 %),
`0.08 GB` stores. **`kChunks` cannot reduce weight traffic by one byte** — it
only multiplies A passes. That single line kills O3 before any measurement.

### 2.3 Standing-kill-list collisions

`aug10/CLAUDE.md:263-264` strikes "chunked/micro-batch decomposition of M6 or M7
(matches our own measured G3/chunk/K-split kill)". The historical record behind
that phrase, recovered tonight:

| what the kill actually was | evidence |
|---|---|
| `G=3` fused **build-gate** kill (512V / 256A / 7–9 spills / 32–40 B scratch) | `exp_59/result.md`, `exp_63/result.md` |
| `G=4` **never built and never timed** — killed on the *standalone* probe alone (256 AGPR sat, 532 B scratch) | `exp_59/rescheck/register_fingerprint.md`; `exp_63/design.md:136` "G=4: dead … **Not attempted**" |
| `kChunks 8→16` **built, gated and timed**: register-clean (476V/220A/0 spill/LDS 154,600) but **1.0023× — loses**; A re-pass tax +866.6 µs vs G amortization −376.4 µs | `exp_65/result.md` |
| K-split | superseded and killed by exp_65 (the accumulator is K-reduced in place; there is no K dimension to split) | `experiments/summary.md` exp_65 ¶ |

Two things follow. **(a) `G=4` was never a measured kill** — the brief is right
to ask. **(b) It does not matter**, because §1.2's LDS wall and §1.1's measured
528 B of spill close it anyway, and §2's measured cost model shrinks the prize
to ≤ −450 µs. And exp_65 *did* measure the `nc` axis, so O3 is a genuine
measured kill and must not be re-litigated.

Also note exp_65's own durable finding, which the brief's framing needs:
**"the 256-AGPR wall is an accumulator-footprint ceiling, not a tile-width
ceiling"** — and its recommendation: *"any future large-M attempt must reduce
the accumulator WITHOUT multiplying re-passes over A."*

---

## 3. O1 — larger `M` (raise intensity) → **KILL**

### 3.1 The feasible-shape enumeration

Legal shapes are `M = 32·kGM`, `N = 128·kColTiles`, subject to:

- **C1 — AGPR:** `acc = M·N/256 ≤ 232` (measured: 232 at (96,512) already
  saturates the *fused* union, which adds ~24 AGPR over the isolated body —
  exp_59 F1) ⇒ `M·N ≤ 59,392`.
- **C2 — LDS:** `1,092·M + 50,596 ≤ 163,840` ⇒ `M ≤ 103.5`.
- **C3 — `DQ2`/amax granularity:** the dequant scale is one FP32 per
  (row, 128-column group of `kInter`), `DQ2` has exactly `kKGroups2 = 16`
  columns (`n2_phase1_gm.cpp:465-468` writes `2g+kb`;
  `n2_phase2_gm_mps.cpp:404-410` reads all 16), and the amax reduction that
  produces it is **CTA-local** (`amax_lds[2][kMrows]`, `j_w = wv>>1`, `:396-406`).
  A K128 group must therefore lie entirely inside one task ⇒ `N/2 = kChunkCols`
  is a multiple of 128 ⇒ **`kColTiles` even** ⇒ `N ∈ {256, 512, 1024}`.
- **C4 — integrality:** `kChunks = 2048/kChunkCols = 32/kColTiles` must be an
  integer.

| (`M`,`N`) | `acc` | C1 | LDS B | over by | C2 | C3 | verdict |
|---|---:|:--:|---:|---:|:--:|:--:|---|
| (96, 512) | 192 | ~at wall | **155,428** *(measured)* | — | ✓ | ✓ | **today** (9 spills / 40 B) |
| (128, 384) | 192 | ✓ | 181,700 | 17,860 | ✗ | ✗ | **illegal** — §3.2 |
| (128, 512) | 256 | ✗ | 190,372 | 26,532 | ✗ | ✓ | dead on both |
| (128, 256) | 128 | ✓ | 173,028 | 9,188 | ✗ | ✓ | LDS ✗; and modelled **worse** (§3.3) |
| (160, 256) | 160 | ✓ | 203,748 | 39,908 | ✗ | ✓ | dead |
| (192, 256) | 192 | ✓ | 234,468 | 70,628 | ✗ | ✓ | dead |
| (96, 1024) | 384 | ✗ | 181,668 | 17,828 | ✗ | ✓ | dead on both |

*LDS convention for the hypothetical rows: **tight** (best-case) strides —
`a2_lds = 64·kColTiles + 16`, and `amax_lds`/`b1s_lds` scaled by
`kColTiles/2` — i.e. each shape is given its most favourable footprint. Note
exp_65's **built** `c16` arm did not tighten `kA2Stride` (a hard `272` at
`n2_phase1_gm.cpp:59`) and measured **154,600 B**, against 154,596 predicted
with the loose stride and 142,308 with the tight one — so real builds land on the
pessimistic side of this table, never the optimistic one.*

**Within both walls the achievable intensity is `2MN/(M+N)` at `M ≤ 103`,
`N ∈ {256,512}` — i.e. 161.7 (today) is already the maximum.** The axis is not
"expensive", it is **empty**.

### 3.2 Verdict on the `M=128, N=384, kColTiles=3` variant — **REFUTED**

The brief's register arithmetic is **correct and I confirm it**:
`acc = 16·kGM·kColTiles = 16·4·3 = 192`, identical to today; and it is in fact
slightly better than neutral, because `bf` shrinks with `kColTiles` faster than
`af`+`aTmp` grow with `kGM`:

```
R(3,4) = 3·(64+24) + 128 = 392
R(4,3) = 4·(48+24) +  96 = 384        →  −8 registers/lane, not merely neutral
intensity 2·128·384/512 = 192          →  +18.7 % over 161.7   (also correct)
```

It nevertheless breaks on **three independent counts**, none of them registers:

1. **Integrality (C4).** `N = 384 ⇒ kChunkCols = 192 ⇒ kChunks = 2048/192 =
   10.67`. There is no integer chunk count. `kChunks` is a `constexpr`
   (`n2_phase1_gm.cpp:50-53`) and `num_tasks = num_tiles·kNChunksP1`
   (`:136-137`).
2. **The `DQ2`/amax contract (C3) — this is the fatal one.** A 192-column chunk
   spans 1.5 of M7's 128-column K-groups, so the group straddling two chunks
   would have its `amax` computed from **half its columns in each of two
   different tasks**, on two different CTAs, and each half would be quantized by
   a *different* scale (`:415-425`) while M7 applies one `DQ2[row][k]` scalar to
   all 128 columns (`:404-410`). That is not awkward, it is **numerically
   wrong**. Repairing it requires either a cross-CTA `atomicMax` two-pass over
   the intermediate, or widening `DQ2`'s granularity — an M7 contract change.
3. **LDS.** At its most favourable strides the `kColTiles=3` variant needs
   `128 × 1,026 + 672 + 49,700 = 181,700 B` — **17,860 B over** the measured
   163,840 ceiling. (Per row: `tok 4 + ascale 224 + amax 6 + a2 208 +
   a_lds 256 + phase2 328`.)

And even if all three were repaired, the **fused** build would still die in
**phase 2**, which shares `N2GM_G`: at `G=4` the isolated phase 2 measures
256V/256A + 136 B scratch (`exp_59` table). Raising `M` for M6 alone requires
splitting `G` per phase (feasible — `a2_done` counts per 32-block either way —
and never tried; exp_63 only rejected *lowering* phase 2's `G`, which indeed
frees nothing).

### 3.3 What `G=4` would actually be worth

Two independent estimates, both below the brief's −531 µs:

```
naive request-bandwidth model:  2,539 × 161.7/204.8              = −531 µs
measured cost model (§2):       G4c8 10,788 cyc/K-step × 478      = 2,344 µs
                                vs clean-model G3c8 9,124 × 625   = 2,591 µs   → −247 µs
measurement-anchored scaling:   the whole G=2→3 step (−33 % weight
   traffic) measured −376.4 µs; G=3→4 is a further −23.5 %, i.e.
   0.71× of that step                                            → ≈ −267 µs
```

`num_tiles(G=4) = 273` (`exp_59/design.md:184` "8–9 super-blocks/expert",
6.2 % pad). So: **−250 to −450 µs, for ≥26,532 B of LDS + ~88 registers/lane +
a per-phase `G` split — three coupled mechanisms, ≥20 build hours, each with its
own gate ladder.** Compare O4: −140 µs for one uncoupled layout change in 5 h.

**Falsifier, pre-registered, should anyone revive it:** build the isolated body
at `G=4` with `kColTiles=2` *and* the O6 LDS union applied and require
`AGPR ≤ 232 ∧ scratch = 0 ∧ LDS_fused ≤ 160,000`. Tonight's sweep says
`G=4/c8` fails this by 24 AGPR and 26,532 B; `G=4/c16` + O6 clears LDS
(154,596 with tight strides, 160,740 with exp_65's loose `kA2Stride`) and
registers (`acc = 128`) — and is then
**modelled at 3,512 µs, worse than today's 3,159**, because `c16` doubles the
expensive A stream. That is the closed loop.

### 3.4 What `G` touches (for the record)

`#error` guard `k0pf6gm_device_tile.hip:152-157` and `_mps.hip:189-192` · host
guard `PF6MPS_G = 3` with fail-closed readback `num_tiles_p[1] = G`
(`_mps.hip:1269-1272`, `ab.py:248-254`) · M4's device tile-table build
(`_mps.hip:1244-1268`) · `kMrows`-indexed LDS in **both** phase bodies
(`n2_phase1_gm.cpp:116-124`, `n2_phase2_gm_mps.cpp:284-291`) · `acc`/`af`/`aTmp`
in both · the 3-sub-block epilogue and its `__syncthreads()` pair (`:407,:429`)
· the per-live-sub-block `a2_done`/`part_done` hooks (`:481-488`) · the
`96+84=180` MFMA census (`PROVENANCE.md:110`) — at `G=4` it becomes `128+112`.

---

## 4. O2 — task-order swizzle → **KILL** (and the brief's reasoning is right, but for a weaker reason than the one that closes it)

### 4.1 Confirming the brief's arithmetic

Per-XCD residency needs `⌈CTAs_per_XCD / tasks_sharing_one_slab⌉` slabs. With
`xcd = bid mod 8` (`moe_mps_adapter.cuh:93-96`) there are 32 CTAs per XCD, and a
`W13` slab `(e, g)` is shared by exactly the `11.1` tiles of expert `e`:

```
min slabs per XCD = ⌈32 / 11.1⌉ = 3   ⇒   3 × 3.67 MB = 11.0 MB  >  4 MB L2
```

No permutation changes either factor, so **no remapping gets the weight working
set under 4 MB at `kColTiles = 4`**. Confirmed: the axis is closed as posed.
(The escapes are all worse: `kGM=1` gives 33.4 tiles/expert but destroys
intensity; `kColTiles=1` shrinks the slab to 0.92 MB but is illegal under C3.)

### 4.2 The stronger kill: capacity is the wrong model, and the limiter is upstream of the L2

Two corrections to §2.3 of `m6_m7_structure.md`:

1. **Reuse among lockstep sharers does not need capacity.** The ~11 CTAs on one
   XCD that share slab `(e,g)` walk the same K-loop at the same rate, so the L2
   only has to hold the *window* (a few K-steps × 65,536 B), not the 3.67 MB
   slab. The "10.6 MB does not fit in 4 MB, so most of the 4.91 TB/s leaves the
   XCD" inference does not follow. `INFERRED`, but it is the only reading
   consistent with measured HBM ≈ compulsory.
2. **The measured marginal rate is 12.1 B/clk/CU** (§2) = 387 B/clk/XCD. Every
   candidate ceiling downstream of the L1 is far above that. So the binding
   resource is the **per-CU vector-memory path** (L1 request/miss-tracker
   throughput), and a swizzle moves bytes between the L2 and the Infinity Cache
   **without changing the number of requests a CU issues**.

Best-case accounting for the best swizzle I could construct (expert-blocked,
chunk-inner: give each XCD all ~11 tiles of one expert × ~3 chunks, so the A
slab gets 3-way sharing and the W slab keeps 11-way):

```
LLC-side traffic = W/S_W + A/S_A
  today   (32 tiles × 1 chunk per XCD): 10.48/11 + 1.95/1  = 2.90 GB
  swizzle (11 tiles × 3 chunks per XCD): 10.48/11 + 1.95/3 = 1.60 GB   (−1.30 GB)
  naive "xcd = tile mod 8" (4 tiles × 8 chunks): 10.48/3.1 + 1.95/8 = 3.61 GB  (WORSE)
request-side traffic (what the CU issues): 12.46 GB in every case    (−0 GB)
```

**Predicted ΔM6 = 0 µs.** Conditional upside if the XCD↔LLC path is in fact the
limiter: `−1.30 GB × 146 µs/GB ≈ −190 µs` (or up to −350 µs if the LLC path is
more expensive per byte than the fitted average). That condition is
`m6_m7_structure.md` §7 item 3, still unresolved.

**Recommendation:** do not build this. **Measure it** first — one `rocprofv3`
PMC pass on the *existing* binary reading `TCC_HIT/TCC_MISS`,
`TCP_TCC_READ_REQ`, and `TCC_EA_RDREQ` splits the 4.91 TB/s into L2-hit vs
LLC-served and settles both this option and §7 item 3 for the cost of one
profiling run. `exp_22`'s ubench (mode b) is the alternative.

---

## 5. O3 — more, smaller chunks → **KILL (measured)**

`nc = 32` is **structurally illegal**: `kColTiles = 1 ⇒ kChunkCols = 64`, half a
K128 group, violating C3 exactly as `N=384` does.

`nc = 16` was **built, gated and timed** by exp_65:

```
G=2/c8  → G=2/c16   (pure chunk doubling, M fixed):  M6 3,154.6 → 4,021.2 µs = +866.6
```

The crossover the brief asks for **does not exist**, and §2.2 says why in one
line: `kChunks` does not appear in the weight-traffic identity
(`W13 = num_tiles × 29.36 MB`). Doubling the chunk count buys **zero** weight
bytes and costs a full extra A pass (`+1.95 GB` at the expensive 0.407 cyc/B
rate = +362 µs of pure byte cost, plus a doubled per-task prologue/epilogue —
together the measured +866.6 µs). For it to pay, an L2 hit would have to be
*negative* cost.

The per-XCD residency claim in the brief is also not delivered: at `nc=16` the
slab is 1.83 MB and the floor is still `⌈32/11.1⌉ = 3` slabs = **5.5 MB > 4 MB**.

---

## 6. O4 — kill the `ascale_lds` line amplification → **BUILD FIRST**

### 6.1 The waste, exactly

`.node/mps_n2_phase1_gm.cpp:187-194`, between `__syncthreads()` at `:184` and
`:195` — so **no MFMA overlaps it**:

```cpp
ascale_lds[k][i] = (token < T) ? A_scale[(size_t)k * T + token] : 0.0f;
```

`T = nvi[1] = T_ext = 32,768`, so consecutive `k` for one token are 131,072 B
apart. Per task the 96 tokens × 56 groups touch **5,376 distinct 64 B lines to
deliver 21,504 B — 16× amplification, 344,064 B/task, 977 MB/epoch (+7.8 % on
the 12.456 GB total)**.

### 6.2 (a) Why group-major? Because nothing wants it.

- The producer is `hk_moe::mps::scale_transpose_row`
  (`moe_mps_adapter.cuh:298-305`, exp_24's extracted half of
  `hkp::zero_part_scale_transpose<14>`), called at
  `k0pf6gm_device_tile_mps.hip:1304-1306` / `:1310-1312`:
  `sc_dst[lane·T_loc + t] = sc_stage_row[lane]`.
- Its **input `sc_stage` is already token-major**: the call passes
  `sc_stage + t·K0P6_NG` with `K0P6_NG = 56` (`k0pf6gm_device_tile.hip:79`), so
  row `t`'s 56 scales are contiguous.
- **`sc_dst` has exactly one consumer: M6.** Exhaustive `K0P6_D_SC_DST` census
  in the kernel — `:144` (define), `:1226` (M5 takes the pointer), `:1304`/`:1310`
  (the transpose writes it), `:1347` (M6 reads it). Nothing else in any mode.
  `sc_dst` is not peer-visible, not in the combine, not in `part`.

So M5 spends 7.34 MiB of scattered 4-B stores per epoch manufacturing a layout
whose only reader is the one phase it hurts. The group-major convention is
inherited from the donor's standalone `input_scale` signature — and note the
standalone host contract actually asserts the *opposite* shape
(`n2_phase1_gm.cpp:531-533` requires `input_scale.cols() == 56`, i.e.
token-major), so even the donor's own contract and its indexing disagree. The
megakernel bypasses that check (`#ifndef N2_KERNELS_ONLY`).

### 6.3 (b) Yes — 14.0×, and the arithmetic is exact

A token-major row is `56 × 4 = 224 B = 3.5` lines, and `224·t mod 64 ∈ {0, 32}`
for every `t`, so **every row touches exactly 4 lines** (never 5):

```
today:      96 tokens × 56 groups              = 5,376 lines = 344,064 B
token-major: 96 tokens ×  4 lines              =   384 lines =  24,576 B
                                                 → 14.00× exactly
per epoch:  (344,064 − 24,576) × 2,840 tasks   = 907.2 MB deleted (−7.28 %)
```

Three independent estimates of the µs, deliberately not one:

| model | arithmetic | Δ |
|---|---|---:|
| measured marginal byte rate (§2) | `0.9072 GB × 146 µs/GB` | **−132 µs** |
| L1-miss-slot model (`INFERRED`: ~64 outstanding lines / ~340 cyc, the pair that reproduces 12.1 B/clk/CU) | `(5,376−384) × 340/64 = 26,520 cyc/task × 11.09 / 2.2e9` | **−134 µs** |
| §5.1's own average-rate model (8.65 B/clk/CU) | `319,488 B / 8.65 × 11.09 / 2.2e9` | **−186 µs** |

Plus M5: the transpose's 56 scattered 4-B stores per row × 32,768 rows touch
1.835 M lines ≈ 117 MB of write-allocate traffic ⇒ **≈ −17 µs** on the ~415 µs
plan phase, and its read side (7.34 MB) is noise. **Total −130 to −190 µs,
point estimate −140 µs.** Note this is a *smaller* claim than
`m6_m7_structure.md` §5.1's ~200 µs, because §5.1 prices the gather at the
phase's average B/clk rather than the measured marginal rate.

### 6.4 (c) What breaks — nothing, and the numerics must be bit-identical

- **Readers of `sc_dst`:** M6 only (§6.2). After the change it has none; leave
  the buffer allocated (descriptor slot 20 stays valid) so the ABI is untouched.
- **`sc_stage` lifetime:** written by M2's unpack
  (`k0pf6gm_device_tile_mps.hip:1160-1161`), read by M5's transpose today. M6
  reading it directly requires it to still be live at M6 — it is: descriptor
  slot 9 is never rebound and nothing writes it between M2 and M7.
- **Publication:** `sc_dst` is published by the M5 grid barrier (`:1327`).
  `sc_stage` is published by the M3 (`:1236`), M4 (`:1291`) *and* both M5
  barriers — **strictly more ordering, no new edge.**
- **Hole rows:** M2 skips hole rows (`off >= s_ns[s] → continue`, `:1159`), so
  `sc_stage` hole rows are uninitialized, whereas `sc_dst` hole rows today carry
  whatever the transpose copied out of those same uninitialized bytes. Neither
  is ever read: `tok_lds` only ever names live receive rows, pad sub-blocks get
  the sentinel `T` and are masked by `token < T` (`:170-172`, `:191`).
- **Bit-identity.** `sc_dst[k·T_ext + t] ≡ sc_stage[t·56 + k]` **by
  construction** (`moe_mps_adapter.cuh:303`). Same float, same predicate, same
  accumulation order ⇒ `A2q`, `DQ2`, and therefore the whole kernel output must
  be **bit-identical**. That is a far stronger gate than a tolerance.

### 6.5 Coordination with exp_24 and exp_27

- **What exp_24 leaves behind:** the `skip_dead_part_zero(cfg4)` fork at
  `k0pf6gm_device_tile_mps.hip:1300-1314`. Fast branch (mode 12, no dual, no
  pull) = `hk_moe::mps::scale_transpose_row(...)` only; fallback branch = the
  original fused `hkp::zero_part_scale_transpose<14>`. exp_24 deletes 448 MiB of
  `part` zero stores and **keeps the transpose**, explicitly because "`sc_dst`
  **is** read by M6" (`exp_24_dead_plan_work/design.md:48-49`).
- **What O4 adds on top:** it removes the *last* reason that fork's fast branch
  exists. With M6 reading `sc_stage`, the fast branch's loop body becomes empty,
  so the entire `for (t = w; t < T_ext; t += nw2)` loop and its CTA work-split
  disappear in mode 12. The fallback branch is left **byte-identical** (it keeps
  writing a now-dead `sc_dst`, which is harmless), which is what keeps this a
  single-variable change.
- **exp_27** owns this mechanism in `aug11/STATUS.md`'s queue. This analysis is
  the arithmetic and the consumer audit for it; `design.md` next to this file is
  written so either agent can execute it. The one thing exp_27 must **not** do is
  the obvious-looking version — adding a second, token-major transpose in M5 —
  because `sc_stage` already *is* the token-major array. The change is a
  deletion.

---

## 7. O5 — deepen the software pipeline → **KILL**

### 7.1 The register bill, priced at today's measured tuple

| deepening | extra registers/lane | extra LDS |
|---|---:|---:|
| B two-deep (`bf[3][2][kColTiles]`) | `+32·kColTiles = +128` | 0 |
| B one extra column-tile stage only | `+32` | 0 |
| A two-deep (`aTmp[kGM][3]` + `a_lds[3]`) | `+8·kGM = +24` | **`+256·kMrows = +24,576 B`** |

Available: the isolated body at `G=3` is 256 ArchVGPR / 232 AGPR = 488 of 512
(§1.1) and the **fused** object is 512 of 512 with 9 spills / 144 B scratch.
There are **24 registers of slack in the isolated body and none in the fused
one**. LDS slack is 8,344 B (§1.2) against the 24,576 B a third `a_lds` buffer
needs. Both variants are unaffordable, and A-deepening is unaffordable twice.

### 7.2 And the mechanism is wrong anyway

Deepening adds *latency tolerance*, not *concurrency*: the same 19 VMEM
instructions per wave per K-step are issued, just earlier. The measured
per-K-step time tracks **bytes**, not depth — exp_65's `c16` arm halves the
bytes per K-step at unchanged pipeline depth and its K-step time falls from
7,456 to 4,751 cycles, i.e. 64 % of the way to the byte-proportional 3,728 —
while sitting 9.3× above its own 512-cycle MFMA floor. A latency-limited loop
would not have moved with bytes at all.

Staging B through phase 2's LDS (the brief's suggestion) does not help either:
it converts 16 direct register loads per lane per K-step into
global→LDS→register, i.e. it moves the B stream onto the A stream's path, which
§2.1 measures at **4.9× the cost per byte**. That is the single most reliable
"loses" prediction in this document.

---

## 8. New options found here

### O6 — `a2_lds` ∪ `ascale_lds`: 21,504 B of free LDS

`ascale_lds`'s last read is inside `mfma_k` (`:283-286`), i.e. before the
K-loop's final `lds_cta_barrier()` at `:349`. `a2_lds`'s first write is at
`:424`, after the epilogue's `__syncthreads()` at `:407`. **Two CTA barriers
separate the two lifetimes**, so they can share storage with no new barrier, no
register change, no traffic change and no math change:

```
saving = min(224, 272)·kMrows = 224·M = 21,504 B at M=96   (28,672 B at M=128)
LDS_fused(M) becomes 868·M + 50,596  ⇒  M ≤ 130.5
```

Worth **0 µs by itself** — occupancy is already 1 block/CU and cannot improve —
so it is not a performance experiment. It is the *only* free LDS currency in
M6, it is the thing any future large-`M` attempt must spend, and it is a
perfect bit-identical control run (same numerics, same registers, −21,504 B).
Build it as a 1–2 h gate exercise whenever the queue has a gap, not as a
candidate.

*(`kA2Stride = 272` for a 256 B row (`:59`) is a deliberate 16 B bank pad, not
slack; and `a_lds` cannot join the union — it is live throughout the K-loop
alongside `ascale_lds`.)*

### O8 — print `num_tiles[0]` and `nvi[0]`

`num_tiles` is device-computed and never printed (`m6_m7_structure.md` §7.1);
every byte, task and FLOP total in this document scales linearly in it, at
±4 %. `num_tiles_p[1]` is already read back host-side for the compiled-`G`
fail-closed check, so `num_tiles_p[0]` costs one extra print. Half an hour,
removes the last systematic uncertainty from the whole M6 model.

### O9 — `global_load_lds` for the A tile → **KILL** (bank arithmetic)

Tempting: it would delete `aTmp` (`8·kGM` = 24 registers), delete `kGM`
`ds_write_b128` per lane per K-step, and cut stall site 1 (§5.3) — the
`vmcnt` wait that couples `load_a(k+2)` to `store_a`. It fails on LDS banking.
LDS-DMA destinations are **lane-linear**, which forces the A-LDS row stride to
be exactly 128 B and therefore forbids the XOR swizzle
`a_cp = a_chunk ^ (a_row & 7)` (`:152`, read side `:249-251`):

```
linear layout: read_a lane (q,r) reads dword 32r + 8q  ⇒ bank = 8q  (independent of r)
               ⇒ the 16 lanes sharing q hit the same 4 banks: 16 accesses/bank
XOR swizzle:   bank = 4·(q ^ (r&7)) ⇒ 8 distinct bank-quads, 2 lanes each
               ⇒ 8 cycles = the 1,024 B / 128 B-per-clk ideal
```

So it would make `read_a` (4·`kGM` = 12 instructions per lane per K-step) ~2×
more expensive to save 3 `ds_write`s — a net loss on the same LDS unit. Padding
the stride to 144 B fixes the banks but breaks the lane-linear contiguity that
LDS-DMA requires. Recorded so nobody spends a night on it.

---

## 9. Hardware numbers used, and their provenance

| number | value | mark |
|---|---|---|
| gfx950 max LDS per workgroup | **163,840 B** | **MEASURED tonight** — compiler rejects 163,841 (`../tools/e28_gsweep.sh`) |
| register file | 512/lane, ≤256 ArchVGPR + ≤256 AGPR | `DOCUMENTED` — CDNA4 ISA §3.6.4, via `aug10/CLAUDE.md` |
| L2 per XCD / LLC / HBM | 4 MB ×8 / 256 MB / 8 TB/s | `DOCUMENTED` — CDNA4 whitepaper, via `aug10/CLAUDE.md` |
| XCD dispatch | round-robin, chunk 1 (`xcd = bid mod 8`) | `DOCUMENTED`/`REPORTED` per `aug10/CLAUDE.md` A8; used at `moe_mps_adapter.cuh:93-96` |
| M6 marginal byte rate | **12.1 B/clk/CU (B stream), 2.5 (A stream)** | **MEASURED** — derived here from exp_65's three paired points |
| L1 vector-cache size, L1↔L2 bandwidth per CU, L1 miss-tracker depth, XCD↔LLC achievable read bandwidth | — | **UNRESOLVED.** I did not find a citable figure and the whitepaper fetch timed out. The §6.3 "64 lines / 340 cyc" pair is `INFERRED` — it is *a* factorization that reproduces the measured 12.1 B/clk/CU, offered only as a cross-check, and no conclusion here rests on it alone. |

I deliberately did **not** reuse the "1 MiB knee / 87 GB/s" family of numbers —
`aug10/CLAUDE.md`'s exp_04 retraction records that they are 4×A100/NVLink, not
CDNA4.

---

## 10. Conclusions

1. **M6's intensity axis is empty, not merely expensive.** Under three measured
   constraints — `M·N ≤ 59,392` (AGPR), `M ≤ 103.5` (LDS), `N ∈ {256,512,1024}`
   (the `DQ2`/amax 128-column group) — `(96, 512)` is already the
   intensity-maximizing legal shape. New: the LDS wall (26,532 B at `G=4`) is
   independent of the register wall that every prior kill cited.
2. **The brief's `M=128, N=384` idea has correct register arithmetic and is
   still refuted**, on integrality, on the `DQ2`/amax granularity (a numerics
   break, not a nuisance), and on LDS.
3. **`G=4` was never built or timed** — the historical kill is a standalone
   register probe (`exp_63/design.md:136`: "Not attempted") — but reviving it is
   still wrong: three coupled mechanisms for a measurement-anchored −250 to
   −450 µs.
4. **`nc=16` is a real measured kill** (exp_65, +866.6 µs) and `nc=32` is
   illegal. The crossover does not exist because weight traffic is independent
   of `kChunks`.
5. **The swizzle family is closed**, both by the brief's `11.1 < 32` argument
   and, more decisively, because the limiter is the per-CU vector-memory path,
   which a swizzle cannot touch. One `rocprofv3` PMC pass would confirm and
   simultaneously close `m6_m7_structure.md` §7 item 3.
6. **Build O4.** −130 to −190 µs, zero registers, zero LDS, bit-identical
   output, one gated deletion. It is the only item in the M6 traffic budget that
   is neither weights nor structurally pinned, and we are the ones who create
   it.

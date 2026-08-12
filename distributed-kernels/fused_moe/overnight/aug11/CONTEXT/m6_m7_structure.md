# M6 / M7 structure map — what the two expert GEMMs actually are, and what limits them

Read-only source map. Every structural claim carries a `file:line`. Every number
is either quoted from a cited measurement or derived here with the arithmetic
shown inline.

## 0. The headline, before the detail

**The premise "M6 is completely CTA-insensitive" is not supported by any
measurement that exists.** The reservation that produced the `+0.5%` figure
happens at **M6.9 — *after* the phase-1 body returns**
(`k0pf6gm_device_tile_mps.hip:1319-1330` calls phase 1, `:1346-1370` is the role
partition). The phase-1 task loop is `for (task = blockIdx.x; task < num_tasks;
task += kCTAs)` with `kCTAs` a compile-time 256
(`.node/mps_n2_phase1_gm.cpp:156`) — there is **no descriptor-derived stride, no
role predicate, and no early exit anywhere between the M5 barrier
(`k0pf6gm_device_tile_mps.hip:1301`) and the phase-1 call (`:1319`)**. In every
mode of every experiment run so far, **all 256 CTAs run all of M6.**

The experiment that generated the number says so itself:

> "**Sanity check that validates the whole table: `plan+M6` is invariant at
> 2,970–3,048 µs across every configuration.** The reservation happens at M6.9,
> so nothing before it should move, and nothing does. Any systematic error in
> the instrument would have shown up here first."
> — `overnight/aug10/experiments/exp_05_dispatch_overlap/stage0_attribution.md:55-58`

`exp_11` / `STATUS.md:197-203` / `KERNEL_BOTTLENECKS.md:61-67` then re-read that
same invariance as "M6 has idle CTA capacity". It is a control, not a
measurement. **No experiment has ever varied M6's CTA count.**

What the arithmetic below says instead: **M6 and M7 are the same fp8 K-loop at
the same arithmetic intensity (161.7 vs 158.1 FLOP per byte of vector-memory
request), running at the same ~7–9 B/clk/CU and the same 14–17% MFMA duty
cycle, and their runtimes are in the ratio their byte counts predict (1.75×
predicted, 1.60× measured).** M7 is measurably CTA-sensitive. There is no
mechanism in the source by which M6 could be anything else.

---

## 1. Exact shapes

### 1.1 The numbers

| quantity | symbol in source | value | where it comes from |
|---|---|---:|---|
| hidden dim | `kHidden`, `K0P6_H`, `K0P5_K` | **7168** | compile-time. `k0pf6gm_device_tile.hip:80`; harness `H = 7168` `.node/mps_remote_e004pf_k0pf_ab.py:103`; host contract "out must have BF16 cols 7168" `n2_phase2_gm_mps.cpp:614-616` |
| intermediate dim | `kInter` | **2048** | compile-time. harness `:103`; "a2q must have BF16-view cols 1024 (=2048 B rows)" `n2_phase2_gm_mps.cpp:603-604` |
| W13 output rows | `kW13N` | **4096** = 2·I (gate `[0,2048)`, up `[2048,4096)`) | compile-time. `.node/mps_n2_phase1_gm.cpp:47` |
| experts **per rank** | `E`, `kExperts` | **32** | descriptor word `K0P6_D_E=46` at runtime (`k0pf6gm_device_tile.hip:132,279`), guarded `E <= K0P6_MAXE = 64` (`:78,303`); harness pins `E = 32` `.node/mps_remote_e004pf_k0pf_ab.py:103`; GEMM host contract requires `[32*4096,3584]` / `[32,56,16]` (`.node/mps_n2_phase1_gm.cpp:534-541`, `n2_phase2_gm_mps.cpp:605-612`) |
| experts **global** | `E_GLOBAL` | **256** = 8 ranks × 32 | harness `:103` |
| top-k | `TOPK` | **8** | descriptor word 45, hard-guarded `TOPK != 8 ⇒ pperr 131072` (`k0pf6gm_device_tile.hip:303,309`); harness `:103` |
| world | `world` | **8** | `moe_host_abi.hpp:23`; guarded `world != 8` (`k0pf6gm_device_tile.hip:303`) |
| input tokens per rank | `T`, `K0P6_D_T` | **4096** | runtime env `K0_T`, default 4096 (`.node/mps_remote_e004pf_k0pf_ab.py:117`); descriptor comment "(= T = 4096 this campaign)" (`k0pf6gm_device_tile.hip:138`) |
| per-source segment stride | `MAXTOK` | **4096** | env `K0_MAXTOK` default 4096 (`.node/…_ab.py:130`), descriptor word 51 |
| receive-row **extent** | `T_ext = world·MAXTOK` | **32,768** | `k0pf6gm_device_tile.hip:312` |
| receive-row capacity | `T_LOC_MAX` | **40,960** | env `K0_T_LOC_MAX` (`.node/…_ab.py:118`) |
| sorted-row capacity | `PADMAX = world·T·TOPK + 31·E` | **263,136** | `.node/…_ab.py:123`; matches `DESIGN_MPS.md:155` |
| A2q/DQ2 row capacity | `_ROWCAP = world·MAXTOK·TOPK + 257·32 − 8` | **270,360** | `.node/…_ab.py:2627-2629` |
| G-stack width | `K0P6GM_G`, `kGM`, `N2GM_G` | **3** | pinned, `#error` on any other value (`k0pf6gm_device_tile.hip:152-157`); build flag `-DK0P6GM_G=3 -DN2GM_G=3` (`BUILDING.md:41`) |
| grid | — | **256 CTAs × 256 threads**, 4 waves/CTA | `__launch_bounds__(256,1)` (`k0pf6gm_device_tile.hip:275`); guard `nct != 256` (`:307`) |
| occupancy | — | **1 block / CU** | 256 ArchVGPR + 256 AGPR and 155,428 B LDS (`PROVENANCE.md:106-110`, `KERNEL_BOTTLENECKS.md:20-23`) |

### 1.2 Dtypes

| tensor | dtype | shape | scale companion |
|---|---|---|---|
| model input `hidden` | **BF16** | `[T, 7168]` | — (M1 quantizes it) |
| dispatched activation `a_dst` / `A_bytes` | **FP8 e4m3 (OCP)** | `[T_ext, 7168]` bytes | `sc_dst` FP32, **group-major** `input_scale[k128·T + token]`, 56 groups (`.node/mps_n2_phase1_gm.cpp:187-193`) |
| `W13` (gate ‖ up) | **FP8 e4m3**, AITER-preshuffled | `[32, 4096, 7168]` | `S13` FP32 `[32, 32, 56]` (`.node/mps_n2_phase1_gm.cpp:96-97,538-541`) |
| `A2q` (M6→M7 handoff) | **FP8 e4m3** | `[rowcap, 2048]` | `DQ2` FP32 `[rowcap, 16]`, one per (row, K128 group) |
| `W2` (down) | **FP8 e4m3**, AITER-preshuffled | `[32, 7168, 2048]` | `S2` FP32 `[32, 56, 16]` (`n2_phase2_gm_mps.cpp:201-202`) |
| `part` (M7 output) | **BF16** | `[T_ext, 7168]` | accumulated via `global_atomic_pk_add_bf16` |
| `out` | **BF16** | `[T, 7168]` | — |

MFMA arithmetic is **FP32 accumulate on FP8 inputs**: `mma_ABt(p, af, bf, p)`
with `zero(p)` per call (a fresh FP32 partial, `.node/mps_n2_phase1_gm.cpp:298-300`,
`n2_phase2_gm_mps.cpp:453-455`); the dequant scale is folded in FP32 afterwards.

### 1.3 How many tokens a rank processes after dispatch

Three different counts, all needed below:

1. **Receive rows** `T_loc` — distinct (token, this-rank) pairs.
   Measured **≈ 21,816** (`exp_01_nil_fault/discriminators.md:66` `T_loc=21816`;
   `KERNEL_BOTTLENECKS.md:138`).
   *Derivation check:* a token picks 8 of 256 experts; P(rank not hit)
   `= ∏_{i=0}^{7}(224−i)/(256−i) = 0.33817`, so P(hit) `= 0.66183`, and
   `8 sources × 4096 tokens × 0.66183 = 21,687` — within 0.6% of the measured
   21,816. Extent (hard bound) is `T_ext = 32,768`.
2. **Sorted rows** — (receive-row, local-expert) pairs, the GEMM's M dimension.
   `8 ranks × 4096 tokens × 8 topk / 8 ranks = ` **32,768** exactly under uniform
   routing. Average local experts per receive row `= 32,768 / 21,816 = 1.50`
   (`LESSONS.md:550` "averages only ~1.5 on this workload").
3. **Padded sorted rows** `nvi[0]` — after per-expert padding to 32-row blocks.
   **≈ 33,440**, i.e. **1,045 32-blocks**. Derived from the measured
   `events_total = (nvi[0] >> 5)·16 ≈ 16,720`
   (`exp_09_arrival_count/design.md:31`) and cross-checked by the arrival total
   `1,045 × 512 = 535,040` (`KERNEL_BOTTLENECKS.md:175-179`).
   Padding overhead = `33,440 − 32,768 = 672` rows = 21/expert.

**Tiles.** M4 builds `tile_desc` on device, one entry per expert-aligned tile of
up to `G=3` consecutive same-expert 32-blocks, packed `(b0<<4)|gcount`
(`k0pf6gm_device_tile.hip:838-861`):

```
num_tiles = Σ_e ceil(nsb_e / 3),  Σ_e nsb_e = 1,045,  E = 32
          ≈ 1045/3 + 32·E[ceil overhead]  ≈ 348.3 + 10.7  ≈ 355     (range 349–380)
```

**UNRESOLVED:** `num_tiles` is device-computed and already stored in
`num_tiles[0]` (`k0pf6gm_device_tile.hip:856`) but is never printed by the
harness. Every task/byte/FLOP total below scales linearly in it. **One host
print of `num_tiles[0]` and `nvi[0]` removes the last ±4% of uncertainty from
this whole document.** All numbers below use `num_tiles = 355`.

Derived work counts:

| | formula | value |
|---|---|---:|
| M6 tasks | `num_tiles × 8` | **2,840** |
| M7 tasks | `num_tiles × 16` | **5,680** |
| M6 tasks per CTA | `/256` | **11.09** |
| M7 tasks per CTA | `/256` | **22.19** |
| MFMA rows (incl. ragged-tile pad) | `num_tiles × 96` | **34,080** |

---

## 2. M6 anatomy — `n2_phase1_gm.cpp`, called at `k0pf6gm_device_tile.hip:894-906`

Local snapshot of the (upstream, read-only) body: `.node/mps_n2_phase1_gm.cpp`.
Provenance: it is the donor `n2_phase1_gm.cpp` — its 22 KB sibling
`.node/mps_n2_phase2_gm.cpp` is byte-comparable to the donor
`n2_phase2_gm.cpp` sha256 `7d8beb03…` that `n2_phase2_gm_mps.cpp:1-11` names,
not to the 29 KB MPS vendored copy in the repo.

### 2.1 The math

Per expert `e` with `n_e` padded sorted rows:

```
A_e [n_e × 7168] fp8   ×   W13_e^T [7168 × 4096] fp8   →   H_e [n_e × 4096] fp32
                                       ↑ 4096 = 2048 gate ‖ 2048 up
then, in registers:  a2[r, j] = SiLU(gate[r, j]) · up[r, j]      j ∈ [0, 2048)
then, per (row, K128 group of 2048): amax → FP8 e4m3 quant → A2q, DQ2
```

Summed over the 32 local experts, using the tiled M actually executed
(`num_tiles × 96 = 34,080`; the MFMA loop has **no `gcount` guard** — it runs all
3 sub-blocks of every tile, `.node/mps_n2_phase1_gm.cpp:281-308`, only the
epilogue store checks liveness at `:443`):

```
[34,080 × 7168] × [7168 × 4096] → [34,080 × 4096]
```

### 2.2 Tile decomposition and task assignment

| item | value | source |
|---|---|---|
| task | `(tile, intermediate chunk g ∈ [0,8))` | `.node/mps_n2_phase1_gm.cpp:136-137,161-162` |
| output tile per task | `[32·G = 96 rows] × [256 gate + 256 up = 512 cols]` | `kChunkCols = kInter/kChunks = 256` `:50-51`; `n_gate/n_up` `:227-228` |
| per-wave slice | 64 intermediate cols → `kColTiles = 4` N16 tiles, ×2 for gate/up | `kWaveCols = 64`, `kColTiles = 4` `:52-53` |
| assignment | **static grid stride, no ticket, no persistence beyond the stride**: `for (task = blockIdx.x; task < num_tasks; task += kCTAs)` | `:156` |
| K-loop depth | `kKGroups = 7168/128 = 56` K128 steps, unrolled by 2 → 28 iterations × 2 halves | `:55,318-350` |
| MFMA instruction | one `mma_ABt` per `(sb, m, gu, c)` = **1 v_mfma_f32_16x16x128_f8f6f4** | `:278-309` |
| MFMA per K-step **per wave** | `kGM(3) × m(2) × gu(2) × kColTiles(4)` = **48** | `:282-305` |
| MFMA per K-loop iteration per wave | **96** (2 halves) | matches `README.md:99` "96 MFMAs per phase-1 K-loop" and the ISA census `96+84=180` (`PROVENANCE.md:110`) |
| MFMA per task per wave | `48 × 56` = **2,688** | |
| MFMA per task per **CTA** | `× 4 waves` = **10,752** | |

**Fragment shapes** (confirmed by the scheduling hints, not assumed):
`read_a` loads 2 × `float4` = **32 B/lane = 2048 B/wave = a 16×128 fp8 A-fragment**
(`:256-259`); `load_bfrag` is **2 VMEM instructions of 16 B/lane** — the VMEM
hint at `:332` is `17 = 1 (load_a) + 8 fragments × 2` at `kGM=1`, which pins it.
So each MFMA consumes 2048 B of A-fragment and 2048 B of B-fragment and produces
`2·16·16·128 = 65,536 FLOP`.

### 2.3 Data movement

Per task, per CTA (all figures are **vector-memory requests**, i.e. L1 misses
reaching L2):

| stream | per K-step per CTA | × 56 K-steps | note |
|---|---:|---:|---|
| `W13` | 4 waves × 8 fragments × 2048 B = **65,536 B** | **3,670,016 B** | = 512 cols × 7168 K, exactly the task's weight slab; **independent of G** |
| `A` (`a_dst`) | 3 sub-blocks × 256 thr × 16 B = **12,288 B** | **688,128 B** | = 96 rows × 7168 B |
| `A2q` store | — | **24,576 B** | 96 rows × 256 B (`:444-452`) |
| `DQ2` store | — | **3,072 B** | 4-way redundant across waves (`:456-471`) |
| **per task** | **77,824 B** | **4,385,792 B** | |

Grid totals for one rank, one epoch (`× 2,840 tasks`):

```
W13 requests   2,840 × 3,670,016 = 10.423 GB
A   requests   2,840 ×   688,128 =  1.954 GB
A2q writes     2,840 ×    24,576 =  0.070 GB
DQ2 writes     2,840 ×     3,072 =  0.009 GB
─────────────────────────────────────────────
TOTAL vector-memory traffic       = 12.456 GB
```

**Achieved L2-request bandwidth:  12.456 GB / 2,539 µs = 4.91 TB/s.**

Compulsory (cold) footprint — every byte that must reach HBM at least once:

```
W13 unique   32 × 4096 × 7168      =   939.5 MB
A live       21,816 × 7168         =   156.4 MB   (extent 234.9 MB)
A2q written  33,440 × 2048         =    68.5 MB
DQ2 written  33,440 × 64           =     2.1 MB
──────────────────────────────────────────────
TOTAL                              = 1,166.5 MB
```

**Achieved HBM bandwidth:  1.167 GB / 2,539 µs = 460 GB/s = 5.7% of 8 TB/s.**

**Does the weight working set fit the 256 MB Infinity Cache?**

- Whole-epoch `W13` = **939.5 MB — no**, 3.7× over.
- **Instantaneous** window: 256 CTAs / 8 tasks-per-tile = **32 tiles in flight**
  ≈ `32 / 11.1 tiles-per-expert` ≈ **2.9 experts** × 29.36 MB/expert =
  **84 MB — yes**, comfortably. So each `W13` byte should be fetched from HBM
  once and served ~11× (the 11 tiles of its expert) from the LLC. That is
  consistent with the 460 GB/s HBM number vs the 4.91 TB/s request number.
- **Per-XCD L2 (4 MB)**: workgroups go round-robin to XCDs with chunk size 1
  (`xcd = bid % 8`, `moe_mps_adapter.cuh:93-96`), and `task = tile·8 + g`, so
  `task ≡ bid (mod 8) ⇒ g = bid mod 8`. **XCD *x* only ever touches chunk
  `g = x`.** Its working set is `2.9 experts × 3.67 MB = 10.6 MB` against a
  **4 MB L2 — does not fit.** Most of the 4.91 TB/s therefore leaves the XCD.

### 2.4 FLOPs and achieved TFLOPs

```
MFMA instructions = 2,840 tasks × 4 waves × 2,688 = 30,535,680
FLOP              = 30,535,680 × 65,536           = 2.001×10¹² = 2.001 TFLOP
achieved          = 2.001 TFLOP / 2,539 µs        = 788 TFLOPS
```

**Peak used: 4.6 PFLOPS dense FP8/MXFP8 for MI350X.** Source: AMD product page
and the ROCm workload-optimization table (MI350X FP8 = 4.6 PF, BF16 = 2.3 PF,
4.6 PF only *with structured sparsity* for BF16; 256 CU, 4 MB L2/XCD, 256 MB
Infinity Cache, 8 TB/s HBM3E). The BF16 2.3 PF matches `CLAUDE.md`'s figure, and
CDNA4 runs FP8 at exactly 2× the 16-bit rate.

```
788 TFLOPS / 4,600 TFLOPS = 17.1% of FP8 peak
```

Cross-check from the other direction: at 2.2 GHz shader clock
(`exp_05/stage0_attribution.md:32-33`) and 32 cycles per
`v_mfma_f32_16x16x128_f8f6f4` (derived: `4.6e15 / (256 CU × 4 SIMD × 2.4e9) =
2,035 FLOP/SIMD/clk`; `65,536 / 2,035 = 32.2`), the MFMA-issue floor is

```
30,535,680 / 1024 SIMDs × 32 cycles = 954,240 cycles = 434 µs
434 / 2,539 = 17.1% MFMA duty cycle          ← identical, both routes agree
```

### 2.5 So what limits M6?

**Answer: the L2/LLC path. M6 is memory-throughput-bound at the tile's
arithmetic intensity, and it is structurally incapable of being
arithmetic-bound at `G=3`.**

The tile intensity is the classic `2MN/(M+N)` with `M = 32G = 96`, `N = 512`:

```
2 × 96 × 512 / (96 + 512) = 161.7 FLOP per byte of vector-memory request
```

which matches the measured totals exactly (`2.001e12 / 12.456e9 = 160.6`).
Therefore:

```
bandwidth needed to reach FP8 peak = 4.6 PFLOPS / 161.7 FLOP/B = 28.4 TB/s
bandwidth actually achieved                                     =  4.91 TB/s
```

**No memory path on this part delivers 28.4 TB/s of L2-request bandwidth to
256 CUs, so M6 cannot be arithmetic-bound at `G=3` regardless of how well it is
scheduled.** Its ceiling is set by intensity, and intensity is set by the tile
shape.

Ruling out the alternatives with the numbers above:

| hypothesis | test | verdict |
|---|---|---|
| **arithmetic-bound** | 17.1% of FP8 peak; MFMA issue occupies 1,536 of 8,992 cycles per K-step | **NO** |
| **HBM-bound** | 460 GB/s compulsory = 5.7% of 8 TB/s; intensity 1,715 FLOP/B vs machine balance 4.6 PF/8 TB/s = 575 FLOP/B, i.e. 3.0× compute-rich against HBM | **NO** |
| **LLC-capacity-bound** | instantaneous weight window 84 MB of 256 MB | **NO** |
| **per-XCD L2-capacity-bound** | 10.6 MB of 4 MB — **exceeded** | **contributing**: forces the whole 4.91 TB/s onto the XCD↔Infinity-Cache path |
| **dependency-bound** | M6 has **no** cross-CTA dependency, **no** readiness poll, **no** peer traffic, and every input is published by the M5 grid barrier (`k0pf6gm_device_tile.hip:889`); `STATUS.md:293-295` "M6 has no readiness poll of any kind" | **NO** |
| **latency-bound (pure)** | 8,992 cycles per K-step = 4.1 µs, far beyond any memory latency; the loop keeps 19 VMEM instructions per wave in flight | **NO** as a pure latency stall, **YES** as bandwidth-at-the-MLP-limit: 8.65 B/clk/CU sustained ≈ (bytes in flight)/(latency) |
| **L2/LLC-throughput-bound** | 4.91 TB/s aggregate, intensity 5.8× below what peak needs | **YES — this is the limiter** |

**And explicitly on "removing 24% of the CTAs costs 0.5%":** *no explanation is
consistent with that, because the experiment did not do it.* The 62 reserved
CTAs execute the entire M6 task loop (§0). `plan+M6` was invariant because
`plan+M6` was unchanged. The three hypotheses that would each be *falsifiable*
by an actual M6 CTA sweep, with their predictions:

| if M6 is… | predicted `M6(194 CTAs) / M6(256 CTAs)` |
|---|---:|
| per-CU throughput limited | **1.32×** (256/194) |
| shared-bandwidth-ceiling limited | **1.00×** |
| like M7 (measured: 254→192 CTAs gives 1.2547× against an ideal 1.3229×, i.e. 76% of linear) | **1.24×** |

**Prediction on the record: an honest M6 CTA sweep will land in 1.20–1.32×, not
1.005×.** The cheapest way to run it is a phase-1 twin of the phase-2
`N2GM_TASK_START` / `N2GM_TASK_STRIDE` macros (§6).

---

## 3. M7 anatomy — `n2_phase2_gm_mps.cpp`, called at `k0pf6gm_device_tile.hip:909-921`

### 3.1 The math

```
A2_e [n_e × 2048] fp8   ×   W2_e^T [2048 × 7168] fp8   →   out_e [n_e × 7168] fp32
→ × sorted_weight → bf16 → global_atomic_pk_add_bf16 into part[xtok, :]
```

Summed over the rank: `[34,080 × 2048] × [2048 × 7168] → [34,080 × 7168]`.

### 3.2 Tiles and tasks

| item | value | source |
|---|---|---|
| task | `(tile, output n-chunk nc ∈ [0,16))` | `n2_phase2_gm_mps.cpp:283-284,331-332` |
| output tile per task | `[96 rows] × [kChunkN = 7168/16 = 448 cols]` | `:67-68` |
| per-wave slice | `kWaveTiles = 448/16/4 = 7` N16 tiles = 112 cols | `:69-70` |
| assignment | grid stride, macro-parameterised: `task = N2GM_TASK_START; task += N2GM_TASK_STRIDE` — defaults `blockIdx.x` / `kCTAs`, overridden by MPS to `compute_id_of(...)` / `(256 − C)` | `:288-294,326-327`; `k0pf6gm_device_tile_mps.hip:273-285,419-420` |
| K-loop depth | `kKGroups2 = 2048/128 = 16`, unrolled by 2 → 8 iterations | `:475` |
| MFMA per K-step per wave | `kGM(3) × kWaveTiles(7) × m(2)` = **42** | `:443-465` |
| MFMA per K-loop iteration per wave | **84** — matches the `96+84=180` ISA census | `PROVENANCE.md:110` |
| MFMA per task per wave | `42 × 16` = **672** | |

### 3.3 Bytes, FLOPs, achieved rates

Per task per CTA:

| stream | per K-step | × 16 | note |
|---|---:|---:|---|
| `W2` | 4 waves × 7 fragments × 2048 B = **57,344 B** | **917,504 B** | = 448 cols × 2048 K |
| `A2` | 3 × 256 × 16 = **12,288 B** | **196,608 B** | = 96 rows × 2048 B |
| `part` atomics | — | **86,016 B** | 96 × 448 × 2 B (`:168-180`) |
| **per task** | **69,632 B** (loop) | **1,200,128 B** | |

```
W2 requests     5,680 × 917,504 = 5.211 GB
A2 requests     5,680 × 196,608 = 1.117 GB
part RMW        5,680 ×  86,016 = 0.489 GB
────────────────────────────────────────────
TOTAL                           = 6.817 GB   →  6.817 GB / 1,586 µs = 4.30 TB/s

compulsory: W2 469.8 MB + A2 68.5 MB + part 312.7 MB distinct = 851 MB → 536 GB/s = 6.7% of 8 TB/s
            (counting the atomic as read+write: 1.516 GB → 956 GB/s = 12.0%)

FLOP = 5,680 × 4 × 672 × 65,536 = 1.001×10¹² = 1.001 TFLOP
     → 1.001 TFLOP / 1,586 µs = 631 TFLOPS = 13.7% of 4.6 PFLOPS
MFMA floor: 15,267,840 / 1024 × 32 cycles = 477,120 cyc = 217 µs → 13.7% duty  ✓
```

Tile intensity `2·96·448/(96+448) = 158.1 FLOP/B` — **within 2.2% of M6's 161.7.**

### 3.4 Why M7 is CTA-linear and M6 "isn't" — with numbers

M7's CTA-linearity is real and directly measured: the phase-2 task stride is
`256 − C` (`k0pf6gm_device_tile_mps.hip:273-276,420`), so mode 0 genuinely runs
M7 on fewer CTAs.

```
254 compute CTAs → M7 = 1,609.7 µs
192 compute CTAs → M7 = 2,019.7 µs          exp_05/stage0_attribution.md:49-50
ratio 1.2547 against an ideal 254/192 = 1.3229  →  76% of linear
```

**M6's "insensitivity" is not a measurement** (§0). The only structural
differences between the two loops are:

| | M6 | M7 |
|---|---:|---:|
| tile | 96 × 512 | 96 × 448 |
| intensity | 161.7 FLOP/B | 158.1 FLOP/B |
| K-steps per task | 56 | 16 |
| MFMA/wave/K-step | 48 | 42 |
| bytes/CTA/K-step | 77,824 | 69,632 |
| achieved B/clk/CU | **8.65** | **7.08** |
| MFMA duty | **17.1%** | **13.7%** |
| **per-XCD weight working set** | 2.9 experts × 3.67 MB = **10.6 MB** (> 4 MB L2) | 1.44 experts × 2 nc-slices × 0.92 MB = **2.64 MB** (< 4 MB L2) |

That last row is the one real structural asymmetry, and it is deliberate:
`nc → XCD (nc mod 8)` is stable across blocks because `16 mod 8 == 0`, so every
block of an expert re-reads its `W2` slice through the *same* XCD's L2
(`n2_phase2_gm_mps.cpp:29-31`). **M7's weight stream is L2-resident; M6's is
not.** Ironically M6 still achieves the *higher* B/clk and the *higher* MFMA
duty — so this asymmetry does not make M6 the slower loop per byte; it makes M6
the loop whose traffic lands on the Infinity Cache instead of the L2.

**The closing arithmetic.** If both phases are the same loop at the same
intensity, their times should be in the ratio of their bytes:

```
predicted  M6/M7 = 12.456 GB / 6.817 GB = 1.83     (or by K-depth × chunk count: 3.5 × 0.5 = 1.75)
measured   M6/M7 = 2,539 / 1,586        = 1.60
```

Within 9–13%. **There is no separate "M6 mystery". M6 is M7 with 1.8× the bytes,
and it runs 1.6× as long.**

---

## 4. The M6 → M7 dependency, in detail

### 4.1 Which M6 outputs does one M7 task consume? The exact index relation.

**M6 task `(tile, g)` writes** (`.node/mps_n2_phase1_gm.cpp:438-472`):

```
A2q[(b0+sb)·32 + cr] [ 256·g + 32·cc  ..  +32 )      cr∈[0,32), cc∈[0,8), sb<gcount
DQ2[(b0+sb)·32 + 16m + r] [ 2g + kb ]                m∈{0,1}, r∈[0,16), kb∈{0,1}
```

i.e. **A2q columns `[256g, 256g+256)` and DQ2 columns `{2g, 2g+1}` for the 96
rows of that tile.**

**M7 task `(tile', nc)` reads** (`n2_phase2_gm_mps.cpp:380-386,404-410,366-376`):

```
A2q[(b0'+sb)·32 + a_row] [ 128·k .. +128 )   for every k ∈ [0,16)   → all 2048 columns
DQ2[(b0'+sb)·32 + i] [k]                     for every k ∈ [0,16)   → all 16 columns
W2 [e, 448·nc .. +448, : ]                                          → its own N-slice
```

**Therefore:**

> **M7 task `(tile, nc)` depends on exactly the 8 M6 tasks `(tile, g=0..7)` — the
> same tile, all chunks — and on nothing else. No other tile, no other expert,
> no neighbouring row.**

And one level finer, which the current protocol throws away:

> **M6 chunk `g` produces exactly K128-group pair `(2g, 2g+1)` of M7's K-loop**
> (256 A2q bytes = 2 × 128, and DQ2 columns `2g`, `2g+1`). So M7's K-step `k`
> needs only M6 chunk `g = k/2` of its own tile. The dependency is a
> **K-wise pipeline**, not a phase barrier.

### 4.2 Is there a grid barrier between M6 and M7?

**No.** Neither port has one.

- Parity port: phase 1 returns at `k0pf6gm_device_tile.hip:906`, phase 2 is
  called at `:910`. The only statement between them is
  `if (debug_stop == 5) return;` (`:907`).
- MPS port: phase 1 at `:1330`, timestamp at `:1336-1344`, role predicate at
  `:1367-1370`, phase 2 at `:1371`. No barrier.
- The last grid barrier before the GEMMs is **M5** (`:889` parity, `:1301`
  MPS); the next one is **M7.5** (`:959-964` parity), *after* M7.

The dependency is enforced **per 32-block, by a counter**:

- **Producer** — `k0p6_a2_arrive` (`k0pf6gm_device_tile.hip:187-194`,
  `k0pf6gm_device_tile_mps.hip:227-234`), fired from
  `N2GM_TASK_DONE_HOOK` per live sub-block at
  `.node/mps_n2_phase1_gm.cpp:481-488`: tid0 does
  `arrive_local_count<true>` = **agent release fence** (`moe_hk_adapter.cuh:92-97`
  → `thread_release<agent>` → `__builtin_amdgcn_fence(RELEASE,"agent")`,
  `include/cdna4/ops/group/distributed/sync.cuh:88-100`) then a **relaxed
  agent `fetch_add`** on `a2_done[b]`.
- **Consumer** — `k0p6_a2_wait` (`k0pf6gm_device_tile.hip:196-211`), fired from
  `N2GM_TASK_WAIT_HOOK` per live sub-block at
  `n2_phase2_gm_mps.cpp:336-344`, **at the very top of the task, before any LDS
  fill**: tid0 spins `poll_local_count<8>(a2_done + b, …)` — target **8**, the
  8 chunk-tasks of the owning tile — then the whole CTA does
  `acquire_payload_agent()` = `cta_acquire<agent>` = `__syncthreads()` +
  agent acquire fence (`moe_hk_adapter.cuh:88-90`, `sync.cuh:176-181`).
- `a2_done` is zeroed grid-wide in M0 (`k0pf6gm_device_tile.hip:344-350`).
- Exactness: expert-aligned tiling gives 1:1 tile ownership of every 32-block,
  so `a2_done[b]` is incremented **exactly 8 times, by its owning tile's 8
  chunk-tasks only** — the pad sub-blocks deliberately do not fire
  (`.node/mps_n2_phase1_gm.cpp:475-488`).

**So the answer to "full grid barrier or per-tile dependency" is: per-tile
(per-32-block), already.**

### 4.3 Could an M7 task begin as soon as its M6 tiles are done?

**It already may — and it never does.** What serialises the phases is **CTA
program order, not synchronisation**: a CTA runs its *entire* phase-1 task
stripe (`for task = blockIdx.x; task < num_tasks; task += 256`,
`.node/mps_n2_phase1_gm.cpp:156`) and only then enters the phase-2 body. All 256
CTAs carry near-identical M6 work, so they all leave M6 within a few
microseconds of each other, and by the time any CTA polls `a2_done[b]` the
counter is long since 8. `exp_10` confirms the same pattern on the dispatch
side; here it means the poll is free and load-bearing only for correctness.

**What already holds (nothing to build):**

| requirement | status |
|---|---|
| a per-tile readiness signal exists | ✅ `a2_done[b]`, target 8 |
| the release/acquire pair is correct | ✅ agent release at producer, agent acquire at consumer (§4.2) |
| the consumer's wait is at task top, before any work | ✅ `n2_phase2_gm_mps.cpp:336-344` |
| no grid barrier stands in the way | ✅ §4.2 |
| the handoff buffer has no slot-reuse hazard | ✅ `A2q` is capacity-sized (§4.4) — every tile owns permanent rows, so **no credit/lifetime protocol is needed** |
| the LDS for both phases is already co-resident | ✅ §4.5 — interleaving costs **zero** extra LDS |
| M7's task loop is already stride-parameterised | ✅ `N2GM_TASK_START` / `N2GM_TASK_STRIDE`, `n2_phase2_gm_mps.cpp:288-294` |

**What is missing, exactly:**

1. **An interleaved or role-split task schedule.** Today the two task spaces are
   two sequential loops in one CTA. To overlap you need either
   (a) a **fused task list** where a CTA picks the next ready item from
   `{M6 tasks} ∪ {M7 tasks with a2_done == 8}` — which needs a dynamic ticket
   plus a cheap readiness probe, or
   (b) a **role split at M6** (some CTAs run phase 1, others poll for ready
   tiles and run phase 2) — the CTA-level shape `CLAUDE.md` mandates, applied at
   the M6/M7 boundary instead of the M7/M8 boundary.
2. **Register/ISA feasibility of (a).** `PROVENANCE.md:98-102` and
   `DESIGN_MPS.md:119-125` both record that *any* value crossing the M6→M7
   boundary costs VGPR-pair spill slots at the phase-2 pointer peak (the
   finish-order ticket was rolled back for exactly +24 B/lane). A fused loop
   carries loop state across both bodies and must be resource-gated first.
   Option (b) avoids this — `blockIdx.x` is free.
3. **Nothing else.** In particular there is no barrier to delete, no buffer to
   double-buffer, and no ordering edge to add.

**The finer K-wise version** (M7's K-step `k` waits only on chunk `k/2`) would
additionally need `a2_done` widened from `[b]` to `[b][8]`, and a wait *inside*
the K-loop — which the current design forbids on the MFMA wave and which
`exp_05`/`exp_20` measured to be exactly the class of thing (scattered
acq_rel RMWs) that inflates a concurrent GEMM by 37–57%. **Do not start there.**

### 4.4 The intermediate buffer

| | value | source |
|---|---|---|
| `A2q` | FP8, `[_ROWCAP = 270,360 × 2048 B]` = **553.7 MB allocated**, ~68.5 MB used (33,440 rows) | `.node/mps_remote_e004pf_k0pf_ab.py:2627-2629`; descriptor slot 27 `K0P6_D_A2Q` |
| `DQ2` | FP32, `[270,360 × 16]` = **17.3 MB** | same |
| sizing rule | `rowcap ≥ sorted_token_ids capacity (PADMAX = 263,136)` | `.node/mps_n2_phase1_gm.cpp:549-551` |
| buffering | **single-buffered, whole-phase** — one permanent row per sorted row, no wraparound, no epoch tag, no zeroing | `.node/mps_n2_phase1_gm.cpp:19-21`: "Both are fully written for every row < nvi[0] on every call… so phase 2 never reads a stale byte — no zeroing, no epoch state, route-swap safe by construction" |
| M7's bound on it | `a2_num_records = nvi[0] × 2048`, a hardware buffer-descriptor bound | `n2_phase2_gm_mps.cpp:302-306` |

The whole-phase sizing is what makes fine-grained M6→M7 overlap *cheap*: there
is no slot to recycle, so no credit protocol, no `bounded_wait_slot_reusable`,
no retirement edge. It costs 571 MB of HBM to buy that.

### 4.5 Where the nonlinearity sits — and a free-LDS bonus

**SiLU/GLU is in the M6 epilogue, in registers**
(`.node/mps_n2_phase1_gm.cpp:370-385`):

```cpp
const float gv = gf[t];
const float hv = (gv / (1.0f + __expf(-gv))) * uf[t];   // :379  SiLU(gate)·up
const float h  = live[m][t] ? hv : 0.0f;
```

It is fused with the **dynamic per-(row, K128) amax** (`:386-406`, a 4-step
`__shfl_xor` butterfly then an LDS `atomicMax`) and the **FP8 e4m3 requant**
(`:410-428`). Nothing nonlinear happens in M7's prologue and there is no
separate pass. Note "THE EPSILON IS NOT OPTIONAL (rank 2 has empty experts;
448/0 = NaN)" at `:409`.

**LDS bonus, checkable from the resource report.** Phase 1 declares 74,240 B of
`__shared__` (`:116-124`: `tok_lds` 384 + `ascale_lds` 21,504 + `b1s_lds` 896 +
`amax_lds` 768 + `a2_lds` 26,112 + `a_lds` 24,576); phase 2 declares 49,664 B
(`n2_phase2_gm_mps.cpp:218-225`: 384 + 384 + 1,792 + 6,144 + 24,576 + 16,384);
the megakernel declares 30,500 B (`k0pf6gm_device_tile.hip:287-294`). Sum =
**154,404 B against a measured 155,428 B** (`PROVENANCE.md:107-108`) — a 1,024 B
alignment delta. **They are summed, not unioned: both phase LDS blocks are
already simultaneously allocated for the whole launch.** Interleaving M6 and M7
tasks inside one CTA therefore needs **no additional LDS**.

---

## 5. What the M6 CTAs are actually waiting on

Complete stall census for one M6 task. Line numbers are
`.node/mps_n2_phase1_gm.cpp`.

### 5.1 Prologue — `:161-205`

| line | what | stall class |
|---|---|---|
| `:168-173` | `tok_lds` fill — global reads of `sorted_ids` | VMEM |
| `:174-176` | `amax_lds` zero | LDS |
| `:177-183` | `b1s_lds` fill — 224 global reads of `S13` | VMEM |
| **`:184`** | **`__syncthreads()`** | **CTA barrier** |
| `:187-194` | **`ascale_lds` fill — the scatter-gather.** `A_scale[k·T + token]` is **group-major** with `T = 32,768`, so consecutive `k` for one token are 131,072 B apart and the 96 tokens of a tile are scattered across the whole range. **56 × 96 = 5,376 distinct 64 B cache lines fetched to deliver 21,504 B of scales — 16× line amplification, 344 KB per task**, and it sits between two barriers so **no MFMA overlaps it**. 21 dependent iterations per thread. | VMEM latency + wasted bandwidth |
| **`:195`** | **`__syncthreads()`** | **CTA barrier** |
| `:198-205` | `a_voff0[]` from `tok_lds` | LDS |

Cost of the gather: `344 KB × 2,840 tasks = 977 MB` of extra line traffic (+7.8%
on the 12.456 GB total); at the phase's own 8.65 B/clk/CU that is
`344,064 / 8.65 = 39,776` cycles per task = **~200 µs of the 2,539 µs, ≈ 8%**.
Compulsory part is only `355 tiles × 344 KB = 122 MB` (the 8 chunk-tasks of a
tile re-read the same lines and should hit L2).
**This is a concrete, isolated, fixable item: transpose `A_scale` to
row-major, or hoist the per-tile scale gather out of the 8 chunk-tasks.**

### 5.2 Pipeline prologue — `:312-316`

`load_a(0,0)`, `load_a(1,1)`, `load_b(0,0)`, `store_a(0,0)`, then
**`lds_cta_barrier()`** at `:316`.

### 5.3 The K-loop — `:318-350` (28 iterations × 2 halves = 56 K128 steps)

Each half is:

```cpp
read_a(0);            // :326  12 × ds_read_b128  (kGM 3 × m 2 × 2 float4)   → lgkmcnt
load_a(ka2, 0);       // :327   3 × buffer_load_b128                        → vmcnt
load_b(kb1, 1);       // :328  16 × 16 B/lane loads (8 fragments × 2)       → vmcnt
mfma_k(k, 0);         // :329  48 × v_mfma_f32_16x16x128_f8f6f4
                      //       + 6 × ds_read_b128 (ascale_lds) + 2 (b1s_lds)
store_a(1, 1);        // :330   3 × ds_write_b128
__builtin_amdgcn_sched_group_barrier(0x100,  4, 0);  // :331  DS read
__builtin_amdgcn_sched_group_barrier(0x020, 17, 0);  // :332  VMEM read
__builtin_amdgcn_sched_group_barrier(0x008, 16, 0);  // :333  MFMA
__builtin_amdgcn_sched_group_barrier(0x200,  1, 0);  // :334  DS write
n2_completion_observation_probe();                   // :335  NO-OP in this build
lds_cta_barrier();                                   // :336  CTA BARRIER
```

**Stall sites, in order of size:**

1. **Compiler-inserted `s_waitcnt vmcnt(N)`** before `store_a` (needs the
   previous half's `load_a`) and before `mfma_k` (needs the previous half's
   `load_b`). There is no explicit `vmcnt` in the source; these are the real
   cost. **The software pipeline is exactly ONE K-step deep** for both A and B:
   `load_b(k+1)` issues in half 0 and is consumed in half 1; `load_a(k+2)`
   issues in half 0 and is stored to LDS in half 1. Latency tolerance per K-step
   = the MFMA time of one K-step = `48 × 32 = 1,536` cycles at `G=3`
   (only 512 cycles at `G=1` — **this is the mechanism by which G-stacking won**).
2. **`lds_cta_barrier()` twice per iteration** (`:336`, `:349`) — **56 CTA
   barriers per task**, 621 per CTA per epoch.
3. **`s_waitcnt lgkmcnt(0)`** before `mfma_k` for the 12 A ds_reads plus 6
   `ascale_lds` and 2 `b1s_lds` reads = 20 LDS reads and 3 LDS writes per lane
   per K-step. At 128 B/clk/SIMD this is ~170–200 cycles of a 8,992-cycle
   K-step — **not** the bottleneck.
4. **`n2_completion_observation_probe()`** (`:335`, `:348`) — an *optional*
   `s_waitcnt vmcnt(0)` behind `N2_FORCE_VMCNT0`, which defaults to 0 (`:70-73`)
   and is **not** defined by the build flags (`BUILDING.md:39-45,85-91`).
   **Compiled out. There is no `vmcnt(0)` inside M6's K-loop.**

**A real scheduling defect, worth one line to fix.** The four
`sched_group_barrier` counts at `:331-334` and `:344-347` are **hardcoded for
`kGM = 1`** and are *not* scaled by `kGM`, unlike phase 2 which scales three of
its four (`n2_phase2_gm_mps.cpp:490-493`: `4*kGM`, `8+7*kGM`, `14*kGM`, `kGM`).
At the shipped `G=3` the actual instruction mix per half is **12 DS-read /
19 VMEM / 48 MFMA / 3 DS-write**, but the hints request **4 / 17 / 16 / 1** — so
the scheduler is steered for only ⅓ of the DS reads, ⅓ of the MFMA and ⅓ of the
DS writes, and the remaining `8 + 2 + 32 + 2` instructions are scheduled freely.
**Candidate experiment: scale phase-1's hints by `kGM` exactly as phase 2 does.
Pure schedule change, no math, no resources, no protocol.**

### 5.4 Epilogue — `:356-430`, run **3× per task** (once per sub-block)

| line | what | stall class |
|---|---|---|
| `:370-385` | SiLU + `fabsf` running max — 32 `__expf` per lane per sub-block | VALU / transcendental (¼ rate) |
| `:386-395` | 4-step `__shfl_xor` butterfly × 8 = 32 shuffles | VALU |
| `:396-406` | 8 LDS `atomicMax` (r == 0 lanes only) | LDS atomic |
| **`:407`** | **`__syncthreads()`** | **CTA barrier** |
| `:410-428` | dequant + **32 byte-wide `ds_write_b8` per lane** into `a2_lds[row][col]` (stride 272 B, `:59`) | LDS, byte-granular |
| **`:429`** | **`__syncthreads()`** | **CTA barrier** |

→ **6 CTA barriers per task from the epilogue alone.**

### 5.5 Store-out and task end — `:438-488`

| line | what |
|---|---|
| `:444-452` | 2 × `uint4` global stores per thread per live sub-block → `A2q` |
| `:456-472` | `DQ2` stores from `q == 0` lanes (4× redundant across waves) |
| **`:474`** | **`__syncthreads()`** |
| `:481-488` | `N2GM_TASK_DONE_HOOK` per live sub-block → `k0p6_a2_arrive` → tid0: **agent release fence + relaxed `fetch_add`** |

### 5.6 What M6 does **not** wait on

- **No poll or spin on data from an earlier phase.** `A_dst`, `sc_dst`, `sti`,
  `sei`, `nvi`, `tile_desc` are all published by the M5 grid barrier
  (`k0pf6gm_device_tile.hip:889`, MPS `:1301`) and read with plain loads.
- **No `a2_done`-style per-tile flag on the input side.** `a2_done` is
  write-only in M6; the wait is M7's.
- **No peer/xGMI traffic, no system-scope fence, no `buffer_wbl2`.** M6 is
  entirely local.
- Confirmed independently: `STATUS.md:293-295` — "M6 has no readiness poll of
  any kind, so there is provably zero dispatch/compute overlap today".

### 5.7 Barrier census

`2 (prologue) + 1 (pipeline prologue) + 56 (K-loop) + 6 (epilogue) + 1 (task end)`
= **66 CTA barriers per M6 task**, × 11.09 tasks = **732 per CTA per epoch**.

---

## 6. Instrumentation already available

### 6.1 What exists today

**Storage.** Descriptor slot 60, `K0P6_D_MPS_STATE`
(`moe_mps_adapter.cuh:24`): 8 `uint32` scalars followed by 8 `uint64` cells,
96 B total, 8-byte aligned (`moe_host_abi.hpp:161,186-188,206-209`).
`K0P6_MPS_ST_WORDS = 8` (`moe_mps_adapter.cuh:38`), `K0P6_MPS_TS_COUNT = 8`
(`:52`).

**Clock.** `realtime_now()` (`moe_mps_adapter.cuh:336-353`):

```cpp
asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)" : "=s"(t) :: "memory");
```

`s_memrealtime` is a **100 MHz constant-rate wall counter**, coherent across
CUs → **1 tick = 0.01 µs** (`exp_05/stage0_attribution.md:31-33`; the shader
clock is 2.2 GHz — using it would be a 22× error). The `s_waitcnt lgkmcnt(0)`
and the `"=s"` scalar-pair constraint are load-bearing: without them four of the
six cells returned literally `1` (`LESSONS.md:288-296`).

**Reduction.** `ts_last(cell)` = `atomicMax(cell, t)`;
`ts_first(cell)` = `atomicMax(cell, ~t)`, recovered host-side as `~cell`, so the
M0 all-zero reset is a valid floor (`moe_mps_adapter.cuh:355-366`).

**Gate.** Every write is behind `cfg.timestamps` = **bit 33 of the packed
config word** (`moe_mps_adapter.cuh:60,79,89`), **off by default**.

**Reset.** M0 zeroes the whole block —
`for i < K0P6_MPS_ST_WORDS + 2*K0P6_MPS_TS_COUNT: mps_state[i] = 0`
(`k0pf6gm_device_tile_mps.hip:745`) — so the block is **per-epoch, not
cumulative**; after a 600-epoch soak every max comes from the final epoch.

**The eight cells and their stamp sites:**

| cell | id | where the stamp is taken | who |
|---|---:|---|---|
| `TS_FIRST_READY` | 0 | `moe_mps_adapter.cuh:675-677` — first tile event observed | service wave, lane 0 |
| `TS_LAST_READY` | 1 | `moe_mps_adapter.cuh:675-677` | service wave, lane 0 |
| `TS_DRAIN` | 2 | `moe_mps_adapter.cuh:836` — service stripe end | service wave |
| `TS_M7_DONE` | 3 | `k0pf6gm_device_tile_mps.hip:1383-1390` — right after the phase-2 body returns | tid 0, compute CTAs |
| `TS_REDUCE_DONE` | 4 | `k0pf6gm_device_tile_mps.hip:1817-1818` — CTA reduction exit | tid 0 |
| `TS_M5_DONE` | 5 | `k0pf6gm_device_tile_mps.hip:1306-1314` — right after the M5 grid barrier | tid 0 |
| `TS_M2_DONE` | 6 | `k0pf6gm_device_tile_mps.hip:1115-1124` — after the M2 Pass-A→B barrier and error gate | tid 0 |
| **`TS_M6_DONE`** | **7** | **`k0pf6gm_device_tile_mps.hip:1336-1344` — immediately after `n2p6gm_phase1_body` returns** | **tid 0** |

**Host derivation** (`moe_mps_adapter.cuh:44-48`):

```
plan (M3–M5) = M5_DONE − M2_DONE
M6           = M6_DONE − M5_DONE          ← the 2,539 µs
M7           = M7_DONE − M6_DONE
combine      = REDUCE_DONE − M7_DONE
dispatch     = epoch_total − (REDUCE_DONE − M2_DONE)
```

**Surfacing.** The state buffer is host-allocated and bound through
`mps_buffer_binding::state` (`moe_host_abi.hpp:161`, sized by
`mps_state_bytes()` `:186-188`). `exp_05` added an **additive, print-only,
exception-guarded host dump**, mirrored into `MPS_OVERNIGHT_HARNESS_NOTE.md`
(`exp_05/stage0_attribution.md:35-38`). Cost: SGPR 104 / VGPR 256 / AGPR 256 /
LDS 155,428 B — **unchanged**, i.e. resource-free.

**Also available but unrelated to M6:** `K0P6_D_SPIN_DBG` (descriptor 52) —
`uint32[2]` of max-spins-to-success / max-spins-at-failure, currently wired only
to the M2 chunk poll (`k0pf6gm_device_tile.hip:139,691`).

### 6.2 The limitation

Every cell is **one grid-wide max (or min) per epoch**. There is no per-CTA,
per-task, or intra-task resolution anywhere. The instrument can say "M6 took
2,539 µs" and *nothing* about where inside M6 that went. That is exactly the
gap this document's §2.5 has to close by arithmetic.

### 6.3 What one-line change would add sub-phase stamps inside M6

The M6 body has **exactly one extension point**: `N2GM_TASK_DONE_HOOK` at
`.node/mps_n2_phase1_gm.cpp:481-488`, fired per live sub-block at the very end
of a task, after the `A2q`/`DQ2` stores and the final `__syncthreads()` at
`:474`. Everything else inside phase 1 is hook-free. So there are two tiers.

**Tier 1 — genuinely one line, zero new files, task granularity.**
File and line: **`k0pf6gm_device_tile_mps.hip:394`**. Change

```cpp
#define N2GM_TASK_DONE_HOOK k0p6_a2_arrive(b, tid, k0p6_desc);
```

to

```cpp
#define N2GM_TASK_DONE_HOOK k0p6_a2_arrive(b, tid, k0p6_desc); k0p6_m6_task_stamp(tid, k0p6_desc);
```

with a `ts_first`/`ts_last` pair into two new cells. That yields **first M6
task-end / last M6 task-end** across the grid, i.e. the M6 task-completion
window and the CTA straggler spread — enough to separate "uniformly slow loop"
(window ≈ 2,539 µs / 11.09 ≈ 229 µs) from "tail straggler" (window ≫ that). It
costs one `s_memrealtime` per (task, live sub-block) per CTA, tid0-only, behind
`cfg.timestamps`, outside both MFMA spans.

**Tier 2 — the change that actually answers "tile-loop entry / first MFMA /
last MFMA / epilogue".** Those four points are *inside* the upstream body, which
is read-only. The clean, precedented path is to **vendor `n2_phase1_gm_mps.cpp`
next to the existing `n2_phase2_gm_mps.cpp`**, byte-derived from the pinned
donor with four macro insertion points and *nothing else changed* — exactly the
MPS-DELTA (1)/(2)/(4) pattern already used for phase 2
(`n2_phase2_gm_mps.cpp:1-11,288-294,326-330,547-549,564-566`, recorded in
`DESIGN_MPS.md:258-272` and `PROVENANCE.md:83-91`). The four sites, in the
current donor's line numbering:

| new macro | insert at | splits out |
|---|---|---|
| `N2GM_P1_TASK_HEAD_HOOK` | `.node/mps_n2_phase1_gm.cpp:161` (top of the task loop) | **the 344 KB `ascale_lds` scatter-gather prologue (§5.1)** |
| `N2GM_P1_KLOOP_ENTER_HOOK` | `:317` (after `lds_cta_barrier()`, before `for (k…)`) | **first MFMA** |
| `N2GM_P1_KLOOP_EXIT_HOOK` | `:351` (immediately after the K-loop closes) | **last MFMA — the K-loop's own duration** |
| `N2GM_P1_EPILOGUE_DONE_HOOK` | `:473` (after the `A2q`/`DQ2` stores, before the final `__syncthreads()`) | **SiLU + amax + quant + store** |

Four `s_memrealtime` per task per CTA, tid0-only, gated, all outside both MFMA
spans — the same resource-free shape `exp_05` already validated. Bump
`K0P6_MPS_TS_COUNT` from 8 to 12 (`moe_mps_adapter.cuh:52`) and
`mps_state_bytes()` accordingly (`moe_host_abi.hpp:186-188`); both are additive
and neither touches the 63-word descriptor ABI.

**Two traps that will silently void the measurement:**

1. **`.cuh`-only and new-`.cpp`-only edits do not invalidate the mori JIT
   cache** — the key hashes only mori's own `_jit-sources` tree, and our headers
   and vendored bodies arrive through `-I` from the read-only DHK mount.
   **`K0P6_MPS_SRC_REV` in `k0pf6gm_device_tile_mps.hip:104` (currently `23`) is
   the hashed file and must be bumped in the same commit**, then a fresh
   `.hsaco` mtime confirmed (`k0pf6gm_device_tile_mps.hip:96-104`, `CLAUDE.md`).
2. **`cfg.timestamps` is bit 33 of `K0P6_D_MPS_CFG` and is off by default**;
   `K0_MPS_CFG` must carry all four of `C,g,mode,flush_rows` or module import
   rejects it and the run *looks* like a pass.

### 6.4 Two more instruments already designed but not built

- **`exp_23`** (`overnight/aug11/exp_23_fig10_timeline/design.md`) specifies a
  `u64[256][24]` per-CTA phase event ring with `M6_START` as phase id 7 — a
  strict superset of Tier 1 above, at 48 KB and ≤ 24 stores per CTA per epoch.
- **`exp_22`** (`overnight/aug11/exp_22_fig7_saturation/design.md`) specifies an
  MFMA-mode saturation ubench built from the **phase-2** tile. **An M6-shaped
  mode `d` fed from a synthetic `tile_desc` would price M6's saturation-vs-CTA
  curve outside the megakernel entirely** — the cleanest way to settle §2.5
  without touching the ratchet.

---

## 7. Summary of open items

1. **`num_tiles` and `nvi[0]` are never printed.** Everything in §2–§3 scales
   linearly in `num_tiles ≈ 355 (349–380)`. One host print closes a ±4% band.
   *(UNRESOLVED, trivially fixable.)*
2. **M6's CTA-scaling has never been measured.** §0. The predicted answer is
   1.20–1.32×, not 1.005×. A phase-1 twin of `N2GM_TASK_START` /
   `N2GM_TASK_STRIDE` is the whole experiment.
3. **UNRESOLVED: the achievable XCD↔Infinity-Cache read ceiling on MI350X.**
   Both GEMMs sit at 4.3–4.9 TB/s of vector-memory request traffic against a
   nominal 8 TB/s HBM figure; the *achievable* number for this access pattern is
   not published and is not measured anywhere in this tree. Until it is, "M6 is
   L2/LLC-throughput-bound" rests on the intensity argument (28.4 TB/s needed
   for peak vs 4.91 TB/s achieved), which is sound, rather than on a measured
   roofline. `exp_22` mode b would supply the missing curve.
4. **UNRESOLVED: the exact ISA `s_waitcnt` placement inside M6's K-loop.** No
   phase-1 disassembly exists in this tree (the ISA gate captured resource
   totals and the MFMA census only, `PROVENANCE.md:106-112`). The claim that the
   pipeline is one K-step deep is read off the source
   (`.node/mps_n2_phase1_gm.cpp:318-350`), not off the emitted `vmcnt` counts.
5. **Three cheap, isolated candidate experiments fall out of this map**, in
   ascending cost:
   - **scale phase-1's four `sched_group_barrier` counts by `kGM`** (§5.3) —
     one-line schedule change, currently steering only ⅓ of the MFMA at `G=3`;
   - **kill the `ascale_lds` scatter-gather** (§5.1) — 16× line amplification,
     344 KB/task, fully exposed between two barriers, ~8% of M6;
   - **raise `G`** (§2.5) — intensity is `2·(32G)·512/(32G+512)`: **60.2 at
     `G=1`, 161.7 at `G=3`, 204.8 at `G=4`, 341.3 at `G=8`.** At a fixed
     4.9 TB/s the model predicts M6 ≈ 2,478 µs at `G=3` (measured 2,539 — a 2.4%
     fit), **1,957 µs at `G=4`** and **1,173 µs at `G=8`**, against a
     G-independent MFMA floor of 434 µs.
     The blocker is registers, and it is quantifiable: `acc[kGM][2][2][4]` is
     `64·G` registers (`.node/mps_n2_phase1_gm.cpp:212`) — 192 at `G=3`, **256 at
     `G=4`**, plus `af` `16G`, `bf` 128 and `aTmp` `8G`. `G=4` totals ~480 of the
     512 VGPR+AGPR budget (feasible, tight); `G=5` needs ~568 (**infeasible
     without reshaping `kColTiles`**). **`G=4` is the last free point on this
     axis and it is worth ~520 µs.**

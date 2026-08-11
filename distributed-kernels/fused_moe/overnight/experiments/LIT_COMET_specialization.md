# LIT: COMET (arXiv 2502.19811) — CTA specialization, mechanically

Read: ar5iv mirror + arXiv PDF + v3 HTML, all three identical (v3 is latest; no
appendix). MLSys 2025. **Labels:** `DOCUMENTED` = in the paper. `CODE` = in the
released implementation (`bytedance/flux`, COMET drop 2025-03-10) and **NOT IN
PAPER**. `INFERRED` = my derivation. `NOT IN PAPER` = paper silent, gap not filled.

**The single most important fact about this paper:** the words *signal, flag,
atomic, barrier, poll, counter, arrive, ready, semaphore, spin, fence* appear
**nowhere in the body** (verified by regex over the full text; the only hits are
"en*counter*s" in the abstract and "abstraction barrier" in a reference title).
The paper describes *which* blocks do *what*, never *how a consumer learns a
producer is done*. Every readiness claim below is therefore `CODE`, not
`DOCUMENTED`. Our bottleneck is precisely the thing the paper omits.

## What COMET actually does

**Grid partition (item 1).** A **static, compile-time tail partition of one fused
kernel's grid** — not a persistent role election, not separate kernels, not
separate SM sets.

- `grid_shape.x += GAHER_RS_N_CTAS` — the launch is `n_compute + n_comm`. `CODE`
- Role test is `if (blockIdx.x + GAHER_RS_N_CTAS >= gridDim.x)` → comm role, else
  GEMM role. So the comm pool is the **contiguous tail** of `blockIdx.x`. `CODE`
- `GAHER_RS_N_CTAS` is a **template parameter**, i.e. baked at compile time. The
  GEMM's tile scheduler is a modified CUTLASS persistent scheduler whose
  effective grid is `gridDim.x - GAHER_RS_N_CTAS`, so compute CTAs never see the
  comm CTAs at all. `CODE`
- This is why §3.2.2 needs "multiple pre-compiled kernels, each with a distinct
  division point"; the optimum is profiled offline into metadata and the right
  binary is selected at runtime. `DOCUMENTED`
- Total = SM count (132 on Hopper). Optimal `n_c` (comm blocks): 18 → 26 as
  M goes 4096 → 16384 at TP=8; 26 → 46 when TP goes 8 → 4 at M=16384, i.e.
  **13.6% / 19.7% / 34.8% of 132**, retuned per shape AND per parallel strategy.
  Fig 8 + §3.2.2. `DOCUMENTED`. Shipped configs corroborate: `gather_rs_ctas` =
  26/28/30 for topk=1/2/3 at n_dim=8192 (`docs/tuning_guide.md`). `CODE`

**What the comm blocks do, instruction-level (item 2).**

- Read the GEMM's results **from ordinary global memory** — no handshake with the
  producing warp, no LDS sharing, no cluster/DSMEM (§3.2.1 Fig 7). `DOCUMENTED`
- Stage through **LDS via `cp.async`, double-buffered**
  (`cutlass::arch::async_load` + `cp.async.commit_group` / `wait_group 0`), do
  the top-k reduce in fp32 in a small register accumulator, then store. Comment
  in source: "double buffer implementation has better performance". `CODE`
- **The comm role aliases the GEMM role's LDS.** Both roles `reinterpret_cast`
  the same `extern __shared__ char smem_buf[]`, guarded by
  `static_assert(sizeof(SharedStorage) > sizeof(TP_Shared_Storage<...>))` — the
  GEMM's shared storage is asserted *larger* so the comm storage fits inside it.
  A comm CTA thus gets the whole MFMA staging area for free. `CODE`
- Remote writes are **direct vector stores through raw peer pointers**
  (`ElementD **inter_Ds`, one base per peer) into the NVSHMEM symmetric heap —
  NVSHMEM provides the address space, not the inner-loop call. §4 + `CODE`
- Comm CTAs stride `blk_m += GAHER_RS_N_CTAS` from `blockIdx.x % GAHER_RS_N_CTAS`
  and start at a **rank-rotated offset**, `swizzled_m = (blk_m + rank *
  blks_per_rank) % blk_count`, so the 8 ranks do not all hit the same rows first. `CODE`
- **Role reflow:** after the GEMM work-fetch loop drains, the *compute* CTAs
  fall through into the same gather-rs routine for the final split
  (`sid = SPLITS-1`, striding over `n_compute_ctas`). The reservation tax is
  partly refunded at the tail. `CODE`, and a notable omission from the paper.
- §3.2.1 explicitly **rejects** intra-CTA comm/compute warp specialization: warp
  thread counts cannot saturate comm bandwidth, and comm warps interfere with
  compute warps in the same block. `DOCUMENTED` — corroborates our own strike.

**The two layers (item 3).**

| | layer0 (comm → compute) | layer1 (compute → comm) |
|---|---|---|
| shared tensor | GEMM **input**, global `(M·topk, N)` | GEMM **output**, global `(M·topk, N)` |
| decompose along | **M** (tokens independent) | **N** only |
| why not the other axis | GEMM reduces along the embedding dim | top-k reduce couples tokens along M |
| reschedule | sort tokens **by source rank**; run local-token tiles first, remote tokens transfer concurrently (Fig 5) | run GroupGEMM **column-block-major across experts** instead of expert-sequential, so reduce+send starts after the first `T_N` columns instead of after the last expert (Fig 6) |
| consumer readiness | per-source-rank flag (below) | per-split arrival counter (below) |

All of the above row 2–4 content is §3.1.1 / §3.1.2 + Figs 4/5/6, `DOCUMENTED`.

## Readiness protocol

`NOT IN PAPER` — reconstructed from the released code. Two protocols, both
**O(world size) or O(splits)**, never per-tile and never per-row.

**layer0 — one flag per source rank.** (`sm90_ag_scatter_fetcher.hpp`)

- `int *barrier_ptr` indexed **by rank**. Producer sets `flag[rank] = 1` once
  per epoch when that rank's all-gather segment has landed.
- Every GEMM work item carries a precomputed dependency **range of source
  ranks**, `ProblemSchedule{source_rank_start, source_rank_end}`. Before
  computing, the CTA loops that range and does
  `Barrier::wait_eq(barrier_ptr, thread_idx, rank, 1)`.
- The wait is `ld.global.acquire` spun by one thread, released to the rest of the
  warpgroup by a **named barrier** (`NamedBarrierSync<NumThreadsPerWarpGroup>`),
  so only the fetching warpgroup blocks, not the whole CTA.
- `revert_order_to_rank` maps a distance-order index to a rank, so the schedule
  is ordered local-rank-first and the wait is usually already satisfied.
- **Total flags: W = 8. Total flag writes per rank per epoch: 8.**

**layer1 — one counter per N-split, static expected count.**
(`sm90_gemm_array_threadblock_specialized.hpp` + `system_barrier.hpp`)

- `int32_t *barrier` of length `SPLITS + 1`, one counter per N-split.
- Compute side: the persistent scheduler visits tiles in split order; when a CTA
  crosses into a new split it does
  `MMABarrier::arrive_inc(params.barrier, mma_thread_idx, sid, add_val)` for each
  split just finished — that is `fence.acq_rel.sys` then **one
  `red.relaxed.sys.global.add.s32`, by one thread, per CTA per split**. At kernel
  end each CTA "catches up" (`for sid = sid_processed+1 .. SPLITS`) so the total
  is **exactly `n_compute` per split regardless of how work was distributed**.
  That is what makes the expected count a compile-time constant.
- Comm side: `Barrier::wait_eq(barrier, threadIdx.x, sid, N_SMS_90 -
  GAHER_RS_N_CTAS)` — spin on `ld.global.acquire.sys.b32` until the counter
  equals **the number of compute CTAs**, then `__syncthreads()`.
- **Total atomics per rank per epoch: `n_compute × SPLITS` ≈ 106 × SPLITS**
  (hundreds). Ours is **535,040** — `INFERRED` ratio ~3 orders of magnitude.
- The comm CTA never learns *which* tile finished; it waits for a whole split to be
  globally complete. The Fig 6 column-major reschedule is what makes that coarse
  question sufficient — **the reschedule is not a nice-to-have, it is the enabler
  of the cheap protocol.** `INFERRED`, and load-bearing for us.

## Measured vs claimed

**Interference (item 4). They never measure it. `NOT IN PAPER`.**

- They *claim* isolation three times — "Comet isolates the impact of
  communication on computation performance" (§1), "the computational efficiency
  of experts is not influenced" (§5.3). `DOCUMENTED` as a claim.
- The evidentiary basis is Fig 11, a four-bar comm/compute time breakdown where
  "computation" lumps in token indexing, dispatch and combine. There is **no
  measurement of GEMM throughput with the comm pool on vs off at a fixed compute
  CTA count** — the control we ran.
- Worse, their methodology **cannot** produce it: Fig 8 sweeps `n_c` holding the
  total at 132, so every point changes the compute CTA count and the interference
  at once. **Our +37% (C=16) / +57% (C=64) is a measurement COMET does not have.**

**Speedups (item 5).** 8× H800 (NVLink), CUDA 12.3, NVSHMEM 2.11, Megatron-LM
6dbe4c; also 8× L20 over PCIe (~25 GB/s). `DOCUMENTED`

- Single MoE layer **1.96×**; end-to-end **1.71×** avg; end-to-end reductions
  34.1% / 42.6% / 44.4% / 31.8% vs Megatron-Cutlass / Megatron-TE / FasterMoE /
  Tutel. Layer range 1.28×–2.37×; L20 1.19×–1.46×. `DOCUMENTED`
- **Attribution to the specialization specifically: absent.** There is no
  ablation isolating thread-block specialization from the shared-tensor
  decomposition, the reschedule, the CUTLASS GEMM swap, or the removal of
  host-side kernel scheduling. Fig 8 is the only specialization-adjacent
  measurement and it only varies `n_c` *within* the final design.
- They explicitly credit part of the win to **host-side scheduling overhead
  removal** ("the advantage is prominent especially when M is small... Comet
  reduces such overhead through kernel scheduling within the fused kernel",
  §5.3), and part to avoiding TP weight-switching (§5.3). Both are orthogonal to
  CTA specialization. So 1.96× is **not** a role-split number.
- Closest thing to a mechanism number: **86.5% of communication latency hidden**,
  vs FasterMoE 29.2% and Tutel 68.6% (§5.3, Fig 11). `DOCUMENTED`

**Granularity and bandwidth (item 6).**

- **CONFIRMED: COMET contains no bandwidth-vs-transfer-size measurement of any
  kind.** There is no §3.3 (sections run 3.1, 3.1.1, 3.1.2, 3.2, 3.2.1, 3.2.2,
  then §4). Figures 1–14 are all latency, breakdown or memory-footprint plots;
  Tables 1–3 are symbols, model configs, NVSHMEM bytes. The only bandwidth
  number in the paper is "~25 GB/s" as an L20 PCIe *testbed property* (§5.4).
  The prior agent's correction stands — the 1 MiB-knee / 87 GB/s curve is **not
  COMET's**. `DOCUMENTED` (by absence, verified)
- Granularity: layer0 unit is **one token row** (N elements); layer1 unit is a
  **`T_N`-column block** where `T_N` is the GroupGEMM N-tile. `DOCUMENTED`.
  Concrete bytes are never stated. `INFERRED` from Table 2: Mixtral N=4096 BF16
  → 8,192 B per token row. Shipped tile is (128,256,64), so `T_N` = 256. `CODE`

## Filter against our constraints

| mechanism | verdict | reasoning |
|---|---|---|
| per-source-rank readiness flags (layer0), W=8 flags, dependency range precomputed per tile | **TRANSFERABLE — top priority** | Kills constraint 3 at the root. Replaces per-(block,row,chunk) atomics with 8 flag writes. Needs no registers, no LDS, no per-lane state. |
| per-split arrival counter, expected count = n_compute, end-of-kernel catch-up | **TRANSFERABLE — top priority** | ~106×SPLITS vs our 535,040. One `red.relaxed` per CTA per split. The catch-up loop is the trick that keeps the expected count static under dynamic routing. |
| column-block-major (nc-major) GroupGEMM reschedule | **TRANSFERABLE** | This is our M8 item, and it is a *prerequisite*, not an alternative: it is what makes split-level readiness sufficient. Do not adopt the counter without it. |
| spread counters one-per-split across distinct lines | **TRANSFERABLE, and required** | Constraint 5 says coalescing RMWs onto few lines is ≥10× worse. SPLITS counters are few but each is hit by ~106 CTAs. Pad each counter to its own cache line. |
| comm role aliases the GEMM role's LDS (union over one `smem_buf`) | **TRANSFERABLE — high value** | Directly dissolves constraint 1. Our comm CTA is starved at 8 KB only because the MFMA's 155,428 B is treated as reserved; in a comm CTA that region is dead. Union it and the staging array never touches registers or scratch. |
| `cp.async`/LDS double-buffered staging in the comm role | **NEEDS ADAPTATION** | Correct shape for constraint 1, but needs the LDS union above first. CDNA4 has direct global→LDS; no `cp.async.commit_group` equivalent, use the `vmcnt` form. |
| static tail-block role partition, C a compile-time constant | **TRANSFERABLE (already have it)** | Our `bid<C`/tail placement work is the same mechanism. Note their pool is the **contiguous tail**, matching our corrected A8 reading. |
| offline-profiled C, per-shape metadata | **TRANSFERABLE** | Confirms C must be swept per shape, and confirms the 14–35% band. Our C≤16 range is below every point they ship. |
| role reflow: compute CTAs join the comm routine after the GEMM drains | **TRANSFERABLE — high value** | Attacks the capacity tax we pay for C. Costs nothing at steady state; refunds C at the tail. `CODE` only — the paper never mentions it. |
| rank-rotated combine start offset | **TRANSFERABLE, cheap** | Two arithmetic ops. Spreads 8 ranks' destination traffic instead of stacking it. |
| device-side packed sorted schedule (`int16` problem_idx/tile_m_start/tile_m_size`), `tile_count` in device memory to avoid host sync | **TRANSFERABLE** | Plan-side, aimed at our 43% pool. See below. |
| gather-index (`gather_A`) instead of physically permuting tokens | **TRANSFERABLE if we permute today** | Plan-side. Permutation costs zero data movement if the mainloop gathers rows through an index. |
| caching gather row-indices **in registers** across the K loop (§4) | **BLOCKED BY constraint 1 → NEEDS ADAPTATION** | We have zero register headroom and already spill 64 B/lane. Adapt: stage the tile's indices in **LDS** — 256 × 4 B = 1 KB, fits even in the current 8 KB headroom, and trivially in the unioned region. |
| local-token-first tile ordering (layer0, Fig 5) | **KILL — blocked by constraint 4** | Its entire purpose is to avoid stalling on remote tokens. We measured max-spins-to-success = 0 of 2,000,000: there is no peer wait to hide. Reordering buys nothing. Do not build it. |
| overlapping the layer0 all-gather under the first GEMM, as a *latency-hiding* play | **KILL — blocked by constraint 4** | Same reason. If dispatch is worth attacking it is as ~1,100 µs of *serial work*, not as exposed wait — a different mechanism with a different justification. |
| intra-CTA comm/compute warp specialization | **already struck, now corroborated** | §3.2.1 rejects it on their own hardware for our reasons. Adds confidence, no action. |

## Ideas for our 43% pool

Does COMET touch the pre-GEMM plan or the first GEMM? **Partly, and the useful
part is not the overlap.** Their layer0 *is* our M0–M2 + plan + M6, but their
layer0 mechanism is remote-fetch latency hiding, which constraint 4 has already
killed for us. **COMET contains nothing that makes the first GEMM's MFMA work
itself faster** — no tiling, no K-split, no data-type play. Their layer0 win
comes from deleting serial all-gather time and host-side scheduling. So:

1. **Attribute the 2,976 µs first.** "plan + M6" is one fused number; every
   plan-side idea below is capped by the plan's true share. This is the cheapest
   next action and it gates the rest. If plan is <200 µs, stop reading here and
   spend the night on the readiness protocol instead.
2. **Packed device-side schedule.** `ProblemSchedV2{int16 problem_idx, int16
   tile_m_start, int16 tile_m_size}` — 6 B per work item, built by a device kernel
   (`get_sorted_problem_schedule_cuda_v2`) from the per-expert `splits` and a
   per-rank cumsum, with `tile_count` **left in device memory "to avoid device
   sync"**. Replaces any wide per-token index tensor or host round-trip. `CODE`
3. **Gather-index, not permutation.** `int32 **gather_A` per problem; the GEMM
   mainloop gathers rows. If our plan physically permutes tokens into
   expert-contiguous order, that copy is pure deletable traffic. `CODE`
4. **Index staging in LDS (our adaptation of §4).** COMET caches the per-K-iter
   row indices in registers; we cannot. 1 KB of LDS gets the same effect and is
   available even before the LDS union. `INFERRED`
5. **The real 43% play is indirect.** Our own finding 3 says the +37–57% GEMM
   inflation is *atomic* traffic, not payload. COMET's protocol has ~1,000× fewer
   atomics. So constraint 2 is plausibly a measurement of **our protocol**, not
   of role specialization — which means re-running the split with a per-split
   counter + per-rank flags is the highest-expected-value experiment available,
   and it is what would let comm ride under M6 at all. Pre-register the
   falsifier: if inflation stays above ~+20% at C=32 with ≤1,000 atomics per
   rank per epoch, the interference is not atomics and finding 3 needs revisiting.

## Sources

- Paper (identical across all four): https://arxiv.org/abs/2502.19811 ·
  https://arxiv.org/pdf/2502.19811 · https://ar5iv.labs.arxiv.org/html/2502.19811
  · https://arxiv.org/html/2502.19811v3 — §3.1.1/3.1.2 decomposition axes +
  reschedule (Figs 4/5/6); §3.2.1 thread-block specialization and the rejection of
  intra-CTA warp specialization (Fig 7); §3.2.2 `n_p`/`n_c`, the 18/26/46-of-132
  optima, offline-profiled pre-compiled kernels (Fig 8); §4 CUTLASS + NVSHMEM +
  row indices in registers; §5.1 testbed; §5.2/5.3 the 1.71×/1.96×/86.5% numbers
  (Figs 9–11); §5.4 L20 ~25 GB/s. **Absence verified:** no synchronization
  vocabulary anywhere, no §3.3, no bandwidth-vs-transfer-size figure.
- https://github.com/bytedance/flux — the COMET drop (README "[2025/03/10] We have
  released COMET"). Every `CODE` claim above:
  `src/moe_gather_rs/sm90_gemm_array_threadblock_specialized.hpp` (grid `+=
  GAHER_RS_N_CTAS`; role test `blockIdx.x + GAHER_RS_N_CTAS >= gridDim.x`;
  `arrive_inc` per CTA per split + end-of-kernel catch-up; `wait_eq(..., N_SMS_90 -
  GAHER_RS_N_CTAS)`; LDS-union `static_assert`; rank-rotated `swizzled_m`; role
  reflow after the work-fetch loop) · `…/sm90_group_tile_scheduler_threadblock_
  specialized.hpp` (compute grid `gridDim.x - GAHER_RS_N_CTAS`, template constant)
  · `include/flux/cuda/system_barrier.hpp` (`ld.global.acquire.sys.b32`;
  `fence.acq_rel.sys` + `red.relaxed.sys.global.add.s32`) ·
  `src/moe_ag_scatter/sm90_ag_scatter_fetcher.hpp` (per-rank flag, per-tile
  source-rank range, named-barrier release) · `…/workspace_util.h`
  (`ProblemSchedV2`, `gather_A`, `tile_count` kept on device).
- https://github.com/bytedance/flux/blob/main/docs/tuning_guide.md —
  `gather_rs_ctas` = 26/28/30 at topk=1/2/3, `tile_shape=(128,256,64)`,
  `n_dim=8192`; "the first value in `make_gather_rs_hparams` refers to the number
  of thread blocks specialized for communication".
- ByteDance Seed blog (secondary, only corroborates Fig 8 percentages and the
  MLSys 2025 5/5/5/4 acceptance): https://seed.bytedance.com/en/blog/comet-has-been-deployed-in-large-scale-clusters-saving-millions-of-gpu-hours-moe-communication-optimization-technology-comet-is-now-open-source

# Yotta Labs' AMD Developer Challenge 2025 write-up — what it says about GEMM-RS

Source: <https://www.yottalabs.ai/post/optimizing-distributed-inference-kernels-for-amd-developer-challenge-2025>
Read 2026-08-11. Section 3.2 is the GEMM-ReduceScatter kernel; §3.1 (All-to-All)
and §3.3 (AllGather-GEMM) contain transferable mechanisms.

Confirms the competition context we assumed: **the metric is a geometric mean
across the problem sizes**, explicitly so that "solutions that excel on some
problems but perform poorly on others" are penalized. Our shapes are their
shapes (64×7168×18432 → 8192×8192×29568, with and without bias).

## 1. They independently found our exp_08 result — from the other direction

> "Remap thread blocks across MI300X's 8 XCDs so each XCD processes 1/8 of
> output matrix rows … **Enables full utilization of all 7 Infinity Fabric
> links simultaneously**"

That is exactly the defect exp_08 fixed: our inherited `WGM = 4` made a tile
group smaller than a 272-CTA round, so a round reached only 2-3 of 8
destinations and we measured **2.02 effective links of 8**. We fixed it with
`WGM = num_pid_m` (column-major decode); they fix it with an explicit XCD
remap. **Two independent teams, same root cause, different mechanism** — which
is strong evidence the mechanism is real and that the remaining question is
only how completely it is exploited.

**Where they go further, and we have not: destination affinity per XCD.** Ours
spreads destinations across XCDs (7.53 effective links, good) but does not bind
a destination to an XCD. Theirs gives each XCD a contiguous 1/8 of the output
rows, so an XCD drives essentially one link and its **per-XCD L2 (4 MB each on
MI300X) holds only that destination's data**. That is E7 in our table and is
still untested. Concretely: CTAs are assigned to XCDs round-robin, so
`blockIdx.x % 8` is the XCD, and the swizzle wanted is one where
`tm / (num_pid_m/8) == pid % 8`.

## 2. Host launch overhead — they quote a number we should compare against

> "Reduces kernel launch overhead from **~120 µs to ~40 µs** … Critical for
> small problem sizes where launch overhead becomes significant."

Achieved by caching the compiled kernel after warmup and calling the AOT
launcher directly, passing a base pointer plus integer offsets instead of full
tensor objects (`A_ptr_index_hack`). That is a Triton-specific problem — they
were paying Python/Triton dispatch — and we are hand-written HIP with a
`pybind11` binding, so our floor should already be lower. But our harness
reports **`hostIss` ≈ 62 µs**, and that column measures *eight* launches issued
serially from one process, so the per-launch figure is ~8 µs. Worth confirming
under the evaluator's one-process-per-rank layout before assuming it is free.

Note this is also the origin of the pointer arithmetic that made rank-1 fail
here: passing a 100-element base tensor plus a −4.4-billion-element offset is
what let Triton 3.6.0 pick `buffer_store_dwordx2` and truncate the offset to
32 bits.

## 3. Their GEMM-RS is TWO kernels with a global barrier — ours is one

> "Split into two kernels: GEMM with scatter epilogue, then separate reduce
> kernel. Use global barrier between kernels … **Leverages kernel boundaries
> for automatic coherence control**. Separate reduce kernel achieves full
> memory bandwidth with optimized grid dimensions."

This is the opposite architectural choice from ours, and the trade is worth
stating precisely because it bears directly on the CTA-split question:

| | theirs (2 kernels + barrier) | ours (1 persistent megakernel, CTA split) |
|---|---|---|
| coherence | free, from the kernel boundary | explicit release/acquire, epochs, per-tile credits |
| reduce width | **full grid** | `NUM_REDUCER_CTAS = 32` of 304 |
| overlap | none between reduce and GEMM | reduce overlaps GEMM continuously |
| launches per call | 2 + a barrier | **1** (our gate 16) |

Their reduce runs at full memory bandwidth because it owns the whole machine;
ours runs on 32 CTAs but never blocks the GEMM. On the current binary our
`reduce` pool is only **86.5 µs of shape 6's 1777.9**, so their advantage there
is small — but our `sync` (191.0) plus `release` (134.9) is **326 µs**, and
that entire cost is the price of doing coherence in-kernel instead of at a
kernel boundary. **That is the real number to weigh against one extra launch.**

The user's stated direction is to keep iterating on CTA-level
compute/communication overlap, so this is recorded as the strategic alternative
rather than a proposal — but if `sync + release` stops shrinking, this is the
axis that makes it structurally zero.

## 4. Smaller, directly portable details

- **`.cg` cache modifier on the remote store** (`tl.store(c_ptrs, c,
  cache_modifier=".cg")`). We currently emit peer payload stores with **no**
  cache-scope bits at all, which is why they land in local L2 and egress at the
  `buffer_wbl2`. Worth an experiment to see whether an explicit policy beats the
  accidental one — but note our own profiling already showed 1.0003× write
  amplification and 99.9% full-64 B transactions, so the current behaviour is
  not obviously improvable.
- **Fine-grained symmetric heap via `hipExtMallocWithFlags`**, 1 GB for GEMM-RS
  (10 GB for All-to-All). We already tested allocation granularity: <1%
  difference. Closed.
- **Template specialization per dimension** to remove runtime branching — we
  already do this via the `config_row` dispatch to compile-time `BM/BN/BK`.
- **`float4` packing rather than Triton's default `float2`** for transfers,
  "achieving full bandwidth with fewer CTAs since only half the memory
  operations are needed". We already emit `global_store_dwordx4` (16 B/lane),
  which is the widest store gfx942 has.
- **§3.3's fine-grained per-chunk signalling with a spin-wait and
  `device_sleep()` backoff** is the same shape as our `bounded_poll_relaxed_into`
  with `s_sleep(4)`. Their AllGather-GEMM dedicates **56 CTAs (8 per remote GPU
  × 7)** to communication — interesting because AMD's own TransferBench also
  saturates the fabric with 8 CUs per link. Ours dedicates 32 CTAs to *reduction*
  and none to sending (producers send from the epilogue). A variant worth
  considering: dedicated sender CTAs at 8-per-peer, which is the natural
  CTA-level boundary the hardware seems to want.

## Ranked implications for our next experiments

1. **E7, XCD-aware remap with destination affinity.** They report it as a
   headline win, we have the same mechanism only partially exploited (7.53 of 8
   links but no XCD↔destination binding, so per-XCD L2 sees all eight
   destinations). Cheap: pure index arithmetic, no protocol change.
2. **Dedicated sender CTAs at ~8 per peer.** Both their AllGather design and
   AMD's TransferBench converge on 8 CUs per link being enough to saturate it.
   That is a direct statement about where the CTA-level producer/consumer
   boundary should sit — which is the project's stated research question.
3. **Attack `sync + release` (326 µs on shape 6)** while keeping one launch.
   Their two-kernel design gets this for free; if we cannot shrink it in-kernel,
   that comparison becomes the argument for changing the architecture.

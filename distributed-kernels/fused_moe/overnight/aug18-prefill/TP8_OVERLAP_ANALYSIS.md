# TP8+EP: what materially changes for our kernel, and the overlap agenda

Author: orchestrator direct analysis (2026-08-18, API-outage fallback — the
multi-agent research pass died; this is first-principles reasoning from the
banked laws, the M15 kernel architecture, and R1-0528's shapes. Every number
here is an estimate to be calibrated by B0/B2 measurements; nothing below is
a measured result.)

Model shapes used throughout: DeepSeek-R1-0528 — 61 layers (3 dense + 58
MoE), hidden 7168, 128 heads, MLA (kv_lora_rank 512 + 64 rope), 256 routed
experts top-8 + 1 shared expert, ~37B active params. bf16 activations
(14 KB/token/layer-boundary), fp8 weights.

## 1. The topology swap, precisely

**DP8/EP8 (ours today).** Eight engines, each with its own batch (B4096/rank).
Attention, dense layers, shared expert: fully replicated compute, ZERO
communication. MoE: experts sharded 32/rank; the ONLY comm is the token
all2all — dispatch (each token's activation to the ranks owning its top-8
experts) and combine (weighted partials back). Our M15 mega fuses
dispatch → expert GEMMs → combine into one persistent kernel and hides the
all2all behind compute with chunk_ready counters, slab rendezvous, and the
cross-rank RMW combine (`global_atomic_pk_add_bf16` under the depth-4 vmcnt
law).

**TP8+EP (AMD's ≤128-concurrency recommendation).** One engine, weights
sharded 8-way. The batch is global (vLLM chunks prefill at
max-num-batched-tokens, AMD rec 16384). Per layer:
- MLA attention: heads sharded 16/rank, o-proj row-parallel → **all-reduce #1**
  (hidden × tokens).
- MoE with EP inside the TP group: the post-AR hidden is REPLICATED on all
  ranks, so routing/gating is computed locally (redundantly, cheap) and each
  rank simply gathers rows destined for its own 32 experts from local memory.
  **The dispatch all2all disappears entirely.** Each rank produces outputs
  only for the tokens its experts served; the combine is a partial-sum
  **all-reduce #2** — which is the all-reduce the row-parallel MLP would need
  anyway. Shared expert: TP-sharded, its output folds into the same AR.
- Dense layers (3) and every norm: standard TP, riding the same two ARs.

So the comm primitive changes species: **all2all (sparse, MoE-only) → two
per-layer all-reduces (dense, on the critical path of every layer)**.

## 2. What this does to the cost structure (c32p prefill, estimates)

Per layer, per 16K-token prefill step, ring-AR cost ≈ 2 × (7/8) × 16K ×
14 KB ≈ 0.4 GB moved per AR; at a realistic achievable single-node AR bus
bandwidth (~300 GB/s class over xGMI) that is ~1.3–1.6 ms per AR-pair per
layer. GEMM side: ~37B active × 2 FLOP × 16K tokens / 61 layers ≈ 19 TFLOP
per layer across 8 GPUs; at effective fp8 rates that is ~3–4 ms per layer.

**Estimated exposed-comm fraction under naive TP8 prefill: ~25–35% of step
time.** That is the Amdahl pool for overlap work — fully hiding it is worth
~+35–50% TP8 throughput. This is the entire opportunity, and it is bounded:
no overlap trick can beat removing the bytes (which DP8/EP8 already does —
it has NO per-layer ARs; that is why the DP mega can still win c32p prefill
even though AMD recommends TP8 below 128 concurrency: AMD's rec is driven by
decode latency and TTFT, not prefill throughput. Expect the per-cell winner
to split along exactly this line; B2/B3 will measure it.)

Second-order but material:
- **Per-rank expert-GEMM M halves** (global 16K tokens vs DP's 8×4096):
  ~16K expert-rows/rank instead of ~32K. Still chunky for prefill; tile
  shapes survive. In DECODE cells (c8/c16/c32) M collapses to a few hundred
  rows and the AR becomes latency- not bandwidth-bound — a different regime
  needing latency tricks (one-shot/LL-style ARs), not bandwidth overlap.
- **The DP-dummy problem evaporates.** No idle-rank lockstep, no
  execute_dummy_batch, no 56%-dummy steps. M24 tier-1's prize is
  DP-topology-specific. Production TP8 doesn't pay the dummy tax either —
  so under TP8 the fill lever is simply gone, for both sides.
- **Memory improves**: attention/dense/shared weights no longer replicated
  (~15 GB/rank freed) → more KV headroom, bigger safe batches.

## 3. Asset-by-asset: what ports, what dies

| Our asset | Fate under TP8+EP |
|---|---|
| Cross-rank RMW combine (atomic pk_add to peer buffers, depth-4 vmcnt) | **PORTS, and becomes the centerpiece**: an in-GEMM-epilogue reduce — expert GEMMs atomically accumulate partials straight into a (replicated or sharded) output buffer across ranks. Done right, AR #2 stops existing as a separate phase. |
| chunk_ready / counted-arrival signaling | **PORTS**: per-chunk readiness of AR #1's output lets the MoE front (gating + row gather + first expert tiles) start on early chunks — counter-dataflow across the attention/MoE boundary. |
| Single grid barrier per launch (T3 law), signaling-over-barriers | **PORTS** unchanged — it is topology-agnostic doctrine. |
| Occupancy-1 256-VGPR GEMM bodies | **PORT for prefill** (M stays large); decode needs new small-M tile variants — new work. |
| Dense tiling from live counts (tile_desc from SC_ERB) | **PORTS**: same mechanism, rows-per-expert now comes from local gating instead of received counts. |
| Slab rendezvous / M8 certification | **REPURPOSED**: from "all ranks' dispatch arrived" to "AR chunk k certified" — coarser and simpler (one producer per chunk, not eight). |
| Shared-expert filler (design doc) | **PORTS with LESS complexity**: shared expert is just another GEMM stream to schedule into AR-drain windows inside the persistent kernel; its output joins the same RMW accumulation, deleting its separate AR. |
| MORI all2all machinery | **DIES** (no dispatch under TP8+EP). |
| M23 ragged seal | **DIES** (per-rank DP batch concept). |
| M24 dummy-skip (tier 1) | **DIES under TP8** — but stays the key lever for the DP8/EP8 high-concurrency deployment (B3's regime). |

## 4. The overlap agenda for TP8+EP, ranked

1. **RS/AR-in-epilogue for the MoE+shared output (biggest, most portable).**
   Expert GEMM epilogues write partial sums via remote atomics directly into
   the layer-output buffer on all (or owning) ranks, with per-chunk arrival
   counters instead of a collective call. AR #2 vanishes into GEMM drain.
   Respects: vmcnt depth-4, coarse chunk signals, no exposed barrier — the
   consumer (next layer's norm/QKV) spins relaxed on chunk certificates.
   Ceiling: roughly half the exposed comm (~15% e2e); cost: M (the indexing
   is new, the primitive is our combine).
   Risk: atomic-add bandwidth over xGMI vs ring efficiency — the ring moves
   2×(7/8)×B; naive 8-way atomic fanout moves (7/8)×B×… per-destination
   once — actually FEWER bytes for the replicated-output case (each partial
   travels once per destination, no reduce-then-broadcast round trip), at
   the price of RMW throughput limits. Must measure early (this is the
   cheapest decisive prototype: a standalone RMW-AR vs RCCL AR microbench
   at 0.4 GB message class, both latency and bandwidth regimes).
2. **Counter-dataflow across AR #1 (attention→MoE boundary).** Chunk the
   o-proj row-parallel reduce; gate + gather + early expert tiles start per
   ready chunk. Ports chunk_ready verbatim. Ceiling: the other ~half of
   exposed comm, minus the unoverlappable head chunk.
3. **Persistent TP8 MoE megakernel** = (1)+(2)+dense tiling+shared-expert
   filler in one launch, one grid barrier. This is "the mega, re-targeted":
   planner consumes local gating counts; service CTAs become AR-chunk
   movers; combine CTAs become epilogue accumulators.
4. **Decode-regime latency track (separate problem)**: small-M tiles;
   one-shot/LL all-reduce (latency-optimal for the few-hundred-token AR);
   possibly fused AR+RMSNorm (AITER has candidates). Do not conflate with
   the prefill bandwidth track — different mechanisms, different cells.
5. **Sequence-parallel variant** (RS+AG instead of AR, norms on shards):
   halves norm-path comm and gives natural chunk boundaries, but token-
   sharded hidden complicates the local row-gather for EP. Secondary;
   evaluate only if (1)/(2) leave exposed norm time.

Falsified-catalogue guardrails that carry over: sync EXPOSURE is deadly
(the AR must never appear as a standalone phase in the ledger); per-row
readiness is probability-zero (chunk granularity everywhere); pools
consume-only in a producer's shadow (AR accumulation buffers are
producer-owned; consumers read only certified chunks).

## 5. Strategic read

TP8+EP does not obsolete the mega — it obsoletes the mega's COMM LAYER and
keeps its execution doctrine. The portfolio that falls out:
- **Low/mid concurrency (TP8 regime)**: new overlap kernel per §4; prize
  ≈ the 25–35% exposed AR fraction. Until built, our DP mega may still hold
  c32p prefill on throughput (B2 measures this today).
- **High concurrency (≥512, AMD's own DP8+EP regime)**: current mega + M24
  fill/dummy work is exactly on-target (B3 measures this today).
- The deployment-policy row then writes itself per cell, honestly.

## 6. What B0/B2/B3 must calibrate before any of §4 is built

- Real exposed-comm fraction at TP8 c32p (from native_tuned_tp's measured
  throughput vs its GEMM-bound roofline; better: a torch-profile pass).
- RCCL AR achieved bandwidth at the 0.4 GB and the 3 MB (decode) message
  points on this box.
- The RMW-AR microbench of §4.1 (cheapest decisive prototype, ~a day).
- Whether vLLM 0.25.1's TP8+EP MoE path actually rides AR #2 as assumed
  (read the resolved config + a profile from the B0 native_tuned_tp arm).

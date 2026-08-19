# G-L1/G-L2 boundary races — the shared-expert filler question, closed

**Date:** 2026-08-19 · **Rig:** m25_boundary_bench @48162d65, `-DM25_G1=1`, node gfx950 ×8 ·
**Workload tuple:** tokens=16,384/rank (the C=32 deployment chunk), slab_rows=256, depth=4,
k_inner=9,040 (ρ_nom=3, deployment-like), bps=2, bf16, 235 MB collective, K=7–9 per arm,
all arms alternated within one invocation (LAW-60). Logs: node `~/g1_race{1..5}*.log`;
medians quoted unless noted; rcclserial max always carries the RCCL first-iter outlier (ignore).

## Arms

- `rcclserial` — PRODUCTION STRUCTURE (verified against vLLM source, see §U2): sharded shared
  expert serialized on the critical path, pre-added, ONE monolithic AR.
- `rccl` — no shared expert (reference for the serial cost).
- `rcclchunk` — chunked AR, no filler (pure chunking tax).
- `rcclhybrid_indep` — filler on a second stream, no AR dependency (models REDUCTION-FREE work).
- `rcclhybrid_merged` — filler chunk i gates AR chunk i (models the SHARDED shared expert whose
  partials must ride the AR).

## Results

| race | config | rcclserial | rccl | candidate | verdict |
|---|---|---:|---:|---:|---|
| 2 | indep, chunks=1, fprio=lo | 8,982.6 | 8,496.2 | 8,868.2 | −114 µs (capture 24%) |
| 4 | indep, chunks=1, **fprio=hi**, fb=256 (mins) | 9,098.4 | 8,437.3 | **8,781.2** | **−317 µs (capture 48%)** |
| 4 | fb=128/64 | — | — | 9,089/9,177 | fewer filler blocks always worse |
| 3 | NCCL channels 16/8 | 13,234/18,097 | 12,587/17,378 | 12,670/17,542 | AR needs default channels (16ch: capture 87% but +4.1 ms AR — dead) |
| 1 | merged, chunks=8 | 9,148.1 | 8,550.9 | 9,979.9 | loses big |
| 5 | merged, chunks=2 | 9,103.1 | (chunk-only 8,649.1) | 9,445.1 | **loses by 342** |
| 5 | merged, chunks=4 | 9,019.7 | (chunk-only 8,697.3) | 9,747.8 | loses worse |

Chunking tax alone: +153 µs at N=2, +201 at N=4, +668 at N=8 (vs monolithic 8,496).
Filler calibration: k_filler=180 → 860.9 µs solo = 1/8 of the 6,784 µs compute arm (the sharded
shared expert's per-rank cost). Session note: this session's compute arm ran ~3.5% above the
banked 6,557 (cross-session drift); all contrasts are in-session.

## Mechanism findings

1. **Filler retention beside a monolithic RCCL AR, in-process:** ~38% at low stream priority,
   **~48% at high priority** (fb=256) — matching the two-process microcosm's 44–53%. Priority
   is the lever; block-count reduction is not (grid-stride filler at 128/64 blocks lengthens its
   own runway more than contention relief buys).
2. **The merged (token-chunked) schedule cannot win at any chunk count:** filler chunk 0 is
   exposed by construction, ~50% retention makes every filler chunk outlive its AR chunk, and
   the chunk tax grows with N. Interior optimum does not exist between N=2 and N=8.
3. **Full replication loses on arithmetic** (G-L4 map §1.5-amended): per-rank replicated shared
   expert = 8× its shard = the ENTIRE per-rank routed load (~R 4.0 ms) vs a ~2.8 ms window.
4. **U2 resolved — production pays ONE all-reduce under TP8+EP.** From container source
   (`moe_runner.py:406-431,700-730`; `_aiter_ops.py:1558,1676`;
   `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=False` in the serving image): the shared-expert
   AR fires only when the fused kernel's `output_is_reduced()` — true only on all2all-manager
   (DP/EP) paths; TP8+EP (dp_size=1, no manager) takes the pre-add branch: `shared + routed`
   summed, one `tensor_model_parallel_all_reduce`. There is no second collective to delete.

## Verdict

**The shared-expert filler is CLOSED as a boundary-level win.** The only arm that beats
production structure (−317 µs, −3.5% of the boundary) models work that needs NO reduction —
and the shared expert is not that work: sharded, its partials must ride the AR (merged loses);
replicated, it is 8× too large. G-L2's ≥25%-of-exposure bar is unmet by any legal variant.

**What the −317 µs arm DOES prove:** ~48% of independent compute rides free beside the AR at
high stream priority. The real candidates for that slot are (a) next-chunk/next-microbatch work
(serving-level chunked-prefill pipelining — the strongest legal filler; G-L4 §4 territory),
(b) next-layer expert-weight prefetch (small, real). The shared expert itself stays serialized,
exactly as production runs it.

**Track 1's surviving path to beat production at C=32:**
1. **G-L1 — the MoE-region GEMM floor** (now the PRIMARY lever, unmeasured): one persistent
   fused kernel for gather + gate/up/down + sum vs AITER's kernel sequence + Python glue at the
   TP8 shapes. M15's machinery won 24.5% at the DP boundary; the TP8 region is GEMM-dominated
   so the edge is smaller but real glue/launch overhead exists to delete.
2. **Serving-level overlap** — next-chunk compute under this chunk's AR (needs G-L4's W1 source
   read; independent-work physics measured here at 48% retention).
3. Weight prefetch during the AR (small).
The projection: G-L1 edge (unknown, measure first) + ~300 µs/boundary from (2)+(3) if the
serving scheduler permits. The +6-10% e2e projection is REVISED DOWN pending the G-L1 floor
measurement — the filler half of it did not survive contact with the dependency structure.

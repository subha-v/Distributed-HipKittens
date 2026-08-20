# R1 TP8 graph-mode DECODE trace analysis — is an M15 port the decode play?

**Source:** `amd-master/EXPERIMENTS/2026-06-deepseek-r1-tp8-graph-profile/artifacts/AMD/R1_graph_mode_rank0.pt.trace.json.gz`
(Kineto, rank 0, 8× MI355X-class gfx9.5 / 256 CU / 288 GiB, 12.35 s span, 1.09 M kernel events,
792 annotated generation steps). Analysis scripts: session scratchpad `tr1/tr2/tr3.py`.
**Workload:** graph-mode DECODE at batch sizes 1 (n=274), 4 (n=250), 16 (n=250) + warmup.
Caveat: rank-0 stream view; per-kernel times include any in-kernel spin/wait.

## Per-step anatomy (µs medians; idle ≤2% everywhere — graph mode leaves no launch gaps)

| family | bs=1 (wall 10,337) | bs=4 (11,973) | bs=16 (14,841) | kernels/step |
|---|---:|---:|---:|---:|
| **all-reduce 1-stage (aiter fused)** | **6,362 (61.5%)** | 1,680 | ~0 | 122–123 (2/layer) |
| **all-reduce 2-stage (RS+AG)** | 418 | — | 3,012 | 244 @bs16 (4/layer) |
| MoE grouped GEMMs (ck) | 1,748 | 3,461 | 4,762 | 116 (2/layer) |
| dense GEMMs (attn proj + shared exp) | 2,734 | 3,013 | 3,093 | 313–428 |
| fp8 activation quant | 1,353 | 1,376 | 1,410 | **305 (5/layer)** |
| MLA attention + reduce | 1,239 | 1,175 | 1,234 | 184 (3/layer) |
| MoE glue (topk, sorting) | 764 | 776 | 841 | 116 |
| fills/memsets + other | 1,266 | 355 | 129 | 20–260 |
| **TOTAL kernels/step** | **1,416** | 1,296 | 1,415 | avg ~7 µs each |

TPOT: **10.34 ms (96.7 tok/s/user) at bs=1**; 11.97 ms at bs=4 (334 tok/s agg); 14.84 ms at
bs=16 (1,078 tok/s agg).

## Findings

1. **At bs=1, 62% of every decode token is all-reduce kernel time** — 122 rendezvous of a
   14 KB payload at 52.1 µs each. This is not bandwidth (14 KB!). And the same kernel runs
   **13.7 µs at bs=4** with a 4× larger payload — so ~75% of the bs=1 AR time is likely
   **inter-rank arrival skew absorbed inside the AR** (at bs=1, EP expert-hits are sparse and
   uneven per rank, so per-rank MoE time varies and the AR waits for the slowest rank) plus a
   per-rendezvous latency floor. Decode at low concurrency is a LATENCY+SKEW problem, not a
   bandwidth problem. NOTE: every transport law we have banked (β=1.5×, towers, knees) is from
   the 235 MB bandwidth regime — the 14–230 KB latency regime is unmeasured on our side
   (θ-F13, the hole the plan already names).
2. **MoE GEMMs at decode are weight-streaming slivers at/near the HBM floor.** bs=16:
   41 µs/kernel ≈ the fp8 weight bytes of the experts hit ÷ ~4.15 TB/s. There is no
   GEMM-efficiency prize here — **an M15 port attacks the wrong bottleneck**. M15's wins
   (tile efficiency on big grouped GEMMs, coarse certificates over a2a) have no decode analog.
3. **~1,400 kernels per step averaging 7 µs** is the second structural cost: even under graph
   replay each kernel pays device-side entry/drain; the 305 fp8-quant launches/step (~1.4 ms)
   move ~14–230 KB each — in a fused design they fold into epilogues for free. Fills alone are
   124 kernels/step at bs=1.
4. **Production config questions (fairness):** (a) is the aiter 1-stage fused-AR the best
   choice at bs=1, given 52 µs/rendezvous — vs vLLM custom-AR or other latency-tuned paths?
   (b) 2-stage RS+AG at bs=16's 229 KB payload (49 µs/layer-pair) — plausible but worth one
   sweep; (c) graphs exist only for bs ∈ {1,4,16} in this trace. Any decode A/B must first
   re-tune these — the operator's "not sure we ran optimal settings" concern is warranted.

## Verdict on "easily port M15 to beat production decode"

**No — an M15 port is the wrong tool for decode.** The decode opportunity is real but is a
DIFFERENT megakernel (the plan's Region-4 "different base", now with measured numbers):
a persistent per-layer (or multi-layer) decode kernel that (i) fuses the sliver ops — quant,
glue, fills, small GEMM epilogues — deleting most of the ~1,400 kernel boundaries (~2–4 ms),
(ii) replaces/absorbs the 122 AR rendezvous with an in-kernel latency-optimized flag AR and
overlaps the skew-wait with other work (~3–4.5 ms at bs=1 IF a 14 KB 8-rank AR can run at
~10–15 µs — unmeasured, θ-F13), (iii) leaves MLA and the weight-streaming GEMM cores as-is
(they are near their floors). Ceiling if both land: bs=1 TPOT 10.3 → ~5–6 ms (≈1.8×).

**Cheapest decisive next measurements:** (1) θ-F13 microbench — 8-rank AR latency at
14/57/229 KB: aiter-1stage vs RCCL vs a bare flag/atomic AR of ours (one small rig arm; our
existing m25 bench's stdrccl arm at small payloads + a new flag-AR arm); (2) production
decode config sweep at bs=1 (AR implementation choice) to establish the FAIR decode baseline;
(3) count the per-rank arrival skew directly (the phase ledger's rank-max vs rank-min on a
decode-shaped run).

# M25: the TP8+EP counter-dataflow layer-pipeline megakernel

Author: orchestrator direct design, 2026-08-18. Companion to
TP8_OVERLAP_ANALYSIS.md (the "why"); this is the "what and how". Grounded in
docs/distributed/OVERLAP_ABSTRACTIONS.md (the five knobs, the CTA-role law,
the occupancy-1 selection rule), the shipped primitive layer under
include/cdna4/ops/group/distributed/, and the measured arch card for gfx950.
Everything here is design, pre-measurement; each mechanism carries its gate.

GOAL: beat genuine production (vLLM 0.25.1, AITER MLA, AMD-recommended
TP8+EP) end-to-end at low/mid-concurrency prefill by hiding the two
per-layer all-reduces that production exposes (~25–35% of step time,
estimate pending B0 calibration), plus deleting redundant replicated work
(norms) and filling residual bubbles (shared expert). Decode gets its own
mode, not a pretense that one mechanism covers both regimes.

---

## 0. Naming and scope

- **M25** = the persistent TP8 kernel. v1 scope: per layer, from the o-proj
  GEMM through the next layer's attention INPUT (both boundary collectives
  swallowed). AITER MLA attention itself stays host-launched between mega
  segments in v1 (clean tensor boundary); v2 options in §8.
- **Primitive deliverables** (device-side, Distributed HipKittens layer —
  these are the reusable artifacts the DP/EP mega ALSO needs; extraction
  order in §7):
  - `credit.cuh` — `throttled_store_packet16<Depth>`,
    `throttled_accumulate_bf162<Depth>` + the ISA issue-run verifier
    (K4 as a primitive; today a hand-rolled vmcnt in n2_phase2, the −615 µs
    knob, silently renegotiable by register pressure — exp_38).
  - `slab.cuh` — `certify_slab(slab, epoch)` / `bounded_wait_slab_into(...)`
    (K3; today open-coded two-slab drain+publish at KERNEL:1694).
  - `order.cuh` — `owner_major_staggered(rank)`, `consumer_major`,
    `source_interleaved` index maps (K2; today NC-major open-coded at
    KERNEL:421).
  - `counter.cuh` addition — `counted_arrive_dynamic_into` (runtime fan-in;
    today ~524k open-coded RMWs/rank/epoch in the MoE adapter).
  - Two composite schedules documented WITH the layer (not new mechanisms —
    knob points, per OVERLAP_ABSTRACTIONS §6):
    - **ERS** (epilogue reduce-scatter): K1=epilogue-carried,
      K2=owner_major_staggered, K3=owner-slab certificates, K4=depth-4,
      K5=owner consumes-in-shadow. This is exactly the §6 ring-RS knob
      point; its prototype EXISTS and is measured — the GEMM-RS adapter at
      376.7 µs vs RCCL 421.8 µs on a comm-dominant shape.
    - **MAG** (multicast all-gather, pull-form): owner certifies a reduced
      slab; CONSUMER ranks pull it into local staging with a quota-bounded
      consuming pool (the M15 front-half-combine pattern, −93/−444 µs),
      never a carrier pool (+340 µs law). Signals: per-(owner,slab) epoch
      words via `publish_epoch_relaxed`/`bounded_poll_epoch_word_relaxed_into`.
  - **CDAR** = ERS + MAG = a chunked all-reduce that never exists as a
    phase: producer side rides the producing GEMM's epilogue, consumer side
    rides the consuming GEMM's ramp. One CDAR instance per layer boundary.

## 1. The TP8+EP layer, recast as dataflows

Per MoE layer (58 of 61), N tokens global, hidden 7168 (14 KB bf16/token):

```
[attn: AITER MLA, host-launched, per-rank 16 heads]
      -> per-rank o-proj INPUT (tokens x 2048)
MEGA SEGMENT (persistent, one launch spans all layers' segments):
  (a) o-proj GEMM  (M=tokens, N=7168, K=2048, per-rank slice)
       epilogue => ERS#1: partial rows accumulate to TOKEN-OWNER rank
                   (owner = token_chunk % 8), +residual preloaded in the
                   owner's accumulator, depth-4 throttled, owner-major-
                   staggered order so every link is busy from slab one
  (b) owner-side: RMSNorm on OWNED shard only (1/8 of tokens — deletes the
      8x-replicated norm production pays), then certify_slab
  (c) MAG#1: all ranks pull certified normed slabs into local staging
      (consuming pool, quota-bounded); router gate GEMM (tokens x 7168 x
      256, tiny) runs per-slab as slabs land => top-8 ids/weights local
  (d) expert row-gather (LOCAL — activations replicated; zero dispatch
      comm, the EP-inside-TP dividend) => rolling planner appends dense
      tiles per certified slab (extends the M15 live-count tile_desc)
  (e) expert GEMMs (occupancy-1 256-VGPR bodies, unchanged) + SHARED
      EXPERT as filler: shared-expert tiles are routing-independent, so
      they are the work queue's fallback whenever routed tiles starve
      waiting on (c) — SHARED_EXPERT_FILLER_DESIGN.md realized with zero
      new machinery, it is just queue priority
  (f) expert down-proj epilogues => ERS#2: partial MoE outputs accumulate
      to token-owner (+residual preload again); shared-expert partials join
      the SAME accumulator => AR#2 and the shared-expert AR merge into one
      CDAR (production pays them inside one fused AR too, via row-parallel;
      we match the merge and hide it)
  (g) owner-side RMSNorm (next layer's input_layernorm) on owned shard,
      certify; MAG#2 pulls => replicated normed hidden for attention L+1
[host launches attention L+1 when MAG#2 completes]
```

Dense layers (3) and the MLP path: same CDAR pattern with the MLP GEMMs as
producer/consumer; strictly simpler (no routing).

### Byte accounting per token per MoE layer (aggregate fabric traffic)

| path | production (ring AR x2) | M25 |
|---|---|---|
| boundary 1 | ring-AR 14 KB replicated-sum ~ 2(N-1)B = 196 KB... shared across both: 2 ARs => ~392 KB | ERS#1 7/8 x 14 = 12.3 KB + MAG#1 7 x 14 = 98 KB |
| boundary 2 | (counted above) | ERS#2 ~ (fanout-1) x 14 ~ 60 KB (partials only from expert-owning ranks) + MAG#2 98 KB |
| total | ~392 KB | **~268 KB (-32%)** |

So M25 moves ~a third FEWER bytes than production's collectives AND hides
them. (Fanout ~5.2 distinct expert ranks/token at top-8 over 8 ranks,
uncorrected for EPLB locality; corpus says rank-0 concentration exists, so
real fanout is lower.) All numbers to be re-derived against B0's measured
step composition.

## 2. Knob settings per boundary (the typed schedule)

This table IS the design in five-knob space — and the shape a CAKE schedule
would carry as typed fields:

| boundary | K1 carrier | K2 order | K3 certification | K4 flow | K5 consumer |
|---|---|---|---|---|---|
| ERS#1 (o-proj out) | epilogue `throttled_accumulate_bf162<4>` (arch card: coalesced atomics 52.8 vs stores 54.9 GB/s — viable ON gfx950 ONLY; gfx942 card says store+local-reduce instead — keep the m15b fold-then-wide-push variant behind a knob) | `owner_major_staggered(rank)` — ring emerges, links busy from slab 1, and the 8-writers-per-owner-slab hotspot is spread by construction | owner slab words (`slab.cuh`), slab = token-chunk x 7168, cut aligned to consumer vector shape | depth 4, verifier-gated | owner's reserved-C pool consumes (residual+norm) in producer shadow |
| MAG#1 (normed hidden) | consumer PULL via `load_peer_packets` into local staging (pool job = consuming certified work — the winning pool economics; never a push/carrier pool) | pull in gate-consumption order (`consumer_major`) | per-(owner,slab) epoch words | pull is load-stream, no injection bound needed; per-link knee 8 CTAs respected by pool sizing | quota-bounded pool + gate GEMM per landed slab |
| ERS#2 (MoE+shared out) | same as ERS#1; contributors = dynamic per-slab fan-in => `counted_arrive_dynamic_into` (fan-in = #expert-ranks serving that slab + 1 shared) | staggered by expert-owner | owner slab words | depth 4 | owner pool: residual+norm in shadow |
| MAG#2 (next-layer input) | pull | attention-consumption order | epoch words | — | v1: pool prefetch, then host attention (exposed tail, §8) |

## 3. Grid organization, roles, planner

- 256 persistent CTAs, occupancy-1, `__launch_bounds__(256,1)` unchanged.
- Static role split (finish_order_partition stays rolled back — 24 B/lane
  spill precedent): C reserved CONSUMING CTAs (MAG pulls, owner norm,
  residual init, slab certification bookkeeping), 256−C−1 GEMM bodies,
  1 rolling planner CTA. C chosen by the response-curve method
  (OVERLAP_ABSTRACTIONS §4.3): same-session sweep {8,16,24,32} x quota —
  never a default.
- Rolling planner: M15's single-CTA planner + live-count tile_desc,
  extended to APPEND tiles as MAG#1 slabs certify (routing for slab k is
  known the moment gate(k) finishes). Work queue priority: routed tiles,
  else shared-expert tiles, else next-boundary consumer work. HBM ticket
  atomics as today.
- ONE grid barrier per launch edge (T3 law). Layer-to-layer transitions
  inside the mega are certificate-gated, never barrier-gated. Buffer reuse
  across layers via `lifetime.cuh` slot retirement (payload-ready vs
  may-overwrite separated; 2-deep accumulator ring per boundary:
  2 x 2 x tokens x 7168 x 2 B ~ 940 MB at 16K tokens — fits gfx950's
  288 GB comfortably next to weights+KV).

## 4. Numerics and parity policy

- ERS accumulation in bf16 packed atomics matches the existing combine's
  numerics class: run-to-run nondeterministic contribution order — gated by
  the measured rel-err envelope discipline (G0a-style), never bit-SHA.
  STRICT knob: fp32 accumulators (2x ERS bytes and 2x buffer — priced, off
  by default). Production's RCCL AR also reorders; class parity, not bit
  parity, is the honest bar and matches campaign v5.1's accuracy machinery
  (c1det per numerics class + cross-class agreement).
- Coalescing REQUIREMENT for the atomic path: accumulate along contiguous
  hidden (rows are contiguous 14 KB — natural). Scattered-atomic cliff is
  13x (52.8 -> 4.1 GB/s); the tile-to-packet mapping must be audited in ISA
  (issue-run gate) not assumed.

## 5. Where the wins come from (Amdahl ledger, all pending calibration)

| lever | mechanism | sized by |
|---|---|---|
| hide AR#1+AR#2 | CDAR (ERS in producer drain + MAG in consumer ramp) | ~25–35% of production's step is exposed collective time (B0 measures the real number); minus ~1/n_slabs ramp/tail |
| move fewer bytes | -32% fabric traffic (§1 table) | headroom against link saturation, not direct time |
| delete replicated norms | owner-shard norms: 8x less norm work + one less full HBM pass per boundary | small but free |
| shared-expert filler | starvation windows filled with routing-independent tiles | bounded by gate-wait bubbles; measured in M15-class ledgers |
| merged shared+routed AR | one CDAR where production... also merges — parity, not a win; claim zero | — |
| rolling planner | routing overlapped with MAG#1 landing (production serializes gate->sort->group kernels) | vLLM kernel-gap time in B0 profile |

Honest prediction band vs vLLM TP8+EP native at c32p prefill: **+20–40%**,
dominated by the first row. Decode cells: NOT claimed; separate mode (§8).

## 6. Gate ladder (each gate kills or funds the next stage)

- **G25-0a (exists, done)**: GEMM-RS adapter beats RCCL (376.7 vs 421.8 µs)
  — ERS producer economics proven on-fabric.
- **G25-0b (1 day, node CPU+GPU micro)**: extend the GEMM-RS harness to
  (i) accumulate-variant vs store+local-reduce at 7168-wide rows,
  (ii) MAG pull bandwidth co-resident with an MFMA body (does the pull pool
  degrade GEMM >3%?), (iii) one-shot direct-write AR latency at decode
  sizes (2–64 rows) vs RCCL. Go/no-go for the K1 choices.
- **G25-1 (MoK-class rig)**: single-boundary CDAR — one expert-GEMM ->
  ERS#2 -> MAG#2 chain vs GEMM + RCCL-AR sequence, same tokens/routing.
  PASS = >=90% of the AR time hidden, rel-err inside envelope, issue-run
  gate clean. This is the decisive experiment for the whole design.
- **G25-2**: two-layer persistent segment with rolling planner + filler;
  phase ledger proves no exposed collective phase.
- **G25-3**: vLLM integration behind default-0 macros (new integration
  mode), build-gate discipline as M24, then campaign v5.1 arms
  (m25 vs native_tuned_tp) under the full protocol + fairness audit.

## 7. Extraction order (primitives first, from PROVEN code)

1. `credit.cuh` from n2_phase2's hand-rolled vmcnt (+ verifier script) —
   benefits M15/M24 immediately (the bound is currently fragile).
2. `slab.cuh` from KERNEL:1694's two-slab protocol; `order.cuh` from
   KERNEL:421's NC-major decode; `counted_arrive_dynamic_into` from the
   MoE adapter's open-coded fan-in. Each lands with its negative-control
   test per the verification-as-layer rule.
3. GEMM-RS adapter re-expressed on 1+2 (no behavior change, gate:
   byte-identical schedule, same 376.7 µs) — proves the primitives carry
   the measured wins.
4. M25 v1 composed from the layer. New open-coded logic is ONLY: rolling
   planner append, owner norm fusion, work-queue priority. Everything else
   is primitive composition.

## 8. Risks and the v2 doors

- **Exposed MAG#2 tail (v1's structural bubble)**: attention is
  host-launched, so the last slabs' pull cannot hide under it. Sizing:
  ~1/n_slabs of MAG#2 + launch gap. v2 doors: (a) bring MLA attention
  in-kernel (HipKittens attention bodies exist; largest lift), (b) chunked
  attention launches on a hipGraph with per-slab events (medium),
  (c) persistent-attention co-kernel sharing the certificate space (the
  T3-style two-kernel handshake). Choose after G25-2's ledger shows the
  tail's real cost.
- **8-writer atomic hotspot per owner slab**: staggered order spreads it;
  fallback K1 = m15b fold-then-wide-push (stores + owner local reduce),
  already a knob.
- **VGPR pressure from epilogue additions**: issue-run gate + the exp_38
  lesson (a bound that can silently die is not a primitive) — the verifier
  ships with credit.cuh, non-negotiable.
- **Decode regime**: chunking collapses below ~2K tokens. M25_DECODE mode:
  one-shot direct-write AR (latency-optimal, bytes irrelevant at that
  size), small-M tile variants — separate design note after G25-0b(iii).
- **EP row-gather locality**: replicated activations make gather local but
  scattered; corpus says hot experts concentrate — gather order should
  follow `source_interleaved` to spread HBM banks. Minor.
- **The fill/dummy lever does not exist under TP8** (no DP lockstep): M24
  remains the DP-topology weapon (B3's regime); M25 does not carry it.

## 9. Relationship to the DP/EP mega (shared foundation)

M15/M24 and M25 become two schedules over ONE primitive layer: same
credit/slab/order/counter/lifetime substrate, same GEMM bodies, same
planner core, different knob tables (M15: all2all dispatch/combine + seal;
M25: CDAR x2 + rolling planner). That is the Distributed-HipKittens story
the primitive extraction (§7) makes real — and it is why building M25 makes
the DP kernel BETTER (credit.cuh alone hardens the −615 µs knob it already
depends on).

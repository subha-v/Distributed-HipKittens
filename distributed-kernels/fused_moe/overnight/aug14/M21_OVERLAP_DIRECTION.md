# Why the MoK win didn't transfer, and where the overlap program goes (2026-08-14 ~23:00 UTC)

Decision record following the m20 serving pair (−19.6%, receipts green; see
M20_SERVING_RESULTS.md), a source-level study of MORI-EP (ROCm/mori) and
MoonEP (MoonshotAI), and new evidence pulled from the banked artifacts.

## 1. The paradox, resolved

"We have kernels that fix imbalance; production has none; why do we lose?"

**(a) In the m20 config we mostly didn't deploy the imbalance fix.** 12/58
layers had the replica cache; 46 ran the plain fused mega. The arm was
mostly a fusion-only arm under production routing.

**(b) Skew's cost is compute concentration, not communication.** At the
measured ~5.5x receive skew, the critical path is the hot rank's GEMM;
comm is a minor term on that path. Comm/compute overlap (fusion's whole
dividend) can only hide comm — under skew there is almost nothing left to
hide, the fused design's fixed costs (rendezvous, planner width, polling)
remain, and replication is the only lever that moves the actual bottleneck
(it relocates compute). That is why m19 = fusion+replication-everywhere won
+39.8% and fusion-alone cannot win the skewed regime.

**(c) Why fusion-alone is WORSE than production in serving when the
aggregate-skew replay said 0.85x — evidence, not vibes:**
- The real per-chunk max-rank-load distribution is TIGHT (n=90,050 calls:
  p50 5.15x, p99 6.35x, max 6.55x). It is NOT a fat tail of extreme
  chunks; the typical real chunk is close to the replay's aggregate skew.
  So the divergence is not skew MAGNITUDE — it is chunk STRUCTURE.
- i.i.d. histogram replay draws each token independently → destinations
  interleave by chance. Real routing is run-correlated (top-10 token types
  carry 26.4% of traffic; same-prompt tokens route alike) → consecutive
  tokens target the same expert/rank. MORI's own intranode source
  documents the consequence on this exact fabric: consecutive
  same-destination traffic drives only 2–3 of the 7 xGMI links (their fix,
  LDS round-robin interleave, measured 822→497.7 µs, 65%), and tight poll
  loops livelock the fabric (they insert s_sleep backoff). Our M7
  remote-RMW combine and LL128 dispatch are exactly the transports that
  degrade under run-concentrated destinations; the i.i.d. replay
  structurally cannot show this.
- The captured-route replay that would have caught it (M19_DESIGN's
  validation contract) WAS NEVER RUN — all 74 MoK result dirs are agg/bal.
- Serving measurement floor: stock self-varied 9.6k–12.7k tok/s across one
  day (±15%); the m15-vs-stock synthetic (balanced) pair measured FLAT
  (10,702 → 10,627), i.e. even the real 1.32x balanced kernel win is
  invisible at n=1 through the serving stack (MoE ≈ 40–54% of step; gate
  self-test double-traffic; eligibility gating). Only 4x-class MoE effects
  (m19) clear the serving noise floor at n=1.

**Per-router-call check:** inter-router-call gaps in the m15pair1 skew
dumps are statistically identical between arms (p50 23.12 vs 23.13 ms), so
the shim adds no host-side drag; the effects are device-side.

## 2. What the systems we studied actually do (source-level)

**MORI-EP** (ROCm/mori): comm kernels only; dispatch/combine are separate
launches; NO comm/compute overlap anywhere in-kernel (delegated to DBO,
which the blog lists as unfinished); NO load balancing (delegated to vLLM
EPLB). It is a fabric-saturation library. MI350X EP8 tuned ceilings @4096
tok: dispatch fp8 342 GB/s (256 CTAs), combine bf16 366 GB/s, combine with
fp8_direct_cast ON WIRE 642 GB/s effective, zero-copy combine 435 GB/s at
just 56 CTAs. Discipline lessons: link interleaving mandatory, spin
backoff mandatory. Our fused design is beyond their architecture; their
numbers are the comm budget envelope for any successor kernel.

**MoonEP** (MoonshotAI): TRAINING-FIRST. Its famed balance is a per-step
centralized planner (rank 0) that rebalances to an exact S*K-token cap per
rank (greedy surplus/deficit transfer matrix; replicas bounded at B=E/R),
with the plan broadcast via NVSwitch multicast and replica weights pulled
PER STEP via TMA pipelines — the per-step weight motion needs NVLink-class
bandwidth (independently confirms our pull/push/cache trilogy: infeasible
on xGMI; persistent cache was right). Its overlap is NOT fusion: separate
kernels chained with PDL plus a priority comm stream that overlaps
dispatch/combine with attention and the shared expert. Transferable
steals: duplicate-top-k dedup, source-side fp32 pre-reduce before combine,
static padded shapes, the surplus/deficit capacity argument.

**CAKE (arXiv 2608.12629)**: NVIDIA-only compiler-agent work, but two
transferable rules: decompose API-level vs GPU-span wins (their fused MoE
"6.2x" was 1.2x in GPU-span — launch-gap removal, not execution), and
per-domain kernel PORTFOLIOS with anti-leakage validation — tuning on
replayed aggregate skew and deploying on real per-chunk skew is precisely
the leakage failure their dispatcher stage forbids.

## 3. The program

**Track A — close serving honestly (cheap, do first):**
1. EPLB baseline pair: stock vs stock + `--enable-eplb`
   `num_redundant_experts=128` (= 48 slots/GPU, the same budget as our
   EL=48). If EPLB recovers most of +39.8%, the serving story for our
   kernel shrinks and we must say so; either way it is the mandatory
   baseline for any claim.
2. Captured-route replay (the never-run decisive experiment): dump raw
   topk_ids via the skew hook, replay real chunks through
   m15/m18/production in MoK; separately, a run-structure A/B (sorted vs
   shuffled token order at fixed histogram) isolates the link-concentration
   mechanism.
3. m20 fix if we still want the serving arm: per-chunk dispatch portfolio
   at the DECIN seam (chunk stats are already computed pre-launch): route
   each chunk → replicated-mega / plain-mega / STOCK production path;
   uncached layers default to stock. Any claim: ≥5 order-balanced pairs
   (protocol doc; stock variance demands it).

**Track B — the successor kernel's overlap surfaces (ranked):**
1. **Shared-expert as filler compute**: DeepSeek's shared expert (~1/8 of
   routed FLOPs) currently runs serially outside the mega; fold it in as
   the work CTAs execute during dispatch/combine waits. New overlap that
   no measured system exploits in-kernel.
2. **fp8-on-wire combine**: MORI's measured 366→642 GB/s lever; our M7/M8
   transport is bf16. Nearly halves combine bytes.
3. **Source-side pre-reduce** (MoonEP): fold same-token multi-expert
   partials at the source before the remote RMW — up to topk-fold fewer
   remote ops.
4. **M21 epoch pipelining**: 2-in-flight epochs (M1 of N+1 under M8 of N);
   the parity-doubled counters already permit it structurally.
5. **Fabric discipline retrofit**: destination-interleave the M7 RMW
   stream under run-correlated routing (m17's RR idea, re-evaluated on
   captured routes, not i.i.d. draws); s_sleep backoff in every poll.

**Track C — training as the main lab (recommended).** Aux-loss/bias
balancing keeps expert loads near-uniform (DeepSeek-V3 uses aux-free bias
+ sequence-level aux; forced-balanced is AMD's own benchmark convention) —
the regime where our 0.756x balanced number lives and imbalance is not our
problem to solve. Static 4096-token microbatches: no chunk raggedness, no
eligibility gate, controlled harness → MoK-fidelity measurement (low
transfer risk — the exact failure mode of serving). The MoE fraction of
step time is far higher in the 4-layer all-MoE proxy. And backward is an
overlap goldmine with no serving analog: wgrad has NO comm dependency —
it is perfect filler compute for dgrad dispatch/combine waits inside a
fused backward — plus grad-reduce overlap. That MoonEP, the strongest EP
system published, is itself training-first is corroborating evidence that
this is where per-step balance + overlap research pays. T0 scaffold is
ready (rocm/primus:v26.5, shipped DeepSeek-V3 config = our exact geometry,
vanilla + turbo baselines).

Sequencing: A1+A2 next node window (cheap), main effort to C with B's
items 1–4 (they all apply to the training forward too; wgrad-filler is the
new frontier), A3 only if the serving arm stays in the paper.

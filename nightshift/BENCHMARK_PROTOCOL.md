# Serving benchmark protocol (operator-approved, 2026-08-18)

Binding for every end-to-end campaign tonight. Source: operator's methodology
session ("the fairest protocol I can construct — informed by every way this
project has fooled itself so far"). Three anchors: the strongest honest
baseline, the most production-like workload, and separation of what you claim
from what you measured. Supersedes weaker practice anywhere it conflicts
(including n=2 pairs).

## 1. Baseline: strongest shipped production, possibly two of them

- **native-default**: the untouched vendor image
  (`vllm/vllm-openai-rocm:v0.25.1`, digest
  `sha256:84459732ca98b40fe2f5338a3f050be6d522504e47a484a5180d58fb75956f86`),
  `vllm serve` with the model's documented flags — what a user gets by
  default.
- **native-tuned** (worth adding): the same untouched image at AMD's
  published recommended deployment settings for this model (their
  MLPerf/blog configs). Fairness cuts both ways — beating a lazy default
  isn't beating production. If the vendor documents a faster config, that's
  the bar.
- **Symmetry rule** for improvements we discovered: the uniform-decode
  rescue roughly doubled throughput, and it's our patch. It must never hide
  inside the kernel's number. Either give it to the baseline too (it's
  upstreamable — that's the three-arm decomposition: native / patched-stock
  / m15) or count it explicitly as "integration effect". The kernel claim is
  only ever m15/patched-stock; the headline is m15/native, with the
  decomposition shown so a reviewer can see which part is kernel and which
  part is engineering both sides could adopt.

## 2. Workload: real prompts, and open-loop arrival

- **Real text** (MLPerf QSL) so routing carries genuine skew; never
  synthetic tokens for a headline.
- **Open-loop cells**: today's corpus showed 31% of steps in the closed-loop
  c32p cell are dummy batches — partly an artifact of concurrency-32 with
  OSL=8 (ranks idle at the tail of every wave). Real production serving is
  arrival-driven. The fairest prefill setting is a request-rate sweep
  (e.g. Poisson arrivals at 50/75/90% of saturation) alongside the saturated
  closed-loop cell: it measures both peak throughput and behavior at
  realistic utilization, and it naturally decides whether the pad/dummy-skip
  advantage is production value or a benchmark artifact.
- **A grid, not a point**: several ISLs and concurrencies, including cells
  where we lose (below the ~1,700 token/rank break-even). The honest
  deliverable is the per-cell table plus one "deployment policy" row — mega
  where it wins, stock elsewhere — because that hybrid is what anyone would
  actually ship, and it's a claim no cherry-picked cell can fake.

## 3. Measurement discipline (each item earned the hard way)

1. Fresh server per arm, order-balanced pairs, **>=5 pairs**, uniform
   cooldowns — the ±18% position effect and ±15% day drift make anything
   less anecdotal.
2. Identical token streams, SHA-verified across arms; same client, seeds,
   warmup.
3. Coverage receipts on the candidate — proof the kernel ran the traffic
   (the pre-M23 era measured an inert kernel for weeks).
4. Accuracy gate: outputs validated (MLPerf AccuracyOnly A/B, or bounded
   rel-err where kernels legitimately differ numerically). A fast wrong
   kernel is a zero.
5. Metrics: aggregate input tok/s (prefill throughput), TTFT p50/p99 (what
   users feel), tok/s/GPU — every number with its complete workload spec
   attached. Median-of-pairs with the spread shown; claim sized against
   measured drift.
6. Costs priced in: memory footprint, KV-cache impact, any replication — in
   the same table as the speedup.
7. Independent adversarial audit with veto before any claim ships (the
   nightshift auditor) — someone whose job is to find the unfairness,
   checking baseline authenticity by evidence (image digest, env, absence of
   patch markers), not intent.

## Notes for tonight's execution

- >=5 pairs supersedes n=2: camp3's +2.7–5.2% (actually +2.70% mean /
  +3.61% geomean, spread +23.5%/−13.1%) is directional only under this
  protocol, never a headline.
- Known open problem: `ordered_output_token_id_stream_sha256` differs even
  stock-vs-stock in camp3 — sampling nondeterminism. The campaign must force
  deterministic decoding for the SHA check or route accuracy through the
  gate-4 path; unresolved parity = no claim.
- Tier-1 dummy-skip wins must be reported per-regime: closed-loop saturated
  vs open-loop utilization cells, so the artifact share is visible.
- Prior-session forensics source (M23 discovery chain, patch/campaign
  forensics, corpus analysis, every measurement with context):
  `/Users/subha/.claude/projects/-Users-subha-repos-Distributed-HipKittens/4077c4da-1965-47cb-aa86-44e5f55b46c4.jsonl`
  (4.3 MB JSONL). Never read wholesale — dispatch an Opus subagent to
  grep/extract specific facts into a compact report.

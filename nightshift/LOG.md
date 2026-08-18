# Nightshift LOG — 2026-08-18 overnight

## 01:12 PDT — session start, Wave 1 launched
Orchestrator (Fable) online. Three background Opus workflows launched in parallel:
- `recon-harvest` (wf_6ad01aab-a7e): read-only harvest of ~/eplb_campaign camp logs
  (order-balanced ratios + receipts only) + node state snapshot (foreign procs,
  containers, image digest, pins, route corpus, disk/mem). No kills yet.
- `fill-aware-design` (wf_59cdde7f-f54): architect writes
  aug18-prefill/FILL_AWARE_DESIGN.md (cost model, mechanisms, parity, macro plan,
  build-gate, MoK gate w/ R0a arbiter, work-list) → 3 adversarial verifiers
  (parity / hardware / measurement lenses) → revise + commit.
- `cleanup-resume` (wf_c23e9d34-05c): finish the pre-M23 void-numbers cleanup in
  both repos, tombstones, SERVING_BENCHMARK_METHODOLOGY.md, README updates,
  commit+push (reconciling the f4968a8x WIP checkpoint).

## 01:17 PDT (08:17 UTC) — recon landed; QUEUE KILLED; R6 launched
Recon (wf_6ad01aab-a7e) findings:
- Camp3 DONE, harvested from ~/20260818_m15_campaign_m23pair2 (c32p, SHA-matched
  prompts b6271f04…): pair_01 m15-first +23.49% (19277.4 vs 15611.2), pair_02
  stock-first −13.06% (17900.0 vs 20588.7). Order-balanced n=2: mean +2.70%,
  geomean +3.61%. Coverage receipt pair_02 m15 rank1: sealed/in_bucket = 251/259
  = 96.9% (NOT the banked 99%). vs PATCHED-stock only — no native arm exists.
- CRITICAL METHOD GAP: ordered_output_token_id_stream_sha256 differs across ALL
  arms including stock-vs-stock → exact-token SHA cannot currently validate
  accuracy parity. Must fix (deterministic sampling) for tonight's campaign.
- Camp1 killed (RC=143), camp2 never ran, capture DONE (route corpus 8 npz,
  49MB, present). camp_grid/4/5 were 0-byte waiters about to fire (~08:25 UTC).
- Node: GPUs idle, 700G disk free, 2.9T RAM free. Native-image digest receipt:
  vllm/vllm-openai-rocm:v0.25.1 = sha256:84459732ca98…56f86. 13 DHK-* pins incl
  ~/DHK-m15. fabric_build/ present. Node DHK checkout 4 days stale (604a9763).
- ANOMALY: apply.py NOT in ~/eplb_campaign/ (only coverage/m23/rr patches);
  copies live in ~/pf4h_vllm_20260729/*/shim/pf4h_integration/apply.py. Must
  confirm patch-chain head before any integration build (Wave 2 gate).
- Foreign but GPU-free: subha_k1, yuhan_dsv4_0806 containers — left alone.

ACTIONS: killed camp_grid/camp4_rr/camp5_native waiters at 08:16 UTC (bracketed
pkill, RC=0 each; pgrep clean; zero data loss — nothing was mid-measurement).
t1full3 waiter (PID 1869717) inspected: passive log-grepper, never takes GPUs —
left alive. No GPU_CLAIM existed. Node is now exclusively ours.
Launched r6-fill-evidence (wf_aa479070-846): R0a arbiter (must repro 0.7556x
±1%) then R6 T-sweep 4096/2048/1024 per DECOMP_RUNBOOK — kernel-level test of
the fixed-cost/fill hypothesis while the design workflow finishes.

## 01:25 PDT — CORPUS_FINDINGS.md folded in; design workflow restarted
Operator pointed at nightshift/CORPUS_FINDINGS.md (new): 31% of sealed calls
are 100% dummy work (full 4096-row MoE for zero real tokens, both arms;
production's captured graph structurally cannot skip); real-row skew 2.61x p50
onto rank 0 (experts 0-7); run-correlation ~nil → fabric arms demoted.
Reprioritized the fill design as TWO-TIER: (1) in-kernel dummy-step early-out
(graph-capture-safe, collective-contract-preserving), (2) n_orig-proportional
work on partial steps. Combined potential ~2.7x less MoE work than production.
Stopped wf_59cdde7f (was ~10 min in), enriched architect+reviewer briefs with
the corpus facts + finalize-contract parity question + apply.py location
anomaly, relaunched as task w7ovje25v (same run id, resumed). RR placement
(rr_patch.py, attacks the 2.61x skew) noted as a candidate e2e arm later.

## 01:32 PDT — operator delivered the benchmark protocol; banked as doctrine
Written to nightshift/BENCHMARK_PROTOCOL.md (binding). Deltas vs prior plan:
- >=5 order-balanced pairs (n=2 is anecdote under drift ±15%/±18%).
- Second baseline: native-TUNED (AMD's published recommended config), not just
  native-default — beating a lazy default isn't beating production.
- Open-loop cells (Poisson at 50/75/90% saturation) alongside closed-loop:
  the 31% dummy share is partly a c32p/OSL-8 wave-tail artifact; open-loop
  decides if dummy-skip is production value. Report tier-1 wins per-regime.
- Real prompts (MLPerf QSL) for headlines; grid incl. losing cells + a hybrid
  "deployment policy" row; TTFT p50/p99 + tok/s/GPU; costs in the same table;
  accuracy gate (deterministic decoding or AccuracyOnly A/B) — the
  stock-vs-stock SHA mismatch must be resolved before any claim.
- Prior-session transcript available for forensics (4.3MB jsonl, path in
  protocol doc) — extract via Opus subagent only, never read wholesale.
Follow-ups queued: fold protocol into SERVING_BENCHMARK_METHODOLOGY.md after
cleanup agent lands; campaign-builder must verify harness support for
open-loop/request-rate mode + MLPerf QSL prompts + deterministic decoding.

Plan: on recon report, decide queue-kill timing (mid-flight arm finishes, then
teardown once first verified fill-aware build is ready — or earlier if the queue
tail is low-value vs getting R6/T-sweep evidence). Wave 2 = implementation
workflow (implement → adversarial verify → fix, compile-only on node while GPUs
busy). Wave 3 = MoK gate (R0a then fill sweep). Wave 4 = e2e campaign vs GENUINE
native baseline, three-way decomposition, cells c32p + c8/c16/c32 (+c512p if
memory), fairness-audit agent before any quoted claim.

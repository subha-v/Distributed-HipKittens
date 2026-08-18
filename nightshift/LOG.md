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

## 01:40 PDT — methodology rev3 landed; campaign prereqs launched
- SERVING_BENCHMARK_METHODOLOGY.md rev 3 committed (b163f93e): protocol folded
  in, >=5-pair hard floor (camp3 demoted to directional-only), native-tuned
  arm + symmetry rule (c3), accuracy gate w/ SHA-nondeterminism open problem,
  open-loop cells, deployment-policy row, fairness items 2b/4b added.
- nightshift commit 0300ebb7 pushed (LOG, BENCHMARK_PROTOCOL). Memory updated
  (nightshift-orchestrator-2026-08-18).
- Launched campaign-prereqs (wf_2a9041f7-7ae), two parallel Opus agents:
  (1) web research: AMD's published recommended R1/MI350X vLLM config for the
  native-tuned arm, cited, with proposed launch command + version caveats;
  (2) harness recon (read-only, local+node): client identity, open-loop
  support, prompt realism (QSL?), determinism diagnosis for the stock-vs-stock
  SHA mismatch, metrics coverage, pairing machinery, change list for the
  protocol campaign.
Still in flight: fill-aware-design (w7ovje25v), r6-fill-evidence (wf_aa479070).

## 01:52 PDT — campaign prereqs landed; campaign-v5 build launched
Prereq findings (full reports in tasks/w3clrpi59.output):
- FAIRNESS EXPOSURE FOUND: v4's "native" arm inherits OUR serve flags
  (gpu-mem-util 0.70, MNBT 4096, max-num-seqs 128, opt-level 2) — it is a
  config-mirrored diagnostic, not shipped-defaults production. Renaming it
  native_mirror; adding genuine native_default + native_tuned arms.
- AMD's documented bar: TP8+EP for <=128 conc (our c8-c32p cells!), DP8+EP
  only >=512 (+16-47% claim), AITER MLA 1.2-1.5x — ranges that swallow our
  +2.7-5.2%. Headline must be vs per-cell best native. ATOM stack (AMD's
  fastest R1 path) can't run in the untouched image — flag as known ceiling
  in REPORT. No absolute AMD tok/s anchor exists for R1/MI350X vLLM.
- Harness: custom closed-loop client (bench_exact_token_ids_v2.py), real
  MLPerf QSL prompts already, temp-0 seeded; stock-vs-stock SHA divergence is
  NUMERIC (chunked-prefill boundaries, prefix cache, DP-rank assignment,
  combine order) — perf cells can't be deterministic; fix = separate c1det
  accuracy cell (conc 1, no prefix cache/chunked prefill). >=5 pairs is pure
  config (--pairs 5). Node summarizer reads WRONG KEYS (silent zeros) — new
  analyzer required. Open-loop = ~60-80 new client lines.
Launched campaign-v5-build (wf_c23ee898-84d): client v3 (open-loop + ITL),
wrapper v5 (native_default/native_tuned_tp/native_tuned_dp/native_mirror,
c1det gate, pairs=5, open-loop cells), v2-schema analyzer, ARMS.md with
per-flag citations, NIGHT_SCHEDULE.md with a fits-by-morning trim option —
implement -> 2-lens adversarial verify -> fix+commit. Node deployment held
until I dispatch measurement.

## 01:55 PDT — operator: approved TP8+EP testing for low concurrency
Operator confirms native_tuned_tp arm and is open to SERVING with TP8+EP.
Campaign build already includes it; headline stays m15 vs per-cell best
native; deployment-policy row will surface TP8+EP-below-break-even if that is
what the data says. Mega integration remains TP1/DP8/EP8 tonight (porting the
mega to TP8+EP is out of overnight scope).

## 02:45 PDT — DESIGN GO (rev 2, 1329aebe); M24 implementation launched
Design workflow done: 3 lenses returned 7 blocking / 24 major / 12 minor; all
5 distinct blockers FIXED, none rebutted; go=true, blocking_open empty.
Mechanism (M24): descriptor slot 71 fill vector {magic,gen,n_orig[world]}
device-sourced from the DP all-reduce (B3 fix); T_eff at exactly 4 row-count
sites; gen staleness fails OPEN to 4096; tier-1a dummy rank publishes
rows_done + 8 chunk_ready cells at fill 0 (peers skip whole segment);
tier-1b NULLWORK keyed on chunk_ready ground truth; ZERO_PAD determinism.
9 macros default-0. Estimate: rho~0.44 => +25-40% e2e central +31%; tier-1
alone +12-17%.
DESIGN CORRECTIONS THAT CHANGE CAMPAIGN/EVIDENCE PLANS:
- Production does NOT scale with fill (VLLM_MOE_SKIP_PADDING ships OFF) —
  add stock+SKIP_PADDING=1 control arm to campaign; fill-awareness is pure
  structural advantage (fairness risk #1, wording matters).
- Bit-identity WITHDRAWN: kernel already run-to-run nondeterministic (bf16
  RMW combine order) — all parity gates become measured rel-err envelopes
  (new G0a); m15 arm can never SHA-match; c1det only validates
  deterministic-numerics arms.
- R6 T-sweep = BATCH-SIZE sweep (K0_MAXTOK=K0_T shrinks capacity too), NOT a
  fill sweep — real fill evidence is the F.3 NORIG_CONST ladder at capacity
  4096 (needs M24 build). Label wf_aa479070 results accordingly.
- C is runtime-config (reserved_comm_ctas), not compile-time 28 => fill-aware
  C possible without recompile. No empty GEMM tiles exist (tiling already
  dense) — the fix is killing garbage rows at M1 dispatch.
- Spin-headroom risk: dummy ranks now spin harder (mode-14 L2 law); monitor
  SPIN_DBG slot 52 every arm; POLL_BACKOFF bundle if fail_max>0.
Launched m24-implement (wf_6bbc1bff-8a2): implement (17 steps, both-config
node compiles) -> 2-lens adversarial diff review -> fix + node build-gate
(disassembly-identical) + commit. GPUs untouched until gates.

## 02:55 PDT — R0a PASS; R6 blocked-by-correctness; NODE DOWN
r6-fill-evidence (wf_aa479070) results (full: tasks/wvdrblr5d.output):
- R0a ARBITER PASS: 0.75187 median-of-5 order-balanced (-0.49% vs banked
  0.7556, inside ±1%), pin DHK-m15 sha 935f555e, gate pass all runs,
  rel_err 0.0083. Harness + anchor are sound; ratchet unconfounded.
- R6 on the M15 body is UNRUNNABLE: chassis is NOT T-parametric — at
  T=2048/1024 mps_mega FAILS the MoK correctness gate (rel 0.874/0.839)
  while production passes (0.0058) in the same runs. tgen body control
  passes at T=2048 (0.8959) => body-specific. The needed kernel work IS M24
  (NORIG_CONST ladder at fixed capacity 4096 — no T-parametric dependence).
- PIN TRAP REALIZED IN ARCHIVE: banked t2048/t1024 "m15" results are
  actually the tgen body (sha 20b8c6bb) — must never be quoted as M15.
- Harness regression at amd-master HEAD (_m20 NameError blocks all
  non-replication MoK arms) — worked around on node via scratch tree;
  proper 1-line fix + runbook amendments (PADMAX=64T+992, R6 scope, ledger
  servicedrain bug, G=3 inert) dispatched as wf_98e0d1c3.
- [MPS SPIN] fail_max=0 at T=4096 100%-fill — no transport degradation.
- NODE UNREACHABLE since ~09:00 UTC (both IPs, confirmed again 09:55 UTC).
  tgen same-pin T-sweep results stranded on node (~/nightshift_r6/tgen_t*/
  summary.json) — 2-min retrieval when node returns. Background watch
  bkpkhvmfu polls every 60s and notifies on recovery. M24/campaign node
  steps (compiles, build-gate) will report blocked; resume on recovery.
  Local implement/verify work continues per doctrine.

## 03:00 PDT — harness+runbook fixes landed
_m20 hoist committed amd-master ed08279a (pure 1-line add before the
REP_EXPERTS block, py_compile pass); DECOMP_RUNBOOK amendments a-f committed
DHK 00501cf1 (new §10 + in-place corrections: R0a re-anchor, PADMAX=64T+992,
R6 scope banner, archive pin-trap warning, ledger servicedrain caveat, G=3
inert note). Both pushed.

## 03:05 PDT — campaign v5.1 READY (a776ac75); night replanned around node window
Build workflow: 12 blocking + 14 major findings across 2 lenses, ALL fixed
(REVIEW_NOTES.md). Key fixes: c1det bit-exactness only within numerics-class
(m15 vs stock; cross-class = agreement fraction), TP8-arm readiness count
(would have 1800s-hung + voided native arms), analyzer refuses unknown
schemas (no silent zeros), native arms denied PF4H env AND mounts, rotation-
balanced pairs (pair count must be multiple of arm count; 6 not 5),
fail-closed seal-coverage gate (>=90%). Deployment steps + Option B schedule
(B0 calibrate -> B1 accuracy gate -> B2 6-pair matrix) in NIGHT_SCHEDULE.md;
full open-issues list in the task output (tasks/wp5jobxxv.output).
CAVEATS BANKED: tonight's absolutes not comparable to camp3 (prefix caching
now off everywhere); prompt SHAs new-generation (no cross-night exact-token
compare); open-loop rate bounded by slowest arm.

NIGHT REPLAN (node still down, ~5h GPU window left at best):
On node return, in order:
1. Retrieve stranded tgen T-sweep summaries (~/nightshift_r6/tgen_t*/).
2. Deploy campaign_v5 (staging cmds in task output); grep a banked
   server.log for exact RAGGED_SEAL_RECEIPT printf format (parse check).
3. Resume M24 node steps (compiles + build-gate; CPU-only, overlaps GPUs).
4. M24 MoK GATES FIRST (R0a ratchet, G0a envelope, F.3 NORIG_CONST ladder,
   ~1-1.5h): the F.3 ladder is the decisive fill evidence, and an M24-
   enabled m15 arm is the only candidate that can plausibly hit +20-30%.
   Current M15 vs native_tuned_tp likely loses — an honest result but not
   worth the whole window at 6 pairs.
5. If M24 gates PASS: build M24 serving pin, campaign m15 arm = M24-enabled.
   If FAIL/no-time: campaign runs current M15 (still the first-ever genuine
   native comparison) and M24 e2e becomes next iteration.
6. B0 -> B1 -> B2 sized by clock at B0 completion (2-arm 6-pair cut
   {m15, native_tuned_tp} if needed; never cut below 5 pairs; stock control
   cut = stated limitation). Campaign may run past morning; REPORT.md is
   written at ~07:30 PDT from completed pairs (within-pair ratios valid per
   pair) and marked "campaign continuing".

## 04:50 PDT — M24 implemented+committed; ready_for_mok=false (node-blocked)
m24-implement (wf_6bbc1bff) done: kernel +431/-4 (9 macros default-0, slot 71
fill vector, publish/commit/teff handshake in M24-owned tail, 5 T_eff sites,
ZERO_PAD, NULLWORK), m15_contracts.py G12, m24_fill.py encoding, MoK patcher
m24_mok_patch.py + tests, m24_compile_check.sh, m24_build_gate.sh. Commits
d99fa55a (impl) + e1bcfc4d (fix pass), pushed. Reviews: 7 blocking + 9 major
across 2 lenses — fixed; 1 rebutted w/ evidence (NULLWORK M2-timeout
uniformity, Appendix R3.2, belt-and-braces added anyway).
ready_for_mok=false because NODE-BLOCKED: build gate + both-config hipcc
never ran (sshd resets banner exchange 30+ attempts/70 min; TCP open; port
22 closed; .87 unroutable). Self-contained gate sequence (P0-P4 + arms) in
tasks/wff3mewsd.output .result.final.mok_sequence.
SERVING-SIDE M24 (G13a-G15: m15_runtime forensics, device write_fill_vector,
PF4H_M24_FILL_V1 apply patch, M24_FILL_RECEIPT) is ABSENT — M24 cannot reach
a server tonight. FINAL NIGHT SHAPE: campaign iteration-0 (current M15 vs
native_tuned_tp) is tonight's e2e deliverable; M24 kernel gates run around
it (compile/build-gate CPU-only in parallel; MoK gates ~1-1.5h GPU after
B1); M24 e2e = next iteration. On node return, GPU order:
tgen-retrieve -> deploy+receipt-grep -> B0 -> B1 -> M24 MoK -> B2 (runs past
morning if needed, REPORT.md from completed pairs ~07:30).
REPORT.md drafting starts NOW (node-independent).

Plan: on recon report, decide queue-kill timing (mid-flight arm finishes, then
teardown once first verified fill-aware build is ready — or earlier if the queue
tail is low-value vs getting R6/T-sweep evidence). Wave 2 = implementation
workflow (implement → adversarial verify → fix, compile-only on node while GPUs
busy). Wave 3 = MoK gate (R0a then fill sweep). Wave 4 = e2e campaign vs GENUINE
native baseline, three-way decomposition, cells c32p + c8/c16/c32 (+c512p if
memory), fairness-audit agent before any quoted claim.

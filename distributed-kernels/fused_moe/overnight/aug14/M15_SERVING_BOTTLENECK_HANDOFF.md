# M15 serving bottleneck — handoff packet (2026-08-14, paused ~04:00 UTC)

**Mission for the next agent:** determine what actually bottlenecks M15 vs
production in real vLLM serving under realistic (skewed) expert routing,
define the minimal additional tests needed to pin it down, then optimize the
kernel against that bottleneck. This packet is self-contained; no prior
transcript is needed.

---

## 1. The one-paragraph situation

M15 (`k0pf6gm_m15_mega`, C=28, mode 12, depth-4 throttle) is **25% faster than
production in the MoK kernel harness** at T=4096 (5,822.0 µs = 0.7544×, 13+
green campaigns). Tonight it was integrated into real DeepSeek-R1 vLLM serving
on **real MLPerf text**. The working hypothesis for this packet — supported but
NOT yet proven — is that M15 is tuned for the balanced routing of the MoK
harness and degrades on the slowest rank under real expert-popularity skew. The
instrument that would prove it (per-rank expert histograms + timing) failed
three times for three different reasons and is one small fix away from working.
That is where work paused.

> **Note (2026-08-18):** this packet's original §2.1 banked an end-to-end
> stock-vs-M15 serving pair. It was removed as obsolete: the megakernel's
> activation seal fired on only ~2% of padded-4096 heavy steps (so the candidate
> arm ran production kernels for ~98% of the heavy MoE work), the candidate arm
> ran with no cudagraph on the unsealed steps, and both arms' baselines were
> depressed by a uniform-decode rank running the whole model eagerly. See
> `../../../../docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md` and
> `../aug18-prefill/M23_RAGGED_SEAL_DESIGN.md`; historical content in `git log`.
> The kernel-level and skew-measurement content below is unaffected.

## 2. What is banked and trustworthy (do not re-run)

### 2.1 Serving-pair record — removed, see the note in §1

Cell `c32p` (the rig itself, still the reference cell): 1,024 prompts, ISL
exactly 4096 (chunked prefill), OSL 8, concurrency 32, real MLPerf R1 QSL text
(concatenated — no real sample reaches 4096 alone). DeepSeek-R1-0528,
TP=1 / DP=8 / EP, `--all2all-backend mori_high_throughput`, image
`vllm/vllm-openai-rocm:v0.25.1`, identical prompt SHAs both arms, fresh server
each arm. M15 activation was receipt-proven (7/7 receipt types × 8 ranks,
`M15_SELECTION_RECEIPT` = 464 = 58 routed layers × 8 ranks; `M15_PPERR = 0`) —
but activation receipts only prove the kernel *loaded*, not that the seal let it
*run*, which is exactly the gap M23 closes with `RAGGED_SEAL_RECEIPT`.

Caveat that still stands: `SAME OUTPUTS: False` — expected under a different
accumulation order + greedy decode, but M15's end-to-end accuracy vs stock is
**unverified**. An MLPerf accuracy-checker run on M15 outputs is required
before any "correct and fast" claim.

### 2.2 Workload skew (real vs synthetic text)

From the banked 4,194,304 input tokens: top-10 token types carry 26.4% of the
stream, top-100 carry **65.9%** (synthetic uniform: 0.33%), Zipf slope −1.42.
The workload is genuinely production-shaped. **Token skew ≠ expert skew**
(routers train with load-balancing losses) — this proves the input half only.

### 2.3 Kernel-side context (MoK harness, all gates green)

- T-response (exp_04_tgen, `ablations-tgen` @ 43291977): M15/prod = **0.756**
  @ T=4096, **0.885** @ 2048, **1.094** @ 1024 (inverts; break-even ≈
  T 1600–1800; pf6gm inverts too ⇒ family-wide small-T property).
- C-sweep monotone at both T=4096 and T=2048; C=28 best measured; C=32 has an
  undiagnosed liveness hang; C>28 finer pacing is the open direction.
- **m17 (RR-interleaved scatter) tied M15 at balanced routing and was parked
  explicitly as skew insurance** (`ablations-m17` branch). If serving skew is
  confirmed as the bottleneck, m17 under skew is the first kernel arm to test.
- Mechanism model: C acts as an injector-concurrency throttle on the M7
  epilogue's remote-RMW stream (exp_03 session 2); combine consumed in
  GEMM-2's shadow via coarse (rank,slab) certification.

## 3. The current bug (why we don't have expert histograms yet)

Goal: per-layer, per-rank expert-popularity counts + per-step timing from
inside the serving workers, for both arms. Three failed designs — each fix
was real, each exposed the next layer:

1. **Cross-process observer** (`m15_expert_skew.py --attach`): monkey-patched
   `PF4HPrepareAndFinalize.prepare` in its *own* interpreter via `docker
   exec`. Structurally blind (patches don't cross process boundaries); also
   would observe nothing on the stock arm (PF4H class never loads there).
   Dead approach — do not resurrect.
2. **In-process hook, correct site, wrong filter.** Site (confirmed correct,
   common to both arms): `vllm/model_executor/layers/fused_moe/router/`
   `fused_moe_router.py` — `select_experts()` (called from
   `moe_runner.py:573`), env-gated wrapper accumulating per-layer expert
   counts per process. First version skipped calls when
   `torch.cuda.is_current_stream_capturing()` — but under CUDA graphs the
   router's Python body executes **only during capture**, never at replay ⇒
   recorded zeros. Fixed: count every call, record a `captured`/`eager`
   split. Also fixed: TP=1/DP=8 makes every worker "rank 0", so dump
   filenames now carry PID + dp_rank. **Smoke-proven on a single rank:
   12,288 tokens routed, hot expert 10,758 — the counter works.**
3. **Dump-at-teardown loses the data.** The hook flushed only at worker
   teardown; the container teardown path kills workers before the flush runs.
   Two runs produced empty/no dumps (stock arm twice, m15 arm suppressed by
   the empty-dump guard). **The designed fix (implement first):** flush
   incrementally every ~50 router calls to a per-PID temp file with atomic
   rename, keeping the teardown flush as bonus. At pause this patch may be
   partially applied — check the vLLM agent's last push on
   `vllm-integration-m15` (was through `d313289f`) and diff before trusting.

Dump path: `/results/skew/` in-container → host
`~/20260813_m15_campaign_m15pair1/skew/<arm>/skew/`.

**Pause-time update: the incremental-flush fix WORKS and banked partial
data.** Before the stand-down halt, the patched hook wrote real dumps for the
stock arm — one forward pass per DP worker (8 files, 50 MoE layers, 204,800
routed rows each):

- **Gini 0.3462**, all 256 experts used; top-1 expert 1.23% of tokens
  (uniform = 0.39%) ⇒ **3.14× hot-to-mean**; top-32 experts 26.64% vs
  bottom-32 1.57%.
- **Per-rank routed_rows identical and step_ms p50 identical (18.1 ms)** —
  mean rank load is flat.
- **step_ms p99 spread 193–1066 ms across workers** — the imbalance lives in
  the tail, not the mean.

Read: the workload IS expert-skewed, EP mean load balances out, and the
per-step tail varies wildly. One pass, stock arm only, **a lead, not a
conclusion**. The m15 arm and full-cell coverage are the remaining
measurement.

Final integration-branch state: **pushed through `e346f8b4`**
(`subha-v/amd-master`, `vllm-integration-m15`). Read handoff **§0a** first —
it holds the five-line resume path: hook `m15_router_skew.py` (patches
`FusedMoERouter.select_experts`), mounted via `skewhook/sitecustomize.py`,
`M15_SKEW_OUT=/results/skew`, dumps `skew_rank<N>_pid<PID>.json`; re-run
with `~/m15pkt_skewonly.sh` (~25 min), analyze with `~/skewreport.py`.
Evidence bundle + `SHA256SUMS` under
`benchmarks/2026-08-12_m15_campaign/evidence/`. Caveat: `~/amd-master-m15pkt`
is dirty on `m15_router_skew.py` (content matches the pushed commit) — `git
checkout --detach github/vllm-integration-m15` there before resuming, or
fetch will abort.

Secondary instrument, not yet attempted properly: the vLLM profiler. In
0.25.1 `VLLM_TORCH_PROFILER_DIR` **does not exist**; `/start_profile` mounts
only when the server is launched with `--profiler-config`. The profile pass
needs that flag wired and probed before any curl (the curl-404 killed one
driver run already; passes are now non-fatal to each other).

## 4. Where everything lives

- **Integration branch:** `vllm-integration-m15` on `subha-v/amd-master`
  (push via `github` remote; node `origin` is a dead bundle). Read
  `HANDOFF_VLLM_M15_20260812.md` §0 (ladder state), §5 (ISOLATION transport
  finding + single-region carve), §10 (risk register), §11 (step log).
- **Node (8× MI350X):** `ssh -p 2425 subvadla@10.5.95.87`. Worktree
  `~/amd-master-m15pkt`; container `subvadla_m15pkt` (install intact,
  `M15_INSTALL_RECEIPT.json`); campaign dir
  `~/20260813_m15_campaign_m15pair1/` (pair_01 = the banked pair); datasets
  `~/mlperf_v6_datasets/` (4,388-sample QSL, sha `2c2ccbbe…`); DHK pin
  worktree `~/DHK-m15pkt` (the shared `~/Distributed-HipKittens` has drifted
  — always set `K0_DHK_ROOT`).
- **GPU etiquette:** one 8-GPU job; claim via `~/GPU_CLAIM`; never SIGKILL a
  server; `MORI_GPU_ARCHS=gfx950`; `pkill -f` matches your own ssh wrapper
  (use PIDs); detached work needs `setsid nohup`; the `~/e39` watchdog is
  dormant and only ever targets `subvadla_k0_mok_tgen*` names.
- **Kernel branches (Distributed-HipKittens):** `ablations` (main line),
  `ablations-m15` (RUN PIN), `ablations-m17` (skew-insurance arm),
  `ablations-tgen` (T-response, clamp fix). MoK harness driver:
  amd-master `k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py`;
  **`K0_SYNTH_ROUTE` exists in the harness env — synthetic route control is
  the bridge for replaying measured skew at kernel level.**

## 5. The plan for the next agent

Phase A — finish the measurement (blocked only on the flush fix):
1. Apply/complete the incremental-flush patch; single-rank smoke; then ONE
   paired skew run (both arms, ~12 min each). Deliverable: per-layer per-rank
   expert histograms + captured/eager split for stock and M15.
2. Reduce to the numbers that matter: per-layer max/mean rank load ratio;
   which ranks are hot; expected slowest-rank overhang per step vs the
   balanced case.

Phase B — attribute the tail (pick what the histograms justify):
3. Wire `--profiler-config`, re-probe `/start_profile`, run the profile pass
   on both arms; attribute step time: attention vs MoE region, and inside the
   region dispatch/GEMM/combine vs M7-epilogue/M8 waits. The question is
   *where the extra p99 milliseconds sit* in M15 steps that coincide with
   skewed batches.
4. Candidate mechanisms to confirm/kill: (a) slowest-rank gating amplified by
   M15's coarse (rank,slab) certification — hot rank's slab certifies late,
   all consumers wait; (b) C=28 tuned for balanced load — hot-rank M7 runs
   long while other ranks' pools idle; (c) an activation-boundary artifact
   (exact-B4096 chunks only; check the non-activated fraction is truly zero
   in tail requests); (d) graph-replay interaction unique to serving.

Phase C — close the loop at kernel speed (this is the optimization path):
5. **Replay the measured expert histogram in the MoK harness via
   `K0_SYNTH_ROUTE`** — reproduce the skewed-routing tail behavior at kernel
   level, where iteration is 139 s per campaign instead of 25 min per pair.
6. Optimize against measured skew: m17 RR scatter under skew (already built),
   C re-sweep under skew, and if coarse certification is the culprit, finer
   slab granularity / work-stealing across the consuming pool.

   **The m17-vs-C decision rule (source-verified 2026-08-14, never yet
   measured under skew):** m17 (`ablations-m17` @ b7561d47, node worktree
   `~/DHK-m17`) is the SAME M15 body with one compile arm
   (`K0P6_M15_SCATTER_RR=1`) that permutes the stage-5 scatter iteration
   source-round-robin (`src = q & (world-1)`, `off = q >> 3`) — bijective,
   same cursors/packing/overflow, only the atomic LANDING order changes.
   Donor order clusters each expert's sorted block by source rank (a span is
   hostage to one peer); RR mixes all 8 sources through every span. So m17
   changes ORDERING, not POPULARITY: the 3.14× hot expert keeps 3.14× the
   work either way. Decision rule: if the p99 tail traces to peer-ordering
   serialization inside expert blocks → m17 is the lever; if it traces to
   expert-popularity concentration itself → C-retune/partition is the lever.
   The banked partial data (flat p50 across workers, p99 spread 193–1066 ms
   — intermittent stalls, not steady-state imbalance) leans m17-ward but is
   one stock-arm pass. The cheap decider: the m15-arm histogram + per-rank
   step_ms (~25 min on the fixed rig), then per-expert block occupancy vs
   tail correlation.
7. Revalidate any kernel change through the standard MoK gate ladder, then
   ONE serving pair to confirm transfer. Before quoting any serving number,
   satisfy `../../../../docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md` in
   full — the M23 patch chain in both arms, a `RAGGED_SEAL_RECEIPT` coverage
   quote, rescued-stock as the baseline, ≥5 order-balanced pairs, and the
   MLPerf accuracy check.

Discipline: stock control first; receipts before activation; never widen
tolerances; every claimed number from an artifact, not an inference; commit
and push as you go (no force-push, no co-author lines).

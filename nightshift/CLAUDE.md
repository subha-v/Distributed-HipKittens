# Nightshift: autonomous megakernel optimization loop

You are the overnight orchestrator for the M15 prefill megakernel
(DeepSeek-R1-0528, 8x MI350X gfx950, vLLM 0.25.1 ROCm, MORI all2all + AITER,
TP1/DP8/EP8). Your mission is an **optimization loop**, not queue-babysitting:

> profile → identify the biggest recoverable cost → implement the fix via
> specialized subagents → verify → run the latest kernel end-to-end on the
> node against the GENUINE production baseline at multiple batch sizes and
> concurrencies → repeat until morning.

**Orchestrate Opus 5 subagents** (`model: opus`; effort `medium` for recon,
harvesting, analysis, cleanup; effort `high` for kernel implementation,
patch work, and every adversarial verification). Use workflows shaped
implement → adversarial verify → fix. Nothing you built touches the node
without a separate verifier subagent signing off — skipping this once hung
8 GPUs on a non-unanimous collective; the verifiers had already found the bug.

## Node access — EXCLUSIVE
`ssh -i ~/.ssh/muhammad-gpu -p 2425 subvadla@10.0.0.228`
(IP is network-dependent; earlier it was `10.5.95.87` — try both). Bash
ssh/scp/rsync needs `dangerouslyDisableSandbox: true`.
**We have exclusive access to this node tonight. If anyone else's processes
appear (foreign GPU processes in `rocm-smi --showpids`, foreign containers
starting work), you have permission to terminate them** (`docker rm -f`,
`kill`) — log what you killed in LOG.md.
**pkill trap**: `pkill -f <pattern>` over ssh kills your own remote shell if
the pattern appears in your own command line — bracket a character
(`pkill -f "cam[p]4"`) and never combine the pkill with commands that name
the same file.

## The legacy queue: harvest it, then SHUT IT DOWN
A gated campaign chain was left running as a fallback in case no orchestrator
was set up (`~/eplb_campaign/camp*_20260818.log`; waiters `camp_grid.sh`,
`camp4_rr.sh`, `camp5_native.sh`). It is NOT your plan. On session start:
1. Read all `~/eplb_campaign/camp*_20260818.log` + harvest any completed
   results (order-balanced ratios only, with coverage receipts).
2. When you are ready to use the node — and specifically once your first
   fill-aware kernel build is verified — **stop the queue**:
   `pkill -f "cam[p]_grid"; pkill -f "cam[p]4_rr"; pkill -f "cam[p]5_native"`
   (separate ssh calls, bracket trick), then stop/remove any `m15_*`
   campaign containers still running, `rm -f ~/GPU_CLAIM`.
   Exception: if a campaign arm is mid-measurement when you arrive, let that
   ARM finish (~15 min) and harvest it before killing the chain — free data.
3. The queued RR (camp4) and tri-arm (camp5) experiments are OPTIONAL
   backlog afterwards; run them only if the main loop is blocked.

## Banked state (2026-08-18 ~01:20 PT — verify against logs before relying)
- **Camp3 (M23 pairs) DONE**: order-balanced n=2 → **m15 +2.7–5.2% vs
  patched-stock** at 99% seal coverage (19,277/17,900 vs 15,611/20,589
  tok/s, cell c32p). Position effect ±18%/arm → ARM_COOLDOWN=240 now in
  wrappers v3/v4; never quote single arms.
- **Route corpus DONE**: 8 worker files × 64 calls of raw topk_ids
  ([4096,8] uint8 + fp16 weights) at
  `~/20260818_m15_campaign_routecap1/pair_01/1_m15/skew/routes/`.
- **M23 discovery chain** (why all pre-2026-08-18 serving numbers are void):
  exact-4096 seal → mega ran ~2% of heavy steps; unsealed steps de-graphed;
  uniform-decode ranks whole-model-eager in both arms. Fixed by
  m23_patch.py (+ rescue). Read `distributed-kernels/fused_moe/overnight/
  aug18-prefill/M23_RAGGED_SEAL_DESIGN.md` and `DECOMP_RUNBOOK.md` FIRST.
- Patch chain in every integration container: apply.py → coverage_patch.py →
  m23_patch.py [→ rr_patch.py], all in `~/eplb_campaign/`, anchor-exact,
  fail-loud, idempotent.

## THE BIGGEST RECOVERABLE COST — start here
Serving receipts show the mega executes 4,096 padded rows while carrying only
~1,539 real tokens/rank/step (**37.6% fill, 2.66× padding multiplier**).
Every banked kernel-level number (0.756x balanced, 0.847x skew-replay) is
100%-fill. The mega's fixed costs — single-CTA planner, C=28 reserved
service CTAs, slab rendezvous, M8 certification, and all per-row work on pad
rows — do not shrink with fill, while production's padded GEMMs do partial-
tile work. This plausibly explains most of the real-routing gap.
**Priority 1: fill-aware megakernel** — make cost scale with n_orig:
early-exit/skip empty tiles, planner + dispatch + service work proportional
to real rows, pad-row handling proven parity-safe (production pads too — the
parity analysis discipline of M23 §3 applies). Design doc → adversarial
review → implement behind default-0 macros → MoK gate (see traps below) →
then take the node for e2e. R6 of DECOMP_RUNBOOK.md (T-sweep 4096/2048/1024,
no capture needed) is the cheap confirmation experiment if you want evidence
before implementing.

## Kernel-level testing traps (from DECOMP_RUNBOOK.md — follow it exactly)
- **Pin trap**: the MoK harness compiles `k0pf6gm_device_tile_mps.hip` from
  DHK_ROOT; on `ablations` that file is the OLD sibling. The serving M15
  body runs only via run pins (e.g. `~/DHK-m15`) with the body inlined
  VERBATIM (the mori JIT hash ignores -I paths). Every arm needs an explicit
  DHK_ROOT pin; R0a must reproduce 0.7556x ±1% (C=28,g=353,mode=12,
  flush_rows=16) or STOP and diagnose.
- Fabric arms (K0P6_M15_POLL_BACKOFF / RMW_INTERLEAVE; worktree
  `.claude/worktrees/wf_b38dfe27-d09-2`, build-gate passed) need generated
  pins (runbook `mkpin`); the harness cannot consume prebuilt hsacos.
- Phase ledger is `[MPS TS]`, rank-0-only, final-soak-epoch semantics;
  `K0_MOK_ROUTE_MAX_CALLS=1` for ledger runs. `[MPS SPIN] fail_max>0` under
  captured routes = direct transport-degradation evidence.

## The GENUINE production baseline — non-negotiable
The end-to-end headline is **latest-kernel vs NATIVE production**: the
untouched `vllm/vllm-openai-rocm:v0.25.1` image, shipped defaults, no PF4H
env, no apply.py, no patches (wrapper v4's `native` arm implements this).
Patched-stock (production ops inside our integration) is only a secondary
diagnostic ratio — it isolates the kernel substitution but inherits our
integration's changes (B4096 stock graph, rescue, runtime footprint). Always
report the three-way decomposition when possible:
`m15/native` (headline) · `m15/patched-stock` (kernel effect) ·
`patched-stock/native` (integration effect). Every quoted number carries the
full workload spec and, for mega arms, its coverage receipt
(`RAGGED_SEAL_RECEIPT sealed/in_bucket`).

## End-to-end sweep matrix (each iteration of the loop)
Serving cells (harness `cell_params`): the prefill-headline **c32p**
(conc 32, 1024 prompts, ISL 4096, OSL 8) plus, for batch/concurrency
coverage, **c8 / c16 / c32** (ISL 1024, OSL 512) and **c512p** (conc 512,
ISL 4096 — defined but never yet run; watch memory). Run order-balanced
pairs (arm order reversed on even pairs), ARM_COOLDOWN active, fresh server
per arm, exact-token SHA. Remember the mega inverts below ~1,600-1,800
tokens/rank — small-batch cells are expected losses for the mega and wins
for the hybrid story; report them honestly, never hide them.

## Fairness audit — MANDATORY before any "beats production" claim
This project spent weeks optimizing over fake baselines. Never again. Launch
a dedicated **fairness-audit Opus subagent** (`model: opus`, effort `high`),
adversarially prompted to PROVE THE COMPARISON UNFAIR, and re-run it for
every new kernel build and every campaign whose numbers you intend to quote.
No headline claim ("our kernel is better than production") ships without its
written sign-off; it holds veto power. Its checklist, grounded in the actual
past failures:
1. **Baseline authenticity**: the production arm is the genuinely untouched
   image at shipped defaults — verify from evidence, not intent: image
   digest, `docker inspect` env (no `VLLM_PF4H_*`), server.log config dump,
   absence of every patch marker (`PF4H_INTEGRATION_PATCH_V3_M15`,
   `PF4H_COVERAGE_PATCH_V1`, `PF4H_M23_RAGGED_SEAL_V1`, `PF4H_RR_*`).
   Past sin: quoting patched-stock as "production".
2. **Baseline not sandbagged**: production gets its best shipped config —
   AITER + MORI env present, same gpu-mem-util/scheduler flags as our arm;
   any deviation from the vendor's recommended deployment documented and
   justified. Past sin: none yet — keep it that way.
3. **Candidate actually ran**: coverage receipts (`RAGGED_SEAL_RECEIPT
   sealed/in_bucket`) prove the megakernel executed the traffic. Past sin:
   the mega was inert on ~98% of heavy steps for every pre-M23 A/B while the
   candidate arm additionally ran de-graphed — a doubly fake candidate.
4. **Identical workload**: exact-token prompt SHA match across arms, same
   cell spec, seeds, client, warmup; any prefix-cache or thermal asymmetry
   neutralized (cooldowns symmetric). Past sin: prewarm/cache asymmetry
   risks; position effect ±18%/arm.
5. **Statistical validity**: order-balanced pairs only, n stated, spread
   across pairs shown, claim sized against the measured drift (±15% day,
   ±18% position). Single arms and cross-pair ratios are never evidence.
   Past sin: n=1 headlines (+39.8%, −19.6%) later voided.
6. **Accuracy parity**: outputs validated (exact-token SHA where applicable;
   otherwise the MoK rel-err policy or an accuracy A/B) — a faster wrong
   kernel is not a win. Past sin: `SAME OUTPUTS: False` left unresolved.
7. **Replay/proxy honesty**: kernel-level rigs (MoK, histogram replay) are
   never quoted as end-to-end; captured-route and fill regimes labeled
   (banked kernel numbers are 100%-fill; serving runs at ~37.6% fill).
   Past sin: i.i.d. replay standing in for real routing.
8. **Claim wording matches measurement**: which cells, which regime, the
   hybrid caveat (mega inverts below ~1,600–1,800 tokens/rank — small-batch
   cells reported, not hidden), memory/replication costs priced in (past
   sin: 41 GB replica caches framed as kernel wins).
REPORT.md must contain a `FAIRNESS_AUDIT` section per quoted claim: verdict,
items checked with evidence pointers, residual caveats.

## Also orchestrate
- **The cleanup agent** (died mid-run at laptop sleep; partial work at
  checkpoint f4968a32 + possibly uncommitted edits): relaunch an Opus
  subagent to DELETE obsolete pre-M23 serving numbers in BOTH repos
  (`/Users/subha/repos/Distributed-HipKittens`,
  `/Users/subha/repos/amd-master`), tombstone wholly-obsolete docs, finish
  `docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md`, commit+push.
  Kernel-level MoK results stay untouched. Runs in parallel with the main
  loop; reconcile, don't duplicate.
- Optional backlog when the main loop stalls: finish BOTTLENECK_SIGNALS.md
  (forensics agent died mid-run), route-corpus run-correlation analysis,
  the RR (`rr_placement/`) and EPLB0 (`m15_eplb0/`) staged composes,
  fp8-wire (`FP8_WIRE_DESIGN.md`, P0 codegen fix first) and shared-expert
  filler (`SHARED_EXPERT_FILLER_DESIGN.md`, gate G1 first) designs.

## Hard rules
- Verify-before-deploy (adversarial subagent) for everything node-bound;
  chain-test patches against the scratchpad mirror pattern
  (`aug18-prefill/m23/test_m23_patch.py`) first.
- Kernel changes behind default-0 macros; default build must be
  instruction-identical (build-gate: resource tuple + disassembly compare —
  see `fabric_build/` precedent on the node).
- Never edit a shell script a running bash has open; deploy new files and
  relaunch waiters instead.
- One B4096 graph per server (memory headroom).
- Git: commit+push `origin ablations` after each meaningful landing, no
  co-author lines, user's account only. Append timestamped progress to
  `nightshift/LOG.md`; write `nightshift/REPORT.md` before morning; keep
  auto-memory updated so a session restart can resume cold.
- GPUs busy = compile-only on the node (hipcc is CPU-only work).

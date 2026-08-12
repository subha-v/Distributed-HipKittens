# CLAUDE.md — aug11 overnight: PAPER EVIDENCE for the fused-MoE megakernel

> **THIS IS YOUR CHARTER. There is a second one in this repo and it is NOT
> yours.** `distributed-kernels/gemm_rs/overnight/aug11/CLAUDE.md` belongs to a
> different agent working the GEMM-RS kernel on 8× MI300X
> (`banff-sc-cs47-05`, gfx942, branch `GEMM-RS`). You must not edit, benchmark,
> or "help with" anything under `distributed-kernels/gemm_rs/**`.
>
> You are fused-MoE: 8× **MI350X** (`gbt350-odcdh2-c05-1`, gfx950, container
> `subha_k1`), branch **`codex/distributed-hipkittens-scaffold`**.

You are a **kernel genius and an orchestrator**. Spawn focused subagents for
read-heavy and write-heavy work and demand concise conclusions plus durable
on-disk artifacts — never burn your own context on whole kernels,
disassemblies, or long logs. You own decomposition, hypotheses, contradiction
resolution, experiment selection, the node lease, and final judgment.
**There is no termination condition: when one experiment lands, start the
next. Do not stop working. Do not end the night early because something is
blocked — fall back to the next queue item and keep going.**

## What tonight is FOR — the central claim, and the figures that prove it

The paper (`docs/distributed/PAPER.md` — read it first) claims: on AMD GPUs,
communication/computation overlap in a megakernel is decided by *scheduling*
decisions — who carries the payload (the producer's epilogue), how many remote
operations are in flight (the throttle), which order producer tasks run, and
how coarse the completion signals are — and NOT by dedicating CTAs to
communication. Tonight's job is to produce the measured figures that prove it.
**A ratchet win is welcome but secondary tonight; the night succeeds if the
figure data lands.** Every experiment below names the paper figure it feeds.

Current state (see `STATUS.md` in this folder): ratchet `mps_mega`
`C=16, g=353, mode=12, flush_rows=16` = **6,568.0 µs = 0.8522× production**;
arms `production` / `pf6gm_mega` / `mps_mega`.

## Mission queue (in order; interleave CPU builds under running GPU campaigns)

0. **Land the exp_32 NaN-poison patch first.** It is written and verified
   (`exp_32_gate_hardening/`), it is a blocking prerequisite for every
   staleness-shaped mechanism tonight (mode 14 especially), and it makes every
   later gate real. `K0_MOK_POISON_OUT` defaults on.
1. **exp_33 — bottleneck attribution at the ratchet (paper Q3, Simran's
   "what step is bottlenecking speed right now").** One timestamps-on
   attribution campaign at `g=353`: per-phase stacked bar (M0–M2 / M3–M5 / M6 /
   M7 split into GEMM vs epilogue surcharge / M8M9), rank-max and rank-0, plus
   the same stamps on `pf6gm_mega` and `production` for the side-by-side.
   Deliverable: `exp_33_attribution/phase_stamps.json` + a result.md table.
   Cheap, do it before anything mutates the tree.
2. **exp_22 — NanoFlow-Fig-7 analog (paper Fig 2 / Q4).** The plan and design
   are already written in `exp_22_fig7_saturation/`. Build the three-mode
   ubench + the concurrent role-split overlay exactly as specified; sweep
   grids are in the plan. Deliverables: `saturation.json` (every point:
   mode, CTAs, MLP, fanout, isolated/concurrent, GB/s or TFLOPS) + the knee
   summary in result.md. Pre-registered hypotheses and the falsifier are in
   the plan — judge against them explicitly.
3. **exp_23 — NanoFlow-v2-Fig-10 analog (paper Fig 3 / Q4).** Plan and design
   in `exp_23_fig10_timeline/`. Per-CTA phase event ring behind the existing
   diagnostics flag; resource-tuple parity gate with the flag compiled in but
   OFF; three arms (production trace via torch-profiler, `pf6gm_mega`,
   `mps_mega` ratchet) on the same routing seed. Deliverables:
   `rank0_events.json` per arm + `timeline_bins.csv` + the two validation
   checks (integral, amd-smi cross-check) recorded in result.md.
4. **exp_34 — mode 14 build (coarse readiness) — the missing waterfall rung.**
   Build exp_30's design (aug11 `STATUS.md` §exp_30): ride mode 12's
   remote-accumulate transport, delete the per-row protocol (`nc_arr`,
   `pushed`, `row_ready` per-row, M8 polls), replace with one grid barrier +
   8 per-source arrival publishes. The NaN poison must be live BEFORE this is
   timed (its failure mode is exactly "row not written"). Full gate ladder,
   then a decision campaign. Pre-registered band: 5,990–6,440 µs; a result
   above 6,568 falsifies the granularity rung and gets reported as such.
   **The degeneration is a finding**: with the protocol gone the service pool
   has no job — bank the number and say so in result.md.
5. **exp_35 — the knob waterfall (paper Fig 4 / Q1 — the money figure).**
   One campaign per rung, same seeds, full ladder each:
   (a) `pf6gm_mega` (homogeneous baseline);
   (b) mode 12 unthrottled (epilogue-carried payload alone);
   (c) mode 12 + depth 4 (add the injection bound; this is the ratchet);
   (d) mode 14 (add coarse signals), if exp_34 gated green;
   (e) + nc-major task order if the reorder is buildable tonight (exp_30's
   companion; if not, log it as the named missing rung).
   Deliverable: `waterfall.json` — rung, arm_p50_us per arm, ratio vs
   production, delta vs previous rung.
6. **exp_36 — sensitivity sweep (paper Fig 7 / Q5, Simran's batch/seqlen/
   imbalance question).** Screens first, campaigns at the corners:
   T ∈ {512, 1024, 2048, 4096} × routing std ∈ {0, 0.032, 0.05}
   (`K0_SYNTH_ROUTE`, `skewed_hot` family; 0.032 = COMET's production skew).
   At every point record `[MPS SPIN]` (the peer-wait instrument) and the best
   config. Pre-registered: small T favors coarse signals/fusion; large T
   favors throttle+order; skew is the ONE regime where a service pool may
   re-enter — at std=0.05 also run C ∈ {0, 8, 16} arms and report whether
   placement ever wins. Deliverable: `sensitivity_grid.json`.
7. **exp_37 — placement adjudication rerun (paper Fig 5 / Q2).** Paired
   same-run: mode 2 best (C=64 g=1) vs mode 12 best (C=16) vs mode 14
   (C ∈ {0, 8, 16} — prediction: flat). Plus F1 (`T6(128)/T6(256)`, the
   phase-1 task macros are live, ~30 min).

Stretch (only if the queue above is green): the Megatron/PyTorch+RCCL eager
arm (paper Q6) — an eager dispatch→GEMM→combine reference through the same
harness shapes, giving the literature-comparable denominator.

## Phase 2 — when the figure queue is done, the optimization loop RESUMES

Getting every plot is not the end of the night; it is the checkpoint where
the standing objective takes back over: **widen the margin over `production`
toward 0.80× (6,172 µs) and beyond**, with the full aug10 ratchet discipline.
The figure data tells you where to strike — exp_33's attribution is the
profile of the current winner, and every figure experiment doubles as a
profiling pass. The live optimization queue, in expected-value order (all
pre-analyzed in `STATUS.md` — read the relevant section before building):

1. **exp_27 — delete M5's scale transpose, point M6 at `sc_stage`**
   (predicted −130…−190 µs, three models agree; the gate is BIT-EXACT
   output, stronger than any tolerance; composes with exp_24's deletion so
   the whole `zero_part_scale_transpose` loop dies).
2. **exp_26 mask ladder on GPU** (mask 0/4/1/5; mask 4 is free at
   donor-identical resources; judge on the M6 phase stamp, 3–9σ expected;
   predicted 75–200 µs).
3. **exp_31 — phase-2 VMEM hint fix** (`14 + kGM`, one line, same
   parameterization class as phase 1's; M7 is 2,660 µs).
4. **nc-major task reorder** if it did not land in the waterfall (lifts the
   combine's unblockable fraction 17%→85%; it is also paper Fig 6 data).
5. **mode 14 C/flush tuning** at the exp_34 winner; then re-run exp_33
   attribution on the new ratchet and pick the next largest term.
6. If all of the above land: M6's K-loop is still ~84% stalled — spawn a
   research+implementer pair on the software-pipeline depth (the one-K-step
   pipeline against a ~9,800-cycle iteration is the biggest single pool of
   cycles left in the kernel).

Loop rule: after every landed optimization, update the ratchet, append to
LESSONS.md, commit+push, re-run the phase stamps, and pick the next target
from the NEW profile — never from tonight's stale one. If a build blocks,
fall back to the next item; if the GPU is busy with a campaign, do CPU-side
builds/ISA work for the next item in parallel. **The night has no done
state.**

## Deliverable discipline — this is a figure-producing night

- Every experiment folder gets `plan.md` (if not already present), `result.md`
  (verdict + numbers + confidence), and **plot-ready data** (`.json`/`.csv`,
  schema stated in result.md). No number lives only in prose.
- Maintain `aug11/PLOTS.md`: one row per paper figure — figure #, data file,
  generating experiment, status. The morning read starts there.
- Update `aug11/STATUS.md` after every landed experiment; append every verdict
  (including negatives and kills) to `aug11/LESSONS.md` (append-only;
  supersede, never delete).
- Commit + push after EVERY completed or failed experiment. Verify
  `git ls-remote` head == local HEAD before starting the next one. The other
  agent pushes to a different branch; you still pull --rebase before push.

## Standing rules (unchanged from aug10 — they are load-bearing)

- Correctness + negative control + 600-epoch soak BEFORE timing, every
  candidate, no exceptions. `pperr != 0` is terminal — never clear-and-retry.
- One variable per arm. Same-run paired denominators only (`summarize.py`
  rank-max medians). Sub-5% deltas re-run or pair against the previous best.
- **Screens resolve ~6.6% end-to-end; phase stamps resolve ~1%.** Screen
  phase-local mechanisms on their phase stamp; end-to-end numbers come from
  5-rotation campaigns only. Screens rank against `production` only.
- Bump `K0P6_MPS_SRC_REV` with any `.cuh`-adjacent edit and confirm a fresh
  `.hsaco` mtime (`stat -L`; directory mtimes lie on cache hits).
- The node checkout IS the arm: `git fetch && git reset --hard origin/<branch>`
  on the node before every run.
- Never change shapes, iteration counts, tolerances, or warmup to make a
  number look better. `K0_MPS_SOAK_ITERS` must stay 600.
- Facts not to re-litigate (measured): dispatch has zero peer wait at std=0
  (exp_10); interference is protocol not payload (exp_20); throttle >8 is a
  cliff (exp_24); coalescing atomics is a trap (exp_07); the M6/M7 interleave
  is capped at +211 µs (exp_25); A-stream bytes cost 4.9× B-stream bytes
  (exp_28).

## Environment (see aug10/CLAUDE.md for the full trap list — it all applies)

- Harness: `~/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/`
  `mok_synthetic_prefill/run_campaign.sh`; arms
  `production,pf6gm_mega,mps_mega`; campaign = 5 processes, 500 warmup /
  100 timed; smoke = 1/1/1. `K0_MPS_CFG` needs all four of
  `C,g,mode,flush_rows` — a partial config LOOKS like a pass.
- Compile container `subha_k1` (CPU work any time); campaigns build in their
  own containers from the read-only bind-mount of this repo.
- Resource gates: current MPS tuple `SGPR 104 / VGPR 256 / AGPR 256 /
  scratch 128 B / LDS 155,428 / MFMA 180 / pk_add_bf16 282`; capture
  `-Rpass-analysis=kernel-resource-usage` on every build; zero scratch ops
  inside either MFMA span.

## Subagents (spawn by name; write OWNERSHIP into every dispatch)

**implementer** (one mechanism at a time, exact files owned) · **profiler**
(stamps/ISA/counters, never concurrent with another GPU job) · **research**
(literature + donor-tree, mechanism-level reasoning) · **protocol-review**
(read-only ordering/epoch review before any new protocol's first GPU run —
mode 14 requires this signoff) · **benchmark-review** (timer fairness and
statistics before any number is reported upward). One deliverable and one
validation command per dispatch; never two writers on one file; never two GPU
jobs at once.

## Node discipline (non-negotiable)

One 8-GPU job of ours at a time (`rocm-smi --showpids` is the check that
works). `setsid` + `timeout` on every long run. Never SIGKILL GPU processes.
We hold the exclusive lease: reclaim the GPUs from foreign processes
(SIGTERM/graceful, log the preemption), and never pause the loop because
someone else started a job.

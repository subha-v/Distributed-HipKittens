# HipKittens

HipKittens is a repository in the ThunderKittens cinematic universe! This work provides minimal, opinionated C++ embedded programming primitives to help you write speedy AMD AI kernels. HipKittens is built from the hardware up: we do what the silicon tells us. 

<div align="center" >
    <img src="assets/hipkittens.png" height=250 alt="HipKittens logo" style="margin-bottom:px"/> 
      <p><em>HipKittens surfing the ~wave~ (not warp).</em></p>
</div>

<br>

**Links**: [Paper (arXiv)](https://arxiv.org/abs/2511.08083) | [Blog: Fast and Furious AMD Kernels](https://hazyresearch.stanford.edu/blog/2025-11-09-hk) | [Blog: AMD GPUs go brrr](https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr)

**AI has largely used a single hardware vendor in the past, but how can we enable a *multi-silicon* future?** Towards the dream of a single software framework that translates across hardware platforms, we explore whether the primitives used in prior DSLs (like TK) suffice for AMD, or whether we need entirely new primitives.

We find that core tile and bulk compute interfaces carry over from TK to HK, but decisions around memory access patterns, scheduling compute and memory, and ordering thread blocks within the chiplet architecture differ. HipKittens features the following types of primitives. 
1. **Tile primitives**: sized according to the tensor core units. Tile memory ops are coalesced, bank conflict free, and eagerly use tensor core layouts. We focus on minimizing address computation costs. 
2. **Python-inspired functions**: bulk compute functions that operate over tiles. These are lightweight, wrapping assembly and HIP.
3. **Asynchronous loads/stores**: hide latencies and address generation using direct buffer loads to shared memory.
4. **Scheduling and overlapping**: we show two core patterns for overlapping compute and memory, 8-wave ping pong and 4-wave interleave, that appear across kernels.

We support CDNA3 and CDNA 4. 

## Distributed HipKittens

This fork is incubating a device-side multi-GPU layer built around parallel
global layouts (`pgl`) and explicit peer-memory, publication, acquire,
replay-lifetime, and role-partition primitives. Kernels remain ordinary HIP
C++: HipKittens does not hide routing, scheduling, dependency keys, or progress
behind a collective or graph API. IRIS provides the fine-grained
symmetric-memory runtime and peer mappings.

Start with [the distributed architecture](docs/distributed/ARCHITECTURE.md),
[source audit](docs/distributed/SOURCE_AUDIT.md), and
[validation matrix](docs/distributed/VALIDATION.md).
The `distributed-kernels/` tree contains the IRIS integration and the GEMM to
ReduceScatter and fused-MoE porting work. Donor measurements and new
abstraction rewrites are kept separate; a port does not inherit a donor's
performance result until its architecture-specific GPU parity gates pass.

The post-draft8 campaign (2026-08-18) lives under
`distributed-kernels/fused_moe/overnight/aug18-ablations/`. The active plan is
**`UNDERSTANDING_PLAN.md`** (r2, twice adversarially reviewed): Track 1 —
**beat production end-to-end at low concurrency** (C=32 TP8+EP; gate ladder
G-L0..G-L4: ρ-swept device phase ledger with directly-measured transport β and
a profiler-confirmed exposed-AR prize, RCCL-hybrid floor, shared-expert filler,
metered AR, then TP8 serving integration) — and Track 2, the **anatomy
campaign**: twelve mechanism-attribution questions (Q1–Q12) over the megakernel
ladder we already own (7,712 → 5,822 µs plus the instructive failures),
producing a MoK-style `MEGAKERNEL_ANATOMY.md` whose abstractions are
**hardware-agnostic data-movement contracts** (Iris/IrisX-aligned; the DHK
headers are one backend). ≈15–22 node-hours total. The earlier
workload-generalization design (`ABLATION_METHODOLOGY.md`, scope-superseded but
kept as the extension plan) and its full evidence chain — grounding briefs,
six design docs, and five adversarial reviews (`review/`) — remain in the same
directory, with `BRIEF.md` carrying the expert feedback that started it.

Current fused-MoE overlap research lives under
`distributed-kernels/fused_moe/overnight/`: the aug11 evidence corpus and
methodology study, and the aug12 design docs —
`OVERLAP_KERNEL_DESIGN_IDEAS.md` (the K0–K8 build queue) plus
`OVERLAP_KERNEL_DESIGN_ADDENDUM.md` (prior-corpus re-grades of the SDMA and
task-order arms, and four new arms: slab-certified combine, staged CU
wide-push combine, source-interleaved scatter, backward plan-reuse, and the
compact-MAXTOK small-T path).

The first new kernel from that program is
`distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip`
(`k0pf6gm_m15_mega`): the mode-12 depth-4 ratchet restructured into an
nc-major, two-slab GEMM-2 with slab-certified coarse readiness, a mid-epoch
front-half combine running in GEMM-2's shadow, and a compile-gated staged
wide-push transport arm. Contract and gate ladder in
`overnight/aug12/M15_DESIGN.md`. **Measured 2026-08-13 on 8× MI350X
(`overnight/aug12/exp_03_m15_slab_combine/result.md`): 6,292.4 µs p50 =
0.8165× the AITER+MORI production arm — −191.4 µs below the prior mode-12
ratchet in the same session, all correctness/poison/soak gates green.** The
same session adjudicated the mode-16 TBO-2 deferred-combine arm at +75.8 µs
(falsified; `overnight/aug12/exp_02_tbo_deferred_combine/result.md`).

The aug13–14 session extended this three ways. (1) **Batch-size response**
(`ablations-tgen` branch, `overnight/aug13/exp_04_tgen/result.md`): a
one-line M2 clamp fixed the T≠4096 correctness defect, and 5-rotation
campaigns measured M15 C=28 at 0.756× (T=4096), 0.885× (T=2048), and 1.094×
(T=1024) of production — the megakernel family inverts at small batch,
break-even ≈ 1,600–1,800 tokens/rank. (2) **Real serving** (amd-master
`vllm-integration-m15` branch): M15 was integrated and ran attested inside
DeepSeek-R1 vLLM serving (58 layers × 8 ranks, receipt-gated) on real MLPerf
text. The throughput A/B taken there is obsolete — see **Serving benchmark
methodology** below. The paused bottleneck study and its resume plan live in
`overnight/aug14/M15_SERVING_BOTTLENECK_HANDOFF.md`. (3) **GEMM-RS**
(`GEMM-RS` branch): the evaluator's debug-flag contamination was removed
(ours beats the GEMM+RCCL reference at 0.764×), and competition-mode
iteration against the leaderboard rank-1 kernel began (gap 1.138× geomean,
first schedule win kept at −0.88%).

The aug14 session closed the serving-bottleneck question and produced
**M18** (`K0P6_M15_REPLICATE`, RUN PIN `ablations-m18`), the static
hot-expert replication arm. The completed skew measurement showed real
MLPerf routing concentrates 51.9% of traffic on experts 0–7 (all one rank;
receive-side load 5.09× fair share aggregate, 5.59× worst layer), a new
`K0_MOK_ROUTE_HIST` harness mode replays measured histograms at kernel
speed, and under that replay M15 degrades to 0.847× production (m17's RR
order buys only ~1%). M18 replicates the hot experts onto every rank and
routes them source-locally (exactly-once via self-source acceptance; combine
untouched): **0.309× production with 8 replicas, 0.242× with 32, 0.217× on
the worst layer — and 0.785× in the balanced control** (all gates green,
5-run campaigns). Full results and fairness caveats in
`overnight/aug14/M18_REPLICATION_RESULTS.md`.

The per-call routing diagnostic then drove the design forward: per-chunk
replica-set coverage is bimodal (p5=1%, p75+=77%) and per-call max rank load
hits 6.25x at p95, so a static aggregate set pays its carry cost on exactly
the chunks it cannot help.  Two fixes followed — per-layer top-16 sets (58
distinct), and **M19** (`K0P6_M15_ADAPTIVE`, `ablations-m19`): per-layer
replica slots plus an in-kernel per-chunk decision (M0.5 own-routing
histogram, theta threshold, sender-published decision bitmaps with mirrored
acceptance), harness reference-match gates green.  Design:
`overnight/aug14/M19_DESIGN.md`.  (The end-to-end serving pairs originally
banked for these arms were retired on 2026-08-18 — see **Serving benchmark
methodology** below.)

The evening session closed the arc with **M20** (`K0P6_M20_SLOTPOOL`,
`ablations-m20`): M19's routing machinery with the 41 GB of per-layer
replicas replaced by a budgeted persistent replica cache and the in-kernel
decision pre-pass replaced by a pre-launch bitmap (deleting the measured
1.2 ms barrier tax).  The road there produced a three-point overlap
measurement on xGMI — pull-based weight streaming +93 ms naive / +23 ms
unrolled (remote-read latency-bound), push-based +54 ms (the
~1.4 GB/s/CTA service-pool law) — proving per-step MoonEP-style weight
movement infeasible on this fabric and motivating the cache.  Final
kernel-level numbers (5-run campaigns, gates green): **0.2483x production
under measured serving skew (6,035 vs 24,283 us) and 0.823x balanced at
theta=64 (~0.76 with theta above uniform), at 672 MB versus M19's 41 GB.**
Serving integration handoff:
`overnight/aug14/M20_SERVING_INTEGRATION_HANDOFF.md`.

### Serving benchmark methodology (2026-08-18) — all earlier serving numbers retired

Per-step instrumentation added on 2026-08-18 showed that **every end-to-end
vLLM serving A/B of the fused megakernel taken before that date is invalid**,
in two compounding ways. (1) *Inert candidate*: the megakernel's activation
seal required all 8 DP ranks at exactly 4096 pre-padding tokens
simultaneously; on the real c32p workload that held on only **~2% of the ~230
padded-4096 heavy steps per rank** (a single 1-token decode batch on any rank
disqualified the step for everyone), so the candidate arms ran production
kernels for ~98% of the heavy MoE work. (2) *De-graphed candidate + broken
baseline*: on unsealed heavy steps the candidate server fell back to **no
cudagraph at all** (it never registers the stock B4096 graph key), and in
**both** arms any rank whose local batch looked like a uniform decode ran the
whole model eagerly at 4096 padded tokens and frequently cancelled DP padding
group-wide.

**M23 "ragged seal" + "uniform-decode rescue"** (implemented and validated
2026-08-18; zero HIP change) makes the seal accept the same padded-4096
batches production already runs — measured coverage **99% of in-bucket steps,
~95% of tokens** — and routes uniform-decode ranks to the graph in both arms.
Fixing the baseline roughly **doubled it**. Two new reference points, same
cell, **n=1 each**:

| arm | input tok/s |
|---|---:|
| rescued stock (production-configured, M23 chain) | **20,918** |
| m15 + M23 (first honest megakernel number) | **19,277** |

Spec for both: c32p cell, concurrency 32, 1,024 MLPerf QSL prompts, ISL 4096,
OSL 8, aggregate node input throughput, DeepSeek-R1-0528 TP=1/DP=8/EP on
8× MI350X. These are **reference points, not a delta** — they are n=1 and
cross-pair, against ±15% stock run-to-run drift.

Consequently the obsolete serving milestones (the PF4H c32/c512/c1024 pairs;
m15, m18, m19, m20 pairs; the balanced-prompt "flat" pair) have been removed
from this repo's docs rather than restated; `git log` preserves them.
**Kernel-level MoK numbers are unaffected** — 0.756x balanced, 0.8467x
skew-replay, the M18/M19/M20 replication ratios above, and all training
numbers never went through the seal.

**First valid order-balanced result (camp3, 2026-08-18).** Two order-balanced
pairs on the same c32p cell (arm order reversed on the even pair,
`ARM_COOLDOWN=240`, fresh server per arm, 99% seal coverage) give
m15 **19,277 / 17,900** tok/s against patched-stock **15,611 / 20,589** tok/s —
**m15 +2.7% to +5.2% within-pair**. That ratio is the *kernel-effect* leg only;
it is **not** a claim against genuine production, and the raw arm numbers show
the ±18% position effect that makes single arms worthless as evidence.

New serving claims must satisfy
[the serving benchmark methodology](docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md),
revised 2026-08-18 (rev 2) to add: the **genuine-native** baseline definition
(untouched `vllm/vllm-openai-rocm:v0.25.1`, shipped defaults, no `VLLM_PF4H_*`
env, no patch markers) as the only baseline a "beats production" headline may
use, with patched-stock demoted to a diagnostic control; the **three-way
decomposition** (`m15/native` headline · `m15/patched-stock` kernel effect ·
`patched-stock/native` integration effect); order-balanced pairs with
`ARM_COOLDOWN=240`, a fresh server per arm, and the measured drift bounds
(±15% day, ±18% position — single arms and cross-pair ratios are never
evidence); `RAGGED_SEAL_RECEIPT sealed/in_bucket` coverage as a validity
*requirement* for every megakernel number; exact-token SHA identity; a full
workload spec; the residual dummy-rank asymmetry; and an **8-point adversarial
fairness audit** that must sign off, with veto power, on any claim of beating
production. Design and implementation:
`overnight/aug18-prefill/M23_RAGGED_SEAL_DESIGN.md` and
`overnight/aug18-prefill/m23/M23_IMPL_NOTES.md`.

The retirement pass was completed on 2026-08-18 across both repos (this one and
`amd-master`): remaining pre-M23 serving deltas were deleted in place, docs
whose entire subject was a void campaign were tombstoned in the lead, and the
methodology document above was finished. Raw JSON/artifacts, receipts and
`SHA256SUMS` were left untouched — they remain valid records of what the machine
did; only their interpretation as an arm comparison is void.

Latest additions to the device primitive layer and the fused-MoE port
(`distributed-kernels/fused_moe/`):

- `include/cdna4/ops/group/distributed/roles.cuh` — minimum-progress role
  specialization primitives: a finish-order compute/service partition
  (`finish_order_partition`), tile-key release/acquire
  (`publish_tile_release` / `wait_tile_acquire_into`), and epoch retirement.
  The model: reserve the *minimum* CTAs needed to guarantee communication
  progress at a phase boundary, never resize inside the phase, and let drained
  compute CTAs flow into service/reduction queues.
- `k0pf6gm_device_tile_mps.hip` (+ `moe_mps_adapter.cuh`,
  `n2_phase2_gm_mps.cpp`, host ABI slots 56–62) — a COMET/MoK-style additive
  sibling of the fused-MoE parity port, and currently the fastest measured
  fused-MoE megakernel in this repository. On world-8 MI350X (gfx950) with
  the MoK synthetic-prefill campaign (two independent 5-rotation campaigns):
  **6,683–6,686 µs = 0.866× `production`, 0.968× the homogeneous megakernel
  `pf6gm_mega`** (exp_21 mode 12, `C=16 g=33 mode=12 flush_rows=16`; all gate
  ladders green incl. 600-epoch soaks). The winning mechanisms on top of the
  MoS stream base:
  - **mode 2 stream** (the prior ratchet): per-`(b, nc)` tile events off the
    GEMM wave, service-wave row bookkeeping, 16-byte slice pushes into
    owner-resident slots, batched owner-only readiness, dynamic-ticket
    combine — 6,866 µs (0.888× production).
  - **mode 12 (exp_21, current best)** — *direct remote bf16 accumulation*:
    the M7 epilogue accumulates each output tile into the owner's slot with
    remote packed-bf16 atomics over xGMI, deleting the local `part` stage and
    the pool's payload copy (936 MB → 312 MB inside the M7 window); the pool
    shrinks to readiness bookkeeping (`C` 64 → 16) and the owner
    consume-and-zero restores the slot invariant. The epilogue's outstanding
    remote RMWs are capped at 8 per thread (`vmcnt(8)`) — the measured cure
    for the fabric stream's rate-shaped interference with the co-resident
    GEMM. Also modes 9/13 (fence-scope diagnostic + consolidated counters),
    the exp_20 diagnostics modes 4–8 (pacing / traffic attributions /
    payload-free stream / poll backoff) and a g-bit dual-write detector that
    certified zero lost remote updates over 600 epochs. Evidence:
    `distributed-kernels/fused_moe/overnight/experiments/exp_21_direct_accumulate/result.md`
    and `overnight/experiments/exp_20_interference/result.md`.
- `include/cdna4/ops/group/distributed/packet.cuh` — new
  `accumulate_peer_bf162` accumulating peer transport (remote packed-bf16
  atomic, the exp_18/21-enabling primitive), next to the existing
  plain/multi-region/streaming packet transports.
- `distributed-kernels/fused_moe/BENCHMARKING.md` — the measurement half of that
  handoff, so "paired timing" resolves to a runnable procedure on the 8× MI350X
  (`gfx950`) node. Pins the MoK synthetic-prefill campaign in `amd-master`
  (`benchmarks/mok_synthetic_prefill/run_campaign.sh`: eager, 500 warmup / 100
  timed, seed `1234+rank`, index-aligned rank-max, median primary, rotated arm
  order, MoK MXFP8 correctness blocking and the strict k0 gate diagnostic), the
  three arms — `production`, `pf6gm_mega` (the past-best megakernel this port
  descends from) and the not-yet-registered `mps_mega` — the cross-repo arm
  registration checklist in `prefill_opt/host/e004pf_k0pf_ab.py`, the rule that
  the `C × g × mode` sweep travels in the descriptor config word rather than as
  arm names, and the `K0_PF6GM_G` default trap that would otherwise compare a
  G=3 candidate against a G=2 reference. The decision number is
  `mps_mega / pf6gm_mega`.

- [`docs/distributed/competition-analysis/`](docs/distributed/competition-analysis/) — an
  analysis of the 17 fastest hand-written submissions to the AMD MI300X×8
  `amd-all2all`, `amd-gemm-rs`, and `amd-ag-gemm` leaderboards
  ([GPUMODE/kernelbot-data](https://huggingface.co/datasets/GPUMODE/kernelbot-data)),
  read against the CTA-level overlap taxonomy above. Archives every submission
  analysed plus per-shape timings. Principal findings: no top-10 kernel on any
  of the three boards wins via CTA-level comm/compute specialization, and three
  competitors independently built and then abandoned it; where static
  specialization *is* used, the comm pool is sized to the interconnect (one CTA
  per XCD per peer link, 2.6–10% of the grid) rather than to a compute/comm time
  balance; the submission implementing our per-tile CAS publication scheme is
  23% slower than one that publishes readiness exactly once. The report also
  documents that the `amd-all2all` benchmark is degenerate — its "expert" is a
  scalar gain, so its rankings cannot be used to compare communication
  strategies. Includes a synchronization cookbook (uncached symmetric heaps that
  delete fences, monotone epoch counters that delete flag resets, wave-execmask
  counted fan-in, NOOP padding for static fan-in) and a prioritized
  do / reconsider / do-not-build list against `sync.cuh`, `completion.cuh`,
  `counter.cuh`, `pgl.cuh`, and `roles.cuh`.

**News**
- [January 2026] HipKittens is accepted to [MLSys 2026 in Seattle]()!
- [February 2026] Will presented HipKittens as a GPU Mode lecture, [check it out](https://www.youtube.com/watch?v=jsYyF03Fs3o)!
- [March 2026] HipKittens is officially an AITER backend! The first [HK kernels have landed in AITER](https://github.com/ROCm/aiter/pull/2039)!

## Setup

```bash
# clone the repo
git clone git@github.com:HazyResearch/HipKittens.git
**or**
git clone https://github.com/HazyResearch/HipKittens.git

# For MI350X and MI355X with gfx950 arch:
# obtain an amd docker using docker pull or podman pull
podman pull docker.io/rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta

# enter the docker
podman run -it \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    -v $(pwd):/workdir/ \
    -e USE_FASTSAFETENSOR=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta \
    bash

# For MI300X/MI325X, use below docker for gfx942 arch:
podman pull rocm/7.0-preview:rocm7.0_rel_30_ubuntu22.04_py3.10_pytorch_release_2.8.0

#enter the docker
podman run -it \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    -v $(pwd):/workdir/ \
    -e USE_FASTSAFETENSOR=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    rocm/7.0-preview:rocm7.0_rel_30_ubuntu22.04_py3.10_pytorch_release_2.8.0 \
    bash

# set the environment variables
cd HipKittens/
source env.src

# install aiter (baseline kernels)
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
```

## Unit tests

We provide unit tests for you to optionally test the correctness of library functions. 

```bash
cd HipKittens/tests/unit
make -j64
```

## Quick start: running kernels

We assume you will run the following on an MI350X or MI355X unless otherwise specified. You should use the CDNA3 branch of HK to run on the MI300X or MI325X.

1. **BF16 GEMM**
```bash
# Defaults to 8192x8192x8192
# This will compare to AITER and PyTorch automatically.
cd kernels/cdna4/gemm/bf16fp32/
make clean && make
python bench.py

# On the mi300x or mi325x run:
git checkout cdna3 # not the main branch!
cd kernels/gemm/bf16fp32/mi325x/8192_256_256_64_16/
make clean && make
python test_python.py
```

2. **Attention forwards (MHA, GQA, Causal, Non-causal, Head dim 128 / 64)**

```bash
# GQA, Non-causal, D=128, N=2048, H=64, H_KV=8, B=16:
# This will compare to AITER automatically. 
cd kernels/cdna4/attn/gqa/
make clean && make
python test_python.py
```

- Modify the ```ATTN_N``` sequence length (e.g., 1024, 2048, 4096, 8192), ```ATTN_H``` query heads and ```ATTN_H_KV``` key value heads (e.g., 16 and 16 for MHA), ```ATTN_D``` head dimension (i.e., 64 or 128) in the Makefile and test_python.py file to try other settings.
- Use the same process for [gqa_causal](https://github.com/HazyResearch/HipKittens/tree/main/kernels/cdna4/attn/gqa_causal).

3. **Attention backwards (MHA, GQA, Causal, Non-causal, Head dim 128 / 64)**

```bash
# GQA, Non-causal, D=128, N=8192, H=64, H_KV=8, B=16:
# This will compare to AITER automatically. 
cd kernels/cdna4/attn/gqa_backwards/
make clean && make
python test_python.py 
```

- Modify the settings in the same way as stated above for forwards.
- Try [gqa_causal_backwards](https://github.com/HazyResearch/HipKittens/tree/main/kernels/cdna4/attn/gqa_causal_backwards).

4. **Memory bound**

```bash
# Rotary (default B=16, H=16, D=128, N=2048)
# This will compare to AITER, PyTorch, PyTorch compiled automatically.
cd kernels/cdna4/rotary/
make clean && make
python test_python.py
```

```bash
# Layernorm fused (default B=16, H=16, D=128, N=4096)
# This will compare to PyTorch, PyTorch compiled automatically.
cd kernels/cdna4/layernorm/
make clean && make
python test_python.py
```

Potential issues:
- If you see a complaint that AITER is not building in the ```test_python.py``` files, then install AITER from source [following this README.md](https://github.com/ROCm/aiter/tree/main). Luckily, it is very quick! You can also comment out AITER from ```test_python.py``` if you only need the HK kernel.
- If you see an error that ```bin/hipcc/``` is not found, then edit the Makefile to replace ROCM_BUILD_DIR with ```/opt/rocm/bin/hipcc```


## Benchmarking

Under [HipKittens/analysis](https://github.com/HazyResearch/HipKittens/tree/main/analysis) we provide scripts and instructions to benchmark all the HK kernels from our paper. This will sweep over different dimensions and settings, and we provide plotting scripts. 

**Note:** We also provide the instructions to reproduce our baselines (Triton, CK, HipBLASLT, Mojo, etc.) in [HipKittens/analysis/baselines](https://github.com/HazyResearch/HipKittens/tree/main/analysis/baselines)! As these are constantly evolving frameworks, we remind that our results are collected in November 2025.

## Training

Under [HipKittens/training](https://github.com/HazyResearch/HipKittens/tree/main/training) we provide instructions to train either BERT or Llama models using HipKittens attention kernels, AITER kernels, or PyTorch kernels. These are lightweight. Run them within the AMD Docker.

## Resources

We provide resources for profiling kernels, dockers, and HipKittens in [HipKittens/docs](https://github.com/HazyResearch/HipKittens/tree/main/docs). Contribute to our [onboarding documents](https://docs.google.com/document/d/15-Zvf6e0NLX1si4ml4sUOWCDlXNMtOWKiuo6CKZMEYA/edit?usp=sharing).

### Get in touch!

Contact: William Hu [willhu@stanford.edu](willhu@stanford.edu) and Simran Arora [simran@cs.stanford.edu](simran@cs.stanford.edu).
Join us on Discord to get involved, [GPU Mode Invite](https://discord.gg/ssgGe4HT) and then you can join the [TK channel](https://discord.com/channels/1189498204333543425/1300872762163728550)! We welcome community contributions.

If you use or build on this work, please consider citing:
```
@misc{hu2025hipkittensfastfuriousamd,
      title={HipKittens: Fast and Furious AMD Kernels}, 
      author={William Hu and Drew Wadsworth and Sean Siddens and Stanley Winata and Daniel Y. Fu and Ryann Swann and Muhammad Osama and Christopher Ré and Simran Arora},
      year={2025},
      eprint={2511.08083},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.08083}, 
}
```

## Training megakernels (branch `ablations`, 2026-08-15)

`distributed-kernels/fused_moe/` gained the T2B backward megakernel family
(saved-plan dY dispatch, z-regeneration, dH2/dX GEMM phases, and the M8.5
in-kernel wgrad phase in `n2_wgrad_gm_t2b.cpp` with its standalone
validation harness `wg_standalone.hip`).  On the 8x MI350X DeepSeek-V3
proxy: per-layer fwd 4.75 ms + backward-dgrad 7.65 ms (vs ~20-24 ms for
the AMD turbo grouped-GEMM production path), end-to-end 19,390 tok/s/GPU
converging — and at matched fp8 precision, the production recipes diverge
(NaN) where this stack's 128x128-blockscale contract trains.  Full results:
amd-master `auto-gpu-kernel/k0_fused_moe/training_bench/T1_RESULTS.md`.

## v6: wgrad-in-bubbles (branch `ablations`, 2026-08-17)

`k0pf6gm_device_tile_t2v6.hip` — additive sibling of the T2B backward that
executes the v2.2 wgrad tiles as FILLER inside the megakernel's own wait
windows (service-CTA slab 0 / post-quota slab 1, and a poll-and-fill M8
certificate wait on all 256 CTAs), driven by two device cursors that respect
the readiness lattice (dW2 claimable from slab 0, dW13 from slab 1) with the
M8.5 phase demoted to a cursor drain; quotas and the cross-microbatch
accumulate ride a packed DWMODE word (accumulate derived in-kernel from the
epoch counter).  A compile-gated PROF arm adds a monotonic per-wait-site
idle-cycle ledger (descriptor slot 73) to size every bubble at training
geometry.  The slice/accumulate plumbing (`tile_lo`/`tile_hi`/`accumulate`)
landed in `n2_wgrad_gm_t2b.cpp` + `wg_standalone.hip` for exactly this.

## fp8-on-wire combine design (branch `ablations`, 2026-08-18)

`distributed-kernels/fused_moe/overnight/aug18-prefill/FP8_WIRE_DESIGN.md` —
design record for fp8 e4m3 combine payloads plus source-side pre-reduce in the
M15 prefill megakernel.  Establishes that the byte lever is reachable only on
top of the staged arm (`K0P6_M15_STAGED`): gfx950 has no fp8 remote RMW, so the
transport becomes local bf16 fold + posted fp8 stores + owner-side
dequant-reduce in M8 (2.931x fewer remote bytes; row 14,336 B -> 7,392 B with
56 inline fp32 group scales).  Includes measured paired-build evidence that the
staged arm's M7 epilogue is instruction-identical to the shipped ratchet (282
`pk_add_bf16` / 96 `vmcnt(4)` / 180 `v_mfma`), so the format change costs the
epilogue nothing — and a P0 finding that `K0P6_M15_STAGED=1` currently emits 96
scratch reloads and 97 `s_waitcnt vmcnt(0)` beside the remote atomic (the
exp_24 pathology with exp_38's throttle-annihilation consequence), which blocks
any A/B of the arm until cleared.  No kernel code changed.

## M23 "ragged seal" design (branch `ablations`, 2026-08-18)

`distributed-kernels/fused_moe/overnight/aug18-prefill/M23_RAGGED_SEAL_DESIGN.md` —
serving-integration design record for the highest-priority coverage fix.  New
per-step instrumentation showed the `k0pf6gm_m15_mega` graph is nearly inert in
real serving: the seal fires only when all eight DP ranks have *exactly* 4096
pre-padding tokens, which is ~2% of the ~230 in-bucket steps per c32p run, while
production's stock B4096 graph runs all 230.  The doc establishes with file:line
evidence that stock vLLM **dispatches every padded row** through Mori+AITER
(`is_padding` is None on the V1 runner, `VLLM_MOE_SKIP_PADDING` defaults off), so
parity needs **no megakernel change and no shim math change** — the megakernel's
`T` is a launch constant and its capacities are already sized `world x T`.  M23
is therefore a pure seal relaxation: a DP-unanimous predicate built only from
all-reduced data (padded counts, synced cudagraph mode, plus a new readiness row
folded into the existing 4x8 all-reduce), a mode override so a uniform-decode
rank cannot split the collective, and real coverage counters
(`RAGGED_SEAL_RECEIPT`) replacing the one-shot receipt latches.  Projected
coverage 2% -> 100% of in-bucket steps (0.77% -> 38.3% of all steps, ~50x).
Also records two findings that invalidate every prior serving A/B: an unsealed
in-bucket step in a PF4H-target server runs with **no cudagraph at all**, and a
rank whose local batch looks like a uniform decode runs the whole model eagerly
at 4096 padded tokens in *both* arms.  Top risk is the fail-closed
`M15_SPIN_LIMIT` against ragged cross-rank arrival skew.  No code changed.

## M23 implementation + adversarial-review fix (branch `ablations`, 2026-08-18)

`distributed-kernels/fused_moe/overnight/aug18-prefill/m23/` — the design landed
as a post-apply, **in-container** patcher (`m23_patch.py`, marker
`PF4H_M23_RAGGED_SEAL_V1`, chained strictly after `coverage_patch.py` and fatal
if the order is inverted), plus CPU tests (`test_m23_patch.py`, 179 checks that
run the real patch chain against byte-copies of the deployed vLLM mirror) and the
L0 offline dispatcher replay (`offline_dispatcher_replay.py`).  Implementation
notes and the full deviation list: `m23/M23_IMPL_NOTES.md`.

Two adversarial reviews then found a **split-collective the fix itself
introduced**, and it is fixed here.  vLLM's DP engine runs `execute_dummy_batch()`
on any rank that had nothing scheduled while the group is running, and that rank
goes through the *same* `_determine_batch_execution_and_padding` and the *same*
all-reduce as its busy peers.  The first implementation's readiness bit ignored
the call context — `_dummy_run`'s `pf4h_graph_target` defaults to `False`, not
`None`, so the idle rank contributed a readiness bit of **1** while the seal
predicate's own `m23_serving` term (which it did check) meant that same rank
could never seal.  Result on a routine c32p step: seven ranks seal and replay the
megakernel graph, the eighth runs stock Mori+AITER eagerly — seven ranks spin to
`M15_SPIN_LIMIT` and fail closed, the eighth's all2all hangs, and every counter
reports green because the R3 assert and `eager_b4096` are gated on the same local
term.  Pre-M23 this was structurally impossible (the idle rank's original count
of 1 broke the `(4096,)*8` gate for everybody).

The fix: serving-ness now reaches the seal **only** through the all-reduced
readiness bit (a `_dummy_run` or capture-drive rank contributes 0, so the whole
group refuses in lockstep and records `refused_peer_not_ready`), `m23_serving` is
removed from `b4096_unanimous` so the predicate really is built from all-reduced
data only, and the local term survives just where a per-rank decision is safe —
the uniform-decode mode rescue.  Coverage for the row: a whole-group predicate
test with a real modelled all-reduce, an H3 (`dummy_run`) class in the L0 replay
with a new gate **G4** that raises `SPLIT SEAL` the instant the eight ranks
disagree, and a regression switch that reproduces the pre-fix readiness bit and
proves G4 catches it.  The patcher also installs atomically now
(`tmp` + `fsync` + `os.replace`), so an interrupted write cannot leave a marker
inside a truncated file.  Still open: `DescriptorSlot.SPIN_DBG` has no reader
anywhere in the shim, so probe P2 (the blocking gate for the throughput A/B)
cannot be executed as written.

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
wide-push transport arm. Unmeasured; contract and gate ladder in
`overnight/aug12/M15_DESIGN.md`.

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

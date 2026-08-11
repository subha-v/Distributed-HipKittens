# Overnight agent handoff — fused-MoE MPS megakernel

You own the 8×MI350X node (`gbt350-odcdh2-c05-1`, gfx950) overnight. Your job, **in
this order**: (1) root-cause and fix a deterministic first-launch
`address (nil)` fault in the `mps_mega` arm; (2) run the full gate ladder
(world-8 correctness, negative control, 600-epoch soak, per mode); (3) benchmark
`mps_mega` against `production` and the previous best megakernel `pf6gm_mega`
(the user's actual question); (4) sweep the role-split space; (5) extend the
same COMET/MoK/TileLink role-split idea to OTHER phases of the kernel
additively — the combine boundary is only the first target. **Do not stop after
a fix or a first benchmark: the round is the point. Report everything with
numbers; negative results are results.**

## What the user wants answered (highest priority)

1. Does `mps_mega` beat `pf6gm_mega` (6,902 µs) and `production` (7,703 µs) on
   the MoK synthetic campaign? That is THE question.
2. After it works: which `C ∈ {0,4,8,16,32}` × `g ∈ {1,2,4}` ×
   `mode ∈ {0,1,2}` × `flush_rows` × `pull_fallback` wins, measured by
   `summarize.py`'s median rank-max p50 ratio, one campaign invocation per
   sweep point, arms `production,pf6gm_mega,mps_mega` every time.
3. Then: does role-splitting other boundaries (e.g. dispatch/M1 or M2 streaming
   under compute, COMET-style layer-1 for M6) pay? COMET reference:
   https://arxiv.org/pdf/2502.19811 — read it before inventing.

## Repos, branches, exact build

- `~/Distributed-HipKittens` branch `codex/distributed-hipkittens-scaffold`.
  The kernel is `distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip`
  (entry `k0pf6gm_mps_mega`). Siblings: `moe_mps_adapter.cuh`,
  `n2_phase2_gm_mps.cpp`, `moe_host_abi.hpp`, `DESIGN_MPS.md`, `BENCHMARKING.md`.
  Both the parity port (`k0pf6gm_device_tile.hip`, entry `k0pf6gm_mega`) and
  upstream amd-master donor trees are read-only for you unless a bug is PROVEN.
- `~/amd-master` branch `debug/pf6-first-launch` (see its own handoff
  `auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/MPS_KERNEL_FIX_HANDOFF.md`
  and this run's `MPS_OVERNIGHT_HARNESS_NOTE.md`).
- Build container: `docker exec subha_k1` (ROCm 7.2.4 / LLVM 22, mori JIT at
  `/usr/local/lib/python3.12/dist-packages/mori/_jit-sources`). Direct genco
  build (verifiable reproduction of the current resource table):

```bash
hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
  -Rpass-analysis=kernel-resource-usage \
  -I$HOME/Distributed-HipKittens/include \
  -I$HOME/Distributed-HipKittens/distributed-kernels/fused_moe \
  -I$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip/hkp \
  -I$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/kernels \
  -I$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip \
  -I$MR -I$MR/include -I$MR/src \
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
  $HOME/Distributed-HipKittens/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip \
  -o k0pf6gm_device_tile_mps.gfx950.hsaco
```

`$MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources`. clang-offload-
bundler `--unbundle` then `llvm-objdump -d --mcpu=gfx950` for ISA work.

## Measured state (ROCm 7.2.4/LLVM 22, this exact source)

| metric | parity port | MPS (current head) |
|---|---:|---:|
| Total SGPR | 106 | 104 |
| VGPR / AGPR | 256 / 256 | 256 / 256 |
| Scratch B/lane | 36 | 60 (+24, cold paths) |
| SGPR spill count (metadata) | 10 | ~94–106 (noisy; 0 s-sourced scratch ops in ISA) |
| VGPR spill count (metadata) | 8 | 14 |
| LDS B/block | 155428 | **155428 (EXACT)** |
| static v_mfma census | 96+84=180 | 96+84=180 (identical) |
| scratch ops inside MFMA spans | 0 (min dist. 705 insns) | 0 (min dist. 511 insns) |

Correctness status: **NOT YET PROVEN — first-launch `address (nil)` fault.**
`production` (rel_L2≈0.0065) and `pf6gm_mega` (rel_L2≈0.0098), both `pperr=0`,
run cleanly in the same harness moments earlier.

## The fault — what is PROVEN today (2026-08-11)

Deterministic `Memory access fault by GPU node-2..9 on address (nil)` on all
eight GPUs at the `mps_mega` arm's first launch under
`K0_MPS_CFG="C=8,g=2,mode=2,flush_rows=16"`.

`K0_MPS_DEBUG_STOP` bisect (now container-forwarded, verified engaged — results
change per stop value):

| stop | phases executed | result |
|---:|---|---|
| 1 | M0 | **CLEAN** (no fault) |
| 2 | M0+M1 qpush | CLEAN |
| 3 | +M2 | CLEAN |
| 4 | +M3–M5 (+plan/scatter/part zero) | CLEAN |
| 5 | +M6 phase-1 GEMM | CLEAN |
| 6 | +M7 phase-2 GEMM **with the stream-mode enqueue hooks** | CLEAN |
| 0 (full) | +M7.6 service waves + M8 dynamic + M9 | **`(nil)` FAULT, all 8 GPUs** |

So the fault lives in `{M7.6 service path internals, M7.6 env setup, M8 dynamic
combine, M9}`. debug-stop runs show `rel_L2=1.00000, pperr=0` (zero output) and
`control_fails=True` — the negative control still fails as required.

**Invalid/annotated evidence (redo before trusting):**
- A "service-loop no-op" probe was made and STILL faulted — but its build
  provenance is unverified (mori-jit rebuild line not confirmed in that log). If
  you need this probe, redo it and confirm the new source hash compiled: mori-jit
  cache keys hash source bytes.
- A "skip the mega launch" probe was INVALID (`K0_MPS_SKIP_LAUNCH` was not
  container-forwarded at the time; ALL K0_MPS_* knobs are forwarded now —
  `K0_MPS_DEBUG_STOP`, `K0_MPS_DESC_DUMP`, `K0_MPS_TRACE`, `K0_MPS_SKIP_LAUNCH`).
- Additional host knobs land in the mps desc: `K0_MPS_DESC_DUMP=1` writes
  `/out/mps_desc_rank*.txt` with the exact 63 words at first launch;
  `K0_MPS_TRACE=1` writes `/out/progress_rank*.log` markers per arm.

**Descriptor/host facts verified by assertions:** `desc_mps.numel()==63`; donor
words 0..54 shared with the working pf6gm arm; slot 55 points to a 9-word
device snapshot `{local_heap_base, peer_base[0..7]}`, host-readback validated
(nonzero, self-slot consistent); config word passes `config_is_valid`; slots
60/61 alignment checks pass.

## Suspect ranking (start at the top, falsify one at a time)

1. **Slot-61 pointer flavor / heap-relative offset.** desc[61] holds the mori
   `shmem_malloc` raw pointer for the 448 MiB slots arena. `peer_ptr()` in the
   push path computes `local - local_heap_base + peer_base` — if `slots` is
   NOT inside `[heapBaseAddr, heapBaseAddr+heapSize)`, translated addresses are
   wrong; a wrong-but-nonnull result would still be a GPU fault. Verify on
   device: dump desc word 61 vs the 9 heap words per rank (`K0_MPS_DESC_DUMP`),
   and assert range membership in the kernel entry guard (one sub-page change)
   before believing anything else. The same check for the `row_ready` slot
   (donor, works in parity — probably fine).
2. **M8 dynamic-claim path** (`s_ns[wid]` broadcast, `m8_next` ticket, slot
   addressing in `k0p6_mps_m8_batch` under `base_slot`). Untouched by all valid
   probes; the only subregion never independently cleared. `K0_MPS_CFG` with
   `pull_fallback=1` keeps streamed flags but switches M8 to remote `part`
   pulls: if THE COMBINATION (mode 2 + pull_fallback) runs clean, the fault is
   in slot addressing; if it still faults, it's in the service/stream protocol.
3. **M7.6 `run_service` internals** (wave stripes, pending-flag flush,
   acq_rel probes). Redo the no-op probe properly (confirm the mori-jit
   rebuild happened).
4. **`fetch_add_relaxed<scope::agent>` on `mps_state + K0P6_MPS_ST_TAIL` in the
   M7 hook** (ran clean at stop 6 — cleared, listed for completeness).

## Gate order for your fix (do not shortcut)

1. Fault root-caused and fixed; **document which side the bug was on
   (kernel vs harness) precisely in `MPS_OVERNIGHT_HARNESS_NOTE.md`**.
2. Quick smoke: `K0_MOK_ARMS=production,mps_mega WARMUP=1 TIMED=1` passes
   `pperr=0` and `control_fails=True` for each of modes 0, 1, 2
   (`mode` via `K0_MPS_CFG`; mode 1 requires `C=0`).
3. 600-epoch soak, `pperr=0`, for at least the mode-2 campaign config.
4. Three-arm campaign (`production,pf6gm_mega,mps_mega`, 5 rotations, 500/100)
   per `BENCHMARKING.md` §5. Report `arm_p50_us` for all three and both ratios.
5. The sweep: mode 0 `C∈{4,8,16}` first (pure tax), then mode 1, then mode 2
   `C×g∈{4,8,16}×{1,2,4}`, then `pull_fallback` at the best point, then a
   timestamps-on attribution run. One campaign per point.
6. Only then: additive research on other boundaries under the same
   correctness/soak discipline (never edit the frozen donor or the parity
   port; additive siblings, same-shape controls, one variable per experiment).

Quick smoke form (work from `~/amd-master/auto-gpu-kernel/k0_fused_moe`):

```bash
setsid timeout 1800 env \
  K0_MOK_ARMS=production,mps_mega \
  K0_MOK_WARMUP_ITERS=1 K0_MOK_TIMED_ITERS=1 \
  K0_MOK_OUTPUT_ROOT=$HOME/k0-mok-synthetic-mps-smoke \
  K0_MOK_RUN_TIMEOUT=1500 \
  K0_MPS_CFG="C=8,g=2,mode=2,flush_rows=16" \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh <tag> 1
```

## Node discipline (non-negotiable)

One 8-GPU job at a time; check `pgrep -af 'torchrun|mpirun'` and
`rocm-smi --showpids` before launching; `setsid` + `timeout` for long runs;
**never `SIGKILL`** a GPU process (leaked IPC wedges the node); other tenants'
containers (`yuhan_dsv4_0806` etc.) are off-limits. `subha_k1` is fine for CPU
compiles. Commit+push after every completed or failed experiment; a commit is
not durable until the remote branch is verified.

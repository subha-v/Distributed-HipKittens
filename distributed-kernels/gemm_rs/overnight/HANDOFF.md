# GEMM-RS MI300X overnight handoff — evidence ledger and blockers

Written 2026-08-11 ~00:00 PT at the end of the bring-up session. Everything
below is either a measurement taken on `banff-sc-cs47-05.dh170.dcgpu` or an
explicitly labelled hypothesis. Read `RESULTS.md` for the full gate-by-gate
record; this file is the *state of play* and the *two blockers*.

## One-paragraph state

Our kernel works. It compiles for gfx942, is numerically correct on all 17
graded + official shapes, survives a 600-epoch skewed soak, fails correctly
under all three negative controls, and runs in graph mode with device-derived
epochs advancing per replay. It has been benchmarked **in our own harness**
(single process, 8 devices, peer access) at a geometric mean of **285.7 µs**
over the six graded shapes. What we do **not** have is a number for our kernel
or for the competitor under the **official evaluator**, which is the only
protocol in which the two are comparable. Both of those are blocked, for
different reasons, and both blockers are described below with suspect rankings.

## What is proven (do not re-litigate)

| Gate | Result |
|---|---|
| M1 gfx942 build, production + control TUs | PASS (7 instantiations; production exports only `gemm_rs_mi300x`, control only `gemm_rs_mi300x_control`) |
| M2 ISA + resources | PARTIAL — occupancy and ISA properties pass, scored 256×256 rows spill |
| M3 world-8 correctness | PASS, 17/17 shapes, worst `max|diff| = 4.883e-4` vs tolerance `1e-2` |
| M4 negative controls | PASS, all three fail in their designed way |
| M5 600-epoch skewed soak, no host reset | PASS, epochs and signals exact at every checkpoint |
| M8 graph mode | PASS, epochs advance per replay, outputs track changing inputs |

Three source fixes were required and are committed:

1. `gemm_rs_mi300x_hk_adapter.cuh` used unqualified `bf16` inside its own
   namespace (`hk_gemm_rs` spells that type `std::uint16_t`) — added a
   `using bf16 = kittens::bf16;` alias.
2. `mi300x_globals::grid()/block()/dynamic_shared_memory()` were non-const but
   called through a `const&`.
3. `hipFuncSetAttribute` was called on **every launch**; it is a property of the
   instantiation, so it is now set once per process.

Also: the Gate M1 command in `MI300X_VALIDATION.md` is missing
`-I$(ROCM_PATH)/include/hip`, without which `<hip_bf16.h>` is not found. The
documented command cannot work as written.

## Measured performance, our harness (pinned clocks, duration-based warmup)

Order-rotated, 3 rotations × 50 pipelined iterations, every timed run verified.

| # | shape | mean µs | stdev | host issue µs | SOL µs | ×SOL |
|---|---|---|---|---|---|---|
| 1 | 64×7168×18432 | 108.2 | 0.09 | 61.8 | 6.46 | 16.7 |
| 2 | 512×4096×12288 | 115.4 | 0.19 | 63.3 | 8.19 | 14.1 |
| 3 | 2048×2880×2880 | 97.3 | 0.66 | 62.4 | 23.04 | 4.2 |
| 4 | 4096×4096×4096 | 203.5 | 1.84 | 62.6 | 65.54 | 3.1 |
| 5 | 8192×4096×14336 | 765.4 | 2.76 | 61.1 | 131.07 | 5.9 |
| 6 | 8192×8192×29568 | 2872.3 | 6.75 | 62.3 | 379.43 | 7.6 |

Geometric mean **285.7 µs**; graph mode **279.5 µs** (only 1.00–1.03× faster,
which is what proves these are device-bound and not host-bound).

**Do not use ×SOL as a target.** The published SOL table is the bf16 MFMA
roofline with *zero* budget for the reduce-scatter: shape 6's
`2·8192·8192·3696 / 1.307e15 = 380 µs` versus a tabulated 379.43 µs. rank-1's
own published score (413.139 µs geomean, on GPU MODE's machine) is ~10.4× that
table. The only denominator that means anything is a competitor measured here.

## Where the time goes (ablation, macro-gated scratch copy; deliverable untouched)

Single-cut deltas, so they overlap and need not sum.

| shape | full | GEMM | XGMI egress | reduce | sync | per-tile release |
|---|---|---|---|---|---|---|
| 64×7168×18432 | 109.5 | ≥45.2 | 2.9 | 1.3 | 11.4 | 0.4 |
| 512×4096×12288 | 116.8 | ≥50.0 | 20.8 | 6.1 | 16.1 | 16.0 |
| 2048×2880×2880 | 98.9 | 23.6 | 32.6 | 8.6 | 21.4 | 12.5 |
| 4096×4096×4096 | 205.0 | 39.9 | 84.8 | 20.3 | 30.3 | 18.5 |
| 8192×4096×14336 | 773.1 | 335.5 | 294.9 | 39.7 | 60.9 | 70.5 |
| 8192×8192×29568 | 2861.7 | 1317.8 | 919.7 | 219.5 | 246.9 | 250.3 |

Resource tuple per instantiation (from `-Rpass-analysis=kernel-resource-usage`):

| BM/BN/BK | VGPR | AGPR | SGPR | scratch | occupancy | VGPR spill |
|---|---|---|---|---|---|---|
| 32/256/32 | 94 | 0 | 106 | 0 | 5 waves/SIMD | 0 |
| 64/64/64 | 93 | 0 | 106 | 0 | 5 | 0 |
| 128/256/32 +tail | 170 | 0 | 106 | 0 | 2 | 0 |
| 256/256/32 (±tail) | 256 | 0 | 106 | **12 B** | 2 | **2** |
| 32/64/64 (±tail) | 91 | 0 | 106 | 0 | 5 | 0 |

`AGPRs: 0` everywhere — accumulators sit in arch VGPRs at the 256 cap. That is
the likely root of the spill and the first thing to try.

## BLOCKER A — our kernel under the official evaluator

**This is the priority.** Status: the multi-process path *works*; the evaluator
integration stalls.

Proven working: `harness/hk_submission.py` + `harness/mp_smoke.py` ran 8 ranks
under real `torch.distributed` with a **HIP IPC symmetric heap**, and produced
`allclose=True, max|diff|=9.766e-04` on 512×4096×12288. Peer IPC exchange,
descriptor construction through the production host ABI, and the kernel itself
are all fine across processes.

Stall: under `eval.py benchmark`, the warmup case completes, then the first
timed case hangs. Instrumented stderr shows only **6 of 8 ranks** reach
`state ready`; two are stuck inside `_ShapeState.__init__`.

### Suspect ranking

1. **`hipIpcOpenMemHandle` returning `hipErrorAlreadyMapped` on a repeated
   shape.** We deliberately never free, but we *do* allocate a fresh heap for
   each new process group. If the allocator hands back an address whose IPC
   handle matches one this process already mapped, the open fails and that rank
   dies inside setup — which would hit some ranks and not others, exactly the
   6-of-8 signature. **Recommended fix: cache state by `(rank, m, n, k, bias)`
   persistently across process groups and never redo the exchange for a shape
   already set up.** All ranks then skip the collectives together, which keeps
   setup symmetric, and reusing the old peer pointers is valid because nothing
   was freed and the processes are the same. Epoch counters stay consistent
   because every rank reuses the same state the same number of times.
2. **A NCCL `dist.barrier()` is enqueued, not blocking.** A
   `torch.cuda.synchronize()` after the setup barrier was added but never got a
   clean run. If setup is not actually ordered before the first launch, a
   producer can write into a peer heap that does not exist yet.
3. **Pool worker respawn breaking collective symmetry.** `eval.py` drives ranks
   with `multiprocessing.Pool(8)` and the worker↔rank map is *not* stable across
   test cases (this already caused a real bug — see below). If the pool
   respawns a worker, that worker has no cached state and takes a different
   branch from its peers.

### Already fixed here, do not re-discover

- **Worker↔rank permutation.** A cache keyed only by shape handed rank 1 an
  allocation belonging to cuda:6. The evaluator reported
  `Output device mismatch: cuda:6 != cuda:1`. State is now keyed by rank and by
  process-group identity, and `torch.cuda.set_device(rank)` is called on entry.
- **Teardown between test cases.** Closing peer IPC mappings while other ranks
  may still hold them, plus the asymmetry when a fresh worker has nothing to
  release, was removed entirely — allocations are retained in `_RETAINED`.

### The invalid numbers

One evaluator run *did* emit timings (303 µs / 463 µs / 2099 µs / 3043 µs /
21509 µs for shapes 1–5). **These are meaningless** — they came from the run
with the device-mismatch bug, where most ranks were operating on another
device's buffers; benchmark mode does not re-check correctness, so it happily
timed garbage. The giant standard deviations (±11 ms on shape 5) are the tell.
Do not quote them.

## BLOCKER B — the competitor (rank-1) on this node

`gemm_rs_rank1_58abcf.py`, sha256
`7940fcb81df06c1d8b1e1a77051f23c934149a688441ef48b2751b3f336f0dc5`, verified
against `MI300X_PROVENANCE.md`.

It was written against an **older iris and an older Triton** than anything on
this node, so running it is version archaeology. Four repairs so far, each
found only after fixing the previous one, each ~25 min to test because failures
surface as opaque multiprocessing timeouts:

| # | Problem | Repair | Verified |
|---|---|---|---|
| 1 | `iris` not importable; rank-1 reads it from a literal `/usr/local/lib/python3.10/dist-packages/iris/__init__.py` | stage the checkout at exactly that path, and put it on PYTHONPATH — one copy satisfies both | yes |
| 2 | rank-1 runs `os.system("sudo sed -i '66,82 s/^/#/' .../iris/__init__.py")`; in the revision here lines 66–82 are `from . import hip` / `experimental` / `logging`, i.e. it would delete the `iris.hip` rank-1 then calls | no-op `sudo` shim early on PATH; iris verified to import unpatched | yes |
| 3 | Triton 3.6.0 has no `wrap_handle_tensor_descriptor` | `tools/compat/sitecustomize.py` supplies a stub that **raises** if called; it is only reached for `tensordesc` args, which these kernels do not use. Verified never called (`grep -c SHIM_WAS_CALLED` = 0) | yes |
| 4 | rank-1's C launcher parses `packed_metadata` as 6 ints; Triton 3.6.0 packs 3 (`num_warps, num_ctas, shared_memory`) | `tools/patch_rank1.py` changes `"iiiiii"`→`"iii"` and pins `clusterDim{X,Y,Z}=1`. **Behaviour-preserving**: those three are forwarded to `_launch` and never read — only `num_warps` and `shared_memory` reach `hipModuleLaunchKernel`. The patcher refuses to run unless the source sha256 matches | yes, error changed |
| 5 | rank-1 calls `iris.hip.hipIpcMemHandle_t()`; **all 8** iris checkouts on this node renamed it `gpuIpcMemHandle_t` | alias added to `sitecustomize.py` | **NOT TESTED** — the run was stopped mid-flight |

After repair 4 the failure became a pure timeout. rank-1 spawns per-rank helper
processes (`CREATE_SHEMEM_CODE`) that allocate an 8 GiB heap, write
`heap_bases_{rank}.pkl`, then *"sleep forever wait for a.txt"* — a file-based
coordination protocol that also has to work here. Repair 5 addresses the most
likely reason those helpers were dying. **Test repair 5 first.**

Cheap progress probe, no log reading required: `heap_bases_*.pkl` in the arm
directory. Those files exist **only** if a helper got past
`iris.hip.hipIpcMemHandle_t()` and `get_ipc_handle()`. Zero of them means the
helpers are still dying at the iris call and repair 5 did not take; eight of
them means the iris side is solved and the remaining hang is in the file-based
rendezvous or further downstream. `tools/check_rank1d.sh` prints this plus how
far each pass got. As of handoff: **zero**, from a run that was killed during
the warm pass, so it is not yet evidence either way.

### Cheaper calibration available right now

`exp026` ran the official evaluator to **11/11 passes in 93 s** on this node,
and its `submission.py` is the **reference implementation** — plain
`torch.matmul` + `torch.distributed.reduce_scatter_tensor`, i.e. the GEMM+RCCL
baseline. Benchmarking that costs one evaluator run and no new engineering, and
gives a real same-node anchor while rank-1 is being revived. It has not been
done. `tools/run_competition_bench.sh` already stages it as the `reference` arm.

## Methodology warnings — all of these cost real time today

- **Pin the clocks.** Idle `sclk` here is ~120–132 MHz against ~1900 MHz under
  load. A fixed 10-iteration warmup measured a still-ramping GPU: the same
  shape read 95.7 µs in a back-to-back sweep and 153.1 µs in a fresh process, a
  60% swing, and an entire phase attribution built on it was wrong. Use
  `tools/set_clocks.sh pin 1900` and duration-based warmup. **Clocks are
  currently left pinned at 1900 MHz.**
- **Never run `sed` over binaries.** A line-ending normalizer that walked every
  file under the harness stripped `0x0D` bytes out of `build/*.so` and corrupted
  them. The symptom was a `dlopen` segfault in every process that looked exactly
  like a multiprocessing/HIP-init bug. `tools/push.ps1` now filters by
  extension; keep it that way.
- **Use `tools/nsh.ps1`, not inline `ssh host "..."`.** PowerShell eats `$(...)`,
  `(`, `<`, `2>/dev/null` out of double-quoted remote commands, and Windows CRLF
  breaks bash. Piping a script body over stdin with CRs stripped avoids both.
- **The graded tolerance has almost no detection power.** With inputs in
  ±0.01, corrupting a whole 1-of-8 reduction contribution produced
  `max|diff| = 7.5e-3`, inside `allclose(1e-2, 1e-2)`. A healthy run is
  `4.9e-4`. Every check in this harness also asserts a tight `2e-3`.
- **`CTRL_REROUTE_SLOT` is only observable across changing inputs**: the
  rerouted band leaves the destination slot holding the previous epoch's bytes,
  which are the correct answer if the inputs did not change. It reported a
  false pass once for exactly this reason.
- **Allocation granularity does not matter** (tested): coarse vs fine-grained
  payload is within 1% on every shape and correct in all arms. The protocol's
  `buffer_wbl2` / `buffer_inv` handshake is doing its job. Signals are still
  allocated fine-grained on principle.

## Exact reproduction

Container `dhk-gemmrs` (created by `tools/setup_container.sh`, image
`vllm/vllm-openai-rocm:v0.26.0`, runs as uid 15523 so artifacts stay
user-owned). A second container `dhk-eval` runs as root and exists only to
write `/usr/local/lib` for the rank-1 iris staging.

```bash
# from the repo root on Windows
powershell -File distributed-kernels/gemm_rs/overnight/tools/push.ps1
powershell -File distributed-kernels/gemm_rs/overnight/tools/nsh.ps1 `
  -Script distributed-kernels/gemm_rs/overnight/tools/validate_tree.sh
```

```bash
# on the node, ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
bash $ON/tools/set_clocks.sh pin 1900          # host, before any timing
docker exec dhk-gemmrs bash $ON/harness/build.sh
docker exec -w $ON/harness dhk-gemmrs python3 smoke.py
docker exec -w $ON/harness dhk-gemmrs python3 m3_correctness.py all
docker exec -w $ON/harness dhk-gemmrs python3 m4_controls.py
docker exec -w $ON/harness dhk-gemmrs python3 m5_soak.py 600 512 4096 12288 1 1
docker exec -w $ON/harness dhk-gemmrs python3 m7_bench.py 3 50
docker exec -w $ON/harness dhk-gemmrs python3 m8_graph.py
docker exec -w $ON/harness dhk-gemmrs python3 exp_ablation.py 40
docker exec -w $ON/harness dhk-gemmrs python3 exp_reducer_sweep.py 40

# multi-process (blocker A), fastest debug loop, full stderr:
docker exec -w $ON/harness dhk-gemmrs bash -c \
  'cp hk_submission.py submission.py; HK_DEBUG=1 python3 -u mp_smoke.py'

# the official evaluator, our kernel:
bash $ON/tools/run_ours_evaluator.sh          # HK_ONE=1 for a single shape
# the official evaluator, rank-1:
bash $ON/tools/run_rank1_bench3.sh
```

## Files and ownership

- Kernel (edit additively, never break the ABI):
  `../gemm_rs_mi300x.cpp`, `../gemm_rs_mi300x_hk_adapter.cuh`,
  `../gemm_rs_mi300x_constants.cuh`, `../gemm_rs_mi300x_host_abi.hpp`.
- Design/gates/provenance: `../MI300X_DESIGN.md`, `../MI300X_VALIDATION.md`,
  `../MI300X_PROVENANCE.md`. `MI300X_VALIDATION.md`'s
  `PENDING_GFX942_VALIDATION` markers for M1/M3/M4/M5/M8 are now discharged and
  should be updated with the measurements in `RESULTS.md`.
- **Read-only:** `../gemm_rs_device_tile.cpp` (the gfx950 sibling), everything
  under `~/amd-master/**`, every other `distributed-kernels/` module, and
  `.node/**` at the repo root — that directory is another agent's fused-MoE MPS
  work, do not touch it.

# GEMM-RS MI300X/gfx942 — first hardware bring-up: measured results

Node: `banff-sc-cs47-05.dh170.dcgpu`, 8x AMD Instinct MI300X (gfx942), SPX mode
(304 CUs/GPU), all-to-all XGMI, ROCm 7.2.4 / HIP 7.2.53211.
Container `dhk-gemmrs` from `vllm/vllm-openai-rocm:v0.26.0`: Python 3.12,
torch 2.11.0+rocm, pybind11 3.0.4.

Everything below is a measurement taken on that node. Before this work the port
had never been compiled for gfx942 and had no runtime binding at all; every gate
in `MI300X_VALIDATION.md` was labelled `PENDING_GFX942_VALIDATION`.

## Gate status

| Gate | Subject | Result |
|---|---|---|
| M1 | gfx942 build, production + negative-control TUs | **PASS** (2 source fixes needed) |
| M2 | ISA + resource inspection per instantiation | **PARTIAL** — occupancy/ISA pass, scored rows spill |
| M3 | world-8 correctness, official + scored + generic | **PASS**, 17/17 shapes |
| M4 | three negative controls fail as designed | **PASS** (after strengthening one detector) |
| M5 | 600-epoch skewed changing-input soak, no reset | **PASS** |
| M6 | semantic twins | not started |
| M7 | paired timing vs baselines | **PARTIAL** — this port measured; baselines pending |
| M8 | graph mode with advancing epochs | **PASS** |

## Source fixes required to compile (M1)

Both were latent bugs that could not have been caught without a gfx942 compiler.

1. `gemm_rs_mi300x_hk_adapter.cuh` used unqualified `bf16` inside
   `namespace hk_gemm_rs_mi300x`. `hk_gemm_rs` spells that type
   `std::uint16_t` throughout, so the name never resolved. Fixed with a
   `using bf16 = kittens::bf16;` alias in the namespace.
2. `mi300x_globals::grid()/block()/dynamic_shared_memory()` were non-const but
   called through a `const mi300x_globals&` in `launch_fixed`.

Also: the Gate M1 command in `MI300X_VALIDATION.md` is missing
`-I$(ROCM_PATH)/include/hip`, which the repository's own `kernels/common.mk:52`
supplies. Without it `<hip_bf16.h>` is not found. The documented command cannot
work as written.

A third fix was made for performance rather than compilation — see
"per-launch host API call" below.

All seven required instantiations are emitted: `32/256/32`, `64/64/64`,
`128/256/32+K_TAIL`, `256/256/32`, `256/256/32+K_TAIL`, and generic `32/64/64`
with and without the runtime K tail. Link-time separation holds: the production
module exports only `gemm_rs_mi300x`, the control module only
`gemm_rs_mi300x_control`.

## Gate M2 — resources per instantiation

| BM/BN/BK | VGPR | AGPR | SGPR | scratch B | occupancy | VGPR spill | LDS B |
|---|---|---|---|---|---|---|---|
| 32/256/32 | 94 | 0 | 106 | 0 | 5 waves/SIMD | 0 | 36864 |
| 64/64/64 | 93 | 0 | 106 | 0 | 5 | 0 | 32768 |
| 128/256/32 +tail | 170 | 0 | 106 | 0 | 2 | 0 | 49152 |
| 256/256/32 | 256 | 0 | 106 | **12** | 2 | **2** | 65536 |
| 256/256/32 +tail | 256 | 0 | 106 | **12** | 2 | **2** | 65536 |
| 32/64/64 | 91 | 0 | 106 | 0 | 5 | 0 | 24576 |

- **Residency assumption holds.** Every instantiation reaches at least
  2 waves/SIMD, and 2 waves/SIMD x 4 SIMDs / 8 waves-per-CTA = 1 CTA/CU, which
  is exactly what `MI300X_DESIGN.md` section 7(a) requires. There is no margin
  above it.
- **Gate M2 fails its own no-spill requirement** on scored rows 4, 5 and 6
  (all `256/256/32`): 2 VGPR spills and 12 B of scratch per lane.
- **`AGPRs: 0` everywhere.** On CDNA3 the accumulators are sitting in arch
  VGPRs, at the 256 cap, instead of the accumulation register file. This is the
  most likely first move for the spill fix.
- LDS is requested dynamically, so `group_segment_fixed_size` reads 0 in the
  metadata; the real per-CTA figure is `2*(BM+BN)*BK*2` as tabulated.

ISA properties, verified in context rather than by counting:

- Emit lowers to 16-byte stores: `global_store_dwordx4` / `flat_store_dwordx4`
  present, with `global_store_short` only on the bounded tail paths.
- Release is directional: `buffer_wbl2 sc0 sc1` followed by
  `s_waitcnt vmcnt(0)`, inside `release_payload_system`, with **no** invalidate.
- Reduction issues `global_load_dwordx4` (16 B) per source.
- Bounded polls back off with `s_sleep`, 2 sites per kernel.
- MFMA is `v_mfma_f32_16x16x16_bf16` only; `v_mfma_f32_32x32x8` is never
  selected, which is a tuning question for the mainloop rewrite.

## Gate M3 — correctness, 17/17

All six graded shapes and all eleven official test cases, on all 8 ranks, over
3 epochs each with changing inputs, against the evaluator's own oracle
(`bf16(x @ w.T) + bias` per rank, then reduce-scatter).

Worst deviation over the whole sweep: `max|diff| = 4.883e-4` against a graded
tolerance of `1e-2`. Epoch cells read exactly the launch count; all touched
ready/credit cells read exactly the final epoch; no error bits anywhere.

Coverage includes the K-tail path (`k_local=360`, `BK=64`), the N-tail path
(row 3, `2880 % 256 = 64`), the generic row for 11 shapes, and the smallest
legal M (`m=64`, i.e. 8 rows per rank).

## Gate M4 — negative controls

| Control | Expected | Observed |
|---|---|---|
| `CTRL_DROP_PUBLICATION` | consumer times out, reads nothing | bit 26 set on exactly the starved rank; its output **100%** untouched sentinel |
| `CTRL_DROP_CREDIT` | starved source's epoch-2 producer times out | epoch 1 clean and correct; epoch 2 sets bit 25 on exactly that source |
| `CTRL_REROUTE_SLOT` | numerics fail | 57133/57344 and 57144/57344 elements differ from golden on ranks 1 and 5 |

Two findings about the controls themselves:

1. **`CTRL_REROUTE_SLOT` is only observable across changing inputs.** The
   rerouted band leaves the true destination slot holding its previous epoch's
   bytes; with unchanged inputs those bytes are already the correct answer, so
   the control is invisible. The first run of this control reported a false
   pass for exactly that reason.
2. **The graded tolerance has very little detection power at this input scale.**
   With inputs in +/-0.01, replacing one of eight reduction contributions with
   same-distribution values produced `max|diff| = 7.5e-3`, inside
   `allclose(rtol=1e-2, atol=1e-2)`. A healthy run sits at `4.9e-4`, so a
   tolerance of `2e-3` separates them cleanly; that tighter bound is now
   asserted alongside the graded one everywhere in the harness.

## Gate M5 — 600-epoch soak

600 eager launches, inputs regenerated every epoch, launch order rotated every
epoch so a different rank is last to start, and no host reset of any signal or
epoch state at any point.

- every one of the 600 epochs verified correct (worst `4.883e-4`)
- final epoch cells: all scheduled cells `== 600`, all unscheduled `== 0`
- final signals: every touched ready/credit cell `== 600`
- no error bits

## Gate M8 — graph mode

Captured graphs replayed with inputs changed in place between replays. Epoch
cells advanced by exactly the captured launch count on every replay (5, 9, 13,
17, 21), and outputs tracked the new inputs each time rather than reproducing a
stale answer. The RadeonFlow stale-`signal_val` failure mode is absent.

Graph mode was also the control for a measurement worry: it is only
1.00-1.03x faster than the eager path, which establishes that the eager numbers
below are device-bound, not host-bound.

## Timing — this port (Gate M7, part 1)

Pinned clocks (`rocm-smi --setperfdeterminism 1900`), duration-based warmup,
3 rotations x 50 pipelined iterations, order-rotated so no shape is
systematically first. Every timed run was verified and its error bits and epoch
cells checked.

| # | shape | mean us | stdev | host issue us | SOL us | x SOL |
|---|---|---|---|---|---|---|
| 1 | 64x7168x18432 | 108.2 | 0.09 | 62 | 6.46 | 16.7 |
| 2 | 512x4096x12288 | 115.4 | 0.19 | 63 | 8.19 | 14.1 |
| 3 | 2048x2880x2880 | 96.5 | 0.02 | 62 | 23.04 | 4.2 |
| 4 | 4096x4096x4096 | 201.7 | 1.52 | 63 | 65.54 | 3.1 |
| 5 | 8192x4096x14336 | 770.7 | 4.40 | 61 | 131.07 | 5.9 |
| 6 | 8192x8192x29568 | 2875.0 | 7.15 | 62 | 379.43 | 7.6 |

Geometric mean 285.1 us against a published SOL geometric mean of 39.80 us.

**These ratios are against an analytically unreachable bound and should not be
used as a target.** The SOL table's own numbers correspond to the bf16 MFMA
roofline (shape 6: `2*8192*8192*3696 / 1.307e15 = 380 us` versus a tabulated
379.43 us), i.e. they budget zero time for the reduce-scatter. The only
meaningful denominator is a competitor measured on this node, which is the
measurement in progress.

## Where the time goes (ablation)

Scratch copy of the kernel with macro-gated cuts; the deliverable source is
untouched, so `MI300X_DESIGN.md` section 9's "no ablation code is present"
remains true. Single-cut deltas, so they overlap and need not sum.

| shape | full | GEMM | XGMI egress | reduce | sync | per-tile release |
|---|---|---|---|---|---|---|
| 64x7168x18432 | 109.5 | >=45.2 | 2.9 | 1.3 | 11.4 | 0.4 |
| 512x4096x12288 | 116.8 | >=50.0 | 20.8 | 6.1 | 16.1 | 16.0 |
| 2048x2880x2880 | 98.9 | 23.6 | 32.6 | 8.6 | 21.4 | 12.5 |
| 4096x4096x4096 | 205.0 | 39.9 | 84.8 | 20.3 | 30.3 | 18.5 |
| 8192x4096x14336 | 773.1 | 335.5 | 294.9 | 39.7 | 60.9 | 70.5 |
| 8192x8192x29568 | 2861.7 | 1317.8 | 919.7 | 219.5 | 246.9 | 250.3 |

Ranked optimization targets, from measurement rather than intuition:

1. **GEMM mainloop, 46% of the largest shape.** This is the documented
   "schedule upgrade path": the current mainloop is correctness-first
   double-buffering with a full `__syncthreads()` per k-iteration and a
   blocking global->LDS load, at 1 CTA/CU so nothing hides the latency.
2. **XGMI egress, 32%.** Shape 6 moves ~117 MB off-rank per epoch in 920 us,
   i.e. ~127 GB/s against roughly 448 GB/s of per-GPU XGMI — about 3.5x off,
   consistent with 16-byte scattered peer stores.
3. **The per-tile release, 9%** (250 us on shape 6). `producer_drain_release`
   is issued once per tile, and each one is a `buffer_wbl2 sc0 sc1` L2
   writeback plus a `vmcnt(0)` drain. Grouping the release across the tiles a
   CTA produces, instead of per tile, is the obvious move.
4. reduce ~8%, cross-rank sync ~9%.

## Tuning experiment 1 — NUM_REDUCER_CTAS

Every split passed correctness. Times in us.

| shape | red tiles | NR=8 | 16 | 24 | 32 | 40 | 48 | best | table | gain |
|---|---|---|---|---|---|---|---|---|---|---|
| 64x7168x18432 | 28 | 127.3 | 114.2 | 114.4 | 108.6 | 108.6 | 108.7 | 40 | 32 | 0.0% |
| 512x4096x12288 | 64 | 149.9 | 126.9 | 120.4 | 114.1 | 114.8 | 115.6 | 32 | 48 | 1.3% |
| 2048x2880x2880 | 24 | 126.7 | 112.1 | 98.5 | 97.3 | 97.6 | 97.8 | 32 | 48 | 0.5% |
| 4096x4096x4096 | 32 | 269.8 | 225.1 | 224.8 | 203.5 | 203.8 | 203.7 | 32 | 48 | 0.1% |
| 8192x4096x14336 | 64 | 894.4 | 805.5 | 787.2 | 766.4 | 769.5 | 772.4 | 32 | 32 | 0.0% |
| 8192x8192x29568 | 128 | 2850.2 | 2678.1 | 2632.9 | 2632.1 | 2653.2 | 2685.7 | 32 | 8 | **8.3%** |

**A single uniform `NUM_REDUCER_CTAS = 32` is optimal or within noise of
optimal for all six shapes**, and beats the inherited RadeonFlow per-shape
table (32/48/48/48/32/8). The table's `NR=8` for shape 6 costs 8.3%. Carrying
over a donor's submitted constants was not free.

## Two negative results worth keeping

- **Allocation granularity does not matter.** The payload heap was allocated
  fine-grained (uncached) on the theory that this penalized even the reducer's
  local reads. Coarse-grained payload is within 1% on every shape, and
  correctness holds in all arms. The hypothesis was wrong; the protocol's
  writeback/invalidate handshake is doing its job.
- **A per-launch host API call was real but not the floor.**
  `hipFuncSetAttribute` was being called on every launch inside
  `launch_fixed`. It is a property of the instantiation, so it is now set once
  per process. This cut host issue cost from ~90 us to ~62 us per operation but
  did not move device time. The remaining host cost is `pyutils` re-introspecting
  the torch tensor arguments per call; the harness now prebuilds the argument
  tuple.

## A methodology trap that produced wrong numbers

Idle `sclk` on this node is ~120-132 MHz against ~1900 MHz under load. An early
version of the harness used a fixed 10-iteration warmup, which for a 100 us
kernel is a few milliseconds of load — nowhere near enough to reach steady
clocks. The same shape measured 95.7 us in a back-to-back sweep and 153.1 us in
a fresh process, a 60% swing, and the phase attribution built on those numbers
was wrong. Fixed by pinning clocks with `rocm-smi --setperfdeterminism 1900`
(see `set_clocks.sh`) and making warmup duration-based. Post-fix run-to-run
spread is under 1%.

## Harness

`MI300X_DESIGN.md` section 3 leaves allocation, IPC and descriptor plumbing to
"the future runtime binding". That binding did not exist, so it was written:

- `harness/dhk_rt.cpp` — symmetric heap allocation (fine and coarse grained),
  peer access, descriptor construction and upload, device/host copies, and the
  shape resolver. It calls the **production** host ABI
  (`snapshot_allocation_descriptors`, `resolve_shape`) unmodified rather than
  reimplementing it, so the kernel receives the ABI's own derived counts.
- `harness/shim/iris/iris.hpp` — a 20-line `iris_device_view` supplying the
  three accessors that ABI consumes, so the real snapshot path can run without
  IRIS or MPI.
- `harness/harness_lib.py` — world-8 driver, evaluator-verbatim input
  generation, oracle, epoch/signal/error inspection, and four timing modes.

**Known limitation, and it matters for the baseline comparison.** World-8 is
realized as one process driving eight devices with peer access enabled. That is
address-space equivalent for this kernel — the descriptor holds each rank's own
base explicitly and `translate_peer` only computes
`peer_base + (local_ptr - local_base)` — but it is *not* the evaluator's
topology, which is one process per rank with `torch.distributed`. Running this
port under the official evaluator requires the symmetric heap to survive across
processes (hipIPC handle exchange, or wiring up IRIS properly) and is the
largest remaining engineering item.

## Reproducing

`ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight` on the node;
kernel work runs inside container `dhk-gemmrs`.

```bash
bash $ON/tools/setup_container.sh          # host: create the container
bash $ON/tools/set_clocks.sh pin 1900      # host: pin clocks before any timing
docker exec dhk-gemmrs bash $ON/harness/build.sh
docker exec -w $ON/harness dhk-gemmrs python3 smoke.py
docker exec -w $ON/harness dhk-gemmrs python3 m3_correctness.py all
docker exec -w $ON/harness dhk-gemmrs python3 m4_controls.py
docker exec -w $ON/harness dhk-gemmrs python3 m5_soak.py 600 512 4096 12288 1 1
docker exec -w $ON/harness dhk-gemmrs python3 m7_bench.py 3 50
docker exec -w $ON/harness dhk-gemmrs python3 m8_graph.py
docker exec -w $ON/harness dhk-gemmrs python3 exp_ablation.py 40
docker exec -w $ON/harness dhk-gemmrs python3 exp_reducer_sweep.py 40
docker exec -w $ON/harness dhk-gemmrs python3 exp_granularity.py 30

bash $ON/tools/run_ours_evaluator.sh       # official evaluator, our kernel
bash $ON/tools/run_rank1_bench3.sh         # official evaluator, rank-1
```

From Windows: `tools/push.ps1` to sync, `tools/nsh.ps1 -Script <file>` to run a
script remotely (never inline `ssh host "..."` — see HANDOFF.md).

# exp_21 design — task graph, buffers, roles, traffic model, rejected alternatives

Companion to `plan.md`. Everything here is about the **ubench module**, which
is standalone: it links no production object, is never imported by the graded
binding, and lives entirely under
`overnight/aug11/exp_21_saturation/` with its own `build/`.

## 0. Why a new module at all

There is **no grid override anywhere in this codebase**. The production grid is
hardcoded `dim3(m3::CU_COUNT)` = 304 (`gemm_rs_mi300x.cpp:140`) and the only
host-varied CTA scalar is `num_gemm_ctas`, which moves the producer/reducer
*boundary* inside a fixed 304-CTA grid. A saturation curve needs the grid
itself as the x-axis, so it needs its own launch, its own buffers and its own
argument POD. Bending the production kernel to accept a grid parameter would
put a diagnostic-only knob in the graded ABI, and would also conflate protocol
with payload (see §7).

The bodies, however, are **not** re-implemented. Mode b calls the shipped
`m3::pull_sum_bf16_strip_mlp8` directly out of
`gemm_rs_mi300x_hk_adapter.cuh`; mode c calls the shipped
`hk_gemm_rs::store_peer_packet16` and `kittens::distributed::translate_peer<8>`
and reproduces `emit_band_packets`' index arithmetic line for line; mode a is a
transcription of the `gemm_rs_mi300x.cpp` k-loop including the two volatile-asm
anchors (`m3::acquire_frags`, `m3::acc_anchor`) and the `s_setprio` window,
because removing any of them changes the schedule and the ISA (exp_09).

## 1. Files

| file | role |
|---|---|
| `sat_ubench.cpp` | the HIP module: 3 mode bodies, 7 kernel instantiations, calibration kernel, fill kernel, destination verify kernel, pybind11 surface (allocation, peer access, device props, launch, calibrate, verify) |
| `build.sh` | hipcc into `exp_21_saturation/build/` — **never** `harness/build/`, which holds another agent's exp_20 ablation arms |
| `isa.sh` | `--save-temps` + resource-tuple extraction + store-width / MFMA opcode probes; writes `isa_report.txt` |
| `ceilings.sh` | this node's own xGMI / HBM / clock reporting, verbatim tool output into `ceilings.txt` |
| `run_saturation.py` | the driver: peer access, allocation, tick calibration, the sweep, `saturation.json` |
| `run_sweep.sh` | node-side wrapper (`setsid`+`timeout`, clock pin check, log capture) |

## 2. Roles and task graph

One launch = one grid of `G` CTAs of 512 threads, 65,536 B dynamic LDS
requested in every arm (so occupancy is 1 CTA/CU in every arm, matching
production).

```
isolated:            G = C, every CTA has the resource role, dense id = blockIdx.x
concurrent:          G = 304
                       pid <  C   -> resource role, dense id = pid
                       pid >= C   -> GEMM filler,   dense id = pid - C, count 304-C
reserve_control:     G = 304
                       pid <  C   -> idle spin to a tick deadline (capacity only)
                       pid >= C   -> GEMM filler, identical code to the live arm
```

Edges (only two, both one-way, both agent-scope, both outside every hot loop):

1. `res_done` — `u32`, incremented once by tid 0 of each resource CTA when it
   finishes its work list (or its deadline). The GEMM filler's stop condition.
2. `work_done[304]` — `u32`, one relaxed store by tid 0 at the end of each CTA:
   tiles completed (GEMM role) or work units completed (resource role). The
   FLOP/byte numerator is built from these, so a work list that did not
   complete cannot inflate a throughput.

The GEMM filler's stop check is made **CTA-uniform** without any extra LDS:
mode a already owns the entire 64 KB LDS budget, so a `__shared__ int` would
overflow the per-workgroup limit. Instead tid 0 writes the flag into the first
4 bytes of the A double buffer at the top of a tile, `__syncthreads()`, every
thread reads it, `__syncthreads()` again, and the tile's `load_commit`
overwrites those bytes before they are used as operands. Two extra barriers per
~113 µs tile. Checking *before* the tile starts (not after) means the GEMM
never overshoots the resource role, so the concurrent window contains no
traffic-free tail.

Timestamps: `stamps[304][2]`, `u64`, `s_memrealtime()` by tid 0 at kernel entry
and kernel exit. Every throughput in this experiment is
`numerator / (max end − min start over that role's CTAs)`. Host wall time is
recorded per point as a cross-check, never as the metric.

## 3. Buffers (per rank)

| buffer | granularity | size at defaults | why |
|---|---|---|---|
| `heap` (symmetric, mode-c destination) | fine-grained by default, `--payload-coarse` for the production-parity cross-check | 8 slots × 32,768 rows × 8,192 cols × 2 B = **4 GB** | slot index = *source* rank exactly as `dst[{me,0,lrow·EB,tn·BN}]`; 32,768 windows per slot ≥ the 32,768-window work list, so every push has a unique destination address and the verify is exact |
| `sig` (symmetric) | fine-grained | 64 KB (guard 64 u32 + 2 cells per (CTA, peer)) | the `--protocol` arm's probe RMW and publication targets; distinct cells per (CTA, peer), so the RMW is uncontended exactly as production's per-(src,lrow,col) cells are |
| `red_src` | coarse (cached) | 8 slots × 4,096 × 8,192 × 2 B = **512 MB** | the reducer reads the *local* heap through L2/MALL in production; > 256 MB so panel b is an HBM curve, not an Infinity-Cache curve |
| `red_dst` | coarse | 4,096 × 8,192 × 2 B = 64 MB | the reducer's local output |
| `a_mat`, `b_mat` | coarse | 2 × 1,024 × 3,712 × 2 B = 15.2 MB | 16 distinct 256×256 tiles, deliberately cache-resident |
| `stamps`, `work_done`, `res_done`, `verify_out` | coarse | < 16 KB | instrumentation |

Peer addressing uses the production path end to end: the 8 heap bases are
packed into `kittens::peer_bases<std::byte>`, wrapped in the 72-byte
`hk_gemm_rs::symmetric_descriptor`, passed by value in the argument POD, and
dereferenced only through `kittens::distributed::translate_peer<8>(local_ptr,
local_allocation_base, bases, rank)` — i.e. `peer_base + (local_ptr −
local_base)`, the same arithmetic the megakernel does. The heap base *is* the
region base, so the offset arithmetic is exercised with a zero suballocation
offset. World-8 is one process driving eight devices with
`hipDeviceEnablePeerAccess` (`rt.enable_peer_access(8)` first, always) — the
harness's model, address-space-equivalent for `translate_peer`'s purposes, and
**not** the evaluator's one-process-per-rank topology. That caveat is repeated
in `result.md`.

## 4. Traffic model (what each metric counts, exactly)

- **a**: `2 · 256 · 256 · (k_iters · 32) · Σ work_done[gemm CTAs]` FLOP over the
  GEMM role's span. `k_iters = 116`, so K = 3,712 and `K_TAIL = false` (the
  instantiation where exp_09 found the scheduler hoisting MFMAs above the
  fragment wait — hence the anchors are mandatory here too).
- **b**: `9 · 256 · 256 · 2 B` per strip (8 packet reads + 1 packet write per
  8 columns, over 256 rows × 32 packet-columns) × strips, over the resource
  role's span. Reads and writes are counted together as HBM traffic, which is
  what the sibling's `(8+1)·bytes·rows` does.
- **c**: `16 B × 1,024 packets × windows` = 16 KB per window, over the resource
  role's span. Per-rank egress; `single` puts every rank's egress on one link of
  a ring (so a per-rank number is a per-link number), `rr7` spreads each rank's
  egress over all 7 peers (so a per-rank number is an aggregate-egress number).
  The JSON records per-rank values and their sum; the point value is the
  **median across the 8 ranks** — ranks are not averaged into a single figure
  without also showing the spread.

## 5. Kernel instantiations (7) and why the set is that shape

`sat_kernel<MODE, CONC, CONTROL>`; `if constexpr` keeps mode a out of the
isolated b/c binaries, so those two get their true small footprints, while the
concurrent binaries contain both roles and therefore inherit the union
footprint — exactly what production does, since every branch of one HIP kernel
inherits the kernel's max resource footprint.

| instantiation | arm |
|---|---|
| `<0,false,false>` | mode a isolated |
| `<1,false,false>` | mode b isolated |
| `<2,false,false>` | mode c isolated |
| `<1,true,false>` / `<1,true,true>` | mode b concurrent / reserve control |
| `<2,true,false>` / `<2,true,true>` | mode c concurrent / reserve control |

Plus `sat_calib_kernel` (tick-rate calibration), `sat_fill_kernel` (operand and
source fill, so no 512 MB host transfer is needed), `sat_verify_kernel`
(destination-side bitwise verify + fold).

## 6. Tick-rate calibration

Two-stage, because the rate is the unknown: stage 1 spins for 1e6 ticks and the
host times it to get a rough Hz; stage 2 spins for `0.2 s × rough_Hz` so the
host-timed interval is ~200 ms and launch/teardown overhead is < 0.1%. Five
reps, median, spread recorded. `tick_rate_hz` and `tick_rate_spread_pct` go in
the JSON header and every µs figure in this experiment is derived from them.
The sibling's 100 MHz is treated as a hypothesis to confirm, not an input.

## 7. Alternatives rejected

- **rocprof / rocprofv3 counters as the bandwidth source.** Cannot separate the
  two roles inside one launch, which is the whole point of the concurrent
  overlay. Kept only as a one-time cross-check on a single isolated point, and
  only if the GPU is free; a counter pass must never run concurrently with
  another GPU job on this node.
- **`hipMemcpyPeer` / `hipMemcpyAsync` between devices.** Prices the copy
  engine (SDMA), not the in-kernel pusher. The paper's claim is about what the
  GEMM epilogue's own stores can achieve, so the measurement has to be the
  epilogue's own store shape issued by CTAs whose count we control.
- **Reusing the full megakernel as the ubench** (e.g. via `num_gemm_ctas`).
  Conflates protocol with payload — every emit is wrapped in credit waits, a
  release and publications — and cannot vary the grid at all (§0). The
  `--protocol` switch exists precisely so the protocol term is measured
  *additively* inside a payload-only baseline.
- **In-kernel byte/packet counters.** Adds atomics to the path being measured.
  Numerators come from `work_done[]` plus host-side arithmetic instead.
- **In-kernel xor accumulation for the checksum.** Would put 4 VALU ops and a
  live 64-bit register per 16 B store into the hot loop, and per-peer folds
  would cost 8 live registers in a kernel already at ~250 VGPRs in the
  concurrent arms. A destination-side bitwise verify against the generating
  function is strictly stronger and free.
- **A `__shared__` stop flag for the filler.** Overflows the 64 KB
  per-workgroup LDS limit that mode a already exhausts (§2).
- **Letting waves disagree on the stop condition** (each wave testing
  `s_memrealtime()` or the counter itself). Divergent arrival at the mainloop's
  `__syncthreads()` is a hang risk; the LDS-flag handshake makes the decision
  CTA-uniform by construction.
- **A per-C-rescaled work list.** Would make "total work identical at every CTA
  count" false and hide a per-point sizing bug inside the throughput ratio.
  Rejected in favour of stating the 10–50 ms band violation at the extremes
  (`plan.md` §4) and recording `rounds` per point.
- **Coarse-grained peer destination with no release** (i.e. the naive isolated
  arm). The issuing XCD's L2 can absorb the stores, so it would report L2
  bandwidth as fabric bandwidth. See `plan.md` §8.8; both granularities are
  runnable and the choice is recorded per point.

## 8. Failure modes and what each looks like

| symptom | most likely cause | check |
|---|---|---|
| mode c GB/s above the node's aggregate xGMI figure | stores absorbed locally | `payload_granularity`, and the `--payload-coarse` vs fine pair at the same point |
| `checksum_ok == false` with plausible GB/s | a destination address collision or a dropped window | `work_done` sum vs the work list; verify per slot |
| concurrent ≈ isolated | the GEMM filler died early | `work_done` for the GEMM role, and `gemm_span / resource_span` (recorded per point) |
| mode a TFLOPS flat in C | operands falling out of cache, or spills | `isa_report.txt` scratch/spill counts; A/B footprint |
| large rotation spread | clocks unpinned | `clocks_before` / `clocks_after` per point |

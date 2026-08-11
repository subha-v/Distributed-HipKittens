# GEMM -> ReduceScatter on MI300X/gfx942 — design and static proofs

Status: implemented source, host ABI, static checks, and offline simulations.
Every GPU-dependent statement is labelled `PENDING_GFX942_VALIDATION` and has an
exact later-node command in `MI300X_VALIDATION.md`. Nothing in this document
claims GPU correctness, performance, ISA parity, or leaderboard victory.

This port is additive: the gfx950 two-launch port in this directory is not
modified. This file is the single place that states the intent before the code;
`gemm_rs_mi300x.cpp` cross-references sections here.

## 1. Operator contract (verified against the evaluator)

Source of truth: the official evaluator in the frozen submission's spec (see
`MI300X_PROVENANCE.md` §1 for identity and hashes).

- World size is exactly 8. The evaluator asserts `m % 8 == 0` and `k % 8 == 0`.
- Rank `r` receives `x: [M, k/8]` bf16 and `w: [N, k/8]` bf16 (local K shards
  generated rank-locally with per-rank seeds), and an optional `bias: [N]` bf16
  re-seeded so the vector is identical on every rank.
- The oracle computes, per rank, `partial_r = bf16(x_r @ w_r^T) + bias` and
  then `reduce_scatter(sum over r of partial_r)`. Therefore the checked result
  contains `world * bias` and bias is applied to **every rank's partial before
  the reduction** (§15 of the static gates).
- Output on rank `r`: rows `[r*(M/8), (r+1)*(M/8))` of the reduced matrix,
  shape `[M/8, N]` bf16. Tolerance: `allclose(rtol=1e-2, atol=1e-2)`.
- Local GEMM K is exactly `k / 8` (confirmed in the evaluator `generate_input`).
  Our shape table is keyed by `(M, N, K_local, has_bias)`.

Arithmetic policy (matches both MI300X donors):

- GEMM accumulates in FP32 over the local K shard; bias is added in FP32 in the
  producer epilogue; the partial tile is rounded **once** to bf16 (RNE).
- The reducer converts the eight bf16 source contributions to FP32 and sums in
  source-ascending order with one RNE bf16 pack at output. No bias at the
  reducer — the `bias*world` reducer-side formulation of the gfx950 port is an
  algebraically equivalent but numerically different schedule used there only;
  both MI300X donors use producer-side bias.

## 2. COMET-inspired schedule (arXiv:2502.19811)

The COMET lesson used here is thread-block specialization: persistent produce
CTAs emit independently consumable tiles; separate communication/reduction CTAs
consume them; consumer CTAs start on the earliest independent shared-tensor
region instead of waiting for the whole operator. We do not import COMET's
NVIDIA implementation. On AMD the specialization is at CTA granularity, not
intra-CTA warp granularity, because:

- all branches of one HIP kernel inherit the kernel's maximum resource
  footprint (registers, LDS), so producer-warp specialization inside a CTA
  costs the producer footprint everywhere;
- MI300X has 304 CUs and 64 KiB LDS per workgroup, and a 512-thread
  MFMA-class CTA is self-sufficient as a reducer.

Topology: one persistent grid of exactly **304 CTAs / 512 threads**.
`NUM_GEMM_CTAS = 304 - NUM_REDUCER_CTAS` with the split a per-shape table
entry. The two roles partition `[0, 304)` exactly (producer pids
`[0, 304-NR)`, reducer pids `[304-NR, 304)`); there is no idle band and no
overlap. The RadeonFlow reducer counts are the initial configuration, not a
presumed optimum: `NUM_REDUCER_CTAS ∈ {8, 16, 24, 32, 40, 48, ...}` is a
retunable table column (§8).

### Per-shape proposed configuration (v1)

`EB` = emit-band height (signal row unit, §3). `lrow_count = (M/8)/EB`,
`col_count = ceil(N/BN)`. All counts are verified by the static checker and
the simulation; LDS bytes are exact, `<= 65536`.

| # | M | N | K_global | K_local | bias | BM | BN | BK | EB | red. CTAs | GEMM tiles | red. tiles | LDS B | SOL µs |
|---|---|---|----------|---------|------|----|----|----|----|-----------|------------|------------|-------|--------|
| 1 | 64 | 7168 | 18432 | 2304 | F | 32 | 256 | 32 | 8 | 32 | 2*28=56 | 1*28=28 | 36864 | 6.46 |
| 2 | 512 | 4096 | 12288 | 1536 | T | 64 | 64 | 64 | 64 | 48 | 8*64=512 | 1*64=64 | 32768 | 8.19 |
| 3 | 2048 | 2880 | 2880 | 360 | T | 128 | 256 | 32 | 128 | 48 | 16*12=192 | 2*12=24 | 49152 | 23.04 |
| 4 | 4096 | 4096 | 4096 | 512 | F | 256 | 256 | 32 | 256 | 48 | 16*16=256 | 2*16=32 | 65536 | 65.54 |
| 5 | 8192 | 4096 | 14336 | 1792 | T | 256 | 256 | 32 | 256 | 32 | 32*16=512 | 4*16=64 | 65536 | 131.07 |
| 6 | 8192 | 8192 | 29568 | 3696 | F | 256 | 256 | 32 | 256 | 8 | 32*32=1024 | 4*32=128 | 65536 | 379.43 |

BM/BN/BK are the frozen-MI300X rank-1 tile choices (its Triton `online_config`,
which `BM | M/8` for every scored shape), not the gfx950 donor's 256x256 layout
and not RadeonFlow's 224-tall tiles, which cannot route per-owner (224 does not
divide `M/8=1024`). Reducer counts are RadeonFlow's submitted scored values.
`512 threads / 8 warps` matches both the rank-1 geometry (num_warps=8) and the
gfx950 donor's producer/consumer thread class.

Why the emit band `EB` exists: a producer compute tile of `BM` rows can
straddle owner boundaries (shape 1: `BM=32`, owner slice `M/8=8` rows). The
payload is emitted at `EB = gcd(BM, M/8)` row granularity, so every emitted
dependency key names **exactly one destination slot** (gate 5). Scored shapes
have `EB = min(BM, M/8)`; the general rule is documented in the host ABI.

Generic (non-table) path: `BM=32, BN=64, BK=64`, `EB = gcd(32, M/8)`, with
register-masked M/K tails, bounded emit/reduce at the N edge, packet fast path
for 16-byte-aligned spans and a masked 2-byte tail path otherwise. Preconditions
(and only these): `M % 8 == 0`, `K % 8 == 0` (evaluator contract) and
`M % 32 == 0` for the ragged-read guard (§6) — every scored and known public
correctness case satisfies it; a general masked-load A path is a documented
extension point, not silently assumed.

## 3. Payload, dependency-key and lifetime layout

All buffers are one-time IRIS symmetric allocations, snapshotted once through
the unchanged `hk_gemm_rs::host_abi::snapshot_allocation_descriptors` helper
into the unchanged 72-byte `hk_gemm_rs::symmetric_descriptor` PODs.

- `c_heap` (symmetric bf16): `[8 sources][M/8][N]`. Producer on source rank
  `me` writes only slot `me` of the **destination** rank. The maximum slice
  bytes per source slot is `(M/8)*N*2`; the host ABI computes exact sizes and
  asserts the largest offset fits the allocation (gate 8).
- `sig` (symmetric u32): two regions in one allocation,
  `ready[src][lrow][col]` at offset 0 and `credit[owner][lrow][col]` at offset
  `ready_words`, with runtime strides `lrow_count/col_count` per launch; every
  index is bounds-checked against launch-era dimensions. Collision-freeness is
  proven by the simulation over every `(src, dest, lrow, col)` tuple (gate 7).
- `ep_cell` (LOCAL u32, two windows): one monotonic epoch cell per CTA per
  role; producer window `[0, 304)`, reducer window `[304, 608)`. Device-derived
  launch ordinals only — no host epoch, no per-call reset (gate 17), safe
  under eager replay and graph capture alike.
- `out` (LOCAL bf16): `[M/8, N]`.
- `err` (LOCAL i32): fail-closed error bits (bit 25 producer credit timeout,
  bit 26 reducer readiness timeout).

Zero-epoch invariant: `sig` and `ep_cell` are zeroed once at setup; epoch 0
means "never produced". The monotonic protocol reserves 32-bit wraparound for
a quiesced reinit; the 600-replay soak is far inside range.

Producer payload movement: FP32 accumulators are converted to bf16 (bias added
first), staged into already-dead A/B LDS at offset 0 (exactly the donor's EV=1
discipline, re-derived at `EB` row bands of at most 32 staged rows), and
emitted as aligned 16-byte packets with `hk_gemm_rs::store_peer_packet16`.
Publication is separate from movement: one `producer_drain_release<system>`
per tile covers that tile's packet stores **and** its (up to `BM/EB`) cheap
completion cells, each a relaxed epoch store on the destination rank.

Consumer movement: `pull_sum_bf16_strip_mlp8` issues all eight source packet
loads for a 16-byte slot before consuming them in ascending FP32 order (the
exp-24 REDV=1 discipline), bounds columns to `N`, rounds once, stores the
output tile locally.

## 4. Memory-order and lifetime protocol (proved in §7)

Producer, per GEMM tile `(tm, tn)`, epoch `e`:

1. `ep = epoch32(ep_cell + pid)` once per launch, convergent per CTA.
2. Mainloop -> FP32 accumulators; bias added in FP32 if present.
3. For each emit band `b` of the tile: leader performs
   `bounded_wait_slot_reusable_into(credit[dest][lrow][tn], e)`; on timeout the
   CTA marks `err bit 25` and abandons the launch. Epoch 1 is the immediate
   fast path. The CTA converges and abandons uniformly (no payload store may
   be issued by any lane after a failed wait — gate 12/14).
4. Stage each band through dead A/B LDS; emit 16-byte packets to the owner
   slot `me` on rank `dest`, columns bounded by `N`.
5. One `producer_drain_release<system>()`.
6. Leader publishes each band's `ready[me][lrow][tn] = e` on the destination
   rank (relaxed, system scope; agent scope when `dest == me`).

Reducer, per output tile `(lrow, col)` of the local slice, epoch `e`:

1. `ep = epoch32(ep_cell + 304 + pid_r)` once per launch.
2. Threads `s in [0, 8)` bounded-poll `ready[s][lrow][col] >= e` (caller-owned
   `wait_result`, relaxed, bounded by `spin_limit`).
3. If any wait timed out: `err bit 26`, converge, abandon — **no peer read is
   reachable on this path** (gate 12).
4. On success: `cta_acquire<system>()`.
5. `pull_sum_bf16_strip_mlp8` -> single RNE bf16 pack -> bounded local stores.
6. `consumer_drain()`; leader publishes `credit[me][lrow][col] = e` back to
   every source rank (relaxed; agent scope locally).

Negative controls (separate module, never in the production binding): suppress
one band publication (consumer must time out without reading), reroute one peer
band slot (numerics must fail), suppress one retirement credit (that source's
epoch-2 producer must time out without writing). GPU expectations are in
`MI300X_VALIDATION.md`.

## 5. One-launch structure

`dispatch_gemm_rs_mi300x` performs exactly one `hipLaunchKernel`-class launch
per call; no barrier kernel, no reduce kernel, no host-sync, no per-call
memset. The compile/launch cache key is `(M, N, K_local, has_bias, arch,
dtype, config-row, tail-specialization)`; every codegen-relevant variable is
either a template parameter or a member of the key (gate 20).

## 6. GEMM tail policy

- K tails (`K_local % BK != 0`, shapes 3 and 6): the exp-07 mechanism —
  load the final tile raggedly, zero A's padding K-columns **in registers**
  before the MFMA. Requires only that the ragged read does not fault (the
  donor's `oob_probe` assumption; `PENDING_GFX942_VALIDATION`).
- M tails (generic path only): same trick on A's row axis.
- N tails (shape 3: `2880 % 256 == 64`): never mask operands; bound the
  emit/reduce column ranges (`cols < N`) and never store or read out of range.
- `BM | M` and `BN | N` hold for every table entry except shape 3's N, so
  scored steady-state has no M tail; K is exact for shapes 1/2/4/5.

## 7. Progress and deadlock proof (written before the wait loops, as required)

Let `NR = NUM_REDUCER_CTAS ∈ [8, 48]`, `NG = 304 - NR ∈ [256, 296]`.

(a) **Residency.** The kernel is one grid of 304 CTAs on 304 CUs. LDS per CTA
<= 65536 B (every table row), so LDS admits at least one CTA per CU; 512
threads per CTA admits at least one per CU. The remaining resource gate
(VGPRs must allow 512-thread residence) is `PENDING_GFX942_VALIDATION` — but
progress below does not depend on full residency.

(b) **Cross-launch invariant.** Calls are stream-serialized: launch `e` begins
only after launch `e-1`'s grid has fully retired. Therefore, at any moment
inside launch `e`, every `credit[...] = e-1` publication already exists on all
ranks. A producer's credit wait (step 3) consequently completes at epoch `e`
without any reducer of epoch `e` running — reducers' completions of epoch `e-1`
precede the launch boundary. **Producers never wait within an epoch; only
reducers wait, and they wait only on producers of the same launch.**

(c) **Producer completion.** A GEMM producer CTA contains only bounded waits
(credit, proven immediately-satisfied) and bounded local barriers; it completes
once scheduled, publishing all its tiles' epochs. No producer waits on any
reducer of its own launch.

(d) **Reducer waiters.** Reducer CTAs spin on `ready` cells written by
producers. At most `NR <= 48` CTAs wait concurrently. Even under the most
adverse residency scenario — every reducer resident and spinning while only a
subset of producers is scheduled — producers are non-blocking, so each
resident producer retires and frees its CU for the next producer; the producer
completion of (c) therefore eventually covers all `NG` producers, all epochs
are published, and every reducer's bounded poll succeeds. Reducers then
complete, publish credits, and the launch ends. The worst-case capacity
argument needed is only "`>= 1` producer CU-slot is scheduleable," which holds
because spinning reducers occupy at most `NR <= 48 < 304` CU slots and waiting
CTAs do not consume scheduler work for producers beyond their own slot. If the
hardware harvests some CUs, the argument is unchanged: reducers cap at 48,
producers are non-blocking, and scheduled producers always terminate. Gate: a
named deadlock detector (the host's per-launch stream sync) plus the bounded
spin itself, which reduces any protocol bug to a clean error report, never a
wedged node.

(e) **Epoch advancement (graph/eager).** Each CTA increments its own device
cell once per launch; the cell persists across launches. After `N` launches
every `ready`/`credit` cell that was touched reads exactly `N`. No host
`signal_val` exists anywhere; the RadeonFlow stale-capture failure mode is
absent by construction.

(f) **Lifetime soundness.** Payload lifetime per `(dest, lrow, col)` source
slot is [publish `e`, credit `e`]. A producer's epoch-`e+1` write to the same
slot is blocked on `credit >= e`, which the unique consumer published only
after `consumer_drain()` following its final read of epoch `e`. Monotonic
readiness alone is never used to justify a write.

(g) **Publication counting.** Per epoch, producer CTA `p` on rank `s` writes
each band of its tiles exactly once: for tile `(tm, tn)` and band `b`, exactly
one publication targets `(dest, s, lrow, tn)` with `dest = owner(tm*BM+b*EB)`,
`lrow = (tm*BM+b*EB) - dest*(M/8) / EB` (integer by `EB | M/8` and `EB | BM`).
A reducer tile `(lrow, col)` on rank `d` therefore waits on exactly the eight
values `{ready[s][lrow][col]}_{s=0..7}` and each maps to exactly one producer
band on each rank (gate 6). Simulation §address-sim enumerates these maps for
every shape and rank.

## 8. Retuning surface

Per `config_row`: `NUM_REDUCER_CTAS`, reducer tile order (col-major vs
row-major sweep), producer WGM group width, source-issue swizzle in the
reducer (issue order only; arithmetic stays ascending), and `EB`. All are
host-table entries or small template knobs; none changes the dependency
layout. The three highest-value experiments are listed in the task deliverable
and `MI300X_VALIDATION.md`.

## 9. What is deliberately not here

No generic collective, graph, host planner, implicit world barrier, automatic
dependency allocator, or hidden scheduler. No rung/ladder ablation code is
present (fresh file); historical correctness-breaking arms of the gfx950 donor
are not carried over. `gemm_rs_mi300x.cpp` compiles to exactly one pybind
entry point in the production module, and the negative-control module is a
distinct translation-unit product that cannot be co-linked by accident
(macro-gated, statically checked).

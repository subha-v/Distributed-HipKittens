# How the fastest MI300X×8 kernels overlap communication and computation

An analysis of the top submissions to the GPU MODE / AMD `amd-all2all` (563),
`amd-gemm-rs` (564), and `amd-ag-gemm` (565) leaderboards, read against the
CTA-level overlap taxonomy in our own work.

Source: [`GPUMODE/kernelbot-data`](https://huggingface.co/datasets/GPUMODE/kernelbot-data),
`successful_submissions.parquet`, `run_mode = 'leaderboard'`, best run per user.
All three problems ran on `MI300x8` (8× MI300X VF, ROCm, xGMI).
Submissions are archived under `submissions/`; raw per-shape timings in
`per_shape_timings.txt`.

---

## 1. Headline

**Nobody in the top 10 of any of the three leaderboards won with CTA-level
communication/computation specialization.** Several competitors implemented it,
measured it, and shipped something else. The mechanisms that actually won are,
in rough order of value:

1. **Deleting synchronization** — uncached symmetric heaps and monotone epoch
   counters that make fences, flag-clearing, and inter-iteration barriers
   unnecessary.
2. **Deleting data passes** — making the collective's landing buffer *be* the
   GEMM's operand, and fusing the collective into the GEMM epilogue or the
   expert-scaling loop.
3. **Choosing the dependency granularity** — gating the consumer on a
   per-(tile, K-chunk) flag so overlap comes from the *K* dimension rather than
   from the *CTA* dimension.
4. **Hiding latency with outstanding-request depth**, not with CU count — 32
   in-flight 16 B remote loads per lane inside 8 CTAs, rather than 64 CTAs
   dedicated to communication.

Our measured result — CTA specialization at 6.866 ms vs homogeneous at
6.911 ms, with the second expert GEMM paying +1.249 ms — is *replicated
independently by three separate competitors who each abandoned the same design.*

### Independent replications of our negative result

| Kernel | What they built | What they shipped |
|---|---|---|
| all2all rank03 (u322) | `MODE==2` combine: `bid%2==0` sends, else receives, commented *"overlap send and recv"* | `MODE==3`, homogeneous — every CTA sends then receives |
| ag-gemm rank01 (u155) | `put_kernel` peer DMA over 7 side streams via `hipMemcpyDeviceToDeviceNoCU` | `init_stream()` commented out; in-kernel stores instead |
| gemm-rs rank01 (u155) | same abandoned copy-engine path | one barrier kernel, no per-tile signalling |
| all2all rank02 (u258) | ping-pong two-buffer barrier, measured at **40–90 µs** | monotone sequence number, **~1 µs** |
| gemm-rs rank02 (u258) | XCD swizzle `(tile%8)*(N/8)+(tile/8)` | commented out; ships with XCD swizzling **off** |

---

## 2. Read this before trusting any number

**The `amd-all2all` benchmark is degenerate and cannot be used to rank
communication strategies.** The reference "expert" is `token → (1 + rank) ·
token`, a scalar gain. The whole dispatch→compute→combine sequence therefore
collapses to a per-token weighted sum computable *entirely locally*:

```python
# all2all/submissions/rank07_user114_sub46436.py:15-23
dst_rank = indices // (cfg.num_experts // world_size)   # "no all2all needed"
token_expanded = x[:, None, :] * (1 + dst_rank)[..., None]
y = (token_expanded * weights[..., None]).sum(dim=1).to(cfg.out_dtype)
```

That submission moves **zero bytes between GPUs** and placed 7th at 405 µs.
Rank 9 likewise contains no `torch.distributed` call at all.

The winner (199 µs) does *not* take this shortcut — its `moe_kernel` genuinely
pulls token payload from peers over xGMI and pushes scaled partials back
(`// this pull the tokens over xGMI`). But it does inherit the degenerate
expert: the GEMM is replaced by `weight * (1 + rank)`, so **the winner has no
compute to overlap with.**

Consequences for how the numbers may be used:

- The all2all "2.04× speedup" is a *HIP-versus-PyTorch-eager* number on tiny
  tensors, not a communication-engineering result.
- No all2all submission examined contains an expert GEMM. Nothing here speaks
  to the regime where we measured the +1.249 ms second-GEMM penalty.
- `gemm-rs` and `ag-gemm` are **not** degenerate and their rankings are
  meaningful. Weight those.

### The overlap headroom is small, and measured

The trivial-wrapper baselines bound how much all this engineering is worth:

| Problem | Best custom | Best trivial wrapper | Gap | Ratio |
|---|---|---|---|---|
| ag-gemm | 373 µs | 475 µs | 102 µs | 1.27× |
| gemm-rs | 413 µs | 515 µs | 102 µs | 1.25× |
| all2all | 199 µs | 405 µs (no comms) | — | invalid |

Three functionally identical gemm-rs wrappers score 510 / 515 / 519 µs — a 9 µs
band that is the **noise floor and the entire ceiling on launch/allocator
tuning**. So ~90% of the gemm-rs win is real device-side work.

Two calibration points that should temper any overlap ambitions:

- The **10th-place** gemm-rs entry is a naive *non-MFMA, scalar `__fmaf_rn`*
  SIMT GEMM plus stock RCCL `reduce_scatter_tensor`, zero overlap — and lands
  within **17%** of the fused persistent megakernel.
- The **4th-place** gemm-rs entry is 2,432 lines of vendored `muillm` with an
  IPC symmetric heap, hand-written P2P push, custom barrier and custom bf16
  reduce (497 µs) — versus an **8-line** `F.linear` + `reduce_scatter_tensor`
  (515 µs). All that machinery buys **18 µs, 3.5%.**

Total headroom from all overlap machinery on gemm-rs is under 20%, and most of
the leaderboard spread is GEMM quality and launch overhead.

---

## 3. The taxonomy, extended

Our three categories describe *who* — which CTAs do communication. The winning
kernels mostly answer a different question: *when*, and *whether the
synchronization needs to exist at all*. The axes actually observed:

| Axis | Description | Who uses it |
|---|---|---|
| **A. Homogeneous** | every CTA walks the phase sequence | gemm-rs rank01 (**winner**), all2all rank02/rank03 |
| **B. Static CTA specialization** | fixed comm pool by `blockIdx` range | ag-gemm rank01 (**winner**, 56/536), ag-gemm rank02 (32/304), gemm-rs rank02 (8–16/304) |
| **C. Phase-adaptive / ticketed** | completion-order role switching | **nobody** — but see C′ |
| **C′. Static role fall-through** | producer CTAs re-join the consumer pool when drained, on a compile-time schedule | all2all rank05 — the cheap approximation of C |
| **D. K-granular consumer gating** | consumer waits per (tile, K-chunk); overlap from the K loop | ag-gemm rank02 — *the most sophisticated design in the set* |
| **E. Dependency-ordered tile scheduling** | tiles that need no remote data are placed first in pid space | ag-gemm rank01 |
| **F. Outstanding-request depth** | few CTAs, deep load pipelines | gemm-rs rank02 (32 in-flight 16 B loads/lane) |
| **G. Elimination** | fuse collective into epilogue; delete barriers/passes | **all winners** |

The critical observation about **B**: where static specialization does win, the
comm pool is sized to the *interconnect*, not to a compute/comm time balance,
and it is far leaner than ours.

| Kernel | Comm CTAs | Total | Share | Rationale |
|---|---|---|---|---|
| **ours** | 64 | 256 | **25%** | tuned by time balance |
| ag-gemm rank01 | 56 | 536 | 10% | `SEND_CTA_PER_DEVICE=8` = one CTA per XCD per peer link |
| ag-gemm rank02 | 32 | 304 | 10% | derived from shape: `ceil(M/8/64)·8·k` |
| gemm-rs rank02 | 8–16 | 304 | **2.6%** | compensated by load-issue depth |

`SEND_CTA_PER_DEVICE = 8` is not a tuning artifact. MI300X dispatches block *i*
to XCD *i % 8*, so exactly 8 senders per peer puts one sender on each XCD, and
every XCD hosts exactly 7 senders — one per outgoing link. Perfect spread, zero
same-link contention, derived from topology.

**Our comm pool is over-provisioned by roughly 8×**, which is a plausible direct
cause of the +0.815 ms memory-contention term.

---

## 4. The synchronization cookbook

This is the most transferable artifact. Every technique below is from a top-4
kernel and maps onto an existing header in `include/`.

### 4.1 Push coherence into the allocator, not into fences

The cleanest lesson in the dataset is the gemm-rs rank03 → rank02 diff (same
author, 475 → 443 µs):

| | rank03 | rank02 |
|---|---|---|
| IPC slab | `hipMalloc` (cached) | `hipExtMallocWithFlags(..., hipDeviceMallocUncached)` |
| flag store | `__ATOMIC_RELEASE` | `__ATOMIC_RELAXED` |
| `__threadfence_system` | present | **commented out** |

Allocating the symmetric heap uncached made every ordering annotation
unnecessary. **Four of the top kernels have zero `__threadfence_system` on the
fast path.** ag-gemm rank02 has no release-ordered store at all between payload
and flag; ordering rests entirely on the uncached MTYPE plus a read-back
throttle.

Relevance: this attacks `sync.cuh` directly. If our peer-write path emits system
fences per tile, an uncached heap may delete them outright. ag-gemm rank02's
author *measured* the cost of the GEMM reading A uncached rather than assuming
it (`custom_kernel_sync`, timing the IPC buffer against a normal-memory clone).

### 4.2 Epoch by monotone counter — never reset a flag

Every top kernel solves flag reuse without clearing, ping-pong buffers, or an
inter-iteration barrier:

- **Host-supplied epoch as a kernel argument.** `next_signal++` per launch;
  both sides compare `== next_signal`. 64 KB of flag state never cleared.
  (ag-gemm rank02, gemm-rs rank02)
- **Monotone offset bump.** `offset += 8` per barrier into a 1M-int array —
  ~125k barriers of headroom, no reset, no ABA. (gemm-rs rank01)
- **Consumer self-clearing.** Consumer nontemporal-stores 0 back after its spin
  exits. (all2all rank03)

all2all rank02 replaced a ping-pong barrier measured at **40–90 µs** with a
monotone sequence number at **~1 µs**. This is the single highest
value-per-line-changed item in the set, and it maps onto `completion.cuh`.

### 4.3 Asymmetric spin scopes

From all2all rank03, with the expensive version left commented directly above:

```cpp
// submissions/all2all/rank03_user322_sub64783.py:185-200
template <int SLEEP = 0, bool RESET_FLAG = true>
__device__ int spin_lock_system(int *addr) {
  int flag = 0;
  //while ((flag = __hip_atomic_load(addr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM)) == 0) {
  while ((flag = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) == 0) {
    if constexpr (SLEEP > 0) __builtin_amdgcn_s_sleep(SLEEP);
  }
  asm volatile("buffer_inv sc1;  invalidate cache outside of for loop");
  if constexpr (RESET_FLAG) __builtin_nontemporal_store(0, addr);
  return flag;
}
```

Writer pays `RELEASE`/`SYSTEM`. Reader spins with the cheapest possible
`RELAXED`/`AGENT` load and pays for coherence **exactly once**, after the loop
exits, with a single `buffer_inv sc1`.

### 4.4 Wave execmask reconvergence is a free counted fan-in

gemm-rs rank02 needs 8-way fan-in per tile. Instead of an atomic counter:

```cpp
// lanes 0-15 map to (8 peers × 2 m-tile rows)
if (lane_id < WORLD_SIZE * 2) {
    while (__hip_atomic_load(&signal_arr[...], __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_SYSTEM) != signal_val) { }
}
```

The wave cannot proceed until all 16 pollers exit. **8-way counted fan-in with
zero atomics and zero LDS counters.** ag-gemm rank02 uses the same trick with
lanes 0–3 for a 4-way AND-wait.

This directly replaces `counter.cuh`'s counted fan-in on the critical path — and
recall our own exp_14 finding that 2 of 4 atomics were redundant at g=1.

### 4.5 Poll cheaply, and back off hard

- Poll with a plain uncached load (`.cv`), **not** an atomic. ag-gemm rank01
  has the `tl.atomic_add(flag, 0, sem="acquire")` alternative sitting dead.
  Hundreds of CTAs polling with atomics serialize on an L2 bank.
- Back off aggressively: `s_sleep 1024 × SLEEP_CYCLE` where `SLEEP_CYCLE` is
  200–500 and is *the most-tuned constant* between ag-gemm rank01 and rank03 —
  it is the only meaningful difference between the #1 and #3 submissions.
- Keep the poller count tiny: 7 threads device-wide (gemm-rs rank01), 32–64
  lanes (gemm-rs rank02), one wave per CTA with the other three parked in
  `s_barrier` (ag-gemm rank02).
- Poll **local** memory where possible — ag-gemm rank02's consumer never
  touches the network to check arrival.

Our +0.815 ms "communication traffic contending for memory" may be as much
*pollers* as senders. This is directly testable.

### 4.6 Make dynamic fan-in static with `NOOP` padding

The reason a combine needs a barrier is that a consumer cannot know how much it
will receive. all2all rank05 removes that unknown entirely. Every combiner block
expects exactly `world × max_num_tokens` signals **always** — a launch-time
constant, independent of the data. Eight notifier blocks then fill each peer's
unused signal slots with an explicit "nothing for you" opcode:

```cpp
// submissions/all2all/rank05_user127_sub59248.py:615-618
const auto signal = pack_signal({}, 0, OPCODE::NOOP);
for (int i = threadIdx.x + tc; i < max_num_tokens; i += threads) {
    atomicExch_system(signals + i, signal);
}
```

Combined with two other ideas from the same kernel, this removes every barrier
from the dispatch→combine sequence:

- **`atomicExch`-to-zero as a *claim*, not a read.** A signal is single-
  consumption, so any of 911 CTAs can serve any arriving tile with no ownership
  arbitration.
- **Metadata packed into the doorbell word itself.** `pack_signal` puts the fp16
  combine weight, token id, and opcode in one dword — versus rank04, which
  ships a parallel `META_DIM=4` metadata array the consumer must re-read.

### 4.7 One poller per GPU beats sixty thousand

The two all2all designs disagree completely on polling, and the measurement is
unambiguous. rank05 spins across 911 blocks × 64 lanes on IPC-mapped memory with
no backoff. rank04 uses a `<<<1,1>>>` barrier kernel — **one lane per GPU** —
spinning on a monotone sequence counter in *host-pinned* memory with
`s_sleep(64)` backoff, and it is **faster** (326 vs 353 µs).

rank04's author put the flag in host memory deliberately: *"we need it to be on
the CPU side so that there is no coherency issues between GPUs (need correct
fine-grained atomic operations)"*. The whole file contains **no `__threadfence`
of any kind** — ordering comes from kernel-launch boundaries plus system-scope
atomics.

rank05's 58k naked pollers are a plausible independent instance of the
request-level contention we identified in `exp_19`.

**Barrier-fused snapshot** (rank04): a count that only becomes valid at the
barrier lives in uncached host memory, but ~16k blocks need it. The barrier
kernel copies it into device memory the instant it releases, so the next kernel
does one PCIe read instead of 16k. A good candidate primitive.

### 4.8 Peer base addresses as compile-time constants

ag-gemm/gemm-rs rank01 pass peer heap bases as `tl.constexpr` int64 element
offsets, so peer address arithmetic constant-folds into the ISA — no table load,
no dependent load, no register pressure in the epilogue. gemm-rs rank05 loads
from a runtime `heap_bases` tensor and is slower. Relevant to `pgl.cuh`/`peer.cuh`.

### 4.7 Do not signal per tile if you can signal once

**The gemm-rs winner publishes no per-tile readiness at all.** Correctness comes
from disjoint address ranges — each rank writes only its own slab of the peer's
heap, so no two ranks ever touch the same address — and ordering comes from a
single barrier kernel with **7 active threads**:

```cpp
// submissions/gemm-rs/rank01_user155_sub62633.py:456-466
__global__ void dist_barrier(IpcCommBlock **d_remote_data, int index, int offset) {
  int dst_rank = threadIdx.x / 64;
  if (threadIdx.x % 64 != 0 || dst_rank == index) { return; }
  store_volatile(&d_remote_data[dst_rank]->barrier[offset + index], 1);
  while (true) {
    if (load_volatile(&d_remote_data[index]->barrier[offset + dst_rank]) == 1) break;
  }
}
```

**The submission that implements our design — flag per tile, CAS release/acquire,
system scope — is rank 5, 23% slower** (508 vs 413 µs). It even has a feature
the winner lacks (per-rank destination rotation, so each rank writes to itself
first) and still loses. Its author's own comment on the producer spin reads
`# not sure why need to spin here...`.

The winner's bet: at 256×256 tiles with a 1024-CTA grid, one barrier costs less
than 1024 tile-level acquire/release pairs plus a spinning reducer. The
measurement supports it. **Per-tile signalling adds to the memory-contention
term rather than hiding it.**

---

## 5. The one design worth copying wholesale

**ag-gemm rank02 (382 µs)** is the most sophisticated kernel in the set and the
one that most directly challenges our framing.

Its answer to "how do you overlap the all-gather with the GEMM" is: *make the
collective's landing buffer be the GEMM's A operand, in uncached IPC memory, and
gate the GEMM's K-loop on per-(m-tile, k-chunk) arrival.*

```cpp
for (int tile_k_id = 1; tile_k_id < (num_tile_k - 1); ++tile_k_id) {
    block_sync_lds();
    load_lds();                                  // VGPR -> LDS
    if ((tile_k_id + 1) % exact_div<COMM_K, BK>() == 0)
        wait_signal(tile_m_id, tile_k_id + 1);   // gate on next 128-wide K slice
    load_vgpr((tile_k_id + 1) * BK);             // prefetch
    frags_mfma();                                // MFMA on k-1
```

`COMM_K=128`, `BK=64`, so the gate fires every 2 K-tiles. MFMAs on slice *k−1*
issue while slice *k+1* is still crossing xGMI. The GEMM starts producing MFMAs
after the **first 128-element K slice of one M-tile** has landed — not after the
gather completes.

What makes it work is that the producer's emission order is arranged to match:
each sender warp owns a fixed `(dst_rank, chunk_m)` and walks `chunk_k`
monotonically, so all destinations and all M-chunks for k=0 go out before any
k=1 — K-major emission feeding a K-major consumer.

The 32 comm CTAs are not the overlap mechanism. They are a pump sized to the
minimum the copy needs. **The overlap lives in the K dimension.** Our taxonomy
asks *who*; this kernel's answer is *when*, and the CTA split falls out as a
consequence.

Cautionary note, and a good one: this kernel ships with a dead XCD swizzle (line
665 computes `tile_id1`, line 670 overwrites it), a consume-signal channel
written by nobody's reader, a producer read-back that names the wrong array, and
two entire dead GEMM implementations — and still took #2. The performance is in
the K-granular fusion and the fence-free uncached heap, not the scheduling polish.

---

## 6. Recommendations

Ordered by expected value per unit of effort.

**Do first — cheap, mechanical, independent of overlap strategy**

1. **Allocate the symmetric heap `hipDeviceMallocUncached` and delete the
   per-tile fences.** Measure the cost of uncached reads separately, as ag-gemm
   rank02's author did. Touches `sync.cuh`. This is the highest
   confidence/effort ratio in the report.
2. **Replace flag clearing with a host-supplied monotone epoch counter.**
   Removes our reset path and any inter-iteration barrier. Touches
   `completion.cuh`. Precedent says 40–90 µs → ~1 µs on the barrier alone.
3. **Switch counted fan-in from atomics to wave execmask reconvergence** where
   fan-in ≤ 64. Touches `counter.cuh`. Follows our own exp_14 result.
4. **Make polls uncached non-atomic loads, add `s_sleep` backoff, and cut the
   poller count hard.** Then re-run the M7 interference measurement — a direct
   test of whether the +0.815 ms is pollers rather than payload. The two all2all
   designs bracket this cleanly: 58k naked pollers (353 µs) versus one sleeping
   poller per GPU (326 µs).
4b. **Convert data-dependent fan-in into a launch-time constant with `NOOP`
   padding**, and make doorbell claims single-consumption via `atomicExch`-to-
   zero so any service CTA can take any tile without arbitration. Together these
   remove the reason a combine needs a barrier at all. Touches `completion.cuh`
   and `roles.cuh`.
4c. **Pack metadata into the doorbell word** rather than shipping a parallel
   metadata array the consumer must re-read.
5. **Hoist peer base addresses to compile-time constants.** Touches `pgl.cuh`.
6. **Vectorize peer transfers to 16 B.** The all2all winner uses scalar 2-byte
   `__half` accesses throughout; our tile loads should beat its transfer
   efficiency by 8× per instruction on identical traffic.

**Then reconsider the overlap model**

7. **Cut the comm pool from 64/256 to ~8–32, and size it to topology.** Use one
   CTA per XCD per peer link (8 per peer) rather than a time balance.
   Compensate with load-issue depth — 4 pipeline stages × 8 peers of 16 B
   loads — rather than CU count.
8. **Prefer pull over push for the reducer.** If compute CTAs never touch a
   remote pointer, comm and compute never contend for the same *direction* of
   memory traffic. This is a candidate mechanism for the +0.815 ms.
9. **Try dependency-ordered tile scheduling before trying harder role
   switching.** Place tiles needing no remote data at the front of pid space so
   they saturate MFMA from cycle 0. Costs zero CTAs. In the ag-gemm winner this,
   not the comm pool, is where the hiding comes from.
10. **Evaluate K-granular consumer gating** (ag-gemm rank02) as an alternative
    to CTA specialization for the expert GEMMs. This is the one design in the
    set that is strictly more sophisticated than ours.

**Do not build**

11. **Completion-order *ticketing*.** Zero of 17 kernels use it. Before building
    the ticket machinery, test the one-line static approximation: let producer
    CTAs fall through into the consumer pool on a compile-time schedule
    (all2all rank05 puts the combiner loop *outside* the role `if`, so 136
    producer CTAs become combiners the instant they drain and the other 775
    were combiners all along). This is our `exp_12` fall-through, and it is the
    cheap version of category 3. Under dependency-ordered tile scheduling even
    that has little to do: a tile that finishes early simply exits, and the next
    tile in the dispatch queue is already the right one to run.
12. **Copy-engine / SDMA peer DMA.** Built and abandoned by two independent
    top-3 authors.
13. **Per-tile CAS signalling for GEMM→RS.** Implemented by gemm-rs rank05 and
    23% slower than one barrier.

**Hard invariant** for any persistent-role abstraction: grid == exactly the CU
count (304) with `__launch_bounds__` guaranteeing co-residency. That is what
makes an in-kernel producer/consumer spin deadlock-free. Both fused megakernels
depend on it.

---

## 7. Open questions this analysis does not answer

- **Everything about overlapping a real expert GEMM.** No all2all submission
  contains one. The regime where we measured +1.249 ms is untouched by this
  dataset. The transferable findings are about the *communication schedule*.
- **Whether uncached heaps stay a win when the payload is re-read by MFMA.**
  ag-gemm rank02 measured it for its A operand; our M7 weights have a different
  reuse pattern.
- **Whether K-granular gating survives the MoE dependency structure**, where the
  reduction is over experts rather than over K.

---

## Appendix: file map

| Problem | Rank | Score | User | Design |
|---|---|---|---|---|
| all2all | 1 | 199 µs | u417 | IPC + `/dev/shm` CPU barriers; **no overlap**; eager metadata, sparse payload |
| all2all | 2 | 241 µs | u258 | push dispatch / **pull** combine, sender-memoized dest index, 1 device barrier |
| all2all | 3 | 291 µs | u322 | sender-private slots, per-token self-clearing flags, **no barrier anywhere** |
| all2all | 4 | 326 µs | u401 | vendored `muillm`; **one-lane barrier on host-pinned memory**; no fences at all |
| all2all | 5 | 353 µs | u127 | fused megakernel; `NOOP` padding + claim-by-exchange; **zero barriers**; role fall-through |
| gemm-rs | 1 | 413 µs | u155 | Triton GEMM, epilogue push to owner, **one 7-thread barrier**, local reduce |
| gemm-rs | 2 | 443 µs | u258 | persistent 304-CTA megakernel, 8–16 comm CTAs, pull + 32 in-flight loads |
| gemm-rs | 5 | 508 µs | u322 | **our design**: per-tile CAS release/acquire — 23% slower |
| gemm-rs | 10 | 520 µs | u461 | naive SIMT GEMM + RCCL, zero overlap — within 17% of #2 |
| ag-gemm | 1 | 373 µs | u155 | 56 comm CTAs (8/peer = 1/XCD/link), per-chunk fan-in, dependency-ordered tiles |
| ag-gemm | 2 | 382 µs | u258 | **K-granular consumer gating**; landing buffer *is* A; fence-free uncached heap |

Near-duplicate lineages (same code, different accounts): ag-gemm rank01≡rank03,
ag-gemm rank02≡rank04 (two-line diff), gemm-rs rank02≈rank03.

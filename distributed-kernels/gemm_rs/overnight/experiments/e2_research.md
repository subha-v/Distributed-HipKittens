# E2 research — XGMI egress ceiling and peer-store mechanics on MI300X (gfx942)

Research subagent, no GPU jobs run, no sources edited. Every claim carries its URL.
Written incrementally; sections appear in the order they were established.

**Question being answered:** producers emit 16-byte packets (`global_store_dwordx4`,
one per lane) directly into peer symmetric heaps from the GEMM epilogue. On shape 6
(8192x8192x29568) ~117 MB moves off-rank in 920 us == ~127 GB/s per GPU. What is the
real ceiling, and what closes the gap?

---

## 1. The real XGMI number for MI300X

### 1.1 Topology — this is the part the working premise had wrong

**MI300X in an 8-GPU UBB node is a fully-connected mesh with exactly ONE xGMI link
per peer.** Each GPU has **seven** Infinity Fabric links, one dedicated to each of the
seven peers. There is no link aggregation between any pair — no dual/quad-link pairs,
no 2-hop peers.

- "Each discrete MI300X offers a 16-lane PCIe Gen 5 host interface and **seven AMD
  Infinity Fabric links** for full connectivity between eight GPUs"
  — AMD MI300X data sheet,
  https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
  (also lists the per-GPU link table as `7x 128 GB/s`, i.e. 7 links x 128 GB/s bidir)
- "Each AMD Instinct MI300X system GPU is connected to its seven peer GPUs via xGMI
  links, forming a fully connected mesh"
  — AMD ROCm blog, *Understanding xGMI and RCCL bandwidth on MI300X*,
  https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html

> **Do not use arXiv:2410.00801 for MI300X per-link numbers.** That paper
> (https://arxiv.org/html/2410.00801v1) measures **MI250X**, which is built from
> *GCDs* wired with single/dual/**quad** 50+50 GB/s links and 2-hop peers. Its
> "quad link = 400 GB/s bidirectional" is an MI250X fact and does not transfer.
> It is still useful for one methodology point, recorded in section 1.4.

### 1.2 Per-link and per-GPU numbers

| quantity | theoretical peak | achievable (AMD-published measurement) |
|---|---|---|
| one xGMI link, **per direction** | **64 GB/s** | **45 - 48 GB/s** (~75% of peak) |
| one xGMI link, bidirectional | 128 GB/s | — |
| **aggregate unidirectional per GPU** | **448 GB/s** (7 x 64) | **315 - 336 GB/s** (7 x 45-48) |

Source for the whole table, verbatim from AMD's own blog:
https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html

> "xGMI's raw bandwidth is rated up to 64 GB/s for each point-to-point link between
> GPUs. ... in reality, this bandwidth is constrained by factors like CRC bit
> correction and protocol overhead, which reduce the usable bandwidth to
> approximately **48 GB/s per link**. This represents about 75% of the theoretical
> peak."

**The 448 GB/s figure in the working premise is the theoretical roofline (7 x 64),
not an achievable one.** The achievable per-GPU egress ceiling is **~316-336 GB/s**.
The ~25% haircut is protocol + CRC and is not recoverable by software.

### 1.3 The kernel-issued P2P store figure (the high-value item)

**AMD's published 45-48 GB/s per link IS a kernel-issued figure, not SDMA.** The blog's
measurement is TransferBench `a2a 64M 8`, and its own printed configuration says:

```
USE_DMA_EXEC   = 0 : Using GFX executor        <-- kernel, NOT SDMA
USE_FINE_GRAIN = 1 : Using fine-grained memory
USE_REMOTE_READ= 0 : Using SRC as executor    <-- the *sender* executes: push/store, like us
GFX_UNROLL     = 2
NUM_SUB_EXEC   = 8 : Using 8 subexecutors/CUs per Transfer
BLOCK_BYTES    = 256
```

Measured result, 56 concurrent transfers (every GPU pushing to all 7 peers at once —
**exactly our reduce-scatter traffic pattern**):

```
Executor: GPU 00 | 315.552 GB/s | 1.489 ms | 469762048 bytes | 326.317 GB/s (sum)
  Transfer 00 | 47.056 GB/s | F0 -> G000:008 -> F1
  ...
  Transfer 06 | 45.210 GB/s | F0 -> G000:008 -> F7    <-- slowest link
```

Per-GPU egress of **315-318 GB/s** sustained, all 8 GPUs simultaneously, all 7 links
live in both directions at once. Same URL as above.

Three conclusions that matter more than the headline number:

1. **The links are effectively full-duplex under our pattern.** In `a2a` every link
   carries F_i -> F_j *and* F_j -> F_i at the same time and each direction still
   reports ~47 GB/s. So concurrent all-peer egress does not halve the per-link rate,
   and the 7 x 47 budget is legitimately available to a reduce-scatter.
2. **A GFX (kernel) executor reaches the same 47 GB/s that AMD quotes as the
   practical link ceiling.** There is no separate, higher "SDMA-only" tier to chase:
   kernel-issued push stores are a first-class way to saturate xGMI. Our approach is
   architecturally sound; only its efficiency is in question.
3. **It took only 8 CUs per link — 56 CUs total — to get there.** This is the single
   most diagnostic number in this report. See section 1.5.

### 1.4 Cross-checks and one contrary datapoint

- Independent 8-GPU MI300X RCCL AllReduce measurement peaks at **433 GB/s algbw /
  3468 GB/s busbw** at 1 GB, consistent with a healthy full mesh:
  https://github.com/namanadep/mi300x-amd-rocm-validation
- AMD's Cluster Validation Framework states a *healthy* 8-GPU MI300X SPX node
  "typically delivers ~250-300 GB/s on the AllReduce benchmark", i.e. real
  collectives land below the 316-336 GB/s point-to-point ceiling:
  https://rocm.blogs.amd.com/software-tools-optimization/cvf/README.html
- arXiv:2510.27583 (*AMD MI300X GPU Performance Analysis*) reports all-reduce scaling
  "64 GB/s (2 GPUs) to 192 GB/s (4 GPUs) and 448 GB/s (8 GPUs), reaching about 70% of
  theoretical peak": https://arxiv.org/pdf/2510.27583 — note this 448 is a
  *node-aggregate collective* number and is numerically coincidental with the per-GPU
  7x64 roofline. Do not conflate the two.
- **hipMemcpyPeer under-utilises links** where multiple links exist (MI250X evidence,
  75% / 50% / 25% utilisation for single / dual / quad links):
  https://arxiv.org/html/2410.00801v1 — on MI300X every peer is single-link, so this
  particular pathology mostly disappears, but it reinforces that a well-written kernel
  is not at a disadvantage versus `hipMemcpyPeer`.

### 1.5 Verdict on 127 GB/s

**127 GB/s is bad — roughly 39% of the achievable ceiling, i.e. ~2.5x of headroom.**
Not the 3.5x the premise assumed (that was measured against the unreachable 448 GB/s
roofline), but 2.5x is still the largest recoverable pool identified in this report.

Arithmetic for the 920 us egress pool on shape 6, 117 MB off-rank:

| scenario | per-GPU egress | egress time | delta vs today |
|---|---|---|---|
| today | 127 GB/s | **920 us** | — |
| theoretical roofline (unreachable) | 448 GB/s | 261 us | -659 us |
| **AMD-measured kernel-issued ceiling** | **~329 GB/s** (7 x 47) | **~356 us** | **-564 us** |
| 75% of that ceiling (realistic target) | ~247 GB/s | ~474 us | -446 us |
| 60% of that ceiling (conservative) | ~197 GB/s | ~594 us | -326 us |

On a 2861.7 us total for shape 6 that is **-11% to -20% end-to-end** from the egress
axis alone, assuming egress is on the critical path rather than hidden under compute.

**The damning comparison:** TransferBench needed **8 CUs per link (56 CUs)** to reach
329 GB/s. We are spending on the order of 272 producer CTAs at 1 CTA/CU and reaching
127 GB/s. We are therefore **not short of parallelism or of packets in flight** — by
roughly 5x more issuing CUs than the reference needs — which means the deficit is
almost certainly **per-transaction efficiency** (coalescing width and fabric
transaction count), not concurrency. That single inference should re-rank the E2
experiment list: *widen and coalesce before adding more in-flight packets.*

---

## 2. Store width and coalescing on CDNA3

### 2.1 Is `global_store_dwordx4` the widest per-lane store? YES.

The complete gfx942 `global_store` family, from the LLVM gfx942 instruction syntax
reference (https://www.llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html):

```
global_store_byte / _byte_d16_hi / _short / _short_d16_hi
global_store_dword / _dwordx2 / _dwordx3 / _dwordx4      <-- dwordx4 = 16 B is the max
```

**There is nothing wider than `dwordx4` (128 bit / 16 B per lane) on gfx942.** No
`dwordx8`. So 1024 B per 64-lane wavefront is the hardware maximum per instruction, and
we are already issuing the widest store available. Note every `global_store_*` accepts
the `sc0 nt sc1` modifiers (visible in the syntax listing above), which matters for
section 3.

AMD-aligned guidance confirms 16 B is the target width, and adds the *contiguity*
requirement: "The optimal memory access size is 16 B or 128 bits, using the
`global_load_dwordx4` and `global_store_dwordx4`. Further, **make sure that the memory
access is subgroup-contiguous**, such that the whole subgroup accesses 512 B at once."
— https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md

So width is already right. **Width was never the problem. Contiguity is.**

### 2.2 The transaction granularity that decides everything

This is the load-bearing fact of the whole report, from AMD's own profiler
documentation (https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-7.2.3/conceptual/l2-cache.html):

> "Requests from the L2 Cache are broken down into two major categories, read requests
> and write requests. ... these requests may be sent across Infinity Fabric as
> **different transaction sizes, 32B or 64B on current CDNA accelerators**."
>
> "a request is either a 32B Write Request OR a 64B Write request, as the flow splits
> at this point."

Same page confirms remote destinations ride the same path: "The backing memory for a
request may be local to this accelerator ... **in a remote accelerator's memory**, or
even in the CPU's memory. Infinity Fabric is responsible for routing these memory
requests."

Supporting cache-line geometry (L1/L2 line 128 B; **MALL/Infinity-Cache line 64 B**):
https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md
and the MI300X Hot Chips 2024 deck (L1D 128 B line; L2 write-back/write-allocate,
"Increase request coalescing", agent-scope coherent; Infinity Cache device/system-scope
coherent): https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf

**Therefore the minimum unit that crosses xGMI is a 32 B transaction.** A 16 B packet
cannot be smaller than one fabric transaction, so:

| lane address pattern across the wave | bytes moved on fabric per 16 B of payload | efficiency |
|---|---|---|
| **contiguous** 16 B/lane -> 1024 B/wave, 64 B-aligned | 16 B (16 payloads share one 64 B txn) | **~100%** |
| scattered, each packet alone in a **32 B** txn | 32 B | **50%** |
| scattered, each packet alone in a **64 B** txn | 64 B | **25%** |

### 2.3 The quantitative match — this is the smoking gun

Measured 127 GB/s against the AMD-measured kernel-issued ceiling of ~329 GB/s is
**38.6% efficiency**. That number sits squarely inside the 25%-50% band predicted for
16 B packets that each occupy their own fabric transaction, and nowhere near the ~100%
of a contiguous pattern.

**Conclusion: the 920 us egress pool is most likely dominated by write amplification of
roughly 2.2x-2.6x — we are pushing ~2.5x more bytes across xGMI than the payload
requires, because scattered 16 B stores cannot fill a 32 B or 64 B fabric
transaction.** This is a mechanism-level explanation that predicts the observed number
without any free parameters, which is why it out-ranks everything else in section 4.

Corroborating the general penalty scale, AMD's HIP hardware documentation
(https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html):

> "**Coalesced access pattern**: When consecutive threads access consecutive memory
> addresses, the hardware can combine all 64 thread requests into as few as 4-8 cache
> line requests. **Non-coalesced access pattern**: When threads access widely separated
> addresses, each thread can generate a separate memory transaction, reducing effective
> bandwidth by up to 16x or more." ... "Coalesced access: Can achieve 70-90% of peak
> bandwidth. Random access: Might achieve only 5-15% of peak bandwidth."

(That page also documents the VMEM unit as the coalescing point: "The coalescing
hardware in the VMEM unit analyzes addresses from all threads in a warp and groups them
into the minimum number of cache line requests." Coalescing is *address-driven*, done in
hardware — we cannot ask for it, we can only make the addresses deserve it.)

### 2.4 What the address pattern must look like

To get full-rate egress the 64 lanes of a wave must write a **single contiguous,
64 B-aligned (prefer 128 B-aligned) 1024 B run** of the peer's heap: lane L writes
`base + 16*L`. Anything that makes lane stride != 16 B breaks a fabric transaction into
partial writes. TransferBench's GFX executor — the thing that actually achieves 47 GB/s
per link — does exactly this, with `BLOCK_BYTES = 256` per CU and `GFX_UNROLL = 2`
(configuration dump in section 1.3).

### 2.5 Outstanding stores per CU — is "more packets in flight" even available?

- **`vmcnt` is 6 bits on gfx9/gfx942: 0..63 outstanding vector memory operations per
  wave.** Encoding `VM_CNT` in bits `15:14, 3:0`, value range `0..63`:
  https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/gfx940_waitcnt.html
  (LLVM clamps to `std::min(63u, vmcnt)` for major version 9:
  https://github.com/llvm/llvm-project — `s_waitcnt` wrapper, chipset major 9 path)
- So one wave can have up to 63 stores in flight; with several waves per CU the
  per-CU in-flight capacity is far beyond what 16 B packets need to saturate a link.
- AMD documents the load/store units as maintaining deep queues: "LSUs manage thousands
  of outstanding memory requests per GPU, dynamically scheduling them to hide memory
  latency ... the LSUs maintain queues of pending transactions"
  (https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html).
- **No citable per-CU remote-request or xGMI credit limit was found.** No citable source
  found for a specific fabric credit depth on gfx942.

**Judgement: "more packets in flight" is NOT our lever.** TransferBench saturates a link
with 8 CUs; we are issuing from ~272 CTAs and reaching 39%. The in-flight budget is
already oversubscribed relative to what the fabric needs. Adding concurrency to a
transaction-size-limited path cannot help and may hurt (more partial-write traffic
competing for the same fabric transactions). Deprioritise the "packets in flight" point
of E2 in favour of the coalescing point.

### 2.6 Free diagnostic — settle this before writing any kernel code

rocprofiler-compute exposes the exact counters to confirm or kill section 2.3 in a
single profiling run, with **no kernel change at all**
(https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-7.2.3/conceptual/l2-cache.html):

- 32B vs 64B **Write Request** counts (the flow splits, so they are exclusive)
- **Remote Write and Atomic Traffic** (% of writes routed off-device — isolates our
  peer traffic from local HBM traffic)
- **Write and Atomic BW** (bytes over Infinity Fabric / duration)
- **Write and Atomic Latency** (cycles in fabric before completion ack)
- **UC Req** / **Uncached Write and Atomic Traffic** (tells us whether the peer heap is
  being treated as uncached/fine-grained, which feeds section 3)

Compute `write_amplification = (32*N32 + 64*N64) / payload_bytes`. If that lands near
2.5, section 2.3 is proven and intervention I1 is the night's work. If it lands near
1.0, the coalescing hypothesis is dead and the bottleneck is elsewhere (latency /
release serialization -> look at I2 and E3).

---

## 3. Cache-policy bits for peer stores on CDNA3

### 3.1 Bit meanings on gfx942

Encoding, from the Triton AMD backend's CDNA3/CDNA4 mapping comment (which cites the
MI300 CDNA3 ISA doc):
`bit 0 = sc0, bit 1 = nt, bit 3 = swz, bit 4 = sc1`
— https://github.com/triton-lang/triton/commit/a357e1a77bbeb16b5b179942b7d8707e4d2ff985

> "Vector Memory instructions (Flat, Global, Scratch, and Buffer) have 3 bits to control
> scope and cacheability: **SC[1:0] System Cache level: 0=wave, 1=group, 2=device,
> 3=system**. NT: Non-Temporal."

Primary reference for these tables is the CDNA3 ISA guide, Table 48 "Load Controls" and
section 9.1.10.2 for `nt`:
https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
(cited for exactly these facts by Triton PRs
https://github.com/triton-lang/triton/pull/4337 and
https://github.com/triton-lang/triton/pull/5338)

### 3.2 Which combination a peer store requires

LLVM's `SIMemoryLegalizer` is the authoritative, executable statement of the gfx940/942
memory model
(https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp,
introduced in https://github.com/llvm/llvm-project/commit/47bac63d3f6b9e64fdf997aff1f145bc948f02d9,
documented in https://reviews.llvm.org/D121397):

```cpp
case SIAtomicScope::SYSTEM:
  if (ST.hasGFX940Insts()) {          // Set SC bits to indicate system scope.
    enableCPolBits(MI, CPol::SC0 | CPol::SC1);
case SIAtomicScope::AGENT:
  if (ST.hasGFX940Insts()) {          // Set SC bits to indicate agent scope.
    enableCPolBits(MI, CPol::SC1);
```

And for the release fence:

```
store atomic release - agent  - global : buffer_wbl2 sc1=1
store atomic release - system - global : buffer_wbl2 sc0=1 sc1=1
```

**A peer GPU is a different HSA agent.** Therefore:

- **Peer-destined data stores must carry `sc0 sc1` (system scope).** `sc1` alone is
  *agent* scope — visible across XCDs of *this* GPU only. An `sc1`-only or bare peer
  store can sit dirty in the local XCD L2 (write-back, write-allocate, "agent scope
  coherent" per the Hot Chips deck) and not cross xGMI until something flushes it.
- **Our existing `buffer_wbl2 sc0 sc1` release is correct** — it is precisely the
  system-scope release LLVM emits. Keep it.

### 3.3 The interaction to check first, and a wait-placement bug to look for

Two precise consequences follow, and both are cheap to check by grepping the existing
ISA dump rather than by running anything:

**(a) If the peer *data* stores are not already `sc0 sc1`,** then they are being
absorbed by the local L2 and the bytes only leave the die in a burst when
`buffer_wbl2 sc0 sc1` runs. That would (i) serialize egress against the release instead
of streaming it under compute, and (ii) pollute the 4 MB XCD L2 with write-once
peer-destined data, taxing the 46% / 1318 us GEMM mainloop pool as collateral damage.
**Grep the ISA for the sc bits on the peer store instruction. This is the single
cheapest high-information action available and it re-ranks everything else.**

**(b) LLVM documents the required wait placement around `buffer_wbl2`, and it is the
opposite of the intuitive one:**

> "Inserting a `S_WAITCNT vmcnt(0)` **before** is **not required** because the hardware
> does not reorder memory operations by the same wave with respect to a following
> `BUFFER_WBL2`. The `BUFFER_WBL2` is guaranteed to initiate writeback of any dirty
> cache lines of earlier writes by the same wave. A `S_WAITCNT vmcnt(0)` is needed
> **after** to ensure the writeback has completed."
> — SIMemoryLegalizer.cpp, same URL

So the correct sequence is `buffer_wbl2 sc0 sc1` **then** `s_waitcnt vmcnt(0)`. If
`producer_drain_release` currently drains with `vmcnt(0)` *before* the writeback as well
as after, **that leading drain is provably unnecessary on this architecture** and is
pure cost paid once per tile across the 250 us E3 pool. Worth a grep.

### 3.4 Does `nt` / `__builtin_nontemporal_store` help peer traffic?

`nt` is an *eviction-policy* hint, not a transaction-size control. AMD guidance: "For
data that is 'streamed' and does not need to be cached, consider using *non-temporal*
loads/stores. **This disables coherency and invalidates cache entries.**"
— https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md

Peer packets are the textbook case for `nt`: written once, never re-read locally.
Expect it to **help modestly, and mostly by protecting L2 for the GEMM rather than by
speeding up the fabric** — it cannot change the 32 B/64 B transaction size, which
section 2.3 argues is the dominant term.

**Three traps, all citable:**

1. **`__builtin_nontemporal_store` sets `nt` but does NOT set `sc0`/`sc1`.** They are
   independent bits (bit 1 vs bits 0 and 4). Using the builtin for a peer store and
   assuming the result is peer-visible would be a **correctness bug**, not a perf
   regression. If you use it, you must still get system scope onto the same instruction.
2. **"Disables coherency" collides with a release protocol.** `nt` invalidating /
   bypassing cache entries interacts directly with what `buffer_wbl2 sc0 sc1` is
   supposed to write back. Do not assume `nt` + `wbl2` composes; this needs
   protocol-review.
3. **The compiler can silently drop `nt`.** Triton had to disable LLVM's
   `vector-combine` pass to preserve the flag
   (https://github.com/triton-lang/triton/pull/4337). **Verify `nt` in the ISA dump; do
   not trust the intrinsic.**

### 3.5 Fine-grained / uncached memory — a bandwidth caution for the heap allocation

Relevant because the symmetric heap is IPC-shared and may be fine-grained:

- "On MI-series GPUs (MI300X, etc.) **fine-grained memory is available by default**."
  — https://rocm.docs.amd.com/projects/rocm-xio/en/beta-0.1.0/conceptual/memory-modes.html
- "**Fine-grained**: Recommended for atomic flags, signals, and small synchronization
  variables. **Typically bypasses the L2 cache** or employs heavy coherency protocols to
  ensure immediate visibility, **resulting in lower bandwidth**. **Coarse-grained**:
  Recommended for bulk data transfers and large memory regions. Utilizes the full L2
  cache, providing high bandwidth for sequential access patterns."
  — https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/coherence_control.html
- On CDNA, fine-grained/uncached traffic is also what Infinity Fabric classifies as
  *atomic*: "requests are only considered atomic by Infinity Fabric if they are targeted
  at fine-grained memory allocations or uncached memory allocations"
  — https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-7.2.3/conceptual/l2-cache.html

**Implication:** the bulk payload heap and the flag/counter heap want *different*
allocation policies. Flags and epoch counters belong in fine-grained (or
`hipExtMallocUncached`) memory; the bulk tile payload wants the highest-bandwidth path
available. If both live in one uniformly fine-grained symmetric heap, the payload is
paying a synchronization-grade memory policy for bulk data. Check `UC Req` /
`Uncached Write and Atomic Traffic` in the section 2.6 profile to see whether this is
actually happening before acting on it.

---

## 4. Ranked interventions

Ranked by expected gain per unit of implementation risk. Baseline for all deltas is the
920 us egress pool on shape 6 (117 MB off-rank, 2861.7 us total).

### I0 (do first, costs nothing): measure write amplification, grep the ISA

Not really an intervention — it is the thing that decides whether I1 or I2 is the
night's work, and it requires no kernel edit.

1. Profile with rocprofiler-compute and compute
   `(32*N32 + 64*N64) / payload_bytes` from the 32B/64B Write Request counters, plus
   `Remote Write and Atomic Traffic` and `Write and Atomic Latency` (section 2.6).
2. Grep the existing ISA dump for the `sc0`/`sc1`/`nt` modifiers on the peer-store
   instruction, and for a redundant `vmcnt(0)` *preceding* `buffer_wbl2` (section 3.3b).

**Expected outcome:** amplification ~2.5 confirms I1 (and predicts ~-450 us);
amplification ~1.0 kills I1 and promotes I2/E3. **Correctness risk: none** (read-only).

### I1 (highest expected gain): make peer stores wave-contiguous via LDS staging

**Mechanism.** Keep `global_store_dwordx4` — it is already the widest store (2.1) — but
change the *addresses* so the 64 lanes of a wave cover one contiguous 64 B-aligned
1024 B run of the peer heap, instead of 64 independent 16 B packets. Stage the tile's
peer-destined contribution in LDS, then stream it out contiguously, which is exactly the
pattern TransferBench uses to hit 47 GB/s/link. This converts 1 fabric transaction per
16 B payload into 1 transaction per 64 B payload.

**Expected effect.** From 38.6% efficiency toward 75-90%: **920 us -> ~400-475 us, i.e.
-450 to -520 us** on shape 6, roughly **-16% to -18% end-to-end** if egress is on the
critical path. Even a partial fix (to 60% efficiency) is -326 us. This is the largest
single pool identified anywhere in this report.

**Correctness risk: MEDIUM, and it does touch the protocol.** It does not change *what*
bytes are written, only their grouping and timing, which is the good news. But it
inserts an LDS staging buffer between the accumulator and the release, so the release
must now cover the staged flush rather than individual stores, and the staging buffer
acquires an epoch lifetime of its own (it cannot be reused until its packets are
released). **Needs protocol-review before its first GPU run** — specifically the
release/acquire pair and the staging buffer's epoch lifetime versus the per-tile
credits. Interacts with E6 (`EB` / `win_rows`), which already governs the staging
window; land them together rather than separately.

### I2 (cheapest real fix, highest information): correct the scope bits on peer stores

**Mechanism.** Ensure the peer data store carries **`sc0 sc1` (system scope)** — a peer
GPU is a different agent, and `sc1` alone is agent scope (3.2). If the bits are wrong
today, peer bytes are sitting dirty in the local XCD L2 and only crossing xGMI in a
burst at `buffer_wbl2` time. Additionally drop the redundant pre-`wbl2` `vmcnt(0)` if
present (3.3b).

**Expected effect.** Bimodal, which is why I0 precedes it. If the bits are already
`sc0 sc1`: **zero**. If they are not: potentially large but hard to bound — it would
convert burst-at-release egress into streaming egress (recovering overlap with compute)
*and* relieve L2 pressure on the 1318 us mainloop pool. The redundant-drain removal is
independently worth a slice of the 250 us E3 pool. A few instruction-modifier characters
of edit either way.

**Correctness risk: HIGH.** This *is* the store-ordering and cache-policy axis. Getting
it wrong produces silent data corruption or a hang, not a slowdown. **Mandatory
protocol-review**, and re-run the full gate ladder at both `1e-2` and `2e-3` plus all
three negative controls. Never relax a scope bit to gain bandwidth.

### I3 (cheap, modest, do alongside I1): `nt` on the peer payload stores

**Mechanism.** Add `nt` to the peer payload stores (not to flags or counters). Peer
packets are write-once and never re-read locally — the canonical non-temporal case.

**Expected effect.** Small on egress directly: `nt` is an eviction hint and cannot
change fabric transaction size, which section 2.3 argues dominates. The real value is
**not polluting the 4 MB XCD L2 with write-once traffic**, which pays out in the 1318 us
GEMM mainloop pool rather than the 920 us egress pool. Call it low single-digit percent
overall, with most of the benefit landing on E1's axis rather than E2's.

**Correctness risk: MEDIUM-HIGH, and non-obvious.** Three specific traps from 3.4:
`__builtin_nontemporal_store` does **not** set the scope bits, so it cannot by itself
make a store peer-visible; `nt` "disables coherency" and therefore interacts with what
`buffer_wbl2 sc0 sc1` writes back; and LLVM's `vector-combine` can silently drop the
flag, so it must be verified in the ISA dump. **Needs protocol-review**, and must never
be applied to the flag/counter stores.

### I4 (nearly free, small, worth landing): rotate the destination sweep by rank

**Verdict: yes, do it — it is a strict improvement and costs almost nothing — but expect
single-digit percent, not a fix. Do not let it displace I1.**

**The reasoning matters more than the number, because the obvious intuition is wrong.**
MI300X gives each GPU **seven independent point-to-point links, one per peer** (1.1).
There is no shared switch and no shared trunk, so "all 8 GPUs hammering the same
destination" does *not* oversubscribe a shared resource the way it would on a fat-tree:
the destination receives on 7 *separate* ingress links, and its HBM (5.3 TB/s) absorbs
329 GB/s without noticing.

Working the two cases in the fully destination-phased limit:

| sweep | phase-0 traffic | senders active | node egress |
|---|---|---|---|
| unrotated (every rank starts at peer 0) | all ranks -> rank 0 | 7 (rank 0 idles) | 7 x 47 = 329 GB/s |
| rotated (rank r -> peer (r+k) mod 8) | a permutation | 8 | 8 x 47 = 376 GB/s |

So rotation buys about **8/7 ~ +14% in the fully-phased limit** and removes the
"one rank idle per phase" artifact. Realistically the sweep is only partially phased
(304 CTAs retire tiles for many different destinations concurrently), so the achievable
slice is smaller: **estimate 0-10% of the egress pool, ~0-90 us, most likely ~5%
(~45 us).**

**The important caveat: rotation does not fix the real defect on this axis.** In *both*
rows above each sender is using **one of its seven egress links**, i.e. ~47 GB/s of a
~329 GB/s budget. The valuable half of this intervention is not the rotation, it is
**ensuring a rank always has traffic in flight to all 7 peers simultaneously** — that
is where the 7x lives. Rank rotation is the cheap hedge; per-CTA destination
interleaving is the actual mechanism. Measure whether egress is destination-phased at
all (via `Remote Write and Atomic Traffic` over time) before spending effort here.

**Correctness risk: LOW but not zero.** Rotation changes only the *order* in which
destinations are visited, not the bytes written, so a credit-based reducer that spins on
per-tile counters is indifferent to it. But it changes packet **arrival order** at every
destination, so any place that implicitly assumes a particular peer's data lands first
would break. **Flag for protocol-review as a one-line question ("does any reducer or
epoch-counter path depend on arrival order across peers?") rather than a full review.**

### Summary table

| # | intervention | expected delta on 920 us | risk | protocol review |
|---|---|---|---|---|
| **I0** | profile write amplification; grep ISA for sc/nt bits and redundant drain | 0 (decides I1 vs I2) | none | no |
| **I1** | LDS-stage peer packets -> wave-contiguous 1024 B stores | **-450 to -520 us** | medium | **yes, mandatory** |
| **I2** | correct peer stores to `sc0 sc1`; drop pre-`wbl2` `vmcnt(0)` | 0 if already correct, else large | **high** | **yes, mandatory** |
| **I3** | `nt` on payload stores only | small on egress; helps the mainloop pool | medium-high | **yes** |
| **I4** | rotate destination sweep by rank | ~0-90 us (est. ~45 us) | low | one question |

### Things this report argues *against* spending the night on

- **"More packets in flight."** TransferBench saturates a link with 8 CUs; we issue from
  ~272 CTAs at 39% efficiency (1.5, 2.5). The path is transaction-size-limited, not
  concurrency-limited. More in-flight partial writes cannot help.
- **A wider store instruction.** `dwordx4` is the architectural maximum on gfx942 (2.1).
  There is nothing to widen.
- **Chasing 448 GB/s.** That is the 7 x 64 theoretical roofline. The ~25% protocol/CRC
  haircut is not recoverable in software; **~316-336 GB/s is the real target** (1.2).


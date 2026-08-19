# Data-Movement Abstractions for MoE Megakernels

**Draft for discussion — Subha ↔ Muhammad Awad (AMD), 2026-08-19**

Purpose: stop describing our MoE kernels as *kernels* and start describing them as *data
movement*. This document (1) restates what the mentor has repeatedly asked for, in his own
words; (2) names iris/irisx's actual primitives and maps each to a concrete need in our
megakernels; (3) proposes six candidate abstractions stated purely in movement terms — no
tiles, no MFMA, no architecture constants; (4) states plainly what `draft8_final.pdf` claimed
and where its workload-generalization gap sits.

All meeting citations are `Meeting Notes/<file>:<line>`.

---

## 1. What the mentor has actually asked for

Seven months of notes contain a small number of asks, repeated. They are not stylistic
preferences; they are a specification for what the abstraction layer must and must not know.

| # | The recurring ask | In his words | Source |
|---|---|---|---|
| 1 | **The abstraction is about memory, not tiles.** A tile that knows which GPU it lives on and which GPU it goes to is the *wrong* object. | "just have it to be generic, dont make each tile have a notion regarding which gpu the tile is on … if you just talk about memory, load and store … if you start tying it to tiles then its tied to tile based stuff … there will be a sweet spot somewhere between" | `tues meeting 5.md:31-42` |
| 2 | **Device-side APIs are the only interesting layer.** Host-side collectives are a stepping stone, not the contribution. | "Critical things are device side APIs … Build on load and store (loading data from remote ranks)"; "Host side APIs are less interesting" | `mawad june 19.md:43-44`, `:35` |
| 3 | **Mid-level primitives are the deliverable.** Above raw load/store, below a fixed collective. | "If I produce a library I would focus on how to provide those mid level primitives like all_reduce_ring. Tile level granularity" | `mawad june 19.md:60` |
| 4 | **Fold the wide barrier into the producer tail and the consumer head.** The real dependency graph is finer than the rendezvous. | "at one point you have this very large barrier but then you will see that the lines do not cross … the producer consumer relationship is finer grained than the entire system … at the beginning you can do it at the GPU level, then one level lower at the CU level … you can technically go to the warp level" | `mawad july 27.md:8-16` |
| 5 | **He does not care about intra-GPU.** Movement between GPUs is the subject. | "i dont care much about the intraGPU" | `mawad july 27.md:8` |
| 6 | **Pipeline depth at tile granularity is the unexplored space.** | "Tile level collectives … the amount of software pipelining you can build is really insane and unexplored. How many tiles you produce before you do the reduce"; "Figuring out tile level communication/abstraction would be the greatest contribution I can make" | `tues meeting 1.md:23-28` |
| 7 | **Build a cost model; apply Amdahl before optimizing.** | "Worth having a little cost model"; "ahmdahls law, look at the top thing where can the speedups come from" | `tues meeting 1.md:11`, `tues meeting 5.md:6` |
| 8 | **Show the data flow first — token → bytes → layout.** | "if we want to do it at the token level thats ok but we need a concrete mapping between a token and the memory that it represents … how it's laid out in memory and how many bytes … were going to need that so we can tell how to fuse"; "just look at the data flow graph and the communication graph" | `tues meeting 5.md:38-43`, `tues meeting 2.md:11` |
| 9 | **Evaluation ladder: standalone collective → fused beside compute → end-to-end tok/s.** | "show the hierarchy. single kernel, fusion and then choose a latest model and then do (tok/s)"; "compare the collective we implemented like all reduce with the RCCL implementation … Then we go one step further. when i fuse this with GEMM" | `mawad june 15.md:74`, `:68-72` |
| 10 | **Independent work is a first-class overlap source**, not just dependent fusion. | "There is a very rich source of speedup where we overlap independent work … if we find opportunities to overlap them (could be comm/computation) those could be another source for speedup" | `mawad june 19.md:19` |
| 11 | **Do not burn CUs as data movers.** Think SDMA / async packets. | "we have wasted some of those resources to just do load and store across GPUs … i just wasted a CU and all I had to use from that CU was its registers. this is wasting … SDMA is a data movement engine … issue the thing then the CU gets control back" | `mawad june 19.md:73-78` |
| 12 | **Everything reduces to acquire/release.** | "Any code for distributed communication is going to look like acquire release in some complicated way" | `mawad june 19.md:89` |
| 13 | **Shape specialization is fine; the *abstraction* must generalize.** His Mirage critique is precisely that a fused thing that only works for one model is not a contribution. | "the problem with this approach is that this may not be generic. it might only work for certain models"; "Its ok to hyper optimize for particular shapes" | `mawad june 15.md:35`, `:85` |
| 14 | **Reality check on topology.** TP8 is not the typical serving config; EP gather/scatter is the universal pattern. | "TP8 is not used typically / TP4 is generally better"; "reduce scatter and gather operation are quite interesting … any type of EP parallelism strategy we will have to run into those operations" | `tues meeting 1.md:4-5`, `:58` |
| 15 | **Routing skew and padding are real limiters.** | "its pretty imbalanced since some experts get a ton and some get a few … that m block could be very small for certain experts due to the imbalance" | `tues meeting 2.md:6-7` |
| 16 | **Graph mode, not eager.** | "Typically people do not care about EAGER mode, most people care about GRAPH mode" | `mawad june 19.md:15-16` |

**The synthesis.** Asks 1, 3 and 13 together define the target: a layer that speaks only in
*bytes, owners, ordering and visibility*, that sits above `load`/`store`/`atomic` and below
`all_reduce`, and whose settings — not whose contract — are architecture-specific. Ask 4 tells
us the object's shape: a rendezvous decomposed into a producer-side publication and a
consumer-side wait. Ask 7 tells us the deliverable must predict, not just report.

---

## 2. iris and irisx: the actual primitives, and what each one is for in our kernel

### 2.1 The inventory (real API names)

**Foundation — symmetric heap addressing.** Every rank allocates the same object at the same
heap offset; a remote pointer is a local pointer plus a base delta.

- Triton: `iris.iris.__translate(ptr, from_rank, to_rank, heap_bases)`; context accessor
  `iris_ctx.get_heap_bases()`.
- C++/HIP (`irisx/development/include/iris/iris.hpp`): `iris_device_view::translate<T>(ptr, rank)`,
  `get_heap_base(rank)`, `cur_rank()`, `world_size()`.

**Device-side movement (Triton, `iris/iris.py`).**
- `iris.load(pointer, to_rank, from_rank, heap_bases, mask=)` — remote read into registers.
- `iris.store(pointer, value, from_rank, to_rank, heap_bases, mask=)` — remote write from registers.
- `iris.get(from_ptr, to_ptr, from_rank, to_rank, heap_bases, mask=)` — remote→local block move.
- `iris.put(from_ptr, to_ptr, from_rank, to_rank, heap_bases, mask=)` — local→remote block move.
- `iris.copy(src_ptr, dst_ptr, from_rank, to_rank, cur_rank, heap_bases, mask=)` — general
  rank-to-rank move.

**Device-side atomics (Triton), all carrying `sem=` and `scope=`.**
`atomic_add`, `atomic_sub`, `atomic_cas`, `atomic_xchg`, `atomic_and`, `atomic_or`,
`atomic_xor`, `atomic_min`, `atomic_max`. The taxonomy's producer/consumer idiom is literally
`atomic_cas(flag, 0, 1, sem="release", scope="sys")` on the producer and a spin on
`atomic_cas(flag, 1, 0, sem="acquire", ...)` on the consumer (`docs/conceptual/taxonomy.md`).

**Device-side movement + ordering (C++, `iris_device_view`).**
`load<T>`, `store<T>`, `atomic_load<T, scope>`, `atomic_store<T, scope>`,
`fetch_add<T, scope>(ptr, val, remote_rank, order)`, `fetch_sub<T, scope>`,
`compare_exchange_strong<T, scope>`, `fence<scope>(order)`. Orders:
`relaxed / consume / acquire / release / acq_rel / seq_cst`. Scopes:
`thread / warp / block / device / system`. Reference kernel: `irisx/benchmarks/all_put.hip`.

**Host side.** `iris.iris(heap_size)` context; symmetric allocators
`zeros / ones / full / empty / arange / randn / rand / randint / linspace / uniform / zeros_like`;
`barrier(stream=)`; `broadcast(value, source_rank)`; `get_rank()`, `get_num_ranks()`,
`get_cu_count()`, `get_device()`.

**Collectives (`iris/ccl/`).** `shmem.ccl.all_reduce(out, in, config=, async_op=, workspace=)`,
`all_reduce_preamble(...)`, `shmem.ccl.all_to_all(...)`. Kernels:
`persistent_all_reduce_atomic / _spinlock / _one_shot / _two_shot / _ring`, plus
`chiplet_transform_chunked` for XCD-aware CTA remapping. `iris.ccl.Config` exposes exactly the
knobs we have been sweeping by hand: `comm_sms`, `block_size_m/n`, `swizzle_size`, `num_xcds`,
`all_reduce_variant ∈ {atomic, spinlock, ring, one_shot, two_shot}`, `all_reduce_num_rings`,
`all_reduce_ring_slice_n`, `all_reduce_distribution`, `use_gluon`.

**The pattern taxonomy** (`docs/conceptual/taxonomy.md`) — five named points, which is the
coordinate system we should be answering in: (1) unfused bulk synchronous; (2) unfused
producer-consumer on two streams; (3) fused sequential; (4) fused producer-consumer with
**workgroup specialization**; (5) fused producer-consumer with **wavefront specialization**
(listed as future work, requires Gluon).

### 2.2 The map: our need ↔ iris primitive ↔ what we currently open-code

| MoE megakernel need | iris/irisx expression | What our kernel does today | Gap |
|---|---|---|---|
| Route each token row to its expert-owning rank (dispatch) | `iris.put` / `store<T>` into a `translate`d slice; sender-owned slices need no remote atomic | `pf2_full` **source-push**: origin writes fp8 rows straight into destination buffers (~0.729 ms saved vs the pull path); `frozen_n2` **destination-pull** for decode | Iris has no *plan* object. The `route_slot[token][topk]` map is the real primitive and is invented per-kernel |
| Claim a slot in a hot expert's buffer | `fetch_add<T, scope>(&count[e], 1, dst_rank, order)` / `iris.atomic_add` | `V0a` variant — known to serialize on skewed routing | Confirms `PRIOR_ART.md:1`: prefer precomputed offsets (PPLX-style sender-owned slices) over remote atomics |
| Deliver GEMM-2 partials to the token owner | `iris.atomic_add` (packed) or `store<T>` towers + local reduce | **m15**: producer epilogue does packed-BF16 remote accumulation directly into the owner slot | Iris exposes the atomic, not the *bound on how many are in flight* — the single decisive knob (−615 µs) |
| Publish "this region is finished and immutable" | `fence<device>(release)` then `atomic_store`/`store` of a monotone epoch word to each peer | **m15** two-slab certificate; **`n2r`** folds the second barrier into the phase-2 tail (release fence → 4-byte flags → peer push) | This is exactly ask #4 (`mawad july 27.md:10-16`) and iris has no name for it |
| Wait for readiness without a barrier kernel | spin on `atomic_cas(..., sem="acquire")`, or relaxed poll + one `fence<device>(acquire)` on success | m15 consumer pool; `n2r` combine waits on local flag copies | Iris's taxonomy shows the `atomic_cas` spin; it does not offer bounded/timeout polling or success-only acquire |
| Variable fan-in per key (tokens/expert differs) | `fetch_add` with a **runtime** target | `moe_mps_adapter.cuh` open-codes ~524k runtime-target RMWs/rank/epoch | Needs `counted_arrive_dynamic_into`; a compile-time-target arrival cannot express it |
| Chunked all-reduce at a TP boundary | `ccl.Config(all_reduce_variant="ring", all_reduce_num_rings=…, all_reduce_ring_slice_n=…, comm_sms=…)` | `rcclserial` / `rcclchunk` / `rcclhybrid_indep` / `rcclhybrid_merged` arms | Iris exposes chunking as a **host** config; we need per-chunk readiness visible to a *device-side* consumer |
| Do useful independent work while a collective is exposed | *nothing in iris* | `filleronly` / shared-expert filler arms (44–53% of exposed AR absorbed free) | Ask #10 (`mawad june 19.md:19`) has no primitive anywhere in the stack |
| Reuse a buffer safely across epochs | *nothing in iris* | m15 directed retirement credits | Readiness ≠ reusability; iris conflates them |
| Cut payload bytes on the wire | quantize before `iris.store` (fp8 + per-128 scales; 14,336 B → 7,392 B/token) | `V1` / QuantTile | Representation is a movement parameter, not a compute detail |

**Two honest observations about iris for the meeting.** First, iris is deliberately a
*substrate*: addressing, visibility, atomics. Everything in ask #3 — the mid-level layer — is
genuinely missing, in both the Triton and the C++ (`irisx`) trees. Second, iris's own taxonomy
stops at pattern (4), workgroup specialization, and its published win for that pattern is a
GEMM+all-scatter. Our MoE campaign says that on an occupancy-1 MFMA body, *labeling* CTAs as
communicators is not the mechanism — which is a direct, useful, empirical amendment to iris's
own taxonomy.

---

## 3. Six candidate abstractions, stated purely as data movement

Rules for this section, taken from ask #1 and ask #13: no tile types, no MFMA, no CU/XCD
counts, no MoE nouns. Every definition is in terms of *bytes, owners, order, visibility, and
admission*. Every architecture number is a **card**, not a constant.

---

### A1 — Certified region stream
*(the "chunked collective with per-chunk readiness signals" ask)*

**Definition.** A logical byte range is partitioned into **regions**. Each region carries an
ordered lifecycle:

```
payload write → drain → release → monotone epoch publication
             → relaxed poll → acquire-on-success → consumption → retirement
```

Publication asserts *immutability and visibility of one region*, not of the whole range.
Retirement is a **separate** edge from readiness: a region may be consumed long before its
storage may be reused.

**Governing rule (the one genuinely general finding we have):** *the region boundary should be
the smallest unit the next consumer can profitably execute — not the smallest unit that is
stored or transferred.* Row-level signalling cost ~926k readiness operations/rank/epoch to
describe progress no consumer could use; the median token was unusable until ~91.7% of
production had elapsed. Two coarse regions beat it outright.

**Instantiated by.** m15's two-slab certificate; `n2r`'s folded completion flags; `rcclchunk`'s
chunk boundaries; m25/CDAR's per-slab owner certificate.

**Parameters and what governs them.**
| Parameter | Governed by |
|---|---|
| Region size | Producer traversal geometry × consumer's minimum executable unit; must sit where both co-align |
| Regions in flight | Available producer shadow after the first frontier — i.e. total work per invocation |
| Retirement horizon | Epoch reuse distance; how far ahead the next epoch's producers run |
| Publication fan-out | World size; whether a leader multicasts or every contributor publishes |

**Iris expression.** `fence<device>(release)` → `store`/`atomic_store` of an epoch word per
peer; consumer relaxed `load` loop + one `fence<device>(acquire)`. Everything needed exists;
the *composition* does not.

**Falsifier.** If, at the target shape, the first region cannot be certified before ~50% of
producer work has elapsed, this abstraction has no shadow to sell and reduces to a barrier with
extra steps.

---

### A2 — Ownership handoff (the direction knob)

**Definition.** Every transfer answers one question: *whose progress defines when it happens?*
Push = source-defined, many-to-one. Pull = destination-defined, one-to-many. The **certificate
of A1 is the handoff point**: before it, producers control progress; after it, consumers do.

**Governing rule.** *Push when progress is producer-defined and many-to-one; pull when progress
is consumer-defined and one-to-many.* Direction is subordinate to ownership: a matched
transport diagnostic put push and pull within 0.64% of each other on wall time (15,723.8 vs
15,623.3 µs at 64 KiB) and within 8% on single-link bandwidth (55.46 vs 59.79 GB/s at 64 MiB).
Direction is not the lever; ownership is.

**Instantiated by.** `pf2_full` push dispatch vs `frozen_n2` pull gather — the same operator,
two directions, chosen by phase. m15's epilogue push. CDAR's ERS (push) → certificate → MAG
(pull).

**Parameters and what governs them.** Fan-in vs fan-out degree at that edge; whether the
payload is already live in the producer's registers (if yes, pulling forces it back to memory
and re-published addresses); whether the consumer knows the destination layout it needs (if
yes, pulling gathers straight into that layout for free).

**Iris expression.** `iris.put` / `store<T>` for push; `iris.get` / `load<T>` for pull; the
handoff is A1's epoch word.

---

### A3 — Credit-bounded injection

**Definition.** A cap on the number of outstanding remote operations a single producing context
may have in flight, enforced at issue time. Admission control and transport selection are **one
decision**, not two.

**Evidence that this is a first-class abstraction and not a tuning detail.** Identical
transport, one knob: unbounded producer-carried accumulation *regressed* to 7,110.8 µs
(+210.6 µs vs homogeneous); depth-4 on the same transport reached 6,495.8 µs — a 615.0 µs swing
from the bound alone. That is the largest single-knob effect in the entire campaign.

**Instantiated by.** m15's hand-rolled `vmcnt` throttle in `n2_phase2_gm_mps.cpp`; the CDAR
standalone gate (where depth 4–32 was *flat*, because transport owned the wave).

**Parameters and what governs them.** Depth is a **coupling** property, not a fabric constant:
it is decisive when transport shares issue capacity with co-resident compute, and irrelevant
when transport runs alone. Governed by co-residency (is a compute body sharing this wave?),
per-link and fan-out saturation knees, and message size. It must be re-swept whenever the
co-residency changes — which means it is a *card*, and the abstraction must ship the sweep
method, not a default.

**Iris expression.** Absent. `ccl.Config.comm_sms` bounds *how many CTAs* inject; nothing
bounds *how deep each one goes*. This is a concrete, small, defensible contribution to iris.

**Failure mode to ship with the primitive.** A bound expressed as a raw wait-count can be
silently renegotiated by register pressure elsewhere in the kernel. A bound that can die
silently is not a primitive; it needs a verifier.

---

### A4 — Counted fan-in with source-private completion

**Definition.** A region's contributors arrive at a **local** counter first; exactly one final
local arrival releases the payload and writes **one private per-source word** to the owner. The
owner polls N private words rather than N sources contending on one shared remote counter. A
source with zero contribution still publishes an explicit empty completion, so the dependency
edge exists on every legal path.

**Why it is separate from A1.** A1 says *when* a region is ready. A4 says *how the fact is
computed* without turning the fabric into a contention point. Remote `fetch_add` on a hot key
serializes across the interconnect, and MoE routing is skewed by construction — 51.9% of
expert-slot traffic hit experts 0–7 in our capture.

**Instantiated by.** m15's counted arrivals; `n2r`'s "last phase-2 CTA pushes a 4-byte flag to
every peer"; CDAR's hierarchical arrival.

**Parameters and what governs them.** Local fan-in degree (fixed vs per-key runtime — the
latter needs `counted_arrive_dynamic_into`); world size (private-word cost is O(N) per region);
skew of the key distribution, which decides whether shared-counter contention is a real risk or
a theoretical one.

**Iris expression.** `fetch_add<T, block>` locally, `store<T>` remotely, `fence<device>`
between — composed. Iris has the atoms; the two-level shape is ours.

---

### A5 — Filler slot (independent-work admission into a wait window)

**Definition.** A wait window created by a collective is an **admissible interval** with a
measured capacity. Independent work — work with no dependency on the region being waited on —
may be admitted into it, subject to: (i) it must be interruptible or short relative to the
window; (ii) admitting it must not delay the collective's own critical path; (iii) it must be
*real* work, not padding.

This is the abstraction for the mentor's ask #10 (`mawad june 19.md:19`): overlapping
**independent** work, which is a distinct source of speedup from fusing dependent kernels, and
which nothing in iris, RCCL, or MORI currently names.

**Instantiated by.** The shared-expert filler arms: `filleronly` (capacity calibration),
`rcclserial` (filler on the critical path — the production structure), `rcclhybrid_indep`
(upper bound), `rcclhybrid_merged` (token-chunked: filler chunk *i* gates AR chunk *i*).
Measured: a 235 MB all-reduce is **not** CU-bound — it loses 0.00% beside a full-GPU MFMA body,
and 44–53% of exposed all-reduce time was absorbed for free.

**Parameters and what governs them.**
| Parameter | Governed by |
|---|---|
| `k_filler` (how much independent work to admit) | Measured window capacity × the collective's own resource footprint |
| Chunking alignment (does filler chunk *i* gate AR chunk *i*?) | Whether the filler is on the critical path in production or genuinely independent |
| Admission trigger | Whether the collective is bandwidth-bound (large window, cheap admission) or latency/protocol-bound (no window to sell) |

**Governing workload property.** The window exists only where the collective is *exposed*.
Amdahl first (ask #7): measure the exposed-collective fraction at the real operating point
before designing the filler. In TP8 prefill this fraction is large; in decode, where the region
is weight-stream-bound (`tues meeting 5.md:50`), it may not be.

**Iris expression.** Absent entirely. Iris's model is that the *communication* fills the
compute's shadow; A5 is the dual — compute filling the communication's shadow — and it is the
cleanest new idea we have for his library.

---

### A6 — Movement plan (addressing and representation as a first-class object)

**Definition.** Before any bytes move, a **plan** answers, for every unit of payload: which
destination(s), at which byte offset, in which representation, and how many bytes. The plan is
an explicit, inspectable, reusable object — not a side effect of the movement kernel.

Three sub-decisions, all pure movement:
1. **Slice ownership.** Deterministic sender-owned slices (`buf[key][source_rank][slot]`)
   versus dynamic claim (remote `fetch_add`). The former needs one small counting pass and
   eliminates remote-atomic contention entirely — and a two-pass design with no contention
   beats a one-pass design with it, because in graph mode the launch gap is ~1 ns.
2. **Deduplication.** One message per (payload, destination *rank*) rather than per (payload,
   destination *key*) — our dispatch already does this and it is why we send fewer messages
   than DeepEP.
3. **Representation.** Quantize before the store, not after: 14,336 B/token bf16 →
   7,392 B/token fp8 + per-128 scales, a 51.6% payload cut. Representation is a parameter of
   movement.

The plan is also the artifact the mentor keeps asking to *see* — "a concrete mapping between a
token and the memory that it represents … how it's laid out in memory and how many bytes"
(`tues meeting 5.md:38-40`). Making it an object rather than an implicit index computation is
how ask #8 becomes an abstraction instead of a diagram.

**Instantiated by.** `pf2_full`'s plan v2 (destination ranks + destination-local row);
`frozen_n2`'s route all-gather + BM32 metadata; `route_slot[token][topk]`; QuantTile's
descriptor-parameterized body.

**Parameters and what governs them.** Skew of the destination distribution (drives whether
static slices need worst-case or measured capacity); payload/metadata byte ratio (drives
whether dedup or quantization is the bigger lever); remote fraction (a payload cut only applies
to cross-rank traffic); replan frequency (static per graph capture vs per step).

**Iris expression.** `translate` gives address arithmetic; nothing above it. The plan is the
layer where iris's symmetric heap stops and our abstraction starts.

---

### 3.7 Composition: why these six and not a collective library

Set A2 = push, A6 = owner-major traversal staggered by rank, A1 = one region per owner block,
A3 = depth-bounded, A5 = off — and the schedule that emerges *is* ring reduce-scatter, with
every link busy from the first region. Change A2 to pull after the certificate and you get
all-gather. Change A6's key from "owner row-block" to "expert" and you get MoE combine.
Two-shot all-reduce is another setting.

The abstraction does not *implement* collectives; it makes collectives and our MoE kernels —
which are not classical collectives — the same six decisions. That is the concrete form of
"tile level collectives … how many tiles you produce before you do the reduce"
(`tues meeting 1.md:23-27`): pipeline depth is A1's region size, and the unexplored
software-pipelining space he describes is the A1×A6×A5 cube, of which we have measured one
diagonal.

**What the layer must not know** (ask #1): that the operator is MoE; that a region is 512 rows;
that depth is 4; that the payload is a tile. Those are cards.

---

## 4. What draft8 claimed, and where the generalization gap is

### 4.1 The claims

**End-to-end (slide 2), DeepSeek-R1 class, ISL 4,096 / OSL 8:**

| Cell | Ours | AMD-recommended | Result |
|---|---|---|---|
| C=32, 1,024 prompts | DP8/EP8, 20,372 tok/s | TP8+EP, 25,844 tok/s | **−21.17%** |
| C=512, 2,048 prompts | DP8/EP8, 42,788 tok/s | DP8/EP8, 39,474 tok/s | **+8.40%** |

**Kernel-level (slides 4/6/7), at 4,096 tokens/rank:** production 7,712.0 µs → homogeneous
megakernel 0.8958× → producer-carried unbounded 0.9226× (*a regression*) → producer-carried
depth-4 0.8407× → m15 with 28 initial consumers **0.7544×**. (32 consumers hung in one of two
runs.)

**Mechanism (slides 5–10).** Dedicated communication CTAs were the wrong causal model: deleting
their entire payload moved ~5.7 µs while ~831 µs of protocol and interference remained. The
replacement is producer-carried, depth-bounded, coalesced delivery + consumer-oriented producer
ordering + one coarse certificate + a quota-bounded consuming pool with producer fallthrough.
Slide 10 states the lifecycle; slides 11–12 map open-coded mechanisms to proposed primitives.

**Portability (slide 13).** Producer-carried remote *accumulation* is a gfx950 property; on
gfx942 remote atomics measured ~3× slower than stores, so the portable form is source-private
store towers + owner-local reduction. Both satisfy the same contract.

### 4.2 The generalization gap — this is the part to bring to Muhammad

**(a) The headline number's regime is on a different slide from the headline.** Slides 4, 6 and
7 present "median kernel time" with no token count, no phase, no GPU count, no precision. Only
slide 14 reveals the sweep:

| Tokens/rank | m15 C=28 / production |
|---|---|
| 4,096 | 0.7557× |
| 2,048 | 0.8845× |
| 1,024 | **1.0940× — a loss** |

The measured crossover is ~1,600–1,800 tokens/rank. A reader of slides 1–13 takes away "0.75×
vs production" as a property of the kernel. It is a property of the kernel *at one point on one
axis*.

**(b) The claim is smaller than the noise.** Slide 15 states that run-to-run production
throughput varies ~15%. The headline is +8.40%, from a single directional pair with our arm run
first, no repeats and no error bars — and slides 2 and 15 are thirteen slides apart with no
cross-reference. Separately, our own forensics on an earlier serving pair found the two arms
were not doing the same work: one ran the padded graph on 249/300 steps versus 205.9/300, a 19%
composition difference that fully accounts for an 8.4% wall gap. Composition confounds at this
magnitude are the norm, not the exception, in this harness.

**(c) The mentor's own topology objection lands directly on the −21% row.** "TP8 is not used
typically / TP4 is generally better" (`tues meeting 1.md:4-5`). Our C=32 row compares DP8/EP8
against TP8+EP — a cross-topology comparison — and the −21% is partly a topology artifact.
Meanwhile the *interesting* configuration by his account (TP4 prefill, DP attention + EP MoE
for decode, gather at the EP boundary, `tues meeting 1.md:58-64`) has not been measured at all.

**(d) There is no cost model — and he asked for one twice.** `tues meeting 1.md:11` and
`tues meeting 5.md:6`. The deck has no roofline, no achievable-bandwidth ceiling, no Amdahl
fraction, and no attribution of the residual. Consequently it cannot predict the *sign* of its
own result at an unmeasured operating point — which is exactly what slide 14 demonstrates by
accident.

**(e) The decode regime is a different problem and the mechanism may not transfer.** Decode is
weight-stream bound; prefill is not (`tues meeting 5.md:50`). Every mechanism in draft8 buys
producer shadow. Where the bottleneck is weight streaming rather than exposed collective time,
there is no shadow to sell.

**(f) Notation collision.** `C` means serving concurrency on slides 2–3, communication-CTA
count on slide 4, and initial-consumer count on slides 7 and 14. Nothing disambiguates.

### 4.3 The cost model we owe him

The model does not need to be accurate; it needs to predict the sign. Minimum viable form, per
region, per operating point:

```
T_region ≈ max( T_produce(region) ,
                T_move(bytes(region), direction, depth, link_state) )
         + T_publish(world_size, fan_out)
         + T_wait_residual
T_kernel ≈ T_head + Σ_regions T_region + T_tail + T_fixed(certify, roles, queues)
overlap pays  ⟺  Σ_{r>1} T_produce(r)  >  T_fixed + T_head + T_tail
```

Fit `T_fixed`, `T_head`, `T_tail` from the three points we already have (4,096 / 2,048 / 1,024
tokens/rank); the model should then *reproduce* the 1,600–1,800 crossover rather than being
told it. That single act converts slide 14 from a caveat into the deck's strongest slide — and
it turns each abstraction in §3 into a parameter with a predicted effect, which is the
difference between "we measured six things" and "we have a design space."

**Extend the same model to the filler slot (A5):** admissible window = exposed-collective time
× (1 − collective's own resource footprint). We have the ingredients — a 235 MB all-reduce
loses 0.00% beside a full-GPU MFMA body, and 44–53% of exposed AR was absorbed free. Those two
numbers are a cost model waiting to be written down.

---

## 5. Three questions for the meeting

1. **Region granularity.** Is a full-width, fixed-row region the right first cut for a TP
   boundary, or should the region follow the *consumer's* natural unit (the next operator's
   staging layout) even when that makes producer traversal awkward? A1's whole value rides on
   this choice.
2. **Kill criterion for a transport.** What amount of co-resident compute degradation should
   disqualify a transport that wins on standalone bandwidth? A transport that moves bytes fast
   but steals 10% of MFMA throughput is not an overlap win, and we currently have no threshold.
3. **Where does A5 (the filler slot) belong?** It is the one abstraction with no precedent in
   iris, RCCL, MORI or DeepEP, it is the direct realization of `mawad june 19.md:19`, and it is
   the cheapest thing on this list to prototype in Triton against `ccl.all_reduce`. Should it
   lead, or should the certified-region stream lead?

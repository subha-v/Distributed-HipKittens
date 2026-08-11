# LIT-M3 — push granularity: is 14 KiB really 70× below the knee?

Research artifact. No GPU runs, no kernel edits. Labels: `DOCUMENTED` (vendor doc or the paper itself) · `REPORTED` (secondary) · `INFERRED` (my reasoning) · `UNCERTAIN`.

## Verdict

**Arithmetically correct, cited to the wrong paper and the wrong hardware, and the conclusion drawn from it does
not follow.** 1 MiB / 14,336 B = 73.1×, so the ratio is real — but it comes from arXiv **2607.19539** (§3.3.2,
Fig 3a), **not** COMET, and was measured on **4× A100 over NVLink with a ~100 GB/s one-way per-peer ceiling**,
where 87 GB/s is **87% of a single peer link**. Our 148 GB/s is **27.5% of 8-way aggregate egress** (148/537.6).
Those are different quantities: a curve about where bandwidth *plateaus* cannot show that 27.5% of peak is
unreachable. Two numbers kill the strong form. (1) At C=32 the cost model needs only **4.63 GB/s per service
CTA** — one 14 KiB push per 3.10 µs — against the **10.9 GB/s per SM** that paper's own consumer sustained at
its plateau. (2) AMD's mori-EP reports **234–420 GB/s xGMI combine on MI355X** at hidden=7168 BF16, whose
per-token payload is **exactly 14,336 B** (`REPORTED`). **Do not run M3 first.** A 3.10 µs budget per 14 KiB
push is enormous relative to the bytes, so a shortfall is almost certainly per-push serialization — the `vmcnt(0)`
drain, `buffer_wbl2`, the flag store — not payload size. Order: **A1 (C ∈ {32,48,64,90})**, then the fence class
**M6 → M1 → M5 → M4**; reclassify M3 from "gates whether ~148 GB/s is reachable at all" to a bandwidth-efficiency
optimization for afterwards. Settle it cheaply with **TransferBench's GFX executor** plus the fence sweep below.

## COMET, as measured

arXiv 2502.19811, *Comet: Fine-grained Computation-communication Overlapping for MoE* (ByteDance Seed + SJTU).
All `DOCUMENTED` from the paper.

- **Testbed (§5.1):** 8× **NVIDIA H800** (80 GB), NVLink; CUDA 12.3, **NVSHMEM 2.11**, PyTorch 2.4.0, Megatron-LM.
  Second cluster (§5.4): 8× **L20** over PCIe bridges, "GPU-to-GPU bandwidth is around 25 GB/s."
- **Structure:** §3.1 (3.1.1 decompose, 3.1.2 reschedule), §3.2 (3.2.1 thread-block specialization, 3.2.2 adaptive
  assignment). **There is no §3.3.** No bandwidth-versus-transfer-size measurement anywhere, no transfer-size
  figure, and the units KiB and MiB never appear in the paper.
- **The only internal design-parameter ablation, Fig 8 / §3.2.2:** total blocks = 132 = Hopper SM count;
  optimal comms blocks `n_c` = **18 → 26** as M goes 4096 → 16384 at TP=8, and **46** at TP=4, M=16384 — i.e.
  **13.6 / 19.7 / 34.8%**. This is the real provenance of our "14–35%", and it checks out.
- **Headlines:** **1.96×** single layer, **1.71×** end-to-end, **86.5% of comm latency hidden** (vs 29.2% FasterMoE,
  68.6% Tutel, Fig 11); 1.28–2.37× across M; 1.16–1.83× across E/topk; 1.19–1.46× on L20.

**The actual source of the granularity numbers** is arXiv **2607.19539**, *Fine-grained Computation-Communication
Overlap via Tile-level Signaling and Scheduling for MoE* — a different paper, in our own shape (persistent GEMM
producer + persistent NVSHMEM consumer on a disjoint SM partition). All `DOCUMENTED`:

- **§3.3.2 "Communication Granularity", Fig 3(a)**, verbatim: "approaching the **87 GB/s peak at 8 SMs around
  1 MiB** and **67 GB/s at 4 SMs around 3 MiB**. A single-tile transfer (**usually 32 KiB or 64 KiB** depending
  on the tile configuration used) lies far from saturation."
- **Hardware (§4.1.1):** "a single node with **four NVIDIA A100 GPUs**… Each NVLink lane delivers 25 GB/s per
  direction, and a GPU-to-GPU pair is connected by 4 NVLink lanes, giving a **one-way bandwidth ceiling of
  roughly 100 GB/s per peer**. Each GPU has **108 SMs**." CUDA 12.1, CUTLASS 3.9, NVSHMEM 3.6.5. **Segment
  policy:** first and last segments are one row band, middle segments coalesce `x` bands — `x=1` ≈ **1–2 MiB**,
  `x=2` ≈ **2–4 MiB**, "neither uniformly best."
- **§4.3.4 / Fig 9, the collapse evidence:** `cCTA` swept 2→24 (default 14). At `cCTA=2` it is "slower than the
  baseline at every shape and router, reaching **1.91x the baseline** under stress_skew routing at M=32768,
  E=4." Lowest latency "usually" at **cCTA ∈ [10,20]** = 9.3–18.5% of 108 SMs. Skew amplifies the penalty.

**Transferability to xGMI — partial, three caveats.** In favor: A100 has no TMA and no copy engine here; the
consumer is NVSHMEM device-initiated loads/stores from CUs, the same mechanism class as our vector stores over
xGMI (`INFERRED` from §3.4). Against: (a) their ceiling is one 100 GB/s link, we have **7 single-hop links and
537.6 GB/s aggregate egress**, so ~7× more link parallelism to hide latency in (`INFERRED`); (b) 8 of 108 SMs on
a monolithic die is not 8–90 of 256 CUs across 8 XCDs with private 4 MB L2s; (c) Fig 3(a)'s x-axis is not stated
to be per-CTA or aggregate (`UNCERTAIN`) — see correction 4. Figs 3(a) and 9 are **measurements**; the row-band
choice, the segment policy and the stream-priority argument are **design reasoning**, with no ablation.

## Corrections to our internal note

1. **Wrong paper and wrong hardware.** "COMET §3.3.2" does not exist. §3.3.2, Fig 3(a), 87 GB/s, 32–64 KiB
   tiles, the ~1 MiB knee and the row-band idea are all **arXiv 2607.19539**, measured on **4× A100 / NVLink /
   108 SMs / ~100 GB/s per peer** — not H800, not H100. Our note attributes the curve to COMET's H800 system.
2. **Wrong kind of number.** 87 GB/s is 87% of **one** peer link; our 148 GB/s is 27.5% of **aggregate egress
   across 7**. Distance from a plateau says nothing about clearing 27.5% of peak.
3. **The conclusion is inference, not measurement, and it is contradicted.** "No amount of tuning `g` can reach
   ~148 GB/s" is refuted in strong form by mori-EP's published MI355X EP8 combine at the same 14,336 B payload.
4. **The note never divides by C — that is what makes 73× look fatal.** If Fig 3(a)'s x-axis is an 8-CTA
   consumer's aggregate payload, the **per-CTA** payload at the knee is 128 KiB and our 14 KiB is **9.1× low, not
   73×**; a C=32 pool holds 448 KiB in flight, within 2.3× of the knee. If the x-axis is per-CTA the gap is 18×.
   Neither reading gives 73× (`INFERRED`; the paper is `UNCERTAIN` on which).
5. **M3's stated purpose overstates it.** It is a bandwidth-efficiency optimization, and in our design it costs
   something real: a ≥256 KiB band delays first publish and couples to M8 slice completion.
6. **M7 and M8 confidence is too high.** Both say "mechanism measured." COMET reports **no separately measured
   speedup for layer-0 or layer-1 rescheduling** — no ablation section exists; both live inside the same 1.96× /
   86.5%-hidden aggregate. Downgrade to *mechanism described, speedup not separately measured*.
7. **Minor slip.** Fig 5 is in **§3.1.2**, not §3.1.1 (§3.1.1 is the axis argument; Figs 5 and 6 are in §3.1.2).
8. **Fleet is not bandwidth evidence.** arXiv 2604.15379 is a **single-GPU** multi-die megakernel (XCD L2
   locality, Qwen3-8B decode vs vLLM). It supports M4's atomic-scope and `buffer_wbl2` claims but has **no xGMI
   peer-write bandwidth data**. Its 8/256 CUs (3.1%) is a *scheduler* role, not comm — no evidence for sizing C.
9. **What the note got right** (so we stop re-deriving it): the 73× arithmetic; COMET's 14–35% `n_c`
   (§3.2.2/Fig 8); the M/N axes (§3.1.1); the `cCTA=2` → 1.91× collapse (2607.19539 §4.3.4/Fig 9 — though its
   optimum band is **9–19%** of 108, tighter and lower than 14–35%); and MI350X 76.8 GB/s per direction per link,
   confirmed independently by ROCm's "1,075.2 GB/s P2P ring aggregate" ⇒ 153.6 bidirectional ÷ 2.

## COMET layer-0 and layer-1

Axis rules, **§3.1.1**, `DOCUMENTED`: layer0's consumer is a GEMM taking the shared tensor as **input**, so tokens
are independent along **M**; "since the computation of a GEMM tile involves multiplication and reduction along the
token embedding dimension, decomposing the shared tensor along this dimension is not feasible." Layer1's consumer
contains a **top-K reduction along M**, creating token interdependence, so it "can only be decomposed along the
**N** dimension."

- **Layer-0 (§3.1.2, Fig 5):** tokens **sorted by source rank**; GroupGEMM tile order chosen "to minimize dependency
  on remote data, with computation beginning from tiles containing local tokens while the transfer of other remote
  tokens proceeds concurrently."
- **Layer-1 (§3.1.2, Fig 6):** "Instead of computing each expert sequentially, GroupGEMM operations are executed
  **column-wise**… allows the reduction and communicate operations to proceed as soon as the **first `T_N` columns**
  of the shared tensors are computed." `T_N` = GroupGEMM tile size along N.
- **Separately measured speedups: none.** No per-layer ablation exists; the reschedule is only ever measured jointly
  with thread-block specialization inside the 1.96× / 1.71× / 86.5%-hidden aggregate.

## AMD-side evidence

**No published transfer-size vs achieved-bandwidth curve exists for device-initiated (in-kernel, non-SDMA) small
peer writes on MI300X or MI350X in the 4 KiB–1 MiB range.** Everything published is a multi-MB peak, an SDMA curve,
or a collective-level curve. Closest available:

- **mori-EP (`REPORTED`, the strongest data point).** ROCm/mori README, DeepSeek-V3, 4096 tokens, hidden 7168,
  top-8, FP8 dispatch + BF16 combine. **MI355X EP8: dispatch 345, combine 420 GB/s XGMI**; EP16-V1 combine 234.
  Latency mode (128 tokens): dispatch 31 µs / 142 GB/s, **combine 36 µs / 276 GB/s**. MI300X+CX7 EP8: 307/330.
  Maintainer (issue #107) states bandwidth-mode XGMI is `total recv bytes / duration` and that the numerator is
  inflated by aggressive token deduplication; latency mode assumes no dedup. **Per-token BF16 payload at
  hidden=7168 is 14,336 B — exactly our push size** (`INFERRED` that a DeepEP-style combine's unit is one
  token's hidden vector; `UNCERTAIN` whether mori coalesces per destination). Even at the conservative
  276 GB/s, 148 is 54% of what AMD ships.
- **In-kernel vs SDMA, MI300A (`REPORTED`).** Schieffer et al.: STREAM copy kernel to a peer APU **103–104 GB/s =
  81% of theoretical IF link bandwidth**, vs 90 GB/s `hipMemcpyPeer` — but the sweep is **2 MB–8 GB only**.
- **SDMA, the contrast case (`REPORTED`).** MI300X `hipMemcpyPeer` needs 16.8 MB for ~80% of its 50 GB/s roofline,
  and DMA collectives average **4.5× / 2.5× slower than CU-based below 32 MB** — confirms M10 ranks last.
- **Link peaks (`DOCUMENTED`).** ROCm workload guide: MI350X IF links 38.4 Gbps, 2 IODs, **1,075.2 GB/s P2P
  ring aggregate**. ROCm RCCL/xGMI blog, MI300X: 64 GB/s per link theoretical, **45–48 realized (~75%)**,
  315–336 of 448 GB/s aggregate — the source of our "~75% of raw" figure.
- **TransferBench (`DOCUMENTED`) — run this before writing any code.** Ships with ROCm. Its GFX executor is
  **kernel-based** CU-issued copies with `NUM_SUB_EXECS` = CU count per transfer, and `SAMPLING_FACTOR=0` sweeps
  **1 KB → 512 MB**. Preset **`rwrite`** is literally "parallel remote writes from a single GPU to other GPUs."
  So most of the curve we want already exists as a supported tool invocation (details under Sources).

## Microbenchmark spec

Standalone HIP, ~250 lines. Purpose: map the achieved-bandwidth surface for device-initiated peer writes on
**our** node and separate *payload size* from *fence cost*. **Run TransferBench first** (`SAMPLING_FACTOR=0
NUM_SUB_EXECS=1,8,32,64 ./TransferBench rwrite`, then `schmoo`) — it answers the size × CU-count half in
minutes. Write the harness only for the fence axis, which TransferBench cannot express.

**Grid.** size `S` ∈ {896 B, 14 KiB, 64 KiB, 256 KiB, 1 MiB, 4 MiB} × pushers `P` ∈ {1, 8, 32, 64} × fence policy
`F` × topology ∈ {1 peer, 7 peers} × heap ∈ {`uncached` (mori default), `normal`} — the last folds in M6 for free.
384 points at ~5 ms each: minutes, not hours.

**The fence axis is the point** — without it the benchmark cannot separate "payload too small" from "release too expensive," which *is* the question.
- `F0` — payload stores only, one `s_waitcnt vmcnt(0)` after all iterations. The bandwidth ceiling.
- `F1` — today's protocol: `vmcnt(0)` drain + `buffer_wbl2` + scoped flag store per push.
- `F2` — M1: `sc0 sc1` payloads, bare `s_waitcnt vmcnt`, scoped flag, **no** `buffer_wbl2`.
- `F3` — M5: partial `s_waitcnt vmcnt(N)` overlapping push *k* / *k+1*.

**Kernel.** Persistent, `P` CTAs × 256 threads, 16 B (`dwordx4`) stores through a `translate_peer` pointer into a
mori symmetric heap. Each CTA loops `R` times: push `S` bytes to peer `p` (round-robin over 7 peers in the
fan-out arm, else fixed peer 1), apply `F`, advance. Per-CTA source = a 1 MiB pre-touched local ring (keeps HBM
read pressure out of the measurement); destination offsets stride by `S`, are disjoint across CTAs and iterations,
and the footprint is **≥ 256 MB** so it does not sit in the peer's Infinity Cache. Log `s_getreg_b32
hwreg(HW_REG_XCC_ID,0,4)` per CTA — that answers A8's placement question for free.

**Timing.** In-kernel `wall_clock64()` on CTA 0 spanning the whole loop, plus host `hipEventElapsedTime` as a
cross-check; they must agree within a few percent or the harness is wrong. Pick `R` so each point runs ≥ 2 ms
and `R ≥ 200` (launch overhead < 1%). 11 launches, discard the first 3, report min/median/max.
`GB/s = P × R × S / elapsed` (10⁹), reported **both** per-link and as aggregate egress. **Hold constant:** clocks
(`rocm-smi --setperfdeterminism`; record `--showclocks` before and after), exclusive node, 256 threads/CTA, 16 B
store width, one push in flight per CTA for `F1`/`F2`, destination footprint, heap type within a sweep.

**Validation (one command).** At `S=4 MiB, P=64, F0`, single peer, the result must land within ~15% of
`NUM_SUB_EXECS=64 ./TransferBench p2p` GFX for the same pair. If not, the harness is broken — interpret nothing
else until it agrees.

**Decision rules.**
- **REFUTES the claim:** at `S=14 KiB, F0`, 7-peer aggregate **≥ 148 GB/s** at any `P ≤ 64` (≥ 4.63 GB/s per CTA
  at P=32). Payload size is not the blocker; M3 drops down the queue and the only open question is `F1` vs `F0`.
- **CONFIRMS it:** at `S=14 KiB, F0` the aggregate stays **< 148 GB/s at every** `P`, **and** `S ≥ 256 KiB`
  clears 148. `F0` has no fences, so a shortfall there is genuinely a granularity effect — the only pattern that
  makes M3 load-bearing.
- **Outcome I expect (`INFERRED`):** `F0` at 14 KiB clears 148 but `F1` does not. Then the blocker is the
  per-push release, the fix is M6 → M1 → M5 → M4 rather than M3, and the gain is bounded by (`F0` − `F1`)/`F1`.
- **Mechanism-level falsifier:** if even `S=4 MiB, P=64, F0` over 7 peers misses 148 GB/s, no granularity change
  saves the cost model — escalate to M9 / M10 / M11.

**Required per-CTA rate, the crux** (`INFERRED`, arithmetic on the cost model): 148 GB/s ÷ C → **C=32: 4.63 GB/s
per CTA** (one 14,336 B push per 3.10 µs); **C=48: 3.08** (per 4.65 µs); **C=64: 2.31** (per 6.20 µs). The cited
A100 plateau is 87 / 8 = **10.9 GB/s per SM**; our C=32 requirement is 42% of that.

## Sources

- https://arxiv.org/abs/2502.19811 — COMET: 8× H800 + 8× L20; §3.1.1 M/N axes; §3.1.2 + Figs 5/6 layer-0/1
  reschedule; §3.2.2 + Fig 8 `n_c` = 18/26/46 of 132; 1.96× / 1.71× / 86.5% hidden. **No §3.3, no
  bandwidth-vs-size measurement.**
  Mirror https://ar5iv.labs.arxiv.org/html/2502.19811 confirms the same section structure.
- https://arxiv.org/html/2607.19539 — **actual** source of §3.3.2 / Fig 3(a): 87 GB/s at 8 SMs ≈ 1 MiB, 67 at 4 SMs ≈ 3 MiB, 32–64 KiB tile "far from saturation"; §4.1.1 = 4× A100, 108 SMs, ~100 GB/s per peer; §4.3.4 + Fig 9 `cCTA=2` → 1.91× baseline, optimum [10,20].
- https://github.com/ROCm/mori — MI355X EP8 combine **420 GB/s**, dispatch 345; latency mode 128 tokens combine 36 µs / 276 GB/s; hidden 7168 BF16 ⇒ 14,336 B per token. https://github.com/ROCm/mori/issues/107 gives the two bandwidth formulas and the deduplication caveat.
- https://arxiv.org/html/2604.15379v1 — Fleet: persistent megakernel on MI350X, **single GPU / multi-die**; 8 of 256 CUs for schedulers; device-scope atomics cross all 8 XCDs; `buffer_wbl2` scales with dirty lines.
- https://rocmdocs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/workload.html — MI350X IF links 38.4 Gbps, **1,075.2 GB/s P2P ring aggregate**, 8-GPU fully connected, 7 links, 256 active CUs, 160 KB LDS.
- https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html — MI300X 64 GB/s per link theoretical, **45–48 realized (~75%)**, 315–336 of 448 GB/s aggregate.
- https://www.memsys.io/wp-content/uploads/ninja-forms/5/Memsys_multigpu_MI300A-6.pdf — in-kernel STREAM peer copy **103–104 GB/s = 81% of IF peak** vs 90 SDMA; sweep **2 MB–8 GB only**.
- https://www.scalarlm.com/blog/scalarlm-benchmarking-mi300x-memcpy-peer/ and https://doi.org/10.48550/arxiv.2511.06605 — the SDMA contrast: `hipMemcpyPeer` needs 16.8 MB for ~80% of roofline; DMA collectives 4.5× / 2.5× slower than CU-based below 32 MB.
- https://rocm.docs.amd.com/_/downloads/TransferBench/en/latest/pdf/ — GFX kernel-based executor, `NUM_SUB_EXECS` = CU count, `SAMPLING_FACTOR=0` sweeps 1 KB–512 MB, presets `rwrite`/`schmoo`/`scaling`/`a2asweep`/`p2p`, `MEM_TYPE` uncached, GFX timing via in-kernel `wall_clock64()`.
- https://rocm.docs.amd.com/projects/rocSHMEM/en/docs-7.2.3/api/rma.html and https://ucfconsortium.org/wp-content/uploads/2024/12/2024_1_UCX-in-the-AMD-Instinct-MI300-Series.pdf — device-side `putmem` / `_wg` / `_wave` exist and MI300X putmem-vs-size plots were shown, but the slides carry **no readable numeric values** and use 1 workgroup only; not a usable curve.

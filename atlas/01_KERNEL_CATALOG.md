# MoE Megakernel Variant / Arm Catalog — Distributed-HipKittens

Scope: every distinct megakernel variant or experimental arm found across the local branches.
Paths verified with `git ls-tree`; `branch:path` form means the file lives on that branch.
Every verdict is quoted from a ledger line, commit message, or source comment; unverified items
are marked **verdict unrecorded**.

---

## A. The DP8/EP8 prefill MoE megakernel lineage (`k0pf6gm_*`, MoK synthetic prefill rig)

All of these share one workload regime unless stated: **MoK synthetic prefill, T=4096 tokens/rank, DP=8 / EP8 (32 experts/rank of 256), DeepSeek-R1-0528 shapes, 8× MI350X gfx950, 100% fill.** Baseline denominator is "production" = AITER fused-MoE + MoRI all2all replayed in the same harness (p50 ≈ 7,712 µs). Ratio < 1 = faster.

### A1. `pf6gm_mega` — the pre-M15 fused baseline megakernel
- **Path:** `ablations:distributed-kernels/fused_moe/k0pf6gm_device_tile.hip`
- **Strategy:** monolithic fused dispatch → expert GEMM → combine; all2all hidden behind expert GEMMs.
- **Verdict:** 6,908.8 µs = **0.8958×** production (`aug12/exp_03_m15_slab_combine/result.md` headline table). Superseded.

### A2. `mode 12` + depth-4 admission — "the ratchet"
- **Path:** `ablations:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip` (rev 26, `.text` 642646fc); host `n2_phase2_gm_mps.cpp` carries the depth-4 vmcnt throttle.
- **Strategy:** remote-accumulate (`global_atomic_pk_add_bf16`) epilogue combine straight from the expert GEMM epilogue, with a g-encoded **depth-4 vmcnt injection throttle** and deferred drain (task t's drain paid at t+1's head). One kernel, no side stream; every producer CTA is also a carrier.
- **Verdict:** **0.8407×** (6,483.8 µs) as the in-session control (`exp_03_m15_slab_combine/result.md`). The depth-4 bound alone is worth **−615.0 µs** (t = −80.2) vs unbounded carrier relocation (LAW-20/21, quoted in `OVERLAP_ATLAS.md` SCREEN-3). Root cause of the win: bounded injection, not more parallelism.

### A3. **M15 — slab-certified pipelined combine** — the standing champion
- **Branch/paths:** `ablations:distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip` (canonical source); RUN PIN `ablations-m15:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip` (m15 body compiled under the harness symbol `k0pf6gm_mps_mega`, commit `5753970b`). Design: `ablations:distributed-kernels/fused_moe/overnight/aug12/M15_DESIGN.md`.
- **Strategy:** M7 task order becomes **NC-major**, the task space cut into **S=2 column-disjoint slabs**. After each slab, one rendezvous + **one coarse epoch word per (rank,slab)** published to all 8 peers — replacing ~926k per-row readiness RMWs with 2×8 slab words. The C-pool CTAs stop carrying and instead **consume front-half combine batches inside the back-half GEMM's shadow**. Epilogue transport is mode-12's remote-accumulate verbatim (depth-4).
- **Verdict — FAST, best in family.** C=16: 6,292.4 µs = **0.8165×**, −191.4 µs vs in-session control, arms disjoint by 173 µs. C=28 (shipping config, g=353, mode=12, flush_rows=16): **5,822.0 µs = 0.7544×**; re-anchored 2026-08-18 at **0.75187** (median of 5 order-balanced, `nightshift/REPORT.md` M4). Attribution: combine residue **324.2 → 180.0 µs (−144.2)**, M7 −48.7, M6 flat; budget closes to 2 µs.
- **Why:** protocol deletion (door b) + moving a latency-bound consumer into a compute shadow (door a) — not "filling stalls" (impossible at occupancy 1).
- **Secondary finding:** C=8 costs +92.8 µs vs C=16 — mode-12's "never dedicate CTAs" placement law **inverts**; dedicating CTAs to *consuming in the producer's shadow* pays, dedicating them to *carrying* does not.

### A4. **mode 16 / TBO-2 deferred combine** (`K0P6_MPS_ENABLE_TBO`)
- **Branch/paths:** `ablations-tbo16:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip` (TBO default-1 pin, `a91cff84`) + `ablations-tbo16:distributed-kernels/fused_moe/n2_phase2_gm_mps.cpp`. Result: `ablations:distributed-kernels/fused_moe/overnight/aug12/exp_02_tbo_deferred_combine/result.md`.
- **Strategy:** two-batch-overlap — defer the combine of epoch N out to launch N+1 (double-buffered output halves, parity-indexed), so combine escapes its `(t/S)^8` readiness ceiling and rides the *next* launch's compute.
- **Verdict — FALSIFIED.** 6,559.6 µs = **0.8513×** vs the same-session 0.8407× control = **+75.8 µs**, past the pre-registered +30 µs falsifier. Cross-epoch deferral bought nothing at this config; the initial "failure" was a host-side 1-based-epoch off-by-one (kernel exonerated).

### A5. **M15b / mode 15b — staged wide-push transport** (`K0P6_M15_STAGED=1`)
- **Branch/paths:** `ablations-m15b:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip` (`27b36303`); arm lives in `ablations:.../k0pf6gm_device_tile_m15.hip:68`.
- **Strategy:** **swap the fabric op class** — epilogue folds partials into a *local* bf16 stage (only m7tab pointer values change, instruction stream identical), intra-producer collisions fold locally, carrier CTAs push folded half-rows as **wide posted packet stores** into owners' slots (~1.44× fewer remote bytes; 0.127 vs 0.571 µs/op on the issue ladder). Requires descriptor slot 63 stage (`K0_M15B=1`), run with g=65.
- **Verdict — FALSIFIED, badly.** 9,087.1 µs = **1.1768×** — correct but ~2.8 ms slower. "Op class is not the tax" (`aug12/STATUS.md` session-2). Later diagnosed in `aug18-prefill/FP8_WIRE_DESIGN.md`: this arm compiles to **96 `scratch_load_dword` + 97 `s_waitcnt vmcnt(0)` within 24 instructions of the remote atomic** (default build: 0 and 1) — exp_24's LDS-base eviction + exp_38's throttle annihilation, both live. Marked P0 blocker.

### A6. **M17 — source-interleaved (round-robin) scatter** (`K0P6_M15_SCATTER_RR`)
- **Branch/paths:** `ablations-m17:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip` (`b7561d47`); arm at `ablations:.../k0pf6gm_device_tile_m15.hip:81`.
- **Strategy:** M15 with the scatter iteration RR-permuted across sources, so landing order interleaves ranks rather than draining donor-major (targets the xGMI-link-underuse pathology MORI documents).
- **Verdict — TIE at balanced, ~1% under skew.** Balanced: 5,845.3 µs (d4) / 5,883.3 (d8) — a tie with M15, depth-4 optimum robust; "parked as skew insurance". Under measured-skew replay: 20,504 µs = 0.8370× vs M15's 20,739 = 0.8467× (agg) — "RR landing order buys ~1–1.5% under skew; the bottleneck is **popularity concentration**, not ordering" (`aug14/M18_REPLICATION_RESULTS.md`).

### A7. **M18 — static hot-expert replication** (`K0P6_M15_REPLICATE`)
- **Branch/paths:** `ablations-m18:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip` (`92440297`); arm at `ablations:.../k0pf6gm_device_tile_m15.hip:99`. Results: `ablations:distributed-kernels/fused_moe/overnight/aug14/M18_REPLICATION_RESULTS.md`.
- **Strategy:** not an overlap change — a **data-movement deletion**. Every rank hosts hot experts' weights in local slots `E..E+nrep-1`; those experts route **source-locally** (`dest = cur`), so their dispatch bytes and M7 remote RMWs leave the fabric entirely (>52% of traffic off xGMI under the measured histogram). Exactly-once via self-source acceptance in M2's histogram + the scatter. M7/M8/M9 combine untouched.
- **Verdict — FASTEST recorded kernel-region result under skew.** Measured-aggregate routing replay: **0.3085×** (8 replicas), **0.2489×** (16), **0.2420×** (32); worst-layer layer-adaptive-16 **0.2169×**. Balanced control with 8 replicas: **0.7848×** (+3.9% carry cost vs M15). Decomposition: ~3.2× from removing the slowest-rank overhang + ~1.3× from the M15 fusion chassis.
- **Regime:** MoK kernel-region, one MoE layer, 4096 tok/rank prefill, fixed replayed MLPerf-R1 route histogram (51.9% of traffic on experts 0–7; 5.09× receive-side skew). **Decode untouched.**

### A8. **M19 — per-chunk adaptive replication** (`K0P6_M15_ADAPTIVE`)
- **Branch/paths:** `ablations-m19:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip` (`7185838e`); arm at `ablations:.../k0pf6gm_device_tile_m15.hip:120`. Design: `ablations:distributed-kernels/fused_moe/overnight/aug14/M19_DESIGN.md`.
- **Strategy:** M18 chassis + an **in-kernel M0.5 pre-pass** that builds an own-routing histogram, applies a θ threshold per expert, and publishes decision bitmaps ordered-before `rows_done`; receiver acceptance mirrors sender bits. Cold chunk == M15, hot chunk == M18.
- **Verdict — mixed / partly retracted.** Kernel gates green (reference-match). Costs **+1.2 ms constant** from the M0.5 pre-pass's two grid-barrier rendezvous (~600 µs each) and **+41 GB/rank** of per-layer replicas (halved the serving KV pool 68.8 → 31 GiB) — both quoted in `aug14/M20_DESIGN.md`. Its e2e "+39.8%" serving pair was **retired 2026-08-18** as invalid (see §D1). Its validating captured-route replay **was never run** (`M21_OVERLAP_DIRECTION.md` §1(d)).

### A9. **M20 — slot-pool replication + cross-layer weight prefetch** (`K0P6_M20_SLOTPOOL`)
- **Branch/paths:** `ablations-m20:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip` (`26c97093`/`c26404aa`); arm at `ablations:.../k0pf6gm_device_tile_m15.hip:147`. Design: `ablations:distributed-kernels/fused_moe/overnight/aug14/M20_DESIGN.md`, handoff `.../M20_SERVING_INTEGRATION_HANDOFF.md`.
- **Strategy:** M19's routing machinery with (a) a **triple-buffered ~1 GB replica slot pool** replacing 41 GB static replicas, (b) **cross-layer weight prefetch by 4 slack service CTAs** whose duty tables split over two upstream kernels (stream order is the certification), (c) decisions **precomputed pre-launch** — deleting M19's two rendezvous. Dual-base weight addressing in both GEMM phases via the threaded descriptor.
- **Verdict — FAST, the production-practical point.** **0.2483×** production under measured serving skew (6,035 vs 24,283 µs) and **0.823×** balanced at θ=64 (~0.76 with θ above uniform), at **672 MB** vs M19's 41 GB (README, aug14 evening). Supporting negative result on the fabric: per-step MoonEP-style weight streaming is infeasible — **pull +93 ms naive / +23 ms unrolled** (remote-read latency-bound), **push +54 ms** (~1.4 GB/s/CTA service-pool law).

### A10. **tgen — T-clamped M15 (batch-size generality)**
- **Branch/paths:** `ablations-tgen:distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip` (SRC_REV 2, M2 hole-sentinel chunk-range clamp, `af030778`); result `ablations-tgen:distributed-kernels/fused_moe/overnight/aug13/exp_04_tgen/result.md` (+ `tgen_donor_fix.diff`, `tgen_rc.diff`). **Note: the result.md exists only on `ablations-tgen`, not on `ablations`.**
- **Strategy:** correctness fix, not a schedule change — clamps M2 hole-sentinel chunk ranges to the segment so T≠4096 certifies.
- **Verdict — mixed.** C=28: **0.756× at T=4096** (no regression: 5,828.8 vs 5,822.0 control), **0.885× at T=2048** (the earlier "inversion" overturned as a w1t1p1 cold-start artifact), **1.094× at T=1024** — the megakernel family **inverts at small batch, break-even ≈ 1,600–1,800 tokens/rank**. C-sweep monotone at T=2048 (C=28 best). Root cause of the inversion: fixed fusion/protocol costs stop amortizing.
- **Critical caveat (`nightshift/REPORT.md` R1):** the banked t2048/t1024 numbers carry the **tgen** source sha `20b8c6bb`, not the M15 chassis sha `935f555e`. The **M15 chassis itself fails the correctness gate at T=2048/1024** (`[MOK GATE] pass=False`, rel 0.874/0.839) — measurement M5.

### A11. **M23 — "ragged seal" + uniform-decode rescue** (integration arm, zero HIP change)
- **Paths:** `ablations:distributed-kernels/fused_moe/overnight/aug18-prefill/M23_RAGGED_SEAL_DESIGN.md`, `.../m23/M23_IMPL_NOTES.md`, `.../m23/m23_patch.py`, `.../m23/offline_dispatcher_replay.py`.
- **Strategy:** host/vLLM-side only. Makes the megakernel's activation seal accept the padded-4096 batches production already runs, and routes uniform-decode ranks to the graph in **both** arms.
- **Verdict — the enabling fix, not a speedup.** Seal coverage **~2% → 99% of in-bucket steps / ~95% of tokens**. Fixing the baseline **roughly doubled it**. First honest pair (n=1, c32p): rescued stock **20,918** vs m15+M23 **19,277** input tok/s.

### A12. **M24 — fill-aware megakernel** (`K0P6_M24_FILL` + 8 sibling macros)
- **Paths:** arms at `ablations:distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip:182-269` — `K0P6_M24_FILL`, `K0P6_M24_TGRAIN` (ship 8), `K0P6_M24_ZERO_PAD`, `K0P6_M24_ALLOW_DIRTY_PAD`, `K0P6_M24_NULLWORK`, `K0P6_M24_NORIG_CONST`, `K0P6_M24_STRICT`, `K0P6_M24_DENSE_SCATTER`, `K0P6_M24_FILL_C`, `K0P6_M24_M8_ADAPT`. Design: `ablations:.../overnight/aug18-prefill/FILL_AWARE_DESIGN.md`; analyses `.../m24/G0B_LOCAL_FILL_HISTOGRAM.md`, `.../m24/DUMMY_SYNCHRONY.md`, `.../m24/G7_TGRAIN_ANALYSIS.md`; host `.../m24/m24_mok_patch.py`, `.../m15_eplb0/m24_fill.py`.
- **Strategy:** make cost scale with `n_orig` rather than the padded 4096 — Tier 1 = two **null-work fast paths** (whole-step skip when all 8 ranks are dummy), Tier 2 = per-`T_eff` clamping of GEMM/epilogue/combine, plus zero-pad/dense-scatter variants.
- **Verdict — DESIGNED, NEVER BUILT OR RUN.** "no codegen evidence exists" (node-blocked, `nightshift/REPORT.md` §1). Modeled win **withdrawn**: fell from +31% to ≈+22% vs patched-stock (UNKNOWN vs native), then the corpus fill distribution (bimodal, mean fill 0.69, aggregate padding only **1.19×**) modeled to **+11–17% e2e**, of which **tier 1 alone captures essentially all** (tier-1-only ρ 0.719 beats full-M24 0.730). Dummy steps are **rank-synchronous** (bimodal on {0,8} for 100% of indices, κ=1.000 over 28 worker pairs) → tier-1 skip carries 1:1 to steps.
- **Supporting measurement:** **62.4%** of every MoE row the m15 arm computed was padding (stock 50.8%) — ≈65 s of a 217.6 s wall (`aug18-prefill/BOTTLENECK_SIGNALS.md`).

### A13. **SHEXP — shared-expert filler inside the M15 prefill mega** (`K0P6_M15_SHEXP`, default 0)
- **Path:** `ablations:distributed-kernels/fused_moe/overnight/aug18-prefill/SHARED_EXPERT_FILLER_DESIGN.md`.
- **Strategy:** the shared expert's 3-GEMM MLP over the same 4096 local rows has **zero dependency** on dispatch/plan/sort/peers, so its GEMM-1 tiles are scheduled into the **M0→M2 dispatch window** where 162 MiB of xGMI push is near fabric-bound and MFMA issue is idle. Today vLLM runs it serialized *before* the routed region.
- **Verdict — design only, no kernel edits in that commit.** Its founding assumption was later measured GO by the filler microcosm (§C3).

### A14. **fp8-on-wire combine + source-side pre-reduce**
- **Path:** `ablations:distributed-kernels/fused_moe/overnight/aug18-prefill/FP8_WIRE_DESIGN.md`.
- **Strategy:** halve combine wire bytes by pushing fp8 slabs; requires the staged arm (A5) as its base since gfx950 has **no fp8 remote RMW**.
- **Verdict — BLOCKED.** Unreachable from the mode-12 RMW epilogue by hardware; reachable only on M15b, which carries the P0 scratch/vmcnt pathology. Zero delta to the M7 instruction stream proved (282 `pk_add_bf16`, 96 `vmcnt(4)`, 180 `v_mfma` identical across builds).

### A15. **EPLB placement-only compose + RR expert placement** (host arms, no kernel change)
- **Paths:** `ablations:.../aug18-prefill/M15_EPLB0_COMPOSE.md` + `.../m15_eplb0/{m15_eplb.py,m15_eplb0.diff,eplb0_predict.py,m15_vllm.py,vllm_full.py}`; `ablations:.../aug18-prefill/rr_placement/RR_PLACEMENT_NOTES.md` + `.../rr_placement/test_rr_patch.py`.
- **Strategy:** RR placement is a **permutation of which logical expert sits in which physical slot**, not a routing change — physical ownership stays contiguous rank-major so MoRI/AITER/M15/ownership attestation are all unmodified. EPLB compose at `num_redundant_experts=0` needs no topk gather, no kernel change.
- **Verdict:** compose "sound, and cheaper than the brief assumed"; one correction — placement-only EPLB is **not** zero-memory (**≈1.31 GiB/rank** transfer buffer). RR patch: **built, unit-tested (140 checks green), NOT yet run on the node** — verdict unrecorded.

---

## B. Training backward megakernel lineage (`t2b` / `t2v6` / `t3` — v6.x)

Regime: **training**, DeepSeek-class MoE backward, FP8, per-iteration ms, 12-iter smokes and 260-iter full runs. Baseline = production FP8 (turbo_gg hybrid+delayed) **1,328 ms/iter, 24,667 tok/s/GPU**.

- **Ledger:** `ablations:distributed-kernels/fused_moe/overnight/aug18/V6_EVIDENCE_AND_DESIGN.md`; designs `.../aug14/{T1_TRAINING_SWAP_DESIGN.md,T2_BACKWARD_MEGA_DESIGN.md,T2_PHASE_MAP.md}`, `.../aug15/{OVERLAP_PROGRAM.md,T2B_MILESTONE.md,T4_BF16_VARIANT_DESIGN.md}`, `.../aug18/T3_COUNTER_DATAFLOW_DESIGN.md`.
- **Kernels:** `ablations:distributed-kernels/fused_moe/k0pf6gm_device_tile_t2b.hip`, `..._t2v6.hip`, `..._t3.hip`, `k0pf6gm_t2b_abi.hpp`, `k0pf6gm_t3_abi.hpp`, `k0pf6gm_t2b_shared.hpp`, hosts `n2_phase1b_gm_t2b.cpp`, `n2_phase1z_gm_t2b.cpp`, `n2_phase2b_gm_t2b.cpp`, `n2_wgrad_gm_t2b.cpp`.

| Arm | Strategy | Verdict (ms/iter) |
|---|---|---|
| **v3b baseline == t2v6 control** (bmm wgrad side-stream) | wgrad on a rocBLAS side stream | 1,670 inst / 1,689 avg — the control |
| **PROF arm** (`K0P6_T2V6_PROF`) | per-wait-site idle-cycle ledger, desc slot 73, brackets outside every poll | **cost-free** (1,673/1,689.7 vs 1,670/1,689) — the instrument shape reused by M25's E-B1 |
| **FILLER, pure in-kernel** (`K0P6_T2V6_FILLER`) | wgrad tiles executed inside the kernel's own wait windows (service slab 0, slab-1 post-quota, M8 certificate poll-and-fill, M8.5 cursor drain), two readiness-lattice cursors | **1,892 — LOSS (+222)** |
| **partition + monolithic 128-CTA s2 slice + fence** (E_m 2/4/8) | | **1,959–1,962 — worst** |
| **partition + dribbled small slices + fence** (the "v6.3 dribble" family) | small wgrad slices dribbled into a queue with fences | **1,988–1,993 — worst** |
| **partition + no fence** (dribble 8 / queue-all / 64-CTA fine) | fence removed | **1,954 / 1,930 / 1,927 — still all losses** |
| **FASTCVT** (`K0P6_T2B_FASTCVT`, in `n2_phase1b_gm_t2b.cpp`) | replace 192 emulated `__hip_cvt_fp8_to_halfraw`/lane with hardware `v_cvt_pk_f32_fp8` | **KEEP** — removed 16% of kernel `.text` (260,544 → 219,264 B); banked full run 1,663.1/1,684.9, loss 9.1648e-3 |
| **`K0_MEGA_WG_GRAPH`** (graph-capture the bmm block) | | **WASH** (1,685.7 vs 1,684) — host time is backpressure, not deletable glue |

- **Why every wgrad-rescheduling variant lost (root cause, stated in the ledger):** (a) in-launch bubble supply is **~150–180 CTA-ms vs ~3,360 CTA-ms of wgrad demand** — ~5%; (b) the s2 idle-CU pool is ~120 ms/iter and bmm's "hiding" was mostly its own host-time accounting; (c) **slice convoys block mega residency** where rocBLAS's sub-ms kernels slip through.
- **The governing law it produced (LAW-52):** iteration pace = the critical (hottest, lowest-clocked) rank's wall clock. Rank 2 is the structural straggler (M2 `rows_done` wait 92 µs on rank 2 vs 4,353 µs on rank 0 per CTA per launch). Modelled no-wgrad floor **≈1,473 ms**.

---

## C. TP8+EP boundary-collective rig (M25 / CDAR) — `distributed-kernels/tp8_mega`

Regime: **one TP8 layer boundary, prefill**, tokens=16,384 (**235 MB collective**), slab_rows=256, depth=4, bps=2, iters=7, 256 blocks, 8× MI350X gfx950. Single binary `m25_boundary_bench`; arms alternate **within one invocation** (LAW-60).

- **Rig source:** `ablations:distributed-kernels/tp8_mega/m25_boundary_bench.hip`, `ablations:distributed-kernels/tp8_mega/m25_cdar.cuh`, gate `ablations:distributed-kernels/tp8_mega/ledger_build_gate.sh`. Raw: `ablations:distributed-kernels/tp8_mega/results/{g25_0b_results.txt,g25_0b_slab_sweep.txt,g25_0b_v1_results.txt}`.
- **Design:** `ablations:.../overnight/aug18-prefill/M25_TP8_MEGA_DESIGN.md`, `.../TP8_OVERLAP_ANALYSIS.md`, `.../aug18-ablations/design/G_L4_INTEGRATION_MAP.md`.

### C1. The four G25-1 base arms (`compute` / `phased` / `fused` / `rccl`)
- **Strategy:** `compute` = MFMA floor, two bursts, no collective. `phased` = MFMA then full **CDAR** (custom counter-dataflow all-reduce: ERS epilogue commits + owner tower reduce + MAG pull) as a separate phase. `fused` = the M25 schedule — CDAR ERS commits **interleaved per slab** inside the matrix-core-active producer, owner/MAG overlap. `rccl` = MFMA kernel then `ncclAllReduce(bf16 sum)` on the **same stream** — production's schedule.
- **Verdict — the fused arm NEVER HIDES.** ρ ladder (`results/RHO_LADDER_G0A.md`, 48/48 PASS, medians of K=3):

| ρ_meas | k_inner | compute | phased | fused | rccl | h_wall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.631 | 2,048 | 1,405.6 | 3,634.6 | 3,622.5 | 2,908.7 | **+0.55%** |
| 0.941 | 3,078 | 2,178.3 | 4,493.6 | 4,462.3 | 3,769.6 | **+1.50%** |
| 1.786 | 6,029 | 4,421.7 | 6,879.0 | 6,864.1 | 5,938.0 | **+0.40%** |
| 2.383 | 9,040 | 6,557.2 | 9,331.7 | **9,398.8** | 8,405.2 | **−2.33%** |

  "Fusion hides nothing at ANY reachable compute intensity" — h_wall ≤1.9% in every repeat, **negative at the top rung with arms disjoint (≥33.7 µs)**. RCCL leads fused by **693–994 µs** at every rung. β = ours/RCCL 1.43–1.60, median **1.48**.
- **Four falsified fused sub-variants** (each encoded in the rig, `results/G25_1_STATUS.md`): v2 defer-drain/arrive-one-item-late — no change; v3 write-once-from-registers (kills the load→store re-copy) — no change; v4 duty-block consuming pool, 256 blocks, producer fallthrough — no change; v5 fragments-per-slab granularity at bps=16 — **worse (+180 µs of protocol)**, still no hiding.
- **Root cause, now isolated:** `transport_fused − transport = +15.0 µs` ⇒ the fused protocol's own overhead is **free**; the ~700 µs deficit is **entirely co-residency/scheduling failure**, not protocol.

### C2. G-L0b standalone arms (`transport`, `transport_fused`, `compute0`, `stdrccl`, `beta`)
- **Paths:** same rig; ledger `ablations:distributed-kernels/tp8_mega/LEDGER_NOTES.md` (compile receipts, zero spills, `.text` parity re-verified on gfx950), results in `.../aug18-ablations/results/FILLER_MICROCOSM.md` §6.
- **Strategy:** host-side aliases only (device `.text` byte-identical at `M25_LEDGER=0/M25_NOCOMPUTE=0`) — `transport` = phased with `k_inner` forced to 0; `stdrccl` = bare timed 235 MB `ncclAllReduce` in the same binary; `beta` alternates the two.
- **Verdict:** `stdrccl` **1,320.9 µs**, `transport` **2,019.3**, `transport_fused` **2,034.3**, `compute0` **132.0**. **Direct β = 1.529** (B4 closed — three methods agree: 1.506 subtraction, 1.43–1.60 ladder, 1.529 direct). Our CDAR transport is ~1.5× RCCL's. **I_co is not constant**: +209.7 µs at ρ=0.63 → **+755.2 µs at ρ=2.38 (3.6× growth)** — the "+335 µs constant tax" is retired.
- **Ops rule banked:** never quote `wall_us_max` on the `rccl` arm — first-iteration RCCL channel setup is **15.8× the median** (45,719 vs 2,893 µs).
- **New unexplained mechanism (Q10c):** the sequential transport phase costs **+23.4% more after a 4.7× longer compute phase** (2,229 → 2,751 µs) at flat clocks (0.45% spread, ≤62 °C — not thermal).

### C3. **FILLER MICROCOSM** (two-process co-residency proxy)
- **Path:** `ablations:distributed-kernels/fused_moe/overnight/aug18-ablations/results/FILLER_MICROCOSM.md`.
- **Strategy:** two separate processes on the same 8 GPUs — one bare `ncclAllReduce` (235 MB), one full-GPU `mfma_f32_16x16x32_bf16` loop at occupancy 1. Two nested configurations so each leg gets a fully-overlapped measured window.
- **Verdict — GO, the load-bearing enabling result.** ΔAR = **0.00%** (worst repeat −0.38%; bar was <10%). Filler retains **φ = 43.9–53.4%** of solo throughput during the collective ⇒ **+571 to +695 µs absorbed free per AR** against a ≥25% bar. **Mechanism card: a 235 MB `ncclAllReduce` on this fabric is NOT CU-bound.** Design consequence: **evaluate the two-stream filler (Option A) before the in-kernel variant.**

### C4. **G-L1/G-L2 champion-candidate arms** (`-DM25_G1=1`, `ablations` HEAD `6fcce244`)
- **Path (all five):** `ablations:distributed-kernels/tp8_mega/m25_boundary_bench.hip` — dispatch table at lines 887–891, docs at 46–68. argv 8 = `chunks`, argv 9 = `k_filler` (auto = `k_inner/128` = exactly a 1/8 MFMA-iteration budget = the true shared-expert:routed FLOP ratio). Default device image unchanged; all host-side.

| Arm | Overlap / data-movement strategy | Verdict |
|---|---|---|
| **`rcclserial`** | burst1 → shared-expert filler **on the critical path** → monolithic 235 MB AR → burst2. The **production-structured baseline**: vLLM takes `SharedExpertsOrder.NO_OVERLAP` on ROCm, so the shared expert is serialized in front of the routed region every layer. | **verdict unrecorded** (arms committed `6fcce244`, sibling running from the flag-OFF build; "ledger cells need a rebuild from this commit") |
| **`rcclchunk`** | the `rccl` schedule with the 235 MB AllReduce split into `chunks` equal token-chunks, one `ncclGroup` per chunk, **no filler**. Isolates the **pure chunking tax** (fabric + per-chunk enqueue) — the premise the whole design rests on. | **verdict unrecorded** |
| **`rcclhybrid_indep`** | chunked AR + filler on a **second stream** with **no dependency** on the AR. Pure overlap — the **upper bound**. | **verdict unrecorded** |
| **`rcclhybrid_merged`** | chunked AR + filler on a second stream, **filler chunk i gates AR chunk i via an event** (per `G_L4_INTEGRATION_MAP.md` §1.5: AR#1 → MoE → AR#2 is strictly serial, so the only dependency-free co-resident work requires the collective to be token-chunked). Filler chunk 0 exposed by construction; chunks 1..N−1 hide under AR chunks 0..N−2. **This is the champion candidate.** | **verdict unrecorded** |
| **`filleronly`** | the filler alone — the **`k_filler` calibration arm**. | **verdict unrecorded** |

- **Prior expectation from the ledger:** G-L2's bar is "hides ≥25% of the G-L0-measured exposed AR"; the microcosm cleared it at 43.9–53.4% in proxy form. Exposed RCCL AR to attack = **1,488–1,925 µs per boundary**.

### C5. E-B1 / E-B2 device phase ledger (`-DM25_LEDGER=1`)
- **Paths:** `ablations:distributed-kernels/tp8_mega/m25_cdar.cuh` (phase codes `mfma1/mfma2/commit/pull/gather_wait/mag_wait/drain/reduce/certify`), `ablations:distributed-kernels/tp8_mega/LEDGER_NOTES.md`.
- **Strategy:** per-CTA raw event ring in the 100 MHz realtime domain; all reduction host-side after join. `h_ledger = |U_compute ∩ U_transport| / |U_transport|`; fabric duty histogram; explicit cross-rank RANKMAX.
- **Verdict:** built to **zero spills / zero scratch**, `.text` parity re-verified on gfx950 (`f4174a8f`). Four whole-loop phase codes (`prod_loop/tx_loop/duty_loop/cons_loop`) were built, measured, and **removed** — they cost SGPR spills in a kernel already at 105 SGPRs. **Not yet run** (node-blocked, R7 harness patch blocked on operator action).

---

## D. Serving-integration arms (vLLM 0.25.1+rocm, campaign v5.1)

### D1. Retracted-serving-methodology block (read before quoting ANY e2e number)
- **Paths:** `ablations:docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md`, `ablations:distributed-kernels/fused_moe/overnight/aug14/M20_SERVING_RESULTS.md` (tombstone), `ablations:.../aug14/M21_OVERLAP_DIRECTION.md` (retraction note).
- **Verdict:** **every** pre-2026-08-18 e2e serving A/B is invalid — inert candidate (seal fired on ~2% of padded-4096 heavy steps), de-graphed candidate on unsealed steps, and a baseline depressed by uniform-decode ranks running the whole model eagerly. Retired numbers include the m15/m18/m19/m20 pairs and M20's "−19.6%". **Kernel-level MoK numbers are unaffected.**

### D2. Campaign v5.1 arms
- **Paths:** `ablations:.../aug18-prefill/campaign_v5/ARMS.md` (+ `run_m15_campaign_eplb_v5.sh`, `analyze_campaign_v5.py`, `bench_exact_token_ids_v3.py`, `upstream/camp5_native.sh`).
- **Arms:** `m15` (candidate, `VLLM_PF4H_INTEGRATION_MODE=m15`, MoRI high-throughput all2all, TP=1/DP=8/EP), `stock` (patched-stock control — denominator of the **kernel** claim only, "never quote as production"), `native_default` (**the production headline baseline**, untouched vendor image), plus AMD-recommended TP8+EP and a deterministic accuracy cell.
- **Verdict — SPECIFIED, NOT RUN.** "camp5_native never ran"; **no e2e ratio vs native production exists, tonight or ever** (`nightshift/REPORT.md` M9). camp3 order-balanced n=2 gives mean **+2.70% / geomean +3.61%** with a ±18% position effect — **DIRECTIONAL ONLY**, below the binding ≥5-pair floor.

---

## E. GEMM-RS (adjacent collective kernel, `GEMM-RS` branch)

- **Paths:** `GEMM-RS:distributed-kernels/gemm_rs/{gemm_rs_device_tile.cpp,gemm_rs_dprime_experimental.cpp,gemm_rs_mi300x.cpp,gemm_rs_mi300x_hk_adapter.cuh,gemm_rs_mi300x_simulation.py}`, `GEMM-RS:distributed-kernels/gemm_rs/overnight/RESULTS.md`, `.../MI300X_{DESIGN,VALIDATION,PROVENANCE}.md`.
- **Strategy:** epilogue reduce-scatter (ERS) — the GEMM epilogue carries the reduce-scatter directly, owner-major staggered order, owner consumes in shadow. This is the prototype for M25's ERS knob point.
- **Verdict — FAST vs the GEMM+RCCL reference, still behind leaderboard rank-1.** ours/reference **0.7641** [0.7268, 0.8033], ours/rank-1 **1.1203** [1.0795, 1.1628] over 5 order-balanced sessions both orders. Evaluator readings ours 397.2 / rank-1 335.5 / reference 489.8. **376.7 µs vs RCCL 421.8 µs** on a comm-dominant shape (quoted in M25 design). `COMMIT_MID` mainloop arm: KEEP at −0.88% GM (s6 −2.90%). Contamination finding: exp_24's ours arm ran with `HK_DEBUG=1` inside the timed region — a **−31%/+44% debug tax**. Regime: **gfx942 MI300X SPX**, world-8, 17/17 shapes correct.

---

## Most evidence-dense ledger files

1. `ablations:distributed-kernels/fused_moe/overnight/aug18-ablations/results/RHO_LADDER_G0A.md` — the ρ ladder, 48 runs, h_wall table, the β band, the RCCL-outlier ops rule.
2. `ablations:distributed-kernels/fused_moe/overnight/aug18-ablations/results/FILLER_MICROCOSM.md` — the G-L2 GO, ΔAR=0.00%, φ=43.9–53.4%, direct β=1.529, absolute I_co.
3. `ablations:distributed-kernels/fused_moe/overnight/aug18-ablations/UNDERSTANDING_PLAN.md` (§8 results ledger) — every G-L gate verdict with commit shas.
4. `ablations:distributed-kernels/fused_moe/overnight/aug12/exp_03_m15_slab_combine/result.md` — M15's full C-response curve, phase attribution, and the m15b/m17 adjudications in one table.
5. `ablations:distributed-kernels/fused_moe/overnight/aug14/M18_REPLICATION_RESULTS.md` — the skew measurement, the replay rig, and the whole replication ladder with CIs.
6. `ablations:distributed-kernels/fused_moe/overnight/aug18/V6_EVIDENCE_AND_DESIGN.md` — the training bubble ledger, every falsified wgrad-filler arm, LAW-52.
7. `ablations:nightshift/REPORT.md` — M1–M12 measurement table with validity classes + seven retractions; the single best "what is and isn't proven" document.
8. `ablations:distributed-kernels/tp8_mega/results/G25_1_STATUS.md` — the four falsified fused-CDAR hypotheses and the K1/K3/K4 knob results.
9. `ablations:distributed-kernels/fused_moe/overnight/aug18-ablations/design/OVERLAP_ATLAS.md` — the eight feasibility screens, the four-door occupancy-1 rule, and its decode corollary; the framework every verdict above is scored against.
10. `ablations:distributed-kernels/fused_moe/overnight/aug12/STATUS.md` — the aug12–13 tournament tables (production → pf6gm → mode-12 → mode-16 → M15) in one place.
11. `ablations:distributed-kernels/tp8_mega/LEDGER_NOTES.md` — the instrument's construction, the five LAW-63 guards, and the calibrated ρ↔`k_inner` map.
12. `distributed-kernels/fused_moe/overnight/aug18-prefill/BOTTLENECK_SIGNALS.md` *(committed on this branch)* — the three-estimator composition correction showing the mega is **not** losing in serving (r ≈ 0.95, [0.78, 1.06]) and the 62.4%-padding finding.

Also worth carrying for law-lookup: `ablations:.../aug18-ablations/grounding/BANKED_LAWS.md`, `.../grounding/TP8_STATE.md`, `.../grounding/KNOB_INVENTORY.md`, and `ablations:docs/distributed/OVERLAP_ABSTRACTIONS.md` (the five-knob model).

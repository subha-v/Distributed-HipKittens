# M20 serving integration — handoff packet (2026-08-14, ~21:10 UTC)

**Mission:** integrate the M20 kernel (replica cache + per-chunk adaptive
routing) into real DeepSeek-R1 vLLM serving and produce ONE receipt-gated
stock-vs-m20 pair on realistic MLPerf prompts (c32p cell), at a fraction of
M19's 41 GB memory cost.  This packet is self-contained.

> **Note (2026-08-18):** the end-to-end serving targets this packet was
> originally written against were removed as obsolete — the megakernel's
> activation seal fired on only ~2% of padded-4096 heavy steps, the candidate
> arm ran with no cudagraph on the rest, and both arms' baselines were
> depressed by a uniform-decode rank running eagerly. The integration
> mechanics, ground truth, and gotchas below are unaffected and still current.
> Any new pair must satisfy
> `../../../../docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md` (M23 patch
> chain in both arms, `RAGGED_SEAL_RECEIPT` coverage quoted, rescued-stock
> baseline, ≥5 order-balanced pairs).

## 1. What M20 is (one paragraph)

M15's fused megakernel (0.756x production, balanced) + source-local expert
replication (the M18 mechanism: replicated experts' tokens are computed by
the rank that owns the tokens — never cross the fabric; exactly-once via a
self-source acceptance rule; combine untouched) + a per-chunk decision
bitmap COMPUTED BEFORE LAUNCH (a bincount+threshold op on the router
output; deletes M19's in-kernel pre-pass and its measured 1.2 ms
grid-barrier tax) + replica weights in a **persistent per-layer cache for a
host-budgeted subset of layers** (~0.7 GB per cached-layer-set slice; NOT
41 GB of all-layer copies, and NOT per-step streaming — measured infeasible
on xGMI: pull ~7 GB/s latency-bound, push capped by the ~1.4 GB/s/CTA
service-pool law from exp_03_push_throughput).  Kernel-level, 5-run
campaigns, all gates green: **0.2483x production under measured serving
skew; 0.823x balanced at theta=64** (theta above the uniform count returns
~0.76; in serving the pre-op computes bits per chunk so this is automatic).

## 2. Ground truth: what runs where

- **Kernel** (Distributed-HipKittens, `github.com/subha-v` remote `origin`):
  `ablations` HEAD holds `K0P6_M20_SLOTPOOL` (requires REPLICATE+ADAPTIVE)
  in `distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip` plus
  dual-base weight addressing in `n2_phase{1,2}_gm_mps.cpp` (slot >= 32
  reads pool bases from descriptor slots 66..69 via the threaded
  `k0p6_desc`).  RUN PIN branch `ablations-m20` (kernel name
  `k0pf6gm_mps_mega`, for the MoK harness only).  Descriptor (M20 arm) = 71
  words: 63 M18R table (lut+bitmap, theta in header word 3), 64 rep_dec
  (symmetric, per ring slot), 65 DECIN (8xu32 per-chunk decision bitmap,
  written pre-launch), 66..69 pool W13P/S13P/W2P/S2P bases
  (parity-free in cache mode — one persistent buffer per cached layer),
  70 prefetch duty table (**nullable**; cache mode passes 0/empty —
  entry guard does NOT require it).  `K0P6_M20_PF_CTAS=8` service CTAs
  no-op on an empty table.
- **Serving shim** (amd-master, remote `github`): branch
  `vllm-integration-m18`, worktree `~/amd-master-m15pkt`.  Already contains
  the FULL m15/m18/m19 serving integration (all receipts-gated):
  `shim/pf4h_integration/{m18_replication.py,
  m15_runtime.py, m15_contracts.py, m15_kernargs.py, m15_sources.py,
  weights.py, apply.py, m15_pin/}` + campaign arms in
  `benchmarks/2026-08-12_m15_campaign/run_m15_campaign.sh`.  m19's serving
  path (65-word descriptor, VLLM_PF4H_M19_THRESHOLD, per-layer sets JSON
  via VLLM_PF4H_M18_REP_EXPERTS) is the direct template for m20.
- **Harness** (amd-master `debug/pf6-first-launch`, worktree `~/amd-master`):
  M20 support behind `K0_MOK_M20=1` — read
  `prefill_opt/host/e004pf_k0pf_ab.py` (search `_m20`) for the working
  reference implementation of: symmetric pool carve, host pre-warm from
  owner weights, DECIN construction, descriptor slots 65..70, and the
  duty-table builder (push-direction; duty 0 in cache mode).
- **Node** (8x MI350X, exclusive): `ssh -p 2425 subvadla@10.5.95.87`.
  Worktrees `~/DHK-m20` (harness pin), `~/DHK-m19pkt` (serving DHK pin for
  m19-era deploy set — a NEW serving deploy set must be staged at the M20
  kernel commit).  Deploy sets: `~/m19_deploy_sources_20260814` (m19-era).
  Campaign results: `~/k0-mok-synthetic-results/m20*_0814T2059/` (the final
  numbers), serving pairs `~/20260814_m15_campaign_{m18pair2,m19pair1}/`.
  Per-layer sets file: `m18_layer_sets.json` in the deploy dirs (58
  distinct top-16 sets from measured histograms).

## 3. The integration plan (m19's steps, plus the cache specifics)

1. **Pin + tables**: add `m15_pin/k0pf6gm_m20_mega.hip` (copy the m19 pin,
   add `#define K0P6_M20_SLOTPOOL 1`, symbol `k0pf6gm_m20_mega`); bump
   `m15_sources.py` (M15_DHK_COMMIT -> the current `ablations` HEAD, the
   changed SHAs for `k0pf6gm_device_tile_m15.hip` AND BOTH
   `n2_phase{1,2}_gm_mps.cpp` — the phase files changed for M20, unlike
   m18/m19!), add the pin to M15_NATIVE_MODULES + M15_PACKET_SOURCES;
   `m15_kernargs.py` M20Mega schema (3-arg ABI, pin after first measure);
   restage the deploy dir from a fresh DHK worktree at that commit and
   preflight-compile ALL pins in `subvadla_m15pkt` (pattern:
   `/tmp/preflight_m19_all.py` on the node).
2. **Runtime mode**: in `m15_runtime.py`, an M20 mode selected by e.g.
   `VLLM_PF4H_M20_BUDGET_LAYERS` (csv of model layer ids, or an int budget
   the host resolves to the most-damaged layers — per-layer residuals are
   computable from the banked histograms; see
   `make_route_hist.py`/`perlayer_sets.py` in /tmp on the node or rewrite).
   descriptor_words=71, kernel `k0pf6gm_m20_mega`.
3. **The cache = m18's extended weights, budgeted**: REUSE the existing
   `extend_shuffled_weights` owner-broadcast path (weights.py →
   m18_replication.py) but ONLY for budgeted layers; pool base slots
   66..69 = the extended allocation's replica region (`ext.data_ptr() +
   E*per_expert_stride` for each tensor — the replica slots ARE the cache;
   no separate pool allocation needed serving-side).  Uncached layers: two
   options — (a) per-layer kernel choice (load BOTH m15 and m20 megas;
   uncached layers launch m15 with 63-word descriptors — cleanest), or
   (b) m20 everywhere with a shared dummy pool + all-zero DECIN (guard
   needs nonzero pool pointers; bits 0 means the pool is never read).
   (a) is recommended; the runtime already loads module lists per mode.
4. **DECIN pre-op**: per routed layer per step, between router and mega
   launch (M15PrepareAndFinalize.prepare is the seam): bitmap =
   (lut[e] >= 32) && (bincount(topk_ids)[e] >= theta), written into the
   layer's persistent DECIN buffer.  Must be graph-capturable: implement as
   torch ops on-stream (bincount + comparisons + a packed-bit reduction, or
   a tiny custom kernel).  theta env: reuse `VLLM_PF4H_M19_THRESHOLD`
   (serving-validated at 64).  NOTE the rep_dec publication/acceptance in
   the kernel is unchanged from m19 — nothing to do there.
5. **Campaign arm `m20`** in run_m15_campaign.sh (copy the m19 case: same
   install, sentinel, receipts + M18_REPLICATION_RECEIPT), then ONE pair:
   `--arms stock,m20 --cells c32p --pairs 1` via a driver script modeled on
   `~/m19pair.sh` (fresh RUN_TAG, M15_SOURCES=new deploy dir, DHK_ROOT=new
   serving worktree).
6. After the pair: the 500-sample AccuracyOnly A/B (mlperf_v6 harness — see
   `OFFICIAL_AB_PROTOCOL_20260730.md`; calibration pkl
   `~/mlperf_v6_datasets/mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl`)
   and >=5 order-balanced pairs before quoting numbers.

## 4. Hard-won gotchas (each cost real time today)

- **SHA pins**: any DHK kernel-file change breaks
  `validate_m15_deploy_sources` for EVERY arm sharing the install (the
  stock arm too).  Bump m15_sources tables + restage + move the DHK
  worktree HEAD together, and never mid-pair.
- **The campaign GPU lock** (`/tmp/k0_mok_synthetic_gpu_lock`, and the
  serving campaign's own claim): NEVER clear it without verifying no
  containers are live (`docker ps`); concurrent MoRI inits fail with
  `symmetric_memory.cpp ... invalid argument`.
- **mori heap**: 32 GB (`MORI_SHMEM_HEAP_SIZE`), but keep single symmetric
  allocations well under ~500 MB each (the 3x672 MB pool carve failed).
- **KV-pool math**: budgeted cache memory comes out of vLLM's KV pool
  (gpu-memory-utilization 0.70).  M19's 41 GB halved it; a ~4–8 GB budget
  is the sane serving default.  Record `Available KV cache memory` per arm.
- Node ssh drops transiently (2–10 min, ~3x today); detach everything with
  `setsid nohup`, poll files, never depend on a live session.
- Campaign teardown prints a cosmetic `so: command not found` (RC=127)
  after `campaign complete` in some paths — results are still valid; grep
  for receipts + c32p.json rather than trusting RC alone.
- `pgrep -f <pattern>` matches your own wrapper — add `| grep -v $$` or
  check containers instead.
- Per-query latency keys in c32p.json: `ttft_ms`, `tpot_ms`, `e2el_ms`
  under `per_query`; throughput under `result`.

## 5. The bar

M20 carries M19's routing machinery minus the per-layer 1.2 ms in-kernel
decision tax (× 58 layers ≈ 70 ms/step) at a small fraction of the memory, so
it should beat M19 structurally at equal coverage.  The end-to-end bar M19
originally set was retired on 2026-08-18 as obsolete (see the note under
**Mission** at the top of this packet);
a new bar has to be re-measured under
`../../../../docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md`.  Discipline:
stock control first, receipts before activation, coverage receipt with every
mega number, every number from an artifact, commit and push as you go (no
force-push, no co-author lines).

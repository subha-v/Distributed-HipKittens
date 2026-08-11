# Fused-MoE benchmarking — MoK campaign, arms, and the MPS sweep

This is the missing half of the handoff. `BUILDING.md` says "same-run paired
timing versus both the parity port and production" and `README.md` says "sweeping
`C × g × flush_rows`"; neither names the harness that produces such a number.
This page does.

Everything below runs on the **8× MI350X (`gfx950`) node**, inside the
`amd-master` checkout — the harness lives there, not in this repository. Nothing
here has been executed by the authoring agent; this is a plan with exact paths,
not a result.

---

## 1. The harness

```text
amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/
  run_campaign.sh      launcher (lease, idle gate, container, rotation, summarize)
  correctness.py       MoK error stats + pass predicate
  synthetic_inputs.py  MoK-policy input generation
  summarize.py         per-arm p50/p95 + candidate/production ratios
  CMakeLists.txt       pybind modules built into a cache outside the source tree
```

The arm dispatcher it launches is
`amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py`
(`run_campaign.sh:145`). **Arm names are registered there, not in the benchmark
directory.**

This is the same campaign that produced every number in `PROVENANCE.md`. The
`6,919.8 µs vs 7,698.0 µs production (0.89846x)` figure at `PROVENANCE.md:20` is
a `production` vs `pf6gm_mega` MoK campaign result, and `"MoK passed"` at
`PROVENANCE.md:22` means `correctness.py::mok_error_passes` returned true.

### Protocol (from `run_campaign.sh:111–126`)

| Property | Value |
|---|---|
| Input mode | `K0_INPUT_MODE=mok_synthetic` |
| Timing protocol | `K0_BENCHMARK_PROTOCOL=mok_eager` (eager, HIP events — **not** graph replay) |
| Warmup | 500 iterations (`K0_MOK_WARMUP_ITERS`) |
| Timed | 100 iterations (`K0_MOK_TIMED_ITERS`) |
| Seed | `1234 + rank`, one device generator per rank (`K0_MOK_SEED_BASE`) |
| Rank aggregation | index-aligned `MAX` across all 8 ranks, per iteration |
| Primary statistic | **median of the 100 rank-max samples**; `p95` also reported |
| Arm ordering | rotated one position per run (`run_campaign.sh:83–89`) |
| Correctness gate | MoK MXFP8 policy: global L1 relative error ≤ 0.1, max abs ≤ 1.0, zero nonfinite. Blocking, per arm, every run |
| Strict k0 gate | `rel_L2 ≤ 0.01` — recorded as a **nonblocking diagnostic**, never as a gate |
| Reference | a fresh same-run `production` output, not a stored corpus |
| Timed region | setup, generation, compilation and correctness are all outside it |

### Shape (fixed, `run_campaign.sh:114–120`)

TP1/DP8/EP8 DeepSeek-R1 prefill: `T=4096` tokens/rank, hidden 7168, intermediate
2048, 256 global experts (32/rank), top-k 8, `MAXTOK=4096` for candidate and
production alike, `BLK=128`, `WARP=16`, `PADMAX=263136`, `T_LOC_MAX=40960`.

This is a **synthetic eager diagnostic**. Independent normal router logits make
expert selection roughly uniform and produce more destination-rank fanout than
DeepSeek's grouped router. It is not evidence of production-route performance,
graph-mode performance, or a serving win. Label every result accordingly.

### Safety interlocks

`run_campaign.sh` refuses to run if `torchrun`/`mpirun`/`orterun` is alive, if
fewer than 8 GPUs report 0% use, if the advisory lease `/tmp/k0_mok_synthetic_gpu_lock`
is held, or if the output directory already holds a completed campaign. Do not
defeat these; a contended node invalidates the ratio.

---

## 2. The three arms

| Arm | What it is | Registered at |
|---|---|---|
| `production` | MoRI EpDispatch → AITER `fmoe_fp8_blockscale_g1u1` → MoRI EpCombine | mandatory — `run_campaign.sh:30` rejects any CSV without it |
| `pf6gm_mega` | **past-best MoE megakernel** — the exp_59/exp_64 G-stacked expert-tile donor this port descends from | `e004pf_k0pf_ab.py:230` (`PF6GM_ARM_NAMES`) |
| `mps_mega` | this repository's MPS sibling | **not yet registered — see §4** |

`summarize.py` is arm-generic: it emits `arm_p50_us` / `arm_p95_us` for every
arm and `candidate_ratios[arm]{p50,p95}` against `production` for every
non-`production` arm (`summarize.py:136–152`). Three arms in one campaign
therefore yields both comparisons you want, measured in the same process on the
same generated tensors, with no cross-run stitching.

### ⚠️ Trap: the reference arm defaults to G=2, and the campaign will not forward G=3

`e004pf_k0pf_ab.py:241` sets `PF6GM_G = int(os.environ.get("K0_PF6GM_G", "2"))`,
and its own comment states that **`K0_PF6GM_G` is not forwarded by the frozen
`run_campaign.sh`**, so the container build is governed by the source default.

This port is `K0P6GM_G=3`. `PROVENANCE.md:18–24` records the best configuration
as G=3/c4 at `amd-master` commit `d34e5510c4672e998eb2e94672e46e2f784ca07f`
(6,919.8 µs), while G=2/c4 measured 7,248 µs (`PROVENANCE.md:12`). Running the
frozen campaign unmodified compares a G=3 candidate against a **G=2** reference
arm — a ~330 µs handicap in the candidate's favour, and an invalid claim.

Before timing, do one of these and record which:

- forward `K0_PF6GM_G=3` into the container (add one `-e` line; note the edit in
  the run log), or
- run `pf6gm_mega` at both G=2 and G=3 as two separate campaigns and report both.

`e004pf_k0pf_ab.py:236` also requires that `PF6GM_G` match `K0P6GM_G` in the
`.hip` source and `N2GM_G` in the `_gm` bodies. Assert this at startup; a silent
mismatch is a wrong-kernel measurement, not a crash. The same comment warns the
fused G=3 build spills (7 VGPR / 32 B scratch); `PROVENANCE.md:23` records 9 VGPR
spills / 40 B scratch for the measured G=3 best. Spilling is expected at G=3, not
a build failure — but all spill traffic must stay outside both MFMA K-loops.

---

## 3. Gate order — nothing is timed before all of this passes

From `BUILDING.md:107–125` and `README.md:90–98`, in order. Each MPS mode is a
separate arm-config and repeats the correctness/control/soak gates.

1. **Build** — gfx950, `K0P6GM_G=3`/`N2GM_G=3`, fresh JIT cache root. One HSACO
   serves the entire `C × g × mode` sweep; do not let the cache alias the parity
   port's object.
2. **Resource/ISA A/B vs the parity port**, both MFMA spans: 96 MFMAs per
   phase-1 K-loop; the task-done `s_waitcnt vmcnt(0)` drain and the hook land
   **outside** both K-loops; no ArchVGPR/AGPR/spill/LDS movement (the loop-carried
   role word is CT-uniform → SGPR expected); exactly six `k0p6_symmetric(desc)`
   sites; `store_peer_packets` lowers to 16-byte stores.
3. **World-8 correctness** — MoK gate blocking, strict k0 gate diagnostic.
4. **Negative control** must fail as required.
5. **600-epoch soak**, `pperr=0` throughout. Any nonzero `pperr` is terminal
   (`BUILDING.md:71`): clearing the error word alone is invalid, because
   same-epoch publications from the failed attempt can satisfy a new wait
   prematurely.
6. **Mode ladder**, in this order — mode 0 (reserved-CTA tax control, `g=4`) →
   mode 1 (push layout, no overlap) → mode 2 (full stream).
7. **Paired timing** — only now, and only against `production` **and**
   `pf6gm_mega` in the same campaign.

`pull_fallback` (cfg bit 32) isolates streamed readiness from push transport;
run it before trusting any mode-2 win. Timestamps (cfg bit 33) are for
attribution runs only — leave them **off** for the scored production-shape
timing.

---

## 4. Registering `mps_mega` — the cross-repo work item

The arm does not exist yet. It must be added to
`amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py`,
mirroring the `pf6gm` blocks:

1. `PF6MPS_ARM_NAMES = ("mps_mega",)` + the `K0_ARMS` filter, next to
   `PF6GM_ARM_NAMES` at line 230.
2. Fold `PF6MPS_REQUESTED` into `PF6_REQUESTED` (line 243) so the shared PF6
   buffer/N2 machinery is built when the arm is requested.
3. `PF6MPS_NAME = "k0pf6gm_mps_mega"` alongside `PF6GM_NAME` (line 274) — this
   must equal the entry point in `k0pf6gm_device_tile_mps.hip`.
4. Add `n2_phase2_gm_mps.cpp` to the conditional source tuple `PF6_N2_FILES`
   (line 294), gated on `PF6MPS_REQUESTED`, the way the `_gm` bodies are.
5. A JIT/source-install + load block modelled on `R["pf6"]` (line 758), with its
   own cache key and its own `source_sha256` / `hsaco_sha256` recording.
6. Allocate and zero the MPS buffers, sized by the formulas in
   `moe_host_abi.hpp:166–193`: `mps_queue_bytes(padmax)`,
   `mps_arrivals_bytes(t_ext)`, `mps_pushed_bytes(t_ext)`,
   `mps_claim_bytes(t_ext)`, `mps_state_bytes()`, `mps_slots_bytes(maxtok)`.
   The **slots buffer must be symmetric** (peer-written); queue/counter/state are
   agent-local. Then `append_mps_descriptor` / `patch_mps_slots` +
   `validate_mps_descriptor` for descriptor slots 56..62, and word 55 via
   `append_symmetric_heap_descriptor` (`BUILDING.md:98–105`).
7. Config word from `K0_MPS_CFG` — see §5.

### ⚠️ Trap: the sweep must not be encoded as arm names

`run_campaign.sh:83–89` rotates over the arm list, so a `C × g × mode` sweep
expressed as arm names becomes a 20+ arm rotation: unusable, and it would put
sweep points in different positions of different runs.

One HSACO already serves the whole sweep through the descriptor word, so:

- **one** arm name, `mps_mega`;
- add a forwarded env var `K0_MPS_CFG` (decoded through
  `hk_moe::mps::encode_config(C, g, mode, flush_rows, pull_fallback, timestamps)`,
  `moe_mps_adapter.cuh:61`);
- **one campaign invocation per sweep point**, always
  `K0_MOK_ARMS=production,pf6gm_mega,mps_mega`.

Every sweep point is then paired against both references inside the same run.
That is the property that makes the ratios comparable across sweep points; a
cross-campaign comparison of two `mps_mega` configs is not.

Config field bounds are enforced by `config_is_valid`
(`moe_mps_adapter.cuh:84`): `C ≤ 64` and `C < 256`, `g ∈ {1, 2, 4, 16}`,
`mode ≤ 2`. `flush_rows` defaults to 16, range 1..64.

---

## 5. Commands

Run from `amd-master/auto-gpu-kernel/k0_fused_moe`. The image must provide ROCm,
PyTorch, AITER and MoRI; the established one is `rocm/atom-dev:vllm-latest`.

**Reference campaign first — establish the two baselines with no candidate.**
This is also the regression check that the harness still reproduces
`PROVENANCE.md`:

```bash
K0_MOK_ARMS=production,pf6gm_mega \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh mps_ref_g3 5
```

Expect `pf6gm_mega/production_p50` ≈ 0.898 if G=3 was forwarded, ≈ 0.941 if it
was not. **That single number tells you which reference you actually built** —
check it before running any candidate.

**Mode ladder** (each is a full 5-run campaign; `C`/`g`/`mode`/`flush_rows`
packed into `K0_MPS_CFG`):

```bash
# mode 0 — reserved-CTA tax control, g=4, C sweep
for C in 4 8 16; do
  K0_MOK_ARMS=production,pf6gm_mega,mps_mega \
  K0_MPS_CFG="C=${C},g=4,mode=0,flush_rows=16" \
    bash benchmarks/mok_synthetic_prefill/run_campaign.sh mps_m0_c${C} 5
done

# mode 1 — push layout, no overlap
for C in 4 8 16; do
  K0_MOK_ARMS=production,pf6gm_mega,mps_mega \
  K0_MPS_CFG="C=${C},g=4,mode=1,flush_rows=16" \
    bash benchmarks/mok_synthetic_prefill/run_campaign.sh mps_m1_c${C} 5
done

# mode 2 — full stream, C x g
for C in 4 8 16; do for G in 1 2 4; do
  K0_MOK_ARMS=production,pf6gm_mega,mps_mega \
  K0_MPS_CFG="C=${C},g=${G},mode=2,flush_rows=16" \
    bash benchmarks/mok_synthetic_prefill/run_campaign.sh mps_m2_c${C}_g${G} 5
done; done

# pull_fallback isolation at the best mode-2 point
K0_MOK_ARMS=production,pf6gm_mega,mps_mega \
K0_MPS_CFG="C=<best>,g=<best>,mode=2,flush_rows=16,pull_fallback=1" \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh mps_m2_pullfb 5

# attribution only — timestamps on, NOT a scored number
K0_MOK_ARMS=production,pf6gm_mega,mps_mega \
K0_MPS_CFG="C=<best>,g=<best>,mode=2,flush_rows=16,timestamps=1" \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh mps_m2_ts 5
```

Results land in `${K0_MOK_OUTPUT_ROOT:-$HOME/k0-mok-synthetic-results/<tag>}`:
eight rank JSONs and a log per run, plus `summary.json` for the campaign. Raw
per-rank and aligned rank-max samples are retained for audit.

---

## 6. Reading the result

From `summary.json`, per campaign:

- `arm_p50_us["production"]`, `["pf6gm_mega"]`, `["mps_mega"]`
- `candidate_ratios["mps_mega"]["p50"]` — vs production
- `candidate_ratios["pf6gm_mega"]["p50"]` — vs production
- `mps_mega / pf6gm_mega` — **the decision number**, computed from the two p50s

MPS is worth keeping only if it beats `pf6gm_mega`, not merely `production`.
Beating production is already done: `PROVENANCE.md:20` records `0.89846x` for the
donor this port descends from.

Interpretation against the cost model (`DESIGN_MPS.md:115`): reserving `C` CTAs
at M6-end costs `1844.5 · 256/(256−C) − 1844.5` µs of M7 — about 60 µs at C=8.
Break-even against the ~217 µs campaign floor needs ≳ 277 µs of recovered tail,
i.e. under half of the estimated 620–769 µs remote-`part` transport, at a
required sustained push rate of only ~148 GB/s. If mode 0 (pure tax, no benefit)
already costs more than the mode-2 gain, the experiment is answered negative and
the ladder stops there.

Report every arm's p50 **and** p95, the per-run spread across all 5 rotations,
and the correctness/control/soak status per arm. A ratio without its rotation
spread is not a result.

---

## 7. What this does not measure

- **Not graph mode.** `mok_eager` is eager. Graph-replay behaviour is a separate
  question and this campaign says nothing about it.
- **Not the production route.** Synthetic uniform routing, more fanout than the
  real grouped router.
- **Not serving.** The end-to-end judge is MLPerf v6.0 LoadGen against live vLLM
  (`k0_fused_moe/harnesses/mlperf_v6/`). No region or campaign ratio is promoted
  to a serving claim without it.
- **Not MI300X.** This node is gfx950/MI350X. `PROVENANCE.md:70` — no
  MI300X/gfx942 or production-vLLM end-to-end claim exists for this port.

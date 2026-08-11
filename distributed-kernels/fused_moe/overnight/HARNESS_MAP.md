# HARNESS_MAP.md — MoK synthetic prefill campaign harness (read-only recon, 2026-08-11)

Citation aliases, all on `gbt350-odcdh2-c05-1`. `HB/` = `~/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/`;
`ab.py` = `~/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py` (7263 lines);
`mori/` = `/opt/venv/lib/python3.12/site-packages/mori/` **inside the campaign image
`rocm/atom-dev:vllm-latest`** — NOT `/usr/local/.../dist-packages/mori`, an older mori with no
`ops/_jit_loader.py`; `subha_k1` is not representative for JIT questions.

## 1. Invocation

Both run from `cd ~/amd-master/auto-gpu-kernel/k0_fused_moe`.
(a) 1-process correctness smoke — the 600-epoch soak still runs (§8), so it is not a 10-second check:

```bash
setsid timeout 2700 env \
  K0_MOK_ARMS=production,mps_mega \
  K0_MOK_WARMUP_ITERS=1 K0_MOK_TIMED_ITERS=1 \
  K0_MOK_OUTPUT_ROOT=$HOME/k0-mok-mps-<tag> \
  K0_MOK_RUN_TIMEOUT=2400 \
  K0_MPS_CFG="C=8,g=2,mode=2,flush_rows=16" \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh <tag> 1
```

(b) 5-process full decision campaign, three arms, rotated:

```bash
setsid timeout 5400 env \
  K0_MOK_ARMS=production,pf6gm_mega,mps_mega \
  K0_MOK_WARMUP_ITERS=500 K0_MOK_TIMED_ITERS=100 \
  K0_MOK_OUTPUT_ROOT=$HOME/k0-mok-mps-<tag> \
  K0_MOK_RUN_TIMEOUT=2400 \
  K0_MPS_CFG="C=32,g=4,mode=2,flush_rows=16" \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh <tag> 5
```

`$1`=tag, `$2`=run_count (rotated processes) (`HB/run_campaign.sh:10-11`); arm order rotates by `(run-1) %
n_arms` (`:89-95`). `K0_MPS_CFG` must be complete whenever `mps_mega` is in the arms (§7);
`K0_MOK_OUTPUT_ROOT` must be fresh, an existing `summary.json`/`runN`/`runN.log` aborts (`:54-57,84-87`); do
not set `K0_PF6GM_G`, the runner forwards 3 (`:136`) which `mps_mega` requires (`ab.py:251-254`).

## 2. Env knob table

Host side, read by `HB/run_campaign.sh`:

| Var | Default | Effect | Cite |
|---|---|---|---|
| `K0_MOK_ARMS` | `production,pf4h` | arm CSV; needs `production` + ≥1 candidate | `:16,36-39` |
| `K0_MOK_OUTPUT_ROOT` | `$HOME/k0-mok-synthetic-results/<tag>` | campaign output root | `:13` |
| `K0_MOK_CACHE_ROOT` | `$HOME/.cache/k0-mok-synthetic-prefill` | `/work` cmake build + `/root/.mori` | `:14,108,144` |
| `K0_MOK_LOCK_DIR` | `/tmp/k0_mok_synthetic_gpu_lock` | mkdir GPU lease; exit 6 if held | `:15,44-47` |
| `K0_MOK_RUN_TIMEOUT` | `2400` | per-run SIGTERM timeout | `:100` |
| `K0PF_IMAGE` | `rocm/atom-dev:vllm-latest` | container image | `:12,40-43` |
| `DHK_ROOT` | `$HOME/Distributed-HipKittens` | DHK tree, bind-mounted read-only | `:8,107` |
| `HIPKITTENS_ROOT` | `<workspace>/HipKittens` | HK tree | `:7,23-26` |
| `K0_MOK_BUILD_JOBS` | `16` | `cmake --build -j` | `:158` |
| `K0_MOK_WARMUP_ITERS` | `500` | untimed iters/arm | `:130`, `ab.py:141` |
| `K0_MOK_TIMED_ITERS` | `100` | timed iters/arm | `:131`, `ab.py:142` |
| `K0_MOK_ABSOLUTE_TOLERANCE` | `0.1` runner / `1.0` ab.py | MoK `abs_error_max` gate | `:132`, `ab.py:144` |
| `K0_MOK_RELATIVE_TOLERANCE` | `0.1` | MoK `relative_error` gate | `:133`, `ab.py:147` |
| `K0_MOK_SEED_BASE` | `1234` | synthetic-input seed | `:129`, `ab.py:1830` |
| `K0_MOK_WEIGHT_CHUNK_ELEMENTS` | `33554432` | weight-gen chunking | `:134`, `ab.py:1832` |
| `K0_PF6GM_G` | `3` runner / `2` ab.py | G-stack width; must be 3 for `mps_mega` | `:136`, `ab.py:248,251-254` |
| `K0_SPIN_LIMIT` | `2000000` | bounded-acquire spin; overflow sets `pperr` | `:142`, `ab.py:1801` |
| `K0_PF6_DEBUG_PHASE` / `K0_PF5_PRIMITIVE_ONLY` | `0` / `0` | pf6 (non-MPS) phase truncation; pf5 primitive-only | `:143,135`, `ab.py:2696` |

MPS-specific:

| Var | Default | Effect | Cite |
|---|---|---|---|
| `K0_MPS_CFG` | `""` | `C,g,mode,flush_rows` required; `pull_fallback,timestamps` optional (0/1); packed by `k0_mps_host_abi.encode_config` into a descriptor word | `:137`, `ab.py:389-427` |
| `K0_MPS_DEBUG_STOP` | `""` | sets `desc[49]=N` (1=M0 … 6=through M7) to truncate the megakernel | `:138`, `ab.py:2761-2772` |
| `K0_MPS_DESC_DUMP` | `""` | one-shot `repr(desc_mps)` → `/out/mps_desc_rank<LOCAL_RANK>.txt` | `:139`, `ab.py:3195-3198` |
| `K0_MPS_TRACE` | `""` | per-arm markers → `/out/progress_rank<RANK>.log` | `:140`, `ab.py:3340-3346,4504-4506` |
| `K0_MPS_SOAK_ITERS` | `600` | soak epochs; any value ≠ 600 **raises** | `:141`, `ab.py:4937-4941` |
| `K0_MPS_SKIP_LAUNCH` | `""` | skips the `mps_mega` launch. **NOT forwarded** by `run_campaign.sh` (absent from `:137-143`); direct torchrun only | `ab.py:3199` |

## 3. Arm → source map and build-cache key

| Arm | Host body / launch | Entry symbol | Device source |
|---|---|---|---|
| `production` | `prod_body` `ab.py:3060-3068,3519` | none | mori `op_prod.dispatch/combine` + aiter `fused_moe`; not JIT'd |
| `pf6gm_mega` | `ab.py:3584`, launch `:3188-3192` | `k0pf6gm_mega` `ab.py:289` | `prefill_opt/kernels/k0pf6gm_mega.hip` (frozen donor) `ab.py:885` |
| `mps_mega` | `ab.py:3339-3358,3596`, launch `:3194-3204` | `k0pf6gm_mps_mega` `ab.py:290` | `$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip` `ab.py:890-892`, plus `moe_hk_adapter.cuh`, `moe_mps_adapter.cuh`, `n2_phase2_gm_mps.cpp`, `k0pf6_mori_heap_snapshot.hip` `ab.py:893-923` |

Flow: rank 0 `shutil.copyfile`s every `_PF6_SOURCE_SPECS` entry (`ab.py:878-923`) into `KERNELS_DIR` =
`<mori>/_jit-sources/src/ops/kernels` (`ab.py:171-174,935-940`), then `ensure_compiled(name)` →
`mori.jit.core.compile_genco` (`ab.py:1193-1194`, `mori/ops/_jit_loader.py:56`). **Exact cache key**
(`mori/jit/core.py:551-561`, `mori/jit/cache.py:42-56,83-90`):
`<MORI_JIT_CACHE_DIR | ~/.mori/jit>/gfx950_mlx5/<hash12>/<kernel>.hsaco`, where `hash12` = `sha256(bytes of
every file under {<mori>/_jit-sources/src/ops, <mori>/include/mori} whose suffix ∈ {.hpp, .h, .cpp, .hip},
path-sorted)[:12]`; existence of that `.hsaco` is the hit. `~/.mori` is bind-mounted from
`$K0_MOK_CACHE_ROOT/mori` (`HB/run_campaign.sh:144`), so the cache survives every container; 250 hash dirs
and 8 distinct `k0pf6gm_mps_mega.hsaco` exist today, so `.hip` invalidation demonstrably works.

**What does NOT invalidate it: `.cuh` is absent from the suffix allow-list.** Editing only
`moe_mps_adapter.cuh` or `moe_hk_adapter.cuh` leaves `hash12` unchanged and the stale
`k0pf6gm_mps_mega.hsaco` is silently reused. A `.cuh`-only change must be paired with a byte change to a
hashed file (e.g. a version-comment bump in `k0pf6gm_device_tile_mps.hip`) or the `<hash12>` dir removed.
Ground truth per run: `R["pf6mps"]["source_sha256"]` / `["hsaco_sha256"]` in every rank JSON
(`ab.py:1222-1240`), surfaced as `summary.json → kernel_source_sha256` / `kernel_hsaco_sha256`
(`HB/summarize.py:91-98`) and enforced as a cross-run invariant (`:185-198`). The host-side
`k0_mps_host_abi` pybind module is a *separate* CMake target keyed on `$DHK` `moe_host_abi.hpp`
(`HB/CMakeLists.txt:19-21,80-88`), cached in `$K0_MOK_CACHE_ROOT/build`.

## 4. Getting a kernel edit picked up

1. Push to `codex/distributed-hipkittens-scaffold`, then on the node `cd ~/Distributed-HipKittens && git pull`
   (tree is clean and on that branch, remote `github.com/subha-v/Distributed-HipKittens`). `DHK_ROOT` defaults
   to exactly that path (`HB/run_campaign.sh:8`), is bind-mounted read-only at
   `/workspace/Distributed-HipKittens` (`:107`), and `K0_DHK_ROOT` points there (`:116`, read at
   `ab.py:304-309`). Editing in place there is equivalent to a `git pull` as far as the harness is concerned.
2. **Nothing else — no container restart, no image rebuild, no manual copy.** Each run spawns a fresh
   `--rm` container and rank 0 re-installs the DHK sources into the container-local `KERNELS_DIR`.
3. Verify pickup: `[mori-jit] Compiling k0pf6gm_mps_mega …` in `runN.log`, and a `pf6mps.hsaco_sha256`
   that differs from the previous campaign. If neither moved, you hit the `.cuh` hole in §3.

## 5. Output layout of a completed run

`$K0_MOK_OUTPUT_ROOT/` → `run1..runN/`, `run1..runN.log`, `summary.json` (`HB/run_campaign.sh:82-83,186-188`);
reference example `~/k0-mok-synthetic-results/mps_ref_g3`. Per-rank JSON `runN/k0pf_mok_synthetic_rank{0..7}.json`
(`ab.py:4889-4891`), exactly 8 required or the campaign aborts, exit 23 (`:167-172`). `runN.log` is the tee'd
all-rank stdout (`:162`). `K0_MPS_DESC_DUMP` / `K0_MPS_TRACE` artifacts land in the same `runN/` (`/out`).

| grep for in `runN.log` | meaning | emitter |
|---|---|---|
| `\[MOK GATE\] <arm> max_abs=… relative=… pass=` | authoritative per-arm correctness | `ab.py:4584-4589` |
| `\[MARK\] control_fails=True` | negative control failed as required | `ab.py:4831` |
| `\[MPS SOAK\] completed=600/600 pperr=0 pass=True` | soak verdict; `pperr != 0` is a terminal bounded-acquire timeout, also on `[MARK] eager` | `ab.py:4511,4521,4962,4977-4982` |
| `\[MOK SYNTHETIC EAGER\] status=valid_diagnostic .* gate_ok=True` | final per-run verdict + ratios | `ab.py:4892-4898` |

`[MARK] eager <arm> … pass=False` is **expected and is not a gate** under `mok_eager`: that line is the strict
k0 gate (`max_abs ≤ 0.02`, `rel_L2 ≤ 0.01`, `ab.py:3477`), which real FP8 output fails at `max_abs ≈
0.031-0.039`. The gate is `[MOK GATE]` → `mok_correctness_all_pass` (`ab.py:4625-4627`). The mandate's
`rel_L1` is `relative_error` = `global sum|diff| / global sum|ref|` (`HB/correctness.py:12-59`); pass =
finite, `nonfinite == 0`, `abs_error_max ≤ 0.1`, `relative_error ≤ 0.1` (`:62-78`).

Summarizer: `python3 HB/summarize.py <run1> … <runN> --output <root>/summary.json`
(`HB/run_campaign.sh:186-187`, `HB/summarize.py:360-371`). The field the mandate calls `arm_p50_us` is
**`summary.json → arm_p50_us[<arm>].median`**: median across runs of each run's `p50_us`, itself the 50th
percentile of the index-aligned `MAX(rank)` sample (`HB/summarize.py:140,271-274`, `ab.py:5033,5050`); ratio
vs production is `primary_mok_results[<arm>].candidate_production_ratio` (`HB/summarize.py:287-302`).

## 6. Timing (measured wall clock, warm caches)

- 5-run × 2-arm at 500/100: `mps_ref_g3` = **3 min 11 s** total (~38 s/run); `k0-mok-exp35-decide5` = **2 min 23 s** (~26 s/run).
- 1-proc smoke, warm hsaco, faulting at the first `mps_mega` touch: **35 s** (`k0-mok-synthetic-mps-smoke`,
  06:31:11→06:31:46). 1-proc VOID (rejected `K0_MPS_CFG`): **13 s** (`k0-mok-mps-mode0c8`, 07:09:27→07:09:40).
- Projected 5-run × 3-arm with `mps_mega` at 500/100: ~38 s/run for the two known arms, plus ~600 iters ×
  ~6.9 ms ≈ 4 s for the third, plus the 600-epoch soak (~4 s of GPU plus 600 sync + `all_reduce` round
  trips) ⇒ **≈60-90 s/run, ≈6-8 min per campaign**.
- Cold-cache `k0pf6gm_mps_mega` genco cost: **UNKNOWN** — no log on the node has a `[mori-jit] Compiling` line
  for it. Paid once per new source hash, run 1 only, inside `K0_MOK_RUN_TIMEOUT` (2400 s); do not reuse
  `CLAUDE.md`'s `timeout 1800 / RUN_TIMEOUT=1500` for a first run on a new hash.

## 7. Gotchas

Already fixed/known (`HB/MPS_OVERNIGHT_HARNESS_NOTE.md:8-20`): the `device=dev` NameError and `.cuda_stream`
int-handle fix in `ab.py`, plus container forwarding of the four MPS debug knobs. Fault status (`:22-29`):
mode-2 first launch faults `address (nil)` on all 8 GPUs, `K0_MPS_DEBUG_STOP=1..6` are CLEAN, descriptor
asserted `numel()==63` (`ab.py:2767-2769`). Resource-parity blocker and prioritized LDS/scratch experiments:
`HB/MPS_KERNEL_FIX_HANDOFF.md:17-92`; validated G=3 reference production 7702.98 µs, `pf6gm_mega` 6902.09 µs,
ratio 0.89603 (`:94-99`).

**VOID vs CLEAN.** A partial/malformed `K0_MPS_CFG` is rejected at *module import*, before any GPU work
(`ab.py:389-427`, called at `:425-427`). Confirmed signature on `~/k0-mok-mps-mode0c8`, which set
`K0_MPS_CFG="C=8,mode=0"`: `runN/` is **empty** (zero rank JSONs), the campaign dies with `run N did not produce eight rank
JSON files`, exit 23 (`HB/run_campaign.sh:169-172`), `runN.log` holds 8 × `ValueError: K0_MPS_CFG requires
C,g,mode,flush_rows and optionally pull_fallback,timestamps; got [...]` then `ChildFailedError`, and there is
**no `[MARK]`, no `[MOK GATE]`, no `[MPS SOAK]`, no `Memory access fault`** anywhere. CLEAN (the real fault)
reaches `[MARK] eager production rel_L2=0.00648 pass=False pperr=0` and then 8 × `Memory access fault by GPU
node-N … on address (nil)` with SIGABRT (exit −6). **Discriminator: does `runN.log` contain ≥1 `[MARK]` line
and does `runN/` hold 8 rank JSONs?** If not it is VOID, nothing was measured. Second VOID path, same quiet
look: `K0_MPS_SKIP_LAUNCH` on a *direct* torchrun (`ab.py:3199`) elides the launch so the arm reads stale
buffers and can "pass"; it is not forwarded, so it cannot silently affect a campaign.

Other traps: the runner refuses to start unless all 8 GPUs read 0% use and no `torchrun`/`mpirun`/`orterun` is
alive (`HB/run_campaign.sh:59-78`, exits 20/21), holds a mkdir lease (exit 6), and never overwrites an existing
`summary.json`/`runN` (exits 7/22). Outside the campaign the absolute tolerance defaults to a *looser* 1.0
(`ab.py:144` vs `HB/run_campaign.sh:132`) and `K0_PF6GM_G` defaults to 2, making `mps_mega` raise
(`ab.py:248-254`) — always go through `run_campaign.sh`. A `pf6gm_mega/production` p50 ratio near 0.896
confirms the G=3 reference; ~0.941 means G=2 was built.

## 8. Negative control and soak — both already exist

- **Negative control: present and mandatory.** A deliberately broken combine kernel,
  `k0_region_combine_push_store` (`fn_combine_bug`, `ab.py:553`), runs over an otherwise correct pipeline
  (`ab.py:4818-4823`); `R["mok_control_fails"]` must be `True` (`:4826-4827`) and is a hard conjunct of both
  the pre-timing gate (`:4906-4911`) and the final timing gate (`:5176-5184`). For `mps_mega` it is also
  recorded as `R["mps_negative_control"] = {kind: "frozen combine store-over-peer …", failed_as_required: …}`
  (`:4932-4935`).
- **600-epoch soak: present, automatic, not configurable.** Whenever `mps_mega` is in the arms,
  `ab.py:4937-4999` runs `K0_MPS_SOAK_ITERS` epochs of the `mps_mega` body, MAX-all_reduces `pperr` after every
  epoch and breaks on the first nonzero, then re-gates the output with `mok_gate`. `K0_MPS_SOAK_ITERS != 600`
  **raises** (`:4938-4941`), so it cannot be shortened. Failure writes status `blocked_pre_timing_mps_soak` and
  exits 3 (`:4986-4999`); success lands in `R["mps_soak"]`.
- **Gap:** `blocked_pre_timing_mps_soak` is **not** in `HB/summarize.py`'s `BLOCKED_STATUSES` (`:28-31`), so a
  soak failure makes `summarize.py` raise `ValueError: … has unsupported status 'blocked_pre_timing_mps_soak'`
  (`:133-134`) instead of emitting a blocked summary; the campaign still exits nonzero
  (`HB/run_campaign.sh:173-178`) but no `summary.json` is written. Smallest fix: add that string to
  `BLOCKED_STATUSES` — one line, no change to accept/reject semantics.

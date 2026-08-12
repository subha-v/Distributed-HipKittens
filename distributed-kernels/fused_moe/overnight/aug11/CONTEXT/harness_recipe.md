# MoK synthetic-prefill harness recipe (aug11 night, GPU-run agent)

Written 2026-08-12 ~03:35Z on `gbt350-odcdh2-c05-1` (8x MI350X, gfx950).
Node checkout verified at **HEAD `0b82cd19`**, branch
`codex/distributed-hipkittens-scaffold`, `K0P6_MPS_SRC_REV 23`.

Authorities read for this document: `benchmarks/mok_synthetic_prefill/run_campaign.sh`,
`benchmarks/mok_synthetic_prefill/summarize.py`,
`prefill_opt/host/e004pf_k0pf_ab.py`, `MPS_OVERNIGHT_HARNESS_NOTE.md`, and the
surviving prior-art drivers in `~/.overnight-scripts/screen_*.sh`.

---

## 1. Exact command lines

Everything runs from `~/amd-master/auto-gpu-kernel/k0_fused_moe`.

### Screen (1 process, 1 warmup, 1 timed iter; ~24 s per point once cached)

```bash
setsid timeout 1800 env \
  K0_MOK_ARMS=production,pf6gm_mega,mps_mega \
  K0_MOK_WARMUP_ITERS=1 K0_MOK_TIMED_ITERS=1 \
  K0_MOK_OUTPUT_ROOT=$HOME/k0-mok-<tag> \
  K0_MOK_RUN_TIMEOUT=1500 \
  K0_MPS_CFG="C=16,g=33,mode=12,flush_rows=16" \
  K0_MPS_TRACE=1 \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh <tag> 1
```

### Campaign (5 rotated processes, 500 warmup, 100 timed; ~18 min)

```bash
setsid timeout 5400 env \
  K0_MOK_ARMS=production,pf6gm_mega,mps_mega \
  K0_MOK_WARMUP_ITERS=500 K0_MOK_TIMED_ITERS=100 \
  K0_MOK_OUTPUT_ROOT=$HOME/k0-mok-<tag> \
  K0_MOK_RUN_TIMEOUT=2400 \
  K0_MPS_CFG="C=16,g=33,mode=12,flush_rows=16" \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh <tag> 5
```

`run_campaign.sh` takes exactly two positional arguments: `<tag>` (used only for
the container name and, if `K0_MOK_OUTPUT_ROOT` is unset, the default output
directory) and `<run_count>` (the number of processes; arms rotate by one
position per process, `rotation = (run-1) % n_arms`).

### The driver that wraps both

```bash
bash ~/.overnight-scripts/screen.sh <TAG> ~/overnight-scratch/<TAG>.cfgs
# campaign mode:
SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5 SCREEN_JOB_TIMEOUT=5400 \
  bash ~/.overnight-scripts/screen.sh <TAG> <cfgfile>
```

Source of truth is `overnight/aug11/tools/screen.sh` in this repo; push it with
`overnight/aug11/tools/push.ps1` (strips CRLF on the way, which plain `scp`
does not).

---

## 2. Environment knobs the harness understands

### 2.1 Read by `run_campaign.sh` on the HOST (never reach the container)

| knob | default | effect |
|---|---|---|
| `K0PF_IMAGE` | `rocm/atom-dev:vllm-latest` | container image. Must already be present locally or the script exits 5. |
| `K0_MOK_OUTPUT_ROOT` | `~/k0-mok-synthetic-results/<tag>` | campaign root. **When you set it, `<tag>` is NOT appended** — the root is used verbatim, so two points sharing it collide. |
| `K0_MOK_CACHE_ROOT` | `~/.cache/k0-mok-synthetic-prefill` | mounted at `/work`; holds the cmake build tree and (via `${cache_root}/mori:/root/.mori`) the mori JIT cache. |
| `K0_MOK_LOCK_DIR` | `/tmp/k0_mok_synthetic_gpu_lock` | `mkdir`-based GPU lease. Exit 6 if it already exists. A crashed run leaves it behind. |
| `K0_MOK_ARMS` | `production,pf4h` | comma list. Must contain `production` and at least one candidate, else exit 4. Order here is the base order; processes rotate it. |
| `K0_MOK_RUN_TIMEOUT` | `2400` | `timeout` on each `docker run`. |
| `K0_MOK_BUILD_JOBS` | `16` | `cmake --build -j`. |
| `HIPKITTENS_ROOT` | `<workspace>/HipKittens` | must contain `include/kittens.cuh`, else exit 3. |
| `DHK_ROOT` | `~/Distributed-HipKittens` | bind-mounted **read-only** at `/workspace/Distributed-HipKittens`. With `mps_mega` in the arms it must contain `distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip`, else exit 3. |

### 2.2 Forwarded into the container (overridable)

| knob | default | effect |
|---|---|---|
| `K0_MOK_SEED_BASE` | `1234` | synthetic-input seed base. |
| `K0_MOK_WARMUP_ITERS` | `500` | warmup iterations per arm. |
| `K0_MOK_TIMED_ITERS` | `100` | timed iterations per arm. |
| `K0_MOK_ABSOLUTE_TOLERANCE` | `0.1` | `max_abs` gate. **A gate, not a dial — never widen.** |
| `K0_MOK_RELATIVE_TOLERANCE` | `0.1` | `relative` gate. Same. |
| `K0_MOK_WEIGHT_CHUNK_ELEMENTS` | `33554432` | host-side weight staging chunk. |
| `K0_PF5_PRIMITIVE_ONLY` | `0` | pf5 primitive-only path. |
| `K0_PF6GM_G` | `3` | the G the megakernel arms are built with. **This is the number that decides whether pf6gm/production reads ~0.896 (G=3) or ~0.941.** |
| `K0_MPS_CFG` | `''` | the MPS config string, see 2.4. |
| `K0_SYNTH_ROUTE` | `''` | synthetic route family (skew). Must be one of `SYNTH_ROUTE_FAMILIES` or ab.py raises. |
| `K0_SYNTH_SEED` | `0` | route seed. |
| `K0_MPS_DEBUG_STOP` | `''` | bisect the megakernel, 1=M0 … 6=through M7. **Aborts the ranks (exit 2) and yields no timings** on this build. |
| `K0_MPS_DESC_DUMP` | `''` | writes `/out/mps_desc_rank*.txt`. |
| `K0_MPS_TRACE` | `''` | writes `/out/progress_rank<N>.log` with `arm <name>`, `pf6mps_mega_body enter`, `pf6mps_mega_body launch-returned`. Three host-side file appends per arm per launch; no device cost. |
| `K0_MPS_SOAK_ITERS` | `600` | soak epochs. **The harness raises unless it is exactly 600. Do not shorten.** |
| `K0_SPIN_LIMIT` | `2000000` | M2 chunk-poll spin limit, reported by `[MPS SPIN]`. |
| `K0_PF6_DEBUG_PHASE` | `0` | pf6 phase debug. |

### 2.3 Hard-wired by `run_campaign.sh` (NOT overridable — the fixed baseline)

`HSA_XNACK=1`, `MORI_GPU_ARCHS=gfx950`, `MORI_SHMEM_HEAP_SIZE=34359738368`,
`K0_INPUT_MODE=mok_synthetic`, `K0_BENCHMARK_PROTOCOL=mok_eager`,
`K0_T=4096`, `K0_T_LOC_MAX=40960`, `K0_PADMAX=263136`, `K0_MAXTOK=4096`,
`K0_MAXTOK_PROD=4096`, `K0_BLK=128`, `K0_WARP=16`.

**`K0_MPS_SKIP_LAUNCH` is read by `e004pf_k0pf_ab.py` but is absent from the
`-e` list**, so it has no effect through the campaign. It only works on a direct
torchrun — which you must not do, because a direct torchrun defaults the
absolute tolerance to 1.0 and `K0_PF6GM_G` to 2.

`summarize.py` reads **no** environment variables at all.

### 2.4 `K0_MPS_CFG` grammar (`_parse_mps_config`, ab.py:389)

- Required keys: `C`, `g`, `mode`, `flush_rows`.
- Optional keys: `pull_fallback`, `timestamps` — each must be `0` or `1`.
- Any other key, any duplicate key, or any missing required key raises
  `ValueError` **at module import on all eight ranks**.
- Values go through `int(value, 0)`, so `0x20` is legal.
- The six fields are packed by `k0_mps_host_abi.encode_config` into one word.

**A partial cfg looks like a pass**: eight `ValueError`s, an empty run dir, no
`[MARK]` line, no `[MOK GATE]` line. Always pass all four required keys.

---

## 3. Log-line formats, verbatim

Quoted from real logs on the node, one example line each.

### Gates that matter

```
[MOK GATE] production max_abs=0.027344 relative=0.005724 pass=True
[MOK GATE] pf6gm_mega max_abs=0.035156 relative=0.008293 pass=True
[MOK GATE] mps_mega max_abs=0.035156 relative=0.008293 pass=True
[MARK] control_fails=True
[MPS SOAK] completed=600/600 pperr=0 pass=True
[MOK SYNTHETIC EAGER] status=valid_diagnostic ratios={"pf6gm_mega/production_p50": 0.8912944056437666, "mps_mega/production_p50": 0.8958896839106377, "pf6gm_mega/production_p95": 0.8912944056437666, "mps_mega/production_p95": 0.8958896839106377} gate_ok=True
```

### The trap line — NOT a gate

```
[MARK] eager production rel_L2=0.00644 pass=False pperr=0
[MARK] eager pf6gm_mega rel_L2=0.00976 pass=False pperr=0
[MARK] eager mps_mega rel_L2=0.00976 pass=False pperr=0
```

`pass=False` here is normal and reads `False` even for `production`. The real
correctness gates are `[MOK GATE]`, `pperr`, `[MARK] control_fails=True`, and
`[MPS SOAK]`. `pperr` appears on both the `[MARK] eager` lines and the
`[MPS SOAK]` line; treat any nonzero as terminal.

### Device stamps (only with `timestamps=1` in `K0_MPS_CFG`)

```
[MPS TS] FIRST_READY_inv=18446634936935660749 LAST_READY=109136774499401 DRAIN=109136774505277 M7_DONE=109136774227934 REDUCE_DONE=109136774515422 M5_DONE=109136773605210 M2_DONE=109136773605210 M6_DONE=109136773910034
[MPS TS SPLIT] plan_M3toM5=<ticks> M6=<ticks> (ticks, 1 tick = 0.01 us)
[MPS TS DELTA] planM6=304824 M7=317900 combine=287488 servicedrain=595243 m2_to_end=910212
[MPS SPIN] chunk_poll success_max=0 fail_max=0 limit=2000000
```

- Field order for `[MPS TS]` comes from ab.py:5000 —
  `FIRST_READY_inv, LAST_READY, DRAIN, M7_DONE, REDUCE_DONE, M5_DONE, M2_DONE, M6_DONE`.
  Logs from earlier on 2026-08-11 print `KSTART_inv` in the `M5_DONE` slot; the
  harness was edited mid-night. Parse by name, never by position.
- **1 tick = 0.01 us** (wall clock is 100 MHz; the 2.2 GHz shader clock is the
  wrong divisor).
- `[MPS TS SPLIT]` prints only when `M2_DONE`, `M5_DONE`, `M6_DONE` are all
  nonzero; `[MPS TS DELTA]` only when `M2_DONE`, `M6_DONE`, `M7_DONE`,
  `REDUCE_DONE` are all nonzero. With `timestamps=0` every cell reads 0 and
  neither line appears — `[MPS TS]` still prints, all zeros.
- These are **max** stamps, so after the soak they describe the FINAL SOAK
  EPOCH, not a timed iteration. Do not sum them and compare to `arm_p50_us`.
- `[MPS SPIN]` counters are running maxima that are never reset, so they cover
  warmup + timed + all 600 soak epochs.

### Stage profiles and structural gates

```
[K0PF GATE] plan_equivalence pass=True diffs={'own': 0, 'tof': 0, ...} perr=0/0 padded=33440 T_loc=21816
[K0PF GATE] quant_exact pass=True byte_diffs=0 scale_diffs=0
[K0PF GATE] gather_exact pass=True byte_diffs=0 scale_diffs=0
[K0PF GATE] combine_bit_exact pass=True bf16_bit_diffs=0
[K0PF PROFILE] production {'dispatch': 925.8490204811096, 'gemm': 5942.897796630859, 'combine': 1251.093029975891}
[K0PF PROFILE] pf_full {'plan': 505.3, 'quant': 32.5, 'b1': 47.9, 'gather': 2054.6, 'zero': 147.6, 'n2_p1': 3315.2, 'n2_p2': 1885.1, 'b2': 866.5, 'combine': 1283.3}
```

### Campaign progress and the JIT

```
starting run=1/1 arms=production,pf6gm_mega,mps_mega at 2026-08-12T03:28:37Z
completed run=1/1 at 2026-08-11T19:10:58Z
campaign complete: /home/subvadla/k0-mok-mps/verify_C64g1mode2flush_rows16/summary.json
[mori-jit]   Cached: /root/.mori/jit/gfx950_mlx5/0e22fda28c09/k0pf6gm_mps_mega.hsaco
```

**The `[mori-jit]` lines are not always emitted.** All six screens run tonight
produced zero `[mori-jit]` lines, so a log-derived JIT hash is empty far more
often than not. See gotcha 6.

### Harness refusals (each is a distinct exit code)

```
refusing to overwrite completed campaign <output_root>      # exit 7
GPU lease is already held: /tmp/k0_mok_synthetic_gpu_lock    # exit 6
refusing to run: torchrun is active                          # exit 20
refusing to run: only N/8 GPUs report 0% use                 # exit 21
refusing to overwrite <run_dir> or <log>                     # exit 22
run N did not produce eight rank JSON files                  # exit 23
campaign blocked before timing: <summary.json>               # propagates run rc
```

Other exits: 2 = k0 root missing, 3 = HipKittens/DHK root missing,
4 = bad `run_count` or bad `K0_MOK_ARMS`, 5 = image missing.

### Where the microseconds actually live

Not in the log. `run_campaign.sh` writes
`<output_root>/summary.json`, and the number to quote is

```
summary["arm_p50_us"][<arm>]["median"]      # median over processes of the rank-max p50
```

with `["values"]` holding the per-process samples and `arm_p95_us` present but
flagged `p95_note: supplementary only`. `summary["primary_mok_results"][<arm>]`
carries `candidate_production_ratio`, `latency_reduction_percent`, and
`median_delta_us`. Per-run rank JSONs are
`<output_root>/run<N>/k0pf_mok_synthetic_rank<0-7>.json`.

`summarize.py` has `BLOCKED_STATUSES = {blocked_pre_timing_correctness,
blocked_pre_timing_mok_correctness}`. `blocked_pre_timing_mps_soak` is **not**
in that set, so a soak failure makes `summarize.py` raise instead of writing a
blocked summary — the campaign then has no `summary.json` at all.

---

## 4. CSV schema written by `screen.sh`

One row per config, appended to `~/overnight-scratch/screen_<TAG>.csv`.
38 columns; `cfg`, `note`, and both `hsaco_*` fields are double-quoted, so the
file is `csv.reader`-parseable (the prior-art CSVs were not — an unquoted `cfg`
containing commas made every row ragged).

| # | column | meaning |
|---:|---|---|
| 1 | `idx` | 1-based position in the batch |
| 2 | `utc` | when the row was written |
| 3 | `cfg` | the `K0_MPS_CFG` string, quoted |
| 4 | `status` | `OK`, `FAIL:rc<N>`, `FAIL:nosummary`, `FAIL:gate=<v>`, `FAIL:soak=<v>`, `FAIL:pperr=<v>`, `FAIL:control=<v>`, or `BUSY` |
| 5 | `iters` | `w<warmup>t<timed>p<procs>` — screens and campaigns are distinguishable |
| 6 | `head` | node checkout short SHA |
| 7 | `src_rev` | `K0P6_MPS_SRC_REV` from the MPS kernel source |
| 8-10 | `prod_us`, `pf6gm_us`, `mps_us` | `arm_p50_us[arm]["median"]` |
| 11 | `ratio_vs_prod` | `mps_us / prod_us` |
| 12 | `ratio_vs_pf6gm` | `mps_us / pf6gm_us` |
| 13 | `pf6gm_vs_prod` | reference check, should sit near 0.896 for G=3 |
| 14-16 | `gate_pass`, `gate_max_abs`, `gate_rel` | from `[MOK GATE] mps_mega` |
| 17 | `control_fails` | from `[MARK] control_fails=` — must be `True` |
| 18 | `pperr` | max over every `pperr=` in the log |
| 19-20 | `soak`, `soak_epochs` | from `[MPS SOAK]`, e.g. `True`, `600/600` |
| 21-25 | `ts_planM6_us`, `ts_M7_us`, `ts_combine_us`, `ts_servicedrain_us`, `ts_m2_to_end_us` | `[MPS TS DELTA]`, converted ticks -> us |
| 26-27 | `ts_planM3toM5_us`, `ts_M6_us` | `[MPS TS SPLIT]`, converted |
| 28-29 | `spin_ok_max`, `spin_fail_max` | `[MPS SPIN]` |
| 30-31 | `eager_status`, `eager_gate_ok` | `[MOK SYNTHETIC EAGER]` |
| 32 | `mps_hsaco` | content hash of the mps_mega build resolved through the `latest` symlink AFTER the run — authoritative |
| 33 | `jit_log` | hash scraped from a `[mori-jit]` line, usually empty |
| 34-35 | `hsaco_before`, `hsaco_after` | `<true mtime>|<hash>` either side of the run. Equal = cache hit |
| 36 | `rc` | exit code of the campaign |
| 37 | `outdir` | basename of the unique per-run output root |
| 38 | `note` | refusal tags (`campaign-dir-collision`, `gpu-lease-held`, `foreign-job`, `gpus-not-idle`, `missing-rank-json`, `bad-mps-cfg`, `blocked-pre-timing`) or the last log line |

Missing values are empty, never a stale carry-over: the parser is a fresh
process per point and re-derives every field from that point's own summary and
log.

---

## 5. Deliverable 2 — the confirmed baseline

`TAG=a11base`, HEAD `0b82cd19`, `SRC_REV 23`, arms `production,pf6gm_mega,mps_mega`,
screen resolution `w1t1p1`, `K0_MPS_TRACE=1`, 2026-08-12 03:28-03:30Z.
Raw: `~/overnight-scratch/screen_a11base.csv`.

| # | cfg | status | prod_us | pf6gm_us | mps_us | mps/prod | mps/pf6gm | pf6gm/prod | gate | max_abs | rel | control_fails | pperr | soak | hsaco |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `C=16,g=33,mode=12,flush_rows=16` | OK | 7870.8 | 6864.3 | **6727.5** | **0.8547** | 0.9801 | 0.8721 | True | 0.035156 | 0.008294 | True | 0 | True 600/600 | `19c7f3873e77` |
| 2 | `C=16,g=33,mode=12,flush_rows=16` | OK | 7802.2 | 7074.3 | **6703.6** | **0.8592** | 0.9476 | 0.9067 | True | 0.035156 | 0.008293 | True | 0 | True 600/600 | `e92e2fb2cce5` |
| 3 | `C=64,g=1,mode=2,flush_rows=16` | OK | 7885.2 | 6950.4 | **6945.7** | **0.8809** | 0.9993 | 0.8814 | True | 0.035156 | 0.008294 | True | 0 | True 600/600 | `e92e2fb2cce5` |

All three: `rc=0`, `eager_status=valid_diagnostic`, `eager_gate_ok=True`,
`spin_ok_max=0`, `spin_fail_max=0`. All `ts_*` columns are empty because none of
these configs carries `timestamps=1`; add it to populate them.

**Cross-check (row 3).** The previous ratchet `C=64,g=1,mode=2,flush_rows=16`
reads `mps/prod = 0.8809` here and `0.8961` on a pilot batch seven minutes
earlier, against an archived `0.896` in `~/overnight-scratch/screen_verify.csv`
from 19:10 the previous evening. The tree is the one we think it is.

### Run-to-run noise band (rows 1 vs 2, the same config twice)

| quantity | run 1 | run 2 | spread |
|---|---|---|---|
| `mps_us` | 6727.5 | 6703.6 | **0.36 %** |
| `prod_us` | 7870.8 | 7802.2 | **0.88 %** |
| `pf6gm_us` | 6864.3 | 7074.3 | **3.01 %** |
| `ratio_vs_prod` | 0.8547 | 0.8592 | **0.53 %** |
| `ratio_vs_pf6gm` | 0.9801 | 0.9476 | **3.37 %** |

n=2 is thin for a noise band, so I ran four more repeats of the identical point
(`TAG=a11noise`, same HEAD, 03:32-03:34Z). Across all **six** screens of
`C=16,g=33,mode=12,flush_rows=16` tonight:

| quantity | mean | range | 1 sigma |
|---|---|---|---|
| `mps_us` | 6740.8 | 6703.6 - 6803.0 (1.47 %) | 39.1 us (**0.58 %**) |
| `prod_us` | 7891.2 | 7802.2 - 7931.8 (1.64 %) | 50.0 us (**0.63 %**) |
| `pf6gm_us` | 6934.6 | 6864.3 - 7074.3 (3.03 %) | 78.6 us (**1.13 %**) |
| `ratio_vs_prod` | 0.8542 | 0.8465 - 0.8592 (1.49 %) | **0.52 %** |
| `ratio_vs_pf6gm` | 0.9722 | 0.9476 - 0.9846 (3.81 %) | **1.34 %** |

**How to use this.** At screen resolution `ratio_vs_prod` is good to about
+/- 0.5 % (1 sigma) with a worst-observed range of 1.5 %; the mode-2 point moved
1.7 % between two screens seven minutes apart, so treat **2 %** as the smallest
ratio-vs-production delta a single screen can call. `ratio_vs_pf6gm` is **not
usable at screen resolution** — `pf6gm_mega` is by far the noisiest arm at one
warmup and one timed iteration (3.0 % range on its own microseconds), so a
screen can move `mps/pf6gm` by 3.8 % with nothing changed. Rank candidates
against `production` on screens; only a 5-process campaign can rank against
`pf6gm_mega`.

Screens also run optimistic in absolute terms: the shipped exp_21 campaign
number for this config is `0.866x` production, while screens of the same commit
cluster at `0.854x`. Screens are for ordering points, not for the ratchet.

---

## 6. Harness gotchas hit tonight

1. **`run_campaign.sh` exits 7 on a reused output root, and the old driver
   silently laundered the stale result.** `if [[ -e "${output_root}/summary.json" ]]` ->
   `refusing to overwrite completed campaign`. The prior driver derived its
   output dir from the config string alone, so re-running one config wrote a row
   with the FIRST run's microseconds next to `VOID,VOID,VOID,NOJIT` gates —
   see the three identical rows in `~/overnight-scratch/screen_E20confirm2.csv`,
   which are one measurement printed three times. `screen.sh` gives every run a
   unique `<idx>_<slug>_<utc>` root, which is what makes "run the same config
   twice" a legitimate operation.
2. **`K0_MOK_OUTPUT_ROOT` swallows the tag.** Set it and the `<tag>` positional
   is used only for the container name. Two points that share the root collide
   per gotcha 1.
3. **The `latest/` JIT directory is a trap inside the documented trap.**
   `~/.cache/.../mori/jit/gfx950_mlx5/latest/` holds **symlinks** into the
   content-hashed build dirs, and every run re-creates them. `stat` without
   `-L` reports the *symlink's* mtime, so it advances on every run whether or
   not anything compiled, and `ls -t */k0pf6gm_mps_mega.hsaco | head -1` always
   returns `latest/`. The pilot batch reported three "rebuilds" that were all
   cache hits. Use `readlink -f` plus `stat -L`; `screen.sh` now does.
4. **Two byte-different builds of `k0pf6gm_mps_mega` are live in the cache and
   the symlink flips between them run to run.** `19c7f3873e77` (02:25:02) and
   `e92e2fb2cce5` (02:28:08), both 175,496 bytes, different md5, different
   accompanying kernel sets (the first ships the pf6gm family, the second ships
   `k0pf_combine/gather/quant`) — i.e. two compile sessions of the same source
   with different embedded build IDs. Across six screens the target alternated
   `19c7 / e92e / e92e / 19c7 / 19c7 / e92e` with `mps_us`
   6727.5 / 6703.6 / 6803.0 / 6711.5 / 6725.1 / 6774.3. Grouped by hash the
   means are 6721.4 us (`19c7`, n=3) and 6760.3 us (`e92e`, n=3), 39 us apart
   against a 39.1 us 1-sigma — **no resolvable correlation**, so the flip is benign here. It is still logged per run in
   `mps_hsaco`, because the day it does correlate is the day someone measures
   the wrong kernel.
5. **`.cuh`-only edits do not invalidate the JIT cache** (the documented trap;
   not re-triggered tonight because HEAD did not move). Bump
   `K0P6_MPS_SRC_REV` in the `.hip` and confirm `hsaco_before != hsaco_after`.
6. **`[mori-jit]` lines are not emitted on a full cache hit.** All six screens
   tonight produced none, so the log-scraped hash is empty and `NOJIT` in the
   old CSVs meant "cache hit", not "no kernel". The column name invited exactly
   the wrong reading; `screen.sh` splits it into `mps_hsaco` (authoritative,
   from the resolved symlink) and `jit_log` (best effort).
7. **`pgrep -af 'torchrun|mpirun'` does not see our own job.** The container
   runs `python -m torch.distributed.run`, whose command line contains no
   literal `torchrun`; `run_campaign.sh`'s own `assert_idle` uses `pgrep -x
   torchrun` and misses it too. The reliable idle check is
   `rocm-smi --showpids` showing nothing but `gpuagent`. `screen.sh` checks
   both.
8. **A crashed run leaves `/tmp/k0_mok_synthetic_gpu_lock` behind** and every
   later campaign exits 6. `screen.sh` removes it only when the node is
   independently proven idle, and says so loudly.
9. **`[MARK] eager ... pass=False` is not a failure** and `[MPS TS]` prints all
   zeros without `timestamps=1`. Neither is a defect.
10. **PowerShell-side:** piping a string into a native command appends a
    platform newline, so a remote bash script gets a trailing `\r` line and
    exits 127 with `$'\r': command not found` even when it succeeded. Either end
    every piped script with `exit 0` or filter through `tr -d '\r'` on the far
    side; `push.ps1` does the latter. Also, a `~` inside a single-quoted remote
    path is not expanded — it creates a literal `~` directory. Use absolute
    paths.

---

## 7. Node state during this session

- Exclusive lease held throughout. `pgrep -af 'torchrun|mpirun'` empty and
  `rocm-smi --showpids` showing only `gpuagent` (pid 44579) before every launch.
- **No foreign GPU process appeared and nothing was preempted or signalled.**
- No stale `/tmp/k0_mok_synthetic_gpu_lock` at any point.
- Disk: 928 GB free on `/`.
- Every run under `setsid -w` + `timeout`; batches launched detached under
  `setsid nohup timeout 5400`.
- Nothing outside `overnight/aug11/` (local) and `~/overnight-scratch/`,
  `~/.overnight-scripts/`, `~/k0-mok-a11base*`, `~/k0-mok-a11noise` (node) was
  written. No kernel source and no harness file was edited.

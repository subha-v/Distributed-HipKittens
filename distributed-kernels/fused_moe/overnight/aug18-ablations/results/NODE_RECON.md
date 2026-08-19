# NODE_RECON.md — node state and exact build/run paths for the anatomy campaign

**Node:** `gbt350-odcdh2-c05-1` (`ssh -i ~/.ssh/muhammad-gpu subvadla@10.145.64.67`), 8× MI350X
(`MARKET_NAME: AMD Instinct MI350X`, `TARGET_GRAPHICS_VERSION: gfx950`, 256 CU/GPU).
Recon session: 2026-08-18 ~22:58–23:10 UTC. All 8 GPUs `GFX_ACTIVITY: 0 %` at session start.

## 0. Checkout state (JOB 0)

| item | before | after |
|---|---|---|
| `~/Distributed-HipKittens` branch | `ablations` | `ablations` |
| HEAD | `604a9763` (wgrad v2.2, 2026-08-16-era training commit) | **`2578a72a`** (UNDERSTANDING_PLAN r2) |
| divergence vs `origin/ablations` | **0 ahead / 57 behind** | 0 / 0 |

The stale checkout was a pure fast-forward — `git branch -r --contains 604a9763` returns
`origin/ablations`, so **no node-local work existed and nothing was discarded**. Update was
`git fetch origin '+refs/heads/*:refs/remotes/origin/*' && git pull --ff-only origin ablations`.

> Trap recorded: the node's git does **not** advance `refs/remotes/origin/*` under
> `git fetch origin <branch>` (it only writes `FETCH_HEAD`), so `origin/ablations` reads stale —
> `git rev-list --left-right --count HEAD...origin/ablations` said "36 ahead / 0 behind" until the
> explicit refspec was used. Always fetch with the explicit refspec on this node before judging
> divergence.

The node's `origin` is the **https** remote (`https://github.com/subha-v/Distributed-HipKittens.git`),
not the ssh remote the laptop uses — fetch works, push would need credentials. Results are
therefore transferred laptop-side and committed from the laptop.

## 1. Rig (a) — TP8 boundary bench, `m25_boundary_bench.hip` (G25-0b / G25-1)

**Working copy is OUTSIDE the repo:** `~/m25_prim_check/distributed-kernels/tp8_mega/`, with the
HipKittens headers at `~/m25_prim_check/include/`. The three sources there are **byte-identical**
to `~/Distributed-HipKittens/distributed-kernels/tp8_mega/` at `2578a72a`
(`m25_boundary_bench.hip`, `m25_cdar_bench.hip`, `m25_cdar.cuh` — all `diff -q` SAME), so the
prebuilt binaries are pinned to the repo source.

Prebuilt binaries present (do not rebuild unless the source moves):

| binary | built | source |
|---|---|---|
| `m25_boundary_bench` | 2026-08-18 20:13 | `m25_boundary_bench.hip` (G25-1 rig, 4 arms) |
| `m25_cdar_bench` | 2026-08-18 18:39 | `m25_cdar_bench.hip` (G25-0b v0) |
| `m25_cdar_bench_v1` | 2026-08-18 19:00 | v1: depth × bps swept |

**Build (from the file header, `m25_boundary_bench.hip:35-37`):**

```bash
cd ~/m25_prim_check/distributed-kernels/tp8_mega
hipcc --offload-arch=gfx950 -std=c++20 -O3 -I ~/m25_prim_check \
  m25_boundary_bench.hip -o m25_boundary_bench -lrccl
```

**Run:**

```bash
./m25_boundary_bench <mode> [tokens=16384] [slab_rows=256] [iters=7] [depth=4] \
                     [k_inner=64] [bps=2]
# mode ∈ {compute, phased, fused, rccl}
```

Single process, opens all 8 GPUs itself; no container, no torchrun, no GPU lease. One invocation
at the banked shape costs **0.7–1.0 s wall** (`compute`/`phased`/`fused`); the `rccl` arm is
slower (RCCL comm init dominates, ~10–30 s). Output is one `BOUNDARY_BENCH …` block with
`wall_us median/min/max`, `gather_timeouts`, `mag_timeouts`, `verify_errors` and a terminal
`BOUNDARY_BENCH_PASS`/`_FAIL` line; exit code 0 iff PASS.

**`k_inner` (the ρ knob) already exists as argv 6** — `m25_boundary_bench.hip:447`, consumed by
`mfma_burst()` (`:113-131`) as the MFMA K-loop trip count. **No build item was needed for
G-L0a's ρ ladder** (the "one argv" of W_VECTOR:339 / EXPERIMENT_LADDER:482 is already shipped).
`bps` (fragments per slab) is argv 7; `depth` is argv 5.

Banked G25-1 numbers were taken at `k_inner=2048, bps=2, slab_rows=256, tokens=16384, 256 blocks`
(`results/G25_1_STATUS.md`) — that configuration, **not** the `k_inner=64` default, is the ρ=0.65
point.

Neighbouring runner scripts (queue-behind-campaign pattern, useful as templates):
`~/m25_prim_check/run_g25_0b_after_b3.sh`, `run_g25_0b_v1_after_b3rev.sh`, `run_slab_sweep.sh`.

## 2. Rig (b) — anchor-A M15 boundary harness (the "MoK" rig, T=4,096/rank)

**Harness:** `~/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/run_campaign.sh`
(documented in `overnight/aug10/HARNESS_MAP.md`).
**M15 kernel tree:** `~/DHK-m15` — separate worktree, branch tip `5753970b`
*"RUN PIN (ablations-m15): compile the M15 slab kernel as k0pf6gm_mps_mega"* on top of `eda897de`.
Clean. This is the **shipping M15 binary** for R7: the kernel is mori-JIT'd from
`$DHK_ROOT/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip` and cached by source hash in
`~/.cache/k0-mok-synthetic-prefill/mori/gfx950_mlx5/<hash12>/`, so **leaving `~/DHK-m15` untouched
is exactly "no rebuilds"** — the cached `.hsaco` is reused and its sha is echoed per run as
`summary.json → kernel_hsaco_sha256`.

**Exact C-ladder invocation** (recovered verbatim from `~/e03_sweep_queue.sh`, the script that
produced the banked C=24/C=32/flush_rows=0 points):

```bash
cd ~/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill
env DHK_ROOT=/home/subvadla/DHK-m15 \
    K0_MOK_ARMS=production,mps_mega \
    K0_MPS_CFG="C=32,g=353,mode=12,flush_rows=16" \
    timeout 3600 bash run_campaign.sh <tag> 5
```

`g=353` is the depth-4 injection bound; `mode=12` is the producer-carried + injection-bound
schedule; `flush_rows` is the pool front-sweep quota per wave. exp_03's best point was `C=16`
(`overnight/aug12/exp_03_m15_slab_combine/result.md`). The M18-replication variant of the same
ladder is `~/mok_csweep_queue.sh` (adds `K0_MOK_ROUTE_HIST`, `K0_MOK_REP_EXPERTS`,
`K0_PADMAX=263648`, `DHK_ROOT=~/DHK-m18`); **plain M15 uses `K0_PADMAX` default 263136 and no
route hist** — do not mix.

**Mechanics that matter for R7** (all from `run_campaign.sh`):

- Arg 1 = tag, arg 2 = run_count. Each *run* executes **all** arms (order rotates by
  `(run-1) % n_arms`), so `run_count=1` = one process launch = one R7 invocation. Restarting the
  process between invocations is required anyway (`pperr` is sticky and never host-reset).
- Per-run wrapper: `setsid -w timeout --signal=TERM ${K0_MOK_RUN_TIMEOUT:-2400} docker run --rm
  --name subvadla_k0_mok_<tag>_r<run> …`. Container name is deterministic — that is the handle for
  `docker exec` forensics and for `docker kill` cleanup.
- The container carries `--cap-add SYS_PTRACE --security-opt seccomp=unconfined`, so **rocgdb runs
  inside the container** (`/opt/rocm/bin/rocgdb` exists on the host too, but host ptrace of the
  container's root-owned ranks is blocked for `subvadla` — `ptrace_scope=1`). `docker exec
  <container> rocgdb -p <rank pid>` is the working attach path.
- `runN.log` is `tee`'d **all-rank** torchrun stdout → the LAW-61c "capture all 8 ranks"
  requirement is satisfied by the harness itself; rank-N-only failures are visible there and
  nowhere in `summary.json`.
- Preconditions: exits 20/21 unless all 8 GPUs read 0% use and no `torchrun`/`mpirun`/`orterun` is
  alive; holds a `mkdir` lease at `/tmp/k0_mok_synthetic_gpu_lock` (exit 6 if held) that a `kill
  -9` will strand — **rmdir it before every invocation**; refuses a pre-existing
  `summary.json`/`runN`/`runN.log` (exits 7/22); aborts with exit 23 if a run does not produce
  exactly 8 rank JSONs.
- VOID vs CLEAN discriminator (HARNESS_MAP §7): a run with **no `[MARK]` line and no 8 rank
  JSONs** measured nothing (malformed `K0_MPS_CFG`, rejected at import). Every R7 row records
  `mark_lines` and `rank_json` so VOID can never be scored as a hang.
- Timing: ~38–90 s per run warm-cache for 2 arms at 500 warmup / 100 timed iters, plus the
  600-epoch soak.

## 3. Session instruments

- `~/anatomy_g0/clocklog.sh <csv>` — 0.5 Hz `amd-smi metric -g 0..7 -c -t --csv` sampler
  (one `amd-smi` call = 0.15 s), emitting `ts,gpu,gfx0_clk_MHz,edge_C,hotspot_C,mem_C`. Runs
  alongside every timed job (GPU2 runs hot; 4–5 % clock spread is expected).
- `~/anatomy_g0/rho/run_rho_ladder.sh` → `rho_ladder.jsonl` + `rho_ladder_raw.txt`.
- `~/anatomy_g0/r7/run_r7.sh` → `r7_runs.jsonl` + per-invocation `<tag>.log` + `<tag>_rocgdb.txt`.
- `~/anatomy_g0/r7/rocgdb_dump.sh <container> <outfile>` — container-side rocgdb attach:
  `info inferiors/threads`, `thread apply all bt 25`, `info agents/queues/dispatches/wavefronts`,
  `detach`. `info wavefronts` is the parked-PC view for `hkp::grid_barrier` (no source in-repo).
- `~/anatomy_g0/chain_r7.sh` — setsid/nohup chain so the queue survives an ssh drop.

**Session discard rule honoured:** the first invocation of the session
(`m25_boundary_bench compute … k_inner=2048`, median 1,357.1 µs) is discarded and is not in
`runs.jsonl`; every ladder number below comes from later invocations.

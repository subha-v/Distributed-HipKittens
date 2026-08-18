# Captured-route replay — the decisive experiment we have never run

**Status:** code complete, locally validated, never executed on GPU.
**Owner action:** deploy (§2), capture (§3), verify (§4), run the grid (§5), read §6.

## 0. What this settles

Every kernel-level MoK number we own was produced by one of two route
generators:

| generator | what it fixes | what it destroys |
|---|---|---|
| balanced (default) | nothing | everything |
| `K0_MOK_ROUTE_HIST` (Gumbel-top-k) | the **marginal** 256-bin expert popularity | per-chunk composition, token-to-token correlation |

Both draw every token independently. Real chunks do not work that way:
adjacent tokens of a prompt route alike, so a contiguous 4,096-row chunk
concentrates destinations far past the aggregate. **The claim that this
run-correlation degrades our transports is inference from MORI source
comments. It has never been measured on our kernel.**

It is also the only surviving explanation for our worst contradiction:
aggregate-histogram replay says the un-replicated M15 mega is **0.8467×
production** (18% faster), while the M20 serving pair measured the same
un-replicated mega losing **~1.8–2×** per layer on real prompts
(`overnight/aug14/M20_SERVING_RESULTS.md`). One of those numbers is not
measuring the workload. This experiment finds out which.

Two arms, one multiset:

* **captured** — the router's rows, verbatim, in the order it emitted them.
* **shuffled** — the same rows, token order permuted with a fixed seed. Every
  expert count, every destination-rank total, the entire aggregate histogram
  is *bit-identical* (the loader asserts it and publishes
  `route_multiset_sha256`). Only the within-chunk correlation is gone.

A captured-vs-shuffled delta therefore has exactly one possible cause.

## 1. What was built

| file (repo-relative, all under `distributed-kernels/fused_moe/overnight/aug18-prefill/`) | role |
|---|---|
| `route_capture/skewhook_v2/sitecustomize.py` | serving hook: v1 histograms verbatim **+** raw per-chunk route capture → `$M15_SKEW_OUT/routes/routes_rank<N>_pid<PID>.npz` |
| `route_capture/mok_patch/mok_synthetic_prefill/route_replay.py` | **new** loader: validation, layer filter, fixed-seed shuffle, provenance + per-call skew stats, CLI |
| `route_capture/mok_patch/mok_synthetic_prefill/synthetic_inputs.py` | `MoKSyntheticConfig.route_file/route_order/route_seed/route_layers/route_max_calls`; captured call 0 becomes the setup + reference route |
| `route_capture/mok_patch/mok_synthetic_prefill/run_campaign.sh` | forwards the new env, bind-mounts the host routes dir at `/routes` |
| `route_capture/mok_patch/prefill_opt_host/e004pf_k0pf_ab.py` | `K0_MOK_ROUTE_*` env + legality gates + device-resident route stack + the per-iteration swap in `_mok_measure_arm` |
| `route_capture/mok_patch/CHANGES.diff` | `diff -u` of all four against the deployed node sources |
| `route_capture/test_route_loader.py` | 22 CPU-only tests — **all pass locally (numpy 2.2.6)** |

Capture bound: 64 calls × (4096×8 uint8 + 4096×8 fp16) ≈ **6 MB resident per
rank**, ~17 MB on-device after the harness widens it to int32/fp32.

### Mechanism, in one paragraph

The symmetric route buffers **are** the live router outputs for every arm
(`R["k0d"]["route_input_contract"]`). Writing captured call *k* into them in
place is exactly the in-place route swap the k0d route-swap gate already
validates: no allocation, no address moves, both arms see the same route on
the same iteration. The swap is issued **between** HIP events
(`end.record()` of *i* → swap → `start.record()` of *i+1*), so it is never
inside a measured window. Only route-live arms are permitted
(`production`, `pf6gm_mega`, `pf6c_mega`, `mps_mega`); a frozen-plan arm
would keep executing call 0's plan while the buffers cycle and would produce
a fast, wrong, entirely plausible number — the harness refuses it. After the
timed loop the harness restores call 0 and runs one untimed epoch, because
`ref` was built from call 0 and every gate compares against it.

Guards that fail the run rather than mis-measure it:
`K0_MOK_ROUTE_FILE` requires `K0_INPUT_MODE=mok_synthetic` **and**
`K0_BENCHMARK_PROTOCOL=mok_eager` (the graph protocol gates each replay
against a single-route reference); combining it with `K0_MOK_ROUTE_HIST` is a
hard error; shape/dtype/id-range/format-version mismatches raise at load.

## 2. Deploy

### 2a. Serving hook → node

```bash
# from the laptop
scp -P 2425 \
  distributed-kernels/fused_moe/overnight/aug18-prefill/route_capture/skewhook_v2/sitecustomize.py \
  subvadla@10.5.95.87:~/eplb_campaign/skewhook_v2/sitecustomize.py
```

It is a drop-in replacement for the v1 `skewhook/sitecustomize.py`: same env
(`M15_SKEW_OUT`), same JSON dumps, same filenames. v1 behavior is preserved
byte-for-byte; the raw capture is additive and gated on
`M15_ROUTE_CAPTURE_MAX > 0`.

### 2b. MoK patch → `~/amd-master`, branch `route-replay`

```bash
ssh -p 2425 subvadla@10.5.95.87
cd ~/amd-master && git checkout -b route-replay          # off whatever the harness pin is
K0=~/amd-master/auto-gpu-kernel/k0_fused_moe
# copy the four files (scp them up first, or apply CHANGES.diff with `git apply`)
cp mok_patch/prefill_opt_host/e004pf_k0pf_ab.py        $K0/prefill_opt/host/
cp mok_patch/mok_synthetic_prefill/synthetic_inputs.py $K0/benchmarks/mok_synthetic_prefill/
cp mok_patch/mok_synthetic_prefill/route_replay.py     $K0/benchmarks/mok_synthetic_prefill/
cp mok_patch/mok_synthetic_prefill/run_campaign.sh     $K0/benchmarks/mok_synthetic_prefill/
python3 -m py_compile $K0/prefill_opt/host/e004pf_k0pf_ab.py \
  $K0/benchmarks/mok_synthetic_prefill/{synthetic_inputs.py,route_replay.py}
bash -n $K0/benchmarks/mok_synthetic_prefill/run_campaign.sh
git -C ~/amd-master add -A && git -C ~/amd-master commit -m "MoK: captured-route replay (K0_MOK_ROUTE_FILE)"
```

`CHANGES.diff` was generated against the sources as deployed on 2026-08-17.
If `git apply` rejects a hunk, the node tree has drifted — diff before
overwriting; the four files are self-contained and the edits are additive.

**Zero-change check** (do this once, it is the ratchet): with no
`K0_MOK_ROUTE_*` env set, `R["route_replay"] == {"enabled": false}` and every
new branch is skipped. A `production,mps_mega` campaign with no route env
must reproduce the banked balanced ratio (≈0.7556) and the banked hist ratio
(≈0.8467 with `K0_MOK_ROUTE_HIST`). If either moved, stop — the patch is not
inert and nothing downstream is interpretable.

## 3. Capture pass (serving)

Capture from the **stock arm** — per the aug14 record the m15 arm ran ~2×
traffic from gate self-tests and diluted its own histogram.

```bash
mkdir -p ~/eplb_campaign/routes
M15_SKEW_HOOK=~/eplb_campaign/skewhook_v2 \
M15_SKEW_OUT=/results/skew \
M15_ROUTE_CAPTURE_MAX=64 \
M15_ROUTE_CAPTURE_TOKENS=4096 \
M15_ROUTE_CAPTURE_FLUSH=16 \
  ~/eplb_campaign/<driver>.sh --skew-pass --arm stock
```

The driver mounts `$M15_SKEW_HOOK` and puts it on `PYTHONPATH` inside each
worker. If your driver spells the flag differently, the only requirements are:
the directory holding `sitecustomize.py` is on the workers' `PYTHONPATH`, and
`M15_SKEW_OUT` points at a mounted, writable directory. Equivalent raw form:

```bash
docker run ... -v ~/eplb_campaign/skewhook_v2:/skewhook_v2:ro \
  -v ~/eplb_campaign/results:/results \
  -e PYTHONPATH=/skewhook_v2 -e M15_SKEW_OUT=/results/skew \
  -e M15_ROUTE_CAPTURE_MAX=64 ...
```

The capture completes after 64 full-size router calls — a few seconds of real
traffic — and is flushed to disk every 16 captures under atomic rename, so it
survives any teardown path (this is the same failure that cost two runs in
August: teardown-only dumps never landed).

Collect:

```bash
cp ~/eplb_campaign/results/skew/routes/routes_rank*_pid*.npz ~/eplb_campaign/routes/
ls -la ~/eplb_campaign/routes/     # expect 8 files, ~6 MB each, one per DP worker
```

**A short capture is fine, an empty one is a stop.** If `routes/` is empty,
read `route_capture.skips` in any `skew_rank*_pid*.json`:
`shape` ≫ 0 means no router call had exactly 4,096 rows (check the driver's
`--max-num-batched-tokens`; if the real chunk is a different size, set
`M15_ROUTE_CAPTURE_TOKENS` to it **and** `K0_T` to the same value in §5);
`capturing` ≫ 0 means the calls ran inside a CUDA-graph capture (the harness
skips those deliberately — a D2H copy inside a capture is illegal); `error`
≫ 0 is a bug, dump the worker stderr.

## 4. Verify the captures before burning GPU time

```bash
cd ~/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill
for r in 0 1 2 3 4 5 6 7; do
  python3 route_replay.py --file ~/eplb_campaign/routes --rank $r --order captured \
    | python3 -c 'import json,sys; m=json.load(sys.stdin); print(m["route_file"], m["route_calls"], m["route_skew"])'
done
python3 route_replay.py --file ~/eplb_campaign/routes --rank 0 --order shuffled > /tmp/shuf.json
python3 route_replay.py --file ~/eplb_campaign/routes --rank 0 --order captured > /tmp/capt.json
python3 - <<'PY'
import json
a=json.load(open('/tmp/capt.json')); b=json.load(open('/tmp/shuf.json'))
assert a["route_multiset_sha256"]==b["route_multiset_sha256"], "shuffle changed the multiset"
print("multiset identical:", a["route_multiset_sha256"][:16])
print("captured p95 max_rank/uniform", a["route_skew"]["max_rank_x_uniform_p95"])
print("shuffled p95 max_rank/uniform", b["route_skew"]["max_rank_x_uniform_p95"])
print("aggregate max_rank/uniform  ", a["route_skew"]["max_rank_x_uniform_aggregate"])
PY
```

What you are reading: `max_rank_x_uniform_*` is the share of routed rows
landing on the busiest destination rank, over fair share. The **aggregate**
value is what `K0_MOK_ROUTE_HIST` reproduces; the **p95** is what the kernel
actually pays chunk by chunk, and wall-time is convex in it. If captured p95
≈ shuffled p95 ≈ aggregate, this workload has no run-correlation to measure
and the grid below will (correctly) return three identical ratios — that is a
result, not a failure, and it kills the hypothesis cheaply.

Also run the unit test on the node (it is CPU-only, needs no GPU lease):

```bash
python3 ~/amd-master/.../route_capture/test_route_loader.py     # 22 tests, expect OK
```

## 5. The decisive grid

Six 5-run campaigns (~139 s per run + build). All use
`K0_MOK_ARMS=production,mps_mega`, `K0_T=4096`, `K0_BENCHMARK_PROTOCOL=mok_eager`
(set by the script). **Reuse the exact `K0_MPS_CFG` / `K0_PF6GM_G` of the
banked aug14 skew campaigns (`skewA–E_*_0814T0456`)** — a different mega
config makes the new numbers incomparable to the 0.8467 baseline.

`K0_MOK_TIMED_ITERS=128` gives exactly two full passes over 64 captured
calls, so every real chunk carries equal weight in the median.

```bash
cd ~/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill
export K0_MOK_ARMS=production,mps_mega
export K0_MPS_CFG=<banked value>          # same as skewA-E
export K0_MOK_TIMED_ITERS=128
export K0_MOK_ROUTE_HOST_DIR=~/eplb_campaign/routes   # bind-mounted at /routes

# A — balanced control (no route env): expect ~0.7556
env -u K0_MOK_ROUTE_FILE -u K0_MOK_ROUTE_HIST bash run_campaign.sh rr_balanced_$(date -u +%m%dT%H%M) 5

# B — aggregate histogram, same session: expect ~0.8467
K0_MOK_ROUTE_HIST=route_hists/stock0814_aggregate.json \
  bash run_campaign.sh rr_hist_$(date -u +%m%dT%H%M) 5

# C — CAPTURED (the experiment)
K0_MOK_ROUTE_FILE=/routes K0_MOK_ROUTE_ORDER=captured \
  bash run_campaign.sh rr_captured_$(date -u +%m%dT%H%M) 5

# D — SHUFFLED (same multiset, correlation destroyed)
K0_MOK_ROUTE_FILE=/routes K0_MOK_ROUTE_ORDER=shuffled K0_MOK_ROUTE_SEED=1234 \
  bash run_campaign.sh rr_shuffled_$(date -u +%m%dT%H%M) 5

# E — CAPTURED on the m17 RR-scatter pin (ordering lever)
DHK_ROOT=~/DHK-m17 K0_MOK_ROUTE_FILE=/routes K0_MOK_ROUTE_ORDER=captured \
  bash run_campaign.sh rr_captured_m17_$(date -u +%m%dT%H%M) 5

# F — SHUFFLED on m17 (only if E moved; isolates m17's effect on correlation)
DHK_ROOT=~/DHK-m17 K0_MOK_ROUTE_FILE=/routes K0_MOK_ROUTE_ORDER=shuffled \
  bash run_campaign.sh rr_shuffled_m17_$(date -u +%m%dT%H%M) 5
```

Before E: confirm the pin really has the RR scatter on —
`grep -n "K0P6_M15_SCATTER_RR" ~/DHK-m17/distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip`
must show the default at `1` in that worktree (the flag defaults to 0 on
`ablations`; a RUN PIN worktree is exactly where it is flipped). Then confirm
after the run that `kernel_source_sha256` in the E summary differs from C's —
if it does not, E rebuilt the wrong tree and the arm is void.

Optional diagnostic (do **not** substitute it for C):
`K0_MOK_ROUTE_LOCKSTEP=1` adds a rank barrier between timed iterations so all
ranks replay the same captured call simultaneously. Off (default) ranks may
drift a call apart and two real chunks mix, which is the honest steady state;
on, the joint per-step load is exact but the loop's overlap is drained. Run it
only if C and D come out close and you suspect drift smeared the tail.

Read each result from `summary.json`:
`ratio = mps_mega p50 / production p50` (below 1 = mega faster). Also confirm
in any rank JSON that `route_replay.route_multiset_sha256` is **identical**
between C and D — that equality is the proof the two arms differ only in
correlation.

## 6. Decision table

Let **r_bal ≈ 0.756**, **r_hist ≈ 0.847** (banked), and r_cap / r_shuf from C
and D. Serving implies **r ≈ 1.8–2.0** for the un-replicated mega on real
prompts (M20 pair, per-layer).

| outcome | reading | what it implies | next move |
|---|---|---|---|
| r_cap ≈ r_shuf ≈ r_hist | replay fidelity does not matter at all | **Hypothesis dead.** Run correlation is not a transport degrader on our kernel, and the aggregate histogram is a sound proxy. The serving gap is then NOT routing: suspect per-layer variance, KV-pool pressure, graph/eager interaction, or the shim's activation boundary | stop optimizing against skew; re-attribute the serving regression with the profiler pass (Phase B of the M15 handoff) |
| r_cap ≫ r_shuf ≈ r_hist | correlation itself is the cost | **Hypothesis confirmed.** Contiguous chunks concentrate destinations; our transport is convex in that concentration and the aggregate replay has been lying to us since aug14 | E decides the lever: see the two rows below |
| …and r_cap(m17) ≪ r_cap | RR landing order recovers it | the mechanism is **peer-ordering serialization** inside expert blocks — a span hostage to one source rank | ship m17 (`K0P6_M15_SCATTER_RR=1`) as default; re-sweep C under skew |
| …and r_cap(m17) ≈ r_cap | ordering is irrelevant | the mechanism is **concentration**, not order | per-chunk adaptive replication (M19 direction, theta) or finer slab certification; the aug14 m17-vs-C decision rule resolves to C |
| r_cap ≈ r_shuf ≫ r_hist | per-chunk **popularity** (not correlation) is worse than the aggregate | chunk-level popularity variance is the degrader — the M20 "uniform per-layer damage / convex tail" finding, now confirmed at kernel speed | retire the aggregate histogram as a predictor; the replay corpus becomes the standard MoK route source; M19 adaptive replication is the direct fix |
| r_shuf > r_cap | correlation **helps** us (dispatch locality) | i.i.d. replay is a *pessimistic* bound, and the serving gap is elsewhere entirely | treat r_hist as a lower bound; go back to the profiler |
| r_cap lands near 1.8–2.0 | kernel replay reproduces serving | **the bench is finally predictive**: 139 s per campaign instead of 25 min per serving pair | move the whole optimization loop onto captured replay before touching serving again |
| any campaign blocked on gates | correctness, not performance | check `mok_eager.status`; the restore-call-0 epoch should make gates route-consistent | if gates fail only under replay, capture `route_replay` metadata + the failing arm and treat as a harness bug, not a kernel result |

The single most valuable line in the output is **C vs B**. B is the number
every design decision since aug14 was made against; C is the same kernel on
the same GPUs against the routing the model actually produced.

## 7. Notes and limits

* Captured call 0 is the setup route, so `ref`, all pre-timing gates and the
  restore epoch are anchored to a **real** chunk, not a synthetic one.
* `hidden` and every weight tensor are byte-identical to the balanced arm:
  the replay path draws and discards the default logits to keep the generator
  stream aligned. Balanced / captured / shuffled differ **only** in routing.
* Router weights are replayed as captured (fp16 → fp32), not re-softmaxed.
  Combine numerics therefore see real weight magnitudes.
* Each rank replays **its own** captured file, so the joint per-step load
  across ranks is the real one; ranks whose capture is short bound the cycle
  (`min` all-reduce), and the count actually used is in
  `config.mok_route_calls`.
* `K0_MOK_REP_EXPERTS=<count>` still requires `K0_MOK_ROUTE_HIST` to pick its
  top-K (unchanged). To run M18/M19 against captured routes, pass the replica
  set explicitly as csv; deriving it from the capture is a follow-on.
* Only `mok_eager` is supported, by design. Under the graph protocol every
  replay is gated against a one-route reference and cycling would fail those
  gates — refusing is correct, not a limitation to route around.

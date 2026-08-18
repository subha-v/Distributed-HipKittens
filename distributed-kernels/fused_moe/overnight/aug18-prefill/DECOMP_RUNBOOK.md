# DECOMP RUNBOOK — localizing the real-routing gap at kernel level

**Written:** 2026-08-18, laptop, no node access. **Executor:** operator on the
8× MI350X node the moment the serving campaign releases the GPUs.
**Status of every command below:** transcribed from the deployed sources in this
repo and the campaign wrappers in the session scratchpad. Nothing here has been
run. Every line marked **⚠ GAP** is something the plan needs that does not exist
yet or that I could not verify without the node.

---

## 0. The number we are chasing

| measurement | value | source |
|---|---|---|
| rescued-stock serving | 20,918 tok/s, 201 s wall | m23pair1 / 1_stock |
| m15+M23 serving | 19,277 tok/s, 218 s wall (**−7.8 %**) | m23pair2 / 1_m15 |
| mega coverage that produced it | sealed 1,976 / in_bucket 1,993 = **99.15 %**, 1,960 ragged + 16 exact, eager_b4096 17/8 ranks | `seal_receipts.txt` |
| back-solved MoE region ratio | **1.15–1.20×** production at f ≈ 0.40–0.54 | arithmetic on the above |
| banked kernel ratio, balanced | **0.7556×** (M15 5,823 µs) | `overnight/aug14/M18_REPLICATION_RESULTS.md:26` |
| banked kernel ratio, aggregate-histogram skew | **0.8467×** (M15 20,739 µs) | same, line 27 |

The gap to localize is **0.85 → 1.17**, i.e. the mega loses ~35–45 % of region
time relative to what every banked replay predicts.

### 0.1 Two numbers I recomputed from the receipts before designing the grid

Parsed from both arms' `seal_receipts.txt` (last receipt per rank, 300 steps ×
8 ranks):

```
p2_m15 : rank-steps=2400  real_tok=3,265,348   mean real tok / rank-step = 1,361
         b4096 rank-steps=1976  padded rows dispatched = 8,093,696
         all real tokens / padded rows = 40.3 %
         in_bucket_sum_orig=24,538,534 over 1,993 rank-steps (8× peer-counted)
         => real tokens per rank per sealed step = 1,539   fill = 37.6 % of 4096
         => PADDED/REAL WORK MULTIPLIER = 2.66×
p1_stock: mean real tok / rank-step = 1,385 ; in-bucket fill 43.3 % (2.31×)
```

**This is the single largest previously-unnamed candidate mechanism.** Every
banked kernel number is `T=4096` **real** rows per rank. Serving runs the mega
on 4,096 **padded** rows carrying ~1,539 real ones. If production's MoE path
does *not* pay the same padding (or pays a smaller fixed-cost share of it), the
whole 0.85 → 1.17 move can come from fill, with zero contribution from routing.
No captured-route experiment can see this, because the replay corpus is
`[4096, 8]` by construction. **That is why the grid below adds a T-sweep (R6)
and ranks it above the remedy arms.**

Mechanism labels used throughout: **(a)** destination concentration degrading
LL128 dispatch / remote-RMW combine; **(b)** per-call compute concentration
(hot-rank critical path vs non-shrinking fixed costs); **(c)** per-step
integration overhead now paid at 99 % duty; **(d)** all-8-eager fallback
(measured 17/2400 rank-steps = 0.7 %, already too small to matter — **treat (d)
as refuted by the receipts, do not spend GPU time on it**); **(e)** ragged/padded
fill (the 2.66× above).

---

## 1. Preconditions and node discovery (5 min, no GPU)

```bash
ssh -p 2425 subvadla@10.5.95.87

# 1.1 What DHK pins exist for the MoK harness?
ls -d ~/DHK-* ~/fabric_build/* 2>/dev/null
for d in ~/DHK-*; do
  printf '%s : ' "$d"
  git -C "$d" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "(not a git tree)"
done

# 1.2 Which kernel does each pin compile under the harness's mps entry point?
for d in ~/DHK-*; do
  printf '%s SCATTER_RR/KERNEL_NAME in mps tile:\n' "$d"
  grep -n "K0P6_M15_KERNEL_NAME\|^#define K0P6_M15_SCATTER_RR\|^#define K0P6_M15_POLL_BACKOFF\|^#define K0P6_M15_RMW_INTERLEAVE" \
    "$d/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" 2>/dev/null | head -5
done

# 1.3 Is the route-replay MoK patch deployed? (ROUTE_REPLAY_PLAN says: never run)
K0=~/amd-master/auto-gpu-kernel/k0_fused_moe
grep -c K0_MOK_ROUTE_FILE $K0/prefill_opt/host/e004pf_k0pf_ab.py \
                          $K0/benchmarks/mok_synthetic_prefill/run_campaign.sh
ls $K0/benchmarks/mok_synthetic_prefill/route_replay.py 2>/dev/null || echo "LOADER MISSING"

# 1.4 The serving campaign's own dirs
ls ~/eplb_campaign/
ls $PACKET/benchmarks/2026-08-12_m15_campaign/ 2>/dev/null
```

### 1.1 THE PIN TRAP — read this before running any MoK campaign

The MoK harness's `mps_mega` arm compiles **`k0pf6gm_device_tile_mps.hip`**
(`e004pf_k0pf_ab.py:480, 997`) and loads the symbol `k0pf6gm_mps_mega`
(`PF6MPS_NAME`, line 386). On the `ablations` tree that file is the *old MPS
sibling* (mode ≤ 2, the 0.898× donor) — **not** the M15 chassis that produced
0.7556 / 0.8467 and not the kernel serving runs. The M15 chassis is compiled
under the harness's name only in a RUN PIN worktree, which prepends

```c
#define K0P6_M15_KERNEL_NAME k0pf6gm_mps_mega
```

and then inlines `k0pf6gm_device_tile_m15.hip` **verbatim** (not `#include` —
the mori JIT content hash does not hash `-I` paths; this is the exp_04 trap).
Verified in this repo: `ablations-m15:…_mps.hip:10` and
`ablations-m17:…_mps.hip:8-9` (m17 additionally sets `K0P6_M15_SCATTER_RR 1`).

**Consequence:** every campaign below must pass an explicit `DHK_ROOT=<pin>`.
Running with the default `~/Distributed-HipKittens` measures a different kernel
and every ratio is void.

**⚠ GAP / correction to `ROUTE_REPLAY_PLAN.md:245`:** that line tells you to
grep `k0pf6gm_device_tile_m15.hip` in `~/DHK-m17` to confirm RR is on. On the
m17 branch that file still shows `#define K0P6_M15_SCATTER_RR 0` (it is the
donor body; the flag is flipped in the **mps** tile). **Grep the mps tile**, as
in §1.2 above.

### 1.2 Config: C=28, not C=24

The banked 0.7556 / 0.8467 pair was measured at
**`C=28, g=353, mode=12, flush_rows=16`** (`M18_REPLICATION_RESULTS.md:24`).
The task brief says C=24; exp_33's phase ledger was C=16 on the *old* mps
ratchet. These are three different configurations. Since the only numbers we
are trying to reconcile are 0.7556 / 0.8467, **use C=28** and let R0a prove it:

```bash
export MPSCFG="C=28,g=353,mode=12,flush_rows=16"
```

If R0a's balanced ratio does not land within ±1 % of 0.7556, stop and sweep
C ∈ {24, 28} before interpreting anything downstream.

**⚠ GAP:** exp_33's banked phase ledger (plan 372.79 / M6 2,453.30 / M7
2,701.84 / combine 324.21 µs) is **balanced-route, C=16, old mps ratchet** —
`overnight/aug11/exp_33_attribution/result.md:5` records
`K0_MPS_CFG="C=16,g=353,mode=12,flush_rows=16,timestamps=1"` and a 0.8424×
*balanced* ratio. It is **not** an aggregate-skew ledger for the M15 chassis, so
it cannot be the comparison baseline for R3. R3 therefore measures its own
balanced M15 ledger in-session (R3a) instead of citing exp_33.

---

## 2. PHASE A — the capture pass (1 server lifetime, ~20 min GPU)

### 2.1 Deploy the v2 hook — and the two wrapper defects that will silently kill it

```bash
# from the laptop
cd /Users/subha/repos/Distributed-HipKittens
scp -P 2425 \
  distributed-kernels/fused_moe/overnight/aug18-prefill/route_capture/skewhook_v2/sitecustomize.py \
  subvadla@10.5.95.87:~/eplb_campaign/skewhook_v2/sitecustomize.py
```

**Defect 1 — `M15_SKEW_HOOK` is overwritten by the wrapper.** Both
`run_m15_campaign_eplb_v2.sh:568-570` and `…_v3.sh:612-614` do:

```bash
M15_SKEW_HOOK="$(cd "$CAMPAIGN_SRC_DIR" && pwd)/skewhook"
mkdir -p "$M15_SKEW_HOOK"
cp "$CAMPAIGN_SRC_DIR/m15_router_skew.py" "$M15_SKEW_HOOK/sitecustomize.py"
export M15_SKEW_HOOK
```

Exporting `M15_SKEW_HOOK=~/eplb_campaign/skewhook_v2` before invoking the driver
does **nothing** — `skew_pass()` clobbers it and installs the v1 hook, which has
no raw route capture. The zero-edit fix is to make v2 *be* the file the wrapper
copies:

```bash
ssh -p 2425 subvadla@10.5.95.87
CSD="$PACKET/benchmarks/2026-08-12_m15_campaign"     # see §3 for PACKET
cp "$CSD/m15_router_skew.py" "$CSD/m15_router_skew.py.v1.bak"
cp ~/eplb_campaign/skewhook_v2/sitecustomize.py "$CSD/m15_router_skew.py"
python3 -m py_compile "$CSD/m15_router_skew.py"      # must be silent
grep -c M15_ROUTE_CAPTURE_MAX "$CSD/m15_router_skew.py"   # expect >= 1
```

(v2 is a byte-compatible superset of v1: same `M15_SKEW_OUT` gate, same JSON
dumps, same filenames — `route_capture/skewhook_v2/sitecustomize.py:24`.)

**Defect 2 — `M15_ROUTE_CAPTURE_*` is not forwarded into the container.** The
docker block forwards only `M15_SKEW_CAPTURE_LAYERS` and `M15_SKEW_CAPTURE_MAX`
(v2 wrapper lines 330-331, v3 lines 373-374). The raw-capture knobs are absent.
**No wrapper edit is required for the default capture**, because the hook's own
defaults are exactly the values `ROUTE_REPLAY_PLAN.md:126-129` asks for:

| env | hook default (`sitecustomize.py:104-106`) | plan's requested value |
|---|---|---|
| `M15_ROUTE_CAPTURE_MAX` | 64 | 64 |
| `M15_ROUTE_CAPTURE_TOKENS` | 4096 | 4096 |
| `M15_ROUTE_CAPTURE_FLUSH` | 16 | 16 |

If you want anything other than the defaults (see §2.3), edit the constants at
the top of the deployed `m15_router_skew.py` **in place** — that is strictly
less invasive than editing the wrapper's docker block.

**Defect 3 (informational) — `skew_pass()` never touches
`/tmp/vllm-pf4h-enable`.** The main pair loop does (`…_v3.sh` line ~707) but the
skew pass does not, so on the `m15` arm the megakernel is installed and never
activated. Irrelevant for us: we capture from the **stock** arm per
`ROUTE_REPLAY_PLAN.md:121` (the m15 arm doubles its own traffic with gate
self-tests and dilutes its histogram). It does mean **do not** try to read
serving performance out of the capture lifetime.

### 2.2 Run the capture — M23 era, patch chain on, stock arm, zero pairs

Reuse `camp3_stock_vs_m15_ragged.sh`'s environment **verbatim** — that is what
"scheduler behavior matches the M23 era" means operationally. `PAIRS=0` makes
the main pair loop a no-op, so the only server lifetime is the skew pass.

```bash
ssh -p 2425 subvadla@10.5.95.87
mkdir -p ~/eplb_campaign/routes

export RUN_TAG=routecap1
export DATE_TAG=20260818
export PACKET=/home/subvadla/amd-master-m15pkt/auto-gpu-kernel/k0_fused_moe/vllm_r1_aiter_e2e
export N2=/home/subvadla/pf4h_vllm_20260729/isolation_v1_20260729/n2/k0_n2.cpython-312-x86_64-linux-gnu.so
export M15_SOURCES=/home/subvadla/m20_deploy_sources_20260814
export DHK_ROOT=/home/subvadla/DHK-m20pkt            # SERVING pin — not the MoK pin
export CLIENT=/home/subvadla/pf4h_vllm_20260729/bench_exact_token_ids_v2.py
export GPU_CLAIM_NAME=m15pkt
export QSL_PKL=/home/subvadla/mlperf_v6_datasets/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl
export PROMPT_SOURCE=qsl
export M23_PATCH=/home/subvadla/eplb_campaign/m23_patch.py     # THE M23 ERA
export EPLB_PREWARM=1

bash /home/subvadla/eplb_campaign/run_m15_campaign_eplb_v2.sh \
  --arms stock --cells c32p --pairs 0 --skew-pass \
  2>&1 | tee ~/eplb_campaign/routecap_20260818.log
echo "ROUTECAP_RC=$?"
```

Expected artifacts:

```
/home/subvadla/20260818_m15_campaign_routecap1/skew/stock/skew/
    skew_rank<N>_pid<PID>.json          # v1 histograms, unchanged
    routes/routes_rank<N>_pid<PID>.npz  # THE NEW ARTIFACT, ~6 MB × 8
```

Collect:

```bash
cp /home/subvadla/20260818_m15_campaign_routecap1/skew/stock/skew/routes/routes_rank*_pid*.npz \
   ~/eplb_campaign/routes/
ls -la ~/eplb_campaign/routes/       # expect 8 files, ~6 MB each
```

### 2.3 ⚠ GAP — the fidelity risk that decides whether Phase A works at all

`sitecustomize.py:180-184` skips any call made inside a CUDA-graph capture
(`route_skips["capturing"]++`) — a D2H copy inside a capture is illegal. Its
comment asserts "4096-token prefill chunks run eager anyway". **The M23 receipts
say otherwise:** `eager_b4096` is 1–3 per rank against ~250 sealed b4096 steps —
**≈1 % of full-size chunks run eager**; the other 99 % are graph replays, during
which `select_experts`'s Python body never executes at all.

Arithmetic that says the capture still fills: each eager b4096 *step* issues one
router call **per MoE layer** (~58 for R1), all at shape `[4096, 8]`. Two eager
steps ⇒ ~116 candidate calls ≥ `ROUTE_MAX=64`. So the default capture should
complete — but it will hold **64 consecutive layers of one or two real chunks**,
not 64 independent chunks. That corpus is still legitimate (each layer's routing
is a distinct realization over real, run-correlated tokens, which is exactly the
property under test), but say so in every result: *cross-layer* variation of one
chunk, not *cross-chunk* variation.

Two contingencies, in order of preference:

1. **If `routes/` is empty or short**, read `route_capture.skips` in any
   `skew_rank*_pid*.json`:
   * `capturing ≫ 0, shape ≈ 0` → graph replay ate the calls. Re-run §2.2 with
     `--enforce-eager` appended to the `vllm serve` line (**⚠ GAP: this is a
     one-line edit to `launch_server()` in the wrapper — it does not exist as a
     flag**), stock arm only. Eager changes step *timing*, not chunk
     *composition* (fcfs + chunked prefill at a fixed `--max-num-batched-tokens
     4096` from a fixed QSL order), and routes depend on hidden states, not on
     graph mode. Validate era-faithfulness afterwards by comparing the eager
     run's `RAGGED_SEAL_RECEIPT` `in_bucket / sum_orig / min_orig / max_orig`
     against m23pair1's — if the orig-token distribution matches, the corpus is
     era-faithful.
   * `shape ≫ 0` → no call had exactly 4,096 rows; set
     `M15_ROUTE_CAPTURE_TOKENS` to the real chunk size **and** `K0_T` to the
     same value in Phase D.
   * `error ≫ 0` → bug; dump worker stderr.
2. **If you take the `--enforce-eager` path**, raise `M15_ROUTE_CAPTURE_MAX` to
   1024 in the deployed hook (≈96 MB/rank resident — still fine) so the corpus
   spans ~17 full layer stacks. Then use `K0_MOK_ROUTE_LAYERS=<one router key>`
   at replay time to get a *cross-chunk* corpus for a single layer, which is the
   stronger test of run-correlation.

**Do not** substitute graph-capture-time calls for real ones: those run on
dummy profiling activations and their routes are meaningless.

---

## 3. PHASE B — verify the capture before burning GPU time (CPU only, ~5 min)

```bash
K0=~/amd-master/auto-gpu-kernel/k0_fused_moe
cd $K0/benchmarks/mok_synthetic_prefill

# per-rank provenance and skew
for r in 0 1 2 3 4 5 6 7; do
  python3 route_replay.py --file ~/eplb_campaign/routes --rank $r --order captured \
    | python3 -c 'import json,sys; m=json.load(sys.stdin); print(m["route_file"], m["route_calls"], m["route_skew"])'
done

# the multiset identity that makes captured-vs-shuffled a one-cause experiment
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

# 22 CPU-only loader tests
python3 ~/amd-master/.../route_capture/test_route_loader.py     # expect OK
```

**Early kill:** if captured p95 ≈ shuffled p95 ≈ aggregate (say within 5 %),
this workload has no run-correlation to measure. **Drop R2, R4, R5 entirely**
and spend the whole night on R6 + R3. That is a result, not a failure, and it is
free.

Also read `route_skew.max_rank_x_uniform_aggregate` against the aug14 measured
value of **5.09× fair share**. If the M23-era aggregate is materially different,
the banked 0.8467 was replaying a *different* workload and the "gap" is partly a
workload drift, not a kernel property — note it in the writeup.

---

## 4. PHASE C — pin construction for the remedy arms (CPU only, ~10 min)

### 4.1 The harness cannot consume a prebuilt hsaco

`run_campaign.sh:186-193` runs `cmake … -DDHK_ROOT="${K0_DHK_ROOT}"` and
`cmake --build` **inside the container**, then the host module JIT-compiles
`k0pf6gm_device_tile_mps.hip` out of the bind-mounted `DHK_ROOT`
(`e004pf_k0pf_ab.py:480, 996-997`). There is no env, no flag, and no code path
that loads an externally built hsaco. **The verified hsacos at `~/fabric_build/`
are for the serving path and are useless to the MoK harness.** Every arm must
arrive as a `DHK_ROOT` source tree — the DHK-m17 pattern, no exception.

### 4.2 Build the three pins from ONE m15 body

Internal consistency matters more than matching any historical pin: the control
and the remedy arms must differ **only** in the flag, so build all of them from
the fabric worktree's `k0pf6gm_device_tile_m15.hip` (which is `ablations` HEAD
+200 guarded lines, statically proven default-equivalent by
`overnight/aug18-prefill/verify_fabric_arms.py`).

```bash
# on the node; FAB = the fabric worktree source tree (branch worktree-wf_b38dfe27-d09-2,
# commit deb863bc "M21 fabric discipline"), already staged at ~/fabric_build/
FAB=~/fabric_build/<the-tree-with-k0pf6gm_device_tile_m15.hip>
grep -c K0P6_M15_POLL_BACKOFF $FAB/distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip   # expect 9

mkpin () {   # $1 = destination pin dir, $2.. = extra #defines
  local dst="$1"; shift
  rm -rf "$dst"; cp -a "$FAB" "$dst"
  local m15="$dst/distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip"
  local mps="$dst/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
  {
    echo "// RUN PIN (generated $(date -u +%FT%TZ)) — M15 chassis under the harness kernel name."
    echo "// Body below is k0pf6gm_device_tile_m15.hip VERBATIM (inline, not #include:"
    echo "// the mori JIT content hash does not hash -I paths — the exp_04 trap)."
    echo "#define K0P6_M15_KERNEL_NAME k0pf6gm_mps_mega"
    for d in "$@"; do echo "#define $d"; done
    cat "$m15"
  } > "$mps"
  echo "built $dst:"; head -8 "$mps" | sed 's/^/    /'
}

mkpin ~/DHK-fab-base                                        # R1/R2/R3/R6 control
mkpin ~/DHK-fab-backoff  "K0P6_M15_POLL_BACKOFF 1"          # R5a
mkpin ~/DHK-fab-rmw      "K0P6_M15_RMW_INTERLEAVE 1"        # R5b (implies SCATTER_RR)
mkpin ~/DHK-fab-rr       "K0P6_M15_SCATTER_RR 1"            # R4 (equivalent to DHK-m17)

# sanity: all four must carry the harness symbol, and only the intended flag
for d in ~/DHK-fab-*; do
  echo "== $d"; grep -n "^#define K0P6_M15_" "$d/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" | head -5
done
```

**⚠ GAP that must be checked before trusting R5:** the M21 fabric flags are
implemented **only** in `k0pf6gm_device_tile_m15.hip`. I verified on the laptop
that the fabric worktree's diff vs `ablations` is
`k0pf6gm_device_tile_m15.hip | 200 +++` and **nothing else** — zero hits for
`POLL_BACKOFF` / `RMW_INTERLEAVE` in `k0pf6gm_device_tile_mps.hip`. The `mkpin`
recipe above is what makes them reachable from the `mps_mega` arm; without it,
`DHK_ROOT=~/fabric_build/... K0P6_M15_POLL_BACKOFF=1` compiles the *old MPS
sibling* and the arm measures nothing. **After every R4/R5 run, diff
`pf6mps.source_sha256` / `hsaco_sha256` in the rank JSON against R1's — if they
match, the arm is void.**

**⚠ GAP:** these flags are compile-time `#ifndef` macros with no host-side env
plumbing in the harness. There is no `-D` forwarding in
`run_campaign.sh`'s cmake block for the JIT'd `.hip`. The pin is the only
mechanism. (If a later session wants env-selectable arms, the change is a
`K0P6_M15_*` → `-D` pass-through in the module JIT call at
`e004pf_k0pf_ab.py:996`, plus a cache-key contribution so the JIT does not alias.)

### 4.3 Deploy the route-replay MoK patch if §1.3 says it is missing

Exactly `ROUTE_REPLAY_PLAN.md` §2b, then its **zero-change ratchet**: with no
`K0_MOK_ROUTE_*` env set, `R["route_replay"] == {"enabled": false}` and R0a must
reproduce 0.7556. If it does not, the patch is not inert and nothing downstream
is interpretable.

---

## 5. PHASE D — the replay grid

Common preamble for every campaign in this section:

```bash
K0=~/amd-master/auto-gpu-kernel/k0_fused_moe
cd $K0/benchmarks/mok_synthetic_prefill

export K0_MOK_ARMS=production,mps_mega
export K0_PF6GM_G=3                 # HARD REQUIREMENT: e004pf_k0pf_ab.py:346-350 raises
                                    # "mps_mega requires its paired pf6gm_mega
                                    #  reference at G=3" if this is not 3.
export MPSCFG="C=28,g=353,mode=12,flush_rows=16"    # see §1.2
export K0_MOK_ROUTE_HOST_DIR=~/eplb_campaign/routes # bind-mounted read-only at /routes
export K0_MOK_OUTPUT_ROOT=/home/subvadla/k0-mok-synthetic-results
TS=$(date -u +%m%dT%H%M)
```

Protocol facts you are relying on (from `BENCHMARKING.md` §1 and
`run_campaign.sh`): `K0_INPUT_MODE=mok_synthetic`, `K0_BENCHMARK_PROTOCOL=mok_eager`
(both set by the script and **required** by the route-replay gate), 500 warmup /
timed iters as set, seed `1234+rank`, index-aligned rank-MAX per iteration,
**median of the rank-max samples** is the statistic, arms rotated one position
per run, MoK correctness gate blocking per arm per run, reference is a fresh
same-run `production` output. Safety interlocks refuse a contended node — do not
defeat them.

`K0_MOK_TIMED_ITERS=128` gives exactly two full passes over 64 captured calls so
every real chunk carries equal weight in the median.

---

### R0a — balanced control / the ratchet (**mandatory, run first**)

```bash
env -u K0_MOK_ROUTE_FILE -u K0_MOK_ROUTE_HIST \
  DHK_ROOT=~/DHK-fab-base K0_MPS_CFG="$MPSCFG" \
  bash run_campaign.sh r0a_balanced_$TS 5
```

**Expect** `candidate_ratios["mps_mega"]["p50"] ≈ 0.7556` (M15 ≈ 5,823 µs).
Out of band ⇒ wrong pin, wrong C, or a non-inert route patch. **Stop and fix.**

---

### R6 — the FILL sweep (**the mechanism nobody has measured; run second**)

Tests **(e)** and **(c)** directly, needs **no capture**, and is therefore the
one arm that still delivers if Phase A fails. Serving runs the mega on 4,096
padded rows carrying 37.6 % real tokens (§0.1). If the mega's fixed costs (single-CTA
planner, C=28 reserved service CTAs, slab rendezvous, M8 certification) do not
shrink with real fill while production's do, the ratio must climb as T falls —
and the whole 0.85 → 1.17 move can be fill, not routing.

```bash
# smoke first: 1 run, prove a non-4096 T is even legal in this harness
DHK_ROOT=~/DHK-fab-base K0_MPS_CFG="$MPSCFG" K0_T=2048 \
  bash run_campaign.sh r6_smoke_t2048_$TS 1

# then the sweep, 3 rotations each (T=4096 is R0a, do not repeat it)
for T in 2048 1024; do
  DHK_ROOT=~/DHK-fab-base K0_MPS_CFG="$MPSCFG" K0_T=$T \
    bash run_campaign.sh r6_t${T}_$TS 3
done
```

`run_campaign.sh` derives `K0_MAXTOK=K0_MAXTOK_PROD=K0_T` (lines 143-144), so
both arms genuinely shrink; `K0_PADMAX=263136` stays generous. **⚠ GAP:** no
banked evidence that `K0_T != 4096` passes the MoK gates — hence the 1-run
smoke. If it fails, capture the failure mode; a `T`-parametric harness is then a
named work item, and it is a cheap one.

**Read:** plot `r(T) = mps_mega_p50 / production_p50` for T ∈ {4096, 2048, 1024}
and also the *absolute* p50s. Two distinct signatures:
* `production_p50` falls ~linearly with T while `mps_mega_p50` flattens ⇒ mega
  fixed cost dominates at real fill ⇒ **(e)/(c) confirmed**, and the fix is
  ragged/variable-T dispatch + fixed-cost reduction, not fabric work.
* both fall together, `r(T)` flat ⇒ **(e) refuted**; padding is symmetric and
  the gap is elsewhere. This retires the biggest confound in one 25-minute pass.

---

### R1 — CAPTURED routes, production vs m15 (**the headline**)

```bash
DHK_ROOT=~/DHK-fab-base K0_MPS_CFG="$MPSCFG" \
  K0_MOK_TIMED_ITERS=128 \
  K0_MOK_ROUTE_FILE=/routes K0_MOK_ROUTE_ORDER=captured \
  bash run_campaign.sh r1_captured_$TS 5
```

`r_cap = candidate_ratios["mps_mega"]["p50"]`. Compare against r_bal = 0.7556,
r_hist = 0.8467, and the serving-implied 1.15–1.20.

---

### R2 — SHUFFLED routes (isolates run-correlation from popularity)

```bash
DHK_ROOT=~/DHK-fab-base K0_MPS_CFG="$MPSCFG" \
  K0_MOK_TIMED_ITERS=128 \
  K0_MOK_ROUTE_FILE=/routes K0_MOK_ROUTE_ORDER=shuffled K0_MOK_ROUTE_SEED=1234 \
  bash run_campaign.sh r2_shuffled_$TS 5
```

**Validity gate:** `route_replay.route_multiset_sha256` in any R2 rank JSON must
equal R1's. That equality is the entire logic of the experiment — without it R1
and R2 differ in more than correlation and the comparison is void.

---

### R3 — the phase ledger (attribution, NOT a scored number)

**⚠ GAP — the ledger does not do what the task brief assumes.** Three
corrections, all verified in the deployed host source:

1. The tag is **`[MPS TS]` / `[MPS TS SPLIT]` / `[MPS TS DELTA]`**, not
   `[K0P6 TS]` (`e004pf_k0pf_ab.py:5584-5593`).
2. It prints **rank 0 only** and reads `pf6_state["mps_state"]`, which is never
   all-reduced and never written to the per-rank JSON. It is a device CTA-max
   *within rank 0* — you cannot get a cross-rank max from it
   (`exp_33/phase_stamps.json:508`). Under per-call load concentration (mechanism
   **b**) the interesting rank is the *hot* one, and this instrument cannot see
   it. **⚠ GAP / work item:** all-reducing the 8 u64 stamps and writing them into
   every rank JSON is a ~15-line host change and would make (b) directly
   measurable. It does not exist.
3. The stamps are running device **maxes that are never reset**, sampled after
   the **600-epoch soak** — they describe the *final soak epoch*, not a timed
   iteration (`phase_stamps.json:92`). Under route replay the soak runs
   **captured call 0 only**, because the per-iteration route swap lives in
   `_mok_measure_arm`. A 64-call rotation therefore produces a ledger for one
   chunk. Do not sum the ledger and compare it to `arm_p50_us`.

Given (3), pin the route so the ledger and the timing describe the same chunk:

```bash
# R3a — balanced ledger for THIS chassis at THIS C (the missing baseline; exp_33 is C=16 on the old ratchet)
env -u K0_MOK_ROUTE_FILE -u K0_MOK_ROUTE_HIST \
  DHK_ROOT=~/DHK-fab-base K0_MPS_CFG="$MPSCFG,timestamps=1" \
  bash run_campaign.sh r3a_ts_balanced_$TS 3

# R3b — captured call 0, single-route: ledger and p50 describe the same real chunk
DHK_ROOT=~/DHK-fab-base K0_MPS_CFG="$MPSCFG,timestamps=1" \
  K0_MOK_ROUTE_FILE=/routes K0_MOK_ROUTE_ORDER=captured K0_MOK_ROUTE_MAX_CALLS=1 \
  bash run_campaign.sh r3b_ts_capt0_$TS 3

# R3c — the highest-skew captured layer (key from Phase B's per-call skew stats)
DHK_ROOT=~/DHK-fab-base K0_MPS_CFG="$MPSCFG,timestamps=1" \
  K0_MOK_ROUTE_FILE=/routes K0_MOK_ROUTE_ORDER=captured K0_MOK_ROUTE_MAX_CALLS=1 \
  K0_MOK_ROUTE_LAYERS=<hot_router_key> \
  bash run_campaign.sh r3c_ts_capthot_$TS 3
```

Harvest:

```bash
for t in r3a_ts_balanced r3b_ts_capt0 r3c_ts_capthot; do
  echo "== $t"; grep -h "MPS TS\|MPS SPIN" $K0_MOK_OUTPUT_ROOT/${t}_$TS/run*.log
done
```

Derived phases (ticks × 0.01 µs): `plan_M3toM5 = M5_DONE − M2_DONE`,
`M6 = M6_DONE − M5_DONE`, `M7 = M7_DONE − M6_DONE`,
`combine = REDUCE_DONE − M7_DONE`, `servicedrain = DRAIN − M6_DONE` (overlaps
M7/combine — **not additive**).

`[MPS SPIN]` is the cheapest transport probe in the whole grid: exp_33 measured
`fail_max=0`, `success_max ≤ 1` against a 2,000,000 limit under balanced routes.
**A nonzero `fail_max` under captured routes is direct evidence for (a).**

---

### R4 — RR scatter (remedy arm; conditional)

```bash
DHK_ROOT=~/DHK-fab-rr K0_MPS_CFG="$MPSCFG" \
  K0_MOK_TIMED_ITERS=128 \
  K0_MOK_ROUTE_FILE=/routes K0_MOK_ROUTE_ORDER=captured \
  bash run_campaign.sh r4_captured_rr_$TS 5
```

**Prior:** aug14 already measured m17/RR under aggregate skew at **0.8370 vs
0.8467** — a 1.1 % gain, and the decision rule there resolved to "the bottleneck
is popularity concentration, not ordering"
(`M18_REPLICATION_RESULTS.md:26-33`). R4 is therefore worth GPU time **only if
R1 ≫ R2**, i.e. only if there is a correlation effect that aggregate replay
never showed. Do not run it otherwise.

Post-run: `pf6mps.hsaco_sha256` must differ from R1's.

---

### R5 — fabric discipline (remedy arms; conditional)

```bash
# R5a — escalating s_sleep backoff on the five cross-rank spin loops
DHK_ROOT=~/DHK-fab-backoff K0_MPS_CFG="$MPSCFG" \
  K0_MOK_TIMED_ITERS=128 \
  K0_MOK_ROUTE_FILE=/routes K0_MOK_ROUTE_ORDER=captured \
  bash run_campaign.sh r5a_captured_backoff_$TS 5

# R5b — destination-interleaved combine + dispatch (implies SCATTER_RR)
DHK_ROOT=~/DHK-fab-rmw K0_MPS_CFG="$MPSCFG" \
  K0_MOK_TIMED_ITERS=128 \
  K0_MOK_ROUTE_FILE=/routes K0_MOK_ROUTE_ORDER=captured \
  bash run_campaign.sh r5b_captured_rmw_$TS 5
```

R5a is only meaningful if R3's `[MPS SPIN] fail_max > 0` under captured routes —
backoff cannot help a spin loop that never spins. R5b subsumes R4 (it implies
SCATTER_RR), so if the night is short run **R5b instead of R4**, and read R4 as
the ordering-only control only if R5b moves.

---

### R7 — optional, high-value: M18 replication under captured routes

M18 at 16 replicas measured **0.2489×** under aggregate skew. If R1 confirms
that captured routes are as punishing as (or worse than) the aggregate, the same
lever should apply, and this is the direct kernel-level candidate for a serving
win rather than a serving loss.

```bash
DHK_ROOT=~/DHK-m18 K0_MPS_CFG="$MPSCFG" \
  K0_MOK_TIMED_ITERS=128 \
  K0_MOK_ROUTE_FILE=/routes K0_MOK_ROUTE_ORDER=captured \
  K0_MOK_REP_EXPERTS=6,5,4,2,0,3,1,7,8,9,20,10,11,19,17,16 \
  bash run_campaign.sh r7_captured_m18r16_$TS 5
```

**⚠ GAP:** `K0_MOK_REP_EXPERTS` normally derives its top-K from
`K0_MOK_ROUTE_HIST`, which is a hard error alongside `K0_MOK_ROUTE_FILE`
(`run_campaign.sh:32`). `ROUTE_REPLAY_PLAN.md:299` says the replica set must be
passed as an explicit csv under replay — the csv above is the aug14 measured
top-16 (also the hook's `M15_SKEW_REP_SET` default). Deriving the set *from the
capture* does not exist and is a named follow-on.

---

## 6. Decision table

Let **r_bal = 0.7556** (banked, re-proved by R0a), **r_hist = 0.8467** (banked),
**r_cap** = R1, **r_shuf** = R2, **r(T)** = R6. Serving implies **≈1.15–1.20**
for the MoE region.

| # | observation | mechanism verdict | what it implies | next optimization |
|---|---|---|---|---|
| 1 | **r(T) rises steeply as T falls** (e.g. r(1024) ≳ 1.1) while r_cap ≈ r_hist | **(e)+(c) CONFIRMED, (a)/(b) not needed** | the gap is *fill*: the mega pays a full 4,096-row fixed cost for ~1,539 real tokens (2.66×). Routing was never the story | ragged/variable-T dispatch in the mega; shrink C with T; make the planner and M8 certification token-count-proportional. **Kill the skew work stream.** |
| 2 | r(T) flat, r_cap ≈ r_shuf ≈ r_hist | replay fidelity and fill both irrelevant | the aggregate histogram is a sound proxy and the kernel region is *not* where the serving loss lives | re-attribute e2e: per-layer variance, KV-pool pressure, the shim's activation boundary, launch_prepare/ring-slot cost per sealed step (**c** measured at the host, not the kernel) |
| 3 | **r_cap ≫ r_shuf ≈ r_hist** | **(a) CONFIRMED** — within-chunk correlation itself is the cost | contiguous chunks concentrate destinations; our transport is convex in that concentration; aggregate replay has been lying since aug14 | R4/R5b decide the lever (rows 4-5); the replay corpus becomes the standard MoK route source |
| 4 | …and r_cap(R4/R5b) ≪ r_cap | peer-ordering serialization | a span is hostage to one source rank | ship `K0P6_M15_SCATTER_RR=1` (or RMW_INTERLEAVE) as default; re-sweep C under captured routes |
| 5 | …and r_cap(R4/R5b) ≈ r_cap | concentration, not order | ordering levers are exhausted (consistent with aug14's 1.1 %) | M18/M19 replication (R7) or per-chunk adaptive replication; finer slab certification |
| 6 | r_cap ≈ r_shuf **≫** r_hist | per-chunk **popularity** variance, not correlation | the aggregate histogram is a biased (optimistic) predictor; the convex tail is real — the M20 "uniform per-layer damage" finding at kernel speed | retire `K0_MOK_ROUTE_HIST` as a predictor; R7 is the direct fix |
| 7 | r_shuf > r_cap | correlation **helps** (dispatch locality) | i.i.d. replay is a *pessimistic* bound | treat r_hist as a lower bound; the gap is (c)/(e) — go to rows 1-2 |
| 8 | R3: `[MPS SPIN] fail_max > 0` under captured, `= 0` balanced | **(a)** localized to the M2 dispatch wait | peers are late, not slow | R5a (backoff) is on-mechanism; also try deeper flush_rows |
| 9 | R3: M7 inflates ≫ M6 between R3a and R3b/c | epilogue/remote-RMW surcharge is the inflating phase | combine transport, not GEMM | attack the epilogue carriage (M7/combine are one coupled block at r = −0.904 — **do not attack combine alone, the time will move, not disappear**) |
| 10 | R3: M6 **and** M7 inflate together, plan/combine flat | **(b) CONFIRMED** — hot-rank GEMM is the critical path | fixed costs don't shrink while one rank's compute grows | replication (R7) is the only lever that moves both; ordering/backoff cannot |
| 11 | r_cap lands near 1.15–1.20 | **the bench is finally predictive** | 139 s/campaign replaces 25 min/serving pair | move the entire optimization loop onto captured replay before touching serving again |
| 12 | any campaign blocked on gates | correctness, not performance | check `mok_eager.status`; the restore-call-0 epoch should make gates route-consistent | if gates fail *only* under replay, capture `route_replay` metadata + the failing arm and treat as a harness bug, not a kernel result |

The single most valuable comparisons, in order: **R6's r(T) slope**, then
**R1 vs r_hist**, then **R1 vs R2**.

---

## 7. Budget and cut-short order

| step | GPU time | cumulative | delivers |
|---|---|---|---|
| §2 capture pass (1 server lifetime) | ~20 min | 0:20 | the corpus; nothing else can proceed without it |
| §3 verification | 0 (CPU) | 0:20 | early kill if p95 ≈ aggregate |
| §4 pins | 0 (CPU) | 0:20 | R4/R5 arms |
| R0a balanced control, 5 runs | ~15 min + first build ~10 min | 0:45 | the ratchet; pin/patch validity |
| R6 smoke + T=2048 + T=1024 (1+3+3 runs) | ~20 min | 1:05 | **mechanism (e)/(c)** — needs no capture |
| R1 captured, 5 runs | ~13 min | 1:18 | **the headline** |
| R2 shuffled, 5 runs | ~13 min | 1:31 | (a) vs popularity split |
| R3a/b/c timestamps, 3 runs each | ~25 min | 1:56 | phase attribution + spin probe |
| R0b hist re-run (same-session 0.8467 anchor), 5 runs | ~13 min | 2:09 | optional; the banked value substitutes |
| R5b rmw captured, 5 runs | ~13 min | 2:22 | remedy (conditional on row 3) |
| R4 rr captured, 5 runs | ~13 min | 2:35 | ordering-only control (conditional) |
| R5a backoff captured, 5 runs | ~13 min | 2:48 | remedy (conditional on row 8) |
| R7 M18 r16 captured, 5 runs | ~13 min | 3:01 | the candidate serving win |

**≈3 h of GPU for the full grid; ≈1 h 20 m for the decisive core.**
(139 s/run × 5 + rotation overhead ≈ 13 min/campaign; the first campaign of the
night pays a ~10 min container cmake build, cached afterwards in
`$K0_MOK_CACHE_ROOT`. Note the first campaign against **each new `DHK_ROOT`
pin** re-JITs the mps module — budget ~3 min extra per pin.)

**If the night gets cut short, run in this order and stop wherever it stops:**

1. **Capture pass** — it is the only step that needs a live server; the GPUs are
   released to the harness afterwards and the corpus keeps forever.
2. **R0a** — without the ratchet nothing else is interpretable.
3. **R6** — highest information per minute, tests the mechanism with the largest
   unexamined prior (2.66× padding), and is the only arm that survives a failed
   capture.
4. **R1** — the direct kernel-level test of the −8 %.
5. **R2** — splits (a) from popularity.
6. **R3b vs R3a** — names the inflating phase and reads `[MPS SPIN]`.
7. Remedies (R5b → R4 → R5a) and **R7**, gated on rows 3/8 of the decision table.

---

## 8. Everything this plan needs that does not exist yet

| # | gap | blocking? | cost to close |
|---|---|---|---|
| G1 | `skew_pass()` overwrites `M15_SKEW_HOOK` with the v1 hook (`v2:568-570`, `v3:612-614`) | **yes** | zero-edit workaround in §2.1 (overwrite `m15_router_skew.py`); a proper fix is 1 line |
| G2 | `M15_ROUTE_CAPTURE_{MAX,TOKENS,FLUSH}` not forwarded into the container | no (defaults match the plan) | 3 `-e` lines if non-default values are wanted |
| G3 | **99 % of b4096 chunks run as graph replays; the router's Python never executes.** The corpus will be ~64 layers of 1–2 eager chunks, not 64 chunks | **maybe** | `--enforce-eager` on the capture lifetime = 1 line in `launch_server()`; raise `ROUTE_CAPTURE_MAX` to 1024 and filter by layer at replay |
| G4 | Fabric flags exist **only** in `k0pf6gm_device_tile_m15.hip`; the harness's `mps_mega` arm compiles `…_mps.hip` | **yes for R4/R5** | the `mkpin` recipe in §4.2 (CPU, minutes) |
| G5 | The MoK harness cannot consume a prebuilt hsaco — it cmake+JITs from `DHK_ROOT` | **yes** | none; the pin pattern is the supported mechanism. The verified `~/fabric_build` hsacos are serving-path only |
| G6 | `ROUTE_REPLAY_PLAN.md:245` tells you to grep the **m15** tile to confirm m17's RR is on; on that branch the flag is set in the **mps** tile | no | §1.2's grep |
| G7 | `[MPS TS]` is rank-0-only, never all-reduced, never in the rank JSON | **yes for mechanism (b)** | ~15 host lines: all-reduce the 8 stamps, write into every rank JSON |
| G8 | `[MPS TS]` is a final-soak-epoch max, so under replay it describes captured call 0 only | no | worked around via `K0_MOK_ROUTE_MAX_CALLS=1` (R3b/c); a per-iteration stamp buffer is the real fix |
| G9 | The task brief's `[K0P6 TS]` tag and "exp_33 aggregate-replay ledger" are both wrong: the tag is `[MPS TS]`, and exp_33 is **balanced, C=16, old mps ratchet, 0.8424×** | no | R3a supplies the correct in-session baseline |
| G10 | Banked skew config is **C=28** (`M18_REPLICATION_RESULTS.md:24`), not the brief's C=24; exp_33 used C=16 | no | R0a arbitrates |
| G11 | No banked evidence that `K0_T != 4096` passes the MoK gates | **yes for R6** | the 1-run smoke; if it fails, a `T`-parametric harness is a named work item |
| G12 | Deriving an M18 replica set from a captured corpus does not exist (`REP_EXPERTS` top-K needs `ROUTE_HIST`, which is mutually exclusive with `ROUTE_FILE`) | no | explicit csv in R7; the derivation is a follow-on |
| G13 | The route-replay MoK patch may not be deployed on the node at all (`ROUTE_REPLAY_PLAN.md:3` — "never executed on GPU") | **yes** | §4.3 = `ROUTE_REPLAY_PLAN.md` §2b + the zero-change ratchet |
| G14 | Mechanism **(c)** (per-sealed-step host overhead: `launch_prepare`, ring slots, descriptor writes at 99 % duty) is a **host** cost and no kernel campaign can see it | no | it needs a serving-side profile pass, not this grid — record it as the residual if rows 1-2 both come back flat |

---

## 9. What to write down for each campaign

From `summary.json`: `arm_p50_us` for both arms, `candidate_ratios["mps_mega"]{p50,p95}`,
and the per-run spread across all rotations (a ratio without its rotation spread
is not a result). From any rank JSON: `route_replay.route_multiset_sha256`,
`route_replay.route_calls`, `config.mok_route_calls`, `pf6mps.source_sha256`,
`pf6mps.hsaco_sha256`, `mok_eager.status`, the MoK gate pass counts, and the
600-epoch soak `pperr`. From the run logs: every `[MPS TS*]` and `[MPS SPIN]`
line. Label every number **kernel-region, one MoE layer, mok_eager, T as stated,
captured-route corpus = N layers of M real chunks** — never as a serving claim.

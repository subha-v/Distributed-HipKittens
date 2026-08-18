# Campaign v5 arms — exact env, exact serve line, citation per flag

Every arm below is produced by `run_m15_campaign_eplb_v5.sh`
(`arm_env()` for env, `serve_flags()` for the serve line). This file is the
human-readable mirror of those two case statements, plus the citation for
every flag that is not a vLLM default, plus which item of the
`nightshift/CLAUDE.md` fairness checklist the arm exists to satisfy.

Reading order for a reviewer: §0 (what every arm shares) → the arm you doubt →
§6 (known deviations, stated up front rather than discovered later).

---

## 0. What every arm shares

Container plumbing, identical for all arms:

```
docker run -d --name <container> \
  --network host --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  -v /data/hf_home:/hf_home:ro \
  -v $QSL_PKL:/qsl/dataset.pkl:ro \
  -v <output_dir>:/results -v <output_dir>:/prof \
  --entrypoint bash vllm/vllm-openai-rocm:v0.25.1 -lc "<command>"
```

Serve flags shared by every arm, and why each is not a tuning advantage:

| flag | value | why every arm has it |
|---|---|---|
| `--served-model-name` | `r1` | the client addresses `model: "r1"`; pure plumbing |
| `--host/--port` | `0.0.0.0:8000` | plumbing |
| `--seed` | `0` | identical on both sides; removes one nondeterminism source |
| `--max-model-len` | `32768` (`MAX_MODEL_LEN`); `4608` on the determinism launch | **capacity guard, not a tuning knob.** The checkpoint's own default is 163,840, which does not fit a KV cache at any gpu-memory-utilization on this node. Identical for every arm, so it cannot tilt a ratio. See §6.7 for why the determinism launch *lowers* it rather than widening the batching window. |
| `--no-enable-prefix-caching` | **off on every arm** (v5.1; `PREFIX_CACHING=1` re-enables it symmetrically) | Prefix caching is a property of the WORKLOAD, not of an arm's tuning. §6.6. |

Workload identity (fairness item 4): all arms are driven by the same
`bench_exact_token_ids_v3.py`, the same MLPerf QSL pickle, the same cell seeds,
the same warmup, and the same `ARM_COOLDOWN=240` before every launch. The
prompt stream is SHA-verified across arms by
`analyze_campaign_v5.py` §1.1.

**v5.1 workload changes (adversarial review).** Two of these change numbers, so
a v5.0 manifest and a v5.1 manifest are not interchangeable:

* **Prefix caching is off for every arm.** v5.0 served our arms with
  `--enable-prefix-caching` and the tuned native arms with
  `--no-enable-prefix-caching`. On its own that is already an asymmetry; with
  the QSL generator's duplicate rate (below) it was worth roughly 11% of the
  headline cell's prefill work, handed to the candidate before the kernel did
  anything. §6.6.
* **The QSL generator no longer produces duplicate prompts.** It drew each
  prompt's starting document as `hash(index) % 4388`; in a 1,024-prompt cell
  that is a birthday problem with ~111 expected exact repeats. It now strides
  the pool by a coprime step, which makes index → starting document a
  permutation. `workload.prompt_generator` moves to
  `mlperf-qsl-concat-v2-stride7` and `workload.duplicate_prompts_in_cell`
  records the residue (0 for any cell up to the pool size).
* **Position balance is a rotation, not a reversal**, and `PAIRS` must be a
  multiple of the arm count. §6.8.

Evidence deposited per arm before any measurement (fairness item 1):
`image_digest.txt`, `docker_env.json`, `docker_cmd.json`,
`server_config_dump.txt`, `patch_markers.txt`, `serve_cmd.txt`.

---

## 1. `m15` — the candidate

**Fairness role:** the thing being claimed for. Not a baseline.

**Env**

```
-e VLLM_PF4H_INTEGRATION_MODE=m15
-e VLLM_PF4H_B4096_GRAPH_TARGET=pf4h
-e VLLM_ALL2ALL_BACKEND=mori_high_throughput
# plus the shared PF4H block:
-e VLLM_PF4H_ACTIVATION_FILE=/tmp/vllm-pf4h-enable
-e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1
-e MORI_SHMEM_MODE=ISOLATION -e MORI_GPU_ARCHS=gfx950
-e HIP_FORCE_DEV_KERNARG=1 -e HSA_ENABLE_IPC_MODE_LEGACY=1
-e HSA_NO_SCRATCH_RECLAIM=1 -e PYTORCH_NVML_BASED_CUDA_CHECK=1
```

**Patch chain:** `apply.py --integration-mode m15` → `coverage_patch.py` →
`m23_patch.py` [→ `rr_patch.py`].

**Serve line**

```
vllm serve $MODEL --served-model-name r1 --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --data-parallel-size 8 --api-server-count 8 \
  --enable-expert-parallel --all2all-backend "$VLLM_ALL2ALL_BACKEND" \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.70 \
  --max-model-len 32768 --max-num-batched-tokens 4096 --max-num-seqs 128 \
  --scheduling-policy fcfs --enable-prefix-caching --enable-chunked-prefill \
  --optimization-level 2 --performance-mode balanced --seed 0
```

**Satisfies:** fairness item 3 — the arm must produce
`M15_*_RECEIPT` × ≥8 and a `RAGGED_SEAL_RECEIPT`, proving the megakernel
executed the traffic rather than sitting inert as it did for every pre-M23 A/B.

---

## 2. `stock` — the patched-stock control

**Fairness role:** the denominator of the KERNEL claim only
(`m15/stock`). It is production's MoE ops inside our integration, so it
inherits the B4096 stock graph, the uniform-decode rescue and our runtime
footprint. **It is not production** and must never be quoted as such — that
is the exact past sin recorded in checklist item 1.

**Env:** as `m15` but `VLLM_PF4H_B4096_GRAPH_TARGET=stock`.
**Patch chain:** identical to `m15`. **Serve line:** identical to `m15`.

**Satisfies:** the middle term of the three-way decomposition, so a reader can
see which part of any win is the kernel and which part is engineering both
sides could adopt (BENCHMARK_PROTOCOL.md §1, symmetry rule).

---

## 3. `native_default` — THE production headline baseline

**Fairness role:** checklist item 1, baseline authenticity. The untouched
vendor image at shipped defaults: what a user gets by default.

**Env:** *none.* `arm_env()` prints nothing and `launch_server` withholds the
entire shared PF4H/MoRI/HSA block and every PF4H bind mount from this arm.
Its whole environment is what the vendor baked into
`vllm/vllm-openai-rocm:v0.25.1`. `capture_config_receipt` FAILS the run if
`VLLM_PF4H_*` appears in `docker inspect` or if any of
`PF4H_INTEGRATION_PATCH_V3_M15`, `PF4H_COVERAGE_PATCH_V1`,
`PF4H_M23_RAGGED_SEAL_V1`, `PF4H_RR_*` appears in its log.

**Patch chain:** none. `apply_command()` returns `true`.

**Serve line**

```
vllm serve $MODEL --served-model-name r1 --host 0.0.0.0 --port 8000 \
  --trust-remote-code --tensor-parallel-size 8 \
  --max-model-len 32768 --seed 0
```

| flag | why it is here | citation |
|---|---|---|
| `--trust-remote-code` | DeepSeek-R1 does not load without it | vLLM Recipes, DeepSeek-R1 page |
| `--tensor-parallel-size 8` | the model does not fit on fewer GPUs; TP8 is the plain default parallelism a user reaches for | vLLM Recipes; AMD ai-ecosystem vLLM-V1 guide |
| `--max-model-len 32768` | shared capacity guard, §0 | — |

**Deliberately ABSENT** (each one is a knob we chose for ourselves and giving
it to the baseline would make the baseline ours, not the vendor's):
`--gpu-memory-utilization`, `--max-num-batched-tokens`, `--max-num-seqs`,
`--optimization-level`, `--performance-mode`, `--enable-prefix-caching`,
`--enable-chunked-prefill`, `--kv-cache-dtype`, `--scheduling-policy`,
`--data-parallel-size`, `--api-server-count`, `--all2all-backend`.

Consequence worth stating: without `--kv-cache-dtype fp8` this arm runs a
larger KV cache than our arms do. That is the shipped default and it costs it
memory, not accuracy — and it means our arms carry an fp8-KV accuracy debt
this arm does not (checklist item 6).

---

## 4. `native_tuned_tp` — AMD's recommended config, ≤128 concurrency

**Fairness role:** checklist item 2, baseline not sandbagged. AMD's own guide
recommends **TP8 + expert-parallel** below 128 concurrency; our DP8/EP8 layout
is the vendor's recommendation for exactly one of our cells (c512p). Beating
only the lazy default would be a new form of the old sin.

**Env**

| variable | citation |
|---|---|
| `VLLM_ROCM_USE_AITER=1` | AMD ai-ecosystem *vLLM V1 performance optimization* — master AITER switch |
| `VLLM_ROCM_USE_AITER_MOE=1` | same guide; vLLM Recipes sets it explicitly |
| `SAFETENSORS_FAST_GPU=1` | same guide, recommended env block |
| `TORCH_BLAS_PREFER_HIPBLASLT=1` | same guide |
| `HIP_FORCE_DEV_KERNARG=1` | same guide |
| `VLLM_RPC_TIMEOUT=1800000` | same guide |
| `NCCL_MIN_NCHANNELS=112` | **NOT SET** — AMD documents it for MI300X/MI325X, not gfx950. Setting it would be our guess, not their recommendation. |

**Serve line**

```
vllm serve $MODEL --served-model-name r1 --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --attention-backend ROCM_AITER_MLA \
  --max-model-len 32768 --max-num-batched-tokens 16384 --max-num-seqs 2048 \
  --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --async-scheduling --seed 0
```

| flag | value | citation |
|---|---|---|
| `--tensor-parallel-size 8 --enable-expert-parallel` | TP8+EP | AMD ai-ecosystem vLLM-V1 optimization guide: recommended parallelism at ≤128 concurrency |
| `--attention-backend ROCM_AITER_MLA` | AITER MLA | vLLM blog 2026-02-27 *Beyond Porting* — "recommended for all workloads", 1.35–1.52× TPS vs TRITON_MLA at conc 64/128 |
| `--max-num-batched-tokens 16384` | 16384 | vLLM blog 2026-02-27, AMD's own DeepSeek-R1-0528 benchmark command |
| `--max-num-seqs 2048` | 2048 | AMD ai-ecosystem guide (V1 default; raise for throughput) |
| `--gpu-memory-utilization 0.9` | 0.9 | vLLM blog 2026-02-27 benchmark command (guide offers 0.95 for single-instance max throughput) |
| `--no-enable-prefix-caching` | off | vLLM blog 2026-02-27 benchmark command. **This removes a cross-arm cache asymmetry only because v5.1 disables prefix caching on OUR arms too** — in v5.0 it created one, since only this side of the comparison paid for the ~11% duplicate prompts the QSL generator then emitted (§6.6). |
| `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'` | full+piecewise | vLLM blog 2026-02-27 benchmark command |
| `--async-scheduling` | on | vLLM blog 2026-02-27 benchmark command |
| `--block-size` | **not set** | vLLM blog 2026-02-27 explicitly de-recommends `--block-size 1` for vLLM ≥0.14; AMD's own R1-0528 run used the default 16 |

**Version caveat, and how it is handled:** AMD's cited command targets vLLM
0.14.0rc2 (Jan 2026); we run 0.25.1, where defaults have moved. Every flag is
therefore verified at startup — a rejected flag trips the loud
`STARTUP FLAG REJECTED` path in `wait_ready` with the log tail attached — and
the *resolved* config is captured to `server_config_dump.txt` rather than
trusted from the flags we believe we passed.

---

## 5. `native_tuned_dp` — AMD's recommended config, ≥512 concurrency

**Fairness role:** checklist item 2 again, for the large-batch cells
(c512, c512p), where AMD's recommendation flips to DP+EP and where they claim
**+16–47% throughput** over the alternative — a range large enough to swallow
our entire banked delta. It is also the parallelism-matched control for our
DP8/EP8 mega.

**Env:** the §4 block **plus** `VLLM_ALL2ALL_BACKEND=allgather_reducescatter`
(AMD ai-ecosystem guide; also the vLLM default). MoRI is deliberately not used
here: AMD documents `mori_*` as the **multi-node** path, so choosing it for a
single-node baseline would be our decision dressed as theirs.

**Serve line** — §4 with the parallelism block replaced by

```
  --tensor-parallel-size 1 --data-parallel-size 8 \
  --api-server-count 8 \
  --enable-expert-parallel --disable-nccl-for-dp-synchronization
```

| flag | citation |
|---|---|
| `--data-parallel-size 8 --enable-expert-parallel --disable-nccl-for-dp-synchronization` | AMD ai-ecosystem vLLM-V1 optimization guide, ≥512-concurrency recommendation (claimed +16–47%) |
| `VLLM_ALL2ALL_BACKEND=allgather_reducescatter` | same guide, single-node DP+EP path |
| `--api-server-count 8` | **documented deviation** — see §6.1 |

---

## 6. Known deviations, stated up front

**6.1 `--api-server-count 8` on `native_tuned_dp`.** AMD's guidance says
nothing about the HTTP front end. At concurrency 512 a single API server can
become the bottleneck, which would let us "win" against a queue rather than
against a model. Giving the baseline the same 8 front-end processes our arm
uses is anti-sandbagging, not tuning; `NATIVE_DP_API_SERVERS=1` reproduces the
literal citation if a reviewer wants the difference measured.

**6.2 `--max-model-len 32768` on every arm,** including `native_default`. §0.

**6.3 `native_mirror` exists and is not production.** It is v4's `native`
arm — untouched image, OUR serve line — renamed so nobody can quote it as
production. `analyze_campaign_v5.py` keeps it out of `PRODUCTION_ARMS` and
labels it "control (NOT production)" in every table. Its purpose is to split
"the patch chain" from "our serve config" inside the integration effect.

*v5.1 correction.* v5.0's `arm_is_native()` withheld the whole common
environment from `native_mirror` — including `MORI_GPU_ARCHS=gfx950`,
`MORI_SHMEM_MODE=ISOLATION` and `VLLM_ROCM_USE_AITER*` — while still serving
`--all2all-backend mori_high_throughput`. That arm would either hang at 8-rank
MoRI init or run a MoRI-without-AITER configuration nobody deploys, and in
either case it controlled for far more than "the patch chain". v5.1 splits the
predicate: `arm_is_native()` still means *untouched image* (no `apply.py`, no
coverage/M23/RR patch, no PF4H mounts), while the new
`arm_is_vendor_config()` — true for `native_default` and `native_tuned_*`,
false for `native_mirror` — is what withholds our runtime environment.
`native_mirror` therefore receives AITER/MoRI/HSA/HIP exactly as our arms do,
and receives **no** `VLLM_PF4H_*` variable, which the authenticity gate in
`capture_config_receipt` still forbids for every native arm.

**6.4 The ceiling this campaign cannot reach.** AMD's fastest published
DeepSeek-R1 path on MI350/MI355 is the **ATOM** engine
(`rocm/atom-dev`) with MTP speculative decoding — a different image, so it
cannot be an arm here. "Beats stock vLLM" is strictly weaker than "beats
AMD's fastest published stack", and REPORT.md must say so (checklist item 8).

**6.5 No external absolute anchor exists.** AMD publishes no absolute vLLM
tok/s for DeepSeek-R1 on MI350X/MI355X — only relative multipliers — and R1 is
absent from AMD's MLPerf v6.0 submission. The audit therefore cannot
sanity-check our native numbers against a published figure and must lean
entirely on internal evidence: image digest, container env, patch-marker
absence, resolved config dump, coverage receipts.

**6.6 Prefix caching is a workload knob and is off everywhere (v5.1).**
v5.0 served `m15`/`stock` with `--enable-prefix-caching` and
`native_tuned_tp`/`native_tuned_dp` with `--no-enable-prefix-caching`, on a
prompt list that then contained ~10.8% exact duplicates inside every
1,024-prompt `c32p` cell. Those ~111 prefills (4,096 tokens each) cost roughly
nothing on the cached side and full price on the uncached side, i.e. the arm
designated as THE headline baseline did ~11% more prefill work than the
candidate on the one cell carrying the headline. Two independent fixes:

1. every arm now serves with prefix caching **off** (`PREFIX_CACHING=1` turns it
   on for all arms at once, never for one); and
2. the QSL generator strides the pool so a cell of up to 4,388 prompts contains
   no exact duplicates at all, with the residue counted in the manifest.

`native_default` therefore carries `--no-enable-prefix-caching` as its **second**
non-default flag (after `--max-model-len`). That is a deliberate trade: a
declared, symmetric workload flag on the shipped-defaults arm is a far smaller
sin than an undeclared asymmetry in the candidate's favour. It also means
tonight's absolute tok/s are not comparable to the banked camp3 receipts, which
ran with caching on; the ratios are, and the ratios are the claim.

**6.7 fp8 KV cache asymmetry, and what the accuracy gate can therefore
prove.** Our arms set `--kv-cache-dtype fp8`; the native arms use the shipped
default. This favours the native arms on accuracy and disfavours them on memory.
It is deliberate — each side gets its own shipped configuration — but it means
`m15` and `native_tuned_tp` are **not bit-comparable by construction**: different
KV quantisation, different parallelism (DP8 vs TP8), different attention backend
(default vs `ROCM_AITER_MLA`), different all2all. v5.0's gate demanded a
byte-identical output SHA across exactly those arms, so it could only ever FAIL,
and per its own doctrine ("unresolved parity = no claim") a night run under it
would have shipped nothing.

v5.1 replaces equality-across-everything with a **numerics class** gate
(`numerics_class()` in the wrapper):

| class | arms |
|---|---|
| `dp8-fp8kv-mori` | `stock`, `pf4h`, `m15`, `m18`, `m19`, `m20`, `native_mirror` |
| `dp8-fp8kv-rccl` | `rccl` |
| `tp8-defaultkv-defaultattn` | `native_default` |
| `tp8-defaultkv-aitermla` | `native_tuned_tp` |
| `dp8-defaultkv-aitermla` | `native_tuned_dp` |

* **Within a class**, `c1det` output streams must be byte-identical. `m15` vs
  `stock` is the comparison the kernel claim actually rests on, and it is a hard
  FAIL. This is why `stock` must be in the accuracy pass: without it the gate
  returns `PASS_WEAK` and no bit-exactness is established for the candidate at
  all.
* **Across classes**, the gate reports a measured per-prompt agreement fraction
  and the first divergent prompt index. It never asserts equality there.
* **Both** cells run: `c1det` (conc 1) is the deterministic one, and `c8det`
  (conc 8, 64 prompts) exists because at concurrency 1 under TP1/DP8 a single
  request lands on one data-parallel rank, the (512,4096] ragged seal cannot go
  unanimous, and the megakernel is *inert* — v5.0's gate would have compared
  stock against stock while wearing the candidate's label. `c8det` puts real
  tokens on all eight ranks; its cross-run nondeterminism means it is reported
  as an agreement fraction, not asserted.
* **Candidate liveness is asserted.** The gate reads the accuracy server's own
  `RAGGED_SEAL_RECEIPT` and FAILs if the candidate shows no sealed steps: a PASS
  produced by a fallback path is the project's oldest sin.
* **A missing arm is a FAIL,** not a smaller comparison. v5.0 skipped an absent
  `c1det.json` and could print `ACCURACY_GATE=PASS arms=2` over `{m15, stock}`
  while the production baseline never started.

The determinism launch also no longer moves the candidate off its operating
point. Disabling chunked prefill requires `max_num_batched_tokens >=
max_model_len`; v5.0 satisfied that by raising the batching window to 32,768,
which took the megakernel off the B4096 graph every performance cell measures
(and left `native_default`, which names no batching flag, unable to start at
all). v5.1 instead lowers `max_model_len` and the batching window together to
4,608 — the smallest value ISL 4096 + OSL 8 needs — so a 4,096-token prefill step
still fills the ragged-seal bucket, and emits both flags on every arm.

---

## 7. Arm → fairness-checklist map

| arm | primary checklist item satisfied |
|---|---|
| `native_default` | 1 — baseline authenticity (untouched image, shipped defaults, verified by evidence) |
| `native_tuned_tp` | 2 — baseline not sandbagged (vendor's documented best for ≤128 conc) |
| `native_tuned_dp` | 2 — baseline not sandbagged (vendor's documented best for ≥512 conc) |
| `native_mirror` | 8 — claim wording: isolates the integration effect so it is not silently folded into the kernel claim |
| `stock` | the kernel term of the decomposition; BENCHMARK_PROTOCOL.md §1 symmetry rule |
| `m15` | 3 — candidate actually ran (receipt gates + ragged-seal coverage) |
| `c1det` + `c8det` cells (all arms) | 6 — accuracy parity, within the numerics class it is achievable in; §6.7 |
| `o50p/o75p/o90p` cells | 7 — replay/proxy honesty: closed-loop saturation vs realistic arrival-driven utilization, reported per regime. Ranked by TTFT p99, never by throughput (§8.7). |

---

## 8. v5.1 protocol mechanics added by the adversarial review

**8.1 Position balance is a rotation.** v5.0 reversed the arm LIST on even
pairs. With three arms that never moves the middle arm — `stock` held position 2
in all five pairs — and with an odd `PAIRS` the first-listed arm gets an extra
position-1 slot. The measured position effect is ±18% per arm (position 2 ran
~25% slower than position 1), i.e. the size of the entire banked claim, and a
median over an odd number of biased pairs lands *on* a biased pair rather than
between them. v5.1 rotates by `(pair-1) mod n_arms` and reverses every `n_arms`
pairs, and **refuses to start unless `PAIRS % n_arms == 0`**
(`ALLOW_UNBALANCED_PAIRS=1` overrides it and stamps
`POSITION_BALANCE=UNBALANCED` into the campaign receipt, which turns every
policy row into "no claim"). Default `PAIRS` is now 6.

**8.2 An arm failure voids its pair, not the campaign.** A rejected startup flag
on a native arm used to kill the driver through `set -e` at, say, pair 3 of 5 —
leaving exactly the unbalanced partial dataset the protocol forbids. v5.1 writes
`PAIR_VOID.txt` and continues; the analyzer drops that pair whole.

**8.3 A receipt shortfall discards the arm's cells, loudly.** With `pipefail`
in force, `check_receipts | tee` was fatal, not discarding. v5.1 records
`RECEIPT_GATE=PASS/FAIL` (now including a ≥90% ragged-seal coverage bar) and the
analyzer excludes `FAIL` records from every ratio and lists them.

**8.4 One `RUN_TAG` per invocation.** `ROOT` and the container names both derive
from `RUN_TAG`; three steps sharing one tag meant the third either hit the
name-collision refusal or wrote into the first's `pair_01`, promoting a
calibration arm into the headline matrix. v5.1 refuses a `ROOT` whose recorded
invocation spec differs, refuses a pair directory that already holds manifests,
and removes (rather than trips over) a stopped leftover container.

**8.5 Synthetic prompts require an explicit acknowledgement.**
`PROMPT_SOURCE` still defaults to `synthetic` for backward compatibility, but
the driver now exits unless it is `qsl` or `ALLOW_SYNTHETIC_PROMPTS=1` is set,
and the analyzer prints a "NOT A HEADLINE" banner if any manifest says
otherwise. None of v5.0's prescribed commands set `PROMPT_SOURCE=qsl`.

**8.6 One offered rate for the open-loop cells, from the SLOWEST arm.**
`OPEN_LOOP_BASE_RATE` is the minimum measured ceiling across the arms in the
invocation, and `OPEN_LOOP_BASE_RATE_SOURCE` (required, recorded) says where it
came from. Driving every arm from the candidate's own ceiling offers the
baseline a load it cannot absorb and then reports its queue as its latency;
driving each arm from its own ceiling makes the arms incomparable, which is why
the analyzer suppresses those ratios entirely.

**8.7 Open-loop cells are ranked by TTFT p99.** At a fixed arrival rate the
delivered token count is fixed, so `input_tokens / wall_seconds` is pinned to the
offered rate: v5.0's decomposition reported ~1.000 for every open cell no matter
how differently the arms served it, and its policy table picked a "best arm" out
of drain-time noise. TTFT p99 is where an arm that cannot absorb the load shows
it, so that is the ranking metric (ratios inverted for display, so >1 still
means the numerator is better).

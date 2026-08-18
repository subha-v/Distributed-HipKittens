# M23 — "Ragged Seal": making the megakernel graph accept the batches production already accepts

**Status:** design, ready to implement.
**Author:** aug18-prefill session.
**Scope:** serving integration only (vLLM 0.25.1+rocm723 patched tree + `pf4h_integration` shim). **No megakernel change.**
**Target file set:** `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/dp_utils.py`, `vllm/v1/cudagraph_dispatcher.py`, `vllm/forward_context.py`, `pf4h_integration/{vllm_full,m15_vllm,contracts}.py`, `pf4h_integration/tests/`.

Path conventions in this document:

* **GMR** = `<vllm>/v1/worker/gpu_model_runner.py`
* **DISP** = `<vllm>/v1/cudagraph_dispatcher.py`
* **DPU** = `<vllm>/v1/worker/dp_utils.py`
* **FCTX** = `<vllm>/forward_context.py`
* **MK** = `<vllm>/model_executor/layers/fused_moe/modular_kernel.py`
* **MORI** = `<vllm>/model_executor/layers/fused_moe/prepare_finalize/mori.py`
* **SHIM** = `pf4h_integration/`
* **KERNEL** = `/Users/subha/repos/Distributed-HipKittens/distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip`

All `<vllm>` paths are rooted at the deployed mirror
`/private/tmp/.../scratchpad/mirror/vllm_patched/` and are byte-identical to the node's
`vllm/` package (the shim copy under `model_executor/layers/fused_moe/experts/pf4h_integration/`
differs from the source-of-truth `pf4h_integration/` only by the presence of `apply.py`,
`tests/`, `m15_pin/`, `README.md`, `m18_replication.py`).

---

## 0. Verdict (read this first)

| Question | Answer | Evidence |
|---|---|---|
| Is the fix feasible? | **Yes.** | Sections 3–4 |
| Does the megakernel need to change? | **No. Zero lines of HIP.** | §3.4, §3.5 |
| Does the *shim* need math changes? | **No.** `_attest_dp_batch` is *already* written against post-padding counts and is ragged-correct as-is; only comments/docstrings and one new refusal are needed. | §4.4 |
| What actually changes? | The **seal predicate** in GMR, a **unanimity bit** added to the DP all-reduce in DPU, and a **mode override** so all 8 ranks make the same graph decision. | §4.2, §4.3 |
| Coverage recovery | **2% → 100% of in-bucket steps** = **0.77% → 38.3% of all steps** = **~50× more sealed steps** (4.6 → 230 per rank per c32p run). | §1 |
| Top risk | Cross-rank **arrival skew** on ragged batches vs. the megakernel's `M15_SPIN_LIMIT = 20_000_000` fail-closed spin bound. | §7.1 |
| Effort | **~8–10 engineer-hours of code + tests**, **~4 hours of node time** for the validation ladder. | §8.5 |

There is a second, independent finding that changes how every past A/B should be read:

> **On an unsealed in-bucket step, a PF4H-target server does not fall back to the stock B4096
> *graph* — it falls back to *no graph at all*.** GMR:3973-3981 forces
> `cudagraph_mode = CUDAGraphMode.NONE` because the PF4H-only server never registered the
> ordinary `regular_b4096` key (DISP:250-255 registers exactly one of the two).
> So the historical candidate arm ran ~225 of ~230 heavy steps **eagerly**, while the baseline
> arm ran all ~230 under PIECEWISE replay. Every serving A/B of the fused megakernel measured
> a **doubly** handicapped candidate: the megakernel was inert *and* the piecewise graph was
> disabled underneath it.

---

## 1. The finding, quantified

### 1.1 What "in bucket" means, and why it is a *global* property

`_determine_batch_execution_and_padding` runs a DP all-reduce (GMR:3916-3932 →
`coordinate_batch_across_dp`, DPU:170-254). When any rank will use a cudagraph,
`should_dp_pad` is true and **every rank is padded up to the maximum padded count across
ranks**:

```python
# DPU:77-89  _post_process_dp_padding
num_tokens_across_dp = tensor[1, :]
if should_dp_pad:
    max_num_tokens = int(num_tokens_across_dp.max().item())
    return torch.tensor([max_num_tokens] * len(num_tokens_across_dp), device="cpu", ...)
```

with `should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch` (DPU:152).

The dispatcher routes **any** non-uniform, non-LoRA batch with
`max_cudagraph_capture_size < num_tokens <= 4096` to the B4096 PIECEWISE key
(DISP:331-359; the generic capture list stops at 512 per the comment at DISP:215-216):

```python
# DISP:331-341
if (
    os.environ.get("VLLM_PF4H_INTEGRATION_MODE") in ("full", "m15")
    and self.keys_initialized
    and CUDAGraphMode.PIECEWISE in allowed_modes
    and max_size is not None
    and max_size < num_tokens <= 4096
    and not uniform_decode
    and not has_lora
    and num_active_loras == 0
):
    ...
    return CUDAGraphMode.PIECEWISE, regular_b4096   # DISP:359
```

Consequence: **if one rank is in bucket, all eight are.** "230 of 600 steps are in bucket"
is therefore the *same* 230 steps on every rank, which is exactly what the coverage
instrumentation measured.

### 1.2 What the seal actually requires today

```python
# GMR:3947-3962
pf4h_exact_b4096 = False
if (
    # PF4H_INTEGRATION_PATCH_V3_M15
    os.environ.get("VLLM_PF4H_INTEGRATION_MODE") in ("full", "m15")
    and self.parallel_config.data_parallel_size == 8
    and not should_ubatch
    and not uniform_decode
    and not has_lora
    and batch_descriptor.num_tokens == 4096
    and original_num_tokens_across_dp is not None
):
    original_counts = tuple(int(v) for v in original_num_tokens_across_dp.tolist())
    pf4h_exact_b4096 = original_counts == (4096,) * 8
```

`original_num_tokens_across_dp` is row 0 of the all-reduce — the **pre-padding** counts
(DPU:131-137 writes `tensor_cpu[0][dp_rank] = orig_num_tokens_per_ubatch`; DPU:161 returns
`tensor[0, :].cpu()`). It is the *only* consumer of that row anywhere in the tree; stock vLLM
never looks at it.

`(4096,)*8` requires eight simultaneous *full* 4096-token chunks. With chunked prefill +
prefix caching at c32p, that is a coincidence, not a regime. The measured `fail_min_tok=1`
says the modal failure is a single rank running a one-token decode batch — which DP padding
then inflates to 4096 anyway.

### 1.3 Coverage arithmetic

Per rank, per c32p run (concurrency 32, ISL 4096, OSL 8, chunked prefill, prefix caching):

| Quantity | Today | After M23 |
|---|---|---|
| Total steps | ~600 | ~600 |
| In-bucket steps (`batch_descriptor.num_tokens == 4096`) | ~230 (38.3%) | ~230 (38.3%) |
| Steps production's stock B4096 graph runs | ~230 (100% of in-bucket) | ~230 |
| Steps the megakernel is sealed for | **~4.6** (2% of in-bucket) | **~230** (100% of in-bucket) |
| Megakernel share of all steps | **0.77%** | **38.3%** |
| Unsealed in-bucket steps running **eager** (PF4H arm, GMR:3973-3981) | ~225 | **0** |

**Coverage multiplier: 230 / 4.6 ≈ 50×.**

MoE-work-weighted (58 routed layers, layers 3..60, `FROZEN_CONTRACT.first_moe_layer=3`,
`last_moe_layer=60`, SHIM `contracts.py:63-64,69-71`): every in-bucket step executes exactly
`4096 × 58` MoE row-layers per rank **in both arms**, because padding is arm-independent
(§3). So the megakernel's coverage of padded-4096 MoE work goes from

* `4.6 × 4096 × 58 ≈ 1.09 M` row-layers/rank/run → `230 × 4096 × 58 ≈ 54.6 M` row-layers/rank/run.

Out-of-bucket steps (~370, each ≤512 real tokens) contribute at most
`370 × 512 × 58 ≈ 11.0 M` row-layers — an upper bound, since most are decode batches of a
few tokens. So the in-bucket window M23 unlocks is **≥83% of all MoE row-layers** even under
the most pessimistic assumption about the remaining steps, and in practice far more.

Token-weighted: c32p issues 4096 prefill tokens and 8 decode tokens per request, so ≥99.8%
of tokens are prefill, and prefill is scheduled through chunks that (with
`max_num_batched_tokens=4096`, asserted at DISP:236-241) routinely exceed 512. **Probe P1
(§9) closes this the rest of the way** by summing `num_tokens` over in-bucket vs. all steps.

---

## 2. Archaeology — *why* was the exactness restriction chosen?

Three independent reasons appear in the source. None of them is a capability limit of the
kernel.

**(a) Graph-key collision safety.** SHIM `vllm_full.py:359-362`, inside `_eligible`:

```python
    # A non-uniform B4096 CUDA graph may later replay for a smaller
    # live batch padded to the same descriptor. The exact runner patch
    # prevents that graph-key collision before this predicate runs.
```

The author's worry was that a *stock*-captured B4096 graph and a *PF4H*-captured B4096 graph
would hash to the same `BatchDescriptor` key and a ragged batch would silently land in the
wrong one. The chosen mitigation was to add the `pf4h_exact_b4096: bool` field to
`BatchDescriptor` (FCTX:59-63) so the two keys are structurally distinct, and then to only
ever *stamp* that field on batches that were provably non-ragged. The distinctness is what
provides the safety; the exactness is belt-and-braces on top of it.

**(b) A stated policy, not a measured limit.** SHIM `contracts.py:272-273`:

```python
# The only PF4H serving bucket proven by the isolation+N2 graph gate.  Smaller
# prefill buckets, decode, and ragged DP batches stay on stock Mori+AITER.
```

and SHIM `vllm_full.py:3-6` (module docstring), and SHIM `m15_vllm.py:3-6`
("the same non-ragged exact-B4096 DP proof"). "Proven by the graph gate" — i.e. ragged was
never *validated*, so it was excluded. It was never shown to be wrong.

**(c) The shim could not reconstruct the distinction.** SHIM `vllm_full.py:268-270`:

```python
        # The exact graph patch adds this key only after DP coordination proves
        # every rank's *original*, pre-padding count is 4096. This distinction
        # cannot be reconstructed from DPMetadata, which intentionally carries
        # post-padding counts for graph replay.
```

This is the load-bearing observation, and it cuts the *other* way for M23: because
`DPMetadata` carries **post-padding** counts, the shim's own attestation

```python
# SHIM vllm_full.py:284-294
dp_metadata = getattr(context, "dp_metadata", None)
across = getattr(dp_metadata, "num_tokens_across_dp_cpu", None)
...
return values == (PREFILL_B4096_BUCKET.tokens_per_rank,) * FROZEN_CONTRACT.dp_size
```

is **already exactly the ragged-seal predicate** — "all eight ranks padded to 4096". The shim
needs no logic change at all. The exactness lives in one place only: GMR:3962.

**Not reasons:** the qpush input shape is `T` from the descriptor and is a launch constant
(§3.4); the capture contract is `_dummy_run(4096)` which fabricates a genuinely-unpadded
4096-token batch on every rank (GMR:5918-5926) and is unaffected; no scale or count plumbing
depends on the original counts anywhere.

---

## 3. Q1 — Padding semantics parity

### 3.1 What fills the padded rows

For a step with `n_orig` real tokens padded to 4096 (note `self.max_num_tokens ==
max_num_batched_tokens == 4096`, GMR:494 and DISP:236-241, so the persistent buffers are
exactly one bucket long):

| Buffer | Padded rows `[n_orig, 4096)` | Site |
|---|---|---|
| `positions` | **zeroed every step** | GMR:3570-3573 `self.positions[num_scheduled_tokens:num_input_tokens].zero_()` |
| `input_ids` | **stale** — token ids left by an earlier step (always valid vocab ids, so the embedding gather is memory-safe) | GMR:748 persistent buffer; writes bounded by `total_num_scheduled_tokens` at GMR:1982/1762/1820; no tail fill on the execute path |
| `inputs_embeds` | **stale** | GMR:3509-3525 writes only `[:num_scheduled_tokens]` |
| `slot_mapping` | in PIECEWISE mode **not allocated for the pad at all** (`pad_attn = cudagraph_mode == CUDAGraphMode.FULL`, GMR:4323 → GMR:4371-4374 sizes it `num_tokens_unpadded`); in FULL mode filled with `-1` (GMR:4107-4109) | — |
| attention output | **untouched garbage** — attention metadata is built unpadded (`num_tokens_padded=None` at GMR:4385), backends slice by `num_actual_tokens` | GMR:4382-4392 |

So the hidden state entering each MoE layer's padded rows is
`embedding(stale token id) + garbage-but-finite attention residue`, propagated through the
same 58 routed layers as the real rows.

### 3.2 What stock does with them: **nothing. It dispatches all 4096.**

Decisive chain, every link cited:

1. **Forward context gets the padded count and no padding mask.**
   GMR:4436-4447 `set_forward_context(..., num_tokens=num_tokens_padded, ...)` — the
   `is_padding=` kwarg (FCTX:227, FCTX:161) is **not passed** by the V1 runner. It defaults
   to `None`.
2. **The V1 runner is the deployed one.** `gpu_worker.py:384-398` selects
   `GPUModelRunnerV2` from `v1/worker/gpu/model_runner.py` only when
   `vllm_config.use_v2_model_runner`; the PF4H patch marker
   `PF4H_INTEGRATION_PATCH_V3_M15` lives in `v1/worker/gpu_model_runner.py` (V1), and only
   the V2 runner ever populates `is_padding` (`v1/worker/gpu/model_runner.py:1013`,
   `v1/worker/gpu/input_batch.py:143-176`).
3. **The one masking site is doubly disabled.** MK:1137-1151:
   ```python
   # Skip cudagraph/DP padding tokens uniformly across all a2a backends:
   # forcing padded rows' expert ids to -1 makes every prepare_finalize drop
   # them ... The V2 model runner marks them in forward_context.is_padding; it
   # is None for runners that do not populate it, leaving topk_ids unchanged.
   # Gated by VLLM_MOE_SKIP_PADDING (off by default) ...
   is_padding = None
   if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
       is_padding = get_forward_context().is_padding
   if is_padding is not None:
       n = topk_ids.shape[0]
       topk_ids = torch.where(is_padding[:n].unsqueeze(1), -1, topk_ids)
   ```
   `VLLM_MOE_SKIP_PADDING` defaults to `0` (`envs.py:191`, `envs.py:1499`).
4. **The MoE block uses the padded row count as its token count.** `deepseek_v2.py:403-419`:
   `num_tokens, hidden_dim = hidden_states.shape` (= 4096), gate and experts both consume all
   4096 rows; `return final_hidden_states.view(num_tokens, hidden_dim)` at :427.
5. **The router emits ordinary top-k for padded rows.** `fused_moe/runner/moe_runner.py:573-586`
   calls `select_experts` over all 4096 rows; the only `zeros`/`masked_fill` on the router
   path are algorithmic over the *expert* axis (`router/grouped_topk_router.py:138,145`).
   Padded rows therefore get **valid expert ids in `[0, 256)`** with ordinary weights.
6. **Mori dispatches them.** MORI:89-95 `self.mori_op.dispatch(a1, topk_weights, scale,
   topk_ids)` — `a1` is passed unsliced, `[4096, 7168]`. The handle is sized
   `max_tokens_per_rank = moe.max_num_tokens = max_num_batched_tokens`
   (`all2all_utils.py:263,270-275`), i.e. worst-case, not live-count.
7. **AITER computes them.** `experts/rocm_aiter_moe.py:526-546` passes
   `num_local_tokens = expert_tokens_meta.expert_num_tokens`, which is Mori's
   `dispatch_recv_token_num` (MORI:97-99) — **post-dispatch receive counts, which include
   the padded rows**. The GEMM bound is `M = a1q.size(0)` (MK:802-805).
8. **Combine writes all 4096 rows back.** MORI:118-124
   `num_token = output.shape[0]` (= 4096, since `output = torch.empty_like(hidden_states)` at
   MK:1413) then `output.copy_(result[:num_token])`.
9. **No DP chunking exists in this tree.** `moe_dp_chunk_size`, `MOE_DP_CHUNK_SIZE`,
   `chunk_by_rank` all have zero hits. The surviving `local_sizes` helper is SP-only
   (FCTX:106-122) and itself consumes the *padded* counts.

> **Verdict: STOCK DISPATCHES ALL 4096 PADDED ROWS.** There is nothing to reproduce and
> nothing to mask. The megakernel already does the identical thing.

### 3.3 Therefore M23 is a pure seal relaxation

The `if` branch in the task brief — "if production masks padded rows, determine the cheapest
equivalent" — **does not fire**. No shim-side `topk_weights` zeroing is needed, no `n_orig`
plumbing, no M4-zeroing trick. Keeping `T = 4096` and dispatching padded rows *is* parity.

### 3.4 The megakernel side, confirmed

`T` is a launch constant read from descriptor slot 44 and written once, out of capture:

```python
# SHIM m15_runtime.py:941
words[int(DescriptorSlot.T)] = self.bucket.tokens_per_rank      # 4096
```
(`DescriptorSlot.T = 44`, SHIM `m15_contracts.py:128`; `M15_PREFILL_B4096_BUCKET =
M15GraphBucket(tokens_per_rank=4096, maxtok=4096, ...)`, SHIM `m15_contracts.py:504-506`.)

M1 qpush iterates it unconditionally (KERNEL:977, 993-1018):

```cpp
const int T = (int)k0p6_dread(desc, K0P6_D_T);                        // KERNEL:977
...
for (int tau = gw; tau < T; tau += nw) {                              // KERNEL:993
  const int s = lane;
  const int eid = (s < TOPK) ? my_ids[(size_t)tau * TOPK + s] : -1;   // KERNEL:995
  ...
  const int dest = (eid >= 0) ? eid / E : -1;                         // KERNEL:1018
```

M3–M5 plan/scatter (KERNEL:1375) and M8 combine (KERNEL:1891, `nbatches = (T + NT - 1) / NT`
at KERNEL:1899) are the same. `out` receives all `T` rows, matching the 4096-row `output`
buffer that `finalize` type-checks (SHIM `m15_vllm.py:275-285`).

**Capacity is already worst-case sized for T=4096 on all eight ranks**, so raggedness can only
*reduce* arrivals:

* `t_loc_max = 40_960 = world × T` (SHIM `contracts.py:277`, invariant at `contracts.py:134-136`);
* `pad_max = 263_136 ≥ world×T×topk + 31×local_experts = 263_136` (`contracts.py:138-142`);
* per-`(source,dest)` reservation is bounded by `MAXTOK = 4096` and a source can push at most
  `T = 4096` deduplicated rows to any one destination, so
  `if (pos >= (unsigned int)MAXTOK) atomicOr(pperr, 65536);` (KERNEL:1037) is unreachable —
  identically to the exact case.

### 3.5 Numerical isolation of padded rows inside the megakernel

Padded-row garbage cannot contaminate real rows:

* the FP8 quantization scale is computed **per row** (per 128-element chunk within a row) —
  `a = fmaxf(a, fabsf(...))` reduced with `__shfl_xor` inside the row, then
  `scale = fmaxf(a, 1.0e-6f) / 448.0f` (KERNEL:1058-1065);
* the N2 GEMMs accumulate per output row; BM32 tiling groups rows but never mixes them;
* the combine sums partials belonging to the **same destination token** only
  (`pull_ptr`/`pull_src` indexed by `tau`, KERNEL:1893-1922);
* routing capacity is per `(source, dest)` pair, so a padded row consumes a slot but cannot
  displace a real row (§3.4).

Even a non-finite padded row therefore stays confined to its own output row — which is never
read (§5). This is the same isolation stock has.

### 3.6 One hard incompatibility to lock down

The megakernel has **no `-1` expert-id sentinel**. If `VLLM_MOE_SKIP_PADDING=1` were ever
enabled, MK:1151 would write `topk_ids = -1` for padded rows and KERNEL:1018 would compute
`dest = -1` for a lane with `s < TOPK`; that lane becomes `primary` (KERNEL:1020-1021, since
`match & ((1ULL << s) - 1ULL) == 0` for the lowest lane in the `dest == -1` group) and calls

```cpp
unsigned int* peer = hk_moe::peer_ptr(dest_counter + cur, dest, symmetric);   // KERNEL:1032-1033
```

with `rank = -1`, which indexes `descriptor->heap_bases[-1]`
(`moe_hk_adapter.cuh:49-54` → `kittens::distributed::translate_peer<8>`) — an out-of-bounds
read producing a garbage remote pointer and a remote atomic into it.

Today this is unreachable (env off, `is_padding` None, and for a real token `eid` is always in
`[0, 256)`; lanes with `s >= TOPK` also carry `dest == -1` but are excluded by the `s < TOPK`
conjunct at KERNEL:1021). M23 makes padded rows a *routine* occurrence and therefore makes the
flag a tempting "fix". **M23 must add an explicit activation-time refusal on
`envs.VLLM_MOE_SKIP_PADDING`** (§8.1, edit E6).

---

## 4. Q2 — The seal relaxation

### 4.1 The new predicate, and why every term must be DP-unanimous

The megakernel is a **collective**: rank *i*'s M1 pushes rows into rank *j*'s symmetric heap
and rank *j* spins on `chunk_ready`/`rows_done` epochs until they arrive. If a subset of ranks
replays the PF4H graph while the rest replay the stock graph or run eager, the PF4H ranks spin
to `spin_limit` and fail closed, and the stock ranks' Mori all2all never completes. **Split
decisions are catastrophic, not merely wrong.**

Classify every term of the current predicate:

| Term (GMR line) | Local or DP-unanimous? | M23 disposition |
|---|---|---|
| `VLLM_PF4H_INTEGRATION_MODE in ("full","m15")` (3950) | Global by deployment env | keep |
| `data_parallel_size == 8` (3952) | Global | keep |
| `not should_ubatch` (3953) | **Unanimous** — `torch.all(tensor[2] == 1)` on the all-reduce, DPU:62 | keep |
| `not uniform_decode` (3954) | **LOCAL** — `_is_uniform_decode` is purely this rank's `max_num_scheduled_tokens == 1 and num_tokens == num_reqs` (GMR:3821-3839) | **remove** (see §4.3) |
| `not has_lora` (3955) | **LOCAL** — `len(self.input_batch.lora_id_to_lora_request)` (GMR:3878-3884) | fold into the readiness bit; also globally false (no `lora_config` in this deployment) |
| `batch_descriptor.num_tokens == 4096` (3956) | Unanimous *only when DP padding happened* | **replace** with the direct test on the all-reduced padded row (§4.2) |
| `original_counts == (4096,)*8` (3962) | Unanimous | **delete** |
| `self._pf4h_graph_operator_enabled()` (3971) | **LOCAL** — a per-rank `os.path.isfile` latch (GMR:4007-4023) | **fold into the readiness bit** — this is a live split-brain bug today, merely masked by the seal's rarity |

Two hazards this table exposes that the brief did not anticipate:

* **H1 — the uniform-decode rank.** A rank whose local batch is a pure decode has
  `uniform_decode = True`. Its post-DP re-dispatch (GMR:3939-3942) passes that flag into
  `dispatch()`, whose PF4H early branch requires `not uniform_decode` (DISP:338), so it falls
  through to DISP:361-368 and returns `(CUDAGraphMode.NONE, BatchDescriptor(4096))`. That rank
  runs the **entire model eagerly at 4096 padded tokens** — today, in both arms. It can never
  seal under the current predicate. Since `fail_min_tok = 1` says this is the modal ragged
  shape, M23 *must* handle it or coverage stalls well below 100%.
* **H2 — the no-DP-padding step.** If any rank's local dispatch returns
  `CUDAGraphMode.NONE`, `synced_cudagraph_mode` is 0 (DPU:92-98, min over ranks; enum values
  `NONE=0, PIECEWISE=1, FULL=2` at `config/compilation.py:59-61`), so `should_dp_pad` is False
  (DPU:152) and `num_tokens_across_dp` holds **ragged per-rank padded counts**. A rank at
  4096 would then see `batch_descriptor.num_tokens == 4096` while its peers are at 1000. The
  naive relaxation "drop the original-count test" would seal *non-unanimously* here. This is
  the single most dangerous way to get M23 wrong.

### 4.2 The M23 predicate

Two derived booleans, both computed **only from all-reduced data**:

```python
# both are DP-unanimous by construction
b4096_unanimous = (
    self.parallel_config.data_parallel_size == 8
    and not should_ubatch                                   # DPU:62, all-reduced
    and synced_cudagraph_mode == CUDAGraphMode.PIECEWISE.value   # DPU:92-98, min over ranks
    and num_tokens_across_dp is not None
    and all(int(v) == 4096 for v in num_tokens_across_dp.tolist())  # DPU:77-89, max over ranks
)
pf4h_ragged_seal = b4096_unanimous and pf4h_ready_all       # new all-reduce row, §4.3
```

* `synced_cudagraph_mode == PIECEWISE` kills **H2**: DP padding provably happened, so
  `num_tokens_across_dp` is `[max]*8` and the `all(... == 4096)` test is a statement about
  *the group*, not about this rank.
* Dropping `uniform_decode` and testing `num_tokens_across_dp` instead of the local
  `batch_descriptor` kills **H1** on the predicate side; §4.3 kills it on the dispatch side.
* `original_num_tokens_across_dp` is **kept**, but demoted from gate to telemetry: it feeds
  the new receipt (§8.2) and the row-restricted accuracy attestation (§5.2).

### 4.3 The unanimity bit, and the mode override

**Readiness bit (new row 4 of the existing all-reduce).** DPU already all-reduces a
`(4, dp_size)` int32 tensor every step (DPU:47-54). Add one row:

```python
# DPU:47-51  (edit E1)
tensor_cpu = torch.zeros(5, dp_size, dtype=torch.int32)
tensor_cpu[0][dp_rank] = orig_num_tokens_per_ubatch
tensor_cpu[1][dp_rank] = padded_num_tokens_per_ubatch
tensor_cpu[2][dp_rank] = 1 if should_ubatch else 0
tensor_cpu[3][dp_rank] = cudagraph_mode
tensor_cpu[4][dp_rank] = 1 if pf4h_ready else 0        # NEW
```

```python
def _post_process_pf4h_ready(tensor: torch.Tensor) -> bool:   # NEW, mirrors DPU:57-74
    return bool(torch.all(tensor[4] == 1).item())
```

Zero extra collectives, zero extra latency (the tensor is already 4×8 int32 = 128 B).

The contributed bit must be computable **before** the all-reduce (it cannot depend on the
result):

```python
# GMR, new helper next to _pf4h_graph_operator_enabled (edit E3)
def _pf4h_local_readiness(self, *, has_lora: bool, pf4h_graph_target: bool | None) -> bool:
    if os.environ.get("VLLM_PF4H_INTEGRATION_MODE") not in ("full", "m15"):
        return False
    if os.environ.get("VLLM_PF4H_B4096_GRAPH_TARGET") != "pf4h":
        return False          # the stock arm never contributes readiness
    if self.parallel_config.data_parallel_size != 8:
        return False
    if has_lora or self.vllm_config.lora_config is not None:
        return False
    if envs.VLLM_MOE_SKIP_PADDING:
        return False          # §3.6 — the mega has no -1 sentinel
    if pf4h_graph_target:
        return True           # capture drive: the activation file does not exist yet
    return self._pf4h_graph_operator_enabled()
```

Note the behavioural change this implies: `_pf4h_graph_operator_enabled` is now polled every
step until it latches, rather than only on steps that would otherwise seal. That is one
`os.path.isfile` per step per rank until the sentinel appears — negligible, and it removes the
split-brain window entirely.

**Mode override (kills H1 on the dispatch side).** Immediately after the seal decision:

```python
# GMR, replacing 3973-3987 (edit E4)
if b4096_unanimous and cudagraph_mode != CUDAGraphMode.PIECEWISE:
    # A rank whose *local* batch looked like a uniform decode (or otherwise
    # missed DISP's early branch) dispatched to NONE even though DP padding
    # put it in the same B4096 bucket as everyone else.  Route it to the same
    # PIECEWISE B4096 graph the other seven ranks are using.
    cudagraph_mode = CUDAGraphMode.PIECEWISE
    batch_descriptor = BatchDescriptor(num_tokens=4096)

if pf4h_ragged_seal:
    batch_descriptor = replace(batch_descriptor, pf4h_exact_b4096=True)
elif (
    batch_descriptor.num_tokens == 4096
    and os.environ.get("VLLM_PF4H_B4096_GRAPH_TARGET") == "pf4h"
):
    # A PF4H-only server has no ordinary B4096 graph key.
    cudagraph_mode = CUDAGraphMode.NONE
```

Two facts make this safe and cheap:

* `BatchDescriptor(num_tokens=4096)` is **field-identical** to DISP's `regular_b4096`
  (`num_reqs=None, uniform=False, has_lora=False, num_active_loras=0`, FCTX:31-63 defaults vs
  DISP:342-348), so `replace(..., pf4h_exact_b4096=True)` reproduces the registered PF4H
  capture key from DISP:250-254 **exactly**. No new key, no new capture.
* PIECEWISE graphs exclude attention, so a decode-shaped rank replaying the 4096 graph is
  fine: attention runs outside the graph on unpadded metadata (GMR:4323, 4382-4392). Forcing
  PIECEWISE is strictly *better* for that rank than today's `NONE`.

**Arm symmetry.** `b4096_unanimous` is target-independent, so the override applies to the
stock arm too (routing its uniform-decode rank into `regular_b4096`, which *is* registered
when `graph_target == "stock"`). This is a genuine improvement to the baseline and it must be
landed in both arms and **re-baselined before the A/B** — otherwise M23 would be credited with
a win it did not produce. It can be held behind `VLLM_PF4H_B4096_UNIFORM_RESCUE` for a
one-run ablation; recommendation is to land it on and re-baseline.

### 4.4 Every downstream consumer of the exactness assumption

| # | Site | Depends on exactness? | Change required |
|---|---|---|---|
| 1 | GMR:3947-3962 seal | **Yes — the only functional one** | rewrite per §4.2 |
| 2 | GMR:3964-3971 capture override / operator gate | Indirect | `pf4h_graph_target` branch unchanged; move the operator gate into readiness (E3) |
| 3 | GMR:3973-3981 eager fallback | Yes (it is the *consequence* of a failed seal) | keep, but now reached ~never on in-bucket steps |
| 4 | GMR:3983-3987 descriptor stamp | No | unchanged (see key-identity note above) |
| 5 | GMR:4260-4283 replay receipts | No — but they are **one-shot latches**, not counters (`self._pf4h_graph_replay_logged`, GMR:849-850) | add real counters (§8.2) |
| 6 | GMR:5936, 5959, 6854, 6866 capture drive | No — `_dummy_run(4096)` fabricates a genuinely unpadded 4096-token batch (GMR:5918-5926), so capture stays exact-4096 on all ranks | unchanged |
| 7 | FCTX:59-63 `BatchDescriptor.pf4h_exact_b4096` + docstring | Docstring only ("*proved that every rank had exactly 4096 original live tokens before graph padding*") | **keep the field name** (graph-key identity, `apply.py` sentinel at `apply.py:46`, SHIM test at `test_full_integration.py:156`); rewrite the docstring |
| 8 | DISP:243-255 key registration | No | unchanged |
| 9 | DISP:331-359 bucket routing | No — it already returns `regular_b4096` for the whole `(512, 4096]` range | unchanged (optionally drop `not uniform_decode` at DISP:338 as an alternative to E4; **not recommended**, it perturbs the dispatcher for all callers) |
| 10 | SHIM `vllm_full.py:262-294` `_attest_dp_batch` | **No — already ragged-correct.** It tests `dp_metadata.num_tokens_across_dp_cpu == (4096,)*8`, i.e. *post*-padding counts | comment rewrite at :268-270 only |
| 11 | SHIM `vllm_full.py:345-409` `_eligible` shape gates | No — `_shape(a1) == (4096, 7168)` etc. hold for ragged batches, since the tensors are padded | comment rewrite at :359-362 |
| 12 | SHIM `vllm_full.py:3-6` and `m15_vllm.py:3-13` docstrings | Documentation | rewrite |
| 13 | SHIM `contracts.py:272-273` `PREFILL_B4096_BUCKET` comment | Documentation | rewrite |
| 14 | SHIM `contracts.py:113-166` `GraphBucket` capacities (`t_loc_max`, `pad_max`, `n2_rowcap`) | **No** — all derived from `world × tokens_per_rank`, i.e. worst case; ragged is strictly under | unchanged |
| 15 | SHIM `m15_graph_body.py` fixed shapes | **No tensors at all** — "The body deliberately takes *no* tensor arguments" (`m15_graph_body.py:9-12`); one launch of `desc/pperr/spin_limit` | unchanged |
| 16 | SHIM `m15_kernargs.py` ABI (`M15_MEGA_KERNARG_PINNED = 280`) | No — the kernarg segment is `desc, pperr, spin_limit`, independent of batch content | unchanged |
| 17 | SHIM `m15_runtime.py:941` `DescriptorSlot.T = 4096` | No — launch constant, stays 4096 | unchanged |
| 18 | SHIM `m15_runtime.py:1053-1115` `bind_dynamic` + `M15-DESC-012` | **Indirectly critical** — see §6.3 | unchanged, but §4.3's "sealed ⇒ graph on all 8 ranks" invariant is what protects it |
| 19 | SHIM `contracts.py:77-107` `FrozenDeploymentContract.validate_observed` | **No** — it validates *model/topology* fields (`tp_size`, `dp_size`, `ep_size`, `global_experts`, `topk`, `hidden_size`, ...), never batch shape. Called once from `validate_full_selection` (`vllm_full.py:74`) | unchanged — but see the `max_nonfinite` risk in §5.3 |
| 20 | SHIM `tests/test_full_integration.py:132-158` source assertions | Yes — they assert the exact source strings of `_attest_dp_batch` | update to the new comments; add a positive assertion that the ragged predicate is present |
| 21 | SHIM `apply.py:44-52,81-85` `PATCH_SENTINELS` / `M15_PATCH_SENTINELS` | No (existing sentinels survive) | **add** a `RAGGED_SEAL` sentinel so an install can be verified |

**Nothing on this list requires a kernel rebuild, a re-capture, a new bucket, or a descriptor
layout change.**

---

## 5. Q3 — Accuracy and the output contract

### 5.1 The padded-row region is ignored by the caller — proven, not assumed

`finalize` writes all 4096 rows of `output` (SHIM `m15_vllm.py:275-285` type-checks
`(4096, 7168)`; KERNEL:1891-1922 M8 writes `T` rows). That is byte-for-byte the stock contract
(MORI:118-124). Downstream:

* **Sampling** — GMR:4491 `sample_hidden_states = hidden_states[logits_indices]`.
  `logits_indices = query_start_loc[1:] - 1` (GMR:2196-2198) with
  `query_start_loc[1:num_reqs+1] = cu_num_tokens` (GMR:2033), so every index is
  `< total_num_scheduled_tokens = n_orig`. The spec-decode variant (GMR:2829-2855) is built
  from the same cumulative offsets.
* **Pooling** — GMR:3382 `hidden_states[:num_scheduled_tokens]`.
* **Prompt logprobs** — GMR:3759 `hidden_states[:num_scheduled_tokens]`.
* **Drafter/EAGLE** — GMR:5227/5244/5265, all unpadded.
* **KV cache** — two independent guards: in PIECEWISE the slot mapping is sized
  `num_tokens_unpadded` so the pad has no slot at all (GMR:4371-4374); in FULL it is
  `-1`-filled (GMR:4107-4109). Padded *requests* are additionally fenced by
  `blk_table_tensor[num_reqs:].fill_(NULL_BLOCK_ID)` (GMR:2297-2299) and
  `seq_lens[num_reqs:].fill_(0)` (GMR:2154).

**No zeroing of the padded output region is required**, and adding one would be a gratuitous
divergence from stock.

### 5.2 The attestation

Two levels, both required before flipping the coverage default on.

**A — exact-token A/B (the gate).** Same binary, same prompts, same seed, greedy decoding
(`temperature=0`), the two arms differing **only** in `VLLM_PF4H_B4096_GRAPH_TARGET`
(`stock` vs `pf4h`). Compare the generated token-id sequences for **bitwise equality**. This
is the right gate precisely because §5.1 proves padded rows cannot reach the sampler: any
divergence is a real-token divergence.

Caveat to control for: BF16 non-associativity means stock and mega need not agree bit-for-bit
on real rows either; the existing tolerance gate is `max_abs ≤ 0.02`, `rel_L2 ≤ 0.01`
(SHIM `contracts.py:65-66`). Run the A/B at both `temperature=0` (expect long common prefixes,
report first-divergence index distribution) and with the tensor-level check below.

**B — row-restricted tensor comparison (the diagnostic).** Add a debug mode that, on a sealed
step, runs both the mega and stock for one layer and compares **only rows `[0, n_orig_local)`**:

```python
n_orig_local = int(original_num_tokens_across_dp[dp_rank])
# published to the shim host-side, non-graph path only:
forward_context.additional_kwargs["pf4h_n_orig"] = n_orig_local
```

Rows `≥ n_orig_local` **must be excluded**: stock and the mega quantize the same garbage
differently and will disagree arbitrarily there. This is a diagnostic-only path; it must never
run inside capture (it reads host state).

### 5.3 The `max_nonfinite = 0` chokepoint

`FROZEN_CONTRACT` carries `max_abs = 0.02`, `max_rel_l2 = 0.01`, **`max_nonfinite = 0`**
(SHIM `contracts.py:65-67`), consumed by `WeightLayoutReceipt.validate`
(SHIM `runtime.py:248-259`) and `CorrectnessReceipt.validate` (SHIM `runtime.py:300-318`).

A whole-tensor nonfinite scan over a ragged batch's 4096-row output **can legitimately fail**:
padded rows carry stale embeddings plus attention-buffer residue, and the megakernel's
per-row FP8 path (`scale = fmaxf(a, 1e-6f)/448`, KERNEL:1065) will propagate a non-finite input
to a non-finite output row. That row is discarded (§5.1) but a naive receipt would refuse
activation.

**Requirement:** every correctness receipt produced from a ragged serving step must be
computed over rows `[0, n_orig)` only, and the receipt payload must record `n_orig` so the
restriction is auditable. `FrozenDeploymentContract.validate_observed` itself
(SHIM `contracts.py:77-107`) is untouched — it validates topology, not batches — but it is the
same tolerance object, so the two receipt validators are the sites to fix.

**Probe P4 (§9)** measures whether non-finite padded rows actually occur, in *both* arms.

---

## 6. Q4 — Graph capture at exact-4096, replay on ragged batches

### 6.1 What the captured PIECEWISE graph reads

| Input | Allocation | Pointer-stable across replays? | Content per replay |
|---|---|---|---|
| `input_ids` | `self.input_ids` `CpuGpuBuffer`, GMR:748, sized `max_num_tokens = 4096` | **Yes** — allocated once in `__init__` | rewritten `[:n_orig]`, stale beyond |
| `positions` | `torch.zeros(max_num_tokens, ...)`, GMR:749-751 | **Yes** | `[:n_orig]` written, `[n_orig:4096]` zeroed (GMR:3572-3573) |
| MoE `output` | `torch.empty_like(hidden_states)`, MK:1413 | **Yes inside the graph** — the allocation is captured; replay reuses the captured pointer | overwritten each replay |
| `topk_ids` / `topk_weights` | router outputs, intra-graph intermediates | **Yes** (captured) | recomputed each replay |
| mega descriptor `desc` | `torch.zeros((words_len,), int64)`, SHIM `m15_runtime.py:964-968`, once per layer | **Yes** | **never rewritten per replay** |
| `pperr`, symmetric-heap buffers | MoRI arena, SHIM `m15_runtime.py:746-820` | **Yes** (symmetric-offset attested, `M15_SYMMETRIC_OFFSET_RECEIPT`, `m15_runtime.py:904-910`) | device-side epoch protocol |
| weights `w13/s13/w2/s2` | model weights | **Yes** — `bind_layer_weights` refuses a change (`M15-WEIGHT-002`, `m15_runtime.py:1023-1032`) | constant |

### 6.2 Does the ragged case need a new per-replay scalar? **No.**

The megakernel reads exactly three kernel arguments — `desc`, `pperr`, `spin_limit`
(SHIM `m15_kernargs.py:12-14`) — and everything else through the descriptor, which is a
**device table written out of capture**:

```python
# SHIM m15_runtime.py:981-990
def _write_descriptor(self, record):
    """Copy the host mirror into the device table.  OUT OF CAPTURE ONLY."""
    if self.torch.cuda.is_current_stream_capturing():
        _fail("M15-DESC-010", ...)
```

There is **no in-graph op that updates the descriptor**, so a design needing per-step `n_orig`
in the kernel would require inventing one (a captured `desc[slot].copy_(pinned_scalar)` on
stream). Since T stays 4096 and no kernel logic depends on `n_orig` (§3.4), **nothing new is
needed.** This is the strongest structural argument for the "dispatch all 4096" design.

### 6.3 What `launch_prepare` / `bind_dynamic` rewrite per replay: nothing

Under M15 (the deployed mode, `VLLM_PF4H_INTEGRATION_MODE=m15`), the four per-call pointers
are latched, not rewritten:

```python
# SHIM m15_runtime.py:1083-1099
if capturing:
    record.pending_dynamic = values
    return
latched = record.latched_dynamic
if latched == values and record.dynamic_bound:
    return                       # steady state: no copy, no synchronize
if latched is not None and latched != values:
    _fail("M15-DESC-012", ... "the descriptor cannot be rewritten per call "
                              "inside a captured region")
```

and are published once after capture closes by `commit_captured_descriptors()`, driven from
GMR:6868-6885 (`M15_POST_CAPTURE_COMMIT` / `M15_GRAPH_COMMIT_RECEIPT`).

**Implication for M23:** a sealed step must *always* be a graph-replay step on all 8 ranks. If
a sealed rank ever ran the mega **eagerly**, `torch.empty_like` (MK:1413) could hand back a
different `output` pointer than the captured one and trip `M15-DESC-012` mid-serve. §4.3's
mode override guarantees `pf4h_ragged_seal ⇒ cudagraph_mode == PIECEWISE` on every rank, which
is exactly the invariant that protects this. **Add an assertion to that effect** (edit E4).

### 6.4 How production plumbs the equivalent into *its* captured graph: it doesn't

`DPMetadata` holds a single field, `num_tokens_across_dp_cpu`, plus an SP-only `local_sizes`
(FCTX:77-82). It is:

* **CPU-resident** (DPU:83-89 returns `torch.tensor(..., device="cpu")`),
* **freshly allocated every step** — both the all-reduce staging tensor (DPU:47) and the
  padding result (DPU:83-87) — hence pointer-*unstable*, which is harmless because it is
  host-side and outside the graph,
* **reconstructed per forward pass** (FCTX:313-315 builds a new `DPMetadata`),
* and **never read by the MoRI/AITER path at all** (the only `dp_metadata` consumers are
  DeepEP-v2, the FlashInfer NVLink prepare/finalizes, batched-DeepGEMM, humming-MoE, and the
  routed-experts capturer; `moe_runner.py:606-611` touches it only to set SP local sizes).

So **production's captured B4096 graph carries no per-step token-count tensor either.** It is
`num_tokens`-agnostic at replay, which is exactly why it can already serve ~230 ragged steps
per run. The megakernel graph inherits the same property for free.

The one place the count *is* read is the shim's own attestation
(`vllm_full.py:284-294`), which runs on the **Python** path — at capture and on eager
fall-through — never during replay (a CUDA-graph replay executes no Python).

---

## 7. Q5 — The dispatch-cost delta

### 7.1 Where the megakernel newly *loses*: cross-rank arrival skew

This is the one real regression channel, and it is not about dispatch volume.

Both arms process 4096 MoE rows per rank on every in-bucket step (§3.2), so the *padded waste*
is arm-symmetric and exactly equal. What is **not** symmetric is what a rank does while
waiting for its peers.

Attention cost scales with `n_orig`, not with 4096 (attention metadata is unpadded in
PIECEWISE, GMR:4323/4382-4392). On a ragged step where rank A has 4096 real tokens and rank B
has 1, rank B reaches each MoE layer far earlier than rank A. Per-layer skew, not whole-model
skew, since the mega is itself a collective that re-synchronizes every layer.

* **Stock:** rank B's Mori dispatch issues its sends and its receive-wait blocks. The GPU is
  idle either way.
* **Mega:** rank B enters `k0pf6gm_m15_mega` with **all 256 CTAs** (`M15_GRID = M15_BLOCK =
  256`, SHIM `m15_contracts.py:440-441`) and spins on peer epochs. This repo's own history
  names the cost — the "mode-14 law": spinning CTAs continuously invalidate L2 under the
  concurrent GEMMs. Inside a single-kernel megakernel the victims are the *arriving peer
  writes* and the local N2 tiles.
* **Hard failure mode:** the spin is bounded. `M15_SPIN_LIMIT = 20_000_000`
  (SHIM `m15_contracts.py:443`), written to `DescriptorSlot.SPIN` (`m15_runtime.py:939`) and
  passed as the third kernel argument (`m15_runtime.py:1253`). Exceeding it sets a `pperr` bit
  and fail-closes the invocation (e.g. `hk_moe::set_error_bit{pperr, 33554432}` at
  KERNEL:1881-1885). That budget was tuned on **exact-4096, near-lockstep** batches. Ragged
  batches are precisely the regime that stresses it.

**This is risk #1** and it is why the validation ladder starts with a spin-headroom probe
(P2, §9) before any throughput claim. The kernel already instruments it: `DescriptorSlot.SPIN_DBG
= 52` (`m15_contracts.py:136`, a 2-word int32 buffer at `m15_contracts.py:869-871`) is written at
KERNEL:1238.

### 7.2 Where the megakernel does *not* lose

| Step class | Stock arm | PF4H arm today | PF4H arm after M23 | Verdict |
|---|---|---|---|---|
| All 8 ranks exactly 4096 (~2% of in-bucket) | B4096 graph | **mega** (graph) | mega (graph) | unchanged |
| In-bucket, ragged, no uniform-decode rank | B4096 graph | **eager, stock ops** | mega (graph) | M23 removes a loss |
| In-bucket, ragged, ≥1 uniform-decode rank | B4096 graph on 7 ranks, **eager on that rank** | eager everywhere | mega (graph) on all 8 | M23 removes a loss on both arms (§4.3) |
| Not unanimous / no DP padding (H2) | eager | eager | eager (seal refuses) | unchanged |
| Out of bucket (≤512 tokens, ~370 steps) | small PIECEWISE or FULL keys | same | same | untouched by M23 |

There is **no step class where M23 causes the mega to run on a batch production would have run
smaller.** The dispatcher pads everything in `(512, 4096]` to 4096 for both arms (DISP:331-359),
and both arms then execute 4096 MoE rows. The megakernel's dispatch volume on a ragged step is
identical to stock's on the same step.

### 7.3 A pre-existing confound the A/B must control

The PF4H patch **removes** the `≤256` mixed PIECEWISE keys whenever
`VLLM_PF4H_INTEGRATION_MODE` is set (DISP:204-212):

```python
if (os.environ.get("VLLM_PF4H_INTEGRATION_MODE") in ("full", "m15")
        and batch_desc.num_tokens <= 256):
    # With an explicit B4096 key present, ROCm 7.2.3 stalls entering the
    # mixed <=256 regime.  FULL remains.
    continue
```

So small mixed batches run without a piecewise graph in **both** arms — provided the baseline
is also run with `VLLM_PF4H_INTEGRATION_MODE=m15` and only `VLLM_PF4H_B4096_GRAPH_TARGET`
flipped. **Any A/B that uses an unpatched vLLM as the baseline is invalid.** State this in the
run book.

---

## 8. Implementation plan

### 8.1 Ordered edits

| # | File : lines | Edit | LOC |
|---|---|---|---|
| **E1** | `DPU:36-54` (`_run_ar`) | widen the staging tensor to `(5, dp_size)`; add `pf4h_ready: bool = False` parameter and `tensor_cpu[4][dp_rank]`. | ~6 |
| **E2** | `DPU:57-167` | add `_post_process_pf4h_ready(tensor)`; thread it through `_synchronize_dp_ranks` and `coordinate_batch_across_dp` (new `pf4h_ready` in, new `pf4h_ready_all` + `synced_cudagraph_mode` out). **Free win while here:** hoist a single `tensor_host = tensor.cpu()` and do all four post-processing reads host-side — today DPU:62, 67-68, 82, 98 each force a separate device sync. | ~35 |
| **E3** | `GMR:4007-4023` | add `_pf4h_local_readiness()` next to `_pf4h_graph_operator_enabled()` (§4.3). | ~18 |
| **E4** | `GMR:3912-3987` | compute readiness before the all-reduce; pass it in; capture `synced_cudagraph_mode` and `pf4h_ready_all` out; replace the seal with `b4096_unanimous` / `pf4h_ragged_seal`; add the mode override and the `pf4h_ragged_seal ⇒ PIECEWISE` assertion. Keep the `pf4h_graph_target` branch (GMR:3964-3969) verbatim so capture still refuses a non-exact drive. Keep `original_num_tokens_across_dp` for telemetry. | ~55 |
| **E5** | `GMR:4260-4283` + `GMR:848-850` | replace the two one-shot latches with real per-step counters and a periodic `RAGGED_SEAL_RECEIPT` (§8.2). | ~40 |
| **E6** | `SHIM vllm_full.py:144-154` (`validate_full_selection`) | add blocker `PF4H-FULL-020` refusing activation when `envs.VLLM_MOE_SKIP_PADDING` is true — the megakernel has no `-1` expert-id sentinel (§3.6). | ~10 |
| **E7** | `SHIM vllm_full.py:3-6, 268-270, 359-362`; `m15_vllm.py:3-13`; `contracts.py:272-273`; `FCTX:59-63` | docstring/comment rewrites: "non-ragged, exactly 4096 original live tokens" → "all eight ranks in the padded B4096 bucket; original counts may be ragged". **No logic changes.** | ~20 |
| **E8** | `SHIM tests/test_full_integration.py:132-158` | update the source-string assertions; add assertions that the ragged predicate and the `VLLM_MOE_SKIP_PADDING` refusal are present. | ~15 |
| **E9** | `SHIM apply.py:44-52, 81-85` | add a `RAGGED_SEAL` patch sentinel (e.g. `"v1/worker/dp_utils.py": "pf4h_ready"` and `"v1/worker/gpu_model_runner.py": "RAGGED_SEAL_RECEIPT"`) under a new `M23_PATCH_SENTINELS`. | ~8 |
| **E10** | `SHIM runtime.py:248-259, 300-318` | make the receipt validators take an explicit `rows_compared` / `n_orig` field and require it to be recorded, so a whole-tensor scan cannot be submitted for a ragged step (§5.3). | ~15 |

**Total: ~220 LOC, no HIP, no re-capture, no ABI change.**

Explicitly **not** changed: `DISP:331-359` (§4.4 row 9), `m15_graph_body.py`,
`m15_kernargs.py`, `m15_runtime.py` descriptor layout, `contracts.py` bucket capacities,
`k0pf6gm_device_tile_m15.hip`.

### 8.2 New receipts

The existing PF4H receipts are one-shot booleans (`self._pf4h_graph_replay_logged`,
`self._pf4h_stock_graph_replay_logged`, GMR:849-850) and cannot measure coverage. Add
monotonic counters on the runner, emitted every N steps (N=100) and once at shutdown:

```
RAGGED_SEAL_RECEIPT rank=%d steps=%d in_bucket=%d sealed=%d sealed_ragged=%d
  sealed_exact=%d refused_not_unanimous=%d refused_not_ready=%d eager_b4096=%d
  min_orig=%d max_orig=%d sum_orig=%d
```

Definitions (all per rank, all cumulative):

* `steps` — every call to `_determine_batch_execution_and_padding` from `execute_model`;
* `in_bucket` — `b4096_unanimous` true (the denominator the fix targets);
* `sealed` — `pf4h_ragged_seal` true;
* `sealed_exact` — sealed **and** `original_counts == (4096,)*8` (the old population; should stay ≈2% of `sealed`);
* `sealed_ragged` — `sealed - sealed_exact` (the population M23 creates);
* `refused_not_unanimous` — `batch_descriptor.num_tokens == 4096` locally but `b4096_unanimous` false (the H2 class; **should be ~0**);
* `refused_not_ready` — `b4096_unanimous` true but `pf4h_ready_all` false (activation not yet latched somewhere; **should go to 0 after warmup**);
* `eager_b4096` — in-bucket steps that ended at `CUDAGraphMode.NONE` (**must be 0 after M23**);
* `min_orig / max_orig / sum_orig` — over `original_num_tokens_across_dp`, for the skew and token-weight analyses.

**Success criterion:** `sealed == in_bucket` and `eager_b4096 == 0` on every rank.

### 8.3 Validation ladder

**L0 — offline dispatcher replay (no GPU).** Extract `(step, rank, n_orig)` from a logged
c32p trace (the `RAGGED_SEAL_RECEIPT` fields, or a one-line-per-step debug log). Re-run the
pure-Python decision function — `_pad_for_sequence_parallelism` → `DISP.dispatch` →
`_post_process_dp_padding` → the M23 predicate — over the trace and assert
`sealed == in_bucket` and `refused_not_unanimous == 0`. This catches H1/H2 before any node
time. ~150 LOC of harness; reuse `overnight/aug18-prefill/m15_router_skew.py` for the trace
plumbing.

**L1 — single-pair serving smoke, coverage only.** One c32p run, `graph_target=pf4h`, mega
activation **deliberately withheld** (no activation file) so `refused_not_ready` counts every
in-bucket step. Confirms the counter wiring and `in_bucket ≈ 230` without risking a hang.

**L2 — single-pair serving, mega live, coverage proof.** Same run with the activation file
present. Gate: `sealed == in_bucket`, `sealed_ragged ≈ 0.98 × in_bucket`, `eager_b4096 == 0`,
zero `pperr`, zero `M15-DESC-012` / `M15-STATE-*` blockers. **This is the coverage claim.**

**L3 — spin headroom (risk #1).** Same run with `SPIN_DBG` readout enabled; record the maximum
observed spin fill per layer and compare with `M15_SPIN_LIMIT = 20_000_000`. Gate: peak
< 25% of budget. If not, raise the limit (a descriptor scalar, no recompile) *and* investigate
the skew.

**L4 — exact-token accuracy A/B.** Two runs, same binary, same seed, `temperature=0`, only
`VLLM_PF4H_B4096_GRAPH_TARGET` flipped. Gate: identical generated token ids, or a documented
first-divergence distribution plus a row-restricted tensor check (§5.2) inside the frozen
tolerance.

**L5 — throughput A/B.** Only after L2–L4 pass. **Re-baseline first** (§4.3): the stock arm
now also gets the uniform-decode rescue. Report tok/s and p50/p99 TTFT/ITL for both arms,
plus the coverage counters from both, so the reader can see the candidate arm was actually
exercised.

### 8.4 Rollback

Every change is behind existing env gates. Setting `VLLM_PF4H_B4096_GRAPH_TARGET=stock`
disables the readiness bit at its source (`_pf4h_local_readiness` returns False), so
`pf4h_ragged_seal` is permanently false and the only residual delta is the
uniform-decode rescue — itself guardable with `VLLM_PF4H_B4096_UNIFORM_RESCUE=0`.

### 8.5 Effort

| Task | Estimate |
|---|---|
| E1–E5 (vLLM patch) | 3 h |
| E6–E10 (shim, tests, apply.py, receipts) | 2 h |
| L0 offline replay harness | 3 h |
| Review + patch-set regeneration (`apply.py`) | 1 h |
| **Code subtotal** | **~9 h** |
| L1–L3 node runs | 2 h |
| L4–L5 node runs (4 serving runs) | 2 h |
| **Node subtotal** | **~4 h** |

---

## 9. Runtime probes the mirror cannot answer

| # | Probe | Why the mirror cannot answer it | How to run |
|---|---|---|---|
| **P1** | Token-weighted coverage: `sum(n_orig)` over in-bucket steps ÷ over all steps. | Depends on the live scheduler's chunking under prefix caching. | `sum_orig` in `RAGGED_SEAL_RECEIPT` (§8.2), split by `in_bucket`. |
| **P2** | Peak spin fill per layer on ragged steps vs. `M15_SPIN_LIMIT`. | Purely dynamic; depends on measured cross-rank attention skew. | Read `DescriptorSlot.SPIN_DBG` (slot 52, `m15_contracts.py:136`, written at KERNEL:1238) after each run; report max and 99th pct. **Blocking gate for L5.** |
| **P3** | Confirm `max_cudagraph_capture_size == 512` and `max_num_batched_tokens == 4096` on the live server. | Config-dependent; the mirror only carries the comment (DISP:215-216) and the assert (DISP:236-241). | Log `compilation_config.max_cudagraph_capture_size` and `cudagraph_capture_sizes` once at startup. |
| **P4** | Do padded rows ever carry non-finite hidden states? Measure in **both** arms. | Depends on stale buffer contents at runtime. | On a debug step, `torch.isfinite(hidden_states[n_orig:]).all()` before the MoE, per layer. Feeds §5.3. |
| **P5** | Confirm stock dispatches padded rows on the wire (independent check of §3.2). | The count is a live MoRI return value. | Log `dispatch_recv_token_num.sum().item()` next to `a1.shape[0]` at MORI:97. With padding included it should sit near `4096 × topk / ep_size` per rank **regardless of `n_orig`**; a masking implementation would scale with `n_orig`. |
| **P6** | Frequency of the H1 (uniform-decode rank) and H2 (no-DP-padding) classes. | Scheduler-dependent. | `refused_not_unanimous` and a new `uniform_rescued` counter in `RAGGED_SEAL_RECEIPT`. |
| **P7** | Whether the activation-file latch is ever non-unanimous in a window. | Filesystem timing. | `refused_not_ready` should reach a fixed value early and never increase again. |

---

## 10. Risk register

| # | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| **R1** | **Spin-limit exhaustion from ragged cross-rank skew.** `M15_SPIN_LIMIT = 20_000_000` was tuned for near-lockstep exact-4096 batches; ragged attention costs differ by up to 4096× between ranks. Exceeding it sets `pperr` and fail-closes the invocation. | **High** — a serving error, not a slowdown | Medium | Probe P2 as a blocking L3 gate; `spin_limit` is a descriptor scalar (`m15_runtime.py:939`), raisable without recompile. |
| **R2** | **Non-unanimous seal ⇒ collective deadlock.** Any local term left in the predicate (`uniform_decode`, `has_lora`, the activation-file latch) can split the group; PF4H ranks spin out while stock ranks' Mori all2all hangs. | **Critical** | Medium if done naively | The whole of §4.2-4.3: predicate built only from all-reduced data, readiness folded into the existing all-reduce, mode override forcing PIECEWISE on all 8. L0 offline replay proves it over a real trace before any node run. |
| **R3** | **`M15-DESC-012` mid-serve.** If a sealed step ever runs the mega *eagerly*, `torch.empty_like` (MK:1413) can return a different `output` pointer than the captured one and the runtime refuses (`m15_runtime.py:1091-1099`). | High | Low, once §4.3 holds | Assert `pf4h_ragged_seal ⇒ cudagraph_mode == PIECEWISE` in E4; counter `eager_b4096` must stay 0. |
| **R4** | **`max_nonfinite = 0` receipts fail on ragged steps.** `SHIM contracts.py:67` consumed by `runtime.py:251, 307`. Padded rows can legitimately be non-finite. | Medium — blocks activation, not correctness | Medium | E10: receipts must record `n_orig` and compare rows `[0, n_orig)` only. Probe P4 quantifies. |
| **R5** | **`VLLM_MOE_SKIP_PADDING` enabled later.** Sets `topk_ids = -1` (MK:1151); the mega has no `-1` sentinel and `peer_ptr(..., -1, ...)` reads `heap_bases[-1]` (KERNEL:1032-1033, `moe_hk_adapter.cuh:49-54`) — out-of-bounds remote atomic. | **Critical** if it happens | Low today (default off, V1 runner never sets `is_padding`) | E6: hard activation refusal `PF4H-FULL-020`. |
| **R6** | **Baseline contamination.** The uniform-decode rescue improves the stock arm too; comparing post-M23 candidate against pre-M23 baseline over-credits M23. Separately, an unpatched-vLLM baseline is invalid because of DISP:204-212. | Medium — a reporting error | High if unmanaged | §4.3 and §7.3: re-baseline with the same binary, flip only `VLLM_PF4H_B4096_GRAPH_TARGET`; report both arms' coverage counters. |
| **R7** | **Graph-key collision** — the original motivation for the exactness gate (§2a). | Low | Low | Structurally prevented by the distinct `pf4h_exact_b4096` field (FCTX:59) and by DISP:243-255 registering exactly one B4096 key per server. Unchanged by M23. |
| **R8** | **Descriptor-word attestation drift.** `attest_descriptor_words` (`m15_runtime.py:1104-1114`, `1131-1141`) validates T/MAXTOK/capacities against the bucket. | Low | Low | Nothing in M23 changes any descriptor word; the attestation should pass byte-identically. Assert this in the L2 log diff. |
| **R9** | **Ragged batches change expert load balance.** Padded rows route by stale-token embeddings, plausibly a different expert distribution than real tokens. | Low — perf only | Medium | Symmetric across arms (both dispatch the same padded rows with the same router). Watch `hcnt` histogram spread if L5 disappoints. |

---

## 11. Evidence index

**Coverage / dispatch**
`DISP:204-212` (≤256 key removal) · `DISP:215-255` (single B4096 key per server) ·
`DISP:331-359` (the `(512,4096]` bucket rule) · `DISP:361-368` (fallthrough to NONE) ·
`DPU:47-54` (the 4-row all-reduce) · `DPU:62` (`should_ubatch` unanimity) ·
`DPU:77-89` (max-padding) · `DPU:92-98` (min cudagraph mode) · `DPU:152` (`should_dp_pad`) ·
`DPU:161` (original counts) · `config/compilation.py:59-61` (mode enum values).

**The seal**
`GMR:3821-3839` (`_is_uniform_decode`, local) · `GMR:3886-3945` (padding chain) ·
`GMR:3947-3962` (the exactness gate) · `GMR:3964-3971` (capture drive + operator gate) ·
`GMR:3973-3981` (eager fallback in a PF4H-only server) · `GMR:3983-3987` (descriptor stamp) ·
`GMR:4007-4023` (`_pf4h_graph_operator_enabled`) · `GMR:4244-4258` (unpack) ·
`GMR:4260-4283` (one-shot receipts) · `GMR:848-850` (the latches) ·
`GMR:5918-5926` (`_dummy_run` fabricates an unpadded 4096 batch) · `GMR:5959, 6854, 6866`
(capture drive) · `GMR:6868-6885` (post-capture descriptor commit) ·
`FCTX:59-63` (`pf4h_exact_b4096`).

**Padding semantics**
`GMR:494, 748-757` (persistent buffers, `max_num_tokens = 4096`) ·
`GMR:3562` (padded `input_ids` reach the model) · `GMR:3570-3573` (positions zeroed) ·
`GMR:4107-4109` (slot-mapping `-1`) · `GMR:4323` (`pad_attn`) ·
`GMR:4371-4392` (unpadded attention metadata) · `GMR:4436-4456` (forward context, no
`is_padding`) · `GMR:4491` (`hidden_states[logits_indices]`) ·
`MK:1137-1151` (the only mask, double-disabled) · `MK:802-805` (`M = a1.size(0)`) ·
`MK:1413` (`torch.empty_like`) · `MORI:89-99` (unsliced dispatch) ·
`MORI:118-124` (combine writes 4096) · `deepseek_v2.py:403-427` ·
`moe_runner.py:573-586` (router over 4096 rows) · `envs.py:191, 1499`
(`VLLM_MOE_SKIP_PADDING=0`) · `gpu_worker.py:384-398` (V1 runner selected) ·
`FCTX:77-133` (DPMetadata: CPU, per-step, unread by MoRI).

**Shim**
`vllm_full.py:3-6` (docstring) · `:262-294` (`_attest_dp_batch`, already ragged-correct) ·
`:345-409` (`_eligible`) · `:411-449` (`prepare`) · `:451-485` (`finalize`) ·
`m15_vllm.py:3-13, 200-318` · `contracts.py:63-67` (tolerances), `:113-166` (bucket
invariants), `:272-280` (`PREFILL_B4096_BUCKET`) · `runtime.py:248-259, 300-318` (receipt
validators) · `m15_contracts.py:81-151` (descriptor slots), `:440-443` (grid/spin),
`:504-506` (M15 bucket) · `m15_runtime.py:916-995` (descriptor construction),
`:1053-1115` (`bind_dynamic`), `:1117-1151` (`commit_pending`), `:1230-1254` (`launch`) ·
`m15_graph_body.py:9-12` (no tensor arguments) · `m15_kernargs.py:12-14, 123` (ABI) ·
`apply.py:44-52, 81-85` (patch sentinels) · `tests/test_full_integration.py:132-158`.

**Kernel**
`KERNEL:226` (`K0P6_D_T = 44`) · `:759` (`input_tokens`) · `:977, 993` (M1 loop over T) ·
`:1018` (`dest = eid / E`) · `:1020-1037` (primary / reserve / `MAXTOK` guard) ·
`:1058-1065` (per-row FP8 scale) · `:1375` (M3-M5) · `:1238` (`SPIN_DBG`) ·
`:1881-1885` (spin fail-closed bit) · `:1891-1922` (M8 combine over T rows) ·
`moe_hk_adapter.cuh:49-54` (`peer_ptr` → `translate_peer<8>`).

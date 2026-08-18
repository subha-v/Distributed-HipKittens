# M15 + EPLB placement-only compose (`num_redundant_experts=0`)

**Verdict: sound, and cheaper than the brief assumed.** The compose needs
*no* topk_ids gather, *no* kernel change, and *no* change to the captured
graph body. vLLM 0.25.1 already resolves logical expert ids to physical slot
ids **inside the router**, before the modular kernel is reached, so the
megakernel's `dest = eid / E` is already indexing a physical slot. The entire
patch is (a) deleting one refusal, (b) four new selection gates, and (c) a
post-rearrangement re-attestation that restores a safety check which graph
replay silently disarms.

**One correction to the brief:** placement-only EPLB is *not* zero-memory.
`EplbState.add_model` allocates a one-layer transfer buffer regardless of
redundancy — **≈1.31 GiB per rank** at our shapes. See §7.

Everything below cites the mirror at
`.../scratchpad/mirror/vllm_patched/` (paths written relative to that root)
and `.../scratchpad/mirror/pf4h_integration/`.

---

## 1. Where the logical→physical mapping happens

`model_executor/layers/fused_moe/router/base_router.py:286-305`:

```python
    # Step 2: Compute routing (delegated to subclass)
    topk_weights, topk_ids = self._compute_routing(...)      # LOGICAL ids

    # Capture logical ids before EPLB mapping.
    if self.capture_fn is not None:
        self.capture_fn(topk_ids)

    # Step 3: Apply EPLB mapping
    topk_ids = self._apply_eplb_mapping(topk_ids)            # -> PHYSICAL ids

    # Step 4: Convert indices dtype
    topk_ids = self._convert_indices_dtype(topk_ids, topk_indices_dtype)
    return topk_weights, topk_ids
```

`_apply_eplb_mapping` (`base_router.py:204-223`) is a no-op when
`eplb_state is None` and otherwise calls the fused Triton kernel
`_eplb_map_and_record_i32_kernel` (`base_router.py:18-93`), which does mapping
**and** load recording in one launch.

The call chain to our shim is unbroken and contains no second mapping stage:

- `runner/moe_runner.py:573` — `topk_weights, topk_ids = self.router.select_experts(...)`
- `runner/moe_runner.py:581-586` — `self.routed_experts.forward_modular(..., topk_ids=topk_ids, ...)`
- → `FusedMoEKernel` → `M15PrepareAndFinalize.prepare(a1, topk_weights, topk_ids, ...)`
  (`pf4h_integration/m15_vllm.py:200-252`)
- → `runtime.bind_dynamic(..., my_ids=int(topk_ids.data_ptr()), ...)`
  (`m15_vllm.py:306-313`), i.e. the mega reads that exact buffer.

**Consequence.** With EPLB on, `topk_ids` reaching `prepare` are physical slot
ids in `[0, 256)`. `dest = eid / 32` selects the owning rank and `eid % 32` the
local slot — which is precisely what the linear physical partition means. The
mega inherits EPLB's balancing with zero shim work. *The gather described in
the brief does not need to be written.*

Corollary for the tooling: `mirror/m15_router_skew.py` wraps
`FusedMoERouter.select_experts` (the public wrapper,
`router/fused_moe_router.py:45-67`), i.e. it observes **post-mapping physical**
ids. With EPLB off (how the 51.9%/rank-0 figure was measured) physical ==
logical, so the existing histograms are valid logical-load input for §8's
predictor. With EPLB on, the same hook measures the *residual* skew — which is
exactly the after-metric we want.

## 2. Physical layout is unchanged by rearrangement

`ExpertMapManager._calculate_expert_maps` (`fused_moe/expert_map_manager.py:440-454`)
calls `determine_expert_map`, whose linear branch is
(`expert_map_manager.py:75-79`):

```python
    if expert_placement_strategy == "linear":
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx : start_idx + local_num_experts] = torch.arange(...)
```

with `global_num_experts = num_experts + num_redundant_experts`
(`fused_moe/layer.py:79`) = 256 + 0 = 256. So each rank owns physical slots
`[rank*32, rank*32+32)` — the exact contiguous EP8 map
`PF4HPrepareAndFinalize._attest_expert_map` proves at warmup
(`pf4h_integration/vllm_full.py:296-343`).

Two facts make this robust rather than coincidental:

1. `determine_expert_placement_strategy` (`expert_map_manager.py:116-149`)
   **forces `linear`** whenever `enable_eplb` is true. EPLB cannot put us on
   round-robin.
2. Nothing in `distributed/eplb/` ever touches `ExpertMapManager`. The
   expert_map is built once at layer construction and is pointer-stable for the
   process lifetime, so `_attest_expert_map`'s pointer latch
   (`vllm_full.py:308-316`) continues to hold across rearrangements with **no
   re-attestation needed**.

## 3. The map tensors are device-resident and pointer-stable

Allocation, once, in `EplbState.add_model` (`distributed/eplb/eplb_state.py:381-390`):

```python
    logical_to_physical_map = torch.full(
        (model.num_logical_experts, max_slots_per_logical_expert), -1, device=self.device)
    logical_replica_count = torch.zeros(
        (model.num_logical_experts,), device=self.device, dtype=torch.long)
```

then `.unsqueeze(0).expand(num_moe_layers, ...).contiguous()`
(`eplb_state.py:406-422`) → shapes `(58, 256, 1024)` and `(58, 256)`.

Commit, in place, `_commit_eplb_maps` (`eplb_state.py:1224-1258`):

```python
    dst.copy_(src, non_blocking=True)                     # physical_to_logical_map
    _pad_out_tensor(src=new_logical, dst=model_state.logical_to_physical_map)
    dst.copy_(src, non_blocking=True)                     # logical_replica_count
```

and `_pad_out_tensor` (`eplb_state.py:1180-1184`) is itself a `dst.copy_`.
Per-layer views are taken once (`EplbLayerState.set_layer_state`,
`eplb_state.py:1052-1061`):

```python
    self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
    self.logical_replica_count = logical_replica_count[moe_layer_idx]
```

Slicing dim-0 of a contiguous tensor yields a contiguous view, so the
`.contiguous()` calls in `_eplb_map_and_record_triton`
(`base_router.py:113-114`) are identity — no hidden copy is baked into the
graph.

**The one rebind that exists** is `eplb_state.py:1242-1245`:

```python
    if src.shape[1] != dst.shape[1]:
        model_state.physical_to_logical_map = src.to(dst.device)   # REBIND
    else:
        dst.copy_(src, non_blocking=True)
```

This fires only when the *physical expert count changes*, i.e. elastic EP.
`logical_to_physical_map` is never rebound. We hard-refuse elastic EP
(`M15-EPLB-007`) and re-check both pointers after every rearrangement anyway.

## 4. Graph capture vs. in-place map mutation — safe, with the mechanism

The mega runs inside a replayed graph, so the question is whether the router's
mapping reflects post-rearrangement content at replay time. It does, for a
reason worth stating precisely:

- The mapping is done by a **Triton kernel captured into the graph**, reading
  `logical_to_physical_map` / `logical_replica_count` **by pointer**
  (`base_router.py:111-126`). Replay re-executes that kernel against whatever
  bytes those addresses hold. Content updates are picked up automatically.
- Even the hypothetical `.contiguous()` copy would be safe: the copy is a
  *kernel*, and it would be captured too, so it would re-run each replay
  against fresh source bytes. The unsafe pattern is a **host-side** read
  (`.item()`, `.tolist()`, a Python branch on device data) inside the captured
  region. There is none in this path.
- vLLM 0.25.1's EPLB is visibly designed for replay: `should_record_tensor` is
  a device scalar updated out of capture with `.fill_()`
  (`eplb_state.py:682-687`) and read on-device as `record_enabled_ptr`
  (`base_router.py:78`); `num_unpadded_tokens_tensors` are pre-allocated
  precisely "so that device pointers remain stable across CUDA-graph replays"
  (`eplb_state.py:210-215`).
- The shim's own capture contract is unaffected. `M15GraphBody.capture_body`
  takes no tensor arguments (`pf4h_integration/m15_graph_body.py:62-63`); the
  mega's addresses live in a device descriptor written out of capture by
  `commit_captured_descriptors` (`m15_runtime.py:1117-1155`, called from
  `v1/worker/gpu_model_runner.py:6868-6885`). EPLB adds one Triton node to the
  captured graph and changes nothing about the descriptor.
- `topk_ids` after mapping is a fresh allocation
  (`out_flat = torch.empty(...)`, `base_router.py:108`) — during capture it
  comes from the graph's private pool, so its address is fixed and identical on
  every replay, which is exactly what `bind_dynamic`'s capture-time latch
  (`m15_vllm.py:305-313`, `m15_runtime.py:1053-1115`) requires. This is the same
  mechanism the current EPLB-off deployment already relies on.

## 5. Weight identity — EPLB writes into the live tensors

`MixtureOfExperts.set_eplb_state` (`model_executor/models/interfaces.py:908-916`)
collects `expert_weights` from `layer.get_expert_weights()`, which returns
(`fused_moe/routed_experts.py:1135-1141`):

```python
        return [
            weight.view(self.local_num_experts, -1)
            for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS and ...
        ]
```

i.e. **views of the live `torch.nn.Parameter`s**, not copies. Rearrangement
writes through those views (`distributed/eplb/rebalance_execute.py`):

```python
    386:  w[dst].copy_(b[dst], non_blocking=True)     # from the transfer buffer
    424:  w[dst].copy_(w[src], non_blocking=True)     # intra-rank slot move
```

The shim binds base addresses of the same parameters:
`M15Experts.apply` → `runtime.bind_layer_weights(record, int(w1.data_ptr()),
int(w1_scale.data_ptr()), int(w2.data_ptr()), int(w2_scale.data_ptr()))`
(`m15_vllm.py:434-446`), where `w1/w2` are the arguments vLLM hands
`AiterExperts.apply` — the layer's live shuffled parameters. So an in-place
rearrangement is **transparently visible** to the mega with no rebinding.

Two supporting details:

- `_validate_weight_layout` (`m15_vllm.py:100-150`) already proves the tensors
  are expert-major and contiguous with shapes `(32, 4096, 7168)`,
  `(32, 7168, 2048)` and the two scale tensors. `weight.view(32, -1)` therefore
  slices exactly one expert per row, so EPLB's per-slot moves are
  layout-correct. The AITER shuffle is intra-expert, so moving whole slabs
  preserves it.
- `_maybe_make_contiguous` (`routed_experts.py:1070-1108`) can hand EPLB a
  *different* `Parameter` object wrapping the same storage
  (`torch.transpose(p.data, 1, 2)`) for transposed scale tensors. Same
  `data_ptr`, so still fine — and our block-FP8 scales are contiguous, so it
  returns `p` unchanged.

**The gap graph replay opens.** `bind_layer_weights` refuses a changed pointer
(`M15-WEIGHT-002`, `m15_runtime.py:1023-1031`) — but it only runs when Python
runs. Under steady-state replay, `M15Experts.apply` is never called, so that
gate is dormant and a hypothetical rebind would go undetected until the next
eager step. §6's re-attestation closes it.

## 6. What the patch actually does

Env: **`VLLM_PF4H_M15_EPLB`**, exactly `0`/`1`, default `0`. With it unset,
`PF4H-FULL-018` stands verbatim and behaviour is byte-identical.

| # | File | Change |
|---|------|--------|
| 1 | `m15_contracts.py` | `EPLB_COMPOSE_ENV` + `eplb_compose_enabled()`, in the style of `ring_slots()`/`allocate_part()`. |
| 2 | `vllm_full.py` | `PF4H-FULL-018` becomes conditional; when the compose is on, `_eplb_compose_gate` (lazy import, so a `full`-mode install without the m15 file set still imports) delegates to `validate_eplb_compose`. |
| 3 | `m15_eplb.py` **(new)** | Selection gates, placement proof, post-rearrangement re-attestation, hook install. |
| 4 | `m15_vllm.py` | `post_init_setup`: `validate_runtime_compose(runtime)` + `install_eplb_hooks()`. `prepare`: `require_attested()`. `M15Experts.apply`: `register_layer_weights(...)`. |

No vLLM patch is required. Hook placement rides the load order:
`process_weights_after_loading` (hence `post_init_setup`) runs inside
`model_loader.load_model`, which `GPUModelRunner.load_model` calls **before**
`EplbState.add_model` (`v1/worker/gpu_model_runner.py:5308-5391`), so wrapping
`add_model` from `post_init_setup` is always in time to prove the initial map.

### Selection gates (all refuse, none warn)

| Code | Refuses | Why |
|---|---|---|
| `M15-EPLB-001` | compose env set without `--enable-eplb` | misconfiguration must be loud |
| `M15-EPLB-003` | `num_redundant_experts != 0` | breaks `dest = eid/32` and re-arms the replica hash |
| `M15-EPLB-004` | physical count != 256 | ditto |
| `M15-EPLB-006` | `eplb_config.use_async` | see F3; **async is the vLLM default** (`config/parallel.py:82`) |
| `M15-EPLB-007` | `enable_elastic_ep` | the one path that rebinds maps and changes the per-rank count |
| `M15-EPLB-008/009` | M18 sets / M20 budget active | see F5 |

### The placement proof (`_prove_permutation`)

Run at `add_model` and again after every rearrangement, out of capture:

- `logical_to_physical_map` is `(L, 256, ≥1)`, `logical_replica_count` is `(L, 256)`.
- every replica count is exactly `1` → the router's
  `replica_idx = hashed % replica_count` (`base_router.py:50-55`) collapses to
  slot 0, so the mapping is deterministic and token-index-independent.
- `logical_to_physical_map[:, :, 0]` is a **permutation of `[0, 256)`** per layer.

That last one is the load-bearing safety check, not a formality — see F6.

### Re-attestation after rearrangement

`EplbState.step` is wrapped; a rearrangement is detected by the counter
resetting (`eplb_state.py:655-656`). The sweep re-checks the two map pointers,
re-runs the permutation proof, and re-reads `data_ptr()` on the four retained
weight tensors per layer. Any drift raises `PF4HActivationError` and kills the
worker — the alternative is a replayed graph reading a stale descriptor, which
is silently wrong on every subsequent token.

Cost: one D2H sync per rearrangement (default every 3000 steps). The steady
state adds a dict lookup in `prepare`.

## 7. Failure modes

**F1 — capture vs in-place map mutation.** *Safe*; §4. Residual risk is a
future vLLM moving the map read host-side. The pointer re-check catches a
rebind; a host-side read would not be caught, but would also be a visible
upstream change.

**F2 — a rearrangement racing a replay.** *Safe by placement.* `eplb_step()`
runs after the model forward (`gpu_model_runner.py:4766-4767`), out of capture,
on the main stream, so sync-mode transfers are stream-ordered before the next
replay. Cross-rank: the mega is itself an 8-rank rendezvous (peer `chunk_ready`
polls, the M7.5 slab rendezvous, the M8 epoch words), so no rank can leave the
last MoE layer until all peers have pushed. Rank A's EPLB collective can only
overlap rank B's *post-MoE tail* (attention, sampler), where no mega is in
flight anywhere.

**F3 — async EPLB overlapping the mega's symmetric traffic.** *Refused.*
`start_async_worker` (`distributed/eplb/async_worker.py:31-46`) runs a daemon
thread with its **own CUDA stream** issuing NCCL P2P concurrently with the
forward. That is the exact axis this project measured as fatal (unbounded
remote injection: +211µs; the depth-4 vmcnt throttle: −615µs). Upstream hit the
same wall from the other side and pins `NCCL_MAX_CTAS=8` to stop NCCL starving
a cooperative mega-MoE launch (`distributed/eplb/eplb_utils.py:64-109`); our
mega is a 256-CTA persistent grid with device-side spin loops, so CTAs stolen
by NCCL turn straight into `spin_limit` trips and `pperr`. `use_async` defaults
to **True**, so the operator must set it false explicitly.

**F4 — weight-pointer drift invisible under replay.** Closed by
`register_layer_weights` + the post-rearrangement sweep (§5, §6).

**F5 — M18/M20 replica pools go stale.** *Refused.* M18/M20 stage **copies** of
expert weights into an extended allocation (`m15_runtime.py:1044-1051`,
`replica_pool_pointers`). EPLB rewrites only the live base slabs, so every
replica slot would serve pre-rearrangement weights — silent numeric corruption,
no protocol error. Rebuilding pools per rearrangement is possible but is a
different project.

**F6 — a `-1` physical id.** `base_router.py:57-62` loads
`logical_to_physical_ptr[...]` with `other=-1`, and unfilled map slots hold
`-1`. Reaching the kernel with `eid = -1` gives `dest = -1/32 == 0` and a
negative local slot — an out-of-bounds write into rank 0's symmetric heap, with
no guard anywhere in the mega. With redundancy 0 and a proven permutation this
is unreachable, which is why `M15-EPLB-016/017` are hard refusals rather than
logs. **Recommended follow-up ratchet:** a `K0P6_M15_EID_GUARD` (default 0) in
`k0pf6gm_device_tile_m15.hip`'s M1 qpush that clamps/`pperr`s on
`eid < 0 || eid >= world*E`. Cheap (one compare per routed slot, in a
bandwidth-bound phase), and it converts the worst failure mode from silent
corruption into a protocol error.

**F7 — memory, contradicting the "zero-cost" framing.** `EplbState.add_model`
allocates `expert_buffer = [torch.empty_like(w) for w in model.expert_weights[0]]`
(`eplb_state.py:464`) — one layer's worth, regardless of redundancy:

| tensor | shape | bytes |
|---|---|---|
| `w13` | `(32, 4096, 7168)` fp8 | 896 MiB |
| `w2` | `(32, 7168, 2048)` fp8 | 448 MiB |
| `w13_scale` | `(32, 32, 56)` f32 | 224 KiB |
| `w2_scale` | `(32, 56, 16)` f32 | 112 KiB |
| **total** | | **≈1.31 GiB / rank** |

Plus `expert_load_window` `(window_size, 58, 256)` int32 = 56.6 MiB at the
default `window_size=1000`, plus a transient profile-run spike of another
buffer set when `communicator.needs_profile_buffer_reservation`
(`rebalance_execute.py:570-588`). Budget **1.4–2.8 GiB/rank**. This collides
directly with the M20 12-layer/8 GiB budget pair and with the single-ring-slot
decision (`ring_slots()` defaults to 1 because a second slot's ~1.6 GiB
exhausts MI350X headroom at `--gpu-memory-utilization 0.70`). Do not stack
EPLB0 on top of M20 in the same server without re-doing the budget — and note
F5 refuses that combination anyway.

**F8 — recording overhead.** The mapping kernel also does
`tl.atomic_add(out_ptr + safe_physical_id, 1, ...)` over `4096*8 = 32768` slots
into a 256-entry buffer, per MoE layer, on every step where recording is
enabled — ~1.9 M contended atomics per forward across 58 layers. Recording is
on for the last `window_size` steps of each interval
(`_should_record_current_step`, `eplb_state.py:660-680`), i.e. **33 % of steps**
at the defaults (`window_size=1000`, `step_interval=3000`). Set
`window_size=128` to push that to ~4 %. Leave `log_balancedness=false` for
timed runs (it adds a per-step all-reduce).

**F9 — `get_expert_weights` asserts contiguity** (`routed_experts.py:1128-1133`)
across every non-excluded parameter on `RoutedExperts`. If the AITER shuffle
leaves any parameter non-contiguous, `add_model` asserts at load — a loud,
early failure, but one to expect on the first boot.

**F10 — tooling semantics shift.** `M15_SKEW_REP_SET` (the M18 hot-expert id
list) and any recorded route capture are in *physical slot* space. Under EPLB
those ids move every rearrangement, so replay harnesses keyed on them become
meaningless. Freeze the map (`step_interval` huge) for any replay comparison.

## 8. Falsify it before spending a node run

`m15_eplb0/eplb0_predict.py` runs **vLLM's own** `DefaultEplbPolicy` over the
router-skew histograms the sitecustomize hook already writes, and reports the
per-rank load before/after. This needs no GPU.

Why it is meaningful rather than a toy: on one 8-GPU box `get_node_count()`
is 1, so `rebalance_experts` takes the hierarchical branch with `num_nodes=1`
(`policy/default.py:309-313`). Step 1 (group→node packing) is then a no-op, and
step 3 is `balanced_packing(tokens_per_phy, 8)` — an **unconstrained** LPT
packing of all 256 experts into 8 bins of exactly 32
(`policy/default.py:167-174`, `policy/default.py:22-71`). The expert-group
constraint that would otherwise limit rearrangement only bites at the *node*
level, and we have one node. So placement-only EPLB really can flatten a
stationary rank-0 concentration.

Read the output as an **upper bound**: the policy fits the *window mean*, so
per-call variance is untouched. The same skew JSON carries
`percall.max_rank_before_hist_0p1_bins`, which bounds how much of the
aggregate relief is actually reachable per call. If the aggregate says 4.1× →
1.05× but the per-call histogram is broad, expect much less.

Gate: if the predictor says the worst-layer critical-rank relief is under
~15 %, do not run the arm.

## 9. Accuracy attestation protocol

**Protocol A — route-frozen equivalence (the receipt).** Isolates "does the
mega compute correctly on a permuted placement" from "does EPLB pick the same
placement".

1. Arm S (stock Mori + EPLB0, `VLLM_PF4H_INTEGRATION_MODE` unset). Run the
   fixed B4096 prompt set past one `step_interval`; dump
   `eplb_state.model_states[k].physical_to_logical_map.cpu()` from rank 0
   (a five-line sitecustomize in the style of `m15_router_skew.py`).
2. Restart **both** arms with `step_interval` = 10**9 (never rearranges) and
   seed that exact map through `GPUModelRunner.setup_eplb_from_mapping`
   (`gpu_model_runner.py:3354-3362` → `EplbState.from_mapping`,
   `eplb_state.py:984-1025`). Both arms now hold byte-identical maps.
3. Compare MoE outputs on the frozen prompt set against the existing PF4H
   tolerances: `max_abs=0.02`, `max_rel_l2=0.01`, `max_nonfinite=0`
   (`pf4h_integration/contracts.py:65-67`). This is the current correctness
   gate with EPLB held constant.

**Protocol B — permutation invariance (cheap sanity).** Same server,
`step_interval` = 10**9. Seed arm 1 with the identity map and arm 2 with a
non-trivial permutation; same prompts. Outputs must agree within the *same*
bf16-RMW tolerance, because a slot permutation changes only the accumulation
**order** of the same eight addends in the combine — and that order is already
nondeterministic under `kittens::distributed::accumulate_peer_bf162`. A failure
here means weights did not follow their slot, i.e. a real bug.

**Protocol C — live.** Rearrangement enabled. Assert
`M15_EPLB_PLACEMENT_RECEIPT` at load and no `PF4HActivationError` afterwards;
sweep `poll_protocol_error()` for `pperr`; confirm the skew hook's *post*-EPLB
histogram flattens toward the predictor's projection.

## 10. If the compose is refused on the node

Fallbacks, in order of preference:

1. **Freeze after one fit.** Run EPLB normally for `N > step_interval` steps to
   learn a placement, dump the map, then restart with
   `setup_eplb_from_mapping` + `step_interval = 10**9`. This keeps every
   benefit that a *stationary* skew offers, removes the rearrangement failure
   surface entirely, drops the recording atomics (F8), and makes the run
   reproducible. **Given the measured skew is an aggregate over a whole run,
   this is arguably the better first experiment even if the compose works.**
2. **Eager B4096 path.** `VLLM_PF4H_B4096_GRAPH_TARGET` unset, so the shim
   activates eagerly (`_eligible` → `self._execution = "eager"`,
   `vllm_full.py:408-409`). Python runs every step, so `bind_layer_weights`'s
   pointer gate and `bind_dynamic`'s latch are live and no re-attestation hook
   is needed at all. Costs the graph-launch win, buys full observability —
   the right shape for the *first* live gate.
3. **Capture-time re-freeze on every rearrangement.** Not recommended:
   re-capturing 58 layers' graphs mid-serving is far more disruptive than the
   attestation this design uses, and the evidence in §3–§5 says it is
   unnecessary.

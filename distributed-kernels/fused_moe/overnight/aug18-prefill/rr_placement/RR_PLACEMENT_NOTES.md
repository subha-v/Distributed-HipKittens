# Round-robin expert placement on the MoRI backend — proof, arithmetic, deploy

**Status: built, unit-tested, NOT yet run on the node.**
Deliverables: `rr_patch.py`, `run_m15_campaign_eplb_v3.sh`, `camp4_rr.sh`
(scratchpad), `test_rr_patch.py` (this directory; 140 checks, all green).

The one-line claim: **round-robin placement is not a routing change, it is a
permutation of which logical expert's weights sit in which physical slot.**
Physical ownership stays contiguous rank-major, so MoRI, AITER, the M15
megakernel and the shim's ownership attestation are all the *unmodified linear
path*. Only two decisions move — the weight loader's slot assignment and one
static gather in the router.

---

## 1. Step-1 verification (every claim cited)

Paths are relative to the deployed-code mirror
`.../scratchpad/mirror/vllm_patched/` and `.../scratchpad/mirror/pf4h_integration/`.

### 1.1 What the round-robin routing tables actually emit — INVARIANT HOLDS

`ExpertMapManager._init_round_robin_expert_routing_tables`
(`model_executor/layers/fused_moe/expert_map_manager.py:477-516`):

```python
    owner       = torch.remainder(global_indices, self.ep_size)          # :489
    local_index = torch.div(global_indices, self.ep_size, "floor")       # :490
    base        = self.global_num_experts // self.ep_size                # :492
    remainder   = self.global_num_experts %  self.ep_size                # :493
    physical_offset = owner * base                                       # :494
    if remainder > 0:                                                    # :496
        physical_offset = physical_offset + torch.minimum(owner, remainder)
    global_to_physical = physical_offset + local_index                   # :503
```

At E=256, EP=8: `base = 32`, `remainder = 0`, therefore

> **p(e) = (e % 8) · 32 + e // 8**

which is exactly the required invariant. Executed and checked in
`test_rr_patch.py` §6: p is a bijection on [0,256); rank r owns *exactly*
physical `[32r, 32r+32)`; `p(e) // 32 == e % 8` and `p(e) % 32 == e // 8` for
all 256 e. **Physical ownership under round-robin is contiguous rank-major.**

Note the same builder asserts `num_fused_shared_experts == 0`
(`expert_map_manager.py:481-483`, "Round robin not supported for AITER") — a
second reason not to route through it on an AITER deployment.

### 1.2 Who consumes those tables — THE BLOCKER

The tables are stored at `expert_map_manager.py:233 / 457-475`, surfaced as
buffers at `routed_experts.py:238-245`, and handed to
`maybe_make_prepare_finalize(routing_tables=...)`
(`fused_moe_method_base.py:108-113`, e.g. `quantization/fp8.py:716`). Inside
`all2all_utils.maybe_make_prepare_finalize` they are unpacked in **exactly two
branches**:

* `all2all_utils.py:174-207` → `DeepEPLLPrepareAndFinalize(...)`
* `all2all_utils.py:325-359` → `NixlEPPrepareAndFinalize(...)`

and consumed by `DeepEPLLPrepareAndFinalize._map_global_to_physical_ids`
(`prepare_finalize/deepep_ll.py:155-158`)

```python
    def _map_global_to_physical_ids(self, topk_ids):
        if self.global_to_physical is None:
            return topk_ids
        return self.global_to_physical[topk_ids]
```

called at `deepep_ll.py:291` (dispatch) and `deepep_ll.py:396` (combine).

**`MoriPrepareAndFinalize` (`prepare_finalize/mori.py:52-124`) contains no
translation of any kind.** `prepare` feeds `topk_ids` straight into
`self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)` (`mori.py:95`) and
`finalize` into `self.mori_op.combine(..., topk_ids)` (`mori.py:119-123`). The
mori handle is constructed with `num_local_experts = num_experts //
all2all_manager.world_size` (`all2all_utils.py:265`), i.e. the wire derives the
destination rank as `id // 32` — the same convention the M15 megakernel uses
(`dest = eid / 32`, proven by `_attest_expert_map`,
`pf4h_integration/vllm_full.py:295-340`).

The router does **not** cover for this either. `BaseRouter._apply_eplb_mapping`
(`router/base_router.py:204-223`) returns `topk_ids` unchanged whenever
`self.eplb_state is None`. The "the router already emits physical ids" property
that `M15_EPLB0_COMPOSE.md` §1 proves is an **EPLB-only** property; it does not
hold for the `round_robin` placement path.

> **Verdict.** The physical numbering is correct, but merely widening
> `determine_expert_placement_strategy` (`expert_map_manager.py:137-147`) and
> `needs_round_robin_routing_tables` (`config.py:1073-1074`) would place the
> **weights** round-robin while leaving **logical** ids on the wire: every
> token for logical expert e would be dispatched to rank `e // 32` while e's
> weights live on rank `e % 8`. Silently wrong output, no exception. The task's
> anticipated alternative — *build the static map ourselves at the router seam*
> — is therefore what is implemented.

### 1.3 The weight loader under an RR map

`determine_expert_map`'s round-robin branch (`expert_map_manager.py:80-87`)
sets `expert_map[e] = e // ep_size` for `e ≡ ep_rank (mod ep_size)`, `-1`
otherwise. That map reaches loading through exactly one call path:

`RoutedExperts.weight_loader` (`routed_experts.py:585-608`)
→ `_map_global_expert_id_to_local_expert_id` (`routed_experts.py:276-278`)
→ `ExpertMapManager.map_global_to_local` (`expert_map_manager.py:336-352`).

So **expert e is placed at local slot `e // 8` on rank `e % 8`** — precisely
physical slot p(e). `is_local_expert` (`:354`) and `get_local_expert_ids`
(`:360`) have **zero in-tree callers** (verified by grep across the whole
mirror), so `map_global_to_local` is the *only* load-time consumer.

Separately, the checkpoint **disk filter** `compute_local_expert_ids`
(`model_loader/ep_weight_filter.py:31-61`, round-robin branch at `:58-59`) is
called with the **raw** `parallel_config.expert_placement_strategy`
(`model_loader/default_loader.py:400-403`) — i.e. it filters round-robin even
when the layer silently downgraded to linear.

> **Pre-existing hazard found (worth reporting on its own).** On stock vLLM
> 0.25.1 + MoRI, `--expert-placement-strategy round_robin` makes each rank
> fetch `{r, r+8, …}` from disk while its `expert_map` expects `[32r, 32r+32)`.
> The rows for the experts it did not fetch are never written, `weight_loader`
> is simply never called for them, and nothing raises — the server serves
> uninitialised expert weights. `rr_patch.py` edit E4 turns that combination
> into a hard `ValueError`.

### 1.4 What the MoRI/AITER path assumes about expert ids

`grep -rn expert_map` over the mori/aiter path:

* `MoriPrepareAndFinalize.prepare` takes an `expert_map` parameter
  (`mori.py:58`) and **never reads it**.
* `RoutedExperts.expert_map` (`routed_experts.py:226-230`) returns
  `self._expert_map` normally and `self.expert_mask` when ROCm AITER
  fused-MoE is enabled.
* `rocm_aiter_fused_experts` (`experts/rocm_aiter_moe.py:234-272`) uses it as
  `expert_mask = expert_map` (`:272`); AITER's moe_sorting derives the local
  slot as the prefix-sum of the mask, so the mask must be indexed by whatever
  id space the tokens carry.

That is the crux: an id space and a mask must agree. Under **linear**
placement the two coincide (logical == physical), which is why the deployment
works today and why `_attest_expert_map` can prove ownership from either the
map or the mask (`vllm_full.py:335-341`).

---

## 2. The implemented design — "RR by permutation"

Keep every runtime tensor **linear** and realise round-robin as a permutation:

| stage | linear deployment (today) | RR by permutation |
|---|---|---|
| checkpoint disk filter | `[32r, 32r+32)` | `{r, r+8, …}` (serve flag) |
| loader slot for logical e | `e − 32r` | `e // 8` on rank `e % 8`  ← **E2** |
| `expert_map` / `expert_mask` | linear | **linear (unchanged)** |
| router output ids | logical == physical | `p(e)` ← **E3** |
| MoRI dispatch rank | `id // 32` | `id // 32` (unchanged) |
| AITER local slot | `id − 32r` | `id − 32r` (unchanged) |
| megakernel `dest = eid / 32` | correct | correct (unchanged) |
| shim ownership attestation | linear map/mask | linear map/mask (unchanged) |

Five edits, marker `PF4H_RR_PLACEMENT_V1`, gate `VLLM_PF4H_RR_PLACEMENT`
(exact `0`/`1`, default `0`):

| id | file | change |
|---|---|---|
| E1 | `fused_moe/expert_map_manager.py` | RR state, refusals, cached p(e) gather table (256+64 entries, identity tail so fused shared-expert ids pass through in ONE kernel), `PF4H_RR_PLACEMENT_RECEIPT` |
| E2 | `fused_moe/expert_map_manager.py` | `map_global_to_local` → `e // ep` on rank `e % ep`, `-1` elsewhere. **Load-time only** (single caller, §1.3) |
| E3 | `fused_moe/router/base_router.py` | `topk_ids = _pf4h_rr_to_physical(topk_ids)` inserted at `_select_experts` **after** `capture_fn` (`base_router.py:296-297`, so route-capture tooling still records LOGICAL ids) and **before** `_apply_eplb_mapping` |
| E4 | `model_loader/ep_weight_filter.py` | hard refusal of `round_robin` without the gate (§1.3 hazard) |
| E5 | shim `vllm_full.py` | blocker `PF4H-FULL-021` (RR + EPLB / redundant experts) and the RR-aware `_attest_expert_map` |

**Neither placement guard is widened.** The layer's effective
`placement_strategy` stays `"linear"`, `routing_tables` stay `None`, and the
AITER assert at `expert_map_manager.py:481` is never reached.

### Why the shim attestation *refuses* rather than *accepts* the RR pattern

The brief asked for "acceptance of the static RR map pattern". Step 1 inverts
that polarity, and E5 implements the inverted form deliberately: under this
design the map handed to the megakernel is **still the linear one**, because
the permutation lives in the loader and the router. If the shim ever saw a
*round-robin-indexed* map or mask it would mean the layer really did switch to
RR ownership at runtime, and then `dest = eid / 32` would push tokens at the
wrong peer. So under `VLLM_PF4H_RR_PLACEMENT=1` the patched attestation:

* accepts the linear map/mask and logs `PF4H_RR_ATTEST_RECEIPT … result=ok`;
* **refuses** (returns `False`, logs `result=refused`) on the RR pattern;
* is byte-identical to today's decision when the gate is `0`.

No re-attestation machinery is needed: the map is static, so the pointer latch
(`vllm_full.py:308-316`) holds for the process lifetime exactly as it does
today.

### Cost

One `table[topk_ids]` gather per MoE layer per forward — the same shape and
allocation pattern EPLB's `out_flat` already has (`base_router.py:108`), so the
CUDA-graph reasoning in `M15_EPLB0_COMPOSE.md` §4 carries over verbatim. The
table is built eagerly in `pf4h_rr_configure` (outside capture) and a lazy
build during capture is refused. **Zero extra memory** — unlike placement-only
EPLB, which still allocates ≈1.31 GiB/rank of transfer buffer
(`M15_EPLB0_COMPOSE.md` §7).

---

## 3. Expected skew arithmetic

Source: the deployed 256-bin aggregate histogram
`mirror/mok_synthetic_prefill/route_hists/stock0814_aggregate.json`
(58 layers, 2.332 × 10⁹ routed slots, gini 0.609), replayed under both
placements.

**Hot set.** Logical experts 0-7 carry **51.93 %** of routed slots, each
6.40 – 6.61 %. Under linear placement all eight land on rank 0. Under
round-robin, expert e goes to rank `e % 8` → exactly one hot expert per rank.

**Aggregate receive share per rank** (share of all routed slots whose owner is
that rank):

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | max/mean | max/min |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | **63.60 %** | 6.93 | 5.46 | 5.29 | 4.74 | 4.59 | 4.77 | 4.61 | **5.088×** | 13.84× |
| round-robin | 13.56 | 13.33 | 12.27 | 12.08 | 12.61 | 11.86 | 12.08 | 12.22 | **1.085×** | 1.143× |

Per-layer worst cases from the same dump:

| layer variant | linear max/mean | round-robin max/mean |
|---|---|---|
| worst-rank layer (gini 0.739) | 5.593× | **1.296×** |
| worst-gini layer (gini 0.829) | 4.160× | **1.157×** |

So RR removes essentially all of the *aggregate* imbalance, for free, with no
replicas, no rearrangement and no extra memory.

**What it does NOT fix — say this out loud before reading campaign 4.** The
m18diag per-call instrument (n = 90,050 calls,
`../../aug14/M18_REPLICATION_RESULTS.md:138-141`) measured **per-call max rank
load p50 = 5.15×, p95 = 6.25×**, and — the load-bearing sentence —

> "even **ideal static redistribution leaves p95 at 3.05×**"

Round-robin *is* a static redistribution, and a weaker one than "ideal": it is
chosen from the aggregate histogram and knows nothing about which experts are
hot in any individual chunk. Therefore

* aggregate skew: **fully addressed** (5.09× → 1.085×),
* per-call p50: partially addressed,
* per-call p95: **bounded below by ≈3.05×** — RR cannot beat the ideal static
  bound, and chunk wall-time is convex in skew, so the tail is where the
  remaining cost lives.

That is why M19's per-chunk adaptive routing (+39.8 % vs stock,
`M18_REPLICATION_RESULTS.md`) beat static replication (+11.3 %) so decisively,
and why campaign 4 is a *measurement*, not a bet.

---

## 4. Deploy checklist

1. Copy `rr_patch.py`, `run_m15_campaign_eplb_v3.sh`, `camp4_rr.sh` to
   `/home/subvadla/eplb_campaign/` on the node. SHA-256 of what was built here:

   ```
   7ac9ee696e4e9d7dbc9866fe59780cf37add78a18ea53dacef1d839b8dd1d48a  rr_patch.py
   378b0df16f89be58a9f7267db68fedfa5287e3b9044645ecae08f2b3c78306af  run_m15_campaign_eplb_v3.sh
   4cac8c3277bce2385c8dad4ca9a38bf566875e04ac8726433c2358f09b070563  camp4_rr.sh
   ```

2. Re-run `python3 test_rr_patch.py` on the node's mirror if the deployed vLLM
   is not byte-identical to `.../scratchpad/mirror/vllm_patched/`. Any anchor
   drift is fatal, by design, and prints the anchor name.
3. Confirm the chain in the container is
   `apply.py → coverage_patch.py → m23_patch.py → rr_patch.py`. The v3 driver
   emits it in that order; `rr_patch` also refuses to run before
   `coverage_patch` on a PF4H image.
4. Create `~/eplb_campaign/RR_READY` only after steps 2-3, then start
   `camp4_rr.sh` (it also waits for `CAMP3_RC=` in `camp3_20260818.log`).
5. **Startup receipts to grep before trusting any number.** Per rank, in
   `server.log`:
   * `PF4H_RR_PLACEMENT_RECEIPT rank=<r> ep_size=8 experts=256 local=32
     ownership=contiguous_rank_major …` — 8 occurrences on every `_rr` arm,
     **zero** on the non-RR arms.
   * `EP weight filter: ep_size=8, ep_rank=<r>, loading 32/256 experts`
     (`default_loader.py:405-412`) — present on both, but the *set* differs.
   * m15_rr only: `PF4H_RR_ATTEST_RECEIPT rank=<r> result=ok
     ownership=contiguous_rank_major`, plus the ordinary M15 receipt list
     (`required_receipts` routes `m15_rr → m15`).
   * Any `PF4H_RR_ATTEST_RECEIPT … result=refused` **voids the cell** — it
     means the layer published round-robin-indexed ownership, which this design
     never intends.
6. **Correctness gate before believing any throughput number.** A placement
   permutation is exactly the kind of change that can be fast and wrong. Run
   the harness reference-match / a short AccuracyOnly A/B on `stock_rr` before
   quoting campaign 4A. If output differs from `stock`, the permutation is
   mis-wired — not a "small numerical difference".
7. **Two log lines that look alarming but are correct.**
   * `Round-robin expert placement currently only supports the DeepEP
     low-latency or NIXL EP backend, but 'mori_high_throughput' was
     configured. Falling back to linear expert placement.`
     (`expert_map_manager.py:141-146`) — **expected**. This design wants the
     layer's effective strategy to stay `linear`; the permutation is applied
     in the loader and the router. Its presence is a *positive* signal that
     neither guard was widened.
   * `[EP Rank r/8] Expert parallelism is enabled. Expert placement strategy:
     linear.` (`expert_map_manager.py:245-256`) — same reason. The authoritative
     RR signal is `PF4H_RR_PLACEMENT_RECEIPT`, not this line.
8. Rollback is `unset VLLM_PF4H_RR_PLACEMENT` and drop
   `--expert-placement-strategy` (i.e. run a non-`_rr` arm). The patched bytes
   stay in place and are decision-inert.

---

## 5. Campaign 4 decision table

**4A `rrcal1` — stock vs stock_rr** (production Mori+AITER both sides, zero
shim involvement; the pure value of placement).
**4B `rrfair1` — stock_rr vs m15_rr** (both arms RR-balanced; the megakernel
measured against a *balanced* baseline instead of inheriting imbalance).

| 4A: stock_rr vs stock | 4B: m15_rr vs stock_rr | reading | next move |
|---|---|---|---|
| ≥ +5 % tok/s | ≥ +10 % | Aggregate skew was real cost **and** the megakernel wins on top of a balanced baseline. Strongest possible outcome. | Ship RR as the default placement for the MoRI arms; re-baseline every M15/M18/M19 number against `stock_rr`. |
| ≥ +5 % | ≈ 0 % / negative | The megakernel's earlier wins were **substantially imbalance arbitrage**. RR captures the same value with no kernel. | Honest re-framing: report the M15 delta against `stock_rr`, not `stock`. Redirect effort to what RR cannot fix — the per-call p95 (M19-style per-chunk adaptivity). |
| ≈ 0 % (±2 %) | ≥ +10 % | Aggregate balance is **not** the binding constraint — consistent with m18diag: the cost lives in the per-call tail (p95 ≥ 3.05× even ideal). The megakernel wins for other reasons (fusion, wire). | Keep RR as a free hygiene default (it costs one gather and no memory), but stop treating balance as the lever. Prioritise per-chunk adaptive work. |
| ≈ 0 % | ≈ 0 % | Neither placement nor the kernel moves this cell. Suspect the cell is not skew-bound at c32p, or coverage is low. | Check `M15_COVERAGE` (b4096_steps ≈ in_bucket) before concluding anything; if coverage is fine, move the question to a larger/longer cell. |
| **negative** | any | RR made it *worse*. Two candidate causes, both checkable: (a) the extra router gather is not free at this layer count, (b) RR broke locality that MoRI's HT path exploits (destination-rank fan-out per token rises from "hot rank" to "all ranks"), turning a few large transfers into many small ones. | Grep the skew pass for per-rank *transfer counts*, not just token counts; if (b), that is a genuine and publishable finding about HT all2all, not a bug. |
| any | `result=refused` / missing receipts | Wiring fault, not a result. | Void the pair, re-check §4 step 5. |

**Two confounds to keep on the table for both campaigns.** (i) n = 2 pairs is
below the ≥ 5 order-balanced pairs this rig has repeatedly needed — treat 4A/4B
as direction, not magnitude. (ii) Node-state drift across a day was ~30 % on
2026-08-14; the paired AB/BA design absorbs it, but do not compare 4A's
`stock_rr` number with 4B's `stock_rr` number across campaigns.

---

## 6. Files

| path | what |
|---|---|
| `.../scratchpad/rr_patch.py` | the patcher (marker `PF4H_RR_PLACEMENT_V1`) |
| `.../scratchpad/run_m15_campaign_eplb_v3.sh` | v2 + the `_rr` arms; byte-compatible for every non-`_rr` arm |
| `.../scratchpad/camp4_rr.sh` | the waiter: 4A `rrcal1` then 4B `rrfair1` |
| `./test_rr_patch.py` | 140 CPU-only checks; chains the real coverage + m23 + rr patchers against mirror copies |
| `./RR_PLACEMENT_NOTES.md` | this file |

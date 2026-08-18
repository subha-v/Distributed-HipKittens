# SHEXP — shared-expert filler compute inside the M15 prefill megakernel (2026-08-17)

Design only. No kernel edits in this commit. Everything below is behind
`K0P6_M15_SHEXP` **default 0**; the default build's `.text` must stay
byte-identical (L7, exp_38: one line of unreachable mode-14 code near the M7
epilogue cost +726.9 µs).

Mirror root used for every vLLM/shim citation:
`…/scratchpad/mirror/` (= `vllm_patched/` is the *vllm package root*, i.e.
`vllm_patched/model_executor/...`, not `vllm_patched/vllm/model_executor/...`).

---

## 0. The one-paragraph claim

DeepSeek-R1's single shared expert is, per DP rank, a 3-GEMM MLP over the
**same 4096 local hidden rows the router just consumed** — it owns **zero**
dependency on dispatch, on the plan, on the sort, or on any peer. It is
therefore the *only* piece of same-epoch work in the forward pass that
`aug12/OVERLAP_KERNEL_DESIGN_IDEAS.md` §5's structural argument does not rule
out. That document's conclusion — "same-epoch dispatch×M6 overlap is blocked by
the sort's global histogram dependency; the correct vehicle is cross-epoch" —
is exactly right *for the routed experts* and exactly wrong for the shared one.
Today vLLM runs it **serialized, outside the megakernel, before** the routed
region. SHEXP moves it inside and schedules its GEMM-1 tiles in the M0→M2
dispatch window, where 162 MiB of xGMI push is close to fabric-bound and a
large fraction of the grid's MFMA issue capacity is idle.

---

## 1. Where the shared expert runs today, and what the shim must do

### 1.1 Construction

`model_executor/models/deepseek_v2.py`

| line | what |
|---|---|
| 295 | `self.n_shared_experts = config.n_shared_experts` (= 1 for R1) |
| 347-360 | `self.shared_experts = DeepseekV2MLP(hidden_size=7168, intermediate_size=config.moe_intermediate_size * n_shared_experts = 2048, hidden_act="silu", quant_config=<block-fp8>, reduce_results=False, prefix=f"{prefix}.shared_experts")` — **same shape as one routed expert** |
| 362-388 | `self.experts = FusedMoE(shared_experts=self.shared_experts, …)` — the MLP is *handed to* the FusedMoE layer, not called by the model |
| 378 | `apply_routed_scale_to_output=not self.is_rocm_aiter_moe_enabled` → **False on our deployment**, so `routed_scaling_factor` (2.5) is folded into `topk_weights` by the router, and the runner's own scale is 1.0 |
| 269-273 | `DeepseekV2MLP.forward`: `gate_up_proj(x)` → `SiluAndMul()` → `down_proj(x)`. Three GEMMs + one elementwise kernel. |
| 398-427 | `DeepseekV2MoE.forward` — calls `self.experts(hidden_states, router_logits)` at :417 and **never touches `shared_experts` itself** |

### 1.2 Execution site (the exact line that runs it under PF4H/M15)

`model_executor/layers/fused_moe/layer.py:408` passes `shared_experts=` into
`MoERunner`. `runner/moe_runner.py:277-284` wraps it:

```
self._shared_experts = SharedExperts(shared_experts, …,
                                     mk_can_overlap_shared_experts=
                                       lambda: self._quant_method.mk_can_overlap_shared_experts)
```

`runner/shared_experts.py:89-109` picks an order:

* `MK_INTERNAL_OVERLAPPED` requires `mk_can_overlap_shared_experts`, which is
  `modular_kernel.py:1564-1569` → `prepare_finalize.supports_async()`.
  **`vllm_full.py:229-232` — `PF4HPrepareAndFinalize.supports_async()` returns
  `False`** (and `m15_vllm.py`'s `M15PrepareAndFinalize` inherits it). So this
  arm is dead for us.
* `MULTI_STREAM_OVERLAPPED` requires `current_platform.is_cuda()`
  (`platforms/interface.py:189-190` — **False on ROCm**). Dead.
* ⇒ **`SharedExpertsOrder.NO_OVERLAP`.**

Therefore the shared expert executes at

> **`runner/moe_runner.py:560-562`**
> ```
> self._maybe_apply_shared_experts(shared_experts_input,
>                                  SharedExpertsOrder.NO_OVERLAP)
> ```
> → `moe_runner.py:538-545` → `runner/shared_experts.py:155-174`, whose
> non-overlap branch is **`shared_experts.py:172`: `self._output[self._output_idx] = self._layer(shared_experts_input)`**

— i.e. on the *main* stream, *before* `routed_experts.forward_modular(...)`
(`moe_runner.py:564-586`), which is what calls PF4H's `prepare()` /
`M15Experts.apply()` / `finalize()` and hence the megakernel. **Fully
serialized in front of the mega, every layer, every step.**

Its input is `shared_experts_input`, produced at `moe_runner.py:369-372`:
`hidden_states if self._shared_experts is not None else None` — the *same*
tensor the gate consumed. Exactly the mega's `K0P6_D_HIDDEN` (bound at
`m15_vllm.py:306-312` from `prepare()`'s `a1.data_ptr()`).

### 1.3 What tensor it contributes to

`moe_runner.py:706-723`:

```
shared_output, fused_output = _unpack(result)
shared_output = self._maybe_reduce_shared_expert_output(shared_output)   # :713
shared_output, fused_output = self._maybe_apply_routed_scale_to_output(…) # :715
...
if shared_output is not None:
    result = shared_output + fused_output                                 # :723
```

Three facts that make the in-kernel fold arithmetically exact:

1. `_maybe_reduce_shared_expert_output` (`:418-434`) all-reduces over the **TP**
   group. `contracts.py:53` — `tp_size = 1`. No-op.
2. `_maybe_apply_routed_scale_to_output` (`:392-407`): `self.routed_scaling_factor`
   is 1.0 because `apply_routed_scale_to_output=False` (deepseek_v2.py:378) —
   the 2.5 is already inside `topk_weights`, which the mega already consumes
   through `K0P6_D_MYWG`. No scaling on either side.
3. `:723` is a **plain bf16 add** of two `[4096, 7168]` bf16 tensors. The mega's
   `out` buffer (`K0P6_D_OUT`) *is* `fused_output`. So folding the shared expert
   into `out` is bit-comparable-to-better than what torch does (we accumulate the
   routed side in fp32 and add a bf16 shared term once, vs. torch's two
   already-rounded bf16 tensors).

**Deleting the shared expert from the serving path therefore also deletes the
`:723` add kernel** — 4096×7168 bf16 × (2 reads + 1 write) = 176 MB ≈ 35-40 µs
per layer, free.

### 1.4 What the shim must do

Two monkeypatches, in the existing `pf4h_integration/apply.py` style, both
gated on `K0P6_M15_SHEXP` being compiled into the pinned `.hsaco` (a new
`M15ProcessRuntime.shexp_enabled` flag derived from the descriptor-word count,
same discipline as `m15_runtime.py:355-370`):

**(a) Claim.** Patch `SharedExperts.forward` (`runner/shared_experts.py:155`).
Before doing anything, ask the layer's M15 controller:

```
claimed = _pf4h_shexp_claim(self._layer)     # -> bool
if claimed:
    self._output[self._output_idx] = None    # sentinel: "the mega owns it"
    return
```

**(b) Suppress the add.** Patch `MoERunner._apply_quant_method`
(`moe_runner.py:547-596`) so its return at `:592-595` yields
`(None, fused_out)` when the claim was taken. `shared_output is None` then makes
`moe_runner.py:722-725` take the `result = fused_output` branch — no zero
tensor, no add kernel, no allocation.

**(c) `_pf4h_shexp_claim` — the pre-check and its fail-closed attestation.**
The problem: the shared expert runs at `:560`, *before* `prepare()` runs the
authoritative `_eligible()` predicate (`vllm_full.py:345-409`). So the claim
must use a *pre-check* built from the subset of `_eligible()` that is already
decidable at `:560`:

* `_shape(hidden_states) == (4096, 7168)`, bf16, contiguous;
* `controller._attest_dp_batch()` (`vllm_full.py:262-294`) — the exact-B4096 DP
  descriptor key, already side-effect free;
* `controller.runtime.operator_activation_enabled(layer_id)` latch state
  (read-only variant — do **not** latch from here);
* the capture-state predicate `_capture_in_progress()`.

The parts it *cannot* see (topk shapes/dtypes, `expert_map` attestation,
`quant_config.is_block_quantized`) are all **invariant for the server's
lifetime** once the first eligible layer has run. So:

> **Attestation rule (new blocker `M15-SHEXP-001`, fail-closed):** the claim is
> recorded on the controller as `self._shexp_claimed = True`. `_eligible()`
> then **raises** if it is about to return `False` while `_shexp_claimed` is
> set. A disagreement is a hard activation error, never a silent wrong answer.
> Symmetrically, `finalize()` clears the flag; a `finalize()` that finds the
> flag still set on the non-active path is also `M15-SHEXP-001`.

**(d) Weights.** `M15Experts.apply` (`m15_vllm.py:369-446`) currently binds four
addresses through `runtime.bind_layer_weights` (`m15_runtime.py:997-1034`) after
`_validate_weight_layout` (`m15_vllm.py:100-137`) checks the exact shuffled AITER
shapes `(32, 4096, 7168) / (32, 7168, 2048) / (32, 32, 56) / (32, 56, 16)`.
SHEXP requires a **33-slot** extended allocation with the shared expert
pre-shuffled into slot 32 — *exactly* the M18 replica-slot mechanism
(`m15_runtime.py:482-501`, and the host side at
`prefill_opt_host/e004pf_k0pf_ab.py:3101-3107`, which already swaps
`desc[35..38]` to `_m18_w13/_m18_fc1/_m18_w2c/_m18_fc2`).

The shared expert's live tensors are *not* in AITER-shuffled layout — they are
`MergedColumnParallelLinear.weight [4096, 7168]` + `weight_scale_inv [32, 56]`
and `RowParallelLinear.weight [7168, 2048]` + `[56, 16]`. So at weight-bind time
(once, at load, never in the hot path):

1. locate the sibling module by prefix substitution: the routed layer's name is
   `…mlp.experts`; the shared expert is `…mlp.shared_experts`
   (deepseek_v2.py:359). Resolve through the model's `named_modules()` — do not
   guess an attribute.
2. `aiter.shuffle_weight` the two weight tensors into slot 32 of the extended
   allocation, copy the scale blocks into slot 32 of the two scale tensors.
3. **free the originals** (`shared_experts.gate_up_proj.weight` /
   `down_proj.weight`) — we never call them again. Net HBM delta ≈ 0
   (44 MB/layer either way; 2.5 GB across the 58 MoE layers if you forget to
   free, which is a budget breach, so make the free a receipt-logged step).

**(e) Accuracy attestation.** Three receipts, all reusing the existing
`FROZEN_CONTRACT` tolerances (`contracts.py:65-67`: `max_abs 0.02`,
`max_rel_l2 0.01`, `max_nonfinite 0`):

* **R1 (composite):** the existing PF4H A/B compares the mega's `out` against
  the stock path. With SHEXP the reference simply becomes
  `stock_shared + stock_routed`, which is what `moe_runner.py:723` already
  produces. Same gate, new reference expression. No new fixture.
* **R2 (shexp isolation):** a one-off eager probe with `topk_weights` zeroed →
  the mega's `out` must equal `DeepseekV2MLP.forward(hidden)` to the same
  tolerance. This isolates the shuffle/quant/slot-32 path from the routed path.
* **R3 (null-arm parity):** with `K0P6_M15_SHEXP` compiled in but the claim
  disabled (`VLLM_M15_SHEXP=0`), the shim must be a no-op and the `.hsaco`
  `.text` sha must equal the non-SHEXP build's. This is the L7 gate expressed
  at the serving layer.

---

## 2. Kernel-side design

### 2.1 Why the naive shapes of this idea are all dead

Read these first; each one is ruled out by something already measured.

| shape | why dead |
|---|---|
| a second inlined call to `n2p6m15_phase1_body` at an earlier program point | `N2_P1_QUAL` is `__device__ __forceinline__` (`k0pf6gm_device_tile_m15.hip:392`). A second call site duplicates the whole body **and unions its live ranges into one whole-function allocation** → the 256-VGPR/1-block-per-CU occupancy law and exp_24's spill cliff. Dead on arrival. |
| a `__noinline__` second instantiation | amdgpu takes the call-graph max VGPR plus ABI reservations and loses inter-procedural scheduling. *Possible*, but it is a gamble against the occupancy law with no upside over §2.3. Not chosen. |
| co-resident "filler wave" inside M6/M7 | already falsified: a 5th wave destroys the AGPR MFMA form. |
| a CTA pool that *carries* the shared expert's payload | already falsified (+340 µs). Only pools that **consume a certified immutable prefix in the producer's shadow** win. §2.5 is built to be exactly that. |
| `pk_add` of the shared output directly into `out` concurrently with the routed combine | **not** benign — see §3.3. `out` is written by *plain* `uint4` stores in `k0p6_m15_m8_batch` (`k0pf6gm_device_tile_m15.hip` ≈ :730-742), not accumulated. A concurrent atomic add is a lost update. |
| per-row readiness signalling for the filler | probability-zero-class machinery for a dependency that does not exist. The shared expert's input is local. |

### 2.2 Descriptor: one slot, one sub-descriptor

Slots 63-70 are already contested: 63 = `K0P6_D_M15_STAGE` (STAGED) /
`K0P6_D_M15_REP` (REPLICATE) / `K0P6_D_M15_SAVEZ` (SAVE_Z, T2B), 64 =
`K0P6_D_M15_REPDEC`, 65-70 = the M20 slot-pool block
(`k0pf6gm_device_tile_m15.hip:246-274`). SHEXP must compose with M18/M19/M20
(the serving arm we actually want) and with SAVE_Z (the training arm), so it
takes **one** slot above all of them and hangs everything off a
sub-descriptor — the M18R pattern (`:246-253`), which the kernel already knows
how to validate.

```c
// SHEXP: shared-expert filler.  ONE descriptor slot; everything else lives in
// a 16-byte-aligned device block so the arm composes with M18/M19/M20 (63..70)
// and with SAVE_Z.  Layout (mirrored by the host driver and by m15_contracts):
//   u32[0]  magic 0x53485850 ("SHXP")
//   u32[1]  n_sh_tiles          (tiles of 32*G rows covering T_sh rows)
//   u32[2]  T_sh                (= T, this rank's live token count)
//   u32[3]  filler_ctas         (F; 0 disables the dispatch-window pass)
//   u64[2]  shexp_out           bf16[T_sh][7168], rank-LOCAL, M0-zeroed
//   u64[3]  sh_tile_desc        int32[n_sh_tiles], host-static (b0<<4)|gcount
//   u64[4]  sh_sorted_ids       int32[n_sh_tiles*32*G], host-static
//   u64[5]  sh_sorted_eid       int32[n_sh_tiles*G],    host-static (all == E)
//   u64[6]  sh_sorted_w         float[n_sh_tiles*32*G], host-static (all 1.0f)
#define K0P6_D_M15_SHEXP 71
```

`K0P6_M15_D_LEN` becomes `<current> + 1` under SHEXP. The host mirror
(`moe_host_abi.hpp`) gains `shexp_descriptor_words`, an
`append_shexp_descriptor` in the style of `append_mps_descriptor` (:254-266),
and a `validate_shexp_binding` mirroring `validate_mps_binding` (:211-232).
`m15_contracts.py:151` gains `M15_SHEXP_DESCRIPTOR_WORDS`.

Weights: the shared expert lives at **slot `E` (=32) of an extended 33-slot
weight allocation** — no new weight pointers, because `sh_sorted_eid[b] == E`
makes `n2_phase1_gm_mps.cpp`'s `const int e = sorted_eid[b0]` (≈ :341) and
phase 2's equivalent index it for free. Guard: `E + 1 <= K0P6_MAXE` (64), and
under `K0P6_M15_REPLICATE` the shexp slot is `E + s_nrep` and the host must
size `EL + 1 <= K0P6_MAXE`.

### 2.3 The single-call-site rule, and the two micro-seams that buy it

**Everything reuses the *existing* `n2p6m15_phase1_body` / `n2p6m15_phase2_body`
call sites. Nothing is instantiated twice.** The shared expert is expressed as
extra *tiles* in the existing task spaces, which costs zero new inlined code.

Two default-empty macro seams are added to the shared body files. Both are
uniform (SGPR-class) per task and must be `.text`-sha-gated in the default arm.

**Seam P1-A — per-task A/scale base select** (`n2_phase1_gm_mps.cpp`, at the
task head, ≈ :339-345, right after `const int e = sorted_eid[b0];`):

```c
#ifndef N2GM_P1_TASK_SRC_HOOK
#define N2GM_P1_TASK_SRC_HOOK
#endif
  const std::uint8_t* A_t = A_bytes;
  const float*        As_t = A_scale;
  N2GM_P1_TASK_SRC_HOOK          // may reassign A_t / As_t (uniform)
```
Default: `A_t` is loop-invariant, hoisted, byte-identical codegen.
*Not actually needed if §2.4's single-buffer trick is used* — see below; it is
listed because it is the fallback if `T_loc_max` cannot be grown.

**Seam P2-A — per-task output target select** (`n2_phase2_gm_mps.cpp`, same
position, ≈ :480-500):

```c
#ifndef N2GM_P2_TASK_TARGET_HOOK
#define N2GM_P2_TASK_TARGET_HOOK
#endif
    __hip_bfloat16* OUT_t = OUT;
    const unsigned long long* ptab_t = m7_peer_tab;
    N2GM_P2_TASK_TARGET_HOOK     // may reassign both (uniform)
```
and the two `epilogue_write<>` calls (≈ :692-704) pass `OUT_t, ptab_t` instead
of `OUT, m7_peer_tab`. This seam is the whole reason the shared expert needs no
new epilogue: **`epilogue_write`'s `peer_tab == nullptr` branch already exists
in the default build's codegen** (`n2_phase2_gm_mps.cpp:246-256`) and does
precisely what we want — `unsafeAtomicAdd` of a packed bf16 pair into
`OUT + xtok[i]*kHidden + col`, no fabric, no throttle, no peer table.

Under SHEXP the m15 includer defines:

```c
#define N2GM_P2_TASK_TARGET_HOOK                                            \
  if (tile >= k0p6_desc_sh_tile0(k0p6_desc)) {                              \
    OUT_t  = k0p6_desc_shexp_out_biased(k0p6_desc);  /* base - T_ext*H */   \
    ptab_t = nullptr;                                                       \
  }
```

`k0p6_desc_sh_tile0` is a single `s_load` of a u32 from the sub-descriptor,
hoistable out of the task loop by the compiler (it is `__restrict__`-clean and
loop-invariant), so the per-task cost is **one scalar compare + one SGPR
select**. Register-pressure delta: 2 SGPRs, 0 VGPRs. This must still be gated:
`-Rpass-analysis=kernel-resource-usage` VGPR ≤ 256, AGPR ≤ 256, scratch
unchanged, and zero scratch ops within 24 instructions of any remote atomic
(the exp_24 accidental-spill gate).

### 2.4 Where the shared expert's rows live (no new indexing anywhere)

Do **not** invent a parallel activation buffer. Put the shared expert's
quantized rows at the **top of the existing `a_dst` / `sc_stage` arrays**:

* receive rows occupy `[0, T_ext)` with `T_ext = world*MAXTOK = 32768`;
* the shared expert's rows occupy `[T_ext, T_ext + T_sh)`;
* `sh_sorted_ids[i] = T_ext + i` (token *i* of this rank), all `& 0x00FFFFFF`-clean
  since `T_ext + 4096 = 36864 < 2^24`;
* `sh_sorted_eid[b] = E`; `sh_sorted_w[i] = 1.0f`.

Consequences, all of them free:

* `n2_phase1_gm_mps.cpp`'s `a_num_records = T * kHidden` (≈ :313) and the p2
  epilogue's liveness guard `xtok[i] < T` both read `T = nvi[1]`. Under SHEXP,
  `k0p6_sort::scan` must publish `nvi[1] = T_ext + T_sh`. Routed pad rows keep
  their `tok_lds[i] = T` sentinel (`n2_phase1_gm_mps.cpp` ≈ :368) which is still
  `>= T`, so they stay masked. **Checkpoint: audit `k0p6_sort::pad`'s pad-token
  sentinel against the new `nvi[1]`** — this is the single highest-risk
  correctness detail in the whole design.
* `shexp_out` is biased by `-T_ext * kHidden * sizeof(bf16)` when handed to
  `epilogue_write`, so `OUT_t + xtok*kHidden` lands on row `xtok - T_ext` of a
  `[T_sh][7168]` buffer. Zero extra ALU.
* A2q/DQ2 gain `n_sh_tiles*32*G` rows above `PADMAX` (host allocation only).
* Entry guard must additionally require `T_ext + T_sh + K0P6_M15_SLABS <= T_loc_max`
  (extends the existing check at `k0pf6gm_device_tile_m15.hip` ≈ :813).

### 2.5 Task scheduling, and the claim discipline

#### 2.5.1 The arithmetic that sets the granularity (read before proposing "yield")

Measured (exp_33 / m6_m7_structure, quoted in
`aug12/OVERLAP_KERNEL_DESIGN_IDEAS.md:22,25`): **M6 = 2,453.3 µs, M7 = 2,701.8
µs** (of which ~817-898 µs is the remote-RMW epilogue surcharge), dispatch
M0-M2 = **642 µs**, plan M3-M5 = **372.8 µs**, combine M8 = **324.2 µs**.

* Routed p1 task space ≈ `ceil(32768 + pad)/96 × 8` ≈ **2,816 tasks**, 256 CTAs
  ⇒ **≈ 230 µs of CTA time per phase-1 task**.
* Routed p2 task space ≈ 352 tiles × 16 nc = **5,632 tasks** ⇒ **≈ 123 µs per
  phase-2 task**.
* Shared expert = 4096 rows = **1/8** of the 32,768 routed rows, same
  `(7168 × 2048 × 3)` per-row shape. So:
  * p1: 43 tiles × 8 = **344 tasks ≈ 79,000 CTA·µs** (= M6/8 × 256 ✓)
  * p2: 43 tiles × 16 = **688 tasks ≈ 86,000 CTA·µs**
  * total **≈ 165,000 CTA·µs ≈ 645 µs of whole-grid time**

**A single filler task is ~230 µs. The entire dispatch window is 642 µs.** Any
design that says "yield the moment real work is ready" at task granularity is
arithmetically impossible: the yield latency would be a third of the window it
is filling. The discipline must therefore be **quota, not yield** — which is
exactly the shape the kernel already ships and already justifies:

> `k0pf6gm_device_tile_m15.hip` ≈ :1610-1621 —
> *"QUOTA, not exhaustion: an unbounded sweep would hold the R1 barrier hostage
> once slab-1 compute finishes … `flush_rows` (default 16) batches per wave
> keeps the pool's R1 arrival inside the compute shadow; leftovers are claimed
> post-R1 by the whole grid on the same cursor."*

SHEXP is that pattern verbatim, one phase earlier.

#### 2.5.2 The window: is the dispatch actually free?

This is the load-bearing hypothesis and it must be measured **first** (gate G1,
§5).

* M1 pushes **162 MiB** per rank per epoch (`aug12/…:27`).
* MORI's tuned MI350X EP8 ceilings (`aug14/M21_OVERLAP_DIRECTION.md:58-63`):
  dispatch fp8 **342 GB/s at 256 CTAs**; zero-copy combine **435 GB/s at just
  56 CTAs**.
* 162 MiB / 342 GB/s = **497 µs**. Our 642 µs is ~77% of that ceiling.
* exp_10 measured **zero peer wait** in M1↔M2 (`[MPS SPIN] 0/0`), i.e. the
  642 µs is own work, not idling on a peer.

If dispatch is fabric-limited (MORI's 56-CTA combine number strongly suggests
the store-issue side saturates well below 256 CTAs), then reserving F CTAs for
filler leaves dispatch span ~unchanged and hands us **F × 642 CTA·µs** of
otherwise-wasted MFMA capacity. At **F = 160** that is **102,720 CTA·µs** —
enough to swallow the *entire* shared-expert phase 1 (79,000) with headroom.

If dispatch is CU-throughput-bound, the whole filler idea collapses to work
conservation (exp_25's law: static role split = −173 µs) and the arm degrades
to §4.3's fallback. **G1 decides this in one MoK run and one macro.**

#### 2.5.3 The schedule

```
M0  retire-wait  ──────────────────────────────────────────────────────────────
    + grid barrier + counter zero + shexp_out zero                (all 256 CTAs)
    ▼
    ┌─ dispatch CTAs  [0, 256-F)                ┌─ filler CTAs  [256-F, 256)
    │  M1 qpush over a DENSE re-numbering       │  S1  self-quantize own token
    │      of the compute set                   │      slab -> a_dst[T_ext+τ],
    │  M2 chunk_ready polls + unpack            │      sc_stage[T_ext+τ]
    │                                           │  S2  claim <= Q p1 tasks from
    │                                           │      shexp_p1_next (ticket)
    │                                           │      -> A2q/DQ2 rows >= PADMAX
    └──────────────────┬────────────────────────┴───────────────┬──────────────
                       ▼                                        ▼
                   M2 grid barrier  (all 256 CTAs; the rendezvous)
                       ▼
    M3 plan (bid==0 appends the 43 static shexp tiles to tile_desc; nt += 43;
             nvi[1] = T_ext + T_sh)             M4 zero/scale-transpose   M5 scatter
                       ▼
    M6  phase-1 body — ONE call site. Routed tiles + any shexp p1 tiles the
        filler pass did not finish (same ticket cursor; the loop already
        strides a dense task space, the leftovers are simply tiles > nt_routed)
                       ▼
    M7  slab loop — ONE call site. shexp p2 tiles ride the nc-major task space:
        slab 0 computes their columns [0, 3584), slab 1 columns [3584, 7168).
        Seam P2-A retargets them to shexp_out with peer_tab = nullptr.
                       ▼
    M7.5 slab rendezvous (unchanged)   ── the slab-0 barrier CERTIFIES the shexp
                                          front-half columns for the M8 pool
                       ▼
    M8  combine — k0p6_m15_m8_batch adds shexp_out[tok][off] into acc before the
        bf16 pack. Front sweep reads [0,3584) (certified by slab 0), back sweep
        reads [3584,7168) (certified by the slab-1 words + R1/R2).
                       ▼
    M9  unchanged
```

#### 2.5.4 Claim discipline, stated as the consuming-pool law

The falsified law says: *pools that CARRY payload lose (+340 µs); only pools
CONSUMING a certified immutable prefix in the producer's shadow win.* SHEXP is
built to satisfy it three times over:

1. **The filler pool carries nothing.** It does not touch `a_ll`, `dest_counter`,
   `rows_done`, `chunk_ready`, `slots`, or any peer address. Its entire working
   set is rank-local: `hidden` (read), `a_dst[T_ext..]`/`sc_stage[T_ext..]`
   (write-then-read, *its own* rows), `A2q/DQ2[PADMAX..]` (write). Zero fabric
   traffic, zero atomics on shared protocol cells except one `fetch_add_relaxed`
   per claimed task.
2. **The prefix it consumes is certified by construction.** A filler CTA
   quantizes the rows of the tiles it will itself GEMM (S1 covers exactly the
   tile slab that S2 claims), so the producer and the consumer are the same
   CTA: `__syncthreads()` + `s_waitcnt vmcnt(0)` is the whole certification.
   **No cross-CTA readiness protocol exists in this arm.** (Cross-CTA would
   require a chunk-ready word per token block; it buys nothing and re-introduces
   the polling class that L3 says is the actual interference.)
3. **It runs in the producer's shadow.** The producer is M1's store/fabric
   stream; the filler is MFMA. exp_22 measured co-resident *store-class* work
   taxing compute at **0.073%**; this is the mirror case, and G1 measures it
   directly rather than assuming it.

**Quota.** `Q = K0P6_M15_SHEXP_QUOTA` (compile-time, swept). A filler CTA claims
at most `Q` p1 tasks, then falls through to the M2 grid barrier unconditionally.
Leftovers are not lost: they are the same tiles the M6 task loop will find above
`nt_routed`, claimed from the same ticket cursor. **A filler CTA never spins and
never polls.** The barrier delay is bounded by *one* task (≤230 µs worst case at
the last claim) and the quota is chosen so the expected finish precedes the
dispatch CTAs' arrival: with F = 160 and the 642 µs window, `Q = 2` is the
conservative point (2 × 230 = 460 µs), `Q = 3` the aggressive one (690 µs).
Sweep `Q ∈ {1,2,3}` × `F ∈ {64, 96, 128, 160}`.

**Ticket cells.** `K0P6_MPS_ST_WORDS = 8` scalar words exist in `mps_state`;
only 0-3 are used (`moe_mps_adapter.cuh:91-95`). Take words **4 = `shexp_p1_next`**
and **5 = `shexp_q_next`** (the S1 quantize-slab cursor). Zeroed by the existing
M0 loop (`k0pf6gm_device_tile_m15.hip` ≈ :890-897), which already strides
`K0P6_MPS_ST_WORDS + 2*K0P6_MPS_TS_COUNT`.

**Dense re-numbering for the dispatch set.** M1's stripe assignment is
`gw = bid*(blockDim.x>>6)+wid; nw = nct*(blockDim.x>>6)` (≈ :1002-1003) and M2's
is `wave/wave_count` (≈ :1206-1207). Under SHEXP these become dense over
`[0, 256-F)` — the *exact* pattern `moe_mps_adapter.cuh:685-692`
(`compute_id_of`) already implements for the service pool, whose contract note
says both branches must produce a dense numbering "or tasks would be skipped or
run twice". Reuse it; do not write a second one.

**Placement.** The filler set should be a *tail* reservation like the service
pool's mode-2 form, i.e. `bid >= nct - F`, which spreads one-eighth per XCD
(`moe_mps_adapter.cuh:376-390`) and touches all eight L2s. Do **not** use whole
dies (mode 3) here: the filler is MFMA-bound, not atomic-bound, so exp_08's
die-isolation motive does not apply, and tail placement keeps the dispatch set
spread across all XCDs where its fabric ports are.

### 2.6 Occupancy accounting

Nothing in this design adds a register to M6 or M7:

* no new call site → no new inlined body → no widened whole-function allocation;
* Seam P2-A adds 2 SGPRs and 0 VGPRs (uniform pointer select);
* the S1 quantizer and the filler control flow live **before** the M1 block and
  are dead by the M2 barrier — but they are in the same function, so their live
  ranges *do* enter the whole-function allocation. Mitigation: S1 reuses the
  exact register shape of M1's quantizer (7 `uint4` + scales, `k0pf6gm_device_tile_m15.hip`
  ≈ :1048-1080) and is written as a separate `__device__ __forceinline__`
  helper called from one place, so its peak is ≤ M1's peak, which the current
  allocation already accommodates. **This is a gate (G2), not an assertion:**
  build with `-Rpass-analysis=kernel-resource-usage` and diff VGPR/AGPR/SGPR/
  scratch against the flag-off build.

---

## 3. Input availability and output placement

### 3.1 Input — the shared expert's activations are free

The shared expert consumes `hidden_states` — descriptor slot
`K0P6_D_HIDDEN` (= 2), bound at `m15_vllm.py:306-312` from `prepare()`'s
`a1.data_ptr()`. It is resident and immutable from the first instruction of the
kernel. **No dependency on M1, M2, the router, or any peer.**

It must be fp8-quantized in the N2 layout (1×128 blocks, 56 scale groups per
row). M1 already computes exactly that quantization per token
(`k0pf6gm_device_tile_m15.hip` ≈ :1048-1080: per-128-lane-group `amax` via
`__shfl_xor`, `scale = max(a,1e-6)/448`, `__hip_cvt_float_to_fp8`, 7 × `uint4`
+ 56 fp32 scales). Two ways to get it:

* **(chosen) S1 self-quantize.** The filler CTA quantizes its own token slab
  into `a_dst[T_ext+τ]` / `sc_stage[(T_ext+τ)*56]`, duplicating ~30 lines under
  `#if K0P6_M15_SHEXP`. Cost: 58.7 MB read + 29.4 MB write + 4096×56×4 B ≈
  **~18 µs of grid-wide bandwidth**. Chosen because it needs **zero** cross-CTA
  certification (§2.5.4 point 2) and touches M1's codegen not at all.
* **(rejected) piggyback on M1's quantizer.** Tempting — M1 already has the
  bytes in registers — but (i) M1's `tau` loop is strided by *global wave*, so a
  tile's 96 tokens are produced by 96 different waves, forcing a real cross-CTA
  readiness protocol; (ii) it edits the hottest loop in the dispatch phase;
  (iii) `if (fanout == 0) continue;` (≈ :1044) short-circuits before the
  quantize, so a token routed nowhere would be skipped. Rejected on (i) alone.

### 3.2 Output — where it accumulates

`shexp_out`: **bf16 `[T_sh][7168]`, rank-local, 58.7 MB**, zeroed in M0
alongside `a2_done`/`part_done` (`k0pf6gm_device_tile_m15.hip` ≈ :884-889) — a
grid-strided store of 58.7 MB ≈ **12 µs**, entirely inside the M0 retire-wait
shadow.

Phase 2 writes it through the **already-compiled** `peer_tab == nullptr` branch
of `epilogue_write` (`n2_phase2_gm_mps.cpp:246-256`): `unsafeAtomicAdd` of a
`__hip_bfloat162` at `OUT_t + xtok*kHidden + col_base + 2*dcol`. Because each
`(shexp row, column)` pair is produced by exactly one `(tile, nc, sb, j)`, the
atomic add on a zeroed buffer is semantically a store — but it costs nothing to
keep it an atomic and it keeps us on the existing instruction stream.

M8 folds it in. In `k0p6_m15_m8_batch` (`k0pf6gm_device_tile_m15.hip` ≈
:730-742), immediately before the bf16 pack:

```c
#if K0P6_M15_SHEXP
    // SHEXP: the shared expert's contribution for this token's column slice.
    // Written by the phase-2 local-atomic branch and certified by the slab
    // rendezvous (front half, slab 0) / the M8 entry poll (back half, slab 1).
    // Column-disjoint by CLO/CHI, so exactly once per element.
    {
      const uint4 sx = *reinterpret_cast<const uint4*>(
          reinterpret_cast<const unsigned char*>(
              shexp_out + (size_t)(tok0 + t) * (size_t)K0P6_H) + off);
      const unsigned short* s16 =
          reinterpret_cast<const unsigned short*>(&sx);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[t][e] += k0p6_bf16_to_f32(s16[e]);
    }
#endif
```

Cost: one extra 16 B load per `(token, chunk)`; 4096 × 14 × 64 lanes × 16 B =
**58.7 MB ≈ 12 µs**, on a line the wave is about to write anyway. Exactly-once
follows from M8's own structure: the front cursor `m8_front` covers chunks
`[0, 7)` and the back cursor `m8_back` covers `[7, 14)`, each batch claimed once
(`k0pf6gm_device_tile_m15.hip` ≈ :1898-1930).

### 3.3 The race analysis the task asked for — and why the obvious answer is wrong

**Claim to reject: "pk_add the shared output straight into the token's `out`
row concurrently with the remote partials; atomic adds commute, so it's
benign."**

It is not benign, for two independent reasons:

1. **`out` is never accumulated.** The routed partials do not land in `out`;
   they land in `slots` (peer RMW, `n2_phase2_gm_mps.cpp:236`
   `accumulate_peer_bf162`). `out` is produced by M8 with a **plain `uint4`
   store** (`k0pf6gm_device_tile_m15.hip` ≈ :740). A concurrent
   `unsafeAtomicAdd` into `out` is a classic lost update: whichever M8 store
   lands after the add erases it. Atomicity of the *add* buys nothing when the
   other party is a non-atomic store.
2. **`slots` is not a legal target either.** Adding into
   `slots[cur*MAXTOK + row]` would require a 9th fanout record per token, and
   the fanout tables are exactly full: `pull_stage` is `[T][TOPK][2]` with
   `TOPK = 8`, and `pull_src_fill` is sized `T*world = 32768` records against a
   max fanout of 8 per token. No spare slot exists, and widening them is a
   protocol change across M1/M3/M8.

**M4 interaction.** `zero_part_scale_transpose` (`k0pf6gm_device_tile_m15.hip`
≈ :1434-1451) zeroes **`part`** (the phase-2 `[T_ext][7168]` staging), not
`out`. So a shared-expert partial deposited into `part` before M4 would be
*erased* by M4, and one deposited after M4 would collide with the routed phase-2
writes. `part` is unusable as the shared expert's accumulator.

⇒ **A dedicated `shexp_out` with the fold performed inside M8's existing
accumulate is the only placement that is exactly-once, race-free, and
zero-protocol.** Its ordering obligations are discharged by rendezvous that
already exist:

* front half (columns `[0, 3584)`, M8 chunks `[0,7)`): written during M7 slab 0,
  certified by the slab-0 `release_cta_payload_system` + grid barrier +
  `acquire_payload_agent` (≈ :1690-1706) — the same certification the existing
  mid-M7 pool sweep relies on, and the existing comment already establishes the
  column-disjointness argument (≈ :1528-1533).
* back half (columns `[3584, 7168)`, M8 chunks `[7,14)`): written during slab 1,
  certified by the slab-1 words polled at M8 entry (≈ :1873-1884).
* any shexp p2 tile completed by the *filler* pass (if §4.3's variant B is
  used) is certified by the M2 grid barrier, which strictly precedes both.

**Poison/negative control.** Fill `shexp_out` with a sentinel instead of zero in
a debug arm and assert M8's output changes by exactly the sentinel — this is the
`K0_MOK_POISON_OUT` idiom (`mok_synthetic_prefill/run_campaign.sh:134`).

---

## 4. Sizing

### 4.1 The FLOP ledger (per rank, per MoE layer, balanced T=4096)

| | rows | MACs | share |
|---|---|---|---|
| routed (M6+M7) | 32,768 receive rows | 32,768 × 7168 × 2048 × 3 = 1.44e12 | 1 |
| shared expert | 4,096 local tokens | 4,096 × 7168 × 2048 × 3 = 1.80e11 | **1/8** |

At the mega's *measured* throughput (M6 2,453.3 + M7 2,701.8 µs, of which
~858 µs is the remote-RMW surcharge the shared expert does not pay):

* shexp p1 ≈ 2,453.3/8 = **307 µs** grid-wide (**79,000 CTA·µs**)
* shexp p2 ≈ (2,701.8 − 858)/8 = **230 µs** grid-wide (**59,000 CTA·µs**), plus
  the local-atomic write of 58.7 MB ≈ 12 µs and the M0 zero ≈ 12 µs
* total in-kernel **≈ 550-645 µs of grid-wide work**

### 4.2 Fitting it to the windows

| window | measured | usable CTA·µs at F=160 | fits |
|---|---|---|---|
| M0 retire-wait | ~0 at steady state (graph-replay loop) | ~0 | `shexp_out` zero only (12 µs, bandwidth) |
| **M0→M2 dispatch** | **642 µs**, zero peer wait at std=0 | **102,720** | **all of p1 (79,000) with 23,000 slack** |
| M2 chunk_ready poll | ~0 at std=0; grows with the critical-rank law under real serving skew | skew-dependent bonus | absorbs quota overshoot |
| M3-M5 plan | 372.8 µs, but 3 grid barriers inside | unusable at 230 µs granularity | no |
| M7 slab space | — | p2 rides as appended tiles | +230 µs to M7 |
| M8 | 324.2 µs | the +12 µs fold | — |

**Net in-region delta (fabric-bound hypothesis holds):**
`+230 µs (p2 in M7) + 12 µs (M0 zero) + 12 µs (M8 fold) + 18 µs (S1 quantize)`
`≈ +272 µs`, plus whatever dispatch slowdown G1 measures (hypothesis: ≈ 0).

**Net in-region delta (hypothesis fails, pure work conservation):**
`≈ +645 µs`.

### 4.3 What we delete outside

The vLLM-side shared expert per layer:

* 3 block-fp8 GEMMs at `[4096,7168]×[7168,4096]`, `[4096,4096]×[4096,7168]`
  (the merged gate/up counts as one N=4096 GEMM), 3.61e11 FLOP total. At the
  AITER/vLLM linear path's realized throughput for these shapes — call it
  600-900 TFLOPS, i.e. **400-600 µs** — plus
* `SiluAndMul` over `[4096, 4096]` bf16 (≈ 50 MB traffic, ~15 µs), plus
* 4 kernel launches on the critical path, plus
* the `shared_output + fused_output` add at `moe_runner.py:723`: 176 MB ≈ **35-40 µs**.

⇒ **S_out ≈ 450-700 µs per MoE layer, fully serialized in front of the mega.**

### 4.4 Expected net

| arm | per-layer Δ | 58 layers | E2E at MoE fraction f = 0.40-0.54 |
|---|---|---|---|
| filler works (G1 green) | −(450…700) + 272 = **−180…−430 µs** | −10.4…−24.9 ms | **−1.6% … −4.6%** |
| work-conserving fallback | −(450…700) + 645 = **−55…+195 µs** | −3.2…+11.3 ms | **−0.6% … +1.2%** |

Two honesty notes that must survive into any writeup:

* Even the *fallback* is not obviously negative, because moving the shared
  expert inside deletes 4 launches and an add kernel per layer and runs the
  same FLOPs at the mega's tiled efficiency rather than three separate
  launch-bounded GEMMs. But it is inside the noise.
* **The serving measurement floor is ±15% at n=1**
  (`aug14/M21_OVERLAP_DIRECTION.md:44-48`: stock self-varied 9.6k-12.7k tok/s
  across one day; the balanced m15-vs-stock pair measured *flat* even though
  the kernel win was 1.32×). A −1.6…−4.6% E2E effect is **not measurable in
  serving at n=1**. SHEXP must be measured at the kernel/MoK level with
  region-time receipts, and in serving only through a many-pair, order-balanced
  protocol — or not claimed at all.

---

## 5. Validation ladder and implementation plan

### 5.1 Gates (in order; each is a hard stop)

**G1 — the fabric-bound hypothesis (do this FIRST; ~2 h, decides the design).**
One macro, no shared-expert code: `K0P6_M15_SHEXP_DRYRUN=F` makes the tail F
CTAs skip M1/M2 entirely (dense re-numbering for the rest) and go straight to
the M2 barrier. Sweep `F ∈ {0, 32, 64, 96, 128, 160}` in MoK and read the M2
timestamp (`K0P6_MPS_TS_M2_DONE`, already stamped at ≈ :1256-1264).
*Green:* M2-done is flat to within ±40 µs up to F ≈ 160 ⇒ the window is real.
*Red:* M2-done scales like `642 × 256/(256−F)` ⇒ dispatch is CU-bound, the
filler pass is work-conserving; drop to variant B (below) and re-price.
**This gate is worth running even if SHEXP is never built** — it is the missing
measurement behind every "dispatch overlap" claim in the aug12 document.

**G2 — codegen parity and resource usage.** Flag-off build's `.text` sha must
equal the pre-SHEXP build's, byte for byte (L7). Flag-on build: VGPR ≤ 256,
AGPR ≤ 256, scratch not increased, `v_mfma` count unchanged in the routed path,
`flat_atomic_pk_add_bf16` count unchanged, zero scratch ops within 24
instructions of any remote atomic.

**G3 — MoK correctness with a synthetic shared expert.** In
`prefill_opt_host/e004pf_k0pf_ab.py`, add `K0_MOK_SHEXP=1`: build the 33-slot
extended weight tensors (reuse the `_m18_w13/_m18_fc1/_m18_w2c/_m18_fc2` path at
:3101-3107) with slot 32 = a freshly generated expert; build the static
`sh_tile_desc/sh_sorted_ids/sh_sorted_eid/sh_sorted_w` tables; allocate
`shexp_out`; append descriptor slot 71. Reference = torch
`silu(x@w_gate)*(x@w_up)@w_down` in fp32 + the existing routed reference.
Gates: `K0_MOK_ABSOLUTE_TOLERANCE/RELATIVE_TOLERANCE`, `K0_MOK_POISON_OUT=1`,
`K0_MOK_POISON_SELFTEST=1`, plus the §3.3 sentinel control, plus a **600-epoch
soak** (the family's standing rule).

**G4 — MoK T-generalization.** T ∈ {1024, 2048, 4096}. The static shexp tables
are T-dependent; `nvi[1] = T_ext + T_sh` and the `k0p6_sort::pad` sentinel audit
(§2.4) are the two places this breaks. This is the exp_04/exp_36 landmine class.

**G5 — MoK timing, F×Q sweep.** `F ∈ {64, 96, 128, 160}` × `Q ∈ {1,2,3}`, ≥5
order-balanced pairs each, judged on **M2-done, M6-done, M7-done, and total**
together (never one alone — M7 and combine are anti-correlated at r = −0.904;
judge their sum).

**G6 — skew replay.** Re-run G5 under `K0_MOK_ROUTE_HIST` with the banked
serving histograms. The filler window *grows* with skew (the M2 poll becomes a
real wait under the critical-rank law), so SHEXP should look better here than
under balanced routing. If it does not, the window model is wrong.

**G7 — serving, shim only, SHEXP off.** Land the two monkeypatches with the
claim hard-disabled; prove R3 (null-arm parity) and that no receipt changes.

**G8 — serving, SHEXP on.** R1 + R2 accuracy receipts on every MoE layer for
the first N steps, then ≥5 order-balanced throughput pairs with
production+EPLB always in the table. **Pre-register that the expected effect
(−1.6…−4.6%) is below the n=1 noise floor**; the deliverable is the receipt
that accuracy holds and region time moved, not a throughput claim.

### 5.2 Implementation plan, file:line, with hours

Line numbers drift ±30; grep for the quoted construct.

| # | file | site | change | h |
|---|---|---|---|---|
| 1 | `k0pf6gm_device_tile_m15.hip` | after :274 (the M20 slot block) | `K0P6_M15_SHEXP` (default 0), `K0P6_M15_SHEXP_CTAS`, `K0P6_M15_SHEXP_QUOTA`, `K0P6_D_M15_SHEXP 71`, `K0P6_M15_D_LEN` bump, `#error` guards vs. every 63-70 claimant | 1 |
| 2 | `moe_host_abi.hpp` | after :266 | `shexp_descriptor_words`, `shexp_binding`, `validate_shexp_binding`, `append_shexp_descriptor` (mirror :211-266) | 1.5 |
| 3 | `k0pf6gm_device_tile_m15.hip` | entry guard ≈ :795-818 | validate the SHXP magic/alignment/`T_ext+T_sh+SLABS <= T_loc_max`/`E+1 <= K0P6_MAXE`; `atomicOr(pperr, K0P6_MPS_ERR_CONFIG)` on failure | 1 |
| 4 | same | M0 zero loop ≈ :884-897 | grid-strided zero of `shexp_out`; ticket words 4/5 already covered by the existing `mps_state` loop | 0.5 |
| 5 | same | new `__device__ __forceinline__ k0p6_m15_shexp_quant_slab(...)` near :640 | S1: duplicate the M1 quantizer (≈ :1048-1080) writing `a_dst[T_ext+τ]` / `sc_stage[(T_ext+τ)*56]`; ticketed on `mps_state[5]` | 3 |
| 6 | same | new filler block between the M0 close (:865) and M1 (:960) | role predicate (tail F), S1 then ≤Q p1 claims from `mps_state[4]`, then fall through to the M2 barrier. **Filler CTAs must not execute M1's `qpush_done` arrival** — audit :1122-1131 | 4 |
| 7 | same | M1 ≈ :1002-1003, M2 ≈ :1206-1207, M2 unpack ≈ :1288 | dense re-numbering over `[0, 256-F)` using `hk_moe::mps::compute_id_of`'s pattern (`moe_mps_adapter.cuh:685-692`) | 2 |
| 8 | same | M3 plan, `bid==0` tile builder ≈ :1400-1419 | append the 43 static shexp tiles after the routed ones; raise the `bcap` check; publish `nt` | 1.5 |
| 9 | `hkp_sort.hpp` (via the m15 call at ≈ :1385) | `k0p6_sort::scan` | `nvi[1] = T_ext + T_sh` under SHEXP; **audit `pad`'s sentinel** (§2.4) | 2 |
| 10 | `n2_phase2_gm_mps.cpp` | task head ≈ :480-500; `epilogue_write` calls ≈ :692-710 | seam P2-A (`OUT_t`/`ptab_t` + `N2GM_P2_TASK_TARGET_HOOK`) | 1.5 |
| 11 | `k0pf6gm_device_tile_m15.hip` | the P2 includer block ≈ :417-455 | define `N2GM_P2_TASK_TARGET_HOOK` under SHEXP; helpers `k0p6_desc_sh_tile0` / `k0p6_desc_shexp_out_biased` | 1 |
| 12 | same | `k0p6_m15_m8_batch` ≈ :730-742 | the SHEXP fold (§3.2); thread `shexp_out` into the template's args at both call sites (:1623, :1916-1941) | 1.5 |
| 13 | `prefill_opt_host/e004pf_k0pf_ab.py` | ≈ :3080-3155 | `K0_MOK_SHEXP`: extended weights, static tables, `shexp_out`, slot 71, `_expected_mps_words` arm, torch reference | 4 |
| 14 | `mok_synthetic_prefill/run_campaign.sh` | :140-144 | `-e K0_MOK_SHEXP`, `-e K0_M15_SHEXP_CTAS`, `-e K0_M15_SHEXP_QUOTA` | 0.5 |
| 15 | `pf4h_integration/m15_contracts.py` | :151, the slot enum ≈ :120-150 | `SHEXP = 71`, `M15_SHEXP_DESCRIPTOR_WORDS`, attestation of the new slot | 1 |
| 16 | `pf4h_integration/m15_runtime.py` | :355-370, :482-501, :997-1034 | shexp weight extension + shuffle + free-the-originals receipt; `shexp_out` allocation in the buffer set; slot-71 write | 5 |
| 17 | `pf4h_integration/m15_vllm.py` | `M15Experts.apply` :369-446 | pass the extended weight bases; `_validate_weight_layout` gains the 33-slot arm | 1.5 |
| 18 | `pf4h_integration/apply.py` | new patch entries | patch `SharedExperts.forward` (`runner/shared_experts.py:155`) and `MoERunner._apply_quant_method` (`runner/moe_runner.py:547`); `_pf4h_shexp_claim`; `M15-SHEXP-001` fail-closed in `_eligible`/`finalize` | 4 |
| 19 | — | — | G1 dry-run macro (item 6's role predicate + an early `goto` to the M2 barrier), standalone | 1 |
| | | | **≈ 38 h engineering + node time** | |

Suggested order: **19 → G1**, then 1-4, 10-12 (+G2), 5-9, 13-14 (+G3, G4),
G5, G6, then 15-18 (+G7, G8).

### 5.3 Operator instructions

1. `git checkout -b ablations-shexp` from `ablations`. Sync the repo to the node
   as usual (`BUILDING.md`).
2. **G1 first.** Build the dry-run macro only:
   `-DK0P6_M15_SHEXP_DRYRUN=<F>` for `F ∈ {0,32,64,96,128,160}`. Run
   `mok_synthetic_prefill/run_campaign.sh` with
   `K0_MOK_ARMS=production,pf4h`, `K0_MOK_TIMED_ITERS=100`, timestamps on.
   Report `K0P6_MPS_TS_M2_DONE` per F. **Stop and report before building
   anything else** — G1 decides whether the rest of this document is worth
   building.
3. Every subsequent build: capture `.text` sha of the flag-off arm and diff it
   against the pre-SHEXP baseline (G2). Capture
   `-Rpass-analysis=kernel-resource-usage` for both arms.
4. Correctness before timing, always: `K0_MOK_POISON_OUT=1`,
   `K0_MOK_POISON_SELFTEST=1`, 600-epoch soak, then the F×Q sweep.
5. No serving run until G3-G6 are green and R2 (shexp isolation) passes in MoK.

---

## 6. Risks, ranked

1. **The dispatch window may not exist (P ≈ 0.4).** If M1 is CU-throughput-bound
   rather than fabric-bound, filler is pure work conservation (exp_25: static
   role split = −173 µs) and the net collapses into the noise. *Mitigated by
   G1 costing 2 hours and running first.* This is the single highest-value
   measurement in the document regardless of SHEXP's fate.
2. **`nvi[1]` / pad-sentinel regression (P ≈ 0.35, severity: silent wrong
   answers).** Raising `nvi[1]` from `T_ext` to `T_ext + T_sh` changes the
   liveness guard `xtok[i] < T` and the phase-1 buffer-resource bound for the
   *routed* path too. The exp_04/exp_36 T-generalization defect was exactly this
   class of off-by-a-segment error and it silently sprayed `recv_eid = -1`
   across segments. *Mitigated by G4 (T ∈ {1024,2048,4096}) plus a dedicated
   unit assertion that routed pad rows still mask.*
3. **Codegen regression from seam P2-A (P ≈ 0.3, severity: +700 µs class).**
   exp_38 is the precedent: an *unreachable* line near the M7 epilogue cost
   +726.9 µs by collapsing the injection window. Two extra SGPRs at the task
   head is a smaller perturbation, but the epilogue is at 256 VGPR / 256 AGPR
   with 128 B/lane of scratch already. *Mitigated by G2's hard gate including
   the "zero scratch ops within 24 instructions of any remote atomic" check.*
4. **Filler CTAs delay the M2 grid barrier (P ≈ 0.3, severity: 100-230 µs).**
   Quota granularity is one ~230 µs task. Over-quota directly lengthens the
   region. *Mitigated by the F×Q sweep and by the fact that leftovers are free
   (they fall into the M6 task space).*
5. **Shim claim/eligibility divergence (P ≈ 0.15, severity: wrong output).**
   The claim is taken at `moe_runner.py:560`, before the authoritative
   `_eligible()` at `prepare()`. *Mitigated by the `M15-SHEXP-001` fail-closed
   rule; a divergence must crash, never degrade.*
6. **Weight-shuffle mismatch (P ≈ 0.15).** The shared expert's linears are not
   in AITER-shuffled layout; the shuffle must match the routed path exactly.
   *Mitigated by R2 (topk-zeroed isolation probe), which fails loudly and
   immediately if the layout is wrong.*
7. **HBM budget (P ≈ 0.1).** `shexp_out` 58.7 MB/layer if allocated per layer
   (3.4 GB across 58) — allocate **one** buffer in the shared state ring, not
   per layer (the ring already exists: `m15_runtime.py` ring slots). The
   extended weight slot is net-zero *only if the originals are freed*; make the
   free a logged receipt.
8. **`a_dst`/`sc_stage`/`A2q`/`DQ2` growth (P ≈ 0.1).** `T_loc_max` must grow by
   `T_sh`; `A2q/DQ2` by `n_sh_tiles*32*G` rows. Host-side only, but it changes
   the attested buffer sizes in `m15_contracts.py` and must not silently
   collide with the TBO parity doubling.
9. **Unmeasurable in serving (P ≈ 0.8, severity: reputational).** The predicted
   E2E effect sits under the ±15% n=1 floor. *Mitigated by pre-registering the
   claim shape: SHEXP's serving deliverable is an accuracy receipt plus a
   region-time delta, not a throughput number.*
10. **Composition with M20 (P ≈ 0.1).** The M20 prefetch engine already carves
    the last `K0P6_M20_PF_CTAS = 8` CTAs from the service pool during M7. SHEXP
    carves a tail set during M0-M2 — a *different* phase, so they do not
    conflict today, but both use tail placement and the `#error` guards plus a
    combined-arm build must be part of G2.

---

## 7. What this design deliberately does not do

* It does not build a general device-side work-queue / CTA-role abstraction.
  `aug15/OVERLAP_PROGRAM.md:60-78` puts that in T2, where the wgrad filler needs
  it. SHEXP reuses the two role mechanisms that already ship
  (`is_service_cta`/`compute_id_of`, and the M8 ticket cursors) and adds two
  ticket words. If T2's role scheduler lands first, SHEXP should be re-expressed
  on top of it rather than the reverse.
* It does not touch the dispatch or combine transports. fp8-on-wire, the
  destination-interleave, and the `s_sleep` backoff hygiene
  (`aug15/OVERLAP_PROGRAM.md:80-92`) are orthogonal and independently ranked.
* It does not attempt cross-epoch pipelining (K3/TBO-2). SHEXP is deliberately
  the *same-epoch* overlap that needs no parity buffers, no parity retirement,
  and no second generation of any protocol cell — which is why it is buildable
  in ~38 h instead of a week.

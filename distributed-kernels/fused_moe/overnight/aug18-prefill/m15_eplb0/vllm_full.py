"""Exact vLLM 0.25.1 hybrid PF4H seam.

Only a non-ragged prefill bucket with exactly 4096 original live tokens on all
eight DP/EP ranks takes the PF4H path. The exact vLLM graph-dispatch patch gives
that case a distinct graph key; every decode, other prefill size, padded B4096
graph, and unattested invocation retains stock Mori+AITER.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterExperts,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)

from .contracts import FROZEN_CONTRACT, PREFILL_B4096_BUCKET
from .errors import ActivationBlocker, PF4HActivationError
from .live_runtime import (
    ACTIVATION_FILE_ENV,
    LayerState,
    PF4HProcessRuntime,
    runtime_for_stock_mori,
)


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _fail(code: str, summary: str, evidence: str) -> None:
    raise PF4HActivationError((ActivationBlocker(code, summary, evidence),))


def _eplb_compose_gate(
    moe_config: Any, blockers: list[ActivationBlocker]
) -> bool:
    """Whether the EPLB placement-only compose owns the EPLB decision.

    Imported lazily: ``m15_contracts``/``m15_eplb`` ship only with
    ``--integration-mode m15``, and a ``full``-mode install must keep this
    module importable.  With the file set absent, or the variable unset, the
    caller falls through to the verbatim PF4H-FULL-018 refusal.
    """

    try:
        from .m15_contracts import eplb_compose_enabled
        from .m15_eplb import validate_eplb_compose
    except ImportError:
        return False
    if not eplb_compose_enabled():
        return False
    blockers.extend(validate_eplb_compose(moe_config))
    return True


def validate_full_selection(
    moe_config: Any,
    weight_key: Any,
    activation_key: Any,
    activation_format: mk.FusedMoEActivationFormat,
) -> None:
    """Reject anything except the frozen DeepSeek-R1 EP8 block-FP8 config."""

    parallel = getattr(moe_config, "moe_parallel_config", None)
    observed = {
        "tp_size": getattr(parallel, "tp_size", None),
        "dp_size": getattr(parallel, "dp_size", None),
        "ep_size": getattr(parallel, "ep_size", None),
        "global_experts": getattr(moe_config, "num_experts", None),
        "local_experts": getattr(moe_config, "num_local_experts", None),
        "topk": getattr(moe_config, "experts_per_token", None),
        "hidden_size": getattr(moe_config, "hidden_dim", None),
        "intermediate_size": getattr(
            moe_config, "intermediate_size_per_partition", None
        ),
    }
    if parallel is None or any(value is None for value in observed.values()):
        missing = tuple(name for name, value in observed.items() if value is None)
        _fail(
            "PF4H-FULL-001",
            f"vLLM config is missing exact fields {missing!r}",
            "the vLLM 0.25.1 FusedMoEConfig/ParallelConfig schema",
        )
    FROZEN_CONTRACT.validate_observed(**observed)
    blockers: list[ActivationBlocker] = []
    if not bool(getattr(moe_config, "use_mori_kernels", False)):
        blockers.append(
            ActivationBlocker(
                "PF4H-FULL-002",
                "PF4H full mode was selected without stock Mori enabled",
                "--all2all-backend mori_high_throughput",
            )
        )
    if not bool(getattr(parallel, "use_ep", False)):
        blockers.append(
            ActivationBlocker(
                "PF4H-FULL-003",
                "expert parallelism is not enabled",
                "contiguous EP8 expert ownership",
            )
        )
    # EPLB is refused by default.  ``VLLM_PF4H_M15_EPLB=1`` swaps this single
    # blocker for the placement-only compose gate: with zero redundant
    # experts vLLM keeps 32 physical slots per rank in the same contiguous
    # linear layout and only permutes which logical expert occupies each slot,
    # and the router already resolves topk_ids to physical slot ids before the
    # modular kernel sees them, so ``dest = eid / E`` inherits the balancing
    # unchanged.  See m15_eplb.py for the evidence and the attestations.
    if not _eplb_compose_gate(moe_config, blockers) and bool(
        getattr(parallel, "enable_eplb", False)
    ):
        blockers.append(
            ActivationBlocker(
                "PF4H-FULL-018",
                "EPLB can remap experts after graph capture",
                "disabled EPLB and fixed linear EP8 ownership, or "
                "VLLM_PF4H_M15_EPLB=1 for the placement-only compose",
            )
        )
    if bool(getattr(parallel, "use_batched_activation_format", False)):
        blockers.append(
            ActivationBlocker(
                "PF4H-FULL-004",
                "batched-experts activation format is enabled",
                "the standard Mori modular activation format",
            )
        )
    if activation_format != mk.FusedMoEActivationFormat.Standard:
        blockers.append(
            ActivationBlocker(
                "PF4H-FULL-005",
                f"activation format is {activation_format!r}",
                "FusedMoEActivationFormat.Standard",
            )
        )
    if (weight_key, activation_key) != (
        kFp8Static128BlockSym,
        kFp8Dynamic128Sym,
    ):
        blockers.append(
            ActivationBlocker(
                "PF4H-FULL-006",
                f"quant keys are {(weight_key, activation_key)!r}",
                "static 128x128 FP8 weights plus dynamic 1x128 activations",
            )
        )
    activation = getattr(moe_config, "activation", MoEActivation.SILU)
    if activation != MoEActivation.SILU:
        blockers.append(
            ActivationBlocker(
                "PF4H-FULL-007",
                f"MoE activation is {activation!r}",
                "the N2 SiLU gate proven by the B4096 receipt",
            )
        )
    if not bool(getattr(moe_config, "is_act_and_mul", True)):
        blockers.append(
            ActivationBlocker(
                "PF4H-FULL-008",
                "the MoE is not using a gated activation-and-multiply",
                "the frozen gate/up N2 layout",
            )
        )
    activation_file = os.environ.get(ACTIVATION_FILE_ENV, "")
    if not activation_file or not Path(activation_file).is_absolute():
        blockers.append(
            ActivationBlocker(
                "PF4H-FULL-019",
                f"{ACTIVATION_FILE_ENV}={activation_file!r}",
                "an absolute sentinel path created only after vLLM is ready",
            )
        )
    if blockers:
        raise PF4HActivationError(blockers)


def _layer_id(layer: Any) -> int:
    candidates = (
        getattr(layer, "layer_name", None),
        getattr(layer, "_layer_name", None),
        getattr(layer, "prefix", None),
    )
    for candidate in candidates:
        if isinstance(candidate, str):
            match = _LAYER_PATTERN.search(candidate)
            if match is not None:
                layer_id = int(match.group(1))
                if layer_id in FROZEN_CONTRACT.moe_layer_ids:
                    return layer_id
    candidate_id = getattr(layer, "layer_id", None)
    if isinstance(candidate_id, int) and candidate_id in FROZEN_CONTRACT.moe_layer_ids:
        return candidate_id
    _fail(
        "PF4H-FULL-009",
        f"could not resolve an exact layer id from {candidates!r}/{candidate_id!r}",
        "a RoutedExperts name containing '.layers.<3..60>.'",
    )
    raise AssertionError("unreachable")


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


class PF4HPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Hybrid controller that retains and delegates to stock Mori."""

    def __init__(self, stock: Any, layer: Any):
        super().__init__()
        if stock.__class__.__name__ != "MoriPrepareAndFinalize":
            _fail(
                "PF4H-FULL-010",
                f"stock prepare/finalize is {stock.__class__.__name__}",
                "the exact live vLLM MoriPrepareAndFinalize",
            )
        if getattr(stock, "mori_op", None) is None:
            _fail(
                "PF4H-FULL-011",
                "stock MoriPrepareAndFinalize has no mori_op",
                "the cached live EpDispatchCombineOp",
            )
        self.stock = stock
        self.mori_op = stock.mori_op
        self.layer = layer
        self.layer_id = _layer_id(layer)
        self.runtime: PF4HProcessRuntime | None = None
        self.state: LayerState | None = None
        self.experts: PF4HExperts | None = None
        self._active = False
        self._execution = "stock"
        self._attested_expert_map_pointer: int | None = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return self.stock.activation_format

    def output_is_reduced(self) -> bool:
        return self.stock.output_is_reduced()

    def num_dispatchers(self) -> int:
        return self.stock.num_dispatchers()

    def max_num_tokens_per_rank(self) -> int | None:
        return self.stock.max_num_tokens_per_rank()

    def topk_indices_dtype(self) -> torch.dtype | None:
        return self.stock.topk_indices_dtype()

    def supports_async(self) -> bool:
        # The exact stock Mori implementation is synchronous and PF4H shares
        # that ordering contract.
        return False

    def on_commit(self) -> None:
        self.stock.on_commit()

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts) -> None:
        self.stock.post_init_setup(fused_experts)
        if not isinstance(fused_experts, PF4HExperts):
            _fail(
                "PF4H-FULL-012",
                f"PF4H controller was paired with {type(fused_experts).__name__}",
                "PF4HExperts behind the same modular kernel",
            )
        runtime = runtime_for_stock_mori(self.stock)
        state = runtime.state_for_layer(self.layer_id)
        self.runtime = runtime
        self.state = state
        self.experts = fused_experts
        fused_experts.bind_controller(self)

    @property
    def active(self) -> bool:
        return self._active

    def _capture_in_progress(self) -> bool:
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except RuntimeError:
            return False

    def _attest_dp_batch(self) -> bool:
        if not is_forward_context_available():
            return False
        context = get_forward_context()
        descriptor = getattr(context, "batch_descriptor", None)
        # The exact graph patch adds this key only after DP coordination proves
        # every rank's *original*, pre-padding count is 4096. This distinction
        # cannot be reconstructed from DPMetadata, which intentionally carries
        # post-padding counts for graph replay.
        if (
            descriptor is None
            or getattr(descriptor, "uniform", None) is not False
            or int(getattr(descriptor, "num_tokens", -1))
            != PREFILL_B4096_BUCKET.tokens_per_rank
            or getattr(descriptor, "pf4h_exact_b4096", None) is not True
        ):
            return False
        additional = getattr(context, "additional_kwargs", {})
        is_dummy = additional.get("pf4h_is_dummy_run") is True
        graph_target = additional.get("pf4h_graph_target") is True
        if is_dummy and not graph_target:
            return False
        dp_metadata = getattr(context, "dp_metadata", None)
        across = getattr(dp_metadata, "num_tokens_across_dp_cpu", None)
        if across is None:
            return False
        try:
            values = tuple(int(value) for value in across.tolist())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
        return values == (PREFILL_B4096_BUCKET.tokens_per_rank,) * (
            FROZEN_CONTRACT.dp_size
        )

    def _attest_expert_map(
        self,
        expert_map: torch.Tensor | None,
        *,
        capture_in_progress: bool,
    ) -> bool:
        runtime = self.runtime
        if runtime is None or expert_map is None:
            return False
        shape = _shape(expert_map)
        if len(shape) != 1 or shape[0] < FROZEN_CONTRACT.global_experts:
            return False
        pointer = int(expert_map.data_ptr())
        # ``tolist`` on a GPU tensor synchronizes the host and is illegal while
        # a graph stream is capturing. The mandatory eager graph warmup proves
        # the ownership map once; capture is then allowed only with that exact
        # pointer, which must remain stable for graph replay.
        if capture_in_progress:
            return self._attested_expert_map_pointer == pointer
        if self._attested_expert_map_pointer is not None:
            return self._attested_expert_map_pointer == pointer
        try:
            values = tuple(
                int(value)
                for value in expert_map[: FROZEN_CONTRACT.global_experts].tolist()
            )
        except (RuntimeError, TypeError, ValueError):
            return False
        begin = runtime.rank * FROZEN_CONTRACT.local_experts
        expected_map = tuple(
            expert - begin
            if begin <= expert < begin + FROZEN_CONTRACT.local_experts
            else -1
            for expert in range(FROZEN_CONTRACT.global_experts)
        )
        expected_mask = tuple(
            1
            if begin <= expert < begin + FROZEN_CONTRACT.local_experts
            else 0
            for expert in range(FROZEN_CONTRACT.global_experts)
        )
        # RoutedExperts supplies global->local mapping normally and the AITER
        # binary mask when ROCm AITER fused-MoE mode is enabled.  Both prove
        # the same fixed contiguous EP8 ownership needed by qpush.
        if values not in (expected_map, expected_mask):
            return False
        self._attested_expert_map_pointer = pointer
        return True

    def _eligible(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: Any,
        defer_input_quant: bool,
    ) -> bool:
        b = PREFILL_B4096_BUCKET
        c = FROZEN_CONTRACT
        capture_in_progress = self._capture_in_progress()
        if (
            # A non-uniform B4096 CUDA graph may later replay for a smaller
            # live batch padded to the same descriptor. The exact runner patch
            # prevents that graph-key collision before this predicate runs.
            self.runtime is None
            or self.state is None
            or _shape(a1) != (b.tokens_per_rank, c.hidden_size)
            or _shape(topk_ids) != (b.tokens_per_rank, c.topk)
            or _shape(topk_weights) != (b.tokens_per_rank, c.topk)
            or a1.dtype != torch.bfloat16
            or topk_ids.dtype != torch.int32
            or topk_weights.dtype != torch.float32
            or not a1.is_contiguous()
            or not topk_ids.is_contiguous()
            or not topk_weights.is_contiguous()
            or num_experts != c.global_experts
            or apply_router_weight_on_input
            or defer_input_quant
            or not bool(getattr(quant_config, "is_block_quantized", False))
        ):
            return False
        runtime = self.runtime
        assert runtime is not None
        if not self._attest_dp_batch():
            return False
        context = get_forward_context()
        graph_target = (
            getattr(context, "additional_kwargs", {}).get("pf4h_graph_target") is True
        )
        if not self._attest_expert_map(
            expert_map,
            capture_in_progress=capture_in_progress,
        ):
            if graph_target:
                _fail(
                    "PF4H-FULL-017",
                    f"layer {self.layer_id} graph capture has an unattested "
                    "or pointer-unstable expert map",
                    "one warmup-attested contiguous EP8 map whose pointer "
                    "remains stable through capture and replay",
                )
            return False
        if capture_in_progress and not graph_target:
            return False
        if graph_target:
            self._execution = (
                "graph-capture" if capture_in_progress else "graph-warmup"
            )
            return True
        self._execution = "eager"
        return runtime.operator_activation_enabled(self.layer_id)

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: Any,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        active = self._eligible(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        self._active = active
        if not active:
            return self.stock.prepare(
                a1,
                topk_weights,
                topk_ids,
                num_experts,
                expert_map,
                apply_router_weight_on_input,
                quant_config,
                defer_input_quant,
            )
        assert self.runtime is not None and self.state is not None
        self.runtime.begin_invocation(self.layer_id, self._execution)
        self.runtime.launch_prepare(self.state, a1, topk_ids, topk_weights)
        # Experts consumes the original input only as an interface placeholder;
        # qpush's persistent a_dst/sc_dst buffers carry the real N2 inputs.
        return a1, None, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if not self._active:
            self.stock.finalize(
                output,
                fused_expert_output,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                weight_and_reduce_impl,
            )
            return
        b = PREFILL_B4096_BUCKET
        c = FROZEN_CONTRACT
        if (
            _shape(output) != (b.tokens_per_rank, c.hidden_size)
            or output.dtype != torch.bfloat16
            or not output.is_contiguous()
        ):
            _fail(
                "PF4H-FULL-013",
                f"active layer {self.layer_id} output is "
                f"{_shape(output)!r}/{output.dtype}/{output.is_contiguous()}",
                f"contiguous BF16 {(b.tokens_per_rank, c.hidden_size)!r}",
            )
        assert self.runtime is not None and self.state is not None
        self.runtime.launch_finalize(self.state, output)
        self.runtime.finish_invocation(self.layer_id)

    def poll_protocol_error(self) -> int:
        """Out-of-capture health hook for the worker/control plane."""

        if self.runtime is None:
            _fail(
                "PF4H-FULL-014",
                f"layer {self.layer_id} runtime has not been initialized",
                "FusedMoEKernel post_init_setup completion",
            )
        return self.runtime.protocol_error(self.layer_id)


class PF4HExperts(AiterExperts):
    """AITER fallback plus dsort/N2 for an active exact-B4096 invocation."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._pf4h_controller: PF4HPrepareAndFinalize | None = None

    def bind_controller(self, controller: PF4HPrepareAndFinalize) -> None:
        if self._pf4h_controller is not None and self._pf4h_controller is not controller:
            _fail(
                "PF4H-FULL-015",
                "one PF4HExperts instance was rebound to another layer controller",
                "one experts/controller/state tuple per MoE layer",
            )
        self._pf4h_controller = controller

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        controller = self._pf4h_controller
        if controller is None:
            _fail(
                "PF4H-FULL-016",
                "PF4HExperts.apply ran before controller binding",
                "FusedMoEKernel post_init_setup",
            )
        if not controller.active:
            super().apply(
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )
            return
        if (
            activation != MoEActivation.SILU
            or global_num_experts != FROZEN_CONTRACT.global_experts
            or expert_tokens_meta is not None
            or apply_router_weight_on_input
            or a1q_scale is not None
            or a2_scale is not None
        ):
            _fail(
                "PF4H-FULL-017",
                "active PF4H experts received a post-prepare contract mismatch",
                "SiLU/global-E256/no-stock-dispatch-metadata/no-input-weighting",
            )
        runtime = controller.runtime
        state = controller.state
        if runtime is None or state is None:
            raise AssertionError("active PF4H controller is not bound")
        runtime.launch_experts(
            state,
            controller.layer_id,
            w1,
            w2,
            self.quant_config.w1_scale,
            self.quant_config.w2_scale,
        )


def wrap_mori_prepare_finalize(stock: Any, layer: Any) -> PF4HPrepareAndFinalize:
    """Construct the hybrid wrapper without replacing the stock Mori object."""

    return PF4HPrepareAndFinalize(stock, layer)

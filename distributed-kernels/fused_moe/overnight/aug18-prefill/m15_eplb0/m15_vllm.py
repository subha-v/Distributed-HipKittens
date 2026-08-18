"""Exact vLLM 0.25.1 hybrid M15 seam (additive to ``vllm_full.py``).

Everything about *when* the accelerated path is legal is inherited verbatim
from :class:`~.vllm_full.PF4HPrepareAndFinalize`: the same non-ragged
exact-B4096 DP proof, the same contiguous-EP8 expert-map attestation, the same
activation sentinel, the same single-stream ownership rule.  Only *what* runs
changes -- six pinned launches become one ``k0pf6gm_m15_mega`` launch driven by
a device-resident descriptor.

Selection: ``VLLM_PF4H_INTEGRATION_MODE=m15``.  With the variable unset, or
set to ``full``, nothing in this module is imported and the PF4H path is
byte-identical.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from .contracts import FROZEN_CONTRACT
from .errors import ActivationBlocker, PF4HActivationError
from .m15_contracts import (
    M15_PREFILL_B4096_BUCKET,
    M15_SERVING_CONFIG,
    encode_config,
    m15_mode_selected,
    ring_slots,
)
from .m15_eplb import (
    eplb_compose_enabled,
    install_eplb_hooks,
    register_layer_weights,
    require_attested,
    validate_runtime_compose,
)
from .m15_graph_body import M15BoundState, M15GraphBody
from .m15_runtime import (
    M15LayerDescriptor,
    M15ProcessRuntime,
    commit_captured_descriptors,
    describe_pperr,
    runtime_for_stock_mori,
)
from .vllm_full import PF4HExperts, PF4HPrepareAndFinalize, validate_full_selection

logger = logging.getLogger(__name__)


def _fail(code: str, summary: str, evidence: str) -> None:
    raise PF4HActivationError((ActivationBlocker(code, summary, evidence),))


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def validate_m15_selection(
    moe_config: Any,
    weight_key: Any,
    activation_key: Any,
    activation_format: mk.FusedMoEActivationFormat,
) -> None:
    """Every PF4H selection gate, plus the M15-only preconditions."""

    validate_full_selection(
        moe_config, weight_key, activation_key, activation_format
    )
    blockers: list[ActivationBlocker] = []
    if not m15_mode_selected():
        blockers.append(
            ActivationBlocker(
                "M15-ENV-001",
                "validate_m15_selection ran without "
                "VLLM_PF4H_INTEGRATION_MODE=m15",
                "the exact opt-in string 'm15'",
            )
        )
    try:
        slots = ring_slots()
    except PF4HActivationError as exc:
        blockers.extend(exc.blockers)
        slots = 2
    bucket = M15_PREFILL_B4096_BUCKET
    if bucket.tokens_per_rank != bucket.maxtok:
        blockers.append(
            ActivationBlocker(
                "M15-BUCKET-001",
                f"T={bucket.tokens_per_rank} != MAXTOK={bucket.maxtok}",
                "the balanced exact-B4096 shape M15 is validated at",
            )
        )
    if blockers:
        raise PF4HActivationError(blockers)
    logger.warning(
        "M15_SELECTION_RECEIPT bucket=%r ring_slots=%d config=%#x",
        bucket.key,
        slots,
        encode_config(M15_SERVING_CONFIG),
    )


def _validate_weight_layout(
    layer_id: int,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> None:
    """The exact shuffled-AITER layout gate PF4H's binder applies.

    M15 consumes the same BF16-packed byte views of the stock shuffled FP8
    allocations, so the shape/contiguity contract is identical; only the
    binding target differs (descriptor words instead of pybind arguments).
    """

    c = FROZEN_CONTRACT
    expected = (
        (c.local_experts, 2 * c.intermediate_size, c.hidden_size),
        (c.local_experts, c.hidden_size, c.intermediate_size),
        (
            c.local_experts,
            (2 * c.intermediate_size) // c.quant_block,
            c.hidden_size // c.quant_block,
        ),
        (
            c.local_experts,
            c.hidden_size // c.quant_block,
            c.intermediate_size // c.quant_block,
        ),
    )
    observed = tuple(
        _shape(tensor) for tensor in (w1, w2, w1_scale, w2_scale)
    )
    if observed != expected:
        _fail(
            "M15-WEIGHT-010",
            f"layer {layer_id} runtime shapes={observed!r}",
            f"exact shuffled weight/scale shapes {expected!r}",
        )
    if (
        int(w1.element_size()) != 1
        or int(w2.element_size()) != 1
        or not w1.is_contiguous()
        or not w2.is_contiguous()
        or not w1_scale.is_contiguous()
        or not w2_scale.is_contiguous()
    ):
        _fail(
            "M15-WEIGHT-011",
            f"layer {layer_id} weights are not contiguous FP8 storage",
            "one-byte contiguous AITER shuffled containers and F32 scales",
        )


class _MegaLaunch:
    """Adapter so :class:`M15GraphBody` never sees the runtime object."""

    def __init__(self, runtime: M15ProcessRuntime, record: M15LayerDescriptor):
        self._runtime = runtime
        self._record = record

    def launch(self, stream: object, state: M15BoundState) -> None:
        del stream, state  # the runtime owns the current-stream handle
        self._runtime.launch(self._record)


class M15PrepareAndFinalize(PF4HPrepareAndFinalize):
    """PF4H's eligibility proof driving one megakernel launch."""

    def __init__(self, stock: Any, layer: Any):
        super().__init__(stock, layer)
        self.m15_runtime: M15ProcessRuntime | None = None
        self.record: M15LayerDescriptor | None = None
        self._pending_inputs: tuple[int, int, int] | None = None

    # -- lifecycle -----------------------------------------------------
    def post_init_setup(self, fused_experts: mk.FusedMoEExperts) -> None:
        # Deliberately NOT super().post_init_setup(): that constructs the PF4H
        # four-kernel runtime and its own two-slot ring.  Only the stock
        # delegate's hook and the experts binding are shared.
        self.stock.post_init_setup(fused_experts)
        if not isinstance(fused_experts, M15Experts):
            _fail(
                "M15-FULL-001",
                f"M15 controller was paired with {type(fused_experts).__name__}",
                "M15Experts behind the same modular kernel",
            )
        runtime = runtime_for_stock_mori(self.stock)
        record = runtime.descriptor_for_layer(self.layer_id)
        self.m15_runtime = runtime
        self.record = record
        # ``_eligible`` and the expert-map attestation are inherited and only
        # need not-None sentinels plus ``runtime.rank`` /
        # ``runtime.operator_activation_enabled``; the M15 runtime provides
        # both, so the PF4H predicate runs verbatim against M15 state.
        self.runtime = runtime  # type: ignore[assignment]
        self.state = record  # type: ignore[assignment]
        self.experts = fused_experts
        fused_experts.bind_controller(self)
        if eplb_compose_enabled():
            # Refuse a replica-pool runtime (EPLB would leave the M18/M20
            # copies stale), then arm the placement proof.  This runs inside
            # process_weights_after_loading, which vLLM completes before
            # EplbState.add_model, so the add_model wrapper installed here is
            # always in time to attest the initial map.
            validate_runtime_compose(runtime)
            install_eplb_hooks()

    # -- invocation ----------------------------------------------------
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
            # Any out-of-capture non-M15 call is a safe place to publish the
            # capture-time descriptor writes, in addition to the runner hook.
            if not self._capture_in_progress():
                commit_captured_descriptors()
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
        # Fail closed: an eligible invocation must never precede the
        # placement proof.  A dict-lookup on the steady-state path.
        require_attested()
        runtime, record = self._require_bound()
        runtime.begin_invocation(self.layer_id, self._execution)
        # M20 cached layers: write this chunk's decision bitmap on-stream
        # between the router and the mega launch (a no-op for every other
        # mode/layer).  Capture-safe by construction; see the runtime.
        runtime.write_decision_bitmap(record, topk_ids)
        # No launch here: the megakernel needs ``out``, which only finalize()
        # owns.  The three input addresses are recorded and validated against
        # the finalize-time set so a mismatched pairing cannot slip through.
        self._pending_inputs = (
            int(topk_ids.data_ptr()),
            int(topk_weights.data_ptr()),
            int(a1.data_ptr()),
        )
        del record
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
        bucket = M15_PREFILL_B4096_BUCKET
        c = FROZEN_CONTRACT
        if (
            _shape(output) != (bucket.tokens_per_rank, c.hidden_size)
            or output.dtype != torch.bfloat16
            or not output.is_contiguous()
        ):
            _fail(
                "M15-FULL-002",
                f"active layer {self.layer_id} output is "
                f"{_shape(output)!r}/{output.dtype}/{output.is_contiguous()}",
                f"contiguous BF16 {(bucket.tokens_per_rank, c.hidden_size)!r}",
            )
        pending = self._pending_inputs
        if pending is None:
            _fail(
                "M15-FULL-003",
                f"layer {self.layer_id} finalize ran without a paired prepare",
                "strict prepare -> experts -> finalize ordering",
            )
        runtime, record = self._require_bound()
        my_ids, my_wgt, hidden = pending
        if (
            int(topk_ids.data_ptr()) != my_ids
            or int(topk_weights.data_ptr()) != my_wgt
        ):
            _fail(
                "M15-FULL-004",
                f"layer {self.layer_id} finalize saw different topk buffers "
                "than prepare",
                "one stable routing tensor pair per invocation",
            )
        capturing = self._capture_in_progress()
        runtime.bind_dynamic(
            record,
            my_ids=my_ids,
            my_wgt=my_wgt,
            hidden=hidden,
            out=int(output.data_ptr()),
            capturing=capturing,
        )
        runtime.launch(record)
        self._pending_inputs = None
        if not capturing:
            runtime.assert_compiled_g(record)
        runtime.finish_invocation(self.layer_id)

    # -- helpers -------------------------------------------------------
    def _require_bound(self) -> tuple[M15ProcessRuntime, M15LayerDescriptor]:
        runtime = self.m15_runtime
        record = self.record
        if runtime is None or record is None:
            _fail(
                "M15-FULL-005",
                f"layer {self.layer_id} M15 runtime has not been initialized",
                "FusedMoEKernel post_init_setup completion",
            )
            raise AssertionError("unreachable")
        return runtime, record

    def graph_body(self) -> M15GraphBody:
        """The one-launch capture body, for standalone capture harnesses."""

        runtime, record = self._require_bound()
        state = M15BoundState(
            layer_id=self.layer_id,
            ring_slot=record.ring_slot,
            bucket_key=runtime.bucket.key,
            descriptor_address=int(record.desc.data_ptr()),
            pperr_address=int(record.buffers.tensors["pperr"].data_ptr()),
            spin_limit=runtime.spin_limit,
        )
        return M15GraphBody(state=state, mega=_MegaLaunch(runtime, record))

    def poll_protocol_error(self) -> int:
        runtime = self.m15_runtime
        if runtime is None:
            _fail(
                "M15-FULL-006",
                f"layer {self.layer_id} M15 runtime has not been initialized",
                "FusedMoEKernel post_init_setup completion",
            )
            raise AssertionError("unreachable")
        value = runtime.protocol_error(self.layer_id)
        if value:
            logger.error(
                "M15_PPERR rank=%d layer=%d value=%#x detail=%s",
                runtime.rank,
                self.layer_id,
                value,
                describe_pperr(value),
            )
        return value


class M15Experts(PF4HExperts):
    """AITER fallback; the active path only binds weights for the megakernel."""

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
                "M15-FULL-007",
                "M15Experts.apply ran before controller binding",
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
                "M15-FULL-008",
                "active M15 experts received a post-prepare contract mismatch",
                "SiLU/global-E256/no-stock-dispatch-metadata/no-input-weighting",
            )
        if not isinstance(controller, M15PrepareAndFinalize):
            _fail(
                "M15-FULL-009",
                f"M15Experts is bound to {type(controller).__name__}",
                "M15PrepareAndFinalize",
            )
        runtime, record = controller._require_bound()
        w1_scale = self.quant_config.w1_scale
        w2_scale = self.quant_config.w2_scale
        _validate_weight_layout(
            controller.layer_id, w1, w2, w1_scale, w2_scale
        )
        runtime.bind_layer_weights(
            record,
            int(w1.data_ptr()),
            int(w1_scale.data_ptr()),
            int(w2.data_ptr()),
            int(w2_scale.data_ptr()),
        )
        # Retain the tensors themselves so the post-rearrangement sweep can
        # re-read data_ptr() -- bind_layer_weights only runs when Python does,
        # and steady-state graph replay runs none.
        register_layer_weights(
            controller.layer_id, w1, w1_scale, w2, w2_scale
        )


def wrap_mori_prepare_finalize(stock: Any, layer: Any) -> M15PrepareAndFinalize:
    """Construct the M15 wrapper without replacing the stock Mori object."""

    return M15PrepareAndFinalize(stock, layer)

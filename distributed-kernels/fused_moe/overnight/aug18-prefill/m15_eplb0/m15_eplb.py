"""EPLB placement-only compose for M15 (additive; ``VLLM_PF4H_M15_EPLB=1``).

Why this module is small: vLLM 0.25.1 resolves logical expert ids to physical
slot ids *inside the router*, not in the dispatch layer.
``BaseRouter._select_experts`` runs

    topk_weights, topk_ids = self._compute_routing(...)   # logical
    topk_ids = self._apply_eplb_mapping(topk_ids)         # -> PHYSICAL
    topk_ids = self._convert_indices_dtype(topk_ids, ...)

(``fused_moe/router/base_router.py`` lines 291-303), and ``MoERunner`` hands
that result straight to ``forward_modular`` -> the modular kernel ->
``M15PrepareAndFinalize.prepare`` (``fused_moe/runner/moe_runner.py`` line
573).  So the megakernel's ``dest = eid / E`` already sees a physical slot id
and inherits EPLB's balancing with **no gather and no kernel change**.

What is left is proof, not plumbing.  Three invariants make the compose sound
and this module attests all three:

1.  **The map is a pure permutation.**  With ``num_redundant_experts == 0``
    every ``logical_replica_count`` entry is 1, so the router's replica hash
    (``replica_idx = hashed % replica_count``) collapses to slot 0 and the
    mapping is deterministic.  ``logical_to_physical_map[..., 0]`` must be a
    permutation of ``[0, 256)`` per layer -- if any entry were ``-1`` (the
    sparse-map sentinel, ``base_router.py`` line 60) the kernel would compute
    ``dest = -1 / 32 == 0`` and a negative local expert index, i.e. an
    out-of-bounds write into rank 0's symmetric heap.

2.  **Pointers survive rearrangement.**  ``_commit_eplb_maps`` writes with
    ``dst.copy_(src)`` into the tensors allocated once in
    ``EplbState.add_model`` (``eplb_state.py`` lines 381-385, 1245, 1252), and
    the per-layer ``EplbLayerState`` holds views of those
    (``eplb_state.py`` line 1060).  The captured Triton mapping kernel reads
    them by pointer, so content updates between replays are picked up for
    free.  Weights move with ``w[dst].copy_(...)`` into the live parameters
    (``rebalance_execute.py`` lines 386, 424), so the descriptor's
    capture-committed W13/S13/W2/S2 addresses stay valid.

3.  **Nothing re-binds.**  Under steady-state graph replay no Python runs, so
    ``bind_layer_weights``'s M15-WEIGHT-002 pointer gate is dormant.  This
    module restores it by re-attesting after every rearrangement, which is the
    only moment any of the three can change.

Hook placement follows the load order: ``process_weights_after_loading`` (and
therefore ``M15PrepareAndFinalize.post_init_setup``) runs inside
``model_loader.load_model``, which ``GPUModelRunner.load_model`` calls *before*
``EplbState.add_model`` (``gpu_model_runner.py`` lines 5308-5391).  Wrapping
``add_model`` from ``post_init_setup`` is therefore always in time to attest
the initial placement before any capture or replay.  Everything here is
out-of-capture and the steady-state cost is one device->host sync per
rearrangement (every ``step_interval`` steps, 3000 by default).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .contracts import FROZEN_CONTRACT
from .errors import ActivationBlocker, PF4HActivationError
from .m15_contracts import EPLB_COMPOSE_ENV, eplb_compose_enabled

logger = logging.getLogger(__name__)


def _fail(code: str, summary: str, evidence: str) -> None:
    raise PF4HActivationError((ActivationBlocker(code, summary, evidence),))


# --------------------------------------------------------------------------
# Selection gate
# --------------------------------------------------------------------------


def validate_eplb_compose(moe_config: Any) -> list[ActivationBlocker]:
    """The compose preconditions, returned as blockers for the caller to raise.

    Called from ``validate_full_selection`` in place of the unconditional
    PF4H-FULL-018 refusal, and only when ``VLLM_PF4H_M15_EPLB=1``.
    """

    blockers: list[ActivationBlocker] = []
    parallel = getattr(moe_config, "moe_parallel_config", None)
    if not bool(getattr(parallel, "enable_eplb", False)):
        blockers.append(
            ActivationBlocker(
                "M15-EPLB-001",
                f"{EPLB_COMPOSE_ENV}=1 without vLLM EPLB enabled",
                "--enable-eplb, or unset the compose variable",
            )
        )

    # Redundant experts are the whole hazard: they push the physical count
    # past 256, break the 32-per-rank partition ``dest = eid / 32`` assumes,
    # and re-arm the replica hash the permutation proof relies on collapsing.
    physical = getattr(moe_config, "num_experts", None)
    logical = getattr(moe_config, "num_logical_experts", None)
    if physical is None or logical is None:
        blockers.append(
            ActivationBlocker(
                "M15-EPLB-002",
                "FusedMoEConfig is missing num_experts/num_logical_experts",
                "the vLLM 0.25.1 FusedMoEConfig schema",
            )
        )
    elif int(physical) != int(logical):
        blockers.append(
            ActivationBlocker(
                "M15-EPLB-003",
                f"num_redundant_experts={int(physical) - int(logical)}",
                "placement-only EPLB: --eplb-config "
                '\'{"num_redundant_experts": 0}\'',
            )
        )
    elif int(physical) != FROZEN_CONTRACT.global_experts:
        blockers.append(
            ActivationBlocker(
                "M15-EPLB-004",
                f"physical expert count is {int(physical)}",
                f"exactly {FROZEN_CONTRACT.global_experts} physical experts",
            )
        )

    eplb_config, parallel_config = _live_eplb_config()
    if eplb_config is None:
        blockers.append(
            ActivationBlocker(
                "M15-EPLB-005",
                "no current VllmConfig to read eplb_config from",
                "activation inside a live worker with a set VllmConfig",
            )
        )
        return blockers

    # Async EPLB runs a daemon thread with its OWN cuda stream issuing NCCL
    # P2P (``eplb/async_worker.py`` lines 31-46) *concurrently with the model
    # forward*.  That is the exact axis this project already measured as
    # fatal: unbounded remote injection cost +211us and the depth-4 vmcnt
    # throttle is the single biggest win (-615us).  Upstream hit the same wall
    # from the other side and pins NCCL_MAX_CTAS=8 to stop NCCL starving a
    # cooperative mega-MoE launch (``eplb/eplb_utils.py`` lines 64-109).  The
    # M15 mega is a 256-CTA persistent grid with device-side spin loops; CTAs
    # stolen by NCCL turn straight into spin_limit trips.
    if bool(getattr(eplb_config, "use_async", True)):
        blockers.append(
            ActivationBlocker(
                "M15-EPLB-006",
                "async EPLB overlaps NCCL weight transfer with the mega",
                '--eplb-config \'{"use_async": false}\' (note: async is the '
                "vLLM default, so this must be set explicitly)",
            )
        )

    # Elastic EP is the one path that REBINDS rather than copies:
    # ``_commit_eplb_maps`` replaces ``physical_to_logical_map`` outright when
    # the physical count changes (``eplb_state.py`` lines 1242-1245), and the
    # per-rank expert count stops being 32.
    if bool(getattr(parallel_config, "enable_elastic_ep", False)):
        blockers.append(
            ActivationBlocker(
                "M15-EPLB-007",
                "elastic EP can change the physical expert count at runtime",
                "a fixed EP8 world with 32 physical experts per rank",
            )
        )
    return blockers


def _live_eplb_config() -> tuple[Any | None, Any | None]:
    try:
        from vllm.config import get_current_vllm_config_or_none
    except ImportError:  # pragma: no cover - container-only path
        return (None, None)
    config = get_current_vllm_config_or_none()
    if config is None:
        return (None, None)
    parallel_config = getattr(config, "parallel_config", None)
    return (getattr(parallel_config, "eplb_config", None), parallel_config)


def validate_runtime_compose(runtime: Any) -> None:
    """Refuse the compose against an M18/M20 replica-pool runtime.

    M18/M20 stage *copies* of expert weights into an extended allocation
    (``m15_runtime.replica_pool_pointers``); EPLB rewrites only the live base
    slabs, so every replica slot would silently serve pre-rearrangement
    weights.  There is no cheap fix -- the pools would have to be rebuilt on
    every rearrangement -- so this is a hard refusal rather than a warning.
    """

    if not eplb_compose_enabled():
        return
    if getattr(runtime, "m18_layer_sets", None) is not None:
        _fail(
            "M15-EPLB-008",
            "EPLB compose requested with M18 per-layer replica tables active",
            "unset the M18 sets file, or run EPLB without replication",
        )
    if getattr(runtime, "m20_budget", ()):
        _fail(
            "M15-EPLB-009",
            "EPLB compose requested with an M20 replica-cache budget active",
            "unset the M20 budget, or run EPLB without the replica cache",
        )


# --------------------------------------------------------------------------
# What a rearrangement has to leave untouched
# --------------------------------------------------------------------------


_LOCK = threading.Lock()
_HOOKS_INSTALLED = False
_MODEL_STATES: list[Any] = []
_MAP_POINTERS: dict[int, tuple[int, int]] = {}
_WEIGHTS: dict[int, tuple[Any, ...]] = {}
_WEIGHT_POINTERS: dict[int, tuple[int, ...]] = {}
_ATTESTED = False


def attested() -> bool:
    """Whether the initial placement proof has run."""

    return _ATTESTED


def require_attested() -> None:
    """Fail closed if an eligible invocation precedes the placement proof."""

    if not eplb_compose_enabled():
        return
    if not _ATTESTED:
        _fail(
            "M15-EPLB-010",
            "an M15 invocation became eligible before the EPLB placement "
            "was attested",
            "EplbState.add_model running after "
            "M15PrepareAndFinalize.post_init_setup, which is the vLLM "
            "load_model order",
        )


def register_layer_weights(
    layer_id: int,
    w13: Any,
    s13: Any,
    w2: Any,
    s2: Any,
) -> None:
    """Retain the four live weight tensors so they can be re-attested.

    ``bind_layer_weights`` already refuses a changed pointer, but it only runs
    when Python runs.  Under steady-state replay the descriptor is the only
    consumer, so the tensors themselves have to be held to re-read
    ``data_ptr()`` after a rearrangement.
    """

    if not eplb_compose_enabled():
        return
    tensors = (w13, s13, w2, s2)
    pointers = tuple(int(tensor.data_ptr()) for tensor in tensors)
    with _LOCK:
        previous = _WEIGHT_POINTERS.get(layer_id)
        if previous is not None and previous != pointers:
            _fail(
                "M15-EPLB-011",
                f"layer {layer_id} weight pointers moved from {previous!r} "
                f"to {pointers!r}",
                "EPLB rearrangement writes in place "
                "(rebalance_execute.py move_from_buffer)",
            )
        _WEIGHTS[layer_id] = tensors
        _WEIGHT_POINTERS[layer_id] = pointers


def install_eplb_hooks() -> None:
    """Wrap ``EplbState.add_model`` and ``EplbState.step`` once per process.

    ``add_model`` allocates the map tensors and is the earliest point the
    placement can be proved.  ``step`` is called by the runner after the model
    forward (``gpu_model_runner.py`` line 4767), out of capture, on the main
    stream, with no mega in flight on this rank; a rearrangement is detected
    by the step counter resetting, which ``EplbState.step`` does immediately
    before calling ``rearrange()`` (``eplb_state.py`` lines 655-656).
    """

    global _HOOKS_INSTALLED
    if not eplb_compose_enabled():
        return
    with _LOCK:
        if _HOOKS_INSTALLED:
            return
        try:
            from vllm.distributed.eplb.eplb_state import EplbState
        except ImportError as exc:  # pragma: no cover - container-only path
            _fail(
                "M15-EPLB-012",
                f"EplbState import failed: {exc}",
                "a vLLM build carrying vllm.distributed.eplb",
            )
            raise AssertionError("unreachable")
        if getattr(EplbState, "_m15_eplb_hooked", False):
            _HOOKS_INSTALLED = True
            return

        original_add_model = EplbState.add_model
        original_step = EplbState.step

        def adding(self, *args: Any, **kwargs: Any):
            result = original_add_model(self, *args, **kwargs)
            _attest_state(self)
            return result

        def stepping(self, *args: Any, **kwargs: Any):
            before = int(getattr(self, "expert_rearrangement_step", 0))
            result = original_step(self, *args, **kwargs)
            after = int(getattr(self, "expert_rearrangement_step", 0))
            if after < before:
                reattest_after_rearrangement()
            return result

        EplbState.add_model = adding  # type: ignore[method-assign]
        EplbState.step = stepping  # type: ignore[method-assign]
        EplbState._m15_eplb_hooked = True  # type: ignore[attr-defined]
        _HOOKS_INSTALLED = True
        logger.warning(
            "M15_EPLB_HOOK_RECEIPT state=installed env=%s", EPLB_COMPOSE_ENV
        )


def _attest_state(eplb_state: Any) -> None:
    """Prove every model state's placement and latch its map pointers."""

    global _ATTESTED
    model_states = list(getattr(eplb_state, "model_states", {}).values())
    if not model_states:
        _fail(
            "M15-EPLB-013",
            "EplbState.add_model produced no model state",
            "one MixtureOfExperts model registered with EPLB",
        )
    for model_state in model_states:
        _prove_permutation(model_state)
        _MAP_POINTERS[id(model_state)] = (
            int(model_state.logical_to_physical_map.data_ptr()),
            int(model_state.logical_replica_count.data_ptr()),
        )
    _MODEL_STATES.clear()
    _MODEL_STATES.extend(model_states)
    _ATTESTED = True
    logger.warning(
        "M15_EPLB_PLACEMENT_RECEIPT models=%d layers=%d experts=%d "
        "state=permutation",
        len(model_states),
        int(model_states[0].logical_to_physical_map.shape[0]),
        FROZEN_CONTRACT.global_experts,
    )


def _prove_permutation(model_state: Any) -> None:
    experts = FROZEN_CONTRACT.global_experts
    log2phy = model_state.logical_to_physical_map
    counts = model_state.logical_replica_count
    map_shape = tuple(int(dim) for dim in log2phy.shape)
    count_shape = tuple(int(dim) for dim in counts.shape)
    if len(map_shape) != 3 or map_shape[1] != experts or map_shape[2] < 1:
        _fail(
            "M15-EPLB-014",
            f"logical_to_physical_map shape is {map_shape!r}",
            f"(num_moe_layers, {experts}, >=1)",
        )
    if count_shape != (map_shape[0], experts):
        _fail(
            "M15-EPLB-015",
            f"logical_replica_count shape is {count_shape!r}",
            f"({map_shape[0]}, {experts})",
        )
    # One host round trip per rearrangement; the router itself never leaves
    # the device, so this is the only sync the compose introduces.
    replica_counts = counts.tolist()
    primary = log2phy[:, :, 0].tolist()
    for layer_index, (layer_counts, layer_primary) in enumerate(
        zip(replica_counts, primary)
    ):
        offenders = [
            expert
            for expert, value in enumerate(layer_counts)
            if int(value) != 1
        ]
        if offenders:
            _fail(
                "M15-EPLB-016",
                f"moe layer {layer_index} logical experts {offenders[:8]!r} "
                "have replica counts != 1",
                "num_redundant_experts=0, which makes every count exactly 1 "
                "and collapses the router's replica hash to slot 0",
            )
        slots = [int(value) for value in layer_primary]
        if sorted(slots) != list(range(experts)):
            missing = sorted(set(range(experts)) - set(slots))
            _fail(
                "M15-EPLB-017",
                f"moe layer {layer_index} logical_to_physical_map[:, 0] is "
                f"not a permutation of [0, {experts}) "
                f"(missing {missing[:8]!r})",
                "a bijective placement -- a -1 slot would make the kernel "
                "compute dest = -1 / E and write out of bounds",
            )


def reattest_after_rearrangement() -> None:
    """Everything a rearrangement is allowed to change is content, not shape.

    Raises :class:`PF4HActivationError` on any drift.  Failing the worker is
    the correct response: the alternative is a replayed graph reading a stale
    descriptor, which produces wrong numbers silently on every subsequent
    token.
    """

    if not eplb_compose_enabled():
        return
    for model_state in tuple(_MODEL_STATES):
        pointers = (
            int(model_state.logical_to_physical_map.data_ptr()),
            int(model_state.logical_replica_count.data_ptr()),
        )
        expected = _MAP_POINTERS.get(id(model_state))
        if expected is not None and pointers != expected:
            _fail(
                "M15-EPLB-018",
                f"EPLB map pointers moved from {expected!r} to {pointers!r}",
                "in-place map commits (eplb_state.py _commit_eplb_maps uses "
                "dst.copy_); a rebind means the physical expert count "
                "changed",
            )
        _prove_permutation(model_state)
    with _LOCK:
        bound = tuple(_WEIGHT_POINTERS.items())
        tensors = dict(_WEIGHTS)
    for layer_id, pointers in bound:
        live_tensors = tensors.get(layer_id)
        if live_tensors is None:
            continue
        live = tuple(int(tensor.data_ptr()) for tensor in live_tensors)
        if live != pointers:
            _fail(
                "M15-EPLB-019",
                f"layer {layer_id} weight pointers moved from {pointers!r} "
                f"to {live!r} across a rearrangement",
                "in-place weight movement (rebalance_execute.py "
                "move_from_buffer w[dst].copy_)",
            )

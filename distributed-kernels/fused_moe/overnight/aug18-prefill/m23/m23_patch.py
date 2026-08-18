#!/usr/bin/env python3
"""M23 "ragged seal" post-apply patcher for the PF4H/M15 serving arms.

Runs INSIDE the serving container AFTER

    1. pf4h_integration/apply.py        (installs the shim + the V3_M15 patch)
    2. python /covpatch/coverage_patch.py   (the M15_COVERAGE counters)

and BEFORE ``vllm serve``.  Chain order is enforced: this script exits
non-zero if ``PF4H_COVERAGE_PATCH_V1`` is absent from a PF4H-patched
gpu_model_runner.py, so the chain can never silently invert.  (On a plain /
rccl image -- no ``PF4H_INTEGRATION_PATCH_V3_M15`` at all -- both scripts
no-op and exit 0.)

WHAT IT DOES  (design: M23_RAGGED_SEAL_DESIGN.md section 8, edits E1-E7, E10)

  E1/E2  dp_utils.py   widen the existing DP all-reduce staging tensor to
                       (5, dp_size); row 4 carries a per-rank PF4H readiness
                       bit; add ``_post_process_pf4h_ready`` and thread
                       ``pf4h_ready`` / ``return_pf4h_ready`` through
                       ``_synchronize_dp_ranks`` / ``coordinate_batch_across_dp``
                       as purely ADDITIVE keyword arguments (every existing
                       caller keeps its current return arity).
  E3     GMR           ``_pf4h_m23_ragged_enabled`` + ``_pf4h_local_readiness``.
                       The readiness bit carries EVERY local term of the seal,
                       INCLUDING this call's context: a rank inside
                       ``execute_dummy_batch`` / ``_dummy_run`` contributes 0,
                       because it can never seal itself and a 1 there would let
                       its seven peers replay the megakernel graph while it ran
                       stock Mori+AITER eagerly -- a split collective (risk R2).
  E4     GMR           replace the all-ranks-exactly-4096 seal with the
                       DP-unanimous ``b4096_unanimous`` / ``pf4h_ragged_seal``
                       predicate (NO local term: serving-ness reaches the seal
                       only through the all-reduced readiness bit), add the
                       uniform-decode mode override, and assert
                       ``sealed => PIECEWISE`` (risk R3).
  E5     GMR           per-step RAGGED_SEAL_RECEIPT counters (design 8.2).
  E6     shim          activation blocker PF4H-FULL-020 refusing
                       VLLM_MOE_SKIP_PADDING (the megakernel has no -1
                       expert-id sentinel; design 3.6 / risk R5).
  E7     shim + FCTX   docstring / comment rewrites; no logic change.
  E10    shim runtime  receipts record ``rows_compared`` so a whole-tensor
                       nonfinite scan cannot be submitted for a ragged step.

ENV GATES

  VLLM_PF4H_M23_RAGGED           default "1"; "0" makes every patched code
                                 path behave exactly as the pre-M23 tree.
  VLLM_PF4H_B4096_UNIFORM_RESCUE default "1"; "0" disables the arm-symmetric
                                 uniform-decode rescue for steps that did NOT
                                 seal (a sealed step is always rescued, which
                                 is what keeps risk R3 structurally impossible).

The patch is a behaviour no-op for a stock-target server
(``VLLM_PF4H_B4096_GRAPH_TARGET=stock``) except for the uniform-decode rescue,
which the design deliberately lands in BOTH arms for A/B fairness (section 4.3,
risk R6) -- re-baseline before comparing.

Idempotent (marker ``PF4H_M23_RAGGED_SEAL_V1``); fatal on any anchor mismatch.
"""

import os
import sys

MARK = "PF4H_M23_RAGGED_SEAL_V1"
COV_MARK = "PF4H_COVERAGE_PATCH_V1"
PF4H_MARK = "PF4H_INTEGRATION_PATCH_V3_M15"

VLLM_ROOT = "/usr/local/lib/python3.12/dist-packages/vllm"
SHIM_ROOT = VLLM_ROOT + "/model_executor/layers/fused_moe/experts/pf4h_integration"

GMR_PATH = VLLM_ROOT + "/v1/worker/gpu_model_runner.py"
DPU_PATH = VLLM_ROOT + "/v1/worker/dp_utils.py"
FCTX_PATH = VLLM_ROOT + "/forward_context.py"
VLLM_FULL_PATH = SHIM_ROOT + "/vllm_full.py"
M15_VLLM_PATH = SHIM_ROOT + "/m15_vllm.py"
CONTRACTS_PATH = SHIM_ROOT + "/contracts.py"
RUNTIME_PATH = SHIM_ROOT + "/runtime.py"


# --------------------------------------------------------------------------
# generic engine: plain string replacement, anchors counted == 1, fatal on
# mismatch, marker-based idempotency.  Same discipline as coverage_patch.py.
# --------------------------------------------------------------------------
def _apply(path, label, edits, *, required=True):
    """Apply (name, anchor, replacement) edits to ``path``.

    Returns 0 on success (including "already applied" and "absent, skipped"),
    1 on any anchor mismatch.
    """
    if not os.path.exists(path):
        if required:
            print(
                f"m23_patch: FATAL {label} missing at {path}",
                file=sys.stderr,
            )
            return 1
        print(f"m23_patch: {label} absent; skipping")
        return 0
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if MARK in src:
        print(f"m23_patch: {label} already applied ({MARK})")
        return 0
    for name, anchor, _repl in edits:
        n = src.count(anchor)
        if n != 1:
            print(
                f"m23_patch: FATAL {label} anchor '{name}' matched {n} times",
                file=sys.stderr,
            )
            return 1
    for name, anchor, repl in edits:
        src = src.replace(anchor, repl, 1)
    if MARK not in src:
        print(
            f"m23_patch: FATAL {label} produced no {MARK} marker",
            file=sys.stderr,
        )
        return 1
    try:
        compile(src, path, "exec")
    except SyntaxError as exc:  # pragma: no cover - defensive
        print(
            f"m23_patch: FATAL {label} patched source does not parse: {exc}",
            file=sys.stderr,
        )
        return 1
    # Atomic install: write a sibling temp file, fsync it, then rename over the
    # target.  A truncate-in-place write that is killed mid-flight would leave a
    # half-written file that still contains MARK, so a rerun would print
    # "already applied" and the server would die on a SyntaxError instead of
    # being re-patched.  os.replace() within the same directory is atomic, so
    # the file is either wholly pre-patch or wholly post-patch.
    tmp_path = path + ".m23tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(src)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        print(
            f"m23_patch: FATAL {label} could not be written: {exc}",
            file=sys.stderr,
        )
        return 1
    print(f"m23_patch: applied {MARK} to {label}")
    return 0


# ==========================================================================
# E1 + E2 -- dp_utils.py: the readiness bit rides row 4 of the existing
# (4, dp_size) all-reduce.  Zero extra collectives.
# ==========================================================================
DPU_A_RUN_AR = (
    "def _run_ar(\n"
    "    should_ubatch: bool,\n"
    "    orig_num_tokens_per_ubatch: int,\n"
    "    padded_num_tokens_per_ubatch: int,\n"
    "    cudagraph_mode: int,\n"
    "    parallel_config: ParallelConfig,\n"
    ") -> torch.Tensor:\n"
    "    dp_size = parallel_config.data_parallel_size\n"
    "    dp_rank = parallel_config.data_parallel_rank\n"
    "    device, group = _get_device_and_group(parallel_config)\n"
    "    # Populate this rank's contribution on CPU to reduce GPU syncs.\n"
    "    tensor_cpu = torch.zeros(4, dp_size, dtype=torch.int32)\n"
    "    tensor_cpu[0][dp_rank] = orig_num_tokens_per_ubatch\n"
    "    tensor_cpu[1][dp_rank] = padded_num_tokens_per_ubatch\n"
    "    tensor_cpu[2][dp_rank] = 1 if should_ubatch else 0\n"
    "    tensor_cpu[3][dp_rank] = cudagraph_mode\n"
)
DPU_R_RUN_AR = (
    "def _run_ar(\n"
    "    should_ubatch: bool,\n"
    "    orig_num_tokens_per_ubatch: int,\n"
    "    padded_num_tokens_per_ubatch: int,\n"
    "    cudagraph_mode: int,\n"
    "    parallel_config: ParallelConfig,\n"
    "    # " + MARK + ": row 4 carries this rank's PF4H readiness bit.\n"
    "    pf4h_ready: bool = False,\n"
    ") -> torch.Tensor:\n"
    "    dp_size = parallel_config.data_parallel_size\n"
    "    dp_rank = parallel_config.data_parallel_rank\n"
    "    device, group = _get_device_and_group(parallel_config)\n"
    "    # Populate this rank's contribution on CPU to reduce GPU syncs.\n"
    "    tensor_cpu = torch.zeros(5, dp_size, dtype=torch.int32)\n"
    "    tensor_cpu[0][dp_rank] = orig_num_tokens_per_ubatch\n"
    "    tensor_cpu[1][dp_rank] = padded_num_tokens_per_ubatch\n"
    "    tensor_cpu[2][dp_rank] = 1 if should_ubatch else 0\n"
    "    tensor_cpu[3][dp_rank] = cudagraph_mode\n"
    "    tensor_cpu[4][dp_rank] = 1 if pf4h_ready else 0\n"
)

DPU_A_POST = "    return int(tensor[3, :].min().item())\n"
DPU_R_POST = (
    DPU_A_POST
    + "\n"
    "\n"
    "def _post_process_pf4h_ready(tensor: torch.Tensor) -> bool:\n"
    '    """' + MARK + ": true only when EVERY DP rank contributed a\n"
    "    readiness bit of 1 on row 4 of the all-reduce.  The PF4H megakernel is\n"
    "    a collective: a split seal decision deadlocks the group, so the seal\n"
    "    predicate may only consume all-reduced data (design section 4.1).\n"
    '    """\n'
    "    if tensor.shape[0] < 5:\n"
    "        return False\n"
    "    return bool(torch.all(tensor[4] == 1).item())\n"
)

DPU_A_SYNC_SIG = (
    "def _synchronize_dp_ranks(\n"
    "    num_tokens_unpadded: int,\n"
    "    num_tokens_padded: int,\n"
    "    should_attempt_ubatching: bool,\n"
    "    cudagraph_mode: int,\n"
    "    parallel_config: ParallelConfig,\n"
    ") -> tuple[bool, torch.Tensor | None, int, torch.Tensor]:\n"
)
DPU_R_SYNC_SIG = (
    "def _synchronize_dp_ranks(\n"
    "    num_tokens_unpadded: int,\n"
    "    num_tokens_padded: int,\n"
    "    should_attempt_ubatching: bool,\n"
    "    cudagraph_mode: int,\n"
    "    parallel_config: ParallelConfig,\n"
    "    # " + MARK + "\n"
    "    pf4h_ready: bool = False,\n"
    ") -> tuple[bool, torch.Tensor | None, int, torch.Tensor, bool]:\n"
)

DPU_A_SYNC_CALL = (
    "    tensor = _run_ar(\n"
    "        should_ubatch=should_attempt_ubatching,\n"
    "        orig_num_tokens_per_ubatch=num_tokens_unpadded,\n"
    "        padded_num_tokens_per_ubatch=num_tokens_padded,\n"
    "        cudagraph_mode=cudagraph_mode,\n"
    "        parallel_config=parallel_config,\n"
    "    )\n"
)
DPU_R_SYNC_CALL = (
    "    tensor = _run_ar(\n"
    "        should_ubatch=should_attempt_ubatching,\n"
    "        orig_num_tokens_per_ubatch=num_tokens_unpadded,\n"
    "        padded_num_tokens_per_ubatch=num_tokens_padded,\n"
    "        cudagraph_mode=cudagraph_mode,\n"
    "        parallel_config=parallel_config,\n"
    "        pf4h_ready=pf4h_ready,\n"
    "    )\n"
)

DPU_A_SYNC_RET = (
    "    original_num_tokens = tensor[0, :].cpu()\n"
    "    return (\n"
    "        should_ubatch,\n"
    "        num_tokens_after_padding,\n"
    "        synced_cudagraph_mode,\n"
    "        original_num_tokens,\n"
    "    )\n"
)
DPU_R_SYNC_RET = (
    "    original_num_tokens = tensor[0, :].cpu()\n"
    "    # " + MARK + "\n"
    "    pf4h_ready_all = _post_process_pf4h_ready(tensor)\n"
    "    return (\n"
    "        should_ubatch,\n"
    "        num_tokens_after_padding,\n"
    "        synced_cudagraph_mode,\n"
    "        original_num_tokens,\n"
    "        pf4h_ready_all,\n"
    "    )\n"
)

DPU_A_COORD_SIG = (
    "    cudagraph_mode: int = 0,\n"
    "    return_unpadded_counts: bool = False,\n"
    ") -> (\n"
    "    tuple[bool, torch.Tensor | None, int]\n"
    "    | tuple[bool, torch.Tensor | None, int, torch.Tensor]\n"
    "):\n"
)
DPU_R_COORD_SIG = (
    "    cudagraph_mode: int = 0,\n"
    "    return_unpadded_counts: bool = False,\n"
    "    # " + MARK + ": additive keyword arguments only -- every existing\n"
    "    # caller keeps its current signature and return arity.\n"
    "    pf4h_ready: bool = False,\n"
    "    return_pf4h_ready: bool = False,\n"
    ") -> (\n"
    "    tuple[bool, torch.Tensor | None, int]\n"
    "    | tuple[bool, torch.Tensor | None, int, torch.Tensor]\n"
    "    | tuple[bool, torch.Tensor | None, int, torch.Tensor, bool]\n"
    "):\n"
)

DPU_A_COORD_EARLY = (
    "    if parallel_config.data_parallel_size == 1:\n"
    "        # Early exit.\n"
    "        if return_unpadded_counts:\n"
    "            return (\n"
    "                False,\n"
    "                None,\n"
    "                cudagraph_mode,\n"
    "                torch.tensor([num_tokens_unpadded], dtype=torch.int32),\n"
    "            )\n"
    "        return False, None, cudagraph_mode\n"
)
DPU_R_COORD_EARLY = (
    "    if parallel_config.data_parallel_size == 1:\n"
    "        # Early exit.\n"
    "        if return_unpadded_counts:\n"
    "            if return_pf4h_ready:\n"
    "                return (\n"
    "                    False,\n"
    "                    None,\n"
    "                    cudagraph_mode,\n"
    "                    torch.tensor([num_tokens_unpadded], dtype=torch.int32),\n"
    "                    False,\n"
    "                )\n"
    "            return (\n"
    "                False,\n"
    "                None,\n"
    "                cudagraph_mode,\n"
    "                torch.tensor([num_tokens_unpadded], dtype=torch.int32),\n"
    "            )\n"
    "        return False, None, cudagraph_mode\n"
)

DPU_A_COORD_TAIL = (
    "    (\n"
    "        should_ubatch,\n"
    "        num_tokens_after_padding,\n"
    "        synced_cudagraph_mode,\n"
    "        original_num_tokens,\n"
    "    ) = (\n"
    "        _synchronize_dp_ranks(\n"
    "            num_tokens_unpadded,\n"
    "            num_tokens_padded,\n"
    "            should_attempt_ubatching,\n"
    "            cudagraph_mode,\n"
    "            parallel_config,\n"
    "        )\n"
    "    )\n"
    "\n"
    "    if return_unpadded_counts:\n"
    "        return (\n"
    "            should_ubatch,\n"
    "            num_tokens_after_padding,\n"
    "            synced_cudagraph_mode,\n"
    "            original_num_tokens,\n"
    "        )\n"
    "    return (should_ubatch, num_tokens_after_padding, synced_cudagraph_mode)\n"
)
DPU_R_COORD_TAIL = (
    "    (\n"
    "        should_ubatch,\n"
    "        num_tokens_after_padding,\n"
    "        synced_cudagraph_mode,\n"
    "        original_num_tokens,\n"
    "        pf4h_ready_all,\n"
    "    ) = (\n"
    "        _synchronize_dp_ranks(\n"
    "            num_tokens_unpadded,\n"
    "            num_tokens_padded,\n"
    "            should_attempt_ubatching,\n"
    "            cudagraph_mode,\n"
    "            parallel_config,\n"
    "            pf4h_ready=pf4h_ready,\n"
    "        )\n"
    "    )\n"
    "\n"
    "    if return_unpadded_counts:\n"
    "        if return_pf4h_ready:\n"
    "            return (\n"
    "                should_ubatch,\n"
    "                num_tokens_after_padding,\n"
    "                synced_cudagraph_mode,\n"
    "                original_num_tokens,\n"
    "                pf4h_ready_all,\n"
    "            )\n"
    "        return (\n"
    "            should_ubatch,\n"
    "            num_tokens_after_padding,\n"
    "            synced_cudagraph_mode,\n"
    "            original_num_tokens,\n"
    "        )\n"
    "    return (should_ubatch, num_tokens_after_padding, synced_cudagraph_mode)\n"
)

DPU_EDITS = (
    ("run_ar", DPU_A_RUN_AR, DPU_R_RUN_AR),
    ("post_process", DPU_A_POST, DPU_R_POST),
    ("sync_sig", DPU_A_SYNC_SIG, DPU_R_SYNC_SIG),
    ("sync_call", DPU_A_SYNC_CALL, DPU_R_SYNC_CALL),
    ("sync_ret", DPU_A_SYNC_RET, DPU_R_SYNC_RET),
    ("coord_sig", DPU_A_COORD_SIG, DPU_R_COORD_SIG),
    ("coord_early", DPU_A_COORD_EARLY, DPU_R_COORD_EARLY),
    ("coord_tail", DPU_A_COORD_TAIL, DPU_R_COORD_TAIL),
)


# ==========================================================================
# E3 + E4 + E5 -- gpu_model_runner.py.  Anchors below the coverage patch's
# seal region are written against the POST-coverage-patch text.
# ==========================================================================

# ---- E3: the readiness helper, next to _pf4h_graph_operator_enabled -------
GMR_A_HELPER = (
    "    def _pf4h_graph_operator_enabled(self) -> bool:\n"
    '        if os.environ.get("VLLM_PF4H_B4096_GRAPH_TARGET") != "pf4h":\n'
    "            return False\n"
)
GMR_R_HELPER = (
    "    def _pf4h_m23_ragged_enabled(self) -> bool:\n"
    '        """' + MARK + ": is the M23 ragged seal in force for this server?\n"
    "\n"
    "        Setting VLLM_PF4H_M23_RAGGED=0 reverts every M23 code path to the\n"
    "        pre-M23 all-ranks-exactly-4096 behaviour.\n"
    '        """\n'
    "        return (\n"
    '            os.environ.get("VLLM_PF4H_M23_RAGGED", "1") != "0"\n'
    '            and os.environ.get("VLLM_PF4H_INTEGRATION_MODE")\n'
    '            in ("full", "m15")\n'
    "            and self.parallel_config.data_parallel_size == 8\n"
    "        )\n"
    "\n"
    "    def _pf4h_local_readiness(\n"
    "        self,\n"
    "        *,\n"
    "        has_lora: bool,\n"
    "        pf4h_graph_target: bool | None,\n"
    "        force_uniform_decode: bool | None,\n"
    "        force_num_active_loras: int | None,\n"
    "    ) -> bool:\n"
    '        """' + MARK + ": this rank's contribution to the DP readiness bit.\n"
    "\n"
    "        Must be computable BEFORE the all-reduce, and must fold in every\n"
    "        LOCAL term that could otherwise split the seal decision across\n"
    "        ranks (design section 4.1/4.3, risk R2).\n"
    "\n"
    "        THE CALL CONTEXT IS ONE OF THOSE TERMS.  The first three conjuncts\n"
    "        below are byte-identical to ``m23_serving``'s, and they have to be:\n"
    "        the DP engine's idle-lockstep path (v1/engine/core.py's busy loop ->\n"
    "        gpu_worker.execute_dummy_batch -> _dummy_run(uniform_decode=True))\n"
    "        drives a rank with no scheduled work through THIS SAME all-reduce\n"
    "        while its seven peers are inside execute_model.  Such a rank can\n"
    "        never seal -- it takes the pre-M23 branch and ends at\n"
    "        CUDAGraphMode.NONE -- so if it contributed a readiness bit of 1 the\n"
    "        other seven would seal and replay the PF4H megakernel graph while it\n"
    "        ran stock Mori+AITER eagerly at 4096 padded tokens.  That is a SPLIT\n"
    "        COLLECTIVE: the seven spin to M15_SPIN_LIMIT and fail closed, the\n"
    "        eighth's all2all waits on peers that never dispatch.  Contributing 0\n"
    "        makes the whole group refuse in lockstep, and the refusal shows up as\n"
    "        refused_not_ready in RAGGED_SEAL_RECEIPT (probe P7).\n"
    "\n"
    "        The activation-file latch is polled every step until it latches --\n"
    "        one os.path.isfile per step per rank -- which is what removes the\n"
    "        split-brain window.\n"
    '        """\n'
    "        if (\n"
    "            pf4h_graph_target is not None\n"
    "            or force_uniform_decode is not None\n"
    "            or force_num_active_loras is not None\n"
    "        ):\n"
    "            # Not a serving step: the capture drive, or any _dummy_run\n"
    "            # (warmup, capture, profile, or the DP idle-lockstep dummy\n"
    "            # batch).  Every one of them passes the force_* overrides.\n"
    "            return False\n"
    "        if not self._pf4h_m23_ragged_enabled():\n"
    "            return False\n"
    '        if os.environ.get("VLLM_PF4H_B4096_GRAPH_TARGET") != "pf4h":\n'
    "            # A stock-target server never contributes readiness.\n"
    "            return False\n"
    "        if has_lora or self.vllm_config.lora_config is not None:\n"
    "            return False\n"
    "        if envs.VLLM_MOE_SKIP_PADDING:\n"
    "            # design section 3.6: the megakernel has no -1 expert-id\n"
    "            # sentinel, so padded rows must never be masked to -1.\n"
    "            return False\n"
    "        return self._pf4h_graph_operator_enabled()\n"
    "\n"
    + GMR_A_HELPER
)

# ---- E4a: readiness in, unanimity out, around the DP all-reduce -----------
GMR_A_DP = (
    "        # Extra coordination when running data-parallel since we need to coordinate\n"
    "        # across ranks\n"
    "        should_ubatch, num_tokens_across_dp = False, None\n"
    "        original_num_tokens_across_dp = None\n"
    "        if self.vllm_config.parallel_config.data_parallel_size > 1:\n"
    "            (\n"
    "                should_ubatch,\n"
    "                num_tokens_across_dp,\n"
    "                synced_cudagraph_mode,\n"
    "                original_num_tokens_across_dp,\n"
    "            ) = (\n"
    "                coordinate_batch_across_dp(\n"
    "                    num_tokens_unpadded=num_tokens,\n"
    "                    parallel_config=self.parallel_config,\n"
    "                    allow_microbatching=allow_microbatching,\n"
    "                    num_tokens_padded=num_tokens_padded,\n"
    "                    uniform_decode=uniform_decode,\n"
    "                    cudagraph_mode=cudagraph_mode.value,\n"
    "                    return_unpadded_counts=True,\n"
    "                )\n"
    "            )\n"
)
GMR_R_DP = (
    "        # Extra coordination when running data-parallel since we need to coordinate\n"
    "        # across ranks\n"
    "        should_ubatch, num_tokens_across_dp = False, None\n"
    "        original_num_tokens_across_dp = None\n"
    "        # " + MARK + ": the readiness bit must be computable BEFORE the\n"
    "        # all-reduce so it can ride row 4 of the existing staging tensor.\n"
    "        # It carries EVERY local term of the seal decision, including this\n"
    "        # call's context -- see _pf4h_local_readiness for why a _dummy_run\n"
    "        # rank contributing 1 would split the collective (risk R2).\n"
    "        synced_cudagraph_mode = cudagraph_mode.value\n"
    "        pf4h_ready_all = False\n"
    "        pf4h_local_ready = self._pf4h_local_readiness(\n"
    "            has_lora=has_lora,\n"
    "            pf4h_graph_target=pf4h_graph_target,\n"
    "            force_uniform_decode=force_uniform_decode,\n"
    "            force_num_active_loras=force_num_active_loras,\n"
    "        )\n"
    "        if self.vllm_config.parallel_config.data_parallel_size > 1:\n"
    "            (\n"
    "                should_ubatch,\n"
    "                num_tokens_across_dp,\n"
    "                synced_cudagraph_mode,\n"
    "                original_num_tokens_across_dp,\n"
    "                pf4h_ready_all,\n"
    "            ) = (\n"
    "                coordinate_batch_across_dp(\n"
    "                    num_tokens_unpadded=num_tokens,\n"
    "                    parallel_config=self.parallel_config,\n"
    "                    allow_microbatching=allow_microbatching,\n"
    "                    num_tokens_padded=num_tokens_padded,\n"
    "                    uniform_decode=uniform_decode,\n"
    "                    cudagraph_mode=cudagraph_mode.value,\n"
    "                    return_unpadded_counts=True,\n"
    "                    pf4h_ready=pf4h_local_ready,\n"
    "                    return_pf4h_ready=True,\n"
    "                )\n"
    "            )\n"
)

# ---- E4b: the new predicate (anchored on POST-coverage-patch text) --------
GMR_A_PREDICATE = (
    '                    _cov["fail_min_tok"] = min(\n'
    '                        _cov["fail_min_tok"], min(original_counts)\n'
    "                    )\n"
    "\n"
    "        if pf4h_graph_target is not None:\n"
    "            if pf4h_graph_target and not pf4h_exact_b4096:\n"
    "                raise RuntimeError(\n"
    '                    "PF4H graph capture requested without exact DP8 B4096 inputs"\n'
    "                )\n"
    "            pf4h_exact_b4096 = pf4h_graph_target\n"
    "        elif pf4h_exact_b4096:\n"
    "            pf4h_exact_b4096 = self._pf4h_graph_operator_enabled()\n"
)
GMR_R_PREDICATE = (
    '                    _cov["fail_min_tok"] = min(\n'
    '                        _cov["fail_min_tok"], min(original_counts)\n'
    "                    )\n"
    "\n"
    "        # " + MARK + ": the ragged seal (design section 4.2).\n"
    "        #\n"
    "        # A *serving* step is one execute_model drives.  The capture drive\n"
    "        # (pf4h_graph_target is not None) and every _dummy_run (which always\n"
    "        # passes the force_* overrides) keep the pre-M23 exact behaviour, so\n"
    "        # capture stays exact-4096 on all ranks (design section 4.4 row 6)\n"
    "        # and _dummy_run's cudagraph_runtime_mode assertion cannot trip.\n"
    "        #\n"
    "        # m23_serving is a LOCAL property, so it must never appear in the\n"
    "        # seal predicate (risk R2).  Serving-ness reaches the seal only\n"
    "        # through the all-reduced readiness bit, which _pf4h_local_readiness\n"
    "        # zeroes for exactly these same call contexts.  Below, m23_serving\n"
    "        # therefore gates only branch selection, the mode rescue, the R3\n"
    "        # assertion and the counters -- never pf4h_ragged_seal.\n"
    "        m23_serving = (\n"
    "            pf4h_graph_target is None\n"
    "            and force_uniform_decode is None\n"
    "            and force_num_active_loras is None\n"
    "            and self._pf4h_m23_ragged_enabled()\n"
    "        )\n"
    "        m23_orig_counts: tuple[int, ...] = ()\n"
    "        if original_num_tokens_across_dp is not None:\n"
    "            m23_orig_counts = tuple(\n"
    "                int(value) for value in original_num_tokens_across_dp.tolist()\n"
    "            )\n"
    "        m23_exact = m23_orig_counts == (4096,) * 8\n"
    "        m23_local_b4096 = batch_descriptor.num_tokens == 4096\n"
    "        # Every term below is DP-unanimous by construction:\n"
    "        # _pf4h_m23_ragged_enabled() is env x env x dp_size (identical on\n"
    "        # every rank of the deployment), should_ubatch is torch.all() over\n"
    "        # the all-reduce, synced_cudagraph_mode is the min across ranks, and\n"
    "        # num_tokens_across_dp is [max]*8 whenever that synced mode is not\n"
    "        # NONE (dp_utils._post_process_dp_padding).  Requiring PIECEWISE is\n"
    "        # what kills hazard H2: it proves DP padding actually happened, so\n"
    "        # all(==4096) is a statement about the GROUP.  Note the deliberate\n"
    "        # absence of m23_serving: a rank in _dummy_run computes the same\n"
    "        # b4096_unanimous its peers do, and it is its readiness bit -- not\n"
    "        # this boolean -- that refuses the seal for everybody.\n"
    "        b4096_unanimous = bool(\n"
    "            self._pf4h_m23_ragged_enabled()\n"
    "            and not should_ubatch\n"
    "            and synced_cudagraph_mode == CUDAGraphMode.PIECEWISE.value\n"
    "            and num_tokens_across_dp is not None\n"
    "            and all(int(v) == 4096 for v in num_tokens_across_dp.tolist())\n"
    "        )\n"
    "        # pf4h_ready_all is torch.all() over row 4 of the same all-reduce,\n"
    "        # so pf4h_ragged_seal is identical on all eight ranks.  It also\n"
    "        # implies m23_serving on every rank: a rank that is not serving\n"
    "        # contributes 0, which drives pf4h_ready_all False group-wide.\n"
    "        pf4h_ragged_seal = bool(b4096_unanimous and pf4h_ready_all)\n"
    "        m23_uniform_rescued = False\n"
    "\n"
    "        if pf4h_graph_target is not None:\n"
    "            if pf4h_graph_target and not pf4h_exact_b4096:\n"
    "                raise RuntimeError(\n"
    '                    "PF4H graph capture requested without exact DP8 B4096 inputs"\n'
    "                )\n"
    "            pf4h_exact_b4096 = pf4h_graph_target\n"
    "        elif m23_serving:\n"
    "            # " + MARK + ": the per-rank activation-file latch is folded\n"
    "            # into the DP-unanimous readiness bit, so no LOCAL term survives\n"
    "            # in the seal decision (design section 4.1, risk R2).\n"
    "            pf4h_exact_b4096 = pf4h_ragged_seal\n"
    "        elif pf4h_exact_b4096:\n"
    "            pf4h_exact_b4096 = self._pf4h_graph_operator_enabled()\n"
)

# ---- E4c: the uniform-decode mode override -------------------------------
GMR_A_RESCUE = (
    "        if (\n"
    "            batch_descriptor.num_tokens == 4096\n"
    '            and os.environ.get("VLLM_PF4H_B4096_GRAPH_TARGET") == "pf4h"\n'
    "            and not pf4h_exact_b4096\n"
    "        ):\n"
    "            # A PF4H-only server has no ordinary B4096 graph. Ragged or\n"
    "            # not-yet-activated large prefill must therefore use the stock\n"
    "            # operators eagerly instead of looking up an absent graph key.\n"
    "            cudagraph_mode = CUDAGraphMode.NONE\n"
)
GMR_R_RESCUE = (
    "        # " + MARK + ": the uniform-decode rescue (design section 4.3,\n"
    "        # hazard H1).  A rank whose LOCAL batch looked like a uniform decode\n"
    "        # -- or that otherwise missed the dispatcher's early B4096 branch --\n"
    "        # landed on CUDAGraphMode.NONE even though DP padding put it in\n"
    "        # exactly the same padded B4096 bucket as its seven peers, and ran\n"
    "        # the whole model eagerly at 4096 tokens.  Route it to the same\n"
    "        # PIECEWISE B4096 graph the other ranks use; BatchDescriptor(4096)\n"
    "        # is field-identical to the dispatcher's regular_b4096 key, so this\n"
    "        # needs no new key and no new capture.\n"
    "        #\n"
    "        # Deliberately arm-symmetric: the stock arm gets the same rescue so\n"
    "        # the A/B stays fair (risk R6) -- RE-BASELINE before comparing.  It\n"
    "        # is forced whenever the step actually sealed, because a sealed step\n"
    "        # that ran eagerly would trip M15-DESC-012 mid-serve (risk R3).\n"
    "        #\n"
    "        # Gated on m23_serving so it can never rewrite the mode inside a\n"
    "        # _dummy_run, where _dummy_run's own\n"
    "        # `assert cudagraph_runtime_mode == _cudagraph_mode` would trip.  A\n"
    "        # non-unanimous RESCUE is harmless (it only picks stock-graph replay\n"
    "        # over eager, which is exactly today's behaviour on that rank); a\n"
    "        # non-unanimous SEAL is not, and the seal is not gated here.  Risk\n"
    "        # R3 still holds: pf4h_ragged_seal implies every rank contributed a\n"
    "        # readiness bit of 1, which implies m23_serving on every rank.\n"
    "        if (\n"
    "            b4096_unanimous\n"
    "            and m23_serving\n"
    "            and cudagraph_mode != CUDAGraphMode.PIECEWISE\n"
    "            and (\n"
    "                pf4h_ragged_seal\n"
    '                or os.environ.get("VLLM_PF4H_B4096_UNIFORM_RESCUE", "1")\n'
    '                != "0"\n'
    "            )\n"
    "        ):\n"
    "            cudagraph_mode = CUDAGraphMode.PIECEWISE\n"
    "            batch_descriptor = BatchDescriptor(num_tokens=4096)\n"
    "            m23_uniform_rescued = True\n"
    "\n"
    + GMR_A_RESCUE
)

# ---- E5: the RAGGED_SEAL_RECEIPT counters --------------------------------
GMR_A_RECEIPT = (
    "        if pf4h_graph_target is None and _cov[\"steps\"] % 100 == 0:\n"
    "            logger.info(\n"
    '                "M15_COVERAGE steps=%d tok=%d b4096_steps=%d "\n'
    '                "b4096_tok=%d in_bucket=%d unanimity_fail=%d "\n'
    '                "fail_min_tok=%d",\n'
    '                _cov["steps"], _cov["tok"], _cov["b4096_steps"],\n'
    '                _cov["b4096_steps"] * 4096, _cov["in_bucket"],\n'
    '                _cov["unanimity_fail"], _cov["fail_min_tok"],\n'
    "            )\n"
)
GMR_R_RECEIPT = (
    GMR_A_RECEIPT
    + "\n"
    "        # " + MARK + ": risk R3 -- a sealed step must be a graph replay on\n"
    "        # all eight ranks.  A sealed step that ran the mega eagerly would\n"
    "        # let torch.empty_like hand back an uncaptured output pointer and\n"
    "        # m15_runtime would refuse with M15-DESC-012 mid-serve.\n"
    "        if m23_serving and pf4h_exact_b4096:\n"
    "            assert cudagraph_mode == CUDAGraphMode.PIECEWISE, (\n"
    '                "PF4H M23: a sealed B4096 step is not PIECEWISE "\n'
    '                f"(mode={cudagraph_mode!r})"\n'
    "            )\n"
    "\n"
    "        # " + MARK + ": per-step coverage counters (design section 8.2).\n"
    "        # Success criterion: sealed == in_bucket and eager_b4096 == 0 on\n"
    "        # every rank.\n"
    "        if m23_serving:\n"
    '            _m23 = getattr(self, "_pf4h_m23", None)\n'
    "            if _m23 is None:\n"
    "                _m23 = self._pf4h_m23 = {\n"
    '                    "steps": 0, "in_bucket": 0, "sealed": 0,\n'
    '                    "sealed_exact": 0, "sealed_ragged": 0,\n'
    '                    "refused_not_unanimous": 0, "refused_not_ready": 0,\n'
    '                    "refused_peer_not_ready": 0,\n'
    '                    "eager_b4096": 0, "uniform_rescued": 0,\n'
    '                    "min_orig": 1 << 30, "max_orig": 0, "sum_orig": 0,\n'
    '                    "in_bucket_sum_orig": 0,\n'
    "                }\n"
    "                # design 8.2: also emit once at shutdown, so the last\n"
    "                # partial window of a run is never lost.  Guarded: at\n"
    "                # interpreter teardown the logging handlers may already\n"
    "                # be closed, and a receipt is not worth an exit traceback.\n"
    "                import atexit as _m23_atexit\n"
    "\n"
    "                def _m23_final(\n"
    "                    _c=_m23,\n"
    "                    _rank=self.parallel_config.data_parallel_rank,\n"
    "                ):\n"
    "                    try:\n"
    "                        logger.warning(\n"
    '                            "RAGGED_SEAL_RECEIPT_FINAL rank=%d %s",\n'
    "                            _rank,\n"
    '                            " ".join(\n'
    '                                f"{k}={v}" for k, v in _c.items()\n'
    "                            ),\n"
    "                        )\n"
    "                    except Exception:\n"
    "                        pass\n"
    "\n"
    "                _m23_atexit.register(_m23_final)\n"
    '            _m23["steps"] += 1\n'
    "            if m23_orig_counts:\n"
    '                _m23["min_orig"] = min(\n'
    '                    _m23["min_orig"], min(m23_orig_counts)\n'
    "                )\n"
    '                _m23["max_orig"] = max(\n'
    '                    _m23["max_orig"], max(m23_orig_counts)\n'
    "                )\n"
    '                _m23["sum_orig"] += int(sum(m23_orig_counts))\n'
    "            if b4096_unanimous:\n"
    '                _m23["in_bucket"] += 1\n'
    '                _m23["in_bucket_sum_orig"] += int(sum(m23_orig_counts))\n'
    "                if not pf4h_ready_all:\n"
    '                    _m23["refused_not_ready"] += 1\n'
    "                    if pf4h_local_ready:\n"
    "                        # This rank was ready, so the refusal came from a\n"
    "                        # PEER: either its activation file has not latched\n"
    "                        # yet, or it is inside execute_dummy_batch for this\n"
    "                        # step (the DP idle-lockstep path).  Both are\n"
    "                        # correct group-wide refusals; the second is the\n"
    "                        # class that would be a split collective if the\n"
    "                        # readiness bit ignored the call context.\n"
    '                        _m23["refused_peer_not_ready"] += 1\n'
    "                if cudagraph_mode == CUDAGraphMode.NONE:\n"
    '                    _m23["eager_b4096"] += 1\n'
    "            elif m23_local_b4096:\n"
    '                _m23["refused_not_unanimous"] += 1\n'
    "            if m23_uniform_rescued and (\n"
    "                cudagraph_mode == CUDAGraphMode.PIECEWISE\n"
    "            ):\n"
    '                _m23["uniform_rescued"] += 1\n'
    "            if pf4h_exact_b4096:\n"
    '                _m23["sealed"] += 1\n'
    "                if m23_exact:\n"
    '                    _m23["sealed_exact"] += 1\n'
    "                else:\n"
    '                    _m23["sealed_ragged"] += 1\n'
    '            if _m23["steps"] % 100 == 0:\n'
    "                logger.info(\n"
    '                    "RAGGED_SEAL_RECEIPT rank=%d steps=%d in_bucket=%d "\n'
    '                    "sealed=%d sealed_ragged=%d sealed_exact=%d "\n'
    '                    "refused_not_unanimous=%d refused_not_ready=%d "\n'
    '                    "refused_peer_not_ready=%d "\n'
    '                    "eager_b4096=%d uniform_rescued=%d min_orig=%d "\n'
    '                    "max_orig=%d sum_orig=%d in_bucket_sum_orig=%d",\n'
    "                    self.parallel_config.data_parallel_rank,\n"
    '                    _m23["steps"], _m23["in_bucket"], _m23["sealed"],\n'
    '                    _m23["sealed_ragged"], _m23["sealed_exact"],\n'
    '                    _m23["refused_not_unanimous"],\n'
    '                    _m23["refused_not_ready"],\n'
    '                    _m23["refused_peer_not_ready"], _m23["eager_b4096"],\n'
    '                    _m23["uniform_rescued"], _m23["min_orig"],\n'
    '                    _m23["max_orig"], _m23["sum_orig"],\n'
    '                    _m23["in_bucket_sum_orig"],\n'
    "                )\n"
)

GMR_EDITS = (
    ("helper", GMR_A_HELPER, GMR_R_HELPER),
    ("dp_coordination", GMR_A_DP, GMR_R_DP),
    ("predicate", GMR_A_PREDICATE, GMR_R_PREDICATE),
    ("uniform_rescue", GMR_A_RESCUE, GMR_R_RESCUE),
    ("receipt", GMR_A_RECEIPT, GMR_R_RECEIPT),
)


# ==========================================================================
# E7 -- forward_context.py: BatchDescriptor.pf4h_exact_b4096 docstring.
# The FIELD NAME is load-bearing (graph-key identity, apply.py sentinel) and
# is deliberately kept.
# ==========================================================================
FCTX_A = (
    "    pf4h_exact_b4096: bool = False\n"
    '    """\n'
    "    True only for the PF4H PIECEWISE graph whose DP coordination proved that\n"
    "    every rank had exactly 4096 original live tokens before graph padding.\n"
    '    """\n'
)
FCTX_R = (
    "    pf4h_exact_b4096: bool = False\n"
    '    """\n'
    "    True only for the PF4H PIECEWISE graph whose DP coordination proved that\n"
    "    all eight ranks are in the same padded B4096 bucket.  " + MARK + ":\n"
    "    the per-rank *original* counts may be ragged -- production pads every\n"
    "    batch in (max_cudagraph_capture_size, 4096] up to 4096 and dispatches\n"
    "    all 4096 rows, so the padded bucket is the honest unit of agreement.\n"
    "    The field name is unchanged: it is what makes this graph key\n"
    "    structurally distinct from the ordinary B4096 key.\n"
    '    """\n'
)

FCTX_EDITS = (("descriptor_doc", FCTX_A, FCTX_R),)


# ==========================================================================
# E6 + E7 -- shim vllm_full.py.  These anchors sit outside the region
# coverage_patch.py rewrites (the PF4H-FULL-018 EPLB blocker).
# ==========================================================================
FULL_A_DOC = (
    '"""Exact vLLM 0.25.1 hybrid PF4H seam.\n'
    "\n"
    "Only a non-ragged prefill bucket with exactly 4096 original live tokens on all\n"
    "eight DP/EP ranks takes the PF4H path. The exact vLLM graph-dispatch patch gives\n"
    "that case a distinct graph key; every decode, other prefill size, padded B4096\n"
    "graph, and unattested invocation retains stock Mori+AITER.\n"
    '"""\n'
)
FULL_R_DOC = (
    '"""Exact vLLM 0.25.1 hybrid PF4H seam.\n'
    "\n"
    + MARK + ": the PF4H path takes a prefill step exactly when all eight DP/EP\n"
    "ranks are in the same padded B4096 bucket; the per-rank *original* counts\n"
    "may be ragged.  Production pads every batch in\n"
    "(max_cudagraph_capture_size, 4096] up to 4096 and dispatches all 4096 rows\n"
    "through Mori+AITER unmasked, so the megakernel dispatching the same 4096\n"
    "rows is parity, not a divergence.  The vLLM graph-dispatch patch gives that\n"
    "case a distinct graph key; every decode, other prefill size, non-unanimous\n"
    "batch, and unattested invocation retains stock Mori+AITER.\n"
    '"""\n'
)

FULL_A_ATTEST = (
    "        # The exact graph patch adds this key only after DP coordination proves\n"
    "        # every rank's *original*, pre-padding count is 4096. This distinction\n"
    "        # cannot be reconstructed from DPMetadata, which intentionally carries\n"
    "        # post-padding counts for graph replay.\n"
)
FULL_R_ATTEST = (
    "        # " + MARK + ": the runner stamps this key only after DP\n"
    "        # coordination proves all eight ranks are in the same PADDED B4096\n"
    "        # bucket.  The check below is already written against DPMetadata's\n"
    "        # post-padding counts, so it is ragged-correct as-is and needs no\n"
    "        # math change -- the pre-padding counts are telemetry only.\n"
)

FULL_A_ELIGIBLE = (
    "            # A non-uniform B4096 CUDA graph may later replay for a smaller\n"
    "            # live batch padded to the same descriptor. The exact runner patch\n"
    "            # prevents that graph-key collision before this predicate runs.\n"
)
FULL_R_ELIGIBLE = (
    "            # " + MARK + ": a non-uniform B4096 CUDA graph does replay for\n"
    "            # smaller live batches padded to the same descriptor -- that is\n"
    "            # the point.  Graph-key collision with the ordinary B4096 key is\n"
    "            # prevented structurally by the distinct pf4h_exact_b4096\n"
    "            # descriptor field, not by the batch being non-ragged.  Every\n"
    "            # shape gate below holds for a ragged batch because the tensors\n"
    "            # themselves are padded to 4096 rows.\n"
)

FULL_A_020 = (
    '    activation_file = os.environ.get(ACTIVATION_FILE_ENV, "")\n'
    "    if not activation_file or not Path(activation_file).is_absolute():\n"
    "        blockers.append(\n"
    "            ActivationBlocker(\n"
    '                "PF4H-FULL-019",\n'
    '                f"{ACTIVATION_FILE_ENV}={activation_file!r}",\n'
    '                "an absolute sentinel path created only after vLLM is ready",\n'
    "            )\n"
    "        )\n"
)
FULL_R_020 = (
    FULL_A_020
    + "    # " + MARK + " (design section 3.6, risk R5): with the ragged seal\n"
    "    # the megakernel routinely sees DP/cudagraph padded rows.\n"
    "    # VLLM_MOE_SKIP_PADDING makes modular_kernel force those rows'\n"
    "    # expert ids to -1, and the megakernel has NO -1 sentinel: qpush\n"
    "    # computes dest = -1 and peer_ptr() then indexes heap_bases[-1], an\n"
    "    # out-of-bounds read followed by a remote atomic into a garbage\n"
    "    # pointer.  Refuse activation rather than corrupt a peer heap.\n"
    "    try:\n"
    "        from vllm import envs as _vllm_envs\n"
    "\n"
    "        _skip_padding = bool(\n"
    '            getattr(_vllm_envs, "VLLM_MOE_SKIP_PADDING", False)\n'
    "        )\n"
    "    except Exception:  # pragma: no cover - envs is always importable\n"
    '        _skip_padding = os.environ.get("VLLM_MOE_SKIP_PADDING", "0") not in (\n'
    '            "",\n'
    '            "0",\n'
    "        )\n"
    "    if _skip_padding:\n"
    "        blockers.append(\n"
    "            ActivationBlocker(\n"
    '                "PF4H-FULL-020",\n'
    '                "VLLM_MOE_SKIP_PADDING forces padded-row expert ids to -1",\n'
    '                "VLLM_MOE_SKIP_PADDING=0 (the megakernel has no -1 "\n'
    '                "expert-id sentinel)",\n'
    "            )\n"
    "        )\n"
)

FULL_EDITS = (
    ("module_doc", FULL_A_DOC, FULL_R_DOC),
    ("attest_comment", FULL_A_ATTEST, FULL_R_ATTEST),
    ("eligible_comment", FULL_A_ELIGIBLE, FULL_R_ELIGIBLE),
    ("full_020", FULL_A_020, FULL_R_020),
)


# ==========================================================================
# E7 -- shim m15_vllm.py + contracts.py docstrings/comments.
# ==========================================================================
M15V_A = (
    "Everything about *when* the accelerated path is legal is inherited verbatim\n"
    "from :class:`~.vllm_full.PF4HPrepareAndFinalize`: the same non-ragged\n"
    "exact-B4096 DP proof, the same contiguous-EP8 expert-map attestation, the same\n"
    "activation sentinel, the same single-stream ownership rule.  Only *what* runs\n"
    "changes -- six pinned launches become one ``k0pf6gm_m15_mega`` launch driven by\n"
    "a device-resident descriptor.\n"
)
M15V_R = (
    "Everything about *when* the accelerated path is legal is inherited verbatim\n"
    "from :class:`~.vllm_full.PF4HPrepareAndFinalize`: the same padded-B4096 DP\n"
    "proof (" + MARK + ": all eight ranks in the same padded bucket, original\n"
    "counts may be ragged), the same contiguous-EP8 expert-map attestation, the\n"
    "same activation sentinel, the same single-stream ownership rule.  Only *what*\n"
    "runs changes -- six pinned launches become one ``k0pf6gm_m15_mega`` launch\n"
    "driven by a device-resident descriptor.\n"
)

M15V_EDITS = (("module_doc", M15V_A, M15V_R),)

CONTRACTS_A = (
    "# The only PF4H serving bucket proven by the isolation+N2 graph gate.  Smaller\n"
    "# prefill buckets, decode, and ragged DP batches stay on stock Mori+AITER.\n"
)
CONTRACTS_R = (
    "# The only PF4H serving bucket proven by the isolation+N2 graph gate.  Smaller\n"
    "# prefill buckets and decode stay on stock Mori+AITER.\n"
    "# " + MARK + ": DP batches whose per-rank ORIGINAL counts are ragged now\n"
    "# take this bucket too, because DP padding puts every rank at exactly\n"
    "# tokens_per_rank and every capacity below is derived from world x\n"
    "# tokens_per_rank, i.e. worst case -- raggedness can only reduce arrivals.\n"
)

CONTRACTS_EDITS = (("bucket_comment", CONTRACTS_A, CONTRACTS_R),)


# ==========================================================================
# E10 -- shim runtime.py: receipts must record the compared row window so a
# whole-tensor nonfinite scan cannot be submitted for a ragged step (5.3/R4).
# ==========================================================================
RT_A_IMPORT = "import hashlib\nfrom dataclasses import dataclass\n"
RT_R_IMPORT = (
    "import hashlib\n"
    "import os  # " + MARK + "\n"
    "from dataclasses import dataclass\n"
)

RT_A_WEIGHT_FIELDS = (
    "    offline_equivalence_layers: tuple[int, ...]\n"
    "    max_abs: float\n"
    "    rel_l2: float\n"
    "    nonfinite: int\n"
    "\n"
    "    def validate(self) -> None:\n"
    "        c = FROZEN_CONTRACT\n"
    "        blockers = []\n"
)
RT_R_WEIGHT_FIELDS = (
    "    offline_equivalence_layers: tuple[int, ...]\n"
    "    max_abs: float\n"
    "    rel_l2: float\n"
    "    nonfinite: int\n"
    "    # " + MARK + ": number of leading rows the comparison covered.\n"
    "    rows_compared: int | None = None\n"
    "\n"
    "    def validate(self) -> None:\n"
    "        c = FROZEN_CONTRACT\n"
    "        blockers = []\n"
    "        blockers.extend(_m23_row_window_blockers(self, \"PF4H-WEIGHT-005\"))\n"
)

RT_A_CORRECTNESS = (
    "@dataclass(frozen=True)\n"
    "class CorrectnessReceipt:\n"
    "    graph_vs_graph: bool\n"
    "    same_session_stock_baseline: bool\n"
    "    max_abs: float\n"
    "    rel_l2: float\n"
    "    nonfinite: int\n"
    "\n"
    "    def validate(self) -> None:\n"
    "        c = FROZEN_CONTRACT\n"
    "        if (\n"
)
RT_R_CORRECTNESS = (
    "def _m23_ragged_seal_active() -> bool:\n"
    '    """' + MARK + ": is this server serving ragged padded B4096 steps?\n"
    "\n"
    "    Only then is a whole-tensor nonfinite scan meaningless: rows beyond the\n"
    "    live token count carry stale embeddings plus attention-buffer residue\n"
    "    and can legitimately be non-finite in BOTH arms (design section 5.3).\n"
    '    """\n'
    "    return (\n"
    '        os.environ.get("VLLM_PF4H_M23_RAGGED", "1") != "0"\n'
    '        and os.environ.get("VLLM_PF4H_B4096_GRAPH_TARGET") == "pf4h"\n'
    "    )\n"
    "\n"
    "\n"
    "def _m23_row_window_blockers(receipt, code: str):\n"
    '    """' + MARK + ": require an auditable ``rows_compared``.\n"
    "\n"
    "    A ragged serving step's padded rows are discarded by every consumer\n"
    "    (sampling, pooling, prompt logprobs, drafter, KV cache), so evidence\n"
    "    must be restricted to rows [0, n_orig) and must SAY so.\n"
    '    """\n'
    "    rows = getattr(receipt, \"rows_compared\", None)\n"
    "    if not _m23_ragged_seal_active():\n"
    "        return ()\n"
    "    if rows is None:\n"
    "        return (\n"
    "            ActivationBlocker(\n"
    "                code,\n"
    '                "the receipt does not record rows_compared, so a "\n'
    '                "whole-tensor scan over padded rows cannot be ruled out",\n'
    '                "rows_compared=n_orig with the comparison restricted to "\n'
    '                "rows [0, n_orig)",\n'
    "            ),\n"
    "        )\n"
    "    if int(rows) <= 0:\n"
    "        return (\n"
    "            ActivationBlocker(\n"
    "                code,\n"
    "                f\"the receipt records rows_compared={rows!r}\",\n"
    '                "a positive live-row count",\n'
    "            ),\n"
    "        )\n"
    "    return ()\n"
    "\n"
    "\n"
    "@dataclass(frozen=True)\n"
    "class CorrectnessReceipt:\n"
    "    graph_vs_graph: bool\n"
    "    same_session_stock_baseline: bool\n"
    "    max_abs: float\n"
    "    rel_l2: float\n"
    "    nonfinite: int\n"
    "    # " + MARK + ": number of leading rows the comparison covered.\n"
    "    rows_compared: int | None = None\n"
    "\n"
    "    def validate(self) -> None:\n"
    "        c = FROZEN_CONTRACT\n"
    "        m23_blockers = _m23_row_window_blockers(\n"
    '            self, "PF4H-CORRECTNESS-002"\n'
    "        )\n"
    "        if m23_blockers:\n"
    "            raise PF4HActivationError(tuple(m23_blockers))\n"
    "        if (\n"
)

RT_EDITS = (
    ("os_import", RT_A_IMPORT, RT_R_IMPORT),
    ("correctness", RT_A_CORRECTNESS, RT_R_CORRECTNESS),
    ("weight_fields", RT_A_WEIGHT_FIELDS, RT_R_WEIGHT_FIELDS),
)


# --------------------------------------------------------------------------
def main() -> int:
    if not os.path.exists(GMR_PATH):
        print(f"m23_patch: FATAL no vLLM at {GMR_PATH}", file=sys.stderr)
        return 1
    with open(GMR_PATH, encoding="utf-8") as f:
        gmr = f.read()

    if PF4H_MARK not in gmr:
        # Plain / rccl image: apply.py never ran, coverage_patch.py no-oped.
        print("m23_patch: PF4H patch absent (plain image); nothing to do")
        return 0

    if MARK not in gmr and COV_MARK not in gmr:
        print(
            "m23_patch: FATAL chain order inverted -- "
            f"{COV_MARK} absent from a PF4H-patched gpu_model_runner.py. "
            "coverage_patch.py MUST run before m23_patch.py.",
            file=sys.stderr,
        )
        return 1

    for path, label, edits, required in (
        (DPU_PATH, "dp_utils.py", DPU_EDITS, True),
        (GMR_PATH, "gpu_model_runner.py", GMR_EDITS, True),
        (FCTX_PATH, "forward_context.py", FCTX_EDITS, True),
        (VLLM_FULL_PATH, "shim vllm_full.py", FULL_EDITS, False),
        (M15_VLLM_PATH, "shim m15_vllm.py", M15V_EDITS, False),
        (CONTRACTS_PATH, "shim contracts.py", CONTRACTS_EDITS, False),
        (RUNTIME_PATH, "shim runtime.py", RT_EDITS, False),
    ):
        rc = _apply(path, label, edits, required=required)
        if rc != 0:
            return rc

    print("m23_patch: RAGGED_SEAL install complete (" + MARK + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())

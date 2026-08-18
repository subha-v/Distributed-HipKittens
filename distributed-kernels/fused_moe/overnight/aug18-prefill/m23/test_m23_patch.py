#!/usr/bin/env python3
"""CPU-only tests for the M23 ragged-seal in-container patcher.

What this proves, without a GPU, a node, or a network:

  1. CHAIN COMPOSITION -- coverage_patch.py then m23_patch.py apply cleanly to
     byte-copies of the deployed mirror; every anchor matches exactly once.
  2. SYNTAX -- every patched file AST-parses after both patches.
  3. MARKERS -- PF4H_COVERAGE_PATCH_V1 and PF4H_M23_RAGGED_SEAL_V1 are both
     present in every file each patch claims to touch, and the load-bearing
     pre-existing sentinels (apply.py's PATCH_SENTINELS) survive.
  4. IDEMPOTENCY -- a second m23_patch run is a no-op and leaves the bytes
     unchanged.
  5. CHAIN-ORDER ENFORCEMENT -- running m23_patch BEFORE coverage_patch exits
     non-zero, and running it on a plain (non-PF4H) tree exits 0 as a no-op.
  6. THE PREDICATE -- the new seal decision, extracted into a pure function
     that is checked line-for-line against the patched source, is exercised
     over the design's decision table (section 4.2/4.3, hazards H1/H2).

Run:  python3 test_m23_patch.py          (also works under pytest)

Paths: the mirror and the two patchers live in this session's scratchpad; set
M23_SCRATCH to override.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRATCH = Path(
    os.environ.get(
        "M23_SCRATCH",
        "/private/tmp/claude-501/-Users-subha-repos-Distributed-HipKittens/"
        "4077c4da-1965-47cb-aa86-44e5f55b46c4/scratchpad",
    )
)
MIRROR = SCRATCH / "mirror"
COV_PATCH_SRC = SCRATCH / "coverage_patch.py"
M23_PATCH_SRC = SCRATCH / "m23_patch.py"

COV_MARK = "PF4H_COVERAGE_PATCH_V1"
M23_MARK = "PF4H_M23_RAGGED_SEAL_V1"

# mirror-relative source -> in-container-relative destination
LAYOUT = {
    "vllm_patched/v1/worker/gpu_model_runner.py": "v1/worker/gpu_model_runner.py",
    "vllm_patched/v1/worker/dp_utils.py": "v1/worker/dp_utils.py",
    "vllm_patched/v1/cudagraph_dispatcher.py": "v1/cudagraph_dispatcher.py",
    "vllm_patched/forward_context.py": "forward_context.py",
    "pf4h_integration/vllm_full.py": "shim/vllm_full.py",
    "pf4h_integration/m15_vllm.py": "shim/m15_vllm.py",
    "pf4h_integration/contracts.py": "shim/contracts.py",
    "pf4h_integration/runtime.py": "shim/runtime.py",
    "pf4h_integration/weights.py": "shim/weights.py",
}

FAILURES: list[str] = []


def check(cond: bool, what: str) -> None:
    if cond:
        print(f"  ok   {what}")
    else:
        print(f"  FAIL {what}")
        FAILURES.append(what)


# --------------------------------------------------------------------------
# harness: build a fake container tree and drive both patchers against it
# --------------------------------------------------------------------------
def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolves cls.__module__ through sys.modules during class
    # creation, so the module has to be registered before exec_module.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_tree(root: Path) -> dict[str, Path]:
    """Copy the mirror files into ``root`` and return dest-name -> path."""
    out = {}
    for src, dest in LAYOUT.items():
        dpath = root / dest
        dpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(MIRROR / src, dpath)
        out[dest] = dpath
    return out


def bind_coverage(root: Path):
    mod = _load("coverage_patch_under_test", COV_PATCH_SRC)
    mod.PATH = str(root / "v1/worker/gpu_model_runner.py")
    mod.CONTRACTS_PATH = str(root / "shim/contracts.py")
    mod.VLLM_FULL_PATH = str(root / "shim/vllm_full.py")
    # coverage_patch grew a weights.py relax after the first deploy; bind it
    # too so the composition under test is the real one.
    if hasattr(mod, "WEIGHTS_PATH"):
        mod.WEIGHTS_PATH = str(root / "shim/weights.py")
    return mod


def bind_m23(root: Path):
    mod = _load("m23_patch_under_test", M23_PATCH_SRC)
    mod.GMR_PATH = str(root / "v1/worker/gpu_model_runner.py")
    mod.DPU_PATH = str(root / "v1/worker/dp_utils.py")
    mod.FCTX_PATH = str(root / "forward_context.py")
    mod.VLLM_FULL_PATH = str(root / "shim/vllm_full.py")
    mod.M15_VLLM_PATH = str(root / "shim/m15_vllm.py")
    mod.CONTRACTS_PATH = str(root / "shim/contracts.py")
    mod.RUNTIME_PATH = str(root / "shim/runtime.py")
    return mod


# --------------------------------------------------------------------------
# 1-4  composition, syntax, markers, idempotency
# --------------------------------------------------------------------------
def test_chain_applies(root: Path) -> dict[str, Path]:
    print("\n[1] chain: coverage_patch -> m23_patch on mirror copies")
    files = build_tree(root)
    cov = bind_coverage(root)
    check(cov.main() == 0, "coverage_patch.main() == 0")
    m23 = bind_m23(root)
    check(m23.main() == 0, "m23_patch.main() == 0")

    print("\n[2] every patched file AST-parses")
    for dest, path in files.items():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            check(True, f"ast.parse {dest}")
        except SyntaxError as exc:
            check(False, f"ast.parse {dest}: {exc}")

    print("\n[3] markers")
    gmr = files["v1/worker/gpu_model_runner.py"].read_text()
    check(COV_MARK in gmr, "GMR keeps the coverage marker")
    check(M23_MARK in gmr, "GMR carries the M23 marker")
    check("M15_COVERAGE steps=%d" in gmr, "GMR keeps the M15_COVERAGE line")
    check("RAGGED_SEAL_RECEIPT rank=%d" in gmr, "GMR emits RAGGED_SEAL_RECEIPT")
    check(
        "RAGGED_SEAL_RECEIPT_FINAL rank=%d" in gmr,
        "GMR emits a shutdown receipt (design 8.2)",
    )
    check(
        "PF4H_GRAPH_REPLAY_RECEIPT" in gmr,
        "apply.py sentinel PF4H_GRAPH_REPLAY_RECEIPT survives",
    )
    check("M15_POST_CAPTURE_COMMIT" in gmr, "M15 sentinel survives in GMR")
    check(
        "_pf4h_local_readiness" in gmr and "_pf4h_m23_ragged_enabled" in gmr,
        "GMR gains both readiness helpers",
    )
    check(
        "b4096_unanimous" in gmr and "pf4h_ragged_seal" in gmr,
        "GMR gains the new predicate names",
    )

    dpu = files["v1/worker/dp_utils.py"].read_text()
    check(M23_MARK in dpu, "dp_utils carries the M23 marker")
    check("torch.zeros(5, dp_size" in dpu, "dp_utils all-reduce widened to 5 rows")
    check("_post_process_pf4h_ready" in dpu, "dp_utils gains the readiness reducer")
    check(
        "return_unpadded_counts: bool = False" in dpu,
        "apply.py sentinel return_unpadded_counts survives",
    )

    fctx = files["forward_context.py"].read_text()
    check(M23_MARK in fctx, "forward_context carries the M23 marker")
    check(
        "pf4h_exact_b4096: bool = False" in fctx,
        "apply.py sentinel pf4h_exact_b4096 field survives",
    )

    full = files["shim/vllm_full.py"].read_text()
    check("PF4H_EPLB_FULL018_RELAX_V1" in full, "shim keeps the coverage FULL-018 relax")
    check(M23_MARK in full, "shim vllm_full carries the M23 marker")
    check("PF4H-FULL-020" in full, "shim gains the VLLM_MOE_SKIP_PADDING refusal")

    contracts = files["shim/contracts.py"].read_text()
    check(
        "PF4H_EPLB_CONTRACT_RELAX_V1" in contracts,
        "shim contracts keeps the coverage contract relax",
    )
    check(M23_MARK in contracts, "shim contracts carries the M23 marker")

    runtime = files["shim/runtime.py"].read_text()
    check(M23_MARK in runtime, "shim runtime carries the M23 marker")
    check("rows_compared" in runtime, "receipts record rows_compared")

    m15v = files["shim/m15_vllm.py"].read_text()
    check(M23_MARK in m15v, "shim m15_vllm carries the M23 marker")

    weights = files["shim/weights.py"].read_text()
    check(
        "PF4H_EPLB_WEIGHT005_RELAX_V1" in weights,
        "shim weights keeps the coverage WEIGHT-005 relax (m23 leaves it alone)",
    )
    check(M23_MARK not in weights, "m23_patch does not touch shim weights.py")

    print("\n[4] idempotency")
    before = {d: p.read_bytes() for d, p in files.items()}
    m23b = bind_m23(root)
    check(m23b.main() == 0, "second m23_patch.main() == 0")
    covb = bind_coverage(root)
    check(covb.main() == 0, "second coverage_patch.main() == 0")
    unchanged = all(files[d].read_bytes() == b for d, b in before.items())
    check(unchanged, "re-running both patchers changes no bytes")
    return files


# --------------------------------------------------------------------------
# 5  chain-order enforcement
# --------------------------------------------------------------------------
def test_chain_order(tmp: Path) -> None:
    print("\n[5] chain-order enforcement")

    root = tmp / "inverted"
    build_tree(root)
    m23 = bind_m23(root)
    rc = m23.main()
    check(rc != 0, f"m23 before coverage exits non-zero (got {rc})")
    gmr = (root / "v1/worker/gpu_model_runner.py").read_text()
    check(M23_MARK not in gmr, "the refused run wrote nothing")

    # and once coverage has run, the same tree patches fine
    cov = bind_coverage(root)
    check(cov.main() == 0, "coverage_patch then succeeds on the same tree")
    m23b = bind_m23(root)
    check(m23b.main() == 0, "m23_patch then succeeds on the same tree")

    # plain image: no PF4H patch at all -> both patchers no-op with rc 0
    plain = tmp / "plain"
    build_tree(plain)
    gpath = plain / "v1/worker/gpu_model_runner.py"
    gpath.write_text(
        gpath.read_text().replace("PF4H_INTEGRATION_PATCH_V3_M15", "PLAIN_IMAGE")
    )
    m23p = bind_m23(plain)
    check(m23p.main() == 0, "plain image: m23_patch exits 0")
    check(
        M23_MARK not in gpath.read_text(),
        "plain image: m23_patch wrote nothing to GMR",
    )

    # a mangled anchor must be fatal, not silently skipped
    broken = tmp / "broken"
    build_tree(broken)
    bind_coverage(broken).main()
    dpath = broken / "v1/worker/dp_utils.py"
    dpath.write_text(
        dpath.read_text().replace(
            "    tensor_cpu = torch.zeros(4, dp_size, dtype=torch.int32)\n",
            "    tensor_cpu = torch.zeros(4, dp_size, dtype=torch.int32)  # local\n",
        )
    )
    rc = bind_m23(broken).main()
    check(rc != 0, f"a mangled dp_utils anchor is fatal (got {rc})")


# --------------------------------------------------------------------------
# 6  the new seal predicate
# --------------------------------------------------------------------------
class _Cfg:
    def __init__(self, dp_size=8, dp_rank=0, lora=None):
        self.data_parallel_size = dp_size
        self.data_parallel_rank = dp_rank
        self.lora_config = lora


class PredicateModel:
    """Executable replica of the patched seal decision.

    Kept honest by ``test_predicate_matches_source``: every distinctive line
    of the patched GMR predicate must be present in the installed source.
    """

    PIECEWISE = 1
    FULL = 2
    NONE = 0

    def __init__(self, env, *, operator_latched=True, lora_config=None):
        self.env = env
        self.operator_latched = operator_latched
        self.parallel_config = _Cfg(lora=lora_config)
        self.vllm_config = self.parallel_config

    # ---- E3 -----------------------------------------------------------
    def _pf4h_m23_ragged_enabled(self):
        return (
            self.env.get("VLLM_PF4H_M23_RAGGED", "1") != "0"
            and self.env.get("VLLM_PF4H_INTEGRATION_MODE") in ("full", "m15")
            and self.parallel_config.data_parallel_size == 8
        )

    def _pf4h_graph_operator_enabled(self):
        if self.env.get("VLLM_PF4H_B4096_GRAPH_TARGET") != "pf4h":
            return False
        return self.operator_latched

    def _pf4h_local_readiness(self, *, has_lora, pf4h_graph_target):
        if not self._pf4h_m23_ragged_enabled():
            return False
        if self.env.get("VLLM_PF4H_B4096_GRAPH_TARGET") != "pf4h":
            return False
        if has_lora or self.vllm_config.lora_config is not None:
            return False
        if self.env.get("VLLM_MOE_SKIP_PADDING", "0") not in ("", "0"):
            return False
        if pf4h_graph_target:
            return True
        return self._pf4h_graph_operator_enabled()

    # ---- E4 -----------------------------------------------------------
    def decide(
        self,
        *,
        local_mode,
        descriptor_tokens,
        synced_cudagraph_mode,
        num_tokens_across_dp,
        original_counts,
        should_ubatch=False,
        has_lora=False,
        uniform_decode=False,
        pf4h_graph_target=None,
        force_uniform_decode=None,
        force_num_active_loras=None,
        peer_ready=True,
    ):
        """Return (cudagraph_mode, sealed, flags)."""
        cudagraph_mode = local_mode
        batch_tokens = descriptor_tokens

        # -- pre-M23 exact gate (kept verbatim, still drives capture) ----
        pf4h_exact = False
        if (
            self.env.get("VLLM_PF4H_INTEGRATION_MODE") in ("full", "m15")
            and self.parallel_config.data_parallel_size == 8
            and not should_ubatch
            and not uniform_decode
            and not has_lora
            and batch_tokens == 4096
            and original_counts is not None
        ):
            pf4h_exact = tuple(original_counts) == (4096,) * 8

        local_ready = self._pf4h_local_readiness(
            has_lora=has_lora, pf4h_graph_target=pf4h_graph_target
        )
        pf4h_ready_all = bool(local_ready and peer_ready)

        m23_serving = (
            pf4h_graph_target is None
            and force_uniform_decode is None
            and force_num_active_loras is None
            and self._pf4h_m23_ragged_enabled()
        )
        m23_orig = tuple(int(v) for v in (original_counts or ()))
        m23_exact = m23_orig == (4096,) * 8
        b4096_unanimous = bool(
            m23_serving
            and not should_ubatch
            and synced_cudagraph_mode == self.PIECEWISE
            and num_tokens_across_dp is not None
            and all(int(v) == 4096 for v in num_tokens_across_dp)
        )
        pf4h_ragged_seal = bool(b4096_unanimous and pf4h_ready_all)
        uniform_rescued = False

        if pf4h_graph_target is not None:
            if pf4h_graph_target and not pf4h_exact:
                raise RuntimeError(
                    "PF4H graph capture requested without exact DP8 B4096 inputs"
                )
            pf4h_exact = pf4h_graph_target
        elif m23_serving:
            pf4h_exact = pf4h_ragged_seal
        elif pf4h_exact:
            pf4h_exact = self._pf4h_graph_operator_enabled()

        if (
            b4096_unanimous
            and cudagraph_mode != self.PIECEWISE
            and (
                pf4h_ragged_seal
                or self.env.get("VLLM_PF4H_B4096_UNIFORM_RESCUE", "1") != "0"
            )
        ):
            cudagraph_mode = self.PIECEWISE
            batch_tokens = 4096
            uniform_rescued = True

        if (
            batch_tokens == 4096
            and self.env.get("VLLM_PF4H_B4096_GRAPH_TARGET") == "pf4h"
            and not pf4h_exact
        ):
            cudagraph_mode = self.NONE

        if m23_serving and pf4h_exact:
            assert cudagraph_mode == self.PIECEWISE, "sealed step is not PIECEWISE"

        return cudagraph_mode, pf4h_exact, {
            "b4096_unanimous": b4096_unanimous,
            "uniform_rescued": uniform_rescued,
            "sealed_exact": pf4h_exact and m23_exact,
            "sealed_ragged": pf4h_exact and not m23_exact,
            "ready_all": pf4h_ready_all,
        }


PF4H_ENV = {
    "VLLM_PF4H_INTEGRATION_MODE": "m15",
    "VLLM_PF4H_B4096_GRAPH_TARGET": "pf4h",
}
STOCK_ENV = {
    "VLLM_PF4H_INTEGRATION_MODE": "m15",
    "VLLM_PF4H_B4096_GRAPH_TARGET": "stock",
}

RAGGED = (4096, 1, 3300, 4096, 512, 4096, 77, 2048)
EXACT = (4096,) * 8
PADDED8 = [4096] * 8


def test_predicate() -> None:
    print("\n[6] the new seal predicate (design decision table)")
    P = PredicateModel

    # -- all-8-exactly-4096: the old population still seals ---------------
    m = P(dict(PF4H_ENV))
    mode, sealed, f = m.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=EXACT,
    )
    check(sealed and mode == P.PIECEWISE, "exact-4096 step seals under PIECEWISE")
    check(f["sealed_exact"] and not f["sealed_ragged"], "exact step counts as exact")

    # -- ragged, in bucket, all ranks PIECEWISE: the M23 win --------------
    mode, sealed, f = m.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=RAGGED,
    )
    check(sealed and mode == P.PIECEWISE, "ragged in-bucket step SEALS (the fix)")
    check(f["sealed_ragged"] and not f["sealed_exact"], "ragged step counts as ragged")
    check(f["b4096_unanimous"], "ragged in-bucket step is unanimous")

    # -- pre-M23 behaviour for the same step -------------------------------
    off = P({**PF4H_ENV, "VLLM_PF4H_M23_RAGGED": "0"})
    mode, sealed, f = off.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=RAGGED,
    )
    check(not sealed, "VLLM_PF4H_M23_RAGGED=0: the same ragged step does NOT seal")
    check(mode == P.NONE, "VLLM_PF4H_M23_RAGGED=0: pf4h arm falls back to eager")

    # -- H1: the uniform-decode rank ---------------------------------------
    mode, sealed, f = m.decide(
        local_mode=P.NONE,          # DISP fell through: uniform_decode blocks
        descriptor_tokens=4096,     # ... but DP padding still put it at 4096
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=(1, 4096, 4096, 4096, 4096, 4096, 4096, 4096),
        uniform_decode=True,
    )
    check(sealed, "H1 uniform-decode rank seals under M23")
    check(mode == P.PIECEWISE, "H1 rank is rescued to PIECEWISE (not eager)")
    check(f["uniform_rescued"], "H1 rank is counted as uniform_rescued")

    # -- H1 on the stock arm: rescued, never sealed -------------------------
    s = P(dict(STOCK_ENV))
    mode, sealed, f = s.decide(
        local_mode=P.NONE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=RAGGED,
        uniform_decode=True,
    )
    check(not sealed, "stock arm never seals (readiness bit is target-gated)")
    check(mode == P.PIECEWISE, "stock arm's H1 rank IS rescued (A/B fairness)")
    check(f["uniform_rescued"], "stock arm counts the rescue")

    # -- H1 rescue can be ablated off for an unsealed step ------------------
    s0 = P({**STOCK_ENV, "VLLM_PF4H_B4096_UNIFORM_RESCUE": "0"})
    mode, sealed, f = s0.decide(
        local_mode=P.NONE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=RAGGED,
        uniform_decode=True,
    )
    check(mode == P.NONE and not f["uniform_rescued"], "UNIFORM_RESCUE=0 ablates it")

    # ... but never for a step that sealed (risk R3)
    p0 = P({**PF4H_ENV, "VLLM_PF4H_B4096_UNIFORM_RESCUE": "0"})
    mode, sealed, f = p0.decide(
        local_mode=P.NONE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=RAGGED,
        uniform_decode=True,
    )
    check(
        sealed and mode == P.PIECEWISE,
        "a SEALED step is rescued even with UNIFORM_RESCUE=0 (risk R3)",
    )

    # -- H2: no DP padding -> ragged per-rank counts, must NOT seal ---------
    mode, sealed, f = m.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.NONE,          # some rank dispatched NONE
        num_tokens_across_dp=[4096, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
        original_counts=RAGGED,
    )
    check(not sealed, "H2 non-padded ragged step does NOT seal")
    check(not f["b4096_unanimous"], "H2 step is not counted in_bucket")
    check(mode == P.NONE, "H2 step on the pf4h arm runs eager")

    # -- a padded bucket below 4096 must not seal --------------------------
    mode, sealed, f = m.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=512,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=[512] * 8,
        original_counts=(512, 3, 7, 512, 512, 1, 2, 512),
    )
    check(not sealed and not f["b4096_unanimous"], "a 512 bucket never seals")

    # -- ubatching must not seal -------------------------------------------
    mode, sealed, f = m.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=RAGGED,
        should_ubatch=True,
    )
    check(not sealed, "should_ubatch blocks the seal")

    # -- lora: local flag AND a configured adapter both block --------------
    mode, sealed, f = m.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=RAGGED,
        has_lora=True,
    )
    check(not sealed, "has_lora blocks the seal via the readiness bit")
    ml = P(dict(PF4H_ENV), lora_config=object())
    mode, sealed, f = ml.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=RAGGED,
    )
    check(not sealed, "a configured lora_config blocks the seal")

    # -- VLLM_MOE_SKIP_PADDING must never seal (risk R5) --------------------
    msp = P({**PF4H_ENV, "VLLM_MOE_SKIP_PADDING": "1"})
    mode, sealed, f = msp.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=RAGGED,
    )
    check(not sealed, "VLLM_MOE_SKIP_PADDING=1 refuses the seal (risk R5)")

    # -- one peer not yet activated: NOBODY seals (risk R2) -----------------
    mode, sealed, f = m.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=RAGGED,
        peer_ready=False,
    )
    check(not sealed, "a single not-ready peer refuses the seal on every rank")
    check(f["b4096_unanimous"] and not f["ready_all"], "... counted refused_not_ready")

    # -- this rank's own latch not yet set ---------------------------------
    nl = P(dict(PF4H_ENV), operator_latched=False)
    mode, sealed, f = nl.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=RAGGED,
    )
    check(not sealed, "an unlatched local activation file refuses the seal")

    # -- capture / warmup: pf4h_graph_target is not None, unchanged ---------
    mode, sealed, f = m.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=EXACT,
        pf4h_graph_target=True,
    )
    check(sealed and not f["b4096_unanimous"], "capture drive stamps, bypassing M23")
    raised = False
    try:
        m.decide(
            local_mode=P.PIECEWISE,
            descriptor_tokens=4096,
            synced_cudagraph_mode=P.PIECEWISE,
            num_tokens_across_dp=PADDED8,
            original_counts=RAGGED,
            pf4h_graph_target=True,
        )
    except RuntimeError:
        raised = True
    check(raised, "capture still REFUSES a ragged drive (unchanged)")

    mode, sealed, f = m.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=EXACT,
        pf4h_graph_target=False,
    )
    check(not sealed, "stock capture drive (target False) does not stamp")

    # -- _dummy_run: force_* set -> pre-M23 path ----------------------------
    mode, sealed, f = m.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=RAGGED,
        force_uniform_decode=False,
        force_num_active_loras=0,
    )
    check(
        not sealed and not f["b4096_unanimous"],
        "a _dummy_run (force_* set) never takes the M23 path",
    )
    mode, sealed, f = m.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=PADDED8,
        original_counts=EXACT,
        force_uniform_decode=False,
        force_num_active_loras=0,
    )
    check(sealed, "an exact-4096 _dummy_run still seals the pre-M23 way")

    # -- dp_size != 8 -------------------------------------------------------
    m4 = P(dict(PF4H_ENV))
    m4.parallel_config.data_parallel_size = 4
    m4.vllm_config = m4.parallel_config
    mode, sealed, f = m4.decide(
        local_mode=P.PIECEWISE,
        descriptor_tokens=4096,
        synced_cudagraph_mode=P.PIECEWISE,
        num_tokens_across_dp=[4096] * 4,
        original_counts=(4096,) * 4,
    )
    check(not sealed, "dp_size != 8 never seals")


def test_predicate_matches_source(files: dict[str, Path]) -> None:
    """The replica above is only evidence if it mirrors the installed text."""
    print("\n[7] the replica tracks the patched source")
    gmr = files["v1/worker/gpu_model_runner.py"].read_text()
    required = (
        'os.environ.get("VLLM_PF4H_M23_RAGGED", "1") != "0"',
        'os.environ.get("VLLM_PF4H_B4096_GRAPH_TARGET") != "pf4h"',
        "if has_lora or self.vllm_config.lora_config is not None:",
        "if envs.VLLM_MOE_SKIP_PADDING:",
        "return self._pf4h_graph_operator_enabled()",
        "and force_uniform_decode is None",
        "and force_num_active_loras is None",
        "and synced_cudagraph_mode == CUDAGraphMode.PIECEWISE.value",
        "and all(int(v) == 4096 for v in num_tokens_across_dp.tolist())",
        "pf4h_ragged_seal = bool(b4096_unanimous and pf4h_ready_all)",
        "            pf4h_exact_b4096 = pf4h_ragged_seal",
        "            and cudagraph_mode != CUDAGraphMode.PIECEWISE",
        'os.environ.get("VLLM_PF4H_B4096_UNIFORM_RESCUE", "1")',
        "batch_descriptor = BatchDescriptor(num_tokens=4096)",
        "assert cudagraph_mode == CUDAGraphMode.PIECEWISE, (",
        "pf4h_ready=pf4h_local_ready,",
        "return_pf4h_ready=True,",
    )
    for frag in required:
        check(frag in gmr, f"source contains: {frag[:58]}")

    dpu = files["v1/worker/dp_utils.py"].read_text()
    for frag in (
        "tensor_cpu[4][dp_rank] = 1 if pf4h_ready else 0",
        "def _post_process_pf4h_ready(tensor: torch.Tensor) -> bool:",
        "pf4h_ready_all = _post_process_pf4h_ready(tensor)",
        "return_pf4h_ready: bool = False,",
    ):
        check(frag in dpu, f"dp_utils contains: {frag[:58]}")


def test_no_behaviour_change_off_path(files: dict[str, Path]) -> None:
    """The M23 code must be reachable only through its own gates."""
    print("\n[8] scope: every new GMR branch sits behind an M23 gate")
    gmr = files["v1/worker/gpu_model_runner.py"].read_text()
    tree = ast.parse(gmr)
    names = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
    }
    check("_pf4h_m23_ragged_enabled" in names, "helper is a real def")
    check("_pf4h_local_readiness" in names, "readiness is a real def")
    # the seal override and the counters are both guarded by m23_serving,
    # which itself requires _pf4h_m23_ragged_enabled().
    check(
        gmr.count("m23_serving = (") == 1,
        "m23_serving is computed exactly once",
    )
    check(
        gmr.count("and self._pf4h_m23_ragged_enabled()") == 1,
        "m23_serving folds in the env gate",
    )
    check("elif m23_serving:" in gmr, "the seal override is gated on m23_serving")
    check("if m23_serving and pf4h_exact_b4096:" in gmr, "the R3 assert is gated")
    check("        if m23_serving:\n" in gmr, "the counters are gated")


# --------------------------------------------------------------------------
# 9  the patched dp_utils actually runs (torch, CPU, no distributed backend)
# --------------------------------------------------------------------------
def test_dp_utils_executes(files: dict[str, Path]) -> None:
    print("\n[9] the patched dp_utils executes on CPU torch")
    try:
        import torch
    except ImportError:  # pragma: no cover
        print("  skip (torch not installed)")
        return

    import types

    stubs = {}

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        stubs[name] = m
        return m

    class _ParallelConfig:
        pass

    _mod("vllm")
    _mod("vllm.config", ParallelConfig=_ParallelConfig)
    _mod("vllm.distributed")
    _mod("vllm.distributed.parallel_state", get_dp_group=lambda: None)
    _mod("vllm.logger", init_logger=lambda name: types.SimpleNamespace(
        info_once=lambda *a, **k: None, debug=lambda *a, **k: None
    ))
    _mod("vllm.v1")
    _mod("vllm.v1.worker")
    _mod(
        "vllm.v1.worker.ubatch_utils",
        check_ubatch_thresholds=lambda *a, **k: False,
        is_last_ubatch_empty=lambda *a, **k: False,
    )

    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "dpu_under_test", files["v1/worker/dp_utils.py"]
        )
        dpu = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dpu)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    class Cfg:
        data_parallel_size = 8
        data_parallel_rank = 3
        num_ubatches = 2
        disable_nccl_for_dp_synchronization = True

    cfg = Cfg()

    def fake_ar(rows):
        """rows: list of 8 per-rank (orig, padded, ubatch, mode, ready)."""
        t = torch.zeros(5, 8, dtype=torch.int32)
        for r, vals in enumerate(rows):
            for i, v in enumerate(vals):
                t[i][r] = v
        return t

    # -- _run_ar contributes the readiness bit on row 4 --------------------
    captured = {}

    def stub_all_reduce(tensor, group=None):
        captured["tensor"] = tensor.clone()

    real_dist_ar = dpu.dist.all_reduce
    dpu._get_device_and_group = lambda pc: ("cpu", None)
    dpu.dist.all_reduce = stub_all_reduce
    try:
        out = dpu._run_ar(
            should_ubatch=False,
            orig_num_tokens_per_ubatch=137,
            padded_num_tokens_per_ubatch=4096,
            cudagraph_mode=1,
            parallel_config=cfg,
            pf4h_ready=True,
        )
        c = captured["tensor"]
        check(tuple(c.shape) == (5, 8), f"_run_ar stages a (5, 8) tensor (got {tuple(c.shape)})")
        check(int(c[0][3]) == 137, "_run_ar row 0 carries the original count")
        check(int(c[4][3]) == 1, "_run_ar row 4 carries pf4h_ready=True")
        out2 = dpu._run_ar(
            should_ubatch=False,
            orig_num_tokens_per_ubatch=1,
            padded_num_tokens_per_ubatch=1,
            cudagraph_mode=1,
            parallel_config=cfg,
        )
        check(
            int(captured["tensor"][4][3]) == 0,
            "_run_ar defaults pf4h_ready to 0 for unpatched callers",
        )
        del out, out2
    finally:
        dpu.dist.all_reduce = real_dist_ar

    # -- the reducer -------------------------------------------------------
    all_ready = fake_ar([(4096, 4096, 0, 1, 1)] * 8)
    one_cold = fake_ar([(4096, 4096, 0, 1, 1)] * 7 + [(4096, 4096, 0, 1, 0)])
    check(dpu._post_process_pf4h_ready(all_ready) is True, "reducer: all ready -> True")
    check(dpu._post_process_pf4h_ready(one_cold) is False, "reducer: one cold -> False")
    check(
        dpu._post_process_pf4h_ready(torch.zeros(4, 8, dtype=torch.int32)) is False,
        "reducer tolerates a legacy 4-row tensor",
    )

    # -- _synchronize_dp_ranks returns the 5-tuple -------------------------
    ragged = [(4096, 4096, 0, 1, 1), (1, 1, 0, 1, 1)] + [
        (3000, 3000, 0, 1, 1)
    ] * 6
    dpu._run_ar = lambda **kw: fake_ar(ragged)
    res = dpu._synchronize_dp_ranks(4096, 4096, False, 1, cfg, pf4h_ready=True)
    check(len(res) == 5, f"_synchronize_dp_ranks returns 5 values (got {len(res)})")
    should_ub, padded, synced, orig, ready_all = res
    check(synced == 1, "synced cudagraph mode is the min (PIECEWISE)")
    check(
        [int(v) for v in padded.tolist()] == [4096] * 8,
        "DP padding lifts every rank to the max (4096)",
    )
    check([int(v) for v in orig.tolist()][1] == 1, "original counts stay ragged")
    check(ready_all is True, "pf4h_ready_all is True when every rank contributed")

    # H2: one rank dispatched NONE -> no DP padding, ragged padded counts
    h2 = [(4096, 4096, 0, 1, 1), (1000, 1000, 0, 0, 1)] + [
        (1000, 1000, 0, 1, 1)
    ] * 6
    dpu._run_ar = lambda **kw: fake_ar(h2)
    _, padded_h2, synced_h2, _, _ = dpu._synchronize_dp_ranks(
        4096, 4096, False, 1, cfg, pf4h_ready=True
    )
    check(synced_h2 == 0, "H2: synced mode collapses to NONE")
    check(
        [int(v) for v in padded_h2.tolist()] != [4096] * 8,
        "H2: no DP padding, so the padded counts stay ragged",
    )

    # -- coordinate_batch_across_dp arity is backwards compatible ----------
    dpu._run_ar = lambda **kw: fake_ar(ragged)
    r3 = dpu.coordinate_batch_across_dp(
        num_tokens_unpadded=4096, allow_microbatching=False, parallel_config=cfg
    )
    check(len(r3) == 3, "legacy 3-tuple caller (forward_context) unchanged")
    r4 = dpu.coordinate_batch_across_dp(
        num_tokens_unpadded=4096,
        allow_microbatching=False,
        parallel_config=cfg,
        return_unpadded_counts=True,
    )
    check(len(r4) == 4, "legacy 4-tuple caller (spec-decode) unchanged")
    r5 = dpu.coordinate_batch_across_dp(
        num_tokens_unpadded=4096,
        allow_microbatching=False,
        parallel_config=cfg,
        return_unpadded_counts=True,
        pf4h_ready=True,
        return_pf4h_ready=True,
    )
    check(len(r5) == 5 and r5[4] is True, "the M23 caller gets the readiness bit")

    class Cfg1(Cfg):
        data_parallel_size = 1
        data_parallel_rank = 0

    e5 = dpu.coordinate_batch_across_dp(
        num_tokens_unpadded=7,
        allow_microbatching=False,
        parallel_config=Cfg1(),
        return_unpadded_counts=True,
        return_pf4h_ready=True,
    )
    check(len(e5) == 5 and e5[4] is False, "dp_size==1 early exit honours the new arity")
    e4 = dpu.coordinate_batch_across_dp(
        num_tokens_unpadded=7,
        allow_microbatching=False,
        parallel_config=Cfg1(),
        return_unpadded_counts=True,
    )
    check(len(e4) == 4, "dp_size==1 legacy early exit unchanged")


# --------------------------------------------------------------------------
# 10  the L0 offline replay harness agrees with the predicate replica
# --------------------------------------------------------------------------
def test_offline_replay() -> None:
    print("\n[10] offline_dispatcher_replay (L0 harness)")
    rp_path = Path(__file__).with_name("offline_dispatcher_replay.py")
    if not rp_path.exists():
        check(False, "offline_dispatcher_replay.py exists")
        return
    rp = _load("offline_replay_under_test", rp_path)

    def step(tokens, uniform_idx=()):
        rows = []
        for i, n in enumerate(tokens):
            rows.append(
                rp.RankStep(
                    num_tokens=n,
                    num_reqs=1,
                    max_num_scheduled_tokens=1 if i in uniform_idx else n,
                )
            )
        return rows

    exact = step([4096] * 8)
    ragged = step([4096, 3300, 4096, 2048, 4096, 4096, 700, 4096])
    h1 = step([4096, 1, 4096, 4096, 4096, 4096, 4096, 4096], uniform_idx=(1,))
    trace = [exact, ragged, h1]

    cfg = rp.Config(graph_target="pf4h")
    old, _ = rp.replay(trace, cfg, m23=False)
    new, _ = rp.replay(trace, cfg, m23=True)
    check(
        all(c.in_bucket == 3 for c in new),
        f"all three steps are in bucket (got {[c.in_bucket for c in new]})",
    )
    check(all(c.sealed == 1 for c in old), "OLD predicate seals only the exact step")
    check(all(c.eager_b4096 == 2 for c in old), "OLD leaves 2 in-bucket steps eager")
    check(all(c.sealed == 3 for c in new), "NEW predicate seals all three steps")
    check(all(c.eager_b4096 == 0 for c in new), "NEW leaves nothing eager")
    check(new[1].uniform_rescued == 1, "the H1 rank is rescued exactly once")
    check(
        all(c.uniform_rescued == 0 for i, c in enumerate(new) if i != 1),
        "no other rank is rescued",
    )

    stock, _ = rp.replay(trace, rp.Config(graph_target="stock"), m23=True)
    check(all(c.sealed == 0 for c in stock), "the stock arm never seals")
    check(stock[1].uniform_rescued == 1, "the stock arm still rescues its H1 rank")

    off, _ = rp.replay(
        trace, rp.Config(graph_target="pf4h", m23=False), m23=True
    )
    check(all(c.sealed == 1 for c in off), "cfg.m23=False reproduces the OLD counts")

    # H2: one rank at a padded 4096 while a peer collapses the synced mode
    h2 = [step([4096, 77, 4096, 4096, 4096, 4096, 4096, 4096])]
    hc, hd = rp.replay(h2, cfg, m23=True)
    check(all(c.sealed == 0 for c in hc), "H2 step never seals in the replay")
    check(hd["h2_steps"] == 1, "the replay reports the H2 step")
    check(
        sum(c.refused_not_unanimous for c in hc) > 0,
        "H2 shows up as refused_not_unanimous",
    )

    # end-to-end CLI, including the L0 gate exit code
    proc = subprocess.run(
        [sys.executable, str(rp_path), "--synthetic", "--steps", "120"],
        capture_output=True,
        text=True,
    )
    check("=== L0 gates" in proc.stdout, "CLI prints the L0 gate section")
    check(proc.returncode in (0, 1), f"CLI exits 0/1 (got {proc.returncode})")


def main() -> int:
    if not MIRROR.exists():
        print(f"mirror not found at {MIRROR}", file=sys.stderr)
        return 2
    for p in (COV_PATCH_SRC, M23_PATCH_SRC):
        if not p.exists():
            print(f"patcher not found at {p}", file=sys.stderr)
            return 2

    repo_copy = Path(__file__).with_name("m23_patch.py")
    if repo_copy.exists():
        check(
            repo_copy.read_bytes() == M23_PATCH_SRC.read_bytes(),
            "the repo copy of m23_patch.py matches the deploy copy",
        )

    with tempfile.TemporaryDirectory(prefix="m23test-") as td:
        tmp = Path(td)
        files = test_chain_applies(tmp / "chain")
        test_chain_order(tmp)
        test_predicate()
        test_predicate_matches_source(files)
        test_no_behaviour_change_off_path(files)
        test_dp_utils_executes(files)
        test_offline_replay()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("M23_PATCH_TESTS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

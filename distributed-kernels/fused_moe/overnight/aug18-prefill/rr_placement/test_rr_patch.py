#!/usr/bin/env python3
"""CPU-only tests for the round-robin expert-placement in-container patcher.

What this proves, without a GPU, a node, or a network:

  1. CHAIN COMPOSITION -- coverage_patch.py, m23_patch.py and rr_patch.py apply
     cleanly, in that order, to byte-copies of the deployed mirror; every
     anchor matches exactly once against the POST-m23 text.
  2. ORDER INDEPENDENCE -- rr_patch's own anchors also match exactly once
     against the PRE-m23 text, so the M23 patch is optional in the chain.
  3. SYNTAX + MARKERS -- every patched file AST-parses; PF4H_RR_PLACEMENT_V1 is
     present in each file rr_patch claims to touch and absent from the ones it
     does not; the coverage / M23 / apply.py sentinels all survive.
  4. IDEMPOTENCY -- re-running all three patchers changes no bytes.
  5. CHAIN-ORDER ENFORCEMENT -- rr before coverage on a PF4H tree exits
     non-zero and writes nothing; a plain image applies the core edits only; a
     mangled anchor is fatal; no .rrtmp file survives.
  6. THE PHYSICAL NUMBERING -- the injected helper block is EXECUTED (real
     torch) and its p(e) is checked against the required invariant: physical
     ownership contiguous rank-major, rank r owning exactly [32r, 32r+32).
  7. THE ROUTER GATHER -- the injected `_pf4h_rr_to_physical` source is
     EXECUTED against the real table and every logical id is shown to land on
     rank e % 8 at local slot e // 8.
  8. THE REFUSALS -- pf4h_rr_configure is executed over the unsupported
     combinations (EPLB, redundant experts, non-MoRI backend, a backend with
     vLLM's own RR tables, VLLM_MOE_SKIP_PADDING, an indivisible expert count,
     a heterogeneous second layer, a malformed env value).
  9. THE ENV GATE -- with VLLM_PF4H_RR_PLACEMENT unset/'0' every patched
     decision equals the pre-patch decision (loader map, router gather,
     shim attestation, activation blockers).  The single documented exception
     is the ep_weight_filter guard, which is exercised on its own.
 10. THE SHIM ATTESTATION -- the round-robin map/mask pattern is built and the
     patched acceptance logic is shown to accept the LINEAR pattern and refuse
     the ROUND-ROBIN pattern (see RR_PLACEMENT_NOTES.md section 3 for why that
     polarity is the correct one).

Run:  python3 test_rr_patch.py          (also works under pytest)

Paths: the mirror and the three patchers live in this session's scratchpad;
set RR_SCRATCH to override.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import torch

SCRATCH = Path(
    os.environ.get(
        "RR_SCRATCH",
        "/private/tmp/claude-501/-Users-subha-repos-Distributed-HipKittens/"
        "4077c4da-1965-47cb-aa86-44e5f55b46c4/scratchpad",
    )
)
MIRROR = SCRATCH / "mirror"
COV_PATCH_SRC = SCRATCH / "coverage_patch.py"
M23_PATCH_SRC = SCRATCH / "m23_patch.py"
RR_PATCH_SRC = SCRATCH / "rr_patch.py"

COV_MARK = "PF4H_COVERAGE_PATCH_V1"
M23_MARK = "PF4H_M23_RAGGED_SEAL_V1"
RR_MARK = "PF4H_RR_PLACEMENT_V1"
RR_ENV = "VLLM_PF4H_RR_PLACEMENT"

E = 256          # DeepSeek-R1 routed experts
EP = 8           # EP8 / DP8
LOCAL = E // EP  # 32

# mirror-relative source -> in-container-relative destination
LAYOUT = {
    "vllm_patched/v1/worker/gpu_model_runner.py": "v1/worker/gpu_model_runner.py",
    "vllm_patched/v1/worker/dp_utils.py": "v1/worker/dp_utils.py",
    "vllm_patched/forward_context.py": "forward_context.py",
    "vllm_patched/model_executor/layers/fused_moe/expert_map_manager.py": (
        "model_executor/layers/fused_moe/expert_map_manager.py"
    ),
    "vllm_patched/model_executor/layers/fused_moe/router/base_router.py": (
        "model_executor/layers/fused_moe/router/base_router.py"
    ),
    "vllm_patched/model_executor/model_loader/ep_weight_filter.py": (
        "model_executor/model_loader/ep_weight_filter.py"
    ),
    "pf4h_integration/vllm_full.py": "shim/vllm_full.py",
    "pf4h_integration/m15_vllm.py": "shim/m15_vllm.py",
    "pf4h_integration/contracts.py": "shim/contracts.py",
    "pf4h_integration/runtime.py": "shim/runtime.py",
    "pf4h_integration/weights.py": "shim/weights.py",
}

EMM_DEST = "model_executor/layers/fused_moe/expert_map_manager.py"
BR_DEST = "model_executor/layers/fused_moe/router/base_router.py"
EPWF_DEST = "model_executor/model_loader/ep_weight_filter.py"
FULL_DEST = "shim/vllm_full.py"

FAILURES: list[str] = []


def check(cond: bool, what: str) -> None:
    if cond:
        print(f"  ok   {what}")
    else:
        print(f"  FAIL {what}")
        FAILURES.append(what)


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_tree(root: Path) -> dict[str, Path]:
    out = {}
    for src, dest in LAYOUT.items():
        dpath = root / dest
        dpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(MIRROR / src, dpath)
        out[dest] = dpath
    return out


def bind_coverage(root: Path):
    mod = _load(f"cov_{id(root)}", COV_PATCH_SRC)
    mod.PATH = str(root / "v1/worker/gpu_model_runner.py")
    mod.CONTRACTS_PATH = str(root / "shim/contracts.py")
    mod.VLLM_FULL_PATH = str(root / "shim/vllm_full.py")
    if hasattr(mod, "WEIGHTS_PATH"):
        mod.WEIGHTS_PATH = str(root / "shim/weights.py")
    return mod


def bind_m23(root: Path):
    mod = _load(f"m23_{id(root)}", M23_PATCH_SRC)
    mod.GMR_PATH = str(root / "v1/worker/gpu_model_runner.py")
    mod.DPU_PATH = str(root / "v1/worker/dp_utils.py")
    mod.FCTX_PATH = str(root / "forward_context.py")
    mod.VLLM_FULL_PATH = str(root / "shim/vllm_full.py")
    mod.M15_VLLM_PATH = str(root / "shim/m15_vllm.py")
    mod.CONTRACTS_PATH = str(root / "shim/contracts.py")
    mod.RUNTIME_PATH = str(root / "shim/runtime.py")
    return mod


def bind_rr(root: Path):
    mod = _load(f"rr_{id(root)}", RR_PATCH_SRC)
    mod.EMM_PATH = str(root / EMM_DEST)
    mod.ROUTER_PATH = str(root / BR_DEST)
    mod.EPWF_PATH = str(root / EPWF_DEST)
    mod.VLLM_FULL_PATH = str(root / FULL_DEST)
    mod.GMR_PATH = str(root / "v1/worker/gpu_model_runner.py")
    return mod


def extract_function(source: str, name: str) -> str:
    """Return the source text of a top-level (or nested) function by name."""
    tree = ast.parse(source)
    lines = source.splitlines(True)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            text = "".join(lines[node.lineno - 1 : node.end_lineno])
            # de-indent so it can be exec'd at module level
            pad = len(text) - len(text.lstrip(" "))
            if pad:
                text = "".join(
                    line[pad:] if line.strip() else line for line in text.splitlines(True)
                )
            return text
    raise AssertionError(f"function {name} not found")


def extract_rr_block(emm_source: str) -> str:
    """The injected helper block from the patched expert_map_manager.py."""
    start = emm_source.index("# --- " + RR_MARK)
    end = emm_source.index("def determine_expert_map(", start)
    return emm_source[start:end]


# --------------------------------------------------------------------------
# 1-4  composition, order independence, syntax, markers, idempotency
# --------------------------------------------------------------------------
def test_chain(root: Path) -> dict[str, Path]:
    print("\n[1] chain: coverage_patch -> m23_patch -> rr_patch on mirror copies")
    files = build_tree(root)
    check(bind_coverage(root).main() == 0, "coverage_patch.main() == 0")
    check(bind_m23(root).main() == 0, "m23_patch.main() == 0")
    check(bind_rr(root).main() == 0, "rr_patch.main() == 0 (anchors vs POST-m23)")

    print("\n[2] every patched file AST-parses")
    for dest, path in files.items():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            check(True, f"ast.parse {dest}")
        except SyntaxError as exc:
            check(False, f"ast.parse {dest}: {exc}")

    print("\n[3] markers")
    for dest in (EMM_DEST, BR_DEST, EPWF_DEST, FULL_DEST):
        check(RR_MARK in files[dest].read_text(), f"{dest} carries the RR marker")
    for dest in (
        "v1/worker/gpu_model_runner.py",
        "v1/worker/dp_utils.py",
        "forward_context.py",
        "shim/m15_vllm.py",
        "shim/contracts.py",
        "shim/runtime.py",
        "shim/weights.py",
    ):
        check(
            RR_MARK not in files[dest].read_text(),
            f"rr_patch does not touch {dest}",
        )
    gmr = files["v1/worker/gpu_model_runner.py"].read_text()
    check(COV_MARK in gmr and M23_MARK in gmr, "coverage + M23 marks survive in GMR")
    full = files[FULL_DEST].read_text()
    check("PF4H_EPLB_FULL018_RELAX_V1" in full, "coverage FULL-018 relax survives")
    check(M23_MARK in full, "M23 marker survives in shim vllm_full.py")
    check("PF4H-FULL-020" in full, "M23's FULL-020 blocker survives")
    check("PF4H-FULL-021" in full, "shim gains the RR/EPLB refusal FULL-021")
    check("PF4H_RR_ATTEST_RECEIPT" in full, "shim emits PF4H_RR_ATTEST_RECEIPT")
    emm = files[EMM_DEST].read_text()
    check("PF4H_RR_PLACEMENT_RECEIPT" in emm, "core emits PF4H_RR_PLACEMENT_RECEIPT")
    check(
        "_init_round_robin_expert_routing_tables" in emm,
        "vLLM's own RR routing-table builder is left intact",
    )
    check(
        'placement_strategy = determine_expert_placement_strategy(' in emm
        and 'moe_parallel_config.needs_round_robin_routing_tables' in emm,
        "neither placement guard was widened (the guards are untouched)",
    )

    print("\n[4] idempotency")
    before = {d: p.read_bytes() for d, p in files.items()}
    check(bind_rr(root).main() == 0, "second rr_patch.main() == 0")
    check(bind_m23(root).main() == 0, "second m23_patch.main() == 0")
    check(bind_coverage(root).main() == 0, "second coverage_patch.main() == 0")
    check(
        all(files[d].read_bytes() == b for d, b in before.items()),
        "re-running all three patchers changes no bytes",
    )
    return files


def test_order_independence(tmp: Path) -> None:
    print("\n[2b] rr_patch anchors also match against PRE-m23 text")
    root = tmp / "no_m23"
    files = build_tree(root)
    check(bind_coverage(root).main() == 0, "coverage_patch.main() == 0")
    check(bind_rr(root).main() == 0, "rr_patch.main() == 0 without the M23 chain")
    check(RR_MARK in files[FULL_DEST].read_text(), "shim patched without M23")
    check(M23_MARK not in files[FULL_DEST].read_text(), "and M23 really was absent")
    # and M23 still applies afterwards -- the two are order-independent
    check(bind_m23(root).main() == 0, "m23_patch still applies after rr_patch")
    try:
        ast.parse(files[FULL_DEST].read_text())
        check(True, "shim vllm_full.py parses after the inverted order")
    except SyntaxError as exc:
        check(False, f"shim parse after inverted order: {exc}")


# --------------------------------------------------------------------------
# 5  chain-order enforcement / failure modes
# --------------------------------------------------------------------------
def test_failure_modes(tmp: Path) -> None:
    print("\n[5] chain-order enforcement and failure modes")

    root = tmp / "inverted"
    files = build_tree(root)
    rc = bind_rr(root).main()
    check(rc != 0, f"rr before coverage on a PF4H tree exits non-zero (got {rc})")
    check(
        RR_MARK not in files[EMM_DEST].read_text(),
        "the refused run wrote nothing",
    )
    check(bind_coverage(root).main() == 0, "coverage then succeeds on the same tree")
    check(bind_rr(root).main() == 0, "rr_patch then succeeds on the same tree")

    plain = tmp / "plain"
    pfiles = build_tree(plain)
    gpath = pfiles["v1/worker/gpu_model_runner.py"]
    gpath.write_text(
        gpath.read_text().replace("PF4H_INTEGRATION_PATCH_V3_M15", "PLAIN_IMAGE")
    )
    check(bind_rr(plain).main() == 0, "plain image: rr_patch exits 0")
    check(
        RR_MARK in pfiles[EMM_DEST].read_text(),
        "plain image: the core vLLM edits still land",
    )

    broken = tmp / "broken"
    bfiles = build_tree(broken)
    bind_coverage(broken).main()
    epath = bfiles[EMM_DEST]
    epath.write_text(
        epath.read_text().replace(
            "        return self._expert_map[global_id].item()\n",
            "        return self._expert_map[global_id].item()  # local\n",
        )
    )
    rc = bind_rr(broken).main()
    check(rc != 0, f"a mangled expert_map_manager anchor is fatal (got {rc})")

    src = RR_PATCH_SRC.read_text()
    check("os.replace(tmp_path, path)" in src, "rr_patch installs via os.replace")
    check('tmp_path = path + ".rrtmp"' in src, "rr_patch stages a sibling temp file")
    leftovers = sorted(str(p) for p in tmp.rglob("*.rrtmp"))
    check(not leftovers, f"no .rrtmp files survive a run (found {leftovers})")


# --------------------------------------------------------------------------
# 6  the physical numbering -- the load-bearing invariant, EXECUTED
# --------------------------------------------------------------------------
def load_rr_namespace(emm_source: str) -> dict:
    """Exec the injected helper block with real torch and a logger stub."""

    class _Logger:
        def __init__(self):
            self.lines = []

        def warning_once(self, fmt, *args):
            self.lines.append(fmt % args)

        warning = warning_once
        info_once = warning_once

    ns: dict = {"torch": torch, "logger": _Logger(), "__name__": "rr_block"}
    exec(compile(extract_rr_block(emm_source), "<rr_block>", "exec"), ns)
    return ns


def test_physical_numbering(files: dict[str, Path]) -> None:
    print("\n[6] physical numbering: contiguous rank-major ownership")
    ns = load_rr_namespace(files[EMM_DEST].read_text())
    p = ns["pf4h_rr_physical_id"]

    expected = [(e % EP) * LOCAL + e // EP for e in range(E)]
    check(
        [p(e, EP, E) for e in range(E)] == expected,
        "p(e) == (e % 8) * 32 + e // 8",
    )
    check(sorted(expected) == list(range(E)), "p is a bijection on [0, 256)")

    contiguous = True
    for r in range(EP):
        owned = sorted(p(e, EP, E) for e in range(E) if e % EP == r)
        if owned != list(range(r * LOCAL, r * LOCAL + LOCAL)):
            contiguous = False
    check(contiguous, "rank r owns exactly physical [32r, 32r+32)")

    slots_ok = all(
        p(e, EP, E) // LOCAL == e % EP and p(e, EP, E) % LOCAL == e // EP
        for e in range(E)
    )
    check(slots_ok, "owner = p // 32 = e % 8 and local slot = p % 32 = e // 8")

    # the hot set 0..7 is spread one per rank
    hot_ranks = sorted(p(e, EP, E) // LOCAL for e in range(8))
    check(hot_ranks == list(range(8)), "logical experts 0-7 land one per rank")

    # the gather table: routed entries permuted, tail is the identity
    table = ns["pf4h_rr_gather_table"](E, EP, torch.device("cpu"), torch.int32)
    check(table.dtype == torch.int32, "gather table honours the requested dtype")
    check(table.numel() == E + 64, "gather table carries a 64-entry tail")
    check(table[:E].tolist() == expected, "gather table[:256] == p(e)")
    check(
        table[E:].tolist() == list(range(E, E + 64)),
        "gather table tail is the identity (fused shared-expert ids pass through)",
    )
    check(
        ns["pf4h_rr_gather_table"](E, EP, torch.device("cpu"), torch.int32) is table,
        "gather table is cached, not rebuilt per call",
    )


# --------------------------------------------------------------------------
# 7  the router gather -- EXECUTED against the real table
# --------------------------------------------------------------------------
def _install_fake_emm(ns: dict) -> None:
    """Register a fake vllm...expert_map_manager exposing the exec'd helpers."""
    chain = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.expert_map_manager",
    ]
    for name in chain:
        mod = sys.modules.get(name)
        if mod is None or not isinstance(mod, types.ModuleType):
            mod = types.ModuleType(name)
            sys.modules[name] = mod
    leaf = sys.modules[chain[-1]]
    leaf.pf4h_rr_gather_table = ns["pf4h_rr_gather_table"]
    leaf.pf4h_rr_state = ns["pf4h_rr_state"]


def test_router_gather(files: dict[str, Path]) -> None:
    print("\n[7] router gather: logical -> physical, executed")
    ns = load_rr_namespace(files[EMM_DEST].read_text())
    _install_fake_emm(ns)
    ns["pf4h_rr_state"]().update(
        active=True, num_experts=E, ep_size=EP, ep_rank=0
    )

    fn_src = extract_function(files[BR_DEST].read_text(), "_pf4h_rr_to_physical")
    check(RR_MARK in fn_src, "the router helper carries the marker")

    for on in (False, True):
        gns: dict = {"torch": torch, "_PF4H_RR_ON": on}
        exec(compile(fn_src, "<router>", "exec"), gns)
        gather = gns["_pf4h_rr_to_physical"]
        logical = torch.arange(E, dtype=torch.int32).reshape(32, 8)
        out = gather(logical)
        if not on:
            check(
                out is logical,
                "env gate OFF: the router gather returns topk_ids unchanged",
            )
            continue
        check(out.dtype == torch.int32, "gather preserves the int32 topk dtype")
        check(out.is_contiguous(), "gather output is contiguous")
        check(out is not logical, "gather output is a fresh allocation")
        flat_in = logical.flatten().tolist()
        flat_out = out.flatten().tolist()
        check(
            all(
                q // LOCAL == e % EP and q % LOCAL == e // EP
                for e, q in zip(flat_in, flat_out)
            ),
            "every token's physical id routes to rank e % 8, slot e // 8",
        )
        # int64 path (deepep-style dtype) works too
        out64 = gather(torch.arange(E, dtype=torch.int64).reshape(8, 32))
        check(out64.dtype == torch.int64, "gather works for int64 topk ids")
        # empty batch short-circuit
        empty = torch.zeros((0, 8), dtype=torch.int32)
        check(gather(empty) is empty, "empty topk_ids short-circuits")


# --------------------------------------------------------------------------
# 8  the refusals -- EXECUTED
# --------------------------------------------------------------------------
class _Cfg:
    def __init__(
        self,
        ep_size=EP,
        ep_rank=0,
        mori=True,
        rr_tables=False,
        backend="mori_high_throughput",
    ):
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.use_mori_kernels = mori
        self.needs_round_robin_routing_tables = rr_tables
        self.all2all_backend = backend


def _fresh(files: dict[str, Path], env: dict[str, str]) -> dict:
    ns = load_rr_namespace(files[EMM_DEST].read_text())
    ns["_pf4h_os"] = types.SimpleNamespace(environ=env)
    return ns


def test_refusals(files: dict[str, Path]) -> None:
    print("\n[8] refusals")
    on = {RR_ENV: "1"}

    def configure(env, **kw):
        ns = _fresh(files, env)
        kwargs = dict(
            moe_parallel_config=_Cfg(**kw.pop("cfg", {})),
            global_num_experts=kw.pop("global_num_experts", E),
            num_redundant_experts=kw.pop("num_redundant_experts", 0),
            enable_eplb=kw.pop("enable_eplb", False),
        )
        ns["pf4h_rr_configure"](**kwargs)
        return ns

    def refuses(what, env, **kw):
        try:
            configure(env, **kw)
        except ValueError as exc:
            check(True, f"refuses {what} ({str(exc).splitlines()[0][:60]}...)")
            return
        except Exception as exc:  # pragma: no cover
            check(False, f"refuses {what}: wrong exception {exc!r}")
            return
        check(False, f"refuses {what}: no exception raised")

    refuses("EPLB", on, enable_eplb=True)
    refuses("redundant experts", on, num_redundant_experts=128)
    refuses("a non-MoRI backend", on, cfg={"mori": False, "backend": "deepep_low_latency"})
    refuses("a backend with vLLM's own RR tables", on, cfg={"rr_tables": True})
    refuses("ep_size == 1", on, cfg={"ep_size": 1})
    refuses("an indivisible expert count", on, global_num_experts=250)
    refuses("VLLM_MOE_SKIP_PADDING", {RR_ENV: "1", "VLLM_MOE_SKIP_PADDING": "1"})

    # malformed env value
    try:
        _fresh(files, {RR_ENV: "true"})["pf4h_rr_requested"]()
        check(False, "refuses a malformed env value: no exception")
    except ValueError:
        check(True, "refuses a malformed env value")

    # heterogeneous second layer
    ns = configure(on)
    check(ns["pf4h_rr_state"]()["active"] is True, "a valid config latches RR state")
    try:
        ns["pf4h_rr_configure"](
            moe_parallel_config=_Cfg(ep_rank=3),
            global_num_experts=E,
            num_redundant_experts=0,
            enable_eplb=False,
        )
        check(False, "refuses a heterogeneous second layer: no exception")
    except ValueError:
        check(True, "refuses a heterogeneous second layer")

    # a second identical layer is a no-op, not a refusal
    ns2 = configure(on)
    ns2["pf4h_rr_configure"](
        moe_parallel_config=_Cfg(),
        global_num_experts=E,
        num_redundant_experts=0,
        enable_eplb=False,
    )
    check(True, "a second IDENTICAL layer re-configures without raising")

    # the receipt
    ns3 = configure(on)
    check(
        any("PF4H_RR_PLACEMENT_RECEIPT" in line for line in ns3["logger"].lines),
        "a successful latch emits PF4H_RR_PLACEMENT_RECEIPT",
    )


# --------------------------------------------------------------------------
# 9  the env gate -- '0' is byte-identical decision-making
# --------------------------------------------------------------------------
def _map_global_to_local_replica(rr_state, expert_map, global_id):
    """Executable replica of the patched loader map.

    Kept honest by test_env_gate's source check: every distinctive line of the
    patched body must be present in the installed file.
    """
    if expert_map is None:
        return global_id
    _rr = rr_state
    if _rr["active"] and 0 <= global_id < _rr["num_experts"]:
        _ep = _rr["ep_size"]
        if global_id % _ep != _rr["ep_rank"]:
            return -1
        return global_id // _ep
    return expert_map[global_id]


def test_env_gate(files: dict[str, Path]) -> None:
    print("\n[9] env gate: '0' == the pre-patch decision")
    emm = files[EMM_DEST].read_text()
    for line in (
        '        _rr = _PF4H_RR\n',
        '        if _rr["active"] and 0 <= global_id < _rr["num_experts"]:\n',
        '            _ep = _rr["ep_size"]\n',
        '            if global_id % _ep != _rr["ep_rank"]:\n',
        "                return -1\n",
        "            return global_id // _ep\n",
    ):
        check(line in emm, f"loader-map replica matches source: {line.strip()!r}")

    linear_map = [e - 0 if 0 <= e < LOCAL else -1 for e in range(E)]
    off = {"active": False, "num_experts": 0, "ep_size": 0, "ep_rank": 0}
    check(
        [_map_global_to_local_replica(off, linear_map, e) for e in range(E)]
        == linear_map,
        "env gate OFF: the loader map is the untouched linear map",
    )
    on_state = {"active": True, "num_experts": E, "ep_size": EP, "ep_rank": 0}
    rr_slots = [
        _map_global_to_local_replica(on_state, linear_map, e) for e in range(E)
    ]
    check(
        [e for e, s in enumerate(rr_slots) if s != -1] == list(range(0, E, EP)),
        "env gate ON: rank 0 owns logical experts {0, 8, 16, ...}",
    )
    check(
        [s for s in rr_slots if s != -1] == list(range(LOCAL)),
        "env gate ON: logical e lands at local slot e // 8",
    )
    # the fused-shared-expert tail (ids >= num_experts) still uses expert_map
    tail_map = linear_map + [LOCAL]
    check(
        _map_global_to_local_replica(on_state, tail_map, E) == LOCAL,
        "ids above the routed range fall through to the stock expert_map",
    )

    # the router module constant is read once at import
    br = files[BR_DEST].read_text()
    check(
        '_PF4H_RR_ON = _pf4h_os.environ.get("' + RR_ENV + '", "0") == "1"' in br,
        "router reads the gate once at import into a module constant",
    )
    check(
        "    if not _PF4H_RR_ON:\n        return topk_ids\n" in br,
        "router gather short-circuits on the module constant",
    )
    check(
        "topk_ids = _pf4h_rr_to_physical(topk_ids)" in br
        and br.index("self.capture_fn(topk_ids)")
        < br.index("topk_ids = _pf4h_rr_to_physical(topk_ids)")
        < br.index("topk_ids = self._apply_eplb_mapping(topk_ids)"),
        "the gather sits after capture_fn and before the EPLB mapping",
    )


# --------------------------------------------------------------------------
# 10  the ep_weight_filter guard (the documented env-gate exception)
# --------------------------------------------------------------------------
def test_weight_filter(files: dict[str, Path]) -> None:
    print("\n[10] ep_weight_filter guard")
    src = files[EPWF_DEST].read_text()
    fn = extract_function(src, "compute_local_expert_ids")
    for env, label in (({}, "unset"), ({RR_ENV: "0"}, "'0'")):
        ns = {"_pf4h_os": types.SimpleNamespace(environ=env)}
        exec(compile(fn, "<epwf>", "exec"), ns)
        f = ns["compute_local_expert_ids"]
        check(
            f(E, EP, 0) == set(range(0, LOCAL)),
            f"env {label}: linear placement unchanged (rank 0 -> [0, 32))",
        )
        check(f(E, 1, 0) is None, f"env {label}: ep_size <= 1 still returns None")
        try:
            f(E, EP, 0, placement="round_robin")
            check(False, f"env {label}: round_robin without the gate must raise")
        except ValueError as exc:
            check(
                "expert_placement_strategy" in str(exc),
                f"env {label}: round_robin without the gate is refused",
            )
    ns = {"_pf4h_os": types.SimpleNamespace(environ={RR_ENV: "1"})}
    exec(compile(fn, "<epwf>", "exec"), ns)
    f = ns["compute_local_expert_ids"]
    for r in range(EP):
        if f(E, EP, r, placement="round_robin") != set(range(r, E, EP)):
            check(False, f"env '1': rank {r} filter is {{r, r+8, ...}}")
            break
    else:
        check(True, "env '1': rank r fetches exactly {r, r+8, ...} from disk")
    check(
        f(E, EP, 3, placement="round_robin")
        == {e for e in range(E) if _map_global_to_local_replica(
            {"active": True, "num_experts": E, "ep_size": EP, "ep_rank": 3},
            [0] * E,
            e,
        ) != -1},
        "the disk filter and the loader map agree on every rank's expert set",
    )


# --------------------------------------------------------------------------
# 11  the shim attestation and FULL-021
# --------------------------------------------------------------------------
def _attest_replica(env, rank, values, expected_map, expected_mask):
    """Executable replica of the patched attestation tail (RR portion)."""
    if env.get(RR_ENV, "0") == "1":
        _ep = E // LOCAL
        _rr_map = tuple(
            expert // _ep if expert % _ep == rank else -1 for expert in range(E)
        )
        _rr_mask = tuple(1 if expert % _ep == rank else 0 for expert in range(E))
        if values in (_rr_map, _rr_mask):
            return False
    if values not in (expected_map, expected_mask):
        return False
    return True


def test_shim(files: dict[str, Path]) -> None:
    print("\n[11] shim: attestation polarity and PF4H-FULL-021")
    full = files[FULL_DEST].read_text()
    for line in (
        '        if os.environ.get("' + RR_ENV + '", "0") == "1":\n',
        "            _ep = FROZEN_CONTRACT.global_experts // FROZEN_CONTRACT.local_experts\n",
        "                expert // _ep if expert % _ep == runtime.rank else -1\n",
        "                1 if expert % _ep == runtime.rank else 0\n",
        "            if values in (_rr_map, _rr_mask):\n",
        "                return False\n",
    ):
        check(line in full, f"attestation replica matches source: {line.strip()!r}")

    for rank in range(EP):
        begin = rank * LOCAL
        linear_map = tuple(
            e - begin if begin <= e < begin + LOCAL else -1 for e in range(E)
        )
        linear_mask = tuple(
            1 if begin <= e < begin + LOCAL else 0 for e in range(E)
        )
        rr_map = tuple(e // EP if e % EP == rank else -1 for e in range(E))
        rr_mask = tuple(1 if e % EP == rank else 0 for e in range(E))
        garbage = tuple((e * 7) % 31 - 1 for e in range(E))

        ok = (
            _attest_replica({RR_ENV: "1"}, rank, linear_map, linear_map, linear_mask)
            and _attest_replica(
                {RR_ENV: "1"}, rank, linear_mask, linear_map, linear_mask
            )
            and not _attest_replica(
                {RR_ENV: "1"}, rank, rr_map, linear_map, linear_mask
            )
            and not _attest_replica(
                {RR_ENV: "1"}, rank, rr_mask, linear_map, linear_mask
            )
            and not _attest_replica(
                {RR_ENV: "1"}, rank, garbage, linear_map, linear_mask
            )
            # env off: the decision is exactly the pre-patch one
            and _attest_replica({}, rank, linear_map, linear_map, linear_mask)
            and not _attest_replica({}, rank, garbage, linear_map, linear_mask)
        )
        if not ok:
            check(False, f"attestation polarity holds at rank {rank}")
            break
    else:
        check(
            True,
            "attestation accepts the LINEAR map/mask, refuses the ROUND-ROBIN "
            "pattern and garbage, at every rank",
        )

    # FULL-021 blocker logic
    def full021(env, enable_eplb=False, n_red=0):
        blockers = []
        _rr_env = env.get(RR_ENV, "0")
        if _rr_env not in ("0", "1"):
            blockers.append("PF4H-FULL-021")
        elif _rr_env == "1":
            if bool(enable_eplb) or n_red > 0:
                blockers.append("PF4H-FULL-021")
        return blockers

    check(full021({}) == [], "FULL-021: env unset raises no blocker")
    check(full021({RR_ENV: "0"}, enable_eplb=True) == [], "FULL-021: gate off is inert")
    check(full021({RR_ENV: "1"}) == [], "FULL-021: RR alone is allowed")
    check(
        full021({RR_ENV: "1"}, enable_eplb=True) == ["PF4H-FULL-021"],
        "FULL-021: RR + EPLB is refused",
    )
    check(
        full021({RR_ENV: "1"}, n_red=128) == ["PF4H-FULL-021"],
        "FULL-021: RR + redundant experts is refused",
    )
    check(
        full021({RR_ENV: "yes"}) == ["PF4H-FULL-021"],
        "FULL-021: a malformed gate value is refused",
    )
    for line in (
        '    _rr_env = os.environ.get("' + RR_ENV + '", "0")\n',
        '    if _rr_env not in ("0", "1"):\n',
        '    elif _rr_env == "1":\n',
        "        if _eplb_on or _n_red > 0:\n",
    ):
        check(line in full, f"FULL-021 replica matches source: {line.strip()!r}")


# --------------------------------------------------------------------------
# 12  the campaign wiring
# --------------------------------------------------------------------------
def test_campaign_scripts() -> None:
    print("\n[12] campaign wiring")
    v2 = (SCRATCH / "run_m15_campaign_eplb_v2.sh").read_text()
    v3_path = SCRATCH / "run_m15_campaign_eplb_v3.sh"
    camp4 = SCRATCH / "camp4_rr.sh"
    if not v3_path.exists():
        check(False, "run_m15_campaign_eplb_v3.sh exists")
        return
    v3 = v3_path.read_text()
    check(
        'base_arm() { local a="${1%_rr}"; printf \'%s\' "${a%_eplb}"; }' in v3,
        "v3 base_arm strips both _rr and _eplb",
    )
    check('arm_is_rr() { [[ "$1" == *_rr ]]; }' in v3, "v3 gains arm_is_rr")
    check('readonly RR_PATCH="${RR_PATCH:-}"' in v3, "v3 gains the RR_PATCH env")
    check(
        '${RR_PATCH:+-v "$RR_PATCH:/covpatch/rr_patch.py:ro"}' in v3,
        "v3 mounts rr_patch.py when RR_PATCH is set",
    )
    check(
        "${m23_chain}${rr_chain}exec vllm serve" in v3,
        "v3 chains rr_patch AFTER the M23 step",
    )
    check(
        '--seed 0${eplb_args}${rr_args}"' in v3,
        "v3 appends --expert-placement-strategy like eplb_args",
    )
    check(
        'env_args+=("-e" "' + RR_ENV + '=1")' in v3,
        "v3 sets the placement gate for _rr arms only",
    )
    check(
        'case "$(base_arm "$1")" in' in v3,
        "required_receipts still routes through base_arm (so m15_rr -> m15)",
    )
    # v2 compatibility for non-RR arms: every v2 line survives in v3
    missing = [
        line
        for line in v2.splitlines()
        if line.strip()
        and line not in v3.splitlines()
        and "M15 serving campaign driver" not in line
        and "base_arm() { printf" not in line
        and "${m23_chain}exec vllm serve" not in line
        and "--seed 0${eplb_args}\"" not in line
        and "arms to run: stock" not in line
    ]
    check(not missing, f"v3 keeps every other v2 line verbatim ({len(missing)} lost)")

    if not camp4.exists():
        check(False, "camp4_rr.sh exists")
        return
    c4 = camp4.read_text()
    # Re-chained by the operator: camp4 queues behind the decomposition grid
    # (camp_grid_20260818.log GRID_RC marker), not directly behind camp3.
    check("camp_grid_20260818.log" in c4 and "GRID_RC=" in c4, "camp4 waits on grid")
    check("/home/subvadla/eplb_campaign/RR_READY" in c4, "camp4 waits on RR_READY")
    check(
        "--arms stock,stock_rr --cells c32p --pairs 2" in c4,
        "camp4 A: stock vs stock_rr, c32p, 2 pairs",
    )
    check(
        "--arms stock_rr,m15_rr --cells c32p --pairs 2" in c4,
        "camp4 B: stock_rr vs m15_rr, c32p, 2 pairs",
    )
    check("RUN_TAG=rrcal1" in c4 and "RUN_TAG=rrfair1" in c4, "camp4 run tags")
    check("export EPLB_PREWARM=0" in c4, "camp4 disables the EPLB prewarm")
    check("export PROMPT_SOURCE=qsl" in c4, "camp4 replays the MLPerf QSL")
    check(
        "export M23_PATCH=/home/subvadla/eplb_campaign/m23_patch.py" in c4
        and "export RR_PATCH=/home/subvadla/eplb_campaign/rr_patch.py" in c4,
        "camp4 points at both node-side patchers",
    )
    check("run_m15_campaign_eplb_v3.sh" in c4, "camp4 drives the v3 wrapper")


# --------------------------------------------------------------------------
def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rr_patch_test_"))
    try:
        files = test_chain(tmp / "chain")
        test_order_independence(tmp)
        test_failure_modes(tmp)
        test_physical_numbering(files)
        test_router_gather(files)
        test_refusals(files)
        test_env_gate(files)
        test_weight_filter(files)
        test_shim(files)
        test_campaign_scripts()
    finally:
        leftovers = sorted(str(p) for p in tmp.rglob("*.rrtmp"))
        shutil.rmtree(tmp, ignore_errors=True)
        if leftovers:
            FAILURES.append(f"leftover temp files {leftovers}")

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


def test_everything():  # pytest entry point
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())

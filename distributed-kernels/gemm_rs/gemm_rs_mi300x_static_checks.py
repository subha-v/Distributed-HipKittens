#!/usr/bin/env python3
"""Static correctness gates for the MI300X/gfx942 GEMM-RS megakernel port.

Additively beside ../common/check_port_invariants.py; that suite is unmodified.
These gates audit: the operator contract encoding, the dependency-key layout,
the memory-order/lifetime discipline spelled out in MI300X_DESIGN.md, the
one-launch structure, and the negative-control quarantine.

With --amd-master, immutable donor blobs are verified through `git show` at the
recorded commits; the donor working tree is never read.

Local semantics per gate (1-20) map one-to-one to the task checklist; the
address/dependency/numerical machine-checked parts live in
gemm_rs_mi300x_simulation.py, which this script runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DK = HERE.parent
KERNEL = HERE / "gemm_rs_mi300x.cpp"
ADAPTER = HERE / "gemm_rs_mi300x_hk_adapter.cuh"
HOST_ABI = HERE / "gemm_rs_mi300x_host_abi.hpp"
CONSTANTS = HERE / "gemm_rs_mi300x_constants.cuh"
DESIGN = HERE / "MI300X_DESIGN.md"
LOCK = HERE / "dependencies.mi300x.lock.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read(path: Path) -> str:
    require(path.is_file(), f"missing {path}")
    return path.read_text(encoding="utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Table extraction (single source of truth: the C++ host ABI header).
# ---------------------------------------------------------------------------
TABLE_RE = re.compile(
    r"\{\{\s*(\d+),\s*(\d+),\s*(\d+),\s*(true|false)"
    r"\s*\},\s*\{\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\}\}")


def parse_table(source: str):
    rows = []
    for m in TABLE_RE.finditer(source):
        (m_, n, k, bias, bm, bn, bk, nred, row) = m.groups()
        rows.append({
            "m": int(m_), "n": int(n), "k_local": int(k),
            "bias": bias == "true",
            "bm": int(bm), "bn": int(bn), "bk": int(bk),
            "nred": int(nred), "row": int(row),
        })
    require(len(rows) == 6, f"expected 6 scored rows, parsed {len(rows)}")
    return rows


SCORED = [
    (64, 7168, 18432, False),
    (512, 4096, 12288, True),
    (2048, 2880, 2880, True),
    (4096, 4096, 4096, False),
    (8192, 4096, 14336, True),
    (8192, 8192, 29568, False),
]
SOL_US = [6.46, 8.19, 23.04, 65.54, 131.07, 379.43]
RADEONFLOW_REDUCERS = [32, 48, 48, 48, 32, 8]


def check_gates() -> None:
    adapter = read(ADAPTER)
    host = read(HOST_ABI)
    source = read(KERNEL)
    consts = read(CONSTANTS)

    # Gate 1: world size is exactly eight.
    require("inline constexpr int WORLD_SIZE       = 8;" in consts,
            "gate 1: WORLD_SIZE must be pinned to 8")
    require('"GEMM-RS mi300x fixes world size 8"' in host,
            "gate 1: host static_assert for world 8 missing")

    # Gate 2 + 4: evaluator contract -- M % 8 == 0 enforced at resolution;
    # local K derived as k_global/8; table keys carry k_local.
    require('if (m % world_size != 0)' in host, "gate 2: M % 8 guard missing")
    require("const int k_local = k_global / world_size;" in host,
            "gate 4: local-K derivation must be k_global/world")
    table = parse_table(host)
    for row, (m, n, kg, bias), nred_rf, sol in zip(
            table, SCORED, RADEONFLOW_REDUCERS, SOL_US):
        require((row["m"], row["n"]) == (m, n), f"table row M/N mismatch: {row}")
        require(kg % 8 == 0, f"gate 4: K_global {kg} not divisible by 8")
        require(row["k_local"] == kg // 8,
                f"gate 4: table k_local {row['k_local']} != {kg // 8}")
        require(row["bias"] == bias, f"table bias mismatch on {m}x{n}")
        require(row["nred"] == nred_rf,
                f"row {row['row']}: reducer {row['nred']} != RadeonFlow {nred_rf}")
        require(m % 8 == 0, f"gate 2: scored shape {m} violates M % 8 == 0")

    # Gate 5: every producer band has exactly one destination slot
    # (EB = gcd(BM, M/8) divides both; check per row).
    import math
    for row in table:
        slice_rows = row["m"] // 8
        eb = math.gcd(row["bm"], slice_rows)
        require(eb >= 1 and slice_rows % eb == 0 and row["bm"] % eb == 0,
                f"gate 5: EB does not divide tile/slice for {row}")
        require(row["bm"] % 32 == 0 and row["bm"] // 2 >= 16,
                f"warp fragment misalignment for {row}")
        require(row["bn"] % 64 == 0 and row["bn"] // 4 >= 16,
                f"warp fragment misalignment for {row}")
    require("plan.eb = gcd_int(cfg.bm, slice_rows);" in host,
            "gate 5: EB derivation missing from host resolution")

    # Gate 8: allocation-size accounting exists and is bounded.
    for token in ("plan.c_heap_bytes =", "plan.signal_words_total =",
                  "SIGNAL_U32_CAP", "plan.signal_words_total > SIGNAL_U32_CAP"):
        require(token in host, f"gate 8: sizing missing: {token}")
    require("inline constexpr int SIGNAL_U32_CAP   =" in consts,
            "gate 8: signal cap constant missing")

    # Gate 3/9: tails bounded; packet contract enforced before packet stores;
    # scalar fallback stores bounded.
    for token in ("emit_band_preflight", "emit_band_packets", "emit_band_scalar",
                  "kittens::distributed::packet_contract("):
        require(token in adapter, f"gate 9: emit packet machinery missing: {token}")
    require("kittens::distributed::store_peer_packets(\n        destination, staged, 16u"
            in read(HERE / "gemm_rs_hk_adapter.cuh"),
            "gate 9: packet16 width drifted")
    # packet stores only after preflight in the kernel's emit path
    emit_span = source[source.index("// ---- egress"):source.index(
        "m3::release_payload_system();")]
    require(emit_span.index("emit_band_preflight") < emit_span.index(
        "emit_band_packets"), "gate 9/10: preflight must precede packet emit")

    # ---- source-order proofs on the kernel (gates 10-14) ----
    kernel = source

    # Gate 10: payload stores precede release & publication.
    tile_span = kernel.index("// ---- egress")
    release_at = kernel.index("m3::release_payload_system();", tile_span)
    publish_at = kernel.index("m3::publish_band_epoch(", tile_span)
    require(tile_span < release_at < publish_at,
            "gate 10: emit -> release -> publish order violated")

    # Gate 14: retirement-credit wait precedes the first payload store of a
    # tile's epoch (reused storage): credit wait appears before first stage.
    credit_wait_at = kernel.index("m3::wait_reuse_credit(", tile_span)
    first_stage_at = kernel.index("m3::stage_fragment_bf16(", tile_span)
    require(credit_wait_at < first_stage_at,
            "gate 14: credit wait must precede first egress store")

    # Reducer span: wait -> error-return -> acquire -> consume -> drain ->
    # credit publish (gates 11, 12, 13).
    red_span = kernel.index("// REDUCER ROLE")
    wait_at = kernel.index("m3::wait_band_epoch(", red_span)
    abort_at = kernel.index(
        "if (m3::error_bit_set(errp, m3::ERR_REDUCER_READY)) return;", red_span)
    acquire_at = kernel.index("m3::acquire_payload_system();", red_span)
    consume_at = kernel.index("m3::pull_sum_bf16_strip_mlp8(", red_span)
    drain_at = kernel.index("m3::finish_tile_consumption();", red_span)
    credit_at = kernel.index("m3::publish_reuse_credit(", red_span)
    require(wait_at < abort_at < acquire_at < consume_at < drain_at < credit_at,
            "gates 11/12/13: reducer ordering violated")
    # No payload read is reachable on the timeout path: between the wait and
    # the error check there is no pull/consume token.
    between = kernel[wait_at:abort_at]
    for token in ("pull_sum", "load_packet", "heap["):
        require(token not in between,
                f"gate 12: payload read token before error gate: {token}")

    # Gate 15: bias and reduction order match the oracle. Producer adds bias
    # per element before the single bf16 pack; reducer span has no bias token;
    # consumption is ascending source order p0..p7; one RNE per element.
    require("bias" in adapter and "float_to_bf16_rne(v.x + b0)" in adapter,
            "gate 15: producer-side bias add missing")
    host_at = kernel.index("// Host dispatch:")
    red_text = kernel[red_span:host_at]
    require("bias" not in red_text,
            "gate 15: reducer must not add bias (producer owns it, oracle form)")
    pull = adapter[adapter.index("pull_sum_bf16_strip_mlp8"):]
    order = [pull.find(f"sources.source{s}") for s in range(8)]
    require(all(i >= 0 for i in order) and order == sorted(order),
            "gate 15: reducer consumption must be source-ascending")
    require(pull.count("float_to_bf16_rne") == 9,
            "gate 15: single RNE pack structure drifted")

    # Gate 16: exactly one GPU kernel launch per call after setup.
    dispatch = kernel[kernel.index("void dispatch_gemm_rs_mi300x"):]
    cases = re.findall(r"case \d+:\s*launch_fixed<[^>]+>\(g\);\s*return;", dispatch)
    require(len(cases) == 6, "gate 16: scored dispatch arms drifted")
    require(dispatch.count("<<<") == 0,
            "gate 16: dispatch itself must not launch")
    require(kernel.count(">>>") == 1,
            "gate 16: exactly one kernel launch site required")
    # exactly one kernel definition and one launch site per instantiation
    require(kernel.count("void gemm_rs_mi300x_kernel(") == 1,
            "gate 16: expected exactly one megakernel definition")

    # Gate 17: no per-call host reset or host-generated epoch.
    for token in ("signal_val", "hipMemset", "CUDAGraph", "round_trip"):
        require(token not in kernel,
                f"gate 17: host-reset/stale-capture pattern present: {token}")
    require("epoch32(" in read(HERE / "gemm_rs_hk_adapter.cuh"),
            "gate 17: epochs must be device-derived")

    # Gate 18: no hidden all-rank completion predicate.
    for token in ("grid_barrier", "dist_barrier", "counted_arrive"):
        require(token not in kernel,
                f"gate 18: hidden rendezvous present: {token}")

    # Gate 19: production binding cannot instantiate the controls.
    ctl = kernel.index("#if HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS")
    prod = kernel.index("#else", ctl)
    endp = kernel.index("#endif", prod)
    prod_binding = kernel[prod:endp]
    require("gemm_rs_mi300x_control" not in prod_binding,
            "gate 19: control entry leaked into the production binding")
    ctl_module = kernel[ctl:prod]
    require("gemm_rs_mi300x_control" in ctl_module,
            "gate 19: control module binding missing")
    neg_branches = kernel.count("#if HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS")
    require(neg_branches == 1 + 3,
            "gate 19: every control seam must be macro-gated (binding + 3)")
    require("#define HK_GEMM_RS_MI300X_NEGATIVE_CONTROLS 0" in kernel,
            "gate 19: negative controls must default to 0")

    # Gate 20: compile/launch cache key carries every codegen variable.
    for field in ("config_row", "even_k", "even_n", "arch", "dtype",
                  "has_bias"):
        require(field in host, f"gate 20: cache key missing {field}")
    require('"gfx942"' in host, "gate 20: arch identity missing")


# ---------------------------------------------------------------------------
# Donor-hash verification through git show (never the working tree).
# ---------------------------------------------------------------------------
def git_blob(repo: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=repo, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout


def check_donors(repo: Path) -> None:
    require((repo / ".git").exists(), f"not a Git checkout: {repo}")
    manifest = json.loads(read(LOCK))
    default_commit = manifest["donor_commit"]
    for entry in manifest["donors"]:
        commit = entry.get("verify_commit", default_commit)
        actual = sha256(git_blob(repo, commit, entry["path"]))
        require(actual == entry["sha256"],
                f"donor hash mismatch: {entry['path']} @ {commit} ({actual})")


def check_legacy_free() -> None:
    forbidden = re.compile(r"flow::|irisx::|#include[^\n]*irisx")
    for path in (KERNEL, ADAPTER, HOST_ABI, CONSTANTS):
        match = forbidden.search(read(path))
        if match is not None:
            raise AssertionError(
                f"legacy dependency in {path.name}: {match.group(0)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amd-master", type=Path,
                        help="optional amd-master checkout for blob verification")
    parser.add_argument("--skip-simulation", action="store_true")
    args = parser.parse_args()

    check_legacy_free()
    check_gates()
    if args.amd_master is not None:
        check_donors(args.amd_master.resolve())
    if not args.skip_simulation:
        sim = HERE / "gemm_rs_mi300x_simulation.py"
        result = subprocess.run([sys.executable, str(sim)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        sys.stdout.write(result.stdout.decode())
        if result.returncode != 0:
            raise AssertionError("gemm_rs_mi300x_simulation failed")
    print("gemm_rs mi300x static checks: PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Static gates for the organized GEMM-RS and fused-MoE ports.

With --amd-master, immutable upstream blobs are read through `git show` at the
recorded commits; the donor checkout's working tree is never trusted or changed.
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
GEMM = DK / "gemm_rs"
MOE = DK / "fused_moe"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: Path) -> str:
    require(path.is_file(), f"missing {path}")
    return path.read_text(encoding="utf-8")


def check_no_legacy_code() -> None:
    forbidden = re.compile(r"flow::|irisx::|#include[^\n]*irisx")
    for directory in (GEMM, MOE):
        for path in directory.rglob("*"):
            if path.suffix not in {".cpp", ".hip", ".cuh", ".hpp"}:
                continue
            match = forbidden.search(read(path))
            if match is not None:
                raise AssertionError(
                    f"legacy code dependency in {path}: {match.group(0)}")


def check_gemm() -> None:
    source = read(GEMM / "gemm_rs_device_tile.cpp")
    adapter = read(GEMM / "gemm_rs_hk_adapter.cuh")
    wrapper = read(GEMM / "gemm_rs_dprime_experimental.cpp")

    start = source.index("template<bool HAS_TAIL, int RUNG>")
    marker = source.index("// THE EGRESS EMIT", start)
    end = source.index("\n", marker) + 1
    hot_schedule = source[start:end].encode()
    require(
        sha256(hot_schedule)
        == "4e2c8da5ddf7ff83b1f2feab27607496a6334c85c9b234a9e6fc94821964f5f6",
        "GEMM schedule before the communication epilogue drifted",
    )

    for token in (
        "#define FV 6",
        "#define EV 1",
        "#define REDV 1",
        "LAD_SIG_U32  =",
        "wait_reuse_credit(",
        "publish_reuse_credit(",
        "pull_sum_bf16_tile_mlp8<BLOCK_SIZE, BLOCK_SIZE>",
    ):
        require(token in source, f"GEMM invariant missing: {token}")

    egress = source.index("THE EGRESS EMIT")
    first_store = source.index("emit_c_tile_wide(", egress)
    credit_wait = source.index("wait_reuse_credit(", egress)
    require(credit_wait < first_store, "reuse credit must precede first egress store")

    for token in (
        "struct symmetric_descriptor",
        "sizeof(symmetric_descriptor) == 72",
        "offsetof(symmetric_descriptor, bases) == 8",
        "make_symmetric_descriptor(",
        "kittens::make_parallel_global_layout<8>",
        "kittens::distributed::translate_peer<8>(",
        "kittens::distributed::epoch32(",
        "kittens::distributed::producer_drain_release(",
        "kittens::distributed::bounded_poll_relaxed_into<",
        "kittens::distributed::cta_acquire<",
        "kittens::distributed::bounded_wait_slot_reusable_into(",
        "kittens::distributed::consumer_drain(",
    ):
        require(token in adapter, f"GEMM adapter primitive missing: {token}")

    require("HK_GEMM_RS_DPRIME_EXPERIMENTAL 1" in wrapper,
            "D-prime must remain a separately labelled entry point")


def check_moe() -> None:
    source = read(MOE / "k0pf6gm_device_tile.hip")
    adapter = read(MOE / "moe_hk_adapter.cuh")
    require("ShmemPtrP2p" not in source + adapter,
            "MoE peer translation must use the explicit IRIS heap descriptor")
    for token in (
        "#define K0P6GM_G 3",
        "#define K0P6_D_SYMMETRIC 55",
        "#define K0P6_D_LEN 56",
        '#include "n2_phase1_gm.cpp"',
        '#include "n2_phase2_gm.cpp"',
        "hk_moe::store_dispatch_row(",
        "hk_moe::release_cta_payload_system(",
        "hk_moe::poll_epoch_system(",
        "hk_moe::acquire_payload_system(",
    ):
        require(token in source, f"MoE invariant missing: {token}")

    for token in (
        "desc == nullptr || pperr == nullptr",
        "E <= 0 || E > K0P6_MAXE",
        "cur < 0 || cur >= world",
        "T_loc_max <= 0",
        "PADMAX <= 0 || MAXTOK <= 0",
        "MAXTOK > T_loc_max / 8",
        "input_tokens < 0",
        "input_tokens > MAXTOK",
        "spin_limit < 0",
    ):
        require(token in source, f"MoE entry guard missing: {token}")

    for token in (
        "struct symmetric_heap_descriptor",
        "sizeof(symmetric_heap_descriptor) == 72",
        "offsetof(symmetric_heap_descriptor, heap_bases) == 8",
        "make_symmetric_heap_descriptor(",
        "kittens::distributed::translate_peer<8>(",
        "k0p6_put_row_payload(",
        "kittens::distributed::producer_drain_release(",
        "kittens::distributed::thread_release<",
        "kittens::distributed::reserve_rows_relaxed<",
        "kittens::distributed::bounded_poll_relaxed_into<",
        "kittens::distributed::publish_epoch_word_relaxed<",
        "kittens::distributed::bounded_poll_epoch_word_relaxed_into<",
        "kittens::distributed::cta_acquire<",
    ):
        require(token in adapter, f"MoE adapter primitive missing: {token}")

    # The explicit heap descriptor is phase-local. In particular, no descriptor
    # pointer/value may be materialized across either full-register N2 MFMA body.
    m6 = source.index("// ================= M6:")
    m75 = source.index("// ================= M7.5", m6)
    mfma_span = source[m6:m75]
    require("symmetric" not in mfma_span and
            "K0P6_D_SYMMETRIC" not in mfma_span,
            "IRIS peer-descriptor state leaked into the M6/M7 MFMA span")
    require(source.count("k0p6_symmetric(desc)") == 4,
            "MoE must reload the IRIS descriptor exactly once in M1, M7.5, M8, and M9")

    # Avoid LP64-dependent pointer mismatches with the adapter's std::uint64_t
    # completion APIs. ULL scalars remain valid for ballot/shuffle masks.
    require(re.search(r"(?:const\s+)?unsigned long long\s*\*", source) is None,
            "MoE completion-word pointers must use std::uint64_t")

    # M0 retirement failure must converge across the full grid before any
    # address-reuse reset. Returning from only one CTA would strand later grid
    # barriers; continuing could overwrite a prior peer consumer's payload.
    m0 = source.index("// ================= M0:")
    m1 = source.index("// ================= M1:", m0)
    m0_span = source[m0:m1]
    retire_poll = m0_span.index("poll_value_at_least_system(")
    retire_barrier = m0_span.index("hkp::grid_barrier(", retire_poll)
    retire_abort = m0_span.index(
        "if (hk_moe::error_bit_set_agent(pperr, 16777216 | 2097152)) return;",
        retire_barrier,
    )
    first_reuse_reset = m0_span.index("a2_done[i] = 0u;", retire_abort)
    require(retire_poll < retire_barrier < retire_abort < first_reuse_reset,
            "M0 retirement timeout is not grid-uniform before address reuse")

    # M2 rows/chunk waits all flow through the existing Pass-A rendezvous. The
    # combined failure gate must precede the first settled-word reread or a_ll
    # payload access, and no pre-rendezvous CTA-local return may reappear.
    m2 = source.index("// ================= M2:")
    m3 = source.index("// ================= M3-M5:", m2)
    m2_span = source[m2:m3]
    require("s_abort" not in m2_span,
            "M2 must not return CTA-locally before the Pass-A grid rendezvous")
    pass_a_barrier = m2_span.index("hkp::grid_barrier(")
    combined_abort = m2_span.index(
        "pperr, 16777216 | 8388608 | 2097152)) return;",
        pass_a_barrier,
    )
    settled_reread = m2_span.index("const std::uint64_t w =", combined_abort)
    payload_acquire = m2_span.index("hk_moe::acquire_payload_system();", combined_abort)
    require(pass_a_barrier < combined_abort < settled_reread < payload_acquire,
            "M2 timeout gate must precede completion rereads and a_ll acquire")

    # M7.5 publication is CTA-leader-only: each publishing thread performs its
    # own post-grid SYSTEM release before its stripe of relaxed row_ready stores.
    # A leader fence must never be used to order another lane's publication.
    m8 = source.index("// ================= M8:", m75)
    publish_span = source[m75:m8]
    leader_gate = publish_span.index("if (phase2_payload_valid && tid == 0)")
    publisher_release = publish_span.index(
        "hk_moe::release_signal_batch_system();", leader_gate)
    leader_stripe = publish_span.index(
        "for (int r = bid; r < T_ext; r += nct)", publisher_release)
    ready_store = publish_span.index(
        "hk_moe::publish_epoch<hk_moe::scope::system>(", leader_stripe)
    require(leader_gate < publisher_release < leader_stripe < ready_store,
            "M7.5 row_ready stores lack same-thread leader release ordering")
    require("bid * (int)blockDim.x + tid" not in publish_span,
            "M7.5 publication must not escape the CTA-leader stripe")


def check_moe_mps() -> None:
    """Additive static gates for the minimum-progress specialization sibling.

    These checks never touch the parity-port assertions above; the MPS kernel,
    vendored phase-2 body, adapter, and host ABI are separate artifacts.
    """
    source = read(MOE / "k0pf6gm_device_tile_mps.hip")
    adapter = read(MOE / "moe_mps_adapter.cuh")
    vendored = read(MOE / "n2_phase2_gm_mps.cpp")
    host_abi = read(MOE / "moe_host_abi.hpp")
    roles = (DK.parent / "include" / "cdna4" / "ops" / "group" /
             "distributed" / "roles.cuh")
    roles_src = read(roles)
    aggregate = read(roles.parent / "distributed.cuh")

    require("ShmemPtrP2p" not in source + adapter,
            "MPS peer translation must use the explicit IRIS heap descriptor")
    for token in (
        "#define K0P6GM_G 3",
        "k0pf6gm_mps_mega",
        '#include "n2_phase1_gm.cpp"',
        '#include "n2_phase2_gm_mps.cpp"',
        "finish_order_partition(",
        "hk_moe::mps::run_service(",
        "hk_moe::mps::enqueue_tile_release(",
        "K0P6_D_MPS_CFG",
        "N2GM_TASK_DONE_DRAIN_HOOK asm volatile(\"s_waitcnt vmcnt(0)\"",
        "bcap > 16383",
        "K0P6_MPS_ERR_CONFIG",
        "K0P6_MPS_ERR_SERVICE",
    ):
        require(token in source, f"MPS kernel invariant missing: {token}")
    require('#include "n2_phase2_gm.cpp"' not in source,
            "MPS kernel must use the vendored phase-2 body, not the donor")
    require(source.count("k0p6_symmetric(desc)") == 6,
            "MPS must reload the IRIS descriptor exactly once in M1, M7.6, "
            "M7.5(mode 0), M7.6b(mode 1), M8, and M9")
    # No descriptor value may be materialized across either MFMA body. The MPS
    # span ends at the M7.6 service phase: that phase legitimately reloads the
    # IRIS descriptor AFTER the phase-2 MFMA body has completed on this CTA.
    m6 = source.index("// ================= M6:")
    m76 = source.index("// ================= M7.6", m6)
    mfma_span = source[m6:m76]
    require("symmetric" not in mfma_span and
            "K0P6_D_SYMMETRIC" not in mfma_span,
            "IRIS peer-descriptor state leaked into the M6/M7 MFMA span")
    # M7's task loop must stride by the logical pool, not the physical grid
    m7 = source.index("// ================= M7:")
    m76 = source.index("// ================= M7.6", m7)
    m7_span = source[m7:m76]
    require("k0p6_role" in m7_span and
            "n2p6gm_mps_phase2_body" in m7_span,
            "M7 must run the vendored body under the packed role")
    require("if (!mps_service)" in m7_span,
            "service CTAs must skip the M7 GEMM body")

    for token in (
        "#define K0P6_D_MPS_Q 56",
        "#define K0P6_D_MPS_NCARR 57",
        "#define K0P6_D_MPS_PUSHED 58",
        "#define K0P6_D_MPS_CLAIM 59",
        "#define K0P6_D_MPS_STATE 60",
        "#define K0P6_D_MPS_SLOTS 61",
        "#define K0P6_D_MPS_CFG 62",
        "#define K0P6_MPS_D_LEN 63",
        "encode_config(",
        "decode_config(",
        "config_is_valid(",
        "wait_event_nonempty(",
        "push_slice_group(",
        "flush_pending(",
        "kittens::distributed::store_peer_packets(",
        "kittens::distributed::publish_tile_release<",
    ):
        require(token in adapter, f"MPS adapter primitive missing: {token}")
    # The group-completeness probe must be an RMW (fetch_add +0), so the last
    # completing bumper provably observes every group member at target.
    require("env.nc_arr + (std::size_t)r * 16u + n2, 0u)" in adapter,
            "MPS group probe must use an RMW, not a relaxed load")

    # Vendored body: donor lineage + the two behavioral deltas, in order.
    require(
        "7d8beb039b8224e614eacdbe9b34b21837095ffa6ec4842f08d72358b7b8e025"
        in vendored, "vendored phase-2 must record the pinned donor sha256")
    start_macro = vendored.index("#ifndef N2GM_TASK_START")
    drain_hook = vendored.index("#ifdef N2GM_TASK_DONE_DRAIN_HOOK")
    refill_note = vendored.index(
        "// The next task refills every LDS buffer this one just read.")
    done_hook = vendored.index("#ifdef N2GM_TASK_DONE_HOOK", drain_hook)
    require(start_macro < drain_hook < refill_note < done_hook,
            "vendored phase-2 deltas out of order: start/stride, drain, "
            "syncthreads, done hook")
    loop_line = vendored.index(
        "for (int task = (int)(N2GM_TASK_START); task < num_tasks;")
    require(loop_line > start_macro,
            "vendored phase-2 task loop must consume the start/stride macros")

    for token in (
        "struct role_partition",
        "finish_order_partition(",
        "publish_tile_release(",
        "wait_tile_acquire_into(",
        "retire_epoch(",
    ):
        require(token in roles_src, f"roles.cuh primitive missing: {token}")
    require('#include "roles.cuh"' in aggregate,
            "distributed.cuh must export roles.cuh")

    for token in (
        "mps_descriptor_words = 63",
        "struct mps_buffer_binding",
        "validate_mps_binding(",
        "patch_mps_slots(",
        "append_mps_descriptor(",
        "validate_mps_descriptor(",
        "mps_slots_bytes(",
    ):
        require(token in host_abi, f"MPS host ABI helper missing: {token}")


def git_blob(repo: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def verify_manifest(repo: Path, manifest_path: Path, commit_key: str,
                    files_key: str) -> None:
    manifest = json.loads(read(manifest_path))
    commit = manifest[commit_key]
    for entry in manifest[files_key]:
        actual = sha256(git_blob(repo, commit, entry["path"]))
        require(actual == entry["sha256"],
                f"upstream hash mismatch: {entry['path']} ({actual})")


def check_upstream(repo: Path) -> None:
    require((repo / ".git").exists(), f"not a Git checkout: {repo}")
    verify_manifest(repo, GEMM / "dependencies.lock.json", "donor_commit", "donors")
    verify_manifest(repo, MOE / "dependencies.lock.json", "source_commit", "files")


def check_gemm_mi300x(amd_master: Path | None) -> None:
    """Run the MI300X/gfx942 port's own static gate suite (additive layer)."""
    cmd = [sys.executable, str(GEMM / "gemm_rs_mi300x_static_checks.py")]
    if amd_master is not None:
        cmd += ["--amd-master", str(amd_master)]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amd-master", type=Path,
                        help="optional amd-master checkout for immutable blob verification")
    args = parser.parse_args()

    check_no_legacy_code()
    check_gemm()
    check_moe()
    check_moe_mps()
    check_gemm_mi300x(args.amd_master.resolve() if args.amd_master else None)
    if args.amd_master is not None:
        check_upstream(args.amd_master.resolve())
    print("distributed kernel port invariants: PASS")


if __name__ == "__main__":
    main()

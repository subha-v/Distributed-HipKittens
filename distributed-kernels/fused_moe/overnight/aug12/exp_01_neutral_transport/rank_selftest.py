#!/usr/bin/env python3
"""CPU-only contract tests for the final rank-per-GPU transport base."""

from __future__ import annotations

import copy
import json
import pathlib
import tempfile

import rank_summarize


HERE = pathlib.Path(__file__).resolve().parent
TOTAL = 64 * (1 << 20)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def point(
    method: str,
    record_bytes: int,
    lifetime: str,
    values: list[float],
) -> dict:
    host = method == "host_copy_path"
    depth = 1 if lifetime == "one_epoch" else 2
    controls = {
        "no_publication": {
            "applicable": True,
            "detected": True,
            "detector": "bounded_poll_timeout",
        },
        "redirected_destination": {
            "applicable": True,
            "detected": True,
            "detector": "digest_sample_poison",
        },
        "early_publication": {
            "applicable": True,
            "detected": True,
            "valid": True,
            "detector": "digest_sample_poison",
        },
        "dropped_credit": {
            "applicable": depth == 2,
            "detected": True,
            "detector": "bounded_credit_timeout",
        },
        "early_reuse": {
            "applicable": depth == 2,
            "detected": True,
            "detector": "generation_digest",
        },
    }
    return {
        "schema_version": rank_summarize.SCHEMA_VERSION,
        "point_id": (
            f"rank2|{method}|rb{record_bytes}|tb{TOTAL}|{lifetime}|d{depth}|rev1"
        ),
        "status": "gated",
        "method": method,
        "comparison_family": "transport_equivalent",
        "ranks": 2,
        "record_bytes": record_bytes,
        "records_per_rank": (TOTAL + record_bytes - 1) // record_bytes,
        "total_bytes_per_rank": TOTAL,
        "tail_record_bytes": TOTAL % record_bytes or record_bytes,
        "direction": "bidirectional",
        "fanout": 1,
        "lifetime_mode": lifetime,
        "ping_pong_depth": depth,
        "memory_type": "hipMalloc_ipc",
        "executor": {
            "label": "unknown" if host else "cu",
            "verified": not host,
            "trace_artifact": None,
            "transport_selector": None,
        },
        "work": {
            "payload_bytes": 2 * TOTAL,
            "control_bytes": 8,
            "publications": 2,
            "descriptors": 2 if host else 0,
            "launches": 8 if host else 10,
            "queue_depth": None,
        },
        "correctness": {
            "digest_pass": True,
            "samples_pass": True,
            "poison_survivors": 0,
            "generation_pass": True,
            "all_rank_errors": 0,
            "negative_controls": controls,
            "soak_epochs": 600,
            "soak_pass": True,
        },
        "timing": {
            "campaigns": 1,
            "iterations_per_campaign": len(values),
            "global_makespan_us": {
                "p50": rank_summarize.quantile(values, 0.5),
                "p95": rank_summarize.quantile(values, 0.95),
                "values": values,
            },
            "tp_us": 10.0,
            "tm_us": 100.0,
            "tcons_us": 20.0,
            "tserial_us": None,
            "tideal_us": None,
            "tjoint_us": rank_summarize.quantile(values, 0.5),
            "enqueue_us": 2.0,
            "completion_after_final_op_us": 5.0 if host else 0.0,
            "gbps": 2 * TOTAL / (rank_summarize.quantile(values, 0.5) * 1e3),
            "launch_skew_us": 1.0,
            "rank_tail_us": 2.0,
            "device_host_timer_agreement_pass": True,
            "per_direction": [
                {
                    "source_rank": source,
                    "destination_rank": 1 - source,
                    "component_spans_us": {
                        "producer": 10.0,
                        "transport_hip_event": 100.0,
                        "consumer_hip_event": 20.0,
                        "transport_device_clock": 99.0,
                        "consumer_device_clock": 20.5,
                    },
                }
                for source in (0, 1)
            ],
        },
        "provenance": {
            "source_rev": 1,
            "arch": "gfx950",
            "process_model": "mpi_rank_per_gpu",
            "mpi_world_size": 2,
            "allocation": "hipMalloc_ipc",
        },
        "blocker": None,
    }


def test_static_source_contract() -> None:
    source = (HERE / "rank_transport.hip").read_text(encoding="utf-8")
    runner = (HERE / "run_rank.sh").read_text(encoding="utf-8")
    build = (HERE / "build_rank.sh").read_text(encoding="utf-8")
    required_source = (
        "EXP01_RANK_SRC_REV",
        "MPI_Comm_split_type",
        "hipIpcGetMemHandle",
        "hipIpcOpenMemHandle",
        "hipDeviceCanAccessPeer",
        "hipMemcpyPeerAsync",
        "cu_push_packets_rank",
        "cu_pull_packets_rank",
        "translate_peer<PeerRank>",
        "store_peer_packets",
        "load_peer_packets",
        "release_and_publish",
        "bounded_observe_acquire_into",
        "bounded_wait_slot_reusable_into",
        "drain_and_retire_slot",
        "RANK_TIMING_FORBIDDEN",
        'memory_type',
        'hipMalloc_ipc',
        'comparison_family',
        'transport_equivalent',
    )
    for needle in required_source:
        require(needle in source, f"rank_transport.hip omits {needle}")
    for needle in (
        "branch --show-current",
        "ls-remote origin refs/heads/ablations",
        "rocm-smi --showpids",
        "setsid timeout --signal=TERM",
        "--bind-to core",
    ):
        require(needle in runner, f"run_rank.sh omits guard {needle}")
    require(
        "blocked_mpi_toolchain" in build,
        "build_rank.sh lacks precise MPI toolchain blocker",
    )


def test_validation_and_summary() -> None:
    push_small = point("cu_push", 256, "ping_pong", [110, 108, 109, 111, 300])
    pull_small = point("cu_pull", 256, "ping_pong", [90, 91, 92, 89, 250])
    push_large = point(
        "cu_push", 1 << 20, "ping_pong", [80, 81, 82, 79, 200]
    )
    pull_large = point(
        "cu_pull", 1 << 20, "ping_pong", [100, 101, 99, 102, 300]
    )
    host = point(
        "host_copy_path", 1 << 20, "ping_pong", [70, 71, 72, 69, 180]
    )
    one_epoch = point("cu_push", 65536, "one_epoch", [95, 96, 94])
    candidates = [
        push_small,
        pull_small,
        push_large,
        pull_large,
        host,
        one_epoch,
    ]
    for candidate in candidates:
        require(
            not rank_summarize.validate_point(candidate),
            f"valid point rejected: {candidate['point_id']}",
        )

    bad_memory = copy.deepcopy(push_small)
    bad_memory["memory_type"] = "hip_device_peer_access"
    require(
        any("hipMalloc_ipc" in error for error in rank_summarize.validate_point(bad_memory)),
        "allocation stratum drift was accepted",
    )

    bad_host = copy.deepcopy(host)
    bad_host["executor"] = {"label": "sdma", "verified": True}
    require(
        any("unknown" in error for error in rank_summarize.validate_point(bad_host)),
        "untraced host copy was mislabeled SDMA",
    )

    bad_control = copy.deepcopy(push_small)
    bad_control["correctness"]["negative_controls"]["early_publication"][
        "valid"
    ] = False
    require(
        any("early publication" in error for error in rank_summarize.validate_point(bad_control)),
        "invalid early-publication control did not block the point",
    )

    bad_soak = copy.deepcopy(push_small)
    bad_soak["correctness"]["soak_epochs"] = 599
    require(
        any("600" in error for error in rank_summarize.validate_point(bad_soak)),
        "strict 600-epoch soak was not enforced",
    )

    with tempfile.TemporaryDirectory() as directory:
        source = pathlib.Path(directory) / "points.jsonl"
        destination = pathlib.Path(directory) / "summary.json"
        with source.open("w", encoding="utf-8") as handle:
            for candidate in candidates:
                handle.write(json.dumps(candidate) + "\n")
            handle.write('{"truncated":')
        loaded, warnings = rank_summarize.load_jsonl(source)
        require(len(loaded) == len(candidates), "truncated JSONL lost valid records")
        require(bool(warnings), "truncated JSONL did not produce a warning")
        document = rank_summarize.summarize(loaded, str(source))
        destination.write_text(json.dumps(document, indent=2), encoding="utf-8")

    require(document["status"] == "gated", "valid points did not summarize gated")
    require(document["n_points"] == len(candidates), "wrong point count")
    require(
        document["crossovers"]["cu_push_vs_cu_pull"],
        "known push/pull ordering flip did not create a crossover bracket",
    )
    require(
        document["points"][0]["timing"]["global_makespan_us"]["p95"]
        < max(document["points"][0]["timing"]["global_makespan_us"]["values"]),
        "p95 collapsed to the maximum",
    )


def main() -> int:
    test_static_source_contract()
    test_validation_and_summary()
    print("SELFTEST_RANK_TRANSPORT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

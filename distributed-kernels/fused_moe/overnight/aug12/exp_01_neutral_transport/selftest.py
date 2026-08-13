#!/usr/bin/env python3
"""CPU-only tests for the neutral-transport JSONL and summary contract."""

from __future__ import annotations

import copy
import json
import pathlib
import tempfile

import summarize


HERE = pathlib.Path(__file__).resolve().parent
MIB = 1 << 20


def point(method: str, size: int, values: list[float]) -> dict:
    total = 64 * MIB
    executor = "unknown" if method == "host_copy_path" else "cu"
    return {
        "schema_version": summarize.SCHEMA_VERSION,
        "point_id": f"{method}|rb{size}|one_epoch",
        "status": "gated",
        "method": method,
        "comparison_family": "transport_equivalent",
        "ranks": 2,
        "record_bytes": size,
        "records_per_rank": (total + size - 1) // size,
        "total_bytes_per_rank": total,
        "tail_record_bytes": total % size or size,
        "direction": "bidirectional",
        "fanout": 1,
        "lifetime_mode": "one_epoch",
        "memory_type": "hip_device_peer_access",
        "executor": {
            "label": executor,
            "verified": method != "host_copy_path",
            "trace_artifact": None,
        },
        "work": {
            "payload_bytes": 2 * total,
            "control_bytes": 8,
            "publications": 2,
            "descriptors": 2 if method == "host_copy_path" else 0,
            "launches": 4,
            "queue_depth": None,
        },
        "correctness": {
            "digest_pass": True,
            "samples_pass": True,
            "poison_survivors": 0,
            "generation_pass": True,
            "all_rank_errors": 0,
            "negative_controls": {
                name: {"applicable": True, "detected": True, "detector": detector}
                for name, detector in (
                    ("redirected_destination", "digest_sample_poison"),
                    ("early_publication", "digest_sample_poison"),
                    ("no_publication", "bounded_poll_timeout"),
                )
            },
            "soak_epochs": 600,
            "soak_pass": True,
        },
        "timing": {
            "campaigns": 1,
            "iterations_per_campaign": len(values),
            "global_makespan_us": {
                "p50": summarize.quantile(values, 0.50),
                "p95": summarize.quantile(values, 0.95),
                "values": values,
            },
            "per_direction": [
                {
                    "source_rank": source,
                    "destination_rank": 1 - source,
                    "component_spans_us": {
                        "producer": 10.0,
                        "transport": 100.0,
                        "consumer": 20.0,
                    },
                }
                for source in (0, 1)
            ],
            "gbps": (2 * total) / (summarize.quantile(values, 0.50) * 1.0e3),
            "device_host_timer_agreement_pass": True,
        },
        "provenance": {"source_rev": 1, "arch": "gfx950"},
        "blocker": None,
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    schema = json.loads((HERE / "schema.json").read_text(encoding="utf-8"))
    methods = set(schema["properties"]["method"]["enum"])
    require(
        {"cu_push", "cu_pull", "host_copy_path"} <= methods,
        "schema omits a diagnostic method",
    )

    push_small = point("cu_push", 256, [110.0, 100.0, 105.0, 102.0, 500.0])
    pull_small = point("cu_pull", 256, [90.0, 92.0, 91.0, 93.0, 400.0])
    push_large = point("cu_push", 1 << 20, [80.0, 81.0, 82.0, 83.0, 300.0])
    pull_large = point("cu_pull", 1 << 20, [100.0, 101.0, 102.0, 103.0, 400.0])
    host = point("host_copy_path", 1 << 20, [70.0, 71.0, 72.0, 73.0, 200.0])

    for candidate in (push_small, pull_small, push_large, pull_large, host):
        require(not summarize.validate_point(candidate), "valid point rejected")

    bad_soak = copy.deepcopy(push_small)
    bad_soak["correctness"]["soak_epochs"] = 599
    require(
        any("600" in err for err in summarize.validate_point(bad_soak)),
        "strict soak was not enforced",
    )

    bad_executor = copy.deepcopy(host)
    bad_executor["executor"]["label"] = "sdma"
    bad_executor["executor"]["verified"] = True
    require(
        any("unknown" in err for err in summarize.validate_point(bad_executor)),
        "untraced host copy was mislabeled",
    )

    bad_control = copy.deepcopy(push_small)
    bad_control["correctness"]["negative_controls"]["early_publication"][
        "detected"
    ] = False
    require(
        any("negative control" in err for err in summarize.validate_point(bad_control)),
        "undetected negative control did not block the point",
    )

    with tempfile.TemporaryDirectory() as td:
        src = pathlib.Path(td) / "points.jsonl"
        dst = pathlib.Path(td) / "summary.json"
        with src.open("w", encoding="utf-8") as handle:
            for candidate in (
                push_small,
                pull_small,
                push_large,
                pull_large,
                host,
            ):
                handle.write(json.dumps(candidate) + "\n")
            handle.write('{"truncated":')
        loaded, warnings = summarize.load_jsonl(src)
        require(len(loaded) == 5, "truncated line handling lost valid points")
        require(warnings, "truncated line did not produce a warning")
        document = summarize.summarize_points(loaded, str(src))
        dst.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    require(document["n_points"] == 5, "wrong point count")
    require(document["status"] == "gated", "valid inputs did not summarize gated")
    require(
        document["methods"]["cu_push"]["points"] == 2,
        "method aggregation is wrong",
    )
    require(
        document["crossovers"]["cu_push_vs_cu_pull"],
        "known push/pull ordering flip did not produce a crossover bracket",
    )
    require(
        document["points"][0]["timing"]["global_makespan_us"]["p95"]
        < max(document["points"][0]["timing"]["global_makespan_us"]["values"]),
        "quantile logic collapsed p95 to maximum",
    )

    print("SELFTEST_NEUTRAL_TRANSPORT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

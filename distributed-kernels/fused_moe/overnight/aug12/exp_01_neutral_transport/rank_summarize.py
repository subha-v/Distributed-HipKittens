#!/usr/bin/env python3
"""Validate and summarize final rank-per-GPU Stage-0/1 transport JSONL.

Usage:
  python3 rank_summarize.py rank-points.jsonl transport_crossover.json [--print]
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Iterable


SCHEMA_VERSION = "exp01.transport-point.v1"
SUMMARY_VERSION = "exp01.rank-transport-summary.v1"
METHODS = ("cu_push", "cu_pull", "host_copy_path")
LIFETIMES = ("one_epoch", "ping_pong")
BASE_CONTROLS = ("no_publication", "redirected_destination", "early_publication")
PING_PONG_CONTROLS = ("dropped_credit", "early_reuse")
STAGE1_SIZES = (
    256,
    1024,
    4096,
    14336,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
)
MIB = 1 << 20


def quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile of empty series")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_jsonl(path: str | pathlib.Path) -> tuple[list[dict], list[str]]:
    points: list[dict] = []
    warnings: list[str] = []
    with pathlib.Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                warnings.append(
                    f"line {line_number}: skipped invalid JSON ({error.msg})"
                )
                continue
            if not isinstance(value, dict):
                warnings.append(f"line {line_number}: skipped non-object JSON")
                continue
            points.append(value)
    return points, warnings


def _require(mapping: object, fields: Iterable[str], prefix: str) -> list[str]:
    if not isinstance(mapping, dict):
        return [f"{prefix} must be an object"]
    return [f"{prefix}.{name} is required" for name in fields if name not in mapping]


def validate_point(point: dict) -> list[str]:
    errors = _require(
        point,
        (
            "schema_version",
            "point_id",
            "status",
            "method",
            "comparison_family",
            "ranks",
            "record_bytes",
            "records_per_rank",
            "total_bytes_per_rank",
            "direction",
            "fanout",
            "lifetime_mode",
            "ping_pong_depth",
            "memory_type",
            "executor",
            "work",
            "correctness",
            "timing",
            "provenance",
        ),
        "point",
    )
    if errors:
        return errors
    if point["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if point["method"] not in METHODS:
        errors.append("method is outside the rank transport method set")
    if point["comparison_family"] != "transport_equivalent":
        errors.append("comparison_family must be transport_equivalent")
    if point["ranks"] != 2 or point["direction"] != "bidirectional":
        errors.append("final base requires two-rank bidirectional points")
    if point["fanout"] != 1:
        errors.append("fanout must be one")
    if point["lifetime_mode"] not in LIFETIMES:
        errors.append("lifetime_mode must be one_epoch or ping_pong")
    expected_depth = 1 if point["lifetime_mode"] == "one_epoch" else 2
    if point["ping_pong_depth"] != expected_depth:
        errors.append("ping_pong_depth disagrees with lifetime_mode")
    if point["memory_type"] != "hipMalloc_ipc":
        errors.append("memory_type must be hipMalloc_ipc")
    if point["total_bytes_per_rank"] != 64 * MIB:
        errors.append("fixed-volume Stage-0/1 points must move 64 MiB/rank")
    if point["record_bytes"] not in STAGE1_SIZES:
        errors.append("record_bytes is outside the pre-registered Stage-1 list")
    expected_records = math.ceil(
        point["total_bytes_per_rank"] / point["record_bytes"]
    )
    if point["records_per_rank"] != expected_records:
        errors.append("records_per_rank does not cover exactly the fixed volume")

    executor = point["executor"]
    errors.extend(_require(executor, ("label", "verified"), "executor"))
    if isinstance(executor, dict):
        if point["method"] == "host_copy_path":
            if executor.get("label") != "unknown" or executor.get("verified") is not False:
                errors.append(
                    "host_copy_path must remain executor=unknown/unverified until trace"
                )
        elif executor.get("label") != "cu" or executor.get("verified") is not True:
            errors.append("CU methods require executor=cu/verified")

    provenance = point["provenance"]
    errors.extend(
        _require(
            provenance,
            ("source_rev", "arch", "process_model", "allocation"),
            "provenance",
        )
    )
    if isinstance(provenance, dict):
        if provenance.get("process_model") != "mpi_rank_per_gpu":
            errors.append("process_model must be mpi_rank_per_gpu")
        if provenance.get("allocation") != "hipMalloc_ipc":
            errors.append("provenance allocation must be hipMalloc_ipc")

    work = point["work"]
    errors.extend(
        _require(
            work,
            ("payload_bytes", "control_bytes", "publications", "descriptors"),
            "work",
        )
    )
    if isinstance(work, dict):
        if work.get("payload_bytes") != 2 * point["total_bytes_per_rank"]:
            errors.append("payload_bytes must include both symmetric directions")
        if work.get("publications") != 2:
            errors.append("exactly one publication per source/destination is required")
        expected_descriptors = 2 if point["method"] == "host_copy_path" else 0
        if work.get("descriptors") != expected_descriptors:
            errors.append("descriptor count disagrees with method")

    correctness = point["correctness"]
    errors.extend(
        _require(
            correctness,
            (
                "digest_pass",
                "samples_pass",
                "poison_survivors",
                "generation_pass",
                "all_rank_errors",
                "negative_controls",
                "soak_epochs",
                "soak_pass",
            ),
            "correctness",
        )
    )
    if isinstance(correctness, dict):
        controls = correctness.get("negative_controls", {})
        required = list(BASE_CONTROLS)
        if point["lifetime_mode"] == "ping_pong":
            required.extend(PING_PONG_CONTROLS)
        for name in required:
            control = controls.get(name) if isinstance(controls, dict) else None
            if not isinstance(control, dict) or control.get("detected") is not True:
                errors.append(f"required negative control {name} did not flip")
        early = controls.get("early_publication") if isinstance(controls, dict) else None
        if not isinstance(early, dict) or early.get("valid") is not True:
            errors.append("early publication control is invalid or silently passed")

        if point["status"] == "gated":
            for name in ("digest_pass", "samples_pass", "generation_pass", "soak_pass"):
                if correctness.get(name) is not True:
                    errors.append(f"gated point requires correctness.{name}=true")
            if correctness.get("poison_survivors") != 0:
                errors.append("gated point has surviving poison")
            if correctness.get("all_rank_errors") != 0:
                errors.append("gated point has rank errors")
            if correctness.get("soak_epochs") != 600:
                errors.append("gated point requires exactly 600 soak epochs")

    timing = point["timing"]
    errors.extend(
        _require(
            timing,
            (
                "campaigns",
                "iterations_per_campaign",
                "global_makespan_us",
                "per_direction",
                "gbps",
                "launch_skew_us",
                "rank_tail_us",
                "device_host_timer_agreement_pass",
            ),
            "timing",
        )
    )
    if isinstance(timing, dict):
        makespan = timing.get("global_makespan_us")
        errors.extend(_require(makespan, ("p50", "p95", "values"), "timing.global_makespan_us"))
        if isinstance(makespan, dict):
            values = makespan.get("values", [])
            if not isinstance(values, list):
                errors.append("global makespan values must be an array")
            elif len(values) != timing.get("iterations_per_campaign"):
                errors.append("iteration count does not match global makespan values")
        directions = timing.get("per_direction")
        if not isinstance(directions, list) or len(directions) != 2:
            errors.append("per_direction must contain both rank directions")
        elif {
            (entry.get("source_rank"), entry.get("destination_rank"))
            for entry in directions
            if isinstance(entry, dict)
        } != {(0, 1), (1, 0)}:
            errors.append("per_direction does not contain exact 0->1 and 1->0 entries")
        if point["status"] == "gated" and timing.get(
            "device_host_timer_agreement_pass"
        ) is not True:
            errors.append("gated point failed local device/HIP-event timer agreement")
    return errors


def canonicalize(point: dict) -> dict:
    result = copy.deepcopy(point)
    values = result["timing"]["global_makespan_us"]["values"]
    result["timing"]["global_makespan_us"]["p50"] = quantile(values, 0.50)
    result["timing"]["global_makespan_us"]["p95"] = quantile(values, 0.95)
    result["timing"]["gbps"] = (
        2.0
        * result["total_bytes_per_rank"]
        / (result["timing"]["global_makespan_us"]["p50"] * 1.0e3)
    )
    return result


def crossover_brackets(points: list[dict], left: str, right: str) -> list[dict]:
    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for point in points:
        grouped[(point["lifetime_mode"], point["record_bytes"])][
            point["method"]
        ] = point
    by_lifetime: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (lifetime, size), methods in grouped.items():
        if left not in methods or right not in methods:
            continue
        delta = (
            methods[left]["timing"]["global_makespan_us"]["p50"]
            - methods[right]["timing"]["global_makespan_us"]["p50"]
        )
        by_lifetime[lifetime].append((size, delta))
    brackets: list[dict] = []
    for lifetime, series in by_lifetime.items():
        series.sort()
        for (lower_size, lower_delta), (upper_size, upper_delta) in zip(
            series, series[1:]
        ):
            if lower_delta == 0 or lower_delta * upper_delta <= 0:
                brackets.append(
                    {
                        "lifetime_mode": lifetime,
                        "lower_record_bytes": lower_size,
                        "upper_record_bytes": upper_size,
                        "lower_delta_us": lower_delta,
                        "upper_delta_us": upper_delta,
                    }
                )
    return brackets


def summarize(points: list[dict], source: str) -> dict:
    latest: dict[str, dict] = {}
    duplicates: set[str] = set()
    for point in points:
        point_id = str(point.get("point_id", ""))
        if point_id in latest:
            duplicates.add(point_id)
        latest[point_id] = point

    canonical: list[dict] = []
    invalid: dict[str, list[str]] = {}
    for point_id, point in latest.items():
        errors = validate_point(point)
        if errors:
            invalid[point_id or "<missing-point-id>"] = errors
        else:
            canonical.append(canonicalize(point))
    canonical.sort(
        key=lambda point: (
            point["lifetime_mode"],
            point["record_bytes"],
            point["method"],
        )
    )

    methods: dict[str, dict] = {}
    for method in METHODS:
        selected = [point for point in canonical if point["method"] == method]
        if selected:
            p50s = [
                point["timing"]["global_makespan_us"]["p50"] for point in selected
            ]
            methods[method] = {
                "points": len(selected),
                "gated_points": sum(point["status"] == "gated" for point in selected),
                "median_point_p50_us": statistics.median(p50s),
                "record_bytes": sorted({point["record_bytes"] for point in selected}),
            }
    all_gated = bool(canonical) and all(
        point["status"] == "gated" for point in canonical
    )
    return {
        "schema_version": SUMMARY_VERSION,
        "experiment": "aug12 exp_01 final rank-per-GPU neutral transport",
        "scope": "two-rank Stage-0/1 transport-equivalent fixed-volume base",
        "source_jsonl": source,
        "status": (
            "invalid"
            if invalid
            else "gated"
            if all_gated
            else "rank_transport_ungated"
        ),
        "n_points": len(canonical),
        "n_invalid": len(invalid),
        "duplicate_point_ids_latest_wins": sorted(duplicates),
        "validation_errors": invalid,
        "methods": methods,
        "crossovers": {
            "cu_push_vs_cu_pull": crossover_brackets(
                canonical, "cu_push", "cu_pull"
            ),
            "cu_push_vs_host_copy_path": crossover_brackets(
                canonical, "cu_push", "host_copy_path"
            ),
            "cu_pull_vs_host_copy_path": crossover_brackets(
                canonical, "cu_pull", "host_copy_path"
            ),
        },
        "points": canonical,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    source, destination = argv[1], argv[2]
    points, warnings = load_jsonl(source)
    document = summarize(points, source)
    document["load_warnings"] = warnings
    with pathlib.Path(destination).open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if "--print" in argv:
        print(
            f"{document['status']}: {document['n_points']} point(s), "
            f"{document['n_invalid']} invalid"
        )
        for warning in warnings:
            print(f"WARN: {warning}", file=sys.stderr)
    return 1 if document["validation_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

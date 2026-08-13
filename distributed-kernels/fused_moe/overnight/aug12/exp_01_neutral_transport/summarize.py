#!/usr/bin/env python3
"""Validate neutral-transport JSONL and emit a plot-ready summary.

Usage:
  python3 summarize.py points.jsonl transport_crossover.json [--print]
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
SUMMARY_VERSION = "exp01.transport-summary.v1"
METHODS = {"cu_push", "cu_pull", "host_copy_path"}
LIFETIMES = {"one_epoch", "ping_pong"}
REQUIRED_CONTROLS = {
    "redirected_destination",
    "early_publication",
    "no_publication",
}


def quantile(values: Iterable[float], q: float) -> float:
    """R-7/NumPy-style linearly interpolated quantile."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile of empty series")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
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
                warnings.append(f"line {line_number}: skipped invalid JSON ({error.msg})")
                continue
            if not isinstance(value, dict):
                warnings.append(f"line {line_number}: skipped non-object JSON")
                continue
            points.append(value)
    return points, warnings


def _missing(mapping: dict, names: Iterable[str], prefix: str) -> list[str]:
    return [f"{prefix}.{name} is required" for name in names if name not in mapping]


def validate_point(point: dict) -> list[str]:
    errors = _missing(
        point,
        (
            "schema_version",
            "point_id",
            "status",
            "method",
            "ranks",
            "record_bytes",
            "records_per_rank",
            "total_bytes_per_rank",
            "direction",
            "fanout",
            "memory_type",
            "executor",
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
        errors.append("method is outside the diagnostic method set")
    if point["ranks"] != 2 or point["direction"] != "bidirectional":
        errors.append("diagnostic points must be two-rank bidirectional")
    if point["fanout"] != 1:
        errors.append("diagnostic fanout must be one")
    if point.get("lifetime_mode") not in LIFETIMES:
        errors.append("lifetime_mode must be one_epoch or ping_pong")
    if point["record_bytes"] <= 0 or point["records_per_rank"] <= 0:
        errors.append("record and count fields must be positive")
    if point["total_bytes_per_rank"] != 64 * (1 << 20):
        errors.append("fixed-volume diagnostic must move exactly 64 MiB/rank")

    executor = point["executor"]
    if not isinstance(executor, dict):
        errors.append("executor must be an object")
    else:
        if point["method"] == "host_copy_path":
            if executor.get("label") != "unknown" or executor.get("verified") is not False:
                errors.append(
                    "host_copy_path executor must remain unknown and unverified"
                )
        elif executor.get("label") != "cu":
            errors.append("CU methods must carry executor label cu")

    correctness = point["correctness"]
    errors.extend(
        _missing(
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
    if point["status"] == "gated" and not errors:
        for name in (
            "digest_pass",
            "samples_pass",
            "generation_pass",
            "soak_pass",
        ):
            if correctness[name] is not True:
                errors.append(f"gated point requires correctness.{name}=true")
        if correctness["poison_survivors"] != 0:
            errors.append("gated point has surviving inbox poison")
        if correctness["all_rank_errors"] != 0:
            errors.append("gated point has rank errors")
        if correctness["soak_epochs"] != 600:
            errors.append("gated point requires exactly 600 soak epochs")
        controls = correctness["negative_controls"]
        for name in sorted(REQUIRED_CONTROLS):
            control = controls.get(name) if isinstance(controls, dict) else None
            if not isinstance(control, dict) or control.get("detected") is not True:
                errors.append(f"required negative control {name} did not flip")

    timing = point["timing"]
    errors.extend(
        _missing(
            timing,
            (
                "campaigns",
                "iterations_per_campaign",
                "global_makespan_us",
                "per_direction",
                "gbps",
                "device_host_timer_agreement_pass",
            ),
            "timing",
        )
    )
    if not errors:
        series = timing["global_makespan_us"].get("values", [])
        if len(series) != timing["iterations_per_campaign"]:
            errors.append("iteration count does not match global makespan series")
        if point["status"] == "gated" and timing[
            "device_host_timer_agreement_pass"
        ] is not True:
            errors.append("gated point failed device/host timer agreement")
        directions = timing["per_direction"]
        if not isinstance(directions, list) or len(directions) != 2:
            errors.append("per_direction must contain both transfer directions")
    return errors


def _canonicalize(point: dict) -> dict:
    result = copy.deepcopy(point)
    values = result["timing"]["global_makespan_us"]["values"]
    if values:
        p50 = quantile(values, 0.50)
        p95 = quantile(values, 0.95)
        result["timing"]["global_makespan_us"]["p50"] = p50
        result["timing"]["global_makespan_us"]["p95"] = p95
        result["timing"]["gbps"] = (
            2.0 * result["total_bytes_per_rank"] / (p50 * 1.0e3)
        )
    return result


def _crossover_brackets(points: list[dict], left: str, right: str) -> list[dict]:
    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for point in points:
        key = (point.get("lifetime_mode", ""), point["record_bytes"])
        grouped[key][point["method"]] = point
    brackets: list[dict] = []
    by_lifetime: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (lifetime, size), methods in grouped.items():
        if left not in methods or right not in methods:
            continue
        left_us = methods[left]["timing"]["global_makespan_us"]["p50"]
        right_us = methods[right]["timing"]["global_makespan_us"]["p50"]
        by_lifetime[lifetime].append((size, left_us - right_us))
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


def summarize_points(points: list[dict], source_jsonl: str) -> dict:
    # Resumption may append a replacement after a blocked point. Latest wins.
    deduplicated: dict[str, dict] = {}
    duplicate_ids: list[str] = []
    for point in points:
        point_id = str(point.get("point_id", ""))
        if point_id in deduplicated:
            duplicate_ids.append(point_id)
        deduplicated[point_id] = point

    canonical: list[dict] = []
    validation_errors: dict[str, list[str]] = {}
    for point_id, point in deduplicated.items():
        errors = validate_point(point)
        if errors:
            validation_errors[point_id or "<missing-point-id>"] = errors
        else:
            canonical.append(_canonicalize(point))
    canonical.sort(
        key=lambda point: (
            point["record_bytes"],
            point.get("lifetime_mode", ""),
            point["method"],
        )
    )

    methods: dict[str, dict] = {}
    for method in sorted(METHODS):
        selected = [point for point in canonical if point["method"] == method]
        if not selected:
            continue
        p50s = [
            point["timing"]["global_makespan_us"]["p50"] for point in selected
        ]
        methods[method] = {
            "points": len(selected),
            "gated_points": sum(point["status"] == "gated" for point in selected),
            "median_point_p50_us": statistics.median(p50s),
            "record_bytes": [point["record_bytes"] for point in selected],
        }

    all_gated = bool(canonical) and all(point["status"] == "gated" for point in canonical)
    return {
        "schema_version": SUMMARY_VERSION,
        "experiment": "aug12 exp_01 one-process neutral transport diagnostic",
        "scope": "fast diagnostic; not final rank-per-GPU MORI/IRIS comparison",
        "source_jsonl": source_jsonl,
        "status": (
            "invalid"
            if validation_errors
            else "gated"
            if all_gated
            else "diagnostic_ungated"
        ),
        "n_points": len(canonical),
        "n_invalid": len(validation_errors),
        "duplicate_point_ids_latest_wins": sorted(set(duplicate_ids)),
        "validation_errors": validation_errors,
        "methods": methods,
        "crossovers": {
            "cu_push_vs_cu_pull": _crossover_brackets(
                canonical, "cu_push", "cu_pull"
            ),
            "cu_push_vs_host_copy_path": _crossover_brackets(
                canonical, "cu_push", "host_copy_path"
            ),
            "cu_pull_vs_host_copy_path": _crossover_brackets(
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
    document = summarize_points(points, source)
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

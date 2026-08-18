#!/usr/bin/env python3
"""Zero-GPU falsifier for the M15 + EPLB placement-only compose.

Runs vLLM's OWN rearrangement policy over the router-skew histograms the
sitecustomize hook already writes, and reports the per-rank load before and
after.  The megakernel's cost is set by the busiest rank's received-row count,
so ``max_rank_after / fair_share`` is the number that decides whether the
compose is worth a node run at all.

Inputs: one or more ``skew_rank*_pid*.json`` files produced by
``mirror/m15_router_skew.py`` (``M15_SKEW_OUT=<dir>``).  Each carries
``per_layer_histogram: {layer_key: [256 counts]}``.

The policy call reproduces ``EplbState.rearrange`` for the placement-only
case exactly: ``num_replicas = 256`` (no redundancy), ``num_groups`` from the
model (8 for DeepSeek-R1), ``num_nodes = 1`` for a single 8-GPU box, and
``num_gpus = 8``.  With ``num_nodes == 1`` the hierarchical policy's group
packing is a no-op and step 3 is an unconstrained balanced packing of all 256
experts into 8 bins of 32 -- which is exactly the mechanism that can break the
measured rank-0 concentration.

Usage:
    python3 eplb0_predict.py --vllm-root <site-packages> skew_rank*.json
    python3 eplb0_predict.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GLOBAL_EXPERTS = 256
NUM_RANKS = 8
LOCAL_EXPERTS = GLOBAL_EXPERTS // NUM_RANKS


def load_histograms(paths: list[Path]) -> dict[str, list[int]]:
    """Sum every rank's per-layer histogram into one global-load table."""

    merged: dict[str, list[int]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        experts = int(payload.get("global_experts", GLOBAL_EXPERTS))
        if experts != GLOBAL_EXPERTS:
            raise SystemExit(
                f"{path}: global_experts={experts}, expected {GLOBAL_EXPERTS}"
            )
        for key, counts in payload["per_layer_histogram"].items():
            row = merged.setdefault(key, [0] * GLOBAL_EXPERTS)
            for index, value in enumerate(counts):
                row[index] += int(value)
    if not merged:
        raise SystemExit("no per_layer_histogram entries found")
    return merged


def rank_loads(counts: list[int], phy2log: list[int] | None) -> list[int]:
    """Per-rank slot load for a placement.

    ``phy2log[p]`` is the logical expert occupying physical slot ``p``; rank
    ``p // 32`` owns it.  ``None`` means the identity placement (EPLB off),
    which is what the megakernel runs today.
    """

    loads = [0] * NUM_RANKS
    if phy2log is None:
        for expert, value in enumerate(counts):
            loads[expert // LOCAL_EXPERTS] += value
        return loads
    for slot, logical in enumerate(phy2log):
        loads[slot // LOCAL_EXPERTS] += counts[logical]
    return loads


def imbalance(loads: list[int]) -> tuple[float, float]:
    """(max rank load as a multiple of fair share, share of the busiest rank)."""

    total = sum(loads)
    if total == 0:
        return (0.0, 0.0)
    peak = max(loads)
    return (peak / (total / NUM_RANKS), peak / total)


def rebalance(weights: list[list[int]], num_groups: int):
    """Call vLLM's DefaultEplbPolicy for the placement-only configuration."""

    import torch
    from vllm.distributed.eplb.policy.default import DefaultEplbPolicy

    weight = torch.tensor(weights, dtype=torch.float32)
    return DefaultEplbPolicy.rebalance_experts(
        weight,
        GLOBAL_EXPERTS,  # num_replicas: no redundancy
        num_groups,
        1,  # num_nodes: one 8-GPU box
        NUM_RANKS,
    ).tolist()


def report(merged: dict[str, list[int]], num_groups: int) -> int:
    keys = sorted(merged)
    weights = [merged[key] for key in keys]
    phy2log = rebalance(weights, num_groups)

    print(f"{'layer':<14}{'before':>10}{'after':>10}{'r0 before':>12}"
          f"{'r0 after':>11}")
    worst_before = worst_after = 0.0
    for index, key in enumerate(keys):
        counts = merged[key]
        before = rank_loads(counts, None)
        after = rank_loads(counts, phy2log[index])
        peak_before, _ = imbalance(before)
        peak_after, _ = imbalance(after)
        total = max(1, sum(counts))
        worst_before = max(worst_before, peak_before)
        worst_after = max(worst_after, peak_after)
        print(
            f"{key:<14}{peak_before:>10.3f}{peak_after:>10.3f}"
            f"{before[0] / total:>12.3f}{after[0] / total:>11.3f}"
        )
    print()
    print(f"worst-layer max-rank load  before={worst_before:.3f}x  "
          f"after={worst_after:.3f}x  fair-share=1.000x")
    # The mega pays the busiest rank on every layer, so the projected dispatch
    # + GEMM ceiling scales with this ratio.  A ratio that barely moves means
    # the skew is per-batch, not stationary, and placement-only cannot help.
    if worst_before > 0:
        print(f"projected per-layer critical-rank relief: "
              f"{100.0 * (1.0 - worst_after / worst_before):.1f}%")
    return 0


def self_test() -> int:
    """No vLLM required: exercise the load accounting on a synthetic skew."""

    counts = [0] * GLOBAL_EXPERTS
    for expert in range(LOCAL_EXPERTS):
        counts[expert] = 100  # everything on rank 0
    for expert in range(LOCAL_EXPERTS, GLOBAL_EXPERTS):
        counts[expert] = 10
    before = rank_loads(counts, None)
    assert before[0] == 3200, before
    assert imbalance(before)[0] > 4.0
    # A placement that spreads the 32 hot experts one per rank... with 8 ranks
    # and 32 slots each, put four hot experts on each rank.
    phy2log = []
    for rank in range(NUM_RANKS):
        phy2log.extend(range(rank * 4, rank * 4 + 4))
        phy2log.extend(
            range(
                LOCAL_EXPERTS + rank * (LOCAL_EXPERTS - 4),
                LOCAL_EXPERTS + (rank + 1) * (LOCAL_EXPERTS - 4),
            )
        )
    assert sorted(phy2log) == list(range(GLOBAL_EXPERTS))
    after = rank_loads(counts, phy2log)
    assert len(set(after)) == 1, after
    assert abs(imbalance(after)[0] - 1.0) < 1e-9
    print("self-test ok: accounting reproduces a 4.0x -> 1.0x flattening")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skew", nargs="*", type=Path)
    parser.add_argument(
        "--num-groups",
        type=int,
        default=8,
        help="model expert groups (DeepSeek-R1: 8)",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.skew:
        parser.error("pass at least one skew_rank*.json, or --self-test")
    return report(load_histograms(args.skew), args.num_groups)


if __name__ == "__main__":
    sys.exit(main())

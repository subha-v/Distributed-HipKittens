#!/usr/bin/env python3
"""Captured-route replay for the MoK synthetic prefill benchmark.

Consumes the .npz files written by the serving router-capture hook
(skewhook_v2/sitecustomize.py) and hands the harness a stack of REAL
per-chunk routes, replayed VERBATIM one per timed iteration.

Why this exists.  Every kernel-level MoK number so far replays either a
balanced i.i.d. route or an aggregate 256-bin popularity histogram
(K0_MOK_ROUTE_HIST, Gumbel-top-k).  Both fix only the MARGINAL expert
popularity and draw each token independently.  Real chunks are
run-correlated: adjacent tokens of a prompt route alike, so a contiguous
4096-row chunk concentrates destinations well beyond the aggregate, and
wall-time is convex in that concentration.  That is the standing (never
measured) explanation for serving regressions the aggregate replay failed to
predict.  Replay here is exact, so the question becomes measurable.

Two orders, one multiset -- the separator:
  captured  rows exactly as the router emitted them (position => prompt =>
            run correlation intact; a chunk is what the kernel really saw)
  shuffled  rows permuted with a fixed seed.  The multiset of routed rows
            (hence every expert count, every destination-rank total, the
            whole aggregate histogram) is IDENTICAL; only the token ORDER
            changes, which is exactly the within-chunk correlation.
A captured-vs-shuffled delta is caused by run correlation and nothing else.

File contract (written by the hook; see CAPTURE_KEYS):
  format_version int32   1
  topk_ids       [N,tokens,topk] uint8    (256 experts fit a byte)
  topk_weights   [N,tokens,topk] float16
  layers         [N] unicode              router key per captured call
  call_index     [N] int32                that layer's own call index
  seq            [N] int32                capture order within the rank
  tokens, topk, num_experts, rank, pid    int32 scalars
  captured_utc   unicode scalar

CLI:
  python3 route_replay.py --file <npz|dir> [--rank 0] [--order shuffled]
  python3 route_replay.py --selftest
"""

from __future__ import annotations

import functools
import glob
import hashlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

CAPTURE_FORMAT_VERSION = 1
CAPTURE_KEYS = (
    "format_version",
    "topk_ids",
    "topk_weights",
    "layers",
    "call_index",
    "seq",
    "tokens",
    "topk",
    "num_experts",
    "rank",
    "pid",
    "captured_utc",
)
ORDERS = ("captured", "shuffled")


@dataclass(frozen=True)
class RouteCapture:
    """A rank's replayable route stack.  Immutable; arrays are read-only."""

    topk_ids: np.ndarray       # [N, tokens, topk] int32
    topk_weights: np.ndarray   # [N, tokens, topk] float32
    layers: tuple[str, ...]
    call_index: tuple[int, ...]
    order: str
    seed: int
    metadata: dict[str, Any]

    @property
    def calls(self) -> int:
        return int(self.topk_ids.shape[0])


def write_capture_npz(
    path: str,
    topk_ids: np.ndarray,
    topk_weights: np.ndarray,
    layers,
    call_index=None,
    *,
    num_experts: int = 256,
    rank: int = 0,
    pid: int = 0,
    captured_utc: str = "19700101T000000Z",
) -> str:
    """Write one capture file in the EXACT hook format (test + tooling path).

    The serving hook cannot import this module (it runs as sitecustomize
    inside vLLM workers, before anything of ours is importable), so it holds
    its own copy of these keys.  test_route_loader.py builds its fixtures
    through THIS function and asserts the key set, which is what keeps the two
    writers from drifting apart.
    """

    ids = np.ascontiguousarray(topk_ids).astype(np.uint8)
    wgt = np.ascontiguousarray(topk_weights).astype(np.float16)
    if ids.ndim != 3 or wgt.shape != ids.shape:
        raise ValueError(f"capture arrays must be [N,tokens,topk]; got {ids.shape} / {wgt.shape}")
    n = ids.shape[0]
    lay = np.array([str(x) for x in layers], dtype="U32")
    if lay.shape != (n,):
        raise ValueError(f"layers must hold {n} entries, got {lay.shape}")
    if call_index is None:
        call_index = np.arange(n, dtype=np.int32)
    cidx = np.asarray(call_index, dtype=np.int32)
    if cidx.shape != (n,):
        raise ValueError(f"call_index must hold {n} entries, got {cidx.shape}")
    payload = dict(
        format_version=np.int32(CAPTURE_FORMAT_VERSION),
        topk_ids=ids,
        topk_weights=wgt,
        layers=lay,
        call_index=cidx,
        seq=np.arange(n, dtype=np.int32),
        tokens=np.int32(ids.shape[1]),
        topk=np.int32(ids.shape[2]),
        num_experts=np.int32(num_experts),
        rank=np.int32(rank),
        pid=np.int32(pid),
        captured_utc=np.array(captured_utc, dtype="U32"),
    )
    tmp = path + ".tmp.npz"
    with open(tmp, "wb") as handle:
        np.savez(handle, **payload)
    os.replace(tmp, path)
    return path


def resolve_capture_path(path: str, rank: int) -> str:
    """Resolve K0_MOK_ROUTE_FILE to ONE file for this rank.

    A file is used as given.  A directory is searched for this rank's own
    capture (routes_rank<rank>_pid*.npz) so every rank replays the routes IT
    produced in serving, preserving the joint per-step load across ranks; when
    a rank has several (one per PID incarnation) the largest capture wins,
    ties broken by name so all ranks are deterministic.  A directory holding
    no file for this rank falls back to the rank-th file in sorted order,
    which keeps a partial capture usable and is recorded in the metadata.
    """

    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"route capture path not found: {path!r}")
    own = sorted(glob.glob(os.path.join(path, f"routes_rank{rank}_pid*.npz")))
    if own:
        return max(own, key=lambda p: (os.path.getsize(p), p))
    every = sorted(glob.glob(os.path.join(path, "routes_rank*_pid*.npz")))
    if not every:
        raise FileNotFoundError(
            f"no routes_rank*_pid*.npz under {path!r} (did the capture pass run?)"
        )
    return every[rank % len(every)]


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def multiset_digest(ids: np.ndarray) -> str:
    """Order-independent digest of the routed-row multiset of a call stack.

    Sorts the (topk-sorted) rows of every call, so it is invariant under the
    token permutation the shuffled order applies and sensitive to any change
    of WHICH rows are present.  The captured/shuffled pair must agree on it --
    that equality is the proof that the arms differ only in correlation.
    """

    ids = np.asarray(ids, dtype=np.int32)
    flat = ids.reshape(ids.shape[0], ids.shape[1], -1)
    rows = np.sort(flat, axis=2)
    order = np.lexsort(tuple(rows[:, :, c] for c in range(rows.shape[2] - 1, -1, -1)))
    sorted_rows = np.take_along_axis(rows, order[:, :, None], axis=1)
    return hashlib.sha256(np.ascontiguousarray(sorted_rows).tobytes(order="C")).hexdigest()


def _permutation(seed: int, call: int, tokens: int) -> np.ndarray:
    """Fixed-seed row permutation: reproducible for (seed, call) forever."""

    return np.random.default_rng([int(seed), int(call)]).permutation(tokens)


def _skew_stats(ids: np.ndarray, world_size: int, num_experts: int) -> dict[str, Any]:
    """Per-call destination-rank concentration -- the quantity that is convex.

    max_rank_x_uniform = (rows landing on the busiest destination rank) /
    (fair share).  The aggregate histogram fixes its MEAN; the tail is what
    the kernel actually pays, and reporting p50/p95/max here is what makes a
    captured-vs-shuffled result interpretable without re-deriving it.
    """

    local = num_experts // world_size
    dest = (ids.astype(np.int64) // local).reshape(ids.shape[0], -1)
    per_call = np.zeros((ids.shape[0], world_size), dtype=np.int64)
    for r in range(world_size):
        per_call[:, r] = (dest == r).sum(axis=1)
    total = per_call.sum(axis=1, keepdims=True).clip(min=1)
    share = per_call / total
    ratio = share.max(axis=1) * world_size
    agg = per_call.sum(axis=0)
    agg_ratio = float(agg.max()) / float(max(agg.sum(), 1)) * world_size
    counts = np.bincount(ids.reshape(-1).astype(np.int64), minlength=num_experts)
    srt = np.sort(counts.astype(np.float64))
    n = srt.size
    total_c = srt.sum() or 1.0
    gini = float((2.0 * np.sum((np.arange(1, n + 1)) * srt)) / (n * total_c) - (n + 1) / n)
    return {
        "max_rank_x_uniform_p50": float(np.percentile(ratio, 50)),
        "max_rank_x_uniform_p95": float(np.percentile(ratio, 95)),
        "max_rank_x_uniform_max": float(ratio.max()),
        "max_rank_x_uniform_aggregate": agg_ratio,
        "expert_gini_aggregate": gini,
        "top1_expert_x_uniform": float(counts.max()) / float(max(counts.sum(), 1)) * num_experts,
        "experts_used": int((counts > 0).sum()),
    }


@functools.lru_cache(maxsize=8)
def load_capture(
    path: str,
    rank: int,
    tokens: int,
    topk: int,
    num_experts: int,
    world_size: int,
    order: str = "captured",
    seed: int = 1234,
    layers: tuple[str, ...] | None = None,
    max_calls: int = 0,
) -> RouteCapture:
    """Load, validate and (optionally) reorder one rank's captured routes.

    Cached: the harness and synthetic_inputs both need the same stack (call 0
    seeds the setup route and the reference; the whole stack drives the timed
    loop) and must not disagree about it.
    """

    if order not in ORDERS:
        raise ValueError(f"route order {order!r} must be one of {ORDERS}")
    resolved = resolve_capture_path(path, rank)
    with np.load(resolved, allow_pickle=False) as blob:
        present = set(blob.files)
        missing = [k for k in CAPTURE_KEYS if k not in present]
        if missing:
            raise ValueError(f"{resolved}: capture is missing {missing}")
        version = int(blob["format_version"])
        if version != CAPTURE_FORMAT_VERSION:
            raise ValueError(
                f"{resolved}: capture format v{version}, this loader speaks "
                f"v{CAPTURE_FORMAT_VERSION}"
            )
        raw_ids = np.asarray(blob["topk_ids"])
        raw_wgt = np.asarray(blob["topk_weights"])
        file_layers = [str(x) for x in blob["layers"].tolist()]
        file_calls = [int(x) for x in blob["call_index"].tolist()]
        file_tokens = int(blob["tokens"])
        file_topk = int(blob["topk"])
        file_experts = int(blob["num_experts"])
        file_rank = int(blob["rank"])
        captured_utc = str(blob["captured_utc"])
    if raw_ids.ndim != 3 or raw_ids.shape[1:] != (tokens, topk):
        raise ValueError(
            f"{resolved}: topk_ids is {raw_ids.shape}, harness needs [N,{tokens},{topk}]"
        )
    if raw_wgt.shape != raw_ids.shape:
        raise ValueError(
            f"{resolved}: topk_weights {raw_wgt.shape} != topk_ids {raw_ids.shape}"
        )
    if (file_tokens, file_topk, file_experts) != (tokens, topk, num_experts):
        raise ValueError(
            f"{resolved}: capture is (tokens={file_tokens}, topk={file_topk}, "
            f"experts={file_experts}); harness is ({tokens}, {topk}, {num_experts})"
        )
    if len(file_layers) != raw_ids.shape[0] or len(file_calls) != raw_ids.shape[0]:
        raise ValueError(f"{resolved}: layers/call_index length disagrees with topk_ids")

    keep = list(range(raw_ids.shape[0]))
    if layers:
        wanted = set(layers)
        keep = [i for i in keep if file_layers[i] in wanted]
        if not keep:
            raise ValueError(
                f"{resolved}: no captured call matches layers={sorted(wanted)}; "
                f"file holds {sorted(set(file_layers))}"
            )
    if max_calls and len(keep) > max_calls:
        keep = keep[:max_calls]

    ids = raw_ids[keep].astype(np.int32)
    wgt = raw_wgt[keep].astype(np.float32)
    if ids.min() < 0 or ids.max() >= num_experts:
        raise ValueError(
            f"{resolved}: expert ids out of range [0,{num_experts}): "
            f"[{int(ids.min())},{int(ids.max())}]"
        )
    captured_digest = multiset_digest(ids)
    if order == "shuffled":
        # Permute TOKEN ROWS only.  Every row travels intact, so the multiset
        # of routed rows -- and therefore the whole aggregate histogram -- is
        # bit-identical; what dies is the correlation between neighbouring
        # rows, i.e. the concentration inside a contiguous chunk.
        for j in range(ids.shape[0]):
            # Key on the file's own capture index, not the per-layer call
            # index: every captured call is the FIRST call of its layer, so
            # call_index is 0 for all of them and would give one shared
            # permutation.  keep[j] is unique and stable under layer filtering.
            perm = _permutation(seed, int(keep[j]), tokens)
            ids[j] = ids[j][perm]
            wgt[j] = wgt[j][perm]
        if multiset_digest(ids) != captured_digest:
            raise AssertionError("shuffled order changed the routed-row multiset")

    ids.setflags(write=False)
    wgt.setflags(write=False)
    metadata = {
        "route_file_arg": path,
        "route_file": os.path.abspath(resolved),
        "route_file_sha256": _sha256_file(resolved),
        "route_file_rank": file_rank,
        "route_file_is_own_rank": file_rank == rank,
        "route_format_version": version,
        "route_captured_utc": captured_utc,
        "route_order": order,
        "route_seed": int(seed),
        "route_calls": int(ids.shape[0]),
        "route_calls_in_file": int(raw_ids.shape[0]),
        "route_layers": [file_layers[i] for i in keep],
        "route_call_index": [file_calls[i] for i in keep],
        "route_multiset_sha256": captured_digest,
        "route_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(ids).tobytes(order="C")
        ).hexdigest(),
        "route_skew": _skew_stats(ids, world_size, num_experts),
    }
    return RouteCapture(
        topk_ids=ids,
        topk_weights=wgt,
        layers=tuple(file_layers[i] for i in keep),
        call_index=tuple(file_calls[i] for i in keep),
        order=order,
        seed=int(seed),
        metadata=metadata,
    )


def load_for_config(config, rank: int) -> RouteCapture:
    """Load through a MoKSyntheticConfig (duck-typed: no import cycle)."""

    return load_capture(
        path=config.route_file,
        rank=int(rank),
        tokens=int(config.tokens_per_rank),
        topk=int(config.topk),
        num_experts=int(config.num_experts),
        world_size=int(config.world_size),
        order=str(config.route_order),
        seed=int(config.route_seed),
        layers=tuple(config.route_layers) if config.route_layers else None,
        max_calls=int(config.route_max_calls),
    )


def _selftest() -> int:
    import tempfile

    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        n, tokens, topk, experts = 4, 64, 8, 256
        ids = np.stack(
            [
                np.stack([rng.choice(experts, size=topk, replace=False) for _ in range(tokens)])
                for _ in range(n)
            ]
        )
        wgt = rng.random((n, tokens, topk), dtype=np.float32)
        path = os.path.join(tmp, "routes_rank0_pid1.npz")
        write_capture_npz(path, ids, wgt, [f"router{i:03d}" for i in range(n)])
        a = load_capture(path, 0, tokens, topk, experts, 8, "captured", 1234)
        b = load_capture(path, 0, tokens, topk, experts, 8, "shuffled", 1234)
        assert a.calls == n and b.calls == n
        assert a.metadata["route_multiset_sha256"] == b.metadata["route_multiset_sha256"]
        assert not np.array_equal(a.topk_ids, b.topk_ids)
        print("route_replay selftest OK")
    return 0


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", help="capture .npz or a directory of them")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--order", default="captured", choices=list(ORDERS))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--layers", default="")
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()
    if not args.file:
        parser.error("--file is required unless --selftest")
    layers = tuple(x.strip() for x in args.layers.split(",") if x.strip()) or None
    capture = load_capture(
        args.file, args.rank, args.tokens, args.topk, args.experts, args.world,
        args.order, args.seed, layers, args.max_calls,
    )
    print(json.dumps(capture.metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

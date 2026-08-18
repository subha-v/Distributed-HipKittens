#!/usr/bin/env python3
"""Closed-concurrency vLLM scout with byte-identical prompt-token replay.

This is an A/B performance scout, not an official MLPerf LoadGen run.  It
records MLPerf-relevant TTFT/TPOT/E2EL distributions and an ordered digest of
the exact token-ID stream so the stock and PF4H arms can prove identical input.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from typing import Any

import aiohttp


def _tokens(index: int, *, seed: int, length: int) -> list[int]:
    """Generate one deterministic, valid-range, sample-unique token sequence."""
    state = (seed ^ ((index + 1) * 0x9E3779B9)) & 0xFFFFFFFF
    result: list[int] = []
    for _ in range(length):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        result.append(1000 + state % 30000)
    # Distinguish samples before any long shared prefix so prefix caching cannot
    # turn this into a cache benchmark.
    result[0] = 1000 + index % 30000
    if length > 1:
        result[1] = 1000 + (index // 30000) % 30000
    return result



# --------------------------------------------------------------------------- #
# Real-text prompt source: the official MLPerf DeepSeek-R1 QSL                 #
# --------------------------------------------------------------------------- #
# Synthetic prompts give uniform-random token IDs, so expert routing sees no
# realistic popularity skew.  This path replays REAL pre-tokenised prompts.
#
# The official 4,388-sample set has NO sample at ISL >= 4096 (measured:
# min 52, mean 803.8, median 901.5, max 3140), so an exact-4096 request is
# built by concatenating consecutive real prompts (~5.1 of them on average)
# and truncating to exactly `length`.  The token IDs stay real, which is what
# expert routing actually keys on; only the document boundary is artificial.
_QSL_CACHE: dict[str, list[list[int]]] = {}


def _qsl_pool(path: str) -> list[list[int]]:
    if path not in _QSL_CACHE:
        import pandas as pd

        frame = pd.read_pickle(path)
        if "tok_input" not in frame.columns:
            raise ValueError(
                f"{path} has no 'tok_input' column; columns={list(frame.columns)}"
            )
        pool = [list(map(int, row)) for row in frame["tok_input"] if len(row) > 0]
        if not pool:
            raise ValueError(f"{path} produced no usable token sequences")
        _QSL_CACHE[path] = pool
    return _QSL_CACHE[path]


def _qsl_tokens(index: int, *, seed: int, length: int, path: str) -> list[int]:
    """Exactly `length` real token IDs, concatenated from the QSL."""
    pool = _qsl_pool(path)
    # Deterministic per-sample start, decorrelated from `index` so consecutive
    # requests do not share a long prefix (prefix caching would otherwise turn
    # this into a cache benchmark).
    state = (seed ^ ((index + 1) * 0x9E3779B9)) & 0xFFFFFFFF
    cursor = state % len(pool)
    out: list[int] = []
    while len(out) < length:
        out.extend(pool[cursor])
        cursor = (cursor + 1) % len(pool)
    return out[:length]


def _row_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def _payload(
    token_ids: list[int],
    *,
    model: str,
    output_len: int,
    seed: int,
) -> bytes:
    return json.dumps(
        {
            "model": model,
            "prompt": token_ids,
            "max_tokens": output_len,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
        },
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass
class RequestResult:
    index: int
    ttft_ms: float | None
    tpot_ms: float | None
    e2el_ms: float
    input_tokens: int | None
    output_tokens: int
    output_token_ids_sha256: str | None
    finish_reason: str | None
    error: str | None

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and self.ttft_ms is not None
            and self.input_tokens is not None
            and self.output_tokens > 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "ok": self.ok,
            "error": self.error,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "output_token_ids_sha256": self.output_token_ids_sha256,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "e2el_ms": self.e2el_ms,
        }


def _token_ids_from_choice(choice: dict[str, Any]) -> list[int]:
    value = choice.get("token_ids")
    if value is None and isinstance(choice.get("delta"), dict):
        value = choice["delta"].get("token_ids")
    if not isinstance(value, list):
        return []
    return [
        int(token_id)
        for token_id in value
        if isinstance(token_id, int) and not isinstance(token_id, bool)
    ]


async def _one(
    session: aiohttp.ClientSession,
    *,
    url: str,
    index: int,
    body: bytes,
    expected_input_tokens: int,
    expected_output_tokens: int,
) -> RequestResult:
    t_issue = time.perf_counter()
    t_first: float | None = None
    t_last: float | None = None
    output_tokens = 0
    output_digest = hashlib.sha256()
    usage_input: int | None = None
    usage_output: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    try:
        async with session.post(
            url,
            data=body,
            headers={"content-type": "application/json"},
        ) as response:
            if response.status != 200:
                detail = (await response.text())[:1000]
                raise RuntimeError(f"HTTP {response.status}: {detail}")
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                usage = event.get("usage")
                if isinstance(usage, dict):
                    if isinstance(usage.get("prompt_tokens"), int):
                        usage_input = int(usage["prompt_tokens"])
                    if isinstance(usage.get("completion_tokens"), int):
                        usage_output = int(usage["completion_tokens"])
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                emitted = _token_ids_from_choice(choice)
                if emitted:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    output_tokens += len(emitted)
                    for token_id in emitted:
                        output_digest.update(struct.pack("<I", token_id))
    except BaseException as exc:  # preserve one bounded failure per request
        error = f"{type(exc).__name__}: {exc}"
    t_end = time.perf_counter()

    if error is None and usage_input != expected_input_tokens:
        error = (
            f"usage.prompt_tokens={usage_input}, "
            f"expected {expected_input_tokens}"
        )
    if error is None and usage_output != expected_output_tokens:
        error = (
            f"usage.completion_tokens={usage_output}, "
            f"expected {expected_output_tokens}"
        )
    if error is None and output_tokens != expected_output_tokens:
        error = (
            f"streamed token_ids={output_tokens}, "
            f"expected {expected_output_tokens}"
        )

    ttft_ms = None if t_first is None else (t_first - t_issue) * 1000.0
    tpot_ms = None
    if t_first is not None and t_last is not None:
        tpot_ms = (
            0.0
            if output_tokens <= 1
            else (t_last - t_first) * 1000.0 / (output_tokens - 1)
        )
    return RequestResult(
        index=index,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2el_ms=(t_end - t_issue) * 1000.0,
        input_tokens=usage_input,
        output_tokens=output_tokens,
        output_token_ids_sha256=(
            output_digest.hexdigest() if output_tokens else None
        ),
        finish_reason=finish_reason,
        error=error,
    )


async def _run_closed(
    *,
    url: str,
    bodies: list[bytes],
    concurrency: int,
    input_len: int,
    output_len: int,
    timeout_s: float,
) -> tuple[list[RequestResult], float]:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(len(bodies)):
        queue.put_nowait(index)

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(
        limit=0,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    results: list[RequestResult | None] = [None] * len(bodies)
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        read_bufsize=64 * 1024,
    ) as session:

        async def worker() -> None:
            while True:
                try:
                    index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results[index] = await _one(
                    session,
                    url=url,
                    index=index,
                    body=bodies[index],
                    expected_input_tokens=input_len,
                    expected_output_tokens=output_len,
                )
                queue.task_done()

        t0 = time.perf_counter()
        workers = [
            asyncio.create_task(worker())
            for _ in range(min(concurrency, len(bodies)))
        ]
        await asyncio.gather(*workers)
        wall_s = time.perf_counter() - t0
    return [result for result in results if result is not None], wall_s


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * percentile / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return (
        sorted_values[low] * (1.0 - fraction)
        + sorted_values[high] * fraction
    )


def _stats(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered) if ordered else None,
        "p50": _percentile(ordered, 50),
        "p90": _percentile(ordered, 90),
        "p99": _percentile(ordered, 99),
        "max": ordered[-1] if ordered else None,
    }


def _build_bodies(
    *,
    num_prompts: int,
    input_len: int,
    output_len: int,
    model: str,
    seed: int,
    prompt_source: str = "synthetic",
    qsl_path: str = "",
    save_prompt_token_ids: "Path | None" = None,
) -> tuple[list[bytes], str]:
    bodies: list[bytes] = []
    ordered_digest = hashlib.sha256()
    stream = None
    if save_prompt_token_ids is not None:
        save_prompt_token_ids.parent.mkdir(parents=True, exist_ok=True)
        stream = save_prompt_token_ids.open("w", encoding="utf-8")
    for index in range(num_prompts):
        if prompt_source == "qsl":
            token_ids = _qsl_tokens(
                index, seed=seed, length=input_len, path=qsl_path
            )
        else:
            token_ids = _tokens(index, seed=seed, length=input_len)
        row_digest = _row_sha256(token_ids)
        ordered_digest.update(f"{index}:{row_digest}\n".encode("ascii"))
        if stream is not None:
            stream.write(
                json.dumps(
                    {
                        "index": index,
                        "sha256": row_digest,
                        "num_tokens": len(token_ids),
                        "token_ids": token_ids,
                    }
                )
                + "\n"
            )
        bodies.append(
            _payload(
                token_ids,
                model=model,
                output_len=output_len,
                seed=seed,
            )
        )
    if stream is not None:
        stream.close()
    return bodies, ordered_digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="r1")
    parser.add_argument("--label", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--warmup-prompts", type=int, default=0)
    parser.add_argument("--warmup-concurrency", type=int, default=64)
    parser.add_argument("--warmup-output-len", type=int, default=8)
    parser.add_argument(
        "--prompt-source",
        choices=("synthetic", "qsl"),
        default="synthetic",
        help="synthetic (default, unchanged) or real MLPerf QSL prompts",
    )
    parser.add_argument(
        "--qsl-path",
        default="",
        help="mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl (with --prompt-source qsl)",
    )
    # The campaign driver passes these three.  They are accepted and
    # VALIDATED rather than ignored: silently reporting percentiles other
    # than the ones requested would be worse than refusing.
    parser.add_argument(
        "--percentile-metrics",
        default="ttft,tpot,itl,e2el",
        help="comma list; must be a subset of the metrics this client reports",
    )
    parser.add_argument(
        "--metric-percentiles",
        default="50,90,99",
        help="comma list; this client computes exactly 50,90,99",
    )
    parser.add_argument(
        "--save-prompt-token-ids",
        type=Path,
        default=None,
        help="write one JSON object per prompt (index, sha256, token_ids)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    if args.concurrency <= 0 or args.num_prompts < args.concurrency:
        raise ValueError("require num_prompts >= concurrency > 0")
    _REPORTED_METRICS = {"ttft", "tpot", "itl", "e2el"}
    _REPORTED_PERCENTILES = {"50", "90", "99"}
    wanted = {m.strip() for m in args.percentile_metrics.split(",") if m.strip()}
    if not wanted <= _REPORTED_METRICS:
        raise ValueError(
            f"--percentile-metrics {sorted(wanted - _REPORTED_METRICS)} not "
            f"reported by this client; it reports {sorted(_REPORTED_METRICS)}"
        )
    pct = {q.strip() for q in args.metric_percentiles.split(",") if q.strip()}
    if pct != _REPORTED_PERCENTILES:
        raise ValueError(
            f"--metric-percentiles {sorted(pct)} but this client computes "
            f"exactly {sorted(_REPORTED_PERCENTILES)}"
        )
    url = args.base_url.rstrip("/") + "/v1/completions"

    if args.warmup_prompts:
        warmup_bodies, _ = _build_bodies(
            num_prompts=args.warmup_prompts,
            input_len=args.input_len,
            output_len=args.warmup_output_len,
            model=args.model,
            seed=args.seed ^ 0xBAD5EED,
            prompt_source=args.prompt_source,
            qsl_path=args.qsl_path,
        )
        warmup, warmup_wall = await _run_closed(
            url=url,
            bodies=warmup_bodies,
            concurrency=min(args.warmup_concurrency, args.warmup_prompts),
            input_len=args.input_len,
            output_len=args.warmup_output_len,
            timeout_s=args.timeout_s,
        )
        warmup_failures = [result for result in warmup if not result.ok]
        if warmup_failures:
            raise RuntimeError(
                f"{len(warmup_failures)} warmup requests failed; "
                f"first={warmup_failures[0].error}"
            )
        print(
            f"[exact-token-bench] warmup={len(warmup)} "
            f"wall={warmup_wall:.3f}s",
            flush=True,
        )

    bodies, prompt_digest = _build_bodies(
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        model=args.model,
        seed=args.seed,
        prompt_source=args.prompt_source,
        qsl_path=args.qsl_path,
        save_prompt_token_ids=args.save_prompt_token_ids,
    )
    print(
        f"[exact-token-bench] prompt_source={args.prompt_source} "
        f"qsl={args.qsl_path or '-'}",
        flush=True,
    )
    print(
        f"[exact-token-bench] label={args.label} c={args.concurrency} "
        f"n={args.num_prompts} prompt_sha256={prompt_digest}",
        flush=True,
    )
    results, wall_s = await _run_closed(
        url=url,
        bodies=bodies,
        concurrency=args.concurrency,
        input_len=args.input_len,
        output_len=args.output_len,
        timeout_s=args.timeout_s,
    )
    ok = [result for result in results if result.ok]
    failed = [result for result in results if not result.ok]
    input_tokens = sum(int(result.input_tokens or 0) for result in ok)
    output_tokens = sum(result.output_tokens for result in ok)
    ordered_output_digest = hashlib.sha256()
    for result in sorted(ok, key=lambda value: value.index):
        ordered_output_digest.update(
            (
                f"{result.index}:{result.output_token_ids_sha256}\n"
            ).encode("ascii")
        )
    manifest = {
        "schema": "pf4h-exact-token-closed-concurrency-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "endpoint": url,
        "model": args.model,
        "workload": {
            "concurrency": args.concurrency,
            "num_prompts": args.num_prompts,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "seed": args.seed,
            "ignore_eos": True,
            "temperature": 0.0,
            "prompt_generator": "lcg32-v1-token-range-[1000,30999]",
            "ordered_prompt_token_id_stream_sha256": prompt_digest,
        },
        "result": {
            "completed": len(ok),
            "failed": len(failed),
            "wall_seconds": wall_s,
            "request_throughput": len(ok) / wall_s,
            "input_tokens_total": input_tokens,
            "output_tokens_total": output_tokens,
            "ordered_output_token_id_stream_sha256": (
                ordered_output_digest.hexdigest()
            ),
            "total_tokens_total": input_tokens + output_tokens,
            "input_tokens_per_second": input_tokens / wall_s,
            "output_tokens_per_second": output_tokens / wall_s,
            "total_tokens_per_second": (input_tokens + output_tokens) / wall_s,
            "ttft_ms": _stats(
                [float(result.ttft_ms) for result in ok if result.ttft_ms is not None]
            ),
            "tpot_ms": _stats(
                [float(result.tpot_ms) for result in ok if result.tpot_ms is not None]
            ),
            "e2el_ms": _stats([result.e2el_ms for result in ok]),
        },
        "per_query": [result.as_dict() for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[exact-token-bench] completed={len(ok)} failed={len(failed)} "
        f"output_tok/s={manifest['result']['output_tokens_per_second']:.3f}",
        flush=True,
    )
    print(
        f"[exact-token-bench] TTFT={manifest['result']['ttft_ms']} "
        f"TPOT={manifest['result']['tpot_ms']}",
        flush=True,
    )
    return 0 if not failed else 2


def main() -> int:
    return asyncio.run(_main_async(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

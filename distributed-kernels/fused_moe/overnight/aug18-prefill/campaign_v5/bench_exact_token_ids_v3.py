#!/usr/bin/env python3
"""Closed- OR open-loop vLLM scout with byte-identical prompt-token replay.

This is an A/B performance scout, not an official MLPerf LoadGen run.  It
records MLPerf-relevant TTFT/TPOT/ITL/E2EL distributions and an ordered digest
of the exact token-ID stream so the stock and PF4H arms can prove identical
input.

v3 (2026-08-18) = v2 plus, additively:

  * ``--request-rate`` / ``--burstiness``: an OPEN-LOOP arrival-driven driver
    (``_run_open``) beside the original closed-concurrency pool
    (``_run_closed``).  BENCHMARK_PROTOCOL.md section 2 requires arrival-driven
    cells: closed-loop c32p with OSL=8 spends ~31% of its steps in dummy
    batches, an artifact of the harness rather than of production traffic.
  * real ITL (inter-token latency) capture in ``_one``.  v2 accepted
    ``--percentile-metrics ttft,tpot,itl,e2el`` but never computed ITL, so the
    campaign was silently asking for a metric that did not exist.

BOTH additions are inert unless asked for: with neither ``--request-rate`` nor
``--burstiness`` on the command line the driver is the v2 closed-loop pool,
issuing the identical bodies through the identical ``_one`` coroutine in the
identical order.  Only the manifest grows (``itl_ms``, ``arrival``) and its
``schema`` string moves to the v2 pair.  The exact-token QSL machinery,
``_row_sha256`` folding and both ordered stream digests are untouched.

v3.1 (2026-08-18, adversarial-review fixes).  These are NOT inert -- each one
changes a number, so a manifest from v3.0 and one from v3.1 are not
interchangeable:

  * QSL prompts are drawn on a STRIDE PERMUTATION of the pool instead of an
    independent per-request hash, which removes the ~10.8% exact duplicates a
    1,024-prompt cell used to contain (and which a prefix cache would serve for
    free on whichever arm had one enabled).  ``workload.prompt_generator``
    therefore moves to ``mlperf-qsl-concat-v2-stride7`` and
    ``workload.duplicate_prompts_in_cell`` records the residue.
  * ITL no longer scores the extra tokens of the FIRST streamed chunk as
    0 ms samples.  They are dropped, and ``result.itl_dropped_first_chunk_total``
    says how many -- a figure worth comparing across arms, because it is the
    coalescing behaviour that used to contaminate the distribution.
  * ``arrival.achieved_request_rate`` counts only SUCCESSFUL requests; the
    error-inclusive figure lives beside it as
    ``returned_request_rate_including_errors``.  The old field made a cell
    where everything errored look like a cell that absorbed the load.
  * ``client_lag_ms`` is measured from the timestamp taken INSIDE the request
    coroutine, not from the spawning wrapper, which could not queue and so
    could never register lag.

On the ``schema`` string: it moved from ``...-v1`` to the ``...-v2`` pair in
v3.0 on purpose, because the manifest gained ``itl_ms`` and ``arrival``.
``analyze_campaign_v5.py`` accepts all three strings and REFUSES an unknown
one; the node's ``summarize_m15_campaign.py`` reads vllm-bench field names this
client has never emitted, so it reports zeros for v1 manifests too and is not
the consumer to preserve compatibility for.

Python 3.9 compatible (``from __future__ import annotations`` keeps the PEP 604
annotations unevaluated); stdlib only at runtime, so the arrival process uses
``random.gammavariate`` seeded from the cell seed -- which makes the arrival
schedule byte-identical across arms, the open-loop analogue of the exact-token
prompt guarantee.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
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


def _coprime_stride(modulus: int, want: int) -> int:
    """The smallest stride >= `want` that is coprime with `modulus`."""
    stride = max(1, want)
    while math.gcd(stride, modulus) != 1:
        stride += 1
    return stride


def _qsl_tokens(index: int, *, seed: int, length: int, path: str) -> list[int]:
    """Exactly `length` real token IDs, concatenated from the QSL.

    v5.1 fix (F2, fairness item 4).  The previous form picked the starting
    document as ``hash(index) % len(pool)``, i.e. independently at random per
    request.  With a 4,388-sample pool and a 1,024-prompt cell that is a
    birthday problem: about 111 prompts -- ~10.8% of the cell -- started at a
    cursor some earlier prompt had already used and were therefore BYTE-
    IDENTICAL 4,096-token repeats.  With prefix caching enabled on one arm and
    disabled on another, those ~11% of prefills cost nothing on one side and
    full price on the other: an ~11% "win" manufactured before the kernel does
    anything.

    The cursor is now ``base + index * stride (mod len(pool))`` with `stride`
    coprime to the pool size, so the map index -> starting document is a
    PERMUTATION: every prompt in a cell of up to len(pool) requests starts at a
    distinct real document and no two prompts are equal.  `base` still depends
    on the seed, so different cells still draw different (but internally
    duplicate-free) prompt sets.  The campaign additionally serves every arm
    with prefix caching off, so the two defences are independent.
    """
    pool = _qsl_pool(path)
    size = len(pool)
    base = (seed ^ 0x9E3779B9) % size
    # ~5.1 documents are consumed per 4096-token prompt; striding by more than
    # that also keeps consecutive prompts from sharing a long tail-to-head run.
    stride = _coprime_stride(size, 7)
    cursor = (base + index * stride) % size
    out: list[int] = []
    while len(out) < length:
        out.extend(pool[cursor])
        cursor = (cursor + 1) % size
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
    # v3: per-token inter-token latencies, milliseconds, one entry per output
    # token AFTER the first (so len == output_tokens - 1 on a clean request).
    itl_ms: list[float] = field(default_factory=list)
    # v5.1: tokens that arrived inside the FIRST streamed chunk, after its
    # first token.  They have no observable inter-token spacing, so they are
    # excluded from itl_ms rather than scored at 0 ms (which is what v3.0 did,
    # asymmetrically per arm, making cross-arm ITL a chunk-coalescing metric).
    itl_dropped_first_chunk: int = 0
    # v3 open loop only: seconds from the measurement epoch to this request's
    # SCHEDULED arrival and to its actual issue.  The gap between them is the
    # client's own scheduling lag -- if it grows, the client, not the server,
    # became the bottleneck and the cell is void.
    #
    # v5.1: `issued_s` is stamped by the SPAWNING wrapper (before any await) and
    # `sent_s` inside the request coroutine itself, at the moment the event loop
    # actually ran it.  client_lag_ms is measured from `sent_s`, because the
    # wrapper's timestamp is taken before anything can queue and so could never
    # grow -- v3.0's guard was structurally incapable of firing.  Connection
    # setup inside aiohttp remains folded into TTFT; the connector is unbounded
    # (limit=0) precisely so that it is not itself a queue.
    scheduled_s: float | None = None
    issued_s: float | None = None
    sent_s: float | None = None

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and self.ttft_ms is not None
            and self.input_tokens is not None
            and self.output_tokens > 0
        )

    def as_dict(self) -> dict[str, Any]:
        # The raw ITL series is deliberately NOT emitted per query: at
        # OSL 512 x 1024 prompts it would be half a million floats of
        # per-query noise.  The aggregate distribution is in result.itl_ms.
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
            "itl_ms_count": len(self.itl_ms),
            "itl_ms_mean": (
                sum(self.itl_ms) / len(self.itl_ms) if self.itl_ms else None
            ),
            "itl_dropped_first_chunk": self.itl_dropped_first_chunk,
            "scheduled_s": self.scheduled_s,
            "issued_s": self.issued_s,
            "sent_s": self.sent_s,
            "client_lag_ms": (
                None
                if self.scheduled_s is None or self.sent_s is None
                else (self.sent_s - self.scheduled_s) * 1000.0
            ),
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
    epoch: float | None = None,
) -> RequestResult:
    # `t_issue` is taken as the FIRST statement of the coroutine body, i.e. at
    # the moment the event loop actually gave this request the CPU.  In the
    # open-loop driver `epoch` is the measurement epoch, so `sent_s` records
    # that moment on the campaign clock and (sent_s - scheduled_s) is the
    # client's own event-loop scheduling lag -- measured where the request is
    # really emitted rather than in the wrapper that spawned it.
    t_issue = time.perf_counter()
    t_first: float | None = None
    t_last: float | None = None
    itl_ms: list[float] = []
    itl_dropped = 0
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
                        # The first token's latency is TTFT, never an ITL.
                        #
                        # v5.1 fix: v3.0 then computed
                        #   prev = t_last if t_last is not None else t_first
                        # which on the FIRST event evaluates to `now`, so a
                        # first chunk carrying k>1 token ids injected k-1
                        # samples of exactly 0.0 ms into the ITL pool.  Under
                        # load vLLM coalesces several ids into that first
                        # chunk, and it coalesces DIFFERENTLY per arm, so the
                        # zero contamination was asymmetric and any cross-arm
                        # ITL comparison was measuring chunk coalescing.
                        # Tokens 2..k of the first chunk have no observable
                        # spacing at all, so the honest handling is to DROP
                        # them from the series and count how many were dropped.
                        t_first = now
                        t_last = now
                        itl_dropped += max(0, len(emitted) - 1)
                        output_tokens += len(emitted)
                        for token_id in emitted:
                            output_digest.update(struct.pack("<I", token_id))
                        continue
                    # ITL, one sample per output token after the first.  A
                    # single SSE event may carry k>1 token ids (vLLM batches
                    # them under load); the event's delta is then shared
                    # evenly across its k tokens, which is the only
                    # attribution the wire supports.
                    prev = t_last if t_last is not None else t_first
                    n_new = len(emitted)
                    if n_new > 0:
                        share = (now - prev) * 1000.0 / n_new
                        itl_ms.extend([share] * n_new)
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
        itl_ms=itl_ms,
        itl_dropped_first_chunk=itl_dropped,
        sent_s=None if epoch is None else (t_issue - epoch),
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


def _arrival_offsets(
    *, count: int, request_rate: float, burstiness: float, seed: int
) -> list[float]:
    """Cumulative arrival times, seconds from the measurement epoch.

    Gamma-distributed inter-arrival gaps with shape ``burstiness`` and mean
    ``1 / request_rate``: burstiness 1.0 is exactly a Poisson process, < 1 is
    burstier (more clumping), > 1 is more uniform.  This is the same family
    ``vllm bench serve`` uses, reimplemented on stdlib ``random`` so the
    client keeps its no-numpy runtime footprint.

    The generator is seeded from the CELL seed, so every arm of every pair
    replays the identical arrival schedule against the identical prompts --
    the open-loop counterpart of the exact-token guarantee.  Without this the
    arrival jitter alone would swamp the effect being measured.
    """
    rng = random.Random(seed ^ 0x0A5E17)
    theta = 1.0 / (request_rate * burstiness)
    offsets: list[float] = []
    clock = 0.0
    for _ in range(count):
        offsets.append(clock)
        clock += rng.gammavariate(burstiness, theta)
    return offsets


async def _run_open(
    *,
    url: str,
    bodies: list[bytes],
    request_rate: float,
    burstiness: float,
    input_len: int,
    output_len: int,
    timeout_s: float,
    seed: int,
) -> "tuple[list[RequestResult], float, dict[str, Any]]":
    """Arrival-driven driver: requests are ISSUED on a schedule, not on a slot.

    The closed-loop pool answers "how fast can the server go when someone is
    always waiting"; this answers "how does the server behave at X req/s",
    which is what a deployment actually experiences and what
    BENCHMARK_PROTOCOL.md section 2 asks for.  Crucially, in-flight
    concurrency is an OUTPUT here, not an input: if the server cannot keep up
    with the offered rate, the queue grows and TTFT p99 explodes -- exactly
    the signal the closed-loop cell cannot produce.

    Every request goes through the same ``_one`` coroutine as the closed
    driver, so per-request accounting is identical.
    """
    offsets = _arrival_offsets(
        count=len(bodies),
        request_rate=request_rate,
        burstiness=burstiness,
        seed=seed,
    )
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(
        limit=0,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    results: list[RequestResult | None] = [None] * len(bodies)
    max_in_flight = 0
    in_flight = 0
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        read_bufsize=64 * 1024,
    ) as session:
        t0 = time.perf_counter()

        async def issue(index: int) -> None:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            if in_flight > max_in_flight:
                max_in_flight = in_flight
            issued = time.perf_counter() - t0
            result = await _one(
                session,
                url=url,
                index=index,
                body=bodies[index],
                expected_input_tokens=input_len,
                expected_output_tokens=output_len,
                epoch=t0,
            )
            result.scheduled_s = offsets[index]
            result.issued_s = issued
            results[index] = result
            in_flight -= 1

        tasks: list[asyncio.Task] = []
        for index in range(len(bodies)):
            delay = offsets[index] - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.ensure_future(issue(index)))
        if tasks:
            await asyncio.gather(*tasks)
        wall_s = time.perf_counter() - t0

    done = [result for result in results if result is not None]
    # v5.1 fix (fairness item: the open-loop cell's own validity check).
    # `achieved_request_rate` used to count every coroutine that RETURNED,
    # errors included.  A cell in which every request 500'd would then finish
    # fast, shrink wall_s toward the arrival span, and report an achieved rate
    # at ~100% of offered -- the analyzer's "did the server absorb the load?"
    # table showing a 0% deficit for a cell that served nothing.  Achieved is
    # now the rate of SUCCESSFUL requests; the error-inclusive figure is kept
    # beside it so the two can be compared.
    succeeded = [result for result in done if result.ok]
    lags = [
        (result.sent_s or 0.0) - (result.scheduled_s or 0.0)
        for result in done
        if result.sent_s is not None and result.scheduled_s is not None
    ]
    arrival = {
        "mode": "open",
        "request_rate": request_rate,
        "burstiness": burstiness,
        "offered_request_rate": request_rate,
        "achieved_request_rate": (
            (len(succeeded) / wall_s) if wall_s > 0 else None
        ),
        "returned_request_rate_including_errors": (
            (len(done) / wall_s) if wall_s > 0 else None
        ),
        "completed_ok": len(succeeded),
        "returned_total": len(done),
        "failed": len(done) - len(succeeded),
        "offered_input_tokens_per_second": request_rate * input_len,
        "arrival_span_seconds": offsets[-1] if offsets else 0.0,
        "max_in_flight_observed": max_in_flight,
        "client_lag_ms": _stats([lag * 1000.0 for lag in lags]),
    }
    return done, wall_s, arrival


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
) -> "tuple[list[bytes], str, int]":
    bodies: list[bytes] = []
    ordered_digest = hashlib.sha256()
    seen_rows: dict[str, int] = {}
    stream = None
    if prompt_source == "qsl":
        pool_size = len(_qsl_pool(qsl_path))
        if num_prompts > pool_size:
            # The stride permutation only guarantees distinct starting
            # documents while num_prompts <= pool size.  Beyond that the cell
            # necessarily contains repeats, which a prefix cache would serve
            # for free -- say so rather than let it pass unremarked.
            print(
                f"[exact-token-bench] WARNING: num_prompts={num_prompts} "
                f"exceeds the QSL pool ({pool_size}); the cell WILL contain "
                f"duplicate prompts. Run with prefix caching disabled on every "
                f"arm, or reduce num_prompts.",
                flush=True,
            )
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
        seen_rows[row_digest] = seen_rows.get(row_digest, 0) + 1
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
    duplicates = num_prompts - len(seen_rows)
    if duplicates:
        # Fairness item 4: an exact duplicate inside a cell is free work for
        # whichever arm has a prefix cache.  Emit the count so the number is
        # in the manifest rather than in someone's later reconstruction.
        print(
            f"[exact-token-bench] WARNING: {duplicates} of {num_prompts} "
            f"prompts are exact duplicates of an earlier prompt in this cell.",
            flush=True,
        )
    return bodies, ordered_digest.hexdigest(), duplicates


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
    # --- v3: open-loop arrival driver -------------------------------------
    # Absent  -> the v2 closed-concurrency pool, unchanged.
    # Present -> arrival-driven; --concurrency then bounds nothing and is kept
    #            only as the label/warmup width, and in-flight concurrency
    #            becomes a measured OUTPUT (result.arrival.max_in_flight).
    parser.add_argument(
        "--request-rate",
        type=float,
        default=None,
        help="requests/second; enables the OPEN-LOOP driver (default: closed)",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=None,
        help="gamma shape of the inter-arrival gaps: 1.0 = Poisson (default), "
        "<1 burstier, >1 more uniform; requires --request-rate",
    )
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
    # v3 open-loop gate.  Refuse the ambiguous combinations loudly rather than
    # silently picking a driver: a cell that ran closed-loop while its label
    # says "o75p" would be exactly the kind of quiet mislabelling this project
    # has already been burned by.
    if args.burstiness is not None and args.request_rate is None:
        raise ValueError("--burstiness requires --request-rate")
    open_loop = args.request_rate is not None
    burstiness = 1.0 if args.burstiness is None else float(args.burstiness)
    if open_loop:
        if args.request_rate <= 0:
            raise ValueError("--request-rate must be > 0")
        if burstiness <= 0:
            raise ValueError("--burstiness must be > 0")
    url = args.base_url.rstrip("/") + "/v1/completions"

    if args.warmup_prompts:
        warmup_bodies, _, _ = _build_bodies(
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

    bodies, prompt_digest, duplicate_prompts = _build_bodies(
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
    if open_loop:
        print(
            f"[exact-token-bench] driver=open rate={args.request_rate} req/s "
            f"burstiness={burstiness}",
            flush=True,
        )
        results, wall_s, arrival = await _run_open(
            url=url,
            bodies=bodies,
            request_rate=float(args.request_rate),
            burstiness=burstiness,
            input_len=args.input_len,
            output_len=args.output_len,
            timeout_s=args.timeout_s,
            seed=args.seed,
        )
    else:
        arrival = {
            "mode": "closed",
            "request_rate": None,
            "burstiness": None,
            "offered_request_rate": None,
            "achieved_request_rate": None,
            "returned_request_rate_including_errors": None,
            "completed_ok": None,
            "returned_total": None,
            "failed": None,
            "offered_input_tokens_per_second": None,
            "arrival_span_seconds": None,
            "max_in_flight_observed": args.concurrency,
            "client_lag_ms": None,
        }
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
    itl_samples: list[float] = []
    for result in ok:
        itl_samples.extend(result.itl_ms)
    manifest = {
        # v3 schema pair.  Both members carry the SAME result-field names, so
        # one extractor reads either; the string says which driver produced
        # the row and must never be edited to make an open-loop cell look
        # like a closed-loop one.
        "schema": (
            "pf4h-exact-token-open-arrival-v2"
            if open_loop
            else "pf4h-exact-token-closed-concurrency-v2"
        ),
        "client_version": "bench_exact_token_ids_v3.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "endpoint": url,
        "model": args.model,
        "workload": {
            "driver": "open" if open_loop else "closed",
            "concurrency": args.concurrency,
            "request_rate": args.request_rate,
            "burstiness": burstiness if open_loop else None,
            "num_prompts": args.num_prompts,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "seed": args.seed,
            "ignore_eos": True,
            "temperature": 0.0,
            "prompt_source": args.prompt_source,
            # v5.1: the QSL generator's cursor is now a stride permutation, so
            # a cell of up to pool-size prompts contains NO exact duplicates.
            # The generator string changes with it -- a stability receipt that
            # silently kept its old name across a behaviour change would be
            # worse than useless.
            "prompt_generator": (
                "mlperf-qsl-concat-v2-stride7"
                if args.prompt_source == "qsl"
                else "lcg32-v1-token-range-[1000,30999]"
            ),
            "duplicate_prompts_in_cell": duplicate_prompts,
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
            "itl_ms": _stats(itl_samples),
            # How many tokens were excluded from itl_ms because they shared the
            # first streamed chunk with the TTFT token.  If this differs a lot
            # between arms, the arms are coalescing differently and the ITL
            # distributions describe different sampling, not different latency.
            "itl_dropped_first_chunk_total": sum(
                result.itl_dropped_first_chunk for result in ok
            ),
            "e2el_ms": _stats([result.e2el_ms for result in ok]),
            "arrival": arrival,
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
    print(
        f"[exact-token-bench] ITL={manifest['result']['itl_ms']}",
        flush=True,
    )
    if open_loop:
        # Offered-vs-achieved is the open-loop cell's own validity check: if
        # achieved << offered the server never absorbed the load and the cell
        # measured a queue, not a throughput.
        print(
            "[exact-token-bench] arrival offered={:.4f} achieved={:.4f} req/s "
            "max_in_flight={} lag_p99_ms={}".format(
                arrival["offered_request_rate"],
                arrival["achieved_request_rate"] or 0.0,
                arrival["max_in_flight_observed"],
                (arrival["client_lag_ms"] or {}).get("p99"),
            ),
            flush=True,
        )
    return 0 if not failed else 2


def main() -> int:
    return asyncio.run(_main_async(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

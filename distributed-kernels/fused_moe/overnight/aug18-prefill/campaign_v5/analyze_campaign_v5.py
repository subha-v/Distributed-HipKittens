#!/usr/bin/env python3
"""Campaign-v5 extractor: order-balanced pair tables and the decomposition.

WHY THIS EXISTS AND NOT summarize_m15_campaign.py
-------------------------------------------------
The node summarizer reads ``output_throughput``, ``p50_ttft_ms``,
``mean_ttft_ms`` -- the key names ``vllm bench serve`` emits.  Our client has
never emitted those.  Its manifest carries
``result.output_tokens_per_second`` and ``result.ttft_ms.p50``, so every
lookup in that summarizer misses and silently returns 0, which means its
derived ``tok_s_per_gpu`` has been reporting zero (or worse, a plausible-
looking number from some older bench) for the entire campaign.  This file
reads the schema the client actually writes and refuses anything else.

WHAT IT REFUSES TO DO
---------------------
* It never divides an arm in one pair by an arm in another pair, and -- v5.1 --
  never divides an arm in one CAMPAIGN ROOT by an arm in another.  The record
  key carries the campaign; two roots' ``pair_03`` directories are different
  experiments run hours apart, and v5.0's key silently fused them while the
  table still said "within-pair".  The measured position effect is +-18% per
  arm and the day drift +-15%: a cross-pair ratio is noise wearing a decimal
  point.  Every ratio is within-pair within-campaign, and the headline is the
  MEDIAN of those per-pair ratios with the full spread printed beside it.
* It never merges an open-loop cell with a closed-loop one, and -- v5.1 -- it
  SUPPRESSES rather than annotates the ratios for an open-loop cell whose arms
  were offered different rates.  v5.0 printed a warning under the per-cell
  table and then emitted the ratio anyway in sections 3 and 4, and the JSON
  carried no marker at all.
* v5.1: it does not score open-loop cells by throughput.  The arrival schedule
  PINS input_tokens/wall to the offered rate, so an open-loop throughput ratio
  is ~1.000 by construction for any arm that keeps up and a pure saturation
  artefact for any arm that does not.  Open-loop cells are ranked by TTFT p99
  (lower is better), which is the metric that answers the question the cells
  were added for.
* It never reports a ratio against ``native_mirror`` as a production headline
  -- that arm is our own serve line on an untouched image, a control.
* v5.1: it EXCLUDES records whose arm failed its receipt gate (RECEIPT_GATE=
  FAIL, including ragged-seal coverage below threshold) and whole pairs marked
  PAIR_VOID, and it lists both.  A candidate that barely ran is not evidence,
  and v5.0's only defence here was grepping for the literal string " = 0".
* v5.1: it refuses to call anything a headline when the campaign receipt says
  the prompts were synthetic or the pair rotation was position-unbalanced.
* It labels, rather than hides, the cells the candidate loses.

Python 3.9 compatible; stdlib only (numpy is permitted but not needed --
median-of-pairs on n=5 does not want a dependency).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

# Pairs dropped because the driver marked them void (an arm failed to start,
# failed authenticity, or failed a cell).  Module-level because load_campaign
# is called once per root and the report wants them all in one place.
_VOID_PAIRS = []  # type: List[Tuple[str, str, str]]

# The manifest schemas this extractor understands.  A stranger here is a hard
# stop: silently reading an unknown schema is how the old summarizer produced
# zeros for weeks.
KNOWN_SCHEMAS = {
    "pf4h-exact-token-closed-concurrency-v1",  # v2 client
    "pf4h-exact-token-closed-concurrency-v2",  # v3 client, closed driver
    "pf4h-exact-token-open-arrival-v2",        # v3 client, open driver
}

# Arms whose ratio may be quoted as "vs production".  native_mirror is
# deliberately NOT in this set.
PRODUCTION_ARMS = ("native_default", "native_tuned_tp", "native_tuned_dp")

# Measured run-to-run noise on this node, from the campaign log: +-18% per-arm
# position effect, +-15% day drift.  A claim smaller than this is directional.
DRIFT_PCT = 15.0

# The minimum ragged-seal coverage an m15-family arm must show before its cells
# count as evidence that the megakernel served the traffic (fairness item 3).
# The driver applies the same bar; this is the reporting-layer copy, because
# v5.0's report annotated coverage without ever thresholding it.
MIN_SEAL_PCT = 90.0

# Metrics where SMALLER is better.  An open-loop cell is ranked by these: the
# arrival schedule fixes the delivered token count, so throughput cannot
# distinguish the arms, while TTFT p99 is exactly where a server that cannot
# absorb the offered load shows it.
LOWER_IS_BETTER = ("ttft_p50", "ttft_p99", "itl_p50", "itl_p99", "tpot_p50",
                   "e2el_p50")
OPEN_LOOP_METRIC = "ttft_p99"


class Record(object):
    """One (campaign, pair, arm, cell) measurement."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, name):
        return getattr(self, name, None)


def _dig(doc, *path):
    node = doc
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _read_text(path):
    try:
        with open(path, "r") as handle:
            return handle.read()
    except (IOError, OSError):
        return ""


def _seal_pct(seal_line):
    """(pct, sealed, in_bucket) parsed out of a RAGGED_SEAL_RECEIPT line."""
    if not seal_line:
        return (None, None, None)
    sealed = re.search(r"sealed[=: ]+([0-9]+)", seal_line)
    bucket = re.search(r"in_bucket[=: ]+([0-9]+)", seal_line)
    if not sealed or not bucket:
        return (None, None, None)
    sealed_n, bucket_n = int(sealed.group(1)), int(bucket.group(1))
    if bucket_n <= 0:
        return (None, sealed_n, bucket_n)
    return (100.0 * sealed_n / bucket_n, sealed_n, bucket_n)


def _receipt_verdict(receipts, seal_line, arm):
    """('PASS'|'FAIL'|'-', reason).  v5.0 looked for the literal ' = 0' and
    nothing else, so an m15 arm logging `M15_ACTIVATION_RECEIPT = 3` -- below
    the driver's own >= 8 bar -- read as OK, and the coverage receipt was
    printed truncated but never compared against anything."""
    text = receipts or ""
    if not text.strip():
        return ("-", "no receipts.txt")
    if "RECEIPT_GATE=FAIL" in text:
        return ("FAIL", "driver receipt gate failed")
    short = []
    for line in text.splitlines():
        hit = re.match(r"\s*receipt\s+(\S+)\s*=\s*([0-9]+)\s*$", line)
        if hit and int(hit.group(2)) < 8:
            short.append("%s=%s" % (hit.group(1), hit.group(2)))
    if short:
        return ("FAIL", "receipt count < 8: " + ", ".join(short))
    if arm.split("_")[0] in ("m15", "m18", "m19", "m20"):
        pct, sealed, bucket = _seal_pct(seal_line)
        if pct is None:
            return ("FAIL", "no parseable RAGGED_SEAL_RECEIPT: the megakernel "
                            "never reported coverage")
        if pct < MIN_SEAL_PCT:
            return ("FAIL", "seal coverage %.1f%% (%s/%s) < %.0f%%"
                    % (pct, sealed, bucket, MIN_SEAL_PCT))
    if "RECEIPT_GATE=PASS" in text:
        return ("PASS", "")
    return ("PASS", "")


def load_campaign(root):
    # type: (str) -> List[Record]
    """Walk <root>/pair_NN/<position>_<arm>/<cell>.json."""
    records = []  # type: List[Record]
    if not os.path.isdir(root):
        raise SystemExit("no such campaign root: %s" % root)
    campaign = os.path.basename(os.path.normpath(root))
    invocation = _read_text(os.path.join(root, "CAMPAIGN_INVOCATION.txt"))
    for pair_dir in sorted(os.listdir(root)):
        if not pair_dir.startswith("pair_"):
            continue
        void_note = _read_text(os.path.join(root, pair_dir, "PAIR_VOID.txt"))
        if void_note.strip():
            # An arm of this pair failed to start, failed authenticity, or
            # failed a cell.  Reporting the survivors would be exactly the
            # unbalanced partial-pair data the protocol forbids.
            sys.stderr.write("VOID pair %s/%s: %s\n"
                             % (campaign, pair_dir,
                                " ".join(void_note.split())))
            _VOID_PAIRS.append((campaign, pair_dir, " ".join(void_note.split())))
            continue
        try:
            pair = int(pair_dir.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        pair_path = os.path.join(root, pair_dir)
        if not os.path.isdir(pair_path):
            continue
        for arm_dir in sorted(os.listdir(pair_path)):
            arm_path = os.path.join(pair_path, arm_dir)
            if not os.path.isdir(arm_path):
                continue
            head, _, arm = arm_dir.partition("_")
            if not arm:
                continue
            try:
                position = int(head)
            except ValueError:
                continue
            # Arm-level receipts, read once and attached to every cell of the
            # arm: they are properties of the SERVER, not of the cell.
            receipts = _read_text(os.path.join(arm_path, "receipts.txt"))
            markers = _read_text(os.path.join(arm_path, "patch_markers.txt"))
            coverage = _read_text(os.path.join(arm_path, "coverage_tail.txt"))
            nclass = _read_text(
                os.path.join(arm_path, "numerics_class.txt")).strip()
            server_log = os.path.join(arm_path, "server.log")
            seal = ""
            if os.path.isfile(server_log):
                # Coverage receipt: proof the megakernel actually executed the
                # traffic (fairness item 3).  Cheap tail scan, not a full read.
                for line in _read_text(server_log).splitlines():
                    if "RAGGED_SEAL_RECEIPT" in line:
                        seal = line.strip()
            if not seal:
                for line in (receipts + "\n" + coverage).splitlines():
                    if "RAGGED_SEAL_RECEIPT" in line:
                        seal = line.strip()
            receipt_verdict, receipt_reason = _receipt_verdict(
                receipts, seal, arm)
            for name in sorted(os.listdir(arm_path)):
                if not name.endswith(".json"):
                    continue
                cell = name[:-5]
                if cell.endswith("_prewarm") or cell in (
                    "image_digest",
                    "docker_env",
                    "docker_cmd",
                ):
                    continue
                path = os.path.join(arm_path, name)
                try:
                    with open(path, "r") as handle:
                        doc = json.load(handle)
                except (ValueError, IOError, OSError) as exc:
                    sys.stderr.write("WARN unreadable %s: %s\n" % (path, exc))
                    continue
                if not isinstance(doc, dict) or "result" not in doc:
                    continue
                record = _record_from_doc(
                    doc,
                    campaign=campaign,
                    invocation=invocation,
                    pair=pair,
                    position=position,
                    arm=arm,
                    cell=cell,
                    path=path,
                    receipts=receipts,
                    markers=markers,
                    coverage=coverage,
                    seal=seal,
                    seal_pct=_seal_pct(seal)[0],
                    numerics_class=nclass,
                    receipt_verdict=receipt_verdict,
                    receipt_reason=receipt_reason,
                )
                records.append(record)
    return records


def _record_from_doc(doc, **meta):
    schema = doc.get("schema")
    if schema not in KNOWN_SCHEMAS:
        raise SystemExit(
            "unknown manifest schema %r in %s\n"
            "  known: %s\n"
            "  refusing to guess field names -- that is exactly how the node "
            "summarizer ended up reporting zeros." % (
                schema, meta["path"], ", ".join(sorted(KNOWN_SCHEMAS))
            )
        )
    result = doc["result"]
    workload = doc.get("workload", {})
    arrival = result.get("arrival") or {}
    return Record(
        schema=schema,
        driver=workload.get("driver") or (
            "open" if schema.endswith("open-arrival-v2") else "closed"
        ),
        concurrency=workload.get("concurrency"),
        num_prompts=workload.get("num_prompts"),
        input_len=workload.get("input_len"),
        output_len=workload.get("output_len"),
        seed=workload.get("seed"),
        request_rate=workload.get("request_rate"),
        burstiness=workload.get("burstiness"),
        prompt_source=workload.get("prompt_source"),
        prompt_generator=workload.get("prompt_generator"),
        duplicate_prompts=workload.get("duplicate_prompts_in_cell"),
        prompt_sha=workload.get("ordered_prompt_token_id_stream_sha256"),
        output_sha=result.get("ordered_output_token_id_stream_sha256"),
        completed=result.get("completed"),
        failed=result.get("failed"),
        wall_seconds=result.get("wall_seconds"),
        request_throughput=result.get("request_throughput"),
        input_tps=result.get("input_tokens_per_second"),
        output_tps=result.get("output_tokens_per_second"),
        total_tps=result.get("total_tokens_per_second"),
        ttft_p50=_dig(result, "ttft_ms", "p50"),
        ttft_p99=_dig(result, "ttft_ms", "p99"),
        tpot_p50=_dig(result, "tpot_ms", "p50"),
        itl_p50=_dig(result, "itl_ms", "p50"),
        itl_p99=_dig(result, "itl_ms", "p99"),
        e2el_p50=_dig(result, "e2el_ms", "p50"),
        offered_rate=arrival.get("offered_request_rate"),
        achieved_rate=arrival.get("achieved_request_rate"),
        max_in_flight=arrival.get("max_in_flight_observed"),
        **meta
    )


# --------------------------------------------------------------------------- #
# aggregation                                                                 #
# --------------------------------------------------------------------------- #

def index_records(records):
    # type: (List[Record]) -> Dict[Tuple[str, str, str, int], Record]
    """(campaign, cell, arm, pair) -> record.

    v5.1: the CAMPAIGN is part of the key.  Without it, two roots' ``pair_03``
    directories collapsed onto one slot: either an arm from root A was divided
    by an arm from root B -- runs hours apart, possibly at different offered
    rates -- and printed as a "within-pair" ratio, or, when the same arm
    appeared in both, one of the two was dropped to stderr and vanished from
    the report and from its own n count.

    A duplicate within one campaign is still a campaign bug, not a merge
    opportunity: the same arm cannot legitimately run the same cell twice in
    one pair, and averaging them would hide whichever one crashed.  Those are
    now surfaced in the report, not only on stderr.
    """
    index = {}
    duplicates = []
    for record in records:
        key = (record.campaign, record.cell, record.arm, record.pair)
        if key in index:
            message = ("duplicate %s %s/%s/pair%s -- keeping %s, IGNORING %s"
                       % (record.campaign, record.cell, record.arm,
                          record.pair, index[key].path, record.path))
            sys.stderr.write("WARN " + message + "\n")
            duplicates.append(message)
            continue
        index[key] = record
    return index, duplicates


def campaign_pairs(index, cell):
    """[(campaign, pair)] present for this cell, sorted."""
    return sorted({(key[0], key[3]) for key in index if key[1] == cell})


def median_and_spread(values):
    """(median, min, max, spread_pct_of_median) over a list of floats."""
    clean = [float(v) for v in values if v is not None and not _isnan(v)]
    if not clean:
        return (None, None, None, None)
    med = statistics.median(clean)
    lo = min(clean)
    hi = max(clean)
    spread = None if med in (0, None) else (hi - lo) / med * 100.0
    return (med, lo, hi, spread)


def _isnan(value):
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def cell_metric(record, default_metric):
    """The metric an OPEN-LOOP cell must be judged by.

    In an open-loop cell the client delivers a fixed prompt list on a fixed
    arrival schedule, so ``input_tokens / wall_seconds`` is pinned to the
    offered rate for any arm that keeps up: v5.0's decomposition reported
    ~1.000 for every open cell no matter how differently the arms served it,
    and its policy table then picked a "best arm" out of drain-time noise.
    TTFT p99 is where an arm that cannot absorb the load actually shows it, so
    that is what the o-cells are ranked by (smaller is better -- see
    ``ratio_is_gain``).
    """
    if getattr(record, "driver", "closed") == "open":
        return OPEN_LOOP_METRIC
    return default_metric


def ratio_is_gain(metric):
    """True when ratio > 1 means the numerator arm is BETTER."""
    return metric not in LOWER_IS_BETTER


def pair_ratios(index, cell, numerator, denominator, metric="input_tps",
                excluded=None):
    """Within-pair AND within-campaign ratios only.

    Returns ``[((campaign, pair), ratio)]``.  Records whose arm failed its
    receipt gate are excluded (``excluded`` collects the reasons for the
    report) -- an arm that did not demonstrably run the kernel under claim is
    not a denominator or a numerator.
    """
    out = []
    for campaign, pair in campaign_pairs(index, cell):
        top = index.get((campaign, cell, numerator, pair))
        bot = index.get((campaign, cell, denominator, pair))
        if top is None or bot is None:
            continue
        skip = False
        for record in (top, bot):
            if getattr(record, "receipt_verdict", "-") == "FAIL":
                if excluded is not None:
                    excluded.append(
                        "%s %s pair %s arm %s: %s"
                        % (campaign, cell, pair, record.arm,
                           getattr(record, "receipt_reason", "receipt gate")))
                skip = True
        if skip:
            continue
        tv = top.get(metric)
        bv = bot.get(metric)
        if tv is None or bv is None or not bv:
            continue
        out.append(((campaign, pair), float(tv) / float(bv)))
    return out


# --------------------------------------------------------------------------- #
# rendering                                                                   #
# --------------------------------------------------------------------------- #

def _fmt(value, digits=1):
    if value is None:
        return "-"
    try:
        return ("%%.%df" % digits) % float(value)
    except (TypeError, ValueError):
        return str(value)


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _campaign_flags(records):
    """Protocol-level facts read off the campaign receipts, not off intent.

    v5.1: two of the review's blocking findings are things a reader of the
    TABLE could not have detected -- a night run on synthetic prompts, and a
    pair count that is not a multiple of the arm count so the rotation never
    balanced position.  Both are recorded by the driver in
    CAMPAIGN_INVOCATION.txt and both are surfaced here as banners that
    explicitly disqualify the run as a headline.
    """
    flags = []
    sources = {r.prompt_source for r in records if r.prompt_source}
    if sources and sources != {"qsl"}:
        flags.append(
            "**NOT A HEADLINE: prompt_source = %s.** BENCHMARK_PROTOCOL.md "
            "section 2 requires real MLPerf QSL text; uniform-random token ids "
            "give expert routing no popularity skew, which is the variable the "
            "MoE claim rests on."
            % ", ".join(sorted(sources)))
    generators = {r.prompt_generator for r in records if r.prompt_generator}
    if len(generators) > 1:
        flags.append(
            "**Prompt generators differ across records (%s).** These runs did "
            "not draw the same prompt distribution and must not be pooled."
            % ", ".join(sorted(generators)))
    dupes = max([int(r.duplicate_prompts or 0) for r in records] or [0])
    if dupes:
        flags.append(
            "**%d exact duplicate prompts inside a cell.** With prefix caching "
            "enabled on any arm those prefills are free for that arm only "
            "(fairness item 4)." % dupes)
    for invocation in {r.invocation for r in records if r.invocation}:
        if "position_balance: UNBALANCED" in invocation:
            flags.append(
                "**NOT A HEADLINE: the driver recorded "
                "POSITION_BALANCE=UNBALANCED** (PAIRS is not a multiple of the "
                "arm count), so some arm held position 1 more often than "
                "another and the measured +-18%/arm position effect is folded "
                "into every ratio below rather than cancelled by it.")
        if "prompt_source: synthetic" in invocation:
            flags.append(
                "**The campaign receipt says prompt_source: synthetic.**")
    return sorted(set(flags))


def _position_balance(index, records):
    """(rows, unbalanced_cells).  One row per (campaign, cell, arm): how often
    the arm ran in each position.  v5.0 had no such check; its balancing was a
    list reversal that, with three arms, pinned the middle arm to position 2 in
    every pair and gave the first-listed arm the extra position-1 slot when
    PAIRS was odd.  A +4-5% artefact is the size of the entire banked claim, so
    the imbalance is now visible in the report itself."""
    counts = {}
    for record in records:
        key = (record.campaign, record.cell, record.arm)
        counts.setdefault(key, {}).setdefault(record.position, 0)
        counts[key][record.position] += 1
    rows = []
    bad = set()
    for (campaign, cell, arm) in sorted(counts):
        seen = counts[(campaign, cell, arm)]
        spread = max(seen.values()) - min(seen.values()) if seen else 0
        # An arm that never occupies some position at all is the worst case.
        positions = sorted(seen)
        balanced = spread == 0
        rows.append([campaign, cell, arm,
                     ", ".join("p%d x%d" % (p, seen[p]) for p in positions),
                     "OK" if balanced else "**IMBALANCED**"])
        if not balanced:
            bad.add((campaign, cell))
    return rows, bad


def _open_loop_comparable(index, cell, arms):
    """(comparable, rates).  Two arms offered different request rates are not
    running the same workload; v5.0 printed a warning under the per-cell table
    and then emitted the ratio anyway in sections 3 and 4, with no marker in
    the JSON at all.  Here the answer is consumed, not just printed."""
    rates = set()
    for campaign, pair in campaign_pairs(index, cell):
        for arm in arms:
            record = index.get((campaign, cell, arm, pair))
            if record is not None and record.offered_rate is not None:
                rates.add(round(float(record.offered_rate), 4))
    return (len(rates) <= 1, sorted(rates))


def render(records, args):
    index, duplicates = index_records(records)
    cells = sorted({record.cell for record in records})
    arms = sorted({record.arm for record in records})
    campaigns_pairs = sorted({(r.campaign, r.pair) for r in records})
    gpus = args.gpus

    report = {
        "campaigns": args.campaign,
        "gpus": gpus,
        "arms": arms,
        "cells": cells,
        "pairs": [list(cp) for cp in campaigns_pairs],
        "metric": args.metric,
        "open_loop_metric": OPEN_LOOP_METRIC,
        "integrity": {},
        "per_cell": {},
        "decomposition": {},
        "policy": [],
        "void_pairs": [list(v) for v in _VOID_PAIRS],
        "duplicate_records": duplicates,
        "disqualifiers": [],
        "excluded_by_receipt_gate": [],
    }
    out = []
    out.append("# Campaign v5 results")
    out.append("")
    out.append(
        "Campaigns: `%s`  ·  arms: %s  ·  cells: %s  ·  (campaign, pair) "
        "slots: %d  ·  GPUs: %d"
        % (", ".join(args.campaign), ", ".join(arms), ", ".join(cells),
           len(campaigns_pairs), gpus)
    )
    out.append("")
    out.append(
        "Headline metric: `%s` for closed-loop cells and `%s` (lower is "
        "better) for open-loop cells -- an open-loop cell's throughput is "
        "pinned to the offered arrival rate by construction and cannot "
        "distinguish the arms. Every ratio below is computed WITHIN one pair of "
        "ONE campaign root and then reduced by median across those pairs; the "
        "spread column is (max-min)/median over the per-pair ratios. Measured "
        "node noise is +-%.0f%% day drift and +-18%% arm-position effect, so a "
        "median ratio whose spread straddles 1.0 is directional, not a result."
        % (args.metric, OPEN_LOOP_METRIC, DRIFT_PCT)
    )
    out.append("")

    # ------------------------------------------------------ disqualifiers
    flags = _campaign_flags(records)
    if flags:
        out.append("## 0. Disqualifiers and protocol flags")
        out.append("")
        for flag in flags:
            out.append("> " + flag)
            out.append("")
        report["disqualifiers"] = flags

    # ---------------------------------------------------------------- integrity
    out.append("## 1. Integrity")
    out.append("")
    if _VOID_PAIRS:
        out.append("### 1.0 Void pairs (dropped, not reported)")
        out.append("")
        out.append(_table(["campaign", "pair", "reason"],
                          [list(v) for v in _VOID_PAIRS]))
        out.append("")
        out.append("> A pair in which any arm failed to start, failed the "
                   "untouched-image authenticity check, or failed a cell is "
                   "dropped WHOLE. Reporting its surviving arms would be the "
                   "unbalanced partial-pair data the protocol forbids.")
        out.append("")
    if duplicates:
        out.append("### 1.0b Duplicate records (ignored)")
        out.append("")
        for message in duplicates:
            out.append("* `%s`" % message)
        out.append("")

    integrity_rows = []
    for cell in cells:
        cell_arms = sorted({r.arm for r in records if r.cell == cell})
        for campaign, pair in campaign_pairs(index, cell):
            shas = {}
            present = []
            for arm in cell_arms:
                record = index.get((campaign, cell, arm, pair))
                if record is not None:
                    present.append(arm)
                    shas.setdefault(record.prompt_sha, []).append(arm)
            if not shas:
                continue
            # v5.1: a pair with ONE arm in it is not an agreement receipt.
            # v5.0 printed `OK` for it, visually indistinguishable from a
            # genuine two-arm match, and `prompt_sha=None` on every arm also
            # collapsed to len(shas)==1 -> OK.
            if list(shas) == [None]:
                ok = "**NO SHA**"
            elif len(present) < 2:
                ok = "**INCOMPLETE (%d arm)**" % len(present)
            elif len(present) < len(cell_arms):
                ok = "**INCOMPLETE (%d of %d arms)**" % (len(present),
                                                         len(cell_arms))
            elif len(shas) == 1:
                ok = "OK"
            else:
                ok = "**MISMATCH**"
            integrity_rows.append([
                campaign, cell, pair, ok, len(shas), ", ".join(sorted(present)),
            ])
    out.append("### 1.1 Identical token streams (fairness item 4)")
    out.append("")
    out.append(_table(
        ["campaign", "cell", "pair", "prompt SHA agreement", "distinct SHAs",
         "arms present"],
        integrity_rows,
    ))
    report["integrity"]["prompt_sha_rows"] = integrity_rows
    out.append("")

    out.append("### 1.2 Baseline authenticity and candidate coverage "
               "(fairness items 1 and 3)")
    out.append("")
    auth_rows = []
    for arm in arms:
        seen = [r for r in records if r.arm == arm]
        markers = seen[0].markers if seen else ""
        seal = ""
        seal_pct = None
        for record in seen:
            if record.seal:
                seal = record.seal
                seal_pct = record.seal_pct
                break
        if arm.startswith("native"):
            verdict = "PASS" if "NATIVE_AUTHENTICITY=PASS" in markers else (
                "**FAIL/absent**")
        else:
            verdict = "n/a (our arm)"
        gate = {r.receipt_verdict for r in seen}
        if "FAIL" in gate:
            reason = next((r.receipt_reason for r in seen
                           if r.receipt_verdict == "FAIL"), "")
            recv = "**VOID: %s**" % reason
        elif gate == {"-"}:
            recv = "-"
        else:
            recv = "OK"
        coverage = "-"
        if seal_pct is not None:
            coverage = "%.1f%% %s" % (
                seal_pct,
                "" if seal_pct >= MIN_SEAL_PCT else "**< %.0f%%**" % MIN_SEAL_PCT)
        elif seal:
            coverage = "**unparsed**: " + (
                seal[:48] + "..." if len(seal) > 48 else seal)
        auth_rows.append([
            arm,
            "production baseline" if arm in PRODUCTION_ARMS else (
                "control (NOT production)" if arm.startswith("native")
                else "our arm"),
            verdict, recv, coverage,
        ])
    out.append(_table(
        ["arm", "role", "untouched-image check", "receipt gate",
         "ragged-seal coverage"],
        auth_rows,
    ))
    report["integrity"]["authenticity_rows"] = auth_rows
    out.append("")
    out.append(
        "> `native_mirror` is our serve line on an untouched image. It is a "
        "control for the integration effect and must never be quoted as "
        "production."
    )
    out.append("")

    # ------------------------------------------------- 1.3 position balance
    balance_rows, unbalanced = _position_balance(index, records)
    out.append("### 1.3 Position balance (fairness item 5)")
    out.append("")
    out.append(_table(
        ["campaign", "cell", "arm", "positions held", "balanced"],
        balance_rows))
    out.append("")
    if unbalanced:
        out.append(
            "> **The arm positions are not balanced in: %s.** The measured "
            "position effect is +-18%% per arm (position 2 ran ~25%% slower "
            "than position 1), so an arm holding position 1 more often than "
            "its comparator carries that bias into every per-pair ratio and "
            "the median does not cancel it. Ratios for those cells are "
            "directional at best."
            % ", ".join("%s/%s" % (c, cell) for c, cell in sorted(unbalanced)))
        out.append("")
    report["integrity"]["position_balance_rows"] = balance_rows
    report["integrity"]["position_unbalanced"] = [
        list(item) for item in sorted(unbalanced)]

    # --------------------------------------------------------------- per cell
    out.append("## 2. Per-cell absolutes (median of pairs)")
    out.append("")
    closed_cells = [c for c in cells if _cell_driver(index, c) == "closed"]
    open_cells = [c for c in cells if _cell_driver(index, c) == "open"]
    comparable = {}
    for cell in closed_cells + open_cells:
        driver = _cell_driver(index, cell)
        spec = _cell_spec(index, cell)
        metric = OPEN_LOOP_METRIC if driver == "open" else args.metric
        out.append("### %s — %s" % (cell, spec))
        out.append("")
        out.append("Ranking metric for this cell: `%s`%s"
                   % (metric, " (lower is better)"
                      if metric in LOWER_IS_BETTER else ""))
        out.append("")
        rows = []
        cell_json = {}
        for arm in arms:
            got = [index[(c, cell, arm, p)]
                   for (c, p) in campaign_pairs(index, cell)
                   if (c, cell, arm, p) in index]
            if not got:
                continue
            med_in, lo_in, hi_in, spr_in = median_and_spread(
                [r.get(metric) for r in got])
            med_tps = median_and_spread([r.input_tps for r in got])[0]
            med_out = median_and_spread([r.output_tps for r in got])[0]
            med_ttft50 = median_and_spread([r.ttft_p50 for r in got])[0]
            med_ttft99 = median_and_spread([r.ttft_p99 for r in got])[0]
            med_itl50 = median_and_spread([r.itl_p50 for r in got])[0]
            med_tpot = median_and_spread([r.tpot_p50 for r in got])[0]
            med_wall = median_and_spread([r.wall_seconds for r in got])[0]
            failed = sum(int(r.failed or 0) for r in got)
            void = sum(1 for r in got if r.receipt_verdict == "FAIL")
            row = [
                arm, len(got),
                _fmt(med_in, 1 if metric in LOWER_IS_BETTER else 0),
                "%s / %s" % (_fmt(lo_in, 1), _fmt(hi_in, 1)),
                _fmt(med_tps, 0),
                _fmt(None if med_tps is None else med_tps / gpus, 1),
                _fmt(med_out, 0),
                _fmt(med_ttft50, 0),
                _fmt(med_ttft99, 0),
                _fmt(med_itl50, 2),
                _fmt(med_tpot, 2),
                _fmt(med_wall, 1),
                failed if failed else "-",
                "**%d VOID**" % void if void else "-",
            ]
            rows.append(row)
            cell_json[arm] = {
                "n_pairs": len(got),
                "metric": metric,
                "median": med_in,
                "min": lo_in,
                "max": hi_in,
                "spread_pct": spr_in,
                "input_tokens_per_second_median": med_tps,
                "median_per_gpu": None if med_tps is None else med_tps / gpus,
                "output_tokens_per_second_median": med_out,
                "ttft_ms_p50_median": med_ttft50,
                "ttft_ms_p99_median": med_ttft99,
                "itl_ms_p50_median": med_itl50,
                "tpot_ms_p50_median": med_tpot,
                "wall_seconds_median": med_wall,
                "failed_total": failed,
                "receipt_void_records": void,
                "driver": driver,
                "per_pair": {
                    "%s/pair%s" % (r.campaign, r.pair): {
                        "position": r.position,
                        metric: r.get(metric),
                        "input_tps": r.input_tps,
                        "ttft_ms_p50": r.ttft_p50,
                        "ttft_ms_p99": r.ttft_p99,
                        "receipt_verdict": r.receipt_verdict,
                        "output_sha": r.output_sha,
                    } for r in got
                },
            }
        out.append(_table(
            ["arm", "n", "%s (median)" % metric, "min / max",
             "input tok/s", "tok/s/GPU", "out tok/s", "TTFT p50 ms",
             "TTFT p99 ms", "ITL p50 ms", "TPOT p50 ms", "wall s", "failed",
             "receipt"],
            rows,
        ))
        out.append("")
        cell_comparable = True
        rates = []
        if driver == "open":
            orows = []
            for arm in arms:
                got = [index[(c, cell, arm, p)]
                       for (c, p) in campaign_pairs(index, cell)
                       if (c, cell, arm, p) in index]
                if not got:
                    continue
                offered = median_and_spread([r.offered_rate for r in got])[0]
                achieved = median_and_spread([r.achieved_rate for r in got])[0]
                inflight = median_and_spread([r.max_in_flight for r in got])[0]
                failed = sum(int(r.failed or 0) for r in got)
                deficit = None
                if offered and achieved:
                    deficit = (achieved / offered - 1.0) * 100.0
                orows.append([arm, _fmt(offered, 3), _fmt(achieved, 3),
                              _fmt(deficit, 1), _fmt(inflight, 0),
                              failed if failed else "-"])
            out.append("Open-loop validity — `achieved req/s` counts only "
                       "SUCCESSFUL requests (v3.1; the earlier field counted "
                       "errored ones too, so a cell where everything failed "
                       "reported a healthy achieved rate). An arm materially "
                       "below its offered rate did not absorb the load and its "
                       "latencies are a queue, not a service time:")
            out.append("")
            out.append(_table(
                ["arm", "offered req/s", "achieved req/s", "deficit %",
                 "max in-flight", "failed"], orows))
            out.append("")
            cell_comparable, rates = _open_loop_comparable(index, cell, arms)
            if not cell_comparable:
                out.append("> **INVALID CELL: arms were offered DIFFERENT "
                           "rates (%s). They did not run the same workload, so "
                           "no ratio is reported for this cell in sections 3 "
                           "or 4.**"
                           % ", ".join(_fmt(x, 3) for x in rates))
                out.append("")
        comparable[cell] = cell_comparable
        report["per_cell"][cell] = {"driver": driver, "spec": spec,
                                    "metric": metric,
                                    "comparable": cell_comparable,
                                    "offered_rates": rates,
                                    "arms": cell_json}

    # ---------------------------------------------------- decomposition
    excluded = []
    out.append("## 3. Three-way decomposition (within-pair, median of pairs)")
    out.append("")
    out.append(
        "`%s / <native>` is the headline; `%s / %s` isolates the kernel "
        "substitution; `%s / <native>` is the integration effect, i.e. the "
        "part of any win that both sides could adopt. A headline is only "
        "honest when all three are shown together. For open-loop cells the "
        "ratio is `%s` and is INVERTED so that >1 still means the numerator "
        "arm is better."
        % (args.candidate, args.candidate, args.control, args.control,
           OPEN_LOOP_METRIC)
    )
    out.append("")
    for cell in cells:
        metric = OPEN_LOOP_METRIC if _cell_driver(index, cell) == "open" \
            else args.metric
        if not comparable.get(cell, True):
            out.append("### %s" % cell)
            out.append("")
            out.append("> **Suppressed: the arms in this cell were offered "
                       "different request rates and are not comparable.**")
            out.append("")
            report["decomposition"][cell] = {"suppressed":
                                             "offered rates differ"}
            continue
        rows = []
        for base in [a for a in arms if a.startswith("native")]:
            head = pair_ratios(index, cell, args.candidate, base, metric,
                               excluded)
            integ = pair_ratios(index, cell, args.control, base, metric,
                                excluded)
            kern = pair_ratios(index, cell, args.candidate, args.control,
                               metric, excluded)
            if not head and not integ and not kern:
                continue
            gain = ratio_is_gain(metric)
            rows.append([
                base + ("" if base in PRODUCTION_ARMS else " (control)"),
                _ratio_cell(head, gain), _ratio_cell(kern, gain),
                _ratio_cell(integ, gain),
                len(head),
            ])
            report["decomposition"].setdefault(cell, {})[base] = {
                "metric": metric,
                "orientation": "higher_is_better" if gain
                else "lower_is_better (ratios inverted for display)",
                "position_balanced": not any(
                    cell == c for _, c in unbalanced),
                "headline_candidate_over_native": _ratio_json(head, gain),
                "kernel_candidate_over_control": _ratio_json(kern, gain),
                "integration_control_over_native": _ratio_json(integ, gain),
            }
        if not rows:
            continue
        out.append("### %s" % cell)
        out.append("")
        out.append(_table(
            ["baseline", "%s / base (headline)" % args.candidate,
             "%s / %s (kernel)" % (args.candidate, args.control),
             "%s / base (integration)" % args.control, "pairs"],
            rows,
        ))
        out.append("")

    # The accuracy gate is read BEFORE the policy table, not after it.  v5.0
    # printed section 4's "SHIP CANDIDATE" and then pasted a FAILED gate into
    # section 5 underneath it; the two never spoke to each other.
    gate_text = ""
    for root in args.campaign:
        gate_text += _read_text(
            os.path.join(root, "accuracy", "ACCURACY_GATE.txt"))
    if "ACCURACY_GATE=FAIL" in gate_text:
        gate_status = "FAIL"
    elif "ACCURACY_GATE=PASS_WEAK" in gate_text:
        gate_status = "PASS_WEAK"
    elif "ACCURACY_GATE=PASS" in gate_text:
        gate_status = "PASS"
    else:
        gate_status = "ABSENT"
    report["accuracy_gate_status"] = gate_status

    # ---------------------------------------------------------- policy row
    out.append("## 4. Per-cell winner and deployment policy")
    out.append("")
    out.append(
        "The deliverable a reader can act on: which arm one would actually "
        "ship in each cell. Cells the candidate loses are listed, not hidden "
        "-- the megakernel is expected to invert below ~1,600-1,800 "
        "tokens/rank."
    )
    out.append("")
    policy_rows = []
    for cell in cells:
        driver = _cell_driver(index, cell)
        metric = OPEN_LOOP_METRIC if driver == "open" else args.metric
        gain = ratio_is_gain(metric)
        if not comparable.get(cell, True):
            policy_rows.append([
                cell, _cell_spec(index, cell), "-", "-", "-", "-", "-", "-",
                "**invalid: offered rates differ**", "no policy"])
            continue
        best_arm, best_val = None, None
        prod_arm, prod_val = None, None
        for arm in arms:
            got = [index[(c, cell, arm, p)]
                   for (c, p) in campaign_pairs(index, cell)
                   if (c, cell, arm, p) in index
                   and index[(c, cell, arm, p)].receipt_verdict != "FAIL"]
            if not got:
                continue
            med = median_and_spread([r.get(metric) for r in got])[0]
            if med is None:
                continue
            if best_val is None or ((med > best_val) if gain
                                    else (med < best_val)):
                best_arm, best_val = arm, med
            if arm in PRODUCTION_ARMS:
                if prod_val is None or ((med > prod_val) if gain
                                        else (med < prod_val)):
                    prod_arm, prod_val = arm, med
        if best_val is None:
            continue
        margin = None
        if prod_val:
            margin = ((best_val / prod_val - 1.0) if gain
                      else (prod_val / best_val - 1.0)) * 100.0
        ratios = pair_ratios(index, cell, args.candidate, prod_arm or "",
                             metric, excluded) if prod_arm else []
        values = [r for _, r in ratios]
        if not gain:
            values = [1.0 / v for v in values if v]
        med_r, lo_r, hi_r, _ = median_and_spread(values)
        verdict = "-"
        if med_r is not None:
            if lo_r is not None and lo_r > 1.0:
                verdict = "candidate wins every pair"
            elif hi_r is not None and hi_r < 1.0:
                verdict = "candidate loses every pair"
            elif margin is not None and abs(margin) < DRIFT_PCT:
                verdict = "inside measured drift — directional only"
            else:
                verdict = "mixed across pairs"
        n_pairs = len(ratios)
        # v5.1: a "SHIP" verdict now also requires enough pairs, a balanced
        # rotation and a clean protocol receipt.  v5.0 could stamp SHIP
        # CANDIDATE next to a FAILED accuracy gate, an unbalanced 5-pair
        # rotation and a synthetic prompt stream.
        blockers = []
        if n_pairs < 5:
            blockers.append("n=%d < 5 pairs" % n_pairs)
        if any(cell == c for _, c in unbalanced):
            blockers.append("position-unbalanced")
        if flags:
            blockers.append("protocol flags in section 0")
        if gate_status in ("FAIL", "ABSENT"):
            blockers.append("accuracy gate %s" % gate_status.lower())
        if blockers:
            policy = "no claim (%s)" % "; ".join(blockers)
        elif best_arm == args.candidate and verdict == "candidate wins every pair":
            policy = "SHIP CANDIDATE"
        else:
            policy = "ship %s" % best_arm
        policy_rows.append([
            cell, _cell_spec(index, cell), best_arm,
            _fmt(best_val, 1 if not gain else 0),
            prod_arm or "-", _fmt(prod_val, 1 if not gain else 0),
            "-" if margin is None else "%+.1f%%" % margin,
            "-" if med_r is None else "%s (%s–%s, n=%d)" % (
                _fmt(med_r, 3), _fmt(lo_r, 3), _fmt(hi_r, 3), n_pairs),
            verdict,
            policy,
        ])
    out.append(_table(
        ["cell", "spec", "best arm", "best metric",
         "best production arm", "its metric", "margin",
         "%s/prod median (min–max)" % args.candidate, "verdict", "policy"],
        policy_rows,
    ))
    report["policy"] = policy_rows
    out.append("")
    if excluded:
        seen = sorted(set(excluded))
        out.append("### 4.1 Records excluded by the receipt gate")
        out.append("")
        for item in seen:
            out.append("* `%s`" % item)
        out.append("")
        report["excluded_by_receipt_gate"] = seen

    # ------------------------------------------------------------ accuracy
    out.append("## 5. Accuracy gate")
    out.append("")
    gate_lines = []
    gate_ok = None
    for root in args.campaign:
        path = os.path.join(root, "accuracy", "ACCURACY_GATE.txt")
        text = _read_text(path)
        if text.strip():
            gate_lines.append("```\n%s\n```" % text.strip())
            if "ACCURACY_GATE=FAIL" in text:
                gate_ok = False
            elif gate_ok is None:
                gate_ok = True
    if gate_lines:
        out.extend(gate_lines)
        report["accuracy_gate"] = "\n".join(gate_lines)
        if gate_ok is False:
            out.append("")
            out.append("> **The accuracy gate FAILED. Per "
                       "BENCHMARK_PROTOCOL.md item 3.4 no performance claim "
                       "ships from this campaign until parity is resolved, "
                       "whatever section 4 says.**")
        if any("PASS_WEAK" in line for line in gate_lines):
            out.append("")
            out.append("> **The gate passed WEAKLY: no arm shared the "
                       "candidate's numerics class, so bit-exactness was never "
                       "established for it. Quote the cross-class agreement "
                       "fractions, not a parity claim.**")
    else:
        out.append("**No ACCURACY_GATE.txt found. Per "
                   "BENCHMARK_PROTOCOL.md item 3.4, unresolved parity means "
                   "no performance claim ships from this campaign.**")
        report["accuracy_gate"] = None
    out.append("")

    out.append("## 6. Caveats this table cannot remove")
    out.append("")
    out.append(
        "* Kernel-level rigs (MoK, histogram replay) are not represented here "
        "and must never be quoted alongside these numbers as if they were "
        "end-to-end.\n"
        "* Serving runs at ~37.6%% fill (~1,539 real tokens/rank/step in a "
        "4,096-row padded mega); the banked kernel numbers are 100% fill.\n"
        "* Memory and replication costs are not in this table. A win that "
        "costs GB of replica cache is not free.\n"
        "* Open-loop cells answer 'how does it behave at X req/s', and X is "
        "bounded by the SLOWEST arm's measured ceiling. They do not show what "
        "the candidate could deliver above that rate.\n"
        "* AMD's fastest published DeepSeek-R1 path on this hardware is the "
        "ATOM engine with MTP speculative decoding, which cannot run inside "
        "the untouched vLLM image. 'Beats stock vLLM' is strictly weaker than "
        "'beats AMD's fastest stack'."
    )
    out.append("")
    return "\n".join(out), report
def _ratio_cell(ratios, gain=True):
    if not ratios:
        return "-"
    values = [r for _, r in ratios]
    if not gain:
        # Lower-is-better metric: display the ratio the way a reader expects,
        # i.e. >1 means the numerator arm is better.
        values = [1.0 / v for v in values if v]
    med, lo, hi, spread = median_and_spread(values)
    return "%s (%s–%s, spr %s%%)" % (
        _fmt(med, 3), _fmt(lo, 3), _fmt(hi, 3), _fmt(spread, 0))


def _ratio_json(ratios, gain=True):
    if not ratios:
        return None
    values = [r for _, r in ratios]
    if not gain:
        values = [1.0 / v for v in values if v]
    med, lo, hi, spread = median_and_spread(values)
    return {
        "median": med, "min": lo, "max": hi, "spread_pct": spread,
        "n_pairs": len(ratios),
        "inverted_for_display": not gain,
        "per_pair": {"%s/pair%s" % (key[0], key[1]):
                     (1.0 / value if (not gain and value) else value)
                     for key, value in ratios},
    }


def _cell_driver(index, cell):
    for key, record in index.items():
        if key[1] == cell:
            return record.driver
    return "closed"


def _cell_spec(index, cell):
    for key, record in index.items():
        if key[1] == cell:
            if record.driver == "open":
                return ("open-loop, %s req/s offered, burstiness %s, "
                        "ISL %s, OSL %s, n=%s"
                        % (_fmt(record.request_rate, 3), record.burstiness,
                           record.input_len, record.output_len,
                           record.num_prompts))
            return ("closed-loop, conc %s, ISL %s, OSL %s, n=%s"
                    % (record.concurrency, record.input_len,
                       record.output_len, record.num_prompts))
    return "?"


def main():
    parser = argparse.ArgumentParser(
        description="campaign-v5 extractor (v2/v3 client schema)")
    parser.add_argument("--campaign", action="append", required=True,
                        help="campaign ROOT (repeatable).  Roots are kept "
                             "SEPARATE: a ratio is never taken across two of "
                             "them, only pooled after being computed within "
                             "one pair of one root.")
    parser.add_argument("--candidate", default="m15",
                        help="the arm being claimed for (default m15)")
    parser.add_argument("--control", default="stock",
                        help="patched-stock control arm (default stock)")
    parser.add_argument("--metric", default="input_tps",
                        choices=("input_tps", "output_tps", "total_tps",
                                 "request_throughput"),
                        help="headline metric for CLOSED-loop cells; "
                             "input_tps = prefill throughput.  Open-loop cells "
                             "always use %s (lower is better) because their "
                             "throughput is pinned to the offered rate."
                             % OPEN_LOOP_METRIC)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    records = []
    for root in args.campaign:
        records.extend(load_campaign(root))
    if not records:
        raise SystemExit("no cell manifests found under %s"
                         % ", ".join(args.campaign))
    markdown, report = render(records, args)
    if args.out_md:
        with open(args.out_md, "w") as handle:
            handle.write(markdown + "\n")
    else:
        sys.stdout.write(markdown + "\n")
    if args.out_json:
        with open(args.out_json, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())

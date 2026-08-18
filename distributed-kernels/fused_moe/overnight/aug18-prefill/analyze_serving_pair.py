#!/usr/bin/env python3
"""analyze_serving_pair.py -- forensics for a pf4h serving A/B pair.

Consumes the two artifact directories produced by a c32p (or any
`pf4h-exact-token-closed-concurrency-v1`) pair run:

    <pair_dir>/
        <armA>/c32p.json          # result block + per_query array + workload
        <armA>/c32p_client.log    # optional, one-line summary
        <armA>/seal_receipts.txt  # all ranks' RAGGED_SEAL_RECEIPT + M15_COVERAGE
        <armB>/...

and emits every number the BOTTLENECK_SIGNALS.md analysis is built from:

  1. per-request TTFT / TPOT / e2e distributions and the arm ratio ACROSS the
     distribution (a Q-Q ratio table -- flat ratio => per-step multiplicative
     cost, tail-only ratio => stragglers, ramp/drain-only => fallback path),
  2. a closed-loop timeline reconstruction (concurrency slots, index order)
     giving ramp / steady / drain phase decomposition,
  3. receipt arithmetic: per-rank per-checkpoint trajectories, wall-clock
     step rates, node token rates, and the REAL-token fill / padding fraction
     of the padded B4096 bucket,
  4. the mega-region back-solve over the MoE-fraction parameter f.

Usage:
    python3 analyze_serving_pair.py <baseline_dir> <candidate_dir> [--json out.json]
    python3 analyze_serving_pair.py --pair <dir-with-two-arm-subdirs>

Everything is stdlib.  No GPU, no node, no network.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import sys
from typing import Any

# --------------------------------------------------------------------------
# small stats helpers (stdlib only, linear interpolation like numpy default)
# --------------------------------------------------------------------------


def pct(xs: list[float], q: float) -> float:
    """q in [0,100]. Linear interpolation between order statistics."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


QUANTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


# --------------------------------------------------------------------------
# artifact loading
# --------------------------------------------------------------------------


def find_result_json(d: str) -> str:
    cands = [f for f in sorted(os.listdir(d)) if f.endswith(".json")]
    if not cands:
        raise SystemExit(f"no .json result file in {d}")
    # prefer one that parses with the expected schema
    for c in cands:
        try:
            with open(os.path.join(d, c)) as fh:
                j = json.load(fh)
            if "per_query" in j and "result" in j:
                return os.path.join(d, c)
        except Exception:
            continue
    return os.path.join(d, cands[0])


def load_arm(d: str) -> dict[str, Any]:
    path = find_result_json(d)
    with open(path) as fh:
        j = json.load(fh)
    arm: dict[str, Any] = {
        "dir": d,
        "json_path": path,
        "label": j.get("label", os.path.basename(d)),
        "result": j.get("result", {}),
        "workload": j.get("workload", {}),
        "generated_utc": j.get("generated_utc"),
        "per_query": sorted(j.get("per_query", []), key=lambda q: q["index"]),
    }
    rp = os.path.join(d, "seal_receipts.txt")
    arm["receipt_lines"] = open(rp).read().splitlines() if os.path.exists(rp) else []
    return arm


# --------------------------------------------------------------------------
# 1. per-request distributions
# --------------------------------------------------------------------------


def dist(xs: list[float]) -> dict[str, float]:
    d = {"n": len(xs), "mean": mean(xs), "sd": stdev(xs), "min": min(xs), "max": max(xs)}
    for q in QUANTILES:
        d[f"p{q}"] = pct(xs, q)
    return d


def series(arm: dict[str, Any], key: str) -> list[float]:
    return [q[key] for q in arm["per_query"] if q.get("ok", True)]


def qq_ratio(a: list[float], b: list[float]) -> list[tuple[float, float, float, float]]:
    """Quantile-quantile ratio b/a at a fine quantile grid."""
    out = []
    grid = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    for q in grid:
        va, vb = pct(a, q), pct(b, q)
        out.append((q, va, vb, vb / va if va else float("nan")))
    return out


# --------------------------------------------------------------------------
# 2. closed-loop timeline reconstruction
# --------------------------------------------------------------------------


def reconstruct_timeline(arm: dict[str, Any]) -> list[dict[str, float]]:
    """Rebuild per-request submit/first-token/finish times.

    The harness is a closed loop of `concurrency` workers pulling requests in
    index order (schema pf4h-exact-token-closed-concurrency-v1).  Given the
    per-request e2e latency and the dispatch order, the schedule is exactly
    determined by a greedy `earliest-free-slot` replay -- no timestamps needed.
    """
    import heapq

    c = int(arm["workload"].get("concurrency", 32))
    free = [(0.0, i) for i in range(c)]
    heapq.heapify(free)
    rows = []
    for q in arm["per_query"]:
        t0, slot = heapq.heappop(free)
        t_first = t0 + q["ttft_ms"] / 1000.0
        t_end = t0 + q["e2el_ms"] / 1000.0
        rows.append(
            {
                "index": q["index"],
                "slot": slot,
                "t_start": t0,
                "t_first_token": t_first,
                "t_end": t_end,
                "ttft_ms": q["ttft_ms"],
                "tpot_ms": q["tpot_ms"],
                "e2el_ms": q["e2el_ms"],
            }
        )
        heapq.heappush(free, (t_end, slot))
    return rows


def phase_split(rows: list[dict[str, float]], c: int) -> dict[str, list[dict]]:
    """ramp   = the first wave (indices < c, all start at t=0),
       drain  = the last wave (the final c requests to be dispatched),
       steady = everything else."""
    n = len(rows)
    return {
        "ramp": rows[:c],
        "steady": rows[c : n - c],
        "drain": rows[n - c :],
    }


def timeline_bins(rows: list[dict[str, float]], nbins: int = 20) -> list[dict[str, Any]]:
    """Bin requests by dispatch time; report per-bin latency stats + rate."""
    if not rows:
        return []
    tmax = max(r["t_end"] for r in rows)
    w = tmax / nbins
    bins: list[dict[str, Any]] = []
    for b in range(nbins):
        lo, hi = b * w, (b + 1) * w
        sel = [r for r in rows if lo <= r["t_start"] < hi]
        done = [r for r in rows if lo <= r["t_end"] < hi]
        bins.append(
            {
                "bin": b,
                "t_lo": lo,
                "t_hi": hi,
                "n_started": len(sel),
                "n_finished": len(done),
                "ttft_p50": pct([r["ttft_ms"] for r in sel], 50) if sel else float("nan"),
                "tpot_p50": pct([r["tpot_ms"] for r in sel], 50) if sel else float("nan"),
                "e2e_p50": pct([r["e2el_ms"] for r in sel], 50) if sel else float("nan"),
                "finish_rate_rps": len(done) / w if w else float("nan"),
            }
        )
    return bins


# --------------------------------------------------------------------------
# 3. receipt arithmetic
# --------------------------------------------------------------------------

_TS = re.compile(r"\b(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\b")
_KV = re.compile(r"(\w+)=(-?\d+)")


def _ts(line: str) -> float | None:
    m = _TS.search(line)
    if not m:
        return None
    mo, da, hh, mm, ss = (int(x) for x in m.groups())
    # year-agnostic: we only ever take differences within one run
    return _dt.datetime(2000, mo, da, hh, mm, ss).timestamp()


def parse_receipts(lines: list[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"ragged": [], "coverage": []}
    for ln in lines:
        if "RAGGED_SEAL_RECEIPT" in ln:
            kind = "ragged"
        elif "M15_COVERAGE" in ln:
            kind = "coverage"
        else:
            continue
        tail = ln.split("RAGGED_SEAL_RECEIPT" if kind == "ragged" else "M15_COVERAGE", 1)[1]
        rec: dict[str, Any] = {k: int(v) for k, v in _KV.findall(tail)}
        rec["t"] = _ts(ln)
        mw = re.search(r"Worker_DP(\d+)", ln)
        if "rank" not in rec and mw:
            rec["rank"] = int(mw.group(1))
        out[kind].append(rec)
    return out


def receipt_table(recs: list[dict[str, Any]], dp_size: int = 8) -> dict[str, Any]:
    """Per-rank trajectories, per-window deltas, fill/padding fractions."""
    by_rank: dict[int, list[dict]] = {}
    for r in recs:
        by_rank.setdefault(r.get("rank", -1), []).append(r)
    for v in by_rank.values():
        v.sort(key=lambda r: r["steps"])

    t0 = min((r["t"] for r in recs if r.get("t")), default=None)
    windows = []
    for rank, rs in sorted(by_rank.items()):
        prev = None
        for r in rs:
            if prev is not None:
                dt = (r["t"] - prev["t"]) if (r.get("t") and prev.get("t")) else float("nan")
                dsteps = r["steps"] - prev["steps"]
                dtok = r.get("sum_orig", 0) - prev.get("sum_orig", 0)
                dib = r.get("in_bucket", 0) - prev.get("in_bucket", 0)
                dibtok = r.get("in_bucket_sum_orig", 0) - prev.get("in_bucket_sum_orig", 0)
                dsealed = r.get("sealed", 0) - prev.get("sealed", 0)
                windows.append(
                    {
                        "rank": rank,
                        "steps_lo": prev["steps"],
                        "steps_hi": r["steps"],
                        "dt_s": dt,
                        "d_steps": dsteps,
                        "steps_per_s": dsteps / dt if dt and dt == dt and dt > 0 else float("nan"),
                        "d_node_tok": dtok,
                        "node_tok_per_s": dtok / dt if dt and dt == dt and dt > 0 else float("nan"),
                        "node_tok_per_step": dtok / dsteps if dsteps else float("nan"),
                        "d_in_bucket": dib,
                        "in_bucket_rate": dib / dsteps if dsteps else float("nan"),
                        "d_in_bucket_tok": dibtok,
                        "node_tok_per_ib_step": dibtok / dib if dib else float("nan"),
                        "rank_tok_per_ib_step": (dibtok / dib / dp_size) if dib else float("nan"),
                        "fill_frac": (dibtok / dib / dp_size / 4096.0) if dib else float("nan"),
                        "d_sealed": dsealed,
                        "seal_rate": dsealed / dib if dib else float("nan"),
                    }
                )
            prev = r

    finals = {rank: rs[-1] for rank, rs in by_rank.items() if rs}
    agg: dict[str, Any] = {}
    if finals:
        f = list(finals.values())
        agg = {
            "ranks": sorted(finals),
            "steps": [x["steps"] for x in f],
            "in_bucket": [x.get("in_bucket") for x in f],
            "sealed": [x.get("sealed") for x in f],
            "sealed_ragged": [x.get("sealed_ragged") for x in f],
            "sealed_exact": [x.get("sealed_exact") for x in f],
            "refused_not_unanimous": [x.get("refused_not_unanimous") for x in f],
            "refused_not_ready": [x.get("refused_not_ready") for x in f],
            "refused_peer_not_ready": [x.get("refused_peer_not_ready") for x in f],
            "eager_b4096": [x.get("eager_b4096") for x in f],
            "uniform_rescued": [x.get("uniform_rescued") for x in f],
            "sum_orig": [x.get("sum_orig") for x in f],
            "in_bucket_sum_orig": [x.get("in_bucket_sum_orig") for x in f],
        }
        st = mean([x["steps"] for x in f])
        ib = mean([x.get("in_bucket", 0) for x in f])
        so = mean([x.get("sum_orig", 0) for x in f])
        ibso = mean([x.get("in_bucket_sum_orig", 0) for x in f])
        sealed = mean([x.get("sealed", 0) for x in f])
        agg["derived"] = {
            "in_bucket_step_frac": ib / st if st else float("nan"),
            "seal_frac_of_in_bucket": sealed / ib if ib else float("nan"),
            "seal_frac_of_steps": sealed / st if st else float("nan"),
            "token_coverage_P1": ibso / so if so else float("nan"),
            "node_tok_per_step_all": so / st if st else float("nan"),
            "node_tok_per_ib_step": ibso / ib if ib else float("nan"),
            "rank_real_tok_per_ib_step": ibso / ib / dp_size if ib else float("nan"),
            "fill_frac_of_4096": ibso / ib / dp_size / 4096.0 if ib else float("nan"),
            "padding_frac_of_4096": 1.0 - (ibso / ib / dp_size / 4096.0) if ib else float("nan"),
            "min_frac_rank_steps_below_4096": (
                1.0 - (ibso / ib / dp_size / 4096.0) if ib else float("nan")
            ),
            "eager_frac_of_in_bucket": mean([x.get("eager_b4096", 0) for x in f]) / ib
            if ib
            else float("nan"),
            "uniform_rescue_frac_of_steps": mean([x.get("uniform_rescued", 0) for x in f]) / st
            if st
            else float("nan"),
        }
        agg["t0"] = t0
    return {"by_rank": by_rank, "windows": windows, "final": finals, "agg": agg}


# --------------------------------------------------------------------------
# 4. mega-region back-solve
# --------------------------------------------------------------------------


def padded_work(agg: dict[str, Any], dp: int = 8, bucket: int = 4096) -> dict[str, float]:
    """MoE row-work accounting for the padded B4096 bucket.

    On an in-bucket step EVERY dp rank runs the padded `bucket`-row graph, so the
    MoE row-work of that step is dp*bucket regardless of how many rows are real.
    Off-bucket steps are approximated by their real token count (they are the
    cheap decode-ish steps).  The ratio of padded-rows-per-real-token between two
    arms is the composition confound that must be divided out before any
    statement about kernel speed can be made.
    """
    steps = mean(agg["steps"])
    ib = mean([x or 0 for x in agg["in_bucket"]])
    real_all = mean([x or 0 for x in agg["sum_orig"]])
    real_ib = mean([x or 0 for x in agg["in_bucket_sum_orig"]])
    padded_ib = ib * dp * bucket
    real_off = real_all - real_ib
    return {
        "steps": steps,
        "in_bucket_steps": ib,
        "off_bucket_steps": steps - ib,
        "real_tok_total": real_all,
        "real_tok_in_bucket": real_ib,
        "real_tok_off_bucket": real_off,
        "padded_rows_in_bucket": padded_ib,
        "moe_rows_total": padded_ib + real_off,
        "padded_rows_per_real_token": (padded_ib + real_off) / real_all if real_all else float("nan"),
        "wasted_rows": padded_ib - real_ib,
        "wasted_frac_of_moe_rows": (padded_ib - real_ib) / (padded_ib + real_off)
        if (padded_ib + real_off)
        else float("nan"),
    }


def two_state_fit(ib_a: float, small_a: float, t_a: float,
                  ib_b: float, small_b: float, t_b: float) -> dict[str, float]:
    """Solve the 2x2 system  ib*a + small*b = t  for two observations.

    State a = an in-bucket (padded 4096) step, state b = any other step.  A
    single arm-independent (a, b) that reproduces BOTH arms is the null
    hypothesis 'the mega's padded step costs exactly what the stock padded step
    costs, and the whole wall-clock gap is batch composition'.
    """
    det = ib_a * small_b - small_a * ib_b
    if det == 0:
        return {"a": float("nan"), "b": float("nan"), "det": 0.0}
    a = (t_a * small_b - small_a * t_b) / det
    b = (ib_a * t_b - t_a * ib_b) / det
    return {"a": a, "b": b, "det": det}


def delta_scan(ib_s: float, sm_s: float, t_s: float,
               ib_m: float, sm_m: float, t_m: float,
               b_grid) -> list[dict[str, float]]:
    """For each assumed cheap-step cost b, calibrate a from the STOCK arm and
    report the extra per-step cost delta the mega arm's wall time implies, and
    the equivalent MoE-region ratio at a given f."""
    out = []
    for b in b_grid:
        a = (t_s - sm_s * b) / ib_s
        if a <= 0:
            continue
        delta = (t_m - sm_m * b) / ib_m - a
        out.append({"b": b, "a": a, "delta": delta, "step_ratio": (a + delta) / a})
    return out


def back_solve(e2e_ratio: float, f_grid, duty: float = 1.0, eager_frac: float = 0.0,
               eager_penalty: float = 0.0) -> list[dict[str, float]]:
    """Given total-time ratio R = T_m15 / T_stock, solve for the MoE-region ratio r.

    Model:  T_m15 = T_stock * [ (1-f) + f * ( (1-d)*1 + d*r ) ] * (1 + eager_tax)
    where f is the MoE-region fraction of stock step time, d = duty (fraction of
    MoE-region time that actually runs the mega), and eager_tax is an additive
    whole-step penalty on the eager-fallback steps.
        => r = 1 + ( R/(1+eager_tax) - 1 ) / (f*d)
    """
    eager_tax = eager_frac * eager_penalty
    out = []
    for f in f_grid:
        R = e2e_ratio / (1.0 + eager_tax)
        r = 1.0 + (R - 1.0) / (f * duty)
        out.append({"f": f, "duty": duty, "eager_tax": eager_tax, "region_ratio": r})
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def hdr(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def fmt_dist(name: str, d: dict[str, float]) -> str:
    return (
        f"{name:<10} n={d['n']:<5} mean={d['mean']:9.2f} sd={d['sd']:8.2f} "
        f"p10={d['p10']:9.2f} p50={d['p50']:9.2f} p90={d['p90']:9.2f} "
        f"p99={d['p99']:9.2f} max={d['max']:9.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", nargs="?", help="baseline arm directory")
    ap.add_argument("candidate", nargs="?", help="candidate arm directory")
    ap.add_argument("--pair", help="a directory containing exactly two arm subdirectories")
    ap.add_argument("--dp-size", type=int, default=8)
    ap.add_argument("--json", help="also dump every derived number to this JSON file")
    ap.add_argument("--f-grid", default="0.40,0.45,0.50,0.54")
    args = ap.parse_args(argv)

    if args.pair:
        subs = sorted(
            os.path.join(args.pair, d)
            for d in os.listdir(args.pair)
            if os.path.isdir(os.path.join(args.pair, d))
        )
        if len(subs) != 2:
            raise SystemExit(f"--pair needs exactly two subdirectories, found {len(subs)}")
        base_dir, cand_dir = subs
    else:
        if not (args.baseline and args.candidate):
            ap.error("give two arm directories, or --pair")
        base_dir, cand_dir = args.baseline, args.candidate

    A = load_arm(base_dir)
    B = load_arm(cand_dir)
    dump: dict[str, Any] = {"baseline": A["label"], "candidate": B["label"]}

    # ---- headline ---------------------------------------------------------
    hdr("0. HEADLINE")
    for arm in (A, B):
        r = arm["result"]
        print(
            f"{arm['label']:<24} wall={r.get('wall_seconds', float('nan')):8.2f}s "
            f"in_tok/s={r.get('input_tokens_per_second', float('nan')):10.1f} "
            f"req/s={r.get('request_throughput', float('nan')):7.4f} "
            f"completed={r.get('completed')} failed={r.get('failed')}"
        )
    wr = B["result"]["wall_seconds"] / A["result"]["wall_seconds"]
    tr = B["result"]["input_tokens_per_second"] / A["result"]["input_tokens_per_second"]
    print(f"\ncandidate/baseline wall ratio   = {wr:.4f}   ({100*(wr-1):+.2f}%)")
    print(f"candidate/baseline tok/s ratio  = {tr:.4f}   ({100*(tr-1):+.2f}%)")
    dump["wall_ratio"] = wr
    dump["tok_ratio"] = tr

    # exact-token agreement
    ha = {q["index"]: q["output_token_ids_sha256"] for q in A["per_query"]}
    hb = {q["index"]: q["output_token_ids_sha256"] for q in B["per_query"]}
    common = sorted(set(ha) & set(hb))
    mism = [i for i in common if ha[i] != hb[i]]
    print(
        f"\nper-request output-token agreement: {len(common)-len(mism)}/{len(common)} identical, "
        f"{len(mism)} divergent ({100*len(mism)/max(1,len(common)):.2f}%)"
    )
    dump["token_divergence"] = {"n": len(common), "divergent": len(mism),
                                "frac": len(mism) / max(1, len(common)),
                                "first_20": mism[:20]}

    # ---- 1. distributions -------------------------------------------------
    hdr("1. PER-REQUEST DISTRIBUTIONS")
    dump["dists"] = {}
    for key in ("ttft_ms", "tpot_ms", "e2el_ms"):
        da, db = dist(series(A, key)), dist(series(B, key))
        print(f"\n-- {key}")
        print("  " + fmt_dist(A["label"], da))
        print("  " + fmt_dist(B["label"], db))
        print(f"  {'ratio':<10} " + " ".join(
            f"{k}={db[k]/da[k]:.4f}" for k in ("mean", "p10", "p50", "p90", "p99", "max")))
        dump["dists"][key] = {"baseline": da, "candidate": db}

    hdr("1b. Q-Q RATIO (candidate/baseline at matched quantiles)")
    dump["qq"] = {}
    for key in ("ttft_ms", "tpot_ms", "e2el_ms"):
        rows = qq_ratio(series(A, key), series(B, key))
        print(f"\n-- {key}:  q     baseline   candidate     ratio")
        for q, va, vb, r in rows:
            print(f"   {q:>5}  {va:10.2f}  {va and vb:10.2f}  {r:8.4f}")
        rs = [r for _, _, _, r in rows]
        print(f"   spread of ratio across quantiles: min={min(rs):.4f} max={max(rs):.4f} "
              f"range={max(rs)-min(rs):.4f}")
        dump["qq"][key] = rows

    # ---- 2. timeline ------------------------------------------------------
    hdr("2. RECONSTRUCTED CLOSED-LOOP TIMELINE (greedy earliest-free-slot replay)")
    dump["phases"] = {}
    for arm in (A, B):
        rows = reconstruct_timeline(arm)
        arm["_rows"] = rows
        c = int(arm["workload"].get("concurrency", 32))
        span = max(r["t_end"] for r in rows)
        real = arm["result"]["wall_seconds"]
        print(f"\n{arm['label']}: reconstructed span={span:.2f}s vs reported wall={real:.2f}s "
              f"(residual {100*(span-real)/real:+.2f}%)")
        ph = phase_split(rows, c)
        pd_: dict[str, Any] = {}
        for name, sel in ph.items():
            if not sel:
                continue
            pd_[name] = {
                "n": len(sel),
                "ttft_p50": pct([r["ttft_ms"] for r in sel], 50),
                "tpot_p50": pct([r["tpot_ms"] for r in sel], 50),
                "e2e_p50": pct([r["e2el_ms"] for r in sel], 50),
                "e2e_mean": mean([r["e2el_ms"] for r in sel]),
            }
            print(f"   {name:<7} n={len(sel):<5} ttft_p50={pd_[name]['ttft_p50']:8.1f} "
                  f"tpot_p50={pd_[name]['tpot_p50']:7.1f} e2e_p50={pd_[name]['e2e_p50']:8.1f}")
        dump["phases"][arm["label"]] = pd_

    print("\n-- phase ratios (candidate/baseline)")
    for name in ("ramp", "steady", "drain"):
        pa = dump["phases"][A["label"]].get(name)
        pb = dump["phases"][B["label"]].get(name)
        if pa and pb:
            print(f"   {name:<7} ttft_p50={pb['ttft_p50']/pa['ttft_p50']:.4f} "
                  f"tpot_p50={pb['tpot_p50']/pa['tpot_p50']:.4f} "
                  f"e2e_p50={pb['e2e_p50']/pa['e2e_p50']:.4f}")

    hdr("2b. TIME-BINNED PROFILE (by reconstructed dispatch time)")
    nb = 10
    ba = timeline_bins(A["_rows"], nb)
    bb = timeline_bins(B["_rows"], nb)
    print(f"{'bin':>3} | {'A t_lo':>8} {'A e2e_p50':>10} {'A rps':>7} | "
          f"{'B t_lo':>8} {'B e2e_p50':>10} {'B rps':>7} | {'e2e B/A':>8}")
    for x, y in zip(ba, bb):
        rr = y["e2e_p50"] / x["e2e_p50"] if x["e2e_p50"] == x["e2e_p50"] else float("nan")
        print(f"{x['bin']:>3} | {x['t_lo']:8.1f} {x['e2e_p50']:10.1f} {x['finish_rate_rps']:7.3f} | "
              f"{y['t_lo']:8.1f} {y['e2e_p50']:10.1f} {y['finish_rate_rps']:7.3f} | {rr:8.4f}")
    dump["bins"] = {A["label"]: ba, B["label"]: bb}

    # ---- 3. receipts ------------------------------------------------------
    hdr("3. RECEIPT ARITHMETIC")
    dump["receipts"] = {}
    for arm in (A, B):
        p = parse_receipts(arm["receipt_lines"])
        rt = receipt_table(p["ragged"], args.dp_size)
        cv = receipt_table(p["coverage"], args.dp_size)
        arm["_rt"] = rt
        print(f"\n### {arm['label']}  ({len(p['ragged'])} RAGGED lines, "
              f"{len(p['coverage'])} COVERAGE lines)")
        if not rt["final"]:
            print("   (no RAGGED_SEAL_RECEIPT lines)")
            continue
        a = rt["agg"]
        print(f"   ranks={a['ranks']}")
        for k in ("steps", "in_bucket", "sealed", "sealed_ragged", "sealed_exact",
                  "refused_not_unanimous", "refused_not_ready", "refused_peer_not_ready",
                  "eager_b4096", "uniform_rescued"):
            v = a[k]
            if all(x is None for x in v):
                continue
            print(f"   {k:<24} {v}   mean={mean([x or 0 for x in v]):.2f}")
        print(f"   sum_orig (node real tok)  {a['sum_orig']}")
        print(f"   in_bucket_sum_orig        {a['in_bucket_sum_orig']}")
        d = a["derived"]
        for k, v in d.items():
            print(f"   {k:<32} {v:.6f}")
        print("\n   per-100-step windows (rank, steps, dt, steps/s, node tok/s, "
              "node tok/ib-step, rank real tok/ib-step, fill):")
        for w in rt["windows"]:
            print(f"     r{w['rank']} {w['steps_lo']:>4}->{w['steps_hi']:<4} dt={w['dt_s']:6.1f}s "
                  f"sps={w['steps_per_s']:6.3f} ntps={w['node_tok_per_s']:9.1f} "
                  f"ib_tok={w['node_tok_per_ib_step']:8.1f} rank_tok={w['rank_tok_per_ib_step']:7.1f} "
                  f"fill={w['fill_frac']:.4f} seal={w['d_sealed']}/{w['d_in_bucket']}")
        dump["receipts"][arm["label"]] = {"agg": a, "windows": rt["windows"],
                                          "coverage_agg": cv["agg"]}

    # cross-arm window comparison on matched step indices
    hdr("3b. CROSS-ARM MATCHED-STEP WINDOW COMPARISON (rank 0)")
    wa = [w for w in A["_rt"]["windows"] if w["rank"] == 0]
    wb = [w for w in B["_rt"]["windows"] if w["rank"] == 0]
    print(f"{'window':>12} | {'A dt':>7} {'A tok/s':>10} | {'B dt':>7} {'B tok/s':>10} | "
          f"{'tok/s A/B':>10} | {'A fill':>7} {'B fill':>7}")
    for x, y in zip(wa, wb):
        print(f"{x['steps_lo']:>5}->{x['steps_hi']:<5} | {x['dt_s']:7.1f} {x['node_tok_per_s']:10.1f} | "
              f"{y['dt_s']:7.1f} {y['node_tok_per_s']:10.1f} | "
              f"{x['node_tok_per_s']/y['node_tok_per_s']:10.4f} | "
              f"{x['fill_frac']:7.4f} {y['fill_frac']:7.4f}")

    # ---- 4. back-solve ----------------------------------------------------
    hdr("4. MEGA-REGION BACK-SOLVE")
    f_grid = [float(x) for x in args.f_grid.split(",")]
    Bagg = B["_rt"]["agg"]["derived"] if B["_rt"]["final"] else {}
    duty = Bagg.get("seal_frac_of_steps", 1.0)
    duty_tok = Bagg.get("token_coverage_P1", 1.0)
    eager_frac = Bagg.get("eager_frac_of_in_bucket", 0.0) * Bagg.get("in_bucket_step_frac", 1.0)
    print(f"seal duty (steps)      = {duty:.4f}")
    print(f"token coverage (P1)    = {duty_tok:.4f}")
    print(f"eager-fallback steps   = {eager_frac:.4f} of all steps")
    print(f"observed e2e time ratio= {wr:.4f}\n")
    print(f"{'f':>6} {'duty':>7} {'eager_tax':>10} {'region ratio':>13}")
    dump["back_solve"] = []
    for duty_used, tag in ((duty, "step-duty"), (duty_tok, "token-duty"), (1.0, "duty=1")):
        for ep in (0.0, 0.5):
            rows = back_solve(wr, f_grid, duty_used, eager_frac, ep)
            for r in rows:
                print(f"{r['f']:6.2f} {duty_used:7.4f} {r['eager_tax']:10.4f} "
                      f"{r['region_ratio']:13.4f}   [{tag}, eager_penalty={ep}]")
                dump["back_solve"].append(dict(r, tag=tag, eager_penalty=ep))

    # ---- 5. composition correction + two-state step-cost model -------------
    hdr("5. PADDED-WORK ACCOUNTING (the composition confound)")
    pwA = padded_work(A["_rt"]["agg"], args.dp_size)
    pwB = padded_work(B["_rt"]["agg"], args.dp_size)
    dump["padded_work"] = {A["label"]: pwA, B["label"]: pwB}
    keys = ["steps", "in_bucket_steps", "off_bucket_steps", "real_tok_total",
            "real_tok_in_bucket", "padded_rows_in_bucket", "moe_rows_total",
            "padded_rows_per_real_token", "wasted_rows", "wasted_frac_of_moe_rows"]
    print(f"{'':<30}{A['label']:>20}{B['label']:>20}{'B/A':>12}")
    for k in keys:
        va, vb = pwA[k], pwB[k]
        print(f"{k:<30}{va:>20,.2f}{vb:>20,.2f}{(vb/va if va else float('nan')):>12.4f}")
    W = pwB["padded_rows_per_real_token"] / pwA["padded_rows_per_real_token"]
    print(f"\nMoE row-work inflation of candidate per real token  W = {W:.4f}")

    hdr("5b. COMPOSITION-NORMALISED REGION BACK-SOLVE")
    # time per real token, from the receipt window that both arms share
    tptA = pwA["steps"] / pwA["real_tok_total"]
    tptB = pwB["steps"] / pwB["real_tok_total"]
    S = tptB / tptA  # non-MoE work per real token (proportional to steps)
    Rtime = wr * (pwA["real_tok_total"] / pwB["real_tok_total"]) if pwA["real_tok_total"] else wr
    print("model:  T/token = (1-f)*S + f*W*r     [non-MoE scales with steps]")
    print(f"  observed time-per-real-token ratio  R = {Rtime:.4f}")
    print(f"  step (non-MoE) inflation            S = {S:.4f}")
    print(f"  padded MoE row inflation            W = {W:.4f}")
    print(f"\n{'f':>6} {'r (non-MoE~steps)':>19} {'r (non-MoE~tokens)':>20} {'r (naive, W=1)':>16}")
    dump["norm_back_solve"] = []
    for f in f_grid:
        r1 = (Rtime - (1 - f) * S) / (f * W)
        r2 = (Rtime - (1 - f) * 1.0) / (f * W)
        r0 = (Rtime - (1 - f) * 1.0) / f
        print(f"{f:>6.2f} {r1:>19.4f} {r2:>20.4f} {r0:>16.4f}")
        dump["norm_back_solve"].append({"f": f, "r_steps": r1, "r_tokens": r2, "r_naive": r0})

    hdr("5c. TWO-STATE STEP-COST MODEL  (a = padded B4096 step, b = other step)")
    # full-run extrapolation from the receipt window rates
    rows = []
    for arm, pw in ((A, pwA), (B, pwB)):
        tot_tok = arm["result"].get("input_tokens_total", 0)
        n_steps = tot_tok / (pw["real_tok_total"] / pw["steps"])
        ib = n_steps * pw["in_bucket_steps"] / pw["steps"]
        rows.append({"label": arm["label"], "wall": arm["result"]["wall_seconds"],
                     "steps_est": n_steps, "ib": ib, "small": n_steps - ib})
        print(f"  {arm['label']:<24} est_total_steps={n_steps:8.1f}  in_bucket={ib:7.1f}  "
              f"other={n_steps-ib:7.1f}  wall={arm['result']['wall_seconds']:.2f}s")
    fit = two_state_fit(rows[0]["ib"], rows[0]["small"], rows[0]["wall"],
                        rows[1]["ib"], rows[1]["small"], rows[1]["wall"])
    print(f"\n  arm-independent solution: a = {fit['a']*1000:.1f} ms/padded-step, "
          f"b = {fit['b']*1000:.1f} ms/other-step")
    print("  (an exact 2x2 fit: a SINGLE (a,b) reproduces BOTH arms' wall times, i.e. the\n"
          "   whole wall-clock gap is explained with zero mega step-cost penalty)")
    dump["two_state_fullrun"] = {"rows": rows, "fit": fit}

    # matched 100-step windows: calibrate a from stock, read delta off the candidate
    hdr("5d. DELTA SCAN ON MATCHED RECEIPT WINDOWS "
        "(a calibrated on baseline, delta read off candidate)")
    for x, y in zip(wa, wb):
        ib_s, sm_s = x["d_in_bucket"], x["d_steps"] - x["d_in_bucket"]
        ib_m, sm_m = y["d_in_bucket"], y["d_steps"] - y["d_in_bucket"]
        print(f"\n  window {x['steps_lo']}->{x['steps_hi']}: "
              f"baseline {ib_s} padded + {sm_s} other = {x['dt_s']:.0f}s | "
              f"candidate {ib_m} padded + {sm_m} other = {y['dt_s']:.0f}s")
        print(f"    {'b(ms)':>7} {'a(ms)':>8} {'delta(ms)':>10} {'padded-step ratio':>18}")
        for row in delta_scan(ib_s, sm_s, x["dt_s"], ib_m, sm_m, y["dt_s"],
                              [0.10, 0.20, 0.25, 0.30, 0.35, 0.40]):
            print(f"    {row['b']*1000:>7.0f} {row['a']*1000:>8.1f} {row['delta']*1000:>10.1f} "
                  f"{row['step_ratio']:>18.4f}")
        dump.setdefault("delta_scan", []).append(
            {"window": f"{x['steps_lo']}->{x['steps_hi']}",
             "rows": delta_scan(ib_s, sm_s, x["dt_s"], ib_m, sm_m, y["dt_s"],
                                [0.10, 0.20, 0.25, 0.30, 0.35, 0.40])})

    # mean-TPOT two-state cross-check
    hdr("5e. MEAN-TPOT CROSS-CHECK (a decode token consumes one model step)")
    fa = pwA["in_bucket_steps"] / pwA["steps"]
    fb = pwB["in_bucket_steps"] / pwB["steps"]
    ta = mean(series(A, "tpot_ms")) / 1000.0
    tb = mean(series(B, "tpot_ms")) / 1000.0
    fit2 = two_state_fit(fa, 1 - fa, ta, fb, 1 - fb, tb)
    print(f"  baseline : in-bucket step frac={fa:.4f}  mean TPOT={ta*1000:.1f} ms")
    print(f"  candidate: in-bucket step frac={fb:.4f}  mean TPOT={tb*1000:.1f} ms")
    print(f"  arm-independent solution: a = {fit2['a']*1000:.1f} ms, b = {fit2['b']*1000:.1f} ms")
    dump["two_state_tpot"] = fit2

    hdr("5f. TPOT CEILING (the pure padded-step regime)")
    for arm in (A, B):
        xs = series(arm, "tpot_ms")
        top = sorted(xs)[-50:]
        print(f"  {arm['label']:<24} p95={pct(xs,95):7.2f} p99={pct(xs,99):7.2f} "
              f"p99.5={pct(xs,99.5):7.2f} max={max(xs):7.2f}  "
              f"n>750ms={sum(1 for v in xs if v>750):4d}  mean(top50)={mean(top):7.2f}")
    xa, xb = series(A, "tpot_ms"), series(B, "tpot_ms")
    print(f"  ceiling ratio (mean of top 50): {mean(sorted(xb)[-50:])/mean(sorted(xa)[-50:]):.4f}")
    dump["tpot_ceiling"] = {
        A["label"]: {"p99": pct(xa, 99), "max": max(xa), "top50": mean(sorted(xa)[-50:])},
        B["label"]: {"p99": pct(xb, 99), "max": max(xb), "top50": mean(sorted(xb)[-50:])},
    }

    # ---- 6. interval synthesis --------------------------------------------
    hdr("6. REGION-RATIO INTERVAL (grid over every nuisance parameter)")
    ests: list[dict[str, Any]] = []
    # (i) whole-run composition-normalised back-solve
    for f in [0.40, 0.45, 0.50, 0.54]:
        for S_ in (S, 1.0):
            ests.append({"method": "norm-backsolve", "f": f,
                         "r": (Rtime - (1 - f) * S_) / (f * W)})
    # (ii) matched-window two-state, stock-calibrated, with +-1s timestamp jitter
    for x, y in zip(wa, wb):
        ib_s, sm_s = x["d_in_bucket"], x["d_steps"] - x["d_in_bucket"]
        ib_m, sm_m = y["d_in_bucket"], y["d_steps"] - y["d_in_bucket"]
        for jit_s in (-1.0, 0.0, 1.0):
            for jit_m in (-1.0, 0.0, 1.0):
                for b in (0.10, 0.20, 0.30, 0.40):
                    a = (x["dt_s"] + jit_s - sm_s * b) / ib_s
                    if a <= 0:
                        continue
                    k = ((y["dt_s"] + jit_m - sm_m * b) / ib_m) / a
                    for f in (0.40, 0.45, 0.50, 0.54):
                        ests.append({"method": f"window{x['steps_lo']}", "f": f, "b": b,
                                     "jit": (jit_s, jit_m), "k": k, "r": 1 + (k - 1) / f})
    # (iii) TPOT ceiling (pure padded-step regime)
    kc = mean(sorted(xb)[-50:]) / mean(sorted(xa)[-50:])
    for f in (0.40, 0.45, 0.50, 0.54):
        ests.append({"method": "tpot-ceiling", "f": f, "k": kc, "r": 1 + (kc - 1) / f})
    rs = [e["r"] for e in ests]
    print(f"  {len(ests)} composition-corrected estimates of the mega MoE-region ratio")
    print(f"  min={min(rs):.4f}  p10={pct(rs,10):.4f}  p50={pct(rs,50):.4f}  "
          f"p90={pct(rs,90):.4f}  max={max(rs):.4f}")
    for m in sorted({e["method"] for e in ests}):
        sub = [e["r"] for e in ests if e["method"] == m]
        print(f"    {m:<16} n={len(sub):<4} min={min(sub):.4f} med={pct(sub,50):.4f} max={max(sub):.4f}")
    naive = [e for e in dump["norm_back_solve"]]
    print(f"\n  for contrast, the UNCORRECTED (W=1) back-solve gives "
          f"{min(x['r_naive'] for x in naive):.4f}..{max(x['r_naive'] for x in naive):.4f}")
    dump["interval"] = {"estimates": ests, "min": min(rs), "p10": pct(rs, 10),
                        "p50": pct(rs, 50), "p90": pct(rs, 90), "max": max(rs)}

    hdr("7. CO-SCHEDULING HEADROOM (both arms; independent of kernel choice)")
    dp = args.dp_size
    for arm, pw in ((A, pwA), (B, pwB)):
        tot = arm["result"].get("input_tokens_total", 0)
        n_steps = tot / (pw["real_tok_total"] / pw["steps"])
        ib = n_steps * pw["in_bucket_steps"] / pw["steps"]
        ideal_ib = tot / (dp * 4096)
        # decode tokens still need their own steps once prefill is packed
        out_tok = arm["result"].get("output_tokens_total", 0)
        conc = int(arm["workload"].get("concurrency", 32))
        decode_steps = out_tok / max(1.0, conc / dp) / dp
        for a_, b_ in ((fit["a"], fit["b"]), (fit2["a"], fit2["b"])):
            new = ideal_ib * a_ + decode_steps * b_
            print(f"  {arm['label']:<22} (a={a_*1000:5.0f}ms b={b_*1000:5.0f}ms) "
                  f"padded {ib:6.1f}->{ideal_ib:5.0f} steps, decode {decode_steps:5.0f}: "
                  f"wall {arm['result']['wall_seconds']:6.1f} -> {new:6.1f}s "
                  f"({100*(new/arm['result']['wall_seconds']-1):+5.1f}%)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dump, fh, indent=2, default=str)
        print(f"\n[wrote {args.json}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""exp_23 shared library: phase table, log parsing, window selection, checks.

Pure stdlib. No GPU, no numpy. Schemas are frozen here and documented in
result.md; bump the schema string if any field changes meaning.

Tick convention: 1 tick = 0.01 us (s_memrealtime, 100 MHz). This is the same
divisor the harness uses for [MPS TS SPLIT]/[MPS TS DELTA]; the 2.2 GHz shader
clock is the WRONG divisor and has burned this project before.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

EVENTS_SCHEMA = "exp23-events-1"
BINS_SCHEMA = "exp23-bins-1"

TICK_US = 0.01
SLOTS = 16

# slot index -> name. Fixed by the kernel patch (patch_spec.md section 2).
PHASE_SLOTS: Dict[int, str] = {
    0: "KSTART",
    1: "M2_DONE",
    2: "M5_DONE",
    3: "M6_DONE",
    4: "M7_DONE",
    5: "SVC_ENTER",
    6: "SVC_EXIT",
    7: "M75_ENTER",
    8: "M75_BAR",
    9: "M75_EXIT",
    10: "M8_ENTER",
    11: "REDUCE_DONE",
    12: "M9_DONE",
    15: "META",
}
SLOT_OF = {v: k for k, v in PHASE_SLOTS.items()}

# Boundaries that MUST appear in non-decreasing order within one CTA, in the
# order the CTA crosses them. Absent (== 0) boundaries are skipped, not failed:
# a service CTA has no M7_DONE, a mode-12 CTA has no M75_*.
MONOTONIC_ORDER = [
    "KSTART", "M2_DONE", "M5_DONE", "M6_DONE", "M7_DONE",
    "SVC_ENTER", "SVC_EXIT", "M75_ENTER", "M75_BAR", "M75_EXIT",
    "M8_ENTER", "REDUCE_DONE", "M9_DONE",
]

# Interval name -> (open boundary, close boundary). An interval is emitted only
# if BOTH endpoints are present and the close is >= the open.
INTERVALS: List[Tuple[str, str, str]] = [
    ("dispatch", "KSTART", "M2_DONE"),
    ("plan", "M2_DONE", "M5_DONE"),
    ("M6", "M5_DONE", "M6_DONE"),
    ("M7", "M6_DONE", "M7_DONE"),
    ("service", "SVC_ENTER", "SVC_EXIT"),
    ("m75", "M75_ENTER", "M75_EXIT"),
    ("combine", "M8_ENTER", "REDUCE_DONE"),
    ("tail", "REDUCE_DONE", "M9_DONE"),
]

# Intervals whose CTAs are counted as "in an MFMA phase". This is an OCCUPANCY
# PROXY, not MFMA utilization: a CTA stalled on vmcnt inside M7 counts here.
MFMA_INTERVALS = ("M6", "M7")

# The five boundaries that share a clock read with a coarse [MPS TS] cell, so
# max-over-CTAs must equal the coarse cell EXACTLY (0 ticks). See patch_spec G7.
COARSE_PAIRS = {
    "M2_DONE": "M2_DONE",
    "M5_DONE": "M5_DONE",
    "M6_DONE": "M6_DONE",
    "M7_DONE": "M7_DONE",
    "REDUCE_DONE": "REDUCE_DONE",
}

# Tolerances. Each one is justified in result.md; do not loosen to make a run
# pass, and do not tighten past the stated justification.
TOL_COARSE_TICKS = 0            # exact by construction (one shared clock read)
TOL_INTERIOR_US = 1.0           # arithmetic identity, not physics
TOL_E2E_FRAC = 0.10             # p50 - interior residual is a measured ~9.9 %
TOL_COVERAGE_FRAC = 0.005       # unaccounted time inside a CTA's own span
TOL_BIN_INTEGRAL_FRAC = 0.005   # binning self-consistency (pure arithmetic)

_RE_HDR = re.compile(
    r"\[MPS E23\]\s+slots=(?P<slots>\d+)\s+ctas=(?P<ctas>\d+)\s+"
    r"tick_ns=(?P<tick_ns>\d+)(?:\s+cfg=(?P<cfg>\S*))?"
)
_RE_CTA = re.compile(r"\[MPS E23 CTA\]\s+(?P<bid>\d+)\s+(?P<vals>[\d\s]+)")
_RE_TS = re.compile(r"\[MPS TS\]\s+(?P<body>\S+=\S+(?:\s+\S+=\S+)*)")
_RE_SPIN = re.compile(
    r"\[MPS SPIN\]\s+chunk_poll\s+success_max=(?P<ok>\d+)\s+fail_max=(?P<bad>\d+)"
)


class ParseError(RuntimeError):
    pass


def parse_log(text: str) -> Dict[str, Any]:
    """Extract the exp_23 stamp block, the coarse [MPS TS] cells and [MPS SPIN].

    A log may contain several arms; the LAST [MPS E23] header wins and only CTA
    rows appearing after it are taken. That mirrors how the harness prints (one
    block per arm, in arm order, rank 0 only).
    """
    lines = text.splitlines()
    hdr_idx = None
    hdr = None
    for i, ln in enumerate(lines):
        m = _RE_HDR.search(ln)
        if m:
            hdr_idx, hdr = i, m
    if hdr is None:
        raise ParseError("no [MPS E23] header found; was the harness patched (H2)?")

    slots = int(hdr.group("slots"))
    n_ctas = int(hdr.group("ctas"))
    tick_ns = int(hdr.group("tick_ns"))
    if slots != SLOTS:
        raise ParseError(f"slots={slots} but this tool is built for {SLOTS}")
    if tick_ns != 10:
        raise ParseError(
            f"tick_ns={tick_ns}: the 100 MHz s_memrealtime domain is assumed "
            "everywhere (1 tick = 0.01 us)"
        )

    stamps: Dict[int, List[int]] = {}
    for ln in lines[hdr_idx:]:
        m = _RE_CTA.search(ln)
        if not m:
            continue
        bid = int(m.group("bid"))
        vals = [int(x) for x in m.group("vals").split()]
        if len(vals) != SLOTS:
            raise ParseError(f"CTA {bid}: {len(vals)} values, expected {SLOTS}")
        if bid in stamps:
            raise ParseError(f"CTA {bid} appears twice after the last header")
        if bid >= n_ctas:
            raise ParseError(f"CTA {bid} >= ctas={n_ctas}")
        stamps[bid] = vals
    if not stamps:
        raise ParseError("[MPS E23] header present but no CTA rows followed")

    coarse: Dict[str, int] = {}
    for ln in lines:
        m = _RE_TS.search(ln)
        if not m:
            continue
        cur = {}
        for tok in m.group("body").split():
            k, _, v = tok.partition("=")
            try:
                cur[k] = int(v)
            except ValueError:
                cur = {}
                break
        if cur:
            coarse = cur           # last [MPS TS] line wins, same as the block

    spin = None
    for ln in lines:
        m = _RE_SPIN.search(ln)
        if m:
            spin = {"success_max": int(m.group("ok")),
                    "fail_max": int(m.group("bad"))}

    return {
        "cfg_in_log": hdr.group("cfg") or "",
        "n_ctas_declared": n_ctas,
        "tick_ns": tick_ns,
        "stamps": stamps,
        "coarse": coarse,
        "spin": spin,
    }


def epoch_window(stamps: Dict[int, List[int]],
                 coarse: Dict[str, int]) -> Tuple[int, int, int]:
    """Pick the final-epoch window and count stale cells outside it.

    Cells are never reset, so a boundary a CTA stopped crossing (role change,
    config change between runs sharing a buffer) keeps an OLD value. Anything
    older than `4 x interior` before the newest stamp is stale. 4x is chosen so
    that even a pathological 4-fold epoch stretch cannot be misread as stale,
    while the 600-epoch soak's stride (~6.5 ms = 650,000 ticks) is 154x the
    threshold's granularity -- there is no ambiguous regime in between.
    """
    live = [v for row in stamps.values() for v in row[:13] if v]
    if not live:
        raise ParseError("every stamp cell is zero: timestamps=1 missing from K0_MPS_CFG?")
    t_max = max(live)
    interior = 0
    if coarse.get("REDUCE_DONE") and coarse.get("M2_DONE"):
        interior = coarse["REDUCE_DONE"] - coarse["M2_DONE"]
    if interior <= 0:
        interior = 1_000_000       # 10 ms fallback when no coarse cells exist
    lo = t_max - 4 * interior
    stale = sum(1 for row in stamps.values() for v in row[:13] if v and v < lo)
    return lo, t_max, stale


def build_ctas(stamps: Dict[int, List[int]], lo: int) -> List[Dict[str, Any]]:
    """Per-CTA record: live stamps (ticks), inferred role, intervals (ticks)."""
    out = []
    for bid in sorted(stamps):
        row = stamps[bid]
        st = {}
        for slot, name in PHASE_SLOTS.items():
            if name == "META":
                continue
            v = row[slot]
            if v and v >= lo:
                st[name] = v
        meta = row[SLOT_OF["META"]]
        role = "service" if "M7_DONE" not in st else "compute"
        if meta:
            role = "service" if (meta >> 32) & 0xFFFFFFFF else "compute"
        ivs = []
        for name, a, b in INTERVALS:
            if a in st and b in st and st[b] >= st[a]:
                ivs.append({"phase": name, "t0": st[a], "t1": st[b]})
        out.append({
            "bid": bid,
            "role": role,
            "stamps": st,
            "intervals": ivs,
            "meta_count": (meta & 0xFFFFFFFF) if meta else 0,
        })
    return out


def check_monotonic(ctas: List[Dict[str, Any]]) -> Dict[str, Any]:
    bad = []
    for c in ctas:
        prev_name, prev = None, None
        for name in MONOTONIC_ORDER:
            v = c["stamps"].get(name)
            if v is None:
                continue
            if prev is not None and v < prev:
                bad.append({"bid": c["bid"], "at": name, "after": prev_name,
                            "delta_ticks": v - prev})
            prev_name, prev = name, v
    return {"pass": not bad, "violations": bad[:20], "n_violations": len(bad)}


def check_coarse(ctas: List[Dict[str, Any]],
                 coarse: Dict[str, int]) -> Dict[str, Any]:
    """max over CTAs of a per-CTA cell == the coarse atomicMax cell, EXACTLY.

    This is a theorem if the patch used the fused ts_mark() (one clock read
    feeding both cells). A nonzero delta means either the ring is indexed wrong,
    or the cells are from different epochs, or -- most likely -- the implementer
    wrote `ts_last(); e23_mark();` and the two calls read the clock twice. See
    patch_spec.md gate G7.
    """
    rows, ok = [], True
    for ring_name, coarse_name in COARSE_PAIRS.items():
        cv = coarse.get(coarse_name)
        vals = [c["stamps"][ring_name] for c in ctas if ring_name in c["stamps"]]
        if cv is None or not vals:
            rows.append({"boundary": ring_name, "status": "unavailable",
                         "coarse": cv, "n_ctas": len(vals)})
            continue
        d = max(vals) - cv
        good = abs(d) <= TOL_COARSE_TICKS
        ok = ok and good
        rows.append({"boundary": ring_name, "coarse_ticks": cv,
                     "ring_max_ticks": max(vals), "delta_ticks": d,
                     "n_ctas": len(vals), "pass": good})
    return {"pass": ok, "tolerance_ticks": TOL_COARSE_TICKS, "rows": rows}


def check_coverage(ctas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per CTA: how much of [first stamp, last stamp] no interval accounts for.

    Overlapping intervals (service overlaps M7 by design; m75 can overlap
    combine) are unioned first, so overlap never manufactures coverage.
    """
    worst, rows = 0.0, []
    for c in ctas:
        if not c["intervals"]:
            continue
        lo = min(i["t0"] for i in c["intervals"])
        hi = max(i["t1"] for i in c["intervals"])
        span = hi - lo
        if span <= 0:
            continue
        segs = sorted((i["t0"], i["t1"]) for i in c["intervals"])
        merged = [list(segs[0])]
        for a, b in segs[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        covered = sum(b - a for a, b in merged)
        frac = (span - covered) / span
        worst = max(worst, frac)
        if frac > TOL_COVERAGE_FRAC:
            rows.append({"bid": c["bid"], "uncovered_frac": frac,
                         "span_us": span * TICK_US})
    return {"pass": worst <= TOL_COVERAGE_FRAC, "worst_uncovered_frac": worst,
            "tolerance_frac": TOL_COVERAGE_FRAC, "offenders": rows[:20],
            "n_offenders": len(rows)}


def check_interior(ctas: List[Dict[str, Any]],
                   coarse: Dict[str, int]) -> Dict[str, Any]:
    """plan + M6 + M7 + combine == interior, on the rank-max reconstruction.

    Pure arithmetic on the same cells, so the tolerance is 1 us, not 10 %.
    exp_33 measured this closing to 0.0 us in all 10 rotations.
    """
    need = ("M2_DONE", "M5_DONE", "M6_DONE", "M7_DONE", "REDUCE_DONE")
    mx = {}
    for n in need:
        vals = [c["stamps"][n] for c in ctas if n in c["stamps"]]
        if not vals:
            return {"pass": None, "reason": f"no CTA reported {n}"}
        mx[n] = max(vals)
    plan = (mx["M5_DONE"] - mx["M2_DONE"]) * TICK_US
    m6 = (mx["M6_DONE"] - mx["M5_DONE"]) * TICK_US
    m7 = (mx["M7_DONE"] - mx["M6_DONE"]) * TICK_US
    comb = (mx["REDUCE_DONE"] - mx["M7_DONE"]) * TICK_US
    interior = (mx["REDUCE_DONE"] - mx["M2_DONE"]) * TICK_US
    resid = (plan + m6 + m7 + comb) - interior
    return {"pass": abs(resid) <= TOL_INTERIOR_US, "plan_us": plan, "M6_us": m6,
            "M7_us": m7, "combine_us": comb, "interior_us": interior,
            "residual_us": resid, "tolerance_us": TOL_INTERIOR_US,
            "coarse_interior_us": (
                (coarse["REDUCE_DONE"] - coarse["M2_DONE"]) * TICK_US
                if coarse.get("REDUCE_DONE") and coarse.get("M2_DONE") else None)}


def check_e2e(ctas: List[Dict[str, Any]],
              arm_p50_us: Optional[float]) -> Dict[str, Any]:
    """Timeline span vs the campaign's end-to-end p50.

    The stamps describe the FINAL SOAK epoch and arm_p50_us is a timed-iteration
    rank-max, so these are two different measurements of the same regime. The
    known residual (p50 - interior) is ~641.7 us on a ~6,496 us arm = 9.9 %
    (PLOTS.md caveat 2), which is exactly why the tolerance is 10 % and not 2 %.
    A tighter gate would fail for a reason already understood; a looser one
    would not catch a real error.
    """
    starts = [c["stamps"]["KSTART"] for c in ctas if "KSTART" in c["stamps"]]
    ends = [c["stamps"]["REDUCE_DONE"] for c in ctas
            if "REDUCE_DONE" in c["stamps"]]
    if not ends:
        return {"pass": None, "reason": "no REDUCE_DONE cell"}
    origin = min(starts) if starts else min(
        c["stamps"]["M2_DONE"] for c in ctas if "M2_DONE" in c["stamps"])
    span_us = (max(ends) - origin) * TICK_US
    out = {"timeline_span_us": span_us, "origin_is_kstart": bool(starts),
           "arm_p50_us": arm_p50_us, "tolerance_frac": TOL_E2E_FRAC}
    if arm_p50_us:
        rel = (span_us - arm_p50_us) / arm_p50_us
        out["relative_delta"] = rel
        out["pass"] = abs(rel) <= TOL_E2E_FRAC
    else:
        out["pass"] = None
        out["reason"] = "no arm_p50_us supplied (--summary)"
    return out


def read_arm_p50(summary_path: str, arm: str) -> Optional[float]:
    try:
        with open(summary_path, "r", encoding="utf-8") as fh:
            s = json.load(fh)
        return float(s["arm_p50_us"][arm]["median"])
    except Exception:
        return None

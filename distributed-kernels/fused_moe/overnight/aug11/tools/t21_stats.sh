#!/usr/bin/env bash
# t21: per-mask device-stamp statistics + Welch t against the mask-0 control.
# Reads every screen_<tag>.csv named on the command line (tag order = report
# order); the FIRST tag is the control every other arm is tested against.
set -uo pipefail
cd "$HOME/overnight-scratch" || exit 1
python3 - "$@" <<'PY'
import csv, math, sys

def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0 / d; h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1.0) < EPS: break
    return h

def betainc(a, b, x):
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log(1.0 - x) + a * math.log(x)) * betacf(b, a, 1.0 - x) / b

def t_two_sided_p(t, df):
    if df <= 0: return float("nan")
    return betainc(df / 2.0, 0.5, df / (df + t * t))

def stats(v):
    n = len(v)
    if n == 0: return (0, float("nan"), float("nan"))
    m = sum(v) / n
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return (n, m, s)

def welch(a, b):
    n1, m1, s1 = stats(a); n2, m2, s2 = stats(b)
    if n1 < 2 or n2 < 2: return (float("nan"),) * 4
    se2 = s1 * s1 / n1 + s2 * s2 / n2
    if se2 <= 0: return (float("nan"),) * 4
    t = (m1 - m2) / math.sqrt(se2)
    num = se2 * se2
    den = (s1 ** 4) / (n1 * n1 * (n1 - 1)) + (s2 ** 4) / (n2 * n2 * (n2 - 1))
    df = num / den if den > 0 else float("nan")
    return (m1 - m2, t, df, t_two_sided_p(t, df))

COLS = ["ts_M6_us", "ts_M7_us", "ts_combine_us", "ts_planM3toM5_us",
        "ts_servicedrain_us", "ts_m2_to_end_us", "mps_us", "ratio_vs_prod"]
tags = sys.argv[1:]
data, meta = {}, {}
for tag in tags:
    rows, bad = [], []
    with open(f"screen_{tag}.csv") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "OK":
                bad.append(r["status"]); continue
            rows.append(r)
    data[tag] = rows
    meta[tag] = (len(rows), bad, sorted({r["head"] for r in rows}),
                 sorted({r["mps_hsaco"] for r in rows}))

print("=" * 96)
for tag in tags:
    n, bad, heads, hs = meta[tag]
    print(f"{tag:10s} n_OK={n:2d}  non-OK={bad if bad else '[]'}  head={heads}  hsaco_dirs={hs}")
print("=" * 96)

ctl = tags[0]
for col in COLS:
    print(f"\n--- {col} " + "-" * (76 - len(col)))
    base = [float(r[col]) for r in data[ctl] if r[col] not in ("", None)]
    nb, mb, sb = stats(base)
    print(f"{'arm':10s} {'n':>2s} {'mean':>10s} {'sd':>7s} {'delta':>9s} "
          f"{'t':>8s} {'df':>6s} {'p2':>10s}")
    for tag in tags:
        v = [float(r[col]) for r in data[tag] if r[col] not in ("", None)]
        n, m, s = stats(v)
        if tag == ctl:
            print(f"{tag:10s} {n:2d} {m:10.2f} {s:7.2f} {'--':>9s} {'--':>8s} {'--':>6s} {'--':>10s}")
        else:
            d, t, df, p = welch(v, base)
            print(f"{tag:10s} {n:2d} {m:10.2f} {s:7.2f} {d:+9.2f} {t:8.2f} {df:6.1f} {p:10.2e}")
    if not math.isnan(sb) and sb > 0:
        print(f"  control sd = {sb:.2f}; a 3-sigma effect on the control's own sd is {3*sb:.1f}")
PY
exit 0

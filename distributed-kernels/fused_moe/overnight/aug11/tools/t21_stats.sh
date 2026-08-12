#!/usr/bin/env bash
# t21: per-mask device-stamp statistics + Welch t against the mask-0 control.
# Reads every screen_<tag>.csv named on the command line (tag order = report
# order); the FIRST tag is the control every other arm is tested against.
set -uo pipefail
cd "$HOME/overnight-scratch" || exit 1
# Control FIRST. Edit as batches land (nsh.ps1 forwards no arguments).
TAGS="e26_m0a e26_m4 e26_m1 e26_m1b e26_m5 e26_m0b e26_m4b e26_m5b"
python3 - $TAGS <<'PY'
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
present = []
for tag in tags:
    rows, bad = [], []
    try:
        fh = open(f"screen_{tag}.csv")
    except FileNotFoundError:
        print(f"(skipping {tag}: no CSV yet)")
        continue
    present.append(tag)
    with fh:
        for r in csv.DictReader(fh):
            if r["status"] != "OK":
                bad.append(r["status"]); continue
            rows.append(r)
    data[tag] = rows
    meta[tag] = (len(rows), bad, sorted({r["head"] for r in rows}),
                 sorted({r["mps_hsaco"] for r in rows}))
tags = present

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

# ---------------------------------------------------------------- 2x2 factorial
# The ladder IS a factorial in (DSR bit 0, MFMA bit 2), so analyse it as one.
# Cells pool every batch of the same mask; batch means are printed so the
# replication is visible rather than hidden inside a pooled sd.
CELL = {0: ["e26_m0a", "e26_m0b"], 1: ["e26_m1", "e26_m1b"],
        4: ["e26_m4", "e26_m4b"], 5: ["e26_m5", "e26_m5b"]}
print("\n" + "=" * 96)
print("2x2 FACTORIAL on ts_M6_us   bits: DSR = mask&1, MFMA = mask&4")
print("=" * 96)
cell = {}
for mask, tg in CELL.items():
    vals, per_batch = [], []
    for t in tg:
        if t not in data: continue
        v = [float(r["ts_M6_us"]) for r in data[t] if r["ts_M6_us"] not in ("", None)]
        if v:
            vals += v
            per_batch.append((t, len(v), sum(v) / len(v)))
    if not vals: continue
    n, m, s = stats(vals)
    cell[mask] = (n, m, s)
    bs = "  ".join(f"{t}={mv:.2f}(n{bn})" for t, bn, mv in per_batch)
    print(f"mask {mask}: DSR={mask & 1} MFMA={(mask & 4) >> 2}  "
          f"n={n:2d} mean={m:8.2f} sd={s:6.2f}   batches: {bs}")

CELLVALS = {}
for mask, tg in CELL.items():
    v = [float(r["ts_M6_us"]) for t in tg if t in data
         for r in data[t] if r["ts_M6_us"] not in ("", None)]
    if v: CELLVALS[mask] = v
if 0 in CELLVALS:
    print("\nper-mask vs the POOLED mask-0 control (both cells, 2 batches, n=10):")
    for mask in sorted(CELLVALS):
        if mask == 0: continue
        d, t, df, p = welch(CELLVALS[mask], CELLVALS[0])
        print(f"  mask {mask}: delta {d:+7.2f} us  t {t:+6.2f}  df {df:5.1f}  p2 {p:9.2e}")

def contrast(coefs):
    """coefs: {mask: weight}. Returns (estimate, se, t, df_min)."""
    est = sum(w * cell[k][1] for k, w in coefs.items())
    var = sum(w * w * cell[k][2] ** 2 / cell[k][0] for k, w in coefs.items())
    se = math.sqrt(var)
    dfs = [cell[k][0] - 1 for k in coefs]
    return est, se, (est / se if se > 0 else float("nan")), min(dfs)

if set(cell) >= {0, 1, 4, 5}:
    print()
    for name, coefs in (
        ("MFMA bit main effect ", {4: 0.5, 5: 0.5, 0: -0.5, 1: -0.5}),
        ("DS-read bit main eff ", {1: 0.5, 5: 0.5, 0: -0.5, 4: -0.5}),
        ("interaction          ", {5: 0.5, 0: 0.5, 1: -0.5, 4: -0.5}),
    ):
        e, se, t, df = contrast(coefs)
        print(f"{name} = {e:+8.2f} us  SE {se:5.2f}  t {t:+6.2f}  "
              f"p2 {t_two_sided_p(t, df):8.2e}  (df>={df})")
    print()
    on = [float(r["ts_M6_us"]) for t in CELL[4] + CELL[5] if t in data
          for r in data[t] if r["ts_M6_us"] not in ("", None)]
    off = [float(r["ts_M6_us"]) for t in CELL[0] + CELL[1] if t in data
           for r in data[t] if r["ts_M6_us"] not in ("", None)]
    d, t, df, p = welch(on, off)
    n1, m1, s1 = stats(on); n0, m0, s0 = stats(off)
    print(f"MFMA on  n={n1:2d} mean={m1:8.2f} sd={s1:6.2f}")
    print(f"MFMA off n={n0:2d} mean={m0:8.2f} sd={s0:6.2f}")
    print(f"pooled Welch on-vs-off: {d:+.2f} us  t={t:+.2f}  df={df:.1f}  p2={p:.2e}")
PY
exit 0

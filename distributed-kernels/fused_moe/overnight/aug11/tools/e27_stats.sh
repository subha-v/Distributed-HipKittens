#!/usr/bin/env bash
# exp_27 final analysis. Two questions, kept separate:
#   A. did the M6 device stamp move, and by how much, with BATCH as a block
#   B. is the output numerically unchanged, at full float precision
#
# On (A): the M6 stamp drifts between batches on a provably identical binary
# (exp_26's control read 2581.7 at 06:35Z; batch 1, whose .text is byte-identical
# to the ratchet, read 2531.6 at 07:07Z -- 50 us, ~6 within-batch sigma). So the
# design is control, candidate, control, candidate and the drift between the two
# CONTROL batches is reported as the yardstick for how much of any candidate delta
# could be time rather than arm.
#
# On (B): the harness cannot test bit-identity of `out` directly, because `out` is
# not bit-reproducible in EITHER arm -- mode 12's combine is a nondeterministic
# bf16 remote-atomic accumulation and the MoK dispatch assigns receive rows by
# fetch_add. So the test is the same-run paired d = mps - pf6gm (pf6gm being an
# untouched arm on the same input), clustered by RUN, plus a check of whether any
# residual shift tracks the M6 TIMING change -- which is the signature of a
# changed atomic order rather than of changed arithmetic.
set -uo pipefail
python3 - <<'PY'
import csv, glob, json, os, statistics as st

S = os.path.expanduser("~/overnight-scratch")
ARM  = {"e27b1_ctl": 0, "e27b2_cand": 1, "e27b3_ctl": 0, "e27b4_cand": 1}
ORDER = ["e27b1_ctl", "e27b2_cand", "e27b3_ctl", "e27b4_cand"]

def welch(a, b):
    if len(a) < 2 or len(b) < 2: return float("nan"), float("nan")
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    se = (va + vb) ** 0.5
    return mb - ma, (mb - ma) / se if se else float("nan")

# ---------------------------------------------------------------- A. the stamp
csvrows = {}
for tag in ORDER:
    p = f"{S}/screen_{tag}.csv"
    if not os.path.exists(p): continue
    csvrows[tag] = [r for r in csv.DictReader(open(p)) if r.get("status") == "OK"]

print("=" * 100)
print("A. THE M6 DEVICE STAMP  (ts_M6_us; 1 tick = 0.01 us)")
print("=" * 100)
print(f"  {'batch':13s} {'arm':>3s} {'n':>3s} {'mean':>9s} {'sd':>7s} {'sem':>6s}   values")
M6 = {}
for tag in ORDER:
    rs = csvrows.get(tag, [])
    v = [float(r["ts_M6_us"]) for r in rs if r.get("ts_M6_us")]
    if not v: continue
    M6[tag] = v
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f"  {tag:13s} {ARM[tag]:3d} {len(v):3d} {st.mean(v):9.2f} {sd:7.2f} "
          f"{sd/len(v)**0.5:6.2f}   {['%.1f'%x for x in v]}")

ctl = [x for t in ORDER if ARM[t] == 0 and t in M6 for x in M6[t]]
cnd = [x for t in ORDER if ARM[t] == 1 and t in M6 for x in M6[t]]
if ctl and cnd:
    print()
    print(f"  pooled control   n={len(ctl):2d}  mean={st.mean(ctl):8.2f}  sd={st.stdev(ctl):6.2f}")
    print(f"  pooled candidate n={len(cnd):2d}  mean={st.mean(cnd):8.2f}  sd={st.stdev(cnd):6.2f}")
    d, t = welch(ctl, cnd)
    print(f"  ==> delta M6 = {d:+.1f} us   (Welch t = {t:+.2f}, runs as units)")
    if "e27b1_ctl" in M6 and "e27b3_ctl" in M6:
        dd, dt = welch(M6["e27b1_ctl"], M6["e27b3_ctl"])
        print(f"  DRIFT YARDSTICK: control batch 1 -> control batch 3 = {dd:+.1f} us (t = {dt:+.2f})")
        print(f"     i.e. the same binary, 8 minutes apart. |arm effect| / |drift| = "
              f"{abs(d)/abs(dd):.1f}x" if dd else "")
    if "e27b2_cand" in M6 and "e27b4_cand" in M6:
        dd2, dt2 = welch(M6["e27b2_cand"], M6["e27b4_cand"])
        print(f"  candidate batch 2 -> batch 4 = {dd2:+.1f} us (t = {dt2:+.2f}) -- the candidate's own reproducibility")
    print()
    print("  per-batch-mean test (BATCH as the unit, 2 vs 2 -- the conservative reading):")
    bc = [st.mean(M6[t]) for t in ORDER if ARM[t] == 0 and t in M6]
    bd = [st.mean(M6[t]) for t in ORDER if ARM[t] == 1 and t in M6]
    print(f"    control batch means   = {['%.1f'%x for x in bc]}")
    print(f"    candidate batch means = {['%.1f'%x for x in bd]}")
    if len(bc) >= 2 and len(bd) >= 2:
        d2, t2 = welch(bc, bd)
        print(f"    ==> delta = {d2:+.1f} us  (t = {t2:+.2f} on 2 vs 2)")
    print()
    print("  OTHER STAMPS (same pooling), for attribution:")
    for k, lbl in (("ts_planM3toM5_us", "plan M3-M5"), ("ts_M7_us", "M7"),
                   ("ts_combine_us", "combine"), ("ts_m2_to_end_us", "M2->end"),
                   ("mps_us", "end-to-end mps"), ("prod_us", "end-to-end production"),
                   ("ratio_vs_prod", "ratio vs prod")):
        a = [float(r[k]) for t in ORDER if ARM[t] == 0 for r in csvrows.get(t, []) if r.get(k)]
        b = [float(r[k]) for t in ORDER if ARM[t] == 1 for r in csvrows.get(t, []) if r.get(k)]
        if len(a) >= 2 and len(b) >= 2:
            d3, t3 = welch(a, b)
            print(f"    {lbl:22s} ctl={st.mean(a):9.2f} cand={st.mean(b):9.2f}  "
                  f"delta={d3:+9.2f}  t={t3:+6.2f}")

# ------------------------------------------------------- B. the numerics, clustered
print()
print("=" * 100)
print("B. NUMERICS AT FULL PRECISION, clustered by RUN (8 ranks in a run share one input)")
print("=" * 100)
runs = {}   # (tag, outdir) -> list of per-rank dicts
for tag in ORDER:
    for rj in sorted(glob.glob(os.path.expanduser(f"~/k0-mok-{tag}/*/run*/k0pf_mok_synthetic_rank*.json"))):
        try: d = json.load(open(rj))
        except Exception: continue
        outdir = os.path.basename(os.path.dirname(os.path.dirname(rj)))
        mc = d.get("mok_correctness", {})
        m = mc.get("mps_mega", {}).get("relative_error")
        p = mc.get("pf6gm_mega", {}).get("relative_error")
        pr = mc.get("production", {}).get("relative_error")
        ab = d.get("eager", {}).get("mps_mega", {}).get("max_abs")
        pb = d.get("eager", {}).get("pf6gm_mega", {}).get("max_abs")
        if m is None or p is None: continue
        runs.setdefault((tag, outdir), []).append(dict(m=m, p=p, pr=pr, ab=ab, pb=pb,
                                                       nf=mc.get("mps_mega", {}).get("nonfinite")))

per_run = []
for (tag, outdir), cells in sorted(runs.items()):
    per_run.append(dict(tag=tag, arm=ARM[tag], outdir=outdir, ncell=len(cells),
                        d=st.mean([c["m"] - c["p"] for c in cells]),
                        m=st.mean([c["m"] for c in cells]),
                        pr=st.mean([c["pr"] for c in cells if c["pr"] is not None]),
                        abeq=all(c["ab"] == c["pb"] for c in cells),
                        nf=sum(c["nf"] or 0 for c in cells)))

print(f"  {'batch':13s} {'arm':>3s} {'cells':>5s} {'mean d = mps-pf6':>18s} {'mps rel_err':>16s} "
      f"{'max_abs==pf6':>13s} {'nonfin':>7s}")
for r in per_run:
    print(f"  {r['tag']:13s} {r['arm']:3d} {r['ncell']:5d} {r['d']:+18.4e} {r['m']:16.12f} "
          f"{str(r['abeq']):>13s} {r['nf']:7d}")

d0 = [r["d"] for r in per_run if r["arm"] == 0]
d1 = [r["d"] for r in per_run if r["arm"] == 1]
if len(d0) >= 2 and len(d1) >= 2:
    dd, tt = welch(d0, d1)
    print()
    print(f"  d, control   n={len(d0)} runs  mean={st.mean(d0):+.4e}  sd={st.stdev(d0):.4e}")
    print(f"  d, candidate n={len(d1)} runs  mean={st.mean(d1):+.4e}  sd={st.stdev(d1):.4e}")
    print(f"  ==> shift in d = {dd:+.4e}  (t = {tt:+.2f}, RUNS as units)")
    prall = [r["pr"] for r in per_run]
    print(f"  untouched `production` rel_err across the same runs: sd = {st.stdev(prall):.4e}")
    print(f"      -> the shift in d is {abs(dd)/st.stdev(prall)*100:.3f}% of the run-to-run input drift")
    nall = sum(r["nf"] for r in per_run)
    ab = sum(1 for r in per_run if r["abeq"])
    print(f"  max_abs(mps) == max_abs(pf6gm) in {ab}/{len(per_run)} runs (all ranks);  total nonfinite = {nall}")

    # Is the residual shift a TIMING artifact? mode 12 accumulates with
    # nondeterministically-ordered bf16 remote atomics, so a faster M6 reorders
    # them and moves the last bits WITHOUT changing any input to the arithmetic.
    print()
    print("  IS THE RESIDUAL A CHANGED ATOMIC ORDER RATHER THAN CHANGED ARITHMETIC?")
    join = {}
    for tag in ORDER:
        for r in csvrows.get(tag, []):
            if r.get("outdir") and r.get("ts_M6_us"):
                join[(tag, r["outdir"])] = float(r["ts_M6_us"])
    pts = [(join[(r["tag"], r["outdir"])], r["d"], r["arm"])
           for r in per_run if (r["tag"], r["outdir"]) in join]
    if len(pts) >= 6:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        mx, my = st.mean(xs), st.mean(ys)
        sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        sxx = sum((x-mx)**2 for x in xs); syy = sum((y-my)**2 for y in ys)
        rho = sxy/(sxx*syy)**0.5 if sxx and syy else float("nan")
        print(f"    corr(ts_M6_us, d) across all {len(pts)} runs, both arms = {rho:+.3f}")
        for a in (0, 1):
            sub = [(x, y) for x, y, ar in pts if ar == a]
            if len(sub) >= 3:
                xs2 = [p[0] for p in sub]; ys2 = [p[1] for p in sub]
                mx2, my2 = st.mean(xs2), st.mean(ys2)
                sxy2 = sum((x-mx2)*(y-my2) for x, y in zip(xs2, ys2))
                sxx2 = sum((x-mx2)**2 for x in xs2); syy2 = sum((y-my2)**2 for y in ys2)
                r2 = sxy2/(sxx2*syy2)**0.5 if sxx2 and syy2 else float("nan")
                print(f"    corr within arm {a} (n={len(sub)}) = {r2:+.3f}   "
                      f"(a strong ACROSS-arm corr with a weak WITHIN-arm corr means the")
                print(f"      shift rides on the timing change, not on the layout)")
PY

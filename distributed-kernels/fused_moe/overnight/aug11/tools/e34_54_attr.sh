#!/usr/bin/env bash
# exp_34: attribute the mode-12 mps_mega regression (6,48x -> 7,22x) to a src_rev.
# EVERY screen row on the node, any tag, any cfg: print rev, cfg, the three arm
# p50s. CPU-only; safe to run while a campaign holds the GPUs.
set -uo pipefail
python3 - <<'PY'
import glob, os, re
HOME = os.path.expanduser("~")
files = set()
for pat in ("overnight-scratch/*.log", "overnight-scratch/*.csv", "*.driver.log",
            "e3*/*.log", "e3*/*.csv", "*.csv", "*.log"):
    files.update(glob.glob(os.path.join(HOME, pat)))
rows = {}
for f in sorted(files):
    try:
        t = open(f, errors="ignore").read()
    except Exception:
        continue
    for line in t.splitlines():
        if not line.startswith("SCREEN ") or '"' not in line:
            continue
        cfg = line.split('"')[1]
        rest = line.split('"')[2].lstrip(",").split(",")
        if len(rest) < 7:
            continue
        status, iters, head, rev = rest[0], rest[1], rest[2], rest[3]
        try:
            prod, pf6, mps = float(rest[4]), float(rest[5]), float(rest[6])
        except ValueError:
            continue
        rows[(os.path.basename(f), line[:40], cfg)] = (
            rev, head, iters, status, cfg, prod, pf6, mps, os.path.basename(f))
by_rev = {}
for v in rows.values():
    by_rev.setdefault((v[0], v[2]), []).append(v)
print(f"{'rev':>4s} {'iters':>10s} {'n':>3s}  {'cfg':46s} {'prod':>8s} {'pf6gm':>8s} {'mps':>8s}")
for (rev, iters), vs in sorted(by_rev.items()):
    groups = {}
    for v in vs:
        groups.setdefault(v[4], []).append(v)
    for cfg, g in sorted(groups.items()):
        import statistics as st
        print(f"{rev:>4s} {iters:>10s} {len(g):3d}  {cfg:46s} "
              f"{st.median([x[5] for x in g]):8.1f} {st.median([x[6] for x in g]):8.1f} "
              f"{st.median([x[7] for x in g]):8.1f}   {sorted({x[8] for x in g})[0]}")
PY
echo
echo "===commit dates for the revs in play==="
for c in 123b9e31 6651c4d4 f113d73f d3d22ce4 291dfa08; do
  git -C "$HOME/Distributed-HipKittens" log -1 --format="%h %ad %s" --date=iso "$c" 2>/dev/null | cut -c1-110
done
echo
echo "===is K0_MOK_POISON_OUT on by default, and what rev added it==="
git -C "$HOME/Distributed-HipKittens" log --oneline -6 291dfa08 | cut -c1-110
echo "===DONE==="

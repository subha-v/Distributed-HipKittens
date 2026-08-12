#!/usr/bin/env bash
# Read the shape-3 recheck: did the synchronized tail reproduce in a fresh pool?
set -u
E=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/experiments/exp_10_rank1
echo "alive: $(pgrep -c -f 'mp_vs_rank1' || true)"
echo "===== recheck3.log ====="
cat "$E/recheck3.log" 2>&1 | tail -25
echo
echo "===== recheck3 tail analysis ====="
python3 - "$E/vs_logs_recheck3" <<'PY'
import glob, json, os, statistics, sys
root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "vs_s2.rank*.json")))
if not files:
    print("no recheck json yet"); raise SystemExit(0)
pool = {"ours": [], "rank1": []}
print(f"{'rank':>5} {'med':>9} {'mean':>9} {'outliers >2x med (index, us)'}")
for p in files:
    rk = int(os.path.basename(p).split(".")[1].replace("rank", ""))
    d = json.load(open(p))
    if not d:
        continue
    for a in pool:
        pool[a] += d["arms"].get(a, [])
    xs = d["arms"]["ours"]
    med = statistics.median(xs)
    bad = [(i, round(x, 1)) for i, x in enumerate(xs) if x > 2 * med]
    print(f"{rk:>5} {med:9.2f} {statistics.mean(xs):9.2f}  {bad}")
print()
for a in ("ours", "rank1"):
    xs = pool[a]
    if not xs:
        continue
    med = statistics.median(xs)
    print(f"{a:>6}: best={min(xs):8.2f} med={med:8.2f} mean={statistics.mean(xs):8.2f} "
          f"sd%={statistics.stdev(xs)/statistics.mean(xs)*100:5.1f} max={max(xs):8.2f} "
          f"n={len(xs)}  >2x med: {sum(1 for x in xs if x > 2*med)}")
if pool["ours"] and pool["rank1"]:
    print(f"  ratio best={min(pool['ours'])/min(pool['rank1']):.3f}x "
          f"med={statistics.median(pool['ours'])/statistics.median(pool['rank1']):.3f}x "
          f"mean={statistics.mean(pool['ours'])/statistics.mean(pool['rank1']):.3f}x")
print()
print("first run for comparison: ours best=194.31 med=218.46 mean=327.77 sd=75.2%")
print("                          rank1 best=157.61 med=162.76 mean=163.47 sd=2.5%")
PY
echo "===== DONE ====="

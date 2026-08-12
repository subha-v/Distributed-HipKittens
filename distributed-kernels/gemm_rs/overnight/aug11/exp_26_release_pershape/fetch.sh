#!/usr/bin/env bash
# Pull the numbers that matter out of the node-side artefacts, in a form that
# can be pasted into result.md. Read-only; no GPU.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
LADDER=$ON/experiments/exp_26_release_pershape/logs

echo "########## gate ladder verdicts ##########"
grep -E 'ALL MODULES BUILT|M2 metadata rows|shapes PASSED|GATE M4|GATE M5|FAILED AT|GATE LADDER COMPLETE' \
  "$D/logs/ladder.log" 2>/dev/null

echo
echo "########## M2 resource tuples (candidate build) ##########"
sed -n '/metadata table/,/ordering ops/p' "$LADDER/m2_report.log" 2>/dev/null \
  | grep -E 'instantiation|BM=' | head -12

echo
echo "########## M7, per-rotation samples -> best / median / mean ##########"
python3 - "$LADDER/m7_results.json" <<'PY'
import json, statistics, sys
d = json.load(open(sys.argv[1]))
names = ["64x7168x18432", "512x4096x12288", "2048x2880x2880",
         "4096x4096x4096", "8192x4096x14336", "8192x8192x29568"]
prev_best = [62.38, 64.52, 83.75, 198.71, 613.70, 1616.63]
floor = [3.48, 0.93, 1.61, 4.90, 13.93, 69.87]
print(f"{'#':>2} {'shape':<18}{'best':>10}{'median':>10}{'mean':>10}"
      f"{'prevbest':>10}{'d_best':>9}{'floor':>8}{'|d|>floor':>10}")
for i, name in enumerate(names):
    s = d["samples_us"][str(i)]
    b, m, mu = min(s), statistics.median(s), statistics.mean(s)
    delta = b - prev_best[i]
    print(f"{i+1:>2} {name:<18}{b:>10.2f}{m:>10.2f}{mu:>10.2f}"
          f"{prev_best[i]:>10.2f}{delta:>+9.2f}{floor[i]:>8.2f}"
          f"{'YES' if abs(delta) > floor[i] else 'no':>10}")
import math
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
best = [min(d["samples_us"][str(i)]) for i in range(6)]
med = [statistics.median(d["samples_us"][str(i)]) for i in range(6)]
print(f"\ngeomean of best   : {gm(best):8.2f} us   (prev best-of-arm "
      f"{gm(prev_best):.2f})")
print(f"geomean of median : {gm(med):8.2f} us")
print(f"geomean of mean   : {d['geomean_us']:8.2f} us")
print(f"all correct       : {d['all_correct']}")
PY

echo
echo "########## M9 (whatever exists) ##########"
for f in m9 m9_shape5 m9_full; do
  [ -f "$D/logs/$f.log" ] || continue
  echo "-- $f --"
  grep -E 'GATE M9|FAIL|detector fired|CTRL_PUBLISH_EARLY|Memory access fault' \
    "$D/logs/$f.log" | head -30
done

echo
echo "########## A/B summaries (whatever exists) ##########"
for f in ab_fwd ab_rev; do
  [ -f "$D/logs/$f.log" ] || continue
  echo "===== $f ====="
  sed -n '/PER-SHAPE SUMMARY/,$p' "$D/logs/$f.log"
done

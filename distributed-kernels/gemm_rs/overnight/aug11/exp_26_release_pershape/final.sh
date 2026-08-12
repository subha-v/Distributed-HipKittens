#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
L=$ON/experiments/exp_26_ps2_land/logs

echo "########## landing ladder verdicts (PERSHAPE=2) ##########"
grep -E 'ALL MODULES BUILT|M2 metadata rows|shapes PASSED|GATE M4|GATE M5|FAILED AT|GATE LADDER COMPLETE' \
  "$D/logs/ladder_ps2.log"

echo
echo "########## M2 resource tuple, shipped build ##########"
sed -n '/metadata table/,/waves\/SIMD/p' "$L/m2_report.log" 2>/dev/null | head -14

echo
echo "########## M9 on the shipped build: per-shape CTRL_PUBLISH_EARLY ##########"
awk '/CTRL_PUBLISH_EARLY: m=/{shape=$0} /detector fired on/{print shape; print "   " $0}' \
  "$D/logs/m9_ps2.log"
echo "-- verdict --"
grep -E 'GATE M9|ok   CTRL_PUBLISH_EARLY|note:' "$D/logs/m9_ps2.log"
echo "-- candidate cases: any failure? --"
grep -c 'FAIL' "$D/logs/m9_ps2.log"

echo
echo "########## M7 of the landing ladder: best / median / mean ##########"
python3 - "$L/m7_results.json" <<'PY'
import json, math, statistics, sys
d = json.load(open(sys.argv[1]))
names = ["64x7168x18432", "512x4096x12288", "2048x2880x2880",
         "4096x4096x4096", "8192x4096x14336", "8192x8192x29568"]
print(f"{'#':>2} {'shape':<18}{'best':>10}{'median':>10}{'mean':>10}")
best = []
for i, n in enumerate(names):
    s = d["samples_us"][str(i)]
    best.append(min(s))
    print(f"{i+1:>2} {n:<18}{min(s):>10.2f}{statistics.median(s):>10.2f}"
          f"{statistics.mean(s):>10.2f}")
gm = math.exp(sum(math.log(x) for x in best) / len(best))
print(f"\ngeomean of best {gm:.2f} us   geomean of mean {d['geomean_us']:.2f} us"
      f"   all correct {d['all_correct']}")
PY

echo
echo "########## shipped source default ##########"
grep -A2 '^#ifndef HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE' \
  /home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp

echo
echo "########## node left clean? ##########"
if [ -d "$ON/aug11/.gpu_lease" ]; then
  echo "lease held by $(cat "$ON/aug11/.gpu_lease/owner" 2>/dev/null)"
else
  echo "lease FREE"
fi
pgrep -af 'exp_26' | head -5 || echo "no exp_26 processes"

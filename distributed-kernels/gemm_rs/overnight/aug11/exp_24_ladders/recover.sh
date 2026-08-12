#!/usr/bin/env bash
# RECOVERY: rebuild the aggregate that tools/push.ps1 clobbered.
#
# push.ps1 scp's the whole overnight tree including *.json, so syncing scripts
# to the node OVERWRITES node-side results with whatever stale copy the Windows
# worktree happens to hold. It replaced the new ladders.json (2 252 816 B) with
# the previous run's (2 244 658 B), and the report text captured right after was
# therefore generated from the wrong file.
#
# Recoverable because the 48 per-rank sample files under raw/ladder/ exist only
# on the node and so were never candidates for the overwrite. Re-aggregate from
# them, re-verify, regenerate the reports. Run via nsh.ps1 ONLY -- nsh copies the
# single script to /tmp itself, so it does not need (and must not trigger) a push.
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders
RL=$D/raw/ladder

echo "=============== the raw samples must be intact ==============="
n=$(ls "$RL"/lad_s*.rank*.json 2>/dev/null | wc -l)
echo "  sample files: $n (need 48)"
[ "$n" -ne 48 ] && { echo "FATAL: raw samples are gone; the run must be repeated"; exit 1; }
bad=0
for f in "$RL"/lad_s*.rank*.json; do
  m=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('rot_mode','ABSENT'))" "$f" 2>/dev/null)
  [ "$m" = "shuffle" ] || { echo "  NOT FROM THIS RUN: $(basename "$f") rot_mode=$m"; bad=$((bad+1)); }
done
echo "  files not from this run: $bad"
[ "$bad" -ne 0 ] && { echo "FATAL"; exit 1; }

echo
echo "=============== instrument B must stay quarantined ==============="
ls -d "$D"/raw/eval_prev_79599cce_DO_NOT_PARSE 2>/dev/null && echo "  quarantine intact"
echo "  raw/eval contents: $(ls "$D"/raw/eval 2>/dev/null | wc -l) entries (want 0)"

echo
echo "=============== re-aggregate from the raw samples ==============="
python3 "$D/ladders.py" --root "$D" --out "$D/ladders.json" --ladder-dir ladder \
  --expect-a 6 --expect-b 0 | tail -18
echo "  size now: $(stat -c%s "$D/ladders.json") B  (expect 2252816)"

echo
echo "=============== regenerate the report artifacts ==============="
python3 "$D/report.py"       > "$D/remeasure_report_ladder.txt" 2>&1
python3 "$D/compare_runs.py" > "$D/remeasure_report_delta.txt"  2>&1
for f in "$RL"/lad_s*.rank0.json; do python3 "$D/show_orders.py" "$f"; done \
  > "$D/remeasure_arm_orders.txt" 2>&1
grep -E 'geomean 1\.|geomean 0\.|GEOMEAN' "$D/remeasure_report_delta.txt" | head -8 | sed 's/^/  /'
for f in ladders.json geometry.json remeasure_delta.json \
         remeasure_report_ladder.txt remeasure_report_delta.txt \
         remeasure_arm_orders.txt; do
  [ -f "$D/$f" ] && echo "  $f  $(stat -c%s "$D/$f") B" || echo "  MISSING $f"
done

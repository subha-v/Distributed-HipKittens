#!/usr/bin/env bash
# exp_24 re-measure, finalize: integrity gate, then the two reports.
#
# The integrity gate exists because LAD_KEEP is not implemented in
# run_ladders.sh -- it never deletes previous samples, it simply overwrites
# lad_s<shape>.rank<r>.json per shape. That is fine when all six shapes succeed
# and silently wrong if one fails, because the stale file from the PREVIOUS
# binary (79599cce) survives and the parser cannot tell. So gate on content, not
# on mtime: every sample file must carry rot_mode == "shuffle", which only this
# run's code writes.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders
RL=$D/raw/ladder

echo "=============== integrity: is every sample file from THIS run? ==============="
n_files=$(ls "$RL"/lad_s*.rank*.json 2>/dev/null | wc -l)
echo "  sample files: $n_files (expect 48 = 6 shapes x 8 ranks)"
bad=0
for f in "$RL"/lad_s*.rank*.json; do
  [ -f "$f" ] || continue
  mode=$(python3 -c "
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(d.get('rot_mode','ABSENT'))
except Exception as e:
    print('UNREADABLE')
" "$f")
  if [ "$mode" != "shuffle" ]; then
    echo "  STALE/BAD $(basename "$f") rot_mode=$mode  mtime=$(date -Is -r "$f")"
    bad=$((bad + 1))
  fi
done
echo "  files not from this run: $bad"
if [ "$bad" -ne 0 ]; then
  echo "  FATAL: refusing to report a ladder that mixes two binaries."
  echo "         Re-run the affected shapes with LAD_SHAPES=<idx>."
  exit 1
fi
echo "  OK: all $n_files sample files carry rot_mode=shuffle"

echo
echo "=============== the arm orders actually walked ==============="
for f in "$RL"/lad_s*.rank0.json; do
  [ -f "$f" ] || continue
  python3 "$D/show_orders.py" "$f"
done

echo
echo "=============== re-aggregate ==============="
python3 "$D/ladders.py" --root "$D" --out "$D/ladders.json" --ladder-dir ladder \
  --expect-a 6 --expect-b 0
echo "  aggregate rc=$?"

echo
echo "=============== report: the ladder itself ==============="
python3 "$D/report.py" 2>&1 | sed 's/^/  /'

echo
echo "=============== report: the paired delta vs the previous run ==============="
python3 "$D/compare_runs.py" 2>&1 | sed 's/^/  /'

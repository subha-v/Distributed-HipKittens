#!/usr/bin/env bash
# Read-only poll of the exp_21 campaign. Touches no GPU.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/aug11/exp_21_saturation
N=${1:-40}

echo "########## lease ##########"
bash "$ON/tools/gpu_lease.sh" status

echo
echo "########## processes ##########"
ps -eo pid,etimes,cmd 2>/dev/null \
  | grep -E 'exp_21_saturation/(logs/run_[^ ]*/)?(campaign|run_sweep)|run_saturation' \
  | grep -v grep || echo "  (none alive)"

echo
echo "########## campaign log ##########"
tail -"$N" "$EXP/logs/campaign_latest.log" 2>/dev/null || echo "  (no campaign log yet)"

echo
echo "########## newest sweep log ##########"
latest=$(ls -t "$EXP"/logs/sweep_*.log 2>/dev/null | head -1)
if [ -n "$latest" ]; then
  echo "-- $latest ($(wc -l <"$latest") lines) --"
  tail -"$N" "$latest"
else
  echo "  (none)"
fi

echo
echo "########## artifacts ##########"
ls -la "$EXP"/*.json "$EXP"/ceilings.txt 2>/dev/null || echo "  (none)"

#!/usr/bin/env bash
# Post-install recheck: which gpu_lease.sh is installed, is the old campaign gone,
# what does the lease log say, and is anyone else patching the same tool.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/aug11/exp_21_saturation

echo "########## installed lease tool ##########"
ls -la --time-style=+%H:%M:%S "$ON"/tools/gpu_lease* | sed 's/^/  /'
grep -c 'zombie_kfd_pids' "$ON/tools/gpu_lease.sh" | sed 's/^/  zombie-aware markers: /'

echo
echo "########## lease ##########"
bash "$ON/tools/gpu_lease.sh" status

echo
echo "########## lease log ##########"
tail -14 "$ON/aug11/gpu_lease.log" 2>/dev/null

echo
echo "########## exp_21 campaign processes ##########"
ps -eo pid,etimes,cmd 2>/dev/null | grep -E 'exp_21_saturation/(campaign|run_sweep)|run_saturation' \
  | grep -v grep || echo "  (none alive)"

echo
echo "########## campaign log ##########"
tail -12 "$EXP/logs/campaign_latest.log" 2>/dev/null

echo
echo "########## any other agent's GPU driver alive ##########"
ps -eo pid,etimes,cmd 2>/dev/null \
  | grep -E 'sweep\.py|ladder|m7_bench|exp_ablation|mp_smoke|eval\.py|m9_|m5_soak|m3_correct' \
  | grep -v grep || echo "  (none)"

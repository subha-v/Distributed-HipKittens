#!/usr/bin/env bash
# Which amd-smi SECTION reports MAX_BANDWIDTH 5325 GB/s, and what are the xGMI units?
# Mislabeling a ceiling is worse than having none, so this prints the surrounding block.
set -uo pipefail
EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_21_saturation
C=$EXP/ceilings.txt

echo "########## context around MAX_BANDWIDTH ##########"
grep -n -B 12 -A 4 'MAX_BANDWIDTH' "$C" | head -50

echo
echo "########## the shownodesbw units line, verbatim ##########"
grep -n -A 2 'Format: min-max' "$C"

echo
echo "########## any other BANDWIDTH / xgmi_bandwidth field in amd-smi ##########"
grep -n -iE 'xgmi.*(bandwidth|speed)|bandwidth.*xgmi|num_links|link_width|link_speed' "$C" | head -20

echo
echo "########## coarse phase progress ##########"
tail -6 "$EXP/logs/campaign_latest.log" 2>/dev/null
ls -la --time-style=+%H:%M:%S "$EXP"/saturation_coarse.json 2>/dev/null || echo "  (coarse json not written yet)"

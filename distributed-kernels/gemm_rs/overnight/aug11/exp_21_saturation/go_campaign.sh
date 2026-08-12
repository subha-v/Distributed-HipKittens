#!/usr/bin/env bash
# Transported by tools/nsh.ps1. Launches the exp_21 campaign DETACHED and returns
# immediately, because campaign.sh blocks on the GPU lease (another agent holds it)
# and a blocking ssh would just time out.
#
#   nsh.ps1 -Script .../go_campaign.sh                       # probe+quick+full+coarse
#   nsh.ps1 -Script .../go_campaign.sh -ArgLine "probe quick"  # a subset
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/aug11/exp_21_saturation
PHASES=${*:-"probe quick full coarse"}

if pgrep -f "exp_21_saturation/campaign.sh" >/dev/null 2>&1; then
  echo "REFUSING: an exp_21 campaign is already running:"
  ps -eo pid,etimes,cmd | grep -E 'exp_21_saturation/(campaign|run_sweep)' | grep -v grep
  exit 1
fi

mkdir -p "$EXP/logs"
echo "########## lease status now ##########"
bash "$ON/tools/gpu_lease.sh" status

echo
echo "########## launching detached campaign: $PHASES ##########"
cd "$EXP"
PHASES="$PHASES" setsid timeout --signal=TERM --kill-after=300 25200 \
  bash "$EXP/campaign.sh" >>"$EXP/logs/campaign_boot.log" 2>&1 &
sleep 6
echo "campaign pid(s): $(pgrep -f 'exp_21_saturation/campaign.sh' | tr '\n' ' ')"
echo
echo "########## first lines of the campaign log ##########"
tail -20 "$EXP/logs/campaign_latest.log" 2>/dev/null || cat "$EXP/logs/campaign_boot.log"

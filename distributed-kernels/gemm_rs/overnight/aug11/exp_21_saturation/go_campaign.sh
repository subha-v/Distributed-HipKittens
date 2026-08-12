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

if pgrep -f "exp_21_saturation/(logs/campaign_run|campaign\.sh)" >/dev/null 2>&1; then
  echo "REFUSING: an exp_21 campaign is already running:"
  ps -eo pid,etimes,cmd | grep -E 'exp_21_saturation/(logs/campaign_run|campaign|run_sweep)' \
    | grep -v grep
  exit 1
fi

mkdir -p "$EXP/logs"
echo "########## lease status now ##########"
bash "$ON/tools/gpu_lease.sh" status

# Run an immutable per-run COPY, never campaign.sh itself. bash reads a script
# incrementally by byte offset, and scp truncates and rewrites the same inode, so a
# scoped push landing during a multi-hour campaign makes the interpreter resume
# mid-token in the new bytes. That is not hypothetical: it killed the 10:01 run with
# a syntax error on a line that was never wrong on disk.
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN=$EXP/logs/campaign_run_$STAMP.sh
cp "$EXP/campaign.sh" "$RUN"
bash -n "$RUN" || { echo "FAIL: campaign.sh does not parse"; exit 1; }
echo "  frozen run script: $RUN (parses OK)"

echo
echo "########## launching detached campaign: $PHASES ##########"
cd "$EXP"
PHASES="$PHASES" setsid timeout --signal=TERM --kill-after=300 25200 \
  bash "$RUN" >>"$EXP/logs/campaign_boot.log" 2>&1 &
sleep 6
echo "campaign pid(s): $(pgrep -f 'campaign_run_' | tr '\n' ' ')"
echo
echo "########## first lines of the campaign log ##########"
tail -20 "$EXP/logs/campaign_latest.log" 2>/dev/null || cat "$EXP/logs/campaign_boot.log"

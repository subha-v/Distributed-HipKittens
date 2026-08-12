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

if pgrep -f "exp_21_saturation/(logs/run_[^ ]*/campaign|campaign)\.sh" >/dev/null 2>&1; then
  echo "REFUSING: an exp_21 campaign is already running:"
  ps -eo pid,etimes,cmd | grep -E 'exp_21_saturation/(logs/run_|campaign|run_sweep)' | grep -v grep
  exit 1
fi

mkdir -p "$EXP/logs"
echo "########## lease status now ##########"
bash "$ON/tools/gpu_lease.sh" status

# Run an immutable, CR-stripped per-run COPY of the WHOLE chain, never the working
# files. Two independent hazards, both observed tonight:
#   1. bash reads a script incrementally by byte offset and scp truncates and
#      rewrites the same inode, so a push landing mid-campaign makes the interpreter
#      resume mid-token in the new bytes. That killed the 10:01 run with a syntax
#      error on a line that was never wrong on disk, and it broke the 10:34 full
#      sweep's exit path after the data had already been written.
#   2. another agent is pushing this tree with CRLF intact, which turns every
#      `do`/`then` into `do\r` and every "$LOG" into "$LOG\r".
# Freezing CR-free copies makes this campaign immune to both for its whole run.
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUNDIR=$EXP/logs/run_$STAMP
mkdir -p "$RUNDIR"
for f in campaign.sh run_sweep.sh run_saturation.py knees.py; do
  sed 's/\r$//' "$EXP/$f" > "$RUNDIR/$f"
done
chmod +x "$RUNDIR"/*.sh
RUN=$RUNDIR/campaign.sh
bash -n "$RUN" || { echo "FAIL: campaign.sh does not parse"; exit 1; }
bash -n "$RUNDIR/run_sweep.sh" || { echo "FAIL: run_sweep.sh does not parse"; exit 1; }
docker exec dhk-gemmrs python3 -m py_compile "$RUNDIR/run_saturation.py" \
  || { echo "FAIL: run_saturation.py does not compile"; exit 1; }
echo "  frozen chain: $RUNDIR (all four files parse/compile)"

echo
echo "########## launching detached campaign: $PHASES ##########"
cd "$EXP"
PHASES="$PHASES" RUNDIR="$RUNDIR" setsid timeout --signal=TERM --kill-after=300 25200 \
  bash "$RUN" >>"$EXP/logs/campaign_boot.log" 2>&1 &
sleep 6
echo "campaign pid(s): $(pgrep -f 'logs/run_.*/campaign.sh' | tr '\n' ' ')"
echo
echo "########## first lines of the campaign log ##########"
tail -20 "$EXP/logs/campaign_latest.log" 2>/dev/null || cat "$EXP/logs/campaign_boot.log"

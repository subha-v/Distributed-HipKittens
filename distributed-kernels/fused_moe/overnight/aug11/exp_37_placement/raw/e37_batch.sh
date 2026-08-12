#!/usr/bin/env bash
# exp_37 placement adjudication -- batch launcher.
#   usage: e37_batch.sh TAG CFGFILE
#
# Pins the node checkout to an EXPLICIT COMMIT (a mode-14 commit is landing on
# the branch during this experiment; SCREEN_SYNC=0 + an explicit pin is the only
# way the binary cannot move mid-campaign) and refuses to start unless the
# pinned tree carries K0P6_MPS_ASCALE_TM 1.
#
# Every config line in CFGFILE is a full 5-rotation campaign
# (500 warmup / 100 timed, K0_MPS_SOAK_ITERS untouched so the harness default
# of 600 stands).
set -uo pipefail
TAG="${1:?usage: e37_batch.sh TAG CFGFILE}"
CFGFILE="${2:?CFGFILE is mandatory}"
PIN="${E37_PIN:-b5215081}"
EXPECT_TM=1

DHK="$HOME/Distributed-HipKittens"
MPSSRC="$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
LOG="$HOME/e37/${TAG}.driver.log"
mkdir -p "$HOME/e37"

echo "== exp_37 pre-flight =="
date -u
pgrep -af 'torchrun|mpirun' || echo "no torchrun/mpirun"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print "KFD: "$1" "$2}'
if pgrep -f 'screen.sh' >/dev/null 2>&1; then
  echo "REFUSING: a screen batch is already running"; exit 9
fi

echo "== pin node checkout to $PIN =="
git -C "$DHK" fetch --all -q
git -C "$DHK" reset -q --hard "$PIN" || { echo "REFUSING: cannot reset to $PIN"; exit 11; }
HEADSHA="$(git -C "$DHK" rev-parse HEAD)"
GOT="$(grep -oE '^#define K0P6_MPS_ASCALE_TM [0-9]+' "$MPSSRC" | awk '{print $3}')"
SRCREV="$(grep -oE '^#define K0P6_MPS_SRC_REV [0-9]+' "$MPSSRC" | awk '{print $3}')"
echo "node HEAD  = $HEADSHA"
echo "SRC_REV    = $SRCREV"
echo "ASCALE_TM  = $GOT (want $EXPECT_TM)"
echo "hip sha256 = $(sha256sum "$MPSSRC" | cut -c1-16)"
git -C "$DHK" status --porcelain | head -5
if [ "$GOT" != "$EXPECT_TM" ]; then
  echo "REFUSING: pinned tree carries ASCALE_TM='$GOT', exp_37 requires '$EXPECT_TM'."
  exit 10
fi

echo "== configs =="
grep -vE '^\s*(#|$)' "$CFGFILE" | nl

rm -f "$LOG"
setsid nohup env \
  SCREEN_ARMS=production,pf6gm_mega,mps_mega \
  SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5 \
  SCREEN_JOB_TIMEOUT=4500 SCREEN_RUN_TIMEOUT=1500 \
  SCREEN_SYNC=0 SCREEN_IDLE_WAIT=600 \
  bash "$HOME/tools/screen.sh" "$TAG" "$CFGFILE" \
  > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! -> $LOG"
sleep 20
head -30 "$LOG"
exit 0

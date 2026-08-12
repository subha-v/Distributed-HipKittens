#!/usr/bin/env bash
# exp_33 launch: ONE 5-rotation campaign at the ratchet config with timestamps=1.
# Uses the repo copy of screen.sh (source of truth) with CRLF stripped.
# Detached: setsid nohup + timeout. Writes only under $HOME/e33, $HOME/k0-mok-e33a,
# $HOME/overnight-scratch.
set -uo pipefail

DHK="$HOME/Distributed-HipKittens"
SRC="$DHK/distributed-kernels/fused_moe/overnight/aug11/tools/screen.sh"
E33="$HOME/e33"
mkdir -p "$E33"

TAG="${E33_TAG:-e33a}"
DRV="$E33/screen_${TAG}.sh"
CFGF="$E33/${TAG}.cfgs"
OUTLOG="$E33/${TAG}_driver.log"

if [ -e "$OUTLOG" ]; then
  echo "FATAL $OUTLOG exists -- pick a new E33_TAG so nothing is overwritten" >&2
  exit 1
fi

tr -d '\r' < "$SRC" > "$DRV"
printf 'C=16,g=353,mode=12,flush_rows=16,timestamps=1\n' > "$CFGF"

echo "driver : $DRV  (md5 $(md5sum "$DRV" | awk '{print $1}'))"
echo "cfg    : $(cat "$CFGF")"
echo "node   : $(git -C "$DHK" rev-parse --short HEAD) $(git -C "$DHK" rev-parse --abbrev-ref HEAD)"

# K0_MOK_POISON_OUT is read by run_campaign.sh from its own environment
# (${K0_MOK_POISON_OUT:-1}); exporting it here makes the default explicit.
export K0_MOK_POISON_OUT=1
export SCREEN_ARMS=production,pf6gm_mega,mps_mega
export SCREEN_WARMUP=500
export SCREEN_TIMED=100
export SCREEN_PROCS=5
export SCREEN_RUN_TIMEOUT=2400
export SCREEN_JOB_TIMEOUT=5400
export SCREEN_TRACE=1
export SCREEN_SOAK_ITERS=''          # empty -> harness default 600. NEVER change.
export SCREEN_SYNC=1
export SCREEN_IDLE_WAIT=300
export SCREEN_BRANCH=codex/distributed-hipkittens-scaffold

setsid nohup timeout 6000 bash "$DRV" "$TAG" "$CFGF" > "$OUTLOG" 2>&1 < /dev/null &
echo "launched pid $! detached; driver log: $OUTLOG"
sleep 12
echo "=== first 40 lines of driver log ==="
sed -n '1,40p' "$OUTLOG"
exit 0

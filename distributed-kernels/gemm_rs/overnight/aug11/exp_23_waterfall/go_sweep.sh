#!/usr/bin/env bash
# Host-side launcher for one exp_23 allocation DRAW. THIS TOUCHES ALL 8 GPUs --
# confirm no other job of ours is running before calling it (one 8-GPU job at a
# time), and confirm clocks are pinned (`tools/set_clocks.sh pin 1900`), because
# idle sclk is ~125 MHz and an unpinned short run is unrepeatable by up to 60%.
#
#   nsh.ps1 -Script ...\exp_23_waterfall\go_sweep.sh -ArgLine "fwd all"
#   nsh.ps1 -Script ...\exp_23_waterfall\go_sweep.sh -ArgLine "rev all"
#
# arg1: fwd|rev   -- arm CONSTRUCTION order. Run half the draws each way: part
#                    of this node's allocation bias is construction order, not
#                    chance, so a one-sided set of draws gives a signed null.
# arg2..: passed through to sweep.py (shapes, passes, iters_scale)
set -uo pipefail

EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_23_waterfall
DIR=${1:-fwd}; shift || true
ARGS=${*:-all}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=$EXP/sweep_${STAMP}_${DIR}.log

REV=0
[ "$DIR" = "rev" ] && REV=1

if [ ! -f "$EXP/fingerprints.json" ]; then
  echo "REFUSING: fingerprints.json absent. Build and fingerprint the rungs first."
  exit 1
fi

echo "exp_23 draw: construction=$DIR args='$ARGS' -> $LOG"
rocm-smi --showclocks 2>/dev/null | grep -iE 'sclk' | head -3

setsid timeout 5400 docker exec -w "$EXP" -e SWEEP_REVERSE=$REV dhk-gemmrs \
  python3 "$EXP/sweep.py" $ARGS >"$LOG" 2>&1 &

echo "launched pid $! -> $LOG"
sleep 5
tail -20 "$LOG"

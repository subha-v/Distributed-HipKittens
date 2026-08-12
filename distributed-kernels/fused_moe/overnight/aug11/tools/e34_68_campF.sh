#!/usr/bin/env bash
# exp_34 decision campaign, final STAMPS-OFF batch: brings the headline arms to
# n >= 4 and adds the stamps-off mode-14 g=65 (throttle-off) point. Interleaved.
set -uo pipefail
TAG="${1:-e34f}"
export SCREEN_ARMS=production,pf6gm_mega,mps_mega
export SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5
export SCREEN_SYNC=0
export SCREEN_TRACE=1
export SCREEN_JOB_TIMEOUT=4500 SCREEN_RUN_TIMEOUT=4200
export SCREEN_IDLE_WAIT=1800

CFG=/tmp/e34_${TAG}_cfgs.txt
cat > "$CFG" <<'EOF'
C=0,g=353,mode=14,flush_rows=16,timestamps=0
C=0,g=481,mode=14,flush_rows=16,timestamps=0
C=16,g=353,mode=12,flush_rows=16,timestamps=0
C=0,g=353,mode=14,flush_rows=16,timestamps=0
C=0,g=481,mode=14,flush_rows=16,timestamps=0
C=8,g=353,mode=14,flush_rows=16,timestamps=0
C=16,g=353,mode=14,flush_rows=16,timestamps=0
C=0,g=65,mode=14,flush_rows=16,timestamps=0
EOF
echo "===HEAD==="; git -C "$HOME/Distributed-HipKittens" rev-parse HEAD
echo "===CFGS==="; cat "$CFG"; date -u
setsid -w timeout 40000 bash "$HOME/tools/screen.sh" "$TAG" "$CFG"
echo "===SCREEN_RC=$?==="; date -u

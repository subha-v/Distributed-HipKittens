#!/usr/bin/env bash
# exp_34 decision campaign, parameterized batch.
#   $1 = tag, $2 = stamps (0|1)
# Interleaved order B,D,A,B,D,A,C8,C16 -- never two of the same arm adjacent, so
# session drift cannot masquerade as a mechanism.
set -uo pipefail
TAG="${1:?tag}"; TS="${2:?stamps 0|1}"
export SCREEN_ARMS=production,pf6gm_mega,mps_mega
export SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5
export SCREEN_SYNC=0
export SCREEN_TRACE=1
export SCREEN_JOB_TIMEOUT=4500 SCREEN_RUN_TIMEOUT=4200
export SCREEN_IDLE_WAIT=900

CFG=/tmp/e34_${TAG}_cfgs.txt
cat > "$CFG" <<EOF
C=0,g=353,mode=14,flush_rows=16,timestamps=$TS
C=16,g=353,mode=12,flush_rows=16,timestamps=$TS
C=0,g=481,mode=14,flush_rows=16,timestamps=$TS
C=0,g=353,mode=14,flush_rows=16,timestamps=$TS
C=16,g=353,mode=12,flush_rows=16,timestamps=$TS
C=0,g=481,mode=14,flush_rows=16,timestamps=$TS
C=8,g=353,mode=14,flush_rows=16,timestamps=$TS
C=16,g=353,mode=14,flush_rows=16,timestamps=$TS
EOF
echo "===HEAD==="; git -C "$HOME/Distributed-HipKittens" rev-parse HEAD
echo "===CFGS==="; cat "$CFG"; date -u
setsid -w timeout 40000 bash "$HOME/tools/screen.sh" "$TAG" "$CFG"
echo "===SCREEN_RC=$?==="; date -u

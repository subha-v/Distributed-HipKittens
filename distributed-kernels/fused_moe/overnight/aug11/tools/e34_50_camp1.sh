#!/usr/bin/env bash
# exp_34 decision campaign, batch 1: STAMPS-OFF headline numbers, interleaved so
# session drift cannot masquerade as a mechanism.
#
#   B = C=0 ,g=353,mode=14   granularity + drain deletion   (the rung)
#   A = C=0 ,g=481,mode=14   granularity alone (drain kept via 0x80)
#   D = C=16,g=353,mode=12   the same-session mode-12 ratchet control
#   plus the placement points C=8 / C=16 at mode 14.
#
# Order is B,D,A,B,D,A,C8,C16 -- never two of the same arm adjacent.
set -uo pipefail
TAG="${1:-e34c1}"
export SCREEN_ARMS=production,pf6gm_mega,mps_mega
export SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5
export SCREEN_SYNC=0             # pinned to 291dfa08
export SCREEN_TRACE=1
export SCREEN_JOB_TIMEOUT=4500 SCREEN_RUN_TIMEOUT=4200
export SCREEN_IDLE_WAIT=600

CFG=/tmp/e34_c1_cfgs.txt
cat > "$CFG" <<'EOF'
C=0,g=353,mode=14,flush_rows=16,timestamps=0
C=16,g=353,mode=12,flush_rows=16,timestamps=0
C=0,g=481,mode=14,flush_rows=16,timestamps=0
C=0,g=353,mode=14,flush_rows=16,timestamps=0
C=16,g=353,mode=12,flush_rows=16,timestamps=0
C=0,g=481,mode=14,flush_rows=16,timestamps=0
C=8,g=353,mode=14,flush_rows=16,timestamps=0
C=16,g=353,mode=14,flush_rows=16,timestamps=0
EOF
echo "===HEAD==="; git -C "$HOME/Distributed-HipKittens" rev-parse HEAD
echo "===CFGS==="; cat "$CFG"; date -u
setsid -w timeout 40000 bash "$HOME/tools/screen.sh" "$TAG" "$CFG"
echo "===SCREEN_RC=$?==="; date -u

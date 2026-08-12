#!/usr/bin/env bash
# exp_34 BLOCKING diagnostic batch. The same-session mode-12 control at the
# ratchet cfg reads 7,222 us where rev 26 read 6,488 -- i.e. it behaves like the
# UNTHROTTLED g=65 (7,111 at rev 26). Two hypotheses, both settled here, all four
# points at the pin:
#   D1  historical cfg string verbatim (NO timestamps token -- no historical row
#       has ever used timestamps=0, so the token itself is a suspect)
#   D2  timestamps=1 (the other historical spelling)
#   D3  timestamps=0 (the spelling that produced 7,222; repeat)
#   D4  g=65, the known-unthrottled reference: if D1==D4 the throttle is dead at
#       the pin for mode 12, independent of the token.
set -uo pipefail
TAG="${1:-e34d}"
export SCREEN_ARMS=production,pf6gm_mega,mps_mega
export SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5
export SCREEN_SYNC=0
export SCREEN_TRACE=1
export SCREEN_JOB_TIMEOUT=4500 SCREEN_RUN_TIMEOUT=4200
export SCREEN_IDLE_WAIT=1200

CFG=/tmp/e34_${TAG}_cfgs.txt
cat > "$CFG" <<'EOF'
C=16,g=353,mode=12,flush_rows=16
C=16,g=353,mode=12,flush_rows=16,timestamps=1
C=16,g=353,mode=12,flush_rows=16,timestamps=0
C=16,g=65,mode=12,flush_rows=16
EOF
echo "===HEAD==="; git -C "$HOME/Distributed-HipKittens" rev-parse HEAD
echo "===CFGS==="; cat "$CFG"; date -u
setsid -w timeout 20000 bash "$HOME/tools/screen.sh" "$TAG" "$CFG"
echo "===SCREEN_RC=$?==="; date -u

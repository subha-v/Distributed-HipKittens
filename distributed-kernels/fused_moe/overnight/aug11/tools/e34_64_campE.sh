#!/usr/bin/env bash
# exp_34 decision campaign, STAMPS-ON batch (the phase story + the service-pool
# degeneration evidence: ts_servicedrain should collapse for mode 14).
#
# It also carries the arms that the throttle finding forces: g=65 is exactly
# g=353 minus the throttle bit (0x20) and the depth selector (0x100), so
# mode14 g=65 vs mode14 g=353 tests whether mode 14's transport lost the
# injection bound the same way mode 12's did at this pin. mode12 g=65 is the
# unthrottled denominator that mode 14 must be compared against here.
set -uo pipefail
TAG="${1:-e34e}"
export SCREEN_ARMS=production,pf6gm_mega,mps_mega
export SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5
export SCREEN_SYNC=0
export SCREEN_TRACE=1
export SCREEN_JOB_TIMEOUT=4500 SCREEN_RUN_TIMEOUT=4200
export SCREEN_IDLE_WAIT=1800

CFG=/tmp/e34_${TAG}_cfgs.txt
cat > "$CFG" <<'EOF'
C=0,g=353,mode=14,flush_rows=16,timestamps=1
C=16,g=353,mode=12,flush_rows=16,timestamps=1
C=0,g=481,mode=14,flush_rows=16,timestamps=1
C=0,g=65,mode=14,flush_rows=16,timestamps=1
C=16,g=65,mode=12,flush_rows=16,timestamps=1
C=8,g=353,mode=14,flush_rows=16,timestamps=1
C=16,g=353,mode=14,flush_rows=16,timestamps=1
C=0,g=353,mode=14,flush_rows=16,timestamps=1
EOF
echo "===HEAD==="; git -C "$HOME/Distributed-HipKittens" rev-parse HEAD
echo "===CFGS==="; cat "$CFG"; date -u
setsid -w timeout 40000 bash "$HOME/tools/screen.sh" "$TAG" "$CFG"
echo "===SCREEN_RC=$?==="; date -u

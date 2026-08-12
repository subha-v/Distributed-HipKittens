#!/usr/bin/env bash
# exp_35 step 6: the waterfall campaigns.
#
# Four 5-rotation campaigns (500 warmup / 100 timed / soak 600), ALTERNATING
# rung (c) and rung (b) so that any time-ordering drift on the node cannot be
# confounded with the mechanism:
#
#   1  C=16,g=353,mode=12,flush_rows=16   rung (c)  throttle ON, depth 4  (ratchet)
#   2  C=16,g=65 ,mode=12,flush_rows=16   rung (b)  throttle OFF
#   3  C=16,g=353,...                     rung (c)  repeat
#   4  C=16,g=65 ,...                     rung (b)  repeat
#
# `production` and `pf6gm_mega` are arms in EVERY campaign, so every rung gets a
# same-run paired denominator, and rung (a) (`pf6gm_mega`, the homogeneous
# baseline) is measured four independent times without a separate campaign.
set -uo pipefail
TAG="${1:-e35w}"
export SCREEN_ARMS=production,pf6gm_mega,mps_mega
export SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5
export SCREEN_SYNC=0            # CRITICAL: the SHA is pinned to ca5b683f; never resync
export SCREEN_TRACE=1
export SCREEN_JOB_TIMEOUT=4500 SCREEN_RUN_TIMEOUT=4200
export SCREEN_IDLE_WAIT=600

CFG=/tmp/e35_wf_cfgs.txt
cat > "$CFG" <<'EOF'
C=16,g=353,mode=12,flush_rows=16,timestamps=1
C=16,g=65,mode=12,flush_rows=16,timestamps=1
C=16,g=353,mode=12,flush_rows=16,timestamps=1
C=16,g=65,mode=12,flush_rows=16,timestamps=1
EOF

echo "===HEAD_BEFORE==="; git -C "$HOME/Distributed-HipKittens" rev-parse HEAD
echo "===CFGS==="; cat "$CFG"
date -u
setsid -w timeout 20000 bash "$HOME/tools/screen.sh" "$TAG" "$CFG"
echo "===SCREEN_RC=$?==="
date -u
echo "===HEAD_AFTER==="; git -C "$HOME/Distributed-HipKittens" rev-parse HEAD

#!/usr/bin/env bash
# exp_35 step 5: EMPIRICAL DECODE PROOF.
# 1-process smoke of the two mode-12 rungs with K0_MPS_TRACE=1 and
# K0_MPS_DESC_DUMP=1, then check the dumped descriptor's packed MPS cfg word
# against the arithmetic from moe_mps_adapter.cuh:encode_config.
#
#   rung (c) ratchet     C=16,g=353,mode=12,flush_rows=16  -> throttle ON depth 4
#   rung (b) unthrottled C=16,g=65 ,mode=12,flush_rows=16  -> throttle OFF
#
# g=353 (0x161) and g=65 (0x041) are identical in physical_g(0x1), detect(0x10)
# and skip_dead_part_zero(0x40); they differ ONLY in the throttle enable bit
# (0x20) and the depth selector (0x300), which config_is_valid FORCES to zero
# whenever the enable bit is clear. So g=65 is the unique legal "throttle bits
# only" neighbour of g=353.
set -uo pipefail
TAG=e35smoke
export SCREEN_ARMS=production,pf6gm_mega,mps_mega
export SCREEN_WARMUP=1 SCREEN_TIMED=1 SCREEN_PROCS=1
export SCREEN_SYNC=0            # CRITICAL: do NOT resync; the SHA is pinned to ca5b683f
export SCREEN_TRACE=1
export SCREEN_JOB_TIMEOUT=2400 SCREEN_RUN_TIMEOUT=2100
export K0_MPS_DESC_DUMP=1

CFG=/tmp/e35_smoke_cfgs.txt
cat > "$CFG" <<'EOF'
C=16,g=353,mode=12,flush_rows=16,timestamps=1
C=16,g=65,mode=12,flush_rows=16,timestamps=1
EOF

echo "===PINNED_HEAD_BEFORE==="
git -C "$HOME/Distributed-HipKittens" rev-parse HEAD
echo "===CFGS==="; cat "$CFG"
echo "===RUN==="
setsid -w timeout 3000 bash "$HOME/tools/screen.sh" "$TAG" "$CFG"
RC=$?
echo "===SCREEN_RC=$RC==="
echo "===PINNED_HEAD_AFTER==="
git -C "$HOME/Distributed-HipKittens" rev-parse HEAD

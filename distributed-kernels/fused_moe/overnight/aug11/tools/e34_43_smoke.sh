#!/usr/bin/env bash
# exp_34 ladder steps 2-6: the FIRST GPU run of mode 14, at C=0 only.
# 1 warmup / 1 timed / 1 rotation, but the harness still runs the full
# correctness gate, the harness negative control, the poison self-test and the
# 600-epoch soak, which is ladder items 3, 4a, 5 and 6.
set -uo pipefail
TAG=e34smoke
export SCREEN_ARMS=production,pf6gm_mega,mps_mega
export SCREEN_WARMUP=1 SCREEN_TIMED=1 SCREEN_PROCS=1
export SCREEN_SYNC=0             # CRITICAL: pinned to 291dfa08, never resync
export SCREEN_TRACE=1
export SCREEN_JOB_TIMEOUT=3600 SCREEN_RUN_TIMEOUT=3300
export SCREEN_IDLE_WAIT=300
export K0_MPS_DESC_DUMP=1

CFG=/tmp/e34_smoke_cfgs.txt
cat > "$CFG" <<'EOF'
C=0,g=353,mode=14,flush_rows=16,timestamps=1
EOF

echo "===HEAD_BEFORE==="; git -C "$HOME/Distributed-HipKittens" rev-parse HEAD
echo "===SRC_REV==="; grep -oE 'K0P6_MPS_SRC_REV [0-9]+' "$HOME/Distributed-HipKittens/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" | tail -1
echo "===CFG==="; cat "$CFG"; date -u
setsid -w timeout 4000 bash "$HOME/tools/screen.sh" "$TAG" "$CFG"
echo "===SCREEN_RC=$?==="; date -u
echo "===HEAD_AFTER==="; git -C "$HOME/Distributed-HipKittens" rev-parse HEAD

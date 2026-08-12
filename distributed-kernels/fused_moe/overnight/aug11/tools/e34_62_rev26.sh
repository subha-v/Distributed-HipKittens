#!/usr/bin/env bash
# exp_34 attribution control: run the ratchet cfg at rev 26 (f113d73f) IN THIS
# SESSION. The node checkout IS the arm. This closes the last alternative
# explanation for the pin's mode-12 M7 inflation (7,220 us / M7 3,472 vs rev 26's
# 6,488 / M7 ~2,660): if rev 26 reproduces ~6,49x right now, the regression is
# attributable to 291dfa08 and not to today's session.
# The pin is restored on EVERY exit path.
set -uo pipefail
TAG="${1:-e34r26}"
REPO=$HOME/Distributed-HipKittens
PIN=291dfa08

restore() {
  echo "[restore] resetting $REPO back to the pin $PIN"
  git -C "$REPO" reset -q --hard "$PIN"
  git -C "$REPO" rev-parse HEAD
}
trap restore EXIT

echo "===before==="; git -C "$REPO" rev-parse HEAD
git -C "$REPO" reset -q --hard f113d73f
echo "===now at==="; git -C "$REPO" rev-parse HEAD
grep -n "K0P6_MPS_SRC_REV" "$REPO/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" | head -3

export SCREEN_ARMS=production,pf6gm_mega,mps_mega
export SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5
export SCREEN_SYNC=0
export SCREEN_TRACE=1
export SCREEN_JOB_TIMEOUT=4500 SCREEN_RUN_TIMEOUT=4200
export SCREEN_IDLE_WAIT=1800

CFG=/tmp/e34_${TAG}_cfgs.txt
cat > "$CFG" <<'EOF'
C=16,g=353,mode=12,flush_rows=16,timestamps=1
C=16,g=353,mode=12,flush_rows=16,timestamps=0
EOF
cat "$CFG"; date -u
setsid -w timeout 12000 bash "$HOME/tools/screen.sh" "$TAG" "$CFG"
echo "===SCREEN_RC=$?==="; date -u

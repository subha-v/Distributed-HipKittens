#!/usr/bin/env bash
# exp_24 decision campaign: 5 rotated processes, 500 warmup / 100 timed, no
# timestamps -- the exact conditions exp_21's 6,685 us ratchet was measured under.
#   point 1: g=97  = Mechanism A at exp_21's throttle depth (the candidate)
#   point 2: g=33  = exp_24's control, which isolates MPS-DELTA (6) (the
#            slot_off fold) against exp_21's published 0.8665/0.8657
set -uo pipefail
SC=$HOME/.overnight-scripts
CFG=$HOME/overnight-scratch/e24c.cfgs
cat > "$CFG" <<'EOF'
C=16,g=97,mode=12,flush_rows=16
C=16,g=33,mode=12,flush_rows=16
EOF
echo "=== node state"
git -C "$HOME/Distributed-HipKittens" rev-parse --short HEAD
grep -oE 'K0P6_MPS_SRC_REV [0-9]+' \
  "$HOME/Distributed-HipKittens/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" | tail -1
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1, $2}'
echo "=== launching campaign detached (2 points x 5 processes)"
cd "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
SCREEN_SYNC=0 SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5 \
  SCREEN_RUN_TIMEOUT=2400 SCREEN_JOB_TIMEOUT=5400 \
  setsid nohup timeout 10800 \
  bash "$SC/screen.sh" e24c "$CFG" \
  > "$HOME/overnight-scratch/e24c_batch.out" 2>&1 < /dev/null &
echo "batch pid=$!"
sleep 15
head -12 "$HOME/overnight-scratch/e24c_batch.out"
exit 0

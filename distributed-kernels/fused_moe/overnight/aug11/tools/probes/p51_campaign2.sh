#!/usr/bin/env bash
# exp_24 campaign 2: confirm g=97 (the new ratchet candidate) and price the one
# composed candidate the screens liked, g=353 = A + throttle depth 4.
set -uo pipefail
SC=$HOME/.overnight-scripts
CFG=$HOME/overnight-scratch/e24d.cfgs
cat > "$CFG" <<'EOF'
C=16,g=97,mode=12,flush_rows=16
C=16,g=353,mode=12,flush_rows=16
C=16,g=97,mode=12,flush_rows=16
EOF
echo "=== node state"
git -C "$HOME/Distributed-HipKittens" rev-parse --short HEAD
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1, $2}'
cd "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
SCREEN_SYNC=0 SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5 \
  SCREEN_RUN_TIMEOUT=2400 SCREEN_JOB_TIMEOUT=5400 \
  setsid nohup timeout 10800 \
  bash "$SC/screen.sh" e24d "$CFG" \
  > "$HOME/overnight-scratch/e24d_batch.out" 2>&1 < /dev/null &
echo "batch pid=$!"
sleep 12
head -11 "$HOME/overnight-scratch/e24d_batch.out"
exit 0

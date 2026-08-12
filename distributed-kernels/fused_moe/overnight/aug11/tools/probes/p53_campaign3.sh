#!/usr/bin/env bash
# exp_24 campaign 3: g=353 (A + depth 4) had n=1 and is the ratchet claim.
# Two more 5-rotation campaigns before it is declared.
set -uo pipefail
SC=$HOME/.overnight-scripts
CFG=$HOME/overnight-scratch/e24e.cfgs
cat > "$CFG" <<'EOF'
C=16,g=353,mode=12,flush_rows=16
C=16,g=353,mode=12,flush_rows=16
EOF
git -C "$HOME/Distributed-HipKittens" rev-parse --short HEAD
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1, $2}'
cd "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
SCREEN_SYNC=0 SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5 \
  SCREEN_RUN_TIMEOUT=2400 SCREEN_JOB_TIMEOUT=5400 \
  setsid nohup timeout 10800 \
  bash "$SC/screen.sh" e24e "$CFG" \
  > "$HOME/overnight-scratch/e24e_batch.out" 2>&1 < /dev/null &
echo "batch pid=$!"
sleep 10
tail -3 "$HOME/overnight-scratch/e24e_batch.out"
exit 0

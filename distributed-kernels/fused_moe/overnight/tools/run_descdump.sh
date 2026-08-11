#!/usr/bin/env bash
# exp_01: dump the 63 mps descriptor words at first launch, on a debug_stop=6 run
# (known CLEAN), so we get the descriptor WITHOUT the fault. Decisive test for
# "is desc[61] (slots arena) actually null?".
set -uo pipefail

cd "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe" || exit 2

echo "=== node preflight ==="
pgrep -af 'torchrun|mpirun' && { echo "REFUSING: gpu job already running"; exit 3; }
rocm-smi --showpids 2>/dev/null | sed -n '1,20p'

OUT="$HOME/k0-mok-mps-descdump"
rm -rf "$OUT"
mkdir -p "$OUT"
LOG="$HOME/exp01_descdump.log"
rm -f "$LOG"

echo "=== launching descdump (debug_stop=6, DESC_DUMP=1, TRACE=1) ==="
nohup setsid timeout 1800 env \
  K0_MOK_ARMS=production,mps_mega \
  K0_MOK_WARMUP_ITERS=1 K0_MOK_TIMED_ITERS=1 \
  K0_MOK_OUTPUT_ROOT="$OUT" \
  K0_MOK_RUN_TIMEOUT=1500 \
  K0_MPS_CFG="C=8,g=2,mode=2,flush_rows=16" \
  K0_MPS_DESC_DUMP=1 \
  K0_MPS_TRACE=1 \
  K0_MPS_DEBUG_STOP=6 \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh descdump 1 \
  > "$LOG" 2>&1 < /dev/null &

echo "launched pid $!"
sleep 5
echo "=== first lines ==="
sed -n '1,15p' "$LOG"

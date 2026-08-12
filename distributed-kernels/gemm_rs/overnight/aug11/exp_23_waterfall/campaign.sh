#!/usr/bin/env bash
# exp_23 GPU campaign: four allocation draws, split evenly between forward and
# reversed arm-construction order, then pool + score.
#
# Draw count and order split are FIXED HERE, before any number is seen. Adding
# draws until a contrast crosses its threshold is the exact procedure that
# manufactures a false positive, so the stopping rule is written down first.
#
# The whole campaign is wrapped in the node's GPU lease and releases it from an
# EXIT trap, so a timeout or a crash cannot leave the lock held.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/aug11/exp_23_waterfall
OWNER=exp_23
DRAWS=("fwd" "rev" "fwd" "rev")

release() {
  echo
  echo "===== releasing GPU lease ====="
  bash "$ON/tools/gpu_lease.sh" release "$OWNER"
}

echo "===== exp_23 campaign start $(date -Is) ====="
bash "$ON/tools/gpu_lease.sh" acquire "$OWNER" 1800 || {
  echo "LEASE NOT ACQUIRED -- aborting without touching the GPUs"
  bash "$ON/tools/gpu_lease.sh" status
  exit 1
}
trap release EXIT

echo
echo "===== clocks BEFORE (pinned expected: ~1900 MHz sclk, perf determinism) ====="
rocm-smi --showclocks 2>/dev/null | grep -iE 'sclk|GPU\[' | head -20
rocm-smi --showperflevel 2>/dev/null | grep -iE 'Performance|GPU\[' | head -20

echo
echo "########## preflight smoke draw (shape 3, 1 rotation, no draw file) ##########"
# Cheap insurance: a Python-level error in sweep.py would otherwise be
# discovered four draws and forty minutes into a held lease. --smoke writes no
# draw file, so it cannot contaminate the pool.
timeout 900 docker exec -w "$EXP" dhk-gemmrs \
  python3 "$EXP/sweep.py" 3 8 0.3 --smoke
smoke=$?
echo "  preflight exit=$smoke"
if [ "$smoke" != "0" ]; then
  echo "PREFLIGHT FAILED -- aborting the campaign and releasing the lease"
  exit 1
fi
sleep 10

n=0
for dir in "${DRAWS[@]}"; do
  n=$((n + 1))
  rev=0
  [ "$dir" = "rev" ] && rev=1
  echo
  echo "########## draw $n/${#DRAWS[@]}: construction=$dir $(date -Is) ##########"
  # SIGTERM only on timeout (the default). This kernel uses HIP IPC and a
  # SIGKILLed process can leak a mapping that wedges the node.
  timeout 2700 docker exec -w "$EXP" -e SWEEP_REVERSE=$rev dhk-gemmrs \
    python3 "$EXP/sweep.py" all
  echo "  draw $n exit=$?"
  # Let any lingering worker retire before the next draw allocates.
  sleep 10
done

echo
echo "===== clocks AFTER ====="
rocm-smi --showclocks 2>/dev/null | grep -iE 'sclk|GPU\[' | head -20

echo
echo "===== pooling all draws -> waterfall.json ====="
docker exec -w "$EXP" dhk-gemmrs python3 "$EXP/sweep.py" --pool
echo "  pool exit=$?"

echo
echo "===== scoring (paired contrasts, exact rank-sum) -> stats.json ====="
docker exec -w "$EXP" dhk-gemmrs python3 "$EXP/pool_stats.py"
echo "  stats exit=$?"

echo
echo "===== exp_23 campaign done $(date -Is) ====="

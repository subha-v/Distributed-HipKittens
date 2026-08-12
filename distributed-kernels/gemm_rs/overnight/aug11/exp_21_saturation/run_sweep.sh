#!/usr/bin/env bash
# exp_21 GPU sweep launcher. THIS IS THE ONLY SCRIPT HERE THAT TOUCHES A GPU.
#
# Node discipline: one 8-GPU job of ours at a time. Refuses to start if another of our jobs is
# alive, uses setsid + timeout, and never SIGKILLs a GPU process (leaked HIP IPC wedges the node).
#
#   bash run_sweep.sh                 # full sweep, ~30-60 min
#   bash run_sweep.sh quick           # smoke, ~2-4 min, 1 rotation
#   bash run_sweep.sh coarse          # production-parity cross-check (coarse payload heap)
set -uo pipefail

REPO=/home/subvadla/dhk
EXP=$REPO/distributed-kernels/gemm_rs/overnight/aug11/exp_21_saturation
# Artifacts always land in EXP; the driver itself may be a frozen per-run copy, so a
# push landing mid-sweep cannot rewrite the interpreter's input (see go_campaign.sh).
PY=${SAT_PY:-$(dirname "${BASH_SOURCE[0]}")/run_saturation.py}
export SAT_BASE=$EXP          # the driver's module and artifacts live here, not next to PY
LOGS=$EXP/logs
mkdir -p "$LOGS"
MODE=${1:-full}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=$LOGS/sweep_${MODE}_${STAMP}.log

echo "########## pre-flight ##########"
if [ ! -f "$EXP/build/sat_ubench.so" ]; then
  echo "FAIL: build/sat_ubench.so missing -- run build.sh first"; exit 1
fi

echo "-- our processes that must not be alive --"
ps -eo pid,etimes,cmd 2>/dev/null | grep -E 'run_saturation|m7_bench|exp_ablation|mp_smoke|eval\.py' \
  | grep -v grep || echo "  (none)"
if ps -eo cmd 2>/dev/null | grep -qE 'run_saturation\.py'; then
  echo "FAIL: an exp_21 sweep is already running"; exit 1
fi

echo
echo "-- clocks (must be pinned ~1900; unpinned short runs are unrepeatable by up to 60%) --"
rocm-smi --showclocks 2>&1 | grep -i sclk || echo "  rocm-smi unavailable in this context"

echo
echo "-- GPU occupancy right now --"
rocm-smi 2>&1 | tail -14 || true

ARGS=""
case "$MODE" in
  probe)  ARGS="--probe --skip-smi --out $EXP/saturation_probe.json" ;;
  quick)  ARGS="--quick --out $EXP/saturation_quick.json" ;;
  coarse) ARGS="--payload-coarse --out $EXP/saturation_coarse.json" ;;
  full)   ARGS="--out $EXP/saturation.json" ;;
  *)      ARGS="$*" ;;
esac

echo
echo "########## launching: python3 $PY $ARGS ##########"
echo "log: $LOG"
cd "$EXP"
setsid timeout --signal=TERM --kill-after=120 7200 \
  python3 "$PY" $ARGS >"$LOG" 2>&1
rc=$?
echo "sweep exit=$rc"
echo
echo "########## tail ##########"
tail -30 "$LOG"
exit $rc

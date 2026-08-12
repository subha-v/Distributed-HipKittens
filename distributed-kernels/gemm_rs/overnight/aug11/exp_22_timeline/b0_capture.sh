#!/usr/bin/env bash
# exp_22 arm (a): capture a rocprofv3 kernel trace of the reference GEMM+RCCL
# baseline on shape 5, rank 0 only.
#
# NEEDS THE GPU.  Do not run until the orchestrator hands the node over.
# Preflight refuses to start if any other KFD process is alive: an interleaved
# tenant would put foreign kernels in our trace and inflate every interval.
#
#   b0_capture.sh [tag] [m n k bias seed]
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E22=$ON/aug11/exp_22_timeline
TAG=${1:-s5}
M=${2:-8192}; N=${3:-4096}; K=${4:-14336}; BIAS=${5:-1}; SEED=${6:-7168}
OUT=$E22/prof/$TAG
mkdir -p "$OUT"

echo "===== preflight ====="
pids=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids: $pids"
if [ "$pids" != "0" ]; then
  echo "ABORT: node is not clean -- another tenant's kernels would land in the trace"
  exit 1
fi

echo "===== clocks (recorded into the arm, per the charter) ====="
bash "$ON/tools/set_clocks.sh" pin 1900 2>&1 | tail -3
rocm-smi --showclocks 2>/dev/null | tee "$OUT/clocks_before.txt" | head -20

echo "===== capture ====="
rm -f "$OUT"/k_*.csv
setsid timeout 1200 docker exec -w "$E22" dhk-gemmrs \
  env MASTER_ADDR=127.0.0.1 MASTER_PORT=29622 \
      TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --nproc_per_node=8 --nnodes=1 \
    --no-python bash b0_rank0_wrap.sh "$OUT" \
      b0_ref.py --m "$M" --n "$N" --k "$K" --bias "$BIAS" --seed "$SEED" \
                --out "$OUT/boundaries.json" \
  > "$OUT/run.log" 2>&1
rc=$?
echo "exit=$rc"
grep -h '^REF ' "$OUT/run.log" || { echo "NO REF LINE; tail:"; tail -25 "$OUT/run.log"; }

rocm-smi --showclocks 2>/dev/null | tee "$OUT/clocks_after.txt" | head -20

echo "===== trace inventory ====="
find "$OUT" -name '*kernel_trace*.csv' -printf '%p  %s bytes\n' || true

echo "===== derive the kernel -> resource-class map from THIS trace ====="
docker exec -w "$E22" dhk-gemmrs python3 b0_kernel_map.py \
  --trace-dir "$OUT" \
  --boundaries "$OUT/boundaries.json" \
  --map-json "$E22/b0_kernel_map.json" \
  --events-json "$E22/events_b0_reference.json" \
  --shape "$M,$N,$K,$BIAS,$SEED"

echo "===== DONE ====="

#!/usr/bin/env bash
# exp_10 -- run the minimal 8-rank rank-1 driver. GPU. Seconds, not 25 minutes.
#   $1 shape index (0..5)   $2 timing iters   $3 tcp port   $4 extra env tag
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ARM=$ON/compbench/rank1
IRISDST=/usr/local/lib/python3.10/dist-packages
SHAPE=${1:-0}
ITERS=${2:-0}
PORT=${3:-12377}
TAG=${4:-serial}

# GPU coordination: never start on top of another job.
echo "kfd_pids before launch: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"

cp "$ON/experiments/exp_10_rank1/r1_min.py" "$ARM/r1_min.py"
LOGS=$ARM/logs_${TAG}_s${SHAPE}
rm -rf "$LOGS"; mkdir -p "$LOGS"

# AMD_SERIALIZE_KERNEL=3 waits for each kernel to complete before returning, so
# a memory fault is attributed to the exact launch that caused it rather than
# to the next synchronizing call.
case "$TAG" in
  serial) EXTRA=(-e AMD_SERIALIZE_KERNEL=3) ;;
  plain)  EXTRA=() ;;
  *)      EXTRA=(-e AMD_SERIALIZE_KERNEL=3) ;;
esac

echo "=== running shape $SHAPE iters $ITERS tag $TAG ==="
set -x
docker exec -w "$ARM" \
  -e PATH="$ON/tools/compat/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
  -e PYTHONPATH="$ON/tools/compat:$IRISDST" \
  -e PYTHONUNBUFFERED=1 \
  -e COMPAT_SHIM_VERBOSE=1 \
  -e TRITON_CACHE_DIR="$ARM/.triton" \
  -e TORCHINDUCTOR_CACHE_DIR="$ARM/.inductor" \
  -e R1_OUTDIR="$LOGS" \
  -e R1_TIMEOUT=420 \
  -e HSA_ENABLE_COREDUMP=0 \
  "${EXTRA[@]}" \
  dhk-gemmrs bash -lc "timeout --signal=TERM 460 setsid python3 -u r1_min.py $SHAPE $ITERS $PORT" \
  > "$LOGS/driver.txt" 2>&1
rc=$?
set +x
echo "driver rc=$rc"

echo
echo "===== driver.txt ====="
cat "$LOGS/driver.txt"

echo
echo "===== per-rank stdout (rank 0 and 1) ====="
for r in 0 1; do
  echo "--- stdout_rank$r ---"; cat "$LOGS/stdout_rank$r.txt" 2>/dev/null | tail -25
done

echo
echo "===== per-rank stderr: error signatures ====="
for r in 0 1 2 3 4 5 6 7; do
  f="$LOGS/stderr_rank$r.txt"
  [ -f "$f" ] || { echo "rank $r: no stderr file"; continue; }
  echo "--- rank $r ($(wc -c < "$f") bytes) ---"
  grep -nE 'Memory access fault|UNI Error|Error at|Traceback|Fatal|SHIM_WAS_CALLED|assert|Aborted|error:' "$f" | head -8
done

echo
echo "===== rank 0 stderr, last 60 lines ====="
tail -60 "$LOGS/stderr_rank0.txt" 2>/dev/null

echo
echo "===== dead-code check: did heap_bases ever appear? (expected: never) ====="
ls -la "$ARM"/heap_bases_*.pkl 2>/dev/null || echo "no heap_bases_*.pkl (CREATE_SHEMEM_CODE is never spawned, so this is expected and NOT a failure signal)"
echo "--- ipc handle files (the REAL bootstrap artifact) ---"
ls -la "$ARM"/ipc_handles_rank*.bin 2>/dev/null | head -10 || echo "none"
echo "===== DONE ====="

#!/usr/bin/env bash
# exp_10 generic runner.
#   $1 driver py (r1_bisect.py | r1_min.py)   $2 shape   $3 port
#   $4 bufops: 0 disables AMDGCN_USE_BUFFER_OPS, 1 leaves Triton's default
#   $5 tag     $6 iters (r1_min only)
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ARM=$ON/compbench/rank1
IRISDST=/usr/local/lib/python3.10/dist-packages
DRV=${1:-r1_bisect.py}
SHAPE=${2:-0}
PORT=${3:-12399}
BUFOPS=${4:-0}
TAG=${5:-bufops$4}
ITERS=${6:-0}

echo "kfd_pids before launch: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"
cp "$ON/experiments/exp_10_rank1/$DRV" "$ARM/$DRV"
LOGS=$ARM/logs_${TAG}_s${SHAPE}
rm -rf "$LOGS"; mkdir -p "$LOGS"
rm -f "$ARM"/ipc_handles_rank*.bin

# The cached hsaco was compiled WITH buffer ops; the knob only takes effect on
# a fresh compile, so the cache must go or the test is vacuous.
rm -rf "$ARM/.triton"

ARGS="$SHAPE $PORT"
[ "$DRV" = "r1_min.py" ] && ARGS="$SHAPE $ITERS $PORT"

docker exec -w "$ARM" \
  -e PATH="$ON/tools/compat/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
  -e PYTHONPATH="$ON/tools/compat:$IRISDST" \
  -e PYTHONUNBUFFERED=1 \
  -e TRITON_CACHE_DIR="$ARM/.triton" \
  -e R1_OUTDIR="$LOGS" \
  -e R1_TIMEOUT=600 \
  -e HSA_ENABLE_COREDUMP=0 \
  -e AMDGCN_USE_BUFFER_OPS="$BUFOPS" \
  dhk-gemmrs bash -lc "timeout --signal=TERM 900 setsid python3 -u $DRV $ARGS" \
  > "$LOGS/driver.txt" 2>&1
echo "driver rc=$?"

echo
echo "===== driver summary ====="
sed -n '/driver summary/,$p' "$LOGS/driver.txt"

echo
echo "===== per-rank stage progress ====="
for r in 0 1 2 3 4 5 6 7; do
  echo "--- rank $r ---"
  grep -E 'STAGE|FAILED|clean exit|correctness|custom_kernel #1|iters,' \
    "$LOGS/stdout_rank$r.txt" 2>/dev/null | sed 's/^/    /'
done

echo
echo "===== faults? ====="
FOUND=0
for r in 0 1 2 3 4 5 6 7; do
  f="$LOGS/stderr_rank$r.txt"
  if grep -q 'Memory access fault' "$f" 2>/dev/null; then
    FOUND=1
    echo "--- rank $r ---"
    grep -E 'Memory access fault' "$f" | head -3
  fi
done
[ "$FOUND" = 0 ] && echo "  NO memory access faults in any rank"

echo
echo "===== SHIM_WAS_CALLED anywhere? (must be absent) ====="
grep -rl 'SHIM_WAS_CALLED' "$LOGS" 2>/dev/null || echo "  absent (good)"

echo
echo "===== ISA check: buffer_store vs global_store in the recompiled kernel ====="
for f in $(find "$ARM/.triton" -name '_kernel1.amdgcn' 2>/dev/null | head -3); do
  echo "--- $(basename "$(dirname "$f")") ---"
  grep -oE '(global|buffer|flat)_store[a-z0-9_]*' "$f" | sort | uniq -c | sed 's/^/    /'
  grep -oE '(global|buffer|flat)_load[a-z0-9_]*' "$f" | sort | uniq -c | sed 's/^/    /'
done
echo "===== DONE ====="

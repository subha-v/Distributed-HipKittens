#!/usr/bin/env bash
# exp_10 -- run the frozen (packed_metadata-repaired) rank-1 submission through
# the OFFICIAL evaluator on the six graded benchmark shapes.
#   $1 arm (rank1|reference)   $2 mode (benchmark|test)   $3 timeout_s
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ROOT=$ON/compbench
IRISDST=/usr/local/lib/python3.10/dist-packages
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
ARMNAME=${1:-rank1}
MODE=${2:-benchmark}
TMO=${3:-1500}
ARM=$ROOT/$ARMNAME

echo "kfd_pids before launch: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"

# The six graded benchmark shapes, verbatim from task.yml `benchmarks:`.
cat > "$ARM/cases_bench.txt" <<'EOF'
world_size: 8; m: 64; n: 7168; k: 18432; has_bias: False; seed: 1234
world_size: 8; m: 512; n: 4096; k: 12288; has_bias: True; seed: 663
world_size: 8; m: 2048; n: 2880; k: 2880; has_bias: True; seed: 166
world_size: 8; m: 4096; n: 4096; k: 4096; has_bias: False; seed: 1371
world_size: 8; m: 8192; n: 4096; k: 14336; has_bias: True; seed: 7168
world_size: 8; m: 8192; n: 8192; k: 29568; has_bias: False; seed: 42
EOF

CASES=cases_bench.txt
[ "$MODE" = "test" ] && CASES=cases_test.txt

rm -rf "$ARM/.triton"
rm -f "$ARM"/ipc_handles_rank*.bin "$ARM"/*.pkl

echo "=== $ARMNAME : eval.py $MODE $CASES (buffer ops disabled) ==="
docker exec -w "$ARM" \
  -e PATH="$ON/tools/compat/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
  -e PYTHONPATH="$ON/tools/compat:$IRISDST" \
  -e PYTHONUNBUFFERED=1 \
  -e POPCORN_FD=3 -e POPCORN_GPUS=8 \
  -e TRITON_CACHE_DIR="$ARM/.triton" \
  -e TORCHINDUCTOR_CACHE_DIR="$ARM/.inductor" \
  -e HSA_ENABLE_COREDUMP=0 \
  -e AMDGCN_USE_BUFFER_OPS=0 \
  dhk-gemmrs bash -lc "timeout --signal=TERM $TMO setsid python3 -u eval.py $MODE $CASES 3>$ARM/$MODE.popcorn.txt >$ARM/$MODE.stdout.txt 2>$ARM/$MODE.stderr.txt"
echo "exit=$?"

echo
echo "===== popcorn output ($ARMNAME $MODE) ====="
cat "$ARM/$MODE.popcorn.txt" 2>/dev/null

echo
echo "===== faults / errors ====="
grep -E 'Memory access fault|SHIM_WAS_CALLED|Traceback|Error|error:' "$ARM/$MODE.stderr.txt" 2>/dev/null | sort | uniq -c | sort -rn | head -15
echo "--- stderr tail ---"
tail -20 "$ARM/$MODE.stderr.txt" 2>/dev/null

echo
echo "===== stdout tail ====="
tail -25 "$ARM/$MODE.stdout.txt" 2>/dev/null
echo "===== DONE ====="

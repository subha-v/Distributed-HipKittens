#!/usr/bin/env bash
# Benchmark the frozen rank-1 MI300X submission (and the reference GEMM+RCCL
# implementation) with the OFFICIAL evaluator on this node, in the graded
# protocol, on the six graded benchmark shapes.
#
# This requires no new engineering: eval.py/task.py/utils.py/reference.py are
# taken verbatim from the working directory that exp026 already ran to 11/11 on
# this machine, and only submission.py is swapped.
#
# Purpose is calibration: the published 413.139 us for rank 1 came from GPU
# MODE's machine, and nothing we have measured is comparable to it until rank 1
# is timed here.
set -uo pipefail

SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
ROOT=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/compbench
mkdir -p "$ROOT"

# The six graded benchmark shapes, verbatim from task.yml `benchmarks:`.
write_bench_cases() {
  cat > "$1" <<'EOF'
world_size: 8; m: 64; n: 7168; k: 18432; has_bias: False; seed: 1234
world_size: 8; m: 512; n: 4096; k: 12288; has_bias: True; seed: 663
world_size: 8; m: 2048; n: 2880; k: 2880; has_bias: True; seed: 166
world_size: 8; m: 4096; n: 4096; k: 4096; has_bias: False; seed: 1371
world_size: 8; m: 8192; n: 4096; k: 14336; has_bias: True; seed: 7168
world_size: 8; m: 8192; n: 8192; k: 29568; has_bias: False; seed: 42
EOF
}

setup_arm() {
  local name="$1" submission="$2"
  local dir="$ROOT/$name"
  rm -rf "$dir"; mkdir -p "$dir"
  cp "$SRC/eval.py" "$SRC/task.py" "$SRC/utils.py" "$SRC/reference.py" "$dir/"
  cp "$submission" "$dir/submission.py"
  write_bench_cases "$dir/cases_bench.txt"
  cp "$SRC/cases.txt" "$dir/cases_test.txt" 2>/dev/null || true
  echo "$dir"
}

run_arm() {
  local name="$1" mode="$2" cases="$3" timeout_s="$4"
  local dir="$ROOT/$name"
  echo
  echo "################################################################"
  echo "# $name : eval.py $mode $cases"
  echo "################################################################"
  ( cd "$dir" && \
    POPCORN_FD=3 POPCORN_GPUS=8 \
    TRITON_CACHE_DIR="$dir/.triton" \
    TORCHINDUCTOR_CACHE_DIR="$dir/.inductor" \
    timeout "$timeout_s" python3 eval.py "$mode" "$cases" \
      3>"$dir/$mode.popcorn.txt" >"$dir/$mode.stdout.txt" 2>"$dir/$mode.stderr.txt" )
  local status=$?
  echo "exit=$status"
  echo "--- popcorn output ---"
  cat "$dir/$mode.popcorn.txt" 2>/dev/null
  if [ "$status" != "0" ]; then
    echo "--- stderr tail ---"
    tail -25 "$dir/$mode.stderr.txt" 2>/dev/null
    echo "--- stdout tail ---"
    tail -15 "$dir/$mode.stdout.txt" 2>/dev/null
  fi
  return $status
}

echo "===== toolchain ====="
python3 -c "import torch, sys; print('torch', torch.__version__, 'gpus', torch.cuda.device_count())"
python3 -c "import triton; print('triton', triton.__version__)" 2>&1 | tail -1

echo
echo "===== rank-1 submission identity ====="
sha256sum "$RANK1"
echo "expected 7940fcb81df06c1d8b1e1a77051f23c934149a688441ef48b2751b3f336f0dc5 (MI300X_PROVENANCE.md)"

setup_arm rank1 "$RANK1" >/dev/null
setup_arm reference "$SRC/submission.py" >/dev/null
echo "arms staged under $ROOT"

# Correctness first: a benchmark number for a wrong kernel is meaningless.
run_arm rank1 test cases_test.txt 900
run_arm rank1 benchmark cases_bench.txt 1200

run_arm reference benchmark cases_bench.txt 1200

echo
echo "===== DONE ====="

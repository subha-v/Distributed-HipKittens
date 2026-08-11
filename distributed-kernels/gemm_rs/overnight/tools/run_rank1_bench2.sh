#!/usr/bin/env bash
# Benchmark the frozen rank-1 MI300X submission with the official evaluator.
#
# Compatibility repairs, all outside the frozen file (see compat/sitecustomize.py
# and the header of run_rank1_bench.sh):
#   1. iris staged at the literal python3.10 path rank-1 reads, and on PYTHONPATH
#   2. no-op `sudo` so rank-1's sed against iris/__init__.py cannot fire
#   3. sitecustomize supplies triton's missing wrap_handle_tensor_descriptor as a
#      stub that raises if called (it is only called for tensordesc args)
#
# Order matters: the first pass warms the Triton JIT cache. eval.py's test mode
# hardcodes a 60 s per-rank timeout, which a cold Triton compile of this kernel
# exceeds, so correctness is checked only after the cache is warm.
set -uo pipefail

NAME=dhk-eval
STAGE=/usr/local/lib/python3.10/dist-packages
DIR=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/compbench/rank1
COMPAT=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/tools/compat

run() { docker exec "$NAME" bash -c "$1"; }

ENVS="PATH=/usr/local/shim:\$PATH PYTHONPATH=$COMPAT:$STAGE POPCORN_FD=3 POPCORN_GPUS=8 \
TRITON_CACHE_DIR=$DIR/.triton TORCH_EXTENSIONS_DIR=$DIR/.ext TMPDIR=$DIR/.tmp \
COMPAT_SHIM_VERBOSE=1"

echo "===== confirm the shim installs and triton still works ====="
run "PYTHONPATH=$COMPAT:$STAGE COMPAT_SHIM_VERBOSE=1 python3 -c \"
from triton.backends.amd import driver
print('wrap_handle_tensor_descriptor present:',
      hasattr(driver,'wrap_handle_tensor_descriptor'))
import triton; print('triton', triton.__version__)
import iris; print('iris ok')
\""

go() {
  local mode="$1" cases="$2" label="$3" tmo="$4"
  echo
  echo "################################################################"
  echo "# rank1 $label : eval.py $mode $cases"
  echo "################################################################"
  run "mkdir -p $DIR/.tmp && cd $DIR && $ENVS timeout $tmo python3 eval.py $mode $cases \
       3>$DIR/$label.popcorn.txt >$DIR/$label.stdout.txt 2>$DIR/$label.stderr.txt"
  echo "exit=$?"
  echo "--- popcorn ---"
  run "cat $DIR/$label.popcorn.txt 2>/dev/null | head -80"
  echo "--- shim called? (must be empty) ---"
  run "grep -c SHIM_WAS_CALLED $DIR/$label.stderr.txt 2>/dev/null || true"
  echo "--- stderr tail ---"
  run "tail -12 $DIR/$label.stderr.txt 2>/dev/null"
}

# Pass 1: warm the Triton cache. Failures here are expected and not reported.
go benchmark cases_bench.txt warm 1800
# Pass 2: correctness, now that compilation is cached.
go test cases_test.txt test 1800
# Pass 3: the number.
go benchmark cases_bench.txt bench 1800

echo
echo "===== DONE ====="

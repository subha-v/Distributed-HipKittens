#!/usr/bin/env bash
# Benchmark the frozen rank-1 MI300X submission with the official evaluator,
# with the three disclosed compatibility repairs applied:
#   1. iris staged at the literal python3.10 path it reads, and on PYTHONPATH
#   2. no-op `sudo` so its sed against iris/__init__.py cannot fire
#   3. compat/sitecustomize.py supplies triton's wrap_handle_tensor_descriptor
#      as a stub that raises if called (dead branch for these kernels)
#   4. patch_rank1.py fixes the packed_metadata field count (6 -> 3), which is
#      behaviour-preserving because the dropped fields are never read
#
# Pass order matters: the first pass warms the Triton JIT cache, because
# eval.py's test mode hardcodes a 60 s per-rank timeout that a cold compile of
# this kernel exceeds.
set -uo pipefail

NAME=dhk-eval
STAGE=/usr/local/lib/python3.10/dist-packages
NODE=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
DIR=$NODE/compbench/rank1
COMPAT=$NODE/compat
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py

run() { docker exec "$NAME" bash -c "$1"; }

echo "===== stage arm + apply the metadata repair ====="
run "rm -rf $DIR && mkdir -p $DIR/.tmp && cp $SRC/eval.py $SRC/task.py $SRC/utils.py $SRC/reference.py $DIR/ && cp $SRC/cases.txt $DIR/cases_test.txt"
run "python3 $NODE/patch_rank1.py $RANK1 $DIR/submission.py"
if [ $? -ne 0 ]; then echo "patch failed - stopping"; exit 1; fi

run "cat > $DIR/cases_bench.txt <<'EOF'
world_size: 8; m: 64; n: 7168; k: 18432; has_bias: False; seed: 1234
world_size: 8; m: 512; n: 4096; k: 12288; has_bias: True; seed: 663
world_size: 8; m: 2048; n: 2880; k: 2880; has_bias: True; seed: 166
world_size: 8; m: 4096; n: 4096; k: 4096; has_bias: False; seed: 1371
world_size: 8; m: 8192; n: 4096; k: 14336; has_bias: True; seed: 7168
world_size: 8; m: 8192; n: 8192; k: 29568; has_bias: False; seed: 42
EOF
echo staged"

ENVS="PATH=/usr/local/shim:\$PATH PYTHONPATH=$COMPAT:$STAGE POPCORN_FD=3 POPCORN_GPUS=8 \
TRITON_CACHE_DIR=$DIR/.triton TORCH_EXTENSIONS_DIR=$DIR/.ext TMPDIR=$DIR/.tmp"

go() {
  local mode="$1" cases="$2" label="$3"
  echo
  echo "################################################################"
  echo "# rank1 [$label] : eval.py $mode $cases"
  echo "################################################################"
  run "cd $DIR && $ENVS timeout 1700 python3 eval.py $mode $cases \
       3>$DIR/$label.popcorn.txt >$DIR/$label.stdout.txt 2>$DIR/$label.stderr.txt"
  echo "exit=$?"
  echo "--- popcorn ---"
  run "cat $DIR/$label.popcorn.txt 2>/dev/null | head -90"
  echo "--- shim invoked? (must print 0) ---"
  run "grep -c SHIM_WAS_CALLED $DIR/$label.stderr.txt 2>/dev/null || echo 0"
  echo "--- stderr tail ---"
  run "tail -14 $DIR/$label.stderr.txt 2>/dev/null"
}

go benchmark cases_bench.txt warm
go test cases_test.txt test
go benchmark cases_bench.txt bench

echo
echo "===== DONE ====="

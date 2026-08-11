#!/usr/bin/env bash
# The reference GEMM+RCCL baseline through the SAME evaluator, in the SAME
# container, on the SAME shapes as our arm.
#
# Why this matters more than the absolute numbers: our own evaluator run showed
# 12-93% relative standard deviations, so no single arm's absolute number is
# trustworthy. But both arms pay the same protocol overhead -- the barrier pair,
# the host issue, the per-iteration _clone_data of the inputs, and the ~100 us
# launch skew -- so the RATIO between two arms measured the same way is
# meaningful even when neither absolute number is.
#
# The reference submission is exp026's own submission.py: plain torch.matmul +
# torch.distributed.reduce_scatter_tensor. It is the naive fused-free baseline
# and beating it is the minimum bar.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
DIR=$ON/compbench/reference

run() { docker exec dhk-gemmrs bash -c "$1"; }

echo "===== preflight: node must be clean ====="
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids: $n"
if [ "$n" != "0" ]; then echo "ABORT: node dirty"; exit 1; fi

echo "===== stage the reference arm ====="
run "rm -rf $DIR && mkdir -p $DIR/.tmp && cp $SRC/eval.py $SRC/task.py $SRC/utils.py $SRC/reference.py $SRC/submission.py $DIR/ && cp $SRC/cases.txt $DIR/cases_test.txt && ls $DIR"

run "cat > $DIR/cases_bench.txt <<'EOF'
world_size: 8; m: 64; n: 7168; k: 18432; has_bias: False; seed: 1234
world_size: 8; m: 512; n: 4096; k: 12288; has_bias: True; seed: 663
world_size: 8; m: 2048; n: 2880; k: 2880; has_bias: True; seed: 166
world_size: 8; m: 4096; n: 4096; k: 4096; has_bias: False; seed: 1371
world_size: 8; m: 8192; n: 4096; k: 14336; has_bias: True; seed: 7168
world_size: 8; m: 8192; n: 8192; k: 29568; has_bias: False; seed: 42
EOF
echo staged"

echo "===== what IS the reference submission? ====="
run "head -40 $DIR/submission.py"

ENVS="POPCORN_FD=3 POPCORN_GPUS=8 TMPDIR=$DIR/.tmp TORCH_EXTENSIONS_DIR=$DIR/.ext"

go() {
  local mode="$1" cases="$2" tmo="$3"
  echo
  echo "################################################################"
  echo "# reference : eval.py $mode $cases"
  echo "################################################################"
  run "cd $DIR && $ENVS timeout $tmo python3 eval.py $mode $cases \
       3>$DIR/$mode.popcorn.txt >$DIR/$mode.stdout.txt 2>$DIR/$mode.stderr.txt"
  echo "exit=$?"
  echo "--- popcorn ---"
  run "cat $DIR/$mode.popcorn.txt 2>/dev/null"
  echo "--- stderr tail ---"
  run "tail -15 $DIR/$mode.stderr.txt 2>/dev/null"
}

go test cases_test.txt 900
go benchmark cases_bench.txt 1500

echo
echo "===== DONE ====="

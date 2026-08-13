#!/usr/bin/env bash
# Run OUR kernel under the official evaluator, in the graded protocol:
# one process per rank, torch.distributed, HIP IPC symmetric heap.
#
# This is the like-for-like measurement. Correctness (test mode) runs first; a
# benchmark number for a wrong kernel is worthless.
set -uo pipefail

NAME=dhk-gemmrs           # our own container, runs as the invoking user
NODE=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
DIR=$NODE/compbench/ours
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd

run() { docker exec "$NAME" bash -c "$1"; }

echo "===== can HIP IPC share each allocation flavour? ====="
run "cd $NODE/harness && python3 -c \"
import importlib.util
spec = importlib.util.spec_from_file_location('dhk_rt', 'build/dhk_rt.so')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(m.ipc_probe(0))
\""

echo
echo "===== stage the arm ====="
run "rm -rf $DIR && mkdir -p $DIR/.tmp && cp $SRC/eval.py $SRC/task.py $SRC/utils.py $SRC/reference.py $DIR/ && cp $SRC/cases.txt $DIR/cases_test.txt && cp $NODE/harness/hk_submission.py $DIR/submission.py && ls $DIR"

run "cat > $DIR/cases_bench.txt <<'EOF'
world_size: 8; m: 64; n: 7168; k: 18432; has_bias: False; seed: 1234
world_size: 8; m: 512; n: 4096; k: 12288; has_bias: True; seed: 663
world_size: 8; m: 2048; n: 2880; k: 2880; has_bias: True; seed: 166
world_size: 8; m: 4096; n: 4096; k: 4096; has_bias: False; seed: 1371
world_size: 8; m: 8192; n: 4096; k: 14336; has_bias: True; seed: 7168
world_size: 8; m: 8192; n: 8192; k: 29568; has_bias: False; seed: 42
EOF
echo staged"

# HK_DEBUG defaults to 0 as of aug13/exp_01. It defaulted to 1, and exp_24
# §12's evaluator "ours" arm inherited it: DEBUG disables hk_submission's
# exp_12 fast path and adds a per-call synchronize + blocking D2H error-bit
# read + ~5 flushed stderr lines per rank INSIDE the evaluator's timed region.
# That self-inflicted tax was most of the "arm-dependent inflation" that put
# ours at 578.7 us geomean (1.6159x vs rank-1); with the fast path live the
# same ladder reads ~397 us and beats reference (aug13/exp_01). Set HK_DEBUG=1
# explicitly only when debugging a hang, and never quote such a run's numbers.
ENVS="POPCORN_FD=3 POPCORN_GPUS=8 HK_BUILD_DIR=$NODE/harness/build \
HK_DEBUG=${HK_DEBUG:-0} TMPDIR=$DIR/.tmp TORCH_EXTENSIONS_DIR=$DIR/.ext"

go() {
  local mode="$1" cases="$2" tmo="$3"
  echo
  echo "################################################################"
  echo "# ours : eval.py $mode $cases"
  echo "################################################################"
  run "cd $DIR && $ENVS timeout $tmo python3 eval.py $mode $cases \
       3>$DIR/$mode.popcorn.txt >$DIR/$mode.stdout.txt 2>$DIR/$mode.stderr.txt"
  echo "exit=$?"
  echo "--- popcorn ---"
  run "cat $DIR/$mode.popcorn.txt 2>/dev/null | head -80"
  echo "--- stderr tail ---"
  run "tail -25 $DIR/$mode.stderr.txt 2>/dev/null"
}

if [ "${HK_ONE:-0}" = "1" ]; then
  # Single graded shape, full stderr: the fastest way to see where setup stalls.
  run "head -1 $DIR/cases_bench.txt > $DIR/cases_one.txt; cat $DIR/cases_one.txt"
  go benchmark cases_one.txt 600
  echo "--- FULL stderr ---"
  run "cat $DIR/benchmark.stderr.txt 2>/dev/null | head -120"
else
  go test cases_test.txt 900
  go benchmark cases_bench.txt 1500
fi

echo
echo "===== DONE ====="

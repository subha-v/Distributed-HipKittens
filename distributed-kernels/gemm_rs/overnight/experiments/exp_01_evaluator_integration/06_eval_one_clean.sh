#!/usr/bin/env bash
# exp_01 step 2b: single graded shape under eval.py, staged and invoked from
# HERE rather than through tools/run_ours_evaluator.sh, because the node copy of
# that script has Windows CRLF endings: its trailing if/else/fi is corrupted, so
# it ran the HK_ONE branch AND the full-suite branch, and the second run
# overwrote benchmark.stderr.txt and destroyed the diagnostics.
#
# HK_DEBUG=1: diagnostic run, no number here is reportable.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
DIR=$ON/compbench/ours
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
OUT=$ON/experiments/exp_01_evaluator_integration/logs
mkdir -p $OUT

echo "===== how does eval.py warm up? (read-only) ====="
sed -n '495,575p' $DIR/eval.py

echo
echo "===== stage the arm ====="
docker exec dhk-gemmrs bash -c "
  rm -rf $DIR && mkdir -p $DIR/.tmp &&
  cp $SRC/eval.py $SRC/task.py $SRC/utils.py $SRC/reference.py $DIR/ &&
  cp $SRC/cases.txt $DIR/cases_test.txt &&
  cp $ON/harness/hk_submission.py $DIR/submission.py &&
  head -1 $ON/tools/../compbench_cases_bench.txt 2>/dev/null;
  printf '%s\n' 'world_size: 8; m: 64; n: 7168; k: 18432; has_bias: False; seed: 1234' > $DIR/cases_one.txt &&
  ls $DIR && cat $DIR/cases_one.txt"

echo
echo "===== run: eval.py benchmark cases_one.txt, HK_DEBUG=1 ====="
docker exec -w $DIR dhk-gemmrs bash -c "
  POPCORN_FD=3 POPCORN_GPUS=8 HK_BUILD_DIR=$ON/harness/build HK_DEBUG=1 \
  HK_DUMP_AFTER=60 TMPDIR=$DIR/.tmp TORCH_EXTENSIONS_DIR=$DIR/.ext \
  timeout 400 python3 -u eval.py benchmark cases_one.txt \
    3>$OUT/one.popcorn.txt >$OUT/one.stdout.txt 2>$OUT/one.stderr.txt"
echo "eval rc=$?"

echo
echo "===== popcorn ====="
cat $OUT/one.popcorn.txt 2>&1

echo
echo "===== stdout ====="
tail -30 $OUT/one.stdout.txt 2>&1

echo
echo "===== stderr size ====="
wc -l $OUT/one.stderr.txt 2>&1

echo
echo "===== step counts across all ranks ====="
for step in 'enter custom_kernel' 'cache vote' 'cache hit' 'cache miss' \
            'allocated c=' 'c_heap: get_ipc_handle done' 'c_heap: all_gather_object done' \
            'c_heap: opened all peers' 'signals: get_ipc_handle done' \
            'signals: all_gather_object done' 'signals: opened all peers' \
            'descriptors uploaded' 'setup barrier returned' \
            'device synchronize returned' 'state ready' 'launch issued' \
            'launch synchronized' 'ERROR BITS' 'FAILED peer' 'watchdog armed'; do
  printf '%-34s %s\n' "$step" "$(grep -c "$step" $OUT/one.stderr.txt 2>/dev/null)"
done

echo
echo "===== cache vote lines ====="
grep 'cache vote' $OUT/one.stderr.txt 2>/dev/null | head -20

echo
echo "===== LAST logged step per pid (this localizes the hang) ====="
grep -o '^\[hk [0-9.]* pid [0-9]* rank [0-9-]*\] .*' $OUT/one.stderr.txt 2>/dev/null \
  | awk '{ pid=$4; rank=$6; sub(/\]/,"",rank);
           line=$0; last[pid]=line; }
         END { for (p in last) print last[p] }' | sort -t' ' -k4 -n

echo
echo "===== ranks per distinct final step ====="
grep -o '^\[hk [0-9.]* pid [0-9]* rank [0-9-]*\] .*' $OUT/one.stderr.txt 2>/dev/null \
  | awk '{ pid=$4; $1="";$2="";$3="";$4="";$5="";$6=""; last[pid]=$0 } END { for (p in last) print last[p] }' \
  | sed 's/^ *//' | sort | uniq -c | sort -rn

echo
echo "===== IPC failures / tracebacks ====="
grep -n 'FAILED peer\|AlreadyMapped\|hipIpc\|Traceback\|Error\|error' $OUT/one.stderr.txt 2>/dev/null | head -25

echo
echo "===== faulthandler dumps (where is each rank blocked?) ====="
grep -n 'Timeout (0:\|hk_submission.py\|eval.py, line\|distributed_c10d\|line [0-9]* in ' \
  $OUT/one.stderr.txt 2>/dev/null | tail -60

echo
echo "===== STEP2B DONE ====="

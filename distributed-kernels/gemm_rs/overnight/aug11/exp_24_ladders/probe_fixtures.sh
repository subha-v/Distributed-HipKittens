#!/usr/bin/env bash
# READ-ONLY: dump the three saved popcorn/stdout fixtures verbatim so the parser
# can be written against real evaluator text rather than a guess, plus the few
# extra facts probe_stage.sh could not resolve.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight

for arm in ours reference rank1; do
  echo "########## FIXTURE $arm : benchmark.popcorn.txt ##########"
  cat "$ON/compbench/$arm/benchmark.popcorn.txt" 2>&1
  echo "########## FIXTURE $arm : test.popcorn.txt ##########"
  cat "$ON/compbench/$arm/test.popcorn.txt" 2>&1 | sed -n '1,12p'
  echo
done

echo "########## eval.py: how benchmark results are emitted ##########"
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
grep -nE 'benchmark-|geomean|def _print|POPCORN_FD|print_?result|log\(|def run_benchmark|duration|best|mean|worst|number_of_runs' "$SRC/eval.py" 2>&1 | sed -n '1,60p'

echo
echo "########## utils.py: the timed region (graded protocol) ##########"
grep -nE 'def benchmark|def _run|perf_counter|synchronize|barrier|clear_l2|clone|Stats|best|mean|def metric' "$SRC/utils.py" 2>&1 | sed -n '1,60p'

echo
echo "########## does /usr/local/shim exist in dhk-eval? (bench3 PATH) ##########"
docker exec dhk-eval bash -lc 'ls -l /usr/local/shim 2>&1; which sudo 2>&1' 2>&1 | sed -n '1,8p'

echo
echo "########## release-granularity / reducer knob names in the sources ##########"
grep -rnoE '[A-Z_]*RELEASE[A-Z_]*|[A-Z_]*REDUCER[A-Z_]*|WGM[A-Z_]*|[A-Z_]*GROUP[A-Z_]*' \
  "$ON/../gemm_rs_mi300x_constants.cuh" "$ON/../gemm_rs_mi300x_host_abi.hpp" 2>&1 \
  | awk -F: '{print $3}' | sort -u | sed -n '1,30p'

echo
echo "########## exp_10 vs_report.py (the aggregator I must stay compatible with) ##########"
sed -n '1,80p' "$ON/experiments/exp_10_rank1/vs_report.py" 2>&1

echo "########## DONE ##########"

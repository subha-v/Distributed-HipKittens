#!/usr/bin/env bash
# Our kernel under the official evaluator, with HK_DEBUG=0.
#
# run_ours_evaluator.sh defaults HK_DEBUG to 1, which puts a
# torch.cuda.synchronize() and unbuffered stderr writes INSIDE every timed
# call. That is a debugging aid, never a measurement. Any number that goes into
# RESULTS.md must come from this wrapper.
#
# Order matters: test mode first. Benchmark mode runs with recheck=False and
# therefore never re-verifies correctness -- a previous session recorded
# 303/463/2099/3043/21509 us from a run where most ranks were operating on
# another device's buffers, and benchmark mode timed it happily.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight

echo "===== preflight: node must be clean ====="
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids: $n"
if [ "$n" != "0" ]; then echo "ABORT: node dirty"; exit 1; fi

export HK_DEBUG=0
echo "HK_DEBUG=$HK_DEBUG  (0 = measurement mode)"
bash $ON/tools/run_ours_evaluator.sh

echo
echo "===== benchmark popcorn, full ====="
docker exec dhk-gemmrs bash -c "cat $ON/compbench/ours/benchmark.popcorn.txt 2>/dev/null"
echo
echo "===== test popcorn, full ====="
docker exec dhk-gemmrs bash -c "cat $ON/compbench/ours/test.popcorn.txt 2>/dev/null"
echo "===== DONE ====="

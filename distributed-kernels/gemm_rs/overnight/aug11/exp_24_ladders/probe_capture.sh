#!/usr/bin/env bash
# rot0/rank1/ contains benchmark.popcorn.txt, which is run_ours_evaluator.sh's
# naming, not run_rank1_bench3.sh's warm/test/bench. Either the arms are being
# captured into the wrong directories -- which would mislabel the cross-check, the
# worst possible failure here -- or bench3 has not run yet and I am looking at
# stale files that were staged into compbench/rank1 by an earlier experiment.
# Read-only; phase B is live, do not disturb it.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders

echo "===== arm order actually executed in rot0 ====="
grep -nE 'instrument B rot=|captured |===== DONE' "$D/logs/full_run.log" | tail -20

echo
echo "===== what is in each capture dir, with sizes ====="
find "$D/raw/eval" -type f 2>/dev/null | sort | while read -r f; do
  echo "  $(stat -c%s "$f" | xargs printf '%8d') ${f#$D/raw/eval/}"
done

echo
echo "===== does rot0/rank1/benchmark.popcorn.txt look like OURS or RANK-1? ====="
echo "--- head of the captured popcorn ---"
head -12 "$D/raw/eval/rot0/rank1/benchmark.popcorn.txt" 2>/dev/null
echo "--- which kernel does its stdout name? ---"
grep -ohE 'hk_submission|HipKittens|\[hk |triton|iris|rank1|submission' \
  "$D/raw/eval/rot0/rank1/benchmark.stdout.txt" 2>/dev/null | sort | uniq -c | sort -rn | head -8

echo
echo "===== where does each driver stage? (the ground truth) ====="
for t in run_ours_evaluator.sh run_reference_arm.sh run_rank1_bench3.sh; do
  echo "--- $t"
  grep -nE '^DIR=|^NAME=|compbench/' "$D/toolsnap/$t" | head -4
done

echo
echo "===== stale files still sitting in compbench/* ====="
for a in ours reference rank1; do
  echo "  $a: $(ls "$ON/compbench/$a"/*.popcorn.txt 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
done

echo
echo "===== eval_arm's capture call sites (is dest ever wrong?) ====="
grep -nE 'capture |dest=' "$D/run_ladders.sh"

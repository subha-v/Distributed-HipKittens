#!/usr/bin/env bash
# Decisive facts, read-only. Two questions:
#  1. Does $NODE (in run_ours_evaluator.sh) equal $ON (in my capture())? If not,
#     the `ours` evaluator arm writes somewhere capture() never looks.
#  2. Where did rot0/rank1 come from, when the only arm headers logged so far are
#     `ours` and `reference` and bench3 has demonstrably not run (no warm/bench
#     popcorn anywhere)?
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders

echo "===== 1. NODE vs ON in run_ours_evaluator.sh ====="
grep -nE '^NODE=|^ON=|^DIR=' "$D/toolsnap/run_ours_evaluator.sh"
echo "my CB in run_ladders.sh:"; grep -nE '^CB=|^ON=' "$D/run_ladders.sh"

echo
echo "===== 2. every 'instrument B' line in the log, no tail ====="
grep -nE 'instrument B rot=' "$D/logs/full_run.log"

echo
echo "===== 3. timestamps: when were rot0/* created? ====="
ls -la --time-style=full-iso "$D/raw/eval/rot0/" 2>/dev/null
for d in "$D/raw/eval/rot0"/*; do
  echo "--- ${d##*/}"; ls -la --time-style=full-iso "$d" 2>/dev/null | sed 1d
done

echo
echo "===== 4. empty capture dirs (find -type f hides these) ====="
find "$D/raw/eval" -type d 2>/dev/null | sed "s|$D/raw/eval|  raw/eval|"

echo
echo "===== 5. where did the ours evaluator ACTUALLY write? ====="
for p in /home/subvadla/dhk/compbench/ours $ON/compbench/ours; do
  echo "  $p : $(ls "$p"/*.popcorn.txt 2>/dev/null | wc -l) popcorn files"
  ls -la --time-style=full-iso "$p"/*.popcorn.txt 2>/dev/null | sed 's/^/     /'
done

echo
echo "===== 6. is rot0/rank1's popcorn identical to the ours evaluator output? ====="
for p in /home/subvadla/dhk/compbench/ours/benchmark.popcorn.txt \
         $ON/compbench/ours/benchmark.popcorn.txt \
         $ON/compbench/rank1/benchmark.popcorn.txt \
         "$D/raw/eval/rot0/rank1/benchmark.popcorn.txt"; do
  [ -f "$p" ] && echo "  $(sha256sum "$p" | cut -c1-16)  $(stat -c%y "$p" | cut -c1-19)  $p"
done

echo
echo "===== 7. the capture/eval_arm block verbatim ====="
sed -n '295,365p' "$D/run_ladders.sh"

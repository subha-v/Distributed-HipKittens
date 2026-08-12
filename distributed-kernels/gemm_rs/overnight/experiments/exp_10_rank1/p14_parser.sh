#!/usr/bin/env bash
# exp_10 probe 14 -- NO GPU (grep only).
# Hypothesis that resolves the contradiction: eval.py parses `has_bias: False`
# from the cases file into the STRING "False", which is truthy, so the official
# evaluator generates a bias for ALL SIX graded shapes -- including the three
# declared has_bias: False. That would explain both rank-1's source comment
# ("bench all have bias?") and why eval.py never hits its own bias-is-None bug.
set -uo pipefail

ARM=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/compbench/rank1

echo "===== A. eval.py get_test_cases / _combine: how are values typed? ====="
awk 'NR>=44 && NR<=96 {printf "%d: %s\n", NR, $0}' "$ARM/eval.py"

echo
echo "===== B. decide it empirically ====="
docker exec -w "$ARM" dhk-gemmrs bash -lc 'python3 -u -c "
import eval as E
cases = E.get_test_cases(\"cases_bench.txt\", None)
for i, t in enumerate(cases):
    a = t.args
    hb = a.get(\"has_bias\")
    print(f\"  case {i}: m={a.get(\\\"m\\\")!r:>7} has_bias={hb!r:<9} type={type(hb).__name__:<5} bool()={bool(hb)}\")
print()
print(\"=> if type is str, EVERY graded shape gets a bias under this evaluator\")
"' 2>&1 | tail -15

echo
echo "===== C. and what does reference.generate_input then produce? ====="
docker exec -w "$ARM" dhk-gemmrs bash -lc 'python3 -u -c "
import eval as E
from reference import generate_input
cases = E.get_test_cases(\"cases_bench.txt\", None)
import torch
for i, t in enumerate(cases):
    a = dict(t.args)
    a.pop(\"world_size\", None)
    # do not touch the GPU: just re-implement the branch decision
    print(f\"  case {i}: has_bias={t.args.get(\\\"has_bias\\\")!r} -> bias tensor created? {bool(t.args.get(\\\"has_bias\\\"))}\")
"' 2>&1 | tail -10
echo "===== DONE p14 ====="

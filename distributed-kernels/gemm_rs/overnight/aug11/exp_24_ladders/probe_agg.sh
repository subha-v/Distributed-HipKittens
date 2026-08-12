#!/usr/bin/env bash
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders
echo "===== full aggregation output + traceback, nothing truncated ====="
docker exec dhk-gemmrs bash -lc \
  "cd $D && python3 ladders.py --root $D --out $D/ladders.json --ladder-dir ladder --expect-a 6 --expect-b 1; echo RC=\$?" 2>&1 | tail -45
echo
echo "===== did it write? ====="
ls -la --time-style=full-iso "$D/ladders.json"
docker exec dhk-gemmrs python3 -c "
import json; d=json.load(open('$D/ladders.json'))
print('rotations now:', list(d['evaluator_crosscheck']['rotations'].keys()))
"

#!/usr/bin/env bash
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders
docker exec dhk-gemmrs python3 -c "
import json
d=json.load(open('$D/ladders.json'))
print('top-level keys:', sorted(d.keys()))
for k in d:
    if 'cross' in k or 'eval' in k or 'instrument' in k:
        print(k, '->', json.dumps(d[k])[:600])
"
echo
echo "===== does build_crosscheck get called, and with what root? ====="
grep -n 'build_crosscheck\|evaluator_crosscheck\|raw..eval\|\"eval\"' "$D/ladders.py"

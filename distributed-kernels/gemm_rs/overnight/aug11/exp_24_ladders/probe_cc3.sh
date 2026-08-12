#!/usr/bin/env bash
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders
echo "===== call build_crosscheck directly ====="
docker exec dhk-gemmrs python3 -c "
import sys, json; sys.path.insert(0,'$D')
import ladders
cc = ladders.build_crosscheck('$D', False)
print('rotations:', list(cc['rotations'].keys()))
for rot, arms in cc['rotations'].items():
    for k,v in arms.items(): print('  ', rot, k, v.get('geomean_us'), v.get('check'))
print('errors:', cc['parse_errors'])
print('skipped:', cc['skipped_test_mode'])
" 2>&1 | tail -25

echo
echo "===== build_crosscheck source ON THE NODE ====="
sed -n '435,480p' "$D/ladders.py"

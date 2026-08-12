#!/usr/bin/env bash
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders
echo "===== glob from inside the container ====="
docker exec dhk-gemmrs python3 -c "
import glob, os
D='$D'
for pat in ['raw/eval/rot*/*/*.popcorn.txt','raw/eval/*/*/*.popcorn.txt','raw/eval/**/*.popcorn.txt']:
    p=os.path.join(D,pat)
    print(pat, '->', len(glob.glob(p, recursive=True)))
print('listdir raw:', sorted(os.listdir(os.path.join(D,'raw'))))
print('listdir raw/eval:', sorted(os.listdir(os.path.join(D,'raw','eval'))))
print('listdir raw/eval/rot0:', sorted(os.listdir(os.path.join(D,'raw','eval','rot0'))))
"
echo
echo "===== how is --root parsed / normalized? ====="
grep -n 'add_argument\|args.root' "$D/ladders.py" | head -20
echo
echo "===== the lines around the call site ====="
sed -n '540,560p' "$D/ladders.py"

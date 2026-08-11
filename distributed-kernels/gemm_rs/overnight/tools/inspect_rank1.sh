#!/usr/bin/env bash
# What does the frozen rank-1 submission need from the `iris` Python package?
set -uo pipefail
R=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py

echo "===== size / imports ====="
wc -l "$R"
grep -nE '^\s*(import|from)\s' "$R" | head -40

echo
echo "===== every mention of iris ====="
grep -n 'iris' "$R" | head -60

echo
echo "===== context around the hardcoded dist-packages path ====="
grep -n -B12 -A25 'dist-packages/iris' "$R" | head -80

echo
echo "===== is the iris python package available anywhere on this box? ====="
for p in /usr/local/lib/python3.10/dist-packages/iris /usr/local/lib/python3.12/dist-packages/iris \
         /usr/lib/python3/dist-packages/iris; do
  [ -e "$p" ] && echo "FOUND $p" || echo "absent $p"
done
echo "-- pip --"
python3 -m pip show iris 2>&1 | head -5
echo "-- any iris checkout in home --"
ls -d /home/subvadla/iris /home/subvadla/*/iris 2>/dev/null | head
echo "-- inside prior campaign containers? --"
ls -d /home/subvadla/ddt-*/container-home/*iris* 2>/dev/null | head

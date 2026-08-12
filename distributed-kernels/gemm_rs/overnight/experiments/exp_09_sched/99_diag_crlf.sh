#!/usr/bin/env bash
# Why did m2_isa.sh stop parsing? Suspect CRLF that push.ps1's sed skipped.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
cd "$ON"
for f in tools/m2_isa.sh tools/m1_build.sh tools/gate_ladder.sh \
         experiments/exp_09_sched/isa_arm.sh experiments/exp_03_mainloop/lds_race_check.sh; do
  printf '%-52s crlf=%-4s owner=%-10s perm=%s\n' "$f" \
    "$(grep -c $'\r' "$f" 2>/dev/null)" \
    "$(stat -c '%U:%G' "$f" 2>/dev/null)" \
    "$(stat -c '%a' "$f" 2>/dev/null)"
done
echo
echo "whoami=$(whoami)  uid=$(id -u)"
echo "-- files under gemm_rs NOT writable by me --"
find /home/subvadla/dhk/distributed-kernels/gemm_rs -type f ! -writable \
  -not -path '*/build/*' -not -path '*/compbench/*' 2>/dev/null | head -20
echo "-- count --"
find /home/subvadla/dhk/distributed-kernels/gemm_rs -type f ! -writable \
  -not -path '*/build/*' -not -path '*/compbench/*' 2>/dev/null | wc -l
echo
echo "-- CRLF survivors --"
grep -rlU $'\r' /home/subvadla/dhk/distributed-kernels/gemm_rs \
  --include='*.sh' --include='*.py' --include='*.cpp' --include='*.cuh' --include='*.hpp' \
  2>/dev/null | head -20

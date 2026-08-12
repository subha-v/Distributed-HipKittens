#!/usr/bin/env bash
# Transported by tools/nsh.ps1. Compilation only -- hipcc needs no GPU, so this does not violate
# the one-GPU-job rule and does not disturb the agent currently holding the devices.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/aug11/exp_21_saturation

echo "########## source freshness ##########"
ls -la "$EXP" | head -20
md5sum "$EXP/sat_ubench.cpp"

echo
docker exec dhk-gemmrs bash "$EXP/build.sh"
rc=$?
echo "build.sh exit=$rc"

if [ "$rc" != "0" ]; then
  echo
  echo "########## first 80 error lines ##########"
  docker exec dhk-gemmrs bash -c "grep -nE 'error:' $EXP/build/sat_ubench.log | head -80"
  echo
  echo "########## context around the first error ##########"
  docker exec dhk-gemmrs bash -c "grep -n -A4 -B2 'error:' $EXP/build/sat_ubench.log | head -60"
fi
exit $rc

#!/usr/bin/env bash
# CPU-only preflight: build the four arm modules and prove the PERSHAPE define
# reached codegen. No GPU is touched, so this is safe to run at any time.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
mkdir -p "$D/logs"

echo "########## source check: the two macros as they will compile ##########"
grep -n 'RELEASE_GROUP_PERSHAPE\|rcap\|rgroup =' \
  /home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp \
  | grep -v '^ *//' | head -20

echo
docker exec dhk-gemmrs bash "$D/build_arms.sh" 2>&1 | tee "$D/logs/build_arms.log"
grep -q 'ARM MODULES BUILT' "$D/logs/build_arms.log" || {
  echo "ABORT: arm build failed"; exit 1; }

echo
docker exec dhk-gemmrs bash "$D/isa_diff.sh" 2>&1 | tee "$D/logs/isa_diff.log"
isa_rc=${PIPESTATUS[0]}
if [ "$isa_rc" != "0" ]; then
  echo "ABORT: isa_diff.sh exited $isa_rc (a preflight that cannot run is not a pass)"
  exit 1
fi
grep -q 'did NOT reach codegen' "$D/logs/isa_diff.log" && {
  echo "ABORT: PERSHAPE did not reach codegen"; exit 1; }
grep -q 'DONE' "$D/logs/isa_diff.log" || {
  echo "ABORT: isa_diff.sh did not reach its end"; exit 1; }

echo
echo "BUILD PREFLIGHT OK"

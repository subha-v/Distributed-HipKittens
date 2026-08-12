#!/usr/bin/env bash
# exp_13 step 4: the landed table against uniform NR=32, same run, interleaved,
# rotated, with a null twin -- the only valid denominator for the change.
#
# This exists because M7's cross-session comparison is contaminated: shape 5
# read 658 us today against a recorded 613.70 with an IDENTICAL configuration
# (its NR did not change and the device ISA is byte-identical), so the
# session-to-session geomean understates the change. Here both tables are
# measured in one process. On shapes 2-5 the two arms are the same
# configuration, which makes them four extra null arms.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_13_cta_split
L=$D/logs
mkdir -p "$L"
STAMP=$(date +%H%M%S)

for i in $(seq 1 90); do
  n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
  [ "$n" = "0" ] && break
  echo "waiting for $n KFD pid(s) to drain (attempt $i/90)"; sleep 10
done
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
if [ "$n" != "0" ]; then echo "NODE STILL DIRTY ($n pids)"; exit 1; fi
echo "node clean"

docker exec -w $ON/harness dhk-gemmrs timeout 3600 \
  python3 -u $D/sweep.py T,32 12 all 1 2>&1 | tee "$L/tableab_$STAMP.log"

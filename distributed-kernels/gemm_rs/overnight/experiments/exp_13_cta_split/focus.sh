#!/usr/bin/env bash
# exp_13 step 2: focused confirmation of the screening winners, with fewer arms
# and more passes so the null floor tightens and the marginal shapes (3 and 5)
# separate. Two runs because the interesting region differs by shape:
#   A: shapes 1 and 3, where the GEMM is one wave for the whole range, so the
#      curve keeps falling past NR=48 and the pick is inside {56,64,80}.
#   B: shapes 2, 4, 5 and 6, where NR>=56 costs a whole extra producer wave, so
#      only 24..48 can win. Longer blocks (scale 1.5) on these.
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

echo "########## FOCUS A: shapes 1,3 x {32,40,48,56,64,80} x 14 passes ##########"
docker exec -w $ON/harness dhk-gemmrs timeout 3600 \
  python3 -u $D/sweep.py 32,40,48,56,64,80 14 1,3 1 2>&1 | tee "$L/focusA_$STAMP.log"

echo
echo "########## FOCUS B: shapes 2,4,5,6 x {24,32,40,48} x 15 passes, 1.5x blocks ##########"
docker exec -w $ON/harness dhk-gemmrs timeout 3600 \
  python3 -u $D/sweep.py 24,32,40,48 15 2,4,5,6 1.5 2>&1 | tee "$L/focusB_$STAMP.log"

ls -l $D/*.json

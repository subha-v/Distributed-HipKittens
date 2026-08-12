#!/usr/bin/env bash
# The paired run's final restore build failed, so the node's build/*.so may be
# the BASE binary sitting under the candidate's source -- exactly the state that
# makes the next measurement meaningless. Diagnose, then rebuild and prove the
# binary matches the source.
set -uo pipefail
CAND=${1:-b_setprio}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_09_sched
SRC=$ON/..

echo "########## why did the restore build fail? ##########"
tail -30 "$EXP/paired_$CAND/build_restore.log" 2>/dev/null || echo "(no log)"

echo
echo "########## CRLF / parse state of the build script ##########"
printf 'harness/build.sh  crlf=%s\n' "$(grep -c $'\r' $ON/harness/build.sh)"
docker exec dhk-gemmrs bash -n $ON/harness/build.sh && echo "  parses OK in container" \
  || echo "  DOES NOT PARSE in container"

echo
echo "########## normalize and re-restore ##########"
find "$SRC" -maxdepth 1 -type f -writable \( -name '*.cpp' -o -name '*.cuh' -o -name '*.hpp' \) \
  -exec sed -i 's/\r$//' {} +
find "$ON" -type f -writable \( -name '*.sh' -o -name '*.py' \) \
  -not -path '*/build/*' -not -path '*/compbench/*' -exec sed -i 's/\r$//' {} +
cp -f "$EXP/arms/$CAND/gemm_rs_mi300x.cpp"            "$SRC/gemm_rs_mi300x.cpp"
cp -f "$EXP/arms/$CAND/gemm_rs_mi300x_hk_adapter.cuh" "$SRC/gemm_rs_mi300x_hk_adapter.cuh"
sha256sum "$SRC/gemm_rs_mi300x.cpp" "$SRC/gemm_rs_mi300x_hk_adapter.cuh" \
          "$EXP/arms/$CAND/gemm_rs_mi300x.cpp" "$EXP/arms/$CAND/gemm_rs_mi300x_hk_adapter.cuh"

echo
for try in 1 2 3; do
  docker exec dhk-gemmrs bash $ON/harness/build.sh > "$EXP/paired_$CAND/build_restore2.log" 2>&1
  if grep -q "ALL MODULES BUILT" "$EXP/paired_$CAND/build_restore2.log"; then
    echo "RESTORE BUILD OK (attempt $try)"; break
  fi
  echo "attempt $try failed:"; tail -12 "$EXP/paired_$CAND/build_restore2.log"; sleep 10
done

echo
echo "########## proof: .so newer than source, and setprio present ##########"
ls -la --time-style=+%H:%M:%S $ON/build/*.so 2>/dev/null | head
stat -c '%y %n' "$SRC/gemm_rs_mi300x.cpp"
grep -c 's_setprio' "$SRC/gemm_rs_mi300x.cpp" || true

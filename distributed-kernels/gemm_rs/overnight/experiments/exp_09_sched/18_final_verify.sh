#!/usr/bin/env bash
# Leave-behind check: the node's source is the shipped arm, the .so was built
# from it, and no GPU process of ours is left running.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_09_sched
SRC=$ON/..

echo "########## CRLF after the last push ##########"
n=$(grep -rlU $'\r' /home/subvadla/dhk/distributed-kernels/gemm_rs \
      --include='*.sh' --include='*.py' --include='*.cpp' --include='*.cuh' 2>/dev/null \
      | grep -v '/build/' | grep -v '/compbench/' | wc -l)
echo "files with CR: $n"
if [ "$n" != "0" ]; then
  echo "normalizing"
  find /home/subvadla/dhk/distributed-kernels/gemm_rs -type f -writable \
    \( -name '*.sh' -o -name '*.py' -o -name '*.cpp' -o -name '*.cuh' -o -name '*.hpp' \) \
    -not -path '*/build/*' -not -path '*/compbench/*' -exec sed -i 's/\r$//' {} +
fi

echo
echo "########## source == shipped arm? ##########"
for f in gemm_rs_mi300x.cpp gemm_rs_mi300x_hk_adapter.cuh; do
  a=$(sha256sum "$SRC/$f" | cut -d' ' -f1)
  b=$(sha256sum "$EXP/arms/b_setprio/$f" | cut -d' ' -f1)
  [ "$a" = "$b" ] && echo "  MATCH  $f  $a" || echo "  !! MISMATCH $f live=$a arm=$b"
done

echo
echo "########## rebuild from the live source and confirm ##########"
docker exec dhk-gemmrs bash -n $ON/harness/build.sh || { echo "build.sh does not parse"; exit 1; }
docker exec dhk-gemmrs bash $ON/harness/build.sh > /tmp/final_build.log 2>&1
grep -q "ALL MODULES BUILT" /tmp/final_build.log && echo "  ALL MODULES BUILT" \
  || { echo "  BUILD FAILED"; tail -12 /tmp/final_build.log; }
ls -la --time-style=+%H:%M:%S $ON/build/*.so | awk '{print "  "$6" "$7}'
printf '  s_setprio in live source: %s\n' "$(grep -c s_setprio $SRC/gemm_rs_mi300x.cpp)"
printf '  acc_anchor in live adapter: %s\n' "$(grep -c acc_anchor $SRC/gemm_rs_mi300x_hk_adapter.cuh)"

echo
echo "########## node left clean? ##########"
rocm-smi --showpids 2>&1 | sed -n '/PID/,/^====/p'
echo "KFD pids: $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)"
rocm-smi 2>&1 | grep -E 'perf_determinism|manual|auto' | head -3

echo
echo "########## artifacts ##########"
du -sh $EXP 2>/dev/null
ls $EXP
ls $EXP/arms

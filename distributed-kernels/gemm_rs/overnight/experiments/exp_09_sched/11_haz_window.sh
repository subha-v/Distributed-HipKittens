#!/usr/bin/env bash
# Are the race-checker's 10 hits real, or a layout-order artifact of the loop
# rotation a1b enabled? Print the load window, the reported consumer window,
# and every s_waitcnt between them in LAYOUT order.
#   11_haz_window.sh <arm> <load_line> <use_line>
set -uo pipefail
ARM=${1:?arm}; L0=${2:?load line}; L1=${3:?use line}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
S=$ON/experiments/exp_09_sched/arms/$ARM/gemm_rs_mi300x.gfx942.s

echo "===== load window: $((L0-14)) .. $((L0+4)) ====="
sed -n "$((L0-14)),$((L0+4))p" "$S" | cat -n | sed "s/^/ /"
echo
echo "===== reported-consumer window: $((L1-12)) .. $((L1+12)) ====="
awk -v a=$((L1-12)) -v b=$((L1+12)) 'NR>=a && NR<=b {printf "%7d| %s\n", NR, $0}' "$S"
echo
echo "===== block boundaries and s_waitcnt between $L0 and $L1 (LAYOUT order) ====="
awk -v a=$L0 -v b=$L1 'NR>a && NR<b && (/^\.LBB/ || /^; %bb\./ || /s_waitcnt/ || /s_cbranch/ || /s_branch/ || /s_endpgm/) {printf "%7d| %s\n", NR, $0}' "$S" | head -40

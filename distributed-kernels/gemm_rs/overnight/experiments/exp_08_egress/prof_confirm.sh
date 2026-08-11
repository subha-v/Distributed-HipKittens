#!/usr/bin/env bash
# exp_08 step 3: confirm the win is the mechanism we claimed and not something
# else. The tile-order change moves no bytes and changes no store instruction,
# so on the WINNING binary the traffic counters must be UNCHANGED --
# 117.48 MB of fabric writes, 99.9% of them 64 B -- while the time drops 27.8%.
# If fabric bytes moved, the change did something other than advertised.
# The write-request stall should fall, because the links that were saturated no
# longer carry the whole round.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_08_egress
WARM=2
MEAS=4

n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
[ "$n" = "0" ] || { echo "ABORT: node dirty ($n KFD pids)"; exit 1; }

g1="TCC_EA0_WRREQ TCC_EA0_WRREQ_64B TCC_EA0_WRREQ_DRAM TCC_EA0_RDREQ"
g2="TCC_WRITEBACK TCC_ALL_TC_OP_WB_WRITEBACK TCC_NORMAL_WRITEBACK TCC_EA0_RDREQ_DRAM"
g3="TCC_EA0_WRREQ_STALL TCC_EA0_WRREQ_GMI_CREDIT_STALL TCC_EA0_WRREQ_IO_CREDIT_STALL TCC_EA0_WRREQ_DRAM_CREDIT_STALL"
g6="TCC_EA0_WRREQ_LEVEL TCC_EA0_WRREQ TCC_TAG_STALL TCC_STREAMING_REQ"

run() {
  local tag=$1 mod=$2 m=$3 nn=$4 k=$5 bias=$6 seed=$7; shift 7
  local out="$D/prof/$tag"
  rm -rf "$out"
  echo "  [$tag] $mod ${m}x${nn}x${k}"
  docker exec -w "$D" dhk-gemmrs timeout 900 \
    rocprofv3 --pmc $* -d "$out" -o p --output-format csv \
      -- python3 -u prof_driver.py "$mod" "$m" "$nn" "$k" "$bias" "$seed" \
         "$WARM" "$MEAS" > "$out.log" 2>&1
  grep -h '^PROF' "$out.log" || tail -8 "$out.log"
}

echo "=== WINNER binary, shape 6 and shape 5 ==="
run w6_g1 gemm_rs_mi300x 8192 8192 29568 0 42 $g1
run w6_g2 gemm_rs_mi300x 8192 8192 29568 0 42 $g2
run w6_g3 gemm_rs_mi300x 8192 8192 29568 0 42 $g3
run w6_g6 gemm_rs_mi300x 8192 8192 29568 0 42 $g6
run w5_g1 gemm_rs_mi300x 8192 4096 14336 1 7168 $g1
run w5_g3 gemm_rs_mi300x 8192 4096 14336 1 7168 $g3

echo
echo "=== stall attribution on the OLD (WGM=4) order, for the same breakdown ==="
docker exec dhk-gemmrs bash -lc "cd $ON/harness && cp build/gemm_rs_mi300x.so build/win_keep.so"
docker exec dhk-gemmrs bash -lc \
  "hipcc -std=c++20 -O3 -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
     -ffast-math --offload-arch=gfx942 -shared -fPIC \
     -I$ON/../../../include -I$ON/../../../include/pyutils \
     -I/opt/rocm/include/hip \
     -I\$(python3 -c 'import pybind11;print(pybind11.get_include())') \
     -I\$(python3 -c 'import sysconfig;print(sysconfig.get_paths()[\"include\"])') \
     -Wno-nan-infinity-disabled -ferror-limit=0 \
     -DHK_GEMM_RS_MI300X_WGM4=1 -DTK_MODNAME=gemm_rs_wgm4 \
     $ON/../gemm_rs_mi300x.cpp -o $ON/harness/build/gemm_rs_wgm4.so" 2>&1 | tail -3
ls -l $ON/harness/build/gemm_rs_wgm4.so 2>/dev/null || echo "control build FAILED"
run o6_g3 gemm_rs_wgm4 8192 8192 29568 0 42 $g3
run o6_g6 gemm_rs_wgm4 8192 8192 29568 0 42 $g6

echo
echo "############ AGGREGATE ############"
for tag in w6_g1 w6_g2 w6_g3 w6_g6 w5_g1 w5_g3 o6_g3 o6_g6; do
  f="$D/prof/$tag/p_counter_collection.csv"
  [ -f "$f" ] || { echo "=== $tag : MISSING"; continue; }
  docker exec -w "$D" dhk-gemmrs python3 prof_agg.py "$f" $WARM $MEAS "$tag"
done

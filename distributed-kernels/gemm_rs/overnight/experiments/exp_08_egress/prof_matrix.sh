#!/usr/bin/env bash
# exp_08 step 1: MEASURE. No kernel code is written until this reports.
#
# Three questions, one matrix:
#   Q1 write amplification -- are the bytes on the fabric equal to the useful
#      bytes, or 2-4x them?  (g1 amp, g4 mtype)
#   Q2 bunched or spread -- do the payload lines leave L2 because `buffer_wbl2`
#      pushed them (bunched at the release) or because capacity evicted them
#      during the emit/mainloop (spread)?  (g2 wb)
#   Q3 is the fabric saturated at all -- xGMI write-credit starvation is the
#      direct backpressure signal.  (g3 stall)
#
# The `emitlocal` arm is the KNOWN-ANSWER CONTROL for the counter labels: it
# writes the identical bytes through the identical instructions to the local
# rank's own slot, so its off-die write requests MUST collapse to ~0. If they
# do not, `TCC_EA0_WRREQ - TCC_EA0_WRREQ_DRAM` does not mean what this
# experiment claims and no conclusion may be drawn from it.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_08_egress
WARM=2
MEAS=4
mkdir -p "$D/prof"

n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
[ "$n" = "0" ] || { echo "ABORT: node dirty ($n KFD pids)"; exit 1; }

g1="TCC_EA0_WRREQ TCC_EA0_WRREQ_64B TCC_EA0_WRREQ_DRAM TCC_EA0_RDREQ"
g2="TCC_WRITEBACK TCC_ALL_TC_OP_WB_WRITEBACK TCC_NORMAL_WRITEBACK TCC_EA0_RDREQ_DRAM"
g3="TCC_EA0_WRREQ_GMI_CREDIT_STALL TCC_TOO_MANY_EA_WRREQS_STALL TCC_EA0_WRREQ_STALL TCC_CYCLE"
g4="TCC_UC_REQ TCC_NC_REQ TCC_CC_REQ TCC_EA0_WR_UNCACHED_32B"
g5="TCC_WRITE TCC_READ TCC_HIT TCC_MISS"

run() {   # run <tag> <module> <m> <n> <k> <bias> <seed> <counters...>
  local tag=$1 mod=$2 m=$3 nn=$4 k=$5 bias=$6 seed=$7; shift 7
  local out="$D/prof/$tag"
  if [ -f "$out/p_counter_collection.csv" ]; then
    echo "  [$tag] already collected"; return 0
  fi
  rm -rf "$out"
  echo "  [$tag] $mod ${m}x${nn}x${k} :: $*"
  docker exec -w "$D" dhk-gemmrs timeout 900 \
    rocprofv3 --pmc $* -d "$out" -o p --output-format csv \
      -- python3 -u prof_driver.py "$mod" "$m" "$nn" "$k" "$bias" "$seed" \
         "$WARM" "$MEAS" > "$out.log" 2>&1
  rc=$?
  grep -h '^PROF' "$out.log" || echo "    NO PROF LINE (rc=$rc) --- tail:"
  [ -f "$out/p_counter_collection.csv" ] || sed -n '$!d;p' "$out.log"
  if [ $rc -ne 0 ]; then tail -12 "$out.log"; fi
}

echo "=== shape 6 (8192x8192x29568), production arm ==="
run s6_prod_g1 gemm_rs_mi300x     8192 8192 29568 0 42 $g1
run s6_prod_g2 gemm_rs_mi300x     8192 8192 29568 0 42 $g2
run s6_prod_g3 gemm_rs_mi300x     8192 8192 29568 0 42 $g3
run s6_prod_g4 gemm_rs_mi300x     8192 8192 29568 0 42 $g4
run s6_prod_g5 gemm_rs_mi300x     8192 8192 29568 0 42 $g5

echo "=== shape 6, emit-LOCAL arm (counter-label control: fabric must vanish) ==="
run s6_loc_g1 gemm_rs_abl_emitlocal 8192 8192 29568 0 42 $g1
run s6_loc_g2 gemm_rs_abl_emitlocal 8192 8192 29568 0 42 $g2
run s6_loc_g3 gemm_rs_abl_emitlocal 8192 8192 29568 0 42 $g3
run s6_loc_g4 gemm_rs_abl_emitlocal 8192 8192 29568 0 42 $g4

echo "=== shape 6, no-mainloop arm (removes A/B operand read traffic) ==="
run s6_nomain_g1 gemm_rs_abl_nomain 8192 8192 29568 0 42 $g1
run s6_nomain_g2 gemm_rs_abl_nomain 8192 8192 29568 0 42 $g2
run s6_nomain_g3 gemm_rs_abl_nomain 8192 8192 29568 0 42 $g3

echo "=== shape 5 (8192x4096x14336), production arm ==="
run s5_prod_g1 gemm_rs_mi300x 8192 4096 14336 1 7168 $g1
run s5_prod_g2 gemm_rs_mi300x 8192 4096 14336 1 7168 $g2
run s5_prod_g3 gemm_rs_mi300x 8192 4096 14336 1 7168 $g3

echo
echo "############ AGGREGATE ############"
for tag in s6_prod_g1 s6_prod_g2 s6_prod_g3 s6_prod_g4 s6_prod_g5 \
           s6_loc_g1 s6_loc_g2 s6_loc_g3 s6_loc_g4 \
           s6_nomain_g1 s6_nomain_g2 s6_nomain_g3 \
           s5_prod_g1 s5_prod_g2 s5_prod_g3; do
  f="$D/prof/$tag/p_counter_collection.csv"
  [ -f "$f" ] || { echo "=== $tag : MISSING"; continue; }
  docker exec -w "$D" dhk-gemmrs python3 prof_agg.py "$f" $WARM $MEAS "$tag"
done

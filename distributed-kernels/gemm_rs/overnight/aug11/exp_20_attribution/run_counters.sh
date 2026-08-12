#!/usr/bin/env bash
# aug11/exp_20: rocprofv3 counter pass at the current best config.
#
# Reuses exp_08's validated collection machinery verbatim -- its prof_driver.py
# is invoked in place, from its own directory, because it computes the harness
# import path from __file__ (dirname^3 + /harness). Only the output directory is
# ours; nothing under experiments/exp_08_egress/ is written.
#
# Arm choice: `gemm_rs_abl_full`, the unablated scratch build the attribution
# table was measured on, rebuilt by reattribute.sh at 04:00 today. mtime is not
# a freshness test (push.ps1 resets every source mtime), so the arm that
# produced the numbers is the arm profiled. `s6_prod_g1` cross-checks the
# production .so against it.
#
# WHAT IS ALREADY COLLECTED, and why this script only adds to it:
#   g1..g3 were collected at 04:10-04:14 today -- AFTER the 04:00 arm rebuild,
#   so they describe the same binary as ablation.json -- over all six shapes,
#   every cell reporting correct=1 errors=none:
#     g1 TCC_EA0_WRREQ TCC_EA0_WRREQ_64B TCC_EA0_WRREQ_DRAM TCC_EA0_RDREQ
#     g2 TCC_EA0_WRREQ_LEVEL TCC_EA0_WRREQ TCC_WRITEBACK TCC_CYCLE
#     g3 TCC_WRITE TCC_NORMAL_WRITEBACK TCC_ALL_TC_OP_WB_WRITEBACK TCC_NC_REQ
#   Re-collecting them would buy nothing. The two groups below are NEW and are
#   collected under NEW TAGS (g4/g5), because `run()` skips on the TAG alone:
#   reusing a tag with a different counter set silently mislabels the data,
#   which is how this script's first version wasted a launch.
#
# COUNTER CHOICE -- exp_08 already paid for this on this ASIC:
#   * The three TCC_EA0_WRREQ_*_CREDIT_STALL counters are UNUSABLE on gfx942
#     (they read ~0 while TCC_EA0_WRREQ_STALL reads 16% of TCC_CYCLE), so g5
#     carries TCC_EA0_WRREQ_STALL + TCC_TAG_STALL instead.
#   * Groups are 4 counters wide to stay inside one pass, so no derived ratio
#     ever divides counters collected from different populations. TCC_CYCLE
#     rides along in g5 because both stall figures are normalized by it.
# g4 is the group the refreshed attribution demands: ~61% of shape 6 is now the
# mainloop, and SQ busy-vs-wait is what says whether that pool is MFMA-bound or
# stall-bound. If an SQ counter does not resolve on gfx942 the cell fails, is
# recorded as unavailable, and the pass continues -- counters do not gate this
# experiment.
#
#   run_counters.sh [warm] [meas]
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E8=$ON/experiments/exp_08_egress
OUT=$ON/aug11/exp_20_attribution
PROF=$OUT/prof
WARM=${1:-2}
MEAS=${2:-4}
mkdir -p "$PROF"

# g4 mainloop occupancy: is the 61% GEMM pool MFMA-bound or wait-bound?
g4="SQ_BUSY_CYCLES SQ_VALU_MFMA_BUSY_CYCLES SQ_WAIT_ANY SQ_INSTS_MFMA"
# g5 backpressure: the two stall counters that DO work on this ASIC.
g5="TCC_EA0_WRREQ_STALL TCC_TAG_STALL TCC_CYCLE TCC_EA0_WRREQ"

# tag m n k bias seed -- shape 6 (largest) and shape 5 (mid), per the brief.
SHAPES=(
  "s6 8192 8192 29568 0 42"
  "s5 8192 4096 14336 1 7168"
)

wait_clean() {
  local tries=${1:-12} n
  for ((i=0; i<tries; i++)); do
    n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
    if [ "$n" = "0" ]; then return 0; fi
    [ "$i" = "0" ] && echo "  KFD pids: $n -- waiting for the node to drain"
    sleep 10
  done
  echo "  WARN: a KFD pid persists after $((tries*10))s; proceeding, because a"
  echo "        counter cell reports per-dispatch counts, not timings"
  return 0
}

run() {   # run <tag> <module> <m> <n> <k> <bias> <seed> <counters...>
  local tag=$1 mod=$2 m=$3 nn=$4 k=$5 bias=$6 seed=$7; shift 7
  local out="$PROF/$tag"
  if [ -f "$out/p_counter_collection.csv" ]; then
    echo "  [$tag] already collected"; return 0
  fi
  wait_clean 12
  rm -rf "$out"
  echo "  [$tag] $mod ${m}x${nn}x${k} :: $*"
  docker exec -w "$E8" dhk-gemmrs timeout 1200 \
    rocprofv3 --pmc $* -d "$out" -o p --output-format csv \
      -- python3 -u prof_driver.py "$mod" "$m" "$nn" "$k" "$bias" "$seed" \
         "$WARM" "$MEAS" > "$out.log" 2>&1
  local rc=$?
  grep -h '^PROF' "$out.log" || { echo "    NO PROF LINE (rc=$rc); tail:";
                                  tail -12 "$out.log"; }
  [ -f "$out/p_counter_collection.csv" ] || echo "    NO CSV (rc=$rc)"
}

echo "############ g4 SQ busy vs wait / MFMA ############"
for s in "${SHAPES[@]}"; do set -- $s
  run "$1_full_g4" gemm_rs_abl_full $2 $3 $4 $5 $6 $g4
done

echo "############ g5 fabric backpressure ############"
for s in "${SHAPES[@]}"; do set -- $s
  run "$1_full_g5" gemm_rs_abl_full $2 $3 $4 $5 $6 $g5
done

echo
echo "############ AGGREGATE (exp_08's prof_agg.py, verbatim) ############"
for f in "$PROF"/*/p_counter_collection.csv; do
  [ -f "$f" ] || continue
  tag=$(basename "$(dirname "$f")")
  docker exec -w "$E8" dhk-gemmrs python3 prof_agg.py "$f" $WARM $MEAS "$tag"
done
echo "COUNTERS DONE"

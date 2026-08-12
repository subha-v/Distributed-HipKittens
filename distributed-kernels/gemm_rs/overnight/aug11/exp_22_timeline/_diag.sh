#!/usr/bin/env bash
# Discriminate: node state vs binary/config provenance vs host-side contention.
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E22=$ON/aug11/exp_22_timeline
echo "===== clocks pinned during the run? ====="
head -30 "$E22/clocks_before.txt" 2>/dev/null | grep -Ei "sclk|mclk|level" | head -12
echo "--- live now ---"
rocm-smi --showclocks 2>/dev/null | grep -Ei "sclk" | head -10
echo
echo "===== host-side load (shapes 1-3 are HOST-bound at ~62 us/op) ====="
uptime; nproc
echo "--- top CPU consumers ---"
ps -eo pcpu,pid,user,comm --sort=-pcpu | head -8
echo
echo "===== other tenants ====="
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Image}}' | head
rocm-smi --showpids 2>/dev/null | head -10
echo
echo "===== prior m7_results.json on the node (the 613.70 provenance) ====="
find "$ON" -name "m7_results.json" -printf "%T@ %TY-%Tm-%Td %TH:%TM  %p\n" 2>/dev/null | sort -rn | head -20
echo
echo "===== grep the tree for the reference vector ====="
grep -rn "613.7" "$ON"/RESULTS.md "$ON"/aug11/*.md "$ON"/experiments/LESSONS.md 2>/dev/null | head -12
echo
echo "===== binary provenance: what has changed in the kernel sources ====="
cd "$ON/.." && git log --oneline -12 -- gemm_rs_mi300x.cpp gemm_rs_mi300x_constants.cuh gemm_rs_mi300x_hk_adapter.cuh 2>&1 | head -14
echo "--- uncommitted ---"
git status --porcelain -- gemm_rs_mi300x.cpp gemm_rs_mi300x_constants.cuh gemm_rs_mi300x_hk_adapter.cuh 2>&1 | head
echo
echo "===== shape table: what NR does row 6 declare? (M7 printed NR=48) ====="
grep -n "num_reducer_ctas\|NUM_REDUCER_CTAS\|29568\|, *48\b" "$ON/../gemm_rs_mi300x_constants.cuh" 2>/dev/null | head -25

#!/usr/bin/env bash
# exp_34 condition 1, take 2: LIVE host file only, no experiment snapshots.
set -uo pipefail
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
HOST=$K0/prefill_opt/host/e004pf_k0pf_ab.py

echo "### host file: $HOST"
ls -l "$HOST"
echo
echo "################ row_ready in the LIVE host file ################"
grep -n -iE "row_ready" "$HOST"
echo
echo "################ T_LOC_MAX in the LIVE host file ################"
grep -n -E "T_LOC_MAX *=|T_LOC_MAX," "$HOST" | head -20
echo
echo "################ symmetric-heap / iris allocation helpers ################"
grep -n -iE "def (sym|_sym|iris_|alloc)" "$HOST" | head -30
echo
echo "################ any zeros/empty alloc with WORLD*T_LOC_MAX shape ################"
grep -n -E "WORLD *\* *T_LOC_MAX|T_LOC_MAX *\* *WORLD" "$HOST" | head -20
echo "===DONE==="

#!/usr/bin/env bash
# exp_34 condition 1, the number that decides it: what T_LOC_MAX and MAXTOK the
# CAMPAIGN actually runs with, and therefore the highest word mode 14 writes.
set -uo pipefail
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
BM=$K0/benchmarks/mok_synthetic_prefill
HOST=$K0/prefill_opt/host/e004pf_k0pf_ab.py

echo "############ campaign env: T_LOC_MAX / MAXTOK / T ############"
grep -rn -E "K0_T_LOC_MAX|MAXTOK|K0_T=|K0_TOKENS|export K0_" "$BM"/run_campaign.sh | head -40
echo "--- any other setter in the benchmark dir ---"
grep -rn -E "K0_T_LOC_MAX|K0P6_D_MAXTOK|MAXTOK" "$BM" 2>/dev/null | grep -v Binary | head -20
echo
echo "############ host defaults + the descriptor MAXTOK slot value ############"
grep -n -E "^T_LOC_MAX|^MAXTOK|MAXTOK *=|K0_MAXTOK" "$HOST" | head -20
echo "--- what goes into descriptor slot 51 (K0P6_D_MAXTOK) ---"
grep -n -B4 -A4 "MAXTOK" "$HOST" | grep -n -E "desc|slot|51" | head -20
grep -n -E "^\s*(MAXTOK|int\(MAXTOK\)),?\s*#.*(51|MAXTOK)" "$HOST" | head
echo "--- descriptor build list around slot 51 ---"
awk '/pf6_state\["row_ready_p"\]/{f=NR} END{}' "$HOST" >/dev/null
grep -n -A3 "K0P6_D_MAXTOK\|# 51\|MAXTOK, *# " "$HOST" | head -20
echo "===DONE==="

#!/usr/bin/env bash
# exp_34 conditions 2 and 5, plus the last ISA line.
set -uo pipefail
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
HOST=$K0/prefill_opt/host/e004pf_k0pf_ab.py
BM=$K0/benchmarks/mok_synthetic_prefill

echo "############ ISA: first vector load after the mode-14 acquire ############"
docker exec subha_k1 bash -lc '
OUT=/home/subvadla/e34/out
L=$(grep -n "E34_M14_ACQ_BEGIN" $OUT/MK.s | cut -d: -f1)
awk -v a=$L "NR>a && /global_load|flat_load|buffer_load/ {print NR\": \"\$0; c++; if (c==3) exit}" $OUT/MK.s
echo "--- and the label that block belongs to ---"
awk -v a=$L "NR>a && /^\.LBB0_/ {lbl=\$0} NR>a && /global_load|flat_load|buffer_load/ {print \"first load is inside: \" lbl; exit}" $OUT/MK.s
' 2>&1

echo
echo "############ 2. EVERY writer of row_ready on the host ############"
echo "--- direct name uses ---"
grep -n -E "_pf6_row_ready|row_ready" "$HOST"
echo "--- zero_() calls anywhere in the host, with the list they belong to ---"
grep -n -B2 -A2 "\.zero_()" "$HOST" | head -60

echo
echo "############ 2b. is K0_PF6GM_DECOMP on in the campaign? ############"
grep -rn -E "PF6GM_DECOMP|DECOMP" "$BM"/*.sh "$BM"/*.py 2>/dev/null | head -20
echo "--- the reset call sites and their sync discipline ---"
grep -n -B6 -A10 "_dc_reset()" "$HOST" | head -80

echo
echo "############ 2c. soak loop: does anything clear state per iteration? ############"
grep -n -B4 -A18 "MPS_SOAK_ITERS|soak" "$HOST" | head -80

echo
echo "############ 5. where pperr is PRINTED (bit 26 visibility) ############"
grep -n -E "pperr|MOK GATE|MPS SPIN" "$HOST" | grep -iE "print|gate|f\"" | head -30
echo "===DONE==="

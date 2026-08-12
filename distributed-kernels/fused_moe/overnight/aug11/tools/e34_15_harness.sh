#!/usr/bin/env bash
# exp_34 step 15: will the HOST side accept mode 14? The kernel validator is now
# `mode > 14u`, but K0_MPS_CFG is parsed and encoded host-side in the harness
# (files this change does not own). Find every place a mode bound is enforced.
# READ ONLY. NO GPU WORK.
set -uo pipefail
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
B=$K0/benchmarks/mok_synthetic_prefill

echo "=== K0_MPS_CFG parse sites ==="
grep -rn "K0_MPS_CFG" "$K0" --include=*.py --include=*.sh --include=*.cpp --include=*.hpp 2>/dev/null | head -30
echo
echo "=== mode bound checks (13 / 14) near mps config ==="
grep -rn -E "mode[^a-zA-Z_].{0,24}(1[0-9])|> *13|>= *14|<= *13" "$K0" \
  --include=*.py --include=*.hpp --include=*.cpp 2>/dev/null | grep -iE "mps|mode" | head -30
echo
echo "=== encode_config / mps config on the host ==="
grep -rn -E "encode_config|reserved_comm_ctas|flush_rows" "$K0" \
  --include=*.py --include=*.hpp --include=*.cpp 2>/dev/null | head -30
echo
echo "=== poison env forwarding (must stay on) ==="
grep -rn "K0_MOK_POISON_OUT" "$K0" 2>/dev/null | head -10
echo "===DONE==="

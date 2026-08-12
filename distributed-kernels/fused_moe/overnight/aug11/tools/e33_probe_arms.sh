#!/usr/bin/env bash
# exp_33 probe 2: is the MPS timestamp block reachable for pf6gm_mega, or only for
# mps_mega? Determines whether the M7 epilogue surcharge is computable from stamps.
# Read-only.
set -uo pipefail
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
AB="$K0/prefill_opt/host/e004pf_k0pf_ab.py"

echo "=== enclosing context: what gates the soak+TS block (search upward for arm guard) ==="
awk 'NR>=4900 && NR<=5060' "$AB" | grep -nE 'MPS_SOAK|mps_soak_iters|if .*mps|MPS_SELECTED|arm|def |for ' | sed -n '1,40p'

echo
echo "=== all occurrences of mps_state / _pf6_mps_state ==="
grep -n '_pf6_mps_state\|"mps_state"\|pf6_state\[' "$AB" | sed -n '1,60p'

echo
echo "=== arm selection flags ==="
grep -nE '^_?(MPS|PF6GM)[A-Z_]*_(SELECTED|REQUESTED)|MPS_SELECTED|PF6GM_SELECTED|pf6gm_mega|mps_mega' "$AB" | sed -n '1,60p'

echo
echo "=== which .hip does each mega arm build from ==="
grep -nE 'k0pf6gm_device_tile(_mps)?\.hip|k0pf6gm_mps_mega|k0pf6gm_mega' "$AB" | sed -n '1,40p'

echo
echo "=== the line number where the soak block starts ==="
grep -n 'MPS SOAK\|_mps_soak_iters =\|K0_MPS_SOAK_ITERS' "$AB" | sed -n '1,20p'
echo "=== PROBE2 DONE ==="
exit 0

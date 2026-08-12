#!/usr/bin/env bash
# exp_24 probe 3: does the HOST harness ever read/validate `part`? what is the
# negative control? is `part` shared across arms?
set -uo pipefail
AB="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py"
echo "=== ab.py: every mention of 'part' (excluding 'partial'/'particip')"
grep -nE "\bpart\b|part_buf|\"part\"|'part'|part_t|d_part|PART" "$AB" | grep -viE "partial|particip|partition|department" | head -60
echo
echo "=== ab.py: negative control"
grep -n "control" "$AB" | head -40
echo
echo "=== ab.py: pf6 state allocation of part / sharing across arms"
grep -n "pf6_state\|_state\[.part.\]\|def _pf6\|_alloc_pf6\|part =" "$AB" | head -60
echo
echo "=== pybind bridge encode_config signature"
BR="$(find "$HOME/amd-master" -name 'mps_host_bridge.cpp' 2>/dev/null | head -1)"
echo "bridge: $BR"
[ -n "$BR" ] && sed -n '1,60p' "$BR"
exit 0

#!/usr/bin/env bash
# exp_36 step 5: build the scratch drivers in $HOME/e36.
#
# WHY THIS EXISTS. run_campaign.sh hard-wires the shape:
#     -e K0_T=4096 -e K0_T_LOC_MAX=40960 -e K0_PADMAX=263136
#     -e K0_MAXTOK=4096 -e K0_MAXTOK_PROD=4096
# and docker gives no way to override an -e from outside. exp_36's whole
# independent variable is T, so the driver has to be parameterised. The harness
# itself is NOT edited: rc_T.sh is generated from it by sed, and the diff is
# printed below so the deviation is auditable. At T=4096 the generated defaults
# reproduce the hard-wired numbers EXACTLY (T_LOC_MAX = 10*T, PADMAX =
# WORLD*T*TOPK + 31*E rounded to 32, MAXTOK = MAXTOK_PROD = T), i.e. the
# T=4096 control run is byte-identical in environment to the stock harness.
set -u
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
SRC="$K0/benchmarks/mok_synthetic_prefill/run_campaign.sh"
E36="$HOME/e36"
mkdir -p "$E36" "$E36/scratch"
DST="$E36/rc_T.sh"

sed -e 's|^SCRIPT_DIR="\$(cd -- "\$(dirname -- "\${BASH_SOURCE\[0\]}")" \&\& pwd)"$|SCRIPT_DIR="${E36_HARNESS_DIR:-$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill}"|' \
    -e 's|^  -e K0_T=4096 \\$|      -e "K0_T=${K0_T}" \\|' \
    -e 's|^      -e K0_T=4096 \\$|      -e "K0_T=${K0_T}" \\|' \
    -e 's|^      -e K0_T_LOC_MAX=40960 \\$|      -e "K0_T_LOC_MAX=${K0_T_LOC_MAX}" \\|' \
    -e 's|^      -e K0_PADMAX=263136 \\$|      -e "K0_PADMAX=${K0_PADMAX}" \\|' \
    -e 's|^      -e K0_MAXTOK=4096 \\$|      -e "K0_MAXTOK=${K0_MAXTOK}" \\|' \
    -e 's|^      -e K0_MAXTOK_PROD=4096 \\$|      -e "K0_MAXTOK_PROD=${K0_MAXTOK_PROD}" \\|' \
    "$SRC" > "$DST"

# insert the shape derivation right after the arms are parsed
python3 - "$DST" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = 'IFS=\',\' read -r -a base_arms <<< "${arm_csv}"\n'
block = anchor + '''
# ---- exp_36 shape parameterisation (the ONLY behavioural delta vs the stock
# run_campaign.sh; every default below reproduces the hard-wired value at
# T=4096, so the T=4096 arm is a true control) --------------------------------
K0_T="${K0_T:-4096}"
K0_T_LOC_MAX="${K0_T_LOC_MAX:-$(( 10 * K0_T ))}"
_pad_safe=$(( 8 * K0_T * 8 + 31 * 32 ))
K0_PADMAX="${K0_PADMAX:-$(( (_pad_safe + 31) / 32 * 32 ))}"
K0_MAXTOK="${K0_MAXTOK:-${K0_T}}"
K0_MAXTOK_PROD="${K0_MAXTOK_PROD:-${K0_T}}"
echo "[exp_36 shape] T=${K0_T} T_LOC_MAX=${K0_T_LOC_MAX} PADMAX=${K0_PADMAX} MAXTOK=${K0_MAXTOK} MAXTOK_PROD=${K0_MAXTOK_PROD}"
'''
assert anchor in s, "anchor not found"
s = s.replace(anchor, block, 1)
open(p, 'w').write(s)
print("shape block inserted")
PY

chmod +x "$DST"
echo "=== diff run_campaign.sh -> rc_T.sh ==="
diff -u "$SRC" "$DST"
echo "=== end diff (rc=$?) ==="

# ------------------------------------------------------------------ screenT.sh
# Source of truth is the REPO copy (pushed to $E36/screen_src.sh by push.ps1):
# the deployed ~/.overnight-scripts/screen.sh predates the exp_32 poison fields
# and reports soak=MALFORMED on every current log.
SCR="$E36/screen_src.sh"
[ -f "$SCR" ] || SCR="$HOME/.overnight-scripts/screen.sh"
echo "screen source: $SCR"
DST2="$E36/screenT.sh"
sed -e 's|^CAMPAIGN="\$K0/benchmarks/mok_synthetic_prefill/run_campaign.sh"$|CAMPAIGN="${SCREEN_CAMPAIGN:-$K0/benchmarks/mok_synthetic_prefill/run_campaign.sh}"|' \
    -e 's|^SYNC="\${SCREEN_SYNC:-1}"$|SYNC="${SCREEN_SYNC:-0}"|' \
    -e 's|^SCRATCH="\$HOME/overnight-scratch"$|SCRATCH="${SCREEN_SCRATCH:-$HOME/overnight-scratch}"|' \
    -e 's|^    K0_MPS_TRACE="\$TRACE" \\$|    K0_MPS_TRACE="$TRACE" \\\n    K0_T="${SCREEN_T:-4096}" \\\n    K0_T_LOC_MAX="${SCREEN_T_LOC_MAX:-}" \\\n    K0_PADMAX="${SCREEN_PADMAX:-}" \\\n    K0_MAXTOK="${SCREEN_MAXTOK:-}" \\\n    K0_MAXTOK_PROD="${SCREEN_MAXTOK_PROD:-}" \\|' \
    "$SCR" > "$DST2"
chmod +x "$DST2"
echo "=== diff screen.sh -> screenT.sh ==="
diff -u "$SCR" "$DST2"
echo "=== end diff ==="
exit 0

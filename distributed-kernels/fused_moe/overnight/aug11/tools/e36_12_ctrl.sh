#!/usr/bin/env bash
# exp_36 step 12: CONTROL. One screen at T=4096 through the scratch driver.
# Must reproduce the stock-harness screen band for the ratchet config
# (mps/prod ~ 0.854 +/- 0.005 at 1 sigma) or the driver copy is not trusted.
set -u
E36="$HOME/e36"
mkdir -p "$E36"
cat > "$E36/e36c1.cfgs" <<'EOF'
C=16,g=353,mode=12,flush_rows=16,timestamps=1
EOF
E36_T=4096 E36_TAG=e36c1 bash "$E36/launch.sh"
exit 0

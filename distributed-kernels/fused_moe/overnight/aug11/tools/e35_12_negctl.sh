#!/usr/bin/env bash
# exp_35 step 5c: FAIL-CLOSED NEGATIVE CONTROL for the throttle encoding.
#
# NOT A RUNG. NOT A CANDIDATE. This run is EXPECTED TO FAIL.
#
# The derivation of rung (b) rests on one claim from moe_mps_adapter.cuh:352-359
# -- "a depth selector without the throttle enabled selects nothing. Reject
# instead of accepting a config that reads as swept." If that rejection is real,
# then g=321 (0x141 = skip_part_zero + depth_sel=1, throttle enable CLEAR) must
# be refused by the device-side config_is_valid at .hip:704, which sets
# pperr |= K0P6_MPS_ERR_CONFIG and returns. That is why rung (b) must be g=65
# (0x041, depth selector zeroed) and CANNOT be g=321: there is no legal config
# that clears the enable bit while keeping depth_sel=1, so g=65 is the UNIQUE
# legal "throttle bits only" neighbour of the ratchet's g=353.
#
# A pperr!=0 / gate-fail here is the EXPECTED result and is recorded as control
# evidence, not as a candidate regression.
set -uo pipefail
TAG=e35neg
export SCREEN_ARMS=production,pf6gm_mega,mps_mega
export SCREEN_WARMUP=1 SCREEN_TIMED=1 SCREEN_PROCS=1
export SCREEN_SYNC=0
export SCREEN_TRACE=1
export SCREEN_JOB_TIMEOUT=2400 SCREEN_RUN_TIMEOUT=2100
export K0_MPS_DESC_DUMP=1

CFG=/tmp/e35_neg_cfgs.txt
cat > "$CFG" <<'EOF'
C=16,g=321,mode=12,flush_rows=16,timestamps=1
EOF
echo "===HEAD==="; git -C "$HOME/Distributed-HipKittens" rev-parse HEAD
setsid -w timeout 3000 bash "$HOME/tools/screen.sh" "$TAG" "$CFG"
echo "===SCREEN_RC=$?==="
echo "===PPERR_EVIDENCE (expect a nonzero pperr / failed gate)==="
grep -hoE 'pperr=[0-9]+|\[MOK GATE\] mps_mega [^ ]+ [^ ]+ pass=\w+|ERR_CONFIG|config' \
  "$HOME"/overnight-scratch/${TAG}_*.log 2>/dev/null | sort | uniq -c | sort -rn | head -20
echo "===K0P6_MPS_ERR_CONFIG value==="
grep -rn 'K0P6_MPS_ERR_CONFIG' "$HOME/Distributed-HipKittens/distributed-kernels/fused_moe/" | head -5

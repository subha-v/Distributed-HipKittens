#!/usr/bin/env bash
# Landing run. The ladder and M9 that exp_26 already passed were run against
# PERSHAPE=1, which the timing then disqualified on its register tuple. The
# shipped rule is PERSHAPE=2, so it gets its own full ladder and its own M9 --
# a gate passed by a different binary is not a gate.
#
# Logs go to experiments/exp_26_ps2_land so the PERSHAPE=1 ladder logs survive
# as the record of what was actually run against what.
#
# No trap, no $(cmd && echo a || echo b): both of this experiment's aborted
# campaigns died in exactly those constructs after transport.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
LEASE=/tmp/gpu_lease_exp26.sh
tr -d '\r' < "$ON/tools/gpu_lease.sh" > "$LEASE"
mkdir -p "$D/logs" "$ON/experiments/exp_26_ps2_land/logs"

stage() {
  printf '\n########## %s ##########\n' "$1"
  date -u +%FT%TZ
}

stage "the default that is about to be gated"
grep -A2 '^#ifndef HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE' \
  /home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp
val=$(grep -A1 '^#ifndef HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE' \
  /home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp \
  | tail -1 | awk '{print $3}')
echo "PERSHAPE default = $val"
if [ "$val" != "2" ]; then
  echo "ABORTED: the source default is not 2; nothing to land"
  exit 1
fi

stage "acquire the lease"
bash "$LEASE" acquire exp_26 10800
if [ $? -ne 0 ]; then
  echo "ABORTED: no lease"
  exit 1
fi

stage "full gate ladder on the shipped rule"
bash "$ON/tools/gate_ladder.sh" exp_26_ps2_land 2>&1 | tee "$D/logs/ladder_ps2.log"
lrc=${PIPESTATUS[0]}
echo "ladder exit: $lrc"

stage "M9 on the shipped rule, golden = the incumbent"
docker exec -w $ON/harness dhk-gemmrs timeout 5400 \
  python3 -u $D/m9_vs_incumbent.py 2>&1 | tee "$D/logs/m9_ps2.log"
echo "m9 exit: ${PIPESTATUS[0]}"

stage "release the lease"
bash "$LEASE" release exp_26 2>&1 | head -2

stage "done"
echo "CAMPAIGN4 COMPLETE (ladder $lrc)"

#!/usr/bin/env bash
# exp_09_sched: ISA-only inspection of one arm. NO GPU, NO harness/build/*.so.
#
#   isa_arm.sh <arm-name>
#
# Rebuilds the resource-usage log and the .s (tools/m1_build.sh + tools/m2_isa.sh
# both write only under overnight/build/), then archives, for this arm:
#   m1a.log resource tuples, the .s, the k-loop schedule shape for the
#   256/256/32 pair and the 32/64/64 pair, and the LDS/VMEM race check.
set -uo pipefail
ARM=${1:?usage: isa_arm.sh <arm-name>}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_09_sched
D=$EXP/arms/$ARM
mkdir -p "$D"
S=$ON/build/isa/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s

# For the first minute or so after tools/push.ps1 returns, the CONTAINER can
# still see a truncated copy of a just-scp'd script even though the host sees
# it whole -- three times now tools/m2_isa.sh died inside dhk-gemmrs with
# "syntax error: unexpected end of file" at a different line each time (its
# `PY` heredoc terminator had not landed yet), which looks exactly like a
# compile failure. Check the view that matters: the container's.
for _try in 1 2 3 4 5 6 7 8; do
  if docker exec dhk-gemmrs bash -n "$ON/tools/m2_isa.sh"  2>/dev/null && \
     docker exec dhk-gemmrs bash -n "$ON/tools/m1_build.sh" 2>/dev/null; then break; fi
  echo "  (tools not fully landed in the container yet; settling $_try)"; sleep 10
done

echo "########## $ARM: source snapshot ##########"
for f in gemm_rs_mi300x.cpp gemm_rs_mi300x_hk_adapter.cuh; do
  cp -f "$ON/../$f" "$D/$f"
  sha256sum "$ON/../$f"
done

echo
echo "########## $ARM: resource tuples + ISA regen ##########"
docker exec dhk-gemmrs bash $ON/tools/m1_build.sh > "$D/m1_build.log" 2>&1
echo "m1_build exit=$?  (tuples come from build/m1a.log)"
docker exec dhk-gemmrs bash $ON/tools/m2_isa.sh  > "$D/m2_isa.log"  2>&1
echo "m2_isa exit=$?"
if grep -qE 'error:' "$D/m2_isa.log"; then
  echo "!!! COMPILE ERRORS !!!"; grep -m20 -E 'error:' "$D/m2_isa.log"; exit 1
fi
sed -n '/instantiation/,/waves\/SIMD ->/p' "$D/m2_isa.log"

echo
echo "########## $ARM: spill / scratch probes ##########"
grep -E 'scratch_store|scratch_load|VGPRs Spill|SGPRs Spill' "$D/m2_isa.log" | head -8
printf 'scratch_store in .s : %s\n' "$(grep -c scratch_store "$S")"
printf 'scratch_load  in .s : %s\n' "$(grep -c scratch_load  "$S")"
printf 's_setprio     in .s : %s\n' "$(grep -c s_setprio     "$S")"
cp -f "$S" "$D/gemm_rs_mi300x.gfx942.s"

echo
echo "########## $ARM: k-loop schedule shape ##########"
python3 "$EXP/sched_isa.py" "$S" '<256,256,32' > "$D/kloop_256.txt" 2>&1
python3 "$EXP/sched_isa.py" "$S" '<32,64,64'   > "$D/kloop_32.txt"  2>&1
sed -n '1,200p' "$D/kloop_256.txt"

echo
echo "########## $ARM: LDS/VMEM race check ##########"
bash $ON/experiments/exp_03_mainloop/lds_race_check.sh "$S" > "$D/lds_race.txt" 2>&1
tail -4 "$D/lds_race.txt"
grep -c 'HAZARD' "$D/lds_race.txt" || true
echo "########## $ARM: ISA PASS ##########"

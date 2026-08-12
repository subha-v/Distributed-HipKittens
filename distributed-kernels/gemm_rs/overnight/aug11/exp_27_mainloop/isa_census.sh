#!/usr/bin/env bash
# exp_27 — CPU-only ISA census of the GEMM k-loop. NO GPU, NO lease, NO write
# outside exp_27_mainloop/. hipcc -c --save-temps needs no device.
#
# Everything lands in overnight/aug11/exp_27_mainloop/isa/ so harness/build/
# and every other experiment's tree are untouched.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
EXP=$GEMM/overnight/aug11/exp_27_mainloop
ISA=$EXP/isa
mkdir -p "$ISA"

echo "################ provenance ################"
for f in gemm_rs_mi300x.cpp gemm_rs_mi300x_hk_adapter.cuh \
         gemm_rs_mi300x_constants.cuh; do
  printf '%-40s %s\n' "$f" "$(sha256sum "$GEMM/$f" | cut -c1-16)"
done
printf '%-40s %s\n' "hipcc" "$(hipcc --version 2>/dev/null | head -1)"
date -u +'%-40s %s' >/dev/null
echo "utc                                      $(date -u +%Y%m%dT%H%M%SZ)"

echo
echo "################ compile (-c, host CPU only) ################"
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

cd "$ISA" || exit 1
rm -f ./*.s ./*.o ./*.bc ./*.hipi ./*.cui 2>/dev/null
t0=$(date +%s)
hipcc -std=c++20 -O3 -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
  -ffast-math --offload-arch=gfx942 -DTK_MODNAME=gemm_rs_mi300x \
  -I"$REPO/include" -I"$REPO/include/pyutils" -I"$ROCM_PATH/include/hip" \
  -I"$PBINC" -I"$PYINC" -Wno-nan-infinity-disabled -ferror-limit=0 \
  -Rpass-analysis=kernel-resource-usage \
  --save-temps -c "$GEMM/gemm_rs_mi300x.cpp" -o "$ISA/exp27.o" \
  >"$ISA/compile.log" 2>&1
rc=$?
echo "compile exit=$rc  wall=$(( $(date +%s) - t0 ))s  (log: isa/compile.log)"
tail -2 "$ISA/compile.log"

S=$(ls "$ISA"/*gfx942*.s 2>/dev/null | head -1)
[ -z "$S" ] && S=$(ls "$ISA"/*.s 2>/dev/null | head -1)
echo "ISA file: $S  ($(wc -l <"$S" 2>/dev/null) lines)"
[ -z "$S" ] && { echo "NO ISA FILE — abort"; exit 1; }

echo
echo "################ resource tuple (from compile.log) ################"
grep -E 'Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|Spill|LDS Size' \
  "$ISA/compile.log" | sed 's/^.*remark: //' | sed 's/ \[-Rpass.*//' \
  | awk '/Function Name/{printf "\n%s\n", $0; next}{printf "   %s\n", $0}' \
  | head -120

echo
echo "################ whole-TU opcode inventory ################"
for pat in v_mfma_f32_16x16x16 v_mfma_f32_32x32x8 v_mfma \
           ds_read_b128 ds_read2_b64 ds_read_b64 ds_read_b32 \
           ds_write_b128 ds_write2_b64 ds_write_b64 ds_write_b32 \
           global_load_dwordx4 s_barrier scratch_store v_accvgpr; do
  printf '%-26s %s\n' "$pat" "$(grep -cE "^\s+$pat" "$S")"
done

echo
echo "################ k-loop census ################"
python3 "$EXP/kloop_hist.py" "$S" "$EXP/kloop_census.json"

echo
echo "################ housekeeping: drop the bulky temps ################"
rm -f "$ISA"/*.bc "$ISA"/*.hipi "$ISA"/*.cui "$ISA"/*.o 2>/dev/null
du -sh "$ISA"
echo "################ DONE ################"

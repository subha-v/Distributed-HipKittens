#!/usr/bin/env bash
# Diagnostics for the two failed fingerprint assertions. Read-only, CPU-only.
set -uo pipefail
EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_23_waterfall
OUT=$EXP/build

echo "===== A6: shape of the resource-usage remarks in the compile log ====="
grep -n -m 24 -E 'Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|Spill|LDS' \
  "$OUT/gemm_rs_w23_c.compile.log"

echo
echo "===== A6: the shipped harness log, for format comparison ====="
grep -n -m 12 -E 'Function Name|SGPRs|VGPRs' \
  /home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/harness/build/gemm_rs_mi300x.log

echo
echo "===== A4: how do c and null ISA differ? ====="
diff "$OUT/isa/gemm_rs_w23_c.s" "$OUT/isa/gemm_rs_w23_null.s" | head -40
echo "-- diff line count: $(diff "$OUT/isa/gemm_rs_w23_c.s" "$OUT/isa/gemm_rs_w23_null.s" | wc -l)"

echo
echo "===== A4: does the module name appear in the device ISA at all? ====="
for m in gemm_rs_w23_a gemm_rs_w23_b gemm_rs_w23_c gemm_rs_w23_null; do
  echo "  $m: $(grep -c "$m" "$OUT/isa/$m.s") occurrences of its own module name"
done

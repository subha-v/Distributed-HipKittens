#!/usr/bin/env bash
# Build counters.json from the collected rocprofv3 CSVs. No GPU work.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/aug11/exp_20_attribution
SRCD=/home/subvadla/dhk/distributed-kernels/gemm_rs
KMD5=$(tr -d '\r' < $SRCD/gemm_rs_mi300x.cpp | md5sum | cut -d' ' -f1)

echo "===== collected cells ====="
ls -1 "$OUT/prof" | grep -v '\.log$' | tr '\n' ' '; echo

python3 "$OUT/mk_counters_json.py" "$OUT/prof" "$OUT/counters.json" 2 4 \
  warm=2 meas=4 \
  arm=gemm_rs_abl_full \
  release_group=4 release_group_full_only=1 wgm4=0 negative_controls=0 \
  shape_table="32/64/128:NR56, 64/128/64:NR32, 128/192/32:NR32, 256/256/32:NR32, 256/256/32:NR32, 256/256/32:NR48" \
  kernel_md5="$KMD5" \
  node=banff-sc-cs47-05.dh170.dcgpu gpus=8xMI300X_gfx942_SPX \
  rocprofv3=1.1.0 clocks=perf_determinism_1900
rc=$?

echo
echo "===== the label control: fabric MUST collapse on emit-local ====="
grep -A14 '=== s6_emitlocal_g1' "$OUT/counters_run.log" | \
  grep -E 'TCC_EA0_WRREQ|NOT DRAM|fabric bytes|destined DRAM|of which'
exit $rc

#!/usr/bin/env bash
# Turn the completed attribution log into ablation.json. No GPU work.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/aug11/exp_20_attribution
SRCD=/home/subvadla/dhk/distributed-kernels/gemm_rs
LOG=$OUT/reattribute_run.log
CANON=$(ls -t $ON/experiments/logs/reattribute_*.log 2>/dev/null | head -1)
KMD5=$(tr -d '\r' < $SRCD/gemm_rs_mi300x.cpp | md5sum | cut -d' ' -f1)
AMD5=$(md5sum $ON/harness/ablate/gemm_rs_ablate.cpp | cut -d' ' -f1)

python3 "$OUT/mk_ablation_json.py" "$LOG" "$OUT/ablation.json" \
  iters=40 \
  expected_shape6_full_us=1670 tolerance_pct=6 \
  freshness=passed \
  release_group=4 release_group_full_only=1 wgm4=0 negative_controls=0 \
  tile_sweep=0 \
  shape_table="32/64/128:NR56, 64/128/64:NR32, 128/192/32:NR32, 256/256/32:NR32, 256/256/32:NR32, 256/256/32:NR48" \
  kernel_md5="$KMD5" ablate_scratch_md5="$AMD5" \
  node=banff-sc-cs47-05.dh170.dcgpu gpus=8xMI300X_gfx942_SPX cu_per_gpu=304 \
  clocks=perf_determinism_1900 \
  canonical_log="$CANON"

echo
echo "===== arm build fingerprints (proves the arms are this run's) ====="
ls -la --time-style=full-iso $ON/harness/build/gemm_rs_abl_*.so | awk '{print $6, $7, $5, $9}'
echo
echo "===== correctness of the `full` arm, from the run log ====="
grep -nE 'correct=|CELL gemm_rs_abl_full' "$LOG" 2>/dev/null | head -5
echo "(the driver only prints per-cell correctness to the cell's own stdout,"
echo " which it captures; the table's absence of FAULT/FAIL notes is the signal)"
grep -cE 'FAULT|FAIL' "$LOG"

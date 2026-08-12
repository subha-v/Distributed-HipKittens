#!/usr/bin/env bash
E22=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_22_timeline
docker exec -w "$E22" dhk-gemmrs python3 -u "$E22/phase_summary.py" \
  "$E22/events_ours_s5.json" "$E22/events_ours_s6.json" \
  --json "$E22/phase_summary.json" 2>&1 | tail -40

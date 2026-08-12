#!/usr/bin/env bash
E22=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_22_timeline
echo "--- alive? ---"; pgrep -af "go_gpu.sh" | head -3; echo "(none = finished)"
echo "--- tail go_gpu.log ---"; tail -45 "$E22/logs/go_gpu.log"
echo "--- artifacts ---"; ls -lt "$E22"/*.json "$E22"/*.csv "$E22"/b0_s5 2>/dev/null | head -20

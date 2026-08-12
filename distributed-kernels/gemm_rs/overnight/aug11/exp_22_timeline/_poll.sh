#!/usr/bin/env bash
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E22=$ON/aug11/exp_22_timeline
echo "--- alive? ---"; pgrep -af "go_armb.sh" | head -3; echo "(none = finished)"
echo "--- lease ---"; bash "$ON/tools/gpu_lease.sh" status 2>&1 | head -5
echo "--- tail ---"; tail -${TAIL:-50} "$E22/logs/go_armb.log"
echo "--- artifacts ---"
ls -lt "$E22"/*.json "$E22"/*.csv 2>/dev/null | head -12

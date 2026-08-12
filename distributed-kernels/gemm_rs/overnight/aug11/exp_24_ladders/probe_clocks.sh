#!/usr/bin/env bash
# READ-ONLY except for re-asserting the clock pin, which the charter mandates
# before any timing. The preflight in validate_dry.sh printed sclk at 120 MHz
# after calling `set_clocks.sh pin 1900`, so establish whether the pin took, is
# just reporting idle DPM state, or needs root.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
echo "=== tools/set_clocks.sh ==="
cat "$ON/tools/set_clocks.sh"
echo
echo "=== current perf level / sclk (idle) ==="
rocm-smi --showperflevel 2>&1 | sed -n '1,16p'
echo
rocm-smi --showgpuclocks 2>&1 | sed -n '1,16p'
echo "=== DONE ==="

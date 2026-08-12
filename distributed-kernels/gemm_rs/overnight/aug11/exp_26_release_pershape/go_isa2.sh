#!/usr/bin/env bash
# isa_diff2.sh compiles, so it runs INSIDE dhk-gemmrs. The host toolchain has no
# pybind11 and a different python; running a build script on the host is the
# mistake that already cost this experiment one round of arm modules.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
mkdir -p "$D/logs"
docker exec dhk-gemmrs bash "$D/isa_diff2.sh" 2>&1 | tee "$D/logs/isa_diff2.log"
exit ${PIPESTATUS[0]}

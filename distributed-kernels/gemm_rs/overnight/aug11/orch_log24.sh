#!/usr/bin/env bash
set -uo pipefail
A=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11
echo "===== exp_24 full_run.log tail ====="
tail -40 "$A/exp_24_ladders/logs/full_run.log" 2>&1
echo
echo "===== does any runner do its own KFD drain check? ====="
grep -n 'showpids' "$A/exp_24_ladders/run_ladders.sh" "$A/exp_24_ladders/full_runner.sh" \
     "$A/exp_26_release_pershape/campaign3.sh" 2>/dev/null | head -20
echo
echo "===== other drain checks in the tree that share the same blindness ====="
grep -rln 'showpids' /home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/tools/ 2>/dev/null
echo done

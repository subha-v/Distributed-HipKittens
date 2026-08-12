#!/usr/bin/env bash
E22=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_22_timeline
echo "===== re-score the M7 run already taken, against the right statistic ====="
docker exec -w "$E22" dhk-gemmrs python3 -u "$E22/sanity_check.py" \
  "$E22/m7_results.json" --json "$E22/sanity.json"
rc=$?
echo "sanity rc=$rc"
[ "$rc" != "0" ] && { echo "still FAIL -- not relaunching"; exit 1; }
echo
echo "===== relaunching arms (a) and (b) ====="
EXP22_REUSE_M7=1 bash "$E22/_launch.sh"

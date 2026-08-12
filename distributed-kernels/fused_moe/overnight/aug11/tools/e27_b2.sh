#!/usr/bin/env bash
# exp_27: fingerprint the arm-0 build that batch 1 actually launched, THEN start
# batch 2 (candidate, arm 1, pinned f113d73f, n=5).
set -uo pipefail
echo "################ FINGERPRINT OF THE ARM BATCH 1 LAUNCHED ################"
bash "$HOME/tools/e27_fp.sh"
cp -f "$HOME/overnight-scratch/e27/fp/j.text.bin" "$HOME/overnight-scratch/e27/fp/arm0.text.bin" 2>/dev/null
echo
echo "################ BATCH 2: CANDIDATE (arm 1) ################"
bash "$HOME/tools/e27_batch.sh" e27b2_cand f113d73f 1 5

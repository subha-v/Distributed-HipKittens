#!/usr/bin/env bash
bash "$HOME/tools/e27_poll.sh" e27b2_cand
echo
echo "################ FINGERPRINT OF THE ARM BATCH 2 LAUNCHED ################"
bash "$HOME/tools/e27_fp.sh"
cp -f "$HOME/overnight-scratch/e27/fp/j.text.bin" "$HOME/overnight-scratch/e27/fp/arm1.text.bin" 2>/dev/null
echo
echo "arm0 vs arm1 launched .text differ? (must differ)"
cmp -s "$HOME/overnight-scratch/e27/fp/arm0.text.bin" "$HOME/overnight-scratch/e27/fp/arm1.text.bin" \
  && echo "  ** IDENTICAL -- the two batches ran the SAME binary, STOP **" \
  || echo "  OK: the two batches ran different binaries"

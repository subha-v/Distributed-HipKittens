#!/usr/bin/env bash
# exp_27 batch 4: CANDIDATE, second independent batch (arm 1), pinned f113d73f.
# Waits for batch 3 to finish first -- ONE GPU job at a time.
set -uo pipefail
for i in $(seq 1 60); do
  pgrep -f 'tools/screen.sh' >/dev/null 2>&1 || break
  [ "$i" = 1 ] && echo "waiting for batch 3 to finish ..."
  sleep 10
done
echo "== batch 3 tail =="
tail -3 "$HOME/overnight-scratch/e27b3_ctl.driver.log" 2>/dev/null | cut -c1-160
echo
echo "################ FINGERPRINT OF THE ARM BATCH 3 LAUNCHED ################"
bash "$HOME/tools/e27_fp.sh"
echo
echo "################ BATCH 4: CANDIDATE (arm 1), second batch ################"
bash "$HOME/tools/e27_batch.sh" e27b4_cand f113d73f 1 5

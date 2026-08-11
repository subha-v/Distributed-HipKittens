#!/usr/bin/env bash
set -uo pipefail
for tag in mode0c8 mode1c0; do
  L="$HOME/exp01_$tag.log"
  echo "############################## $tag ##############################"
  echo "--- size ---"; wc -l "$L"
  echo "--- any python exception / config rejection ---"
  grep -n -i -E 'Error|Exception|Traceback|invalid|reject|assert|config|abort|raise' "$L" | head -25
  echo "--- last 25 lines ---"
  tail -n 25 "$L"
  echo
done

echo "############################## pullfb correctness detail ##############################"
grep -n -E '\[MARK\]|rel_L2|rel_L1|max_abs|pperr|control_fails|pass=' "$HOME/exp01_pullfb.log" | head -30
exit 0

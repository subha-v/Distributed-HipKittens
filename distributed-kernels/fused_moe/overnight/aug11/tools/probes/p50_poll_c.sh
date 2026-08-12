#!/usr/bin/env bash
# exp_24: poll the e24c campaign.
set -uo pipefail
OUT=$HOME/overnight-scratch/e24c_batch.out
CSV=$HOME/overnight-scratch/screen_e24c.csv
echo "=== alive?"; pgrep -af 'screen.sh e24c' | head -2
echo "=== progress"; grep -E '^\[|^SCREEN|completed run|starting run' "$OUT" 2>/dev/null | tail -10
echo "=== per-run progress inside the current campaign"
LOG=$(ls -t $HOME/overnight-scratch/e24c_*.log 2>/dev/null | head -1)
echo "log: $LOG"
[ -n "$LOG" ] && grep -E 'starting run|completed run|MOK GATE|MPS SOAK|campaign complete' "$LOG" | tail -12
echo
if [ -f "$CSV" ]; then
python3 - "$CSV" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
h = rows[0]; i = {k: h.index(k) for k in h}
cols = ["idx","cfg","status","iters","mps_us","prod_us","pf6gm_us",
        "ratio_vs_prod","ratio_vs_pf6gm","pf6gm_vs_prod","gate_pass",
        "control_fails","pperr","soak_epochs","mps_hsaco"]
print(" | ".join(cols))
for r in rows[1:]:
    print(" | ".join(r[i[c]] if i[c] < len(r) else "" for c in cols))
PY
else
  echo "(no csv yet)"
fi
exit 0

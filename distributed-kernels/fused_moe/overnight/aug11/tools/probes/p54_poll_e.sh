#!/usr/bin/env bash
# exp_24: poll the e24e ratchet-confirmation campaign (g=353 x2).
set -uo pipefail
OUT=$HOME/overnight-scratch/e24e_batch.out
CSV=$HOME/overnight-scratch/screen_e24e.csv
echo "=== alive?"; pgrep -af 'screen.sh e24e' | head -2
echo "=== progress"; grep -E '^\[' "$OUT" 2>/dev/null | tail -6
if [ -f "$CSV" ]; then
echo
python3 - "$CSV" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
h = rows[0]; i = {k: h.index(k) for k in h}
cols = ["idx","cfg","status","mps_us","prod_us","pf6gm_us","ratio_vs_prod",
        "ratio_vs_pf6gm","pf6gm_vs_prod","gate_pass","control_fails","pperr",
        "soak_epochs","mps_hsaco"]
print(" | ".join(cols))
for r in rows[1:]:
    print(" | ".join(r[i[c]] if i[c] < len(r) else "" for c in cols))
PY
else
  echo "(no csv yet)"
fi
exit 0

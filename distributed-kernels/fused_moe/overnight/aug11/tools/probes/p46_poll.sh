#!/usr/bin/env bash
# exp_24: poll the e24a batch.
set -uo pipefail
OUT=$HOME/overnight-scratch/e24a_batch.out
CSV=$HOME/overnight-scratch/screen_e24a.csv
echo "=== batch alive?"
pgrep -af 'screen.sh e24a' | head -2
echo "=== progress lines"
grep -E '^\[|hsaco|SCREEN ' "$OUT" 2>/dev/null | tail -22
echo
echo "=== csv so far (idx,cfg,status,prod,pf6gm,mps,ratio_prod,ratio_pf6gm,pf6gm_prod,gate,control,pperr,soak,tsplan,tsM6,tsM7,hsaco_before,hsaco_after)"
if [ -f "$CSV" ]; then
python3 - "$CSV" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
if not rows: sys.exit(0)
h = rows[0]
idx = {k: h.index(k) for k in h}
cols = ["idx","cfg","status","prod_us","pf6gm_us","mps_us","ratio_vs_prod",
        "ratio_vs_pf6gm","pf6gm_vs_prod","gate_pass","control_fails","pperr",
        "soak_epochs","ts_planM3toM5_us","ts_M6_us","ts_M7_us","ts_combine_us",
        "hsaco_before","hsaco_after"]
print(" | ".join(c.replace("_us","").replace("ratio_vs_","v") for c in cols))
for r in rows[1:]:
    print(" | ".join((r[idx[c]] if idx[c] < len(r) else "") for c in cols))
PY
else
  echo "(no csv yet)"
fi
exit 0

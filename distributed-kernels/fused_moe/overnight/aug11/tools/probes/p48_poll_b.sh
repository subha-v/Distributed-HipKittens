#!/usr/bin/env bash
# exp_24: poll e24b and print the paired table.
set -uo pipefail
OUT=$HOME/overnight-scratch/e24b_batch.out
CSV=$HOME/overnight-scratch/screen_e24b.csv
echo "=== alive?"; pgrep -af 'screen.sh e24b' | head -2
echo "=== tail"; grep -E '^\[' "$OUT" 2>/dev/null | tail -6
echo
for C in "$CSV" "$HOME/overnight-scratch/screen_e24a.csv"; do
[ -f "$C" ] || continue
echo "===== $(basename "$C")"
python3 - "$C" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
h = rows[0]; i = {k: h.index(k) for k in h}
def g(r, k):
    j = i[k]
    return r[j] if j < len(r) else ""
cols = [("idx","idx"),("cfg","cfg"),("status","st"),("mps_us","mps_us"),
        ("prod_us","prod_us"),("pf6gm_us","pf6gm"),("ratio_vs_prod","v_prod"),
        ("ts_planM3toM5_us","plan35"),("ts_M6_us","M6"),("ts_M7_us","M7"),
        ("ts_combine_us","comb"),("ts_m2_to_end_us","m2end"),
        ("ts_servicedrain_us","drain"),("mps_hsaco","hsaco")]
w = [len(n) for _, n in cols]
out = []
for r in rows[1:]:
    vals = [g(r, k) for k, _ in cols]
    vals[1] = vals[1].replace("C=16,","").replace("mode=12,","").replace("flush_rows=16","fr16")
    out.append(vals)
    w = [max(a, len(b)) for a, b in zip(w, vals)]
print("  ".join(n.ljust(x) for (_, n), x in zip(cols, w)))
for v in out:
    print("  ".join(a.ljust(x) for a, x in zip(v, w)))
PY
echo
done
exit 0

#!/usr/bin/env bash
# exp_36 poll: batch progress + the rows landed so far.
set -u
E36="$HOME/e36"
TAG="${E36_TAG:-}"
echo "== $(date -u +%FT%TZ) =="
echo "-- gpu --"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1" "$2}'
echo "-- screen procs --"
pgrep -af 'screenT.sh|rc_T.sh|docker run' | cut -c1-140
echo "-- batch logs --"
for f in "$E36"/*.batchlog; do
  [ -e "$f" ] || continue
  echo "### $f  (mtime $(stat -c %y "$f" | cut -c1-19))"
  tail -6 "$f"
done
echo "-- csv rows --"
for c in "$E36"/scratch/screen_*.csv; do
  [ -e "$c" ] || continue
  echo "### $c"
  python3 - "$c" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
h = rows[0]
i = {n: k for k, n in enumerate(h)}
for r in rows[1:]:
    g = lambda n: r[i[n]] if i.get(n) is not None and i[n] < len(r) else ""
    print(f"{g('idx'):>2} {g('cfg'):<48} {g('status'):<14} "
          f"prod={g('prod_us'):>8} pf6={g('pf6gm_us'):>8} mps={g('mps_us'):>8} "
          f"r={g('ratio_vs_prod'):>7} gate={g('gate_pass')} pperr={g('pperr')} "
          f"soak={g('soak')}/{g('soak_epochs')} spin={g('spin_ok_max')}/{g('spin_fail_max')} "
          f"M6={g('ts_M6_us')} M7={g('ts_M7_us')} comb={g('ts_combine_us')} note={g('note')[:40]}")
PY
done
exit 0

#!/usr/bin/env bash
set -uo pipefail
S=$HOME/overnight-scratch
echo "== $(date -u) =="
pgrep -af 'tools/screen.sh|torchrun' | head -3 || echo "(nothing running)"
echo
tail -12 "$S/e27camp_cand.driver.log" 2>/dev/null | cut -c1-200
echo
echo "== gate lines so far =="
for f in $(ls -1t "$S/e27camp_cand"_*.log 2>/dev/null | head -1); do
  echo "---- $(basename "$f") ----"
  grep -cE 'run[0-9]+ ' "$f" 2>/dev/null | head -1
  grep -hE '\[MOK GATE\]|\[MPS SOAK\]|\[MARK\] control_fails|\[POISON SELFTEST\]|arm_p50|blocked|Traceback|Error' "$f" 2>/dev/null | tail -25
  echo "--- progress: which run of 5 ---"
  grep -oE 'run[0-9]+' "$f" 2>/dev/null | sort -u | tr '\n' ' '
  echo
done
echo
echo "== summary.json if present =="
D=$(ls -1dt "$HOME/k0-mok-e27camp_cand"/*/ 2>/dev/null | head -1)
echo "outdir: $D"
if [ -f "$D/summary.json" ]; then
  python3 -c "
import json
d=json.load(open('$D/summary.json'))
ap=d.get('arm_p50_us',{})
for a,v in ap.items(): print(f'  {a:16s} median={v.get(\"median\")}  n={len(v.get(\"values\",[]) or [])}  values={v.get(\"values\")}')
p=ap.get('production',{}).get('median'); 
for a in ('pf6gm_mega','mps_mega'):
    m=ap.get(a,{}).get('median')
    if p and m: print(f'  ratio {a} / production = {m/p:.4f}')
f=ap.get('pf6gm_mega',{}).get('median'); m=ap.get('mps_mega',{}).get('median')
if f and m: print(f'  ratio mps_mega / pf6gm_mega = {m/f:.4f}')
"
else echo "(no summary.json yet)"; fi

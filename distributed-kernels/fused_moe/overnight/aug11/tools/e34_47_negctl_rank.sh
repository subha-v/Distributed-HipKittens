#!/usr/bin/env bash
# exp_34: close condition 4 -- find the RANK-LEVEL evidence for the control's
# failure (the harness prints pperr from rank 0 only, and the predicted failure
# is on rank 7).
set -uo pipefail
L=$(ls -t $HOME/overnight-scratch/e34neg_*.log | head -1)
D=$(ls -td $HOME/k0-mok-e34neg/*/ | head -1)
echo "### log: $L"; echo "### outdir: $D"
echo
echo "############ poison / survivor lines ############"
grep -hE "POISON|survivors|nonfinite" "$L" | head -20
echo
echo "############ any per-rank diagnostic ############"
grep -hE "rank=[0-9]|RANK [0-9]|\[rank" "$L" | grep -viE "torch|WARNING" | head -20
echo
echo "############ blocking message ############"
grep -hE "blocked|refus|Traceback|RuntimeError|assert" "$L" | head -20
echo
echo "############ rank JSONs produced? ############"
ls -l "$D" | head -20
for f in "$D"rank_*.json "$D"*rank*.json; do
  [ -f "$f" ] || continue
  echo "--- $f"
  python3 -c "
import json,sys
d=json.load(open('$f'))
def walk(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items():
            if 'pperr' in str(k) or 'rank'==k: print(p+'/'+str(k), '=', v if not isinstance(v,(dict,list)) else type(v).__name__)
            walk(v,p+'/'+str(k))
walk(d)" 2>/dev/null | head -20
done
echo
echo "############ the 29,360,128 number in context ############"
grep -hE "29360128" "$L" | head -5
python3 -c "print('T*H =', 4096*7168, '= one rank full output')"
echo "===DONE==="

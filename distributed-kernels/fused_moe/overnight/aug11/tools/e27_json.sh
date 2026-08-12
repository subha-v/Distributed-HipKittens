#!/usr/bin/env bash
# exp_27: is there more precision available than the 6 printed [MOK GATE] digits?
#
# The control batch -- ONE binary, .text-fingerprinted as identical to the ratchet
# -- printed two different `relative` values across its five runs
# (0.008293 / 0.008294), and `production`, an arm exp_27 does not touch, printed
# two different max_abs values (0.031250 / 0.027344). So the printed digits are
# NOT run-reproducible for a fixed binary, and "reproduce the ratchet's digits
# exactly" cannot be evaluated as stated. Look for a higher-precision record.
set -uo pipefail
for TAG in e27b1_ctl e27b2_cand; do
  D=$(ls -1dt "$HOME/k0-mok-${TAG}"/*/ 2>/dev/null | head -1)
  echo "################ $TAG -> $D ################"
  [ -n "$D" ] || continue
  echo "--- files ---"
  find "$D" -maxdepth 2 -type f | head -20
  echo "--- summary.json keys ---"
  python3 -c "
import json,sys
d=json.load(open('$D/summary.json'))
def walk(o,p='',dep=0):
    if dep>2: return
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)): walk(v,p+'/'+str(k),dep+1)
            else: print(f'{p}/{k} = {v!r}')
    elif isinstance(o,list) and o and not isinstance(o[0],(dict,list)):
        print(f'{p} = {o!r}'[:200])
walk(d)
" 2>&1 | grep -iE 'max_abs|relative|rel_|l1|l2|nonfinite|gate|pperr' | head -30
  echo "--- a rank JSON, correctness fields at full precision ---"
  RJ=$(find "$D" -name '*rank*.json' | head -1)
  echo "rank json: $RJ"
  [ -n "$RJ" ] && python3 -c "
import json
d=json.load(open('$RJ'))
def walk(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v,p+'/'+str(k))
    elif isinstance(o,list):
        if o and not isinstance(o[0],(dict,list)): print(f'{p} = {o!r}'[:240])
        else:
            for i,v in enumerate(o[:3]): walk(v,f'{p}[{i}]')
    else:
        print(f'{p} = {o!r}')
walk(d)
" 2>&1 | grep -iE 'max_abs|relative|rel_|nonfinite|arm|pperr' | head -40
  echo
done
echo "===DONE==="

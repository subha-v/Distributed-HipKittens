#!/usr/bin/env bash
echo "===summary.json (exp21/e21_m13_C16)==="
python3 -m json.tool ~/k0-mok-exp21/e21_m13_C16/summary.json | head -80
echo; echo "===summary.json dec21a==="
python3 -m json.tool ~/k0-mok-dec21a/summary.json | head -80
echo; echo "===screen_e21.out==="
cat ~/k0-mok-exp21/screen_e21.out
echo; echo "===timestamps=1 MPS TS line==="
grep -h 'MPS TS\|MPS SPIN' ~/overnight-scratch/ts2_C64g1mode2flush_rows16timestamps1.log
grep -h 'MPS TS\|MPS SPIN' ~/overnight-scratch/ts2_C32g1mode2flush_rows16timestamps1.log
echo; echo "===any surviving driver scripts==="
ls -la ~/overnight-scratch/*.sh 2>&1 | head
find ~ -maxdepth 2 -name 'screen*.sh' -o -maxdepth 2 -name 'drive*.sh' 2>/dev/null | head -20
exit 0

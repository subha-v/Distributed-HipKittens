#!/usr/bin/env bash
# exp_34: quote the ladder evidence out of the smoke log, and confirm the
# descriptor really carried mode=14 / C=0.
set -uo pipefail
L=$(ls -t $HOME/overnight-scratch/e34smoke_*.log | head -1)
echo "### log: $L"
echo
echo "############ ladder evidence ############"
grep -hE "\[MOK GATE\]|\[MARK\] control_fails|\[MPS SOAK\]|\[POISON SELFTEST\]|\[POISON\] |\[MPS SPIN\]" "$L" | sort -u | head -40
echo
echo "############ pperr values seen (bit 26 = 67108864 must be absent) ############"
grep -hoE "pperr=[0-9]+" "$L" | sort | uniq -c | sort -rn | head
echo
echo "############ descriptor dump: mode / C actually used ############"
grep -hE "DESC|desc|MPS CFG|cfg word|mps_cfg" "$L" | head -20
echo
echo "############ mode-14-specific evidence: service drain stamp ############"
grep -hE "\[MPS TS DELTA\]|\[MPS TS SPLIT\]" "$L" | tail -4
echo
echo "############ per-arm p50 from summary.json ############"
S=$HOME/k0-mok-e34smoke/*/summary.json
python3 - $S <<'PY'
import json,sys,glob
for p in sys.argv[1:]:
    d=json.load(open(p))
    print(p.split('/')[-2])
    for arm,v in d.get("arm_p50_us",{}).items():
        print(f"   {arm:14s} median={v.get('median')}")
PY
echo "===DONE==="

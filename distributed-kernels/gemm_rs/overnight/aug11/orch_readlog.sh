#!/usr/bin/env bash
# Read-only: print the newest reattribute log's table plus whatever exp_20 has
# written so far. Orchestrator use; touches nothing.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
L=$(ls -t "$ON"/experiments/logs/reattribute_*.log 2>/dev/null | head -1)
echo "===== newest log: $L ====="
tail -60 "$L" 2>&1

echo
echo "===== exp_20 artifacts ====="
find "$ON"/aug11/exp_20_attribution -maxdepth 1 -type f -printf '%10s  %TY-%Tm-%Td %TH:%TM  %f\n' 2>&1 | sort -k4

echo
echo "===== ablation.json if present ====="
if [ -f "$ON"/aug11/exp_20_attribution/ablation.json ]; then
  python3 -c "
import json,sys
d=json.load(open('$ON/aug11/exp_20_attribution/ablation.json'))
print('keys:', list(d.keys()))
for s in d.get('shapes', []):
    st=s.get('stages_us', {})
    print(f\"{s.get('shape'):>22}  full={s.get('full_us')}  \" + '  '.join(f'{k}={v}' for k,v in st.items()))
" 2>&1
else
  echo "  not written yet"
fi
echo "done"

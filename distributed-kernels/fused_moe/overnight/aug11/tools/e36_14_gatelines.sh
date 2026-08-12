#!/usr/bin/env bash
# exp_36: dump the exact gate lines from the control screen so the CSV parser's
# soak=MALFORMED can be adjudicated against the real log format.
set -u
L="$(ls -t $HOME/e36/scratch/e36c1_*.log 2>/dev/null | head -1)"
echo "log: $L"
echo "=== MPS SOAK ==="
grep -n 'MPS SOAK' "$L" | head -12
echo "=== POISON ==="
grep -n 'POISON' "$L" | head -12
echo "=== MOK GATE / MARK ==="
grep -n '\[MOK GATE\]\|\[MARK\] control_fails\|pperr=' "$L" | head -20
echo "=== MPS TS / SPIN ==="
grep -n '\[MPS TS\]\|\[MPS TS SPLIT\]\|\[MPS TS DELTA\]\|\[MPS SPIN\]' "$L" | tail -12
echo "=== shape echo ==="
grep -n 'exp_36 shape' "$L" | head -3
echo "=== route stats from rank0 json ==="
J="$(ls -t $HOME/k0-mok-e36c1/*/run1/k0pf_mok_synthetic_rank0.json 2>/dev/null | head -1)"
echo "json: $J"
python3 - "$J" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
si = d.get("synthetic_inputs", {})
keep = ["fanout_mean","fanout_max","expert_assignment_min","expert_assignment_max",
        "expert_assignments_per_destination_rank","hidden_shape","rank","seed"]
for k in keep:
    if k in si: print(k, "=", si[k])
print("top keys:", sorted(d.keys())[:25])
PY
exit 0

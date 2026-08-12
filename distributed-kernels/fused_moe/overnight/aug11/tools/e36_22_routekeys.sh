#!/usr/bin/env bash
# exp_36: where do the realised-route statistics actually live in the rank JSON?
set -u
J="$(ls -t $HOME/k0-mok-e36T4096/1_*/run1/k0pf_mok_synthetic_rank0.json 2>/dev/null | head -1)"
echo "json: $J"
python3 - "$J" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("top:", sorted(d.keys()))
si = d.get("synthetic_inputs")
print("synthetic_inputs type:", type(si))
if isinstance(si, dict):
    print("si keys:", sorted(si.keys()))
    for k in ("expert_assignments_per_destination_rank", "fanout_mean",
              "expert_assignment_min", "expert_assignment_max", "hidden_shape"):
        print(" ", k, "=", str(si.get(k))[:120])
PY
echo "=== campaign A progress ==="
tail -4 "$HOME/e36/e36campA.batchlog"
ls -1 "$HOME"/k0-mok-e36campA/*/run* -d 2>/dev/null | tail -6
exit 0

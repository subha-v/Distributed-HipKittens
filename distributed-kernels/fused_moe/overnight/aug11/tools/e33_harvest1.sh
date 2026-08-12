#!/usr/bin/env bash
# exp_33 harvest stage 1: node idleness after the run, full gate ladder, every
# stamp line from every rotation, and the exact shapes of summary.json / rank JSON.
set -uo pipefail
OUT="$HOME/k0-mok-e33a/1_C16g353mode12flush_rows16timestamps1_20260812T085807Z"
LG="$HOME/overnight-scratch/e33a_1_C16g353mode12flush_rows16timestamps1_20260812T085807Z.log"

echo "=== node idle after run ==="
pgrep -af 'torchrun' 2>/dev/null || echo "(none: torchrun)"
pgrep -af 'mpirun' 2>/dev/null || echo "(none: mpirun)"
/opt/rocm/bin/rocm-smi --showpids 2>&1 | grep -E '^[0-9]+[ \t]' || echo "(no KFD rows)"
ls -ld /tmp/k0_mok_synthetic_gpu_lock 2>&1 || echo "(no lock dir)"

echo
echo "=== campaign rc / dirs ==="
ls -1 "$OUT"
echo "log lines: $(wc -l < "$LG")"

echo
echo "=== FULL GATE LADDER (all occurrences) ==="
grep -nE '\[MOK GATE\]|\[MARK\] control_fails|\[MPS SOAK\]|\[POISON SELFTEST\]|\[POISON\] |pperr=' "$LG"

echo
echo "=== ALL STAMP LINES, in log order ==="
grep -nE '\[MPS TS\]|\[MPS TS SPLIT\]|\[MPS TS DELTA\]|\[MPS SPIN\]' "$LG"

echo
echo "=== ALL K0PF PROFILE production LINES ==="
grep -nE '\[K0PF PROFILE\] production' "$LG"

echo
echo "=== K0PF GATE lines ==="
grep -nE '\[K0PF GATE\]' "$LG" | sed -n '1,12p'

echo
echo "=== run boundaries (to map stamp lines -> rotation) ==="
grep -nE 'starting run=|completed run=' "$LG"

echo
echo "=== summary.json shape ==="
python3 - "$OUT/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("top keys:", sorted(d.keys()))
print(json.dumps({k:d[k] for k in ("arm_p50_us","arm_p95_us") if k in d}, indent=1)[:2500])
for k in ("primary_mok_results","config","meta","runs"):
    if k in d:
        print("---",k,"---")
        print(json.dumps(d[k], indent=1)[:1500])
PY

echo
echo "=== rank JSON shape (run1 rank0) ==="
python3 - "$OUT/run1/k0pf_mok_synthetic_rank0.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("top keys:", sorted(d.keys()))
def peek(k):
    if k in d:
        print("---",k,"---"); print(json.dumps(d[k], indent=1)[:1800])
for k in ("stage_profile","mok_eager","mps_soak","arms","warmup_iters","timed_iters","mps_cfg","mok_warmup_iters"):
    peek(k)
print("=== any key containing 'ts' or 'stamp' or 'mps' ===")
def walk(o,p=""):
    if isinstance(o,dict):
        for k,v in o.items():
            kp=p+"/"+str(k)
            if any(s in str(k).lower() for s in ("stamp","_ts","ts_","mps")): print(kp, type(v).__name__)
            walk(v,kp)
walk(d)
PY
echo "=== HARVEST1 DONE ==="
exit 0

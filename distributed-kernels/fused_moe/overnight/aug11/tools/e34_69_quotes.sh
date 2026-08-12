#!/usr/bin/env bash
# exp_34: harvest the LITERAL ladder lines that result.md must quote, from the
# C=0 first-run log and from the protocol negative control, plus the rank-7 pperr
# decode that proves the control failed on the intended rank only.
set -uo pipefail
echo "############ LADDER: first GPU run, C=0 (tag e34smoke) ############"
L=$(ls -t $HOME/overnight-scratch/e34smoke_*.log | head -1); echo "log: $L"
grep -hE "\[MOK GATE\] mps_mega|\[MARK\] control_fails|\[POISON SELFTEST\]|\[POISON\] |\[MPS SOAK\]|\[MPS TS|\[MPS SPIN\]" "$L" | sort -u
echo
echo "############ NEGATIVE CONTROL (tag e34neg) ############"
N=$(ls -t $HOME/overnight-scratch/e34neg_*.log | head -1); echo "log: $N"
grep -hE "\[MOK GATE\] mps_mega|\[MARK\]|SRC_REV|survivors=|pperr" "$N" | sort -u | head -14
echo
echo "---- per-rank pperr from the control's rank JSONs (bit 25 = 33554432) ----"
D=$(ls -dt $HOME/k0-mok-e34neg/*/ | head -1)
python3 - "$D" <<'PY'
import glob, json, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "**", "k0pf_mok_synthetic_rank*.json"),
                          recursive=True)):
    j = json.load(open(f))
    e = (j.get("eager") or {}).get("mps_mega") or {}
    p = e.get("pperr")
    bits = [b for b in range(32) if p and (p >> b) & 1]
    print(f"  rank {j.get('rank')}: pperr={p} bits={bits} "
          f"mok_correctness.pass={((j.get('mok_correctness') or {}).get('mps_mega') or {}).get('pass')}")
PY
echo
echo "############ bit 26 (K0P6_MPS_ERR_SERVICE=67108864) audit over EVERY mode-14 run ############"
python3 - <<'PY'
import glob, json, os, re
HOME = os.path.expanduser("~")
tot = bad = 0
vals = set()
for tag in ("e34smoke", "e34c1", "e34e", "e34f"):
    for f in glob.glob(f"{HOME}/k0-mok-{tag}/*/**/k0pf_mok_synthetic_rank*.json", recursive=True):
        if "mode14" not in f:
            continue
        try:
            j = json.load(open(f))
        except Exception:
            continue
        def walk(o, path=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield from walk(v, path + "/" + str(k))
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    yield from walk(v, path + f"[{i}]")
            else:
                yield path, o
        for p, v in walk(j):
            if p.lower().endswith("pperr") and isinstance(v, int):
                tot += 1
                vals.add(v)
                if v & 67108864:
                    bad += 1
print(f"mode-14 pperr readings checked={tot}  distinct values={sorted(vals)}  "
      f"with bit 26 set={bad}")
PY
echo "===DONE==="

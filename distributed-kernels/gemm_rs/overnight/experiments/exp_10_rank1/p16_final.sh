#!/usr/bin/env bash
# exp_10 -- final aggregate over all six shapes, plus the provenance record.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/experiments/exp_10_rank1/vs_logs
ARM=$ON/compbench/rank1
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py

echo "===== provenance ====="
echo "frozen source:"
sha256sum "$RANK1"
echo "expected     : 7940fcb81df06c1d8b1e1a77051f23c934149a688441ef48b2751b3f336f0dc5"
echo "staged copy  :"
sha256sum "$ARM/submission.py"
echo "diff line count (frozen vs staged):"
diff "$RANK1" "$ARM/submission.py" | grep -c '^[<>]'
echo "iris revision: $(git -C /home/subvadla/amd-master/iris rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "bias config  : VS_FORCE_BIAS=1 for shapes 1,4,6 (reproduces eval.py's own parser)"

echo
echo "===== which shapes have results, and in what bias config? ====="
python3 - "$OUT" <<'PYEOF'
import glob, json, os, sys
root = sys.argv[1]
for i in range(6):
    files = sorted(glob.glob(os.path.join(root, f"vs_s{i}.rank*.json")))
    ok = 0
    forced = present = None
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d:
            ok += 1
            forced = d.get("bias_forced")
            present = d.get("bias_present")
    print(f"  shape index {i}: {ok}/8 ranks reported  bias_forced={forced} "
          f"bias_present={present}")
PYEOF

echo
python3 "$ON/experiments/exp_10_rank1/vs_report.py" "$OUT"
echo "===== DONE ====="

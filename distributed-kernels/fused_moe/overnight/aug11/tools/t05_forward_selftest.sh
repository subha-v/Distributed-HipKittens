#!/usr/bin/env bash
# t05: forward K0_MOK_POISON_SELFTEST into the campaign containers. Without this
# the default lives inside ab.py and the off-switch is a lie from the host --
# the documented K0_MPS_SKIP_LAUNCH trap.
set -uo pipefail
RC="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/run_campaign.sh"
python3 - "$RC" <<'PY'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = '      -e "K0_MOK_POISON_OUT=${K0_MOK_POISON_OUT:-1}" \\\n'
if a not in s:
    print("ANCHOR MISSING"); sys.exit(3)
if "K0_MOK_POISON_SELFTEST" in s:
    print("already present"); sys.exit(0)
add = '      -e "K0_MOK_POISON_SELFTEST=${K0_MOK_POISON_SELFTEST:-1}" \\\n'
open(p, "w", encoding="utf-8").write(s.replace(a, a + add))
print("forwarded")
PY
rc=$?
[ "$rc" -le 0 ] || exit "$rc"
bash -n "$RC" && echo "bash -n OK"
grep -n 'K0_MOK_POISON' "$RC"
echo "sha: $(sha256sum "$RC" | cut -c1-16)"
exit 0

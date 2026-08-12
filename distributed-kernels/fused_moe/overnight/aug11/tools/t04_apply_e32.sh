#!/usr/bin/env bash
# t04: apply exp_32_poison.patch + the V2a detector self-test to the node harness.
# Backs up first, validates syntax after, prints every changed region.
set -uo pipefail
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
DHK="$HOME/Distributed-HipKittens"
PATCH="$DHK/distributed-kernels/fused_moe/overnight/aug11/exp_32_gate_hardening/exp_32_poison.patch"
AB="$K0/prefill_opt/host/e004pf_k0pf_ab.py"
RC="$K0/benchmarks/mok_synthetic_prefill/run_campaign.sh"
BK="$HOME/harness-backups/e32/$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$BK"
cp -p "$AB" "$BK/e004pf_k0pf_ab.py.pre"
cp -p "$RC" "$BK/run_campaign.sh.pre"
echo "== backup =="; ls -l "$BK"
echo "pre sha: $(sha256sum "$AB" | cut -c1-16) $(sha256sum "$RC" | cut -c1-16)"

cd "$K0" || exit 1
echo "== git apply =="
git apply -p1 --verbose "$PATCH" || { echo "APPLY FAILED"; exit 3; }
echo "applied. post sha: $(sha256sum "$AB" | cut -c1-16) $(sha256sum "$RC" | cut -c1-16)"
echo "ab.py lines: $(wc -l < "$AB")  (was 7309)"

echo "== insert V2a detector self-test =="
python3 - "$AB" <<'PY'
import sys, io
p = sys.argv[1]
src = open(p, encoding="utf-8").read()
anchor = ('    if name not in _PROD_LIKE:\n'
          '        R.setdefault("poison", {})[name] = _poison_report("eager", name)\n')
n = src.count(anchor)
if n != 1:
    print(f"ANCHOR COUNT {n} -- refusing"); sys.exit(4)
add = (
'        # exp_32 V2a: prove the detector is wired to a GATE, not just a print.\n'
'        # Poison ONE row of an otherwise-correct buffer; mok_gate MUST fail.\n'
'        # Restores the row bit-for-bit, so this cannot perturb any later gate.\n'
'        if MOK_POISON_OUT and os.environ.get("K0_MOK_POISON_SELFTEST", "1") != "0":\n'
'            _st_row = cand_out[T - 1].clone()\n'
'            cand_out.view(torch.int16)[T - 1].fill_(MOK_POISON_I16)\n'
'            _st = mok_gate(_obuf(name))\n'
'            cand_out[T - 1].copy_(_st_row)\n'
'            R["poison"][name + "_selftest_fails"] = bool(not _st["pass"])\n'
'            if rank == 0:\n'
'                print(f"[POISON SELFTEST] arm={name} one_row_poisoned_fails="\n'
'                      f"{not _st[\'pass\']} nonfinite={_st[\'nonfinite\']} "\n'
'                      f"relative={_st[\'relative_error\']}", flush=True)\n'
)
open(p, "w", encoding="utf-8").write(src.replace(anchor, anchor + add))
print("V2a inserted")
PY
[ $? -eq 0 ] || { echo "V2A INSERT FAILED"; exit 5; }

echo "== validate =="
python3 -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read()); print('ast.parse OK')" "$AB" || exit 6
bash -n "$RC" && echo "bash -n OK" || exit 7
echo "post-V2a sha: $(sha256sum "$AB" | cut -c1-16)  lines $(wc -l < "$AB")"

echo "== changed regions (diff vs backup) =="
diff -u "$BK/e004pf_k0pf_ab.py.pre" "$AB" | head -260
echo "---- run_campaign.sh ----"
diff -u "$BK/run_campaign.sh.pre" "$RC"
echo "== grep: all poison sites =="
grep -n '_poison_out()\|_poison_report\|_poison_count\|MOK_POISON\|POISON SELFTEST\|_mok_eager_stash' "$AB"
echo "== backup dir: $BK"
echo "== done =="
exit 0

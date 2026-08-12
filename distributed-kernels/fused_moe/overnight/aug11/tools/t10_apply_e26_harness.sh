#!/usr/bin/env bash
# t10: exp_26 activate.md §4a-4d -- install + route the vendored phase-1 body.
# 4a/4b are mandatory (without them the JIT compile dies FileNotFoundError).
# 4c/4d keep the source contract and the provenance hash pointed at the file
# that is actually compiled. The donor n2_phase1_gm.cpp STAYS installed:
# pf6gm_mega still includes it and that is what preserves the reference arm.
set -uo pipefail
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
AB="$K0/prefill_opt/host/e004pf_k0pf_ab.py"
BK="$HOME/harness-backups/e26/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BK"; cp -p "$AB" "$BK/e004pf_k0pf_ab.py.pre"
echo "pre sha $(sha256sum "$AB" | cut -c1-16)  backup $BK"

python3 - "$AB" <<'PY'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
edits = [
 # ---- 4a: allowlist. Anything not in PF6_N2_FILES is never copied into
 #      KERNELS_DIR and the JIT compile raises FileNotFoundError.
 ('4a',
  '    ("n2_phase2_gm_mps.cpp",) if PF6MPS_REQUESTED else ()\n)\n',
  '    ("n2_phase2_gm_mps.cpp",) if PF6MPS_REQUESTED else ()\n'
  ') + (\n'
  '    # exp_26: vendored phase-1 body for mps_mega. The donor n2_phase1_gm.cpp\n'
  '    # above STAYS installed -- pf6gm_mega still includes it.\n'
  '    ("n2_phase1_gm_mps.cpp",) if PF6MPS_REQUESTED else ()\n'
  ')\n'),
 # ---- 4b: route it to the DHK tree, not amd-master.
 ('4b',
  '                PF6MPS_SOURCE_DIR\n'
  '                if _source_name == "n2_phase2_gm_mps.cpp"\n'
  '                else PF6_N2_SOURCE_DIR\n',
  '                PF6MPS_SOURCE_DIR\n'
  '                if _source_name in ("n2_phase1_gm_mps.cpp",\n'
  '                                    "n2_phase2_gm_mps.cpp")\n'
  '                else PF6_N2_SOURCE_DIR\n'),
 # ---- 4c: validate the phase body mps_mega actually includes.
 ('4c',
  '                os.path.join(PF6_N2_SOURCE_DIR, "n2_phase1_gm.cpp"),\n'
  '                os.path.join(PF6MPS_SOURCE_DIR, "n2_phase2_gm_mps.cpp"),\n',
  '                os.path.join(PF6MPS_SOURCE_DIR, "n2_phase1_gm_mps.cpp"),\n'
  '                os.path.join(PF6MPS_SOURCE_DIR, "n2_phase2_gm_mps.cpp"),\n'),
 # ---- 4d: provenance hashes the compiled file.
 ('4d',
  '                "n2_phase1_gm.cpp",\n'
  '                "n2_phase2_gm_mps.cpp",\n',
  '                "n2_phase1_gm_mps.cpp",\n'
  '                "n2_phase2_gm_mps.cpp",\n'),
]
for tag, old, new in edits:
    n = s.count(old)
    if n != 1:
        print(f"{tag}: ANCHOR COUNT {n} -- refusing, no edit written")
        sys.exit(4)
    s = s.replace(old, new)
    print(f"{tag}: ok")
open(p, "w", encoding="utf-8").write(s)
PY
[ $? -eq 0 ] || { echo "EDIT FAILED -- restoring"; cp -p "$BK/e004pf_k0pf_ab.py.pre" "$AB"; exit 4; }

python3 -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read()); print('ast.parse OK')" "$AB" \
  || { echo "PARSE FAILED -- restoring"; cp -p "$BK/e004pf_k0pf_ab.py.pre" "$AB"; exit 6; }
echo "post sha $(sha256sum "$AB" | cut -c1-16)"
echo "== diff =="
diff -u "$BK/e004pf_k0pf_ab.py.pre" "$AB"
echo "== vendored body carries the kGM contract token? =="
grep -n 'constexpr int kGM = N2GM_G' \
  "$HOME/Distributed-HipKittens/distributed-kernels/fused_moe/n2_phase1_gm_mps.cpp" | head -3
exit 0

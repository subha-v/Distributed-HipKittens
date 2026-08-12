#!/usr/bin/env bash
set -uo pipefail
SC=$HOME/overnight-scratch/e26
cat > "$SC/py/diag.py" <<'PYEOF'
import re, sys, collections
sys.path.insert(0,'/home/subvadla/overnight-scratch/e26/py')
exec(open('/home/subvadla/overnight-scratch/e26/py/mask.py').read().split('rows = []')[0])
for tag in sys.argv[1:]:
    insns = load(f'{tag}.isa')
    print(tag, 'insns', len(insns), 'mfma', sum(1 for x in insns if cls(x[1])=='MFMA'))
    seen = {}
    for lo,hi in backedges(insns):
        b = body(insns, lo, hi)
        n = sum(1 for x in b if cls(x[1])=='MFMA')
        if n >= 30:
            print(f'   loop 0x{lo:x}..0x{hi:x} mfma={n} insns={len(b)}')
PYEOF
docker exec subha_k1 bash -lc "cd /home/subvadla/overnight-scratch/e26/out2 && python3 ../py/diag.py D M15"
echo "--- compare with first run ---"
docker exec subha_k1 bash -lc "cd /home/subvadla/overnight-scratch/e26/out && python3 ../py/diag.py B0 B2" 2>&1 | head -20

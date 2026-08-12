#!/usr/bin/env bash
# Dump the race-check hazards and locate the MFMA loop for one arm.
#   10_hazards.sh <arm-name>
set -uo pipefail
ARM=${1:?usage: 10_hazards.sh <arm>}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_09_sched/arms/$ARM
S=$D/gemm_rs_mi300x.gfx942.s

echo "########## hazards ##########"
grep -B3 -A8 'HAZARD' "$D/lds_race.txt" | head -80

echo
echo "########## <256,256,32,false>: where are the MFMAs? ##########"
python3 - "$S" <<'PY'
import re, sys
lines = open(sys.argv[1], errors='replace').read().splitlines()
re_fstart = re.compile(r'^(_Z\S+):\s*(;.*)?$')
re_fend   = re.compile(r'^\.Lfunc_end(\d+):')
funcs, cur = [], None
for i, L in enumerate(lines, 1):
    m = re_fstart.match(L)
    if m: cur = (m.group(1), i); continue
    if re_fend.match(L) and cur: funcs.append([cur[0], cur[1], i]); cur = None
for nm, a, b in funcs:
    if 'ILi256ELi256ELi32ELb0E' not in nm: continue
    print("symbol lines %d..%d" % (a, b))
    # every block header line with its loop annotation, plus mfma/global_load counts
    blk = re.compile(r'^(\.LBB[\w.$]+):|^; (%bb\.[\w.$]+):')
    marks = []
    for i in range(a, b + 1):
        m = blk.match(lines[i - 1])
        if m: marks.append((i, m.group(1) or m.group(2), lines[i-1]))
    marks.append((b, 'END', ''))
    for j in range(len(marks) - 1):
        i0, nmv, raw = marks[j]; i1 = marks[j+1][0]
        seg = "\n".join(lines[i0-1:i1-1])
        nmf = seg.count('v_mfma'); ngl = seg.count('global_load_dwordx4')
        ndr = seg.count('ds_read_b64'); ndw = seg.count('ds_write_b64')
        if nmf or ngl:
            ann = ''
            for t in range(i0-1, min(i0+2, len(lines))):
                mm = re.search(r'(in Loop: Header=\S+ Depth=\d+|This (?:Inner )?Loop Header: Depth=\d+)', lines[t])
                if mm: ann = mm.group(1); break
            print("  %-12s %6d..%-6d mfma=%-3d gl=%-2d dsr=%-3d dsw=%-2d  %s"
                  % (nmv, i0, i1-1, nmf, ngl, ndr, ndw, ann))
PY

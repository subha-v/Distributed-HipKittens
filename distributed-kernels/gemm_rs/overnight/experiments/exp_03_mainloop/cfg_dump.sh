#!/usr/bin/env bash
# Print the CFG (labels + every branch target) and the raw text of the mainloop
# region of <256,256,32,false>, so block membership is read rather than inferred
# from the assembler's loop-depth comments.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
S=${1:-$ON/build/isa/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s}
LO=${2:-10190}
HI=${3:-10950}

python3 - "$S" "$LO" "$HI" <<'PY'
import re, sys
lines = open(sys.argv[1], errors='replace').read().splitlines()
lo, hi = int(sys.argv[2]), int(sys.argv[3])

print("############ CFG: labels and branches in %d..%d ############" % (lo, hi))
for i in range(lo, min(hi, len(lines)) + 1):
    L = lines[i - 1]
    st = L.split(';')[0].strip()
    if re.match(r'^\.LBB[\w.$]+:', st) or L.strip().startswith('; %bb.'):
        print("%7d| LABEL   %s" % (i, L.strip()[:110]))
    elif st.startswith('s_branch') or st.startswith('s_cbranch'):
        print("%7d|   br    %s" % (i, st))
print()
print("############ raw %d..%d ############" % (lo, hi))
for i in range(lo, min(hi, len(lines)) + 1):
    print("%7d| %s" % (i, lines[i - 1]))
PY

#!/usr/bin/env bash
# READ-ONLY: inspect the two tools the parent just pushed before wiring them in.
# I need the lease's exact CLI contract (does `acquire` block? what is arg 2?
# what does release do if we do not hold it?) and I need to confirm bench3's four
# repairs are actually present, because I am about to disclose them in result.md
# as behaviour-preserving and that claim has to rest on the file, not on a note.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight

echo "########## tools/gpu_lease.sh ##########"
if [ -f "$ON/tools/gpu_lease.sh" ]; then cat "$ON/tools/gpu_lease.sh"; else echo "MISSING"; fi

echo
echo "########## tools/run_rank1_bench3.sh (repaired?) ##########"
cat "$ON/tools/run_rank1_bench3.sh" 2>&1

echo
echo "########## the four repairs, checked mechanically ##########"
B=$ON/tools/run_rank1_bench3.sh
echo -n "1. AMDGCN_USE_BUFFER_OPS=0 present : "; grep -c 'AMDGCN_USE_BUFFER_OPS' "$B"
echo -n "   ...passed via docker exec -e   : "; grep -c '\-e AMDGCN_USE_BUFFER_OPS' "$B"
echo -n "2. PYTHONPATH -> tools/compat      : "; grep -c 'tools/compat' "$B"
echo -n "   ...stale \$NODE/compat gone      : "; grep -cE 'COMPAT=\$(NODE|ON)/compat$' "$B"
echo -n "3. patch_rank1.py at tools/ path   : "; grep -c 'tools/patch_rank1.py' "$B"
echo -n "4. container dhk-gemmrs            : "; grep -c 'dhk-gemmrs' "$B"
echo -n "   ...dhk-eval gone                : "; grep -c 'dhk-eval' "$B"
echo -n "   warm->test->bench order kept    : "; grep -nE '^go (benchmark|test)' "$B" | tr '\n' ' '
echo
echo -n "   node-clean preflight added      : "; grep -c 'showpids' "$B"
echo -n "   TRITON_CACHE_DIR cleared        : "; grep -c 'TRITON_CACHE_DIR' "$B"

echo
echo "########## frozen submission hash (unchanged?) ##########"
sha256sum /home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
echo "expected 7940fcb81df06c1d8b1e1a77051f23c934149a688441ef48b2751b3f336f0dc5"

echo
echo "########## lease state right now ##########"
ls -la "$ON/.lease" "$ON"/*.lease /tmp/*lease* 2>/dev/null | head -20
bash "$ON/tools/gpu_lease.sh" status 2>&1 | head -20

echo
echo "########## node state ##########"
rocm-smi --showpids 2>&1 | sed -n '3,12p'
echo "kfd fds: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"
echo "########## DONE ##########"

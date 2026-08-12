#!/usr/bin/env bash
# exp_26 probe: donor provenance + load_bfrag census + build env survey.
set -u
SOL="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip"
echo "===DONOR==="
sha256sum "$SOL/n2_phase1_gm.cpp" "$SOL/n2_phase2_gm.cpp"
echo "===LOAD_BFRAG==="
grep -n -A 24 'load_bfrag' "$SOL/n2_device_common.cuh" | head -80
echo "===STAGE_AGPR==="
grep -n -A 14 'stage_agpr' "$SOL/n2_device_common.cuh" | head -40
echo "===LDS_CTA_BARRIER==="
grep -n -A 10 'lds_cta_barrier' "$SOL/n2_device_common.cuh" | head -30
echo "===SCHED_HINTS_P2_DONOR==="
grep -n 'sched_group_barrier' "$SOL/n2_phase2_gm.cpp"
echo "===DOCKER==="
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
echo "===GPUJOBS==="
pgrep -af 'torchrun|mpirun' || echo "(none)"
echo "===MORI_JIT==="
docker exec subha_k1 bash -lc 'ls -d /usr/local/lib/python3.12/dist-packages/mori/_jit-sources 2>/dev/null || echo MISSING'
echo "===HOME_LAYOUT==="
ls -d "$HOME/Distributed-HipKittens" "$HOME/amd-master" 2>/dev/null

#!/usr/bin/env bash
# The runner acquired the lease at 05:06 and then LOST it: gpu_lease.sh's own
# drain wait aborted after 300s on KFD pid 3001610, which `status` now reports as
# STALE (exiting/zombie, cannot dispatch). So the tool has been updated since my
# probe. Before relaunching I need to know whether the stale-pid distinction was
# applied to the ACQUIRE DRAIN LOOP or only to `status` -- if only to status, a
# relaunch aborts again in exactly the same place after another 300s.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight

echo "===== does the drain loop know about stale pids? ====="
sed -n '/kfd_pids\|kfd_live\|stale/,+3p' "$ON/tools/gpu_lease.sh" | head -60
echo
echo "----- the acquire drain loop verbatim -----"
sed -n '/mkdir "\$LOCK"/,/^        done/p' "$ON/tools/gpu_lease.sh"

echo
echo "===== is 3001610 actually dead? ====="
echo -n "state: "; ps -o pid=,stat=,etime=,cmd= -p 3001610 2>&1 | head -3 || echo "gone from ps"
echo -n "/proc exists: "; [ -d /proc/3001610 ] && echo yes || echo no
echo -n "kfd fd held : "; ls -l /proc/3001610/fd 2>/dev/null | grep -c kfd || echo "n/a"
echo -n "parent      : "; awk '{print "ppid="$4" state="$3}' /proc/3001610/stat 2>/dev/null || echo "n/a"
echo "rocm-smi view:"; rocm-smi --showpids 2>&1 | sed -n '3,12p'

echo
echo "===== can a GPU actually be used right now? ====="
timeout 120 docker exec dhk-gemmrs python3 -c "
import torch
print('devices', torch.cuda.device_count())
x = torch.ones(1024, 1024, device='cuda:0')
print('alloc+matmul ok:', bool((x@x).sum().item() == 1024**3))
" 2>&1 | tail -5

echo
echo "===== lease + clocks ====="
bash "$ON/tools/gpu_lease.sh" status
rocm-smi --showclocks 2>/dev/null | grep -i sclk | head -3

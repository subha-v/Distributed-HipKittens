#!/usr/bin/env bash
# Is the node actually wedged, or merely holding a stale pid?
#
# The distinction decides the night: if a GPU queue is hung the driver is waiting
# on a reset and nothing will run; if teardown is merely slow, the GPUs are
# usable and only our own drain gate (which requires 0 KFD pids) is blocking the
# queue. Diagnose before doing anything destructive.
set -uo pipefail

echo "===== 1. kernel ring buffer: GPU faults / resets / queue evictions ====="
(dmesg -T 2>/dev/null || sudo dmesg -T 2>/dev/null || cat /var/log/kern.log 2>/dev/null | tail -400) \
  | grep -iE 'amdgpu|kfd|ring .* timeout|GPU reset|VM_L2|page fault|hang|evict|IOMMU' \
  | tail -40
echo "  (empty above = no fault/reset logged)"

echo
echo "===== 2. per-GPU health ====="
rocm-smi --showtemp --showpower --showuse 2>/dev/null | grep -iE 'GPU\[|Temperature \(Sensor edge\)|Average Graphics Package|GPU use' | head -30

echo
echo "===== 3. can a trivial HIP job actually run on all 8 GPUs? ====="
docker exec dhk-gemmrs timeout 180 python3 -c "
import torch, sys
try:
    n = torch.cuda.device_count()
    print('  visible devices:', n)
    ok = []
    for i in range(n):
        torch.cuda.set_device(i)
        a = torch.randn(2048, 2048, device=f'cuda:{i}', dtype=torch.bfloat16)
        c = (a @ a).float().sum().item()
        free, total = torch.cuda.mem_get_info(i)
        ok.append(i)
        print(f'  gpu {i}: matmul OK  sum={c:.3e}  free={free/2**30:.1f}/{total/2**30:.1f} GiB')
    print('  ALL_OK' if len(ok) == n else '  PARTIAL')
except Exception as e:
    print('  FAILED:', type(e).__name__, e); sys.exit(1)
" 2>&1 | tail -20

echo
echo "===== 4. is the stuck pid visible INSIDE the container? ====="
docker exec dhk-gemmrs bash -lc 'ps -eo pid,stat,etime,cmd --no-headers | grep -E "m9_stale|python3 -u" | grep -v grep' 2>&1 | head -5
echo "  (a pid stuck in exit_mm is often already gone from the container's namespace)"

echo
echo "===== 5. lease + queue ====="
bash /home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/tools/gpu_lease.sh status
echo done

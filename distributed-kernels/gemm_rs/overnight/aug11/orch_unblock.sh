#!/usr/bin/env bash
# Swap in the stale-pid-aware lease and free the queue that is stuck behind a
# corpse.
#
# Sequence matters. exp_21's `acquire` is currently executing the previous
# version of the tool from its own open inode, so it will keep waiting on the
# dead pid no matter what we install. We therefore: install the fix by atomic
# rename (never truncate a script a process is executing), then terminate the
# WAITING BASH WRAPPERS -- these are plain shell, not GPU processes, so SIGTERM
# is safe and the no-SIGKILL-on-GPU-processes rule does not apply -- then release
# the lease they left behind.
#
# Nothing here signals the wedged GPU process: a task in exit_mm cannot be
# signalled at all, and attempting it is what escalates a stale entry into a
# genuinely wedged node.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
T=$ON/tools

echo "===== 1. install the stale-pid-aware lease ====="
[ -s "$T/gpu_lease.new.sh" ] || { echo "staged file missing"; exit 1; }
bash -n "$T/gpu_lease.new.sh" || { echo "staged file does not parse; refusing"; exit 1; }
chmod +x "$T/gpu_lease.new.sh"
mv -f "$T/gpu_lease.new.sh" "$T/gpu_lease.sh"
echo "installed."

echo
echo "===== 2. what does it now see? ====="
bash "$T/gpu_lease.sh" status

echo
echo "===== 3. terminate the stuck waiters (plain bash, not GPU processes) ====="
for pat in 'gpu_lease.sh acquire exp_21' 'gpu_lease.sh acquire exp_24' \
           'exp_21_saturation/campaign.sh' 'exp_24_ladders/full_runner.sh'; do
  for p in $(pgrep -f "$pat" 2>/dev/null); do
    echo "  SIGTERM $p ($pat)"
    kill -TERM "$p" 2>/dev/null || true
  done
done
sleep 5
echo "  survivors:"
pgrep -af 'gpu_lease.sh acquire|campaign.sh|full_runner.sh' 2>/dev/null | head -10 || echo "  (none)"

echo
echo "===== 4. release any lease they left behind ====="
for who in exp_21 exp_24 exp_23 exp_22; do
  bash "$T/gpu_lease.sh" release "$who" 2>/dev/null | grep -v 'no lease held' || true
done
bash "$T/gpu_lease.sh" status

echo
echo "===== 5. prove the GPUs are usable right now ====="
docker exec dhk-gemmrs timeout 180 python3 -c "
import torch
n = torch.cuda.device_count()
for i in range(n):
    torch.cuda.set_device(i)
    a = torch.randn(4096, 4096, device=f'cuda:{i}', dtype=torch.bfloat16)
    (a @ a).float().sum().item()
print(f'  {n}/8 GPUs ran a 4096 bf16 matmul: OK')
" 2>&1 | tail -3
echo done

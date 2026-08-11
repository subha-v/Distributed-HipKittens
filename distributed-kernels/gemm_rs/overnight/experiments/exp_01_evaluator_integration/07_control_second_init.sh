#!/usr/bin/env bash
# exp_01 step 3: is the second init_process_group hang OURS or ENVIRONMENTAL?
#
# The single-shape evaluator run proved our kernel completes 101 consecutive
# calls on all 8 ranks and then every rank blocks in init_process_group. Our
# module pins the previous ProcessGroup in the module-global _LAST_PG (a
# deliberate strong reference), which survives destroy_process_group and could
# stop the old NCCL communicator and its TCPStore from being finalized.
#
# Control: the same two-case mp_smoke driver against a plain matmul +
# reduce_scatter submission that has no globals and no IPC. If the control also
# hangs at case 1, the strong reference is exonerated and the fault is in
# torch/NCCL/the container.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_01_evaluator_integration
OUT=$EXP/logs
mkdir -p $OUT

echo "===== is a ProcessGroup weakref-able? does it expose group_name? ====="
docker exec dhk-gemmrs bash -c "cd /tmp && MASTER_ADDR=127.0.0.1 MASTER_PORT=12801 \
  timeout 120 python3 -u -c \"
import weakref, torch, torch.distributed as dist
torch.cuda.set_device(0)
dist.init_process_group('nccl', init_method='env://', rank=0, world_size=1,
                        device_id=torch.device('cuda:0'))
g = dist.distributed_c10d._get_default_group()
print('type', type(g).__name__)
try:
    r = weakref.ref(g); print('weakref OK ->', r() is g)
except TypeError as e:
    print('weakref UNSUPPORTED:', e)
print('group_name', getattr(g, 'group_name', '<none>'))
dist.destroy_process_group()
print('probe done')
\"" 2>&1 | tail -12

echo
echo "===== CONTROL: reference submission, mp_smoke two cases ====="
cp $EXP/ref_submission.py $ON/harness/submission.py
docker exec -w $ON/harness dhk-gemmrs bash -c \
  'HK_DEBUG=1 timeout 240 python3 -u mp_smoke.py' > $OUT/control_mp.log 2>&1
echo "control rc=$?"

echo
echo "--- control: per-case progress ---"
grep -E 'case 0|case 1|correctness|DONE|TIMED OUT|exit codes' $OUT/control_mp.log | head -40

echo
echo "--- control: where blocked / errors ---"
grep -E 'init_process_group|DistBackendError|ncclRemote|Connection refused|Timeout \(0:' \
  $OUT/control_mp.log | head -20

echo
echo "--- control: case-1 reached by how many ranks? ---"
printf 'case 0 entered: %s\ncase 1 entered: %s\ncase 0 correct: %s\ncase 1 correct: %s\nDONE: %s\n' \
  "$(grep -c 'case 0:' $OUT/control_mp.log)" \
  "$(grep -c 'case 1:' $OUT/control_mp.log)" \
  "$(grep -c 'correctness allclose=True' $OUT/control_mp.log)" \
  "$(grep -c 'DONE' $OUT/control_mp.log)" \
  "$(grep -c 'DONE' $OUT/control_mp.log)"

echo
echo "--- control tail ---"
tail -20 $OUT/control_mp.log

echo
echo "===== reap ====="
pkill -TERM -f mp_smoke 2>/dev/null; pkill -TERM -f multiprocessing.spawn 2>/dev/null; sleep 5
rocm-smi --showpids 2>&1 | grep -c 'No KFD PIDs' || true

echo
echo "===== STEP3 DONE ====="

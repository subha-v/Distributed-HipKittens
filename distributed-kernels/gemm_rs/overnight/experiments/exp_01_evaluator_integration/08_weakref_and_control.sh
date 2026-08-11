#!/usr/bin/env bash
# exp_01 step 4: two arms for the second-init hang, both driven by mp_smoke's
# two-case script (case 0 = 512x4096x12288, case 1 = 2048x2880x2880, one
# process group each, so case 1 is the second init_process_group).
#
#   CONTROL   : plain matmul + reduce_scatter, no globals, no IPC.
#               If this hangs at case 1 too, our retained state is exonerated
#               and the fault is torch/NCCL/container.
#   CANDIDATE : hk_submission with _LAST_PG demoted from a strong reference to
#               a weakref. A strong reference outlives destroy_process_group and
#               pins the old NCCL communicator + TCPStore.
#
# Both arms are HK_DEBUG=1 diagnostics. No timing is reported from here.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_01_evaluator_integration
OUT=$EXP/logs
mkdir -p $OUT

summarize() {   # $1 = log, $2 = label
  printf '%-10s case0_entered=%s case1_entered=%s allclose_true=%s DONE=%s\n' "$2" \
    "$(grep -c 'case 0:' $1)" "$(grep -c 'case 1:' $1)" \
    "$(grep -c 'correctness allclose=True' $1)" "$(grep -c '] DONE' $1)"
  grep -E 'exit codes|TIMED OUT' $1 | head -3
  grep -E 'init_process_group|ncclRemote|Connection refused' $1 | head -3
}

################################ CONTROL #################################
cat > $EXP/ref_submission.py <<'PYEOF'
"""CONTROL: plain matmul + reduce_scatter. No module globals, no HIP IPC."""
import torch
import torch.distributed as dist

WORLD = 8


def custom_kernel(data):
    x, w, bias = data
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    partial = torch.matmul(x, w.T)
    if bias is not None:
        partial = partial + bias
    rows = x.shape[0] // WORLD
    out = torch.empty((rows, w.shape[0]), dtype=partial.dtype,
                      device=partial.device)
    dist.reduce_scatter_tensor(out, partial.contiguous())
    return out
PYEOF

echo "===== ARM 1 / CONTROL: reference submission, two cases ====="
cp $EXP/ref_submission.py $ON/harness/submission.py
docker exec -w $ON/harness dhk-gemmrs bash -c \
  'HK_DEBUG=1 timeout 200 python3 -u mp_smoke.py' > $OUT/arm_control.log 2>&1
echo "control rc=$?"
summarize $OUT/arm_control.log CONTROL
pkill -TERM -f mp_smoke 2>/dev/null; pkill -TERM -f multiprocessing.spawn 2>/dev/null; sleep 6

############################### CANDIDATE ################################
echo
echo "===== patch hk_submission.py: _LAST_PG strong ref -> weakref ====="
python3 - "$ON/harness/hk_submission.py" <<'PYEOF'
import hashlib, sys
path = sys.argv[1]
src = open(path).read()
before = hashlib.sha256(src.encode()).hexdigest()

edits = [
    ("import sys\nimport time\n",
     "import sys\nimport time\nimport weakref\n"),
    ("differs from the last one. _LAST_PG holds a strong reference so a recycled\n"
     "# object address cannot make a new group look like the old one.",
     "differs from the last one. _LAST_PG is a WEAKREF, not a strong reference:\n"
     "# a strong reference here outlives destroy_process_group and pins the old\n"
     "# NCCL communicator and its TCPStore, and the next init_process_group on the\n"
     "# same MASTER_PORT then never completes -- measured under eval.py, where all\n"
     "# eight ranks finished 101 calls and then blocked in\n"
     "# _new_process_group_helper. A weakref cannot make a recycled address look\n"
     "# like the old group either, because the test below compares the referent."),
    ("    settled = (state is not None and group is not None\n"
     "               and group is _LAST_PG and key == _LAST_KEY)",
     "    settled = (state is not None and group is not None\n"
     "               and _LAST_PG is not None and _LAST_PG() is group\n"
     "               and key == _LAST_KEY)"),
    ("        _LAST_PG = group\n",
     "        _LAST_PG = weakref.ref(group) if group is not None else None\n"),
]
for old, new in edits:
    if src.count(old) != 1:
        print(f"PATCH ABORT: {src.count(old)} matches for {old[:60]!r}")
        sys.exit(3)
    src = src.replace(old, new)
open(path, "w").write(src)
print("patched", path)
print("sha256 before", before)
print("sha256 after ", hashlib.sha256(src.encode()).hexdigest())
PYEOF
prc=$?
echo "patch rc=$prc"
[ $prc -ne 0 ] && { echo "PATCH FAILED, stopping"; exit $prc; }

echo
echo "--- the patched decision points ---"
grep -n 'weakref' $ON/harness/hk_submission.py

echo
echo "===== ARM 2 / CANDIDATE: weakref hk_submission, two cases ====="
cp $ON/harness/hk_submission.py $ON/harness/submission.py
docker exec -w $ON/harness dhk-gemmrs bash -c \
  'HK_DEBUG=1 timeout 260 python3 -u mp_smoke.py' > $OUT/arm_weakref.log 2>&1
echo "candidate rc=$?"
summarize $OUT/arm_weakref.log CANDIDATE

echo
echo "--- candidate: cache vote lines (symmetry check) ---"
grep 'cache vote' $OUT/arm_weakref.log | head -20

echo
echo "--- candidate: max|diff| per rank ---"
grep 'correctness' $OUT/arm_weakref.log | head -20

echo
echo "--- candidate tail ---"
tail -15 $OUT/arm_weakref.log

echo
echo "===== reap + node state ====="
pkill -TERM -f mp_smoke 2>/dev/null; pkill -TERM -f multiprocessing.spawn 2>/dev/null; sleep 6
pgrep -a -f 'mp_smoke|multiprocessing.spawn' | head -5 || echo "no stragglers"
rocm-smi --showpids 2>&1 | grep 'No KFD PIDs' || echo "WARNING: KFD pids remain"

echo
echo "===== STEP4 DONE ====="

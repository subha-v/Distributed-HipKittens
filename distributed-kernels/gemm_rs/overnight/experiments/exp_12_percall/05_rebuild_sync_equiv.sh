#!/usr/bin/env bash
# exp_12: rebuild, resync harness/submission.py from hk_submission.py, and prove
# the prebound launch path is bit-identical to the duck-typed one.
#
# submission.py is a COPY of hk_submission.py that lives only on the node (the
# evaluator and every mp_* harness import `submission`). Editing hk_submission
# without recopying measures the old code and looks like a null result.
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_12_percall
PORT=${1:-12631}

sed -i 's/\r$//' "$EXP"/*.py "$EXP"/*.sh 2>/dev/null

echo "===== node clean check ====="
if ! rocm-smi --showpids 2>&1 | grep -q 'No KFD PIDs currently running'; then
  echo "REFUSING: GPU processes present"; rocm-smi --showpids 2>&1 | head -20
  exit 3
fi

echo "===== was submission.py still a faithful copy before this edit? ====="
if [ -f "$ON/harness/submission.py" ]; then
  echo "  submission.py    $(sha256sum "$ON/harness/submission.py" | cut -c1-16)"
  echo "  hk_submission.py $(sha256sum "$ON/harness/hk_submission.py" | cut -c1-16)"
  diff <(sed '/fast path/,$d' "$ON/harness/submission.py") /dev/null >/dev/null 2>&1
  echo "  (differing now is expected: hk_submission carries the exp_12 edit)"
else
  echo "  submission.py ABSENT"
fi

echo "===== M1 build ====="
docker exec dhk-gemmrs bash "$ON/harness/build.sh" 2>&1 | grep -E 'OK|FAIL|error|entry points|ALL MODULES'
grep_rc=$?

echo "===== resync submission.py ====="
cp "$ON/harness/hk_submission.py" "$ON/harness/submission.py"
cmp -s "$ON/harness/hk_submission.py" "$ON/harness/submission.py" \
  && echo "  submission.py == hk_submission.py" || { echo "  COPY FAILED"; exit 4; }

echo "===== equivalence: prebound vs duck-typed, bit-exact, 6 shapes ====="
docker exec -w "$EXP" dhk-gemmrs python3 -m py_compile 04_equiv.py || exit 4
setsid timeout --signal=TERM 900 \
  docker exec -w "$EXP" dhk-gemmrs python3 -u 04_equiv.py "$PORT" \
  2>&1 | tee "$EXP/logs/equiv.txt"
rc=${PIPESTATUS[0]}

sleep 3
rocm-smi --showpids 2>&1 | grep -E 'No KFD PIDs|PID' | head -5
echo "exit=$rc"
echo "DONE-equiv"
exit $rc

#!/usr/bin/env bash
# exp_10 -- interleaved same-run paired comparison, OUR kernel vs frozen rank-1,
# graded protocol, one fresh 8-process pool per shape.
#   $1 shapes (comma list, default all)  $2 iters  $3 reps  $4 base port
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ARM=$ON/compbench/rank1
IRISDST=/usr/local/lib/python3.10/dist-packages
SHAPES=${1:-0,1,2,3,4,5}
ITERS=${2:-15}
REPS=${3:-2}
PORT0=${4:-12500}
VS_FORCE_BIAS=${5:-0}
export VS_FORCE_BIAS
OUT=$ON/experiments/exp_10_rank1/vs_logs
mkdir -p "$OUT"

echo "kfd_pids before launch: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"

echo "===== integrity: does OUR submission collide with rank-1's torch op ns? ====="
grep -nE 'my_ops|TORCH_LIBRARY|load_inline|def custom_kernel' "$ON/harness/submission.py" 2>/dev/null | head
echo "(rank-1 registers 'my_ops'; ours must not)"

echo
echo "===== integrity: rank-1 shape configs (silent torch fallback check) ====="
docker exec -w "$ARM" \
  -e PATH="$ON/tools/compat/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
  -e PYTHONPATH="$ON/tools/compat:$IRISDST" \
  dhk-gemmrs bash -lc 'python3 -u -c "
import submission as s
SH=[(64,7168,18432),(512,4096,12288),(2048,2880,2880),(4096,4096,4096),(8192,4096,14336),(8192,8192,29568)]
conf=None
for nm in (\"__conf\",\"_submission__conf\"):
    conf=getattr(s,nm,conf)
print(\"online_config keys:\", sorted(s.online_config.keys()))
print(\"group keys:\", sorted(s.online_config_group.keys()))
for (m,n,k) in SH:
    lk=k//8; key=(m,n,lk)
    print(f\"  m={m:<5} n={n:<5} lk={lk:<5} in___conf={key in conf if conf else None} in_online_config={key in s.online_config} group={(m//8)*n in s.online_config_group}\")
"' 2>&1 | tail -20

cp "$ON/experiments/exp_10_rank1/mp_vs_rank1.py" "$ARM/mp_vs_rank1.py"

IFS=',' read -ra LIST <<< "$SHAPES"
for s in "${LIST[@]}"; do
  PORT=$((PORT0 + s))
  echo
  echo "################ shape $s ################"
  rm -f "$ARM"/ipc_handles_rank*.bin
  docker exec -w "$ARM" \
    -e PATH="$ON/tools/compat/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
    -e PYTHONPATH="$ON/tools/compat:$IRISDST" \
    -e PYTHONUNBUFFERED=1 \
    -e HK_DEBUG=0 \
    -e TRITON_CACHE_DIR="$ARM/.triton" \
    -e HSA_ENABLE_COREDUMP=0 \
    -e AMDGCN_USE_BUFFER_OPS=0 \
    -e VS_FORCE_BIAS="${VS_FORCE_BIAS:-0}" \
    -e VS_OUT="$OUT/vs_s${s}" \
    dhk-gemmrs bash -lc "timeout --signal=TERM 1200 setsid python3 -u mp_vs_rank1.py $s $ITERS $REPS $PORT" \
    2>&1 | grep -vE '^\[1/|^\[2/|^\[3/|hipcc|^ *[0-9]+ \||warning:|^ *\^|preprocessed|replaced kernel|unsupported CUDA' | tail -25
  echo "rc=$?"
done

echo
echo "===== aggregate ====="
python3 "$ON/experiments/exp_10_rank1/vs_report.py" "$OUT"
echo "===== DONE ====="

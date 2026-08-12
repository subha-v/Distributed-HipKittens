#!/usr/bin/env bash
# exp_10 probe 15 -- why did shape 4 (4096x4096x4096) die with SIGTERM, and can
# it be re-run alone? Everything else in the sweep passed.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ARM=$ON/compbench/rank1
OUT=$ON/experiments/exp_10_rank1/vs_logs
IRISDST=/usr/local/lib/python3.10/dist-packages

echo "===== A. what did shape 3's ranks say? ====="
for r in 0 1 2 3 4 5 6 7; do
  f="$OUT/vs_s3.rank$r.stderr"
  [ -f "$f" ] || { echo "rank $r: no stderr file"; continue; }
  echo "--- rank $r ($(wc -c < "$f") bytes) ---"
  grep -nE 'waiting for ipc handle|UNI Error|Memory access fault|out of memory|Error|Traceback|assert' "$f" | head -6
  tail -4 "$f"
done

echo
echo "===== B. json results present? ====="
ls -la "$OUT"/ | head -30

echo
echo "===== C. host memory + GPU memory right now ====="
free -g | head -3
rocm-smi --showmeminfo vram 2>/dev/null | grep -E 'GPU|Total|Used' | head -20

echo
echo "===== D. re-run shape 3 alone, bias forced, fresh state ====="
echo "kfd_pids: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"
rm -f "$ARM"/ipc_handles_rank*.bin
cp "$ON/experiments/exp_10_rank1/mp_vs_rank1.py" "$ARM/mp_vs_rank1.py"

docker exec -w "$ARM" \
  -e PATH="$ON/tools/compat/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
  -e PYTHONPATH="$ON/tools/compat:$IRISDST" \
  -e PYTHONUNBUFFERED=1 -e HK_DEBUG=0 \
  -e TRITON_CACHE_DIR="$ARM/.triton" \
  -e HSA_ENABLE_COREDUMP=0 \
  -e AMDGCN_USE_BUFFER_OPS=0 \
  -e VS_FORCE_BIAS=1 \
  -e VS_OUT="$OUT/vs_s3" \
  dhk-gemmrs bash -lc "timeout --signal=TERM 1500 setsid python3 -u mp_vs_rank1.py 3 15 2 12777" \
  2>&1 | grep -vE '^\[1/|^\[2/|^\[3/|hipcc|warning:|^ *\^|preprocessed|replaced kernel|unsupported CUDA|clear cost|start clear' | tail -20
echo "rc=$?"

echo
echo "===== E. shape 3 stderr after the retry ====="
for r in 0 1; do
  f="$OUT/vs_s3.rank$r.stderr"
  echo "--- rank $r ---"; tail -8 "$f" 2>/dev/null
done
echo "===== DONE p15 ====="

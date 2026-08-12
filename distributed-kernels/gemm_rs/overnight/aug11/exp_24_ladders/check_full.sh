#!/usr/bin/env bash
# Poll the detached full ladder. Read-only; safe to call at any time.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders

echo "=== $(date -Is) ==="
echo -n "runner: "; pgrep -af 'full_runner.sh' | grep -v pgrep || echo "NOT RUNNING"
echo -n "ladder: "; pgrep -af 'ladder_mp.py|eval.py' | grep -v pgrep | head -3 || echo "no python arm active"
# tools/gpu_lease.sh currently ships CRLF, which bash rejects; prefer the runner's
# normalized snapshot, and fall back to a CR-stripped read of the live tool.
if [ -f "$D/toolsnap/gpu_lease.sh" ]; then
  bash "$D/toolsnap/gpu_lease.sh" status
else
  bash <(tr -d '\r' < "$ON/tools/gpu_lease.sh") status
fi
echo "kfd fds: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"

echo
echo "=== progress: instrument A shapes completed ==="
ls "$D/raw/ladder"/lad_s*.rank0.json 2>/dev/null | sed 's/.*lad_s/  shape idx /;s/\.rank0\.json//' || echo "  none yet"
echo "  (8 rank files per shape; 6 shapes expected)"
echo -n "  rank-file count: "; ls "$D/raw/ladder"/lad_s*.rank*.json 2>/dev/null | wc -l

echo
echo "=== progress: instrument B captures ==="
find "$D/raw/eval" -name '*.popcorn.txt' 2>/dev/null | sed "s|$D/raw/eval/|  |" | sort || echo "  none yet"

echo
echo "=== log tail ==="
tail -${1:-45} "$D/logs/full_run.log" 2>/dev/null || echo "(no log yet)"

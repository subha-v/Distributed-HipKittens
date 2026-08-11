#!/usr/bin/env bash
# Node state probe: clocks, stale GPU jobs, containers, rank-1 progress marker.
# Host-side only; touches no GPU compute.
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight

echo "===== date ====="
date -u +"%Y-%m-%dT%H:%M:%SZ"

echo "===== containers ====="
docker ps --format '{{.Names}}\t{{.Status}}' 2>/dev/null || echo "docker ps failed"

echo "===== our stale GPU processes (evaluator / mp_smoke / python harness) ====="
ps -eo pid,etime,user,cmd 2>/dev/null | grep -E 'mp_smoke|eval\.py|m[0-9]_|exp_|hk_submission|torchrun' | grep -v grep || echo "  none"

echo "===== GPU busy / clocks ====="
rocm-smi --showuse --showclocks --showperfdeterminism 2>/dev/null | grep -Ei 'GPU\[|use|sclk|determin' | head -60 || echo "rocm-smi unavailable"

echo "===== rank-1 progress marker: heap_bases_*.pkl ====="
for d in $ON/compbench $ON/../compbench /home/subvadla/dhk/compbench; do
  [ -d "$d" ] && echo "-- $d --" && find "$d" -name 'heap_bases_*.pkl' 2>/dev/null | head -20
done
echo "  (count below)"
find /home/subvadla -name 'heap_bases_*.pkl' 2>/dev/null | wc -l

echo "===== compbench tree ====="
ls -la $ON/compbench 2>/dev/null | head -30 || echo "  no compbench dir under overnight/"

echo "===== disk ====="
df -h /home/subvadla | tail -1

echo "===== DONE ====="

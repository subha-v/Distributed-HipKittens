#!/usr/bin/env bash
set -u
echo "=== pre-timing cleanliness $(date -u +%FT%TZ) ==="
rocm-smi --showpids
echo
echo "=== perf level ==="
rocm-smi --showperflevel
echo
echo "=== our stray processes ==="
ps -eo pid,user,etime,cmd | grep -E "mp_smoke|eval\.py|torchrun|m7_bench|m5_soak|m3_correctness|m4_controls" | grep -v grep
echo "(end stray list)"
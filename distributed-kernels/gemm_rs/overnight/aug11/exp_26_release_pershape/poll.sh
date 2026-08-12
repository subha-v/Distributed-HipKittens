#!/usr/bin/env bash
# Progress read for whatever exp_26 has running. Read-only; touches no GPU.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
WHICH=${1:-ladder}

echo "===== $(date -u +%FT%TZ) ====="
echo "-- alive --"
pgrep -af 'gate_ladder.sh exp_26|ab_pershape|m9_stale_slot|run_ab.sh' | head -5 \
  || echo "  (nothing of ours)"
echo "-- KFD pids --"
rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l
echo "-- tail of $WHICH --"
tail -40 "$D/logs/$WHICH.log" 2>/dev/null || echo "  (no log yet)"
echo "-- gate markers so far --"
grep -E '^##########|GATE LADDER|FAILED AT|PASSED|shapes PASSED' \
  "$D/logs/$WHICH.log" 2>/dev/null | tail -20

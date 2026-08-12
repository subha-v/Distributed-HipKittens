#!/usr/bin/env bash
set -uo pipefail
LOG="$HOME/exp01_descdump.log"
OUT="$HOME/k0-mok-mps-descdump"

echo "=== running? ==="
pgrep -af 'torchrun|run_campaign' | head -5
echo "=== log tail ==="
tail -n 30 "$LOG" 2>/dev/null
echo "=== desc dumps found ==="
find "$OUT" -name 'mps_desc_rank*.txt' 2>/dev/null | head -20
echo "=== progress logs found ==="
find "$OUT" -name 'progress_rank*.log' 2>/dev/null | head -20

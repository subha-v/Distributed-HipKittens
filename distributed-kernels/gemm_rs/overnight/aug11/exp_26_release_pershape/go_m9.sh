#!/usr/bin/env bash
# Gate M9 (mandatory for anything touching publication order), detached.
#
# m9 compares the production module bitwise against the frozen pre-E3 golden
# under a NaN-poisoned heap, and then runs CTRL_PUBLISH_EARLY, which MUST still
# fail. A gate whose control stops failing is blind, and a pass from a blind
# gate means nothing -- so this wrapper checks the golden module exists first
# rather than letting m9 fall back to anything.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
GOLD=$ON/harness/build/gemm_rs_mi300x_e3base.so
LOG=$D/logs/m9.log
mkdir -p "$D/logs"

[ -f "$GOLD" ] || { echo "REFUSED: golden module missing: $GOLD"; exit 1; }
echo "golden: $(stat -c'%s bytes  %y' "$GOLD")"

n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids before launch: $n"
[ "$n" = "0" ] || { echo "REFUSED: node dirty"; exit 1; }

: > "$LOG"
setsid nohup docker exec -w $ON/harness dhk-gemmrs timeout 7200 \
  python3 -u m9_stale_slot.py >> "$LOG" 2>&1 &
echo "launched pid $! ; log $LOG"
sleep 8
tail -12 "$LOG"

#!/usr/bin/env bash
# Sync the exp_22 sources to the node and normalize line endings.
# Windows scp writes CRLF, and a CR inside a shell script or an inline-asm
# string fails in ways that look like a compiler bug (it cost an hour once).
#
#   bash tools/push.sh              # sync only
#   bash tools/push.sh gate         # sync, then run the full CPU gate
set -euo pipefail
HOST=subvadla@gbt350-odcdh2-c05-1.png-odc.dcgpu
HERE="$(cd "$(dirname "$0")/.." && pwd)"

ssh "$HOST" 'mkdir -p /home/subvadla/e22/src /home/subvadla/e22/out'
scp -q "$HERE/e22_saturation.hip" "$HERE/build_in.sh" "$HERE/remark.py" \
       "$HERE/isa_census.py" "$HERE/summarize.py" \
       "$HERE/selftest_summarize.py" "$HERE/run_saturation.sh" \
       "$HERE/tools/fixnl.py" "$HERE/tools/e22_gate.sh" \
       "$HOST":/home/subvadla/e22/src/
ssh "$HOST" 'python3 /home/subvadla/e22/src/fixnl.py /home/subvadla/e22/src'

if [ "${1:-}" = "gate" ]; then
  ssh "$HOST" 'bash /home/subvadla/e22/src/e22_gate.sh'
fi

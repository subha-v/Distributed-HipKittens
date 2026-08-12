#!/usr/bin/env bash
# Generic poll of the staged overnight job output.
set -uo pipefail
F="${1:-$HOME/ovn/discriminators.out}"
echo "=== gpu jobs ==="
pgrep -af 'torchrun|mpirun' | head -3 || echo "(none)"
echo "=== driver alive? ==="
pgrep -af 'ovn/.*\.sh' | head -3 || echo "(driver finished)"
echo "=== output ==="
cat "$F" 2>/dev/null
exit 0

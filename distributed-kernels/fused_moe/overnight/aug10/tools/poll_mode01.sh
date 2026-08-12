#!/usr/bin/env bash
set -uo pipefail
echo "=== gpu jobs ==="
pgrep -af 'torchrun|mpirun' | head -3 || echo "(none)"
echo "=== mode01 driver ==="
pgrep -af 'ovn/mode01' | head -2 || echo "(driver finished)"
echo
echo "=== mode01 output ==="
cat "$HOME/ovn/mode01.out" 2>/dev/null
exit 0

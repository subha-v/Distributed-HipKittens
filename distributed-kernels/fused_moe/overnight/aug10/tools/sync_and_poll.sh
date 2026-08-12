#!/usr/bin/env bash
set -uo pipefail
echo "=== sync node repo ==="
cd "$HOME/Distributed-HipKittens" && git pull --ff-only 2>&1 | tail -2 && git rev-parse HEAD

echo
echo "=== gpu jobs ==="
pgrep -af 'torchrun|mpirun' | head -3 || echo "(none)"
echo "=== discriminator driver ==="
pgrep -af 'ovn/discriminators' | head -2 || echo "(driver finished)"
echo
echo "=== discriminator output ==="
cat "$HOME/ovn/discriminators.out" 2>/dev/null
exit 0

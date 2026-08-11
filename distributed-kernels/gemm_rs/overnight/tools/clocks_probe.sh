#!/usr/bin/env bash
# Where does rocm-smi live, and are clocks pinned? Host first, then container.
set -u
echo "===== which rocm-smi (host) ====="
command -v rocm-smi || echo "  not on host PATH"
ls -la /opt/rocm/bin/rocm-smi 2>/dev/null || echo "  no /opt/rocm/bin/rocm-smi on host"

echo "===== host attempt ====="
rocm-smi --showclocks 2>&1 | head -20

echo "===== container attempt (dhk-gemmrs) ====="
docker exec dhk-gemmrs rocm-smi --showclocks 2>&1 | grep -Ei 'sclk|determin' | head -20

echo "===== container perfdeterminism state ====="
docker exec dhk-gemmrs rocm-smi 2>&1 | tail -16

echo "===== DONE rc=$? ====="

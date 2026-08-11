#!/usr/bin/env bash
set -u
echo "=== rocm-smi --showperflevel ==="
rocm-smi --showperflevel
echo
echo "=== rocm-smi default table ==="
rocm-smi
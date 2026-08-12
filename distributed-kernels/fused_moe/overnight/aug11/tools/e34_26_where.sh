#!/usr/bin/env bash
set -uo pipefail
echo "### e34 scratch:"; ls -l ~/e34/ 2>/dev/null
echo "### deliverable folder (node copy):"
ls -l ~/e34/DHK/distributed-kernels/fused_moe/overnight/aug11/exp_34_mode14/ 2>/dev/null
echo "### build_log.md compile command:"
grep -n -B2 -A20 -iE "hipcc|genco" ~/e34/DHK/distributed-kernels/fused_moe/overnight/aug11/exp_34_mode14/build_log.md 2>/dev/null | head -70
echo "### existing build scripts in ~/e34:"
ls -l ~/e34/*.sh 2>/dev/null | head -30
echo "===DONE==="

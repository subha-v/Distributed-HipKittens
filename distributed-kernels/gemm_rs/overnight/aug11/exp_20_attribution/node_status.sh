#!/usr/bin/env bash
# Read-only node reconnaissance for aug11/exp_13 (attribution refresh).
# Answers, before anything is pushed or run:
#   1. is the node idle (KFD pids) and are clocks pinned?
#   2. do the node's kernel sources match the local worktree (md5)?
#   3. what ablation arm .so files already exist (the stale-table trap)?
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
SRCD=/home/subvadla/dhk/distributed-kernels/gemm_rs

echo "===== 1. GPU occupancy ====="
rocm-smi --showpids 2>&1 | sed -n '/PID/,/^====/p'
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pid count: $n"

echo
echo "===== 2. clocks / determinism ====="
rocm-smi --showclocks 2>/dev/null | grep -iE 'sclk|fclk|socclk' | head -12
echo "-- perf level --"
rocm-smi --showperflevel 2>/dev/null | grep -iE 'perf|GPU' | head -12

echo
echo "===== 3. kernel source md5 (node side) ====="
md5sum $SRCD/gemm_rs_mi300x.cpp \
       $SRCD/gemm_rs_mi300x_hk_adapter.cuh \
       $SRCD/gemm_rs_mi300x_constants.cuh \
       $SRCD/gemm_rs_mi300x_host_abi.hpp 2>&1

echo
echo "===== 4. config defaults as they stand on the node ====="
grep -nE '#define HK_GEMM_RS_MI300X_(NEGATIVE_CONTROLS|WGM4|RELEASE_GROUP|RELEASE_GROUP_FULL_ONLY|TILE_SWEEP)' \
  $SRCD/gemm_rs_mi300x.cpp

echo
echo "===== 5. existing ablation artifacts (stale-table trap) ====="
ls -la $ON/harness/build/gemm_rs_abl_*.so 2>&1 | head -20
ls -la $ON/harness/ablate/ 2>&1 | head -10

echo
echo "===== 6. does exp_ablation.py's anchor set still apply? ====="
docker exec -w $ON/harness dhk-gemmrs python3 - <<'PY' 2>&1
import re
SRC = "/home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp"
import importlib.util
spec = importlib.util.spec_from_file_location("abl", "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/harness/exp_ablation.py")
mod = importlib.util.module_from_spec(spec)
# do not execute main(); just pull PATCHES out textually
text = open("/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/harness/exp_ablation.py").read()
ns = {}
exec(text.split("PROLOGUE =")[0].split("import harness_lib")[0].replace("import os","").replace("import subprocess","").replace("import sys",""), ns)
src = open(SRC).read()
for i, (anchor, _) in enumerate(ns["PATCHES"], 1):
    c = src.count(anchor)
    flag = "OK " if c == 1 else "BAD"
    print(f"  anchor {i}: count={c} {flag}  {anchor.strip().splitlines()[0][:70]!r}")
PY

echo
echo "===== 7. shape/plan table sanity (NR + tiles), read-only ====="
grep -nE 'num_reducer_ctas|BM|BN|BK' $SRCD/gemm_rs_mi300x_constants.cuh | head -40

echo
echo "===== 8. rocprofv3 present? ====="
docker exec dhk-gemmrs bash -lc 'which rocprofv3 && rocprofv3 --version 2>&1 | head -3' 2>&1
echo "done"

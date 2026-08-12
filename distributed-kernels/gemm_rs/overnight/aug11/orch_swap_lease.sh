#!/usr/bin/env bash
# Atomically replace tools/gpu_lease.sh with the staged version.
#
# `mv` and not `cp`/scp: bash reads a script incrementally from its open file
# descriptor, so TRUNCATING a script that a process is currently executing makes
# it resume at a byte offset in different text -- which fails in whatever way the
# new bytes happen to parse as. A rename leaves the running process on its
# original inode and only affects the next invocation.
set -uo pipefail
T=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/tools
[ -s "$T/gpu_lease.new.sh" ] || { echo "staged file missing/empty"; exit 1; }
bash -n "$T/gpu_lease.new.sh" || { echo "staged file does not parse; refusing to swap"; exit 1; }
chmod +x "$T/gpu_lease.new.sh"
mv -f "$T/gpu_lease.new.sh" "$T/gpu_lease.sh"
echo "swapped:"
grep -n "caller's remaining budget\|CALLER'S remaining budget" "$T/gpu_lease.sh" | head -3
echo "--- current lease state ---"
bash "$T/gpu_lease.sh" status

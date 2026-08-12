#!/usr/bin/env bash
# Repair CRLF line endings on the node's SHARED tools, and report the blast radius.
#
# Someone pushed with CRLF intact at ~10:37Z. Every affected bash script is now
# broken for every agent ("set: pipefail: invalid option name", "$'\r': command not
# found"), including tools/gpu_lease.sh, which is the node's mutual exclusion. It
# also truncated and corrupted an exp_21 script that was executing at the time.
#
# sed -i is safe to run against a script another agent is executing: it writes a
# temp file and RENAMES it, so a running interpreter keeps the old inode. (scp does
# not -- it truncates in place, which is what caused the corruption.)
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight

echo "########## blast radius: files with CR ##########"
for d in tools harness aug11; do
  n=$(grep -rlI $'\r' "$ON/$d" --include='*.sh' --include='*.py' 2>/dev/null | wc -l)
  echo "  $d: $n file(s) with CR"
done
echo "-- shared tools affected --"
grep -rlI $'\r' "$ON/tools" --include='*.sh' --include='*.py' 2>/dev/null | sed 's/^/    /'
echo "-- other experiments affected (NOT repaired here; their owners must know) --"
grep -rlI $'\r' "$ON/aug11" --include='*.sh' --include='*.py' 2>/dev/null \
  | grep -v exp_21_saturation | sed 's/^/    /' || true

echo
echo "########## repairing tools/ and exp_21_saturation/ (atomic rename per file) ##########"
fixed=0
for f in $(grep -rlI $'\r' "$ON/tools" "$ON/aug11/exp_21_saturation" \
             --include='*.sh' --include='*.py' 2>/dev/null); do
  sed -i 's/\r$//' "$f" && { echo "  fixed $f"; fixed=$((fixed+1)); }
done
echo "  $fixed file(s) repaired"

echo
echo "########## verify the shared tools parse again ##########"
for f in "$ON"/tools/*.sh; do
  bash -n "$f" 2>/dev/null || echo "  STILL BROKEN: $f"
done
echo "  (silence above means every tools/*.sh parses)"
bash "$ON/tools/gpu_lease.sh" status

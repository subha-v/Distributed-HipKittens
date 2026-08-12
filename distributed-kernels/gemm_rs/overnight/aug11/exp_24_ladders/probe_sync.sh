#!/usr/bin/env bash
# The running job printed the OLD drain message ("waiting for kfd drain (fds=1)")
# even though the new drain() logs once as "draining our workers:". Either the push
# did not land run_ladders.sh, or bash is executing a version from before it did.
# This matters for phase B, not for the shape-3 run already in flight, so check
# read-only and do not disturb anything.
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders

echo "===== which drain() is on disk? ====="
echo -n "new marker 'draining our workers' : "; grep -c 'draining our workers' "$D/run_ladders.sh"
echo -n "old marker 'waiting for kfd drain': "; grep -c 'waiting for kfd drain' "$D/run_ladders.sh"
echo -n "kfd_holders defined              : "; grep -c 'kfd_holders()' "$D/run_ladders.sh"
echo -n "DRAIN_IGNORE present             : "; grep -c 'DRAIN_IGNORE' "$D/run_ladders.sh"
echo -n "LAD_EXPECT_A present             : "; grep -c 'LAD_EXPECT_A' "$D/run_ladders.sh"
echo -n "TOOLS indirection present        : "; grep -c 'TOOLS=\${LAD_TOOLSNAP' "$D/run_ladders.sh"
echo "mtime: $(date -Is -r "$D/run_ladders.sh")   size: $(stat -c%s "$D/run_ladders.sh")"

echo
echo "===== is the file CRLF-damaged (my own scripts this time)? ====="
for f in run_ladders.sh full_runner.sh go_full.sh check_full.sh; do
  echo "  $f CR count: $(grep -c $'\r' "$D/$f" 2>/dev/null || true)"
done

echo
echo "===== shape 3 progress (do not disturb) ====="
ls -la "$D/raw/ladder"/lad_s3.rank*.json 2>/dev/null | wc -l
tail -6 "$D/logs/full_run.log"

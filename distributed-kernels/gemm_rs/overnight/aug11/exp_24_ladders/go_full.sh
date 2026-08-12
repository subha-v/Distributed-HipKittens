#!/usr/bin/env bash
# Launch the full ladder DETACHED and return immediately, so the multi-hour run
# (lease wait + six shapes + the evaluator cross-check) does not depend on the
# ssh connection that started it.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders

echo "=== pre-launch checks $(date -Is) ==="
bash -n "$D/run_ladders.sh"     && echo "OK run_ladders.sh"
bash -n "$D/full_runner.sh"     && echo "OK full_runner.sh"
docker exec dhk-gemmrs bash -lc "cd $D && python3 -m py_compile ladders.py ladder_mp.py && echo 'OK python'" 2>&1 | tail -2

echo
echo "=== already running? ==="
pgrep -af 'full_runner.sh|ladder_mp.py' | grep -v pgrep || echo "nothing of ours running"

echo
echo "=== lease status before launch ==="
# Read the lease tool through a CR strip: it currently ships with CRLF line
# endings, which bash rejects outright. full_runner.sh snapshots and normalizes it
# properly; this is just the pre-launch readout.
bash <(tr -d '\r' < "$ON/tools/gpu_lease.sh") status
echo -n "CRLF in tools/gpu_lease.sh: "; grep -c $'\r' "$ON/tools/gpu_lease.sh" || true

# Fresh log for this attempt, and clear any stale instrument-A samples so a
# previous partial cannot be aggregated into tonight's ladder.
#
# LAD_KEEP=1 suppresses the wipe, for RESUMING after a preemption. This guard is
# the difference between filling in one missing shape and destroying five good
# ones: the 06:40 SIGTERM left shapes 0,1,2,4,5 complete on disk, and an
# unconditional rm here would have thrown away 35 minutes of measured data on the
# next launch.
mkdir -p "$D/logs" "$D/raw/ladder"
mv -f "$D/logs/full_run.log" "$D/logs/full_run.prev.log" 2>/dev/null
if [ "${LAD_KEEP:-0}" = "1" ]; then
  echo "LAD_KEEP=1 -- preserving existing samples (resume):"
  for i in 0 1 2 3 4 5; do
    echo "  shape idx $i : $(ls "$D/raw/ladder"/lad_s${i}.rank*.json 2>/dev/null | wc -l)/8 rank files"
  done
else
  rm -f "$D/raw/ladder"/lad_s*.rank*.json "$D/raw/ladder"/lad_s*.rank*.stderr
  rm -rf "$D/raw/eval"
fi

echo
echo "=== launching detached $(date -Is) ==="
setsid nohup bash "$D/full_runner.sh" >/dev/null 2>&1 &
disown 2>/dev/null || true
sleep 12
echo "runner pids:"
pgrep -af 'full_runner.sh' | grep -v pgrep || echo "  (none -- LAUNCH FAILED)"
echo
echo "=== first lines of the log ==="
tail -25 "$D/logs/full_run.log" 2>/dev/null
echo
echo "=== launched. Poll with check_full.sh ==="

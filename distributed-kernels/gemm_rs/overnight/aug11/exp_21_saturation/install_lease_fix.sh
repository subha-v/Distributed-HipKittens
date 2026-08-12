#!/usr/bin/env bash
# Install the drain-check fix into tools/gpu_lease.sh, then restart the exp_21
# campaign so it picks up the patient acquire loop.
#
# The install is an atomic `mv`, not an in-place overwrite: bash reads a script
# incrementally, so rewriting the bytes of a file another agent is currently
# executing can make it resume mid-token. `mv` swaps the inode and leaves every
# running interpreter on the old one.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/aug11/exp_21_saturation

echo "########## syntax check the replacement ##########"
bash -n "$ON/tools/gpu_lease_new.sh" || { echo "FAIL: replacement does not parse"; exit 1; }
echo "  parses OK"

echo
echo "########## install (atomic mv) ##########"
cp -p "$ON/tools/gpu_lease.sh" "$ON/tools/gpu_lease.sh.bak.$(date -u +%H%M%S)"
mv "$ON/tools/gpu_lease_new.sh" "$ON/tools/gpu_lease.sh"
chmod +x "$ON/tools/gpu_lease.sh"
ls -la --time-style=+%H:%M:%S "$ON"/tools/gpu_lease* | sed 's/^/  /'

echo
echo "########## the drain check now, with the wedged pid classified ##########"
bash "$ON/tools/gpu_lease.sh" status

echo
echo "########## stop the old exp_21 campaign gracefully (SIGTERM; its trap releases) ##########"
pids=$(pgrep -f "exp_21_saturation/campaign.sh" || true)
if [ -n "$pids" ]; then
  echo "  SIGTERM -> $pids"
  kill -TERM $pids 2>/dev/null
  for i in $(seq 1 20); do
    sleep 3
    pgrep -f "exp_21_saturation/campaign.sh" >/dev/null || break
  done
fi
pgrep -f "exp_21_saturation/campaign.sh" >/dev/null \
  && { echo "  still alive; refusing to escalate (no SIGKILL on this node)"; exit 1; } \
  || echo "  old campaign gone"

# The trap releases only if we still held the lease; make sure it is not left ours.
bash "$ON/tools/gpu_lease.sh" release exp_21 >/dev/null 2>&1 || true
echo
echo "########## lease before relaunch ##########"
bash "$ON/tools/gpu_lease.sh" status
